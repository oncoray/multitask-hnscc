import json
import numpy as np
import sys

import pandas as pd
import pytorch_lightning as pl
import torch

from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, WeightedRandomSampler
from pathlib import Path
from pprint import pprint

from survival_plus_x.data.dataset import GensheimerDatasetInMemory
from survival_plus_x.data.transforms import get_preprocess_transforms,\
    get_train_transforms_segmentation
from survival_plus_x.models.survival_plus_unet import MultitaskPlusUNET
from survival_plus_x.models.survival_plus_unetr import MultitaskPlusUNETR
from survival_plus_x.utils.commandline_params import parser_with_common_args
from survival_plus_x.models.cox_lightning import init_memory_bank_from_volume_model
from survival_plus_x.models.mae_lightning import MAELightning
from survival_plus_x.models.unet import UNET
from survival_plus_x.models.unetr import UNETR


def check_disjoint_trainvalid(train_ids, valid_ids):
    # check that no intersection
    if len(set(train_ids).intersection(set(valid_ids))) > 0:
        raise ValueError("Train ids and validation ids are not disjoint!")
    print()
    print(f"Number of training patients {len(train_ids)}")
    print(f"number of validation patients {len(valid_ids)}")


def collate_fn(batch):
    """
    we have to ignore the 'img_transforms' and 'mask_transforms' keys, otherwise
    torch will complain that batches are not equally long for all elements during
    training somehow
    """

    return {
        "img": torch.stack([elem["img"] for elem in batch]),
        "mask": torch.stack([elem["mask"] for elem in batch]),
        "label": torch.stack([elem["label"] for elem in batch]),
        "label_gensheimer": torch.stack(
            [elem["label_gensheimer"] for elem in batch]),
        "patient": [elem["patient"] for elem in batch]
    }


def use_pretrained_params(model, pretrained_model):
    """
    All the matching sub-entries of the pretrained model state dict
    will be used for updating the models state dict
    """

    sd_model = model.state_dict()
    sd_pretrained = pretrained_model.state_dict()
    for k in sd_pretrained:
        if k in sd_model:
            sd_model[k] = sd_pretrained[k]
    model.load_state_dict(sd_model)
    return model


def main(args):
    pl.seed_everything(args.seed, workers=True)

    train_ids = pd.read_csv(args.train_id_file,
                            header=None).values.squeeze().tolist()
    valid_ids = pd.read_csv(args.valid_id_file,
                            header=None).values.squeeze().tolist()

    check_disjoint_trainvalid(train_ids, valid_ids)

    # interval_breaks = [0, 6, 12, 18, 24, 30, 36, 48, 60, 84, 120]
    interval_breaks = args.gensheimer_interval_breaks
    train_dataset = GensheimerDatasetInMemory(
        image_directories=args.input,
        image_filename=args.img_filename,
        mask_filename=args.mask_filename,
        patient_ids=train_ids,
        outcome_file=args.outcome,
        outcome_file_sep=args.outcome_sep,
        outcome_file_id_column=args.id_col,
        outcome_file_time_column=args.time_col,
        outcome_file_event_column=args.event_col,
        interval_breaks=interval_breaks,
        preprocess_transform=get_preprocess_transforms(
            (1.25 * np.array(args.image_size)).astype(int).tolist()),  # increase spatial size so we can random crop
        augmentation_transform=get_train_transforms_segmentation(
            list(args.image_size)))
    print()
    print("len(train_dataset)", len(train_dataset))

    if args.balance_nevents:
        print("Balance events/censoring in batches during training!")
        # use a sampler that puts more weights on under-represented patients in terms of events/censoring
        n_samples = len(train_dataset)
        # censoring indicator, 1 is event, 0 is censoring
        event_or_cens = [d["label"].numpy()[1] for d in train_dataset.data]
        n_events = sum(event_or_cens)
        n_censored = n_samples - n_events
        print(f"Training: n_samples={n_samples}, n_events={n_events}")
        class_count = [n_censored, n_events]  # censored, event
        weights = n_samples / torch.tensor(class_count, dtype=torch.float)
        # weights[0] = weight for censoring, weights[1] = weight for event
        sample_weights = torch.tensor(
            [weights[int(cc)] for cc in event_or_cens])

        train_sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=n_samples,
            replacement=True)
        train_shuffle = False
    else:
        train_sampler = None
        train_shuffle = True

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=train_shuffle,
        sampler=train_sampler,
        collate_fn=collate_fn)

    val_dataset = GensheimerDatasetInMemory(
        image_directories=args.input,
        image_filename=args.img_filename,
        mask_filename=args.mask_filename,
        patient_ids=valid_ids,
        outcome_file=args.outcome,
        outcome_file_sep=args.outcome_sep,
        outcome_file_id_column=args.id_col,
        outcome_file_time_column=args.time_col,
        outcome_file_event_column=args.event_col,
        interval_breaks=interval_breaks,
        preprocess_transform=get_preprocess_transforms(
            list(args.image_size)),  # in inference use the given size
        augmentation_transform=None)
    print()
    print("len(validation_dataset)", len(val_dataset))

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        sampler=None,
        collate_fn=collate_fn)

    outcome_df_train = pd.DataFrame.from_dict(
        train_dataset.dataset.outcome_dict,
        orient="index")
    training_labels = outcome_df_train[
        outcome_df_train.index.isin(train_ids)].to_numpy()

    # model construction
    if args.memory_bank_init_from_volume:
        memory_bank_init_dict = init_memory_bank_from_volume_model(
            train_dataset, val_dataset,
            output_dir=args.default_root_dir)
    else:
        memory_bank_init_dict = None

    mt_model_args = dict(
        learning_rate=args.learning_rate,
        # for the lognormal and the gensheimer head
        timepoints_cindex=args.timepoints_cindex,
        timepoints_brier=args.timepoints_brier,
        training_labels=training_labels,
        # for the gensheimer
        gensheimer_interval_breaks=interval_breaks,
        # for the cox head
        cox_output_activation=args.output_activation,
        cox_nll_on_batch_only=args.nll_on_batch_only,
        cox_average_over_events_only=args.average_over_events_only,
        cox_training_labels=train_dataset.dataset.outcome_dict,
        cox_validation_labels=val_dataset.dataset.outcome_dict,
        cox_memory_bank_init_dict=memory_bank_init_dict,
        cox_memory_bank_decay_factor=args.memory_bank_decay_factor,
        cox_bias=False,  # not args.no_bias_at_output,
        # choose which heads and which weights
        # tested with cox + gensheimer, adding others makes training unstable
        heads_to_use=args.heads_to_use,
    )

    if args.vit_or_cnn == "vit":
        vit_args = dict(
            vit_image_size=list(args.image_size),
            vit_patch_size=list(args.patch_size),
            vit_dim=args.dim,
            vit_depth=args.depth,
            vit_heads=args.heads,
            vit_dim_head=args.dim_head,
            vit_mlp_dim=args.mlp_dim,
            vit_channels=args.channels,
            vit_dropout=args.dropout,
            vit_emb_dropout=args.emb_dropout,
            vit_output_token="cls",
        )
        mt_model_args.update(vit_args)

        unetr_args = dict(
            unetr_res_block=True,
            unetr_conv_block=True,
            unetr_norm_name=args.unetr_norm_name,
            # n filters in the conv part of UNETR
            unetr_feature_size=args.unetr_feature_size,
            unetr_attention_layer_output_idx=args.unetr_attention_layer_output_idx,
            with_segmentation_loss=args.with_segmentation_loss,
            with_densenet=args.with_densenet)

        mt_model_args.update(unetr_args)

        model = MultitaskPlusUNETR(**mt_model_args)

        # handle potentially pretrained
        if args.mae_pretrained_ckpt is not None and args.seg_pretrained_ckpt is not None:
            raise ValueError(
                "Pretrained checkpoints for both, MAE and Segmentation"
                " are given, please choose only one")

        if args.mae_pretrained_ckpt is not None:
            ckpt_file = Path(args.mae_pretrained_ckpt)
            assert ckpt_file.exists()
            print("Loading pretrained vit from MAE pretraining!")
            restored_model = MAELightning.load_from_checkpoint(ckpt_file)
            model.vit = restored_model.vit
        elif args.seg_pretrained_ckpt is not None:
            ckpt_file = Path(args.seg_pretrained_ckpt)
            assert ckpt_file.exists()
            print("Loading pretrained vit from segmentation!")
            restored_model = UNETR.load_from_checkpoint(ckpt_file)
            model = use_pretrained_params(model, restored_model)

    elif args.vit_or_cnn == "cnn":
        unet_args = dict(
            unet_in_channels=args.channels,
            unet_features_start=8,
            unet_feature_size_for_surv=32,
            with_segmentation_loss=args.with_segmentation_loss,
            with_densenet=args.with_densenet
        )
        mt_model_args.update(**unet_args)

        model = MultitaskPlusUNET(**mt_model_args)

        if args.seg_pretrained_ckpt is not None:
            ckpt_file = Path(args.seg_pretrained_ckpt)
            assert ckpt_file.exists()
            print("Loading pretrained unet from segmentation!")
            restored_model = UNET.load_from_checkpoint(ckpt_file)
            model = use_pretrained_params(model, restored_model)

    print(model)
    assert model.hparams['with_segmentation_loss'] == args.with_segmentation_loss
    assert model.hparams['with_densenet'] == args.with_densenet

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="{val_loss:.2f}-{train_loss:.2f}-{epoch:02d}",
        mode="min",
        save_last=True,
        save_top_k=args.num_best_checkpoints,
        every_n_epochs=args.checkpoint_every_n_epochs)

    trainer = pl.Trainer.from_argparse_args(
        args,
        callbacks=[
            checkpoint_callback
        ])

    print()
    print("Start training")
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    parser = parser_with_common_args(
        "Train Multi-loss model, optionally together with segmentation")
    parser = pl.Trainer.add_argparse_args(parser)
    parser = MultitaskPlusUNETR.add_model_specific_args(parser)
    parser.add_argument(
        "--vit_or_cnn",
        choices=["vit", "cnn"],
        default="vit")
    parser.add_argument(
        "--mae_pretrained_ckpt",
        type=str)
    parser.add_argument(
        "--seg_pretrained_ckpt",
        type=str)

    args = parser.parse_args()
    print("\nParsed args are\n")
    pprint(args)

    default_root_dir = args.default_root_dir
    if default_root_dir is None:
        exp_name = "mt_{args.vit_or_cnn}"
        if args.with_segmentation_loss:
            exp_name += "+seg"
        default_root_dir = f"./experiments/{exp_name}/training"
    if not isinstance(default_root_dir, Path):
        default_root_dir = Path(default_root_dir)

    if not default_root_dir.is_dir():
        default_root_dir.mkdir(parents=True)
    else:
        raise ValueError(
            f"Default_root_dir {default_root_dir} already exists!")

    print(f"\nUsing {default_root_dir} as output directory.")

    # storing the commandline arguments to a json file
    with open(default_root_dir / "commandline_args.json", 'w') as of:
        json.dump(vars(args), of, indent=2)

    # now convert to a pathlib object (not done in the beginning because they
    # could not be json-serialized)
    args.default_root_dir = default_root_dir
    assert isinstance(args.default_root_dir, Path)
    args.input = [Path(inp) for inp in args.input]

    retval = main(args)
    sys.exit(retval)
