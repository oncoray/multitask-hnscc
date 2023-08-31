import copy
import json
import pandas as pd
import sys
import torch
import pytorch_lightning as pl
import numpy as np

from monai.metrics import DiceMetric
from pathlib import Path
from torch.utils.data import DataLoader

from argparse import ArgumentParser

from survival_plus_x.data.dataset import GensheimerDatasetInMemory
from survival_plus_x.data.transforms import get_preprocess_transforms
from survival_plus_x.models.survival_plus_unetr import MultitaskPlusUNETR
from survival_plus_x.models.survival_plus_unet import MultitaskPlusUNET
from survival_plus_x.models.multitask import multitask_metrics_from_step_outputs
from survival_plus_x.utils.commandline_params import add_common_args
from survival_plus_x.models.cox_lightning import compute_stratification_logrank_pvalue


def inference_single_sample(args, test_ids, model, seg_threshold=0.5):
    test_dataset = GensheimerDatasetInMemory(
        image_directories=args.input,
        image_filename=args.img_filename,
        mask_filename=args.mask_filename,
        patient_ids=test_ids,
        outcome_file=args.outcome,
        outcome_file_sep=args.outcome_sep,
        outcome_file_id_column=args.id_col,
        outcome_file_time_column=args.time_col,
        outcome_file_event_column=args.event_col,
        interval_breaks=model.hparams.gensheimer_interval_breaks,
        preprocess_transform=get_preprocess_transforms(list(args.image_size)),
        augmentation_transform=None)

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    dice_metric = DiceMetric(include_background=True, reduction="none")

    patients = []
    dices = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            pats = batch['patient']
            pred_dict = model.predict_step(batch, batch_idx)

            pred_seg = pred_dict["segmentation"]
            gt = torch.tensor(pred_seg["mask"], dtype=torch.uint8).cpu()
            pred = pred_seg["prediction"].cpu() # with sigmoid output
            pred_bin = (pred >= seg_threshold).to(torch.uint8)
            # a dice score per 3d sample in the batch
            dice_batch = dice_metric(gt, pred_bin).cpu().numpy()[:, 0]
            dices.extend(dice_batch)  # this automagically remains a a list, not np.array
            patients.extend(pats)

    metrics_per_patient = pd.DataFrame({
        "patient": patients,
        "dice_score": dices,
    })

    # metrics = pd.DataFrame({
    #     "avg_dice": [metrics_per_patient["dice_score"].mean()]
    # })
    return metrics_per_patient



def main(args):
    pl.seed_everything(args.seed)
    test_ids = pd.read_csv(args.test_id_file,
                           header=None).values.squeeze().tolist()

    if args.vit_or_cnn == "vit":
        cls = MultitaskPlusUNETR

    elif args.vit_or_cnn == "cnn":
        cls = MultitaskPlusUNET

    model = cls.load_from_checkpoint(
        checkpoint_path=args.ckpt_file)
    model.eval()
    model.freeze()
    print(f"Loaded trained model from checkpoint {args.ckpt_file}.")

    if args.n_samples > 1:
        raise NotImplementedError
        # test_pred_step_outputs = inference_multiple_samples(
        #     args, test_ids, model)
    elif args.n_samples == 1:
        test_metrics_per_patient = inference_single_sample(
            args, test_ids, model, seg_threshold=0.5)
    else:
        raise ValueError(
            f"n_samples must be >= 1, not "
            f"{args.n_samples}")

    print()
    print(f"metrics (storing to {args.output_dir})")
    print(test_metrics_per_patient.describe())
    # also store metrics per patient
    test_metrics_per_patient.to_csv(args.output_dir / "segmentation_metrics.csv", index=False)

    return 0


if __name__ == "__main__":
    parser = ArgumentParser("Segmentation inference")
    parser = add_common_args(parser)
    parser.add_argument(
        '--ckpt_file',
        type=str,
        help='Full path to a checkpoint file of the trained model.')
    parser.add_argument(
        '--output_dir',
        type=str,
        default=None)
    parser.add_argument(
        '--gpus',
        type=int,
        default=0)
    parser.add_argument(
        '--n_samples',
        type=int,
        default=1,
        help="Number of random crops to create per patient and from which to make a final prediction"
    )
    parser.add_argument(
        '--sample_aggregation',
        type=str,
        choices=["mean", "median", "min", "max"],
        default="mean"
    )
    parser.add_argument(
        "--vit_or_cnn",
        type=str,
        choices=["vit", "cnn"],
        #default="vit"
        )

    args = parser.parse_args()
    print(f"parsed args are\n{args}")

    if args.output_dir is None:
        args.output_dir = "./cox_vit/seg_inference"
    if not isinstance(args.output_dir, Path):
        args.output_dir = Path(args.output_dir)

    if not args.output_dir.is_dir():
        args.output_dir.mkdir(parents=True)
    else:
        raise ValueError(f"Output_dir {args.output_dir} already exists!")

    # storing the commandline arguments to a json file
    with open(args.output_dir / "commandline_args.json", 'w') as of:
        # pathlib objects cant be serialized so we convert to string
        storage_args = vars(copy.deepcopy(args))
        storage_args["output_dir"] = str(
            storage_args["output_dir"])

        json.dump(storage_args, of, indent=2)

    args.input = [Path(inp) for inp in args.input]

    retval = main(args)
    sys.exit(retval)
