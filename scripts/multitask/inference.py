import copy
import json
import pandas as pd
import sys
import torch
import pytorch_lightning as pl
import numpy as np

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


def inference_single_sample(args, test_ids, model):
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

    trainer = pl.Trainer(gpus=args.gpus)

    test_pred_step_outputs = trainer.predict(
        model,
        dataloaders=test_loader)

    return test_pred_step_outputs


def inference_multiple_samples(args, test_ids, model):
    # we have to set up the dataset in the way that initially a larger
    # crop is chosen and then a transform that creates random crops of the
    # actually wanted size
    from monai.transforms import RandSpatialCropSamplesd, Resized, Compose

    n_samples = args.n_samples
    aggregation_fn = {
        "min": torch.min,
        "max": torch.max,
        "mean": torch.mean,
        "median": torch.median
    }[args.sample_aggregation]

    random_crops_transform = Compose([
        RandSpatialCropSamplesd(
            keys=["img", "mask"],
            roi_size=args.image_size,
            random_size=False,
            random_center=True,
            num_samples=n_samples
        ),
        Resized(
            keys=["img", "mask"],
            spatial_size=args.image_size)
    ])

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
        preprocess_transform=get_preprocess_transforms(
            (1.25 * np.array(args.image_size)).astype(int).tolist()),  # increase spatial size so we can random crop
        augmentation_transform=random_crops_transform)

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers)

    # now each batch the data loader produces is a list of dicts of length n_samples
    # for which we have to make predictions and boil down results to a single dict
    # per batch

    step_outputs = []

    surv_heads = model.hparams.heads_to_use
    print("models survival heads", surv_heads)
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            # batch is now a list of dicts, one dict for each sample
            # and we have to aggregate over all samples to make a final
            # prediction for each patient in the batch
            aggregated_step_result = {}
            for sample_idx, sample_dict in enumerate(batch):
                print(
                    f"\nPredicting for Batch {batch_idx+1}, sample {sample_idx + 1}\n")

                # results for the first sample of the batch patients
                # has keys 'survival' and 'segmentation' where
                # survival is another dict for each head containing the keys 'patient', 'label' and 'prediction'
                sample_result = model.predict_step(
                    sample_dict, batch_idx=None)  # batch_idx is not used anyway
                # print(sample_result.keys())
                # print("sample_result['survival'].keys()",
                #       sample_result["survival"].keys())

                if sample_idx == 0:
                    for head in surv_heads:  # sample_result["survival"]:
                        aggregated_step_result[head] = dict()
                        # copy all non-prediction keys
                        for k in sample_result["survival"][head].keys():
                            if "prediction" in k:
                                continue
                            aggregated_step_result[head][k] = sample_result["survival"][head][k]

                        aggregated_step_result[head]["sample_predictions"] = [
                            sample_result["survival"][head]["prediction"].detach()]

                else:
                    for head in surv_heads:  # sample_result["survival"]:
                        aggregated_step_result[head]["sample_predictions"].append(
                            sample_result["survival"][head]["prediction"].detach())

            # stack all the predictions we aggregated along the second dimension,
            # so the output has shape B, n_samples, n_predictions for each head
            for head in surv_heads:
                aggregated_step_result[head]["sample_predictions"] = torch.stack(
                    aggregated_step_result[head]["sample_predictions"], dim=1)

            # now final aggregation
            for head in surv_heads:
                aggregated = aggregation_fn(
                    aggregated_step_result[head]["sample_predictions"], dim=1)

                # NOTE: for min, max and median, torch calls return a tuple of
                # values and indices if dim= argument is passed (but not for mean)
                if not isinstance(aggregated, torch.Tensor):
                    assert len(aggregated) == 2
                    vals, _ = aggregated
                    aggregated = vals
                aggregated_step_result[head]["sample_predictions_std"] = torch.std(
                    aggregated_step_result[head]["sample_predictions"],
                    dim=1,
                    unbiased=False
                )

                aggregated_step_result[head]["prediction"] = aggregated

            step_outputs.append(dict(survival=aggregated_step_result))

    return step_outputs


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
        test_pred_step_outputs = inference_multiple_samples(
            args, test_ids, model)
    elif args.n_samples == 1:
        test_pred_step_outputs = inference_single_sample(
            args, test_ids, model)
    else:
        raise ValueError(
            f"n_samples must be >= 1, not "
            f"{args.n_samples}")

    # NOTE: information of variance among multiple predictions gets lost here
    # since only the "prediction" keys are taken into account when evaluating
    # metrics and returning test_pred
    # TODO: maybe write out the step_outputs in some meaningful way as well?

    test_metrics, test_pred = multitask_metrics_from_step_outputs(
        [d["survival"] for d in test_pred_step_outputs],
        task_names=model.hparams.heads_to_use,
        timepoints_cindex=model.hparams.timepoints_cindex,
        timepoints_brier=model.hparams.timepoints_brier,
        training_labels=model.hparams.training_labels,
        gensheimer_interval_breaks=model.hparams.gensheimer_interval_breaks
    )

    # if inference_multiple_samples -> also add the standard deviations for the predictions
    # to the test_pred dataframe
    if args.n_samples > 1:
        for head in test_pred:
            stds = torch.cat([d['survival'][head]['sample_predictions_std']
                              for d in test_pred_step_outputs])
            #pats = [d['survival'][head]['patient']]
            stds_dict = {}
            for i in range(stds.shape[1]):
                stds_dict[f'std_prediction_{i}'] = stds[:, i]
            stds_dict = pd.DataFrame(stds_dict)
            test_pred[head] = pd.concat([test_pred[head], stds_dict], axis=1)

    print()
    print(f"Storing test predictions to {args.output_dir}")
    for head in test_pred:
        pred_df = test_pred[head]
        pred_df.set_index("patient").to_csv(
            args.output_dir / f"{head}_predictions.csv")

    for head in test_metrics:
        metrics = pd.DataFrame(test_metrics[head], index=[0])

        # stratification cutoff for cox model only
        # TODO: can we use stratification cutoff for other losses?
        if head == "cox":
            if args.stratification_cutoff_cox is None:
                print("Note: No stratification cutoff was provided. Will "
                      "determine it as median of predictions. This might "
                      "not be intended for data other than the training "
                      "data! If you are not using the training data now "
                      "you should determine the cutoff from that beforehand!")
                stratification_cutoff = np.median(
                    test_pred[head]['prediction'])
            else:
                stratification_cutoff = args.stratification_cutoff_cox

            test_logrank_pval = compute_stratification_logrank_pvalue(
                test_pred[head], cutoff=stratification_cutoff)

            metrics['stratification_cutoff'] = stratification_cutoff
            metrics['stratification_logrank_pval'] = test_logrank_pval

        print()
        print(f"{head.capitalize()} metrics (storing to {args.output_dir})")
        print(metrics)
        metrics.to_csv(args.output_dir / f"{head}_metrics.csv", index=False)

    return 0


if __name__ == "__main__":
    parser = ArgumentParser("Inference")
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
        '--stratification_cutoff_cox',
        type=float,
        help="Cutoff value applied to the test predictions to divide into low and high risk groups.")
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

    # parser.add_argument('--plot_predictions',
    #                     action="store_true",
    #                     default=False,
    #                     help="Flag to decide whether predictions for each"
    #                          "  patient should be plotted after training.")

    args = parser.parse_args()
    print(f"parsed args are\n{args}")

    if args.output_dir is None:
        args.output_dir = "./cox_vit/inference"
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
