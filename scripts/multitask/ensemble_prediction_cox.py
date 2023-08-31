import argparse
import copy
import json
import pandas as pd
import numpy as np
import sys

from pathlib import Path

from survival_plus_x.models.cox_lightning import compute_cindex, compute_stratification_logrank_pvalue


def main(args):

    # configure the reduction function of the ensemble
    assert args.ensemble_reduction_fn in ["min", "max", "mean", "median"]
    if args.ensemble_reduction_fn == "min":
        ensemble_fn = np.min
    elif args.ensemble_reduction_fn == "max":
        ensemble_fn = np.max
    elif args.ensemble_reduction_fn == "mean":
        ensemble_fn = np.mean
    elif args.ensemble_reduction_fn == "median":
        ensemble_fn = np.median

    # read all the prediction files
    prediction_data = [
        pd.read_csv(f, sep=args.prediction_file_sep)
        for f in args.prediction_files]

    # get all the patients and collect predictions for each patient
    patient_data = {}
    for df in prediction_data:
        n_patients = len(df)
        patient_ids = df[args.prediction_file_id_col].to_numpy()
        preds = df[args.prediction_file_pred_col].to_numpy()
        times = df[args.prediction_file_time_col].to_numpy()
        events = df[args.prediction_file_event_col].to_numpy()

        for i in range(n_patients):
            pat_id = patient_ids[i]
            pred = preds[i]
            time = times[i]
            event = events[i]

            if pat_id not in patient_data:
                patient_data[pat_id] = {
                    "event_time": time,
                    "event": event,
                    "model_predictions": [pred]
                }
            else:
                patient_data[pat_id]["model_predictions"].append(pred)

    # each patient should have an equal number of predictions
    patient_ids = sorted(patient_data.keys())
    n_preds_per_patient = [len(patient_data[pat]["model_predictions"])
                           for pat in patient_ids]
    assert len(set(n_preds_per_patient)
               ) == 1, "All patients should have the same amount of predictions"
    n_preds_per_patient = n_preds_per_patient[0]
    print()
    print(
        f"Using {n_preds_per_patient} predictions for each patient for ensembling!")

    # now compute ensemble prediction as well as metrics of uncertainty
    for pat_id in patient_ids:
        model_predictions = patient_data[pat_id]["model_predictions"]
        ensemble_prediction = ensemble_fn(model_predictions)

        # NOTE: numpy uses the 1/N function which is not an unbiased estimator
        # but works also in case of a single prediction for each patient
        # whereas the unbiased estimator with 1/(N-1) * ... would fail here.
        prediction_std = np.std(
            model_predictions, ddof=1 if n_preds_per_patient > 1 else 0)

        patient_data[pat_id]["ensemble_prediction"] = ensemble_prediction
        patient_data[pat_id]["prediction_std"] = prediction_std

    # pprint(patient_data)

    # prepare a dictionary for computing metrics and storage
    # where we just leave out the individual predictions

    ensemble_pred_and_label = pd.DataFrame({
        "patient": patient_ids,
        "event_time": np.array([patient_data[pat_id]["event_time"] for pat_id in patient_ids]),
        "event": np.array([patient_data[pat_id]["event"] for pat_id in patient_ids]),
        "prediction": np.array([patient_data[pat_id]["ensemble_prediction"] for pat_id in patient_ids]),
        "prediction_std": np.array([patient_data[pat_id]["prediction_std"] for pat_id in patient_ids]),
    })

    ensemble_pred_and_label.rename(
        {"prediction": "ensemble_prediction"}, axis=1).to_csv(
            args.output_dir / f"{args.method}_predictions.csv")

    # compute metrics and store to disk

    ensemble_c_index = compute_cindex(ensemble_pred_and_label)

    print()
    if args.stratification_cutoff is None:
        stratification_cutoff = np.median(
            ensemble_pred_and_label["prediction"])
        print("Computing stratification cutoff as median "
              f"of ensemble predictions: {stratification_cutoff}")
    else:
        stratification_cutoff = args.stratification_cutoff
        print(f"Using provided stratification cutoff: {stratification_cutoff}")

    ensemble_pval = compute_stratification_logrank_pvalue(
        ensemble_pred_and_label, stratification_cutoff)

    ensemble_metrics = {
        "c_index": ensemble_c_index,
        "stratification_cutoff": stratification_cutoff,
        "stratification_logrank_pval": ensemble_pval
    }

    ensemble_metrics = pd.DataFrame(ensemble_metrics, index=[0])
    print(f"Ensemble metrics (storing to {args.output_dir})")
    print(ensemble_metrics)
    ensemble_metrics.to_csv(
        args.output_dir / f"{args.method}_metrics.csv", index=False)

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Cox cancer VIT Model ensembles")
    parser.add_argument(
        "--prediction_files",
        nargs="+",
        type=str,
        help="Full paths to csv files containing predictions for each patient."
    )
    parser.add_argument(
        '--prediction_file_sep',
        type=str,
        default=",",
        help="Column separator token for the prediction csv files.")
    parser.add_argument(
        "--prediction_file_id_col",
        type=str,
        help="Name of patient id column in the prediction csv files."
    )
    parser.add_argument(
        "--prediction_file_pred_col",
        type=str,
        help="Name of prediction column in the prediction csv files."
    )
    parser.add_argument(
        "--prediction_file_time_col",
        type=str,
        help="Name of event time column in the prediction csv files."
    )
    parser.add_argument(
        "--prediction_file_event_col",
        type=str,
        help="Name of event indicator column in the prediction csv files."
    )
    parser.add_argument(
        "--ensemble_reduction_fn",
        type=str,
        choices=["mean", "median", "min", "max"],
        help="Function that determines how individual model predictions "
             "should be aggregated to obtain an ensemble prediction."
    )
    parser.add_argument(
        "--output_dir",
        type=str)
    parser.add_argument(
        "--stratification_cutoff",
        type=float,
        help="Cutoff to stratify patients into low and high risk groups. "
             "Needed for computation of p-value of logrank test for ensemble. "
             "If not provided, will use the median of the computed ensemble "
             "predictions."
    )
    parser.add_argument(
        "--method",
        type=str,
        default="cox",
        help="To specify the output filenames"
    )

    print("\n\nComputing ensemble performance (Cox)!\n")
    args = parser.parse_args()
    print(f"parsed args are\n{args}")

    if args.output_dir is None:
        args.output_dir = "./ensemble_cox"
    if not isinstance(args.output_dir, Path):
        args.output_dir = Path(args.output_dir)

    if not args.output_dir.is_dir():
        args.output_dir.mkdir(parents=True)

    # storing the commandline arguments to a json file
    with open(args.output_dir / "commandline_args.json", 'w') as of:
        # pathlib objects cant be serialized so we convert to string
        storage_args = vars(copy.deepcopy(args))
        storage_args["output_dir"] = str(
            storage_args["output_dir"])

        json.dump(storage_args, of, indent=2)

    retval = main(args)

    sys.exit(retval)
