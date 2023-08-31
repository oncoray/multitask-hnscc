import argparse
import copy
import json
import pandas as pd
import numpy as np
import sys

from pathlib import Path

from sksurv.metrics import integrated_brier_score,\
    concordance_index_censored
from sksurv.util import Surv

from survival_plus_x.models.cox_lightning import compute_stratification_logrank_pvalue


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

    # we have possibly multiple prediction columns for which we need
    # to do ensembling
    pred_cols = [c for c in prediction_data[0].columns if c.startswith(
        args.prediction_file_pred_col)]

    print("Will compute ensemble for each column in", pred_cols)

    # get all the patients and collect predictions for each patient
    patient_data = {}
    for df in prediction_data:

        n_patients = len(df)
        patient_ids = df[args.prediction_file_id_col].to_numpy()
        preds = df[pred_cols].to_numpy()
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
                }
                for j, p_col in enumerate(pred_cols):
                    patient_data[pat_id][p_col] = [pred[j]]
            else:
                for j, p_col in enumerate(pred_cols):
                    patient_data[pat_id][p_col].append(pred[j])

    # each patient should have an equal number of predictions
    patient_ids = sorted(patient_data.keys())

    for p_col in pred_cols:
        n_preds_per_patient = [
            len(patient_data[pat][p_col]) for pat in patient_ids]
        # print()
        # print(p_col)
        # print({pat: patient_data[pat][p_col] for pat in patient_ids})

        assert len(set(n_preds_per_patient)
                   ) == 1, f"All patients should have the same amount of entries for {p_col} but found {set(n_preds_per_patient)}"

    n_preds_per_patient = n_preds_per_patient[0]
    print()
    print(
        f"Using {n_preds_per_patient} predictions for each patient for ensembling!")

    # now compute ensemble prediction as well as metrics of uncertainty
    for pat_id in patient_ids:
        for p_col in pred_cols:
            model_predictions = patient_data[pat_id][p_col]
            ensemble_prediction = ensemble_fn(model_predictions)

            # NOTE: numpy uses the 1/N function which is not an unbiased estimator
            # but works also in case of a single prediction for each patient
            # whereas the unbiased estimator with 1/(N-1) * ... would fail here.
            prediction_std = np.std(
                model_predictions, ddof=1 if n_preds_per_patient > 1 else 0)

            patient_data[pat_id][f"ensemble_{p_col}"] = ensemble_prediction
            patient_data[pat_id][f"std_{p_col}"] = prediction_std

    # pprint(patient_data)

    # prepare a dictionary for computing metrics and storage
    # where we just leave out the individual predictions

    ensemble_pred_and_label = {
        "patient": patient_ids,
        "event_time": np.array([patient_data[pat_id]["event_time"] for pat_id in patient_ids]),
        "event": np.array([patient_data[pat_id]["event"] for pat_id in patient_ids]),
        # "prediction": np.array([patient_data[pat_id]["ensemble_prediction"] for pat_id in patient_ids]),
        # "prediction_std": np.array([patient_data[pat_id]["prediction_std"] for pat_id in patient_ids]),
    }
    for p_col in pred_cols:
        ensemble_pred_and_label[p_col] = np.array(
            [patient_data[pat_id][f"ensemble_{p_col}"] for pat_id in patient_ids])
        ensemble_pred_and_label[f"std_{p_col}"] = np.array(
            [patient_data[pat_id][f"std_{p_col}"] for pat_id in patient_ids])

    ensemble_pred_and_label = pd.DataFrame(ensemble_pred_and_label)

    ensemble_pred_and_label.to_csv(
        args.output_dir / f"{args.method}_predictions.csv")

    # compute metrics and store to disk

    # ensemble_c_index = compute_cindex(ensemble_pred_and_label)

    # print()
    # if args.stratification_cutoff is None:
    #     stratification_cutoff = np.median(
    #         ensemble_pred_and_label["prediction"])
    #     print("Computing stratification cutoff as median "
    #           f"of ensemble predictions: {stratification_cutoff}")
    # else:
    #     stratification_cutoff = args.stratification_cutoff
    #     print(f"Using provided stratification cutoff: {stratification_cutoff}")

    # ensemble_pval = compute_stratification_logrank_pvalue(
    #     ensemble_pred_and_label, stratification_cutoff)

    # ensemble_metrics = {
    #     "c_index": ensemble_c_index,
    #     "stratification_cutoff": stratification_cutoff,
    #     "stratification_logrank_pval": ensemble_pval
    # }

    # compute brier for ensemble
    # read training outcomes
    training_labels = pd.read_csv(
        args.train_outcome_file, sep=args.train_outcome_file_sep)
    training_labels = training_labels[[
        args.prediction_file_time_col,
        args.prediction_file_event_col]].to_numpy()

    survival_train = Surv.from_arrays(
        event=training_labels[:, 1].astype(np.uint8),
        time=training_labels[:, 0].astype(np.float32))

    survival = Surv.from_arrays(
        event=ensemble_pred_and_label["event"].to_numpy().astype(np.uint8),
        time=ensemble_pred_and_label["event_time"].to_numpy().astype(np.float32))

    brier_cols = sorted(
        [c for c in ensemble_pred_and_label.columns if "predicted_survival_for_brier_time_" in c and "std" not in c])
    pred_survival_brier = ensemble_pred_and_label[brier_cols].to_numpy().astype(
        np.float32)

    timepoints_brier = [
        float(c.split("predicted_survival_for_brier_time_")[1]) for c in brier_cols]

    try:
        integrated_brier = integrated_brier_score(
            survival_train=survival_train,
            survival_test=survival,
            estimate=pred_survival_brier,
            times=timepoints_brier)
    except ValueError as e:
        print("[W]: Failed to compute integrated brier score", e)
        integrated_brier = np.nan

    metrics = {
        'integrated_brier_score': integrated_brier,
    }

    # compute c_index for ensemble
    # TODO: this can now also be many columns!
    cindex_cols = [
        c for c in ensemble_pred_and_label.columns if "predicted_survival_for_cindex_time_" in c and "std" not in c]
    if args.stratification_cutoff is not None:
        # load the file containing the cutoffs
        stratification_cutoff = pd.read_csv(args.stratification_cutoff)
    else:
        stratification_cutoff = None

    cutoffs_to_store = {}
    for cindex_col in cindex_cols:
        timepoint = cindex_col.split("predicted_survival_for_cindex_time_")[1]
        pred_survival_cindex = ensemble_pred_and_label[cindex_col].to_numpy().astype(
            np.float32)

        c_index = concordance_index_censored(
            event_indicator=ensemble_pred_and_label["event"].to_numpy().astype(
                bool),
            event_time=ensemble_pred_and_label["event_time"].to_numpy().astype(
                np.float32),
            estimate=pred_survival_cindex)[0]

        metrics[f'c_index_{timepoint}'] = 1. - c_index

        # compute stratification for c_index prediction as well
        if stratification_cutoff is None:
            cutoff = np.median(
                ensemble_pred_and_label[cindex_col])
            print("Computing stratification cutoff as median "
                  f"of ensemble predictions for cindex timepoint {timepoint}: {cutoff}")
        else:
            # TODO: handle this with multiple cindex values
            cutoff = stratification_cutoff[cindex_col].to_numpy()[0]
            print(
                f"Using provided stratification cutoff: {cutoff} for cindex timepoint {timepoint}")

        pval = compute_stratification_logrank_pvalue(
            pd.DataFrame({
                "prediction": ensemble_pred_and_label[cindex_col].to_numpy(),
                "event_time": ensemble_pred_and_label["event_time"].to_numpy(),
                "event": ensemble_pred_and_label["event"].to_numpy(),
            }),
            cutoff)
        print(f"logrank p={pval}")
        metrics[f"stratification_logrank_pval_{timepoint}"] = pval
        cutoffs_to_store[cindex_col] = cutoff
        print()

    ensemble_metrics = pd.DataFrame(metrics, index=[0])
    print(f"Ensemble metrics (storing to {args.output_dir})")
    print(ensemble_metrics[[
          c for c in ensemble_metrics.columns if "stratification" not in c]])
    ensemble_metrics.to_csv(
        args.output_dir / f"{args.method}_metrics.csv", index=False)

    # also store the cutoffs
    pd.DataFrame(cutoffs_to_store, index=[0]).to_csv(
        args.output_dir / f"{args.method}_stratification_cutoffs.csv",
        index=False
    )

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Gensheimer cancer VIT Model ensembles")
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
        help="Prefix of prediction columns in the prediction csv files. All columns starting with that value will be ensembled."
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
        "--train_outcome_file",
        type=str,
        help="Full path to a csv with outcome of training patients for metric computation. Same id_col, time_col and event_col are expected as given above."
    )
    parser.add_argument(
        "--train_outcome_file_sep",
        type=str,
        default=",",
        help="Column separator of the train_outcome_file"
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
        type=str,
        help="Location of a csv file that contains the cutoffs to stratify patients "
             "into low and high risk groups based on the gh predictions at certain times. "
             "Needed for computation of p-value of logrank test for ensemble. "
             "If not provided, will use the medians of the computed ensemble "
             "predictions."
    )
    parser.add_argument(
        "--method",
        type=str,
        default="gensheimer",
        help="To specify the output filenames"
    )

    print("\n\nComputing ensemble performance (Gensheimer)!\n")
    args = parser.parse_args()
    print(f"parsed args are\n{args}")

    if args.output_dir is None:
        args.output_dir = "./ensemble_gensheimer"
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
