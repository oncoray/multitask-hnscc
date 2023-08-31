import argparse
import numpy as np
import pandas as pd

from sklearn.model_selection import RepeatedStratifiedKFold
from pathlib import Path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create CV folds')
    parser.add_argument('--folds', type=int, default=5,
                        help='number of Folds to create')
    parser.add_argument('--reps', type=int, default=1,
                        help='number of repetitions of fold creation')
    parser.add_argument('--output', type=str, default="",
                        help="Directory for writing patient id files for the splits")
    # parser.add_argument('--validation_size', type=int,
    #                     help="Number of samples to use in validation splits.", default=30)
    parser.add_argument('--id_file', type=str, help="Path to the file containing"
                        " the patient ids that should be split.")
    parser.add_argument('--outcome', type=str,
                        help="Path to the outcome file (*.csv)")
    parser.add_argument('--event_col', type=str, default="OS",
                        help="name of the column containing the event indicator within the outcome. Used for stratified CV")
    parser.add_argument('--id_col', type=str, default="ID_Radiomics",
                        help="name of the column containing patient ids within the outcome.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outcome_sep", type=str, default=";",
                        help="column separation character in outcome file")

    args = parser.parse_args()
    outcome = pd.read_csv(args.outcome, sep=args.outcome_sep)[[args.id_col, args.event_col]]
    print("\noutcome shape: {}".format(outcome.shape))

    trainvalid_ids = pd.read_csv(args.id_file,
                                 header=None).iloc[:, 0].values
    print(f"{len(trainvalid_ids)} ids found in id_file.")

    trainvalid_outcome = outcome[outcome[args.id_col].isin(trainvalid_ids)]
    trainvalid_os = trainvalid_outcome[args.event_col].values.squeeze()

    skf = RepeatedStratifiedKFold(
        n_splits=args.folds,
        n_repeats=args.reps,
        random_state=args.seed)

    output_dir = Path(args.output)
    if not output_dir.exists:
        output_dir.mkdir(parents=True)

    for i, (train_index, valid_index) in enumerate(skf.split(trainvalid_ids, trainvalid_os)):

        rep_idx = i // args.folds
        fold_idx = i % args.folds

        rep_dir = output_dir / f"rep_{rep_idx}"
        if fold_idx == 0:
            # started a new rep and have to create directory
            rep_dir.mkdir(parents=True)
            print("\n=============")

        print(i, "rep", rep_idx, "fold", fold_idx)
        fraction_train = np.mean(trainvalid_os[train_index])
        fraction_valid = np.mean(trainvalid_os[valid_index])
        print(train_index.shape, valid_index.shape,
              fraction_train, fraction_valid)

        train_ids = sorted(trainvalid_ids[train_index])
        valid_ids = sorted(trainvalid_ids[valid_index])
        print(train_ids[:5])
        print(valid_ids[:5])
        pd.DataFrame(train_ids).to_csv(
            rep_dir / f"train_ids_fold_{fold_idx+1}.csv", index=False, header=None)
        pd.DataFrame(valid_ids).to_csv(
            rep_dir / f"valid_ids_fold_{fold_idx+1}.csv", index=False, header=None)
        print()
