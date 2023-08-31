import pandas as pd
from pathlib import Path
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--result_dir",
        type=str,
        required=True,
        help="Full path to directory that contains directories with results for each fold"
    )
    parser.add_argument(
        "--metric_filename",
        type=str,
        default="metrics.csv",
        help="Name of file that contains the metrics"
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default=None
    )

    args = parser.parse_args()
    print("parsed args")
    print(args)

    result_dir = Path(args.result_dir)
    print("result_dir", result_dir)
    files = sorted(result_dir.rglob(f"**/{args.metric_filename}"))
    print(files)
    files_relative = [f.relative_to(result_dir) for f in files]
    print(files_relative)

    metric_dfs = []
    for f in files_relative:
        # split the relative path into parts and check if some part begins with rep_ and some begins with fold_
        parts = f.parts
        poss_rep = [int(part.split("rep_")[1]) for part in parts if part.startswith("rep_")]
        if len(poss_rep) == 0:
            rep = 0
        elif len(poss_rep) > 1:
            raise ValueError("'rep_' should not occur more than once in the path")
        else:
            rep = poss_rep[0]

        poss_fold = [int(part.split("fold_")[1]) for part in parts if part.startswith("fold_")]
        if len(poss_fold) == 0:
            fold_nr = 0
        elif len(poss_fold) > 1:
            raise ValueError("'fold_' should not occur more than once in the path")
        else:
            fold_nr = poss_fold[0]

        subdir = f.parent.name

        df = pd.read_csv(result_dir / f)
        df["rep"] = rep
        df["fold"] = fold_nr
        df["subset"] = subdir
        metric_dfs.append(df)

    metric_dfs = pd.concat(metric_dfs)
    # move fold and subset columns to the front
    cols = [c for c in metric_dfs.columns if "Unnamed" not in c]
    first_cols = ["rep", "fold", "subset"]
    col_order = first_cols + [c for c in cols if c not in first_cols]

    metric_dfs = metric_dfs[col_order]
    print(metric_dfs)

    if args.output_filename is None:
        output_filename = args.metric_filename
    else:
        output_filename = args.output_filename
    save_path = result_dir / f"cv-results_{output_filename}"
    print("Storing cv results to", save_path)
    metric_dfs.to_csv(save_path, index=False)

    for subset, df in metric_dfs.groupby("subset"):
        print(subset)
        print(df.describe())
        if "stratification_logrank_pval" in metric_dfs.columns:
            n_sign = df[df["stratification_logrank_pval"] < 0.05].shape[0]
            print(f"{n_sign}/{len(df)} with significant stratification.")
        print()
