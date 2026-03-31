#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


def sanitize_name(value: str) -> str:
    token = re.sub(r"[^0-9A-Za-z]+", "_", value).strip("_").lower()
    return token or "experiment"


def random_holdout_split(
    df: pd.DataFrame,
    holdout_ratio: float,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < holdout_ratio < 1.0:
        raise ValueError("holdout_ratio must be between 0 and 1.")
    if len(df) < 2:
        raise ValueError("Need at least 2 samples to create holdout split.")

    rng = np.random.default_rng(random_seed)
    indices = df.index.to_numpy().copy()
    rng.shuffle(indices)

    holdout_count = int(round(len(indices) * holdout_ratio))
    holdout_count = min(max(1, holdout_count), len(indices) - 1)

    holdout_idx = indices[:holdout_count]
    train_idx = indices[holdout_count:]
    return df.loc[train_idx].copy(), df.loc[holdout_idx].copy()


def assign_random_folds(train_df: pd.DataFrame, n_folds: int, random_seed: int) -> pd.DataFrame:
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2.")
    if len(train_df) < n_folds:
        raise ValueError("Training sample count must be >= n_folds.")

    rng = np.random.default_rng(random_seed)
    indices = train_df.index.to_numpy().copy()
    rng.shuffle(indices)

    fold_df = train_df.copy()
    fold_df["cv_fold"] = -1
    for order, index in enumerate(indices):
        fold_df.at[index, "cv_fold"] = order % n_folds
    return fold_df


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create an experiment split from a feature dataset. "
            "This script filters one training batch, applies random 80/20 holdout, "
            "assigns K-fold CV inside the training subset, and optionally attaches "
            "a whole-batch external test set."
        )
    )
    parser.add_argument(
        "--dataset-csv",
        default="data/features/default/dataset.csv",
        help="Feature dataset CSV path. Default: data/features/default/dataset.csv",
    )
    parser.add_argument(
        "--train-batch",
        default=None,
        help="Exact source_batch value to use for train/holdout/CV.",
    )
    parser.add_argument(
        "--train-batches",
        nargs="+",
        default=None,
        help="One or more source_batch values to use jointly for train/holdout/CV.",
    )
    parser.add_argument(
        "--external-test-batch",
        default=None,
        help="Exact source_batch value to use as a whole external test set.",
    )
    parser.add_argument(
        "--holdout-ratio",
        type=float,
        default=0.2,
        help="Holdout validation ratio. Default: 0.2",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=4,
        help="Number of CV folds within train split. Default: 4",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed. Default: 42",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Default: data/experiments/<train-batch>_holdout_cv",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    dataset_csv = Path(args.dataset_csv)
    df = pd.read_csv(dataset_csv)
    df = df.reset_index().rename(columns={"index": "row_id"})

    train_batches = list(args.train_batches or ([] if args.train_batch is None else [args.train_batch]))
    if not train_batches:
        raise RuntimeError("Provide --train-batch or --train-batches.")

    subset = df[df["source_batch"].isin(train_batches)].copy()
    if subset.empty:
        raise RuntimeError(f"No rows found for train batches: {train_batches}")

    external_test_df = pd.DataFrame(columns=df.columns)
    if args.external_test_batch:
        external_test_df = df[df["source_batch"] == args.external_test_batch].copy()
        if external_test_df.empty:
            raise RuntimeError(f"No rows found for external test batch: {args.external_test_batch}")

    train_df, holdout_df = random_holdout_split(
        subset,
        holdout_ratio=args.holdout_ratio,
        random_seed=args.random_seed,
    )
    train_df = assign_random_folds(
        train_df,
        n_folds=args.n_folds,
        random_seed=args.random_seed,
    )
    train_df["split"] = "train"
    holdout_df["split"] = "holdout"
    holdout_df["cv_fold"] = -1
    if not external_test_df.empty:
        external_test_df["split"] = "external_test"
        external_test_df["cv_fold"] = -1

    all_parts = [train_df, holdout_df]
    if not external_test_df.empty:
        all_parts.append(external_test_df)
    all_df = pd.concat(all_parts, ignore_index=True)
    all_df = all_df.sort_values(["split", "cell_id"]).reset_index(drop=True)
    train_df = train_df.sort_values(["cv_fold", "cell_id"]).reset_index(drop=True)
    holdout_df = holdout_df.sort_values(["cell_id"]).reset_index(drop=True)
    external_test_df = external_test_df.sort_values(["cell_id"]).reset_index(drop=True)

    default_name = "__".join(sanitize_name(batch) for batch in train_batches)
    default_dir = Path("data/experiments") / f"{default_name}_holdout_cv"
    output_dir = Path(args.output_dir) if args.output_dir else default_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    samples_path = output_dir / "samples.csv"
    train_path = output_dir / "train_cv.csv"
    holdout_path = output_dir / "holdout.csv"
    external_test_path = output_dir / "external_test.csv"
    meta_path = output_dir / "meta.json"

    all_df.to_csv(samples_path, index=False)
    train_df.to_csv(train_path, index=False)
    holdout_df.to_csv(holdout_path, index=False)
    if not external_test_df.empty:
        external_test_df.to_csv(external_test_path, index=False)

    meta = {
        "dataset_csv": str(dataset_csv),
        "train_batch": "__".join(train_batches),
        "train_batches": train_batches,
        "external_test_batch": args.external_test_batch or "",
        "holdout_ratio": args.holdout_ratio,
        "n_folds": args.n_folds,
        "random_seed": args.random_seed,
        "n_total_train_batch": int(len(subset)),
        "train_batch_counts": {
            str(batch): int(count) for batch, count in subset["source_batch"].value_counts().sort_index().items()
        },
        "n_train": int(len(train_df)),
        "n_holdout": int(len(holdout_df)),
        "n_external_test": int(len(external_test_df)),
        "fold_sizes": {
            str(int(fold)): int(count)
            for fold, count in train_df["cv_fold"].value_counts().sort_index().items()
        },
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

    print(f"[saved] {samples_path}")
    print(f"[saved] {train_path}")
    print(f"[saved] {holdout_path}")
    if not external_test_df.empty:
        print(f"[saved] {external_test_path}")
    print(f"[saved] {meta_path}")
    print(
        "[experiment] "
        f"train_batches={','.join(train_batches)} total={len(subset)} train={len(train_df)} "
        f"holdout={len(holdout_df)} external_test={len(external_test_df)} folds={args.n_folds}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
