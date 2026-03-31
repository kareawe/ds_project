#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiment_tracker import log_experiment, make_run_id, utc_timestamp


EXCLUDED_COLUMNS = {
    "row_id",
    "source_batch",
    "cell_id",
    "cell_path",
    "policy",
    "policy_readable",
    "target",
    "split",
    "cv_fold",
}


def numeric_feature_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for column in df.columns:
        if column in EXCLUDED_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df[column]):
            cols.append(column)
    return cols


def impute_fit(X: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        fill_values = np.nanmean(X, axis=0)
    fill_values[np.isnan(fill_values)] = 0.0
    return fill_values


def impute_apply(X: np.ndarray, fill_values: np.ndarray) -> np.ndarray:
    return np.where(np.isnan(X), fill_values, X)


def standardize_fit(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0.0] = 1.0
    return mean, std


def standardize_apply(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (X - mean) / std


def fit_ridge_regression(X: np.ndarray, y: np.ndarray, reg_lambda: float) -> tuple[np.ndarray, float]:
    n_features = X.shape[1]
    X_aug = np.hstack([X, np.ones((X.shape[0], 1), dtype=np.float64)])
    if reg_lambda > 0.0:
        ridge_rows = np.sqrt(reg_lambda) * np.eye(n_features + 1, dtype=np.float64)
        ridge_rows[-1, -1] = 0.0
        A = np.vstack([X_aug, ridge_rows])
        b = np.concatenate([y, np.zeros(n_features + 1, dtype=np.float64)])
    else:
        A = X_aug
        b = y

    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    return coef[:-1], float(coef[-1])


def predict(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        return X @ weights + bias


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
    error = y_pred - y_true
    mse = float(np.mean(error**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(error)))
    nonzero_mask = y_true != 0.0
    if np.any(nonzero_mask):
        avg_percent_error = float(
            np.mean(np.abs(error[nonzero_mask]) / np.abs(y_true[nonzero_mask])) * 100.0
        )
    else:
        avg_percent_error = float("nan")

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot != 0.0 else 0.0

    return {
        "n_samples": int(len(y_true)),
        "rmse": rmse,
        "avg_percent_error": avg_percent_error,
        "mae": mae,
        "r2": r2,
    }


def prepare_arrays(df: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray]:
    X = df[feature_cols].to_numpy(dtype=np.float64)
    y = df["target"].to_numpy(dtype=np.float64)
    return X, y


def fit_and_eval(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_cols: list[str],
    reg_lambda: float,
) -> tuple[dict[str, Any], np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    X_train, y_train = prepare_arrays(train_df, feature_cols)
    X_eval, y_eval = prepare_arrays(eval_df, feature_cols)

    fill_values = impute_fit(X_train)
    X_train = impute_apply(X_train, fill_values)
    X_eval = impute_apply(X_eval, fill_values)

    mean, std = standardize_fit(X_train)
    X_train = standardize_apply(X_train, mean, std)
    X_eval = standardize_apply(X_eval, mean, std)

    weights, bias = fit_ridge_regression(X_train, y_train, reg_lambda)
    y_pred = predict(X_eval, weights, bias)
    metrics = regression_metrics(y_eval, y_pred)
    return metrics, weights, bias, fill_values, mean, std


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train linear regression from experiment split files. "
            "Metadata columns are dropped automatically at train time."
        )
    )
    parser.add_argument(
        "--split-dir",
        default="data/experiments/2017-05-12_holdout_cv",
        help="Experiment split directory containing train_cv.csv and holdout.csv.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to store model and metrics. Default: <split-dir>/linear_regression",
    )
    parser.add_argument(
        "--reg-lambda",
        type=float,
        default=1.0,
        help="L2 regularization strength for ridge regression. Default: 1.0",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    split_dir = Path(args.split_dir)
    train_df = pd.read_csv(split_dir / "train_cv.csv")
    holdout_df = pd.read_csv(split_dir / "holdout.csv")
    external_test_path = split_dir / "external_test.csv"
    external_test_df = pd.read_csv(external_test_path) if external_test_path.exists() else None
    split_meta = {}
    split_meta_path = split_dir / "meta.json"
    if split_meta_path.exists():
        split_meta = json.loads(split_meta_path.read_text())

    dataset_meta = {}
    dataset_csv_path = Path(split_meta.get("dataset_csv", "")) if split_meta else None
    if dataset_csv_path and dataset_csv_path.exists():
        candidate = dataset_csv_path.parent / "meta.json"
        if candidate.exists():
            dataset_meta = json.loads(candidate.read_text())

    feature_cols = numeric_feature_columns(train_df)
    if not feature_cols:
        raise RuntimeError("No numeric feature columns found for training.")

    fold_metrics = []
    for fold_id in sorted(train_df["cv_fold"].unique().tolist()):
        fold_train = train_df[train_df["cv_fold"] != fold_id].reset_index(drop=True)
        fold_val = train_df[train_df["cv_fold"] == fold_id].reset_index(drop=True)
        metrics, _, _, _, _, _ = fit_and_eval(
            train_df=fold_train,
            eval_df=fold_val,
            feature_cols=feature_cols,
            reg_lambda=args.reg_lambda,
        )
        metrics["fold"] = int(fold_id)
        fold_metrics.append(metrics)

    holdout_metrics, weights, bias, fill_values, mean, std = fit_and_eval(
        train_df=train_df,
        eval_df=holdout_df,
        feature_cols=feature_cols,
        reg_lambda=args.reg_lambda,
    )
    external_test_metrics = None
    if external_test_df is not None and not external_test_df.empty:
        external_test_metrics, _, _, _, _, _ = fit_and_eval(
            train_df=train_df,
            eval_df=external_test_df,
            feature_cols=feature_cols,
            reg_lambda=args.reg_lambda,
        )

    cv_summary = {}
    for key in ["rmse", "avg_percent_error", "mae", "r2"]:
        values = [metric[key] for metric in fold_metrics]
        cv_summary[f"{key}_mean"] = float(np.mean(values))
        cv_summary[f"{key}_std"] = float(np.std(values))

    output_dir = Path(args.output_dir) if args.output_dir else split_dir / "linear_regression"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "model.npz"
    metrics_path = output_dir / "metrics.json"

    np.savez_compressed(
        model_path,
        weights=weights,
        bias=np.asarray(bias, dtype=np.float64),
        impute_values=fill_values,
        mean=mean,
        std=std,
        feature_names=np.asarray(feature_cols, dtype=str),
    )

    metrics_path.write_text(
        json.dumps(
            {
                "split_dir": str(split_dir),
                "reg_lambda": args.reg_lambda,
                "n_train": int(len(train_df)),
                "n_holdout": int(len(holdout_df)),
                "n_external_test": int(len(external_test_df)) if external_test_df is not None else 0,
                "n_features": int(len(feature_cols)),
                "excluded_columns": sorted(EXCLUDED_COLUMNS),
                "cv_folds": fold_metrics,
                "cv_summary": cv_summary,
                "holdout": holdout_metrics,
                "external_test": external_test_metrics,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    run_record = {
        "timestamp_utc": utc_timestamp(),
        "run_id": make_run_id("linear"),
        "model_name": "linear_regression",
        "split_dir": str(split_dir),
        "split_batch": split_meta.get("train_batch", split_meta.get("batch", "")),
        "external_test_batch": split_meta.get("external_test_batch", ""),
        "dataset_csv": str(dataset_csv_path) if dataset_csv_path else "",
        "n_train": int(len(train_df)),
        "n_holdout": int(len(holdout_df)),
        "n_external_test": int(len(external_test_df)) if external_test_df is not None else 0,
        "n_features": int(len(feature_cols)),
        "reg_lambda": float(args.reg_lambda),
        "feature_names_path": str(model_path),
        "metrics_path": str(metrics_path),
        "model_path": str(model_path),
        "cv_rmse_mean": cv_summary["rmse_mean"],
        "cv_rmse_std": cv_summary["rmse_std"],
        "cv_avg_percent_error_mean": cv_summary["avg_percent_error_mean"],
        "cv_avg_percent_error_std": cv_summary["avg_percent_error_std"],
        "cv_mae_mean": cv_summary["mae_mean"],
        "cv_mae_std": cv_summary["mae_std"],
        "cv_r2_mean": cv_summary["r2_mean"],
        "cv_r2_std": cv_summary["r2_std"],
        "holdout_rmse": holdout_metrics["rmse"],
        "holdout_avg_percent_error": holdout_metrics["avg_percent_error"],
        "holdout_mae": holdout_metrics["mae"],
        "holdout_r2": holdout_metrics["r2"],
        "external_test_rmse": external_test_metrics["rmse"] if external_test_metrics else None,
        "external_test_avg_percent_error": external_test_metrics["avg_percent_error"] if external_test_metrics else None,
        "external_test_mae": external_test_metrics["mae"] if external_test_metrics else None,
        "external_test_r2": external_test_metrics["r2"] if external_test_metrics else None,
        "feature_config": dataset_meta.get("config", {}),
        "feature_meta_path": str((dataset_csv_path.parent / 'meta.json')) if dataset_csv_path else "",
        "cv_folds": fold_metrics,
    }
    log_experiment(Path("analysis"), run_record)

    print(f"[saved] {model_path}")
    print(f"[saved] {metrics_path}")
    print("[logged] analysis/experiment_registry.csv")
    print(
        "[cv] "
        f"rmse_mean={cv_summary['rmse_mean']:.4f} "
        f"avg_percent_error_mean={cv_summary['avg_percent_error_mean']:.4f} "
        f"mae_mean={cv_summary['mae_mean']:.4f} "
        f"r2_mean={cv_summary['r2_mean']:.4f}"
    )
    print(
        "[holdout] "
        f"rmse={holdout_metrics['rmse']:.4f} "
        f"avg_percent_error={holdout_metrics['avg_percent_error']:.4f} "
        f"mae={holdout_metrics['mae']:.4f} "
        f"r2={holdout_metrics['r2']:.4f}"
    )
    if external_test_metrics:
        print(
            "[external_test] "
            f"rmse={external_test_metrics['rmse']:.4f} "
            f"avg_percent_error={external_test_metrics['avg_percent_error']:.4f} "
            f"mae={external_test_metrics['mae']:.4f} "
            f"r2={external_test_metrics['r2']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
