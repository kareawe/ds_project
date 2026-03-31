#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet

from logger import log_experiment, make_run_id, utc_timestamp


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


def fit_elastic_net_sklearn(
    X: np.ndarray,
    y_log: np.ndarray,
    reg_lambda: float,
    l1_ratio: float,
    max_iter: int,
    tol: float,
    sample_weight: np.ndarray | None,
) -> tuple[np.ndarray, float, int, bool]:
    model = ElasticNet(
        alpha=reg_lambda,
        l1_ratio=l1_ratio,
        fit_intercept=True,
        max_iter=max_iter,
        tol=tol,
        selection="cyclic",
        random_state=0,
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(X, y_log, sample_weight=sample_weight)
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    return (
        model.coef_.astype(np.float64, copy=False),
        float(model.intercept_),
        int(model.n_iter_),
        converged,
    )


def predict_log_target(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        return X @ weights + bias


def clip_log_predictions(
    y_pred_log: np.ndarray,
    y_train_log_min: float,
    y_train_log_max: float,
    mode: str,
) -> np.ndarray:
    if mode == "none":
        return y_pred_log
    if mode == "upper":
        return np.minimum(y_pred_log, y_train_log_max)
    if mode == "range":
        return np.clip(y_pred_log, y_train_log_min, y_train_log_max)
    raise ValueError(f"Unsupported log prediction clip mode: {mode}")


def compute_sample_weights(y: np.ndarray, scheme: str) -> np.ndarray | None:
    if scheme == "none":
        return None

    y = np.asarray(y, dtype=np.float64)
    if np.any(y <= 0.0):
        raise ValueError("Sample weighting requires strictly positive targets.")

    if scheme == "inverse_cycle_life":
        raw = 1.0 / y
    elif scheme == "inverse_sqrt_cycle_life":
        raw = 1.0 / np.sqrt(y)
    elif scheme == "linear_low_life":
        span = float(np.max(y) - np.min(y))
        if span <= 0.0:
            raw = np.ones_like(y)
        else:
            progress = (np.max(y) - y) / span
            raw = 1.0 + progress
    elif scheme == "quadratic_low_life":
        span = float(np.max(y) - np.min(y))
        if span <= 0.0:
            raw = np.ones_like(y)
        else:
            progress = (np.max(y) - y) / span
            raw = 1.0 + progress**2
    else:
        raise ValueError(f"Unsupported sample-weight scheme: {scheme}")

    mean = float(np.mean(raw))
    if mean <= 0.0:
        return None
    return raw / mean


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


def preprocess_train_eval(
    X_train_raw: np.ndarray,
    X_eval_raw: np.ndarray,
    zscore_clip: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    fill_values = impute_fit(X_train_raw)
    X_train = impute_apply(X_train_raw, fill_values)
    X_eval = impute_apply(X_eval_raw, fill_values)

    mean, std = standardize_fit(X_train)
    X_train = standardize_apply(X_train, mean, std)
    X_eval = standardize_apply(X_eval, mean, std)
    if zscore_clip is not None and zscore_clip > 0.0:
        X_train = np.clip(X_train, -zscore_clip, zscore_clip)
        X_eval = np.clip(X_eval, -zscore_clip, zscore_clip)
    return X_train, X_eval, fill_values, mean, std


def fit_and_eval(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    feature_cols: list[str],
    reg_lambda: float,
    l1_ratio: float,
    max_iter: int,
    tol: float,
    zscore_clip: float | None,
    log_prediction_clip_mode: str,
    sample_weight_scheme: str,
) -> tuple[dict[str, Any], np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
    X_train_raw, y_train = prepare_arrays(train_df, feature_cols)
    X_eval_raw, y_eval = prepare_arrays(eval_df, feature_cols)
    if np.any(y_train <= 0.0) or np.any(y_eval <= 0.0):
        raise ValueError("log(cycle_life) requires strictly positive target values.")

    X_train, X_eval, fill_values, mean, std = preprocess_train_eval(
        X_train_raw,
        X_eval_raw,
        zscore_clip=zscore_clip,
    )
    y_train_log = np.log(y_train)
    sample_weight = compute_sample_weights(y_train, sample_weight_scheme)
    y_train_log_min = float(np.min(y_train_log))
    y_train_log_max = float(np.max(y_train_log))

    weights, bias, n_iter, converged = fit_elastic_net_sklearn(
        X_train,
        y_train_log,
        reg_lambda=reg_lambda,
        l1_ratio=l1_ratio,
        max_iter=max_iter,
        tol=tol,
        sample_weight=sample_weight,
    )
    y_eval_log_pred = predict_log_target(X_eval, weights, bias)
    unclipped_log_pred = y_eval_log_pred.copy()
    y_eval_log_pred = clip_log_predictions(
        y_eval_log_pred,
        y_train_log_min=y_train_log_min,
        y_train_log_max=y_train_log_max,
        mode=log_prediction_clip_mode,
    )
    y_eval_pred = np.exp(y_eval_log_pred)
    metrics = regression_metrics(y_eval, y_eval_pred)
    metrics["optimizer_iters"] = int(n_iter)
    metrics["optimizer_converged"] = bool(converged)
    metrics["n_nonzero_features"] = int(np.count_nonzero(np.abs(weights) > 1e-12))
    metrics["pred_log_min"] = float(np.min(y_eval_log_pred))
    metrics["pred_log_max"] = float(np.max(y_eval_log_pred))
    metrics["pred_cycle_min"] = float(np.min(y_eval_pred))
    metrics["pred_cycle_max"] = float(np.max(y_eval_pred))
    metrics["n_clipped_log_predictions"] = int(
        np.count_nonzero(np.abs(unclipped_log_pred - y_eval_log_pred) > 1e-12)
    )
    metrics["train_log_target_min"] = y_train_log_min
    metrics["train_log_target_max"] = y_train_log_max
    metrics["log_prediction_clip_mode"] = log_prediction_clip_mode
    metrics["sample_weight_scheme"] = sample_weight_scheme
    if sample_weight is not None:
        metrics["train_sample_weight_min"] = float(np.min(sample_weight))
        metrics["train_sample_weight_max"] = float(np.max(sample_weight))
    return metrics, weights, bias, fill_values, mean, std


def summarize_cv_results(results: list[dict[str, Any]]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for key in ["rmse", "avg_percent_error", "mae", "r2", "n_nonzero_features"]:
        values = [float(result[key]) for result in results]
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_std"] = float(np.std(values))
    converged_ratio = np.mean([1.0 if result["optimizer_converged"] else 0.0 for result in results])
    summary["optimizer_converged_ratio"] = float(converged_ratio)
    summary["optimizer_iters_mean"] = float(np.mean([result["optimizer_iters"] for result in results]))
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train elastic net on log(cycle_life) from experiment split files. "
            "Evaluation is reported on the original cycle-life scale."
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
        help="Directory to store model and metrics. Default: <split-dir>/elastic_net_log",
    )
    parser.add_argument(
        "--lambda-grid",
        nargs="+",
        type=float,
        default=[0.001, 0.01, 0.1, 1.0, 10.0],
        help="Regularization strengths to try.",
    )
    parser.add_argument(
        "--l1-ratio-grid",
        nargs="+",
        type=float,
        default=[0.1, 0.3, 0.5, 0.7, 0.9],
        help="Elastic net L1 ratios to try. 1.0 becomes lasso, 0.0 becomes ridge-like.",
    )
    parser.add_argument(
        "--selection-metric",
        default="rmse",
        choices=["rmse", "avg_percent_error"],
        help="Cross-validation metric used to select best hyperparameters.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=5000,
        help="Maximum coordinate descent iterations per fit.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-4,
        help="Coordinate descent tolerance.",
    )
    parser.add_argument(
        "--zscore-clip",
        type=float,
        default=5.0,
        help="Clip standardized features into [-zscore_clip, zscore_clip]. Use <=0 to disable.",
    )
    parser.add_argument(
        "--disable-log-target-clipping",
        action="store_true",
        help="Disable clipping predicted log(cycle_life) to the training target range.",
    )
    parser.add_argument(
        "--log-prediction-clip-mode",
        default=None,
        choices=["range", "upper", "none"],
        help="Override log-prediction clipping mode. Defaults to 'range' unless --disable-log-target-clipping is set.",
    )
    parser.add_argument(
        "--sample-weight-scheme",
        default="none",
        choices=[
            "none",
            "inverse_cycle_life",
            "inverse_sqrt_cycle_life",
            "linear_low_life",
            "quadratic_low_life",
        ],
        help="Optional sample-weighting scheme applied on training targets.",
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
    zscore_clip = args.zscore_clip if args.zscore_clip and args.zscore_clip > 0.0 else None
    if args.log_prediction_clip_mode is not None:
        log_prediction_clip_mode = args.log_prediction_clip_mode
    else:
        log_prediction_clip_mode = "none" if args.disable_log_target_clipping else "range"

    search_results = []
    best_candidate: dict[str, Any] | None = None

    for reg_lambda in args.lambda_grid:
        for l1_ratio in args.l1_ratio_grid:
            fold_metrics = []
            for fold_id in sorted(train_df["cv_fold"].unique().tolist()):
                fold_train = train_df[train_df["cv_fold"] != fold_id].reset_index(drop=True)
                fold_val = train_df[train_df["cv_fold"] == fold_id].reset_index(drop=True)
                metrics, _, _, _, _, _ = fit_and_eval(
                    train_df=fold_train,
                    eval_df=fold_val,
                    feature_cols=feature_cols,
                    reg_lambda=reg_lambda,
                    l1_ratio=l1_ratio,
                    max_iter=args.max_iter,
                    tol=args.tol,
                    zscore_clip=zscore_clip,
                    log_prediction_clip_mode=log_prediction_clip_mode,
                    sample_weight_scheme=args.sample_weight_scheme,
                )
                metrics["fold"] = int(fold_id)
                fold_metrics.append(metrics)

            cv_summary = summarize_cv_results(fold_metrics)
            candidate = {
                "reg_lambda": float(reg_lambda),
                "l1_ratio": float(l1_ratio),
                "selection_score": float(cv_summary[f"{args.selection_metric}_mean"]),
                "cv_summary": cv_summary,
                "cv_folds": fold_metrics,
            }
            search_results.append(candidate)

            if best_candidate is None:
                best_candidate = candidate
                continue

            current_score = candidate["selection_score"]
            best_score = best_candidate["selection_score"]
            if current_score < best_score - 1e-12:
                best_candidate = candidate
            elif abs(current_score - best_score) <= 1e-12:
                if candidate["cv_summary"]["avg_percent_error_mean"] < best_candidate["cv_summary"]["avg_percent_error_mean"]:
                    best_candidate = candidate

    if best_candidate is None:
        raise RuntimeError("Elastic net hyperparameter search produced no candidates.")

    best_lambda = float(best_candidate["reg_lambda"])
    best_l1_ratio = float(best_candidate["l1_ratio"])
    holdout_metrics, weights, bias, fill_values, mean, std = fit_and_eval(
        train_df=train_df,
        eval_df=holdout_df,
        feature_cols=feature_cols,
        reg_lambda=best_lambda,
        l1_ratio=best_l1_ratio,
        max_iter=args.max_iter,
        tol=args.tol,
        zscore_clip=zscore_clip,
        log_prediction_clip_mode=log_prediction_clip_mode,
        sample_weight_scheme=args.sample_weight_scheme,
    )
    external_test_metrics = None
    if external_test_df is not None and not external_test_df.empty:
        external_test_metrics, _, _, _, _, _ = fit_and_eval(
            train_df=train_df,
            eval_df=external_test_df,
            feature_cols=feature_cols,
            reg_lambda=best_lambda,
            l1_ratio=best_l1_ratio,
            max_iter=args.max_iter,
            tol=args.tol,
            zscore_clip=zscore_clip,
            log_prediction_clip_mode=log_prediction_clip_mode,
            sample_weight_scheme=args.sample_weight_scheme,
        )

    output_dir = Path(args.output_dir) if args.output_dir else split_dir / "elastic_net_log"
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
        reg_lambda=np.asarray(best_lambda, dtype=np.float64),
        l1_ratio=np.asarray(best_l1_ratio, dtype=np.float64),
        target_transform=np.asarray("log", dtype=str),
        zscore_clip=np.asarray(-1.0 if zscore_clip is None else zscore_clip, dtype=np.float64),
        log_prediction_clip_mode=np.asarray(log_prediction_clip_mode, dtype=str),
        sample_weight_scheme=np.asarray(args.sample_weight_scheme, dtype=str),
    )

    metrics_payload = {
        "split_dir": str(split_dir),
        "model_name": "elastic_net",
        "target_transform": "log",
        "selection_metric": args.selection_metric,
        "zscore_clip": zscore_clip,
        "log_prediction_clip_mode": log_prediction_clip_mode,
        "sample_weight_scheme": args.sample_weight_scheme,
        "lambda_grid": [float(value) for value in args.lambda_grid],
        "l1_ratio_grid": [float(value) for value in args.l1_ratio_grid],
        "best_reg_lambda": best_lambda,
        "best_l1_ratio": best_l1_ratio,
        "n_train": int(len(train_df)),
        "n_holdout": int(len(holdout_df)),
        "n_external_test": int(len(external_test_df)) if external_test_df is not None else 0,
        "n_features": int(len(feature_cols)),
        "n_nonzero_features": int(np.count_nonzero(np.abs(weights) > 1e-12)),
        "excluded_columns": sorted(EXCLUDED_COLUMNS),
        "search_results": search_results,
        "cv_folds": best_candidate["cv_folds"],
        "cv_summary": best_candidate["cv_summary"],
        "holdout": holdout_metrics,
        "external_test": external_test_metrics,
    }
    metrics_path.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2))

    run_record = {
        "timestamp_utc": utc_timestamp(),
        "run_id": make_run_id("elastic"),
        "model_name": "elastic_net_log_cycle_life",
        "split_dir": str(split_dir),
        "split_batch": split_meta.get("train_batch", split_meta.get("batch", "")),
        "external_test_batch": split_meta.get("external_test_batch", ""),
        "dataset_csv": str(dataset_csv_path) if dataset_csv_path else "",
        "n_train": int(len(train_df)),
        "n_holdout": int(len(holdout_df)),
        "n_external_test": int(len(external_test_df)) if external_test_df is not None else 0,
        "n_features": int(len(feature_cols)),
        "n_nonzero_features": int(np.count_nonzero(np.abs(weights) > 1e-12)),
        "best_reg_lambda": best_lambda,
        "best_l1_ratio": best_l1_ratio,
        "selection_metric": args.selection_metric,
        "zscore_clip": zscore_clip,
        "log_prediction_clip_mode": log_prediction_clip_mode,
        "sample_weight_scheme": args.sample_weight_scheme,
        "feature_names_path": str(model_path),
        "metrics_path": str(metrics_path),
        "model_path": str(model_path),
        "cv_rmse_mean": best_candidate["cv_summary"]["rmse_mean"],
        "cv_rmse_std": best_candidate["cv_summary"]["rmse_std"],
        "cv_avg_percent_error_mean": best_candidate["cv_summary"]["avg_percent_error_mean"],
        "cv_avg_percent_error_std": best_candidate["cv_summary"]["avg_percent_error_std"],
        "cv_mae_mean": best_candidate["cv_summary"]["mae_mean"],
        "cv_mae_std": best_candidate["cv_summary"]["mae_std"],
        "cv_r2_mean": best_candidate["cv_summary"]["r2_mean"],
        "cv_r2_std": best_candidate["cv_summary"]["r2_std"],
        "holdout_rmse": holdout_metrics["rmse"],
        "holdout_avg_percent_error": holdout_metrics["avg_percent_error"],
        "holdout_mae": holdout_metrics["mae"],
        "holdout_r2": holdout_metrics["r2"],
        "external_test_rmse": external_test_metrics["rmse"] if external_test_metrics else None,
        "external_test_avg_percent_error": external_test_metrics["avg_percent_error"] if external_test_metrics else None,
        "external_test_mae": external_test_metrics["mae"] if external_test_metrics else None,
        "external_test_r2": external_test_metrics["r2"] if external_test_metrics else None,
        "target_transform": "log",
        "feature_config": dataset_meta.get("config", {}),
        "feature_meta_path": str((dataset_csv_path.parent / "meta.json")) if dataset_csv_path else "",
        "cv_folds": best_candidate["cv_folds"],
    }
    log_experiment(Path("analysis"), run_record)

    print(f"[saved] {model_path}")
    print(f"[saved] {metrics_path}")
    print("[logged] analysis/experiment_registry.csv")
    print(
        "[best] "
        f"lambda={best_lambda:.6f} "
        f"l1_ratio={best_l1_ratio:.3f} "
        f"nonzero={run_record['n_nonzero_features']}"
    )
    print(
        "[cv] "
        f"rmse_mean={best_candidate['cv_summary']['rmse_mean']:.4f} "
        f"avg_percent_error_mean={best_candidate['cv_summary']['avg_percent_error_mean']:.4f}"
    )
    print(
        "[holdout] "
        f"rmse={holdout_metrics['rmse']:.4f} "
        f"avg_percent_error={holdout_metrics['avg_percent_error']:.4f}"
    )
    if external_test_metrics:
        print(
            "[external_test] "
            f"rmse={external_test_metrics['rmse']:.4f} "
            f"avg_percent_error={external_test_metrics['avg_percent_error']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
