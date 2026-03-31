#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


DEFAULT_CONFIG = {
    "early_cycles": 100,
    "target_key": "cycle_life",
    "include_sequence": True,
    "include_policy_one_hot": True,
    "policy_key": "policy",
    "summary_features": [
        "QDischarge",
        "QCharge",
        "IR",
        "Tmax",
        "Tavg",
        "Tmin",
        "chargetime",
    ],
    "aggregations": ["mean", "std", "min", "max", "first", "last", "delta", "slope"],
    "qdlin_pairs": [],
    "qdlin_stats": [],
    "core_features": [],
    "core_feature_params": {
        "cycle_a": 10,
        "cycle_b": 100,
        "charge_time_window": 10,
        "ir_early_window": 10,
        "ir_late_window": 10,
    },
}


def load_config(path: Path | None) -> dict[str, Any]:
    config = dict(DEFAULT_CONFIG)
    if path is None:
        return config
    user_config = json.loads(path.read_text())
    config.update(user_config)
    return config


def compute_slope(values: np.ndarray) -> float:
    x = np.arange(1, len(values) + 1, dtype=np.float64)
    mask = ~np.isnan(values)
    if mask.sum() < 2:
        return np.nan
    x = x[mask]
    y = values[mask]
    x_centered = x - x.mean()
    denom = np.dot(x_centered, x_centered)
    if denom == 0:
        return np.nan
    return float(np.dot(x_centered, y - y.mean()) / denom)


def aggregate(values: np.ndarray) -> dict[str, float]:
    return {
        "mean": float(np.nanmean(values)),
        "std": float(np.nanstd(values)),
        "min": float(np.nanmin(values)),
        "max": float(np.nanmax(values)),
        "first": float(values[0]),
        "last": float(values[-1]),
        "delta": float(values[-1] - values[0]),
        "slope": compute_slope(values),
    }


def summarize_curve_delta(delta: np.ndarray) -> dict[str, float]:
    clean = np.asarray(delta, dtype=np.float64).reshape(-1)
    peak_index = int(np.nanargmax(np.abs(clean))) if len(clean) else 0
    return {
        "mean": float(np.nanmean(clean)),
        "std": float(np.nanstd(clean)),
        "min": float(np.nanmin(clean)),
        "max": float(np.nanmax(clean)),
        "abs_mean": float(np.nanmean(np.abs(clean))),
        "abs_max": float(np.nanmax(np.abs(clean))),
        "l2": float(np.sqrt(np.nansum(clean**2))),
        "end_to_end": float(clean[-1] - clean[0]) if len(clean) else np.nan,
        "logvar": float(np.log(np.nanvar(clean) + 1e-12)),
        "peak_value": float(clean[peak_index]) if len(clean) else np.nan,
        "peak_index": float(peak_index) if len(clean) else np.nan,
    }


def normalize_string(value: Any) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None:
        return ""
    return str(value)


def sanitize_token(value: str) -> str:
    token = value.strip()
    if not token:
        return "empty"
    token = re.sub(r"[^0-9A-Za-z]+", "_", token)
    token = token.strip("_").lower()
    return token or "empty"


def load_summary(cell: Any, feature_name: str, early_cycles: int) -> np.ndarray | None:
    key = f"summary__{feature_name}"
    if key not in cell.files:
        return None
    values = np.asarray(cell[key], dtype=np.float64).reshape(-1)
    if len(values) < early_cycles:
        return None
    return values[:early_cycles]


def load_summary_full(cell: Any, feature_name: str) -> np.ndarray | None:
    key = f"summary__{feature_name}"
    if key not in cell.files:
        return None
    values = np.asarray(cell[key], dtype=np.float64).reshape(-1)
    return values if len(values) else None


def load_cycle_vector(cell: Any, field_name: str, cycle_number: int) -> np.ndarray | None:
    key = f"cycles__{field_name}"
    if key not in cell.files:
        return None
    values = cell[key]
    index = cycle_number - 1
    if index < 0 or index >= len(values):
        return None
    vector = np.asarray(values[index], dtype=np.float64).reshape(-1)
    if vector.size == 0:
        return None
    return vector


def cell_to_row(cell_path: Path, root_dir: Path, config: dict[str, Any]) -> dict[str, Any] | None:
    cell = np.load(cell_path, allow_pickle=True)
    target_key = config["target_key"]
    target = float(cell[target_key])
    if np.isnan(target):
        return None

    early_cycles = int(config["early_cycles"])
    if int(cell["cycle_count"]) < early_cycles:
        return None

    row: dict[str, Any] = {
        "source_batch": str(cell["source_batch"]),
        "cell_id": int(cell["cell_id"]),
        "cell_path": str(cell_path.relative_to(root_dir)),
        "policy": normalize_string(cell["policy"]),
        "target": target,
    }
    feature_aggregations = config.get("feature_aggregations", {})

    for feature_name in config["summary_features"]:
        values = load_summary(cell, feature_name, early_cycles)
        if values is None:
            return None

        if config["include_sequence"]:
            for cycle_idx, value in enumerate(values, start=1):
                row[f"{feature_name}_c{cycle_idx:03d}"] = float(value)

        stats = aggregate(values)
        agg_list = feature_aggregations.get(feature_name, config["aggregations"])
        for agg_name in agg_list:
            row[f"{feature_name}_{agg_name}"] = stats[agg_name]

    for pair in config.get("qdlin_pairs", []):
        if len(pair) != 2:
            raise RuntimeError(f"qdlin_pairs entry must have length 2: {pair}")
        cycle_a, cycle_b = int(pair[0]), int(pair[1])
        vec_a = load_cycle_vector(cell, "Qdlin", cycle_a)
        vec_b = load_cycle_vector(cell, "Qdlin", cycle_b)
        if vec_a is None or vec_b is None:
            return None
        if vec_a.shape != vec_b.shape:
            return None
        delta_stats = summarize_curve_delta(vec_a - vec_b)
        vdlin = np.asarray(cell["Vdlin"], dtype=np.float64).reshape(-1) if "Vdlin" in cell.files else np.array([])
        if vdlin.size == vec_a.size and "peak_index" in delta_stats:
            peak_idx = int(delta_stats["peak_index"])
            delta_stats["peak_voltage"] = float(vdlin[peak_idx])
        prefix = f"Qdlin_delta_c{cycle_a:03d}_c{cycle_b:03d}"
        for stat_name in config.get("qdlin_stats", []):
            if stat_name not in delta_stats:
                raise RuntimeError(f"Unsupported qdlin stat: {stat_name}")
            row[f"{prefix}_{stat_name}"] = delta_stats[stat_name]

    core_features = config.get("core_features", [])
    if core_features:
        params = config.get("core_feature_params", {})
        cycle_a = int(params.get("cycle_a", 10))
        cycle_b = int(params.get("cycle_b", early_cycles))
        charge_time_window = int(params.get("charge_time_window", 10))
        ir_early_window = int(params.get("ir_early_window", 10))
        ir_late_window = int(params.get("ir_late_window", 10))

        if "delta_q_peak_value" in core_features:
            vec_a = load_cycle_vector(cell, "Qdlin", cycle_a)
            vec_b = load_cycle_vector(cell, "Qdlin", cycle_b)
            if vec_a is None or vec_b is None or vec_a.shape != vec_b.shape:
                return None
            delta_q = vec_b - vec_a
            peak_index = int(np.nanargmax(np.abs(delta_q)))
            row[f"delta_q_peak_value_c{cycle_b:03d}_c{cycle_a:03d}"] = float(delta_q[peak_index])

        if "charge_time_first_mean" in core_features:
            ct_all = load_summary_full(cell, "chargetime")
            if ct_all is None or len(ct_all) < charge_time_window:
                return None
            row[f"charge_time_first_{charge_time_window:03d}_mean"] = float(
                np.nanmean(ct_all[:charge_time_window])
            )

        if "ir_increase_rate_early_cycles" in core_features:
            ir_early = load_summary(cell, "IR", early_cycles)
            if ir_early is None or len(ir_early) < max(ir_early_window, ir_late_window):
                return None
            first_window = ir_early[:ir_early_window]
            last_window = ir_early[-ir_late_window:]
            first_mean = float(np.nanmean(first_window))
            last_mean = float(np.nanmean(last_window))
            if first_mean == 0.0 or np.isnan(first_mean) or np.isnan(last_mean):
                return None
            row[f"ir_increase_rate_first{ir_early_window:03d}_last{ir_late_window:03d}_within_{early_cycles:03d}"] = (
                (last_mean - first_mean) / first_mean
            )

        if "ir_increase_rate_full_life" in core_features:
            ir_all = load_summary_full(cell, "IR")
            if ir_all is None or len(ir_all) < max(ir_early_window, ir_late_window):
                return None
            first_window = ir_all[:ir_early_window]
            last_window = ir_all[-ir_late_window:]
            first_mean = float(np.nanmean(first_window))
            last_mean = float(np.nanmean(last_window))
            if first_mean == 0.0 or np.isnan(first_mean) or np.isnan(last_mean):
                return None
            row[f"ir_increase_rate_first{ir_early_window:03d}_last{ir_late_window:03d}_full_life"] = (
                (last_mean - first_mean) / first_mean
            )

    return row


def collect_rows(canonical_dir: Path, config: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for cell_path in sorted(canonical_dir.glob("*/*cell_*.npz")):
        row = cell_to_row(cell_path, canonical_dir, config)
        if row is not None:
            rows.append(row)
    return rows


def collect_policy_vocabulary(canonical_dir: Path, policy_key: str) -> list[str]:
    values = set()
    raw_key = policy_key
    for cell_path in sorted(canonical_dir.glob("*/*cell_*.npz")):
        cell = np.load(cell_path, allow_pickle=True)
        if raw_key not in cell.files:
            continue
        value = normalize_string(cell[raw_key])
        if value:
            values.add(value)
    return sorted(values)


def append_policy_one_hot(
    df: pd.DataFrame,
    canonical_dir: Path,
    policy_key: str,
) -> tuple[pd.DataFrame, dict[str, str]]:
    vocabulary = collect_policy_vocabulary(canonical_dir, policy_key)
    mapping: dict[str, str] = {}
    encoded = pd.DataFrame(index=df.index)

    series = df[policy_key].fillna("").astype(str)
    for raw_value in vocabulary:
        column_name = f"policy__{sanitize_token(raw_value)}"
        suffix = 1
        base_name = column_name
        while column_name in mapping.values():
            suffix += 1
            column_name = f"{base_name}_{suffix}"
        mapping[raw_value] = column_name
        encoded[column_name] = (series == raw_value).astype(float)

    return pd.concat([df, encoded], axis=1), mapping


def feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {"source_batch", "cell_id", "cell_path", "policy", "target"}
    return [column for column in df.columns if column not in excluded]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a training dataset from canonical cell data. "
            "Feature selection is controlled by a JSON config."
        )
    )
    parser.add_argument(
        "--canonical-dir",
        default="data/canonical",
        help="Canonical dataset directory. Default: data/canonical",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to feature config JSON. Default: built-in config",
    )
    parser.add_argument(
        "--output-dir",
        default="data/features/default",
        help="Output directory for feature dataset. Default: data/features/default",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    config = load_config(Path(args.config) if args.config else None)
    canonical_dir = Path(args.canonical_dir)
    rows = collect_rows(canonical_dir, config)
    if not rows:
        raise RuntimeError("No valid samples produced. Check canonical data and config.")

    df = pd.DataFrame(rows).sort_values(["source_batch", "cell_id"]).reset_index(drop=True)
    policy_mapping: dict[str, str] = {}

    if config["include_policy_one_hot"]:
        policy_key = config.get("policy_key", "policy_readable")
        if policy_key != "policy":
            raise RuntimeError("Current default pipeline only supports policy_key='policy'.")
        df, policy_mapping = append_policy_one_hot(df, canonical_dir, policy_key)

    cols = feature_columns(df)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "dataset.csv"
    npz_path = output_dir / "dataset.npz"
    meta_path = output_dir / "meta.json"

    df.to_csv(csv_path, index=False)
    np.savez_compressed(
        npz_path,
        X=df[cols].to_numpy(dtype=np.float64),
        y=df["target"].to_numpy(dtype=np.float64),
        feature_names=np.asarray(cols, dtype=str),
        source_batch=df["source_batch"].to_numpy(dtype=str),
        cell_id=df["cell_id"].to_numpy(dtype=np.int32),
        cell_path=df["cell_path"].to_numpy(dtype=str),
    )
    meta_path.write_text(
        json.dumps(
            {
                "config": config,
                "n_samples": int(df.shape[0]),
                "n_features": int(len(cols)),
                "feature_names": cols,
                "policy_one_hot_mapping": policy_mapping,
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    print(f"[saved] {csv_path}")
    print(f"[saved] {npz_path}")
    print(f"[saved] {meta_path}")
    print(f"[dataset] samples={df.shape[0]} features={len(cols)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
