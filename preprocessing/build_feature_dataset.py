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

    for feature_name in config["summary_features"]:
        values = load_summary(cell, feature_name, early_cycles)
        if values is None:
            return None

        if config["include_sequence"]:
            for cycle_idx, value in enumerate(values, start=1):
                row[f"{feature_name}_c{cycle_idx:03d}"] = float(value)

        stats = aggregate(values)
        for agg_name in config["aggregations"]:
            row[f"{feature_name}_{agg_name}"] = stats[agg_name]

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
