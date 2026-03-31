#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from common import load_mat, normalize_batch, normalize_float, normalize_string, resolve_paths


def to_object_array(values: list[Any]) -> np.ndarray:
    array = np.empty(len(values), dtype=object)
    for index, value in enumerate(values):
        array[index] = value
    return array


def cycle_field_names(cycles: list[dict[str, Any]]) -> list[str]:
    names: set[str] = set()
    for cycle in cycles:
        names.update(cycle.keys())
    return sorted(names)


def export_cell(cell: dict[str, Any], cell_id: int, batch_name: str, output_path: Path) -> None:
    cycles = cell.get("cycles") or []
    summary = cell.get("summary") or {}

    arrays: dict[str, np.ndarray] = {
        "source_batch": np.asarray(batch_name),
        "cell_id": np.asarray(cell_id, dtype=np.int32),
        "cycle_life": np.asarray(normalize_float(cell.get("cycle_life")), dtype=np.float64),
        "policy": np.asarray(normalize_string(cell.get("policy"))),
        "policy_readable": np.asarray(normalize_string(cell.get("policy_readable"))),
        "barcode": np.asarray(normalize_string(cell.get("barcode"))),
        "channel_id": np.asarray(normalize_string(cell.get("channel_id"))),
        "cycle_count": np.asarray(len(cycles), dtype=np.int32),
        "Vdlin": np.asarray(cell.get("Vdlin")).reshape(-1) if cell.get("Vdlin") is not None else np.array([]),
    }

    for key, value in sorted(summary.items()):
        arrays[f"summary__{key}"] = np.asarray(value).reshape(-1) if value is not None else np.array([])

    arrays["cycles__index"] = np.arange(1, len(cycles) + 1, dtype=np.int32)
    for field_name in cycle_field_names(cycles):
        series = []
        for cycle in cycles:
            value = cycle.get(field_name)
            series.append(np.asarray(value) if value is not None else np.array([]))
        arrays[f"cycles__{field_name}"] = to_object_array(series)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)


def export_batch(mat_path: Path, output_root: Path) -> tuple[Path, int]:
    raw = load_mat(mat_path)
    if "batch" not in raw:
        raise KeyError(f"'batch' key not found in {mat_path.name}")

    batch = normalize_batch(raw["batch"])
    batch_name = mat_path.stem
    batch_dir = output_root / batch_name
    batch_dir.mkdir(parents=True, exist_ok=True)

    cell_paths = []
    cycle_life = []
    policies = []
    policies_readable = []

    for cell_id, cell in enumerate(batch):
        cell_path = batch_dir / f"cell_{cell_id:03d}.npz"
        export_cell(cell, cell_id, batch_name, cell_path)
        cell_paths.append(str(cell_path.relative_to(output_root)))
        cycle_life.append(normalize_float(cell.get("cycle_life")))
        policies.append(normalize_string(cell.get("policy")))
        policies_readable.append(normalize_string(cell.get("policy_readable")))

    index_path = batch_dir / "_index.npz"
    np.savez_compressed(
        index_path,
        source_batch=np.asarray(batch_name),
        cell_id=np.arange(len(batch), dtype=np.int32),
        cycle_life=np.asarray(cycle_life, dtype=np.float64),
        policy=np.asarray(policies, dtype=str),
        policy_readable=np.asarray(policies_readable, dtype=str),
        cell_path=np.asarray(cell_paths, dtype=str),
    )
    return index_path, len(batch)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a canonical cell-level dataset from .mat files. "
            "This stage preserves raw metadata, summary arrays, and cycle series."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Input .mat files, directories, or glob patterns. Default: data/*.mat",
    )
    parser.add_argument(
        "--output-dir",
        default="data/canonical",
        help="Output directory for canonical cell dataset. Default: data/canonical",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    inputs = resolve_paths(args.inputs, default_glob="data/*.mat")
    if not inputs:
        parser.error("No input .mat files found.")

    output_dir = Path(args.output_dir)
    total_cells = 0
    for path in inputs:
        index_path, count = export_batch(path, output_dir)
        total_cells += count
        print(f"[saved] {index_path} cells={count}")

    print(f"[canonical] batches={len(inputs)} total_cells={total_cells}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
