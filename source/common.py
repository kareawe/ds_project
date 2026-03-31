from __future__ import annotations

import glob
import importlib
import logging
from pathlib import Path
from typing import Any

import numpy as np


def import_optional(module_name: str) -> Any | None:
    try:
        return importlib.import_module(module_name)
    except ModuleNotFoundError:
        return None


def load_mat(path: Path) -> dict[str, Any]:
    errors: list[str] = []

    mat73 = import_optional("mat73")
    if mat73 is not None:
        try:
            previous_disable = logging.root.manager.disable
            logging.disable(logging.CRITICAL)
            try:
                return mat73.loadmat(str(path))
            finally:
                logging.disable(previous_disable)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"mat73.loadmat failed: {exc}")

    scipy_io = import_optional("scipy.io")
    if scipy_io is not None:
        try:
            return scipy_io.loadmat(str(path), simplify_cells=True)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"scipy.io.loadmat failed: {exc}")

    if mat73 is None and scipy_io is None:
        raise RuntimeError(
            "Cannot read .mat files because neither 'mat73' nor 'scipy' is installed."
        )

    raise RuntimeError(f"Failed to load {path.name}: {' | '.join(errors)}")


def maybe_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def normalize_leaf(value: Any) -> Any:
    value = maybe_scalar(value)
    if isinstance(value, dict):
        return {key: normalize_leaf(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_leaf(item) for item in value]
    if isinstance(value, tuple):
        return tuple(normalize_leaf(item) for item in value)
    return value


def dict_of_lists_to_list_of_dicts(data: dict[str, Any]) -> list[dict[str, Any]]:
    if not data:
        return []
    keys = list(data.keys())
    size = len(data[keys[0]])
    rows = []
    for index in range(size):
        rows.append({key: normalize_leaf(data[key][index]) for key in keys})
    return rows


def normalize_batch(raw_batch: Any) -> list[dict[str, Any]]:
    if isinstance(raw_batch, dict):
        batch = dict_of_lists_to_list_of_dicts(raw_batch)
    elif isinstance(raw_batch, np.ndarray):
        batch = [normalize_leaf(item) for item in raw_batch.tolist()]
    else:
        batch = [normalize_leaf(item) for item in raw_batch]

    normalized = []
    for cell in batch:
        if not isinstance(cell, dict):
            raise TypeError(f"Unexpected cell type: {type(cell)!r}")

        copied = {key: normalize_leaf(value) for key, value in cell.items()}
        if isinstance(copied.get("cycles"), dict):
            copied["cycles"] = dict_of_lists_to_list_of_dicts(copied["cycles"])
        if isinstance(copied.get("summary"), dict):
            copied["summary"] = {
                key: np.asarray(value).reshape(-1) if value is not None else np.array([])
                for key, value in copied["summary"].items()
            }
        normalized.append(copied)
    return normalized


def normalize_float(value: Any) -> float:
    value = maybe_scalar(value)
    if value is None:
        return np.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def normalize_string(value: Any) -> str:
    value = maybe_scalar(value)
    if value is None:
        return ""
    return str(value)


def resolve_paths(values: list[str], default_glob: str = "data/*.mat") -> list[Path]:
    patterns = values or [default_glob]
    paths: list[Path] = []
    for value in patterns:
        matched = [Path(item) for item in glob.glob(value)]
        if matched:
            paths.extend(matched)
            continue
        path = Path(value)
        if path.is_dir():
            paths.extend(sorted(path.glob("*.mat")))
            paths.extend(sorted(path.glob("*.npy")))
        else:
            paths.append(path)

    unique: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return sorted(unique)
