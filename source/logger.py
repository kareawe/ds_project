from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def make_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{uuid4().hex[:8]}"


def to_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, (dict, list)):
            flat[key] = to_json(value)
        else:
            flat[key] = value
    return flat


def append_registry_csv(csv_path: Path, record: dict[str, Any]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    flat = flatten_record(record)

    fieldnames = []
    if csv_path.exists():
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = [name for name in (reader.fieldnames or []) if name]

    for key in flat:
        if key not in fieldnames:
            fieldnames.append(key)

    existing_rows: list[dict[str, Any]] = []
    if csv_path.exists():
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                cleaned = {key: value for key, value in row.items() if key}
                existing_rows.append({key: cleaned.get(key, "") for key in fieldnames})

    existing_rows.append({key: flat.get(key, "") for key in fieldnames})

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(existing_rows)


def append_registry_jsonl(jsonl_path: Path, record: dict[str, Any]) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a") as f:
        f.write(to_json(record) + "\n")


def log_experiment(registry_dir: Path, record: dict[str, Any]) -> None:
    registry_dir.mkdir(parents=True, exist_ok=True)
    append_registry_csv(registry_dir / "experiment_registry.csv", record)
    append_registry_jsonl(registry_dir / "experiment_registry.jsonl", record)
