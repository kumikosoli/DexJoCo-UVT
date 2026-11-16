#!/usr/bin/env python3
"""Fail-fast checks for the DexJoCo LeRobot v2.1 tactile dataset."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys


TACTILE_COLUMN = "observation.tactile.contact_features"
TACTILE_DIM = 216


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ValueError(f"missing required file: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {path}: {exc}") from exc


def _read_jsonl(path: Path) -> list[dict]:
    try:
        lines = [line for line in path.read_text().splitlines() if line.strip()]
    except FileNotFoundError as exc:
        raise ValueError(f"missing required file: {path}") from exc
    try:
        return [json.loads(line) for line in lines]
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSONL: {path}: {exc}") from exc


def _require_slice(modality: dict, group: str, *, expected_dim: int | None = None) -> dict:
    if group not in modality:
        raise ValueError(f"meta/modality.json is missing group {group!r}")
    item = modality[group]
    if not isinstance(item, dict) or "start" not in item or "end" not in item:
        raise ValueError(f"modality group {group!r} needs integer start/end")
    start, end = item["start"], item["end"]
    if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end <= start:
        raise ValueError(f"invalid slice for {group!r}: start={start!r}, end={end!r}")
    if expected_dim is not None and end - start != expected_dim:
        raise ValueError(
            f"{group!r} has dimension {end - start}; expected {expected_dim}"
        )
    return item


def _feature_shape(info: dict, name: str) -> tuple[int, ...]:
    features = info.get("features", {})
    if name not in features:
        raise ValueError(f"meta/info.json features is missing {name!r}")
    shape = features[name].get("shape")
    if not isinstance(shape, list) or not all(isinstance(v, int) for v in shape):
        raise ValueError(f"feature {name!r} has invalid shape: {shape!r}")
    return tuple(shape)


def _scan_tactile(dataset: Path, info: dict, max_files: int) -> tuple[int, float]:
    try:
        import numpy as np
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ValueError(
            "numpy and pyarrow are required for the nonzero tactile scan inside the RLDX image"
        ) from exc

    pattern = info.get("data_path")
    if not isinstance(pattern, str):
        raise ValueError("meta/info.json is missing data_path")
    glob_pattern = pattern.replace("{episode_chunk:03d}", "*").replace(
        "{episode_index:06d}", "*"
    )
    glob_pattern = glob_pattern.replace("{episode_chunk}", "*").replace(
        "{episode_index}", "*"
    )
    files = sorted(dataset.glob(glob_pattern))
    if not files:
        files = sorted((dataset / "data").rglob("*.parquet"))
    if not files:
        raise ValueError("no parquet episodes found under dataset/data")

    rows = 0
    max_abs = 0.0
    found_nonzero = False
    for path in files[:max_files]:
        table = pq.read_table(path, columns=[TACTILE_COLUMN])
        for chunk in table[TACTILE_COLUMN].chunks:
            for value in chunk.to_pylist():
                arr = np.asarray(value, dtype=np.float32).reshape(-1)
                if arr.size != TACTILE_DIM:
                    raise ValueError(
                        f"{path}: tactile row dimension {arr.size}; expected {TACTILE_DIM}"
                    )
                if not np.all(np.isfinite(arr)):
                    raise ValueError(f"{path}: tactile row contains NaN or Inf")
                rows += 1
                local_max = float(np.max(np.abs(arr))) if arr.size else 0.0
                max_abs = max(max_abs, local_max)
                found_nonzero = found_nonzero or local_max > 0.0
        if found_nonzero:
            break
    if rows == 0:
        raise ValueError("tactile column contains no rows")
    if not found_nonzero or not math.isfinite(max_abs):
        raise ValueError(
            "scanned tactile data are all zero; this cannot test tactile dependence"
        )
    return rows, max_abs


def validate(dataset: Path, *, scan_values: bool, max_files: int) -> dict:
    meta = dataset / "meta"
    info = _read_json(meta / "info.json")
    modality = _read_json(meta / "modality.json")
    stats = _read_json(meta / "stats.json")
    episodes = _read_jsonl(meta / "episodes.jsonl")
    tasks = _read_jsonl(meta / "tasks.jsonl")

    if not episodes:
        raise ValueError("dataset contains no episodes")
    if not tasks:
        raise ValueError("dataset contains no tasks")
    too_short = [ep.get("episode_index") for ep in episodes if int(ep.get("length", 0)) < 17]
    if too_short:
        raise ValueError(f"episodes shorter than the 16-step horizon: {too_short[:8]}")

    for camera in ("front", "wrist"):
        _feature_shape(info, f"observation.images.{camera}")
        if camera not in modality.get("video", {}):
            raise ValueError(f"meta/modality.json video is missing {camera!r}")
    state_shape = _feature_shape(info, "observation.state")
    action_shape = _feature_shape(info, "action")
    if state_shape != (23,):
        raise ValueError(f"observation.state has shape {state_shape}; expected (23,)")
    if action_shape != (22,):
        raise ValueError(f"action has shape {action_shape}; expected (22,)")
    tactile_shape = _feature_shape(info, TACTILE_COLUMN)
    if tactile_shape != (TACTILE_DIM,):
        raise ValueError(
            f"{TACTILE_COLUMN} has shape {tactile_shape}; expected ({TACTILE_DIM},)"
        )

    state_meta = _require_slice(
        modality.get("state", {}), "right_hand_state", expected_dim=23
    )
    action_meta = _require_slice(
        modality.get("action", {}), "right_hand_action_rotvec", expected_dim=22
    )
    if state_meta.get("original_key", "observation.state") != "observation.state":
        raise ValueError("state.right_hand_state must point to 'observation.state'")
    if action_meta.get("original_key", "action") != "action":
        raise ValueError("action.right_hand_action_rotvec must point to 'action'")
    tactile_meta = _require_slice(
        modality.get("tactile", {}), "contact_features", expected_dim=TACTILE_DIM
    )
    if tactile_meta.get("original_key", TACTILE_COLUMN) != TACTILE_COLUMN:
        raise ValueError(
            "tactile.contact_features original_key must be " + repr(TACTILE_COLUMN)
        )
    if TACTILE_COLUMN not in stats:
        raise ValueError(f"meta/stats.json is missing {TACTILE_COLUMN!r}")

    rows, max_abs = (0, 0.0)
    if scan_values:
        rows, max_abs = _scan_tactile(dataset, info, max_files)
    return {
        "dataset": str(dataset.resolve()),
        "episodes": len(episodes),
        "tasks": len(tasks),
        "tactile_dim": TACTILE_DIM,
        "scanned_rows": rows,
        "max_abs_tactile": max_abs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--max-files", type=int, default=32)
    args = parser.parse_args()
    try:
        result = validate(
            args.dataset,
            scan_values=not args.metadata_only,
            max_files=args.max_files,
        )
    except ValueError as exc:
        print(f"DATASET PREFLIGHT FAILED: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
