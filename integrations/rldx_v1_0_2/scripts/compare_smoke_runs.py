#!/usr/bin/env python3
"""Compare the two saved smoke configs and fail on an unfair ablation."""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import sys

import yaml


SUCCESS_MARKER_NAME = "dexjoco_run_success.json"
MODEL_INDEX_NAMES = (
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)


def _load(run: Path) -> dict:
    config = run / "experiment_cfg/conf.yaml"
    if not config.is_file():
        raise ValueError(f"missing saved config: {config}")
    value = yaml.safe_load(config.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"saved config is not a mapping: {config}")
    return value


def _load_json(path: Path, label: str) -> dict:
    if not path.is_file():
        raise ValueError(f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a mapping: {path}")
    return value


def _require_success_marker(run: Path, expected_job: str) -> dict:
    marker = _load_json(run / SUCCESS_MARKER_NAME, "post-exit success marker")
    required = {
        "status": "success",
        "exit_code": 0,
        "job": expected_job,
        "mode": "smoke",
        "max_steps": 2,
        "logging_steps": 1,
        "logging_nan_inf_filter": False,
    }
    for key, expected in required.items():
        if marker.get(key) != expected:
            raise ValueError(
                f"{expected_job} success marker {key}={marker.get(key)!r}; "
                f"expected {expected!r}"
            )
    if not marker.get("completed_at"):
        raise ValueError(f"{expected_job} success marker has no completion timestamp")
    return marker


def _trainer_state_candidates(run: Path) -> list[Path]:
    candidates = []
    root_state = run / "trainer_state.json"
    if root_state.is_file():
        candidates.append(root_state)
    candidates.extend(sorted(run.glob("checkpoint-*/trainer_state.json")))
    return candidates


def _require_two_step_finite_losses(run: Path, name: str) -> tuple[Path, dict]:
    candidates = _trainer_state_candidates(run)
    if not candidates:
        raise ValueError(
            f"{name} has no trainer_state.json in the run root or checkpoint-* directory"
        )

    parsed = [(path, _load_json(path, "trainer state")) for path in candidates]
    state_path, state = max(
        parsed,
        key=lambda item: int(item[1].get("global_step", -1)),
    )
    if state.get("global_step") != 2:
        raise ValueError(
            f"{name} trainer global_step={state.get('global_step')!r}; expected 2 "
            f"in {state_path}"
        )

    log_history = state.get("log_history")
    if not isinstance(log_history, list):
        raise ValueError(f"{name} trainer log_history is missing or not a list")
    loss_steps: set[int] = set()
    for record in log_history:
        if not isinstance(record, dict) or "loss" not in record:
            continue
        loss = record["loss"]
        if isinstance(loss, bool) or not isinstance(loss, (int, float)):
            raise ValueError(f"{name} logged a non-numeric loss: {loss!r}")
        if not math.isfinite(float(loss)):
            raise ValueError(f"{name} logged a non-finite loss: {loss!r}")
        step = record.get("step")
        if isinstance(step, int):
            loss_steps.add(step)
        elif isinstance(step, float) and step.is_integer():
            loss_steps.add(int(step))
    if not {1, 2}.issubset(loss_steps):
        raise ValueError(
            f"{name} trainer state lacks finite per-step losses for steps 1 and 2; "
            f"found loss steps {sorted(loss_steps)} in {state_path}"
        )
    return state_path, state


def _require_complete_root_model(run: Path, name: str) -> tuple[Path, tuple[Path, ...]]:
    """Require the final gathered model index and every shard it references."""

    indexes = [run / filename for filename in MODEL_INDEX_NAMES if (run / filename).is_file()]
    if len(indexes) != 1:
        found = [path.name for path in indexes]
        raise ValueError(
            f"{name} must have exactly one final model index at the run root; "
            f"expected one of {list(MODEL_INDEX_NAMES)}, found {found}"
        )
    index_path = indexes[0]
    index = _load_json(index_path, "final model index")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise ValueError(f"{name} final model index has an empty or invalid weight_map")

    shard_names = set()
    for parameter, shard_name in weight_map.items():
        if not isinstance(parameter, str) or not isinstance(shard_name, str):
            raise ValueError(f"{name} final model weight_map contains a non-string entry")
        shard_rel = Path(shard_name)
        if shard_rel.is_absolute() or ".." in shard_rel.parts:
            raise ValueError(
                f"{name} final model index references an unsafe shard path: {shard_name!r}"
            )
        shard_names.add(shard_name)
    shards = tuple(sorted((run / shard_name for shard_name in shard_names), key=str))
    for shard in shards:
        if not shard.is_file() or shard.stat().st_size <= 0:
            raise ValueError(
                f"{name} final model shard is missing or empty: {shard} "
                f"(referenced by {index_path})"
            )
    return index_path, shards


def _strip_expected_differences(config: dict) -> dict:
    value = copy.deepcopy(config)
    training = value.get("training", {})
    training.pop("experiment_name", None)
    data = value.get("data", {})
    modality_configs = data.get("modality_configs", {})
    for embodiment_config in modality_configs.values():
        if isinstance(embodiment_config, dict):
            tactile_config = embodiment_config.get("tactile")
            if isinstance(tactile_config, dict):
                tactile_config.pop("modality_keys", None)
    return value


def compare(tactile_run: Path, mask0_run: Path) -> None:
    _require_success_marker(tactile_run, "tactile")
    _require_success_marker(mask0_run, "mask0")
    _require_two_step_finite_losses(tactile_run, "tactile")
    _require_two_step_finite_losses(mask0_run, "mask0")
    _require_complete_root_model(tactile_run, "tactile")
    _require_complete_root_model(mask0_run, "mask0")
    tactile = _load(tactile_run)
    mask0 = _load(mask0_run)
    if tactile.get("model") != mask0.get("model"):
        raise ValueError("model configs differ; the two arms are not architecture-matched")
    for name, cfg in (("tactile", tactile), ("mask0", mask0)):
        model = cfg.get("model", {})
        data = cfg.get("data", {})
        training = cfg.get("training", {})
        required = {
            "use_physics": True,
            "physics_keys": ["tactile"],
            "physics_dims": [216],
            "action_model_use_lora": True,
            "action_model_lora_rank": 16,
            "video_length": 1,
            "freeze_cog_tokens": True,
        }
        for key, expected in required.items():
            if model.get(key) != expected:
                raise ValueError(
                    f"{name} model.{key}={model.get(key)!r}; expected {expected!r}"
                )
        if data.get("seed") != 42:
            raise ValueError(f"{name} data.seed is not 42")
        if training.get("global_batch_size") != 2 or training.get("num_gpus") != 2:
            raise ValueError(f"{name} no longer has microbatch=1 per GPU")
        if training.get("gradient_checkpointing") is not True:
            raise ValueError(f"{name} gradient checkpointing is disabled")
        if training.get("max_steps") != 2:
            raise ValueError(f"{name} training.max_steps is not 2")
        if training.get("logging_steps") != 1:
            raise ValueError(f"{name} training.logging_steps is not 1")
        if training.get("logging_nan_inf_filter") is not False:
            raise ValueError(
                f"{name} logging_nan_inf_filter must be false for auditable loss"
            )

    tactile_modalities = tactile.get("data", {}).get("modality_configs", {})
    mask0_modalities = mask0.get("data", {}).get("modality_configs", {})
    def tactile_keys(modality_configs: dict) -> list:
        values = []
        for embodiment in modality_configs.values():
            if not isinstance(embodiment, dict) or "tactile" not in embodiment:
                continue
            tactile_cfg = embodiment["tactile"]
            if isinstance(tactile_cfg, dict):
                values.extend(tactile_cfg.get("modality_keys", []))
        return values

    tactile_keys_loaded = tactile_keys(tactile_modalities)
    mask0_keys_loaded = tactile_keys(mask0_modalities)
    if tactile_keys_loaded != ["contact_features"] or mask0_keys_loaded != []:
        raise ValueError(
            "expected identical tactile temporal config with data keys only in the tactile run; "
            f"got tactile={tactile_keys_loaded}, mask0={mask0_keys_loaded}"
        )
    if _strip_expected_differences(tactile) != _strip_expected_differences(mask0):
        raise ValueError("saved configs differ beyond experiment name and tactile modality keys")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("tactile_run", type=Path)
    parser.add_argument("mask0_run", type=Path)
    args = parser.parse_args()
    try:
        compare(args.tactile_run, args.mask0_run)
    except ValueError as exc:
        print(f"SMOKE COMPARISON FAILED: {exc}", file=sys.stderr)
        return 1
    print(
        "Smoke runs completed at step 2 with finite step-1/step-2 losses; "
        "final model indexes/shards are complete; configs match and only tactile "
        "loading is ablated."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
