#!/usr/bin/env python3
"""Exercise the patched RLDX physics-value/missing-sensor contract end to end."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from types import MethodType, SimpleNamespace

import numpy as np


TIME_STEPS = 17
TACTILE_DIM = 216


def _import_rldx(rldx_root: Path):
    sys.path.insert(0, str(rldx_root))
    from rldx.data.embodiment_tags import EmbodimentTag
    from rldx.data.state_action.state_action_processor import StateActionProcessor
    from rldx.model.core.processing_rldx import RLDXProcessor

    return EmbodimentTag, StateActionProcessor, RLDXProcessor


def _make_state_action_processor(state_action_cls):
    processor = state_action_cls.__new__(state_action_cls)
    processor.clip_outliers = True
    processor.norm_params = {
        "general_embodiment": {
            "tactile": {
                "contact_features": {
                    "q01": -np.ones(TACTILE_DIM, dtype=np.float32),
                    "q99": np.ones(TACTILE_DIM, dtype=np.float32),
                }
            }
        }
    }
    return processor


def _make_upper_processor(rldx_processor_cls, state_action_processor):
    processor = rldx_processor_cls.__new__(rldx_processor_cls)
    processor.physics_keys = ["tactile"]
    processor.physics_dims = [TACTILE_DIM]
    processor.allow_missing_physics = True
    processor._physics_t_len = TIME_STEPS
    processor.state_action_processor = state_action_processor
    processor.modality_configs = {
        "general_embodiment": {
            "state": SimpleNamespace(modality_keys=["state"]),
            "action": SimpleNamespace(modality_keys=["action"]),
            "video": SimpleNamespace(modality_keys=[]),
        }
    }
    processor.max_state_dim = 1
    processor.max_action_dim = 1
    processor.max_action_horizon = 1
    processor.formalize_language = False
    processor.memory_length = 1
    processor.training = False
    processor.train_image_transform = None
    processor.eval_image_transform = None
    processor.general_embodiment_train_ratio = 0.0
    processor.embodiment_id_mapping = {"general_embodiment": 0}
    processor._get_vlm_inputs = MethodType(lambda _self, **_kwargs: {}, processor)
    state_action_processor.apply = lambda **_kwargs: (
        {"state": np.zeros((1, 1), dtype=np.float32)},
        {"action": np.zeros((1, 1), dtype=np.float32)},
    )
    return processor


def _content(embodiment, physics: dict[str, np.ndarray]):
    return SimpleNamespace(
        embodiment=embodiment,
        actions={"action": np.zeros((1, 1), dtype=np.float32)},
        states={"state": np.zeros((1, 1), dtype=np.float32)},
        images={},
        text="pinch",
        physics=physics,
    )


def verify(rldx_root: Path) -> None:
    embodiment_cls, state_action_cls, rldx_processor_cls = _import_rldx(rldx_root)
    state_action = _make_state_action_processor(state_action_cls)
    zeros = np.zeros((TIME_STEPS, TACTILE_DIM), dtype=np.float32)
    nonzero = zeros.copy()
    nonzero[8, 17] = 0.25

    normalized_zero = state_action.apply_physics(
        {"tactile.contact_features": zeros}, "general_embodiment"
    )
    if "tactile.contact_features" not in normalized_zero:
        raise AssertionError("present all-zero tactile key was incorrectly dropped")
    np.testing.assert_array_equal(normalized_zero["tactile.contact_features"], zeros)

    normalized_nonzero = state_action.apply_physics(
        {"tactile.contact_features": nonzero}, "general_embodiment"
    )
    if "tactile.contact_features" not in normalized_nonzero:
        raise AssertionError("present non-zero tactile key was incorrectly dropped")

    upper = _make_upper_processor(rldx_processor_cls, state_action)
    embodiment = embodiment_cls.GENERAL_EMBODIMENT
    zero_output = upper([{"content": _content(embodiment, {"tactile.contact_features": zeros})}])
    np.testing.assert_array_equal(
        zero_output["physics_mask"], np.array([1.0], dtype=np.float32)
    )
    np.testing.assert_array_equal(zero_output["physics"].numpy(), zeros)

    missing_output = upper([{"content": _content(embodiment, {})}])
    np.testing.assert_array_equal(
        missing_output["physics_mask"], np.array([0.0], dtype=np.float32)
    )
    np.testing.assert_array_equal(missing_output["physics"].numpy(), zeros)

    nonzero_output = upper(
        [{"content": _content(embodiment, {"tactile.contact_features": nonzero})}]
    )
    np.testing.assert_array_equal(
        nonzero_output["physics_mask"], np.array([1.0], dtype=np.float32)
    )
    if float(nonzero_output["physics"][8, 17]) == 0.0:
        raise AssertionError("non-zero tactile value was not preserved through RLDXProcessor")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("rldx_root", type=Path)
    args = parser.parse_args()
    verify(args.rldx_root.resolve())
    print(
        "RLDX physics contract verified: present zero/non-zero tactile => mask=1; "
        "missing tactile => zero fill + mask=0."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
