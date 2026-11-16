"""Smoke-test a converted dataset through the real RLDX v1.0.2 loader.

Run this script from an installed RLDX-1 v1.0.2 environment.  It exercises
``LeRobotEpisodeLoader``, ``StandardSingleStepDataset`` and a two-item PyTorch
``DataLoader`` batch without loading model weights.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rldx
import torch
from rldx.data.dataset.standard_single_step_dataset import StandardSingleStepDataset
from rldx.data.embodiment_tags import EmbodimentTag
from rldx.data.types import ModalityConfig
from torch.utils.data import DataLoader


class _ShapeSmokeProcessor:
    """Turn one RLDX VLAStepData object into collatable tensors."""

    def __call__(self, messages):
        content = messages[0]["content"]
        return {
            "state": torch.from_numpy(content.states["right_hand_state"]),
            "action": torch.from_numpy(content.actions["right_hand_action_rotvec"]),
            "tactile": torch.from_numpy(content.physics["tactile.contact_features"]),
            "front": torch.from_numpy(
                np.ascontiguousarray(np.stack(content.images["front"]))
            ),
            "wrist": torch.from_numpy(
                np.ascontiguousarray(np.stack(content.images["wrist"]))
            ),
        }


def _modality_configs() -> dict[str, ModalityConfig]:
    return {
        "video": ModalityConfig(
            delta_indices=[0],
            modality_keys=["front", "wrist"],
        ),
        "state": ModalityConfig(
            delta_indices=[0],
            modality_keys=["right_hand_state"],
        ),
        "action": ModalityConfig(
            delta_indices=list(range(16)),
            modality_keys=["right_hand_action_rotvec"],
        ),
        "language": ModalityConfig(delta_indices=[0], modality_keys=["task"]),
        "tactile": ModalityConfig(
            delta_indices=[0],
            modality_keys=["contact_features"],
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--video-backend", default="opencv")
    args = parser.parse_args()

    dataset = StandardSingleStepDataset(
        dataset_path=args.dataset,
        embodiment_tag=EmbodimentTag.GENERAL_EMBODIMENT,
        modality_configs=_modality_configs(),
        video_backend=args.video_backend,
        allow_padding=False,
    )
    dataset.set_processor(_ShapeSmokeProcessor())

    single = dataset[0]
    expected_single_shapes = {
        "state": (1, 23),
        "action": (16, 22),
        "tactile": (1, 216),
    }
    for key, expected_shape in expected_single_shapes.items():
        if tuple(single[key].shape) != expected_shape:
            raise AssertionError(
                f"single {key}: {tuple(single[key].shape)} != {expected_shape}"
            )
    if single["front"].shape[0] != 1 or single["wrist"].shape[0] != 1:
        raise AssertionError("single video history is not one frame")

    batch = next(iter(DataLoader(dataset, batch_size=2, num_workers=0)))
    expected_batch_shapes = {
        "state": (2, 1, 23),
        "action": (2, 16, 22),
        "tactile": (2, 1, 216),
    }
    for key, expected_shape in expected_batch_shapes.items():
        if tuple(batch[key].shape) != expected_shape:
            raise AssertionError(
                f"batch {key}: {tuple(batch[key].shape)} != {expected_shape}"
            )
    if batch["front"].shape[:2] != (2, 1) or batch["wrist"].shape[:2] != (2, 1):
        raise AssertionError("batched videos are not shaped (batch=2, history=1, ...)")

    statistics = dataset.get_dataset_statistics()
    tactile_stats = statistics["tactile"]["contact_features"]
    for name in ("min", "max", "mean", "std", "q01", "q99"):
        if len(tactile_stats[name]) != 216:
            raise AssertionError(f"tactile stats {name} does not have 216 values")

    print(f"RLDX version: {getattr(rldx, '__version__', 'unknown')}")
    print(f"single: { {key: tuple(value.shape) for key, value in single.items()} }")
    print(f"batch: { {key: tuple(value.shape) for key, value in batch.items()} }")
    print(f"steps: {len(dataset)}")
    print("RLDX v1.0.2 episode + batch loader smoke test: PASS")


if __name__ == "__main__":
    main()
