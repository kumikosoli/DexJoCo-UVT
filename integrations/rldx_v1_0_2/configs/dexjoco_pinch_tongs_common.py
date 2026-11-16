"""Shared RLDX modality layout for the single-arm DexJoCo pinch_tongs dataset."""

from rldx.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)


TACTILE_DIM = 4 * 16 * 3 + 4 * 6  # v3 surface features plus per-finger summaries


def build_modality_config(*, include_tactile: bool) -> dict[str, ModalityConfig]:
    """Return identical core modalities, optionally loading the tactile column.

    The no-tactile arm keeps the same tactile temporal configuration but uses
    an empty ``modality_keys`` list.  This lets RLDX infer the exact same
    physics token layout without reading the tactile column.  Together with
    ``--allow-missing-physics``, RLDX emits zeros with ``physics_mask=0`` while
    retaining the same MSAT architecture and parameter count.
    """

    config = {
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
            action_configs=[
                ActionConfig(
                    rep=ActionRepresentation.ABSOLUTE,
                    type=ActionType.NON_EEF,
                    format=ActionFormat.DEFAULT,
                )
            ],
        ),
        "language": ModalityConfig(
            delta_indices=[0],
            modality_keys=["task"],
        ),
    }
    config["tactile"] = ModalityConfig(
        # Current tactile input plus t+1...t+16 prediction targets.
        # --allow-padding is required at episode boundaries.  Keep these
        # deltas in the mask0 arm so PhysicsHead has the same 1+16 tokens.
        delta_indices=list(range(17)),
        modality_keys=["contact_features"] if include_tactile else [],
    )
    return config
