"""CLI for the DexJoCo replay -> RLDX LeRobot v2.1 adapter."""

from __future__ import annotations

import argparse
from pathlib import Path

from dexjoco_data_converter.to_lerobot.rldx_replay_lerobot import (
    prepare_rldx_lerobot_dataset,
    validate_rldx_lerobot_dataset,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert replay_demos_zarr.py episode directories into a fail-closed, "
            "RLDX-1 v1.0.2 compatible LeRobot v2.1 dataset."
        )
    )
    parser.add_argument("--input", type=Path, help="Replay episode root")
    parser.add_argument("--output", type=Path, required=True, help="New dataset path")
    parser.add_argument("--task", help="Language instruction for every episode")
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Drop this many leading rows and frames from every modality",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Expected data FPS; default infers and validates the replay timestamps",
    )
    parser.add_argument(
        "--robot-type",
        default="dexjoco_allegro_right",
        help="LeRobot info.json robot_type",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate --output without reading --input or converting",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if args.validate_only:
        summary = validate_rldx_lerobot_dataset(args.output)
    else:
        if args.input is None:
            raise SystemExit("--input is required unless --validate-only is used")
        if not args.task:
            raise SystemExit("--task is required unless --validate-only is used")
        summary = prepare_rldx_lerobot_dataset(
            args.input,
            args.output,
            task=args.task,
            start_idx=args.start_idx,
            fps=args.fps,
            robot_type=args.robot_type,
        )
    print(
        "RLDX LeRobot v2.1 dataset OK: "
        f"{summary.output_dir} | episodes={summary.total_episodes} "
        f"frames={summary.total_frames} fps={summary.fps}"
    )


if __name__ == "__main__":
    main()
