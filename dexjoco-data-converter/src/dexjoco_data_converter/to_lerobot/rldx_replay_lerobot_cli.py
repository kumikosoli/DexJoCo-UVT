"""Installed console entry point for the RLDX replay adapter."""

from __future__ import annotations

import argparse
from pathlib import Path

from .rldx_replay_lerobot import (
    prepare_rldx_lerobot_dataset,
    validate_rldx_lerobot_dataset,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare DexJoCo replay episodes as RLDX LeRobot v2.1 data"
    )
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task")
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--robot-type", default="dexjoco_allegro_right")
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.validate_only:
        summary = validate_rldx_lerobot_dataset(args.output)
    else:
        if args.input is None or not args.task:
            raise SystemExit("--input and --task are required for conversion")
        summary = prepare_rldx_lerobot_dataset(
            args.input,
            args.output,
            task=args.task,
            start_idx=args.start_idx,
            fps=args.fps,
            robot_type=args.robot_type,
        )
    print(
        f"RLDX dataset OK: {summary.output_dir} | "
        f"episodes={summary.total_episodes} frames={summary.total_frames} "
        f"fps={summary.fps}"
    )
