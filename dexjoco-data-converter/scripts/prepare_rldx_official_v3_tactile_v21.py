#!/usr/bin/env python3
"""Build RLDX v2.1 data by joining official DexJoCo v3 with tactile replay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dexjoco_data_converter.to_lerobot.rldx_official_tactile_join import (
    prepare_official_v3_tactile_rldx_v21,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Join official DexJoCo LeRobot v3 RGB/state/action with current "
            "MuJoCo tactile replay and write an atomic RLDX LeRobot v2.1 dataset."
        )
    )
    parser.add_argument(
        "--official-v3",
        type=Path,
        required=True,
        help="Read-only official DexJoCo LeRobot v3 task directory",
    )
    parser.add_argument(
        "--raw",
        type=Path,
        required=True,
        help="Read-only root containing all raw replay.zarr episodes",
    )
    parser.add_argument(
        "--tactile-replay",
        type=Path,
        required=True,
        help="Read-only data-only replay root containing current tactile_gt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New RLDX LeRobot v2.1 directory; must not already exist",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Build only the first N episodes for a 1/3-episode smoke test",
    )
    parser.add_argument("--robot-type", default="dexjoco_allegro_right")
    parser.add_argument(
        "--video-encoding",
        choices=("h264", "copy"),
        default="h264",
        help=(
            "h264 (default) is compatible with the RLDX OpenCV loader; copy "
            "preserves source packets but independently split AV1 may not decode"
        ),
    )
    parser.add_argument(
        "--replay-audit-json",
        type=Path,
        default=None,
        help="Optional strict/diagnostic replay outcome audit embedded in the receipt",
    )
    parser.add_argument(
        "--video-workers",
        type=int,
        default=4,
        help="Number of bounded parallel official-video conversions (default: 4)",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    replay_audit = None
    if args.replay_audit_json is not None:
        replay_audit = json.loads(
            args.replay_audit_json.read_text(encoding="utf-8")
        )
        if not isinstance(replay_audit, dict):
            raise SystemExit("--replay-audit-json must contain a JSON object")
    summary = prepare_official_v3_tactile_rldx_v21(
        args.official_v3,
        args.raw,
        args.tactile_replay,
        args.output,
        max_episodes=args.max_episodes,
        robot_type=args.robot_type,
        video_encoding=args.video_encoding,
        video_workers=args.video_workers,
        replay_audit=replay_audit,
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "output": str(summary.output_dir),
                "episodes": summary.total_episodes,
                "frames": summary.total_frames,
                "fps": summary.fps,
                "start_indices": list(summary.start_indices),
                "video_encoding": summary.video_encoding,
                "video_reencoded": summary.video_encoding != "copy",
                "mujoco_video_rerendered": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
