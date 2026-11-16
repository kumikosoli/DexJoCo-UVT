from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import unittest
from pathlib import Path

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr

from dexjoco_data_converter.to_lerobot.rldx_official_tactile_join import (
    JOIN_RECEIPT_NAME,
    prepare_official_v3_tactile_rldx_v21,
)
from dexjoco_data_converter.to_lerobot.rldx_replay_lerobot import (
    ACTION_COLUMN,
    STATE_COLUMN,
    TACTILE_COLUMN,
    DatasetValidationError,
    validate_rldx_lerobot_dataset,
)

HAS_FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
TASK = "Grasp the tongs and perform three consecutive open-close motions."


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(path for path in root.rglob("*") if path.is_file()):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _fixed_list(array: np.ndarray) -> pa.FixedSizeListArray:
    value = np.ascontiguousarray(array, dtype=np.float32)
    return pa.FixedSizeListArray.from_arrays(
        pa.array(value.reshape(-1), type=pa.float32()),
        list_size=value.shape[1],
    )


def _matrix(table: pa.Table, key: str, width: int) -> np.ndarray:
    column = table[key].combine_chunks()
    return column.values.to_numpy(zero_copy_only=False).reshape(-1, width)


def _write_packed_video(path: Path, frame_values: list[int], fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("h264", rate=fps)
        stream.width = 64
        stream.height = 48
        stream.pix_fmt = "yuv420p"
        stream.codec_context.gop_size = 1
        stream.codec_context.max_b_frames = 0
        stream.options = {"crf": "12"}
        for value in frame_values:
            image = np.full((48, 64, 3), value, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _write_raw_episode(
    path: Path,
    official_action: np.ndarray,
    official_state: np.ndarray,
    start_idx: int,
) -> tuple[np.ndarray, np.ndarray]:
    path.parent.mkdir(parents=True, exist_ok=True)
    action = np.concatenate(
        [np.repeat(official_action[:1], start_idx, axis=0), official_action], axis=0
    ).astype(np.float64)
    selected_state = np.concatenate(
        [np.repeat(official_state[:1], start_idx, axis=0), official_state], axis=0
    ).astype(np.float32)
    full_state = np.zeros((action.shape[0], 1, 31), dtype=np.float32)
    full_state[:, 0, :23] = selected_state

    root = zarr.open(str(path), mode="w")
    data = root.create_group("data")
    quaternion_action = np.concatenate(
        [
            action[:, :3],
            np.repeat(
                np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float64),
                action.shape[0],
                axis=0,
            ),
            action[:, 6:22],
        ],
        axis=1,
    )
    data.create_dataset("action", data=quaternion_action)
    data.create_dataset("action_rotvec", data=action)
    data.create_dataset("state", data=full_state)
    data.create_dataset("timestamp", data=np.arange(len(action)) / 30.0)
    return action, full_state


def _tactile_v3_metadata() -> dict:
    return {
        "schema_version": "dexjoco.tactile_gt.v3",
        "frame": "per-sample fingertip body-local frame at the body origin",
        "hand_order": ["right"],
        "fingertip_order": ["index", "middle", "ring", "thumb"],
        "taxel_channels": ["tangent_u", "tangent_v", "normal_inward"],
        "taxels_per_fingertip": 16,
        "taxel_grid_shape": [4, 4],
        "taxel_frame": "per-taxel local frame",
        "taxel_projection": {
            "method": "normal/shear separated facing Gaussian projection",
            "normalization": "per-contact taxel weights sum to one",
            "support": "only taxels facing the contact inward normal",
            "layout": "2 cap rings + 2 cylinder rings x 4 columns",
        },
        "field_units": {"taxel_force_local": ["N", "N", "N"]},
        "alignment": {
            "tactile_gt.*": "pre-action observation at row t",
            "next_tactile_gt.*": "post-action response to action at row t",
        },
        "field_specs": {
            "taxel_force_local": {
                "axes": ["time", "hand", "fingertip", "taxel", "xyz"],
                "dtype": "float32",
            }
        },
    }


def _write_tactile_replay(
    path: Path,
    action: np.ndarray,
    raw_full_state: np.ndarray,
    episode_index: int,
    *,
    provenance: str = "legacy",
) -> np.ndarray:
    path.parent.mkdir(parents=True, exist_ok=True)
    length = action.shape[0]
    tactile = (
        np.arange(length * 1 * 4 * 16 * 3, dtype=np.float32).reshape(
            length, 1, 4, 16, 3
        )
        + np.float32(episode_index * 100000)
    )
    root = zarr.open(str(path), mode="w")
    data = root.create_group("data")
    data.create_dataset("action_rotvec", data=action)
    data.create_dataset(
        "state", data=np.asarray(raw_full_state[:, 0, :], dtype=np.float64)
    )
    data.create_dataset("tactile_gt.taxel_force_local", data=tactile)
    data.create_dataset("tactile_gt.fingertip_wrench_local", data=np.zeros((length, 1, 4, 6), np.float32))
    for name in ("fingertip_normal", "fingertip_force_peak", "contact_fraction"):
        data.create_dataset("tactile_gt." + name, data=np.zeros((length, 1, 4), np.float32))
    data.create_dataset(
        "next_tactile_gt.taxel_force_local",
        data=tactile + np.float32(5000000),
    )
    root.attrs["tactile_gt"] = _tactile_v3_metadata()
    replay_output = {
        "mode": "data_only",
        "data_only": True,
        "rgb_observations_generated": False,
        "rgb_videos_written": False,
        "depth_outputs_written": False,
    }
    if provenance == "v1":
        replay_output.update(
            {
                "schema_version": "dexjoco.replay_output.v1",
                "task": "pinch_tongs",
                "randomize": False,
                "randomize_dynamics": False,
                "restore_state_requested": True,
                "initial_state_actually_restored": True,
                "extend_steps": 0,
                "source_action_steps": int(length),
            }
        )
    elif provenance != "legacy":
        raise ValueError(f"Unsupported fixture provenance: {provenance}")
    root.attrs["replay_output"] = replay_output
    return tactile


def _upgrade_replay_to_v1(path: Path) -> None:
    root = zarr.open(str(path), mode="a")
    replay_output = dict(root.attrs["replay_output"])
    replay_output.update(
        {
            "schema_version": "dexjoco.replay_output.v1",
            "task": "pinch_tongs",
            "randomize": False,
            "randomize_dynamics": False,
            "restore_state_requested": True,
            "initial_state_actually_restored": True,
            "extend_steps": 0,
            "source_action_steps": int(root["data"]["action_rotvec"].shape[0]),
        }
    )
    root.attrs["replay_output"] = replay_output


def _make_fixture(root: Path) -> dict:
    official_root = root / "official_v3"
    raw_root = root / "raw"
    replay_root = root / "replay"
    fps = 10
    lengths = [5, 6, 7]
    start_indices = [2, 3, 1]

    action_chunks = []
    state_chunks = []
    episode_rows = []
    raw_actions = []
    tactile_by_episode = []
    global_index = 0
    video_frame_values = []
    replay_names = ["replay_z", "replay_a", "replay_m"]

    for episode_index, (length, start_idx) in enumerate(
        zip(lengths, start_indices, strict=True)
    ):
        row = np.arange(length, dtype=np.float32)[:, None]
        action = (
            np.float32(episode_index * 1000)
            + row * np.float32(10)
            + np.arange(22, dtype=np.float32)[None, :]
        )
        action[:, 3:6] = np.float32(0.0)
        state = (
            np.float32(episode_index * 2000)
            + row * np.float32(20)
            + np.arange(23, dtype=np.float32)[None, :]
        )
        action_chunks.append(action)
        state_chunks.append(state)

        raw_name = f"raw_episode_{episode_index:02d}"
        raw_action, full_state = _write_raw_episode(
            raw_root / raw_name / "replay.zarr",
            action,
            state,
            start_idx,
        )
        raw_actions.append(raw_action)
        tactile = _write_tactile_replay(
            replay_root / replay_names[episode_index] / "replay.zarr",
            raw_action,
            full_state,
            episode_index,
        )
        tactile_by_episode.append(tactile)

        start_time = global_index / fps
        stop_time = (global_index + length) / fps
        episode_rows.append(
            {
                "episode_index": episode_index,
                "tasks": [TASK],
                "length": length,
                "data/chunk_index": 0,
                "data/file_index": 0,
                "dataset_from_index": global_index,
                "dataset_to_index": global_index + length,
                "videos/observation.images.front/chunk_index": 0,
                "videos/observation.images.front/file_index": 0,
                "videos/observation.images.front/from_timestamp": start_time,
                "videos/observation.images.front/to_timestamp": stop_time,
                "videos/observation.images.wrist/chunk_index": 0,
                "videos/observation.images.wrist/file_index": 0,
                "videos/observation.images.wrist/from_timestamp": start_time,
                "videos/observation.images.wrist/to_timestamp": stop_time,
            }
        )
        video_frame_values.extend(
            [20 + episode_index * 50 + frame for frame in range(length)]
        )
        global_index += length

    actions = np.concatenate(action_chunks, axis=0)
    states = np.concatenate(state_chunks, axis=0)
    timestamps = np.concatenate(
        [np.arange(length, dtype=np.float32) / fps for length in lengths]
    )
    frame_indices = np.concatenate([np.arange(length) for length in lengths])
    episode_indices = np.concatenate(
        [np.full(length, index) for index, length in enumerate(lengths)]
    )
    table = pa.table(
        {
            ACTION_COLUMN: _fixed_list(actions),
            STATE_COLUMN: _fixed_list(states),
            "timestamp": pa.array(timestamps, type=pa.float32()),
            "frame_index": pa.array(frame_indices, type=pa.int64()),
            "episode_index": pa.array(episode_indices, type=pa.int64()),
            "index": pa.array(np.arange(global_index), type=pa.int64()),
            "task_index": pa.array(np.zeros(global_index), type=pa.int64()),
        }
    )
    data_path = official_root / "data/chunk-000/file-000.parquet"
    data_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, data_path)
    episode_path = official_root / "meta/episodes/chunk-000/file-000.parquet"
    episode_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(episode_rows), episode_path)
    pq.write_table(
        pa.table(
            {
                "task_index": pa.array([0], type=pa.int64()),
                "__index_level_0__": pa.array([TASK]),
            }
        ),
        official_root / "meta/tasks.parquet",
    )

    for key in ("observation.images.front", "observation.images.wrist"):
        _write_packed_video(
            official_root / f"videos/{key}/chunk-000/file-000.mp4",
            video_frame_values,
            fps,
        )
    info = {
        "codebase_version": "v3.0",
        "robot_type": None,
        "total_episodes": len(lengths),
        "total_frames": global_index,
        "total_tasks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{len(lengths)}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": (
            "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
        ),
        "features": {
            STATE_COLUMN: {"dtype": "float32", "shape": [23]},
            ACTION_COLUMN: {"dtype": "float32", "shape": [22]},
            "observation.images.front": {
                "dtype": "video",
                "shape": [48, 64, 3],
                "info": {"video.codec": "h264", "video.fps": fps},
            },
            "observation.images.wrist": {
                "dtype": "video",
                "shape": [48, 64, 3],
                "info": {"video.codec": "h264", "video.fps": fps},
            },
        },
    }
    (official_root / "meta/info.json").write_text(
        json.dumps(info), encoding="utf-8"
    )
    return {
        "official": official_root,
        "raw": raw_root,
        "replay": replay_root,
        "lengths": lengths,
        "start_indices": start_indices,
        "raw_actions": raw_actions,
        "tactile": tactile_by_episode,
        "replay_names": replay_names,
    }


class TrajectoryAlignmentRegressionTest(unittest.TestCase):
    def test_join_rejects_drift_after_identical_initial_state_and_actions(self):
        from unittest.mock import patch
        from dexjoco_data_converter.to_lerobot import rldx_official_tactile_join as module
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _make_fixture(Path(tmp).resolve())
            # This unit test isolates trajectory matching from ffprobe. Actual
            # video output is covered separately by the integration suite.
            with patch.object(module, "_validate_official_videos", return_value={}):
                _, plans, _ = module._build_join_plans(
                    fixture["official"], fixture["raw"], fixture["replay"], max_episodes=1)
                replay = zarr.open(str(plans[0].replay_episode_dir / "replay.zarr"), mode="a")
                states = replay["data/state"][:]
                states[2, 0] += .01
                replay["data/state"][:] = states
                with self.assertRaisesRegex(DatasetValidationError, "trajectory mismatch.*first row 2"):
                    module._build_join_plans(fixture["official"], fixture["raw"],
                                             fixture["replay"], max_episodes=1)

    def test_quaternion_sign_equivalence_and_separate_joint_tolerance(self):
        from dexjoco_data_converter.to_lerobot.rldx_official_tactile_join import _validate_replay_trajectory
        raw = np.zeros((3, 31), np.float32)
        raw[:, 3] = raw[:, 26] = 1.
        replay = raw.copy()
        replay[1:, 3:7] *= -1
        replay[1:, 26:30] *= -1
        replay[1, 7] = .0005
        report = _validate_replay_trajectory(raw, replay)
        self.assertEqual(report["checked_rows"], 3)
        replay[1, 7] = .01
        with self.assertRaisesRegex(DatasetValidationError, "finger_joint_rad"):
            _validate_replay_trajectory(raw, replay)


@unittest.skipUnless(HAS_FFMPEG, "ffmpeg and ffprobe are required")
class OfficialTactileJoinTest(unittest.TestCase):
    def test_three_episode_join_uses_content_match_and_current_tactile(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = _make_fixture(root)
            before = {
                key: _tree_digest(fixture[key])
                for key in ("official", "raw", "replay")
            }
            output = root / "output"
            summary = prepare_official_v3_tactile_rldx_v21(
                fixture["official"], fixture["raw"], fixture["replay"], output
            )

            self.assertEqual(summary.total_episodes, 3)
            self.assertEqual(summary.total_frames, sum(fixture["lengths"]))
            self.assertEqual(summary.fps, 10)
            self.assertEqual(summary.start_indices, tuple(fixture["start_indices"]))
            self.assertEqual(output.stat().st_mode & 0o777, 0o755)
            validate_rldx_lerobot_dataset(output)
            self.assertEqual(
                {
                    key: _tree_digest(fixture[key])
                    for key in ("official", "raw", "replay")
                },
                before,
            )

            receipt = json.loads(
                (output / "meta" / JOIN_RECEIPT_NAME).read_text(encoding="utf-8")
            )
            self.assertTrue(receipt["video_reencoded"])
            self.assertFalse(receipt["mujoco_video_rerendered"])
            self.assertEqual(receipt["video_encoding"], "h264")
            info = json.loads((output / "meta/info.json").read_text(encoding="utf-8"))
            self.assertEqual(
                info["features"]["observation.images.front"]["info"]["video.codec"],
                "h264",
            )
            for episode_index, length in enumerate(fixture["lengths"]):
                record = receipt["episodes"][episode_index]
                self.assertEqual(
                    record["tactile_replay_episode"],
                    fixture["replay_names"][episode_index],
                )
                self.assertTrue(record["initial_state_float32_bitwise_equal"])
                self.assertEqual(
                    record["initial_state_evidence"]["compared_dimensions"], 31
                )
                self.assertTrue(
                    record["initial_state_evidence"][
                        "replay_length_equals_raw_length"
                    ]
                )
                provenance = record["replay_data_only_metadata_contract"]
                self.assertEqual(
                    provenance["provenance_class"],
                    "legacy_without_schema_version",
                )
                self.assertIsNone(provenance["schema_version"])
                self.assertEqual(
                    provenance["validated_fields"],
                    {
                        "data_only": True,
                        "rgb_observations_generated": False,
                        "rgb_videos_written": False,
                        "depth_outputs_written": False,
                    },
                )
                table = pq.read_table(
                    output / f"data/chunk-000/episode_{episode_index:06d}.parquet"
                )
                np.testing.assert_array_equal(
                    _matrix(table, TACTILE_COLUMN, 216)[:, :192],
                    fixture["tactile"][episode_index][
                        fixture["start_indices"][episode_index] :
                    ].reshape(length, 192),
                )
                self.assertFalse(
                    any(name.startswith("next_tactile_gt") for name in table.column_names)
                )
                for camera in ("front", "wrist"):
                    video = output / (
                        f"videos/chunk-000/observation.images.{camera}/"
                        f"episode_{episode_index:06d}.mp4"
                    )
                    with av.open(str(video)) as container:
                        frames = [frame.to_ndarray(format="rgb24")
                                  for frame in container.decode(video=0)]
                    self.assertEqual(len(frames), length)
                    # Check decoded endpoints, not only container frame counts.
                    for index in (0, length - 1):
                        expected = 20 + episode_index * 50 + index
                        self.assertLessEqual(abs(float(frames[index].mean()) - expected), 3)

    def test_correct_v1_provenance_passes_and_is_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = _make_fixture(root)
            replay_zarr = (
                fixture["replay"] / fixture["replay_names"][0] / "replay.zarr"
            )
            _upgrade_replay_to_v1(replay_zarr)
            output = root / "output"

            summary = prepare_official_v3_tactile_rldx_v21(
                fixture["official"],
                fixture["raw"],
                fixture["replay"],
                output,
                max_episodes=1,
            )

            self.assertEqual(summary.total_episodes, 1)
            receipt = json.loads(
                (output / "meta" / JOIN_RECEIPT_NAME).read_text(encoding="utf-8")
            )
            record = receipt["episodes"][0]
            provenance = record["replay_data_only_metadata_contract"]
            self.assertEqual(provenance["provenance_class"], "versioned_v1")
            self.assertEqual(
                provenance["schema_version"], "dexjoco.replay_output.v1"
            )
            self.assertEqual(
                provenance["validated_fields"]["source_action_steps"],
                fixture["raw_actions"][0].shape[0],
            )
            self.assertTrue(record["initial_state_float32_bitwise_equal"])

    def test_one_episode_prefix_smoke(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = _make_fixture(root)
            output = root / "smoke"
            summary = prepare_official_v3_tactile_rldx_v21(
                fixture["official"],
                fixture["raw"],
                fixture["replay"],
                output,
                max_episodes=1,
                video_encoding="copy",
            )
            self.assertEqual(summary.total_episodes, 1)
            self.assertEqual(summary.total_frames, fixture["lengths"][0])
            self.assertEqual(summary.start_indices, (fixture["start_indices"][0],))
            self.assertEqual(summary.video_encoding, "copy")

    def test_fails_when_start_idx_is_not_first_action_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = _make_fixture(root)
            raw_zarr = sorted(fixture["raw"].glob("*/replay.zarr"))[0]
            data = zarr.open(str(raw_zarr), mode="a")["data"]
            action = np.asarray(data["action_rotvec"])
            action[0, 0] += 0.25
            data["action_rotvec"][:] = action
            quaternion_action = np.asarray(data["action"])
            quaternion_action[0, 0] += 0.25
            data["action"][:] = quaternion_action
            output = root / "output"

            with self.assertRaisesRegex(
                DatasetValidationError, "first adjacent-action change"
            ):
                prepare_official_v3_tactile_rldx_v21(
                    fixture["official"],
                    fixture["raw"],
                    fixture["replay"],
                    output,
                    max_episodes=1,
                )
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".output.staging-*")), [])

    def test_uses_raw_quaternion_when_legacy_cached_rotvec_prefix_is_wrong(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = _make_fixture(root)
            raw_zarr = sorted(fixture["raw"].glob("*/replay.zarr"))[0]
            data = zarr.open(str(raw_zarr), mode="a")["data"]
            cached = np.asarray(data["action_rotvec"])
            start_idx = fixture["start_indices"][0]
            cached[: start_idx + 1] = 0.0
            data["action_rotvec"][:] = cached
            official_data_path = (
                fixture["official"] / "data/chunk-000/file-000.parquet"
            )
            official_table = pq.read_table(official_data_path)
            official_action = _matrix(official_table, ACTION_COLUMN, 22).copy()
            official_action[0] = 0.0
            action_column_index = official_table.schema.get_field_index(ACTION_COLUMN)
            official_table = official_table.set_column(
                action_column_index,
                ACTION_COLUMN,
                _fixed_list(official_action),
            )
            pq.write_table(official_table, official_data_path)
            output = root / "output"

            summary = prepare_official_v3_tactile_rldx_v21(
                fixture["official"],
                fixture["raw"],
                fixture["replay"],
                output,
                max_episodes=1,
            )
            self.assertEqual(summary.total_episodes, 1)
            receipt = json.loads(
                (output / "meta" / JOIN_RECEIPT_NAME).read_text(encoding="utf-8")
            )
            self.assertEqual(
                receipt["episodes"][0]["cached_action_rotvec_mismatch_rows"],
                list(range(start_idx + 1)),
            )
            evidence = receipt["episodes"][0]["action_alignment_evidence"]
            self.assertEqual(
                evidence["cached_vs_canonical"],
                "legacy_zero_filled_prefix_only",
            )
            self.assertEqual(
                evidence["selected_official_vs_canonical_mismatch_rows"], [0]
            )

    def test_rejects_nonzero_cached_prefix_as_legacy_anomaly(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = _make_fixture(root)
            raw_zarr = sorted(fixture["raw"].glob("*/replay.zarr"))[0]
            data = zarr.open(str(raw_zarr), mode="a")["data"]
            cached = np.asarray(data["action_rotvec"])
            start_idx = fixture["start_indices"][0]
            cached[: start_idx + 1] = 0.0
            cached[: start_idx + 1, 7] = 1.0
            data["action_rotvec"][:] = cached
            official_data_path = (
                fixture["official"] / "data/chunk-000/file-000.parquet"
            )
            official_table = pq.read_table(official_data_path)
            official_action = _matrix(official_table, ACTION_COLUMN, 22).copy()
            official_action[0] = cached[start_idx]
            action_column_index = official_table.schema.get_field_index(ACTION_COLUMN)
            official_table = official_table.set_column(
                action_column_index,
                ACTION_COLUMN,
                _fixed_list(official_action),
            )
            pq.write_table(official_table, official_data_path)

            with self.assertRaisesRegex(
                DatasetValidationError, "legacy cached action prefix must be exactly zero"
            ):
                prepare_official_v3_tactile_rldx_v21(
                    fixture["official"],
                    fixture["raw"],
                    fixture["replay"],
                    root / "output",
                    max_episodes=1,
                )

    def test_rejects_missing_or_false_data_only_metadata_fields(self):
        cases = (
            ("missing_data_only", "data_only", None),
            ("false_data_only", "data_only", False),
            (
                "missing_rgb_observations",
                "rgb_observations_generated",
                None,
            ),
            ("true_rgb_observations", "rgb_observations_generated", True),
            ("missing_rgb_videos", "rgb_videos_written", None),
            ("true_rgb_videos", "rgb_videos_written", True),
            ("missing_depth_outputs", "depth_outputs_written", None),
            ("true_depth_outputs", "depth_outputs_written", True),
        )
        for label, key, replacement in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                fixture = _make_fixture(root)
                replay_zarr = (
                    fixture["replay"]
                    / fixture["replay_names"][0]
                    / "replay.zarr"
                )
                replay_root = zarr.open(str(replay_zarr), mode="a")
                metadata = dict(replay_root.attrs["replay_output"])
                if replacement is None:
                    metadata.pop(key)
                else:
                    metadata[key] = replacement
                replay_root.attrs["replay_output"] = metadata

                with self.assertRaisesRegex(
                    DatasetValidationError, rf"replay_output\.{key}"
                ):
                    prepare_official_v3_tactile_rldx_v21(
                        fixture["official"],
                        fixture["raw"],
                        fixture["replay"],
                        root / "output",
                        max_episodes=1,
                    )

    def test_rejects_replay_with_changed_initial_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = _make_fixture(root)
            replay_zarr = (
                fixture["replay"] / fixture["replay_names"][0] / "replay.zarr"
            )
            replay_data = zarr.open(str(replay_zarr), mode="a")["data"]
            replay_state = np.asarray(replay_data["state"])
            replay_state[0, 30] += 0.25
            replay_data["state"][:] = replay_state

            with self.assertRaisesRegex(
                DatasetValidationError, "31-dimensional replay state row 0"
            ):
                prepare_official_v3_tactile_rldx_v21(
                    fixture["official"],
                    fixture["raw"],
                    fixture["replay"],
                    root / "output",
                    max_episodes=1,
                )

    def test_rejects_replay_state_with_extra_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = _make_fixture(root)
            replay_zarr = (
                fixture["replay"] / fixture["replay_names"][0] / "replay.zarr"
            )
            replay_data = zarr.open(str(replay_zarr), mode="a")["data"]
            replay_state = np.asarray(replay_data["state"])
            del replay_data["state"]
            replay_data.create_dataset(
                "state",
                data=np.concatenate([replay_state, replay_state[-1:]], axis=0),
            )

            with self.assertRaisesRegex(
                DatasetValidationError, "replay state.*raw length"
            ):
                prepare_official_v3_tactile_rldx_v21(
                    fixture["official"],
                    fixture["raw"],
                    fixture["replay"],
                    root / "output",
                    max_episodes=1,
                )

    def test_rejects_invalid_v1_provenance(self):
        cases = (
            ("wrong_task", "task", "unlock_ipad"),
            (
                "state_not_restored",
                "initial_state_actually_restored",
                False,
            ),
            ("extended", "extend_steps", 1),
            ("wrong_source_length", "source_action_steps", -1),
        )
        for label, key, replacement in cases:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                fixture = _make_fixture(root)
                replay_zarr = (
                    fixture["replay"]
                    / fixture["replay_names"][0]
                    / "replay.zarr"
                )
                _upgrade_replay_to_v1(replay_zarr)
                replay_root = zarr.open(str(replay_zarr), mode="a")
                metadata = dict(replay_root.attrs["replay_output"])
                metadata[key] = replacement
                replay_root.attrs["replay_output"] = metadata

                with self.assertRaisesRegex(
                    DatasetValidationError, rf"replay_output\.{key}"
                ):
                    prepare_official_v3_tactile_rldx_v21(
                        fixture["official"],
                        fixture["raw"],
                        fixture["replay"],
                        root / "output",
                        max_episodes=1,
                    )

    def test_fails_when_only_future_tactile_is_available(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = _make_fixture(root)
            replay_zarr = fixture["replay"] / fixture["replay_names"][0] / "replay.zarr"
            data = zarr.open(str(replay_zarr), mode="a")["data"]
            del data["tactile_gt.taxel_force_local"]
            output = root / "output"

            with self.assertRaisesRegex(DatasetValidationError, "next_tactile_gt"):
                prepare_official_v3_tactile_rldx_v21(
                    fixture["official"],
                    fixture["raw"],
                    fixture["replay"],
                    output,
                    max_episodes=1,
                )
            self.assertFalse(output.exists())

    def test_fails_when_raw_quaternion_action_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = _make_fixture(root)
            raw_zarr = min(fixture["raw"].glob("*/replay.zarr"))
            data = zarr.open(str(raw_zarr), mode="a")["data"]
            del data["action"]
            output = root / "output"

            with self.assertRaisesRegex(
                DatasetValidationError, "missing raw arrays.*action"
            ):
                prepare_official_v3_tactile_rldx_v21(
                    fixture["official"],
                    fixture["raw"],
                    fixture["replay"],
                    output,
                    max_episodes=1,
                )
            self.assertFalse(output.exists())
            self.assertEqual(list(root.glob(".output.staging-*")), [])

    def test_refuses_existing_destination(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = _make_fixture(root)
            output = root / "output"
            output.mkdir()
            marker = output / "keep.txt"
            marker.write_text("keep", encoding="utf-8")

            with self.assertRaises(FileExistsError):
                prepare_official_v3_tactile_rldx_v21(
                    fixture["official"],
                    fixture["raw"],
                    fixture["replay"],
                    output,
                    max_episodes=1,
                )
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
