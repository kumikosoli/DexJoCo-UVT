from __future__ import annotations

import hashlib
import json
import os
import tempfile
import unittest
from pathlib import Path

import av
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import zarr
from dexjoco_data_converter.to_lerobot.rldx_replay_lerobot import (
    ACTION_COLUMN,
    STATE_COLUMN,
    TACTILE_COLUMN,
    DatasetValidationError,
    prepare_rldx_lerobot_dataset,
    validate_rldx_lerobot_dataset,
)

_REAL_REPLAY_ROOT_ENV = os.environ.get("DEXJOCO_REAL_REPLAY_ROOT")
REAL_REPLAY_ROOT = (
    Path(_REAL_REPLAY_ROOT_ENV) if _REAL_REPLAY_ROOT_ENV else None
)


def _tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(path for path in root.rglob("*") if path.is_file()):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_video(path: Path, values: list[int], *, fps: int = 30) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), mode="w") as container:
        stream = container.add_stream("h264", rate=fps)
        stream.width = 64
        stream.height = 48
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": "12"}
        for value in values:
            image = np.full((48, 64, 3), value, dtype=np.uint8)
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def _decode_video_means(path: Path) -> list[float]:
    with av.open(str(path), mode="r") as container:
        return [
            float(frame.to_ndarray(format="rgb24").mean())
            for frame in container.decode(video=0)
        ]


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


def _make_replay_episode(root: Path, name: str, *, length: int = 8) -> Path:
    episode_dir = root / name
    zarr_path = episode_dir / "replay.zarr"
    group = zarr.open(str(zarr_path), mode="w")
    data = group.create_group("data")

    row = np.arange(length, dtype=np.float32)[:, None]
    state = row * np.float32(100.0) + np.arange(31, dtype=np.float32)[None, :]
    action = row * np.float32(10.0) + np.arange(22, dtype=np.float32)[None, :]
    tactile = np.arange(length * 1 * 4 * 16 * 3, dtype=np.float32).reshape(
        length, 1, 4, 16, 3
    )
    timestamp = np.arange(length, dtype=np.float64) / 50.0

    data.create_dataset("state", data=state)
    data.create_dataset("action_rotvec", data=action)
    data.create_dataset("tactile_gt.taxel_force_local", data=tactile)
    data.create_dataset("tactile_gt.fingertip_wrench_local", data=np.zeros((length, 1, 4, 6), np.float32))
    for name in ("fingertip_normal", "fingertip_force_peak", "contact_fraction"):
        data.create_dataset("tactile_gt." + name, data=np.zeros((length, 1, 4), np.float32))
    data.create_dataset(
        "next_tactile_gt.taxel_force_local",
        data=tactile + np.float32(100000.0),
    )
    data.create_dataset("timestamp", data=timestamp)
    data.create_dataset("tactile_gt.sim_time", data=timestamp[:, None])
    group.attrs["tactile_gt"] = _tactile_v3_metadata()

    values = [10 + 25 * index for index in range(length)]
    _write_video(episode_dir / "videos" / "random_camera.mp4", values)
    _write_video(episode_dir / "videos" / "wrist.mp4", values)
    return episode_dir


def _fixed_list_to_numpy(table, key: str, width: int) -> np.ndarray:
    column = table[key].combine_chunks()
    return column.values.to_numpy(zero_copy_only=False).reshape(-1, width)


class ContactFeatureRegressionTest(unittest.TestCase):
    def test_summary_preserves_force_peak_and_contact_fraction(self):
        from dexjoco_data_converter.to_lerobot.rldx_replay_lerobot import _current_contact_features
        data = {"tactile_gt.taxel_force_local": np.zeros((2, 1, 4, 16, 3), np.float32),
                "tactile_gt.fingertip_wrench_local": np.zeros((2, 1, 4, 6), np.float32),
                "tactile_gt.fingertip_normal": np.full((2, 1, 4), .1, np.float32),
                "tactile_gt.fingertip_force_peak": np.ones((2, 1, 4), np.float32),
                "tactile_gt.contact_fraction": np.full((2, 1, 4), .1, np.float32)}
        data["tactile_gt.fingertip_wrench_local"][..., :3] = [1., 2., 3.]
        packed = _current_contact_features(data, 2, Path("fixture"))
        self.assertEqual(packed.shape, (2, 216))
        np.testing.assert_allclose(packed[:, 192:].reshape(2, 4, 6),
                                  np.broadcast_to([1., 2., 3., .1, 1., .1], (2, 4, 6)))
        del data["tactile_gt.fingertip_force_peak"]
        with self.assertRaisesRegex(DatasetValidationError, "regenerate v3 replay"):
            _current_contact_features(data, 2, Path("fixture"))


class RldxReplayLerobotTest(unittest.TestCase):
    def test_conversion_uses_current_tactile_and_one_start_idx(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_root = tmp_path / "input"
            episode_dir = _make_replay_episode(input_root, "episode_b")
            _make_replay_episode(input_root, "episode_a")
            before = _tree_digest(input_root)
            output = tmp_path / "output"

            summary = prepare_rldx_lerobot_dataset(
                input_root,
                output,
                task="Grasp the tongs and open and close them three times.",
                start_idx=2,
            )

            self.assertEqual(summary.total_episodes, 2)
            self.assertEqual(summary.total_frames, 12)
            self.assertEqual(summary.fps, 50)
            self.assertEqual(_tree_digest(input_root), before)
            validate_rldx_lerobot_dataset(output)

            table = pq.read_table(output / "data/chunk-000/episode_000001.parquet")
            source = zarr.open(str(episode_dir / "replay.zarr"), mode="r")["data"]
            np.testing.assert_array_equal(
                _fixed_list_to_numpy(table, STATE_COLUMN, 23),
                np.asarray(source["state"])[2:, :23].astype(np.float32),
            )
            np.testing.assert_array_equal(
                _fixed_list_to_numpy(table, ACTION_COLUMN, 22),
                np.asarray(source["action_rotvec"])[2:].astype(np.float32),
            )
            output_tactile = _fixed_list_to_numpy(table, TACTILE_COLUMN, 216)[:, :192]
            current_tactile = np.asarray(source["tactile_gt.taxel_force_local"])[
                2:
            ].reshape(-1, 192)
            post_action_tactile = np.asarray(
                source["next_tactile_gt.taxel_force_local"]
            )[2:].reshape(-1, 192)
            np.testing.assert_array_equal(output_tactile, current_tactile)
            self.assertFalse(np.array_equal(output_tactile, post_action_tactile))

            np.testing.assert_allclose(
                table["timestamp"].combine_chunks().to_numpy(),
                np.arange(6) / 50.0,
                atol=1e-7,
            )
            front_path = (
                output / "videos/chunk-000/observation.images.front/episode_000001.mp4"
            )
            means = _decode_video_means(front_path)
            self.assertEqual(len(means), 6)
            self.assertAlmostEqual(means[0], 60.0, delta=4.0)

            # RLDX v1.0.2 uses pandas.read_parquet and specifically checks
            # whether each vector cell is a numpy.ndarray before slicing it.
            dataframe = pd.read_parquet(
                output / "data/chunk-000/episode_000000.parquet"
            )
            self.assertIsInstance(dataframe[STATE_COLUMN].iloc[0], np.ndarray)
            self.assertIsInstance(dataframe[ACTION_COLUMN].iloc[0], np.ndarray)
            self.assertIsInstance(dataframe[TACTILE_COLUMN].iloc[0], np.ndarray)
            self.assertEqual(dataframe[TACTILE_COLUMN].iloc[0].shape, (216,))

            modality = json.loads((output / "meta/modality.json").read_text())
            self.assertEqual(
                modality["tactile"]["contact_features"]["original_key"],
                TACTILE_COLUMN,
            )
            self.assertEqual(
                json.loads((output / "meta/relative_stats.json").read_text()), {}
            )

    def test_fail_closed_on_missing_current_tactile(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_root = tmp_path / "input"
            episode = _make_replay_episode(input_root, "episode")
            data = zarr.open(str(episode / "replay.zarr"), mode="a")["data"]
            del data["tactile_gt.taxel_force_local"]
            output = tmp_path / "output"

            with self.assertRaisesRegex(DatasetValidationError, "current row-t"):
                prepare_rldx_lerobot_dataset(
                    input_root,
                    output,
                    task="Grasp the tongs.",
                )
            self.assertFalse(output.exists())

    def test_fail_closed_on_incompatible_tactile_v3_metadata(self):
        cases = (
            ("schema_version", ("schema_version",), "dexjoco.tactile_gt.v1"),
            ("schema_version", ("schema_version",), "dexjoco.tactile_gt.v2"),
            (
                "taxel_channels",
                ("taxel_channels",),
                ["normal_inward", "tangent_u", "tangent_v"],
            ),
            ("taxel_frame", ("taxel_frame",), "fingertip body frame"),
            (
                "field_units.taxel_force_local",
                ("field_units", "taxel_force_local"),
                ["kg", "kg", "kg"],
            ),
            (
                "field_specs.taxel_force_local.axes",
                ("field_specs", "taxel_force_local", "axes"),
                ["time", "hand", "fingertip", "xyz", "taxel"],
            ),
            (
                "field_specs.taxel_force_local.dtype",
                ("field_specs", "taxel_force_local", "dtype"),
                "float64",
            ),
            (
                "alignment.tactile_gt.*",
                ("alignment", "tactile_gt.*"),
                "post-action response to action at row t",
            ),
            (
                "taxel_projection.method",
                ("taxel_projection", "method"),
                "nearest taxel only",
            ),
        )
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_root = tmp_path / "input"
            episode = _make_replay_episode(input_root, "episode")
            zarr_root = zarr.open(str(episode / "replay.zarr"), mode="a")

            for case_index, (label, path, invalid_value) in enumerate(cases):
                with self.subTest(label=label):
                    metadata = json.loads(json.dumps(_tactile_v3_metadata()))
                    target = metadata
                    for component in path[:-1]:
                        target = target[component]
                    target[path[-1]] = invalid_value
                    zarr_root.attrs["tactile_gt"] = metadata
                    output = tmp_path / f"output_{case_index}"

                    with self.assertRaises(DatasetValidationError) as raised:
                        prepare_rldx_lerobot_dataset(
                            input_root,
                            output,
                            task="Grasp the tongs.",
                        )
                    self.assertIn(label, str(raised.exception))
                    self.assertFalse(output.exists())

    def test_validator_recomputes_every_vector_stat_from_parquet(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_root = tmp_path / "input"
            _make_replay_episode(input_root, "episode")
            output = tmp_path / "output"
            prepare_rldx_lerobot_dataset(
                input_root,
                output,
                task="Grasp the tongs.",
                start_idx=2,
            )
            stats_path = output / "meta/stats.json"
            expected_stats = json.loads(stats_path.read_text(encoding="utf-8"))

            for key in (STATE_COLUMN, ACTION_COLUMN, TACTILE_COLUMN):
                for stat_name in ("count", "min", "max", "mean", "std", "q01", "q99"):
                    with self.subTest(key=key, stat_name=stat_name):
                        corrupted = json.loads(json.dumps(expected_stats))
                        increment = 1 if stat_name == "count" else 0.25
                        corrupted[key][stat_name][0] += increment
                        stats_path.write_text(
                            json.dumps(corrupted),
                            encoding="utf-8",
                        )
                        with self.assertRaises(DatasetValidationError) as raised:
                            validate_rldx_lerobot_dataset(output)
                        self.assertIn(f"{key}.{stat_name}", str(raised.exception))

            stats_path.write_text(json.dumps(expected_stats), encoding="utf-8")
            validate_rldx_lerobot_dataset(output)

    def test_refuses_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_root = tmp_path / "input"
            _make_replay_episode(input_root, "episode")
            output = tmp_path / "output"
            output.mkdir()
            marker = output / "keep.txt"
            marker.write_text("do not overwrite")

            with self.assertRaises(FileExistsError):
                prepare_rldx_lerobot_dataset(
                    input_root,
                    output,
                    task="Grasp the tongs.",
                )
            self.assertEqual(marker.read_text(), "do not overwrite")

    def test_fail_closed_when_a_video_drops_a_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_root = tmp_path / "input"
            episode = _make_replay_episode(input_root, "episode", length=8)
            _write_video(
                episode / "videos" / "wrist.mp4",
                [10 + 25 * index for index in range(7)],
            )
            output = tmp_path / "output"

            with self.assertRaisesRegex(DatasetValidationError, "decoded 7 frames"):
                prepare_rldx_lerobot_dataset(
                    input_root,
                    output,
                    task="Grasp the tongs.",
                )
            self.assertFalse(output.exists())

    @unittest.skipUnless(
        REAL_REPLAY_ROOT is not None and REAL_REPLAY_ROOT.is_dir(),
        "set DEXJOCO_REAL_REPLAY_ROOT to run the real replay sample test",
    )
    def test_real_375_frame_replay_sample(self):
        assert REAL_REPLAY_ROOT is not None
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "rldx_dataset"
            source_episode = next(REAL_REPLAY_ROOT.glob("*/replay.zarr")).parent
            summary = prepare_rldx_lerobot_dataset(
                REAL_REPLAY_ROOT,
                output,
                task="Grasp the tongs and perform three open-close motions.",
            )
            self.assertEqual(summary.total_episodes, 1)
            self.assertEqual(summary.total_frames, 375)
            self.assertEqual(summary.fps, 50)

            table = pq.read_table(output / "data/chunk-000/episode_000000.parquet")
            source = zarr.open(str(source_episode / "replay.zarr"), mode="r")["data"]
            np.testing.assert_array_equal(
                _fixed_list_to_numpy(table, TACTILE_COLUMN, 216)[:, :192],
                np.asarray(source["tactile_gt.taxel_force_local"]).reshape(375, 192),
            )
            self.assertEqual(
                len(
                    _decode_video_means(
                        output / "videos/chunk-000/observation.images.front/"
                        "episode_000000.mp4"
                    )
                ),
                375,
            )


if __name__ == "__main__":
    unittest.main()
