"""Prepare DexJoCo replay episodes for the RLDX-1 v1.0.2 data loader.

This adapter is intentionally separate from the general DexJoCo converter.  Its
input contract is the output of ``scripts/replay_demos_zarr.py`` and its output
contract is a LeRobot v2.1 directory with the extra metadata consumed by RLDX.

The adapter is fail-closed:

* every required array and video is validated before the output is created;
* all modalities use exactly the same ``start_idx``;
* the destination must not already exist;
* files are first written to a sibling staging directory and atomically renamed;
* the input tree is only opened for reading.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
from collections.abc import Iterable
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr

STATE_SOURCE_KEY = "state"
ACTION_SOURCE_KEY = "action_rotvec"
TACTILE_SOURCE_KEY = "tactile_gt.taxel_force_local"
TIMESTAMP_SOURCE_KEY = "timestamp"

STATE_COLUMN = "observation.state"
ACTION_COLUMN = "action"
TACTILE_COLUMN = "observation.tactile.contact_features"

STATE_DIM = 23
ACTION_DIM = 22
TAXEL_DIM = 1 * 4 * 16 * 3
TACTILE_DIM = TAXEL_DIM + 4 * 6
CHUNKS_SIZE = 1000
TACTILE_SCHEMA_VERSION = "dexjoco.tactile_gt.v3"
CAMERA_MAP = {
    "front": "random_camera",
    "wrist": "wrist",
}

DEFAULT_DATA_PATH = "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
DEFAULT_VIDEO_PATH = (
    "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"
)


@dataclass(frozen=True)
class VideoSpec:
    """Validated properties of one source video."""

    path: Path
    frame_count: int
    height: int
    width: int
    fps: float


@dataclass(frozen=True)
class EpisodeSpec:
    """Read-only conversion plan for one replay episode."""

    source_dir: Path
    zarr_path: Path
    source_length: int
    output_length: int
    data_fps: int
    videos: dict[str, VideoSpec]


@dataclass(frozen=True)
class ConversionSummary:
    """Summary returned after conversion or validation."""

    output_dir: Path
    total_episodes: int
    total_frames: int
    fps: int
    start_idx: int


class DatasetValidationError(ValueError):
    """Raised when a source or output violates the RLDX dataset contract."""


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False)
        file.write("\n")


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            file.write("\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise DatasetValidationError(
                    f"Invalid JSONL at {path}:{line_number}: {error}"
                ) from error
    return rows


def _require_finite(name: str, array: np.ndarray) -> None:
    if not np.issubdtype(array.dtype, np.number):
        raise DatasetValidationError(f"{name} must be numeric, got {array.dtype}")
    if not np.isfinite(array).all():
        raise DatasetValidationError(f"{name} contains NaN or infinity")


def _discover_episode_dirs(input_dir: Path) -> list[Path]:
    input_dir = input_dir.expanduser().resolve(strict=True)
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)

    if (input_dir / "replay.zarr").is_dir():
        episode_dirs = [input_dir]
    else:
        episode_dirs = sorted(
            (path.parent for path in input_dir.glob("*/replay.zarr") if path.is_dir()),
            key=lambda path: path.name,
        )

    if not episode_dirs:
        raise DatasetValidationError(
            f"No episode directory containing replay.zarr found under {input_dir}"
        )
    return episode_dirs


def _inspect_video(path: Path) -> VideoSpec:
    if not path.is_file():
        raise DatasetValidationError(f"Required video is missing: {path}")

    frame_count = 0
    height = None
    width = None
    video_fps = None
    try:
        with av.open(str(path), mode="r") as container:
            if not container.streams.video:
                raise DatasetValidationError(f"No video stream in {path}")
            video_stream = container.streams.video[0]
            rate = video_stream.average_rate or video_stream.base_rate
            if rate is not None:
                video_fps = float(rate)
            for frame in container.decode(video=0):
                if height is None:
                    height, width = frame.height, frame.width
                elif (frame.height, frame.width) != (height, width):
                    raise DatasetValidationError(
                        f"Video frame shape changes inside {path}: "
                        f"expected {(height, width)}, got {(frame.height, frame.width)}"
                    )
                frame_count += 1
    except (av.error.FFmpegError, OSError) as error:
        raise DatasetValidationError(f"Cannot decode {path}: {error}") from error

    if frame_count == 0 or height is None or width is None or video_fps is None:
        raise DatasetValidationError(f"Video contains no decodable frames: {path}")
    return VideoSpec(
        path=path,
        frame_count=frame_count,
        height=int(height),
        width=int(width),
        fps=video_fps,
    )


def _validate_tactile_metadata(root: zarr.Group, zarr_path: Path) -> None:
    tactile_meta = root.attrs.get("tactile_gt")
    if not isinstance(tactile_meta, dict):
        raise DatasetValidationError(f"Missing tactile_gt metadata in {zarr_path}")
    expected_pairs = {
        "schema_version": TACTILE_SCHEMA_VERSION,
        "frame": "per-sample fingertip body-local frame at the body origin",
        "hand_order": ["right"],
        "fingertip_order": ["index", "middle", "ring", "thumb"],
        "taxel_channels": ["tangent_u", "tangent_v", "normal_inward"],
        "taxels_per_fingertip": 16,
        "taxel_grid_shape": [4, 4],
        "taxel_frame": "per-taxel local frame",
    }
    for key, expected in expected_pairs.items():
        if tactile_meta.get(key) != expected:
            raise DatasetValidationError(
                f"{zarr_path}: tactile_gt.{key}={tactile_meta.get(key)!r}; "
                f"expected {expected!r}"
            )
    nested_expected = {
        "alignment": {
            "tactile_gt.*": "pre-action observation at row t",
            "next_tactile_gt.*": "post-action response to action at row t",
        },
        "field_units": {"taxel_force_local": ["N", "N", "N"]},
        "field_specs": {
            "taxel_force_local": {
                "axes": ["time", "hand", "fingertip", "taxel", "xyz"],
                "dtype": "float32",
            }
        },
        "taxel_projection": {
            "method": "normal/shear separated facing Gaussian projection",
            "normalization": "per-contact taxel weights sum to one",
            "support": "only taxels facing the contact inward normal",
            "layout": "2 cap rings + 2 cylinder rings x 4 columns",
        },
    }
    for section_name, fields in nested_expected.items():
        section = tactile_meta.get(section_name)
        if not isinstance(section, dict):
            raise DatasetValidationError(
                f"{zarr_path}: tactile_gt.{section_name} must be a metadata object"
            )
        for key, expected in fields.items():
            actual = section.get(key)
            if isinstance(expected, dict):
                if not isinstance(actual, dict):
                    raise DatasetValidationError(
                        f"{zarr_path}: tactile_gt.{section_name}.{key} must be "
                        "a metadata object"
                    )
                for nested_key, nested_value in expected.items():
                    if actual.get(nested_key) != nested_value:
                        raise DatasetValidationError(
                            f"{zarr_path}: tactile_gt.{section_name}.{key}."
                            f"{nested_key}={actual.get(nested_key)!r}; expected "
                            f"{nested_value!r}"
                        )
            elif actual != expected:
                raise DatasetValidationError(
                    f"{zarr_path}: tactile_gt.{section_name}.{key}={actual!r}; "
                    f"expected {expected!r}"
                )


def _current_contact_features(data, length: int, path: Path) -> np.ndarray:
    """Pack v3 taxels plus per-finger [Fx,Fy,Fz,normal,peak,contact_fraction]."""
    specs = {
        "fingertip_wrench_local": (length, 1, 4, 6),
        "fingertip_normal": (length, 1, 4),
        "fingertip_force_peak": (length, 1, 4),
        "contact_fraction": (length, 1, 4),
    }
    values = {}
    for name, shape in specs.items():
        key = "tactile_gt." + name
        if key not in data:
            raise DatasetValidationError(f"{path}: missing {key}; regenerate v3 replay")
        value = np.asarray(data[key])
        if value.shape != shape or value.dtype != np.float32:
            raise DatasetValidationError(f"{path}: {key} must be float32 {shape}")
        _require_finite(key, value)
        if name != "fingertip_wrench_local" and np.any(value < 0):
            raise DatasetValidationError(f"{path}: {key} must be nonnegative")
        values[name] = value
    if np.any(values["contact_fraction"] > 1):
        raise DatasetValidationError(f"{path}: contact_fraction exceeds one")
    if np.any(values["fingertip_force_peak"] + 1e-6 < values["fingertip_normal"]):
        raise DatasetValidationError(f"{path}: force peak is smaller than mean")
    summary = np.concatenate(
        [
            values["fingertip_wrench_local"][..., :3],
            *[
                values[name][..., None]
                for name in ("fingertip_normal", "fingertip_force_peak", "contact_fraction")
            ],
        ],
        axis=-1,
    )
    return np.ascontiguousarray(
        np.concatenate(
            [
                np.asarray(data[TACTILE_SOURCE_KEY]).reshape(length, TAXEL_DIM),
                summary.reshape(length, 24),
            ],
            axis=1,
        ),
        dtype=np.float32,
    )


def _load_source_arrays(
    zarr_path: Path,
    *,
    start_idx: int,
) -> dict[str, np.ndarray]:
    root = zarr.open(str(zarr_path), mode="r")
    if "data" not in root:
        raise DatasetValidationError(f"Missing data group in {zarr_path}")
    _validate_tactile_metadata(root, zarr_path)
    data = root["data"]

    required = (
        STATE_SOURCE_KEY,
        ACTION_SOURCE_KEY,
        TACTILE_SOURCE_KEY,
        TIMESTAMP_SOURCE_KEY,
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise DatasetValidationError(
            f"{zarr_path}: missing required arrays {missing}; "
            f"the current row-t tactile tensor is mandatory"
        )

    state_raw = np.asarray(data[STATE_SOURCE_KEY])
    action_raw = np.asarray(data[ACTION_SOURCE_KEY])
    tactile_raw = np.asarray(data[TACTILE_SOURCE_KEY])
    timestamp_raw = np.asarray(data[TIMESTAMP_SOURCE_KEY])

    if state_raw.ndim != 2 or state_raw.shape[1] < STATE_DIM:
        raise DatasetValidationError(
            f"{zarr_path}: state must have shape (T, >= {STATE_DIM}), got {state_raw.shape}"
        )
    if action_raw.ndim != 2 or action_raw.shape[1] != ACTION_DIM:
        raise DatasetValidationError(
            f"{zarr_path}: action_rotvec must have shape (T, {ACTION_DIM}), "
            f"got {action_raw.shape}"
        )
    expected_tactile_tail = (1, 4, 16, 3)
    if tactile_raw.ndim != 5 or tactile_raw.shape[1:] != expected_tactile_tail:
        raise DatasetValidationError(
            f"{zarr_path}: {TACTILE_SOURCE_KEY} must have shape "
            f"(T, 1, 4, 16, 3), got {tactile_raw.shape}"
        )
    if tactile_raw.dtype != np.float32:
        raise DatasetValidationError(
            f"{zarr_path}: {TACTILE_SOURCE_KEY} must be float32, got {tactile_raw.dtype}"
        )
    if timestamp_raw.ndim != 1:
        raise DatasetValidationError(
            f"{zarr_path}: timestamp must have shape (T,), got {timestamp_raw.shape}"
        )

    length = state_raw.shape[0]
    lengths = {
        STATE_SOURCE_KEY: length,
        ACTION_SOURCE_KEY: action_raw.shape[0],
        TACTILE_SOURCE_KEY: tactile_raw.shape[0],
        TIMESTAMP_SOURCE_KEY: timestamp_raw.shape[0],
    }
    if len(set(lengths.values())) != 1:
        raise DatasetValidationError(
            f"{zarr_path}: array lengths are not aligned: {lengths}"
        )
    if start_idx < 0 or start_idx >= length:
        raise DatasetValidationError(
            f"{zarr_path}: start_idx={start_idx} must satisfy 0 <= start_idx < {length}"
        )

    _require_finite(STATE_SOURCE_KEY, state_raw)
    _require_finite(ACTION_SOURCE_KEY, action_raw)
    _require_finite(TACTILE_SOURCE_KEY, tactile_raw)
    _require_finite(TIMESTAMP_SOURCE_KEY, timestamp_raw)

    timestamp = timestamp_raw.astype(np.float64, copy=False)
    deltas = np.diff(timestamp)
    if deltas.size == 0 or np.any(deltas <= 0):
        raise DatasetValidationError(
            f"{zarr_path}: timestamp must be strictly increasing and contain at least 2 rows"
        )

    tactile_sim_time_key = "tactile_gt.sim_time"
    if tactile_sim_time_key in data:
        tactile_time = np.asarray(data[tactile_sim_time_key]).reshape(-1)
        if tactile_time.shape != timestamp.shape or not np.allclose(
            tactile_time, timestamp, rtol=0.0, atol=1e-7
        ):
            raise DatasetValidationError(
                f"{zarr_path}: timestamp and tactile_gt.sim_time are not row-aligned"
            )

    sliced_timestamp = timestamp[start_idx:] - timestamp[start_idx]
    return {
        STATE_COLUMN: np.ascontiguousarray(
            state_raw[start_idx:, :STATE_DIM], dtype=np.float32
        ),
        ACTION_COLUMN: np.ascontiguousarray(action_raw[start_idx:], dtype=np.float32),
        TACTILE_COLUMN: np.ascontiguousarray(
            _current_contact_features(data, length, zarr_path)[start_idx:], dtype=np.float32
        ),
        "timestamp": np.ascontiguousarray(sliced_timestamp, dtype=np.float32),
        "_source_timestamp": timestamp,
    }


def _infer_data_fps(timestamp: np.ndarray, zarr_path: Path) -> int:
    deltas = np.diff(timestamp.astype(np.float64, copy=False))
    median_dt = float(np.median(deltas))
    if median_dt <= 0:
        raise DatasetValidationError(
            f"{zarr_path}: invalid timestamp interval {median_dt}"
        )
    if not np.allclose(deltas, median_dt, rtol=1e-4, atol=1e-7):
        raise DatasetValidationError(
            f"{zarr_path}: non-uniform data clock; cannot emit a fixed-fps LeRobot dataset"
        )
    exact_fps = 1.0 / median_dt
    fps = round(exact_fps)
    if fps <= 0 or not math.isclose(exact_fps, fps, rel_tol=1e-4, abs_tol=1e-4):
        raise DatasetValidationError(
            f"{zarr_path}: inferred FPS {exact_fps:.9f} is not a positive integer"
        )
    return fps


def _plan_episode(
    episode_dir: Path,
    *,
    start_idx: int,
    requested_fps: int | None,
) -> EpisodeSpec:
    zarr_path = episode_dir / "replay.zarr"
    arrays = _load_source_arrays(zarr_path, start_idx=start_idx)
    source_timestamp = arrays.pop("_source_timestamp")
    inferred_fps = _infer_data_fps(source_timestamp, zarr_path)
    if requested_fps is not None and inferred_fps != requested_fps:
        raise DatasetValidationError(
            f"{zarr_path}: inferred data FPS {inferred_fps} != requested FPS {requested_fps}"
        )
    data_fps = requested_fps if requested_fps is not None else inferred_fps
    source_length = int(source_timestamp.shape[0])

    videos = {}
    for output_camera, source_camera in CAMERA_MAP.items():
        spec = _inspect_video(episode_dir / "videos" / f"{source_camera}.mp4")
        if spec.frame_count != source_length:
            raise DatasetValidationError(
                f"{spec.path}: decoded {spec.frame_count} frames, but replay.zarr has "
                f"{source_length} rows"
            )
        videos[output_camera] = spec

    return EpisodeSpec(
        source_dir=episode_dir,
        zarr_path=zarr_path,
        source_length=source_length,
        output_length=source_length - start_idx,
        data_fps=data_fps,
        videos=videos,
    )


def _encode_trimmed_video(
    source: VideoSpec,
    destination: Path,
    *,
    start_idx: int,
    fps: int,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    try:
        with (
            av.open(str(source.path), mode="r") as input_container,
            av.open(str(destination), mode="w") as output_container,
        ):
            stream = output_container.add_stream("h264", rate=Fraction(fps, 1))
            stream.width = source.width
            stream.height = source.height
            stream.pix_fmt = "yuv420p"
            stream.options = {"crf": "18", "profile": "high"}

            for source_index, frame in enumerate(input_container.decode(video=0)):
                if source_index < start_idx:
                    continue
                rgb = frame.to_ndarray(format="rgb24")
                output_frame = av.VideoFrame.from_ndarray(
                    np.ascontiguousarray(rgb), format="rgb24"
                )
                for packet in stream.encode(output_frame):
                    output_container.mux(packet)
                written += 1
            for packet in stream.encode():
                output_container.mux(packet)
    except (av.error.FFmpegError, OSError) as error:
        raise DatasetValidationError(
            f"Failed to encode aligned video {source.path} -> {destination}: {error}"
        ) from error

    expected = source.frame_count - start_idx
    if written != expected:
        raise DatasetValidationError(
            f"{source.path}: wrote {written} frames after start_idx={start_idx}; "
            f"expected {expected}"
        )
    output_spec = _inspect_video(destination)
    if output_spec.frame_count != expected:
        raise DatasetValidationError(
            f"{destination}: encoded frame count {output_spec.frame_count}; expected {expected}"
        )
    if not math.isclose(output_spec.fps, fps, rel_tol=0.0, abs_tol=1e-6):
        raise DatasetValidationError(
            f"{destination}: encoded FPS {output_spec.fps}; expected data FPS {fps}"
        )


def _fixed_size_float_array(array: np.ndarray) -> pa.FixedSizeListArray:
    if array.ndim != 2:
        raise ValueError(f"Expected a 2-D matrix, got {array.shape}")
    values = pa.array(array.reshape(-1), type=pa.float32())
    return pa.FixedSizeListArray.from_arrays(values, list_size=array.shape[1])


def _write_episode_parquet(
    destination: Path,
    arrays: dict[str, np.ndarray],
    *,
    episode_index: int,
    global_start_index: int,
) -> None:
    length = arrays[STATE_COLUMN].shape[0]
    columns = {
        STATE_COLUMN: _fixed_size_float_array(arrays[STATE_COLUMN]),
        ACTION_COLUMN: _fixed_size_float_array(arrays[ACTION_COLUMN]),
        TACTILE_COLUMN: _fixed_size_float_array(arrays[TACTILE_COLUMN]),
        "timestamp": pa.array(arrays["timestamp"], type=pa.float32()),
        "frame_index": pa.array(np.arange(length, dtype=np.int64)),
        "episode_index": pa.array(np.full(length, episode_index, dtype=np.int64)),
        "index": pa.array(
            np.arange(global_start_index, global_start_index + length, dtype=np.int64)
        ),
        "task_index": pa.array(np.zeros(length, dtype=np.int64)),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table(columns), destination, compression="zstd")


def _stats(array: np.ndarray, *, include_quantiles: bool) -> dict[str, Any]:
    values = np.asarray(array)
    if values.ndim == 1:
        values = values[:, None]
    values64 = values.astype(np.float64, copy=False)
    result = {
        "min": np.min(values64, axis=0).tolist(),
        "max": np.max(values64, axis=0).tolist(),
        "mean": np.mean(values64, axis=0).tolist(),
        "std": np.std(values64, axis=0).tolist(),
        "count": [int(values64.shape[0])],
    }
    if include_quantiles:
        result["q01"] = np.quantile(values64, 0.01, axis=0).tolist()
        result["q99"] = np.quantile(values64, 0.99, axis=0).tolist()
    return result


def _state_names() -> list[str]:
    return (
        ["eef_x", "eef_y", "eef_z"]
        + ["eef_qw", "eef_qx", "eef_qy", "eef_qz"]
        + [f"allegro_joint_{index:02d}" for index in range(16)]
    )


def _action_names() -> list[str]:
    return (
        ["eef_x", "eef_y", "eef_z"]
        + ["eef_rotvec_x", "eef_rotvec_y", "eef_rotvec_z"]
        + [f"allegro_joint_{index:02d}" for index in range(16)]
    )


def _tactile_names() -> list[str]:
    names = []
    for fingertip in ("index", "middle", "ring", "thumb"):
        for taxel in range(16):
            for axis in ("tangent_u", "tangent_v", "normal_inward"):
                names.append(f"right/{fingertip}/taxel_{taxel:02d}/{axis}")
    for fingertip in ("index", "middle", "ring", "thumb"):
        for field in ("Fx", "Fy", "Fz", "normal", "peak", "contact_fraction"):
            names.append(f"right/{fingertip}/summary/{field}")
    return names


def _build_info(
    plans: list[EpisodeSpec],
    *,
    robot_type: str,
) -> dict[str, Any]:
    fps = plans[0].data_fps
    camera_features = {}
    for camera_key in CAMERA_MAP:
        spec = plans[0].videos[camera_key]
        camera_features[f"observation.images.{camera_key}"] = {
            "dtype": "video",
            "shape": [spec.height, spec.width, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.fps": fps,
                "video.codec": "h264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.channels": 3,
                "has_audio": False,
            },
        }
    features = {
        STATE_COLUMN: {
            "dtype": "float32",
            "shape": [STATE_DIM],
            "names": _state_names(),
        },
        ACTION_COLUMN: {
            "dtype": "float32",
            "shape": [ACTION_DIM],
            "names": _action_names(),
        },
        TACTILE_COLUMN: {
            "dtype": "float32",
            "shape": [TACTILE_DIM],
            "names": _tactile_names(),
        },
        **camera_features,
        "timestamp": {"dtype": "float32", "shape": [1], "names": None},
        "frame_index": {"dtype": "int64", "shape": [1], "names": None},
        "episode_index": {"dtype": "int64", "shape": [1], "names": None},
        "index": {"dtype": "int64", "shape": [1], "names": None},
        "task_index": {"dtype": "int64", "shape": [1], "names": None},
    }
    total_episodes = len(plans)
    return {
        "codebase_version": "v2.1",
        "robot_type": robot_type,
        "total_episodes": total_episodes,
        "total_frames": sum(plan.output_length for plan in plans),
        "total_tasks": 1,
        "total_videos": total_episodes * len(CAMERA_MAP),
        "total_chunks": math.ceil(total_episodes / CHUNKS_SIZE),
        "chunks_size": CHUNKS_SIZE,
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": DEFAULT_DATA_PATH,
        "video_path": DEFAULT_VIDEO_PATH,
        "features": features,
    }


def _build_modality_metadata() -> dict[str, Any]:
    return {
        "state": {
            "right_hand_state": {
                "start": 0,
                "end": STATE_DIM,
                "original_key": STATE_COLUMN,
            }
        },
        "action": {
            "right_hand_action_rotvec": {
                "start": 0,
                "end": ACTION_DIM,
                "original_key": ACTION_COLUMN,
            }
        },
        "tactile": {
            "contact_features": {
                "start": 0,
                "end": TACTILE_DIM,
                "original_key": TACTILE_COLUMN,
            }
        },
        "video": {
            camera_key: {"original_key": f"observation.images.{camera_key}"}
            for camera_key in CAMERA_MAP
        },
    }


def _write_dataset(
    staging_dir: Path,
    plans: list[EpisodeSpec],
    *,
    start_idx: int,
    task: str,
    robot_type: str,
) -> None:
    global_start_index = 0
    episode_rows = []
    episode_stats_rows = []
    global_stats_arrays: dict[str, list[np.ndarray]] = {
        STATE_COLUMN: [],
        ACTION_COLUMN: [],
        TACTILE_COLUMN: [],
        "timestamp": [],
    }

    for episode_index, plan in enumerate(plans):
        arrays = _load_source_arrays(plan.zarr_path, start_idx=start_idx)
        arrays.pop("_source_timestamp")
        length = arrays[STATE_COLUMN].shape[0]
        if length != plan.output_length:
            raise DatasetValidationError(
                f"{plan.zarr_path}: source changed after validation "
                f"({length} != {plan.output_length})"
            )

        chunk_index = episode_index // CHUNKS_SIZE
        parquet_path = staging_dir / DEFAULT_DATA_PATH.format(
            episode_chunk=chunk_index,
            episode_index=episode_index,
        )
        _write_episode_parquet(
            parquet_path,
            arrays,
            episode_index=episode_index,
            global_start_index=global_start_index,
        )

        for output_camera, source_video in plan.videos.items():
            video_key = f"observation.images.{output_camera}"
            video_path = staging_dir / DEFAULT_VIDEO_PATH.format(
                episode_chunk=chunk_index,
                video_key=video_key,
                episode_index=episode_index,
            )
            _encode_trimmed_video(
                source_video,
                video_path,
                start_idx=start_idx,
                fps=plan.data_fps,
            )

        episode_rows.append(
            {
                "episode_index": episode_index,
                "tasks": [task],
                "length": length,
            }
        )
        episode_stats_rows.append(
            {
                "episode_index": episode_index,
                "stats": {
                    key: _stats(value, include_quantiles=False)
                    for key, value in arrays.items()
                },
            }
        )
        for key, chunks in global_stats_arrays.items():
            chunks.append(arrays[key])
        global_start_index += length

    info = _build_info(plans, robot_type=robot_type)
    meta_dir = staging_dir / "meta"
    _write_json(meta_dir / "info.json", info)
    _write_jsonl(meta_dir / "episodes.jsonl", episode_rows)
    _write_jsonl(meta_dir / "episodes_stats.jsonl", episode_stats_rows)
    _write_jsonl(meta_dir / "tasks.jsonl", [{"task_index": 0, "task": task}])
    _write_json(meta_dir / "modality.json", _build_modality_metadata())
    _write_json(
        meta_dir / "stats.json",
        {
            key: _stats(np.concatenate(chunks, axis=0), include_quantiles=True)
            for key, chunks in global_stats_arrays.items()
        },
    )

    # The selected 23-D state (quaternion) and 22-D absolute action (rotvec)
    # do not share a valid coordinate-wise relative-action definition.  Keeping
    # this empty makes an accidental RELATIVE ActionConfig fail loudly in RLDX.
    _write_json(meta_dir / "relative_stats.json", {})


def _validate_stats_entry(
    stats: dict[str, Any],
    key: str,
    expected_dim: int,
    observed: np.ndarray,
) -> None:
    if key not in stats or not isinstance(stats[key], dict):
        raise DatasetValidationError(f"stats.json is missing {key}")
    expected = _stats(observed, include_quantiles=True)
    for stat_name in ("count", "min", "max", "mean", "std", "q01", "q99"):
        try:
            values = np.asarray(stats[key].get(stat_name), dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise DatasetValidationError(
                f"stats.json {key}.{stat_name} must contain numeric values"
            ) from error
        expected_shape = (1,) if stat_name == "count" else (expected_dim,)
        if values.shape != expected_shape or not np.isfinite(values).all():
            raise DatasetValidationError(
                f"stats.json {key}.{stat_name} must have shape {expected_shape} "
                "and contain only finite numbers"
            )
        expected_values = np.asarray(expected[stat_name], dtype=np.float64)
        if stat_name == "count":
            matches = np.array_equal(values, expected_values)
        else:
            matches = np.allclose(
                values,
                expected_values,
                rtol=1e-6,
                atol=1e-7,
                equal_nan=False,
            )
        if not matches:
            max_error = float(np.max(np.abs(values - expected_values)))
            raise DatasetValidationError(
                f"stats.json {key}.{stat_name} does not match Parquet data "
                f"(max_abs_error={max_error:.9g})"
            )


def _fixed_list_matrix(table: pa.Table, key: str, expected_dim: int) -> np.ndarray:
    column = table[key].combine_chunks()
    if (
        not pa.types.is_fixed_size_list(column.type)
        or column.type.list_size != expected_dim
    ):
        raise DatasetValidationError(
            f"Parquet column {key} must be fixed_size_list<float32>[{expected_dim}], "
            f"got {column.type}"
        )
    if column.type.value_type != pa.float32():
        raise DatasetValidationError(f"Parquet column {key} must contain float32")
    values = column.values.to_numpy(zero_copy_only=False).reshape(-1, expected_dim)
    _require_finite(key, values)
    return values


def validate_rldx_lerobot_dataset(output_dir: Path | str) -> ConversionSummary:
    """Validate an existing output against the RLDX v1.0.2 loader contract."""

    root = Path(output_dir).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    meta_dir = root / "meta"
    required_meta = (
        "info.json",
        "episodes.jsonl",
        "episodes_stats.jsonl",
        "tasks.jsonl",
        "stats.json",
        "modality.json",
        "relative_stats.json",
    )
    missing_meta = [name for name in required_meta if not (meta_dir / name).is_file()]
    if missing_meta:
        raise DatasetValidationError(f"Missing metadata files: {missing_meta}")

    info = _read_json(meta_dir / "info.json")
    episodes = _read_jsonl(meta_dir / "episodes.jsonl")
    episode_stats = _read_jsonl(meta_dir / "episodes_stats.jsonl")
    tasks = _read_jsonl(meta_dir / "tasks.jsonl")
    modality = _read_json(meta_dir / "modality.json")
    stats = _read_json(meta_dir / "stats.json")
    relative_stats = _read_json(meta_dir / "relative_stats.json")

    if info.get("codebase_version") != "v2.1":
        raise DatasetValidationError("info.json codebase_version must be v2.1")
    total_episodes = int(info.get("total_episodes", -1))
    total_frames = int(info.get("total_frames", -1))
    fps = int(info.get("fps", -1))
    if total_episodes <= 0 or total_frames <= 0 or fps <= 0:
        raise DatasetValidationError("info.json totals and fps must be positive")
    if len(episodes) != total_episodes or len(episode_stats) != total_episodes:
        raise DatasetValidationError("Episode metadata count disagrees with info.json")
    if len(tasks) != 1 or tasks[0].get("task_index") != 0 or not tasks[0].get("task"):
        raise DatasetValidationError("tasks.jsonl must contain exactly task_index 0")
    if not isinstance(relative_stats, dict):
        raise DatasetValidationError("relative_stats.json must contain a JSON object")

    features = info.get("features", {})
    expected_features = {
        STATE_COLUMN: STATE_DIM,
        ACTION_COLUMN: ACTION_DIM,
        TACTILE_COLUMN: TACTILE_DIM,
    }
    for key, dim in expected_features.items():
        feature = features.get(key, {})
        if feature.get("dtype") != "float32" or feature.get("shape") != [dim]:
            raise DatasetValidationError(
                f"info.json feature {key} must be float32 with shape [{dim}]"
            )

    required_modality_entries = {
        ("state", "right_hand_state", STATE_COLUMN, STATE_DIM),
        ("action", "right_hand_action_rotvec", ACTION_COLUMN, ACTION_DIM),
        ("tactile", "contact_features", TACTILE_COLUMN, TACTILE_DIM),
    }
    for modality_name, group_name, original_key, dim in required_modality_entries:
        entry = modality.get(modality_name, {}).get(group_name, {})
        if entry != {"start": 0, "end": dim, "original_key": original_key}:
            raise DatasetValidationError(
                f"Invalid modality.json entry {modality_name}.{group_name}: {entry}"
            )
    for camera_key in CAMERA_MAP:
        original_key = f"observation.images.{camera_key}"
        if (
            modality.get("video", {}).get(camera_key, {}).get("original_key")
            != original_key
        ):
            raise DatasetValidationError(f"Invalid video mapping for {camera_key}")
        if features.get(original_key, {}).get("dtype") != "video":
            raise DatasetValidationError(f"Missing video feature {original_key}")

    expected_global_index = 0
    observed_total_frames = 0
    observed_stats: dict[str, list[np.ndarray]] = {
        key: [] for key in expected_features
    }
    for episode_index, episode in enumerate(episodes):
        if episode.get("episode_index") != episode_index:
            raise DatasetValidationError(
                "episodes.jsonl indices must be contiguous from zero"
            )
        length = int(episode.get("length", -1))
        if length <= 0 or not episode.get("tasks"):
            raise DatasetValidationError(
                f"Invalid episode metadata at index {episode_index}"
            )
        chunk_index = episode_index // int(info["chunks_size"])
        parquet_path = root / info["data_path"].format(
            episode_chunk=chunk_index,
            episode_index=episode_index,
        )
        if not parquet_path.is_file():
            raise DatasetValidationError(f"Missing parquet file: {parquet_path}")
        table = pq.read_table(parquet_path)
        if table.num_rows != length:
            raise DatasetValidationError(
                f"{parquet_path}: {table.num_rows} rows, expected {length}"
            )
        if any(name.startswith("next_tactile_gt") for name in table.column_names):
            raise DatasetValidationError(
                f"Post-action tactile column found in {parquet_path}"
            )
        for key, dim in expected_features.items():
            observed_stats[key].append(_fixed_list_matrix(table, key, dim))

        frame_index = table["frame_index"].combine_chunks().to_numpy()
        episode_column = table["episode_index"].combine_chunks().to_numpy()
        global_index = table["index"].combine_chunks().to_numpy()
        task_index = table["task_index"].combine_chunks().to_numpy()
        if not np.array_equal(frame_index, np.arange(length)):
            raise DatasetValidationError(f"{parquet_path}: invalid frame_index")
        if not np.array_equal(episode_column, np.full(length, episode_index)):
            raise DatasetValidationError(f"{parquet_path}: invalid episode_index")
        if not np.array_equal(
            global_index,
            np.arange(expected_global_index, expected_global_index + length),
        ):
            raise DatasetValidationError(f"{parquet_path}: invalid global index")
        if not np.array_equal(task_index, np.zeros(length, dtype=np.int64)):
            raise DatasetValidationError(f"{parquet_path}: invalid task_index")
        timestamp = table["timestamp"].combine_chunks().to_numpy()
        if timestamp.shape != (length,) or not np.allclose(
            timestamp,
            np.arange(length, dtype=np.float64) / fps,
            rtol=1e-4,
            atol=1e-6,
        ):
            raise DatasetValidationError(
                f"{parquet_path}: timestamps are not aligned to fps"
            )

        for camera_key in CAMERA_MAP:
            video_key = f"observation.images.{camera_key}"
            video_path = root / info["video_path"].format(
                episode_chunk=chunk_index,
                video_key=video_key,
                episode_index=episode_index,
            )
            video_spec = _inspect_video(video_path)
            if video_spec.frame_count != length:
                raise DatasetValidationError(
                    f"{video_path}: {video_spec.frame_count} frames, expected {length}"
                )
            if not math.isclose(video_spec.fps, fps, rel_tol=0.0, abs_tol=1e-6):
                raise DatasetValidationError(
                    f"{video_path}: FPS {video_spec.fps}, expected {fps}"
                )
            feature_shape = features[video_key]["shape"]
            if feature_shape != [video_spec.height, video_spec.width, 3]:
                raise DatasetValidationError(
                    f"{video_path}: decoded shape does not match info.json"
                )

        expected_global_index += length
        observed_total_frames += length

    if observed_total_frames != total_frames:
        raise DatasetValidationError(
            f"Observed {observed_total_frames} frames, info.json says {total_frames}"
        )
    for key, dim in expected_features.items():
        _validate_stats_entry(
            stats,
            key,
            dim,
            np.concatenate(observed_stats[key], axis=0),
        )
    return ConversionSummary(
        output_dir=root,
        total_episodes=total_episodes,
        total_frames=total_frames,
        fps=fps,
        start_idx=0,
    )


def prepare_rldx_lerobot_dataset(
    input_dir: Path | str,
    output_dir: Path | str,
    *,
    task: str,
    start_idx: int = 0,
    fps: int | None = None,
    robot_type: str = "dexjoco_allegro_right",
) -> ConversionSummary:
    """Convert replay episode directories into an RLDX-ready LeRobot v2.1 dataset."""

    if not task.strip():
        raise ValueError("task must be a non-empty language instruction")
    if start_idx < 0:
        raise ValueError("start_idx must be non-negative")
    if fps is not None and fps <= 0:
        raise ValueError("fps must be positive")

    source_root = Path(input_dir).expanduser().resolve(strict=True)
    destination = Path(output_dir).expanduser().resolve(strict=False)
    if destination.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing output (fail-closed): {destination}"
        )
    if destination == source_root or source_root in destination.parents:
        raise DatasetValidationError(
            f"Output must not be inside the input tree: {destination}"
        )

    episode_dirs = _discover_episode_dirs(source_root)
    plans = [
        _plan_episode(
            episode_dir,
            start_idx=start_idx,
            requested_fps=fps,
        )
        for episode_dir in episode_dirs
    ]
    data_fps_values = {plan.data_fps for plan in plans}
    if len(data_fps_values) != 1:
        raise DatasetValidationError(
            f"All episodes must share one data FPS, got {sorted(data_fps_values)}"
        )
    for camera_key in CAMERA_MAP:
        shapes = {
            (plan.videos[camera_key].height, plan.videos[camera_key].width)
            for plan in plans
        }
        if len(shapes) != 1:
            raise DatasetValidationError(
                f"Camera {camera_key} changes resolution across episodes: {sorted(shapes)}"
            )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-",
            dir=str(destination.parent),
        )
    )
    try:
        _write_dataset(
            staging_dir,
            plans,
            start_idx=start_idx,
            task=task.strip(),
            robot_type=robot_type,
        )
        validated = validate_rldx_lerobot_dataset(staging_dir)
        if validated.total_frames != sum(plan.output_length for plan in plans):
            raise DatasetValidationError("Validated frame count changed unexpectedly")
        os.replace(staging_dir, destination)
    except Exception:
        shutil.rmtree(staging_dir, ignore_errors=True)
        raise

    return ConversionSummary(
        output_dir=destination,
        total_episodes=len(plans),
        total_frames=sum(plan.output_length for plan in plans),
        fps=plans[0].data_fps,
        start_idx=start_idx,
    )
