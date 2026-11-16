"""Canonicalize optional tactile observations for merged Zarr datasets.

DexJoCo records each demonstration in its own Zarr store.  Sensor groups are
therefore allowed to be absent from a whole episode, while a merged replay
buffer requires every episode to contain exactly the same arrays.  This module
discovers the sensor schema before an output store is created and materializes
missing streams with zero payloads plus explicit validity masks.

Only the semantic hand axis is padded.  All other dimensions and all dtypes
must agree across sensor-enabled episodes; incompatible sensor schemas fail
instead of being silently flattened or widened.
"""

from __future__ import annotations

import copy
import json
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import zarr


SENSOR_GROUPS = ("tactile_gt",)
_RETIRED_SENSOR_GROUPS = ("tactile",)
TARGET_HAND_ORDER = ("right", "left")
SENSOR_PADDING_SCHEMA_VERSION = "dexjoco.sensor_padding.v1"
_GENERATED_FIELDS = frozenset(("sensor_present", "hand_valid"))
_HAND_LAYOUT_FIELDS = ("fingertip_body_names",)
_TACTILE_GT_V1_FIELDS = frozenset(
    (
        "fingertip_wrench_local",
        "fingertip_normal",
        "fingertip_force_peak",
        "fingertip_impulse_local",
        "contact_fraction",
        "sim_time",
    )
)
# v1 remains readable as its original six-field contract.  v2 is deliberately
# a distinct merge schema: a missing taxel array is unknown data, not zero load,
# so the converter never synthesizes it when v1 and v2 inputs are mixed.
_EXPECTED_FIELDS_BY_SCHEMA = {
    "dexjoco.tactile_gt.v1": _TACTILE_GT_V1_FIELDS,
    "dexjoco.tactile_gt.v2": _TACTILE_GT_V1_FIELDS
    | frozenset(("taxel_force_local",)),
}
_TACTILE_GT_V1_AXES = {
    "fingertip_wrench_local": ("hand", "fingertip", "wrench_channel"),
    "fingertip_normal": ("hand", "fingertip"),
    "fingertip_force_peak": ("hand", "fingertip"),
    "fingertip_impulse_local": ("hand", "fingertip", "xyz"),
    "contact_fraction": ("hand", "fingertip"),
    "sim_time": ("singleton",),
}
_TACTILE_GT_V1_DTYPES = {
    "fingertip_wrench_local": np.dtype(np.float32),
    "fingertip_normal": np.dtype(np.float32),
    "fingertip_force_peak": np.dtype(np.float32),
    "fingertip_impulse_local": np.dtype(np.float32),
    "contact_fraction": np.dtype(np.float32),
    "sim_time": np.dtype(np.float64),
}
_EXPECTED_AXES_BY_SCHEMA = {
    "dexjoco.tactile_gt.v1": _TACTILE_GT_V1_AXES,
    "dexjoco.tactile_gt.v2": {
        **_TACTILE_GT_V1_AXES,
        "taxel_force_local": ("hand", "fingertip", "taxel", "xyz"),
    },
}
_EXPECTED_DTYPES_BY_SCHEMA = {
    "dexjoco.tactile_gt.v1": _TACTILE_GT_V1_DTYPES,
    "dexjoco.tactile_gt.v2": {
        **_TACTILE_GT_V1_DTYPES,
        "taxel_force_local": np.dtype(np.float32),
    },
}
_EXPECTED_AXIS_SIZES_BY_SCHEMA = {
    "dexjoco.tactile_gt.v2": {
        "taxel_force_local": {"fingertip": 4, "taxel": 16, "xyz": 3},
    },
}

# v3 has the same tensor shapes as v2 but different surface semantics.
# Schema identity remains part of merge compatibility; never relabel v2.
for _schemas in (_EXPECTED_FIELDS_BY_SCHEMA, _EXPECTED_AXES_BY_SCHEMA,
                 _EXPECTED_DTYPES_BY_SCHEMA, _EXPECTED_AXIS_SIZES_BY_SCHEMA):
    _schemas["dexjoco.tactile_gt.v3"] = _schemas["dexjoco.tactile_gt.v2"]


@dataclass(frozen=True)
class SensorFieldPlan:
    """Canonical specification for one time-major sensor payload field."""

    name: str
    dtype: np.dtype[Any]
    shape: tuple[int, ...]
    axes: tuple[str, ...]

    @property
    def hand_axis(self) -> int | None:
        try:
            return self.axes.index("hand")
        except ValueError:
            return None


@dataclass(frozen=True)
class SensorGroupPlan:
    """Canonical payload and stream schema for one semantic sensor group."""

    name: str
    stream_prefixes: tuple[str, ...]
    fields: tuple[SensorFieldPlan, ...]
    metadata: dict[str, Any]

    @property
    def field_map(self) -> dict[str, SensorFieldPlan]:
        return {field.name: field for field in self.fields}


@dataclass(frozen=True)
class SensorMergePlan:
    """Schema shared by all episodes written to one merged replay buffer."""

    groups: tuple[SensorGroupPlan, ...]
    target_hand_order: tuple[str, ...] = TARGET_HAND_ORDER

    @property
    def output_keys(self) -> frozenset[str]:
        keys: set[str] = set()
        for group in self.groups:
            for prefix in group.stream_prefixes:
                stream_name = f"{prefix}{group.name}"
                keys.update(
                    f"{stream_name}.{field.name}" for field in group.fields
                )
                keys.add(f"{stream_name}.sensor_present")
                keys.add(f"{stream_name}.hand_valid")
        return frozenset(keys)


@dataclass(frozen=True)
class _StreamRecord:
    zarr_path: Path
    group_name: str
    prefix: str
    fields: tuple[str, ...]
    metadata: dict[str, Any]


def is_sensor_field(key: str) -> bool:
    """Return whether ``key`` belongs to a supported tactile sensor stream."""

    return _belongs_to_sensor_group(key, SENSOR_GROUPS)


def _belongs_to_sensor_group(key: str, group_names: Sequence[str]) -> bool:
    return any(
        key.startswith(f"{prefix}{group_name}.")
        for group_name in group_names
        for prefix in ("", "next_")
    )


def _is_retired_sensor_field(key: str) -> bool:
    """Return whether ``key`` belongs to a recognized retired sensor stream."""

    return _belongs_to_sensor_group(key, _RETIRED_SENSOR_GROUPS)


def discover_sensor_merge_plan(
    zarr_paths: Sequence[Path],
    *,
    target_hand_order: Sequence[str] = TARGET_HAND_ORDER,
) -> SensorMergePlan:
    """Inspect input episode stores and build a canonical optional-sensor schema.

    A sensor stream may be wholly absent from an episode.  Once present, its
    payload field set must be complete and compatible with every other present
    instance of that group, including the corresponding ``next_`` stream.
    """

    paths = [Path(path) for path in zarr_paths]
    if not paths:
        raise ValueError("At least one input Zarr path is required.")

    canonical_hands = tuple(str(name) for name in target_hand_order)
    if not canonical_hands or len(set(canonical_hands)) != len(canonical_hands):
        raise ValueError(
            f"target_hand_order must contain unique labels, got {canonical_hands}."
        )

    records_by_group: dict[str, list[_StreamRecord]] = {
        group_name: [] for group_name in SENSOR_GROUPS
    }
    for zarr_path in paths:
        root = zarr.open_group(str(zarr_path), mode="r")
        if "data" not in root:
            raise ValueError(f"Missing data group in {zarr_path}.")
        data_group = root["data"]
        data_keys = set(data_group.array_keys())
        _validate_sensor_time_lengths(root, zarr_path, data_keys)

        for group_name in SENSOR_GROUPS:
            metadata = _read_group_metadata(root, group_name, zarr_path)
            path_records: list[_StreamRecord] = []
            for prefix in ("", "next_"):
                stream_name = f"{prefix}{group_name}"
                key_prefix = f"{stream_name}."
                fields = tuple(
                    sorted(
                        key[len(key_prefix) :]
                        for key in data_keys
                        if key.startswith(key_prefix)
                        and key[len(key_prefix) :] not in _GENERATED_FIELDS
                    )
                )
                if fields:
                    path_records.append(
                        _StreamRecord(
                            zarr_path=zarr_path,
                            group_name=group_name,
                            prefix=prefix,
                            fields=fields,
                            metadata=metadata,
                        )
                    )
            if len(path_records) == 1:
                present_stream = f"{path_records[0].prefix}{group_name}"
                missing_stream = (
                    f"next_{group_name}"
                    if path_records[0].prefix == ""
                    else group_name
                )
                raise ValueError(
                    f"Incomplete sensor transition in {zarr_path}: "
                    f"{present_stream!r} is present but {missing_stream!r} is "
                    "missing. Current and next sensor streams must be paired."
                )
            records_by_group[group_name].extend(path_records)

    group_plans: list[SensorGroupPlan] = []
    for group_name in SENSOR_GROUPS:
        records = records_by_group[group_name]
        if not records:
            continue
        group_plans.append(
            _build_group_plan(group_name, records, canonical_hands)
        )

    return SensorMergePlan(
        groups=tuple(group_plans), target_hand_order=canonical_hands
    )


def materialize_sensor_fields(
    episode_data: dict[str, np.ndarray],
    *,
    zarr_path: Path,
    plan: SensorMergePlan,
) -> None:
    """Replace raw optional sensor arrays with the canonical padded schema.

    ``episode_data`` is modified in place.  Raw sensor arrays are removed first
    so unselected/generated source masks cannot leak into the merged output.
    """

    raw_sensor_data = {
        key: np.asarray(episode_data.pop(key))
        for key in list(episode_data)
        if is_sensor_field(key) or _is_retired_sensor_field(key)
    }
    retired_sensor_keys = sorted(
        key for key in raw_sensor_data if _is_retired_sensor_field(key)
    )
    if retired_sensor_keys:
        warnings.warn(
            f"Dropping retired tactile fields from {zarr_path}: "
            f"{retired_sensor_keys}. The source Zarr is not modified.",
            RuntimeWarning,
            stacklevel=2,
        )
        for key in retired_sensor_keys:
            raw_sensor_data.pop(key)
    if episode_data:
        lengths = {int(value.shape[0]) for value in episode_data.values()}
        if len(lengths) != 1:
            raise ValueError(
                f"Non-sensor arrays in {zarr_path} have inconsistent lengths: "
                f"{sorted(lengths)}."
            )
        n_steps = lengths.pop()
    elif raw_sensor_data:
        lengths = {int(value.shape[0]) for value in raw_sensor_data.values()}
        if len(lengths) != 1:
            raise ValueError(
                f"Sensor arrays in {zarr_path} have inconsistent lengths: "
                f"{sorted(lengths)}."
            )
        n_steps = lengths.pop()
    else:
        raise ValueError(f"Episode {zarr_path} contains no arrays.")

    root = zarr.open_group(str(zarr_path), mode="r")
    for group_plan in plan.groups:
        metadata = _read_group_metadata(root, group_plan.name, zarr_path)
        for prefix in group_plan.stream_prefixes:
            stream_name = f"{prefix}{group_plan.name}"
            source_fields = {
                key[len(stream_name) + 1 :]: value
                for key, value in raw_sensor_data.items()
                if key.startswith(f"{stream_name}.")
                and key[len(stream_name) + 1 :] not in _GENERATED_FIELDS
            }
            _materialize_stream(
                episode_data=episode_data,
                source_fields=source_fields,
                n_steps=n_steps,
                stream_name=stream_name,
                group_plan=group_plan,
                metadata=metadata,
                target_hand_order=plan.target_hand_order,
                zarr_path=zarr_path,
            )

    unexpected_sensor_keys = sorted(
        key for key in raw_sensor_data if not _key_is_planned(key, plan)
    )
    if unexpected_sensor_keys:
        raise ValueError(
            f"Episode {zarr_path} contains sensor fields outside the discovered "
            f"canonical plan: {unexpected_sensor_keys}."
        )


def write_sensor_metadata(root: zarr.Group, plan: SensorMergePlan) -> None:
    """Write canonical sensor metadata to a merged Zarr root."""

    for group in plan.groups:
        root.attrs[group.name] = copy.deepcopy(group.metadata)


def validate_episode_for_append(
    episode_data: Mapping[str, np.ndarray],
    *,
    expected_schema: Mapping[
        str, tuple[np.dtype[Any], tuple[int, ...]]
    ]
    | None = None,
) -> dict[str, tuple[np.dtype[Any], tuple[int, ...]]]:
    """Validate and describe a complete episode before ReplayBuffer append."""

    if not episode_data:
        raise ValueError("episode_data must not be empty.")
    keys = frozenset(episode_data)
    if expected_schema is not None and keys != frozenset(expected_schema):
        missing = sorted(set(expected_schema) - set(keys))
        extra = sorted(set(keys) - set(expected_schema))
        raise ValueError(
            f"Canonical episode key mismatch: missing={missing}, extra={extra}."
        )

    lengths: dict[str, int] = {}
    schema: dict[str, tuple[np.dtype[Any], tuple[int, ...]]] = {}
    for key, value in episode_data.items():
        array = np.asarray(value)
        if array.ndim < 1:
            raise ValueError(f"Episode field {key!r} has no time axis.")
        lengths[key] = int(array.shape[0])
        schema[key] = (array.dtype, tuple(int(dim) for dim in array.shape[1:]))
    unique_lengths = set(lengths.values())
    if len(unique_lengths) != 1:
        raise ValueError(f"Episode fields have inconsistent lengths: {lengths}.")
    if next(iter(unique_lengths)) <= 0:
        raise ValueError("Cannot append an empty episode.")
    if expected_schema is not None:
        mismatches = {
            key: {"expected": expected_schema[key], "actual": schema[key]}
            for key in sorted(schema)
            if schema[key] != expected_schema[key]
        }
        if mismatches:
            raise ValueError(
                "Canonical episode dtype/shape mismatch before append: "
                f"{mismatches}."
            )
    return schema


def validate_materialized_sensor_fields(
    episode_data: Mapping[str, np.ndarray], plan: SensorMergePlan
) -> None:
    """Assert that post-processing did not alter the canonical sensor schema."""

    for group in plan.groups:
        for prefix in group.stream_prefixes:
            stream_name = f"{prefix}{group.name}"
            for field in group.fields:
                key = f"{stream_name}.{field.name}"
                if key not in episode_data:
                    raise ValueError(f"Missing canonical sensor field {key!r}.")
                array = np.asarray(episode_data[key])
                if array.dtype != field.dtype:
                    raise ValueError(
                        f"Canonical sensor field {key!r} changed dtype: "
                        f"{array.dtype} != {field.dtype}."
                    )
                if tuple(array.shape[1:]) != field.shape:
                    raise ValueError(
                        f"Canonical sensor field {key!r} changed per-frame shape: "
                        f"{array.shape[1:]} != {field.shape}."
                    )

            mask_specs = {
                "sensor_present": (1,),
                "hand_valid": (len(plan.target_hand_order),),
            }
            for mask_name, expected_shape in mask_specs.items():
                key = f"{stream_name}.{mask_name}"
                if key not in episode_data:
                    raise ValueError(f"Missing canonical sensor mask {key!r}.")
                array = np.asarray(episode_data[key])
                if array.dtype != np.dtype(np.bool_):
                    raise ValueError(
                        f"Canonical sensor mask {key!r} changed dtype: "
                        f"{array.dtype} != bool."
                    )
                if tuple(array.shape[1:]) != expected_shape:
                    raise ValueError(
                        f"Canonical sensor mask {key!r} changed per-frame shape: "
                        f"{array.shape[1:]} != {expected_shape}."
                    )


def validate_replay_buffer_lengths(buffer: Any) -> None:
    """Assert that every ReplayBuffer array reaches its final episode end."""

    expected_length = int(buffer.n_steps)
    bad = {
        key: int(value.shape[0])
        for key, value in buffer.data.items()
        if int(value.shape[0]) != expected_length
    }
    if bad:
        raise RuntimeError(
            "Merged replay buffer contains arrays shorter than episode_ends: "
            f"expected={expected_length}, got={bad}."
        )


def _build_group_plan(
    group_name: str,
    records: Sequence[_StreamRecord],
    target_hand_order: tuple[str, ...],
) -> SensorGroupPlan:
    versions_by_path = {
        record.zarr_path: str(record.metadata.get("schema_version"))
        for record in records
    }
    schema_versions = set(versions_by_path.values())
    if len(schema_versions) != 1:
        formatted_versions = {
            str(path): version
            for path, version in sorted(
                versions_by_path.items(), key=lambda item: str(item[0])
            )
        }
        raise ValueError(
            f"Incompatible {group_name} schema versions in one merge: "
            f"{formatted_versions}. Different schema versions must be "
            "converted separately; missing v1 taxel_force_local values are "
            "not synthesized or zero-filled."
        )

    expected_fields = records[0].fields
    schema_version = next(iter(schema_versions))
    schema_fields = _EXPECTED_FIELDS_BY_SCHEMA.get(str(schema_version))
    if schema_fields is not None:
        for record in records:
            if set(record.fields) == schema_fields:
                continue
            missing = sorted(schema_fields - set(record.fields))
            extra = sorted(set(record.fields) - schema_fields)
            raise ValueError(
                f"{record.prefix}{group_name} stream in {record.zarr_path} "
                f"does not satisfy schema {schema_version}: "
                f"missing={missing}, extra={extra}."
            )
    for record in records[1:]:
        if record.fields != expected_fields:
            missing = sorted(set(expected_fields) - set(record.fields))
            extra = sorted(set(record.fields) - set(expected_fields))
            raise ValueError(
                f"Partial {group_name} stream in {record.zarr_path}: "
                f"missing={missing}, extra={extra}."
            )

    metadata_by_path: dict[Path, dict[str, Any]] = {}
    for record in records:
        metadata_by_path.setdefault(record.zarr_path, record.metadata)
    first_metadata = records[0].metadata
    first_signature = _metadata_signature(first_metadata)
    for zarr_path, metadata in metadata_by_path.items():
        if _metadata_signature(metadata) != first_signature:
            raise ValueError(
                f"Incompatible {group_name} metadata in {zarr_path}. "
                "Only hand_order and normalized hand-layout metadata may "
                "vary across a canonical merge."
            )

    field_plans: list[SensorFieldPlan] = []
    source_hand_orders: set[tuple[str, ...]] = set()
    for field_name in expected_fields:
        reference_dtype: np.dtype | None = None
        reference_axes: tuple[str, ...] | None = None
        reference_non_hand_shape: tuple[int, ...] | None = None
        reference_shape: tuple[int, ...] | None = None

        for record in records:
            root = zarr.open_group(str(record.zarr_path), mode="r")
            array = root["data"][
                f"{record.prefix}{group_name}.{field_name}"
            ]
            source_shape = tuple(int(dim) for dim in array.shape[1:])
            source_dtype = np.dtype(array.dtype)
            expected_dtype = _EXPECTED_DTYPES_BY_SCHEMA.get(
                str(schema_version), {}
            ).get(field_name)
            if expected_dtype is not None and source_dtype != expected_dtype:
                raise ValueError(
                    f"{group_name}.{field_name} dtype {source_dtype} does not "
                    f"satisfy schema {schema_version}; expected "
                    f"{expected_dtype} in {record.zarr_path}."
                )
            axes = _resolve_field_axes(
                group_name,
                field_name,
                source_shape,
                record.metadata,
                record.zarr_path,
            )
            expected_axes = _EXPECTED_AXES_BY_SCHEMA.get(
                str(schema_version), {}
            ).get(field_name)
            if expected_axes is not None and axes != expected_axes:
                raise ValueError(
                    f"{group_name}.{field_name} axes {axes} do not satisfy "
                    f"schema {schema_version}; expected {expected_axes} in "
                    f"{record.zarr_path}."
                )
            expected_axis_sizes = _EXPECTED_AXIS_SIZES_BY_SCHEMA.get(
                str(schema_version), {}
            ).get(field_name, {})
            for axis_name, expected_size in expected_axis_sizes.items():
                actual_size = source_shape[axes.index(axis_name)]
                if actual_size != expected_size:
                    raise ValueError(
                        f"{group_name}.{field_name} axis {axis_name!r} has "
                        f"size {actual_size}, but schema {schema_version} "
                        f"requires {expected_size} in {record.zarr_path}."
                    )
            _validate_declared_field_metadata(
                field_name,
                source_shape,
                source_dtype,
                axes,
                record.metadata,
                record.zarr_path,
            )

            hand_axis = axes.index("hand") if "hand" in axes else None
            if hand_axis is None:
                non_hand_shape = source_shape
            else:
                hand_count = source_shape[hand_axis]
                hand_order = _source_hand_order(
                    record.metadata,
                    hand_count,
                    target_hand_order,
                    record.zarr_path,
                )
                source_hand_orders.add(hand_order)
                non_hand_shape = (
                    source_shape[:hand_axis] + source_shape[hand_axis + 1 :]
                )

            if reference_dtype is None:
                reference_dtype = source_dtype
                reference_axes = axes
                reference_non_hand_shape = non_hand_shape
                reference_shape = source_shape
            else:
                if source_dtype != reference_dtype:
                    raise ValueError(
                        f"{group_name}.{field_name} dtype differs in "
                        f"{record.zarr_path}: {source_dtype} != {reference_dtype}."
                    )
                if axes != reference_axes:
                    raise ValueError(
                        f"{group_name}.{field_name} axes differ in "
                        f"{record.zarr_path}: {axes} != {reference_axes}."
                    )
                if non_hand_shape != reference_non_hand_shape:
                    raise ValueError(
                        f"{group_name}.{field_name} non-hand shape differs in "
                        f"{record.zarr_path}: {non_hand_shape} != "
                        f"{reference_non_hand_shape}."
                    )

        assert reference_dtype is not None
        assert reference_axes is not None
        assert reference_shape is not None
        canonical_shape = list(reference_shape)
        if "hand" in reference_axes:
            canonical_shape[reference_axes.index("hand")] = len(target_hand_order)
        field_plans.append(
            SensorFieldPlan(
                name=field_name,
                dtype=reference_dtype,
                shape=tuple(canonical_shape),
                axes=reference_axes,
            )
        )

    streams = tuple(
        prefix for prefix in ("", "next_") if any(r.prefix == prefix for r in records)
    )
    hand_layouts = _collect_hand_layouts(
        group_name=group_name,
        records=records,
        target_hand_order=target_hand_order,
    )
    metadata = _canonical_group_metadata(
        group_name=group_name,
        source_metadata=first_metadata,
        fields=field_plans,
        streams=streams,
        target_hand_order=target_hand_order,
        source_hand_orders=source_hand_orders,
        hand_layouts=hand_layouts,
    )
    return SensorGroupPlan(
        name=group_name,
        stream_prefixes=streams,
        fields=tuple(field_plans),
        metadata=metadata,
    )


def _collect_hand_layouts(
    *,
    group_name: str,
    records: Sequence[_StreamRecord],
    target_hand_order: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    """Normalize flat producer geometry into explicit per-hand metadata."""

    layouts: dict[str, dict[str, Any]] = {}
    records_by_path: dict[Path, _StreamRecord] = {}
    for record in records:
        records_by_path.setdefault(record.zarr_path, record)

    for record in records_by_path.values():
        normalized_layouts = record.metadata.get("hand_layouts")
        if isinstance(normalized_layouts, Mapping):
            for hand_name, details in normalized_layouts.items():
                hand_name = str(hand_name)
                if hand_name not in target_hand_order:
                    raise ValueError(
                        f"hand_layouts in {record.zarr_path} contains unknown "
                        f"hand {hand_name!r}."
                    )
                if not isinstance(details, Mapping):
                    raise ValueError(
                        f"hand_layouts[{hand_name!r}] in {record.zarr_path} "
                        "must be a mapping."
                    )
                _merge_hand_layout(
                    layouts,
                    hand_name,
                    copy.deepcopy(dict(details)),
                    record.zarr_path,
                )

        if not any(name in record.metadata for name in _HAND_LAYOUT_FIELDS):
            continue

        root = zarr.open_group(str(record.zarr_path), mode="r")
        layout_shape: tuple[int, int] | None = None
        for field_name in record.fields:
            array = root["data"][
                f"{record.prefix}{group_name}.{field_name}"
            ]
            source_shape = tuple(int(dim) for dim in array.shape[1:])
            axes = _resolve_field_axes(
                group_name,
                field_name,
                source_shape,
                record.metadata,
                record.zarr_path,
            )
            if "hand" in axes and "fingertip" in axes:
                layout_shape = (
                    source_shape[axes.index("hand")],
                    source_shape[axes.index("fingertip")],
                )
                break
        if layout_shape is None:
            raise ValueError(
                f"Cannot associate hand-indexed layout metadata in "
                f"{record.zarr_path} with a hand/fingertip sensor field."
            )

        hand_count, fingertip_count = layout_shape
        source_hand_order = _source_hand_order(
            record.metadata,
            hand_count,
            target_hand_order,
            record.zarr_path,
        )
        per_hand: dict[str, dict[str, Any]] = {
            hand_name: {"fingertip_count": fingertip_count}
            for hand_name in source_hand_order
        }
        flat_fingertip_count = hand_count * fingertip_count
        for field_name in _HAND_LAYOUT_FIELDS:
            if field_name not in record.metadata:
                continue
            values = np.asarray(record.metadata[field_name])
            if values.ndim < 1 or values.shape[0] != flat_fingertip_count:
                raise ValueError(
                    f"Root attr {field_name!r} in {record.zarr_path} must have "
                    f"{flat_fingertip_count} leading fingertip entries, got "
                    f"shape {values.shape}."
                )
            hand_values = values.reshape(
                hand_count,
                fingertip_count,
                *values.shape[1:],
            )
            for hand_index, hand_name in enumerate(source_hand_order):
                hand_value = hand_values[hand_index].tolist()
                if field_name == "fingertip_body_names":
                    per_hand[hand_name]["fingertip_body_name_variants"] = [
                        hand_value
                    ]
                else:
                    per_hand[hand_name][field_name] = hand_value

        for hand_name, details in per_hand.items():
            _merge_hand_layout(
                layouts,
                hand_name,
                details,
                record.zarr_path,
            )

    return {
        hand_name: layouts[hand_name]
        for hand_name in target_hand_order
        if hand_name in layouts
    }


def _merge_hand_layout(
    layouts: dict[str, dict[str, Any]],
    hand_name: str,
    candidate: Mapping[str, Any],
    zarr_path: Path,
) -> None:
    existing = layouts.setdefault(hand_name, {})
    for key, value in candidate.items():
        if key == "fingertip_body_names":
            key = "fingertip_body_name_variants"
            value = [value]
        if key == "fingertip_body_name_variants":
            variants = existing.setdefault(key, [])
            for variant in value:
                if variant not in variants:
                    variants.append(copy.deepcopy(variant))
            variants.sort(
                key=lambda variant: json.dumps(
                    variant, sort_keys=True, separators=(",", ":")
                )
            )
            continue
        if key in existing and existing[key] != value:
            raise ValueError(
                f"Incompatible {hand_name!r} hand layout field {key!r} in "
                f"{zarr_path}."
            )
        existing[key] = copy.deepcopy(value)


def _materialize_stream(
    *,
    episode_data: dict[str, np.ndarray],
    source_fields: Mapping[str, np.ndarray],
    n_steps: int,
    stream_name: str,
    group_plan: SensorGroupPlan,
    metadata: Mapping[str, Any],
    target_hand_order: tuple[str, ...],
    zarr_path: Path,
) -> None:
    expected_fields = set(group_plan.field_map)
    actual_fields = set(source_fields)
    present = bool(actual_fields)
    if present and actual_fields != expected_fields:
        missing = sorted(expected_fields - actual_fields)
        extra = sorted(actual_fields - expected_fields)
        raise ValueError(
            f"Partial {stream_name} stream in {zarr_path}: "
            f"missing={missing}, extra={extra}."
        )

    source_hand_order: tuple[str, ...] = ()
    if present:
        hand_counts = {
            tuple(np.asarray(source_fields[field.name]).shape[1:])[field.hand_axis]
            for field in group_plan.fields
            if field.hand_axis is not None
        }
        if len(hand_counts) > 1:
            raise ValueError(
                f"Hand-bearing fields in {stream_name} disagree on hand count: "
                f"{sorted(hand_counts)}."
            )
        if hand_counts:
            source_hand_order = _source_hand_order(
                metadata,
                hand_counts.pop(),
                target_hand_order,
                zarr_path,
            )

    for field in group_plan.fields:
        output_key = f"{stream_name}.{field.name}"
        target = np.zeros((n_steps, *field.shape), dtype=field.dtype)
        if present:
            source = np.asarray(source_fields[field.name])
            if source.shape[0] != n_steps:
                raise ValueError(
                    f"{output_key} length {source.shape[0]} != episode length "
                    f"{n_steps} in {zarr_path}."
                )
            if source.dtype != field.dtype:
                raise ValueError(
                    f"{output_key} dtype {source.dtype} != canonical "
                    f"{field.dtype} in {zarr_path}."
                )
            if field.hand_axis is None:
                if tuple(source.shape[1:]) != field.shape:
                    raise ValueError(
                        f"{output_key} shape {source.shape[1:]} != canonical "
                        f"{field.shape} in {zarr_path}."
                    )
                target[...] = source
            else:
                source_non_hand = (
                    tuple(source.shape[1 : 1 + field.hand_axis])
                    + tuple(source.shape[2 + field.hand_axis :])
                )
                target_non_hand = (
                    field.shape[: field.hand_axis]
                    + field.shape[field.hand_axis + 1 :]
                )
                if source_non_hand != target_non_hand:
                    raise ValueError(
                        f"{output_key} non-hand shape {source_non_hand} != "
                        f"canonical {target_non_hand} in {zarr_path}."
                    )
                for source_index, hand_name in enumerate(source_hand_order):
                    target_index = target_hand_order.index(hand_name)
                    source_slice: list[Any] = [slice(None)] * source.ndim
                    target_slice: list[Any] = [slice(None)] * target.ndim
                    source_slice[field.hand_axis + 1] = source_index
                    target_slice[field.hand_axis + 1] = target_index
                    target[tuple(target_slice)] = source[tuple(source_slice)]
        episode_data[output_key] = target

    sensor_present = np.full((n_steps, 1), present, dtype=np.bool_)
    hand_valid = np.zeros((n_steps, len(target_hand_order)), dtype=np.bool_)
    if present:
        if source_hand_order:
            for hand_name in source_hand_order:
                hand_valid[:, target_hand_order.index(hand_name)] = True
        else:
            # A structurally present group with no hand-bearing payload is still
            # represented by sensor_present; no canonical hand slot is claimed.
            hand_valid.fill(False)
    episode_data[f"{stream_name}.sensor_present"] = sensor_present
    episode_data[f"{stream_name}.hand_valid"] = hand_valid


def _read_group_metadata(
    root: zarr.Group, group_name: str, zarr_path: Path
) -> dict[str, Any]:
    if group_name in root.attrs:
        metadata = root.attrs[group_name]
        if not isinstance(metadata, Mapping):
            raise ValueError(
                f"Root attribute {group_name!r} in {zarr_path} must be a mapping."
            )
        return copy.deepcopy(dict(metadata))

    data_group = root.get("data")
    has_group = data_group is not None and any(
        key.startswith(f"{prefix}{group_name}.")
        for prefix in ("", "next_")
        for key in data_group.array_keys()
    )
    if has_group:
        warnings.warn(
            f"{zarr_path} contains {group_name} arrays without root metadata; "
            "using the converter's built-in schema fallback.",
            RuntimeWarning,
            stacklevel=2,
        )
        return _default_group_metadata(group_name)
    return {}


def _default_group_metadata(group_name: str) -> dict[str, Any]:
    if group_name == "tactile_gt":
        return {
            "schema_version": "dexjoco.tactile_gt.v1",
            "frame": "per-sample fingertip body-local frame at the body origin",
            "fingertip_order": ["index", "middle", "ring", "thumb"],
            "wrench_channels": ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"],
            "field_units": {
                "fingertip_wrench_local": [
                    "N",
                    "N",
                    "N",
                    "N*m",
                    "N*m",
                    "N*m",
                ],
                "fingertip_normal": "N",
                "fingertip_force_peak": "N",
                "fingertip_impulse_local": "N*s",
                "contact_fraction": "1",
                "sim_time": "s",
            },
            "metadata_status": "inferred by dexjoco-data-converter",
        }
    raise ValueError(f"Unsupported sensor group {group_name!r}.")


def _field_metadata(metadata: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return producer field metadata, accepting the legacy key on read."""

    for key in ("field_specs", "fields"):
        value = metadata.get(key)
        if isinstance(value, Mapping):
            return value
    return {}


def _resolve_field_axes(
    group_name: str,
    field_name: str,
    source_shape: tuple[int, ...],
    metadata: Mapping[str, Any],
    zarr_path: Path,
) -> tuple[str, ...]:
    fields_metadata = _field_metadata(metadata)
    if isinstance(fields_metadata, Mapping):
        field_metadata = fields_metadata.get(field_name, {})
        if isinstance(field_metadata, Mapping) and "axes" in field_metadata:
            axes = tuple(str(axis) for axis in field_metadata["axes"])
            if axes and axes[0] == "time":
                axes = axes[1:]
            if len(axes) != len(source_shape):
                raise ValueError(
                    f"Declared axes {axes} for {group_name}.{field_name} do not "
                    f"match per-frame shape {source_shape} in {zarr_path}."
                )
            if len(set(axes)) != len(axes):
                raise ValueError(
                    f"Axes for {group_name}.{field_name} must be unique, got {axes}."
                )
            return axes

    axes = _builtin_field_axes(group_name, field_name, len(source_shape))
    if axes is None:
        raise ValueError(
            f"No axis metadata is available for {group_name}.{field_name} in "
            f"{zarr_path}; add root attrs[{group_name!r}]['field_specs']"
            f"[{field_name!r}]['axes'] before merging."
        )
    return axes


def _builtin_field_axes(
    group_name: str, field_name: str, ndim: int
) -> tuple[str, ...] | None:
    tactile_gt_axes = {
        "fingertip_wrench_local": ("hand", "fingertip", "wrench_channel"),
        "taxel_force_local": ("hand", "fingertip", "taxel", "xyz"),
        "fingertip_normal": ("hand", "fingertip"),
        "fingertip_force_peak": ("hand", "fingertip"),
        "fingertip_impulse_local": ("hand", "fingertip", "xyz"),
        "contact_fraction": ("hand", "fingertip"),
        "sim_time": ("singleton",),
    }
    if group_name == "tactile_gt":
        axes = tactile_gt_axes.get(field_name)
        return axes if axes is not None and len(axes) == ndim else None

    return None


def _validate_declared_field_metadata(
    field_name: str,
    source_shape: tuple[int, ...],
    source_dtype: np.dtype,
    axes: tuple[str, ...],
    metadata: Mapping[str, Any],
    zarr_path: Path,
) -> None:
    fields_metadata = _field_metadata(metadata)
    if not isinstance(fields_metadata, Mapping):
        return
    declared = fields_metadata.get(field_name)
    if not isinstance(declared, Mapping):
        return
    if "dtype" in declared and np.dtype(declared["dtype"]) != source_dtype:
        raise ValueError(
            f"Declared dtype {declared['dtype']} for {field_name} does not match "
            f"{source_dtype} in {zarr_path}."
        )
    if "shape" in declared:
        declared_shape = tuple(int(dim) for dim in declared["shape"])
        if declared_shape != source_shape:
            raise ValueError(
                f"Declared shape {declared_shape} for {field_name} does not match "
                f"{source_shape} in {zarr_path}."
            )
    if axes.count("hand") > 1:
        raise ValueError(f"Field {field_name} declares more than one hand axis.")


def _source_hand_order(
    metadata: Mapping[str, Any],
    hand_count: int,
    target_hand_order: tuple[str, ...],
    zarr_path: Path,
) -> tuple[str, ...]:
    configured = metadata.get("hand_order")
    if configured is None:
        if hand_count == 1:
            order = ("right",)
        elif hand_count == len(target_hand_order):
            order = target_hand_order
        else:
            raise ValueError(
                f"Cannot infer {hand_count} source hands in {zarr_path}; add "
                "a hand_order root attribute."
            )
    else:
        order = tuple(str(name) for name in configured)

    if len(order) != hand_count:
        raise ValueError(
            f"hand_order {order} has length {len(order)}, but sensor arrays in "
            f"{zarr_path} have {hand_count} hands."
        )
    if len(set(order)) != len(order):
        raise ValueError(
            f"hand_order contains duplicate labels in {zarr_path}: {order}."
        )
    unknown = sorted(set(order) - set(target_hand_order))
    if unknown:
        raise ValueError(
            f"hand_order in {zarr_path} contains labels outside canonical "
            f"{target_hand_order}: {unknown}."
        )
    return order


def _canonical_group_metadata(
    *,
    group_name: str,
    source_metadata: Mapping[str, Any],
    fields: Sequence[SensorFieldPlan],
    streams: tuple[str, ...],
    target_hand_order: tuple[str, ...],
    source_hand_orders: set[tuple[str, ...]],
    hand_layouts: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    metadata = copy.deepcopy(dict(source_metadata))
    source_field_metadata = _field_metadata(metadata)
    metadata.pop("fields", None)
    metadata.pop("hand_layouts", None)
    for field_name in _HAND_LAYOUT_FIELDS:
        metadata.pop(field_name, None)

    canonical_fields: dict[str, Any] = {}
    for field in fields:
        details = copy.deepcopy(dict(source_field_metadata.get(field.name, {})))
        details.update(
            {
                "dtype": _dtype_metadata_name(field.dtype),
                "shape": list(field.shape),
                "axes": ["time", *field.axes],
            }
        )
        canonical_fields[field.name] = details
    canonical_fields["sensor_present"] = {
        "dtype": "bool",
        "shape": [1],
        "axes": ["time", "singleton"],
        "meaning": "this source episode contains the sensor stream",
    }
    canonical_fields["hand_valid"] = {
        "dtype": "bool",
        "shape": [len(target_hand_order)],
        "axes": ["time", "hand"],
        "meaning": "the canonical hand slot is backed by source sensor data",
    }

    metadata.update(
        {
            "storage_schema_version": SENSOR_PADDING_SCHEMA_VERSION,
            "hand_order": list(target_hand_order),
            "source_hand_orders": [
                list(order) for order in sorted(source_hand_orders)
            ],
            "streams": [f"{prefix}{group_name}" for prefix in streams],
            "field_specs": canonical_fields,
            "hand_layouts": copy.deepcopy(dict(hand_layouts)),
            "padding": {
                "fill_value": 0,
                "sensor_present_field": "sensor_present",
                "hand_valid_field": "hand_valid",
                "target_hand_order": list(target_hand_order),
            },
        }
    )
    return metadata


def _dtype_metadata_name(dtype: np.dtype[Any]) -> str:
    """Use readable native names while retaining non-native byte order."""

    return dtype.name if dtype.isnative else dtype.str


def _metadata_signature(metadata: Mapping[str, Any]) -> str:
    normalized = copy.deepcopy(dict(metadata))
    for key in (
        "hand_order",
        "hand_layouts",
        "source_hand_orders",
        "streams",
        "padding",
        "storage_schema_version",
    ):
        normalized.pop(key, None)
    for field_name in _HAND_LAYOUT_FIELDS:
        normalized.pop(field_name, None)
    fields = _field_metadata(normalized)
    normalized.pop("field_specs", None)
    normalized.pop("fields", None)
    if fields:
        normalized_fields: dict[str, Any] = {}
        for name, details in fields.items():
            if isinstance(details, Mapping):
                details = dict(details)
                details.pop("shape", None)
                axes = details.get("axes")
                if (
                    isinstance(axes, Sequence)
                    and not isinstance(axes, (str, bytes))
                    and axes
                    and axes[0] == "time"
                ):
                    details["axes"] = list(axes[1:])
            normalized_fields[str(name)] = details
        normalized["field_specs"] = normalized_fields
    return json.dumps(normalized, sort_keys=True, separators=(",", ":"))


def _validate_sensor_time_lengths(
    root: zarr.Group, zarr_path: Path, data_keys: set[str]
) -> None:
    if "meta" not in root or "episode_ends" not in root["meta"]:
        raise ValueError(f"Missing meta/episode_ends in {zarr_path}.")
    episode_ends = np.asarray(root["meta"]["episode_ends"][:])
    if episode_ends.size != 1:
        raise ValueError(
            f"Expected one raw episode in {zarr_path}, got {episode_ends.size}."
        )
    expected_length = int(episode_ends[-1])
    bad = {
        key: int(root["data"][key].shape[0])
        for key in data_keys
        if is_sensor_field(key)
        and int(root["data"][key].shape[0]) != expected_length
    }
    if bad:
        raise ValueError(
            f"Sensor arrays in {zarr_path} do not match episode length "
            f"{expected_length}: {bad}."
        )


def _key_is_planned(key: str, plan: SensorMergePlan) -> bool:
    if not is_sensor_field(key):
        return True
    if key in plan.output_keys:
        return True
    return any(
        key in {
            f"{prefix}{group.name}.sensor_present",
            f"{prefix}{group.name}.hand_valid",
        }
        for group in plan.groups
        for prefix in group.stream_prefixes
    )
