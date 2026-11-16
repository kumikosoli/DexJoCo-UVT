from __future__ import annotations

import sys
import tempfile
import types
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np
import zarr

from dexjoco_data_converter.sensor_fields import (
    discover_sensor_merge_plan,
    materialize_sensor_fields,
    write_sensor_metadata,
)


def _tactile_gt_metadata(hand_order: list[str]) -> dict[str, Any]:
    return {
        "schema_version": "dexjoco.tactile_gt.v1",
        "frame": "per-sample fingertip body-local frame at the body origin",
        "hand_order": hand_order,
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
        "reduction": {
            "fingertip_wrench_local": "mean over all physics substeps",
            "fingertip_normal": "mean summed normal load",
            "fingertip_force_peak": "max summed normal load in one substep",
            "fingertip_impulse_local": "sum of local force times physics_dt",
            "contact_fraction": "fraction of substeps above contact threshold",
        },
        "alignment": {
            "tactile_gt.*": "pre-action observation at row t",
            "next_tactile_gt.*": "post-action response to action at row t",
        },
        "field_specs": {
            "fingertip_wrench_local": {
                "axes": ["time", "hand", "fingertip", "wrench_channel"],
                "dtype": "float32",
            },
            "fingertip_normal": {
                "axes": ["time", "hand", "fingertip"],
                "dtype": "float32",
            },
            "fingertip_force_peak": {
                "axes": ["time", "hand", "fingertip"],
                "dtype": "float32",
            },
            "fingertip_impulse_local": {
                "axes": ["time", "hand", "fingertip", "xyz"],
                "dtype": "float32",
            },
            "contact_fraction": {
                "axes": ["time", "hand", "fingertip"],
                "dtype": "float32",
            },
            "sim_time": {
                "axes": ["time", "singleton"],
                "dtype": "float64",
            },
        },
    }


def _tactile_gt_v2_metadata(hand_order: list[str]) -> dict[str, Any]:
    metadata = _tactile_gt_metadata(hand_order)
    metadata["schema_version"] = "dexjoco.tactile_gt.v2"
    metadata["taxel_channels"] = [
        "tangent_u",
        "tangent_v",
        "normal_inward",
    ]
    metadata["taxels_per_fingertip"] = 16
    metadata["taxel_grid_shape"] = [4, 4]
    metadata["taxel_frame"] = "per-taxel local frame"
    metadata["taxel_projection"] = {
        "method": "deterministic normalized Gaussian projection",
        "source": "MuJoCo contact positions and local contact forces",
        "normalization": "per-contact taxel weights sum to one",
        "sigma": (
            "per-fingertip median nearest-neighbor taxel-center spacing when "
            "unspecified"
        ),
        "support": "all 16 taxels on the fingertip",
        "layout": "4 axial rows x 4 circumferential columns",
        "layout_source": (
            "taxel centers and local frames are deterministically derived from "
            "each fingertip collision capsule, including the MuJoCo geom pose"
        ),
        "aggregation": (
            "sum projected contact events per physics substep, then arithmetic "
            "mean over control-step substeps"
        ),
        "conservation": (
            "projected force sums to the source force after each taxel vector "
            "is rotated back to the fingertip body frame"
        ),
    }
    metadata["field_units"]["taxel_force_local"] = ["N", "N", "N"]
    metadata["reduction"]["taxel_force_local"] = (
        "mean over all physics substeps after deterministic Gaussian projection"
    )
    metadata["field_specs"]["taxel_force_local"] = {
        "axes": ["time", "hand", "fingertip", "taxel", "xyz"],
        "dtype": "float32",
    }
    return metadata


def _make_tactile_gt(
    steps: int,
    hands: int,
    *,
    offset: float = 0.0,
    all_zero: bool = False,
) -> dict[str, np.ndarray]:
    def encoded(shape: tuple[int, ...], extra: float = 0.0) -> np.ndarray:
        if all_zero:
            return np.zeros(shape, dtype=np.float32)
        values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        return values + np.float32(offset + extra)

    current = {
        "fingertip_wrench_local": encoded((steps, hands, 4, 6)),
        "fingertip_normal": encoded((steps, hands, 4), 100.0),
        "fingertip_force_peak": encoded((steps, hands, 4), 200.0),
        "fingertip_impulse_local": encoded((steps, hands, 4, 3), 300.0),
        "contact_fraction": encoded((steps, hands, 4), 400.0),
        "sim_time": np.arange(steps, dtype=np.float64)[:, None] * 0.02,
    }
    arrays: dict[str, np.ndarray] = {}
    for prefix, extra in (("", 0.0), ("next_", 1000.0)):
        for name, value in current.items():
            if name == "sim_time":
                shifted = value + (0.02 if prefix else 0.0)
            else:
                shifted = value + np.float32(extra)
            arrays[f"{prefix}tactile_gt.{name}"] = shifted
    return arrays


def _make_tactile_gt_v2(
    steps: int,
    hands: int,
    *,
    offset: float = 0.0,
    all_zero: bool = False,
) -> dict[str, np.ndarray]:
    arrays = _make_tactile_gt(
        steps,
        hands,
        offset=offset,
        all_zero=all_zero,
    )
    shape = (steps, hands, 4, 16, 3)
    if all_zero:
        current = np.zeros(shape, dtype=np.float32)
    else:
        current = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
        current += np.float32(offset + 500.0)
    arrays["tactile_gt.taxel_force_local"] = current
    arrays["next_tactile_gt.taxel_force_local"] = current + np.float32(1000.0)
    return arrays


def _write_episode(
    episode_dir: Path,
    arrays: dict[str, np.ndarray],
    *,
    attrs: dict[str, Any] | None = None,
) -> Path:
    episode_dir.mkdir(parents=True)
    (episode_dir / "videos").mkdir()
    zarr_path = episode_dir / "replay.zarr"
    root = zarr.group(store=zarr.DirectoryStore(str(zarr_path)))
    data_group = root.require_group("data")
    meta_group = root.require_group("meta")
    lengths = {int(value.shape[0]) for value in arrays.values()}
    if len(lengths) != 1:
        raise ValueError(f"Fixture arrays have inconsistent lengths: {lengths}")
    steps = lengths.pop()
    for key, value in arrays.items():
        value = np.asarray(value)
        data_group.array(
            key,
            data=value,
            chunks=value.shape,
            compressor=None,
        )
    meta_group.array(
        "episode_ends",
        data=np.asarray([steps], dtype=np.int64),
        chunks=(1,),
        compressor=None,
    )
    for key, value in (attrs or {}).items():
        root.attrs[key] = value
    return zarr_path


def _base_arrays(steps: int, *, nested_action: bool = False) -> dict[str, np.ndarray]:
    action_shape = (steps, 1, 3) if nested_action else (steps, 3)
    state_shape = (steps, 1, 4) if nested_action else (steps, 4)
    return {
        "action": np.ones(action_shape, dtype=np.float32),
        "state": np.ones(state_shape, dtype=np.float32),
        "timestamp": np.arange(steps, dtype=np.float64) / 30.0,
    }


def _load_episode_arrays(zarr_path: Path) -> dict[str, np.ndarray]:
    root = zarr.open_group(str(zarr_path), mode="r")
    return {key: np.asarray(root["data"][key][:]) for key in root["data"].array_keys()}


class SensorFieldsTest(unittest.TestCase):
    def test_v2_taxels_canonicalize_single_bimanual_and_missing_episodes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root_dir = Path(tmp)
            right_path = _write_episode(
                root_dir / "right_v2",
                {**_base_arrays(2), **_make_tactile_gt_v2(2, 1, offset=10.0)},
                attrs={"tactile_gt": _tactile_gt_v2_metadata(["right"])},
            )
            both_path = _write_episode(
                root_dir / "both_v2",
                {**_base_arrays(1), **_make_tactile_gt_v2(1, 2, offset=20.0)},
                attrs={
                    "tactile_gt": _tactile_gt_v2_metadata(["right", "left"])
                },
            )
            missing_path = _write_episode(
                root_dir / "missing_sensor",
                _base_arrays(3),
            )

            plan = discover_sensor_merge_plan(
                [right_path, both_path, missing_path]
            )
            self.assertIn("tactile_gt.taxel_force_local", plan.output_keys)
            self.assertIn("next_tactile_gt.taxel_force_local", plan.output_keys)
            metadata = plan.groups[0].metadata
            self.assertEqual(metadata["schema_version"], "dexjoco.tactile_gt.v2")
            self.assertEqual(metadata["taxels_per_fingertip"], 16)
            self.assertEqual(metadata["taxel_grid_shape"], [4, 4])
            self.assertEqual(
                metadata["field_specs"]["taxel_force_local"]["shape"],
                [2, 4, 16, 3],
            )

            right = _load_episode_arrays(right_path)
            source_right = right["tactile_gt.taxel_force_local"].copy()
            source_right_next = right[
                "next_tactile_gt.taxel_force_local"
            ].copy()
            materialize_sensor_fields(right, zarr_path=right_path, plan=plan)
            self.assertEqual(
                right["tactile_gt.taxel_force_local"].shape,
                (2, 2, 4, 16, 3),
            )
            self.assertEqual(
                right["tactile_gt.taxel_force_local"].dtype,
                np.float32,
            )
            np.testing.assert_array_equal(
                right["tactile_gt.taxel_force_local"][:, 0],
                source_right[:, 0],
            )
            np.testing.assert_array_equal(
                right["next_tactile_gt.taxel_force_local"][:, 0],
                source_right_next[:, 0],
            )
            np.testing.assert_array_equal(
                right["tactile_gt.taxel_force_local"][:, 1], 0.0
            )
            np.testing.assert_array_equal(
                right["tactile_gt.hand_valid"], [[True, False]] * 2
            )

            both = _load_episode_arrays(both_path)
            source_both = both["tactile_gt.taxel_force_local"].copy()
            materialize_sensor_fields(both, zarr_path=both_path, plan=plan)
            np.testing.assert_array_equal(
                both["tactile_gt.taxel_force_local"], source_both
            )
            np.testing.assert_array_equal(
                both["tactile_gt.hand_valid"], [[True, True]]
            )

            missing = _load_episode_arrays(missing_path)
            materialize_sensor_fields(
                missing, zarr_path=missing_path, plan=plan
            )
            self.assertEqual(
                missing["tactile_gt.taxel_force_local"].shape,
                (3, 2, 4, 16, 3),
            )
            np.testing.assert_array_equal(
                missing["tactile_gt.taxel_force_local"], 0.0
            )
            np.testing.assert_array_equal(
                missing["tactile_gt.sensor_present"], False
            )
            np.testing.assert_array_equal(
                missing["tactile_gt.hand_valid"], False
            )

    def test_v1_remains_convertible_without_synthesizing_taxels(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_episode(
                Path(tmp) / "v1",
                {**_base_arrays(1), **_make_tactile_gt(1, 1)},
                attrs={"tactile_gt": _tactile_gt_metadata(["right"])},
            )
            plan = discover_sensor_merge_plan([path])
            self.assertEqual(
                plan.groups[0].metadata["schema_version"],
                "dexjoco.tactile_gt.v1",
            )
            self.assertFalse(
                any("taxel_force_local" in key for key in plan.output_keys)
            )

            episode = _load_episode_arrays(path)
            materialize_sensor_fields(episode, zarr_path=path, plan=plan)
            self.assertFalse(
                any("taxel_force_local" in key for key in episode)
            )

    def test_v1_and_v2_cannot_merge_by_zero_filling_unknown_taxels(self):
        with tempfile.TemporaryDirectory() as tmp:
            root_dir = Path(tmp)
            v1_path = _write_episode(
                root_dir / "v1",
                {**_base_arrays(1), **_make_tactile_gt(1, 1)},
                attrs={"tactile_gt": _tactile_gt_metadata(["right"])},
            )
            v2_path = _write_episode(
                root_dir / "v2",
                {**_base_arrays(1), **_make_tactile_gt_v2(1, 1)},
                attrs={"tactile_gt": _tactile_gt_v2_metadata(["right"])},
            )
            with self.assertRaisesRegex(
                ValueError, "Incompatible tactile_gt schema versions"
            ) as caught:
                discover_sensor_merge_plan([v1_path, v2_path])
            self.assertIn("not synthesized or zero-filled", str(caught.exception))

    def test_v3_is_readable_but_cannot_merge_with_v2(self):
        with tempfile.TemporaryDirectory() as tmp:
            root_dir = Path(tmp)
            metadata = _tactile_gt_v2_metadata(["right"])
            metadata["schema_version"] = "dexjoco.tactile_gt.v3"
            metadata["taxel_projection"] = {
                "method": "normal/shear separated facing Gaussian projection",
                "normalization": "per-contact taxel weights sum to one",
                "support": "only taxels facing the contact inward normal",
                "layout": "2 cap rings + 2 cylinder rings x 4 columns",
            }
            arrays = {**_base_arrays(1), **_make_tactile_gt_v2(1, 1)}
            v3_path = _write_episode(root_dir / "v3", arrays,
                                     attrs={"tactile_gt": metadata})
            plan = discover_sensor_merge_plan([v3_path])
            self.assertEqual(plan.groups[0].metadata["schema_version"],
                             "dexjoco.tactile_gt.v3")
            episode = _load_episode_arrays(v3_path)
            materialize_sensor_fields(episode, zarr_path=v3_path, plan=plan)
            self.assertEqual(episode["tactile_gt.taxel_force_local"].shape,
                             (1, 2, 4, 16, 3))
            v2_path = _write_episode(root_dir / "v2", arrays,
                attrs={"tactile_gt": _tactile_gt_v2_metadata(["right"])})
            with self.assertRaisesRegex(ValueError, "Incompatible tactile_gt schema versions"):
                discover_sensor_merge_plan([v2_path, v3_path])

    def test_v2_requires_paired_taxel_fields_with_exact_shape(self):
        with tempfile.TemporaryDirectory() as tmp:
            root_dir = Path(tmp)
            incomplete = {**_base_arrays(1), **_make_tactile_gt_v2(1, 1)}
            incomplete.pop("next_tactile_gt.taxel_force_local")
            incomplete_path = _write_episode(
                root_dir / "incomplete",
                incomplete,
                attrs={"tactile_gt": _tactile_gt_v2_metadata(["right"])},
            )
            with self.assertRaisesRegex(ValueError, "taxel_force_local"):
                discover_sensor_merge_plan([incomplete_path])

            malformed = {**_base_arrays(1), **_make_tactile_gt_v2(1, 1)}
            for prefix in ("", "next_"):
                malformed[f"{prefix}tactile_gt.taxel_force_local"] = np.zeros(
                    (1, 1, 4, 15, 3), dtype=np.float32
                )
            malformed_path = _write_episode(
                root_dir / "malformed",
                malformed,
                attrs={"tactile_gt": _tactile_gt_v2_metadata(["right"])},
            )
            with self.assertRaisesRegex(ValueError, "requires 16"):
                discover_sensor_merge_plan([malformed_path])

    def test_v2_rejects_wrong_taxel_dtype_even_when_metadata_agrees(self):
        with tempfile.TemporaryDirectory() as tmp:
            arrays = {**_base_arrays(1), **_make_tactile_gt_v2(1, 1)}
            for prefix in ("", "next_"):
                key = f"{prefix}tactile_gt.taxel_force_local"
                arrays[key] = arrays[key].astype(np.float64)

            metadata = _tactile_gt_v2_metadata(["right"])
            metadata["field_specs"]["taxel_force_local"]["dtype"] = "float64"
            path = _write_episode(
                Path(tmp) / "wrong_dtype",
                arrays,
                attrs={"tactile_gt": metadata},
            )

            with self.assertRaisesRegex(
                ValueError,
                r"taxel_force_local dtype float64.*expected float32",
            ):
                discover_sensor_merge_plan([path])

    def test_optional_single_bimanual_and_legacy_are_canonical(self):
        with tempfile.TemporaryDirectory() as tmp:
            root_dir = Path(tmp)
            right_path = _write_episode(
                root_dir / "right",
                {**_base_arrays(3), **_make_tactile_gt(3, 1, offset=10.0)},
                attrs={"tactile_gt": _tactile_gt_metadata(["right"])},
            )
            legacy_metadata = _tactile_gt_metadata(["right", "left"])
            legacy_metadata["fields"] = legacy_metadata.pop("field_specs")
            for field_spec in legacy_metadata["fields"].values():
                field_spec["axes"] = field_spec["axes"][1:]
            both_path = _write_episode(
                root_dir / "both",
                {**_base_arrays(1), **_make_tactile_gt(1, 2, offset=20.0)},
                attrs={"tactile_gt": legacy_metadata},
            )
            legacy_path = _write_episode(
                root_dir / "legacy", _base_arrays(2)
            )

            plans = [
                discover_sensor_merge_plan(
                    [right_path, both_path, legacy_path]
                ),
                discover_sensor_merge_plan(
                    [legacy_path, both_path, right_path]
                ),
            ]
            self.assertEqual(plans[0].output_keys, plans[1].output_keys)
            self.assertEqual(plans[0].groups[0].metadata, plans[1].groups[0].metadata)

            right = _load_episode_arrays(right_path)
            materialize_sensor_fields(
                right, zarr_path=right_path, plan=plans[0]
            )
            self.assertEqual(
                right["tactile_gt.fingertip_normal"].shape, (3, 2, 4)
            )
            self.assertEqual(
                right["tactile_gt.fingertip_wrench_local"].shape,
                (3, 2, 4, 6),
            )
            self.assertEqual(right["tactile_gt.sim_time"].shape, (3, 1))
            self.assertEqual(right["tactile_gt.sim_time"].dtype, np.float64)
            self.assertEqual(
                right["tactile_gt.sensor_present"].dtype, np.bool_
            )
            np.testing.assert_array_equal(
                right["tactile_gt.hand_valid"],
                np.asarray([[True, False]] * 3),
            )
            np.testing.assert_array_equal(
                right["next_tactile_gt.hand_valid"],
                np.asarray([[True, False]] * 3),
            )
            np.testing.assert_array_equal(
                right["tactile_gt.fingertip_normal"][:, 1], 0.0
            )

            both = _load_episode_arrays(both_path)
            original_both = both["tactile_gt.fingertip_normal"].copy()
            materialize_sensor_fields(
                both, zarr_path=both_path, plan=plans[0]
            )
            np.testing.assert_array_equal(
                both["tactile_gt.fingertip_normal"], original_both
            )
            np.testing.assert_array_equal(
                both["tactile_gt.hand_valid"], [[True, True]]
            )

            legacy = _load_episode_arrays(legacy_path)
            materialize_sensor_fields(
                legacy, zarr_path=legacy_path, plan=plans[0]
            )
            self.assertEqual(
                legacy["tactile_gt.fingertip_normal"].shape, (2, 2, 4)
            )
            np.testing.assert_array_equal(
                legacy["tactile_gt.fingertip_normal"], 0.0
            )
            np.testing.assert_array_equal(
                legacy["tactile_gt.sensor_present"], False
            )
            np.testing.assert_array_equal(legacy["tactile_gt.hand_valid"], False)

    def test_valid_zero_contact_is_distinct_from_missing_sensor(self):
        with tempfile.TemporaryDirectory() as tmp:
            root_dir = Path(tmp)
            zero_path = _write_episode(
                root_dir / "zero_contact",
                {
                    **_base_arrays(2),
                    **_make_tactile_gt(2, 1, all_zero=True),
                },
                attrs={"tactile_gt": _tactile_gt_metadata(["right"])},
            )
            missing_path = _write_episode(root_dir / "missing", _base_arrays(2))
            plan = discover_sensor_merge_plan([zero_path, missing_path])

            zero = _load_episode_arrays(zero_path)
            materialize_sensor_fields(zero, zarr_path=zero_path, plan=plan)
            missing = _load_episode_arrays(missing_path)
            materialize_sensor_fields(
                missing, zarr_path=missing_path, plan=plan
            )

            np.testing.assert_array_equal(
                zero["tactile_gt.fingertip_wrench_local"], 0.0
            )
            np.testing.assert_array_equal(
                missing["tactile_gt.fingertip_wrench_local"], 0.0
            )
            np.testing.assert_array_equal(
                zero["tactile_gt.sensor_present"], True
            )
            np.testing.assert_array_equal(
                zero["tactile_gt.hand_valid"], [[True, False]] * 2
            )
            np.testing.assert_array_equal(
                missing["tactile_gt.sensor_present"], False
            )
            np.testing.assert_array_equal(
                missing["tactile_gt.hand_valid"], False
            )

    def test_retired_tactile_is_warned_dropped_and_source_is_read_only(
        self,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            retired_current = np.ones((2, 1, 4, 16, 3), dtype=np.float32)
            retired_next = np.full((2, 1, 4, 16, 3), 2.0, dtype=np.float32)
            path = _write_episode(
                Path(tmp) / "retired_stream",
                {
                    **_base_arrays(2),
                    **_make_tactile_gt(2, 1),
                    "tactile.retired_field": retired_current,
                    "next_tactile.retired_field": retired_next,
                },
                attrs={
                    "tactile_gt": _tactile_gt_metadata(["right"]),
                    "tactile": {
                        "schema_version": "retired.tactile.v1"
                    },
                },
            )

            plan = discover_sensor_merge_plan([path])
            self.assertEqual([group.name for group in plan.groups], ["tactile_gt"])
            self.assertFalse(
                any(
                    key.startswith(("tactile.", "next_tactile."))
                    for key in plan.output_keys
                )
            )

            episode = _load_episode_arrays(path)
            with self.assertWarnsRegex(
                RuntimeWarning, "Dropping retired tactile fields"
            ):
                materialize_sensor_fields(episode, zarr_path=path, plan=plan)

            self.assertIn("tactile_gt.fingertip_normal", episode)
            self.assertIn("next_tactile_gt.fingertip_normal", episode)
            self.assertFalse(
                any(
                    key.startswith(("tactile.", "next_tactile."))
                    for key in episode
                )
            )

            source = zarr.open_group(str(path), mode="r")
            np.testing.assert_array_equal(
                source["data"]["tactile.retired_field"][:],
                retired_current,
            )
            np.testing.assert_array_equal(
                source["data"]["next_tactile.retired_field"][:],
                retired_next,
            )

            output = zarr.group(store=zarr.MemoryStore())
            write_sensor_metadata(output, plan)
            self.assertIn("tactile_gt", output.attrs)
            self.assertNotIn("tactile", output.attrs)

    def test_left_only_source_maps_by_hand_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_episode(
                Path(tmp) / "left",
                {**_base_arrays(1), **_make_tactile_gt(1, 1, offset=50.0)},
                attrs={"tactile_gt": _tactile_gt_metadata(["left"])},
            )
            plan = discover_sensor_merge_plan([path])
            episode = _load_episode_arrays(path)
            source = episode["tactile_gt.fingertip_normal"].copy()
            materialize_sensor_fields(episode, zarr_path=path, plan=plan)
            np.testing.assert_array_equal(
                episode["tactile_gt.fingertip_normal"][:, 0], 0.0
            )
            np.testing.assert_array_equal(
                episode["tactile_gt.fingertip_normal"][:, 1], source[:, 0]
            )
            np.testing.assert_array_equal(
                episode["tactile_gt.hand_valid"], [[False, True]]
            )

    def test_incompatible_dtype_fails_during_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            root_dir = Path(tmp)
            first_arrays = {**_base_arrays(1), **_make_tactile_gt(1, 1)}
            second_arrays = {**_base_arrays(1), **_make_tactile_gt(1, 1)}
            second_arrays["tactile_gt.fingertip_normal"] = second_arrays[
                "tactile_gt.fingertip_normal"
            ].astype(np.float64)
            first = _write_episode(
                root_dir / "first",
                first_arrays,
                attrs={"tactile_gt": _tactile_gt_metadata(["right"])},
            )
            second = _write_episode(
                root_dir / "second",
                second_arrays,
                attrs={"tactile_gt": _tactile_gt_metadata(["right"])},
            )
            with self.assertRaisesRegex(ValueError, "dtype"):
                discover_sensor_merge_plan([first, second])

    def test_partial_known_schema_fails_during_preflight(self):
        with tempfile.TemporaryDirectory() as tmp:
            arrays = {**_base_arrays(1), **_make_tactile_gt(1, 1)}
            arrays.pop("tactile_gt.contact_fraction")
            arrays.pop("next_tactile_gt.contact_fraction")
            path = _write_episode(
                Path(tmp) / "partial",
                arrays,
                attrs={"tactile_gt": _tactile_gt_metadata(["right"])},
            )
            with self.assertRaisesRegex(ValueError, "does not satisfy schema"):
                discover_sensor_merge_plan([path])

    def test_known_schema_cannot_relabel_the_hand_axis(self):
        with tempfile.TemporaryDirectory() as tmp:
            metadata = _tactile_gt_metadata(["right"])
            metadata["field_specs"]["fingertip_normal"]["axes"] = [
                "time",
                "slot",
                "fingertip",
            ]
            path = _write_episode(
                Path(tmp) / "bad_axes",
                {**_base_arrays(1), **_make_tactile_gt(1, 1)},
                attrs={"tactile_gt": metadata},
            )
            with self.assertRaisesRegex(ValueError, "do not satisfy schema"):
                discover_sensor_merge_plan([path])

    def test_current_and_next_sensor_streams_must_be_paired(self):
        with tempfile.TemporaryDirectory() as tmp:
            arrays = {**_base_arrays(1), **_make_tactile_gt(1, 1)}
            for key in list(arrays):
                if key.startswith("next_tactile_gt."):
                    arrays.pop(key)
            path = _write_episode(
                Path(tmp) / "current_only",
                arrays,
                attrs={"tactile_gt": _tactile_gt_metadata(["right"])},
            )
            with self.assertRaisesRegex(ValueError, "must be paired"):
                discover_sensor_merge_plan([path])


class _FakeReplayBuffer:
    """Small behavioral clone of the vendored ReplayBuffer methods under test."""

    def __init__(self, root: zarr.Group):
        self.root = root

    @classmethod
    def create_from_path(cls, zarr_path: str, mode: str = "r"):
        return cls(zarr.open_group(zarr_path, mode=mode))

    @classmethod
    def create_empty_zarr(cls, storage):
        root = zarr.group(store=storage)
        root.require_group("data")
        meta = root.require_group("meta")
        meta.array(
            "episode_ends",
            data=np.zeros((0,), dtype=np.int64),
            chunks=(1024,),
            compressor=None,
        )
        return cls(root)

    @property
    def data(self):
        return self.root["data"]

    @property
    def episode_ends(self):
        return self.root["meta"]["episode_ends"]

    @property
    def n_steps(self):
        return 0 if len(self.episode_ends) == 0 else int(self.episode_ends[-1])

    def get_episode(self, index: int, copy: bool = False):
        start = 0 if index == 0 else int(self.episode_ends[index - 1])
        end = int(self.episode_ends[index])
        return {key: value[start:end] for key, value in self.data.items()}

    def add_episode(self, data: dict[str, np.ndarray], compressors="disk"):
        del compressors
        current_length = self.n_steps
        episode_length = len(next(iter(data.values())))
        new_length = current_length + episode_length
        for key, value in data.items():
            value = np.asarray(value)
            new_shape = (new_length, *value.shape[1:])
            if key not in self.data:
                array = self.data.zeros(
                    key,
                    shape=new_shape,
                    chunks=value.shape,
                    dtype=value.dtype,
                    compressor=None,
                )
            else:
                array = self.data[key]
                if tuple(array.shape[1:]) != tuple(value.shape[1:]):
                    raise AssertionError((array.shape, value.shape))
                array.resize(new_shape)
            array[-episode_length:] = value
        self.episode_ends.resize((len(self.episode_ends) + 1,))
        self.episode_ends[-1] = new_length


class _ImmediateExecutor:
    def __init__(self, *args, **kwargs):
        del args, kwargs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        del exc_type, exc, traceback

    def map(self, function, iterable):
        return map(function, iterable)


class _MessageQueue:
    def __init__(self):
        self.messages = []

    def put(self, message):
        self.messages.append(message)


def _install_fake_diffusion_policy() -> None:
    diffusion_policy = types.ModuleType("diffusion_policy")
    common = types.ModuleType("diffusion_policy.common")
    replay_buffer = types.ModuleType("diffusion_policy.common.replay_buffer")
    replay_buffer.ReplayBuffer = _FakeReplayBuffer
    diffusion_policy.common = common
    common.replay_buffer = replay_buffer
    sys.modules["diffusion_policy"] = diffusion_policy
    sys.modules["diffusion_policy.common"] = common
    sys.modules["diffusion_policy.common.replay_buffer"] = replay_buffer


class ZarrMergerIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            from diffusion_policy.common.replay_buffer import ReplayBuffer  # noqa: F401
        except (ImportError, ModuleNotFoundError):
            _install_fake_diffusion_policy()
        from dexjoco_data_converter.to_zarr import merge_datasets_zarr
        from dexjoco_data_converter.to_zarr import merge_episode_zarr

        cls.merge_datasets_zarr = merge_datasets_zarr
        cls.merge_episode_zarr = merge_episode_zarr

    def test_single_merge_sensor_first_legacy_last_is_well_formed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root_dir = Path(tmp)
            input_dir = root_dir / "input"
            input_dir.mkdir()
            _write_episode(
                input_dir / "episode_000000",
                {
                    **_base_arrays(2, nested_action=True),
                    **_make_tactile_gt(2, 1, offset=10.0),
                },
                attrs={"tactile_gt": _tactile_gt_metadata(["right"])},
            )
            _write_episode(
                input_dir / "episode_000001",
                _base_arrays(3, nested_action=True),
            )
            output_dir = root_dir / "output"

            with mock.patch.object(
                self.merge_episode_zarr,
                "ProcessPoolExecutor",
                _ImmediateExecutor,
            ):
                self.merge_episode_zarr.merge_episode_worker(
                    input_dir=input_dir,
                    output_dir=output_dir,
                    dataset_name="single",
                    message_queue=_MessageQueue(),
                    slice_cfg={},
                    skip_static_frames=False,
                )

            output = zarr.open_group(
                str(output_dir / "replay_buffer.zarr"), mode="r"
            )
            np.testing.assert_array_equal(
                output["meta"]["episode_ends"][:], [2, 5]
            )
            for _, array in output["data"].arrays():
                self.assertEqual(array.shape[0], 5)
            self.assertEqual(output["data"]["action"].shape, (5, 3))
            self.assertEqual(output["data"]["state"].shape, (5, 4))
            self.assertEqual(
                output["data"]["tactile_gt.fingertip_normal"].shape,
                (5, 2, 4),
            )
            self.assertEqual(
                output["data"]["tactile_gt.sim_time"].dtype, np.float64
            )
            np.testing.assert_array_equal(
                output["data"]["tactile_gt.sensor_present"][:2], True
            )
            np.testing.assert_array_equal(
                output["data"]["tactile_gt.sensor_present"][2:], False
            )
            np.testing.assert_array_equal(
                output["data"]["tactile_gt.hand_valid"][2:], False
            )
            self.assertEqual(
                output.attrs["tactile_gt"]["hand_order"], ["right", "left"]
            )

    def test_shape_mismatch_is_rejected_before_partial_append(self):
        with tempfile.TemporaryDirectory() as tmp:
            root_dir = Path(tmp)
            input_dir = root_dir / "input"
            input_dir.mkdir()
            _write_episode(
                input_dir / "episode_000000",
                {
                    **_base_arrays(2),
                    "aux": np.ones((2, 2), dtype=np.float32),
                },
            )
            _write_episode(
                input_dir / "episode_000001",
                {
                    **_base_arrays(3),
                    "aux": np.ones((3, 3), dtype=np.float32),
                },
            )
            output_dir = root_dir / "output"

            with mock.patch.object(
                self.merge_episode_zarr,
                "ProcessPoolExecutor",
                _ImmediateExecutor,
            ):
                with self.assertRaisesRegex(ValueError, "before append"):
                    self.merge_episode_zarr.merge_episode_worker(
                        input_dir=input_dir,
                        output_dir=output_dir,
                        dataset_name="shape_guard",
                        message_queue=_MessageQueue(),
                        slice_cfg={},
                        skip_static_frames=False,
                    )

            output = zarr.open_group(
                str(output_dir / "replay_buffer.zarr"), mode="r"
            )
            np.testing.assert_array_equal(
                output["meta"]["episode_ends"][:], [2]
            )
            for _, array in output["data"].arrays():
                self.assertEqual(array.shape[0], 2)

    def test_multi_merge_is_safe_in_both_dataset_orders(self):
        with tempfile.TemporaryDirectory() as tmp:
            root_dir = Path(tmp)
            tactile_dataset = root_dir / "touch_ds"
            legacy_dataset = root_dir / "legacy_ds"
            tactile_dataset.mkdir()
            legacy_dataset.mkdir()
            _write_episode(
                tactile_dataset / "episode_000000",
                {**_base_arrays(2), **_make_tactile_gt(2, 1, offset=10.0)},
                attrs={"tactile_gt": _tactile_gt_metadata(["right"])},
            )
            _write_episode(
                legacy_dataset / "episode_000000", _base_arrays(3)
            )

            selected = {
                "touch_ds": {
                    "action": "action",
                    "state": "state",
                    "cameras": {},
                },
                "legacy_ds": {
                    "action": "action",
                    "state": "state",
                    "cameras": {},
                },
            }
            slices = {"touch_ds": {}, "legacy_ds": {}}

            for index, order in enumerate(
                (
                    [tactile_dataset, legacy_dataset],
                    [legacy_dataset, tactile_dataset],
                )
            ):
                output_dir = root_dir / f"output_{index}"
                with mock.patch.object(
                    self.merge_datasets_zarr,
                    "ProcessPoolExecutor",
                    _ImmediateExecutor,
                ):
                    self.merge_datasets_zarr.merge_datasets_worker(
                        dataset_dirs=order,
                        output_dir=output_dir,
                        dataset_name=f"multi_{index}",
                        message_queue=_MessageQueue(),
                        slice_config=slices,
                        selected_data_config=selected,
                        target_action_dim=3,
                        target_state_dim=4,
                        skip_static_frames=False,
                    )

                output = zarr.open_group(
                    str(output_dir / "replay_buffer.zarr"), mode="r"
                )
                first_episode = (
                    order[0] / "episode_000000" / "replay.zarr"
                )
                first_length = len(_load_episode_arrays(first_episode)["action"])
                np.testing.assert_array_equal(
                    output["meta"]["episode_ends"][:], [first_length, 5]
                )
                for _, array in output["data"].arrays():
                    self.assertEqual(array.shape[0], 5)
                present = output["data"]["tactile_gt.sensor_present"][:, 0]
                if order[0].name == "touch_ds":
                    np.testing.assert_array_equal(
                        present, [True, True, False, False, False]
                    )
                else:
                    np.testing.assert_array_equal(
                        present, [False, False, False, True, True]
                    )
                self.assertEqual(
                    output["data"]["tactile_gt.fingertip_normal"].shape,
                    (5, 2, 4),
                )
                self.assertEqual(
                    output.attrs["tactile_gt"]["storage_schema_version"],
                    "dexjoco.sensor_padding.v1",
                )

if __name__ == "__main__":
    unittest.main()
