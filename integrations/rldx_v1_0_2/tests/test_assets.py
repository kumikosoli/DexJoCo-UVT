from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock


ASSET_ROOT = Path(__file__).resolve().parents[1]


def _load_launcher():
    path = ASSET_ROOT / "scripts/launch_pair.py"
    spec = importlib.util.spec_from_file_location("dexjoco_rldx_launch_pair", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_comparer():
    path = ASSET_ROOT / "scripts/compare_smoke_runs.py"
    spec = importlib.util.spec_from_file_location("dexjoco_rldx_compare_smoke", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    try:
        import yaml  # noqa: F401
    except ImportError:
        fake_yaml = types.ModuleType("yaml")
        fake_yaml.safe_load = json.loads
        with mock.patch.dict(sys.modules, {"yaml": fake_yaml}):
            spec.loader.exec_module(module)
    else:
        spec.loader.exec_module(module)
    return module


class LaunchPlanTest(unittest.TestCase):
    def test_smoke_plan_is_matched_and_uses_four_disjoint_gpus(self):
        launcher = _load_launcher()
        with tempfile.TemporaryDirectory() as tmp:
            env = {
                "RLDX_ROOT": str(Path(tmp) / "rldx"),
                "DATASET_PATH": str(Path(tmp) / "dataset"),
                "OUTPUT_ROOT": str(Path(tmp) / "output"),
            }
            with mock.patch.dict(os.environ, env, clear=False):
                tactile, mask0, metadata = launcher.build_jobs("smoke", "unit")

        self.assertEqual(tactile.gpus, "0,1")
        self.assertEqual(mask0.gpus, "2,3")
        self.assertNotEqual(tactile.master_port, mask0.master_port)
        self.assertEqual(metadata["seed"], 42)
        self.assertEqual(metadata["max_steps"], 2)
        self.assertEqual(metadata["microbatch_per_gpu"], 1)
        for job in (tactile, mask0):
            command = list(job.command)
            self.assertEqual(
                command[0], "/storage/envs/rldx-train/bin/torchrun"
            )
            self.assertIn("RLWRLD/RLDX-1-PT-IMG", command)
            self.assertEqual(command[command.index("--video-length") + 1], "1")
            self.assertEqual(command[command.index("--global-batch-size") + 1], "2")
            self.assertEqual(command[command.index("--num-gpus") + 1], "2")
            self.assertEqual(command[command.index("--max-steps") + 1], "2")
            self.assertEqual(command[command.index("--logging-steps") + 1], "1")
            self.assertIn("--no-logging-nan-inf-filter", command)
            self.assertEqual(command[command.index("--action-model-lora-rank") + 1], "16")
            self.assertEqual(command[command.index("--physics-dims") + 1], "216")
            self.assertIn("--gradient-checkpointing", command)
            self.assertIn("--allow-padding", command)
            self.assertIn("--freeze-cog-tokens", command)
        tactile_cmd = list(tactile.command)
        mask0_cmd = list(mask0.command)
        for flag in ("--modality-config-path", "--experiment-name"):
            tactile_cmd[tactile_cmd.index(flag) + 1] = "<EXPECTED_DIFFERENCE>"
            mask0_cmd[mask0_cmd.index(flag) + 1] = "<EXPECTED_DIFFERENCE>"
        tactile_cmd[2] = "--master_port=<EXPECTED_DIFFERENCE>"
        mask0_cmd[2] = "--master_port=<EXPECTED_DIFFERENCE>"
        self.assertEqual(tactile_cmd, mask0_cmd)
        self.assertEqual(metadata["logging_steps"], 1)
        self.assertIs(metadata["logging_nan_inf_filter"], False)

    def test_local_backbone_path_is_forwarded(self):
        launcher = _load_launcher()
        with tempfile.TemporaryDirectory() as tmp:
            env = {
                "RLDX_ROOT": str(Path(tmp) / "rldx"),
                "DATASET_PATH": str(Path(tmp) / "dataset"),
                "OUTPUT_ROOT": str(Path(tmp) / "output"),
                "BACKBONE_PATH": "/workspace/checkpoints/backbone_assets",
            }
            with mock.patch.dict(os.environ, env, clear=False):
                tactile, mask0, metadata = launcher.build_jobs("smoke", "unit")
        for job in (tactile, mask0):
            command = list(job.command)
            self.assertEqual(
                command[command.index("--backbone-path") + 1],
                "/workspace/checkpoints/backbone_assets",
            )
        self.assertEqual(
            metadata["backbone_path"], "/workspace/checkpoints/backbone_assets"
        )

    def test_long_train_requires_explicit_confirmation(self):
        launcher = _load_launcher()
        with mock.patch.dict(os.environ, {"RLDX_CONFIRM_LONG_TRAIN": "NO"}, clear=False):
            with self.assertRaisesRegex(ValueError, "guarded"):
                launcher.build_jobs("train", "unit")

    def test_run_id_is_restricted_to_a_safe_single_path_component(self):
        launcher = _load_launcher()
        for invalid in ("", ".", "..", "../escape", "a/b", "space here", "-starts-dash"):
            with self.subTest(run_id=invalid):
                with self.assertRaisesRegex(ValueError, "run_id must match"):
                    launcher.build_jobs("smoke", invalid)
        tactile, mask0, metadata = launcher.build_jobs("smoke", "A.b-2_3")
        self.assertEqual(metadata["run_id"], "A.b-2_3")
        self.assertTrue(tactile.run_dir.name.endswith("A.b-2_3"))
        self.assertTrue(mask0.run_dir.name.endswith("A.b-2_3"))

    def test_gpu_overlap_is_rejected(self):
        launcher = _load_launcher()
        with mock.patch.dict(
            os.environ,
            {"TACTILE_GPUS": "0,1", "MASK0_GPUS": "1,2"},
            clear=False,
        ):
            with self.assertRaisesRegex(ValueError, "disjoint"):
                launcher.build_jobs("smoke", "unit")

    def test_single_mask0_smoke_ignores_unselected_tactile_gpu_and_port(self):
        launcher = _load_launcher()
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(
                os.environ,
                {
                    "RLDX_ROOT": str(Path(tmp) / "rldx"),
                    "DATASET_PATH": str(Path(tmp) / "dataset"),
                    "OUTPUT_ROOT": str(Path(tmp) / "output"),
                    "TACTILE_GPUS": "not-a-valid-pair",
                    "TACTILE_MASTER_PORT": "not-an-integer",
                    "MASK0_GPUS": "1,2",
                    "MASK0_MASTER_PORT": "29612",
                },
                clear=False,
            ):
                _tactile, mask0, metadata = launcher.build_jobs(
                    "smoke", "unit", only="mask0"
                )
        self.assertEqual(mask0.gpus, "1,2")
        self.assertEqual(mask0.master_port, 29612)
        self.assertEqual(metadata["only"], "mask0")

    def test_long_train_defaults_to_three_thousand_steps_and_accumulation_eight(self):
        launcher = _load_launcher()
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(
                os.environ,
                {
                    "RLDX_ROOT": str(Path(tmp) / "rldx"),
                    "DATASET_PATH": str(Path(tmp) / "dataset"),
                    "OUTPUT_ROOT": str(Path(tmp) / "output"),
                    "RLDX_CONFIRM_LONG_TRAIN": "YES",
                },
                clear=False,
            ):
                tactile, mask0, metadata = launcher.build_jobs("train", "unit")
        self.assertEqual(metadata["max_steps"], 3000)
        self.assertEqual(metadata["gradient_accumulation_steps"], 8)
        self.assertEqual(metadata["logging_steps"], 10)
        self.assertIs(metadata["logging_nan_inf_filter"], False)
        for job in (tactile, mask0):
            command = list(job.command)
            self.assertEqual(command[command.index("--max-steps") + 1], "3000")
            self.assertEqual(command[command.index("--logging-steps") + 1], "10")
            self.assertIn("--no-logging-nan-inf-filter", command)

    def test_gpu_preflight_rejects_existing_process_without_starting_jobs(self):
        launcher = _load_launcher()
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(
                os.environ,
                {
                    "RLDX_ROOT": str(Path(tmp) / "rldx"),
                    "DATASET_PATH": str(Path(tmp) / "dataset"),
                    "OUTPUT_ROOT": str(Path(tmp) / "output"),
                },
                clear=False,
            ):
                tactile, mask0, _ = launcher.build_jobs("smoke", "unit")
        gpu_rows = "\n".join(
            f"{index}, GPU-{index}, 24576, 24000" for index in range(6)
        )
        process_rows = "GPU-3, 9988, python, 18000\n"
        with mock.patch.object(
            launcher, "_nvidia_smi", side_effect=[gpu_rows, process_rows]
        ):
            with self.assertRaisesRegex(ValueError, "GPU 3: active compute process"):
                launcher.assert_gpus_available((tactile, mask0))

    def test_gpu_preflight_accepts_four_idle_3090s(self):
        launcher = _load_launcher()
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(
                os.environ,
                {
                    "RLDX_ROOT": str(Path(tmp) / "rldx"),
                    "DATASET_PATH": str(Path(tmp) / "dataset"),
                    "OUTPUT_ROOT": str(Path(tmp) / "output"),
                },
                clear=False,
            ):
                tactile, mask0, _ = launcher.build_jobs("smoke", "unit")
        gpu_rows = "\n".join(
            f"{index}, GPU-{index}, 24576, 24000" for index in range(4)
        )
        with mock.patch.object(launcher, "_nvidia_smi", side_effect=[gpu_rows, ""]):
            statuses = launcher.assert_gpus_available((tactile, mask0))
        self.assertEqual(set(statuses), {"0", "1", "2", "3"})

    def test_single_arm_smoke_preflight_accepts_exactly_two_idle_gpus(self):
        launcher = _load_launcher()
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(
                os.environ,
                {
                    "RLDX_ROOT": str(Path(tmp) / "rldx"),
                    "DATASET_PATH": str(Path(tmp) / "dataset"),
                    "OUTPUT_ROOT": str(Path(tmp) / "output"),
                },
                clear=False,
            ):
                tactile, _mask0, _ = launcher.build_jobs("smoke", "unit")
        gpu_rows = "\n".join(
            f"{index}, GPU-{index}, 24576, 24000" for index in range(2)
        )
        with mock.patch.object(launcher, "_nvidia_smi", side_effect=[gpu_rows, ""]):
            statuses = launcher.assert_gpus_available((tactile,))
        self.assertEqual(set(statuses), {"0", "1"})

    def test_port_preflight_rejects_a_bound_master_port(self):
        import socket

        launcher = _load_launcher()
        bound = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.addCleanup(bound.close)
        bound.bind(("127.0.0.1", 0))
        occupied_port = bound.getsockname()[1]
        spare = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        spare.bind(("127.0.0.1", 0))
        free_port = spare.getsockname()[1]
        spare.close()
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(
                os.environ,
                {
                    "RLDX_ROOT": str(Path(tmp) / "rldx"),
                    "DATASET_PATH": str(Path(tmp) / "dataset"),
                    "OUTPUT_ROOT": str(Path(tmp) / "output"),
                    "TACTILE_MASTER_PORT": str(occupied_port),
                    "MASK0_MASTER_PORT": str(free_port),
                },
                clear=False,
            ):
                tactile, mask0, _ = launcher.build_jobs("smoke", "unit")
        with self.assertRaisesRegex(ValueError, "already in use"):
            launcher.assert_ports_available((tactile, mask0))

    def test_success_marker_is_written_only_by_post_exit_helper(self):
        launcher = _load_launcher()
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(
                os.environ,
                {
                    "RLDX_ROOT": str(Path(tmp) / "rldx"),
                    "DATASET_PATH": str(Path(tmp) / "dataset"),
                    "OUTPUT_ROOT": str(Path(tmp) / "output"),
                },
                clear=False,
            ):
                tactile, _mask0, metadata = launcher.build_jobs(
                    "smoke", "unit", only="tactile"
                )
            tactile.run_dir.mkdir(parents=True)
            self.assertFalse(tactile.success_marker.exists())
            launcher._write_success_markers((tactile,), metadata)
            marker = json.loads(tactile.success_marker.read_text())
        self.assertEqual(marker["status"], "success")
        self.assertEqual(marker["exit_code"], 0)
        self.assertEqual(marker["job"], "tactile")
        self.assertEqual(marker["max_steps"], 2)
        self.assertEqual(marker["logging_steps"], 1)
        self.assertIs(marker["logging_nan_inf_filter"], False)

    def test_output_preflight_refuses_existing_run_directory_or_log(self):
        launcher = _load_launcher()
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(
                os.environ,
                {
                    "RLDX_ROOT": str(Path(tmp) / "rldx"),
                    "DATASET_PATH": str(Path(tmp) / "dataset"),
                    "OUTPUT_ROOT": str(Path(tmp) / "output"),
                },
                clear=False,
            ):
                tactile, mask0, _ = launcher.build_jobs("smoke", "unit")
            tactile.run_dir.mkdir(parents=True)
            mask0.log_path.parent.mkdir(parents=True)
            mask0.log_path.write_text("old log")
            with self.assertRaisesRegex(ValueError, "silent resume/overwrite"):
                launcher.assert_output_paths_available((tactile, mask0))

    def test_terminate_signals_each_live_torchrun_process_group(self):
        launcher = _load_launcher()
        proc = mock.Mock()
        proc.pid = 4321
        proc.poll.return_value = None
        with mock.patch.object(launcher.os, "killpg") as killpg:
            launcher._terminate([proc], timeout=0.0)
        self.assertEqual(
            killpg.call_args_list,
            [
                mock.call(4321, launcher.signal.SIGTERM),
                mock.call(4321, launcher.signal.SIGKILL),
            ],
        )

    def test_partial_spawn_failure_runs_finally_cleanup(self):
        launcher = _load_launcher()
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.dict(
                os.environ,
                {
                    "RLDX_ROOT": str(Path(tmp) / "rldx"),
                    "DATASET_PATH": str(Path(tmp) / "dataset"),
                    "OUTPUT_ROOT": str(Path(tmp) / "output"),
                },
                clear=False,
            ):
                tactile, mask0, metadata = launcher.build_jobs("smoke", "partial")
            first_process = mock.Mock(pid=7654)
            with (
                mock.patch.object(
                    launcher.subprocess,
                    "Popen",
                    side_effect=[first_process, OSError("second spawn failed")],
                ),
                mock.patch.object(launcher, "_terminate") as terminate,
                mock.patch.object(
                    launcher.signal,
                    "signal",
                    return_value=launcher.signal.SIG_DFL,
                ),
            ):
                with self.assertRaisesRegex(OSError, "second spawn failed"):
                    launcher.run_jobs(
                        (tactile, mask0),
                        rldx_root=Path(tmp) / "rldx",
                        metadata=metadata,
                    )
            terminate.assert_called_once_with([first_process])


class ModalityConfigTest(unittest.TestCase):
    def test_mask0_keeps_identical_physics_tokens_but_loads_no_column(self):
        class ModalityConfig:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        class ActionConfig(ModalityConfig):
            pass

        class _Value:
            ABSOLUTE = "absolute"
            NON_EEF = "non_eef"
            DEFAULT = "default"

        fake_types = types.ModuleType("rldx.data.types")
        fake_types.ModalityConfig = ModalityConfig
        fake_types.ActionConfig = ActionConfig
        fake_types.ActionFormat = _Value
        fake_types.ActionRepresentation = _Value
        fake_types.ActionType = _Value
        fake_data = types.ModuleType("rldx.data")
        fake_rldx = types.ModuleType("rldx")
        path = ASSET_ROOT / "configs/dexjoco_pinch_tongs_common.py"
        spec = importlib.util.spec_from_file_location("dexjoco_pinch_tongs_common_test", path)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        with mock.patch.dict(
            sys.modules,
            {
                "rldx": fake_rldx,
                "rldx.data": fake_data,
                "rldx.data.types": fake_types,
            },
        ):
            spec.loader.exec_module(module)
        tactile = module.build_modality_config(include_tactile=True)
        mask0 = module.build_modality_config(include_tactile=False)
        self.assertEqual(tactile["tactile"].delta_indices, list(range(17)))
        self.assertEqual(mask0["tactile"].delta_indices, list(range(17)))
        self.assertEqual(tactile["tactile"].modality_keys, ["contact_features"])
        self.assertEqual(mask0["tactile"].modality_keys, [])
        self.assertEqual(
            sum(index <= 0 for index in mask0["tactile"].delta_indices), 1
        )
        self.assertEqual(
            sum(index > 0 for index in mask0["tactile"].delta_indices), 16
        )


class DatasetMetadataTest(unittest.TestCase):
    def _make_dataset(self, root: Path):
        meta = root / "meta"
        meta.mkdir(parents=True)
        info = {
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "features": {
                "observation.images.front": {"shape": [256, 256, 3]},
                "observation.images.wrist": {"shape": [256, 256, 3]},
                "observation.state": {"shape": [23]},
                "action": {"shape": [22]},
                "observation.tactile.contact_features": {"shape": [216]},
            },
        }
        modality = {
            "video": {"front": {}, "wrist": {}},
            "state": {"right_hand_state": {"start": 0, "end": 23}},
            "action": {"right_hand_action_rotvec": {"start": 0, "end": 22}},
            "tactile": {
                "contact_features": {
                    "start": 0,
                    "end": 216,
                    "original_key": "observation.tactile.contact_features",
                }
            },
        }
        (meta / "info.json").write_text(json.dumps(info))
        (meta / "modality.json").write_text(json.dumps(modality))
        (meta / "stats.json").write_text(
            json.dumps({"observation.tactile.contact_features": {}})
        )
        (meta / "episodes.jsonl").write_text(
            json.dumps({"episode_index": 0, "length": 20, "tasks": ["pinch"]}) + "\n"
        )
        (meta / "tasks.jsonl").write_text(
            json.dumps({"task_index": 0, "task": "pinch"}) + "\n"
        )

    def test_metadata_contract_accepts_216d_taxels(self):
        validator_path = ASSET_ROOT / "scripts/validate_dataset.py"
        spec = importlib.util.spec_from_file_location("dexjoco_validate_dataset", validator_path)
        validator = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(validator)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_dataset(root)
            result = validator.validate(root, scan_values=False, max_files=1)
        self.assertEqual(result["tactile_dim"], 216)
        self.assertEqual(result["episodes"], 1)

    def test_wrong_tactile_dimension_fails(self):
        validator_path = ASSET_ROOT / "scripts/validate_dataset.py"
        spec = importlib.util.spec_from_file_location("dexjoco_validate_dataset_bad", validator_path)
        validator = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(validator)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._make_dataset(root)
            info_path = root / "meta/info.json"
            info = json.loads(info_path.read_text())
            info["features"]["observation.tactile.contact_features"]["shape"] = [191]
            info_path.write_text(json.dumps(info))
            with self.assertRaisesRegex(ValueError, "expected"):
                validator.validate(root, scan_values=False, max_files=1)


class CompareSmokeEvidenceTest(unittest.TestCase):
    def _config(self, *, tactile: bool) -> dict:
        return {
            "model": {
                "use_physics": True,
                "physics_keys": ["tactile"],
                "physics_dims": [216],
                "action_model_use_lora": True,
                "action_model_lora_rank": 16,
                "video_length": 1,
                "freeze_cog_tokens": True,
            },
            "data": {
                "seed": 42,
                "modality_configs": {
                    "general_embodiment": {
                        "tactile": {
                            "delta_indices": list(range(17)),
                            "modality_keys": ["contact_features"] if tactile else [],
                        }
                    }
                },
            },
            "training": {
                "experiment_name": "tactile" if tactile else "mask0",
                "global_batch_size": 2,
                "num_gpus": 2,
                "gradient_checkpointing": True,
                "max_steps": 2,
                "logging_steps": 1,
                "logging_nan_inf_filter": False,
            },
        }

    def _make_run(self, root: Path, name: str, *, tactile: bool) -> Path:
        run = root / name
        config_dir = run / "experiment_cfg"
        state_dir = run / "checkpoint-2"
        config_dir.mkdir(parents=True)
        state_dir.mkdir(parents=True)
        (config_dir / "conf.yaml").write_text(
            json.dumps(self._config(tactile=tactile), sort_keys=True)
        )
        (run / "dexjoco_run_success.json").write_text(
            json.dumps(
                {
                    "status": "success",
                    "exit_code": 0,
                    "job": name,
                    "mode": "smoke",
                    "max_steps": 2,
                    "logging_steps": 1,
                    "logging_nan_inf_filter": False,
                    "completed_at": "2026-08-15T00:00:00+00:00",
                }
            )
        )
        (state_dir / "trainer_state.json").write_text(
            json.dumps(
                {
                    "global_step": 2,
                    "log_history": [
                        {"loss": 1.25, "step": 1},
                        {"loss": 0.75, "step": 2},
                    ],
                }
            )
        )
        shard = run / "model-00001-of-00001.safetensors"
        shard.write_bytes(b"non-empty-test-shard")
        (run / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "metadata": {"total_size": shard.stat().st_size},
                    "weight_map": {"model.test.weight": shard.name},
                }
            )
        )
        return run

    def test_compare_requires_completed_finite_two_step_runs(self):
        comparer = _load_comparer()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tactile = self._make_run(root, "tactile", tactile=True)
            mask0 = self._make_run(root, "mask0", tactile=False)
            comparer.compare(tactile, mask0)

    def test_compare_rejects_missing_post_exit_marker(self):
        comparer = _load_comparer()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tactile = self._make_run(root, "tactile", tactile=True)
            mask0 = self._make_run(root, "mask0", tactile=False)
            (tactile / "dexjoco_run_success.json").unlink()
            with self.assertRaisesRegex(ValueError, "success marker"):
                comparer.compare(tactile, mask0)

    def test_compare_rejects_nonfinite_logged_loss(self):
        comparer = _load_comparer()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tactile = self._make_run(root, "tactile", tactile=True)
            mask0 = self._make_run(root, "mask0", tactile=False)
            state_path = tactile / "checkpoint-2/trainer_state.json"
            state = json.loads(state_path.read_text())
            state["log_history"][1]["loss"] = float("nan")
            state_path.write_text(json.dumps(state))
            with self.assertRaisesRegex(ValueError, "non-finite loss"):
                comparer.compare(tactile, mask0)

    def test_compare_rejects_incomplete_global_step(self):
        comparer = _load_comparer()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tactile = self._make_run(root, "tactile", tactile=True)
            mask0 = self._make_run(root, "mask0", tactile=False)
            state_path = mask0 / "checkpoint-2/trainer_state.json"
            state = json.loads(state_path.read_text())
            state["global_step"] = 1
            state_path.write_text(json.dumps(state))
            with self.assertRaisesRegex(ValueError, "global_step"):
                comparer.compare(tactile, mask0)

    def test_compare_rejects_missing_final_model_shard(self):
        comparer = _load_comparer()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tactile = self._make_run(root, "tactile", tactile=True)
            mask0 = self._make_run(root, "mask0", tactile=False)
            (tactile / "model-00001-of-00001.safetensors").unlink()
            with self.assertRaisesRegex(ValueError, "shard is missing or empty"):
                comparer.compare(tactile, mask0)

    def test_compare_rejects_model_index_only_inside_checkpoint(self):
        comparer = _load_comparer()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tactile = self._make_run(root, "tactile", tactile=True)
            mask0 = self._make_run(root, "mask0", tactile=False)
            (tactile / "model.safetensors.index.json").unlink()
            (tactile / "checkpoint-2/model.safetensors.index.json").write_text(
                json.dumps({"weight_map": {"model.test.weight": "unused"}})
            )
            with self.assertRaisesRegex(ValueError, "run root"):
                comparer.compare(tactile, mask0)

    def test_compare_rejects_tactile_delta_indices_mismatch(self):
        comparer = _load_comparer()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            tactile = self._make_run(root, "tactile", tactile=True)
            mask0 = self._make_run(root, "mask0", tactile=False)
            config_path = mask0 / "experiment_cfg/conf.yaml"
            config = json.loads(config_path.read_text())
            config["data"]["modality_configs"]["general_embodiment"]["tactile"][
                "delta_indices"
            ] = list(range(16))
            config_path.write_text(json.dumps(config, sort_keys=True))
            with self.assertRaisesRegex(ValueError, "differ beyond"):
                comparer.compare(tactile, mask0)


class StaticAssetsTest(unittest.TestCase):
    def test_shell_scripts_parse(self):
        for name in ("apply_v1_0_2_patch.sh", "run_in_docker.sh"):
            result = subprocess.run(
                ["bash", "-n", str(ASSET_ROOT / "scripts" / name)],
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_docker_runner_uses_existing_rldx_venv(self):
        script = (ASSET_ROOT / "scripts/run_in_docker.sh").read_text()
        self.assertIn("/storage/envs/rldx-train/bin/python", script)
        self.assertIn("TORCHRUN_BIN=/storage/envs/rldx-train/bin/torchrun", script)
        self.assertIn("PYTHONPATH=/workspace/RLDX-1", script)
        self.assertIn("TRITON_CACHE_DIR=/tmp/dexjoco-rldx-triton-cache", script)
        self.assertIn(
            "TORCHINDUCTOR_CACHE_DIR=/tmp/dexjoco-rldx-torchinductor-cache",
            script,
        )
        self.assertIn('MAX_STEPS="${MAX_STEPS:-3000}"', script)
        self.assertNotIn("\n    python /workspace/DexJoCo", script)

    def test_python_scripts_compile(self):
        scripts = sorted((ASSET_ROOT / "scripts").glob("*.py"))
        result = subprocess.run(
            [sys.executable, "-m", "py_compile", *map(str, scripts)],
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_patches_contain_only_the_expected_rldx_files(self):
        patch_paths = sorted((ASSET_ROOT / "patches").glob("*.patch"))
        self.assertEqual(len(patch_paths), 5)
        paths = set()
        for patch_path in patch_paths:
            result = subprocess.run(
                ["git", "apply", "--numstat", str(patch_path)],
                text=True,
                capture_output=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            paths.update(
                line.removeprefix("diff --git a/").split(" b/", 1)[0]
                for line in patch_path.read_text().splitlines()
                if line.startswith("diff --git a/")
            )
        self.assertEqual(
            paths,
            {
                "rldx/configs/train_config.py",
                "rldx/experiment/assembly.py",
                "rldx/data/stats.py",
                "rldx/data/state_action/state_action_processor.py",
                "rldx/configs/training/training_config.py",
                "rldx/experiment/experiment.py",
                "rldx/experiment/trainer.py",
            },
        )

    def test_zero_physics_patch_changes_only_the_presence_semantics(self):
        patch_path = ASSET_ROOT / "patches/0003-preserve-valid-zero-physics.patch"
        text = patch_path.read_text()
        changed_paths = [
            line.removeprefix("diff --git a/").split(" b/", 1)[0]
            for line in text.splitlines()
            if line.startswith("diff --git a/")
        ]
        self.assertEqual(
            changed_paths,
            ["rldx/data/state_action/state_action_processor.py"],
        )
        self.assertIn("zero is a valid no-contact/no-load measurement", text)
        self.assertIn(
            '-                if modality in ("torque", "tactile") and np.all(values == 0):',
            text,
        )
        self.assertNotIn(
            '+                if modality in ("torque", "tactile") and np.all(values == 0):',
            text,
        )

    def test_nonfinite_loss_patch_is_unconditional_and_precedes_metric_logging(self):
        patch_path = ASSET_ROOT / "patches/0005-fail-fast-nonfinite-training-loss.patch"
        text = patch_path.read_text()
        changed_paths = [
            line.removeprefix("diff --git a/").split(" b/", 1)[0]
            for line in text.splitlines()
            if line.startswith("diff --git a/")
        ]
        self.assertEqual(changed_paths, ["rldx/experiment/trainer.py"])
        guard = "+        if not torch.isfinite(loss.detach()).all().item():"
        self.assertIn(guard, text)
        self.assertLess(text.index(guard), text.index("         self.loss = loss"))
        self.assertNotIn("local_rank", text)

    @unittest.skipUnless(
        os.environ.get("RLDX_PATCHED_ROOT"),
        "set RLDX_PATCHED_ROOT to run the patched RLDX runtime contract",
    )
    def test_patched_rldx_zero_physics_runtime_contract(self):
        for verifier_name in (
            "verify_zero_physics_contract.py",
            "verify_nonfinite_loss_contract.py",
        ):
            with self.subTest(verifier=verifier_name):
                verifier = ASSET_ROOT / "scripts" / verifier_name
                result = subprocess.run(
                    [sys.executable, str(verifier), os.environ["RLDX_PATCHED_ROOT"]],
                    text=True,
                    capture_output=True,
                )
                self.assertEqual(result.returncode, 0, result.stderr or result.stdout)


if __name__ == "__main__":
    unittest.main()
