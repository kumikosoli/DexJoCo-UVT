#!/usr/bin/env python3
"""Launch the matched tactile/mask0 RLDX LoRA pair inside one Docker container."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import datetime as dt
import json
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import sys
import time


RLDX_V102_COMMIT = "cf67c31f7f168ebbca6d06e8c3c23914cb94435c"
FIXED_SEED = 42  # RLDX v1.0.2 DataConfig default; identical in both processes.
SUCCESS_MARKER_NAME = "dexjoco_run_success.json"
RUN_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*\Z")


@dataclass(frozen=True)
class Job:
    name: str
    gpus: str
    master_port: int
    modality_config: Path
    command: tuple[str, ...]
    log_path: Path
    run_dir: Path
    success_marker: Path


@dataclass(frozen=True)
class GPUStatus:
    index: str
    uuid: str
    total_mib: int
    free_mib: int
    processes: tuple[tuple[int, str, int], ...]


def _env_int(name: str, default: int, *, minimum: int = 0) -> int:
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc
    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{name} must be one of true/false, 1/0, yes/no, on/off; got {raw!r}"
    )


def _validate_run_id(run_id: str) -> str:
    """Keep experiment names inside their intended output directory."""

    if run_id in {".", ".."} or RUN_ID_PATTERN.fullmatch(run_id) is None:
        raise ValueError(
            "run_id must match [A-Za-z0-9][A-Za-z0-9._-]* and cannot be '.' or '..'; "
            f"got {run_id!r}"
        )
    return run_id


def _parse_gpus(value: str, name: str) -> tuple[str, str]:
    items = tuple(part.strip() for part in value.split(",") if part.strip())
    if len(items) != 2 or len(set(items)) != 2:
        raise ValueError(f"{name} must contain exactly two distinct GPU ids, got {value!r}")
    return items  # type: ignore[return-value]


def _nvidia_smi(*args: str) -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", *args],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise ValueError("nvidia-smi is unavailable inside the training container") from exc
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.strip() or exc.stdout.strip() or f"exit {exc.returncode}"
        raise ValueError(f"nvidia-smi failed: {detail}") from exc
    return result.stdout


def query_gpu_status() -> dict[str, GPUStatus]:
    """Return physical GPU occupancy without mutating or signaling any process."""

    gpu_output = _nvidia_smi(
        "--query-gpu=index,uuid,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    )
    inventory: dict[str, tuple[str, int, int]] = {}
    for line in gpu_output.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 4:
            raise ValueError(f"unexpected nvidia-smi GPU row: {line!r}")
        index, uuid, total, free = parts
        try:
            inventory[index] = (uuid, int(total), int(free))
        except ValueError as exc:
            raise ValueError(f"unexpected nvidia-smi memory value: {line!r}") from exc

    process_output = _nvidia_smi(
        "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
        "--format=csv,noheader,nounits",
    )
    by_uuid: dict[str, list[tuple[int, str, int]]] = {}
    for line in process_output.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",", 3)]
        if len(parts) != 4:
            raise ValueError(f"unexpected nvidia-smi process row: {line!r}")
        uuid, pid, process_name, used = parts
        try:
            by_uuid.setdefault(uuid, []).append((int(pid), process_name, int(used)))
        except ValueError as exc:
            raise ValueError(f"unexpected nvidia-smi process value: {line!r}") from exc

    return {
        index: GPUStatus(
            index=index,
            uuid=uuid,
            total_mib=total,
            free_mib=free,
            processes=tuple(by_uuid.get(uuid, [])),
        )
        for index, (uuid, total, free) in inventory.items()
    }


def assert_gpus_available(jobs: tuple[Job, ...]) -> dict[str, GPUStatus]:
    """Fail before either job starts if any requested GPU is occupied or too full."""

    requested = [gpu for job in jobs for gpu in _parse_gpus(job.gpus, job.name)]
    expected = 2 * len(jobs)
    if len(set(requested)) != expected:
        raise ValueError(
            f"the requested run requires {expected} unique GPUs, got {requested}"
        )
    minimum_free = _env_int("MIN_FREE_GPU_MIB", 22000, minimum=1)
    statuses = query_gpu_status()
    problems = []
    for gpu in requested:
        status = statuses.get(gpu)
        if status is None:
            problems.append(f"GPU {gpu}: not visible in this container")
            continue
        if status.processes:
            details = "; ".join(
                f"pid={pid} name={name} used={used}MiB"
                for pid, name, used in status.processes
            )
            problems.append(f"GPU {gpu}: active compute process ({details})")
        if status.free_mib < minimum_free:
            problems.append(
                f"GPU {gpu}: only {status.free_mib}MiB free; require >= {minimum_free}MiB"
            )
    if problems:
        raise ValueError(
            "GPU preflight refused to start either job; no existing process was touched:\n  - "
            + "\n  - ".join(problems)
        )
    return {gpu: statuses[gpu] for gpu in requested}


def assert_ports_available(jobs: tuple[Job, ...]) -> None:
    """Reject a colliding torchrun rendezvous port without touching its owner."""

    for job in jobs:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
            sock.bind(("0.0.0.0", job.master_port))
        except OSError as exc:
            raise ValueError(
                f"{job.name} master port {job.master_port} is already in use"
            ) from exc
        finally:
            sock.close()


def _assert_rldx_ready(root: Path) -> None:
    required = {
        root / "rldx/configs/train_config.py": (
            "deepspeed_stage: int = 2",
            "gradient_checkpointing: bool = False",
            "allow_padding: bool = False",
            "logging_steps: int = 10",
            "logging_nan_inf_filter: bool = True",
        ),
        root / "rldx/experiment/assembly.py": (
            "run_config.training.deepspeed_stage = cli.deepspeed_stage",
            "run_config.training.gradient_checkpointing = cli.gradient_checkpointing",
            "run_config.data.allow_padding = cli.allow_padding",
            "run_config.training.logging_steps = cli.logging_steps",
            "run_config.training.logging_nan_inf_filter = cli.logging_nan_inf_filter",
        ),
        root / "rldx/data/stats.py": ("if not action_keys:",),
        root / "rldx/data/state_action/state_action_processor.py": (
            "zero is a valid no-contact/no-load measurement",
        ),
        root / "rldx/configs/training/training_config.py": (
            "logging_nan_inf_filter: bool = True",
        ),
        root / "rldx/experiment/experiment.py": (
            "logging_nan_inf_filter=config.training.logging_nan_inf_filter",
        ),
        root / "rldx/experiment/trainer.py": (
            "Non-finite total training loss",
        ),
        root / "rldx/experiment/launch_train.py": (),
    }
    for path, needles in required.items():
        if not path.is_file():
            raise ValueError(f"missing RLDX source file: {path}")
        text = path.read_text()
        for needle in needles:
            if needle not in text:
                raise ValueError(
                    f"RLDX v1.0.2 CLI patch is missing {needle!r}; "
                    "run scripts/apply_v1_0_2_patch.sh first"
                )
    zero_skip = 'if modality in ("torque", "tactile") and np.all(values == 0):'
    state_action_text = (
        root / "rldx/data/state_action/state_action_processor.py"
    ).read_text()
    if zero_skip in state_action_text:
        raise ValueError(
            "RLDX still treats a present all-zero tactile array as missing; "
            "run scripts/apply_v1_0_2_patch.sh first"
        )
    git_dir = root / ".git"
    if git_dir.exists():
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        ).stdout.strip()
        if commit != RLDX_V102_COMMIT:
            raise ValueError(
                f"RLDX HEAD is {commit}; expected v1.0.2 {RLDX_V102_COMMIT}"
            )


def _base_args(
    *,
    base_model: str,
    backbone_path: str | None,
    dataset: Path,
    output_root: Path,
    max_steps: int,
    accumulation: int,
    dataloader_workers: int,
    deepspeed_stage: int,
    logging_steps: int,
    logging_nan_inf_filter: bool,
) -> list[str]:
    args = [
        "--base-model-path",
        base_model,
        "--dataset-path",
        str(dataset),
        "--embodiment-tag",
        "GENERAL_EMBODIMENT",
        "--video-length",
        "1",
        "--n-cog-tokens",
        "64",
        "--action-horizon",
        "16",
        "--dataset-mode",
        "standard",
        "--allow-padding",
        "--global-batch-size",
        "2",
        "--gradient-accumulation-steps",
        str(accumulation),
        "--num-gpus",
        "2",
        "--dataloader-num-workers",
        str(dataloader_workers),
        "--learning-rate",
        os.environ.get("LEARNING_RATE", "1e-4"),
        "--max-steps",
        str(max_steps),
        "--logging-steps",
        str(logging_steps),
        "--save-steps",
        os.environ.get("SAVE_STEPS", "1000"),
        "--save-total-limit",
        os.environ.get("SAVE_TOTAL_LIMIT", "2"),
        "--output-dir",
        str(output_root),
        "--action-model-use-lora",
        "--action-model-lora-rank",
        "16",
        "--action-model-lora-alpha",
        "32",
        "--action-model-lora-dropout",
        "0.0",
        "--tune-top-llm-layers",
        "0",
        "--freeze-cog-tokens",
        "--use-physics",
        "--physics-keys",
        "tactile",
        "--physics-dims",
        "216",
        "--physics-loss-weight",
        "0.1",
        "--physics-dropout-prob",
        "0.0",
        "--allow-missing-physics",
        "--deepspeed-stage",
        str(deepspeed_stage),
        "--gradient-checkpointing",
    ]
    if not logging_nan_inf_filter:
        args.append("--no-logging-nan-inf-filter")
    if backbone_path:
        args.extend(("--backbone-path", backbone_path))
    return args


def build_jobs(
    mode: str,
    run_id: str,
    *,
    only: str | None = None,
) -> tuple[Job, Job, dict]:
    run_id = _validate_run_id(run_id)
    asset_root = Path(__file__).resolve().parents[1]
    rldx_root = Path(os.environ.get("RLDX_ROOT", "/workspace/RLDX-1"))
    dataset = Path(
        os.environ.get("DATASET_PATH", "/workspace/data/dexjoco/pinch_tongs")
    )
    output_root = Path(
        os.environ.get("OUTPUT_ROOT", "/workspace/outputs/dexjoco_rldx")
    )
    base_model = os.environ.get("BASE_MODEL_PATH", "RLWRLD/RLDX-1-PT-IMG")
    backbone_path = os.environ.get("BACKBONE_PATH") or None
    torchrun = os.environ.get(
        "TORCHRUN_BIN", "/storage/envs/rldx-train/bin/torchrun"
    )
    tactile_gpus = os.environ.get("TACTILE_GPUS", "0,1")
    mask0_gpus = os.environ.get("MASK0_GPUS", "2,3")
    if only not in (None, "tactile", "mask0"):
        raise ValueError(f"only must be tactile, mask0, or None; got {only!r}")
    if only != "mask0":
        tactile_ids = _parse_gpus(tactile_gpus, "TACTILE_GPUS")
        tactile_port = _env_int("TACTILE_MASTER_PORT", 29611, minimum=1024)
    else:
        tactile_ids = ()
        tactile_port = 29611
    if only != "tactile":
        mask0_ids = _parse_gpus(mask0_gpus, "MASK0_GPUS")
        mask0_port = _env_int("MASK0_MASTER_PORT", 29612, minimum=1024)
    else:
        mask0_ids = ()
        mask0_port = 29612
    if only is None and set(tactile_ids) & set(mask0_ids):
        raise ValueError("TACTILE_GPUS and MASK0_GPUS must be disjoint")
    if only is None and tactile_port == mask0_port:
        raise ValueError("the two torchrun master ports must differ")

    if mode == "smoke":
        max_steps = 2
        accumulation = 1
        logging_steps = 1
        logging_nan_inf_filter = False
    else:
        if os.environ.get("RLDX_CONFIRM_LONG_TRAIN") != "YES":
            raise ValueError(
                "long training is guarded; export RLDX_CONFIRM_LONG_TRAIN=YES after smoke passes"
            )
        max_steps = _env_int("MAX_STEPS", 3000, minimum=3)
        accumulation = _env_int("GRADIENT_ACCUMULATION_STEPS", 8, minimum=1)
        logging_steps = _env_int("LOGGING_STEPS", 10, minimum=1)
        logging_nan_inf_filter = _env_bool("LOGGING_NAN_INF_FILTER", False)
    workers = _env_int("DATALOADER_NUM_WORKERS", 2, minimum=0)
    stage = _env_int("DEEPSPEED_STAGE", 3, minimum=2)
    if stage not in (2, 3):
        raise ValueError("DEEPSPEED_STAGE must be 2 or 3")

    common = _base_args(
        base_model=base_model,
        backbone_path=backbone_path,
        dataset=dataset,
        output_root=output_root,
        max_steps=max_steps,
        accumulation=accumulation,
        dataloader_workers=workers,
        deepspeed_stage=stage,
        logging_steps=logging_steps,
        logging_nan_inf_filter=logging_nan_inf_filter,
    )
    logs = output_root / "logs"
    configs = asset_root / "configs"

    def make_job(name: str, gpus: str, port: int, config_name: str) -> Job:
        config = configs / config_name
        exp_name = f"pinch_tongs_{name}_{run_id}"
        run_dir = output_root / exp_name
        command = (
            torchrun,
            "--nproc_per_node=2",
            f"--master_port={port}",
            "rldx/experiment/launch_train.py",
            *common,
            "--modality-config-path",
            str(config),
            "--experiment-name",
            exp_name,
        )
        return Job(
            name=name,
            gpus=gpus,
            master_port=port,
            modality_config=config,
            command=command,
            log_path=logs / f"{exp_name}.log",
            run_dir=run_dir,
            success_marker=run_dir / SUCCESS_MARKER_NAME,
        )

    tactile = make_job(
        "tactile", tactile_gpus, tactile_port, "dexjoco_pinch_tongs_tactile.py"
    )
    mask0 = make_job(
        "mask0", mask0_gpus, mask0_port, "dexjoco_pinch_tongs_mask0.py"
    )
    metadata = {
        "rldx_version": "v1.0.2",
        "rldx_commit": RLDX_V102_COMMIT,
        "docker_image": "rldx1-train:local",
        "mode": mode,
        "seed": FIXED_SEED,
        "dataset": str(dataset),
        "base_model": base_model,
        "backbone_path": backbone_path,
        "torchrun": torchrun,
        "video_length": 1,
        "tactile_dim": 216,
        "microbatch_per_gpu": 1,
        "gradient_accumulation_steps": accumulation,
        "max_steps": max_steps,
        "logging_steps": logging_steps,
        "logging_nan_inf_filter": logging_nan_inf_filter,
        "deepspeed_stage": stage,
        "output_root": str(output_root),
        "rldx_root": str(rldx_root),
        "gpu_preflight_min_free_mib": _env_int("MIN_FREE_GPU_MIB", 22000, minimum=1),
        "only": only,
        "run_id": run_id,
    }
    return tactile, mask0, metadata


def _print_plan(jobs: tuple[Job, ...], metadata: dict) -> None:
    plan = {
        **metadata,
        "jobs": [
            {
                "name": job.name,
                "cuda_visible_devices": job.gpus,
                "master_port": job.master_port,
                "modality_config": str(job.modality_config),
                "log": str(job.log_path),
                "run_dir": str(job.run_dir),
                "success_marker": str(job.success_marker),
                "command": list(job.command),
                "shell": f"CUDA_VISIBLE_DEVICES={shlex.quote(job.gpus)} "
                + shlex.join(job.command),
            }
            for job in jobs
        ],
    }
    print(json.dumps(plan, indent=2, sort_keys=True))


def _signal_process_group(proc: subprocess.Popen, signum: int) -> None:
    """Signal the complete torchrun session created with start_new_session=True."""

    try:
        os.killpg(proc.pid, signum)
    except ProcessLookupError:
        pass


def _terminate(processes: list[subprocess.Popen], timeout: float = 20.0) -> None:
    for proc in processes:
        if proc.poll() is None:
            _signal_process_group(proc, signal.SIGTERM)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and any(proc.poll() is None for proc in processes):
        time.sleep(0.2)
    for proc in processes:
        if proc.poll() is None:
            _signal_process_group(proc, signal.SIGKILL)


def assert_output_paths_available(jobs: tuple[Job, ...]) -> None:
    """Refuse silent resume or overwrite before creating either process."""

    conflicts = []
    for job in jobs:
        if os.path.lexists(job.run_dir):
            conflicts.append(f"{job.name} run directory already exists: {job.run_dir}")
        if os.path.lexists(job.log_path):
            conflicts.append(f"{job.name} log already exists: {job.log_path}")
    if conflicts:
        raise ValueError(
            "output preflight refused silent resume/overwrite; choose a new --run-id:\n  - "
            + "\n  - ".join(conflicts)
        )


def _write_success_markers(jobs: tuple[Job, ...], metadata: dict) -> None:
    """Write completion evidence only after every selected process exits zero."""

    completed_at = dt.datetime.now(dt.timezone.utc).isoformat()
    for job in jobs:
        if not job.run_dir.is_dir():
            raise ValueError(
                f"{job.name} exited zero but did not create its run directory: {job.run_dir}"
            )
        payload = {
            "schema_version": 1,
            "status": "success",
            "exit_code": 0,
            "job": job.name,
            "mode": metadata["mode"],
            "max_steps": metadata["max_steps"],
            "logging_steps": metadata["logging_steps"],
            "logging_nan_inf_filter": metadata["logging_nan_inf_filter"],
            "completed_at": completed_at,
            "run_dir": str(job.run_dir),
            "log_path": str(job.log_path),
            "command": list(job.command),
        }
        temporary = job.success_marker.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        temporary.replace(job.success_marker)


def run_jobs(jobs: tuple[Job, ...], *, rldx_root: Path, metadata: dict) -> int:
    processes: list[subprocess.Popen] = []
    log_files = []
    old_int = None
    old_term = None
    assert_output_paths_available(jobs)

    def stop_handler(signum, _frame):
        _terminate(processes)
        raise SystemExit(128 + signum)

    try:
        old_int = signal.signal(signal.SIGINT, stop_handler)
        old_term = signal.signal(signal.SIGTERM, stop_handler)
        for job in jobs:
            job.log_path.parent.mkdir(parents=True, exist_ok=True)
            # Exclusive creation closes the race between preflight and spawn.
            log_file = job.log_path.open("x")
            log_files.append(log_file)
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = job.gpus
            env.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
            process = subprocess.Popen(
                job.command,
                cwd=rldx_root,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            processes.append(process)
            print(f"started {job.name}: pid={process.pid}, log={job.log_path}")

        while True:
            statuses = [proc.poll() for proc in processes]
            failed = [status for status in statuses if status not in (None, 0)]
            if failed:
                _terminate(processes)
                return failed[0]
            if all(status == 0 for status in statuses):
                _write_success_markers(jobs, metadata)
                return 0
            time.sleep(2.0)
    finally:
        # Also covers a failure while spawning the second arm.
        _terminate(processes)
        if old_int is not None:
            signal.signal(signal.SIGINT, old_int)
        if old_term is not None:
            signal.signal(signal.SIGTERM, old_term)
        for log_file in log_files:
            log_file.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", nargs="?", choices=("smoke", "train"), default="smoke")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--only",
        choices=("tactile", "mask0"),
        help="Run one arm for smoke diagnostics only; long training remains paired.",
    )
    parser.add_argument("--skip-dataset-preflight", action="store_true")
    parser.add_argument("--run-id")
    args = parser.parse_args()
    run_id = args.run_id or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        if args.only and args.mode != "smoke":
            raise ValueError("--only is restricted to two-step smoke diagnostics")
        tactile, mask0, metadata = build_jobs(args.mode, run_id, only=args.only)
        jobs = (
            (tactile,)
            if args.only == "tactile"
            else (mask0,)
            if args.only == "mask0"
            else (tactile, mask0)
        )
        rldx_root = Path(metadata["rldx_root"])
        dataset = Path(metadata["dataset"])
        _assert_rldx_ready(rldx_root)
        if not args.skip_dataset_preflight:
            validator = Path(__file__).with_name("validate_dataset.py")
            subprocess.run([sys.executable, str(validator), str(dataset)], check=True)
        assert_output_paths_available(jobs)
        _print_plan(jobs, metadata)
        if args.dry_run:
            return 0
        assert_ports_available(jobs)
        available = assert_gpus_available(jobs)
        print(
            "GPU preflight passed: "
            + ", ".join(
                f"GPU {gpu} free={status.free_mib}MiB"
                for gpu, status in available.items()
            )
        )
        return run_jobs(jobs, rldx_root=rldx_root, metadata=metadata)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"RLDX PAIR LAUNCH FAILED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
