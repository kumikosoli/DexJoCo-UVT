#!/usr/bin/env bash
# 3-task task-balanced π0.5 baseline (water_plant / pick_bucket / hammer_nail).
# Equal mixture 1/3. Init from action_dim=44 base (single-arm states are right-padded).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OPENPI_DIR="${OPENPI_DIR:-$ROOT/openpi}"
DATA_ROOT="${DATA_ROOT:-$ROOT/datasets/dexjoco_lerobot_datasets}"
# Prefer sibling datasets path used on this machine if local openpi datasets is empty.
if [[ ! -d "$DATA_ROOT/water_plant" ]]; then
  DATA_ROOT="/HOME/sysu_xdliang/sysu_xdliang_1/HDD_POOL/linmin/datasets/DexJoCo-Datasets-LeRobot/dexjoco_lerobot_datasets"
fi
INIT_PARAMS="${INIT_PARAMS:-$ROOT/checkpoints/pi05_base_action_dim_44/params}"
CKPT_BASE="${CKPT_BASE:-$ROOT/checkpoints}"
ASSETS_BASE="${ASSETS_BASE:-$OPENPI_DIR/assets}"
EXP_NAME="${EXP_NAME:-pilot3_task_balanced_v1}"
NUM_TRAIN_STEPS="${NUM_TRAIN_STEPS:-30000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
VIDEO_BACKEND="${VIDEO_BACKEND:-torchcodec}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
KEEP_PERIOD="${KEEP_PERIOD:-10000}"
FSDP_DEVICES="${FSDP_DEVICES:-1}"
RESUME="${RESUME:-0}"
OPERATION="${1:-train}"  # inspect | norm-stats | train

cd "$OPENPI_DIR"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-openpi}"

SITE="$(python -c 'import site; print(site.getsitepackages()[0])')"
# Prefer the env's NVIDIA wheels for both torch and JAX/cuDNN (avoids mixed lib bus errors).
export LD_LIBRARY_PATH="$SITE/cusparselt/lib:$SITE/nvidia/cudnn/lib:$SITE/nvidia/cublas/lib:$SITE/nvidia/cuda_runtime/lib:$SITE/nvidia/cuda_nvrtc/lib:$SITE/nvidia/cusparse/lib:$SITE/nvidia/cusolver/lib:$SITE/nvidia/nccl/lib:$SITE/nvidia/nvjitlink/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION="${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.85}"
export TOKENIZERS_PARALLELISM=false
# Do not set unknown XLA_FLAGS — JAX aborts immediately on unrecognized flags.

COMMON_ARGS=(
  --operation "$OPERATION"
  --pilot3
  --data-root "$DATA_ROOT"
  --init-params-path "$INIT_PARAMS"
  --exp-name "$EXP_NAME"
  --checkpoint-base-dir "$CKPT_BASE"
  --assets-base-dir "$ASSETS_BASE"
  --project-name dexjoco-pi05-pilot3-task-balanced
  --batch-size "$BATCH_SIZE"
  --num-workers "${NUM_WORKERS:-4}"
  --num-train-steps "$NUM_TRAIN_STEPS"
  --save-interval "$SAVE_INTERVAL"
  --keep-period "$KEEP_PERIOD"
  --fsdp-devices "$FSDP_DEVICES"
  --video-backend "${VIDEO_BACKEND:-torchcodec}"
)
if [[ "$RESUME" == "1" || "$RESUME" == "true" ]]; then
  COMMON_ARGS+=(--resume)
fi

echo "operation=$OPERATION tasks=water_plant,pick_bucket,hammer_nail steps=$NUM_TRAIN_STEPS"
python scripts/dexjoco_11_task_balanced.py "${COMMON_ARGS[@]}"
