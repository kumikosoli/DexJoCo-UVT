#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
IMAGE="${RLDX_DOCKER_IMAGE:-rldx1-train:local}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEXJOCO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
RLDX_HOST_ROOT="${RLDX_HOST_ROOT:-}"
DATASET_HOST_PATH="${DATASET_HOST_PATH:-}"
OUTPUT_HOST_ROOT="${OUTPUT_HOST_ROOT:-}"
HF_HOST_CACHE="${HF_HOST_CACHE:-$HOME/.cache/huggingface}"
BASE_MODEL_HOST_PATH="${BASE_MODEL_HOST_PATH:-}"
BACKBONE_HOST_PATH="${BACKBONE_HOST_PATH:-}"
RLDX_RUNTIME_HOST_ROOT="${RLDX_RUNTIME_HOST_ROOT:-/21231_data1/rldx1}"

BASE_MODEL_CONTAINER_PATH="${BASE_MODEL_PATH:-RLWRLD/RLDX-1-PT-IMG}"
if [[ -n "$BASE_MODEL_HOST_PATH" ]]; then
    if [[ ! -d "$BASE_MODEL_HOST_PATH" ]]; then
        echo "BASE_MODEL_HOST_PATH is not a directory: $BASE_MODEL_HOST_PATH" >&2
        exit 2
    fi
    BASE_MODEL_CONTAINER_PATH=/workspace/checkpoints/base_model
fi
BACKBONE_CONTAINER_PATH="${BACKBONE_PATH:-}"
if [[ -n "$BACKBONE_HOST_PATH" ]]; then
    if [[ ! -d "$BACKBONE_HOST_PATH" ]]; then
        echo "BACKBONE_HOST_PATH is not a directory: $BACKBONE_HOST_PATH" >&2
        exit 2
    fi
    BACKBONE_CONTAINER_PATH=/workspace/checkpoints/backbone_assets
fi

for name in RLDX_HOST_ROOT DATASET_HOST_PATH OUTPUT_HOST_ROOT; do
    if [[ -z "${!name}" ]]; then
        echo "$name must be set" >&2
        exit 2
    fi
done
if [[ ! -x "$RLDX_RUNTIME_HOST_ROOT/envs/rldx-train/bin/python" \
    || ! -x "$RLDX_RUNTIME_HOST_ROOT/envs/rldx-train/bin/torchrun" ]]; then
    echo "RLDX runtime is missing python/torchrun under $RLDX_RUNTIME_HOST_ROOT/envs/rldx-train" >&2
    exit 2
fi
if [[ "$MODE" != "smoke" && "$MODE" != "train" ]]; then
    echo "mode must be smoke or train" >&2
    exit 2
fi
if [[ "$MODE" == "train" && "${RLDX_CONFIRM_LONG_TRAIN:-}" != "YES" ]]; then
    echo "export RLDX_CONFIRM_LONG_TRAIN=YES only after the two-step smoke passes" >&2
    exit 3
fi

mkdir -p "$OUTPUT_HOST_ROOT/.runtime_home" "$HF_HOST_CACHE"
docker image inspect "$IMAGE" >/dev/null

runtime_uid="$(id -u)"
runtime_gid="$(id -g)"
LAUNCH_ARGS=("$MODE")
if [[ -n "${RLDX_ONLY:-}" ]]; then
    LAUNCH_ARGS+=(--only "$RLDX_ONLY")
fi
if [[ -n "${RLDX_RUN_ID:-}" ]]; then
    LAUNCH_ARGS+=(--run-id "$RLDX_RUN_ID")
fi

DOCKER_ARGS=(
    run --rm
    --name "dexjoco-rldx-ablation-${USER:-user}"
    --user "$runtime_uid:$runtime_gid"
    --gpus all
    --ipc=host
    --network=host
    --ulimit memlock=-1
    -v "$RLDX_RUNTIME_HOST_ROOT:/storage:ro"
    -v "$RLDX_HOST_ROOT:/workspace/RLDX-1:rw"
    -v "$DEXJOCO_ROOT:/workspace/DexJoCo:ro"
    -v "$DATASET_HOST_PATH:/workspace/data/dexjoco/pinch_tongs:ro"
    -v "$OUTPUT_HOST_ROOT:/workspace/outputs/dexjoco_rldx:rw"
    -v "$HF_HOST_CACHE:/workspace/huggingface:rw"
)
if [[ -n "$BASE_MODEL_HOST_PATH" ]]; then
    DOCKER_ARGS+=(
        -v "$BASE_MODEL_HOST_PATH:/workspace/checkpoints/base_model:ro"
    )
fi
if [[ -n "$BACKBONE_HOST_PATH" ]]; then
    DOCKER_ARGS+=(
        -v "$BACKBONE_HOST_PATH:/workspace/checkpoints/backbone_assets:ro"
    )
fi
DOCKER_ARGS+=(
    -e RLDX_ROOT=/workspace/RLDX-1
    -e DATASET_PATH=/workspace/data/dexjoco/pinch_tongs
    -e OUTPUT_ROOT=/workspace/outputs/dexjoco_rldx
    -e BASE_MODEL_PATH="$BASE_MODEL_CONTAINER_PATH"
    -e BACKBONE_PATH="$BACKBONE_CONTAINER_PATH"
    -e TORCHRUN_BIN=/storage/envs/rldx-train/bin/torchrun
    -e PYTHONPATH=/workspace/RLDX-1
    -e PYTHONPYCACHEPREFIX=/tmp/dexjoco-rldx-pycache
    -e TRITON_CACHE_DIR=/tmp/dexjoco-rldx-triton-cache
    -e TORCHINDUCTOR_CACHE_DIR=/tmp/dexjoco-rldx-torchinductor-cache
    -e HOME=/workspace/outputs/dexjoco_rldx/.runtime_home
    -e HF_HOME=/workspace/huggingface
    -e HF_HUB_CACHE=/workspace/huggingface/hub
    -e HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
    -e TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
    -e NO_ALBUMENTATIONS_UPDATE=1
    -e TACTILE_GPUS="${TACTILE_GPUS:-0,1}"
    -e MASK0_GPUS="${MASK0_GPUS:-2,3}"
    -e TACTILE_MASTER_PORT="${TACTILE_MASTER_PORT:-29611}"
    -e MASK0_MASTER_PORT="${MASK0_MASTER_PORT:-29612}"
    -e MIN_FREE_GPU_MIB="${MIN_FREE_GPU_MIB:-22000}"
    -e DEEPSPEED_STAGE="${DEEPSPEED_STAGE:-3}"
    -e MAX_STEPS="${MAX_STEPS:-3000}"
    -e GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-8}"
    -e LOGGING_STEPS="${LOGGING_STEPS:-10}"
    -e LOGGING_NAN_INF_FILTER="${LOGGING_NAN_INF_FILTER:-false}"
    -e DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-2}"
    -e LEARNING_RATE="${LEARNING_RATE:-1e-4}"
    -e SAVE_STEPS="${SAVE_STEPS:-1000}"
    -e SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"
    -e RLDX_CONFIRM_LONG_TRAIN="${RLDX_CONFIRM_LONG_TRAIN:-NO}"
    "$IMAGE"
    /storage/envs/rldx-train/bin/python \
    /workspace/DexJoCo/integrations/rldx_v1_0_2/scripts/launch_pair.py \
    "${LAUNCH_ARGS[@]}"
)
docker "${DOCKER_ARGS[@]}"
