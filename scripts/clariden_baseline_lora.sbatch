#!/bin/bash
#SBATCH --job-name=hardex-flat-60k
#SBATCH --account=aa004
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --gpus-per-node=4
#SBATCH --time=12:00:00
#SBATCH --uenv=pytorch/v2.8.0:v1
#SBATCH --view=default
#SBATCH --open-mode=append
#SBATCH --output=/iopsstor/scratch/cscs/zjingchen/zbk/HARDex_runtime/logs/hardex-flat-60k-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/zjingchen/zbk/HARDex_runtime/logs/hardex-flat-60k-%j.err

set -euo pipefail

CODE_DIR=/users/zjingchen/zbk/HARDex/openpi
RUNTIME_DIR=/iopsstor/scratch/cscs/zjingchen/zbk/HARDex_runtime
PYTHON=/users/zjingchen/zbk/ENVS/hardex-openpi/bin/python
DATA_ROOT="$RUNTIME_DIR/datasets/dexjoco_lerobot_datasets"
INIT_PARAMS="$RUNTIME_DIR/checkpoints/pi05_base_action_dim_44/params"
ASSETS_DIR="$RUNTIME_DIR/assets"
CHECKPOINT_DIR="$RUNTIME_DIR/training_checkpoints"

NUM_TRAIN_STEPS=${NUM_TRAIN_STEPS:-60000}
SAVE_INTERVAL=${SAVE_INTERVAL:-2000}
KEEP_PERIOD=${KEEP_PERIOD:-10000}
EXP_NAME=${EXP_NAME:-baseline_lora_all11_b32_s60000_v1}
RESUME=${RESUME:-false}
RUN_DIR="$CHECKPOINT_DIR/dexjoco_11_task_balanced/$EXP_NAME"

FINAL_STEP=""
if [[ -d "$RUN_DIR" ]]; then
    FINAL_STEP=$(
        find "$RUN_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
            | awk '/^[0-9]+$/' \
            | sort -n \
            | tail -n 1
    )
fi
if [[ -n "$FINAL_STEP" && "$FINAL_STEP" -ge $((NUM_TRAIN_STEPS - 1)) ]]; then
    echo "Baseline training is already complete at checkpoint $FINAL_STEP"
    exit 0
fi

for required in \
    "$CODE_DIR/scripts/dexjoco_11_task_balanced.py" \
    "$DATA_ROOT/hammer_nail/meta/info.json" \
    "$DATA_ROOT/bimanual_assembly/meta/info.json" \
    "$INIT_PARAMS/_METADATA" \
    "$ASSETS_DIR/dexjoco_11_task_balanced/local_repo/norm_stats.json"; do
    if [[ ! -s "$required" ]]; then
        echo "Required input is missing or empty: $required" >&2
        exit 2
    fi
done

mkdir -p "$ASSETS_DIR" "$CHECKPOINT_DIR" "$RUNTIME_DIR/logs"
cd "$CODE_DIR"

export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

ARGS=(
    --operation train
    --data-root "$DATA_ROOT"
    --init-params-path "$INIT_PARAMS"
    --exp-name "$EXP_NAME"
    --checkpoint-base-dir "$CHECKPOINT_DIR"
    --assets-base-dir "$ASSETS_DIR"
    --project-name dexjoco-pi05-hand-flat-baseline
    --batch-size 32
    --num-workers 16
    --num-train-steps "$NUM_TRAIN_STEPS"
    --save-interval "$SAVE_INTERVAL"
    --keep-period "$KEEP_PERIOD"
    --fsdp-devices 1
    --video-backend pyav
)

if [[ "$RESUME" == "true" && -n "$FINAL_STEP" ]]; then
    ARGS+=(--resume)
fi

echo "Starting flat-hand LoRA baseline: exp=$EXP_NAME steps=$NUM_TRAIN_STEPS resume=$RESUME"
srun -ul "$PYTHON" scripts/dexjoco_11_task_balanced.py "${ARGS[@]}"
