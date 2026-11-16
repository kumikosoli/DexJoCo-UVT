#!/bin/bash
#SBATCH --job-name=hardex-flat-official
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
#SBATCH --output=/iopsstor/scratch/cscs/zjingchen/zbk/HARDex_runtime/logs/hardex-flat-official-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/zjingchen/zbk/HARDex_runtime/logs/hardex-flat-official-%j.err

set -euo pipefail

CODE_ROOT=${CODE_ROOT:-/users/zjingchen/zbk/HARDex}
OPENPI_DIR="$CODE_ROOT/openpi"
RUNTIME_DIR=/iopsstor/scratch/cscs/zjingchen/zbk/HARDex_runtime
PYTHON=/users/zjingchen/zbk/ENVS/hardex-openpi/bin/python
DATA_ROOT="$RUNTIME_DIR/datasets/dexjoco_lerobot_datasets"
INIT_PARAMS="$RUNTIME_DIR/checkpoints/pi05_base_action_dim_44/params"
ASSETS_BASE="$RUNTIME_DIR/assets"
CHECKPOINT_BASE="$RUNTIME_DIR/training_checkpoints"

CONFIG_NAME=dexjoco_11_task_official_sampling
EXP_NAME=${EXP_NAME:-flat_lora_all11_official_sampling_b32_s60000_v1}
NUM_TRAIN_STEPS=${NUM_TRAIN_STEPS:-60000}
SAVE_INTERVAL=${SAVE_INTERVAL:-10000}
KEEP_PERIOD=${KEEP_PERIOD:-5000}
RESUME=${RESUME:-false}
RUN_DIR="$CHECKPOINT_BASE/$CONFIG_NAME/$EXP_NAME"
NORM_STATS="$ASSETS_BASE/$CONFIG_NAME/local_repo/norm_stats.json"
PAUSE_AFTER_TRAIN_FILE=${PAUSE_AFTER_TRAIN_FILE:-"$RUNTIME_DIR/orchestration/official_flat_to_v2_master_v1/pause_after_flat_train"}

pause_master_after_training_if_requested() {
    if [[ ! -f "$PAUSE_AFTER_TRAIN_FILE" ]]; then
        return 0
    fi
    if [[ -z "${SLURM_JOB_ID:-}" ]]; then
        echo "Pause-after-train was requested, but SLURM_JOB_ID is unavailable" >&2
        exit 3
    fi

    echo "PAUSE_AFTER_FLAT_TRAIN job=$SLURM_JOB_ID marker=$PAUSE_AFTER_TRAIN_FILE"
    # Requeue the master in a user-held state. This releases the allocation
    # immediately, preserves the same Slurm job and checkpoint, and prevents
    # the master from falling through into formal evaluation until the user
    # explicitly removes the marker and releases the job.
    scontrol requeuehold "$SLURM_JOB_ID"
    sleep 600
    exit 0
}

latest_step=""
if [[ -d "$RUN_DIR" ]]; then
    latest_step=$(
        find "$RUN_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
            | awk '/^[0-9]+$/' \
            | sort -n \
            | tail -n 1
    )
fi
if [[ -n "$latest_step" && "$latest_step" -ge $((NUM_TRAIN_STEPS - 1)) ]]; then
    echo "Flat official reproduction is already complete at checkpoint $latest_step"
    pause_master_after_training_if_requested
    exit 0
fi

for required in \
    "$OPENPI_DIR/scripts/dexjoco_11_task_official_sampling.py" \
    "$OPENPI_DIR/scripts/compute_proportional_dexjoco_11_norm_stats.py" \
    "$DATA_ROOT/hammer_nail/meta/info.json" \
    "$DATA_ROOT/bimanual_assembly/meta/info.json" \
    "$INIT_PARAMS/_METADATA"; do
    if [[ ! -s "$required" ]]; then
        echo "Required input is missing or empty: $required" >&2
        exit 2
    fi
done

mkdir -p "$ASSETS_BASE" "$CHECKPOINT_BASE" "$RUNTIME_DIR/logs"
cd "$OPENPI_DIR"

export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_MODE=disabled
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8

COMMON_ARGS=(
    --data-root "$DATA_ROOT"
    --init-params-path "$INIT_PARAMS"
    --exp-name "$EXP_NAME"
    --checkpoint-base-dir "$CHECKPOINT_BASE"
    --assets-base-dir "$ASSETS_BASE"
    --project-name dexjoco-pi05-flat-official-reproduction
    --batch-size 32
    --num-workers 4
    --num-train-steps "$NUM_TRAIN_STEPS"
    --save-interval "$SAVE_INTERVAL"
    --keep-period "$KEEP_PERIOD"
    --fsdp-devices 1
    --video-backend pyav
)

if [[ ! -s "$NORM_STATS" ]]; then
    echo "Computing flat official frame-proportional normalization statistics"
    srun -ul "$PYTHON" scripts/dexjoco_11_task_official_sampling.py \
        --operation norm-stats \
        "${COMMON_ARGS[@]}"
fi
test -s "$NORM_STATS"

TRAIN_ARGS=(--operation train "${COMMON_ARGS[@]}")
if [[ "$RESUME" == "true" && -n "$latest_step" ]]; then
    TRAIN_ARGS+=(--resume)
fi

echo "Starting flat official reproduction: exp=$EXP_NAME steps=$NUM_TRAIN_STEPS resume=$RESUME"
srun -ul "$PYTHON" scripts/dexjoco_11_task_official_sampling.py "${TRAIN_ARGS[@]}"

latest_step=$(
    find "$RUN_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' \
        | awk '/^[0-9]+$/' \
        | sort -n \
        | tail -n 1
)
if [[ -z "$latest_step" || "$latest_step" -lt $((NUM_TRAIN_STEPS - 1)) ]]; then
    echo "Training exited without the final checkpoint: latest=${latest_step:-none}" >&2
    exit 4
fi
for required in "$RUN_DIR/$latest_step/params" "$RUN_DIR/$latest_step/assets"; do
    if [[ ! -d "$required" ]]; then
        echo "Final checkpoint is incomplete: missing $required" >&2
        exit 5
    fi
done

pause_master_after_training_if_requested
