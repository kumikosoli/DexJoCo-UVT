#!/bin/bash
#SBATCH --job-name=hardex-flat-norm
#SBATCH --account=aa004
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --gpus-per-node=4
#SBATCH --time=00:30:00
#SBATCH --uenv=pytorch/v2.8.0:v1
#SBATCH --view=default
#SBATCH --output=/iopsstor/scratch/cscs/zjingchen/zbk/HARDex_runtime/logs/hardex-flat-norm-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/zjingchen/zbk/HARDex_runtime/logs/hardex-flat-norm-%j.err

set -euo pipefail

CODE_DIR=/users/zjingchen/zbk/HARDex/openpi
RUNTIME_DIR=/iopsstor/scratch/cscs/zjingchen/zbk/HARDex_runtime
PYTHON=/users/zjingchen/zbk/ENVS/hardex-openpi/bin/python
NORM_STATS="$RUNTIME_DIR/assets/dexjoco_11_task_balanced/local_repo/norm_stats.json"

if [[ -s "$NORM_STATS" ]]; then
    echo "Flat task-balanced normalization statistics already exist: $NORM_STATS"
    exit 0
fi

mkdir -p "$RUNTIME_DIR/assets" "$RUNTIME_DIR/training_checkpoints" "$RUNTIME_DIR/logs"
cd "$CODE_DIR"

export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled

srun -ul "$PYTHON" scripts/dexjoco_11_task_balanced.py \
    --operation norm-stats \
    --data-root "$RUNTIME_DIR/datasets/dexjoco_lerobot_datasets" \
    --init-params-path "$RUNTIME_DIR/checkpoints/pi05_base_action_dim_44/params" \
    --exp-name baseline_lora_all11_b32_s60000_v1 \
    --checkpoint-base-dir "$RUNTIME_DIR/training_checkpoints" \
    --assets-base-dir "$RUNTIME_DIR/assets" \
    --batch-size 32 \
    --num-workers 16 \
    --num-train-steps 60000 \
    --save-interval 2000 \
    --keep-period 10000 \
    --fsdp-devices 1 \
    --video-backend pyav

test -s "$NORM_STATS"
echo "BASELINE_NORM_DONE $NORM_STATS"
