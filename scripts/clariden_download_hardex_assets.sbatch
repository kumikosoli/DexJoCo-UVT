#!/bin/bash
#SBATCH --job-name=hardex-download
#SBATCH --account=aa004
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --gpus-per-node=4
#SBATCH --time=01:30:00
#SBATCH --uenv=pytorch/v2.8.0:v1
#SBATCH --view=default
#SBATCH --output=/iopsstor/scratch/cscs/zjingchen/zbk/HARDex_runtime/logs/hardex-download-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/zjingchen/zbk/HARDex_runtime/logs/hardex-download-%j.err

set -euo pipefail

CODE_DIR=/users/zjingchen/zbk/HARDex/openpi
RUNTIME_DIR=/iopsstor/scratch/cscs/zjingchen/zbk/HARDex_runtime
PYTHON=/users/zjingchen/zbk/ENVS/hardex-openpi/bin/python
STAGING_DIR="$RUNTIME_DIR/hf_staging"

mkdir -p "$RUNTIME_DIR/logs" "$STAGING_DIR"
cd "$CODE_DIR"

export PYTHONUNBUFFERED=1
export HF_HOME="$RUNTIME_DIR/hf_cache"
export HF_HUB_DISABLE_TELEMETRY=1

srun -ul "$PYTHON" scripts/download_dexjoco_hf_assets.py \
    --output-root "$STAGING_DIR" \
    --max-workers 8
