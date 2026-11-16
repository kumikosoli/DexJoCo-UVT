#!/bin/bash
#SBATCH --job-name=hardex-eval-summary
#SBATCH --account=aa004
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gpus-per-node=1
#SBATCH --time=00:15:00
#SBATCH --uenv=pytorch/v2.8.0:v1
#SBATCH --view=default
#SBATCH --output=/iopsstor/scratch/cscs/zjingchen/zbk/HARDex_runtime/logs/hardex-eval-summary-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/zjingchen/zbk/HARDex_runtime/logs/hardex-eval-summary-%j.err

set -euo pipefail

CODE_ROOT=/users/zjingchen/zbk/HARDex
RUNTIME_DIR=/iopsstor/scratch/cscs/zjingchen/zbk/HARDex_runtime
PYTHON=/users/zjingchen/zbk/ENVS/hardex-openpi/bin/python
EVAL_NAME=${EVAL_NAME:?Set EVAL_NAME to the evaluation directory name}
LABEL=${LABEL:?Set LABEL to a human-readable model label}
EPISODES=${EPISODES:-50}

"$PYTHON" "$CODE_ROOT/scripts/summarize_dexjoco_eval.py" \
    --root "$RUNTIME_DIR/evaluations/$EVAL_NAME" \
    --episodes "$EPISODES" \
    --label "$LABEL"
