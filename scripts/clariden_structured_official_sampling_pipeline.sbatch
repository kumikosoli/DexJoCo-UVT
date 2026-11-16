#!/bin/bash
#SBATCH --job-name=hardex-prop-train
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
#SBATCH --output=/iopsstor/scratch/cscs/zjingchen/zbk/HARDex_runtime/logs/hardex-prop-pipeline-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/zjingchen/zbk/HARDex_runtime/logs/hardex-prop-pipeline-%j.err

set -euo pipefail

CODE_ROOT=/users/zjingchen/zbk/HARDex
RUNTIME_DIR=/iopsstor/scratch/cscs/zjingchen/zbk/HARDex_runtime
CHECKPOINT_BASE="$RUNTIME_DIR/training_checkpoints"
CONFIG_NAME=dexjoco_11_task_structured_official_sampling
EXP_NAME=${EXP_NAME:-structured_lora_all11_official_sampling_b32_s60000_v1}
EVAL_NAME=${EVAL_NAME:-structured_lora_all11_official_sampling_b32_s60000_final}
EPISODES=${EPISODES:-50}
EVAL_WORKERS=${EVAL_WORKERS:-22}
SUBMIT_EVAL=${SUBMIT_EVAL:-true}
RUN_DIR="$CHECKPOINT_BASE/$CONFIG_NAME/$EXP_NAME"
EVAL_ROOT="$RUNTIME_DIR/evaluations/$EVAL_NAME"

cd "$CODE_ROOT"
mkdir -p "$EVAL_ROOT" "$RUNTIME_DIR/logs"

# The training script also computes frame-proportional normalization statistics
# inside this allocation when they are not present yet.
EXP_NAME="$EXP_NAME" NUM_TRAIN_STEPS=60000 RESUME=${RESUME:-false} \
    bash scripts/clariden_structured_official_sampling_lora.sbatch

FINAL_STEP=$(
    find "$RUN_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null \
        | awk '/^[0-9]+$/' \
        | sort -n \
        | tail -n 1
)
if [[ -z "$FINAL_STEP" || "$FINAL_STEP" -lt 59999 ]]; then
    echo "Training returned without a complete checkpoint 59999 (latest=${FINAL_STEP:-none})" >&2
    exit 10
fi
for required in "$RUN_DIR/$FINAL_STEP/params" "$RUN_DIR/$FINAL_STEP/assets"; do
    if [[ ! -d "$required" ]]; then
        echo "Final checkpoint is incomplete: missing $required" >&2
        exit 11
    fi
done

result_count=$(find "$EVAL_ROOT" -type f -name "success_rate_*_${EPISODES}.txt" 2>/dev/null | wc -l)
if [[ "$result_count" -eq 33 ]]; then
    echo "Formal evaluation already contains all 33 result markers; summarizing in place"
    /users/zjingchen/zbk/ENVS/hardex-openpi/bin/python \
        scripts/summarize_dexjoco_eval.py \
        --root "$EVAL_ROOT" \
        --episodes "$EPISODES" \
        --label "Structured hand-state LoRA, official frame-proportional sampling, checkpoint $FINAL_STEP"
    exit 0
fi

if [[ "$SUBMIT_EVAL" != "true" ]]; then
    echo "AUTO_EVAL_PREQUEUED_BY_DEPENDENCY checkpoint=$FINAL_STEP existing_markers=$result_count"
    exit 0
fi

# Clariden normal jobs are limited to 12 hours. Training alone takes about
# 7.5 hours, so submit the one formal evaluation allocation automatically
# after checkpoint verification instead of risking a mid-evaluation timeout.
eval_job=$(
    sbatch --parsable \
        --export=ALL,MODEL_KIND=structured_official_sampling,EXP_NAME="$EXP_NAME",EVAL_NAME="$EVAL_NAME",EPISODES="$EPISODES",WORKERS="$EVAL_WORKERS",XLA_MEM_FRACTION=0.14,AUTO_SUMMARY=true,SUMMARY_LABEL=structured_official_frame_proportional_checkpoint_59999 \
        scripts/clariden_eval_bundle.sbatch
)
echo "AUTO_EVAL_SUBMITTED job=$eval_job checkpoint=$FINAL_STEP existing_markers=$result_count"
