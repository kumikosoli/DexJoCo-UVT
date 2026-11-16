#!/bin/bash
#SBATCH --job-name=hardex-flat-repro
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
#SBATCH --output=/iopsstor/scratch/cscs/zjingchen/zbk/HARDex_runtime/logs/hardex-flat-repro-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/zjingchen/zbk/HARDex_runtime/logs/hardex-flat-repro-%j.err

set -euo pipefail

CODE_ROOT=${CODE_ROOT:-/users/zjingchen/zbk/HARDex}
RUNTIME_DIR=/iopsstor/scratch/cscs/zjingchen/zbk/HARDex_runtime
CHECKPOINT_BASE="$RUNTIME_DIR/training_checkpoints"
CONFIG_NAME=dexjoco_11_task_official_sampling
EXP_NAME=${EXP_NAME:-flat_lora_all11_official_sampling_b32_s60000_v1}
EVAL_NAME=${EVAL_NAME:-flat_lora_all11_official_sampling_b32_s60000_final}
EPISODES=${EPISODES:-50}
EVAL_WORKERS=${EVAL_WORKERS:-22}
RUN_DIR="$CHECKPOINT_BASE/$CONFIG_NAME/$EXP_NAME"
EVAL_ROOT="$RUNTIME_DIR/evaluations/$EVAL_NAME"

cd "$CODE_ROOT"
mkdir -p "$EVAL_ROOT" "$RUNTIME_DIR/logs"

CODE_ROOT="$CODE_ROOT" \
EXP_NAME="$EXP_NAME" \
NUM_TRAIN_STEPS=60000 \
RESUME=${RESUME:-false} \
    bash scripts/clariden_flat_official_sampling_lora.sbatch

final_step=$(
    find "$RUN_DIR" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null \
        | awk '/^[0-9]+$/' \
        | sort -n \
        | tail -n 1
)
if [[ -z "$final_step" || "$final_step" -lt 59999 ]]; then
    echo "Training returned without a complete checkpoint 59999 (latest=${final_step:-none})" >&2
    exit 10
fi
for required in "$RUN_DIR/$final_step/params" "$RUN_DIR/$final_step/assets"; do
    if [[ ! -d "$required" ]]; then
        echo "Final checkpoint is incomplete: missing $required" >&2
        exit 11
    fi
done

result_count=$(find "$EVAL_ROOT" -type f -name "success_rate_*_${EPISODES}.txt" 2>/dev/null | wc -l)
if [[ "$result_count" -eq 33 ]]; then
    echo "Flat official evaluation already contains all 33 result markers; summarizing in place"
    /users/zjingchen/zbk/ENVS/hardex-openpi/bin/python \
        scripts/summarize_dexjoco_eval.py \
        --root "$EVAL_ROOT" \
        --episodes "$EPISODES" \
        --label "Flat π0.5 official-setting reproduction, checkpoint $final_step"
    exit 0
fi

# Keep training and evaluation as two allocations: the 60k run already uses
# most of the 12-hour normal-partition limit. Submit exactly one packed eval.
eval_job=$(
    sbatch --parsable \
        --export=ALL,CODE_ROOT="$CODE_ROOT",MODEL_KIND=flat_official_sampling,EXP_NAME="$EXP_NAME",EVAL_NAME="$EVAL_NAME",EPISODES="$EPISODES",WORKERS="$EVAL_WORKERS",XLA_MEM_FRACTION=0.14,AUTO_SUMMARY=true,SUMMARY_LABEL=flat_pi05_official_reproduction_checkpoint_59999 \
        scripts/clariden_eval_bundle.sbatch
)
echo "FLAT_OFFICIAL_EVAL_SUBMITTED job=$eval_job checkpoint=$final_step existing_markers=$result_count"
