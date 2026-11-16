#!/bin/bash
#SBATCH --job-name=hardex-eval-all11
#SBATCH --account=aa004
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --uenv=pytorch/v2.8.0:v1
#SBATCH --view=default
#SBATCH --array=0-32%11
#SBATCH --output=/iopsstor/scratch/cscs/zjingchen/zbk/HARDex_runtime/logs/hardex-eval-all11-%A_%a.out
#SBATCH --error=/iopsstor/scratch/cscs/zjingchen/zbk/HARDex_runtime/logs/hardex-eval-all11-%A_%a.err

set -euo pipefail

CODE_ROOT=/users/zjingchen/zbk/HARDex
OPENPI_DIR="$CODE_ROOT/openpi"
RUNTIME_DIR=/iopsstor/scratch/cscs/zjingchen/zbk/HARDex_runtime
PYTHON=/users/zjingchen/zbk/ENVS/hardex-openpi/bin/python
DATA_ROOT="$RUNTIME_DIR/datasets/dexjoco_lerobot_datasets"
INIT_PARAMS="$RUNTIME_DIR/checkpoints/pi05_base_action_dim_44/params"
CHECKPOINT_BASE="$RUNTIME_DIR/training_checkpoints"
ASSETS_BASE="$RUNTIME_DIR/assets"

MODEL_KIND=${MODEL_KIND:?Set MODEL_KIND to structured or baseline}
EXP_NAME=${EXP_NAME:?Set EXP_NAME to the training experiment name}
EVAL_NAME=${EVAL_NAME:-$EXP_NAME}
EPISODES=${EPISODES:-50}

case "$MODEL_KIND" in
    structured)
        CONFIG_NAME=dexjoco_11_task_structured
        STRUCTURED_ARG=(--structured-hand-state)
        STRUCTURED_EVAL_ARG=(--structured-hand-state)
        ;;
    baseline)
        CONFIG_NAME=dexjoco_11_task_balanced
        STRUCTURED_ARG=()
        STRUCTURED_EVAL_ARG=()
        ;;
    *)
        echo "MODEL_KIND must be structured or baseline, got: $MODEL_KIND" >&2
        exit 2
        ;;
esac

TASKS=(
    bimanual_assembly
    bimanual_hanoi
    bimanual_microwave_cook
    bimanual_photograph
    bimanual_unlock_ipad
    click_mouse
    fold_glasses
    hammer_nail
    pick_bucket
    pinch_tongs
    water_plant
)

TASK_INDEX=$((SLURM_ARRAY_TASK_ID / 3))
SEED=$((SLURM_ARRAY_TASK_ID % 3))
TASK=${TASKS[$TASK_INDEX]}
CHECKPOINT_ROOT="$CHECKPOINT_BASE/$CONFIG_NAME/$EXP_NAME"

FINAL_STEP=$(
    find "$CHECKPOINT_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' 2>/dev/null \
        | awk '/^[0-9]+$/' \
        | sort -n \
        | tail -n 1
)
if [[ -z "$FINAL_STEP" || "$FINAL_STEP" -lt 59999 ]]; then
    echo "Final checkpoint is unavailable under $CHECKPOINT_ROOT (latest=${FINAL_STEP:-none})" >&2
    exit 3
fi

CHECKPOINT_DIR="$CHECKPOINT_ROOT/$FINAL_STEP"
for required in "$CHECKPOINT_DIR/params" "$CHECKPOINT_DIR/assets"; do
    if [[ ! -d "$required" ]]; then
        echo "Final checkpoint is incomplete: missing $required" >&2
        exit 4
    fi
done

OUTPUT_DIR="$RUNTIME_DIR/evaluations/$EVAL_NAME/$TASK/seed$SEED"
mkdir -p "$(dirname "$OUTPUT_DIR")" "$RUNTIME_DIR/logs"
if compgen -G "$OUTPUT_DIR/success_rate_*_${EPISODES}.txt" >/dev/null; then
    echo "Evaluation already complete: $OUTPUT_DIR"
    exit 0
fi
if [[ -d "$OUTPUT_DIR" ]]; then
    mv "$OUTPUT_DIR" "${OUTPUT_DIR}.incomplete.${SLURM_JOB_ID}"
fi
mkdir -p "$OUTPUT_DIR"

export PYTHONPATH="$OPENPI_DIR/src:$OPENPI_DIR/scripts:$CODE_ROOT/dexjoco${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.82
export TOKENIZERS_PARALLELISM=false

PORT=$((18000 + SLURM_ARRAY_TASK_ID))
SERVER_LOG="$OUTPUT_DIR/policy_server.log"
cd "$OPENPI_DIR"

"$PYTHON" scripts/serve_dexjoco_11_task_checkpoint.py \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --data-root "$DATA_ROOT" \
    --init-params-path "$INIT_PARAMS" \
    --checkpoint-base-dir "$CHECKPOINT_BASE" \
    --assets-base-dir "$ASSETS_BASE" \
    --exp-name "$EXP_NAME" \
    --port "$PORT" \
    "${STRUCTURED_ARG[@]}" \
    >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
trap 'kill "$SERVER_PID" 2>/dev/null || true; wait "$SERVER_PID" 2>/dev/null || true' EXIT

SERVER_READY=false
for _ in $(seq 1 180); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "Policy server exited before becoming ready" >&2
        tail -n 120 "$SERVER_LOG" >&2
        exit 5
    fi
    if "$PYTHON" -c \
        "import socket; s=socket.create_connection(('127.0.0.1', $PORT), timeout=1); s.close()" \
        >/dev/null 2>&1; then
        SERVER_READY=true
        break
    fi
    sleep 2
done
if [[ "$SERVER_READY" != true ]]; then
    echo "Timed out waiting for policy server on port $PORT" >&2
    tail -n 120 "$SERVER_LOG" >&2
    exit 6
fi

echo "Evaluating model=$MODEL_KIND checkpoint=$FINAL_STEP task=$TASK seed=$SEED episodes=$EPISODES"
"$PYTHON" -m dexjoco_openpi_client.cli.evaluate \
    --config "$CODE_ROOT/configs/rand_obj/$TASK.yaml" \
    --seed "$SEED" \
    --host 127.0.0.1 \
    --port "$PORT" \
    --output "$OUTPUT_DIR" \
    --episodes "$EPISODES" \
    --pad-state-dim46 \
    "${STRUCTURED_EVAL_ARG[@]}"

if ! compgen -G "$OUTPUT_DIR/success_rate_*_${EPISODES}.txt" >/dev/null; then
    echo "Evaluation finished without a success-rate marker: $OUTPUT_DIR" >&2
    exit 7
fi
echo "FORMAL_DONE model=$MODEL_KIND checkpoint=$FINAL_STEP task=$TASK seed=$SEED"
