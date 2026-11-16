#!/bin/bash
#SBATCH --job-name=hardex-eval-bundle
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
#SBATCH --output=/iopsstor/scratch/cscs/zjingchen/zbk/HARDex_runtime/logs/hardex-eval-bundle-%j.out
#SBATCH --error=/iopsstor/scratch/cscs/zjingchen/zbk/HARDex_runtime/logs/hardex-eval-bundle-%j.err

set -euo pipefail

CODE_ROOT=${CODE_ROOT:-/users/zjingchen/zbk/HARDex}
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
WORKERS=${WORKERS:-4}
AUTO_SUMMARY=${AUTO_SUMMARY:-false}
SUMMARY_LABEL=${SUMMARY_LABEL:-$EVAL_NAME}

# The long-running master is deliberately submitted in four-hour backfill
# slices.  Keep one policy server per GH200 so each 50-episode group completes
# and writes its marker inside a slice; excessive model concurrency makes an
# individual group slower and can prevent any resumable progress.
REQUIRE_EXPLICIT_EGL_BINDING=false
if [[ "${SLURM_JOB_NAME:-}" == "hardex-official-master" ]]; then
    REQUIRE_EXPLICIT_EGL_BINDING=true
    if (( WORKERS > 4 )); then
        echo "MASTER_BACKFILL_WORKER_CAP requested=$WORKERS effective=4"
        WORKERS=4
    fi
fi
HAND_STATE_AUX_LOSS_WEIGHT=${HAND_STATE_AUX_LOSS_WEIGHT:-0.05}
GPUS_PER_NODE=4
STRUCTURED_ARG=()
STRUCTURED_EVAL_ARG=()
SAMPLING_ARG=()

if (( WORKERS < 1 || WORKERS > 24 )); then
    echo "WORKERS must be between 1 and 24, got: $WORKERS" >&2
    exit 2
fi

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
        SAMPLING_ARG=()
        ;;
    flat_official_sampling)
        CONFIG_NAME=dexjoco_11_task_official_sampling
        STRUCTURED_ARG=()
        STRUCTURED_EVAL_ARG=()
        SAMPLING_ARG=(--sampling-balance proportional)
        ;;
    structured_official_sampling)
        CONFIG_NAME=dexjoco_11_task_structured_official_sampling
        STRUCTURED_ARG=(--structured-hand-state)
        STRUCTURED_EVAL_ARG=(--structured-hand-state)
        SAMPLING_ARG=(--sampling-balance proportional)
        ;;
    structured_handwise_v2)
        CONFIG_NAME=dexjoco_11_task_structured_handwise_v2
        STRUCTURED_ARG=(
            --structured-hand-state
            --handwise-action-experts
            --hand-state-aux-loss-weight "$HAND_STATE_AUX_LOSS_WEIGHT"
        )
        STRUCTURED_EVAL_ARG=(--structured-hand-state)
        ;;
    structured_handwise_v2_official_sampling)
        CONFIG_NAME=dexjoco_11_task_structured_handwise_v2_official_sampling
        STRUCTURED_ARG=(
            --structured-hand-state
            --handwise-action-experts
            --hand-state-aux-loss-weight "$HAND_STATE_AUX_LOSS_WEIGHT"
        )
        STRUCTURED_EVAL_ARG=(--structured-hand-state)
        SAMPLING_ARG=(--sampling-balance proportional)
        ;;
    *)
        echo "Unsupported MODEL_KIND: $MODEL_KIND" >&2
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

EVAL_ROOT="$RUNTIME_DIR/evaluations/$EVAL_NAME"
LEGACY_EVAL_ROOT="$RUNTIME_DIR/evaluations_pre_egl_fix/$EVAL_NAME"
mkdir -p "$EVAL_ROOT" "$LEGACY_EVAL_ROOT" "$RUNTIME_DIR/logs"

export PYTHONPATH="$OPENPI_DIR/src:$OPENPI_DIR/scripts:$OPENPI_DIR/packages/openpi-client/src:$CODE_ROOT/dexjoco${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Up to six model servers fit comfortably on each 98-GiB GH200. A structured
# server uses about 8.9 GiB in measured inference; the per-process ceiling
# leaves additional room for compilation and MuJoCo rendering.
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_MEM_FRACTION:-0.14}
export TOKENIZERS_PARALLELISM=false

evaluate_worker() (
    set -euo pipefail
    worker=$1
    gpu=$((worker % GPUS_PER_NODE))
    export CUDA_VISIBLE_DEVICES=$gpu
    # MuJoCo's EGL backend enumerates physical EGL devices independently of
    # CUDA_VISIBLE_DEVICES.  Without this, every simulator renders on EGL
    # device 0 even though the four policy servers are correctly spread over
    # GPUs 0-3, serializing all camera rendering on the first GH200.
    export MUJOCO_EGL_DEVICE_ID=$gpu
    port=$((19000 + worker))
    server_log="$EVAL_ROOT/policy_server_worker${worker}.log"

    cd "$OPENPI_DIR"
    "$PYTHON" scripts/serve_dexjoco_11_task_checkpoint.py \
        --checkpoint-dir "$CHECKPOINT_DIR" \
        --data-root "$DATA_ROOT" \
        --init-params-path "$INIT_PARAMS" \
        --checkpoint-base-dir "$CHECKPOINT_BASE" \
        --assets-base-dir "$ASSETS_BASE" \
        --exp-name "$EXP_NAME" \
        --port "$port" \
        "${SAMPLING_ARG[@]}" \
        "${STRUCTURED_ARG[@]}" \
        >"$server_log" 2>&1 &
    server_pid=$!
    trap 'kill "$server_pid" 2>/dev/null || true; wait "$server_pid" 2>/dev/null || true' EXIT

    server_ready=false
    for _ in $(seq 1 180); do
        if ! kill -0 "$server_pid" 2>/dev/null; then
            echo "Worker $worker policy server exited before becoming ready" >&2
            tail -n 120 "$server_log" >&2
            exit 5
        fi
        if "$PYTHON" -c \
            "import socket; s=socket.create_connection(('127.0.0.1', $port), timeout=1); s.close()" \
            >/dev/null 2>&1; then
            server_ready=true
            break
        fi
        sleep 2
    done
    if [[ "$server_ready" != true ]]; then
        echo "Worker $worker timed out waiting for policy server on port $port" >&2
        tail -n 120 "$server_log" >&2
        exit 6
    fi

    group_index=0
    for task in "${TASKS[@]}"; do
        for seed in 0 1 2; do
            if (( group_index % WORKERS != worker )); then
                group_index=$((group_index + 1))
                continue
            fi

            output_dir="$EVAL_ROOT/$task/seed$seed"
            mkdir -p "$(dirname "$output_dir")"
            if compgen -G "$output_dir/success_rate_*_${EPISODES}.txt" >/dev/null; then
                egl_marker="$output_dir/.explicit_mujoco_egl_binding"
                if [[ "$REQUIRE_EXPLICIT_EGL_BINDING" != true || -f "$egl_marker" ]]; then
                    echo "Worker $worker skipping completed group: task=$task seed=$seed"
                    group_index=$((group_index + 1))
                    continue
                fi

                # Results produced before the explicit EGL-device fix used
                # asynchronous inference while every simulator rendered on
                # physical GPU 0. Preserve them for audit, but do not mix them
                # into the formal post-fix comparison because runtime changes
                # can alter action-chunk freshness.
                legacy_dir="$LEGACY_EVAL_ROOT/$task/seed${seed}.job${SLURM_JOB_ID}.worker${worker}"
                mkdir -p "$(dirname "$legacy_dir")"
                echo "Worker $worker archiving pre-EGL-fix result: $output_dir -> $legacy_dir"
                mv "$output_dir" "$legacy_dir"
            fi
            if [[ -d "$output_dir" ]]; then
                mv "$output_dir" "${output_dir}.incomplete.${SLURM_JOB_ID}.worker${worker}"
            fi
            mkdir -p "$output_dir"

            echo "Worker $worker gpu=$gpu evaluating model=$MODEL_KIND checkpoint=$FINAL_STEP task=$task seed=$seed episodes=$EPISODES"
            "$PYTHON" -m dexjoco_openpi_client.cli.evaluate \
                --config "$CODE_ROOT/configs/rand_obj/$task.yaml" \
                --seed "$seed" \
                --host 127.0.0.1 \
                --port "$port" \
                --output "$output_dir" \
                --episodes "$EPISODES" \
                --pad-state-dim46 \
                "${STRUCTURED_EVAL_ARG[@]}"

            if ! compgen -G "$output_dir/success_rate_*_${EPISODES}.txt" >/dev/null; then
                echo "Worker $worker finished without a success-rate marker: $output_dir" >&2
                exit 7
            fi
            if [[ "$REQUIRE_EXPLICIT_EGL_BINDING" == true ]]; then
                printf 'worker=%s\nphysical_egl_device=%s\n' "$worker" "$gpu" \
                    >"$output_dir/.explicit_mujoco_egl_binding"
            fi
            echo "FORMAL_DONE worker=$worker model=$MODEL_KIND checkpoint=$FINAL_STEP task=$task seed=$seed"
            group_index=$((group_index + 1))
        done
    done
)

worker_pids=()
for worker in $(seq 0 $((WORKERS - 1))); do
    evaluate_worker "$worker" &
    worker_pids+=("$!")
done

failed=0
for worker_pid in "${worker_pids[@]}"; do
    if ! wait "$worker_pid"; then
        failed=1
    fi
done
if (( failed != 0 )); then
    echo "At least one bundled evaluation worker failed" >&2
    exit 8
fi

result_count=0
for task in "${TASKS[@]}"; do
    for seed in 0 1 2; do
        output_dir="$EVAL_ROOT/$task/seed$seed"
        if ! compgen -G "$output_dir/success_rate_*_${EPISODES}.txt" >/dev/null; then
            continue
        fi
        if [[ "$REQUIRE_EXPLICIT_EGL_BINDING" == true \
            && ! -f "$output_dir/.explicit_mujoco_egl_binding" ]]; then
            continue
        fi
        result_count=$((result_count + 1))
    done
done
if [[ "$result_count" -ne 33 ]]; then
    echo "Expected 33 canonical formal result groups, found $result_count under $EVAL_ROOT" >&2
    exit 9
fi
echo "BUNDLED_FORMAL_DONE model=$MODEL_KIND checkpoint=$FINAL_STEP groups=$result_count"

if [[ "$AUTO_SUMMARY" == "true" ]]; then
    "$PYTHON" "$CODE_ROOT/scripts/summarize_dexjoco_eval.py" \
        --root "$EVAL_ROOT" \
        --episodes "$EPISODES" \
        --label "$SUMMARY_LABEL"
fi
