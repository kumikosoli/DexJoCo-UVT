#!/usr/bin/env bash
set -euo pipefail

EXPECTED_COMMIT="cf67c31f7f168ebbca6d06e8c3c23914cb94435c"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASSET_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
CLI_PATCH="$ASSET_ROOT/patches/0001-expose-training-memory-and-padding-cli.patch"
ABSOLUTE_STATS_PATCH="$ASSET_ROOT/patches/0002-keep-absolute-action-datasets-read-only.patch"
ZERO_PHYSICS_PATCH="$ASSET_ROOT/patches/0003-preserve-valid-zero-physics.patch"
LOSS_LOGGING_PATCH="$ASSET_ROOT/patches/0004-expose-auditable-loss-logging-cli.patch"
NONFINITE_LOSS_PATCH="$ASSET_ROOT/patches/0005-fail-fast-nonfinite-training-loss.patch"
RLDX_ROOT="${1:-${RLDX_ROOT:-}}"

if [[ -z "$RLDX_ROOT" ]]; then
    echo "usage: $0 /path/to/RLDX-1-v1.0.2" >&2
    exit 2
fi
if [[ ! -d "$RLDX_ROOT/.git" ]]; then
    echo "RLDX_ROOT is not a git checkout: $RLDX_ROOT" >&2
    exit 2
fi

actual_commit="$(git -C "$RLDX_ROOT" rev-parse HEAD)"
if [[ "$actual_commit" != "$EXPECTED_COMMIT" ]]; then
    echo "refusing to patch RLDX commit $actual_commit; expected v1.0.2 $EXPECTED_COMMIT" >&2
    exit 3
fi

train_config="$RLDX_ROOT/rldx/configs/train_config.py"
assembly="$RLDX_ROOT/rldx/experiment/assembly.py"
stats="$RLDX_ROOT/rldx/data/stats.py"
state_action_processor="$RLDX_ROOT/rldx/data/state_action/state_action_processor.py"
training_config="$RLDX_ROOT/rldx/configs/training/training_config.py"
experiment="$RLDX_ROOT/rldx/experiment/experiment.py"
trainer="$RLDX_ROOT/rldx/experiment/trainer.py"
cli_patch_present=false
if grep -q 'deepspeed_stage: int = 2' "$train_config" \
    && grep -q 'gradient_checkpointing: bool = False' "$train_config" \
    && grep -q 'allow_padding: bool = False' "$train_config" \
    && grep -q 'run_config.training.deepspeed_stage = cli.deepspeed_stage' "$assembly" \
    && grep -q 'run_config.training.gradient_checkpointing = cli.gradient_checkpointing' "$assembly" \
    && grep -q 'run_config.data.allow_padding = cli.allow_padding' "$assembly"; then
    cli_patch_present=true
fi

if [[ "$cli_patch_present" != true ]]; then
    git -C "$RLDX_ROOT" apply --check "$CLI_PATCH"
    git -C "$RLDX_ROOT" apply "$CLI_PATCH"
    echo "Applied RLDX v1.0.2 CLI patch to $RLDX_ROOT"
fi

if ! grep -q 'if not action_keys:' "$stats"; then
    git -C "$RLDX_ROOT" apply --check "$ABSOLUTE_STATS_PATCH"
    git -C "$RLDX_ROOT" apply "$ABSOLUTE_STATS_PATCH"
    echo "Applied read-only absolute-action stats patch to $RLDX_ROOT"
fi

zero_physics_patch_present=false
if grep -q 'zero is a valid no-contact/no-load measurement' "$state_action_processor" \
    && ! grep -Fq 'if modality in ("torque", "tactile") and np.all(values == 0):' \
        "$state_action_processor"; then
    zero_physics_patch_present=true
fi
if [[ "$zero_physics_patch_present" != true ]]; then
    git -C "$RLDX_ROOT" apply --check "$ZERO_PHYSICS_PATCH"
    git -C "$RLDX_ROOT" apply "$ZERO_PHYSICS_PATCH"
    echo "Applied valid-zero physics patch to $RLDX_ROOT"
fi

loss_logging_patch_present=false
if grep -q 'logging_steps: int = 10' "$train_config" \
    && grep -q 'logging_nan_inf_filter: bool = True' "$train_config" \
    && grep -q 'logging_nan_inf_filter: bool = True' "$training_config" \
    && grep -q 'run_config.training.logging_steps = cli.logging_steps' "$assembly" \
    && grep -q 'run_config.training.logging_nan_inf_filter = cli.logging_nan_inf_filter' \
        "$assembly" \
    && grep -q 'logging_nan_inf_filter=config.training.logging_nan_inf_filter' \
        "$experiment"; then
    loss_logging_patch_present=true
fi
if [[ "$loss_logging_patch_present" != true ]]; then
    git -C "$RLDX_ROOT" apply --check "$LOSS_LOGGING_PATCH"
    git -C "$RLDX_ROOT" apply "$LOSS_LOGGING_PATCH"
    echo "Applied auditable loss-logging patch to $RLDX_ROOT"
fi

if ! grep -q 'Non-finite total training loss' "$trainer"; then
    git -C "$RLDX_ROOT" apply --check "$NONFINITE_LOSS_PATCH"
    git -C "$RLDX_ROOT" apply "$NONFINITE_LOSS_PATCH"
    echo "Applied non-finite total-loss fail-fast patch to $RLDX_ROOT"
fi

if grep -q 'deepspeed_stage: int = 2' "$train_config" \
    && grep -q 'if not action_keys:' "$stats" \
    && grep -q 'zero is a valid no-contact/no-load measurement' "$state_action_processor" \
    && grep -q 'logging_nan_inf_filter=config.training.logging_nan_inf_filter' \
        "$experiment" \
    && grep -q 'Non-finite total training loss' "$trainer"; then
    echo "All RLDX v1.0.2 DexJoCo patches are present."
fi
