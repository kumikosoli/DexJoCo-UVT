#!/usr/bin/env bash
# 3-task pilot cache + reminder of the train commands.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
TASKS=(water_plant pick_bucket hammer_nail)

python -m effect_vla.scripts.cache_effect_targets \
  --task "${TASKS[0]}" --task "${TASKS[1]}" --task "${TASKS[2]}" \
  --skip-existing "$@"

echo
echo "Next (openpi env, cd openpi):"
for t in "${TASKS[@]}"; do
  echo "  python scripts/train.py ${t}_continue"
  echo "  python scripts/train.py ${t}_effect_v1_bootstrap"
  echo "  python scripts/train.py ${t}_effect_v1"
done
