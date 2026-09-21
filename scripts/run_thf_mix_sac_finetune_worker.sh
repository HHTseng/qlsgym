#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-$HOME/qlsgym_FNO_RL_optuna}
gpu=${GPU:?set GPU to 0 or 2}
output=${OUTPUT:-results/thf_mix_sac_exact_finetune}
shift 0

cd "$repo"
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate qlsgym
export PYTHONPATH="$repo/src:$repo/scripts${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$gpu"

mode=${MODE:-confirm}
if [[ $# -eq 0 ]]; then
  echo "usage: GPU=0 MODE=confirm $0 SEED [SEED ...]" >&2
  echo "   or: GPU=0 MODE=diagnose $0 LR_INDEX [LR_INDEX ...]" >&2
  exit 2
fi
for item in "$@"; do
  if [[ "$mode" == "diagnose" ]]; then
    python scripts/fine_tune_thf_mix_sac.py diagnose-lr \
      --output "$output" --device cuda:0 --lr-index "$item"
  else
    python scripts/fine_tune_thf_mix_sac.py confirm-seed \
      --output "$output" --device cuda:0 --source-seed "$item"
  fi
done
