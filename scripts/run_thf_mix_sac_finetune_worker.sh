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

if [[ $# -eq 0 ]]; then
  echo "usage: GPU=0 $0 SEED [SEED ...]" >&2
  exit 2
fi
for seed in "$@"; do
  python scripts/fine_tune_thf_mix_sac.py confirm-seed \
    --output "$output" --device cuda:0 --source-seed "$seed"
done
