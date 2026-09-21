#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-$HOME/qlsgym_FNO_RL_optuna}
gpu=${GPU:-2}
output=${OUTPUT:-results/thf_mix_sac_exact_finetune}

cd "$repo"
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate qlsgym
export PYTHONPATH="$repo/src:$repo/scripts${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$gpu"
mkdir -p "$output/logs"

python scripts/fine_tune_thf_mix_sac.py run --output "$output" --device cuda:0
