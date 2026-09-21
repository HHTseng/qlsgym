#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-$HOME/qlsgym_FNO_RL_optuna}
output=${OUTPUT:-results/thf_fno_rl_improvement}
manifest=${MANIFEST:-$HOME/qlsgym_work/checkpoints/thf/mix.json}
gpu=${GPU:-0}

cd "$repo"
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate qlsgym
export PYTHONPATH="$repo/src:$repo/scripts${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$gpu"

python scripts/improve_thf_rl.py diagnose \
  --output "$output" --manifest "$manifest" --device cuda:0
python scripts/improve_thf_rl.py select \
  --output "$output" --manifest "$manifest" --device cuda:0
python scripts/improve_thf_rl.py confirm \
  --output "$output" --manifest "$manifest" --device cuda:0
python scripts/improve_thf_rl.py summarize \
  --output "$output" --manifest "$manifest" --device cuda:0
