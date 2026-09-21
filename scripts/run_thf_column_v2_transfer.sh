#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-$HOME/qlsgym_FNO_RL_optuna}
output=${OUTPUT:-results/thf_column_fno_v2_transfer}
gpu0=${GPU0:-0}
gpu1=${GPU1:-2}

cd "$repo"
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate qlsgym
export PYTHONPATH="$repo/src:$repo/scripts${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$output/logs"

CUDA_VISIBLE_DEVICES="$gpu0" python scripts/evaluate_thf_column_v2_transfer.py baseline \
  --output "$output" --device cuda:0 >"$output/logs/baseline.log" 2>&1

CUDA_VISIBLE_DEVICES="$gpu0" python scripts/evaluate_thf_column_v2_transfer.py run \
  --output "$output" --device cuda:0 --seeds 0,2,4 >"$output/logs/gpu${gpu0}.log" 2>&1 &
pid0=$!
CUDA_VISIBLE_DEVICES="$gpu1" python scripts/evaluate_thf_column_v2_transfer.py run \
  --output "$output" --device cuda:0 --seeds 1,3 >"$output/logs/gpu${gpu1}.log" 2>&1 &
pid1=$!
wait "$pid0"
wait "$pid1"

python scripts/evaluate_thf_column_v2_transfer.py summarize --output "$output" \
  >"$output/logs/summarize.log" 2>&1
