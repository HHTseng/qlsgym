#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-$HOME/qlsgym_FNO_RL_optuna}
output=${OUTPUT:-results/thf_descending_hybrid}
gpu0=${GPU0:-0}
gpu1=${GPU1:-2}

cd "$repo"
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate qlsgym
export PYTHONPATH="$repo/src:$repo/scripts${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$output/logs"

CUDA_VISIBLE_DEVICES="$gpu0" python scripts/improve_thf_descending_hybrid.py diagnose \
  --output "$output" --device cuda:0 --indices 0,2,4,6,8,10,12 \
  >"$output/logs/diagnose_gpu${gpu0}.log" 2>&1 &
pid0=$!
CUDA_VISIBLE_DEVICES="$gpu1" python scripts/improve_thf_descending_hybrid.py diagnose \
  --output "$output" --device cuda:0 --indices 1,3,5,7,9,11,13 \
  >"$output/logs/diagnose_gpu${gpu1}.log" 2>&1 &
pid1=$!
wait "$pid0"
wait "$pid1"

python scripts/improve_thf_descending_hybrid.py select --output "$output" \
  >"$output/logs/select.log" 2>&1

CUDA_VISIBLE_DEVICES="$gpu0" python scripts/improve_thf_descending_hybrid.py confirm \
  --output "$output" --device cuda:0 --seeds 0,2,4 \
  >"$output/logs/confirm_gpu${gpu0}.log" 2>&1 &
pid0=$!
CUDA_VISIBLE_DEVICES="$gpu1" python scripts/improve_thf_descending_hybrid.py confirm \
  --output "$output" --device cuda:0 --seeds 1,3 \
  >"$output/logs/confirm_gpu${gpu1}.log" 2>&1 &
pid1=$!
wait "$pid0"
wait "$pid1"

python scripts/improve_thf_descending_hybrid.py summarize --output "$output" \
  >"$output/logs/summarize.log" 2>&1
