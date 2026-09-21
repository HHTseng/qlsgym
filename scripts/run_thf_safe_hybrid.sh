#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-$HOME/qlsgym_FNO_RL_optuna}
output=${OUTPUT:-results/thf_safe_hybrid}
gpu0=${GPU0:-0}
gpu1=${GPU1:-2}

cd "$repo"
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate qlsgym
export PYTHONPATH="$repo/src:$repo/scripts${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$output/logs"

run_gpu() {
  local gpu=$1
  shift
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/improve_thf_safe_hybrid.py "$@" \
    --output "$output" --device cuda:0
}

run_gpu "$gpu0" fallback-diagnose 2>&1 | tee "$output/logs/fallback_diagnose.log"
python scripts/improve_thf_safe_hybrid.py fallback-select --output "$output" \
  2>&1 | tee "$output/logs/fallback_select.log"

run_gpu "$gpu0" risk-diagnose --candidate-indices 0,2,4,6 \
  >"$output/logs/risk_diagnose_gpu${gpu0}.log" 2>&1 &
pid0=$!
run_gpu "$gpu1" risk-diagnose --candidate-indices 1,3,5,7 \
  >"$output/logs/risk_diagnose_gpu${gpu1}.log" 2>&1 &
pid1=$!
wait "$pid0"
wait "$pid1"
python scripts/improve_thf_safe_hybrid.py risk-select --output "$output" \
  2>&1 | tee "$output/logs/risk_select.log"

run_gpu "$gpu0" confirm --seeds 0,2,4 \
  >"$output/logs/confirm_gpu${gpu0}.log" 2>&1 &
pid0=$!
run_gpu "$gpu1" confirm --seeds 1,3 \
  >"$output/logs/confirm_gpu${gpu1}.log" 2>&1 &
pid1=$!
wait "$pid0"
wait "$pid1"

python scripts/improve_thf_safe_hybrid.py summarize --output "$output" \
  2>&1 | tee "$output/logs/summarize.log"
