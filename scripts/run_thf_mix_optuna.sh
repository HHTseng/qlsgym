#!/usr/bin/env bash
set -euo pipefail

repo=$(cd "$(dirname "$0")/.." && pwd)
cd "$repo"

python_bin=${PYTHON_BIN:-$HOME/.conda/envs/qlsgym/bin/python}
output=${OUTPUT:-results/thf_rl_optuna_mix}
manifest=${MANIFEST:-$HOME/qlsgym_work/checkpoints/thf/mix.json}
trials_per_worker=${TRIALS_PER_WORKER:-40}
gpus=(${GPUS:-0 2})

if (( ${#gpus[@]} > 2 )); then
  echo "Refusing to use more than two GPUs" >&2
  exit 2
fi

mkdir -p "$output/logs"
export PYTHONUNBUFFERED=1
export QLSGYM_WORK=${QLSGYM_WORK:-$HOME/qlsgym_work}
export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"

run_pair() {
  local stage=$1
  shift
  local pids=()
  local worker=0
  for gpu in "${gpus[@]}"; do
    CUDA_VISIBLE_DEVICES=$gpu "$python_bin" scripts/optimize_thf_mix_rl.py "$@" \
      --worker-id "$worker" --device cuda:0 --output "$output" --manifest "$manifest" \
      >"$output/logs/${stage}_worker${worker}.log" 2>&1 &
    pids+=("$!")
    worker=$((worker + 1))
  done
  local failed=0
  for pid in "${pids[@]}"; do
    wait "$pid" || failed=1
  done
  if (( failed )); then
    echo "$stage failed; inspect $output/logs" >&2
    exit 1
  fi
}

date -Is >"$output/started_at.txt"
"$python_bin" scripts/optimize_thf_mix_rl.py init --output "$output" \
  >"$output/logs/init.log" 2>&1
run_pair broad optimize --trials-per-agent "$trials_per_worker"

"$python_bin" scripts/optimize_thf_mix_rl.py prepare --output "$output" \
  >"$output/logs/prepare.log" 2>&1
run_pair promotion run-queue --stage promotion

"$python_bin" scripts/optimize_thf_mix_rl.py select --output "$output" \
  >"$output/logs/select.log" 2>&1
run_pair final run-queue --stage final

"$python_bin" scripts/optimize_thf_mix_rl.py summarize --output "$output" \
  >"$output/logs/summarize.log" 2>&1
date -Is >"$output/completed_at.txt"

echo "complete: $output/summary.md"
