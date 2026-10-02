#!/usr/bin/env bash
set -euo pipefail

GPU0=${GPU0:-0}
GPU1=${GPU1:-2}
OUTPUT=${OUTPUT:-results/thf_jose_ppo_comparison}
MANIFEST=${MANIFEST:-${QLSGYM_WORK:?set QLSGYM_WORK}/checkpoints/thf/mix.json}

run_one() {
  local gpu=$1 profile=$2 selection=$3 seed=$4
  local name="final_ppo_${profile}_${selection}_s${seed}"
  if [[ -f "$OUTPUT/runs/${name}.json" ]]; then
    echo "[skip] $name"
    return
  fi
  echo "[run] gpu=$gpu profile=$profile selection=$selection seed=$seed"
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/thf_rl_agents.py run \
    --agent ppo --preset final --ppo-profile "$profile" \
    --ppo-selection "$selection" --manifest "$MANIFEST" \
    --fno-tag mix --min-manifest-coverage 1.0 --seed "$seed" \
    --device cuda --eval-batch 128 --output "$OUTPUT"
}

worker() {
  local gpu=$1
  shift
  local task
  for task in "$@"; do
    read -r profile selection seed <<<"$task"
    run_one "$gpu" "$profile" "$selection" "$seed"
  done
}

mkdir -p "$OUTPUT/logs"
tasks0=(
  "jose_matched fno 0"
  "jose_main fno 0"
  "jose_tuned fno 0"
  "jose_main exact 0"
  "jose_matched fno 2"
  "jose_main fno 3"
  "jose_tuned fno 3"
  "jose_matched fno 4"
  "jose_main fno 4"
)
tasks1=(
  "jose_matched fno 1"
  "jose_main fno 1"
  "jose_tuned fno 1"
  "jose_main exact 1"
  "jose_main fno 2"
  "jose_tuned fno 2"
  "jose_main exact 2"
  "jose_matched fno 3"
  "jose_tuned fno 4"
)

worker "$GPU0" "${tasks0[@]}" >"$OUTPUT/logs/worker_gpu${GPU0}.log" 2>&1 &
pid0=$!
worker "$GPU1" "${tasks1[@]}" >"$OUTPUT/logs/worker_gpu${GPU1}.log" 2>&1 &
pid1=$!
wait "$pid0"
wait "$pid1"

python scripts/summarize_jose_ppo.py \
  --runs "$OUTPUT/runs" \
  --baseline-runs results/thf_rl_mix_tara/runs \
  --output "$OUTPUT"
