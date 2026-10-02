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
  local parity=$1 gpu=$2
  for seed in 0 1 2 3 4; do
    if (( seed % 2 != parity )); then
      continue
    fi
    run_one "$gpu" jose_matched fno "$seed"
    run_one "$gpu" jose_main fno "$seed"
    run_one "$gpu" jose_tuned fno "$seed"
    if (( seed < 3 )); then
      run_one "$gpu" jose_main exact "$seed"
    fi
  done
}

mkdir -p "$OUTPUT/logs"
worker 0 "$GPU0" >"$OUTPUT/logs/worker_gpu${GPU0}.log" 2>&1 &
pid0=$!
worker 1 "$GPU1" >"$OUTPUT/logs/worker_gpu${GPU1}.log" 2>&1 &
pid1=$!
wait "$pid0"
wait "$pid1"

python scripts/summarize_jose_ppo.py \
  --runs "$OUTPUT/runs" \
  --baseline-runs results/thf_rl_mix_tara/runs \
  --output "$OUTPUT"
