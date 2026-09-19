#!/usr/bin/env bash
set -euo pipefail

mode=${1:?usage: run_thf_mix_tara_study.sh baseline|worker|extra_a|extra_b GPU}
gpu=${2:-0}

repo="$HOME/qlsgym_FNO_RL_agents"
work="$HOME/qlsgym_work"
python_bin="$HOME/.conda/envs/qlsgym/bin/python"
manifest="$work/checkpoints/thf/mix.json"
output="$repo/results/thf_rl_mix_tara"

export QLSGYM_WORK="$work"
export PYTHONPATH="$repo/src${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$gpu"
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export MPLBACKEND=Agg

mkdir -p "$output/logs"
cd "$repo"

run_one() {
  local agent=$1
  local seed=$2
  local result="$output/runs/final_${agent}_s${seed}.json"
  local log="$output/logs/${agent}_s${seed}.log"
  if [[ -f "$result" ]]; then
    echo "existing $result"
    return
  fi
  echo "$(date -Is) start agent=$agent seed=$seed physical_gpu=$gpu"
  "$python_bin" scripts/thf_rl_agents.py run \
    --agent "$agent" \
    --preset final \
    --manifest "$manifest" \
    --fno-tag mix \
    --min-manifest-coverage 1.0 \
    --seed "$seed" \
    --device cuda:0 \
    --eval-batch 128 \
    --output "$output" >"$log" 2>&1
  echo "$(date -Is) finish agent=$agent seed=$seed physical_gpu=$gpu"
}

if [[ "$mode" == baseline ]]; then
  for agent in sweeping random physics_elimination; do
    run_one "$agent" 0
  done
  exit 0
fi

if [[ "$mode" == extra_a ]]; then
  run_one ppo 1
  run_one sac_discrete 0
  exit 0
fi

if [[ "$mode" == extra_b ]]; then
  run_one sac_discrete 4
  run_one ddqn 3
  exit 0
fi

if [[ "$mode" != worker ]]; then
  echo "unknown mode: $mode" >&2
  exit 2
fi

index=0
for agent in ppo sac_discrete ddqn; do
  for seed in 0 1 2 3 4; do
    if (( index % 4 == gpu )); then
      run_one "$agent" "$seed"
    fi
    index=$((index + 1))
  done
done
