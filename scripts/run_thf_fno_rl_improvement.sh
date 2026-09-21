#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-$HOME/qlsgym_FNO_RL_optuna}
gpu=${GPU:-0}
export REPO="$repo"
export GPU="$gpu"
export CUDA_VISIBLE_DEVICES="$gpu"

cd "$repo"
mkdir -p results/thf_fno_rl_improvement/logs

bash scripts/run_thf_exact_rl_diagnosis.sh \
  2>&1 | tee results/thf_fno_rl_improvement/logs/exact_rl.log
bash scripts/run_thf_mix_structural_audit.sh \
  2>&1 | tee results/thf_fno_rl_improvement/logs/mix_structural_audit.log
