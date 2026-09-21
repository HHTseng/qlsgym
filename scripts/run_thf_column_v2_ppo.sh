#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-$HOME/qlsgym_FNO_RL_optuna}
output=${OUTPUT:-results/thf_column_fno_v2_ppo}
audit=${AUDIT:-results/thf_column_fno_v2_full_audit/summary.json}
gpu0=${GPU0:-0}
gpu1=${GPU1:-2}

cd "$repo"
passing=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["pairs_passing_all"])' "$audit")
if [[ "$passing" != "24" ]]; then
  echo "Refusing improved-FNO PPO: only $passing/24 structural audits pass." >&2
  exit 2
fi

source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate qlsgym
export PYTHONPATH="$repo/src:$repo/scripts${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$output/logs"

CUDA_VISIBLE_DEVICES="$gpu0" python scripts/train_thf_column_v2_ppo.py run \
  --output "$output" --device cuda:0 --seeds 0,2,4 \
  >"$output/logs/gpu${gpu0}.log" 2>&1 &
pid0=$!
CUDA_VISIBLE_DEVICES="$gpu1" python scripts/train_thf_column_v2_ppo.py run \
  --output "$output" --device cuda:0 --seeds 1,3 \
  >"$output/logs/gpu${gpu1}.log" 2>&1 &
pid1=$!
wait "$pid0"
wait "$pid1"

python scripts/train_thf_column_v2_ppo.py summarize --output "$output" \
  >"$output/logs/summarize.log" 2>&1
