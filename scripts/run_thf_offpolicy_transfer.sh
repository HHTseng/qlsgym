#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUTPUT=${OUTPUT:-$ROOT/results/thf_offpolicy_transfer}
GPU0=${GPU0:-0}
GPU1=${GPU1:-2}

if [[ "$GPU0" == "$GPU1" ]]; then
  echo "GPU0 and GPU1 must identify different physical GPUs" >&2
  exit 2
fi

mkdir -p "$OUTPUT/screen/runs" "$OUTPUT/final/runs"

if [[ "${1:-run}" == "status" ]]; then
  echo "screen $(find "$OUTPUT/screen/runs" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l | tr -d ' ')/12"
  [[ -f "$OUTPUT/selection.json" ]] && echo "selection complete" || echo "selection pending"
  echo "final $(find "$OUTPUT/final/runs" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l | tr -d ' ')/10"
  [[ -f "$OUTPUT/summary.json" ]] && echo "summary complete" || echo "summary pending"
  exit 0
fi

MANIFEST=${MANIFEST:-${QLSGYM_WORK:?set QLSGYM_WORK}/checkpoints/thf/mix.json}
mkdir -p "$OUTPUT/screen/logs" "$OUTPUT/final/logs"

run_one() {
  local stage=$1
  local agent=$2
  local profile=$3
  local seed=$4
  local physical_gpu=$5
  local evaluation=$6
  local stage_output="$OUTPUT/$stage"
  local stem="final_${agent}_${profile}_fno_s${seed}"
  local record="$stage_output/runs/$stem.json"
  local log="$stage_output/logs/$stem.log"
  if [[ -f "$record" ]]; then
    echo "skip $stage $profile seed=$seed"
    return
  fi
  echo "run $stage $profile seed=$seed gpu=$physical_gpu"
  CUDA_VISIBLE_DEVICES="$physical_gpu" python "$ROOT/scripts/thf_rl_agents.py" run \
    --agent "$agent" --preset final \
    --offpolicy-profile "$profile" --offpolicy-selection fno \
    --snapshot-count 8 --snapshot-eval-episodes 256 --snapshot-eval-seed 31001 \
    --manifest "$MANIFEST" --fno-tag mix --min-manifest-coverage 1.0 \
    --seed "$seed" --eval-seed 20001 --eval-batch 128 \
    --evaluation-dynamics "$evaluation" --device cuda:0 \
    --output "$stage_output" >"$log" 2>&1
  echo "done $stage $profile seed=$seed gpu=$physical_gpu"
}

screen_worker() {
  local physical_gpu=$1
  local seed=$2
  run_one screen sac_discrete sac_refined_1m "$seed" "$physical_gpu" fno
  run_one screen sac_discrete sac_refined_2m "$seed" "$physical_gpu" fno
  run_one screen ddqn ddqn_optuna_1m "$seed" "$physical_gpu" fno
  run_one screen ddqn ddqn_optuna_2m "$seed" "$physical_gpu" fno
  run_one screen ddqn ddqn_scaled20_1m "$seed" "$physical_gpu" fno
  run_one screen ddqn ddqn_raw20_1m "$seed" "$physical_gpu" fno
}

screen_worker "$GPU0" 100 &
worker0=$!
screen_worker "$GPU1" 101 &
worker1=$!
failed=0
wait "$worker0" || failed=1
wait "$worker1" || failed=1
if ((failed)); then
  echo "a screening worker failed; inspect $OUTPUT/screen/logs" >&2
  exit 4
fi

python "$ROOT/scripts/summarize_thf_offpolicy_transfer.py" select \
  --runs "$OUTPUT/screen/runs" --output "$OUTPUT/selection.json"

mapfile -t selected < <(python - "$OUTPUT/selection.json" <<'PY'
import json, sys
profiles = json.load(open(sys.argv[1]))["selected_profiles"]
print(profiles["sac_discrete"])
print(profiles["ddqn"])
PY
)
if [[ ${#selected[@]} -ne 2 ]]; then
  echo "selection did not produce one SAC and one DDQN profile" >&2
  exit 3
fi

final_worker0() {
  run_one final sac_discrete "${selected[0]}" 0 "$GPU0" both
  run_one final sac_discrete "${selected[0]}" 2 "$GPU0" both
  run_one final sac_discrete "${selected[0]}" 4 "$GPU0" both
  run_one final ddqn "${selected[1]}" 1 "$GPU0" both
  run_one final ddqn "${selected[1]}" 3 "$GPU0" both
}

final_worker1() {
  run_one final sac_discrete "${selected[0]}" 1 "$GPU1" both
  run_one final sac_discrete "${selected[0]}" 3 "$GPU1" both
  run_one final ddqn "${selected[1]}" 0 "$GPU1" both
  run_one final ddqn "${selected[1]}" 2 "$GPU1" both
  run_one final ddqn "${selected[1]}" 4 "$GPU1" both
}

final_worker0 &
worker0=$!
final_worker1 &
worker1=$!
failed=0
wait "$worker0" || failed=1
wait "$worker1" || failed=1
if ((failed)); then
  echo "a final worker failed; inspect $OUTPUT/final/logs" >&2
  exit 5
fi

python "$ROOT/scripts/summarize_thf_offpolicy_transfer.py" summarize \
  --runs "$OUTPUT/final/runs" \
  --baseline-runs "$ROOT/results/thf_rl_mix_tara/runs" \
  --selection "$OUTPUT/selection.json" --output "$OUTPUT"

echo "off-policy transfer study complete"
