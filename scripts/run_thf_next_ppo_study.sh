#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
OUTPUT=${OUTPUT:-$ROOT/results/thf_next_ppo}
GPU0=${GPU0:-0}
GPU1=${GPU1:-2}

if [[ "$GPU0" == "$GPU1" ]]; then
  echo "GPU0 and GPU1 must identify different physical GPUs" >&2
  exit 2
fi

mkdir -p "$OUTPUT/screen/runs" "$OUTPUT/final/runs"

if [[ "${1:-run}" == "status" ]]; then
  echo "screen $(find "$OUTPUT/screen/runs" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l | tr -d ' ')/16"
  if [[ -f "$OUTPUT/selection.json" ]]; then
    python - "$OUTPUT/selection.json" <<'PY'
import json, sys
data = json.load(open(sys.argv[1]))
print("promoted", ",".join(data["promoted_profiles"]))
PY
  else
    echo "promoted pending"
  fi
  echo "final $(find "$OUTPUT/final/runs" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l | tr -d ' ')/10"
  [[ -f "$OUTPUT/summary.json" ]] && echo "summary complete" || echo "summary pending"
  exit 0
fi

MANIFEST=${MANIFEST:-${QLSGYM_WORK:?set QLSGYM_WORK}/checkpoints/thf/mix.json}
mkdir -p "$OUTPUT/screen/logs" "$OUTPUT/final/logs"

run_one() {
  local stage=$1
  local profile=$2
  local seed=$3
  local physical_gpu=$4
  local evaluation=$5
  local stage_output="$OUTPUT/$stage"
  local stem="final_ppo_${profile}_fno_s${seed}"
  local record="$stage_output/runs/$stem.json"
  local log="$stage_output/logs/$stem.log"
  if [[ -f "$record" ]]; then
    echo "skip $stage $profile seed=$seed"
    return
  fi
  echo "run $stage $profile seed=$seed gpu=$physical_gpu"
  CUDA_VISIBLE_DEVICES="$physical_gpu" python "$ROOT/scripts/thf_rl_agents.py" run \
    --agent ppo --preset final \
    --ppo-profile "$profile" --ppo-selection fno \
    --manifest "$MANIFEST" --fno-tag mix --min-manifest-coverage 1.0 \
    --seed "$seed" --eval-seed 20001 --eval-batch 128 \
    --evaluation-dynamics "$evaluation" --device cuda:0 \
    --output "$stage_output" >"$log" 2>&1
  echo "done $stage $profile seed=$seed gpu=$physical_gpu"
}

screen_worker() {
  local physical_gpu=$1
  local seed=$2
  local profile
  for profile in \
    jose_qmdp jose_aux_01 jose_aux_03 \
    qmdp_tuned_250k qmdp_tuned_1m qmdp_tuned_2m \
    qmdp_lr10_1m qmdp_lr17_1m
  do
    run_one screen "$profile" "$seed" "$physical_gpu" fno
  done
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

python "$ROOT/scripts/summarize_thf_next_ppo.py" select \
  --runs "$OUTPUT/screen/runs" \
  --output "$OUTPUT/selection.json"

mapfile -t promoted < <(python - "$OUTPUT/selection.json" <<'PY'
import json, sys
for profile in json.load(open(sys.argv[1]))["promoted_profiles"]:
    print(profile)
PY
)
if [[ ${#promoted[@]} -ne 2 ]]; then
  echo "selection did not produce two profiles" >&2
  exit 3
fi

final_worker0() {
  run_one final "${promoted[0]}" 0 "$GPU0" both
  run_one final "${promoted[0]}" 2 "$GPU0" both
  run_one final "${promoted[0]}" 4 "$GPU0" both
  run_one final "${promoted[1]}" 1 "$GPU0" both
  run_one final "${promoted[1]}" 3 "$GPU0" both
}

final_worker1() {
  run_one final "${promoted[0]}" 1 "$GPU1" both
  run_one final "${promoted[0]}" 3 "$GPU1" both
  run_one final "${promoted[1]}" 0 "$GPU1" both
  run_one final "${promoted[1]}" 2 "$GPU1" both
  run_one final "${promoted[1]}" 4 "$GPU1" both
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

python "$ROOT/scripts/summarize_thf_next_ppo.py" summarize \
  --runs "$OUTPUT/final/runs" \
  --baseline-runs "$ROOT/results/thf_jose_ppo_comparison/runs" \
  --selection "$OUTPUT/selection.json" \
  --output "$OUTPUT"

echo "next-PPO study complete"
