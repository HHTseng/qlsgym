#!/usr/bin/env bash
# Run inside tmux. Two GPU queues, locked full-grid aggregation, then publish.
set -euo pipefail

STUDY_REPO=$(cd "$(dirname "$0")/.." && pwd)
STUDY_WORK=${QLSGYM_WORK:-/home/htseng/Downloads/qlsgym_work}
STUDY_PYTHON=${STUDY_PYTHON:-/home/htseng/anaconda3/envs/qlsgym/bin/python}
STUDY_TAG=${STUDY_TAG:-rlprod120v2}
STUDY_OUTPUT="$STUDY_REPO/results/thf_rl_final"
export QLSGYM_WORK="$STUDY_WORK"
export PYTHONPATH="$STUDY_REPO/src:${PYTHONPATH:-}"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg
cd "$STUDY_REPO"
mkdir -p "$STUDY_WORK/logs" "$STUDY_OUTPUT/logs"

if [[ ${1:-coordinator} == worker ]]; then
  STUDY_GPU=$2
  # Assign whole agent/seed jobs to a GPU, no concurrent jobs on one card.
  STUDY_INDEX=0
  for STUDY_AGENT in ppo sac_discrete ddqn; do
    for STUDY_SEED in 0 1 2 3 4; do
      if (( STUDY_INDEX % 2 == STUDY_GPU )); then
        STUDY_RESULT="$STUDY_OUTPUT/runs/final_${STUDY_AGENT}_s${STUDY_SEED}.json"
        if [[ ! -f "$STUDY_RESULT" ]]; then
          echo "$(date -Is) start $STUDY_AGENT seed=$STUDY_SEED gpu=$STUDY_GPU"
          "$STUDY_PYTHON" scripts/thf_rl_agents.py run --agent "$STUDY_AGENT" \
            --preset final --manifest "$STUDY_WORK/checkpoints/thf/$STUDY_TAG.json" \
            --fno-tag "$STUDY_TAG" --min-manifest-coverage 1.0 --seed "$STUDY_SEED" \
            --device "cuda:$STUDY_GPU" --eval-batch 128 --output "$STUDY_OUTPUT" \
            > "$STUDY_OUTPUT/logs/${STUDY_AGENT}_s${STUDY_SEED}.log" 2>&1
        else
          echo "existing $STUDY_RESULT; final aggregator will verify its full contract"
        fi
      fi
      STUDY_INDEX=$((STUDY_INDEX + 1))
    done
  done
  exit 0
fi

echo "$(date -Is) waiting for all production FNO checkpoints"
while tmux has-session -t thf_fno_full_sp 2>/dev/null || tmux has-session -t thf_fno_full_sm 2>/dev/null; do
  sleep 60
done
# Failure stops the pipeline; never substitute exact fallback for missing FNOs.
"$STUDY_PYTHON" scripts/collect_thf_fno.py --work "$STUDY_WORK" --tag "$STUDY_TAG" --require-complete
"$STUDY_PYTHON" scripts/prepare_thf_fno.py manifest --work "$STUDY_WORK" --tag "$STUDY_TAG" --sigmas both --device cpu
cp "$STUDY_WORK/checkpoints/thf/$STUDY_TAG.json" "results/thf_fno_blocks/manifest_$STUDY_TAG.json"

echo "$(date -Is) full paper-style held-out audits (100 frequencies x 500 initial states per stratum)"
"$STUDY_PYTHON" scripts/thf_fno_paper_metrics.py --work "$STUDY_WORK" --tag "$STUDY_TAG" \
  --block 0 --sigma + --device cuda:0 > "$STUDY_WORK/logs/paper_block0_sp.log" 2>&1 &
STUDY_AUDIT_PID=$!
"$STUDY_PYTHON" scripts/thf_fno_paper_metrics.py --work "$STUDY_WORK" --tag "$STUDY_TAG" \
  --block 1 --sigma - --device cuda:1 > "$STUDY_WORK/logs/paper_block1_sm.log" 2>&1
wait "$STUDY_AUDIT_PID"

echo "$(date -Is) final baseline evaluation"
for STUDY_AGENT in sweeping random physics_elimination; do
  if [[ ! -f "$STUDY_OUTPUT/runs/final_${STUDY_AGENT}_s0.json" ]]; then
    "$STUDY_PYTHON" scripts/thf_rl_agents.py run --agent "$STUDY_AGENT" --preset final \
      --manifest "$STUDY_WORK/checkpoints/thf/$STUDY_TAG.json" --fno-tag "$STUDY_TAG" \
      --min-manifest-coverage 1.0 --seed 0 --device cuda:0 --eval-batch 128 --output "$STUDY_OUTPUT" \
      > "$STUDY_OUTPUT/logs/${STUDY_AGENT}_s0.log" 2>&1
  fi
done
echo "$(date -Is) start full learned-agent grid"
bash scripts/run_thf_final_study.sh worker 0 > "$STUDY_WORK/logs/final_gpu0.log" 2>&1 &
STUDY_WORKER_PID=$!
bash scripts/run_thf_final_study.sh worker 1 > "$STUDY_WORK/logs/final_gpu1.log" 2>&1
wait "$STUDY_WORKER_PID"

"$STUDY_PYTHON" scripts/summarize_thf_study.py
"$STUDY_PYTHON" scripts/write_thf_report.py
"$STUDY_PYTHON" -m pytest -q -m 'not slow'
git add README.md docs/THF_RESULTS.md results/thf_fno_blocks results/thf_fno_validation results/thf_rl_final
if ! git diff --cached --quiet; then
  git commit -m "Report full ThF FNO validation and five-seed RL ranking"
fi
git push origin HEAD:FNO_RL_agents
echo "$(date -Is) complete; final ranking and report pushed"
