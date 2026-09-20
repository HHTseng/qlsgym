#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/htseng/qlsgym_FNO_RL_optuna
OUTPUT="$ROOT/results/thf_rl_optuna_mix_sac_refine"
MANIFEST=/home/htseng/qlsgym_work/checkpoints/thf/mix.json

cd "$ROOT"
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate qlsgym
export QLSGYM_WORK=/home/htseng/qlsgym_work
export PYTHONPATH="$ROOT/src:$ROOT/scripts:${PYTHONPATH:-}"
mkdir -p "$OUTPUT/logs"

python scripts/refine_thf_mix_sac.py init --output "$OUTPUT"

CUDA_VISIBLE_DEVICES=0 python scripts/refine_thf_mix_sac.py optimize \
  --output "$OUTPUT" --manifest "$MANIFEST" --device cuda:0 \
  --worker-id 0 --target-trials 24 >"$OUTPUT/logs/optimize_gpu0.log" 2>&1 &
pid0=$!
CUDA_VISIBLE_DEVICES=2 python scripts/refine_thf_mix_sac.py optimize \
  --output "$OUTPUT" --manifest "$MANIFEST" --device cuda:0 \
  --worker-id 1 --target-trials 24 >"$OUTPUT/logs/optimize_gpu2.log" 2>&1 &
pid1=$!
failed=0
wait "$pid0" || failed=1
wait "$pid1" || failed=1
test "$failed" -eq 0

python scripts/refine_thf_mix_sac.py prepare --output "$OUTPUT" --top-k 4

CUDA_VISIBLE_DEVICES=0 python scripts/refine_thf_mix_sac.py run-queue \
  --output "$OUTPUT" --manifest "$MANIFEST" --device cuda:0 \
  --stage promotion --worker-id 0 >"$OUTPUT/logs/promotion_gpu0.log" 2>&1 &
pid0=$!
CUDA_VISIBLE_DEVICES=2 python scripts/refine_thf_mix_sac.py run-queue \
  --output "$OUTPUT" --manifest "$MANIFEST" --device cuda:0 \
  --stage promotion --worker-id 1 >"$OUTPUT/logs/promotion_gpu2.log" 2>&1 &
pid1=$!
failed=0
wait "$pid0" || failed=1
wait "$pid1" || failed=1
test "$failed" -eq 0

python scripts/refine_thf_mix_sac.py select --output "$OUTPUT"

CUDA_VISIBLE_DEVICES=0 python scripts/refine_thf_mix_sac.py run-queue \
  --output "$OUTPUT" --manifest "$MANIFEST" --device cuda:0 \
  --stage final --worker-id 0 >"$OUTPUT/logs/final_gpu0.log" 2>&1 &
pid0=$!
CUDA_VISIBLE_DEVICES=2 python scripts/refine_thf_mix_sac.py run-queue \
  --output "$OUTPUT" --manifest "$MANIFEST" --device cuda:0 \
  --stage final --worker-id 1 >"$OUTPUT/logs/final_gpu2.log" 2>&1 &
pid1=$!
failed=0
wait "$pid0" || failed=1
wait "$pid1" || failed=1
test "$failed" -eq 0

python scripts/refine_thf_mix_sac.py summarize --output "$OUTPUT"
