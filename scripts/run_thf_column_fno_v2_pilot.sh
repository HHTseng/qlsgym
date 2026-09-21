#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-$HOME/qlsgym_FNO_RL_optuna}
work=${QLSGYM_WORK:-$HOME/qlsgym_work}
tag=${COLUMN_TAG:-column_v2}
gpu0=${GPU0:-0}
gpu1=${GPU1:-2}
output=${OUTPUT:-results/thf_column_fno_v2_pilot_audit}
current=${CURRENT_AUDIT:-results/thf_mix_structural_audit}
comparison=${COMPARISON:-results/thf_column_fno_v2_pilot_comparison.json}

cd "$repo"
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate qlsgym
export PYTHONPATH="$repo/src:$repo/scripts${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$output/logs"

train_pair() {
  local gpu=$1
  local block=$2
  local sigma=$3
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/train_thf_column_fno.py train-pair \
    --work "$work" --tag "$tag" --block "$block" --sigma "$sigma" \
    --device cuda:0 --train-freq 768 --val-freq 96 --states 32 \
    --batch-size 4 --epochs 100 --lr 3e-4 --modes 50 --hidden 96 --layers 4 \
    --identity-logit-bias 6.0 --column-tv-weight 4.0 --column-ce-weight 2.0 \
    --joint-tv-weight 2.0 \
    --branch-mass-weight 8.0 --conditional-weight 4.0 --infidelity-weight 1.0
}

(
  train_pair "$gpu0" 0 +
  train_pair "$gpu0" 1 +
) >"$output/logs/train_gpu${gpu0}.log" 2>&1 &
pid0=$!
(
  train_pair "$gpu1" 0 -
  train_pair "$gpu1" 1 -
) >"$output/logs/train_gpu${gpu1}.log" 2>&1 &
pid1=$!
wait "$pid0"
wait "$pid1"

python scripts/train_thf_column_fno.py manifest \
  --work "$work" --tag "$tag" --pairs '0:+,0:-,1:+,1:-' --device cpu

manifest="$work/checkpoints/thf/$tag.json"
audit_pair() {
  local gpu=$1
  local block=$2
  local sigma=$3
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/audit_thf_fno_manifest.py pair \
    --manifest "$manifest" --output "$output" --block "$block" \
    --sigma "$sigma" --device cuda:0
}

(
  audit_pair "$gpu0" 0 +
  audit_pair "$gpu0" 1 +
) >"$output/logs/audit_gpu${gpu0}.log" 2>&1 &
pid0=$!
(
  audit_pair "$gpu1" 0 -
  audit_pair "$gpu1" 1 -
) >"$output/logs/audit_gpu${gpu1}.log" 2>&1 &
pid1=$!
wait "$pid0"
wait "$pid1"

set +e
python scripts/compare_thf_column_pilot.py \
  --current "$current" --candidate "$output" --output "$comparison"
decision=$?
set -e
if [[ $decision -ne 0 ]]; then
  echo "Column-v2 pilot did not satisfy the preregistered promotion rule."
  exit 2
fi
echo "Column-v2 pilot passed; full 24-pair training is allowed."
