#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-$HOME/qlsgym_seqFNO_RL}
work=${QLSGYM_WORK:-$HOME/qlsgym_work}
gpu0=${GPU0:-0}
gpu1=${GPU1:-2}
tag=${TEMPORAL_TAG:-seqfno_temporal2_long_pilot}
output=${OUTPUT:-results/thf_temporal_fno_long_pilot}
cache=${CACHE_DIR:-$work/cache/temporal_column_exact}
derivative_weight=${DERIVATIVE_WEIGHT:-0.0}
spectral_weight=${SPECTRAL_WEIGHT:-0.0}
pairs='0:+,0:-,1:+,1:-,9:+,11:+'

cd "$repo"
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate qlsgym
export PYTHONPATH="$repo/src:$repo/scripts${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$output/logs" "$cache"

train_pair() {
  local gpu=$1 block=$2 sigma=$3
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/train_thf_column_fno.py train-pair \
    --work "$work" --tag "$tag" --block "$block" --sigma "$sigma" \
    --device cuda:0 --architecture temporal --cache-dir "$cache" \
    --train-freq 768 --val-freq 96 --states 32 --batch-size 4 \
    --epochs 100 --lr 3e-4 --modes 50 --hidden 96 --layers 4 \
    --d-model 128 --attention-heads 4 --attention-layers 2 --ff-dim 256 \
    --attention-gate-init 0.075 --identity-logit-bias 6.0 \
    --column-tv-weight 4.0 --column-ce-weight 2.0 --joint-tv-weight 2.0 \
    --branch-mass-weight 8.0 --conditional-weight 4.0 --infidelity-weight 1.0 \
    --derivative-weight "$derivative_weight" --spectral-weight "$spectral_weight"
}

(
  train_pair "$gpu0" 0 +
  train_pair "$gpu0" 1 +
  train_pair "$gpu0" 11 +
) >"$output/logs/train_gpu${gpu0}.log" 2>&1 &
pid0=$!
(
  train_pair "$gpu1" 0 -
  train_pair "$gpu1" 1 -
  train_pair "$gpu1" 9 +
) >"$output/logs/train_gpu${gpu1}.log" 2>&1 &
pid1=$!
wait "$pid0"
wait "$pid1"

python scripts/train_thf_column_fno.py manifest \
  --work "$work" --tag "$tag" --pairs "$pairs" --device cpu \
  >"$output/logs/manifest.log" 2>&1

audit_pair() {
  local gpu=$1 block=$2 sigma=$3
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/audit_thf_fno_manifest.py pair \
    --manifest "$work/checkpoints/thf/$tag.json" --output "$output/$tag" \
    --block "$block" --sigma "$sigma" --device cuda:0 --n-freq 16 --n-init 64
}

(
  audit_pair "$gpu0" 0 +
  audit_pair "$gpu0" 1 +
  audit_pair "$gpu0" 11 +
) >"$output/logs/audit_gpu${gpu0}.log" 2>&1 &
pid0=$!
(
  audit_pair "$gpu1" 0 -
  audit_pair "$gpu1" 1 -
  audit_pair "$gpu1" 9 +
) >"$output/logs/audit_gpu${gpu1}.log" 2>&1 &
pid1=$!
wait "$pid0"
wait "$pid1"

python scripts/summarize_thf_temporal_study.py \
  --study "downloaded_mix=$HOME/qlsgym_FNO_RL_optuna/results/thf_mix_structural_audit" \
  --study "column_v2=$HOME/qlsgym_FNO_RL_optuna/results/thf_column_fno_v2_full_audit" \
  --study "temporal_2layer_long=$output/$tag" \
  --candidate temporal_2layer_long \
  --output "$output/comparison" >"$output/logs/summarize.log" 2>&1
