#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-$HOME/qlsgym_seqFNO_RL}
work=${QLSGYM_WORK:-$HOME/qlsgym_work}
gpu0=${GPU0:-0}
gpu1=${GPU1:-2}
attention_layers=${ATTENTION_LAYERS:?set ATTENTION_LAYERS to the promoted pilot depth}
tag=${TEMPORAL_TAG:-seqfno_temporal_full}
output=${OUTPUT:-results/thf_temporal_fno_full_audit}
cache=${CACHE_DIR:-$work/cache/temporal_column_exact}

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
    --epochs 100 --lr 3e-4 --modes 40 --hidden 96 --layers 3 \
    --d-model 128 --attention-heads 4 --attention-layers "$attention_layers" \
    --ff-dim 256 --attention-gate-init 0.075 \
    --identity-logit-bias 6.0 --column-tv-weight 4.0 --column-ce-weight 2.0 \
    --joint-tv-weight 2.0 --branch-mass-weight 8.0 --conditional-weight 4.0 \
    --infidelity-weight 1.0
}

worker() {
  local gpu=$1 parity=$2
  for block in $(seq "$parity" 2 11); do
    train_pair "$gpu" "$block" +
    train_pair "$gpu" "$block" -
  done
}

worker "$gpu0" 0 >"$output/logs/train_gpu${gpu0}.log" 2>&1 &
pid0=$!
worker "$gpu1" 1 >"$output/logs/train_gpu${gpu1}.log" 2>&1 &
pid1=$!
wait "$pid0"
wait "$pid1"

pairs=$(python - <<'PY'
print(','.join(f'{block}:{sigma}' for block in range(12) for sigma in ('+', '-')))
PY
)
python scripts/train_thf_column_fno.py manifest \
  --work "$work" --tag "$tag" --pairs "$pairs" --device cpu \
  >"$output/logs/manifest.log" 2>&1

audit_worker() {
  local gpu=$1 parity=$2
  for block in $(seq "$parity" 2 11); do
    for sigma in + -; do
      label=sp
      [[ "$sigma" == "-" ]] && label=sm
      [[ -f "$output/pairs/block${block}_${label}.json" ]] && continue
      CUDA_VISIBLE_DEVICES="$gpu" python scripts/audit_thf_fno_manifest.py pair \
        --manifest "$work/checkpoints/thf/$tag.json" --output "$output" \
        --block "$block" --sigma "$sigma" --device cuda:0
    done
  done
}

audit_worker "$gpu0" 0 >"$output/logs/audit_gpu${gpu0}.log" 2>&1 &
pid0=$!
audit_worker "$gpu1" 1 >"$output/logs/audit_gpu${gpu1}.log" 2>&1 &
pid1=$!
wait "$pid0"
wait "$pid1"
python scripts/audit_thf_fno_manifest.py summarize --output "$output" \
  >"$output/logs/summarize.log" 2>&1
