#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-$HOME/qlsgym_seqFNO_RL}
work=${QLSGYM_WORK:-$HOME/qlsgym_work}
gpu0=${GPU0:-0}
gpu1=${GPU1:-2}
output=${OUTPUT:-results/thf_temporal_fno_pilot}
cache=${CACHE_DIR:-$work/cache/temporal_column_exact}
pairs='0:+,0:-,1:+,1:-,9:+,11:+'

cd "$repo"
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate qlsgym
export PYTHONPATH="$repo/src:$repo/scripts${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$output/logs" "$cache"

state() {
  python - "$output/pipeline_state.json" "$1" <<'PY'
import datetime, json, pathlib, sys
path = pathlib.Path(sys.argv[1])
path.write_text(json.dumps({
    "stage": sys.argv[2],
    "updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}, indent=2) + "\n")
PY
}

train_pair() {
  local gpu=$1 block=$2 sigma=$3 tag=$4 architecture=$5 attention_layers=$6
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/train_thf_column_fno.py train-pair \
    --work "$work" --tag "$tag" --block "$block" --sigma "$sigma" \
    --device cuda:0 --architecture "$architecture" --cache-dir "$cache" \
    --train-freq 512 --val-freq 96 --states 32 --batch-size 4 \
    --epochs 80 --lr 3e-4 --modes 40 --hidden 96 --layers 3 \
    --d-model 128 --attention-heads 4 --attention-layers "$attention_layers" \
    --ff-dim 256 --attention-gate-init 0.075 \
    --identity-logit-bias 6.0 --column-tv-weight 4.0 --column-ce-weight 2.0 \
    --joint-tv-weight 2.0 --branch-mass-weight 8.0 --conditional-weight 4.0 \
    --infidelity-weight 1.0
}

train_all_for_pair() {
  local gpu=$1 block=$2 sigma=$3
  train_pair "$gpu" "$block" "$sigma" seqfno_column_pilot column 1
  train_pair "$gpu" "$block" "$sigma" seqfno_temporal1_pilot temporal 1
  train_pair "$gpu" "$block" "$sigma" seqfno_temporal2_pilot temporal 2
}

state training
(
  train_all_for_pair "$gpu0" 0 +
  train_all_for_pair "$gpu0" 1 +
  train_all_for_pair "$gpu0" 11 +
) >"$output/logs/train_gpu${gpu0}.log" 2>&1 &
pid0=$!
(
  train_all_for_pair "$gpu1" 0 -
  train_all_for_pair "$gpu1" 1 -
  train_all_for_pair "$gpu1" 9 +
) >"$output/logs/train_gpu${gpu1}.log" 2>&1 &
pid1=$!
wait "$pid0"
wait "$pid1"

state manifests
for tag in seqfno_column_pilot seqfno_temporal1_pilot seqfno_temporal2_pilot; do
  python scripts/train_thf_column_fno.py manifest \
    --work "$work" --tag "$tag" --pairs "$pairs" --device cpu \
    >"$output/logs/manifest_${tag}.log" 2>&1
done

audit_pair() {
  local gpu=$1 block=$2 sigma=$3 tag=$4
  local audit="$output/$tag"
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/audit_thf_fno_manifest.py pair \
    --manifest "$work/checkpoints/thf/$tag.json" --output "$audit" \
    --block "$block" --sigma "$sigma" --device cuda:0 --n-freq 8 --n-init 32
}

audit_all_for_pair() {
  local gpu=$1 block=$2 sigma=$3
  for tag in seqfno_column_pilot seqfno_temporal1_pilot seqfno_temporal2_pilot; do
    audit_pair "$gpu" "$block" "$sigma" "$tag"
  done
}

state auditing
(
  audit_all_for_pair "$gpu0" 0 +
  audit_all_for_pair "$gpu0" 1 +
  audit_all_for_pair "$gpu0" 11 +
) >"$output/logs/audit_gpu${gpu0}.log" 2>&1 &
pid0=$!
(
  audit_all_for_pair "$gpu1" 0 -
  audit_all_for_pair "$gpu1" 1 -
  audit_all_for_pair "$gpu1" 9 +
) >"$output/logs/audit_gpu${gpu1}.log" 2>&1 &
pid1=$!
wait "$pid0"
wait "$pid1"

state summarizing
python scripts/summarize_thf_temporal_study.py \
  --study "downloaded_mix=$HOME/qlsgym_FNO_RL_optuna/results/thf_mix_structural_audit" \
  --study "column_v2=$HOME/qlsgym_FNO_RL_optuna/results/thf_column_fno_v2_full_audit" \
  --study "column_retrained=$output/seqfno_column_pilot" \
  --study "temporal_1layer=$output/seqfno_temporal1_pilot" \
  --study "temporal_2layer=$output/seqfno_temporal2_pilot" \
  --candidate temporal_1layer --candidate temporal_2layer \
  --output "$output/comparison" >"$output/logs/summarize.log" 2>&1
state complete
