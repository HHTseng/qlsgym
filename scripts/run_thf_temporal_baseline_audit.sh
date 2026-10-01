#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-$HOME/qlsgym_seqFNO_RL}
work=${QLSGYM_WORK:-$HOME/qlsgym_work}
gpu0=${GPU0:-0}
gpu1=${GPU1:-2}
output=${OUTPUT:-results/thf_temporal_baseline_audit}
long_audit=${LONG_AUDIT:-results/thf_temporal_fno_long_pilot/seqfno_temporal2_long_pilot}

cd "$repo"
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate qlsgym
export PYTHONPATH="$repo/src:$repo/scripts${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$output/logs"

audit_pair() {
  local gpu=$1 block=$2 sigma=$3 tag=$4 manifest=$5
  CUDA_VISIBLE_DEVICES="$gpu" python scripts/audit_thf_fno_manifest.py pair \
    --manifest "$manifest" --output "$output/$tag" --block "$block" \
    --sigma "$sigma" --device cuda:0 --n-freq 16 --n-init 64
}

audit_all_for_pair() {
  local gpu=$1 block=$2 sigma=$3
  audit_pair "$gpu" "$block" "$sigma" downloaded_mix "$work/checkpoints/thf/mix.json"
  audit_pair "$gpu" "$block" "$sigma" column_v2 "$work/checkpoints/thf/column_v2.json"
}

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

python scripts/summarize_thf_temporal_study.py \
  --study "downloaded_mix=$output/downloaded_mix" \
  --study "column_v2=$output/column_v2" \
  --study "temporal_2layer_long=$long_audit" \
  --candidate temporal_2layer_long \
  --output "$output/comparison" >"$output/logs/summarize.log" 2>&1
