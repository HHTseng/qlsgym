#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-$HOME/qlsgym_FNO_RL_optuna}
work=${QLSGYM_WORK:-$HOME/qlsgym_work}
tag=${COLUMN_TAG:-column_v1}
gpu=${GPU:-0}
output=${OUTPUT:-results/thf_column_fno_full_audit}

cd "$repo"
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate qlsgym
export PYTHONPATH="$repo/src:$repo/scripts${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$gpu"

for block in $(seq 0 11); do
  for sigma in + -; do
    python scripts/train_thf_column_fno.py train-pair \
      --work "$work" --tag "$tag" --block "$block" --sigma "$sigma" \
      --device cuda:0
  done
done

python scripts/train_thf_column_fno.py manifest \
  --work "$work" --tag "$tag" \
  --pairs '0:+,0:-,1:+,1:-,2:+,2:-,3:+,3:-,4:+,4:-,5:+,5:-,6:+,6:-,7:+,7:-,8:+,8:-,9:+,9:-,10:+,10:-,11:+,11:-' \
  --device cpu

manifest="$work/checkpoints/thf/$tag.json"
for block in $(seq 0 11); do
  for sigma in + -; do
    label=sp
    [[ "$sigma" == "-" ]] && label=sm
    [[ -f "$output/pairs/block${block}_${label}.json" ]] && continue
    python scripts/audit_thf_fno_manifest.py pair \
      --manifest "$manifest" --output "$output" --block "$block" \
      --sigma "$sigma" --device cuda:0
  done
done
python scripts/audit_thf_fno_manifest.py summarize --output "$output"
