#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-$HOME/qlsgym_FNO_RL_optuna}
work=${QLSGYM_WORK:-$HOME/qlsgym_work}
tag=${COLUMN_TAG:-column_v1}
gpu=${GPU:-0}
output=${OUTPUT:-results/thf_column_fno_pilot_audit}
current=${CURRENT_AUDIT:-results/thf_mix_structural_audit}
comparison=${COMPARISON:-results/thf_column_fno_pilot_comparison.json}

cd "$repo"
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate qlsgym
export PYTHONPATH="$repo/src:$repo/scripts${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$gpu"

for pair in 0:+ 0:- 1:+ 1:-; do
  block=${pair%%:*}
  sigma=${pair#*:}
  python scripts/train_thf_column_fno.py train-pair \
    --work "$work" --tag "$tag" --block "$block" --sigma "$sigma" \
    --device cuda:0
done

python scripts/train_thf_column_fno.py manifest \
  --work "$work" --tag "$tag" --pairs '0:+,0:-,1:+,1:-' --device cpu

manifest="$work/checkpoints/thf/$tag.json"
for pair in 0:+ 0:- 1:+ 1:-; do
  block=${pair%%:*}
  sigma=${pair#*:}
  python scripts/audit_thf_fno_manifest.py pair \
    --manifest "$manifest" --output "$output" --block "$block" \
    --sigma "$sigma" --device cuda:0
done

set +e
python scripts/compare_thf_column_pilot.py \
  --current "$current" --candidate "$output" --output "$comparison"
decision=$?
set -e
if [[ $decision -ne 0 ]]; then
  echo "Transfer-column pilot did not satisfy the preregistered promotion rule."
  exit 2
fi
echo "Transfer-column pilot passed; full 24-pair training is allowed."
