#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-$HOME/qlsgym_FNO_RL_optuna}
manifest=${MANIFEST:-$HOME/qlsgym_work/checkpoints/thf/mix.json}
output=${OUTPUT:-results/thf_mix_structural_audit}
gpu=${GPU:-0}

cd "$repo"
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate qlsgym
export PYTHONPATH="$repo/src:$repo/scripts${PYTHONPATH:+:$PYTHONPATH}"
export CUDA_VISIBLE_DEVICES="$gpu"

for block in $(seq 0 11); do
  for sigma in + -; do
    tag=sp
    [[ "$sigma" == "-" ]] && tag=sm
    destination="$output/pairs/block${block}_${tag}.json"
    [[ -f "$destination" ]] && continue
    python scripts/audit_thf_fno_manifest.py pair \
      --manifest "$manifest" --output "$output" --block "$block" \
      --sigma "$sigma" --device cuda:0
  done
done
python scripts/audit_thf_fno_manifest.py summarize --output "$output"
