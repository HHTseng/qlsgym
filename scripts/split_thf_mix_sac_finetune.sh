#!/usr/bin/env bash
set -euo pipefail

repo=${REPO:-$HOME/qlsgym_FNO_RL_optuna}
output=${OUTPUT:-results/thf_mix_sac_exact_finetune}

cd "$repo"
source /usr/local/anaconda3/etc/profile.d/conda.sh
conda activate qlsgym
export PYTHONPATH="$repo/src:$repo/scripts${PYTHONPATH:+:$PYTHONPATH}"

while [[ ! -f "$output/selection.json" ]]; do
  sleep 5
done

# The original runner immediately enters confirmation after selection. Stop it
# before assigning disjoint source seeds to the two GPU workers. Completed JSON
# records are durable and each worker skips any seed already written.
if tmux has-session -t thf_mix_sac_finetune 2>/dev/null; then
  tmux send-keys -t thf_mix_sac_finetune C-c
  sleep 3
  tmux kill-session -t thf_mix_sac_finetune 2>/dev/null || true
fi

tmux new-session -d -s thf_mix_sac_gpu0 \
  "cd $repo && GPU=0 MODE=confirm OUTPUT=$output bash scripts/run_thf_mix_sac_finetune_worker.sh 0 2 4 > $output/logs/gpu0.log 2>&1"
tmux new-session -d -s thf_mix_sac_gpu2 \
  "cd $repo && GPU=2 MODE=confirm OUTPUT=$output bash scripts/run_thf_mix_sac_finetune_worker.sh 1 3 > $output/logs/gpu2.log 2>&1"

while [[ $(find "$output/confirmation" -maxdepth 1 -name 'source_s*.json' 2>/dev/null | wc -l) -lt 5 ]]; do
  sleep 10
done
python scripts/fine_tune_thf_mix_sac.py summarize --output "$output"
