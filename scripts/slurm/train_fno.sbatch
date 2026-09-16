#!/bin/bash
# Array-train one FNO surrogate per block: ThF has 12 blocks (0-11), H3O has 10 (0-9).
#
#   sbatch --array=0-11 scripts/slurm/train_fno.sbatch
#   MOLECULE=h3o SIGMA=+ TAG=mix ALPHA=-2 sbatch --array=0-9 scripts/slurm/train_fno.sbatch
#
#SBATCH --job-name=qlsgym_fno
#SBATCH --account=iaifi_lab
#SBATCH --partition=gpu_requeue,iaifi_gpu_requeue,gpu,iaifi_gpu
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --array=0-11
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$REPO/scripts/env.sh"
mkdir -p "$QLSGYM_WORK/logs" "$QLSGYM_WORK/runs"
exec > "${QLSGYM_WORK}/logs/fno_${SLURM_JOB_NAME:-train}_${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-0}.out" 2>&1

MOLECULE="${MOLECULE:-thf}"
SIGMA="${SIGMA:-+}"
TAG="${TAG:-prod_a${ALPHA:-1.0}}"
ALPHA="${ALPHA:-1.0}"
FREQ_SAMPLING="${FREQ_SAMPLING:-mixture}"
NFREQ="${NFREQ:-24000}"
NPAIRS="${NPAIRS:-32000}"
EPOCHS="${EPOCHS:-80}"

b="$SLURM_ARRAY_TASK_ID"

nvidia-smi -L || true

srun python "$REPO/scripts/train_fno.py" \
  --molecule "$MOLECULE" --block "$b" --sigma "$SIGMA" --tag "$TAG" \
  --freq-sampling "$FREQ_SAMPLING" --alpha "$ALPHA" \
  --n-freq "$NFREQ" --n-init 1000 --n-pairs "$NPAIRS" \
  --epochs "$EPOCHS" --scheduler-step 8 --scheduler-gamma 0.75 \
  --val-n-freq 100 --val-n-init 100 --val-n-pairs 400 \
  --test-n-freq 100 --test-n-init 500 \
  --log-every 5
