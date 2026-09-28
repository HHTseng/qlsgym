#!/bin/bash
# The GPU half of tests/test_cudaq_engine.py (CudaqEngine vs ExactEngine).
#
#   sbatch scripts/slurm/test_cudaq_engine.sbatch
#
#SBATCH --job-name=qlsgym_cudaq
#SBATCH --account=iaifi_lab
#SBATCH --partition=gpu_requeue,iaifi_gpu_requeue,gpu,iaifi_gpu
#SBATCH --gres=gpu:1
#SBATCH --requeue
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

# sbatch runs a spooled copy, so BASH_SOURCE points at /var/spool; name the repo explicitly
REPO="${QLSGYM_REPO:-/n/home02/josemm/RESEARCH/PROJECTS/MOLESQLS/qlsgym}"
source "$REPO/scripts/env.sh"
mkdir -p "$QLSGYM_WORK/logs"
exec > "${QLSGYM_WORK}/logs/cudaq_engine_${SLURM_JOB_ID:-local}.out" 2>&1

nvidia-smi -L
cd "$REPO"
python -m pytest tests/test_cudaq_engine.py -m gpu -q -s -rs
