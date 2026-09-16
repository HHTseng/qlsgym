#!/bin/bash
#SBATCH --job-name=qls_bench
#SBATCH --account=iaifi_lab
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#
# One benchmark arm.  Everything is chosen by environment variable:
# ARM MOL NEPS SEED DEVICE LIBRARY FNO_TAG NPOOL MAXPULSES EXTRA OUTROOT ENGINE
#
#   ARM=sweeping MOL=h3o sbatch --partition=sapphire --time=2:00:00 \
#       --cpus-per-task=4 --mem=16G scripts/slurm/benchmark.sbatch
#
#   ARM=rl-fno MOL=h3o DEVICE=cuda sbatch --partition=gpu \
#       --gres=gpu:nvidia_a100-sxm4-80gb:1 --time=12:00:00 \
#       --cpus-per-task=8 --mem=64G scripts/slurm/benchmark.sbatch
#
# A long arm is chunked over an array; merge the parts afterwards with
# `python scripts/benchmark.py merge <mol> <arm>`:
#
#   ARM=planner-fno MOL=h3o NEPS=100 sbatch --array=0-9 ... benchmark.sbatch
#
set -euo pipefail

: "${ARM:?set ARM to one of sweeping random planner-exact planner-fno rl-exact rl-fno}"
: "${MOL:=h3o}"
: "${NEPS:=1000}"
: "${DEVICE:=cpu}"
: "${LIBRARY:=physics_subset}"
: "${FNO_TAG:=prod}"
: "${NPOOL:=16}"
# the published arms were measured with the exact engine; ENGINE=auto opts into CUDA-Q
: "${ENGINE:=${QLSGYM_ENGINE:-exact}}"
export QLSGYM_ENGINE="$ENGINE"
: "${MAXPULSES:=}"
: "${EXTRA:=}"

MAXP_ARG=()
if [[ -n "$MAXPULSES" ]]; then
  MAXP_ARG=(--max-pulses "$MAXPULSES")
elif [[ "$EXTRA" != *--max-pulses* ]]; then
  echo "WARNING: MAXPULSES is unset and EXTRA carries no --max-pulses; the horizon falls back" >&2
  echo "WARNING: to molecule.task.max_pulses, which is shorter than one sweep of the library." >&2
fi

QLSGYM_REPO=/n/home02/josemm/RESEARCH/PROJECTS/MOLESQLS/qlsgym
source "${QLSGYM_REPO}/scripts/env.sh"
# OUTROOT sends results outside the repository (see outputs_dir).
if [[ -n "${OUTROOT:-}" ]]; then export QLSGYM_OUTPUTS="$OUTROOT"; fi
LOGS="${QLSGYM_WORK}/logs"
mkdir -p "$LOGS"

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  PART="${SLURM_ARRAY_TASK_ID}"
  SEED="${SEED:-777}"; SEED=$((SEED + 1000 * PART))   # distinct per part; merge checks this
  PART_ARG=(--part "$PART")
else
  SEED="${SEED:-777}"
  PART_ARG=()
fi
STAMP="${SLURM_JOB_ID:-local}${SLURM_ARRAY_TASK_ID:+_$SLURM_ARRAY_TASK_ID}"
exec > "${LOGS}/bench_${MOL}_${ARM}_${STAMP}.out" 2>&1

echo "arm=$ARM molecule=$MOL episodes=$NEPS seed=$SEED device=$DEVICE library=$LIBRARY max_pulses=${MAXPULSES:-<molecule default>} engine=$ENGINE"
[[ "$DEVICE" == cuda* ]] && nvidia-smi -L || true

srun python "${QLSGYM_REPO}/scripts/benchmark.py" run "$MOL" "$ARM" \
  --library "$LIBRARY" --n-episodes "$NEPS" --seed "$SEED" --device "$DEVICE" \
  --fno-tag "$FNO_TAG" --n-pool "$NPOOL" "${MAXP_ARG[@]}" "${PART_ARG[@]}" $EXTRA
