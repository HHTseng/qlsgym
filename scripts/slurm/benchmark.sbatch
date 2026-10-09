#!/bin/bash
#SBATCH --job-name=qls_bench
#SBATCH --account=iaifi_lab
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

: "${ARM:?set ARM to one of sweeping random planner-exact planner-fno rl-exact rl-fno}"
: "${MOL:?set MOL to h3o or thf}"
: "${NEPS:=1000}"
: "${DEVICE:=cpu}"
: "${LIBRARY:=physics_subset}"
: "${NPOOL:=16}"
: "${FNO_TAG:=}"
: "${ENGINE:=${QLSGYM_ENGINE:-exact}}"
export QLSGYM_ENGINE="$ENGINE"
: "${MAXPULSES:=}"
: "${EXTRA:=}"

OPT_ARGS=()
if [[ -n "$MAXPULSES" ]]; then OPT_ARGS+=(--max-pulses "$MAXPULSES"); fi
if [[ -n "$FNO_TAG" ]]; then OPT_ARGS+=(--fno-tag "$FNO_TAG"); fi

QLSGYM_REPO=/n/home02/josemm/RESEARCH/PROJECTS/MOLESQLS/qlsgym
source "${QLSGYM_REPO}/scripts/env.sh"
if [[ -n "${OUTROOT:-}" ]]; then export QLSGYM_OUTPUTS="$OUTROOT"; fi
LOGS="${QLSGYM_WORK}/logs"
mkdir -p "$LOGS"

if [[ -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  PART="${SLURM_ARRAY_TASK_ID}"
  SEED="${SEED:-777}"; SEED=$((SEED + 1000 * PART))
  PART_ARG=(--part "$PART")
else
  SEED="${SEED:-777}"
  PART_ARG=()
fi
STAMP="${SLURM_JOB_ID:-local}${SLURM_ARRAY_TASK_ID:+_$SLURM_ARRAY_TASK_ID}"
exec > "${LOGS}/bench_${MOL}_${ARM}_${STAMP}.out" 2>&1

echo "arm=$ARM molecule=$MOL episodes=$NEPS seed=$SEED device=$DEVICE library=$LIBRARY max_pulses=${MAXPULSES:-<molecule default>} fno_tag=${FNO_TAG:-<molecule default>} engine=$ENGINE"
[[ "$DEVICE" == cuda* ]] && nvidia-smi -L || true

srun python "${QLSGYM_REPO}/scripts/benchmark.py" run "$MOL" "$ARM" \
  --library "$LIBRARY" --n-episodes "$NEPS" --seed "$SEED" --device "$DEVICE" \
  --n-pool "$NPOOL" "${OPT_ARGS[@]}" "${PART_ARG[@]}" $EXTRA
