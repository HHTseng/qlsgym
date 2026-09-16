#!/bin/bash
# Replication stage s05_planner: FNO-SPMP against exact and trivial, matched
# budget.  CPU only -- the planner is a closed-loop search, not a training job.
#
# One (arm, seed) unit per array task: --array=0-14 for the default five seeds
# (n_seeds x 3 otherwise), units 0-4 exact, 5-9 trivial, 10-14 fno.  Each unit
# checkpoints into $QLSGYM_WORK/repl_units/ as it finishes, so a killed task
# loses at most one unit and a re-submission resumes.
#
# Run the dry-run first -- it prints the projected cost for your configuration,
# which is per molecule.  H3O+ is <= 4.7 h per unit; ThF+ projects ~17 h, which
# does NOT fit the --time below.
#
#   python replication/run.py s05_planner --dry-run
#
#   QLSGYM_WORK=/your/scratch sbatch --array=0-14 replication/slurm/planner.sbatch
#
# The fno units need s03_train's manifest on disk (train_blocks.sbatch first).
#
#SBATCH --job-name=qlsrepl_planner
#SBATCH --account=iaifi_lab
#SBATCH --partition=sapphire,serial_requeue
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=16:00:00
#SBATCH --array=0-14
#SBATCH --output=/n/holylabs/iaifi_lab/Lab/josemm/QLSGYM/logs/%x_%A_%a.out
#SBATCH --error=/n/holylabs/iaifi_lab/Lab/josemm/QLSGYM/logs/%x_%A_%a.err

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Set QLSGYM_WORK before env.sh so its default does not win; the #SBATCH log
# paths above are literal and must be edited to match if you move it.
export QLSGYM_WORK="${QLSGYM_WORK:-/n/holylabs/iaifi_lab/Lab/josemm/QLSGYM}"
source "$REPO/scripts/env.sh"
mkdir -p "$QLSGYM_WORK/logs" "$QLSGYM_WORK/repl_array" "$QLSGYM_WORK/repl_units"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

MOLECULE="${MOLECULE:-h3o}"
TAG="${TAG:-repl}"
ROLLOUTS="${ROLLOUTS:-200}"
MAX_PULSES="${MAX_PULSES:-80}"

export QLSGYM_REPL_S05_UNITS="${SLURM_ARRAY_TASK_ID}"
# A subset run is a partial payload, so it must not land in the one output tree.
export QLSGYM_REPL_OUTPUTS="$QLSGYM_WORK/repl_array/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$QLSGYM_REPL_OUTPUTS"

ARGS=(--device cpu --set "molecule=$MOLECULE" --set "tag=$TAG"
      --set "n_rollouts=$ROLLOUTS" --set "max_pulses=$MAX_PULSES")

echo "QLSGYM_WORK=$QLSGYM_WORK"
echo "QLSGYM_REPL_OUTPUTS=$QLSGYM_REPL_OUTPUTS"
echo "QLSGYM_REPL_S05_UNITS=$QLSGYM_REPL_S05_UNITS"
echo "+ python $REPO/replication/run.py s05_planner ${ARGS[*]}"
srun python "$REPO/replication/run.py" s05_planner "${ARGS[@]}"

echo
echo "unit $SLURM_ARRAY_TASK_ID done; its checkpoint is under"
echo "  $QLSGYM_WORK/repl_units/s05_planner/"
echo "When the whole array has finished, write the real result on a login node"
echo "(seconds, nothing recomputed):"
echo "  source $REPO/scripts/env.sh"
echo "  unset QLSGYM_REPL_OUTPUTS"
echo "  QLSGYM_REPL_S05_UNITS=none python $REPO/replication/run.py s05_planner ${ARGS[*]}"
