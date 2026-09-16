#!/bin/bash
# Replication stages s01_physics + s02_data + s03_train, one array task per
# unique block.  The array bound is per molecule -- 6 for h3o, 12 for thf:
#
#   python -c "from replication.config import Config, resolve_blocks; \
#              print(resolve_blocks(Config(molecule='h3o')))"
#
#   QLSGYM_WORK=/your/scratch sbatch --array=0-5 replication/slurm/train_blocks.sbatch
#
# Disk: the materialised splits are ~35-40 GB for six blocks (s02_data --dry-run
# prints the exact figure).  Each task writes its own stage JSON under
# $QLSGYM_WORK/repl_array/; the aggregation commands are echoed when it finishes.
#
#SBATCH --job-name=qlsrepl_train
#SBATCH --account=iaifi_lab
#SBATCH --partition=gpu,iaifi_gpu,iaifi_gpu_requeue
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --array=0-5
#SBATCH --output=/n/holylabs/iaifi_lab/Lab/josemm/QLSGYM/logs/%x_%A_%a.out
#SBATCH --error=/n/holylabs/iaifi_lab/Lab/josemm/QLSGYM/logs/%x_%A_%a.err

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Set QLSGYM_WORK before env.sh so its default does not win; the #SBATCH log
# paths above are literal and must be edited to match if you move it.
export QLSGYM_WORK="${QLSGYM_WORK:-/n/holylabs/iaifi_lab/Lab/josemm/QLSGYM}"
source "$REPO/scripts/env.sh"
mkdir -p "$QLSGYM_WORK/logs" "$QLSGYM_WORK/runs" "$QLSGYM_WORK/repl_array"

MOLECULE="${MOLECULE:-h3o}"
SIGMA="${SIGMA:-+}"
TAG="${TAG:-repl}"
# production sizes (replication/config.py's defaults are the small ones)
NFREQ="${NFREQ:-24000}"
NINIT="${NINIT:-1000}"
NPAIRS="${NPAIRS:-32000}"
EPOCHS="${EPOCHS:-80}"

BLOCK="$(python -c "import sys; sys.path.insert(0, '$REPO'); \
from replication.config import Config, resolve_blocks; \
print(resolve_blocks(Config(molecule='$MOLECULE'))[${SLURM_ARRAY_TASK_ID}])")"

export QLSGYM_REPL_OUTPUTS="$QLSGYM_WORK/repl_array/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$QLSGYM_REPL_OUTPUTS"

nvidia-smi -L || true
echo "QLSGYM_WORK=$QLSGYM_WORK"
echo "QLSGYM_REPL_OUTPUTS=$QLSGYM_REPL_OUTPUTS"

# These sizes must match the aggregation pass, or it names a different split.
SIZES=(--set "molecule=$MOLECULE" --set "sigmas=$SIGMA" --set "tag=$TAG"
       --set "n_freq=$NFREQ" --set "n_init=$NINIT" --set "n_pairs=$NPAIRS"
       --set "epochs=$EPOCHS")
COMMON=(--device cuda --set "blocks=$BLOCK" "${SIZES[@]}")

for STAGE in s01_physics s02_data s03_train; do
  echo "+ python $REPO/replication/run.py $STAGE ${COMMON[*]}"
  srun python "$REPO/replication/run.py" "$STAGE" "${COMMON[@]}"
done

echo
echo "block $BLOCK done.  When the whole array has finished, write the real"
echo "OUTPUTS/replication/*.json on a login node (nothing is recomputed):"
echo "  source $REPO/scripts/env.sh"
for STAGE in s01_physics s02_data s03_train; do
  echo "  python $REPO/replication/run.py $STAGE ${SIZES[*]}"
done
