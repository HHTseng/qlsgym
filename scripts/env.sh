source /n/sw/Miniforge3-26.1.0-0/etc/profile.d/conda.sh
conda activate fnorepl
export QLSGYM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export QLSGYM_WORK="${QLSGYM_WORK:-/n/netscratch/iaifi_lab/Lab/josemm/QLSGYM}"
export HEFF_ROOT="${HEFF_ROOT:-/n/home02/josemm/RESEARCH/PROJECTS/MOLESQLS/MolecularHamiltonians/heff}"
export PYTHONPATH="$QLSGYM_ROOT/src:$HEFF_ROOT:${PYTHONPATH:-}"
mkdir -p "$QLSGYM_WORK" 2>/dev/null || echo "warning: could not create QLSGYM_WORK=$QLSGYM_WORK (quota?); set it to a writable path" >&2
