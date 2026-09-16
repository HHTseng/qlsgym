# source scripts/env.sh
source /n/sw/Miniforge3-26.1.0-0/etc/profile.d/conda.sh
conda activate fnorepl
export QLSGYM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Caches and trained artefacts.  netscratch is purged after 90 days, so nothing
# irreplaceable lives here.
export QLSGYM_WORK="${QLSGYM_WORK:-/n/netscratch/iaifi_lab/Lab/josemm/QLSGYM}"
# source repos, used ONLY by the parity tests (tests/test_parity_*.py) and make_manifest
export FNOREPL_ROOT="${FNOREPL_ROOT:-/n/home02/josemm/RESEARCH/PROJECTS/MOLESQLS/FNO_REPL}"
export THFFNO_ROOT="${THFFNO_ROOT:-/n/home02/josemm/RESEARCH/PROJECTS/MOLESQLS/FNO_NEW_MOLECULE/ThF_232_19}"
export FNOREPL_WORK="${FNOREPL_WORK:-/n/holylabs/iaifi_lab/Lab/josemm/FNO_REPL}"
export THFFNO_WORK="${THFFNO_WORK:-/n/holylabs/iaifi_lab/Lab/josemm/ThF_232_19}"
export HEFF_ROOT="${HEFF_ROOT:-/n/home02/josemm/RESEARCH/PROJECTS/MOLESQLS/MolecularHamiltonians/heff}"
export PYTHONPATH="$QLSGYM_ROOT/src:$FNOREPL_ROOT/src:$THFFNO_ROOT/src:${PYTHONPATH:-}"
mkdir -p "$QLSGYM_WORK" 2>/dev/null || echo "warning: could not create QLSGYM_WORK=$QLSGYM_WORK (quota?); set it to a writable path" >&2
