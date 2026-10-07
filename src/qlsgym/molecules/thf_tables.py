"""Generate the ThF+ level and sigma+ two-photon coupling tables in data/thf/ from heff.

This is the only module that talks to heff (the effective-Hamiltonian package at $HEFF_ROOT, which
scripts/env.sh puts on PYTHONPATH). It is a build-time dependency: scripts/build_thf_tables.py calls
build_tables() once and writes two plain-text tables, and qlsgym.molecules.thf reads those files.
Regenerate the tables whenever a setting here or heff's parameter set changes; their headers carry
the heff git revision, its spec hash and the full parameter table. docs/thf_system.md explains
every modelling choice below.

1. Levels. heff builds H = sum_k c_k M_k in the Hund's case (c) basis |J, Omega = +-1, F, m_F>
   (I = 1/2 for 19F), J = 1..J_max, at static collinear fields (E_z, B_z). Collinear fields conserve
   signed m_F, so each m_F block is diagonalised separately. Every eigenstate is labelled by its
   dominant ket (J, F, m_F), its exact parity <P> = +-1 (exact at E_z = 0), and an energy-ordered
   index xi inside its (J, parity, m_F) class: the 5-tuple (J, F, parity, m_F, xi).

2. Two-photon couplings, model A. No Raman rate inside X 3Delta1 has been measured, and the
   candidate intermediate states disagree by a factor of two in energy between ab-initio papers, so
   the simplest well-defined model is used:

       T_eff = (d . eps2*) P_int (d . eps1) / Delta

   with P_int the projector onto a single Omega = 1 electronic state, summed over all its
   rotational and hyperfine sublevels |J'', Omega'' = Omega, F'', m''>, J'' <= J_max + 1, at a
   common detuning Delta. Consequences (tests/test_thf_tables.py checks each):

   * each E1 leg is Omega-diagonal (q = 0), so it is exactly heff.elements_c.dipole_geometry;
   * the operator is parity-even with no Delta-Omega = +-2 part, so in the ket basis the
     Omega-doublet components e and f see identical couplings. In the eigenbasis they differ at the
     1e-3 level, because heff's Zeeman tensor term G_Delta = (G_xx - G_yy)/2 is itself a
     Delta-Omega = +-2 operator that mixes Delta F = +-1 with opposite sign on e and f;
   * the antisymmetric rank-1 part vanishes identically, which makes the sigma- channel the exact
     reverse of the sigma+ channel (the property qlsgym.physics.hamiltonian relies on);
   * the reduced dipole and Delta are one overall scale, fixed by setting REFERENCE_TRANSITION to
     |Omega|/2pi = REFERENCE_RABI_KHZ.

   It is an approximation on two counts: a real experiment resolves the intermediate rotational
   ladder (B_int ~ 7 GHz against a few-GHz detuning), and Omega = 0 intermediates would add a
   Delta-Omega = +-2 amplitude that makes e and f inequivalent in the coupling.

3. Channel labelling. The sigma+ channel is the set of couplings with Delta m_F = +1. Which Jones
   vector on the second beam realises that sign depends on whether photon 2 is emitted or absorbed,
   so the channel is selected by Delta m_F and the Jones pair used is recorded in the table header.

Units: heff works in MHz; the tables store level energies in MHz and couplings in rad/ms.
"""

from __future__ import annotations

import datetime as _dt
import os
import subprocess
from dataclasses import dataclass

import numpy as np

from ..spec import TWO_PI
from .thf import B_FIELD_GAUSS, E_FIELD_V_PER_CM, J_MAX, REFERENCE_RABI_KHZ, table_tag

__all__ = ["REFERENCE_TRANSITION", "JONES", "Levels", "Couplings", "build_levels", "two_photon_amplitudes",
           "sigma_plus_couplings", "build_tables", "write_tables", "table_files"]


def _heff():
    """Import heff lazily so the rest of qlsgym never needs it."""
    try:
        import heff
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on the environment
        raise ModuleNotFoundError("heff is not importable; `source scripts/env.sh` puts $HEFF_ROOT on "
                                  "PYTHONPATH (needed only to build the tables)") from exc
    return heff


# (J, F, parity, m_F, xi) of the initial and final state whose sigma+ coupling is normalised to
# |Omega|/2pi = REFERENCE_RABI_KHZ: the m_F = +1/2 -> +3/2 rung of the J = 1, F = 3/2 ladder on the
# parity-+ doublet component, which at J = 1 is f, the upper one (heff puts e, parity +(-1)^J,
# below f at every J). The e partner's same rung comes out at 3.0007 kHz.
REFERENCE_TRANSITION = (
    (1, 1.5, "+", 0.5, 0),
    (1, 1.5, "+", 1.5, 0),
)


# Levels


@dataclass(frozen=True)
class Levels:
    """Eigenstates of the field-dressed molecular Hamiltonian.

    labels[n] = (J, F, parity, m_F, xi); energies_MHz ascending; vectors[:, n] is eigenstate n in
    the global kets basis; ef[n] is the Brown-1975 e/f label, kept for the table only.
    """

    kets: np.ndarray
    labels: list
    energies_MHz: np.ndarray
    vectors: np.ndarray
    ef: list
    spec_hash: str
    params_table: str


def build_levels(j_max: int = J_MAX, b_gauss: float = B_FIELD_GAUSS, e_v_per_cm: float = E_FIELD_V_PER_CM,
                 pset=None) -> Levels:
    """Diagonalise heff's Hamiltonian at (E_z, B_z) and label the eigenstates.

    pset replaces heff's thf_v1() parameter set (e.g. a control with G_xx = G_yy built with
    ParamSet.with_). The shipped tables always use the default.
    """
    heff = _heff()
    spec = heff.thf_spec(J_max=j_max)
    pset = heff.thf_v1() if pset is None else pset
    ctx = heff.ctx_from(spec, pset)
    kets = heff.enumerate_kets(spec)
    blocks = heff.block_by_mF(kets)
    n = len(kets)

    energies = np.empty(n)
    vectors = np.zeros((n, n), dtype=np.complex128)
    parity = np.empty(n)
    dom = np.empty(n, dtype=kets.dtype)
    efs: list = [None] * n
    spec_hash = None
    col = 0
    for mf in sorted(blocks.index):
        idx = np.asarray(blocks.index[mf])
        sub = kets[idx]
        tm = heff.build_term_matrices(sub, ctx)
        spec_hash = spec_hash or tm.manifest["spec_hash"]
        # heff's term list carries the whole Zeeman tensor diag(G_xx, G_yy, G_zz), so the
        # parity-dependent part G_Delta is already in here
        h = heff.hamiltonian(tm, pset, {"E_z": e_v_per_cm, "B_z": b_gauss})
        w, v = np.linalg.eigh(h)
        pop = heff.parity_operator(sub, ctx.S, ell=0.0, s=0.0)
        lab = heff.label_lines(sub, v, ctx.S, rule=pset.conventions.ef_rule, ell=0.0, s=0.0)
        for s in range(len(w)):
            energies[col] = w[s]
            vectors[idx, col] = v[:, s]
            parity[col] = float(np.real(np.vdot(v[:, s], pop @ v[:, s])))
            dom[col] = sub[int(np.argmax(np.abs(v[:, s])))]
            efs[col] = lab[s]["ef"]
            col += 1
    assert col == n

    order = np.argsort(energies, kind="stable")
    energies, vectors, parity, dom = energies[order], vectors[:, order], parity[order], dom[order]
    efs = [efs[int(k)] for k in order]

    if np.max(np.abs(np.abs(parity) - 1.0)) > 1e-9:
        raise ValueError("parity is not exact; is E_z != 0? The block keys (J, parity) assume exact parity.")

    # xi: energy-ordered index inside each (J, parity, m_F) class
    counters: dict = {}
    labels = []
    for k in range(n):
        J = int(round(float(dom["J"][k])))
        F = float(dom["F"][k])
        mF = float(dom["mF"][k])
        p = "+" if parity[k] > 0 else "-"
        xi = counters.get((J, p, mF), 0)
        counters[(J, p, mF)] = xi + 1
        labels.append((J, F, p, mF, xi))

    return Levels(kets=kets, labels=labels, energies_MHz=energies, vectors=vectors, ef=efs,
                  spec_hash=spec_hash, params_table=pset.table())


# Two-photon amplitudes, coupling model A

# Cartesian Jones vectors; sigma+ = -(x + i y)/sqrt2 is the Condon-Shortley e_{+1}, pi = z
JONES = {
    "pi": np.array([0.0, 0.0, 1.0], dtype=complex),
    "sigma+": np.array([-1.0, -1.0j, 0.0]) / np.sqrt(2.0),
    "sigma-": np.array([1.0, -1.0j, 0.0]) / np.sqrt(2.0),
}


def _leg_weights(eps: np.ndarray) -> dict:
    """c[p] = (-1)^p eps_{-p} for d . eps = sum_p c[p] T^1_p(d)."""
    from heff.twophoton import _leg
    return _leg(eps)


def _intermediate_kets(kets: np.ndarray, j_max_int: int) -> np.ndarray:
    """|J'', Omega'' = +-1, F'', m''> for J'' = 1..j_max_int, in the X kets' dtype and phase convention
    so dipole_geometry serves both legs. Omega'' = +-1 models an Omega = 1 electronic state.
    """
    rows = []
    for J in range(1, j_max_int + 1):
        for Om in (-1.0, 1.0):
            for F in (J - 0.5, J + 0.5):
                for k in range(int(2 * F) + 1):
                    rows.append((float(J), Om, F, -F + k))
    return np.array(rows, dtype=kets.dtype)


def two_photon_amplitudes(kets: np.ndarray, I: float, eps1: np.ndarray, eps2: np.ndarray,
                          j_max_int: int) -> np.ndarray:
    """A[a, b] = <a| (d.eps2*) P_int (d.eps1) |b> in the X ket basis.

    Dimensionless: the intermediate reduced dipole and the common detuning are set to one. eps2 is
    conjugated (Raman reading: photon 2 is emitted).
    """
    from heff.elements_c import dipole_geometry

    ints = _intermediate_kets(kets, j_max_int)
    nx, ni = len(kets), len(ints)
    # d_up[p][a, i] = <X_a| T^1_p |int_i>,  d_dn[p][i, b] = <int_i| T^1_p |X_b>
    d_up = {p: np.zeros((nx, ni)) for p in (-1, 0, 1)}
    d_dn = {p: np.zeros((ni, nx)) for p in (-1, 0, 1)}
    for p in (-1, 0, 1):
        for a in range(nx):
            for i in range(ni):
                d_up[p][a, i] = dipole_geometry(kets[a], ints[i], I, p)
        for i in range(ni):
            for b in range(nx):
                d_dn[p][i, b] = dipole_geometry(ints[i], kets[b], I, p)
    c_a = _leg_weights(np.conj(np.asarray(eps2, dtype=complex)))
    c_b = _leg_weights(np.asarray(eps1, dtype=complex))
    out = np.zeros((nx, nx), dtype=np.complex128)
    for pa in (-1, 0, 1):
        if c_a[pa] == 0:
            continue
        for pb in (-1, 0, 1):
            if c_b[pb] == 0:
                continue
            out += c_a[pa] * c_b[pb] * (d_up[pa] @ d_dn[pb])
    return out


@dataclass(frozen=True)
class Couplings:
    """sigma+ couplings between eigenstates, i -> f with m_F(f) = m_F(i) + 1."""

    i_idx: np.ndarray
    f_idx: np.ndarray
    omega: np.ndarray            # complex, rad/ms, reference-normalised
    jones: tuple                 # (name of eps1, name of eps2) that gave Delta m_F = +1
    raw_reference: float         # |A| of the reference transition before scaling


def sigma_plus_couplings(levels: Levels, j_max_int: int | None = None, reference: tuple = REFERENCE_TRANSITION,
                         reference_rabi_khz: float = REFERENCE_RABI_KHZ,
                         min_rel_amplitude: float = 1e-9) -> Couplings:
    """The Delta m_F = +1 couplings above min_rel_amplitude of the largest, scaled so the reference
    transition has |Omega|/2pi = reference_rabi_khz.
    """
    heff = _heff()
    j_max = int(max(l[0] for l in levels.labels))
    I = heff.thf_spec(J_max=j_max).I
    j_max_int = j_max_int or j_max + 1

    # try both Jones assignments for the second beam and keep the one realising Delta m_F = +1
    mfs = np.array([l[3] for l in levels.labels])
    dm = mfs[:, None] - mfs[None, :]            # dm[f, i] = m_f - m_i
    chosen = None
    for name2 in ("sigma+", "sigma-"):
        a_ket = two_photon_amplitudes(levels.kets, I, JONES["pi"], JONES[name2], j_max_int)
        a_eig = levels.vectors.conj().T @ a_ket @ levels.vectors
        mask = np.abs(a_eig) > min_rel_amplitude * np.max(np.abs(a_eig))
        signs = set(np.unique(dm[mask]).tolist())
        if signs == {1.0}:
            chosen = (("pi", name2), a_eig)
            break
        if signs != {-1.0}:
            raise ValueError(f"unexpected Delta m_F set {signs} for (pi, {name2})")
    if chosen is None:
        raise ValueError("neither Jones assignment gives a pure Delta m_F = +1 channel")
    jones, a_eig = chosen

    f_idx, i_idx = np.nonzero(np.abs(a_eig) > min_rel_amplitude * np.max(np.abs(a_eig)))
    vals = a_eig[f_idx, i_idx]
    # parity is exact at E_z = 0 and the operator is parity-even, so anything crossing parity is
    # round-off (measured <= 6e-12 relative at J_max = 6)
    par = np.array([l[2] for l in levels.labels])
    cross = par[f_idx] != par[i_idx]
    if cross.any():
        raise ValueError(f"{cross.sum()} cross-parity couplings survive the {min_rel_amplitude:g} floor "
                         f"(max {np.abs(vals[cross]).max() / np.max(np.abs(a_eig)):.2e} rel.)")

    def _find(t):
        for n, l in enumerate(levels.labels):
            if l[0] == t[0] and abs(l[1] - t[1]) < 1e-9 and l[2] == t[2] \
                    and abs(l[3] - t[3]) < 1e-9 and l[4] == t[4]:
                return n
        raise ValueError(f"reference level {t} not found")
    i0, f0 = (_find(t) for t in reference)
    k = np.where((i_idx == i0) & (f_idx == f0))[0]
    if k.size != 1:
        raise ValueError("reference transition not uniquely present in the sigma+ channel")
    raw = float(abs(vals[k[0]]))
    scale = reference_rabi_khz / raw * TWO_PI          # -> rad/ms
    order = np.lexsort((f_idx, i_idx))
    return Couplings(i_idx=i_idx[order], f_idx=f_idx[order], omega=vals[order] * scale,
                     jones=jones, raw_reference=raw)


# Tables


def _git_hash(path: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "--short", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:  # pragma: no cover
        return "unknown"


def build_tables(j_max: int = J_MAX, b_gauss: float = B_FIELD_GAUSS,
                 e_v_per_cm: float = E_FIELD_V_PER_CM) -> tuple[Levels, Couplings]:
    levels = build_levels(j_max, b_gauss, e_v_per_cm)
    return levels, sigma_plus_couplings(levels)


def table_files(outdir, j_max: int = J_MAX, b_gauss: float = B_FIELD_GAUSS,
                e_v_per_cm: float = E_FIELD_V_PER_CM) -> tuple[str, str]:
    """(levels_file, rabi_file) under outdir, named as qlsgym.molecules.thf expects."""
    tag = table_tag(j_max, b_gauss, e_v_per_cm)
    return (os.path.join(outdir, f"ThF232_levels_{tag}.txt"),
            os.path.join(outdir, f"ThF232_two_photon_sigma_plus_modelA_{tag}.txt"))


def write_tables(levels: Levels, cpl: Couplings, levels_file: str, rabi_file: str, j_max: int = J_MAX,
                 b_gauss: float = B_FIELD_GAUSS, e_v_per_cm: float = E_FIELD_V_PER_CM) -> None:
    """Write both tables. Line 1 (timestamp, heff revision, spec hash) is the only line that can
    differ between two builds from the same heff revision.
    """
    heff_root = os.environ.get("HEFF_ROOT", "?")
    stamp = _dt.datetime.now().isoformat(timespec="seconds")
    common = [
        f"# generated {stamp} by qlsgym.molecules.thf_tables (heff @ {_git_hash(heff_root)}, "
        f"spec_hash {levels.spec_hash})",
        f"# molecule 232Th19F+ X 3Delta1 v=0, J = 1..{j_max}, I(19F) = 1/2, "
        f"basis |J, Omega=+-1, F, m_F>, {len(levels.labels)} states",
        f"# fields E_z = {e_v_per_cm} V/cm, B_z = {b_gauss} G (= {b_gauss / 10:.3f} mT)",
        "# Zeeman: heff's full effective tensor G = diag(G_xx, G_yy, G_zz) (below); the parity-dependent",
        "#   part is its G_Delta = (G_xx - G_yy)/2, so nothing is added here.",
        "#   Omega-doublet ordering: e (parity +(-1)^J) lies BELOW f at every J.",
        "# heff parameter set thf_v1:",
    ] + ["#   " + line for line in levels.params_table.splitlines()]
    with open(levels_file, "w") as fh:
        fh.write("\n".join(common) + "\n")
        fh.write("# parity: physical parity <P> (exact at E_z = 0); ef: Brown-1975 label\n")
        fh.write("# xi: energy-ordered index inside the (J, parity, m_F) class\n")
        fh.write("# columns: idx  J  F  parity  ef  mF  xi  E_MHz (17 significant digits, lossless)\n")
        for n, (l, e, ef) in enumerate(zip(levels.labels, levels.energies_MHz, levels.ef)):
            fh.write(f"{n:4d} {l[0]:2d} {l[1]:4.1f} {l[2]:>2s} {ef:>2s} {l[3]:5.1f} {l[4]:2d} {e:.17g}\n")
    with open(rabi_file, "w") as fh:
        fh.write("\n".join(common) + "\n")
        fh.write("# coupling model A: single Omega = 1 intermediate electronic state, all its\n"
                 f"#   rotational/hyperfine sublevels J'' <= {j_max + 1}, common detuning\n"
                 "#   (see docs/thf_system.md).  Parity-even, no Delta-Omega = +-2 part,\n"
                 "#   K = 1 part identically zero.\n"
                 f"# sigma+ channel := Delta m_F = +1, realised by Jones pair (eps1, eps2) = "
                 f"{cpl.jones} in the Raman reading (eps2 conjugated)\n"
                 f"# scale: reference {REFERENCE_TRANSITION[0]} -> {REFERENCE_TRANSITION[1]} "
                 f"set to |Omega|/2pi = {REFERENCE_RABI_KHZ} kHz (raw geometry {cpl.raw_reference:.9g})\n"
                 "# each coupling listed once, direction i -> f; Omega in rad/ms\n"
                 "# columns: i  f  Re(Omega)  Im(Omega)  |Omega|/2pi_kHz\n")
        for i, f, w in zip(cpl.i_idx, cpl.f_idx, cpl.omega):
            fh.write(f"{i:4d} {f:4d} {w.real:+.12e} {w.imag:+.12e} {abs(w) / TWO_PI:.9g}\n")
