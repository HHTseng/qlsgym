from __future__ import annotations

import datetime as _dt
import os
from dataclasses import dataclass, replace

import numpy as np

from ..config import molecule_config
from ..spec import TWO_PI
from .thf_tables import _git_hash, _heff

__all__ = ["CONFIG", "Levels", "Couplings", "build_levels", "sigma_plus_couplings", "build_tables",
           "write_tables", "table_files", "table_tag"]

CONFIG = molecule_config("thf229")
_TAB = CONFIG["tables"]
ISOTOPOLOGUE = str(_TAB["isotopologue"])
PARAMETER_SET = str(_TAB["heff_parameter_set"])
ZEROED = tuple(_TAB["zeroed_parameters"])
J_MAX = int(_TAB["j_max"])
B_FIELD_GAUSS = float(_TAB["b_field_gauss"])
E_FIELD_V_PER_CM = float(_TAB["e_field_v_per_cm"])
RAMAN_RANK = int(_TAB["raman_rank"])
RAMAN_D_OMEGA = int(_TAB["raman_d_omega"])
REFERENCE_TRANSITION = tuple(tuple(t) for t in _TAB["reference_transition"])
REFERENCE_RABI_KHZ = float(_TAB["reference_rabi_khz"])
MIN_REL_AMPLITUDE = float(_TAB["min_rel_amplitude"])


def table_tag(j_max: int = J_MAX, b_gauss: float = B_FIELD_GAUSS, e_v_per_cm: float = E_FIELD_V_PER_CM) -> str:
    return f"Jmax_{j_max}_B_{b_gauss:g}G_E_{e_v_per_cm:g}Vcm"


def table_files(outdir, j_max: int = J_MAX, b_gauss: float = B_FIELD_GAUSS,
                e_v_per_cm: float = E_FIELD_V_PER_CM) -> tuple[str, str]:
    tag = table_tag(j_max, b_gauss, e_v_per_cm)
    return (os.path.join(outdir, f"ThF229_levels_{tag}.txt"),
            os.path.join(outdir, f"ThF229_two_photon_sigma_plus_K{RAMAN_RANK}dOm{RAMAN_D_OMEGA}_{tag}.txt"))


@dataclass(frozen=True)
class Levels:
    kets: np.ndarray
    labels: list
    energies_MHz: np.ndarray
    vectors: np.ndarray
    ef: list
    spec_hash: str
    params_table: str
    ctx: object


@dataclass(frozen=True)
class Couplings:
    i_idx: np.ndarray
    f_idx: np.ndarray
    omega: np.ndarray
    raw_reference: float


def parameter_set():
    heff = _heff()
    pset = getattr(heff, PARAMETER_SET)(ISOTOPOLOGUE)
    zero = {s: replace(pset.params[s], value=0.0) for s in ZEROED if s in pset.params}
    return pset.with_(**zero) if zero else pset


def build_levels(j_max: int = J_MAX, b_gauss: float = B_FIELD_GAUSS,
                 e_v_per_cm: float = E_FIELD_V_PER_CM, pset=None) -> Levels:
    heff = _heff()
    from heff.elements_c2 import REGISTRY_C2

    spec = heff.thf_spec(ISOTOPOLOGUE, J_max=j_max)
    pset = parameter_set() if pset is None else pset
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
        tm = heff.build_term_matrices(sub, ctx, case="c2", registry=REGISTRY_C2)
        spec_hash = spec_hash or tm.manifest["spec_hash"]
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
        raise ValueError("parity is not exact; is E_z != 0? The block keys (J, parity, F1) assume exact parity.")

    counters: dict = {}
    labels = []
    for k in range(n):
        J = int(round(float(dom["J"][k])))
        F1 = float(dom["F1"][k])
        F = float(dom["F"][k])
        mF = float(dom["mF"][k])
        p = "+" if parity[k] > 0 else "-"
        xi = counters.get((J, F1, F, p, mF), 0)
        counters[(J, F1, F, p, mF)] = xi + 1
        labels.append((J, F1, F, p, mF, xi))
    return Levels(kets=kets, labels=labels, energies_MHz=energies, vectors=vectors, ef=efs,
                  spec_hash=spec_hash, params_table=pset.table(), ctx=ctx)


def sigma_plus_couplings(levels: Levels, reference: tuple = REFERENCE_TRANSITION,
                         reference_rabi_khz: float = REFERENCE_RABI_KHZ,
                         min_rel_amplitude: float = MIN_REL_AMPLITUDE) -> Couplings:
    from heff.twophoton import two_photon_matrix

    t_ket = np.asarray(two_photon_matrix(levels.kets, levels.kets, levels.ctx, K=RAMAN_RANK,
                                         dOmega=RAMAN_D_OMEGA, P=1), dtype=np.complex128)
    a = levels.vectors.conj().T @ t_ket @ levels.vectors
    big = np.abs(a) > min_rel_amplitude * np.max(np.abs(a))
    f_idx, i_idx = np.nonzero(big)
    vals = a[f_idx, i_idx]
    mfs = np.array([l[4] for l in levels.labels])
    if not np.allclose(mfs[f_idx] - mfs[i_idx], 1.0):
        raise ValueError("the P = +1 channel is not a pure Delta m_F = +1 operator")
    par = np.array([l[3] for l in levels.labels])
    if np.any(par[f_idx] != par[i_idx]):
        raise ValueError("cross-parity two-photon couplings survive")

    def _find(t):
        for n, l in enumerate(levels.labels):
            if (l[0] == t[0] and abs(l[1] - t[1]) < 1e-9 and abs(l[2] - t[2]) < 1e-9 and l[3] == t[3]
                    and abs(l[4] - t[4]) < 1e-9 and l[5] == t[5]):
                return n
        raise ValueError(f"reference level {t} not found")
    i0, f0 = (_find(t) for t in reference)
    k = np.where((i_idx == i0) & (f_idx == f0))[0]
    if k.size != 1:
        raise ValueError("reference transition not present in the sigma+ channel")
    raw = float(abs(vals[k[0]]))
    scale = reference_rabi_khz / raw * TWO_PI
    order = np.lexsort((f_idx, i_idx))
    return Couplings(i_idx=i_idx[order], f_idx=f_idx[order], omega=vals[order] * scale, raw_reference=raw)


def build_tables(j_max: int = J_MAX, b_gauss: float = B_FIELD_GAUSS,
                 e_v_per_cm: float = E_FIELD_V_PER_CM) -> tuple[Levels, Couplings]:
    levels = build_levels(j_max, b_gauss, e_v_per_cm)
    return levels, sigma_plus_couplings(levels)


def write_tables(levels: Levels, cpl: Couplings, levels_file: str, rabi_file: str, j_max: int = J_MAX,
                 b_gauss: float = B_FIELD_GAUSS, e_v_per_cm: float = E_FIELD_V_PER_CM) -> None:
    heff_root = os.environ.get("HEFF_ROOT", "?")
    stamp = _dt.datetime.now().isoformat(timespec="seconds")
    common = [
        f"# generated {stamp} by qlsgym.molecules.thf229_tables (heff @ {_git_hash(heff_root)}, "
        f"spec_hash {levels.spec_hash})",
        f"# molecule 229Th19F+ X 3Delta1 v=0, J = 1..{j_max}, I(229Th) = 5/2, I(19F) = 1/2, "
        f"basis |J, Omega=+-1, F1, F, m_F>, {len(levels.labels)} states",
        f"# fields E_z = {e_v_per_cm} V/cm, B_z = {b_gauss} G",
        f"# heff parameter set {PARAMETER_SET}({ISOTOPOLOGUE!r}) with {', '.join(ZEROED)} set to 0:",
    ] + ["#   " + line for line in levels.params_table.splitlines()]
    with open(levels_file, "w") as fh:
        fh.write("\n".join(common) + "\n")
        fh.write("# parity: physical parity <P> (exact at E_z = 0); ef: Brown-1975 label\n")
        fh.write("# xi: energy-ordered index inside the (J, F1, F, parity, m_F) class\n")
        fh.write("# columns: idx  J  F1  F  parity  ef  mF  xi  E_MHz (17 significant digits, lossless)\n")
        for n, (l, e, ef) in enumerate(zip(levels.labels, levels.energies_MHz, levels.ef)):
            fh.write(f"{n:4d} {l[0]:2d} {l[1]:4.1f} {l[2]:4.1f} {l[3]:>2s} {ef:>2s} {l[4]:5.1f} {l[5]:2d} {e:.17g}\n")
    with open(rabi_file, "w") as fh:
        fh.write("\n".join(common) + "\n")
        fh.write(f"# coupling: heff effective two-photon operator, rank K = {RAMAN_RANK}, dOmega = {RAMAN_D_OMEGA}, "
                 "component P = +1 (Delta m_F = +1), closure over unresolved intermediate structure\n"
                 f"# scale: reference {REFERENCE_TRANSITION[0]} -> {REFERENCE_TRANSITION[1]} "
                 f"set to |Omega|/2pi = {REFERENCE_RABI_KHZ} kHz (raw geometry {cpl.raw_reference:.9g})\n"
                 "# each coupling listed once, direction i -> f; Omega in rad/ms\n"
                 "# columns: i  f  Re(Omega)  Im(Omega)  |Omega|/2pi_kHz\n")
        for i, f, w in zip(cpl.i_idx, cpl.f_idx, cpl.omega):
            fh.write(f"{i:4d} {f:4d} {w.real:+.12e} {w.imag:+.12e} {abs(w) / TWO_PI:.9g}\n")
