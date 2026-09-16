"""232Th19F+ / Ca+: 192 hyperfine-resolved states (X 3Delta1, v = 0, J = 1..6, Omega = +-1, F = J
+- 1/2), 12 (J, parity) blocks.
"""

from __future__ import annotations

import dataclasses
import os
import re
import subprocess
from pathlib import Path

import numpy as np

from ..spec import Block, Molecule, Primitive, System, Task, Trap, Window, TWO_PI

# table tag; mirrors thffno.units.table_tag(J_MAX, B_FIELD_GAUSS, E_FIELD_V_PER_CM)
# (kept as a literal so qlsgym stays importable without the source package, G10)
_TAG = "Jmax_6_B_20G_E_0Vcm"
LEVELS_FILE = f"ThF232_levels_{_TAG}.txt"
RABI_FILE = f"ThF232_two_photon_sigma_plus_modelA_{_TAG}.txt"
SOURCE_REPO = "/n/home02/josemm/RESEARCH/PROJECTS/MOLESQLS/FNO_NEW_MOLECULE/ThF_232_19"

J_MAX = 6
B_FIELD_GAUSS = 20.0
E_FIELD_V_PER_CM = 0.0
# effective Zeeman tensor components read back out of the table header
G_SYMBOLS = ("G_xx", "G_yy", "G_zz")
REFERENCE_RABI_KHZ = 3.0
NU_F_OVER_2PI_KHZ = 1112.7182
ETA = 0.06776
N_NU = 7
OMEGA_HALF_WIDTH_OVER_2PI_KHZ = 470.0
TAU_MAX_MS, N_TAU = 6.0, 200
T_INT_K = 4.0
RWA_CUTOFF = 1.0e6
DELTA_MAX = 1.0e4
OMEGA_MIN_COUPLING = 1.0


def data_dir() -> Path:
    root = os.environ.get("QLSGYM_DATA")
    return Path(root) / "thf" if root else Path(__file__).resolve().parents[3] / "data" / "thf"


def _git_rev(repo: str):
    try:
        out = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=5)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


_GENERATED_RE = re.compile(r"\(heff @ (?P<rev>\S+?), spec_hash (?P<hash>\S+?)\)")
_PARAM_RE = re.compile(r"^#\s{2,}(?P<symbol>[A-Za-z][A-Za-z0-9_]*)\s+(?P<value>[-+0-9.eE]+)\s")


def read_table_header(filename) -> dict:
    """Provenance carried in the # preamble thffno.molecule writes."""
    rev = spec_hash = None
    params: dict = {}
    with open(filename) as fh:
        for line in fh:
            if not line.startswith("#"):
                break
            m = _GENERATED_RE.search(line)
            if m is not None:
                rev, spec_hash = m.group("rev"), m.group("hash")
                continue
            m = _PARAM_RE.match(line)
            if m is not None:
                try:
                    params[m.group("symbol")] = float(m.group("value"))
                except ValueError:                              # pragma: no cover - not a number column
                    pass
    return {"heff_git_rev": rev, "heff_spec_hash": spec_hash, "params": params}


# Readers (byte-compatible with thffno.system)


def read_energy_levels(filename):
    """(levels, energies): levels[j] = (J, F, parity, mF, xi), energies in rad/ms (E_MHz * 1e3 * 2
    pi).
    """
    levels, energies = [], []
    with open(filename) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            idx, J, F, parity, ef, mF, xi, e_mhz = line.split()
            assert int(idx) == len(levels), "level table must be contiguous"
            if parity not in ("+", "-"):
                raise ValueError(f"unexpected parity {parity!r}")
            levels.append((int(J), float(F), parity, float(mF), int(xi)))
            energies.append(float(e_mhz) * 1e3 * TWO_PI)
    return levels, np.asarray(energies, dtype=np.float64)


def read_rabi_file(filename):
    """(i_idx, f_idx, omega); omega complex, rad/ms, already normalised by the table builder."""
    i_idx, f_idx, vals = [], [], []
    with open(filename) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            i, f, re, im, _mag = line.split()
            i_idx.append(int(i))
            f_idx.append(int(f))
            vals.append(complex(float(re), float(im)))
    return (np.asarray(i_idx, dtype=np.int64), np.asarray(f_idx, dtype=np.int64),
            np.asarray(vals, dtype=np.complex128))


# System


def build_system(levels_file, rabi_file) -> System:
    """Load the tables and derive the (J, parity) block decomposition (ordered by descending size,
    ties by the smallest global state index, i.e. the lower-energy parity first).
    """
    levels, energies = read_energy_levels(levels_file)
    i_idx, f_idx, omega_c = read_rabi_file(rabi_file)
    keys = [(l[0], l[2]) for l in levels]
    uniq = sorted(set(keys), key=lambda k: (-keys.count(k), keys.index(k)))
    remap = {k: b for b, k in enumerate(uniq)}
    block_of_state = np.array([remap[k] for k in keys], dtype=np.int64)

    same_j = np.array([levels[int(a)][0] == levels[int(b)][0] for a, b in zip(i_idx, f_idx)])
    intra = block_of_state[i_idx] == block_of_state[f_idx]
    if not np.array_equal(intra, same_j):
        raise ValueError("coupling table violates the (J, parity) block structure")

    blocks = []
    for bidx, key in enumerate(uniq):
        states = np.where(block_of_state == bidx)[0]
        local = {int(g): k for k, g in enumerate(states)}
        sel = intra & np.isin(i_idx, states)
        blocks.append(Block(
            index=bidx, states=states, key=key,
            i_local=np.array([local[int(x)] for x in i_idx[sel]], dtype=np.int64),
            f_local=np.array([local[int(x)] for x in f_idx[sel]], dtype=np.int64),
            omega=omega_c[sel].copy(),
        ))
    return System(tuple(levels), energies, i_idx, f_idx, omega_c, tuple(blocks), block_of_state)


# Cross-J primitives (thffno.planner.cross_j_primitives, ported)


def cross_j_primitives(molecule: Molecule, n_primitives: int = 24, temperature_k: float | None = None,
                       delta_j: tuple = (1, 2), dedupe_khz: float = 1.0, max_per_family: int | None = None,
                       min_coupling: float | None = None, mode: str = "dark_state") -> tuple:
    """The cross-J sideband primitive library (ThF decisions D8, OPEN-1)."""
    from ..physics.thermal import boltzmann

    s, eta, nu_f = molecule.system, molecule.trap.eta, molecule.trap.nu_f
    tau_max = molecule.window.tau_max_ms
    min_coupling = molecule.window.omega_min_coupling if min_coupling is None else min_coupling
    p_th = boltzmann(molecule, temperature_k)
    lv = s.levels
    cands = []
    for k in range(s.omega_c.size):
        i, f = int(s.i_idx[k]), int(s.f_idx[k])
        dj = abs(lv[f][0] - lv[i][0])
        om = abs(s.omega_c[k])
        if dj not in delta_j or om < min_coupling:
            continue
        tau_pi = np.pi / (eta * np.exp(-eta ** 2 / 2.0) * om)       # 2|g| = eta e^{-eta^2/2} |Omega|
        if tau_pi > tau_max:
            continue
        d = float(s.energies[f] - s.energies[i])
        for sigma, w, src, tgt in (("+", nu_f + d, i, f), ("-", nu_f - d, f, i)):
            cands.append((float(p_th[src]), float(om), sigma, float(w), src, tgt, float(tau_pi)))
    cands.sort(key=lambda c: (-c[0], -c[1]))

    def _mk(pop, om, sigma, w, src, tgt, tau_pi) -> Primitive:
        Js, Fs, ps, ms, _ = lv[src]
        Jt, Ft, pt, mt, _ = lv[tgt]
        label = (f"sigma{sigma} |J{Js} F{Fs} {ps} m{ms:+.1f}> -> |J{Jt} F{Ft} {pt} m{mt:+.1f}>  "
                 f"w/2pi={w / TWO_PI / 1e6:+.5f} GHz |Om|/2pi={om / TWO_PI:.2f} kHz "
                 f"pi={tau_pi:.2f} ms src_pop={pop:.4f}")
        return Primitive(label, sigma, w, tau_pi, (), source=src, target=tgt)

    if mode == "dark_state":
        ladders: dict = {}
        for n, l in enumerate(lv):
            ladders.setdefault((l[0], l[1], l[2]), []).append(n)
        pumpable = sorted((k for k in ladders if k[1] >= 1.5),
                          key=lambda k: -sum(p_th[n] for n in ladders[k]))
        wanted: list = []
        for k in pumpable[:max(0, n_primitives - 4)]:
            J, F, par = k
            wanted.append(next(n for n in ladders[k] if abs(lv[n][3] - F) < 1e-9))
        for par in ("+", "-"):
            for m in (-0.5, 0.5):
                wanted.append(next(n for n, l in enumerate(lv)
                                   if l[0] == 1 and l[1] == 0.5 and l[2] == par and abs(l[3] - m) < 1e-9))
        out = []
        for src in wanted[:n_primitives]:
            best = max((c for c in cands if c[4] == src), key=lambda c: c[1], default=None)
            if best is None:
                raise ValueError(f"no cross-J coupling from {lv[src]} fits the pulse window")
            out.append(_mk(*best))
        return tuple(out)
    if mode != "thermal":
        raise ValueError(f"unknown mode {mode!r}")

    classes = sorted({(l[0], l[2]) for l in lv})
    mass = {c: float(sum(p_th[n] for n, l in enumerate(lv) if (l[0], l[2]) == c)) for c in classes}
    total = sum(mass.values())
    raw = {c: n_primitives * mass[c] / total for c in classes}
    budget = {c: int(np.floor(raw[c])) for c in classes}
    for c in sorted(classes, key=lambda c: -(raw[c] - budget[c]))[: n_primitives - sum(budget.values())]:
        budget[c] += 1
    n_dir: dict = {}

    def _pick(chosen: list, used: set, allow_repeat: bool, balance: bool) -> None:
        per_family: dict = {}
        for pop, om, sigma, w, src, tgt, tau_pi in cands:
            c = (lv[src][0], lv[src][2])
            if budget[c] <= 0:
                continue
            up = lv[tgt][0] > lv[src][0]
            if balance and n_dir.get((c, up), 0) > n_dir.get((c, not up), 0):
                continue
            if not allow_repeat and src in used:
                continue
            if any(p.sigma == sigma and abs(p.omega - w) < TWO_PI * dedupe_khz for p in chosen):
                continue
            fam = (lv[src][0], lv[tgt][0], sigma)
            if max_per_family is not None and per_family.get(fam, 0) >= max_per_family:
                continue
            per_family[fam] = per_family.get(fam, 0) + 1
            budget[c] -= 1
            n_dir[(c, up)] = n_dir.get((c, up), 0) + 1
            used.add(src)
            chosen.append(_mk(pop, om, sigma, w, src, tgt, tau_pi))

    out: list = []
    used: set = set()
    for allow_repeat, balance in ((False, True), (True, True), (False, False), (True, False)):
        if len(out) >= n_primitives:
            break
        _pick(out, used, allow_repeat=allow_repeat, balance=balance)
    return tuple(out)


def _with_sector_blocks(molecule: Molecule, prims) -> tuple:
    """Set each primitive's blocks to the union of the blocks its RWA sectors touch (the (J, p) U
    (J', p) union of D8).
    """
    from ..physics.hamiltonian import rwa_sectors

    out = []
    for p in prims:
        secs = rwa_sectors(molecule, p.sigma, p.omega)
        blocks = sorted({int(b) for sec in secs for b in sec.key[3]})
        if int(molecule.system.block_of_state[p.source]) not in blocks:
            raise RuntimeError(f"primitive {p.label!r} does not touch its own source block")
        out.append(dataclasses.replace(p, blocks=tuple(blocks)))
    return tuple(out)


# Entry point


def build(n_primitives: int = 24, mode: str = "dark_state", n_nu: int = N_NU,
          temperature_k: float = T_INT_K, max_pulses: int = 80, p_target: float = 0.98,
          levels_file=None, rabi_file=None) -> Molecule:
    """The ThF+ molecule pack."""
    ddir = data_dir()
    lf = Path(levels_file) if levels_file else ddir / LEVELS_FILE
    rf = Path(rabi_file) if rabi_file else ddir / RABI_FILE
    system = build_system(lf, rf)
    nu_f = TWO_PI * NU_F_OVER_2PI_KHZ
    trap = Trap(nu_f=nu_f, eta=ETA, n_nu=int(n_nu))
    # computed in kHz first, exactly as thffno.units does (bit-identical bounds)
    window = Window(omega_min=TWO_PI * (NU_F_OVER_2PI_KHZ - OMEGA_HALF_WIDTH_OVER_2PI_KHZ),
                    omega_max=TWO_PI * (NU_F_OVER_2PI_KHZ + OMEGA_HALF_WIDTH_OVER_2PI_KHZ),
                    tau_max_ms=TAU_MAX_MS, n_tau=N_TAU,
                    rwa_cutoff=RWA_CUTOFF, omega_min_coupling=OMEGA_MIN_COUPLING, carrier_bands=())
    task = Task(temperature_k=float(temperature_k), p_target=float(p_target), max_pulses=int(max_pulses))
    hdr = read_table_header(lf)
    g = hdr["params"]
    missing = [s for s in G_SYMBOLS if s not in g]
    if missing:
        raise ValueError(f"{lf}: level table header has no {missing} -- stale table? "
                         "qlsgym needs the heff G-tensor tables (2026-09-14 or later)")
    provenance = {
        "levels_file": str(lf), "rabi_file": str(rf),
        "b_field_gauss": B_FIELD_GAUSS, "e_field_v_per_cm": E_FIELD_V_PER_CM, "j_max": J_MAX,
        "coupling_model": "A",
        "zeeman": "heff G tensor (G_xx, G_yy, G_zz)",
        "g_xx": g["G_xx"], "g_yy": g["G_yy"], "g_zz": g["G_zz"],
        "g_delta": 0.5 * (g["G_xx"] - g["G_yy"]),
        "doublet_ordering": "e below f (heff 2026-09-14)",
        "heff_git_rev": hdr["heff_git_rev"], "heff_spec_hash": hdr["heff_spec_hash"],
        "reference_rabi_khz": REFERENCE_RABI_KHZ,
        "delta_max": DELTA_MAX, "s_emb": 0.05, "beta_emb": 0.01,
        "source_repo": SOURCE_REPO, "source_package": "thffno", "source_git_rev": _git_rev(SOURCE_REPO),
        "primitives": f"thffno.planner.cross_j_primitives(mode={mode!r}, n_primitives={n_primitives})",
        "max_pulses_in_source_results": 150,
    }
    from ..physics.spectrum import carrier_bands
    mol = Molecule("thf", system, trap, window, task, (), provenance)
    bands = tuple(tuple(map(float, b)) for sg in ("+", "-") for b in carrier_bands(mol, sg))
    mol = dataclasses.replace(mol, window=dataclasses.replace(window, carrier_bands=bands))
    if n_primitives > 0:
        mol = dataclasses.replace(mol, primitives=_with_sector_blocks(
            mol, cross_j_primitives(mol, n_primitives, mode=mode)))
    return mol
