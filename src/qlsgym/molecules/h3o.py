from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import numpy as np

from ..config import molecule_config
from ..spec import Block, Molecule, Primitive, System, Task, Trap, Window, TWO_PI

CONFIG = molecule_config("h3o")
_TAB, _TRAP, _WIN = CONFIG["tables"], CONFIG["trap"], CONFIG["window"]
_EMB, _TASK, _PRIM = CONFIG["embedding"], CONFIG["task"], CONFIG["primitives"]

LEVELS_FILE = str(_TAB["levels_file"])
RABI_FILE = str(_TAB["rabi_file"])

REFERENCE_TRANSITION = tuple(tuple(t) for t in _TAB["reference_transition"])
REFERENCE_RABI_KHZ = float(_TAB["reference_rabi_khz"])

NU_F_OVER_2PI_KHZ = float(_TRAP["nu_f_khz"])
ETA = float(_TRAP["eta"])
N_NU = int(_TRAP["n_nu"])
OMEGA_MIN_OVER_2PI_KHZ, OMEGA_MAX_OVER_2PI_KHZ = float(_WIN["omega_min_khz"]), float(_WIN["omega_max_khz"])
TAU_MAX_MS, N_TAU = float(_WIN["tau_max_ms"]), int(_WIN["n_tau"])
T_INT_K = float(_TASK["temperature_k"])
RWA_CUTOFF = float(_WIN["rwa_cutoff_rad_per_ms"])
DELTA_MAX = float(_EMB["delta_max_rad_per_ms"])
OMEGA_MIN_COUPLING = float(_WIN["omega_min_coupling_rad_per_ms"])


def data_dir() -> Path:
    root = os.environ.get("QLSGYM_DATA")
    return Path(root) / "h3o" if root else Path(__file__).resolve().parents[3] / "data" / "h3o"


def read_energy_levels(filename):
    levels, energies = [], []
    with open(filename) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            J, S, K, mF, xi, e_mhz = int(parts[0]), int(parts[1]), float(parts[2]), float(parts[3]), int(parts[4]), float(parts[5])
            if S == 1:
                parity = "+"
            elif S == -1:
                parity = "-"
            else:
                raise ValueError(f"unexpected S value: {S}")
            levels.append((J, K, parity, mF, xi))
            energies.append(e_mhz * 1e3 * TWO_PI)
    return levels, np.asarray(energies, dtype=np.float64)


def read_rabi_file(filename, levels, target_ref_rabi_khz: float = REFERENCE_RABI_KHZ):
    i_idx, f_idx, vals = [], [], []
    with open(filename) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith(("Type", "#")):
                continue
            parts = line.split()
            if len(parts) < 15:
                continue
            i_idx.append(int(parts[1]))
            f_idx.append(int(parts[7]))
            vals.append(complex(float(parts[13]), float(parts[14])))
    i_arr = np.asarray(i_idx, dtype=np.int64)
    f_arr = np.asarray(f_idx, dtype=np.int64)
    v_arr = np.asarray(vals, dtype=np.complex128)

    def _find(t):
        for n, l in enumerate(levels):
            if l[0] == t[0] and l[1] == t[1] and l[2] == t[2] and abs(l[3] - t[3]) < 1e-9 and l[4] == t[4]:
                return n
        raise ValueError(f"reference level {t} not found in {filename}")

    i0, f0 = (_find(t) for t in REFERENCE_TRANSITION)
    k = np.where((i_arr == i0) & (f_arr == f0))[0]
    if k.size != 1:
        raise ValueError("reference transition not uniquely present")
    scale = float(target_ref_rabi_khz) / abs(v_arr[k[0]]) * TWO_PI
    return i_arr, f_arr, v_arr * scale


def build_system(levels_file, rabi_file) -> System:
    from ..physics.hamiltonian import components

    levels, energies = read_energy_levels(levels_file)
    i_idx, f_idx, omega_c = read_rabi_file(rabi_file, levels)
    n = len(levels)
    labels = components(n, i_idx, f_idx)
    order = sorted(np.unique(labels),
                   key=lambda L: (-int((labels == L).sum()), int(np.where(labels == L)[0][0])))
    remap = {int(L): k for k, L in enumerate(order)}
    block_of_state = np.array([remap[int(L)] for L in labels], dtype=np.int64)

    blocks = []
    for bidx in range(len(order)):
        states = np.where(block_of_state == bidx)[0]
        local = {int(g): k for k, g in enumerate(states)}
        sel = np.isin(i_idx, states)
        assert np.array_equal(sel, np.isin(f_idx, states))
        keys = {(levels[int(g)][1], levels[int(g)][2]) for g in states}
        assert len(keys) == 1, f"block {bidx} spans several (K, parity) sectors: {keys}"
        blocks.append(Block(
            index=bidx, states=states, key=next(iter(keys)),
            i_local=np.array([local[int(x)] for x in i_idx[sel]], dtype=np.int64),
            f_local=np.array([local[int(x)] for x in f_idx[sel]], dtype=np.int64),
            omega=omega_c[sel].copy(),
        ))
    return System(tuple(levels), energies, i_idx, f_idx, omega_c, tuple(blocks), block_of_state)


def thz_primitives(molecule: Molecule, n_families: int = int(_PRIM["n_families"]), temperature_k: float | None = None,
                   min_source_population: float = float(_PRIM["min_source_population"]), max_per_family: int | None = None) -> tuple:
    from ..physics.thermal import boltzmann

    s, eta, nu_f = molecule.system, molecule.trap.eta, molecule.trap.nu_f
    tau_max = molecule.window.tau_max_ms
    p_th = boltzmann(molecule, temperature_k)
    families: dict = {}
    for b in s.blocks:
        for k in range(b.omega.size):
            gi, gf = int(b.states[b.i_local[k]]), int(b.states[b.f_local[k]])
            ji, jf = s.levels[gi][0], s.levels[gf][0]
            if ji == jf:
                continue
            d = s.energies[gf] - s.energies[gi]
            if d <= 0:
                continue
            om = abs(b.omega[k])
            if om <= 0 or np.pi / (eta * om) > tau_max:
                continue
            rec = families.setdefault((ji, jf), {"mass": 0.0, "res": []})
            rec["mass"] += float(p_th[gi])
            rec["res"].append((b.index, float(nu_f + d), om, float(p_th[gi]), gi, gf))

    ranked = sorted(families.items(), key=lambda kv: -kv[1]["mass"])[:n_families]
    out = []
    for (ji, jf), rec in ranked:
        seen: set = set()
        kept = 0
        for bidx, w, om, src, gi, gf in sorted(rec["res"], key=lambda r: (-r[3], -r[2])):
            if src < min_source_population:
                continue
            key = round(w, 3)
            if key in seen:
                continue
            if max_per_family is not None and kept >= max_per_family:
                break
            seen.add(key)
            kept += 1
            tau_pi = np.pi / (eta * np.exp(-eta ** 2 / 2.0) * om)
            label = (f"J{ji}->{jf} {w / TWO_PI / 1e6:.4f}GHz |Om|/2pi={om / TWO_PI:.2f}kHz "
                     f"src_pop={src:.4f}")
            out.append(Primitive(label, "+", w, float(tau_pi), (int(bidx),), source=gi, target=gf))
    return tuple(out)


def _with_sector_blocks(molecule: Molecule, prims) -> tuple:
    from ..physics.hamiltonian import rwa_sectors

    out = []
    for p in prims:
        secs = rwa_sectors(molecule, p.sigma, p.omega)
        blocks = sorted({int(b) for sec in secs for b in sec.key[3]})
        if p.source >= 0 and int(molecule.system.block_of_state[p.source]) not in blocks:
            raise RuntimeError(f"primitive {p.label!r} does not touch its own source block")
        out.append(dataclasses.replace(p, blocks=tuple(blocks)))
    return tuple(out)


def build(n_families: int = int(_PRIM["n_families"]), n_nu: int = N_NU, temperature_k: float = T_INT_K,
          max_pulses: int = int(_TASK["max_pulses"]), p_target: float = float(_TASK["p_target"]),
          primitives: bool = True,
          levels_file=None, rabi_file=None) -> Molecule:
    ddir = data_dir()
    lf = Path(levels_file) if levels_file else ddir / LEVELS_FILE
    rf = Path(rabi_file) if rabi_file else ddir / RABI_FILE
    system = build_system(lf, rf)
    trap = Trap(nu_f=TWO_PI * NU_F_OVER_2PI_KHZ, eta=ETA, n_nu=int(n_nu))
    window = Window(omega_min=TWO_PI * OMEGA_MIN_OVER_2PI_KHZ, omega_max=TWO_PI * OMEGA_MAX_OVER_2PI_KHZ,
                    tau_max_ms=TAU_MAX_MS, n_tau=N_TAU, rwa_cutoff=RWA_CUTOFF,
                    omega_min_coupling=OMEGA_MIN_COUPLING, carrier_bands=(),
                    s_emb=float(_EMB["s_emb"]), beta_emb=float(_EMB["beta_emb"]))
    task = Task(temperature_k=float(temperature_k), p_target=float(p_target), max_pulses=int(max_pulses))
    provenance = {
        "levels_file": str(lf), "rabi_file": str(rf),
        "b_field_mt": float(_TAB["b_field_mt"]), "j_max": int(_TAB["j_max"]),
        "reference_transition": REFERENCE_TRANSITION, "reference_rabi_khz": REFERENCE_RABI_KHZ,
        "delta_max": DELTA_MAX, "s_emb": window.s_emb, "beta_emb": window.beta_emb,
        "primitives": f"thz_primitives(n_families={n_families}), sigma+ at their pi-time",
        "paper": "arXiv:2410.11839",
    }
    from ..physics.spectrum import carrier_bands
    mol = Molecule("h3o", system, trap, window, task, (), provenance)
    bands = tuple(tuple(map(float, b)) for sg in ("+", "-") for b in carrier_bands(
        mol, sg, half_width_khz=float(_WIN["carrier_half_width_khz"]),
        min_coupling=TWO_PI * float(_WIN["carrier_min_coupling_khz"])))
    mol = dataclasses.replace(mol, window=dataclasses.replace(window, carrier_bands=bands))
    if primitives and n_families > 0:
        mol = dataclasses.replace(mol, primitives=_with_sector_blocks(mol, thz_primitives(mol, n_families)))
    return mol
