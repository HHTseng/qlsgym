from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import numpy as np

from ..config import molecule_config
from ..spec import Block, Molecule, System, Task, Trap, Window, TWO_PI
from .thf import read_rabi_file, read_table_header

CONFIG = molecule_config("thf229")
_TAB, _TRAP, _WIN = CONFIG["tables"], CONFIG["trap"], CONFIG["window"]
_EMB, _TASK = CONFIG["embedding"], CONFIG["task"]

J_MAX = int(_TAB["j_max"])
B_FIELD_GAUSS = float(_TAB["b_field_gauss"])
E_FIELD_V_PER_CM = float(_TAB["e_field_v_per_cm"])
RAMAN_RANK, RAMAN_D_OMEGA = int(_TAB["raman_rank"]), int(_TAB["raman_d_omega"])
NU_F_OVER_2PI_KHZ = float(_TRAP["nu_f_khz"])
ETA = float(_TRAP["eta"])
N_NU = int(_TRAP["n_nu"])
OMEGA_HALF_WIDTH_OVER_2PI_KHZ = float(_WIN["half_width_khz"])
TAU_MAX_MS, N_TAU = float(_WIN["tau_max_ms"]), int(_WIN["n_tau"])
T_INT_K = float(_TASK["temperature_k"])
RWA_CUTOFF = float(_WIN["rwa_cutoff_rad_per_ms"])
DELTA_MAX = float(_EMB["delta_max_rad_per_ms"])
OMEGA_MIN_COUPLING = float(_WIN["omega_min_coupling_rad_per_ms"])


def table_tag(j_max: int = J_MAX, b_gauss: float = B_FIELD_GAUSS, e_v_per_cm: float = E_FIELD_V_PER_CM) -> str:
    return f"Jmax_{j_max}_B_{b_gauss:g}G_E_{e_v_per_cm:g}Vcm"


LEVELS_FILE = f"ThF229_levels_{table_tag()}.txt"
RABI_FILE = f"ThF229_two_photon_sigma_plus_K{RAMAN_RANK}dOm{RAMAN_D_OMEGA}_{table_tag()}.txt"


def data_dir() -> Path:
    root = os.environ.get("QLSGYM_DATA")
    return Path(root) / "thf229" if root else Path(__file__).resolve().parents[3] / "data" / "thf229"


def read_energy_levels(filename):
    levels, energies = [], []
    with open(filename) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            idx, J, F1, F, parity, ef, mF, xi, e_mhz = line.split()
            assert int(idx) == len(levels), "level table must be contiguous"
            if parity not in ("+", "-"):
                raise ValueError(f"unexpected parity {parity!r}")
            levels.append((int(J), float(F1), float(F), parity, float(mF), int(xi)))
            energies.append(float(e_mhz) * 1e3 * TWO_PI)
    return levels, np.asarray(energies, dtype=np.float64)


def _window_bounds() -> tuple[float, float]:
    return (TWO_PI * (NU_F_OVER_2PI_KHZ - OMEGA_HALF_WIDTH_OVER_2PI_KHZ),
            TWO_PI * (NU_F_OVER_2PI_KHZ + OMEGA_HALF_WIDTH_OVER_2PI_KHZ))


def build_system(levels_file, rabi_file) -> System:
    levels, energies = read_energy_levels(levels_file)
    i_idx, f_idx, omega_c = read_rabi_file(rabi_file)
    lo, hi = _window_bounds()
    nu_f = TWO_PI * NU_F_OVER_2PI_KHZ
    d = energies[f_idx] - energies[i_idx]
    near = np.zeros(i_idx.size, dtype=bool)
    for w_res in (nu_f + d, nu_f - d):
        near |= np.maximum(0.0, np.maximum(lo - w_res, w_res - hi)) <= RWA_CUTOFF

    groups = [(l[0], l[3], l[1]) for l in levels]
    names = sorted(set(groups), key=groups.index)
    parent = {g: g for g in names}

    def root(g):
        while parent[g] != g:
            parent[g] = parent[parent[g]]
            g = parent[g]
        return g
    for k in np.where(near)[0]:
        a, b = root(groups[int(i_idx[k])]), root(groups[int(f_idx[k])])
        if a != b:
            parent[b] = a
    members: dict = {}
    for g in names:
        members.setdefault(root(g), []).append(g)

    def key_of(gs):
        js = sorted({g[0] for g in gs})
        return (js[0] if len(js) == 1 else tuple(js), gs[0][1], tuple(sorted({g[2] for g in gs})))
    comp_key = {g: key_of(gs) for gs in members.values() for g in gs}
    keys = [comp_key[g] for g in groups]
    uniq = sorted(set(keys), key=lambda k: (-keys.count(k), keys.index(k)))
    remap = {k: b for b, k in enumerate(uniq)}
    block_of_state = np.array([remap[k] for k in keys], dtype=np.int64)
    intra = block_of_state[i_idx] == block_of_state[f_idx]
    assert not np.any(near & ~intra)

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


def build(n_nu: int = N_NU, temperature_k: float = T_INT_K, max_pulses: int = int(_TASK["max_pulses"]),
          p_target: float = float(_TASK["p_target"]), levels_file=None, rabi_file=None) -> Molecule:
    ddir = data_dir()
    lf = Path(levels_file) if levels_file else ddir / LEVELS_FILE
    rf = Path(rabi_file) if rabi_file else ddir / RABI_FILE
    system = build_system(lf, rf)
    trap = Trap(nu_f=TWO_PI * NU_F_OVER_2PI_KHZ, eta=ETA, n_nu=int(n_nu))
    lo, hi = _window_bounds()
    window = Window(omega_min=lo, omega_max=hi, tau_max_ms=TAU_MAX_MS, n_tau=N_TAU,
                    rwa_cutoff=RWA_CUTOFF, omega_min_coupling=OMEGA_MIN_COUPLING, carrier_bands=(),
                    s_emb=float(_EMB["s_emb"]), beta_emb=float(_EMB["beta_emb"]))
    task = Task(temperature_k=float(temperature_k), p_target=float(p_target), max_pulses=int(max_pulses))
    hdr = read_table_header(lf)
    provenance = {
        "levels_file": str(lf), "rabi_file": str(rf),
        "isotopologue": "229Th19F+", "b_field_gauss": B_FIELD_GAUSS, "e_field_v_per_cm": E_FIELD_V_PER_CM,
        "j_max": J_MAX, "coupling_model": f"heff two-photon K={RAMAN_RANK} dOmega={RAMAN_D_OMEGA}",
        "zeroed_parameters": list(_TAB["zeroed_parameters"]),
        "heff_git_rev": hdr["heff_git_rev"], "heff_spec_hash": hdr["heff_spec_hash"],
        "reference_rabi_khz": float(_TAB["reference_rabi_khz"]),
        "delta_max": DELTA_MAX, "s_emb": window.s_emb, "beta_emb": window.beta_emb,
        "primitives": "none",
    }
    from ..physics.spectrum import carrier_bands
    mol = Molecule("thf229", system, trap, window, task, (), provenance)
    bands = tuple(tuple(map(float, b)) for sg in ("+", "-") for b in carrier_bands(
        mol, sg, half_width_khz=float(_WIN["carrier_half_width_khz"]),
        min_coupling=TWO_PI * float(_WIN["carrier_min_coupling_khz"])))
    return dataclasses.replace(mol, window=dataclasses.replace(window, carrier_bands=bands))
