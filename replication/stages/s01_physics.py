"""Stage 1 -- the molecule the paper describes, checked against the paper."""

from __future__ import annotations

import numpy as np

from ..config import Config, load_molecule

NAME = "s01_physics"
TITLE = "structure and control settings vs Sec. II (H3O+: 444 states, Eqs. 43-44)"
PAPER = "Sec. II, Eqs. 43-44"
REQUIRES: tuple = ()
COST = "seconds, CPU"

TWO_PI = 2.0 * np.pi

# what the paper states about H3O+ (Sec. II, Eqs. 43-44)

# Sec. II: the 444 hyperfine-resolved states of H3O+ up to J = 4 at 0.36 mT.
H3O_N_STATES = 444
# Eq. 43, which lists 2 M_f for the ten (K, parity) sectors.
H3O_BLOCK_DIMS: tuple = (64, 64, 60, 48, 48, 42, 42, 40, 18, 18)
# Eq. 44: the (K, +) / (K, -) partners are degenerate, so these six
# blocks carry all the distinct dynamics.
H3O_UNIQUE_BLOCKS: tuple = (0, 2, 3, 5, 7, 8)
# Sec. II trap and control settings (nu_f / 2pi kHz, Lamb-Dicke eta).
H3O_NU_F_OVER_2PI_KHZ = 5164.0
H3O_ETA = 0.09
# The drive window the surrogate is trained on, kHz.
H3O_WINDOW_OVER_2PI_KHZ: tuple = (5050.0, 5300.0)
# Pulse-time grid: 200 points over [0, 4 ms], endpoints included (Eq. 19).
H3O_N_TAU = 200
H3O_TAU_MAX_MS = 4.0
# Thermal state and stopping rule (Sec. II.4).
H3O_TEMPERATURE_K = 20.0
H3O_P_TARGET = 0.98
# "two discrete THz transitions" (Sec. II, control library).  The paper names
# the two Delta J families; qlsgym's selection rule resolves them into 21
# m_F-resolved primitives (molecules.h3o.thz_primitives), which is the
# number checked here.
H3O_N_THZ_FAMILIES = 2
H3O_N_PRIMITIVES = 21

# The paper propagates nu <= 1, i.e. two motional levels;
# Molecule.trap.n_nu is the molecule's own answer and is reported beside it.
PAPER_NU_MAX = 1
PAPER_N_NU = PAPER_NU_MAX + 1

# Two blocks count as parity partners when their retained sigma+ resonances
# coincide to this fraction of a half-linewidth (H3O+ measures ~1e-4).
PARTNER_TOL_HWHM = 1.0e-2
# Monte-Carlo samples for the active-fraction measurement.
ACTIVE_FRACTION_SAMPLES = 200_000


class StructureError(RuntimeError):
    """A structural check on the molecule failed."""


# Measurements


def _khz(omega: float) -> float:
    return float(omega) / TWO_PI


def block_table(molecule) -> list:
    """One row per block: dimensions, label, and Eq. 43's 2 M_f."""
    rows = []
    for b in molecule.blocks:
        key = tuple(float(k) if isinstance(k, (int, float, np.floating)) else str(k) for k in b.key)
        rows.append({"index": int(b.index), "key": key, "m_f": int(b.n_states),
                     "two_m_f": 2 * int(b.n_states), "dim_propagated": int(b.dim(molecule.trap.n_nu)),
                     "n_couplings": int(b.omega.size)})
    return rows


def resonance_geometry(molecule, sigmas: tuple = ("+", "-")) -> list:
    """Per (block, sigma): how much of the drive window is active."""
    from qlsgym.surrogate.metrics import active_fraction, is_on_resonance
    from qlsgym.surrogate.metrics import resonance_geometry as geometry

    w = molecule.window
    out = []
    for b in molecule.blocks:
        for sigma in sigmas:
            w_res, hwhm = geometry(molecule, b.index, sigma)
            inside = (w_res >= w.omega_min) & (w_res <= w.omega_max)
            row = {"block": int(b.index), "sigma": sigma, "m_f": int(b.n_states),
                   "n_retained": int(w_res.size), "n_in_window": int(inside.sum()),
                   "active_fraction": active_fraction(molecule, b.index, sigma, 1.0,
                                                      n_samples=ACTIVE_FRACTION_SAMPLES),
                   "hwhm_hz_median": None, "hwhm_hz_min": None, "hwhm_hz_max": None}
            if w_res.size:
                hz = hwhm / TWO_PI * 1e3
                row.update(hwhm_hz_median=float(np.median(hz)), hwhm_hz_min=float(hz.min()),
                           hwhm_hz_max=float(hz.max()))
            if inside.any():
                # the on-resonance mask is what every stratified number is cut on
                row["on_resonance_at_own_resonances"] = bool(
                    is_on_resonance(molecule, w_res[inside], b.index, sigma, 1.0).all())
            out.append(row)
    return out


def parity_partners(molecule, tol_hwhm: float = PARTNER_TOL_HWHM) -> dict:
    """Group blocks whose sigma+ resonance spectra are degenerate (Eq. 44)."""
    from qlsgym.surrogate.metrics import resonance_geometry as geometry

    specs = []
    for b in molecule.blocks:
        w_res, hwhm = geometry(molecule, b.index, "+")
        order = np.argsort(w_res)
        specs.append((w_res[order], hwhm[order], int(b.n_states)))

    groups: list = []
    worst = 0.0
    for i, (w_i, h_i, m_i) in enumerate(specs):
        for g in groups:
            w_j, h_j, m_j = specs[g[0]]
            if m_i != m_j or w_i.size != w_j.size or w_i.size == 0:
                continue
            scale = np.maximum(np.minimum(h_i, h_j), 1e-12)
            split = float(np.max(np.abs(w_i - w_j) / scale))
            if split <= tol_hwhm:
                g.append(i)
                worst = max(worst, float(np.max(np.abs(w_i - w_j))))
                break
        else:
            groups.append([i])
    return {"groups": [list(map(int, g)) for g in groups],
            "unique_blocks": [int(g[0]) for g in groups],
            "n_partnered": int(sum(len(g) - 1 for g in groups)),
            "max_partner_split_hz": worst / TWO_PI * 1e3,
            "tol_hwhm": float(tol_hwhm)}


def thermal_summary(molecule) -> dict:
    """The thermal state the protocol starts from, and what the THz primitives reach
    (qlsgym.physics.thermal.boltzmann).
    """
    from qlsgym.physics.thermal import boltzmann

    p = boltzmann(molecule)
    sources = sorted({int(pr.source) for pr in molecule.primitives if pr.source >= 0})
    return {"temperature_k": float(molecule.task.temperature_k),
            "sum": float(p.sum()), "max_population": float(p.max()),
            "n_states_above_1pct": int((p > 0.01).sum()),
            "primitive_source_states": len(sources),
            "primitive_source_mass": float(p[sources].sum()) if sources else 0.0}


# Checks


def _universal_checks(molecule, checks: dict) -> None:
    """Invariants every molecule pack must satisfy (spec.py's contract)."""
    s, w, trap = molecule.system, molecule.window, molecule.trap
    dims = [b.n_states for b in molecule.blocks]
    covered = np.concatenate([b.states for b in molecule.blocks]) if molecule.blocks else np.zeros(0, int)
    checks["blocks_partition_states"] = bool(
        covered.size == sum(dims) and np.array_equal(np.sort(covered), np.arange(molecule.n_states)))
    checks["block_of_state_agrees"] = bool(all(
        np.all(s.block_of_state[b.states] == b.index) for b in molecule.blocks))
    checks["couplings_stay_inside_blocks"] = bool(all(
        b.i_local.size == b.f_local.size == b.omega.size
        and (b.i_local.size == 0 or (b.i_local.max() < b.n_states and b.f_local.max() < b.n_states))
        for b in molecule.blocks))
    tau = molecule.tau_grid()
    checks["tau_grid_matches_window"] = bool(
        tau.size == w.n_tau and tau[0] == 0.0 and np.isclose(tau[-1], w.tau_max_ms))
    checks["window_brackets_sideband"] = bool(w.omega_min < trap.nu_f < w.omega_max)
    checks["n_nu_at_least_two"] = bool(trap.n_nu >= 2)
    checks["energies_match_states"] = bool(s.energies.shape == (molecule.n_states,))


def _h3o_checks(molecule, measured: dict, checks: dict) -> None:
    """The paper's own numbers for H3O+ (Sec. II, Eqs. 43-44)."""
    w, trap, task = molecule.window, molecule.trap, molecule.task
    dims = tuple(int(b.n_states) for b in molecule.blocks)
    keys = [b.key for b in molecule.blocks]

    checks["h3o_n_states_444"] = molecule.n_states == H3O_N_STATES
    checks["h3o_ten_blocks"] = len(molecule.blocks) == len(H3O_BLOCK_DIMS)
    checks["h3o_eq43_block_dims"] = dims == H3O_BLOCK_DIMS
    checks["h3o_blocks_are_k_parity_sectors"] = bool(
        len(set(keys)) == len(keys) and all(len(k) == 2 and k[1] in ("+", "-") for k in keys))
    checks["h3o_eq44_unique_blocks"] = (
        tuple(measured["parity_partners"]["unique_blocks"]) == H3O_UNIQUE_BLOCKS)
    checks["h3o_partners_share_k"] = bool(all(
        len({keys[i][0] for i in g}) == 1 and len({keys[i][1] for i in g}) == len(g)
        for g in measured["parity_partners"]["groups"]))

    checks["h3o_nu_f"] = np.isclose(_khz(trap.nu_f), H3O_NU_F_OVER_2PI_KHZ)
    checks["h3o_eta"] = np.isclose(trap.eta, H3O_ETA)
    checks["h3o_window"] = bool(np.isclose(_khz(w.omega_min), H3O_WINDOW_OVER_2PI_KHZ[0])
                                and np.isclose(_khz(w.omega_max), H3O_WINDOW_OVER_2PI_KHZ[1]))
    checks["h3o_tau_grid"] = bool(w.n_tau == H3O_N_TAU and np.isclose(w.tau_max_ms, H3O_TAU_MAX_MS))
    checks["h3o_temperature"] = np.isclose(task.temperature_k, H3O_TEMPERATURE_K)
    checks["h3o_p_target"] = np.isclose(task.p_target, H3O_P_TARGET)
    checks["h3o_nu_truncation"] = trap.n_nu == PAPER_N_NU
    checks["h3o_n_primitives"] = len(molecule.primitives) == H3O_N_PRIMITIVES
    checks["h3o_thz_families"] = len(measured["primitives"]["families"]) == H3O_N_THZ_FAMILIES
    checks["h3o_no_carrier_in_window"] = len(w.carrier_bands) == 0
    checks["h3o_every_block_has_resonances"] = bool(
        all(r["n_in_window"] > 0 for r in measured["resonance_geometry"]))


# Stage protocol


def plan(cfg: Config) -> list:
    lines = [f"load molecule {cfg.molecule!r} through qlsgym.load_molecule (no Hamiltonian is built)",
             "check the block decomposition, tau grid, drive window and motional truncation",
             f"measure the active fraction of the drive window per (block, sigma), "
             f"{ACTIVE_FRACTION_SAMPLES:,} Monte-Carlo samples",
             "group the parity-degenerate blocks and report the unique set"]
    if cfg.molecule == "h3o":
        lines += [f"assert the paper's numbers: {H3O_N_STATES} states, Eq. 43 dims "
                  f"{list(H3O_BLOCK_DIMS)} (the paper lists 2 M_f), Eq. 44 unique blocks "
                  f"{list(H3O_UNIQUE_BLOCKS)}",
                  f"assert nu_f/2pi = {H3O_NU_F_OVER_2PI_KHZ} kHz, eta = {H3O_ETA}, window "
                  f"{H3O_WINDOW_OVER_2PI_KHZ[0]}-{H3O_WINDOW_OVER_2PI_KHZ[1]} kHz, "
                  f"n_tau = {H3O_N_TAU}, tau_max = {H3O_TAU_MAX_MS} ms, T = {H3O_TEMPERATURE_K} K, "
                  f"p_target = {H3O_P_TARGET}, {H3O_N_PRIMITIVES} THz primitives, nu <= {PAPER_NU_MAX}"]
    else:
        lines.append(f"molecule is {cfg.molecule!r}, not 'h3o': the Eq. 43/44 numbers are reported, "
                     f"not asserted")
    lines.append("raise StructureError if any check fails; write no bulk data")
    return lines


def run(cfg: Config) -> dict:
    molecule = load_molecule(cfg)
    w, trap, task = molecule.window, molecule.trap, molecule.task

    measured: dict = {
        "molecule": molecule.name,
        "fingerprint": molecule.fingerprint(),
        "n_states": int(molecule.n_states),
        "n_blocks": int(len(molecule.blocks)),
        "blocks": block_table(molecule),
        "block_dims": [int(b.n_states) for b in molecule.blocks],
        "eq43_two_m_f": [2 * int(b.n_states) for b in molecule.blocks],
        "trap": {"nu_f_over_2pi_khz": _khz(trap.nu_f), "eta": float(trap.eta),
                 "n_nu": int(trap.n_nu)},
        "truncation": {"paper_nu_max": PAPER_NU_MAX, "paper_n_nu": PAPER_N_NU,
                       "molecule_n_nu": int(trap.n_nu),
                       "matches_paper": bool(trap.n_nu == PAPER_N_NU),
                       "note": "the paper propagates nu <= 1 (two motional levels); "
                               "qlsgym makes n_nu a molecule property"},
        "window": {"omega_min_over_2pi_khz": _khz(w.omega_min),
                   "omega_max_over_2pi_khz": _khz(w.omega_max),
                   "width_khz": _khz(w.omega_max - w.omega_min),
                   "tau_max_ms": float(w.tau_max_ms), "n_tau": int(w.n_tau),
                   "delta_max": float(molecule.provenance.get("delta_max", 1.0e4)),
                   "omega_min_coupling": float(w.omega_min_coupling),
                   "carrier_bands_khz": [[_khz(lo), _khz(hi)] for lo, hi in w.carrier_bands]},
        "task": {"temperature_k": float(task.temperature_k), "p_target": float(task.p_target),
                 "max_pulses": int(task.max_pulses)},
        "primitives": {"n": len(molecule.primitives),
                       "families": sorted({p.label.split()[0] for p in molecule.primitives}),
                       "sigmas": sorted({p.sigma for p in molecule.primitives}),
                       "blocks": sorted({int(b) for p in molecule.primitives for b in p.blocks}),
                       "tau_ms_min": min((float(p.tau_ms) for p in molecule.primitives), default=None),
                       "tau_ms_max": max((float(p.tau_ms) for p in molecule.primitives), default=None)},
        "provenance": {k: str(v) for k, v in molecule.provenance.items()
                       if k in ("levels_file", "rabi_file", "source_package", "paper",
                                "b_field_mt", "j_max", "reference_rabi_khz")},
    }
    measured["parity_partners"] = parity_partners(molecule)
    measured["resonance_geometry"] = resonance_geometry(molecule)
    measured["thermal"] = thermal_summary(molecule)

    active = [r["active_fraction"] for r in measured["resonance_geometry"] if r["n_retained"]]
    measured["active_window_fraction"] = {
        "min": min(active) if active else None, "max": max(active) if active else None,
        "median": float(np.median(active)) if active else None,
        "note": "fraction of the drive window within one half-linewidth of a retained transition; "
                "a uniform-frequency median is dominated by the rest",
    }

    checks: dict = {}
    _universal_checks(molecule, checks)
    if cfg.molecule == "h3o":
        _h3o_checks(molecule, measured, checks)
    measured["checks"] = {k: bool(v) for k, v in checks.items()}

    failed = [k for k, v in measured["checks"].items() if not v]
    if failed:
        raise StructureError(
            f"{molecule.name} (fingerprint {molecule.fingerprint()}) failed {len(failed)} of "
            f"{len(checks)} structural checks: {', '.join(failed)}. The Hamiltonian or the "
            f"control settings are not the paper's; every later stage would quote numbers from "
            f"different physics. Inspect qlsgym/data/{molecule.name}/ and "
            f"qlsgym.molecules.{molecule.name}.")
    return measured
