"""Stage 5 -- the Eq. 25 score planner with three engines in the loop (Sec. II.4, Fig. 5)."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np

from ..config import Config, read_result, resolve_blocks, work_dir

NAME = "s05_planner"
TITLE = "FNO-SPMP vs exact vs trivial, in the loop at a matched budget"
PAPER = "Sec. II.4, Fig. 5, Eq. 25"
REQUIRES = ("s03_train",)
COST = "~60 CPU-h, CPU (array of 15 resumable (arm, seed) units)"

# The arms, in report order.  exact first because it is the reference the
# other two are a degradation of.
ARMS = ("exact", "trivial", "fno")

# Production planner settings of the run the target numbers come from
# (OUTPUTS/ranking/inloop_comparison.json, settings).  They are the
# settings the comparison was measured at, not qlsgym's ScoreConfig
# defaults; cfg.overrides moves any of them.
SCORE_DEFAULTS = {"w_tr": 0.5, "w_br": 1.5, "w1_scale": 10.0, "br_mode": "best"}
GRID_DEFAULTS = {"d_omega_khz": 1.0, "tau_step": 1}

# Matched sample size: the number of planning seeds every arm is compared at.
N_SEEDS = 5

# Milliseconds per drive evaluation of ScorePlannerPolicy against
# ExactEngine -- H3O+, tau_step = 1 (101 tau rows), one core.
# **Measured, not guessed**: warm, a drive costs 2.4 ms at d_omega =
# 12.5 kHz (1.4 GB of cached columns) and 3.1 ms at 5.0 kHz (3.4 GB),
# because its (101, 2 M, M) column block is ~36 MB and is re-read from RAM
# every time; the production 1 kHz cache is ~17 GB, so the value below carries
# that trend plus a margin.  The *first* visit to a drive costs ~0.13 s (it
# builds the columns), once per engine per drive -- ~70 s per arm, under 1 % of
# a unit -- and is deliberately not in the constant.
MS_PER_DRIVE = 4.0

# Pulses a rollout spends, for the *estimate* only.  The previous
# replication's exact arm averaged 20.203 over its first five seeds, with
# 12.4 % of rollouts spending the full 80
# (FNO_REPL/OUTPUTS/fig5/dw1.0.json); coarse-grid runs here spend 15-24.
MEAN_PULSES = 20.2

# Environment variable selecting which units this process runs.  Unset or
# empty: every unit, computing whatever is not checkpointed yet.  A
# comma-separated list of indices into units (3, 0,4, 0-4):
# only those, which is how the SLURM array splits the work.  none: compute
# nothing, assemble the checkpoints that exist and fail naming those that do
# not -- the cheap login-node aggregation pass.
UNITS_ENV = "QLSGYM_REPL_S05_UNITS"

# Quotations from the previous replication (NOT from arXiv:2608.03702).
# FNO_REPL/paper/replication.tex, Sec. "The registered control": "Matched
# at n=5 [...] the three arms are 0.8452+-0.0076 (exact), 0.8134+-0.0061
# (trivial) and 0.6972+-0.0666 (FNO)."  The +- is the sample standard
# deviation over the five seeds.  Drawn and tabulated beside ours, never in
# place of ours (the same rule as paper.py),
# and read caveat before putting the two columns in one sentence.
PRIOR_REPLICATION = {
    "source": "FNO_REPL/paper/replication.tex, Sec. 'The registered control' and Tab. 'loop'",
    "n_seeds": 5,
    "uncertainty": "sample standard deviation over seeds (ddof=1)",
    "n_rollouts_per_seed": 1000,
    "max_pulses": 80,
    "d_omega_khz": 1.0,
    "caveat": (
        "NOT like-for-like with headline.ours, although it sits beside it.  The "
        "statistical protocol does match: 5 planning seeds, matched seed count, '+-' "
        "is the sample sd with ddof=1, and the quoted number is exact propagation of "
        "the chosen pulses.  The planner does not.  The prior arms were FNO-SPMP, an "
        "OFFLINE sequence planned once per seed and then replayed "
        "(FNO_REPL/OUTPUTS/fig5/dw1.0.json: side_depth 3, expand_both 1, thz 2, "
        "main_branch_len 80, ~268 s of planning per seed), evaluated with 1000 "
        "rollouts per seed over a 54944-action library, and its 10 seeds were "
        "truncated to the first 5 to match.  Ours is ScorePlannerPolicy, a one-step "
        "greedy re-planner with no tree and no lookahead, evaluated with 200 rollouts "
        "per seed over 50723 actions.  A gap between the two columns is a difference "
        "of planner and of rollout count as much as of engine."),
    "arms": {
        "exact": {"validated": 0.8452, "sd": 0.0076, "view": None,
                  "substituted_fraction": 0.0, "mean_pulses_successful": None},
        "trivial": {"validated": 0.8134, "sd": 0.0061, "view": 0.815,
                    "substituted_fraction": 0.276, "mean_pulses_successful": 10.5},
        "fno": {"validated": 0.6972, "sd": 0.0666, "view": 0.809,
                "substituted_fraction": 0.254, "mean_pulses_successful": 29.7},
    },
}


# protocol pieces


def seeds_of(cfg: Config) -> list:
    n = int(cfg.overrides.get("n_seeds", N_SEEDS))
    return [int(cfg.seed) + i for i in range(n)]


def wants_view(cfg: Config) -> bool:
    """Whether the calibration pass runs."""
    return bool(cfg.overrides.get("view", True))


def score_config(cfg: Config, molecule):
    from qlsgym.policies.score import ScoreConfig

    kw = {**SCORE_DEFAULTS, **{k: v for k, v in cfg.overrides.items() if k in SCORE_DEFAULTS}}
    return ScoreConfig(**kw).for_molecule(molecule)


def control_grid(cfg: Config, molecule):
    """The Eq. 23 uniform grid at the configured spacing."""
    from qlsgym.env.actions import ControlGrid

    return ControlGrid.uniform(
        molecule,
        d_omega_khz=float(cfg.overrides.get("d_omega_khz", GRID_DEFAULTS["d_omega_khz"])),
        tau_lo=cfg.overrides.get("tau_lo"), tau_hi=cfg.overrides.get("tau_hi"),
        tau_step=int(cfg.overrides.get("tau_step", GRID_DEFAULTS["tau_step"])))


def build_library(cfg: Config, molecule):
    """The library every arm searches, and the tau rows the engines must carry."""
    from qlsgym.env.actions import ActionLibrary

    try:
        lib = ActionLibrary(molecule, control_grid(cfg, molecule), include_primitives=True)
    except ValueError as exc:
        raise ValueError(
            f"{exc}  d_omega_khz = "
            f"{cfg.overrides.get('d_omega_khz', GRID_DEFAULTS['d_omega_khz'])} does not "
            f"give {molecule.name} a usable grid; move it with "
            f"Config(overrides={{'d_omega_khz': ...}}).") from exc
    ti = set(int(t) for t in lib.tau_indices)
    ti.update(int(lib.primitive_tau_index(k)) for k in range(lib.n_primitives))
    return lib, np.asarray(sorted(ti), dtype=np.int64)


def drives_per_pulse(molecule, lib) -> int:
    """Distinct (sigma, omega) the planner scores at *every* pulse."""
    keys = {(sg, round(float(w), 6)) for sg, w, _a, _t in lib.drives()}
    keys |= {(q.sigma, round(float(q.omega), 6)) for q in molecule.primitives}
    return len(keys)


def cache_estimate_gb(molecule, grid, n_tau_rows: int) -> float:
    """Rough upper bound on one engine's per-drive column cache."""
    per_drive = sum(2 * b.n_states ** 2 for b in molecule.blocks) * n_tau_rows * 8
    return per_drive * grid.n_freq * len(grid.sigmas) / 1024 ** 3


def cost_estimate(cfg: Config, molecule, lib) -> dict:
    """Projected CPU cost of this configuration, from MS_PER_DRIVE."""
    nd = drives_per_pulse(molecule, lib)
    pulses = min(MEAN_PULSES, float(cfg.max_pulses))
    evals = float(cfg.n_rollouts) * pulses * nd
    hours = evals * MS_PER_DRIVE / 1e3 / 3600.0
    n_seeds = len(seeds_of(cfg))
    passes = len(ARMS) + (len(ARMS) - 1 if wants_view(cfg) else 0)
    return {"drives_per_pulse": nd, "mean_pulses_assumed": pulses,
            "ms_per_drive": MS_PER_DRIVE, "drive_evals_per_pass": int(evals),
            "hours_per_pass": hours, "passes_per_seed": passes,
            "hours_per_unit_max": hours * (2 if wants_view(cfg) else 1),
            "n_units": len(ARMS) * n_seeds, "cpu_hours": hours * passes * n_seeds}


def make_engine(arm: str, cfg: Config, molecule, tau_indices, exact,
                manifest: str | None = None):
    """One arm's in-loop engine."""
    if arm == "exact":
        return exact
    if arm == "trivial":
        from qlsgym.physics.engines import TrivialEngine

        return TrivialEngine(molecule, tau_indices=tau_indices,
                             blocks=list(resolve_blocks(cfg, molecule)),
                             sigmas=tuple(cfg.sigmas))
    if arm == "fno":
        from qlsgym.surrogate.manifest import load_manifest, manifest_path

        path = manifest if manifest and os.path.exists(manifest) else None
        try:
            return load_manifest(molecule, cfg.tag, tau_indices=tau_indices,
                                 device=cfg.resolved_device(), fallback=exact, path=path)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"s05_planner: the 'fno' arm needs the surrogate manifest "
                f"{path or manifest_path(molecule.name, cfg.tag)}, which stage 3 writes; "
                f"run `python replication/run.py s03_train` (or point cfg.tag at the run "
                f"that has it)") from exc
    raise ValueError(f"unknown arm {arm!r}")


def engine_calls(engine) -> dict | None:
    """The per-category call counters an approximate engine keeps, or None."""
    calls = getattr(engine, "calls", None)
    return {k: int(v) for k, v in calls.items()} if isinstance(calls, dict) else None


def _delta(before: dict | None, after: dict | None) -> dict | None:
    if before is None or after is None:
        return None
    return {k: int(after.get(k, 0)) - int(before.get(k, 0)) for k in after}


def _substituted_fraction(calls: dict | None) -> float | None:
    """Share of branch evaluations the approximate engine answered itself."""
    if not calls:
        return None
    total = sum(calls.values())
    if total <= 0:
        return None
    served = calls.get("fno", 0) + calls.get("substituted", 0)
    return served / total


def _finite(x):
    """A JSON number, or None."""
    if x is None:
        return None
    x = float(x)
    return x if np.isfinite(x) else None


def stats(values) -> dict:
    """Mean over seeds with the sample sd (ddof = 1) the target quotes."""
    v = np.asarray(values, dtype=float)
    sd = float(v.std(ddof=1)) if v.size > 1 else float("nan")
    return {"n": int(v.size), "per_seed": [float(x) for x in v],
            "mean": _finite(v.mean()) if v.size else None,
            "sd": _finite(sd), "sem": _finite(sd / np.sqrt(v.size)) if v.size > 1 else None}


def contrast(a, b) -> dict:
    """Both tests over the seed list."""
    from scipy.stats import ttest_ind, ttest_rel

    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    out = {"diff": _finite(a.mean() - b.mean())}
    if a.size > 1 and b.size > 1:
        w = ttest_ind(a, b, equal_var=False)
        out.update(welch_t=_finite(w.statistic), welch_p=_finite(w.pvalue))
        if a.size == b.size:
            r = ttest_rel(a, b)
            out.update(paired_t=_finite(r.statistic), paired_p=_finite(r.pvalue))
    return out


# units: one (arm, seed) of work, checkpointed the moment it finishes


def units(cfg: Config) -> list:
    """Every (arm, seed) unit in a fixed order: index i is array task i."""
    return [(arm, seed) for arm in ARMS for seed in seeds_of(cfg)]


def selected_units(cfg: Config) -> tuple:
    """(units to assemble, may this process compute them) -- see UNITS_ENV."""
    spec = os.environ.get(UNITS_ENV, "").strip().lower()
    every = units(cfg)
    if not spec:
        return every, True
    if spec == "none":
        return every, False
    want: set = set()
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        lo, _, hi = part.partition("-")
        want.update(range(int(lo), int(hi) + 1) if hi else [int(lo)])
    bad = sorted(i for i in want if not 0 <= i < len(every))
    if bad:
        raise ValueError(f"${UNITS_ENV}={spec!r} selects unit(s) {bad}, but this "
                         f"configuration has {len(every)} (0-{len(every) - 1})")
    return [every[i] for i in sorted(want)], True


def unit_signature(cfg: Config, molecule, arm: str, seed: int,
                   surrogate: str | None = None) -> dict:
    """Everything a unit's number depends on."""
    def pick(defaults):
        return {**defaults, **{k: v for k, v in cfg.overrides.items() if k in defaults}}

    return {
        "stage": NAME, "arm": arm, "seed": int(seed),
        "molecule": molecule.name, "fingerprint": molecule.fingerprint(), "tag": cfg.tag,
        "n_rollouts": int(cfg.n_rollouts), "max_pulses": int(cfg.max_pulses),
        "view": wants_view(cfg),
        "blocks": [int(b) for b in resolve_blocks(cfg, molecule)],
        "sigmas": [str(s) for s in cfg.sigmas],
        "score": pick(SCORE_DEFAULTS),
        "grid": {**pick(GRID_DEFAULTS), "tau_lo": cfg.overrides.get("tau_lo"),
                 "tau_hi": cfg.overrides.get("tau_hi")},
        "surrogate": surrogate if arm == "fno" else None,
    }


def surrogate_stamp(cfg: Config, molecule, manifest: str | None) -> str | None:
    """name:mtime of the manifest the fno arm would load, so that retraining invalidates its
    checkpoints instead of being reused.
    """
    from qlsgym.surrogate.manifest import manifest_path

    path = manifest if manifest and os.path.exists(manifest) else str(
        manifest_path(molecule.name, cfg.tag))
    return (f"{os.path.basename(path)}:{int(os.path.getmtime(path))}"
            if os.path.exists(path) else None)


def unit_dir() -> Path:
    """Where finished units live: $QLSGYM_WORK, never the output tree (R3)."""
    return work_dir() / "repl_units" / NAME


def unit_path(sig: dict) -> Path:
    h = hashlib.sha1(json.dumps(sig, sort_keys=True, default=str).encode()).hexdigest()[:10]
    return unit_dir() / f"{sig['molecule']}_{sig['tag']}_{sig['arm']}_seed{sig['seed']}_{h}.json"


def load_unit(sig: dict) -> dict | None:
    """The finished unit on disk, or None if it is absent, unreadable, or was computed at a
    different signature.
    """
    try:
        doc = json.loads(unit_path(sig).read_text())
    except (OSError, ValueError):
        return None
    return doc.get("result") if doc.get("signature") == sig else None


def save_unit(sig: dict, result: dict) -> Path:
    path = unit_path(sig)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"signature": sig, "result": result}, indent=1))
    os.replace(tmp, path)
    return path


def run_unit(cfg: Config, molecule, lib, cfg_score, exact, engine, arm: str, seed: int) -> dict:
    """One (arm, seed): the validated pass and, for an approximate arm, view."""
    from qlsgym.policies.baselines import ScorePlannerPolicy
    from qlsgym.policies.rollout import rollout

    t0, before = time.time(), engine_calls(engine)
    kw = dict(n_rollouts=cfg.n_rollouts, seed=int(seed), max_pulses=cfg.max_pulses,
              p_target=float(molecule.task.p_target), library=lib)

    def planner():
        return ScorePlannerPolicy(engine, lib, cfg_score, tau_mode="library")

    r = rollout(exact, planner(), **kw)
    view = None
    if wants_view(cfg) and arm != "exact":
        view = float(rollout(engine, planner(), **kw).success_fraction)
    return {"arm": arm, "seed": int(seed), "validated": float(r.success_fraction),
            "view": view, "mean_pulses": float(r.mean_pulses),
            "mean_pulses_successful": _finite(r.mean_pulses_successful),
            "calls": _delta(before, engine_calls(engine)),
            "seconds": float(time.time() - t0)}


def _arm_payload(arm: str, recs: list, engine_name: str, sub_blocks: list) -> dict:
    """One arm's block of the payload, assembled from its finished units."""
    val = stats([r["validated"] for r in recs])
    views = [r["view"] for r in recs if r["view"] is not None]
    view = stats(views) if views else None
    calls: dict = {}
    for r in recs:
        for k, v in (r["calls"] or {}).items():
            calls[k] = calls.get(k, 0) + int(v)
    ok = [r["mean_pulses_successful"] for r in recs if r["mean_pulses_successful"] is not None]
    return {
        "engine": engine_name,
        "validated": val,
        "view": view,
        "calibration_gap": (None if view is None or view["mean"] is None or val["mean"] is None
                            else float(view["mean"] - val["mean"])),
        "mean_pulses": _finite(np.mean([r["mean_pulses"] for r in recs])),
        "mean_pulses_per_seed": [float(r["mean_pulses"]) for r in recs],
        "mean_pulses_successful": _finite(np.mean(ok)) if ok else None,
        "calls": calls or None,
        "substituted_fraction": _substituted_fraction(calls),
        "substituted_blocks": None if arm == "exact" else sub_blocks,
        "seconds": float(sum(r["seconds"] for r in recs)),
        "prior_replication": PRIOR_REPLICATION["arms"][arm],
    }


# stage protocol


def plan(cfg: Config) -> list:
    from ..config import load_molecule

    mol = load_molecule(cfg)
    grid = control_grid(cfg, mol)
    lib, tau_indices = build_library(cfg, mol)
    sd = seeds_of(cfg)
    blocks = list(resolve_blocks(cfg, mol))
    fallback = sorted(set(range(len(mol.blocks))) - set(blocks))
    est = cost_estimate(cfg, mol, lib)
    todo, may_compute = selected_units(cfg)
    stamp = surrogate_stamp(cfg, mol, None)
    done = sum(load_unit(unit_signature(cfg, mol, *u, surrogate=stamp)) is not None for u in todo)
    return [
        "read s03_train.json, then load the manifest of surrogates for the 'fno' arm",
        f"library: {lib.n_actions} actions ({grid.n_freq} frequencies x "
        f"{grid.n_tau_slots} durations x {len(grid.sigmas)} polarisations + "
        f"{lib.n_primitives} primitives); {est['drives_per_pulse']} drives scored per "
        f"pulse, {tau_indices.size} tau rows carried by every engine",
        f"score config (Eq. 25): {score_config_repr(cfg)}",
        f"three arms {ARMS} x {len(sd)} planning seeds {sd} x {cfg.n_rollouts} rollouts, "
        f"max {cfg.max_pulses} pulses; each arm "
        + ("twice -- ExactEngine advances the belief for the reported number, the "
           "engine itself for its view" if wants_view(cfg) else
           "once, view OFF -- ExactEngine advances the belief"),
        "matched: identical library, score weights, seeds, rollout count, pulse budget, "
        "thermal initial belief and substituted (block, sigma) footprint across arms",
        "uncertainty: mean over seeds, +- the sample sd (ddof=1); the paired t is the "
        "headline contrast (the arms share one outcome stream), Welch's t sits beside it",
        f"substituted footprint: blocks {blocks} x sigmas {list(cfg.sigmas)}; "
        + (f"blocks {fallback} (the parity partners qlsgym does not relabel), every "
           if fallback else
           "no block falls back on parity (every block of this molecule has distinct "
           "dynamics), but every ")
        + "sigma- pulse and every off-window primitive stay exact -- "
        "arms.<arm>.calls shows the share actually substituted",
        f"cost: {est['drives_per_pulse']} drives/pulse x {cfg.n_rollouts} rollouts x "
        f"~{est['mean_pulses_assumed']:.1f} pulses = "
        f"{est['drive_evals_per_pass'] / 1e6:.2f}M drive evaluations per pass at "
        f"{MS_PER_DRIVE:.1f} ms each (measured) -> ~{est['hours_per_pass']:.1f} h per pass, "
        f"<= {est['hours_per_unit_max']:.1f} h per (arm, seed) unit, "
        f"~{est['cpu_hours']:.0f} CPU-hours in total "
        f"({est['passes_per_seed']} passes x {len(sd)} seeds); "
        f"~{cache_estimate_gb(mol, grid, tau_indices.size):.0f} GB of cached columns per "
        f"engine, and an approximate arm holds two (its own and the exact validator's)",
        f"units: {done}/{len(todo)} selected of {len(units(cfg))} are already "
        f"checkpointed under {unit_dir()}; "
        + (f"this process assembles them and computes nothing more"
           if done == len(todo) else
           f"this process {'computes' if may_compute else 'refuses to compute'} the "
           f"remaining {len(todo) - done}")
        + f" (${UNITS_ENV}={os.environ.get(UNITS_ENV) or 'unset'})",
    ]


def score_config_repr(cfg: Config) -> str:
    kw = {**SCORE_DEFAULTS, **{k: v for k, v in cfg.overrides.items() if k in SCORE_DEFAULTS}}
    return ", ".join(f"{k}={v}" for k, v in kw.items())


def run(cfg: Config) -> dict:
    # The unit selection is checked before anything is read: a subset produces a
    # partial payload, which must never land on the real result.
    todo, may_compute = selected_units(cfg)
    if len(todo) != len(units(cfg)) and not os.environ.get("QLSGYM_REPL_OUTPUTS"):
        raise RuntimeError(
            f"${UNITS_ENV} restricts this run to {len(todo)} of {len(units(cfg))} units, so "
            f"its payload is partial; point $QLSGYM_REPL_OUTPUTS at a per-task directory "
            f"first so it cannot overwrite the real {NAME}.json (slurm/planner.sbatch does)")
    # then read_result: a missing input must surface as MissingResult naming
    # the stage to run, not as an ImportError from something below.
    s03 = read_result("s03_train", cfg)

    from .. import paper as P
    from ..config import load_molecule

    molecule = load_molecule(cfg)
    lib, tau_indices = build_library(cfg, molecule)
    cfg_score = score_config(cfg, molecule)
    sd = seeds_of(cfg)
    p_target = float(molecule.task.p_target)
    sub_blocks = [int(b) for b in resolve_blocks(cfg, molecule)]

    s03_manifest = s03.get("manifest") if isinstance(s03, dict) else None
    stamp = surrogate_stamp(cfg, molecule, s03_manifest)
    sigs = {u: unit_signature(cfg, molecule, *u, surrogate=stamp) for u in todo}
    cached = {u: load_unit(sigs[u]) for u in todo}
    pending = [u for u in todo if cached[u] is None]
    if pending and not may_compute:
        raise RuntimeError(
            f"${UNITS_ENV}=none assembles finished units only, and {len(pending)} of "
            f"{len(todo)} are missing: "
            f"{', '.join(f'{a}/seed {s}' for a, s in pending[:8])}"
            f"{' ...' if len(pending) > 8 else ''}.  Re-submit array task(s) "
            f"{[units(cfg).index(u) for u in pending]} and aggregate again.")

    exact, engines = None, {}
    if pending:
        from qlsgym.physics.engines import ExactEngine

        est = cost_estimate(cfg, molecule, lib)
        print(f"  {len(todo) - len(pending)}/{len(todo)} units cached; computing "
              f"{len(pending)} at <= {est['hours_per_unit_max']:.1f} h each "
              f"(~{est['hours_per_unit_max'] * len(pending):.0f} CPU-hours)", flush=True)
        # every engine before any rollout: a missing manifest must fail in
        # seconds, not after the first unit has spent its CPU-hours.
        exact = ExactEngine(molecule, tau_indices=tau_indices)
        engines = {a: make_engine(a, cfg, molecule, tau_indices, exact, s03_manifest)
                   for a in sorted({a for a, _ in pending})}

    recs: dict = {a: [] for a in ARMS}
    for unit in todo:
        arm, seed = unit
        rec = cached[unit]
        if rec is None:
            rec = run_unit(cfg, molecule, lib, cfg_score, exact, engines[arm], arm, seed)
            print(f"  {arm}/seed {seed}: validated {rec['validated']:.4f} in "
                  f"{rec['seconds'] / 60:.1f} min -> {save_unit(sigs[unit], rec)}", flush=True)
        recs[arm].append(rec)

    default_names = {"exact": "ExactEngine", "trivial": "TrivialEngine", "fno": "FnoEngine"}
    arms = {a: _arm_payload(a, recs[a],
                            type(engines[a]).__name__ if a in engines else default_names[a],
                            sub_blocks)
            for a in ARMS if recs[a]}
    per_seed = {a: arms[a]["validated"]["per_seed"] for a in arms}

    return {
        "protocol": {
            "arms": list(ARMS),
            "n_seeds": len(sd), "seeds": sd,
            "n_rollouts_per_seed": cfg.n_rollouts, "max_pulses": cfg.max_pulses,
            "p_target": p_target,
            "score": {**SCORE_DEFAULTS,
                      **{k: v for k, v in cfg.overrides.items() if k in SCORE_DEFAULTS},
                      "p_target": cfg_score.p_target},
            "library": {"kind": "uniform grid + off-window primitives",
                        "d_omega_khz": float(cfg.overrides.get(
                            "d_omega_khz", GRID_DEFAULTS["d_omega_khz"])),
                        "n_actions": int(lib.n_actions), "n_grid": int(lib.n_grid),
                        "n_primitives": int(lib.n_primitives),
                        "n_drives": len(lib.drives()),
                        "drives_scored_per_pulse": drives_per_pulse(molecule, lib),
                        "tau_rows": int(tau_indices.size)},
            "initial_belief": "thermal (qlsgym.env.env.boltzmann_belief)",
            "headline_is": "arms.<arm>.validated.mean",
            "headline_test": "contrasts.<pair>.paired_p (see s05_planner.contrast)",
            "view": wants_view(cfg),
            "substituted_footprint": {"blocks": sub_blocks, "sigmas": list(cfg.sigmas)},
            "exact_fallback_blocks": sorted(set(range(len(molecule.blocks))) - set(sub_blocks)),
            "note": ("The arms are one planner at one setting and differ only in the "
                     "in-loop engine. 'validated' (ExactEngine advancing the belief) is the "
                     "result; 'view' (the arm's own engine advancing it) is calibration only "
                     "(R7). 'sd' is the sample sd, ddof=1, over seeds. exact_fallback_blocks, "
                     "every sigma- pulse and every off-window primitive stay exact in the "
                     "approximate arms."),
        },
        "units": {"total": len(units(cfg)), "assembled": len(todo),
                  "computed_here": len(pending),
                  "complete": len(todo) == len(units(cfg)),
                  "checkpoints": os.fspath(unit_dir()),
                  "cost": cost_estimate(cfg, molecule, lib)},
        "arms": arms,
        "contrasts": {f"{a}_vs_{b}": contrast(per_seed[a], per_seed[b])
                      for a, b in (("exact", "trivial"), ("exact", "fno"), ("trivial", "fno"))
                      if a in per_seed and b in per_seed},
        "headline": {
            "ours": {a: {"validated": arms[a]["validated"]["mean"],
                         "sd": arms[a]["validated"]["sd"]} for a in arms},
            "prior_replication": PRIOR_REPLICATION,
            "cost_of_approximation": {
                a: _finite(arms["exact"]["validated"]["mean"] - arms[a]["validated"]["mean"])
                for a in ("trivial", "fno") if a in arms and "exact" in arms
            },
        },
        "paper": {
            "fig5a_dw1.0": P.FIG5A_PAPER["dw1.0"],
            "p_target": P.P_TARGET,
            "max_pulses": P.MAX_PULSES,
            "n_mc_rollouts": P.N_MC_ROLLOUTS,
            "note": ("the paper's Fig. 5a numbers are its own FNO-SPMP tree search, not "
                     "this one-step re-planning loop; they are quoted for scale, not as a "
                     "like-for-like target"),
        },
        "s03_tag": s03.get("tag", cfg.tag) if isinstance(s03, dict) else cfg.tag,
    }
