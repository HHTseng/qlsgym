"""Stage 4 -- forward accuracy of the surrogates, stratified (Fig. 3, Eqs. 27-30)."""

from __future__ import annotations

import os

import numpy as np

from ..config import Config, read_result, run_dir

NAME = "s04_accuracy"
TITLE = "Stratified forward accuracy of the FNO surrogates"
PAPER = "Fig. 3, Eqs. 27-30"
REQUIRES = ("s03_train",)
COST = "~20 min, GPU"

# Fig. 3a plotting rule: the paper draws "only states with a population change
# greater than 1e-4".  Also the active-channel mask of the MRE.
ACTIVE_THRESHOLD = 1e-4

# Percentiles of the Fig. 3b band stack.
BAND_PERCENTILES = (5, 25, 50, 75, 95)

# Most Fig. 3a channels to export per manifold; the rest go to the caption as
# a count.  Bounds the committed JSON without dropping a measurement.
MAX_FIG3A_CHANNELS = 24

# Threshold of the fig3c_frac_below_bound summary.  It is defined here, not
# read from paper.py: a stage computes nothing from a quotation (R5).  It
# happens to equal the bound Sec. III.1 states, which is quoted beside it.
FIG3C_BOUND = 3.0e-3


# small helpers


def _fl(x, sig: int = 6):
    """A float, or a list of them, rounded for a committed JSON."""
    if np.ndim(x) == 0:
        v = float(x)
        return v if not np.isfinite(v) else float(f"{v:.{sig}g}")
    return [_fl(v, sig) for v in np.asarray(x)]


def _n_on(cfg: Config) -> int:
    return int(cfg.overrides.get("n_on", cfg.n_test_freq))


def _n_off(cfg: Config) -> int:
    return int(cfg.overrides.get("n_off", cfg.n_test_freq))


def _n_linewidths(cfg: Config) -> float:
    return float(cfg.overrides.get("n_linewidths", 1.0))


def _freq_seed(cfg: Config) -> int:
    return int(cfg.overrides.get("test_seed", 20260101))


# where stage 3 left the checkpoints


def resolve_checkpoints(cfg: Config, s03: dict) -> dict:
    """(block, sigma) -> checkpoint path, from stage 3's payload alone."""
    from qlsgym.surrogate.manifest import parse_key

    from ..config import pairs as cfg_pairs

    table = {parse_key(str(k)): str(v)
             for k, v in ((s03 or {}).get("checkpoints") or {}).items()}
    out, missing = {}, []
    for block, sigma in cfg_pairs(cfg):
        path = table.get((int(block), sigma))
        if path and os.path.exists(path):
            out[(int(block), sigma)] = path
        else:
            missing.append(f"block {block} sigma{sigma}"
                           + (f" ({path} is gone)" if path else " (not in s03_train.checkpoints)"))
    if missing:
        raise FileNotFoundError(
            "s04_accuracy: no checkpoint for " + "; ".join(missing)
            + " -- run `python replication/run.py s03_train` (or point cfg.tag at the "
              "run that has them)")
    return out


# re-keying the flat stratified summary


def strata_from_summary(summary: dict, test_set: str) -> dict:
    """The {stratum: {...}} block of one test set, from the flat keys
    qlsgym.surrogate.metrics.stratified_summary produces.
    """
    out: dict = {}
    for stratum in ("all", "on", "off"):
        n = summary.get(f"{test_set}_{stratum}_n")
        if not n:
            continue
        g = lambda k: summary.get(f"{test_set}_{stratum}_{k}")  # noqa: E731
        out[stratum] = {
            "n": int(n),
            "median": _fl(g("median")),
            "p95": _fl(g("p95")),
            "max": _fl(g("max")),
            "trivial_median": _fl(g("static_median")),
            "ratio_median": _fl(g("ratio_to_static_median")),
            "ratio_p95": _fl(g("ratio_to_static_p95")),
            "beats_trivial": None,
        }
    return out


def _bands(curves: np.ndarray) -> dict:
    from qlsgym.surrogate.metrics import percentile_bands

    return {str(q): _fl(v) for q, v in percentile_bands(curves, BAND_PERCENTILES).items()}


def _set_payload(ev: dict) -> tuple[dict, dict]:
    """(fig3b_block, fig3c_block) for one test-frequency set."""
    two_pi = 2.0 * np.pi
    fig3b = {"bands": _bands(ev["curves"]),
             "trivial_median": _fl(np.median(ev["static_curves"], axis=0)),
             "n_freq": int(ev["curves"].shape[0])}
    fig3c = {"omega_khz": _fl(np.asarray(ev["omegas"]) / two_pi),
             "time_avg": _fl(ev["time_avg"]),
             "trivial_time_avg": _fl(ev["static_time_avg"]),
             "on_resonance": [bool(v) for v in ev["on_resonance"]],
             "swing": _fl(ev["swing"])}
    return fig3b, fig3c


# Fig. 3a: one showcase trajectory, exact against the surrogate


def showcase_trajectory(model, molecule, block: int, sigma: str, omega: float,
                        n_init: int, seed: int, device) -> dict:
    """Populations vs pulse time at one drive frequency, exact and predicted."""
    import torch

    from qlsgym.surrogate.dataset import (random_mixed_populations,
                                          trajectories_from_columns, transfer_columns)
    from qlsgym.surrogate.embedding import TorchEmbedding
    from qlsgym.surrogate.metrics import (infidelity_curve, mean_relative_error,
                                          static_baseline)

    dev = torch.device(device)
    m = molecule.blocks[block].n_states
    rng = np.random.default_rng(seed)
    p0 = torch.as_tensor(random_mixed_populations(m, n_init, rng, 1.0),
                         dtype=torch.float64, device=dev)
    w = torch.as_tensor([float(omega)], dtype=torch.float64, device=dev)
    cols = transfer_columns(molecule, block, w, sigma, device=dev, out_dtype=torch.float64)[0]
    p_true = trajectories_from_columns(cols[None], p0)
    emb = TorchEmbedding(molecule, block, sigma, dev)
    x = emb.build(p0, w.expand(n_init), out_dtype=torch.float32)
    with torch.no_grad():
        p_pred = model.to(dev).eval()(x).transpose(1, 2).double()

    infid = infidelity_curve(p_pred, p_true).mean(0)
    trivial = static_baseline(p_true).mean(0)
    mre = mean_relative_error(p_pred, p_true, ACTIVE_THRESHOLD).mean(0)
    mean_true = p_true.mean(0).cpu().numpy()
    mean_pred = p_pred.mean(0).cpu().numpy()

    swing = np.abs(mean_true - mean_true[:1, :]).max(axis=0)
    active = np.flatnonzero(swing > ACTIVE_THRESHOLD)
    nu = (active >= m).astype(int)
    keep = []
    for manifold in (0, 1):
        sel = active[nu == manifold]
        keep.extend(sel[np.argsort(-swing[sel])][:MAX_FIG3A_CHANNELS].tolist())
    keep = sorted(keep)
    tau = molecule.tau_grid()

    return {
        "block": int(block), "sigma": sigma,
        "omega_khz": _fl(float(omega) / (2.0 * np.pi)),
        "n_init": int(n_init), "n_channels": int(2 * m),
        "n_active": int(active.size), "n_drawn": len(keep),
        "channel": [int(c) for c in keep],
        "nu": [int(c >= m) for c in keep],
        "tau_ms": _fl(tau),
        "p_true": [_fl(mean_true[:, c]) for c in keep],
        "p_pred": [_fl(mean_pred[:, c]) for c in keep],
        "infidelity": _fl(infid.cpu().numpy()),
        "trivial_infidelity": _fl(trivial.cpu().numpy()),
        "mre": _fl(mre.cpu().numpy()),
        "time_avg_infidelity": _fl(float(infid.mean())),
        "trivial_time_avg_infidelity": _fl(float(trivial.mean())),
        "max_infidelity": _fl(float(infid.max())),
        "time_avg_mre": _fl(float(mre.mean())),
        "max_mre": _fl(float(mre.max())),
        "active_threshold": ACTIVE_THRESHOLD,
    }


def pick_showcase_frequency(raw: dict, molecule, block: int, sigma: str,
                            cfg: Config) -> tuple[float, str]:
    """The drive frequency Fig. 3a is drawn at, and why it was chosen."""
    pinned = cfg.overrides.get("fig3a_omega_khz")
    if pinned is not None:
        return 2.0 * np.pi * float(pinned), "pinned by cfg.overrides['fig3a_omega_khz']"
    for name in ("on", "uniform", "off"):
        ev = raw.get(name)
        if ev is None or not len(ev["omegas"]):
            continue
        k = int(np.argmax(ev["swing"]))
        return float(ev["omegas"][k]), f"largest population swing in the '{name}' test set"
    raise RuntimeError(f"block {block} sigma{sigma}: no test frequencies to draw Fig. 3a at")


# stage protocol


def plan(cfg: Config) -> list:
    """Cheap and side-effect free: never loads a checkpoint."""
    from ..config import pairs as cfg_pairs

    pr = cfg_pairs(cfg)
    lines = [
        f"read s03_train.json for the {len(pr)} trained (block, sigma) pairs: "
        + ", ".join(f"{b}{s}" for b, s in pr),
        f"per pair, evaluate Eqs. 27-30 on three frequency draws: uniform "
        f"({cfg.n_test_freq}), on ({_n_on(cfg)}), off ({_n_off(cfg)}), each with "
        f"{cfg.n_test_init} random mixed initial states "
        f"(on/off at {_n_linewidths(cfg)} half-width; seed {_freq_seed(cfg)})",
        "report per stratum the time-averaged infidelity, the trivial p(tau) = p(0) "
        "predictor on the same frequencies, and the per-frequency ratio "
        "surrogate/trivial, plus the active fraction of the window",
        f"export the {len(BAND_PERCENTILES)}-percentile Fig. 3b bands and the Fig. 3c "
        "scatter per stratum so stage 7 recomputes no physics",
        "draw one Fig. 3a showcase trajectory (exact vs surrogate, infidelity and MRE) "
        "for the pair with the best on-resonance ratio",
        "carry the paper's FIG3 constants beside ours as {'ours', 'paper', 'comparable'}, "
        "with a note wherever the two are not the same quantity",
        f"device {cfg.resolved_device()}; checkpoints from s03_train.json, under "
        + (str(run_dir(cfg, pr[0][0], pr[0][1]).parent) if pr else "$QLSGYM_WORK/runs"),
    ]
    return lines


def run(cfg: Config) -> dict:
    # read_result first: a missing input must surface as MissingResult
    # naming the stage to run, not as an ImportError from something below.
    s03 = read_result("s03_train", cfg)
    from qlsgym.surrogate.metrics import is_on_resonance
    from qlsgym.surrogate.train import evaluate_stratified, load_model

    from .. import paper as P
    from ..config import load_molecule
    from ..config import pairs as cfg_pairs

    molecule = load_molecule(cfg)
    device = cfg.resolved_device()
    ckpts = resolve_checkpoints(cfg, s03)

    freq_seed, n_lw = _freq_seed(cfg), _n_linewidths(cfg)
    pairs_out: list = []
    models: dict = {}              # live torch modules: never in the payload
    raw: dict = {}                 # per-pair evaluation arrays; only Fig. 3a needs them
    pooled_curves: dict = {}
    pooled_static: dict = {}
    pooled_time: dict = {}
    pooled_static_time: dict = {}

    for block, sigma in cfg_pairs(cfg):
        path = ckpts[(block, sigma)]
        model = load_model(path, device=device, molecule=molecule)
        summary, ev_sets = evaluate_stratified(
            model, molecule, block, sigma, n_uniform=cfg.n_test_freq, n_on=_n_on(cfg),
            n_off=_n_off(cfg), n_init=cfg.n_test_init, freq_seed=freq_seed,
            init_seed=freq_seed + 1, n_linewidths=n_lw, device=device)

        entry: dict = {
            "block": int(block), "sigma": sigma,
            "n_states": int(molecule.blocks[block].n_states),
            "checkpoint": path,
            "active_fraction": _fl(summary["active_fraction"]),
            "n_linewidths": n_lw,
            "strata": {}, "swing_median": {},
            "fig3b": {"tau_ms": _fl(molecule.tau_grid())}, "fig3c": {},
        }
        for name in ("uniform", "on", "off"):
            ev = ev_sets[name]
            entry["strata"][name] = strata_from_summary(summary, name)
            entry["swing_median"][name] = _fl(summary.get(f"{name}_swing_median"))
            for stratum, blk in entry["strata"][name].items():
                sel = {"all": np.ones_like(ev["on_resonance"]),
                       "on": ev["on_resonance"], "off": ~ev["on_resonance"]}[stratum]
                blk["beats_trivial"] = _fl(float(
                    np.mean(ev["time_avg"][sel] < ev["static_time_avg"][sel]))) if sel.any() else None
            b3, c3 = _set_payload(ev)
            entry["fig3b"][name] = b3
            entry["fig3c"][name] = c3
            pooled_curves.setdefault(name, []).append(ev["curves"])
            pooled_static.setdefault(name, []).append(ev["static_curves"])
            pooled_time.setdefault(name, []).append(ev["time_avg"])
            pooled_static_time.setdefault(name, []).append(ev["static_time_avg"])

        models[(block, sigma)] = model
        raw[(block, sigma)] = ev_sets
        pairs_out.append(entry)

    # Fig. 3a on the pair that learned the most on resonance
    def _on_ratio(e):
        r = e["strata"]["on"].get("all", {}).get("ratio_median")
        return float("inf") if r is None else float(r)

    showcase = min(pairs_out, key=_on_ratio)
    key = (showcase["block"], showcase["sigma"])
    omega, why = pick_showcase_frequency(raw[key], molecule, *key, cfg)
    fig3a = showcase_trajectory(models[key], molecule, *key, omega, cfg.n_test_init,
                                freq_seed + 2, device)
    fig3a["selected_by"] = why
    fig3a["on_resonance"] = bool(is_on_resonance(
        molecule, np.asarray([omega]), showcase["block"], showcase["sigma"], n_lw)[0])

    # pooled over pairs
    pooled: dict = {"n_pairs": len(pairs_out), "strata": {},
                    "fig3b": {"tau_ms": _fl(molecule.tau_grid())}, "fig3c": {}}
    for name in ("uniform", "on", "off"):
        cur = np.concatenate(pooled_curves[name])
        sta = np.concatenate(pooled_static[name])
        ta = np.concatenate(pooled_time[name])
        st = np.concatenate(pooled_static_time[name])
        ratio = ta / np.maximum(st, 1e-30)
        pooled["strata"][name] = {"all": {
            "n": int(ta.size), "median": _fl(np.median(ta)),
            "p95": _fl(np.percentile(ta, 95)), "max": _fl(np.max(ta)),
            "trivial_median": _fl(np.median(st)),
            "ratio_median": _fl(np.median(ratio)),
            "ratio_p95": _fl(np.percentile(ratio, 95)),
            "beats_trivial": _fl(float(np.mean(ta < st))),
        }}
        pooled["fig3b"][name] = {"bands": _bands(cur),
                                 "trivial_median": _fl(np.median(sta, axis=0)),
                                 "n_freq": int(cur.shape[0])}
        pooled["fig3c"][name] = {
            "omega_khz": _fl(np.concatenate(
                [np.asarray(e["fig3c"][name]["omega_khz"]) for e in pairs_out])),
            "time_avg": _fl(ta), "trivial_time_avg": _fl(st),
            "on_resonance": [b for e in pairs_out for b in e["fig3c"][name]["on_resonance"]]}

    pooled["active_fraction_median"] = _fl(
        float(np.median([e["active_fraction"] for e in pairs_out])))

    uni = pooled["strata"]["uniform"]["all"]
    band50 = np.asarray(pooled["fig3b"]["uniform"]["bands"]["50"], dtype=float)
    band95 = np.asarray(pooled["fig3b"]["uniform"]["bands"]["95"], dtype=float)

    return {
        "protocol": {
            "n_test_freq": cfg.n_test_freq, "n_on": _n_on(cfg), "n_off": _n_off(cfg),
            "n_test_init": cfg.n_test_init, "n_linewidths": n_lw,
            "freq_seed": freq_seed, "init_seed": freq_seed + 1,
            "device": device, "checkpoints_from": "s03_train.checkpoints",
            "band_percentiles": list(BAND_PERCENTILES),
            "active_threshold": ACTIVE_THRESHOLD,
            "equations": "Eq. 27 F_p, Eq. 28 I_p, Eq. 29 ensemble mean, Eq. 30 time average",
            "reporting": (
                "every infidelity is reported beside the trivial p(tau) = p(0) predictor on "
                "the same frequencies and as the per-frequency ratio surrogate/trivial, "
                "stratified on and off resonance; ratio >= 1 on resonance "
                "means nothing was learned, whatever the absolute infidelity says"),
            "pooling": (
                "'pooled' concatenates the per-block test frequencies; it is NOT the paper's "
                "full-space quantity over all ten blocks, which would combine the blocks by "
                "Eq. 18 with their population weights"),
            "mre": (
                "mean relative error over the channels whose true population moves by more "
                f"than {ACTIVE_THRESHOLD:g}; the ratio is undefined where the reference "
                "population vanishes and qlsgym clamps the denominator at 1e-12"),
        },
        "pairs": pairs_out,
        "pooled": pooled,
        "showcase": {"block": showcase["block"], "sigma": showcase["sigma"],
                     "on_resonance_ratio_median": _on_ratio(showcase)},
        "fig3a": fig3a,
        "comparison": {
            # Every pair is {ours, paper}; 'comparable' says whether the two are
            # the same quantity, and when they are not the note says how they
            # differ.  A paper value is a quotation and nothing here is computed
            # from one.
            "fig3b_median": {
                "ours": _fl(np.median(band50)), "paper": P.FIG3B_MEDIAN,
                "comparable": False,
                "note": "ours pools the test frequencies of the trained blocks; the "
                        "paper's median is over the full 444-state space (Eq. 18)"},
            "fig3b_p95": {
                "ours": _fl(np.max(band95)), "paper": P.FIG3B_P95_BOUND,
                "comparable": False,
                "note": "same pooling caveat as fig3b_median; ours is the worst tau of "
                        "the 95th-percentile curve, the paper states a bound"},
            "fig3a_time_avg_infidelity": {
                "ours": fig3a["time_avg_infidelity"], "paper": P.FIG3A_TIME_AVG_INFIDELITY,
                "comparable": False,
                "note": f"ours is block {showcase['block']} sigma{showcase['sigma']} alone at "
                        "one drive frequency; the paper's number is over all 444 states"},
            "mre": {
                "ours": fig3a["time_avg_mre"], "paper": P.FIG3A_MRE_BOUND,
                "comparable": False,
                "note": "ours is over the channels of the showcase block that move by more "
                        f"than {ACTIVE_THRESHOLD:g}; the paper's is over its plotted states"},
            "fig3c_frac_below_bound": {
                "ours": _fl(float(np.mean(
                    np.asarray(pooled["fig3c"]["uniform"]["time_avg"], dtype=float)
                    < FIG3C_BOUND))),
                "threshold": FIG3C_BOUND,
                "paper": P.FIG3C_TYPICAL_BOUND,
                "comparable": False,
                "note": "the paper states the bound and 'most test frequencies', not a "
                        "fraction; ours is the fraction below our own threshold"},
            "n_init": {
                "ours": cfg.n_test_init, "paper": P.FIG3B_L_INIT,
                "comparable": True},
            "uniform_median": {
                "ours": uni["median"], "paper": P.FIG3B_MEDIAN,
                "comparable": False,
                "note": "ours is the median over test frequencies of the time-averaged "
                        "infidelity; the paper's is the median of the curve at each tau, "
                        "and over the full space"},
        },
    }
