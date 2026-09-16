"""The measurement and figure half of the replication: stages 4, 5 and 7."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from replication import config as cfgmod                     # noqa: E402
from replication.config import Config, MissingResult         # noqa: E402
from replication.stages import base                          # noqa: E402

MINE = ("s04_accuracy", "s05_planner", "s07_figures")

@pytest.fixture(scope="session")
def paper():
    """The constants stage 7 quotes, from the real replication/paper.py."""
    from replication.stages.s07_figures import paper_constants

    return paper_constants()


@pytest.fixture
def outputs(tmp_path, monkeypatch):
    """Point the whole result tree at tmp_path (never the real OUTPUTS)."""
    d = tmp_path / "outputs"
    d.mkdir()
    monkeypatch.setenv("QLSGYM_REPL_OUTPUTS", str(d))
    assert cfgmod.outputs_dir() == d
    assert d != cfgmod.ROOT / "OUTPUTS" / "replication"
    return d


@pytest.fixture(scope="session")
def synthetic():
    from qlsgym import load_molecule

    return load_molecule("synthetic")


def cfg_for(molecule_name: str = "synthetic", **kw) -> Config:
    base_kw = dict(molecule=molecule_name, blocks=(0,), sigmas=("+",), device="cpu")
    base_kw.update(kw)
    return Config(**base_kw)


# the stage protocol


@pytest.mark.parametrize("name", MINE)
def test_module_satisfies_stage_protocol(name):
    stage = base.load(name)
    assert isinstance(stage, base.Stage)
    assert stage.NAME == name and name in base.ORDER
    for attr in ("TITLE", "PAPER", "COST"):
        assert isinstance(getattr(stage, attr), str) and getattr(stage, attr)
    assert isinstance(stage.REQUIRES, tuple)
    assert all(r in base.ORDER for r in stage.REQUIRES)


def test_declared_requirements_match_the_stage_table():
    assert base.load("s04_accuracy").REQUIRES == ("s03_train",)
    assert base.load("s05_planner").REQUIRES == ("s03_train",)
    assert base.load("s07_figures").REQUIRES == ("s01_physics", "s04_accuracy", "s05_planner")
    # every requirement runs before the stage that needs it
    for name in MINE:
        stage = base.load(name)
        for r in stage.REQUIRES:
            assert base.ORDER.index(r) < base.ORDER.index(name)


@pytest.mark.parametrize("name", MINE)
def test_plan_is_non_empty_and_survives_missing_inputs(name, outputs):
    """plan must describe the work with nothing on disk, not crash (R6)."""
    stage = base.load(name)
    assert not any(cfgmod.result_path(r).exists() for r in stage.REQUIRES)
    lines = stage.plan(cfg_for())
    assert isinstance(lines, list) and lines
    assert all(isinstance(line, str) and line.strip() for line in lines)
    assert not list(outputs.iterdir()), "plan must touch nothing"


def test_plan_of_s07_names_the_missing_inputs(outputs):
    lines = "\n".join(base.load("s07_figures").plan(cfg_for()))
    for stage in ("s01_physics", "s04_accuracy", "s05_planner"):
        assert stage in lines
    assert "MISSING" in lines


def test_plan_of_s05_describes_the_matching_rule(outputs):
    lines = "\n".join(base.load("s05_planner").plan(cfg_for()))
    assert "matched" in lines.lower()
    assert "exact" in lines and "trivial" in lines


# a missing input names the stage to run first


@pytest.mark.parametrize("name,needs", [("s04_accuracy", "s03_train"),
                                        ("s05_planner", "s03_train"),
                                        ("s07_figures", "s01_physics")])
def test_run_raises_missing_result_naming_the_stage(name, needs, outputs):
    stage = base.load(name)
    with pytest.raises(MissingResult) as exc:
        stage.run(cfg_for())
    msg = str(exc.value)
    assert needs in msg and "run" in msg
    assert not list(outputs.rglob("*.json")), "a failed run must write nothing"


def test_check_requirements_reports_the_gap(outputs):
    cfg = cfg_for()
    for name in MINE:
        stage = base.load(name)
        assert base.check_requirements(stage, cfg) == list(stage.REQUIRES)


# stage 4 pieces that need no checkpoint


def test_resolve_checkpoints_reads_the_s03_payload(tmp_path):
    from replication.stages import s04_accuracy as s04

    ck = tmp_path / "best_onres.pt"
    ck.write_bytes(b"not a real checkpoint")
    got = s04.resolve_checkpoints(cfg_for(), {"checkpoints": {"0,+": str(ck)}})
    assert got == {(0, "+"): str(ck)}


@pytest.mark.parametrize("s03", [{}, {"checkpoints": {"0,+": "/nowhere/best.pt"}}])
def test_resolve_checkpoints_names_the_stage_to_run(s03):
    """No checkpoint is an error naming stage 3, never a guess at a path."""
    from replication.stages import s04_accuracy as s04

    with pytest.raises(FileNotFoundError) as exc:
        s04.resolve_checkpoints(cfg_for(), s03)
    assert "s03_train" in str(exc.value) and "block 0 sigma+" in str(exc.value)


def test_strata_from_summary_rekeys_and_keeps_the_ratio(synthetic):
    from qlsgym.surrogate.metrics import stratified_summary
    from replication.stages.s04_accuracy import strata_from_summary

    rng = np.random.default_rng(0)
    ta = rng.uniform(1e-4, 1e-2, 40)
    static = rng.uniform(1e-4, 1e-2, 40)
    on = np.zeros(40, dtype=bool)
    on[:12] = True
    flat = stratified_summary(ta, static, on, prefix="uniform_")
    out = strata_from_summary(flat, "uniform")
    assert set(out) == {"all", "on", "off"}
    assert out["on"]["n"] == 12 and out["off"]["n"] == 28 and out["all"]["n"] == 40
    # the ratio is the median of the per-frequency ratio, not a ratio of medians
    assert out["all"]["ratio_median"] == pytest.approx(
        float(np.median(ta / static)), rel=1e-5)
    assert out["all"]["trivial_median"] == pytest.approx(float(np.median(static)), rel=1e-5)


def test_showcase_trajectory_against_a_stub_model(synthetic):
    """Exercises the Fig. 3a plumbing (columns, embedding, Eqs. 28-30, MRE) without building an
    FNO: the stub returns the right shape and nothing else.
    """
    import torch

    from replication.stages.s04_accuracy import showcase_trajectory

    mol = synthetic
    m = mol.blocks[0].n_states

    class _Stub(torch.nn.Module):
        def forward(self, x):                      # (B, C, P) -> (B, 2M, P)
            b, _, p = x.shape
            out = torch.zeros(b, 2 * m, p, dtype=torch.float32, device=x.device)
            out[:, :m, :] = 1.0 / m
            return out

    omega = 0.5 * (mol.window.omega_min + mol.window.omega_max)
    tr = showcase_trajectory(_Stub(), mol, 0, "+", omega, n_init=4, seed=3, device="cpu")
    n_tau = mol.window.n_tau
    assert tr["n_channels"] == 2 * m
    assert len(tr["tau_ms"]) == n_tau
    assert len(tr["infidelity"]) == n_tau and len(tr["mre"]) == n_tau
    assert len(tr["trivial_infidelity"]) == n_tau
    assert len(tr["p_true"]) == len(tr["p_pred"]) == tr["n_drawn"] == len(tr["channel"])
    assert 0.0 <= tr["time_avg_infidelity"] <= 1.0
    assert all(np.isfinite(tr["infidelity"]))
    assert set(tr["nu"]) <= {0, 1}


def test_pick_showcase_frequency_honours_the_override(synthetic):
    from replication.stages.s04_accuracy import pick_showcase_frequency

    raw = {"on": {"omegas": np.array([1.0, 2.0]), "swing": np.array([0.1, 0.9])}}
    w, why = pick_showcase_frequency(raw, synthetic, 0, "+", cfg_for())
    assert w == 2.0 and "swing" in why
    w2, why2 = pick_showcase_frequency(
        raw, synthetic, 0, "+", cfg_for(overrides={"fig3a_omega_khz": 100.0}))
    assert w2 == pytest.approx(2.0 * np.pi * 100.0) and "pinned" in why2


# stage 5 pieces: statistics, matching, and a tiny end-to-end on synthetic


def test_stats_uses_the_sample_sd_the_target_quotes():
    from replication.stages.s05_planner import stats

    v = [0.84, 0.85, 0.83, 0.86, 0.85]
    s = stats(v)
    assert s["n"] == 5 and s["per_seed"] == v
    assert s["mean"] == pytest.approx(float(np.mean(v)))
    assert s["sd"] == pytest.approx(float(np.std(v, ddof=1)))
    assert s["sem"] == pytest.approx(s["sd"] / np.sqrt(5))


def test_prior_replication_is_a_quotation_with_its_source():
    from replication.stages.s05_planner import ARMS, PRIOR_REPLICATION

    assert "replication.tex" in PRIOR_REPLICATION["source"]
    assert set(PRIOR_REPLICATION["arms"]) == set(ARMS)
    assert PRIOR_REPLICATION["arms"]["exact"]["validated"] == 0.8452
    assert PRIOR_REPLICATION["arms"]["trivial"]["validated"] == 0.8134
    assert PRIOR_REPLICATION["arms"]["fno"]["validated"] == 0.6972
    assert "ddof=1" in PRIOR_REPLICATION["uncertainty"]


def test_seeds_are_identical_across_arms():
    cfg = cfg_for(seed=7)
    from replication.stages.s05_planner import seeds_of

    assert seeds_of(cfg) == [7, 8, 9, 10, 11]
    assert seeds_of(cfg_for(seed=7, overrides={"n_seeds": 2})) == [7, 8]


def test_substituted_fraction_reads_both_engine_call_conventions():
    from replication.stages.s05_planner import _substituted_fraction

    assert _substituted_fraction({"fno": 1, "exact_primitive": 3}) == pytest.approx(0.25)
    assert _substituted_fraction(
        {"substituted": 2, "exact_untrained": 2}) == pytest.approx(0.5)
    assert _substituted_fraction(None) is None
    assert _substituted_fraction({"fno": 0, "exact_primitive": 0}) is None


@pytest.fixture(scope="module")
def tiny_planner(request):
    """A tiny library and the two engines that need no checkpoint."""
    from qlsgym import load_molecule
    from qlsgym.physics.engines import ExactEngine

    from replication.stages.s05_planner import build_library, make_engine, score_config

    mol = load_molecule("synthetic")
    cfg = Config(molecule="synthetic", blocks=(0, 1), sigmas=("+",), device="cpu",
                 n_rollouts=3, max_pulses=3, seed=11,
                 overrides={"n_seeds": 2, "d_omega_khz": 70.0})
    lib, ti = build_library(cfg, mol)
    exact = ExactEngine(mol, tau_indices=ti)
    return mol, cfg, lib, ti, exact, score_config(cfg, mol), make_engine


def test_library_and_engines_share_the_tau_rows(tiny_planner):
    mol, cfg, lib, ti, exact, sc, make_engine = tiny_planner
    assert lib.n_actions > 0 and lib.n_primitives == len(mol.primitives)
    needed = set(int(t) for t in lib.tau_indices)
    needed |= {int(lib.primitive_tau_index(k)) for k in range(lib.n_primitives)}
    assert needed <= set(int(t) for t in ti)
    triv = make_engine("trivial", cfg, mol, ti, exact)
    assert type(triv).__name__ == "TrivialEngine"
    assert np.array_equal(triv.tau_indices, ti)
    assert make_engine("exact", cfg, mol, ti, exact) is exact
    with pytest.raises(ValueError):
        make_engine("bogus", cfg, mol, ti, exact)


def test_trivial_arm_substitutes_exactly_the_configured_footprint(tiny_planner):
    mol, cfg, lib, ti, exact, sc, make_engine = tiny_planner
    triv = make_engine("trivial", cfg, mol, ti, exact)
    assert triv.sub_blocks == {0, 1}
    assert triv.sub_sigmas == ("+",)


def test_tiny_end_to_end_planner_on_synthetic(tiny_planner):
    """Two arms, two seeds, three rollouts: the whole stage-5 loop in miniature."""
    from qlsgym.policies.baselines import ScorePlannerPolicy
    from qlsgym.policies.rollout import rollout

    from replication.stages.s05_planner import (contrast, engine_calls, seeds_of,
                                                stats, _delta, _substituted_fraction)

    mol, cfg, lib, ti, exact, sc, make_engine = tiny_planner
    seeds = seeds_of(cfg)
    per_arm = {}
    for arm in ("exact", "trivial"):
        engine = make_engine(arm, cfg, mol, ti, exact)
        before = engine_calls(engine)
        vals = []
        for seed in seeds:
            pol = ScorePlannerPolicy(engine, lib, sc, tau_mode="library")
            r = rollout(exact, pol, n_rollouts=cfg.n_rollouts, seed=seed,
                        max_pulses=cfg.max_pulses, p_target=mol.task.p_target,
                        library=lib)
            assert r.n_rollouts == cfg.n_rollouts and r.max_pulses == cfg.max_pulses
            assert 0.0 <= r.success_fraction <= 1.0
            assert pol.calls > 0
            vals.append(r.success_fraction)
        per_arm[arm] = stats(vals)
        calls = _delta(before, engine_calls(engine))
        if arm == "trivial":
            assert calls["substituted"] > 0
            frac = _substituted_fraction(calls)
            assert 0.0 < frac < 1.0, "some evaluations must still fall back to exact"
        else:
            assert calls is None
    assert set(per_arm) == {"exact", "trivial"}
    assert all(s["n"] == len(seeds) for s in per_arm.values())
    c = contrast(per_arm["exact"]["per_seed"], per_arm["trivial"]["per_seed"])
    assert "diff" in c


# stage 7: figures from hand-written minimal payloads


def _curve(n, v):
    return [float(v)] * n


def minimal_s04(n_tau: int = 8, n_freq: int = 6) -> dict:
    """The smallest stage-4 payload Fig. 3 can be drawn from."""
    khz = list(np.linspace(5050.0, 5300.0, n_freq))
    sets = {}
    for name, base_v in (("uniform", 1e-4), ("on", 2e-3), ("off", 5e-6)):
        sets[name] = {
            "b": {"bands": {q: _curve(n_tau, base_v * (1 + i))
                            for i, q in enumerate(("5", "25", "50", "75", "95"))},
                  "trivial_median": [1e-18] + _curve(n_tau - 1, 3e-4),
                  "n_freq": n_freq},
            "c": {"omega_khz": khz,
                  "time_avg": list(np.linspace(base_v, 3 * base_v, n_freq)),
                  "trivial_time_avg": _curve(n_freq, 3e-4),
                  "on_resonance": [name == "on"] * n_freq},
        }
    strata = {name: {"all": {"n": n_freq, "median": 1e-4, "p95": 2e-4, "max": 3e-4,
                             "trivial_median": 3e-4, "ratio_median": 1.4,
                             "ratio_p95": 2.0, "beats_trivial": 0.25}}
              for name in sets}
    tau = list(np.linspace(0.0, 2.0, n_tau))
    return {
        "protocol": {"n_test_freq": n_freq, "n_on": n_freq, "n_off": n_freq,
                     "n_test_init": 8, "n_linewidths": 1.0, "freq_seed": 1,
                     "init_seed": 2, "device": "cpu",
                     "checkpoints_from": "s03_train.checkpoints",
                     "band_percentiles": [5, 25, 50, 75, 95], "active_threshold": 1e-4,
                     "equations": "Eqs. 27-30",
                     "reporting": "ratio to the trivial predictor, stratified",
                     "pooling": "pooled over blocks, not the full-space quantity",
                     "mre": "undefined where the reference population vanishes"},
        "pairs": [{"block": 0, "sigma": "+", "n_states": 4,
                   "checkpoint": "/nowhere/best_onres.pt",
                   "active_fraction": 0.02, "n_linewidths": 1.0,
                   "strata": strata, "swing_median": {k: 0.1 for k in sets},
                   "fig3b": {"tau_ms": tau, **{k: v["b"] for k, v in sets.items()}},
                   "fig3c": {k: v["c"] for k, v in sets.items()}}],
        "pooled": {"n_pairs": 1, "strata": strata, "active_fraction_median": 0.02,
                   "fig3b": {"tau_ms": tau, **{k: v["b"] for k, v in sets.items()}},
                   "fig3c": {k: v["c"] for k, v in sets.items()}},
        "showcase": {"block": 0, "sigma": "+", "on_resonance_ratio_median": 1.4},
        "fig3a": {"block": 0, "sigma": "+", "omega_khz": 5159.6, "n_init": 8,
                  "n_channels": 8, "n_active": 4, "n_drawn": 4,
                  "channel": [0, 1, 4, 5], "nu": [0, 0, 1, 1], "tau_ms": tau,
                  "p_true": [_curve(n_tau, 0.3), _curve(n_tau, 0.2),
                             _curve(n_tau, 0.1), _curve(n_tau, 0.05)],
                  "p_pred": [_curve(n_tau, 0.31), _curve(n_tau, 0.19),
                             _curve(n_tau, 0.11), _curve(n_tau, 0.04)],
                  "infidelity": _curve(n_tau, 2e-3),
                  "trivial_infidelity": [1e-18] + _curve(n_tau - 1, 4e-3),
                  "mre": _curve(n_tau, 0.09),
                  "time_avg_infidelity": 2e-3, "trivial_time_avg_infidelity": 4e-3,
                  "max_infidelity": 3e-3, "time_avg_mre": 0.09, "max_mre": 0.15,
                  "active_threshold": 1e-4, "selected_by": "test fixture",
                  "on_resonance": True},
    }


def minimal_s05(n_seeds: int = 3) -> dict:
    """The smallest stage-5 payload Figs. 5 and 6 can be drawn from."""
    def arm(mean, spread, view_mean, calls):
        per = list(np.round(mean + spread * np.linspace(-1, 1, n_seeds), 6))
        v = None if view_mean is None else {
            "n": n_seeds, "per_seed": list(np.round(
                view_mean + spread * np.linspace(-1, 1, n_seeds), 6)),
            "mean": view_mean, "sd": float(spread), "sem": float(spread)}
        return {"engine": "Stub", "validated": {
            "n": n_seeds, "per_seed": per, "mean": float(np.mean(per)),
            "sd": float(np.std(per, ddof=1)), "sem": float(np.std(per, ddof=1) / np.sqrt(n_seeds))},
            "view": v,
            "calibration_gap": None if v is None else float(view_mean - np.mean(per)),
            "mean_pulses": 20.0, "mean_pulses_per_seed": [20.0] * n_seeds,
            "mean_pulses_successful": 11.0,
            "planner_drives_scored_per_seed": [10] * n_seeds,
            "calls": calls,
            "substituted_fraction": (None if calls is None else
                                     (calls.get("fno", 0) + calls.get("substituted", 0))
                                     / sum(calls.values())),
            "substituted_blocks": None if calls is None else [0],
            "outcomes_per_seed": [{"success": 1.0}] * n_seeds, "seconds": 1.0,
            "prior_replication": {"validated": 0.8452, "sd": 0.0076, "view": None,
                                  "substituted_fraction": 0.0,
                                  "mean_pulses_successful": None}}

    arms = {"exact": arm(0.845, 0.008, None, None),
            "trivial": arm(0.813, 0.006, 0.815, {"substituted": 27, "exact_untrained": 73}),
            "fno": arm(0.697, 0.067, 0.809, {"fno": 25, "exact_primitive": 75})}
    return {
        "protocol": {"arms": ["exact", "trivial", "fno"], "n_seeds": n_seeds,
                     "seeds": list(range(n_seeds)), "n_rollouts_per_seed": 50,
                     "max_pulses": 80, "p_target": 0.98,
                     "score": {"w_tr": 0.5, "w_br": 1.5, "w1_scale": 10.0,
                               "br_mode": "best", "p_target": 0.98},
                     "library": {"kind": "uniform", "d_omega_khz": 1.0, "n_actions": 100,
                                 "n_grid": 90, "n_primitives": 10, "n_drives": 20,
                                 "tau_rows": 5},
                     "initial_belief": "thermal", "validation": "exact propagation",
                     "headline_is": "arms.<arm>.validated.mean", "view_is": "calibration",
                     "uncertainty": "sample sd over seeds",
                     "exact_fallback": "blocks [1] fall back to exact"},
        "matching": {"rule": "same planner, different engine", "matched": ["seeds"],
                     "n_matched_seeds": n_seeds,
                     "substituted_footprint": {"blocks": [0], "sigmas": ["+"]},
                     "note": "no truncation needed"},
        "arms": arms,
        "contrasts": {"exact_vs_trivial": {"diff": 0.032, "welch_t": 7.3, "welch_p": 1e-4},
                      "exact_vs_fno": {"diff": 0.148, "welch_t": 4.9, "welch_p": 0.007},
                      "trivial_vs_fno": {"diff": 0.116, "welch_t": 3.9, "welch_p": 0.017}},
        "headline": {"ours": {a: {"validated": arms[a]["validated"]["mean"],
                                  "sd": arms[a]["validated"]["sd"]} for a in arms},
                     "prior_replication": {"source": "fixture"},
                     "cost_of_approximation": {"trivial": 0.032, "fno": 0.148}},
        "paper": {"fig5a_dw1.0": {}, "p_target": 0.98, "max_pulses": 80,
                  "n_mc_rollouts": 1000,
                  "note": "the paper's Fig. 5a is a tree search, not this loop"},
        "s03_tag": "repl",
    }


def minimal_s01() -> dict:
    return {"n_states": 10, "n_blocks": 3, "blocks": [{"index": i} for i in range(3)]}


@pytest.fixture
def rendered(outputs, synthetic):
    """Write the three inputs, run s07, return (payload, figures_dir)."""
    cfg = cfg_for()
    for stage, payload in (("s01_physics", minimal_s01()),
                           ("s04_accuracy", minimal_s04()),
                           ("s05_planner", minimal_s05())):
        cfgmod.write_result(stage, payload, cfg, molecule=synthetic)
    payload = base.load("s07_figures").run(cfg)
    return payload, cfgmod.figures_dir()


def test_s07_writes_pdf_and_png_for_every_figure(rendered):
    payload, figdir = rendered
    for name in ("fig3", "fig5", "fig6"):
        for ext in ("pdf", "png"):
            p = figdir / f"{name}.{ext}"
            assert p.exists() and p.stat().st_size > 1000, f"{p} is missing or empty"
    assert sorted(Path(x).name for x in payload["files"]) == [
        f"{n}.{e}" for n in ("fig3", "fig5", "fig6") for e in ("pdf", "png")]


def test_s07_payload_is_only_the_files_it_wrote(rendered):
    payload, figdir = rendered
    assert set(payload) == {"figures_dir", "files"}
    assert payload["figures_dir"] == str(figdir)


def test_s07_refuses_a_stale_vintage(outputs, synthetic):
    """A result from a different Hamiltonian must not be plotted as current."""
    from replication.config import VintageMismatch

    cfg = cfg_for()
    for stage, payload in (("s01_physics", minimal_s01()),
                           ("s04_accuracy", minimal_s04()),
                           ("s05_planner", minimal_s05())):
        cfgmod.write_result(stage, payload, cfg, molecule=synthetic)
    path = cfgmod.result_path("s04_accuracy")
    doc = json.loads(path.read_text())
    doc["provenance"]["fingerprint"] = "deadbeef0000"
    path.write_text(json.dumps(doc))
    with pytest.raises(VintageMismatch):
        base.load("s07_figures").run(cfg)


def test_s07_renders_without_importing_torch(outputs, synthetic):
    """The claim is about run, so the fresh process renders the whole set."""
    cfg = cfg_for()
    for stage, payload in (("s01_physics", minimal_s01()),
                           ("s04_accuracy", minimal_s04()),
                           ("s05_planner", minimal_s05())):
        cfgmod.write_result(stage, payload, cfg, molecule=synthetic)
    code = (
        "import sys; sys.path[:0] = [%r, %r]\n"
        "from replication.config import Config\n"
        "from replication.stages import base\n"
        "out = base.load('s07_figures').run("
        "Config(molecule='synthetic', blocks=(0,), sigmas=('+',), device='cpu'))\n"
        "assert len(out['files']) == 6, out\n"
        "assert 'torch' not in sys.modules, sorted(k for k in sys.modules if 'torch' in k)\n"
        "print('ok')\n" % (str(ROOT), str(ROOT / "src"))
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={**os.environ, "QLSGYM_REPL_OUTPUTS": str(outputs)})
    assert out.returncode == 0, out.stderr
    assert "ok" in out.stdout


# the figure modules themselves


def test_binned_median_is_stable_and_drops_empty_bins():
    from replication.figures.fig3 import binned_median

    x = np.array([0.0, 0.0, 1.0, 1.0, 9.0])
    y = np.array([1.0, 3.0, 10.0, 20.0, 100.0])
    c, m = binned_median(x, y, n_bins=3)          # edges 0, 3, 6, 9; the middle bin is empty
    assert c.size == m.size == 2
    assert m[0] == pytest.approx(6.5) and m[-1] == pytest.approx(100.0)
    assert c[0] == pytest.approx(1.5) and c[-1] == pytest.approx(7.5)
    assert binned_median([], [], 3)[0].size == 0


def test_paper_values_are_drawn_in_the_reserved_style(outputs, paper):
    """R5: a quoted value must be tellable from a measurement at a glance."""
    from replication.figures import fig3, style

    assert style.PAPER_KW["linestyle"] == "--" and style.PAPER_KW["color"] == "0.35"
    assert style.TRIVIAL_LS != style.PAPER_KW["linestyle"]
    paths = fig3.render(minimal_s04(), paper, outputs, name="fig3_style")
    assert [p.endswith(e) for p, e in zip(paths, (".pdf", ".png"))] == [True, True]


@pytest.mark.parametrize("drop", ["none", "null", "absent"])
def test_fig6_renders_with_no_view_data(outputs, paper, drop):
    """With stage 5's view pass off there is nothing to pair; the panel must still render, empty,
    rather than raise.
    """
    from replication.figures import fig6

    s05 = minimal_s05()
    for a in s05["arms"].values():
        if drop == "null":
            a["view"] = None
        elif drop == "absent":
            a.pop("view", None)
    paths = fig6.render(s05, paper, outputs, name=f"fig6_{drop}")
    assert len(paths) == 2 and (outputs / f"fig6_{drop}.pdf").exists()


def _rendered_text(module, payload, paper, outdir, monkeypatch):
    """Every non-empty string the renderer draws, and the ones an axes explains itself with (tick
    labels, axis labels, its title, its legend entries).
    """
    import matplotlib.pyplot as plt
    from matplotlib.text import Text

    grabbed = {}
    monkeypatch.setattr(module, "save", lambda fig, name, out: grabbed.setdefault("fig", fig) or [])
    module.render(payload, paper, outdir, name="probe")
    fig = grabbed["fig"]
    fig.canvas.draw()
    allowed = set()
    for ax in fig.axes:
        allowed |= {ax.get_title(), ax.get_xlabel(), ax.get_ylabel()}
        for axis in (ax.xaxis, ax.yaxis):
            allowed |= {t.get_text() for t in axis.get_ticklabels(which="both")}
            allowed.add(axis.get_offset_text().get_text())
        legend = ax.get_legend()
        if legend is not None:
            allowed |= {t.get_text() for t in legend.get_texts()}
    drawn = {t.get_text() for t in fig.findobj(Text)}
    plt.close(fig)
    return {s for s in drawn if s.strip()}, {s for s in allowed if s.strip()}


def test_figures_draw_nothing_but_the_data(outputs, paper, monkeypatch):
    """No caption, note or caveat: every string on a figure is a tick label, an axis label, an axes
    title or a legend entry.
    """
    from replication.figures import fig3, fig5, fig6

    for module, payload in ((fig3, minimal_s04()), (fig5, minimal_s05()),
                            (fig6, minimal_s05())):
        drawn, allowed = _rendered_text(module, payload, paper, outputs, monkeypatch)
        assert drawn, f"{module.__name__} drew no text at all"
        assert drawn <= allowed, f"{module.__name__} draws {sorted(drawn - allowed)}"


# the contract with replication/paper.py


def test_paper_module_provides_every_constant_stage_7_quotes(paper):
    from replication.stages.s07_figures import PAPER_CONSTANTS

    assert set(paper) == set(PAPER_CONSTANTS)
    assert isinstance(paper["FIG5A_PAPER"], dict) and "dw1.0" in paper["FIG5A_PAPER"]


# What a quotation may pass through on its way into a payload -- indexing one
# (P.FIG5A_PAPER["dw1.0"]), nesting it in a literal -- and where it lands.
_CARRIES = (ast.Subscript, ast.Dict, ast.List, ast.Tuple, ast.Set)
_LANDS = (ast.Assign, ast.AnnAssign, ast.Return)


def _is_quotation(node, parents) -> bool:
    """True if this P.<NAME> is only carried into a payload."""
    while True:
        parent = parents.get(node)
        if parent is None:
            return False
        if isinstance(parent, _LANDS):
            return getattr(parent, "value", None) is node
        if isinstance(parent, ast.Dict):
            if node not in parent.values:          # a key, not a value
                return False
        elif isinstance(parent, ast.Subscript):
            if parent.value is not node:           # the index, not the quotation
                return False
        elif not isinstance(parent, _CARRIES):     # a call, an operator, a test...
            return False
        node = parent


@pytest.mark.parametrize("path", sorted((ROOT / "replication" / "stages").glob("s0*.py")),
                         ids=lambda p: p.stem)
def test_no_stage_computes_from_a_paper_constant(path):
    """A stage may quote paper.py into its payload and nothing else; computing from one is a defect."""
    tree = ast.parse(path.read_text())
    parents = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}
    aliases = {a.asname or a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
               for a in n.names if a.name == "paper"}
    aliases |= {a.asname or a.name.split(".")[0] for n in ast.walk(tree)
                if isinstance(n, ast.Import) for a in n.names if a.name.endswith("paper")}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id in aliases):
            assert _is_quotation(node, parents), (
                f"{path.name}:{node.lineno} computes from {node.value.id}.{node.attr}: "
                "a paper value is a quotation")
