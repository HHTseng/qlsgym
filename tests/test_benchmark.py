"""qlsgym.benchmark: the finished-episodes curve arithmetic, part merging, result stamping/refusal, and
the cheap arms end to end on the synthetic molecule (exact physics tables).
"""
import json
import os
from dataclasses import replace

import numpy as np
import pytest

from qlsgym import load_molecule
from qlsgym.benchmark import (ARMS, BenchmarkConfig, finished_curve, from_rollout, merge_results,
                              pulses_to, read_results, run_arm, unmerged_parts, write_result)
from qlsgym.benchmark.figure import (GUIDE_COLOUR, STYLE, draw, render, style_legend,
                                    table)
from qlsgym.benchmark.protocol import (build_library, exact_env, rl_run_dir, score_config,
                                       train_or_load_actor)
from qlsgym.rl.ppo import PPOConfig


def test_curve_is_the_cdf_of_successful_lengths():
    lengths = [1, 3, 3, 5, 5, 5]
    succ = [True, True, True, False, True, False]     # two failures at the budget
    c = finished_curve(lengths, succ, max_pulses=5)
    assert c.shape == (6,)
    assert np.allclose(c, [0, 1 / 6, 1 / 6, 3 / 6, 3 / 6, 4 / 6])
    assert c[-1] == pytest.approx(np.mean(succ))
    assert pulses_to(c, 0.5) == 3 and pulses_to(c, 4 / 6) == 5 and pulses_to(c, 0.85) is None
    with pytest.raises(ValueError):
        finished_curve([6], [True], max_pulses=5)
    with pytest.raises(ValueError):
        finished_curve([], [], max_pulses=5)


def _record(arm, seed, lengths, succ, max_pulses=10, engine=None, created="2000-01-01T00:00:00"):
    c = finished_curve(lengths, succ, max_pulses)
    return {"arm": arm, "fingerprint": "f", "max_pulses": max_pulses, "p_target": 0.98,
            "library": {"tag": "L"}, "config": {"seed": seed, "n_episodes": len(lengths)},
            "lengths": lengths, "successes": succ, "created": created,
            "curve": c.tolist(), "p85": pulses_to(c), "success": float(np.mean(succ)),
            "mean_pulses": float(np.mean(lengths)), "n_episodes": len(lengths), "seconds": 1.0,
            "engine": engine}


def test_merge_concatenates_parts_and_refuses_mismatches():
    a = _record("sweeping", 1, [2, 10], [True, False])
    b = _record("sweeping", 2, [4, 4], [True, True])
    m = merge_results([a, b])
    assert m["n_episodes"] == 4 and m["success"] == 0.75 and m["mean_pulses"] == 5.0
    assert np.allclose(m["curve"], finished_curve([2, 10, 4, 4], [True, False, True, True], 10))
    assert m["merged_from"] == [{"seed": 1, "n_episodes": 2}, {"seed": 2, "n_episodes": 2}]
    with pytest.raises(ValueError):
        merge_results([a, _record("sweeping", 1, [1], [True])])          # same seed twice
    with pytest.raises(ValueError):
        merge_results([a, _record("random", 3, [1], [True])])            # different arm


def test_merged_record_describes_the_merge_not_part_zero():
    e1 = {"kind": "FnoEngine", "manifest_tag": "prod", "manifest_fingerprint": "f",
          "calls": {"fno": 10, "exact_untrained": 1}, "surrogate_fraction": 10 / 11}
    e2 = {"kind": "FnoEngine", "manifest_tag": "prod", "manifest_fingerprint": "f",
          "calls": {"fno": 30, "exact_primitive": 9}, "surrogate_fraction": 30 / 39}
    a = _record("planner-fno", 1, [2, 10], [True, False], engine=e1)
    b = _record("planner-fno", 2, [4, 4], [True, True], engine=e2)
    m = merge_results([a, b])
    # the config must not still claim one part's episode count or seed
    assert m["config"]["n_episodes"] == 4 == m["n_episodes"]
    assert m["config"]["seed"] is None and m["seeds"] == [1, 2]
    assert m["created"] != a["created"]
    # engine call counts are totals, like `seconds` already was
    assert m["engine"]["calls"] == {"fno": 40, "exact_untrained": 1, "exact_primitive": 9}
    assert m["engine"]["surrogate_fraction"] == pytest.approx(40 / 50)
    assert m["engine"]["merged_from_parts"] == 2
    assert m["seconds"] == a["seconds"] + b["seconds"]
    # parts that consulted different dynamics must not be merged
    c = _record("planner-fno", 3, [4], [True], engine=dict(e1, manifest_tag="mix"))
    with pytest.raises(ValueError, match="different dynamics"):
        merge_results([a, c])


def test_write_result_refuses_non_finite_numbers(tmp_path):
    r = _record("sweeping", 1, [1, 2], [True, True])
    r["engine"] = {"kind": "FnoEngine", "surrogate_fraction": float("nan")}
    with pytest.raises(ValueError, match="non-finite"):
        write_result(r, str(tmp_path / "sweeping.json"))
    assert not os.path.exists(str(tmp_path / "sweeping.json"))
    r["engine"]["surrogate_fraction"] = None
    write_result(r, str(tmp_path / "sweeping.json"))
    with open(tmp_path / "sweeping.json") as fh:
        assert json.load(fh)["engine"]["surrogate_fraction"] is None


def test_write_read_refuses_stale_fingerprint(tmp_path):
    mol = load_molecule("synthetic")
    ok = _record("sweeping", 1, [1, 2], [True, True]); ok["fingerprint"] = mol.fingerprint()
    stale = _record("random", 1, [1, 2], [True, True])
    write_result(ok, str(tmp_path / "sweeping.json"))
    write_result(stale, str(tmp_path / "random.json"))
    write_result(ok, str(tmp_path / "sweeping.part00.json"))
    warnings = []
    got = read_results(mol, str(tmp_path), warn=warnings.append)
    assert [r["arm"] for r in got] == ["sweeping"] and len(warnings) == 1 and "random.json" in warnings[0]


def test_render_and_table(tmp_path):
    recs = [_record("sweeping", 1, [2, 3, 10], [True, True, False]), _record("rl-fno", 1, [1, 1, 2], [True] * 3)]
    paths = render(recs, str(tmp_path / "fig"))
    assert all(os.path.exists(p) for p in paths)
    txt = table(recs)
    assert "sweeping" in txt and "rl-fno" in txt
    # the numbers, then the figure key that replaces the (forbidden) on-image legend
    assert txt.splitlines()[:3] == [l for l in txt.splitlines()[:3]] and len(txt.splitlines()) > 5
    assert "colour" in txt and "style" in txt


def test_figure_names_the_arms_and_carries_nothing_else():
    """The legend names the curves -- that is content.  Titles, captions and
    annotations are not, and must stay off the image."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = [_record(arm, 1, [2, 3, 10], [True, True, False]) for arm in ARMS]
    fig, ax = plt.subplots()
    draw(ax, recs, budget=5)
    try:
        leg = ax.get_legend()
        assert leg is not None, "the arms must be named on the figure"
        assert {t.get_text() for t in leg.get_texts()} == set(ARMS)
        assert ax.get_title() == "" and ax.get_title("left") == "" and ax.get_title("right") == ""
        assert [t for t in ax.texts if t.get_text()] == [], "no annotation may be drawn on the axes"
        assert ax.get_xlabel() and ax.get_ylabel()
    finally:
        plt.close(fig)
    # legend=False gives the bare curves
    fig, ax = plt.subplots()
    draw(ax, recs, budget=5, legend=False)
    try:
        assert ax.get_legend() is None
    finally:
        plt.close(fig)
    # three unambiguous classes, so the plot still reads in greyscale
    assert {STYLE[a][1] for a in ("sweeping", "random")} == {":"}
    assert {STYLE[a][1] for a in ("planner-exact", "rl-exact")} == {"--"}
    assert {STYLE[a][1] for a in ("planner-fno", "rl-fno")} == {"-"}
    # the exact/fno pair of a family shares a colour and differs in style
    for a, b in (("planner-exact", "planner-fno"), ("rl-exact", "rl-fno")):
        assert STYLE[a][0] == STYLE[b][0] and STYLE[a][1] != STYLE[b][1]
    # no guide may be mistaken for an arm
    assert (GUIDE_COLOUR, "-") not in {(c, ls) for c, ls, _ in STYLE.values()}
    txt = style_legend(recs)
    for arm in ARMS:
        assert arm in txt
    assert "85% guide" in txt


def test_unmerged_parts_are_reported_not_silently_dropped(tmp_path):
    mol = load_molecule("synthetic")
    ok = _record("sweeping", 1, [1, 2], [True, True]); ok["fingerprint"] = mol.fingerprint()
    write_result(ok, str(tmp_path / "sweeping.json"))
    # planner-exact exists only as a part file: it would vanish from the figure
    write_result(ok, str(tmp_path / "planner-exact.part90.json"))
    assert list(unmerged_parts(str(tmp_path))) == ["planner-exact"]
    warnings = []
    got = read_results(mol, str(tmp_path), warn=warnings.append)
    assert [r["arm"] for r in got] == ["sweeping"]
    assert any("planner-exact" in w and "MISSING" in w for w in warnings)
    # once merged, no warning
    write_result(ok, str(tmp_path / "planner-exact.json"))
    assert unmerged_parts(str(tmp_path)) == {}


def test_p_target_override_reaches_the_planner_score():
    mol = load_molecule("synthetic")
    assert score_config(BenchmarkConfig(molecule="synthetic"), mol).p_target == mol.task.p_target
    sc = score_config(BenchmarkConfig(molecule="synthetic", p_target=0.999), mol)
    assert sc.p_target == 0.999      # the env would stop at 0.999; so must the planner


def test_config_validation():
    with pytest.raises(ValueError):
        BenchmarkConfig(library="grid")
    with pytest.raises(ValueError):
        BenchmarkConfig(n_episodes=0)
    with pytest.raises(ValueError):
        run_arm("sweep", BenchmarkConfig(molecule="synthetic"))


@pytest.fixture(scope="module")
def synthetic_cfg(tmp_path_factory):
    work = tmp_path_factory.mktemp("work")
    os.environ["QLSGYM_WORK"] = str(work)
    return BenchmarkConfig(molecule="synthetic", n_episodes=12, seed=5, n_pool=3, delta_s=0.0,
                           ppo=PPOConfig(n_envs=8, n_steps=4, total_steps=64, minibatches=2, epochs=1,
                                         eval_every=1, eval_rollouts=4, hidden=16))


@pytest.mark.parametrize("arm", ["sweeping", "random", "planner-exact", "rl-exact"])
def test_cheap_arms_end_to_end(synthetic_cfg, arm, tmp_path):
    lines = []
    r = run_arm(arm, synthetic_cfg, log=lines.append)
    mol = load_molecule("synthetic")
    assert r["arm"] == arm and r["fingerprint"] == mol.fingerprint()
    assert r["n_episodes"] == 12 and len(r["curve"]) == mol.task.max_pulses + 1
    assert r["curve"][-1] == pytest.approx(r["success"])
    assert r["library"]["n_actions"] > 0 and r["config"]["seed"] == 5
    if arm == "rl-exact":
        assert r["rl"]["tables_builder"] == "physics" and os.path.exists(r["rl"]["run_dir"] + "/actor_best.pt")
        # a second call loads the saved actor instead of training
        r2 = run_arm(arm, synthetic_cfg)
        assert r2["rl"]["loaded_from"].endswith("actor_best.pt")
        assert r2["lengths"] == r["lengths"]
    if arm == "planner-exact":
        assert r["engine"]["kind"] == "ExactEngine" and "n_pool=3" in r["policy"]
    path = write_result(r, str(tmp_path / f"{arm}.json"))
    with open(path) as fh:
        assert json.load(fh)["arm"] == arm


def test_rl_run_dir_separates_materially_different_configs(synthetic_cfg):
    mol = load_molecule("synthetic")
    lib = build_library(synthetic_cfg, mol)
    base = rl_run_dir(synthetic_cfg, lib, "rl-exact", molecule=mol)
    for other in (replace(synthetic_cfg, max_pulses=7),
                  replace(synthetic_cfg, rho=0.5),
                  replace(synthetic_cfg, penalty_mode="proportional"),
                  replace(synthetic_cfg, p_target=0.999),
                  replace(synthetic_cfg, ppo=replace(synthetic_cfg.ppo, hidden=32)),
                  replace(synthetic_cfg, ppo=replace(synthetic_cfg.ppo, obs="p")),
                  replace(synthetic_cfg, ppo=replace(synthetic_cfg.ppo, total_steps=128))):
        assert rl_run_dir(other, lib, "rl-exact", molecule=mol) != base, other
    # the surrogate manifest, not just its tag, is part of the key
    class _Eng:
        pass
    e1, e2 = _Eng(), _Eng()
    e1.manifest = type("M", (), {"to_json": lambda self: {"entries": {"0,+": {"path": "/a"}}}})()
    e2.manifest = type("M", (), {"to_json": lambda self: {"entries": {"0,+": {"path": "/b"}}}})()
    d1 = rl_run_dir(synthetic_cfg, lib, "rl-fno", molecule=mol, engine=e1)
    d2 = rl_run_dir(synthetic_cfg, lib, "rl-fno", molecule=mol, engine=e2)
    assert d1 != d2


def test_saved_actor_is_refused_when_it_does_not_match_the_request(synthetic_cfg, tmp_path):
    """load_policy rebuilds the net from the SAVED ppo config, so a mismatch used to
    evaluate one network while the result JSON reported another."""
    mol = load_molecule("synthetic")
    lib = build_library(synthetic_cfg, mol)
    env = exact_env(synthetic_cfg, mol, lib, batch=synthetic_cfg.n_episodes)
    env_train = env.clone(synthetic_cfg.ppo.n_envs)
    run_dir = str(tmp_path / "run")
    _, meta = train_or_load_actor(synthetic_cfg, mol, lib, env_train, env, run_dir, "physics")
    assert meta["ppo_config"]["hidden"] == synthetic_cfg.ppo.hidden
    # same directory, different network: must refuse rather than silently reuse
    other = replace(synthetic_cfg, ppo=replace(synthetic_cfg.ppo, hidden=32))
    with pytest.raises(RuntimeError, match="hidden"):
        train_or_load_actor(other, mol, lib, env_train, env, run_dir, "physics")
    # unchanged config still loads
    pol, meta2 = train_or_load_actor(synthetic_cfg, mol, lib, env_train, env, run_dir, "physics")
    assert meta2["loaded_from"].endswith("actor_best.pt")


def test_benchmark_sbatch_passes_the_horizon_through():
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sb = open(os.path.join(here, "scripts", "slurm", "benchmark.sbatch")).read()
    assert "MAXPULSES" in sb
    assert "--max-pulses" in sb and '"${MAXP_ARG[@]}"' in sb


def test_fno_arm_needs_a_manifest(synthetic_cfg):
    with pytest.raises(FileNotFoundError):
        run_arm("planner-fno", synthetic_cfg)


def test_render_marks_the_task_budget(tmp_path):
    recs = [_record("sweeping", 1, [5, 30, 30], [True, True, True], max_pulses=30)]
    paths = render(recs, str(tmp_path / "fig"), budget=10)
    assert all(os.path.exists(p) for p in paths)
    # a budget at or beyond the horizon draws nothing and must not raise
    assert render(recs, str(tmp_path / "fig2"), budget=30)
