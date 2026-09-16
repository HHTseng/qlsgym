"""The qMDP-DQN baseline port (replication.rl) and its stage (s06_rl)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
for p in (str(ROOT), str(ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from qlsgym.env.actions import ActionLibrary                      # noqa: E402
from qlsgym.env.cache import build_action_tables                  # noqa: E402
from qlsgym.env.env import EnvConfig, PurificationEnv             # noqa: E402
from qlsgym.policies.baselines import Policy                      # noqa: E402
from qlsgym.policies.rollout import rollout                       # noqa: E402

from replication import rl                                        # noqa: E402
from replication.config import Config, MissingResult, load_molecule, result_path  # noqa: E402
from replication.stages import base                               # noqa: E402
from replication.stages import s06_rl                             # noqa: E402

MICRO_AGENT = {"hidden": 8, "batch_size": 8, "memory": 64, "snapshot_every": 2,
               "snapshot_rollouts": 8, "eps_decay_steps": 40}
MICRO = {**MICRO_AGENT, "library": "physics_subset", "moving_window": 3, "verbose": False}


def micro_cfg(**kw) -> Config:
    over = dict(MICRO)
    over.update(kw.pop("overrides", {}))
    fields = dict(molecule="synthetic", rl_episodes=4, n_rollouts=12, max_pulses=6, seed=1)
    fields.update(kw)
    return Config(overrides=over, **fields)


@pytest.fixture(scope="module")
def world():
    """Synthetic molecule, its resonant action library and exact tables."""
    cfg = micro_cfg()
    mol = load_molecule(cfg)
    lib = ActionLibrary.physics_subset(mol, clip_tau=True)
    tables = build_action_tables(mol, lib, write=False)
    return cfg, mol, lib, tables


@pytest.fixture
def env(world):
    _, mol, lib, tables = world
    return PurificationEnv(mol, lib, tables, EnvConfig(max_pulses=6, rho=2.0), batch=1)


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """An empty output tree and an empty $QLSGYM_WORK."""
    monkeypatch.setenv("QLSGYM_REPL_OUTPUTS", str(tmp_path / "outputs"))
    monkeypatch.setenv("QLSGYM_WORK", str(tmp_path / "work"))
    return tmp_path


def write_s01(molecule) -> None:
    p = result_path("s01_physics")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"stage": "s01_physics",
                             "provenance": {"fingerprint": molecule.fingerprint()},
                             "result": {"ok": True}}))


# the stage contract


def test_stage_satisfies_the_protocol():
    assert isinstance(s06_rl, base.Stage)
    assert s06_rl.NAME == "s06_rl" and s06_rl.NAME in base.ORDER
    assert s06_rl.PAPER == "App. C.1"
    # the RL baseline runs on exact tables: it needs the physics check, not the surrogate
    assert s06_rl.REQUIRES == ("s01_physics",)
    assert base.load("s06_rl") is s06_rl


def test_plan_works_with_nothing_on_disk(isolated, world):
    cfg, mol, lib, _ = world
    lines = s06_rl.plan(cfg)
    assert lines and all(isinstance(x, str) for x in lines)
    joined = "\n".join(lines)
    assert f"{lib.n_actions} actions" in joined          # library size
    assert "ABSENT" in joined                            # the cache it would need
    assert "floats/action" in joined
    assert str(len(rl.DEVIATIONS)) in joined


def test_plan_sees_a_built_cache(isolated, world):
    cfg, mol, lib, _ = world
    build_action_tables(mol, lib, write=True)
    assert "PRESENT" in "\n".join(s06_rl.plan(cfg))


def test_plan_warns_when_no_snapshot_would_be_taken(isolated, world):
    cfg, _, _, _ = world
    lines = s06_rl.plan(micro_cfg(rl_episodes=1))
    assert any("no snapshot" in ln for ln in lines)


def test_run_without_its_requirement_says_which_stage(isolated, world):
    cfg, _, _, _ = world
    with pytest.raises(MissingResult) as e:
        s06_rl.run(cfg)
    assert "s01_physics" in str(e.value)


def test_run_end_to_end(isolated, world):
    cfg, mol, lib, _ = world
    write_s01(mol)
    payload = s06_rl.run(cfg)
    json.dumps(payload)                                   # the stage payload must be JSON
    assert payload["library"]["n_actions"] == lib.n_actions
    assert payload["training"]["episodes"] == cfg.rl_episodes
    assert payload["training"]["curve"][-1][0] == cfg.rl_episodes
    assert payload["training"]["snapshots"], "the snapshot protocol produced nothing"
    for snap in payload["training"]["snapshots"]:
        assert Path(snap["path"]).exists()
        assert str(isolated / "work") in snap["path"]     # bulk stays out of git (R3)
    assert payload["selected_snapshot"]["episode"] in [s["episode"] for s in
                                                       payload["training"]["snapshots"]]
    for who in ("trained_policy", "anchor"):
        assert 0.0 <= payload[who]["converged_fraction"] <= 1.0
        assert payload[who]["mean_pulses"] <= cfg.max_pulses
        assert payload[who]["n_rollouts"] == cfg.n_rollouts
        assert payload[who]["eval_seed"] == rl.EVAL_SEED
    assert payload["deviations"] == list(rl.DEVIATIONS)
    assert payload["reference"]["paper"]["converged_fraction"] == 0.428
    assert payload["reference"]["fnorepl"]["converged_fraction"] == 0.226
    assert payload["seconds"]["wall"] > 0


# the agent is a qlsgym policy


def test_qnet_policy_satisfies_the_policy_protocol(world):
    _, mol, lib, _ = world
    cfg = rl.AgentConfig(hidden=8)
    pol = rl.QNetPolicy(rl.make_qnet(mol.n_states, lib.n_actions, cfg), lib.n_actions)
    assert isinstance(pol, Policy)
    rng = np.random.default_rng(0)
    belief = np.full(mol.n_states, 1.0 / mol.n_states)
    a = pol.act(belief, 0, rng)
    assert isinstance(a, int) and 0 <= a < lib.n_actions
    batch = pol.act_batch(np.tile(belief, (5, 1)), 0, rng)
    assert batch.shape == (5,) and batch.dtype == np.int64
    assert (batch == a).all()                              # greedy and stateless


def test_qnet_policy_explores_when_epsilon_is_one(world):
    _, mol, lib, _ = world
    net = rl.make_qnet(mol.n_states, lib.n_actions, rl.AgentConfig(hidden=8))
    pol = rl.QNetPolicy(net, lib.n_actions, eps=1.0)
    belief = np.full(mol.n_states, 1.0 / mol.n_states)
    picks = pol.act_batch(np.tile(belief, (64, 1)), 0, np.random.default_rng(0))
    assert len(set(picks.tolist())) > 1


def test_rollout_scores_the_agent_like_any_policy(env, world):
    _, mol, lib, _ = world
    pol = rl.QNetPolicy(rl.make_qnet(mol.n_states, lib.n_actions, rl.AgentConfig(hidden=8)),
                        lib.n_actions)
    res = rollout(env, pol, n_rollouts=10, seed=rl.EVAL_SEED)
    assert res.policy == "QNetPolicy" and res.n_rollouts == 10
    assert 0.0 <= res.success_fraction <= 1.0


def test_fixed_action_anchor(env, world):
    _, _, lib, _ = world
    pick = rl.best_fixed_action(env)
    assert 0 <= pick["action"] < lib.n_actions
    assert 0.0 <= pick["pi1"] <= 1.0 and 0.0 <= pick["purity_nu1"] <= 1.0
    pol = rl.FixedActionPolicy(pick["action"], lib.n_actions)
    assert isinstance(pol, Policy)
    assert (pol.act_batch(np.zeros((4, 3)), 0, None) == pick["action"]).all()
    res = rollout(env, pol, n_rollouts=10, seed=rl.EVAL_SEED)
    assert res.policy == "FixedActionPolicy"
    with pytest.raises(IndexError):
        rl.FixedActionPolicy(lib.n_actions, lib.n_actions)


# micro training


def _micro_train(world, snapshot_dir=None):
    _, mol, lib, tables = world
    agent = rl.AgentConfig(seed=3, **MICRO_AGENT)
    return rl.train(mol, lib, tables, agent, EnvConfig(max_pulses=6, rho=2.0),
                    episodes=4, snapshot_dir=snapshot_dir, moving_window=3, verbose=False)


def test_micro_training_runs_end_to_end(world, tmp_path):
    out = _micro_train(world, str(tmp_path / "snaps"))
    assert len(out["curve"]) == 4
    assert out["env_steps"] > 0 and out["seconds_train"] > 0
    assert [s["episode"] for s in out["snapshots"]] == [2, 4]
    for snap in out["snapshots"]:
        assert Path(snap["path"]).exists()
        assert 0.0 <= snap["frac_untrained_action_rows"] <= 1.0
    assert (tmp_path / "snaps" / "run.json").exists()
    best = rl.select_best(out["snapshots"])
    assert best in out["snapshots"]
    _, mol, lib, _ = world
    pol = rl.load_policy(best["path"], mol.n_states, lib.n_actions,
                         rl.AgentConfig(hidden=8))
    assert isinstance(pol, Policy)


def test_micro_training_is_reproducible(world):
    keys = ("episode", "length", "reward", "solved", "moving_success", "epsilon", "env_steps")

    def deterministic(out):
        return [{k: c[k] for k in keys} for c in out["curve"]]

    a = _micro_train(world)
    b = _micro_train(world)
    assert deterministic(a) == deterministic(b)
    assert [s["success_fraction"] for s in a["snapshots"]] == \
           [s["success_fraction"] for s in b["snapshots"]]
    assert a["run_tag"] == b["run_tag"]


def test_snapshot_dir_refuses_another_run(world, tmp_path):
    d = str(tmp_path / "snaps")
    rl.stamp_snapshot_dir(d, "aaaaaaaaaaaa", {})
    rl.stamp_snapshot_dir(d, "aaaaaaaaaaaa", {})            # same run: fine
    with pytest.raises(RuntimeError):
        rl.stamp_snapshot_dir(d, "bbbbbbbbbbbb", {})


def test_select_best_needs_snapshots():
    with pytest.raises(ValueError):
        rl.select_best([])
    snaps = [{"episode": 1, "success_fraction": 0.2, "mean_pulses": 9.0},
             {"episode": 2, "success_fraction": 0.4, "mean_pulses": 7.0},
             {"episode": 3, "success_fraction": 0.4, "mean_pulses": 8.0}]
    assert rl.select_best(snaps)["episode"] == 2
    assert rl.select_best(snaps, "mean_length")["episode"] == 2
    with pytest.raises(ValueError):
        rl.select_best(snaps, "coin flip")


# units: replay buffer and epsilon schedule


def test_replay_buffer_stores_both_branches_and_wraps():
    torch = pytest.importorskip("torch")
    n, cap = 5, 4
    buf = rl.Replay(cap, n, torch.device("cpu"), torch)
    assert len(buf) == 0

    def push(k, a):
        one = torch.full((1, n), float(k))
        buf.push(one, torch.tensor([a]), one + 1, one + 2, torch.tensor([0.25]),
                 torch.tensor([0.75]), torch.tensor([-1.0]), torch.tensor([-3.0]),
                 torch.tensor([False]), torch.tensor([True]))

    for k in range(3):
        push(k, k)
    assert len(buf) == 3 and buf.pos == 3
    s, a, s0, s1, pi0, pi1, r0, r1, d0, d1 = buf.sample(7)
    assert s.shape == (7, n) and a.shape == (7,)
    assert torch.allclose(s0 - s, torch.ones_like(s)) and torch.allclose(s1 - s, 2 * torch.ones_like(s))
    assert torch.allclose(pi0 + pi1, torch.ones_like(pi0))
    assert (r0 == -1.0).all() and (r1 == -3.0).all()
    assert (~d0).all() and d1.all()
    for k in range(3, 9):
        push(k, k % cap)
    assert len(buf) == cap and buf.pos == 9 % cap
    assert float(buf.s[buf.pos - 1][0]) == 8.0              # newest entry survived the wrap


def test_epsilon_schedule_reaches_its_end_value():
    cfg = rl.AgentConfig(eps_end=0.025, eps_decay_steps=1000)
    assert rl.epsilon(0, cfg) == pytest.approx(1.0)
    assert rl.epsilon(500, cfg) == pytest.approx(0.5125)
    assert rl.epsilon(1000, cfg) == pytest.approx(cfg.eps_end)
    assert rl.epsilon(10_000, cfg) == pytest.approx(cfg.eps_end)    # clamped, never below
    steps = [rl.epsilon(s, cfg) for s in range(0, 1200, 50)]
    assert all(b <= a + 1e-12 for a, b in zip(steps, steps[1:]))    # monotone
    with pytest.raises(ValueError):
        rl.epsilon(0, rl.AgentConfig(eps_schedule="sigmoid"))


def test_untrained_action_rows_flags_only_the_untouched_rows():
    torch = pytest.importorskip("torch")
    init = torch.randn(6, 4)
    now = init.clone() * 0.97                       # decoupled weight decay: collinear
    now[2] += 1.0                                   # this row received a gradient
    mask = rl.untrained_action_rows(now, init)
    assert bool(mask[0]) and bool(mask[5]) and not bool(mask[2])
