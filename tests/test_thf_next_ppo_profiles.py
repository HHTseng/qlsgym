"""Contracts for the leak-free next-PPO screening profiles."""

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "thf_rl_agents.py"
SPEC = importlib.util.spec_from_file_location("thf_rl_agents", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
PRESET = MODULE.PRESETS["final"]


@pytest.mark.parametrize(
    "profile, steps, lr",
    [
        ("qmdp_tuned_250k", 250_000, 1.346363908197788e-3),
        ("qmdp_tuned_1m", 1_000_000, 1.346363908197788e-3),
        ("qmdp_tuned_2m", 2_000_000, 1.346363908197788e-3),
        ("qmdp_lr10_1m", 1_000_000, 1e-3),
        ("qmdp_lr17_1m", 1_000_000, 1.7e-3),
    ],
)
def test_qmdp_profiles_preserve_the_optuna_target(profile, steps, lr):
    config = MODULE.ppo_config(profile, PRESET, seed=7)
    assert config.value_target == "qmdp"
    assert config.total_steps == steps
    assert config.lr == pytest.approx(lr)
    assert config.seed == 7


@pytest.mark.parametrize("profile, coefficient", [("jose_aux_01", 0.1), ("jose_aux_03", 0.3)])
def test_auxiliary_profiles_keep_jose_actor_gae(profile, coefficient):
    config = MODULE.ppo_config(profile, PRESET, seed=3)
    assert config.value_target == "gae"
    assert config.branch_aux_coef == coefficient
    assert config.total_steps == 2_000_000


def test_jose_qmdp_changes_only_the_value_target():
    jose = MODULE.ppo_config("jose_main", PRESET, seed=0).as_dict()
    branch = MODULE.ppo_config("jose_qmdp", PRESET, seed=0).as_dict()
    differing = {key for key in jose if jose[key] != branch[key]}
    assert differing == {"value_target"}
