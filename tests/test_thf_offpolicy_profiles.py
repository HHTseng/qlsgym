"""Contracts for the branch-aware SAC/DDQN transfer profiles."""

import importlib.util
from pathlib import Path

import pytest
import torch

from qlsgym.rl.off_policy import transform_belief


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "thf_rl_agents.py"
SPEC = importlib.util.spec_from_file_location("thf_rl_agents_offpolicy", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
PRESET = MODULE.PRESETS["final"]


def test_sac_refinement_preserves_scale_aware_optuna_contract():
    config = MODULE.offpolicy_config(
        "sac_discrete", "sac_refined_2m", PRESET, seed=7
    )
    assert config.total_steps == 2_000_000
    assert config.n_envs == 16
    assert config.return_scale == 20.0
    assert config.autotune_alpha
    assert config.alpha == pytest.approx(0.013268918812005082)
    assert config.obs == "sqrt"
    assert config.seed == 7


@pytest.mark.parametrize(
    "profile, steps, scale, obs",
    [
        ("ddqn_optuna_1m", 1_000_000, None, "sqrt"),
        ("ddqn_optuna_2m", 2_000_000, None, "sqrt"),
        ("ddqn_scaled20_1m", 1_000_000, 20.0, "sqrt"),
        ("ddqn_raw20_1m", 1_000_000, 20.0, "p"),
    ],
)
def test_ddqn_profiles_change_only_declared_factors(profile, steps, scale, obs):
    config = MODULE.offpolicy_config("ddqn", profile, PRESET, seed=11)
    assert config.total_steps == steps
    assert config.return_scale == scale
    assert config.obs == obs
    assert config.seed == 11


def test_profiles_reject_the_wrong_agent():
    with pytest.raises(ValueError, match="requires"):
        MODULE.offpolicy_config("ddqn", "sac_refined_1m", PRESET, seed=0)


def test_belief_transform_has_explicit_domain_mapping():
    probability = torch.tensor([[0.0, 0.25, 0.75]])
    assert torch.equal(transform_belief(probability, "p"), probability)
    assert torch.allclose(
        transform_belief(probability, "sqrt"), probability.sqrt()
    )
    with pytest.raises(ValueError, match="belief transform"):
        transform_belief(probability, "log")
