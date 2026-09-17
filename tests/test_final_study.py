"""A missing job or changed training/metric contract must not rank."""

import copy
import importlib.util
import sys
from pathlib import Path

import pytest


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
SPEC = importlib.util.spec_from_file_location("final_study", SCRIPTS / "summarize_thf_study.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.fixture
def grid():
    records = []
    for agent in MODULE.AGENTS:
        for seed in (range(5) if agent in MODULE.LEARNED else [0]):
            config = {"total_steps": 1000000, "n_envs": 128, "lr": 3e-4, "gamma": 1.0}
            if agent == "ppo":
                config.update(value_target="qmdp_gae", n_steps=32, epochs=4, minibatches=8,
                              ent_coef=0.01, gae_lambda=0.95, eval_greedy=False)
            if agent == "ddqn":
                config.update(eps_fraction=0.72, gradient_steps=4, learning_starts=10240)
            if agent == "sac_discrete":
                config.update(gamma=0.99, alpha=0.05/80, target_entropy_ratio=0.5,
                              gradient_steps=4, learning_starts=10240)
            result = {"lengths": [80]*5000, "successes": [False]*5000,
                      "average_actions": 80, "unfinished_fraction": 1}
            records.append({"job": {"agent": agent, "train_seed": seed, "preset": "final",
                                    "eval_episodes": 5000, "eval_seed": 20001, "evaluation_batch": 128},
                            "contract": {"manifest_pair_coverage": 1., "p_target": .98, "max_pulses": 80,
                                         "rho": 0., "n_states": 192, "n_actions": 312, "n_nu": 7,
                                         "checkpoint_sha256": {str(k): "hash" for k in range(24)}},
                            "training": {"config": config, "env_steps": 1000064},
                            "evaluation": {"exact": copy.deepcopy(result), "fno": copy.deepcopy(result)}})
    return records


def test_accept_locked_grid(grid):
    MODULE.validate_grid(grid)


def test_reject_missing_seed(grid):
    with pytest.raises(ValueError, match="complete final grid"):
        MODULE.validate_grid(grid[:-1])


def test_reject_duplicate_job(grid):
    with pytest.raises(ValueError, match="complete final grid"):
        MODULE.validate_grid(grid + [grid[-1]])


def test_reject_mislabeled_failures(grid):
    grid[0]["evaluation"]["exact"]["lengths"][0] = 0
    with pytest.raises(ValueError, match="scoring"):
        MODULE.validate_grid(grid)


def test_reject_unscaled_sac_temperature(grid):
    for record in grid:
        if record["job"]["agent"] == "sac_discrete":
            record["training"]["config"]["alpha"] = .05
    with pytest.raises(ValueError, match="hyperparameters"):
        MODULE.validate_grid(grid)
