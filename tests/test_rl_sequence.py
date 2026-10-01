"""Episode safety and smoke tests for sequence-aware PPO/SAC."""

import numpy as np
import pytest
import torch

from qlsgym import load_molecule
from qlsgym.env import ActionLibrary, ControlGrid, EnvConfig, PurificationEnv
from qlsgym.rl.off_policy import DiscreteSACAgent, SACConfig, sac_policy
from qlsgym.rl.ppo import PPOConfig, policy_from_state_dict, train_ppo
from qlsgym.rl.sequence import RollingHistory, branch_histories
from qlsgym.rl.sequence import HistoryEncoder

from _fake_tables import FakeEngine, fake_tables


@pytest.fixture(scope="module")
def sequence_env():
    molecule = load_molecule("synthetic")
    library = ActionLibrary(
        molecule, ControlGrid.rl_discrete(molecule, n_freq=4, n_tau_slots=2, seed=3)
    )
    tables = fake_tables(molecule, library, FakeEngine(molecule, seed=4))
    return PurificationEnv(
        molecule, library, tables, EnvConfig(rho=0.0), batch=4
    )


def test_history_is_causal_and_reset_does_not_cross_episodes():
    initial = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    history = RollingHistory(initial, context_len=3, n_actions=5, max_pulses=10)
    history.append(
        torch.tensor([[0.8, 0.2], [0.1, 0.9]]),
        torch.tensor([1, 2]), torch.tensor([0, 1]), torch.tensor([1, 1]),
    )
    assert history.view()["valid"].sum(1).tolist() == [2, 2]
    history.reset_rows(torch.tensor([True, False]), initial)
    assert history.view()["valid"].sum(1).tolist() == [1, 2]
    assert history.view()["previous_action"][0, -1].item() == 5
    torch.testing.assert_close(history.view()["state"][0, -1], initial[0])


def test_counterfactual_histories_keep_both_measurement_branches():
    initial = torch.tensor([[0.5, 0.5]])
    history = RollingHistory(initial, context_len=2, n_actions=3, max_pulses=8)
    branches = branch_histories(
        history.view(),
        torch.tensor([[[0.9, 0.1], [0.2, 0.8]]]),
        torch.tensor([2]), torch.tensor([0.875]),
    )
    torch.testing.assert_close(
        branches["state"][:, -1], torch.tensor([[0.9, 0.1], [0.2, 0.8]])
    )
    assert branches["previous_action"][:, -1].tolist() == [2, 2]
    assert branches["previous_outcome"][:, -1].tolist() == [0, 1]


def test_causal_transformer_is_finite_with_leading_padding():
    history = RollingHistory(
        torch.tensor([[0.5, 0.5]]), context_len=4, n_actions=3, max_pulses=8
    )
    encoder = HistoryEncoder(
        n_states=2, n_actions=3, context_len=4, encoder="transformer",
        d_model=8, n_layers=2, n_heads=2, ff_dim=16, dropout=0.0,
        obs="sqrt",
    )
    assert torch.isfinite(encoder(history.view())).all()


def test_token_layer_norm_is_optional_and_finite():
    history = RollingHistory(
        torch.tensor([[0.5, 0.5]]), context_len=4, n_actions=3, max_pulses=8
    )
    encoder = HistoryEncoder(
        n_states=2, n_actions=3, context_len=4, encoder="transformer",
        d_model=8, n_layers=1, n_heads=2, ff_dim=16, dropout=0.0,
        obs="sqrt", token_norm=True,
    )
    encoded = encoder(history.view())
    assert encoded.shape == (1, 8)
    assert torch.isfinite(encoded).all()


@pytest.mark.parametrize("encoder", ["stack", "gru", "transformer"])
def test_sequence_ppo_runs_with_s18_targets(sequence_env, encoder):
    env = sequence_env.clone(4)
    cfg = PPOConfig(
        n_envs=4, n_steps=2, total_steps=8, epochs=1, minibatches=1,
        eval_every=1, eval_rollouts=2, value_target="qmdp",
        encoder=encoder, context_len=2, d_model=16, sequence_layers=1,
        n_heads=4, ff_dim=32, seed=2,
    )
    result = train_ppo(env, cfg)
    assert result.env_steps == 8
    policy = policy_from_state_dict(
        result.final_state_dict, result.n_in, result.n_actions, cfg,
        max_pulses=env.cfg.max_pulses,
    )
    action = policy.act(env.p_init, 0, np.random.default_rng(0))
    assert 0 <= action < env.n_actions


@pytest.mark.parametrize("encoder", ["stack", "gru", "transformer"])
def test_sequence_sac_runs_with_episode_aware_replay(sequence_env, encoder):
    cfg = SACConfig(
        n_envs=2, total_steps=4, batch_size=2, buffer_size=16,
        learning_starts=0, encoder=encoder, context_len=2, d_model=16,
        sequence_layers=1, n_heads=4, ff_dim=32, autotune_alpha=False,
        seed=5,
    )
    agent = DiscreteSACAgent(sequence_env, cfg)
    stats = agent.train(log_points=1)
    assert stats.env_steps == 4 and stats.gradient_steps == 2
    policy = sac_policy(agent)
    action = policy.act(sequence_env.p_init, 0, np.random.default_rng(0))
    assert 0 <= action < sequence_env.n_actions
