import numpy as np
import pytest
import torch

from qlsgym.rl.off_policy import (
    DDQNAgent, DDQNConfig, DiscreteSACAgent, SACConfig, transform_belief,
)


class TinyTransition:
    def __init__(self, belief):
        batch = len(belief)
        self.s0 = belief.clone()
        self.s1 = belief.clone()
        self.pi0 = torch.full((batch,), 0.5, device=belief.device)
        self.pi1 = torch.full((batch,), 0.5, device=belief.device)
        self.r0 = torch.zeros(batch, device=belief.device)
        self.r1 = torch.zeros(batch, device=belief.device)
        self.done0 = torch.zeros(batch, dtype=torch.bool, device=belief.device)
        self.done1 = torch.zeros(batch, dtype=torch.bool, device=belief.device)
        self.done = torch.zeros(batch, dtype=torch.bool, device=belief.device)
        self.truncated = torch.zeros(batch, dtype=torch.bool, device=belief.device)
        self.belief = belief


class TinyEnv:
    n_states = 3
    n_actions = 2
    device = torch.device("cpu")

    class Config:
        max_pulses = 100

    cfg = Config()

    def __init__(self, batch=2):
        self.batch = batch
        self.state = torch.full((batch, self.n_states), 1 / self.n_states)
        self.steps = torch.zeros(batch, dtype=torch.long)

    def clone(self, batch):
        return TinyEnv(batch)

    def reset(self, seed=None, batch=None):
        if batch is not None:
            self.batch = batch
        self.state = torch.full((self.batch, self.n_states), 1 / self.n_states)
        self.steps = torch.zeros(self.batch, dtype=torch.long)
        return self.state

    def reset_rows(self, rows):
        self.state[rows] = 1 / self.n_states
        self.steps[rows] = 0

    def step(self, action):
        self.steps += 1
        return TinyTransition(self.state)


def test_ddqn_training_callback_runs_at_each_log_point():
    config = DDQNConfig(
        n_envs=2, total_steps=8, learning_starts=100, buffer_size=16,
        batch_size=2, seed=3,
    )
    agent = DDQNAgent(TinyEnv(), config)
    calls = []
    stats = agent.train(log_points=4, on_log=lambda model, record: calls.append(record["env_steps"]))
    assert calls == [2, 4, 6, 8]
    assert stats.env_steps == 8


def test_sac_uses_configured_return_scale(monkeypatch):
    captured = {}

    def fake_target(branch_reward, branch_probability, branch_continue,
                    next_logits, target_next_q1, target_next_q2, alpha, gamma):
        captured["branch_reward"] = branch_reward.detach().clone()
        return torch.zeros(len(branch_reward), device=branch_reward.device)

    monkeypatch.setattr("qlsgym.rl.off_policy.sac_s18_target", fake_target)
    config = SACConfig(
        n_envs=2, total_steps=2, batch_size=2, buffer_size=4,
        learning_starts=0, return_scale=5.0, autotune_alpha=False, seed=4,
    )
    agent = DiscreteSACAgent(TinyEnv(), config)
    state = agent.env.reset(seed=4, batch=2)
    action = torch.zeros(2, dtype=torch.long)
    transition = agent.env.step(action)
    transition.r0.fill_(-1.0)
    transition.r1.fill_(-1.0)
    agent._record_transition(state, action, transition)
    agent._update()
    np.testing.assert_allclose(captured["branch_reward"].numpy(), -0.2)


def test_belief_transform_supports_raw_and_sqrt_inputs():
    belief = torch.tensor([[0.0, 0.25, 1.0]])
    torch.testing.assert_close(transform_belief(belief, "p"), belief)
    torch.testing.assert_close(
        transform_belief(belief, "sqrt"), torch.tensor([[0.0, 0.5, 1.0]])
    )
