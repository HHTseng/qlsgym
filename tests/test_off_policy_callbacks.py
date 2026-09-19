import numpy as np
import pytest
import torch

from qlsgym.rl.off_policy import DDQNAgent, DDQNConfig


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
