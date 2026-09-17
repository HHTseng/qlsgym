"""Numerical contracts for the qlsgym off-policy S18 targets."""

import torch

from qlsgym.rl.off_policy import ddqn_s18_target, sac_s18_target


def test_ddqn_uses_online_selection_and_target_evaluation():
    reward = torch.tensor([[-0.1, -0.2]])
    probability = torch.tensor([[0.25, 0.75]])
    continuation = torch.tensor([[1.0, 0.0]])
    online = torch.tensor([[[1.0, 3.0], [9.0, 0.0]]])
    target = torch.tensor([[[8.0, 5.0], [2.0, 7.0]]])

    got = ddqn_s18_target(
        reward, probability, continuation, online, target, gamma=1.0
    )
    # Branch 0 selects action 1 with online Q and evaluates it as 5 in target Q;
    # branch 1 is physically terminal and has no continuation.
    want = 0.25 * (-0.1 + 5.0) + 0.75 * (-0.2)
    assert torch.allclose(got, torch.tensor([want]))


def test_sac_exactly_averages_quantum_branches():
    reward = torch.tensor([[-1.0, -2.0]])
    probability = torch.tensor([[0.4, 0.6]])
    continuation = torch.tensor([[1.0, 0.0]])
    logits = torch.zeros((1, 2, 2))
    q1 = torch.tensor([[[2.0, 4.0], [10.0, 20.0]]])
    q2 = torch.tensor([[[3.0, 3.0], [11.0, 19.0]]])
    alpha = torch.tensor(0.5)

    got = sac_s18_target(
        reward, probability, continuation, logits, q1, q2, alpha, gamma=0.9
    )
    entropy_term = 0.5 * torch.log(torch.tensor(2.0))
    soft_value0 = 0.5 * (2.0 + 3.0) + entropy_term
    want = 0.4 * (-1.0 + 0.9 * soft_value0) + 0.6 * (-2.0)
    assert torch.allclose(got, want[None])


def test_sac_reward_normalization_requires_temperature_normalization():
    reward = torch.tensor([[-1.0, -1.0]])
    probability = torch.tensor([[0.4, 0.6]])
    continuation = torch.ones_like(reward)
    logits = torch.zeros((1, 2, 3))
    q1 = torch.tensor([[[2.0, 4.0, 1.0], [1.0, 2.0, 3.0]]])
    q2 = q1 + 1
    alpha = torch.tensor(0.05)
    raw = sac_s18_target(reward, probability, continuation, logits, q1, q2, alpha, 0.99)
    scaled = sac_s18_target(
        reward / 80, probability, continuation, logits, q1 / 80, q2 / 80,
        alpha / 80, 0.99,
    )
    assert torch.allclose(scaled, raw / 80)
