"""Branch residuals must not leak across episode resets."""

import torch
from qlsgym.rl.ppo import compute_advantages


def test_qmdp_gae_resets():
    reward = torch.zeros(3, 2)
    value = torch.ones(3, 2)
    branch = value + torch.tensor([[1., 2.], [3., 4.], [5., 6.]])
    terminal = torch.tensor([[0., 1.], [1., 0.], [0., 0.]])
    advantage, target = compute_advantages(reward, value, terminal, torch.zeros(2), branch, 1., .5, "qmdp_gae")
    torch.testing.assert_close(advantage, torch.tensor([[2.5, 2.], [3., 7.], [5., 6.]]))
    torch.testing.assert_close(target, advantage + value)


def test_zero_lambda_reproduces_one_step_qmdp():
    reward, value, branch = (torch.randn(5, 3) for _ in range(3))
    args = reward, value, torch.zeros_like(value), torch.zeros(3), branch, .99, 0.
    expected = compute_advantages(*args, "qmdp")
    actual = compute_advantages(*args, "qmdp_gae")
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b)
