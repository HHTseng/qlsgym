import numpy as np
import pytest

from qlsgym import load_molecule
from qlsgym.env import ActionLibrary, ControlGrid, EnvConfig, PurificationEnv
from qlsgym.policies import RandomPolicy
from qlsgym.rl.evaluation import evaluate_policy
from _fake_tables import FakeEngine, fake_tables


@pytest.fixture
def environment():
    mol = load_molecule("synthetic")
    lib = ActionLibrary(mol, ControlGrid.rl_discrete(mol, n_freq=3, n_tau_slots=2))
    return PurificationEnv(mol, lib, fake_tables(mol, lib, FakeEngine(mol)),
                           EnvConfig(p_target=0.999, max_pulses=4))


def test_chunked_evaluation_is_repeatable(environment):
    policy = RandomPolicy(environment.n_actions)
    first = evaluate_policy(environment, policy, 13, 17, batch_size=4)
    second = evaluate_policy(environment, policy, 13, 17, batch_size=4)
    assert first["lengths"] == second["lengths"]
    assert first["successes"] == second["successes"]
    assert len(first["lengths"]) == 13
    assert first["average_actions"] == np.mean(first["lengths"])


def test_stateful_policy_has_private_row_memory(environment):
    class Policy:
        stateful = True

        def reset(self):
            self.history = []

        def act(self, belief, t, rng):
            assert len(self.history) == t
            self.history.append(t)
            return 0

    policy = Policy()
    policy.reset()
    result = evaluate_policy(environment, policy, 13, 17, batch_size=4)
    assert len(result["successes"]) == 13
    assert policy.history == []


def test_no_action_failures_receive_horizon_cost(environment):
    class Stop:
        def act(self, belief, t, rng):
            return None

    result = evaluate_policy(environment, Stop(), 13, 17, batch_size=4)
    assert result["unfinished_fraction"] == 1
    assert result["average_actions"] == 4
    assert result["actual_lengths"] == [0] * 13
    assert result["outcomes"]["no_action"] == 1
