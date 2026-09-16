"""Legacy-table regression path: the old gym's .npz tables as a single-block PurificationEnv."""
import os

import numpy as np
import pytest

from qlsgym.env.table import legacy_table_env, legacy_tables
from qlsgym.policies import SweepingPolicy, rollout

CAH_NPZ = ("/n/home02/josemm/RESEARCH/PROJECTS/MOLESQLS/OLD_GYM/"
           "torch-quantum-spect-gym/src/qlsgym_torch/data/cah_j12.npz")

pytestmark = pytest.mark.skipif(not os.path.exists(CAH_NPZ), reason=f"{CAH_NPZ} not found")


def test_legacy_tables_load_cah_j12():
    mol, lib, tables, p_init, meta = legacy_tables(CAH_NPZ, p_target=0.99, max_pulses=1000)
    assert mol.n_states == 16
    assert lib.n_actions == 13
    assert tables.blocks[0].shape == (13, 32, 16)
    assert np.isclose(p_init.sum(), 1.0)


def test_legacy_table_env_is_a_single_block_purification_env():
    env = legacy_table_env(CAH_NPZ, p_target=0.99, max_pulses=1000, rho=0.0)
    s = env.reset(seed=0)
    assert s.shape == (1, 16)
    assert np.isclose(float(s.sum()), 1.0)
    tr = env.step(np.array([0]))
    assert tr.belief.shape == (1, 16)


def test_sweeping_reproduces_cah_expected_length():
    """Old gym's test_acceptance.py: sweeping on CaH+ has E[L] = 9.8038 exactly (an exact tree sum
    there, not a Monte-Carlo estimate).
    """
    env = legacy_table_env(CAH_NPZ, p_target=0.99, max_pulses=1000, rho=0.0, batch=20000)
    policy = SweepingPolicy(env.n_actions)
    res = rollout(env, policy, n_rollouts=20000, seed=0)
    assert res.success_fraction > 0.999
    sem = float(np.std(res.lengths)) / np.sqrt(res.n_rollouts)
    assert abs(res.mean_pulses - 9.8038) < max(4 * sem, 0.05)
