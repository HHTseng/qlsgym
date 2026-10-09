import numpy as np
import pytest

from qlsgym.molecules import thf229
from qlsgym.physics.propagate import transfer_matrix
from qlsgym.physics.spectrum import resonant_frequencies
from qlsgym.spec import TWO_PI


@pytest.fixture(scope="module")
def mol():
    return thf229.build()


def test_structure_and_closed_blocks(mol):
    lv = mol.system.levels
    assert mol.n_states == 24 * sum(2 * j + 1 for j in range(1, thf229.J_MAX + 1))
    assert max(b.n_states for b in mol.blocks) == 24 and len(mol.blocks) == 28
    s = mol.system
    w = mol.window
    d = s.energies[s.f_idx] - s.energies[s.i_idx]
    near = np.zeros(d.size, dtype=bool)
    for w_res in (mol.trap.nu_f + d, mol.trap.nu_f - d):
        near |= np.maximum(0.0, np.maximum(w.omega_min - w_res, w_res - w.omega_max)) <= w.rwa_cutoff
    assert np.all(s.block_of_state[s.i_idx[near]] == s.block_of_state[s.f_idx[near]])
    mf = np.array([l[4] for l in lv])
    par = np.array([l[3] for l in lv])
    assert np.allclose(mf[s.f_idx] - mf[s.i_idx], 1.0) and np.all(par[s.f_idx] == par[s.i_idx])
    for b in mol.blocks:
        assert len({(lv[g][0], lv[g][3]) for g in b.states}) == 1


def test_reference_rung_sets_the_scale(mol):
    ref = thf229.CONFIG["tables"]["reference_transition"]
    idx = {l: n for n, l in enumerate(mol.system.levels)}
    i, f = (idx[(t[0], float(t[1]), float(t[2]), t[3], float(t[4]), t[5])] for t in ref)
    k = np.where((mol.system.i_idx == i) & (mol.system.f_idx == f))[0]
    assert k.size == 1
    assert abs(mol.system.omega_c[k[0]]) / TWO_PI == pytest.approx(thf229.CONFIG["tables"]["reference_rabi_khz"])


def test_propagation_conserves_probability(mol):
    w = mol.window
    for b in (0, 6, 18):
        r = resonant_frequencies(mol, b, "+")
        om = r[(r >= w.omega_min) & (r <= w.omega_max)][0]
        for sigma in "+-":
            t = transfer_matrix(mol, b, om, sigma)
            np.testing.assert_allclose(t.sum(1), 1.0, atol=1e-10)
            assert t.min() >= -1e-12


@pytest.mark.slow
def test_motional_cutoff_closes_the_largest_ladders():
    a, c = thf229.build(n_nu=thf229.N_NU), thf229.build(n_nu=thf229.N_NU + 3)
    w = a.window
    for b in (0, 2, 6):
        r = resonant_frequencies(a, b, "+")
        for om in r[(r >= w.omega_min) & (r <= w.omega_max)]:
            assert np.abs(transfer_matrix(a, b, om, "+") - transfer_matrix(c, b, om, "+")).max() < 1e-8


@pytest.mark.slow
def test_shipped_tables_equal_a_fresh_build(tmp_path):
    pytest.importorskip("heff", reason="heff not on PYTHONPATH: source scripts/env.sh")
    from qlsgym.molecules import thf229_tables as T
    levels, cpl = T.build_tables()
    fresh = T.table_files(tmp_path)
    T.write_tables(levels, cpl, *fresh)
    for got, want in zip(fresh, T.table_files(thf229.data_dir())):
        with open(got) as a, open(want) as b:
            assert a.read().splitlines()[1:] == b.read().splitlines()[1:], want
