"""Acceptance: the ThF+ pack reproduces thffno."""
import numpy as np
import pytest

thffno = pytest.importorskip("thffno", reason="thffno not on PYTHONPATH: source scripts/env.sh (parity test SKIPPED)")

from thffno import planner as tplanner, propagate as tpropagate, system as tsystem, dataset as tdataset  # noqa: E402

from qlsgym import load_molecule  # noqa: E402
from qlsgym.physics.engines import ExactEngine, TorchExactEngine  # noqa: E402
from qlsgym.physics.hamiltonian import make_operator, rwa_sectors  # noqa: E402
from qlsgym.physics.propagate import transfer_matrix  # noqa: E402
from qlsgym.physics.spectrum import carrier_bands, embedding_transitions, resonant_frequencies  # noqa: E402
from qlsgym.physics.thermal import boltzmann  # noqa: E402

TOL = 1e-10
BLOCKS = (0, 3, 6, 11)
TAU_IDX = np.array([0, 40, 99, 150, 199])


@pytest.fixture(scope="module")
def mol():
    return load_molecule("thf")


@pytest.fixture(scope="module")
def src():
    return tsystem.default_system()


@pytest.fixture(scope="module")
def omegas(mol):
    w = mol.window
    res = resonant_frequencies(mol, 0, "+")
    res = np.sort(res[(res > w.omega_min) & (res < w.omega_max)])
    return np.array([w.omega_min, res[0], res[len(res) // 2], 0.5 * (w.omega_min + w.omega_max) + 7.0, w.omega_max])


def test_blocks_identical(mol, src):
    assert mol.n_states == src.n_states == 192
    assert np.array_equal(mol.system.energies, src.energies)
    assert np.array_equal(mol.system.i_idx, src.i_idx) and np.array_equal(mol.system.f_idx, src.f_idx)
    assert np.array_equal(mol.system.omega_c, src.omega_c)
    assert np.array_equal(mol.system.block_of_state, src.block_of_state)
    assert len(mol.blocks) == len(src.blocks) == 12
    for a, b in zip(mol.blocks, src.blocks):
        assert a.index == b.index and a.key == b.key
        assert np.array_equal(a.states, b.states)
        assert np.array_equal(a.i_local, b.i_local) and np.array_equal(a.f_local, b.f_local)
        assert np.array_equal(a.omega, b.omega)
    assert list(mol.system.levels) == list(src.levels)


def test_boltzmann_identical(mol, src):
    assert np.array_equal(boltzmann(mol), tdataset.boltzmann_populations(src, 4.0))
    assert np.array_equal(boltzmann(mol), tplanner.initial_state(src))


def test_window_constants(mol):
    from thffno import units as u
    assert mol.trap.nu_f == u.NU_F and mol.trap.eta == u.ETA and mol.trap.n_nu == u.N_NU
    assert mol.window.omega_min == u.OMEGA_MIN and mol.window.omega_max == u.OMEGA_MAX
    assert mol.window.tau_max_ms == u.TAU_MAX_MS and mol.window.n_tau == u.N_TAU
    assert mol.task.temperature_k == u.T_INT_K
    src_bands = tsystem.carrier_bands(sigma="+")
    assert np.allclose(np.array(mol.window.carrier_bands), src_bands)
    assert np.allclose(carrier_bands(mol, "+"), src_bands) and carrier_bands(mol, "-").shape == (0, 2)


def test_embedding_transitions_identical(mol, src):
    for b in BLOCKS:
        for sg in "+-":
            assert np.array_equal(embedding_transitions(mol, b, sg),
                                  tsystem.embedding_transitions(src, src.blocks[b], sg))


def test_hamiltonian_and_transfer_matrix(mol, src, omegas):
    """build(omega) equals thffno's build_multi(omega, 7) exactly, and transfer_matrix equals
    transfer_columns_lumped.
    """
    from thffno.hamiltonian import make_operator as t_make
    taus = mol.tau_grid()[TAU_IDX]
    worst = 0.0
    for b in BLOCKS:
        m = mol.blocks[b].n_states
        for sg in "+-":
            op, top = make_operator(mol, b, sg), t_make(src, src.blocks[b], sg)
            for w in omegas:
                assert np.array_equal(op.build(w), top.build_multi(w, 7))
                assert np.array_equal(op.build(w, n_nu=2), top.build(w))
                t = transfer_matrix(mol, b, w, sg, TAU_IDX)
                tt = tpropagate.transfer_columns_lumped(src, src.blocks[b], w, sg, taus, 7)
                assert t.shape == tt.shape == (TAU_IDX.size, 2 * m, m)
                worst = max(worst, np.abs(t - tt).max())
    print(f"\nThF+ transfer_matrix residual {worst:.3e}")
    assert worst < TOL


def test_engine_branches_identical(mol, src, omegas):
    p = boltzmann(mol)
    te_ = tplanner.ExactTauBatchedEngine(src, tau_indices=TAU_IDX)
    te_.taus = mol.tau_grid()
    me = ExactEngine(mol, TAU_IDX)
    worst = 0.0
    for sg in "+-":
        for w in omegas:
            a0, a1 = me.branches_all_tau(p, w, sg)
            b0, b1 = te_.branches_all_tau(p, w, sg)
            worst = max(worst, np.abs(a0 - b0).max(), np.abs(a1 - b1).max())
    print(f"\nThF+ branches_all_tau residual {worst:.3e}")
    assert worst < TOL
    fe = tplanner.ExactTauBatchedEngine(src); fe.taus = mol.tau_grid()
    me = ExactEngine(mol)
    for sg in "+-":
        a0, a1 = me.branches_all_tau(p, omegas[1], sg)
        b0, b1 = fe.branches_all_tau(p, omegas[1], sg)
        assert a0.shape == (200, 192)
        assert np.abs(a0 - b0).max() < TOL and np.abs(a1 - b1).max() < TOL


def test_primitives_match_source(mol, src):
    prims = tplanner.cross_j_primitives(src, n_primitives=24, mode="dark_state")
    assert len(prims) == len(mol.primitives) == 24
    for p, q in zip(mol.primitives, prims):
        assert p.sigma == q.sigma and abs(p.omega - q.omega) < 1e-9 * abs(q.omega)
        assert abs(p.tau_ms - q.tau_pi) < 1e-12 and p.source == q.source and p.target == q.target
        assert p.label == q.label
    # sectors and the engine at two primitives (one per polarisation)
    p = boltzmann(mol)
    fe = tplanner.ExactTauBatchedEngine(src, tau_indices=TAU_IDX); fe.taus = mol.tau_grid()
    me = ExactEngine(mol, TAU_IDX)
    worst = 0.0
    for pr in (next(q for q in mol.primitives if q.sigma == "+"), next(q for q in mol.primitives if q.sigma == "-")):
        mine, theirs = rwa_sectors(mol, pr.sigma, pr.omega), tplanner.rwa_sectors(src, pr.sigma, pr.omega)
        assert len(mine) == len(theirs)
        for a, b in zip(mine, theirs):
            assert np.array_equal(a.states, b.states) and a.key[3] == b.key[3]
            assert np.array_equal(a.omega, b.omega)
        assert sorted({b for s in mine for b in s.key[3]}) == sorted(pr.blocks)
        a0, a1 = me.branches_all_tau(p, pr.omega, pr.sigma)
        b0, b1 = fe.branches_all_tau(p, pr.omega, pr.sigma)
        worst = max(worst, np.abs(a0 - b0).max(), np.abs(a1 - b1).max())
        assert a1.sum(1).max() > 1e-3
    print(f"\nThF+ primitive residual {worst:.3e}")
    assert worst < TOL


def test_torch_engine_matches_source(mol, src, omegas):
    p = boltzmann(mol)
    fe = tplanner.ExactTauBatchedEngine(src, tau_indices=TAU_IDX); fe.taus = mol.tau_grid()
    te = TorchExactEngine(mol, "cpu", TAU_IDX)
    for sg in "+-":
        c0, c1 = te.branches_batch(p, omegas[:3], sg)
        for k, w in enumerate(omegas[:3]):
            b0, b1 = fe.branches_all_tau(p, w, sg)
            assert np.abs(c0[k] - b0).max() < TOL and np.abs(c1[k] - b1).max() < TOL
