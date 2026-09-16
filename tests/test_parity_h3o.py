"""Acceptance: the H3O+ pack reproduces fnorepl."""
import numpy as np
import pytest

fnorepl = pytest.importorskip("fnorepl", reason="fnorepl not on PYTHONPATH: source scripts/env.sh (parity test SKIPPED)")

from fnorepl import planner as fplanner, propagate as fpropagate, system as fsystem, dataset as fdataset  # noqa: E402

from qlsgym import load_molecule  # noqa: E402
from qlsgym.physics.engines import ExactEngine, TorchExactEngine  # noqa: E402
from qlsgym.physics.hamiltonian import make_operator, rwa_sectors  # noqa: E402
from qlsgym.physics.propagate import transfer_matrix  # noqa: E402
from qlsgym.physics.spectrum import embedding_transitions, resonant_frequencies  # noqa: E402
from qlsgym.physics.thermal import boltzmann  # noqa: E402

TOL = 1e-10
BLOCKS = (0, 2, 5, 8)
TAU_IDX = np.array([0, 40, 99, 150, 199])


@pytest.fixture(scope="module")
def mol():
    return load_molecule("h3o")


@pytest.fixture(scope="module")
def src():
    return fsystem.default_system()


@pytest.fixture(scope="module")
def omegas(mol):
    w = mol.window
    res = resonant_frequencies(mol, 0, "+")
    res = np.sort(res[(res > w.omega_min) & (res < w.omega_max)])
    return np.array([w.omega_min, res[0], res[len(res) // 2], 0.5 * (w.omega_min + w.omega_max) + 7.0, w.omega_max])


def test_blocks_identical(mol, src):
    assert mol.n_states == src.n_states == 444
    assert np.array_equal(mol.system.energies, src.energies)
    assert np.array_equal(mol.system.i_idx, src.i_idx) and np.array_equal(mol.system.f_idx, src.f_idx)
    assert np.array_equal(mol.system.omega_c, src.omega_c)
    assert np.array_equal(mol.system.block_of_state, src.block_of_state)
    assert len(mol.blocks) == len(src.blocks)
    for a, b in zip(mol.blocks, src.blocks):
        assert a.index == b.index and a.key == b.key
        assert np.array_equal(a.states, b.states)
        assert np.array_equal(a.i_local, b.i_local) and np.array_equal(a.f_local, b.f_local)
        assert np.array_equal(a.omega, b.omega)
    assert list(mol.system.levels) == list(src.levels)


def test_boltzmann_identical(mol, src):
    assert np.array_equal(boltzmann(mol), fdataset.boltzmann_populations(src, 20.0))
    assert np.array_equal(boltzmann(mol), fplanner.initial_state(src))


def test_embedding_transitions_identical(mol, src):
    for b in BLOCKS:
        for sg in "+-":
            assert np.array_equal(embedding_transitions(mol, b, sg),
                                  fsystem.embedding_transitions(src, src.blocks[b], sg))


def test_hamiltonian_and_transfer_matrix(mol, src, omegas):
    """n_nu = 2 reproduces fnorepl.hamiltonian.make_operator(...).build exactly (same basis
    ordering) and transfer_matrix equals population_transfer_matrix's nu = 0 columns.
    """
    from fnorepl.hamiltonian import make_operator as f_make
    taus = mol.tau_grid()[TAU_IDX]
    worst = 0.0
    for b in BLOCKS:
        m = mol.blocks[b].n_states
        for sg in "+-":
            op, fop = make_operator(mol, b, sg), f_make(src, src.blocks[b], sg)
            for w in omegas:
                assert np.array_equal(op.build(w), fop.build(w))
                t = transfer_matrix(mol, b, w, sg, TAU_IDX)
                ft = fpropagate.population_transfer_matrix(src, src.blocks[b], w, sg, taus)[:, :, :m]
                assert t.shape == ft.shape == (TAU_IDX.size, 2 * m, m)
                worst = max(worst, np.abs(t - ft).max())
    print(f"\nH3O+ transfer_matrix residual {worst:.3e}")
    assert worst < TOL


def test_engine_branches_identical(mol, src, omegas):
    p = boltzmann(mol)
    fe = fplanner.ExactTauBatchedEngine(src, tau_indices=TAU_IDX)
    fe.taus = mol.tau_grid()                      # see module docstring
    me = ExactEngine(mol, TAU_IDX)
    worst = 0.0
    for sg in "+-":
        for w in omegas:
            a0, a1 = me.branches_all_tau(p, w, sg)
            b0, b1 = fe.branches_all_tau(p, w, sg)
            worst = max(worst, np.abs(a0 - b0).max(), np.abs(a1 - b1).max())
    print(f"\nH3O+ branches_all_tau residual {worst:.3e}")
    assert worst < TOL
    # one full-grid call, both channels
    fe = fplanner.ExactTauBatchedEngine(src); fe.taus = mol.tau_grid()
    me = ExactEngine(mol)
    for sg in "+-":
        a0, a1 = me.branches_all_tau(p, omegas[1], sg)
        b0, b1 = fe.branches_all_tau(p, omegas[1], sg)
        assert a0.shape == (200, 444)
        assert np.abs(a0 - b0).max() < TOL and np.abs(a1 - b1).max() < TOL


def test_primitives_match_source(mol, src):
    thz = fplanner.thz_primitives(src, n_families=2)
    assert len(thz) == len(mol.primitives) == 21
    for p, (bidx, w, label) in zip(mol.primitives, thz):
        assert p.blocks == (bidx,) and abs(p.omega - w) < 1e-9 * abs(w) and p.sigma == "+"
        assert label.split(" ")[0] == p.label.split(" ")[0]
    # the engine at one primitive (fnorepl propagates the whole block with the
    # RWA taken at that drive; we propagate the RWA sector -- same physics)
    p = boltzmann(mol)
    pr = mol.primitives[0]
    fe = fplanner.ExactTauBatchedEngine(src, tau_indices=TAU_IDX); fe.taus = mol.tau_grid()
    me = ExactEngine(mol, TAU_IDX)
    a0, a1 = me.branches_all_tau(p, pr.omega, pr.sigma)
    b0, b1 = fe.branches_all_tau(p, pr.omega, pr.sigma)
    res = max(np.abs(a0 - b0).max(), np.abs(a1 - b1).max())
    print(f"\nH3O+ primitive residual {res:.3e}  (Pi_1 max {a1.sum(1).max():.4f})")
    assert res < TOL and a1.sum(1).max() > 0.01
    secs = rwa_sectors(mol, pr.sigma, pr.omega)
    assert all(set(s.key[3]) <= set(pr.blocks) for s in secs)


def test_torch_engine_matches_source(mol, src, omegas):
    p = boltzmann(mol)
    fe = fplanner.ExactTauBatchedEngine(src, tau_indices=TAU_IDX); fe.taus = mol.tau_grid()
    te = TorchExactEngine(mol, "cpu", TAU_IDX)
    for sg in "+-":
        c0, c1 = te.branches_batch(p, omegas[:3], sg)
        for k, w in enumerate(omegas[:3]):
            b0, b1 = fe.branches_all_tau(p, w, sg)
            assert np.abs(c0[k] - b0).max() < TOL and np.abs(c1[k] - b1).max() < TOL
