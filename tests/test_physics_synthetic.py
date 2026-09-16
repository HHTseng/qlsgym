"""Physics core on the synthetic molecule: fast, no tables, no GPU."""
import numpy as np
import pytest
import torch

from qlsgym import load_molecule
from qlsgym.spec import TauBatchedEngine, BatchedEngine, check_branches, branches_batch_fallback
from qlsgym.physics.hamiltonian import (hermiticity_error, lump, make_operator, operator,
                                        rwa_sectors, sideband_matrix_elements)
from qlsgym.physics.propagate import propagator_series, transfer_matrix
from qlsgym.physics.torch_ops import transfer_columns
from qlsgym.physics.engines import (ExactEngine, TorchExactEngine, TrivialEngine, NoisyExactEngine,
                                    TableEngine)
from qlsgym.physics.thermal import boltzmann
from qlsgym.physics.spectrum import resonant_frequencies, embedding_transitions


@pytest.fixture(scope="module")
def mol():
    return load_molecule("synthetic", n_nu=3)


@pytest.fixture(scope="module")
def omegas(mol):
    w = mol.window
    res = resonant_frequencies(mol, 0, "+")
    inside = res[(res > w.omega_min) & (res < w.omega_max)]
    return np.array([w.omega_min + 1.0, 0.5 * (w.omega_min + w.omega_max), w.omega_max - 1.0] + list(inside[:2]))


def test_sideband_elements_debye_waller():
    eta = 0.08
    d = sideband_matrix_elements(eta, 4)
    assert abs(d[0] - 1j * eta * np.exp(-eta ** 2 / 2)) < 1e-12
    assert abs(d[1] - 1j * eta * np.sqrt(2) * np.exp(-eta ** 2 / 2) * (1 - eta ** 2 / 2)) < 1e-12


def test_hermitian_and_dims(mol, omegas):
    for b in mol.blocks:
        for sg in ("+", "-"):
            op = make_operator(mol, b, sg)
            for w in omegas:
                h = op.build(w)
                assert h.shape == (mol.trap.n_nu * b.n_states,) * 2
                assert hermiticity_error(h) < 1e-12
                # component referencing: every decoupled component has zero-mean diagonal
                d = np.real(np.diag(h))
                for c in range(op.n_component_n):
                    assert abs(d[op.component_n == c].mean()) < 1e-9 * max(1.0, np.abs(d).max())
            h2 = op.build(omegas[0], n_nu=2)
            assert h2.shape == (2 * b.n_states,) * 2 and hermiticity_error(h2) < 1e-12
            # the nu = 0 -> 1 couplings are the same in both truncations
            m = b.n_states
            assert np.array_equal(h2[m:, :m], op.build(omegas[0])[m:2 * m, :m])


def test_operator_cache_keyed_on_fingerprint(mol):
    a = operator(mol, 0, "+")
    assert operator(mol, 0, "+") is a
    other = load_molecule("synthetic", seed=1, n_nu=3)
    assert other.fingerprint() != mol.fingerprint()
    assert operator(other, 0, "+") is not a


def test_probability_conservation_and_lump(mol, omegas):
    taus = mol.tau_grid()
    for b in mol.blocks:
        m = b.n_states
        op = make_operator(mol, b, "+")
        u = propagator_series(op.build(omegas[1]), taus)
        t = np.abs(u[:, :, :m]) ** 2
        assert t.shape == (taus.size, mol.trap.n_nu * m, m)
        assert np.allclose(t.sum(1), 1.0, atol=1e-12)          # column stochastic
        tl = lump(t, m)
        assert tl.shape == (taus.size, 2 * m, m)
        assert np.allclose(tl[:, :m], t[:, :m]) and np.allclose(tl[:, m:], t[:, m:].reshape(taus.size, -1, m, m).sum(1))
        assert np.allclose(transfer_matrix(mol, b.index, omegas[1], "+"), tl)
    sub = transfer_matrix(mol, 0, omegas[1], "-", tau_indices=[0, 5])
    assert sub.shape == (2, 2 * mol.blocks[0].n_states, mol.blocks[0].n_states)


def test_transfer_columns_torch_matches_numpy(mol, omegas):
    for sg in ("+", "-"):
        tc = transfer_columns(mol, 1, omegas, sg, out_dtype=torch.float64)
        assert tc.shape == (omegas.size, mol.window.n_tau, 2 * mol.blocks[1].n_states, mol.blocks[1].n_states)
        for k, w in enumerate(omegas):
            assert np.abs(tc[k].numpy() - transfer_matrix(mol, 1, w, sg)).max() < 1e-12
    # primitive frequency: one off-window omega, RWA at that drive
    pr = mol.primitives[0]
    sec = rwa_sectors(mol, pr.sigma, pr.omega)
    for s in sec:
        tc = transfer_columns(mol, s, np.array([pr.omega]), pr.sigma, out_dtype=torch.float64)[0].numpy()
        assert np.abs(tc - transfer_matrix(mol, s, pr.omega, pr.sigma)).max() < 1e-12
    with pytest.raises(ValueError):
        transfer_columns(mol, 0, np.array([omegas[0], pr.omega]), "+")


def test_engines_satisfy_protocol_and_conserve(mol, omegas):
    p = boltzmann(mol)
    assert abs(p.sum() - 1) < 1e-12
    engines = [ExactEngine(mol), TorchExactEngine(mol), TrivialEngine(mol),
               NoisyExactEngine(mol, rel_l1=0.05, purity_gamma=0.2, halluc_nu1=0.1, seed=3)]
    engines.append(TableEngine.from_engine(engines[0], [(sg, w) for sg in "+-" for w in omegas]
                                           + [(pr.sigma, pr.omega) for pr in mol.primitives]))
    for e in engines:
        assert isinstance(e, TauBatchedEngine)
        assert e.tau_indices.size == mol.window.n_tau
        for sg in ("+", "-"):
            for w in omegas:
                p0, p1 = e.branches_all_tau(p, w, sg)
                assert p0.shape == (mol.window.n_tau, mol.n_states) and p0.dtype == np.float64
                check_branches(p0, p1, mol.n_states)
        for pr in mol.primitives:
            p0, p1 = e.branches_all_tau(p, pr.omega, pr.sigma)
            check_branches(p0, p1, mol.n_states)
        with pytest.raises((ValueError, KeyError)):
            e.branches_all_tau(p, mol.window.omega_max + 12345.0, "+")
    assert isinstance(engines[1], BatchedEngine)


def test_trivial_engine_semantics(mol, omegas):
    p = boltzmann(mol)
    e = TrivialEngine(mol)
    p0, p1 = e.branches_all_tau(p, omegas[3], "+")
    assert np.allclose(p0, p[None, :]) and np.all(p1 == 0)      # sigma+ in-window: p(tau) = p(0)
    q0, q1 = e.branches_all_tau(p, omegas[3], "-")             # sigma- is exact, as in the source
    ex = ExactEngine(mol).branches_all_tau(p, omegas[3], "-")
    assert np.allclose(q0, ex[0]) and np.allclose(q1, ex[1])
    both = TrivialEngine(mol, sigmas=("+", "-"))
    r0, r1 = both.branches_all_tau(p, omegas[3], "-")
    assert np.allclose(r0, p[None, :]) and np.all(r1 == 0)
    two = two_block_molecule()                                  # primitives are exact
    pr = two.primitives[0]
    q = np.full(two.n_states, 0.25)
    a = TrivialEngine(two).branches_all_tau(q, pr.omega, pr.sigma)
    b = ExactEngine(two).branches_all_tau(q, pr.omega, pr.sigma)
    assert np.allclose(a[0], b[0]) and np.allclose(a[1], b[1]) and a[1].sum() > 1e-3


def test_exact_equals_torch_and_batch(mol, omegas):
    p = boltzmann(mol)
    idx = np.array([0, 3, 7, mol.window.n_tau - 1])
    ne, te = ExactEngine(mol, tau_indices=idx), TorchExactEngine(mol, tau_indices=idx)
    for sg in ("+", "-"):
        for w in list(omegas) + [pr.omega for pr in mol.primitives]:
            a0, a1 = ne.branches_all_tau(p, w, sg)
            b0, b1 = te.branches_all_tau(p, w, sg)
            assert a0.shape == (idx.size, mol.n_states)
            assert np.abs(a0 - b0).max() < 1e-10 and np.abs(a1 - b1).max() < 1e-10
        c0, c1 = te.branches_batch(p, omegas, sg)
        d0, d1 = branches_batch_fallback(ne, p, omegas, sg)
        assert c0.shape == (omegas.size, idx.size, mol.n_states)
        assert np.abs(c0 - d0).max() < 1e-10 and np.abs(c1 - d1).max() < 1e-10
        mixed = np.array([omegas[0], mol.primitives[0].omega])
        e0, e1 = te.branches_batch(p, mixed, sg)
        f0, f1 = branches_batch_fallback(ne, p, mixed, sg)
        assert np.abs(e0 - f0).max() < 1e-10 and np.abs(e1 - f1).max() < 1e-10
    # single-tau accessor agrees with the row
    a0, a1 = ne.branches(p, omegas[1], "+", 7)
    assert np.allclose(a0, ne.branches_all_tau(p, omegas[1], "+")[0][2])


def two_block_molecule():
    """Two 2-state blocks ("J manifolds" 2pi*3000 rad/ms apart) with one cross-block coupling 1 ->
    2 and a sigma+ primitive resonant with it.
    """
    from qlsgym.spec import Block, Molecule, Primitive, System, Task, Trap, Window, TWO_PI
    nu_f, eta = TWO_PI * 1000.0, 0.08
    e = TWO_PI * np.array([0.0, 20.0, 3000.0, 3020.0])
    w_intra, w_cross = TWO_PI * 1.5 * np.exp(0.3j), TWO_PI * 5.0 * np.exp(-0.7j)   # pi-time 1.26 ms fits the 2 ms grid
    blocks = (Block(0, np.array([0, 1]), (0, "+"), np.array([0]), np.array([1]), np.array([w_intra])),
              Block(1, np.array([2, 3]), (1, "+"), np.array([0]), np.array([1]), np.array([w_intra])))
    system = System(((0, 0), (0, 1), (1, 0), (1, 1)), e, np.array([0, 2, 1]), np.array([1, 3, 2]),
                    np.array([w_intra, w_intra, w_cross]), blocks, np.array([0, 0, 1, 1]))
    window = Window(nu_f - TWO_PI * 100, nu_f + TWO_PI * 100, 2.0, 16, rwa_cutoff=TWO_PI * 200, omega_min_coupling=1.0)
    tau_pi = np.pi / (eta * np.exp(-eta ** 2 / 2) * abs(w_cross))
    prim = Primitive("cross", "+", nu_f + (e[2] - e[1]), tau_pi, (0, 1), source=1, target=2)
    return Molecule("two_block", system, Trap(nu_f, eta, 3), window, Task(1.0), (prim,))


def test_synthetic_primitive_is_identity(mol):
    """The lead's synthetic primitive sits at 2pi*5e6 rad/ms where no coupling resonates: the RWA
    keeps nothing and the pulse is the identity.
    """
    pr = mol.primitives[0]
    assert rwa_sectors(mol, pr.sigma, pr.omega) == ()
    p = boltzmann(mol)
    p0, p1 = ExactEngine(mol).branches_all_tau(p, pr.omega, pr.sigma)
    assert np.array_equal(p0, np.tile(p, (mol.window.n_tau, 1))) and np.all(p1 == 0)


def test_primitive_couples_its_blocks():
    two = two_block_molecule()
    pr = two.primitives[0]
    secs = rwa_sectors(two, pr.sigma, pr.omega)
    assert len(secs) == 1 and np.array_equal(secs[0].states, [1, 2]) and secs[0].key[3] == (0, 1)
    taus = two.tau_grid()
    p = np.zeros(4); p[pr.source] = 1.0
    for eng in (ExactEngine(two), TorchExactEngine(two)):
        p0, p1 = eng.branches_all_tau(p, pr.omega, pr.sigma)
        check_branches(p0, p1, 4)
        k = int(np.argmin(np.abs(taus - pr.tau_ms)))
        assert p1[k, pr.target] > 0.95                          # pi-pulse moves source -> target, nu = 1
        assert np.all(p1[:, [0, 3]] == 0) and np.all(p0[:, [0, 3]] == 0)   # untouched states are untouched
        # a sigma- pulse at the same frequency keeps no coupling: identity
        q0, q1 = eng.branches_all_tau(p, pr.omega, "-")
        assert np.array_equal(q0, np.tile(p, (taus.size, 1))) and np.all(q1 == 0)
    # in-window the two blocks are independent (the cross coupling is not in any block)
    w = two.trap.nu_f + (two.system.energies[1] - two.system.energies[0])
    p0, p1 = ExactEngine(two).branches_all_tau(p, w, "+")
    assert np.all(p1[:, [2, 3]] == 0) and p1[:, 1].max() < 1e-6


def test_embedding_transitions(mol):
    idx = embedding_transitions(mol, 0, "+")
    assert idx.dtype.kind == "i" and idx.size == mol.blocks[0].omega.size
