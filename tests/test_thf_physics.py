import numpy as np
import pytest
from scipy.linalg import expm
from scipy.special import eval_genlaguerre

from qlsgym.molecules import thf
from qlsgym.physics.propagate import transfer_matrix
from qlsgym.physics.thermal import boltzmann
from qlsgym.spec import TWO_PI

H_PLANCK = 6.62607015e-34
K_B = 1.380649e-23


@pytest.fixture(scope="module")
def mol():
    return thf.build()


def _raw_tables():
    d = thf.data_dir()
    lv = [ln.split() for ln in open(d / thf.LEVELS_FILE) if ln.strip() and not ln.startswith("#")]
    jp = [(int(r[1]), r[3]) for r in lv]
    mf = np.array([float(r[5]) for r in lv])
    e = np.array([float(r[7]) for r in lv]) * 1e3 * TWO_PI
    rb = np.array([[float(x) for x in ln.split()[:4]] for ln in open(d / thf.RABI_FILE)
                   if ln.strip() and not ln.startswith("#")])
    return jp, mf, e, rb[:, 0].astype(int), rb[:, 1].astype(int), rb[:, 2] + 1j * rb[:, 3]


def _sideband(eta: float, n: int) -> complex:
    return 1j * eta * np.exp(-eta ** 2 / 2) * eval_genlaguerre(n, 1, eta ** 2) / np.sqrt(n + 1)


def _reference_columns(mol, key, omega, sigma, taus, n_nu):
    jp, mf, e, ci, cf, om = _raw_tables()
    states = np.array([k for k, x in enumerate(jp) if x == key])
    loc = {int(g): k for k, g in enumerate(states)}
    w, nu_f, eta = mol.window, mol.trap.nu_f, mol.trap.eta
    inside = np.isin(ci, states) & np.isin(cf, states)
    d = e[cf] - e[ci]
    w_res = nu_f + d if sigma == "+" else nu_f - d
    near = np.maximum(0.0, np.maximum(w.omega_min - w_res, w_res - w.omega_max)) <= w.rwa_cutoff
    sel = np.where(inside & near)[0]
    m = states.size
    dim = n_nu * m
    e_loc = e[states] - e[states].mean()
    h = np.zeros((dim, dim), dtype=complex)
    for nu in range(n_nu):
        h[np.arange(m) + nu * m, np.arange(m) + nu * m] = e_loc + nu * (nu_f - omega)
    for k in sel:
        src, dst = (loc[ci[k]], loc[cf[k]]) if sigma == "+" else (loc[cf[k]], loc[ci[k]])
        for nu in range(n_nu - 1):
            g = 0.5 * om[k] * _sideband(eta, nu)
            h[dst + (nu + 1) * m, src + nu * m] += g
            h[src + nu * m, dst + (nu + 1) * m] += np.conj(g)
    out = np.empty((len(taus), 2 * m, m))
    for t, tau in enumerate(taus):
        pop = np.abs(expm(-1j * h * tau)[:, :m]) ** 2
        out[t, :m] = pop[:m]
        out[t, m:] = pop[m:].reshape(n_nu - 1, m, m).sum(0)
    return out


def _drives(mol, block, sigma, n_random, rng):
    from qlsgym.physics.spectrum import resonant_frequencies
    w = mol.window
    res = resonant_frequencies(mol, block, sigma)
    res = res[(res >= w.omega_min) & (res <= w.omega_max)]
    return np.r_[res, rng.uniform(w.omega_min, w.omega_max, n_random)]


def _check_propagation(mol, blocks, n_random, tau_stride):
    rng = np.random.default_rng(0)
    tau_idx = np.arange(0, mol.window.n_tau, tau_stride)
    taus = mol.tau_grid()[tau_idx]
    worst = 0.0
    for b in blocks:
        for sigma in ("+", "-"):
            for omega in _drives(mol, b, sigma, n_random, rng):
                got = transfer_matrix(mol, b, omega, sigma, tau_indices=tau_idx)
                want = _reference_columns(mol, mol.blocks[b].key, omega, sigma, taus, mol.trap.n_nu)
                worst = max(worst, float(np.abs(got - want).max()))
    return worst


def test_propagation_matches_independent_expm(mol):
    assert _check_propagation(mol, blocks=(13, 8, 0), n_random=2, tau_stride=40) < 1e-8


@pytest.mark.slow
def test_propagation_matches_independent_expm_every_block(mol):
    assert _check_propagation(mol, blocks=range(len(mol.blocks)), n_random=4, tau_stride=20) < 1e-8


def test_reference_reproduces_the_pi_pulse_of_the_reference_rung(mol):
    from qlsgym.env.actions import pi_time_ms
    labels = mol.system.levels
    i = labels.index((1, 1.5, "+", 0.5, 0))
    f = labels.index((1, 1.5, "+", 1.5, 0))
    b = int(mol.system.block_of_state[i])
    states = mol.blocks[b].states
    li, lf = int(np.where(states == i)[0][0]), int(np.where(states == f)[0][0])
    omega = mol.trap.nu_f + mol.system.energies[f] - mol.system.energies[i]
    tau = pi_time_ms(mol, TWO_PI * 3.0)
    cols = _reference_columns(mol, mol.blocks[b].key, omega, "+", [tau], mol.trap.n_nu)[0]
    m = states.size
    assert cols[m + lf, li] > 0.99


def _cutoff_error(blocks, n_random, other_n_nu=16):
    mb, mo = thf.build(n_primitives=0), thf.build(n_nu=other_n_nu, n_primitives=0)
    rng = np.random.default_rng(1)
    worst = {}
    for b in blocks:
        d = 0.0
        for sigma in ("+", "-"):
            for omega in _drives(mb, b, sigma, n_random, rng):
                d = max(d, float(np.abs(transfer_matrix(mb, b, omega, sigma)
                                        - transfer_matrix(mo, b, omega, sigma)).max()))
        worst[b] = d
    return worst


@pytest.mark.slow
def test_motional_cutoff_converged_on_the_hardest_block():
    assert _cutoff_error([0], n_random=2)[0] < 2e-3


@pytest.mark.slow
def test_motional_cutoff_converged_on_every_block():
    bound = {7: 2e-3, 6: 1.5e-3, 5: 6e-4, 4: 5e-5, 3: 1e-9, 2: 1e-9, 1: 1e-9}
    mol = thf.build(n_primitives=0)
    worst = _cutoff_error(range(len(mol.blocks)), n_random=4)
    for b, d in worst.items():
        assert d < bound[mol.blocks[b].key[0]], (b, mol.blocks[b].key, d)


def _library_cutoff_error(n_nu_ref=16):
    from qlsgym.env.actions import ActionLibrary
    from qlsgym.env.cache import build_action_tables

    tables = {}
    for n_nu in (thf.N_NU, n_nu_ref):
        m = thf.build(n_nu=n_nu)
        tables[n_nu] = build_action_tables(m, ActionLibrary.physics_subset(m), dtype=np.float64, write=False)
    mb = thf.build()
    p = boltzmann(mb)
    entry, thermal = [], []
    for b, (a, c) in enumerate(zip(tables[thf.N_NU].blocks, tables[n_nu_ref].blocks)):
        M = mb.blocks[b].n_states
        pb = p[mb.blocks[b].states] / p[mb.blocks[b].states].sum()
        entry.append(float(np.abs(a - c).max()))
        thermal.append(float(np.abs(a[:, M:, :].sum(1) @ pb - c[:, M:, :].sum(1) @ pb).max()))
    prim = max(float(np.abs(x.table - y.table).max())
               for x, y in zip(tables[thf.N_NU].primitives, tables[n_nu_ref].primitives))
    return entry, thermal, prim


@pytest.mark.slow
def test_motional_cutoff_converged_on_the_action_library():
    entry, thermal, prim = _library_cutoff_error()
    assert max(entry) < 3e-4, entry
    assert max(thermal) < 1e-6, thermal
    assert prim < 1e-11


def test_boltzmann_prior_matches_SI_calculation(mol):
    e_mhz = np.array([float(ln.split()[7]) for ln in open(thf.data_dir() / thf.LEVELS_FILE)
                      if ln.strip() and not ln.startswith("#")])
    x = -H_PLANCK * (e_mhz - e_mhz.min()) * 1e6 / (K_B * mol.task.temperature_k)
    want = np.exp(x) / np.exp(x).sum()
    np.testing.assert_allclose(boltzmann(mol), want, rtol=1e-9, atol=0)
    assert boltzmann(mol).sum() == pytest.approx(1.0, abs=1e-14)
