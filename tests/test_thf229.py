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
    assert max(b.n_states for b in mol.blocks) == 108 and len(mol.blocks) == 66
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
        assert {lv[g][1] for g in b.states} == set(b.key[2])


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
    a, c = thf229.build(n_nu=thf229.N_NU, n_primitives=0), thf229.build(n_nu=thf229.N_NU + 4, n_primitives=0)
    w = a.window
    for b in range(len(a.blocks)):
        r = resonant_frequencies(a, b, "+")
        r = r[(r >= w.omega_min) & (r <= w.omega_max)]
        for om in r[::max(1, r.size // 4)][:4]:
            assert np.abs(transfer_matrix(a, b, om, "+") - transfer_matrix(c, b, om, "+")).max() < 1e-8, b


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


def _raw_tables():
    from qlsgym.spec import TWO_PI as _TP
    d = thf229.data_dir()
    lv = [ln.split() for ln in open(d / thf229.LEVELS_FILE) if ln.strip() and not ln.startswith("#")]
    e = np.array([float(r[8]) for r in lv]) * 1e3 * _TP
    rb = np.array([[float(x) for x in ln.split()[:4]] for ln in open(d / thf229.RABI_FILE)
                   if ln.strip() and not ln.startswith("#")])
    return e, rb[:, 0].astype(int), rb[:, 1].astype(int), rb[:, 2] + 1j * rb[:, 3]


def _reference_columns(mol, states, omega, sigma, taus, n_nu):
    from scipy.linalg import expm
    from scipy.special import eval_genlaguerre
    e, ci, cf, om = _raw_tables()
    loc = {int(g): k for k, g in enumerate(states)}
    w, nu_f, eta = mol.window, mol.trap.nu_f, mol.trap.eta
    inside = np.isin(ci, states) & np.isin(cf, states)
    d = e[cf] - e[ci]
    w_res = nu_f + d if sigma == "+" else nu_f - d
    near = np.maximum(0.0, np.maximum(w.omega_min - w_res, w_res - w.omega_max)) <= w.rwa_cutoff
    m = states.size
    h = np.zeros((n_nu * m, n_nu * m), dtype=complex)
    e_loc = e[states] - e[states].mean()
    for nu in range(n_nu):
        h[np.arange(m) + nu * m, np.arange(m) + nu * m] = e_loc + nu * (nu_f - omega)
    for k in np.where(inside & near)[0]:
        src, dst = (loc[ci[k]], loc[cf[k]]) if sigma == "+" else (loc[cf[k]], loc[ci[k]])
        for nu in range(n_nu - 1):
            g = 0.5 * om[k] * 1j * eta * np.exp(-eta ** 2 / 2) * eval_genlaguerre(nu, 1, eta ** 2) / np.sqrt(nu + 1)
            h[dst + (nu + 1) * m, src + nu * m] += g
            h[src + nu * m, dst + (nu + 1) * m] += np.conj(g)
    out = np.empty((len(taus), 2 * m, m))
    for t, tau in enumerate(taus):
        pop = np.abs(expm(-1j * h * tau)[:, :m]) ** 2
        out[t, :m] = pop[:m]
        out[t, m:] = pop[m:].reshape(n_nu - 1, m, m).sum(0)
    return out


@pytest.mark.slow
def test_propagation_matches_independent_expm(mol):
    w = mol.window
    tau_idx = np.array([20, 90, 199])
    taus = mol.tau_grid()[tau_idx]
    merged = next(b.index for b in mol.blocks if len(b.key[2]) > 1)
    small = min(range(len(mol.blocks)), key=lambda k: (mol.blocks[k].n_states < 8, mol.blocks[k].n_states))
    worst = 0.0
    for b in (small, merged):
        r = resonant_frequencies(mol, b, "+")
        r = r[(r >= w.omega_min) & (r <= w.omega_max)]
        for sigma in "+-":
            for om in (r[0] if sigma == "+" else 2 * mol.trap.nu_f - r[0], 0.5 * (w.omega_min + w.omega_max)):
                got = transfer_matrix(mol, b, om, sigma, tau_indices=tau_idx)
                want = _reference_columns(mol, mol.blocks[b].states, om, sigma, taus, mol.trap.n_nu)
                worst = max(worst, float(np.abs(got - want).max()))
    assert worst < 1e-8, worst


@pytest.mark.slow
def test_every_primitive_moves_its_source_into_its_target(mol):
    from qlsgym.env.actions import snap_tau
    from qlsgym.physics.engines import ExactEngine
    assert len(mol.primitives) == thf229.CONFIG["primitives"]["n"]
    worst = 1.0
    for p in mol.primitives:
        k = snap_tau(mol, p.tau_ms)
        eng = ExactEngine(mol, tau_indices=np.array([k]))
        p0 = np.zeros(mol.n_states)
        p0[p.source] = 1.0
        b0, b1 = eng.branches_all_tau(p0, p.omega, p.sigma)
        worst = min(worst, float(b1[0, p.target]))
    assert worst > 0.9, worst


def test_boltzmann_prior_matches_SI_calculation(mol):
    from qlsgym.physics.thermal import boltzmann
    e_mhz = np.array([float(ln.split()[8]) for ln in open(thf229.data_dir() / thf229.LEVELS_FILE)
                      if ln.strip() and not ln.startswith("#")])
    x = -6.62607015e-34 * (e_mhz - e_mhz.min()) * 1e6 / (1.380649e-23 * mol.task.temperature_k)
    want = np.exp(x) / np.exp(x).sum()
    np.testing.assert_allclose(boltzmann(mol), want, rtol=1e-9, atol=0)
