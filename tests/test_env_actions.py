"""Action library: grids, encode/decode bijection, stable ordering, physics subset."""
import numpy as np
import pytest

from qlsgym import load_molecule
from qlsgym.env.actions import ActionLibrary, ControlGrid, mask_carriers, resonant_frequencies, pi_time_ms
from qlsgym.spec import Action, TWO_PI


@pytest.fixture(scope="module")
def mol():
    return load_molecule("synthetic")


def test_rl_discrete_matches_paper_slots_for_200_point_grid(mol):
    from dataclasses import replace
    m = replace(mol, window=replace(mol.window, n_tau=200))
    g = ControlGrid.rl_discrete(m, n_freq=7, seed=3)
    assert g.tau_indices.tolist() == (np.arange(150, 200, 5) - 1).tolist()
    assert g.n_actions == 2 * 7 * 10
    assert np.all(np.diff(g.omegas) >= 0) and g.omegas.min() >= m.window.omega_min
    # bit-identical to the source draw (no carrier bands on this molecule)
    ref = np.sort(np.random.default_rng(3).uniform(m.window.omega_min, m.window.omega_max, 7))
    assert np.array_equal(g.omegas, ref)


def test_uniform_grid_masks_carrier_bands(mol):
    from dataclasses import replace
    w = mol.window
    lo, hi = w.omega_min + TWO_PI * 10, w.omega_min + TWO_PI * 20
    m = replace(mol, window=replace(w, carrier_bands=((lo, hi),)))
    g = ControlGrid.uniform(m, d_omega_khz=1.0)
    assert not np.any((g.omegas >= lo) & (g.omegas <= hi))
    g_all = ControlGrid.uniform(m, d_omega_khz=1.0, mask_carrier_bands=False)
    assert g_all.n_freq > g.n_freq
    assert g.tau_indices[0] == m.window.n_tau // 2 - 1 and g.tau_indices[-1] == m.window.n_tau - 1
    assert not mask_carriers(m, np.array([0.5 * (lo + hi)]))[0]


def test_grid_library_roundtrip_and_ordering(mol):
    g = ControlGrid.rl_discrete(mol, n_freq=4, n_tau_slots=3, seed=1)
    lib = ActionLibrary(mol, g)
    assert lib.n_grid == 2 * 4 * 3 and lib.n_primitives == 1 and lib.n_actions == 25
    seen = set()
    for a in range(lib.n_actions):
        act = lib.decode(a)
        assert lib.encode(act) == a
        assert lib.is_primitive(a) == (a >= lib.n_grid) == act.is_primitive
        seen.add((act.sigma, round(act.omega, 9), act.tau_index, act.primitive))
    assert len(seen) == lib.n_actions
    # sigma -> omega -> tau ordering, as in fnorepl.rl.ActionLibrary
    a = (1 * 4 + 2) * 3 + 1
    act = lib.decode(a)
    assert act == Action("-", float(g.omegas[2]), int(g.tau_indices[1]), -1)
    # primitive decodes with its own snapped duration
    p = lib.decode(lib.n_grid)
    assert p.primitive == 0 and p.omega == mol.primitives[0].omega
    assert p.tau_index == int(np.argmin(np.abs(mol.tau_grid() - mol.primitives[0].tau_ms)))
    assert list(lib.actions)[-1] == p
    with pytest.raises(IndexError):
        lib.decode(lib.n_actions)
    with pytest.raises(KeyError):
        lib.encode(Action("+", float(g.omegas[0]) + 1e-3, 0))


def test_ordering_is_stable_across_constructions(mol):
    a = ActionLibrary(mol, ControlGrid.rl_discrete(mol, n_freq=5, n_tau_slots=2))
    b = ActionLibrary(mol, ControlGrid.rl_discrete(mol, n_freq=5, n_tau_slots=2))
    assert a.tag() == b.tag()
    assert [x for x in a.actions] == [x for x in b.actions]
    c = ActionLibrary(mol, ControlGrid.rl_discrete(mol, n_freq=5, n_tau_slots=2, seed=9))
    assert c.tag() != a.tag()
    d = ActionLibrary(mol, ControlGrid.rl_discrete(mol, n_freq=5, n_tau_slots=2), include_primitives=False)
    assert d.tag() != a.tag() and d.n_actions == a.n_grid


def test_off_window_grid_action_is_rejected(mol):
    with pytest.raises(ValueError):
        ActionLibrary.from_actions(mol, [("+", mol.window.omega_max + 1.0, 0)])
    with pytest.raises(ValueError):
        ActionLibrary.from_actions(mol, [("+", mol.window.omega_min, mol.window.n_tau)])


def test_drives_group_actions_by_sigma_omega(mol):
    g = ControlGrid.rl_discrete(mol, n_freq=3, n_tau_slots=2)
    lib = ActionLibrary(mol, g)
    drives = lib.drives()
    assert len(drives) == 6
    for sg, w, a_idx, t_idx in drives:
        for a, t in zip(a_idx, t_idx):
            act = lib.decode(a)
            assert act.sigma == sg and act.omega == w and act.tau_index == t


def test_physics_subset_synthetic(mol):
    lib = ActionLibrary.physics_subset(mol)           # every pi-time exceeds tau_max here
    assert lib.n_grid == 0 and lib.n_primitives == 1
    lib = ActionLibrary.physics_subset(mol, clip_tau=True)
    n_couplings = sum(b.omega.size for b in mol.blocks)
    assert lib.n_grid == 2 * n_couplings
    src, tgt = lib.meta["sources"], lib.meta["targets"]
    for a in range(0, lib.n_grid, 2):
        plus, minus = lib.decode(a), lib.decode(a + 1)
        assert plus.sigma == "+" and minus.sigma == "-"
        assert minus.omega == pytest.approx(mol.sideband_mirror(plus.omega))
        assert plus.tau_index == minus.tau_index == mol.window.n_tau - 1
        assert (src[a], tgt[a]) == (tgt[a + 1], src[a + 1])
        assert mol.system.block_of_state[src[a]] == lib.meta["blocks"][a]
    # resonance = nu_f + (E_f - E_i)
    b0 = mol.blocks[0]
    w = resonant_frequencies(mol, 0, "+")
    e = mol.system.energies
    assert np.allclose(w, mol.trap.nu_f + e[b0.states[b0.f_local]] - e[b0.states[b0.i_local]])
    assert pi_time_ms(mol, 2.0) == pytest.approx(np.pi / (mol.trap.eta * np.exp(-mol.trap.eta ** 2 / 2) * 2.0))


@pytest.mark.integration
def test_physics_subset_thf_matches_baseline_script():
    """The ported action set equals baseline_protocol.build_actions on ThF+, in the SAME ORDER."""
    pytest.importorskip("qlsgym.physics.engines")
    thf_sys = pytest.importorskip("thffno.system")
    import os, sys
    root = os.environ.get("THFFNO_ROOT")
    if not root:
        pytest.skip("THFFNO_ROOT not set (source scripts/env.sh)")
    sys.path.insert(0, os.path.join(root, "scripts"))
    from baseline_protocol import build_actions
    from thffno.units import tau_grid
    mol = load_molecule("thf")
    ref = build_actions(thf_sys.default_system(), tau_grid())
    lib = ActionLibrary.physics_subset(mol)

    assert np.allclose(mol.tau_grid(), tau_grid()), "tau grids diverged; port no longer comparable"
    ref_ordered = [(a[0], round(a[1], 6), int(a[2])) for v in ref.values() for a in v]
    got_ordered = [(str(s), round(float(w), 6), int(t))
                   for s, w, t in zip(lib.sigmas, lib.omegas, lib.tau_indices)]
    assert got_ordered == ref_ordered, (
        f"action order/content diverged: {len(got_ordered)} got vs {len(ref_ordered)} ref")


@pytest.mark.parametrize("name,d_omega_khz", [("synthetic", 1.0), ("synthetic", 100.0),
                                              ("h3o", 1.0), ("thf", 1.0)])
def test_uniform_grid_stays_inside_the_window(name, d_omega_khz):
    """A window width that is an exact multiple of the step once pushed the last point out."""
    mol = load_molecule(name)
    grid = ControlGrid.uniform(mol, d_omega_khz)
    assert np.all(grid.omegas >= mol.window.omega_min)
    assert np.all(grid.omegas <= mol.window.omega_max)
    ActionLibrary(mol, grid)
