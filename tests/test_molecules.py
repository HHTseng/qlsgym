"""Structure of the two real molecule packs."""
import re

import numpy as np
import pytest

from qlsgym import load_molecule
from qlsgym.spec import Molecule, TWO_PI
from qlsgym.physics.thermal import boltzmann
from qlsgym.physics.engines import ExactEngine

# mu_B / h in MHz/G, only used to turn a tabulated Zeeman spacing back into a
# g-factor for the ThF+ Omega-doublet test below.
MU_B_MHZ_PER_G = 1.3996245


@pytest.fixture(scope="module")
def h3o():
    return load_molecule("h3o")


@pytest.fixture(scope="module")
def thf():
    return load_molecule("thf")


def test_h3o_structure(h3o):
    assert isinstance(h3o, Molecule) and h3o.name == "h3o"
    assert h3o.n_states == 444 and h3o.system.n_blocks == 10
    assert sorted(b.n_states for b in h3o.blocks) == sorted([64, 64, 60, 48, 48, 42, 42, 40, 18, 18])
    assert [b.n_states for b in h3o.blocks] == sorted([b.n_states for b in h3o.blocks], reverse=True)
    for b in h3o.blocks:
        assert len(b.key) == 2 and b.key[1] in "+-" and isinstance(b.key[0], float)
        assert np.array_equal(b.states, np.sort(b.states))
        assert np.all(h3o.system.block_of_state[b.states] == b.index)
    assert h3o.trap.n_nu == 2 and abs(h3o.trap.nu_f / (2 * np.pi) - 5164.0) < 1e-9 and h3o.trap.eta == 0.09
    assert abs(h3o.window.omega_min / (2 * np.pi) - 5050) < 1e-9 and abs(h3o.window.omega_max / (2 * np.pi) - 5300) < 1e-9
    assert h3o.window.tau_max_ms == 4.0 and h3o.window.n_tau == 200 and h3o.task.temperature_k == 20.0
    assert h3o.task.max_pulses == 80 and h3o.window.carrier_bands == ()
    assert len(h3o.primitives) == 21 and all(p.sigma == "+" for p in h3o.primitives)
    for p in h3o.primitives:
        assert not h3o.window.contains(p.omega) and 0 < p.tau_ms <= 1.01 * h3o.window.tau_max_ms
        assert len(p.blocks) == 1 and h3o.system.block_of_state[p.source] == p.blocks[0]
        assert h3o.system.levels[p.source][0] == 1 and h3o.system.levels[p.target][0] in (2, 3)
    assert h3o.provenance["j_max"] == 4 and h3o.provenance["b_field_mt"] == 0.36


def test_thf_structure(thf):
    assert thf.name == "thf" and thf.n_states == 192 and thf.system.n_blocks == 12
    assert [b.n_states for b in thf.blocks] == [26, 26, 22, 22, 18, 18, 14, 14, 10, 10, 6, 6]
    for b in thf.blocks:
        J, par = b.key
        assert b.n_states == 4 * J + 2 and par in "+-"
        assert all(thf.system.levels[s][0] == J and thf.system.levels[s][2] == par for s in b.states)
    assert thf.trap.n_nu == 7 and abs(thf.trap.nu_f / (2 * np.pi) - 1112.7182) < 1e-9 and thf.trap.eta == 0.06776
    assert abs((thf.window.omega_max - thf.window.omega_min) / (2 * np.pi) - 940.0) < 1e-9
    assert thf.window.tau_max_ms == 6.0 and thf.window.n_tau == 200 and thf.task.temperature_k == 4.0
    # the two J = 1, F = 1/2 sigma+ carriers inside the window (ThF finding F5);
    # with the heff G tensor they sit at 906.3 kHz (f) and 921.7 kHz (e)
    bands = np.array(thf.window.carrier_bands) / (2 * np.pi)
    assert bands.shape == (2, 2) and np.allclose(bands.mean(1), [906.324, 921.657], atol=2e-3)
    assert len(thf.primitives) == 24
    sig = [p.sigma for p in thf.primitives]
    assert "+" in sig and "-" in sig
    for p in thf.primitives:
        assert not thf.window.contains(p.omega) and 0 < p.tau_ms <= thf.window.tau_max_ms
        assert thf.system.block_of_state[p.source] in p.blocks and thf.system.block_of_state[p.target] in p.blocks
        assert len(p.blocks) >= 2
    assert thf.provenance["coupling_model"] == "A"
    assert thf.provenance["b_field_gauss"] == 20.0 and thf.provenance["j_max"] == 6
    # D11 (hand-added Delta_g) is retired: the parity-dependent Zeeman effect
    # is now the G tensor's own G_Delta = (G_xx - G_yy) / 2.
    assert "delta_g" not in thf.provenance and "delta_g_j_scaling" not in thf.provenance
    assert thf.provenance["zeeman"] == "heff G tensor (G_xx, G_yy, G_zz)"
    assert (thf.provenance["g_xx"], thf.provenance["g_yy"], thf.provenance["g_zz"]) == \
        (8.2211e-4, 1.23327e-3, 0.046532)
    assert thf.provenance["g_delta"] == pytest.approx(-2.0558e-4, abs=1e-12)
    assert thf.provenance["doublet_ordering"].startswith("e below f")
    # parsed out of the table header, not hard-coded
    assert re.fullmatch(r"[0-9a-f]{7,40}", thf.provenance["heff_git_rev"])
    assert re.fullmatch(r"[0-9a-f]{64}", thf.provenance["heff_spec_hash"])


def test_thf_omega_doublet_ordering_and_g_tensor(thf):
    """heff 2026-09-14: e lies below f at every J, and G_Delta splits their g-factors at J = 1."""
    lv = thf.system.levels
    for J in range(1, 7):
        e_parity = "+" if J % 2 == 0 else "-"
        lo = {p: min(thf.system.energies[i] for i, l in enumerate(lv) if l[0] == J and l[2] == p)
              for p in "+-"}
        assert min(lo, key=lo.get) == e_parity, f"J={J}: lower doublet component is not e"
    # block order follows the table order, which follows energy
    assert [b.key for b in thf.blocks] == [(6, "+"), (6, "-"), (5, "-"), (5, "+"), (4, "+"), (4, "-"),
                                           (3, "-"), (3, "+"), (2, "+"), (2, "-"), (1, "-"), (1, "+")]

    def g_magnitude(J, F, parity):
        idx = [i for i, l in enumerate(lv) if l[0] == J and l[1] == F and l[2] == parity]
        e_mhz = np.sort(thf.system.energies[idx]) / (1e3 * TWO_PI)
        return float(np.diff(e_mhz).mean() / (MU_B_MHZ_PER_G * thf.provenance["b_field_gauss"]))

    g_f, g_e = g_magnitude(1, 1.5, "+"), g_magnitude(1, 1.5, "-")      # f upper, e lower
    assert g_f == pytest.approx(0.014765, abs=1e-6)
    assert g_e == pytest.approx(0.015039, abs=1e-6)
    assert g_f - g_e == pytest.approx(-2.7405e-4, abs=1e-8)            # = -(g(f) - g(e)) in magnitude


def test_fingerprints_and_boltzmann(h3o, thf):
    assert h3o.fingerprint() != thf.fingerprint()
    assert h3o.fingerprint() == load_molecule("h3o").fingerprint()
    assert load_molecule("thf", n_nu=2).fingerprint() != thf.fingerprint()
    for mol in (h3o, thf):
        p = boltzmann(mol)
        assert p.shape == (mol.n_states,) and abs(p.sum() - 1.0) < 1e-12 and np.all(p > 0)
        assert np.argmax(p) == np.argmin(mol.system.energies)
    assert boltzmann(h3o, 1.0).max() > boltzmann(h3o).max()


def test_two_molecules_do_not_interfere(h3o, thf):
    """Caches are keyed on the fingerprint: alternating molecules gives the same numbers as using
    each alone.
    """
    idx = np.array([50, 199])
    ph, pt = boltzmann(h3o), boltzmann(thf)
    wh = 0.5 * (h3o.window.omega_min + h3o.window.omega_max)
    wt = 0.5 * (thf.window.omega_min + thf.window.omega_max)
    eh, et = ExactEngine(h3o, idx), ExactEngine(thf, idx)
    a = eh.branches_all_tau(ph, wh, "+")
    b = et.branches_all_tau(pt, wt, "+")
    a2 = ExactEngine(h3o, idx).branches_all_tau(ph, wh, "+")
    b2 = ExactEngine(thf, idx).branches_all_tau(pt, wt, "+")
    assert a[0].shape == (2, 444) and b[0].shape == (2, 192)
    assert np.array_equal(a[0], a2[0]) and np.array_equal(b[1], b2[1])
    with pytest.raises(ValueError):
        et.branches_all_tau(pt, wh, "+")      # the H3O+ window is off-window for ThF+
