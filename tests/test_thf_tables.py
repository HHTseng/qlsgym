"""The ThF+ table generator (qlsgym.molecules.thf_tables): coupling model A checked against the
claims in its docstring, and the shipped tables in data/thf against a fresh build.

Needs heff on PYTHONPATH (source scripts/env.sh); skipped otherwise.
"""
from dataclasses import replace

import numpy as np
import pytest

heff = pytest.importorskip("heff", reason="heff not on PYTHONPATH: source scripts/env.sh")

from qlsgym.molecules import thf, thf_tables  # noqa: E402
from qlsgym.spec import TWO_PI  # noqa: E402

J_SMALL = 2      # enough structure to exercise every claim, fast


@pytest.fixture(scope="module")
def levels():
    return thf_tables.build_levels(j_max=J_SMALL)


@pytest.fixture(scope="module")
def amp(levels):
    """Two-photon amplitudes in the eigenbasis, keyed by the second beam's Jones vector."""
    I = heff.thf_spec(J_max=J_SMALL).I
    a = {}
    for name2 in ("sigma+", "sigma-"):
        a_ket = thf_tables.two_photon_amplitudes(levels.kets, I, thf_tables.JONES["pi"],
                                                 thf_tables.JONES[name2], J_SMALL + 1)
        a[name2] = levels.vectors.conj().T @ a_ket @ levels.vectors
    return a


def test_levels_match_heff_tutorial():
    """J = 1 at zero field: Omega-doublet 5.290 MHz, hyperfine (3/4)|A| - (3/2)c_I = 15.0476 MHz
    (heff tutorial section 4); e (parity +(-1)^J) lies below f by omega_ef J(J+1)/2 at every J.
    """
    levels = thf_tables.build_levels(j_max=J_SMALL, b_gauss=0.0)
    e = levels.energies_MHz - levels.energies_MHz.min()
    lab = levels.labels
    j1 = [(l, x) for l, x in zip(lab, e) if l[0] == 1]
    f32p = [x for l, x in j1 if l[1] == 1.5 and l[2] == "+"]
    f32m = [x for l, x in j1 if l[1] == 1.5 and l[2] == "-"]
    f12m = [x for l, x in j1 if l[1] == 0.5 and l[2] == "-"]
    assert np.mean(f32p) - np.mean(f32m) == pytest.approx(5.290, abs=2e-3)   # parity - is e at J = 1
    assert np.mean(f12m) - np.mean(f32m) == pytest.approx(15.0476, abs=5e-3)
    assert len(f32m) == 4 and len(f12m) == 2
    for l, ef in zip(levels.labels, levels.ef):          # heff's e/f labels agree with the parity rule
        e_parity = "+" if l[0] % 2 == 0 else "-"
        assert ef == ("e" if l[2] == e_parity else "f"), (l, ef)
    for J in range(1, J_SMALL + 1):
        mean = {p: np.mean([x for l, x in zip(lab, e) if l[0] == J and l[1] == J + 0.5 and l[2] == p])
                for p in ("+", "-")}
        lower, upper = ("+", "-") if J % 2 == 0 else ("-", "+")
        assert mean[upper] - mean[lower] == pytest.approx(5.29 * J * (J + 1) / 2.0, abs=3e-3)


def test_parity_dependent_zeeman_is_heffs_G_tensor():
    """heff's G = diag(G_xx, G_yy, G_zz) carries the e/f g-factor difference:

        g_J(e/f) = -G_perp - (G_zz - G_perp)/[J(J+1)] +- G_Delta,   G_Delta = (G_xx - G_yy)/2

    so at J = 1, F = 3/2 (F-projection factor 2/3) g_f - g_e = -2 G_Delta * 2/3 = +2.741e-4, with e
    the larger |g|, and the mean is the measured |g_F| = 0.0149. Read from the shipped table, the
    numbers the rest of qlsgym runs on.
    """
    ps = heff.thf_v1()
    g_delta = 0.5 * (ps.value("G_xx") - ps.value("G_yy"))
    assert g_delta == pytest.approx(-2.0558e-4, rel=1e-4)

    mu_b = 1.3996245                                   # MHz/G
    lab, energies = thf.read_energy_levels(thf.data_dir() / thf.LEVELS_FILE)
    e = energies / TWO_PI / 1e3                        # rad/ms -> MHz

    def g_of(parity):
        ms = sorted((l[3], e[n]) for n, l in enumerate(lab) if l[0] == 1 and l[1] == 1.5 and l[2] == parity)
        return -float(np.mean(np.diff([x[1] for x in ms]))) / (mu_b * thf.B_FIELD_GAUSS)

    g_e, g_f = g_of("-"), g_of("+")                    # at J = 1, e is parity -
    assert g_e == pytest.approx(-0.015039, abs=2e-6)
    assert g_f == pytest.approx(-0.014765, abs=2e-6)
    assert 0.5 * (g_e + g_f) == pytest.approx(-0.014902, abs=2e-6)
    assert g_f - g_e == pytest.approx(-2.0 * g_delta * (2.0 / 3.0), rel=2e-3)


def test_parity_is_exact_and_operator_is_parity_even(levels, amp):
    par = np.array([l[2] for l in levels.labels])
    for a in amp.values():
        assert np.max(np.abs(a[par[:, None] != par[None, :]])) < 1e-10 * np.max(np.abs(a))


def test_channels_are_pure_delta_mF(levels, amp):
    mf = np.array([l[3] for l in levels.labels])
    dm = mf[:, None] - mf[None, :]                     # dm[f, i]
    for name2, sign in (("sigma-", +1.0), ("sigma+", -1.0)):
        a = amp[name2]
        big = np.abs(a) > 1e-9 * np.max(np.abs(a))
        assert set(np.unique(dm[big]).tolist()) == {sign}


def test_sigma_minus_is_the_exact_reverse_of_sigma_plus(amp):
    """|A_-| = |A_+|^T, which qlsgym.physics.hamiltonian builds the sigma- channel from. True because
    the rank-1 part vanishes for an Omega-diagonal intermediate; an Omega = 0 one would break it.
    """
    a_plus, a_minus = amp["sigma-"], amp["sigma+"]     # Delta m_F = +1 and -1
    assert np.max(np.abs(np.abs(a_minus) - np.abs(a_plus).T)) < 1e-12 * np.max(np.abs(a_plus))


def _sigma_plus_eigen(levels):
    I = heff.thf_spec(J_max=J_SMALL).I
    a_ket = thf_tables.two_photon_amplitudes(levels.kets, I, thf_tables.JONES["pi"],
                                             thf_tables.JONES["sigma-"], J_SMALL + 1)
    return levels.vectors.conj().T @ a_ket @ levels.vectors


def _ef_coupling_asymmetry(levels, a):
    """max | |a[m, n]| - |a[twin(m), twin(n)]| | / max |a| over Omega-doublet twins."""
    idx = {l: n for n, l in enumerate(levels.labels)}
    worst = 0.0
    for (J, F, p, mF, xi), n in idx.items():
        q = "-" if p == "+" else "+"
        twin = idx[(J, F, q, mF, xi)]
        for (J2, F2, p2, mF2, xi2), m in idx.items():
            if p2 != p:
                continue
            worst = max(worst, abs(abs(a[m, n]) - abs(a[idx[(J2, F2, q, mF2, xi2)], twin])))
    return worst / np.max(np.abs(a))


def test_parity_partners_couple_alike_except_through_G_Delta(levels, amp):
    """Twins (J, +) and (J, -) see the same |Omega| in the ket basis. In the eigenbasis they differ
    by ~1e-3 of the largest coupling (measured 9.7e-4 at J_max = 2), because G_Delta is a
    Delta-Omega = +-2 operator that changes the Delta F = +-1 Zeeman admixture with opposite sign on
    e and f. Control: with G_xx = G_yy (G_Delta = 0, all else equal) the difference collapses.
    """
    rel = _ef_coupling_asymmetry(levels, amp["sigma-"])
    assert 1e-4 < rel < 2e-3, rel

    ps = heff.thf_v1()
    g_perp = 0.5 * (ps.value("G_xx") + ps.value("G_yy"))
    no_g_delta = ps.with_(G_xx=replace(ps.params["G_xx"], value=g_perp),
                          G_yy=replace(ps.params["G_yy"], value=g_perp))
    assert no_g_delta.value("G_xx") == no_g_delta.value("G_yy")
    ctrl = thf_tables.build_levels(j_max=J_SMALL, pset=no_g_delta)
    assert _ef_coupling_asymmetry(ctrl, _sigma_plus_eigen(ctrl)) < 1e-6


def test_no_rank_1_component_in_model_A(levels):
    """Swapping the two beams transposes the operator iff the two legs commute on X, i.e. iff the
    antisymmetric rank-1 (pi x sigma) part is zero.
    """
    I = heff.thf_spec(J_max=J_SMALL).I
    z, s = thf_tables.JONES["pi"], thf_tables.JONES["sigma+"]
    t_zs = thf_tables.two_photon_amplitudes(levels.kets, I, z, s, J_SMALL + 1)
    t_sz = thf_tables.two_photon_amplitudes(levels.kets, I, s, z, J_SMALL + 1)
    assert np.max(np.abs(t_sz - t_zs.conj().T)) < 1e-12 * np.max(np.abs(t_zs))


def test_delta_mF_amplitudes_are_pure_rank_2(levels):
    """Wigner-Eckart in the ket basis: for fixed (J, Omega, F) -> (J', Omega, F') the Delta m_F = +1
    amplitudes follow the rank-2 3j (F' 2 F; -m-1 1 m) with an m-independent constant -- the algebraic
    content of "no rank-1 part", and why the -1/2 -> +1/2 rung of every ladder vanishes.
    """
    from heff.wigner import w3j
    I = heff.thf_spec(J_max=J_SMALL).I
    a = thf_tables.two_photon_amplitudes(levels.kets, I, thf_tables.JONES["pi"], thf_tables.JONES["sigma-"],
                                         J_SMALL + 1)
    scale = np.max(np.abs(a))
    groups: dict = {}
    for n, k in enumerate(levels.kets):
        groups.setdefault((float(k["J"]), float(k["Om"]), float(k["F"])), {})[float(k["mF"])] = n
    checked = 0
    for (Ji, Oi, Fi), gi in groups.items():
        for (Jf, Of, Ff), gf in groups.items():
            if Oi != Of:
                continue
            ratios = []
            for mi, ni in gi.items():
                nf = gf.get(mi + 1.0)
                if nf is None:
                    continue
                w = (-1.0) ** (Ff - (mi + 1.0)) * w3j(Ff, 2, Fi, -(mi + 1.0), 1.0, mi)
                if abs(w) < 1e-12:
                    assert abs(a[nf, ni]) < 1e-12 * scale
                else:
                    ratios.append(a[nf, ni] / w)
            if len(ratios) > 1:
                assert np.max(np.abs(np.array(ratios) - ratios[0])) < 1e-12 * scale
                checked += 1
    assert checked > 5


@pytest.mark.slow
def test_shipped_tables_equal_a_fresh_build(tmp_path):
    """data/thf must be byte-identical to a fresh build at the current settings and heff revision,
    except line 1 (timestamp). Fails after any change to heff, its parameters or the settings here:
    re-run scripts/build_thf_tables.py (and retrain every surrogate).
    """
    levels, cpl = thf_tables.build_tables()
    fresh = thf_tables.table_files(tmp_path)
    thf_tables.write_tables(levels, cpl, *fresh)
    shipped = thf_tables.table_files(thf.data_dir())
    for got, want in zip(fresh, shipped):
        with open(got) as a, open(want) as b:
            assert a.read().splitlines()[1:] == b.read().splitlines()[1:], want
