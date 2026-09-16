"""Eq. 25 score (qlsgym.policies.score): scalar/batched consistency and numeric parity against
thffno.planner.score_batch on random inputs.
"""
import numpy as np
import pytest

from qlsgym.policies.score import ScoreConfig, score, score_batch, score_terms


def _random_branches(rng, n=12):
    p_in = rng.dirichlet(np.full(n, 0.5))
    # unnormalised branches that still conserve total probability
    split = rng.uniform(0.05, 0.95)
    p0 = rng.dirichlet(np.full(n, 0.5)) * split
    p1 = rng.dirichlet(np.full(n, 0.5)) * (1 - split)
    return p_in, p0, p1


@pytest.mark.parametrize("br_mode", ["expected", "best", "success", "progress"])
def test_score_batch_matches_scalar_score(br_mode):
    rng = np.random.default_rng(0)
    cfg = ScoreConfig(br_mode=br_mode)
    for _ in range(20):
        p_in, p0, p1 = _random_branches(rng)
        s_scalar = score(p_in, p0, p1, cfg)
        s_batch = float(score_batch(p_in, p0, p1, cfg))
        assert s_scalar == pytest.approx(s_batch, abs=1e-10)
        t_scalar = score_terms(p_in, p0, p1, cfg)
        assert all(np.isfinite(t_scalar))


def test_score_batch_vectorises_over_leading_axes():
    rng = np.random.default_rng(1)
    n = 10
    p_in = rng.dirichlet(np.full(n, 0.5))
    cands = [_random_branches(rng, n) for _ in range(5)]
    p0 = np.stack([c[1] for c in cands])
    p1 = np.stack([c[2] for c in cands])
    cfg = ScoreConfig()
    batched = score_batch(p_in, p0, p1, cfg)
    assert batched.shape == (5,)
    for k in range(5):
        assert batched[k] == pytest.approx(score(p_in, p0[k], p1[k], cfg))


def test_unknown_br_mode_raises():
    rng = np.random.default_rng(2)
    p_in, p0, p1 = _random_branches(rng)
    with pytest.raises(ValueError):
        score_batch(p_in, p0, p1, ScoreConfig(br_mode="bogus"))


@pytest.mark.parametrize("br_mode", ["expected", "best", "success", "progress"])
def test_parity_with_thffno_planner_score_batch(br_mode):
    """qlsgym.policies.score is a verbatim port of thffno.planner's Eq. 25."""
    thf_planner = pytest.importorskip("thffno.planner")
    rng = np.random.default_rng(3)
    n = 40
    ref_cfg = thf_planner.ScoreConfig(br_mode=br_mode)
    cfg = ScoreConfig(br_mode=br_mode, p_target=ref_cfg.p_target,
                      success_width=ref_cfg.success_width)
    for _ in range(10):
        p_in, p0, p1 = _random_branches(rng, n)
        got = score_batch(p_in, p0, p1, cfg)
        want = thf_planner.score_batch(p_in, p0, p1, ref_cfg)
        assert float(got) == pytest.approx(float(want), abs=1e-10)
