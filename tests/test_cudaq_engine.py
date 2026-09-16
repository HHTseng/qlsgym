"""CudaqEngine: everything but the ODE solve runs on CPU; the solve itself needs a GPU."""
import numpy as np
import pytest

from qlsgym import load_molecule
from qlsgym.benchmark.protocol import engine_provenance
from qlsgym.physics import cudaq_engine as ce
from qlsgym.physics.cudaq_engine import (CudaqEngine, columns_from_amplitudes, dynamics_available,
                                         make_drive, rk_steps, schedule_and_rows, step_size)
from qlsgym.physics.engines import ExactEngine, engine_choice, select_engine
from qlsgym.physics.hamiltonian import rwa_sectors
from qlsgym.physics.propagate import propagator_series
from qlsgym.spec import check_branches

TAU_IDX = np.array([0, 3, 7, 15])


@pytest.fixture(scope="module")
def mol():
    return load_molecule("synthetic", n_nu=3)


class ExactEvolve(CudaqEngine):
    """CudaqEngine with cudaq.evolve replaced by the exact propagator, so that the schedule,
    the lumping and the tau gathering can be pinned without a GPU.
    """

    def _evolve(self, drive, omega, sched, dt):
        u = propagator_series(drive.dense(omega), sched)
        return np.moveaxis(u[:, :, :drive.n_states], -1, 0)


@pytest.fixture
def stub(monkeypatch, mol):
    monkeypatch.setattr(ce, "dynamics_available", lambda *a, **k: (True, "stubbed"))
    return ExactEvolve(mol, TAU_IDX)


# the Hamiltonian handed to CUDA-Q


def test_affine_split_reproduces_the_block_operator(mol):
    """H(omega) = h_ref + (omega - w_ref) grad must be the operator ExactEngine propagates."""
    for b in mol.blocks:
        for sigma in ("+", "-"):
            op = ce._block_operator(mol, b, sigma, mol.trap.nu_f)
            drive = make_drive(mol, b, sigma, mol.trap.nu_f)
            for w in np.linspace(mol.window.omega_min, mol.window.omega_max, 5):
                assert np.abs(drive.dense(w) - op.build(float(w))).max() < 1e-9


def test_stubbed_engine_matches_exact_engine_on_h3o(monkeypatch):
    """The real molecule, in-window and at an off-window primitive (RWA sectors)."""
    monkeypatch.setattr(ce, "dynamics_available", lambda *a, **k: (True, "stubbed"))
    mol = load_molecule("h3o")
    idx = np.array([0, 40, 199])
    stub, exact = ExactEvolve(mol, idx), ExactEngine(mol, idx)
    pr = mol.primitives[0]
    assert rwa_sectors(mol, pr.sigma, pr.omega)
    p = np.full(mol.n_states, 1.0 / mol.n_states)
    for w, sigma in [(0.5 * (mol.window.omega_min + mol.window.omega_max), "+"),
                     (pr.omega, pr.sigma)]:
        got, want = stub.branches_all_tau(p, w, sigma), exact.branches_all_tau(p, w, sigma)
        assert np.abs(got[0] - want[0]).max() < 1e-12
        assert np.abs(got[1] - want[1]).max() < 1e-12


# schedule, step size, norm check


def test_schedule_starts_at_zero_and_rows_point_back(mol):
    taus = mol.tau_grid()[[7, 3, 15, 3]]
    sched, rows = schedule_and_rows(taus)
    assert sched[0] == 0.0 and np.all(np.diff(sched) > 0)
    assert np.array_equal(sched[rows], taus)


def test_step_size_is_capped_by_the_hamiltonian():
    small = np.diag([1.0, -1.0]).astype(complex)
    assert step_size(small, 3e-5, 0.05) == 3e-5
    big = np.diag([1e5, -1e5]).astype(complex)
    assert step_size(big, 3e-5, 0.05) == pytest.approx(0.05 / 1e5)
    assert rk_steps(np.array([0.0, 1.0]), 0.25) == 4


def test_lost_norm_raises_instead_of_returning_a_wrong_answer():
    psi = np.zeros((2, 3, 4), dtype=complex)
    psi[0, :, 0] = 1.0
    psi[1, :, 1] = 1.0
    t = columns_from_amplitudes(psi, 2)
    assert t.shape == (3, 4, 2) and np.allclose(t.sum(-2), 1.0)
    with pytest.raises(RuntimeError, match="lost"):
        columns_from_amplitudes(0.9 * psi, 2)


# the engine, with the solve stubbed out


def test_stubbed_engine_matches_exact_engine(mol, stub):
    exact = ExactEngine(mol, TAU_IDX)
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(mol.n_states))
    drives = [(0.5 * (mol.window.omega_min + mol.window.omega_max), "+"),
              (mol.window.omega_min + 1.0, "-")]
    for w, sigma in drives:
        got = stub.branches_all_tau(p, w, sigma)
        want = exact.branches_all_tau(p, w, sigma)
        check_branches(got[0], got[1], mol.n_states)
        assert np.abs(got[0] - want[0]).max() < 1e-12
        assert np.abs(got[1] - want[1]).max() < 1e-12
    assert stub.stats["solves"] == 2 * mol.system.n_blocks and stub.stats["rk_steps"] > 0


def test_one_solve_per_drive_covers_every_tau(mol, stub):
    p = np.full(mol.n_states, 1.0 / mol.n_states)
    w = 0.5 * (mol.window.omega_min + mol.window.omega_max)
    stub.branches_all_tau(p, w, "+")
    solves = stub.stats["solves"]
    assert solves == mol.system.n_blocks
    p0, _ = stub.branches_all_tau(p, w, "+")
    assert p0.shape == (TAU_IDX.size, mol.n_states)
    assert stub.stats["solves"] == solves          # cached, and every tau came from one solve


# selection


def test_engine_choice_reads_the_environment(monkeypatch):
    monkeypatch.delenv("QLSGYM_ENGINE", raising=False)
    assert engine_choice() == "auto"
    monkeypatch.setenv("QLSGYM_ENGINE", "Exact")
    assert engine_choice() == "exact"
    assert engine_choice("cudaq") == "cudaq"
    with pytest.raises(ValueError):
        engine_choice("gpu")


def test_forced_exact_never_touches_cudaq(mol, monkeypatch):
    monkeypatch.setattr(ce, "dynamics_available", lambda *a, **k: pytest.fail("probed"))
    engine = select_engine(mol, TAU_IDX, prefer="exact", log=lambda s: None)
    assert isinstance(engine, ExactEngine)
    assert engine.selection["selected"] == "ExactEngine"


def test_auto_reports_the_engine_it_picked_and_why(mol):
    said = []
    engine = select_engine(mol, TAU_IDX, prefer="auto", log=said.append)
    ok, reason = dynamics_available()
    assert type(engine).__name__ == ("CudaqEngine" if ok else "ExactEngine")
    assert engine.selection == {"requested": "auto", "selected": type(engine).__name__,
                                "reason": reason}
    assert said and type(engine).__name__ in said[0] and reason in said[0]
    prov = engine_provenance(engine)
    assert prov["kind"] == type(engine).__name__ and prov["selection"] == engine.selection


def test_forced_cudaq_fails_loudly_where_it_cannot_run(mol):
    ok, reason = dynamics_available()
    if ok:
        pytest.skip("CUDA-Q dynamics is available here")
    with pytest.raises(RuntimeError, match="cannot run here"):
        select_engine(mol, TAU_IDX, prefer="cudaq", log=lambda s: None)


# the real thing


def _require_gpu():
    ok, reason = dynamics_available()
    if not ok:
        pytest.skip(reason)


def _compare_with_exact(mol, block, idx, atol):
    engine, exact = CudaqEngine(mol, idx), ExactEngine(mol, idx)
    p = np.zeros(mol.n_states)
    p[mol.blocks[block].states] = 1.0 / mol.blocks[block].n_states
    w = 0.5 * (mol.window.omega_min + mol.window.omega_max)
    got, want = engine.branches_all_tau(p, w, "+"), exact.branches_all_tau(p, w, "+")
    check_branches(got[0], got[1], mol.n_states)
    assert engine.stats["solves"] == 1
    err = max(np.abs(got[0] - want[0]).max(), np.abs(got[1] - want[1]).max())
    assert err < atol, err
    return err


@pytest.mark.gpu
@pytest.mark.parametrize("name,block", [("h3o", 0), ("thf", 11)])
def test_cudaq_matches_exact_engine(name, block):
    """Short pulses: enough dynamics to be a real check, few enough RK steps to be a test."""
    _require_gpu()
    mol = load_molecule(name)
    print(name, "max error", _compare_with_exact(mol, block, np.array([0, 1, 2, 5]), 1e-5))


@pytest.mark.gpu
@pytest.mark.slow
def test_cudaq_matches_exact_engine_over_the_whole_tau_grid():
    mol = load_molecule("h3o")
    _require_gpu()
    print("h3o full grid max error", _compare_with_exact(mol, 0, None, 1e-5))


@pytest.mark.gpu
@pytest.mark.slow
def test_a_reckless_step_size_raises():
    """The reason max_step_size is explicit: CUDA-Q's own default diverges on this system."""
    _require_gpu()
    mol = load_molecule("h3o")
    engine = CudaqEngine(mol, np.array([0, 199]), max_step_size=1.0, max_phase_per_step=1e3)
    p = np.full(mol.n_states, 1.0 / mol.n_states)
    with pytest.raises(RuntimeError, match="lost"):
        engine.branches_all_tau(p, 0.5 * (mol.window.omega_min + mol.window.omega_max), "+")
