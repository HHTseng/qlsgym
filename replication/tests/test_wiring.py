"""The folder is wired correctly: stages, CLI, result I/O. Fast and CPU-only."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from replication import config as cfgmod                          # noqa: E402
from replication.config import (Config, MissingResult, VintageMismatch,   # noqa: E402
                                read_result, result_path, write_result)
from replication.stages import base                               # noqa: E402
from replication.stages import s01_physics                        # noqa: E402

REPL = ROOT / "replication"
RUN_PY = REPL / "run.py"


@pytest.fixture
def outputs(tmp_path, monkeypatch):
    """Point the one output tree at tmp_path: a test never writes into OUTPUTS/replication."""
    monkeypatch.setenv("QLSGYM_REPL_OUTPUTS", str(tmp_path))
    assert cfgmod.outputs_dir() == tmp_path
    return tmp_path


def cli(*args, cwd=ROOT, outputs=None, timeout=900) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": f"{ROOT}:{ROOT / 'src'}"}
    if outputs is not None:
        env["QLSGYM_REPL_OUTPUTS"] = str(outputs)
    return subprocess.run([sys.executable, str(RUN_PY), *args], cwd=str(cwd), env=env,
                          capture_output=True, text=True, timeout=timeout)


# The stage registry


def test_order_matches_the_modules_on_disk():
    disk = sorted(p.stem for p in (REPL / "stages").glob("s[0-9]*.py"))
    assert list(base.ORDER) == disk, (f"stages/ has {disk}, ORDER has {list(base.ORDER)}: "
                                      f"adding a stage means adding a module *and* a line in ORDER")


@pytest.mark.parametrize("name", base.ORDER)
def test_stage_satisfies_the_protocol(name):
    stage = base.load(name)
    assert stage.NAME == name
    assert isinstance(stage, base.Stage)
    for attr in ("TITLE", "PAPER", "COST"):
        assert isinstance(getattr(stage, attr), str) and getattr(stage, attr).strip()
    assert isinstance(stage.REQUIRES, tuple)
    assert all(r in base.ORDER for r in stage.REQUIRES)
    assert all(base.ORDER.index(r) < base.ORDER.index(name) for r in stage.REQUIRES)


@pytest.mark.parametrize("name", base.ORDER)
def test_plan_is_non_empty_and_cheap(name, outputs):
    lines = base.load(name).plan(Config())
    assert lines and all(isinstance(line, str) and line.strip() for line in lines)
    assert not list(outputs.iterdir()), f"{name}.plan() wrote into the output tree"


def test_blocks_are_resolved_against_the_molecule():
    """An empty Config.blocks means the molecule's unique set; an absent block is refused."""
    assert cfgmod.resolve_blocks(Config()) == s01_physics.H3O_UNIQUE_BLOCKS
    assert cfgmod.pairs(Config(molecule="synthetic")) == [(0, "+"), (1, "+"), (2, "+")]
    with pytest.raises(ValueError, match="99"):
        cfgmod.pairs(Config(molecule="synthetic", blocks=(99,)))


# The CLI


def test_cli_list(outputs):
    out = cli("--list", outputs=outputs)
    assert out.returncode == 0, out.stderr
    for name in base.ORDER:
        assert name in out.stdout
    assert str(outputs) in out.stdout


@pytest.mark.parametrize("molecule", ["h3o", "synthetic", "thf"])
def test_cli_all_dry_run_touches_nothing(molecule, outputs):
    out = cli("--all", "--dry-run", "--set", f"molecule={molecule}", outputs=outputs)
    assert out.returncode == 0, out.stderr
    assert "dry run: nothing executed" in out.stdout
    for name in base.ORDER:
        assert name in out.stdout
    assert not list(outputs.iterdir())


# Result I/O


def test_result_round_trip(outputs):
    cfg = Config(molecule="synthetic")
    payload = {"answer": 42, "blocks": [1, 2, 3]}
    path = write_result("s01_physics", payload, cfg)
    assert path == result_path("s01_physics") and path.exists()
    assert read_result("s01_physics", cfg) == payload
    assert cfgmod.stage_is_done("s01_physics")

    doc = json.loads(path.read_text())
    prov = doc["provenance"]
    assert prov["molecule"] == "synthetic" and prov["fingerprint"]
    assert prov["config"]["molecule"] == "synthetic"


def test_missing_result_names_the_stage(outputs):
    with pytest.raises(MissingResult) as excinfo:
        read_result("s02_data", Config(molecule="synthetic"))
    assert "s02_data" in str(excinfo.value)


def test_an_unreadable_result_names_the_stage_too(outputs):
    """A stage's input fails with "run stage X first", never a KeyError or a bare
    JSONDecodeError.
    """
    path = result_path("s01_physics")
    path.parent.mkdir(parents=True, exist_ok=True)
    for text in ('{"stage": "s01_physics", "provenance": {}}', '{"stage": "s01_phy'):
        path.write_text(text)
        with pytest.raises(MissingResult, match="s01_physics"):
            read_result("s01_physics", Config(molecule="synthetic"))


def test_vintage_mismatch_is_refused(outputs):
    cfg = Config(molecule="synthetic")
    path = write_result("s01_physics", {"answer": 42}, cfg)
    doc = json.loads(path.read_text())
    doc["provenance"]["fingerprint"] = "deadbeefcafe"
    path.write_text(json.dumps(doc))
    with pytest.raises(VintageMismatch):
        read_result("s01_physics", cfg)
    assert read_result("s01_physics", cfg, check_fingerprint=False) == {"answer": 42}


# s01_physics on the synthetic molecule


def test_s01_physics_on_synthetic():
    payload = s01_physics.run(Config(molecule="synthetic"))
    assert payload["checks"] and all(payload["checks"].values())
    assert payload["molecule"] == "synthetic"
    assert payload["n_states"] == sum(payload["block_dims"])
    assert payload["eq43_two_m_f"] == [2 * m for m in payload["block_dims"]]
    assert payload["truncation"]["molecule_n_nu"] == 2
    assert len(payload["resonance_geometry"]) == 2 * payload["n_blocks"]
    assert 0.0 <= payload["active_window_fraction"]["min"] <= 1.0
    # the paper's own numbers are asserted for h3o only
    assert not any(k.startswith("h3o_") for k in payload["checks"])


def test_s01_physics_crashes_on_broken_physics(monkeypatch):
    """A structural failure must raise, not be reported as a False in a dict."""
    import dataclasses

    from qlsgym import load_molecule as _load

    mol = _load("synthetic")
    broken = dataclasses.replace(mol, window=dataclasses.replace(
        mol.window, omega_min=mol.trap.nu_f + 1.0, omega_max=mol.trap.nu_f + 2.0))
    monkeypatch.setattr(s01_physics, "load_molecule", lambda cfg: broken)
    with pytest.raises(s01_physics.StructureError, match="window_brackets_sideband"):
        s01_physics.run(Config(molecule="synthetic"))


# The two import rules


def _imported_modules(path: Path) -> set:
    """Top-level module names a file imports, parsed rather than grepped (a docstring that
    *mentions* an import must not count).
    """
    tree = ast.parse(path.read_text())
    names: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_nothing_imports_a_source_physics_repo():
    sources = [p for p in REPL.rglob("*.py") if "__pycache__" not in p.parts]
    offenders = [str(p.relative_to(ROOT)) for p in sources
                 if _imported_modules(p) & {"fnorepl", "thffno"}]
    assert not offenders, (f"{offenders} depend on a source physics repo at runtime; "
                           f"everything must come from qlsgym")


def test_paper_is_a_leaf_of_quotations():
    """paper.py computes nothing: it imports nothing but __future__ and its body is assignments, so
    a quotation can never be produced by a calculation of ours.
    """
    import replication.paper as paper

    path = REPL / "paper.py"
    assert _imported_modules(path) == {"__future__"}
    body = ast.parse(path.read_text()).body
    assert all(isinstance(n, (ast.Assign, ast.AnnAssign, ast.Expr, ast.ImportFrom)) for n in body), \
        [type(n).__name__ for n in body]
    public = [n for n in vars(paper) if not n.startswith("_") and n != "annotations"]
    assert public and all(n.isupper() or n[0].isupper() for n in public), public
    assert "arXiv:2608.03702" in paper.__doc__ and "quotation" in paper.__doc__
