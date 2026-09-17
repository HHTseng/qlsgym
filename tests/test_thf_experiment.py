"""Reject aggregation that would conceal a changed task or surrogate."""

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "thf_rl_agents.py"
SPEC = importlib.util.spec_from_file_location("fno_experiment", SCRIPT)
DRIVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DRIVER)


@pytest.fixture
def records(tmp_path):
    record = {
        "contract": {
            "molecule_fingerprint": "physics-v1",
            "library_tag": "library-v1",
            "manifest_tag": "pilot",
            "manifest_sha256": "weights-v1",
            "p_target": 0.98,
            "max_pulses": 80,
            "rho": 0.0,
        },
        "job": {"preset": "smoke"},
    }
    (tmp_path / "first.json").write_text(json.dumps(record))
    return tmp_path, record


@pytest.mark.parametrize("field,value", [
    ("manifest_sha256", "weights-v2"),
    ("manifest_tag", "production"),
    ("p_target", 0.99),
    ("max_pulses", 200),
    ("rho", 0.1),
])
def test_reject_changed_contract(records, field, value):
    directory, record = records
    record["contract"][field] = value
    (directory / "second.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="contracts"):
        DRIVER.load_records(directory)


def test_reject_changed_budget(records):
    directory, record = records
    record["job"]["preset"] = "primary"
    (directory / "second.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="contracts"):
        DRIVER.load_records(directory)


def test_accept_same_contract(records):
    directory, record = records
    (directory / "second.json").write_text(json.dumps(record))
    assert len(DRIVER.load_records(directory)) == 2


def test_reject_changed_agent_hyperparameters(records):
    directory, record = records
    record["job"]["agent"] = "sac_discrete"
    record["training"] = {"config": {"seed": 0, "alpha": 0.05}}
    (directory / "first.json").write_text(json.dumps(record))
    record["training"]["config"] = {"seed": 1, "alpha": 0.000625}
    (directory / "second.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="hyperparameter contracts"):
        DRIVER.load_records(directory)
