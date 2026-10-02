"""The off-policy screen must select one profile per agent using only FNO."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts" / "summarize_thf_offpolicy_transfer.py"
)
SPEC = importlib.util.spec_from_file_location("offpolicy_summary", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_screen_selection_is_separate_by_agent_and_fno_only(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    contract = {
        "molecule_fingerprint": "molecule",
        "library_tag": "library",
        "manifest_sha256": "manifest",
        "p_target": 0.98,
        "max_pulses": 80,
        "rho": 0.0,
    }
    for agent, profiles in MODULE.SCREEN.items():
        for rank, profile in enumerate(profiles):
            for seed in MODULE.SCREEN_SEEDS:
                record = {
                    "job": {
                        "agent": agent,
                        "offpolicy_profile": profile,
                        "train_seed": seed,
                    },
                    "contract": contract,
                    "training": {"config": {"seed": seed}},
                    "evaluation": {
                        "fno": {
                            "unfinished_fraction": 0.2 + 0.1 * rank,
                            "average_actions": 40 + rank,
                        }
                    },
                    "wall_clock_s": 1.0,
                }
                (runs / f"{profile}_{seed}.json").write_text(json.dumps(record))

    output = tmp_path / "selection.json"
    MODULE.select(SimpleNamespace(runs=runs, output=output))
    selection = json.loads(output.read_text())
    assert selection["selected_profiles"] == {
        agent: profiles[0] for agent, profiles in MODULE.SCREEN.items()
    }
    assert all("exact_failure_mean" not in item for item in selection["rows"])
