"""The stage protocol and the registry."""

from __future__ import annotations

import importlib
from typing import Protocol, runtime_checkable

from ..config import Config


@runtime_checkable
class Stage(Protocol):
    NAME: str
    TITLE: str
    PAPER: str            # the section/figure of arXiv:2608.03702 it targets
    REQUIRES: tuple       # NAMEs of stages whose JSON this one reads
    COST: str             # human-readable, e.g. "~4 h, GPU (array of 6)"

    def plan(self, cfg: Config) -> list: ...
    def run(self, cfg: Config) -> dict: ...


# execution order.  Adding a stage means adding a module and a line here.
ORDER = (
    "s01_physics",
    "s02_data",
    "s03_train",
    "s04_accuracy",
    "s05_planner",
    "s06_rl",
    "s07_figures",
)


def load(name: str):
    """Import one stage module by NAME."""
    if name not in ORDER:
        raise KeyError(f"unknown stage {name!r}; known: {', '.join(ORDER)}")
    return importlib.import_module(f".{name}", __package__)


def all_stages() -> list:
    return [load(n) for n in ORDER]


def check_requirements(stage, cfg: Config) -> list:
    """REQUIRES that are not on disk yet (empty means ready to run)."""
    from ..config import stage_is_done
    return [r for r in getattr(stage, "REQUIRES", ()) if not stage_is_done(r)]
