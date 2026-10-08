from __future__ import annotations

import functools
import os
import tomllib
from pathlib import Path

ENV = "QLSGYM_PHYSICS_CONFIG"
DEFAULT_PATH = Path(__file__).resolve().with_name("physics.config")


def config_path() -> Path:
    return Path(os.environ.get(ENV) or DEFAULT_PATH)


@functools.cache
def _load(path: str) -> dict:
    with open(path, "rb") as fh:
        return tomllib.load(fh)


def physics_config() -> dict:
    return _load(str(config_path()))


def constants() -> dict:
    return physics_config()["constants"]


def molecule_config(name: str) -> dict:
    cfg = physics_config()
    if name not in cfg:
        raise KeyError(f"{config_path()} has no [{name}] section")
    return cfg[name]
