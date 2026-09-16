"""Molecule packs. Each module exposes build(**kw) -> Molecule."""
from __future__ import annotations
import importlib
from ..spec import Molecule

_REGISTRY = {"h3o": "qlsgym.molecules.h3o", "thf": "qlsgym.molecules.thf",
             "synthetic": "qlsgym.molecules.synthetic"}


def load(name: str, **kw) -> Molecule:
    if name not in _REGISTRY:
        raise KeyError(f"unknown molecule {name!r}; known: {sorted(_REGISTRY)}")
    return importlib.import_module(_REGISTRY[name]).build(**kw)


def register(name: str, module: str) -> None:
    _REGISTRY[name] = module
