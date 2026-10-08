from __future__ import annotations

import numpy as np

from ..spec import Molecule, TWO_PI

KB_OVER_HBAR = TWO_PI * 2.0836619123e7


def boltzmann(molecule: Molecule, temperature_k: float | None = None) -> np.ndarray:
    t = molecule.task.temperature_k if temperature_k is None else float(temperature_k)
    e = molecule.system.energies - molecule.system.energies.min()
    w = np.exp(-e / (KB_OVER_HBAR * t))
    return w / w.sum()
