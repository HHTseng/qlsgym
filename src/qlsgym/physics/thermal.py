"""Thermal (Boltzmann) populations over the molecular states (paper Eq. 5)."""

from __future__ import annotations

import numpy as np

from ..spec import Molecule, TWO_PI

# Boltzmann constant in the internal angular-frequency units, k_B / hbar
# in rad/ms per kelvin: k_B/h = 20.836619123 GHz/K so
# k_B/hbar = 2 pi * 2.0836619123e7 kHz/K (both source projects).
KB_OVER_HBAR = TWO_PI * 2.0836619123e7


def boltzmann(molecule: Molecule, temperature_k: float | None = None) -> np.ndarray:
    """Global thermal populations over all n_states molecular states."""
    t = molecule.task.temperature_k if temperature_k is None else float(temperature_k)
    e = molecule.system.energies - molecule.system.energies.min()
    w = np.exp(-e / (KB_OVER_HBAR * t))
    return w / w.sum()
