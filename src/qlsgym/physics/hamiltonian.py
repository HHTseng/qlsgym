"""Block Hamiltonians for a single Raman sideband pulse."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..spec import Block, Molecule
from .spectrum import resonant_frequencies

__all__ = [
    "BlockOperator", "make_operator", "operator", "lump", "rwa_sectors",
    "sideband_matrix_elements", "components", "hermiticity_error",
]


# Small helpers


def sideband_matrix_elements(eta: float, n_nu: int, n_fock: int = 40) -> np.ndarray:
    """<n+1| exp(i eta (a + a^dag)) |n> for n = 0 .. n_nu - 2, from the displacement operator
    itself rather than a first-order Lamb-Dicke expansion.
    """
    n = np.arange(1, n_fock)
    a = np.diag(np.sqrt(n), 1)
    x = a + a.conj().T
    w, v = np.linalg.eigh(x)
    d = (v * np.exp(1j * eta * w)) @ v.conj().T
    return np.array([d[k + 1, k] for k in range(n_nu - 1)], dtype=np.complex128)


def components(n: int, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Connected-component label per node of the undirected graph {src[k] -- dst[k]} on n nodes
    (union-find, labels 0..n_comp-1 in order of first appearance of the root).
    """
    parent = np.arange(n)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in zip(np.asarray(src).tolist(), np.asarray(dst).tolist()):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
    roots = np.array([find(a) for a in range(n)])
    _, labels = np.unique(roots, return_inverse=True)
    return labels


def hermiticity_error(h: np.ndarray) -> float:
    return float(np.max(np.abs(h - h.conj().T)))


def lump(t: np.ndarray, n_states: int) -> np.ndarray:
    """Collapse a nu-resolved transfer array onto the measured branches."""
    m = int(n_states)
    rest = t[..., m:, :]
    lumped = rest.reshape(*t.shape[:-2], -1, m, t.shape[-1]).sum(axis=-3)
    return np.concatenate([t[..., :m, :], lumped], axis=-2)


# The operator


@dataclass(frozen=True)
class BlockOperator:
    """Pre-computed, well-conditioned Hamiltonian builder for one block or RWA sector of one
    molecule.
    """

    block_index: int
    key: tuple
    n_states: int                 # M_f
    sigma: str
    nu_f: float
    eta: float
    n_nu: int
    e_shifted: np.ndarray         # (M_f,) molecular energies, coarse per-component origin
    src: np.ndarray               # (n_keep,) local index, nu side
    dst: np.ndarray               # (n_keep,) local index, nu + 1 side
    g: np.ndarray                 # (n_keep,) complex (Omega/2) <1|D(i eta)|0> = i eta e^{-eta^2/2} Omega/2
    rung: np.ndarray              # (n_nu - 1,) complex d_nu / d_0; rung[0] == 1 exactly
    component: np.ndarray         # (M_f,) component id per molecular state
    keep: np.ndarray              # (n_cpl,) bool mask into Block.omega
    e_origin: np.ndarray          # (n_comp,) subtracted energy origins
    # dynamically decoupled components of the nu-enlarged space (n_nu M_f,)
    component_n: np.ndarray
    n_component_n: int

    @property
    def dim(self) -> int:
        return self.n_nu * self.n_states

    def _graph(self, n_nu: int):
        m = self.n_states
        src = np.concatenate([self.src + k * m for k in range(n_nu - 1)])
        dst = np.concatenate([self.dst + (k + 1) * m for k in range(n_nu - 1)])
        gval = np.concatenate(
            [self.g if k == 0 else self.g * self.rung[k] for k in range(n_nu - 1)]
        )
        return src, dst, gval

    def build(self, omega: float, n_nu: int | None = None) -> np.ndarray:
        """Dense (n_nu M_f, n_nu M_f) Hermitian Hamiltonian at drive omega (rad/ms), nu-major."""
        n_nu = self.n_nu if n_nu is None else int(n_nu)
        if n_nu < 2:
            raise ValueError("n_nu must be at least 2")
        if n_nu > self.rung.size + 1:
            raise ValueError(f"operator was built for n_nu <= {self.rung.size + 1}")
        m = self.n_states
        dim = n_nu * m
        src, dst, gval = self._graph(n_nu)
        diag = np.concatenate([self.e_shifted + k * (self.nu_f - omega) for k in range(n_nu)])
        if n_nu == self.n_nu:
            comp, n_comp = self.component_n, self.n_component_n
        else:
            comp = components(dim, src, dst)
            n_comp = int(comp.max()) + 1
        means = np.bincount(comp, weights=diag, minlength=n_comp) / np.bincount(comp, minlength=n_comp)
        diag = diag - means[comp]

        h = np.zeros((dim, dim), dtype=np.complex128)
        idx = np.arange(dim)
        h[idx, idx] = diag
        np.add.at(h, (dst, src), gval)
        np.add.at(h, (src, dst), np.conjugate(gval))
        return h

    def detunings(self, omega: float) -> np.ndarray:
        """Delta = E(J',nu=1) - E(J,nu=0) for the retained couplings."""
        return self.e_shifted[self.dst] + (self.nu_f - omega) - self.e_shifted[self.src]


def make_operator(
    molecule: Molecule,
    block,
    sigma: str = "+",
    omega_lo: float | None = None,
    omega_hi: float | None = None,
    rwa_cutoff: float | None = None,
) -> BlockOperator:
    """Build the BlockOperator for block (index, block or RWA sector) and polarisation sigma."""
    s, trap, w = molecule.system, molecule.trap, molecule.window
    b = s.blocks[int(block)] if not isinstance(block, Block) else block
    lo = w.omega_min if omega_lo is None else float(omega_lo)
    hi = w.omega_max if omega_hi is None else float(omega_hi)
    cutoff = w.rwa_cutoff if rwa_cutoff is None else float(rwa_cutoff)

    w_res = resonant_frequencies(molecule, b, sigma)
    dist = np.maximum(0.0, np.maximum(lo - w_res, w_res - hi))
    keep = dist <= cutoff

    if sigma == "+":
        src, dst = b.i_local[keep], b.f_local[keep]
    elif sigma == "-":
        src, dst = b.f_local[keep], b.i_local[keep]
    else:
        raise ValueError(f"sigma must be '+' or '-', got {sigma!r}")
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)

    m = b.n_states
    comp = components(m, src, dst)
    e_mol = s.energies[b.states]
    n_comp = int(comp.max()) + 1 if m else 0
    # Coarse origin on the molecular energies: enough to keep e_shifted small
    # for MHz drives; build() then does the exact per-component referencing in
    # the nu-enlarged space.  Use the nu = 0 (source) side of each component,
    # because for a THz drive the two motional layers sit in different
    # rotational manifolds and their mean would leave a ~1e9 rad/ms residual.
    origin = np.empty(n_comp, dtype=np.float64)
    src_set = set(src.tolist())
    for c in range(n_comp):
        members = np.where(comp == c)[0]
        sources = [int(x) for x in members if int(x) in src_set]
        origin[c] = e_mol[sources].mean() if sources else e_mol[members].mean()
    e_shifted = e_mol - origin[comp]

    eta, n_nu = trap.eta, int(trap.n_nu)
    d = sideband_matrix_elements(eta, max(n_nu, 2))
    rung = d / d[0]
    rung[0] = 1.0
    g = 1j * eta * np.exp(-(eta ** 2) / 2.0) * b.omega[keep] / 2.0

    src_n = np.concatenate([src + k * m for k in range(n_nu - 1)])
    dst_n = np.concatenate([dst + (k + 1) * m for k in range(n_nu - 1)])
    comp_n = components(n_nu * m, src_n, dst_n)

    return BlockOperator(
        block_index=int(b.index), key=tuple(b.key), n_states=m, sigma=sigma,
        nu_f=float(trap.nu_f), eta=float(eta), n_nu=n_nu,
        e_shifted=e_shifted, src=src, dst=dst, g=g, rung=rung,
        component=comp, keep=keep, e_origin=origin,
        component_n=comp_n, n_component_n=int(comp_n.max()) + 1 if m else 0,
    )


_OPERATORS: dict = {}


def operator(molecule: Molecule, block_index: int, sigma: str = "+",
             omega: float | None = None) -> BlockOperator:
    """Cached make_operator, keyed on (fingerprint, block, sigma, omega)."""
    key = (molecule.fingerprint(), int(block_index), sigma, None if omega is None else float(omega))
    op = _OPERATORS.get(key)
    if op is None:
        if omega is None:
            op = make_operator(molecule, block_index, sigma)
        else:
            op = make_operator(molecule, block_index, sigma, omega_lo=omega, omega_hi=omega)
        _OPERATORS[key] = op
    return op


# Off-window primitives: RWA sectors


def rwa_sectors(molecule: Molecule, sigma: str, omega: float,
                rwa_cutoff: float | None = None) -> tuple:
    """Dynamically coupled sectors at drive omega: connected components of the *global* couplings
    that survive the RWA there.
    """
    s = molecule.system
    cutoff = molecule.window.rwa_cutoff if rwa_cutoff is None else float(rwa_cutoff)
    d = s.energies[s.f_idx] - s.energies[s.i_idx]
    w_res = molecule.trap.nu_f + d if sigma == "+" else molecule.trap.nu_f - d
    keep = np.abs(w_res - omega) <= cutoff
    if not keep.any():
        return ()
    i_k, f_k, w_k = s.i_idx[keep], s.f_idx[keep], s.omega_c[keep]
    n = s.n_states
    labels = components(n, i_k, f_k)
    touched = np.unique(np.concatenate([i_k, f_k]))
    roots = labels[touched]
    sectors = []
    for k, r in enumerate(np.unique(roots)):
        states = np.sort(touched[roots == r])
        local = {int(gidx): q for q, gidx in enumerate(states)}
        sel = np.isin(i_k, states)
        blocks = sorted({int(s.block_of_state[gidx]) for gidx in states})
        sectors.append(Block(
            index=-1 - k, states=states,
            key=("sector", round(float(omega), 6), sigma, tuple(blocks)),
            i_local=np.array([local[int(x)] for x in i_k[sel]], dtype=np.int64),
            f_local=np.array([local[int(x)] for x in f_k[sel]], dtype=np.int64),
            omega=w_k[sel].copy(),
        ))
    return tuple(sectors)
