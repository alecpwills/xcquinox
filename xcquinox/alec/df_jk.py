"""Density-fitting (RI) helpers for the SCF Coulomb build.

PBE is a pure GGA (no exact exchange), so the SCF only needs the Coulomb
matrix ``J``. The density-fitted tensor ``cderi`` (shape ``(naux, nao, nao)``)
depends only on geometry + basis (NOT on the NN parameters), so it is built
once with pyscf at precompute time and contracted in JAX inside the
differentiable SCF loop::

    J = einsum('Lij,L->ij', cderi, einsum('Lkl,kl->L', cderi, D))

which is differentiable w.r.t. ``D`` (cderi is a constant). This replaces the
full 4-index ERI (``8 * nao**4`` bytes) with a ``naux * nao**2`` tensor,
making larger bases (def2-tzvp) memory-feasible.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

# Orbital-basis -> standard JK fitting basis. Explicit pairs avoid surprises for
# the bases we actually run. The per-basis ``-jkfit`` sets are the tighter
# matched fits and are preferred where they exist; the diffuse/large def2 bases
# (def2-tzvpd, def2-tzvppd, def2-qzvp) have no dedicated ``-jkfit`` and use the
# Weigend universal Coulomb-fitting set ``def2-universal-jkfit``, which is
# designed to fit J (and K) for ALL def2 orbital bases.
#   (L) F. Weigend, Phys. Chem. Chem. Phys. 8, 1057 (2006) — accurate
#       Coulomb-fitting (def2-universal-JKFIT) basis sets for H to Rn.
_DEF2_UNIVERSAL_JKFIT = "def2-universal-jkfit"
_AUXBASIS_TABLE = {
    "def2-svp": "def2-svp-jkfit",
    "def2-svpd": "def2-svp-jkfit",
    "def2-tzvp": "def2-tzvp-jkfit",
    "def2-tzvpp": "def2-tzvp-jkfit",
    "def2-tzvpd": _DEF2_UNIVERSAL_JKFIT,
    "def2-tzvppd": _DEF2_UNIVERSAL_JKFIT,
    "def2-qzvp": _DEF2_UNIVERSAL_JKFIT,
}


def default_auxbasis(orbital_basis: str | None) -> str | None:
    """Map an orbital basis to its JK fitting basis.

    Returns the matched ``-jkfit`` set when one is tabulated; for any other
    ``def2-*`` orbital basis returns the cited Weigend ``def2-universal-jkfit``
    (a *reproducible* default rather than pyscf's internal auto-select). For
    non-def2 / unknown bases (e.g. sto-3g, cc-pVDZ) returns ``None`` so pyscf
    auto-selects an appropriate fit. ``None`` orbital basis -> ``None``.
    """
    if orbital_basis is None:
        return None
    key = orbital_basis.lower()
    if key in _AUXBASIS_TABLE:
        return _AUXBASIS_TABLE[key]
    if key.startswith("def2-"):
        return _DEF2_UNIVERSAL_JKFIT
    return None


def build_cderi(mol, auxbasis: str | None = None) -> jnp.ndarray:
    """Build the unpacked 3-index DF tensor ``(naux, nao, nao)`` for ``mol``.

    ``auxbasis=None`` resolves via :func:`default_auxbasis(mol.basis)`, and if
    that is also None, pyscf auto-selects. Heavy (pyscf libcint); called once
    per molecule at precompute time, never inside the SCF loop.
    """
    from pyscf import df, lib
    aux = auxbasis if auxbasis is not None else default_auxbasis(
        getattr(mol, "basis", None))
    dfobj = df.DF(mol) if aux is None else df.DF(mol, auxbasis=aux)
    dfobj.build()
    # dfobj._cderi is (naux, nao_pair) lower-triangular-packed; unpack to the
    # symmetric (naux, nao, nao) so the JAX contraction is a plain einsum.
    cderi_packed = np.asarray(dfobj._cderi)
    cderi = lib.unpack_tril(cderi_packed)               # (naux, nao, nao)
    return jnp.asarray(cderi)


def compute_j_df(D: jnp.ndarray, cderi: jnp.ndarray) -> jnp.ndarray:
    """Coulomb matrix from the DF tensor: ``J[D] = cderi . (cderi : D)``.

    Differentiable in ``D``; ``cderi`` is a constant precompute.
    """
    jaux = jnp.einsum("Lkl,kl->L", cderi, D)            # (naux,)
    return jnp.einsum("Lij,L->ij", cderi, jaux)         # (nao, nao)
