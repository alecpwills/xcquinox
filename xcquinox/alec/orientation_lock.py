"""Orientation lock: a deterministic degeneracy-lifting bias for h_core.

Some open-shell radicals have an orbitally-degenerate ground state -- OH and NO
are X-2-Pi doublets, whose singly-occupied pi hole may sit in any linear
combination of the degenerate (pi_x, pi_y) pair. The total ENERGY is invariant
to that choice, but the single-determinant DENSITY on a fixed real-space grid is
not: threaded-BLAS non-associativity tips the near-degenerate SCF to a different
pi component from one process/machine to the next, so the density is not
reproducible even though the energy is. When such a density is used as a training
target (DFS-style density training weights the density ~20x) or reported as a
benchmark, the arbitrary orientation is a physical artifact, not a code bug.

The fix is an **orientation lock**: add a small, fixed, deterministic anisotropic
one-electron operator to h_core so the SCF always relaxes to the SAME
representative of the degenerate manifold. Applied *identically* in the CCSD
reference generation, the PBE seed, training, and eval (all off the same
geometry+basis, so the operator matrix is byte-identical), the reference and the
functional lock the same pi component, and the density becomes reproducible.

Physically this **selects one representative of the degenerate 2-Pi manifold** --
standard broken-symmetry practice; the density-only comparison is well defined
because both sides pick the same one.

The operator is a **traceless, generic-anisotropic quadrupole** about the
nuclear-charge centroid:

    M = sum_ij W_ij <chi_mu | r_i r_j | chi_nu>          (AO basis)
    bias = strength * M

* **Traceless** ``W`` (Tr W = 0): the first-order energy shift ``strength *
  Tr(M D)`` vanishes for a (near-)isotropic density, so the lock splits the pi
  pair without materially shifting total energies. Non-cylindrical closed-shell
  systems get only a small O(strength) shift, kept negligible by ``strength``.
* **Distinct eigenvalues + generic (non-axis-aligned) principal axes**: lifts the
  p/pi degeneracy for *any* molecular orientation. A purely diagonal ``x^2 - z^2``
  form would vanish on a bond lying in the x=z plane; the off-diagonal terms
  avoid every such pathological axis.
* **Density-independent and deterministic**: identical (geometry, basis) -> the
  identical matrix, which is what guarantees ref and eval lock the same
  component.

``DEFAULT_STRENGTH`` is calibrated so the induced pi splitting (~1e-6..1e-5 Ha)
sits ~4 orders above float64/BLAS noise and the intrinsic finite-basis pi
asymmetry (so it reliably locks), while a closed-shell PBE total energy shifts by
< 0.1 kcal/mol (negligible for the demo's ~1.7 kcal/mol AE-MAE, and it cancels in
the like-for-like NN-vs-PBE and density comparisons that both use the biased
h_core).
"""
import numpy as np

# Recommended lock strength (Hartree per Bohr^2 coefficient on the traceless
# quadrupole). See the module docstring for the calibration argument.
DEFAULT_STRENGTH: float = 3e-5

# Fixed, symmetric, TRACELESS weight matrix with three distinct eigenvalues
# (~ -1.21, -0.83, +2.03) and generic (non-axis-aligned) principal axes. Do not
# change casually: ref-gen and eval must use the identical operator, and the
# distinct-eigenvalue / generic-axis properties are what make the lock work for
# any orientation.
_W: np.ndarray = np.array(
    [[2.0, 0.3, 0.1],
     [0.3, -1.0, 0.2],
     [0.1, 0.2, -1.0]],
    dtype=float,
)


def orientation_lock_bias(mol, strength: float) -> np.ndarray:
    """Return the AO-basis orientation-lock bias matrix ``strength * M``.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        The molecule (already built). Its geometry+basis fully determine the
        operator; ``mol`` is not mutated (the common origin is set inside a
        context manager and restored).
    strength : float
        Coefficient on the traceless quadrupole. ``0.0`` -> a zero matrix (the
        off / byte-identical path).

    Returns
    -------
    numpy.ndarray
        Symmetric ``(nao, nao)`` matrix to add to h_core. Deterministic and
        density-independent.
    """
    strength = float(strength)
    nao = mol.nao
    if strength == 0.0:
        return np.zeros((nao, nao))

    charges = np.asarray(mol.atom_charges(), dtype=float)
    coords = np.asarray(mol.atom_coords(), dtype=float)
    centroid = (charges[:, None] * coords).sum(axis=0) / charges.sum()

    # int1e_rr -> <mu| r_i r_j |nu>, flattened (9, nao, nao); origin-dependent,
    # so anchor it at the nuclear-charge centroid (translation-invariant) via a
    # context manager that restores mol's prior common origin.
    with mol.with_common_origin(centroid):
        q = np.asarray(mol.intor("int1e_rr")).reshape(3, 3, nao, nao)

    m = np.einsum("ij,ijmn->mn", _W, q)
    m = 0.5 * (m + m.T)  # symmetrize against integral round-off
    return strength * m
