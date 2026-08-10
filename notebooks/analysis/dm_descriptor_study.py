"""Candidate replacements for dm_entropy, screened against the three criteria.

The requirement that killed both dm_entropy and the participation ratio:

    For a single determinant the natural occupations of DS are EXACTLY
    {2,...,2,0,...,0} (RKS). So ANY function of the SPECTRUM of DS alone is a
    function of N_occ only, hence CONSTANT on the idempotent manifold at fixed
    electron count -- and every converged SCF density lies on that manifold.

That predicts an entire class is dead, not just the two already tried. The
discriminating test is therefore: hold N fixed, change the BONDING, and see
whether the candidate moves. H2 at 0.74 A vs 2.00 A is the cleanest such pair --
same molecule, same electron count, both idempotent, completely different
bonding. N2 vs CO is the same test across species (both 14 electrons).

Screened on:
  (a) VARIES on the idempotent manifold (the killer criterion)
  (b) size-INTENSIVE (a global feature is broadcast to every grid point, so a
      size-extensive one leaks molecule identity -- the original dm_entropy bug)
  (c) exact gradient at a converged density, eigh-free
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from pyscf import gto, dft


def scf(atom, basis="def2-svp", spin=0, charge=0):
    mol = gto.M(atom=atom, basis=basis, spin=spin, charge=charge, verbose=0)
    mf = dft.RKS(mol); mf.xc = "pbe"; mf.kernel()
    return mol, jnp.asarray(mf.make_rdm1()), jnp.asarray(mol.intor("int1e_ovlp"))


def atom_slices(mol):
    return [(int(a[2]), int(a[3])) for a in mol.aoslice_by_atom()]


# ---------------------------------------------------------------- candidates
def trace_power(P, S, n):
    """Tr[(DS)^n] / N_e -- a pure spectral invariant. Predicted CONSTANT."""
    M = P @ S
    A = M
    for _ in range(n - 1):
        A = A @ M
    return jnp.trace(A) / jnp.trace(M)


def participation_ratio(P, S):
    """(Tr[DS])^2 / Tr[(DS)^2] -- the already-rejected candidate."""
    M = P @ S
    return jnp.trace(M) ** 2 / jnp.trace(M @ M)


def mayer_bond_order_mean(P, S, slices):
    """Mean Mayer bond order per atom.

    B_AB = sum_{mu in A} sum_{nu in B} (PS)_{mu nu} (PS)_{nu mu}
    (Mayer, Chem. Phys. Lett. 97, 270 (1983)). Pure matrix products, so
    polynomial in P: no eigendecomposition, no degeneracy problem. It probes
    the EIGENVECTORS (how density is shared between atomic blocks) rather than
    the spectrum, which is why it can move on the idempotent manifold.
    Normalized by atom count to stay size-intensive.
    """
    M = P @ S
    tot = 0.0
    n_at = len(slices)
    for a in range(n_at):
        i0, i1 = slices[a]
        for b in range(a + 1, n_at):
            j0, j1 = slices[b]
            blk = M[i0:i1, j0:j1] * M[j0:j1, i0:i1].T
            tot = tot + jnp.sum(blk)
    return tot / n_at


def charge_dispersion(P, S, slices):
    """Population spread across atoms: sqrt(mean_A (q_A - mean q)^2) / mean q.

    q_A = sum_{mu in A} (PS)_{mu mu} is the Mulliken population. The relative
    dispersion is intensive by construction and polynomial in P.
    """
    M = P @ S
    q = jnp.stack([jnp.sum(jnp.diag(M)[i0:i1]) for i0, i1 in slices])
    mean_q = jnp.mean(q)
    return jnp.sqrt(jnp.mean((q - mean_q) ** 2)) / (mean_q + 1e-12)


def interatomic_delocalization(P, S, slices):
    """Fraction of ||PS||_F^2 carried by INTER-atomic blocks.

    Intensive (a ratio), polynomial, and directly a covalency measure: 0 for a
    fully atom-localized density, larger the more the density is shared.
    """
    M = P @ S
    total = jnp.sum(M * M)
    intra = 0.0
    for i0, i1 in slices:
        blk = M[i0:i1, i0:i1]
        intra = intra + jnp.sum(blk * blk)
    return (total - intra) / (total + 1e-12)


CANDIDATES = {
    "Tr[(DS)^3]/N (spectral)": lambda P, S, sl: trace_power(P, S, 3),
    "Tr[(DS)^4]/N (spectral)": lambda P, S, sl: trace_power(P, S, 4),
    "participation ratio": lambda P, S, sl: participation_ratio(P, S),
    "Mayer bond order / atom": mayer_bond_order_mean,
    "charge dispersion": charge_dispersion,
    "interatomic delocalization": interatomic_delocalization,
}


def main():
    print("=" * 92)
    print("(a) DOES IT VARY ON THE IDEMPOTENT MANIFOLD?")
    print("    Same electron count, different bonding. A spectrum-only")
    print("    invariant must give identical values; a useful one must not.")
    print("=" * 92)
    systems = [
        ("H2 @0.74A", "H 0 0 0; H 0 0 0.74", 0, 0),
        ("H2 @2.00A", "H 0 0 0; H 0 0 2.00", 0, 0),
        ("N2  (14 e)", "N 0 0 0; N 0 0 1.098", 0, 0),
        ("CO  (14 e)", "C 0 0 0; O 0 0 1.128", 0, 0),
    ]
    built = []
    for name, atom, spin, charge in systems:
        mol, P, S = scf(atom, spin=spin, charge=charge)
        built.append((name, mol, P, S, atom_slices(mol)))
    hdr = "".join(f"{n:>14}" for n, *_ in built)
    print(f"{'candidate':<30}{hdr}")
    for label, fn in CANDIDATES.items():
        vals = [float(fn(P, S, sl)) for _n, _m, P, S, sl in built]
        print(f"{label:<30}" + "".join(f"{v:14.6f}" for v in vals))

    print()
    print("  VERDICT per candidate:")
    for label, fn in CANDIDATES.items():
        v = [float(fn(P, S, sl)) for _n, _m, P, S, sl in built]
        h2_moves = abs(v[0] - v[1]) > 1e-6 * max(abs(v[0]), 1.0)
        n2co_moves = abs(v[2] - v[3]) > 1e-6 * max(abs(v[2]), 1.0)
        print(f"    {label:<30} H2 stretch: "
              f"{'VARIES' if h2_moves else 'CONSTANT':<9} "
              f"N2 vs CO: {'VARIES' if n2co_moves else 'CONSTANT'}")

    print()
    print("=" * 92)
    print("(b) SIZE-INTENSIVE? one molecule vs two copies 100 A apart")
    print("=" * 92)
    mol1, P1, S1 = scf("H 0 0 0; H 0 0 0.74")
    mol2, P2, S2 = scf("H 0 0 0; H 0 0 0.74; H 100 0 0; H 100 0 0.74")
    sl1, sl2 = atom_slices(mol1), atom_slices(mol2)
    print(f"{'candidate':<30}{'H2':>14}{'2x H2':>14}{'ratio':>10}  verdict")
    for label, fn in CANDIDATES.items():
        a = float(fn(P1, S1, sl1)); b = float(fn(P2, S2, sl2))
        r = b / a if abs(a) > 1e-14 else float("nan")
        ok = "INTENSIVE" if abs(r - 1.0) < 1e-4 else f"extensive ({r:.3f}x)"
        print(f"{label:<30}{a:14.6f}{b:14.6f}{r:10.4f}  {ok}")

    print()
    print("=" * 92)
    print("(c) EXACT GRADIENT AT A CONVERGED (IDEMPOTENT) DENSITY?")
    print("=" * 92)
    mol, P, S = scf("O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469")
    sl = atom_slices(mol)
    rng = np.random.default_rng(3)
    W = rng.normal(size=P.shape); W = jnp.asarray(0.5 * (W + W.T))
    print(f"{'candidate':<30}{'value':>14}{'FD':>14}{'autodiff':>14}{'rel':>11}")
    for label, fn in CANDIDATES.items():
        f = lambda Q: fn(Q, S, sl)  # noqa: E731
        ad = float(jnp.sum(jax.grad(f)(P) * W))
        eps = 1e-6
        fd = float((f(P + eps * W) - f(P - eps * W)) / (2 * eps))
        rel = abs(fd - ad) / max(abs(fd), abs(ad), 1e-30)
        print(f"{label:<30}{float(f(P)):14.6f}{fd:14.6e}{ad:14.6e}{rel:11.2e}")


if __name__ == "__main__":
    main()
