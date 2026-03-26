# One-Shot DM Prediction Fix Plans

## Problem Summary
The one-shot DM prediction fails for atomic systems (H, O) due to near-degenerate
eigenvalues in the Fock matrix. The gradient through `jnp.linalg.eigh` involves
terms like `1/(λ_i - λ_j)`, which explode when eigenvalues are nearly degenerate.

- H atom: 3 exactly degenerate p-orbitals (gap = 0)
- O atom: 8 near-degenerate pairs (smallest gap = 5.8e-5 Ha)
- H2O: No severe degeneracies (smallest gap = 0.0124 Ha)

---

## Plan 1: Exclude Atoms from One-Shot DM Matching

**Rationale:** Atoms have degenerate orbitals by symmetry. Only match DMs for
molecules where eigenvalue gaps are larger.

### Changes Required

#### 1.1 Modify `make_loss_B_ae_dm` (notebook cell ~27)

**Current code:**
```python
def make_loss_B_ae_dm(arch_config, dm_weight=0.1):
    @eqx.filter_value_and_grad
    def loss_fn(model, mols, refs_arr):
        # ... AE calculation ...

        # DM Matching (one-shot prediction)
        dm_loss = 0.0
        for idx, (name, mol) in enumerate(training_mols.items()):
            dm_pred, E_pred = oneshot_dm_prediction(model, name, arch_config)
            dm_ccsd = refs['ccsd']['dms'][name]
            dm_diff = dm_pred - dm_ccsd
            mol_dm_loss = jnp.sum(dm_diff**2)
            dm_loss += mol_dm_loss

        total_loss = ae_loss + dm_weight * dm_loss
        return total_loss
    return loss_fn
```

**New code:**
```python
def make_loss_B_ae_dm(arch_config, dm_weight=0.1, molecules_only=True):
    """AE + DM matching loss.

    Args:
        arch_config: Architecture configuration dict
        dm_weight: Weight for DM matching loss
        molecules_only: If True, only match DMs for molecules (skip atoms)
    """
    # Define which systems are atoms (have degenerate orbitals)
    ATOMIC_SYSTEMS = {'H', 'O', 'C', 'N', 'F', 'S', 'Cl'}  # Extendable

    @eqx.filter_value_and_grad
    def loss_fn(model, mols, refs_arr):
        # === Atomization Energy (fixed-density) ===
        E_H = compute_total_energy_nn(model, 'H', arch_config)
        E_O = compute_total_energy_nn(model, 'O', arch_config)
        E_H2O = compute_total_energy_nn(model, 'H2O', arch_config)

        AE_pred = E_H2O - 2*E_H - E_O
        AE_target = refs['lit']['H2O_AE']
        ae_loss = (AE_pred - AE_target)**2 / (AE_target**2 + 1e-8)

        # === DM Matching (molecules only to avoid degeneracy issues) ===
        dm_loss = 0.0
        dm_count = 0
        for name in training_mols.keys():
            # Skip atomic systems if molecules_only=True
            if molecules_only and name in ATOMIC_SYSTEMS:
                jax.debug.print("    Skipping {n} (atomic system)", n=name)
                continue

            jax.debug.print("    Processing {n} for DM matching...", n=name)
            dm_pred, E_pred = oneshot_dm_prediction(model, name, arch_config)
            dm_ccsd = refs['ccsd']['dms'][name]

            # Frobenius norm of difference, normalized by system size
            dm_diff = dm_pred - dm_ccsd
            nao = dm_ccsd.shape[-1]
            mol_dm_loss = jnp.sum(dm_diff**2) / (nao * nao)  # Normalized
            dm_loss += mol_dm_loss
            dm_count += 1

            jax.debug.print("      {n}: dm_loss={l:.6f}", n=name, l=mol_dm_loss)

        # Average over molecules processed
        if dm_count > 0:
            dm_loss = dm_loss / dm_count

        total_loss = ae_loss + dm_weight * dm_loss

        jax.debug.print("  TOTAL: ae_loss={a:.6f}, dm_loss={d:.6f}, total={t:.6f}",
                       a=ae_loss, d=dm_loss, t=total_loss)

        return total_loss
    return loss_fn
```

#### 1.2 Modify `make_loss_D2_delta_dm` similarly (notebook cell ~30)

Same pattern: add `molecules_only=True` parameter and skip atomic systems.

#### 1.3 Update parallel worker if needed

File: `scripts/parallel_train_worker.py`

The worker uses these loss functions, so changes propagate automatically if
the notebook cell definitions are re-exported or if the worker imports from
the module.

### Testing

1. Run single training with B_ae_dm approach on 'shallow' architecture
2. Verify no NaN errors
3. Check that DM loss is reasonable (should be much smaller now)
4. Run full parallel training for Phase 2

### Expected Outcome

- DM matching only on H2O (1 molecule)
- Loss values should be stable (no gradient explosion from atomic degeneracies)
- Training should complete without NaNs

---

## Plan 2: Increase Degeneracy-Breaking Perturbation

**Rationale:** If Plan 1 still has issues with molecular near-degeneracies,
add a larger perturbation to the Fock matrix before diagonalization.

### Changes Required

#### 2.1 Modify `oneshot_dm_prediction_fast` (notebook cell ~25)

**Current code:**
```python
# Add small perturbation to break degeneracies
fock_orth = fock_orth + 1e-10 * jnp.eye(nao)
mo_energy, mo_coeff_orth = jnp.linalg.eigh(fock_orth)
```

**New code:**
```python
# Add perturbation to break degeneracies
# 1e-6 Ha = 0.03 meV, small enough to not affect physical results
# but large enough to regularize gradient computation
DEGENERACY_PERTURBATION = 1e-6

# Use a non-uniform perturbation to break symmetry more effectively
# This ensures no two diagonal elements are exactly equal
perturbation = DEGENERACY_PERTURBATION * jnp.diag(jnp.arange(1, nao + 1, dtype=fock_orth.dtype) / nao)
fock_orth = fock_orth + perturbation

mo_energy, mo_coeff_orth = jnp.linalg.eigh(fock_orth)
```

#### 2.2 Make perturbation configurable

Add parameter to `oneshot_dm_prediction_fast`:

```python
def oneshot_dm_prediction_fast(model, name, arch_config, degeneracy_eps=1e-6):
    """Fast one-shot DM prediction.

    Args:
        degeneracy_eps: Perturbation magnitude for breaking eigenvalue degeneracies.
                       Larger values = more stable gradients but less accurate energies.
    """
    # ... existing code ...

    perturbation = degeneracy_eps * jnp.diag(jnp.arange(1, nao + 1, dtype=fock_orth.dtype) / nao)
    fock_orth = fock_orth + perturbation
```

### Testing

1. Test with degeneracy_eps = 1e-6 (default)
2. If still unstable, try 1e-5 or 1e-4
3. Monitor orbital energy shifts to ensure physical accuracy isn't compromised

### Expected Outcome

- Eigenvalue gaps increased by ~1e-6 Ha minimum
- Gradient amplification reduced from 17,000x to ~1,000x or less
- More stable training

---

## Plan 3: Löwdin (Symmetric) Orthogonalization

**Rationale:** Cholesky-based transformation can amplify errors when the overlap
matrix is ill-conditioned. Löwdin orthogonalization using SVD is more stable.

### Changes Required

#### 3.1 Replace Cholesky with Löwdin in `oneshot_dm_prediction_fast`

**Current code (Cholesky-based):**
```python
# Cholesky: S = LL^T
overlap_reg = overlap + 1e-10 * jnp.eye(nao)
L = jnp.linalg.cholesky(overlap_reg)
L_inv = jax.scipy.linalg.solve_triangular(L, jnp.eye(nao), lower=True)

# Transform: F' = L^{-1} F L^{-T}
fock_orth = L_inv @ fock @ L_inv.T

# ... diagonalize ...

# Back-transform: C = L^{-T} C'
mo_coeff = L_inv.T @ mo_coeff_orth
```

**New code (Löwdin-based):**
```python
def lowdin_orthogonalization(overlap, regularization=1e-8):
    """Compute S^{-1/2} using eigendecomposition (Löwdin orthogonalization).

    More numerically stable than Cholesky for ill-conditioned overlap matrices.

    Args:
        overlap: Overlap matrix S
        regularization: Minimum eigenvalue (smaller eigenvalues are clipped)

    Returns:
        S_inv_sqrt: S^{-1/2} matrix
        S_sqrt: S^{1/2} matrix (for back-transformation)
    """
    # Eigendecomposition: S = U Λ U^T
    eigvals, U = jnp.linalg.eigh(overlap)

    # Clip small eigenvalues for numerical stability
    eigvals_safe = jnp.maximum(eigvals, regularization)

    # S^{-1/2} = U Λ^{-1/2} U^T
    S_inv_sqrt = U @ jnp.diag(1.0 / jnp.sqrt(eigvals_safe)) @ U.T

    # S^{1/2} = U Λ^{1/2} U^T (for back-transformation)
    S_sqrt = U @ jnp.diag(jnp.sqrt(eigvals_safe)) @ U.T

    return S_inv_sqrt, S_sqrt


def oneshot_dm_prediction_fast(model, name, arch_config, use_lowdin=True):
    """Fast one-shot DM prediction.

    Args:
        use_lowdin: If True, use Löwdin orthogonalization (more stable).
                   If False, use Cholesky (faster but less stable).
    """
    # ... get data and compute Fock matrix ...

    if use_lowdin:
        # Löwdin orthogonalization (symmetric, more stable)
        S_inv_sqrt, S_sqrt = lowdin_orthogonalization(overlap, regularization=1e-8)

        # Transform: F' = S^{-1/2} F S^{-1/2}
        fock_orth = S_inv_sqrt @ fock @ S_inv_sqrt

        # Diagonalize
        fock_orth = fock_orth + 1e-6 * jnp.diag(jnp.arange(1, nao + 1) / nao)
        mo_energy, mo_coeff_orth = jnp.linalg.eigh(fock_orth)

        # Back-transform: C = S^{-1/2} C' (Löwdin MOs)
        mo_coeff = S_inv_sqrt @ mo_coeff_orth
    else:
        # Original Cholesky-based code
        # ... existing implementation ...
```

### Key Differences

| Aspect | Cholesky | Löwdin |
|--------|----------|--------|
| Factorization | S = LL^T | S = UΛU^T |
| Transform | L^{-1} F L^{-T} | S^{-1/2} F S^{-1/2} |
| Back-transform | L^{-T} C' | S^{-1/2} C' |
| Stability | Sensitive to conditioning | Eigenvalue clipping possible |
| Symmetry | Asymmetric | Symmetric |

### Testing

1. Verify MO orthonormality: C^T S C = I
2. Compare energies with Cholesky version (should match within 1e-8)
3. Test gradient stability on O atom

### Expected Outcome

- More stable for ill-conditioned overlap matrices
- Explicit control over small eigenvalue handling
- Symmetric transformation preserves more numerical precision

---

## Plan 4: Regularized Eigenvalue Gradient

**Rationale:** The core issue is in the backward pass of `eigh`, not the forward
pass. Use a custom VJP that regularizes gradients for near-degenerate eigenvalues.

### Changes Required

#### 4.1 Create custom `safe_eigh` function

```python
import jax
from jax import custom_vjp
import jax.numpy as jnp


@custom_vjp
def safe_eigh(a, min_gap=1e-5):
    """Eigendecomposition with regularized gradients for near-degenerate eigenvalues.

    Args:
        a: Symmetric matrix to diagonalize
        min_gap: Minimum eigenvalue gap for gradient computation.
                Smaller gaps are clipped to this value.

    Returns:
        eigenvalues, eigenvectors (same as jnp.linalg.eigh)
    """
    return jnp.linalg.eigh(a)


def safe_eigh_fwd(a, min_gap):
    """Forward pass: standard eigendecomposition."""
    eigvals, eigvecs = jnp.linalg.eigh(a)
    return (eigvals, eigvecs), (eigvals, eigvecs, min_gap)


def safe_eigh_bwd(res, g):
    """Backward pass: regularized gradient for near-degenerate eigenvalues.

    Standard eigh gradient:
        ∂L/∂A = U @ (F ⊙ (U^T @ G_v @ U)) @ U^T + diag terms
    where F_ij = 1/(λ_i - λ_j) for i ≠ j

    We regularize F_ij by clipping small denominators.
    """
    eigvals, eigvecs, min_gap = res
    g_eigvals, g_eigvecs = g

    n = eigvals.shape[0]

    # Compute eigenvalue differences
    # diff[i,j] = λ_i - λ_j
    diff = eigvals[:, None] - eigvals[None, :]

    # Regularize: ensure |diff| >= min_gap for i != j
    # This prevents gradient explosion for near-degenerate eigenvalues
    diff_sign = jnp.sign(diff)
    diff_abs = jnp.abs(diff)
    diff_regularized = diff_sign * jnp.maximum(diff_abs, min_gap)

    # Set diagonal to 1 to avoid division by zero (diagonal terms handled separately)
    diff_regularized = diff_regularized.at[jnp.diag_indices(n)].set(1.0)

    # F matrix: F_ij = 1/(λ_i - λ_j) for i != j, 0 on diagonal
    F = 1.0 / diff_regularized
    F = F.at[jnp.diag_indices(n)].set(0.0)

    # Gradient contribution from eigenvectors
    # ∂L/∂A (from eigvecs) = U @ (F ⊙ (U^T @ G_v @ U)) @ U^T
    if g_eigvecs is not None:
        middle = eigvecs.T @ g_eigvecs
        middle = F * (middle + middle.T) / 2  # Symmetrize
        grad_a = eigvecs @ middle @ eigvecs.T
    else:
        grad_a = jnp.zeros_like(eigvecs @ eigvecs.T)

    # Gradient contribution from eigenvalues
    # ∂L/∂A (from eigvals) = U @ diag(g_λ) @ U^T
    if g_eigvals is not None:
        grad_a = grad_a + eigvecs @ jnp.diag(g_eigvals) @ eigvecs.T

    # Symmetrize the gradient (A is symmetric)
    grad_a = (grad_a + grad_a.T) / 2

    return (grad_a, None)  # None for min_gap gradient


safe_eigh.defvjp(safe_eigh_fwd, safe_eigh_bwd)
```

#### 4.2 Use `safe_eigh` in `oneshot_dm_prediction_fast`

```python
def oneshot_dm_prediction_fast(model, name, arch_config):
    # ... build Fock matrix ...

    # Use safe_eigh instead of jnp.linalg.eigh
    mo_energy, mo_coeff_orth = safe_eigh(fock_orth, min_gap=1e-5)

    # ... rest of the function ...
```

### Testing

1. Verify forward pass matches `jnp.linalg.eigh` exactly
2. Test backward pass on matrix with known degenerate eigenvalues
3. Compare gradients with finite differences
4. Test on O atom (8 near-degenerate pairs)

### Expected Outcome

- Forward pass: identical to standard eigh
- Backward pass: gradients clipped when eigenvalue gaps < min_gap
- Maximum gradient amplification: 1/min_gap = 100,000 (for min_gap=1e-5)
- Stable training even with atomic systems

---

## Implementation Order

1. **Plan 1** (Exclude atoms): Simplest, minimal code changes
2. **Plan 2** (Larger perturbation): Quick fix if Plan 1 isn't enough
3. **Plan 3** (Löwdin): More robust orthogonalization
4. **Plan 4** (Regularized gradient): Most comprehensive but most complex

## Success Criteria

- Training completes without NaN errors
- Loss decreases monotonically (or with reasonable noise)
- Final DM RMSE < 0.1 for molecules
- AE error < 5 kcal/mol

## Rollback Plan

If all plans fail, revert to fixed-density training only (approaches A, D1)
and investigate alternative DM matching formulations (e.g., DM fitting in
post-processing rather than during training).
