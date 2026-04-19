"""Tests for jax.grad through the pyscfad backend.

pyscfad's custom ``eigh_gen_p`` primitive has no CUDA kernel, so
``jax.grad`` through pyscfad fails on GPU with:
    UNIMPLEMENTED: No registered implementation for custom call to
    cusolver_sygvd_ffi

``run_pyscfad_scf`` wraps its body in a ``jax.default_device(cpu)``
context to pin the pyscfad subgraph to CPU. This test exercises
``eqx.filter_grad`` through a FIXED_J pyscfad SCF on whatever device JAX
defaults to (GPU on CUDA hosts) — the CPU-pin context is responsible for
producing finite, nonzero gradients w.r.t. NN parameters regardless of
the host default device.
"""


def test_grad_through_pyscfad_fixed_j_is_finite():
    """eqx.filter_grad on a loss that runs FIXED_J via pyscfad must produce
    finite, nonzero gradients w.r.t. NN params (CPU pinning handles eigh_gen)."""
    import jax
    import jax.numpy as jnp
    import equinox as eqx

    import xcquinox.alec as alec
    from xcquinox.alec.data import precompute_fixed_density_data
    from xcquinox.alec.solver import (
        SolverConfig, SolverBackend, SolverMode, run_scf,
    )
    from xcquinox.alec.tests.fixtures.molecules import h2_molecule

    # Use the same fixture as the passing pyscfad FIXED_J test; at
    # grid_level=1 the precompute (pyscf) and pyscfad grids mismatch by
    # ~16 points, which is a separate pre-existing issue independent of
    # the CPU-pin concern that Task 18 targets.
    md = precompute_fixed_density_data(h2_molecule(), required_keys=("eri",))
    arch = alec.get_architecture("deep")
    xnet, cnet = alec.create_network_pair(arch, seed=0)
    model = alec.AlecGGAModel.from_arch(arch, xnet=xnet, cnet=cnet)
    cfg = SolverConfig(
        backend=SolverBackend.PYSCFAD, mode=SolverMode.FIXED_J,
        max_cycles=2, conv_tol=1e-4,
    )

    def loss_fn(model):
        result = run_scf(cfg, model, md)
        # Use total_energy as the scalar — it depends explicitly on the
        # XC functional output and thus produces a meaningful gradient
        # flowing through the pyscfad eval_xc callback. (A loss based on
        # the converged density_matrix alone yields near-zero gradients
        # because the SCF fixed-point is nearly stationary w.r.t. model
        # parameters at typical random init.)
        return result.total_energy

    grad_model = eqx.filter_grad(loss_fn)(model)
    # Traverse gradient tree for finite, nonzero leaves.
    leaves = jax.tree_util.tree_leaves(grad_model)
    arr_leaves = [l for l in leaves if hasattr(l, 'shape')]
    # Need at least one non-trivial leaf.
    assert len(arr_leaves) > 0
    all_finite = all(bool(jnp.all(jnp.isfinite(l))) for l in arr_leaves)
    assert all_finite, "gradient contains non-finite values"
    max_abs = max(float(jnp.max(jnp.abs(l))) for l in arr_leaves if l.size > 0)
    assert max_abs > 0, f"gradient is all zero: max|grad|={max_abs}"
    # Sanity-check that the gradient is not just floating-point noise —
    # a meaningful NN gradient through a converged SCF should be well
    # above the 1e-20 regime.
    assert max_abs > 1e-8, (
        f"gradient suspiciously small (likely dead path): max|grad|={max_abs}"
    )
