"""Phase 0 discovery probes for SCF solver implementation.

Each test here is a fail-fast probe verifying a spec assumption before
dependent phases are unblocked. Failures here mean the spec must be
revised, not that the test should be weakened.

See: docs/superpowers/plans/2026-04-14-alec-scf-solver-and-ref-density-rename.md
"""
import pytest
import jax
import jax.numpy as jnp

from xcquinox.alec.config import ArchitectureConfig
from xcquinox.alec.models import AlecGGAModel
from xcquinox.alec.data import precompute_fixed_density_data
from xcquinox.alec.tests.fixtures.molecules import h2_molecule


def _make_h2_model_and_data(seed: int = 0):
    arch = ArchitectureConfig(
        name="probe", depth=2, nodes=8, attention=False,
        descriptors=(), x_constraints=(), c_constraints=(),
        double_lob_clamp_allowed=False,
    )
    model = AlecGGAModel.from_arch(arch, seed=seed)
    data = precompute_fixed_density_data(h2_molecule())
    return model, data


def test_p01_compute_vxc_nn_flows_grad_through_dynamic_rho():
    """P0.1: compute_vxc_nn must accept dynamic rho/sigma and let jax.grad
    flow through. Otherwise the manual SCF backend's D → rho → F → D' loop
    is not differentiable."""
    from xcquinox.alec.oneshot import compute_vxc_nn
    from xcquinox.alec.descriptors import assemble_descriptor_features

    model, data = _make_h2_model_and_data()
    features = assemble_descriptor_features(model.descriptors, data)
    ao_grid = data["ao_grid"]
    grid_weights = data["grid_weights"]

    def scalar_from_vxc(rho_dyn, sigma_dyn):
        # PRE-07: compute_vxc_nn refuses to silently drop the GGA v_sigma
        # term, so the LDA-only v_rho path (which this gradient-flow probe
        # exercises) must be requested explicitly via lda_only=True.
        vxc = compute_vxc_nn(
            model, rho_dyn, sigma_dyn, features, ao_grid, grid_weights,
            lda_only=True,
        )
        return jnp.sum(vxc ** 2)

    rho0 = data["rho_grid"]
    sigma0 = data["sigma_grid"]

    grad_rho = jax.grad(scalar_from_vxc, argnums=0)(rho0, sigma0)
    grad_sigma = jax.grad(scalar_from_vxc, argnums=1)(rho0, sigma0)

    assert jnp.all(jnp.isfinite(grad_rho))
    assert jnp.all(jnp.isfinite(grad_sigma))
    assert grad_rho.shape == rho0.shape
    assert grad_sigma.shape == sigma0.shape


def test_p02_mol_data_has_metadata_for_pyscfad_rebuild():
    """P0.2: mol_data must contain enough metadata to rebuild a pyscf.gto.Mole
    object. Required: atom spec, basis, charge, spin."""
    _, data = _make_h2_model_and_data()
    assert "mol_metadata" in data, (
        "mol_data lacks 'mol_metadata' — extend precompute_fixed_density_data "
        "to stash atom/basis/charge/spin for pyscfad backend rebuild."
    )
    md = data["mol_metadata"]
    for k in ("atom", "basis", "charge", "spin"):
        assert k in md, f"mol_metadata missing required key {k!r}"
    assert isinstance(md["atom"], str)
    assert isinstance(md["basis"], str)
    assert isinstance(md["charge"], int)
    assert isinstance(md["spin"], int)


def test_p03_pyscfad_get_fock_sees_current_dm_per_cycle():
    """P0.3: pyscfad's SCF driver must invoke mf.get_fock with a DM kwarg
    at the start of each SCF cycle, so the REASSEMBLE closure can read
    the DM before the XC callback runs."""
    import pyscfad.gto
    import pyscfad.dft

    # Force CPU — pyscfad's custom eigh_gen primitive lacks a CUDA impl,
    # so this test fails when JAX defaults to GPU in the full suite.
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        return _run_p03_get_fock_probe()


def _run_p03_get_fock_probe():
    import pyscfad.gto
    import pyscfad.dft

    mol = pyscfad.gto.Mole()
    mol.atom = "H 0 0 0; H 0 0 0.74"
    mol.basis = "sto-3g"
    mol.charge = 0
    mol.spin = 0
    mol.verbose = 0
    mol.build()
    mf = pyscfad.dft.RKS(mol)
    mf.xc = "pbe"
    mf.max_cycle = 3
    mf.conv_tol = 1e-6

    dm_observations = []
    original_get_fock = mf.get_fock

    def patched(*args, **kwargs):
        dm_arg = kwargs.get("dm", None)
        if dm_arg is None and len(args) >= 4:
            dm_arg = args[3]
        dm_observations.append(dm_arg is not None)
        return original_get_fock(*args, **kwargs)

    mf.get_fock = patched
    mf.kernel()

    assert len(dm_observations) >= 2, (
        f"get_fock invoked {len(dm_observations)} times — expected ≥2 "
        f"across an SCF with max_cycle=3"
    )
    assert any(dm_observations), (
        "get_fock was never invoked with a DM argument; "
        "REASSEMBLE policy cannot read the current DM from this hook"
    )


def test_p04_pyscfad_get_j_monkey_patch_propagates():
    """P0.4: monkey-patching mf.get_j must actually intercept J in the Fock
    build. Use a sentinel offset that, if bypassed, leaves the total energy
    indistinguishable from the baseline."""
    # Force CPU — pyscfad's custom eigh_gen primitive lacks a CUDA impl.
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        return _run_p04_get_j_probe()


def _run_p04_get_j_probe():
    import pyscfad.gto
    import pyscfad.dft
    import numpy as np

    mol = pyscfad.gto.Mole()
    mol.atom = "H 0 0 0; H 0 0 0.74"
    mol.basis = "sto-3g"
    mol.charge = 0
    mol.spin = 0
    mol.verbose = 0
    mol.build()

    # Baseline: unpatched PBE SCF
    mf_ref = pyscfad.dft.RKS(mol)
    mf_ref.xc = "pbe"
    mf_ref.max_cycle = 30
    mf_ref.kernel()
    e_ref = float(mf_ref.e_tot)

    # Patched: get_j returns a clearly-offset matrix
    mf_patched = pyscfad.dft.RKS(mol)
    mf_patched.xc = "pbe"
    mf_patched.max_cycle = 30
    nao = mol.nao
    sentinel = 1e-3 * np.eye(nao)
    original_get_j = mf_patched.get_j

    def patched_get_j(*args, **kwargs):
        j_real = original_get_j(*args, **kwargs)
        return np.asarray(j_real) + sentinel

    mf_patched.get_j = patched_get_j
    mf_patched.kernel()
    e_patched = float(mf_patched.e_tot)

    assert abs(e_patched - e_ref) > 1e-6, (
        f"get_j monkey-patch did not propagate: "
        f"e_patched={e_patched} e_ref={e_ref} |Δ|={abs(e_patched - e_ref):.2e}. "
        f"Pyscfad bypasses the overridden get_j — fixed_j pyscfad mode "
        f"must fall back to manual backend."
    )
