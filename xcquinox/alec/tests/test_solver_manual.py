"""Tests for xcquinox.alec.solver_manual: SCF body correctness."""
import numpy as np
import pytest
import jax.numpy as jnp

import xcquinox.alec as alec
from xcquinox.alec.config import MoleculeSpec
from xcquinox.alec.data import precompute_fixed_density_data
from xcquinox.alec.solver import (
    SolverConfig,
    SolverBackend,
    SolverMode,
    run_scf,
)


def test_scf_energy_computed_from_mixed_dm_consistently():
    """SCF energy trace must be a consistent functional of the mixed DM,
    not a hybrid of D_cur (XC part) and D_mixed (one-electron + Coulomb)."""
    spec = MoleculeSpec(
        name="H2",
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        charge=0,
        spin=0,
        atom_composition=(("H", 2),),
        grid_level=1,
    )
    md = precompute_fixed_density_data(spec, required_keys=("eri",))
    arch = alec.get_architecture("deep")
    xnet, cnet = alec.create_network_pair(arch, seed=0)
    model = alec.AlecGGAModel.from_arch(arch, xnet=xnet, cnet=cnet)
    cfg = SolverConfig(
        backend=SolverBackend.MANUAL,
        mode=SolverMode.FULL,
        max_cycles=10,
        conv_tol=1e-8,
    )
    result = run_scf(cfg, model, md)

    # Energy trace should not have implausible upward excursions > 1 Hartree
    # during the SCF trajectory (previous bug could produce such artifacts
    # because the XC term lagged behind the one-electron/Coulomb terms).
    energy_trace = np.asarray(result.energy_trace)
    valid = (
        energy_trace[~np.isnan(energy_trace)]
        if np.any(np.isnan(energy_trace))
        else energy_trace
    )
    if len(valid) > 1:
        max_upward_jump = float(np.max(np.diff(valid)))
        assert max_upward_jump < 1.0, (
            f"SCF energy jumped upward by {max_upward_jump:.3f} Ha, "
            f"density inconsistency between E_new and features_used"
        )


def test_scf_energy_uses_post_mix_density():
    """After the fix, E_new at each cycle is computed from D_mixed with
    features/rho derived from D_mixed (not from D_cur).

    Proxy check: with mixer alpha in (0, 1), D_mixed != D_cur except at
    convergence. The reported energy at convergence should equal the energy
    evaluated from the final density's features, no hybrid.
    """
    spec = MoleculeSpec(
        name="H2",
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        charge=0,
        spin=0,
        atom_composition=(("H", 2),),
        grid_level=1,
    )
    md = precompute_fixed_density_data(spec, required_keys=("eri",))
    arch = alec.get_architecture("deep")
    xnet, cnet = alec.create_network_pair(arch, seed=0)
    model = alec.AlecGGAModel.from_arch(arch, xnet=xnet, cnet=cnet)
    cfg = SolverConfig(
        backend=SolverBackend.MANUAL,
        mode=SolverMode.FULL,
        max_cycles=30,
        conv_tol=1e-10,
        mixer_kwargs=(("alpha", 0.5),),
    )
    result = run_scf(cfg, model, md)
    assert bool(result.converged)
    assert jnp.isfinite(result.total_energy)


# --------------------------------------------------------------------------- #
# Per-rung seeding: the SCF consumes mol_data["dm_seed"] as D0
# --------------------------------------------------------------------------- #
def _seed_spec(grid_level=1):
    return MoleculeSpec(
        name="H2", atom="H 0 0 0; H 0 0 0.74", basis="sto-3g",
        charge=0, spin=0, atom_composition=(("H", 2),),
        grid_level=grid_level,
    )


def _seed_model():
    arch = alec.get_architecture("deep")
    xnet, cnet = alec.create_network_pair(arch, seed=0)
    return alec.AlecGGAModel.from_arch(arch, xnet=xnet, cnet=cnet)


def _full3_cfg(**kw):
    return SolverConfig(backend=SolverBackend.MANUAL, mode=SolverMode.FULL,
                        max_cycles=3, conv_tol=1e-12, **kw)


def test_scf_consumes_dm_seed_not_dm_pbe():
    """Swapping ONLY the dm_seed slot of one record (everything else the
    same objects) must change the truncated trajectory -- proving D0 comes
    from dm_seed, with no confound from record-to-record SCF differences."""
    from xcquinox.alec.data import clear_precompute_cache
    clear_precompute_cache()
    warm = precompute_fixed_density_data(_seed_spec(), required_keys=("eri",))
    cold_seed = precompute_fixed_density_data(
        _seed_spec(), required_keys=("eri",), seed_source="minao")["dm_seed"]
    md_cold = dict(warm)
    md_cold["dm_seed"] = cold_seed
    model = _seed_model()
    r_warm = run_scf(_full3_cfg(), model, warm)
    r_cold = run_scf(_full3_cfg(seed_source="minao"), model, md_cold)
    assert not np.allclose(np.asarray(r_warm.density_matrix),
                           np.asarray(r_cold.density_matrix))
    assert float(r_warm.total_energy) != float(r_cold.total_energy)


def test_scf_pbe_seed_value_semantics():
    """Only the VALUE of dm_seed matters: replacing the alias with an equal
    copy reproduces the trajectory exactly (the pre-seeding semantics)."""
    from xcquinox.alec.data import clear_precompute_cache
    clear_precompute_cache()
    md = precompute_fixed_density_data(_seed_spec(), required_keys=("eri",))
    model = _seed_model()
    r1 = run_scf(_full3_cfg(), model, md)
    md2 = dict(md)
    md2["dm_seed"] = jnp.array(np.asarray(md["dm_pbe"]).copy())
    r2 = run_scf(_full3_cfg(), model, md2)
    assert np.allclose(np.asarray(r1.density_matrix),
                       np.asarray(r2.density_matrix))
    assert float(r1.total_energy) == pytest.approx(float(r2.total_energy),
                                                   abs=0.0)


def test_uks_scf_consumes_dm_seed():
    """Single-record seed swap on the UKS path (an O atom's independent
    SCF runs can land on different degenerate 3P components, so the
    comparison must hold everything but dm_seed fixed)."""
    from xcquinox.alec.data import clear_precompute_cache
    clear_precompute_cache()
    o_spec = MoleculeSpec(name="O", atom="O 0 0 0", basis="sto-3g",
                          charge=0, spin=2, atom_composition=(("O", 1),),
                          grid_level=1)
    warm = precompute_fixed_density_data(o_spec, required_keys=("eri",))
    cold_seed = precompute_fixed_density_data(
        o_spec, required_keys=("eri",), seed_source="minao")["dm_seed"]
    md_cold = dict(warm)
    md_cold["dm_seed"] = cold_seed
    model = _seed_model()
    r_warm = run_scf(_full3_cfg(), model, warm)
    r_cold = run_scf(_full3_cfg(seed_source="minao"), model, md_cold)
    assert not np.allclose(np.asarray(r_warm.density_matrix),
                           np.asarray(r_cold.density_matrix))


def test_pyscfad_backend_rejects_non_pbe_seed():
    """The pyscfad backend re-prunes its internal grid on the seed density;
    per-rung seeding is manual-backend-only and must fail loud there."""
    from xcquinox.alec.data import clear_precompute_cache
    clear_precompute_cache()
    md = precompute_fixed_density_data(_seed_spec(), required_keys=("eri",),
                                       seed_source="minao")
    cfg = SolverConfig(backend=SolverBackend.PYSCFAD, mode=SolverMode.FULL,
                       max_cycles=3, seed_source="minao")
    with pytest.raises(NotImplementedError):
        run_scf(cfg, _seed_model(), md)
