"""Tests for xcquinox.pipeline.solver_manual: SCF body correctness."""
import numpy as np
import pytest
import jax.numpy as jnp

import xcquinox.pipeline as pipeline
from xcquinox.pipeline.config import MoleculeSpec
from xcquinox.pipeline.data import precompute_fixed_density_data
from xcquinox.pipeline.solver import (
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
    arch = pipeline.get_architecture("deep")
    xnet, cnet = pipeline.create_network_pair(arch, seed=0)
    model = pipeline.AlecGGAModel.from_arch(arch, xnet=xnet, cnet=cnet)
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
    arch = pipeline.get_architecture("deep")
    xnet, cnet = pipeline.create_network_pair(arch, seed=0)
    model = pipeline.AlecGGAModel.from_arch(arch, xnet=xnet, cnet=cnet)
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
    arch = pipeline.get_architecture("deep")
    xnet, cnet = pipeline.create_network_pair(arch, seed=0)
    return pipeline.AlecGGAModel.from_arch(arch, xnet=xnet, cnet=cnet)


def _full3_cfg(**kw):
    return SolverConfig(backend=SolverBackend.MANUAL, mode=SolverMode.FULL,
                        max_cycles=3, conv_tol=1e-12, **kw)


def test_scf_consumes_dm_seed_not_dm_pbe():
    """Swapping ONLY the dm_seed slot of one record (everything else the
    same objects) must change the truncated trajectory -- proving D0 comes
    from dm_seed, with no confound from record-to-record SCF differences."""
    from xcquinox.pipeline.data import clear_precompute_cache
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
    from xcquinox.pipeline.data import clear_precompute_cache
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
    from xcquinox.pipeline.data import clear_precompute_cache
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
    from xcquinox.pipeline.data import clear_precompute_cache
    clear_precompute_cache()
    md = precompute_fixed_density_data(_seed_spec(), required_keys=("eri",),
                                       seed_source="minao")
    cfg = SolverConfig(backend=SolverBackend.PYSCFAD, mode=SolverMode.FULL,
                       max_cycles=3, seed_source="minao")
    with pytest.raises(NotImplementedError):
        run_scf(cfg, _seed_model(), md)



# --------------------------------------------------------------------------- #
# freeze_on_convergence (spec C4): whether the truncated SCF stops advancing
# once the criterion has fired, or runs every configured cycle as dpyscf's
# loop does.
# --------------------------------------------------------------------------- #
def _freeze_cfg(tol, **kw):
    """A four-cycle manual solver at the linear mixer, parametrized by the
    convergence tolerance."""
    return SolverConfig(backend=SolverBackend.MANUAL, mode=SolverMode.FULL,
                        max_cycles=4, conv_tol=tol,
                        mixer_kwargs=(("alpha", 0.5),), **kw)


def _assert_freeze_semantics(md, max_cycles=4):
    """The two flag settings on a toy seeded at its own fixed point.

    These sto-3g toys sit on the fixed point of the untrained network from the
    PBE seed to about 1e-11 (H2) and 1e-9 (O) Hartree per cycle, so an energy
    criterion at any looser tolerance fires at the FIRST cycle: ``conv_tol``
    1e3 makes the firing independent of that residue, and 1e-12 is the control
    whose criterion never fires inside four cycles.

    With the freeze the scan stops advancing there (one cycle counted, the
    energy trace held at its first entry). Without it the run is bit-identical
    to the control -- same cycle count, same trace, same density, so the
    density, the energy, the counter and the mixer state all advanced
    unconditionally -- while the ``converged`` flag reports the last cycle's
    criterion, which at this tolerance fires on every cycle, so it still
    reports the convergence the control never reached.
    """
    model = _seed_model()
    frozen = run_scf(_freeze_cfg(1e3, freeze_on_convergence=True), model, md)
    running = run_scf(_freeze_cfg(1e3, freeze_on_convergence=False), model, md)
    never = run_scf(_freeze_cfg(1e-12), model, md)

    assert bool(frozen.converged) and bool(running.converged)
    assert not bool(never.converged)          # the control's criterion never fires
    assert int(frozen.cycles_run) == 1, int(frozen.cycles_run)
    assert int(running.cycles_run) == max_cycles, int(running.cycles_run)
    assert int(never.cycles_run) == max_cycles, int(never.cycles_run)

    tf = np.asarray(frozen.energy_trace)
    tr = np.asarray(running.energy_trace)
    tn = np.asarray(never.energy_trace)
    assert tf.shape == tr.shape == tn.shape == (max_cycles,)
    assert np.ptp(tf) == 0.0, tf              # held at the converged cycle
    assert np.ptp(tn) > 0.0, tn               # the control keeps moving
    np.testing.assert_array_equal(tr, tn)     # the unfrozen run IS the control

    d_frozen = np.asarray(frozen.density_matrix)
    d_running = np.asarray(running.density_matrix)
    d_never = np.asarray(never.density_matrix)
    np.testing.assert_allclose(d_running, d_never, rtol=0, atol=1e-12)
    # the frozen density stopped three cycles earlier, which the residue of
    # those cycles separates from the control's by far more than the identity
    # tolerance above.
    assert np.linalg.norm(d_frozen - d_never) > 1e-9, (
        np.linalg.norm(d_frozen - d_never))


def test_freeze_on_convergence_governs_the_rks_cycle_count():
    """RKS: with the flag set (the default, every campaign through v7) the scan
    stops advancing at the cycle the criterion fires on; with it clear the
    truncated SCF runs all ``max_cycles``, which is what the reference protocol
    does and what the tail-weighted loss is defined over."""
    from xcquinox.pipeline.data import clear_precompute_cache
    clear_precompute_cache()
    md = precompute_fixed_density_data(_seed_spec(), required_keys=("eri",))
    _assert_freeze_semantics(md)


def test_freeze_on_convergence_governs_the_uks_cycle_count():
    """UKS: the same, on the spin-resolved body -- the two SCF bodies carry
    separate copies of the freeze, so a flag honored in one of them leaves
    every open-shell species (every atom anchor of the pool) on the old
    trajectory while the record says otherwise."""
    from xcquinox.pipeline.data import clear_precompute_cache
    clear_precompute_cache()
    o_spec = MoleculeSpec(name="O", atom="O 0 0 0", basis="sto-3g",
                          charge=0, spin=2, atom_composition=(("O", 1),),
                          grid_level=1)
    md = precompute_fixed_density_data(o_spec, required_keys=("eri",))
    _assert_freeze_semantics(md)


def test_solver_config_describe_reports_freeze_on_convergence():
    """``describe()`` is what ``train_metadata.json`` records the solver by, so
    the flag has to appear there: two runs differing only in it would otherwise
    be indistinguishable in their artifacts. Default True keeps every existing
    run's trajectory."""
    assert SolverConfig().describe()["freeze_on_convergence"] is True
    cfg = SolverConfig(backend=SolverBackend.MANUAL, mode=SolverMode.FULL,
                       max_cycles=3, freeze_on_convergence=False)
    assert cfg.freeze_on_convergence is False
    assert cfg.describe()["freeze_on_convergence"] is False


# --------------------------------------------------------------------------- #
# The reported flags: ``converged`` is the LAST cycle's criterion, the latched
# OR is ``converged_ever``; ``dm_residual`` is the last cycle's density change.
# --------------------------------------------------------------------------- #
def _flags_over_energies(energies, tol, freeze):
    """Drive ``_convergence_flags`` over consecutive energies as the SCF body
    does; returns the final ``(converged, converged_ever)``."""
    from xcquinox.pipeline.solver import EnergyConvergence
    from xcquinox.pipeline.solver_manual import _convergence_flags
    crit = EnergyConvergence(tol=tol)
    ever = jnp.bool_(False)
    conv = jnp.bool_(False)
    for e_prev, e_cur in zip(energies[:-1], energies[1:]):
        is_conv = crit.is_converged_from_energies(jnp.asarray(e_prev),
                                                  jnp.asarray(e_cur))
        conv, ever = _convergence_flags(ever, is_conv, freeze)
    return bool(conv), bool(ever)


def test_convergence_flags_an_early_dip_does_not_report_a_moving_end():
    """Energies 0, 1e-8, 1, 2 at tol 1e-6: the criterion fires on the first
    step and never again. Without the freeze every cycle runs, so the run
    ends unconverged (converged False) though it converged once (converged_ever
    True); with the freeze the latched flag stays True, as the state would
    have stopped advancing at the first step."""
    energies = [0.0, 1e-8, 1.0, 2.0]
    assert _flags_over_energies(energies, 1e-6, freeze=False) == (False, True)
    assert _flags_over_energies(energies, 1e-6, freeze=True) == (True, True)
    # a run that converges at the end reports both flags True either way
    assert _flags_over_energies([0.0, 1.0, 1.0 + 1e-8], 1e-6, False) == (True, True)
    # a run that never converges reports both False either way
    assert _flags_over_energies([0.0, 1.0, 2.0], 1e-6, False) == (False, False)
    assert _flags_over_energies([0.0, 1.0, 2.0], 1e-6, True) == (False, False)


def test_rks_result_reports_the_flags_and_the_density_residual():
    """A real manual RKS SCF: ``converged`` agrees with the last step of its
    own energy trace, ``converged_ever`` is implied by any step of the trace
    below the tolerance (the trace omits the initial energy, so the first
    cycle's criterion is not visible from it and the OR is only bracketed
    from below), ``dm_residual`` is finite and, on a one-cycle run, is
    exactly max|D_final - D_seed|."""
    from xcquinox.pipeline.data import clear_precompute_cache
    clear_precompute_cache()
    md = precompute_fixed_density_data(_seed_spec(), required_keys=("eri",))
    model = _seed_model()
    for tol in (1e3, 1e-12):
        res = run_scf(_freeze_cfg(tol, freeze_on_convergence=False), model, md)
        tr = np.asarray(res.energy_trace)
        steps = np.abs(np.diff(tr)) < tol            # cycles 2..max_cycles
        assert bool(res.converged) == bool(steps[-1]), (tol, tr)
        assert bool(res.converged_ever) >= bool(steps.any()), (tol, tr)
        assert bool(res.converged_ever) >= bool(res.converged)
        assert np.isfinite(float(res.dm_residual))
        assert float(res.dm_residual) > 0.0
    one = run_scf(SolverConfig(backend=SolverBackend.MANUAL, mode=SolverMode.FULL,
                               max_cycles=1, conv_tol=1e-12,
                               mixer_kwargs=(("alpha", 0.5),)), model, md)
    expected = float(np.max(np.abs(np.asarray(one.density_matrix)
                                   - np.asarray(md["dm_seed"]))))
    assert float(one.dm_residual) == expected
    # under the freeze the two flags coincide
    frozen = run_scf(_freeze_cfg(1e3, freeze_on_convergence=True), model, md)
    assert bool(frozen.converged) and bool(frozen.converged_ever)
    assert np.isfinite(float(frozen.dm_residual))


# --------------------------------------------------------------------------- #
# Fock noise: fresh per update and per molecule when the keys are carried.
# --------------------------------------------------------------------------- #
def _noise_cfg(**kw):
    return SolverConfig(backend=SolverBackend.MANUAL, mode=SolverMode.FULL,
                        max_cycles=2, conv_tol=1e-12,
                        fock_noise_std=1.0e-5, fock_noise_seed=0, **kw)


def test_fock_noise_draw_keys_on_update_and_molecule_when_present():
    import jax
    from xcquinox.pipeline.solver_manual import _fock_noise
    cfg = _noise_cfg()
    shape = (3, 3)
    u0, u1 = jnp.int32(0), jnp.int32(1)
    m0, m1 = jnp.int32(0), jnp.int32(1)
    base = np.asarray(_fock_noise(cfg, jnp.int32(1), shape, 0, update=u0, mol_id=m0))
    assert not np.array_equal(base, np.asarray(
        _fock_noise(cfg, jnp.int32(1), shape, 0, update=u1, mol_id=m0)))
    assert not np.array_equal(base, np.asarray(
        _fock_noise(cfg, jnp.int32(1), shape, 0, update=u0, mol_id=m1)))
    np.testing.assert_array_equal(base, np.asarray(
        _fock_noise(cfg, jnp.int32(1), shape, 0, update=u0, mol_id=m0)))
    # the symmetrized half-normal form is kept
    assert np.array_equal(base, base.T) and np.all(base >= 0.0)
    # absent keys: the draw of the cycle and channel alone, bit for bit
    key = jax.random.fold_in(jax.random.PRNGKey(cfg.fock_noise_seed), jnp.int32(1))
    key = jax.random.fold_in(key, 0)
    n = cfg.fock_noise_std * jnp.abs(jax.random.normal(key, shape))
    np.testing.assert_array_equal(np.asarray(n + n.T),
                                  np.asarray(_fock_noise(cfg, jnp.int32(1), shape, 0)))


def test_scf_fock_noise_differs_across_updates_and_molecules():
    """The keys enter the SCF through mol_data: two update counters (and two
    molecule ids) give two trajectories of the same record, while a record
    without the keys follows the keyless draw."""
    from xcquinox.pipeline.data import clear_precompute_cache
    from xcquinox.pipeline.solver_manual import (
        FOCK_NOISE_MOL_ID_KEY, FOCK_NOISE_UPDATE_KEY)
    clear_precompute_cache()
    md = precompute_fixed_density_data(_seed_spec(), required_keys=("eri",))
    model = _seed_model()
    cfg = _noise_cfg()

    def _run(update=None, mol_id=None):
        rec = dict(md)
        if update is not None:
            rec[FOCK_NOISE_UPDATE_KEY] = jnp.asarray(update, dtype=jnp.int32)
        if mol_id is not None:
            rec[FOCK_NOISE_MOL_ID_KEY] = jnp.asarray(mol_id, dtype=jnp.int32)
        return np.asarray(run_scf(cfg, model, rec).energy_trace)

    plain = _run()
    np.testing.assert_array_equal(plain, _run())
    u0m0 = _run(0, 0)
    np.testing.assert_array_equal(u0m0, _run(0, 0))
    assert not np.array_equal(u0m0, _run(1, 0))
    assert not np.array_equal(u0m0, _run(0, 1))
    # the forward-only readout carries no noise whatever the keys say
    exact = np.asarray(run_scf(cfg, model, md, forward_only=True).energy_trace)
    rec = dict(md, **{FOCK_NOISE_UPDATE_KEY: jnp.int32(3)})
    np.testing.assert_array_equal(
        exact, np.asarray(run_scf(cfg, model, rec, forward_only=True).energy_trace))
