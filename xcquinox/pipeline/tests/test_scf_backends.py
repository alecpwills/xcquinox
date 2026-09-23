"""Integration tests for SCF backends.

Golden system: H2/STO-3G (2 AOs, 1 occ). Runs in milliseconds.
"""
import pytest
import jax.numpy as jnp

from xcquinox.pipeline.config import ArchitectureConfig
from xcquinox.pipeline.models import AlecGGAModel
from xcquinox.pipeline.data import precompute_fixed_density_data
from xcquinox.pipeline.solver import (
    SolverConfig, SolverBackend, SolverMode, FeaturePolicy, run_scf,
)
from xcquinox.pipeline.oneshot import fixed_density_total_energy
from xcquinox.pipeline.tests.fixtures.molecules import h2_molecule


def _make_h2():
    arch = ArchitectureConfig(
        name="t", depth=2, nodes=8, attention=False,
        descriptors=(), x_constraints=(), c_constraints=(),
        double_lob_clamp_allowed=False,
    )
    model = AlecGGAModel.from_arch(arch, seed=0)
    data = precompute_fixed_density_data(h2_molecule())
    return model, data


def test_manual_oneshot_matches_legacy():
    """manual backend, oneshot mode, zero cycles, byte-identical to legacy path."""
    model, data = _make_h2()
    cfg = SolverConfig(backend=SolverBackend.MANUAL, mode=SolverMode.ONESHOT)
    result = run_scf(cfg, model, data)
    e_legacy = float(fixed_density_total_energy(model, data))
    assert float(result.total_energy) == pytest.approx(e_legacy, abs=1e-12)
    assert int(result.cycles_run) == 0
    assert bool(result.converged) is True


def test_manual_full_converges_on_h2_with_eri():
    """FULL mode requires the eri tensor in mol_data; test converges in <=15 cycles."""
    from xcquinox.pipeline.config import ArchitectureConfig, FeatureSpec
    from xcquinox.pipeline.models import AlecGGAModel
    from xcquinox.pipeline.data import precompute_fixed_density_data

    arch = ArchitectureConfig(
        name="t", depth=2, nodes=8, attention=False,
        descriptors=(FeatureSpec.of("cusp"), FeatureSpec.of("dm_statistics")),
        x_constraints=(), c_constraints=(),
        double_lob_clamp_allowed=False,
    )
    model = AlecGGAModel.from_arch(arch, seed=0)
    data = precompute_fixed_density_data(
        h2_molecule(),
        descriptors=arch.materialize_descriptors(),
        required_keys=("eri",),
    )
    cfg = SolverConfig(
        backend=SolverBackend.MANUAL, mode=SolverMode.FULL,
        max_cycles=15, conv_tol=1e-6,
    )
    result = run_scf(cfg, model, data)
    assert bool(result.converged) is True
    assert jnp.isfinite(result.total_energy)


def test_oneshot_and_scf_total_energy_agree_at_D_PBE():
    """Contract test: the ONESHOT fast-path (via fixed_density_total_energy)
    and the SCF code path (via _compute_total_energy) must produce the
    same number when D=D_PBE and J=J[D_PBE]. Spec Section 5.2 "One-shot
    regression guarantee": this test enforces the algebraic equivalence.
    """
    from xcquinox.pipeline.oneshot import fixed_density_total_energy
    from xcquinox.pipeline.descriptors import assemble_descriptor_features
    from xcquinox.pipeline.solver_manual import _compute_total_energy

    model, data = _make_h2()
    e_oneshot = float(fixed_density_total_energy(model, data))

    features = assemble_descriptor_features(model.descriptors, data)
    e_scf = float(_compute_total_energy(
        model=model,
        D=data["dm_pbe"],
        rho=data["rho_grid"],
        sigma=data["sigma_grid"],
        features=features,
        grid_weights=data["grid_weights"],
        h_core=data["h_core"],
        J=data["j_matrix"],
        e_nuc=jnp.asarray(data["e_nuc"]),
    ))

    assert abs(e_oneshot - e_scf) < 1e-12, (
        f"one-shot and SCF total-energy code paths diverged at D=D_PBE: "
        f"|delta|={abs(e_oneshot - e_scf):.3e} Ha"
    )


def test_pyscfad_full_converges_on_h2():
    model, data = _make_h2()
    cfg = SolverConfig(
        backend=SolverBackend.PYSCFAD, mode=SolverMode.FULL,
        max_cycles=15, conv_tol=1e-6,
    )
    result = run_scf(cfg, model, data)
    assert bool(result.converged) is True
    assert jnp.isfinite(result.total_energy)


def test_backends_agree_fixed_j_on_h2():
    model, data = _make_h2()
    cfg_m = SolverConfig(
        backend=SolverBackend.MANUAL, mode=SolverMode.FIXED_J,
        max_cycles=20, conv_tol=1e-8,
    )
    cfg_p = SolverConfig(
        backend=SolverBackend.PYSCFAD, mode=SolverMode.FIXED_J,
        max_cycles=20, conv_tol=1e-8,
    )
    e_m = float(run_scf(cfg_m, model, data).total_energy)
    e_p = float(run_scf(cfg_p, model, data).total_energy)
    assert abs(e_m - e_p) < 1e-4, f"manual={e_m} pyscfad={e_p}"


def test_backends_agree_full_on_h2():
    model, data = _make_h2()
    data_with_eri = dict(data)
    if data_with_eri.get("eri") is None:
        from xcquinox.pipeline.data import precompute_fixed_density_data
        data_with_eri = precompute_fixed_density_data(
            h2_molecule(), required_keys=("eri",),
        )
    cfg_m = SolverConfig(
        backend=SolverBackend.MANUAL, mode=SolverMode.FULL,
        max_cycles=30, conv_tol=1e-8,
    )
    cfg_p = SolverConfig(
        backend=SolverBackend.PYSCFAD, mode=SolverMode.FULL,
        max_cycles=30, conv_tol=1e-8,
    )
    e_m = float(run_scf(cfg_m, model, data_with_eri).total_energy)
    e_p = float(run_scf(cfg_p, model, data_with_eri).total_energy)
    assert abs(e_m - e_p) < 1e-3, f"manual={e_m} pyscfad={e_p}"


def test_backends_agree_fixed_j_uks_on_o_atom():
    """Open-shell UKS manual-vs-pyscfad energy agreement (SP4 review).

    A multi-cycle FIXED_J UKS SCF builds the per-spin Fock from V_xc on BOTH
    backends, but via different assemblers: the manual solver uses the explicit
    spin-resolved V_xc (oneshot._uks_spin_resolved_vxc), while pyscfad routes
    the callback's libxc-convention (vrho, vsigma=(uu,ud,dd)) through pyscf's
    nr_uks numint. If the (uu,ud,dd) vsigma convention (esp. the ud cross term)
    were wrong, the two SCFs would converge to different energies. Agreement
    here is the regression guard the matrix-level tests in
    test_solv01_split_xc.py explicitly punt on.
    """
    import xcquinox.pipeline as pipeline
    from xcquinox.pipeline.config import MoleculeSpec

    spec = MoleculeSpec(
        name="O", atom="O 0 0 0", basis="sto-3g",
        charge=0, spin=2, atom_composition=(("O", 1),), grid_level=1,
    )
    md = precompute_fixed_density_data(spec, required_keys=("eri",))
    arch = pipeline.get_architecture("deep")
    xnet, cnet = pipeline.create_network_pair(arch, seed=0)
    model = AlecGGAModel.from_arch(arch, xnet=xnet, cnet=cnet)
    cfg_m = SolverConfig(
        backend=SolverBackend.MANUAL, mode=SolverMode.FIXED_J,
        max_cycles=20, conv_tol=1e-8,
    )
    cfg_p = SolverConfig(
        backend=SolverBackend.PYSCFAD, mode=SolverMode.FIXED_J,
        max_cycles=20, conv_tol=1e-8,
    )
    e_m = float(run_scf(cfg_m, model, md).total_energy)
    e_p = float(run_scf(cfg_p, model, md).total_energy)
    assert abs(e_m - e_p) < 1e-4, (
        f"open-shell UKS backends disagree (vsigma uu/ud/dd convention?): "
        f"manual={e_m} pyscfad={e_p}")


def test_install_functional_sets_what_pyscfad_reads():
    """The installer must write the functional where pyscfad reads it.

    pyscfad's integration loops read ``_xc_type`` and ``eval_xc`` on the numint
    instance, which PySCF's numint-level installer sets, while the mean-field
    method reaches an installer that pyscfad 0.3.4's wrapper module does not
    carry. The three attributes asserted below (of the five the installer
    sets), the returned object and the callback's use by a short kernel run
    are therefore the contract the backend depends on; ``hybrid_coeff`` is the
    installer's contract on the numint, not a switch of the Fock build, whose
    exact-exchange term pyscfad decides from the mean-field's xc string.
    """
    import pyscfad.dft
    from xcquinox.pipeline.solver_pyscfad import (
        install_functional, _rebuild_mol_from_mol_data, _make_pipeline_eval_xc,
    )
    model, data = _make_h2()

    mol = _rebuild_mol_from_mol_data(data)
    mf = pyscfad.dft.RKS(mol)
    pipeline_cb = _make_pipeline_eval_xc(
        model, model.descriptors, data, FeaturePolicy.FROZEN,
    )
    calls = {"n": 0}

    def cb(*args, **kwargs):
        calls["n"] += 1
        return pipeline_cb(*args, **kwargs)

    out = install_functional(mf, cb, "GGA")
    assert out is mf, "install_functional must return the mean-field it installed on"
    assert mf._numint._xc_type("anything") == "GGA"
    assert mf._numint.eval_xc is cb
    assert mf._numint.hybrid_coeff() == 0

    mf.max_cycle = 3
    mf.conv_tol = 1e-4
    mf.kernel(dm0=data["dm_pbe"])
    assert calls["n"] > 0, (
        "the installed callback was never reached by pyscfad's integration "
        "loops; the functional is not the one this SCF used"
    )


def test_pyscfad_refuses_dm_dependent_descriptors_under_reassemble(monkeypatch):
    """A wrong potential must fail loudly rather than converge quietly.

    ``eval_xc_pipeline_gga`` is a libxc-compatible per-point callback returning
    ``(exc, vrho, vsigma)``; pyscf's numint builds the Fock matrix from those
    three arrays alone, so there is no channel for the global
    ``sum_g w_g (de/dfeatures)_g . dfeatures_g/dP`` term that any DM-dependent
    descriptor contributes. Returning V_xc without it means the SCF converges a
    density that does not minimise its own energy.

    FROZEN is exempt: the features are constant in P, so the term is identically
    zero and the existing callback is already exact.
    """
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    import dataclasses
    import pytest as _pytest
    import xcquinox.pipeline as pipeline
    from xcquinox.pipeline.solver_pyscfad import _reject_dm_dependent_descriptors
    from xcquinox.pipeline.solver import FeaturePolicy

    def _model(name):
        arch = dataclasses.replace(pipeline.get_architecture(name),
                                   zero_init_final_layer=False)
        return pipeline.AlecGGAModel.from_arch(arch, seed=0)

    for name in ("deep_dm", "deep_mgga_3x16", "deep_rung35_3x16",
                 "deep_rung35ms_3x16"):
        with _pytest.raises(NotImplementedError, match="pyscfad"):
            _reject_dm_dependent_descriptors(_model(name),
                                             FeaturePolicy.REASSEMBLE)
        # FROZEN makes the missing term vanish, so it stays supported.
        _reject_dm_dependent_descriptors(_model(name), FeaturePolicy.FROZEN)

    # Geometry-only and descriptor-free architectures are unaffected: cusp is
    # the only descriptor without a compute_from_dm.
    for name in ("deep_3x16", "deep_attn_3x16", "deep_cusp_3x16",
                 "deep_geom_3x16", "deep_geom_attn_3x16"):
        for policy in (FeaturePolicy.REASSEMBLE, FeaturePolicy.FROZEN):
            _reject_dm_dependent_descriptors(_model(name), policy)


def test_build_pyscfad_mf_pins_the_cutoff_whatever_pyscfads_default(
        monkeypatch):
    """The backend's mean-field carries the reference density cutoff.

    ``_build_pyscfad_mf`` settles the quadrature the SCF integrates on, and
    its UKS branch prunes that grid at ``mf.small_rho_cutoff``. The record the
    frozen features were assembled on was taken on the grid pruned at
    ``pyscf_determinism.REFERENCE_SMALL_RHO_CUTOFF``, so an inherited default
    of 0 -- where pyscf 2.14.0 moved the Kohn-Sham class attribute from 1e-7,
    and where any downstream wrapper is free to put it -- would hand the
    backend a different quadrature from the precompute's. The class default is
    set to 0 below, so a value of 1e-7 on the built object can only come from
    the builder's own pin.
    """
    import pyscfad.dft
    import pyscfad.dft.rks
    import pyscfad.dft.uks
    from xcquinox.pipeline import pyscf_determinism as pd
    from xcquinox.pipeline.solver_pyscfad import (
        _build_pyscfad_mf, _rebuild_mol_from_mol_data)

    for cls in (pyscfad.dft.rks.RKS, pyscfad.dft.uks.UKS):
        # The attribute is inherited from the Kohn-Sham base class rather
        # than defined on these classes, hence raising=False.
        monkeypatch.setattr(cls, "small_rho_cutoff", 0.0, raising=False)

    _model, data = _make_h2()
    mol = _rebuild_mol_from_mol_data(data)
    # The patched default reaches a plain construction: without this the
    # assertion below could hold for want of a patch rather than for a pin.
    assert pyscfad.dft.RKS(mol).small_rho_cutoff == 0.0

    mf = _build_pyscfad_mf(mol, data)
    assert mf.small_rho_cutoff == pd.REFERENCE_SMALL_RHO_CUTOFF == 1e-7

    # The branch where the pin acts inside the builder itself: the UKS branch
    # prunes the grid here, on the stored PBE density, at the pinned value. On
    # the H atom (def2-svp, grid level 1) the converged and the initial-guess
    # densities prune the same points, so the builder's grid is the record's
    # (2336 points, measured) with the pin and the unpruned grid (2472) with
    # the patched default; a pin that reached the RKS branch alone leaves the
    # UKS grid at the default and fails here.
    from xcquinox.pipeline.config import MoleculeSpec
    h_spec = MoleculeSpec(name="H_cutoff_pin", atom="H 0 0 0", basis="def2-svp",
                          charge=0, spin=1, atom_composition=(("H", 1),),
                          grid_level=1)
    h_data = precompute_fixed_density_data(h_spec)
    assert pyscfad.dft.UKS(_rebuild_mol_from_mol_data(h_data)).small_rho_cutoff == 0.0
    h_mf = _build_pyscfad_mf(_rebuild_mol_from_mol_data(h_data), h_data)
    assert h_mf.small_rho_cutoff == pd.REFERENCE_SMALL_RHO_CUTOFF
    assert int(h_mf.grids.weights.size) == int(h_data["grid_weights"].shape[0])

