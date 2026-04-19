"""Integration tests for SCF backends.

Golden system: H2/STO-3G (2 AOs, 1 occ). Runs in milliseconds.
"""
import pytest
import jax.numpy as jnp

from xcquinox.alec.config import ArchitectureConfig
from xcquinox.alec.models import AlecGGAModel
from xcquinox.alec.data import precompute_fixed_density_data
from xcquinox.alec.solver import (
    SolverConfig, SolverBackend, SolverMode, FeaturePolicy, run_scf,
)
from xcquinox.alec.oneshot import fixed_density_total_energy
from xcquinox.alec.tests.fixtures.molecules import h2_molecule


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
    """manual backend, oneshot mode, zero cycles — byte-identical to legacy path."""
    model, data = _make_h2()
    cfg = SolverConfig(backend=SolverBackend.MANUAL, mode=SolverMode.ONESHOT)
    result = run_scf(cfg, model, data)
    e_legacy = float(fixed_density_total_energy(model, data))
    assert float(result.total_energy) == pytest.approx(e_legacy, abs=1e-12)
    assert int(result.cycles_run) == 0
    assert bool(result.converged) is True


def test_manual_fixed_j_converges_on_h2():
    """H2/STO-3G fixed_j should converge in <=10 cycles at default tol."""
    model, data = _make_h2()
    cfg = SolverConfig(
        backend=SolverBackend.MANUAL, mode=SolverMode.FIXED_J,
        max_cycles=10, conv_tol=1e-8,
    )
    result = run_scf(cfg, model, data)
    assert bool(result.converged) is True
    assert int(result.cycles_run) <= 10
    assert int(result.cycles_run) >= 1
    assert jnp.isfinite(result.total_energy)


def test_manual_full_converges_on_h2_with_eri():
    """FULL mode requires the eri tensor in mol_data; test converges in <=15 cycles."""
    from xcquinox.alec.config import ArchitectureConfig, FeatureSpec
    from xcquinox.alec.models import AlecGGAModel
    from xcquinox.alec.data import precompute_fixed_density_data

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
    regression guarantee" — this test enforces the algebraic equivalence.
    """
    import numpy as np
    from xcquinox.alec.oneshot import fixed_density_total_energy
    from xcquinox.alec.descriptors import assemble_descriptor_features
    from xcquinox.alec.solver_manual import _compute_total_energy

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


def test_pyscfad_oneshot_matches_legacy():
    """pyscfad backend, ONESHOT mode — same byte-identical fast path."""
    model, data = _make_h2()
    cfg = SolverConfig(backend=SolverBackend.PYSCFAD, mode=SolverMode.ONESHOT)
    result = run_scf(cfg, model, data)
    e_legacy = float(fixed_density_total_energy(model, data))
    assert float(result.total_energy) == pytest.approx(e_legacy, abs=1e-12)
    assert int(result.cycles_run) == 0


def test_pyscfad_fixed_j_converges_on_h2():
    """pyscfad backend, FIXED_J mode, FROZEN features. 10 cycles on H2."""
    model, data = _make_h2()
    cfg = SolverConfig(
        backend=SolverBackend.PYSCFAD, mode=SolverMode.FIXED_J,
        max_cycles=10, conv_tol=1e-6,
    )
    result = run_scf(cfg, model, data)
    assert bool(result.converged) is True
    assert jnp.isfinite(result.total_energy)


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
        from xcquinox.alec.data import precompute_fixed_density_data
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


def test_pyscfad_uks_oneshot_o_atom():
    """pyscfad backend ONESHOT UKS on O atom — returns (2, nao, nao) DM."""
    import numpy as np
    import xcquinox.alec as alec
    from xcquinox.alec.config import MoleculeSpec

    spec = MoleculeSpec(
        name="O", atom="O 0 0 0", basis="sto-3g",
        charge=0, spin=2, atom_composition=(("O", 1),), grid_level=1,
    )
    md = precompute_fixed_density_data(spec)
    arch = alec.get_architecture("deep")
    xnet, cnet = alec.create_network_pair(arch, seed=0)
    model = AlecGGAModel.from_arch(arch, xnet=xnet, cnet=cnet)
    cfg = SolverConfig(backend=SolverBackend.PYSCFAD, mode=SolverMode.ONESHOT)
    result = run_scf(cfg, model, md)
    dm = np.asarray(result.density_matrix)
    assert dm.ndim == 3 and dm.shape[0] == 2


def test_pyscfad_uks_fixed_j_runs():
    """pyscfad backend FIXED_J UKS on O atom runs without error."""
    import numpy as np
    import xcquinox.alec as alec
    from xcquinox.alec.config import MoleculeSpec

    spec = MoleculeSpec(
        name="O", atom="O 0 0 0", basis="sto-3g",
        charge=0, spin=2, atom_composition=(("O", 1),), grid_level=1,
    )
    md = precompute_fixed_density_data(spec, required_keys=("eri",))
    arch = alec.get_architecture("deep")
    xnet, cnet = alec.create_network_pair(arch, seed=0)
    model = AlecGGAModel.from_arch(arch, xnet=xnet, cnet=cnet)
    cfg = SolverConfig(
        backend=SolverBackend.PYSCFAD, mode=SolverMode.FIXED_J,
        max_cycles=3, conv_tol=1e-4,
    )
    result = run_scf(cfg, model, md)
    dm = np.asarray(result.density_matrix)
    assert dm.ndim == 3 and dm.shape[0] == 2


def test_pyscfad_fixed_j_monkey_patched_get_j_is_called():
    """Fragility guard (spec Section 6.6): verify the monkey-patched get_j
    is actually used by pyscfad's Fock build, via sentinel propagation.
    If this test fails silently (e.g., by passing without the sentinel
    changing the energy), fixed_j pyscfad must fall back to manual."""
    import pyscfad.dft
    from xcquinox.alec.solver_pyscfad import _rebuild_mol_from_mol_data, _make_alec_eval_xc
    model, data = _make_h2()

    mol = _rebuild_mol_from_mol_data(data)
    mf = pyscfad.dft.RKS(mol)
    mf.define_xc_(
        _make_alec_eval_xc(model, model.descriptors, data, FeaturePolicy.FROZEN),
        "GGA",
    )
    mf.max_cycle = 5
    mf.conv_tol = 1e-4

    J_pinned = data["j_matrix"]
    called = {"flag": False}

    def fixed_get_j(*args, **kwargs):
        called["flag"] = True
        return J_pinned

    mf.get_j = fixed_get_j
    mf.kernel(dm0=data["dm_pbe"])
    assert called["flag"], (
        "pyscfad SCF driver bypassed the overridden get_j — "
        "fixed_j pyscfad mode cannot guarantee J pinning"
    )


def test_precompute_caches_pyscfad_mol():
    """precompute_fixed_density_data caches a pyscfad Mole in mol_data.

    The pyscfad SCF backend uses this cached Mole to avoid calling
    Mole.build() inside filter_jit'd training steps (Mole.build() invokes
    numpy.__array__ which raises TracerArrayConversionError under jit).
    """
    import importlib
    from xcquinox.alec.config import MoleculeSpec
    from xcquinox.alec.data import precompute_fixed_density_data

    spec = MoleculeSpec(
        name="H2", atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g", charge=0, spin=0,
        atom_composition=(("H", 2),), grid_level=1,
    )
    md = precompute_fixed_density_data(spec)
    # _pyscfad_mol key must always be present; value is either a Mole or None.
    assert "_pyscfad_mol" in md, "pyscfad Mole slot missing from mol_data"
    # pyscfad is installed on the dev machine; cache must be populated.
    if importlib.util.find_spec("pyscfad") is not None:
        assert md["_pyscfad_mol"] is not None, (
            "pyscfad installed but cached Mole is None"
        )
        import pyscfad.gto as pyscfad_gto
        assert isinstance(md["_pyscfad_mol"], pyscfad_gto.Mole), (
            "cached _pyscfad_mol is not a pyscfad.gto.Mole instance"
        )


def test_pyscfad_cycles_run_tracks_actual_iterations(monkeypatch):
    """pyscfad backend SCFResult.cycles_run must reflect actual SCF cycles,
    not always return config.max_cycles."""
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    import xcquinox.alec as alec
    from xcquinox.alec.config import MoleculeSpec
    from xcquinox.alec.data import precompute_fixed_density_data
    from xcquinox.alec.solver import SolverConfig, SolverBackend, SolverMode, run_scf

    spec = MoleculeSpec(name="H2", atom="H 0 0 0; H 0 0 0.74",
                       basis="sto-3g", charge=0, spin=0,
                       atom_composition=(("H", 2),), grid_level=1)
    md = precompute_fixed_density_data(spec, required_keys=("eri",))
    arch = alec.get_architecture("deep")
    xnet, cnet = alec.create_network_pair(arch, seed=0)
    model = alec.AlecGGAModel.from_arch(arch, xnet=xnet, cnet=cnet)
    # Set max_cycles high so SCF converges before hitting cap
    cfg = SolverConfig(backend=SolverBackend.PYSCFAD, mode=SolverMode.FIXED_J,
                      max_cycles=50, conv_tol=1e-6)
    result = run_scf(cfg, model, md)
    # Should converge well under max_cycles
    assert int(result.cycles_run) < 50, (
        f"cycles_run = {int(result.cycles_run)}; expected < 50 if pyscfad converged"
    )
    assert int(result.cycles_run) > 0


def test_pyscfad_reassemble_policy_no_warning(monkeypatch):
    """REASSEMBLE policy on pyscfad must NOT emit the 'not yet implemented'
    warning anymore."""
    import warnings
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    import xcquinox.alec as alec
    from xcquinox.alec.config import MoleculeSpec
    from xcquinox.alec.data import precompute_fixed_density_data
    from xcquinox.alec.solver import (SolverConfig, SolverBackend, SolverMode,
                                      FeaturePolicy, run_scf)
    spec = MoleculeSpec(name="H2", atom="H 0 0 0; H 0 0 0.74",
                       basis="sto-3g", charge=0, spin=0,
                       atom_composition=(("H", 2),), grid_level=1)
    arch = alec.get_architecture("deep_cusp")
    md = precompute_fixed_density_data(
        spec, required_keys=("eri",),
        descriptors=arch.materialize_descriptors(),
    )
    xnet, cnet = alec.create_network_pair(arch, seed=0)
    model = alec.AlecGGAModel.from_arch(arch, xnet=xnet, cnet=cnet)
    cfg = SolverConfig(backend=SolverBackend.PYSCFAD, mode=SolverMode.FIXED_J,
                      feature_policy=FeaturePolicy.REASSEMBLE,
                      max_cycles=3, conv_tol=1e-4)
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        run_scf(cfg, model, md)
    # Filter for the specific warning — it should NOT be present
    for w in recorded:
        assert "REASSEMBLE policy on pyscfad backend is not yet implemented" \
               not in str(w.message), (
            f"REASSEMBLE downgrade warning still emitted: {w.message}"
        )


def test_pyscfad_reassemble_policy_differs_from_frozen(monkeypatch):
    """REASSEMBLE and FROZEN should produce different DMs for a descriptor-ful
    architecture when the SCF is still iterating.

    Setup:
      - Use 6-31g so the DM has enough degrees of freedom to distinguish
        the two feature policies (sto-3g H2 collapses to the same
        converged DM regardless of features).
      - Perturb dm_pbe so SCF starts away from self-consistency.
      - Cap at max_cycles=2 with conv_tol=1e-12 to prevent convergence
        erasing the feature-policy difference.
    """
    import numpy as np
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    import xcquinox.alec as alec
    from xcquinox.alec.config import MoleculeSpec
    from xcquinox.alec.data import precompute_fixed_density_data
    from xcquinox.alec.solver import (SolverConfig, SolverBackend, SolverMode,
                                      FeaturePolicy, run_scf)
    spec = MoleculeSpec(name="H2", atom="H 0 0 0; H 0 0 0.74",
                       basis="6-31g", charge=0, spin=0,
                       atom_composition=(("H", 2),), grid_level=1)
    arch = alec.get_architecture("deep_dm")
    md = precompute_fixed_density_data(
        spec, required_keys=("eri",),
        descriptors=arch.materialize_descriptors(),
    )
    xnet, cnet = alec.create_network_pair(arch, seed=0)
    model = alec.AlecGGAModel.from_arch(arch, xnet=xnet, cnet=cnet)

    # Perturb dm_pbe so SCF actually iterates — REASSEMBLE vs FROZEN
    # must diverge when features are recomputed from an evolving DM.
    rng = np.random.default_rng(42)
    dm0 = np.asarray(md["dm_pbe"])
    perturb = 0.2 * rng.standard_normal(dm0.shape)
    perturb = 0.5 * (perturb + perturb.T)  # keep DM symmetric
    md_perturbed = dict(md)
    md_perturbed["dm_pbe"] = dm0 + perturb

    cfg_frozen = SolverConfig(backend=SolverBackend.PYSCFAD, mode=SolverMode.FIXED_J,
                             feature_policy=FeaturePolicy.FROZEN, max_cycles=2,
                             conv_tol=1e-12)
    cfg_reass = SolverConfig(backend=SolverBackend.PYSCFAD, mode=SolverMode.FIXED_J,
                            feature_policy=FeaturePolicy.REASSEMBLE, max_cycles=2,
                            conv_tol=1e-12)

    dm_frozen = np.asarray(run_scf(cfg_frozen, model, md_perturbed).density_matrix)
    dm_reass = np.asarray(run_scf(cfg_reass, model, md_perturbed).density_matrix)

    # The two DMs should differ because REASSEMBLE updates features per cycle
    diff = float(np.max(np.abs(dm_frozen - dm_reass)))
    assert diff > 1e-8, f"REASSEMBLE gives same DM as FROZEN: diff={diff}"

