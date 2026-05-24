"""Tests for xcquinox.alec.evaluation -- Metric ABC, 4 metrics, run_test.

Implements Task 5.3 test suite: 36 tests.

Tests 1-12: Per-metric registry+compute+schema (4 metrics x 3 = 12).
Tests 13-15: Registry-level.
Tests 16-24: TestSpec.validate negative paths (9 total).
Tests 25-36: run_test integration + misc.
"""
import csv
import json
import math
import os
import tempfile

import jax.numpy as jnp
import numpy as np
import pytest
import equinox as eqx

from xcquinox.alec.config import (
    ArchitectureConfig,
    FeatureSpec,
    MoleculeSpec,
    TestSpec,
    TrainingSpec,
    get_architecture,
)
from xcquinox.alec.evaluation import (
    METRIC_REGISTRY,
    AtomizationEnergyMetric,
    ConstraintViolationsMetric,
    DensityRMSEMetric,
    Metric,
    TotalEnergyMetric,
    _flatten_constraint_report,
    list_metrics,
    make_metric,
    run_test,
)
from xcquinox.alec.models import AlecGGAModel
from xcquinox.alec.data import precompute_fixed_density_data
from xcquinox.alec.tests.fixtures.molecules import (
    h_atom,
    h2o_molecule,
    o_atom,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_arch(**overrides):
    defaults = dict(
        name="t", depth=2, nodes=8, attention=False,
        descriptors=(), x_constraints=(), c_constraints=(),
        double_lob_clamp_allowed=False,
    )
    defaults.update(overrides)
    return ArchitectureConfig(**defaults)


def _make_model(arch=None, seed=0):
    if arch is None:
        arch = _make_arch()
    return AlecGGAModel.from_arch(arch, seed=seed)


# ---------------------------------------------------------------------------
# Module-scoped fixtures (PySCF -- expensive, computed once)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def h_data():
    return precompute_fixed_density_data(h_atom())


@pytest.fixture(scope="module")
def o_data():
    return precompute_fixed_density_data(o_atom())


@pytest.fixture(scope="module")
def h2o_data():
    return precompute_fixed_density_data(h2o_molecule())


@pytest.fixture(scope="module")
def tiny_model():
    return _make_model(seed=0)


@pytest.fixture(scope="module")
def trained_checkpoint(h_data, o_data, h2o_data):
    """Train a tiny model for 3 steps and serialize it. Return (path, arch)."""
    from xcquinox.alec.train import run_training

    tmpdir = tempfile.mkdtemp()
    ckdir = os.path.join(tmpdir, "ckpt")

    h = h_atom()
    o = o_atom()
    h2o = h2o_molecule()

    ae_h2o = float(h_data["E_pbe"] * 2 + o_data["E_pbe"] - h2o_data["E_pbe"])
    targets = {
        "H": float(h_data["E_pbe"]),
        "O": float(o_data["E_pbe"]),
        "H2O": max(ae_h2o, 0.001),
    }
    atom_energies = {
        "H": float(h_data["E_pbe"]),
        "O": float(o_data["E_pbe"]),
    }

    arch = _make_arch()
    spec = TrainingSpec.from_dicts(
        arch=arch,
        molecules=(h, o, h2o),
        targets=targets,
        atom_energies=atom_energies,
        loss_name="A_atomization",
        n_steps=3,
        checkpoint_dir=ckdir,
        seed=42,
    )
    run_training(spec)
    model_path = os.path.join(ckdir, "model.eqx")
    return model_path, arch, atom_energies


# ---------------------------------------------------------------------------
# Tests 1-3: TotalEnergyMetric
# ---------------------------------------------------------------------------

# (1a) Registry roundtrip
def test_total_energy_registry_roundtrip():
    m = make_metric("total_energy")
    assert isinstance(m, TotalEnergyMetric)


# (1b) Compute on tiny model returns dict
def test_total_energy_compute(tiny_model, h2o_data):
    m = TotalEnergyMetric()
    result = m.compute(tiny_model, h2o_data)
    assert isinstance(result, dict)
    assert math.isfinite(result["E_total_nn"])
    assert math.isfinite(result["E_pbe"])


# (1c) Output keys match schema
def test_total_energy_schema(tiny_model, h2o_data):
    m = TotalEnergyMetric()
    result = m.compute(tiny_model, h2o_data)
    assert "E_total_nn" in result
    assert "E_pbe" in result
    # No E_ref_literature set, so no error keys
    assert "E_error_hartree" not in result

    # Now test with E_ref_literature set
    mol_data_ref = dict(h2o_data)
    mol_data_ref["E_ref_literature"] = -76.0
    result_ref = m.compute(tiny_model, mol_data_ref)
    assert "E_error_hartree" in result_ref
    assert "E_error_kcalmol" in result_ref
    assert math.isfinite(result_ref["E_error_hartree"])
    assert math.isfinite(result_ref["E_error_kcalmol"])


# ---------------------------------------------------------------------------
# Tests 4-6: AtomizationEnergyMetric
# ---------------------------------------------------------------------------

# (4a) Registry roundtrip
def test_atomization_energy_registry_roundtrip():
    m = make_metric("atomization_energy", atom_energies={"H": -0.5, "O": -74.8})
    assert isinstance(m, AtomizationEnergyMetric)


# (4b) Compute on tiny model returns dict
def test_atomization_energy_compute(tiny_model, h_data, o_data, h2o_data):
    ae = {"H": float(h_data["E_pbe"]), "O": float(o_data["E_pbe"])}
    m = AtomizationEnergyMetric(atom_energies=ae)
    result = m.compute(tiny_model, h2o_data)
    assert isinstance(result, dict)
    assert "AE_nn" in result
    assert math.isfinite(result["AE_nn"])


# (4c) Output keys match schema (with reference)
def test_atomization_energy_schema(tiny_model, h_data, o_data, h2o_data):
    ae = {"H": float(h_data["E_pbe"]), "O": float(o_data["E_pbe"])}
    ref = {"H2O": 232.0}
    m = AtomizationEnergyMetric(atom_energies=ae, reference_ae_kcalmol=ref)
    result = m.compute(tiny_model, h2o_data)
    assert "AE_nn" in result
    assert "AE_ref_kcalmol" in result
    assert "AE_error_hartree" in result
    assert "AE_error_kcalmol" in result


# ---------------------------------------------------------------------------
# Tests 7-9: DensityRMSEMetric
# ---------------------------------------------------------------------------

# (7a) Registry roundtrip
def test_density_rmse_registry_roundtrip():
    m = make_metric("density_rmse")
    assert isinstance(m, DensityRMSEMetric)


# (7b) Compute on tiny model returns dict
def test_density_rmse_compute(tiny_model, h2o_data):
    mol_data = dict(h2o_data)
    mol_data["rho_ref_grid"] = mol_data["rho_grid"] + 0.001 * jnp.ones_like(
        mol_data["rho_grid"]
    )
    mol_data["ref_density_method"] = "hf"
    m = DensityRMSEMetric()
    result = m.compute(tiny_model, mol_data)
    assert isinstance(result, dict)
    assert "density_rmse" in result
    assert result["density_rmse"] is not None
    assert result["density_rmse"] > 0.0


# Solver-aware density metric: the value must depend on solver_config so
# that training's DM/density loss (which uses the spec's solver_config) and
# evaluation's density_rmse measure the same quantity. Without this plumbing,
# training with FIXED_J / FULL optimizes one density (SCF-iterated) while
# eval measures another (oneshot 1-Roothaan-step), exactly analogous to the
# 2026-04-24 energy-functional bug.
def test_run_test_with_full_solver_config_passes_eri_to_precompute(
    trained_checkpoint, h2o_data,
):
    """Regression: eval with TestSpec(solver_config=FULL) requires the
    4-index ERI tensor in mol_data because run_scf(FULL) builds J[D]
    every cycle. run_test must include ``eri`` in required_keys whenever
    spec.solver_config.mode == FULL; without this guard the eval crashes
    inside opt_einsum with 'Cannot determine the shape of None'."""
    import numpy as np
    from xcquinox.alec.config import MoleculeSpec
    from xcquinox.alec.solver import SolverConfig, SolverMode

    model_path, arch, atom_energies = trained_checkpoint
    rho_ref = np.asarray(h2o_data["rho_grid"])

    with tempfile.TemporaryDirectory() as tmpdir:
        ext_path = os.path.join(tmpdir, "H2O.npz")
        np.savez(
            ext_path, rho_ref_grid=rho_ref, ref_density_method="pbe",
            dm_target=np.asarray(h2o_data["dm_pbe"]),
            E_ref_literature=float(h2o_data["E_pbe"]),
        )
        h2o = MoleculeSpec(
            name="H2O",
            atom="O 0 0 0; H 0 0 0.96; H 0.96 0 0",
            basis="sto-3g", charge=0, spin=0,
            atom_composition=(("H", 2), ("O", 1)),
            external_data_path=ext_path,
        )
        spec = TestSpec.from_dicts(
            model_checkpoint=model_path, arch=arch, molecules=(h2o,),
            metrics=("density_rmse",),
            output_dir=os.path.join(tmpdir, "out"),
            solver_config=SolverConfig(
                mode=SolverMode.FULL, max_cycles=2, conv_tol=1e-4,
            ),
        )
        # Must not raise.
        result = run_test(spec)
        rmse = result["per_molecule"][0]["density_rmse"]
        assert rmse is not None and math.isfinite(rmse)


def test_pbe_reference_metric_computes_pbe_ae_error(tiny_model, h_data, o_data, h2o_data):
    """PBEReferenceMetric reports the PBE-level atomization energy and its
    error vs literature reference, using ONLY E_pbe from mol_data + an
    ``atom_energies`` dict (PBE-consistent atomic totals). Zero
    NN forward pass. This is the 'what if we just used PBE?' baseline
    shown alongside trained-NN results on the notebook's comparison plots."""
    from xcquinox.alec.evaluation import PBEReferenceMetric
    # PBE-consistent atom anchors (same convention as AtomizationEnergyMetric).
    atom_energies = {"H": float(h_data["E_pbe"]), "O": float(o_data["E_pbe"])}
    # Literature reference in kcal/mol; we use the PBE-derived AE itself
    # as the "reference" so the error should come out near zero.
    ae_h2o_ha = float(h_data["E_pbe"] * 2 + o_data["E_pbe"] - h2o_data["E_pbe"])
    HA_TO_KCAL = 627.5094740631
    ref_ae_kcalmol = {"H2O": ae_h2o_ha * HA_TO_KCAL}
    m = PBEReferenceMetric(
        atom_energies=atom_energies, reference_ae_kcalmol=ref_ae_kcalmol,
    )
    # h2o_data must have mol name so reference lookup works.
    mol_data = dict(h2o_data); mol_data["name"] = "H2O"
    result = m.compute(tiny_model, mol_data)
    assert "AE_pbe" in result
    assert "AE_error_pbe_kcalmol" in result
    # Because ref_ae == AE_pbe computed the same way, error should be ~0.
    assert abs(result["AE_error_pbe_kcalmol"]) < 1e-6, (
        f"PBE error vs self-reference should be ~0, got "
        f"{result['AE_error_pbe_kcalmol']}"
    )


def test_pbe_reference_metric_is_model_independent(tiny_model, h_data, o_data, h2o_data):
    """PBEReferenceMetric must not depend on the NN model — it's a
    hardware-free baseline. Calling with two different models on the
    same mol_data must produce identical output."""
    from xcquinox.alec.evaluation import PBEReferenceMetric
    atom_energies = {"H": float(h_data["E_pbe"]), "O": float(o_data["E_pbe"])}
    m = PBEReferenceMetric(atom_energies=atom_energies)
    mol_data = dict(h2o_data); mol_data["name"] = "H2O"
    out1 = m.compute(tiny_model, mol_data)
    # Any other "model" object — PBEReferenceMetric should ignore it.
    out2 = m.compute(None, mol_data)
    for k, v in out1.items():
        if isinstance(v, float):
            assert math.isclose(v, out2[k]), f"{k}: {v} vs {out2[k]}"


def test_density_rmse_honors_solver_config(tiny_model, h2o_data):
    from xcquinox.alec.solver import SolverConfig, SolverMode
    mol_data = dict(h2o_data)
    mol_data["rho_ref_grid"] = mol_data["rho_grid"]
    mol_data["ref_density_method"] = "hf"
    m = DensityRMSEMetric()
    # Oneshot path (no SCF) vs FIXED_J path (3 SCF cycles with J pinned).
    r_oneshot = m.compute(tiny_model, mol_data)
    r_fixed_j = m.compute(
        tiny_model, mol_data,
        solver_config=SolverConfig(mode=SolverMode.FIXED_J, max_cycles=3),
    )
    # Both must produce a finite RMSE.
    assert r_oneshot["density_rmse"] is not None and math.isfinite(r_oneshot["density_rmse"])
    assert r_fixed_j["density_rmse"] is not None and math.isfinite(r_fixed_j["density_rmse"])
    # And they must differ (SCF iteration changes the density for a random NN).
    assert abs(r_oneshot["density_rmse"] - r_fixed_j["density_rmse"]) > 1e-10, (
        f"DensityRMSEMetric must consume solver_config; oneshot and FIXED_J "
        f"produced the same RMSE ({r_oneshot['density_rmse']} == {r_fixed_j['density_rmse']})"
    )


# (7c) Output keys match schema
def test_density_rmse_schema(tiny_model, h2o_data):
    mol_data = dict(h2o_data)
    mol_data["rho_ref_grid"] = mol_data["rho_grid"] + 0.001 * jnp.ones_like(
        mol_data["rho_grid"]
    )
    mol_data["ref_density_method"] = "hf"
    m = DensityRMSEMetric()
    result = m.compute(tiny_model, mol_data)
    assert "density_rmse" in result
    assert "density_l1" in result
    assert math.isfinite(result["density_rmse"])
    assert math.isfinite(result["density_l1"])


# ---------------------------------------------------------------------------
# Tests 10-12: ConstraintViolationsMetric
# ---------------------------------------------------------------------------

# (10a) Registry roundtrip
def test_constraint_violations_registry_roundtrip():
    m = make_metric("constraint_violations")
    assert isinstance(m, ConstraintViolationsMetric)


# (10b) Compute on tiny model returns dict
def test_constraint_violations_compute(tiny_model, h2o_data):
    m = ConstraintViolationsMetric()
    result = m.compute(tiny_model, h2o_data)
    assert isinstance(result, dict)
    # shallow arch has no constraints, so result dict should be empty
    assert len(result) == 0


# (10c) Compute with constrained arch returns expected keys
def test_constraint_violations_schema(h2o_data):
    arch = ArchitectureConfig.from_spec(
        "constrained_test", 2, 8,
        x_constraints=["lieb_oxford"],
        c_constraints=["non_negative_correlation"],
    )
    model = AlecGGAModel.from_arch(arch, seed=0)
    m = ConstraintViolationsMetric()
    result = m.compute(model, h2o_data)
    assert isinstance(result, dict)
    # Should have keys for x_lieb_oxford and c_non_negative_correlation
    assert any("lieb_oxford" in k for k in result)
    assert any("non_negative_correlation" in k for k in result)
    for key in ("max", "mean", "l2"):
        assert any(key in k for k in result)


# ---------------------------------------------------------------------------
# Tests 13-15: Registry-level
# ---------------------------------------------------------------------------

# (13) METRIC_REGISTRY has exactly 5 entries
# (was 4 prior to the 2026-04-24 addition of PBEReferenceMetric, the
# model-independent PBE-baseline metric used by the notebook's comparison plots)
def test_metric_registry_has_6_entries():
    assert len(METRIC_REGISTRY) == 6


# (14) list_metrics returns sorted names
def test_list_metrics_sorted():
    names = list_metrics()
    assert names == sorted(names)
    assert set(names) == {
        "total_energy", "atomization_energy", "density_rmse",
        "constraint_violations", "pbe_reference", "scf_convergence",
    }


# (15) make_metric("not_a_metric") raises KeyError
def test_make_metric_unknown_raises():
    with pytest.raises(KeyError, match="unknown metric"):
        make_metric("not_a_metric")


# ---------------------------------------------------------------------------
# Tests 16-24: TestSpec.validate negative paths
# ---------------------------------------------------------------------------

def _make_real_checkpoint(arch=None, tmpdir=None):
    """Create a valid model.eqx file for TestSpec.validate tests."""
    if arch is None:
        arch = _make_arch()
    if tmpdir is None:
        tmpdir = tempfile.mkdtemp()
    model = AlecGGAModel.from_arch(arch, seed=0)
    ckpt_path = os.path.join(tmpdir, "model.eqx")
    eqx.tree_serialise_leaves(ckpt_path, model)
    return ckpt_path


def _make_test_spec(**overrides):
    """Build a minimal valid TestSpec."""
    tmpdir = tempfile.mkdtemp()
    arch = overrides.pop("arch", _make_arch())
    ckpt = overrides.pop("model_checkpoint", _make_real_checkpoint(arch, tmpdir))
    outdir = overrides.pop("output_dir", os.path.join(tmpdir, "output"))
    h = h_atom()
    o = o_atom()
    h2o = h2o_molecule()
    defaults = dict(
        model_checkpoint=ckpt,
        arch=arch,
        molecules=(h, o, h2o),
        metrics=("total_energy",),
        output_dir=outdir,
    )
    defaults.update(overrides)
    return TestSpec(**defaults)


# (16-i) missing model_checkpoint file
def test_validate_missing_checkpoint():
    spec = _make_test_spec(model_checkpoint="/tmp/nonexistent_model.eqx")
    with pytest.raises(ValueError, match="model_checkpoint not found"):
        spec.validate()


# (16-ii) unknown metric name
def test_validate_unknown_metric():
    spec = _make_test_spec(metrics=("total_energy", "bogus_metric"))
    with pytest.raises(ValueError, match="unknown metrics"):
        spec.validate()


# (16-iii) atomization_energy without atom_energies
def test_validate_ae_without_atom_energies():
    spec = _make_test_spec(metrics=("atomization_energy",), atom_energies=())
    with pytest.raises(ValueError, match="atomization_energy metric requires atom_energies"):
        spec.validate()


# (16-iv) non-finite atom_energies
def test_validate_nonfinite_atom_energies():
    spec = _make_test_spec(
        metrics=("atomization_energy",),
        atom_energies=(("H", float("nan")), ("O", -74.8)),
    )
    with pytest.raises(ValueError, match="must be finite"):
        spec.validate()


# (16-v) metric_kwargs set for metric not in self.metrics
def test_validate_metric_kwargs_for_absent_metric():
    spec = _make_test_spec(
        metrics=("total_energy",),
        metric_kwargs=(("density_rmse", (("foo", 1),)),),
    )
    with pytest.raises(ValueError, match="metric_kwargs.*is set but.*is not in self.metrics"):
        spec.validate()


# (16-vi) unknown metric_kwargs key
def test_validate_unknown_metric_kwargs_key():
    spec = _make_test_spec(
        metrics=("atomization_energy",),
        atom_energies=(("H", -0.5), ("O", -74.8)),
        metric_kwargs=(("atomization_energy", (("totally_bogus_key", 1),)),),
    )
    with pytest.raises(ValueError, match="unknown keys"):
        spec.validate()


# (16-vii) output_dir as file
def test_validate_output_dir_is_file():
    tmpdir = tempfile.mkdtemp()
    file_path = os.path.join(tmpdir, "not_a_dir.out")
    with open(file_path, "w") as f:
        f.write("x")
    spec = _make_test_spec(output_dir=file_path)
    with pytest.raises(ValueError, match="output_dir exists but is not a directory"):
        spec.validate()


# (16-viii) empty molecules
def test_validate_empty_molecules():
    spec = _make_test_spec(molecules=())
    with pytest.raises(ValueError, match="molecules must be non-empty"):
        spec.validate()


# (16-ix) empty metrics
def test_validate_empty_metrics():
    spec = _make_test_spec(metrics=())
    with pytest.raises(ValueError, match="metrics must be non-empty"):
        spec.validate()


# ---------------------------------------------------------------------------
# Tests 25-36: run_test integration + misc
# ---------------------------------------------------------------------------

def test_run_test_forwards_solver_config_to_density_metric(
    trained_checkpoint, h2o_data,
):
    """End-to-end: TestSpec.solver_config must reach DensityRMSEMetric.
    Equivalent specs that differ only in ``solver_config`` (None vs
    FIXED_J) must produce different ``density_rmse`` values for the same
    trained model — otherwise run_test is silently ignoring the solver
    distinction that training used."""
    import numpy as np
    from xcquinox.alec.config import MoleculeSpec
    from xcquinox.alec.solver import SolverConfig, SolverMode

    model_path, arch, atom_energies = trained_checkpoint
    rho_ref = np.asarray(h2o_data["rho_grid"])  # use PBE rho as the reference

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write a minimal external_data npz so DensityRMSEMetric has
        # rho_ref_grid available during eval.
        ext_path = os.path.join(tmpdir, "H2O.npz")
        np.savez(
            ext_path,
            rho_ref_grid=rho_ref,
            ref_density_method="pbe",
            dm_target=np.asarray(h2o_data["dm_pbe"]),
            E_ref_literature=float(h2o_data["E_pbe"]),
        )
        h2o = MoleculeSpec(
            name="H2O",
            atom="O 0 0 0; H 0 0 0.96; H 0.96 0 0",
            basis="sto-3g", charge=0, spin=0,
            atom_composition=(("H", 2), ("O", 1)),
            external_data_path=ext_path,
        )

        spec_oneshot = TestSpec.from_dicts(
            model_checkpoint=model_path, arch=arch, molecules=(h2o,),
            metrics=("density_rmse",),
            output_dir=os.path.join(tmpdir, "oneshot"),
            solver_config=None,
        )
        result_oneshot = run_test(spec_oneshot)

        spec_fixedj = TestSpec.from_dicts(
            model_checkpoint=model_path, arch=arch, molecules=(h2o,),
            metrics=("density_rmse",),
            output_dir=os.path.join(tmpdir, "fixedj"),
            solver_config=SolverConfig(mode=SolverMode.FIXED_J, max_cycles=3),
        )
        result_fixedj = run_test(spec_fixedj)

    rmse_oneshot = result_oneshot["per_molecule"][0]["density_rmse"]
    rmse_fixedj = result_fixedj["per_molecule"][0]["density_rmse"]
    assert rmse_oneshot is not None and math.isfinite(rmse_oneshot)
    assert rmse_fixedj is not None and math.isfinite(rmse_fixedj)
    assert abs(rmse_oneshot - rmse_fixedj) > 1e-10, (
        f"TestSpec.solver_config must propagate into DensityRMSEMetric; "
        f"oneshot={rmse_oneshot!r} fixed_j={rmse_fixedj!r}"
    )


# (25) run_test on 2-molecule spec returns {per_molecule, aggregate}
def test_run_test_basic(trained_checkpoint):
    model_path, arch, atom_energies = trained_checkpoint
    h = h_atom()
    h2o = h2o_molecule()
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = os.path.join(tmpdir, "out")
        spec = TestSpec.from_dicts(
            model_checkpoint=model_path,
            arch=arch,
            molecules=(h, h2o),
            metrics=("total_energy",),
            output_dir=outdir,
        )
        result = run_test(spec)
        assert "per_molecule" in result
        assert "aggregate" in result
        assert len(result["per_molecule"]) == 2


# (26) per_molecule.json roundtrips
@pytest.mark.slow
def test_per_molecule_json_roundtrip(trained_checkpoint):
    model_path, arch, atom_energies = trained_checkpoint
    h = h_atom()
    h2o = h2o_molecule()
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = os.path.join(tmpdir, "out")
        spec = TestSpec.from_dicts(
            model_checkpoint=model_path,
            arch=arch,
            molecules=(h, h2o),
            metrics=("total_energy",),
            output_dir=outdir,
        )
        result = run_test(spec)
        pm_path = os.path.join(outdir, "per_molecule.json")
        assert os.path.isfile(pm_path)
        with open(pm_path) as f:
            loaded = json.load(f)
        assert len(loaded) == 2
        assert loaded[0]["molecule"] == "H"
        assert loaded[1]["molecule"] == "H2O"


# (27) per_molecule.csv roundtrips
@pytest.mark.slow
def test_per_molecule_csv_roundtrip(trained_checkpoint):
    model_path, arch, atom_energies = trained_checkpoint
    h = h_atom()
    h2o = h2o_molecule()
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = os.path.join(tmpdir, "out")
        spec = TestSpec.from_dicts(
            model_checkpoint=model_path,
            arch=arch,
            molecules=(h, h2o),
            metrics=("total_energy",),
            output_dir=outdir,
        )
        run_test(spec)
        csv_path = os.path.join(outdir, "per_molecule.csv")
        assert os.path.isfile(csv_path)
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["molecule"] == "H"
        assert "E_total_nn" in rows[0]


# (28) aggregate.json includes mean/MAE/RMSE/max/count
@pytest.mark.slow
def test_aggregate_json_stats(trained_checkpoint):
    model_path, arch, atom_energies = trained_checkpoint
    h = h_atom()
    h2o = h2o_molecule()
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = os.path.join(tmpdir, "out")
        spec = TestSpec.from_dicts(
            model_checkpoint=model_path,
            arch=arch,
            molecules=(h, h2o),
            metrics=("total_energy",),
            output_dir=outdir,
        )
        result = run_test(spec)
        agg = result["aggregate"]
        assert "E_total_nn" in agg
        stats = agg["E_total_nn"]
        for key in ("mean", "MAE", "RMSE", "max", "count"):
            assert key in stats, f"aggregate E_total_nn missing {key}"
        assert stats["count"] == 2


# (29) test_metadata.json roundtrips with all fields
@pytest.mark.slow
def test_metadata_json_roundtrip(trained_checkpoint):
    model_path, arch, atom_energies = trained_checkpoint
    h = h_atom()
    h2o = h2o_molecule()
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = os.path.join(tmpdir, "out")
        spec = TestSpec.from_dicts(
            model_checkpoint=model_path,
            arch=arch,
            molecules=(h, h2o),
            metrics=("total_energy",),
            output_dir=outdir,
        )
        run_test(spec)
        md_path = os.path.join(outdir, "test_metadata.json")
        assert os.path.isfile(md_path)
        with open(md_path) as f:
            md = json.load(f)
        required_fields = {
            "arch_name", "model_checkpoint", "metrics", "molecules",
            "metric_kwargs", "atom_energies", "output_dir",
            "save_per_molecule", "save_aggregate",
            "timestamp", "duration_seconds",
        }
        missing = required_fields - set(md.keys())
        assert not missing, f"test_metadata.json missing keys: {missing}"


# (30) metric_kwargs override works
@pytest.mark.slow
def test_metric_kwargs_override(trained_checkpoint):
    model_path, arch, atom_energies = trained_checkpoint
    h = h_atom()
    h2o = h2o_molecule()
    ref_ae = {"H2O": 232.0}
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = os.path.join(tmpdir, "out")
        spec = TestSpec.from_dicts(
            model_checkpoint=model_path,
            arch=arch,
            molecules=(h, h2o),
            metrics=("atomization_energy",),
            atom_energies=atom_energies,
            metric_kwargs={
                "atomization_energy": {"reference_ae_kcalmol": ref_ae},
            },
            output_dir=outdir,
        )
        result = run_test(spec)
        h2o_result = result["per_molecule"][1]
        assert "AE_ref_kcalmol" in h2o_result
        assert h2o_result["AE_ref_kcalmol"] == 232.0


# (31) TestSpec.describe returns dict
def test_testspec_describe():
    spec = _make_test_spec()
    desc = spec.describe()
    assert isinstance(desc, dict)
    assert "arch" in desc
    assert "metrics" in desc


# (32) save_per_molecule=False omits per-molecule artifacts
@pytest.mark.slow
def test_save_per_molecule_false(trained_checkpoint):
    model_path, arch, atom_energies = trained_checkpoint
    h = h_atom()
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = os.path.join(tmpdir, "out")
        spec = TestSpec.from_dicts(
            model_checkpoint=model_path,
            arch=arch,
            molecules=(h,),
            metrics=("total_energy",),
            output_dir=outdir,
            save_per_molecule=False,
        )
        run_test(spec)
        assert not os.path.exists(os.path.join(outdir, "per_molecule.json"))
        assert not os.path.exists(os.path.join(outdir, "per_molecule.csv"))
        # aggregate should still exist
        assert os.path.isfile(os.path.join(outdir, "aggregate.json"))


# (33) save_aggregate=False omits aggregate artifact
@pytest.mark.slow
def test_save_aggregate_false(trained_checkpoint):
    model_path, arch, atom_energies = trained_checkpoint
    h = h_atom()
    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = os.path.join(tmpdir, "out")
        spec = TestSpec.from_dicts(
            model_checkpoint=model_path,
            arch=arch,
            molecules=(h,),
            metrics=("total_energy",),
            output_dir=outdir,
            save_aggregate=False,
        )
        run_test(spec)
        assert not os.path.exists(os.path.join(outdir, "aggregate.json"))
        # per_molecule should still exist
        assert os.path.isfile(os.path.join(outdir, "per_molecule.json"))


# (34) D-H4: constraint_violations raises on missing descriptor key
def test_constraint_violations_missing_descriptor_key():
    arch = ArchitectureConfig.from_spec(
        "desc_test", 2, 8, descriptors=["cusp"],
    )
    model = AlecGGAModel.from_arch(arch, seed=0)
    # mol_data without cusp_features should fail during assemble_descriptor_features
    fake_rho = jnp.array([0.1, 0.2, 0.3])
    fake_sigma = jnp.array([0.01, 0.02, 0.03])
    mol_data = {
        "rho_grid": fake_rho,
        "sigma_grid": fake_sigma,
        "cusp_features": None,
    }
    m = ConstraintViolationsMetric()
    with pytest.raises((TypeError, KeyError, Exception)):
        m.compute(model, mol_data)


# (35) E-H2: DensityRMSEMetric on UKS compound returns finite positive scalar
@pytest.mark.slow
def test_density_rmse_uks_compound(o_data):
    """DensityRMSEMetric on a UKS compound (O treated as non-atom).

    For this test, we create a fake compound mol_data based on o_data
    but with atom_composition summing to > 1, so the metric does not skip.
    """
    arch = _make_arch()
    model = _make_model(arch, seed=0)

    # Create a fake H2 molecule data based on h2o_data to get UKS behavior
    # Actually use o_data but mark it as a compound
    mol_data = dict(o_data)
    mol_data["atom_composition"] = (("O", 2),)
    mol_data["rho_ref_grid"] = mol_data["rho_grid"] + 0.001 * jnp.ones_like(
        mol_data["rho_grid"]
    )
    mol_data["ref_density_method"] = "hf"

    m = DensityRMSEMetric()
    result = m.compute(model, mol_data)
    assert result["density_rmse"] is not None
    assert result["density_rmse"] > 0.0
    assert math.isfinite(result["density_rmse"])


def test_density_rmse_carries_ref_method():
    """DensityRMSEMetric.compute output echoes mol_data['ref_density_method']."""
    from xcquinox.alec.evaluation import DensityRMSEMetric
    from xcquinox.alec.models import AlecGGAModel
    from xcquinox.alec.config import ArchitectureConfig
    from xcquinox.alec.data import precompute_fixed_density_data
    from xcquinox.alec.tests.fixtures.molecules import h2_molecule

    arch = ArchitectureConfig(
        name="t", depth=2, nodes=8, attention=False,
        descriptors=(), x_constraints=(), c_constraints=(),
        double_lob_clamp_allowed=False,
    )
    model = AlecGGAModel.from_arch(arch, seed=0)
    data = dict(precompute_fixed_density_data(h2_molecule()))
    data["rho_ref_grid"] = data["rho_grid"]
    data["ref_density_method"] = "hf"
    metric = DensityRMSEMetric()
    out = metric.compute(model, data)
    assert out["ref_density_method"] == "hf"


# (36) E-M6: DensityRMSEMetric on atom returns skip schema
def test_density_rmse_atom_skip(tiny_model, h_data):
    m = DensityRMSEMetric()
    result = m.compute(tiny_model, h_data)
    assert result["density_rmse"] is None
    assert result["density_l1"] is None
    assert result["skipped"] is True
    assert result["skip_reason"] == "atomic_system"


# (37) SCFConvergenceMetric returns sentinel for ONESHOT / no solver_config
def test_scf_convergence_metric_oneshot_sentinel(tiny_model, h2o_data):
    """ONESHOT (and solver_config=None) skip the SCF loop, so the metric
    must emit cycles_run=0 + scf_converged=True without raising."""
    from xcquinox.alec.evaluation import SCFConvergenceMetric
    from xcquinox.alec.solver import SolverConfig, SolverMode
    m = SCFConvergenceMetric()
    out_none = m.compute(tiny_model, h2o_data, solver_config=None)
    assert out_none == {"cycles_run": 0, "scf_converged": True}
    out_oneshot = m.compute(
        tiny_model, h2o_data,
        solver_config=SolverConfig(mode=SolverMode.ONESHOT),
    )
    assert out_oneshot == {"cycles_run": 0, "scf_converged": True}


# (38) SCFConvergenceMetric records per-cycle |E_n - E_final| trace under FIXED_J
def test_scf_convergence_metric_records_residual_trace(tiny_model, h2o_data):
    """Under a real SCF backend (FIXED_J via pyscfad), the metric must
    emit ``scf_energy_residual_<i>`` keys for each executed cycle.

    Each residual is |E_n - E_final| -- so the residual at the final
    cycle should be much smaller than at cycle 0 (energy decay during
    SCF). This catches both (a) the pyscfad backend forgetting to
    populate energy_trace and (b) the metric forgetting to surface it.
    """
    from xcquinox.alec.evaluation import SCFConvergenceMetric
    from xcquinox.alec.solver import SolverConfig, SolverMode
    m = SCFConvergenceMetric()
    cfg = SolverConfig(mode=SolverMode.FIXED_J, max_cycles=4, conv_tol=1e-6)
    out = m.compute(tiny_model, h2o_data, solver_config=cfg)
    # Core fields always present
    assert "cycles_run" in out
    assert "scf_converged" in out
    assert "scf_total_energy" in out
    # Per-cycle residual fields when the backend recorded a trace
    residual_keys = [k for k in out if k.startswith("scf_energy_residual_")]
    assert residual_keys, (
        "SCFConvergenceMetric did not surface scf_energy_residual_<i> rows; "
        "energy_trace likely not populated by the pyscfad backend."
    )
    # All residuals are non-negative finite floats; final cycle residual
    # is the smallest (decay toward convergence).
    indices = sorted(int(k.split("_")[-1]) for k in residual_keys)
    residuals = [out[f"scf_energy_residual_{i}"] for i in indices]
    for r in residuals:
        assert math.isfinite(r) and r >= 0.0
    # The trace must show actual decay from the first to the last cycle.
    assert residuals[-1] <= residuals[0] + 1e-12


# ---------------------------------------------------------------------------
# DATA-04: _aggregate_results coverage transparency
# ---------------------------------------------------------------------------

def test_aggregate_results_records_n_total_and_n_skipped():
    """DATA-04: aggregate stats must expose n_total and n_skipped per metric.

    Scenario: 5 molecules, but density_rmse is None for 3 of them (e.g. atoms
    that were skipped).  The aggregate for density_rmse should include:
      - ``count`` (or ``n_included``) == 2   (molecules that contributed)
      - ``n_total`` == 5                      (total molecules considered)
      - ``n_skipped`` == 3                    (molecules that were excluded)

    Without these fields a 2-of-5 aggregate is indistinguishable from a
    5-of-5 aggregate, so a near-empty result can read as a full-population
    statistic.
    """
    from xcquinox.alec.evaluation import _aggregate_results

    per_molecule = [
        {"molecule": "H",   "density_rmse": None,  "E_total_nn": -0.5},
        {"molecule": "He",  "density_rmse": None,  "E_total_nn": -2.9},
        {"molecule": "Li",  "density_rmse": None,  "E_total_nn": -7.4},
        {"molecule": "H2",  "density_rmse": 0.012, "E_total_nn": -1.1},
        {"molecule": "LiH", "density_rmse": 0.034, "E_total_nn": -8.0},
    ]

    agg = _aggregate_results(per_molecule)

    # density_rmse: only 2 of 5 contributed
    drms = agg["density_rmse"]
    assert drms["count"] == 2, (
        f"expected count=2, got {drms['count']}"
    )
    assert drms["n_total"] == 5, (
        f"expected n_total=5, got {drms.get('n_total')}"
    )
    assert drms["n_skipped"] == 3, (
        f"expected n_skipped=3, got {drms.get('n_skipped')}"
    )

    # E_total_nn: all 5 contributed; n_skipped must be 0
    etn = agg["E_total_nn"]
    assert etn["count"] == 5, f"expected count=5, got {etn['count']}"
    assert etn["n_total"] == 5, f"expected n_total=5, got {etn.get('n_total')}"
    assert etn["n_skipped"] == 0, f"expected n_skipped=0, got {etn.get('n_skipped')}"

    # Numeric stats for the included molecules must be unchanged
    assert math.isclose(drms["mean"], (0.012 + 0.034) / 2)
    assert math.isclose(drms["MAE"],  (0.012 + 0.034) / 2)
    assert math.isclose(
        drms["RMSE"],
        math.sqrt((0.012**2 + 0.034**2) / 2),
    )
    assert math.isclose(drms["max"],  0.034)
