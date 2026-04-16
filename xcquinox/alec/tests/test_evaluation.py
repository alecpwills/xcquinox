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

# (13) METRIC_REGISTRY has exactly 4 entries
def test_metric_registry_has_4_entries():
    assert len(METRIC_REGISTRY) == 4


# (14) list_metrics returns sorted names
def test_list_metrics_sorted():
    names = list_metrics()
    assert names == sorted(names)
    assert set(names) == {"total_energy", "atomization_energy", "density_rmse", "constraint_violations"}


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

# (25) run_test on 2-molecule spec returns {per_molecule, aggregate}
@pytest.mark.slow
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
