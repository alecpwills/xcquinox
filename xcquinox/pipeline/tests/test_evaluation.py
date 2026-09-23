"""Tests for xcquinox.pipeline.evaluation -- Metric ABC, 4 metrics, run_test.

Implements Task 5.3 test suite: 36 tests.

Tests 1-12: Per-metric registry+compute+schema (4 metrics x 3 = 12).
Tests 13-15: Registry-level.
Tests 16-24: TestSpec.validate negative paths (9 total).
Tests 25-36: run_test integration + misc.
"""
import json
import math
import os
import tempfile

import jax.numpy as jnp
import numpy as np
import pytest
import equinox as eqx

from xcquinox.pipeline.config import (
    ArchitectureConfig,
    TestSpec,
    TrainingSpec,
)
from xcquinox.pipeline.evaluation import (
    AtomizationEnergyMetric,
    ConstraintViolationsMetric,
    DensityRMSEMetric,
    TotalEnergyMetric,
    make_metric,
    pbe_density_eps,
    run_test,
)
from xcquinox.pipeline.models import AlecGGAModel
from xcquinox.pipeline.data import precompute_fixed_density_data
from xcquinox.pipeline.tests.fixtures.molecules import (
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
    from xcquinox.pipeline.train import run_training

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


# (4b) Compute on tiny model returns dict
def test_atomization_energy_compute(tiny_model, h_data, o_data, h2o_data):
    ae = {"H": float(h_data["E_pbe"]), "O": float(o_data["E_pbe"])}
    m = AtomizationEnergyMetric(atom_energies=ae)
    result = m.compute(tiny_model, h2o_data)
    assert isinstance(result, dict)
    assert "AE_nn" in result
    assert math.isfinite(result["AE_nn"])


# (4c) Output keys match schema (with reference)


# ---------------------------------------------------------------------------
# Tests 7-9: DensityRMSEMetric
# ---------------------------------------------------------------------------

# (7a) Registry roundtrip


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


def test_pbe_reference_metric_computes_pbe_ae_error(tiny_model, h_data, o_data, h2o_data):
    """PBEReferenceMetric reports the PBE-level atomization energy and its
    error vs literature reference, using ONLY E_pbe from mol_data + an
    ``atom_energies`` dict (PBE-consistent atomic totals). Zero
    NN forward pass. This is the 'what if we just used PBE?' baseline
    shown alongside trained-NN results on the notebook's comparison plots."""
    from xcquinox.pipeline.evaluation import PBEReferenceMetric
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
    """PBEReferenceMetric must not depend on the NN model, it's a
    hardware-free baseline. Calling with two different models on the
    same mol_data must produce identical output."""
    from xcquinox.pipeline.evaluation import PBEReferenceMetric
    atom_energies = {"H": float(h_data["E_pbe"]), "O": float(o_data["E_pbe"])}
    m = PBEReferenceMetric(atom_energies=atom_energies)
    mol_data = dict(h2o_data); mol_data["name"] = "H2O"
    out1 = m.compute(tiny_model, mol_data)
    # Any other "model" object, PBEReferenceMetric should ignore it.
    out2 = m.compute(None, mol_data)
    for k, v in out1.items():
        if isinstance(v, float):
            assert math.isclose(v, out2[k]), f"{k}: {v} vs {out2[k]}"


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


# DFS Letter Eq. 20 support: eps = sum(w|drho|)/N_e with N_e the quadrature
# integral of the reference density (the dpyscf per-electron convention).
def test_pbe_density_eps_closed_form():
    md = {
        "rho_ref_grid": np.array([2.0, 1.0]),
        "rho_grid": np.array([2.5, 0.5]),
        "grid_weights": np.array([3.0, 1.0]),
    }
    eps, n_e, wsum = pbe_density_eps(md)
    # sum(w|diff|) = 3*0.5 + 1*0.5 = 2 ; N_e = sum(w*rho_ref) = 7 ; wsum = 4
    assert eps == pytest.approx(2.0 / 7.0)
    assert n_e == pytest.approx(7.0)
    assert wsum == pytest.approx(4.0)
    # deliberately distinct from the volume-averaged L1 (= 0.5) so the two
    # normalizations cannot be confused by a passing test
    assert abs(eps - 0.5) > 0.1
    # missing reference -> the historical all-None skip semantics
    assert pbe_density_eps({"rho_ref_grid": None}) == (None, None, None)


def test_density_rmse_emits_eps_and_bookkeeping(tiny_model, h2o_data):
    mol_data = dict(h2o_data)
    mol_data["rho_ref_grid"] = mol_data["rho_grid"] + 0.001 * jnp.ones_like(
        mol_data["rho_grid"]
    )
    mol_data["ref_density_method"] = "hf"
    out = DensityRMSEMetric().compute(tiny_model, mol_data)
    w = np.asarray(mol_data["grid_weights"])
    n_e = float(np.sum(w * np.asarray(mol_data["rho_ref_grid"])))
    wsum = float(np.sum(w))
    assert out["n_electrons"] == pytest.approx(n_e, rel=1e-10)
    assert out["grid_weight_sum"] == pytest.approx(wsum, rel=1e-10)
    # the REAL PBE density integrates to ~10 electrons for H2O; the synthetic
    # reference (+0.001 everywhere) inflates N_e by exactly 0.001 * sum(w)
    n_e_pbe = float(np.sum(w * np.asarray(mol_data["rho_grid"])))
    assert 9.5 < n_e_pbe < 10.5
    assert out["n_electrons"] == pytest.approx(n_e_pbe + 0.001 * wsum,
                                               rel=1e-8)
    # exact identity: eps (per-electron L1) = (volume-averaged L1) * wsum/N_e
    assert out["density_eps_l1"] == pytest.approx(
        out["density_l1"] * wsum / n_e, rel=1e-10)
    assert out["density_eps_l1_pbe"] == pytest.approx(
        out["density_l1_pbe"] * wsum / n_e, rel=1e-10)


def test_aggregate_excludes_quadrature_bookkeeping():
    """n_electrons / grid_weight_sum are bookkeeping, not error metrics --
    aggregate.json must not carry their mean/MAE/RMSE as pseudo-metrics,
    while the genuine eps error metric still aggregates."""
    from xcquinox.pipeline.evaluation import _aggregate_results
    agg = _aggregate_results([
        {"molecule": "h2o", "density_eps_l1": 1e-3, "n_electrons": 10.0,
         "grid_weight_sum": 2.1e5},
        {"molecule": "nh3", "density_eps_l1": 2e-3, "n_electrons": 10.0,
         "grid_weight_sum": 1.9e5},
    ])
    assert "density_eps_l1" in agg
    assert agg["density_eps_l1"]["mean"] == pytest.approx(1.5e-3)
    assert "n_electrons" not in agg
    assert "grid_weight_sum" not in agg


# ---------------------------------------------------------------------------
# Tests 10-12: ConstraintViolationsMetric
# ---------------------------------------------------------------------------

# (10a) Registry roundtrip


# (10b) Compute on tiny model returns dict
def test_constraint_violations_compute(tiny_model, h2o_data):
    m = ConstraintViolationsMetric()
    result = m.compute(tiny_model, h2o_data)
    assert isinstance(result, dict)
    # shallow arch has no constraints, so result dict should be empty
    assert len(result) == 0


# (10c) Compute with constrained arch returns expected keys


# ---------------------------------------------------------------------------
# Tests 13-15: Registry-level
# ---------------------------------------------------------------------------

# (13) METRIC_REGISTRY has exactly 5 entries
# (was 4 prior to the 2026-04-24 addition of PBEReferenceMetric, the
# model-independent PBE-baseline metric used by the notebook's comparison plots)


# (14) list_metrics returns sorted names


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


# (16-ii) unknown metric name


# (16-iii) atomization_energy without atom_energies
def test_validate_ae_without_atom_energies():
    spec = _make_test_spec(metrics=("atomization_energy",), atom_energies=())
    with pytest.raises(ValueError, match="atomization_energy metric requires atom_energies"):
        spec.validate()


# (16-iv) non-finite atom_energies


# (16-v) metric_kwargs set for metric not in self.metrics


# (16-vi) unknown metric_kwargs key


# (16-vii) output_dir as file


# (16-viii) empty molecules


# (16-ix) empty metrics


# ---------------------------------------------------------------------------
# Tests 25-36: run_test integration + misc
# ---------------------------------------------------------------------------


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


# (30) metric_kwargs override works


# (31) TestSpec.describe returns dict


# (32) save_per_molecule=False omits per-molecule artifacts


# (33) save_aggregate=False omits aggregate artifact


# (34) D-H4: constraint_violations raises on missing descriptor key


# (35) E-H2: DensityRMSEMetric on UKS compound returns finite positive scalar


# (36) E-M6: DensityRMSEMetric on atom returns skip schema


# (37) SCFConvergenceMetric returns sentinel for ONESHOT / no solver_config


# (38) SCFConvergenceMetric records per-cycle |E_n - E_final| trace under FIXED_J
def test_scf_convergence_metric_records_residual_trace(tiny_model, h2o_data):
    """Under a real SCF backend (FIXED_J on the default manual backend), the metric must
    emit ``scf_energy_residual_<i>`` keys for each executed cycle.

    Each residual is |E_n - E_final| -- so the residual at the final
    cycle should be much smaller than at cycle 0 (energy decay during
    SCF). This catches both (a) the backend forgetting to populate
    energy_trace and (b) the metric forgetting to surface it.
    """
    from xcquinox.pipeline.evaluation import SCFConvergenceMetric
    from xcquinox.pipeline.solver import SolverConfig, SolverMode
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


# ---------------------------------------------------------------------------
# pbe_density_errors -- model-free PBE-vs-reference baseline
# ---------------------------------------------------------------------------


def test_density_rmse_metric_emits_pbe_channel(tiny_model, h2o_data):
    import jax.numpy as jnp
    m = DensityRMSEMetric()
    h2o = dict(h2o_data)
    h2o["rho_ref_grid"] = jnp.asarray(h2o["rho_grid"]) * 1.01
    h2o["ref_density_method"] = "ccsd"
    out = m.compute(tiny_model, h2o)
    assert out["density_rmse_pbe"] is not None and out["density_rmse_pbe"] > 0
    assert out["density_l1_pbe"] is not None
    # skip branches carry the new keys as None (schema stability)
    atom = dict(h2o, atom_composition=(("O", 1),))
    out_atom = m.compute(tiny_model, atom)
    assert out_atom["density_rmse_pbe"] is None
    no_ref = dict(h2o)
    no_ref["rho_ref_grid"] = None
    out_no_ref = m.compute(tiny_model, no_ref)
    assert out_no_ref["density_rmse_pbe"] is None


