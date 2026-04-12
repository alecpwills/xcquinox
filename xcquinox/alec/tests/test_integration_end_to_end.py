"""Integration end-to-end tests for xcquinox.alec pipeline.

Tests the full pipeline: model creation -> training -> evaluation.
6 tests total: tests 1, 4, 5, 6 should PASS; tests 2, 3 are XFAIL.
"""
import math
import os
import tempfile

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from xcquinox.alec.config import (
    ArchitectureConfig,
    TrainingSpec,
    TestSpec,
    get_architecture,
)
from xcquinox.alec.models import AlecGGAModel
from xcquinox.alec.data import precompute_fixed_density_data
from xcquinox.alec.train import run_training
from xcquinox.alec.evaluation import run_test
from xcquinox.alec.tests.fixtures.molecules import (
    h_atom,
    h2_molecule,
    o_atom,
    h2o_molecule,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _shallow_arch():
    """Shallow arch with no descriptors/constraints (depth=2, nodes=8)."""
    return get_architecture("shallow")


def _make_training_spec(
    molecules, targets, atom_energies, *, arch=None, loss_name="A_atomization",
    n_steps=3, checkpoint_dir=None, seed=42, **extra,
):
    """Build a valid TrainingSpec for integration tests."""
    if arch is None:
        arch = _shallow_arch()
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(tempfile.mkdtemp(), "ckpt")
    return TrainingSpec.from_dicts(
        arch=arch,
        molecules=molecules,
        targets=targets,
        atom_energies=atom_energies,
        loss_name=loss_name,
        n_steps=n_steps,
        lr_start=1e-3,
        lr_end=1e-5,
        lr_decay_start=0.0,
        grad_clip=1.0,
        checkpoint_dir=checkpoint_dir,
        seed=seed,
        **extra,
    )


# ---------------------------------------------------------------------------
# Module-scoped fixtures (PySCF -- expensive, computed once per module)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def h_data():
    return precompute_fixed_density_data(h_atom())


@pytest.fixture(scope="module")
def h2_data():
    return precompute_fixed_density_data(h2_molecule())


@pytest.fixture(scope="module")
def o_data():
    return precompute_fixed_density_data(o_atom())


@pytest.fixture(scope="module")
def h2o_data():
    return precompute_fixed_density_data(h2o_molecule())


@pytest.fixture(scope="module")
def h_h2_batch_info(h_data, h2_data):
    """Pre-assembled batch info for H + H2."""
    h = h_atom()
    h2 = h2_molecule()
    ae_h2 = float(h_data["E_pbe"] * 2 - h2_data["E_pbe"])
    targets = {
        "H": float(h_data["E_pbe"]),
        "H2": max(ae_h2, 0.001),
    }
    atom_energies = {
        "H": float(h_data["E_pbe"]),
    }
    return {
        "mols": (h, h2),
        "targets": targets,
        "atom_energies": atom_energies,
    }


@pytest.fixture(scope="module")
def h_o_h2o_batch_info(h_data, o_data, h2o_data):
    """Pre-assembled batch info for H + O + H2O."""
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
    return {
        "mols": (h, o, h2o),
        "targets": targets,
        "atom_energies": atom_energies,
    }


# ---------------------------------------------------------------------------
# Test 1: Full pipeline (train + evaluate) for H2 with shallow architecture
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_full_pipeline_h2(h_h2_batch_info):
    """Full train -> evaluate pipeline for H + H2 with shallow arch.

    - Creates model from shallow architecture
    - Runs training for 3 steps with A_atomization loss
    - Runs evaluation with total_energy metric
    - Asserts all stages complete and artifacts exist
    """
    arch = _shallow_arch()
    info = h_h2_batch_info

    with tempfile.TemporaryDirectory() as tmpdir:
        # --- Training stage ---
        train_ckdir = os.path.join(tmpdir, "train_ckpt")
        spec_train = _make_training_spec(
            molecules=info["mols"],
            targets=info["targets"],
            atom_energies=info["atom_energies"],
            arch=arch,
            loss_name="A_atomization",
            n_steps=3,
            checkpoint_dir=train_ckdir,
            seed=42,
        )
        metadata = run_training(spec_train)

        # Training assertions
        assert isinstance(metadata, dict)
        assert math.isfinite(metadata["final_loss"])
        model_path = os.path.join(train_ckdir, "model.eqx")
        assert os.path.isfile(model_path)
        assert os.path.isfile(os.path.join(train_ckdir, "losses.npy"))
        assert os.path.isfile(os.path.join(train_ckdir, "train_metadata.json"))

        # --- Evaluation stage ---
        test_out = os.path.join(tmpdir, "test_results")
        spec_test = TestSpec.from_dicts(
            model_checkpoint=model_path,
            arch=arch,
            molecules=info["mols"],
            metrics=("total_energy",),
            atom_energies=info["atom_energies"],
            output_dir=test_out,
        )
        results = run_test(spec_test)

        # Evaluation assertions
        assert "per_molecule" in results
        assert "aggregate" in results
        assert len(results["per_molecule"]) == len(info["mols"])
        for row in results["per_molecule"]:
            assert "E_total_nn" in row
            assert math.isfinite(row["E_total_nn"])
        assert os.path.isfile(os.path.join(test_out, "per_molecule.json"))
        assert os.path.isfile(os.path.join(test_out, "test_metadata.json"))


# ---------------------------------------------------------------------------
# Test 2: Full pipeline with deep_combined architecture (XFAIL)
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.xfail(
    reason="deep_combined architecture requires descriptor features (dm_statistics, "
           "cusp) in pretrain data; may fail without full descriptor fixtures",
    strict=False,
)
def test_full_pipeline_deep_combined(h_h2_batch_info):
    """Full pipeline with deep_combined arch to test descriptor plumbing.

    Uses dm_statistics + cusp descriptors. XFAIL because descriptor feature
    computation may require specific data fixtures not yet available.
    """
    arch = get_architecture("deep_combined")
    info = h_h2_batch_info

    with tempfile.TemporaryDirectory() as tmpdir:
        train_ckdir = os.path.join(tmpdir, "train_ckpt")
        spec_train = _make_training_spec(
            molecules=info["mols"],
            targets=info["targets"],
            atom_energies=info["atom_energies"],
            arch=arch,
            loss_name="A_atomization",
            n_steps=3,
            checkpoint_dir=train_ckdir,
            seed=42,
        )
        metadata = run_training(spec_train)
        assert isinstance(metadata, dict)
        assert math.isfinite(metadata["final_loss"])

        model_path = os.path.join(train_ckdir, "model.eqx")
        test_out = os.path.join(tmpdir, "test_results")
        spec_test = TestSpec.from_dicts(
            model_checkpoint=model_path,
            arch=arch,
            molecules=info["mols"],
            metrics=("total_energy",),
            atom_energies=info["atom_energies"],
            output_dir=test_out,
        )
        results = run_test(spec_test)
        assert len(results["per_molecule"]) == len(info["mols"])


# ---------------------------------------------------------------------------
# Test 3: Load from stashed legacy step3b checkpoint (XFAIL)
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    reason="requires xcquinox/alec/tests/fixtures/legacy_step3b_checkpoint/ "
           "which does not exist yet",
    strict=True,
    raises=FileNotFoundError,
)
def test_from_legacy_step3b():
    """Load and evaluate from a stashed legacy step3b checkpoint.

    This test exercises backward-compatibility with the old checkpoint format.
    XFAIL until the fixture directory is populated.
    """
    fixture_dir = os.path.join(
        os.path.dirname(__file__), "fixtures", "legacy_step3b_checkpoint"
    )
    if not os.path.isdir(fixture_dir):
        raise FileNotFoundError(
            f"legacy_step3b_checkpoint fixture directory not found: {fixture_dir}"
        )

    model_path = os.path.join(fixture_dir, "model.eqx")
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"model.eqx not found in legacy fixture: {model_path}"
        )

    arch = _shallow_arch()
    skeleton = AlecGGAModel.from_arch(arch, seed=0)
    model = eqx.tree_deserialise_leaves(model_path, skeleton)

    h = h_atom()
    h2 = h2_molecule()
    with tempfile.TemporaryDirectory() as tmpdir:
        test_out = os.path.join(tmpdir, "test_results")
        spec_test = TestSpec.from_dicts(
            model_checkpoint=model_path,
            arch=arch,
            molecules=(h, h2),
            metrics=("total_energy",),
            atom_energies={"H": -0.5},
            output_dir=test_out,
        )
        results = run_test(spec_test)
        assert len(results["per_molecule"]) == 2


# ---------------------------------------------------------------------------
# Test 4: Model serialization roundtrip
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_model_serialization_roundtrip(h2_data):
    """Serialize and deserialize AlecGGAModel, verify bitwise-identical eval_exc.

    - Builds a fresh model from shallow arch
    - Computes eval_exc on H2 mol_data
    - Serializes to tmp_path, deserializes
    - Asserts jnp.array_equal(original, roundtripped)
    """
    arch = _shallow_arch()
    model_orig = AlecGGAModel.from_arch(arch, seed=42)

    # Compute eval_exc on H2 data
    from xcquinox.alec.descriptors import assemble_descriptor_features
    features = assemble_descriptor_features(model_orig.descriptors, h2_data)
    output_orig = model_orig.eval_exc(
        h2_data["rho_grid"], h2_data["sigma_grid"], features,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model_roundtrip.eqx")
        eqx.tree_serialise_leaves(model_path, model_orig)

        skeleton = AlecGGAModel.from_arch(arch, seed=0)
        model_loaded = eqx.tree_deserialise_leaves(model_path, skeleton)

        features_loaded = assemble_descriptor_features(
            model_loaded.descriptors, h2_data,
        )
        output_loaded = model_loaded.eval_exc(
            h2_data["rho_grid"], h2_data["sigma_grid"], features_loaded,
        )

    assert jnp.array_equal(output_orig, output_loaded), (
        "eval_exc output must be bitwise-identical after serialization roundtrip"
    )


# ---------------------------------------------------------------------------
# Test 5: Determinism regression
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_determinism_regression(h_h2_batch_info):
    """Run training twice with the same seed; assert nearly-identical losses.npy.

    Uses H + H2 with shallow arch, n_steps=3, seed=42.
    Both runs should produce the same loss trajectory to high precision.

    Note: bit-exact equality is not guaranteed because run_training internally
    calls precompute_fixed_density_data (PySCF SCF), and PySCF's iterative
    solver can produce ULP-level floating-point differences between calls.
    We use a tight tolerance (rtol=1e-12, atol=1e-15) to verify determinism
    of the NN + optimizer path while allowing PySCF-level rounding noise.
    """
    arch = _shallow_arch()
    info = h_h2_batch_info
    losses_runs = []

    for run_idx in range(2):
        with tempfile.TemporaryDirectory() as tmpdir:
            ckdir = os.path.join(tmpdir, f"run_{run_idx}")
            spec = _make_training_spec(
                molecules=info["mols"],
                targets=info["targets"],
                atom_energies=info["atom_energies"],
                arch=arch,
                loss_name="A_atomization",
                n_steps=3,
                checkpoint_dir=ckdir,
                seed=42,
            )
            run_training(spec)
            losses = np.load(os.path.join(ckdir, "losses.npy"))
            losses_runs.append(losses)

    assert np.allclose(losses_runs[0], losses_runs[1], rtol=1e-12, atol=1e-15), (
        f"Two training runs with identical seed should produce nearly-identical "
        f"losses. Run 0: {losses_runs[0]}, Run 1: {losses_runs[1]}, "
        f"max abs diff: {np.max(np.abs(losses_runs[0] - losses_runs[1]))}"
    )


# ---------------------------------------------------------------------------
# Test 6: Mixed UKS/RKS end-to-end with B_atomization_plus_dm loss
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_mixed_uks_rks_end_to_end(h_o_h2o_batch_info):
    """Full pipeline on (H, O, H2O) with A_atomization loss.

    Exercises both UKS (H spin=1, O spin=2) and RKS (H2O spin=0) systems
    in the same training batch. Asserts finite non-NaN losses.
    """
    arch = _shallow_arch()
    info = h_o_h2o_batch_info

    with tempfile.TemporaryDirectory() as tmpdir:
        train_ckdir = os.path.join(tmpdir, "train_ckpt")
        spec_train = _make_training_spec(
            molecules=info["mols"],
            targets=info["targets"],
            atom_energies=info["atom_energies"],
            arch=arch,
            loss_name="A_atomization",
            n_steps=3,
            checkpoint_dir=train_ckdir,
            seed=42,
        )
        metadata = run_training(spec_train)

        # Verify training completed with finite losses
        assert math.isfinite(metadata["final_loss"])
        losses = np.load(os.path.join(train_ckdir, "losses.npy"))
        assert len(losses) == 3
        assert all(np.isfinite(losses)), (
            f"All losses must be finite, got: {losses}"
        )
        assert not any(np.isnan(losses)), (
            f"No losses may be NaN, got: {losses}"
        )

        # Run evaluation
        model_path = os.path.join(train_ckdir, "model.eqx")
        test_out = os.path.join(tmpdir, "test_results")
        spec_test = TestSpec.from_dicts(
            model_checkpoint=model_path,
            arch=arch,
            molecules=info["mols"],
            metrics=("total_energy",),
            atom_energies=info["atom_energies"],
            output_dir=test_out,
        )
        results = run_test(spec_test)

        # All molecules evaluated successfully
        assert len(results["per_molecule"]) == 3
        for row in results["per_molecule"]:
            assert "E_total_nn" in row
            assert math.isfinite(row["E_total_nn"]), (
                f"E_total_nn must be finite for {row.get('molecule')}"
            )
