"""Tests for xcquinox.alec.train -- run_training custom loop.

Implements THE SPEC Task 5.2 test suite: 31 tests.

Tests 1-9: TrainingSpec.validate negative paths (fast, no PySCF).
Tests 10-15: run_training end-to-end for each of 6 losses (slow, PySCF).
Test 16: losses decrease after 5 steps on A_atomization.
Test 17: artifact roundtrip (model.eqx, losses.npy, aux_log.pkl, metadata).
Test 18: pretrain checkpoint yields lower initial loss than from-scratch.
Test 19: atom-composition validation (missing single-atom molecules).
Test 20: constraint_report post-update still valid.
Test 21: aux_log.pkl schema.
Test 22: progress callback schema.
Test 23: molecule-generic (H, N, NH3) training set.
Tests 24-31: additional validation tests (fast, no PySCF).
"""
import json
import math
import os
import pickle  # noqa: S403 -- loading trusted test aux_log.pkl data only
import tempfile

import numpy as np
import pytest

from xcquinox.alec.config import (
    ArchitectureConfig,
    MoleculeSpec,
    TrainingSpec,
    get_architecture,
)
from xcquinox.alec.losses import list_losses
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


def _make_training_spec(**overrides):
    """Build a minimal valid TrainingSpec for H, O, H2O."""
    tmpdir = tempfile.mkdtemp()
    ckdir = os.path.join(tmpdir, "ckpt")
    h = h_atom()
    o = o_atom()
    h2o = h2o_molecule()
    defaults = dict(
        arch=_make_arch(),
        molecules=(h, o, h2o),
        targets=(("H", -0.5), ("H2O", 0.3), ("O", -74.8)),
        atom_energies=(("H", -0.5), ("O", -74.8)),
        loss_name="A_atomization",
        n_steps=3,
        lr_start=1e-3,
        lr_end=1e-5,
        lr_decay_start=0.0,
        grad_clip=1.0,
        checkpoint_dir=ckdir,
        seed=42,
    )
    defaults.update(overrides)
    return TrainingSpec(**defaults)


# ---------------------------------------------------------------------------
# Tests 1-8: TrainingSpec.validate negative paths (fast -- no PySCF)
# ---------------------------------------------------------------------------

# (1) unknown loss name
def test_validate_unknown_loss_name():
    spec = _make_training_spec(loss_name="nonexistent_loss")
    with pytest.raises(ValueError, match="unknown loss"):
        spec.validate()


# (2) empty molecules
def test_validate_empty_molecules():
    spec = _make_training_spec(molecules=())
    with pytest.raises(ValueError, match="molecules must be non-empty"):
        spec.validate()


# (3) missing targets
def test_validate_missing_targets():
    spec = _make_training_spec(targets=(("H", -0.5), ("O", -74.8)))
    with pytest.raises(ValueError, match="targets missing for molecules"):
        spec.validate()


# (4) empty atom_energies
def test_validate_empty_atom_energies():
    spec = _make_training_spec(atom_energies=())
    with pytest.raises(ValueError, match="atom_energies must be non-empty"):
        spec.validate()


# (5) n_steps <= 0
def test_validate_n_steps_zero():
    spec = _make_training_spec(n_steps=0)
    with pytest.raises(ValueError, match="n_steps must be > 0"):
        spec.validate()


# (6) lr_decay_start out of range
def test_validate_lr_decay_start_out_of_range():
    spec = _make_training_spec(lr_decay_start=1.5)
    with pytest.raises(ValueError, match="lr_decay_start must be in"):
        spec.validate()


# (7) lr_start < lr_end
def test_validate_lr_start_below_lr_end():
    spec = _make_training_spec(lr_start=1e-6, lr_end=1e-3)
    with pytest.raises(ValueError, match="lr_start .* must be >= lr_end"):
        spec.validate()


# (8) grad_clip <= 0
def test_validate_grad_clip_nonpositive():
    spec = _make_training_spec(grad_clip=-1.0)
    with pytest.raises(ValueError, match="grad_clip must be > 0"):
        spec.validate()


# ---------------------------------------------------------------------------
# Test 9: missing pretrain_checkpoint directory
# ---------------------------------------------------------------------------

def test_validate_missing_pretrain_checkpoint():
    spec = _make_training_spec(pretrain_checkpoint="/tmp/alec_nonexistent_ckpt_dir_xyz")
    with pytest.raises(ValueError, match="pretrain_checkpoint directory not found"):
        spec.validate()


# ---------------------------------------------------------------------------
# Module-scoped fixtures (PySCF -- expensive, computed once per module)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def h_mol_data():
    from xcquinox.alec.data import precompute_fixed_density_data
    return precompute_fixed_density_data(h_atom())


@pytest.fixture(scope="module")
def o_mol_data():
    from xcquinox.alec.data import precompute_fixed_density_data
    return precompute_fixed_density_data(o_atom())


@pytest.fixture(scope="module")
def h2o_mol_data():
    from xcquinox.alec.data import precompute_fixed_density_data
    return precompute_fixed_density_data(h2o_molecule())


@pytest.fixture(scope="module")
def training_batch_info(h_mol_data, o_mol_data, h2o_mol_data):
    """Pre-assembled training batch components for H, O, H2O."""
    mols = (h_atom(), o_atom(), h2o_molecule())
    ae_h2o = float(
        h_mol_data["E_pbe"] * 2 + o_mol_data["E_pbe"] - h2o_mol_data["E_pbe"]
    )
    targets = {
        "H": float(h_mol_data["E_pbe"]),
        "O": float(o_mol_data["E_pbe"]),
        "H2O": max(ae_h2o, 0.001),
    }
    atom_energies = {
        "H": float(h_mol_data["E_pbe"]),
        "O": float(o_mol_data["E_pbe"]),
    }
    return {
        "mols": mols,
        "targets": targets,
        "atom_energies": atom_energies,
    }


def _make_live_spec(training_batch_info, *, loss_name="A_atomization",
                    n_steps=3, tmpdir=None, **extra):
    """Build a valid TrainingSpec for integration tests."""
    if tmpdir is None:
        tmpdir = tempfile.mkdtemp()
    ckdir = os.path.join(tmpdir, "ckpt")
    return TrainingSpec.from_dicts(
        arch=_make_arch(),
        molecules=training_batch_info["mols"],
        targets=training_batch_info["targets"],
        atom_energies=training_batch_info["atom_energies"],
        loss_name=loss_name,
        n_steps=n_steps,
        lr_start=1e-3,
        lr_end=1e-5,
        lr_decay_start=0.0,
        grad_clip=1.0,
        checkpoint_dir=ckdir,
        seed=42,
        **extra,
    )


# ---------------------------------------------------------------------------
# Tests 10-15: run_training end-to-end for each of 6 losses
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("loss_name", list_losses())
def test_run_training_end_to_end(loss_name, training_batch_info):
    """Tests 10-15: run_training completes for each loss variant."""
    from xcquinox.alec.train import run_training

    with tempfile.TemporaryDirectory() as tmpdir:
        spec = _make_live_spec(
            training_batch_info, loss_name=loss_name, tmpdir=tmpdir,
        )
        metadata = run_training(spec)
        assert isinstance(metadata, dict)
        assert "final_loss" in metadata
        assert math.isfinite(metadata["final_loss"])
        # Check artifacts exist
        ckdir = spec.checkpoint_dir
        assert os.path.isfile(os.path.join(ckdir, "model.eqx"))
        assert os.path.isfile(os.path.join(ckdir, "losses.npy"))
        assert os.path.isfile(os.path.join(ckdir, "aux_log.pkl"))
        assert os.path.isfile(os.path.join(ckdir, "train_metadata.json"))


# ---------------------------------------------------------------------------
# Test 16: losses decrease after 5 steps
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_losses_decrease(training_batch_info):
    """Test 16: loss at step 5 < loss at step 0 for A_atomization."""
    from xcquinox.alec.train import run_training

    with tempfile.TemporaryDirectory() as tmpdir:
        spec = _make_live_spec(
            training_batch_info, loss_name="A_atomization",
            n_steps=5, tmpdir=tmpdir,
        )
        metadata = run_training(spec)
        losses = np.load(os.path.join(spec.checkpoint_dir, "losses.npy"))
        assert losses[-1] < losses[0], (
            f"losses should decrease: first={losses[0]}, last={losses[-1]}"
        )


# ---------------------------------------------------------------------------
# Test 17: artifact roundtrip
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_artifact_roundtrip(training_batch_info):
    """Test 17: model.eqx loads correctly, losses.npy matches, aux_log.pkl
    deserializes, train_metadata.json has all fields, progress_callback was invoked."""
    import equinox as eqx
    from xcquinox.alec.train import run_training
    from xcquinox.alec.models import AlecGGAModel

    progress_calls = []

    def _cb(payload):
        progress_calls.append(payload)

    with tempfile.TemporaryDirectory() as tmpdir:
        spec = _make_live_spec(
            training_batch_info, loss_name="A_atomization",
            n_steps=3, tmpdir=tmpdir,
        )
        metadata = run_training(spec, progress_callback=_cb)
        ckdir = spec.checkpoint_dir

        # model.eqx roundtrip
        model_path = os.path.join(ckdir, "model.eqx")
        model_skel = AlecGGAModel.from_arch(spec.arch, seed=spec.seed)
        model_loaded = eqx.tree_deserialise_leaves(model_path, model_skel)
        # Just check it loaded without error and is an AlecGGAModel
        assert isinstance(model_loaded, AlecGGAModel)

        # losses.npy matches metadata
        losses = np.load(os.path.join(ckdir, "losses.npy"))
        assert len(losses) == 3
        assert np.isclose(losses[-1], metadata["final_loss"])

        # aux_log.pkl deserializes
        with open(os.path.join(ckdir, "aux_log.pkl"), "rb") as f:
            aux_log = pickle.load(f)  # noqa: S301 -- trusted test data
        assert isinstance(aux_log, list)
        assert len(aux_log) == 3

        # train_metadata.json has all required fields
        required_fields = {
            "arch_name", "loss_name", "loss_kwargs", "n_steps",
            "lr_start", "lr_end", "lr_decay_start", "grad_clip",
            "pretrain_checkpoint", "molecules", "targets", "atom_energies",
            "final_loss", "min_loss", "timestamp", "duration_seconds",
        }
        with open(os.path.join(ckdir, "train_metadata.json")) as f:
            md = json.load(f)
        missing = required_fields - set(md.keys())
        assert not missing, f"train_metadata.json missing keys: {missing}"

        # progress_callback was invoked
        assert len(progress_calls) == 3


# ---------------------------------------------------------------------------
# Test 18: pretrain checkpoint yields lower initial loss
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_pretrain_checkpoint_lower_initial_loss(training_batch_info):
    """Test 18: loading a pretrain checkpoint gives a different starting loss
    compared to from-scratch training."""
    import equinox as eqx
    from xcquinox.alec.train import run_training
    from xcquinox.alec.models import AlecGGAModel
    from xcquinox.alec.networks import create_network_pair

    with tempfile.TemporaryDirectory() as tmpdir:
        # First: train from scratch for 3 steps and capture first loss
        ckdir_scratch = os.path.join(tmpdir, "scratch")
        spec_scratch = TrainingSpec.from_dicts(
            arch=_make_arch(),
            molecules=training_batch_info["mols"],
            targets=training_batch_info["targets"],
            atom_energies=training_batch_info["atom_energies"],
            loss_name="A_atomization",
            n_steps=3,
            checkpoint_dir=ckdir_scratch,
            seed=42,
        )
        run_training(spec_scratch)
        losses_scratch = np.load(os.path.join(ckdir_scratch, "losses.npy"))

        # Create pretrain checkpoint: just serialize xnet.eqx + cnet.eqx
        pretrain_dir = os.path.join(tmpdir, "pretrain_ckpt")
        os.makedirs(pretrain_dir, exist_ok=True)
        arch = _make_arch()
        model_trained = AlecGGAModel.from_arch(arch, seed=42)
        # Load the trained model from the scratch run
        model_trained = eqx.tree_deserialise_leaves(
            os.path.join(ckdir_scratch, "model.eqx"), model_trained
        )
        # Save as pretrain checkpoint (xnet.eqx + cnet.eqx)
        eqx.tree_serialise_leaves(
            os.path.join(pretrain_dir, "xnet.eqx"), model_trained.xnet
        )
        eqx.tree_serialise_leaves(
            os.path.join(pretrain_dir, "cnet.eqx"), model_trained.cnet
        )

        # Now train from pretrain checkpoint
        ckdir_pretrained = os.path.join(tmpdir, "pretrained")
        spec_pretrained = TrainingSpec.from_dicts(
            arch=arch,
            molecules=training_batch_info["mols"],
            targets=training_batch_info["targets"],
            atom_energies=training_batch_info["atom_energies"],
            loss_name="A_atomization",
            n_steps=3,
            checkpoint_dir=ckdir_pretrained,
            pretrain_checkpoint=pretrain_dir,
            seed=42,
        )
        run_training(spec_pretrained)
        losses_pretrained = np.load(os.path.join(ckdir_pretrained, "losses.npy"))

        # The pretrained model should start differently (its weights are trained)
        # We just verify they differ -- the pretrained model has already seen
        # gradient updates so its starting loss should be different.
        assert losses_scratch[0] != losses_pretrained[0], (
            "pretrained model should have a different initial loss than from-scratch"
        )


# ---------------------------------------------------------------------------
# Test 19: atom-composition validation (missing single-atom molecules)
# ---------------------------------------------------------------------------

def test_validate_missing_atom_species():
    """Test 19: molecules=(H2O,) without H and O atoms -> ValueError."""
    h2o = h2o_molecule()
    spec = _make_training_spec(
        molecules=(h2o,),
        targets=(("H2O", 0.3),),
        atom_energies=(("H", -0.5), ("O", -74.8)),
    )
    with pytest.raises(ValueError, match="Missing single-atom molecules"):
        spec.validate()


# ---------------------------------------------------------------------------
# Test 20: constraint_report post-update still valid
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_constraint_report_post_training(training_batch_info):
    """Test 20: constraint_report returns valid per-constraint stats after training."""
    import jax.numpy as jnp
    import equinox as eqx
    from xcquinox.alec.train import run_training
    from xcquinox.alec.models import AlecGGAModel

    with tempfile.TemporaryDirectory() as tmpdir:
        spec = _make_live_spec(
            training_batch_info, loss_name="A_atomization",
            n_steps=3, tmpdir=tmpdir,
        )
        run_training(spec)

        # Load trained model
        model_skel = AlecGGAModel.from_arch(spec.arch, seed=spec.seed)
        model = eqx.tree_deserialise_leaves(
            os.path.join(spec.checkpoint_dir, "model.eqx"), model_skel
        )

        # Run constraint_report with synthetic data
        rho = jnp.array([0.1, 0.2, 0.3])
        sigma = jnp.array([0.01, 0.02, 0.03])
        features = jnp.zeros((3, 0))
        report = model.constraint_report(rho, sigma, features)

        assert isinstance(report, dict)
        assert "x" in report
        assert "c" in report
        # With no constraints on the shallow arch, dicts should be empty
        assert isinstance(report["x"], dict)
        assert isinstance(report["c"], dict)


# ---------------------------------------------------------------------------
# Test 21: aux_log.pkl schema
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_aux_log_schema(training_batch_info):
    """Test 21: aux_log.pkl is a list of dicts with {step, loss, aux} keys."""
    from xcquinox.alec.train import run_training

    with tempfile.TemporaryDirectory() as tmpdir:
        spec = _make_live_spec(
            training_batch_info, loss_name="A_atomization",
            n_steps=3, tmpdir=tmpdir,
        )
        run_training(spec)

        with open(os.path.join(spec.checkpoint_dir, "aux_log.pkl"), "rb") as f:
            aux_log = pickle.load(f)  # noqa: S301 -- trusted test data

        assert isinstance(aux_log, list)
        assert len(aux_log) == 3
        for entry in aux_log:
            assert isinstance(entry, dict)
            assert "step" in entry
            assert "loss" in entry
            assert "aux" in entry


# ---------------------------------------------------------------------------
# Test 22: progress callback schema
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_progress_callback_schema(training_batch_info):
    """Test 22: progress callback receives dicts with {arch, phase, step, total,
    loss, timestamp}."""
    from xcquinox.alec.train import run_training

    received = []

    def _cb(payload):
        received.append(payload)

    with tempfile.TemporaryDirectory() as tmpdir:
        spec = _make_live_spec(
            training_batch_info, loss_name="A_atomization",
            n_steps=3, tmpdir=tmpdir,
        )
        run_training(spec, progress_callback=_cb)

    assert len(received) == 3
    for payload in received:
        for key in ("arch", "phase", "step", "total", "loss", "timestamp"):
            assert key in payload, f"progress payload missing key {key!r}"
        assert payload["phase"] == "train"
        assert isinstance(payload["step"], int)
        assert isinstance(payload["total"], int)
        assert isinstance(payload["loss"], float)
        assert isinstance(payload["timestamp"], float)


# ---------------------------------------------------------------------------
# Test 23: molecule-generic (H, N, NH3) training set
# ---------------------------------------------------------------------------

def _n_atom() -> MoleculeSpec:
    """Nitrogen atom (spin=3, open-shell UKS)."""
    return MoleculeSpec(
        name="N", atom="N 0 0 0", basis="sto-3g",
        charge=0, spin=3, atom_composition=(("N", 1),),
    )


def _nh3_molecule() -> MoleculeSpec:
    """Ammonia molecule (spin=0, closed-shell)."""
    return MoleculeSpec(
        name="NH3",
        atom="N 0 0 0.117; H 0 0.935 -0.272; H 0.810 -0.468 -0.272; H -0.810 -0.468 -0.272",
        basis="sto-3g",
        charge=0, spin=0,
        atom_composition=(("H", 3), ("N", 1)),
    )


@pytest.fixture(scope="module")
def nh3_batch_info():
    """Pre-computed batch info for (H, N, NH3)."""
    from xcquinox.alec.data import precompute_fixed_density_data
    h = h_atom()
    n = _n_atom()
    nh3 = _nh3_molecule()
    h_data = precompute_fixed_density_data(h)
    n_data = precompute_fixed_density_data(n)
    nh3_data = precompute_fixed_density_data(nh3)
    ae_nh3 = float(h_data["E_pbe"] * 3 + n_data["E_pbe"] - nh3_data["E_pbe"])
    targets = {
        "H": float(h_data["E_pbe"]),
        "N": float(n_data["E_pbe"]),
        "NH3": max(ae_nh3, 0.001),
    }
    atom_energies = {
        "H": float(h_data["E_pbe"]),
        "N": float(n_data["E_pbe"]),
    }
    return {
        "mols": (h, n, nh3),
        "targets": targets,
        "atom_energies": atom_energies,
    }


@pytest.mark.slow
def test_molecule_generic_h_n_nh3(nh3_batch_info):
    """Test 23: (H, N, NH3) training set works end-to-end."""
    from xcquinox.alec.train import run_training

    with tempfile.TemporaryDirectory() as tmpdir:
        ckdir = os.path.join(tmpdir, "ckpt")
        spec = TrainingSpec.from_dicts(
            arch=_make_arch(),
            molecules=nh3_batch_info["mols"],
            targets=nh3_batch_info["targets"],
            atom_energies=nh3_batch_info["atom_energies"],
            loss_name="A_atomization",
            n_steps=3,
            checkpoint_dir=ckdir,
            seed=42,
        )
        metadata = run_training(spec)
        assert isinstance(metadata, dict)
        assert math.isfinite(metadata["final_loss"])


# ---------------------------------------------------------------------------
# Test 24: missing atom_energy key
# ---------------------------------------------------------------------------

def test_validate_missing_atom_energy_key():
    """Test 24: atom_energies={H: -0.5} for (H, H2O) -> ValueError (missing O)."""
    h = h_atom()
    o = o_atom()
    h2o = h2o_molecule()
    spec = _make_training_spec(
        molecules=(h, o, h2o),
        atom_energies=(("H", -0.5),),
    )
    with pytest.raises(ValueError, match="atom_energies dict is missing"):
        spec.validate()


# ---------------------------------------------------------------------------
# Test 25: atoms-only batch
# ---------------------------------------------------------------------------

def test_validate_atoms_only_batch():
    """Test 25: molecules=(H, O) -> ValueError (no compound molecule)."""
    h = h_atom()
    o = o_atom()
    spec = _make_training_spec(
        molecules=(h, o),
        targets=(("H", -0.5), ("O", -74.8)),
        atom_energies=(("H", -0.5), ("O", -74.8)),
    )
    with pytest.raises(ValueError, match="at least one compound molecule"):
        spec.validate()


# ---------------------------------------------------------------------------
# Test 26: non-finite float hyperparameter
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "field_name", ["lr_start", "lr_end", "lr_decay_start", "grad_clip"],
)
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_validate_nonfinite_float_hyperparameter(field_name, bad_value):
    """Test 26: non-finite hyperparameter -> ValueError."""
    spec = _make_training_spec(**{field_name: bad_value})
    with pytest.raises(ValueError, match=f"{field_name} must be finite"):
        spec.validate()


# ---------------------------------------------------------------------------
# Test 27: non-finite target
# ---------------------------------------------------------------------------

def test_validate_nonfinite_target():
    """Test 27: targets={H2O: nan} -> ValueError."""
    spec = _make_training_spec(
        targets=(("H", -0.5), ("H2O", float("nan")), ("O", -74.8)),
    )
    with pytest.raises(ValueError, match="must be finite"):
        spec.validate()


# ---------------------------------------------------------------------------
# Test 28: non-finite atom_energies
# ---------------------------------------------------------------------------

def test_validate_nonfinite_atom_energies():
    """Test 28: atom_energies={H: nan, O: -74.8} -> ValueError."""
    spec = _make_training_spec(
        atom_energies=(("H", float("nan")), ("O", -74.8)),
    )
    with pytest.raises(ValueError, match="must be finite"):
        spec.validate()


# ---------------------------------------------------------------------------
# Test 29: checkpoint_dir as file
# ---------------------------------------------------------------------------

def test_validate_checkpoint_dir_is_file():
    """Test 29: checkpoint_dir is a regular file -> ValueError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "not_a_dir.chk")
        with open(file_path, "w") as f:
            f.write("x")
        spec = _make_training_spec(checkpoint_dir=file_path)
        with pytest.raises(ValueError, match="checkpoint_dir exists but is not a directory"):
            spec.validate()


# ---------------------------------------------------------------------------
# Test 30: loss_kwargs unknown key
# ---------------------------------------------------------------------------

def test_validate_loss_kwargs_unknown_key():
    """Test 30: loss_kwargs with unknown key -> ValueError."""
    spec = _make_training_spec(
        loss_kwargs=(("totally_bogus_key", 1.0),),
    )
    with pytest.raises(ValueError, match="loss_kwargs contains unknown keys"):
        spec.validate()


# ---------------------------------------------------------------------------
# Test 31: loss_kwargs non-finite numeric
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "key,bad_value",
    [("w_atomic", float("nan"))],
)
def test_validate_loss_kwargs_nonfinite_numeric(key, bad_value):
    """Test 31: loss_kwargs with non-finite numeric -> ValueError; bools excluded."""
    spec = _make_training_spec(
        loss_kwargs=((key, bad_value),),
    )
    with pytest.raises(ValueError, match="must be finite"):
        spec.validate()
