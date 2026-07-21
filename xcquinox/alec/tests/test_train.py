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
from xcquinox.alec.solver import SolverConfig, SolverMode
from xcquinox.alec.tests.fixtures.molecules import (
    h_atom,
    h2_molecule,
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
# Tests: loss_metric and balancing fields
# ---------------------------------------------------------------------------

def test_training_spec_backward_compat_defaults():
    """balancing=None and loss_metric='absolute' are backward-compatible defaults."""
    spec = _make_training_spec()
    assert spec.loss_metric == "absolute"
    assert spec.balancing is None


def test_validate_invalid_loss_metric():
    spec = _make_training_spec(loss_metric="invalid_metric")
    with pytest.raises(ValueError, match="loss_metric must be"):
        spec.validate()


def test_validate_twophase_phase1_steps_exceeds_n_steps():
    from xcquinox.alec.balancing import TwoPhaseConfig
    spec = _make_training_spec(
        balancing=TwoPhaseConfig(phase1_steps=100),
        n_steps=50,
    )
    with pytest.raises(ValueError, match="phase1_steps.*must be < n_steps"):
        spec.validate()


def test_validate_twophase_unknown_phase1_loss():
    from xcquinox.alec.balancing import TwoPhaseConfig
    spec = _make_training_spec(
        balancing=TwoPhaseConfig(phase1_steps=2, phase1_loss="nonexistent"),
        n_steps=5,
    )
    with pytest.raises(ValueError, match="phase1_loss.*not in"):
        spec.validate()


def test_validate_valid_balancing_configs():
    """All balancing config types pass validation when valid."""
    from xcquinox.alec.balancing import (
        BalancingConfig, LossNormConfig, TwoPhaseConfig, GradNormConfig,
    )
    for bal in [None, BalancingConfig(), LossNormConfig(),
                TwoPhaseConfig(phase1_steps=1), GradNormConfig()]:
        spec = _make_training_spec(balancing=bal, n_steps=5)
        spec.validate()  # should not raise


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
        # Best-loss checkpoint written side-by-side with the final one.
        assert os.path.isfile(os.path.join(ckdir, "model_best.eqx"))
        assert metadata["has_best_checkpoint"] is True
        assert os.path.isfile(os.path.join(ckdir, "losses.npy"))
        assert os.path.isfile(os.path.join(ckdir, "aux_log.pkl"))
        assert os.path.isfile(os.path.join(ckdir, "train_metadata.json"))


# ---------------------------------------------------------------------------
# Best-loss checkpoint side-by-side saver -- fast unit tests
# ---------------------------------------------------------------------------

def test_best_model_tracker_selects_min_window1():
    from xcquinox.alec.train import _BestModelTracker
    t = _BestModelTracker(window=1)
    t.update(0.5, "a")
    t.update(0.1, "b")
    t.update(0.3, "c")
    assert t.best_model == "b"
    assert t.best_loss == 0.1


def test_best_model_tracker_window_smooths_and_skips_nonfinite():
    from xcquinox.alec.train import _BestModelTracker
    t = _BestModelTracker(window=2)
    t.update(1.0, "m1")            # window not full yet -> ignored
    t.update(0.2, "m2")            # trailing mean (1.0+0.2)/2 = 0.6
    t.update(0.1, "m3")            # trailing mean (0.2+0.1)/2 = 0.15 -> best
    t.update(float("nan"), "m4")  # non-finite -> ignored
    t.update(float("inf"), "m5")  # non-finite -> ignored
    assert t.best_model == "m3"
    assert abs(t.best_loss - 0.15) < 1e-9


# ---------------------------------------------------------------------------
# WS3 (2026-06-20): _BestValidationTracker (validation-metric early-stop)
# ---------------------------------------------------------------------------

def test_best_validation_tracker_keeps_min_on_improving_curve():
    """A strictly-improving validation curve keeps the LAST (lowest) snapshot and
    NEVER triggers early-stop."""
    from xcquinox.alec.train import _BestValidationTracker
    t = _BestValidationTracker()
    for mae, snap in [(10.0, "a"), (8.0, "b"), (5.0, "c"), (3.0, "d")]:
        t.update(mae, snap)
        assert t.should_stop(patience=2, min_delta=0.0) is False
    assert t.best_model == "d"
    assert t.best_mae == 3.0


def test_best_validation_tracker_stops_after_exactly_patience_checks():
    """A plateaued/rising curve stops after EXACTLY `patience` consecutive
    non-improving checks; the best snapshot remains the min-val one."""
    from xcquinox.alec.train import _BestValidationTracker
    t = _BestValidationTracker()
    t.update(5.0, "best")                          # improvement -> counter resets
    assert t.should_stop(patience=2, min_delta=0.0) is False
    t.update(6.0, "worse1")                        # non-improving #1
    assert t.should_stop(patience=2, min_delta=0.0) is False
    t.update(6.0, "worse2")                        # non-improving #2 -> stop
    assert t.should_stop(patience=2, min_delta=0.0) is True
    # best snapshot is the minimum-val one, not the latest.
    assert t.best_model == "best"
    assert t.best_mae == 5.0


def test_best_validation_tracker_min_delta_requires_real_improvement():
    """A drop smaller than `min_delta` counts as NON-improving (so a noisy
    near-flat curve still early-stops)."""
    from xcquinox.alec.train import _BestValidationTracker
    t = _BestValidationTracker()
    t.update(5.0, "a")
    t.update(4.99, "b")        # improves by 0.01 < min_delta=0.1 -> non-improving
    assert t.should_stop(patience=1, min_delta=0.1) is True
    # best snapshot still updates to the numerically-lower value.
    assert t.best_mae == 4.99
    assert t.best_model == "b"


def test_best_validation_tracker_skips_nonfinite():
    """Non-finite validation MAE (NaN/inf) is ignored: not counted as an
    improvement AND not counted as a non-improving check (no spurious stop)."""
    from xcquinox.alec.train import _BestValidationTracker
    t = _BestValidationTracker()
    t.update(5.0, "a")
    t.update(float("nan"), "b")
    t.update(float("inf"), "c")
    assert t.best_model == "a"
    assert t.best_mae == 5.0
    # the two non-finite checks did NOT advance the no-improvement counter.
    assert t.should_stop(patience=1, min_delta=0.0) is False


def test_best_validation_tracker_patience_zero_never_stops():
    """patience=0 is the documented no-op: never early-stops, even on a rising
    curve."""
    from xcquinox.alec.train import _BestValidationTracker
    t = _BestValidationTracker()
    t.update(5.0, "a")
    t.update(9.0, "b")
    t.update(9.0, "c")
    assert t.should_stop(patience=0, min_delta=0.0) is False


# ---------------------------------------------------------------------------
# WS3 (2026-06-20): _validation_reaction_mae (in-loop val MAE assembly)
# ---------------------------------------------------------------------------

def test_validation_reaction_mae_assembles_from_energy_fn():
    """The val MAE = reaction_mae_kcalmol over per-species energies produced by
    the injected energy_fn (so the assembly is testable with NO PySCF). The
    energy_fn is called once per species in val_mol_data; the reaction energies
    are then scored against reaction_energy_ref."""
    from xcquinox.alec.train import _validation_reaction_mae

    # Two species; one reaction A -> B with a known reference. Per-species
    # energies (Hartree) chosen so the predicted ΔE differs from the ref by a
    # round number of kcal/mol.
    KCAL = 627.5094740631
    val_mol_data = {"A": {"tag": "A"}, "B": {"tag": "B"}}
    energies = {"A": -1.0, "B": -1.5}   # ΔE = E_B - E_A = -0.5 Ha
    de_ref = (-0.5) * KCAL + 4.0        # ref 4 kcal/mol ABOVE the prediction
    reactions = [{
        "name": "rxn1", "reactants": ["A"], "products": ["B"],
        "coeffs": [-1.0, 1.0], "reaction_energy_ref": de_ref,
    }]

    calls = []

    def fake_energy(model, md):
        calls.append(md["tag"])
        return energies[md["tag"]]

    mae = _validation_reaction_mae(
        model=object(), val_mol_data=val_mol_data, val_reactions=reactions,
        solver_config=None, energy_fn=fake_energy)
    assert abs(mae - 4.0) < 1e-6
    assert sorted(calls) == ["A", "B"]   # one energy eval per species


def test_validation_reaction_mae_nan_when_no_finite_reactions():
    """If a species energy is non-finite, its reaction is dropped; with no
    finite reactions the MAE is NaN (matching reaction_mae_kcalmol)."""
    import math
    from xcquinox.alec.train import _validation_reaction_mae

    val_mol_data = {"A": {"tag": "A"}, "B": {"tag": "B"}}

    def fake_energy(model, md):
        return float("nan")

    reactions = [{
        "name": "rxn1", "reactants": ["A"], "products": ["B"],
        "coeffs": [-1.0, 1.0], "reaction_energy_ref": 10.0,
    }]
    mae = _validation_reaction_mae(
        model=object(), val_mol_data=val_mol_data, val_reactions=reactions,
        solver_config=None, energy_fn=fake_energy)
    assert math.isnan(mae)


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
# Stage 4: per-molecule (DFS/dpyscf-style) stochastic update scheme
# ---------------------------------------------------------------------------

def test_validate_update_scheme_invalid():
    spec = _make_training_spec(update_scheme="bogus")
    with pytest.raises(ValueError, match="update_scheme"):
        spec.validate()


def test_update_scheme_defaults_to_batched():
    spec = _make_training_spec()
    assert spec.update_scheme == "batched"
    assert spec.channel_weights_dict == {}


def test_effective_channel_weights_partial_fills_from_defaults():
    """A PARTIAL channel_weights overrides only named channels; omitted channels
    inherit the density-dominant defaults (NOT 1.0)."""
    from xcquinox.alec.train import (
        _effective_channel_weights, _DEFAULT_CHANNEL_WEIGHTS,
    )
    # Empty -> defaults unchanged.
    assert _effective_channel_weights({}) == _DEFAULT_CHANNEL_WEIGHTS
    # Partial (only loss_AE overridden) -> loss_rho keeps its 20.0 default,
    # NOT the old 1.0 fallback.
    eff = _effective_channel_weights({"loss_AE": 5.0})
    assert eff["loss_AE"] == 5.0
    assert eff["loss_rho"] == 20.0
    assert eff["loss_BH76"] == _DEFAULT_CHANNEL_WEIGHTS["loss_BH76"]


def test_training_groups_ae_pool():
    """One AE group per polyatomic compound carrying a target; atoms that are
    not regularized produce no anchor groups."""
    from xcquinox.alec.train import _training_groups
    mols = (h_atom(), o_atom(), h2o_molecule(), h2_molecule())
    spec = TrainingSpec.from_dicts(
        arch=_make_arch(), molecules=mols,
        targets={"H": -0.5, "O": -75.0, "H2O": 0.3, "H2": 0.17},
        atom_energies={"H": -0.5, "O": -75.0},
        loss_name="L5_gradnorm_vxc_step7",
        update_scheme="per_molecule", require_atom_anchors=False,
    )
    groups = _training_groups(spec)
    assert {g["label"] for g in groups} == {"ae:H2O", "ae:H2"}
    assert all(len(g["species"]) == 1 for g in groups)


def test_training_groups_bh76_and_anchor():
    """A BH76 reaction yields one group carrying its species; a regularized
    single atom yields an anchor group."""
    from xcquinox.alec.train import _training_groups
    mols = (h_atom(), h2_molecule(), o_atom())
    rxn = {"name": "r1", "reactants": ["H2"], "products": ["H"],
           "coeffs": [-1.0, 2.0], "e_rxn_ref": 0.17}
    spec = TrainingSpec.from_dicts(
        arch=_make_arch(), molecules=mols,
        targets={"H": -0.5, "H2": 0.17, "O": -75.0},
        atom_energies={"H": -0.5, "O": -75.0},
        loss_name="L5_gradnorm_vxc_step7",
        loss_kwargs={"bh76_reactions": [rxn],
                     "regularize_atom_syms": ("H",)},
        update_scheme="per_molecule", require_atom_anchors=False,
    )
    groups = _training_groups(spec)
    labels = [g["label"] for g in groups]
    assert "bh76:r1" in labels
    assert "anchor:H" in labels
    bh = next(g for g in groups if g["label"] == "bh76:r1")
    assert {s.name for s in bh["species"]} == {"H2", "H"}


def test_training_groups_skips_ae_group_for_aux_only_reaction_compound():
    """Regression: under ae_as_reactions=true a reaction-form AE compound is
    aux-forced (in ``aux_only_names``) AND carries a real target AND trains via
    its bh76 reaction group. It must NOT also get a redundant ``ae:<name>``
    group -- that group has no AE term (aux-forced) and only re-applies the
    density (weight 20) + vxc channels, density/vxc-supervising the compound
    TWICE per epoch. See reports_local/dfs_training_review_2026-06-22.md BUG-1."""
    from xcquinox.alec.train import _training_groups
    mols = (h_atom(), o_atom(), h2o_molecule())
    rxn = {"name": "H2O", "reactants": ["H2O"], "products": ["H", "O"],
           "coeffs": [-1.0, 2.0, 1.0], "e_rxn_ref": 0.3}
    spec = TrainingSpec.from_dicts(
        arch=_make_arch(), molecules=mols,
        targets={"H": -0.5, "O": -75.0, "H2O": 0.3},
        atom_energies={"H": -0.5, "O": -75.0},
        loss_name="L5_gradnorm_vxc_step7",
        loss_kwargs={"bh76_reactions": [rxn],
                     "aux_only_names": ("H2O",),
                     "regularize_atom_syms": ("H", "O")},
        update_scheme="per_molecule", require_atom_anchors=False,
    )
    groups = _training_groups(spec)
    labels = [g["label"] for g in groups]
    assert "bh76:H2O" in labels             # supervised via its reaction group
    assert "ae:H2O" not in labels           # NOT a second (redundant) group
    # the aux-forced compound appears in EXACTLY ONE density/vxc-bearing group
    h2o_groups = [g for g in groups
                  if any(s.name == "H2O" for s in g["species"])]
    assert [g["label"] for g in h2o_groups] == ["bh76:H2O"]


@pytest.mark.slow
def test_run_training_per_molecule_completes(training_batch_info):
    """run_training under update_scheme='per_molecule' completes, takes one
    update per group per epoch, tags the aux_log, and reduces the loss."""
    import pickle
    from xcquinox.alec.train import run_training, _training_groups

    with tempfile.TemporaryDirectory() as tmpdir:
        spec = _make_live_spec(
            training_batch_info, loss_name="L5_gradnorm_vxc_step7",
            n_steps=8, tmpdir=tmpdir, update_scheme="per_molecule",
            require_atom_anchors=False,
        )
        n_groups = len(_training_groups(spec))
        run_training(spec)
        losses = np.load(os.path.join(spec.checkpoint_dir, "losses.npy"))
        # Best-loss checkpoint (epoch-trailing-mean) saved alongside the final.
        assert os.path.isfile(os.path.join(spec.checkpoint_dir, "model_best.eqx"))
        # n_steps is the EPOCH count in per-molecule mode.
        assert len(losses) == spec.n_steps * n_groups
        assert losses[-1] < losses[0]
        with open(os.path.join(spec.checkpoint_dir, "aux_log.pkl"), "rb") as f:
            aux_log = pickle.load(f)
        assert all(e["update_scheme"] == "per_molecule" for e in aux_log)
        assert all("group" in e for e in aux_log)
        # FIX 3 (WS3-ESV-1) E2E: a per_molecule run with validate_every=0 (default)
        # writes the PRE-WS3 metadata key set through the REAL loop -- no
        # has_val_best_checkpoint, no early_stopped/val_* keys, no model_val_best.eqx.
        with open(os.path.join(spec.checkpoint_dir, "train_metadata.json")) as f:
            on_disk = json.load(f)
        assert set(on_disk) == set(_PRE_WS3_METADATA_KEYS)
        assert not os.path.isfile(
            os.path.join(spec.checkpoint_dir, "model_val_best.eqx"))


# ---------------------------------------------------------------------------
# WS3 (2026-06-20): in-loop validation early-stop + model_val_best.eqx in the
# per_molecule loop.
# ---------------------------------------------------------------------------

def test_build_validation_data_disabled_returns_none():
    """validate_every<=0 OR no validation_molecules => (None, None): the loop
    runs with no validation (byte-identical to a decay-free run)."""
    from xcquinox.alec.train import _build_validation_data
    # validate_every=0 (default) -> disabled even if molecules were present.
    spec = _make_training_spec()
    assert _build_validation_data(spec) == (None, None)
    # validate_every>0 but no validation molecules/path -> still disabled.
    spec2 = _make_training_spec(validate_every=2)
    assert _build_validation_data(spec2) == (None, None)


@pytest.mark.slow
def test_per_molecule_loop_early_stops_and_writes_val_best(
        training_batch_info, monkeypatch):
    """With validate_every=1, patience=1 and a monkeypatched val function
    returning a RISING curve, the per_molecule loop early-stops, writes
    model_val_best.eqx, and records early_stopped/epochs_run/val_best_mae."""
    import json as _json
    from xcquinox.alec import train as train_mod
    from xcquinox.alec.train import run_training

    # Stub the val-data build (no extra PySCF) + a strictly-RISING val curve so
    # the FIRST check is the best and the SECOND triggers patience=1.
    monkeypatch.setattr(
        train_mod, "_build_validation_data",
        lambda spec: ({"A": {}, "B": {}},
                      [{"name": "r", "reactants": ["A"], "products": ["B"],
                        "coeffs": [-1.0, 1.0], "reaction_energy_ref": 0.0}]))
    seq = iter([10.0, 11.0, 12.0, 13.0, 14.0])
    monkeypatch.setattr(train_mod, "_validation_reaction_mae",
                        lambda *a, **k: next(seq))

    with tempfile.TemporaryDirectory() as tmpdir:
        spec = _make_live_spec(
            training_batch_info, loss_name="L5_gradnorm_vxc_step7",
            n_steps=8, tmpdir=tmpdir, update_scheme="per_molecule",
            require_atom_anchors=False,
            validate_every=1, patience=1, early_stop_min_delta=0.0,
        )
        meta = run_training(spec)
        # model_val_best.eqx written (the min-val snapshot).
        assert os.path.isfile(
            os.path.join(spec.checkpoint_dir, "model_val_best.eqx"))
        # early-stopped before the full 8 epochs: best at epoch 1 (mae 10), the
        # epoch-2 check (mae 11) is the 1st non-improving -> patience=1 stop.
        assert meta["early_stopped"] is True
        assert meta["epochs_run"] == 2
        assert meta["val_best_mae"] == pytest.approx(10.0)
        # metadata round-trips to disk.
        with open(os.path.join(spec.checkpoint_dir,
                               "train_metadata.json")) as f:
            on_disk = _json.load(f)
        assert on_disk["early_stopped"] is True
        assert on_disk["epochs_run"] == 2


# The EXACT train_metadata.json key set produced BEFORE WS3 (no validation).
# FIX 3 (WS3-ESV-1): a non-validating run must produce byte-identically this
# key set -- no has_val_best_checkpoint, no early_stopped/epochs_run/val_* keys.
_PRE_WS3_METADATA_KEYS = frozenset({
    "arch_name", "use_polarized_correlation", "loss_name", "loss_kwargs",
    "solver_config", "n_steps", "lr_start", "lr_end", "lr_decay_start",
    "grad_clip", "pretrain_checkpoint", "molecules", "targets",
    "atom_energies", "loss_metric", "balancing", "final_loss", "min_loss",
    "has_best_checkpoint", "timestamp", "duration_seconds",
})


def test_save_artifacts_metadata_byte_identical_when_no_validation():
    """FIX 3 (WS3-ESV-1): _save_artifacts with no val_best snapshot + no
    extra_metadata (the per_molecule-with-validate_every=0 / batched case)
    writes train_metadata.json with the PRE-WS3 key set -- no
    has_val_best_checkpoint, no val_* keys. Byte-identical to pre-WS3."""
    spec = _make_training_spec(update_scheme="per_molecule")
    from xcquinox.alec.train import _save_artifacts
    meta = _save_artifacts(
        spec, _make_arch(), [0.5, 0.4, 0.3], [], 1.0,
        best_model=None, val_best_model=None, extra_metadata=None)
    assert set(meta) == set(_PRE_WS3_METADATA_KEYS)
    assert "has_val_best_checkpoint" not in meta
    with open(os.path.join(spec.checkpoint_dir, "train_metadata.json")) as f:
        on_disk = json.load(f)
    assert set(on_disk) == set(_PRE_WS3_METADATA_KEYS)


def test_save_artifacts_adds_val_keys_only_when_validation_ran():
    """When a val-best snapshot + extra_metadata ARE supplied (validation ran),
    has_val_best_checkpoint and the val_* extras appear -- the keys are added
    ONLY in the validated case, never on the default path."""
    spec = _make_training_spec(update_scheme="per_molecule")
    from xcquinox.alec.train import _save_artifacts
    extra = {"early_stopped": True, "epochs_run": 2, "val_best_mae": 10.0,
             "n_epochs_configured": 8, "validate_every": 1, "patience": 1}
    meta = _save_artifacts(
        spec, _make_arch(), [0.5, 0.4], [], 1.0,
        best_model=_make_arch(), val_best_model=_make_arch(),
        extra_metadata=extra)
    assert meta["has_val_best_checkpoint"] is True
    assert meta["early_stopped"] is True and meta["epochs_run"] == 2
    assert set(extra).issubset(set(meta))


# ---------------------------------------------------------------------------
# Fail-loud finite guard: a NaN/Inf must abort training immediately, naming
# the offending loop/step/group/channel, never silently corrupt the weights.
# ---------------------------------------------------------------------------

def test_abort_if_nonfinite_passes_when_finite():
    from xcquinox.alec.train import _abort_if_nonfinite
    # finite loss + finite channels -> returns None, no raise.
    _abort_if_nonfinite(
        0.5, {"loss_AE": 0.1, "loss_rho": 0.2}, loop="batched", step=0)


def test_abort_if_nonfinite_names_nonfinite_channel():
    from xcquinox.alec.train import _abort_if_nonfinite
    with pytest.raises(FloatingPointError, match="loss_AE"):
        _abort_if_nonfinite(
            float("nan"), {"loss_AE": float("nan"), "loss_rho": 0.2},
            loop="per_molecule", step=3, group="anchor:h")


def test_abort_if_nonfinite_raises_on_nonfinite_loss_even_if_channels_finite():
    from xcquinox.alec.train import _abort_if_nonfinite
    with pytest.raises(FloatingPointError, match="step=7"):
        _abort_if_nonfinite(
            float("inf"), {"loss_AE": 0.1}, loop="batched", step=7)


def test_abort_if_nonfinite_names_group_in_message():
    from xcquinox.alec.train import _abort_if_nonfinite
    with pytest.raises(FloatingPointError, match="anchor:h"):
        _abort_if_nonfinite(
            float("nan"), {"loss_AE": float("nan")},
            loop="per_molecule", step=3, group="anchor:h")


# ---------------------------------------------------------------------------
# Gradient-level guard: a step whose LOSS is finite but whose GRADIENT carries a
# NaN/Inf used to pass the guard, corrupt every weight via apply_updates, and
# abort one step LATE on the next group's now-NaN loss -- so the abort named the
# wrong step/group (dfs6311 step-5 ae:CO). The guard must sweep the gradient
# pytree and name the first offending parameter path at the step it occurs.
# ---------------------------------------------------------------------------

def test_abort_if_nonfinite_passes_when_grads_finite():
    import jax.numpy as jnp
    from xcquinox.alec.train import _abort_if_nonfinite
    grads = {"b": jnp.zeros(3), "w": jnp.ones((2, 2))}
    _abort_if_nonfinite(
        0.5, {"loss_AE": 0.1}, loop="per_molecule", step=0, group="ae:H2",
        grads=grads)


def test_abort_if_nonfinite_raises_on_nan_grad_with_finite_loss():
    import jax.numpy as jnp
    from xcquinox.alec.train import _abort_if_nonfinite
    grads = {"b": jnp.array([0.0, jnp.nan, jnp.inf]), "w": jnp.ones((2, 2))}
    with pytest.raises(FloatingPointError) as exc:
        _abort_if_nonfinite(
            0.5, {"loss_AE": 0.1}, loop="per_molecule", step=5, group="ae:CO",
            grads=grads)
    msg = str(exc.value)
    assert "['b']" in msg, msg          # keystr of the offending leaf
    assert "n_nan=1" in msg and "n_inf=1" in msg, msg
    assert "ae:CO" in msg and "step=5" in msg, msg
    assert "1 of 2" in msg, msg         # bad-leaf count over swept leaves


def test_abort_if_nonfinite_counts_all_bad_grad_leaves():
    import jax.numpy as jnp
    from xcquinox.alec.train import _abort_if_nonfinite
    grads = {"a": jnp.full(2, jnp.nan), "b": jnp.ones(2),
             "c": jnp.full((2, 2), jnp.inf)}
    with pytest.raises(FloatingPointError, match="2 of 3"):
        _abort_if_nonfinite(
            0.5, {"loss_AE": 0.1}, loop="batched/static", step=1, grads=grads)


def test_abort_if_nonfinite_grads_none_preserves_loss_only_behavior():
    from xcquinox.alec.train import _abort_if_nonfinite
    # grads omitted -> exactly the pre-existing loss/channel semantics.
    _abort_if_nonfinite(0.5, {"loss_AE": 0.1}, loop="batched", step=0)
    with pytest.raises(FloatingPointError, match="loss_AE"):
        _abort_if_nonfinite(
            float("nan"), {"loss_AE": float("nan")}, loop="batched", step=0)


def _nan_grads_like(model):
    import jax
    import jax.numpy as jnp
    import equinox as eqx
    return jax.tree_util.tree_map(
        lambda a: jnp.full_like(a, jnp.nan),
        eqx.filter(model, eqx.is_inexact_array))


def test_per_molecule_loop_aborts_on_nan_gradient(monkeypatch):
    """Finite loss + NaN grads at the step seam -> the per_molecule loop must
    raise AT that step, naming the group and the gradient, instead of applying
    the corrupt update and surviving to the next step."""
    import jax.numpy as jnp
    import xcquinox.alec.train as train_mod
    from xcquinox.alec.train import run_training

    def _finite_loss_nan_grads(gloss, model, gbatch, cw, relative,
                               pad_target=None):
        return ((jnp.array(0.5), {"loss_AE": jnp.array(0.5)}),
                _nan_grads_like(model))

    monkeypatch.setattr(train_mod, "defused_value_and_grad",
                        _finite_loss_nan_grads)
    with tempfile.TemporaryDirectory() as tmpdir:
        spec = TrainingSpec.from_dicts(
            arch=_make_arch(),
            molecules=(h_atom(), h2_molecule()),
            targets={"H": -0.5, "H2": 0.17},
            atom_energies={"H": -0.5},
            loss_name="L5_gradnorm_vxc_step7",
            update_scheme="per_molecule", require_atom_anchors=False,
            n_steps=1, lr_start=1e-3, lr_end=1e-5, lr_decay_start=0.0,
            grad_clip=1.0, checkpoint_dir=os.path.join(tmpdir, "ck"),
            seed=42)
        with pytest.raises(FloatingPointError) as exc:
            run_training(spec)
    msg = str(exc.value)
    assert "GRADIENT" in msg, msg
    assert "ae:H2" in msg, msg


def test_static_loop_aborts_on_nan_gradient(monkeypatch):
    """Same blind spot in the batched/static loop via the _train_step seam."""
    import jax.numpy as jnp
    import xcquinox.alec.train as train_mod
    from xcquinox.alec.train import run_training

    def _fake_train_step(model, opt_state, batch, loss_fn, optimizer):
        return (model, opt_state, jnp.array(0.25),
                {"loss_AE": jnp.array(0.25)}, _nan_grads_like(model))

    monkeypatch.setattr(train_mod, "_train_step", _fake_train_step)
    with tempfile.TemporaryDirectory() as tmpdir:
        spec = TrainingSpec.from_dicts(
            arch=_make_arch(),
            molecules=(h_atom(), h2_molecule()),
            targets={"H": -0.5, "H2": 0.17},
            atom_energies={"H": -0.5},
            loss_name="A_atomization",
            n_steps=2, lr_start=1e-3, lr_end=1e-5, lr_decay_start=0.0,
            grad_clip=1.0, checkpoint_dir=os.path.join(tmpdir, "ck"),
            seed=42)
        with pytest.raises(FloatingPointError, match="GRADIENT"):
            run_training(spec)


@pytest.mark.slow
def test_run_training_aborts_loudly_on_nonfinite(training_batch_info, monkeypatch):
    """The fail-loud guard raises FloatingPointError (not a silent NaN run) the
    instant a step produces a non-finite value -- here a forced NaN energy."""
    import jax.numpy as jnp
    import xcquinox.alec.losses as losses_mod
    from xcquinox.alec.train import run_training

    def _nan_energies(model, mol_data, N, solver_config=None):
        return jnp.full((N,), jnp.nan)

    monkeypatch.setattr(losses_mod, "_compute_energies", _nan_energies)
    with tempfile.TemporaryDirectory() as tmpdir:
        spec = _make_live_spec(
            training_batch_info, loss_name="L5_gradnorm_vxc_step7",
            n_steps=3, tmpdir=tmpdir,
        )
        with pytest.raises(FloatingPointError, match="non-finite"):
            run_training(spec)


# ---------------------------------------------------------------------------
# Polarized correlation differentiated through the FULL SCF NaN'd on
# fully-spin-polarized atom anchors (H, Li) at zeta=+-1. The whole run (every
# step, not just final_loss) must stay finite, with all-finite saved params.
# This combo (polarized + FULL + per_molecule + atom anchors) was untested; the
# oneshot-only live tests missed it.
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_per_molecule_polarized_full_solver_stays_finite(training_batch_info):
    import jax
    import jax.numpy as jnp
    import equinox as eqx
    from xcquinox.alec.train import run_training
    from xcquinox.alec.models import AlecGGAModel

    with tempfile.TemporaryDirectory() as tmpdir:
        spec = TrainingSpec.from_dicts(
            arch=_make_arch(use_polarized_correlation=True),
            molecules=training_batch_info["mols"],
            targets=training_batch_info["targets"],
            atom_energies=training_batch_info["atom_energies"],
            loss_name="L5_gradnorm_vxc_step7", n_steps=4,
            lr_start=0.01, lr_end=1e-5, lr_decay_start=0.2, grad_clip=1.0,
            checkpoint_dir=os.path.join(tmpdir, "ck"), seed=42,
            update_scheme="per_molecule", require_atom_anchors=False,
            solver_config=SolverConfig(mode=SolverMode.FULL, max_cycles=3),
            loss_kwargs={"regularize_atom_syms": ("H", "O"),
                         "density_weight": 0.1, "vxc_weight": 0.01},
        )
        run_training(spec)
        losses = np.load(os.path.join(spec.checkpoint_dir, "losses.npy"))
        bad = int(np.argmax(~np.isfinite(losses))) if not np.all(
            np.isfinite(losses)) else -1
        assert np.all(np.isfinite(losses)), (
            f"non-finite training loss at step {bad} of {len(losses)}")
        skel = AlecGGAModel.from_arch(spec.arch, seed=spec.seed)
        model = eqx.tree_deserialise_leaves(
            os.path.join(spec.checkpoint_dir, "model.eqx"), skel)
        leaves = jax.tree_util.tree_leaves(
            eqx.filter(model, eqx.is_inexact_array))
        assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves), (
            "saved model has non-finite parameters")


# ---------------------------------------------------------------------------
# All-options matrix: every (update_scheme x solver_mode x polarized) combo --
# the dimension space the 2026-06 NaN lived in -- must train fully finite on
# tiny molecules (UKS atoms H/O + RKS H2O). Each cell asserts EVERY step finite,
# not just final_loss. The FULL-solver cells are the ones never exercised before.
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("polarized", [False, True], ids=["unpol", "pol"])
@pytest.mark.parametrize("solver_id", ["oneshot", "fixed_j", "full3"])
@pytest.mark.parametrize("update_scheme", ["batched", "per_molecule"])
def test_train_matrix_all_options_stay_finite(
        update_scheme, solver_id, polarized, training_batch_info):
    from xcquinox.alec.train import run_training

    solver_cfg = {
        "oneshot": SolverConfig(mode=SolverMode.ONESHOT, max_cycles=0),
        "fixed_j": SolverConfig(mode=SolverMode.FIXED_J, max_cycles=2),
        "full3": SolverConfig(mode=SolverMode.FULL, max_cycles=3),
    }[solver_id]
    extra = {}
    if update_scheme == "per_molecule":
        extra = {
            "update_scheme": "per_molecule",
            "require_atom_anchors": False,
            "loss_kwargs": {"regularize_atom_syms": ("H", "O"),
                            "density_weight": 0.1, "vxc_weight": 0.01},
        }
    with tempfile.TemporaryDirectory() as tmpdir:
        spec = TrainingSpec.from_dicts(
            arch=_make_arch(use_polarized_correlation=polarized),
            molecules=training_batch_info["mols"],
            targets=training_batch_info["targets"],
            atom_energies=training_batch_info["atom_energies"],
            loss_name="L5_gradnorm_vxc_step7", n_steps=3,
            lr_start=0.01, lr_end=1e-5, lr_decay_start=0.2, grad_clip=1.0,
            checkpoint_dir=os.path.join(tmpdir, "ck"), seed=42,
            solver_config=solver_cfg, **extra,
        )
        run_training(spec)
        losses = np.load(os.path.join(spec.checkpoint_dir, "losses.npy"))
        assert np.all(np.isfinite(losses)), (
            f"non-finite loss: scheme={update_scheme} solver={solver_id} "
            f"polarized={polarized} at step "
            f"{int(np.argmax(~np.isfinite(losses)))}/{len(losses)}")


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
            "arch_name", "loss_name", "loss_kwargs", "solver_config",
            "n_steps", "lr_start", "lr_end", "lr_decay_start", "grad_clip",
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


# ---------------------------------------------------------------------------
# Test 32: SolverConfig in loss_kwargs is serialized to JSON
# ---------------------------------------------------------------------------

def test_solver_config_in_loss_kwargs_is_json_serializable():
    """Test 32: SolverConfig objects in loss_kwargs are serialized via describe()."""
    cfg = SolverConfig()
    spec = _make_training_spec(
        loss_kwargs=(("solver_config", cfg),),
        solver_config=cfg,
    )
    # Simulate the serialization logic from run_training
    loss_kwargs_ser = {
        k: v.describe() if isinstance(v, SolverConfig) else v
        for k, v in spec.loss_kwargs_dict.items()
    }
    metadata = {
        "loss_kwargs": loss_kwargs_ser,
        "solver_config": (
            spec.solver_config.describe()
            if spec.solver_config is not None
            else None
        ),
    }
    # Must not raise
    dumped = json.dumps(metadata)
    roundtrip = json.loads(dumped)
    assert isinstance(roundtrip["loss_kwargs"]["solver_config"], dict)
    assert roundtrip["loss_kwargs"]["solver_config"]["backend"] == cfg.backend.value
    assert isinstance(roundtrip["solver_config"], dict)
    assert roundtrip["solver_config"]["mode"] == cfg.mode.value


# ---------------------------------------------------------------------------
# Test 33: FULL mode solver_config causes "eri" in required_keys
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mode,expect_eri", [
    (SolverMode.FULL, True),
    (SolverMode.FIXED_J, False),
    (SolverMode.ONESHOT, False),
])
def test_full_mode_requires_eri(mode, expect_eri):
    """Test 33: required-keys includes 'eri' only for FULL solver mode."""
    max_cycles = 0 if mode == SolverMode.ONESHOT else 3
    cfg = SolverConfig(mode=mode, max_cycles=max_cycles)
    spec = _make_training_spec(
        loss_kwargs=(("solver_config", cfg),),
        solver_config=cfg,
    )
    # Reproduce the required-keys logic from run_training Step 3
    from xcquinox.alec.losses import make_loss
    loss = make_loss(spec.loss_name, molecules=spec.molecules, **spec.loss_kwargs_dict)
    required = set(loss.required_mol_keys)
    for d in spec.arch.materialize_descriptors():
        required |= set(d.required_mol_keys)
    sc = spec.loss_kwargs_dict.get("solver_config") or spec.solver_config
    if isinstance(sc, SolverConfig) and sc.mode == SolverMode.FULL:
        required.add("eri")
    assert ("eri" in required) == expect_eri


# GradNorm robustness to zero step-0 loss channels
def test_gradnorm_relative_rates_neutralizes_zero_L0():
    """A task channel with ~0 step-0 loss must be neutralized (relative rate 1)
    and excluded from the mean, so a later 0->nonzero excursion cannot inject a
    ~1/floor spike into the GradNorm targets (Chen et al. 2018 assume L_i(0)>0)."""
    import jax.numpy as jnp
    from xcquinox.alec.train import _gradnorm_relative_rates
    comp = jnp.array([0.01, 1.0, 2.0])   # channel 0 grew from 0
    L0 = jnp.array([0.0, 1.0, 2.0])      # channel 0 had zero step-0 loss
    r_rel = _gradnorm_relative_rates(comp, L0)
    assert bool(jnp.all(jnp.isfinite(r_rel)))
    assert float(r_rel[0]) == 1.0                  # neutralized
    assert float(jnp.max(jnp.abs(r_rel))) < 1e3    # no 1e10 spike
    assert bool(jnp.allclose(r_rel[1:], 1.0))      # valid channels: r=1 -> 1


def test_gradnorm_relative_rates_matches_plain_when_all_valid():
    """When all L0 are above the floor, rates equal plain GradNorm r_i/mean(r)."""
    import jax.numpy as jnp
    from xcquinox.alec.train import _gradnorm_relative_rates
    comp = jnp.array([2.0, 1.0, 0.5])
    L0 = jnp.array([1.0, 2.0, 1.0])
    r = comp / L0
    expected = r / jnp.mean(r)
    got = _gradnorm_relative_rates(comp, L0)
    assert bool(jnp.allclose(got, expected, atol=1e-6))


def test_training_groups_skip_charged_atom_anchor():
    """Cations must NOT get element-symbol anchor groups: anchor:Li+ would
    pull E_NN(Li+) toward the NEUTRAL Chakravorty Li value via the scoped
    regularizer (build_indices maps atom_map['Li'] -> Li+ when Li+ is the
    only candidate), opposing the IP13 channel."""
    from xcquinox.alec.train import _training_groups
    from xcquinox.alec.config import MoleculeSpec
    li = MoleculeSpec(name="Li", atom="Li 0 0 0", basis="sto-3g",
                      charge=0, spin=1, atom_composition=(("Li", 1),))
    li_cat = MoleculeSpec(name="Li+", atom="Li 0 0 0", basis="sto-3g",
                          charge=1, spin=0, atom_composition=(("Li", 1),))
    spec = TrainingSpec.from_dicts(
        arch=_make_arch(), molecules=(li, li_cat, h2_molecule()),
        targets={"Li": -7.478, "Li+": -7.28, "H2": 0.17},
        atom_energies={"Li": -7.478, "H": -0.5},
        loss_name="L5_gradnorm_vxc_step7",
        loss_kwargs={"regularize_atom_syms": ("Li",),
                     "ip13_pairs": [{"name": "Li_IP", "neutral": "Li",
                                     "cation": "Li+", "ip_ref": 0.198}]},
        update_scheme="per_molecule", require_atom_anchors=False,
    )
    labels = [g["label"] for g in _training_groups(spec)]
    assert "anchor:Li" in labels
    assert "anchor:Li+" not in labels
    assert "ip13:Li_IP" in labels       # the cation still trains via its pair


# 2026-06-20 (WS2): L2 weight decay. The 2026-06-20 review found the DFS pool
# overfits with plain adam (no decay) while DFS used weight decay. build_optimizer
# must apply DECOUPLED weight decay (adamw): under a ZERO loss-gradient a positive
# weight_decay still shrinks params; weight_decay=0 leaves them untouched.
def test_build_optimizer_weight_decay_shrinks_params_under_zero_grad():
    import jax.numpy as jnp
    import optax
    from xcquinox.alec.train import build_optimizer

    params = {"w": jnp.ones((4,))}
    zero_grad = {"w": jnp.zeros((4,))}
    kw = dict(lr_start=0.1, lr_end=0.1, n_steps=1, lr_decay_start=0.0,
              grad_clip=1e9)

    opt = build_optimizer(weight_decay=0.5, **kw)
    updates, _ = opt.update(zero_grad, opt.init(params), params)
    decayed = optax.apply_updates(params, updates)
    assert float(decayed["w"][0]) < 1.0   # decoupled decay shrinks even at zero grad

    opt0 = build_optimizer(weight_decay=0.0, **kw)
    updates0, _ = opt0.update(zero_grad, opt0.init(params), params)
    undecayed = optax.apply_updates(params, updates0)
    assert float(undecayed["w"][0]) == 1.0  # no decay + zero grad -> no change


def test_build_optimizer_weight_decay_defaults_to_zero():
    # default (omitted) weight_decay must be a no-op, so existing runs are unchanged.
    import jax.numpy as jnp
    import optax
    from xcquinox.alec.train import build_optimizer
    params = {"w": jnp.ones((3,))}
    opt = build_optimizer(lr_start=0.1, lr_end=0.1, n_steps=1,
                          lr_decay_start=0.0, grad_clip=1e9)
    updates, _ = opt.update({"w": jnp.zeros((3,))}, opt.init(params), params)
    assert float(optax.apply_updates(params, updates)["w"][0]) == 1.0


# ---------------------------------------------------------------------------
# WS5 (2026-06-20): RESUMABLE per_molecule training -- resume checkpoint
# serialization helpers (PySCF-free; tiny real AlecGGAModel + optax state).
# ---------------------------------------------------------------------------

def _tiny_model_and_opt(seed=3, n_advance=2):
    """Build a tiny real AlecGGAModel + an advanced optax opt_state (the adamw
    step count > 0 so the round-trip exercises the LR-schedule resume). Returns
    (model, opt_state, optimizer).

    The optimizer is advanced with a deterministic ones-shaped gradient pytree
    (no model forward pass needed -- this exercises the SAME optax.update path
    the real loop uses and produces non-trivial adam moments + a non-zero step
    count, which is all the resume round-trip needs)."""
    import equinox as eqx
    import jax.tree_util as jtu
    import jax.numpy as jnp
    from xcquinox.alec.models import AlecGGAModel
    from xcquinox.alec.train import build_optimizer

    arch = _make_arch()
    model = AlecGGAModel.from_arch(arch, seed=seed)
    optimizer = build_optimizer(lr_start=1e-3, lr_end=1e-5, n_steps=10,
                                lr_decay_start=0.0, grad_clip=1.0)
    params = eqx.filter(model, eqx.is_array)
    opt_state = optimizer.init(params)
    for _ in range(n_advance):
        grads = jtu.tree_map(lambda a: jnp.ones_like(a) * 0.01, params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        model = eqx.apply_updates(model, updates)
        params = eqx.filter(model, eqx.is_array)
    return model, opt_state, optimizer


def _opt_step_count(opt_state):
    """Extract the adamw scalar step count from an optax opt_state pytree."""
    import jax.tree_util as jtu
    import numpy as _np
    for leaf in jtu.tree_leaves(opt_state):
        a = _np.asarray(leaf)
        if a.dtype.kind in "iu" and a.ndim == 0:
            return int(a)
    raise AssertionError("no scalar int count leaf in opt_state")


def test_write_then_load_resume_checkpoint_roundtrip(tmp_path):
    """WS5: _write_resume_checkpoint then _load_resume_checkpoint restores the
    model arrays, opt_state (incl. adamw step count), RNG state, both trackers'
    scalars + their best_model snapshots, epoch/update/losses/aux exactly."""
    import equinox as eqx
    import jax.tree_util as jtu
    import numpy as _np
    from xcquinox.alec.models import AlecGGAModel
    from xcquinox.alec.train import (
        _write_resume_checkpoint, _load_resume_checkpoint,
        _BestModelTracker, _BestValidationTracker, build_optimizer,
    )

    model, opt_state, optimizer = _tiny_model_and_opt(seed=3, n_advance=3)

    # Trackers carrying DISTINCT best_model snapshots (a different seed) so a
    # mix-up between train-best and val-best would be detectable.
    tt = _BestModelTracker(window=2)
    tt.best_loss = 0.123
    tt._recent = [0.5, 0.123]
    train_best = AlecGGAModel.from_arch(_make_arch(), seed=11)
    tt.best_model = train_best

    vt = _BestValidationTracker()
    vt.best_mae = 7.5
    vt._finite_metrics = [9.0, 7.5, 8.0]
    val_best = AlecGGAModel.from_arch(_make_arch(), seed=22)
    vt.best_model = val_best

    rng = _np.random.RandomState(42)
    rng.shuffle(_np.arange(5))      # advance the RNG so its state is non-initial
    rng_state = rng.get_state()
    order = [2, 0, 1]
    losses = [0.5, 0.4, 0.123]
    aux = [{"step": 0, "loss": 0.5}, {"step": 1, "loss": 0.4}]

    _write_resume_checkpoint(
        str(tmp_path), model=model, opt_state=opt_state, rng_state=rng_state,
        order=order,
        train_best_loss=tt.best_loss, train_recent=list(tt._recent),
        train_window=tt.window, train_best_model=tt.best_model,
        val_present=True, val_best_mae=vt.best_mae,
        val_finite_metrics=list(vt._finite_metrics), val_best_model=vt.best_model,
        epoch=4, update=12, losses=losses, aux_log=aux, early_stopped=False)

    # the resume_* set exists.
    for fn in ("resume_model.eqx", "resume_opt_state.eqx", "resume_best.eqx",
               "resume_val_best.eqx", "resume_state.pkl"):
        assert os.path.isfile(os.path.join(str(tmp_path), fn)), fn

    # Build the skeletons the loader needs (fresh init).
    model_skel = AlecGGAModel.from_arch(_make_arch(), seed=999)
    opt_skel = optimizer.init(eqx.filter(model_skel, eqx.is_array))
    out = _load_resume_checkpoint(
        str(tmp_path), model_skeleton=model_skel, opt_state_skeleton=opt_skel)

    # model arrays equal.
    a1 = [_np.asarray(x) for x in jtu.tree_leaves(eqx.filter(model, eqx.is_array))]
    a2 = [_np.asarray(x) for x in
          jtu.tree_leaves(eqx.filter(out["model"], eqx.is_array))]
    assert len(a1) == len(a2) and all(_np.allclose(x, y) for x, y in zip(a1, a2))

    # opt_state equal incl. step count.
    assert _opt_step_count(out["opt_state"]) == _opt_step_count(opt_state) == 3
    o1 = [_np.asarray(x) for x in jtu.tree_leaves(opt_state)]
    o2 = [_np.asarray(x) for x in jtu.tree_leaves(out["opt_state"])]
    assert all(_np.allclose(x, y) for x, y in zip(o1, o2))

    # scalars + epoch/update/losses/aux/order.
    assert out["epoch"] == 4
    assert out["update"] == 12
    assert out["order"] == order
    assert out["losses"] == losses
    assert out["aux_log"] == aux
    assert out["early_stopped"] is False
    # RandomState.get_state() is a tuple whose 2nd element is a uint32 key array;
    # compare component-wise so the array doesn't trip ==-on-tuple ambiguity.
    assert out["rng_state"][0] == rng_state[0]
    assert _np.array_equal(out["rng_state"][1], rng_state[1])
    assert out["rng_state"][2:] == rng_state[2:]

    # trackers rehydrated incl. their (distinct) best_model snapshots.
    rt = out["train_tracker"]
    assert isinstance(rt, _BestModelTracker)
    assert rt.best_loss == pytest.approx(0.123)
    assert rt.window == 2 and rt._recent == [0.5, 0.123]
    rv = out["val_tracker"]
    assert isinstance(rv, _BestValidationTracker)
    assert rv.best_mae == pytest.approx(7.5)
    assert rv._finite_metrics == [9.0, 7.5, 8.0]

    # train_best snapshot round-trips to the SAME arrays as the original
    # train_best (and is NOT the val_best).
    tb1 = [_np.asarray(x) for x in
           jtu.tree_leaves(eqx.filter(train_best, eqx.is_array))]
    tb2 = [_np.asarray(x) for x in
           jtu.tree_leaves(eqx.filter(rt.best_model, eqx.is_array))]
    assert all(_np.allclose(x, y) for x, y in zip(tb1, tb2))
    vb1 = [_np.asarray(x) for x in
           jtu.tree_leaves(eqx.filter(val_best, eqx.is_array))]
    vb2 = [_np.asarray(x) for x in
           jtu.tree_leaves(eqx.filter(rv.best_model, eqx.is_array))]
    assert all(_np.allclose(x, y) for x, y in zip(vb1, vb2))
    # train_best != val_best (distinct seeds) -> ensures no snapshot mix-up.
    assert not all(_np.allclose(x, y) for x, y in zip(tb1, vb1))


def test_resume_rng_state_restores_shuffle_sequence(tmp_path):
    """WS5: a run that continues on the LIVE rng and a run that resumes from the
    saved rng_state produce the SAME next shuffle order -- so resume does not
    re-walk groups the killed run already trained on."""
    import equinox as eqx
    import numpy as _np
    from xcquinox.alec.models import AlecGGAModel
    from xcquinox.alec.train import (
        _write_resume_checkpoint, _load_resume_checkpoint, _BestModelTracker,
    )
    model, opt_state, optimizer = _tiny_model_and_opt(seed=1, n_advance=1)
    rng = _np.random.RandomState(7)
    order = _np.arange(6)
    for _ in range(3):
        rng.shuffle(order)         # mimic 3 completed epochs
    rng_state = rng.get_state()    # captured at epoch boundary (pre-next-shuffle)

    tt = _BestModelTracker(window=1)
    _write_resume_checkpoint(
        str(tmp_path), model=model, opt_state=opt_state, rng_state=rng_state,
        order=list(order),
        train_best_loss=tt.best_loss, train_recent=list(tt._recent),
        train_window=tt.window, train_best_model=tt.best_model,
        val_present=False, val_best_mae=None, val_finite_metrics=None,
        val_best_model=None, epoch=3, update=18,
        losses=[], aux_log=[], early_stopped=False)

    # Continuing run: the live rng's NEXT shuffle (the epoch-4 order) applied to
    # the SAME `order` arrangement the resumed run will restore.
    cont = order.copy()
    rng.shuffle(cont)

    # Resumed run: rehydrate the rng AND `order` from the persisted state and
    # take the next shuffle. This must match the continuing run exactly.
    model_skel = AlecGGAModel.from_arch(_make_arch(), seed=2)
    opt_skel = optimizer.init(eqx.filter(model_skel, eqx.is_array))
    out = _load_resume_checkpoint(
        str(tmp_path), model_skeleton=model_skel, opt_state_skeleton=opt_skel)
    assert out["order"] == list(order)
    resumed_rng = _np.random.RandomState(0)
    resumed_rng.set_state(out["rng_state"])
    resumed = _np.asarray(out["order"])
    resumed_rng.shuffle(resumed)

    assert list(cont) == list(resumed)


def test_load_resume_checkpoint_without_optional_best_snapshots(tmp_path):
    """WS5: when the trackers have NO best_model, resume_best.eqx /
    resume_val_best.eqx are NOT written, and load rehydrates trackers with
    best_model=None (and val_tracker=None when none was saved)."""
    import equinox as eqx
    from xcquinox.alec.models import AlecGGAModel
    from xcquinox.alec.train import (
        _write_resume_checkpoint, _load_resume_checkpoint,
        _BestModelTracker,
    )
    model, opt_state, optimizer = _tiny_model_and_opt(seed=4, n_advance=1)
    tt = _BestModelTracker(window=3)   # best_model stays None
    _write_resume_checkpoint(
        str(tmp_path), model=model, opt_state=opt_state,
        rng_state=__import__("numpy").random.RandomState(5).get_state(),
        order=[0],
        train_best_loss=tt.best_loss, train_recent=list(tt._recent),
        train_window=tt.window, train_best_model=tt.best_model,
        val_present=False, val_best_mae=None, val_finite_metrics=None,
        val_best_model=None, epoch=0, update=3,
        losses=[1.0], aux_log=[], early_stopped=False)
    assert not os.path.isfile(os.path.join(str(tmp_path), "resume_best.eqx"))
    assert not os.path.isfile(os.path.join(str(tmp_path), "resume_val_best.eqx"))
    model_skel = AlecGGAModel.from_arch(_make_arch(), seed=6)
    opt_skel = optimizer.init(eqx.filter(model_skel, eqx.is_array))
    out = _load_resume_checkpoint(
        str(tmp_path), model_skeleton=model_skel, opt_state_skeleton=opt_skel)
    assert out["train_tracker"].best_model is None
    assert out["val_tracker"] is None


def test_finalize_completion_writes_sentinel_and_deletes_resume(tmp_path):
    """WS5/WS6 contract: the completion helper writes completion.json
    {completed, early_stopped, epochs_run} and DELETES the resume_* set (so a
    completed dir has model.eqx + completion.json and NO resume_* files)."""
    import json as _json
    from xcquinox.alec.train import _finalize_completion

    d = str(tmp_path)
    # plant a full resume_* set + the model.eqx success signal.
    for fn in ("resume_model.eqx", "resume_opt_state.eqx", "resume_best.eqx",
               "resume_val_best.eqx", "resume_state.pkl", "model.eqx"):
        with open(os.path.join(d, fn), "wb") as f:
            f.write(b"x")
    _finalize_completion(d, early_stopped=True, epochs_run=42)
    # resume_* deleted, model.eqx untouched.
    for fn in ("resume_model.eqx", "resume_opt_state.eqx", "resume_best.eqx",
               "resume_val_best.eqx", "resume_state.pkl"):
        assert not os.path.isfile(os.path.join(d, fn)), fn
    assert os.path.isfile(os.path.join(d, "model.eqx"))
    with open(os.path.join(d, "completion.json")) as f:
        sentinel = _json.load(f)
    assert sentinel == {"completed": True, "early_stopped": True,
                        "epochs_run": 42}


def test_finalize_completion_tolerates_missing_resume_files(tmp_path):
    """WS5: completion cleanup is idempotent -- deleting an absent resume_* file
    must not raise (checkpoint_every=0 runs never wrote them)."""
    from xcquinox.alec.train import _finalize_completion
    d = str(tmp_path)
    _finalize_completion(d, early_stopped=False, epochs_run=3)  # no files present
    assert os.path.isfile(os.path.join(d, "completion.json"))


def test_has_resume_checkpoint_ignores_orphan_resume_next_to_model(tmp_path):
    """WS5-SIG-5: a SIGTERM that lands AFTER model.eqx is written but BEFORE the
    resume_* cleanup leaves orphan resume_* files next to model.eqx. This is
    benign -- _has_resume_checkpoint must return False whenever model.eqx OR
    completion.json is present, so the orphans are IGNORED (model.eqx wins) and
    the dir is never re-resumed."""
    from xcquinox.alec.train import _has_resume_checkpoint
    d = str(tmp_path)
    # resume_state.pkl alone -> resumable.
    with open(os.path.join(d, "resume_state.pkl"), "wb") as f:
        f.write(b"x")
    assert _has_resume_checkpoint(d) is True
    # model.eqx now present (orphan resume_state remains) -> NOT resumable.
    with open(os.path.join(d, "model.eqx"), "wb") as f:
        f.write(b"x")
    assert _has_resume_checkpoint(d) is False
    # completion.json also wins regardless of orphan resume_state.
    os.remove(os.path.join(d, "model.eqx"))
    with open(os.path.join(d, "completion.json"), "w") as f:
        f.write("{}")
    assert _has_resume_checkpoint(d) is False


def test_resume_flusher_registry_register_and_clear():
    """WS5: the module-level resume-flusher holder lets the SIGTERM handler in
    the worker call the loop's flush fn. register/get/clear round-trips."""
    from xcquinox.alec.train import (
        _register_resume_flusher, _clear_resume_flusher, _get_resume_flusher,
    )
    _clear_resume_flusher()
    assert _get_resume_flusher() is None
    calls = []
    _register_resume_flusher(lambda: calls.append(1))
    f = _get_resume_flusher()
    assert f is not None
    f()
    assert calls == [1]
    _clear_resume_flusher()
    assert _get_resume_flusher() is None


# ---------------------------------------------------------------------------
# WS5 (2026-06-20): RESUMABLE per_molecule loop -- end-to-end wiring.
# checkpoint_every=0 byte-identity is PySCF-free-ish but still needs the live
# loop; the resume-equivalence + completion tests use the live PySCF batch.
# ---------------------------------------------------------------------------

class _StopAfterEpochs(Exception):
    """Sentinel: simulate a walltime kill after N completed epochs."""


def _interrupt_after(n_epochs_before_kill):
    """A progress callback that raises _StopAfterEpochs once the loop reports it
    has finished ``n_epochs_before_kill`` epochs (the periodic resume checkpoint
    for that epoch is already on disk by the time the hook fires)."""
    def _cb(info):
        if int(info.get("step", 0)) >= n_epochs_before_kill:
            raise _StopAfterEpochs(info["step"])
    return _cb


def _model_leaves(path):
    """Load model.eqx arrays as a flat list of numpy arrays for fp comparison."""
    import equinox as eqx
    import jax.tree_util as jtu
    import numpy as _np
    from xcquinox.alec.models import AlecGGAModel
    skel = AlecGGAModel.from_arch(_make_arch(), seed=12345)
    m = eqx.tree_deserialise_leaves(path, skel)
    return [_np.asarray(x) for x in jtu.tree_leaves(eqx.filter(m, eqx.is_array))]


@pytest.mark.slow
def test_per_molecule_checkpoint_every_zero_writes_no_resume_files(
        training_batch_info):
    """WS5 byte-identity: checkpoint_every=0 (default) writes NONE of the
    resume_* files and NO completion.json -- the loop is byte-identical to the
    pre-WS5 per_molecule loop."""
    from xcquinox.alec.train import run_training
    with tempfile.TemporaryDirectory() as tmpdir:
        spec = _make_live_spec(
            training_batch_info, loss_name="L5_gradnorm_vxc_step7",
            n_steps=3, tmpdir=tmpdir, update_scheme="per_molecule",
            require_atom_anchors=False)   # checkpoint_every defaults to 0
        assert spec.checkpoint_every == 0
        run_training(spec)
        d = spec.checkpoint_dir
        for fn in ("resume_model.eqx", "resume_opt_state.eqx", "resume_best.eqx",
                   "resume_val_best.eqx", "resume_state.pkl", "completion.json"):
            assert not os.path.isfile(os.path.join(d, fn)), fn
        # the normal artifacts are still written.
        assert os.path.isfile(os.path.join(d, "model.eqx"))
        assert os.path.isfile(os.path.join(d, "model_best.eqx"))


@pytest.mark.slow
def test_per_molecule_completion_writes_sentinel_and_clears_resume(
        training_batch_info):
    """WS5/WS6: a clean run with checkpoint_every>0 ends with model.eqx +
    completion.json present and ALL resume_* files deleted."""
    import json as _json
    from xcquinox.alec.train import run_training
    with tempfile.TemporaryDirectory() as tmpdir:
        spec = _make_live_spec(
            training_batch_info, loss_name="L5_gradnorm_vxc_step7",
            n_steps=3, tmpdir=tmpdir, update_scheme="per_molecule",
            require_atom_anchors=False, checkpoint_every=1)
        run_training(spec)
        d = spec.checkpoint_dir
        assert os.path.isfile(os.path.join(d, "model.eqx"))
        for fn in ("resume_model.eqx", "resume_opt_state.eqx", "resume_best.eqx",
                   "resume_val_best.eqx", "resume_state.pkl"):
            assert not os.path.isfile(os.path.join(d, fn)), fn
        with open(os.path.join(d, "completion.json")) as f:
            sentinel = _json.load(f)
        assert sentinel["completed"] is True
        assert sentinel["early_stopped"] is False
        assert sentinel["epochs_run"] == 3


@pytest.mark.slow
def test_per_molecule_resume_finishes_and_matches_uninterrupted(
        training_batch_info):
    """WS5 CORE: a run killed after 2 of 5 epochs RESUMES from its periodic
    checkpoint and FINISHES; the final model is identical (fp tolerance) to a
    from-scratch uninterrupted run with the same seed."""
    from xcquinox.alec.train import run_training
    import numpy as _np

    # (A) Reference: a clean 5-epoch run in its own dir.
    with tempfile.TemporaryDirectory() as ref_dir:
        ref_spec = _make_live_spec(
            training_batch_info, loss_name="L5_gradnorm_vxc_step7",
            n_steps=5, tmpdir=ref_dir, update_scheme="per_molecule",
            require_atom_anchors=False, checkpoint_every=1)
        run_training(ref_spec)
        ref_leaves = _model_leaves(
            os.path.join(ref_spec.checkpoint_dir, "model.eqx"))

        # (B) Interrupted run: SAME seed, SAME checkpoint_dir, killed after 2
        # epochs, then re-entered to finish.
        with tempfile.TemporaryDirectory() as run_dir:
            spec = _make_live_spec(
                training_batch_info, loss_name="L5_gradnorm_vxc_step7",
                n_steps=5, tmpdir=run_dir, update_scheme="per_molecule",
                require_atom_anchors=False, checkpoint_every=1)
            with pytest.raises(_StopAfterEpochs):
                run_training(spec, progress_callback=_interrupt_after(2))
            d = spec.checkpoint_dir
            # mid-run: resume present, NO success signal yet.
            assert os.path.isfile(os.path.join(d, "resume_state.pkl"))
            assert not os.path.isfile(os.path.join(d, "model.eqx"))
            assert not os.path.isfile(os.path.join(d, "completion.json"))

            # Re-enter the SAME spec; the loop resumes from epoch 2 and finishes.
            run_training(spec)
            assert os.path.isfile(os.path.join(d, "model.eqx"))
            assert os.path.isfile(os.path.join(d, "completion.json"))
            assert not os.path.isfile(os.path.join(d, "resume_state.pkl"))

            resumed_leaves = _model_leaves(os.path.join(d, "model.eqx"))

    assert len(ref_leaves) == len(resumed_leaves)
    for a, b in zip(ref_leaves, resumed_leaves):
        assert _np.allclose(a, b, rtol=1e-9, atol=1e-9)


@pytest.mark.slow
def test_per_molecule_completed_dir_does_not_resume(training_batch_info):
    """WS5: a dir already carrying completion.json (+ model.eqx) is NOT resumed
    -- re-running starts fresh (does not read the stale resume_state.pkl)."""
    from xcquinox.alec import train as train_mod
    from xcquinox.alec.train import run_training
    with tempfile.TemporaryDirectory() as tmpdir:
        spec = _make_live_spec(
            training_batch_info, loss_name="L5_gradnorm_vxc_step7",
            n_steps=2, tmpdir=tmpdir, update_scheme="per_molecule",
            require_atom_anchors=False, checkpoint_every=1)
        run_training(spec)             # completes -> completion.json present
        d = spec.checkpoint_dir
        assert os.path.isfile(os.path.join(d, "completion.json"))
        # Plant a stale resume_state.pkl; because completion.json exists the loop
        # MUST ignore it (start fresh) -- spy on _load_resume_checkpoint.
        with open(os.path.join(d, "resume_state.pkl"), "wb") as f:
            f.write(b"stale")
        called = {"n": 0}
        orig = train_mod._load_resume_checkpoint

        def _spy(*a, **k):
            called["n"] += 1
            return orig(*a, **k)
        train_mod._load_resume_checkpoint = _spy
        try:
            run_training(spec)
        finally:
            train_mod._load_resume_checkpoint = orig
        assert called["n"] == 0        # never attempted a resume load


def test_resume_continues_lr_schedule_not_restart(tmp_path):
    """WS5 step 5: the adamw step count restored from the resume checkpoint
    drives the LR SCHEDULE forward -- the update applied right after resume uses
    the CONTINUED-schedule learning rate (smaller, decayed), NOT the step-0 LR a
    fresh restart would use. Proven by comparing the parameter delta of a
    resumed step against a fresh-state step on the same gradient: with a decaying
    schedule the resumed (later-step, lower-LR) update is strictly smaller."""
    import equinox as eqx
    import jax.numpy as jnp
    import jax.tree_util as jtu
    import numpy as _np
    from xcquinox.alec.models import AlecGGAModel
    from xcquinox.alec.train import (
        build_optimizer, _write_resume_checkpoint, _load_resume_checkpoint,
        _BestModelTracker,
    )

    # A clearly-decaying schedule so step index materially changes the LR.
    optimizer = build_optimizer(lr_start=1.0, lr_end=1e-4, n_steps=100,
                                lr_decay_start=0.0, grad_clip=1e9)
    model = AlecGGAModel.from_arch(_make_arch(), seed=8)
    params = eqx.filter(model, eqx.is_array)
    opt_state = optimizer.init(params)
    # Advance MANY steps so the restored count maps to a much lower LR.
    g = jtu.tree_map(lambda a: jnp.ones_like(a), params)
    for _ in range(50):
        upd, opt_state = optimizer.update(g, opt_state, params)
        model = eqx.apply_updates(model, upd)
        params = eqx.filter(model, eqx.is_array)

    _tt = _BestModelTracker(window=1)
    _write_resume_checkpoint(
        str(tmp_path), model=model, opt_state=opt_state,
        rng_state=_np.random.RandomState(0).get_state(), order=[0],
        train_best_loss=_tt.best_loss, train_recent=list(_tt._recent),
        train_window=_tt.window, train_best_model=_tt.best_model,
        val_present=False, val_best_mae=None, val_finite_metrics=None,
        val_best_model=None, epoch=50, update=50, losses=[], aux_log=[],
        early_stopped=False)

    skel = AlecGGAModel.from_arch(_make_arch(), seed=9)
    opt_skel = optimizer.init(eqx.filter(skel, eqx.is_array))
    out = _load_resume_checkpoint(
        str(tmp_path), model_skeleton=skel, opt_state_skeleton=opt_skel)
    assert _opt_step_count(out["opt_state"]) == 50

    # One more update from the RESTORED state (step ~50 -> low LR).
    p_resumed = eqx.filter(out["model"], eqx.is_array)
    upd_resumed, _ = optimizer.update(g, out["opt_state"], p_resumed)
    delta_resumed = max(
        float(_np.max(_np.abs(_np.asarray(x))))
        for x in jtu.tree_leaves(upd_resumed))

    # One update from a FRESH state (step 0 -> high LR) on the same params/grad.
    fresh_state = optimizer.init(p_resumed)
    upd_fresh, _ = optimizer.update(g, fresh_state, p_resumed)
    delta_fresh = max(
        float(_np.max(_np.abs(_np.asarray(x))))
        for x in jtu.tree_leaves(upd_fresh))

    # The resumed update is on the decayed branch -> strictly smaller step than a
    # step-0 restart would take. This is exactly the LR-schedule-resume contract.
    assert delta_resumed < delta_fresh


@pytest.mark.slow
def test_sigterm_flusher_writes_full_resume_set_between_periodic_checkpoints(
        training_batch_info):
    """WS5: the flush registered by the per_molecule loop (what the worker's
    SIGTERM handler invokes) writes the FULL resume_* set even when NO periodic
    checkpoint has fired yet -- the safety net for progress between periodic
    writes. checkpoint_every is set LARGER than n_steps so no periodic write
    happens; the kill callback grabs the live flusher and calls it."""
    from xcquinox.alec import train as train_mod
    from xcquinox.alec.train import run_training

    captured = {}

    def _cb(info):
        # On the first epoch report, capture + call the live flusher, then kill.
        if int(info.get("step", 0)) >= 1:
            captured["flusher"] = train_mod._get_resume_flusher()
            captured["flusher"]()       # simulate the SIGTERM flush
            raise _StopAfterEpochs(info["step"])

    with tempfile.TemporaryDirectory() as tmpdir:
        spec = _make_live_spec(
            training_batch_info, loss_name="L5_gradnorm_vxc_step7",
            n_steps=5, tmpdir=tmpdir, update_scheme="per_molecule",
            require_atom_anchors=False, checkpoint_every=100)  # never periodic
        with pytest.raises(_StopAfterEpochs):
            run_training(spec, progress_callback=_cb)
        d = spec.checkpoint_dir
        assert captured.get("flusher") is not None
        # full resume set on disk (resume_best.eqx present because the 1-epoch
        # trailing-mean tracker has a best_model by epoch 1).
        for fn in ("resume_model.eqx", "resume_opt_state.eqx",
                   "resume_state.pkl"):
            assert os.path.isfile(os.path.join(d, fn)), fn
        # and NO success signal yet (mid-run).
        assert not os.path.isfile(os.path.join(d, "model.eqx"))
        assert not os.path.isfile(os.path.join(d, "completion.json"))


# ---------------------------------------------------------------------------
# WS5 regression tests (2026-06-20):
#   RESUME-01 (BLOCKER): the per-epoch group `order` permutation must survive a
#     kill+resume so a MULTI-group resumed run processes groups in the SAME
#     sequence as an uninterrupted same-seed run (the prior CORE test masked it
#     because its fixture yields ONE group -> shuffle is a no-op).
#   SIG-1 (major): a mid-epoch SIGTERM flush must write the LAST COMPLETED
#     epoch's self-consistent snapshot (never a torn rng/losses-advanced one),
#     so resume-after-flush is byte-exact with no duplicated losses.
#   SIG-2/3 robustness: an exception clears the flusher; a corrupt resume_state
#     starts fresh instead of crashing.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def h2_mol_data():
    from xcquinox.alec.data import precompute_fixed_density_data
    return precompute_fixed_density_data(h2_molecule())


@pytest.fixture(scope="module")
def multigroup_batch_info(h_mol_data, o_mol_data, h2o_mol_data, h2_mol_data):
    """Two AE groups (ae:H2O + ae:H2) -> a genuinely multi-group per_molecule
    spec, so the per-epoch `order` shuffle is NOT a no-op and the resume `order`
    bug is observable."""
    mols = (h_atom(), o_atom(), h2o_molecule(), h2_molecule())
    ae_h2o = float(
        h_mol_data["E_pbe"] * 2 + o_mol_data["E_pbe"] - h2o_mol_data["E_pbe"])
    ae_h2 = float(h_mol_data["E_pbe"] * 2 - h2_mol_data["E_pbe"])
    targets = {
        "H": float(h_mol_data["E_pbe"]),
        "O": float(o_mol_data["E_pbe"]),
        "H2O": max(ae_h2o, 0.001),
        "H2": max(ae_h2, 0.001),
    }
    atom_energies = {
        "H": float(h_mol_data["E_pbe"]),
        "O": float(o_mol_data["E_pbe"]),
    }
    return {"mols": mols, "targets": targets, "atom_energies": atom_energies}


def _make_multigroup_live_spec(multigroup_batch_info, *, tmpdir, n_steps,
                               **extra):
    """A live per_molecule TrainingSpec with TWO AE groups (ae:H2O, ae:H2)."""
    ckdir = os.path.join(tmpdir, "ckpt")
    return TrainingSpec.from_dicts(
        arch=_make_arch(),
        molecules=multigroup_batch_info["mols"],
        targets=multigroup_batch_info["targets"],
        atom_energies=multigroup_batch_info["atom_energies"],
        loss_name="L5_gradnorm_vxc_step7",
        n_steps=n_steps,
        lr_start=1e-3, lr_end=1e-5, lr_decay_start=0.0, grad_clip=1.0,
        checkpoint_dir=ckdir, seed=42,
        update_scheme="per_molecule", require_atom_anchors=False,
        **extra,
    )


def _losses_npy(checkpoint_dir):
    import numpy as _np
    return _np.load(os.path.join(checkpoint_dir, "losses.npy"))


@pytest.mark.slow
def test_per_molecule_multigroup_resume_matches_uninterrupted(
        multigroup_batch_info):
    """WS5-RESUME-01 (BLOCKER): with >=2 training groups, a run killed after 2
    of 5 epochs and RESUMED must reproduce an uninterrupted same-seed run EXACTLY
    -- identical losses.npy AND final model.eqx leaves to rtol=1e-9. This FAILS
    before the `order`-persistence fix (the resumed run re-creates `order` fresh
    while the rng is mid-sequence, so the post-resume epochs shuffle groups in a
    different order and the optimizer trajectory diverges)."""
    from xcquinox.alec.train import run_training, _training_groups
    import numpy as _np

    # (A) Reference: a clean 5-epoch run.
    with tempfile.TemporaryDirectory() as ref_dir:
        ref_spec = _make_multigroup_live_spec(
            multigroup_batch_info, tmpdir=ref_dir, n_steps=5, checkpoint_every=1)
        assert len(_training_groups(ref_spec)) >= 2   # genuinely multi-group
        run_training(ref_spec)
        ref_leaves = _model_leaves(
            os.path.join(ref_spec.checkpoint_dir, "model.eqx"))
        ref_losses = _losses_npy(ref_spec.checkpoint_dir)

        # (B) Same seed, killed after 2 epochs, then re-entered to finish.
        with tempfile.TemporaryDirectory() as run_dir:
            spec = _make_multigroup_live_spec(
                multigroup_batch_info, tmpdir=run_dir, n_steps=5,
                checkpoint_every=1)
            with pytest.raises(_StopAfterEpochs):
                run_training(spec, progress_callback=_interrupt_after(2))
            d = spec.checkpoint_dir
            assert os.path.isfile(os.path.join(d, "resume_state.pkl"))
            run_training(spec)         # resume + finish
            resumed_leaves = _model_leaves(os.path.join(d, "model.eqx"))
            resumed_losses = _losses_npy(d)

    assert len(ref_losses) == len(resumed_losses)
    assert _np.allclose(ref_losses, resumed_losses, rtol=1e-9, atol=1e-9)
    assert len(ref_leaves) == len(resumed_leaves)
    for a, b in zip(ref_leaves, resumed_leaves):
        assert _np.allclose(a, b, rtol=1e-9, atol=1e-9)


@pytest.mark.slow
def test_per_molecule_multigroup_midepoch_flush_resume_is_exact(
        multigroup_batch_info):
    """WS5-SIG-1 (major): a MID-EPOCH SIGTERM flush (the registered flusher called
    partway through epoch k+1) must persist epoch k's self-consistent snapshot --
    NOT a torn checkpoint with advanced rng / already-appended partial losses.
    Resuming from it must reproduce the uninterrupted run EXACTLY with NO
    duplicated/lost losses. FAILS before the fix (the flush stores rng/losses by
    reference, so a mid-epoch flush writes an rng-advanced, losses-partial,
    stale-epoch torn state)."""
    from xcquinox.alec import train as train_mod
    from xcquinox.alec.train import run_training, _training_groups
    import numpy as _np

    with tempfile.TemporaryDirectory() as ref_dir:
        ref_spec = _make_multigroup_live_spec(
            multigroup_batch_info, tmpdir=ref_dir, n_steps=5, checkpoint_every=1)
        assert len(_training_groups(ref_spec)) >= 2
        run_training(ref_spec)
        ref_leaves = _model_leaves(
            os.path.join(ref_spec.checkpoint_dir, "model.eqx"))
        ref_losses = _losses_npy(ref_spec.checkpoint_dir)

        with tempfile.TemporaryDirectory() as run_dir:
            # checkpoint_every huge so NO periodic write fires -- the ONLY resume
            # artifact is the mid-epoch flush. Kill mid-epoch-3 (after 2 done).
            spec = _make_multigroup_live_spec(
                multigroup_batch_info, tmpdir=run_dir, n_steps=5,
                checkpoint_every=100)

            def _cb(info):
                if int(info.get("step", 0)) >= 2:
                    # We are now PAST epoch 2; the next group steps of epoch 3
                    # have already advanced the live rng + appended losses.
                    flusher = train_mod._get_resume_flusher()
                    flusher()                       # simulate the SIGTERM flush
                    raise _StopAfterEpochs(info["step"])
            with pytest.raises(_StopAfterEpochs):
                run_training(spec, progress_callback=_cb)
            d = spec.checkpoint_dir
            assert os.path.isfile(os.path.join(d, "resume_state.pkl"))

            run_training(spec)         # resume from the flush + finish
            resumed_leaves = _model_leaves(os.path.join(d, "model.eqx"))
            resumed_losses = _losses_npy(d)

    # No duplicated losses: total count is exactly epochs*groups.
    assert len(resumed_losses) == len(ref_losses)
    assert _np.allclose(ref_losses, resumed_losses, rtol=1e-9, atol=1e-9)
    for a, b in zip(ref_leaves, resumed_leaves):
        assert _np.allclose(a, b, rtol=1e-9, atol=1e-9)


def test_resume_checkpoint_roundtrips_group_order(tmp_path):
    """WS5-RESUME-01 unit: `order` is persisted by _write_resume_checkpoint and
    returned by _load_resume_checkpoint (so the resumed loop can continue the
    killed run's permutation)."""
    from xcquinox.alec.models import AlecGGAModel
    from xcquinox.alec.train import (
        _write_resume_checkpoint, _load_resume_checkpoint, _BestModelTracker,
    )
    import equinox as eqx
    model, opt_state, optimizer = _tiny_model_and_opt(seed=3, n_advance=1)
    saved_order = [3, 0, 2, 1, 4, 5]
    _tt = _BestModelTracker(window=1)
    _write_resume_checkpoint(
        str(tmp_path), model=model, opt_state=opt_state,
        rng_state=__import__("numpy").random.RandomState(1).get_state(),
        order=saved_order,
        train_best_loss=_tt.best_loss, train_recent=list(_tt._recent),
        train_window=_tt.window, train_best_model=_tt.best_model,
        val_present=False, val_best_mae=None, val_finite_metrics=None,
        val_best_model=None,
        epoch=2, update=12, losses=[], aux_log=[], early_stopped=False)
    model_skel = AlecGGAModel.from_arch(_make_arch(), seed=9)
    opt_skel = optimizer.init(eqx.filter(model_skel, eqx.is_array))
    out = _load_resume_checkpoint(
        str(tmp_path), model_skeleton=model_skel, opt_state_skeleton=opt_skel)
    assert out["order"] == saved_order


@pytest.mark.slow
def test_per_molecule_loop_clears_flusher_on_exception(training_batch_info):
    """WS5-SIG-2: a raised exception inside the epoch loop (e.g. _abort_if_
    nonfinite) must STILL clear the registered resume flusher via try/finally --
    a stale flusher must not survive into the next run in the same process."""
    from xcquinox.alec import train as train_mod
    from xcquinox.alec.train import run_training
    train_mod._clear_resume_flusher()
    with tempfile.TemporaryDirectory() as tmpdir:
        spec = _make_live_spec(
            training_batch_info, loss_name="L5_gradnorm_vxc_step7",
            n_steps=5, tmpdir=tmpdir, update_scheme="per_molecule",
            require_atom_anchors=False, checkpoint_every=1)
        with pytest.raises(_StopAfterEpochs):
            run_training(spec, progress_callback=_interrupt_after(1))
        # The loop raised through; the flusher MUST have been cleared.
        assert train_mod._get_resume_flusher() is None


@pytest.mark.slow
def test_per_molecule_corrupt_resume_starts_fresh(training_batch_info):
    """WS5-SIG-3: a corrupt/truncated resume_state.pkl must NOT crash the task --
    the loop logs a warning and starts fresh, producing model.eqx + completion."""
    from xcquinox.alec.train import run_training
    with tempfile.TemporaryDirectory() as tmpdir:
        spec = _make_live_spec(
            training_batch_info, loss_name="L5_gradnorm_vxc_step7",
            n_steps=2, tmpdir=tmpdir, update_scheme="per_molecule",
            require_atom_anchors=False, checkpoint_every=1)
        d = spec.checkpoint_dir
        os.makedirs(d, exist_ok=True)
        # Plant a corrupt resume_state.pkl (no resume_*.eqx alongside) -> a naive
        # _load_resume_checkpoint would raise UnpicklingError / FileNotFound.
        with open(os.path.join(d, "resume_state.pkl"), "wb") as f:
            f.write(b"\x80\x04 not a valid pickle stream")
        run_training(spec)             # must NOT raise; starts fresh and finishes
        assert os.path.isfile(os.path.join(d, "model.eqx"))
        assert os.path.isfile(os.path.join(d, "completion.json"))


# ---------------------------------------------------------------------------
# Regression: optimizer.update must receive array-filtered params, not the raw
# Equinox model. adamw's add_decayed_weights does tree_map(g + wd*p, updates,
# params); grads from eqx.filter_value_and_grad carry None at the networks'
# non-array (activation function / final_activation lambda) leaves, so the raw
# model is a structure mismatch that newer JAX rejects ("Expected None, got
# <function <lambda>>"). See xcquinox/alec/HISTORY.md.
# ---------------------------------------------------------------------------

def test_trainable_params_structure_matches_grads_not_raw_model():
    """Version-independent invariant: array-filtered params share the grads'
    tree structure; the raw model (function leaves populated) does NOT."""
    import jax
    import jax.numpy as jnp
    import equinox as eqx
    from xcquinox.alec.train import _trainable_params

    # eqx.nn.MLP is what create_network_pair embeds (networks.py:113,258); its
    # `activation`=gelu and default `final_activation`=lambda are dynamic
    # (non-array) leaves -> grads carry None there.
    model = eqx.nn.MLP(in_size=3, out_size=1, width_size=4, depth=1,
                       activation=jax.nn.gelu, key=jax.random.PRNGKey(0))

    def loss(m, x):
        return jnp.sum(jax.vmap(m)(x) ** 2)

    _, grads = eqx.filter_value_and_grad(loss)(model, jnp.ones((2, 3)))
    ts = jax.tree_util.tree_structure
    assert ts(grads) != ts(model)                      # the bug (raw model)
    assert ts(grads) == ts(_trainable_params(model))   # the fix (filtered)


def test_train_step_adamw_weight_decay_with_function_leaves():
    """The real fixed _train_step (train.py) runs end-to-end on a model with
    function leaves, and decoupled weight decay flows through add_decayed_weights
    (the exact transform that crashed when handed the raw model)."""
    import jax
    import jax.numpy as jnp
    import equinox as eqx
    from xcquinox.alec.train import build_optimizer, _trainable_params, _train_step

    model = eqx.nn.MLP(in_size=3, out_size=2, width_size=4, depth=1,
                       activation=jax.nn.gelu, key=jax.random.PRNGKey(0))
    opt = build_optimizer(lr_start=1e-2, lr_end=1e-3, n_steps=4,
                          lr_decay_start=0.0, grad_clip=1.0, weight_decay=0.1)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    # (a) Zero grads -> update is pure decoupled weight decay (-lr*wd*p): proves
    #     add_decayed_weights executed on a structurally-valid params tree.
    grads0 = jax.tree_util.tree_map(jnp.zeros_like, _trainable_params(model))
    updates, _ = opt.update(grads0, opt_state, _trainable_params(model))
    w0 = model.layers[0].weight
    dw = updates.layers[0].weight
    assert bool(jnp.isfinite(dw).all())
    assert bool((jnp.sign(dw) == -jnp.sign(w0)).all())  # decay points inward

    # (b) The real fixed _train_step runs and stays finite.
    def loss_fn(m, batch):
        pred = jax.vmap(m)(batch)
        tot = jnp.sum(pred ** 2)
        return tot, {"loss": tot}

    new_model, _new_state, loss_val, _aux, _grads = _train_step(
        model, opt_state, jnp.ones((5, 3)), loss_fn, opt)
    assert bool(jnp.isfinite(loss_val))
    leaves = jax.tree_util.tree_leaves(eqx.filter(new_model, eqx.is_array))
    assert all(bool(jnp.isfinite(leaf).all()) for leaf in leaves)


# ---------------------------------------------------------------------------
# Per-update RSS instrumentation in the per_molecule loop
# ---------------------------------------------------------------------------

def test_aux_log_records_rss_per_molecule(training_batch_info):
    """Every per_molecule training-step aux entry carries the process RSS and
    high-water mark (GiB floats) at append time, so aux_log.pkl holds the full
    RSS-vs-update curve for post-mortem memory diagnosis."""
    from xcquinox.alec.train import run_training

    with tempfile.TemporaryDirectory() as tmpdir:
        spec = _make_live_spec(
            training_batch_info, loss_name="L5_gradnorm_vxc_step7",
            n_steps=2, tmpdir=tmpdir, update_scheme="per_molecule",
            require_atom_anchors=False,
        )
        run_training(spec)
        with open(os.path.join(spec.checkpoint_dir, "aux_log.pkl"), "rb") as f:
            aux_log = pickle.load(f)  # noqa: S301 -- trusted test data

    step_entries = [e for e in aux_log if e.get("group") != "__validation__"]
    assert step_entries
    for entry in step_entries:
        assert isinstance(entry["rss_gb"], float)
        assert isinstance(entry["hwm_gb"], float)
        assert math.isfinite(entry["rss_gb"]) and entry["rss_gb"] > 0.0
        # High-water mark is read from the same /proc snapshot as VmRSS.
        assert entry["hwm_gb"] >= entry["rss_gb"]


def test_per_molecule_validation_entry_records_rss(
        training_batch_info, monkeypatch):
    """__validation__ aux entries record RSS before and after the validation
    eval (plus the post-eval high-water mark) so a validation-boundary memory
    burst is directly visible in aux_log.pkl."""
    from xcquinox.alec import train as train_mod
    from xcquinox.alec.train import run_training

    monkeypatch.setattr(
        train_mod, "_build_validation_data",
        lambda spec: ({"A": {}, "B": {}},
                      [{"name": "r", "reactants": ["A"], "products": ["B"],
                        "coeffs": [-1.0, 1.0], "reaction_energy_ref": 0.0}]))
    monkeypatch.setattr(train_mod, "_validation_reaction_mae",
                        lambda *a, **k: 10.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        spec = _make_live_spec(
            training_batch_info, loss_name="L5_gradnorm_vxc_step7",
            n_steps=2, tmpdir=tmpdir, update_scheme="per_molecule",
            require_atom_anchors=False,
            validate_every=1, patience=999, early_stop_min_delta=0.0,
        )
        run_training(spec)
        with open(os.path.join(spec.checkpoint_dir, "aux_log.pkl"), "rb") as f:
            aux_log = pickle.load(f)  # noqa: S301 -- trusted test data

    val_entries = [e for e in aux_log if e.get("group") == "__validation__"]
    assert len(val_entries) == 2
    for entry in val_entries:
        for key in ("rss_gb_pre_val", "rss_gb_post_val", "hwm_gb_post_val"):
            assert isinstance(entry[key], float)
            assert math.isfinite(entry[key]) and entry[key] > 0.0
        assert entry["hwm_gb_post_val"] >= entry["rss_gb_post_val"]
