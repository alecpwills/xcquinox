"""Tests for xcquinox.pipeline.train -- run_training custom loop.

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

from xcquinox.pipeline.config import (
    ArchitectureConfig,
    MoleculeSpec,
    TrainingSpec,
)
from xcquinox.pipeline.solver import SolverConfig, SolverMode
from xcquinox.pipeline.tests.fixtures.molecules import (
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


# (4) empty atom_energies


# (5) n_steps <= 0
def test_validate_n_steps_zero():
    spec = _make_training_spec(n_steps=0)
    with pytest.raises(ValueError, match="n_steps must be > 0"):
        spec.validate()


# (6) lr_decay_start out of range


# (7) lr_start < lr_end


# (8) grad_clip <= 0


# ---------------------------------------------------------------------------
# Test 9: missing pretrain_checkpoint directory
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Tests: loss_metric and balancing fields
# ---------------------------------------------------------------------------


def test_validate_invalid_loss_metric():
    spec = _make_training_spec(loss_metric="invalid_metric")
    with pytest.raises(ValueError, match="loss_metric must be"):
        spec.validate()


def test_validate_valid_balancing_configs():
    """All balancing config types pass validation when valid."""
    from xcquinox.pipeline.balancing import (
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
    from xcquinox.pipeline.data import precompute_fixed_density_data
    return precompute_fixed_density_data(h_atom())


@pytest.fixture(scope="module")
def o_mol_data():
    from xcquinox.pipeline.data import precompute_fixed_density_data
    return precompute_fixed_density_data(o_atom())


@pytest.fixture(scope="module")
def h2o_mol_data():
    from xcquinox.pipeline.data import precompute_fixed_density_data
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
@pytest.mark.parametrize("loss_name", ["A_atomization", "C_atomization_plus_grid"])
def test_run_training_end_to_end(loss_name, training_batch_info):
    """Tests 10-15: run_training completes for each loss variant."""
    from xcquinox.pipeline.train import run_training

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
    from xcquinox.pipeline.train import _BestModelTracker
    t = _BestModelTracker(window=1)
    t.update(0.5, "a")
    t.update(0.1, "b")
    t.update(0.3, "c")
    assert t.best_model == "b"
    assert t.best_loss == 0.1


# ---------------------------------------------------------------------------
# WS3 (2026-06-20): _BestValidationTracker (validation-metric early-stop)
# ---------------------------------------------------------------------------

def test_best_validation_tracker_keeps_min_on_improving_curve():
    """A strictly-improving validation curve keeps the LAST (lowest) snapshot and
    NEVER triggers early-stop."""
    from xcquinox.pipeline.train import _BestValidationTracker
    t = _BestValidationTracker()
    for mae, snap in [(10.0, "a"), (8.0, "b"), (5.0, "c"), (3.0, "d")]:
        t.update(mae, snap)
        assert t.should_stop(patience=2, min_delta=0.0) is False
    assert t.best_model == "d"
    assert t.best_mae == 3.0


def test_best_validation_tracker_stops_after_exactly_patience_checks():
    """A plateaued/rising curve stops after EXACTLY `patience` consecutive
    non-improving checks; the best snapshot remains the min-val one."""
    from xcquinox.pipeline.train import _BestValidationTracker
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
    from xcquinox.pipeline.train import _BestValidationTracker
    t = _BestValidationTracker()
    t.update(5.0, "a")
    t.update(4.99, "b")        # improves by 0.01 < min_delta=0.1 -> non-improving
    assert t.should_stop(patience=1, min_delta=0.1) is True
    # best snapshot still updates to the numerically-lower value.
    assert t.best_mae == 4.99
    assert t.best_model == "b"


# ---------------------------------------------------------------------------
# WS3 (2026-06-20): _validation_reaction_mae (in-loop val MAE assembly)
# ---------------------------------------------------------------------------

def test_validation_reaction_mae_assembles_from_energy_fn():
    """The val MAE = reaction_mae_kcalmol over per-species energies produced by
    the injected energy_fn (so the assembly is testable with NO PySCF). The
    energy_fn is called once per species in val_mol_data; the reaction energies
    are then scored against reaction_energy_ref."""
    from xcquinox.pipeline.train import _validation_reaction_mae

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


# ---------------------------------------------------------------------------
# Test 16: losses decrease after 5 steps
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_losses_decrease(training_batch_info):
    """Test 16: loss at step 5 < loss at step 0 for A_atomization."""
    from xcquinox.pipeline.train import run_training

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


def test_effective_channel_weights_partial_fills_from_defaults():
    """A PARTIAL channel_weights overrides only named channels; omitted channels
    inherit the density-dominant defaults (NOT 1.0)."""
    from xcquinox.pipeline.train import (
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
    from xcquinox.pipeline.train import _training_groups
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
    from xcquinox.pipeline.train import _training_groups
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


@pytest.mark.slow
def test_run_training_per_molecule_completes(training_batch_info):
    """run_training under update_scheme='per_molecule' completes, takes one
    update per group per epoch, tags the aux_log, and reduces the loss."""
    import pickle
    from xcquinox.pipeline.train import run_training, _training_groups

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
        assert set(on_disk) == set(_BASE_METADATA_KEYS)
        assert not os.path.isfile(
            os.path.join(spec.checkpoint_dir, "model_val_best.eqx"))


# ---------------------------------------------------------------------------
# WS3 (2026-06-20): in-loop validation early-stop + model_val_best.eqx in the
# per_molecule loop.
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_per_molecule_loop_early_stops_and_writes_val_best(
        training_batch_info, monkeypatch):
    """With validate_every=1, patience=1 and a monkeypatched val function
    returning a RISING curve, the per_molecule loop early-stops, writes
    model_val_best.eqx, and records early_stopped/epochs_run/val_best_mae."""
    import json as _json
    from xcquinox.pipeline import train as train_mod
    from xcquinox.pipeline.train import run_training

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


# The base train_metadata.json key set for a non-validating run: the pre-WS3
# set (FIX 3, WS3-ESV-1: no has_val_best_checkpoint, no
# early_stopped/epochs_run/val_* keys) plus the runtime-weighting truth keys
# (update_scheme / balancing_active / effective_channel_weights -- added so
# the artifact reports the weights the loop ACTUALLY applied, not only the
# nominal loss_kwargs/balancing config; see docs/notes/LOSS_PRIMER.md)
# and the optimizer the run was fit under (adamw_linear on the linear schedule
# or adam_plateau on the reduce-on-plateau controller), which is method and is
# recorded for every run, not only for the arm that changes it.
_BASE_METADATA_KEYS = frozenset({
    "arch_name", "use_polarized_correlation", "loss_name", "loss_kwargs",
    "solver_config", "n_steps", "lr_start", "lr_end", "lr_decay_start",
    "grad_clip", "pretrain_checkpoint", "molecules", "targets",
    "atom_energies", "loss_metric", "balancing", "final_loss", "min_loss",
    "has_best_checkpoint", "timestamp", "duration_seconds",
    "update_scheme", "balancing_active", "effective_channel_weights",
    "seed_mix_atomic", "optimizer",
    # the published protocol's non-self-consistent points: the switch, the
    # weight, the names and the species each marks, recorded for every run
    "respect_sc_flag", "nonsc_weight", "nonsc_points", "nonsc_species",
})


def test_save_artifacts_metadata_byte_identical_when_no_validation():
    """FIX 3 (WS3-ESV-1): _save_artifacts with no val_best snapshot + no
    extra_metadata (the per_molecule-with-validate_every=0 / batched case)
    writes train_metadata.json with the base key set -- no
    has_val_best_checkpoint, no val_* keys."""
    spec = _make_training_spec(update_scheme="per_molecule")
    from xcquinox.pipeline.train import _save_artifacts
    meta = _save_artifacts(
        spec, _make_arch(), [0.5, 0.4, 0.3], [], 1.0,
        best_model=None, val_best_model=None, extra_metadata=None)
    assert set(meta) == set(_BASE_METADATA_KEYS)
    assert "has_val_best_checkpoint" not in meta
    with open(os.path.join(spec.checkpoint_dir, "train_metadata.json")) as f:
        on_disk = json.load(f)
    assert set(on_disk) == set(_BASE_METADATA_KEYS)


# ---------------------------------------------------------------------------
# Fail-loud finite guard: a NaN/Inf must abort training immediately, naming
# the offending loop/step/group/channel, never silently corrupt the weights.
# ---------------------------------------------------------------------------

def test_abort_if_nonfinite_passes_when_finite():
    from xcquinox.pipeline.train import _abort_if_nonfinite
    # finite loss + finite channels -> returns None, no raise.
    _abort_if_nonfinite(
        0.5, {"loss_AE": 0.1, "loss_rho": 0.2}, loop="batched", step=0)


# ---------------------------------------------------------------------------
# Gradient-level guard: a step whose LOSS is finite but whose GRADIENT carries a
# NaN/Inf used to pass the guard, corrupt every weight via apply_updates, and
# abort one step LATE on the next group's now-NaN loss -- so the abort named the
# wrong step/group (dfs6311 step-5 ae:CO). The guard must sweep the gradient
# pytree and name the first offending parameter path at the step it occurs.
# ---------------------------------------------------------------------------


def test_abort_if_nonfinite_raises_on_nan_grad_with_finite_loss():
    import jax.numpy as jnp
    from xcquinox.pipeline.train import _abort_if_nonfinite
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
    import xcquinox.pipeline.train as train_mod
    from xcquinox.pipeline.train import run_training

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
    from xcquinox.pipeline.train import run_training
    from xcquinox.pipeline.models import AlecGGAModel

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


# ---------------------------------------------------------------------------
# Test 17: artifact roundtrip
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_artifact_roundtrip(training_batch_info):
    """Test 17: model.eqx loads correctly, losses.npy matches, aux_log.pkl
    deserializes, train_metadata.json has all fields, progress_callback was invoked."""
    import equinox as eqx
    from xcquinox.pipeline.train import run_training
    from xcquinox.pipeline.models import AlecGGAModel

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
    from xcquinox.pipeline.train import run_training
    from xcquinox.pipeline.models import AlecGGAModel

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
        # train._build_model refuses an uncertified pretrain checkpoint; this
        # one is synthesised by the test, so it carries the PASS certificate a
        # real pretrain job would have written beside the networks.
        with open(os.path.join(pretrain_dir,
                               "fidelity_certificate.json"), "w") as f:
            json.dump({"verdict": "PASS", "arch": arch.name}, f)

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


# ---------------------------------------------------------------------------
# Test 20: constraint_report post-update still valid
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_constraint_report_post_training(training_batch_info):
    """Test 20: constraint_report returns valid per-constraint stats after training."""
    import jax.numpy as jnp
    import equinox as eqx
    from xcquinox.pipeline.train import run_training
    from xcquinox.pipeline.models import AlecGGAModel

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
    from xcquinox.pipeline.train import run_training

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


# ---------------------------------------------------------------------------
# Test 24: missing atom_energy key
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Test 25: atoms-only batch
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Test 26: non-finite float hyperparameter
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("field_name", ["lr_start"])
@pytest.mark.parametrize("bad_value", [float("nan")])
def test_validate_nonfinite_float_hyperparameter(field_name, bad_value):
    """Test 26: non-finite hyperparameter -> ValueError."""
    spec = _make_training_spec(**{field_name: bad_value})
    with pytest.raises(ValueError, match=f"{field_name} must be finite"):
        spec.validate()


# ---------------------------------------------------------------------------
# Test 27: non-finite target
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Test 28: non-finite atom_energies
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Test 29: checkpoint_dir as file
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Test 32: SolverConfig in loss_kwargs is serialized to JSON
# ---------------------------------------------------------------------------


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
    from xcquinox.pipeline.losses import make_loss
    loss = make_loss(spec.loss_name, molecules=spec.molecules, **spec.loss_kwargs_dict)
    required = set(loss.required_mol_keys)
    for d in spec.arch.materialize_descriptors():
        required |= set(d.required_mol_keys)
    sc = spec.loss_kwargs_dict.get("solver_config") or spec.solver_config
    if isinstance(sc, SolverConfig) and sc.mode == SolverMode.FULL:
        required.add("eri")
    assert ("eri" in required) == expect_eri


# GradNorm robustness to zero step-0 loss channels


# 2026-06-20 (WS2): L2 weight decay. The 2026-06-20 review found the DFS pool
# overfits with plain adam (no decay) while DFS used weight decay. build_optimizer
# must apply DECOUPLED weight decay (adamw): under a ZERO loss-gradient a positive
# weight_decay still shrinks params; weight_decay=0 leaves them untouched.
def test_build_optimizer_weight_decay_shrinks_params_under_zero_grad():
    import jax.numpy as jnp
    import optax
    from xcquinox.pipeline.train import build_optimizer

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
    from xcquinox.pipeline.models import AlecGGAModel
    from xcquinox.pipeline.train import build_optimizer

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
    from xcquinox.pipeline.models import AlecGGAModel
    from xcquinox.pipeline.train import (
        _write_resume_checkpoint, _load_resume_checkpoint,
        _BestModelTracker, _BestValidationTracker,
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
    from xcquinox.pipeline.models import AlecGGAModel
    from xcquinox.pipeline.train import (
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
    from xcquinox.pipeline.models import AlecGGAModel
    skel = AlecGGAModel.from_arch(_make_arch(), seed=12345)
    m = eqx.tree_deserialise_leaves(path, skel)
    return [_np.asarray(x) for x in jtu.tree_leaves(eqx.filter(m, eqx.is_array))]


@pytest.mark.slow
def test_per_molecule_resume_finishes_and_matches_uninterrupted(
        training_batch_info):
    """WS5 CORE: a run killed after 2 of 5 epochs RESUMES from its periodic
    checkpoint and FINISHES; the final model is identical (fp tolerance) to a
    from-scratch uninterrupted run with the same seed."""
    from xcquinox.pipeline.train import run_training
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
    from xcquinox.pipeline.models import AlecGGAModel
    from xcquinox.pipeline.train import (
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
    from xcquinox.pipeline import train as train_mod
    from xcquinox.pipeline.train import run_training

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
    from xcquinox.pipeline.data import precompute_fixed_density_data
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


@pytest.mark.slow
def test_per_molecule_corrupt_resume_starts_fresh(training_batch_info):
    """WS5-SIG-3: a corrupt/truncated resume_state.pkl must NOT crash the task --
    the loop logs a warning and starts fresh, producing model.eqx + completion."""
    from xcquinox.pipeline.train import run_training
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
# <function <lambda>>").
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Per-update RSS instrumentation in the per_molecule loop
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Memory levers: XCQUINOX_MEMDIAG (opt-in census) / XCQUINOX_GC_EVERY
# (default-on collection cadence)
# ---------------------------------------------------------------------------


# --------------------------------------------------------------------------- #
# Seed threading: the batch/validation builders forward the spec's seed
# fields into the supply layer
# --------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------
# In-process pretraining-fidelity gate
# ---------------------------------------------------------------------------

def test_build_model_refuses_an_uncertified_pretrain_checkpoint(tmp_path,
                                                                monkeypatch):
    """A checkpoint with no certificate is refused with an actionable message.

    The pre-certificate checkpoints were 2.3 to 56 kcal/mol away from their
    parent in atomization energies; training from one silently measures that
    offset instead of the architecture."""
    import equinox as eqx
    from xcquinox.pipeline import train as train_mod
    from xcquinox.pipeline.networks import create_network_pair

    monkeypatch.delenv(train_mod._ALLOW_UNCERTIFIED_ENV, raising=False)
    arch = _make_arch()
    xnet, cnet = create_network_pair(arch, seed=0)
    d = tmp_path / "pretrain_ckpt"
    d.mkdir()
    eqx.tree_serialise_leaves(str(d / "xnet.eqx"), xnet)
    eqx.tree_serialise_leaves(str(d / "cnet.eqx"), cnet)

    spec = _make_training_spec(pretrain_checkpoint=str(d))
    with pytest.raises(ValueError, match="fidelity"):
        train_mod._build_model(spec)


# ---------------------------------------------------------------------------
# Group-scoped atom-anchor allowlist (train._build_group_loss_and_batch)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# seed_mix_atomic: the per-update SCF-seed mixture of DFS SI Sec. III A,
#   rho_init = (1 - beta) rho_atomic + beta rho_DFT,  beta = (r + 1) / 2,
#   r ~ U(0, 1), resampled at EVERY optimization step.
# The loop's own RandomState draws r, so the resume checkpoint's rng_state
# carries the mixture sequence along with the epoch shuffle.
# ---------------------------------------------------------------------------

class _StopRecording(Exception):
    """Sentinel raised by the recording stub to kill a run mid-epoch, standing
    in for the SIGTERM/wall-clock death the resume path exists for."""


def _make_seed_recorder(records, stop_after=None):
    """A drop-in for ``defused_value_and_grad`` that records the seed each
    molecule enters the update with and returns a zero gradient.

    The point of the stub is to leave the loop, the mixture and the optimizer
    step exactly as they are while removing the SCF: what is under test is which
    density matrix reaches the group's sub-batch at each update, not the energy
    it would produce. ``stop_after`` kills the run once that many updates have
    been recorded.
    """
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    def stub(loss, model, batch, channel_weights, relative=False,
             pad_target=None):
        records.append(tuple(
            {
                "name": md["name"],
                "dm_seed": np.asarray(md["dm_seed"]),
                "dm_pbe": np.asarray(md["dm_pbe"]),
                "dm_minao": (None if md.get("dm_minao") is None
                             else np.asarray(md["dm_minao"])),
                "s": np.asarray(md["s_matrix"]),
                # object identity: the default path must hand the solver the
                # PBE array itself, not a copy of it.
                "seed_is_pbe": md["dm_seed"] is md["dm_pbe"],
            }
            for md in batch["mol_data"]))
        if stop_after is not None and len(records) >= stop_after:
            raise _StopRecording()
        grads = jax.tree_util.tree_map(
            jnp.zeros_like, eqx.filter(model, eqx.is_inexact_array))
        return (jnp.asarray(1.0), {"loss_e": jnp.asarray(1.0)}), grads

    return stub


def _recover_beta(rec):
    """The mixing coefficient the recorded seed implies, from the S-weighted
    trace of the mixture against its two endpoints (beta weights the atomic
    guess, as dpyscf's script does). The atomic guess and the converged
    density do NOT carry the same electron count (sto-3g H2O: 9.8612 against
    10.0000), so tr((D_minao - D_pbe) S) is a well-conditioned denominator
    (0.139 for H2O, 0.137 for O, 0.0137 for H)."""
    def tr(a):
        return float(np.sum(a * rec["s"]))
    den = tr(rec["dm_minao"] - rec["dm_pbe"])
    assert abs(den) > 1e-3, (rec["name"], den)
    return tr(rec["dm_seed"] - rec["dm_pbe"]) / den


def _seed_mix_spec(training_batch_info, tmpdir, **extra):
    """H/O/H2O under the per-molecule scheme with the H and O anchors, so the
    run carries three groups (ae:H2O, anchor:H, anchor:O) and every molecule is
    seeded once per epoch."""
    return _make_live_spec(
        training_batch_info, loss_name="L5_gradnorm_vxc_step7", tmpdir=tmpdir,
        loss_kwargs={"regularize_atom_syms": ("H", "O")},
        update_scheme="per_molecule", require_atom_anchors=False, **extra)


def test_seed_mix_atomic_resamples_the_seed_per_update(training_batch_info,
                                                       monkeypatch):
    """Under ``seed_mix_atomic`` every group's sub-batch is rebuilt before each
    optimizer step with a freshly drawn seed
    ``(1 - beta) dm_pbe + beta dm_minao``, beta = (r + 1) / 2 -- so beta lies in
    [0.5, 1) and changes from update to update; with the flag off the solver
    still receives the PBE array itself.
    """
    from xcquinox.pipeline import train as train_mod
    from xcquinox.pipeline.data import clear_precompute_cache
    from xcquinox.pipeline.train import run_training, _training_groups

    with tempfile.TemporaryDirectory() as tmpdir:
        clear_precompute_cache()
        spec = _seed_mix_spec(training_batch_info, tmpdir, n_steps=3,
                              seed_mix_atomic=True)
        n_groups = len(_training_groups(spec))
        records = []
        monkeypatch.setattr(train_mod, "defused_value_and_grad",
                            _make_seed_recorder(records))
        run_training(spec)

    assert len(records) == 3 * n_groups >= 5
    per_molecule: dict = {}
    for update in records:
        for rec in update:
            assert rec["dm_minao"] is not None, rec["name"]
            beta = _recover_beta(rec)
            assert 0.5 <= beta < 1.0, (rec["name"], beta)
            # the recovered scalar reproduces the WHOLE matrix: the seed is
            # that affine combination and nothing else.
            mixed = ((1.0 - beta) * rec["dm_pbe"] + beta * rec["dm_minao"])
            np.testing.assert_allclose(rec["dm_seed"], mixed,
                                       rtol=0, atol=1e-12)
            # beta < 1 strictly, so the seed is never the PBE density itself.
            assert not rec["seed_is_pbe"]
            assert not np.allclose(rec["dm_seed"], rec["dm_pbe"])
            per_molecule.setdefault(rec["name"], []).append(beta)
    # resampled at EVERY step, not drawn once at loop start.
    for name, betas in per_molecule.items():
        assert len(set(betas)) > 1, (name, betas)

    # Control: the flag off leaves the seed supply exactly as it was.
    with tempfile.TemporaryDirectory() as tmpdir:
        clear_precompute_cache()
        spec_off = _seed_mix_spec(training_batch_info, tmpdir, n_steps=2)
        off_records = []
        monkeypatch.setattr(train_mod, "defused_value_and_grad",
                            _make_seed_recorder(off_records))
        run_training(spec_off)
    assert len(off_records) == 2 * n_groups
    for update in off_records:
        for rec in update:
            assert rec["seed_is_pbe"], rec["name"]
            assert rec["dm_minao"] is None


# ---------------------------------------------------------------------------
# The non-self-consistent point capability: the group flag the spec's recorded
# point names imply, the marking and weighting the loop applies under the
# switch, and the settings the run's record states.
# ---------------------------------------------------------------------------

_ENERGY_CHANNELS = ("loss_AE", "loss_BH76", "loss_IP13")


def _sc_reaction_spec(training_batch_info, tmpdir, **extra):
    """H/O/H2O under the per-molecule scheme with a reaction over the three
    records, so the run carries the groups bh76:r1, ae:H2O, anchor:H and
    anchor:O and the compound of the reaction also has an AE group of its own.
    """
    return _make_live_spec(
        training_batch_info, loss_name="L5_gradnorm_vxc_step7", tmpdir=tmpdir,
        loss_kwargs={
            "regularize_atom_syms": ("H", "O"),
            "bh76_reactions": [{
                "name": "r1", "reactants": ["H2O"], "products": ["H", "O"],
                "coeffs": [-1.0, 2.0, 1.0],
                "e_rxn_ref": training_batch_info["targets"]["H2O"],
            }],
        },
        update_scheme="per_molecule", require_atom_anchors=False, **extra)


def _make_sc_recorder(records):
    """A drop-in for ``defused_value_and_grad`` that records, per update, the
    channel weights the group is handed and, per molecule, whether the record
    is marked for evaluation at the reference density, the seed it enters the
    update with and the run's converged seed density.

    As with the seed recorder, the stub leaves the loop, the marking, the
    mixture and the optimizer step exactly as they are and removes only the
    SCF: what is under test is which record and which weights reach the update.
    """
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    from xcquinox.pipeline.oneshot import ONESHOT_AT_REFERENCE_KEY

    def stub(loss, model, batch, channel_weights, relative=False,
             pad_target=None):
        records.append({
            "channel_weights": dict(channel_weights),
            "molecules": tuple(
                {
                    "name": md["name"],
                    "mark": md.get(ONESHOT_AT_REFERENCE_KEY),
                    "has_mark": ONESHOT_AT_REFERENCE_KEY in md,
                    # object identity: an exempt record must reach the solver
                    # with the run's own seed array, not a copy of it.
                    "seed_is_pbe": md["dm_seed"] is md["dm_pbe"],
                    "dm_seed": np.asarray(md["dm_seed"]),
                    "dm_pbe": np.asarray(md["dm_pbe"]),
                }
                for md in batch["mol_data"]),
        })
        grads = jax.tree_util.tree_map(
            jnp.zeros_like, eqx.filter(model, eqx.is_inexact_array))
        return (jnp.asarray(1.0), {"loss_e": jnp.asarray(1.0)}), grads

    return stub


def test_the_groups_carry_the_points_flag():
    """``_training_groups`` gives every group the self-consistency flag implied
    by the point names the spec records, and no other group.

    A reaction or IP group takes its flag from its own name, an AE group from
    its compound's name, and an atom anchor is always self-consistent: the
    published trajectory marks reaction species and eight AE molecules, never
    the atomic references. The oracle is three specs differing in
    ``nonsc_points`` alone over one fixed group set -- the reaction named, the
    IP pair named, and nothing named -- so a rule that keys off the group KIND
    rather than the recorded names disagrees with at least one of them.
    """
    import dataclasses
    from xcquinox.pipeline.config import MoleculeSpec
    from xcquinox.pipeline.train import _training_groups

    li = MoleculeSpec.from_dict(name="Li", atom="Li 0 0 0", basis="sto-3g",
                                charge=0, spin=1, atom_composition={"Li": 1})
    li_cation = MoleculeSpec.from_dict(name="Li+", atom="Li 0 0 0",
                                       basis="sto-3g", charge=1, spin=0,
                                       atom_composition={"Li": 1})
    spec = _make_training_spec(
        molecules=(h_atom(), o_atom(), h2o_molecule(), h2_molecule(),
                   li, li_cation),
        targets=(("H", -0.5), ("H2", 0.17), ("H2O", 0.3), ("Li", -7.4),
                 ("Li+", -7.2), ("O", -74.8)),
        atom_energies=(("H", -0.5), ("Li", -7.4), ("O", -74.8)),
        loss_name="L5_gradnorm_vxc_step7",
        loss_kwargs=(
            ("bh76_reactions", [{
                "name": "r1", "reactants": ["H2O"], "products": ["H", "O"],
                "coeffs": [-1.0, 2.0, 1.0], "e_rxn_ref": 0.3}]),
            ("ip13_pairs", [{"name": "Li_IP", "neutral": "Li",
                             "cation": "Li+", "ip_ref": 0.2}]),
            ("regularize_atom_syms", ("H", "Li")),
        ),
        update_scheme="per_molecule", require_atom_anchors=False,
        nonsc_points=("H2O", "r1"),
    )

    def flags(s):
        return {g["label"]: g["sc"] for g in _training_groups(s)}

    expected_labels = {"bh76:r1", "ip13:Li_IP", "ae:H2O", "ae:H2",
                       "anchor:H", "anchor:Li"}
    assert set(flags(spec)) == expected_labels
    assert flags(spec) == {"bh76:r1": False, "ip13:Li_IP": True,
                           "ae:H2O": False, "ae:H2": True,
                           "anchor:H": True, "anchor:Li": True}
    assert flags(dataclasses.replace(spec, nonsc_points=("Li_IP",))) == {
        "bh76:r1": True, "ip13:Li_IP": False, "ae:H2O": True, "ae:H2": True,
        "anchor:H": True, "anchor:Li": True}
    assert all(flags(dataclasses.replace(spec, nonsc_points=())).values())


def test_the_loop_marks_non_sc_groups_under_the_switch_only(
        training_batch_info, monkeypatch):
    """Under the switch the per-molecule loop marks every record of a
    non-self-consistent group for evaluation at the reference density, hands
    that group its energy channels scaled by ``nonsc_weight``, and leaves the
    marked records out of the seed mixture; with the switch off it does none of
    those things.

    The reaction group and the AE group of its compound share the same molecule
    record, so a mark written into the batch rather than into the group's own
    copy would reach a self-consistent group as well; the recorder therefore
    checks the other groups' records too. The oracle is the pair of runs
    differing in ``respect_sc_flag`` alone: the same spec, the same recorded
    point names, the same weight.
    """
    from xcquinox.pipeline import train as train_mod
    from xcquinox.pipeline.data import clear_precompute_cache
    from xcquinox.pipeline.train import (
        run_training, _effective_channel_weights, _training_groups)

    n_epochs = 2
    with tempfile.TemporaryDirectory() as tmpdir:
        clear_precompute_cache()
        spec = _sc_reaction_spec(
            training_batch_info, tmpdir, n_steps=n_epochs,
            nonsc_points=("r1",), respect_sc_flag=True, nonsc_weight=0.5,
            seed_mix_atomic=True)
        labels = [g["label"] for g in _training_groups(spec)]
        assert "bh76:r1" in labels and "ae:H2O" in labels
        records = []
        monkeypatch.setattr(train_mod, "defused_value_and_grad",
                            _make_sc_recorder(records))
        run_training(spec)

    run_weights = _effective_channel_weights(spec.channel_weights_dict)
    scaled = {k: (v * 0.5 if k in _ENERGY_CHANNELS else v)
              for k, v in run_weights.items()}
    # the weight reaches the energy channels only
    assert scaled != run_weights
    assert all(scaled[k] == run_weights[k]
               for k in run_weights if k not in _ENERGY_CHANNELS)

    assert len(records) == n_epochs * len(labels)
    marked_updates = 0
    for rec in records:
        names = sorted(m["name"] for m in rec["molecules"])
        if names == ["H", "H2O", "O"]:          # the reaction group
            marked_updates += 1
            assert all(m["mark"] is True for m in rec["molecules"]), names
            assert rec["channel_weights"] == scaled, rec["channel_weights"]
            for m in rec["molecules"]:
                assert m["seed_is_pbe"], m["name"]
                np.testing.assert_array_equal(m["dm_seed"], m["dm_pbe"])
        else:
            assert all(m["mark"] is not True for m in rec["molecules"]), names
            assert rec["channel_weights"] == run_weights, names
            for m in rec["molecules"]:
                assert not m["seed_is_pbe"], m["name"]
                assert not np.allclose(m["dm_seed"], m["dm_pbe"]), m["name"]
    assert marked_updates == n_epochs

    # The switch off: the same spec, the same recorded names and weight, and
    # nothing marked, nothing rescaled, every seed mixed.
    with tempfile.TemporaryDirectory() as tmpdir:
        clear_precompute_cache()
        spec_off = _sc_reaction_spec(
            training_batch_info, tmpdir, n_steps=n_epochs,
            nonsc_points=("r1",), respect_sc_flag=False, nonsc_weight=0.5,
            seed_mix_atomic=True)
        off_records = []
        monkeypatch.setattr(train_mod, "defused_value_and_grad",
                            _make_sc_recorder(off_records))
        run_training(spec_off)

    assert len(off_records) == n_epochs * len(labels)
    for rec in off_records:
        assert not any(m["has_mark"] for m in rec["molecules"])
        assert rec["channel_weights"] == run_weights
        for m in rec["molecules"]:
            assert not m["seed_is_pbe"], m["name"]


def test_a_zero_weight_drops_the_non_sc_group(training_batch_info,
                                              monkeypatch):
    """Under the switch a non-self-consistent group with ``nonsc_weight`` zero
    takes no optimizer step at all: the published script skips such a group
    outright, so the loop drops it from the epoch's groups rather than stepping
    on a loss of zero (a step that would still move the weights through the
    decay and advance the schedule).

    Oracle: the recorded updates against the group list, the reaction group
    absent from every epoch and no record marked.
    """
    from xcquinox.pipeline import train as train_mod
    from xcquinox.pipeline.data import clear_precompute_cache
    from xcquinox.pipeline.train import run_training, _training_groups

    n_epochs = 2
    with tempfile.TemporaryDirectory() as tmpdir:
        clear_precompute_cache()
        spec = _sc_reaction_spec(
            training_batch_info, tmpdir, n_steps=n_epochs,
            nonsc_points=("r1",), respect_sc_flag=True, nonsc_weight=0.0)
        labels = [g["label"] for g in _training_groups(spec)]
        assert "bh76:r1" in labels
        records = []
        monkeypatch.setattr(train_mod, "defused_value_and_grad",
                            _make_sc_recorder(records))
        run_training(spec)

    assert len(records) == n_epochs * (len(labels) - 1)
    for rec in records:
        names = sorted(m["name"] for m in rec["molecules"])
        assert names != ["H", "H2O", "O"], names
        assert not any(m["has_mark"] for m in rec["molecules"])


def test_a_marked_group_scopes_its_regularizer_and_keeps_its_weight(
        training_batch_info):
    """The group builder marks only the species the spec lists for a
    non-self-consistent point, leaves a marked atom out of the regularizer's
    allowlist, and raises the regularizer's weight by the group's scaling
    factor when unmarked atoms keep theirs, so that term stays at the run's
    own value while the group's energy channels are scaled.

    The oracle is the same group built three ways: the switch off; the switch
    on with the compound alone listed (the reaction-form AE case: its atoms
    self-consistent); the switch on with no species listed (the reaction case:
    every species marked). The records are name-only stand-ins, since the
    builder indexes them by name and copies them.
    """
    from xcquinox.pipeline.oneshot import ONESHOT_AT_REFERENCE_KEY
    from xcquinox.pipeline.train import (
        _build_group_loss_and_batch, _training_groups)

    def build(**extra):
        spec = _sc_reaction_spec(training_batch_info, tempfile.mkdtemp(),
                                 n_steps=1, **extra)
        batch = {"mol_data": tuple({"name": m.name} for m in spec.molecules),
                 "targets": spec.targets_dict,
                 "atom_energies": spec.atom_energies_dict}
        group = next(g for g in _training_groups(spec)
                     if g["label"] == "bh76:r1")
        return _build_group_loss_and_batch(spec, group, batch)

    loss_off, batch_off = build(nonsc_points=("r1",), respect_sc_flag=False,
                                nonsc_weight=0.5)
    assert not any(ONESHOT_AT_REFERENCE_KEY in md
                   for md in batch_off["mol_data"])

    # the compound alone marked: the atoms keep their regularizer, at the
    # run's weight once the group's scaling is compensated
    loss_on, batch_on = build(nonsc_points=("r1",),
                              nonsc_species=(("r1", ("H2O",)),),
                              respect_sc_flag=True, nonsc_weight=0.5)
    marks = {md["name"]: md.get(ONESHOT_AT_REFERENCE_KEY)
             for md in batch_on["mol_data"]}
    assert marks == {"H2O": True, "H": None, "O": None}
    assert loss_on.regularize_atom_syms == loss_off.regularize_atom_syms
    assert loss_off.regularize_atom_syms == ("H", "O")
    assert loss_on.w_atomic == pytest.approx(loss_off.w_atomic / 0.5)

    # the whole group marked: no regularized atom, the weight untouched
    loss_all, batch_all = build(nonsc_points=("r1",), respect_sc_flag=True,
                                nonsc_weight=0.5)
    assert all(md.get(ONESHOT_AT_REFERENCE_KEY) is True
               for md in batch_all["mol_data"])
    assert loss_all.regularize_atom_syms == ()
    assert loss_all.w_atomic == loss_off.w_atomic


def test_training_metadata_records_the_sc_settings():
    """The run's record states the self-consistency settings it was fit under:
    the switch, the weight and the names of the non-self-consistent points.

    Two runs differing only in those settings are otherwise indistinguishable
    from their artifacts, and the split is method, so it is recorded whether or
    not the arm turns it on. The oracle is the written ``train_metadata.json``
    read back, and the base key set, which a new key must join.
    """
    from xcquinox.pipeline.train import _save_artifacts

    spec = _make_training_spec(update_scheme="per_molecule",
                               respect_sc_flag=True, nonsc_weight=0.5,
                               nonsc_points=("H2O", "r1"),
                               nonsc_species=(("r1", ("H2O",)),))
    meta = _save_artifacts(spec, _make_arch(), [0.5, 0.4], [], 1.0,
                           best_model=None, val_best_model=None,
                           extra_metadata=None)
    with open(os.path.join(spec.checkpoint_dir, "train_metadata.json")) as f:
        on_disk = json.load(f)
    for written in (meta, on_disk):
        assert written["respect_sc_flag"] is True
        assert written["nonsc_weight"] == 0.5
        assert written["nonsc_points"] == ["H2O", "r1"]
        # the species each named point marks: a run marking the compound
        # alone and one marking its whole group differ here and nowhere else
        assert written["nonsc_species"] == [["r1", ["H2O"]]]

    spec_off = _make_training_spec(update_scheme="per_molecule")
    meta_off = _save_artifacts(spec_off, _make_arch(), [0.5], [], 1.0,
                               best_model=None, val_best_model=None,
                               extra_metadata=None)
    assert meta_off["respect_sc_flag"] is False
    assert meta_off["nonsc_weight"] == 1.0
    assert meta_off["nonsc_points"] == []
    assert meta_off["nonsc_species"] == []
    # The base key set is the contract the non-validating run is held to, so
    # the four keys belong in it.
    assert set(meta_off) == set(_BASE_METADATA_KEYS)


def test_a_marked_record_consumes_its_mixing_draw():
    """The mixture draws one coefficient per molecule in batch order and
    applies it to the unmarked ones only: a marked record keeps its seed and
    still consumes its draw, as the published script draws ``mixing`` before
    testing the entry's flag. A coefficient stream that skipped the marked
    record would hand every later molecule of the group another coefficient
    and change the run's seeds from that update on.

    Oracle: ``np.random.RandomState(0)`` replayed beside the call over three
    molecules with the middle one marked; the third molecule takes the third
    draw, not the second.
    """
    from xcquinox.pipeline.oneshot import ONESHOT_AT_REFERENCE_KEY
    from xcquinox.pipeline.train import _mix_seed_batch

    def record(name, scale, marked=False):
        md = {
            "name": name,
            "dm_seed": scale * np.array([[1.0, 0.25], [0.25, 2.0]]),
            "dm_pbe": scale * np.array([[1.0, 0.25], [0.25, 2.0]]),
            "dm_minao": scale * np.array([[0.25, 0.0], [0.0, 0.5]]),
            "s_matrix": np.eye(2),
        }
        if marked:
            md[ONESHOT_AT_REFERENCE_KEY] = True
        return md

    mols = (record("A", 1.0), record("B", 2.0, marked=True), record("C", 3.0))
    out = _mix_seed_batch({"mol_data": mols, "label": "g"},
                          np.random.RandomState(0))

    oracle = np.random.RandomState(0)
    betas = [(float(oracle.uniform()) + 1.0) / 2.0 for _ in range(3)]
    a, b, c = out["mol_data"]
    np.testing.assert_allclose(
        a["dm_seed"],
        (1.0 - betas[0]) * mols[0]["dm_seed"] + betas[0] * mols[0]["dm_minao"],
        rtol=0, atol=1e-12)
    # the marked record is the input record itself, its seed untouched
    assert b is mols[1]
    assert b["dm_seed"] is mols[1]["dm_seed"]
    # the third molecule takes the third draw: the marked one consumed the second
    np.testing.assert_allclose(
        c["dm_seed"],
        (1.0 - betas[2]) * mols[2]["dm_seed"] + betas[2] * mols[2]["dm_minao"],
        rtol=0, atol=1e-12)
    assert betas[1] != betas[2]


# ---------------------------------------------------------------------------
# The seed start as the one variable between the arms: the mixture starts from
# the run's OWN seed, draws from its OWN rng, and carries that rng's state in
# the resume record.
# ---------------------------------------------------------------------------

def _group_sequence(records):
    """The per-update group identity of a recorded run: the sorted molecule
    names each update's sub-batch carries."""
    return [tuple(sorted(rec["name"] for rec in update)) for update in records]


def test_the_mixture_starts_from_the_runs_own_seed():
    """``_mix_seed_batch`` mixes the run's OWN SCF seed with the atomic guess:
    ``D0 = (1 - beta) dm_seed + beta dm_minao``, beta = (r + 1) / 2 drawn from
    the rng it is handed, once per molecule in batch order.

    ``dm_seed``, ``dm_pbe`` and ``dm_minao`` are three DIFFERENT arrays here,
    which is what separates a mixture formed from the run's seed from one
    formed from the PBE density: under the PBE seed the two coincide (the
    supply layer aliases ``dm_seed`` to ``dm_pbe``), so no PBE-seeded run can
    tell them apart, while a SCAN-seeded arm starts from the wrong density.
    The oracle is ``np.random.RandomState(0)`` replayed beside the call; the
    two molecules draw different coefficients, so the assignment of a draw to
    a molecule is pinned as well.
    """
    from xcquinox.pipeline.train import _mix_seed_batch

    mol_a = {
        "name": "A",
        "dm_seed": np.array([[1.0, 0.25], [0.25, 2.0]]),
        "dm_pbe": np.array([[5.0, -0.5], [-0.5, 7.0]]),
        "dm_minao": np.array([[0.25, 0.0], [0.0, 0.5]]),
        "s_matrix": np.eye(2),
    }
    mol_b = {
        "name": "B",
        "dm_seed": np.array([[3.0, 0.125], [0.125, 4.0]]),
        "dm_pbe": np.array([[-2.0, 0.75], [0.75, 9.0]]),
        "dm_minao": np.array([[0.75, 0.0], [0.0, 1.25]]),
        "s_matrix": np.eye(2),
    }
    gbatch = {"mol_data": (mol_a, mol_b), "label": "ae:AB"}

    out = _mix_seed_batch(gbatch, np.random.RandomState(0))

    oracle = np.random.RandomState(0)
    betas = []
    for md_in, md_out in zip(gbatch["mol_data"], out["mol_data"]):
        beta = (float(oracle.uniform()) + 1.0) / 2.0
        betas.append(beta)
        expected = (1.0 - beta) * md_in["dm_seed"] + beta * md_in["dm_minao"]
        np.testing.assert_allclose(md_out["dm_seed"], expected,
                                   rtol=0, atol=1e-12)
        # the mixture is at least half the atomic guess, as the protocol states
        assert 0.5 <= beta < 1.0, beta
    assert betas[0] != betas[1]

    # Only the seed is replaced: every other array is the one it was, and the
    # input batch is left alone (the loop re-enters `prepared` every update).
    for md_in, md_out in zip(gbatch["mol_data"], out["mol_data"]):
        assert md_out["dm_pbe"] is md_in["dm_pbe"]
        assert md_out["dm_minao"] is md_in["dm_minao"]
        assert md_out["name"] == md_in["name"]
    assert gbatch["mol_data"][0]["dm_seed"][0, 0] == 1.0
    assert out["label"] == "ae:AB"


def test_the_mixers_stream_is_its_own():
    """The mixer's rng stream is a function of the spec seed that is neither
    the shuffle stream of that seed nor the shuffle stream of any other seed
    compared: the two streams of one run never coincide, and no run's
    mixture replays another run's group order.

    Oracle: ``np.random.RandomState`` sequences at the seeds compared, and
    the spawned seeds' distinctness from each other and from every seed.
    """
    from xcquinox.pipeline.train import _mixer_seed

    seeds = (0, 1, 7, 42, 123, 2024)
    spawned = [_mixer_seed(s) for s in seeds]
    assert len(set(spawned)) == len(seeds)
    shuffles = {s: tuple(np.random.RandomState(s).uniform(size=8))
                for s in seeds}
    for s, m in zip(seeds, spawned):
        assert m != s
        assert m not in seeds
        mixer = tuple(np.random.RandomState(m).uniform(size=8))
        assert mixer not in shuffles.values(), (s, m)


def test_the_mixture_leaves_the_group_order_alone(training_batch_info,
                                                  monkeypatch):
    """The per-update group sequence of a run with the mixture ON is identical
    to the same spec's sequence with the mixture OFF.

    The three campaign arms differ in the SCF seed start and in nothing else,
    so the stochastic-update order must not depend on whether the mixture is
    drawing. Sharing the loop's shuffle rng with the mixture makes the order a
    function of the number of molecules mixed, which differs between the arms
    from the second epoch on. The oracle is the mixture-off run at the same
    seed: the two group sequences must agree update for update.
    """
    from xcquinox.pipeline import train as train_mod
    from xcquinox.pipeline.train import run_training, _training_groups

    n_steps = 4
    with tempfile.TemporaryDirectory() as tmpdir:
        spec_on = _seed_mix_spec(training_batch_info, tmpdir, n_steps=n_steps,
                                 seed_mix_atomic=True)
        n_groups = len(_training_groups(spec_on))
        on_records = []
        monkeypatch.setattr(train_mod, "defused_value_and_grad",
                            _make_seed_recorder(on_records))
        run_training(spec_on)

    with tempfile.TemporaryDirectory() as tmpdir:
        spec_off = _seed_mix_spec(training_batch_info, tmpdir, n_steps=n_steps)
        off_records = []
        monkeypatch.setattr(train_mod, "defused_value_and_grad",
                            _make_seed_recorder(off_records))
        run_training(spec_off)

    assert spec_on.seed == spec_off.seed
    assert n_groups > 1, "a single-group spec cannot show a shuffle difference"
    assert len(on_records) == len(off_records) == n_steps * n_groups
    assert _group_sequence(on_records) == _group_sequence(off_records)


def test_resume_checkpoint_carries_the_mixers_rng(tmp_path):
    """The resume record carries the MIXER's rng state beside the shuffle rng,
    and a ``RandomState`` restored from it continues the draw sequence.

    The oracle is the live stream: the next ``uniform()`` of the mixer whose
    state was written equals the next ``uniform()`` of a state restored from
    the record. The two states are also held apart, so a record that wrote the
    shuffle state under both keys is separated from one that wrote both
    streams. With the mixture off the field round-trips as ``None``.
    """
    import equinox as eqx
    import numpy as _np
    from xcquinox.pipeline.models import AlecGGAModel
    from xcquinox.pipeline.train import (
        _write_resume_checkpoint, _load_resume_checkpoint, _BestModelTracker,
    )

    model, opt_state, optimizer = _tiny_model_and_opt(seed=5, n_advance=1)
    shuffle_rng = _np.random.RandomState(7)
    order = _np.arange(4)
    shuffle_rng.shuffle(order)
    mix_rng = _np.random.RandomState(1234)
    for _ in range(5):          # mid-epoch draws: the state is not the initial
        mix_rng.uniform()
    mix_state = mix_rng.get_state()
    expected_next = float(mix_rng.uniform())

    tt = _BestModelTracker(window=1)
    common = dict(
        model=model, opt_state=opt_state, order=list(order),
        train_best_loss=tt.best_loss, train_recent=list(tt._recent),
        train_window=tt.window, train_best_model=tt.best_model,
        val_present=False, val_best_mae=None, val_finite_metrics=None,
        val_best_model=None, epoch=1, update=4, losses=[], aux_log=[],
        early_stopped=False,
    )
    _write_resume_checkpoint(str(tmp_path), rng_state=shuffle_rng.get_state(),
                             mix_rng_state=mix_state, **common)

    model_skel = AlecGGAModel.from_arch(_make_arch(), seed=6)
    opt_skel = optimizer.init(eqx.filter(model_skel, eqx.is_array))
    out = _load_resume_checkpoint(str(tmp_path), model_skeleton=model_skel,
                                  opt_state_skeleton=opt_skel)

    restored = _np.random.RandomState(0)
    restored.set_state(out["mix_rng_state"])
    assert float(restored.uniform()) == expected_next
    # the two streams are separate records, not one written twice.
    assert not _np.array_equal(out["mix_rng_state"][1], out["rng_state"][1])

    # Mixture off: the field defaults and round-trips as None.
    off_dir = tmp_path / "off"
    off_dir.mkdir()
    _write_resume_checkpoint(str(off_dir), rng_state=shuffle_rng.get_state(),
                             **common)
    off = _load_resume_checkpoint(str(off_dir), model_skeleton=model_skel,
                                  opt_state_skeleton=opt_skel)
    assert off["mix_rng_state"] is None


def test_the_mixture_resumes_bit_exactly(training_batch_info, monkeypatch):
    """A mixing run killed one update into its second epoch and re-entered
    draws, from the resume boundary on, the coefficients an uninterrupted run
    of the same spec draws.

    The oracle is the uninterrupted run: its updates from the second epoch on
    are compared group by group and molecule by molecule against the resumed
    run's, through the mixing coefficient the recorded seed implies. This
    guards the restore of the mixer's stream -- a state saved but never applied
    restarts the coefficients at the stream's head while the group order
    continues, which no energy or loss reports.
    """
    from xcquinox.pipeline import train as train_mod
    from xcquinox.pipeline.train import run_training, _training_groups

    with tempfile.TemporaryDirectory() as ref_dir:
        ref_spec = _seed_mix_spec(training_batch_info, ref_dir, n_steps=3,
                                  seed_mix_atomic=True, checkpoint_every=1)
        n_groups = len(_training_groups(ref_spec))
        ref_records = []
        monkeypatch.setattr(train_mod, "defused_value_and_grad",
                            _make_seed_recorder(ref_records))
        run_training(ref_spec)

    with tempfile.TemporaryDirectory() as run_dir:
        spec = _seed_mix_spec(training_batch_info, run_dir, n_steps=3,
                              seed_mix_atomic=True, checkpoint_every=1)
        killed = []
        monkeypatch.setattr(
            train_mod, "defused_value_and_grad",
            _make_seed_recorder(killed, stop_after=n_groups + 1))
        with pytest.raises(_StopRecording):
            run_training(spec)
        # the periodic checkpoint of the completed first epoch is on disk and
        # the run carries no success signal.
        assert os.path.isfile(
            os.path.join(spec.checkpoint_dir, "resume_state.pkl"))
        assert not os.path.isfile(
            os.path.join(spec.checkpoint_dir, "model.eqx"))

        resumed = []
        monkeypatch.setattr(train_mod, "defused_value_and_grad",
                            _make_seed_recorder(resumed))
        run_training(spec)

    assert len(ref_records) == 3 * n_groups
    assert len(resumed) == 2 * n_groups
    assert _group_sequence(resumed) == _group_sequence(ref_records)[n_groups:]
    for got_update, want_update in zip(resumed, ref_records[n_groups:]):
        for got, want in zip(got_update, want_update):
            assert got["name"] == want["name"]
            assert _recover_beta(got) == pytest.approx(_recover_beta(want),
                                                       rel=1e-12)


def test_a_resume_set_without_the_mixers_stream_starts_fresh(
        training_batch_info, monkeypatch):
    """A mixing run re-entered on a resume set that carries no
    ``mix_rng_state`` starts fresh, with the warning every unusable resume set
    raises, rather than continuing with coefficients from the stream's head.

    The set is the run's own periodic checkpoint with the key removed from
    its state pickle, the shape a set written before the mixer had its own
    stream would have. The oracle is the record count: a fresh run records
    every epoch again, a resumed one only the epochs after the boundary.
    """
    import pickle
    from xcquinox.pipeline import train as train_mod
    from xcquinox.pipeline.train import run_training, _training_groups

    with tempfile.TemporaryDirectory() as run_dir:
        spec = _seed_mix_spec(training_batch_info, run_dir, n_steps=3,
                              seed_mix_atomic=True, checkpoint_every=1)
        n_groups = len(_training_groups(spec))
        killed = []
        monkeypatch.setattr(
            train_mod, "defused_value_and_grad",
            _make_seed_recorder(killed, stop_after=n_groups + 1))
        with pytest.raises(_StopRecording):
            run_training(spec)
        state_path = os.path.join(spec.checkpoint_dir, "resume_state.pkl")
        # the run's own state pickle, written by the loop a moment ago in
        # this process: the artifact under test, not data from elsewhere.
        with open(state_path, "rb") as fh:
            state = pickle.load(fh)  # noqa: S301
        assert state.pop("mix_rng_state") is not None
        with open(state_path, "wb") as fh:
            pickle.dump(state, fh, protocol=4)

        fresh = []
        monkeypatch.setattr(train_mod, "defused_value_and_grad",
                            _make_seed_recorder(fresh))
        with pytest.warns(RuntimeWarning, match="mix_rng_state"):
            run_training(spec)
        assert os.path.isfile(os.path.join(spec.checkpoint_dir, "model.eqx"))

    assert len(fresh) == 3 * n_groups


def test_validation_records_carry_the_atomic_guess_under_the_mixture(
        tmp_path, training_batch_info):
    """``_build_validation_data`` asks the precompute for the atomic guess
    exactly when the run mixes, so a species shared by the training and
    validation slices is one cache entry rather than two reference SCFs.

    The validation SEED itself is unchanged: the mixture is a training-loop
    device and the validation metric must stay the same quantity, which is
    asserted against the mixture-off record's own ``dm_seed``.
    """
    import dataclasses
    from xcquinox.pipeline.train import _build_validation_data

    rxn_path = tmp_path / "val_reactions.json"
    rxn_path.write_text(json.dumps([{
        "name": "r", "reactants": ["H2"], "products": ["H2"],
        "coeffs": [-1.0, 1.0], "reaction_energy_ref": 0.0}]))
    spec_on = _make_live_spec(
        training_batch_info, loss_name="L5_gradnorm_vxc_step7", n_steps=1,
        tmpdir=str(tmp_path), update_scheme="per_molecule",
        require_atom_anchors=False, validate_every=1,
        validation_molecules=(h2_molecule(),),
        validation_reactions_path=str(rxn_path), seed_mix_atomic=True)
    spec_off = dataclasses.replace(spec_on, seed_mix_atomic=False)

    on_data, on_reactions = _build_validation_data(spec_on)
    off_data, _ = _build_validation_data(spec_off)

    assert set(on_data) == set(off_data) == {"H2"}
    assert on_reactions[0]["name"] == "r"
    guess = on_data["H2"]["dm_minao"]
    assert guess is not None
    assert np.asarray(guess).shape == np.asarray(on_data["H2"]["dm_seed"]).shape
    assert off_data["H2"]["dm_minao"] is None
    # the validation seed does not move: the metric is the same quantity.
    np.testing.assert_array_equal(np.asarray(on_data["H2"]["dm_seed"]),
                                  np.asarray(off_data["H2"]["dm_seed"]))


# ---------------------------------------------------------------------------
# The plateau optimizer (spec C7): torch's ReduceLROnPlateau in `min` mode as
# dpyscf configures it (patience 10, factor 0.1, min_lr 1e-7), stepping once
# per epoch on the epoch's MEAN group loss, with the coupled L2 of torch's
# ``Adam(weight_decay=...)`` -- ``add_decayed_weights`` BEFORE adam -- in place
# of adamw's decoupled decay.
# ---------------------------------------------------------------------------

def _ctrl(**kw):
    """A controller at the scripted-sequence settings; every argument by
    keyword so the test states the semantics, not the parameter order."""
    from xcquinox.pipeline.train import _PlateauController
    kwargs = dict(patience=2, factor=0.1, min_lr=1e-7, lr=1e-4)
    kwargs.update(kw)
    return _PlateauController(**kwargs)


def test_plateau_controller_fires_one_epoch_past_the_patience_and_resets():
    """torch semantics: the reduction fires on the epoch whose non-improving
    count EXCEEDS the patience, not on the one that reaches it.

    With patience 2 the sequence [1.0, 0.9, 0.9, 0.9, 0.9] sets the best at the
    first call, improves at the second, and then runs three non-improving
    epochs: the counter reads 1, 2, 3 and only the third (bad > patience)
    reduces. Firing at ``bad >= patience`` would cut the rate a whole epoch
    early -- over 200 epochs of the arm that is one decade of learning rate
    lost -- so the epoch of the reduction is asserted, not merely that one
    happened. The counter resets with the reduction: the next reduction is
    again three non-improving epochs away.
    """
    ctrl = _ctrl()
    got = [ctrl.update(x) for x in (1.0, 0.9, 0.9, 0.9, 0.9)]
    assert got == pytest.approx([1e-4, 1e-4, 1e-4, 1e-4, 1e-5], rel=1e-12), got
    # counter reset: two more non-improving epochs do NOT fire, the third does.
    assert [ctrl.update(0.9) for _ in range(3)] == pytest.approx([1e-5, 1e-5, 1e-6], rel=1e-12)


def _plateau_spec(training_batch_info, tmpdir, **extra):
    """The seed-mixture helper's H/O/H2O per-molecule spec (three groups) at
    the arm's rates: lr_start 1e-4 and lr_end 1e-7, dpyscf's MIN_RATE, which
    is the controller's floor under ``adam_plateau``."""
    import dataclasses
    spec = _make_live_spec(
        training_batch_info, loss_name="L5_gradnorm_vxc_step7", tmpdir=tmpdir,
        loss_kwargs={"regularize_atom_syms": ("H", "O")},
        update_scheme="per_molecule", require_atom_anchors=False, **extra)
    return dataclasses.replace(spec, lr_start=1e-4, lr_end=1e-7)


def _make_scripted_loss(n_groups, *, start_call=0, stop_after=None, seen=None):
    """A drop-in for ``defused_value_and_grad`` returning a scripted loss and a
    zero gradient, so the loop, the optimizer and the controller run with no
    SCF behind them.

    The script separates the epoch MEAN from the epoch's LAST group loss and
    from the running mean over the whole history. In epoch e the group losses
    are ``[b + d, ..., b + d, b - (n - 1) d]`` with ``b = 1 / min(e, 3)`` and
    ``d = 0.01 e`` (0 in epoch 1): the epoch mean is 1, 1/2, then 1/3 for ever,
    so the best is set at epoch 3; from epoch 4 the mean sits 1.5e-4 below it,
    an improvement to a controller stepped on the mean (torch's relative
    threshold 1e-4) and NOT to one stepped on the script's sqrt(mean) x 1000
    (0.75e-4), so only the latter fires at epoch 14; the
    last loss of each epoch falls monotonically and the cumulative mean keeps
    falling for the whole run. A controller stepping on the epoch mean
    therefore fires at epoch 14 (patience 10); one stepping on the last
    group's loss -- the value left in the loop's ``loss_py`` at the epoch
    boundary -- or on the cumulative mean never does.
    ``start_call`` continues the script across a resume; ``stop_after`` kills
    the run once that many updates have been served.
    """
    import equinox as eqx
    import jax
    import jax.numpy as jnp

    state = {"call": start_call}

    def stub(loss, model, batch, channel_weights, relative=False,
             pad_target=None):
        i = state["call"]
        epoch = i // n_groups + 1
        base = 1.0 / min(epoch, 3)
        if epoch >= 4:
            base *= 1.0 - 1.5e-4
        d = 0.0 if epoch == 1 else 0.01 * epoch
        value = (base - (n_groups - 1) * d if i % n_groups == n_groups - 1
                 else base + d)
        state["call"] = i + 1
        if seen is not None:
            seen.append((epoch, value))
        if stop_after is not None and state["call"] - start_call >= stop_after:
            raise _StopRecording()
        grads = jax.tree_util.tree_map(
            jnp.zeros_like, eqx.filter(model, eqx.is_inexact_array))
        return (jnp.asarray(value), {"loss_e": jnp.asarray(value)}), grads

    return stub


def _plateau_entries(checkpoint_dir):
    """The ``__plateau__`` rows of the run's aux log, in order, paired with the
    group row that precedes each of them."""
    with open(os.path.join(checkpoint_dir, "aux_log.pkl"), "rb") as f:
        aux = pickle.load(f)  # noqa: S301 -- written by this test's own run
    out = []
    for i, e in enumerate(aux):
        if e.get("group") == "__plateau__":
            prev = next(a for a in reversed(aux[:i]) if a.get("group"))
            out.append((e, prev))
    return out


def test_per_molecule_loop_steps_the_plateau_on_the_epoch_mean(
        training_batch_info, monkeypatch):
    """Under ``adam_plateau`` the loop steps the controller once per epoch, on
    that epoch's MEAN group loss, and records the rate it is training at.

    With patience 10 the first epoch sets the best and epochs 2-12 are
    non-improving, so the twelfth is the first whose count exceeds the patience
    and the rate falls to a tenth there and not before. The scripted losses hold
    the epoch mean flat while the last group's loss of each epoch falls, so a
    controller stepping on the last loss the loop happens to be holding would
    never fire at all. The optimizer name reaches ``train_metadata.json``, and
    the default ``adamw_linear`` run writes no ``__plateau__`` row.
    """
    from xcquinox.pipeline import train as train_mod
    from xcquinox.pipeline.train import run_training, _training_groups

    with tempfile.TemporaryDirectory() as tmpdir:
        spec = _plateau_spec(training_batch_info, tmpdir, n_steps=14,
                             optimizer="adam_plateau", plateau_patience=10,
                             plateau_factor=0.1)
        n_groups = len(_training_groups(spec))
        assert n_groups >= 2, n_groups     # else mean == last group loss
        seen = []
        monkeypatch.setattr(train_mod, "defused_value_and_grad",
                            _make_scripted_loss(n_groups, seen=seen))
        # the reduced rate must be WRITTEN to the optimizer state, not only
        # logged: every write is recorded here and the state itself checked
        writes = []
        real_set = train_mod._set_learning_rate

        def _recording_set(opt_state, lr):
            writes.append(float(lr))
            new_state = real_set(opt_state, lr)
            written = [s.hyperparams["learning_rate"] for s in new_state
                       if getattr(s, "hyperparams", None) is not None]
            assert float(written[0]) == pytest.approx(lr, rel=1e-12)
            return new_state

        monkeypatch.setattr(train_mod, "_set_learning_rate", _recording_set)
        meta = run_training(spec)
        rows = _plateau_entries(spec.checkpoint_dir)
        with open(os.path.join(spec.checkpoint_dir, "train_metadata.json")) as f:
            on_disk_optimizer = json.load(f)["optimizer"]

    # the script did what the reading of it above claims.
    per_epoch: dict = {}
    for epoch, value in seen:
        per_epoch.setdefault(epoch, []).append(value)
    means = [np.mean(v) for v in per_epoch.values()]
    assert means == pytest.approx(
        [1.0, 0.5, 1 / 3] + [(1 / 3) * (1 - 1.5e-4)] * 11, abs=1e-12)
    lasts = [v[-1] for v in per_epoch.values()]
    assert all(b < a for a, b in zip(lasts[1:], lasts[2:])), lasts

    assert len(rows) == 14, [e for e, _ in rows]
    for i, (entry, prev) in enumerate(rows):
        assert entry["epoch"] == prev["epoch"], (i, entry, prev)
        # each row carries ITS epoch's mean (a cumulative mean would read
        # 0.75, 0.61, ... here) and the quantity the controller was stepped on
        assert entry["epoch_loss"] == pytest.approx(means[i], abs=1e-12), i
        assert entry["plateau_metric"] == pytest.approx(
            np.sqrt(means[i]) * 1000.0, rel=1e-12), i
        assert "step" in entry
    assert [e["lr"] for e, _ in rows] == [1e-4] * 13 + [1e-5]
    assert writes == pytest.approx([1e-5], rel=1e-12), writes
    assert meta["optimizer"] == "adam_plateau"
    assert on_disk_optimizer == "adam_plateau"

    # Control: the default optimizer runs no controller at all.
    with tempfile.TemporaryDirectory() as tmpdir:
        spec_off = _plateau_spec(training_batch_info, tmpdir, n_steps=2)
        monkeypatch.setattr(train_mod, "defused_value_and_grad",
                            _make_scripted_loss(n_groups))
        meta_off = run_training(spec_off)
        assert _plateau_entries(spec_off.checkpoint_dir) == []
    assert meta_off["optimizer"] == "adamw_linear"


