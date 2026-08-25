"""The model-class record beside a TRAINED checkpoint
(:mod:`xcquinox.alec.checkpoint_class`), its writers in the training stage and
its readers in the two evaluation loaders and the resume path.

The property under test is the one the leaf stream cannot state: the parent
anchor and the descriptor coordinates are static fields with no parameters of
their own, so a checkpoint written by one model class deserialises into
another class's skeleton silently and evaluates as a model that is neither.
The first case below measures exactly that -- a cross-class
``tree_deserialise_leaves`` succeeding -- so the refusals that follow are held
against a real hazard rather than an assumed one.

Costs: no PySCF. The evaluation entry point is reached with a valid spec and
raises before any precompute; the acceptance leg is observed with a sentinel
raised at the step after the check.
"""
import dataclasses
import json
import os

import equinox as eqx
import jax.tree_util as jtu
import numpy as np
import pytest

from xcquinox.alec.checkpoint_class import (CLASS_RECORD_SUFFIX,
                                            class_record_path,
                                            model_class_of_arch,
                                            model_class_of_model,
                                            read_class_record,
                                            require_matching_class,
                                            write_class_record)
from xcquinox.alec import config as alec_config
from xcquinox.alec.config import ArchitectureConfig, TrainingSpec, anchored
from xcquinox.alec.models import AlecGGAModel
from xcquinox.alec.tests.fixtures.molecules import h_atom, h2o_molecule, o_atom


# ---------------------------------------------------------------------------
# Architectures: three model classes with IDENTICAL parameter shapes
# ---------------------------------------------------------------------------

def _base_arch(**overrides):
    """A small architecture with polarized correlation, the shape the anchor
    needs (the anchored correlation parent divides by the polarized PW92
    baseline, so a zeta-blind anchored net is refused at construction). Every
    class below is this one with static fields changed, so all three have the
    same leaves."""
    defaults = dict(
        name="t", depth=2, nodes=8, attention=False,
        descriptors=(), x_constraints=(), c_constraints=(),
        double_lob_clamp_allowed=False, use_polarized_correlation=True,
    )
    defaults.update(overrides)
    return ArchitectureConfig(**defaults)


def _legacy_arch():
    """Unanchored, legacy coordinates: what every run before the anchor was."""
    return _base_arch()


def _anchored_arch():
    """Anchored, legacy coordinates."""
    return anchored(_base_arch())


def _anchored_dfs_arch():
    """Anchored, DFS coordinates: differs from the above in the coordinate
    set alone, which changes no width either."""
    return dataclasses.replace(anchored(_base_arch()),
                               descriptor_coordinates="dfs")


def _dfs_arch():
    """DFS coordinates with NO anchor: the fourth reachable class, and the one
    the two compared fields separate on their own."""
    return _base_arch(descriptor_coordinates="dfs")


def _model(arch, seed=0):
    return AlecGGAModel.from_arch(arch, seed=seed)


def _write_checkpoint(path, arch, *, record=True, seed=0):
    """A trained checkpoint of ``arch``'s class, with or without its record."""
    model = _model(arch, seed=seed)
    if record:
        write_class_record(path, arch)
    eqx.tree_serialise_leaves(path, model)
    return model


def _arrays(model):
    return [np.asarray(x) for x in jtu.tree_leaves(eqx.filter(model, eqx.is_array))]


# ---------------------------------------------------------------------------
# The hazard the record exists for
# ---------------------------------------------------------------------------

def test_a_checkpoint_of_another_class_deserialises_without_complaint(tmp_path):
    """The premise: nothing in the leaf stream distinguishes the classes.

    An anchored DFS checkpoint loads into an unanchored legacy skeleton, and
    the reverse, with every array equal -- so the model that comes out is the
    other class's weights read as this class's, and no shape, dtype or count
    objects. This is why the record is written and compared; it is not a
    belt-and-braces check over something equinox would catch.
    """
    path = str(tmp_path / "model.eqx")
    written = _write_checkpoint(path, _anchored_dfs_arch(), record=False)
    loaded = eqx.tree_deserialise_leaves(path, _model(_legacy_arch(), seed=7))
    a, b = _arrays(written), _arrays(loaded)
    assert len(a) == len(b) and a
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    # ... and the skeleton it landed in still says it is the legacy class.
    assert model_class_of_model(loaded) == {"parent_anchor": False,
                                            "descriptor_coordinates": "legacy"}


# ---------------------------------------------------------------------------
# The writer: every trained checkpoint carries its class
# ---------------------------------------------------------------------------

def _make_training_spec(arch, checkpoint_dir):
    return TrainingSpec(
        arch=arch, molecules=(h_atom(), o_atom(), h2o_molecule()),
        targets=(("H", -0.5), ("H2O", 0.3), ("O", -74.8)),
        atom_energies=(("H", -0.5), ("O", -74.8)),
        loss_name="A_atomization", n_steps=3, lr_start=1e-3, lr_end=1e-5,
        lr_decay_start=0.0, grad_clip=1.0, checkpoint_dir=str(checkpoint_dir),
        seed=42)


def test_save_artifacts_records_the_class_of_every_checkpoint_it_writes(tmp_path):
    """The final, best-loss and validation-best checkpoints each get the
    record, with the class the spec's arch states and the provenance keys the
    neighbouring records use."""
    from xcquinox.alec.cluster.materialize import running_xcquinox_version
    from xcquinox.alec.train import _save_artifacts

    arch = _anchored_dfs_arch()
    spec = _make_training_spec(arch, tmp_path)
    model = _model(arch)
    _save_artifacts(spec, model, [0.5, 0.4], [], 1.0,
                    best_model=_model(arch, seed=1),
                    val_best_model=_model(arch, seed=2))

    for name in ("model.eqx", "model_best.eqx", "model_val_best.eqx"):
        ckpt = os.path.join(str(tmp_path), name)
        assert os.path.isfile(ckpt), name
        record = read_class_record(ckpt)
        assert record is not None, name
        assert record["parent_anchor"] is True
        assert record["descriptor_coordinates"] == "dfs"
        assert record["parent"] == "pbe"
        assert record["arch_name"] == "t"
        assert record["meta_gga"] is False
        assert record["use_polarized_correlation"] is True
        assert record["xcquinox_version"] == running_xcquinox_version()


def test_an_unanchored_run_records_the_legacy_class(tmp_path):
    """The record is written for every class, not only the anchored one: a
    legacy run states its class rather than leaving the file unrecorded, and
    names no parent."""
    from xcquinox.alec.train import _save_artifacts

    arch = _legacy_arch()
    spec = _make_training_spec(arch, tmp_path)
    _save_artifacts(spec, _model(arch), [0.5], [], 1.0)

    record = read_class_record(os.path.join(str(tmp_path), "model.eqx"))
    assert record["parent_anchor"] is False
    assert record["descriptor_coordinates"] == "legacy"
    assert record["parent"] is None


def test_the_record_sits_at_the_checkpoints_own_path(tmp_path):
    """The record's path follows from the checkpoint's, which is what lets a
    reader holding one FILE (``TestSpec.model_checkpoint``, the eval task's
    choice among model.eqx / model_best.eqx / model_val_best.eqx) find it with
    no knowledge of the run layout, and what keeps the two together when a
    single checkpoint is copied out on its own."""
    ckpt = str(tmp_path / "model_val_best.eqx")
    assert class_record_path(ckpt) == ckpt + CLASS_RECORD_SUFFIX
    write_class_record(ckpt, _anchored_arch())
    assert os.path.isfile(str(tmp_path / "model_val_best.eqx.class.json"))
    with open(str(tmp_path / "model_val_best.eqx.class.json")) as f:
        assert json.load(f)["parent_anchor"] is True


# ---------------------------------------------------------------------------
# The readers: evaluation.run_test
# ---------------------------------------------------------------------------

class _ReachedTheEval(Exception):
    """Raised at the step AFTER the class check, to observe acceptance
    without running an evaluation."""


def _make_test_spec(arch, checkpoint, output_dir):
    # ``config.TestSpec`` by attribute: the bare name would be collected as a
    # test class by pytest and warned about.
    return alec_config.TestSpec(
        model_checkpoint=str(checkpoint), arch=arch,
        molecules=(h_atom(), o_atom(), h2o_molecule()),
        metrics=("total_energy",), output_dir=str(output_dir))


@pytest.mark.parametrize("written,wanted,names", [
    (_anchored_dfs_arch, _legacy_arch,
     ("parent_anchor=True", "parent_anchor=False")),
    (_legacy_arch, _anchored_dfs_arch,
     ("parent_anchor=False", "parent_anchor=True")),
    (_anchored_dfs_arch, _anchored_arch,
     ("descriptor_coordinates='dfs'", "descriptor_coordinates='legacy'")),
])
def test_run_test_refuses_a_checkpoint_of_another_class(tmp_path, written,
                                                        wanted, names):
    """Both directions of the anchor, and the coordinates on their own: the
    refusal names the class the checkpoint was written as AND the class being
    built."""
    from xcquinox.alec.evaluation import run_test

    ckpt = tmp_path / "model.eqx"
    _write_checkpoint(str(ckpt), written())
    spec = _make_test_spec(wanted(), ckpt, tmp_path / "out")
    with pytest.raises(ValueError) as excinfo:
        run_test(spec)
    message = str(excinfo.value)
    assert "refusing to load" in message
    for fragment in names:
        assert fragment in message, message


def test_run_test_refuses_an_unrecorded_checkpoint_into_an_anchored_model(tmp_path):
    """No record means the legacy class, because every anchored or dfs run
    writes one. An anchored model is therefore refused with the reason stated
    -- the checkpoint carries no class record -- rather than loading weights
    nothing describes."""
    from xcquinox.alec.evaluation import run_test

    ckpt = tmp_path / "model.eqx"
    _write_checkpoint(str(ckpt), _legacy_arch(), record=False)
    spec = _make_test_spec(_anchored_arch(), ckpt, tmp_path / "out")
    with pytest.raises(ValueError, match="no model-class record"):
        run_test(spec)


def test_run_test_accepts_an_unrecorded_checkpoint_into_a_legacy_model(
        tmp_path, monkeypatch):
    """The other side of the same rule: every checkpoint written before the
    record existed still loads into the class that wrote it. Observed at the
    step after the load (metric construction), so no evaluation runs."""
    import xcquinox.alec.evaluation as evaluation

    ckpt = tmp_path / "model.eqx"
    _write_checkpoint(str(ckpt), _legacy_arch(), record=False)
    spec = _make_test_spec(_legacy_arch(), ckpt, tmp_path / "out")

    def _stop(*args, **kwargs):
        raise _ReachedTheEval

    monkeypatch.setattr(evaluation, "make_metric", _stop)
    with pytest.raises(_ReachedTheEval):
        evaluation.run_test(spec)


# ---------------------------------------------------------------------------
# The readers: eval_holdout.load_trained_model
# ---------------------------------------------------------------------------

class _SpecStub:
    def __init__(self, arch):
        self.arch = arch


@pytest.mark.parametrize("written,wanted", [
    (_anchored_dfs_arch, _legacy_arch),
    (_legacy_arch, _anchored_dfs_arch),
    (_anchored_arch, _anchored_dfs_arch),
])
def test_load_trained_model_refuses_a_checkpoint_of_another_class(
        tmp_path, written, wanted):
    """The held-out loader (the cluster eval task's and the cold-start
    channel's single entry point) holds the same rule."""
    from xcquinox.alec.eval_holdout import load_trained_model

    ckpt = tmp_path / "model.eqx"
    _write_checkpoint(str(ckpt), written())
    with pytest.raises(ValueError, match="different model classes"):
        load_trained_model(_SpecStub(wanted()), ckpt)


def test_load_trained_model_and_the_unrecorded_checkpoint(tmp_path):
    """Unrecorded: refused by an anchored skeleton, accepted by a legacy one
    (which loads and returns the model)."""
    from xcquinox.alec.eval_holdout import load_trained_model

    ckpt = tmp_path / "model.eqx"
    written = _write_checkpoint(str(ckpt), _legacy_arch(), record=False)
    with pytest.raises(ValueError, match="no model-class record"):
        load_trained_model(_SpecStub(_anchored_arch()), ckpt)

    loaded = load_trained_model(_SpecStub(_legacy_arch()), ckpt)
    a, b = _arrays(written), _arrays(loaded)
    assert a and all(np.array_equal(x, y) for x, y in zip(a, b))


def test_load_trained_model_accepts_the_class_it_was_written_as(tmp_path):
    """The matching case runs: an anchored dfs checkpoint into an anchored dfs
    skeleton returns the model, so the refusals above are not simply refusing
    everything."""
    from xcquinox.alec.eval_holdout import load_trained_model

    ckpt = tmp_path / "model.eqx"
    _write_checkpoint(str(ckpt), _anchored_dfs_arch())
    loaded = load_trained_model(_SpecStub(_anchored_dfs_arch()), ckpt)
    assert model_class_of_model(loaded) == {"parent_anchor": True,
                                            "descriptor_coordinates": "dfs"}


# ---------------------------------------------------------------------------
# The readers: the resume path
# ---------------------------------------------------------------------------

def _write_resume_set(checkpoint_dir, arch, *, with_arch=True):
    """One periodic resume checkpoint of ``arch``'s class."""
    from xcquinox.alec.train import _write_resume_checkpoint, build_optimizer

    model = _model(arch)
    optimizer = build_optimizer(lr_start=1e-3, lr_end=1e-5, n_steps=10,
                                lr_decay_start=0.0, grad_clip=1.0)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    _write_resume_checkpoint(
        str(checkpoint_dir), model=model, opt_state=opt_state,
        rng_state=np.random.RandomState(0).get_state(), order=[0, 1],
        train_best_loss=0.5, train_recent=[0.5], train_window=2,
        train_best_model=_model(arch, seed=1), val_present=True,
        val_best_mae=1.0, val_finite_metrics=[1.0],
        val_best_model=_model(arch, seed=2), epoch=1, update=2,
        losses=[0.5], aux_log=[], early_stopped=False,
        arch=(arch if with_arch else None))
    return model, optimizer


def test_the_resume_set_carries_a_record_for_every_model_it_writes(tmp_path):
    """The periodic snapshots are trained checkpoints too. Each model file
    gets a record; the optimizer state, which is not a model, does not."""
    _write_resume_set(tmp_path, _anchored_dfs_arch())
    for name in ("resume_model.eqx", "resume_best.eqx", "resume_val_best.eqx"):
        record = read_class_record(os.path.join(str(tmp_path), name))
        assert record is not None, name
        assert record["parent_anchor"] is True
        assert record["descriptor_coordinates"] == "dfs"
    assert not os.path.isfile(
        os.path.join(str(tmp_path), "resume_opt_state.eqx" + CLASS_RECORD_SUFFIX))


def test_the_resume_loader_refuses_a_snapshot_of_another_class(tmp_path):
    """A run whose configuration changed class between the kill and the
    restart must not resume from the other class's weights. The loader holds
    each snapshot to the class of the skeleton it is about to fill, read off
    the skeleton's own static fields."""
    from xcquinox.alec.train import _load_resume_checkpoint

    _model_written, optimizer = _write_resume_set(tmp_path, _anchored_dfs_arch())
    skeleton = _model(_legacy_arch(), seed=9)
    opt_skeleton = optimizer.init(eqx.filter(skeleton, eqx.is_array))
    with pytest.raises(ValueError) as excinfo:
        _load_resume_checkpoint(str(tmp_path), model_skeleton=skeleton,
                                opt_state_skeleton=opt_skeleton)
    message = str(excinfo.value)
    assert "resume checkpoint" in message
    assert "descriptor_coordinates='dfs'" in message
    assert "descriptor_coordinates='legacy'" in message


def test_the_resume_loader_accepts_its_own_class(tmp_path):
    """The matching case restores, so the refusal above is a class check and
    not a broken loader."""
    from xcquinox.alec.train import _load_resume_checkpoint

    written, optimizer = _write_resume_set(tmp_path, _anchored_dfs_arch())
    skeleton = _model(_anchored_dfs_arch(), seed=9)
    opt_skeleton = optimizer.init(eqx.filter(skeleton, eqx.is_array))
    out = _load_resume_checkpoint(str(tmp_path), model_skeleton=skeleton,
                                  opt_state_skeleton=opt_skeleton)
    a, b = _arrays(written), _arrays(out["model"])
    assert a and all(np.array_equal(x, y) for x, y in zip(a, b))
    assert out["epoch"] == 1 and out["update"] == 2


def test_an_unrecorded_resume_set_is_readable_only_by_the_legacy_class(tmp_path):
    """The rule applies to the resume path as it does to the evaluation
    loaders: a set written with no class stated (a legacy caller) loads into a
    legacy skeleton and is refused by an anchored one."""
    from xcquinox.alec.train import _load_resume_checkpoint

    _written, optimizer = _write_resume_set(tmp_path, _legacy_arch(),
                                            with_arch=False)
    assert read_class_record(
        os.path.join(str(tmp_path), "resume_model.eqx")) is None

    legacy_skeleton = _model(_legacy_arch(), seed=9)
    opt_skeleton = optimizer.init(eqx.filter(legacy_skeleton, eqx.is_array))
    out = _load_resume_checkpoint(str(tmp_path), model_skeleton=legacy_skeleton,
                                  opt_state_skeleton=opt_skeleton)
    assert out["epoch"] == 1

    anchored_skeleton = _model(_anchored_arch(), seed=9)
    with pytest.raises(ValueError, match="no model-class record"):
        _load_resume_checkpoint(str(tmp_path),
                                model_skeleton=anchored_skeleton,
                                opt_state_skeleton=opt_skeleton)


class _KilledInTheRename(Exception):
    """Stands for a kill inside the atomic write, at the instant the leaves
    have been written to their temporary sibling and the rename that puts them
    in place has not returned."""


def _kill_the_rename_of(monkeypatch, basename):
    """Make ``os.replace`` raise :class:`_KilledInTheRename` when it is asked to
    move something onto ``basename``, and behave normally otherwise. The atomic
    writer's leaf rename is then the only step that dies."""
    real_replace = os.replace

    def _replace(src, dst, *args, **kwargs):
        if os.path.basename(os.fspath(dst)) == basename:
            raise _KilledInTheRename(dst)
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "replace", _replace)


def test_a_killed_atomic_write_leaves_the_record_describing_the_leaves(
        tmp_path, monkeypatch):
    """The atomic writer's two files change together or not at all.

    Unlike the truncating write, an aborted atomic write leaves the PREVIOUS
    ``.eqx`` in place. A record put down before the rename would therefore come
    to stand beside the old class's leaves -- a checkpoint labelled as a class
    it is not, which every reader then accepts. After a kill at the rename the
    record beside the surviving checkpoint must still be the surviving
    checkpoint's own, or absent (absent is the legacy class, which an anchored
    skeleton refuses).
    """
    from xcquinox.alec.train import _serialise_trained_model

    _write_resume_set(tmp_path, _legacy_arch())
    ckpt = os.path.join(str(tmp_path), "resume_model.eqx")
    with open(ckpt, "rb") as f:
        leaves_before = f.read()
    assert read_class_record(ckpt)["descriptor_coordinates"] == "legacy"

    _kill_the_rename_of(monkeypatch, "resume_model.eqx")
    other = _anchored_dfs_arch()
    with pytest.raises(_KilledInTheRename):
        _serialise_trained_model(ckpt, _model(other, seed=3), other, atomic=True)
    monkeypatch.undo()

    with open(ckpt, "rb") as f:
        assert f.read() == leaves_before, "the previous leaves did not survive"
    record = read_class_record(ckpt)
    assert record is None or (record["parent_anchor"] is False
                              and record["descriptor_coordinates"] == "legacy"), (
        "the record was renamed into place while the leaves beside it are "
        f"still the previous class's: {record}")


def test_the_non_atomic_write_still_puts_the_record_down_first(tmp_path,
                                                               monkeypatch):
    """The other half of the same invariant, pinned so the two branches are
    not unified into one order by mistake.

    ``eqx.tree_serialise_leaves`` truncates its target, so its leaves cannot
    survive a kill and the record may safely precede them. The order the
    atomic branch needs would, here, leave a COMPLETE checkpoint of this class
    with no record beside it -- which a legacy skeleton accepts and reads as
    legacy, the load this module exists to refuse.
    """
    from xcquinox.alec import train

    ckpt = str(tmp_path / "model.eqx")
    arch = _anchored_dfs_arch()

    def _die(*args, **kwargs):
        raise _KilledInTheRename(ckpt)

    monkeypatch.setattr(train.eqx, "tree_serialise_leaves", _die)
    with pytest.raises(_KilledInTheRename):
        train._serialise_trained_model(ckpt, _model(arch), arch)
    monkeypatch.undo()

    assert not os.path.isfile(ckpt), "the leaves were written after all"
    assert read_class_record(ckpt)["descriptor_coordinates"] == "dfs"


def test_a_refused_set_then_a_killed_write_does_not_resume_the_other_class(
        tmp_path, monkeypatch):
    """The sequence the record exists to make impossible, end to end.

    A run of class A leaves its resume set behind; a restart configured as
    class B is refused, as it must be; class B's first periodic checkpoint is
    killed inside the atomic write; class B starts again. Whatever the last
    step loads, it must not be class A's weights -- either the set is gone and
    the run starts fresh, or the load is refused. Resuming from them is the
    silent cross-class load this module exists to prevent.
    """
    from xcquinox.alec.train import (_has_resume_checkpoint,
                                     _load_resume_checkpoint,
                                     _write_resume_checkpoint)

    class_a_arch, class_b = _legacy_arch(), _anchored_dfs_arch()
    _model_a, optimizer = _write_resume_set(tmp_path, class_a_arch)
    class_a_leaves = [_arrays(_model(class_a_arch, seed=s)) for s in (0, 1, 2)]
    skeleton = _model(class_b, seed=9)
    opt_skeleton = optimizer.init(eqx.filter(skeleton, eqx.is_array))

    with pytest.raises(ValueError):
        _load_resume_checkpoint(str(tmp_path), model_skeleton=skeleton,
                                opt_state_skeleton=opt_skeleton)

    # The class B run's first periodic checkpoint, killed at the LAST model
    # file it writes: the snapshots before it are class B's throughout, and the
    # validation-best file is the one left holding class A's leaves.
    _kill_the_rename_of(monkeypatch, "resume_val_best.eqx")
    with pytest.raises(_KilledInTheRename):
        _write_resume_checkpoint(
            str(tmp_path), model=_model(class_b, seed=3), opt_state=opt_skeleton,
            rng_state=np.random.RandomState(0).get_state(), order=[0, 1],
            train_best_loss=0.4, train_recent=[0.4], train_window=2,
            train_best_model=_model(class_b, seed=4), val_present=True,
            val_best_mae=0.9, val_finite_metrics=[0.9],
            val_best_model=_model(class_b, seed=5), epoch=2, update=4,
            losses=[0.4], aux_log=[], early_stopped=False, arch=class_b)
    monkeypatch.undo()

    resumed = []
    if _has_resume_checkpoint(str(tmp_path)):
        try:
            out = _load_resume_checkpoint(str(tmp_path), model_skeleton=skeleton,
                                          opt_state_skeleton=opt_skeleton)
        except Exception:  # noqa: BLE001 -- refused: the fail-safe outcome
            pass
        else:
            resumed = [out["model"], out["train_tracker"].best_model,
                       out["val_tracker"].best_model]
    for loaded in resumed:
        if loaded is None:
            continue
        got = _arrays(loaded)
        for wrote in class_a_leaves:
            assert not all(np.array_equal(x, y) for x, y in zip(wrote, got)), (
                "the class B restart resumed from class A's weights")


def test_a_refused_resume_load_removes_the_stale_set_with_its_records(tmp_path):
    """A refused set is discarded, not left for a later write to re-label.

    The refusal is permanent -- the configuration's class will not change back
    within the run that was just refused -- and the run starts fresh, so the
    set is superseded either way. Leaving it is what makes the sequence above
    reachable: its records and its leaves can afterwards be updated
    independently. The message states the removal, so the caller's warning
    reports it.
    """
    from xcquinox.alec.train import _load_resume_checkpoint

    _written, optimizer = _write_resume_set(tmp_path, _anchored_dfs_arch())
    skeleton = _model(_legacy_arch(), seed=9)
    opt_skeleton = optimizer.init(eqx.filter(skeleton, eqx.is_array))

    with pytest.raises(ValueError) as excinfo:
        _load_resume_checkpoint(str(tmp_path), model_skeleton=skeleton,
                                opt_state_skeleton=opt_skeleton)
    assert "removed" in str(excinfo.value).lower(), str(excinfo.value)
    left = sorted(name for name in os.listdir(str(tmp_path))
                  if name.startswith("resume_"))
    assert left == [], left


def test_completion_deletes_the_resume_records_with_their_checkpoints(tmp_path):
    """No record outlives the checkpoint it describes: completion clears the
    resume set and its records together, so the next run in the same directory
    cannot read a class record belonging to a deleted snapshot."""
    from xcquinox.alec.train import _finalize_completion

    _write_resume_set(tmp_path, _anchored_dfs_arch())
    _finalize_completion(str(tmp_path), early_stopped=False, epochs_run=1)
    left = sorted(name for name in os.listdir(str(tmp_path))
                  if name.startswith("resume_"))
    assert left == []


# ---------------------------------------------------------------------------
# The record itself
# ---------------------------------------------------------------------------

def test_an_unreadable_record_is_refused_rather_than_read_as_legacy(tmp_path):
    """A record that cannot be parsed is not the same as no record: answering
    "legacy" for it would be a guess at the one thing the file states."""
    ckpt = tmp_path / "model.eqx"
    _write_checkpoint(str(ckpt), _anchored_arch(), record=False)
    with open(class_record_path(str(ckpt)), "w") as f:
        f.write("{not json")
    with pytest.raises(ValueError, match="could not be read"):
        require_matching_class(str(ckpt), model_class_of_arch(_anchored_arch()))


def test_the_class_of_an_arch_and_of_the_model_it_builds_agree():
    """``create_network_pair`` carries the configuration's class into the
    built networks' static fields, so the two readings the loaders use --
    from the spec's arch, and from the skeleton itself -- answer the same.

    All FOUR reachable classes: the anchor and the coordinates are independent
    fields, so unanchored ``dfs`` is a class of its own and is checked here
    with the other three."""
    classes = set()
    for arch in (_legacy_arch(), _anchored_arch(), _anchored_dfs_arch(),
                 _dfs_arch()):
        assert model_class_of_arch(arch) == model_class_of_model(_model(arch))
        classes.add(tuple(sorted(model_class_of_arch(arch).items())))
    assert len(classes) == 4, classes
