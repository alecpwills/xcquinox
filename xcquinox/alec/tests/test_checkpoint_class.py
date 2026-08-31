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

The record and the checkpoint are two files with one rename each, so the
second property under test is that the record DESCRIBES the leaves it stands
beside rather than merely arriving in a particular order: the kill-point cases
interrupt the writer at each boundary and hold every reader to the state that
is left. The digests the cases compare against are taken with ``hashlib``
here, not with the module's own helper, so the two are independent.

Costs: no PySCF. The evaluation entry point is reached with a valid spec and
raises before any precompute; the acceptance leg is observed with a sentinel
raised at the step after the check.
"""
import dataclasses
import hashlib
import json
import os
import time

import equinox as eqx
import jax.tree_util as jtu
import numpy as np
import pytest

from xcquinox.alec.checkpoint_class import (CLASS_RECORD_SUFFIX,
                                            TEMPORARY_GRACE_SECONDS,
                                            ClassRecordStale,
                                            class_record_path,
                                            load_trained_checkpoint,
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
    """A trained checkpoint of ``arch``'s class, with or without its record.

    The record is written for the leaves that are already on disk, which is
    what makes it describe them; the training stage's own writer stages it
    around the leaves' rename instead (``train._serialise_trained_model``).
    """
    model = _model(arch, seed=seed)
    eqx.tree_serialise_leaves(path, model)
    if record:
        write_class_record(path, arch)
    return model


def _sha256_of(path):
    """The digest of the file at ``path``, taken here rather than through the
    module under test."""
    with open(os.fspath(path), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


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
    no knowledge of the run layout.

    The pull the local re-evaluation workflow uses narrows per spec DIRECTORY
    (``cluster.sync.build_rsync_command``), so every record comes down beside
    its own checkpoint. A bare copy of the ``.eqx`` alone leaves the record
    behind and reads as a checkpoint with no record, which an anchored or dfs
    skeleton refuses and a LEGACY skeleton accepts and loads as legacy -- so
    the record must be copied with the checkpoint whenever one is moved
    outside that pull."""
    ckpt = str(tmp_path / "model_val_best.eqx")
    assert class_record_path(ckpt) == ckpt + CLASS_RECORD_SUFFIX
    _write_checkpoint(ckpt, _anchored_arch())
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
    """Stands for a kill inside the write, at the instant one of the two
    renames that commit a checkpoint and its record has not returned."""


def _kill_the_rename_onto(monkeypatch, basename):
    """Make ``os.replace`` raise :class:`_KilledInTheRename` when it is asked to
    move something onto ``basename``, and behave normally otherwise, so one
    named step of the write is the only one that dies."""
    real_replace = os.replace

    def _replace(src, dst, *args, **kwargs):
        if os.path.basename(os.fspath(dst)) == basename:
            raise _KilledInTheRename(dst)
        return real_replace(src, dst, *args, **kwargs)

    monkeypatch.setattr(os, "replace", _replace)


def _run_to_the_kill(fn, *args, **kwargs):
    """Call ``fn`` and return the :class:`_KilledInTheRename` it raised, or
    ``None`` if it ran to completion.

    The kill is not asserted here: what the case is about is the state left on
    disk, and a writer that reached further than the kill point says so more
    plainly through that state than through a missing exception.
    """
    try:
        fn(*args, **kwargs)
    except _KilledInTheRename as exc:
        return exc
    return None


def test_a_kill_between_the_record_and_the_leaves_is_refused_by_every_reader(
        tmp_path, monkeypatch):
    """Kill point "4 to 5" of the writer's table: the record has been renamed
    into place and the leaves have not.

    What stands on disk is then the NEW record over the PREVIOUS run's
    complete ``.eqx``. That is the state a record-first write left ACCEPTED --
    an anchored record over a legacy run's weights, read as anchored by both
    evaluation loaders and by the resume path, in silence. The record carries
    the digest of the leaves it was written for, so here every reader refuses
    it, and refuses it whichever class its own skeleton is: a record that does
    not describe these leaves is not evidence about them.
    """
    from xcquinox.alec.eval_holdout import load_trained_model
    from xcquinox.alec.evaluation import run_test
    from xcquinox.alec.train import (_load_resume_checkpoint,
                                     _serialise_trained_model)

    legacy, other = _legacy_arch(), _anchored_dfs_arch()
    _written, optimizer = _write_resume_set(tmp_path, legacy)
    ckpt = os.path.join(str(tmp_path), "resume_model.eqx")
    with open(ckpt, "rb") as f:
        leaves_before = f.read()

    _kill_the_rename_onto(monkeypatch, "resume_model.eqx")
    killed = _run_to_the_kill(_serialise_trained_model, ckpt,
                              _model(other, seed=3), other)
    monkeypatch.undo()

    with open(ckpt, "rb") as f:
        assert f.read() == leaves_before, (
            "the write reached the leaves: the previous run's checkpoint was "
            "overwritten by a write that was killed before its rename")
    assert killed is not None, "the leaves' rename was never reached"
    record = read_class_record(ckpt)
    assert record["descriptor_coordinates"] == "dfs", record
    assert record["sha256"] != _sha256_of(ckpt), (
        "the crossing state was not built: the record describes these leaves")

    for skeleton_arch in (legacy, other):
        with pytest.raises(ValueError) as excinfo:
            load_trained_model(_SpecStub(skeleton_arch), ckpt)
        assert isinstance(excinfo.value, ClassRecordStale), excinfo.value
        assert _sha256_of(ckpt) in str(excinfo.value)

        spec = _make_test_spec(skeleton_arch, ckpt, tmp_path / "out")
        with pytest.raises(ValueError) as excinfo:
            run_test(spec)
        assert isinstance(excinfo.value, ClassRecordStale), excinfo.value

        skeleton = _model(skeleton_arch, seed=9)
        opt_skeleton = optimizer.init(eqx.filter(skeleton, eqx.is_array))
        with pytest.raises(ValueError) as excinfo:
            _load_resume_checkpoint(str(tmp_path), model_skeleton=skeleton,
                                    opt_state_skeleton=opt_skeleton)
        assert isinstance(excinfo.value, ClassRecordStale), excinfo.value


def test_a_record_committed_over_no_checkpoint_is_refused(tmp_path,
                                                          monkeypatch):
    """The same kill point in an EMPTY directory, the first write a run makes:
    the record lands and the leaves do not, so the record describes a
    checkpoint that is not there.

    A record outliving -- or here outrunning -- its ``.eqx`` states the class
    of leaves nothing on disk holds, so it is refused rather than applied to
    whatever is written next. ``model.eqx`` is the harness's completion
    signal, and it is the rename that never happened, so the run reads as
    unfinished, which it is.
    """
    from xcquinox.alec.train import _serialise_trained_model

    ckpt = str(tmp_path / "model.eqx")
    arch = _anchored_dfs_arch()
    _kill_the_rename_onto(monkeypatch, "model.eqx")
    killed = _run_to_the_kill(_serialise_trained_model, ckpt, _model(arch), arch)
    monkeypatch.undo()

    assert not os.path.isfile(ckpt), (
        "the leaves landed: this is not the state a kill at the rename leaves")
    assert killed is not None, "the leaves' rename was never reached"
    assert _temporaries_in(tmp_path) == [], "the write left a temporary behind"
    assert read_class_record(ckpt)["descriptor_coordinates"] == "dfs"

    with pytest.raises(ValueError) as excinfo:
        require_matching_class(ckpt, model_class_of_arch(arch))
    assert isinstance(excinfo.value, ClassRecordStale), excinfo.value
    assert "model.eqx" in str(excinfo.value)


def test_a_kill_before_the_record_commit_leaves_the_previous_checkpoint(
        tmp_path, monkeypatch):
    """Kill points 1 to 3 of the same table: neither file has been renamed, so
    the directory is exactly as the write found it.

    Found here as an unrecorded LEGACY checkpoint -- what every run before the
    anchor left -- and the rule for one of those is unchanged by anything the
    digest adds: the legacy skeleton loads those leaves, and the anchored
    skeleton is refused because nothing on disk states what they are.
    """
    from xcquinox.alec.eval_holdout import load_trained_model
    from xcquinox.alec.train import _serialise_trained_model

    ckpt = str(tmp_path / "model.eqx")
    legacy_model = _write_checkpoint(ckpt, _legacy_arch(), record=False)
    with open(ckpt, "rb") as f:
        leaves_before = f.read()

    _kill_the_rename_onto(monkeypatch, "model.eqx" + CLASS_RECORD_SUFFIX)
    other = _anchored_dfs_arch()
    killed = _run_to_the_kill(_serialise_trained_model, ckpt,
                              _model(other, seed=3), other)
    monkeypatch.undo()

    with open(ckpt, "rb") as f:
        assert f.read() == leaves_before, (
            "the write reached the leaves after its record was killed")
    assert killed is not None, "the record's rename was never reached"
    assert read_class_record(ckpt) is None, (
        "a record was left behind by a write that committed neither file")
    assert _temporaries_in(tmp_path) == [], "the write left a temporary behind"

    loaded = load_trained_model(_SpecStub(_legacy_arch()), ckpt)
    a, b = _arrays(legacy_model), _arrays(loaded)
    assert a and all(np.array_equal(x, y) for x, y in zip(a, b))
    with pytest.raises(ValueError, match="no model-class record"):
        load_trained_model(_SpecStub(_anchored_arch()), ckpt)


def test_a_kill_before_the_leaves_are_serialised_never_commits_the_record(
        tmp_path, monkeypatch):
    """Kill point 1, with the serialisation itself raising: the record on disk
    is the previous one, or none -- never the new one.

    This is the window a record-first write opened at its widest. The
    truncation the earlier order relied on happens when
    ``tree_serialise_leaves`` OPENS its target, so a failure of that open
    (EACCES, EMFILE, a read-only remount) or a kill in the interval before it
    left the previous complete ``.eqx`` under the new record. Here the leaves
    are the first thing written and they are written to a temporary, so a
    failure there has touched neither file the readers look at.
    """
    from xcquinox.alec import train
    from xcquinox.alec.eval_holdout import load_trained_model

    ckpt = str(tmp_path / "model.eqx")
    legacy_model = _write_checkpoint(ckpt, _legacy_arch())
    with open(ckpt, "rb") as f:
        leaves_before = f.read()

    def _die(*args, **kwargs):
        raise _KilledInTheRename(ckpt)

    monkeypatch.setattr(train.eqx, "tree_serialise_leaves", _die)
    other = _anchored_dfs_arch()
    with pytest.raises(_KilledInTheRename):
        train._serialise_trained_model(ckpt, _model(other, seed=3), other)
    monkeypatch.undo()

    record = read_class_record(ckpt)
    assert record["descriptor_coordinates"] == "legacy", (
        "the new class's record was committed over the previous run's "
        f"checkpoint: {record}")
    with open(ckpt, "rb") as f:
        assert f.read() == leaves_before
    assert record["sha256"] == _sha256_of(ckpt)

    loaded = load_trained_model(_SpecStub(_legacy_arch()), ckpt)
    a, b = _arrays(legacy_model), _arrays(loaded)
    assert a and all(np.array_equal(x, y) for x, y in zip(a, b))


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
    _kill_the_rename_onto(monkeypatch, "resume_val_best.eqx")
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


def test_the_resume_loader_refuses_a_record_that_describes_other_leaves(tmp_path):
    """The crossing state, built on disk rather than by interrupting a writer:
    one run's record over another run's leaves, with ``resume_state.pkl``
    present so the set is offered to the loader.

    ``_has_resume_checkpoint`` requires that state pickle, which is written
    after every snapshot it names, so a HALF-WRITTEN set is never read at all.
    That gate says nothing about the state here -- a set that was complete
    when it was written and has since had one of its files replaced by another
    run's -- and the class comparison says nothing about it either, since the
    record it reads is the one the skeleton's own class wrote. What refuses it
    is the digest.
    """
    from xcquinox.alec.train import (_has_resume_checkpoint,
                                     _load_resume_checkpoint)

    _written, optimizer = _write_resume_set(tmp_path, _legacy_arch())
    ckpt = os.path.join(str(tmp_path), "resume_model.eqx")
    recorded = read_class_record(ckpt)
    eqx.tree_serialise_leaves(ckpt, _model(_anchored_dfs_arch(), seed=3))
    assert _has_resume_checkpoint(str(tmp_path)), "the state gate is not set"
    assert recorded["descriptor_coordinates"] == "legacy"

    skeleton = _model(_legacy_arch(), seed=9)
    opt_skeleton = optimizer.init(eqx.filter(skeleton, eqx.is_array))
    with pytest.raises(ValueError) as excinfo:
        _load_resume_checkpoint(str(tmp_path), model_skeleton=skeleton,
                                opt_state_skeleton=opt_skeleton)
    assert isinstance(excinfo.value, ClassRecordStale), excinfo.value
    assert _sha256_of(ckpt) in str(excinfo.value)


# ---------------------------------------------------------------------------
# The write temporaries: one name per write, not one per checkpoint
# ---------------------------------------------------------------------------

def _temporaries_in(directory):
    """Every temporary sibling left in ``directory``, whatever drew it."""
    return sorted(name for name in os.listdir(os.fspath(directory))
                  if name.endswith(".tmp"))


def test_two_writers_of_one_checkpoint_do_not_share_a_temporary(tmp_path):
    """Two writes of the same checkpoint, one nested inside the other, both
    complete and the pair left on disk is consistent.

    The state is reachable by operator action: ``cluster.__main__.cmd_resubmit``
    re-submits an index classified ``no_evidence`` into the SAME run directory
    without establishing that the earlier task has stopped, so one spec
    directory can hold two live writers of ``model.eqx``. Under one temporary
    name per checkpoint the second writer serialises INTO the first writer's
    half-written file and renames it away, and the first writer then fails on
    a file that is no longer there. Here each write draws its own name
    (``checkpoint_class.new_temporary``), so the writer that renames last puts
    down its own leaves under its own record.

    The inner write runs immediately after the outer one has serialised its
    leaves, which is the widest window between a writer's temporary and its
    own rename.
    """
    from xcquinox.alec import train

    ckpt = str(tmp_path / "model.eqx")
    outer_arch, inner_arch = _anchored_dfs_arch(), _legacy_arch()
    outer_model, inner_model = _model(outer_arch, seed=3), _model(inner_arch, seed=8)

    real_serialise = eqx.tree_serialise_leaves
    nested = []

    def _serialise(path, pytree, *args, **kwargs):
        out = real_serialise(path, pytree, *args, **kwargs)
        if not nested:
            nested.append(True)
            # The whole second write, between the first writer's serialise and
            # its own rename.
            train._serialise_trained_model(ckpt, inner_model, inner_arch)
        return out

    monkeypatched = pytest.MonkeyPatch()
    monkeypatched.setattr(train.eqx, "tree_serialise_leaves", _serialise)
    try:
        train._serialise_trained_model(ckpt, outer_model, outer_arch)
    finally:
        monkeypatched.undo()
    assert nested, "the second write never ran"

    assert sorted(os.listdir(str(tmp_path))) == [
        "model.eqx", "model.eqx" + CLASS_RECORD_SUFFIX], os.listdir(str(tmp_path))
    record = read_class_record(ckpt)
    assert record["sha256"] == _sha256_of(ckpt), (
        "the record on disk does not describe the leaves on disk: the two "
        "writers crossed")
    # The outer write renamed last, so what stands is its pair, not a record
    # of one class over the other's leaves.
    assert record["descriptor_coordinates"] == "dfs", record
    loaded = load_trained_checkpoint(ckpt, _model(outer_arch, seed=1))
    a, b = _arrays(outer_model), _arrays(loaded)
    assert a and all(np.array_equal(x, y) for x, y in zip(a, b))


def test_a_successful_write_clears_the_temporaries_a_killed_one_left(tmp_path):
    """A write killed between its temporary and its own rename leaves that
    temporary behind, and the next COMPLETED write of the same checkpoint
    removes it.

    With the names drawn, the next write no longer reuses the abandoned one,
    so without the sweep one file would accumulate per kill. Nothing reads a
    temporary -- the eval task picks its three checkpoint names literally --
    but a spec directory that is repeatedly resubmitted would otherwise fill
    with them.

    The kill is modelled as a SIGKILL rather than an exception: the writer's
    own ``except`` clause is suppressed for the killed write, so nothing runs
    after the failure, which is what a signal leaves.
    """
    from xcquinox.alec import train

    ckpt = str(tmp_path / "model.eqx")
    arch = _anchored_dfs_arch()

    monkeypatched = pytest.MonkeyPatch()
    monkeypatched.setattr(train, "_discard", lambda path: None)
    monkeypatched.setattr(train, "discard_staged_record",
                          lambda path, staged=None: None)
    _kill_the_rename_onto(monkeypatched, "model.eqx" + CLASS_RECORD_SUFFIX)
    killed = _run_to_the_kill(train._serialise_trained_model, ckpt,
                              _model(arch, seed=3), arch)
    monkeypatched.undo()
    assert killed is not None, "the record's rename was never reached"

    left = _temporaries_in(tmp_path)
    assert len(left) == 2, left
    assert "model.eqx.tmp" not in left, (
        "the killed write's temporary is the name the next write would use: "
        "the two writes share one temporary per checkpoint")
    assert "model.eqx" + CLASS_RECORD_SUFFIX + ".tmp" not in left, left

    # Dated past the grace the sweep applies. In a run the killed write and
    # the one that follows it are separated by a resubmission; here they are
    # milliseconds apart, and a temporary that recent is deliberately spared
    # (it could belong to a writer that has just started).
    aged = time.time() - 10.0 * TEMPORARY_GRACE_SECONDS
    for name in left:
        os.utime(os.path.join(str(tmp_path), name), (aged, aged))

    train._serialise_trained_model(ckpt, _model(arch, seed=4), arch)
    assert sorted(os.listdir(str(tmp_path))) == [
        "model.eqx", "model.eqx" + CLASS_RECORD_SUFFIX], os.listdir(str(tmp_path))
    assert read_class_record(ckpt)["sha256"] == _sha256_of(ckpt)


def test_a_live_writers_temporary_is_left_alone_by_the_sweep(tmp_path):
    """The sweep removes only what a live write cannot be holding: the set is
    read BEFORE this write draws anything, and every member of it is already
    older than any write can run for.

    Two ways a temporary escapes it, both tested here: drawn AFTER this write
    began (it is not in the set), and drawn moments before (it is in the
    directory but not old enough). Without both, the cleanup would delete a
    concurrent writer's staging file and cost that writer its rename. The
    second is not a theoretical margin -- a file's timestamp comes from the
    kernel's coarse clock and can read milliseconds BEFORE the wall clock at
    the instant it was created, so "older than the moment this write started"
    is a test a just-created file can fail.
    """
    from xcquinox.alec import train
    from xcquinox.alec.checkpoint_class import new_temporary

    ckpt = str(tmp_path / "model.eqx")
    arch = _anchored_dfs_arch()
    abandoned = new_temporary(ckpt)
    with open(abandoned, "wb") as f:
        f.write(b"an abandoned write")
    old = time.time() - 3600.0
    os.utime(abandoned, (old, old))
    just_started = new_temporary(ckpt)

    later = []
    real_serialise = eqx.tree_serialise_leaves

    def _serialise(path, pytree, *args, **kwargs):
        if not later:
            later.append(new_temporary(ckpt))
        return real_serialise(path, pytree, *args, **kwargs)

    monkeypatched = pytest.MonkeyPatch()
    monkeypatched.setattr(train.eqx, "tree_serialise_leaves", _serialise)
    try:
        train._serialise_trained_model(ckpt, _model(arch, seed=3), arch)
    finally:
        monkeypatched.undo()

    assert not os.path.exists(abandoned), "the abandoned temporary was not swept"
    assert os.path.exists(just_started), (
        "a temporary drawn moments before this write began was swept: a "
        "writer that had just started would have lost its staging file")
    assert os.path.exists(later[0]), (
        "a temporary drawn after this write began was swept: a concurrent "
        "writer would have lost its staging file")
    os.remove(just_started)
    os.remove(later[0])


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


def test_a_tampered_checkpoint_is_refused_beside_its_own_record(tmp_path):
    """One byte of the ``.eqx`` changed, its length unchanged, the record left
    exactly as it was written.

    The refusal is the digest's alone: the size the record also carries still
    agrees, and the class it names is still the class of the skeleton asking.
    A checkpoint that is not the one the record was written for is refused
    whatever made it differ -- an interrupted write, a partial copy, a file
    edited in place.
    """
    from xcquinox.alec.eval_holdout import load_trained_model

    ckpt = tmp_path / "model.eqx"
    _write_checkpoint(str(ckpt), _anchored_dfs_arch())
    before = ckpt.read_bytes()
    tampered = bytearray(before)
    tampered[-1] ^= 0x01
    ckpt.write_bytes(bytes(tampered))
    assert os.path.getsize(str(ckpt)) == len(before)

    with pytest.raises(ValueError) as excinfo:
        require_matching_class(str(ckpt),
                               model_class_of_arch(_anchored_dfs_arch()))
    assert isinstance(excinfo.value, ClassRecordStale), excinfo.value
    assert _sha256_of(ckpt) in str(excinfo.value)
    # Both sides of the message carry the same byte count: the size the record
    # also states still agrees, and the digest is what refused.
    assert str(excinfo.value).count(f"({len(before)} bytes)") == 2, excinfo.value
    with pytest.raises(ValueError, match="sha256"):
        load_trained_model(_SpecStub(_anchored_dfs_arch()), ckpt)


def test_a_record_without_a_digest_is_unreadable_rather_than_believed(tmp_path):
    """A record that states no ``sha256`` cannot be held to the leaves beside
    it, and a record that cannot be checked is not evidence about them: it
    raises, exactly as a record that will not parse does, rather than being
    taken at its word about the class.

    Nothing in production predates the digest -- the v6 groups are
    unsubmitted -- so this rule downgrades no checkpoint that exists.
    """
    ckpt = tmp_path / "model.eqx"
    _write_checkpoint(str(ckpt), _anchored_arch())
    record = read_class_record(str(ckpt))
    record.pop("sha256", None)
    record.pop("size", None)
    with open(class_record_path(str(ckpt)), "w") as f:
        json.dump(record, f, indent=2)

    with pytest.raises(ValueError, match="sha256"):
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
