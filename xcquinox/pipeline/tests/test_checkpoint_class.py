"""The model-class record beside a TRAINED checkpoint
(:mod:`xcquinox.pipeline.checkpoint_class`), its writers in the training stage and
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
import io
import pathlib
import pickle
import os

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import pytest

from xcquinox.pipeline.checkpoint_class import (CLASS_RECORD_SUFFIX,
                                            ClassRecordStale,
                                            ModelClassMismatch,
                                            load_trained_checkpoint,
                                            model_class_of_arch,
                                            model_class_of_model,
                                            read_class_record,
                                            require_matching_class,
                                            write_class_record)
from xcquinox.pipeline import config as pipeline_config
from xcquinox.pipeline.config import ArchitectureConfig, TrainingSpec, anchored
from xcquinox.pipeline.models import AlecGGAModel
from xcquinox.pipeline.tests.fixtures.molecules import h_atom, h2o_molecule, o_atom
from xcquinox.pipeline.tests.fixtures.old_name_pickle import pickled_under_the_old_name


# ---------------------------------------------------------------------------
# The package's import surface
# ---------------------------------------------------------------------------

def test_the_package_does_not_import_the_pipeline_eagerly():
    """``import xcquinox`` leaves the subpackage alone: it pulls the quantum-chemistry stack
    behind it, and nothing reaches it through the package's namespace. Importing
    it by name works, which is how every caller does it. Checked in a fresh interpreter: in
    this process the test module has imported the subpackage itself."""
    import subprocess
    import sys
    code = ("import sys, xcquinox; "
            "assert 'xcquinox.pipeline' not in sys.modules, 'the init imported it'; "
            "import xcquinox.pipeline; "
            "print(xcquinox.pipeline.__name__)")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          timeout=600)
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert proc.stdout.strip() == "xcquinox.pipeline"


def test_load_pickle_maps_the_old_class_paths():
    """A spec pickled under the old name fails the plain loader (the module is gone) and
    loads through the mapper as the same object."""
    from xcquinox.pipeline import checkpoint_class as cc
    arch = _anchored_dfs_arch()
    patched = pickled_under_the_old_name(arch)
    with pytest.raises(ModuleNotFoundError):
        pickle.loads(patched)
    loaded = cc.load_pickle(io.BytesIO(patched))
    assert loaded == arch
    assert type(loaded) is type(arch)


def test_record_schema_carries_no_module_path():
    """The record names the model class by its fields and never by a dotted module path, so
    the subpackage's rename changes nothing a record states and every record written before
    it loads unchanged; the writer is reached through the renamed package."""
    from xcquinox.pipeline import checkpoint_class as cc
    record = cc.class_record(_anchored_dfs_arch(), sha256="0" * 64, size=1)
    assert set(record) == {
        "parent_anchor", "descriptor_coordinates", cc.LOG_TRANSFORM_FIELD, "arch_name",
        "meta_gga", "use_polarized_correlation", "parent", "xcquinox_version", "sha256",
        "size"}
    assert not any("module" in key or "class_path" in key for key in record)
    assert not any(isinstance(value, str) and value.startswith("xcquinox.")
                   for value in record.values())


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
    from xcquinox.pipeline.cluster.materialize import running_xcquinox_version
    from xcquinox.pipeline.train import _save_artifacts

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
        assert record["descriptor_log_transform"] is False
        assert record["parent"] == "pbe"
        assert record["arch_name"] == "t"
        assert record["meta_gga"] is False
        assert record["use_polarized_correlation"] is True
        assert record["xcquinox_version"] == running_xcquinox_version()


# ---------------------------------------------------------------------------
# The readers: evaluation.run_test
# ---------------------------------------------------------------------------


def _make_test_spec(arch, checkpoint, output_dir):
    # ``config.TestSpec`` by attribute: the bare name would be collected as a
    # test class by pytest and warned about.
    return pipeline_config.TestSpec(
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
    from xcquinox.pipeline.evaluation import run_test

    ckpt = tmp_path / "model.eqx"
    _write_checkpoint(str(ckpt), written())
    spec = _make_test_spec(wanted(), ckpt, tmp_path / "out")
    with pytest.raises(ValueError) as excinfo:
        run_test(spec)
    message = str(excinfo.value)
    assert "refusing to load" in message
    for fragment in names:
        assert fragment in message, message


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
    from xcquinox.pipeline.eval_holdout import load_trained_model

    ckpt = tmp_path / "model.eqx"
    _write_checkpoint(str(ckpt), written())
    with pytest.raises(ValueError, match="different model classes"):
        load_trained_model(_SpecStub(wanted()), ckpt)


# ---------------------------------------------------------------------------
# The readers: the resume path
# ---------------------------------------------------------------------------

def _write_resume_set(checkpoint_dir, arch, *, with_arch=True):
    """One periodic resume checkpoint of ``arch``'s class."""
    from xcquinox.pipeline.train import _write_resume_checkpoint, build_optimizer

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
    from xcquinox.pipeline.train import _load_resume_checkpoint

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
    from xcquinox.pipeline.eval_holdout import load_trained_model
    from xcquinox.pipeline.evaluation import run_test
    from xcquinox.pipeline.train import (_load_resume_checkpoint,
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


# ---------------------------------------------------------------------------
# The write temporaries: one name per write, not one per checkpoint
# ---------------------------------------------------------------------------


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
    from xcquinox.pipeline import train

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


def test_completion_deletes_the_resume_records_with_their_checkpoints(tmp_path):
    """No record outlives the checkpoint it describes: completion clears the
    resume set and its records together, so the next run in the same directory
    cannot read a class record belonging to a deleted snapshot."""
    from xcquinox.pipeline.train import _finalize_completion

    _write_resume_set(tmp_path, _anchored_dfs_arch())
    _finalize_completion(str(tmp_path), early_stopped=False, epochs_run=1)
    left = sorted(name for name in os.listdir(str(tmp_path))
                  if name.startswith("resume_"))
    assert left == []


# ---------------------------------------------------------------------------
# The record itself
# ---------------------------------------------------------------------------


def test_a_tampered_checkpoint_is_refused_beside_its_own_record(tmp_path):
    """One byte of the ``.eqx`` changed, its length unchanged, the record left
    exactly as it was written.

    The refusal is the digest's alone: the size the record also carries still
    agrees, and the class it names is still the class of the skeleton asking.
    A checkpoint that is not the one the record was written for is refused
    whatever made it differ -- an interrupted write, a partial copy, a file
    edited in place.
    """
    from xcquinox.pipeline.eval_holdout import load_trained_model

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


# ---------------------------------------------------------------------------
# The descriptor log transform: recorded, and compared when the record states it
# ---------------------------------------------------------------------------

def _lt_arch(on):
    """The base architecture with the Dick XCDiff compression on or off.

    ``descriptor_log_transform`` alone separates the two: the anchor, the
    coordinates, the descriptors and every width are the base architecture's,
    so one's leaves fit the other's skeleton exactly as they do across the
    anchor.
    """
    return _base_arch(descriptor_log_transform=on)


def _fx(model):
    """The exchange enhancement factor at three ``(rho, sigma)`` points, which
    is what the flag changes on the legacy coordinates (the MLP is fed
    ``(1 - exp(-s^2)) log(s + 1)`` in place of the raw reduced gradient,
    ``networks.AlecGGA_XNet._core``)."""
    return np.array([float(np.asarray(model.xnet(jnp.array([rho, sigma]))))
                     for rho, sigma in ((0.1, 0.02), (1.0, 0.5), (5.0, 12.0))])


@pytest.mark.parametrize("written_on,wanted_on", [(True, False), (False, True)])
def test_a_checkpoint_written_under_the_other_log_transform_is_refused(
        tmp_path, written_on, wanted_on):
    """Both directions: the record states the flag its checkpoint was written
    under, and a skeleton of the other value is refused with both named.

    The matching skeleton still loads, so this is a comparison and not a
    loader that refuses everything.
    """
    ckpt = str(tmp_path / "model.eqx")
    written = _write_checkpoint(ckpt, _lt_arch(written_on))
    assert read_class_record(ckpt)["descriptor_log_transform"] is written_on

    with pytest.raises(ModelClassMismatch) as excinfo:
        require_matching_class(ckpt, model_class_of_arch(_lt_arch(wanted_on)))
    message = str(excinfo.value)
    assert f"descriptor_log_transform={written_on}" in message, message
    assert f"descriptor_log_transform={wanted_on}" in message, message

    with pytest.raises(ModelClassMismatch):
        load_trained_checkpoint(ckpt, _model(_lt_arch(wanted_on), seed=1))

    loaded = load_trained_checkpoint(ckpt, _model(_lt_arch(written_on), seed=1))
    a, b = _arrays(written), _arrays(loaded)
    assert a and all(np.array_equal(x, y) for x, y in zip(a, b))


def test_the_class_of_an_arch_and_of_the_model_it_builds_agree():
    """``create_network_pair`` carries the configuration's class into the
    built networks' static fields, so the two readings the loaders use --
    from the spec's arch, and from the skeleton itself -- answer the same.

    The four classes the two compared fields reach -- the anchor and the
    coordinates are independent, so unanchored ``dfs`` is one of its own --
    and the log transform, which is read off the same architecture and off the
    exchange network it builds (``networks.create_network_pair`` passes it to
    both nets), giving five distinct readings here."""
    classes = set()
    for arch in (_legacy_arch(), _anchored_arch(), _anchored_dfs_arch(),
                 _dfs_arch(), _lt_arch(True)):
        assert model_class_of_arch(arch) == model_class_of_model(_model(arch))
        classes.add(tuple(sorted(model_class_of_arch(arch).items())))
    assert len(classes) == 5, classes


# ---------------------------------------------------------------------------
# A real checkpoint of the v7 campaign, through the same path the evaluation
# loaders take
# ---------------------------------------------------------------------------

#: a cell of the v7 campaign pulled on 2026-09-08: its trained checkpoint with the class
#: record the training stage wrote beside it, and the spec the harness pickled under the
#: module path older pickles carry (the pull lands outside the tree)
_V7_RUN = (pathlib.Path.home() / "Documents/Research/xcquinox-results/runs/dfs_step7"
           / "dfs6311_grid3_v7g1_c25/runs/run_20260908T153856Z")
_V7_CHECKPOINT = _V7_RUN / "checkpoints/spec_0000/model.eqx"
_V7_SPEC = _V7_RUN / "specs/spec_0000.spec"
_V7_PRESENT = _V7_CHECKPOINT.is_file() and _V7_SPEC.is_file()


