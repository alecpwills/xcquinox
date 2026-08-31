"""The one loader every trained checkpoint outside the library is read
through, and a scan of the repository for the readers that bypass it.

``checkpoint_class.load_trained_checkpoint`` is
``require_matching_class`` followed by ``eqx.tree_deserialise_leaves``. The
first half of this module holds it to the three outcomes that matter: a
checkpoint of another model class is refused, a checkpoint with no record is
the legacy class and loads into a legacy skeleton, and a record that does not
describe the leaves beside it is refused before either class is compared.

The second half is the repository scan. A per-site test is not available for
the notebook BUILDERS -- their loaders live inside cell source strings that
are only executed by a notebook run over a real campaign directory -- so what
is tested instead is the property those sites were changed to have: outside a
stated list, no tracked ``.py`` file fills a skeleton with a bare
``eqx.tree_deserialise_leaves``. The list is in :data:`ALLOWED`, with the
reason each entry is on it, and a new bare call anywhere else fails here.
"""
import dataclasses
import os
import re
import subprocess
from pathlib import Path

import equinox as eqx
import jax.tree_util as jtu
import numpy as np
import pytest

from xcquinox.alec.checkpoint_class import (ClassRecordStale,
                                            ModelClassMismatch,
                                            class_record_path,
                                            load_trained_checkpoint,
                                            write_class_record)
from xcquinox.alec.config import ArchitectureConfig, anchored
from xcquinox.alec.models import AlecGGAModel

REPO = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# Architectures: two model classes with IDENTICAL parameter shapes
# ---------------------------------------------------------------------------

def _base_arch(**overrides):
    """A small architecture with polarized correlation -- the shape the anchor
    needs -- so the classes below differ in static fields alone."""
    defaults = dict(
        name="t", depth=2, nodes=8, attention=False,
        descriptors=(), x_constraints=(), c_constraints=(),
        double_lob_clamp_allowed=False, use_polarized_correlation=True,
    )
    defaults.update(overrides)
    return ArchitectureConfig(**defaults)


def _legacy_arch():
    return _base_arch()


def _anchored_dfs_arch():
    return dataclasses.replace(anchored(_base_arch()),
                               descriptor_coordinates="dfs")


def _model(arch, seed=0):
    return AlecGGAModel.from_arch(arch, seed=seed)


def _arrays(model):
    return [np.asarray(x)
            for x in jtu.tree_leaves(eqx.filter(model, eqx.is_array))]


def _write(path, arch, *, record=True, seed=0):
    model = _model(arch, seed=seed)
    eqx.tree_serialise_leaves(path, model)
    if record:
        write_class_record(path, arch)
    return model


# ---------------------------------------------------------------------------
# load_trained_checkpoint
# ---------------------------------------------------------------------------

def test_the_loader_refuses_a_checkpoint_of_another_class(tmp_path):
    """The hazard the loader exists for, and the bare call beside it.

    An anchored DFS checkpoint read into an unanchored legacy skeleton:
    ``eqx.tree_deserialise_leaves`` returns a model with every array equal to
    what was written and nothing raising -- the anchor and the coordinates are
    static fields with no parameters of their own -- while
    ``load_trained_checkpoint`` refuses it by name. The bare call is asserted
    first so the refusal is held against a hazard that was shown, not assumed.
    """
    path = str(tmp_path / "model.eqx")
    written = _write(path, _anchored_dfs_arch())

    silent = eqx.tree_deserialise_leaves(path, _model(_legacy_arch(), seed=7))
    a, b = _arrays(written), _arrays(silent)
    assert a and all(np.array_equal(x, y) for x, y in zip(a, b))

    with pytest.raises(ModelClassMismatch) as excinfo:
        load_trained_checkpoint(path, _model(_legacy_arch(), seed=7))
    assert "descriptor_coordinates='dfs'" in str(excinfo.value)
    assert "descriptor_coordinates='legacy'" in str(excinfo.value)


def test_the_loader_honours_the_record_and_returns_the_leaves(tmp_path):
    """The accepting leg: the recorded class is the skeleton's, so the leaves
    are read, and what comes back is the model that was written."""
    path = str(tmp_path / "model.eqx")
    arch = _anchored_dfs_arch()
    written = _write(path, arch, seed=3)

    loaded = load_trained_checkpoint(path, _model(arch, seed=11))
    a, b = _arrays(written), _arrays(loaded)
    assert a and all(np.array_equal(x, y) for x, y in zip(a, b))


def test_the_loader_accepts_a_legacy_checkpoint_with_no_record(tmp_path):
    """A checkpoint with no record beside it is the unanchored legacy class:
    a legacy skeleton reads it exactly as it did before the record existed --
    which is what the pulled v3/v4/v5 campaign directories are -- and any
    other skeleton is refused, since nothing on disk states what the leaves
    are."""
    path = str(tmp_path / "model.eqx")
    written = _write(path, _legacy_arch(), record=False, seed=5)
    assert not os.path.isfile(class_record_path(path))

    loaded = load_trained_checkpoint(path, _model(_legacy_arch(), seed=1))
    a, b = _arrays(written), _arrays(loaded)
    assert a and all(np.array_equal(x, y) for x, y in zip(a, b))

    with pytest.raises(ModelClassMismatch, match="no model-class record"):
        load_trained_checkpoint(path, _model(_anchored_dfs_arch(), seed=1))


def test_the_loader_refuses_a_record_that_describes_other_leaves(tmp_path):
    """The record is held to the ``.eqx`` beside it before either class is
    compared: leaves replaced under a record that stays put are refused as
    stale, whichever class the skeleton is."""
    path = str(tmp_path / "model.eqx")
    arch = _anchored_dfs_arch()
    _write(path, arch)
    eqx.tree_serialise_leaves(path, _model(arch, seed=17))

    for skeleton_arch in (arch, _legacy_arch()):
        with pytest.raises(ClassRecordStale):
            load_trained_checkpoint(path, _model(skeleton_arch, seed=1))


def test_the_loader_names_the_kind_of_file_it_refused(tmp_path):
    """``what`` reaches the message, so a caller reading several kinds of
    checkpoint out of one directory says which one was refused."""
    path = str(tmp_path / "model_val_best.eqx")
    _write(path, _anchored_dfs_arch())
    with pytest.raises(ModelClassMismatch, match="validation-best snapshot"):
        load_trained_checkpoint(path, _model(_legacy_arch()),
                                what="validation-best snapshot")


# ---------------------------------------------------------------------------
# The repository scan
# ---------------------------------------------------------------------------

#: What a bare deserialise looks like in source. The identifier followed by an
#: opening parenthesis, so the prose that NAMES the function in a docstring is
#: not a hit.
_CALL = re.compile(r"tree_deserialise_leaves\s*\(")

#: Files whose bare calls are allowed, and why. The categories:
#:
#:   library_reader   The package's own trained-checkpoint readers, which do
#:                    the same check inline and keep their own sequence:
#:                    ``evaluation.run_test`` and
#:                    ``eval_holdout.load_trained_model`` compare the class of
#:                    the SPEC's arch (the authority there -- the skeleton is
#:                    built from it two lines above) and attach their own
#:                    message to a deserialisation failure;
#:                    ``train._load_resume_checkpoint`` must DELETE the refused
#:                    resume set around its own call, which a loader that only
#:                    returns a model cannot do. ``checkpoint_class`` is the
#:                    sanctioned loader itself.
#:   pretrain_reader  Reads a pretrain ``xnet.eqx`` / ``cnet.eqx``, whose class
#:                    is guarded by ``pretrain_metadata.json``
#:                    (``train._require_matching_model_class``) and not by the
#:                    checkpoint record. Every call in such a file must NAME
#:                    xnet or cnet, checked below, so a trained-model load
#:                    added to one of these files is not covered by its entry.
#:   library_pretrain ``pretrain.py``'s own loaders: the library-format
#:                    networks and the step3b legacy bridge, neither of which
#:                    is an AlecGGAModel checkpoint.
#:   legacy_package   ``xcquinox.net`` / ``xcquinox.xc`` and the scripts that
#:                    drive them. Those models are not AlecGGAModel and carry
#:                    neither the anchor nor the descriptor coordinates, so
#:                    there is no class for a record to state.
#:
#: Tests are allowed wholesale (see :func:`_category`): a test that builds the
#: cross-class load on purpose is how the hazard is stated.
ALLOWED = {
    "xcquinox/alec/checkpoint_class.py": "library_reader",
    "xcquinox/alec/evaluation.py": "library_reader",
    "xcquinox/alec/eval_holdout.py": "library_reader",
    "xcquinox/alec/train.py": "library_reader",
    "xcquinox/alec/pretrain.py": "library_pretrain",
    "xcquinox/alec/cluster/fidelity.py": "pretrain_reader",
    "notebooks/_build_step4_notebook.py": "pretrain_reader",
    "notebooks/_build_step5_notebook.py": "pretrain_reader",
    "notebooks/_build_step6_notebook.py": "pretrain_reader",
    "notebooks/_patch_plots.py": "pretrain_reader",
    "notebooks/parallel_pretrain_cells.py": "pretrain_reader",
    "notebooks/analysis/constraint_pretrain_gmtkn55_demo.py": "pretrain_reader",
    "notebooks/analysis/mgga_diagnosis_evidence.py": "pretrain_reader",
    "notebooks/analysis/multimode_constraint_eval.py": "pretrain_reader",
    "hpcjobs/dfs6311_nan_verify.py": "pretrain_reader",
    "hpcjobs/dfs6311_pretrained_holdout.py": "pretrain_reader",
    "hpcjobs/probe_pretrain_energy_weight.py": "pretrain_reader",
    "scripts/parallel_train_worker.py": "pretrain_reader",
    "xcquinox/net.py": "legacy_package",
    "xcquinox/xc.py": "legacy_package",
    "scripts/calculate_traj.py": "legacy_package",
    "scripts/pt_validation.py": "legacy_package",
    "scripts/train_traj.py": "legacy_package",
}


def _category(relpath):
    """The allowed category for one repository-relative path, or ``None``."""
    parts = relpath.split("/")
    if "tests" in parts or parts[-1].startswith("test_"):
        return "tests"
    return ALLOWED.get(relpath)


def _tracked_python_files():
    """Every tracked ``.py`` path, repository-relative and POSIX-separated.

    ``git ls-files`` rather than a directory walk: an untracked working file
    is not part of the repository and must not turn this red, and the
    untracked figure and scratch directories are skipped for free.
    """
    try:
        out = subprocess.run(["git", "-C", str(REPO), "ls-files", "-z", "--", "*.py"],
                             check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"the tracked file list is unavailable: {exc}")
    return [name for name in out.stdout.decode().split("\0") if name]


def _statement_at(lines, index):
    """The call's whole statement: its line plus the continuations it needs to
    balance its parentheses (bounded, so a malformed file cannot run away)."""
    text = ""
    depth = 0
    for line in lines[index:index + 8]:
        text += line
        depth += line.count("(") - line.count(")")
        if depth <= 0:
            break
    return text


def test_no_new_bare_deserialise_fills_a_trained_model_skeleton():
    """No tracked ``.py`` outside :data:`ALLOWED` deserialises into a skeleton
    without checking the checkpoint's model class first.

    The anchor and the descriptor coordinates change no parameter shape, so a
    bare call loads one model class's weights into another class's skeleton
    with every array equal and nothing raising, and what comes out is the
    parent plus a correction trained as the whole factor -- a plausible curve
    with no error attached. The readers outside the package go through
    ``checkpoint_class.load_trained_checkpoint`` instead; this case is what
    keeps the next one from being written.
    """
    offenders = []
    for relpath in _tracked_python_files():
        try:
            source = (REPO / relpath).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        if not _CALL.search(source):
            continue
        category = _category(relpath)
        if category is None:
            offenders.append(
                f"{relpath}: a bare tree_deserialise_leaves with no entry in "
                "ALLOWED. Read the checkpoint through "
                "checkpoint_class.load_trained_checkpoint, or add the file to "
                "ALLOWED with the reason its call is not a trained-model load.")
    assert not offenders, "\n".join(offenders)


def test_every_allowed_pretrain_reader_still_reads_only_pretrain_networks():
    """The ``pretrain_reader`` entries are allowed for the ``xnet.eqx`` /
    ``cnet.eqx`` they read, not for the file as a whole.

    A pretrain load always names one of the two networks -- it is the
    subnetwork it fills -- so a call in one of these files that names neither
    is a load of something else, which is exactly the trained-model load the
    entry does not cover. Without this, a bare trained-checkpoint reader added
    to a notebook builder would inherit that builder's entry.

    Prose is held to the same rule: a docstring that WRITES OUT a bare call
    with a full-model skeleton is flagged like the call itself, which is how
    ``_build_step6_notebook.py``'s "canonical pattern" comment -- naming a
    pattern the library's readers no longer use -- would be caught if it came
    back.
    """
    offenders = []
    for relpath, category in sorted(ALLOWED.items()):
        if category != "pretrain_reader":
            continue
        path = REPO / relpath
        assert path.is_file(), f"{relpath} is in ALLOWED and does not exist"
        lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
        for index, line in enumerate(lines):
            if not _CALL.search(line):
                continue
            statement = _statement_at(lines, index)
            if "xnet" not in statement and "cnet" not in statement:
                offenders.append(
                    f"{relpath}:{index + 1}: {statement.strip()!r} names "
                    "neither xnet nor cnet, so it is not the pretrain read "
                    "this file is allowed for")
    assert not offenders, "\n".join(offenders)


def test_every_allowed_entry_is_still_needed():
    """An entry whose file no longer has a bare call is removed rather than
    left standing: the list is what the next reader is checked against, and a
    stale entry silently widens it."""
    stale = []
    for relpath in sorted(ALLOWED):
        path = REPO / relpath
        if not path.is_file():
            stale.append(f"{relpath}: not in the tree")
        elif not _CALL.search(path.read_text(encoding="utf-8")):
            stale.append(f"{relpath}: no bare call left; drop the entry")
    assert not stale, "\n".join(stale)
