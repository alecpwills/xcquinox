"""The model class a TRAINED checkpoint was written as, recorded beside it.

The parent anchor (``ArchitectureConfig.parent_anchor``), the descriptor
coordinates (``ArchitectureConfig.descriptor_coordinates``) and the descriptor
log transform (``ArchitectureConfig.descriptor_log_transform``) are STATIC
properties of the networks: none changes a parameter shape, so an
``.eqx`` leaf stream written by one class deserialises into the skeleton of
another without complaint, and the resulting model is the parent plus a
correction trained as the whole factor -- O(1) wrong everywhere, with nothing
raising. The pretraining stage already guards its own hand-off this way
(``pretrain_metadata.json``, read by ``train._require_matching_model_class``);
this module is the same record for the checkpoints the TRAINING stage writes,
which the evaluation loaders and the resume path fill from ``spec.arch``.

The record is one JSON file per checkpoint, at the checkpoint's own path plus
:data:`CLASS_RECORD_SUFFIX` (``model.eqx`` -> ``model.eqx.class.json``),
rather than one file per checkpoint directory. The readers are handed a FILE
(``TestSpec.model_checkpoint``; ``eval_holdout.load_trained_model``'s
``model_path``; the eval task picks among ``model.eqx``,
``model_best.eqx`` and ``model_val_best.eqx`` in the same directory), so the
record's path follows from what the reader already holds, with no knowledge of
the run layout.

What that buys operationally is that the record travels with the checkpoint
under the pull the local re-evaluation workflow uses:
``cluster.sync.build_rsync_command`` narrows per spec DIRECTORY (its
``spec_indices`` emit ``--include=/checkpoints/spec_NNNN/***``), so every
record comes down beside its own ``.eqx``. A file copied out BY HAND does not
carry its sibling, and the residual risk is asymmetric: the bare copy then
reads as a checkpoint with no record, which an anchored or ``dfs`` skeleton
refuses (fail-safe) but a LEGACY skeleton accepts and loads as legacy -- the
silent cross-class load this module exists to prevent, back again, because the
one file that stated the class was left behind. The record must therefore be
copied with the checkpoint whenever a checkpoint is moved outside the pull.

The record DESCRIBES the checkpoint rather than merely standing beside it: it
carries the SHA-256 and the byte count of the exact ``.eqx`` it was written
for, and :func:`require_matching_class` digests the file on disk and compares
before it compares classes. Two files cannot be renamed in one step, so some
kill always lands between them and no write ORDER can keep them consistent on
its own; the digest is what tells a reader which side of that kill it is
looking at. A record whose digest is not the digest of the leaves beside it is
a record of a checkpoint that is no longer there, and is refused
(:class:`ClassRecordStale`) rather than believed.

Payload, in the vocabulary of the records it sits beside
(``pretrain_metadata.json`` for the two class fields, ``arch_name``,
``meta_gga`` and ``use_polarized_correlation``; the fidelity certificate for
``parent`` and ``xcquinox_version``)::

    parent_anchor              bool   the class, compared by the readers
    descriptor_coordinates     str    the class, compared by the readers
    descriptor_log_transform   bool   the class, compared by the readers WHEN
                                      THE RECORD STATES IT (below)
    arch_name                  str    provenance
    meta_gga                   bool   provenance (ArchitectureConfig.is_meta_gga)
    use_polarized_correlation  bool   provenance (a shape-CHANGING flag, which
                                      the loaders check on the network itself)
    parent                     str    the parent functional when anchored, else
                                      null (``parents.parent_for_arch``)
    xcquinox_version           str    provenance
    sha256                     str    the digest of the .eqx this record
                                      describes, verified before the class is
                                      compared
    size                       int    that checkpoint's size in bytes

The first three are what "the model class" means: each is a static field of
the networks, none changes a parameter shape, and so none of the three is
revealed by a leaf stream. ``sha256`` and ``size`` say which leaves the record
refers to. The rest states what wrote the file.

``descriptor_log_transform`` is compared ONLY WHEN THE RECORD STATES IT. Every
record written before the field was added carries the other keys and not this
one, and is read exactly as it was: such a record is accepted by a skeleton of
either value, and a checkpoint with NO record is the legacy class whatever the
flag (:func:`is_legacy_class` is the first two fields alone -- 23 of the 31
registry architectures set the transform, so folding it into that judgement
would refuse every unrecorded v3/v4/v5 checkpoint of those architectures to
the very skeleton that wrote it). What the flag changes, measured at identical
leaves: on the legacy coordinates the MLPs are fed
``(1 - exp(-x^2)) log(x + 1)`` in place of the raw reduced gradient and r_s,
which moved F_x by 1.9e-3 on a depth-2 untrained pair; and on EVERY coordinate
set the cusp descriptor's second column is log-compressed before its tanh
(``config.ArchitectureConfig.materialize_descriptors`` ->
``features.compute_cusp_descriptor``: 0.51 apart on a bounded (-1, 1) column
at 0.3 to 4 bohr from an oxygen nucleus), which 13 registry architectures
carry. The ``dfs`` coordinates bypass the network transform and nothing else,
so the field is not the coordinates under another name.

A checkpoint with NO record beside it is a legacy checkpoint -- unanchored, on
the legacy coordinates -- because every run that writes an anchored or a
``dfs`` checkpoint writes the record in the same call that writes the ``.eqx``
(``train._serialise_trained_model``, the one write path, through
:func:`stage_class_record` and :func:`commit_class_record`). A legacy skeleton
therefore accepts it and any other skeleton refuses it, which is the rule
``train._require_matching_model_class`` applies to a pretrain directory with
no metadata.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time

#: Appended to a checkpoint's own path to give its class record's path.
CLASS_RECORD_SUFFIX = ".class.json"

#: Ends the name of every temporary this module and the training writer put
#: down beside a file they are about to replace. The name in between is drawn
#: by :func:`new_temporary` and is unique per write, so two writers in one
#: directory never share one. Nothing reads a temporary.
TEMPORARY_SUFFIX = ".tmp"

#: Read size for :func:`file_digest`. The whole checkpoint is never held in
#: memory, and one chunk covers every checkpoint this project writes.
DIGEST_CHUNK_BYTES = 1 << 20

#: The class of a checkpoint with no record: what everything written before
#: the anchor existed is. The two compared fields alone -- a checkpoint with no
#: record states no descriptor log transform either, and the readers compare
#: that field only where it is stated (:func:`require_matching_log_transform`).
LEGACY_CLASS = {"parent_anchor": False, "descriptor_coordinates": "legacy"}

#: The record field naming the descriptor log transform the checkpoint was
#: written under. Optional in the record, unlike the two class fields: absent
#: in every record written before it was added.
LOG_TRANSFORM_FIELD = "descriptor_log_transform"


class ModelClassMismatch(ValueError):
    """A checkpoint's recorded class is not the class being built.

    A ``ValueError``, so every caller that already handles the refusal sees
    what it saw before; a type of its own so a caller that can ACT on it can
    tell a class refusal -- which is permanent for that pair of file and
    configuration -- from a missing file or a record that could not be read.
    ``train._load_resume_checkpoint`` is the caller that acts: it discards the
    resume set it refused.
    """


class ClassRecordStale(ValueError):
    """A class record does not describe the checkpoint lying beside it.

    Raised when the SHA-256 (or the byte count) the record carries is not the
    one the ``.eqx`` on disk has, and when the record stands beside no ``.eqx``
    at all. The record and the leaves are separate files, so a write
    interrupted between the two renames leaves exactly this state -- the new
    record over the previous run's complete checkpoint -- and it is the state
    in which a reader would otherwise load one model class believing the
    record's word for another. A ``ValueError``, so every caller that already
    treats a refusal as "no usable checkpoint" sees what it saw before; a type
    of its own so the condition can be told from a class mismatch (which is a
    correctly described checkpoint of the wrong class) and from an unreadable
    record.
    """


def class_record_path(checkpoint_path) -> str:
    """The class record's path for the checkpoint at ``checkpoint_path``."""
    return f"{os.fspath(checkpoint_path)}{CLASS_RECORD_SUFFIX}"


def _temporary_pattern(target_path):
    """Matches the basenames of the temporaries belonging to ``target_path``.

    ``<name>.<drawn>.tmp`` for the ones :func:`new_temporary` draws, and the
    bare ``<name>.tmp`` a writer from before the names were drawn left behind.
    The drawn part is ``mkstemp``'s alphabet, which carries no ``.``, so the
    pattern for ``model.eqx`` does NOT match ``model.eqx.class.json.X.tmp``:
    the leaves' temporaries and the record's are swept separately, each by its
    own target.
    """
    base = os.path.basename(os.fspath(target_path))
    return re.compile(r"\A" + re.escape(base) + r"(\.[A-Za-z0-9_]+)?"
                      + re.escape(TEMPORARY_SUFFIX) + r"\Z")


def new_temporary(target_path) -> str:
    """An empty temporary beside ``target_path``, under a name no other write
    can be using, and return its path.

    ``tempfile.mkstemp`` in ``target_path``'s OWN directory, so the rename
    that commits it stays within one filesystem and is therefore atomic. The
    name matters because the temporary is not private to one process: two
    generations of the same training task write the same checkpoint directory
    (``cluster.__main__.cmd_resubmit`` re-submits a retryable index into the
    run directory it was classified in), and under one fixed name per
    checkpoint the later writer would serialise into the earlier writer's
    half-written file and each would rename the other's bytes into place. With
    a drawn name each writer commits its own leaves under its own record, and
    the interleavings that remain are the two renames crossing, which the
    record's digest refuses (:func:`require_matching_digest`).
    """
    target = os.fspath(target_path)
    directory = os.path.dirname(os.path.abspath(target)) or "."
    fd, path = tempfile.mkstemp(dir=directory,
                                prefix=os.path.basename(target) + ".",
                                suffix=TEMPORARY_SUFFIX)
    os.close(fd)
    return path


def _temporaries_of(target_path):
    """Every temporary of ``target_path`` on disk now, as paths. A directory
    that cannot be listed yields none."""
    target = os.fspath(target_path)
    directory = os.path.dirname(os.path.abspath(target)) or "."
    pattern = _temporary_pattern(target)
    try:
        names = os.listdir(directory)
    except OSError:
        return ()
    return tuple(os.path.join(directory, name)
                 for name in sorted(names) if pattern.match(name))


#: How old a temporary must be before a write treats it as abandoned. A
#: temporary a LIVE write is still holding is at most as old as that write:
#: the whole sequence -- serialise, fsync, digest, record, two renames, the
#: directory fsync -- costs 22 ms per checkpoint measured on this workstation,
#: essentially all of it the three fsyncs, and the serialise of the largest
#: ``.eqx`` in the tree is 0.9 ms. Sixty seconds is over three orders above
#: that, so nothing this old is being written; and since a temporary is inert
#: (no reader globs for one), leaving a recent one for the write after this
#: one costs nothing while deleting a live writer's staging file would cost
#: that writer its rename.
TEMPORARY_GRACE_SECONDS = 60.0


def stale_temporaries(target_path, *,
                      grace: float = TEMPORARY_GRACE_SECONDS) -> tuple:
    """The temporaries of ``target_path`` that no live write can be holding:
    those already older than ``grace`` at the moment this is called.

    Read at the START of a write and removed when it succeeds
    (``train._serialise_trained_model``), so the set can contain nothing the
    write itself, or anything that began after it, drew. That ordering is what
    makes the sweep safe against a second writer in the same directory; the
    age bound is what makes it safe against one that started moments earlier.
    File timestamps come from the kernel's coarse clock and can read a few
    milliseconds BEFORE the wall clock at the instant the file was created, so
    a bound tighter than that clock's granularity would not hold.
    """
    now = time.time()
    out = []
    for path in _temporaries_of(target_path):
        try:
            if now - os.stat(path).st_mtime > float(grace):
                out.append(path)
        except OSError:
            pass
    return tuple(out)


def discard_temporaries(target_path) -> None:
    """Remove EVERY temporary of ``target_path``, whatever drew it; absent is
    fine.

    For the caller that is deleting ``target_path`` itself
    (``train._remove_resume_set``): nothing that belongs to the file can
    matter once the file is gone.
    """
    for path in _temporaries_of(target_path):
        try:
            os.remove(path)
        except OSError:
            pass


def file_digest(path) -> tuple:
    """``(sha256_hex, size_in_bytes)`` of the file at ``path``.

    Read in :data:`DIGEST_CHUNK_BYTES` chunks, so an arbitrarily large
    checkpoint costs one buffer rather than its own size in memory. Measured
    on this project's checkpoints: 13,968 bytes (``deep_rung35ms_mgga_3x16``)
    in 0.015 ms, and the largest ``.eqx`` in the tree, 131,122 bytes
    (a ``deep_combined_attn``), in 0.091 ms -- against 0.90 ms to serialise
    the first of the two, so the digest is under 2 percent of the write it
    protects and is not on any inner loop.
    """
    digest = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(DIGEST_CHUNK_BYTES)
            if not chunk:
                break
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def fsync_file(path) -> None:
    """Flush the file at ``path`` to stable storage.

    A crash, as opposed to a clean kill, can otherwise reorder the two writes
    arbitrarily: the small record reaches the platter while the checkpoint it
    describes has not. Both temporaries are fsync'd before either is renamed,
    so the only ordering a reader can see is the order of the renames.
    """
    fd = os.open(os.fspath(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def fsync_dir_of(path) -> None:
    """Flush the DIRECTORY entry of ``path``, so the renames themselves are
    durable and not only the bytes they point at.

    A filesystem that does not permit the directory to be opened or synced
    (nothing this project runs on) is tolerated: the renames are still atomic
    there, only their durability across a crash is the filesystem's business.
    """
    directory = os.path.dirname(os.path.abspath(os.fspath(path))) or "."
    try:
        fd = os.open(directory, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    except OSError:
        return
    try:
        os.fsync(fd)
    except OSError:
        pass
    finally:
        os.close(fd)


def model_class_of_arch(arch) -> dict:
    """The model class ``arch`` states: the fields the readers compare.

    The defaults are the legacy class, so an arch-like object from before the
    fields existed answers as the class it is. ``descriptor_log_transform``
    rides here with the two class fields because it is the only channel the
    readers have -- ``evaluation.run_test`` and ``eval_holdout`` hand this
    dict straight to :func:`require_matching_class` -- and it is compared
    against a record that states it, never against one that does not.
    """
    return {
        "parent_anchor": bool(getattr(arch, "parent_anchor", False)),
        "descriptor_coordinates": str(
            getattr(arch, "descriptor_coordinates", "legacy")),
        LOG_TRANSFORM_FIELD: bool(
            getattr(arch, LOG_TRANSFORM_FIELD, False)),
    }


def model_class_of_model(model) -> dict:
    """The model class a BUILT model carries, read off its exchange network's
    static fields (``AlecGGA_XNet.parent``, ``.descriptor_coordinates``,
    ``.descriptor_log_transform``).

    The same question as :func:`model_class_of_arch` asked of the object
    rather than of the configuration; ``networks.create_network_pair`` is the
    single site that carries one into the other. Used where a reader holds
    the skeleton it is about to fill but not the arch that produced it
    (``train._load_resume_checkpoint``).
    """
    xnet = getattr(model, "xnet", None)
    return {
        "parent_anchor": getattr(xnet, "parent", None) is not None,
        "descriptor_coordinates": str(
            getattr(xnet, "descriptor_coordinates", "legacy")),
        LOG_TRANSFORM_FIELD: bool(
            getattr(xnet, LOG_TRANSFORM_FIELD, False)),
    }


def is_legacy_class(model_class) -> bool:
    """Whether ``model_class`` is the unanchored legacy class.

    The anchor and the coordinates alone. The descriptor log transform is
    deliberately no part of this: what it decides is whether a checkpoint with
    NO record beside it may be read, and the campaigns that left those
    checkpoints set the transform on most of their architectures, so a rule
    that read it here would refuse them all to the class that wrote them.
    """
    return (not model_class["parent_anchor"]
            and model_class["descriptor_coordinates"] == "legacy")


def describe_class(model_class) -> str:
    """One line naming a class, in the loaders' shared vocabulary.

    The two fields every record states. The descriptor log transform is not
    named here because it is compared only where the record carries it, and
    its refusal (:func:`require_matching_log_transform`) names both values
    itself.
    """
    return (f"parent_anchor={model_class['parent_anchor']}, "
            f"descriptor_coordinates="
            f"{model_class['descriptor_coordinates']!r}")


def class_record(arch, *, sha256, size) -> dict:
    """The full payload written beside a checkpoint built from ``arch``.

    ``sha256`` and ``size`` are the digest and the byte count of the exact
    ``.eqx`` this record is being written for -- of the temporary the leaves
    were just serialised to, in the one write path, since that is the file
    about to become the checkpoint. They are what makes the record describe a
    particular set of leaves rather than a location.

    The parent name comes from ``parents.parent_for_arch`` rather than from a
    second reading of the rung, so the record cannot name a parent the
    anchored networks were not built against. Imported inside: this module is
    read by the cheap loaders, and ``parents`` pulls JAX in.
    """
    from xcquinox.alec import parents
    from xcquinox.alec.cluster.materialize import running_xcquinox_version
    from xcquinox.alec.config import ArchitectureConfig

    record = dict(model_class_of_arch(arch))
    record.update({
        "arch_name": getattr(arch, "name", None),
        "meta_gga": bool(ArchitectureConfig.is_meta_gga(arch)),
        "use_polarized_correlation": bool(
            getattr(arch, "use_polarized_correlation", False)),
        "parent": (parents.parent_for_arch(arch)
                   if record["parent_anchor"] else None),
        "xcquinox_version": running_xcquinox_version(),
        "sha256": str(sha256),
        "size": int(size),
    })
    return record


def stage_class_record(checkpoint_path, arch, *, sha256, size) -> str:
    """Write the record for ``checkpoint_path`` to its STAGED path, flush it,
    and return that path: the first half of the two-phase record write.

    The record is written where nothing reads it and moved into place by
    :func:`commit_class_record`, so the record on disk is at every instant
    either wholly the previous one or wholly this one -- never a half-written
    JSON file some reader has to parse. ``sha256`` / ``size`` describe the
    leaves this record is being committed for; the reader compares them
    against the ``.eqx`` it finds, which is what tells it whether the two
    files are the pair they claim to be.

    The staged path is DRAWN (:func:`new_temporary`) rather than derived from
    the checkpoint's name, so it belongs to this write alone; the caller hands
    the returned path back to :func:`commit_class_record`.
    """
    staged = new_temporary(class_record_path(checkpoint_path))
    with open(staged, "w") as f:
        json.dump(class_record(arch, sha256=sha256, size=size), f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    return staged


def commit_class_record(checkpoint_path, staged) -> str:
    """Move the record ``staged`` for ``checkpoint_path`` into place, atomically
    (``os.replace``), and return the record's path.

    ``staged`` is what :func:`stage_class_record` returned for this write; no
    other write's temporary is named by any path this function derives.
    """
    path = class_record_path(checkpoint_path)
    os.replace(os.fspath(staged), path)
    return path


def write_class_record(checkpoint_path, arch) -> str:
    """Record ``arch``'s class for the checkpoint ALREADY on disk at
    ``checkpoint_path``, and return the record's path.

    The digest is taken from the file as it stands at this moment, so the
    record cannot describe leaves other than the ones it was written for. The
    checkpoint must exist; a record for an absent ``.eqx`` is the stale state
    the readers refuse, and writing one deliberately is never what a caller
    means.

    This is the form for a checkpoint that is already in place -- adopting one
    written elsewhere, or re-recording one whose class is known. The training
    stage does not use it: ``train._serialise_trained_model`` stages the
    record around the leaves' own rename so that a kill between the two is a
    state the digest can name.
    """
    sha256, size = file_digest(checkpoint_path)
    staged = stage_class_record(checkpoint_path, arch, sha256=sha256, size=size)
    path = commit_class_record(checkpoint_path, staged)
    fsync_dir_of(path)
    return path


def discard_staged_record(checkpoint_path, staged=None) -> None:
    """Remove a staged record whose checkpoint never landed; absent is fine.

    ``staged`` is the path :func:`stage_class_record` returned, and is what a
    failed write passes so that it discards its OWN temporary and no other
    writer's. With no argument every record temporary of ``checkpoint_path``
    goes, which is what the caller that is deleting the checkpoint itself
    means (``train._remove_resume_set``).

    The record on disk is left exactly as it was, which is what keeps a failed
    write from re-labelling the checkpoint that survived it.
    """
    if staged is None:
        discard_temporaries(class_record_path(checkpoint_path))
        return
    try:
        os.remove(os.fspath(staged))
    except FileNotFoundError:
        pass


def remove_class_record(checkpoint_path) -> None:
    """Delete the class record beside ``checkpoint_path``; absent is fine.

    Called wherever the checkpoint itself is deleted (the resume set on
    completion), so a stale record can never outlive its ``.eqx``.
    """
    try:
        os.remove(class_record_path(checkpoint_path))
    except FileNotFoundError:
        pass


def _digest_fields(record, path):
    """``(sha256, size)`` from a parsed record, or ``ValueError``.

    A record that states no digest cannot be checked against the checkpoint,
    and a record that cannot be checked is not evidence of anything: it is
    read as unreadable rather than trusted on its class alone. Nothing in
    production predates the digest -- the v6 groups are unsubmitted and every
    record a test writes is made by the writers above.
    """
    sha256 = record.get("sha256") if isinstance(record, dict) else None
    size = record.get("size") if isinstance(record, dict) else None
    if not isinstance(sha256, str) or len(sha256) != 64:
        raise ValueError(
            f"the model-class record {path!r} states no usable sha256 for the "
            "checkpoint it describes, so it cannot be checked against the "
            "leaves on disk: the record and the checkpoint are separate "
            f"files, and the digest is what says they are a pair (got "
            f"{sha256!r})")
    try:
        int(sha256, 16)
    except ValueError as exc:
        raise ValueError(
            f"the model-class record {path!r} states a sha256 that is not "
            f"hexadecimal: {sha256!r}") from exc
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise ValueError(
            f"the model-class record {path!r} states no usable size in bytes "
            f"for the checkpoint it describes (got {size!r})")
    return sha256.lower(), size


def read_class_record(checkpoint_path):
    """The record beside ``checkpoint_path``, or ``None`` when there is none.

    A record that cannot be read, cannot be parsed, or does not carry the
    ``sha256`` / ``size`` of the checkpoint it describes raises ``ValueError``:
    an unreadable record is not the same as no record, and answering "legacy"
    for it would be a guess at the one thing this file exists to state.
    """
    path = class_record_path(checkpoint_path)
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            record = json.load(f)
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"the model-class record {path!r} beside checkpoint "
            f"{os.fspath(checkpoint_path)!r} could not be read: {exc}") from exc
    _digest_fields(record, path)
    return record


def require_matching_digest(checkpoint_path, record) -> None:
    """Refuse a record that does not describe the ``.eqx`` beside it.

    The record carries the SHA-256 and the byte count of the leaves it was
    written for; here they are compared against the file on disk. The two are
    separate files and cannot be renamed in one step, so the state this
    catches is a real one: a write interrupted between the record's rename and
    the checkpoint's leaves the NEW record over the PREVIOUS run's complete
    ``.eqx``, and without the digest a reader takes the record's word for
    leaves that were written by another run, of another class
    (``train._serialise_trained_model`` sets out every kill point). A record
    beside no checkpoint at all is the same failure with the leaves missing
    rather than stale.

    Raises :class:`ClassRecordStale`, naming the file and both digests.
    """
    path = os.fspath(checkpoint_path)
    recorded_sha, recorded_size = _digest_fields(record, class_record_path(path))
    try:
        got_sha, got_size = file_digest(path)
    except FileNotFoundError as exc:
        raise ClassRecordStale(
            f"the model-class record {class_record_path(path)!r} describes a "
            f"checkpoint that is not on disk: {path!r} is absent, while the "
            f"record states sha256={recorded_sha} over {recorded_size} bytes. "
            "A record outliving its checkpoint states the class of leaves "
            "that are gone, so it is refused rather than applied to whatever "
            "is written next.") from exc
    if got_sha != recorded_sha or got_size != recorded_size:
        raise ClassRecordStale(
            f"refusing to read the model class of {path!r} from the record "
            f"{class_record_path(path)!r}: the record describes leaves with "
            f"sha256={recorded_sha} ({recorded_size} bytes), and the "
            f"checkpoint on disk is sha256={got_sha} ({got_size} bytes). The "
            "record and the checkpoint are separate files; a write "
            "interrupted between them leaves exactly this pair, and the "
            "record's class is not the class of these leaves. Rewrite the "
            "checkpoint, or delete both and retrain -- nothing on disk states "
            "what this .eqx is.")


def require_matching_log_transform(checkpoint_path, record, want_class, *,
                                   what: str = "trained checkpoint") -> None:
    """Refuse a checkpoint written under the other descriptor log transform,
    where the record says which one that was.

    ``descriptor_log_transform`` is a static field of both networks and of the
    cusp descriptor, and it changes no parameter shape, so a checkpoint
    written with the compression on deserialises into a skeleton with it off
    in silence -- and the model that comes out reads identical leaves through
    a different map. On the legacy coordinates the MLPs are fed
    ``(1 - exp(-x^2)) log(x + 1)`` in place of the raw reduced gradient and
    r_s (``networks.AlecGGA_XNet._core``, ``AlecGGA_CNet._core``); on EVERY
    coordinate set the cusp descriptor's second column is log-compressed
    before its tanh (``config.ArchitectureConfig.materialize_descriptors``
    hands the flag to ``features.compute_cusp_descriptor``), so the ``dfs``
    coordinates, which bypass the network transform, do not make the field
    inert for an architecture carrying that descriptor.

    Compared only where BOTH sides state it. A record written before the
    field existed states nothing, and is accepted by a skeleton of either
    value: those are the records standing beside every checkpoint written
    before this check, and how they load is unchanged. A ``want_class``
    that states nothing is a caller that read the flag off nothing, which is
    not evidence either.

    Raises :class:`ModelClassMismatch`, naming both values.
    """
    recorded = (record.get(LOG_TRANSFORM_FIELD)
                if isinstance(record, dict) else None)
    wanted = (want_class.get(LOG_TRANSFORM_FIELD)
              if isinstance(want_class, dict) else None)
    if recorded is None or wanted is None:
        return
    if bool(recorded) != bool(wanted):
        raise ModelClassMismatch(
            f"refusing to load the {what} {os.fspath(checkpoint_path)!r}: it "
            f"was written with descriptor_log_transform={bool(recorded)}, and "
            f"the model being built has "
            f"descriptor_log_transform={bool(wanted)}. The flag changes what "
            "the networks read -- the log compression of the MLP inputs on "
            "the legacy coordinates, and the cusp descriptor's weighted-Z "
            "column on every coordinate set -- and changes no parameter "
            "shape, so loading across it would silently produce a model that "
            "is neither.")


def require_matching_class(checkpoint_path, want_class, *,
                           what: str = "trained checkpoint") -> dict:
    """Refuse to load a checkpoint written as another model class.

    ``want_class`` is the class of the skeleton about to be filled, from
    :func:`model_class_of_arch` (a spec's arch) or
    :func:`model_class_of_model` (a built skeleton). Returns the two fields
    every record states -- :data:`LEGACY_CLASS` when there is no record -- so
    a caller can log what it accepted.

    The record is held to the ``.eqx`` on disk BEFORE the classes are
    compared (:func:`require_matching_digest`): a record that does not
    describe these leaves is no evidence about them, whichever class it names,
    so the same refusal answers a skeleton of either class. Raises
    :class:`ClassRecordStale` there.

    Raises :class:`ModelClassMismatch` (a ``ValueError``) when the record
    names another class, and when there is NO record and the skeleton is not
    the legacy class: nothing then states what the anchored model would be
    loading, and the leaves do not reveal it. The descriptor log transform is
    compared after those two fields agree, and only where the record states
    it (:func:`require_matching_log_transform`), so a record written before
    that field existed loads exactly as it did.
    """
    record = read_class_record(checkpoint_path)
    path = os.fspath(checkpoint_path)
    if record is None:
        if not is_legacy_class(want_class):
            raise ModelClassMismatch(
                f"refusing to load the {what} {path!r} into a model with "
                f"{describe_class(want_class)}: no model-class record "
                f"({class_record_path(os.path.basename(path))}) stands beside "
                "it, and the checkpoint's leaves do not reveal the class (the "
                "anchor and the coordinates are static fields with no "
                "parameters of their own). Every run that writes an anchored "
                "or dfs checkpoint writes the record with it, so a checkpoint "
                "without one was written by the unanchored legacy class.")
        return dict(LEGACY_CLASS)
    require_matching_digest(path, record)
    got_class = {
        "parent_anchor": bool(record.get("parent_anchor", False)),
        "descriptor_coordinates": str(
            record.get("descriptor_coordinates", "legacy")),
    }
    if got_class != {key: want_class.get(key) for key in got_class}:
        raise ModelClassMismatch(
            f"refusing to load the {what} {path!r}: it was written as "
            f"{describe_class(got_class)}, but the model being built is "
            f"{describe_class(want_class)}. The two are different model "
            "classes with identical parameter shapes; loading across them "
            "would silently produce a model that is neither.")
    require_matching_log_transform(path, record, want_class, what=what)
    return got_class


def load_trained_checkpoint(checkpoint_path, skeleton, *,
                            what: str = "trained checkpoint"):
    """Fill ``skeleton`` from the TRAINED checkpoint at ``checkpoint_path``,
    with the record checked first, and return the filled model.

    The one call every reader of a trained checkpoint outside this package's
    own loaders makes: :func:`require_matching_class` against the class the
    skeleton itself carries (:func:`model_class_of_model`), then
    ``eqx.tree_deserialise_leaves``. A bare deserialise is what the record
    cannot guard -- the anchor, the descriptor coordinates and the descriptor
    log transform change no parameter shape, so another class's leaves land in
    this skeleton with every array equal and nothing raising -- and the
    analysis scripts and
    notebook cells that read ``model.eqx``, ``model_best.eqx`` and
    ``model_val_best.eqx`` are exactly where such a load is not noticed,
    because what comes out of it is a plausible curve.

    A checkpoint with no record beside it is the legacy class and is accepted
    by a legacy skeleton, so the campaigns written before the anchor read as
    they did before.

    ``equinox`` is imported here rather than at module scope: this module is
    read by callers that only want the record.
    """
    import equinox as eqx

    require_matching_class(checkpoint_path, model_class_of_model(skeleton),
                           what=what)
    return eqx.tree_deserialise_leaves(os.fspath(checkpoint_path), skeleton)
