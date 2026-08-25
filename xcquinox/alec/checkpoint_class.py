"""The model class a TRAINED checkpoint was written as, recorded beside it.

The parent anchor (``ArchitectureConfig.parent_anchor``) and the descriptor
coordinates (``ArchitectureConfig.descriptor_coordinates``) are STATIC
properties of the networks: neither changes a parameter shape, so an
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
(``pretrain_metadata.json`` for the first five keys, the fidelity certificate
for the two after them)::

    parent_anchor              bool   the class, compared by the readers
    descriptor_coordinates     str    the class, compared by the readers
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

Only ``parent_anchor`` and ``descriptor_coordinates`` are compared as the
class: they are what "the model class" means, and they are the two a leaf
stream cannot reveal. ``sha256`` and ``size`` say which leaves the two refer
to. The rest states what wrote the file.

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

#: Appended to a checkpoint's own path to give its class record's path.
CLASS_RECORD_SUFFIX = ".class.json"

#: Appended to a class record's path to give the path a record is STAGED at
#: while the checkpoint it describes is still being written. Nothing reads it.
STAGED_RECORD_SUFFIX = ".tmp"

#: Read size for :func:`file_digest`. The whole checkpoint is never held in
#: memory, and one chunk covers every checkpoint this project writes.
DIGEST_CHUNK_BYTES = 1 << 20

#: The class of a checkpoint with no record: what everything written before
#: the anchor existed is.
LEGACY_CLASS = {"parent_anchor": False, "descriptor_coordinates": "legacy"}


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


def staged_class_record_path(checkpoint_path) -> str:
    """Where a record for ``checkpoint_path`` is staged before it is committed
    (:func:`stage_class_record`)."""
    return f"{class_record_path(checkpoint_path)}{STAGED_RECORD_SUFFIX}"


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
    """The model class ``arch`` states: the two fields the readers compare.

    The defaults are the legacy class, so an arch-like object from before the
    fields existed answers as the class it is.
    """
    return {
        "parent_anchor": bool(getattr(arch, "parent_anchor", False)),
        "descriptor_coordinates": str(
            getattr(arch, "descriptor_coordinates", "legacy")),
    }


def model_class_of_model(model) -> dict:
    """The model class a BUILT model carries, read off its exchange network's
    static fields (``AlecGGA_XNet.parent``, ``.descriptor_coordinates``).

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
    }


def is_legacy_class(model_class) -> bool:
    """Whether ``model_class`` is the unanchored legacy class."""
    return (not model_class["parent_anchor"]
            and model_class["descriptor_coordinates"] == "legacy")


def describe_class(model_class) -> str:
    """One line naming a class, in the loaders' shared vocabulary."""
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
    """
    staged = staged_class_record_path(checkpoint_path)
    with open(staged, "w") as f:
        json.dump(class_record(arch, sha256=sha256, size=size), f, indent=2)
        f.flush()
        os.fsync(f.fileno())
    return staged


def commit_class_record(checkpoint_path) -> str:
    """Move the record staged for ``checkpoint_path`` into place, atomically
    (``os.replace``), and return the record's path."""
    path = class_record_path(checkpoint_path)
    os.replace(staged_class_record_path(checkpoint_path), path)
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
    stage_class_record(checkpoint_path, arch, sha256=sha256, size=size)
    path = commit_class_record(checkpoint_path)
    fsync_dir_of(path)
    return path


def discard_staged_record(checkpoint_path) -> None:
    """Remove a staged record whose checkpoint never landed; absent is fine.

    The record on disk is left exactly as it was, which is what keeps a failed
    write from re-labelling the checkpoint that survived it.
    """
    try:
        os.remove(staged_class_record_path(checkpoint_path))
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


def require_matching_class(checkpoint_path, want_class, *,
                           what: str = "trained checkpoint") -> dict:
    """Refuse to load a checkpoint written as another model class.

    ``want_class`` is the class of the skeleton about to be filled, from
    :func:`model_class_of_arch` (a spec's arch) or
    :func:`model_class_of_model` (a built skeleton). Returns the recorded
    class -- :data:`LEGACY_CLASS` when there is no record -- so a caller can
    log what it accepted.

    The record is held to the ``.eqx`` on disk BEFORE the classes are
    compared (:func:`require_matching_digest`): a record that does not
    describe these leaves is no evidence about them, whichever class it names,
    so the same refusal answers a skeleton of either class. Raises
    :class:`ClassRecordStale` there.

    Raises :class:`ModelClassMismatch` (a ``ValueError``) when the record
    names another class, and when there is NO record and the skeleton is not
    the legacy class: nothing then states what the anchored model would be
    loading, and the leaves do not reveal it.
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
    if got_class != want_class:
        raise ModelClassMismatch(
            f"refusing to load the {what} {path!r}: it was written as "
            f"{describe_class(got_class)}, but the model being built is "
            f"{describe_class(want_class)}. The two are different model "
            "classes with identical parameter shapes; loading across them "
            "would silently produce a model that is neither.")
    return got_class
