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
the run layout; and a checkpoint copied out on its own -- the surgical
single-file pull ``cluster.sync`` supports, the local re-evaluation workflow
-- carries its class with it.

Payload, in the vocabulary of the records it sits beside
(``pretrain_metadata.json`` for the first five keys, the fidelity certificate
for the last two)::

    parent_anchor              bool   the class, compared by the readers
    descriptor_coordinates     str    the class, compared by the readers
    arch_name                  str    provenance
    meta_gga                   bool   provenance (ArchitectureConfig.is_meta_gga)
    use_polarized_correlation  bool   provenance (a shape-CHANGING flag, which
                                      the loaders check on the network itself)
    parent                     str    the parent functional when anchored, else
                                      null (``parents.parent_for_arch``)
    xcquinox_version           str    provenance

Only the first two are compared: they are what "the model class" means, and
they are the two a leaf stream cannot reveal. The rest states what wrote the
file.

A checkpoint with NO record beside it is a legacy checkpoint -- unanchored, on
the legacy coordinates -- because every run that writes an anchored or a
``dfs`` checkpoint writes the record in the same call that writes the ``.eqx``
(:func:`write_class_record`). A legacy skeleton therefore accepts it and any
other skeleton refuses it, which is the rule
``train._require_matching_model_class`` applies to a pretrain directory with
no metadata.
"""
from __future__ import annotations

import json
import os

#: Appended to a checkpoint's own path to give its class record's path.
CLASS_RECORD_SUFFIX = ".class.json"

#: The class of a checkpoint with no record: what everything written before
#: the anchor existed is.
LEGACY_CLASS = {"parent_anchor": False, "descriptor_coordinates": "legacy"}


def class_record_path(checkpoint_path) -> str:
    """The class record's path for the checkpoint at ``checkpoint_path``."""
    return f"{os.fspath(checkpoint_path)}{CLASS_RECORD_SUFFIX}"


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


def class_record(arch) -> dict:
    """The full payload written beside a checkpoint built from ``arch``.

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
    })
    return record


def write_class_record(checkpoint_path, arch) -> str:
    """Write the class record for the checkpoint at ``checkpoint_path``.

    THE writer: every trained ``.eqx`` gets its record from this one call, in
    the same code path that serialises the leaves, so the two cannot drift.
    Returns the record's path.
    """
    path = class_record_path(checkpoint_path)
    with open(path, "w") as f:
        json.dump(class_record(arch), f, indent=2)
    return path


def remove_class_record(checkpoint_path) -> None:
    """Delete the class record beside ``checkpoint_path``; absent is fine.

    Called wherever the checkpoint itself is deleted (the resume set on
    completion), so a stale record can never outlive its ``.eqx``.
    """
    try:
        os.remove(class_record_path(checkpoint_path))
    except FileNotFoundError:
        pass


def read_class_record(checkpoint_path):
    """The record beside ``checkpoint_path``, or ``None`` when there is none.

    A record that cannot be read or parsed raises ``ValueError``: an
    unreadable record is not the same as no record, and answering "legacy"
    for it would be a guess at the one thing this file exists to state.
    """
    path = class_record_path(checkpoint_path)
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"the model-class record {path!r} beside checkpoint "
            f"{os.fspath(checkpoint_path)!r} could not be read: {exc}") from exc


def require_matching_class(checkpoint_path, want_class, *,
                           what: str = "trained checkpoint") -> dict:
    """Refuse to load a checkpoint written as another model class.

    ``want_class`` is the class of the skeleton about to be filled, from
    :func:`model_class_of_arch` (a spec's arch) or
    :func:`model_class_of_model` (a built skeleton). Returns the recorded
    class -- :data:`LEGACY_CLASS` when there is no record -- so a caller can
    log what it accepted.

    Raises ``ValueError`` when the record names another class, and when there
    is NO record and the skeleton is not the legacy class: nothing then states
    what the anchored model would be loading, and the leaves do not reveal it.
    """
    record = read_class_record(checkpoint_path)
    path = os.fspath(checkpoint_path)
    if record is None:
        if not is_legacy_class(want_class):
            raise ValueError(
                f"refusing to load the {what} {path!r} into a model with "
                f"{describe_class(want_class)}: no model-class record "
                f"({class_record_path(os.path.basename(path))}) stands beside "
                "it, and the checkpoint's leaves do not reveal the class (the "
                "anchor and the coordinates are static fields with no "
                "parameters of their own). Every run that writes an anchored "
                "or dfs checkpoint writes the record with it, so a checkpoint "
                "without one was written by the unanchored legacy class.")
        return dict(LEGACY_CLASS)
    got_class = {
        "parent_anchor": bool(record.get("parent_anchor", False)),
        "descriptor_coordinates": str(
            record.get("descriptor_coordinates", "legacy")),
    }
    if got_class != want_class:
        raise ValueError(
            f"refusing to load the {what} {path!r}: it was written as "
            f"{describe_class(got_class)}, but the model being built is "
            f"{describe_class(want_class)}. The two are different model "
            "classes with identical parameter shapes; loading across them "
            "would silently produce a model that is neither.")
    return got_class
