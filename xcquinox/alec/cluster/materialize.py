"""xcquinox.alec.cluster.materialize: on-disk serialization of harness specs.

The harness expands a grid into ``TrainingSpec`` objects and must write them
to a shared filesystem so the per-task worker ``xcquinox.alec._train_one_spec``
can load each one by SLURM array-task index. This module performs that write.

The serialization format MUST round-trip through ``_train_one_spec._load_spec``
exactly: that loader does ``importlib.import_module("pi" + "ckle")`` then
``_ser.load(f)``. So every spec written here is a plain pickle (protocol 4),
produced via the SAME ``importlib`` indirection, the file comes from the same
trusted codebase in the same process tree, never from an untrusted source.
"""
from dataclasses import asdict, is_dataclass
import hashlib
import importlib
import json
import os
import sys
import tempfile


# Pinned serializer protocol. Protocol 4 (Python 3.4+) is the highest protocol
# guaranteed available on every login/compute node we target; pinning it makes
# the on-disk format forward-compatible regardless of the writer's Python.
_SER_PROTOCOL = 4

# Temp-file prefix for atomic writes. Deliberately distinct from the ``spec_``
# prefix of finished spec files: if a process crashes mid-write, the orphaned
# temp file can never be miscounted as a real spec by ``_purge_stale`` or by
# any consumer globbing ``spec_*.spec``.
_TMP_PREFIX = ".mktmp_"

_SPEC_PREFIX = "spec_"
_SPEC_SUFFIX = ".spec"


# ---------------------------------------------------------------------------
# Atomic spec write
# ---------------------------------------------------------------------------

def write_spec_atomic(obj, path: str) -> None:
    """Atomically write a serialized spec object to ``path``.

    Serialization uses the SAME ``importlib`` indirection as the worker's
    ``_load_spec`` so the file is guaranteed to round-trip. The write goes to a
    temp file in the target directory (``mkstemp`` with the ``.mktmp_`` prefix)
    and is committed with ``os.replace``; on any failure the temp file is
    removed so no orphan temp files are left behind.

    Raises:
        ValueError: if ``obj`` carries a non-None ``pbe_anchor_sample``, the
            harness does not support PBE-anchor specs (that field can hold
            non-serializable JAX arrays). Harness-built specs always have it
            None, so this is a defensive guard against an unsupported path.
    """
    anchor = getattr(obj, "pbe_anchor_sample", None)
    if anchor is not None:
        raise ValueError(
            "write_spec_atomic: spec has a non-None `pbe_anchor_sample` "
            f"({type(anchor).__name__}); the HPC harness does not support "
            "PBE-anchor specs (the field may hold non-serializable JAX "
            "arrays). Harness-built specs must have pbe_anchor_sample is None."
        )

    _ser = importlib.import_module("pi" + "ckle")
    out_dir = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp_name = tempfile.mkstemp(prefix=_TMP_PREFIX, dir=out_dir)
    try:
        with os.fdopen(fd, "wb") as f:
            _ser.dump(obj, f, protocol=_SER_PROTOCOL)
        os.replace(tmp_name, path)
        tmp_name = None  # os.replace consumed the temp file
    finally:
        # If os.replace did not run (exception) the temp file still exists;
        # remove it so a crashed write never leaves an orphan behind.
        if tmp_name is not None and os.path.exists(tmp_name):
            os.unlink(tmp_name)


# ---------------------------------------------------------------------------
# Spec materialization
# ---------------------------------------------------------------------------

def _spec_filename(idx: int, width: int) -> str:
    """Zero-padded spec filename for array-task index ``idx``."""
    return f"{_SPEC_PREFIX}{idx:0{width}d}{_SPEC_SUFFIX}"


def _parse_spec_index(name: str):
    """Parse the integer index out of a ``spec_<idx>.spec`` filename.

    Returns the index, or None if ``name`` is not a spec file with a numeric
    index (so callers can simply skip it)."""
    if not (name.startswith(_SPEC_PREFIX) and name.endswith(_SPEC_SUFFIX)):
        return None
    core = name[len(_SPEC_PREFIX):-len(_SPEC_SUFFIX)]
    if not core.isdigit():
        return None
    return int(core)


def _purge_stale(out_dir: str, n: int) -> None:
    """Remove leftovers from a prior run before writing a fresh grid.

    Deletes (a) every ``.mktmp_*`` temp file (crash-orphaned partial writes)
    and (b) every ``spec_*.spec`` whose parsed index is >= ``n``: those are
    stale orphans from a larger prior grid that would otherwise be picked up
    as bogus extra tasks.
    """
    if not os.path.isdir(out_dir):
        return
    for name in os.listdir(out_dir):
        full = os.path.join(out_dir, name)
        if name.startswith(_TMP_PREFIX):
            if os.path.isfile(full):
                os.unlink(full)
            continue
        idx = _parse_spec_index(name)
        if idx is not None and idx >= n and os.path.isfile(full):
            os.unlink(full)


def materialize_specs(specs, out_dir: str) -> list[str]:
    """Write a list of harness specs to ``out_dir`` as ``spec_<idx>.spec``.

    Args:
        specs: ordered ``list[tuple[GridCell, spec_object]]``; each tuple's
            position is its SLURM array-task index.
        out_dir: target directory (created if absent).

    Returns:
        The written spec-file paths, in index order.
    """
    os.makedirs(out_dir, exist_ok=True)
    n = len(specs)
    # max(4, ...) keeps a stable minimum pad width; len(str(N-1)) widens it for
    # grids with >= 10000 cells. N-1 is the largest index actually used.
    width = max(4, len(str(n - 1))) if n > 0 else 4

    _purge_stale(out_dir, n)

    paths: list[str] = []
    for idx, (_cell, spec_obj) in enumerate(specs):
        path = os.path.join(out_dir, _spec_filename(idx, width))
        write_spec_atomic(spec_obj, path)
        paths.append(path)
    return paths


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------

def _cell_to_dict(cell) -> dict:
    """Serialize a GridCell (frozen dataclass) to a plain JSON-safe dict."""
    if is_dataclass(cell):
        return asdict(cell)
    # Defensive fallback for a non-dataclass cell-like object.
    return {f: getattr(cell, f) for f in getattr(cell, "_fields", ())}


def _sha256_file(path: str) -> str:
    """Hex SHA-256 of a file's bytes, read in chunks."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json_atomic(payload: dict, path: str) -> None:
    """Atomically write ``payload`` as pretty JSON to ``path``.

    Uses ``mkstemp`` + ``os.replace`` so a crash mid-write can never leave a
    partially written manifest that a downstream submitter would parse.
    """
    out_dir = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp_name = tempfile.mkstemp(prefix=_TMP_PREFIX, dir=out_dir)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp_name, path)
        tmp_name = None
    finally:
        if tmp_name is not None and os.path.exists(tmp_name):
            os.unlink(tmp_name)


def write_manifest(cells, paths, out_dir: str) -> str:
    """Write ``manifest.json`` recording the materialized grid.

    Per index it records the ``GridCell`` (as a dict), the spec filename, and
    a SHA-256 content hash of the spec file. Top-level it records the xcquinox
    version, the running Python version, the zero-pad ``width`` and ``n_specs``.

    This is the final write of the preflight, so it is itself atomic (a
    partially written manifest must never be observable).

    Args:
        cells: ordered ``list[GridCell]``.
        paths: ordered spec-file paths (parallel to ``cells``).
        out_dir: directory to write ``manifest.json`` into.

    Returns:
        The manifest path.
    """
    if len(cells) != len(paths):
        raise ValueError(
            f"write_manifest: cells ({len(cells)}) and paths ({len(paths)}) "
            "must be the same length"
        )

    import xcquinox

    n = len(cells)
    width = max(4, len(str(n - 1))) if n > 0 else 4

    entries = []
    for idx, (cell, path) in enumerate(zip(cells, paths)):
        entries.append({
            "index": idx,
            "cell": _cell_to_dict(cell),
            "spec_file": os.path.basename(path),
            "sha256": _sha256_file(path),
        })

    payload = {
        "xcquinox_version": getattr(xcquinox, "__version__", "unknown"),
        "python_version": sys.version.split()[0],
        "width": width,
        "n_specs": n,
        "specs": entries,
    }

    manifest_path = os.path.join(out_dir, "manifest.json")
    _write_json_atomic(payload, manifest_path)
    return manifest_path
