"""xcquinox.alec.cluster.sync: pure helpers for the ``pull`` subcommand.

The ``pull`` subcommand of ``python -m xcquinox.alec.cluster`` rsyncs a sweep
run dir from the cluster back to a local results tree. This module owns the
pure parts of that workflow:

  - :func:`build_rsync_command`: assemble the exact ``rsync`` argv for a
    (host, remote-root, local-root, run-id, profile, dry-run) tuple, including
    the right ``--filter='. <pkg>/filters/<profile>.filter'`` invocation.
  - :func:`resolve_run_id`: normalize the user's ``run_id`` argument:
    pass a well-formed ``run_YYYYmmddTHHMMSSZ`` stamp through unchanged; turn
    the literal ``"latest"`` into the newest such stamp under ``remote_root``
    (resolved via an injected ``ssh_runner`` callable so tests stay offline).

All subprocess / SSH / filesystem side effects live in ``__main__.cmd_pull``;
this module is import-clean (no ``subprocess`` / ``socket`` / network I/O) and
thus trivially unit-testable.

The filter files (``filters/summaries.filter`` and ``filters/full.filter``)
ship as package data, see ``pyproject.toml`` ``[tool.setuptools.package-data]``
and the contract comments at the top of each filter file.
"""
from __future__ import annotations

import posixpath
import re
from datetime import datetime, timezone
from importlib import resources
from pathlib import Path
from typing import Callable, Iterable, Sequence

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Profiles understood by :func:`build_rsync_command`. Each name maps 1:1 to a
#: file ``filters/<name>.filter`` shipped with this package.
VALID_PROFILES: tuple[str, ...] = ("summaries", "full")

#: Pattern a run-id must match, ``run_YYYYmmddTHHMMSSZ`` (UTC stamp, matches
#: ``__main__._utc_stamp``). Used by :func:`resolve_run_id` to reject typos
#: before they reach ``rsync``.
_RUN_ID_RE = re.compile(r"^run_\d{8}T\d{6}Z$")

#: Base rsync flags used for every profile and direction. ``-a`` preserves
#: timestamps/permissions; ``-v`` is operator-friendly; ``-z`` compresses the
#: JSON-heavy summaries profile and is a near-no-op on already-binary
#: ``.eqx`` files; ``--partial`` makes resuming after a dropped SSH session
#: cheap; ``--info=progress2`` shows a single-line progress bar (rsync >= 3.1).
_BASE_FLAGS: tuple[str, ...] = ("-a", "-v", "-z", "--partial", "--info=progress2")


# ---------------------------------------------------------------------------
# Filter-file lookup
# ---------------------------------------------------------------------------

def filter_file_path(profile: str) -> Path:
    """Resolve the absolute filesystem path of the rsync filter for ``profile``.

    Raises :class:`ValueError` if ``profile`` is not in :data:`VALID_PROFILES`,
    or :class:`FileNotFoundError` if the package data file is missing (which
    indicates a packaging bug, the test suite covers this).
    """
    if profile not in VALID_PROFILES:
        raise ValueError(
            f"unknown pull profile {profile!r} "
            f"(expected one of {VALID_PROFILES})"
        )
    # ``files()`` returns a Traversable; for an installed wheel or an editable
    # ``pip install -e .`` it resolves to a real filesystem path, which is
    # what rsync needs. (``as_file`` would handle the zipped-wheel edge case
    # but this package is never zip-installed.)
    pkg = resources.files("xcquinox.alec.cluster")
    path = Path(str(pkg / "filters" / f"{profile}.filter"))
    if not path.is_file():
        raise FileNotFoundError(
            f"filter file missing from package data: {path} "
            "(check pyproject.toml [tool.setuptools.package-data])"
        )
    return path


# ---------------------------------------------------------------------------
# Argv builder
# ---------------------------------------------------------------------------

def _join_run_path(root: str, category: str, run_id: str) -> str:
    """Posixpath-join ``<root>/<category>/<run_id>`` (category may be empty).

    Both the remote and the local destination use this so the local tree
    mirrors the remote category layout (avoids collisions when the same
    ``run_<UTC>Z`` stamp appears in two different categories). ``posixpath``
    is correct for both: the remote is always Linux, and the local POSIX
    machines we ship to also use ``/`` (Windows is not a supported caller).
    """
    parts = [root.rstrip("/")]
    if category:
        parts.append(category.strip("/"))
    parts.append(run_id)
    return posixpath.join(*parts)


def build_rsync_command(
    *,
    host: str,
    remote_root: str,
    local_root: str,
    run_id: str,
    category: str = "",
    profile: str = "summaries",
    dry_run: bool = False,
    extra_flags: Sequence[str] = (),
    spec_indices: Sequence[int] = (),
) -> list[str]:
    """Build the argv that the caller will hand to :func:`subprocess.run`.

    Pure: no I/O is performed beyond resolving the filter-file path (which is
    a packaged read-only lookup). The returned list is ready to invoke; the
    caller is responsible for ``mkdir -p`` on the full local destination
    (which is ``<local_root>/<category>/<run_id>/`` when ``category`` is
    non-empty) before running it, rsync's ``--mkpath`` requires rsync >= 3.2
    and is not available on every cluster image.

    Source spec
    -----------
    When ``host`` is a non-empty string the source is
    ``<host>:<remote_root>/<category>/<run_id>/`` (SSH transport). When
    ``host`` is the empty string the source is the plain local path
    ``<remote_root>/<category>/<run_id>/``: this is the "local-to-local"
    mode the end-to-end canary test in ``test_cluster_sync.py`` uses to
    exercise the filter file against a fixture tree without needing SSH.

    The destination mirrors the category layout: it is
    ``<local_root>/<category>/<run_id>/``. When ``category=""`` the
    behavior reduces exactly to the pre-category contract:
    ``<host>:<remote_root>/<run_id>/`` -> ``<local_root>/<run_id>/``.

    Trailing slashes
    ----------------
    Both source and destination paths end in ``/`` (rsync semantics: "copy
    the contents of this directory into the destination").

    Parameters
    ----------
    host
        SSH host (``"login.seawulf.stonybrook.edu"``, ``"seawulf"`` after
        ``~/.ssh/config`` aliasing) or ``""`` to source from the local
        filesystem.
    remote_root
        Base scratch directory on ``host`` (or locally, when ``host=""``).
        Per-sweep ``run_<UTC>Z`` dirs live under
        ``<remote_root>/<category>/``.
    local_root
        Directory on the caller's machine under which the (mirrored)
        ``<category>/<run_id>/`` tree is created.
    run_id
        Must match :data:`_RUN_ID_RE` (``run_YYYYmmddTHHMMSSZ``). Use
        :func:`resolve_run_id` to turn the literal ``"latest"`` into a real
        stamp before passing it here.
    category
        Optional path segment (possibly multi-level, e.g. ``"alpha_off/runs"``
        or ``"polarized/alpha_on"``) that selects an experiment-series
        subdirectory under ``remote_root``. Empty (the default) preserves the
        original behavior. Leading/trailing slashes are trimmed.
    profile
        Which packaged filter file to load. One of :data:`VALID_PROFILES`.
    dry_run
        When True, ``--dry-run`` is appended so rsync only reports what it
        would do.
    extra_flags
        Extra rsync flags appended verbatim (after the standard set, before
        the source/destination args). The ``pull`` CLI uses this to attach
        the SSH multiplexing transport (``("-e", "ssh -o ControlMaster=auto
        ...")``); it also suits e.g. a ``--bwlimit=...`` flag.
    spec_indices
        Optional list of per-spec indices to restrict the pull to. When
        non-empty, only the requested ``checkpoints/spec_<NNNN>/`` subtrees
        are pulled, every other ``spec_*`` directory under
        ``checkpoints/`` is excluded. Use this with ``profile="full"`` for
        the surgical "pull just these N specs' trained model.eqx" workflow
        described in ``hpcjobs/SEAWULF_RUNBOOK.md`` §10.5 (local test-set
        re-evaluation). Spec ids are zero-padded to width 4 to match the
        harness naming (``__main__._utc_stamp`` / preflight conventions).
        Each id must be ``>= 0`` (raises ``ValueError`` otherwise).

    Returns
    -------
    list[str]
        ``["rsync", -a, -v, -z, --partial, --info=progress2,
        (--include=/checkpoints/spec_NNNN/*** ...)?,
        --filter=. <abs filter path>, [--dry-run,] <src>, <dst>]``.
    """
    if not _RUN_ID_RE.match(run_id):
        raise ValueError(
            f"run_id must match {_RUN_ID_RE.pattern!r}, got {run_id!r}; "
            "pass 'latest' or a stamp from `ssh <host> ls <remote_root>`"
        )
    filt = filter_file_path(profile)  # also validates `profile`

    src_prefix = f"{host}:" if host else ""
    src = f"{src_prefix}{_join_run_path(remote_root, category, run_id)}/"
    dst = f"{_join_run_path(local_root, category, run_id)}/"

    argv: list[str] = ["rsync", *_BASE_FLAGS]

    # Per-spec narrowing, emit BEFORE the --filter so the includes win
    # the first-match race against the catch-all `- /checkpoints/spec_*`
    # exclude that follows them (rsync semantics: first matching rule).
    if spec_indices:
        for idx in spec_indices:
            if not isinstance(idx, int) or idx < 0:
                raise ValueError(
                    f"spec_indices entries must be non-negative ints, "
                    f"got {idx!r}"
                )
        # rsync needs to descend into /checkpoints/ first, then into each
        # named spec_NNNN/, then take everything inside (``***``).
        argv.append("--include=/checkpoints/")
        # The harness pads spec dir names to width max(4, len(str(n_specs-1))),
        # which depends on the grid size and is NOT known here (the pull command
        # does not load the remote manifest). Emit an include for every plausible
        # zero-pad width (4..8, i.e. grids up to 100M specs, far beyond any SLURM
        # array limit); only the width matching the grid hits a real dir, the
        # rest match nothing and are harmless. A single fixed ``:04d`` silently
        # pulled NOTHING once a grid had >9999 specs (dirs padded to width >=5).
        for idx in spec_indices:
            seen: set[str] = set()
            for width in range(4, 9):
                name = f"spec_{idx:0{width}d}"
                if name in seen:
                    continue            # idx already wider than `width`
                seen.add(name)
                argv.append(f"--include=/checkpoints/{name}/")
                argv.append(f"--include=/checkpoints/{name}/***")
        argv.append("--exclude=/checkpoints/spec_*")

    argv.append(f"--filter=. {filt}")
    if dry_run:
        argv.append("--dry-run")
    argv.extend(extra_flags)
    argv.extend([src, dst])
    return argv


# ---------------------------------------------------------------------------
# run_id resolution (``latest`` -> newest stamp under ``remote_root``)
# ---------------------------------------------------------------------------

def resolve_run_id(
    run_id: str,
    *,
    ssh_runner: Callable[[Sequence[str]], Iterable[str]],
    remote_root: str,
    category: str = "",
) -> str:
    """Normalize ``run_id`` and (if it is the literal ``"latest"``) resolve it.

    The injected ``ssh_runner`` lets tests stub the SSH call. In production
    ``__main__.cmd_pull`` wraps :func:`subprocess.run`:

    .. code-block:: python

        ssh_runner=lambda argv: subprocess.run(
            ["ssh", host, *argv], check=True, capture_output=True, text=True,
        ).stdout.splitlines()

    "Latest" is resolved by ``ls -1tr <remote_root>/<category>``: ``ls -t``
    sorts by mtime descending, ``-r`` reverses to ascending, so the last line
    is the most recently modified entry. We additionally filter the listing
    to lines matching :data:`_RUN_ID_RE` so stray files (e.g. an old
    ``runs.tar.gz`` snapshot) cannot trick us.

    Parameters
    ----------
    run_id
        Either ``"latest"`` (triggers the ``ls``) or a well-formed stamp.
    ssh_runner
        Injected SSH wrapper; see usage above.
    remote_root
        Base scratch directory on the cluster.
    category
        Optional path segment under ``remote_root`` that directly contains
        the ``run_<UTC>Z`` dirs. Empty (the default) restores the original
        behavior of listing ``remote_root`` itself.

    Raises
    ------
    ValueError
        If a non-"latest" ``run_id`` does not match the stamp pattern, or if
        ``"latest"`` is requested but no ``run_*`` entry is found.
    """
    if run_id == "latest":
        ls_path = remote_root.rstrip("/")
        if category:
            ls_path = posixpath.join(ls_path, category.strip("/"))
        entries = list(ssh_runner(["ls", "-1tr", ls_path]))
        candidates = [e.strip() for e in entries if _RUN_ID_RE.match(e.strip())]
        if not candidates:
            raise ValueError(
                f"no run_<UTC>Z entries found under {ls_path!r} "
                f"(ssh ls returned {len(entries)} non-matching lines); "
                "is your --remote-root and --category correct? "
                "Try `python -m xcquinox.alec.cluster list-runs` to "
                "discover what's actually there."
            )
        return candidates[-1]

    if not _RUN_ID_RE.match(run_id):
        raise ValueError(
            f"run_id must be 'latest' or match {_RUN_ID_RE.pattern!r}, "
            f"got {run_id!r}"
        )
    return run_id


# ---------------------------------------------------------------------------
# discovery (``list-runs``) and stderr formatting
# ---------------------------------------------------------------------------

def discover_runs(
    *,
    ssh_runner: Callable[[Sequence[str]], Iterable[str]],
    remote_root: str,
    max_depth: int = 5,
) -> dict[str, list[str]]:
    """Walk ``remote_root`` for ``run_<UTC>Z`` dirs, grouped by category.

    Issues a single ``find`` over SSH:

    .. code-block:: bash

        find <remote_root> -mindepth 1 -maxdepth <N> -type d \\
             -name 'run_*Z' -prune -print

    The ``-prune`` flag is critical, it stops descent into matched dirs so
    we do not accidentally pick up anything inside ``run_<UTC>Z/checkpoints``
    or similar.

    Returns a dict mapping the relative parent directory (the "category"
    you would pass to :func:`build_rsync_command`, e.g. ``"alpha_off/runs"``
    or ``""`` if the run lives directly under ``remote_root``) to the list of
    run-id basenames found there, sorted oldest-first (so ``groups[cat][-1]``
    is the latest run for that category).

    Stray paths the regex rejects (e.g. a file named ``runs.tar.gz``) are
    silently filtered out, same robustness story as :func:`resolve_run_id`.

    Parameters
    ----------
    ssh_runner
        Injected SSH wrapper; see :func:`resolve_run_id`.
    remote_root
        Base scratch directory; the same value you pass to
        :func:`build_rsync_command` / :func:`resolve_run_id`.
    max_depth
        Maximum directory levels to descend below ``remote_root``. The
        deepest production layout currently shipped is
        ``polarized/<axis>/runs/run_<UTC>Z`` (4 levels), so the default of
        5 leaves one level of headroom for future categorization (e.g. an
        enclosing experiment-batch segment) without forcing every caller to
        override. ``-prune`` short-circuits descent into matched run dirs,
        so the cost of a higher cap is just ``find`` walking through
        unmatched directories, bounded by the user's xcquinox scratch
        root.

    Returns
    -------
    dict[str, list[str]]
        ``{category: [run_id_oldest, ..., run_id_latest]}``. Empty when no
        run dirs are found.
    """
    if max_depth < 1:
        raise ValueError(f"max_depth must be >= 1, got {max_depth}")
    rr = remote_root.rstrip("/")
    find_args = [
        "find", rr,
        "-mindepth", "1",
        "-maxdepth", str(max_depth),
        "-type", "d",
        "-name", "run_*Z",
        "-prune", "-print",
    ]
    lines = list(ssh_runner(find_args))
    groups: dict[str, list[str]] = {}
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        basename = posixpath.basename(line)
        if not _RUN_ID_RE.match(basename):
            continue  # stray match, silently skip
        parent_abs = posixpath.dirname(line)
        if parent_abs == rr:
            category = ""
        elif parent_abs.startswith(rr + "/"):
            category = parent_abs[len(rr) + 1:]
        else:
            # outside the requested root (symlink? bind mount?), ignore.
            continue
        groups.setdefault(category, []).append(basename)
    for cat in groups:
        # stamps sort lexicographically <=> chronologically (zero-padded ISO).
        groups[cat].sort()
    return groups


# ---------------------------------------------------------------------------
# multi-run pull (``pull auto``): activity discovery, filter transform, argv
# ---------------------------------------------------------------------------

def run_stamp_datetime(run_id: str):
    """The UTC datetime encoded in a ``run_<UTC>Z`` id, or ``None``.

    Display-only helper (roster age annotation in ``pull auto``); selection
    is by file activity, never by this stamp. Returns ``None`` both for
    non-stamp names and for regex-valid but calendar-invalid stamps
    (``run_20260230T120000Z``), so a stray dir name can never abort a pull.
    """
    if not _RUN_ID_RE.match(run_id):
        return None
    try:
        return datetime.strptime(
            run_id, "run_%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def ssh_control_opts(persist_seconds: int) -> tuple[str, ...]:
    """SSH option words enabling connection multiplexing.

    The first connection (discovery) authenticates and becomes the master;
    every later connection in the batch rides its socket, so an N-run pull
    costs ONE interactive verification. The socket path is expressed with
    ssh's OWN tokens -- ``%d`` (local user's home directory) and ``%C`` (a
    fixed-length hash of the connection identity; the expanded path is 65
    chars, inside the sockaddr_un limit) -- so the option string never
    contains a space even when ``$HOME`` does: these words also travel
    inside rsync's ``-e "ssh ..."`` value, which rsync word-splits.
    ``ControlMaster=auto`` composes with any user config: it reuses an
    existing master or becomes one. The caller is responsible for the
    ``~/.ssh`` mkdir (ssh silently skips multiplexing when the socket dir
    cannot be opened).

    ``persist_seconds`` must be >= 1: ssh defines ``ControlPersist=0`` as
    "persist forever", the OPPOSITE of disable, so 0 is refused here --
    disabling multiplexing is the caller's ``--no-control-master``.
    """
    persist = int(persist_seconds)
    if persist < 1:
        raise ValueError(
            f"persist_seconds must be >= 1 (ssh treats ControlPersist=0 as "
            f"'persist forever'), got {persist_seconds!r}")
    return ("-o", "ControlMaster=auto",
            "-o", "ControlPath=%d/.ssh/xcq-cm-%C",
            "-o", f"ControlPersist={persist}")


def discover_runs_with_activity(
    *,
    ssh_runner: Callable[[Sequence[str]], Iterable[str]],
    remote_root: str,
    active_within_epoch: int | None,
    max_depth: int = 5,
) -> dict[str, list[tuple[str, bool]]]:
    """Like :func:`discover_runs`, with a per-run activity tag, in ONE shot.

    A run is ACTIVE when any file under it has mtime at or after
    ``active_within_epoch`` (Unix seconds, UTC). Activity, not the name
    stamp, is the selection signal: a long-running campaign's run dir can be
    older than any cutoff while its evaluations are still being written, and
    a freshly created run's own writes are in-window mtimes, so the activity
    rule strictly supersets any stamp rule.

    Mechanically the discovery ``find`` gains a per-run predicate::

        -exec sh -c 'if find "$1" -newermt @<epoch> -print -quit \
                        | grep -q . ; then echo "A $1"; \
                        else echo "I $1"; fi' _ {} ;

    The inner ``find`` short-circuits at the first in-window file in
    traversal order -- an active run usually stops within a few entries,
    though a run whose only fresh writes sit deep in the tree is walked up
    to that point; a dead run pays one full metadata walk per invocation
    (1e4-1e5 entries for a large training run). The ``grep -q``
    is load-bearing: ``find`` exits 0 whether or not anything matched. The
    cutoff travels as ``@<epoch>`` (GNU find), immune to remote-timezone
    interpretation. NOTE the payload reaches the remote shell only through
    a runner that quotes each word (``__main__._make_ssh_lines``); a
    space-joining runner would splice the script into the find argv.

    ``active_within_epoch=None`` issues the plain :func:`discover_runs`
    argv byte-identically and tags every run active (the ``--days 0``
    no-horizon path).

    Returns ``{category: [(run_id, active), ...]}`` sorted oldest-first,
    with the same stray-path filtering as :func:`discover_runs`.
    """
    if active_within_epoch is None:
        groups = discover_runs(ssh_runner=ssh_runner, remote_root=remote_root,
                               max_depth=max_depth)
        return {cat: [(rid, True) for rid in rids]
                for cat, rids in groups.items()}
    if max_depth < 1:
        raise ValueError(f"max_depth must be >= 1, got {max_depth}")
    epoch = int(active_within_epoch)
    if epoch < 0:
        raise ValueError(f"active_within_epoch must be >= 0, got {epoch}")
    rr = remote_root.rstrip("/")
    script = (f'if find "$1" -newermt @{epoch} -print -quit | grep -q . ; '
              f'then echo "A $1"; else echo "I $1"; fi')
    find_args = [
        "find", rr,
        "-mindepth", "1",
        "-maxdepth", str(max_depth),
        "-type", "d",
        "-name", "run_*Z",
        "-prune",
        "-exec", "sh", "-c", script, "_", "{}", ";",
    ]
    lines = list(ssh_runner(find_args))
    groups: dict[str, list[tuple[str, bool]]] = {}
    for raw in lines:
        line = raw.strip()
        if not line or " " not in line:
            continue
        tag, path = line.split(" ", 1)
        if tag not in ("A", "I"):
            continue
        basename = posixpath.basename(path)
        if not _RUN_ID_RE.match(basename):
            continue
        parent_abs = posixpath.dirname(path)
        if parent_abs == rr:
            category = ""
        elif parent_abs.startswith(rr + "/"):
            category = parent_abs[len(rr) + 1:]
        else:
            continue
        groups.setdefault(category, []).append((basename, tag == "A"))
    for cat in groups:
        groups[cat].sort()
    return groups


#: Anchored include rule in a packaged filter file: ``+ /<path>``, where the
#: path may end in ``/`` (directory descent) or ``***`` (whole subtree).
_ANCHORED_INCLUDE_RE = re.compile(r"^\+ (/\S+)$")


def _checked_run_paths(run_paths: Sequence[str]) -> list[str]:
    """Normalize and validate run paths for the multi-run transfer."""
    paths = [str(p).strip("/") for p in run_paths]
    if not paths:
        raise ValueError("run_paths must be non-empty")
    for p in paths:
        if not p or any(ch in p for ch in " \t\n'\"`"):
            raise ValueError(
                f"unsafe or empty run path {p!r} (rsync word-splits remote "
                "args; whitespace and quote characters are refused)")
    return paths


def build_multi_filter(packaged_text: str, run_paths: Sequence[str]) -> str:
    """Prefix-expand a packaged run-dir-anchored filter for ``rsync -R``.

    The packaged profiles anchor their rules at the RUN DIR as transfer
    root. A multi-source ``rsync -R`` transfer is rooted at the remote root
    instead, so each anchored rule ``+ /X`` is re-emitted per selected run
    path R as ``+ /R/X`` (order and trailing ``/`` / ``***`` preserved),
    preceded by the ancestor descent includes rsync needs to reach R's
    implied directories, and followed by ONE terminal ``- *``. The packaged
    file stays the single source of truth for WHAT a profile carries.

    Accepted packaged forms: ``# comment``, blank, anchored include
    ``+ /...`` (including the full profile's single ``+ /***``), and one
    optional terminal ``- *`` (the summaries profile carries it, full does
    not; this transform always appends its own). Any other rule raises
    ``ValueError`` naming the line, so a future filter edit that breaks the
    transform assumption fails loudly instead of silently mis-pulling.
    """
    paths = _checked_run_paths(run_paths)
    includes: list[str] = []
    seen_terminal = False
    for lineno, raw in enumerate(packaged_text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if seen_terminal:
            raise ValueError(
                f"filter rule after the terminal '- *' at line {lineno}: "
                f"{raw!r}")
        if line == "- *":
            seen_terminal = True
            continue
        m = _ANCHORED_INCLUDE_RE.match(line)
        if m is None:
            raise ValueError(
                f"unsupported filter rule at line {lineno}: {raw!r} (the "
                "multi-run transform accepts comments, blanks, anchored "
                "includes '+ /...', and one terminal '- *')")
        includes.append(m.group(1))
    out: list[str] = []
    seen_dirs: set[str] = set()
    for p in paths:
        parts = p.split("/")
        for k in range(1, len(parts) + 1):
            rule = "+ /" + "/".join(parts[:k]) + "/"
            if rule not in seen_dirs:
                seen_dirs.add(rule)
                out.append(rule)
    for inc in includes:
        for p in paths:
            out.append(f"+ /{p}{inc}")
    out.append("- *")
    return "\n".join(out) + "\n"


def build_multi_rsync_command(
    *,
    host: str,
    remote_root: str,
    local_root: str,
    run_paths: Sequence[str],
    filter_path: str,
    dry_run: bool = False,
    extra_flags: Sequence[str] = (),
) -> list[str]:
    """Argv for the ONE multi-source pull (``pull auto``).

    Sources use rsync's ``--relative`` (``-R``) ``/./`` marker, so every
    selected run lands at ``<local_root>/<run_path>`` -- the same mirror
    :func:`build_rsync_command` produces for
    ``category=dirname(run_path)``. All sources name the same host, so
    rsync serves the whole batch over ONE connection. ``filter_path`` is
    the :func:`build_multi_filter` output written to a file (the caller
    owns its lifecycle). Base flags are shared with
    :func:`build_rsync_command` (:data:`_BASE_FLAGS`), so transfer behavior
    cannot drift between the two builders. The caller is responsible for
    ``mkdir -p`` on ``local_root``: rsync does not create multi-level
    destination roots (rc=11).
    """
    paths = _checked_run_paths(run_paths)
    for p in paths:
        if not _RUN_ID_RE.match(posixpath.basename(p)):
            raise ValueError(
                f"run path {p!r} does not end in a run_<UTC>Z stamp")
    rr = remote_root.rstrip("/")
    argv: list[str] = ["rsync", *_BASE_FLAGS, "-R",
                       f"--filter=. {filter_path}"]
    if dry_run:
        argv.append("--dry-run")
    argv.extend(extra_flags)
    prefix = f"{host}:" if host else ""
    for p in paths:
        argv.append(f"{prefix}{rr}/./{p}")
    argv.append(local_root.rstrip("/") + "/")
    return argv


def format_ssh_stderr_tail(stderr: str, n: int = 3) -> str:
    """Return the last ``n`` non-blank lines of ssh stderr.

    Cluster login nodes commonly emit a multi-line compliance banner on every
    SSH connection (the Stony Brook SeaWulf banner is ~10 lines about
    AI-training restrictions). On a failed command the underlying tool's
    actual error (e.g. ``ls: cannot access ...: No such file or directory``)
    lands at the tail of stderr, after the banner. Showing only the last
    ``n`` non-blank lines drops the banner from view without suppressing
    real error output.

    Pure: no I/O. ``stderr`` may be empty or ``None``-equivalent (an empty
    string), in which case ``""`` is returned.
    """
    lines = [ln.rstrip() for ln in (stderr or "").splitlines() if ln.strip()]
    return "\n".join(lines[-n:])
