"""xcquinox.alec.cluster.sync — pure helpers for the ``pull`` subcommand.

The ``pull`` subcommand of ``python -m xcquinox.alec.cluster`` rsyncs a sweep
run dir from the cluster back to a local results tree. This module owns the
*pure* parts of that workflow:

  - :func:`build_rsync_command` — assemble the exact ``rsync`` argv for a
    (host, remote-root, local-root, run-id, profile, dry-run) tuple, including
    the right ``--filter='. <pkg>/filters/<profile>.filter'`` invocation.
  - :func:`resolve_run_id`      — normalize the user's ``run_id`` argument:
    pass a well-formed ``run_YYYYmmddTHHMMSSZ`` stamp through unchanged; turn
    the literal ``"latest"`` into the newest such stamp under ``remote_root``
    (resolved via an injected ``ssh_runner`` callable so tests stay offline).

All subprocess / SSH / filesystem side effects live in ``__main__.cmd_pull``;
this module is import-clean (no ``subprocess`` / ``socket`` / network I/O) and
thus trivially unit-testable.

The filter files (``filters/summaries.filter`` and ``filters/full.filter``)
ship as package data — see ``pyproject.toml`` ``[tool.setuptools.package-data]``
and the contract comments at the top of each filter file.
"""
from __future__ import annotations

import posixpath
import re
from importlib import resources
from pathlib import Path
from typing import Callable, Iterable, Sequence

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Profiles understood by :func:`build_rsync_command`. Each name maps 1:1 to a
#: file ``filters/<name>.filter`` shipped with this package.
VALID_PROFILES: tuple[str, ...] = ("summaries", "full")

#: Pattern a run-id must match — ``run_YYYYmmddTHHMMSSZ`` (UTC stamp, matches
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
    indicates a packaging bug — the test suite covers this).
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
) -> list[str]:
    """Build the argv that the caller will hand to :func:`subprocess.run`.

    Pure: no I/O is performed beyond resolving the filter-file path (which is
    a packaged read-only lookup). The returned list is ready to invoke; the
    caller is responsible for ``mkdir -p`` on the full local destination
    (which is ``<local_root>/<category>/<run_id>/`` when ``category`` is
    non-empty) before running it — rsync's ``--mkpath`` requires rsync >= 3.2
    and is not available on every cluster image.

    Source spec
    -----------
    When ``host`` is a non-empty string the source is
    ``<host>:<remote_root>/<category>/<run_id>/`` (SSH transport). When
    ``host`` is the empty string the source is the plain local path
    ``<remote_root>/<category>/<run_id>/`` — this is the "local-to-local"
    mode the end-to-end canary test in ``test_cluster_sync.py`` uses to
    exercise the filter file against a fixture tree without needing SSH.

    The destination mirrors the category layout: it is
    ``<local_root>/<category>/<run_id>/``. When ``category=""`` the
    behavior reduces exactly to the pre-category contract:
    ``<host>:<remote_root>/<run_id>/`` -> ``<local_root>/<run_id>/``.

    Trailing slashes
    ----------------
    Both source and destination paths end in ``/`` (rsync semantics: "copy
    the *contents* of this directory into the destination").

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
        When True, ``--dry-run`` is appended so rsync only *reports* what it
        would do.
    extra_flags
        Extra rsync flags appended verbatim (after the standard set, before
        the source/destination args). Reserved for future use (e.g. a
        ``--bwlimit=...`` flag); not currently exposed on the CLI.

    Returns
    -------
    list[str]
        ``["rsync", -a, -v, -z, --partial, --info=progress2,
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

    argv: list[str] = ["rsync", *_BASE_FLAGS, f"--filter=. {filt}"]
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

    The ``-prune`` flag is critical — it stops descent into matched dirs so
    we do not accidentally pick up anything inside ``run_<UTC>Z/checkpoints``
    or similar.

    Returns a dict mapping the *relative* parent directory (the "category"
    you would pass to :func:`build_rsync_command`, e.g. ``"alpha_off/runs"``
    or ``""`` if the run lives directly under ``remote_root``) to the list of
    run-id basenames found there, sorted oldest-first (so ``groups[cat][-1]``
    is the latest run for that category).

    Stray paths the regex rejects (e.g. a file named ``runs.tar.gz``) are
    silently filtered out — same robustness story as :func:`resolve_run_id`.

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
        *unmatched* directories — bounded by the user's xcquinox scratch
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
            continue  # stray match — silently skip
        parent_abs = posixpath.dirname(line)
        if parent_abs == rr:
            category = ""
        elif parent_abs.startswith(rr + "/"):
            category = parent_abs[len(rr) + 1:]
        else:
            # outside the requested root (symlink? bind mount?) — ignore.
            continue
        groups.setdefault(category, []).append(basename)
    for cat in groups:
        # stamps sort lexicographically <=> chronologically (zero-padded ISO).
        groups[cat].sort()
    return groups


def format_ssh_stderr_tail(stderr: str, n: int = 3) -> str:
    """Return the last ``n`` non-blank lines of ssh stderr.

    Cluster login nodes commonly emit a multi-line compliance banner on every
    SSH connection (the Stony Brook SeaWulf banner is ~10 lines about
    AI-training restrictions). On a *failed* command the underlying tool's
    actual error (e.g. ``ls: cannot access ...: No such file or directory``)
    lands at the tail of stderr, *after* the banner. Showing only the last
    ``n`` non-blank lines drops the banner from view without suppressing
    real error output.

    Pure: no I/O. ``stderr`` may be empty or ``None``-equivalent (an empty
    string), in which case ``""`` is returned.
    """
    lines = [ln.rstrip() for ln in (stderr or "").splitlines() if ln.strip()]
    return "\n".join(lines[-n:])
