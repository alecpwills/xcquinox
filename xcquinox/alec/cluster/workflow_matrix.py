"""Per-architecture workflow matrix: the harness stage sequence at a tiny
identity, once per registered architecture.

SPEC_pretrain_fidelity_program.md 3.4 requires that, before any campaign YAML
is rendered, every architecture in the registry be driven through the harness
stage sequence -- datagen -> pretrain -> certificate -> preflight -> train
(two cells, subset sizes 1 and 2) -> eval -> validate_run -- at def2-svp /
grid level 1 against the cached subset ledger and CCSD references, so a wiring
defect surfaces in minutes instead of inside a queued campaign. Nothing the
matrix produces is a physics result.

This module carries that identity: the checked-in template
(``examples/workflow_matrix_template.yaml``), the renderer that writes one
architecture's grid config, and the staging of the cached inputs the identity
consumes.

Prerequisite: both cached inputs are UNTRACKED. ``.gitignore`` excludes
``notebooks/checkpoints_step7/``, so a fresh clone, a git worktree and the
cluster checkout carry neither the CCSD references nor the subset ledger; that
directory has to be staged into the tree from a machine that holds it before
the matrix can run. Its absence raises :class:`CachedInputsMissing`, naming the
path.
"""
from __future__ import annotations

import os
import re
import shutil
from pathlib import Path

from xcquinox.alec.config import ARCHITECTURES

#: Cached inputs of the tiny identity, relative to the repository root.
CACHED_REFS_RELPATH = "notebooks/checkpoints_step7/external_refs"
CACHED_LEDGER_RELPATH = "notebooks/checkpoints_step7/alpha_on/subset_index_log.json"

#: Rendered grid config filename inside an architecture's work directory.
GRID_FILENAME = "grid.yaml"

#: Completion manifest of a staged reference directory. It lists every file the
#: copy carries, so a later call can tell a finished copy from an interrupted or
#: damaged one; its presence alone is not the criterion (see
#: ``_stage_is_complete``).
STAGE_MARKER = "_stage_complete"

#: A YAML 1.1 sexagesimal token. An unquoted ``8:00:00`` resolves to the base-60
#: integer 28800, which ``submit.render_sbatch`` would substitute into
#: ``#SBATCH --time=${TIME}``; SLURM reads a bare integer as MINUTES, so the
#: 8-hour wall would become 20 days. Both the load side
#: (``_restore_clock_strings``) and the dump side (``_quoting_dumper``) key off
#: this pattern.
_CLOCK_RE = re.compile(r"^[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+(?:\.[0-9_]*)?$")

#: Lazily built YAML dumper (the ``yaml`` import stays function-local, matching
#: ``grid_config.load_grid_config``).
_DUMPER = None


class CachedInputsMissing(FileNotFoundError):
    """The untracked step-7 cache the tiny identity consumes is not present.

    Raised in place of a ``shutil.copytree`` traceback or a compute-node
    failure further down the stage sequence: the message names the missing file
    and the directory that has to be staged.
    """


def repo_root_path() -> Path:
    """The repository root, four parents up from this file.

    ``<root>/xcquinox/alec/cluster/workflow_matrix.py`` -> ``<root>``.
    """
    return Path(__file__).resolve().parents[3]


def template_path() -> Path:
    """The checked-in one-architecture template (package data)."""
    return Path(__file__).resolve().parent / "examples" / \
        "workflow_matrix_template.yaml"


def _missing_cache_message(path, what: str) -> str:
    """Message naming the missing input and the directory to stage."""
    return (
        f"{what} not found at {path}. The workflow matrix consumes the step-7 "
        "cache, which is untracked (.gitignore excludes "
        "notebooks/checkpoints_step7/): stage that directory into the "
        "repository from a machine that holds it before running the matrix."
    )


def cached_refs_dir(repo_root) -> Path:
    """The cached CCSD reference directory under ``repo_root``, checked."""
    refs = Path(repo_root) / CACHED_REFS_RELPATH
    if not refs.is_dir():
        raise CachedInputsMissing(
            _missing_cache_message(refs, "cached CCSD references"))
    return refs


def cached_ledger_path(repo_root) -> Path:
    """The cached subset ledger under ``repo_root``, checked.

    Both branches of :func:`write_matrix_yaml` resolve the ledger through this
    function. With a shared ``external_refs_dir`` the staging branch is skipped,
    and ``validate_grid_semantics`` never stats ``inputs.subset_ledger_path``,
    so without the check a wrong ``repo_root`` renders a config that validates
    on the login node and then fails in the spec builder on a compute node.
    """
    ledger = (Path(repo_root) / CACHED_LEDGER_RELPATH).resolve()
    if not ledger.is_file():
        raise CachedInputsMissing(
            _missing_cache_message(ledger, "cached subset ledger"))
    return ledger


def _staged_files(root: Path) -> list:
    """Relative paths of every file below ``root``, manifest excluded."""
    return sorted(str(p.relative_to(root)) for p in root.rglob("*")
                  if p.is_file() and p.name != STAGE_MARKER)


def _write_stage_manifest(root: Path, source: Path) -> None:
    """Write the completion manifest: a source header, then one relative path
    per staged file. Written inside the partial copy, before the rename."""
    lines = [f"source: {source}"] + _staged_files(root)
    (root / STAGE_MARKER).write_text("\n".join(lines) + "\n")


def _stage_is_complete(root: Path) -> bool:
    """True when the manifest is present and every file it lists is still there.

    Files ADDED under the staged directory are tolerated (a stage writes its own
    artefacts there); files MISSING from it are not, because a destination left
    half-populated -- an interrupted copy, or files removed underneath it --
    would otherwise be reported as a finished stage and the run would start
    against references that are not there.
    """
    marker = root / STAGE_MARKER
    if not marker.is_file():
        return False
    try:
        lines = marker.read_text().splitlines()
    except OSError:
        return False
    recorded = [line for line in lines[1:] if line]
    if not recorded:
        return False
    return all((root / rel).is_file() for rel in recorded)


def stage_cached_inputs(dest_root, *, repo_root) -> dict:
    """Copy the cached CCSD references into ``dest_root`` and locate the ledger.

    The cache is never used as the working copy: ``external_refs.precompute_all``
    creates its cache directory, migrates legacy filenames inside it and writes
    a ``_run_log_<UTC>.json`` on EVERY call, and ``run_oep_cascade`` may rewrite
    a species npz. The matrix therefore works on a copy (74 MB, one per work
    root, shared by every architecture) rather than on a symlink farm, which
    would carry those writes back into the cache the next run reads. Existing
    run logs are not copied.

    The copy is built in ``external_refs.partial-<pid>`` and moved into place
    with ``os.replace`` only after its manifest is written, so an interrupted
    copy is never visible as a finished one, and a destination whose manifest is
    missing or incomplete is removed and staged again.

    The subset ledger is read-only for the harness (only the JSON is read; no
    ``subset.traj`` is opened, see ``spec_builder``), so it is consumed in
    place.
    """
    dest_root = Path(dest_root)
    refs_src = cached_refs_dir(repo_root)
    ledger = cached_ledger_path(repo_root)
    refs_dst = dest_root / "_inputs" / "external_refs"
    if refs_dst.exists() and not _stage_is_complete(refs_dst):
        shutil.rmtree(refs_dst)
    if not refs_dst.exists():
        refs_dst.parent.mkdir(parents=True, exist_ok=True)
        partial = refs_dst.parent / f"{refs_dst.name}.partial-{os.getpid()}"
        if partial.exists():
            shutil.rmtree(partial)
        shutil.copytree(
            refs_src, partial,
            ignore=shutil.ignore_patterns("_run_log_*.json"))
        _write_stage_manifest(partial, refs_src)
        os.replace(partial, refs_dst)
    return {"external_refs_dir": str(refs_dst),
            "subset_ledger_path": str(ledger)}


def _restore_clock_strings(text: str, raw: dict) -> None:
    """Keep the template's walltimes as the strings they were written as.

    ``yaml.safe_load`` applies the YAML 1.1 implicit resolvers, so an UNQUOTED
    ``8:00:00`` arrives as the integer 28800 and would be rendered into
    ``#SBATCH --time=28800`` -- 28800 minutes to SLURM, not 8 hours. Every
    ``cluster`` walltime that did not load as a string is restored from the
    literal token in the template text when that token is clock-shaped, and
    refused otherwise: a bare number is ambiguous (SLURM reads it as minutes
    while the field is documented as HH:MM:SS), so it is a template defect
    rather than something to guess at.
    """
    cluster = raw.get("cluster")
    if not isinstance(cluster, dict):
        return
    for key, value in list(cluster.items()):
        name = str(key)
        if value is None or isinstance(value, str) or not name.endswith("time"):
            continue
        match = re.search(rf"^\s+{re.escape(name)}:\s*(\S+)\s*$", text, re.M)
        token = match.group(1) if match else ""
        if _CLOCK_RE.match(token):
            cluster[key] = token
        else:
            raise ValueError(
                f"cluster.{name} in {template_path()} loaded as {value!r} "
                f"({type(value).__name__}) rather than a walltime string; "
                'write it quoted, as "HH:MM:SS"')


def _quoting_dumper():
    """A ``yaml.SafeDumper`` that emits clock-shaped strings in quotes.

    ``yaml.safe_dump`` writes ``00:30:00`` unquoted, which round-trips only
    because YAML 1.1's sexagesimal integer requires a leading 1-9; the same
    value one hour longer, ``8:00:00``, would be re-read as 28800. Quoting on
    the dump side removes that dependency, so every stage that reloads the
    rendered config reads a walltime string.
    """
    global _DUMPER
    if _DUMPER is None:
        import yaml

        class _ClockQuotingDumper(yaml.SafeDumper):
            pass

        def _represent_str(dumper, data):
            style = "'" if _CLOCK_RE.match(data) else None
            return dumper.represent_scalar(
                "tag:yaml.org,2002:str", data, style=style)

        _ClockQuotingDumper.add_representer(str, _represent_str)
        _DUMPER = _ClockQuotingDumper
    return _DUMPER


def write_matrix_yaml(arch, out_dir, *, repo_root,
                      external_refs_dir=None, pretrain_data_dir=None) -> Path:
    """Render the one-architecture tiny grid config into ``<out_dir>/grid.yaml``.

    The template is parsed and its four CHANGE_ME values are replaced as data,
    not as text, so a malformed substitution cannot produce a syntactically
    valid but semantically wrong config. ``external_refs_dir`` and
    ``pretrain_data_dir`` default to per-architecture directories under
    ``out_dir``; the matrix passes shared ones so the 74 MB reference copy and
    the pretrain-data generation are paid once per shard instead of once per
    architecture. Both directories are created here, whether defaulted or
    supplied.
    """
    import yaml

    if arch not in ARCHITECTURES:
        raise ValueError(
            f"{arch!r} is not a registered architecture; "
            f"valid names: {sorted(ARCHITECTURES)}"
        )
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    text = template_path().read_text()
    raw = yaml.safe_load(text)
    _restore_clock_strings(text, raw)

    if external_refs_dir is None:
        staged = stage_cached_inputs(out_dir, repo_root=repo_root)
        refs = Path(staged["external_refs_dir"])
        ledger = staged["subset_ledger_path"]
    else:
        refs = Path(external_refs_dir).resolve()
        ledger = str(cached_ledger_path(repo_root))
    data_dir = Path(pretrain_data_dir).resolve() if pretrain_data_dir \
        else out_dir / "pretrain_data"
    # Both input directories exist before any stage runs: datagen writes the
    # pretraining data into data_dir, the reference precompute writes into
    # external_refs_dir, and validate_grid_semantics reports a missing
    # directory the same way for either.
    refs.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    raw["sweep"]["arch"] = [arch]
    raw["inputs"]["external_refs_dir"] = str(refs)
    raw["inputs"]["subset_ledger_path"] = str(ledger)
    raw["inputs"]["output_root"] = str(out_dir)
    raw["pretrain"]["data_dir"] = str(data_dir)

    path = out_dir / GRID_FILENAME
    with path.open("w") as f:
        yaml.dump(raw, f, Dumper=_quoting_dumper(),
                  default_flow_style=False, sort_keys=True)
    return path
