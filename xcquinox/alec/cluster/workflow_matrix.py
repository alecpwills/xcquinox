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

from xcquinox.alec.cluster.grid_config import normalize_cluster_walltimes
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
#: 8-hour wall would become 20 days. The DUMP side keys off this pattern, so
#: that a walltime leaves this module quoted whatever it arrived as; what
#: counts as a walltime on the LOAD side is ``grid_config``'s rule, applied
#: through :func:`_restore_clock_strings`.
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
        # copy_function is passed rather than left to the default, which
        # copytree binds at definition time: the per-file copy is then the one
        # shutil holds at call time, so a caller can substitute it.
        shutil.copytree(
            refs_src, partial, copy_function=shutil.copy2,
            ignore=shutil.ignore_patterns("_run_log_*.json"))
        _write_stage_manifest(partial, refs_src)
        os.replace(partial, refs_dst)
    return {"external_refs_dir": str(refs_dst),
            "subset_ledger_path": str(ledger)}


def staged_refs_dir(refs_dir) -> Path:
    """A SUPPLIED ``external_refs_dir``, checked as an input.

    The matrix passes one shared reference copy per shard, so a supplied
    directory is the output of an earlier :func:`stage_cached_inputs` and is an
    INPUT here: it must already hold the references, and it is never created.
    ``grid_config.validate_grid_semantics`` does not stat
    ``inputs.external_refs_dir`` -- only ``pretrain.data_dir`` and the parent of
    ``inputs.output_root`` are checked, and both only advisorily -- so a
    mistyped path created empty at this point would pass the login-node gate
    and surface on a compute node, at the first reference read of a queued job.

    The criterion is what the staging writes: its completion manifest, or the
    per-species ``.npz`` files it copies (55 of them at the measured cache
    size, beside the ``_intermediates`` directory). Either one distinguishes a
    staged directory from an empty or wrong one.
    """
    refs = Path(refs_dir).resolve()
    if not refs.is_dir():
        detail = "does not exist"
    elif not (refs / STAGE_MARKER).is_file() and not any(refs.glob("*.npz")):
        detail = (f"holds neither the staging manifest ({STAGE_MARKER}) nor "
                  "any species .npz, so it carries no references")
    else:
        return refs
    raise CachedInputsMissing(
        f"supplied external_refs_dir {refs} {detail}. It is an INPUT -- the "
        "shared copy stage_cached_inputs writes, one per work root -- and is "
        "not created here: pass the directory an earlier stage_cached_inputs "
        "wrote, or omit external_refs_dir to stage a fresh one.")


def _restore_clock_strings(text: str, raw: dict) -> None:
    """Restore and check the template's walltimes, in place.

    ``yaml.safe_load`` applies the YAML 1.1 implicit resolvers, so an UNQUOTED
    ``8:00:00`` arrives as the integer 28800 and would be rendered into
    ``#SBATCH --time=28800`` -- 28800 minutes to SLURM, not 8 hours. The
    template is read here, before any :class:`GridConfig` exists, so the rule
    that applies to a loaded config is applied to the template through
    ``grid_config.normalize_cluster_walltimes``: a wall the loader resolved to
    a number is restored from its literal, and every field is then checked
    against the accepted shapes (``H:MM:SS`` and ``D-HH:MM:SS``).

    Sharing that function rather than restating the rule is what keeps the two
    load paths from drifting; the shapes it refuses -- a bare number, ``MM:SS``
    -- are legal SLURM meaning MINUTES, and the quoting is no protection, since
    ``time: "30"`` loads as a string and renders ``--time=30`` exactly as
    ``time: 30`` does.
    """
    cluster = raw.get("cluster")
    if not isinstance(cluster, dict):
        return
    raw["cluster"] = normalize_cluster_walltimes(
        cluster, text=text, source=str(template_path()))


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
    architecture.

    The two behave differently because they are not the same kind of path.
    ``pretrain_data_dir`` is an OUTPUT -- datagen writes the pretraining set
    into it -- and is created here whether defaulted or supplied. A supplied
    ``external_refs_dir`` is an INPUT and is only checked, by
    :func:`staged_refs_dir`; creating it empty would hide a mistyped path from
    the login-node validation and surface it on a compute node.
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
        ledger = str(cached_ledger_path(repo_root))
        refs = staged_refs_dir(external_refs_dir)
    data_dir = Path(pretrain_data_dir).resolve() if pretrain_data_dir \
        else out_dir / "pretrain_data"
    # data_dir is an output and exists before datagen runs; external_refs_dir
    # is an input, already checked above and never created here.
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


# ---------------------------------------------------------------------------
# Oracle selection
# ---------------------------------------------------------------------------

#: Test module of the spin-scaling oracles O1-O4
#: (SPEC_pretrain_fidelity_program.md 3.1). Its architecture-carrying oracles
#: are parametrized over ``sorted(ARCHITECTURES)``, so a node id ends in
#: ``[<arch>]`` (or ``[<species>-<arch>]``) and one architecture's oracles are
#: selectable with ``-k``.
ORACLE_MODULE = "test_spin_scaling_oracles"

#: Collection target for the oracle run. A directory rather than the module
#: path so the module name in the selector is what pins the module; pytest
#: matches ``-k`` against the module name as well as the test name.
ORACLE_TEST_TARGET = "xcquinox/alec/tests"


def oracle_selector(arch, archs=None) -> str:
    """A pytest ``-k`` expression selecting one architecture's oracles.

    ``-k`` matches SUBSTRINGS of the node id, and the registry contains names
    that are prefixes of others (``deep`` of ``deep_attn``, ``deep_cusp`` of
    ``deep_cusp_mgga_3x16``, ``shallow`` of ``shallow_attn``), so a bare name
    would silently pull in a sibling architecture's cases and report them as
    this one's. Every longer registry name containing this one is therefore
    excluded explicitly. Every registry name is a Python identifier, so each
    term lexes as a single term of pytest's expression grammar rather than as
    several.

    The exclusions are exact only while the oracle module's test names carry no
    architecture name of their own: ``-k`` matches the function name as well as
    the parametrisation id, so a test called ``test_deep_scaling`` would answer
    to every ``deep*`` term. ``test_oracle_selector_selects_this_architecture_only``
    checks the collected set against that once the module is installed.
    """
    names = sorted(ARCHITECTURES) if archs is None else sorted(archs)
    if arch not in names:
        raise ValueError(
            f"{arch!r} is not a registered architecture; "
            f"valid names: {names}"
        )
    terms = [ORACLE_MODULE, arch]
    terms += [f"not {other}" for other in names
              if other != arch and arch in other]
    return " and ".join(terms)
