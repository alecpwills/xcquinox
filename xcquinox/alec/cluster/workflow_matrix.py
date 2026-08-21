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

import dataclasses
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from xcquinox.alec.cluster.fidelity import (
    CERTIFICATE_FILENAME as _CERTIFICATE_FILENAME,
    certificate_enforced_in as _certificate_enforced_in,
    certificate_path as _certificate_path,
    gate_certificate as _gate_certificate,
    read_certificate as _read_certificate,
)
from xcquinox.alec.cluster.grid_config import (
    normalize_cluster_walltimes, pretrain_checkpoint_dir,
)
from xcquinox.alec.config import ARCHITECTURES
from xcquinox.alec.eval_holdout import (
    EVAL_METADATA_NAME as _EVAL_STAMP_NAME,
    SLICED_MARKER_NAME as _SLICE_MARKER_NAME,
)
from xcquinox.alec.full_benchmark_pools import (
    HELDOUT_SPECIES_SLICE_ENV as _HELDOUT_SLICE_ENV,
)

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

    The criterion is what the staging writes: its completion manifest CHECKED
    against the files it records (:func:`_stage_is_complete`), or the
    per-species ``.npz`` files it copies (55 of them at the measured cache
    size, beside the ``_intermediates`` directory). Either one distinguishes a
    staged directory from an empty or wrong one. The manifest's mere presence
    is not the criterion: a manifest that records no file, or one whose files
    have been removed, describes a directory carrying no references, which is
    the case this refusal exists for.
    """
    refs = Path(refs_dir).resolve()
    if not refs.is_dir():
        detail = "does not exist"
    elif not _stage_is_complete(refs) and not any(refs.glob("*.npz")):
        detail = (f"holds neither a complete staging manifest ({STAGE_MARKER} "
                  "listing files that are all present) nor any species .npz, "
                  "so it carries no references")
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
    excluded explicitly. Containment is tested case-INSENSITIVELY, because
    that is how ``-k`` matches (``KeywordMatcher.__call__`` lowercases both
    sides): a registry entry differing from a longer one only in case would
    otherwise be left unexcluded while pytest still collected it. Every
    registry name is a Python identifier, so each term lexes as a single term
    of pytest's expression grammar rather than as several.

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
              if other != arch and arch.lower() in other.lower()]
    return " and ".join(terms)


def oracle_function_name_conflicts(node_ids, archs=None) -> list:
    """Collected oracle node ids whose FUNCTION name carries a registry name.

    :func:`oracle_selector` is exact only while the oracle module's test names
    carry no architecture name of their own: ``-k`` matches the function name
    as well as the parametrisation id, so a test called
    ``test_deep_channel_gradient`` answers to every selector naming ``deep``
    and would be collected -- and reported -- as that architecture's oracle
    whatever its own parameters say. No expression can exclude it, since the
    architecture term is exactly what has to match; the collected names are
    therefore checked instead, against the same case-insensitive rule ``-k``
    applies. Returns one message per offending node id, empty when the module's
    function names carry none.
    """
    names = sorted(ARCHITECTURES) if archs is None else sorted(archs)
    conflicts = []
    for node_id in node_ids:
        function = str(node_id).split("::")[-1].split("[")[0]
        carried = [name for name in names if name.lower() in function.lower()]
        if carried:
            conflicts.append(
                f"{node_id}: the test function {function!r} carries the "
                f"registry name(s) {carried}, so every selector naming one of "
                "them collects this case as that architecture's oracle")
    return conflicts


# ---------------------------------------------------------------------------
# Stage table
# ---------------------------------------------------------------------------

#: Six species of the BH76 + W4-11 held-out pool closing three reactions -- one
#: BH76 barrier (h + n2o -> n2ohts) and two W4-11 atomizations (h2, oh) -- over
#: both spin types (RKS h2 / n2o, UKS h / o / oh / n2ohts). The full pool is 216
#: reactions over 214 species and hours of SCF per grid cell
#: (SPEC_pretrain_fidelity_program.md 3.4), and it is not narrowable from the
#: grid config. The W4-11 leg is atomization energies, so every one of its
#: reactions carries single-atom legs: a slice of six MOLECULES with no atoms
#: closes no atomization at all and would leave that half of the reaction math
#: untested, which is why the atoms are in it.
HELDOUT_SPECIES_SLICE = "h,h2,o,oh,n2o,n2ohts"

#: Reactions :data:`HELDOUT_SPECIES_SLICE` closes: ``bh76_h_n2o_to_n2ohts``,
#: ``w411_h2_atomization`` and ``w411_oh_atomization``.
#: ``full_benchmark_pools.slice_held_out_pools`` keeps a reaction only when
#: every reactant and product lies inside the slice, so this is a property of
#: the pool rather than a choice; it is measured against the loaded pool by
#: ``test_the_slice_constant_closes_the_reactions_it_claims``. The eval stage
#: records the same number in its ``sliced_eval.json`` mark, which is what
#: :func:`_slice_check` compares against.
SLICE_CLOSED_REACTIONS = 3

#: Prefix ``cluster/validate_run.main`` prints one failure line under, before
#: its count line and its exit 1. It is the only machine-readable form that
#: module offers: the validator writes no JSON, so the expected-refusal test
#: below reads these lines.
_VALIDATE_FAIL_PREFIX = "[validate_run] FAIL: "

#: What the report records for a ``validate_run`` that refused the run for the
#: one reason this identity expects.
VALIDATE_RUN_EXPECTED_DETAIL = "refused the waived certificate as expected"

#: Return code recorded for a certificate stage that wrote no certificate.
#: The template's waiver (``fidelity.enforce: false``) covers a FAIL VERDICT,
#: which is the expected outcome of a 50-step pretrain, and nothing else: an
#: absent certificate is refused by ``fidelity.gate_certificate``, so the
#: preflight sweep and the train task would refuse the run, and the matrix
#: would have no verdict to report. It is therefore a failure of the stage
#: whatever the stage's own exit code said.
CERTIFICATE_MISSING_RC = 3

#: Per-stage wall-clock cap. It is a hang detector, not a schedule: the tiny
#: identity is 3 training steps and 50 pretraining steps at def2-svp / grid
#: level 1, two orders of magnitude below the 30-minute SLURM walls the
#: template renders for the same stages at production size, so a stage still
#: running after an hour is stuck rather than slow.
DEFAULT_STAGE_TIMEOUT_S = 3600

#: Return code recorded for a stage killed by the timeout: the convention of
#: ``timeout(1)`` (GNU coreutils), which exits 124 when the command runs on
#: past its limit.
TIMEOUT_RC = 124

#: pytest's exit code for a run that collected nothing
#: (``pytest.ExitCode.NO_TESTS_COLLECTED``). The oracle module belongs to spec
#: 3.1; while it is absent, or if a selector stops matching it, the oracle
#: stage exits with this code and must report the cause by name rather than as
#: an anonymous failure -- and never as a pass.
ORACLE_NO_TESTS_RC = 5

#: Grid cells of the tiny identity, in spec-index order (subset sizes 1 and 2
#: of the cached ledger). One train stage and one eval stage run per cell;
#: ``test_the_stage_table_covers_every_cell_the_template_expands_to`` checks
#: this against what the template's grid expands to.
_SPEC_INDICES = (0, 1)

#: Stage names in execution order; the report's column legend.
STAGE_ORDER = ("submit", "datagen", "pretrain", "certificate", "preflight",
               "train[0]", "train[1]", "eval[0]", "eval[1]", "validate_run")

_RUN_DIR_LINE = re.compile(r"^submit: run dir = (?P<path>\S.*)$")


@dataclasses.dataclass(frozen=True)
class Stage:
    """One stage invocation: its name, its argv, and its failure policy."""

    name: str
    argv: tuple
    #: The certificate is the one stage whose non-zero exit does not stop the
    #: sequence: spec 3.4 records its verdict, it does not require a PASS.
    allow_nonzero: bool = False
    #: Extra environment for this stage only, as ``((key, value), ...)``.
    env_extra: tuple = ()


def stage_plan(run_dir, *, species_slice=HELDOUT_SPECIES_SLICE,
               device="cpu") -> tuple:
    """The nine stages after ``submit``, in the order the job graph runs them.

    Each stage is the module SLURM would invoke, with the same argument vector,
    so the matrix verifies the code the cluster executes rather than an
    in-process re-implementation of it. The species slice reaches the eval
    stages ONLY: no other stage reads it, and confining it here keeps the
    training pool provably untouched.
    """
    py = sys.executable
    run_dir = str(run_dir)
    slice_env = ((_HELDOUT_SLICE_ENV, species_slice),) if species_slice else ()
    stages = [
        Stage("datagen",
              (py, "-m", "xcquinox.alec.cluster._datagen", run_dir)),
        Stage("pretrain",
              (py, "-m", "xcquinox.alec.cluster._pretrain", run_dir, "0")),
        Stage("certificate",
              (py, "-m", "xcquinox.alec.cluster.fidelity", run_dir, "0"),
              allow_nonzero=True),
        Stage("preflight",
              (py, "-m", "xcquinox.alec.cluster._preflight", run_dir)),
    ]
    stages += [
        Stage(f"train[{idx}]",
              (py, "-m", "xcquinox.alec.cluster._train_task", run_dir,
               str(idx), "--device", device))
        for idx in _SPEC_INDICES
    ]
    stages += [
        Stage(f"eval[{idx}]",
              (py, "-m", "xcquinox.alec.cluster._eval_one_spec", run_dir,
               str(idx)),
              env_extra=slice_env)
        for idx in _SPEC_INDICES
    ]
    stages.append(
        Stage("validate_run",
              (py, "-m", "xcquinox.alec.cluster.validate_run", run_dir)))
    return tuple(stages)


def _base_env(threads):
    """Process environment shared by every stage.

    fp32 versus fp64 silently changes every energy, and the matrix runs several
    architectures at once on one box, so the JAX backend and the BLAS thread
    caps are pinned here rather than inherited. Any inherited species slice is
    dropped: only the eval stages get one, and only from :func:`stage_plan`.

    The certificate's enforcement is NOT configured here. It is the rendered
    config's ``fidelity`` block (``enforce: false`` with a non-empty
    ``override_reason``, ``grid_config.FidelityConfig``), which the certificate
    copies into its own record and ``fidelity.gate_certificate`` re-checks
    there; no environment variable opens that gate.
    """
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[key] = str(threads)
    env.pop(_HELDOUT_SLICE_ENV, None)
    return env


def _run_stage(name, argv, log_path, *, runner, env, timeout_s, cwd):
    """Run one stage into its own log; return the stage record."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    with log_path.open("w") as fh:
        fh.write(f"$ {' '.join(str(a) for a in argv)}\n")
        fh.flush()
        try:
            completed = runner(list(argv), stdout=fh,
                               stderr=subprocess.STDOUT, cwd=str(cwd),
                               env=env, timeout=timeout_s, check=False)
            rc = int(completed.returncode)
        except subprocess.TimeoutExpired:
            fh.write(f"\n[workflow_matrix] {name} exceeded {timeout_s} s and "
                     "was killed\n")
            rc = TIMEOUT_RC
    return {"name": name, "rc": rc,
            "seconds": round(time.monotonic() - t0, 1), "log": str(log_path)}


def _parse_run_dir(log_path):
    """The run directory ``submit`` created, from its own log.

    ``cmd_submit`` ends with ``submit: run dir = <path>``; reading it is exact,
    where globbing ``<output_root>/runs`` would race a concurrent shard.
    """
    run_dir = None
    with Path(log_path).open(errors="replace") as fh:
        for line in fh:
            match = _RUN_DIR_LINE.match(line.strip())
            if match:
                run_dir = match.group("path").strip()
    return run_dir


def _manifest_width(run_dir):
    """Zero-pad width of the spec indices, from the manifest (default 4)."""
    path = Path(run_dir) / "manifest.json"
    try:
        with path.open() as fh:
            return int(json.load(fh)["width"])
    except (OSError, ValueError, KeyError, TypeError):
        return 4


def _spec_dir(run_dir, idx, width):
    """``<run_dir>/checkpoints/spec_NNNN`` for one grid cell."""
    return Path(run_dir) / "checkpoints" / f"spec_{idx:0{width}d}"


def _polarized_data(grid_path, arch) -> bool:
    """Whether datagen writes the POLARIZED pretraining file for this run.

    Mirrors ``cluster/_datagen._required_polarization_flags``: a run-level
    ``use_polarized_correlation`` forces the polarized file for every
    architecture, and otherwise the architecture's own flag decides. Read out
    of the rendered YAML rather than through ``load_grid_config`` because one
    boolean is wanted and an unreadable config is not a fatal condition here --
    the artefact record then simply names the unpolarized file, and the stage
    logs carry the real failure.
    """
    import yaml

    try:
        raw = yaml.safe_load(Path(grid_path).read_text())
    except (OSError, yaml.YAMLError):
        raw = None
    run_level = bool(isinstance(raw, dict)
                     and raw.get("use_polarized_correlation", False))
    return run_level or bool(getattr(ARCHITECTURES[arch],
                                     "use_polarized_correlation", False))


def _artefact_paths(run_dir, arch, data_dir, polarized=False):
    """Every artefact the stage sequence is expected to leave behind."""
    from xcquinox.alec.pretrain_data_gen import pretrain_data_filename

    run = Path(run_dir)
    width = _manifest_width(run)
    pre = Path(pretrain_checkpoint_dir(str(run), arch))
    scripts = run / "scripts"
    labels = {
        "resolved_config": run / "resolved_config.yaml",
        "script_datagen": scripts / "datagen.sbatch",
        "script_pretrain": scripts / "pretrain.sbatch",
        "script_preflight": scripts / "preflight.sbatch",
        "script_train": scripts / "train_array.sbatch",
        "script_eval": scripts / "eval_array.sbatch",
        # The generator's own naming function, so the two cannot drift; the
        # datagen stage calls it with the default (PBE) reference density.
        "pretrain_data": Path(data_dir) / pretrain_data_filename(polarized),
        "pretrain_xnet": pre / "xnet.eqx",
        "pretrain_cnet": pre / "cnet.eqx",
        "pretrain_metadata": pre / "pretrain_metadata.json",
        "certificate": pre / _CERTIFICATE_FILENAME,
        "manifest": run / "manifest.json",
    }
    for idx in _SPEC_INDICES:
        ckpt = _spec_dir(run, idx, width)
        labels[f"spec[{idx}]"] = run / "specs" / f"spec_{idx:0{width}d}.spec"
        labels[f"model[{idx}]"] = ckpt / "model.eqx"
        labels[f"eval_df[{idx}]"] = ckpt / "eval_df.csv"
        labels[f"holdout_test_set[{idx}]"] = \
            ckpt / "eval_holdout" / "test_set.csv"
        labels[f"holdout_metadata[{idx}]"] = \
            ckpt / "eval_holdout" / _EVAL_STAMP_NAME
        labels[f"holdout_sliced[{idx}]"] = \
            ckpt / "eval_holdout" / _SLICE_MARKER_NAME
    return {name: {"path": str(path), "exists": path.exists()}
            for name, path in labels.items()}


def _certificate_record(run_dir, arch) -> dict:
    """What the certificate stage left behind, as the report carries it.

    The verdict, the waiver the certificate records and whether the ON-NODE
    gates will release the run are all read through ``cluster/fidelity``'s own
    predicates (``read_certificate``, ``certificate_enforced_in``,
    ``gate_certificate``), so the matrix reports what ``_preflight`` and
    ``_train_task`` will act on rather than a second reading of the same file.

    At this identity a FAIL verdict is the EXPECTED outcome -- 50 pretraining
    steps on two atoms cannot reproduce the parent functional to
    tol_AE = 1.0 kcal/mol -- and the template's ``fidelity.enforce: false``
    waiver is what lets the sequence continue past it with the verdict on
    record. An ABSENT or unreadable certificate is waived by nothing: the gate
    refuses it, and ``present`` False is what :func:`run_arch` turns into a
    stage failure.
    """
    if run_dir is None:
        return {"present": False, "path": None, "verdict": None,
                "enforced": None, "override_reason": None,
                "gate_released": False,
                "gate_message": ("no run directory; the certificate stage did "
                                 "not run")}
    pretrain_dir = pretrain_checkpoint_dir(str(run_dir), arch)
    payload = _read_certificate(pretrain_dir)
    if not isinstance(payload, dict):
        payload = None
    tolerances = payload.get("tolerances") if payload else None
    allowed, message = _gate_certificate(str(run_dir), arch)
    return {
        "present": payload is not None,
        "path": _certificate_path(str(run_dir), arch),
        "verdict": payload.get("verdict") if payload else None,
        "enforced": (_certificate_enforced_in(pretrain_dir) if payload
                     else None),
        "override_reason": (tolerances.get("override_reason")
                            if isinstance(tolerances, dict) else None),
        "gate_released": bool(allowed),
        "gate_message": message,
    }


def _validate_run_failures(log_path):
    """The failures ``validate_run`` printed, in order, without their prefix."""
    try:
        text = Path(log_path).read_text(errors="replace")
    except OSError:
        return []
    return [line.strip()[len(_VALIDATE_FAIL_PREFIX):].strip()
            for line in text.splitlines()
            if line.strip().startswith(_VALIDATE_FAIL_PREFIX)]


def _is_certificate_refusal(failure, arch) -> bool:
    """True for ``validate_run``'s certificate-VERDICT refusal of ``arch``.

    The validator writes
    ``pretrain/<arch>: fidelity certificate verdict <v>, expected 'PASS' ...``
    for that one check and a differently-shaped line for each of its others
    (identity block, named architecture, parent functional, code version,
    checkpoint digests), so matching the shape identifies the check without a
    machine-readable report to key on.
    """
    return (failure.startswith(f"pretrain/{arch}: fidelity certificate "
                               "verdict ")
            and "expected 'PASS'" in failure)


def _validate_run_outcome(log_path, rc, arch, certificate) -> dict:
    """Whether ``validate_run`` ended the way this identity requires.

    ``validate_run`` is a RECORD layer and stays strict: it requires
    ``verdict == "PASS"`` and ignores the certificate's ``enforced`` field by
    design, so a run carrying the matrix's waived FAIL certificate MUST be
    refused by it -- and refused for exactly that reason. Three outcomes are
    therefore distinguished, and only the first is expected:

    * one failure, the certificate-verdict refusal of the architecture under
      test: the record layer did its job and the matrix's own assertions are
      unaffected;
    * a zero exit: the record layer accepted a run whose certificate is a
      recorded FAIL, which is the guarantee that keeps a workflow run out of
      the results;
    * anything else it refused: a second failure would otherwise hide behind
      the expected one, since both produce the same exit code.

    With no waiver in play (a PASS certificate) the ordinary contract applies:
    exit zero is the expected outcome.
    """
    failures = _validate_run_failures(log_path)
    waived = (certificate.get("verdict") == "FAIL"
              and certificate.get("enforced") is False)
    if not waived:
        return {"expected": rc == 0, "rc": rc, "failures": failures,
                "detail": ("clean" if rc == 0 else
                           f"exited {rc} with {len(failures)} failure(s): "
                           + "; ".join(failures))}
    if rc == 0:
        return {"expected": False, "rc": rc, "failures": failures,
                "detail": ("validate_run exited 0 on a run whose certificate "
                           "records a FAIL under the matrix waiver; the record "
                           "layer requires verdict PASS and ignores the "
                           "waiver, so a clean exit means that requirement is "
                           "no longer imposed")}
    if len(failures) == 1 and _is_certificate_refusal(failures[0], arch):
        return {"expected": True, "rc": rc, "failures": failures,
                "detail": VALIDATE_RUN_EXPECTED_DETAIL}
    if not failures:
        return {"expected": False, "rc": rc, "failures": failures,
                "detail": (f"validate_run exited {rc} but printed no "
                           f"{_VALIDATE_FAIL_PREFIX!r} line, so what it "
                           "refused cannot be read")}
    others = [f for f in failures if not _is_certificate_refusal(f, arch)]
    return {"expected": False, "rc": rc, "failures": failures,
            "detail": (f"validate_run reported {len(failures)} failures, "
                       f"{len(others)} of them not the certificate refusal "
                       f"for {arch}: " + "; ".join(failures))}


# ---------------------------------------------------------------------------
# Held-out channel: a sliced evaluation has to be MARKED as one
# ---------------------------------------------------------------------------

def _read_json_or_none(path):
    """Parsed JSON, or None when the file is unreadable or malformed."""
    try:
        with Path(path).open() as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _check_one_channel(channel, wanted, rel):
    """Refusal text for one held-out channel directory, or None when it is
    marked as the slice that was asked for.

    ``cluster/_eval_one_spec`` marks a sliced channel twice -- the marker
    before any energy is computed, the stamp after the evaluation -- and the
    figure layer refuses either mark, so a channel missing them is read
    downstream as a FULL-pool channel. The reaction count in the marker is the
    slice's own closure (the stamp's is what survived the validation-complement
    filter and is not comparable), so it is the marker's count that is checked
    against :data:`SLICE_CLOSED_REACTIONS`.
    """
    marker = _read_json_or_none(channel / _SLICE_MARKER_NAME)
    if marker is None:
        return (f"{rel} carries no readable {_SLICE_MARKER_NAME}: the held-out "
                "evaluation did not record a species slice, so the channel is "
                "either a full-pool evaluation or an unmarked sliced one")
    names = marker.get("species_slice") if isinstance(marker, dict) else None
    if list(names or ()) != list(wanted):
        return (f"{rel}/{_SLICE_MARKER_NAME} records species_slice={names!r}, "
                f"not the requested {list(wanted)!r}")
    n_reactions = marker.get("n_reactions")
    if n_reactions != SLICE_CLOSED_REACTIONS:
        return (f"{rel}/{_SLICE_MARKER_NAME} records "
                f"n_reactions={n_reactions!r}; the slice closes "
                f"{SLICE_CLOSED_REACTIONS} reactions, so a different pool was "
                "evaluated")
    stamp = _read_json_or_none(channel / _EVAL_STAMP_NAME)
    if not isinstance(stamp, dict):
        return (f"{rel} carries no readable {_EVAL_STAMP_NAME}: the "
                "evaluation did not finish, or its provenance stamp failed to "
                "write")
    if not stamp.get("species_slice"):
        return (f"{rel}/{_EVAL_STAMP_NAME} records "
                f"species_slice={stamp.get('species_slice')!r}, which is what "
                "a FULL-pool evaluation stamps; it contradicts the marker "
                "written before the energies")
    return None


def _slice_check(run_dir, *, species_slice, evaluated):
    """Whether the held-out channels the eval stages wrote are marked sliced.

    The matrix's held-out assertion is that the eval stage ran on the named
    species slice: the slice is what makes the channel affordable, and an
    unmarked channel is indistinguishable downstream from a full-pool one. The
    check therefore reads what ``cluster/_eval_one_spec`` writes -- the marker
    before the energies and the ``species_slice`` entry of the provenance stamp
    after them -- for EVERY ``eval_holdout*`` channel of every cell, and
    compares the marker's reaction count with the closure of the slice.

    ``ok`` is None when there is nothing to judge: the eval stages did not
    complete, so the stage return codes already carry the failure and a slice
    verdict on top of them would report a second, spurious one. ``ok`` is True
    with ``checked`` False when no slice was asked for, which is a full-pool
    evaluation by request.
    """
    if not species_slice:
        return {"checked": False, "ok": True, "channels": [],
                "n_reactions": None,
                "detail": ("no species slice requested; the full held-out "
                           "pool was evaluated")}
    if not evaluated:
        return {"checked": False, "ok": None, "channels": [],
                "n_reactions": None,
                "detail": ("the eval stages did not complete; no held-out "
                           "channel to check")}
    wanted = [name.strip() for name in str(species_slice).split(",")
              if name.strip()]
    run = Path(run_dir)
    width = _manifest_width(run)
    channels, problems = [], []
    for idx in _SPEC_INDICES:
        ckpt = _spec_dir(run, idx, width)
        found = sorted(p for p in ckpt.glob("eval_holdout*") if p.is_dir())
        if not found:
            problems.append(f"{ckpt.name} carries no held-out channel "
                            "directory (eval_holdout*)")
            continue
        for channel in found:
            rel = f"{ckpt.name}/{channel.name}"
            channels.append(rel)
            problem = _check_one_channel(channel, wanted, rel)
            if problem:
                problems.append(problem)
    if problems:
        return {"checked": True, "ok": False, "channels": channels,
                "n_reactions": None, "detail": "; ".join(problems)}
    return {"checked": True, "ok": True, "channels": channels,
            "n_reactions": SLICE_CLOSED_REACTIONS,
            "detail": (f"{len(channels)} held-out channel(s) marked sliced to "
                       f"{len(wanted)} species / {SLICE_CLOSED_REACTIONS} "
                       "reactions")}


# ---------------------------------------------------------------------------
# Oracles
# ---------------------------------------------------------------------------

def _summary_line(log_path):
    """The last non-empty line of a pytest log -- its one-line result."""
    try:
        lines = [ln.strip() for ln in
                 Path(log_path).read_text(errors="replace").splitlines()
                 if ln.strip()]
    except OSError:
        return ""
    return lines[-1] if lines else ""


def _oracle_module_path(repo_root):
    """Where the spec-3.1 oracle module lives when it is installed."""
    return (Path(repo_root) / "xcquinox" / "alec" / "tests"
            / f"{ORACLE_MODULE}.py")


def _oracle_failure_note(rc, module_path, selector):
    """Name what a non-zero oracle exit means, or None to keep pytest's own
    summary line.

    The oracle module is spec 3.1's and this runner does not own it. Absent, it
    collects nothing and pytest exits :data:`ORACLE_NO_TESTS_RC`; reported as a
    bare non-zero code that is indistinguishable from a failing oracle, and
    reported as anything but a failure it would certify an architecture no
    oracle was ever run against. Both cases are therefore named, in a form
    short enough for the report's table, with the full sentence in the log.
    """
    if not Path(module_path).is_file():
        return (f"no oracle module: {ORACLE_MODULE}.py is not installed",
                f"[workflow_matrix] the spin-scaling oracle module is not "
                f"installed at {module_path} (SPEC_pretrain_fidelity_program"
                f".md 3.1), so the selector {selector!r} can match nothing "
                f"and this architecture has no oracle result")
    if rc == ORACLE_NO_TESTS_RC:
        return (f"no oracle collected: {ORACLE_MODULE} matched nothing",
                f"[workflow_matrix] pytest collected no test under "
                f"{ORACLE_TEST_TARGET} for the selector {selector!r} (exit "
                f"{ORACLE_NO_TESTS_RC}, no tests collected); the module is "
                f"present at {module_path}, so its node ids no longer carry "
                f"the architecture the selector names")
    return None


def _run_oracles(arch, log_path, *, runner, env, timeout_s, cwd):
    """Run this architecture's slice of the spin-scaling oracles O1-O4."""
    selector = oracle_selector(arch)
    argv = (sys.executable, "-m", "pytest", ORACLE_TEST_TARGET,
            "-k", selector, "-q", "-p", "no:randomly")
    record = _run_stage("oracles", argv, log_path, runner=runner, env=env,
                        timeout_s=timeout_s, cwd=cwd)
    summary = _summary_line(log_path)
    if record["rc"] != 0:
        note = _oracle_failure_note(
            record["rc"], _oracle_module_path(cwd), selector)
        if note is not None:
            summary, detail = note
            with Path(log_path).open("a") as fh:
                fh.write(f"\n{detail}\n")
    return {"rc": record["rc"], "summary_line": summary,
            "log": record["log"], "selector": selector,
            "seconds": record["seconds"]}


# ---------------------------------------------------------------------------
# One architecture
# ---------------------------------------------------------------------------

def run_arch(arch, work_root, *, runner=subprocess.run,
             timeout_s=DEFAULT_STAGE_TIMEOUT_S, repo_root=None,
             external_refs_dir=None, pretrain_data_dir=None,
             species_slice=HELDOUT_SPECIES_SLICE, threads=4,
             run_oracles=True) -> dict:
    """Drive one architecture through the whole stage sequence.

    The certificate stage is the one stage whose non-zero exit does not stop
    the sequence, and that tolerance is tied to the certificate FILE: a FAIL
    verdict under the template's ``fidelity.enforce: false`` waiver is the
    expected outcome at this identity and is recorded, while a stage that
    wrote no certificate is a stage failure (:data:`CERTIFICATE_MISSING_RC`).
    ``validate_run`` is judged on its REPORT rather than on its exit code: it
    is a record layer, it stays strict, and under the waiver its expected
    outcome is the certificate refusal and nothing else
    (:func:`_validate_run_outcome`).

    ``submit`` runs in its default DRY-RUN, which creates the run directory,
    writes ``resolved_config.yaml`` and renders every sbatch script without
    calling SLURM (``--submit`` is the opt-in that queues, and the matrix never
    passes it); the matrix then invokes each stage module itself. The sequence
    stops at the first non-zero exit -- a stage's inputs are the previous
    stage's outputs, so continuing past a failure measures nothing -- except
    for the certificate, whose verdict is recorded rather than required. The
    oracles run regardless: they are a property of the installed code, not of
    this run directory.
    """
    root = Path(repo_root) if repo_root is not None else repo_root_path()
    arch_root = Path(work_root).resolve() / arch
    logs_dir = arch_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    grid_path = write_matrix_yaml(
        arch, arch_root, repo_root=root,
        external_refs_dir=external_refs_dir,
        pretrain_data_dir=pretrain_data_dir)
    data_dir = (Path(pretrain_data_dir) if pretrain_data_dir
                else arch_root / "pretrain_data")
    env = _base_env(threads)
    t_all = time.monotonic()

    stages = [_run_stage(
        "submit",
        (sys.executable, "-m", "xcquinox.alec.cluster", "submit",
         str(grid_path), "--partition", "local"),
        logs_dir / "submit.log", runner=runner, env=env, timeout_s=timeout_s,
        cwd=root)]
    run_dir = _parse_run_dir(stages[0]["log"]) if stages[0]["rc"] == 0 else None
    if stages[0]["rc"] == 0 and run_dir is None:
        # A zero exit with no run-dir line means the dry-run changed its
        # contract; every later stage takes the run dir as argv[0], so there is
        # nothing to run.
        stages[0]["rc"] = 2
        with (logs_dir / "submit.log").open("a") as fh:
            fh.write("\n[workflow_matrix] submit printed no "
                     "'submit: run dir = ' line\n")

    certificate = _certificate_record(run_dir, arch)
    validate = {"expected": None, "rc": None, "failures": [],
                "detail": "validate_run did not run"}
    if run_dir is not None and stages[0]["rc"] == 0:
        for stage in stage_plan(run_dir, species_slice=species_slice):
            stage_env = dict(env)
            stage_env.update(dict(stage.env_extra))
            log_name = stage.name.replace("[", "_").replace("]", "")
            log_path = logs_dir / f"{log_name}.log"
            stages.append(_run_stage(
                stage.name, stage.argv, log_path, runner=runner,
                env=stage_env, timeout_s=timeout_s, cwd=root))
            tolerated = stage.allow_nonzero
            if stage.name == "certificate":
                certificate = _certificate_record(run_dir, arch)
                if not certificate["present"]:
                    # The stage's tolerance is tied to this: a FAIL verdict is
                    # recorded and the sequence continues, a certificate that
                    # was never written is a failure of the stage.
                    tolerated = False
                    if stages[-1]["rc"] == 0:
                        stages[-1]["rc"] = CERTIFICATE_MISSING_RC
                    with log_path.open("a") as fh:
                        fh.write("\n[workflow_matrix] the certificate stage "
                                 f"wrote no readable {_CERTIFICATE_FILENAME} "
                                 f"at {certificate['path']}; "
                                 f"{certificate['gate_message']}\n")
            if stage.name == "validate_run":
                validate = _validate_run_outcome(
                    log_path, stages[-1]["rc"], arch, certificate)
                tolerated = tolerated or validate["expected"]
            if stages[-1]["rc"] != 0 and not tolerated:
                break

    oracle_tests = {"rc": None, "summary_line": "", "log": None,
                    "selector": oracle_selector(arch), "seconds": 0.0}
    if run_oracles:
        oracle_tests = _run_oracles(
            arch, logs_dir / "oracles.log", runner=runner, env=env,
            timeout_s=timeout_s, cwd=root)

    rc_by_stage = {stage["name"]: stage["rc"] for stage in stages}
    evaluated = all(rc_by_stage.get(f"eval[{idx}]") == 0
                    for idx in _SPEC_INDICES)
    artefacts = (_artefact_paths(run_dir, arch, data_dir,
                                 _polarized_data(grid_path, arch))
                 if run_dir else {})
    return {
        "arch": arch,
        "run_dir": run_dir,
        "seconds": round(time.monotonic() - t_all, 1),
        "stages": stages,
        "artefacts": artefacts,
        "certificate": certificate,
        "certificate_verdict": certificate["verdict"],
        "validate_run": validate,
        "slice_check": (_slice_check(run_dir, species_slice=species_slice,
                                     evaluated=evaluated) if run_dir else
                        {"checked": False, "ok": None, "channels": [],
                         "n_reactions": None,
                         "detail": "no run directory; no channel to check"}),
        "oracle_tests": oracle_tests,
    }


# ---------------------------------------------------------------------------
# The matrix
# ---------------------------------------------------------------------------

#: Concurrency ceiling. Each shard runs one stage subprocess at a time and
#: :func:`run_matrix` divides the BLAS thread cap by the shard count, so on
#: this workstation (20 logical cores, ``os.cpu_count()``) four shards already
#: put each stage at five threads; past that the stages contend for memory
#: bandwidth and the per-architecture wall stops being comparable between
#: shards, which is the number the report exists to carry.
MAX_SHARDS = 4


def run_matrix(archs, work_root, *, shards=1, runner=subprocess.run,
               timeout_s=DEFAULT_STAGE_TIMEOUT_S, repo_root=None,
               external_refs_dir=None,
               species_slice=HELDOUT_SPECIES_SLICE, threads=None,
               run_oracles=True, progress=None) -> list:
    """Run the stage sequence for every architecture in ``archs``.

    Architectures are dealt round-robin into ``shards`` groups, each group run
    serially by one thread; the threads only wait on subprocesses, so the work
    is in the stages, not in this process. Every shard gets its OWN
    pretrain-data directory because the generator writes a fixed filename and
    two concurrent datagen stages would race on it; inside a shard the second
    architecture's datagen is a skip-if-current no-op.

    The reference copy is staged ONCE here, before any thread starts, so the
    copy itself cannot race; ``external_refs_dir`` re-uses an already staged
    copy instead. ``progress`` is called with each finished result and may be
    called from several threads.

    An architecture whose sequence raises is RECORDED and the matrix carries
    on: the point of the pass is a row per architecture, and one launch
    failure taking the rest of the registry with it would cost a run of the
    whole matrix to learn one thing. The exception text is kept in the record's
    ``error`` field, which :func:`matrix_exit_code` counts as a failure.
    """
    archs = list(archs)
    if not archs:
        raise ValueError("run_matrix: no architectures given")
    unknown = [a for a in archs if a not in ARCHITECTURES]
    if unknown:
        raise ValueError(
            f"run_matrix: {unknown} are not registered architectures; "
            f"valid names: {sorted(ARCHITECTURES)}"
        )
    shards = int(shards)
    if not 1 <= shards <= MAX_SHARDS:
        raise ValueError(
            f"run_matrix: shards must satisfy 1 <= shards <= {MAX_SHARDS}, "
            f"got {shards}. Each shard runs one SCF-heavy stage at a time."
        )
    root = Path(repo_root) if repo_root is not None else repo_root_path()
    work_root = Path(work_root).resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    refs = (str(Path(external_refs_dir).resolve()) if external_refs_dir
            else stage_cached_inputs(work_root,
                                     repo_root=root)["external_refs_dir"])
    if threads is None:
        threads = max(1, (os.cpu_count() or 4) // shards)
    groups = [archs[k::shards] for k in range(shards)]
    data_dirs = [work_root / "_inputs" / f"pretrain_data_shard{k}"
                 for k in range(shards)]

    def _run_group(k):
        out = []
        for arch in groups[k]:
            t0 = time.monotonic()
            try:
                out.append(run_arch(
                    arch, work_root, runner=runner, timeout_s=timeout_s,
                    repo_root=root, external_refs_dir=refs,
                    pretrain_data_dir=data_dirs[k],
                    species_slice=species_slice, threads=threads,
                    run_oracles=run_oracles))
            except Exception as exc:  # noqa: BLE001 -- recorded, not raised
                out.append({
                    "arch": arch, "run_dir": None,
                    "seconds": round(time.monotonic() - t0, 1),
                    "stages": [], "artefacts": {},
                    "certificate_verdict": None,
                    "slice_check": {"checked": False, "ok": None,
                                    "channels": [], "n_reactions": None,
                                    "detail": "the sequence did not run"},
                    "oracle_tests": {"rc": None, "summary_line": "",
                                     "log": None,
                                     "selector": oracle_selector(arch),
                                     "seconds": 0.0},
                    "error": f"{type(exc).__name__}: {exc}",
                })
            if progress is not None:
                progress(out[-1])
        return out

    if shards == 1:
        collected = _run_group(0)
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=shards) as pool:
            collected = [record
                         for group in pool.map(_run_group, range(shards))
                         for record in group]
    order = {name: i for i, name in enumerate(archs)}
    return sorted(collected, key=lambda record: order[record["arch"]])


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def _fmt_wall(seconds):
    """Compact wall-clock: ``11m42s`` under an hour, ``1h02m`` above."""
    total = int(round(float(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m{secs:02d}s"


def arch_row(result) -> dict:
    """One report row: architecture, stage return codes, certificate, oracles,
    wall.

    ``stages_rc`` is one field per entry of :data:`STAGE_ORDER`, ``-`` for a
    stage the sequence never reached, so the column is a fixed-width
    fingerprint of the run and two matrices diff line by line.
    """
    by_name = {s["name"]: s["rc"] for s in result.get("stages", ())}
    validate = result.get("validate_run") or {}
    # A validate_run refusal the matrix expects is not a failed stage, and the
    # fingerprint has to say so: two matrices are diffed on this column, and a
    # bare 1 there would read as the run having broken.
    expected_refusal = bool(validate.get("expected"))

    def _cell(name):
        if name not in by_name:
            return "-"
        rc = by_name[name]
        if name == "validate_run" and expected_refusal and rc != 0:
            return f"{rc}w"
        return str(rc)

    stages_rc = ".".join(_cell(name) for name in STAGE_ORDER)
    oracles = result.get("oracle_tests") or {}
    if oracles.get("rc") is None:
        oracle_cell = "skipped"
    else:
        oracle_cell = f"{oracles['rc']} ({oracles.get('summary_line', '')})"
    # The verdict, and whether it was acted on: a FAIL the run's own waiver
    # covers is the expected outcome here, while a FAIL that nothing waives
    # would have blocked the on-node gates, and the two must not read alike.
    record = result.get("certificate") or {}
    verdict = result.get("certificate_verdict") or record.get("verdict")
    if record and not record.get("present", True):
        certificate_cell = "missing"
    elif verdict and record.get("enforced") is False:
        certificate_cell = f"{verdict} (waived)"
    else:
        certificate_cell = verdict or "-"
    return {
        "arch": result["arch"],
        "stages_rc": stages_rc,
        "certificate": certificate_cell,
        "oracles": oracle_cell,
        "wall": _fmt_wall(result.get("seconds", 0.0)),
    }


def _is_clean(result) -> bool:
    """True iff this architecture met every acceptance item of spec 3.4.

    Every stage ran and ended the way this identity requires -- exit zero, or,
    for ``validate_run`` under the waiver, the certificate refusal and nothing
    else -- the certificate exists, the held-out
    channel is marked as the slice it was evaluated on, and the oracles passed
    (or were not asked for). The certificate's VERDICT is exempt and is not
    read here: spec 3.4 records the verdict, it does not require a PASS from a
    50-step pretrain on two atoms. Its EXISTENCE is not exempt -- a run with no
    certificate has no record of what its networks reproduce.
    """
    if result.get("error"):
        return False
    record = result.get("certificate")
    if record is not None and not record.get("present", True):
        return False
    if len(result.get("stages", ())) != len(STAGE_ORDER):
        return False
    validate = result.get("validate_run") or {}
    for stage in result["stages"]:
        if stage["name"] == "certificate":
            continue
        if (stage["name"] == "validate_run" and validate.get("expected")
                and stage["rc"] != 0):
            continue
        if stage["rc"] != 0:
            return False
    if validate.get("expected") is False:
        return False
    if (result.get("slice_check") or {}).get("ok") is False:
        return False
    rc = (result.get("oracle_tests") or {}).get("rc")
    return rc in (0, None)


def matrix_exit_code(results) -> int:
    """``0`` when every architecture met the acceptance list, ``1`` otherwise.

    The matrix runs to completion whatever any one architecture does, so the
    verdict is carried by the exit status rather than by an early stop: a
    caller (a shell, a later job) sees one number for the whole pass and the
    report carries which architecture failed what.
    """
    return 0 if all(_is_clean(record) for record in results) else 1


def write_matrix_report(results, path) -> Path:
    """Write the matrix table as markdown, and the full records as JSON.

    The markdown table is the HISTORY baseline entry (columns: architecture,
    stage return codes, certificate verdict, oracle result, wall), followed by
    a findings block naming what any non-clean architecture failed. The JSON
    sidecar beside it keeps every stage's wall clock and log path and every
    artefact record, which the table cannot hold and a later comparison needs.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    results = list(results)
    rows = [arch_row(r) for r in results]
    clean = sum(1 for r in results if _is_clean(r))
    lines = [
        "# Per-architecture workflow matrix",
        "",
        "Identity: def2-svp, grid level 1, solver oneshot, 2 cells "
        "(subset sizes 1 and 2), 3 training steps, 50 pretraining steps on "
        "H and O; certificate at the production identity; held-out eval on "
        f"the species slice {HELDOUT_SPECIES_SLICE} "
        f"({SLICE_CLOSED_REACTIONS} closed reactions).",
        "",
        "The certificate is computed under the template's "
        "`fidelity.enforce: false` waiver: a FAIL verdict is the EXPECTED "
        "outcome of a 50-step pretrain on two atoms and is recorded rather "
        "than required (`waived` in the certificate column), while a stage "
        "that wrote no certificate is a stage failure. Such a run is refused "
        "by validate_run, merge_v4_arms and the figure loaders regardless of "
        "the waiver, so it can never become a quantitative result.",
        "",
        "Stage order of the `stages rc` column (`-` = never reached; `<rc>w` = "
        "a non-zero exit the matrix expects, i.e. validate_run refusing the "
        "waived certificate; the certificate's own non-zero exit does not stop "
        "the sequence, its verdict is recorded): "
        + ", ".join(STAGE_ORDER) + ".",
        "",
        f"{clean} of {len(results)} architectures completed every stage with "
        "its expected outcome, a held-out channel marked sliced, and passing "
        "oracles.",
        "",
        "| arch | stages rc | certificate | oracles | wall |",
        "|---|---|---|---|---|",
    ]
    lines += [f"| {r['arch']} | {r['stages_rc']} | {r['certificate']} | "
              f"{r['oracles']} | {r['wall']} |" for r in rows]
    findings = []
    for record in results:
        if record.get("error"):
            findings.append(f"- {record['arch']}: the sequence did not run "
                            f"({record['error']}).")
        certificate = record.get("certificate") or {}
        if certificate and not certificate.get("present", True):
            findings.append(
                f"- {record['arch']}: the certificate stage wrote no "
                f"certificate at {certificate.get('path')} -- "
                f"{certificate.get('gate_message', '')}")
        check = record.get("slice_check") or {}
        if check.get("ok") is False:
            findings.append(f"- {record['arch']}: held-out channel not marked "
                            f"sliced -- {check.get('detail', '')}")
        validate = record.get("validate_run") or {}
        if validate.get("expected") is False:
            findings.append(f"- {record['arch']}: validate_run -- "
                            f"{validate.get('detail', '')}")
    expected = [f"- {record['arch']}: validate_run: "
                f"{VALIDATE_RUN_EXPECTED_DETAIL}"
                for record in results
                if (record.get("validate_run") or {}).get("expected")
                and (record.get("validate_run") or {}).get("rc")]
    if expected:
        lines += ["", "## Expected outcomes", "",
                  "validate_run is a record layer and stays strict: it "
                  "requires a PASS certificate and ignores the waiver, so it "
                  "MUST refuse a run rendered from this template, and only "
                  "that one refusal is expected of it."] + [""] + expected
    if findings:
        lines += ["", "## Findings", ""] + findings
    lines.append("")
    path.write_text("\n".join(lines))
    sidecar = path.with_suffix(".json")
    with sidecar.open("w") as fh:
        json.dump({"stage_order": list(STAGE_ORDER),
                   "species_slice": HELDOUT_SPECIES_SLICE,
                   "slice_closed_reactions": SLICE_CLOSED_REACTIONS,
                   "n_clean": clean,
                   "exit_code": matrix_exit_code(results),
                   "results": results}, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return path
