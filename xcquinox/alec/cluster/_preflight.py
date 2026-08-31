"""xcquinox.alec.cluster._preflight: the SLURM preflight-job entrypoint.

The HPC harness submits a five-stage job graph (see ``submit.py``):

    datagen (single job)
        |  --dependency=afterok:<datagen>
    pretrain (one job per architecture)
        |  --dependency=afterok:<pretrain>
    preflight (single job)
        |  --dependency=afterok:<pretrain>:<preflight>
    train (array job)  --->  eval (array job)   (aftercorr)

This module is the body of the preflight job. The rendered
``preflight.sbatch`` invokes it as::

    python -m xcquinox.alec.cluster._preflight ${RUN_DIR}

Subset selection is a finished pre-process, the harness does NOT run it.
The preflight CONSUMES the existing subset ledger (``subset_index_log.json``)
read-only. Its job is, on a compute node, before the train array starts:

  1. Sweep stale temp files from a prior crashed run (via ``materialize``).
  2. ``prepare_inputs(cfg)``: build the training-point pool, load the EXISTING
     subset ledger (fail-fast on a missing required ``(metric, r)`` cell), and
     ensure CCSD external references via ``precompute_all`` (skip-if-cached).
  3. Build one :class:`~xcquinox.alec.config.TrainingSpec` per grid cell via
     :func:`xcquinox.alec.cluster.spec_builder.build_training_specs`,
     ``spec.validate()`` every spec, then ``materialize_specs``.
  4. Write ``<run_dir>/manifest.json`` (atomic, the last write).
  5. Self-check: assert every ``spec_<idx>`` file exists and the manifest
     records all ``N`` cells.
  6. (optional, ``cluster.preflight_compile_smoke``) Compile the single heaviest
     attention cell once on this exclusive node (``_train_one_spec --smoke``,
     n_steps=1). A host-OOM at that epoch-0 compile exits non-zero so the whole
     train array is blocked -- one cheap failure instead of every large-basis
     task OOMing at XLA/LLVM compile time.
  7. Sweep the per-architecture pretraining-fidelity certificates
     (``<run_dir>/pretrain/<arch>/fidelity_certificate.json``) through
     ``fidelity.gate_certificate``. Every distinct architecture of the sweep
     must carry one that releases the gate; the uncertified ones are named and
     the preflight exits non-zero, so an architecture pretrained under another
     submission, a deleted certificate, or a partial pretrain array SLURM
     reported as complete all block the train array here.

If anything is incomplete, :func:`main` returns a non-zero exit code so the
train array's ``afterok:<preflight>`` dependency correctly blocks.

The pretrained checkpoint is a harness PRODUCT of the pretrain stage (written
to ``<run_dir>/pretrain/<arch>/`` before the preflight runs); the preflight does
not pre-stage it and does not re-check its shape, ``TrainingSpec.validate()``
only checks the path when the directory exists. What the preflight does check
is the certificate the pretrain stage wrote beside it (step 7): whether those
networks were shown to reproduce their parent functional.

on_precompute_failure policy
----------------------------
:class:`~xcquinox.alec.cluster.grid_config.GridConfig` carries an
``on_precompute_failure`` field (``"abort"`` / ``"drop_failed_species"``):

  - ``abort`` (default): a precompute ``RuntimeError`` blocks the whole grid,
    the preflight logs the failed-species list and exits 1.
  - ``drop_failed_species``: subsets are FIXED, the preflight cannot
    re-select. It instead catches the ``RuntimeError``, extracts the failed
    species, builds all specs anyway (refs for non-failed species are cached),
    writes a ``precompute_failed_species`` ``failure.json`` into the checkpoint
    dir of every spec whose species union intersects the failed set (so that
    spec's train task exits fast and ``reduce_outcomes`` reports it cleanly),
    materializes ALL specs, writes the manifest, and exits 0.

Mockable seams
--------------
The three heavy calls are bound to module-level names so a test can run
:func:`main` end-to-end with the heavy work monkeypatched:

  - :data:`_prepare_inputs`: wraps ``inputs.prepare_inputs``
  - :data:`_build_training_specs`: wraps ``spec_builder.build_training_specs``
  - :data:`_materialize_specs`: wraps ``materialize.materialize_specs``

:func:`main` itself is orchestration-only.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

from xcquinox.alec import parallel
from xcquinox.alec.config import get_architecture
from xcquinox.alec.cluster.fidelity import gate_certificate
from xcquinox.alec.cluster.grid_config import load_grid_config
from xcquinox.alec.cluster.domain import get_domain_profile
from xcquinox.alec.cluster._train_task import (
    _CPU_OOM_MARKERS,
    _GPU_OOM_MARKERS,
)
from xcquinox.alec.cluster.inputs import prepare_inputs as _inputs_prepare_inputs
from xcquinox.alec.cluster.spec_builder import (
    build_training_specs as _spec_build_training_specs,
)
from xcquinox.alec.cluster.materialize import (
    materialize_specs as _materialize_materialize_specs,
    write_manifest,
)


# ---------------------------------------------------------------------------
# Mockable heavy-call seams
# ---------------------------------------------------------------------------
# These module-level references are the indirection points tests monkeypatch
# (``monkeypatch.setattr(_preflight, "_prepare_inputs", fake)``) so the
# orchestration can run end-to-end without real CCSD / SCF / pickle work.

_prepare_inputs = _inputs_prepare_inputs
_build_training_specs = _spec_build_training_specs
_materialize_specs = _materialize_materialize_specs


# ---------------------------------------------------------------------------
# Optional compile-smoke gate
# ---------------------------------------------------------------------------

def _arch_compile_size(arch_name):
    """``(network size, descriptor feature columns)`` for an architecture name.

    The two quantities that decide how large the epoch-0 fused kernel is, once
    the molecule set is fixed: the network's own size (``depth * nodes``, the
    parameter-count proxy) and the number of EXTRA feature columns its
    descriptors add to every grid row (``n_extra_features``).

    A name the registry does not carry contributes ``(0, 0)`` rather than
    raising: this selector runs inside the preflight, ahead of the train
    array, and an unresolvable name must not be the thing that blocks it. Such
    a name is refused earlier anyway, by ``validate_grid_semantics`` on the
    login node and by ``build_training_specs`` here.
    """
    try:
        arch = get_architecture(str(arch_name))
        return (int(arch.depth) * int(arch.nodes), int(arch.n_extra_features))
    except (KeyError, AttributeError, TypeError, ValueError):
        return (0, 0)


def _select_smoke_cell(specs):
    """Pick the single heaviest cell to compile-smoke. Returns ``(index, cell)``.

    ``specs`` is the preflight's ``[(cell, spec), ...]`` list. The heaviest cell
    is the one whose epoch-0 XLA/LLVM compile is most likely to exhaust node
    RAM: among the attention archs (``str(cell.arch)`` contains ``"attn"``),
    which compile the largest fused kernels, otherwise over every cell.

    Within that pool the order is TOTAL and reads only the cell's own
    properties, so the choice depends on neither the axis order nor which
    equal-subset cell happens to appear first:

      1. ``subset_size`` -- the molecule count, which dominates everything else;
      2. network size, ``depth * nodes``;
      3. descriptor feature columns, ``n_extra_features``;
      4. the architecture name, purely to make a full tie deterministic.

    Rules 2 and 3 are the correction of a real gap. With ONE attention
    architecture on the axis (through v5) the subset size identified the cell
    outright; with twelve (v6) it does not, and Python's ``max`` returns the
    FIRST maximal element -- over v6's canonical, alphabetically sorted axis
    that is ``deep_attn``, depth 4 / width 32 with NO descriptors, while
    ``deep_combined_attn`` sits at the same shape with four descriptor columns.
    The descriptor-carrying attention forms are both the heavier compile and
    the architectures whose parent offsets SPEC_pretrain_fidelity_program.md
    Section 2 records as the largest, so they are exactly the cells the gate
    exists to fail cheaply on.
    """
    attn = [
        (i, cell) for i, (cell, _spec) in enumerate(specs)
        if "attn" in str(cell.arch)
    ]
    pool = attn if attn else [(i, cell) for i, (cell, _spec) in enumerate(specs)]

    def _weight(item):
        _i, cell = item
        size, columns = _arch_compile_size(cell.arch)
        return (int(cell.subset_size), size, columns, str(cell.arch))

    return max(pool, key=_weight)


def _compile_smoke_impl(specs, paths, run_dir) -> bool:
    """Compile-smoke the single heaviest grid cell before the train array.

    ``specs`` is the preflight's ``[(cell, spec), ...]`` list; ``paths`` are the
    same-indexed materialized spec-file paths. The cell is chosen by
    :func:`_select_smoke_cell`, whose docstring states the order. That spec is
    run through ``_train_one_spec --smoke`` (n_steps=1, throwaway checkpoint
    dir) on the CPU so the heaviest per-molecule kernel is actually compiled
    once on this exclusive preflight node.

    Returns ``True`` iff the probe compiled and ran its single epoch without a
    host OOM; ``False`` on a host/GPU OOM or any other non-completion. A ``False``
    return makes :func:`main` exit non-zero so the train array's ``afterok``
    dependency blocks -- one cheap failure instead of the whole array OOMing at
    compile time (the 6-311++G(3df,2pd)+grid3 regression this gate guards).
    """
    i_heavy, heavy = _select_smoke_cell(specs)
    _size, _columns = _arch_compile_size(heavy.arch)
    _log(
        f"compile-smoke: heaviest cell is index {i_heavy} "
        f"(arch={heavy.arch!r}, subset_size={heavy.subset_size}, "
        f"depth*nodes={_size}, descriptor columns={_columns}); compiling it "
        f"once via _train_one_spec --smoke on the CPU"
    )

    # Reproduce the PRODUCTION train-node compile environment so the gate is a
    # faithful proxy. The probe MUST NOT inherit the preflight shell's
    # OMP/MKL/OPENBLAS=$SLURM_CPUS_PER_TASK (24) with no XLA trims: at that thread
    # count the heaviest attention cell's LLVM parallel codegen fails to spawn its
    # threads (``pthread_create failed`` -> "Resource tracker became defunct",
    # rc -11), a FALSE-POSITIVE OOM block -- even though production training compiles
    # the same de-fused per-molecule kernel fine at BLAS = SLURM_CPUS_ON_NODE/12 with
    # the compile-memory XLA trims (train_eval_inline_cpu.sbatch.tmpl). Match
    # production here via the single-source ``parallel._thread_env``. The node core
    # count and /12 slice mirror the train template exactly.
    cores = int(os.environ.get("SLURM_CPUS_ON_NODE") or os.cpu_count() or 12)
    blas_threads = max(1, cores // 12)
    probe_env = {**os.environ,
                 **parallel._thread_env(blas_threads, bound_worker=False)}
    # The probe mirrors the train array's environment: strip any bind
    # request or slot inherited from THIS process so the probe cannot be
    # pinned to a slice the array would not be.
    probe_env.pop(parallel.WORKER_BIND_CPUS_ENV, None)
    probe_env.pop(parallel.WORKER_SLOT_ENV, None)

    proc = subprocess.run(
        [
            sys.executable, "-m", "xcquinox.alec._train_one_spec",
            paths[i_heavy], "--device", "cpu", "--smoke", "--no-progress",
        ],
        capture_output=True, text=True, env=probe_env,
    )
    stdout = proc.stdout or ""
    text = stdout + "\n" + (proc.stderr or "")
    rc = proc.returncode
    tail = text[-500:]

    # Persist the FULL probe output. The 500-char ``tail`` logged below drops the
    # head where the actual OOM/pthread marker + traceback live, which is exactly
    # what made this failure hard to diagnose off-cluster. Best-effort: a write
    # failure must never change the gate verdict.
    try:
        probe_log = os.path.join(run_dir, "logs", "compile_smoke_probe.out")
        os.makedirs(os.path.dirname(probe_log), exist_ok=True)
        with open(probe_log, "w") as fh:
            fh.write(
                f"# compile-smoke probe: spec index {i_heavy}, "
                f"arch={heavy.arch!r}, subset_size={heavy.subset_size}, "
                f"blas_threads={blas_threads}, rc={rc}\n"
                f"# argv: _train_one_spec {paths[i_heavy]} --device cpu --smoke\n\n"
            )
            fh.write(text)
    except OSError as exc:
        _log(f"compile-smoke: could not persist full probe output ({exc})")
    # The worker emits ``{"kind": "done", ...}`` (json.dumps -> spaced form) at a
    # clean finish; accept the compact form too so a trivial JSON-whitespace
    # difference cannot misclassify a genuinely-completed probe.
    done_seen = ('"kind": "done"' in stdout) or ('"kind":"done"' in stdout)

    # Classify on the OOM TEXT signature, not the exit signal. A real host/GPU
    # OOM prints a marker ("Cannot allocate memory", "std::bad_alloc",
    # "RESOURCE_EXHAUSTED", ...) and crashes BEFORE the one-epoch completion. A
    # glibc heap-corruption abort at process TEARDOWN ("corrupted size vs.
    # prev_size" / "double free", SIGABRT -6/134 or SIGSEGV -11, with NO OOM
    # text) is a known-benign artifact of this JAX/PySCF/OpenBLAS stack; once the
    # worker has printed its completion marker the compile has already succeeded,
    # so a teardown signal must NOT be read as a compile OOM (which would wrongly
    # block the whole array). done_seen => the epoch finished => no OOM occurred.
    oom_text = any(m in text for m in (_CPU_OOM_MARKERS + _GPU_OOM_MARKERS))
    if oom_text:
        _log(
            f"compile-smoke FAILED: host-OOM signature in the heaviest cell "
            f"(index {i_heavy}, arch={heavy.arch!r}, "
            f"subset_size={heavy.subset_size}) output (rc={rc}). Tail:\n{tail}"
        )
        return False
    if done_seen or rc == 0:
        _log(
            f"compile-smoke PASSED: the heaviest cell (index {i_heavy}, "
            f"arch={heavy.arch!r}, subset_size={heavy.subset_size}) compiled + "
            f"ran one epoch (rc={rc}; a teardown signal after completion is benign)"
        )
        return True
    _log(
        f"compile-smoke FAILED: the heaviest cell (index {i_heavy}, "
        f"arch={heavy.arch!r}, subset_size={heavy.subset_size}) did not complete "
        f"one epoch and shows no OOM text (rc={rc}); blocking the array. "
        f"Tail:\n{tail}"
    )
    return False


# Seam: tests monkeypatch ``_preflight._compile_smoke`` (main() looks it up as a
# module global, so a patched value is honored) to run the gate wiring without
# spawning the real worker subprocess.
_compile_smoke = _compile_smoke_impl


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    """Emit a legible progress line to stdout (the preflight SLURM log).

    Project rule: long-running steps must emit progress so a running job is
    visibly distinct from a hung one. ``flush=True`` so the line appears in the
    SLURM ``.out`` immediately rather than sitting in a buffer.
    """
    print(f"[preflight] {msg}", flush=True)


# ---------------------------------------------------------------------------
# precompute-failure handling
# ---------------------------------------------------------------------------

def _failed_species_from_error(exc: Exception) -> list[str]:
    """Extract the failed-species list from a precompute ``RuntimeError``.

    ``external_refs.precompute_all`` raises (see its body) with a message of
    the exact form::

        Cell 0.5 pre-compute failed for N species: ['A', 'B']. Inspect ...

    where ``['A', 'B']`` is the Python ``repr`` of the ``failures`` list of
    ``SpeciesEntry.name`` strings. This parses the bracketed portion back to a
    list of names; on any parse failure an empty list is returned (the message
    itself is still logged verbatim by the caller). The names match
    ``MoleculeSpec.name`` (both use Hill formula for compounds, atomic symbol /
    ``symbol+"+"`` for atoms / cations), so they can be intersected directly
    against a spec's molecule set.
    """
    msg = str(exc)
    marker = "species: ["
    start = msg.find(marker)
    if start == -1:
        return []
    start += len(marker)
    end = msg.find("]", start)
    if end == -1:
        return []
    inner = msg[start:end]
    out: list[str] = []
    for part in inner.split(","):
        token = part.strip().strip("'\"")
        if token:
            out.append(token)
    return out


def _write_failure_json(checkpoint_dir: str, payload: dict) -> None:
    """Write a ``failure.json`` into a spec's checkpoint dir.

    The checkpoint dir is created if absent (``spec.validate()`` would also
    create it, but the drop path may mark a spec before validation runs).
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, "failure.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _spec_species(spec) -> set[str]:
    """Return the set of molecule names carried by a ``TrainingSpec``.

    Each ``MoleculeSpec.name`` is the Hill formula / explicit ``info['name']``:
    the same naming convention ``precompute_all`` reports its failed species
    in, so this set can be intersected directly with the failed-species set.
    """
    return {
        getattr(m, "name", None)
        for m in (getattr(spec, "molecules", ()) or ())
        if getattr(m, "name", None) is not None
    }


def _mark_failed_species_specs(specs, failed_species) -> int:
    """Write a ``precompute_failed_species`` ``failure.json`` for every spec
    whose molecule species union intersects the failed-species set.

    The harness cannot re-select a subset (subsets are fixed), so a spec that
    references a species whose CCSD reference failed to precompute cannot
    train. It is marked instead of being submitted blind: the per-spec train
    task reads ``failure.json`` and exits fast, and ``reduce_outcomes`` reports
    it cleanly. Unaffected specs are left untouched and train normally.

    Returns the number of specs marked.
    """
    failed = set(failed_species)
    failed_sorted = sorted(failed)
    marked = 0
    for idx, (cell, spec) in enumerate(specs):
        affected = sorted(_spec_species(spec) & failed)
        if not affected:
            continue
        _write_failure_json(
            spec.checkpoint_dir,
            {
                "classification": "precompute_failed_species",
                "index": idx,
                "cell": {
                    "arch": cell.arch,
                    "loss": cell.loss,
                    "metric": cell.metric,
                    "subset_size": cell.subset_size,
                    "solver": cell.solver,
                },
                "species": affected,
                "failed_species": failed_sorted,
                "detail": (
                    "this grid cell's subset references species whose CCSD "
                    "external reference failed to pre-compute; the harness "
                    "cannot re-select a fixed subset, so the train task is "
                    "marked failed instead of submitted blind."
                ),
            },
        )
        marked += 1
    return marked


def _stage_inputs(cfg, run_dir=None):
    """Run :data:`_prepare_inputs`, applying ``cfg.on_precompute_failure``.

    Returns ``(staged, failed_species)``. ``failed_species`` is the list of
    species whose CCSD precompute failed under the ``drop_failed_species``
    policy (empty otherwise).

    Under ``abort`` (the default) a precompute ``RuntimeError`` re-raises for
    :func:`main` to turn into exit code 1. Under ``drop_failed_species`` the
    error is parsed for its failed-species list and ``prepare_inputs`` is
    re-invoked with ``recompute_refs=False``: this re-uses the already-built
    pool + ledger WITHOUT re-running the (just-failed) precompute, so the
    affected specs can be built and marked while the unaffected ones train.
    A ``drop_failed_species`` failure whose species list cannot be parsed is
    treated as ``abort``.
    """
    try:
        staged = _prepare_inputs(cfg, run_dir=run_dir)
        return staged, []
    except RuntimeError as exc:
        failed = _failed_species_from_error(exc)
        policy = cfg.on_precompute_failure
        if policy == "drop_failed_species":
            if not failed:
                # Could not parse the failed-species list, without it the
                # affected specs cannot be identified; treat as abort.
                _log(
                    "PRECOMPUTE FAILURE, on_precompute_failure='drop_failed_"
                    "species' but the failed-species list could not be parsed "
                    "from the precompute error; cannot identify affected "
                    "specs. Aborting the grid."
                )
                _log(f"precompute error detail: {exc}")
                raise
            _log(
                "PRECOMPUTE FAILURE, on_precompute_failure='drop_failed_"
                f"species'; affected specs will be marked. Failed species: "
                f"{failed}"
            )
            _log(f"precompute error detail: {exc}")
            # Re-stage with recompute_refs=False: the pool + ledger load is
            # cheap and deterministic; skipping the precompute avoids re-
            # running the work that just failed. References for non-failed
            # species remain cached on disk for their specs to consume.
            staged = _prepare_inputs(cfg, recompute_refs=False, run_dir=run_dir)
            _log(
                f"re-staged inputs (precompute skipped); {len(failed)} "
                f"species failed: {sorted(failed)}"
            )
            return staged, failed
        # policy == "abort" (the default), one bad species blocks the grid.
        _log(
            "PRECOMPUTE FAILURE, on_precompute_failure='abort'; the whole "
            f"grid is blocked. Failed species: {failed or '<unparsed>'}"
        )
        _log(f"precompute error detail: {exc}")
        raise


# ---------------------------------------------------------------------------
# Run-dir provenance
# ---------------------------------------------------------------------------

def _write_ledger_provenance(run_dir: str, subset_ledger) -> None:
    """Write a provenance copy of the loaded subset ledger into the run dir.

    The canonical ledger is the EXISTING ``subset_index_log.json``: consumed
    read-only. This copy (``<run_dir>/subset_ledger.json``) records, alongside
    the run, exactly which subset selection the specs were built from. It is
    never written back to the source. A write failure here is non-fatal, it
    is provenance only, so it is logged and swallowed.
    """
    path = os.path.join(run_dir, "subset_ledger.json")
    try:
        with open(path, "w") as f:
            json.dump(subset_ledger, f, indent=2, sort_keys=True)
            f.write("\n")
    except (OSError, TypeError, ValueError) as exc:
        _log(f"WARNING: could not write subset-ledger provenance copy ({exc})")
        return
    _log(f"wrote subset-ledger provenance copy to {path}")


# ---------------------------------------------------------------------------
# Self-check
# ---------------------------------------------------------------------------

def _self_check(run_dir, specs_dir, n) -> bool:
    """Verify exactly ``N`` spec files and a complete manifest exist.

    Uses an explicit per-index existence check (the zero-pad ``width`` is read
    back from the manifest, NOT inferred from a glob) so a stray ``spec_*``
    file from a prior larger grid cannot mask a missing real spec.

    Returns ``True`` iff every check passes; logs the first failure otherwise.
    """
    manifest_path = os.path.join(run_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        _log(f"SELF-CHECK FAILED: manifest.json missing at {manifest_path}")
        return False
    try:
        with open(manifest_path) as f:
            manifest = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        _log(f"SELF-CHECK FAILED: manifest.json unreadable ({exc})")
        return False

    m_n = manifest.get("n_specs")
    if m_n != n:
        _log(
            f"SELF-CHECK FAILED: manifest n_specs={m_n!r} but the grid has "
            f"{n} cells"
        )
        return False
    spec_entries = manifest.get("specs")
    if not isinstance(spec_entries, list) or len(spec_entries) != n:
        got = len(spec_entries) if isinstance(spec_entries, list) else spec_entries
        _log(
            "SELF-CHECK FAILED: manifest 'specs' list does not record all "
            f"{n} cells (got {got!r})"
        )
        return False

    width = manifest.get("width")
    if not isinstance(width, int) or width < 1:
        _log(f"SELF-CHECK FAILED: manifest 'width' is invalid ({width!r})")
        return False

    for idx in range(n):
        fname = f"spec_{idx:0{width}d}.spec"
        fpath = os.path.join(specs_dir, fname)
        if not os.path.isfile(fpath):
            _log(
                f"SELF-CHECK FAILED: expected spec file {fname} is missing "
                f"from {specs_dir}"
            )
            return False
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    """Preflight-job entrypoint. Returns a process exit code (0 = success).

    Parameters
    ----------
    argv : list[str] | None
        The argument vector (run dir at ``argv[0]``). ``None`` -> ``sys.argv[1:]``.

    Returns
    -------
    int
        ``0`` on a fully-staged, self-checked run; ``1`` on any failure (so the
        train array's ``afterok`` dependency blocks).
    """
    if argv is None:
        argv = sys.argv[1:]
    if len(argv) < 1:
        _log("ERROR: no run directory given; usage: _preflight <run_dir>")
        return 1
    run_dir = os.path.abspath(argv[0])
    _log(f"starting preflight for run_dir={run_dir}")

    # --- 1. load config + domain -------------------------------------------
    cfg_path = os.path.join(run_dir, "resolved_config.yaml")
    if not os.path.isfile(cfg_path):
        _log(f"ERROR: resolved_config.yaml not found at {cfg_path}")
        return 1
    try:
        cfg = load_grid_config(cfg_path)
        domain = get_domain_profile(cfg.domain_profile)
    except (ValueError, ImportError, OSError) as exc:
        _log(f"ERROR: failed to load resolved config: {exc}")
        return 1
    _log(
        f"config loaded: domain_profile={cfg.domain_profile!r}, "
        f"on_precompute_failure={cfg.on_precompute_failure!r}"
    )

    # --- 2. stage inputs ----------------------------------------------------
    # prepare_inputs builds the pool, loads the EXISTING subset ledger
    # (fail-fast on a missing required (metric, r) cell), and ensures the CCSD
    # external references via precompute_all (skip-if-cached). Subset selection
    # is a finished pre-process, the preflight does NOT run it.
    _log("staging inputs (training-point pool, subset ledger, CCSD refs)...")
    try:
        staged, failed_species = _stage_inputs(cfg, run_dir)
    except RuntimeError:
        # _stage_inputs already logged the failed-species list.
        return 1
    except (ValueError, OSError) as exc:
        _log(f"ERROR: input staging failed: {exc}")
        return 1
    _log(f"inputs staged: {len(staged.points)} training points in the pool")
    refs = tuple(getattr(staged, "reference_species", ()) or ())
    if refs:
        canonical = int(getattr(staged, "canonical_species_count", 0) or 0)
        scope = f" of the {canonical} canonical" if canonical else ""
        _log(
            f"CCSD references ensured for the {len(refs)}{scope} species the "
            f"run's cells name: {', '.join(refs)}"
        )
    outside = tuple(getattr(staged, "cell_species_without_reference", ()) or ())
    if outside:
        named = ", ".join(f"{n} (charge {c:+d}, 2S {sp})" for n, c, sp in outside)
        _log(
            f"{len(outside)} species the run's cells name carry no reference "
            f"in the canonical set and train without a density target, as in "
            f"every run before: {named}"
        )

    # Provenance copy of the consumed (read-only) subset ledger.
    _write_ledger_provenance(run_dir, staged.subset_ledger)

    # --- 3. build specs -----------------------------------------------------
    _log("building TrainingSpecs from the subset ledger...")
    try:
        specs = _build_training_specs(
            staged.points, staged.subset_ledger, cfg, domain, run_dir
        )
    except (ValueError, KeyError) as exc:
        _log(f"ERROR: spec building failed: {exc}")
        return 1
    n = len(specs)
    if n == 0:
        _log("ERROR: the grid expanded to 0 specs, nothing to materialize")
        return 1
    _log(f"built {n} TrainingSpec(s)")

    # --- drop_failed_species marking ---------------------------------------
    if failed_species:
        marked = _mark_failed_species_specs(specs, failed_species)
        _log(
            f"drop_failed_species: wrote precompute_failed_species "
            f"failure.json for {marked} of {n} spec(s) whose subset "
            f"references a failed species ({sorted(failed_species)})"
        )

    # --- 4. validate every spec --------------------------------------------
    _log("validating specs (creates per-spec checkpoint dirs)...")
    for idx, (cell, spec) in enumerate(specs):
        try:
            spec.validate()
        except Exception as exc:  # validate() may raise ValueError or worse
            _log(
                f"ERROR: spec {idx} failed validation for cell "
                f"(arch={cell.arch!r}, loss={cell.loss!r}, "
                f"metric={cell.metric!r}, subset_size={cell.subset_size}, "
                f"solver={cell.solver!r}): {exc}"
            )
            return 1
    _log(f"all {n} specs validated")

    # --- 5. materialize -----------------------------------------------------
    # materialize_specs purges stale spec_* / temp files from a prior crashed
    # run, then writes spec_0000 .. spec_(N-1).
    specs_dir = os.path.join(run_dir, "specs")
    _log(f"materializing {n} spec file(s) to {specs_dir}...")
    try:
        paths = _materialize_specs(specs, specs_dir)
    except OSError as exc:
        idx = getattr(exc, "_spec_index", "?")
        errno = getattr(exc, "errno", "?")
        _log(
            f"ERROR: materialization failed (spec index {idx}, errno "
            f"{errno}): {exc}"
        )
        return 1
    except Exception as exc:
        _log(f"ERROR: materialization failed: {exc}")
        return 1
    _log(f"materialized {len(paths)} spec file(s)")

    # --- 6. manifest (final, atomic write) ---------------------------------
    _log("writing manifest.json...")
    try:
        manifest_path = write_manifest(
            [cell for cell, _ in specs], paths, run_dir, cfg=cfg,
        )
    except Exception as exc:
        _log(f"ERROR: manifest write failed: {exc}")
        return 1
    _log(f"manifest written to {manifest_path}")

    # --- 7. self-check ------------------------------------------------------
    _log("self-checking materialized grid...")
    if not _self_check(run_dir, specs_dir, n):
        return 1

    # --- 8. optional compile-smoke gate ------------------------------------
    # When enabled, compile the single heaviest attention cell once on this
    # exclusive node before the train array launches. A host-OOM at that compile
    # exits the preflight non-zero so the array's afterok dependency blocks (one
    # cheap failure instead of the whole array OOMing at XLA/LLVM compile time).
    if getattr(getattr(cfg, "cluster", None), "preflight_compile_smoke", False):
        _log("compile-smoke gate enabled: probing the heaviest cell before the "
             "array ...")
        if not _compile_smoke(specs, paths, run_dir):
            _log("ERROR: compile-smoke gate FAILED -- blocking the train array")
            return 1
        _log("compile-smoke gate PASSED")

    # --- 9. per-architecture fidelity certificates -------------------------
    # The preflight is submitted afterok on the pretrain array, so by this
    # point every distinct architecture has been certified on its own node.
    # This sweep is the run-level cross-check: it catches an architecture that
    # was pretrained under a different submission, a certificate that was
    # deleted, and a partial pretrain array that SLURM reported as complete.
    # ``gate_certificate`` honours a run configured with
    # ``fidelity.enforce: false`` (the workflow-verification matrix), which
    # ``validate_run``, ``merge_v4_arms`` and the figure suite still refuse.
    archs = sorted(set(cfg.sweep.arch))
    uncertified = []
    for arch in archs:
        allowed, message = gate_certificate(run_dir, arch)
        if allowed:
            _log(f"fidelity gate for arch {arch}: {message}")
            continue
        uncertified.append(arch)
        _log(f"ERROR: fidelity certificate for arch {arch} does not release "
             f"the gate: {message}")
    if uncertified:
        _log(f"ERROR: fidelity gate FAILED for {len(uncertified)}/"
             f"{len(archs)} architecture(s) ({', '.join(uncertified)}) -- "
             "blocking the train array")
        return 1
    _log(f"fidelity gate PASSED: {len(archs)}/{len(archs)} architecture "
         "certificate(s) released the gate")

    _log(f"preflight SUCCEEDED: {n} specs staged + verified")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess
    # The stage's verdict is the status this process hands SLURM, and
    # JAX's atexit teardown can abort the interpreter AFTER main() has
    # returned it (cluster job 2134455: the pretrain worker logged
    # "pretrain SUCCEEDED" and then died in glibc's "corrupted size vs.
    # prev_size", rc -6, so the stage read as FAILED and the dependent
    # array never ran). run_and_exit flushes and leaves through os._exit,
    # so the status is the verdict. See xcquinox/alec/cluster/_exit.py.
    # Imported HERE rather than in the module body: several of these
    # modules pin what their import pulls in (``fidelity`` is held to a
    # whitelist of cheap readers so the on-node gates can read a
    # certificate without the training stack), and the helper is needed
    # only when the module is RUN.
    from xcquinox.alec.cluster._exit import run_and_exit
    run_and_exit(main)
