"""xcquinox.alec.cluster._preflight — the SLURM preflight-job entrypoint.

The HPC harness submits a four-stage job graph (see ``submit.py``):

    pretrain (one job per architecture)
        |  --dependency=afterok:<pretrain>
    preflight (single job)
        |  --dependency=afterok:<pretrain>:<preflight>
    train (array job)  --->  eval (array job)   (aftercorr)

This module is the body of the **preflight job**. The rendered
``preflight.sbatch`` invokes it as::

    python -m xcquinox.alec.cluster._preflight ${RUN_DIR}

Subset selection is a *finished pre-process* — the harness does NOT run it.
The preflight CONSUMES the existing subset ledger (``subset_index_log.json``)
read-only. Its job is, on a compute node, before the train array starts:

  1. Sweep stale temp files from a prior crashed run (via ``materialize``).
  2. ``prepare_inputs(cfg)`` — build the training-point pool, load the EXISTING
     subset ledger (fail-fast on a missing required ``(metric, r)`` cell), and
     ensure CCSD external references via ``precompute_all`` (skip-if-cached).
  3. Build one :class:`~xcquinox.alec.config.TrainingSpec` per grid cell via
     :func:`xcquinox.alec.cluster.spec_builder.build_training_specs`,
     ``spec.validate()`` every spec, then ``materialize_specs``.
  4. Write ``<run_dir>/manifest.json`` (atomic — the last write).
  5. Self-check: assert every ``spec_<idx>`` file exists and the manifest
     records all ``N`` cells.

If anything is incomplete, :func:`main` returns a non-zero exit code so the
train array's ``afterok:<preflight>`` dependency correctly blocks.

The pretrained checkpoint is a harness PRODUCT of the pretrain stage (written
to ``<pretrain_root>/<run_id>/<arch>/`` before the preflight runs); the preflight does
not pre-stage or validate it — ``TrainingSpec.validate()`` only checks the
path when the directory exists.

on_precompute_failure policy
----------------------------
:class:`~xcquinox.alec.cluster.grid_config.GridConfig` carries an
``on_precompute_failure`` field (``"abort"`` / ``"drop_failed_species"``):

  - ``abort`` (default): a precompute ``RuntimeError`` blocks the whole grid —
    the preflight logs the failed-species list and exits 1.
  - ``drop_failed_species``: subsets are FIXED — the preflight cannot
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

  - :data:`_prepare_inputs`       — wraps ``inputs.prepare_inputs``
  - :data:`_build_training_specs` — wraps ``spec_builder.build_training_specs``
  - :data:`_materialize_specs`    — wraps ``materialize.materialize_specs``

:func:`main` itself is orchestration-only.
"""
from __future__ import annotations

import json
import os
import sys

from xcquinox.alec.cluster.grid_config import load_grid_config
from xcquinox.alec.cluster.domain import get_domain_profile
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

    Each ``MoleculeSpec.name`` is the Hill formula / explicit ``info['name']``
    — the same naming convention ``precompute_all`` reports its failed species
    in — so this set can be intersected directly with the failed-species set.
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


def _stage_inputs(cfg):
    """Run :data:`_prepare_inputs`, applying ``cfg.on_precompute_failure``.

    Returns ``(staged, failed_species)``. ``failed_species`` is the list of
    species whose CCSD precompute failed under the ``drop_failed_species``
    policy (empty otherwise).

    Under ``abort`` (the default) a precompute ``RuntimeError`` re-raises for
    :func:`main` to turn into exit code 1. Under ``drop_failed_species`` the
    error is parsed for its failed-species list and ``prepare_inputs`` is
    re-invoked with ``recompute_refs=False`` — this re-uses the already-built
    pool + ledger WITHOUT re-running the (just-failed) precompute, so the
    affected specs can be built and marked while the unaffected ones train.
    A ``drop_failed_species`` failure whose species list cannot be parsed is
    treated as ``abort``.
    """
    try:
        staged = _prepare_inputs(cfg)
        return staged, []
    except RuntimeError as exc:
        failed = _failed_species_from_error(exc)
        policy = cfg.on_precompute_failure
        if policy == "drop_failed_species":
            if not failed:
                # Could not parse the failed-species list — without it the
                # affected specs cannot be identified; treat as abort.
                _log(
                    "PRECOMPUTE FAILURE — on_precompute_failure='drop_failed_"
                    "species' but the failed-species list could not be parsed "
                    "from the precompute error; cannot identify affected "
                    "specs. Aborting the grid."
                )
                _log(f"precompute error detail: {exc}")
                raise
            _log(
                "PRECOMPUTE FAILURE — on_precompute_failure='drop_failed_"
                f"species'; affected specs will be marked. Failed species: "
                f"{failed}"
            )
            _log(f"precompute error detail: {exc}")
            # Re-stage with recompute_refs=False: the pool + ledger load is
            # cheap and deterministic; skipping the precompute avoids re-
            # running the work that just failed. References for non-failed
            # species remain cached on disk for their specs to consume.
            staged = _prepare_inputs(cfg, recompute_refs=False)
            _log(
                f"re-staged inputs (precompute skipped); {len(failed)} "
                f"species failed: {sorted(failed)}"
            )
            return staged, failed
        # policy == "abort" (the default) — one bad species blocks the grid.
        _log(
            "PRECOMPUTE FAILURE — on_precompute_failure='abort'; the whole "
            f"grid is blocked. Failed species: {failed or '<unparsed>'}"
        )
        _log(f"precompute error detail: {exc}")
        raise


# ---------------------------------------------------------------------------
# Run-dir provenance
# ---------------------------------------------------------------------------

def _write_ledger_provenance(run_dir: str, subset_ledger) -> None:
    """Write a provenance *copy* of the loaded subset ledger into the run dir.

    The canonical ledger is the EXISTING ``subset_index_log.json`` — consumed
    read-only. This copy (``<run_dir>/subset_ledger.json``) records, alongside
    the run, exactly which subset selection the specs were built from. It is
    never written back to the source. A write failure here is non-fatal — it
    is provenance only — so it is logged and swallowed.
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
    # is a finished pre-process — the preflight does NOT run it.
    _log("staging inputs (training-point pool, subset ledger, CCSD refs)...")
    try:
        staged, failed_species = _stage_inputs(cfg)
    except RuntimeError:
        # _stage_inputs already logged the failed-species list.
        return 1
    except (ValueError, OSError) as exc:
        _log(f"ERROR: input staging failed: {exc}")
        return 1
    _log(f"inputs staged: {len(staged.points)} training points in the pool")

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
        _log("ERROR: the grid expanded to 0 specs — nothing to materialize")
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
            [cell for cell, _ in specs], paths, run_dir
        )
    except Exception as exc:
        _log(f"ERROR: manifest write failed: {exc}")
        return 1
    _log(f"manifest written to {manifest_path}")

    # --- 7. self-check ------------------------------------------------------
    _log("self-checking materialized grid...")
    if not _self_check(run_dir, specs_dir, n):
        return 1

    _log(f"preflight SUCCEEDED: {n} specs staged + verified")
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess
    sys.exit(main())
