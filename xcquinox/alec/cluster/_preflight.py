"""xcquinox.alec.cluster._preflight — the SLURM preflight-job entrypoint.

The HPC harness submits three SLURM jobs (see ``submit.py``):

    preflight (single job)
        |  --dependency=afterok:<preflight>
    train (array job)  --->  eval (array job)

This module is the body of the **preflight job**. The rendered
``preflight.sbatch`` invokes it as::

    python -m xcquinox.alec.cluster._preflight ${RUN_DIR}

Its job is to do — on a compute node, before the train array starts — every
piece of work that must succeed for the array tasks to have something to load:

  1. Load the resolved config (``<run_dir>/resolved_config.yaml``, written by
     the ``submit`` step) and the domain profile.
  2. Stage the input artifacts (CCSD external references, descriptor caches,
     subset ledger) via :func:`xcquinox.alec.cluster.inputs.prepare_inputs`.
  3. Build one :class:`~xcquinox.alec.config.TrainingSpec` per grid cell via
     :func:`xcquinox.alec.cluster.spec_builder.build_training_specs`.
  4. ``spec.validate()`` every spec — this creates the per-spec checkpoint dir
     and fails fast on a bad spec / missing ``pretrain_checkpoint``.
  5. Materialize the specs to ``<run_dir>/specs/spec_<idx>.spec``.
  6. Write ``<run_dir>/manifest.json`` (atomic — the last write).
  7. Self-check: assert every ``spec_<idx>`` file exists and the manifest
     records all ``N`` cells.

If anything is incomplete, :func:`main` returns a non-zero exit code so the
train array's ``afterok:<preflight>`` dependency correctly blocks.

Regenerate-vs-reuse decision
----------------------------
:func:`xcquinox.alec.cluster.inputs.prepare_inputs` takes a ``regenerate``
bool. :class:`~xcquinox.alec.cluster.grid_config.GridConfig` has no such field,
so the preflight reads it **directly from the resolved-config YAML**: a
top-level ``regenerate`` boolean key, or a top-level ``mode`` string key
(``"regenerate"`` / ``"reuse"``). When neither is present the preflight
**defaults to regenerate=True** — a fresh run with no pre-staged ledger is the
common case, and ``prepare_inputs(regenerate=False)`` would otherwise fail-fast
on a missing ledger.

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
# resolved_config.yaml helpers
# ---------------------------------------------------------------------------

def _read_raw_resolved_config(path: str) -> dict:
    """Load the resolved-config file as a raw mapping (no dataclass build).

    Used only to read the optional top-level ``regenerate`` / ``mode`` keys
    that :class:`GridConfig` does not model. Mirrors ``load_grid_config``'s
    lazy-YAML / stdlib-JSON dispatch.
    """
    lower = path.lower()
    if lower.endswith((".yaml", ".yml")):
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise ImportError(
                "reading a YAML resolved config requires PyYAML — "
                "install it with `pip install pyyaml`"
            ) from exc
        with open(path) as f:
            raw = yaml.safe_load(f)
    elif lower.endswith(".json"):
        with open(path) as f:
            raw = json.load(f)
    else:
        raise ValueError(
            f"unsupported resolved-config extension for {path!r}: "
            "expected .yaml, .yml, or .json"
        )
    if not isinstance(raw, dict):
        raise ValueError(
            f"resolved config {path!r}: top-level must be a mapping, got "
            f"{type(raw).__name__}"
        )
    return raw


def _resolve_regenerate(raw_cfg: dict) -> bool:
    """Decide ``regenerate`` for :func:`prepare_inputs` from the raw config.

    Precedence:
      1. a top-level boolean ``regenerate`` key — used verbatim;
      2. a top-level string ``mode`` key — ``"reuse"`` -> ``False``,
         ``"regenerate"`` -> ``True`` (any other value is an error);
      3. neither present -> default ``True`` (regenerate).
    """
    if "regenerate" in raw_cfg:
        val = raw_cfg["regenerate"]
        if not isinstance(val, bool):
            raise ValueError(
                f"resolved config 'regenerate' must be a boolean, got "
                f"{type(val).__name__}"
            )
        return val
    if "mode" in raw_cfg:
        mode = raw_cfg["mode"]
        if mode == "reuse":
            return False
        if mode == "regenerate":
            return True
        raise ValueError(
            f"resolved config 'mode' must be 'regenerate' or 'reuse', got "
            f"{mode!r}"
        )
    return True


# ---------------------------------------------------------------------------
# drop_failed_species handling
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
    ``TrainingPoint`` species ``info['name']`` (both use Hill formula for
    compounds, atomic symbol / ``symbol+"+"`` for atoms), so they can be
    handed straight to ``prepare_inputs(drop_species=...)``.
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


def _mark_dropped_species_specs(specs, dropped_species) -> int:
    """Write a ``precompute_dropped_species`` ``failure.json`` for every spec
    whose chosen subset is affected by a dropped species.

    A spec is *affected* when, after the failed species were dropped from the
    pool, its resolved subset (the molecules its ``TrainingSpec`` carries) is
    empty — i.e. every training point it would have trained on referenced a
    dropped species, leaving it with nothing to train. Such a spec cannot run,
    so it is marked instead of being submitted blind.

    Returns the number of specs marked.
    """
    dropped = sorted(set(dropped_species))
    marked = 0
    for idx, (cell, spec) in enumerate(specs):
        molecules = getattr(spec, "molecules", ()) or ()
        if len(molecules) == 0:
            _write_failure_json(
                spec.checkpoint_dir,
                {
                    "classification": "precompute_dropped_species",
                    "index": idx,
                    "cell": {
                        "arch": cell.arch,
                        "loss": cell.loss,
                        "metric": cell.metric,
                        "subset_size": cell.subset_size,
                        "solver": cell.solver,
                    },
                    "dropped_species": dropped,
                    "detail": (
                        "the chosen subset is empty after the failed CCSD "
                        "species were dropped from the pool; this grid cell "
                        "has no training points and cannot be trained."
                    ),
                },
            )
            marked += 1
    return marked


def _stage_inputs(cfg, regenerate):
    """Run :data:`_prepare_inputs`, applying ``cfg.on_precompute_failure``.

    Returns ``(staged, dropped_species)``. ``dropped_species`` is the list of
    species dropped under the ``drop_failed_species`` policy (empty otherwise).

    Under the ``drop_failed_species`` policy a precompute ``RuntimeError`` is
    parsed for its failed-species list, then ``prepare_inputs`` is re-invoked
    with ``drop_species=<failed>`` — the deliberate survivor-pool path that
    rebuilds the ledger over the shrunk pool. A precompute failure under the
    ``abort`` policy (or a ``drop_failed_species`` failure whose species list
    cannot be parsed) re-raises the ``RuntimeError`` for :func:`main` to turn
    into exit code 1.
    """
    try:
        staged = _prepare_inputs(cfg, regenerate)
        return staged, []
    except RuntimeError as exc:
        failed = _failed_species_from_error(exc)
        policy = cfg.on_precompute_failure
        if policy == "drop_failed_species":
            if not failed:
                # Could not parse the failed-species list, so a deliberate
                # survivor-pool re-stage is impossible — treat as abort.
                _log(
                    "PRECOMPUTE FAILURE — on_precompute_failure='drop_failed_"
                    "species' but the failed-species list could not be parsed "
                    "from the precompute error; cannot build a survivor pool. "
                    "Aborting the grid."
                )
                _log(f"precompute error detail: {exc}")
                raise
            _log(
                "PRECOMPUTE FAILURE — on_precompute_failure='drop_failed_"
                f"species'; dropping failed species and re-staging. "
                f"Failed species: {failed}"
            )
            _log(f"precompute error detail: {exc}")
            # Re-stage deliberately: prepare_inputs(drop_species=...) builds
            # the survivor pool (every TrainingPoint referencing a failed
            # species removed), re-runs select_subset over it, and writes a
            # ledger with pool_was_shrunk=True / dropped_species=sorted(failed).
            staged = _prepare_inputs(
                cfg, regenerate, drop_species=tuple(failed)
            )
            _log(
                f"re-stage after drop succeeded; {len(failed)} species "
                f"dropped from the candidate pool: {sorted(failed)}"
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
        raw_cfg = _read_raw_resolved_config(cfg_path)
        domain = get_domain_profile(cfg.domain_profile)
        regenerate = _resolve_regenerate(raw_cfg)
    except (ValueError, ImportError, OSError) as exc:
        _log(f"ERROR: failed to load resolved config: {exc}")
        return 1
    _log(
        f"config loaded: domain_profile={cfg.domain_profile!r}, "
        f"regenerate={regenerate}, on_precompute_failure="
        f"{cfg.on_precompute_failure!r}"
    )

    # --- 2. stage inputs ----------------------------------------------------
    _log("staging inputs (CCSD refs, descriptors, subset ledger)...")
    try:
        staged, dropped_species = _stage_inputs(cfg, regenerate)
    except RuntimeError:
        # _stage_inputs already logged the failed-species list.
        return 1
    except (ValueError, OSError) as exc:
        _log(f"ERROR: input staging failed: {exc}")
        return 1
    _log(f"inputs staged: {len(staged.points)} training points in the pool")

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
    if dropped_species:
        marked = _mark_dropped_species_specs(specs, dropped_species)
        _log(
            f"drop_failed_species: wrote precompute_dropped_species "
            f"failure.json for {marked} affected spec(s)"
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
