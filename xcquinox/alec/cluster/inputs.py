"""xcquinox.alec.cluster.inputs — input-artifact staging for the HPC harness.

This module prepares the harness's input artifacts before any training job is
submitted:

  - the per-species CCSD external references (``external_refs_dir/<name>.npz``),
  - the descriptor / reference-histogram caches,
  - the **subset ledger** — the JSON file recording, per ``(metric, subset_size)``
    grid-cell pair, which :class:`~xcquinox.alec.training_points.TrainingPoint`
    objects that cell trains on.

Two modes
---------
``prepare_inputs(cfg, regenerate=True)`` does the heavy precompute: CCSD
references via :func:`xcquinox.alec.external_refs.precompute_all`, descriptor
extraction, reference-histogram construction, and one
:func:`xcquinox.alec.subset_selection.select_subset` enumeration per distinct
``(metric, subset_size)`` pair appearing in the grid. It then writes the subset
ledger as one whole-file atomic write.

``prepare_inputs(cfg, regenerate=False)`` reuses pre-staged artifacts: it loads
the previously-written ledger and fails fast (with a named-cause
:class:`ValueError`) on any inconsistency with the current config / pool.

The subset-ledger schema
------------------------
This is the canonical schema consumed by
:func:`xcquinox.alec.cluster.spec_builder.build_training_specs`::

    {
        "pool_fingerprint": "<hex sha256 from pool_fingerprint(points)>",
        "basis": "<basis>",
        "grid_level": <int>,
        "bh76_mode": "<reaction_energy|barrier_height>",
        "pool_was_shrunk": <bool>,
        "dropped_species": [<name>, ...],
        "entries": {
            "<metric>:<subset_size>": {
                "metric": "<metric>",
                "subset_size": <int>,
                "point_names": [<name>, ...],
            },
            ...
        },
    }

The ``entries`` keys are JSON-string ``"<metric>:<subset_size>"`` keys (the
spec-builder's ``_lookup_entry`` accepts both that and the in-memory tuple
form). The ``basis`` / ``grid_level`` / ``bh76_mode`` / ``pool_was_shrunk`` /
``dropped_species`` top-level fields exist so reuse-mode can validate the
ledger against the current config.

The ``drop_failed_species`` path
--------------------------------
``prepare_inputs(cfg, regenerate=True, drop_species=("F2", "O3"))`` runs the
**deliberate survivor-pool** path: it builds the pool, removes every
:class:`~xcquinox.alec.training_points.TrainingPoint` whose species set
intersects ``drop_species`` (an AE point with a dropped compound/anchor, a
BH76/IP13 point referencing a dropped species — all get dropped), re-runs
descriptors / reference histograms / ``select_subset`` over the *survivor*
pool, and writes a ledger whose ``pool_fingerprint`` is computed from the
survivor pool, with ``pool_was_shrunk = True`` and
``dropped_species = sorted(drop_species)``. When ``drop_species`` is empty
(the default) the no-shrink path runs and those fields stay ``False`` / ``[]``.

Mockable seams
--------------
The four heavy calls are bound to module-level names so tests can monkeypatch
them without doing real CCSD / SCF work:

  - :data:`_precompute_all`        — wraps ``external_refs.precompute_all``
  - :data:`_build_species_union`   — wraps ``external_refs.build_species_union``
  - :data:`_extract_descriptors`   — wraps ``subset_selection.extract_descriptors_for_species``
  - :data:`_build_reference_histograms` — wraps ``subset_selection.build_reference_histograms``
  - :data:`_select_subset`         — wraps ``subset_selection.select_subset``

``prepare_inputs`` itself is orchestration-only.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass

from xcquinox.alec import external_refs as _external_refs
from xcquinox.alec import subset_selection as _subset_selection
from xcquinox.alec.training_points import (
    build_dfs_pool_points,
    species_union_from_points,
)
from xcquinox.alec.cluster.grid_config import GridConfig, expand_grid
from xcquinox.alec.cluster.spec_builder import pool_fingerprint


# ---------------------------------------------------------------------------
# Mockable heavy-call seams
# ---------------------------------------------------------------------------
# These module-level references are the indirection points tests monkeypatch
# (``monkeypatch.setattr(inputs, "_precompute_all", fake)``) so the regen-mode
# pipeline can run without real CCSD / SCF compute.

_precompute_all = _external_refs.precompute_all
_build_species_union = _external_refs.build_species_union
_extract_descriptors = _subset_selection.extract_descriptors_for_species
_build_reference_histograms = _subset_selection.build_reference_histograms
_select_subset = _subset_selection.select_subset
_concatenate_point_descriptors = _subset_selection.concatenate_point_descriptors


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StagedInputs:
    """Result of :func:`prepare_inputs`.

    Attributes
    ----------
    points : list[TrainingPoint]
        The training-point pool — the full pool from ``build_dfs_pool_points``
        in the no-drop path, or the *survivor* pool (dropped-species points
        removed) when ``prepare_inputs`` was called with ``drop_species``.
    subset_ledger : dict
        The name-based subset ledger — see the module docstring for the
        canonical schema. Ready to hand to
        :func:`xcquinox.alec.cluster.spec_builder.build_training_specs`.
    """

    points: list
    subset_ledger: dict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _entry_key(metric: str, subset_size: int) -> str:
    """The canonical JSON-string ledger entry key for a ``(metric, r)`` pair."""
    return f"{metric}:{int(subset_size)}"


def _metric_size_pairs(cfg: GridConfig) -> list[tuple[str, int]]:
    """Distinct ``(metric, subset_size)`` pairs across the expanded grid.

    Deterministically ordered (sorted) so the regen pipeline runs
    ``select_subset`` in a reproducible sequence.
    """
    pairs = {(c.metric, int(c.subset_size)) for c in expand_grid(cfg)}
    return sorted(pairs)


def _point_species_names(tp) -> set:
    """The set of species ``info['name']`` carried by a TrainingPoint.

    Each entry of ``TrainingPoint.species`` is an ASE ``Atoms`` whose
    ``info['name']`` is the species' canonical name — the same naming
    ``external_refs.build_species_union`` / ``SpeciesEntry.name`` use (Hill
    formula for compounds, atomic symbol or ``symbol+"+"`` for atoms /
    cations). ``TrainingPoint.__post_init__`` guarantees every species has
    ``info['name']``, so this is total.
    """
    return {s.info["name"] for s in tp.species}


def _survivor_pool(points, drop_species: tuple) -> list:
    """Return ``points`` with every TrainingPoint that references a dropped
    species removed.

    A point is dropped when its species-name set intersects ``drop_species``
    — that covers an AE point whose compound (or H/Li anchor) was a failed
    CCSD species and a BH76/IP13 point referencing any failed species.
    """
    drop = set(drop_species)
    return [tp for tp in points if not (_point_species_names(tp) & drop)]


def _write_ledger_atomic(ledger: dict, path: str) -> None:
    """Whole-file atomic write of the ledger JSON.

    Uses ``tempfile.mkstemp`` in the destination directory + ``os.replace`` so
    an interrupted write cannot leave a corrupt partial ledger that reuse-mode
    would mis-parse (matches the atomic-write precedent in
    ``external_refs.py``).
    """
    path = os.path.abspath(path)
    out_dir = os.path.dirname(path) or "."
    os.makedirs(out_dir, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=out_dir, suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(ledger, f, indent=2, sort_keys=True)
        os.replace(tmp_name, path)
    except Exception:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


# ---------------------------------------------------------------------------
# Regenerate mode
# ---------------------------------------------------------------------------

def _prepare_inputs_regenerate(
    cfg: GridConfig, drop_species: tuple = ()
) -> StagedInputs:
    """Heavy-precompute path of :func:`prepare_inputs` — see its docstring.

    When ``drop_species`` is non-empty the pool is shrunk to the survivor
    pool (dropped-species points removed) BEFORE descriptors / histograms /
    ``select_subset`` run, and the ledger records the shrink truthfully.
    """
    points = build_dfs_pool_points(bh76_mode=cfg.bh76_mode)

    # --- CCSD external references ------------------------------------------
    # build_species_union assembles its own canonical species set (training +
    # probe + HBPT); precompute_all is skip-if-cached / idempotent and raises
    # RuntimeError with a failed-species list on failure.
    species = _build_species_union()
    _precompute_all(
        species,
        cache_dir=cfg.inputs.external_refs_dir,
        basis=cfg.inputs.basis,
        grid_level=cfg.inputs.grid_level,
    )

    # --- survivor-pool shrink (drop_failed_species path) -------------------
    # Remove every point referencing a failed CCSD species so descriptors,
    # reference histograms, and select_subset all run over the survivor pool.
    if drop_species:
        points = _survivor_pool(points, drop_species)

    # --- descriptors + reference histograms --------------------------------
    # Extract per-species descriptors over the union of every point's species,
    # then concatenate per point to get the candidate pool select_subset
    # enumerates over.
    species_atoms = species_union_from_points(points)
    species_descriptors = _extract_descriptors(
        species_atoms,
        basis=cfg.inputs.basis,
        grid_level=cfg.inputs.grid_level,
        cache_dir=cfg.inputs.descriptor_cache,
    )
    pool_descriptors = _concatenate_point_descriptors(
        points, species_descriptors
    )
    h_ref, edges = _build_reference_histograms(pool_descriptors)

    # --- subset selection per (metric, subset_size) pair -------------------
    entries: dict = {}
    for metric, subset_size in _metric_size_pairs(cfg):
        result = _select_subset(
            pool_descriptors,
            edges,
            h_ref,
            r=subset_size,
            metric=metric,
        )
        # select_subset returns (best_combo, best_val[, vals, idx_array]);
        # best_combo is integer POSITIONS into the pool — convert to names.
        best_combo = result[0]
        point_names = [points[i].name for i in best_combo]
        entries[_entry_key(metric, subset_size)] = {
            "metric": metric,
            "subset_size": int(subset_size),
            "point_names": point_names,
        }

    ledger = {
        # pool_fingerprint is computed from the (possibly survivor) pool, so
        # reuse-mode validates against exactly the pool the ledger describes.
        "pool_fingerprint": pool_fingerprint(points),
        "basis": cfg.inputs.basis,
        "grid_level": int(cfg.inputs.grid_level),
        "bh76_mode": cfg.bh76_mode,
        "pool_was_shrunk": bool(drop_species),
        "dropped_species": sorted(drop_species),
        "entries": entries,
    }
    _write_ledger_atomic(ledger, cfg.inputs.subset_ledger_path)
    return StagedInputs(points=points, subset_ledger=ledger)


# ---------------------------------------------------------------------------
# Reuse mode
# ---------------------------------------------------------------------------

def _prepare_inputs_reuse(cfg: GridConfig) -> StagedInputs:
    """Reuse path of :func:`prepare_inputs` — see its docstring."""
    points = build_dfs_pool_points(bh76_mode=cfg.bh76_mode)
    ledger_path = cfg.inputs.subset_ledger_path

    # --- load the ledger ----------------------------------------------------
    if not os.path.isfile(ledger_path):
        raise ValueError(
            f"reuse-mode subset ledger not found at {ledger_path!r}; run "
            "prepare_inputs(cfg, regenerate=True) first to stage it."
        )
    try:
        with open(ledger_path) as f:
            ledger = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(
            f"reuse-mode subset ledger at {ledger_path!r} is unreadable / "
            f"unparseable ({exc}); regenerate it with "
            "prepare_inputs(cfg, regenerate=True)."
        ) from exc
    if not isinstance(ledger, dict):
        raise ValueError(
            f"reuse-mode subset ledger at {ledger_path!r} is not a JSON "
            f"object (got {type(ledger).__name__}); regenerate it."
        )

    # --- basis / grid_level guard ------------------------------------------
    ledger_basis = ledger.get("basis")
    if ledger_basis != cfg.inputs.basis:
        raise ValueError(
            f"reuse-mode subset ledger basis mismatch: ledger was generated "
            f"with basis {ledger_basis!r} but cfg.inputs.basis is "
            f"{cfg.inputs.basis!r} — regenerate the ledger against the "
            "current config."
        )
    ledger_grid_level = ledger.get("grid_level")
    if ledger_grid_level != int(cfg.inputs.grid_level):
        raise ValueError(
            f"reuse-mode subset ledger grid_level mismatch: ledger was "
            f"generated with grid_level {ledger_grid_level!r} but "
            f"cfg.inputs.grid_level is {cfg.inputs.grid_level!r} — "
            "regenerate the ledger against the current config."
        )

    # --- bh76_mode guard ----------------------------------------------------
    ledger_bh76_mode = ledger.get("bh76_mode")
    if ledger_bh76_mode != cfg.bh76_mode:
        raise ValueError(
            f"reuse-mode subset ledger bh76_mode mismatch: ledger was "
            f"generated with bh76_mode {ledger_bh76_mode!r} but cfg.bh76_mode "
            f"is {cfg.bh76_mode!r} — regenerate the ledger against the "
            "current config."
        )

    # --- pool fingerprint guard --------------------------------------------
    actual_fp = pool_fingerprint(points)
    ledger_fp = ledger.get("pool_fingerprint")
    if ledger_fp != actual_fp:
        raise ValueError(
            f"reuse-mode subset ledger pool_fingerprint mismatch: ledger "
            f"fingerprint {ledger_fp!r} != current pool fingerprint "
            f"{actual_fp!r} — the ledger was generated against a different "
            "pool; regenerate or use a fresh ledger."
        )

    # --- every required (metric, subset_size) cell present -----------------
    entries = ledger.get("entries")
    if not isinstance(entries, dict):
        raise ValueError(
            f"reuse-mode subset ledger at {ledger_path!r} has no 'entries' "
            "mapping; regenerate it with prepare_inputs(cfg, regenerate=True)."
        )
    missing = [
        (metric, subset_size)
        for metric, subset_size in _metric_size_pairs(cfg)
        if _entry_key(metric, subset_size) not in entries
        and (metric, subset_size) not in entries
    ]
    if missing:
        raise ValueError(
            f"reuse-mode subset ledger at {ledger_path!r} is missing entries "
            f"for grid cells {missing}; the grid requires a subset for every "
            "(metric, subset_size) pair — regenerate the ledger."
        )

    return StagedInputs(points=points, subset_ledger=ledger)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def prepare_inputs(
    cfg: GridConfig,
    regenerate: bool,
    *,
    drop_species: tuple = (),
) -> StagedInputs:
    """Stage the harness's input artifacts.

    Parameters
    ----------
    cfg : GridConfig
        The harness config — supplies the swept axes (which determine the
        ``(metric, subset_size)`` pairs to select subsets for), the input
        paths, the basis / grid_level, and ``bh76_mode``.
    regenerate : bool
        ``True`` — run the heavy precompute (CCSD references, descriptors,
        reference histograms, ``select_subset`` per ``(metric, subset_size)``
        pair) and write the subset ledger atomically.
        ``False`` — reuse pre-staged artifacts: load the previously-written
        ledger and fail fast on any inconsistency with ``cfg`` / the pool.
    drop_species : tuple[str, ...], keyword-only, default ``()``
        The ``drop_failed_species`` survivor-pool path (regenerate mode only).
        When non-empty, every ``TrainingPoint`` whose species set intersects
        ``drop_species`` is removed from the pool *before* descriptors /
        histograms / ``select_subset`` run, and the written ledger carries
        ``pool_was_shrunk=True`` / ``dropped_species=sorted(drop_species)``.
        Empty (the default) — the no-shrink path; the ledger records
        ``pool_was_shrunk=False`` / ``dropped_species=[]``. Ignored in reuse
        mode (a reuse-mode ledger was already shrunk when written).

    Returns
    -------
    StagedInputs
        ``points`` (the ``TrainingPoint`` pool — the survivor pool when
        ``drop_species`` was given) + ``subset_ledger`` (the dict consumed by
        ``spec_builder.build_training_specs``).

    Raises
    ------
    ValueError
        In reuse mode: the ledger is missing / unparseable, or its
        ``basis`` / ``grid_level`` / ``bh76_mode`` / ``pool_fingerprint`` do
        not match the current config / pool, or a required
        ``(metric, subset_size)`` grid cell has no ledger entry.
    RuntimeError
        In regenerate mode: ``precompute_all`` failed for one or more species
        (it raises with the failed-species list).
    """
    if regenerate:
        return _prepare_inputs_regenerate(cfg, drop_species=tuple(drop_species))
    return _prepare_inputs_reuse(cfg)
