"""xcquinox.alec.cluster.inputs — input-artifact staging for the HPC harness.

This module prepares the harness's input artifacts before any training job is
submitted. Subset selection is a **finished pre-process** — the harness does
NOT run it. ``prepare_inputs`` is therefore *consume-only* for subsets:

  - it builds the training-point pool,
  - it loads the EXISTING subset ledger (``subset_index_log.json``), and
  - it ensures the per-species CCSD external references are staged (skip-if-
    cached, so already-computed references are a no-op).

The subset-ledger schema
------------------------
``cfg.inputs.subset_ledger_path`` points at the existing
``subset_index_log.json`` produced by the (already-finished) subset-selection
pre-process. Its schema is::

    {
        "<metric>/<r>": {
            "chosen_indices": [<int>, ...],
            "metric_value": <float>,
            "point_kinds": [<kind>, ...],
            "point_names": [<name>, ...],
            "tag": "bin<NN>"
        },
        ...
    }

The keys are ``"<metric>/<subset_size>"`` strings (e.g. ``"l2/3"``).
``prepare_inputs`` loads this raw notebook-format dict and returns it verbatim
as ``StagedInputs.subset_ledger``; a SEPARATE ``spec_builder`` refactor task
adapts :func:`xcquinox.alec.cluster.spec_builder.build_training_specs` to
consume this format.

Mockable seam
-------------
The one heavy call — the CCSD external-reference precompute — is bound to a
module-level name so tests can monkeypatch it without doing real CCSD / SCF
work:

  - :data:`_precompute_all`      — wraps ``external_refs.precompute_all``
  - :data:`_build_species_union` — wraps ``external_refs.build_species_union``

``prepare_inputs`` itself is orchestration-only.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

from xcquinox.alec import external_refs as _external_refs
from xcquinox.alec import pretrain_data_gen as _pretrain_data_gen
from xcquinox.alec.training_points import build_dfs_pool_points
from xcquinox.alec.cluster.grid_config import GridConfig, expand_grid


# ---------------------------------------------------------------------------
# Mockable heavy-call seams
# ---------------------------------------------------------------------------
# These module-level references are the indirection points tests monkeypatch
# (``monkeypatch.setattr(inputs, "_precompute_all", fake)``) so input staging
# can run without real CCSD / SCF compute.

_precompute_all = _external_refs.precompute_all
_build_species_union = _external_refs.build_species_union
_ensure_pretrain_data = _pretrain_data_gen.ensure_pretrain_data


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StagedInputs:
    """Result of :func:`prepare_inputs`.

    Attributes
    ----------
    points : list[TrainingPoint]
        The training-point pool from ``build_dfs_pool_points``.
    subset_ledger : dict
        The raw, notebook-format ``subset_index_log.json`` dict — keys are
        ``"<metric>/<subset_size>"`` strings. Handed unchanged to
        :func:`xcquinox.alec.cluster.spec_builder.build_training_specs` (the
        spec-builder refactor task adapts it to consume this format).
    """

    points: list
    subset_ledger: dict


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ledger_key(metric: str, subset_size: int) -> str:
    """The ``subset_index_log.json`` entry key for a ``(metric, r)`` pair."""
    return f"{metric}/{int(subset_size)}"


def _metric_size_pairs(cfg: GridConfig) -> list[tuple[str, int]]:
    """Distinct ``(metric, subset_size)`` pairs across the expanded grid,
    deterministically ordered (sorted)."""
    pairs = {(c.metric, int(c.subset_size)) for c in expand_grid(cfg)}
    return sorted(pairs)


def _load_subset_ledger(cfg: GridConfig) -> dict:
    """Load + validate the existing ``subset_index_log.json`` ledger.

    Fails fast with a clear :class:`ValueError` if the file is missing /
    unparseable / not a JSON object, or if any ``(metric, subset_size)`` cell
    required by ``expand_grid(cfg)`` is absent from it.
    """
    ledger_path = cfg.inputs.subset_ledger_path

    if not os.path.isfile(ledger_path):
        raise ValueError(
            f"subset ledger not found at {ledger_path!r}; the harness "
            "consumes the EXISTING subset_index_log.json produced by the "
            "(already-finished) subset-selection pre-process — stage it "
            "before running prepare_inputs."
        )
    try:
        with open(ledger_path) as f:
            ledger = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        raise ValueError(
            f"subset ledger at {ledger_path!r} is unreadable / unparseable "
            f"({exc})."
        ) from exc
    if not isinstance(ledger, dict):
        raise ValueError(
            f"subset ledger at {ledger_path!r} is not a JSON object (got "
            f"{type(ledger).__name__})."
        )

    # Every (metric, subset_size) cell the grid sweeps must have a ledger
    # entry — the harness cannot select a subset for a missing cell.
    missing = [
        (metric, subset_size)
        for metric, subset_size in _metric_size_pairs(cfg)
        if _ledger_key(metric, subset_size) not in ledger
    ]
    if missing:
        missing_keys = [_ledger_key(m, r) for m, r in missing]
        raise ValueError(
            f"subset ledger at {ledger_path!r} is missing entries for grid "
            f"cells {missing} (keys {missing_keys}); the grid requires a "
            "pre-selected subset for every (metric, subset_size) pair."
        )

    return ledger


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def prepare_inputs(
    cfg: GridConfig,
    *,
    recompute_refs: bool = True,
) -> StagedInputs:
    """Stage the harness's input artifacts (consume-only for subsets).

    Steps:

      1. Build the training-point pool via ``build_dfs_pool_points``.
      2. Load the EXISTING subset ledger at ``cfg.inputs.subset_ledger_path``
         (the ``subset_index_log.json`` format) — fail fast on a missing /
         unparseable ledger or a missing required ``(metric, subset_size)``
         grid cell.
      3. Ensure the per-species CCSD external references are staged under
         ``cfg.inputs.external_refs_dir`` via ``precompute_all`` — skip-if-
         cached, so already-staged references are a no-op and missing ones
         get computed.

    Parameters
    ----------
    cfg : GridConfig
        The harness config — supplies the swept axes (which determine the
        ``(metric, subset_size)`` cells the ledger must cover), the input
        paths, the basis / grid_level, and ``bh76_mode``.
    recompute_refs : bool, keyword-only, default ``True``
        ``True`` — call ``precompute_all`` to ensure CCSD external references
        are staged (skip-if-cached, idempotent). ``False`` — skip the
        precompute entirely; use this only when the references are known to
        be already staged.

    Returns
    -------
    StagedInputs
        ``points`` (the ``TrainingPoint`` pool) + ``subset_ledger`` (the raw
        ``subset_index_log.json`` dict, ready for ``spec_builder``).

    Raises
    ------
    ValueError
        The subset ledger is missing / unparseable, or a required
        ``(metric, subset_size)`` grid cell has no ledger entry.
    RuntimeError
        ``recompute_refs`` is ``True`` and ``precompute_all`` failed for one
        or more species (it raises with the failed-species list).
    """
    # --- 1. training-point pool --------------------------------------------
    points = build_dfs_pool_points(bh76_mode=cfg.bh76_mode)

    # --- 2. load the EXISTING subset ledger --------------------------------
    subset_ledger = _load_subset_ledger(cfg)

    # --- 3. ensure CCSD external references --------------------------------
    # build_species_union assembles its own canonical species set (training +
    # probe + HBPT); precompute_all is skip-if-cached / idempotent and raises
    # RuntimeError with a failed-species list on failure.
    if recompute_refs:
        species = _build_species_union()
        _precompute_all(
            species,
            cache_dir=cfg.inputs.external_refs_dir,
            basis=cfg.inputs.basis,
            grid_level=cfg.inputs.grid_level,
            density_fit=cfg.inputs.density_fit,
            auxbasis=cfg.inputs.auxbasis,
        )

        # --- 4. ensure pretrain data matches the configured basis -----------
        # The per-atom Fx/Fc pretrain targets are basis-dependent, so a basis
        # change must regenerate them (skip-if-current via the data's manifest)
        # rather than silently training on stale def2-svp data. Density-fit the
        # per-atom SCF when the run does, so the whole pipeline shares one
        # Coulomb backend and a large basis stays within node RAM.
        _ensure_pretrain_data(
            cfg.pretrain.data_dir,
            basis=cfg.inputs.basis,
            grid_level=cfg.inputs.grid_level,
            density_fit=cfg.inputs.density_fit,
            polarized=cfg.use_polarized_correlation,
        )

    return StagedInputs(points=points, subset_ledger=subset_ledger)
