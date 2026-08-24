"""xcquinox.alec.cluster.inputs: input-artifact staging for the HPC harness.

This module prepares the harness's input artifacts before any training job is
submitted. Subset selection is a finished pre-process, the harness does
NOT run it. ``prepare_inputs`` is therefore consume-only for subsets:

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
The one heavy call, the CCSD external-reference precompute, is bound to a
module-level name so tests can monkeypatch it without doing real CCSD / SCF
work:

  - :data:`_precompute_all`: wraps ``external_refs.precompute_all``
  - :data:`_build_species_union`: wraps ``external_refs.build_species_union``

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


def _load_full_held_out_pools(basis="def2-svp", grid_level=1, refs_dir=None):
    """Seam wrapping ``full_benchmark_pools.load_full_held_out_pools`` (module-
    level so the WS3 val-slice staging tests can stub the heavy pool load)."""
    from xcquinox.alec.full_benchmark_pools import load_full_held_out_pools
    return load_full_held_out_pools(basis=basis, grid_level=grid_level,
                                    refs_dir=refs_dir)


def _get_domain_profile(name):
    """Seam wrapping ``domain.get_domain_profile`` (module-level so tests can
    monkeypatch the pool/CCSD strategy without the real heavy pool build)."""
    from xcquinox.alec.cluster.domain import get_domain_profile
    return get_domain_profile(name)


def _ledger_scoped_species(points, subset_ledger):
    """CCSD species to precompute = the union of species across every training
    point named in the loaded subset ledger (training-subset species only), plus
    a ``{(name,charge,spin): ASE Atoms}`` geometry map for ``precompute_all``.

    Generalizable: works for any name-keyed ledger over any pool. Returns
    ``(list[SpeciesEntry], atoms_by_key)``."""
    from xcquinox.alec.external_refs import SpeciesEntry
    from xcquinox.alec.training_points import species_union_from_points

    by_name = {p.name: p for p in points}
    chosen = []
    seen = set()
    for entry in subset_ledger.values():
        for pn in entry.get("point_names", ()):
            if pn in by_name and pn not in seen:
                seen.add(pn)
                chosen.append(by_name[pn])
    union_atoms = species_union_from_points(chosen)
    species = []
    atoms_by_key = {}
    for a in union_atoms:
        name = a.info["name"]
        charge = int(a.info.get("charge", 0))
        spin = int(a.info.get("spin", 0))
        species.append(SpeciesEntry(name=name, charge=charge, spin=spin,
                                    source="reaction_pool"))
        atoms_by_key[(name, charge, spin)] = a
    return species, atoms_by_key


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
        The raw, notebook-format ``subset_index_log.json`` dict, keys are
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
            "(already-finished) subset-selection pre-process, stage it "
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
    # entry, the harness cannot select a subset for a missing cell.
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

def _stage_validation_slice(cfg: GridConfig, run_dir: str):
    """Stage the held-out VALIDATION slice (WS3).

    Loads the held-out pools at the run's basis/grid, splits the reactions
    val/test via :func:`eval_holdout.split_held_out` (``cfg.hyperparams.val_frac``
    -- the SAME deterministic split the eval reports the test side of), and writes
    the val reaction dicts to ``<run_dir>/validation/val_reactions.json``.

    FIX 4 (WS3-INPUTS-01): there is NO SCF precompute here. The in-loop
    validation MAE is reaction-energy-only against PUBLISHED references, and the
    val MoleculeData is rebuilt at TRAIN time by
    :func:`xcquinox.alec.data.precompute_fixed_density_data` (via
    :func:`train._build_validation_data`), which runs its OWN PBE SCF and never
    consults any ``val_refs_dir`` cache. The previous ``_precompute_val_refs``
    call wrote ``<val_refs_dir>/_intermediates/<name>_g..._scf.npz`` that NOTHING
    read (and that no spec's ``external_data_path`` ever resolved to), so it was
    pure wasted compute -- removed. Only the ``val_reactions.json`` staging is
    needed (the train loop loads those reaction dicts).

    Returns the val reaction list. The caller (``prepare_inputs``) only invokes
    this when ``cfg.inputs.val_refs_dir`` AND ``run_dir`` are both set, so it is a
    no-op for runs that do not configure in-loop validation."""
    from xcquinox.alec.eval_holdout import split_held_out

    val_frac = float(getattr(cfg.hyperparams, "val_frac", 0.2))
    _mols_by_name, reactions = _load_full_held_out_pools(
        basis=cfg.inputs.basis, grid_level=cfg.inputs.grid_level)
    val_rxns, _test_rxns = split_held_out(reactions, val_frac=val_frac)

    val_dir = os.path.join(run_dir, "validation")
    os.makedirs(val_dir, exist_ok=True)
    with open(os.path.join(val_dir, "val_reactions.json"), "w") as f:
        json.dump(list(val_rxns), f, indent=2)
    print(f"[val-slice] staged {len(val_rxns)} val reactions "
          f"(val_frac={val_frac}); reactions -> "
          f"{val_dir}/val_reactions.json. Val density is rebuilt at train time "
          f"(no SCF precompute).", flush=True)
    return list(val_rxns)


def prepare_inputs(
    cfg: GridConfig,
    *,
    recompute_refs: bool = True,
    run_dir: str | None = None,
) -> StagedInputs:
    """Stage the harness's input artifacts (consume-only for subsets).

    Steps:

      1. Build the training-point pool via ``build_dfs_pool_points``.
      2. Load the EXISTING subset ledger at ``cfg.inputs.subset_ledger_path``
         (the ``subset_index_log.json`` format), fail fast on a missing /
         unparseable ledger or a missing required ``(metric, subset_size)``
         grid cell.
      3. Ensure the per-species CCSD external references are staged under
         ``cfg.inputs.external_refs_dir`` via ``precompute_all``: skip-if-
         cached, so already-staged references are a no-op and missing ones
         get computed.

    Parameters
    ----------
    cfg : GridConfig
        The harness config, supplies the swept axes (which determine the
        ``(metric, subset_size)`` cells the ledger must cover), the input
        paths, the basis / grid_level, and ``bh76_mode``.
    recompute_refs : bool, keyword-only, default ``True``
        ``True``: call ``precompute_all`` to ensure CCSD external references
        are staged (skip-if-cached, idempotent). ``False``: skip the
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
    # --- 1. training-point pool (domain-selected) --------------------------
    domain = _get_domain_profile(cfg.domain_profile)
    points = domain.pool_builder(cfg)

    # --- 2. load the EXISTING subset ledger --------------------------------
    subset_ledger = _load_subset_ledger(cfg)

    # --- 3. ensure CCSD external references --------------------------------
    # DFS domain: build_species_union assembles its own canonical species set
    # (training + probe + HBPT). External pools (ccsd_species_from_ledger):
    # restrict CCSD to the ledger's training-subset species and hand their
    # geometries to precompute_all directly. Either way precompute_all is
    # skip-if-cached / idempotent and raises RuntimeError with the failed-species
    # list on failure.
    if recompute_refs:
        if domain.ccsd_species_from_ledger:
            species, atoms_by_key = _ledger_scoped_species(points, subset_ledger)
            _precompute_all(
                species,
                cache_dir=cfg.inputs.external_refs_dir,
                basis=cfg.inputs.basis,
                grid_level=cfg.inputs.grid_level,
                density_fit=cfg.inputs.density_fit,
                auxbasis=cfg.inputs.auxbasis,
                orientation_lock_strength=cfg.inputs.orientation_lock_strength,
                atoms_by_key=atoms_by_key,
                validate_overrides=False,
                run_preflight=False,
            )
        else:
            species = _build_species_union()
            _precompute_all(
                species,
                cache_dir=cfg.inputs.external_refs_dir,
                basis=cfg.inputs.basis,
                grid_level=cfg.inputs.grid_level,
                density_fit=cfg.inputs.density_fit,
                auxbasis=cfg.inputs.auxbasis,
                # Lock the TRAINING CCSD references to the same degenerate density
                # component the functional and held-out refs use (spec_builder /
                # submit); otherwise OH/CH/NO radical training densities are
                # orientation-arbitrary and inject noise into the density loss.
                orientation_lock_strength=cfg.inputs.orientation_lock_strength,
            )

        # --- 4. ensure pretrain data matches the configured basis -----------
        # The per-atom Fx/Fc pretrain targets are basis-dependent, so a basis
        # change must regenerate them (skip-if-current via the data's manifest)
        # rather than silently training on stale def2-svp data. Density-fit the
        # per-atom SCF when the run does, so the whole pipeline shares one
        # Coulomb backend and a large basis stays within node RAM.
        #
        # Which files are required and which protocol keywords they are built
        # with are taken from the datagen stage's own derivations rather than
        # restated here: the preflight and the datagen stage then agree on the
        # set of files by construction, including the two-parent split a
        # mixed-rung sweep gets under ``pretrain.parent_density: auto``. The
        # historical call passed the run-level polarization flag alone, which
        # cannot express that split.
        from xcquinox.alec.cluster._datagen import (_protocol_keywords,
                                                    _required_data_specs)
        _extra = _protocol_keywords(cfg.pretrain)
        for _polarized, _reference_xc in _required_data_specs(cfg):
            _call = dict(_extra)
            # The reference density is named only when the call is not the
            # historical one, so a pre-protocol configuration reaches the
            # generator with exactly the keyword set it always did and its
            # existing data file stays current.
            if _call or _reference_xc != "pbe":
                _call["reference_xc"] = _reference_xc
            _ensure_pretrain_data(
                cfg.pretrain.data_dir,
                basis=cfg.inputs.basis,
                grid_level=cfg.inputs.grid_level,
                density_fit=cfg.inputs.density_fit,
                auxbasis=cfg.inputs.auxbasis,
                polarized=_polarized,
                # Part of the identity for the reason the datagen stage states
                # it: the harness default is the generator's own, but a run
                # pinned at another lock (0.0 in the pre-lock campaigns) must
                # not be served the locked file.
                orientation_lock_strength=cfg.inputs.orientation_lock_strength,
                **_call,
            )

    # --- 5. stage the held-out VALIDATION slice (WS3, option a) -------------
    # Only when a val_refs_dir is configured AND a run_dir is given (the
    # val_reactions.json lives under run_dir). No-op otherwise so existing runs
    # stay byte-identical. Density-only (PBE SCF), skip-if-cached, no CCSD/OEP.
    if getattr(cfg.inputs, "val_refs_dir", None) and run_dir:
        _stage_validation_slice(cfg, run_dir)

    return StagedInputs(points=points, subset_ledger=subset_ledger)
