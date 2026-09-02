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
from xcquinox.alec.training_points import build_dfs_pool_points, _atom_anchor_atoms
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


def _regularizer_anchor_atoms(domain) -> list:
    """The neutral ground-state atoms ``spec_builder`` injects into every spec
    whose chosen subset lacks them as single atoms -- one per symbol in
    ``domain.regularize_atom_syms`` (the Dick regularizer set, H and Li) --
    built by the same helper the natural AE path uses, so their names, charges
    and spins are the ones the specs carry. A run's references must cover
    them whether or not a chosen point names them: a single-cell run on a
    point without lithium still carries the Li anchor in its spec, and on a
    cold cache the anchor would otherwise reach training without a density
    reference."""
    return [_atom_anchor_atoms(sym)
            for sym in tuple(getattr(domain, "regularize_atom_syms", ()) or ())]


def _ledger_scoped_species(points, subset_ledger, *, extra_atoms=()):
    """CCSD species to precompute = the union of species across every training
    point named in the loaded subset ledger (training-subset species only), plus
    ``extra_atoms`` (the regularizer anchors, :func:`_regularizer_anchor_atoms`)
    deduplicated on the same key, plus a ``{(name,charge,spin): ASE Atoms}``
    geometry map for ``precompute_all``.

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
    union_atoms = list(species_union_from_points(chosen))
    have = {(a.info["name"], int(a.info.get("charge", 0)),
             int(a.info.get("spin", 0))) for a in union_atoms}
    for a in extra_atoms:
        key = (a.info["name"], int(a.info.get("charge", 0)),
               int(a.info.get("spin", 0)))
        if key not in have:
            have.add(key)
            union_atoms.append(a)
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


def _run_cells_ledger(cfg: GridConfig, subset_ledger: dict) -> dict:
    """The ledger entries of THIS run's ``(metric, subset_size)`` cells.

    :func:`_load_subset_ledger` has already required every one of them, so
    the lookup cannot miss; what this drops is the entries the file carries
    for cells outside the grid (the notebook ledger holds every cell ever
    selected, whichever subset sizes a run sweeps)."""
    return {_ledger_key(m, r): subset_ledger[_ledger_key(m, r)]
            for m, r in _metric_size_pairs(cfg)}


def _run_scoped_canonical_species(points, run_ledger, anchor_atoms=()):
    """DFS domain: the canonical species (:func:`build_species_union`, with
    their own geometry sources) that the run's specs carry -- the species its
    cells name plus the regularizer anchors ``spec_builder`` injects
    (``anchor_atoms``) -- in canonical order.

    Returns ``(species, canonical_count, outside)``: the entries to build, the
    size of the canonical set, and the sorted ``(name, charge, spin)`` keys the
    specs carry that the canonical set does not."""
    scoped, _atoms = _ledger_scoped_species(points, run_ledger,
                                            extra_atoms=anchor_atoms)
    wanted = {(s.name, s.charge, s.spin) for s in scoped}
    canonical = _build_species_union()
    species = [s for s in canonical if (s.name, s.charge, s.spin) in wanted]
    have = {(s.name, s.charge, s.spin) for s in species}
    return species, len(canonical), sorted(wanted - have)


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
    #: Names of the species whose CCSD references this staging ensured, in
    #: build order: the species the run's own cells name (empty when the
    #: precompute was skipped).
    reference_species: tuple = ()
    #: Size of the canonical species set the DFS build selects from (0 for
    #: an external pool, whose species come from the ledger alone).
    canonical_species_count: int = 0
    #: ``(name, charge, spin)`` keys the run's cells name that the canonical
    #: set does not carry: no reference exists for them and none is built, so
    #: they train without a density target, as in every run before.
    cell_species_without_reference: tuple = ()


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
         get computed. The species are the ones the run's own cells name
         (:func:`_run_cells_ledger`), and for the DFS domain they are taken
         from the canonical set (:func:`_run_scoped_canonical_species`).
      4. Check the pretrain-data ``.npz`` currency against the run identity
         (``ensure_pretrain_data``). Steps 3 and 4 share the
         ``recompute_refs`` gate: ``recompute_refs=False`` skips BOTH.
      5. Stage the validation slice record
         (``validation/val_reactions.json``) into the run directory.

    Parameters
    ----------
    cfg : GridConfig
        The harness config, supplies the swept axes (which determine the
        ``(metric, subset_size)`` cells the ledger must cover), the input
        paths, the basis / grid_level, and ``bh76_mode``.
    recompute_refs : bool, keyword-only, default ``True``
        ``True``: call ``precompute_all`` to ensure CCSD external references
        are staged (skip-if-cached, idempotent). ``False``: skip the
        precompute entirely -- INCLUDING the pretrain-data currency check
        that lives in the same block (step 4 below), so a stale pretrain
        ``.npz`` is not detected on this path; use only when the references
        AND the pretrain data are known to be already staged and current
        (the drop_failed_species preflight path).

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
    # Scoped to THIS run: the species named by the training points of the
    # run's own (metric, subset_size) cells, read off the ledger entries the
    # grid requires -- the ledger file carries every cell ever selected --
    # plus the regularizer anchors spec_builder injects into every spec whose
    # subset lacks them (H and Li), since the specs carry them either way. DFS
    # domain: those species taken from the canonical set build_species_union
    # assembles (training, probe and HBPT species), so a species outside that
    # set -- an atom an AE reaction names that no reference was ever built
    # for -- is reported and trains as before, without a density target,
    # rather than acquiring one here. External pools (ccsd_species_from_ledger):
    # the cells' species with their own geometries handed to precompute_all
    # directly. Either way precompute_all is skip-if-cached / idempotent and
    # raises RuntimeError with the failed-species list on failure.
    #
    # The canonical set holds 55 species. The 26-point DFS pool's cells name
    # 30 of them at subset size 26 and 7 at subset sizes 1 and 2 (the workflow
    # matrix); the rest are held-out probe species whose references the
    # evaluation reads from benchmark_refs_dir. Building all 55 for every run
    # put the longest OEP tails (O2S, H2O2, F2O) on the critical path of every
    # preflight, for species no cell trains on.
    run_ledger = _run_cells_ledger(cfg, subset_ledger)
    anchors = _regularizer_anchor_atoms(domain)
    reference_species: tuple = ()
    canonical_count = 0
    without_reference: tuple = ()
    if recompute_refs:
        if domain.ccsd_species_from_ledger:
            species, atoms_by_key = _ledger_scoped_species(
                points, run_ledger, extra_atoms=anchors)
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
            species, canonical_count, outside = _run_scoped_canonical_species(
                points, run_ledger, anchors)
            without_reference = tuple(outside)
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

        reference_species = tuple(s.name for s in species)

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
        # The refusal the generator applies to the requested identity (a
        # spatially degenerate free atom below grid level 3, or with the lock
        # off) precedes the currency check, so the preflight raises on a file
        # that is already on disk and current unless the run's waiver reaches
        # it here too. Stated only where it is granted, for the reason the
        # datagen stage states it only there.
        _waiver = ({"allow_irreproducible_degenerate": True}
                   if bool(getattr(cfg.inputs,
                                   "allow_irreproducible_degenerate", False))
                   else {})
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
                **_waiver,
                **_call,
            )

    # --- 5. stage the held-out VALIDATION slice (WS3, option a) -------------
    # Only when a val_refs_dir is configured AND a run_dir is given (the
    # val_reactions.json lives under run_dir). No-op otherwise so existing runs
    # stay byte-identical. Density-only (PBE SCF), skip-if-cached, no CCSD/OEP.
    if getattr(cfg.inputs, "val_refs_dir", None) and run_dir:
        _stage_validation_slice(cfg, run_dir)

    return StagedInputs(points=points, subset_ledger=subset_ledger,
                        reference_species=reference_species,
                        canonical_species_count=canonical_count,
                        cell_species_without_reference=without_reference)
