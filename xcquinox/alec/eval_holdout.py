"""Shared held-out evaluation primitives.

Pulls the pure functions (reaction math, MAE, writers) and the
side-effectful NN forward + PBE precompute helpers OUT of
``notebooks/analysis/local_reeval.py`` into a place the cluster eval
(``xcquinox.alec.cluster._eval_one_spec``) can also import.

The local CLI (``notebooks/analysis/local_reeval.py``) re-exports these
names so existing imports and tests continue to work byte-identically.

Schema for the reaction dicts these helpers consume mirrors
``xcquinox.alec.eval_probes.PROBE_C_BH76_OUT_OF_TRAINING``::

    {
        "name":               str,                    # reaction id
        "source_pool":        "bh76" | "w411" | ...,
        "reactants":          list[str],
        "products":           list[str],
        "coeffs":             list[float],
        "reaction_energy_ref": float,                 # kcal/mol
        "species_spins":      dict[str, int],         # optional, 2S
        "species_charges":    dict[str, int],         # optional
        "source":             str,                    # optional citation
    }
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import pickle  # noqa: S403 -- trusted local .spec files, written by this codebase
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


#: CODATA-2018 hartree -> kcal/mol. Matches
#: ``notebooks/analysis/constraint_pretrain_gmtkn55_demo.KCAL_PER_HA``.
KCAL_PER_HA: float = 627.5094740631


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

# Cold-start trajectory diagnostic (the eval_holdout_coldstart channel).
# 25 cycles = the DFS Letter's SCF step count; conv_tol far below any
# per-cycle energy step so the latched |dE| freeze never masks the
# trajectory (all cycles execute and are recorded).
COLDSTART_MAX_CYCLES = 25
COLDSTART_CONV_TOL = 1e-12


def coldstart_solver_config(sc):
    """The cold-start override of a trained FULL-mode solver config.

    Functional-free minao seed, :data:`COLDSTART_MAX_CYCLES` cycles,
    :data:`COLDSTART_CONV_TOL`; mode stays FULL (a ONESHOT-shaped config
    would arm the dormant one-shot energy path and pin J) and every other
    trained knob (mixer, tail loss, orientation lock, DF) is preserved.
    Applied in ONE place by both the orchestrator (before the
    parallel/serial dispatch, covering the in-process serial tiers) and
    the shard workers (via ``--coldstart``, since they reload the spec
    pickle themselves) -- a single source of truth for the channel's
    protocol. Raises ``ValueError`` for a non-FULL solver (the
    SolverConfig validation: a non-pbe seed requires FULL mode).
    """
    import dataclasses

    return dataclasses.replace(
        sc, seed_source="minao", max_cycles=COLDSTART_MAX_CYCLES,
        conv_tol=COLDSTART_CONV_TOL, seed_cache_dir=None)


def load_training_spec(spec_path: Path):
    """Read the harness's serialized ``spec_<NNNN>.spec`` file.

    The ``.spec`` file is a plain pickle produced and consumed by the same
    codebase (the harness's ``_train_one_spec._load_spec`` writes it), so it
    round-trips byte-for-byte; it is trusted local data.
    """
    with open(spec_path, "rb") as f:
        return pickle.load(f)  # noqa: S301 -- trusted local .spec, written by this codebase


def held_out_pool_names(
    training_molecule_names: Sequence[str],
    pool_specs: Dict[str, Any],
) -> List[str]:
    """``pool_specs.keys() - training_molecule_names``, sorted lex.

    Pure. The lex sort makes the script's per-molecule output order
    deterministic across runs (so the per_molecule.json diffs cleanly in
    a long-running comparison loop).
    """
    training = set(training_molecule_names)
    return sorted(name for name in pool_specs if name not in training)


def reaction_overlap(
    reaction: Dict[str, Any], training_names: set,
) -> Tuple[bool, List[str]]:
    """``(any_overlap, [names that are in_sample])`` for one reaction.

    Used by the reaction filter to decide whether a reaction is strictly
    held-out (no overlap), or carries an in-sample side (overlap is the
    list of species names present in both the reaction and the training
    set).

    CASE-INSENSITIVE: BH76 and W4-11 name the SAME molecule with different case
    (``CH4`` vs ``ch4``, ``NH3`` vs ``nh3``, ``H2S`` vs ``h2s``). An exact-name
    match would let a W4-11 lower-case molecule be trained while its upper-case
    BH76 reaction stayed "held-out" -> train/test leakage. Every cross-pool
    case-twin in these pools is verified to be the same molecule (identical
    atom_composition), so case-folding the membership test is safe here.
    """
    names = set(reaction.get("reactants", [])) | set(
        reaction.get("products", []))
    training_cf = {str(t).casefold() for t in training_names}
    in_sample = sorted(n for n in names if str(n).casefold() in training_cf)
    return (len(in_sample) > 0, in_sample)


def filter_reactions(
    reactions: Sequence[Dict[str, Any]],
    training_names: Sequence[str],
    *,
    strict: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Partition ``reactions`` into ``(kept, dropped)``.

    Loose mode (default, ``strict=False``): every reaction is kept and
    each gains an ``"in_sample_overlap"`` key listing the overlapping
    species (empty list when held-out). Strict mode (``strict=True``): a
    reaction with ANY species in ``training_names`` is dropped. ``training_names``
    MUST be MOLECULE-level -- build it via :func:`training_molecule_names`, which
    excludes single ATOMS. Otherwise shared reference atoms (h, c, n, o, ...)
    count as overlap and drop nearly the ENTIRE atomization held-out set: every
    W4-11/BH76 atomization shares atoms with any non-empty training set, so
    strict atom-disjointness is unachievable (this was a real bug -- training on
    6 reactions dropped ~135/140 W4-11 reactions purely on shared atoms).
    """
    training = set(training_names)
    kept: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    for rxn in reactions:
        has_overlap, overlap = reaction_overlap(rxn, training)
        if has_overlap and strict:
            dropped.append({**rxn, "in_sample_overlap": overlap})
        else:
            kept.append({**rxn, "in_sample_overlap": overlap})
    return kept, dropped


def reaction_identity_key(rxn: Dict[str, Any]) -> str:
    """Order-invariant physical identity of a reaction: its sorted casefolded
    reactant and product name tuples, serialized. The pool lists four BH76
    barriers twice under permuted-reactant names (``bh76_h_hf_to_hfhts`` vs
    ``bh76_hf_h_to_hfhts``); a NAME-keyed split can put one copy in the
    validation slice and its twin in the test slice, so validation-best
    selection sees a reported test barrier. Falls back to the name when the
    species lists are absent."""
    reac = rxn.get("reactants")
    prod = rxn.get("products")
    if not reac or not prod:
        return f"name:{rxn.get('name')}"
    return repr((tuple(sorted(str(x).casefold() for x in reac)),
                 tuple(sorted(str(x).casefold() for x in prod))))


def split_held_out(
    reactions: Sequence[Dict[str, Any]],
    val_frac: float = 0.2,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Deterministically partition held-out ``reactions`` into ``(val, test)``.

    The val slice (~``val_frac``) drives in-training early-stop and
    validation-best model selection; the test slice is what the held-out eval
    REPORTS. Assignment is by a STABLE hash (hashlib, not the salted builtin
    ``hash``) of the reaction's PHYSICAL identity
    (:func:`reaction_identity_key`), so the split is identical across runs,
    processes and input order, a reaction can never land in both, and the
    pool's permuted-name duplicate entries land on the SAME side (a name-keyed
    hash put one twin per slice -- validation-best selection then saw four
    reported test barriers). 2026-06-20 (WS3); identity-keyed 2026-08-13.
    """
    if not (0.0 < val_frac < 1.0):
        raise ValueError(f"val_frac must be in (0, 1), got {val_frac}")
    val: List[Dict[str, Any]] = []
    test: List[Dict[str, Any]] = []
    for rxn in reactions:
        digest = hashlib.md5(
            reaction_identity_key(rxn).encode("utf-8")).hexdigest()
        frac = int(digest[:8], 16) / 0xFFFFFFFF   # deterministic [0, 1]
        (val if frac < val_frac else test).append(rxn)
    return val, test


def _spec_is_atom(mol_spec) -> bool:
    """True iff ``mol_spec`` is a NEUTRAL single atom -- a universal reference
    anchor present in every atomization/barrier reaction.

    Charge matters: monatomic ANIONS (``f-``, ``cl-``) are specific SN2 reactant
    SPECIES, NOT universal anchors. Treating them as atoms (excluding them from
    the overlap filter) leaks every reaction sharing the anion into the held-out
    set once any SN2 reaction is trained. So only NEUTRAL monatomics are anchors.
    """
    comp = getattr(mol_spec, "atom_composition", ()) or ()
    try:
        single = sum(int(n) for _, n in comp) == 1
    except (TypeError, ValueError):
        return False
    return single and int(getattr(mol_spec, "charge", 0) or 0) == 0


def training_molecule_names(training_spec) -> Tuple[str, ...]:
    """Names of the MULTI-ATOM training species, EXCLUDING single atoms.

    Held-out overlap must be MOLECULE-level: atoms (h, c, n, o, f, ...) are
    universal reference anchors present in every atomization reaction, so
    including them in ``training_names`` makes :func:`filter_reactions` drop
    nearly the entire W4-11/BH76 held-out set (every atomization shares atoms
    with any non-empty training set; strict atom-disjointness is unachievable).
    Pass this as the ``training_names`` for the held-out OVERLAP filter; use the
    FULL molecule list only for the per-molecule ``in_training_subset`` flag.
    """
    return tuple(
        getattr(m, "name", None)
        for m in getattr(training_spec, "molecules", ()) or ()
        if getattr(m, "name", None) is not None and not _spec_is_atom(m)
    )


def trained_reaction_exclusion(training_spec, pool_specs
                               ) -> Tuple[set, Dict[str, Tuple[str, ...]]]:
    """``(identity set, species key map)`` of the spec's VERBATIM supervised
    reactions -- the reaction-form training points recorded in
    ``loss_kwargs["bh76_reactions"]`` (the AE-as-reactions and the trained
    barrier reactions; IP13 pairs are not reactions in the held-out pools).

    Held-out exclusion is by verbatim supervised reaction, not by species
    membership: a test reaction merely CONTAINING a trained molecule is a
    genuine generalization target, while the trained reaction itself (e.g.
    the ``w411_*_atomization`` twin of a trained AE molecule, under the
    pool's naming) was a training target and must leave the reported set.
    Identities are canonical (composition/charge/spin with geometric isomer
    classes), so cross-vocabulary and permuted-name twins coincide. The key
    map covers pool AND trained names so callers can key pool reactions with
    the same vocabulary. ``(set(), {})`` when the spec records no reaction
    points."""
    from xcquinox.alec.species_matching import (canonical_species_keys,
                                                reaction_identity_keys)
    lk: Dict[str, Any] = {}
    # TrainingSpec exposes ``loss_kwargs_dict`` as a PROPERTY (attribute
    # access yields the dict directly); test stubs may model it as a method;
    # raw ``loss_kwargs`` may be a dict or the spec's tuple-of-pairs form.
    got = getattr(training_spec, "loss_kwargs_dict", None)
    if isinstance(got, dict):
        lk = got
    elif callable(got):
        lk = got() or {}
    else:
        raw = getattr(training_spec, "loss_kwargs", None)
        if isinstance(raw, dict):
            lk = raw
        elif raw:
            try:
                lk = dict(raw)
            except (TypeError, ValueError):
                lk = {}
    entries = []
    trained_names: set = set()
    for r in (lk.get("bh76_reactions") or []):
        get = (r.get if isinstance(r, dict)
               else lambda k, d=None, s=r: getattr(s, k, d))
        e = {"reactants": [str(x) for x in (get("reactants") or [])],
             "products": [str(x) for x in (get("products") or [])],
             "coeffs": list(get("coeffs") or [])}
        if not e["reactants"] or not e["products"]:
            continue
        entries.append(e)
        trained_names.update(e["reactants"] + e["products"])
    if not entries:
        return set(), {}
    key_map = canonical_species_keys(pool_specs, sorted(trained_names))
    identities: set = set()
    for e in entries:
        identities.update(reaction_identity_keys(e, key_map))
    return identities, key_map


def held_out_filter_names_with_aliases(training_spec,
                                       pool_specs) -> Tuple[str, ...]:
    """:func:`training_molecule_names` plus the pool species physically
    identical to a trained molecule under a different naming scheme.

    The training vocabulary carries ASE Hill formulas (``CHN``, ``H3N``,
    ``HO``) while the pools name the same molecules in GMTKN55 style
    (``hcn``, ``nh3``, ``oh``); the name-based (even case-folded) overlap
    test cannot connect them, so without this expansion the strict filter
    keeps trained molecules' reactions in the "held-out" set. Identity is
    matched on (element composition, charge, spin) via
    ``species_matching.trained_pool_aliases``."""
    from xcquinox.alec.species_matching import trained_pool_aliases
    names = training_molecule_names(training_spec)
    return tuple(sorted(set(names)
                        | trained_pool_aliases(names, pool_specs)))


def per_reaction_errors(
    energies_ha: Dict[str, float],
    reactions: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Per-reaction predicted ΔE + signed error vs ``reaction_energy_ref``.

    Returns a list of dicts, one per reaction, with keys: ``name``,
    ``de_kcalmol`` (predicted ΔE in kcal/mol), ``ref_kcalmol``,
    ``error_kcalmol`` (signed = predicted - reference), and
    ``abs_error_kcalmol``. Reactions whose species set has any non-finite
    energy get ``de_kcalmol = NaN`` so downstream consumers (per-reaction
    figures, MAE aggregators) can filter consistently. Pure.
    """
    out: List[Dict[str, Any]] = []
    for rxn in reactions:
        names = list(rxn.get("reactants", [])) + list(rxn.get("products", []))
        coeffs = list(rxn.get("coeffs", []))
        ref = float(rxn["reaction_energy_ref"])
        if len(names) != len(coeffs):
            out.append({"name": rxn.get("name"),
                        "de_kcalmol": float("nan"),
                        "ref_kcalmol": ref,
                        "error_kcalmol": float("nan"),
                        "abs_error_kcalmol": float("nan")})
            continue
        es = [energies_ha.get(n) for n in names]
        if any(e is None or not math.isfinite(e) for e in es):
            de_kc = float("nan")
            err = float("nan")
        else:
            de_ha = sum(c * e for c, e in zip(coeffs, es))
            de_kc = de_ha * KCAL_PER_HA
            err = de_kc - ref
        out.append({
            "name": rxn.get("name"),
            "de_kcalmol": de_kc,
            "ref_kcalmol": ref,
            "error_kcalmol": err,
            "abs_error_kcalmol": abs(err) if math.isfinite(err) else float("nan"),
        })
    return out


def reaction_mae_kcalmol(
    energies_ha: Dict[str, float],
    reactions: Sequence[Dict[str, Any]],
) -> Tuple[float, int, int]:
    """``(MAE in kcal/mol, n_reactions_used, n_dropped_nan)``.

    Thin wrapper around :func:`per_reaction_errors` that averages the
    finite absolute errors and reports how many reactions were silently
    dropped because a species energy was missing / non-finite.
    """
    rxns = list(reactions)
    err_rows = list(per_reaction_errors(energies_ha, rxns))
    abs_errs = [r["abs_error_kcalmol"] for r in err_rows
                if math.isfinite(r["abs_error_kcalmol"])]
    n_dropped_nan = len(err_rows) - len(abs_errs)
    if not abs_errs:
        return float("nan"), 0, n_dropped_nan
    return float(sum(abs_errs) / len(abs_errs)), len(abs_errs), n_dropped_nan


_DENSITY_RECORD_KEYS: Tuple[str, ...] = (
    "density_rmse", "density_l1", "density_rmse_pbe", "density_l1_pbe",
    # DFS Letter Eq. 20 per-electron L1 (eps) + its model-free PBE twin,
    # with the quadrature bookkeeping that makes any renormalization
    # reconstructible offline (evaluation.density_eps_terms)
    "density_eps_l1", "density_eps_l1_pbe", "n_electrons", "grid_weight_sum",
    "ref_density_method",
)


def density_errors_for_record(model, md: Dict[str, Any], *,
                              solver_config=None) -> Dict[str, Any]:
    """NN-vs-CCSD and PBE-vs-CCSD density errors for one held-out species.

    Returns ``{density_rmse, density_l1, density_rmse_pbe, density_l1_pbe,
    density_eps_l1, density_eps_l1_pbe, n_electrons, grid_weight_sum,
    ref_density_method}`` (= ``_DENSITY_RECORD_KEYS``; the eps pair is the
    DFS Letter Eq. 20 per-electron L1 with its quadrature bookkeeping) --
    all ``None`` for atoms or when no CCSD reference density was loaded
    (``rho_ref_grid is None``; e.g. the species' ``external_data_path`` was
    unresolved), so runs without benchmark refs are byte-identical to the
    historical all-None schema.

    The NN channel reuses :class:`~xcquinox.alec.evaluation.DensityRMSEMetric`
    (solver-aware; with a FULL/FIXED_J ``solver_config`` this re-runs the SCF
    to get the self-consistent NN density -- roughly doubling per-species eval
    cost WHEN refs are present, same trade the in-sample eval makes). The PBE
    channel needs NO model: ``md['rho_grid']`` IS the PBE density on the same
    pruned grid the CCSD reference was evaluated on (data.py precompute), so
    it is the same weighted RMSE/L1 with rho_pbe in place of rho_nn
    (formula: evaluation.py DensityRMSEMetric). fp64 note: the cluster eval
    path already forces JAX_ENABLE_X64 before importing jax."""
    none_result = {k: None for k in _DENSITY_RECORD_KEYS}
    comp = md.get("atom_composition") or ()
    if sum(n for _, n in comp) == 1:
        return none_result                      # atoms: density matching skipped
    if md.get("rho_ref_grid") is None:
        return none_result
    import xcquinox.alec.evaluation as evaluation
    nn = evaluation.DensityRMSEMetric().compute(model, md,
                                                solver_config=solver_config)
    rmse_pbe, l1_pbe = evaluation.pbe_density_errors(md)
    eps_pbe, n_e, wsum = evaluation.pbe_density_eps(md)
    return {
        "density_rmse": nn.get("density_rmse"),
        "density_l1": nn.get("density_l1"),
        "density_rmse_pbe": rmse_pbe,
        "density_l1_pbe": l1_pbe,
        "density_eps_l1": nn.get("density_eps_l1"),
        "density_eps_l1_pbe": eps_pbe,
        "n_electrons": n_e,
        "grid_weight_sum": wsum,
        "ref_density_method": nn.get("ref_density_method")
                              or md.get("ref_density_method"),
    }


def make_per_molecule_record(
    name: str,
    mol_data: Dict[str, Any],
    e_nn_ha: float,
    *,
    in_training_subset: bool,
    scf: Optional[Dict[str, Any]] = None,
    density: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Schema-compatible with the cluster's ``eval/per_molecule.json`` so
    the existing :func:`collect_per_molecule_rows` can read it without
    modification. Adds a ``from_training_subset`` flag for downstream
    splitting.

    ``AE_error_kcalmol`` is left None (it only makes sense within a reaction
    context, which the per-reaction CSV captures). The density fields
    (``density_rmse``/``density_l1`` NN-vs-CCSD, ``density_rmse_pbe``/
    ``density_l1_pbe`` PBE-vs-CCSD, ``density_eps_l1``/``density_eps_l1_pbe``
    the DFS Eq. 20 per-electron L1 pair, ``n_electrons``/``grid_weight_sum``
    the quadrature bookkeeping, ``ref_density_method``) come from the
    optional ``density`` dict (:func:`density_errors_for_record`) and stay
    None when it is omitted -- runs without benchmark CCSD reference
    densities keep the historical all-None schema.

    ``scf``: optional per-molecule SCF convergence info captured during the
    NN self-consistent eval (see :func:`evaluate_holdout`'s ``scf_info_out``).
    Expected keys: ``cycles_run`` (int), ``converged`` (bool),
    ``total_energy`` (float), ``energy_trace`` (list[float], one entry per SCF
    cycle). When present, the record gains, FOR EACH cycle ``i`` that ran:
    ``scf_energy_step_<i>`` (the total energy after cycle ``i``, Hartree) and
    ``scf_energy_residual_<i>`` (``|E_i - E_final|``), the per-molecule,
    per-SCF-step convergence trace. ``cycles_run`` / ``scf_converged`` /
    ``scf_total_energy`` reflect the actual SCF (vs the one-shot sentinels).

    Failure rows: when ``scf`` carries an ``eval_error`` (the species' forward
    pass raised, see :func:`evaluate_holdout`), or when ``scf`` is absent AND
    ``e_nn_ha`` is non-finite, no SCF ran and the (0, True) sentinels would
    assert a convergence that never happened -- ``cycles_run`` and
    ``scf_converged`` are then ``None``, and the ``eval_error`` text is carried
    on the record when it is known. A no-SCF row with a FINITE energy is the
    ONESHOT/FIXED_J case and keeps the historical sentinels, matching
    ``evaluation.SCFConvergenceMetric``.
    """
    e_pbe = mol_data.get("E_pbe")
    e_pbe_f = float(e_pbe) if e_pbe is not None else None
    record: Dict[str, Any] = {
        "molecule": name,
        "E_total_nn": e_nn_ha if math.isfinite(e_nn_ha) else None,
        "E_pbe": e_pbe_f,
        "AE_nn": ((e_nn_ha - e_pbe_f)
                  if (e_pbe_f is not None and math.isfinite(e_nn_ha))
                  else None),
        "AE_error_kcalmol": None,
        "density_rmse": None,
        "density_l1": None,
        "density_rmse_pbe": None,
        "density_l1_pbe": None,
        "density_eps_l1": None,
        "density_eps_l1_pbe": None,
        "n_electrons": None,
        "grid_weight_sum": None,
        "ref_density_method": None,
        "cycles_run": 0,
        "scf_converged": True,
        "from_training_subset": bool(in_training_subset),
    }
    if density is not None:
        for k in _DENSITY_RECORD_KEYS:
            record[k] = density.get(k)
    eval_error = scf.get("eval_error") if scf is not None else None
    if eval_error:
        # The species' evaluation raised: no SCF ran, so the sentinels would
        # assert a converged zero-cycle SCF that never happened. Null out the
        # convergence pair and carry the exception text instead.
        record["cycles_run"] = None
        record["scf_converged"] = None
        record["eval_error"] = str(eval_error)
    elif scf is None and not math.isfinite(e_nn_ha):
        # No SCF info and no energy either (a failure recorded without an
        # ``scf_info_out`` map, or a species missing from the energy map):
        # same false statement, same null pair.
        record["cycles_run"] = None
        record["scf_converged"] = None
    elif scf is not None:
        record["cycles_run"] = int(scf.get("cycles_run", 0))
        record["scf_converged"] = bool(scf.get("converged", False))
        e_final = scf.get("total_energy")
        if e_final is not None and math.isfinite(float(e_final)):
            record["scf_total_energy"] = float(e_final)
        trace = scf.get("energy_trace") or []
        e_final_f = (float(e_final) if e_final is not None
                     and math.isfinite(float(e_final)) else None)
        for i, e_step in enumerate(trace):
            try:
                e_i = float(e_step)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(e_i):
                continue
            record[f"scf_energy_step_{i}"] = e_i
            if e_final_f is not None:
                record[f"scf_energy_residual_{i}"] = abs(e_i - e_final_f)
    return record


def make_per_reaction_records(
    reactions: Sequence[Dict[str, Any]],
    nn_errors: Sequence[Dict[str, Any]],
    pbe_errors: Sequence[Dict[str, Any]],
    training_names: Sequence[str],
) -> List[Dict[str, Any]]:
    """Per-reaction record dicts paired across NN and PBE.

    Output schema (per record): ``name, pool, reactants, products, coeffs,
    reaction_energy_ref_kcalmol, de_nn_kcalmol, de_pbe_kcalmol,
    error_nn_kcalmol, error_pbe_kcalmol, abs_error_nn_kcalmol,
    abs_error_pbe_kcalmol, in_sample_overlap``. Stable order = the input
    reactions list.
    """
    training = set(training_names)
    records: List[Dict[str, Any]] = []
    for rxn, nn, pbe in zip(reactions, nn_errors, pbe_errors):
        # Case-insensitive, via the shared overlap helper, so the per-reaction
        # in_sample_overlap flag matches the strict-drop filter exactly.
        _, overlap = reaction_overlap(rxn, training)
        records.append({
            "name": rxn.get("name"),
            "pool": rxn.get("source_pool"),
            "reactants": list(rxn.get("reactants", [])),
            "products": list(rxn.get("products", [])),
            "coeffs": list(rxn.get("coeffs", [])),
            "reaction_energy_ref_kcalmol": float(rxn["reaction_energy_ref"]),
            "de_nn_kcalmol": nn["de_kcalmol"],
            "de_pbe_kcalmol": pbe["de_kcalmol"],
            "error_nn_kcalmol": nn["error_kcalmol"],
            "error_pbe_kcalmol": pbe["error_kcalmol"],
            "abs_error_nn_kcalmol": nn["abs_error_kcalmol"],
            "abs_error_pbe_kcalmol": pbe["abs_error_kcalmol"],
            "in_sample_overlap": overlap,
        })
    return records


# ---------------------------------------------------------------------------
# Side-effectful: PBE precompute + NN forward
# ---------------------------------------------------------------------------

def precompute_holdout(
    mol_specs: Dict[str, Any],
    descriptors: Sequence[Any] = (),
    *,
    required_keys: Sequence[str] = (),
    auxbasis: str | None = None,
    orientation_lock_strength: float = 0.0,
    seed_source: str = "pbe",
    seed_cache_dir: str | None = None,
    seed_density_fit: bool = False,
) -> Dict[str, Any]:
    """Run the PBE precompute over a held-out pool of species.

    ``descriptors`` must be the SAME descriptor list the trained model uses
    (typically ``training_spec.arch.materialize_descriptors()``). With
    empty descriptors, models that consume descriptor features (e.g.
    ``deep_combined_attn`` with ``dm_statistics`` + ``cusp``) will raise
    ``TypeError`` from ``jnp.concatenate`` at evaluation time because the
    descriptor columns will be ``None`` in the resulting ``mol_data``.

    ``required_keys`` lets callers ask for additional precomputed fields
    beyond the baseline (e.g. ``("eri",)`` when an SCF solver will be run
    against the precomputed data, without ``eri``, ``run_scf`` raises
    ``ValueError: Cannot determine the shape of None`` because the Coulomb
    rebuild needs the electron-repulsion integrals).

    Per-molecule progress logged (~1-10 s/molecule for PBE SCF + grid build).
    """
    import xcquinox.alec as alec
    out: Dict[str, Any] = {}
    n = len(mol_specs)
    t0 = time.time()
    for i, (name, spec) in enumerate(mol_specs.items(), start=1):
        t1 = time.time()
        try:
            out[name] = alec.precompute_fixed_density_data(
                spec, descriptors=tuple(descriptors),
                required_keys=tuple(required_keys), auxbasis=auxbasis,
                orientation_lock_strength=orientation_lock_strength,
                seed_source=seed_source, seed_cache_dir=seed_cache_dir,
                seed_density_fit=seed_density_fit)
        except Exception as exc:  # noqa: BLE001
            print(f"  [precompute {i}/{n}] {name}: FAILED ({exc})",
                  flush=True)
            continue
        print(f"  [precompute {i}/{n}] {name}  "
              f"({time.time() - t1:.1f}s)", flush=True)
    print(f"  precompute done in {time.time() - t0:.1f}s "
          f"({len(out)}/{n} succeeded)", flush=True)
    return out


def evaluate_holdout(model, mol_data: Dict[str, Any],
                     *, solver_config=None,
                     verbose_failures: bool = True,
                     scf_info_out: Optional[Dict[str, Dict[str, Any]]] = None,
                     ) -> Dict[str, float]:
    """Per-species total energy.

    Energy source follows the same solver-MODE rule as training
    (``losses._compute_energies`` / ``oneshot.total_energy_for_solver``):

    * ``FULL`` -> the self-consistent ``alec.run_scf(...).total_energy`` (and the
      per-SCF-step trace is captured into ``scf_info_out``), matching what
      FULL-mode training optimizes.
    * ``ONESHOT`` / ``FIXED_J`` / ``None`` -> the one-shot
      ``alec.fixed_density_total_energy(model, mol_data[name])`` on ρ_PBE.
      FIXED_J stays one-shot deliberately (its run_scf energy is an incoherent
      J-pinned hybrid), so a FIXED_J-trained spec is not silently evaluated on
      that hybrid here.

    ``scf_info_out``: optional dict; when provided AND an SCF runs, it is
    populated ``{name: {cycles_run, converged, total_energy, energy_trace}}``
    per species, the per-molecule, per-SCF-step convergence trace that
    :func:`make_per_molecule_record` turns into ``scf_energy_step_<i>`` /
    ``scf_energy_residual_<i>`` columns. Captures even when the final energy
    is non-finite (so a diverged SCF's trace is still recorded). A species
    whose evaluation RAISES gets ``{name: {"eval_error": "<Type>: <msg>"}}``
    instead, so the failure survives into the per-molecule record rather than
    reading as an SCF that was never attempted.

    NaN on exception. When ``verbose_failures`` is True (default), EVERY
    exception in a batch is printed to stderr with its full message so the
    operator sees real errors instead of a silent column of NaNs.
    """
    import xcquinox.alec as alec
    from xcquinox.alec.solver import run_scf, SolverMode
    from xcquinox.alec.oneshot import tail_weighted_mean_energy
    import numpy as _np
    out: Dict[str, float] = {}
    n_failed = 0
    # FULL -> self-consistent run_scf energy (+ trace); else one-shot. Matches
    # oneshot.total_energy_for_solver so train == in-sample-eval == held-out-eval.
    use_scf = (solver_config is not None
               and getattr(solver_config, "mode", None) == SolverMode.FULL)
    for name, md in mol_data.items():
        try:
            if use_scf:
                # forward_only: held-out energy eval is a forward pass (no grad),
                # so run the SCF cycles in a python loop -> skip the giant fused
                # per-molecule XLA compile (see solver_manual._iterate_scf). The
                # held-out DENSITY path already de-fuses via DensityRMSEMetric.
                result = run_scf(solver_config, model, md, forward_only=True)
                e_final = float(result.total_energy)
                trace = getattr(result, "energy_trace", None)
                # Convergence-aware reported energy: the DFS tail-weighted mean
                # (denoised) when the tail loss is enabled, else the final
                # cycle. Matches oneshot.total_energy_for_solver so train ==
                # in-sample-eval == held-out-eval all report the same quantity.
                if (getattr(solver_config, "scf_loss_use_tail", False)
                        and trace is not None):
                    e = float(tail_weighted_mean_energy(
                        trace, solver_config.scf_loss_tail,
                        solver_config.scf_loss_weight_power))
                else:
                    e = e_final
                if scf_info_out is not None:
                    scf_info_out[name] = {
                        "cycles_run": int(getattr(result, "cycles_run", 0)),
                        "converged": bool(getattr(result, "converged", False)),
                        # raw final-cycle energy, kept for forensics ...
                        "total_energy": e_final,
                        # ... alongside the energy the metric actually used.
                        "reported_energy": e,
                        "energy_trace": ([float(x) for x in _np.asarray(trace)]
                                         if trace is not None else []),
                    }
            else:
                e = float(alec.fixed_density_total_energy(model, md))
        except Exception as exc:  # noqa: BLE001
            # EVERY failure is named, on stderr: one shard can silently lose
            # dozens of species to transient allocation/compile faults, and a
            # single first-failure line on stdout hides the rest (stdout also
            # carries the worker's JSON status line).
            if verbose_failures:
                print(f"  eval[{name}] FAILED: {type(exc).__name__}: {exc}",
                      file=sys.stderr, flush=True)
            if scf_info_out is not None:
                # An ABSENT scf_info entry is indistinguishable downstream from
                # a one-shot eval, which is how a raised species came to be
                # written as a converged zero-cycle SCF; record the failure.
                scf_info_out[name] = {
                    "eval_error": f"{type(exc).__name__}: {exc}"}
            n_failed += 1
            e = float("nan")
        out[name] = e if math.isfinite(e) else float("nan")
    if n_failed:
        print(f"  eval: {n_failed}/{len(mol_data)} species failed "
              "(NaN energy; see the eval[...] FAILED lines on stderr)",
              flush=True)
    return out


# ---------------------------------------------------------------------------
# Channel integrity: a sliced channel is not a full-pool channel
# ---------------------------------------------------------------------------

#: Slice marker, written by ``cluster/_eval_one_spec._apply_species_slice``
#: into the channel directory BEFORE any energy is computed, so an
#: interrupted or failed sliced evaluation stays unmistakable.
SLICED_MARKER_NAME = "sliced_eval.json"
#: Channel provenance stamp, written by
#: ``cluster/_eval_one_spec._run_held_out_eval`` AFTER the evaluation; its
#: ``species_slice`` entry is None for the full pool and a list otherwise.
EVAL_METADATA_NAME = "eval_metadata.json"


class SlicedChannelError(RuntimeError):
    """A held-out channel evaluated on a species slice, read as a full pool.

    Subclasses ``RuntimeError`` so a caller that only wants "this channel is
    unusable" can keep catching the broader type.
    """


def _sliced_channel_message(mark: Path, mark_note: str, spec_dir: Path,
                            eval_subdir: str, slice_names) -> str:
    """The refusal text: which mark fired, where, on what slice, and why a
    sliced channel cannot stand in for a full-pool one."""
    from xcquinox.alec.full_benchmark_pools import HELDOUT_SPECIES_SLICE_ENV
    # <run>/checkpoints/spec_NNNN is the layout every pull uses; a spec dir
    # placed anywhere else still names its own parent.
    run_dir = (spec_dir.parent.parent if spec_dir.parent.name == "checkpoints"
               else spec_dir.parent)
    # Only a list/tuple is a species list. Any other value is reported as
    # unknown WITH the value itself: iterating it would render a string per
    # character and a mapping as its keys -- either reads as a species list
    # that was never there -- and a number or bool would raise TypeError out
    # of the guard instead of the refusal the caller is written against.
    if isinstance(slice_names, (list, tuple)) and slice_names:
        named = ", ".join(repr(str(n)) for n in slice_names)
    elif slice_names is None:
        named = "unknown"
    else:
        named = f"unknown ({slice_names!r})"
    return (
        f"{mark} marks a SLICED held-out channel ({mark_note}): "
        f"run {run_dir}, spec {spec_dir.name}, channel {eval_subdir}, "
        f"species_slice={named}. A slice covers only the species named in "
        f"{HELDOUT_SPECIES_SLICE_ENV} for a workflow test, so this channel "
        "averages a different reaction set than the full BH76 + W4-11 "
        "held-out pool and its MAE is not a pool MAE; re-evaluate the "
        "channel without the variable, or drop the run."
    )


def _read_json_or_none(path: Path):
    """Parsed JSON, or None when the file is unreadable or malformed."""
    try:
        with path.open() as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
        return None


def assert_channel_not_sliced(spec_dir: Path, eval_subdir: str) -> None:
    """Refuse a held-out channel evaluated on a species slice.

    A slice covers the handful of species named for a workflow test
    (``XCQUINOX_HELDOUT_SPECIES_SLICE``, SPEC_pretrain_fidelity_program.md
    3.4), not the 216-reaction BH76 + W4-11 pool the architectures are
    compared on; its MAE is a different quantity, so a reader that averages
    one into a full-pool number redefines the metric with no visible signal.
    Every reader of a held-out channel calls this before its first read of
    that channel, and every writer that rewrites one calls it before the
    rewrite.

    ``cluster/_eval_one_spec`` marks a sliced channel TWICE --
    :data:`SLICED_MARKER_NAME` written before the energies and a
    ``species_slice`` entry in :data:`EVAL_METADATA_NAME` written after them
    -- and either mark is fatal here, so an interrupted sliced evaluation and
    a stale marker left by an earlier sliced pass are refused as surely as a
    complete one. The refusal keys on the MARKS only, never on reaction
    counts: the two files' counts differ by construction (the marker records
    the slice's own counts, the stamp records what survived the
    validation-complement filter), so a count comparison would be a false
    signal.

    Passing states: no channel directory, no mark, ``species_slice: null``
    (what a full-pool evaluation writes), an unparseable stamp, and a stamp
    that is valid JSON but not an object -- none is a slice signal, and the
    malformed cases are left to the readers below, which already tolerate
    unreadable JSON. The marker's own presence is the signal regardless of
    its contents; when those cannot be read the slice is reported as
    unknown.

    The two marks therefore part company on an EMPTY species list: a stamp
    carrying ``species_slice: []`` loads, because an empty slice restricts
    nothing, while a marker carrying it still refuses, with the slice
    reported unknown, because the marker is a slice signal by its presence
    alone. Neither state is reachable from the writer, which records None
    for the full pool and a non-empty list otherwise. A ``species_slice``
    that is neither null nor a list -- a number, a bool, a string, a mapping
    -- refuses on the MARKER whatever its value, and on the stamp only when
    it is truthy (the stamp branch tests the value, so ``0``, ``false`` and
    ``""`` load there, as the empty list does), and is reported as unknown
    beside its own value, never iterated.

    Raises:
        SlicedChannelError: the channel carries either mark.
    """
    spec_dir = Path(spec_dir)
    channel = spec_dir / eval_subdir
    marker = channel / SLICED_MARKER_NAME
    if marker.is_file():
        payload = _read_json_or_none(marker)
        names = (payload.get("species_slice")
                 if isinstance(payload, dict) else None)
        raise SlicedChannelError(_sliced_channel_message(
            marker, "written before the energies", spec_dir, eval_subdir,
            names))
    stamp = channel / EVAL_METADATA_NAME
    if not stamp.is_file():
        return
    payload = _read_json_or_none(stamp)
    if not isinstance(payload, dict):
        return
    names = payload.get("species_slice")
    if names:
        raise SlicedChannelError(_sliced_channel_message(
            stamp, "written after the evaluation", spec_dir, eval_subdir,
            names))


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

# Default filenames (kept as constants so callers can construct paths in
# their own subdir conventions without re-deriving the suffix).
DEFAULT_CSV_NAME = "test_set.csv"
DEFAULT_PER_MOLECULE_NAME = "per_molecule.json"
DEFAULT_PER_REACTION_NAME = "per_reaction.json"


def write_test_set_csv(
    out_path: Path,
    per_pool_mae: Dict[str, Tuple[float, float, int, int, int]],
    combined_mae: Tuple[float, float, int, int, int],
    strict: bool,
) -> Path:
    """Write the per-spec MAE summary CSV (one row per pool + one combined).

    ``per_pool_mae`` maps pool token (``"bh76"``, ``"w411"``) to
    ``(mae_nn_kcalmol, mae_pbe_kcalmol, n_used, n_dropped_overlap,
    n_dropped_nan)``. The PBE MAE comes from re-evaluating the same
    reactions against ``mol_data["E_pbe"]`` instead of the NN, costs
    nothing extra since the PBE energies are by-products of the precompute
    step, and gives the operator a direct apples-to-apples NN-vs-PBE
    comparison on the SAME pool the NN was scored on.

    The ``n_dropped_nan`` column reports reactions silently dropped because
    their species energies were missing or non-finite. It is distinct from
    ``n_dropped_overlap`` (training-set
    overlap drops in strict mode). A row with ``n_dropped_nan > 0``
    indicates the MAE was computed on a SMALLER reaction set than expected.

    Writes to ``out_path`` (caller controls the full path so cluster uses
    ``<ckpt>/eval_holdout/test_set.csv`` and the local CLI uses
    ``<ckpt>/local_test_set.csv``).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["set", "mae_nn_kcalmol", "mae_pbe_kcalmol",
                  "delta_nn_minus_pbe", "n_reactions", "n_dropped_overlap",
                  "n_dropped_nan", "note"]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for pool_name, vals in per_pool_mae.items():
            mae_nn, mae_pbe, n_used, n_dropped, n_nan = vals
            note_parts = []
            if strict and n_dropped:
                note_parts.append(
                    f"strict (held-out only); {n_dropped} verbatim-"
                    "supervised reactions dropped")
            elif strict:
                note_parts.append("strict (held-out only)")
            else:
                note_parts.append("loose (verbatim-supervised reactions "
                                  "kept; species overlap flagged in "
                                  "per_molecule.json)")
            if n_nan:
                note_parts.append(
                    f"{n_nan} reactions silently dropped (missing/NaN "
                    "species energies)")
            delta = (mae_nn - mae_pbe
                     if math.isfinite(mae_nn) and math.isfinite(mae_pbe)
                     else float("nan"))
            w.writerow({
                "set": f"test_set_{pool_name}",
                "mae_nn_kcalmol": ("" if not math.isfinite(mae_nn)
                                   else f"{mae_nn:.6f}"),
                "mae_pbe_kcalmol": ("" if not math.isfinite(mae_pbe)
                                    else f"{mae_pbe:.6f}"),
                "delta_nn_minus_pbe": ("" if not math.isfinite(delta)
                                       else f"{delta:+.6f}"),
                "n_reactions": n_used,
                "n_dropped_overlap": n_dropped,
                "n_dropped_nan": n_nan,
                "note": "; ".join(note_parts),
            })
        mae_c_nn, mae_c_pbe, n_used_c, n_dropped_c, n_nan_c = combined_mae
        delta_c = (mae_c_nn - mae_c_pbe
                   if math.isfinite(mae_c_nn) and math.isfinite(mae_c_pbe)
                   else float("nan"))
        combined_note_parts = ["combined across pools"
                                + (" (strict)" if strict else " (loose)")]
        if n_nan_c:
            combined_note_parts.append(
                f"{n_nan_c} reactions silently dropped (missing/NaN species)")
        w.writerow({
            "set": "test_set_held_out_combined",
            "mae_nn_kcalmol": ("" if not math.isfinite(mae_c_nn)
                               else f"{mae_c_nn:.6f}"),
            "mae_pbe_kcalmol": ("" if not math.isfinite(mae_c_pbe)
                                else f"{mae_c_pbe:.6f}"),
            "delta_nn_minus_pbe": ("" if not math.isfinite(delta_c)
                                   else f"{delta_c:+.6f}"),
            "n_reactions": n_used_c,
            "n_dropped_overlap": n_dropped_c,
            "n_dropped_nan": n_nan_c,
            "note": "; ".join(combined_note_parts),
        })
    return out_path


def write_per_molecule_json(out_path: Path,
                             records: List[Dict[str, Any]]) -> Path:
    """Write per-species records to JSON at ``out_path``."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(records, f, indent=2)
    return out_path


def write_per_reaction_json(out_path: Path,
                             records: List[Dict[str, Any]]) -> Path:
    """Write per-reaction records to JSON at ``out_path``."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(records, f, indent=2)
    return out_path


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def arch_polarized_flag(arch) -> bool:
    """``True`` iff this architecture trained with spin-polarized
    correlation (the ``use_polarized_correlation`` field on
    ``ArchitectureConfig``). Pure."""
    return bool(getattr(arch, "use_polarized_correlation", False))


def load_trained_model(training_spec, model_path: Path):
    """Build the AlecGGAModel skeleton from ``training_spec.arch`` and
    deserialize ``model.eqx`` into it. Returns the trained
    ``AlecGGAModel``.

    Logs the polarization mode (RKS-trained vs UKS-trained) so the
    operator sees at-a-glance which path the eval will take for open-shell
    species. The actual routing is decided per-molecule by
    ``xcquinox.alec.oneshot.fixed_density_total_energy`` based on the
    cnet's ``use_spin_polarization`` flag, which is exactly what
    ``AlecGGAModel.from_arch`` sets from
    ``arch.use_polarized_correlation``.

    The checkpoint's model class -- the parent anchor and the descriptor
    coordinates, neither of which changes a parameter shape and so neither of
    which the leaf stream reveals -- is compared with the class of the
    skeleton before the leaves are read, from the record written beside the
    checkpoint by the training stage (``checkpoint_class``). A checkpoint of
    the other class would otherwise load here in silence and be evaluated as
    a model that is neither."""
    import equinox as eqx
    from xcquinox.alec.checkpoint_class import (model_class_of_arch,
                                                require_matching_class)
    from xcquinox.alec.models import AlecGGAModel
    skeleton = AlecGGAModel.from_arch(training_spec.arch, seed=0)
    require_matching_class(model_path,
                           model_class_of_arch(training_spec.arch))
    polarized = arch_polarized_flag(training_spec.arch)
    mode = "polarized (UKS for open-shell)" if polarized else "unpolarized (RKS)"
    print(f"  arch: {getattr(training_spec.arch, 'name', '?')}  "
          f"[{mode}]", flush=True)
    return eqx.tree_deserialise_leaves(str(model_path), skeleton)


# ---------------------------------------------------------------------------
# Cluster-side high-level driver
# ---------------------------------------------------------------------------

def descriptors_and_required_keys(training_spec):
    """``(descriptors, required_keys, mode_str)`` for a training spec.

    ``descriptors`` are the arch's materialized descriptors (matched to what
    the trained model consumes); ``required_keys`` is ``("eri",)`` when the
    training solver is a non-ONESHOT SCF (so ``run_scf`` can rebuild J), else
    ``()``. These determine the precompute and are identical for every spec
    that shares an arch descriptor signature + solver mode, which is what
    lets a caller precompute ONCE and reuse the result across specs."""
    try:
        descriptors = tuple(training_spec.arch.materialize_descriptors())
    except AttributeError:
        descriptors = ()
    spec_solver_config = getattr(training_spec, "solver_config", None)
    needs_scf = (
        spec_solver_config is not None
        and hasattr(spec_solver_config, "mode")
        and spec_solver_config.mode.value != "oneshot"
    )
    if needs_scf:
        # DF path needs the 3-index cderi instead of the full 4-index eri.
        required_keys = (("cderi",)
                         if getattr(spec_solver_config, "density_fit", False)
                         else ("eri",))
    else:
        required_keys = ()
    mode_str = (
        spec_solver_config.mode.name if spec_solver_config is not None
        and hasattr(spec_solver_config, "mode") else "fixed_density"
    )
    return descriptors, required_keys, mode_str


def precompute_holdout_for_spec(training_spec, mol_specs: Dict[str, Any]):
    """Precompute PBE + grid + (eri) + descriptor features for one spec's arch.

    The expensive part (PBE SCF + integrals) depends only on the geometry,
    basis, descriptor signature and solver mode, NOT on the trained weights,
    so the returned ``mol_data`` is reusable by ``run_full_holdout_eval`` (via
    its ``mol_data=`` argument) for EVERY spec that shares the same descriptor
    signature + solver mode. This is what turns an N-spec re-eval from N
    precomputes into one-per-descriptor-group."""
    descriptors, required_keys, mode_str = descriptors_and_required_keys(
        training_spec)
    # Forward the spec's DF auxbasis so held-out cderi uses the same fitting
    # basis as training (else build_cderi falls back to a possibly-different
    # auto aux). Only meaningful when density_fit is on; None otherwise.
    sc = getattr(training_spec, "solver_config", None)
    auxbasis = (getattr(sc, "auxbasis", None)
                if getattr(sc, "density_fit", False) else None)
    orientation_lock_strength = getattr(sc, "orientation_lock_strength", 0.0)
    seed_source = getattr(sc, "seed_source", "pbe")
    print(f"[holdout] precomputing {len(mol_specs)} species "
          f"(descriptors: {[type(d).__name__ for d in descriptors] or 'none'}; "
          f"solver: {mode_str}; seed: {seed_source}; extra precompute keys: "
          f"{list(required_keys) or 'none'}) ...", flush=True)
    return precompute_holdout(mol_specs, descriptors=descriptors,
                              required_keys=required_keys, auxbasis=auxbasis,
                              orientation_lock_strength=orientation_lock_strength,
                              seed_source=seed_source,
                              seed_cache_dir=getattr(sc, "seed_cache_dir", None),
                              seed_density_fit=bool(
                                  getattr(sc, "density_fit", False)))


def run_full_holdout_eval(
    training_spec,
    model,
    mol_specs: Dict[str, Any],
    reactions: Sequence[Dict[str, Any]],
    out_dir: Path,
    *,
    strict: Optional[bool] = None,
    mol_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """End-to-end held-out eval driver, what the cluster eval task calls.

    Workflow:
      1. Decide ``required_keys=("eri",)`` if the training spec's
         solver_config requests a non-ONESHOT SCF (so ``run_scf`` has the
         electron-repulsion integrals it needs).
      2. Precompute PBE on every species in ``mol_specs``.
      3. NN forward (one-shot or SCF, matching the training solver) on
         every species.
      4. Filter reactions per ``strict`` (env var
         ``XCQUINOX_HELDOUT_STRICT=1`` overrides the kwarg; default is
         loose).
      5. Compute per-pool MAE + per-reaction NN/PBE error records.
      6. Write three artifacts under ``out_dir``:
         - ``test_set.csv``: per-pool + combined MAE rows
         - ``per_molecule.json``: per-species E_nn + E_pbe + flags
         - ``per_reaction.json``: per-reaction NN + PBE errors with
           ``in_sample_overlap`` lists

    ``mol_data``: when provided (from :func:`precompute_holdout_for_spec`),
    the expensive PBE/grid/eri precompute is SKIPPED and the supplied data is
    reused: the caller is responsible for ensuring it was built with a
    matching descriptor signature + solver mode (same arch group).

    Returns a small summary dict (counts + output paths) for the caller's
    log line.
    """
    if strict is None:
        strict = os.environ.get("XCQUINOX_HELDOUT_STRICT") == "1"
    per = compute_holdout_per_molecule(
        training_spec, model, mol_specs, mol_data=mol_data)
    excl, key_map = trained_reaction_exclusion(training_spec, mol_specs)
    return _finalize_holdout_outputs(
        reactions, per["energies"], per["pbe_energies"], per["mol_records"],
        per["training_names"], per["n_species"], out_dir, strict=strict,
        excluded_identities=excl, species_key_map=key_map)


def compute_holdout_per_molecule(training_spec, model, mol_specs: Dict[str, Any],
                                 *, mol_data: Optional[Dict[str, Any]] = None
                                 ) -> Dict[str, Any]:
    """Per-molecule stage of the held-out eval (the parallelizable part).

    Precomputes PBE/grid/(eri) and runs the NN forward (one-shot or SCF) on
    every species in ``mol_specs``, then builds the per-molecule records. This
    is pure of reaction aggregation and file IO, so a sharded driver can call it
    on a SUBSET of ``mol_specs`` and merge the returned maps (the loops iterate
    the dict, so a shard is just a smaller ``mol_specs``).

    Returns a dict with: ``energies`` (name -> E_nn), ``pbe_energies``
    (name -> E_pbe), ``scf_info`` (name -> SCF convergence dict), ``mol_records``
    (per_molecule.json rows, sorted by name), ``n_species`` and ``training_names``.
    """
    _descriptors, _extra_required, mode_str = descriptors_and_required_keys(
        training_spec)
    spec_solver_config = getattr(training_spec, "solver_config", None)

    # FULL species list (incl. atoms) -- ONLY for the per-molecule
    # `in_training_subset` flag below.
    training_names = tuple(
        getattr(m, "name", "?") for m in
        getattr(training_spec, "molecules", ())
    )
    # MOLECULE-level names (single atoms excluded) -- what the held-out OVERLAP
    # filter must use; atoms are universal anchors, not held-out molecules.
    # Expanded with the pool names physically identical to a trained molecule
    # under a different naming scheme (Hill 'CHN' vs pool 'hcn'): name
    # matching alone leaves those trained twins inside the "held-out" set.
    held_out_filter_names = held_out_filter_names_with_aliases(
        training_spec, mol_specs)

    if mol_data is None:
        mol_data = precompute_holdout_for_spec(training_spec, mol_specs)
    else:
        print(f"[holdout] reusing precomputed {len(mol_data)} species "
              f"(solver: {mode_str}) ...", flush=True)

    print(f"[holdout] evaluating model on {len(mol_data)} species "
          f"(solver: {mode_str}) ...", flush=True)
    scf_info: Dict[str, Dict[str, Any]] = {}
    energies = evaluate_holdout(model, mol_data,
                                 solver_config=spec_solver_config,
                                 scf_info_out=scf_info)

    pbe_energies = {n: float(md.get("E_pbe"))
                    for n, md in mol_data.items()
                    if md.get("E_pbe") is not None
                    and math.isfinite(float(md.get("E_pbe")))}

    # Case-insensitive, to match the case-insensitive reaction overlap: CH4
    # (BH76) and ch4 (W4-11) are the SAME molecule, so the descriptive
    # in_training_subset flag must agree with the strict-drop filter --
    # including the composition-level aliases (Hill vs pool naming).
    from xcquinox.alec.species_matching import trained_pool_aliases
    flag_names = set(training_names) | trained_pool_aliases(
        training_names, mol_specs, verbose=False)
    training_cf = {str(t).casefold() for t in flag_names}
    mol_records: List[Dict[str, Any]] = []
    for name in sorted(mol_data):
        mol_records.append(make_per_molecule_record(
            name, mol_data[name], energies.get(name, float("nan")),
            in_training_subset=(str(name).casefold() in training_cf),
            scf=scf_info.get(name),
            # all-None without benchmark CCSD refs (rho_ref_grid is None then)
            density=density_errors_for_record(
                model, mol_data[name], solver_config=spec_solver_config),
        ))

    return {
        "energies": energies,
        "pbe_energies": pbe_energies,
        "scf_info": scf_info,
        "mol_records": mol_records,
        "n_species": len(mol_data),
        "training_names": held_out_filter_names,
    }


def is_finite_energy(value: Any) -> bool:
    """True only for a real, finite number. ``None`` (an absent shard entry)
    and ``NaN`` (a species whose evaluation raised) are both non-finite, so
    the two failure representations are handled by one predicate."""
    return (isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(float(value)))


def merge_holdout_shards(shard_payloads: Sequence[Dict[str, Any]]
                         ) -> Tuple[Dict[str, float], Dict[str, float],
                                    List[Dict[str, Any]]]:
    """Merge per-shard ``{energies, pbe_energies, mol_records}`` payloads into
    the combined maps the finalize stage consumes.

    A species evaluated in ONE shard only (the common case) passes through
    unchanged. A species that was RE-QUEUED to a lower-parallelism tier appears
    in several payloads, so the union is resolved by precedence: a finite value
    beats a non-finite one (the retry that succeeded is the result), and among
    equally-finite (or equally non-finite) values the LAST payload wins, i.e.
    the lowest-parallelism tier that ran it. ``mol_records`` follow the same
    rule on their ``E_total_nn`` and are de-duplicated by molecule name, so
    per_molecule.json keeps exactly one row per species; the records are
    re-sorted by name to match the serial ordering."""
    energies: Dict[str, float] = {}
    pbe_energies: Dict[str, float] = {}
    records_by_name: Dict[str, Dict[str, Any]] = {}
    unnamed_records: List[Dict[str, Any]] = []

    def _accept(dest: Dict[str, Any], key: str, value: Any) -> None:
        if key in dest and not is_finite_energy(value) \
                and is_finite_energy(dest[key]):
            return                             # keep the finite earlier value
        dest[key] = value

    for payload in shard_payloads:
        for name, e in (payload.get("energies") or {}).items():
            _accept(energies, name, e)
        for name, e in (payload.get("pbe_energies") or {}).items():
            _accept(pbe_energies, name, e)
        for rec in payload.get("mol_records") or []:
            name = rec.get("molecule")
            if name is None:
                unnamed_records.append(rec)
                continue
            prev = records_by_name.get(name)
            if prev is not None \
                    and not is_finite_energy(rec.get("E_total_nn")) \
                    and is_finite_energy(prev.get("E_total_nn")):
                continue
            records_by_name[name] = rec

    mol_records = list(records_by_name.values()) + unnamed_records
    mol_records.sort(key=lambda r: str(r.get("molecule") or ""))
    return energies, pbe_energies, mol_records


def _n_nan_union(energies_ha: Dict[str, float],
                 pbe_energies_ha: Dict[str, float],
                 reactions: Sequence[Dict[str, Any]]) -> int:
    """Count reactions dropped (non-finite abs error) in EITHER the NN or the
    PBE metric, i.e. the union. The two metrics can drop DIFFERENT reactions, so
    max(n_nan_nn, n_nan_pbe) undercounts the true dropped set."""
    nn = list(per_reaction_errors(energies_ha, reactions))
    pb = list(per_reaction_errors(pbe_energies_ha, reactions))
    return sum(
        1 for a, b in zip(nn, pb)
        if not (math.isfinite(a["abs_error_kcalmol"])
                and math.isfinite(b["abs_error_kcalmol"]))
    )


def _finalize_holdout_outputs(reactions: Sequence[Dict[str, Any]],
                              energies: Dict[str, float],
                              pbe_energies: Dict[str, float],
                              mol_records: List[Dict[str, Any]],
                              training_names: Sequence[str],
                              n_species: int,
                              out_dir: Path, *, strict: bool,
                              excluded_identities: Optional[set] = None,
                              species_key_map: Optional[
                                  Dict[str, Tuple[str, ...]]] = None
                              ) -> Dict[str, Any]:
    """Reaction aggregation + artifact writing, the fast serial tail of the
    held-out eval, shared by the serial driver and the sharded/parallel driver.

    Needs ALL molecule energies (reactions span the whole pool), so it runs once
    after every shard has finished. Writes ``test_set.csv``, ``per_molecule.json``
    and ``per_reaction.json`` under ``out_dir`` and returns the summary dict.

    ``strict`` drops the VERBATIM supervised reactions: rows whose canonical
    identity (``species_key_map``) intersects ``excluded_identities`` (built
    by :func:`trained_reaction_exclusion`; the recorded validation slice was
    already removed upstream). Species-level overlap no longer drops anything
    -- a reaction merely containing a trained molecule is a generalization
    target -- but is still ANNOTATED per row (``in_sample_overlap``, via the
    loose :func:`filter_reactions` mode) and per molecule
    (``in_training_subset``)."""
    from xcquinox.alec.species_matching import reaction_identity_keys
    if strict and not excluded_identities and training_names:
        print("[holdout] WARNING: strict mode with an EMPTY verbatim-"
              "exclusion set while the spec records trained molecules -- "
              "the training record may predate reaction-form points; no "
              "supervised reaction will be dropped", flush=True)
    # Partition reactions by source_pool so we can write per-pool rows.
    by_pool: Dict[str, List[Dict[str, Any]]] = {}
    for r in reactions:
        by_pool.setdefault(r.get("source_pool", "unknown"), []).append(r)

    per_pool_mae: Dict[str, Tuple[float, float, int, int, int]] = {}
    all_kept: List[Dict[str, Any]] = []
    n_dropped_total = 0
    n_nan_total = 0
    for pool, pool_rxns in by_pool.items():
        # loose mode: every reaction kept, species overlap annotated
        kept, _ = filter_reactions(pool_rxns, training_names, strict=False)
        dropped: List[Dict[str, Any]] = []
        if strict and excluded_identities:
            kept2: List[Dict[str, Any]] = []
            for rxn in kept:
                ids = reaction_identity_keys(rxn, species_key_map or {})
                if ids and set(ids) & excluded_identities:
                    dropped.append(rxn)
                else:
                    kept2.append(rxn)
            kept = kept2
        n_dropped_pool = len(dropped)
        mae_nn, n_used, n_nan_nn = reaction_mae_kcalmol(energies, kept)
        mae_pbe, _, n_nan_pbe = reaction_mae_kcalmol(pbe_energies, kept)
        # Union: NN and PBE can drop DIFFERENT reactions, so max() undercounts.
        n_nan = _n_nan_union(energies, pbe_energies, kept)
        per_pool_mae[pool] = (mae_nn, mae_pbe, n_used, n_dropped_pool, n_nan)
        all_kept.extend(kept)
        n_dropped_total += n_dropped_pool
        n_nan_total += n_nan
    combined_mae_nn, combined_n_used, combined_n_nan_nn = reaction_mae_kcalmol(
        energies, all_kept)
    combined_mae_pbe, _, combined_n_nan_pbe = reaction_mae_kcalmol(
        pbe_energies, all_kept)
    combined = (combined_mae_nn, combined_mae_pbe, combined_n_used,
                n_dropped_total, _n_nan_union(energies, pbe_energies, all_kept))

    nn_per_rxn = per_reaction_errors(energies, all_kept)
    pbe_per_rxn = per_reaction_errors(pbe_energies, all_kept)
    rxn_records = make_per_reaction_records(
        all_kept, nn_per_rxn, pbe_per_rxn, training_names)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = write_test_set_csv(
        out_dir / DEFAULT_CSV_NAME, per_pool_mae, combined, strict)
    mol_json_path = write_per_molecule_json(
        out_dir / DEFAULT_PER_MOLECULE_NAME, mol_records)
    rxn_json_path = write_per_reaction_json(
        out_dir / DEFAULT_PER_REACTION_NAME, rxn_records)
    print(f"[holdout] wrote {csv_path.name}, "
          f"{mol_json_path.name}, {rxn_json_path.name} "
          f"({len(all_kept)} reactions, {n_nan_total} NaN-drops, "
          f"{n_dropped_total} verbatim-supervised drops)", flush=True)

    return {
        "n_reactions": len(all_kept),
        "n_species": n_species,
        "n_dropped_overlap": n_dropped_total,
        "n_dropped_nan": n_nan_total,
        "per_pool_mae": per_pool_mae,
        "combined": combined,
        "csv_path": str(csv_path),
        "per_molecule_path": str(mol_json_path),
        "per_reaction_path": str(rxn_json_path),
    }
