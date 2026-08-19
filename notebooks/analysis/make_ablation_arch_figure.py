#!/usr/bin/env python
"""Architecture-ablation figures for the ``ablation_notransform`` sweep.

The existing :mod:`make_cluster_pulls_figure` renders a category-level suite
but keys every series on ``(metric, solver)`` -- so it collapses the eight
architectures of this ablation (which holds ``metric=jsd``/``solver=full_3``
fixed and varies ``arch``) into a single line. This module fills that gap:
every figure here is **architecture-aware**, and the parity figure is modeled
on Figure 5 of Navarro-Rodriguez et al., *Constraint-aware functional cloning*
(MLXC_Constraints, 2026) -- predicted-vs-reference scatter with a y=x diagonal
and a per-network mean-error inset.

It reuses the data collectors and house style from
``make_cluster_pulls_figure`` verbatim (no re-parsing of the run dir), adding
only an ``eval_holdout/per_reaction.json`` collector (the cluster-side held-out
reaction eval -- same schema as the local-reeval ``local_per_reaction.json``
that the existing module reads, but a different source path).

Scientific provenance carried on every figure:
  * The pulled run ``run_20260529T165503Z`` predates the ``dm_entropy`` fix
    from the 2026-05-29 forensic review -- these are PRE-FIX numbers.
  * On the held-out reactions ``de_nn ≈ de_pbe`` while both sit far from the
    benchmark refs: the network faithfully *reproduces PBE*, it does not beat
    it. The two parity panels make that explicit rather than hiding it.
  * Coverage is partial (57/80 specs trained; only 32 carry held-out
    reactions) -- incomplete grid cells are drawn hatched, never dropped.

Figures written (PNG):
  A. ``ablation_parity.png``       -- Fig-5 analog, 2 panels (NN-vs-PBE and
     NN&PBE-vs-benchmark), points colored by arch, per-arch MAE inset bars.
  B. ``ablation_arch_subset_heatmap.png`` -- arch × subset_size MAE heatmap
     (held-out reaction MAE + in-sample atomization-energy MAE).
  C. ``ablation_mae_by_arch.png``  -- per-arch MAE bars (log-y), held-out
     reaction + in-sample AE, with the PBE-vs-benchmark baseline line.
  D. ``ablation_mae_vs_subset.png``-- MAE vs subset_size, one line per arch.
  E. ``ablation_ae_parity.png``    -- HELD-OUT atomization-energy parity (W4-11,
     the held-out set's atomization-energy pool): predicted vs reference AE with
     PBE drawn as the baseline (grey × + dashed). Panel (a) by architecture,
     panel (b) colored by training-subset size with an NN-MAE-vs-subset inset.
     The AE analog of the held-out reaction parity (A).

Usage:
    python notebooks/analysis/make_ablation_arch_figure.py \
        [--run-dir <pulled run dir>] \
        [--outdir notebooks/analysis/figures_ablation_notransform]
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# ---------------------------------------------------------------------------
# Reuse the sibling module's collectors + style (load by path; this directory
# is not an importable package).
# ---------------------------------------------------------------------------

_CCP_PATH = Path(__file__).resolve().parent / "make_cluster_pulls_figure.py"
_ccp_spec = importlib.util.spec_from_file_location(
    "make_cluster_pulls_figure", _CCP_PATH)
ccp = importlib.util.module_from_spec(_ccp_spec)  # type: ignore[arg-type]
sys.modules["make_cluster_pulls_figure"] = ccp
_ccp_spec.loader.exec_module(ccp)  # type: ignore[union-attr]

HA_TO_KCAL = 627.5094740631  # CODATA-2018, matches analyze.HA_TO_KCAL

# ---------------------------------------------------------------------------
# Ablation axes + palette
# ---------------------------------------------------------------------------

# Fixed display order: baseline first, then attention / descriptor variants,
# then the notransform pair (the headline of *this* ablation) last.
# ARCH_ORDER / ARCH_COLOR / SUBSET_SIZES and the Jacob's-ladder rung taxonomy now
# live in the shared ``arch_style`` module, imported by every figure script (this
# suite, the DFS demo notebook, plot_pretraining_curves.py) so the rung /
# meta-GGA-vs-SCAN story reads consistently. Loaded by PATH -- this directory is
# not an importable package (identical mechanism to the ``ccp`` sibling above).
_AS_PATH = Path(__file__).resolve().parent / "arch_style.py"
_as_spec = importlib.util.spec_from_file_location("arch_style", _AS_PATH)
arch_style = importlib.util.module_from_spec(_as_spec)  # type: ignore[arg-type]
sys.modules["arch_style"] = arch_style
_as_spec.loader.exec_module(arch_style)  # type: ignore[union-attr]

ARCH_ORDER: Tuple[str, ...] = arch_style.ARCH_ORDER
ARCH_COLOR: Dict[str, str] = arch_style.ARCH_COLOR
SUBSET_SIZES: Tuple[int, ...] = arch_style.SUBSET_SIZES
POOL_MARKER: Dict[str, str] = {"bh76": "o", "w411": "^"}

# Compact rung tags for tight gutters/labels (the full RUNG_ORDER names are too
# long for a heatmap gutter or a 1-arch-tall rung span).
_RUNG_SHORT: Dict[str, str] = {
    arch_style.RUNG_GGA: "GGA",
    arch_style.RUNG_MGGA: "mGGA",
    arch_style.RUNG_R35: "r3.5",
    arch_style.RUNG_R35_MGGA: "r3.5+m",
}

_STYLE = dict(ccp._STYLE)

# Provenance banner (static methodology note). The PBE baseline and the
# NN-vs-PBE headline are computed LIVE per-run and appended -- no hardcoded
# benchmark numbers (see pbe_pool_baseline / provenance_footer /
# nn_vs_pbe_caveat below). build_all() stamps the dynamic strings; direct calls
# to a plot fn fall back to this base banner.
_PROVENANCE_BASE = (
    "Held-out: GMTKN55-BH76 barrier heights + W4-11 atomization energies "
    "(reaction energies, kcal/mol)."
)


def _is_num(v: Any) -> bool:
    """True iff v is a real, finite number."""
    return isinstance(v, (int, float)) and math.isfinite(v)


# ---------------------------------------------------------------------------
# Data ingest
# ---------------------------------------------------------------------------

_POOL_SPECS_CACHE: Optional[Dict[str, Any]] = None
_POOL_CACHE: Optional[Tuple[Dict[str, Any], List[Dict[str, Any]]]] = None


def _canonical_pool() -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """``(pool species specs, pool reactions)`` -- the canonical held-out
    benchmark pool the cluster eval scores against. Lazy + cached (one load
    per process); monkeypatchable test seam."""
    global _POOL_CACHE
    if _POOL_CACHE is None:
        from xcquinox.alec.full_benchmark_pools import (
            load_full_held_out_pools)
        specs, rxns = load_full_held_out_pools()
        _POOL_CACHE = (specs, list(rxns))
    return _POOL_CACHE


def _pool_specs_for_aliasing() -> Dict[str, Any]:
    """Benchmark-pool species specs for composition-level alias matching --
    the figure-layer twin of the eval-side
    ``held_out_filter_names_with_aliases`` expansion. Lazy + cached (one pool
    load per process); monkeypatchable test seam."""
    global _POOL_SPECS_CACHE
    if _POOL_SPECS_CACHE is None:
        from xcquinox.alec.full_benchmark_pools import (
            load_full_held_out_pools)
        _POOL_SPECS_CACHE = load_full_held_out_pools()[0]
    return _POOL_SPECS_CACHE


def _spec_alias_names(spec_dir: Path) -> set:
    """Casefolded pool names physically identical to this spec's trained
    molecules under a DIFFERENT name (Hill ``CHN`` vs pool ``hcn``) --
    exactly the set the cluster-side name-level strict filter could not see.
    Name-visible trained species were already dropped there, so only these
    aliases need removing here. Empty set when metadata is absent."""
    tm = spec_dir / "train_metadata.json"
    if not tm.is_file():
        return set()
    try:
        with tm.open() as f:
            mols = json.load(f).get("molecules") or []
    except (json.JSONDecodeError, OSError):
        return set()
    from xcquinox.alec.species_matching import (is_atomic,
                                                parse_formula_name,
                                                trained_pool_aliases)
    mol_level = []
    for n in mols:
        parsed = parse_formula_name(str(n))
        if parsed is not None and is_atomic(parsed[0]):
            continue  # atoms are universal anchors, never held-out species
        mol_level.append(str(n))
    if not mol_level:
        return set()
    aliases = trained_pool_aliases(mol_level, _pool_specs_for_aliasing(),
                                   verbose=False)
    return {str(a).casefold() for a in aliases}


def _reaction_identity(r: Dict[str, Any]) -> Optional[Tuple]:
    """Order-invariant physical identity of a reaction row: the sorted
    casefolded reactant and product name tuples. ``None`` when either side is
    missing (older rows without species lists are never identity-matched)."""
    reac = r.get("reactants")
    prod = r.get("products")
    if not reac or not prod:
        return None
    return (tuple(sorted(str(x).casefold() for x in reac)),
            tuple(sorted(str(x).casefold() for x in prod)))


def _val_reaction_identities(run_dir: Path) -> set:
    """Identities of the run's validation-slice reactions, from
    ``validation/val_reactions.json`` -- checked in ``run_dir`` itself and in
    every source run a merged symlink view resolves into. Empty set when no
    validation record exists (pre-validation runs render unchanged)."""
    cands = [Path(run_dir) / "validation" / "val_reactions.json"]
    for _idx, sd in ccp._spec_dirs(run_dir):
        if sd.is_symlink():
            cands.append(sd.resolve().parent.parent / "validation"
                         / "val_reactions.json")
    out: set = set()
    seen_files: set = set()
    for p in cands:
        try:
            rp = p.resolve()
        except OSError:
            continue
        if rp in seen_files or not p.is_file():
            continue
        seen_files.add(rp)
        try:
            with p.open() as f:
                entries = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        for e in entries:
            ident = _reaction_identity(e)
            if ident is not None:
                out.add(ident)
    return out


def collect_holdout_reaction_rows(run_dir: Path,
                                  eval_subdir: str = "eval_holdout"
                                  ) -> List[Dict[str, Any]]:
    """Read every ``checkpoints/spec_*/<eval_subdir>/per_reaction.json`` (the
    cluster-side held-out reaction eval) and join with the manifest cell.

    One row per (spec, reaction); schema mirrors
    ``ccp.collect_per_reaction_rows`` but sourced from ``<eval_subdir>/`` rather
    than the local-reeval ``eval/local_per_reaction.json``. Specs without the
    file (e.g. the 25 specs whose held-out eval did not run) are skipped.
    ``eval_subdir`` selects the checkpoint variant: ``eval_holdout`` (final-step
    weights, default) or ``eval_holdout_val_best`` (held-out validation-best weights).

    Two strict-holdout repairs applied on read (each printed when it fires):
    rows whose reaction contains a pool species physically identical to one of
    that spec's trained molecules under a different name (``_spec_alias_names``
    -- the cluster-side name filter is blind to the Hill-vs-pool naming split)
    are dropped, and rows whose reaction is a permuted-name twin of a
    validation-slice reaction (``_val_reaction_identities`` -- validation-best
    selection saw that barrier) are dropped.

    Rows require a finite comparator (PBE) leg only: reactions whose NN leg
    is NaN are kept with NaN NN columns on BOTH ingest paths (the cluster
    rows already carry them; the reconstruction now emits them too), so
    per-cell comparator reductions cover each cell's full test slice while
    NN reducers skip the unscored rows via their per-key finiteness
    filters."""
    cells = ccp._read_manifest_cells(run_dir)
    rows: List[Dict[str, Any]] = []
    # -- verbatim-holdout reconstruction (specs whose per_molecule carries the
    #    per-species energies) --------------------------------------------
    recon_stats = {"specs": 0, "verbatim": 0, "val": 0, "nan_pbe": 0,
                   "nan_nn": 0}
    legacy_specs: List[Tuple[int, Path]] = []
    for idx, spec_dir in ccp._spec_dirs(run_dir):
        got = _reconstruct_spec_rows(run_dir, idx, spec_dir, cells,
                                     eval_subdir, recon_stats)
        if got is None:
            legacy_specs.append((idx, spec_dir))
        else:
            rows.extend(got)
    if recon_stats["specs"]:
        print(f"  (verbatim holdout: reconstructed {recon_stats['specs']} "
              f"specs' test slices from per-species energies; excluded "
              f"{recon_stats['verbatim']} verbatim-supervised and "
              f"{recon_stats['val']} validation rows; "
              f"{recon_stats['nan_pbe']} comparator-NaN-dropped; "
              f"{recon_stats['nan_nn']} NN-NaN rows kept "
              f"(comparator leg only))")
    if not legacy_specs:
        return rows
    # -- legacy path: cluster-written per_reaction.json (pulls whose
    #    per_molecule.json predates the energy columns), with the
    #    species-alias and validation-twin repairs ------------------------
    val_ids = _val_reaction_identities(run_dir)
    n_alias = 0
    alias_hits: set = set()
    n_twin = 0
    twin_hits: set = set()
    n_specs_alias = 0
    for idx, spec_dir in legacy_specs:
        rj_path = spec_dir / eval_subdir / "per_reaction.json"
        if not rj_path.is_file():
            continue
        try:
            with rj_path.open() as f:
                payload = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        cell = cells.get(idx, {})
        aliases_cf = _spec_alias_names(spec_dir)
        spec_had_alias_drop = False
        for r in payload:
            species_cf = [str(x).casefold()
                          for x in ((r.get("reactants") or [])
                                    + (r.get("products") or []))]
            if aliases_cf and any(s in aliases_cf for s in species_cf):
                n_alias += 1
                spec_had_alias_drop = True
                alias_hits.update(s for s in species_cf if s in aliases_cf)
                continue
            if val_ids:
                ident = _reaction_identity(r)
                if ident is not None and ident in val_ids:
                    n_twin += 1
                    twin_hits.add(str(r.get("name")))
                    continue
            rows.append({
                "idx": idx,
                "arch": cell.get("arch"),
                "subset_size": cell.get("subset_size"),
                "name": r.get("name"),
                "pool": r.get("pool"),
                "ref_kcalmol": r.get("reaction_energy_ref_kcalmol"),
                "de_nn_kcalmol": r.get("de_nn_kcalmol"),
                "de_pbe_kcalmol": r.get("de_pbe_kcalmol"),
                "abs_error_nn_kcalmol": r.get("abs_error_nn_kcalmol"),
                "abs_error_pbe_kcalmol": r.get("abs_error_pbe_kcalmol"),
                # species membership (for per-channel density/ED views)
                "reactants": r.get("reactants"),
                "products": r.get("products"),
            })
        if spec_had_alias_drop:
            n_specs_alias += 1
    if n_alias:
        print(f"  (strict-holdout repair: dropped {n_alias} reaction rows "
              f"across {n_specs_alias} specs containing trained species "
              f"under pool names {sorted(alias_hits)})")
    if n_twin:
        print(f"  (validation-twin repair: dropped {n_twin} test rows whose "
              f"reaction is a permuted-name twin of a validation reaction: "
              f"{sorted(twin_hits)})")
    return rows


class _MetadataSpec:
    """Duck-typed training-spec view over a pulled ``train_metadata.json``,
    exposing exactly what ``trained_reaction_exclusion`` consumes."""
    def __init__(self, meta: Dict[str, Any]):
        self._lk = dict(meta.get("loss_kwargs") or {})
    def loss_kwargs_dict(self) -> Dict[str, Any]:
        return self._lk


def _val_reaction_entries(run_dir: Path) -> List[Dict[str, Any]]:
    """Raw ``validation/val_reactions.json`` entries, from ``run_dir`` and
    every source run a merged symlink view resolves into (deduplicated by
    file). Empty list when no validation record exists."""
    cands = [Path(run_dir) / "validation" / "val_reactions.json"]
    for _idx, sd in ccp._spec_dirs(run_dir):
        if sd.is_symlink():
            cands.append(sd.resolve().parent.parent / "validation"
                         / "val_reactions.json")
    out: List[Dict[str, Any]] = []
    seen: set = set()
    for p in cands:
        try:
            rp = p.resolve()
        except OSError:
            continue
        if rp in seen or not p.is_file():
            continue
        seen.add(rp)
        try:
            with p.open() as f:
                out.extend(json.load(f))
        except (json.JSONDecodeError, OSError):
            continue
    return out


_VAL_IDENTITY_CACHE: Dict[Tuple[str, int], set] = {}
_POOL_KEY_MAP_CACHE: Optional[Tuple[Any, Dict[str, Tuple[str, ...]]]] = None


def _reconstruct_spec_rows(run_dir: Path, idx: int, spec_dir: Path,
                           cells: Dict[int, Dict[str, Any]],
                           eval_subdir: str,
                           stats: Dict[str, int]
                           ) -> Optional[List[Dict[str, Any]]]:
    """One spec's VERBATIM-HOLDOUT test slice, reconstructed from its
    per-species energies (``E_total_nn`` / ``E_pbe`` in
    ``<eval_subdir>/per_molecule.json``) over the canonical pool with the
    cluster's own reaction math (``eval_holdout.per_reaction_errors``).

    Exclusions -- by canonical reaction identity -- are exactly the spec's
    verbatim supervised reactions (``trained_reaction_exclusion`` over the
    training record's reaction points) and the recorded validation slice;
    a reaction merely containing a trained molecule STAYS. Rows require a
    finite COMPARATOR (PBE) leg only: reactions the NN failed to score are
    kept with NaN NN columns, so comparator reductions cover the cell's
    full test slice regardless of NN convergence (the cluster-written
    per_reaction.json carries the same convention). ``None`` when the
    spec's per_molecule predates the energy columns (caller falls back to
    the cluster-written per_reaction.json)."""
    pm_path = spec_dir / eval_subdir / "per_molecule.json"
    if not pm_path.is_file():
        return None
    try:
        with pm_path.open() as f:
            pm = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    e_nn = {str(r.get("molecule")): float(r["E_total_nn"]) for r in pm
            if _is_num(r.get("E_total_nn"))}
    e_pbe = {str(r.get("molecule")): float(r["E_pbe"]) for r in pm
             if _is_num(r.get("E_pbe"))}
    if not e_nn or not e_pbe:
        return None
    from xcquinox.alec.eval_holdout import (per_reaction_errors,
                                            trained_reaction_exclusion)
    from xcquinox.alec.species_matching import reaction_identity_keys
    pool_specs, pool_rxns = _canonical_pool()
    tm_path = spec_dir / "train_metadata.json"
    meta: Dict[str, Any] = {}
    if tm_path.is_file():
        try:
            with tm_path.open() as f:
                meta = json.load(f)
        except (json.JSONDecodeError, OSError):
            meta = {}
    excl, key_map = trained_reaction_exclusion(_MetadataSpec(meta),
                                               pool_specs)
    # A pool-name key map always exists (pool-name keys are identical under
    # any trained-name extension, so identities computed with either map
    # coincide on pool reactions); the trained map is preferred when the
    # spec records reaction points, since the exclusion set was built on it.
    global _POOL_KEY_MAP_CACHE
    if _POOL_KEY_MAP_CACHE is None or _POOL_KEY_MAP_CACHE[0] is not pool_specs:
        from xcquinox.alec.species_matching import canonical_species_keys
        _POOL_KEY_MAP_CACHE = (pool_specs,
                               canonical_species_keys(pool_specs))
    id_map = key_map or _POOL_KEY_MAP_CACHE[1]
    cache_key = (str(Path(run_dir).resolve()), id(pool_rxns))
    val_ids = _VAL_IDENTITY_CACHE.get(cache_key)
    if val_ids is None:
        val_ids = set()
        for e in _val_reaction_entries(run_dir):
            val_ids.update(reaction_identity_keys(
                e, _POOL_KEY_MAP_CACHE[1]))
        _VAL_IDENTITY_CACHE[cache_key] = val_ids
    nn_err = per_reaction_errors(e_nn, pool_rxns)
    pbe_err = per_reaction_errors(e_pbe, pool_rxns)
    cell = cells.get(idx, {})
    out: List[Dict[str, Any]] = []
    for rxn, rn, rp in zip(pool_rxns, nn_err, pbe_err):
        if not _is_num(rp.get("abs_error_kcalmol")):
            stats["nan_pbe"] += 1     # comparator leg undefined: not in slice
            continue
        ids = set(reaction_identity_keys(rxn, id_map))
        if excl and ids & excl:
            stats["verbatim"] += 1
            continue
        if val_ids and ids & val_ids:
            stats["val"] += 1
            continue
        if not _is_num(rn.get("abs_error_kcalmol")):
            stats["nan_nn"] += 1      # NN leg NaN: row kept, NN columns NaN
        out.append({
            "idx": idx,
            "arch": cell.get("arch"),
            "subset_size": cell.get("subset_size"),
            "name": rxn.get("name"),
            "pool": rxn.get("source_pool"),
            "ref_kcalmol": rn.get("ref_kcalmol"),
            "de_nn_kcalmol": rn.get("de_kcalmol"),
            "de_pbe_kcalmol": rp.get("de_kcalmol"),
            "abs_error_nn_kcalmol": rn.get("abs_error_kcalmol"),
            "abs_error_pbe_kcalmol": rp.get("abs_error_kcalmol"),
            "reactants": list(rxn.get("reactants") or []),
            "products": list(rxn.get("products") or []),
        })
    stats["specs"] += 1
    return out


def collect_insample_ae_rows(run_dir: Path) -> List[Dict[str, Any]]:
    """In-sample atomization-energy errors from ``eval/per_molecule.json``
    (reusing ccp.collect_per_molecule_rows), filtered to molecules carrying a
    finite ``AE_error_kcalmol``. Atoms (skipped) and null-AE rows drop out."""
    out: List[Dict[str, Any]] = []
    for r in ccp.collect_per_molecule_rows(run_dir):
        if r.get("skipped"):
            continue
        if not _is_num(r.get("AE_error_kcalmol")):
            continue
        out.append(r)
    return out


def trained_spec_count(run_dir: Path,
                       eval_subdir: str = "eval_holdout") -> int:
    """Number of specs that ran training -- evidenced by a materialized
    ``model.eqx`` OR any eval output (``<eval_subdir>/per_reaction.json`` or
    ``eval/per_molecule.json``). Eval output implies the spec trained even when
    its weights were not pulled (eval-only sync), so the figure's coverage count
    is not understated to ``1/48`` when only one model.eqx came down."""
    n = 0
    for _idx, spec_dir in ccp._spec_dirs(run_dir):
        if ((spec_dir / "model.eqx").is_file()
                or (spec_dir / eval_subdir / "per_reaction.json").is_file()
                or (spec_dir / "eval" / "per_molecule.json").is_file()):
            n += 1
    return n


def _spec_in_progress(spec_dir: Path) -> bool:
    """Mid-training evidence: a resume checkpoint on disk (the array task
    writes ``resume_*.eqx`` before ``resume_state.pkl`` and deletes the set
    at completion, so the ``or`` covers both crash windows), gated on no
    completion sentinel; the no-final-weights half is the caller's check."""
    if (spec_dir / "completion.json").is_file():
        return False
    return ((spec_dir / "resume_state.pkl").is_file()
            or any(spec_dir.glob("resume_*.eqx")))


def arch_coverage(run_dir: Path,
                  eval_subdir: str = "eval_holdout") -> Dict[str, List[str]]:
    """Per-arch coverage of this (partial) run, computed from disk.

    Returns ``{"trained": [...], "holdout": [...], "insample": [...],
    "untrained": [...], "in_progress": [...]}`` -- arch names in
    ``ARCH_ORDER`` order. ``trained`` = has ``model.eqx``; ``holdout`` = has
    ``<eval_subdir>/per_reaction.json``; ``insample`` = has
    ``eval/per_molecule.json``; ``untrained`` = arch in the manifest grid with
    no trained spec at all; ``in_progress`` = arch with at least one spec
    carrying a resume checkpoint and no final weights FOR THAT SPEC --
    independent evidence, not a training-status verdict: an arch can be in
    ``in_progress`` and ``trained`` at once (finished cells plus a resuming
    one). Only ``coverage_note`` intersects it with ``untrained`` to decide
    the IN PROGRESS wording.
    """
    cells = ccp._read_manifest_cells(run_dir)
    trained: set = set()
    holdout: set = set()
    insample: set = set()
    in_progress: set = set()
    grid_archs: set = {c.get("arch") for c in cells.values() if c.get("arch")}
    for idx, spec_dir in ccp._spec_dirs(run_dir):
        arch = cells.get(idx, {}).get("arch")
        if arch is None:
            continue
        if (spec_dir / "model.eqx").is_file():
            trained.add(arch)
        elif _spec_in_progress(spec_dir):
            in_progress.add(arch)
        if (spec_dir / eval_subdir / "per_reaction.json").is_file():
            holdout.add(arch)
        if (spec_dir / "eval" / "per_molecule.json").is_file():
            insample.add(arch)

    def _ordered(s: set) -> List[str]:
        return ([a for a in ARCH_ORDER if a in s]
                + sorted(s - set(ARCH_ORDER)))

    return {
        "trained": _ordered(trained),
        "holdout": _ordered(holdout),
        "insample": _ordered(insample),
        # An arch with held-out or in-sample eval was obviously trained, so it is
        # NOT untrained even when its model.eqx was not pulled (eval-only sync).
        "untrained": _ordered(grid_archs - trained - holdout - insample),
        "in_progress": _ordered(in_progress),
    }


def coverage_note(run_dir: Path, eval_subdir: str = "eval_holdout") -> str:
    """One-line human summary of arch coverage for figure footers -- makes the
    partial-run gaps explicit (no silent truncation). An arch whose only
    on-disk evidence is a resume checkpoint reads IN PROGRESS, not NOT
    TRAINED -- a running array is not an absent one."""
    cov = arch_coverage(run_dir, eval_subdir=eval_subdir)
    parts = [f"Held-out reactions: {len(cov['holdout'])}/{len(ARCH_ORDER)} archs "
             f"({', '.join(cov['holdout']) or 'none'})."]
    if cov["untrained"]:
        inprog = [a for a in cov["untrained"]
                  if a in cov.get("in_progress", [])]
        not_started = [a for a in cov["untrained"] if a not in inprog]
        if inprog:
            parts.append("IN PROGRESS (resume checkpoint, no final weights "
                         f"yet): {', '.join(inprog)}.")
        if not_started:
            parts.append(
                f"NOT TRAINED in this run: {', '.join(not_started)}.")
    trained_no_holdout = [a for a in cov["trained"] if a not in cov["holdout"]]
    if trained_no_holdout:
        parts.append(f"Trained but no held-out eval: "
                     f"{', '.join(trained_no_holdout)}.")
    return "  ".join(parts)


def lockfix_boundary(run_dir: Path) -> Dict[str, Any]:
    """Mid-run density-reference swap boundary, or ``{}`` when the run has none.

    Reads ``lockfix_swap_manifest.json``, written into the run directory by
    ``hpcjobs/dfs6311_lockfix_swap.py`` when degenerate-radical references are
    relocked while a sweep is in flight. Specs that had started before the swap
    trained against the OLD references; those that had not train against the
    new ones, so a density comparison spanning the boundary mixes two
    reference sets and must say so. Returns ``{swap_time, species, pre, post}``
    with ``pre``/``post`` as sets of spec INDICES.
    """
    path = Path(run_dir) / "lockfix_swap_manifest.json"
    if not path.is_file():
        return {}
    try:
        with path.open() as fh:
            man = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return {}
    part = man.get("spec_partition_at_swap") or {}

    def _idx(names):
        out = set()
        for n in names or ():
            try:
                out.add(int(str(n).rsplit("_", 1)[-1]))
            except ValueError:
                continue
        return out

    # Three classes, not two. A spec that was mid-TRAINING at the swap loaded
    # its references at task start (old) but runs its EVAL afterwards, which
    # re-reads them (new) -- so its density metrics are computed against
    # references it did not train on. Confirmed on this run: specs 0020/0021
    # were in flight at the swap and their eval reports the RELOCKED CH error
    # (9.82e-03) while their training channel still shows the old frozen
    # signature. Those cells are MIXED and cannot be read on the density axis.
    pre = _idx(part.get("complete"))
    mixed = _idx(part.get("in_flight"))
    post = _idx(part.get("not_started"))
    if not (pre or mixed or post):
        return {}
    return {"swap_time": man.get("swap_time_local", "unknown"),
            "species": sorted((man.get("species") or {}).keys()),
            "pre": pre, "mixed": mixed, "post": post}


def lockfix_cell_classes(run_dir: Path, eval_subdir: str = "eval_holdout"
                         ) -> Dict[str, set]:
    """``{"relocked": {cell...}, "mixed": {cell...}}`` for figure glyphs.

    A cell is only classified when its TRAINING SET contains a relocked
    species -- every other cell saw byte-identical references either side of
    the swap and must carry no marker. ``relocked`` trained on the fixed
    references; ``mixed`` was mid-training at the swap (trained old, evaluated
    new) and is not interpretable on the density axis. Empty dicts for runs
    with no swap, so their figures are unmarked.
    """
    b = lockfix_boundary(run_dir)
    out: Dict[str, set] = {"relocked": set(), "mixed": set()}
    if not b:
        return out
    swapped = set(b["species"])
    cells = ccp._read_manifest_cells(Path(run_dir))
    for idx, spec_dir in ccp._spec_dirs(Path(run_dir)):
        if not (spec_dir / eval_subdir / "per_molecule.json").is_file():
            continue
        meta = spec_dir / "train_metadata.json"
        cell = cells.get(idx) or {}
        arch, ss = cell.get("arch"), cell.get("subset_size")
        if not (meta.is_file() and arch is not None and ss is not None):
            continue
        try:
            with meta.open() as fh:
                mols = set(json.load(fh).get("molecules", []))
        except (json.JSONDecodeError, OSError):
            continue
        if not (mols & swapped):
            continue
        key = (arch, int(ss))
        if idx in b.get("mixed", set()):
            out["mixed"].add(key)
        elif idx in b["post"]:
            out["relocked"].add(key)
    return out


def lockfix_note(run_dir: Path, eval_subdir: str = "eval_holdout") -> str:
    """Disclosure naming how the PLOTTED cells fall either side of a mid-run
    density-reference swap. Empty string when the run has no swap manifest, so
    runs without one are unchanged."""
    b = lockfix_boundary(run_dir)
    if not b:
        return ""
    # Only cells whose TRAINING SET contains a relocked species are affected:
    # for every other cell the references are byte-identical either side, so
    # they stay fully comparable across the boundary and must not be warned
    # about. (On this run CH enters at subset_size 6, so the whole ss<6 block
    # is unaffected regardless of which side it fell on.)
    swapped = set(b["species"])
    plotted = set()
    for idx, spec_dir in ccp._spec_dirs(Path(run_dir)):
        if not (spec_dir / eval_subdir / "per_molecule.json").is_file():
            continue
        meta = spec_dir / "train_metadata.json"
        if not meta.is_file():
            continue
        try:
            with meta.open() as fh:
                mols = set(json.load(fh).get("molecules", []))
        except (json.JSONDecodeError, OSError):
            continue
        if mols & swapped:
            plotted.add(idx)
    if not plotted:
        return (f"DENSITY-REFERENCE BOUNDARY: {'/'.join(b['species'])} "
                f"references were relocked mid-run ({b['swap_time']}), but no "
                "plotted cell trains on those species, so every cell here is "
                "unaffected and comparable across the boundary.")
    pre = sorted(plotted & b["pre"])
    mixed = sorted(plotted & b.get("mixed", set()))
    post = sorted(plotted & b["post"])
    species = "/".join(b["species"]) or "degenerate-radical"

    def _rng(ix):
        return (f"spec {ix[0]:04d}" if len(ix) == 1
                else f"spec {ix[0]:04d}-{ix[-1]:04d}")

    head = (f"DENSITY-REFERENCE BOUNDARY: {species} references were relocked "
            f"mid-run ({b['swap_time']}); only cells TRAINING on those species "
            "are affected (all others are comparable across it).")
    parts = []
    if pre:
        parts.append(f"{len(pre)} affected cell(s) pre-swap [{_rng(pre)}], OLD "
                     "unlocked references")
    if post:
        parts.append(f"{len(post)} post-swap [{_rng(post)}], relocked "
                     "references")
    if not parts:
        return head
    out = head + " " + "; ".join(parts) + "."
    if pre and post:
        out += " Density comparisons across the boundary mix two reference sets."
    if mixed:
        out += (f" {len(mixed)} cell(s) [{_rng(mixed)}] were mid-training at "
                "the swap: they TRAINED on the old references but their eval "
                "re-read the new ones, so their density numbers are NOT "
                "interpretable and are excluded from either side.")
    return out


# ---------------------------------------------------------------------------
# Live (non-hardcoded) footer baselines
# ---------------------------------------------------------------------------

# Above the ~2.5e-6 Ha cross-spec SCF noise, far below any physical artifact
# (the c2 grid-drift class disagrees by 6e-2 Ha).
_PBE_CONSISTENCY_TOL_HA = 1e-4


def _value_clusters(pairs, tol: float) -> List[Tuple[float, List[str]]]:
    """Cluster ``(label, value)`` pairs by value: sorted ascending, a new
    cluster opens when the gap to the current cluster's minimum exceeds
    ``tol`` (the same hi-lo criterion the consistency guards apply within a
    cluster). Returns ``[(cluster_value, sorted_labels)]`` ordered
    largest-cluster-first (ties by value), so ``[0]`` is the majority
    candidate when one exists; ``cluster_value`` is the cluster's smallest
    member -- an actually-reported value, never a synthetic mean."""
    pts = sorted((float(v), str(lbl)) for lbl, v in pairs if _is_num(v))
    clusters: List[Tuple[float, List[str]]] = []
    cur_vals: List[float] = []
    cur_lbls: List[str] = []
    for v, lbl in pts:
        if cur_vals and v - cur_vals[0] > tol:
            clusters.append((cur_vals[0], sorted(cur_lbls)))
            cur_vals, cur_lbls = [], []
        cur_vals.append(v)
        cur_lbls.append(lbl)
    if cur_vals:
        clusters.append((cur_vals[0], sorted(cur_lbls)))
    return sorted(clusters, key=lambda c: (-len(c[1]), c[0]))


def _outlier_clause(clusters: List[Tuple[float, List[str]]],
                    fmt: str = "{:.6f}") -> str:
    """Spec attribution for a flagged cross-spec disagreement. When one
    cluster holds a STRICT MAJORITY of all evals, the remaining evals are
    named individually -- they are the re-evaluation targets, the majority
    reference is intact. Otherwise every side is named: there is no
    majority reference to prefer. A single cluster (reachable only by
    callers outside the disagreement guards) reports plain agreement."""
    if not clusters:
        return ""
    major, rest = clusters[0], clusters[1:]
    if not rest:
        return f"{len(major[1])} specs agree @ {fmt.format(major[0])}"
    if len(major[1]) > sum(len(c[1]) for c in rest):
        minority = "; ".join(
            f"{', '.join(c[1])} @ {fmt.format(c[0])}" for c in rest)
        return (f"outlier eval(s): {minority} vs {len(major[1])} specs @ "
                f"{fmt.format(major[0])} -- re-evaluate the outlier spec(s)")
    sides = "; ".join(
        f"{', '.join(c[1])} @ {fmt.format(c[0])}" for c in clusters)
    return f"multi-spec split: {sides} -- no majority reference"


def _first_pbe_energies(run_dir: Path,
                        eval_subdir: str = "eval_holdout") -> Dict[str, float]:
    """PBE energy map (molecule -> Hartree) pooled across every spec carrying
    an ``<eval_subdir>/per_molecule.json``. PBE is invariant to the trained NN
    (SCF noise ~2.5e-6 Ha across specs), so the specs must agree; a species
    whose values spread beyond :data:`_PBE_CONSISTENCY_TOL_HA` indicates a
    drifted reference (the c2 grid-drift class -- on a merged multi-arm view
    the arms can disagree) and is EXCLUDED from the map with a printed
    warning naming the disagreeing spec(s), so the reference baselines skip
    its reactions on every leg instead of silently inheriting whichever spec
    sorts first. Each checkpoint channel (``eval_subdir``) pools its OWN
    per_molecule files: a retried eval can re-converge PBE to a different
    SCF solution in one channel only (observed: spec_0042's ``eval_holdout``
    c2), so the channels may exclude different molecules and the exclusion
    set is per-channel."""
    vals: Dict[str, List[Tuple[str, float]]] = {}
    for pm in sorted(Path(run_dir).glob(
            f"checkpoints/spec_*/{eval_subdir}/per_molecule.json")):
        spec = pm.parent.parent.name
        for r in json.loads(pm.read_text()):
            if isinstance(r.get("E_pbe"), (int, float)):
                vals.setdefault(r["molecule"], []).append(
                    (spec, float(r["E_pbe"])))
    out: Dict[str, float] = {}
    bad: Dict[str, Tuple[float, float, str]] = {}
    for m, vv in vals.items():
        nums = [v for _spec, v in vv]
        lo, hi = min(nums), max(nums)
        if hi - lo > _PBE_CONSISTENCY_TOL_HA:
            clause = _outlier_clause(
                _value_clusters(vv, _PBE_CONSISTENCY_TOL_HA))
            bad[m] = (lo, hi, clause)
        else:
            out[m] = nums[0]
    if bad:
        det = "; ".join(
            f"{m}: {lo:.6f}..{hi:.6f} Ha "
            f"({(hi - lo) * 627.5094740631:.2f} kcal/mol; {clause})"
            for m, (lo, hi, clause) in sorted(bad.items()))
        print("  WARNING: PBE reference disagreement across specs -- excluded "
              f"from the reference baselines: {det}")
    return out


def pbe_pool_baseline(run_dir: Path, *, eval_subdir: str = "eval_holdout",
                      _loader=None) -> Dict[str, Any]:
    """Full-pool PBE reaction-energy MAE (kcal/mol): ``{bh76, w411, combined}``
    plus a ``coverage`` map.

    The benchmark's inherent difficulty, independent of any train/held-out split
    (PBE does not depend on the trained NN), computed LIVE so the figure footers
    are never stale. Reuses the validated reaction math in
    ``xcquinox.alec.eval_holdout`` over the canonical pool from
    ``load_full_held_out_pools`` -- so it covers ALL 76 BH76 / 140 W4-11
    reactions, including the few that are in-sample in every spec and thus absent
    from any held-out file. ``coverage[pool] = {"used": n, "reference": m}``
    counts the reactions actually averaged against the UNRESTRICTED canonical
    leg size, so a consistency-guard exclusion (:func:`_first_pbe_energies`) or
    missing per_molecule energies shows as ``used < reference`` on the footers
    instead of silently shrinking the pool. Counts are canonical pool ENTRIES
    (76 + 140 = 216), the benchmarks' own accounting -- BH76 lists its
    symmetric identity reactions as two entries with identical signatures, so
    entry counts sit above distinct-signature counts by those pairs.
    ``_loader`` is a test seam (default: ``load_full_held_out_pools``)."""
    if _loader is None:
        from xcquinox.alec.full_benchmark_pools import load_full_held_out_pools
        _loader = load_full_held_out_pools
    from xcquinox.alec.eval_holdout import reaction_mae_kcalmol
    _, full_rxns = _loader()
    pbe = _first_pbe_energies(run_dir, eval_subdir=eval_subdir)
    out: Dict[str, Any] = {}
    cov: Dict[str, Dict[str, int]] = {}
    legs = [("bh76", [r for r in full_rxns if r.get("source_pool") == "bh76"]),
            ("w411", [r for r in full_rxns if r.get("source_pool") == "w411"]),
            ("combined", list(full_rxns))]
    for key, rx in legs:
        if rx:
            mae, n_used, _n_drop = reaction_mae_kcalmol(pbe, rx)
        else:
            mae, n_used = float("nan"), 0
        out[key] = mae
        cov[key] = {"used": int(n_used), "reference": len(rx)}
    out["coverage"] = cov
    return out


def _fmt_mae(x: Any) -> str:
    return (f"{x:.2f}" if isinstance(x, (int, float)) and math.isfinite(x)
            else "n/a")


def pool_line_suffix(baseline: Optional[Dict[str, Any]],
                     key: str = "combined") -> str:
    """``", u/r"`` when the baseline's ``key`` leg averaged fewer reactions
    than its unrestricted reference set (a consistency-guard exclusion or
    missing energies) -- the same label form :func:`scan_line_value` uses, so
    a reduced pooled line can never read as the full pool. Empty at full
    coverage and for legacy baselines carrying no ``coverage``."""
    used, ref = scan_coverage(baseline, key)
    return f", {used}/{ref}" if ref and used < ref else ""


def _pool_cov_bracket(baseline: Optional[Dict[str, Any]],
                      key: str = "combined") -> str:
    """`` [u/r reactions]`` for footer text when the ``key`` leg is reduced
    (same trigger as :func:`pool_line_suffix`), else ``""`` so full-coverage
    and legacy baselines render byte-identically to before."""
    used, ref = scan_coverage(baseline, key)
    return f" [{used}/{ref} reactions]" if ref and used < ref else ""


def provenance_footer(baseline: Dict[str, float],
                      scan_baseline: Optional[Dict[str, float]] = None) -> str:
    """Static methodology banner + the LIVE full-pool PBE baseline, and -- when a
    SCAN-energy cache is present (``scan_baseline`` carries a finite value) -- the
    parallel full-pool SCAN meta-GGA baseline. Absent SCAN -> the string is
    byte-identical to the PBE-only footer (backward compatible). A baseline
    whose combined leg averaged fewer reactions than the canonical pool (a
    consistency-guard exclusion) carries a ``[u/r reactions]`` bracket, so a
    reduced pool is visible on-figure; full coverage renders unchanged."""
    s = (_PROVENANCE_BASE + " PBE (full pool):"
         f" BH76 {_fmt_mae(baseline.get('bh76'))}"
         f" / W4-11 {_fmt_mae(baseline.get('w411'))}"
         f" / combined {_fmt_mae(baseline.get('combined'))}"
         f"{_pool_cov_bracket(baseline)}.")
    if scan_baseline and any(_is_num(scan_baseline.get(k))
                             for k in ("bh76", "w411", "combined")):
        s += (" SCAN (full pool):"
              f" BH76 {_fmt_mae(scan_baseline.get('bh76'))}"
              f" / W4-11 {_fmt_mae(scan_baseline.get('w411'))}"
              f" / combined {_fmt_mae(scan_baseline.get('combined'))}"
              f"{_pool_cov_bracket(scan_baseline)}.")
    return s


def nn_vs_pbe_caveat(reaction_rows: List[Dict[str, Any]],
                     baseline: Dict[str, float]) -> str:
    """Data-derived NN-vs-PBE headline for the parity figure: the live BH76 PBE
    baseline, the best NN arch/subset cell on BH76 barriers, and how many
    arch x subset cells beat PBE. Replaces the old hardcoded claim."""
    pbe_bh76 = baseline.get("bh76")
    cells: Dict[Tuple[str, int], List[float]] = {}
    for r in reaction_rows:
        if r.get("pool") == "bh76" and _is_num(r.get("abs_error_nn_kcalmol")):
            cells.setdefault((r.get("arch"), r.get("subset_size")), []).append(
                float(r["abs_error_nn_kcalmol"]))
    cell_mae = {k: sum(v) / len(v) for k, v in cells.items() if v}
    if not cell_mae or not _is_num(pbe_bh76):
        return "NN vs PBE on BH76 barriers: insufficient held-out data."
    (best_arch, best_ss), best = min(cell_mae.items(), key=lambda kv: kv[1])
    n_beat = sum(1 for m in cell_mae.values() if m < pbe_bh76)
    return (f"PBE BH76 baseline {pbe_bh76:.2f} kcal/mol; best NN "
            f"{best_arch}/subset-{best_ss} ({best:.2f} kcal/mol); "
            f"{n_beat}/{len(cell_mae)} arch x subset cell(s) beat PBE on barriers.")


def _scan_cache_name(basis: str) -> str:
    """Filename for the SCAN full-pool energy cache at ``basis``
    (``scan_pool_energies_<basis>.json`` written by ``precompute_scan_pool.py``).
    The ``+DF`` density-fit suffix is dropped -- SCAN reference energies span the
    same species set regardless -- and any path-unsafe char is mapped to ``_``."""
    b = (basis or "def2-svp").replace("+DF", "").strip() or "def2-svp"
    safe = "".join(c if (c.isalnum() or c in "-.+") else "_" for c in b)
    return f"scan_pool_energies_{safe}.json"


def _scan_energies(run_dir: Path, *, basis: Optional[str] = None,
                   cache_dir: Optional[Path] = None) -> Dict[str, float]:
    """Load the cached ``{molecule_name: E_scan_hartree}`` map written by
    ``precompute_scan_pool.py``. Searches ``cache_dir`` (default: ``run_dir``)
    for ``scan_pool_energies_<basis>.json`` (basis from ``run_basis_label`` when
    not given). Returns ``{}`` when the cache is absent, so the SCAN baseline
    degrades to all-NaN (figures then omit the SCAN line)."""
    try:
        basis = basis or run_basis_label(run_dir)
    except Exception:
        basis = basis or "def2-svp"
    fname = _scan_cache_name(basis)
    base = Path(cache_dir) if cache_dir else Path(run_dir)
    p = base / fname
    if not p.is_file():
        return {}
    try:
        raw = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return {k: float(v) for k, v in raw.items()
            if isinstance(v, (int, float)) and math.isfinite(v)}


# A SCAN reference drawn beside a PBE reference must average the SAME
# reactions/species, or the comparison is between two different benchmarks. SCAN
# can legitimately lose species (a meta-GGA SCF on a diffuse basis does not
# always converge), so coverage is measured against what PBE achieved and the
# line is qualified or withdrawn rather than silently drawn over a subset.
_SCAN_COVERAGE_FLOOR = 0.9


def _nan_baseline() -> Dict[str, Any]:
    """All-NaN baseline with empty coverage -- what every SCAN loader returns
    when its cache is absent, so the figures omit the line."""
    return {"bh76": float("nan"), "w411": float("nan"),
            "combined": float("nan"), "coverage": {}}


def scan_coverage(baseline: Optional[Dict[str, Any]],
                  key: str = "combined") -> Tuple[int, int]:
    """``(n_used, n_reference)`` for one leg of a pooled baseline -- how many
    reactions/species the leg averaged, against the unrestricted reference
    set. ``(0, 0)`` when the baseline carries no coverage (absent cache or a
    legacy dict)."""
    cov = ((baseline or {}).get("coverage") or {}).get(key) or {}
    return int(cov.get("used", 0)), int(cov.get("reference", 0))


def scan_line_value(baseline: Optional[Dict[str, Any]], key: str = "combined"
                    ) -> Tuple[Optional[float], str]:
    """``(value, suffix)`` for drawing one SCAN reference line, or
    ``(None, "")`` when it must not be drawn.

    The line is withdrawn when SCAN covers less than
    :data:`_SCAN_COVERAGE_FLOOR` of what PBE covers -- below that it is a
    different benchmark, not a reference. Between the floor and full coverage it
    is drawn with the covered count in its label, so a partial SCAN can never
    read as a like-for-like comparison."""
    val = (baseline or {}).get(key)
    if not _is_num(val):
        return None, ""
    used, ref = scan_coverage(baseline, key)
    if ref <= 0:
        return float(val), ""
    frac = used / ref
    if frac < _SCAN_COVERAGE_FLOOR:
        return None, ""
    return float(val), ("" if used >= ref else f", {used}/{ref}")


def scan_pool_baseline(run_dir: Path, *, basis: Optional[str] = None,
                       cache_dir: Optional[Path] = None, _loader=None,
                       _energies: Optional[Dict[str, float]] = None,
                       _pbe_energies: Optional[Dict[str, float]] = None,
                       eval_subdir: str = "eval_holdout") -> Dict[str, Any]:
    """Full-pool SCAN reaction-energy MAE (kcal/mol): ``{bh76, w411, combined}``
    plus a ``coverage`` map.

    The meta-GGA reference the ``_mgga`` archs clone, computed LIVE from a
    precomputed SCAN-energy cache (``scan_pool_energies_<basis>.json`` from
    ``precompute_scan_pool.py``) over the SAME canonical pool as
    :func:`pbe_pool_baseline`, so PBE and SCAN reference lines are directly
    comparable. Returns all-NaN (so figures OMIT the SCAN line) when the cache is
    absent -- older runs render exactly as before. MIRRORS
    :func:`pbe_pool_baseline`.

    ``coverage[pool] = {"used": n, "reference": m}`` records how many reactions
    SCAN averaged against the UNRESTRICTED canonical leg size (pool ENTRIES,
    matching :func:`pbe_pool_baseline`'s accounting), so ``used < reference``
    whenever SCAN misses a species OR a PBE consistency-guard exclusion
    dropped reactions from both legs -- either reduction is then visible in
    the ``, u/r`` label suffix and the coverage floor counts it (deliberate: a
    mass exclusion withdraws the SCAN line while the PBE line stays with its
    own ``[u/r]`` disclosure; a run in that regime carries dozens of
    reference warnings and neither pooled line is trustworthy).
    ``reaction_mae_kcalmol`` silently drops a reaction whose species energy is
    missing, so without this the two lines could average different reaction sets
    and still be drawn as a like-for-like pair. PBE's own numbers are NOT
    restricted to SCAN's coverage -- the PBE reference must not move because a
    SCAN species failed. ``_loader`` / ``_energies`` / ``_pbe_energies`` are test
    seams."""
    scan = (_energies if _energies is not None
            else _scan_energies(run_dir, basis=basis, cache_dir=cache_dir))
    if not scan:
        return _nan_baseline()   # no cache -> no SCAN line (no xcquinox import)
    if _loader is None:
        from xcquinox.alec.full_benchmark_pools import load_full_held_out_pools
        _loader = load_full_held_out_pools
    from xcquinox.alec.eval_holdout import reaction_mae_kcalmol
    _, full_rxns = _loader()
    pbe = (_pbe_energies if _pbe_energies is not None
           else _first_pbe_energies(run_dir, eval_subdir=eval_subdir))
    out: Dict[str, Any] = {}
    cov: Dict[str, Dict[str, int]] = {}
    legs = [("bh76", [r for r in full_rxns if r.get("source_pool") == "bh76"]),
            ("w411", [r for r in full_rxns if r.get("source_pool") == "w411"]),
            ("combined", list(full_rxns))]
    def _pbe_computable(rxns):
        # Keep the SCAN and PBE legs averaging the SAME reactions: a species
        # excluded from the PBE map (cross-spec disagreement in
        # _first_pbe_energies) drops its reactions from BOTH legs.
        return [r for r in rxns
                if all(n in pbe for n in (list(r.get("reactants", []))
                                          + list(r.get("products", []))))]
    for key, rx in legs:
        rx_eff = _pbe_computable(rx) if pbe else rx
        if not rx_eff:
            out[key] = float("nan")
            cov[key] = {"used": 0, "reference": len(rx)}
            continue
        mae, n_used, _n_drop = reaction_mae_kcalmol(scan, rx_eff)
        out[key] = mae
        cov[key] = {"used": int(n_used), "reference": len(rx)}
    out["coverage"] = cov
    return out


def _scan_density_cache_name(basis: str) -> str:
    """Filename for the SCAN density cache at ``basis``
    (``scan_pool_density_<basis>.json`` written by ``precompute_scan_pool.py``).
    Same slug rule as :func:`_scan_cache_name`, so one basis label resolves the
    energy and density caches together."""
    b = (basis or "def2-svp").replace("+DF", "").strip() or "def2-svp"
    safe = "".join(c if (c.isalnum() or c in "-.+") else "_" for c in b)
    return f"scan_pool_density_{safe}.json"


def _scan_density_records(run_dir: Path, *, basis: Optional[str] = None,
                          cache_dir: Optional[Path] = None
                          ) -> Dict[str, Dict[str, Any]]:
    """``{molecule: {density_rmse_scan, density_eps_l1_scan}}`` from the cache
    ``precompute_scan_pool.py --with-density`` writes. ``{}`` when absent, so the
    SCAN density baseline degrades to all-NaN and the figures omit the line."""
    try:
        basis = basis or run_basis_label(run_dir)
    except Exception:
        basis = basis or "def2-svp"
    base = Path(cache_dir) if cache_dir else Path(run_dir)
    p = base / _scan_density_cache_name(basis)
    if not p.is_file():
        return {}
    try:
        raw = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return {k: v for k, v in raw.items() if isinstance(v, dict)}


#: SCAN twin of each PBE density column, so one selector drives both legs.
_SCAN_DENSITY_KEY: Dict[str, str] = {
    "density_rmse_pbe": "density_rmse_scan",
    "density_l1_pbe": "density_l1_scan",
    "density_eps_l1_pbe": "density_eps_l1_scan",
}


def scan_density_mean(records: Optional[Dict[str, Dict[str, Any]]],
                      molecules, key: str = "density_rmse_pbe"
                      ) -> Tuple[float, int, int]:
    """``(mean, n_used, n_reference)`` of SCAN's density error over exactly
    ``molecules`` -- the species set the PBE anchor beside it averages.

    ONE rule for every SCAN density anchor on every panel, so a per-channel bar
    row and the pooled line can never be averaged over different species sets.
    ``(nan, 0, n)`` when the cache holds none of them."""
    scan_key = _SCAN_DENSITY_KEY.get(key)
    mols = sorted(molecules)
    if not records or scan_key is None:
        return float("nan"), 0, len(mols)
    vals = [records[m][scan_key] for m in mols
            if m in records and _is_num(records[m].get(scan_key))]
    return (float(np.mean(vals)) if vals else float("nan"),
            len(vals), len(mols))


def scan_density_line_counts(records: Optional[Dict[str, Dict[str, Any]]],
                             molecules, key: str = "density_rmse_pbe"
                             ) -> Tuple[Optional[float], int, int]:
    """``(value, n_used, n_reference)`` behind :func:`scan_density_line` --
    the same withdrawal rule, with the coverage counts kept so callers can
    qualify a drawn-but-partial line (``_scan_ed_suffix``)."""
    mean, used, ref = scan_density_mean(records, molecules, key=key)
    if not (_is_num(mean) and mean > 0.0) or ref <= 0:
        return None, used, ref
    return ((mean if (used / ref) >= _SCAN_COVERAGE_FLOOR else None),
            used, ref)


def scan_density_line(records: Optional[Dict[str, Dict[str, Any]]],
                      molecules, key: str = "density_rmse_pbe"
                      ) -> Optional[float]:
    """The SCAN density value to draw beside a PBE anchor over ``molecules``,
    or ``None`` when it must not be drawn (absent cache, or coverage below
    :data:`_SCAN_COVERAGE_FLOOR` of the PBE anchor's species)."""
    return scan_density_line_counts(records, molecules, key=key)[0]


def _scan_ed_suffix(e_used: int, e_ref: int, d_used: int, d_ref: int) -> str:
    """Label suffix qualifying a drawn-but-partially-covered SCAN ED
    comparator, mirroring :func:`scan_line_value`'s ``", used/ref"``
    convention leg-by-leg: ``""`` when both legs cover their full reference
    sets, else the partial legs as ``", E 5/6 D 3/4"`` (E = reactions of the
    energy leg, D = species of the density leg)."""
    parts = []
    if e_ref > 0 and 0 < e_used < e_ref:
        parts.append(f"E {e_used}/{e_ref}")
    if d_ref > 0 and 0 < d_used < d_ref:
        parts.append(f"D {d_used}/{d_ref}")
    return (", " + " ".join(parts)) if parts else ""


def scan_density_baseline(hd_rows: List[Dict[str, Any]], run_dir: Path, *,
                          pbe_table: Optional[Dict[str, Dict[str, float]]] = None,
                          key: str = "density_rmse_pbe",
                          basis: Optional[str] = None,
                          cache_dir: Optional[Path] = None,
                          _records: Optional[Dict[str, Dict[str, Any]]] = None
                          ) -> Dict[str, Any]:
    """Pooled SCAN density-vs-CCSD anchor ``D_SCAN``, the twin of
    :func:`pbe_density_baseline`: ``{"value": float, "coverage": {...}}``.

    Averaged over EXACTLY the molecules :func:`pbe_density_baseline` averages
    for the same ``key`` -- taking SCAN's own mean over whatever the cache holds
    would put two different species sets on one axis. ``key`` selects the PBE
    column (grid-weighted RMSE by default, ``density_eps_l1_pbe`` for the DFS
    Eq. 20 per-electron anchor) and :data:`_SCAN_DENSITY_KEY` maps it to SCAN's.
    ``coverage["value"]`` counts how many of those molecules the SCAN cache
    carries, so :func:`scan_line_value` can qualify or withdraw the line.
    All-NaN when the cache is absent, so figures render unchanged without it.
    ``_records`` is a test seam."""
    recs = (_records if _records is not None
            else _scan_density_records(run_dir, basis=basis,
                                       cache_dir=cache_dir))
    if not recs:
        return {"value": float("nan"), "coverage": {}}
    # The molecule set the PBE anchor averages -- same helper, same dedup.
    pbe_mol = _pbe_density_map(hd_rows, pbe_table, key=key)
    mean, used, ref = scan_density_mean(recs, pbe_mol, key=key)
    return {"value": mean,
            "coverage": {"value": {"used": used, "reference": ref}}}


def _report_scan_density(scan_density: Optional[Dict[str, Any]]) -> None:
    """Print the SCAN density anchor + its species coverage when it resolves
    (the density half of ``_report_scan_coverage``); silent otherwise."""
    if scan_density is not None and _is_num(scan_density.get("value")):
        d_used, d_ref = scan_coverage(scan_density, "value")
        print(f"  SCAN density vs CCSD: {scan_density['value']:.3e} "
              f"[{d_used}/{d_ref} species]")


def _report_scan_coverage(scan_baseline: Optional[Dict[str, Any]],
                          scan_density: Optional[Dict[str, Any]] = None) -> None:
    """Print the SCAN baseline + its coverage, or say the cache is absent.

    A silent omission is how the SCAN line went unnoticed for so long: the
    loader degrades to all-NaN and the figures simply drop the line, so the
    console never mentions SCAN at all. This makes both states explicit.
    ``build_all`` owns the energy-side report; ``build_density_energy_figures``
    reports only the density anchor (``_report_scan_density``), so a suite
    pass prints each state exactly once."""
    e_used, e_ref = scan_coverage(scan_baseline, "combined")
    if not _is_num((scan_baseline or {}).get("combined")):
        print("  (no SCAN cache next to the run -- SCAN reference lines omitted; "
              "generate with notebooks/analysis/precompute_scan_pool.py)")
        return
    print(f"  SCAN baseline (full pool): "
          f"BH76 {_fmt_mae((scan_baseline or {}).get('bh76'))} / "
          f"W4-11 {_fmt_mae((scan_baseline or {}).get('w411'))} / "
          f"combined {_fmt_mae((scan_baseline or {}).get('combined'))} "
          f"[{e_used}/{e_ref} reactions]")
    if scan_density is not None and _is_num(scan_density.get("value")):
        d_used, d_ref = scan_coverage(scan_density, "value")
        print(f"  SCAN density vs CCSD: {scan_density['value']:.3e} "
              f"[{d_used}/{d_ref} species]")


def arch_reference_kinds(archs) -> Dict[str, str]:
    """``"pbe"`` for pure-GGA architectures, ``"scan"`` for any architecture
    carrying beyond-GGA information (meta-GGA, rung-3.5, and their stacks).

    The green beats marker claims improvement over the arch's OWN-RUNG
    nonempirical reference; crediting a beyond-GGA architecture for merely
    beating PBE overstates it. The rung-3.5 families have no same-rung
    nonempirical reference (nonlocal DM information but no tau, so neither
    PBE's nor SCAN's input set contains theirs); they are held to SCAN, the
    conservative assignment."""
    return {a: ("pbe" if arch_style.rung_of(a) == arch_style.RUNG_GGA
                else "scan") for a in archs}


# The cell-rows comparator mark: ONE capped horizontal span per subset-size
# group (error-bar style: a horizontal segment with small vertical end caps
# demarking the group's extent) at the group's cell-slice value -- the
# group's cells share one test slice and the comparator reduces it
# independent of NN convergence, so their anchors agree to fp noise; a
# disagreeing group falls back to per-bar spans. Every figure drawing it
# must carry _CELL_ROWS_GLYPH_NOTE.
_CELL_ROWS_GLYPH_NOTE = (
    "Capped horizontal spans across each subset-size group mark the "
    "group's cell-slice reference value (black = PBE, grey = SCAN, "
    "reduced over the cell's full test slice -- every slice reaction "
    "with a finite comparator leg, independent of NN convergence); a "
    "group whose cells disagree (a degraded comparator) shows per-bar "
    "spans instead. Starred bars: the NN scored fewer reactions than "
    "the slice (named in the note band). The dash-dot (PBE) / dotted "
    "(SCAN) lines are the pooled-set reductions.")

# A subset-size group's cells share one test slice, and the comparator
# anchors reduce that slice independent of NN convergence, so present
# cells agree to fp summation noise (~1e-15 relative); a genuinely
# different slice (a degraded comparator leg: a NaN or drifted E_pbe)
# moves the anchor by >= 1e-3 relative. 1e-6 separates the two regimes
# by three decades each way.
_GROUP_ANCHOR_REL_TOL = 1e-6


def _group_span_points(by_cell: Optional[Dict[Tuple[str, int], float]],
                       archs: List[str], subsets: List[int], bw: float
                       ) -> Tuple[List[float], List[float], List[float]]:
    """Cell-rows span geometry: ONE capped span per subset-size group --
    centered on the group (x = the subset's tick position) with half-width
    covering the group's bar cluster -- at the group's shared anchor, when
    every present cell agrees within :data:`_GROUP_ANCHOR_REL_TOL`
    (relative); the drawn value is the first present cell's own value,
    never a synthetic mean. A disagreeing group falls back to per-bar
    spans (bar-width, slightly shrunk so neighbours stay separable), so a
    divergent anchor stays visible instead of being averaged away.
    ``(xs, ys, half_widths)`` for ``ax.errorbar(..., xerr=half_widths)``."""
    xs_out: List[float] = []
    ys_out: List[float] = []
    hw_out: List[float] = []
    if by_cell is None:
        return xs_out, ys_out, hw_out
    group_half = max(1, len(archs)) * bw / 2.0
    for i, s in enumerate(subsets):
        vals = [(j, float(by_cell[(a, s)]))
                for j, a in enumerate(archs)
                if _is_num(by_cell.get((a, s)))]
        if not vals:
            continue
        vv = [v for _j, v in vals]
        lo, hi = min(vv), max(vv)
        scale = max(abs(lo), abs(hi))
        if scale == 0.0 or (hi - lo) / scale <= _GROUP_ANCHOR_REL_TOL:
            xs_out.append(float(i))
            ys_out.append(vv[0])
            hw_out.append(group_half)
        else:
            for j, v in vals:
                xs_out.append(i + (j - (len(archs) - 1) / 2) * bw)
                ys_out.append(v)
                hw_out.append(0.45 * bw)
    return xs_out, ys_out, hw_out


def _cell_anchor_note(by_cell: Optional[Dict[Tuple[str, int], float]],
                      unit: str = "kcal/mol", glyphs: bool = True) -> str:
    """Disclosure line for panels whose beats marks are judged against
    cell-slice anchors: names the convention, the PBE anchors' range, and
    -- on figures that actually draw the capped spans -- the mark's key.
    ``glyphs=False`` is for the ED line/scatter figures, whose verdicts are
    cell-matched but which draw no bars: they point at the CSV columns
    instead of describing glyphs they do not carry. Empty when no cell
    anchor resolved (marks then fell back to the pooled lines)."""
    vals = [float(v) for v in (by_cell or {}).values() if _is_num(v)]
    if not vals:
        return ""
    head = ("beats marks: each bar against its own-rung reference's "
            "cell-slice anchor (PBE for GGA architectures, SCAN for "
            "meta-GGA/rung-3.5; PBE anchors "
            f"{min(vals):.3g}-{max(vals):.3g} {unit}). ")
    if glyphs:
        return head + _CELL_ROWS_GLYPH_NOTE
    return (head + "Cell-slice anchor values are recorded in the CSV "
            "(ED_pbe_cell/ED_scan_cell columns); the dashed/dotted lines "
            "are the pooled-set reductions.")


def _beats_pbe_marks(xs, heights, pbe_line) -> List[Tuple[float, float]]:
    """``(x, height)`` for every bar whose height is strictly BELOW ``pbe_line``
    -- the positions to stamp a beats-PBE marker. Empty when ``pbe_line`` is not
    finite or nothing beats it. Pure + NaN-safe so both bar figures share one
    tested rule for "this cell beats PBE"."""
    if not _is_num(pbe_line):
        return []
    return [(float(x), float(h)) for x, h in zip(xs, heights)
            if _is_num(h) and h < pbe_line]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _mae(values: List[float]) -> Optional[float]:
    vals = [abs(v) for v in values if _is_num(v)]
    return float(np.mean(vals)) if vals else None


def reaction_mae_by_arch_subset(
    rows: List[Dict[str, Any]], *, key: str = "abs_error_nn_kcalmol",
) -> Dict[Tuple[str, int], float]:
    """``{(arch, subset_size): MAE}`` over held-out reactions for ``key``
    (``abs_error_nn_kcalmol`` or ``abs_error_pbe_kcalmol``). Deduplicated by
    reaction name WITHIN each cell (first FINITE value per key wins) -- the
    canonical pool lists four reactions twice under one name, and the PBE
    baselines dedup, so an un-deduped cell metric would double-count those
    rows against a deduped reference line. Finiteness is tested BEFORE the
    name slot is consumed (matching ``_cell_counts``): a NaN first instance
    of a duplicated name cannot discard its finite twin."""
    buckets: Dict[Tuple[str, int], List[float]] = {}
    seen: set = set()
    for r in rows:
        arch, ss = r.get("arch"), r.get("subset_size")
        if arch is None or ss is None or not _is_num(r.get(key)):
            continue
        nm = r.get("name")
        if nm is not None:
            if (arch, ss, nm) in seen:
                continue
            seen.add((arch, ss, nm))
        buckets.setdefault((arch, ss), []).append(r[key])
    return {k: float(np.mean(v)) for k, v in buckets.items() if v}


def ae_mae_by_arch_subset(
    rows: List[Dict[str, Any]],
) -> Dict[Tuple[str, int], float]:
    """``{(arch, subset_size): MAE}`` over in-sample |AE_error_kcalmol|."""
    buckets: Dict[Tuple[str, int], List[float]] = {}
    for r in rows:
        arch, ss = r.get("arch"), r.get("subset_size")
        if arch is None or ss is None:
            continue
        buckets.setdefault((arch, ss), []).append(r["AE_error_kcalmol"])
    return {k: m for k, v in buckets.items() if (m := _mae(v)) is not None}


def _archs_present(rows: List[Dict[str, Any]]) -> List[str]:
    present = {r.get("arch") for r in rows if r.get("arch")}
    ordered = [a for a in ARCH_ORDER if a in present]
    # Append any unexpected arch names (defensive) in sorted order.
    ordered += sorted(present - set(ordered))
    return ordered


def _best_subset_per_arch(rows: List[Dict[str, Any]]) -> Dict[str, int]:
    """For each arch, the LARGEST subset_size that has held-out reactions --
    the most-trained representative used for the parity scatter. Principled
    and stated on the figure (no MAE-based cherry-picking)."""
    by_arch: Dict[str, int] = {}
    for r in rows:
        arch, ss = r.get("arch"), r.get("subset_size")
        if arch is None or ss is None:
            continue
        if arch not in by_arch or ss > by_arch[arch]:
            by_arch[arch] = ss
    return by_arch


# ---------------------------------------------------------------------------
# Unequal training-depth coverage.
#
# A partial sweep fills the arch x subset_size grid column by column, so an arch
# that entered late carries only its smallest subsets while the early archs have
# reached the full pool. Figures that aggregate OVER subset_size (the per-arch
# bars, the Jacob's-ladder rung summary) then place a shallowly-trained arch next
# to a fully-trained one, and the difference reads as an architecture result when
# it is a training-set-size result. These helpers identify that mismatch so the
# affected bars can be marked on the figure itself.
# ---------------------------------------------------------------------------

# Visual channel for the mark, one definition shared by both figures. The
# per-arch bars distinguish their series by COLOR and the rungs by background
# band, leaving hatch free; the rung summary already spends hatch on the
# BH76/W4-11 split and face color on rung identity, so it fades the face and
# switches the edge instead.
_SHALLOW_HATCH = "x"
_SHALLOW_EDGE = "#d62728"
# Thin the hatch stroke where the mark is drawn: at the default width an "x"
# fill swamps the bar's series color and reads as "no data" rather than
# "shallower depth". Applied per-figure, never globally.
_SHALLOW_HATCH_RC = {"hatch.linewidth": 0.5}

# ---------------------------------------------------------------------------
# V_xc-consistency provenance. The 2026-08-06 correction added the
# feature-response term sum_g w_g (de/df)_g . df_g/dP to V_xc for
# DM-dependent descriptors; cells trained before it used a potential that was
# not the exact functional derivative of the energy for those descriptors.
# Grid-local architectures (GGA + cusp) never had the term, so their cells
# are unaffected. Production measurement (HISTORY 2026-08-10): the
# converged-energy effect is <= 0.34 kcal/mol on stable configurations, so
# pre-correction ENERGIES remain interpretable, but the meta-GGA SCF
# oscillates under the corrected potential on tail-dominated species -- so
# those cells' re-runs wait on SCF stabilization, while rung-3.5 cells
# re-run safely. Runs started after the fix date carry corrected code and
# draw no marks.
# ---------------------------------------------------------------------------
_VXC_FIX_DATE = "20260806"
_VXC_PRE_GATED = ("deep_mgga_3x16", "deep_mgga_attn_3x16")
_VXC_PRE_READY = ("deep_rung35_3x16", "deep_rung35_attn_3x16",
                  "deep_rung35_mgga_3x16")
_VXC_HATCH_GATED = "\\\\"
_VXC_HATCH_READY = ".."
_VXC_DISCLOSURE = (
    "V_xc PROVENANCE: this run predates the 2026-08-06 feature-response "
    "correction; hatched architectures trained with a potential that was not "
    "the exact functional derivative for their DM-dependent descriptors "
    "(measured converged-energy effect <= 0.34 kcal/mol on stable "
    "configurations). '\\\\' meta-GGA: re-run gated on SCF stabilization; "
    "'..' rung-3.5: safe to re-run corrected. GGA/cusp architectures "
    "unaffected.")


def _run_predates_vxc_fix(run_id: str) -> bool:
    """True when the run's encoded start date predates the V_xc correction.

    Run directories encode their start as ``run_YYYYMMDDTHHMMSSZ``; an id
    without that stamp is conservatively treated as post-fix (no marks) so
    synthetic/test ids do not acquire provenance hatching."""
    import re as _re
    m = _re.search(r"run_(\d{8})T", str(run_id))
    return bool(m) and m.group(1) < _VXC_FIX_DATE


def _vxc_hatch(arch: str) -> Optional[str]:
    if arch in _VXC_PRE_GATED:
        return _VXC_HATCH_GATED
    if arch in _VXC_PRE_READY:
        return _VXC_HATCH_READY
    return None


def _subset_coverage(rows: List[Dict[str, Any]]) -> Dict[str, Tuple[int, int]]:
    """``{arch: (smallest, largest)}`` subset_size that actually carries rows --
    the training depth each arch reached in this run."""
    cov: Dict[str, Tuple[int, int]] = {}
    for r in rows:
        arch, ss = r.get("arch"), r.get("subset_size")
        if arch is None or ss is None:
            continue
        ss = int(ss)
        lo, hi = cov.get(arch, (ss, ss))
        cov[arch] = (min(lo, ss), max(hi, ss))
    return cov


def _coverage_span(bounds: Tuple[int, int]) -> str:
    """``(1, 26) -> "1-26"``, ``(1, 1) -> "1"`` -- compact axis/annotation tag."""
    lo, hi = bounds
    return str(lo) if lo == hi else f"{lo}-{hi}"


def _shallow_archs(rows: List[Dict[str, Any]]) -> Tuple[Set[str], int]:
    """``(archs trained less deeply than the run's deepest, that depth)``.

    An arch is SHALLOW when its largest subset_size falls below the largest any
    arch reached: its bars aggregate a smaller training set, so a difference
    against an unmarked arch confounds architecture with training depth. Returns
    an empty set when every arch reached the same depth, so a complete grid
    renders exactly as it did before the mark existed.
    """
    cov = _subset_coverage(rows)
    if not cov:
        return set(), 0
    deepest = max(hi for _lo, hi in cov.values())
    return {a for a, (_lo, hi) in cov.items() if hi < deepest}, deepest


def _shallow_rungs(rows: List[Dict[str, Any]]) -> Tuple[Set[str], int]:
    """``(rungs whose DEEPEST arch is shallower than the run's, that depth)``.

    Judged on the rung's best-covered arch, not its mean: a rung containing one
    fully-trained arch has been probed at full depth and must not be marked just
    because a sibling lags. Empty set when every rung reached the run's depth.
    """
    cov = _subset_coverage(rows)
    if not cov:
        return set(), 0
    deepest = max(hi for _lo, hi in cov.values())
    out: Set[str] = set()
    for rung, archs in arch_style.by_rung(_archs_present(rows)).items():
        his = [cov[a][1] for a in archs if a in cov]
        if his and max(his) < deepest:
            out.add(rung)
    return out, deepest


def _rung_coverage(rows: List[Dict[str, Any]]) -> Dict[str, Tuple[int, int]]:
    """``{rung: (smallest, largest)}`` over the BEST subset_size of each of the
    rung's archs -- the depths the rung summary's bars actually aggregate."""
    best = _best_subset_per_arch(rows)
    out: Dict[str, Tuple[int, int]] = {}
    for rung, archs in arch_style.by_rung(_archs_present(rows)).items():
        ss = [best[a] for a in archs if a in best]
        if ss:
            out[rung] = (min(ss), max(ss))
    return out


def _coverage_caveat(rows: List[Dict[str, Any]], by_rung: bool = False) -> str:
    """Disclosure naming the architectures (or Jacob's-ladder rungs) whose bars
    aggregate a shallower training depth than the run's deepest, so an unequal-
    coverage gap is never read as an architecture result. Empty string when
    coverage is level -- complete grids keep their original footer."""
    if by_rung:
        shallow, deepest = _shallow_rungs(rows)
        cov = _rung_coverage(rows)
        keys = [r for r in arch_style.RUNG_ORDER if r in shallow]
        what = "rung"
    else:
        shallow, deepest = _shallow_archs(rows)
        cov = _subset_coverage(rows)
        keys = [a for a in _archs_present(rows) if a in shallow]
        what = "architecture"
    parts = [f"{k} at subset_size {_coverage_span(cov[k])}"
             for k in keys if k in cov]
    if not parts:
        return ""
    return ("UNEQUAL TRAINING DEPTH (marked on the bars): " + "; ".join(parts)
            + f", against {deepest} for the deepest. A difference involving a "
            f"marked {what} confounds it with training-set size.")


def _fade(color: Any, frac: float = 0.55) -> Tuple[float, float, float]:
    """``color`` blended ``frac`` of the way to white. Keeps the hue -- so a
    faded bar is still identifiable as its rung -- while reading as provisional.
    """
    r, g, b = matplotlib.colors.to_rgb(color)
    f = min(max(float(frac), 0.0), 1.0)
    return (r + (1.0 - r) * f, g + (1.0 - g) * f, b + (1.0 - b) * f)


# ---------------------------------------------------------------------------
# Figure A -- Fig-5-style parity
# ---------------------------------------------------------------------------

def _draw_mae_inset(ax, mae_by_arch: Dict[str, float], archs: List[str], *,
                    title: str, baseline: Optional[float] = None,
                    baseline_label: str = "PBE") -> None:
    """Per-arch MAE bar inset (lower-right), color-matched to the scatter --
    the analog of the paper's Fig-5 mean-relative-error inset."""
    # Low-right, but lifted just enough that the angled (40°) tick labels clear
    # the outer panel's x-axis -- low enough that the inset body stays under the
    # y=x diagonal (which passes through ~axes-fraction 0.56 at the inset's left
    # edge).
    inset = ax.inset_axes([0.56, 0.13, 0.41, 0.33])
    xs = np.arange(len(archs))
    heights = [mae_by_arch.get(a, np.nan) for a in archs]
    inset.bar(xs, heights, color=[ARCH_COLOR[a] for a in archs],
              edgecolor="k", linewidth=0.3)
    if baseline is not None and math.isfinite(baseline):
        inset.axhline(baseline, ls="--", color="k", linewidth=1.0,
                      label=baseline_label)
        inset.legend(fontsize=5, loc="upper left", framealpha=0.6)
    inset.set_xticks(xs)
    _short = {"deep": "base", "deep_attn": "attn", "deep_cusp": "cusp",
              "deep_dm": "dm", "deep_combined": "comb",
              "deep_combined_attn": "comb_at",
              "deep_notransform": "notr", "deep_notransform_attn": "notr_at"}
    inset.set_xticklabels(
        [_short.get(a, a.replace("deep_", "").replace("deep", "base") or "base")
         for a in archs],
        rotation=40, ha="right", rotation_mode="anchor", fontsize=5)
    inset.tick_params(axis="y", labelsize=5)
    inset.set_title(title, fontsize=6)
    inset.set_ylabel("MAE", fontsize=5)
    inset.grid(True, axis="y", alpha=0.3)


def _robust_limits(vals: List[float], q: Tuple[float, float] = (1.0, 99.0),
                   pad: float = 0.08) -> Optional[Tuple[float, float]]:
    """Symmetric-ish [lo, hi] window from percentiles ``q`` of finite vals,
    padded. Returns None when there is nothing finite to bound."""
    finite = np.array([v for v in vals if _is_num(v)], dtype=float)
    if finite.size == 0:
        return None
    lo, hi = np.percentile(finite, q)
    span = (hi - lo) or 1.0
    return float(lo - pad * span), float(hi + pad * span)


def _diagonal(ax, xs: List[float], ys: List[float],
              limits: Optional[Tuple[float, float]] = None) -> int:
    """Draw the y=x line and set equal axis limits. If ``limits`` is given,
    clamp to that window and return the number of (x, y) points falling
    outside it (so the caller can annotate clipped outliers); otherwise use
    the full finite range and return 0."""
    finite = [v for v in (xs + ys) if _is_num(v)]
    if not finite:
        return 0
    if limits is None:
        lo, hi = min(finite), max(finite)
        pad = 0.05 * (hi - lo or 1.0)
        line = [lo - pad, hi + pad]
        n_out = 0
    else:
        line = list(limits)
        n_out = sum(1 for x, y in zip(xs, ys)
                    if _is_num(x) and _is_num(y)
                    and (not line[0] <= x <= line[1]
                         or not line[0] <= y <= line[1]))
    ax.plot(line, line, color="k", ls="-", linewidth=1.0, zorder=1,
            label="y = x (perfect)")
    ax.set_xlim(line)
    ax.set_ylim(line)
    return n_out


def plot_parity(rows: List[Dict[str, Any]], out_path: Path, run_id: str,
                note: str = "", provenance: Optional[str] = None,
                caveat: Optional[str] = None) -> Path:
    """Figure A -- two-panel parity, points colored by arch, y=x diagonal,
    per-arch MAE inset. Each arch contributes its most-trained (largest
    subset_size) spec's held-out reactions."""
    with plt.rc_context(_STYLE):
        archs = _archs_present(rows)
        best = _best_subset_per_arch(rows)
        # Restrict scatter to each arch's representative spec.
        sel = [r for r in rows
               if r.get("arch") in best
               and r.get("subset_size") == best[r["arch"]]]

        fig, (axa, axb) = plt.subplots(1, 2, figsize=(13, 7.4))

        # Panel (a): optimized NN vs PBE -- how far subset training moved the
        # network from its PBE starting point (the PBE "clone" is the PRETRAIN;
        # these are the post-pretrain, subset-OPTIMIZED networks). ------------
        xs_a, ys_a = [], []
        for arch in archs:
            for pool, marker in POOL_MARKER.items():
                pts = [(r["de_pbe_kcalmol"], r["de_nn_kcalmol"]) for r in sel
                       if r.get("arch") == arch and r.get("pool") == pool
                       and _is_num(r.get("de_pbe_kcalmol"))
                       and _is_num(r.get("de_nn_kcalmol"))]
                if not pts:
                    continue
                xx, yy = zip(*pts)
                xs_a += list(xx); ys_a += list(yy)
                axa.scatter(xx, yy, s=14, marker=marker, alpha=0.55,
                            color=ARCH_COLOR[arch], edgecolor="none", zorder=3)
        _diagonal(axa, xs_a, ys_a)
        axa.set_xlabel("PBE reaction energy  de_pbe  (kcal/mol)")
        axa.set_ylabel("NN reaction energy  de_nn  (kcal/mol)")
        axa.set_title("(a) optimized NN vs PBE reaction energy")
        mae_nn_vs_pbe = {
            a: m for a in archs
            if (m := _mae([r["de_nn_kcalmol"] - r["de_pbe_kcalmol"]
                           for r in sel if r.get("arch") == a
                           and _is_num(r.get("de_nn_kcalmol"))
                           and _is_num(r.get("de_pbe_kcalmol"))])) is not None
        }
        _draw_mae_inset(axa, mae_nn_vs_pbe, archs,
                        title="per-arch |NN−PBE| MAE")

        # Panel (b): NN & PBE vs benchmark reference -----------------------
        xs_b, ys_b = [], []
        for arch in archs:
            pts = [(r["ref_kcalmol"], r["de_nn_kcalmol"]) for r in sel
                   if r.get("arch") == arch and _is_num(r.get("ref_kcalmol"))
                   and _is_num(r.get("de_nn_kcalmol"))]
            if not pts:
                continue
            xx, yy = zip(*pts)
            xs_b += list(xx); ys_b += list(yy)
            axb.scatter(xx, yy, s=14, alpha=0.55, color=ARCH_COLOR[arch],
                        edgecolor="none", zorder=3, label=arch)
        # PBE-vs-ref as a single grey baseline series (same for every arch).
        pbe_pts = [(r["ref_kcalmol"], r["de_pbe_kcalmol"]) for r in sel
                   if _is_num(r.get("ref_kcalmol"))
                   and _is_num(r.get("de_pbe_kcalmol"))]
        if pbe_pts:
            xx, yy = zip(*pbe_pts)
            xs_b += list(xx); ys_b += list(yy)
            axb.scatter(xx, yy, s=10, marker="x", alpha=0.35, color="0.4",
                        zorder=2, label="PBE")
        # Robust window: catastrophic outlier predictions (down to ~-7000)
        # otherwise compress the diagonal. Clip to the 1-99 pct window of the
        # predicted values and annotate how many points fall outside.
        limits_b = _robust_limits(ys_b + xs_b, q=(1.0, 99.0))
        n_out = _diagonal(axb, xs_b, ys_b, limits=limits_b)
        if n_out:
            axb.text(0.02, 0.97, f"{n_out} point(s) beyond axis",
                     transform=axb.transAxes, fontsize=6.5, va="top",
                     color="#a33")
        axb.set_xlabel("Benchmark reference reaction energy  (kcal/mol)")
        axb.set_ylabel("Predicted reaction energy  (kcal/mol)")
        axb.set_title("(b) NN & PBE vs benchmark reference")
        mae_nn_vs_ref = {a: m for a in archs
                         if (m := _mae([r["abs_error_nn_kcalmol"] for r in sel
                                        if r.get("arch") == a])) is not None}
        pbe_vs_ref = _mae([r["abs_error_pbe_kcalmol"] for r in sel])
        _draw_mae_inset(axb, mae_nn_vs_ref, archs,
                        title="per-arch NN-vs-ref MAE", baseline=pbe_vs_ref)

        # Shared arch legend below the panels.
        # Each arch's representative is its own deepest spec, and archs enter
        # the sweep at different depths -- so the cloud can mix a 1-molecule
        # net with a full-pool one. The legend states which depth each is.
        handles = [Patch(facecolor=ARCH_COLOR[a],
                         label=f"{a} (ss {best[a]})" if a in best else a)
                   for a in archs]
        handles.append(plt.Line2D([], [], marker="o", ls="", color="0.4",
                                   label="bh76 (●) / w411 (▲) by marker"))
        # Shared arch legend in its own reserved band below the panels -- the
        # bottom strip is stacked (legend > note > provenance) with no overlap.
        fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=7,
                   frameon=False, bbox_to_anchor=(0.5, 0.085))

        fig.suptitle(
            "Reaction-energy parity (Fig-5 analog) -- "
            f"each arch at its largest available subset_size · {run_id}",
            fontsize=11, y=0.985)
        if caveat:
            fig.text(0.5, 0.925, caveat, ha="center", fontsize=7.5,
                     style="italic", color="#444444")
        if note:
            fig.text(0.5, 0.05, note, ha="center", fontsize=6.5,
                     color="#a33", wrap=True)
        fig.text(0.5, 0.016, provenance or _PROVENANCE_BASE, ha="center",
                 fontsize=6, color="#777777")
        fig.tight_layout(rect=(0, 0.155, 1, 0.90))
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Figure B -- arch × subset_size heatmaps
# ---------------------------------------------------------------------------

def _heatmap_panel(ax, mae_map: Dict[Tuple[str, int], float], archs: List[str],
                   *, title: str, cbar_label: str,
                   center: Optional[float] = None,
                   subset_sizes: Optional[Sequence[int]] = None,
                   rung_separators: bool = False,
                   vxc_pre_fix: bool = False) -> None:
    """arch x subset_size heatmap. Default: log-scaled viridis (raw MAE spanning
    decades). With ``center`` set (e.g. 1.0 for a MAE/PBE ratio): a diverging
    RdBu_r map about ``center`` -- below center is blue (better than the
    reference), above is red (worse). Missing cells are hatched either way.
    ``subset_sizes`` overrides the column axis (default the global SUBSET_SIZES);
    pass the present sizes to drop empty trailing columns. ``rung_separators``
    (for a rung-ordered ``archs`` axis) draws a colored rung gutter to the left
    plus thin lines between rung groups -- the heatmap analog of
    :func:`plot_mae_by_arch`'s rung bands."""
    ss_axis = list(subset_sizes) if subset_sizes is not None else list(SUBSET_SIZES)
    n_a, n_s = len(archs), len(ss_axis)
    grid = np.full((n_a, n_s), np.nan)
    for i, a in enumerate(archs):
        for j, ss in enumerate(ss_axis):
            v = mae_map.get((a, ss))
            if v is not None and math.isfinite(v):
                grid[i, j] = v
    finite = grid[np.isfinite(grid)]
    if center is not None and finite.size:
        # diverging about `center` (TwoSlopeNorm needs vmin < vcenter < vmax)
        vmin = min(float(finite.min()), center * 0.999)
        vmax = max(float(finite.max()), center * 1.001)
        norm = matplotlib.colors.TwoSlopeNorm(vcenter=center, vmin=vmin, vmax=vmax)
        cmap = plt.get_cmap("RdBu_r").copy()
        fmt = "{:.2f}"
    elif finite.size:
        # log color scale (MAE spans decades)
        norm = matplotlib.colors.LogNorm(vmin=max(float(finite.min()), 1e-3),
                                         vmax=float(finite.max()))
        cmap = plt.get_cmap("viridis").copy()
        fmt = "{:.1f}"
    else:
        norm, cmap, fmt = None, plt.get_cmap("viridis").copy(), "{:.1f}"
    cmap.set_bad("none")
    im = ax.imshow(np.ma.masked_invalid(grid), aspect="auto", cmap=cmap,
                   norm=norm, origin="upper")
    # Hatch the missing cells so partial coverage is visible, not silent.
    for i in range(n_a):
        for j in range(n_s):
            if not math.isfinite(grid[i, j]):
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1,
                                           fill=False, hatch="//////",
                                           edgecolor="0.7", linewidth=0))
                continue
            if norm is not None and center is not None:
                # white text only on the dark (far-from-center) cells
                dark = abs(norm(grid[i, j]) - 0.5) > 0.32
            else:
                dark = grid[i, j] < (norm.vmax if norm else 1)
            ax.text(j, i, fmt.format(grid[i, j]), ha="center", va="center",
                    fontsize=5.5, color="white" if dark else "black")
    ax.set_xticks(range(n_s))
    ax.set_xticklabels(ss_axis, fontsize=7)
    ax.set_yticks(range(n_a))
    ylabels = ([a + " [pre-Vxc]" if _vxc_hatch(a) else a for a in archs]
               if vxc_pre_fix else list(archs))
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.set_xlabel("training subset_size")
    ax.set_title(title, fontsize=10)
    # Rung gutter + separators (left of the grid, between the spine and column 0),
    # so the Jacob's-ladder grouping of the rung-ordered arch axis reads at a
    # glance. Collision-free: arch tick labels sit outside the spine.
    if rung_separators and n_a:
        ax.set_xlim(-1.75, n_s - 0.5)
        for _rg, _s, _e in arch_style.rung_bands(archs):
            ax.add_patch(plt.Rectangle(
                (-1.62, _s - 0.5), 1.0, _e - _s, clip_on=False, zorder=2,
                facecolor=arch_style.RUNG_ACCENT.get(_rg, "0.5"), alpha=0.85,
                edgecolor="none"))
            ax.text(-1.12, (_s + _e - 1) / 2.0, _RUNG_SHORT.get(_rg, _rg),
                    rotation=90, va="center", ha="center", fontsize=5.5,
                    color="white", fontweight="bold", zorder=3, clip_on=False)
            if _s > 0:
                ax.axhline(_s - 0.5, color="0.15", lw=1.0, zorder=4)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)


def _heatmap_subset_axis(reaction_rows: List[Dict[str, Any]],
                         insample_rows: List[Dict[str, Any]]) -> List[int]:
    """Data-driven column axis for the arch x subset heatmaps: every subset
    size present in either row set, falling back to the historical
    SUBSET_SIZES grid when both are empty. Keeps sizes outside that grid
    (e.g. the full 26-pt dfs_step7 pool) from being silently dropped."""
    present = sorted(set(_present_subsets(reaction_rows))
                     | set(_present_subsets(insample_rows)))
    return present or list(SUBSET_SIZES)


def _heatmap_arch_axis(reaction_rows: List[Dict[str, Any]],
                       insample_rows: List[Dict[str, Any]]) -> List[str]:
    """Rung-sorted arch (y) axis for the arch x subset heatmaps: every arch that
    has a held-out reaction or in-sample AE cell, ordered by Jacob's-ladder rung
    so the rung separators drawn on the panel are contiguous (GGA -> meta-GGA ->
    rung-3.5 -> combined)."""
    present = (set(_archs_present(reaction_rows))
               | set(_archs_present(insample_rows)))
    ordered = arch_style.sort_by_rung([a for a in ARCH_ORDER if a in present])
    # Fall back to the full (rung-sorted) ladder when nothing is present, so the
    # panel never renders a zero-row grid (matches the pre-rung behavior).
    return ordered or arch_style.sort_by_rung(list(ARCH_ORDER))


def plot_arch_subset_heatmap(reaction_rows: List[Dict[str, Any]],
                             insample_rows: List[Dict[str, Any]],
                             out_path: Path, run_id: str, *,
                             n_trained: int, n_total: int,
                             n_holdout: int, note: str = "",
                             provenance: Optional[str] = None) -> Path:
    """Figure B -- arch × subset_size MAE heatmaps (held-out reactions +
    in-sample AE). Missing cells hatched; coverage stated in the footer."""
    with plt.rc_context(_STYLE):
        # Rung-ordered arch (y) axis so the rung separators are contiguous.
        all_archs = _heatmap_arch_axis(reaction_rows, insample_rows)
        ss_axis = _heatmap_subset_axis(reaction_rows, insample_rows)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.6))
        _hm_vxc = _run_predates_vxc_fix(run_id)
        _heatmap_panel(ax1, reaction_mae_by_arch_subset(reaction_rows),
                       all_archs, title="Held-out reaction-energy MAE (NN)",
                       cbar_label="MAE (kcal/mol)", subset_sizes=ss_axis,
                       rung_separators=True, vxc_pre_fix=_hm_vxc)
        _heatmap_panel(ax2, ae_mae_by_arch_subset(insample_rows), all_archs,
                       title="In-sample atomization-energy MAE",
                       cbar_label="MAE (kcal/mol)", subset_sizes=ss_axis,
                       rung_separators=True, vxc_pre_fix=_hm_vxc)
        fig.suptitle(f"Architecture × subset_size error grid · {run_id}",
                     fontsize=11)
        fig.text(0.5, 0.028,
                 f"Coverage: {n_trained}/{n_total} specs trained · "
                 f"{n_holdout} carry held-out reactions · hatched = no data. "
                 + (provenance or _PROVENANCE_BASE), ha="center",
                 fontsize=6.5, color="#777777")
        if note:
            fig.text(0.5, 0.006, note, ha="center", fontsize=6.5, color="#a33")
        fig.tight_layout(rect=(0, 0.06, 1, 0.95))
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return out_path


def plot_arch_subset_heatmap_vs_pbe(reaction_rows: List[Dict[str, Any]],
                                    out_path: Path, run_id: str, *,
                                    note: str = "",
                                    provenance: Optional[str] = None) -> Path:
    """Figure B2 -- arch × subset_size NN/PBE MAE-ratio grid (held-out
    reactions): diverging about 1.0, blue = the NN beats PBE on that cell's
    strict held-out set, red = worse. The per-cell PBE MAE is computed on
    exactly the reactions the NN was scored on, so every ratio is
    like-for-like; missing cells stay hatched."""
    with plt.rc_context(_STYLE):
        archs = _heatmap_arch_axis(reaction_rows, [])
        ss_axis = _heatmap_subset_axis(reaction_rows, [])
        nn = reaction_mae_by_arch_subset(reaction_rows)
        pbe = reaction_mae_by_arch_subset(reaction_rows,
                                          key="abs_error_pbe_kcalmol")
        ratio = {cell: nn[cell] / pbe[cell]
                 for cell in nn if _is_num(pbe.get(cell)) and pbe[cell] > 0}
        fig, ax = plt.subplots(figsize=(8.4, 5.6))
        _heatmap_panel(ax, ratio, archs,
                       title="Held-out reaction-energy MAE ratio NN/PBE "
                             "(<1 = NN better)",
                       cbar_label="MAE ratio NN/PBE", center=1.0,
                       subset_sizes=ss_axis, rung_separators=True,
                       vxc_pre_fix=_run_predates_vxc_fix(run_id))
        fig.suptitle(f"NN-vs-PBE cell grid · {run_id}", fontsize=11)
        fig.text(0.5, 0.028, provenance or _PROVENANCE_BASE, ha="center",
                 fontsize=6.5, color="#777777")
        if note:
            fig.text(0.5, 0.006, note, ha="center", fontsize=6.5, color="#a33")
        fig.tight_layout(rect=(0, 0.06, 1, 0.95))
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Figure C -- per-arch MAE bars
# ---------------------------------------------------------------------------

def plot_mae_by_arch(reaction_rows: List[Dict[str, Any]],
                     insample_rows: List[Dict[str, Any]],
                     out_path: Path, run_id: str, note: str = "",
                     provenance: Optional[str] = None,
                     scan_baseline: Optional[Dict[str, float]] = None,
                     scan_errors: Optional[Dict[str, float]] = None) -> Path:
    """Figure C -- per-arch MAE bars (log-y): held-out reaction MAE (mean &
    best over available subsets) + in-sample AE MAE, with PBE-vs-ref line.
    Archs are rung-ordered with Jacob's-ladder rung bands; a beats-PBE marker
    tags every held-out reaction bar below the PBE line; a dotted SCAN
    reference line is drawn row-matched to the same deduped rows as the PBE
    line when ``scan_errors`` covers them (:func:`scan_row_matched_ref`),
    falling back to the full-pool ``scan_baseline`` value (absent SCAN
    cache -> unchanged).

    Both bar statistics aggregate OVER subset_size, so an arch that has only
    reached the small subsets is not comparable to one trained on the full pool.
    Every bar of such an arch is hatched (:data:`_SHALLOW_HATCH`) and its
    subset_size span is stated under the tick, with the mismatch named in a
    footer line -- on a level grid no bar is hatched and the footer is
    unchanged."""
    with plt.rc_context({**_STYLE, **_SHALLOW_HATCH_RC}):
        rxn_map = reaction_mae_by_arch_subset(reaction_rows)
        ae_map = ae_mae_by_arch_subset(insample_rows)
        archs = arch_style.sort_by_rung([a for a in ARCH_ORDER
                 if any(k[0] == a for k in rxn_map)
                 or any(k[0] == a for k in ae_map)])
        # Depth is judged over BOTH row sets, matching the arch axis above (an
        # arch can carry in-sample AE cells without held-out reactions yet).
        depth_rows = list(reaction_rows) + list(insample_rows)
        shallow, _deepest = _shallow_archs(depth_rows)
        ss_cov = _subset_coverage(depth_rows)
        cov_caveat = _coverage_caveat(depth_rows)

        def _arch_stat(mp, arch, stat):
            vals = [v for (a, _ss), v in mp.items() if a == arch]
            if not vals:
                return np.nan
            return float(np.mean(vals)) if stat == "mean" else float(np.min(vals))

        xs = np.arange(len(archs))
        w = 0.27
        rxn_mean = [_arch_stat(rxn_map, a, "mean") for a in archs]
        rxn_best = [_arch_stat(rxn_map, a, "best") for a in archs]
        ae_mean = [_arch_stat(ae_map, a, "mean") for a in archs]

        fig, ax = plt.subplots(figsize=(11, 5.6))
        # Rung bands: shade GGA / meta-GGA / rung-3.5 / combined groups so the
        # "does climbing Jacob's ladder help?" comparison is legible (archs are
        # rung-ordered above; the flat layout hid the rung structure).
        for _rg, _s, _e in arch_style.rung_bands(archs):
            ax.axvspan(_s - 0.5, _e - 0.5, color=arch_style.RUNG_BAND[_rg],
                       alpha=0.6, zorder=0)
        bars = [
            ax.bar(xs - w, rxn_mean, w, label="held-out reaction MAE (mean)",
                   color="#4f81bd", edgecolor="k", linewidth=0.3),
            ax.bar(xs, rxn_best, w,
                   label="held-out reaction MAE (best subset-size)",
                   color="#9dc3e6", edgecolor="k", linewidth=0.3),
            ax.bar(xs + w, ae_mean, w, label="in-sample AE MAE (mean)",
                   color="#c0504d", edgecolor="k", linewidth=0.3),
        ]
        # Hatch every bar of an arch that has not been trained as deep as the
        # deepest arch here. Set per-patch (not via bar(hatch=...)) so the mark
        # does not depend on a matplotlib version that broadcasts the kwarg.
        for cont in bars:
            for patch, arch in zip(cont.patches, archs):
                if arch in shallow:
                    patch.set_hatch(_SHALLOW_HATCH)

        # Row-matched pair: PBE deduped over the pulled rows (the same
        # arithmetic as the ED anchor), SCAN over the SAME rows when the
        # cache covers them -- previously the SCAN line was the full-pool
        # value beside a row-matched PBE line, two different reaction sets
        # inside one figure.
        pbe_vs_ref = pbe_reaction_mae_baseline(reaction_rows)
        scan_c, scan_label = scan_row_matched_ref(reaction_rows, scan_errors,
                                                  scan_baseline)
        if _is_num(pbe_vs_ref):
            # The footer's PBE/SCAN values are full-pool; the reference
            # lines here are (or can be) reduced over this figure's own
            # rows. Two numbers for one functional on one figure need the
            # distinction stated where they are read.
            ref_note = ("Reference lines: reduced over this figure's own "
                        "deduped held-out rows (PBE always; SCAN when "
                        "row-matched); the grey footer's PBE/SCAN values "
                        "are full-pool.")
            note = (note + "  " + ref_note) if note else ref_note
        if _is_num(pbe_vs_ref):
            ax.axhline(pbe_vs_ref, ls="--", color="k", linewidth=1.2,
                       label=f"PBE-vs-benchmark MAE ({pbe_vs_ref:.1f})")
            # Mark held-out reaction bars (mean + best-subset) below their
            # OWN-RUNG reference: PBE for GGA archs, the SCAN reference for
            # beyond-GGA ones (unmarked when the SCAN cache is absent --
            # a beyond-GGA arch is never credited for merely beating PBE).
            kinds = arch_reference_kinds(archs)
            marks = []
            for i, arch in enumerate(archs):
                ref = (scan_c if kinds.get(arch) == "scan" else pbe_vs_ref)
                if not _is_num(ref):
                    continue
                for xpos, h in ((xs[i] - w, rxn_mean[i]), (xs[i], rxn_best[i])):
                    if _is_num(h) and h < float(ref):
                        marks.append((xpos, h))
            if marks:
                mx, mh = zip(*marks)
                ax.scatter(mx, mh, marker="v", s=22, color="#2ca02c",
                           edgecolor="k", linewidths=0.3, zorder=6,
                           label="beats rung reference")
        if scan_c is not None:
            ax.axhline(scan_c, ls=":", color="#555555", linewidth=1.3,
                       label=scan_label)
        ax.axhline(1.0, ls=":", color="green", linewidth=1.0,
                   label="chemical accuracy (1 kcal/mol)")

        ax.set_yscale("log")
        ax.set_xticks(xs)
        # Tick labels carry the subset_size span each arch's bars aggregate, so
        # the depth behind a bar is readable without consulting the heatmap.
        ax.set_xticklabels(
            [f"{a}\n(ss {_coverage_span(ss_cov[a])})" if a in ss_cov else a
             for a in archs], rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("MAE (kcal/mol, log scale)")
        ax.set_title(f"Per-architecture error · {run_id}")
        handles, _labels = ax.get_legend_handles_labels()
        if shallow:
            handles.append(Patch(facecolor="0.8", edgecolor="k",
                                 hatch=_SHALLOW_HATCH,
                                 label="shallower training depth (see footer)"))
        ax.legend(handles=handles, fontsize=7, ncol=2)
        ax.grid(True, axis="y", which="both", alpha=0.3)
        # Without a coverage caveat the footer stack is unchanged; with one the
        # two lines are lifted to make room rather than overprinting.
        y_note, rect_bottom = (0.045, 0.075) if cov_caveat else (0.028, 0.06)
        if note:
            fig.text(0.5, y_note, note, ha="center", fontsize=6.5, color="#a33")
        if cov_caveat:
            fig.text(0.5, 0.026, cov_caveat, ha="center", fontsize=6.0,
                     color="#a33")
        fig.text(0.5, 0.006, provenance or _PROVENANCE_BASE, ha="center",
                 fontsize=6, color="#777777")
        fig.tight_layout(rect=(0, rect_bottom, 1, 1))
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Figure: Jacob's-ladder rung summary (headline "does climbing help?")
# ---------------------------------------------------------------------------

def plot_rung_summary(rows: List[Dict[str, Any]], out_path: Path, run_id: str, *,
                      pbe_baseline: Optional[Dict[str, float]] = None,
                      scan_baseline: Optional[Dict[str, float]] = None,
                      note: str = "", provenance: Optional[str] = None,
                      caveat: Optional[str] = None,
                      dataset: Optional[str] = None) -> Path:
    """Headline rung figure -- "does climbing Jacob's ladder to meta-GGA help?".

    Per rung (in :data:`arch_style.RUNG_ORDER`, only those present), the MEAN over
    that rung's architectures of each arch's BEST-subset (largest subset_size)
    held-out reaction MAE, split BH76 barriers vs W4-11 atomization. Grouped bars
    (BH76 solid, W4-11 hatched ``//``) colored by :data:`arch_style.RUNG_ACCENT`,
    each value annotated. The full-pool PBE (dashed) and -- when a SCAN cache is
    present -- SCAN (dotted) combined baselines are drawn as labeled reference
    lines. Absent SCAN -> no SCAN line (guarded, backward compatible).

    A rung that entered the sweep late sits at a smaller subset_size than the
    rungs before it, and the ladder difference would then be a training-set-size
    difference. Hatch is already spent on the BH76/W4-11 split and face color on
    rung identity, so under-trained rungs are marked by FADING the face toward
    white (the rung hue survives) and switching to a dotted warning edge; the
    subset_size each rung aggregates is annotated under every bar pair. On a
    level grid nothing fades and the figure is unchanged."""
    with plt.rc_context(_STYLE):
        by_r = arch_style.by_rung(_archs_present(rows))
        rungs = [r for r in arch_style.RUNG_ORDER if r in by_r]
        best = _best_subset_per_arch(rows)
        shallow_rungs, _deepest = _shallow_rungs(rows)
        rung_cov = _rung_coverage(rows)
        cov_caveat = _coverage_caveat(rows, by_rung=True)
        pool_mae = {p: reaction_mae_by_arch_subset(
                        [r for r in rows if r.get("pool") == p])
                    for p in ("bh76", "w411")}

        def _rung_pool_mean(rung: str, pool: str) -> float:
            vals = []
            for a in by_r.get(rung, []):
                ss = best.get(a)
                v = pool_mae[pool].get((a, ss)) if ss is not None else None
                if _is_num(v):
                    vals.append(v)
            return float(np.mean(vals)) if vals else float("nan")

        bh = [_rung_pool_mean(r, "bh76") for r in rungs]
        w4 = [_rung_pool_mean(r, "w411") for r in rungs]
        xs = np.arange(len(rungs))
        w = 0.38
        # A two-rung run makes a narrow canvas; widen it when the coverage
        # disclosure has to fit, so the line is readable instead of wrapping
        # into the provenance stamp.
        fig_w = max(6.0, 1.9 * len(rungs) + 3.0)
        if cov_caveat:
            fig_w = max(fig_w, 10.0)
        fig, ax = plt.subplots(figsize=(fig_w, 5.4))
        for i, rg in enumerate(rungs):
            c = arch_style.RUNG_ACCENT.get(rg, "0.5")
            is_shallow = rg in shallow_rungs
            face = _fade(c) if is_shallow else c
            edge = _SHALLOW_EDGE if is_shallow else "k"
            elw = 1.4 if is_shallow else 0.4
            els = ":" if is_shallow else "-"
            ax.bar(xs[i] - w / 2, bh[i], w, color=face, edgecolor=edge,
                   linewidth=elw, linestyle=els)
            ax.bar(xs[i] + w / 2, w4[i], w, color=face, edgecolor=edge,
                   linewidth=elw, linestyle=els, hatch="//")
            for xc, val in ((xs[i] - w / 2, bh[i]), (xs[i] + w / 2, w4[i])):
                if _is_num(val):
                    ax.annotate(f"{val:.1f}", (xc, val), ha="center", va="bottom",
                                fontsize=6.5, xytext=(0, 1.5),
                                textcoords="offset points")
        pbe_c = (pbe_baseline or {}).get("combined")
        if _is_num(pbe_c):
            ax.axhline(pbe_c, ls="--", color="0.35", linewidth=1.4,
                       label=(f"PBE (combined {pbe_c:.1f}"
                              f"{pool_line_suffix(pbe_baseline)})"))
        scan_c, scan_cov = scan_line_value(scan_baseline, "combined")
        if scan_c is not None:
            ax.axhline(scan_c, ls=":", color="#555555", linewidth=1.6,
                       label=f"SCAN (combined {scan_c:.1f}{scan_cov})")
        # BH76/W4-11 hatch key + the reference lines (+ the coverage key when
        # some rung is shallower than the deepest one plotted).
        style_handles = [
            Patch(facecolor="0.75", edgecolor="k", label="BH76 barriers"),
            Patch(facecolor="0.75", edgecolor="k", hatch="//",
                  label="W4-11 atomization")]
        if shallow_rungs:
            style_handles.append(
                Patch(facecolor=_fade("0.75"), edgecolor=_SHALLOW_EDGE,
                      linewidth=1.4, linestyle=":",
                      label="shallower training depth (see footer)"))
        ref_h, ref_l = ax.get_legend_handles_labels()
        ax.legend(handles=style_handles + ref_h, fontsize=7, ncol=2, loc="best",
                  framealpha=0.7)
        ax.set_xticks(xs)
        # The arch count and the subset_size span each rung aggregates both ride
        # in the tick label (one text row, no collision with the axis label), and
        # a shallow rung's label takes the warning color to match its bars.
        ax.set_xticklabels(
            [f"{r}\n(n={len(by_r.get(r, []))}"
             + (f", ss {_coverage_span(rung_cov[r])})" if r in rung_cov else ")")
             for r in rungs], fontsize=9)
        for lbl, rg in zip(ax.get_xticklabels(), rungs):
            if rg in shallow_rungs:
                lbl.set_color(_SHALLOW_EDGE)
        ax.set_xlabel("Jacob's-ladder rung  (mean over the rung's architectures, "
                      "each at its best subset_size)", fontsize=8.5)
        ax.set_ylabel("held-out reaction-energy MAE (kcal/mol)", fontsize=9)
        ax.grid(True, axis="y", alpha=0.3)
        _stamp_parity_footer(
            fig, run_id=run_id, note=note, provenance=provenance, caveat=caveat,
            dataset=dataset, extra_note=cov_caveat or None,
            title="Jacob's-ladder rung summary -- held-out MAE (BH76 | W4-11)")
        fig.tight_layout(rect=(0, 0.075 if not cov_caveat else 0.115, 1, 0.93))
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Figure D (bonus) -- MAE vs subset_size, one line per arch
# ---------------------------------------------------------------------------

_RUNG_LS: Dict[str, str] = {
    arch_style.RUNG_GGA: "-", arch_style.RUNG_MGGA: "--",
    arch_style.RUNG_R35: "-.", arch_style.RUNG_R35_MGGA: ":",
}


def _rung_linestyles(archs: List[str]) -> Dict[str, str]:
    """One linestyle per Jacob's-ladder rung (solid GGA through dotted
    rung-3.5+meta-GGA) so a multi-rung frame separates the families without
    leaning on color alone."""
    return {a: _RUNG_LS.get(arch_style.rung_of(a), "-") for a in archs}


def _mae_vs_subset_panel(ax, mae_map: Dict[Tuple[str, int], float],
                         archs: List[str], *, title: str,
                         pbe_line: Optional[float] = None,
                         pbe_suffix: str = "",
                         scan_line: Optional[float] = None,
                         scan_suffix: str = "",
                         ls_for: Optional[Dict[str, str]] = None) -> None:
    for a in archs:
        pts = sorted((ss, v) for (aa, ss), v in mae_map.items() if aa == a)
        if not pts:
            continue
        xx, yy = zip(*pts)
        ax.plot(xx, yy, marker="o", ms=4, linewidth=1.3,
                color=ARCH_COLOR.get(a, "0.5"),
                ls=(ls_for or {}).get(a, "-"), label=a)
    if pbe_line is not None and math.isfinite(pbe_line):
        ax.axhline(pbe_line, ls="--", color="0.35", lw=1.2,
                   label=f"PBE full-pool MAE ({pbe_line:.1f}{pbe_suffix})")
    if scan_line is not None and math.isfinite(scan_line):
        ax.axhline(scan_line, ls=":", color="#555555", lw=1.4,
                   label=f"SCAN full-pool MAE ({scan_line:.1f}{scan_suffix})")
    ax.set_yscale("log")
    ax.set_xlabel("training subset_size")
    ax.set_ylabel("MAE (kcal/mol, log)")
    ax.set_title(title, fontsize=10)
    ax.grid(True, which="both", alpha=0.3)


def plot_mae_vs_subset(reaction_rows: List[Dict[str, Any]],
                       insample_rows: List[Dict[str, Any]],
                       out_path: Path, run_id: str, note: str = "",
                       provenance: Optional[str] = None, *,
                       pbe_baseline: Optional[Dict[str, Any]] = None,
                       scan_baseline: Optional[Dict[str, Any]] = None) -> Path:
    """Figure D -- learning curves: MAE vs subset_size, one line per arch,
    rung-ordered with one linestyle per Jacob's-ladder rung. ``pbe_baseline``
    / ``scan_baseline`` (pool-baseline dicts) draw the dashed-PBE and
    dotted-SCAN full-pool reference lines on the held-out panel; the SCAN
    line obeys the coverage gate (:func:`scan_line_value`)."""
    with plt.rc_context(_STYLE):
        archs = [a for a in _heatmap_arch_axis(reaction_rows, insample_rows)
                 if a in _archs_present(reaction_rows)
                 or a in _archs_present(insample_rows)]
        ls_for = _rung_linestyles(archs)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.4))
        pbe_c = (pbe_baseline or {}).get("combined")
        held_pbe = float(pbe_c) if _is_num(pbe_c) else None
        held_scan, held_scan_sfx = scan_line_value(scan_baseline, "combined")
        _mae_vs_subset_panel(ax1, reaction_mae_by_arch_subset(reaction_rows),
                             archs, title="Held-out reaction-energy MAE",
                             pbe_line=held_pbe,
                             pbe_suffix=pool_line_suffix(pbe_baseline),
                             scan_line=held_scan,
                             scan_suffix=held_scan_sfx, ls_for=ls_for)
        _mae_vs_subset_panel(ax2, ae_mae_by_arch_subset(insample_rows),
                             archs, title="In-sample atomization-energy MAE",
                             ls_for=ls_for)
        ax1.legend(fontsize=6, ncol=2)
        fig.suptitle(f"Error vs training-subset size · {run_id}", fontsize=11)
        if note:
            fig.text(0.5, 0.03, note, ha="center", fontsize=6.5, color="#a33")
        fig.text(0.5, 0.008, provenance or _PROVENANCE_BASE, ha="center",
                 fontsize=6, color="#777777")
        fig.tight_layout(rect=(0, 0.06, 1, 0.95))
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Figure E -- held-out atomization-energy parity (W4-11)
# ---------------------------------------------------------------------------
# W4-11 is the held-out set's atomization-energy benchmark: each W4-11 "reaction"
# is a molecule -> constituent-atoms atomization, so its ``de_nn``/``de_pbe``
# ARE predicted atomization energies and ``ref`` is the reference AE. This is
# the atomization-energy analog of the held-out reaction parity (plot_parity),
# and -- unlike an in-sample training-fit plot -- it shows generalization, with
# PBE drawn as the baseline so "does the NN beat PBE?" is legible.

def _w411_rows(reaction_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Held-out W4-11 reactions (the atomization-energy pool) with finite
    NN / PBE / reference values."""
    return [r for r in reaction_rows
            if r.get("pool") == "w411"
            and _is_num(r.get("ref_kcalmol"))
            and _is_num(r.get("de_nn_kcalmol"))
            and _is_num(r.get("de_pbe_kcalmol"))]


def _w411_mae_by_subset(rows: List[Dict[str, Any]], *,
                        key: str = "abs_error_nn_kcalmol") -> Dict[int, float]:
    """``{subset_size: MAE}`` over held-out W4-11 ``key`` (NN or PBE abs error),
    pooled across architectures."""
    buckets: Dict[int, List[float]] = {}
    for r in rows:
        ss = r.get("subset_size")
        if ss is None or not _is_num(r.get(key)):
            continue
        buckets.setdefault(ss, []).append(r[key])
    return {k: float(np.mean(v)) for k, v in buckets.items() if v}


def plot_ae_parity(reaction_rows: List[Dict[str, Any]], out_path: Path,
                   run_id: str, note: str = "",
                   provenance: Optional[str] = None) -> Path:
    """Figure E -- HELD-OUT atomization-energy parity (W4-11): predicted vs
    reference atomization energy (kcal/mol), the AE analog of
    :func:`plot_parity`. PBE is drawn as the baseline throughout so the
    NN-vs-PBE comparison is explicit.

    Panel (a) colors points by architecture (each arch's largest-subset spec),
    overlays PBE as grey ×, and shows a per-arch NN-vs-ref MAE inset with the
    PBE-vs-ref MAE as the dashed baseline. Panel (b) colors every held-out
    W4-11 point by training-subset size with an NN-MAE-vs-subset inset that
    also draws the PBE baseline."""
    with plt.rc_context(_STYLE):
        rows = _w411_rows(reaction_rows)
        archs = _archs_present(rows)
        best = _best_subset_per_arch(rows)
        fig, (axa, axb) = plt.subplots(1, 2, figsize=(13, 7.4))

        # Panel (a): by architecture, each arch's representative (largest) spec.
        sel = [r for r in rows if r.get("arch") in best
               and r.get("subset_size") == best[r["arch"]]]
        xs_a, ys_a = [], []
        for arch in archs:
            pts = [(r["ref_kcalmol"], r["de_nn_kcalmol"]) for r in sel
                   if r.get("arch") == arch]
            if not pts:
                continue
            xx, yy = zip(*pts)
            xs_a += list(xx); ys_a += list(yy)
            axa.scatter(xx, yy, s=16, alpha=0.6, color=ARCH_COLOR[arch],
                        edgecolor="none", zorder=3, label=arch)
        # PBE baseline points (same physical PBE for every arch -- draw once).
        pbe_pts = [(r["ref_kcalmol"], r["de_pbe_kcalmol"]) for r in sel]
        if pbe_pts:
            xx, yy = zip(*pbe_pts)
            xs_a += list(xx); ys_a += list(yy)
            axa.scatter(xx, yy, s=12, marker="x", alpha=0.4, color="0.4",
                        zorder=2, label="PBE")
        limits_a = _robust_limits(xs_a + ys_a, q=(1.0, 99.0))
        n_out_a = _diagonal(axa, xs_a, ys_a, limits=limits_a)
        if n_out_a:
            axa.text(0.02, 0.97, f"{n_out_a} point(s) beyond axis",
                     transform=axa.transAxes, fontsize=6.5, va="top",
                     color="#a33")
        axa.set_xlabel("Reference W4-11 atomization energy  (kcal/mol)")
        axa.set_ylabel("Predicted atomization energy  (kcal/mol)")
        axa.set_title("(a) held-out W4-11 AE vs reference -- by architecture")
        mae_by_arch = {
            a: m for a in archs
            if (m := _mae([r["abs_error_nn_kcalmol"] for r in sel
                           if r.get("arch") == a
                           and _is_num(r.get("abs_error_nn_kcalmol"))])) is not None
        }
        pbe_vs_ref = _mae([r["abs_error_pbe_kcalmol"] for r in sel])
        _draw_mae_inset(axa, mae_by_arch, archs,
                        title="per-arch NN-vs-ref MAE", baseline=pbe_vs_ref)

        # Panel (b): every held-out W4-11 point, colored by subset_size.
        all_pts = [(r["ref_kcalmol"], r["de_nn_kcalmol"], r.get("subset_size"))
                   for r in rows if _is_num(r.get("subset_size"))]
        cmap = plt.get_cmap("viridis")
        if all_pts:
            xs_b = [p[0] for p in all_pts]
            ys_b = [p[1] for p in all_pts]
            css = [p[2] for p in all_pts]
            ss_present = sorted(set(css))
            norm = matplotlib.colors.Normalize(
                vmin=min(ss_present), vmax=max(ss_present))
            sc = axb.scatter(xs_b, ys_b, s=16, alpha=0.6, c=css, cmap=cmap,
                             norm=norm, edgecolor="none", zorder=3)
            n_out_b = _diagonal(axb, xs_b, ys_b,
                                limits=_robust_limits(xs_b + ys_b, q=(1.0, 99.0)))
            if n_out_b:
                axb.text(0.02, 0.97, f"{n_out_b} point(s) beyond axis",
                         transform=axb.transAxes, fontsize=6.5, va="top",
                         color="#a33")
            cbar = fig.colorbar(sc, ax=axb, fraction=0.046, pad=0.04)
            cbar.set_label("training subset_size", fontsize=7)
            cbar.ax.tick_params(labelsize=6)
            # NN-MAE-vs-subset inset with the PBE baseline.
            mae_by_ss = _w411_mae_by_subset(rows)
            if mae_by_ss:
                inset = axb.inset_axes([0.56, 0.13, 0.41, 0.33])
                sss = sorted(mae_by_ss)
                xs_i = np.arange(len(sss))
                inset.bar(xs_i, [mae_by_ss[s] for s in sss],
                          color=[cmap(norm(s)) for s in sss],
                          edgecolor="k", linewidth=0.3)
                pbe_b = _mae([r["abs_error_pbe_kcalmol"] for r in rows])
                if pbe_b is not None and math.isfinite(pbe_b):
                    inset.axhline(pbe_b, ls="--", color="k", linewidth=1.0,
                                  label="PBE")
                    inset.legend(fontsize=5, loc="upper left", framealpha=0.6)
                inset.set_xticks(xs_i)
                inset.set_xticklabels([str(s) for s in sss], fontsize=5)
                inset.set_title("NN AE MAE vs subset_size", fontsize=6)
                inset.set_ylabel("MAE", fontsize=5)
                inset.set_xlabel("subset_size", fontsize=5)
                inset.tick_params(axis="y", labelsize=5)
                inset.grid(True, axis="y", alpha=0.3)
        axb.set_xlabel("Reference W4-11 atomization energy  (kcal/mol)")
        axb.set_ylabel("NN atomization energy  (kcal/mol)")
        axb.set_title("(b) held-out W4-11 AE vs reference -- by train-set size")

        # Each arch's representative is its own deepest spec, and archs enter
        # the sweep at different depths -- so the cloud can mix a 1-molecule
        # net with a full-pool one. The legend states which depth each is.
        handles = [Patch(facecolor=ARCH_COLOR[a],
                         label=f"{a} (ss {best[a]})" if a in best else a)
                   for a in archs]
        handles.append(plt.Line2D([], [], marker="x", ls="", color="0.4",
                                   label="PBE"))
        fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=7,
                   frameon=False, bbox_to_anchor=(0.5, 0.085))
        fig.suptitle(
            "Held-out atomization-energy parity (W4-11) -- "
            f"each arch at its largest available subset_size · {run_id}",
            fontsize=11, y=0.985)
        fig.text(0.5, 0.925,
                 "W4-11 (held-out): each reaction is a molecule->atoms "
                 "atomization energy. PBE (grey ×, dashed baseline) is the bar "
                 "to beat; points above/below the diagonal over/under-bind.",
                 ha="center", fontsize=7.5, style="italic", color="#444444")
        if note:
            fig.text(0.5, 0.05, note, ha="center", fontsize=6.5,
                     color="#a33", wrap=True)
        fig.text(0.5, 0.016, provenance or _PROVENANCE_BASE, ha="center",
                 fontsize=6, color="#777777")
        fig.tight_layout(rect=(0, 0.155, 1, 0.90))
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Parity layout variants (pools separated by scale; ALL arch x subset shown)
# ---------------------------------------------------------------------------
# Five candidate parity figures, all fixing the same three defects of the
# original plot_parity: (1) only one spec/arch plotted, (2) both pools crushed
# onto one shared axis, (3) inset reflecting a single spec. Each renders every
# (arch, subset_size) cell, separates the two pools onto their own scales, draws
# PBE once per panel (PBE is NN-invariant), and uses the live footers.

_POOL_LABEL = {"bh76": "BH76 (barriers)", "w411": "W4-11 (atomizations)"}
_ARCH_SHORT = {"deep": "base", "deep_attn": "attn", "deep_cusp": "cusp",
               "deep_dm": "dm", "deep_combined": "comb",
               "deep_combined_attn": "comb_at",
               "deep_notransform": "notr", "deep_notransform_attn": "notr_at"}


def _present_pools(rows: List[Dict[str, Any]]) -> List[str]:
    return [p for p in ("bh76", "w411") if any(r.get("pool") == p for r in rows)]


def _present_subsets(rows: List[Dict[str, Any]]) -> List[int]:
    return sorted({r["subset_size"] for r in rows
                   if r.get("subset_size") is not None})


def _pool_parity_limits(rows: List[Dict[str, Any]], pool: str
                        ) -> Optional[Tuple[float, float]]:
    """Robust square parity window for ONE pool, over ref + de_nn + de_pbe (so
    the grey PBE cloud stays on-frame). Each pool gets its own scale -- the fix
    for BH76 (+-150 kcal/mol) being crushed by W4-11 (0..1300)."""
    pr = [r for r in rows if r.get("pool") == pool]
    vals: List[Any] = []
    for key in ("ref_kcalmol", "de_nn_kcalmol", "de_pbe_kcalmol"):
        vals += [r.get(key) for r in pr]
    return _robust_limits(vals, q=(1.0, 99.0))


def _parity_scatter(ax, panel_rows: List[Dict[str, Any]], *, color_by: str,
                    limits: Optional[Tuple[float, float]],
                    subset_values: Optional[List[int]] = None,
                    draw_pbe: bool = True, point_size: float = 11.0):
    """Draw one parity panel: NN de_nn (y) vs reference (x), colored by ``arch``
    (discrete ``ARCH_COLOR``) or ``subset`` (viridis ``Normalize``). PBE grey-x
    drawn once. y=x clipped to ``limits`` via :func:`_diagonal`; off-axis count
    annotated. Returns ``(n_out, mappable)`` -- ``mappable`` is the viridis
    scatter (for a colorbar) or None."""
    xs: List[float] = []
    ys: List[float] = []
    mappable = None
    if color_by == "arch":
        for a in _archs_present(panel_rows):
            pts = [(r["ref_kcalmol"], r["de_nn_kcalmol"]) for r in panel_rows
                   if r.get("arch") == a and _is_num(r.get("ref_kcalmol"))
                   and _is_num(r.get("de_nn_kcalmol"))]
            if not pts:
                continue
            xx, yy = zip(*pts)
            xs += list(xx); ys += list(yy)
            ax.scatter(xx, yy, s=point_size, alpha=0.5, color=ARCH_COLOR[a],
                       edgecolor="none", zorder=3, label=a)
    else:  # subset_size -> viridis
        sv = subset_values or _present_subsets(panel_rows)
        norm = matplotlib.colors.Normalize(
            vmin=min(sv) if sv else 0, vmax=max(sv) if sv else 1)
        pts = [(r["ref_kcalmol"], r["de_nn_kcalmol"], r["subset_size"])
               for r in panel_rows if _is_num(r.get("ref_kcalmol"))
               and _is_num(r.get("de_nn_kcalmol"))
               and r.get("subset_size") is not None]
        if pts:
            xx, yy, ss = zip(*pts)
            xs += list(xx); ys += list(yy)
            mappable = ax.scatter(xx, yy, s=point_size, alpha=0.55, c=ss,
                                  cmap="viridis", norm=norm, edgecolor="none",
                                  zorder=3)
    if draw_pbe:
        pbe = [(r["ref_kcalmol"], r["de_pbe_kcalmol"]) for r in panel_rows
               if _is_num(r.get("ref_kcalmol"))
               and _is_num(r.get("de_pbe_kcalmol"))]
        if pbe:
            xx, yy = zip(*pbe)
            xs += list(xx); ys += list(yy)
            ax.scatter(xx, yy, s=max(6.0, point_size - 3), marker="x",
                       alpha=0.3, color="0.5", zorder=2, label="PBE")
    if not xs:
        ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center",
                va="center", fontsize=7, color="0.6")
    n_out = _diagonal(ax, xs, ys, limits=limits)
    if n_out:
        ax.text(0.03, 0.97, f"{n_out} off-axis", transform=ax.transAxes,
                fontsize=5.5, va="top", color="#a33")
    return n_out, mappable


def _combined_mae_inset(ax, rows_for_subset: List[Dict[str, Any]],
                        archs: List[str],
                        pbe_combined: Optional[float]) -> None:
    """Inset (upper-left of ``ax``): per-arch COMBINED (BH76+W4-11) held-out
    NN-MAE bars for one subset_size, with the PBE combined baseline dashed.
    Honest across all archs at that subset (no single-spec cherry-pick)."""
    inset = ax.inset_axes([0.085, 0.60, 0.36, 0.35])
    xs = np.arange(len(archs))
    heights = []
    for a in archs:
        errs = [r["abs_error_nn_kcalmol"] for r in rows_for_subset
                if r.get("arch") == a and _is_num(r.get("abs_error_nn_kcalmol"))]
        heights.append(float(np.mean(errs)) if errs else np.nan)
    inset.bar(xs, heights, color=[ARCH_COLOR[a] for a in archs],
              edgecolor="k", linewidth=0.3)
    if pbe_combined is not None and math.isfinite(pbe_combined):
        inset.axhline(pbe_combined, ls="--", color="k", linewidth=0.8)
    inset.set_xticks(xs)
    inset.set_xticklabels([_ARCH_SHORT.get(a, a) for a in archs],
                          rotation=40, ha="right", fontsize=4)
    inset.tick_params(axis="y", labelsize=4)
    inset.set_title("combined MAE", fontsize=5)
    inset.grid(True, axis="y", alpha=0.3)


def _arch_pbe_legend_handles(archs: List[str], *, pools: Optional[List[str]] = None):
    # Rung-ordered so the compact (ncol ~ #rungs) legend reads up Jacob's ladder.
    handles = [Patch(facecolor=ARCH_COLOR[a], label=a)
               for a in arch_style.sort_by_rung(archs)]
    if pools:
        handles += [plt.Line2D([], [], marker=POOL_MARKER[p], ls="", color="0.3",
                                label=p.upper()) for p in pools]
    handles.append(plt.Line2D([], [], marker="x", ls="", color="0.5", label="PBE"))
    return handles


def plot_parity_by_class(reaction_rows: List[Dict[str, Any]], out_path: Path,
                         run_id: str, note: str = "",
                         provenance: Optional[str] = None,
                         caveat: Optional[str] = None,
                         dataset: Optional[str] = None) -> Path:
    """Held-out parity split by reaction class -- W4-11 atomization | BH76
    barriers | total -- predicted vs benchmark reference. Top row: by
    architecture (each arch's largest-subset spec, PBE as grey x); bottom
    row: every cell's points colored by training subset_size with one
    shared colorbar. Each class column carries its own square limits
    (:func:`_pool_parity_limits` -- the fix for BH76's +-150 kcal/mol being
    crushed by W4-11's 0..1300), shared down the column so the two rows of
    one class compare directly. Panel bodies are :func:`_parity_scatter`,
    the same machinery as the parity variants."""
    with plt.rc_context(_STYLE):
        best = _best_subset_per_arch(reaction_rows)
        sel = [r for r in reaction_rows if r.get("arch") in best
               and r.get("subset_size") == best[r["arch"]]]
        ss_all = sorted({r.get("subset_size") for r in reaction_rows
                         if _is_num(r.get("subset_size"))})
        classes = (("w411", "W4-11 AE"), ("bh76", "BH76 barriers"),
                   (None, "total (BH76+W4-11)"))
        fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.8), squeeze=False)
        mappable = None
        for c, (pool, lab) in enumerate(classes):
            if pool is None:
                cls_all = list(reaction_rows)
                cls_sel = list(sel)
                vals: List[Any] = []
                for key in ("ref_kcalmol", "de_nn_kcalmol",
                            "de_pbe_kcalmol"):
                    vals += [r.get(key) for r in reaction_rows]
                limits = _robust_limits(vals, q=(1.0, 99.0))
            else:
                cls_all = [r for r in reaction_rows
                           if r.get("pool") == pool]
                cls_sel = [r for r in sel if r.get("pool") == pool]
                limits = _pool_parity_limits(reaction_rows, pool)
            axa = axes[0][c]
            _parity_scatter(axa, cls_sel, color_by="arch", limits=limits)
            axa.set_title(f"({'abc'[c]}) {lab} -- by architecture "
                          "(largest subset per arch)", fontsize=9)
            if c == 0:
                axa.set_ylabel("predicted reaction energy (kcal/mol)")
            axb = axes[1][c]
            _n_out, mp = _parity_scatter(axb, cls_all, color_by="subset",
                                         limits=limits,
                                         subset_values=ss_all)
            if mp is not None:
                mappable = mp
            axb.set_title(f"({'def'[c]}) {lab} -- by training subset_size "
                          "(all cells)", fontsize=9)
            axb.set_xlabel("reference reaction energy (kcal/mol)")
            if c == 0:
                axb.set_ylabel("predicted reaction energy (kcal/mol)")
        # Reserve the footer/legend band explicitly: tight_layout fights the
        # ax-spanning colorbar, so the margins are set directly.
        fig.subplots_adjust(bottom=0.16, top=0.90, hspace=0.30, wspace=0.22)
        archs = _archs_present(reaction_rows)
        if archs:
            fig.legend(handles=_arch_pbe_legend_handles(archs),
                       loc="lower center", ncol=min(len(archs) + 1, 7),
                       fontsize=7, frameon=False,
                       bbox_to_anchor=(0.5, 0.065))
        if mappable is not None:
            cbar = fig.colorbar(mappable,
                                ax=[axes[1][k] for k in range(3)],
                                fraction=0.025, pad=0.01)
            cbar.set_label("training subset_size", fontsize=7)
            cbar.ax.tick_params(labelsize=6)
        _stamp_parity_footer(
            fig, run_id=run_id, note=note, provenance=provenance,
            caveat=caveat, dataset=dataset,
            title="Held-out parity by reaction class -- AE | barriers | "
                  "total")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return out_path


def _stamp_parity_footer(fig, *, run_id: str, title: str, note: str,
                         provenance: Optional[str], caveat: Optional[str],
                         dataset: Optional[str] = None,
                         extra_note: Optional[str] = None) -> None:
    """``extra_note`` adds a SECOND red footer line above ``note`` (used for the
    unequal-training-depth disclosure). Left None -- the default -- the footer
    stack sits exactly where it did before the parameter existed."""
    if _run_predates_vxc_fix(run_id):
        # Every figure of a pre-correction run carries the V_xc provenance,
        # bar panels or not; post-fix runs stamp nothing.
        note = (note + "  " if note else "") + _VXC_DISCLOSURE
    fig.suptitle(f"{title}  ·  {run_id}", fontsize=11.5, y=0.997)
    if caveat:
        fig.text(0.5, 0.945, caveat, ha="center", fontsize=7.5, style="italic",
                 color="#444444")
    if dataset:
        # what the eval set IS (live counts) -- sits between the caveat and
        # the axes; None (the default) renders every legacy figure unchanged
        fig.text(0.5, 0.922, dataset, ha="center", fontsize=5.6,
                 color="#555555")
    if extra_note:
        fig.text(0.5, 0.030, extra_note, ha="center", fontsize=5.6,
                 color="#a33", wrap=True)
    if note:
        fig.text(0.5, 0.032 if not extra_note else 0.058, note, ha="center",
                 fontsize=5.6, color="#a33", wrap=True)
    fig.text(0.5, 0.010, provenance or _PROVENANCE_BASE, ha="center",
             fontsize=5.6, color="#777777")


def _add_subset_colorbar(fig, mappable, *, x=0.945):
    if mappable is None:
        return
    cax = fig.add_axes([x, 0.22, 0.012, 0.50])
    cbar = fig.colorbar(mappable, cax=cax)
    cbar.set_label("subset_size", fontsize=7)
    cbar.ax.tick_params(labelsize=6)


def plot_parity_marginal(rows: List[Dict[str, Any]], out_path: Path, run_id: str,
                         note: str = "", provenance: Optional[str] = None,
                         caveat: Optional[str] = None,
                         dataset: Optional[str] = None) -> Path:
    """L1 -- compact 2x2: rows = pool (own scale); col0 by ARCH, col1 by
    SUBSET (viridis). Arch & subset as separate marginal views."""
    with plt.rc_context(_STYLE):
        pools = _present_pools(rows) or ["bh76"]
        subset_values = _present_subsets(rows)
        fig, axes = plt.subplots(len(pools), 2, figsize=(12, 5.3 * len(pools)),
                                 squeeze=False)
        mappable = None
        for i, pool in enumerate(pools):
            lim = _pool_parity_limits(rows, pool)
            pr = [r for r in rows if r.get("pool") == pool]
            _parity_scatter(axes[i][0], pr, color_by="arch", limits=lim)
            _, mp = _parity_scatter(axes[i][1], pr, color_by="subset",
                                    limits=lim, subset_values=subset_values)
            mappable = mp or mappable
            axes[i][0].set_ylabel(
                f"{_POOL_LABEL[pool]}\nNN reaction energy (kcal/mol)", fontsize=8)
            for j, sub in enumerate(("by architecture", "by training subset_size")):
                axes[i][j].set_title(f"({pool}) {sub}", fontsize=9)
                axes[i][j].set_xlabel("reference reaction energy (kcal/mol)",
                                      fontsize=8)
        fig.legend(handles=_arch_pbe_legend_handles(_archs_present(rows)),
                   loc="lower center", ncol=len(arch_style.RUNG_ORDER),
                   fontsize=7, frameon=False,
                   bbox_to_anchor=(0.5, 0.052))
        _stamp_parity_footer(fig, run_id=run_id, note=note,
                             provenance=provenance, dataset=dataset,
                             caveat=caveat,
                             title="Reaction-energy parity -- marginal (arch | subset)")
        fig.tight_layout(rect=(0, 0.085, 0.92, 0.915))
        _add_subset_colorbar(fig, mappable)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return out_path


def plot_parity_facet_subset(rows: List[Dict[str, Any]], out_path: Path,
                             run_id: str, note: str = "",
                             provenance: Optional[str] = None,
                             caveat: Optional[str] = None,
                             dataset: Optional[str] = None) -> Path:
    """L2 -- rows = pool x cols = subset_size; arch = color within each facet.
    Joint arch x subset."""
    with plt.rc_context(_STYLE):
        pools = _present_pools(rows) or ["bh76"]
        subset_values = _present_subsets(rows) or [1]
        nr, nc = len(pools), len(subset_values)
        fig, axes = plt.subplots(nr, nc, figsize=(2.7 * nc + 1.2, 3.6 * nr + 1.0),
                                 squeeze=False)
        for i, pool in enumerate(pools):
            lim = _pool_parity_limits(rows, pool)
            for j, s in enumerate(subset_values):
                pr = [r for r in rows if r.get("pool") == pool
                      and r.get("subset_size") == s]
                _parity_scatter(axes[i][j], pr, color_by="arch", limits=lim,
                                point_size=8)
                if i == 0:
                    axes[i][j].set_title(f"subset={s}", fontsize=8)
                if i == nr - 1:
                    axes[i][j].set_xlabel("reference (kcal/mol)", fontsize=6.5)
                if j == 0:
                    axes[i][j].set_ylabel(f"{_POOL_LABEL[pool]}\nNN de", fontsize=7)
                axes[i][j].tick_params(labelsize=6)
        fig.legend(handles=_arch_pbe_legend_handles(_archs_present(rows)),
                   loc="lower center", ncol=len(arch_style.RUNG_ORDER),
                   fontsize=7, frameon=False,
                   bbox_to_anchor=(0.5, 0.05))
        _stamp_parity_footer(fig, run_id=run_id, note=note,
                             provenance=provenance, dataset=dataset,
                             caveat=caveat,
                             title="Reaction-energy parity -- pool x subset facets (arch=color)")
        fig.tight_layout(rect=(0, 0.085, 1, 0.915))
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return out_path


def plot_parity_arch_cols(rows: List[Dict[str, Any]], out_path: Path,
                          run_id: str, note: str = "",
                          provenance: Optional[str] = None,
                          caveat: Optional[str] = None,
                          dataset: Optional[str] = None) -> Path:
    """L3 -- rows = pool x cols = arch; subset_size = viridis within each panel.
    All subsets per arch, individually colored."""
    with plt.rc_context(_STYLE):
        pools = _present_pools(rows) or ["bh76"]
        archs = _archs_present(rows) or ["deep"]
        subset_values = _present_subsets(rows)
        nr, nc = len(pools), len(archs)
        fig, axes = plt.subplots(nr, nc, figsize=(3.2 * nc + 1.2, 3.8 * nr + 1.0),
                                 squeeze=False)
        mappable = None
        for i, pool in enumerate(pools):
            lim = _pool_parity_limits(rows, pool)
            for j, a in enumerate(archs):
                pr = [r for r in rows if r.get("pool") == pool
                      and r.get("arch") == a]
                _, mp = _parity_scatter(axes[i][j], pr, color_by="subset",
                                        limits=lim, subset_values=subset_values,
                                        point_size=9)
                mappable = mp or mappable
                if i == 0:
                    axes[i][j].set_title(a, fontsize=8)
                if i == nr - 1:
                    axes[i][j].set_xlabel("reference (kcal/mol)", fontsize=6.5)
                if j == 0:
                    axes[i][j].set_ylabel(f"{_POOL_LABEL[pool]}\nNN de", fontsize=7)
                axes[i][j].tick_params(labelsize=6)
        fig.legend(handles=[plt.Line2D([], [], marker="x", ls="", color="0.5",
                                       label="PBE")],
                   loc="lower center", fontsize=7, frameon=False,
                   bbox_to_anchor=(0.5, 0.05))
        _stamp_parity_footer(fig, run_id=run_id, note=note,
                             provenance=provenance, dataset=dataset,
                             caveat=caveat,
                             title="Reaction-energy parity -- pool x arch panels (subset=viridis)")
        fig.tight_layout(rect=(0, 0.075, 0.92, 0.915))
        _add_subset_colorbar(fig, mappable)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return out_path


def plot_parity_errbars_by_subset(rows: List[Dict[str, Any]], out_path: Path,
                                  run_id: str, note: str = "",
                                  provenance: Optional[str] = None,
                                  caveat: Optional[str] = None,
                                  dataset: Optional[str] = None) -> Path:
    """L4 -- 3x2 by subset_size: each subplot = AGGREGATE parity, one marker per
    (arch, pool) at (mean ref, mean de_nn) with a vertical error bar = that
    cell's reaction-energy MAE. Pool by marker, arch by color, y=x line."""
    with plt.rc_context(_STYLE):
        pools = _present_pools(rows) or ["bh76"]
        archs = _archs_present(rows) or ["deep"]
        subset_values = _present_subsets(rows) or [1]
        # aggregate per (subset, arch, pool): (mean_ref, mean_de_nn, mae)
        agg: Dict[Tuple[int, str, str], Tuple[float, float, float]] = {}
        for s in subset_values:
            for a in archs:
                for pool in pools:
                    cell = [r for r in rows if r.get("subset_size") == s
                            and r.get("arch") == a and r.get("pool") == pool
                            # x=mean(ref) and y=mean(de_nn) must average the
                            # SAME rows; NN-NaN slice rows stay out of both
                            and _is_num(r.get("de_nn_kcalmol"))]
                    refs = [r["ref_kcalmol"] for r in cell if _is_num(r.get("ref_kcalmol"))]
                    des = [r["de_nn_kcalmol"] for r in cell if _is_num(r.get("de_nn_kcalmol"))]
                    maes = [r["abs_error_nn_kcalmol"] for r in cell
                            if _is_num(r.get("abs_error_nn_kcalmol"))]
                    if refs and des:
                        agg[(s, a, pool)] = (float(np.mean(refs)),
                                             float(np.mean(des)),
                                             float(np.mean(maes)) if maes else 0.0)
        gv: List[float] = []
        for mref, mde, mae in agg.values():
            gv += [mref, mde, mde - mae, mde + mae]
        glim = _robust_limits(gv, q=(0.0, 100.0))
        n = len(subset_values)
        ncols = 2 if n > 1 else 1
        nrows = max(1, math.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(6.0 * ncols, 4.1 * nrows),
                                 squeeze=False)
        flat = axes.ravel()
        for idx, s in enumerate(subset_values):
            ax = flat[idx]
            xs, ys = [], []
            for a in archs:
                for pool in pools:
                    if (s, a, pool) not in agg:
                        continue
                    mref, mde, mae = agg[(s, a, pool)]
                    ax.errorbar(mref, mde, yerr=mae, fmt=POOL_MARKER[pool],
                                color=ARCH_COLOR[a], ms=7, capsize=3,
                                elinewidth=1.0, alpha=0.9, zorder=3)
                    xs.append(mref); ys.append(mde)
            anchor = list(glim) if glim else []
            _diagonal(ax, xs + anchor, ys + anchor, limits=glim)
            ax.set_title(f"subset_size = {s}", fontsize=9)
            ax.set_xlabel("mean reference (kcal/mol)", fontsize=7)
            ax.set_ylabel("mean NN reaction energy +- MAE (kcal/mol)", fontsize=7)
            ax.tick_params(labelsize=6)
        for k in range(n, len(flat)):
            flat[k].axis("off")
        fig.legend(handles=_arch_pbe_legend_handles(archs, pools=pools),
                   loc="lower center", ncol=len(arch_style.RUNG_ORDER),
                   fontsize=7, frameon=False,
                   bbox_to_anchor=(0.5, 0.05))
        _stamp_parity_footer(fig, run_id=run_id, note=note,
                             provenance=provenance, dataset=dataset,
                             caveat=caveat,
                             title="Reaction-energy parity + error bars -- 3x2 by subset (aggregate)")
        fig.tight_layout(rect=(0, 0.085, 1, 0.915))
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return out_path


def plot_parity_grid_by_subset(rows: List[Dict[str, Any]], out_path: Path,
                               run_id: str, note: str = "",
                               provenance: Optional[str] = None,
                               caveat: Optional[str] = None,
                               dataset: Optional[str] = None) -> Path:
    """L5 -- 6x2 grid: rows = subset_size, cols = pool (BH76 | W4-11). Each cell
    = per-reaction parity (arch=color), robust window per pool-column. Each
    subset ROW carries one combined-MAE-per-arch inset on its W4-11 panel."""
    with plt.rc_context(_STYLE):
        pools = _present_pools(rows) or ["bh76"]
        archs = _archs_present(rows) or ["deep"]
        subset_values = _present_subsets(rows) or [1]
        nr, nc = len(subset_values), len(pools)
        fig, axes = plt.subplots(nr, nc, figsize=(5.0 * nc + 0.6, 3.3 * nr),
                                 squeeze=False)
        col_lims = {pool: _pool_parity_limits(rows, pool) for pool in pools}
        inset_col = nc - 1
        for i, s in enumerate(subset_values):
            for j, pool in enumerate(pools):
                pr = [r for r in rows if r.get("subset_size") == s
                      and r.get("pool") == pool]
                _parity_scatter(axes[i][j], pr, color_by="arch",
                                limits=col_lims[pool], point_size=8)
                if i == 0:
                    axes[i][j].set_title(_POOL_LABEL[pool], fontsize=9)
                if i == nr - 1:
                    axes[i][j].set_xlabel("reference (kcal/mol)", fontsize=6.5)
                if j == 0:
                    axes[i][j].set_ylabel(f"subset={s}\nNN de", fontsize=7)
                axes[i][j].tick_params(labelsize=6)
            rows_s = [r for r in rows if r.get("subset_size") == s]
            pbe_s = _mae([r["abs_error_pbe_kcalmol"] for r in rows_s])
            _combined_mae_inset(axes[i][inset_col], rows_s, archs, pbe_s)
        fig.legend(handles=_arch_pbe_legend_handles(archs),
                   loc="lower center", ncol=len(arch_style.RUNG_ORDER),
                   fontsize=7, frameon=False,
                   bbox_to_anchor=(0.5, 0.04))
        _stamp_parity_footer(fig, run_id=run_id, note=note,
                             provenance=provenance, dataset=dataset,
                             caveat=caveat,
                             title="Reaction-energy parity -- 6x2 subset x pool, per-subset combined-MAE inset")
        fig.tight_layout(rect=(0, 0.065, 1, 0.93))
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return out_path


def build_parity_variants(run_dir: Path, outdir: Path,
                          eval_subdir: str = "eval_holdout") -> List[Path]:
    """Render all five parity-layout candidates into ``outdir`` for comparison."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = collect_holdout_reaction_rows(run_dir, eval_subdir=eval_subdir)
    run_id = f"{run_dir.name} · {_ckpt_label(eval_subdir)}"
    note = coverage_note(run_dir, eval_subdir=eval_subdir)
    try:
        baseline = pbe_pool_baseline(run_dir, eval_subdir=eval_subdir)
    except Exception as exc:  # pool unavailable
        print(f"  (PBE baseline unavailable: {exc})")
        baseline = {"bh76": float("nan"), "w411": float("nan"),
                    "combined": float("nan")}
    prov = provenance_footer(baseline)
    caveat = nn_vs_pbe_caveat(rows, baseline)
    ds_e = _holdout_eval_note(rows, [])
    variants = [
        (plot_parity_arch_cols, "ablation_parity_arch_cols.png"),
        (plot_parity_marginal, "ablation_parity_marginal_2x2.png"),
        (plot_parity_facet_subset, "ablation_parity_facet_subset.png"),
        (plot_parity_errbars_by_subset, "ablation_parity_errbars_by_subset.png"),
        (plot_parity_grid_by_subset, "ablation_parity_grid_by_subset.png"),
    ]
    written: List[Path] = []
    for fn, name in variants:
        written.append(fn(rows, outdir / name, run_id, note=note,
                          provenance=prov, caveat=caveat, dataset=ds_e))
    return written


# ---------------------------------------------------------------------------
# 2-subset WTMAD-2 energy metric + in-sample density-vs-CCSD diagnostic
# ---------------------------------------------------------------------------
# Energy: a 2-subset (BH76 / W4-11) WTMAD-2 (GMTKN55 Eq.14 style) that rebalances
# the ~16x BH76-vs-W4-11 magnitude gap a plain combined MAE buries -- a LABELED
# reweighting, NOT a full 55-subset GMTKN55 WTMAD-2. Density: the in-sample
# (training-set) density error vs the CCSD reference, the actual density training
# signal (Dick & Fernandez-Serra). Held-out density does not exist yet (no CCSD
# reference densities for the held-out pool), so the two are kept SEPARATE.

_GMTKN55_SCALE = 56.84  # kcal/mol, global mean |dE| over GMTKN55 (Goerigk 2017)


def _dedup_rows_by_name(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """One row per reaction name (PBE de is spec-invariant), so a pooled PBE
    baseline isn't multiply-counted across specs."""
    seen: set = set()
    out: List[Dict[str, Any]] = []
    for r in rows:
        nm = r.get("name")
        if nm in seen:
            continue
        seen.add(nm)
        out.append(r)
    return out


def _wtmad2_over_pools(pool_rows: Dict[str, List[Tuple[float, float]]],
                       scale: float) -> Optional[float]:
    """WTMAD-2 = (scale/N_total) * sum_i N_i * MAD_i/|ref|_i over the pool buckets,
    each bucket a list of (|err|, |ref|). None if a bucket has zero |ref| mean or
    nothing finite."""
    n_total = sum(len(v) for v in pool_rows.values())
    if n_total == 0:
        return None
    acc = 0.0
    for vals in pool_rows.values():
        n_i = len(vals)
        absref_i = sum(rf for _, rf in vals) / n_i
        if absref_i <= 0:
            return None
        mad_i = sum(e for e, _ in vals) / n_i
        acc += n_i * mad_i / absref_i
    return scale / n_total * acc


def wtmad2_by_arch_subset(rows: List[Dict[str, Any]], scale: float = _GMTKN55_SCALE
                          ) -> Dict[Tuple[str, int], float]:
    """2-subset (BH76/W4-11) WTMAD-2 per (arch, subset_size) cell, vs the
    benchmark ``reaction_energy_ref_kcalmol``. NOTE: only 2 GMTKN55 subsets are
    present here, so this is a LABELED reweighting, not a full-GMTKN55 WTMAD-2.
    Deduplicated by reaction name WITHIN each cell (first FINITE pair wins,
    finiteness tested before the name slot is consumed -- a NaN first
    instance of a duplicated name cannot discard its finite twin), matching
    the PBE baselines' convention -- the pool lists four reactions twice
    under one name."""
    cells: Dict[Tuple[str, int], Dict[str, List[Tuple[float, float]]]] = {}
    seen: set = set()
    for r in rows:
        a, s, pool = r.get("arch"), r.get("subset_size"), r.get("pool")
        e = r.get("abs_error_nn_kcalmol")
        ref = r.get("ref_kcalmol")           # key used by collect_holdout_reaction_rows
        if ref is None:
            ref = r.get("reaction_energy_ref_kcalmol")   # raw per_reaction.json key
        if a is None or s is None or pool is None:
            continue
        if not (_is_num(e) and _is_num(ref)):
            continue
        nm = r.get("name")
        if nm is not None:
            if (a, s, nm) in seen:
                continue
            seen.add((a, s, nm))
        cells.setdefault((a, s), {}).setdefault(pool, []).append((abs(e), abs(ref)))
    out: Dict[Tuple[str, int], float] = {}
    for cell, pools in cells.items():
        w = _wtmad2_over_pools(pools, scale)
        if w is not None:
            out[cell] = w
    return out


def wtmad2_pbe_by_arch_subset(rows: List[Dict[str, Any]],
                              scale: float = _GMTKN55_SCALE
                              ) -> Dict[Tuple[str, int], float]:
    """Per-cell PBE 2-subset WTMAD-2 over the cell's FULL TEST SLICE (every
    slice reaction with a finite PBE leg, independent of NN convergence) --
    the anchor for beats-PBE verdicts. Cells score training-subset-dependent
    slices (their own trained reactions are excluded), so the pooled-union
    anchor over- or under-states PBE on a given cell's slice; and a
    reference reduction never follows a single arch's NN-scored subset --
    an NN SCF failure must not move the reference."""
    out: Dict[Tuple[str, int], float] = {}
    by_cell: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for r in rows:
        a, s = r.get("arch"), r.get("subset_size")
        if a is None or s is None:
            continue
        by_cell.setdefault((a, s), []).append(r)
    for cell, cell_rows in by_cell.items():
        w = wtmad2_pbe_baseline(cell_rows, scale)
        if _is_num(w):
            out[cell] = w
    return out


def pbe_reaction_mae_by_cell(rows: List[Dict[str, Any]]
                             ) -> Dict[Tuple[str, int], float]:
    """Per-cell PBE reaction MAE over that cell's full test slice
    (name-dedup; finite PBE legs, independent of NN convergence) -- the
    MAE-leg twin of :func:`wtmad2_pbe_by_arch_subset`."""
    return reaction_mae_by_arch_subset(rows, key="abs_error_pbe_kcalmol")


def wtmad2_pbe_baseline(rows: List[Dict[str, Any]], scale: float = _GMTKN55_SCALE
                        ) -> float:
    """2-subset WTMAD-2 for the PBE baseline over the held-out pool (dedup by
    reaction name). A single reference value for the figure's dashed line."""
    pools: Dict[str, List[Tuple[float, float]]] = {}
    for r in _dedup_rows_by_name(rows):
        pool = r.get("pool")
        e = r.get("abs_error_pbe_kcalmol")
        ref = r.get("ref_kcalmol")
        if ref is None:
            ref = r.get("reaction_energy_ref_kcalmol")
        if pool is None or not (_is_num(e) and _is_num(ref)):
            continue
        pools.setdefault(pool, []).append((abs(e), abs(ref)))
    w = _wtmad2_over_pools(pools, scale)
    return w if w is not None else float("nan")


def scan_reaction_errors(run_dir: Path, *, basis: Optional[str] = None,
                         cache_dir: Optional[Path] = None, _loader=None,
                         _energies: Optional[Dict[str, float]] = None
                         ) -> Dict[str, float]:
    """``{reaction_name: |SCAN error| kcal/mol}`` from the SCAN energy cache.

    The per-reaction form behind :func:`wtmad2_scan_baseline`, computed with the
    same validated reaction math the NN/PBE legs use
    (``eval_holdout.per_reaction_errors``). ``{}`` when the cache is absent."""
    scan = (_energies if _energies is not None
            else _scan_energies(run_dir, basis=basis, cache_dir=cache_dir))
    if not scan:
        return {}
    if _loader is None:
        from xcquinox.alec.full_benchmark_pools import load_full_held_out_pools
        _loader = load_full_held_out_pools
    from xcquinox.alec.eval_holdout import per_reaction_errors
    _, full_rxns = _loader()
    out: Dict[str, float] = {}
    for r in per_reaction_errors(scan, list(full_rxns)):
        name, err = r.get("name"), r.get("abs_error_kcalmol")
        if name is not None and _is_num(err):
            out[str(name)] = float(err)
    return out


def wtmad2_scan_baseline(rows: List[Dict[str, Any]],
                         scan_errors: Optional[Dict[str, float]],
                         scale: float = _GMTKN55_SCALE
                         ) -> Tuple[float, int, int]:
    """``(WTMAD-2, n_used, n_reference)`` for SCAN over the SAME reactions
    :func:`wtmad2_pbe_baseline` reduces.

    The reaction set comes from ``rows`` (the deduped held-out eval), NOT from
    the SCAN cache, so the two reference lines reduce the same benchmark; a
    reaction SCAN could not score is counted as missing rather than quietly
    shrinking the set. Rows without a finite ``abs_error_pbe_kcalmol`` are
    excluded entirely (from ``n_ref`` AND the average) -- they are not part
    of the set :func:`wtmad2_pbe_baseline` reduces, and counting them in
    ``n_ref`` alone would let unscored-by-both rows push the coverage
    fraction under the floor while the MAE twin
    (:func:`scan_reaction_mae_baseline`, which always had the gate) stays
    above it. NaN when nothing usable."""
    pools: Dict[str, List[Tuple[float, float]]] = {}
    n_ref = 0
    for r in _dedup_rows_by_name(rows):
        pool = r.get("pool")
        ref = r.get("ref_kcalmol")
        if ref is None:
            ref = r.get("reaction_energy_ref_kcalmol")
        if pool is None or not _is_num(ref):
            continue
        if not _is_num(r.get("abs_error_pbe_kcalmol")):
            continue
        n_ref += 1
        e = (scan_errors or {}).get(str(r.get("name")))
        if not _is_num(e):
            continue
        pools.setdefault(pool, []).append((abs(e), abs(ref)))
    n_used = sum(len(v) for v in pools.values())
    w = _wtmad2_over_pools(pools, scale)
    return (w if w is not None else float("nan")), n_used, n_ref


def scan_reaction_mae_baseline(rows: List[Dict[str, Any]],
                               scan_errors: Optional[Dict[str, float]]
                               ) -> Tuple[float, int, int]:
    """``(combined reaction MAE, n_used, n_reference)`` for SCAN over the SAME
    deduped reactions :func:`pbe_reaction_mae_baseline` reduces -- the MAE-leg
    twin of :func:`wtmad2_scan_baseline`, so the headline ED figure's mae leg
    can carry a like-for-like SCAN comparator. NaN when nothing usable."""
    errs: List[float] = []
    n_ref = 0
    for r in _dedup_rows_by_name(rows):
        if not _is_num(r.get("abs_error_pbe_kcalmol")):
            continue
        n_ref += 1
        e = (scan_errors or {}).get(str(r.get("name")))
        if _is_num(e):
            errs.append(abs(e))
    return ((float(np.mean(errs)) if errs else float("nan")),
            len(errs), n_ref)


def scan_row_matched_ref(reaction_rows: List[Dict[str, Any]],
                         scan_errors: Optional[Dict[str, float]],
                         scan_baseline: Optional[Dict[str, Any]]
                         ) -> Tuple[Optional[float], str]:
    """``(value, label)`` for the per-arch MAE figure's SCAN reference line.

    Preferred: :func:`scan_reaction_mae_baseline` over the SAME deduped rows
    the figure's PBE line averages -- a like-for-like pair -- when the cache
    covers at least :data:`_SCAN_COVERAGE_FLOOR` of THIS figure's rows
    (partial coverage is qualified ``, u/r`` in the label). The gate is on
    the figure's own row set, so a row-matched line can draw even where the
    POOLED line's coverage floor would withdraw it -- the rows here are the
    figure's benchmark, fully covered or gated on their own terms. Otherwise
    the full-pool fallback with its previous label (absent cache ->
    ``(None, "")`` and no line is drawn), so the figure never silently mixes
    a row-matched PBE line with a different-set SCAN one when the
    row-matched reduction is available."""
    val, used, ref = scan_reaction_mae_baseline(reaction_rows, scan_errors)
    if _is_num(val) and ref > 0 and used / ref >= _SCAN_COVERAGE_FLOOR:
        sfx = f", {used}/{ref}" if used < ref else ""
        return float(val), f"SCAN row-matched MAE ({val:.1f}{sfx})"
    scan_c, scan_cov = scan_line_value(scan_baseline, "combined")
    if scan_c is None:
        return None, ""
    return scan_c, f"SCAN full-pool MAE ({scan_c:.1f}{scan_cov})"


def wtmad2_scan_by_cell(rows: List[Dict[str, Any]],
                        scan_errors: Optional[Dict[str, float]],
                        scale: float = _GMTKN55_SCALE
                        ) -> Dict[Tuple[str, int], float]:
    """Per-cell SCAN 2-subset WTMAD-2 over the cell's full test slice,
    coverage-gated PER CELL at :data:`_SCAN_COVERAGE_FLOOR` (the floor now
    measures the cache's coverage of the slice) -- the SCAN twin of
    :func:`wtmad2_pbe_by_arch_subset`. Cells whose slice the cache covers
    too thinly are absent (their marks are withdrawn; the pooled comparator
    lines remain for scale only)."""
    out: Dict[Tuple[str, int], float] = {}
    by_cell: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for r in rows:
        a, s = r.get("arch"), r.get("subset_size")
        if a is None or s is None:
            continue
        by_cell.setdefault((a, s), []).append(r)
    for cell, cell_rows in by_cell.items():
        v, used, ref = wtmad2_scan_baseline(cell_rows, scan_errors, scale)
        if _is_num(v) and ref and (used / ref) >= _SCAN_COVERAGE_FLOOR:
            out[cell] = float(v)
    return out


def scan_reaction_mae_by_cell(rows: List[Dict[str, Any]],
                              scan_errors: Optional[Dict[str, float]]
                              ) -> Dict[Tuple[str, int], float]:
    """Per-cell SCAN reaction MAE over the cell's full test slice,
    coverage-gated per cell -- the MAE-leg twin of
    :func:`wtmad2_scan_by_cell`."""
    out: Dict[Tuple[str, int], float] = {}
    by_cell: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
    for r in rows:
        a, s = r.get("arch"), r.get("subset_size")
        if a is None or s is None:
            continue
        by_cell.setdefault((a, s), []).append(r)
    for cell, cell_rows in by_cell.items():
        v, used, ref = scan_reaction_mae_baseline(cell_rows, scan_errors)
        if _is_num(v) and ref and (used / ref) >= _SCAN_COVERAGE_FLOOR:
            out[cell] = float(v)
    return out


def scan_density_by_cell(hd_rows: List[Dict[str, Any]],
                         records: Optional[Dict[str, Dict[str, Any]]],
                         pbe_table: Optional[Dict[str, Dict[str, float]]]
                         = None, *, nn_key: str = "density_rmse",
                         pbe_key: str = "density_rmse_pbe",
                         _pbe_mol: Optional[Dict[str, float]] = None
                         ) -> Dict[Tuple[str, int], float]:
    """Per-cell SCAN density anchor over exactly the species each cell's PBE
    density anchor averages (the comparator species set -- membership in the
    PBE map, independent of the NN leg), coverage-gated per cell via
    :func:`scan_density_line` -- the SCAN twin of
    :func:`pbe_density_by_cell`. ``nn_key`` is kept for call-site symmetry
    but no longer keys the set."""
    del nn_key  # comparator coverage is NN-independent
    pbe_mol = (_pbe_mol if _pbe_mol is not None
               else _pbe_density_map(hd_rows, pbe_table, key=pbe_key))
    by_cell: Dict[Tuple[str, int], set] = {}
    for r in hd_rows:
        a, s = r.get("arch"), r.get("subset_size")
        if a is None or s is None:
            continue
        m = r.get("molecule")
        if m in pbe_mol:
            by_cell.setdefault((a, s), set()).add(m)
    out: Dict[Tuple[str, int], float] = {}
    for cell, mols in by_cell.items():
        v = scan_density_line(records, mols, key=pbe_key)
        if _is_num(v):
            out[cell] = float(v)
    return out


def collect_insample_density_rows(run_dir: Path) -> List[Dict[str, Any]]:
    """In-sample density-vs-CCSD errors from ``eval/per_molecule.json``: trained
    multi-atom species carrying a finite ``density_rmse`` (atoms are skipped at
    eval time -> None), joined with the manifest arch/subset_size. Read directly
    (not via ``ccp.collect_per_molecule_rows``, which drops ``ref_density_method``)."""
    cells = ccp._read_manifest_cells(run_dir)
    rows: List[Dict[str, Any]] = []
    for idx, spec_dir in ccp._spec_dirs(run_dir):
        pm_path = spec_dir / "eval" / "per_molecule.json"
        if not pm_path.is_file():
            continue
        try:
            with pm_path.open() as f:
                payload = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        cell = cells.get(idx, {})
        for r in payload:
            if r.get("skipped") or not _is_num(r.get("density_rmse")):
                continue
            rows.append({
                "idx": idx,
                "arch": cell.get("arch"),
                "subset_size": cell.get("subset_size"),
                "molecule": r.get("molecule"),
                "density_rmse": r.get("density_rmse"),
                "density_l1": r.get("density_l1"),
                # PBE-vs-CCSD baseline on the same grid (model-free; emitted by
                # newer evals only -- None on older per_molecule.json)
                "density_rmse_pbe": r.get("density_rmse_pbe"),
                "density_l1_pbe": r.get("density_l1_pbe"),
                "density_eps_l1": r.get("density_eps_l1"),
                "density_eps_l1_pbe": r.get("density_eps_l1_pbe"),
                "ref_density_method": r.get("ref_density_method"),
                "from_training_subset": r.get("from_training_subset"),
            })
    return rows


# Cross-spec relative spread above which a species' PBE density reference is
# reference-inconsistent (the c2 class: two arms carrying incompatible c2
# references differ 11x; within-arm scatter is ~0). The energy-side twin of
# this guard is _first_pbe_energies' absolute 1e-4 Ha tolerance.
_PBE_DENSITY_CONSISTENCY_REL = 0.05


def _inconsistent_pbe_density_species(raw_rows: List[Dict[str, Any]]
                                      ) -> Dict[str, Tuple[float, float]]:
    """``{molecule: (min, max)}`` for species whose model-free PBE density
    reference disagrees across specs beyond
    :data:`_PBE_DENSITY_CONSISTENCY_REL` on EITHER error channel (RMSE or
    Eq. 20 eps). Such a species' reference is broken (the c2 drift class):
    averaging it into anchors weights the candidates by pull coverage, and
    its NN column is judged against whichever reference its arm carries."""
    out: Dict[str, Tuple[float, float]] = {}
    for key in ("density_rmse_pbe", "density_eps_l1_pbe"):
        acc: Dict[str, List[float]] = {}
        for r in raw_rows:
            m = r.get("molecule")
            if m and _is_num(r.get(key)):
                acc.setdefault(str(m), []).append(float(r[key]))
        for m, vals in acc.items():
            lo, hi = min(vals), max(vals)
            if hi > 0.0 and (hi - lo) / hi > _PBE_DENSITY_CONSISTENCY_REL:
                out.setdefault(m, (lo, hi))
    return out


def _pbe_density_outlier_clauses(raw_rows: List[Dict[str, Any]],
                                 bad: Dict[str, Tuple[float, float]]
                                 ) -> Dict[str, str]:
    """Per-flagged-molecule spec attribution for the density-reference
    warning, from the first error channel whose relative spread trips the
    guard (matching :func:`_inconsistent_pbe_density_species`); the clause is
    tagged with that channel. Rows whose spec index is missing are omitted
    from the attribution (they cannot be named); if the attributable rows
    alone no longer trip the guard, the molecule gets no clause and the
    range-only warning stands."""
    out: Dict[str, str] = {}
    for m in bad:
        for key in ("density_rmse_pbe", "density_eps_l1_pbe"):
            pairs = [((f"spec_{r['idx']:04d}"
                       if isinstance(r.get("idx"), int)
                       else str(r.get("idx"))), float(r[key]))
                     for r in raw_rows
                     if str(r.get("molecule")) == m and _is_num(r.get(key))
                     and r.get("idx") is not None]
            if not pairs:
                continue
            nums = [v for _spec, v in pairs]
            lo, hi = min(nums), max(nums)
            if hi > 0.0 and (hi - lo) / hi > _PBE_DENSITY_CONSISTENCY_REL:
                clusters = _value_clusters(
                    pairs, hi * _PBE_DENSITY_CONSISTENCY_REL)
                out[m] = (_outlier_clause(clusters, fmt="{:.6g}")
                          + f" [{key}]")
                break
    return out


def collect_holdout_density_rows(run_dir: Path,
                                 eval_subdir: str = "eval_holdout"
                                 ) -> List[Dict[str, Any]]:
    """Held-out density-vs-CCSD errors from ``<eval_subdir>/per_molecule.json``
    (the un-stubbed NN ``density_rmse`` + model-free ``density_rmse_pbe``
    columns; both None until benchmark CCSD reference densities are wired),
    joined with the manifest arch/subset_size. Rows are kept when EITHER
    channel is finite, so a PBE-only re-eval still produces the baseline.

    Two repairs applied on read (printed when they fire): rows for species
    that are pool twins of the spec's trained molecules under a different
    name are dropped (``_spec_alias_names`` -- the held-out density mean must
    not average supervised species), and rows for species whose model-free
    PBE reference disagrees across specs
    (``_inconsistent_pbe_density_species``, the c2 reference-drift class) are
    dropped entirely -- from the anchors AND the per-cell means -- so no
    anchor can drift with pull coverage."""
    cells = ccp._read_manifest_cells(run_dir)
    raw: List[Dict[str, Any]] = []
    n_alias = 0
    alias_hits: set = set()
    for idx, spec_dir in ccp._spec_dirs(run_dir):
        pm_path = spec_dir / eval_subdir / "per_molecule.json"
        if not pm_path.is_file():
            continue
        try:
            with pm_path.open() as f:
                payload = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        cell = cells.get(idx, {})
        aliases_cf = _spec_alias_names(spec_dir)
        for r in payload:
            if not (_is_num(r.get("density_rmse"))
                    or _is_num(r.get("density_rmse_pbe"))):
                continue
            mol = r.get("molecule")
            if aliases_cf and str(mol).casefold() in aliases_cf:
                n_alias += 1
                alias_hits.add(str(mol))
                continue
            raw.append({
                "idx": idx,
                "arch": cell.get("arch"),
                "subset_size": cell.get("subset_size"),
                "molecule": mol,
                "density_rmse": r.get("density_rmse"),
                "density_l1": r.get("density_l1"),
                "density_rmse_pbe": r.get("density_rmse_pbe"),
                "density_l1_pbe": r.get("density_l1_pbe"),
                # DFS Eq. 20 per-electron L1 columns (emitted by newer evals;
                # None on older pulls)
                "density_eps_l1": r.get("density_eps_l1"),
                "density_eps_l1_pbe": r.get("density_eps_l1_pbe"),
                "ref_density_method": r.get("ref_density_method"),
                "from_training_subset": r.get("from_training_subset"),
            })
    if n_alias:
        print(f"  (strict-holdout repair: dropped {n_alias} density rows for "
              f"trained species under pool names {sorted(alias_hits)})")
    bad = _inconsistent_pbe_density_species(raw)
    if bad:
        clauses = _pbe_density_outlier_clauses(raw, bad)
        detail = ", ".join(
            f"{m} ({lo:.6g}..{hi:.6g}"
            + (f"; {clauses[m]}" if m in clauses else "") + ")"
            for m, (lo, hi) in sorted(bad.items()))
        print("  (WARNING: cross-spec-inconsistent PBE density reference -- "
              f"excluding from all density anchors and cell means: {detail})")
        raw = [r for r in raw if str(r.get("molecule")) not in bad]
    return raw


def load_pbe_density_table(run_dir: Path) -> Dict[str, Dict[str, float]]:
    """``{molecule: {density_rmse_pbe, density_l1_pbe}}`` from the run-level
    ``pbe_density_errors.json`` written by ``reeval_holdout_fixed.py
    --pbe-density-only`` (model-free, shared across every spec/arch of the
    run). Empty dict when absent."""
    p = Path(run_dir) / "pbe_density_errors.json"
    if not p.is_file():
        return {}
    try:
        return dict(json.loads(p.read_text()).get("errors", {}))
    except (json.JSONDecodeError, OSError):
        return {}


# ---------------------------------------------------------------------------
# DFS Eq. 21 combined energy-density metric (ED)
# ---------------------------------------------------------------------------
# ED combines the held-out energy error with the held-out density-vs-CCSD error
# in the metric of Dick & Fernandez-Serra, PRB 104, L161109 (2021), Eq. 21: the
# harmonic mean of an energy error and a density error rescaled to an energy,
# ED = 2/(1/E + 1/(gamma*D)). Three deviations from the Letter, all stamped on
# the figure:
#   * gamma is SELF-CALIBRATED per energy leg from the pooled PBE anchors
#     (gamma = E_PBE/D_PBE, so ED_PBE == E_PBE by construction). The Letter's
#     gamma = 1084.87 kcal/mol is the slope of a zero-intercept regression of
#     WTMAD-2 on its per-electron L1 density error (Eq. 20) across six
#     nonempirical functionals, and is dimensionally tied to those units --
#     applying it to the grid-weight-averaged RMSE stored by the eval pipeline
#     would be wrong.
#   * the energy legs are the suite's 2-subset WTMAD-2 (Eq. 19 form over
#     BH76+W4-11 only, a labeled reweighting) and the plain combined reaction
#     MAE -- not the Letter's diet-GMTKN55-150 WTMAD-2.
#   * the density leg is the grid-weight-averaged RMSE vs CCSD (not CCSD(T))
#     emitted by the eval pipeline, not Eq. 20's per-electron L1. The Letter's
#     SI Sec. VI reports the ranking "largely independent of the density error
#     metric chosen" (its L2 variant, SI Eq. 8, correlates best with WTMAD-2,
#     R^2 = 0.98), and gamma absorbs the unit change.
# Semantics match ``combined_energy_density`` in
# notebooks/dfs_selfconsistent_density/dfs_demo.py, reimplemented here because
# dfs_demo pulls in the TestSpec/xcquinox import chain (too heavy for a
# plotting-only script).


def holdout_density_by_arch_subset(hd_rows: List[Dict[str, Any]],
                                   key: str = "density_rmse"
                                   ) -> Dict[Tuple[str, int], float]:
    """``{(arch, subset_size): mean held-out density error}`` over rows with
    a finite NN channel -- the D leg of ED. ``key`` selects the error column
    (default the grid-weighted RMSE; ``density_eps_l1`` gives the DFS Eq. 20
    per-electron L1 when the eval emitted it). Same bucketing rule as
    ``reaction_mae_by_arch_subset``; atoms never contribute (their density
    columns are None at eval time)."""
    buckets: Dict[Tuple[str, int], List[float]] = {}
    for r in hd_rows:
        arch, ss = r.get("arch"), r.get("subset_size")
        if arch is None or ss is None:
            continue
        if _is_num(r.get(key)):
            buckets.setdefault((arch, ss), []).append(r[key])
    return {k: float(np.mean(v)) for k, v in buckets.items() if v}


def _pbe_density_map(hd_rows: List[Dict[str, Any]],
                     pbe_table: Optional[Dict[str, Dict[str, float]]] = None,
                     key: str = "density_rmse_pbe") -> Dict[str, float]:
    """Per-molecule PBE density-error map behind ``pbe_density_baseline`` and
    ``_pbe_anchor_coverage_warning``: the run-level table when given (finite
    entries only), else the per-molecule mean of the inline ``key`` columns
    (default the grid-weighted RMSE; ``density_eps_l1_pbe`` selects the DFS
    Eq. 20 per-electron L1 twin)."""
    pbe_mol: Dict[str, float] = {
        m: d[key] for m, d in (pbe_table or {}).items()
        if _is_num(d.get(key))}
    if not pbe_mol:
        acc: Dict[str, List[float]] = {}
        for r in hd_rows:
            if _is_num(r.get(key)) and r.get("molecule"):
                acc.setdefault(r["molecule"], []).append(r[key])
        pbe_mol = {m: float(np.mean(v)) for m, v in acc.items()}
    return pbe_mol


def pbe_density_baseline(hd_rows: List[Dict[str, Any]],
                         pbe_table: Optional[Dict[str, Dict[str, float]]] = None,
                         key: str = "density_rmse_pbe") -> float:
    """Pooled PBE density-vs-CCSD anchor ``D_PBE``: the run-level
    ``pbe_density_errors.json`` table when given, else the per-molecule mean of
    the inline ``key`` columns (default the grid-weighted RMSE;
    ``density_eps_l1_pbe`` selects the DFS Eq. 20 per-electron L1 anchor);
    then the mean over molecules. The per-molecule dedup matters -- the PBE
    channel is model-free and identical across specs, so a row-weighted mean
    would multiply-count each molecule by its spec coverage. NaN when nothing
    finite (older evals without the PBE columns), mirroring
    ``wtmad2_pbe_baseline``'s NaN-degrade convention. An anchor set wider or
    narrower than the NN density union is flagged by
    ``_pbe_anchor_coverage_warning``."""
    pbe_mol = _pbe_density_map(hd_rows, pbe_table, key=key)
    return float(np.mean(list(pbe_mol.values()))) if pbe_mol else float("nan")


def _pbe_anchor_coverage_warning(hd_rows: List[Dict[str, Any]],
                                 pbe_table: Optional[Dict[str, Dict[str, float]]]
                                 = None, *, nn_key: str = "density_rmse",
                                 pbe_key: str = "density_rmse_pbe") -> str:
    """'' when the PBE density anchor's molecule set equals the set of
    molecules with a finite NN density; otherwise names the symmetric
    difference. Guards ``D_PBE`` against silently averaging species the NN
    legs never see -- possible when a run-level ``pbe_density_errors.json``
    spans more species than the NN eval, or when the NN channel failed for
    rows whose PBE column survived. ``nn_key``/``pbe_key`` select the error
    channel (defaults: the RMSE pair; the ``density_eps_l1`` pair guards the
    DFS-units legs, where a partial backfill shrinks both sets)."""
    anchor = set(_pbe_density_map(hd_rows, pbe_table, key=pbe_key))
    nn = {r.get("molecule") for r in hd_rows if _is_num(r.get(nn_key))}
    if not anchor or anchor == nn:
        return ""

    def _fmt(ms: List[str]) -> str:
        shown = ", ".join(ms[:6])
        return shown + (f" (+{len(ms) - 6} more)" if len(ms) > 6 else "")

    parts = []
    extra = sorted(m for m in anchor - nn if m)
    missing = sorted(m for m in nn - anchor if m)
    if extra:
        parts.append("anchor-only: " + _fmt(extra))
    if missing:
        parts.append("NN-only: " + _fmt(missing))
    return ("PBE density anchor set differs from the NN density union -- "
            + "; ".join(parts) + ".")


def pbe_density_by_cell(hd_rows: List[Dict[str, Any]],
                        pbe_table: Optional[Dict[str, Dict[str, float]]]
                        = None, *, nn_key: str = "density_rmse",
                        pbe_key: str = "density_rmse_pbe",
                        _pbe_mol: Optional[Dict[str, float]] = None
                        ) -> Dict[Tuple[str, int], float]:
    """Per-cell PBE density anchor over the cell's comparator species set --
    every held-out species present in the per-molecule PBE map, independent
    of the NN density leg (the PBE column is model-free, so an NN-failed
    species still belongs to the slice) -- from the same per-molecule PBE
    map as the pooled anchor: the density leg of the cell-slice beats-PBE
    verdict. ``nn_key`` is kept for call-site symmetry with the NN means but
    no longer keys the set. ``_pbe_mol`` injects a prebuilt per-molecule map
    (the 3x3's channel-filtered map) so channel views reuse exactly the map
    their pooled anchor averaged."""
    del nn_key  # comparator coverage is NN-independent
    pbe_mol = (_pbe_mol if _pbe_mol is not None
               else _pbe_density_map(hd_rows, pbe_table, key=pbe_key))
    by_cell: Dict[Tuple[str, int], set] = {}
    for r in hd_rows:
        a, s = r.get("arch"), r.get("subset_size")
        if a is None or s is None:
            continue
        m = r.get("molecule")
        if m in pbe_mol:
            by_cell.setdefault((a, s), set()).add(m)
    out: Dict[Tuple[str, int], float] = {}
    for cell, mols in by_cell.items():
        # Sorted species order: np.mean over a hash-ordered set permutes the
        # fp summation between processes (PYTHONHASHSEED), moving the cell
        # anchors by ulps between renders of identical data. The SCAN twin
        # (scan_density_line) already sorts.
        vals = [pbe_mol[m] for m in sorted(mols, key=str) if m in pbe_mol]
        if vals:
            out[cell] = float(np.mean(vals))
    return out


def pbe_reaction_mae_baseline(rows: List[Dict[str, Any]]) -> float:
    """Pooled PBE reaction MAE (kcal/mol) over the held-out rows, dedup by
    reaction name -- the MAE-leg anchor ``E_PBE``, the same arithmetic as the
    inline expression in ``plot_energy_wtmad_mae``. NOT ``pbe_pool_baseline``'s
    'combined': that one spans the full canonical pool (including reactions
    absent from the pulled ``per_reaction.json``), and gamma self-calibration
    needs ``E_PBE`` on the same reaction set as the per-cell ``E_NN``. NaN when
    empty."""
    val = _mae([r.get("abs_error_pbe_kcalmol")
                for r in _dedup_rows_by_name(rows)])
    return float("nan") if val is None else float(val)


def _harmonic_mean(a: float, b: float) -> float:
    """``2/(1/a + 1/b)``; 0.0 when either input is non-positive or non-finite
    (the guard of dfs_demo's ``combined_energy_density`` -- a leg at or below
    zero has no harmonic-mean meaning and would otherwise divide by zero)."""
    if not (_is_num(a) and _is_num(b)) or a <= 0.0 or b <= 0.0:
        return 0.0
    return 2.0 / (1.0 / a + 1.0 / b)


def _cell_pbe_ed(cell: Tuple[str, int], gamma: float,
                 ed_pooled: Optional[float],
                 e_by_cell: Optional[Dict[Tuple[str, int], float]],
                 d_by_cell: Optional[Dict[Tuple[str, int], float]]
                 ) -> Tuple[Optional[float], Optional[float]]:
    """``(verdict anchor, ed_cell-or-None)`` for one cell of a reference
    functional: the harmonic ED of the CELL-SLICE legs under the summary's
    gamma when both are available and positive, else the pooled fallback
    (which may itself be None -- comparator withdrawn). Cells score
    training-subset-dependent slices, so a verdict against the pooled
    anchor misgrades cells whose own slice is easier or harder for the
    reference than the union. Shared by the PBE and SCAN legs."""
    if e_by_cell is None or d_by_cell is None:
        return ed_pooled, None
    e_c = e_by_cell.get(cell)
    d_c = d_by_cell.get(cell)
    if (_is_num(e_c) and float(e_c) > 0.0
            and _is_num(d_c) and float(d_c) > 0.0):
        ed_c = _harmonic_mean(float(e_c), gamma * float(d_c))
        if ed_c > 0.0:
            return ed_c, ed_c
    return ed_pooled, None


def combined_ed_by_cell(energy_by_cell: Dict[Tuple[str, int], float],
                        e_pbe: float,
                        density_by_cell: Dict[Tuple[str, int], float],
                        d_pbe: float,
                        e_scan: Optional[float] = None,
                        d_scan: Optional[float] = None,
                        e_pbe_by_cell: Optional[
                            Dict[Tuple[str, int], float]] = None,
                        d_pbe_by_cell: Optional[
                            Dict[Tuple[str, int], float]] = None,
                        e_scan_by_cell: Optional[
                            Dict[Tuple[str, int], float]] = None,
                        d_scan_by_cell: Optional[
                            Dict[Tuple[str, int], float]] = None
                        ) -> Dict[str, Any]:
    """DFS Eq. 21 ED per (arch, subset_size) cell (kcal/mol), NN vs PBE.

    ``gamma = e_pbe / d_pbe`` (self-calibrated; see the section note above for
    why the Letter's fixed 1084.87 kcal/mol cannot be used here), so
    ``ed_pbe == e_pbe`` by construction and every cell's
    ``ED = 2/(1/E + 1/(gamma*D))`` shares the PBE kcal/mol scale. Only cells
    with a finite value in BOTH maps are emitted -- callers surface the
    excluded cells via ``_ed_exclusion_note``. Raises ValueError on
    non-finite/non-positive anchors (no silent NaN ED).

    ``e_scan`` / ``d_scan`` add the meta-GGA comparator: ``ed_scan`` is the SAME
    harmonic mean under the SAME gamma, so it is on the cells' scale and can be
    drawn as a second reference. Unlike ``ed_pbe`` it is NOT equal to its energy
    leg -- gamma is calibrated on PBE, not on SCAN, so SCAN's density leg moves
    it. ``None`` when either SCAN leg is missing, so the figures omit the line.

    ``e_pbe_by_cell`` / ``d_pbe_by_cell`` supply CELL-MATCHED PBE reductions
    (PBE over exactly the reactions/species each cell scored): the per-cell
    ``beats_pbe`` verdict is then judged against ``ed_pbe_cell`` (stored on
    the cell, under the same gamma) instead of the pooled anchor, which over-
    or under-states PBE on cells whose scored subset differs from the union.
    ``e_scan_by_cell`` / ``d_scan_by_cell`` are the SCAN twins behind
    ``beats_scan`` / ``ed_scan_cell`` -- the pooled SCAN comparator carries
    the same misgrading (on the current data it understates SCAN on every
    cell's surviving set).
    """
    if not (_is_num(e_pbe) and e_pbe > 0.0):
        raise ValueError(
            f"PBE energy anchor must be finite and positive, got {e_pbe!r}")
    if not (_is_num(d_pbe) and d_pbe > 0.0):
        raise ValueError(
            f"PBE density anchor must be finite and positive, got {d_pbe!r}")
    gamma = float(e_pbe) / float(d_pbe)
    ed_pbe = _harmonic_mean(float(e_pbe), gamma * float(d_pbe))
    ed_scan = None
    if (_is_num(e_scan) and float(e_scan) > 0.0
            and _is_num(d_scan) and float(d_scan) > 0.0):
        ed_scan = _harmonic_mean(float(e_scan), gamma * float(d_scan))
    cells: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for cell, e in energy_by_cell.items():
        d = density_by_cell.get(cell)
        if not (_is_num(e) and _is_num(d)):
            continue
        ed = _harmonic_mean(float(e), gamma * float(d))
        anchor, ed_pbe_cell = _cell_pbe_ed(cell, gamma, ed_pbe,
                                           e_pbe_by_cell, d_pbe_by_cell)
        s_anchor, ed_scan_cell = _cell_pbe_ed(cell, gamma, ed_scan,
                                              e_scan_by_cell, d_scan_by_cell)
        cells[cell] = {"E": float(e), "D": float(d),
                       "gammaD": gamma * float(d), "ED": ed,
                       "beats_pbe": bool(ed < anchor),
                       "ed_pbe_cell": ed_pbe_cell,
                       "ed_scan_cell": ed_scan_cell,
                       "beats_scan": (bool(ed < s_anchor)
                                      if s_anchor is not None else None)}
    return {"gamma": gamma, "gamma_mode": "self_calibrated",
            "e_pbe": float(e_pbe), "d_pbe": float(d_pbe),
            "ed_pbe": ed_pbe,
            "e_scan": (float(e_scan) if _is_num(e_scan) else None),
            "d_scan": (float(d_scan) if _is_num(d_scan) else None),
            "ed_scan": ed_scan, "cells": cells}


# The Letter's published conversion slope (kcal/mol per unit of its Eq. 20
# per-electron L1 density error): the zero-intercept regression of WTMAD-2 on
# eps across PW91/PBE/TPSS/revTPSS/SCAN/PBE0 (Dick & Fernandez-Serra, PRB 104,
# L161109 (2021), Fig. 3). Dimensionally valid ONLY against density errors in
# the same Eq. 20 units (the density_eps_l1 columns) -- never against the
# grid-weighted RMSE.
_DFS_GAMMA_KCAL = 1084.87

# DFS-paper notation (PRB 104, L161109): the combined energy-density metric
# is the CALLIGRAPHIC ED -- generic form (main text, "we propose a metric
# ED"), and ED_{|n|} when its density leg is the Eq. 20 per-electron L1
# eps_{|n|} (Eq. 21, Table I, Fig. 2's ylabel). Figure text uses the
# paper's symbols; CSV column names keep their ASCII schema.
_ED_SYM = r"$\mathcal{ED}$"
_ED_N_SYM = r"$\mathcal{ED}_{|n|}$"
_EPS_N_SYM = r"$\varepsilon_{|n|}$"
_EPS_N_EQ = r"$\varepsilon_{|n|} = \frac{1}{N_e}\int|n - n_{ref}|$"


def gamma_zero_intercept(pairs) -> float:
    """Zero-intercept least-squares slope of W on eps over ``(eps, W)`` pairs:
    ``gamma = sum(eps*W) / sum(eps^2)`` -- the Letter's Fig. 3 regression
    procedure. NaN when no pair has a positive eps."""
    num = 0.0
    den = 0.0
    for eps, wt in pairs:
        if _is_num(eps) and _is_num(wt) and eps > 0.0:
            num += eps * wt
            den += eps * eps
    return num / den if den > 0.0 else float("nan")


def combined_ed_fixed_gamma(energy_by_cell: Dict[Tuple[str, int], float],
                            e_pbe: float,
                            density_by_cell: Dict[Tuple[str, int], float],
                            d_pbe: float, gamma: float, *,
                            gamma_source: Optional[str] = None,
                            e_scan: Optional[float] = None,
                            d_scan: Optional[float] = None,
                            e_pbe_by_cell: Optional[
                                Dict[Tuple[str, int], float]] = None,
                            d_pbe_by_cell: Optional[
                                Dict[Tuple[str, int], float]] = None,
                            e_scan_by_cell: Optional[
                                Dict[Tuple[str, int], float]] = None,
                            d_scan_by_cell: Optional[
                                Dict[Tuple[str, int], float]] = None
                            ) -> Dict[str, Any]:
    """DFS Eq. 21 ED per cell with an EXTERNALLY FIXED gamma (the Letter's
    published 1084.87 on Eq. 20 units, or an own-axes regression slope from
    the nonempirical pool cache) -- unlike ``combined_ed_by_cell``, gamma is
    NOT derived from the anchors, so ``ed_pbe`` is generally NOT equal to
    ``e_pbe`` and PBE sits off the y=x locus wherever it sits off the
    calibration trend. Same summary contract as ``combined_ed_by_cell``
    except ``gamma_mode = "fixed"`` -- the ED panels branch their stamps and
    PBE labels on that key, so a fixed-gamma summary never renders the
    self-calibration claims (ED_PBE = E_PBE, PBE-on-y=x), which are false
    here. ``gamma_source`` (e.g. "DFS published" / "own-axes fit") is stored
    additively when given, and the gamma stamp names it -- the value alone
    cannot tell a reader WHICH external gamma a panel plots."""
    if not (_is_num(e_pbe) and e_pbe > 0.0):
        raise ValueError(
            f"PBE energy anchor must be finite and positive, got {e_pbe!r}")
    if not (_is_num(d_pbe) and d_pbe > 0.0):
        raise ValueError(
            f"PBE density anchor must be finite and positive, got {d_pbe!r}")
    if not (_is_num(gamma) and gamma > 0.0):
        raise ValueError(f"gamma must be finite and positive, got {gamma!r}")
    ed_pbe = _harmonic_mean(float(e_pbe), gamma * float(d_pbe))
    ed_scan = None
    if (_is_num(e_scan) and float(e_scan) > 0.0
            and _is_num(d_scan) and float(d_scan) > 0.0):
        ed_scan = _harmonic_mean(float(e_scan), gamma * float(d_scan))
    cells: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for cell, e in energy_by_cell.items():
        d = density_by_cell.get(cell)
        if not (_is_num(e) and _is_num(d)):
            continue
        ed = _harmonic_mean(float(e), gamma * float(d))
        anchor, ed_pbe_cell = _cell_pbe_ed(cell, float(gamma), ed_pbe,
                                           e_pbe_by_cell, d_pbe_by_cell)
        s_anchor, ed_scan_cell = _cell_pbe_ed(cell, float(gamma), ed_scan,
                                              e_scan_by_cell, d_scan_by_cell)
        cells[cell] = {"E": float(e), "D": float(d),
                       "gammaD": gamma * float(d), "ED": ed,
                       "beats_pbe": bool(ed < anchor),
                       "ed_pbe_cell": ed_pbe_cell,
                       "ed_scan_cell": ed_scan_cell,
                       "beats_scan": (bool(ed < s_anchor)
                                      if s_anchor is not None else None)}
    out: Dict[str, Any] = {"gamma": float(gamma), "gamma_mode": "fixed",
                           "e_pbe": float(e_pbe), "d_pbe": float(d_pbe),
                           "ed_pbe": ed_pbe,
                           "e_scan": (float(e_scan) if _is_num(e_scan) else None),
                           "d_scan": (float(d_scan) if _is_num(d_scan) else None),
                           "ed_scan": ed_scan, "cells": cells}
    if gamma_source:
        out["gamma_source"] = gamma_source
    return out


def _nonempirical_cache_name(basis: str) -> str:
    """Filename of the nonempirical-functional pool cache at ``basis`` --
    kept identical to ``precompute_nonempirical_pool._pool_cache_name``."""
    b = (basis or "def2-svp").replace("+DF", "").strip() or "def2-svp"
    safe = "".join(c if (c.isalnum() or c in "-.+") else "_" for c in b)
    return f"nonempirical_pool_{safe}.json"


def _wtmad2_from_energies(energies: Dict[str, float],
                          _loader=None) -> float:
    """2-subset WTMAD-2 for one functional from its ``{species: E_tot(Ha)}``
    map over the canonical full held-out pool -- the validated reaction math
    in ``eval_holdout.per_reaction_errors``, bucketed by ``source_pool`` and
    reweighted by :func:`_wtmad2_over_pools` (lazy pool import; ``_loader``
    is the test seam). NaN when no reaction is computable."""
    if _loader is None:
        from xcquinox.alec.full_benchmark_pools import load_full_held_out_pools
        _loader = load_full_held_out_pools
    from xcquinox.alec.eval_holdout import per_reaction_errors
    _, rxns = _loader()
    rxns = list(rxns)
    pools: Dict[str, List[Tuple[float, float]]] = {}
    for rxn, er in zip(rxns, per_reaction_errors(energies, rxns)):
        if math.isfinite(er["abs_error_kcalmol"]):
            pools.setdefault(rxn.get("source_pool"), []).append(
                (er["abs_error_kcalmol"], abs(er["ref_kcalmol"])))
    w = _wtmad2_over_pools(pools, _GMTKN55_SCALE)
    return float(w) if w is not None else float("nan")


def nonempirical_gamma(run_dir: Path, *, basis: Optional[str] = None,
                       cache_dir: Optional[Path] = None,
                       _wtmad: Optional[Dict[str, float]] = None
                       ) -> Dict[str, Any]:
    """The DFS-procedure gamma on OUR axes: load the nonempirical pool cache
    (``precompute_nonempirical_pool.py``), build per-functional
    ``(eps, WTMAD-2)`` pairs, and fit the zero-intercept slope.

    A partially-filled cache (timed-out job, per-species failures) leaves the
    functionals with UNEQUAL species support; fitting each functional's mean
    over its own set would bias the slope by coverage, not physics. So eps is
    averaged over the COMMON species intersection across all functionals, and
    the real WTMAD-2 path restricts each functional's energies to the common
    energy-species set (making the computable reaction list identical across
    functionals). Coverage is reported: ``n_species`` (the intersection) and
    ``n_species_dropped`` (union minus intersection; nonzero means partial
    support, worth stating wherever the fitted gamma is used). Returns
    ``{gamma, pairs, n_functionals, n_species, n_species_dropped}`` or ``{}``
    when the cache is absent/empty/malformed or the intersection is empty.
    ``_wtmad`` injects per-functional WTMAD-2 values directly (test seam).
    Cache-name note: like the SCAN cache, the slug drops ``+DF``, so DF and
    non-DF runs at one basis share the calibration cache (the DF error is far
    below the eps signal, the same trade ``run_pbe_density_table`` makes)."""
    b = basis or run_basis_label(Path(run_dir)) or "def2-svp"
    candidates = [Path(d) / _nonempirical_cache_name(b)
                  for d in (cache_dir, run_dir) if d is not None]
    cache = None
    for p in candidates:
        if p.is_file():
            try:
                cache = json.loads(p.read_text())
                break
            except (json.JSONDecodeError, OSError):
                cache = None
    if not cache or not isinstance(cache, dict):
        return {}
    by_xc: Dict[str, Dict[str, float]] = {}
    energies: Dict[str, Dict[str, float]] = {}
    for name, per_xc in cache.items():
        if not isinstance(per_xc, dict):
            continue
        for xc, entry in per_xc.items():
            if not isinstance(entry, dict):
                continue
            if _is_num(entry.get("density_eps_l1")):
                by_xc.setdefault(xc, {})[name] = entry["density_eps_l1"]
            if _is_num(entry.get("e_tot")):
                energies.setdefault(xc, {})[name] = entry["e_tot"]
    if not by_xc:
        return {}
    common = set.intersection(*(set(v) for v in by_xc.values()))
    if not common:
        return {}
    union = set().union(*(set(v) for v in by_xc.values()))
    e_common: set = set()
    if _wtmad is None:
        e_sets = [set(energies.get(xc, {})) for xc in by_xc]
        e_common = set.intersection(*e_sets) if all(e_sets) else set()
    pairs: Dict[str, Tuple[float, float]] = {}
    for xc, eps_by_mol in by_xc.items():
        # sorted: set iteration is hash-ordered and float summation is
        # order-dependent -- an unsorted mean makes the fitted gamma's last
        # ulp vary across processes
        eps = float(np.mean([eps_by_mol[n] for n in sorted(common)]))
        if _wtmad is not None:
            wt = _wtmad.get(xc, float("nan"))
        else:
            wt = _wtmad2_from_energies(
                {n: e for n, e in energies.get(xc, {}).items()
                 if n in e_common})
        if _is_num(eps) and _is_num(wt):
            pairs[xc] = (eps, wt)
    if not pairs:
        return {}
    return {"gamma": gamma_zero_intercept(pairs.values()),
            "pairs": pairs, "n_functionals": len(pairs),
            "n_species": len(common),
            "n_species_dropped": len(union) - len(common)}


def _cell_tag(cell: Tuple[str, int]) -> str:
    return f"{cell[0]}/ss{cell[1]}"


def _ed_exclusion_note(energy_by_cell: Dict[Tuple[str, int], float],
                       density_by_cell: Dict[Tuple[str, int], float]) -> str:
    """Names every cell excluded from ED -- present in only one leg, or keyed
    but non-finite in either -- mirroring ``combined_ed_by_cell``'s
    finite-in-both emission rule so nothing drops silently; '' when every
    keyed cell is emitted."""
    e_ok = {c for c, v in energy_by_cell.items() if _is_num(v)}
    d_ok = {c for c, v in density_by_cell.items() if _is_num(v)}
    e_only = sorted(e_ok - d_ok)
    d_only = sorted(d_ok - e_ok)
    nonfinite = sorted((set(energy_by_cell) | set(density_by_cell))
                       - e_ok - d_ok)
    if not e_only and not d_only and not nonfinite:
        return ""
    parts = []
    if e_only:
        parts.append("energy-only: " + ", ".join(_cell_tag(c) for c in e_only))
    if d_only:
        parts.append("density-only: " + ", ".join(_cell_tag(c) for c in d_only))
    if nonfinite:
        parts.append("non-finite: " + ", ".join(_cell_tag(c)
                                                for c in nonfinite))
    return ("ED excludes cells lacking a usable leg -- "
            + "; ".join(parts) + ".")


def _density_cell_coverage_warning(hd_rows: List[Dict[str, Any]],
                                   key: str = "density_rmse") -> str:
    """'' when every (arch, subset_size) cell's set of finite-NN density
    species equals the pooled union; otherwise names the divergent cells with
    their counts. Cells are expected to share one held-out species set -- a
    partial eval would silently bias that cell's mean D. ``key`` selects the
    error channel (default the grid-weighted RMSE; ``density_eps_l1`` runs
    the same homogeneity check on the DFS-units channel, where a per-species
    partial backfill is the realistic cause)."""
    per_cell: Dict[Tuple[str, int], set] = {}
    for r in hd_rows:
        arch, ss = r.get("arch"), r.get("subset_size")
        if arch is None or ss is None or not _is_num(r.get(key)):
            continue
        per_cell.setdefault((arch, ss), set()).add(r.get("molecule"))
    if not per_cell:
        return ""
    union = set().union(*per_cell.values())
    bad = {c: s for c, s in per_cell.items() if s != union}
    if not bad:
        return ""
    frag = ", ".join(f"{_cell_tag(c)} n={len(s)}"
                     for c, s in sorted(bad.items()))
    return (f"per-cell density species differ from the pooled union "
            f"(n={len(union)}): {frag}.")


def _incomplete_energy_cells(rows: List[Dict[str, Any]]
                             ) -> Dict[Tuple[str, int],
                                       Tuple[int, int, List[str]]]:
    """``{(arch, subset_size): (n_scored, n_slice, missing_names)}`` for
    cells whose NN-scored reaction names are a PROPER subset of the cell's
    test slice (the finite-comparator rows). Per-cell, within-cell: energy
    slices legitimately differ across subset sizes (each excludes its own
    trained reactions), so no cross-cell union is implied -- unlike
    :func:`_density_cell_coverage_warning`. Name-level sets: a name counts
    as scored when any of its rows carries a finite NN leg, mirroring
    :func:`_cell_counts`."""
    slice_names: Dict[Tuple[str, int], set] = {}
    nn_names: Dict[Tuple[str, int], set] = {}
    for r in rows:
        a, s, nm = r.get("arch"), r.get("subset_size"), r.get("name")
        if a is None or s is None or nm is None:
            continue
        if _is_num(r.get("abs_error_pbe_kcalmol")):
            slice_names.setdefault((a, s), set()).add(nm)
        if _is_num(r.get("abs_error_nn_kcalmol")):
            nn_names.setdefault((a, s), set()).add(nm)
    out: Dict[Tuple[str, int], Tuple[int, int, List[str]]] = {}
    for cell, sl in slice_names.items():
        nn = nn_names.get(cell, set())
        missing = sorted(str(m) for m in (sl - nn))
        if missing:
            out[cell] = (len(nn & sl), len(sl), missing)
    return out


def _energy_cell_coverage_warning(rows: List[Dict[str, Any]], *,
                                  max_names: int = 6) -> str:
    """'' when every cell's NN scored its whole test slice; otherwise names
    each shortfall cell with scored/slice counts and the missing reaction
    names (capped at ``max_names``). NN-side only by construction: a missing
    COMPARATOR leg changes the slice itself, which the per-bar span fallback
    and the cross-spec PBE consistency guard surface -- not this warning."""
    bad = _incomplete_energy_cells(rows)
    if not bad:
        return ""
    frags = []
    for cell, (n_scored, n_slice, missing) in sorted(bad.items()):
        shown = ", ".join(missing[:max_names])
        if len(missing) > max_names:
            shown += f" (+{len(missing) - max_names} more)"
        frags.append(f"{_cell_tag(cell)} {n_scored}/{n_slice} "
                     f"(missing: {shown})")
    return ("incomplete hold-out eval (NN leg unscored on part of the "
            "slice) -- " + "; ".join(frags) + ".")


def _spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Spearman rank correlation via double argsort + Pearson on the ranks.
    NaN for n < 2, length mismatch, or a constant series; ties are not
    rank-averaged (ED values are continuous -- exact ties arise only from
    degenerate inputs)."""
    xv = [float(x) for x in xs]
    yv = [float(y) for y in ys]
    if len(xv) < 2 or len(xv) != len(yv):
        return float("nan")
    if len(set(xv)) < 2 or len(set(yv)) < 2:
        return float("nan")
    rx = np.argsort(np.argsort(xv))
    ry = np.argsort(np.argsort(yv))
    return float(np.corrcoef(rx, ry)[0, 1])


def _cell_counts(rows: List[Dict[str, Any]], key: str
                 ) -> Dict[Tuple[str, int], int]:
    """``{(arch, subset_size): #rows with a finite key}`` -- the n_* columns
    of the ED CSV. Rows carrying a reaction ``name`` are deduplicated by
    name within each cell, matching the effective N of the deduped cell
    metrics (the pool lists four reactions twice under one name); rows
    without a ``name`` (the per-molecule density rows) count as before."""
    out: Dict[Tuple[str, int], int] = {}
    seen: set = set()
    for r in rows:
        arch, ss = r.get("arch"), r.get("subset_size")
        if arch is None or ss is None or not _is_num(r.get(key)):
            continue
        nm = r.get("name")
        if nm is not None:
            if (arch, ss, nm) in seen:
                continue
            seen.add((arch, ss, nm))
        out[(arch, ss)] = out.get((arch, ss), 0) + 1
    return out


def _holdout_eval_note(rows: List[Dict[str, Any]],
                       hd_rows: List[Dict[str, Any]]) -> str:
    """One-line description of WHAT the held-out eval is, with live counts:
    name-deduplicated reactions per pool (from the per_reaction rows) and the
    density-species coverage (from the per_molecule rows). Stamped on the
    held-out figures via the footer's ``dataset`` line so each figure is
    self-describing. '' when both inputs are empty; either clause is omitted
    when its input is absent."""
    pool_label = {"bh76": "BH76", "w411": "W4-11"}
    parts: List[str] = []
    pools: Dict[str, int] = {}
    for r in _dedup_rows_by_name(rows):
        p = r.get("pool")
        if p:
            pools[p] = pools.get(p, 0) + 1
    if pools:
        frag = " + ".join(f"{pool_label.get(p, str(p).upper())} {n}"
                          for p, n in sorted(pools.items()))
        parts.append(f"{frag} reactions (name-dedup; reaction energies, "
                     "kcal/mol)")
    n_nn = len({r.get("molecule") for r in hd_rows
                if r.get("molecule") and _is_num(r.get("density_rmse"))})
    n_pbe = len({r.get("molecule") for r in hd_rows
                 if r.get("molecule") and _is_num(r.get("density_rmse_pbe"))})
    if n_nn or n_pbe:
        cnt = (f"{n_nn} species" if n_nn == n_pbe
               else f"{n_nn} NN / {n_pbe} PBE species")
        parts.append(f"density: {cnt} vs CCSD refs at matching basis/grid "
                     "(atoms excluded)")
    if not parts:
        return ""
    return "Held-out eval: " + "; ".join(parts) + "."


def _species_pools(rows: List[Dict[str, Any]]) -> Dict[str, set]:
    """``{molecule: {pools}}`` from the held-out reaction rows' reactants and
    products. Species appearing in reactions of both pools (the BH76/W4-11
    overlap) map to both -- the per-channel density panels show them in each
    channel, stated on the figure. Pool-less rows are ignored."""
    out: Dict[str, set] = {}
    for r in rows:
        p = r.get("pool")
        if not p:
            continue
        for sp in list(r.get("reactants") or []) + list(r.get("products") or []):
            if sp:
                out.setdefault(sp, set()).add(p)
    return out


def channel_ed_summaries(rows: List[Dict[str, Any]],
                         hd_rows: List[Dict[str, Any]],
                         pbe_table: Optional[Dict[str, Dict[str, float]]]
                         = None, *,
                         fixed_gamma: Optional[float] = None,
                         gamma_source: Optional[str] = None,
                         density_key: str = "density_rmse",
                         pbe_density_key: str = "density_rmse_pbe",
                         scan_errors: Optional[Dict[str, float]] = None,
                         scan_density_records: Optional[
                             Dict[str, Dict[str, Any]]] = None
                         ) -> Dict[str, Optional[Dict[str, Any]]]:
    """Per-channel DFS Eq. 21 summaries for ``bh76`` / ``w411`` / ``combined``.

    Each channel filters the reaction rows by pool (its energy leg is then the
    one-bucket WTMAD-2 reduction; the combined channel is the genuine 2-subset
    form) and the density rows by species membership (``_species_pools``;
    overlap species contribute to both channels). By default each channel's
    gamma is self-calibrated from ITS OWN pool-filtered PBE anchors, so EDs
    are comparable within a channel, not across channels. With ``fixed_gamma``
    every channel shares that external gamma (``gamma_mode="fixed"``
    summaries) -- EDs then DO compare across channels; the DFS-units twins
    pass the published slope with ``density_key="density_eps_l1"`` /
    ``pbe_density_key="density_eps_l1_pbe"`` so D is the Letter's Eq. 20
    per-electron L1. A channel whose anchors are missing/non-positive maps to
    None (callers render placeholders). When a run-level ``pbe_table`` is
    given but carries no finite entries for the chosen key on a channel's
    species, that channel's density anchor falls back to the inline PBE
    columns (the ``pbe_density_baseline`` contract)."""
    pools_of = _species_pools(rows)
    out: Dict[str, Optional[Dict[str, Any]]] = {}
    for ch in ("bh76", "w411", "combined"):
        if ch == "combined":
            ch_rows, ch_hd, ch_tab = rows, hd_rows, pbe_table
        else:
            ch_rows = [r for r in rows if r.get("pool") == ch]
            ch_hd = [r for r in hd_rows
                     if ch in pools_of.get(r.get("molecule"), ())]
            ch_tab = None
            if pbe_table:
                ch_tab = {m: v for m, v in pbe_table.items()
                          if ch in pools_of.get(m, ())}
        e_cells = wtmad2_by_arch_subset(ch_rows)
        e_pbe = wtmad2_pbe_baseline(ch_rows)
        d_cells = holdout_density_by_arch_subset(ch_hd, key=density_key)
        d_pbe = pbe_density_baseline(ch_hd, ch_tab, key=pbe_density_key)
        # cell-matched anchors for the beats verdicts, on this channel's
        # own reactions/species (same reductions as the pooled anchors above)
        e_pbe_cells = wtmad2_pbe_by_arch_subset(ch_rows)
        d_pbe_cells = pbe_density_by_cell(ch_hd, ch_tab, nn_key=density_key,
                                          pbe_key=pbe_density_key)
        e_scan_cells = wtmad2_scan_by_cell(ch_rows, scan_errors)
        d_scan_cells = scan_density_by_cell(ch_hd, scan_density_records,
                                            ch_tab, nn_key=density_key,
                                            pbe_key=pbe_density_key)
        # SCAN comparator legs on THIS channel: its WTMAD-2 over the same
        # reactions PBE reduced, and its density over the same species the PBE
        # anchor averaged. Either missing (or too thinly covered) -> None, and
        # the summary's ed_scan is None so the panels omit the line.
        e_scan, e_used, e_ref = wtmad2_scan_baseline(ch_rows, scan_errors)
        if not (e_ref and (e_used / e_ref) >= _SCAN_COVERAGE_FLOOR):
            e_scan = None
        d_scan, d_used, d_ref = scan_density_line_counts(
            scan_density_records, _pbe_density_map(ch_hd, ch_tab,
                                                   key=pbe_density_key),
            key=pbe_density_key)
        if (e_cells and _is_num(e_pbe) and e_pbe > 0.0 and d_cells
                and _is_num(d_pbe) and d_pbe > 0.0):
            out[ch] = (combined_ed_fixed_gamma(e_cells, e_pbe, d_cells,
                                               d_pbe, fixed_gamma,
                                               gamma_source=gamma_source,
                                               e_scan=e_scan, d_scan=d_scan,
                                               e_pbe_by_cell=e_pbe_cells,
                                               d_pbe_by_cell=d_pbe_cells,
                                               e_scan_by_cell=e_scan_cells,
                                               d_scan_by_cell=d_scan_cells)
                       if fixed_gamma is not None else
                       combined_ed_by_cell(e_cells, e_pbe, d_cells, d_pbe,
                                           e_scan=e_scan, d_scan=d_scan,
                                           e_pbe_by_cell=e_pbe_cells,
                                           d_pbe_by_cell=d_pbe_cells,
                                           e_scan_by_cell=e_scan_cells,
                                           d_scan_by_cell=d_scan_cells))
            if out[ch].get("ed_scan") is not None:
                out[ch]["scan_suffix"] = _scan_ed_suffix(e_used, e_ref,
                                                         d_used, d_ref)
        else:
            out[ch] = None
    return out


_ED_CSV_FIELDS = ["leg", "arch", "subset_size", "n_reactions",
                  "n_density_species", "E_kcalmol", "D_rmse", "gamma",
                  "gammaD_kcalmol", "ED_kcalmol", "E_pbe_kcalmol",
                  "D_pbe_rmse", "ED_pbe_kcalmol", "beats_pbe",
                  "E_scan_kcalmol", "D_scan_rmse", "ED_scan_kcalmol",
                  "beats_scan", "ED_pbe_cell_kcalmol",
                  "ED_scan_cell_kcalmol", "n_reactions_slice"]


def _blank_if_none(x: Any) -> Any:
    """Empty CSV cell for absent SCAN legs (cache not pulled / coverage-gated)."""
    return "" if x is None else x


def write_combined_ed_csv(legs: Dict[str, Optional[Dict[str, Any]]],
                          out_path: Path, *,
                          n_reactions: Dict[Tuple[str, int], int],
                          n_density: Dict[Tuple[str, int], int],
                          n_reactions_slice: Optional[
                              Dict[Tuple[str, int], int]] = None,
                          counts_by_leg: Optional[Dict[str, Tuple[
                              Dict[Tuple[str, int], int], ...]]] = None
                          ) -> Path:
    """Per-cell ED table for the given energy legs, alongside the figure --
    the machine-readable source for a paper table. One row per (leg, cell),
    cells in ARCH_ORDER-then-subset order; None legs skipped.
    ``n_reactions`` counts the cell's NN-scored deduped reactions;
    ``n_reactions_slice`` the cell's full test slice (finite-comparator
    rows) -- ``n_reactions < n_reactions_slice`` is the machine-readable
    incomplete-eval condition behind the figures' starred bars.
    ``counts_by_leg`` optionally overrides the flat count maps per leg
    (the per-channel 3x3 CSV, where each channel counts only its own pool's
    rows/species); 2-tuples (older call shape) leave the slice column
    blank, 3-tuples carry it. The CSV path is NOT appended to the figure
    list returned by ``build_density_energy_figures`` (that return contract
    stays PNG-only)."""
    order = {a: i for i, a in enumerate(ARCH_ORDER)}
    out_path = Path(out_path)
    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=_ED_CSV_FIELDS)
        w.writeheader()
        for leg, summary in legs.items():
            if not summary:
                continue
            nr, nd = n_reactions, n_density
            ns = n_reactions_slice or {}
            if counts_by_leg and leg in counts_by_leg:
                cl = counts_by_leg[leg]
                nr, nd = cl[0], cl[1]
                ns = cl[2] if len(cl) > 2 else {}
            cells = sorted(summary["cells"].items(),
                           key=lambda kv: (order.get(kv[0][0], len(order)),
                                           kv[0][0], kv[0][1]))
            for (arch, ss), c in cells:
                w.writerow({
                    "leg": leg, "arch": arch, "subset_size": ss,
                    "n_reactions": nr.get((arch, ss), ""),
                    "n_density_species": nd.get((arch, ss), ""),
                    "E_kcalmol": c["E"], "D_rmse": c["D"],
                    "gamma": summary["gamma"],
                    "gammaD_kcalmol": c["gammaD"], "ED_kcalmol": c["ED"],
                    "E_pbe_kcalmol": summary["e_pbe"],
                    "D_pbe_rmse": summary["d_pbe"],
                    "ED_pbe_kcalmol": summary["ed_pbe"],
                    "beats_pbe": c["beats_pbe"],
                    "E_scan_kcalmol": _blank_if_none(summary.get("e_scan")),
                    "D_scan_rmse": _blank_if_none(summary.get("d_scan")),
                    "ED_scan_kcalmol": _blank_if_none(summary.get("ed_scan")),
                    "beats_scan": _blank_if_none(c.get("beats_scan")),
                    # cell-matched anchors behind the verdicts (blank when a
                    # verdict fell back to its pooled column)
                    "ED_pbe_cell_kcalmol": _blank_if_none(
                        c.get("ed_pbe_cell")),
                    "ED_scan_cell_kcalmol": _blank_if_none(
                        c.get("ed_scan_cell")),
                    "n_reactions_slice": ns.get((arch, ss), ""),
                })
    return out_path


_ELEMENT_SYMBOLS = frozenset(
    "h he li be b c n o f ne na mg al si p s cl ar k ca sc ti v cr mn fe co "
    "ni cu zn ga ge as se br kr".split())


def training_subsets_by_size(run_dir: Path) -> Dict[int, List[str]]:
    """``{subset_size: sorted non-atom training molecules}`` from each spec's
    ``train_metadata.json``. The training subset is shared across archs for a
    given subset_size (verified), so the first spec per size wins. Single-element
    anchors (h, c, n, o, ...) are dropped for legibility -- the molecules are
    what each subset actually trained on."""
    cells = ccp._read_manifest_cells(run_dir)
    out: Dict[int, List[str]] = {}
    for idx, spec_dir in ccp._spec_dirs(run_dir):
        ss = cells.get(idx, {}).get("subset_size")
        meta_path = spec_dir / "train_metadata.json"
        if ss is None or ss in out or not meta_path.is_file():
            continue
        try:
            mols = json.loads(meta_path.read_text()).get("molecules", [])
        except (json.JSONDecodeError, OSError):
            continue
        out[ss] = sorted(m for m in mols
                         if str(m).casefold() not in _ELEMENT_SYMBOLS)
    return out


_REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_bh76_reactions() -> Dict[str, Dict[str, Any]]:
    """``{reaction_name: {reactants, products, coeffs}}`` from the in-repo BH76
    pool JSON -- the authoritative reactants->products definitions."""
    p = _REPO_ROOT / "xcquinox/alec/data/bh76_full_pool.json"
    if not p.is_file():
        return {}
    try:
        pool = json.loads(p.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return {r["name"]: r for r in pool.get("reactions", []) if "name" in r}


def training_reactions_by_size(run_dir: Path,
                               ledgers_dir: Optional[Path] = None
                               ) -> Dict[int, Dict[str, List[Any]]]:
    """``{subset_size: {"ae": [W4-11 molecule, ...], "rxn": [(reactants, products),
    ...]}}`` -- the AUTHORITATIVE training content from the subset-selection
    ledger (``resolved_config.yaml: subset_ledger_path``), so W4-11 atomization
    points (``w411_X_atomization`` -> molecule X) and BH76 reaction points
    (``bh76_..._to_...`` -> reactants->products, looked up in the BH76 pool) are
    distinguished and reactions are NOT split into separate species. Returns ``{}``
    if the ledger is not found locally."""
    cfg = Path(run_dir) / "resolved_config.yaml"
    ledger_name = None
    if cfg.is_file():
        for line in cfg.read_text().splitlines():
            s = line.strip()
            if s.startswith("subset_ledger_path:"):
                ledger_name = Path(s.split(":", 1)[1].strip()).name
    if not ledger_name:
        return {}
    ledgers_dir = Path(ledgers_dir) if ledgers_dir else _REPO_ROOT / "hpcjobs/ledgers"
    ledger_path = ledgers_dir / ledger_name
    if not ledger_path.is_file():
        return {}
    try:
        ledger = json.loads(ledger_path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    rxn_lookup = _load_bh76_reactions()
    out: Dict[int, Dict[str, List[Any]]] = {}
    for key, entry in ledger.items():
        if not key.startswith("jsd/") or not isinstance(entry, dict):
            continue
        try:
            ss = int(key.split("/", 1)[1])
        except ValueError:
            continue
        ae: List[str] = []
        rxn: List[Tuple[List[str], List[str]]] = []
        for nm in entry.get("point_names", []):
            if nm.startswith("w411_") and nm.endswith("_atomization"):
                ae.append(nm[len("w411_"):-len("_atomization")])
            elif nm.startswith("bh76_"):
                r = rxn_lookup.get(nm)
                if r:
                    rxn.append((list(r.get("reactants", [])),
                                list(r.get("products", []))))
                else:  # fall back to parsing the name "A_B_to_C"
                    core = nm[len("bh76_"):]
                    if "_to_" in core:
                        lhs, rhs = core.split("_to_", 1)
                        rxn.append((lhs.split("_"), rhs.split("_")))
        out[ss] = {"ae": ae, "rxn": rxn}
    return out


def _energy_arch_axis(rows: List[Dict[str, Any]]) -> List[str]:
    """Rung-sorted arch order for the per-cell energy bars
    (:func:`plot_energy_wtmad_mae`) so the grouped bars climb Jacob's ladder."""
    return arch_style.sort_by_rung(_archs_present(rows) or ["deep"])


def _grouped_arch_bars(ax, metric: Dict[Tuple[str, int], float],
                       archs: List[str], subsets: List[int], *,
                       pbe_line: Optional[float] = None, title: str,
                       scan_line: Optional[float] = None,
                       scan_suffix: str = "",
                       pbe_by_cell: Optional[
                           Dict[Tuple[str, int], float]] = None,
                       scan_by_cell: Optional[
                           Dict[Tuple[str, int], float]] = None,
                       reference_by_arch: Optional[Dict[str, str]] = None,
                       relocked_cells: Optional[set] = None,
                       mixed_cells: Optional[set] = None,
                       incomplete_cells: Optional[set] = None,
                       vxc_pre_fix: bool = False) -> None:
    """Grouped per-(arch, subset_size) bar panel: one bar group per arch
    (rung-ordered by the caller), x = subset_size, PBE dashed / SCAN dotted
    reference lines when finite, green beats markers on bars strictly below
    their reference (see ``reference_by_arch``). ``pbe_line=None`` silently
    draws no line and no marks (the in-sample AE panel has no baseline).
    ``scan_suffix`` qualifies a drawn-but-partially-covered SCAN line in its
    legend label (the ``", used/ref"`` convention).

    ``pbe_by_cell`` supplies CELL-SLICE PBE reductions (PBE over the cell's
    full test slice -- every slice reaction/species with a finite
    comparator leg, independent of NN convergence): the beats-PBE marks are
    judged bar-by-bar against the cell's own anchor, and ONE black capped
    horizontal span per subset-size group -- spanning the group's bar
    cluster via :func:`_group_span_points`, per-bar only when the group's
    cells disagree -- shows the anchor while the dash-dot pooled line stays
    for cross-cell scale. Cells score training-subset-dependent slices, so
    the pooled line alone over- or under-states PBE on individual cells.
    ``scan_by_cell`` draws the SCAN twins as grey capped spans
    (coverage-gated per cell by the caller) and relabels the pooled dotted
    line "SCAN (pooled)". Bars whose cell is in ``incomplete_cells`` (the
    NN scored fewer reactions than the cell's slice) carry a star above the
    bar -- a disclosure of the incomplete hold-out eval, not a grading
    change: the spans and beats anchors are slice reductions regardless. A
    cell with no finite bar height cannot carry the star; the callers'
    footer warning still names it with its scored/slice counts. Shared by
    ``plot_energy_wtmad_mae`` and the overview composites."""
    bw = 0.8 / max(1, len(archs))
    beat_x: List[float] = []
    beat_h: List[float] = []
    # Reference-provenance glyphs: cells trained on relocked degenerate-radical
    # references, and cells whose references changed mid-training (not
    # interpretable). Empty/None on every run without a mid-run swap.
    relocked = relocked_cells or set()
    mixed = mixed_cells or set()
    incomplete = incomplete_cells or set()
    relock_x: List[float] = []
    relock_h: List[float] = []
    mixed_x: List[float] = []
    mixed_h: List[float] = []
    inc_x: List[float] = []
    inc_h: List[float] = []
    for j, a in enumerate(archs):
        xs = [i + (j - (len(archs) - 1) / 2) * bw
              for i in range(len(subsets))]
        hs = [metric.get((a, s), float("nan")) for s in subsets]
        # Cell-level mixed-references hatch takes precedence over the
        # arch-level V_xc-provenance hatch (a mixed cell is uninterpretable
        # regardless of which potential trained it).
        vh = _vxc_hatch(a) if vxc_pre_fix else None
        hatches = [("//" if (a, s) in mixed else vh) for s in subsets]
        if any(hatches):
            for x, h, s, hatch in zip(xs, hs, subsets, hatches):
                ax.bar([x], [h], width=bw, color=ARCH_COLOR.get(a, "0.5"),
                       edgecolor="k", linewidth=0.3, hatch=hatch,
                       label=a if s == subsets[0] else None)
        else:
            ax.bar(xs, hs, width=bw, color=ARCH_COLOR.get(a, "0.5"),
                   edgecolor="k", linewidth=0.3, label=a)
        for x, h, s in zip(xs, hs, subsets):
            if not _is_num(h):
                continue
            if (a, s) in relocked:
                relock_x.append(x); relock_h.append(h)
            elif (a, s) in mixed:
                mixed_x.append(x); mixed_h.append(h)
            if (a, s) in incomplete:
                inc_x.append(x); inc_h.append(h)
        # beats marks: each bar against its own-rung reference (PBE for GGA
        # archs, SCAN for beyond-GGA ones when reference_by_arch is given).
        # With a cell-anchor map present, a cell absent from it stays
        # UNMARKED -- the cell-matched comparator was withdrawn (e.g. SCAN
        # coverage under the per-cell floor), and falling back to the pooled
        # line would reintroduce the different-set flattery the ticks
        # eliminate. Only legacy callers with no map at all mark against the
        # pooled line.
        ref_kind = (reference_by_arch or {}).get(a, "pbe")
        if ref_kind == "scan":
            anchor_map, fallback = scan_by_cell, scan_line
        else:
            anchor_map, fallback = pbe_by_cell, pbe_line
        if anchor_map is not None:
            for x, h, s in zip(xs, hs, subsets):
                anchor = anchor_map.get((a, s))
                if _is_num(h) and _is_num(anchor) and h < float(anchor):
                    beat_x.append(x); beat_h.append(h)
        else:
            for x, h in _beats_pbe_marks(xs, hs, fallback):
                beat_x.append(x); beat_h.append(h)
    tick_x, tick_h, tick_w = _group_span_points(pbe_by_cell, archs,
                                                subsets, bw)
    stick_x, stick_h, stick_w = _group_span_points(scan_by_cell, archs,
                                                   subsets, bw)
    if relock_x:
        ax.scatter(relock_x, relock_h, marker="*", s=70, color="#1f77b4",
                   edgecolor="k", linewidths=0.4, zorder=7,
                   label="relocked refs")
    if mixed_x:
        ax.scatter(mixed_x, mixed_h, marker="X", s=42, color="#d62728",
                   edgecolor="k", linewidths=0.4, zorder=7,
                   label="refs changed mid-training (not interpretable)")
    # Incomplete-eval star: a text annotation, NOT a star scatter marker --
    # the relocked-refs glyph above already owns marker="*".
    for x, h in zip(inc_x, inc_h):
        ax.annotate("*", (x, h), xytext=(0, 1.5),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=10, color="k", zorder=8)
    if inc_x:
        ax.plot([], [], ls="", marker="$*$", color="k",
                label="* incomplete hold-out eval (NN scored < cell slice)")
    if _is_num(pbe_line):
        ax.axhline(pbe_line, ls="-.", color="k", linewidth=1.0,
                   label=("PBE (pooled)" if tick_x else "PBE"))
    if tick_x:
        ax.errorbar(tick_x, tick_h, xerr=tick_w, fmt="none", ecolor="k",
                    elinewidth=1.2, capsize=3.0, capthick=1.2, alpha=0.85,
                    zorder=5, label="PBE (cell rows)")
    if stick_x:
        ax.errorbar(stick_x, stick_h, xerr=stick_w, fmt="none",
                    ecolor="#555555", elinewidth=1.2, capsize=3.0,
                    capthick=1.2, alpha=0.85, zorder=5,
                    label="SCAN (cell rows)")
    if _is_num(scan_line):
        ax.axhline(scan_line, ls=":", color="#555555", linewidth=1.3,
                   label=(f"SCAN (pooled){scan_suffix}" if stick_x
                          else f"SCAN{scan_suffix}"))
        # A SCAN line above every bar otherwise lands flush against the top
        # spine, where the gamma stamp sits on the ED rows. Give it headroom
        # rather than letting the two overprint.
        lo, hi = ax.get_ylim()
        if float(scan_line) > 0.0:
            ax.set_ylim(lo, max(hi, float(scan_line) * 1.14))
    if beat_x:
        ax.scatter(beat_x, beat_h, marker="v", s=16, color="#2ca02c",
                   edgecolor="k", linewidths=0.3, zorder=6,
                   label=("beats rung reference" if reference_by_arch
                          else "beats PBE"))
    if vxc_pre_fix:
        # Zero-size legend proxies, one per V_xc class actually on the panel
        # (attached to the axes so get_legend_handles_labels sees them).
        if any(a in _VXC_PRE_GATED for a in archs):
            ax.add_patch(plt.Rectangle(
                (0, 0), 0, 0, fill=False, edgecolor="0.25",
                hatch=_VXC_HATCH_GATED * 2,
                label="pre-correction V_xc (re-run gated on SCF stabilization)"))
        if any(a in _VXC_PRE_READY for a in archs):
            ax.add_patch(plt.Rectangle(
                (0, 0), 0, 0, fill=False, edgecolor="0.25",
                hatch=_VXC_HATCH_READY * 2,
                label="pre-correction V_xc (safe to re-run)"))
    ax.set_xticks(range(len(subsets)))
    ax.set_xticklabels(subsets)
    ax.set_xlabel("training subset_size", fontsize=8)
    ax.set_ylabel("kcal/mol", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)


def plot_energy_wtmad_mae(rows: List[Dict[str, Any]], out_path: Path, run_id: str,
                          note: str = "", provenance: Optional[str] = None,
                          caveat: Optional[str] = None,
                          training_subsets: Optional[Dict[int, List[str]]] = None,
                          scan_baseline: Optional[Dict[str, float]] = None,
                          scan_errors: Optional[Dict[str, float]] = None,
                          dataset: Optional[str] = None) -> Path:
    """Held-out energy: ONE bar per (arch, subset_size) cell -- combined
    reaction-energy MAE (panel a) and 2-subset WTMAD-2 (panel b) -- grouped by
    arch (rung-ordered) within each subset_size on the x-axis. NO error bars:
    each cell is a single model trained on a distinct subset and evaluated on a
    fixed held-out set, so a within-sample spread would be arbitrary and
    cross-subset aggregation is invalid (the six subset models per arch are not
    comparable). The subset trend is the x-axis. WTMAD-2 here = 2-subset, NOT
    full GMTKN55. Green beats markers are judged against each architecture's
    OWN-RUNG reference's cell-slice anchor (black capped spans for GGA
    archs' PBE anchors, grey spans for meta-GGA/rung-3.5's SCAN anchors;
    ``scan_errors`` supplies the per-reaction SCAN errors behind the grey
    spans; bars whose NN scored fewer reactions than the cell's slice carry
    the incomplete-eval star), and a dotted SCAN full-pool reference line is
    added to the MAE panel when ``scan_baseline`` carries a finite combined
    MAE (absent SCAN cache -> unchanged)."""
    with plt.rc_context(_STYLE):
        archs = _energy_arch_axis(rows)
        subsets = _present_subsets(rows) or [1]
        mae = reaction_mae_by_arch_subset(rows)
        wt = wtmad2_by_arch_subset(rows)
        pbe_mae = _mae([r["abs_error_pbe_kcalmol"] for r in _dedup_rows_by_name(rows)])
        pbe_wt = wtmad2_pbe_baseline(rows)
        mae_cell_anchors = pbe_reaction_mae_by_cell(rows)
        wt_cell_anchors = wtmad2_pbe_by_arch_subset(rows)
        mae_scan_anchors = scan_reaction_mae_by_cell(rows, scan_errors)
        wt_scan_anchors = wtmad2_scan_by_cell(rows, scan_errors)
        inc_cells = set(_incomplete_energy_cells(rows))
        # Either panel drawing glyphs obliges the key: rows without pool
        # labels empty the WTMAD-2 anchors while the MAE anchors survive.
        anote = _cell_anchor_note(wt_cell_anchors or mae_cell_anchors)
        if anote:
            note = (note + "  " + anote) if note else anote
        has_ts = bool(training_subsets)
        # Taller bottom band than the pre-rung layout: the rung-grouped legend
        # (ncol ~ #rungs) stacks into ~3 rows, which must clear the training-
        # subset text block + footer without overlap.
        fig, axes = plt.subplots(1, 2, figsize=(13, 7.8 if has_ts else 5.6),
                                 squeeze=False)
        # SCAN full-pool combined MAE only tracks panel (a) (there is no
        # 2-subset SCAN WTMAD-2 to draw); guarded so absent SCAN changes nothing,
        # and withdrawn outright when SCAN covers too little of PBE's pool.
        scan_c, scan_sfx = scan_line_value(scan_baseline, "combined")
        vxc_pre = _run_predates_vxc_fix(run_id)
        _grouped_arch_bars(
            axes[0][0], mae, archs, subsets, pbe_line=pbe_mae,
            title="Held-out reaction-energy MAE (combined), per (arch, subset)",
            scan_line=scan_c, scan_suffix=scan_sfx, vxc_pre_fix=vxc_pre,
            pbe_by_cell=mae_cell_anchors,
            scan_by_cell=(mae_scan_anchors or None),
            reference_by_arch=arch_reference_kinds(archs),
            incomplete_cells=inc_cells)
        _grouped_arch_bars(
            axes[0][1], wt, archs, subsets, pbe_line=pbe_wt,
            title="2-subset WTMAD-2 (BH76+W4-11), per (arch, subset)",
            vxc_pre_fix=vxc_pre,
            pbe_by_cell=wt_cell_anchors,
            scan_by_cell=(wt_scan_anchors or None),
            reference_by_arch=arch_reference_kinds(archs),
            incomplete_cells=inc_cells)
        handles, labels = axes[0][0].get_legend_handles_labels()
        if labels:
            fig.legend(handles, labels, loc="lower center",
                       ncol=len(arch_style.RUNG_ORDER), fontsize=7,
                       frameon=False, bbox_to_anchor=(0.5, 0.05))
        if has_ts:
            lines = ["Training subsets (held-in molecules; + element anchors):"]
            for ss in sorted(training_subsets):
                ms = training_subsets[ss]
                lines.append(f"  {ss}:  {', '.join(ms) if ms else '(atoms only)'}")
            # Anchored high (just under the axes) so it clears the taller legend.
            fig.text(0.06, 0.35, "\n".join(lines), ha="left", va="top",
                     fontsize=6, family="monospace", color="#333333")
        _stamp_parity_footer(
            fig, run_id=run_id, note=note, provenance=provenance, caveat=caveat,
            dataset=dataset,
            title="Held-out energy: per-cell combined MAE + 2-subset WTMAD-2 (NOT full GMTKN55)")
        fig.tight_layout(rect=(0, 0.37 if has_ts else 0.16, 1, 0.93))
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return out_path


def _insample_density_lines_panel(ax, density_rows: List[Dict[str, Any]]
                                  ) -> None:
    """Per-arch mean in-sample density RMSE vs subset_size (n annotated), with
    the grey dashed PBE-vs-CCSD line over subset_size when the model-free
    ``density_rmse_pbe`` column is present. Panel body shared by
    ``plot_insample_density_ccsd`` and ``plot_insample_overview``."""
    archs = _archs_present(density_rows) or ["deep"]
    for a in archs:
        by_s: Dict[int, List[float]] = {}
        for r in density_rows:
            if r.get("arch") == a and _is_num(r.get("density_rmse")):
                by_s.setdefault(r["subset_size"], []).append(r["density_rmse"])
        pts = sorted((s, float(np.mean(v)), len(v)) for s, v in by_s.items())
        if pts:
            ax.plot([s for s, _, _ in pts], [m for _, m, _ in pts],
                    marker="o", ms=5, color=ARCH_COLOR[a], label=a)
            for s, m, n in pts:
                ax.annotate(f"n={n}", (s, m), fontsize=5,
                            color=ARCH_COLOR[a], xytext=(0, 4),
                            textcoords="offset points")
    # PBE-vs-CCSD baseline (arch-independent): mean over the molecules
    # present at each subset_size, grey dashed
    pbe_by_s: Dict[int, List[float]] = {}
    for r in density_rows:
        if _is_num(r.get("density_rmse_pbe")):
            pbe_by_s.setdefault(r["subset_size"], []).append(
                r["density_rmse_pbe"])
    pbe_pts = sorted((s, float(np.mean(v))) for s, v in pbe_by_s.items())
    if pbe_pts:
        ax.plot([s for s, _ in pbe_pts], [m for _, m in pbe_pts],
                ls="--", color="0.35", marker="x", ms=5, lw=1.2,
                label="PBE vs CCSD")
    ax.set_yscale("log")
    ax.set_xlabel("training subset_size", fontsize=8)
    ax.set_ylabel("density RMSE vs CCSD (grid, weighted-mean)", fontsize=8)
    ax.set_title("In-sample density fit vs CCSD (per arch)", fontsize=9)
    if ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=6, ncol=2)
    ax.grid(True, which="both", alpha=0.3)


def _insample_density_strip_panel(ax, density_rows: List[Dict[str, Any]]
                                  ) -> None:
    """Per-molecule in-sample density strip (every point, arch-jittered) with
    one grey PBE x per molecule when the PBE column is present. Panel body
    shared by ``plot_insample_density_ccsd`` and ``plot_insample_overview``."""
    archs = _archs_present(density_rows) or ["deep"]
    arch_idx = {a: i for i, a in enumerate(archs)}
    mols = sorted({r["molecule"] for r in density_rows if r.get("molecule")})
    mol_x = {m: i for i, m in enumerate(mols)}
    noff = max(1, len(archs))
    for r in density_rows:
        if not _is_num(r.get("density_rmse")) or r.get("molecule") not in mol_x:
            continue
        jit = (arch_idx.get(r.get("arch"), 0) - (noff - 1) / 2) * 0.12
        ax.scatter(mol_x[r["molecule"]] + jit, r["density_rmse"], s=18,
                   alpha=0.75, color=ARCH_COLOR.get(r.get("arch"), "0.5"),
                   edgecolor="none")
    # PBE baseline per molecule (arch-independent -> one grey x each)
    pbe_by_mol: Dict[str, List[float]] = {}
    for r in density_rows:
        if _is_num(r.get("density_rmse_pbe")) and r.get("molecule") in mol_x:
            pbe_by_mol.setdefault(r["molecule"], []).append(
                r["density_rmse_pbe"])
    for m, vals in pbe_by_mol.items():
        ax.scatter(mol_x[m], float(np.mean(vals)), s=26, marker="x",
                   color="0.35", lw=1.2, zorder=3)
    ax.set_yscale("log")
    ax.set_xticks(range(len(mols)))
    ax.set_xticklabels(mols, rotation=60, ha="right", fontsize=6)
    ax.set_ylabel("density RMSE vs CCSD", fontsize=8)
    ax.set_title(f"Per-molecule (every point; {len(mols)} trained species)",
                 fontsize=9)
    ax.grid(True, axis="y", which="both", alpha=0.3)


def _insample_ae_strip_panel(ax, ae_rows: List[Dict[str, Any]]) -> None:
    """Per-molecule in-sample |AE error| strip (kcal/mol, arch-jittered), the
    AE analog of ``_insample_density_strip_panel``. Rows without a molecule or
    a finite nonzero ``AE_error_kcalmol`` are dropped (log axis). No PBE
    series: the in-sample per_molecule.json carries no PBE AE column."""
    plotted = [r for r in ae_rows
               if r.get("molecule") and _is_num(r.get("AE_error_kcalmol"))
               and abs(r["AE_error_kcalmol"]) > 0.0]
    archs = _archs_present(plotted) or ["deep"]
    arch_idx = {a: i for i, a in enumerate(archs)}
    mols = sorted({r["molecule"] for r in plotted})
    mol_x = {m: i for i, m in enumerate(mols)}
    noff = max(1, len(archs))
    for r in plotted:
        jit = (arch_idx.get(r.get("arch"), 0) - (noff - 1) / 2) * 0.12
        ax.scatter(mol_x[r["molecule"]] + jit, abs(r["AE_error_kcalmol"]),
                   s=18, alpha=0.75,
                   color=ARCH_COLOR.get(r.get("arch"), "0.5"),
                   edgecolor="none")
    ax.set_yscale("log")
    ax.set_xticks(range(len(mols)))
    ax.set_xticklabels(mols, rotation=60, ha="right", fontsize=6)
    ax.set_ylabel("|AE error| (kcal/mol)", fontsize=8)
    ax.set_title(f"Per-molecule |AE error| (every point; {len(mols)} trained "
                 "species)", fontsize=9)
    ax.grid(True, axis="y", which="both", alpha=0.3)


def plot_insample_density_ccsd(density_rows: List[Dict[str, Any]], out_path: Path,
                               run_id: str, note: str = "",
                               provenance: Optional[str] = None,
                               caveat: Optional[str] = None) -> Path:
    """In-sample density error vs CCSD (Dick-style diagnostic): (left) per-arch
    density RMSE vs subset_size with n annotated; (right) per-molecule strip
    (every point, since the trained-species set is tiny). Labeled IN-SAMPLE.
    Rows carrying ``density_rmse_pbe`` (the model-free PBE-vs-CCSD baseline on
    the same grid; emitted by newer evals) add a grey dashed PBE baseline to
    both panels; older runs without the column render exactly as before.
    Panel bodies live in ``_insample_density_lines_panel`` /
    ``_insample_density_strip_panel`` (shared with the in-sample overview)."""
    with plt.rc_context(_STYLE):
        archs = _archs_present(density_rows) or ["deep"]
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), squeeze=False)
        _insample_density_lines_panel(axes[0][0], density_rows)
        _insample_density_strip_panel(axes[0][1], density_rows)

        arch_handles = [Patch(facecolor=ARCH_COLOR[a], label=a)
                        for a in arch_style.sort_by_rung(archs)]
        fig.legend(handles=arch_handles, loc="lower center",
                   ncol=len(arch_style.RUNG_ORDER), fontsize=7,
                   frameon=False, bbox_to_anchor=(0.5, 0.02))
        insample = ("IN-SAMPLE density fit on TRAINING molecules (atoms excluded; "
                    "weighted-mean grid RMSE vs CCSD, NOT N_e-normalized). "
                    "Training-set fit, NOT generalization; not comparable to the "
                    "held-out energy panels.")
        _stamp_parity_footer(
            fig, run_id=run_id, note=note, provenance=provenance, caveat=insample,
            title="In-sample density error vs CCSD (Dick-style diagnostic)")
        fig.tight_layout(rect=(0, 0.08, 1, 0.92))
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return out_path


def _holdout_density_lines_panel(ax, density_rows: List[Dict[str, Any]],
                                 pbe_mol: Dict[str, float],
                                 scan_records: Optional[
                                     Dict[str, Dict[str, Any]]] = None) -> None:
    """Per-arch mean held-out density RMSE vs subset_size (n annotated) with
    the grey dashed PBE pool-mean line, and -- when a SCAN density cache is
    present -- the dotted SCAN meta-GGA line over the SAME species. Panel body
    shared by ``plot_holdout_density_ccsd`` and
    ``plot_density_energy_overview``. The SCAN line is the comparator the
    ``_mgga`` archs are judged against (they pretrain to SCAN); absent cache ->
    no line, panel unchanged."""
    for a in (_archs_present(density_rows) or []):
        by_s: Dict[int, List[float]] = {}
        for r in density_rows:
            if r.get("arch") == a and _is_num(r.get("density_rmse")):
                by_s.setdefault(r["subset_size"], []).append(
                    r["density_rmse"])
        pts = sorted((s, float(np.mean(v)), len(v))
                     for s, v in by_s.items())
        if pts:
            ax.plot([s for s, _, _ in pts], [m for _, m, _ in pts],
                    marker="o", ms=5, color=ARCH_COLOR.get(a, "0.5"),
                    label=a)
            for s, m, n in pts:
                ax.annotate(f"n={n}", (s, m), fontsize=5,
                            color=ARCH_COLOR.get(a, "0.5"),
                            xytext=(0, 4), textcoords="offset points")
    if pbe_mol:
        pbe_mean = float(np.mean(list(pbe_mol.values())))
        ax.axhline(pbe_mean, ls="--", color="0.35", lw=1.2,
                   label=f"PBE vs CCSD (pool mean {pbe_mean:.1e})")
        scan_mean = scan_density_line(scan_records, pbe_mol)
        if scan_mean is not None:
            ax.axhline(scan_mean, ls=":", color="#555555", lw=1.4,
                       label=f"SCAN vs CCSD (pool mean {scan_mean:.1e})")
    ax.set_yscale("log")
    ax.set_xlabel("training subset_size", fontsize=8)
    ax.set_ylabel("held-out density RMSE vs CCSD (grid, weighted-mean)",
                  fontsize=8)
    ax.set_title("Held-out density error vs CCSD (per arch)", fontsize=9)
    if ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=6, ncol=2)
    ax.grid(True, which="both", alpha=0.3)


def _density_parity_panel(ax, density_rows: List[Dict[str, Any]],
                          pbe_mol: Dict[str, float], *,
                          nn_key: str = "density_rmse",
                          unit_label: str = "density RMSE",
                          limits: Optional[Tuple[float, float]] = None
                          ) -> None:
    """Per-species NN-vs-PBE density parity (log-log; below the diagonal =
    the NN density is closer to CCSD than PBE is), with a PBE-only sorted
    strip as the fallback when no NN channel exists. Panel body shared by
    ``plot_holdout_density_ccsd`` and ``plot_density_energy_overview``.
    ``nn_key``/``unit_label`` select the error channel (defaults reproduce
    the grid-weighted RMSE panel; the DFS-units twins pass
    ``density_eps_l1`` with an eps label, and the caller supplies a
    ``pbe_mol`` built on the matching PBE key). ``limits`` imposes an
    external square (lo, hi) on both axes -- the 3x3 figures pass one
    row-wide envelope so their three channel panels share a frame and are
    directly comparable; None keeps the own-data envelope."""
    n_pairs = 0
    fin_xy: List[float] = []
    for r in density_rows:
        x = pbe_mol.get(r.get("molecule"))
        y = r.get(nn_key)
        if not (_is_num(x) and _is_num(y)):
            continue
        ax.scatter(x, y, s=14, alpha=0.6,
                   color=ARCH_COLOR.get(r.get("arch"), "0.5"),
                   edgecolor="none")
        fin_xy.extend((float(x), float(y)))
        n_pairs += 1
    pos_xy = [v for v in fin_xy if v > 0.0]
    lims = limits if limits is not None else (
        (0.8 * min(pos_xy), 1.25 * max(pos_xy)) if pos_xy else None)
    if n_pairs and lims is not None:
        # square shared log limits (external, or from the pooled POSITIVE
        # own data): without them the two axes autoscale independently (the
        # NN axis outlier-stretched, the PBE axis tight), the cloud drifts
        # off-center and the y=x diagonal exits mid-frame instead of
        # corner-to-corner. A zero-valued error (unrenderable on the log
        # axes anyway) must not poison lo -- set_xlim(0, ...) on a log axis
        # is silently ignored and the panel falls back to non-square.
        lo, hi = lims
        ax.plot([lo, hi], [lo, hi], ls=":", color="0.5", lw=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
    elif n_pairs:
        ax.set_xscale("log")
        ax.set_yscale("log")
    elif pbe_mol:
        # PBE-only data (no NN density yet): per-species PBE strip
        vals = sorted(pbe_mol.values(), reverse=True)
        ax.scatter(range(len(vals)), vals, s=10, color="0.35")
        ax.set_yscale("log")
        ax.set_xlabel("species (sorted by PBE error)", fontsize=8)
    ax.set_xlabel(ax.get_xlabel() or f"PBE {unit_label} vs CCSD",
                  fontsize=8)
    ax.set_ylabel(f"NN {unit_label} vs CCSD" if n_pairs
                  else f"PBE {unit_label} vs CCSD", fontsize=8)
    ax.set_title(f"Per-species NN vs PBE (both vs CCSD; {n_pairs} points)"
                 if n_pairs else
                 f"PBE-vs-CCSD per species ({len(pbe_mol)} refs)",
                 fontsize=9)
    ax.grid(True, which="both", alpha=0.3)


_HOLDOUT_DENSITY_CAVEAT = (
    "HELD-OUT density error vs CCSD reference densities on the W4-11+BH76 "
    "benchmark species (atoms excluded; weighted-mean grid RMSE, NOT "
    "N_e-normalized). PBE baseline is model-free on the same grid. Density "
    "generalization; separate from the held-out ENERGY panels by design.")


def plot_holdout_density_ccsd(density_rows: List[Dict[str, Any]],
                              out_path: Path, run_id: str, *,
                              pbe_table: Optional[Dict[str, Dict[str, float]]]
                              = None,
                              note: str = "",
                              provenance: Optional[str] = None,
                              scan_density_records: Optional[
                                  Dict[str, Dict[str, Any]]] = None,
                              dataset: Optional[str] = None) -> Path:
    """HELD-OUT density error vs CCSD on the W4-11+BH76 benchmark species:
    (left) per-arch weighted-mean grid RMSE vs subset_size with the grey
    dashed PBE-vs-CCSD pool baseline; (right) per-species NN-vs-PBE parity
    (log-log; below the diagonal = the NN density is closer to CCSD than PBE
    is). The PBE channel is model-free and shared across every spec --
    ``pbe_table`` (the run-level ``pbe_density_errors.json``) supplies it for
    PBE-only re-evals; rows carrying ``density_rmse_pbe`` work too. Panel
    bodies live in ``_holdout_density_lines_panel`` / ``_density_parity_panel``
    (shared with the held-out overview)."""
    # per-molecule PBE map: explicit table first, else from the rows
    pbe_mol = _pbe_density_map(density_rows, pbe_table)

    with plt.rc_context(_STYLE):
        archs = _archs_present(density_rows) or []
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), squeeze=False)
        _holdout_density_lines_panel(axes[0][0], density_rows, pbe_mol,
                                     scan_density_records)
        _density_parity_panel(axes[0][1], density_rows, pbe_mol)

        arch_handles = [Patch(facecolor=ARCH_COLOR[a], label=a)
                        for a in arch_style.sort_by_rung(archs) if a in ARCH_COLOR]
        if arch_handles:
            fig.legend(handles=arch_handles, loc="lower center",
                       ncol=len(arch_style.RUNG_ORDER),
                       fontsize=7, frameon=False, bbox_to_anchor=(0.5, 0.02))
        _stamp_parity_footer(
            fig, run_id=run_id, note=note, provenance=provenance,
            caveat=_HOLDOUT_DENSITY_CAVEAT, dataset=dataset,
            title="Held-out density error vs CCSD (NN vs PBE)")
        # rect top 0.90 (was 0.92) makes room for the dataset footer line
        fig.tight_layout(rect=(0, 0.08, 1, 0.90))
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return out_path


def plot_holdout_density_per_arch(hd_rows: List[Dict[str, Any]],
                                  out_path: Path, run_id: str, *,
                                  pbe_table: Optional[Dict[str, Dict[str, float]]]
                                  = None,
                                  note: str = "",
                                  provenance: Optional[str] = None,
                                  scan_density_records: Optional[
                                      Dict[str, Dict[str, Any]]] = None,
                                  dataset: Optional[str] = None) -> Path:
    """Standalone single-panel figure of the per-arch held-out density trend
    vs subset_size (grid weighted-mean RMSE vs CCSD, PBE pool-mean dashed) --
    the left panel of ``plot_holdout_density_ccsd`` promoted to its own
    figure after the held-out overview swapped this slot for the parity and
    iso-ED decomposition panels. Same panel body
    (``_holdout_density_lines_panel``), same caveat."""
    pbe_mol = _pbe_density_map(hd_rows, pbe_table)
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(1, 1, figsize=(12, 5.6), squeeze=False)
        # the panel's own axes legend identifies archs + the PBE line; a
        # bottom fig-level legend would duplicate it and collide with the
        # note band on a single-panel figure
        _holdout_density_lines_panel(axes[0][0], hd_rows, pbe_mol,
                                     scan_density_records)
        _stamp_parity_footer(
            fig, run_id=run_id, note=note, provenance=provenance,
            caveat=_HOLDOUT_DENSITY_CAVEAT, dataset=dataset,
            title="Held-out density error vs CCSD per arch "
                  "(grid weighted-mean)")
        fig.tight_layout(rect=(0, 0.10, 1, 0.90))
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return out_path


_ED_CAVEAT = (_ED_SYM + " = 2/(1/E + 1/(gamma*D)) (Dick & Fernandez-Serra, "
              "PRB 104, L161109 (2021), Eq. 21); gamma self-calibrated per "
              "leg from pooled PBE anchors (gamma = E_PBE/D_PBE -- the "
              "Letter's regression slope, calibrated here on PBE alone), so "
              + _ED_SYM + " of PBE == E_PBE by construction (dashed).")

# kept near _ED_CAVEAT's length: the caveat renders as ONE figtext line and
# savefig.bbox="tight" widens the canvas to the longest line
_ED_DFS_UNITS_CAVEAT = (
    _ED_N_SYM + " = 2/(1/E + 1/(gamma " + _EPS_N_SYM + ")) (Dick & "
    "Fernandez-Serra, PRB 104, L161109 (2021), Eq. 21); " + _EPS_N_EQ
    + " (Eq. 20 per species; quadrature sum_i(w_i|rho-rho_ref|_i)/N_e); "
    "gamma "
    "EXTERNALLY FIXED (published Fig. 3 slope 1084.87 or own-axes refit) "
    "-- NOT self-calibrated: " + _ED_N_SYM + " of PBE != E_PBE, "
    "PBE off y=x.")

_HOLDOUT_OVERVIEW_CAVEAT = (
    "Single-pool 'WTMAD-2' (panels A, B) reduces to 56.84 * MAD_pool / "
    "mean|ref|_pool -- a scaled relative error, NOT a reweighting; only "
    "panel C (2-subset) reweights BH76 vs W4-11, and it is NOT full GMTKN55. "
    "E/F: " + _ED_SYM + " = 2/(1/E + 1/(gamma*D)), gamma = E_PBE/D_PBE "
    "self-calibrated from the pooled PBE anchors (value printed in the "
    "panels), so " + _ED_SYM + " of PBE == E_PBE (dashed).")

_INSAMPLE_OVERVIEW_CAVEAT = (
    "IN-SAMPLE (training-fit) overview on trained molecules -- NOT "
    "generalization; final checkpoint only (eval/ has no val-best variant, so "
    "the panels are identical in the final-step and val-best dirs; only the "
    "title's checkpoint stamp differs).\n"
    "No PBE AE baseline exists in-sample (per_molecule.json has no PBE AE "
    "column); no in-sample " + _ED_SYM + " (no PBE energy anchor to "
    "self-calibrate gamma).")


def _gamma_stamp_text(summary: Dict[str, Any]) -> str:
    """The gamma provenance text, branched on ``gamma_mode`` --
    self-calibrated summaries state the E_PBE/D_PBE construction; fixed ones
    the external value plus its source when the summary carries
    ``gamma_source`` (the value alone cannot tell the published slope from
    the own-axes fit). Single source for EVERY panel that prints a gamma
    (lines, bars, and the rich decomposition's inline stamp) so the
    truthfulness contract cannot fork."""
    self_cal = summary.get("gamma_mode", "self_calibrated") == "self_calibrated"
    src = summary.get("gamma_source")
    if self_cal:
        return ("$\\gamma$ = E$_{\\rm PBE}$/D$_{\\rm PBE}$ = "
                f"{summary['gamma']:.4g} (self-calibrated)")
    if src:
        return f"$\\gamma$ = {summary['gamma']:.6g} (fixed: {src})"
    return f"$\\gamma$ = {summary['gamma']:.6g} (fixed, external)"


def _gamma_stamp(ax, summary: Dict[str, Any]) -> None:
    """``_gamma_stamp_text`` placed top-right of an ED panel."""
    ax.text(0.98, 0.98, _gamma_stamp_text(summary), transform=ax.transAxes,
            fontsize=6, color="#444444", ha="right", va="top")


def _ed_lines_panel(ax, summary: Dict[str, Any], title: str,
                    reference_by_arch: Optional[Dict[str, str]] = None
                    ) -> None:
    """Per-arch ED vs subset_size lines with the dashed PBE line at
    ``ed_pbe`` (== the energy anchor under gamma self-calibration; a plain
    PBE level under a fixed gamma -- labels and the gamma stamp branch on
    ``gamma_mode``), green beats markers. With ``reference_by_arch`` each
    arch's marker follows its OWN-RUNG reference verdict (``beats_pbe`` for
    GGA archs, ``beats_scan`` for beyond-GGA ones); without it, the
    beats-PBE verdict, as before. Panel body shared by
    ``plot_combined_energy_density`` and ``plot_density_energy_overview``."""
    cells = summary["cells"]
    archs = arch_style.sort_by_rung(sorted({a for a, _ in cells}))
    for a in archs:
        pts = sorted((ss, c["ED"]) for (aa, ss), c in cells.items()
                     if aa == a and c["ED"] > 0.0)
        if not pts:
            continue
        ax.plot([s for s, _ in pts], [e for _, e in pts], marker="o",
                ms=5, color=ARCH_COLOR.get(a, "0.5"), label=a)
    self_cal = summary.get("gamma_mode", "self_calibrated") == "self_calibrated"
    if _is_num(summary["ed_pbe"]) and summary["ed_pbe"] > 0.0:
        ax.axhline(summary["ed_pbe"], ls="--", color="k", lw=1.0,
                   label=("PBE (ED = E by self-calibration)" if self_cal
                          else "PBE"))
    # The meta-GGA comparator, on the same gamma as the cells. Absent SCAN
    # legs leave ed_scan None and the panel is unchanged; a partial (but
    # floor-passing) coverage is named in the label via scan_suffix.
    ed_scan = summary.get("ed_scan")
    if _is_num(ed_scan) and ed_scan > 0.0:
        sfx = summary.get("scan_suffix") or ""
        ax.axhline(ed_scan, ls=":", color="#555555", lw=1.4,
                   label=f"SCAN{sfx}")
    beat = []
    for (aa, ss), c in cells.items():
        kind = (reference_by_arch or {}).get(aa, "pbe")
        verdict = (c.get("beats_scan") if kind == "scan"
                   else c.get("beats_pbe"))
        if verdict and c["ED"] > 0.0:
            beat.append((ss, c["ED"]))
    if beat:
        ax.scatter([s for s, _ in beat], [e for _, e in beat], marker="v",
                   s=16, color="#2ca02c", edgecolor="k", linewidths=0.3,
                   zorder=6, label=("beats rung reference"
                                    if reference_by_arch else "beats PBE"))
    ax.set_yscale("log")
    ax.set_xlabel("training subset_size", fontsize=8)
    ax.set_ylabel(f"{_ED_SYM} (kcal/mol)", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    _gamma_stamp(ax, summary)


def _ed_decomposition_panel(ax, summary: Dict[str, Any]) -> None:
    """Per-cell decomposition in the (E, gamma*D) plane, log-log: dotted y=x
    self-calibration locus (PBE sits on it exactly under self-calibration,
    off it under a fixed gamma -- the label branches on ``gamma_mode``; grey
    x), thin iso-ED harmonic contours at {0.5, 1, 2} x the PBE ED,
    subset_size digits on the cell points; below the locus =
    density-limited, above = energy-limited. Panel body shared by
    ``plot_combined_energy_density`` and ``plot_density_energy_overview``."""
    cells = summary["cells"]
    for (a, ss), c in sorted(cells.items()):
        if c["E"] <= 0.0 or c["gammaD"] <= 0.0:
            continue
        ax.scatter(c["E"], c["gammaD"], s=18, alpha=0.8,
                   color=ARCH_COLOR.get(a, "0.5"), edgecolor="none")
        ax.annotate(str(ss), (c["E"], c["gammaD"]), fontsize=5,
                    color=ARCH_COLOR.get(a, "0.5"), xytext=(2, 2),
                    textcoords="offset points")
    e_pbe = summary["e_pbe"]
    gd_pbe = summary["gamma"] * summary["d_pbe"]
    ax.scatter([e_pbe], [gd_pbe], marker="x", s=40, color="0.4",
               label=("PBE (on y=x by construction)"
                      if summary.get("gamma_mode",
                                     "self_calibrated") == "self_calibrated"
                      else "PBE"))
    fin_e = [c["E"] for c in cells.values() if c["E"] > 0.0] + [e_pbe]
    fin_g = [c["gammaD"] for c in cells.values()
             if c["gammaD"] > 0.0] + [gd_pbe]
    lo = 0.5 * min(fin_e + fin_g)
    hi = 2.0 * max(fin_e + fin_g)
    xs = np.geomspace(lo, hi, 256)
    ax.plot(xs, xs, ls=":", color="0.5", lw=1.0)
    # iso-ED harmonic contours: ED = c  <=>  y = 1/(2/c - 1/x), x > c/2
    for k in (0.5, 1.0, 2.0):
        cval = k * summary["ed_pbe"]
        xv = xs[xs > cval / 2.0 * (1.0 + 1e-9)]
        if not len(xv):
            continue
        yv = 1.0 / (2.0 / cval - 1.0 / xv)
        ax.plot(xv, yv, lw=1.1 if k == 1.0 else 0.7,
                color="0.55" if k == 1.0 else "0.75", zorder=1)
        ax.annotate(f"{_ED_SYM}={cval:.3g}", (xv[-1], yv[-1]), fontsize=5,
                    color="0.5", xytext=(-2, 2),
                    textcoords="offset points", ha="right")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("E, 2-subset WTMAD-2 (kcal/mol)", fontsize=8)
    ax.set_ylabel("$\\gamma$ * D (kcal/mol)", fontsize=8)
    ax.set_title("Energy vs rescaled density error per cell "
                 f"(iso-{_ED_SYM} contours)", fontsize=9)
    ax.grid(True, which="both", alpha=0.3)


def _ed_decomposition_rich_panel(ax, summary: Dict[str, Any]) -> None:
    """Enriched (E, gamma*D) decomposition: labeled iso-ED contour family at
    {0.25, 0.5, 0.75, 1, 1.5, 2, 3} x ED_PBE, light shading of the beats-PBE
    region (every point with ED < ED_PBE -- including the whole strip
    E < ED_PBE/2, where the harmonic mean cannot reach ED_PBE for any density
    error), per-arch subset-ordered trajectories through the cells, and the
    PBE anchor (on the dotted y=x locus under gamma self-calibration, off it
    under a fixed gamma -- label and stamp branch on ``gamma_mode``). Same
    summary contract as ``_ed_decomposition_panel`` (the compact version
    used by the ED figure and the held-out overview)."""
    cells = summary["cells"]
    e_pbe = summary["e_pbe"]
    gd_pbe = summary["gamma"] * summary["d_pbe"]
    ed_pbe = summary["ed_pbe"]
    fin_e = [c["E"] for c in cells.values() if c["E"] > 0.0] + [e_pbe]
    fin_g = [c["gammaD"] for c in cells.values()
             if c["gammaD"] > 0.0] + [gd_pbe]
    lo = 0.4 * min(fin_e + fin_g)
    hi = 2.5 * max(fin_e + fin_g)
    xs = np.geomspace(lo, hi, 512)
    # beats-PBE region: harmonic(x, y) < ED_PBE <=> y < 1/(2/ED_PBE - 1/x)
    # for x > ED_PBE/2, and every y when x <= ED_PBE/2
    with np.errstate(divide="ignore"):
        upper = np.where(xs > ed_pbe / 2.0 * (1.0 + 1e-9),
                         1.0 / (2.0 / ed_pbe - 1.0 / xs), hi)
    upper = np.clip(upper, lo, hi)
    ax.fill_between(xs, lo, upper, color="#2ca02c", alpha=0.08, zorder=0)
    # iso-ED contour family, labeled where each curve crosses the y=x locus
    for k in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
        cval = k * ed_pbe
        xv = xs[xs > cval / 2.0 * (1.0 + 1e-9)]
        if not len(xv):
            continue
        yv = 1.0 / (2.0 / cval - 1.0 / xv)
        ax.plot(xv, yv, lw=1.4 if k == 1.0 else 0.7,
                color="0.35" if k == 1.0 else "0.72", zorder=1)
        if lo < cval < hi:
            ax.annotate(f"{_ED_SYM} = {cval:.3g}" if k == 1.0
                        else f"{k:g}x",
                        (cval, cval), fontsize=5.5, color="0.4",
                        ha="left", va="bottom", rotation=-40,
                        xytext=(1, 1), textcoords="offset points")
    ax.plot(xs, xs, ls=":", color="0.5", lw=1.0, zorder=1)
    # per-arch subset-ordered trajectories through the cells
    archs = arch_style.sort_by_rung(sorted({a for a, _ in cells}))
    for a in archs:
        pts = sorted((ss, c["E"], c["gammaD"])
                     for (aa, ss), c in cells.items()
                     if aa == a and c["E"] > 0.0 and c["gammaD"] > 0.0)
        if not pts:
            continue
        col = ARCH_COLOR.get(a, "0.5")
        ax.plot([p[1] for p in pts], [p[2] for p in pts], lw=0.8,
                alpha=0.45, color=col, zorder=2)
        ax.scatter([p[1] for p in pts], [p[2] for p in pts], s=22,
                   alpha=0.9, color=col, edgecolor="k", linewidths=0.2,
                   zorder=3, label=a)
        for ss, e, g in pts:
            ax.annotate(str(ss), (e, g), fontsize=5, color=col,
                        xytext=(2, 2), textcoords="offset points")
    self_cal = summary.get("gamma_mode", "self_calibrated") == "self_calibrated"
    ax.scatter([e_pbe], [gd_pbe], marker="x", s=60, color="k", zorder=5,
               label=("PBE (on y=x by construction)" if self_cal else "PBE"))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("E, 2-subset WTMAD-2 (kcal/mol)", fontsize=9)
    ax.set_ylabel("$\\gamma$ * D (kcal/mol)", fontsize=9)
    ax.set_title(f"{_ED_SYM} decomposition -- iso-{_ED_SYM} contours, "
                 "beats-PBE region shaded, per-arch subset trajectories",
                 fontsize=10)
    ax.grid(True, which="both", alpha=0.25)
    ax.text(0.02, 0.02,
            _gamma_stamp_text(summary)
            + f"; shaded: {_ED_SYM} < {_ED_SYM} of PBE",
            transform=ax.transAxes, fontsize=6.5, color="#444444")
    if ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=6, ncol=2, loc="upper left")


def plot_ed_decomposition(summary: Dict[str, Any], out_path: Path,
                          run_id: str, *, note: str = "",
                          provenance: Optional[str] = None,
                          caveat: Optional[str] = None,
                          dataset: Optional[str] = None,
                          title: Optional[str] = None) -> Path:
    """Standalone enriched ED decomposition (WTMAD-2 leg): the ED figure's
    (E, gamma*D) panel promoted to its own canvas with a labeled iso-ED
    contour family, the beats-PBE region shaded, and per-arch subset-ordered
    trajectories. Consumes the same ``combined_ed_by_cell`` summary as the ED
    figure's headline, so the two views cannot drift. ``title`` overrides the
    footer title (the DFS-units variant renders a ``combined_ed_fixed_gamma``
    summary under its own heading; default preserved otherwise)."""
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(1, 1, figsize=(9.0, 8.0), squeeze=False)
        _ed_decomposition_rich_panel(axes[0][0], summary)
        _stamp_parity_footer(
            fig, run_id=run_id, note=note, provenance=provenance,
            caveat=caveat or _ED_CAVEAT, dataset=dataset,
            title=title or f"{_ED_SYM} decomposition (DFS Eq. 21) -- "
                           "held-out, NN vs PBE")
        fig.tight_layout(rect=(0, 0.06, 1, 0.90))
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return out_path


def plot_combined_energy_density(wt_summary: Dict[str, Any],
                                 mae_summary: Optional[Dict[str, Any]],
                                 out_path: Path, run_id: str, *,
                                 note: str = "",
                                 provenance: Optional[str] = None,
                                 caveat: Optional[str] = None,
                                 dataset: Optional[str] = None,
                                 panel_titles: Optional[Tuple[str, str]]
                                 = None,
                                 second_leg_placeholder: str
                                 = "MAE leg unavailable",
                                 title: Optional[str] = None) -> Path:
    """DFS Eq. 21 combined energy-density ED, held-out, NN vs PBE:
    (a) headline ED with the 2-subset WTMAD-2 energy leg vs subset_size per
    arch, PBE dashed at its own ED (== its energy error under gamma
    self-calibration; off it under a fixed gamma); (b) the per-cell
    (E, gamma*D) decomposition with iso-ED harmonic contours -- PBE sits on
    the dotted y=x locus exactly when gamma is self-calibrated, cells below
    it are density-limited, above it energy-limited;
    (c) the reaction-MAE-leg ED, the leg-independence check (its own gamma).
    ``wt_summary`` / ``mae_summary`` are ``combined_ed_by_cell`` outputs; a
    None/empty ``mae_summary`` renders a placeholder panel. Non-positive
    points are dropped defensively (log axes; cannot occur for real
    MAE/WTMAD-2/D > 0). The line-panel body lives in ``_ed_lines_panel``
    (shared with the held-out overview). See the ED section note for the
    deviations from the Letter.

    ``panel_titles`` (a, c), ``second_leg_placeholder``, and ``title``
    override the panel headings, the empty-panel-C text, and the footer
    title; defaults reproduce the historical WTMAD-2/MAE figure exactly. The
    DFS-units variant passes ``combined_ed_fixed_gamma`` summaries here (the
    panel bodies branch their stamps/labels on ``gamma_mode`` themselves)
    with panel C = the own-axes-fit leg instead of the MAE leg."""

    title_a, title_c = panel_titles or (
        f"{_ED_SYM}, energy leg = 2-subset WTMAD-2 (headline)",
        f"{_ED_SYM}, energy leg = combined reaction MAE "
        "(leg-independence check)")
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6), squeeze=False)
        axA, axB, axC = axes[0]
        _ed_lines_panel(axA, wt_summary, title_a,
                        reference_by_arch=arch_reference_kinds(
                            {a for a, _ in wt_summary['cells']}))

        # (b) decomposition: one point per cell in (E, gamma*D) space
        _ed_decomposition_panel(axB, wt_summary)

        if mae_summary and mae_summary.get("cells"):
            _ed_lines_panel(axC, mae_summary, title_c,
                            reference_by_arch=arch_reference_kinds(
                                {a for a, _ in mae_summary['cells']}))
        else:
            axC.text(0.5, 0.5, second_leg_placeholder, ha="center",
                     va="center", transform=axC.transAxes, fontsize=9,
                     color="0.5")
            axC.set_title(title_c, fontsize=9)

        seen: Dict[str, Any] = {}
        for ax in (axA, axB, axC):
            hs, ls = ax.get_legend_handles_labels()
            for h, l in zip(hs, ls):
                seen.setdefault(l, h)
        if seen:
            fig.legend(list(seen.values()), list(seen.keys()),
                       loc="lower center",
                       ncol=max(4, len(arch_style.RUNG_ORDER)), fontsize=7,
                       frameon=False, bbox_to_anchor=(0.5, 0.04))
        _stamp_parity_footer(
            fig, run_id=run_id, note=note, provenance=provenance,
            caveat=caveat or _ED_CAVEAT, dataset=dataset,
            title=title or f"Combined energy-density {_ED_SYM} (DFS "
                           "Eq. 21, harmonic mean) -- held-out, NN vs PBE")
        fig.tight_layout(rect=(0, 0.10, 1, 0.90))
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return out_path


def _overview_provenance(ed_summary: Optional[Dict[str, Any]]) -> str:
    """Footer for the held-out overview composite. The SCAN sentence tracks
    what panel F actually draws -- the SCAN ED comparator appears exactly when
    ``ed_summary`` carries a finite ``ed_scan`` (``_ed_lines_panel``), so a
    fixed footer would contradict the panel in one of the two states."""
    scan_part = ("SCAN: only panel F's ED comparator line (coverage-"
                 "gated); A-E carry no SCAN references. "
                 if _is_num((ed_summary or {}).get("ed_scan"))
                 else "no SCAN lines (no SCAN WTMAD-2 cache). ")
    return ("Held-out overview. A/B: one-bucket WTMAD-2 reduction "
            "56.84*MAD/mean|ref| per pool; C: 2-subset WTMAD-2; "
            + scan_part +
            "D/E: grid-weight-averaged density RMSE vs CCSD (not CCSD(T)) "
            "refs at matching basis/grid, PBE model-free on the same grid. "
            "F: ED -- full diagnostics on "
            "ablation_combined_energy_density.png.")


def plot_density_energy_overview(rows: List[Dict[str, Any]],
                                 hd_rows: List[Dict[str, Any]],
                                 out_path: Path, run_id: str, *,
                                 pbe_table: Optional[Dict[str, Dict[str, float]]]
                                 = None,
                                 ed_summary: Optional[Dict[str, Any]] = None,
                                 note: str = "",
                                 provenance: Optional[str] = None,
                                 caveat: Optional[str] = None,
                                 dataset: Optional[str] = None,
                                 parity_nn_key: str = "density_rmse",
                                 parity_pbe_key: str = "density_rmse_pbe",
                                 parity_unit_label: str = "density RMSE",
                                 title: Optional[str] = None) -> Path:
    """Held-out overview composite -- energy above, the energy-density TRADE
    below: (A)/(B) single-pool WTMAD-2 bars per (arch, subset_size) for BH76 /
    W4-11 (with one pool the WTMAD-2 sum collapses to
    56.84 * MAD_pool / mean|ref|_pool -- a scaled relative error, stamped as
    such), (C) the genuine 2-subset WTMAD-2, (D) per-species NN-vs-PBE density
    parity, (E) the per-cell (E, gamma*D) iso-ED decomposition, (F) the DFS
    Eq. 21 ED headline. (E)/(F) take ``ed_summary`` (a ``combined_ed_by_cell``
    output) and degrade to grey placeholders when it is missing/empty. The
    per-arch density-vs-subset trend lives in its own figure
    (``plot_holdout_density_per_arch``) and in the left panel of
    ``plot_holdout_density_ccsd``. Panel bodies are the same ax-level helpers
    the dedicated figures use, so the views cannot drift apart. Panel F draws
    the SCAN ED comparator line when ``ed_summary`` carries a finite
    ``ed_scan`` (label coverage-qualified via ``scan_suffix``); panels A-E
    never draw SCAN references, and ``_overview_provenance`` states which
    state rendered. Each top panel carries its own pool-filtered PBE dashed
    line.

    ``parity_nn_key``/``parity_pbe_key``/``parity_unit_label`` select panel
    D's density channel and ``title`` overrides the footer title; defaults
    reproduce the historical figure exactly. The DFS-units twin passes a
    ``combined_ed_fixed_gamma`` summary (panels E/F branch their stamps on
    ``gamma_mode`` themselves) with the eps parity keys."""
    pbe_mol = _pbe_density_map(hd_rows, pbe_table, key=parity_pbe_key)
    with plt.rc_context(_STYLE):
        archs = _energy_arch_axis(rows)
        subsets = _present_subsets(rows) or [1]
        fig, axes = plt.subplots(2, 3, figsize=(18.0, 9.6), squeeze=False)
        (axA, axB, axC), (axD, axE, axF) = axes
        for ax, pool, tag in ((axA, "bh76", "(A) WTMAD-2, BH76 only"),
                              (axB, "w411", "(B) WTMAD-2, W4-11 only")):
            pr = [r for r in rows if r.get("pool") == pool]
            _grouped_arch_bars(ax, wtmad2_by_arch_subset(pr), archs, subsets,
                               pbe_line=wtmad2_pbe_baseline(pr),
                               pbe_by_cell=wtmad2_pbe_by_arch_subset(pr),
                               reference_by_arch=arch_reference_kinds(archs),
                               incomplete_cells=set(
                                   _incomplete_energy_cells(pr)),
                               title=tag + " -- one-bucket reduction "
                                     "(scaled relative error)",
                               vxc_pre_fix=_run_predates_vxc_fix(run_id))
        _grouped_arch_bars(axC, wtmad2_by_arch_subset(rows), archs, subsets,
                           pbe_line=wtmad2_pbe_baseline(rows),
                           pbe_by_cell=wtmad2_pbe_by_arch_subset(rows),
                           reference_by_arch=arch_reference_kinds(archs),
                           incomplete_cells=set(
                               _incomplete_energy_cells(rows)),
                           title="(C) 2-subset WTMAD-2 (BH76+W4-11), "
                                 "per (arch, subset)",
                           vxc_pre_fix=_run_predates_vxc_fix(run_id))
        _density_parity_panel(axD, hd_rows, pbe_mol, nn_key=parity_nn_key,
                              unit_label=parity_unit_label)
        axD.set_title("(D) " + axD.get_title(), fontsize=9)
        if ed_summary and ed_summary.get("cells"):
            _ed_decomposition_panel(axE, ed_summary)
            axE.set_title("(E) " + axE.get_title(), fontsize=9)
            _ed_lines_panel(axF, ed_summary,
                            f"(F) {_ED_SYM}, energy leg = 2-subset "
                            "WTMAD-2 (headline)",
                            reference_by_arch=arch_reference_kinds(
                                {a for a, _ in ed_summary['cells']}))
        else:
            axE.text(0.5, 0.5, f"{_ED_SYM} decomposition unavailable",
                     ha="center", va="center", transform=axE.transAxes,
                     fontsize=9, color="0.5")
            axE.set_title("(E) Energy vs rescaled density error per cell "
                          f"(iso-{_ED_SYM} contours)", fontsize=9)
            axF.text(0.5, 0.5, f"{_ED_SYM} unavailable", ha="center",
                     va="center", transform=axF.transAxes, fontsize=9,
                     color="0.5")
            axF.set_title(f"(F) {_ED_SYM}, energy leg = 2-subset WTMAD-2 "
                          "(headline)", fontsize=9)
        seen: Dict[str, Any] = {}
        for ax in (axA, axB, axC, axD, axE, axF):
            hs, ls = ax.get_legend_handles_labels()
            for h, l in zip(hs, ls):
                seen.setdefault(l, h)
        if seen:
            fig.legend(list(seen.values()), list(seen.keys()),
                       loc="lower center",
                       ncol=max(4, len(arch_style.RUNG_ORDER)), fontsize=7,
                       frameon=False, bbox_to_anchor=(0.5, 0.04))
        _stamp_parity_footer(
            fig, run_id=run_id, note=note, provenance=provenance,
            caveat=caveat or _HOLDOUT_OVERVIEW_CAVEAT, dataset=dataset,
            title=title or "Held-out overview: WTMAD-2 by pool + density "
                           "vs CCSD + ED")
        fig.tight_layout(rect=(0, 0.10, 1, 0.90))
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return out_path


# Two-line caveats (\n): the 14-inch 3x3 canvas has room for a second
# caveat line above the dataset band, and the shorter longest-line keeps
# bbox="tight" from widening the canvas. Line 1 defines the single-pool
# "one-bucket" WTMAD-2 reduction explicitly.
_3X3_CAVEAT = (
    "Columns are channels: BH76 | W4-11 | combined. A/B and the ED legs in "
    "G/H use the SINGLE-POOL 'WTMAD-2', which collapses to "
    "56.84*MAD_pool/mean|dE_ref|_pool -- the pool's mean abs deviation "
    "rescaled by its mean abs reference energy: a scaled relative error, "
    "NOT a reweighting (only the combined column reweights; NOT full "
    "GMTKN55).\n"
    + _ED_SYM + " = 2/(1/E + 1/(gamma*D)) with gamma = E_PBE/D_PBE from "
    "that channel's OWN anchors (value printed in each panel), so "
    + _ED_SYM + " of PBE == E_PBE per channel and " + _ED_SYM + " never "
    "compares across channels. Overlap species appear in both density "
    "channels; per-species parity in "
    "ablation_density_parity_by_channel.png.")

_3X3_DFS_UNITS_CAVEAT = (
    "Columns are channels: BH76 | W4-11 | combined. A/B and the G/H ED "
    "energy legs use the SINGLE-POOL 'WTMAD-2', which collapses to "
    "56.84*MAD_pool/mean|dE_ref|_pool -- the pool's mean abs deviation "
    "rescaled by its mean abs reference energy: a scaled relative error, "
    "NOT a reweighting (only the combined column reweights; NOT full "
    "GMTKN55).\n"
    + _ED_N_SYM + " = 2/(1/E + 1/(gamma " + _EPS_N_SYM + ")); " + _EPS_N_EQ
    + " (Eq. 20 per species); ONE gamma SHARED by all channels, value + "
    "source stamped "
    "in each panel (own-axes six-functional fit when the calibration cache "
    "resolves, the Letter's published 1084.87 otherwise) -- "
    + _ED_N_SYM + " compares across columns; " + _ED_N_SYM + " of PBE != "
    "E_PBE. Density row = cell-mean " + _EPS_N_SYM + "; per-species parity "
    "in ablation_density_parity_by_channel_dfs_units.png.")

_HOLDOUT_OVERVIEW_DFS_UNITS_CAVEAT = (
    "Single-pool 'WTMAD-2' (A, B) reduces to 56.84*MAD_pool/mean|ref|_pool "
    "-- a scaled relative error, not a reweighting; only (C) reweights (NOT "
    "full GMTKN55). E/F: " + _ED_N_SYM + " = 2/(1/E + 1/(gamma "
    + _EPS_N_SYM + ")); " + _EPS_N_EQ + " (Eq. 20 per species); gamma "
    "EXTERNALLY "
    "FIXED, value + source stamped in-panel (own-axes fit when the "
    "calibration cache resolves, published 1084.87 otherwise) -- "
    + _ED_N_SYM + " of PBE != E_PBE, PBE off y=x. (D) parity in "
    + _EPS_N_SYM + " units.")


def plot_density_energy_3x3(rows: List[Dict[str, Any]],
                            hd_rows: List[Dict[str, Any]],
                            out_path: Path, run_id: str, *,
                            pbe_table: Optional[Dict[str, Dict[str, float]]]
                            = None,
                            ch_summaries: Optional[Dict[str, Optional[
                                Dict[str, Any]]]] = None,
                            note: str = "",
                            provenance: Optional[str] = None,
                            caveat: Optional[str] = None,
                            dataset: Optional[str] = None,
                            density_nn_key: str = "density_rmse",
                            density_pbe_key: str = "density_rmse_pbe",
                            density_unit_label: str = "density RMSE",
                            ed_gamma_label: str = "own gamma",
                            lockfix_cells: Optional[Dict[str, set]] = None,
                            scan_density_records: Optional[
                                Dict[str, Dict[str, Any]]] = None,
                            scan_errors: Optional[Dict[str, float]] = None,
                            title: Optional[str] = None) -> Path:
    """Per-channel held-out story, ALL BARS, one column per channel
    (BH76 | W4-11 | combined): row 1 = WTMAD-2 bars per (arch, subset_size)
    (A/B the one-bucket reduction, C the genuine 2-subset form); row 2 =
    the density-error bars on the ``density_nn_key`` channel, restricted to
    that channel's species (cell mean; PBE dashed at the channel's
    deduplicated anchor; overlap species contribute to both single-pool
    columns, stated in the caveat); row 3 = the DFS Eq. 21 combined metric
    per channel as bars (PBE dashed at its own combined value, the gamma
    stamp in-panel). Channels missing data render grey placeholders. The
    per-species parity view lives in its own figure,
    :func:`plot_density_parity_by_channel`.

    The keyword overrides (density keys/label, ``ed_gamma_label``,
    ``title``) default to the grid-weighted-RMSE original; the DFS-units
    twin passes fixed-gamma ``ch_summaries`` (from ``channel_ed_summaries``
    with ``fixed_gamma``/eps keys), the eps density keys, the
    ``$\\varepsilon_{|n|}$`` unit label, and a clean row-3 gamma tag."""
    if ch_summaries is None:
        ch_summaries = channel_ed_summaries(rows, hd_rows, pbe_table)
    pools_of = _species_pools(rows)
    pbe_mol = _pbe_density_map(hd_rows, pbe_table, key=density_pbe_key)
    # reference-provenance glyphs; empty dict -> bars render exactly as before
    _lf = ({"relocked_cells": (lockfix_cells or {}).get("relocked"),
            "mixed_cells": (lockfix_cells or {}).get("mixed")}
           if lockfix_cells else {})
    _lf["vxc_pre_fix"] = _run_predates_vxc_fix(run_id)
    with plt.rc_context(_STYLE):
        archs = _energy_arch_axis(rows)
        subsets = _present_subsets(rows) or [1]
        fig, axes = plt.subplots(3, 3, figsize=(18.0, 14.0), squeeze=False)
        chans = (("bh76", "BH76"), ("w411", "W4-11"),
                 ("combined", "combined"))
        letters = "ABCDEFGHI"
        # Row-1 incomplete-eval star sets, per channel; row 3's ED bars
        # reuse them (their energy leg is the incomplete one; the density
        # row has complete coverage and its own union check).
        inc_by_ch: Dict[str, set] = {}
        for j, (ch, lab) in enumerate(chans):
            pr = rows if ch == "combined" else [
                r for r in rows if r.get("pool") == ch]
            inc_by_ch[ch] = set(_incomplete_energy_cells(pr))
            ttl = (f"({letters[j]}) 2-subset WTMAD-2 (BH76+W4-11), "
                   "per (arch, subset)" if ch == "combined" else
                   f"({letters[j]}) WTMAD-2, {lab} only -- one-bucket "
                   "reduction (scaled relative error)")
            # SCAN's WTMAD-2 on this channel, reduced over the reactions the
            # PBE leg reduced; None below the coverage floor or with no cache.
            e_scan_ch, _u, _r = wtmad2_scan_baseline(pr, scan_errors)
            if not (_r and (_u / _r) >= _SCAN_COVERAGE_FLOOR):
                e_scan_ch = None
            _grouped_arch_bars(axes[0][j], wtmad2_by_arch_subset(pr), archs,
                               subsets, pbe_line=wtmad2_pbe_baseline(pr),
                               pbe_by_cell=wtmad2_pbe_by_arch_subset(pr),
                               reference_by_arch=arch_reference_kinds(archs),
                               incomplete_cells=inc_by_ch[ch],
                               scan_line=e_scan_ch,
                               scan_by_cell=wtmad2_scan_by_cell(pr,
                                                                scan_errors),
                               scan_suffix=(f", {_u}/{_r}"
                                            if e_scan_ch is not None
                                            and _u < _r else ""),
                               title=ttl, **_lf)
        for j, (ch, lab) in enumerate(chans):
            ax = axes[1][j]
            if ch == "combined":
                hd_ch, pbe_ch = hd_rows, pbe_mol
            else:
                hd_ch = [r for r in hd_rows
                         if ch in pools_of.get(r.get("molecule"), ())]
                pbe_ch = {m: v for m, v in pbe_mol.items()
                          if ch in pools_of.get(m, ())}
            d_map = holdout_density_by_arch_subset(hd_ch,
                                                   key=density_nn_key)
            ttl_d = (f"({letters[3 + j]}) {density_unit_label} vs CCSD, "
                     f"{lab} species -- cell mean")
            if d_map:
                d_pbe_ch = (float(np.mean(list(pbe_ch.values())))
                            if pbe_ch else float("nan"))
                # SCAN over the SAME channel species the PBE anchor averages.
                d_scan_ch, d_ch_u, d_ch_r = scan_density_line_counts(
                    scan_density_records, pbe_ch, key=density_pbe_key)
                _grouped_arch_bars(
                    ax, d_map, archs, subsets,
                    pbe_line=(d_pbe_ch if _is_num(d_pbe_ch)
                              and d_pbe_ch > 0.0 else None),
                    pbe_by_cell=pbe_density_by_cell(
                        hd_ch, None, nn_key=density_nn_key,
                        pbe_key=density_pbe_key, _pbe_mol=pbe_ch),
                    reference_by_arch=arch_reference_kinds(archs),
                    scan_line=d_scan_ch,
                    scan_by_cell=scan_density_by_cell(
                        hd_ch, scan_density_records, None,
                        nn_key=density_nn_key, pbe_key=density_pbe_key,
                        _pbe_mol=pbe_ch),
                    scan_suffix=(f", {d_ch_u}/{d_ch_r}"
                                 if d_scan_ch is not None
                                 and d_ch_u < d_ch_r else ""),
                    title=ttl_d, **_lf)
                ax.set_ylabel(f"{density_unit_label} vs CCSD", fontsize=8)
            else:
                ax.text(0.5, 0.5, "density unavailable", ha="center",
                        va="center", transform=ax.transAxes, fontsize=9,
                        color="0.5")
                ax.set_title(ttl_d, fontsize=9)
        for j, (ch, lab) in enumerate(chans):
            ax = axes[2][j]
            s = ch_summaries.get(ch)
            ttl = (f"({letters[6 + j]}) {_ED_SYM}, {lab} channel"
                   + (f" ({ed_gamma_label})" if ed_gamma_label else ""))
            if s and s.get("cells"):
                ed_map = {c: v["ED"] for c, v in s["cells"].items()
                          if _is_num(v.get("ED")) and v["ED"] > 0.0}
                ed_pbe = s.get("ed_pbe")
                ed_scan = s.get("ed_scan")
                ed_cell_anchors = {c: v["ed_pbe_cell"]
                                   for c, v in s["cells"].items()
                                   if _is_num(v.get("ed_pbe_cell"))}
                ed_scan_anchors = {c: v["ed_scan_cell"]
                                   for c, v in s["cells"].items()
                                   if _is_num(v.get("ed_scan_cell"))}
                _grouped_arch_bars(
                    ax, ed_map, archs, subsets,
                    pbe_line=(float(ed_pbe) if _is_num(ed_pbe)
                              and ed_pbe > 0.0 else None),
                    pbe_by_cell=(ed_cell_anchors or None),
                    reference_by_arch=arch_reference_kinds(archs),
                    incomplete_cells=inc_by_ch.get(ch),
                    scan_line=(float(ed_scan) if _is_num(ed_scan)
                               and ed_scan > 0.0 else None),
                    scan_by_cell=(ed_scan_anchors or None),
                    scan_suffix=(s.get("scan_suffix") or ""),
                    title=ttl, **_lf)
                ax.set_ylabel(f"{_ED_SYM} (kcal/mol)", fontsize=8)
                _gamma_stamp(ax, s)
            else:
                ax.text(0.5, 0.5, f"{_ED_SYM} unavailable", ha="center",
                        va="center", transform=ax.transAxes, fontsize=9,
                        color="0.5")
                ax.set_title(ttl, fontsize=9)
        seen: Dict[str, Any] = {}
        for row_axes in axes:
            for ax in row_axes:
                hs, ls = ax.get_legend_handles_labels()
                for h, l in zip(hs, ls):
                    seen.setdefault(l, h)
        if seen:
            # anchored above the red note band (0.032) so a two-row legend
            # stacks upward into the reserved bottom margin, never onto it
            fig.legend(list(seen.values()), list(seen.keys()),
                       loc="lower center",
                       ncol=max(4, len(arch_style.RUNG_ORDER)), fontsize=7,
                       frameon=False, bbox_to_anchor=(0.5, 0.045))
        _stamp_parity_footer(
            fig, run_id=run_id, note=note, provenance=provenance,
            caveat=caveat or _3X3_CAVEAT, dataset=dataset,
            title=title or "Per-channel held-out story: WTMAD-2 | density "
                           "error | ED (BH76, W4-11, combined)")
        fig.tight_layout(rect=(0, 0.085, 1, 0.90))
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return out_path


_PARITY_BY_CHANNEL_DFS_UNITS_CAVEAT = (
    "Per-species " + _EPS_N_SYM + " vs CCSD (DFS Eq. 20: "
    + _EPS_N_EQ + " per species; quadrature "
    "sum_i(w_i|rho-rho_ref|_i)/N_e), NN vs the "
    "model-free PBE baseline on the same grid; below the diagonal = the NN "
    "density is closer to CCSD. One shared square frame across the three "
    "channel panels; overlap species appear in both single-pool panels.")


def plot_density_parity_by_channel(rows: List[Dict[str, Any]],
                                   hd_rows: List[Dict[str, Any]],
                                   out_path: Path, run_id: str, *,
                                   pbe_table: Optional[Dict[str, Dict[
                                       str, float]]] = None,
                                   nn_key: str = "density_rmse",
                                   pbe_key: str = "density_rmse_pbe",
                                   unit_label: str = "density RMSE",
                                   note: str = "",
                                   provenance: Optional[str] = None,
                                   caveat: Optional[str] = None,
                                   dataset: Optional[str] = None,
                                   title: Optional[str] = None) -> Path:
    """Per-species NN-vs-PBE density parity, one panel per channel
    (BH76 | W4-11 | combined species; membership from the reactions'
    reactants+products via ``_species_pools``, overlap species in both
    single-pool panels), all three panels in ONE shared square frame (the
    pooled positive envelope) so the channels are directly comparable --
    the 3x3's former parity row promoted to its own figure. ``nn_key`` /
    ``pbe_key`` / ``unit_label`` select the error channel exactly as in
    ``_density_parity_panel``."""
    pools_of = _species_pools(rows)
    pbe_mol = _pbe_density_map(hd_rows, pbe_table, key=pbe_key)
    # one shared square envelope over ALL species (the combined panel's
    # superset): identical x and y limits on every panel
    row_pairs: List[float] = []
    for r in hd_rows:
        x = pbe_mol.get(r.get("molecule"))
        y = r.get(nn_key)
        if _is_num(x) and x > 0.0 and _is_num(y) and y > 0.0:
            row_pairs.extend((float(x), float(y)))
    row_limits = ((0.8 * min(row_pairs), 1.25 * max(row_pairs))
                  if row_pairs else None)
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6), squeeze=False)
        chans = (("bh76", "BH76"), ("w411", "W4-11"),
                 ("combined", "combined"))
        for j, (ch, lab) in enumerate(chans):
            ax = axes[0][j]
            if ch == "combined":
                hd_ch, pbe_ch = hd_rows, pbe_mol
            else:
                hd_ch = [r for r in hd_rows
                         if ch in pools_of.get(r.get("molecule"), ())]
                pbe_ch = {m: v for m, v in pbe_mol.items()
                          if ch in pools_of.get(m, ())}
            _density_parity_panel(ax, hd_ch, pbe_ch, nn_key=nn_key,
                                  unit_label=unit_label, limits=row_limits)
            ax.set_title(f"({'ABC'[j]}) {lab} species -- "
                         + ax.get_title(), fontsize=9)
        # The panels color every species point by architecture with no
        # per-point labels; without a figure legend the colors are
        # undecodable.
        archs_present = arch_style.sort_by_rung(
            sorted({str(r.get("arch")) for r in hd_rows if r.get("arch")}))
        handles = [plt.Line2D([], [], marker="o", ls="",
                              color=ARCH_COLOR.get(a, "0.5"),
                              markersize=6, label=a)
                   for a in archs_present]
        if handles:
            fig.legend(handles=handles, loc="lower center",
                       ncol=min(len(handles), 6), fontsize=7,
                       frameon=False, bbox_to_anchor=(0.5, 0.075))
        _stamp_parity_footer(
            fig, run_id=run_id, note=note, provenance=provenance,
            caveat=caveat or _HOLDOUT_DENSITY_CAVEAT, dataset=dataset,
            title=title or "Per-species density parity by channel -- "
                           "held-out, NN vs PBE")
        fig.tight_layout(rect=(0, 0.13, 1, 0.90))
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return out_path


def plot_insample_overview(ae_rows: List[Dict[str, Any]],
                           density_rows: List[Dict[str, Any]],
                           out_path: Path, run_id: str, *,
                           note: str = "",
                           provenance: Optional[str] = None,
                           caveat: Optional[str] = None) -> Path:
    """In-sample companion composite (training fit, final checkpoint):
    (A) in-sample AE MAE bars per (arch, subset_size) -- NN only, no PBE line
    (the in-sample per_molecule.json has no PBE AE column, which is also why
    no in-sample ED exists: no PBE energy anchor to self-calibrate gamma);
    (B) per-molecule |AE error| strip; (C) in-sample density RMSE vs
    subset_size with the PBE dashed line when the model-free column is
    present; (D) per-molecule density strip with one grey PBE x per molecule.
    ``eval/`` has no val-best variant, so the panels are identical in the
    final-step and val-best output dirs -- only the title's checkpoint stamp
    differs (stated in the caveat). Panel bodies are the same ax-level
    helpers ``plot_insample_density_ccsd`` uses."""
    with plt.rc_context(_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(13, 9.6), squeeze=False)
        (axA, axB), (axC, axD) = axes
        _grouped_arch_bars(axA, ae_mae_by_arch_subset(ae_rows),
                           _energy_arch_axis(ae_rows),
                           _present_subsets(ae_rows) or [1],
                           pbe_line=None,
                           vxc_pre_fix=_run_predates_vxc_fix(run_id),
                           title="(A) In-sample AE MAE per (arch, subset) -- "
                                 "NN only (no in-sample PBE AE)")
        axA.set_ylabel("in-sample AE MAE (kcal/mol)", fontsize=8)
        _insample_ae_strip_panel(axB, ae_rows)
        axB.set_title("(B) " + axB.get_title(), fontsize=9)
        _insample_density_lines_panel(axC, density_rows)
        axC.set_title("(C) " + axC.get_title(), fontsize=9)
        _insample_density_strip_panel(axD, density_rows)
        axD.set_title("(D) " + axD.get_title(), fontsize=9)
        seen: Dict[str, Any] = {}
        for ax in (axA, axB, axC, axD):
            hs, ls = ax.get_legend_handles_labels()
            for h, l in zip(hs, ls):
                seen.setdefault(l, h)
        if seen:
            fig.legend(list(seen.values()), list(seen.keys()),
                       loc="lower center",
                       ncol=max(4, len(arch_style.RUNG_ORDER)), fontsize=7,
                       frameon=False, bbox_to_anchor=(0.5, 0.04))
        _stamp_parity_footer(
            fig, run_id=run_id, note=note, provenance=provenance,
            caveat=caveat or _INSAMPLE_OVERVIEW_CAVEAT,
            title="In-sample overview: AE + density vs CCSD "
                  "(training fit; final checkpoint)")
        fig.tight_layout(rect=(0, 0.10, 1, 0.90))
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return out_path


def collect_training_losses(run_dir: Path,
                            basis_label: Optional[str] = None
                            ) -> List[Dict[str, Any]]:
    """Per-spec training-loss trajectory from ``losses.npy`` (the per-group-update
    loss recorded during training), joined with the manifest arch/subset_size.
    Each row is tagged with ``basis`` (``basis_label``) so several runs can be
    merged into one cumulative plot."""
    cells = ccp._read_manifest_cells(run_dir)
    rows: List[Dict[str, Any]] = []
    for idx, spec_dir in ccp._spec_dirs(run_dir):
        lp = spec_dir / "losses.npy"
        if not lp.is_file():
            continue
        try:
            # losses.npy is a trusted plain float array (this run's own training
            # output) -> no pickle needed (allow_pickle stays False).
            losses = np.asarray(np.load(lp), float).ravel()
        except (ValueError, OSError):
            continue
        cell = cells.get(idx, {})
        rows.append({"idx": idx, "arch": cell.get("arch"),
                     "subset_size": cell.get("subset_size"), "losses": losses,
                     "basis": basis_label})
    return rows


def collect_training_losses_multi(runs: List[Tuple[Path, str]]
                                  ) -> List[Dict[str, Any]]:
    """Concatenate :func:`collect_training_losses` across several
    ``(run_dir, basis_label)`` pairs so EVERY trained cell from EVERY run lands in
    one cumulative loss plot. Cells trained in more than one basis (e.g. ``deep``,
    ``deep_attn``) yield one row per basis."""
    rows: List[Dict[str, Any]] = []
    for run_dir, basis_label in runs:
        rows.extend(collect_training_losses(run_dir, basis_label=basis_label))
    return rows


def _rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or x.size < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="valid")


def plot_training_losses(loss_rows: List[Dict[str, Any]], out_path: Path,
                         run_id: str, note: str = "",
                         provenance: Optional[str] = None,
                         highlight: Optional[List[Tuple[str, int]]] = None) -> Path:
    """Training-loss trajectories faceted by arch, one curve per subset_size
    (viridis), rolling-mean-smoothed, log-y. A run that destabilizes late (its
    loss climbs back up) stands out -- e.g. deep_attn ss6. When the rows carry
    more than one ``basis`` (e.g. def2-svp + def2-tzvpd+DF), basis is shown by
    LINESTYLE so every trained cell from every run appears together (cells shared
    across bases get one curve per basis)."""
    _LS = ["-", "--", "-.", ":"]
    with plt.rc_context(_STYLE):
        present = {r["arch"] for r in loss_rows if r.get("arch")}
        archs = [a for a in ARCH_ORDER if a in present]
        archs += sorted(present - set(archs))
        archs = archs or ["deep"]
        subset_values = sorted({r["subset_size"] for r in loss_rows
                                if r.get("subset_size") is not None})
        # basis -> linestyle (stable order: bases as first seen in the rows)
        bases: List[Any] = []
        for r in loss_rows:
            b = r.get("basis")
            if b not in bases:
                bases.append(b)
        ls_for = {b: _LS[i % len(_LS)] for i, b in enumerate(bases)}
        multi_basis = len([b for b in bases if b is not None]) > 1
        hl = set(highlight or [])
        n = len(archs)
        ncols = 2 if n > 1 else 1
        nrows = max(1, math.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(6.7 * ncols, 4.2 * nrows),
                                 squeeze=False)
        flat = axes.ravel()
        norm = matplotlib.colors.Normalize(
            vmin=min(subset_values) if subset_values else 0,
            vmax=max(subset_values) if subset_values else 1)
        cmap = plt.get_cmap("viridis")
        for ai, arch in enumerate(archs):
            ax = flat[ai]
            for r in sorted((r for r in loss_rows if r.get("arch") == arch),
                            key=lambda r: (r.get("subset_size") or 0,
                                           str(r.get("basis")))):
                L = r["losses"]
                if L.size == 0:
                    continue
                s = _rolling_mean(L, max(1, L.size // 75))
                xs = np.linspace(0.0, 1.0, s.size)
                is_hl = (arch, r.get("subset_size")) in hl
                ax.plot(xs, np.clip(s, 1e-14, None), color=cmap(norm(r["subset_size"])),
                        ls=ls_for.get(r.get("basis"), "-"),
                        lw=2.6 if is_hl else 1.0, alpha=0.95 if is_hl else 0.8,
                        zorder=5 if is_hl else 3)
                if is_hl:
                    ax.annotate(f"ss{r['subset_size']} (unstable)", (xs[-1], s[-1]),
                                fontsize=6.5, color="#a33", ha="right",
                                xytext=(0, 6), textcoords="offset points")
            ax.set_yscale("log")
            ax.set_title(arch, fontsize=9)
            ax.set_xlabel("training progress (fraction of updates)", fontsize=7.5)
            ax.set_ylabel("loss (rolling mean, log)", fontsize=7.5)
            ax.grid(True, which="both", alpha=0.3)
            ax.tick_params(labelsize=6.5)
        for k in range(len(archs), len(flat)):
            flat[k].axis("off")
        # basis legend (linestyle key), only when several bases are overlaid
        if multi_basis:
            handles = [plt.Line2D([], [], color="0.3", ls=ls_for[b],
                                  label=str(b)) for b in bases if b is not None]
            flat[0].legend(handles=handles, title="basis", fontsize=6.5,
                           title_fontsize=6.5, loc="upper right", framealpha=0.7)
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        fig.tight_layout(rect=(0, 0.05, 0.93, 0.93))
        cax = fig.add_axes([0.945, 0.22, 0.012, 0.5])
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label("training subset_size", fontsize=7)
        cbar.ax.tick_params(labelsize=6)
        title = "Training-loss trajectories by architecture (per-subset"
        title += ", basis=linestyle)" if multi_basis else ")"
        _stamp_parity_footer(
            fig, run_id=run_id, note=note, provenance=provenance, caveat=None,
            title=title)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return out_path


def _final_window_loss(losses: Any, n: int = 50) -> float:
    """Mean of the last ``n`` training-loss steps -- represents the FINAL
    checkpoint the eval actually loads (``model.eqx``), so a late blow-up shows
    here even though the best-ever loss was tiny."""
    L = np.asarray(losses, float).ravel()
    if L.size == 0:
        return float("nan")
    return float(np.mean(L[-min(n, L.size):]))


def _classify_cell(heldout_mae: Optional[float], pbe_mae: Optional[float],
                   final_loss: float, cohort_median: float,
                   instab_factor: float = 5.0) -> str:
    """Mechanism of a single cell. ``pass`` = held-out MAE <= PBE. Among the
    failures: ``late_instability`` when the FINAL-window loss is an ABSOLUTE
    outlier vs the cohort (training itself diverged late -- the deep_attn-ss6
    case); otherwise ``generalization_gap`` (train loss is healthy but the model
    overfits the tiny held-in subset). The ratio final/best is NOT used -- it is
    huge for healthy cells too (noisy per-batch SCF loss)."""
    if not _is_num(heldout_mae) or not _is_num(pbe_mae):
        return "pass"
    if heldout_mae <= pbe_mae:
        return "pass"
    if (_is_num(final_loss) and _is_num(cohort_median) and cohort_median > 0
            and final_loss > instab_factor * cohort_median):
        return "late_instability"
    return "generalization_gap"


def _heldout_pbe_ratio(cell: Dict[str, Any]) -> Optional[float]:
    """A cell's held-out MAE relative to ITS OWN per-cell PBE baseline. The
    held-out reaction set differs per spec (each trains on a different subset), so
    the per-cell PBE -- NOT a cohort mean -- is the honest denominator and the one
    :func:`_classify_cell` uses. ``> 1`` <=> worse than PBE (a failure);
    ``<= 1`` <=> beats PBE (pass), so the ratio's side of 1.0 matches the cell's
    pass/fail colour exactly. None if either MAE is missing or PBE <= 0."""
    hm, pm = cell.get("heldout_mae"), cell.get("pbe_mae")
    return hm / pm if _is_num(hm) and _is_num(pm) and pm > 0 else None


def classify_failures(runs: List[Tuple[Path, str]], *, instab_factor: float = 5.0,
                      final_window: int = 50,
                      eval_subdir: str = "eval_holdout") -> List[Dict[str, Any]]:
    """Per (arch, subset_size, basis) held-out diagnosis across several runs:
    combined held-out MAE + BH76/W4-11 split, that cell's per-reaction PBE
    baseline, the final-window training loss, and a :func:`_classify_cell`
    mechanism label. Reuses :func:`collect_holdout_reaction_rows`,
    :func:`reaction_mae_by_arch_subset`, and :func:`collect_training_losses`."""
    cells: List[Dict[str, Any]] = []
    for run_dir, basis in runs:
        rows = collect_holdout_reaction_rows(run_dir, eval_subdir=eval_subdir)
        bh = [r for r in rows if r.get("pool") == "bh76"]
        w4 = [r for r in rows if r.get("pool") == "w411"]
        nn = reaction_mae_by_arch_subset(rows)
        pbe = reaction_mae_by_arch_subset(rows, key="abs_error_pbe_kcalmol")
        bh_nn = reaction_mae_by_arch_subset(bh)
        bh_pbe = reaction_mae_by_arch_subset(bh, key="abs_error_pbe_kcalmol")
        w4_nn = reaction_mae_by_arch_subset(w4)
        w4_pbe = reaction_mae_by_arch_subset(w4, key="abs_error_pbe_kcalmol")
        losses = {(r["arch"], r["subset_size"]): r["losses"]
                  for r in collect_training_losses(run_dir)}
        for (arch, ss), mae in nn.items():
            L = losses.get((arch, ss))
            cells.append({
                "run_dir": str(run_dir), "basis": basis, "arch": arch,
                "subset_size": ss, "heldout_mae": mae,
                "pbe_mae": pbe.get((arch, ss)),
                "bh76_mae": bh_nn.get((arch, ss)), "bh76_pbe": bh_pbe.get((arch, ss)),
                "w411_mae": w4_nn.get((arch, ss)), "w411_pbe": w4_pbe.get((arch, ss)),
                "final_loss": _final_window_loss(L, final_window)
                if L is not None else float("nan")})
    fins = [c["final_loss"] for c in cells if _is_num(c["final_loss"])]
    med = float(np.median(fins)) if fins else float("nan")
    for c in cells:
        c["cohort_median_loss"] = med
        c["classification"] = _classify_cell(
            c["heldout_mae"], c["pbe_mae"], c["final_loss"], med, instab_factor)
    return cells


_FAIL_COLORS = {"pass": "#2a9d3a", "generalization_gap": "#e08214",
                "late_instability": "#c0392b"}
_FAIL_LABEL = {"pass": "pass (<= PBE)",
               "generalization_gap": "generalization gap (overfit)",
               "late_instability": "late training instability"}


def _primary_basis(cells: List[Dict[str, Any]]) -> Optional[str]:
    """The basis with the most evaluated cells -- the dense run (def2-svp) used
    for the ss-resolved bars / heatmaps; the sparse run is shown in the lines."""
    counts: Dict[Any, int] = {}
    for c in cells:
        counts[c["basis"]] = counts.get(c["basis"], 0) + 1
    return max(counts, key=counts.get) if counts else None


def _ladder_bases(cells: List[Dict[str, Any]]) -> List[str]:
    """The bases rendered (one stacked capacity-ladder sub-panel each) in the
    failure diagnostic's right column, in run order (def2-svp first, the sparse
    DF run second). De-duplicated, order preserved."""
    return list(dict.fromkeys(c["basis"] for c in cells))


def _failure_caption(cells: List[Dict[str, Any]],
                     bases: List[str]) -> str:
    """Two-line failure-mechanism key for the figure footer: the late-instability
    cells (eval uses the bad final checkpoint) and the cells that beat PBE. The
    generalization-gap cells are read off Panel B's capacity ladder, so they are
    NOT enumerated here (the list was unreadably long)."""
    multi = len(bases) > 1

    def _grp(label: str) -> str:
        items = sorted(f"{c['arch']} ss{c['subset_size']}"
                       + (f" ({c['basis']})" if multi else "")
                       for c in cells if c["classification"] == label)
        return ", ".join(items) if items else "(none)"

    return (f"Late training instability (final loss is an outlier; eval uses the "
            f"bad final checkpoint):  {_grp('late_instability')}.\n"
            f"Beats PBE (pass):  {_grp('pass')}.")


def plot_failure_diagnostic(runs: List[Tuple[Path, str]], out_path: Path,
                            run_id: str, note: str = "",
                            provenance: Optional[str] = None,
                            eval_subdir: str = "eval_holdout") -> Path:
    """Explain WHY each network fails. Panel A: the DECOUPLING -- final-window
    training loss (x, log) vs held-out combined MAE (y); nearly all cells reach a
    low train loss yet scatter widely in held-out error (they overfit the tiny
    held-in subset), and only deep_attn-ss6 also has a high train loss (the lone
    genuine training failure). Panel B: the capacity ladder -- held-out MAE/PBE by
    arch family, split BH76 (barriers) vs W4-11 (atomization), showing extra
    descriptors + attention worsen overfitting and the damage lands on W4-11."""
    cells = classify_failures(runs, eval_subdir=eval_subdir)
    bases = _ladder_bases(cells)
    mk = ["o", "^", "s", "D"]
    marker_for = {b: mk[i % len(mk)] for i, b in enumerate(bases)}
    with plt.rc_context(_STYLE):
        fig = plt.figure(figsize=(14.0, 8.0))
        gs = fig.add_gridspec(1, 2, left=0.06, right=0.985, top=0.92, bottom=0.26,
                              wspace=0.22)
        axA = fig.add_subplot(gs[0, 0])  # axB sub-panels built by _broken_bar_panel

        # --- Panel A: decoupling scatter. y = each cell's held-out MAE RELATIVE
        # TO ITS OWN PBE (per-cell, since the held-out set differs per spec), so
        # PBE parity is a single exact line at 1.0 and the pass/fail COLOUR matches
        # the point's side of the line (green below, orange/red above). A cohort
        # mean-PBE line was misleading: a cell can sit below the mean yet still
        # lose to its own (lower) PBE. ---
        for c in cells:
            r = _heldout_pbe_ratio(c)
            if not (_is_num(c["final_loss"]) and _is_num(r)):
                continue
            axA.scatter(c["final_loss"], r,
                        color=_FAIL_COLORS[c["classification"]],
                        marker=marker_for.get(c["basis"], "o"), s=40,
                        edgecolor="k", linewidth=0.4, zorder=3)
        axA.axhline(1.0, ls="--", color="0.4", lw=1.0)  # PBE parity (held-out = PBE)
        med = cells[0].get("cohort_median_loss") if cells else None
        if _is_num(med):
            axA.axvline(5.0 * med, ls=":", color="#c0392b", lw=1.0)
        # label the worst few cells (largest held-out/PBE ratio)
        for c in sorted((c for c in cells if _is_num(_heldout_pbe_ratio(c))),
                        key=lambda c: -_heldout_pbe_ratio(c))[:4]:
            if _is_num(c["final_loss"]):
                axA.annotate(f"{c['arch']} ss{c['subset_size']}",
                             (c["final_loss"], _heldout_pbe_ratio(c)), fontsize=6,
                             xytext=(4, 2), textcoords="offset points")
        axA.set_xscale("log")
        axA.set_xlabel("final-window training loss (mean of last 50 steps, log)",
                       fontsize=8)
        axA.set_ylabel("held-out MAE / PBE (per cell;  >1 = worse than PBE)",
                       fontsize=8)
        axA.set_title("Training loss is decoupled from held-out accuracy vs PBE",
                      fontsize=9)
        axA.grid(True, which="both", alpha=0.3)
        cls_handles = [Patch(facecolor=_FAIL_COLORS[k], edgecolor="k",
                             label=_FAIL_LABEL[k])
                       for k in ("pass", "generalization_gap", "late_instability")]
        cls_handles += [plt.Line2D([], [], ls="--", color="0.4",
                                   label="PBE parity (held-out = PBE)"),
                        plt.Line2D([], [], ls=":", color="#c0392b",
                                   label="instability cut (5x median loss)")]
        if len(bases) > 1:
            cls_handles += [plt.Line2D([], [], ls="", marker=marker_for[b],
                                       color="0.3", label=str(b)) for b in bases]
        axA.legend(handles=cls_handles, fontsize=6.3, loc="upper left",
                   framealpha=0.7)

        # --- Panel B: ss-RESOLVED capacity-ladder bars, ONE STACKED SUB-PANEL PER
        # BASIS (def2-svp on top, def2-tzvpd+DF below) so the density-fitting run
        # is shown alongside the dense svp run. Bars are NEVER averaged over
        # subset_size -- at fixed (small) ss the capacity ladder is clean
        # (deep < attn < cusp < combined < combined_attn) and each arch falls
        # toward PBE as ss grows (overfitting relieved by data).
        present = {c["arch"] for c in cells}
        archs = [a for a in ARCH_ORDER if a in present]
        archs += sorted(present - set(archs))
        # GLOBAL ss scale (over BOTH bases) so the viridis subset colours match
        # between the two sub-panels.
        ss_vals = sorted({c["subset_size"] for c in cells
                          if _is_num(c.get("subset_size"))})
        nss = max(1, len(ss_vals))
        bw = 0.82 / nss
        norm_ss = matplotlib.colors.Normalize(min(ss_vals), max(ss_vals)) \
            if ss_vals else None
        cmap_ss = plt.get_cmap("viridis")
        ss_colors = [cmap_ss(norm_ss(ss)) if norm_ss else "0.5" for ss in ss_vals]
        gsB = gs[0, 1].subgridspec(max(1, len(bases)), 1, hspace=0.6)
        for bi, basis in enumerate(bases):
            pcells = [c for c in cells if c["basis"] == basis]
            rmap = {(c["arch"], c["subset_size"]): c["heldout_mae"] / c["pbe_mae"]
                    for c in pcells if _is_num(c["heldout_mae"])
                    and _is_num(c["pbe_mae"]) and c["pbe_mae"] > 0}
            series = [(f"ss{ss}", [rmap.get((a, ss), float("nan")) for a in archs])
                      for ss in ss_vals]
            # Broken y-axis (reused, tested helper): a lone spike (e.g. svp
            # deep_attn-ss6 ~5.4) shows at its TRUE height in an upper band instead
            # of crushing the bulk -- the break is decided independently per basis.
            axB = _broken_bar_panel(
                fig, gsB[bi, 0], series, archs, [],
                f"Capacity ladder per subset_size ({basis});  >1 = worse than PBE",
                "held-out MAE / PBE", ss_colors, bw)
            axB.axhline(1.0, ls="--", color="0.3", lw=1.0)  # PBE parity
            if bi == 0:
                ss_handles = [Patch(facecolor=ss_colors[k], edgecolor="k",
                                    label=f"ss{ss}")
                              for k, ss in enumerate(ss_vals)]
                axB.legend(handles=ss_handles, fontsize=6.0, ncol=max(1, nss // 2),
                           title="subset", title_fontsize=6.0, loc="upper left",
                           framealpha=0.7)

        # --- classification key: late-instability + beats-PBE cells (the
        # generalization-gap cells are read off the capacity ladder above) ---
        fig.text(0.06, 0.185, _failure_caption(cells, bases), ha="left", va="top",
                 fontsize=6.6, family="serif", wrap=True)
        _stamp_parity_footer(
            fig, run_id=run_id, note=note, provenance=provenance, caveat=None,
            title="Failure-mechanism diagnostic (held-out vs training loss)")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return out_path


def plot_capacity_trends(runs: List[Tuple[Path, str]], out_path: Path,
                         run_id: str, note: str = "",
                         provenance: Optional[str] = None,
                         eval_subdir: str = "eval_holdout") -> Path:
    """Secondary descriptive views of the same MAE/PBE structure: two diverging
    ratio heatmaps (BH76 barriers + W4-11 atomization, arch x ss, centered at PBE
    parity 1.0) showing the damage lands on W4-11; and a MAE/PBE-vs-subset_size
    line plot (one line per arch, basis = linestyle) making the capacity ordering
    and the fall-to-PBE-with-more-data trend explicit."""
    cells = classify_failures(runs, eval_subdir=eval_subdir)
    present = {c["arch"] for c in cells}
    archs = [a for a in ARCH_ORDER if a in present]
    archs += sorted(present - set(archs))
    bases = list(dict.fromkeys(c["basis"] for c in cells))
    prim = _primary_basis(cells)
    ss_axis = sorted({c["subset_size"] for c in cells if c["basis"] == prim})

    def _ratio_map(num: str, den: str, basis: Any) -> Dict[Tuple[str, int], float]:
        return {(c["arch"], c["subset_size"]): c[num] / c[den] for c in cells
                if c["basis"] == basis and _is_num(c.get(num))
                and _is_num(c.get(den)) and c[den] > 0}

    with plt.rc_context(_STYLE):
        fig = plt.figure(figsize=(15.5, 5.2))
        gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.15], left=0.055,
                              right=0.975, top=0.84, bottom=0.2, wspace=0.42)
        axH1, axH2, axL = (fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]),
                           fig.add_subplot(gs[0, 2]))
        _heatmap_panel(axH1, _ratio_map("bh76_mae", "bh76_pbe", prim), archs,
                       title=f"BH76 barriers  MAE/PBE ({prim})",
                       cbar_label="MAE / PBE", center=1.0, subset_sizes=ss_axis,
                       vxc_pre_fix=_run_predates_vxc_fix(run_id))
        _heatmap_panel(axH2, _ratio_map("w411_mae", "w411_pbe", prim), archs,
                       title=f"W4-11 atomization  MAE/PBE ({prim})",
                       cbar_label="MAE / PBE", center=1.0, subset_sizes=ss_axis,
                       vxc_pre_fix=_run_predates_vxc_fix(run_id))
        # line plot: combined MAE/PBE vs subset_size, arch = color, basis = ls
        arch_color = {a: plt.get_cmap("tab10")(i % 10) for i, a in enumerate(archs)}
        ls_for = {b: ["-", "--", "-.", ":"][i % 4] for i, b in enumerate(bases)}
        for a in archs:
            for b in bases:
                pts = sorted((c["subset_size"], c["heldout_mae"] / c["pbe_mae"])
                             for c in cells if c["arch"] == a and c["basis"] == b
                             and _is_num(c["heldout_mae"]) and _is_num(c["pbe_mae"])
                             and c["pbe_mae"] > 0)
                if not pts:
                    continue
                xs, ys = zip(*pts)
                axL.plot(xs, ys, marker="o", ms=3, color=arch_color[a],
                         ls=ls_for[b], lw=1.3,
                         label=a if b == bases[0] else None)
        for c in cells:
            if (c["classification"] == "late_instability" and _is_num(c["heldout_mae"])
                    and _is_num(c["pbe_mae"]) and c["pbe_mae"] > 0):
                axL.annotate(f"{c['arch']} ss{c['subset_size']}",
                             (c["subset_size"], c["heldout_mae"] / c["pbe_mae"]),
                             fontsize=6, color="#c0392b", ha="right", va="bottom",
                             xytext=(-3, 1), textcoords="offset points")
        axL.axhline(1.0, ls="--", color="0.3", lw=1.0)
        axL.set_xlabel("training subset_size", fontsize=8)
        axL.set_ylabel("held-out combined MAE / PBE", fontsize=8)
        axL.set_title("Overfitting relieved by more held-in molecules\n"
                      "(basis = linestyle)", fontsize=8.0)
        axL.grid(True, alpha=0.3)
        axL.legend(fontsize=5.8, ncol=2, framealpha=0.7)
        _stamp_parity_footer(
            fig, run_id=run_id, note=note, provenance=provenance, caveat=None,
            title="Capacity / data-relief trends (held-out MAE / PBE, per cell)")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return out_path


def _w411_natoms_map() -> Dict[str, int]:
    """``{W4-11 reaction name: molecule atom count}`` from the canonical pool --
    the number of atom products in each ``molecule -> atoms`` atomization. Used
    to expose the size-consistency failure (error vs molecule size)."""
    from xcquinox.alec.full_benchmark_pools import load_full_held_out_pools
    _, full = load_full_held_out_pools()
    out: Dict[str, int] = {}
    for r in full:
        if r.get("source_pool") != "w411":
            continue
        names = list(r.get("reactants", [])) + list(r.get("products", []))
        coeffs = list(r.get("coeffs", []))
        n = sum(int(round(abs(c))) for nm, c in zip(names, coeffs)
                if str(nm).casefold() in _ELEMENT_SYMBOLS)
        if n:
            out[r.get("name")] = n
    return out


def plot_size_consistency_diagnostic(rows: List[Dict[str, Any]], out_path: Path,
                                     run_id: str,
                                     cells: List[Tuple[str, int]], *,
                                     note: str = "",
                                     provenance: Optional[str] = None,
                                     dataset: Optional[str] = None) -> Path:
    """Diagnostic for the size-consistency (additivity) failure across a few
    chosen (arch, subset_size) cells: (a) W4-11 atomization |error| vs molecule
    atom-count with a per-cell linear fit -- a steep slope is a non-additive
    error that grows with molecule size; (b) BH76 (barriers) vs W4-11
    (atomizations) MAE per cell -- the asymmetry that is the fingerprint of lost
    size-consistency (barriers cancel the per-atom error, atomizations expose it)."""
    with plt.rc_context(_STYLE):
        natoms = _w411_natoms_map()
        palette = plt.get_cmap("tab10")
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), squeeze=False)
        axA, axB = axes[0][0], axes[0][1]
        bw = 0.38
        for i, (arch, ss) in enumerate(cells):
            col = palette(i % 10)
            sub = [r for r in rows if r.get("arch") == arch
                   and r.get("subset_size") == ss]
            pts = [(natoms[r["name"]], r["abs_error_nn_kcalmol"]) for r in sub
                   if r.get("pool") == "w411" and r.get("name") in natoms
                   and _is_num(r.get("abs_error_nn_kcalmol"))]
            if pts:
                xx, yy = zip(*pts)
                axA.scatter(xx, yy, s=13, alpha=0.45, color=col, edgecolor="none")
                if len(set(xx)) > 1:
                    a, b = np.polyfit(np.array(xx, float), np.array(yy, float), 1)
                    xr = np.array([min(xx), max(xx)], float)
                    axA.plot(xr, a * xr + b, color=col, lw=1.8,
                             label=f"{arch}/ss{ss}  ({a:.1f}/atom)")
                else:
                    axA.plot([], [], color=col, lw=1.8, label=f"{arch}/ss{ss}")
            bh = _mae([r["abs_error_nn_kcalmol"] for r in sub
                       if r.get("pool") == "bh76"])
            w4 = _mae([r["abs_error_nn_kcalmol"] for r in sub
                       if r.get("pool") == "w411"])
            axB.bar(i - bw / 2, bh if bh is not None else np.nan, width=bw,
                    color="#4477aa", edgecolor="k", linewidth=0.4)
            axB.bar(i + bw / 2, w4 if w4 is not None else np.nan, width=bw,
                    color="#cc6677", edgecolor="k", linewidth=0.4)
        axA.set_xlabel("molecule atom count", fontsize=8)
        axA.set_ylabel("W4-11 atomization |error|  (kcal/mol)", fontsize=8)
        axA.set_title("(a) Size-consistency: atomization error vs molecule size",
                      fontsize=9)
        if axA.get_legend_handles_labels()[1]:
            axA.legend(fontsize=7, title="fit slope = kcal/mol per atom",
                       title_fontsize=6)
        axA.grid(True, alpha=0.3)
        axB.bar([], [], color="#4477aa", label="BH76 (barriers)")
        axB.bar([], [], color="#cc6677", label="W4-11 (atomizations)")
        axB.set_xticks(range(len(cells)))
        axB.set_xticklabels([f"{a}/ss{s}" for a, s in cells], rotation=25,
                            ha="right", fontsize=7)
        axB.set_ylabel("MAE (kcal/mol)", fontsize=8)
        axB.set_title("(b) Barriers vs atomizations -- the cancellation fingerprint",
                      fontsize=9)
        axB.legend(fontsize=7)
        axB.grid(True, axis="y", alpha=0.3)
        _stamp_parity_footer(
            fig, run_id=run_id, note=note, provenance=provenance, caveat=None,
            dataset=dataset,
            title="Why deep_attn ss=6 fails: a size-consistency (additivity) breakdown")
        fig.tight_layout(rect=(0, 0.04, 1, 0.93))
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return out_path


def _break_limits(vals: List[Any]):
    """If one value dwarfs the bulk, return brokenaxes ylims
    ``((0, bulk_hi), (upper_lo, upper_hi))``; else None (use a normal axis)."""
    v = sorted(float(x) for x in vals if _is_num(x) and x >= 0)
    if len(v) < 4:
        return None
    rng = v[-1] - v[0]
    if rng <= 1e-9:
        return None
    # Break at the LARGEST gap between consecutive sorted values -- i.e. the empty
    # band separating the bulk from a lone outlier (e.g. ~45 -> ~77). The lower
    # band keeps EVERY non-outlier bar; only the empty gap is collapsed.
    gap, idx = max((v[i + 1] - v[i], i) for i in range(len(v) - 1))
    if gap < 0.35 * rng:           # no clear separation -> normal axis
        return None
    low_hi = v[idx] + 0.12 * gap    # just above the bulk maximum
    up_lo = v[idx + 1] - 0.12 * gap  # just below the lowest outlier
    return ((0.0, low_hi), (up_lo, v[-1] * 1.04))


def _broken_bar_panel(fig, subplot_spec, series, labels, pbe_lines, title, ylab,
                      colors, bw):
    """Grouped-bar panel placed in ``subplot_spec`` (a GridSpec cell): uses a
    BROKEN y-axis (brokenaxes) when one bar dwarfs the rest, else a normal axis.
    ``series`` = [(label, heights)]; ``pbe_lines`` = [(label, y)]."""
    all_vals = [h for _, hs in series for h in hs]
    lims = _break_limits(all_vals)
    n = len(labels)
    nb = max(1, len(series))
    if lims is not None:
        from brokenaxes import brokenaxes  # optional dep (xcq env)
        ax = brokenaxes(ylims=lims, subplot_spec=subplot_spec, hspace=0.08,
                        d=0.008, despine=False)
        bottom = min(ax.axs, key=lambda a: a.get_ylim()[0])
    else:
        ax = fig.add_subplot(subplot_spec)
        bottom = ax
    for j, (label, hs) in enumerate(series):
        xs = [i + (j - (nb - 1) / 2) * bw for i in range(n)]
        ax.bar(xs, hs, width=bw, color=colors[j % len(colors)], edgecolor="k",
               linewidth=0.3, label=label)
    for j, (label, y) in enumerate(pbe_lines):
        if _is_num(y):
            ax.axhline(y, ls="--", lw=1.0, color=colors[j % len(colors)], alpha=0.8)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        bottom.set_xticks(range(n))
        bottom.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
    ax.set_ylabel(ylab, fontsize=8)
    ax.set_title(title, fontsize=9)
    if lims is None:
        ax.grid(True, axis="y", alpha=0.3)
    return ax


# 2-letter element symbols that actually occur in the bh76w411 pool (H,C,N,O,F,P,
# S,Cl,Al,Si,B,Be,Li,Na,Mg). Restricted to these so greedy matching never mistakes
# H-O-C-N-S substrings for Ho/Os/Co/Cs/Ni etc.
_CHEM_2L = frozenset({"cl", "al", "si", "be", "li", "na", "mg"})


def _chem_latex(name: str) -> str:
    """Render a training-species token as a chemical formula in matplotlib
    mathtext: element symbols capitalized, digit counts subscripted. Recognizes a
    transition-state suffix ``ts`` (double-dagger) and a complex suffix ``comp``
    ((c) subscript); passes reaction-label species (``RKT...``) through verbatim."""
    if name.lower().startswith("rkt"):
        return name
    core, suffix = name, ""
    if name.lower().endswith("ts"):
        core, suffix = name[:-2], r"$^{\ddagger}$"
    elif name.lower().endswith("comp"):
        core, suffix = name[:-4], r"$_{\mathrm{(c)}}$"
    out: List[str] = []
    i = 0
    while i < len(core):
        if core[i : i + 2].lower() in _CHEM_2L:
            out.append(core[i : i + 2].capitalize())
            i += 2
        elif core[i].isalpha():
            out.append(core[i].upper())
            i += 1
        elif core[i].isdigit():
            j = i
            while j < len(core) and core[j].isdigit():
                j += 1
            sub = core[i:j]
            out.append(r"$_%s$" % sub if len(sub) == 1 else r"$_{%s}$" % sub)
            i = j
        else:
            i += 1
    return "".join(out) + suffix


def _methods_columns(subsets: Dict[int, List[str]]) -> List[List[str]]:
    """The three methods columns (checked against networks.py / config.py / models.py) (placed under panels a/b/c).
    Strings checked against networks.py / config.py / features.py / losses.py /
    train.py: GGA inputs + log-transform + constraints (col 1); pretrain +
    optimization + attention (col 2); extra descriptors + training subsets
    (col 3). Subset molecules are rendered as chemical formulas."""
    col1 = [
        "DESCRIPTORS (inputs to the exchange/correlation MLPs):",
        r"  $x_2{=}s{=}|\nabla\rho|/(2(3\pi^2)^{1/3}\rho^{4/3})$  reduced gradient [1,4].",
        r"  $r_s{=}(3/4\pi\rho)^{1/3}$  Wigner-Seitz radius = the PW92 LDA-",
        r"     correlation variable [2].  X-net in $(x_2)$; C-net in $(r_s,x_2,x_1)$.",
        r"  $x_1{=}\frac{1}{2}[(1{+}\zeta)^{4/3}{+}(1{-}\zeta)^{4/3}]$  spin feature [4],",
        r"     $\zeta{=}(\rho_\alpha{-}\rho_\beta)/\rho$;  $(1{\pm}\zeta)^{4/3}$ = exchange spin-scaling",
        r"     factor [3], in the PW92 $f(\zeta)$ numerator [2].  $x_1{=}1$ at $\zeta{=}0$ (RKS).",
        "  Log-transform (this work): the MLP is fed",
        r"     $\tilde{x}_2{=}(1{-}e^{-x_2^2})\ln(x_2{+}1)$ ($\sim x_2^3{\to}0$, preserving UEG); $r_s$",
        r"     likewise; $x_1$ raw.  ([4] Eq.9 form; [4] also log-transform $x_1$, Eq.8.)",
        r"  Spin clip (this work): $\zeta$ clamped to $\pm(1{-}10^{-6})$.  PW92 $f(\zeta)$",
        r"     [2] has $f''{\sim}(1{\mp}\zeta)^{-2/3}{\to}\infty$ at full polarization ($\rho_\beta{\to}0$,",
        r"     free atoms); the SCF differentiates $v_c{=}\partial E_c/\partial\rho$ a 2nd time, so",
        r"     the gradient is non-finite at the unclamped boundary ($10^{-6}$ keeps $f''$ finite).",
        "",
        "CONSTRAINTS / BOUNDS:",
        r"  $E_{xc}{=}\int\rho\,(\epsilon_x^{UEG}F_x{+}\epsilon_c^{PW92}F_c)$ [2,4];  $F{=}1{+}\mathrm{LOB}_L(\tanh^2\!x_2\cdot\mathrm{MLP})$.",
        r"  $\mathrm{LOB}_L(x){=}L\sigma(x{-}\ln(L{-}1)){-}1$ maps $\mathbb{R}{\to}({-}1,L{-}1)$, so",
        r"     $F{\in}(0,L)$, $F(0){=}1$  ($=$ DFS $I_L$ [4] Eq.11).",
        r"  $\tanh^2\!x_2$ UEG gate: $F{\to}1$ at $x_2{=}0$ (exact GGA limit [1]; this-",
        r"     work gate vs [4]'s $\tilde{x}_2{+}\tanh^2\tilde{x}_3$ meta-GGA form).",
        r"  $F_x$: $L{=}1.804{=}1{+}\kappa$ ($\kappa{=}0.804$), the PBE exchange ceiling set",
        r"     by the local Lieb-Oxford bound [1,5]; [4] use a tighter 1.174 [6].",
        r"  $F_c$: $L{=}2$, a non-negativity squash ([4] $I_2$ Eq.13), NOT a LO",
        r"     bound on $F_c$.  Exchange spin-scaled $E_x{=}\frac{1}{2}[E_x(2\rho_\alpha){+}E_x(2\rho_\beta)]$ [3].",
    ]
    col2 = [
        "LOSS  (channel forms = this work, losses.py; the density-",
        " dominant weights + per-molecule scheme follow dpyscf/DFS [4,15]):",
        r"  $L(\omega){=}\sum_k w_k L_k$,  $w{=}\{$AE 1, BH76 1, IP13 1, $v_{xc}$ 1, $\rho$ 20$\}$.",
        "  Mixed metric (loss_metric = absolute):",
        r"   reaction energy (absolute), the L5 'BH76' channel: $\langle(\sum_s\nu_s E^{NN}_s{-}e^{ref}_{rxn})^2\rangle$,",
        r"     $E^{NN}_s$=SCF energy.  Trains BOTH W4-11 atomizations (molecule$\to$atoms,",
        r"     $e^{ref}$=W4-11 [17]) and BH76 barriers (reactants$\to$TS, $e^{ref}$=W2-F12 [16]).",
        r"   L5's relative-AE $\langle(A^{NN}{-}A^{ref})^2/((A^{ref})^2{+}10^{-8})\rangle$, $A^{NN}{=}\sum_Z n_Z E_Z{-}E^{NN}$",
        r"     ($E_Z$ atom totals [18]), and the IP13 channel, are not populated by this pool.",
        r"   $v_{xc}$ (per-elem MSE): $\langle\|V^{NN}_{xc}{-}V^{ref}_{xc}\|_F^2/n_{ao}^2\rangle$ (AO matrix).",
        r"   $\rho$ (grid-$L_2$): $\langle\sum_g w_g(\rho^{NN}_g{-}\rho^{ref}_g)^2\rangle$ ($w_g$ quadrature wt).",
        r"  SCF: $E^{NN}$ and $\rho^{NN}$ are the FINAL state of a fixed 3-cycle",
        r"   differentiable Kohn-Sham SCF (rebuild $J{+}V_{xc}$ from the NN density",
        r"   each cycle, backprop through all 3) [14, our implementation];",
        r"   $V^{NN}_{xc}$ is one-shot.  Not iterated-to-tolerance.",
        "  Per-molecule update: one optimizer step per molecule-group,",
        r"   all groups/epoch, 250 epochs; LR $0.01$ held 0.2 then linear $\to10^{-5}$.",
        r"   GradNorm ($\alpha{=}1.5$) [13] is CONFIGURED BUT DORMANT (per-molecule",
        "   bypasses it; the weights stay fixed).",
    ]
    col3 = [
        "PRETRAIN (this work, [4]-style; 2500 steps, per-grid-point, spin-resolved):",
        r"  GGA/rung-3.5: $F_x{=}F_x^{PBE}/F_x^{LDA}{-}1$, $F_c{=}F_c^{PBE}/F_c^{LDA}{-}1$;",
        r"  _mgga archs: the SAME ratios to SCAN [20] (the meta-GGA they clone).",
        "ATTENTION (_attn / _combined_attn, heads=4): per-grid-point",
        r"  channel attn $\mathrm{softmax}(QK^T\!/\sqrt{d_k})V$ [19] over MLP-1 units, 4 tokens.",
        "",
        "EXTENDED DESCRIPTORS (defined in this work):",
        r"  _cusp $(x_4,x_5)$:",
        r"   $x_4{=}e^{-2Z_{near}r_{min}}$: Slater density envelope at the nearest",
        r"     nucleus.  Cusp: wavefn $(\partial\bar\psi/\partial r)_0{=}{-}Z\psi(0)$ [7]; density",
        r"     $(\partial\bar\rho/\partial r)_0{=}{-}2Z\rho(0)$, $\rho{\sim}e^{-2Zr}$ [8].",
        r"   $x_5{=}\tanh(\ln(\sum_A Z_A/r_A)/5)$: $\sum_A Z_A/r_A$ = magnitude of the",
        r"     bare-nuclei electrostatic potential $={-}V_{ext}$ ($V_{ext}{=}{-}\sum_A Z_A/|r{-}R_A|$",
        r"     [12]); the $\ln,/5,\tanh$ map it to $(-1,1)$ (this work; log convention [4]).",
        r"  _dm $(x_6,x_7)$ from the 1-particle density matrix $D$ ($D'{=}D/2$",
        r"   RKS, $D_\sigma$ UKS; a 3rd feature, the occupation-spread entropy, was",
        r"   removed 2026-08-06: its gradient vanishes identically at any converged",
        r"   density, so it carried no trainable signal):",
        r"   $x_6{=}\|D'SD'{-}D'\|_F^2/\mathrm{Tr}(D'S)$: idempotency, $=0$ EXACTLY for one",
        r"     Slater determinant ($PSP{=}P$ [10]; squared norm, smooth at 0), $>0$ under",
        r"     the fractional natural occupation of multireference states [11]. On the",
        r"     single-determinant KS densities evaluated here it is zero in value and",
        r"     gradient; the block's live channel is $x_7$.",
        r"   $x_7{=}\|D_{off}\|_F/\mathrm{Tr}(D)$: relative off-diagonal weight of $D$.",
        r"  _rung35 $(x_8,x_9)$: per-spin localized-DM occupancy",
        r"   $n_\sigma(r){=}A(r)^T\!D^\sigma A(r)\in[0,1]$, $A_\mu{=}\langle\chi_\mu|\phi^G_r\rangle$",
        r"     a Gaussian projector (Rung-3.5 [21]; leak-free, replaces global _dm).",
        r"  _mgga $(x_{10})$: iso-orbital $\alpha{=}(\tau{-}\tau_W)/\tau_{unif}$ [20]",
        r"     (meta-GGA; $F_x$ ceiling 1.174 not 1.804; UEG gate on $(s,\alpha)$).",
        r"  _rung35ms $(x_{11}..x_{16})$: the $x_8,x_9$ occupancy at projector widths",
        r"     $\alpha_w{\in}\{0.05,0.2,0.8\}$, alpha-major then spin (rung35ms archs).",
        r"  _combined: cusp & DM;   _notransform: log-transform off.",
    ]
    return [col1, col2, col3]


def _methods_references() -> List[str]:
    """Full-width numbered references key for the methods box (each equation in
    the columns cites [n]). Every reference checked against the primary source."""
    return [
        "References   [1] Perdew, Burke, Ernzerhof, PRL 77, 3865 (1996).   "
        "[2] Perdew & Wang, PRB 45, 13244 (1992).   [3] Oliver & Perdew, PRA 20, 397 (1979).   "
        "[4] Dick & Fernandez-Serra (\"DFS\"), PRB 104, L161109 (2021).   [5] Lieb & Oxford, IJQC 19, 427 (1981).   "
        "[6] Perdew, Ruzsinszky, Sun, Burke, JCP 140, 18A533 (2014).",
        "[7] Kato, Commun. Pure Appl. Math. 10, 151 (1957).   [8] Steiner, JCP 39, 2365 (1963).   "
        "[9] Loewdin, Phys. Rev. 97, 1474 (1955).   [10] Szabo & Ostlund (1996) / Pople & Nesbet, JCP 22, 571 (1954).   "
        "[11] Boguslawski et al., JPCL 3, 3129 (2012); Xu et al., JCTC 20, 721 (2024).   "
        "[12] Parr & Yang, DFT of Atoms and Molecules (1989).",
        "[13] Chen et al. (GradNorm), ICML 2018 / arXiv:1711.02257.   [14] Li et al., PRL 126, 036401 (2021).   "
        "[15] dpyscf / [4] (density-dominant weights + per-molecule scheme).   "
        "[16] Goerigk et al. (GMTKN55-BH76), PCCP 19, 32184 (2017).   [17] Karton, Daon, Martin (W4-11), CPL 510, 165 (2011).   "
        "[18] Chakravorty et al., PRA 47, 3649 (1993).   [19] Vaswani et al., NeurIPS 2017 (scaled-dot-product attention).   "
        "[20] Sun, Ruzsinszky, Perdew (SCAN), PRL 115, 036402 (2015).   "
        "[21] Janesko (Rung-3.5), JCP 133, 104103 (2010) / Verma et al. (M11plus), JCTC 15, 4804 (2019).",
    ]


_DESCRIPTOR_X_LABELS: Dict[str, Tuple[str, ...]] = {
    # x-labels each descriptor group contributes, in feature order (col3 defines
    # x_4,x_5 = cusp and x_6,x_7 = the 1-RDM statistics; x_8,x_9 = the rung-3.5
    # per-spin localized-DM occupancies n_alpha, n_beta; x_10 = the meta-GGA
    # iso-orbital alpha = (tau - tau_W)/tau_unif; x_11..x_16 = the multi-width
    # rung-3.5 occupancies, alpha-major then spin over 3 projector widths).
    # dm_statistics went 3 -> 2 on 2026-08-06: dm_entropy was removed (no
    # usable gradient at any converged density), so the labels here MUST track
    # each descriptor's n_features -- pinned by
    # test_descriptor_x_labels_match_registry_widths.
    "cusp": ("x_4", "x_5"),
    "dm_statistics": ("x_6", "x_7"),
    "rung35": ("x_8", "x_9"),
    "metagga": ("x_10",),
    "rung35_multishell": ("x_11", "x_12", "x_13", "x_14", "x_15", "x_16"),
}


def _arch_input_forms(arch_names: Tuple[str, ...] = ARCH_ORDER,
                      polarized: bool = True) -> Dict[str, Dict[str, Any]]:
    """Per-arch X-net (F_x) and C-net (F_c) MLP input signatures, derived from
    each ArchitectureConfig's descriptor list. Descriptor extras are concatenated
    in the arch's descriptor order -- exactly the order networks.py packs them
    (descriptors.py: ``concatenate([d.compute() for d in descriptors])``) -- so
    e.g. deep_combined (descriptors ``[dm_statistics, cusp]``) packs the DM block
    x_6,x_7 BEFORE the cusp block x_4,x_5.  Both nets receive the same extras;
    ``polarized`` reflects the run-wide ``use_polarized_correlation`` override
    (True for the bh76w411 runs), which adds x_1 to the C-net.  Source of truth:
    ``xcquinox.alec.config.ARCHITECTURES``."""
    from xcquinox.alec.config import ARCHITECTURES
    out: Dict[str, Dict[str, Any]] = {}
    for name in arch_names:
        cfg = ARCHITECTURES[name]
        extras: List[str] = []
        for spec in cfg.descriptors:
            extras.extend(_DESCRIPTOR_X_LABELS[spec.name])
        out[name] = {
            "fx": ["x_2", *extras],
            "fc": ["r_s", "x_2", *(["x_1"] if polarized else []), *extras],
            "attention": cfg.attention,
            "log_transform": cfg.descriptor_log_transform,
        }
    return out


def _arch_forms_lines(arch_names: Tuple[str, ...] = ARCH_ORDER,
                      polarized: bool = True) -> List[str]:
    """Full-width methods lines giving the explicit $F_x$/$F_c$ MLP-input form of
    each figure architecture.  Archs sharing an identical form are grouped (the
    attention variants share their base's inputs; notransform shares deep's
    inputs but unleashed from the log-transform)."""
    forms = _arch_input_forms(arch_names, polarized=polarized)
    groups: List[Tuple[Tuple[Any, ...], List[str]]] = []
    for name in arch_names:
        f = forms[name]
        key = (tuple(f["fx"]), tuple(f["fc"]), f["log_transform"])
        for k, names in groups:
            if k == key:
                names.append(name)
                break
        else:
            groups.append((key, [name]))
    lines = [r"Per-arch MLP inputs (X-net $F_x$, C-net $F_c$; descriptor extras "
             r"concatenated in arch order):"]
    for (fx, fc, logt), names in groups:
        raw = "" if logt else r"   [$s,r_s$ raw -- no log-transform]"
        lines.append(rf"  {', '.join(names)}:  "
                     rf"$F_x({', '.join(fx)})$,  $F_c({', '.join(fc)})${raw}")
    lines.append(r"  $\_$attn variants add per-grid channel attention [19]; "
                 r"MLP inputs unchanged.")
    return lines


def _render_reaction(reactants: List[str], products: List[str]) -> str:
    """``reactants -> products`` in mathtext, species via :func:`_chem_latex`."""
    lhs = r" $+$ ".join(_chem_latex(r) for r in reactants)
    rhs = r" $+$ ".join(_chem_latex(p) for p in products)
    return f"{lhs} " + r"$\to$" + f" {rhs}"


def _subset_reaction_lines(reactions: Dict[int, Dict[str, List[Any]]]) -> List[str]:
    """Full-width footer lines making the per-subset training content explicit:
    W4-11 atomization molecules + BH76 barrier reactions (reactants->TS)."""
    lines = ["Training content per held-in subset  (W4-11 atomization energies: "
             "molecule -> atoms;  BH76 barriers: reactants -> transition state).  "
             r"Superscript $\ddagger$ = transition state;  subscript (c) = reactant complex:"]
    for ss in sorted(reactions):
        ae = ", ".join(_chem_latex(m) for m in reactions[ss].get("ae", []))
        rx = ";  ".join(_render_reaction(r, p)
                        for r, p in reactions[ss].get("rxn", []))
        parts = []
        if ae:
            parts.append("AE: " + ae)
        if rx:
            parts.append("barriers: " + rx)
        lines.append(f"  ss{ss} -- " + "    ".join(parts))
    return lines


def _methods_textblock(fig, subsets: Dict[int, List[str]], y_top: float = 0.28,
                       archs: Optional[List[str]] = None,
                       xs: Tuple[float, float, float] = (0.05, 0.385, 0.715),
                       y_deltas: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                       fontsize: float = 6.2,
                       reactions: Optional[Dict[int, Dict[str, List[Any]]]] = None,
                       fig_h: Optional[float] = None,
                       include_references: bool = True) -> int:
    """Place the three methods columns (mathtext) under panels a/b/c at ``xs``,
    each offset vertically by ``y_deltas`` (figure fraction; negative = lower).
    When ``reactions`` + ``fig_h`` are given, a FULL-WIDTH training-content footer
    (W4-11 atomizations + BH76 reactions) is placed below the columns. With
    ``include_references=False`` the full-width references key is omitted (the
    columns keep their ``[n]`` cites) and the training-content footer slides up
    into the freed slot. Returns the total effective line count (columns +
    references + footer) so a caller can size the figure."""
    cols = _methods_columns(subsets)
    for x, dy, col in zip(xs, y_deltas, cols):
        fig.text(x, y_top + dy, "\n".join(col), va="top", ha="left",
                 fontsize=fontsize, family="serif")
    max_col = max(len(c) for c in cols)
    arch_lines = _arch_forms_lines()
    refs = _methods_references() if include_references else []
    footer = _subset_reaction_lines(reactions) if reactions else []
    if fig_h:
        line_frac = fontsize * 1.58 / (72.0 * fig_h)
        # full-width per-arch F_x/F_c forms, clear of the tallest column ...
        y = y_top - (max_col + 2.0) * line_frac
        fig.text(xs[0], y, "\n".join(arch_lines), va="top", ha="left",
                 fontsize=fontsize, family="serif")
        y -= (len(arch_lines) + 2.0) * line_frac
        if refs:                                   # ... the references key (optional) ...
            fig.text(xs[0], y, "\n".join(refs), va="top", ha="left",
                     fontsize=fontsize - 0.5, family="serif")
            y -= (len(refs) + 2.0) * line_frac
        if footer:                                 # ... and the training-content footer (kept)
            fig.text(xs[0], y, "\n".join(footer), va="top", ha="left",
                     fontsize=fontsize, family="serif")
    return max_col + len(arch_lines) + len(refs) + (len(footer) + 8 if footer else 6)


def run_basis_label(run_dir: Path) -> str:
    """Short basis tag from ``resolved_config.yaml`` (e.g. ``def2-svp``,
    ``def2-tzvpd+DF``). Line-parsed -- no yaml dependency."""
    cfg = Path(run_dir) / "resolved_config.yaml"
    basis, df = "unknown", False
    if cfg.is_file():
        for line in cfg.read_text().splitlines():
            s = line.strip()
            if s.startswith("basis:"):
                basis = s.split(":", 1)[1].strip()
            elif s.startswith("density_fit:"):
                df = "true" in s.split(":", 1)[1].strip().lower()
    return f"{basis}+DF" if df else basis


def run_solver_label(run_dir: Path) -> str:
    """SCF-cycle tag (e.g. ``full_3`` / ``full_25``) from the ``solver:`` entry of
    ``resolved_config.yaml`` -- the variable that distinguishes two runs sharing a
    basis (3-cycle vs 25-cycle SCF). Reads the same file as :func:`run_basis_label`
    (no yaml dependency); returns ``""`` when no solver entry is present."""
    cfg = Path(run_dir) / "resolved_config.yaml"
    if not cfg.is_file():
        return ""
    in_solver = False
    for line in cfg.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("solver:"):
            inline = stripped.split(":", 1)[1].strip()      # "solver: full_3"
            if inline and not inline.startswith("#"):
                return inline.strip("[]'\" ")
            in_solver = True                                # block form follows
            continue
        if in_solver:
            if stripped.startswith("-"):                    # "- full_3"
                return stripped[1:].strip().strip("'\"")
            if stripped and not line.startswith((" ", "\t")):
                break                                       # left the solver block
    return ""


def _disambiguated_run_labels(run_dirs: List[Path]) -> List[str]:
    """Display labels for a set of runs, guaranteed pairwise-distinct. Uses the
    bare basis tag (:func:`run_basis_label`) when those are already unique; when
    two runs share a basis (e.g. def2-svp full_3 vs full_25) the SCF-cycle tag is
    appended, and as a last resort the basis-subdir alias -- so the comparison
    legend/title can always tell the series apart."""
    dirs = [Path(rd) for rd in run_dirs]
    base = [run_basis_label(rd) for rd in dirs]
    if len(set(base)) == len(base):
        return base
    tagged = []
    for rd, b in zip(dirs, base):
        tag = run_solver_label(rd)
        tagged.append(f"{b} · {tag}" if tag else b)
    if len(set(tagged)) == len(tagged):
        return tagged
    out = []
    for rd, b in zip(dirs, base):                           # last resort: subdir alias
        try:
            alias = _basis_fig_alias(rd.parents[1].name)
        except IndexError:
            alias = rd.name
        out.append(f"{b} ({alias})")
    return out


def _ckpt_label(eval_subdir: str) -> str:
    """Human tag for which checkpoint a figure set was scored from: final-step
    weights (``eval_holdout``), the held-out-validation-best weights
    (``eval_holdout_val_best``), or the legacy training-loss-best weights
    (``eval_holdout_best``, no longer plotted)."""
    return {
        "eval_holdout": "final-step",
        "eval_holdout_val_best": "val-best",
        "eval_holdout_best": "train-best",
        "eval_holdout_coldstart": "cold-start",
    }.get(eval_subdir, "final-step")


_BASIS_COLORS = ("#4477aa", "#cc6677", "#228833", "#ccbb44")


def _comparison_cells(cellsets: List[set],
                      archs: Optional[Sequence[str]] = None
                      ) -> List[Tuple[str, int]]:
    """Sorted union of (arch, subset_size) cells across the per-run cell sets.
    ``archs`` (when given) restricts the union to the named architectures; the
    full range of subset sizes within those archs is kept. The figure width
    scales with the cell count, so the restriction is what keeps a focused
    comparison readable when the full union spans many arch x subset columns."""
    cells = sorted(set.union(*cellsets)) if cellsets else []
    if archs is not None:
        wanted = set(archs)
        cells = [c for c in cells if c[0] in wanted]
    return cells


def plot_basis_comparison(runs: List[Tuple[Path, str]], out_path: Path,
                          run_id: str, note: str = "",
                          provenance: Optional[str] = None,
                          include_references: bool = True,
                          bars_only: bool = False,
                          eval_subdir: str = "eval_holdout",
                          archs: Optional[Sequence[str]] = None) -> Path:
    """Cross-basis comparison over the UNION of (arch, subset) cells present in
    ANY run: (a) combined held-out reaction-energy MAE, (b) 2-subset WTMAD-2, (c)
    in-sample density RMSE vs CCSD -- grouped bars by basis. A basis's bar is
    simply absent for a cell it hasn't run yet (leaving room as later runs, e.g.
    DF, fill in) -- completed cells are NEVER dropped for lack of a counterpart.
    Per-basis PBE baselines are dashed lines on the energy panels; the held-out
    benchmark reference is basis-independent, so NN errors ARE comparable.

    ``bars_only`` drops EVERY bottom annotation (the 3-column methods block, the
    per-arch forms, the references key, the subset-reaction footer and the
    provenance line), leaving just the three panels + legend + title -- a compact,
    easy-to-read variant for a slide/email. ``include_references`` is ignored when
    ``bars_only`` is set. ``archs`` restricts the plotted cells to the named
    architectures (see :func:`_comparison_cells`)."""
    with plt.rc_context(_STYLE):
        data = []
        cellsets = []
        for rd, label in runs:
            rows = collect_holdout_reaction_rows(rd, eval_subdir=eval_subdir)
            mae = reaction_mae_by_arch_subset(rows)
            wt = wtmad2_by_arch_subset(rows)
            pbe_mae = _mae([r["abs_error_pbe_kcalmol"]
                            for r in _dedup_rows_by_name(rows)])
            pbe_wt = wtmad2_pbe_baseline(rows)
            dmap: Dict[Tuple[str, int], List[float]] = {}
            for r in collect_insample_density_rows(rd):
                if _is_num(r.get("density_rmse")):
                    dmap.setdefault((r.get("arch"), r.get("subset_size")),
                                    []).append(r["density_rmse"])
            data.append((label, mae, wt, pbe_mae, pbe_wt, dmap))
            cellsets.append(set(mae.keys()))
        cells = _comparison_cells(cellsets, archs)
        if archs is not None:
            present = {a for cs in cellsets for (a, _s) in cs}
            missing = [a for a in dict.fromkeys(archs) if a not in present]
            if not cells:
                raise ValueError(
                    f"archs {list(archs)} match no (arch, subset) cell in any "
                    f"run (present archs: {sorted(present)}); a blank "
                    "comparison would render. Check the names (e.g. the "
                    "_3x16 suffix).")
            if missing:
                print(f"  (comparison archs with no cell in any run, "
                      f"skipped: {missing})")
        labels = [f"{a}/ss{s}" for a, s in cells]
        pw = max(6.0, 0.42 * max(1, len(cells)))
        nb = max(1, len(data))
        bw = 0.8 / nb
        # Size the figure to its content (inches): the methods band is placed
        # snug above the provenance so there is no trailing whitespace, and the
        # legend goes ABOVE the panels (clear of the rotated x-axis labels).
        subsets = training_subsets_by_size(runs[0][0]) if runs and not bars_only else {}
        reactions = training_reactions_by_size(runs[0][0]) if runs and not bars_only else {}
        FS = 6.2
        # height = tallest column + full-width per-arch forms + references key
        # + subset footer, each separated by a 2-line gap (+ top/bottom pads).
        if bars_only:
            meth_h = 0.0                            # no methods/footer/provenance
        else:
            n_cols = max(len(c) for c in _methods_columns(subsets))
            n_arch = len(_arch_forms_lines())
            n_refs = len(_methods_references()) if include_references else 0
            n_footlines = len(_subset_reaction_lines(reactions)) if reactions else 0
            n_meth = (n_cols + 2 + n_arch
                      + ((2 + n_refs) if include_references else 0)
                      + ((2 + n_footlines) if n_footlines else 0) + 3)
            meth_h = n_meth * FS * 1.58 / 72.0 + 0.06  # methods block (~1.2 linespace)
        panels_h, xlabel_h = 3.5, 0.72              # panels + rotated cell labels
        legend_h, gap1, gap2 = 0.30, 0.06, 0.10     # legend band: methods | legend | labels
        gap1 = 0.0 if bars_only else gap1           # no methods gap in bars-only
        top_pad = 0.68                              # suptitle + panel-title clearance
        bot_pad = 0.12 if bars_only else 0.24       # provenance line (none in bars-only)
        fig_h = (bot_pad + meth_h + gap1 + legend_h + gap2 + xlabel_h
                 + panels_h + top_pad)
        fig = plt.figure(figsize=(pw * 3, fig_h))

        def _f(inches: float) -> float:             # inches-from-bottom -> fraction
            return inches / fig_h

        top = fig.add_gridspec(
            1, 3, left=0.05, right=0.975, top=1.0 - _f(top_pad),
            bottom=_f(bot_pad + meth_h + gap1 + legend_h + gap2 + xlabel_h),
            wspace=0.26)

        def _panel(ax, getval, pbe_attr, title, ylab, logy=False):
            for j, (label, mae, wt, pbe_mae, pbe_wt, dmap) in enumerate(data):
                xs = [i + (j - (nb - 1) / 2) * bw for i in range(len(cells))]
                hs = [getval(mae, wt, dmap, c) for c in cells]
                col = _BASIS_COLORS[j % len(_BASIS_COLORS)]
                ax.bar(xs, hs, width=bw, color=col, edgecolor="k", linewidth=0.3,
                       label=label)
                if pbe_attr is not None:
                    base = pbe_mae if pbe_attr == "mae" else pbe_wt
                    if _is_num(base):
                        ax.axhline(base, ls="--", lw=1.0, color=col, alpha=0.8)
            ax.set_xticks(range(len(cells)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
            ax.set_ylabel(ylab, fontsize=8)
            ax.set_title(title, fontsize=9)
            if logy:
                ax.set_yscale("log")
            ax.grid(True, axis="y", which="both", alpha=0.3)

        # Panel (a) MAE -- BROKEN y-axis when an outlier (e.g. deep_attn ss6) dominates.
        mae_series = [(lbl, [mae.get(c, float("nan")) for c in cells])
                      for (lbl, mae, wt, pm, pwt, dm) in data]
        mae_pbe = [(lbl, pm) for (lbl, mae, wt, pm, pwt, dm) in data]
        _broken_bar_panel(fig, top[0, 0], mae_series, labels, mae_pbe,
                          "Held-out reaction-energy MAE (combined)", "kcal/mol",
                          _BASIS_COLORS, bw)
        _panel(fig.add_subplot(top[0, 1]),
               lambda mae, wt, d, c: wt.get(c, float("nan")), "wt",
               "2-subset WTMAD-2 (BH76+W4-11)", "kcal/mol")
        _panel(fig.add_subplot(top[0, 2]),
               lambda mae, wt, d, c: (float(np.mean(d[c])) if d.get(c)
                                      else float("nan")), None,
               "In-sample density RMSE vs CCSD", "density RMSE", logy=True)
        # Legend in its own band BELOW the panels' x-labels and ABOVE the methods
        # (solid bar = NN vs benchmark; dashed = that basis's PBE on the energy
        # panels). The dedicated band keeps it clear of the rotated cell labels.
        handles = []
        for j, (label, *_rest) in enumerate(data):
            col = _BASIS_COLORS[j % len(_BASIS_COLORS)]
            handles.append(Patch(facecolor=col, edgecolor="k",
                                  label=f"{label}: NN (bars)"))
            handles.append(plt.Line2D(
                [], [], ls="--", color=col,
                label=f"{label}: PBE (dashed; energy panels)"))
        fig.legend(handles=handles, loc="center", ncol=min(4, 2 * nb),
                   fontsize=7.5, frameon=False,
                   bbox_to_anchor=(0.5, _f(bot_pad + meth_h + gap1 + legend_h / 2)))
        # Methods: 3 columns under panels a/b/c + a full-width subset-reaction
        # footer below them (top-aligned columns -- the dense content no longer
        # leaves room for the old middle-column nudge). Skipped in bars-only mode.
        if not bars_only:
            _methods_textblock(fig, subsets, y_top=_f(bot_pad + meth_h),
                               fontsize=FS, xs=(0.05, 0.37, 0.69),
                               reactions=reactions, fig_h=fig_h,
                               include_references=include_references)
        fig.suptitle(
            "Cross-basis comparison (union of arch x subset cells; bar absent "
            "where a basis hasn't run) -- NN bars vs benchmark, PBE dashed"
            f"  ·  {run_id}", fontsize=11, y=1.0 - _f(0.16))
        if not bars_only:
            fig.text(0.5, _f(0.09), provenance or _PROVENANCE_BASE, ha="center",
                     fontsize=6, color="#777777")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return out_path


def build_basis_comparison_figures(run_dirs: List[Path], outdir: Path,
                                   eval_subdir: str = "eval_holdout",
                                   archs: Optional[Sequence[str]] = None
                                   ) -> List[Path]:
    """Render the cross-basis comparison for the given run dirs (each labeled by
    its basis+DF from resolved_config.yaml). ``archs`` narrows the comparison to
    the named architectures and switches the filenames to a ``_focus`` stem, so
    the full-union trio is never overwritten by a focused render."""
    if archs is not None and not archs:
        raise ValueError(
            "archs must be non-empty when given (an empty filter would "
            "render blank figures under the full-union filenames); pass "
            "None for the unfiltered comparison.")
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    labels = _disambiguated_run_labels(run_dirs)
    runs = list(zip((Path(rd) for rd in run_dirs), labels))
    rid = " vs ".join(labels) + f" [{_ckpt_label(eval_subdir)}]"
    sfx = ""
    if archs:
        rid += " · " + " + ".join(dict.fromkeys(archs))
        sfx = "_focus"
    return [
        plot_basis_comparison(runs, outdir / f"basis_comparison{sfx}.png", rid,
                              eval_subdir=eval_subdir, archs=archs),
        # variant without the lower references key (columns + subset footer kept)
        plot_basis_comparison(runs, outdir / f"basis_comparison{sfx}_no_refs.png",
                              rid, include_references=False,
                              eval_subdir=eval_subdir, archs=archs),
        # bars-only variant: no bottom notes at all (panels + legend + title) --
        # the easy-to-read figure for a slide/email
        plot_basis_comparison(runs, outdir / f"basis_comparison{sfx}_clean.png",
                              rid, bars_only=True, eval_subdir=eval_subdir,
                              archs=archs),
    ]


def build_diagnostic_figures(run_dirs: List[Path], outdir: Path,
                             eval_subdir: str = "eval_holdout") -> List[Path]:
    """Render the CUMULATIVE (multi-basis) training-loss trajectories -- every
    trained cell from every run, basis by linestyle -- plus the failure-mechanism
    diagnostic that classifies and explains each failing cell."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    labels = _disambiguated_run_labels(run_dirs)
    runs = list(zip((Path(rd) for rd in run_dirs), labels))
    rid = " + ".join(labels) + f" [{_ckpt_label(eval_subdir)}]"
    loss_rows = collect_training_losses_multi(runs)
    return [
        plot_training_losses(loss_rows, outdir / "diagnostic_training_losses.png",
                             rid, highlight=[("deep_attn", 6)]),
        plot_failure_diagnostic(runs, outdir / "diagnostic_failure_mechanisms.png",
                                rid, eval_subdir=eval_subdir),
        plot_capacity_trends(runs, outdir / "diagnostic_capacity_trends.png", rid,
                             eval_subdir=eval_subdir),
    ]


def build_density_energy_figures(run_dir: Path, outdir: Path,
                                 eval_subdir: str = "eval_holdout") -> List[Path]:
    """Render the held-out energy (MAE + 2-subset WTMAD-2) figure and the
    in-sample density-vs-CCSD diagnostic, kept SEPARATE. The in-sample density
    panel always reads ``eval/`` (the final-checkpoint in-sample eval); only the
    held-out energy panels follow ``eval_subdir``. When the held-out
    per-molecule files carry the density columns, the held-out density figure
    AND the DFS Eq. 21 combined energy-density figure
    (``ablation_combined_energy_density.png``) are rendered too; the ED
    per-cell CSV is written alongside but its path is NOT in the returned
    list (the return contract stays PNG-only). Two overview composites ride
    along: ``ablation_insample_overview.png`` is ALWAYS rendered (in-sample
    AE + density; final-checkpoint data, so its panels are identical in the
    final and val-best output dirs), and ``ablation_density_energy_overview.png``
    + the standalone ``ablation_holdout_density_per_arch.png`` + the
    per-channel ``ablation_density_energy_3x3.png`` (with its own per-channel
    ED CSV, path printed, never returned) render whenever the held-out
    density figure does, with placeholder panels where a channel's ED anchors
    are missing; the enriched ``ablation_ed_decomposition.png`` renders with
    the ED figure. The held-out figures carry a ``dataset`` footer line
    stating what the held-out eval is (live reaction/species counts from
    ``_holdout_eval_note``)."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = collect_holdout_reaction_rows(run_dir, eval_subdir=eval_subdir)
    drows = collect_insample_density_rows(run_dir)
    ae_rows = collect_insample_ae_rows(run_dir)
    run_id = f"{run_dir.name} · {_ckpt_label(eval_subdir)}"
    note = coverage_note(run_dir, eval_subdir=eval_subdir)
    # A mid-run density-reference swap makes cells on either side incomparable
    # on the density axis; stamp it on every figure this builder renders (all
    # carry a density or ED panel). Empty for runs without a swap, so their
    # figures are unchanged.
    _lockfix = lockfix_note(run_dir, eval_subdir=eval_subdir)
    lf_cells = lockfix_cell_classes(run_dir, eval_subdir=eval_subdir)
    if not (lf_cells["relocked"] or lf_cells["mixed"]):
        lf_cells = None          # no glyphs on runs without an affected cell
    if _lockfix:
        print(f"  ({_lockfix})")
        note = f"{note}  {_lockfix}" if note else _lockfix
    ecw = _energy_cell_coverage_warning(rows)
    if ecw:
        print(f"  ({ecw})")
        note = f"{note}  {ecw}" if note else ecw
    try:
        baseline = pbe_pool_baseline(run_dir, eval_subdir=eval_subdir)
    except Exception as exc:
        print(f"  (PBE baseline unavailable: {exc})")
        baseline = {"bh76": float("nan"), "w411": float("nan"),
                    "combined": float("nan")}
    # SCAN meta-GGA baseline: all-NaN (no line) unless a precomputed SCAN cache
    # sits by the run -- see scan_pool_baseline / precompute_scan_pool.py.
    try:
        scan_baseline = scan_pool_baseline(run_dir, eval_subdir=eval_subdir)
    except Exception as exc:
        print(f"  (SCAN baseline unavailable: {exc})")
        scan_baseline = _nan_baseline()
    # The SCAN caches are read ONCE here; each panel's anchor is then formed
    # over the species/reactions THAT panel plots (held-out and in-sample are
    # different sets), mirroring how pbe_density_baseline is called per figure.
    try:
        scan_dens_recs = _scan_density_records(run_dir)
    except Exception as exc:
        print(f"  (SCAN density cache unavailable: {exc})")
        scan_dens_recs = {}
    try:
        scan_errs = scan_reaction_errors(run_dir)
    except Exception as exc:
        print(f"  (SCAN reaction errors unavailable: {exc})")
        scan_errs = {}
    _report_scan_density(
        scan_density_baseline(collect_holdout_density_rows(
            run_dir, eval_subdir=eval_subdir), run_dir,
            _records=scan_dens_recs) if scan_dens_recs else None)
    prov = provenance_footer(baseline, scan_baseline)
    caveat = nn_vs_pbe_caveat(rows, baseline)
    dens_prov = ("In-sample density vs CCSD: grid weighted-mean RMSE/L1 on trained "
                 "species (atoms excluded).")
    tsubsets = training_subsets_by_size(run_dir)
    ds_e = _holdout_eval_note(rows, [])
    written = [
        plot_rung_summary(rows, outdir / "ablation_rung_summary.png", run_id,
                          pbe_baseline=baseline, scan_baseline=scan_baseline,
                          note=note, provenance=prov, caveat=caveat,
                          dataset=ds_e),
        plot_energy_wtmad_mae(rows, outdir / "ablation_energy_wtmad_mae.png",
                              run_id, note=note, provenance=prov, caveat=caveat,
                              training_subsets=tsubsets,
                              scan_baseline=scan_baseline,
                              scan_errors=scan_errs, dataset=ds_e),
        plot_insample_density_ccsd(drows,
                                   outdir / "ablation_insample_density_ccsd.png",
                                   run_id, note=note, provenance=dens_prov),
        plot_insample_overview(
            ae_rows, drows, outdir / "ablation_insample_overview.png", run_id,
            note=note,
            provenance=("In-sample AE + density vs CCSD from "
                        "eval/per_molecule.json (atoms excluded); density "
                        "RMSE grid-weight-averaged, NOT N_e-normalized; AE "
                        "vs benchmark reference atomization energies.")),
    ]
    # Held-out density family: only renderable once benchmark CCSD reference
    # densities exist (eval_holdout density columns and/or the run-level
    # pbe_density_errors.json from a --pbe-density-only re-eval); skipped with
    # a note otherwise so current refs-free runs are unchanged.
    hd_rows = collect_holdout_density_rows(run_dir, eval_subdir=eval_subdir)
    pbe_table = load_pbe_density_table(run_dir)
    if hd_rows or pbe_table:
        hd_prov = ("Held-out density vs CCSD: benchmark reference densities "
                   "(xcquinox.alec.benchmark_refs); PBE baseline model-free "
                   "on the same grid.")
        ds = _holdout_eval_note(rows, hd_rows)
        written.append(plot_holdout_density_ccsd(
            hd_rows, outdir / "ablation_holdout_density_ccsd.png", run_id,
            pbe_table=pbe_table, note=note, provenance=hd_prov, dataset=ds,
            scan_density_records=scan_dens_recs))
        written.append(plot_holdout_density_per_arch(
            hd_rows, outdir / "ablation_holdout_density_per_arch.png", run_id,
            pbe_table=pbe_table, note=note, provenance=hd_prov, dataset=ds,
            scan_density_records=scan_dens_recs))
        # DFS Eq. 21 combined ED: needs the NN held-out density (finite
        # density_rmse rows) AND positive PBE anchors on both legs; a
        # pbe_table-only re-eval reaches here but cannot produce ED.
        d_cells = holdout_density_by_arch_subset(hd_rows)
        d_pbe = pbe_density_baseline(hd_rows, pbe_table)
        wt_cells = wtmad2_by_arch_subset(rows)
        e_pbe_wt = wtmad2_pbe_baseline(rows)
        wt_summary: Optional[Dict[str, Any]] = None
        if (d_cells and _is_num(d_pbe) and d_pbe > 0.0 and wt_cells
                and _is_num(e_pbe_wt) and e_pbe_wt > 0.0):
            # SCAN comparator legs, coverage-gated exactly like the
            # per-channel 3x3 path -- previously omitted here, so the
            # headline ED figures never drew ed_scan even with both caches
            # present.
            e_scan_wt, wt_u, wt_r = wtmad2_scan_baseline(rows, scan_errs)
            if not (wt_r and (wt_u / wt_r) >= _SCAN_COVERAGE_FLOOR):
                e_scan_wt = None
            e_scan_mae, mae_u, mae_r = scan_reaction_mae_baseline(rows,
                                                                  scan_errs)
            if not (mae_r and (mae_u / mae_r) >= _SCAN_COVERAGE_FLOOR):
                e_scan_mae = None
            d_scan, d_u, d_r = scan_density_line_counts(
                scan_dens_recs, _pbe_density_map(hd_rows, pbe_table))
            # cell-matched anchors: verdicts compare same-set reductions,
            # for the PBE marks AND the SCAN comparator (the pooled SCAN
            # value understates SCAN on the surviving per-cell sets)
            e_pbe_wt_cells = wtmad2_pbe_by_arch_subset(rows)
            e_pbe_mae_cells = pbe_reaction_mae_by_cell(rows)
            d_pbe_cells = pbe_density_by_cell(hd_rows, pbe_table)
            e_scan_wt_cells = wtmad2_scan_by_cell(rows, scan_errs)
            e_scan_mae_cells = scan_reaction_mae_by_cell(rows, scan_errs)
            d_scan_cells = scan_density_by_cell(hd_rows, scan_dens_recs,
                                                pbe_table)
            wt_summary = combined_ed_by_cell(wt_cells, e_pbe_wt,
                                             d_cells, d_pbe,
                                             e_scan=e_scan_wt, d_scan=d_scan,
                                             e_pbe_by_cell=e_pbe_wt_cells,
                                             d_pbe_by_cell=d_pbe_cells,
                                             e_scan_by_cell=e_scan_wt_cells,
                                             d_scan_by_cell=d_scan_cells)
            if wt_summary.get("ed_scan") is not None:
                wt_summary["scan_suffix"] = _scan_ed_suffix(wt_u, wt_r,
                                                            d_u, d_r)
            mae_cells = reaction_mae_by_arch_subset(rows)
            e_pbe_mae = pbe_reaction_mae_baseline(rows)
            mae_summary = (combined_ed_by_cell(mae_cells, e_pbe_mae,
                                               d_cells, d_pbe,
                                               e_scan=e_scan_mae,
                                               d_scan=d_scan,
                                               e_pbe_by_cell=e_pbe_mae_cells,
                                               d_pbe_by_cell=d_pbe_cells,
                                               e_scan_by_cell=e_scan_mae_cells,
                                               d_scan_by_cell=d_scan_cells)
                           if mae_cells and _is_num(e_pbe_mae)
                           and e_pbe_mae > 0.0 else None)
            if mae_summary and mae_summary.get("ed_scan") is not None:
                mae_summary["scan_suffix"] = _scan_ed_suffix(mae_u, mae_r,
                                                             d_u, d_r)
            ed_prov = ("Energy legs: 2-subset WTMAD-2 (BH76+W4-11 labeled "
                       "reweighting, NOT full GMTKN55) and combined reaction "
                       "MAE. Density leg: held-out grid-weight-averaged RMSE "
                       "vs CCSD (not the Letter's per-electron L1, Eq. 20; "
                       "SI Sec. VI: ranking largely metric-independent, SI "
                       "Eq. 8 R^2 = 0.98). CCSD (not CCSD(T)) references at "
                       "matching basis/grid.")
            if wt_summary.get("ed_scan") is not None:
                ed_prov += (" SCAN comparator legs (coverage-gated): "
                            f"WTMAD-2 over {wt_u}/{wt_r} reactions, density "
                            f"over {d_u}/{d_r} species.")
            ed_anchor_note = _cell_anchor_note(
                {c: v["ed_pbe_cell"] for c, v in wt_summary["cells"].items()
                 if _is_num(v.get("ed_pbe_cell"))},
                glyphs=False)   # line/scatter figures: no bars, no glyphs
            if ed_anchor_note:
                ed_prov += " " + ed_anchor_note
            scan_cell_vals = [v["ed_scan_cell"]
                              for v in wt_summary["cells"].values()
                              if _is_num(v.get("ed_scan_cell"))]
            if scan_cell_vals:
                ed_prov += (" beats-SCAN verdicts likewise cell-matched "
                            f"(anchors {min(scan_cell_vals):.3g}-"
                            f"{max(scan_cell_vals):.3g} kcal/mol); the "
                            "dotted line is the pooled SCAN.")
            extra = [note] if note else []
            excl = _ed_exclusion_note(wt_cells, d_cells)
            if excl:
                extra.append(excl)
            covw = _density_cell_coverage_warning(hd_rows)
            if covw:
                extra.append(covw)
            aw = _pbe_anchor_coverage_warning(hd_rows, pbe_table)
            if aw:
                extra.append(aw)
            if mae_summary:
                shared = [c for c in wt_summary["cells"]
                          if c in mae_summary["cells"]]
                rho = _spearman(
                    [wt_summary["cells"][c]["ED"] for c in shared],
                    [mae_summary["cells"][c]["ED"] for c in shared])
                if _is_num(rho):
                    extra.append(f"Leg agreement: Spearman rho = {rho:.2f} "
                                 f"over {len(shared)} shared cells.")
            written.append(plot_combined_energy_density(
                wt_summary, mae_summary,
                outdir / "ablation_combined_energy_density.png", run_id,
                note="  ".join(extra), provenance=ed_prov, dataset=ds))
            written.append(plot_ed_decomposition(
                wt_summary, outdir / "ablation_ed_decomposition.png",
                run_id, note="  ".join(extra), provenance=ed_prov,
                dataset=ds))
            legs_main: Dict[str, Optional[Dict[str, Any]]] = {
                "wtmad2": wt_summary, "mae": mae_summary}
            counts_main: Dict[str, Tuple[Dict, ...]] = {}
            # DFS-units ED legs: when the pulled data carries the Eq. 20 eps
            # columns, the SAME WTMAD-2 energy cells are re-scored with
            # D = per-cell mean eps and gamma = the Letter's published
            # 1084.87 (dimensionally valid on eps units), plus the own-axes
            # six-functional regression slope when the nonempirical pool
            # cache resolves. Older pulls lack the columns, so the CSV is
            # unchanged.
            eps_cells = holdout_density_by_arch_subset(
                hd_rows, key="density_eps_l1")
            eps_pbe = pbe_density_baseline(hd_rows, pbe_table,
                                           key="density_eps_l1_pbe")
            if eps_cells and _is_num(eps_pbe) and eps_pbe > 0.0:
                # the RMSE legs' coverage guards, re-run on the eps channel:
                # a partial backfill (only some specs re-evaled) leaves the
                # eps legs covering fewer cells/molecules than the RMSE legs
                # -- disclose it on stdout AND in the DFS-units figures' note
                # band rather than shipping silently narrower outputs
                eps_extra = [note] if note else []
                eps_missing = sorted(set(d_cells) - set(eps_cells))
                if eps_missing:
                    miss = ("DFS-units ED: eps columns cover "
                            f"{len(eps_cells)}/{len(d_cells)} density cells; "
                            "missing "
                            + ", ".join(_cell_tag(c) for c in eps_missing)
                            + " -- partial backfill?")
                    print(f"  ({miss})")
                    eps_extra.append(miss)
                aw_eps = _pbe_anchor_coverage_warning(
                    hd_rows, pbe_table, nn_key="density_eps_l1",
                    pbe_key="density_eps_l1_pbe")
                if aw_eps:
                    print(f"  (DFS-units ED eps anchor: {aw_eps})")
                    eps_extra.append(f"Eps anchor: {aw_eps}")
                cw_eps = _density_cell_coverage_warning(
                    hd_rows, key="density_eps_l1")
                if cw_eps:
                    print(f"  (DFS-units ED eps cells: {cw_eps})")
                    eps_extra.append(f"Eps cells: {cw_eps}")
                eps_counts = (_cell_counts(rows, "abs_error_nn_kcalmol"),
                              _cell_counts(hd_rows, "density_eps_l1"),
                              _cell_counts(rows, "abs_error_pbe_kcalmol"))
                # SCAN comparator legs for the DFS-units summaries: the SAME
                # coverage-gated WTMAD-2 energy leg as the RMSE-channel
                # headline, and the density leg re-anchored on the Eq. 20
                # eps channel -- previously omitted here, so the same cell
                # carried ED_scan in the 3x3 DFS-units CSV but a blank in
                # ablation_combined_energy_density.csv.
                d_scan_eps, deps_u, deps_r = scan_density_line_counts(
                    scan_dens_recs,
                    _pbe_density_map(hd_rows, pbe_table,
                                     key="density_eps_l1_pbe"),
                    key="density_eps_l1_pbe")
                d_pbe_eps_cells = pbe_density_by_cell(
                    hd_rows, pbe_table, nn_key="density_eps_l1",
                    pbe_key="density_eps_l1_pbe")
                d_scan_eps_cells = scan_density_by_cell(
                    hd_rows, scan_dens_recs, pbe_table,
                    nn_key="density_eps_l1", pbe_key="density_eps_l1_pbe")
                dfs_summary = combined_ed_fixed_gamma(
                    wt_cells, e_pbe_wt, eps_cells, eps_pbe, _DFS_GAMMA_KCAL,
                    gamma_source="DFS published",
                    e_scan=e_scan_wt, d_scan=d_scan_eps,
                    e_pbe_by_cell=e_pbe_wt_cells,
                    d_pbe_by_cell=d_pbe_eps_cells,
                    e_scan_by_cell=e_scan_wt_cells,
                    d_scan_by_cell=d_scan_eps_cells)
                if dfs_summary.get("ed_scan") is not None:
                    dfs_summary["scan_suffix"] = _scan_ed_suffix(
                        wt_u, wt_r, deps_u, deps_r)
                legs_main["wtmad2_eps_gamma_dfs"] = dfs_summary
                counts_main["wtmad2_eps_gamma_dfs"] = eps_counts
                fit = nonempirical_gamma(run_dir)
                fit_ok = bool(fit and _is_num(fit.get("gamma"))
                              and fit["gamma"] > 0.0)
                fit_summary = None
                if fit_ok:
                    fit_summary = combined_ed_fixed_gamma(
                        wt_cells, e_pbe_wt, eps_cells, eps_pbe, fit["gamma"],
                        gamma_source="own-axes fit",
                        e_scan=e_scan_wt, d_scan=d_scan_eps,
                        e_pbe_by_cell=e_pbe_wt_cells,
                        d_pbe_by_cell=d_pbe_eps_cells,
                        e_scan_by_cell=e_scan_wt_cells,
                        d_scan_by_cell=d_scan_eps_cells)
                    if fit_summary.get("ed_scan") is not None:
                        fit_summary["scan_suffix"] = _scan_ed_suffix(
                            wt_u, wt_r, deps_u, deps_r)
                    legs_main["wtmad2_eps_gamma_fit"] = fit_summary
                    counts_main["wtmad2_eps_gamma_fit"] = eps_counts
                    fit_msg = (f"own-axes gamma = {fit['gamma']:.6g} kcal/mol "
                               f"from {fit['n_functionals']} nonempirical "
                               f"functionals over {fit['n_species']} common "
                               "species"
                               + (f"; {fit['n_species_dropped']} species "
                                  "dropped for unequal support"
                                  if fit.get("n_species_dropped") else ""))
                    print(f"  (DFS-units ED: {fit_msg})")
                    eps_extra.append(fit_msg)
                # the OPERATIVE gamma for every single-gamma DFS-units view:
                # the own-axes six-functional fit when its cache resolves
                # (the calibration performed on THIS data's axes), the
                # published slope only as the fallback -- each panel's stamp
                # names which one it plots
                op_summary = fit_summary if fit_ok else dfs_summary
                # DFS-units parity ED figures: the same panel bodies as the
                # RMSE-channel ED figure, rendered from the fixed-gamma
                # summaries (gamma_mode="fixed" keeps the self-calibration
                # claims off the stamps/labels); panel C carries the
                # own-axes-fit leg when the nonempirical pool cache resolves
                # next to the run dir, a placeholder otherwise.
                eps_prov = (
                    "Energy legs: 2-subset WTMAD-2 (BH76+W4-11 labeled "
                    "reweighting, NOT full GMTKN55) in BOTH line panels. "
                    "Density leg: " + _EPS_N_SYM + " (DFS Eq. 20) vs "
                    "CCSD references at matching basis/grid. CCSD (not "
                    "CCSD(T)) references.")
                if dfs_summary.get("ed_scan") is not None:
                    eps_prov += (" SCAN comparator legs (coverage-gated): "
                                 f"WTMAD-2 over {wt_u}/{wt_r} reactions, "
                                 + _EPS_N_SYM +
                                 f" over {deps_u}/{deps_r} species.")
                eps_anchor_note = _cell_anchor_note(
                    {c: v["ed_pbe_cell"]
                     for c, v in dfs_summary["cells"].items()
                     if _is_num(v.get("ed_pbe_cell"))},
                    glyphs=False)   # line/scatter figures: no glyphs
                if eps_anchor_note:
                    eps_prov += " " + eps_anchor_note
                eps_scan_cell_vals = [v["ed_scan_cell"]
                                      for v in dfs_summary["cells"].values()
                                      if _is_num(v.get("ed_scan_cell"))]
                if eps_scan_cell_vals:
                    eps_prov += (" beats-SCAN verdicts likewise "
                                 "cell-matched (anchors "
                                 f"{min(eps_scan_cell_vals):.3g}-"
                                 f"{max(eps_scan_cell_vals):.3g} kcal/mol); "
                                 "the dotted line is the pooled SCAN.")
                written.append(plot_combined_energy_density(
                    dfs_summary, fit_summary,
                    outdir / "ablation_combined_energy_density_dfs_units.png",
                    run_id, note="  ".join(eps_extra), provenance=eps_prov,
                    caveat=_ED_DFS_UNITS_CAVEAT, dataset=ds,
                    panel_titles=(
                        _ED_N_SYM + ", $\\gamma$ = "
                        f"{_DFS_GAMMA_KCAL:g} (published)",
                        _ED_N_SYM + ", $\\gamma$ = own-axes "
                        "nonempirical fit"),
                    second_leg_placeholder=(
                        "own-axes $\\gamma$ unavailable\n(no nonempirical "
                        "pool cache next to the run dir)"),
                    title=f"Combined {_ED_N_SYM} (DFS Eq. 21 on "
                          f"{_EPS_N_SYM}) -- held-out, NN vs PBE"))
                written.append(plot_ed_decomposition(
                    op_summary,
                    outdir / "ablation_ed_decomposition_dfs_units.png",
                    run_id, note="  ".join(eps_extra), provenance=eps_prov,
                    caveat=_ED_DFS_UNITS_CAVEAT, dataset=ds,
                    title=f"{_ED_N_SYM} decomposition (DFS Eq. 21 on "
                          f"{_EPS_N_SYM}) -- held-out, NN vs PBE"))
                # DFS-units twins of the composite ED surfaces: the held-out
                # overview (E/F under the operative gamma, D parity in eps
                # units) and the per-channel 3x3 (row 3 under ONE shared
                # gamma -- EDs compare across columns, unlike the
                # self-calibrated original -- row 2 parity in eps units,
                # one row-shared frame).
                written.append(plot_density_energy_overview(
                    rows, hd_rows,
                    outdir / "ablation_density_energy_overview_dfs_units.png",
                    run_id, pbe_table=pbe_table, ed_summary=op_summary,
                    note="  ".join(eps_extra),
                    provenance=(
                        "Held-out overview, DFS units. A/B: one-bucket "
                        "WTMAD-2 reduction per pool; C: 2-subset WTMAD-2. "
                        "D: per-species " + _EPS_N_SYM + " (DFS Eq. 20) "
                        "parity vs CCSD refs, PBE model-free. E/F: "
                        + _ED_N_SYM + " with the gamma stamped in-panel -- "
                        "full diagnostics on "
                        "ablation_combined_energy_density_dfs_units.png. "
                        + _CELL_ROWS_GLYPH_NOTE),
                    caveat=_HOLDOUT_OVERVIEW_DFS_UNITS_CAVEAT, dataset=ds,
                    parity_nn_key="density_eps_l1",
                    parity_pbe_key="density_eps_l1_pbe",
                    parity_unit_label=_EPS_N_SYM,
                    title="Held-out overview (DFS units): WTMAD-2 by pool "
                          f"+ {_EPS_N_SYM} vs CCSD + {_ED_N_SYM}"))
                ch_eps_dfs = channel_ed_summaries(
                    rows, hd_rows, pbe_table, fixed_gamma=_DFS_GAMMA_KCAL,
                    gamma_source="DFS published",
                    density_key="density_eps_l1",
                    pbe_density_key="density_eps_l1_pbe",
                    scan_errors=scan_errs,
                    scan_density_records=scan_dens_recs)
                ch_eps_fit = (channel_ed_summaries(
                    rows, hd_rows, pbe_table, fixed_gamma=fit["gamma"],
                    gamma_source="own-axes fit",
                    density_key="density_eps_l1",
                    pbe_density_key="density_eps_l1_pbe",
                    scan_errors=scan_errs,
                    scan_density_records=scan_dens_recs)
                    if fit_ok else None)
                written.append(plot_density_energy_3x3(
                    rows, hd_rows,
                    outdir / "ablation_density_energy_3x3_dfs_units.png",
                    run_id, pbe_table=pbe_table,
                    ch_summaries=(ch_eps_fit if fit_ok else ch_eps_dfs),
                    note="  ".join(eps_extra),
                    provenance=(
                        "Channels: pool-filtered reactions (energy legs) "
                        "and species-membership-filtered densities (overlap "
                        "species in both channels). Density rows: "
                        + _EPS_N_SYM + " (DFS Eq. 20) vs CCSD; ONE gamma "
                        "shared by all channels, stamped in each panel. "
                        + _CELL_ROWS_GLYPH_NOTE),
                    caveat=_3X3_DFS_UNITS_CAVEAT, dataset=ds,
                    density_nn_key="density_eps_l1",
                    density_pbe_key="density_eps_l1_pbe",
                    density_unit_label=_EPS_N_SYM,
                    ed_gamma_label="",
                    lockfix_cells=lf_cells,
                    scan_density_records=scan_dens_recs,
                    scan_errors=scan_errs,
                    title="Per-channel held-out story (DFS units): WTMAD-2 "
                          f"| {_EPS_N_SYM} | {_ED_N_SYM} "
                          "(BH76, W4-11, combined)"))
                written.append(plot_density_parity_by_channel(
                    rows, hd_rows,
                    outdir
                    / "ablation_density_parity_by_channel_dfs_units.png",
                    run_id, pbe_table=pbe_table,
                    nn_key="density_eps_l1", pbe_key="density_eps_l1_pbe",
                    unit_label=_EPS_N_SYM, note="  ".join(eps_extra),
                    provenance=(
                        "Per-species " + _EPS_N_SYM + " (DFS Eq. 20) vs "
                        "CCSD references at matching basis/grid; PBE "
                        "model-free on the same grid. Channel membership "
                        "from the reactions' reactants+products."),
                    caveat=_PARITY_BY_CHANNEL_DFS_UNITS_CAVEAT, dataset=ds,
                    title="Per-species " + _EPS_N_SYM + " parity by "
                          "channel (DFS units) -- held-out, NN vs PBE"))
                pools_of_eps = _species_pools(rows)
                legs3_eps: Dict[str, Optional[Dict[str, Any]]] = {}
                counts3_eps: Dict[str, Tuple[Dict, ...]] = {}
                for ch in ch_eps_dfs:
                    ch_rows_ = rows if ch == "combined" else [
                        r for r in rows if r.get("pool") == ch]
                    ch_hd_ = hd_rows if ch == "combined" else [
                        r for r in hd_rows
                        if ch in pools_of_eps.get(r.get("molecule"), ())]
                    ch_counts = (
                        _cell_counts(ch_rows_, "abs_error_nn_kcalmol"),
                        _cell_counts(ch_hd_, "density_eps_l1"),
                        _cell_counts(ch_rows_, "abs_error_pbe_kcalmol"))
                    legs3_eps[f"{ch}_wtmad2_eps_gamma_dfs"] = ch_eps_dfs[ch]
                    counts3_eps[f"{ch}_wtmad2_eps_gamma_dfs"] = ch_counts
                    if ch_eps_fit is not None:
                        legs3_eps[f"{ch}_wtmad2_eps_gamma_fit"] = \
                            ch_eps_fit[ch]
                        counts3_eps[f"{ch}_wtmad2_eps_gamma_fit"] = ch_counts
                csv3_eps = write_combined_ed_csv(
                    legs3_eps,
                    outdir / "ablation_density_energy_3x3_dfs_units.csv",
                    n_reactions={}, n_density={}, counts_by_leg=counts3_eps)
                print(f"  (per-channel DFS-units ED: wrote {csv3_eps})")
            else:
                print("  (no Eq. 20 eps columns / positive eps PBE anchor in "
                      "this pull -- skipping the DFS-units ED legs and the "
                      "_dfs_units figure twins (combined ED, decomposition, "
                      "overview, 3x3 + CSV, parity-by-channel); a stale "
                      "file from a prior render persists, as with the "
                      "holdout density figure)")
            csv_path = write_combined_ed_csv(
                legs_main,
                outdir / "ablation_combined_energy_density.csv",
                n_reactions=_cell_counts(rows, "abs_error_nn_kcalmol"),
                n_density=_cell_counts(hd_rows, "density_rmse"),
                n_reactions_slice=_cell_counts(rows,
                                               "abs_error_pbe_kcalmol"),
                counts_by_leg=counts_main or None)
            gtxt = f"gamma_wt = {wt_summary['gamma']:.4g}"
            if mae_summary:
                gtxt += f", gamma_mae = {mae_summary['gamma']:.4g}"
            print(f"  (combined ED: {gtxt}; wrote {csv_path})")
        else:
            print("  (no NN held-out density and/or positive PBE anchors -- "
                  "skipping ablation_combined_energy_density.png/.csv, "
                  "ablation_ed_decomposition.png, and their _dfs_units "
                  "twins; a stale file from a prior render persists, as "
                  "with the holdout density figure)")
        # Held-out overview composite: same gate as the holdout density figure
        # (renders whenever it does); panel F degrades to the "ED unavailable"
        # placeholder when the ED anchors above were missing (wt_summary None).
        ov_prov = _overview_provenance(wt_summary)
        ov_anchor_note = _cell_anchor_note(wtmad2_pbe_by_arch_subset(rows))
        if ov_anchor_note:
            ov_prov += " " + ov_anchor_note
        written.append(plot_density_energy_overview(
            rows, hd_rows,
            outdir / "ablation_density_energy_overview.png", run_id,
            pbe_table=pbe_table, ed_summary=wt_summary, note=note,
            provenance=ov_prov, dataset=ds))
        # Per-channel 3x3 + its CSV: renders whenever held-out density
        # exists; channels degrade individually inside the figure.
        ch_summaries = channel_ed_summaries(
            rows, hd_rows, pbe_table, scan_errors=scan_errs,
            scan_density_records=scan_dens_recs)
        prov_3x3 = ("Channels: pool-filtered reactions (energy legs) and "
                    "species-membership-filtered densities (membership from "
                    "reaction reactants+products; overlap species in both "
                    "channels). Density leg: grid-weight-averaged RMSE vs "
                    "CCSD; per-channel gamma = that channel's E_PBE/D_PBE. "
                    "beats marks: each arch vs its own-rung reference's "
                    "cell-matched anchor, per channel (PBE for GGA archs, "
                    "SCAN for meta-GGA/rung-3.5). " + _CELL_ROWS_GLYPH_NOTE)
        written.append(plot_density_energy_3x3(
            rows, hd_rows, outdir / "ablation_density_energy_3x3.png",
            run_id, pbe_table=pbe_table, ch_summaries=ch_summaries,
            note=note, provenance=prov_3x3, dataset=ds,
            lockfix_cells=lf_cells, scan_density_records=scan_dens_recs,
            scan_errors=scan_errs))
        pools_of = _species_pools(rows)
        legs3: Dict[str, Optional[Dict[str, Any]]] = {}
        counts3: Dict[str, Tuple[Dict, ...]] = {}
        for ch, s in ch_summaries.items():
            leg = f"{ch}_wtmad2"
            legs3[leg] = s
            ch_rows = rows if ch == "combined" else [
                r for r in rows if r.get("pool") == ch]
            ch_hd = hd_rows if ch == "combined" else [
                r for r in hd_rows
                if ch in pools_of.get(r.get("molecule"), ())]
            counts3[leg] = (_cell_counts(ch_rows, "abs_error_nn_kcalmol"),
                            _cell_counts(ch_hd, "density_rmse"),
                            _cell_counts(ch_rows, "abs_error_pbe_kcalmol"))
        csv3 = write_combined_ed_csv(
            legs3, outdir / "ablation_density_energy_3x3.csv",
            n_reactions={}, n_density={}, counts_by_leg=counts3)
        print(f"  (per-channel ED: wrote {csv3})")
        # the 3x3's former parity row as its own figure (RMSE channel)
        written.append(plot_density_parity_by_channel(
            rows, hd_rows,
            outdir / "ablation_density_parity_by_channel.png", run_id,
            pbe_table=pbe_table, note=note,
            provenance=("Per-species grid-weighted density RMSE vs CCSD "
                        "references at matching basis/grid; PBE model-free "
                        "on the same grid. Channel membership from the "
                        "reactions' reactants+products."),
            dataset=ds))
    else:
        print("  (no held-out density data -- skipping "
              "ablation_holdout_density_ccsd.png; needs benchmark CCSD refs)")
    return written


def build_per_run_diagnostics(run_dir: Path, outdir: Path,
                              basis_label: Optional[str] = None,
                              eval_subdir: str = "eval_holdout") -> List[Path]:
    """Per-run diagnostics kept in each basis's own ``figures_<alias>/`` dir: the
    size-consistency (additivity) diagnostic over the capacity ladder at the
    smallest subset_size (where overfitting -- and the per-atom error it produces
    -- is worst), and the single-run training-loss trajectories. Wired into
    :func:`build_bh76w411_suite` so a fresh pull refreshes them too (they were
    previously generated by hand and went stale)."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    run_id = f"{run_dir.name} · {_ckpt_label(eval_subdir)}"
    note = coverage_note(run_dir, eval_subdir=eval_subdir)
    rows = collect_holdout_reaction_rows(run_dir, eval_subdir=eval_subdir)
    ecw = _energy_cell_coverage_warning(rows)
    if ecw:
        print(f"  ({ecw})")
        note = f"{note}  {ecw}" if note else ecw
    written: List[Path] = []
    # Capacity ladder at the smallest available subset_size: added capacity
    # steepens the per-atom (size-consistency) error, clearest at small ss.
    present = list(reaction_mae_by_arch_subset(rows).keys())
    if present:
        ss0 = min(ss for _, ss in present)
        order = {a: i for i, a in enumerate(ARCH_ORDER)}
        sc_cells = sorted((cell for cell in present if cell[1] == ss0),
                          key=lambda c: (order.get(c[0], len(ARCH_ORDER)), c[0]))
        written.append(plot_size_consistency_diagnostic(
            rows, outdir / "diagnostic_size_consistency.png", run_id, sc_cells,
            note=note, dataset=_holdout_eval_note(rows, [])))
    loss_rows = collect_training_losses(
        run_dir, basis_label=basis_label or run_basis_label(run_dir))
    written.append(plot_training_losses(
        loss_rows, outdir / "diagnostic_training_losses.png", run_id, note=note,
        highlight=[("deep_attn", 6)]))
    return written


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

_DEFAULT_LOCAL_ROOT = Path.home() / "Documents/Research/xcquinox-results/runs"
_DEFAULT_CATEGORY = "ablation_notransform/polarized/runs"


def _resolve_run_dir(run_dir: Optional[str]) -> Path:
    if run_dir:
        return Path(run_dir).expanduser().resolve()
    cats = ccp.discover_pulled_categories(_DEFAULT_LOCAL_ROOT)
    rd = cats.get(_DEFAULT_CATEGORY)
    if rd is None:
        raise SystemExit(
            f"No pulled run found under {_DEFAULT_LOCAL_ROOT / _DEFAULT_CATEGORY}; "
            "pass --run-dir explicitly.")
    return rd


def build_all(run_dir: Path, outdir: Path,
              eval_subdir: str = "eval_holdout") -> List[Path]:
    """Collect once, render every figure. Returns the written PNG paths."""
    outdir.mkdir(parents=True, exist_ok=True)
    run_id = f"{run_dir.name} · {_ckpt_label(eval_subdir)}"
    reaction_rows = collect_holdout_reaction_rows(run_dir, eval_subdir=eval_subdir)
    insample_rows = collect_insample_ae_rows(run_dir)
    n_trained = trained_spec_count(run_dir, eval_subdir=eval_subdir)
    n_total = len(ccp._read_manifest_cells(run_dir)) or len(ccp._spec_dirs(run_dir))
    n_holdout = len({r["idx"] for r in reaction_rows})
    note = coverage_note(run_dir, eval_subdir=eval_subdir)
    print(f"  coverage: {note}")
    ecw = _energy_cell_coverage_warning(reaction_rows)
    if ecw:
        print(f"  ({ecw})")
        note = f"{note}  {ecw}" if note else ecw

    # Live, non-hardcoded footers (degrade to "n/a" if the pool can't be loaded).
    try:
        baseline = pbe_pool_baseline(run_dir, eval_subdir=eval_subdir)
    except Exception as exc:  # pool unavailable (e.g. GMTKN55 clone absent)
        print(f"  (PBE baseline unavailable: {exc})")
        baseline = {"bh76": float("nan"), "w411": float("nan"),
                    "combined": float("nan")}
    # SCAN meta-GGA baseline: all-NaN (no SCAN line) unless a precomputed cache
    # sits by the run (precompute_scan_pool.py); older runs render as before.
    try:
        scan_baseline = scan_pool_baseline(run_dir, eval_subdir=eval_subdir)
    except Exception as exc:
        print(f"  (SCAN baseline unavailable: {exc})")
        scan_baseline = {"bh76": float("nan"), "w411": float("nan"),
                         "combined": float("nan")}
    # Per-reaction SCAN errors back the row-matched SCAN line on the per-arch
    # MAE figure; {} when the cache is absent (the figure then falls back to
    # the full-pool scan_baseline value, exactly as before).
    try:
        scan_errs = scan_reaction_errors(run_dir)
    except Exception as exc:
        print(f"  (SCAN per-reaction errors unavailable: {exc})")
        scan_errs = {}
    prov = provenance_footer(baseline, scan_baseline)
    # These five figures stamp their footers with bespoke fig.text stacks (no
    # _stamp_parity_footer dataset slot), so the dataset sentence rides the
    # grey provenance line instead of a dedicated line.
    ds_e = _holdout_eval_note(reaction_rows, [])
    if ds_e:
        prov = prov + " " + ds_e
    caveat = nn_vs_pbe_caveat(reaction_rows, baseline)
    print(f"  PBE baseline (full pool): BH76 {_fmt_mae(baseline['bh76'])} / "
          f"W4-11 {_fmt_mae(baseline['w411'])} / "
          f"combined {_fmt_mae(baseline['combined'])}"
          f"{_pool_cov_bracket(baseline)}")
    _report_scan_coverage(scan_baseline)

    written: List[Path] = []
    written.append(plot_parity(
        reaction_rows, outdir / "ablation_parity.png", run_id, note=note,
        provenance=prov, caveat=caveat))
    written.append(plot_arch_subset_heatmap(
        reaction_rows, insample_rows, outdir / "ablation_arch_subset_heatmap.png",
        run_id, n_trained=n_trained, n_total=n_total, n_holdout=n_holdout,
        note=note, provenance=prov))
    written.append(plot_arch_subset_heatmap_vs_pbe(
        reaction_rows, outdir / "ablation_arch_subset_heatmap_vs_pbe.png",
        run_id, note=note, provenance=prov))
    written.append(plot_mae_by_arch(
        reaction_rows, insample_rows, outdir / "ablation_mae_by_arch.png", run_id,
        note=note, provenance=prov, scan_baseline=scan_baseline,
        scan_errors=scan_errs))
    written.append(plot_mae_vs_subset(
        reaction_rows, insample_rows, outdir / "ablation_mae_vs_subset.png", run_id,
        note=note, provenance=prov, pbe_baseline=baseline,
        scan_baseline=scan_baseline))
    written.append(plot_ae_parity(
        reaction_rows, outdir / "ablation_ae_parity.png", run_id, note=note,
        provenance=prov))
    written.append(plot_parity_by_class(
        reaction_rows, outdir / "ablation_parity_by_class.png", run_id,
        note=note, provenance=prov, caveat=caveat))
    return written


_BH76W411_BASES: Tuple[str, ...] = ("svp_grid2", "tzvpd_grid2_df")


def _basis_fig_alias(basis: str) -> str:
    """Output-dir alias for a basis: ``svp_grid2`` -> ``svp`` (figures_svp),
    ``tzvpd_grid2_df`` -> ``tzvpd_df`` (figures_tzvpd_df)."""
    return basis.replace("_grid2", "")


def _newest_run_per_basis(results_root: Path,
                          bases: Tuple[str, ...] = _BH76W411_BASES,
                          domain: str = "bh76w411_repr") -> Dict[str, Path]:
    """Map each basis -> its newest ``run_*`` dir under
    ``<results_root>/<domain>/<basis>/runs/``. ISO-Z timestamps sort
    lexicographically, so the last ``sorted`` entry is the latest pull."""
    out: Dict[str, Path] = {}
    for basis in bases:
        runs_dir = Path(results_root) / domain / basis / "runs"
        runs = sorted(runs_dir.glob("run_*"))
        if not runs:
            raise FileNotFoundError(f"no run_* dir under {runs_dir}")
        out[basis] = runs[-1]
    return out


def figure_cell_coverage(run_dir: Path,
                         eval_subdir: str = "eval_holdout") -> Dict[str, Any]:
    """What the figures will actually render for a run: every held-out (arch,
    subset_size) cell, plus a guard list ``archs_not_in_order`` of archs present
    in the data but absent from ``ARCH_ORDER`` (the per-arch plots cannot
    order/colour those, so they would be silently dropped)."""
    mae = reaction_mae_by_arch_subset(
        collect_holdout_reaction_rows(run_dir, eval_subdir=eval_subdir))
    cells = sorted(mae.keys())
    archs = sorted({a for a, _ in cells})
    return {
        "run": run_dir.name,
        "n_cells": len(cells),
        "cells": cells,
        "archs": archs,
        "subsets": sorted({s for _, s in cells}),
        # archs the figure CANNOT render (present in data, absent from ARCH_ORDER)
        "archs_not_in_order": [a for a in archs if a not in ARCH_ORDER],
        # ARCH_ORDER archs with NO held-out eval cell yet (run still in progress);
        # judged by eval coverage, not model.eqx (weights are often not pulled)
        "archs_missing": [a for a in ARCH_ORDER if a not in archs],
        "coverage": arch_coverage(run_dir, eval_subdir=eval_subdir),
    }


def build_bh76w411_suite(results_root: Optional[Path] = None,
                         outroot: Optional[Path] = None,
                         bases: Tuple[str, ...] = _BH76W411_BASES,
                         domain: str = "bh76w411_repr",
                         comparison_archs: Optional[Tuple[str, ...]] = None
                         ) -> List[Path]:
    """Regenerate EVERY figure family for ``domain`` from the newest run per
    basis, so a fresh spec pull lands on all figures in one call. Per basis: the
    arch-aware ablation set (:func:`build_all`), the held-out energy/density set
    (:func:`build_density_energy_figures`), the five parity-layout variants
    (:func:`build_parity_variants`) and the per-run size-consistency/training-loss
    diagnostics (:func:`build_per_run_diagnostics`) into ``figures_<alias>/``.
    Cross-basis (only when >= 2 bases have eval coverage -- a one-run "basis
    comparison" is misleading): the basis comparison + its no-references variant
    (:func:`build_basis_comparison_figures`) and the diagnostic set
    (:func:`build_diagnostic_figures`) into ``figures_basis_comparison/``.
    ``comparison_archs`` additionally renders the ``basis_comparison_focus*``
    trio restricted to the named archs (the full-union files are kept).

    For a non-default ``domain`` (e.g. ``dfs_step7``) every output dir name is
    prefixed with the domain (``figures_dfs_step7_svp/``) so the bh76w411 sets
    are never overwritten.

    Emits TWO parallel figure sets per the checkpoint variant the cluster now
    evaluates: the final-step set from ``eval_holdout/`` (into ``figures_<alias>/``
    + ``figures_basis_comparison/``) and the val-best set from
    ``eval_holdout_val_best/`` (into ``figures_<alias>_val_best/`` +
    ``figures_basis_comparison_val_best/``) -- scored from the held-out
    validation-best weights, which (unlike the min-training-loss checkpoint) do not
    select the most-overfit step. The val-best set is produced for every basis whose
    ``eval_holdout_val_best/`` data was pulled.

    Prints a per-run coverage report and FAILS LOUD if a run carries an arch
    outside ``ARCH_ORDER`` (which the per-arch plots would drop); incomplete runs
    (archs not yet eval'd) are reported, not masked. Returns every written path.
    Figures are regenerated outputs -- callers do not version-control them."""
    results_root = Path(results_root) if results_root else _DEFAULT_LOCAL_ROOT
    outroot = Path(outroot) if outroot else Path(__file__).resolve().parent
    prefix = "" if domain == "bh76w411_repr" else f"{domain}_"
    runs = _newest_run_per_basis(results_root, bases, domain=domain)
    written: List[Path] = []
    for eval_subdir, suffix in (("eval_holdout", ""),
                                ("eval_holdout_val_best", "_val_best")):
        is_val_best = eval_subdir != "eval_holdout"
        ordered_runs: List[Path] = []
        for basis in bases:
            run = runs[basis]
            cov = figure_cell_coverage(run, eval_subdir=eval_subdir)
            if is_val_best and cov["n_cells"] == 0:
                continue  # no val-best eval pulled for this basis yet
            ordered_runs.append(run)
            print(f"[{basis} | {eval_subdir}] {cov['run']}: {cov['n_cells']} "
                  f"cells  archs={cov['archs']}  subsets={cov['subsets']}")
            if cov["archs_not_in_order"]:
                raise ValueError(
                    f"{basis} {cov['run']} has archs not in ARCH_ORDER "
                    f"{cov['archs_not_in_order']}; add them to ARCH_ORDER/"
                    "ARCH_COLOR (and the per-arch F_x/F_c forms) before "
                    "regenerating, else they are dropped from the figures.")
            if cov["archs_missing"]:
                print(f"   (incomplete -- ARCH_ORDER archs with no held-out eval "
                      f"cell yet: {cov['archs_missing']})")
            fdir = outroot / f"figures_{prefix}{_basis_fig_alias(basis)}{suffix}"
            written += build_all(run, fdir, eval_subdir=eval_subdir)
            written += build_density_energy_figures(run, fdir,
                                                    eval_subdir=eval_subdir)
            written += build_parity_variants(run, fdir, eval_subdir=eval_subdir)
            written += build_per_run_diagnostics(run, fdir, run_basis_label(run),
                                                 eval_subdir=eval_subdir)
        if not ordered_runs:
            if is_val_best:
                print("   (no eval_holdout_val_best/ data found -- skipping the "
                      "val-best figure set)")
            continue
        if len(ordered_runs) < 2:
            print(f"   (only one basis with {eval_subdir}/ coverage -- "
                  "skipping the basis-comparison figure set)")
            continue
        cmp_dir = outroot / f"figures_{prefix}basis_comparison{suffix}"
        written += build_basis_comparison_figures(ordered_runs, cmp_dir,
                                                  eval_subdir=eval_subdir)
        if comparison_archs:
            # focused trio (basis_comparison_focus*) alongside the full union --
            # readable column count when the union spans many arch x subset cells
            written += build_basis_comparison_figures(ordered_runs, cmp_dir,
                                                      eval_subdir=eval_subdir,
                                                      archs=comparison_archs)
        written += build_diagnostic_figures(ordered_runs, cmp_dir,
                                            eval_subdir=eval_subdir)
    return written


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", default=None,
                   help="pulled run dir (default: latest under "
                        f"{_DEFAULT_LOCAL_ROOT / _DEFAULT_CATEGORY})")
    p.add_argument("--outdir", default=str(
        Path(__file__).resolve().parent / "figures_ablation_notransform"),
        help="output directory for PNGs (single-run mode)")
    p.add_argument("--suite", action="store_true",
                   help="regenerate ALL figure families for --domain (every "
                        "--bases basis) from the newest run per basis, into "
                        "figures_<basis>/ + figures_basis_comparison/ under "
                        "--outroot")
    p.add_argument("--results-root", default=None,
                   help="results runs root for --suite "
                        f"(default: {_DEFAULT_LOCAL_ROOT})")
    p.add_argument("--domain", default="bh76w411_repr",
                   help="results domain under the runs root for --suite; "
                        "non-default domains get domain-prefixed figure dirs, "
                        "e.g. --domain dfs_step7 -> figures_dfs_step7_svp/ "
                        "(default: bh76w411_repr)")
    p.add_argument("--bases", default=",".join(_BH76W411_BASES),
                   help="comma-separated basis subdirs to render for --suite "
                        f"(default: {','.join(_BH76W411_BASES)})")
    p.add_argument("--outroot", default=None,
                   help="directory the figures_* dirs are written under for "
                        "--suite (default: next to this script)")
    p.add_argument("--comparison-archs", default=None,
                   help="comma-separated arch names; when given, --suite ALSO "
                        "writes a basis_comparison_focus* trio restricted to "
                        "these archs (readable column count when the full "
                        "union of arch x subset cells is wide)")
    args = p.parse_args(argv)

    if args.suite:
        bases = tuple(b.strip() for b in args.bases.split(",") if b.strip())
        cmp_archs = (tuple(a.strip() for a in args.comparison_archs.split(",")
                           if a.strip())
                     if args.comparison_archs else None)
        written = build_bh76w411_suite(results_root=args.results_root,
                                       outroot=args.outroot,
                                       bases=bases,
                                       domain=args.domain,
                                       comparison_archs=cmp_archs)
        for pth in written:
            print(f"  wrote {pth}")
        return 0

    run_dir = _resolve_run_dir(args.run_dir)
    outdir = Path(args.outdir).expanduser().resolve()
    print(f"run_dir: {run_dir}")
    written = build_all(run_dir, outdir)
    for pth in written:
        print(f"  wrote {pth}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
