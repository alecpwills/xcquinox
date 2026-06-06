#!/usr/bin/env python
"""Architecture-ablation figures for the ``ablation_notransform`` sweep.

The existing :mod:`make_cluster_pulls_figure` renders a category-level suite
but keys every series on ``(metric, solver)`` — so it collapses the eight
architectures of this ablation (which holds ``metric=jsd``/``solver=full_3``
fixed and varies ``arch``) into a single line. This module fills that gap:
every figure here is **architecture-aware**, and the parity figure is modeled
on Figure 5 of Navarro-Rodriguez et al., *Constraint-aware functional cloning*
(MLXC_Constraints, 2026) — predicted-vs-reference scatter with a y=x diagonal
and a per-network mean-error inset.

It reuses the data collectors and house style from
``make_cluster_pulls_figure`` verbatim (no re-parsing of the run dir), adding
only an ``eval_holdout/per_reaction.json`` collector (the cluster-side held-out
reaction eval — same schema as the local-reeval ``local_per_reaction.json``
that the existing module reads, but a different source path).

Scientific provenance carried on every figure:
  * The pulled run ``run_20260529T165503Z`` predates the ``dm_entropy`` fix
    from the 2026-05-29 forensic review — these are PRE-FIX numbers.
  * On the held-out reactions ``de_nn ≈ de_pbe`` while both sit far from the
    benchmark refs: the network faithfully *reproduces PBE*, it does not beat
    it. The two parity panels make that explicit rather than hiding it.
  * Coverage is partial (57/80 specs trained; only 32 carry held-out
    reactions) — incomplete grid cells are drawn hatched, never dropped.

Figures written (PNG):
  A. ``ablation_parity.png``       — Fig-5 analog, 2 panels (NN-vs-PBE and
     NN&PBE-vs-benchmark), points colored by arch, per-arch MAE inset bars.
  B. ``ablation_arch_subset_heatmap.png`` — arch × subset_size MAE heatmap
     (held-out reaction MAE + in-sample atomization-energy MAE).
  C. ``ablation_mae_by_arch.png``  — per-arch MAE bars (log-y), held-out
     reaction + in-sample AE, with the PBE-vs-benchmark baseline line.
  D. ``ablation_mae_vs_subset.png``— MAE vs subset_size, one line per arch.
  E. ``ablation_ae_parity.png``    — HELD-OUT atomization-energy parity (W4-11,
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
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
ARCH_ORDER: Tuple[str, ...] = (
    "deep", "deep_attn", "deep_cusp", "deep_dm",
    "deep_combined", "deep_combined_attn",
    "deep_notransform", "deep_notransform_attn",
)
_ARCH_TAB = plt.get_cmap("tab10")
ARCH_COLOR: Dict[str, str] = {
    a: matplotlib.colors.to_hex(_ARCH_TAB(i)) for i, a in enumerate(ARCH_ORDER)
}
SUBSET_SIZES: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 12, 15, 18)
POOL_MARKER: Dict[str, str] = {"bh76": "o", "w411": "^"}

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

def collect_holdout_reaction_rows(run_dir: Path) -> List[Dict[str, Any]]:
    """Read every ``checkpoints/spec_*/eval_holdout/per_reaction.json`` (the
    cluster-side held-out reaction eval) and join with the manifest cell.

    One row per (spec, reaction); schema mirrors
    ``ccp.collect_per_reaction_rows`` but sourced from ``eval_holdout/`` rather
    than the local-reeval ``eval/local_per_reaction.json``. Specs without the
    file (e.g. the 25 specs whose held-out eval did not run) are skipped.
    """
    cells = ccp._read_manifest_cells(run_dir)
    rows: List[Dict[str, Any]] = []
    for idx, spec_dir in ccp._spec_dirs(run_dir):
        rj_path = spec_dir / "eval_holdout" / "per_reaction.json"
        if not rj_path.is_file():
            continue
        try:
            with rj_path.open() as f:
                payload = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue
        cell = cells.get(idx, {})
        for r in payload:
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
            })
    return rows


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


def trained_spec_count(run_dir: Path) -> int:
    """Number of specs with a materialized ``model.eqx`` (trained)."""
    n = 0
    for _idx, spec_dir in ccp._spec_dirs(run_dir):
        if (spec_dir / "model.eqx").is_file():
            n += 1
    return n


def arch_coverage(run_dir: Path) -> Dict[str, List[str]]:
    """Per-arch coverage of this (partial) run, computed from disk.

    Returns ``{"trained": [...], "holdout": [...], "insample": [...],
    "untrained": [...]}`` — arch names in ``ARCH_ORDER`` order. ``trained``
    = has ``model.eqx``; ``holdout`` = has ``eval_holdout/per_reaction.json``;
    ``insample`` = has ``eval/per_molecule.json``; ``untrained`` = arch in the
    manifest grid with no trained spec at all.
    """
    cells = ccp._read_manifest_cells(run_dir)
    trained: set = set()
    holdout: set = set()
    insample: set = set()
    grid_archs: set = {c.get("arch") for c in cells.values() if c.get("arch")}
    for idx, spec_dir in ccp._spec_dirs(run_dir):
        arch = cells.get(idx, {}).get("arch")
        if arch is None:
            continue
        if (spec_dir / "model.eqx").is_file():
            trained.add(arch)
        if (spec_dir / "eval_holdout" / "per_reaction.json").is_file():
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
        "untrained": _ordered(grid_archs - trained),
    }


def coverage_note(run_dir: Path) -> str:
    """One-line human summary of arch coverage for figure footers — makes the
    partial-run gaps explicit (no silent truncation)."""
    cov = arch_coverage(run_dir)
    parts = [f"Held-out reactions: {len(cov['holdout'])}/{len(ARCH_ORDER)} archs "
             f"({', '.join(cov['holdout']) or 'none'})."]
    if cov["untrained"]:
        parts.append(f"NOT TRAINED in this run: {', '.join(cov['untrained'])}.")
    trained_no_holdout = [a for a in cov["trained"] if a not in cov["holdout"]]
    if trained_no_holdout:
        parts.append(f"Trained but no held-out eval: "
                     f"{', '.join(trained_no_holdout)}.")
    return "  ".join(parts)


# ---------------------------------------------------------------------------
# Live (non-hardcoded) footer baselines
# ---------------------------------------------------------------------------

def _first_pbe_energies(run_dir: Path) -> Dict[str, float]:
    """PBE energy map (molecule -> Hartree) from the first spec that carries an
    ``eval_holdout/per_molecule.json``. PBE is invariant to the trained NN (only
    SCF noise ~2.5e-6 Ha across specs), so any spec serves as the baseline."""
    for pm in sorted(Path(run_dir).glob(
            "checkpoints/spec_*/eval_holdout/per_molecule.json")):
        energies = {r["molecule"]: r["E_pbe"]
                    for r in json.loads(pm.read_text())
                    if isinstance(r.get("E_pbe"), (int, float))}
        if energies:
            return energies
    return {}


def pbe_pool_baseline(run_dir: Path, *, _loader=None) -> Dict[str, float]:
    """Full-pool PBE reaction-energy MAE (kcal/mol): ``{bh76, w411, combined}``.

    The benchmark's inherent difficulty, independent of any train/held-out split
    (PBE does not depend on the trained NN), computed LIVE so the figure footers
    are never stale. Reuses the validated reaction math in
    ``xcquinox.alec.eval_holdout`` over the canonical pool from
    ``load_full_held_out_pools`` -- so it covers ALL 76 BH76 / 140 W4-11
    reactions, including the few that are in-sample in every spec and thus absent
    from any held-out file. ``_loader`` is a test seam (default:
    ``load_full_held_out_pools``)."""
    if _loader is None:
        from xcquinox.alec.full_benchmark_pools import load_full_held_out_pools
        _loader = load_full_held_out_pools
    from xcquinox.alec.eval_holdout import reaction_mae_kcalmol
    _, full_rxns = _loader()
    pbe = _first_pbe_energies(run_dir)
    out: Dict[str, float] = {}
    for pool in ("bh76", "w411"):
        rx = [r for r in full_rxns if r.get("source_pool") == pool]
        out[pool] = reaction_mae_kcalmol(pbe, rx)[0] if rx else float("nan")
    out["combined"] = (reaction_mae_kcalmol(pbe, list(full_rxns))[0]
                       if full_rxns else float("nan"))
    return out


def _fmt_mae(x: Any) -> str:
    return (f"{x:.2f}" if isinstance(x, (int, float)) and math.isfinite(x)
            else "n/a")


def provenance_footer(baseline: Dict[str, float]) -> str:
    """Static methodology banner + the LIVE full-pool PBE baseline."""
    return (_PROVENANCE_BASE + f" PBE: BH76 {_fmt_mae(baseline.get('bh76'))}"
            f" / W4-11 {_fmt_mae(baseline.get('w411'))}"
            f" / combined {_fmt_mae(baseline.get('combined'))}.")


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
    (``abs_error_nn_kcalmol`` or ``abs_error_pbe_kcalmol``)."""
    buckets: Dict[Tuple[str, int], List[float]] = {}
    for r in rows:
        arch, ss = r.get("arch"), r.get("subset_size")
        if arch is None or ss is None:
            continue
        if _is_num(r.get(key)):
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
    """For each arch, the LARGEST subset_size that has held-out reactions —
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
# Figure A — Fig-5-style parity
# ---------------------------------------------------------------------------

def _draw_mae_inset(ax, mae_by_arch: Dict[str, float], archs: List[str], *,
                    title: str, baseline: Optional[float] = None,
                    baseline_label: str = "PBE") -> None:
    """Per-arch MAE bar inset (lower-right), color-matched to the scatter —
    the analog of the paper's Fig-5 mean-relative-error inset."""
    # Low-right, but lifted just enough that the angled (40°) tick labels clear
    # the outer panel's x-axis — low enough that the inset body stays under the
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
    """Figure A — two-panel parity, points colored by arch, y=x diagonal,
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

        # Panel (a): optimized NN vs PBE — how far subset training moved the
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
        handles = [Patch(facecolor=ARCH_COLOR[a], label=a) for a in archs]
        handles.append(plt.Line2D([], [], marker="o", ls="", color="0.4",
                                   label="bh76 (●) / w411 (▲) by marker"))
        # Shared arch legend in its own reserved band below the panels — the
        # bottom strip is stacked (legend > note > provenance) with no overlap.
        fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=7,
                   frameon=False, bbox_to_anchor=(0.5, 0.085))

        fig.suptitle(
            "Reaction-energy parity (Fig-5 analog) — "
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
# Figure B — arch × subset_size heatmaps
# ---------------------------------------------------------------------------

def _heatmap_panel(ax, mae_map: Dict[Tuple[str, int], float], archs: List[str],
                   *, title: str, cbar_label: str,
                   center: Optional[float] = None,
                   subset_sizes: Optional[Sequence[int]] = None) -> None:
    """arch x subset_size heatmap. Default: log-scaled viridis (raw MAE spanning
    decades). With ``center`` set (e.g. 1.0 for a MAE/PBE ratio): a diverging
    RdBu_r map about ``center`` -- below center is blue (better than the
    reference), above is red (worse). Missing cells are hatched either way.
    ``subset_sizes`` overrides the column axis (default the global SUBSET_SIZES);
    pass the present sizes to drop empty trailing columns."""
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
    ax.set_yticklabels(archs, fontsize=7)
    ax.set_xlabel("training subset_size")
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)


def plot_arch_subset_heatmap(reaction_rows: List[Dict[str, Any]],
                             insample_rows: List[Dict[str, Any]],
                             out_path: Path, run_id: str, *,
                             n_trained: int, n_total: int,
                             n_holdout: int, note: str = "",
                             provenance: Optional[str] = None) -> Path:
    """Figure B — arch × subset_size MAE heatmaps (held-out reactions +
    in-sample AE). Missing cells hatched; coverage stated in the footer."""
    with plt.rc_context(_STYLE):
        archs = _archs_present(reaction_rows) or list(ARCH_ORDER)
        archs_ae = _archs_present(insample_rows)
        all_archs = [a for a in ARCH_ORDER if a in set(archs) | set(archs_ae)]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.6))
        _heatmap_panel(ax1, reaction_mae_by_arch_subset(reaction_rows),
                       all_archs, title="Held-out reaction-energy MAE (NN)",
                       cbar_label="MAE (kcal/mol)")
        _heatmap_panel(ax2, ae_mae_by_arch_subset(insample_rows), all_archs,
                       title="In-sample atomization-energy MAE",
                       cbar_label="MAE (kcal/mol)")
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


# ---------------------------------------------------------------------------
# Figure C — per-arch MAE bars
# ---------------------------------------------------------------------------

def plot_mae_by_arch(reaction_rows: List[Dict[str, Any]],
                     insample_rows: List[Dict[str, Any]],
                     out_path: Path, run_id: str, note: str = "",
                     provenance: Optional[str] = None) -> Path:
    """Figure C — per-arch MAE bars (log-y): held-out reaction MAE (mean &
    best over available subsets) + in-sample AE MAE, with PBE-vs-ref line."""
    with plt.rc_context(_STYLE):
        rxn_map = reaction_mae_by_arch_subset(reaction_rows)
        ae_map = ae_mae_by_arch_subset(insample_rows)
        archs = [a for a in ARCH_ORDER
                 if any(k[0] == a for k in rxn_map)
                 or any(k[0] == a for k in ae_map)]

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
        ax.bar(xs - w, rxn_mean, w, label="held-out reaction MAE (mean)",
               color="#4f81bd", edgecolor="k", linewidth=0.3)
        ax.bar(xs, rxn_best, w, label="held-out reaction MAE (best subset)",
               color="#9dc3e6", edgecolor="k", linewidth=0.3)
        ax.bar(xs + w, ae_mean, w, label="in-sample AE MAE (mean)",
               color="#c0504d", edgecolor="k", linewidth=0.3)

        pbe_vs_ref = _mae([r["abs_error_pbe_kcalmol"] for r in reaction_rows])
        if pbe_vs_ref is not None:
            ax.axhline(pbe_vs_ref, ls="--", color="k", linewidth=1.2,
                       label=f"PBE-vs-benchmark MAE ({pbe_vs_ref:.1f})")
        ax.axhline(1.0, ls=":", color="green", linewidth=1.0,
                   label="chemical accuracy (1 kcal/mol)")

        ax.set_yscale("log")
        ax.set_xticks(xs)
        ax.set_xticklabels(archs, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("MAE (kcal/mol, log scale)")
        ax.set_title(f"Per-architecture error · {run_id}")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, axis="y", which="both", alpha=0.3)
        if note:
            fig.text(0.5, 0.028, note, ha="center", fontsize=6.5, color="#a33")
        fig.text(0.5, 0.006, provenance or _PROVENANCE_BASE, ha="center",
                 fontsize=6, color="#777777")
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Figure D (bonus) — MAE vs subset_size, one line per arch
# ---------------------------------------------------------------------------

def _mae_vs_subset_panel(ax, mae_map: Dict[Tuple[str, int], float],
                         archs: List[str], *, title: str) -> None:
    for a in archs:
        pts = sorted((ss, v) for (aa, ss), v in mae_map.items() if aa == a)
        if not pts:
            continue
        xx, yy = zip(*pts)
        ax.plot(xx, yy, marker="o", ms=4, linewidth=1.3, color=ARCH_COLOR[a],
                label=a)
    ax.set_yscale("log")
    ax.set_xlabel("training subset_size")
    ax.set_ylabel("MAE (kcal/mol, log)")
    ax.set_title(title, fontsize=10)
    ax.grid(True, which="both", alpha=0.3)


def plot_mae_vs_subset(reaction_rows: List[Dict[str, Any]],
                       insample_rows: List[Dict[str, Any]],
                       out_path: Path, run_id: str, note: str = "",
                       provenance: Optional[str] = None) -> Path:
    """Figure D — learning curves: MAE vs subset_size, one line per arch."""
    with plt.rc_context(_STYLE):
        archs = [a for a in ARCH_ORDER
                 if a in _archs_present(reaction_rows)
                 or a in _archs_present(insample_rows)]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.4))
        _mae_vs_subset_panel(ax1, reaction_mae_by_arch_subset(reaction_rows),
                             archs, title="Held-out reaction-energy MAE")
        _mae_vs_subset_panel(ax2, ae_mae_by_arch_subset(insample_rows),
                             archs, title="In-sample atomization-energy MAE")
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
# Figure E — held-out atomization-energy parity (W4-11)
# ---------------------------------------------------------------------------
# W4-11 is the held-out set's atomization-energy benchmark: each W4-11 "reaction"
# is a molecule -> constituent-atoms atomization, so its ``de_nn``/``de_pbe``
# ARE predicted atomization energies and ``ref`` is the reference AE. This is
# the atomization-energy analog of the held-out reaction parity (plot_parity),
# and — unlike an in-sample training-fit plot — it shows generalization, with
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
    """Figure E — HELD-OUT atomization-energy parity (W4-11): predicted vs
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
        # PBE baseline points (same physical PBE for every arch — draw once).
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
        axa.set_title("(a) held-out W4-11 AE vs reference — by architecture")
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
        axb.set_title("(b) held-out W4-11 AE vs reference — by train-set size")

        handles = [Patch(facecolor=ARCH_COLOR[a], label=a) for a in archs]
        handles.append(plt.Line2D([], [], marker="x", ls="", color="0.4",
                                   label="PBE"))
        fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=7,
                   frameon=False, bbox_to_anchor=(0.5, 0.085))
        fig.suptitle(
            "Held-out atomization-energy parity (W4-11) — "
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
    handles = [Patch(facecolor=ARCH_COLOR[a], label=a) for a in archs]
    if pools:
        handles += [plt.Line2D([], [], marker=POOL_MARKER[p], ls="", color="0.3",
                                label=p.upper()) for p in pools]
    handles.append(plt.Line2D([], [], marker="x", ls="", color="0.5", label="PBE"))
    return handles


def _stamp_parity_footer(fig, *, run_id: str, title: str, note: str,
                         provenance: Optional[str], caveat: Optional[str]) -> None:
    fig.suptitle(f"{title}  ·  {run_id}", fontsize=11.5, y=0.997)
    if caveat:
        fig.text(0.5, 0.945, caveat, ha="center", fontsize=7.5, style="italic",
                 color="#444444")
    if note:
        fig.text(0.5, 0.032, note, ha="center", fontsize=5.6, color="#a33",
                 wrap=True)
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
                         caveat: Optional[str] = None) -> Path:
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
                   loc="lower center", ncol=8, fontsize=7, frameon=False,
                   bbox_to_anchor=(0.5, 0.052))
        _stamp_parity_footer(fig, run_id=run_id, note=note, provenance=provenance,
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
                             caveat: Optional[str] = None) -> Path:
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
                   loc="lower center", ncol=8, fontsize=7, frameon=False,
                   bbox_to_anchor=(0.5, 0.05))
        _stamp_parity_footer(fig, run_id=run_id, note=note, provenance=provenance,
                             caveat=caveat,
                             title="Reaction-energy parity -- pool x subset facets (arch=color)")
        fig.tight_layout(rect=(0, 0.085, 1, 0.915))
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return out_path


def plot_parity_arch_cols(rows: List[Dict[str, Any]], out_path: Path,
                          run_id: str, note: str = "",
                          provenance: Optional[str] = None,
                          caveat: Optional[str] = None) -> Path:
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
        _stamp_parity_footer(fig, run_id=run_id, note=note, provenance=provenance,
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
                                  caveat: Optional[str] = None) -> Path:
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
                            and r.get("arch") == a and r.get("pool") == pool]
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
                   loc="lower center", ncol=8, fontsize=7, frameon=False,
                   bbox_to_anchor=(0.5, 0.05))
        _stamp_parity_footer(fig, run_id=run_id, note=note, provenance=provenance,
                             caveat=caveat,
                             title="Reaction-energy parity + error bars -- 3x2 by subset (aggregate)")
        fig.tight_layout(rect=(0, 0.085, 1, 0.915))
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return out_path


def plot_parity_grid_by_subset(rows: List[Dict[str, Any]], out_path: Path,
                               run_id: str, note: str = "",
                               provenance: Optional[str] = None,
                               caveat: Optional[str] = None) -> Path:
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
                   loc="lower center", ncol=8, fontsize=7, frameon=False,
                   bbox_to_anchor=(0.5, 0.04))
        _stamp_parity_footer(fig, run_id=run_id, note=note, provenance=provenance,
                             caveat=caveat,
                             title="Reaction-energy parity -- 6x2 subset x pool, per-subset combined-MAE inset")
        fig.tight_layout(rect=(0, 0.065, 1, 0.93))
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return out_path


def build_parity_variants(run_dir: Path, outdir: Path) -> List[Path]:
    """Render all five parity-layout candidates into ``outdir`` for comparison."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = collect_holdout_reaction_rows(run_dir)
    run_id = run_dir.name
    note = coverage_note(run_dir)
    try:
        baseline = pbe_pool_baseline(run_dir)
    except Exception as exc:  # pool unavailable
        print(f"  (PBE baseline unavailable: {exc})")
        baseline = {"bh76": float("nan"), "w411": float("nan"),
                    "combined": float("nan")}
    prov = provenance_footer(baseline)
    caveat = nn_vs_pbe_caveat(rows, baseline)
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
                          provenance=prov, caveat=caveat))
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
    present here, so this is a LABELED reweighting, not a full-GMTKN55 WTMAD-2."""
    cells: Dict[Tuple[str, int], Dict[str, List[Tuple[float, float]]]] = {}
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
        cells.setdefault((a, s), {}).setdefault(pool, []).append((abs(e), abs(ref)))
    out: Dict[Tuple[str, int], float] = {}
    for cell, pools in cells.items():
        w = _wtmad2_over_pools(pools, scale)
        if w is not None:
            out[cell] = w
    return out


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
                "ref_density_method": r.get("ref_density_method"),
                "from_training_subset": r.get("from_training_subset"),
            })
    return rows


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


def plot_energy_wtmad_mae(rows: List[Dict[str, Any]], out_path: Path, run_id: str,
                          note: str = "", provenance: Optional[str] = None,
                          caveat: Optional[str] = None,
                          training_subsets: Optional[Dict[int, List[str]]] = None
                          ) -> Path:
    """Held-out energy: ONE bar per (arch, subset_size) cell -- combined
    reaction-energy MAE (panel a) and 2-subset WTMAD-2 (panel b) -- grouped by
    arch within each subset_size on the x-axis. NO error bars: each cell is a
    single model trained on a distinct subset and evaluated on a fixed held-out
    set, so a within-sample spread would be arbitrary and cross-subset
    aggregation is invalid (the six subset models per arch are not comparable).
    The subset trend is the x-axis. WTMAD-2 here = 2-subset, NOT full GMTKN55."""
    with plt.rc_context(_STYLE):
        archs = _archs_present(rows) or ["deep"]
        subsets = _present_subsets(rows) or [1]
        mae = reaction_mae_by_arch_subset(rows)
        wt = wtmad2_by_arch_subset(rows)
        pbe_mae = _mae([r["abs_error_pbe_kcalmol"] for r in _dedup_rows_by_name(rows)])
        pbe_wt = wtmad2_pbe_baseline(rows)
        has_ts = bool(training_subsets)
        fig, axes = plt.subplots(1, 2, figsize=(13, 6.4 if has_ts else 5),
                                 squeeze=False)
        bw = 0.8 / max(1, len(archs))

        def _grouped(ax, metric, pbe_line, title):
            for j, a in enumerate(archs):
                xs = [i + (j - (len(archs) - 1) / 2) * bw
                      for i in range(len(subsets))]
                hs = [metric.get((a, s), float("nan")) for s in subsets]
                ax.bar(xs, hs, width=bw, color=ARCH_COLOR[a], edgecolor="k",
                       linewidth=0.3, label=a)
            if _is_num(pbe_line):
                ax.axhline(pbe_line, ls="--", color="k", linewidth=1.0, label="PBE")
            ax.set_xticks(range(len(subsets)))
            ax.set_xticklabels(subsets)
            ax.set_xlabel("training subset_size", fontsize=8)
            ax.set_ylabel("kcal/mol", fontsize=8)
            ax.set_title(title, fontsize=9)
            ax.grid(True, axis="y", alpha=0.3)

        _grouped(axes[0][0], mae, pbe_mae,
                 "Held-out reaction-energy MAE (combined), per (arch, subset)")
        _grouped(axes[0][1], wt, pbe_wt,
                 "2-subset WTMAD-2 (BH76+W4-11), per (arch, subset)")
        handles, labels = axes[0][0].get_legend_handles_labels()
        if labels:
            fig.legend(handles, labels, loc="lower center", ncol=8, fontsize=7,
                       frameon=False, bbox_to_anchor=(0.5, 0.045))
        if has_ts:
            lines = ["Training subsets (held-in molecules; + element anchors):"]
            for ss in sorted(training_subsets):
                ms = training_subsets[ss]
                lines.append(f"  {ss}:  {', '.join(ms) if ms else '(atoms only)'}")
            fig.text(0.06, 0.265, "\n".join(lines), ha="left", va="top",
                     fontsize=6, family="monospace", color="#333333")
        _stamp_parity_footer(
            fig, run_id=run_id, note=note, provenance=provenance, caveat=caveat,
            title="Held-out energy: per-cell combined MAE + 2-subset WTMAD-2 (NOT full GMTKN55)")
        fig.tight_layout(rect=(0, 0.31 if has_ts else 0.10, 1, 0.93))
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return out_path


def plot_insample_density_ccsd(density_rows: List[Dict[str, Any]], out_path: Path,
                               run_id: str, note: str = "",
                               provenance: Optional[str] = None,
                               caveat: Optional[str] = None) -> Path:
    """In-sample density error vs CCSD (Dick-style diagnostic): (left) per-arch
    density RMSE vs subset_size with n annotated; (right) per-molecule strip
    (every point, since the trained-species set is tiny). Labeled IN-SAMPLE; no
    PBE density baseline exists."""
    with plt.rc_context(_STYLE):
        archs = _archs_present(density_rows) or ["deep"]
        arch_idx = {a: i for i, a in enumerate(archs)}
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), squeeze=False)
        axL = axes[0][0]
        for a in archs:
            by_s: Dict[int, List[float]] = {}
            for r in density_rows:
                if r.get("arch") == a and _is_num(r.get("density_rmse")):
                    by_s.setdefault(r["subset_size"], []).append(r["density_rmse"])
            pts = sorted((s, float(np.mean(v)), len(v)) for s, v in by_s.items())
            if pts:
                axL.plot([s for s, _, _ in pts], [m for _, m, _ in pts],
                         marker="o", ms=5, color=ARCH_COLOR[a], label=a)
                for s, m, n in pts:
                    axL.annotate(f"n={n}", (s, m), fontsize=5,
                                 color=ARCH_COLOR[a], xytext=(0, 4),
                                 textcoords="offset points")
        axL.set_yscale("log")
        axL.set_xlabel("training subset_size", fontsize=8)
        axL.set_ylabel("density RMSE vs CCSD (grid, weighted-mean)", fontsize=8)
        axL.set_title("In-sample density fit vs CCSD (per arch)", fontsize=9)
        if axL.get_legend_handles_labels()[1]:
            axL.legend(fontsize=6, ncol=2)
        axL.grid(True, which="both", alpha=0.3)

        axR = axes[0][1]
        mols = sorted({r["molecule"] for r in density_rows if r.get("molecule")})
        mol_x = {m: i for i, m in enumerate(mols)}
        noff = max(1, len(archs))
        for r in density_rows:
            if not _is_num(r.get("density_rmse")) or r.get("molecule") not in mol_x:
                continue
            jit = (arch_idx.get(r.get("arch"), 0) - (noff - 1) / 2) * 0.12
            axR.scatter(mol_x[r["molecule"]] + jit, r["density_rmse"], s=18,
                        alpha=0.75, color=ARCH_COLOR.get(r.get("arch"), "0.5"),
                        edgecolor="none")
        axR.set_yscale("log")
        axR.set_xticks(range(len(mols)))
        axR.set_xticklabels(mols, rotation=60, ha="right", fontsize=6)
        axR.set_ylabel("density RMSE vs CCSD", fontsize=8)
        axR.set_title(f"Per-molecule (every point; {len(mols)} trained species)",
                      fontsize=9)
        axR.grid(True, axis="y", which="both", alpha=0.3)

        arch_handles = [Patch(facecolor=ARCH_COLOR[a], label=a) for a in archs]
        fig.legend(handles=arch_handles, loc="lower center", ncol=8, fontsize=7,
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


def classify_failures(runs: List[Tuple[Path, str]], *, instab_factor: float = 5.0,
                      final_window: int = 50) -> List[Dict[str, Any]]:
    """Per (arch, subset_size, basis) held-out diagnosis across several runs:
    combined held-out MAE + BH76/W4-11 split, that cell's per-reaction PBE
    baseline, the final-window training loss, and a :func:`_classify_cell`
    mechanism label. Reuses :func:`collect_holdout_reaction_rows`,
    :func:`reaction_mae_by_arch_subset`, and :func:`collect_training_losses`."""
    cells: List[Dict[str, Any]] = []
    for run_dir, basis in runs:
        rows = collect_holdout_reaction_rows(run_dir)
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


def plot_failure_diagnostic(runs: List[Tuple[Path, str]], out_path: Path,
                            run_id: str, note: str = "",
                            provenance: Optional[str] = None) -> Path:
    """Explain WHY each network fails. Panel A: the DECOUPLING -- final-window
    training loss (x, log) vs held-out combined MAE (y); nearly all cells reach a
    low train loss yet scatter widely in held-out error (they overfit the tiny
    held-in subset), and only deep_attn-ss6 also has a high train loss (the lone
    genuine training failure). Panel B: the capacity ladder -- held-out MAE/PBE by
    arch family, split BH76 (barriers) vs W4-11 (atomization), showing extra
    descriptors + attention worsen overfitting and the damage lands on W4-11."""
    cells = classify_failures(runs)
    bases = list(dict.fromkeys(c["basis"] for c in cells))
    mk = ["o", "^", "s", "D"]
    marker_for = {b: mk[i % len(mk)] for i, b in enumerate(bases)}
    with plt.rc_context(_STYLE):
        fig = plt.figure(figsize=(14.0, 6.8))
        gs = fig.add_gridspec(1, 2, left=0.06, right=0.985, top=0.9, bottom=0.30,
                              wspace=0.22)
        axA = fig.add_subplot(gs[0, 0])  # axB is built by _broken_bar_panel below

        # --- Panel A: decoupling scatter ---
        for c in cells:
            if not (_is_num(c["final_loss"]) and _is_num(c["heldout_mae"])):
                continue
            axA.scatter(c["final_loss"], c["heldout_mae"],
                        color=_FAIL_COLORS[c["classification"]],
                        marker=marker_for.get(c["basis"], "o"), s=40,
                        edgecolor="k", linewidth=0.4, zorder=3)
        pbes = [c["pbe_mae"] for c in cells if _is_num(c["pbe_mae"])]
        if pbes:
            axA.axhspan(min(pbes), max(pbes), color="0.75", alpha=0.35, zorder=0)
            axA.axhline(float(np.mean(pbes)), ls="--", color="0.4", lw=1.0)
        med = cells[0].get("cohort_median_loss") if cells else None
        if _is_num(med):
            axA.axvline(5.0 * med, ls=":", color="#c0392b", lw=1.0)
        # label the worst few cells
        for c in sorted((c for c in cells if _is_num(c["heldout_mae"])),
                        key=lambda c: -c["heldout_mae"])[:4]:
            if _is_num(c["final_loss"]):
                axA.annotate(f"{c['arch']} ss{c['subset_size']}",
                             (c["final_loss"], c["heldout_mae"]), fontsize=6,
                             xytext=(4, 2), textcoords="offset points")
        axA.set_xscale("log")
        axA.set_xlabel("final-window training loss (mean of last 50 steps, log)",
                       fontsize=8)
        axA.set_ylabel("held-out combined reaction MAE (kcal/mol)", fontsize=8)
        axA.set_title("Training loss is decoupled from held-out error", fontsize=9)
        axA.grid(True, which="both", alpha=0.3)
        cls_handles = [Patch(facecolor=_FAIL_COLORS[k], edgecolor="k",
                             label=_FAIL_LABEL[k])
                       for k in ("pass", "generalization_gap", "late_instability")]
        cls_handles += [plt.Line2D([], [], ls="--", color="0.4",
                                   label="PBE baseline (band)"),
                        plt.Line2D([], [], ls=":", color="#c0392b",
                                   label="instability cut (5x median loss)")]
        if len(bases) > 1:
            cls_handles += [plt.Line2D([], [], ls="", marker=marker_for[b],
                                       color="0.3", label=str(b)) for b in bases]
        axA.legend(handles=cls_handles, fontsize=6.3, loc="upper left",
                   framealpha=0.7)

        # --- Panel B: ss-RESOLVED capacity-ladder bars (one bar per arch x ss) ---
        # NEVER averaged over subset_size -- at fixed (small) ss the capacity
        # ladder is clean (deep < attn < cusp < combined < combined_attn) and each
        # arch falls toward PBE as ss grows (overfitting relieved by data).
        present = {c["arch"] for c in cells}
        archs = [a for a in ARCH_ORDER if a in present]
        archs += sorted(present - set(archs))
        prim = _primary_basis(cells)  # densest run (svp); the sparse one is in trends
        pcells = [c for c in cells if c["basis"] == prim]
        ss_vals = sorted({c["subset_size"] for c in pcells})
        rmap = {(c["arch"], c["subset_size"]): c["heldout_mae"] / c["pbe_mae"]
                for c in pcells if _is_num(c["heldout_mae"])
                and _is_num(c["pbe_mae"]) and c["pbe_mae"] > 0}
        nss = max(1, len(ss_vals))
        bw = 0.82 / nss
        norm_ss = matplotlib.colors.Normalize(min(ss_vals), max(ss_vals)) \
            if ss_vals else None
        cmap_ss = plt.get_cmap("viridis")
        ss_colors = [cmap_ss(norm_ss(ss)) if norm_ss else "0.5" for ss in ss_vals]
        series = [(f"ss{ss}", [rmap.get((a, ss), float("nan")) for a in archs])
                  for ss in ss_vals]
        # Broken y-axis (reused, tested helper): the lone deep_attn-ss6 spike (5.4)
        # shows at its TRUE height in an upper band (break ~3.4->5.1) instead of
        # crushing the 0.6-3.1 bulk -- no cap, no floating label.
        axB = _broken_bar_panel(
            fig, gs[0, 1], series, archs, [],
            f"Capacity ladder per subset_size ({prim});  >1 = worse than PBE",
            "held-out MAE / PBE", ss_colors, bw)
        axB.axhline(1.0, ls="--", color="0.3", lw=1.0)  # PBE parity
        ss_handles = [Patch(facecolor=ss_colors[k], edgecolor="k", label=f"ss{ss}")
                      for k, ss in enumerate(ss_vals)]
        axB.legend(handles=ss_handles, fontsize=6.0, ncol=max(1, nss // 2),
                   title="subset", title_fontsize=6.0, loc="upper left",
                   framealpha=0.7)

        # --- classification key: every FAILING cell -> mechanism ---
        def _grp(label: str) -> str:
            items = sorted(f"{c['arch']} ss{c['subset_size']}"
                           + (f" ({c['basis']})" if len(bases) > 1 else "")
                           for c in cells if c["classification"] == label)
            return ", ".join(items) if items else "(none)"

        key = (f"Late training instability (final loss is an outlier; eval uses the "
               f"bad final checkpoint):  {_grp('late_instability')}.\n"
               f"Generalization gap (clean low train loss, but overfits the "
               f"{ '~3-11' }-molecule held-in subset; damage on W4-11 atomization):  "
               f"{_grp('generalization_gap')}.\n"
               f"Beats PBE (pass):  {_grp('pass')}.")
        fig.text(0.06, 0.2, key, ha="left", va="top", fontsize=6.6, family="serif",
                 wrap=True)
        _stamp_parity_footer(
            fig, run_id=run_id, note=note, provenance=provenance, caveat=None,
            title="Failure-mechanism diagnostic (held-out vs training loss)")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return out_path


def plot_capacity_trends(runs: List[Tuple[Path, str]], out_path: Path,
                         run_id: str, note: str = "",
                         provenance: Optional[str] = None) -> Path:
    """Secondary descriptive views of the same MAE/PBE structure: two diverging
    ratio heatmaps (BH76 barriers + W4-11 atomization, arch x ss, centered at PBE
    parity 1.0) showing the damage lands on W4-11; and a MAE/PBE-vs-subset_size
    line plot (one line per arch, basis = linestyle) making the capacity ordering
    and the fall-to-PBE-with-more-data trend explicit."""
    cells = classify_failures(runs)
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
                       cbar_label="MAE / PBE", center=1.0, subset_sizes=ss_axis)
        _heatmap_panel(axH2, _ratio_map("w411_mae", "w411_pbe", prim), archs,
                       title=f"W4-11 atomization  MAE/PBE ({prim})",
                       cbar_label="MAE / PBE", center=1.0, subset_sizes=ss_axis)
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
                                     provenance: Optional[str] = None) -> Path:
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
    """The three source-verified methods columns (placed under panels a/b/c).
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
        r"  $F_x{=}F_x^{PBE}/F_x^{LDA}{-}1$,  $F_c{=}F_c^{PBE}/F_c^{LDA}{-}1$.",
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
        r"  _dm $(x_6,x_7,x_8)$ from the 1-particle density matrix $D$ ($D'{=}D/2$",
        r"   RKS, $D_\sigma$ UKS):",
        r"   $x_6{=}\|D'SD'{-}D'\|_F/\mathrm{Tr}(D'S)$: idempotency, $=0$ EXACTLY for one",
        r"     Slater determinant ($PSP{=}P$ [10]).",
        r"   $x_7{=}-\!\sum p_i\ln p_i/\ln\max(n_{orb}^{eff},2)$, $p_i$=normalized occupations",
        r"     (eig $DS$ [9]): occupation-spread entropy, size-INTENSIVE $\in[0,1]$, used as a",
        r"     multireference indicator [11].  Nonzero for one determinant.",
        r"   $x_8{=}\|D_{off}\|_F/\mathrm{Tr}(D)$: relative off-diagonal weight of $D$.",
        r"  _combined: cusp & DM;   _notransform: log-transform off.",
    ]
    return [col1, col2, col3]


def _methods_references() -> List[str]:
    """Full-width numbered references key for the methods box (each equation in
    the columns cites [n]). Every entry verified real + accurate (consensus of
    multiple opus reviewers + two citation-verifiers)."""
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
        "[18] Chakravorty et al., PRA 47, 3649 (1993).   [19] Vaswani et al., NeurIPS 2017 (scaled-dot-product attention).",
    ]


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
                       fig_h: Optional[float] = None) -> int:
    """Place the three methods columns (mathtext) under panels a/b/c at ``xs``,
    each offset vertically by ``y_deltas`` (figure fraction; negative = lower).
    When ``reactions`` + ``fig_h`` are given, a FULL-WIDTH training-content footer
    (W4-11 atomizations + BH76 reactions) is placed below the columns. Returns the
    total effective line count (columns + footer) so a caller can size the figure."""
    cols = _methods_columns(subsets)
    for x, dy, col in zip(xs, y_deltas, cols):
        fig.text(x, y_top + dy, "\n".join(col), va="top", ha="left",
                 fontsize=fontsize, family="serif")
    max_col = max(len(c) for c in cols)
    refs = _methods_references()
    footer = _subset_reaction_lines(reactions) if reactions else []
    if fig_h:
        line_frac = fontsize * 1.58 / (72.0 * fig_h)
        # full-width references key, clear of the tallest column ...
        y = y_top - (max_col + 3.0) * line_frac
        fig.text(xs[0], y, "\n".join(refs), va="top", ha="left",
                 fontsize=fontsize - 0.5, family="serif")
        # ... then the training-content reaction footer below the references
        if footer:
            y -= (len(refs) + 2.0) * line_frac
            fig.text(xs[0], y, "\n".join(footer), va="top", ha="left",
                     fontsize=fontsize, family="serif")
    return max_col + len(refs) + (len(footer) + 6 if footer else 4)


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


_BASIS_COLORS = ("#4477aa", "#cc6677", "#228833", "#ccbb44")


def plot_basis_comparison(runs: List[Tuple[Path, str]], out_path: Path,
                          run_id: str, note: str = "",
                          provenance: Optional[str] = None) -> Path:
    """Cross-basis comparison over the UNION of (arch, subset) cells present in
    ANY run: (a) combined held-out reaction-energy MAE, (b) 2-subset WTMAD-2, (c)
    in-sample density RMSE vs CCSD -- grouped bars by basis. A basis's bar is
    simply absent for a cell it hasn't run yet (leaving room as later runs, e.g.
    DF, fill in) -- completed cells are NEVER dropped for lack of a counterpart.
    Per-basis PBE baselines are dashed lines on the energy panels; the held-out
    benchmark reference is basis-independent, so NN errors ARE comparable."""
    with plt.rc_context(_STYLE):
        data = []
        cellsets = []
        for rd, label in runs:
            rows = collect_holdout_reaction_rows(rd)
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
        cells = sorted(set.union(*cellsets)) if cellsets else []
        labels = [f"{a}/ss{s}" for a, s in cells]
        pw = max(6.0, 0.42 * max(1, len(cells)))
        nb = max(1, len(data))
        bw = 0.8 / nb
        # Size the figure to its content (inches): the methods band is placed
        # snug above the provenance so there is no trailing whitespace, and the
        # legend goes ABOVE the panels (clear of the rotated x-axis labels).
        subsets = training_subsets_by_size(runs[0][0]) if runs else {}
        reactions = training_reactions_by_size(runs[0][0]) if runs else {}
        FS = 6.2
        # height = tallest column + full-width references key + subset footer (+gaps)
        n_cols = max(len(c) for c in _methods_columns(subsets))
        n_refs = len(_methods_references())
        n_foot = (len(_subset_reaction_lines(reactions)) + 2) if reactions else 0
        n_meth = n_cols + n_refs + 6 + n_foot
        meth_h = n_meth * FS * 1.58 / 72.0 + 0.06   # methods text block (~1.2 linespacing)
        panels_h, xlabel_h = 3.5, 0.72              # panels + rotated cell labels
        legend_h, gap1, gap2 = 0.30, 0.06, 0.10     # legend band: methods | legend | labels
        top_pad, bot_pad = 0.68, 0.24               # suptitle + panel-title clearance ; provenance
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
        # leaves room for the old middle-column nudge).
        _methods_textblock(fig, subsets, y_top=_f(bot_pad + meth_h), fontsize=FS,
                           xs=(0.05, 0.37, 0.69), reactions=reactions, fig_h=fig_h)
        fig.suptitle(
            "Cross-basis comparison (union of arch x subset cells; bar absent "
            "where a basis hasn't run) -- NN bars vs benchmark, PBE dashed"
            f"  ·  {run_id}", fontsize=11, y=1.0 - _f(0.16))
        fig.text(0.5, _f(0.09), provenance or _PROVENANCE_BASE, ha="center",
                 fontsize=6, color="#777777")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return out_path


def build_basis_comparison_figures(run_dirs: List[Path], outdir: Path) -> List[Path]:
    """Render the cross-basis comparison for the given run dirs (each labeled by
    its basis+DF from resolved_config.yaml)."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    runs = [(Path(rd), run_basis_label(rd)) for rd in run_dirs]
    rid = " vs ".join(lbl for _, lbl in runs)
    return [plot_basis_comparison(runs, outdir / "basis_comparison.png", rid)]


def build_diagnostic_figures(run_dirs: List[Path], outdir: Path) -> List[Path]:
    """Render the CUMULATIVE (multi-basis) training-loss trajectories -- every
    trained cell from every run, basis by linestyle -- plus the failure-mechanism
    diagnostic that classifies and explains each failing cell."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    runs = [(Path(rd), run_basis_label(rd)) for rd in run_dirs]
    rid = " + ".join(lbl for _, lbl in runs)
    loss_rows = collect_training_losses_multi(runs)
    return [
        plot_training_losses(loss_rows, outdir / "diagnostic_training_losses.png",
                             rid, highlight=[("deep_attn", 6)]),
        plot_failure_diagnostic(runs, outdir / "diagnostic_failure_mechanisms.png",
                                rid),
        plot_capacity_trends(runs, outdir / "diagnostic_capacity_trends.png", rid),
    ]


def build_density_energy_figures(run_dir: Path, outdir: Path) -> List[Path]:
    """Render the held-out energy (MAE + 2-subset WTMAD-2) figure and the
    in-sample density-vs-CCSD diagnostic, kept SEPARATE."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rows = collect_holdout_reaction_rows(run_dir)
    drows = collect_insample_density_rows(run_dir)
    run_id = run_dir.name
    note = coverage_note(run_dir)
    try:
        baseline = pbe_pool_baseline(run_dir)
    except Exception as exc:
        print(f"  (PBE baseline unavailable: {exc})")
        baseline = {"bh76": float("nan"), "w411": float("nan"),
                    "combined": float("nan")}
    prov = provenance_footer(baseline)
    caveat = nn_vs_pbe_caveat(rows, baseline)
    dens_prov = ("In-sample density vs CCSD: grid weighted-mean RMSE/L1 on trained "
                 "species (atoms excluded).")
    tsubsets = training_subsets_by_size(run_dir)
    written = [
        plot_energy_wtmad_mae(rows, outdir / "ablation_energy_wtmad_mae.png",
                              run_id, note=note, provenance=prov, caveat=caveat,
                              training_subsets=tsubsets),
        plot_insample_density_ccsd(drows,
                                   outdir / "ablation_insample_density_ccsd.png",
                                   run_id, note=note, provenance=dens_prov),
    ]
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


def build_all(run_dir: Path, outdir: Path) -> List[Path]:
    """Collect once, render every figure. Returns the written PNG paths."""
    outdir.mkdir(parents=True, exist_ok=True)
    run_id = run_dir.name
    reaction_rows = collect_holdout_reaction_rows(run_dir)
    insample_rows = collect_insample_ae_rows(run_dir)
    n_trained = trained_spec_count(run_dir)
    n_total = len(ccp._read_manifest_cells(run_dir)) or len(ccp._spec_dirs(run_dir))
    n_holdout = len({r["idx"] for r in reaction_rows})
    note = coverage_note(run_dir)
    print(f"  coverage: {note}")

    # Live, non-hardcoded footers (degrade to "n/a" if the pool can't be loaded).
    try:
        baseline = pbe_pool_baseline(run_dir)
    except Exception as exc:  # pool unavailable (e.g. GMTKN55 clone absent)
        print(f"  (PBE baseline unavailable: {exc})")
        baseline = {"bh76": float("nan"), "w411": float("nan"),
                    "combined": float("nan")}
    prov = provenance_footer(baseline)
    caveat = nn_vs_pbe_caveat(reaction_rows, baseline)
    print(f"  PBE baseline (full pool): BH76 {_fmt_mae(baseline['bh76'])} / "
          f"W4-11 {_fmt_mae(baseline['w411'])} / "
          f"combined {_fmt_mae(baseline['combined'])}")

    written: List[Path] = []
    written.append(plot_parity(
        reaction_rows, outdir / "ablation_parity.png", run_id, note=note,
        provenance=prov, caveat=caveat))
    written.append(plot_arch_subset_heatmap(
        reaction_rows, insample_rows, outdir / "ablation_arch_subset_heatmap.png",
        run_id, n_trained=n_trained, n_total=n_total, n_holdout=n_holdout,
        note=note, provenance=prov))
    written.append(plot_mae_by_arch(
        reaction_rows, insample_rows, outdir / "ablation_mae_by_arch.png", run_id,
        note=note, provenance=prov))
    written.append(plot_mae_vs_subset(
        reaction_rows, insample_rows, outdir / "ablation_mae_vs_subset.png", run_id,
        note=note, provenance=prov))
    written.append(plot_ae_parity(
        reaction_rows, outdir / "ablation_ae_parity.png", run_id, note=note,
        provenance=prov))
    return written


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", default=None,
                   help="pulled run dir (default: latest under "
                        f"{_DEFAULT_LOCAL_ROOT / _DEFAULT_CATEGORY})")
    p.add_argument("--outdir", default=str(
        Path(__file__).resolve().parent / "figures_ablation_notransform"),
        help="output directory for PNGs")
    args = p.parse_args(argv)

    run_dir = _resolve_run_dir(args.run_dir)
    outdir = Path(args.outdir).expanduser().resolve()
    print(f"run_dir: {run_dir}")
    written = build_all(run_dir, outdir)
    for pth in written:
        print(f"  wrote {pth}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
