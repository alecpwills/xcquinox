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
from typing import Any, Dict, List, Optional, Tuple

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

# Provenance + caveat banners stamped on every figure.
_PROVENANCE = (
    "Geometry-fixed local re-eval (struc.xyz read as angstrom; 2026-05-31). "
    "Held-out: GMTKN55-BH76 + W4-11 reaction energies, kcal/mol. PBE: BH76 "
    "11.83 / W4-11 15.93 / combined 14.49."
)
_NNPBE_CAVEAT = (
    "Corrected geometry: PBE is physical (~12-16 kcal/mol) and NN no longer "
    "tracks it. At 1-18 training points the NN sits above PBE; only "
    "deep_combined_attn/subset-3 beats PBE on BH76 barriers (11.49)."
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
                note: str = "") -> Path:
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
        fig.text(0.5, 0.925, _NNPBE_CAVEAT, ha="center", fontsize=7.5,
                 style="italic", color="#444444")
        if note:
            fig.text(0.5, 0.05, note, ha="center", fontsize=6.5,
                     color="#a33", wrap=True)
        fig.text(0.5, 0.016, _PROVENANCE, ha="center", fontsize=6,
                 color="#777777")
        fig.tight_layout(rect=(0, 0.155, 1, 0.90))
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Figure B — arch × subset_size heatmaps
# ---------------------------------------------------------------------------

def _heatmap_panel(ax, mae_map: Dict[Tuple[str, int], float], archs: List[str],
                   *, title: str, cbar_label: str) -> None:
    n_a, n_s = len(archs), len(SUBSET_SIZES)
    grid = np.full((n_a, n_s), np.nan)
    for i, a in enumerate(archs):
        for j, ss in enumerate(SUBSET_SIZES):
            v = mae_map.get((a, ss))
            if v is not None and math.isfinite(v):
                grid[i, j] = v
    # log color scale (MAE spans decades); mask NaN as hatched "no data".
    finite = grid[np.isfinite(grid)]
    if finite.size:
        norm = matplotlib.colors.LogNorm(vmin=max(finite.min(), 1e-3),
                                         vmax=finite.max())
    else:
        norm = None
    cmap = plt.get_cmap("viridis").copy()
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
            else:
                ax.text(j, i, f"{grid[i, j]:.1f}", ha="center", va="center",
                        fontsize=5.5,
                        color="white" if grid[i, j] < (norm.vmax if norm else 1)
                        else "black")
    ax.set_xticks(range(n_s))
    ax.set_xticklabels(SUBSET_SIZES, fontsize=7)
    ax.set_yticks(range(n_a))
    ax.set_yticklabels(archs, fontsize=7)
    ax.set_xlabel("training subset_size")
    ax.set_title(title, fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=cbar_label)


def plot_arch_subset_heatmap(reaction_rows: List[Dict[str, Any]],
                             insample_rows: List[Dict[str, Any]],
                             out_path: Path, run_id: str, *,
                             n_trained: int, n_total: int,
                             n_holdout: int, note: str = "") -> Path:
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
                 + _PROVENANCE, ha="center", fontsize=6.5, color="#777777")
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
                     out_path: Path, run_id: str, note: str = "") -> Path:
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
        fig.text(0.5, 0.006, _PROVENANCE, ha="center", fontsize=6,
                 color="#777777")
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
                       out_path: Path, run_id: str, note: str = "") -> Path:
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
        fig.text(0.5, 0.008, _PROVENANCE, ha="center", fontsize=6,
                 color="#777777")
        fig.tight_layout(rect=(0, 0.06, 1, 0.95))
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return out_path


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

    written: List[Path] = []
    written.append(plot_parity(
        reaction_rows, outdir / "ablation_parity.png", run_id, note=note))
    written.append(plot_arch_subset_heatmap(
        reaction_rows, insample_rows, outdir / "ablation_arch_subset_heatmap.png",
        run_id, n_trained=n_trained, n_total=n_total, n_holdout=n_holdout,
        note=note))
    written.append(plot_mae_by_arch(
        reaction_rows, insample_rows, outdir / "ablation_mae_by_arch.png", run_id,
        note=note))
    written.append(plot_mae_vs_subset(
        reaction_rows, insample_rows, outdir / "ablation_mae_vs_subset.png", run_id,
        note=note))
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
