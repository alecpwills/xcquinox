#!/usr/bin/env python
"""Training-subset descriptor *completeness* vs held-out accuracy -- over the
FULL per-architecture descriptor set the network actually consumes.

The held-out reaction-energy MAE is non-monotonic in ``subset_size`` because
the JSD subsets are NOT nested -- each size trains on a differently-composed
set, so what matters is *which* region of the functional's INPUT (descriptor)
space the subset covers, not the point count. This module makes that explicit
by binning every descriptor dimension the network sees and correlating coverage
to accuracy.

Descriptor set per spec (exactly the network inputs):
  * ``s``   = |grad rho| / (2 (3 pi^2)^{1/3} rho^{4/3})   (reduced gradient; xnet+cnet)
  * ``r_s`` = (3 / (4 pi rho))^{1/3}                       (Wigner-Seitz; cnet)
  * the arch's EXTRA descriptors, per grid point:
        - ``CuspDescriptor_{0,1}``        (cusp archs)         -- spatial
        - ``DMStatisticsDescriptor_{0,1,2}`` (dm archs)        -- per-molecule, tiled
    (``deep``/``deep_attn``/``notransform`` have none -> set = {s, r_s}).

All dimensions are sampled per grid point from one cached PBE precompute (with
the cusp+dm superset, so every column is available and selected per arch), then
grid-weight-histogrammed. ``s``/``r_s`` use fixed chemically-meaningful supports;
the cusp/dm dimensions use adaptive supports from the held-out range.

Completeness ``C = <sum_bins min(P_train, P_held)>_{dims} in [0,1]`` is the mean
histogram-intersection coverage of the held-out descriptor distribution by the
training subset, averaged over THAT arch's dimensions (1 = fully spans them).

Figures:
  * ``ablation_descriptor_completeness_vs_mae.png`` -- C (x) vs held-out NN MAE
    (y), one point per spec, colored by arch, Spearman rho. "Does coverage
    predict accuracy?"
  * ``ablation_descriptor_histograms.png`` -- one panel per descriptor dimension;
    per-subset training distributions overlaid on the shaded held-out reference.

Heavy step (PBE density precompute on ~200 unique molecules; cached, no SCF)
runs once in the driver; the histogram/coverage math is pure and unit-tested.

Usage:
    python notebooks/analysis/subset_descriptor_coverage.py \
        [--run-dir <run dir>] [--outdir <dir>]
"""
from __future__ import annotations

import argparse
import importlib.util
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_SIB = Path(__file__).resolve().parent / "make_ablation_arch_figure.py"
_sib_spec = importlib.util.spec_from_file_location("make_ablation_arch_figure", _SIB)
sib = importlib.util.module_from_spec(_sib_spec)
sys.modules["make_ablation_arch_figure"] = sib
_sib_spec.loader.exec_module(sib)

ARCH_ORDER = sib.ARCH_ORDER
ARCH_COLOR = sib.ARCH_COLOR

#: Fixed supports for the universal GGA descriptors (chemically-relevant).
FIXED_EDGES: Dict[str, np.ndarray] = {
    "s": np.linspace(0.0, 3.0, 31),
    "rs": np.linspace(0.0, 6.0, 31),
}
_N_ADAPTIVE_BINS = 30
_RHO_FLOOR = 1e-10


# ---------------------------------------------------------------------------
# Pure descriptor + histogram math (unit-tested, no compute deps)
# ---------------------------------------------------------------------------

def point_s_rs(rho: np.ndarray, sigma: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Per-grid-point ``(s, r_s)`` from total density ``rho`` and ``sigma =
    |grad rho|^2``. rho below a floor -> NaN (dropped by the histogram)."""
    rho = np.asarray(rho, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    rho_safe = np.where(rho > _RHO_FLOOR, rho, np.nan)
    k_f = (3.0 * np.pi ** 2 * rho_safe) ** (1.0 / 3.0)
    s = np.sqrt(np.clip(sigma, 0.0, None)) / (2.0 * k_f * rho_safe)
    r_s = (3.0 / (4.0 * np.pi * rho_safe)) ** (1.0 / 3.0)
    return s, r_s


def weighted_hist(values: np.ndarray, weights: np.ndarray,
                  edges: np.ndarray) -> np.ndarray:
    """Grid-weighted, area-normalized histogram (discrete PDF over ``edges``).
    Non-finite values are ignored; zeros if no finite weighted samples."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    m = np.isfinite(values) & np.isfinite(weights)
    if not m.any():
        return np.zeros(len(edges) - 1)
    h, _ = np.histogram(values[m], bins=edges, weights=weights[m])
    total = h.sum()
    return h / total if total > 0 else h


def histogram_intersection(p: np.ndarray, q: np.ndarray) -> float:
    """``sum min(p_i, q_i)`` for two normalized histograms (overlap fraction).
    1.0 = identical, 0.0 = disjoint."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    if p.sum() <= 0 or q.sum() <= 0:
        return 0.0
    return float(np.minimum(p, q).sum())


def completeness(train_hists: Dict[str, np.ndarray],
                 held_hists: Dict[str, np.ndarray],
                 dims: Optional[Sequence[str]] = None) -> float:
    """Mean histogram-intersection coverage over ``dims`` (default: the dims
    present in both). This is averaged over THAT arch's descriptor set."""
    keys = list(dims) if dims is not None else [
        k for k in train_hists if k in held_hists]
    keys = [k for k in keys if k in train_hists and k in held_hists]
    if not keys:
        return 0.0
    return float(np.mean([histogram_intersection(train_hists[k], held_hists[k])
                          for k in keys]))


def adaptive_edges(values: np.ndarray, n_bins: int = _N_ADAPTIVE_BINS,
                   q: Tuple[float, float] = (0.5, 99.5)) -> np.ndarray:
    """Histogram edges spanning robust percentiles of ``values`` (for the
    cusp/dm dimensions whose ranges are not known a priori). Falls back to a
    unit interval when degenerate."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return np.linspace(0.0, 1.0, n_bins + 1)
    lo, hi = np.percentile(v, q)
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        lo, hi = float(v.min()), float(v.max())
        if hi <= lo:
            hi = lo + 1.0
    return np.linspace(lo, hi, n_bins + 1)


def hist_pool_from_values(values_by_mol: Sequence[np.ndarray],
                          weights_by_mol: Sequence[np.ndarray],
                          edges: np.ndarray) -> np.ndarray:
    """Pool per-molecule grid-weighted histograms (equal weight per molecule)
    into one normalized distribution over ``edges``."""
    acc = np.zeros(len(edges) - 1)
    for vals, w in zip(values_by_mol, weights_by_mol):
        h = weighted_hist(vals, w, edges)
        if h.sum() > 0:
            acc += h
    return acc / acc.sum() if acc.sum() > 0 else acc


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rank correlation (no scipy); average ranks for ties."""
    def _rank(a):
        order = np.argsort(a, kind="mergesort")
        ranks = np.empty(len(a), dtype=float)
        ranks[order] = np.arange(len(a), dtype=float)
        _, inv, counts = np.unique(a, return_inverse=True, return_counts=True)
        sums = np.zeros(len(counts))
        np.add.at(sums, inv, ranks)
        return (sums / counts)[inv]
    rx, ry = _rank(np.asarray(x, float)), _rank(np.asarray(y, float))
    rx -= rx.mean(); ry -= ry.mean()
    denom = math.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else 0.0


# ---------------------------------------------------------------------------
# Heavy: full per-molecule descriptor grids (cached PBE precompute)
# ---------------------------------------------------------------------------

def _superset_descriptors():
    """The cusp+dm descriptor instances (from ``deep_combined``), so one
    precompute yields every extra column; per-arch subsets select from it.
    Assumes a single descriptor configuration across archs (true for this
    sweep: all deep_* use dm_entropy_intensive=True)."""
    from xcquinox.alec.config import get_architecture
    return tuple(get_architecture("deep_combined").materialize_descriptors())


def superset_dim_names() -> List[str]:
    """``['s','rs', <each extra column name>]`` for the superset."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import extract_subset_descriptors as esd
    return ["s", "rs"] + list(esd.feature_column_names(_superset_descriptors()))


def arch_dim_names(arch) -> List[str]:
    """The descriptor dimensions one arch's network actually consumes:
    ``s``, ``r_s``, plus the arch's own extra descriptor columns."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import extract_subset_descriptors as esd
    try:
        extras = list(esd.feature_column_names(
            tuple(arch.materialize_descriptors())))
    except AttributeError:
        extras = []
    return ["s", "rs"] + extras


def molecule_descriptor_grids(mol_spec, cache: Dict[str, Dict[str, np.ndarray]]):
    """Cached ``{dim_name: per-grid values, '_w': grid_weights}`` for one
    MoleculeSpec, over the full superset (s, r_s, cusp, dm). Heavy (pyscf)."""
    name = getattr(mol_spec, "name", None) or repr(mol_spec)
    if name in cache:
        return cache[name]
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import extract_subset_descriptors as esd
    import xcquinox.alec as alec
    descriptors = _superset_descriptors()
    local = esd._strip_external_data_path(mol_spec)
    md = alec.precompute_fixed_density_data(local, descriptors=descriptors)
    rho = np.asarray(md["rho_grid"], dtype=float)
    sigma = np.asarray(md["sigma_grid"], dtype=float)
    w = np.asarray(md["grid_weights"], dtype=float)
    s, r_s = point_s_rs(rho, sigma)
    out: Dict[str, np.ndarray] = {"s": s, "rs": r_s, "_w": w}
    feats = np.asarray(np.concatenate(
        [np.asarray(d.compute(md)) for d in descriptors], axis=1), dtype=float)
    for j, col in enumerate(esd.feature_column_names(descriptors)):
        out[col] = feats[:, j]
    cache[name] = out
    return out


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_completeness_vs_mae(rows: List[Dict[str, Any]], out_path: Path,
                             run_id: str) -> Path:
    """Scatter: full-descriptor-set completeness (x) vs held-out NN MAE (y),
    one point per spec, colored by arch, Spearman rho in the title."""
    with plt.rc_context(sib._STYLE):
        fig, ax = plt.subplots(figsize=(8.5, 6))
        archs = [a for a in ARCH_ORDER if any(r.get("arch") == a for r in rows)]
        for a in archs:
            pts = [(r["completeness"], r["mae"]) for r in rows
                   if r.get("arch") == a and sib._is_num(r.get("completeness"))
                   and sib._is_num(r.get("mae"))]
            if not pts:
                continue
            xs, ys = zip(*pts)
            ax.scatter(xs, ys, s=40, color=ARCH_COLOR[a], edgecolor="k",
                       linewidth=0.3, label=a, alpha=0.85)
        allpts = [(r["completeness"], r["mae"]) for r in rows
                  if sib._is_num(r.get("completeness")) and sib._is_num(r.get("mae"))]
        rho_txt = "n<3 (correlation deferred)"
        if len(allpts) >= 3:
            xs, ys = map(np.asarray, zip(*allpts))
            rho_txt = f"Spearman rho = {_spearman(xs, ys):+.2f}  (n={len(allpts)})"
        pbe = next((r.get("pbe_mae") for r in rows if r.get("pbe_mae")), None)
        if pbe:
            ax.axhline(pbe, ls="--", color="k", linewidth=1.2,
                       label=f"PBE baseline ({pbe:.1f})")
        ax.set_xlabel("training-subset descriptor completeness C "
                      r"(mean $\sum\min(P_\mathrm{train},P_\mathrm{held})$ over "
                      "the arch's descriptors)")
        ax.set_ylabel("held-out reaction-energy NN MAE (kcal/mol)")
        ax.set_title(f"Descriptor completeness vs accuracy · {run_id}\n{rho_txt}",
                     fontsize=10)
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        fig.text(0.5, 0.005,
                 "C = histogram-intersection coverage of the held-out descriptor "
                 "distribution by the training subset, over s, r_s + the arch's "
                 "cusp/dm inputs. Hypothesis: higher C -> lower MAE.",
                 ha="center", fontsize=6.5, color="#777777")
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return out_path


def plot_descriptor_histograms(subset_hists: Dict[int, Dict[str, np.ndarray]],
                               held_hists: Dict[str, np.ndarray],
                               edges_by_dim: Dict[str, np.ndarray],
                               out_path: Path, run_id: str) -> Path:
    """One panel per descriptor dimension; per-subset training distributions
    overlaid on the shaded held-out reference. ``subset_hists``: {ss: {dim: h}}."""
    dims = (["s", "rs"]
            + sorted(k for k in held_hists if k not in ("s", "rs")))
    dims = [d for d in dims if d in held_hists]
    with plt.rc_context(sib._STYLE):
        ncol = min(4, max(1, len(dims)))
        nrow = max(1, math.ceil(len(dims) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 2.9 * nrow),
                                 squeeze=False)
        sizes = sorted(subset_hists)
        cmap = plt.get_cmap("viridis")
        norm = matplotlib.colors.Normalize(
            vmin=min(sizes) if sizes else 0, vmax=max(sizes) if sizes else 1)
        for k, dim in enumerate(dims):
            ax = axes[k // ncol][k % ncol]
            edges = edges_by_dim.get(dim, FIXED_EDGES.get(dim))
            if edges is None:
                continue
            centers = 0.5 * (edges[:-1] + edges[1:])
            if dim in held_hists:
                ax.fill_between(centers, held_hists[dim], step="mid",
                                color="0.7", alpha=0.6, label="held-out")
            for ss in sizes:
                h = subset_hists[ss].get(dim)
                if h is not None and h.sum() > 0:
                    ax.plot(centers, h, color=cmap(norm(ss)), linewidth=1.1,
                            label=f"subset {ss}")
            ax.set_title(dim, fontsize=8)
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.3)
        # Blank any unused axes.
        for k in range(len(dims), nrow * ncol):
            axes[k // ncol][k % ncol].axis("off")
        axes[0][0].legend(fontsize=5, ncol=2)
        fig.suptitle("Training-subset vs held-out descriptor distributions · "
                     f"{run_id}", fontsize=11)
        fig.supylabel("grid-weighted density (norm.)", fontsize=8)
        fig.tight_layout(rect=(0.02, 0, 1, 0.95))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def build(run_dir: Path, outdir: Path) -> List[Path]:
    """Full pipeline: full-descriptor held-out reference + per-subset coverage,
    join to corrected MAE, render both figures. Heavy (cached PBE precompute)."""
    from xcquinox.alec.eval_holdout import load_training_spec
    from xcquinox.alec.full_benchmark_pools import load_full_held_out_pools

    cache: Dict[str, Dict[str, np.ndarray]] = {}
    run_id = run_dir.name
    all_dims = superset_dim_names()

    # ---- Held-out reference: raw per-molecule descriptor grids -------------
    held_specs, _ = load_full_held_out_pools(basis="def2-svp", grid_level=1)
    print(f"[coverage] held-out reference: {len(held_specs)} species "
          f"(dims: {all_dims}) ...", flush=True)
    held_grids: List[Dict[str, np.ndarray]] = []
    for i, (nm, ms) in enumerate(held_specs.items(), 1):
        try:
            held_grids.append(molecule_descriptor_grids(ms, cache))
        except Exception as exc:  # noqa: BLE001
            print(f"  [warn] held-out {nm}: {type(exc).__name__}: {exc}", flush=True)
        if i % 25 == 0:
            print(f"  ... {i}/{len(held_specs)}", flush=True)

    # Per-dim edges: fixed for s/rs, adaptive (held-out range) for extras.
    edges_by_dim: Dict[str, np.ndarray] = {}
    for dim in all_dims:
        if dim in FIXED_EDGES:
            edges_by_dim[dim] = FIXED_EDGES[dim]
        else:
            pooled = np.concatenate([g[dim] for g in held_grids if dim in g]) \
                if held_grids else np.array([])
            edges_by_dim[dim] = adaptive_edges(pooled)
    held_hists = {dim: hist_pool_from_values(
        [g[dim] for g in held_grids if dim in g],
        [g["_w"] for g in held_grids if dim in g], edges_by_dim[dim])
        for dim in all_dims}

    # ---- Per spec: MAE + arch dims + training-subset coverage --------------
    reaction_rows = sib.collect_holdout_reaction_rows(run_dir)
    by_idx: Dict[int, List[float]] = {}
    arch_sub: Dict[int, Tuple[Optional[str], Optional[int]]] = {}
    for r in reaction_rows:
        if sib._is_num(r.get("abs_error_nn_kcalmol")):
            by_idx.setdefault(r["idx"], []).append(r["abs_error_nn_kcalmol"])
            arch_sub[r["idx"]] = (r.get("arch"), r.get("subset_size"))
    pbe_mae = sib._mae([r["abs_error_pbe_kcalmol"] for r in reaction_rows])

    rows: List[Dict[str, Any]] = []
    subset_hists: Dict[int, Dict[str, np.ndarray]] = {}
    for idx, errs in sorted(by_idx.items()):
        spec_path = run_dir / "specs" / f"spec_{idx:04d}.spec"
        if not spec_path.is_file():
            continue
        ts = load_training_spec(spec_path)
        dims = arch_dim_names(ts.arch)
        grids = []
        for mol in getattr(ts, "molecules", ()):
            try:
                grids.append(molecule_descriptor_grids(mol, cache))
            except Exception as exc:  # noqa: BLE001
                print(f"  [warn] train mol spec {idx}: {exc}", flush=True)
        th = {dim: hist_pool_from_values(
            [g[dim] for g in grids if dim in g],
            [g["_w"] for g in grids if dim in g], edges_by_dim[dim])
            for dim in dims}
        c = completeness(th, held_hists, dims=dims)
        arch, ss = arch_sub.get(idx, (None, None))
        rows.append({"idx": idx, "arch": arch, "subset_size": ss,
                     "completeness": c, "mae": float(np.mean(np.abs(errs))),
                     "pbe_mae": pbe_mae, "dims": dims})
        if ss is not None:
            subset_hists[ss] = th
        print(f"  spec {idx} ({arch}, ss={ss}): C={c:.3f} over {len(dims)} dims  "
              f"MAE={np.mean(np.abs(errs)):.2f}", flush=True)

    written = [
        plot_completeness_vs_mae(
            rows, outdir / "ablation_descriptor_completeness_vs_mae.png", run_id),
        plot_descriptor_histograms(
            subset_hists, held_hists, edges_by_dim,
            outdir / "ablation_descriptor_histograms.png", run_id),
    ]
    return written


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", default=None)
    p.add_argument("--outdir", default=str(
        Path(__file__).resolve().parent / "figures_ablation_notransform"))
    args = p.parse_args(argv)
    run_dir = sib._resolve_run_dir(args.run_dir)
    outdir = Path(args.outdir).expanduser().resolve()
    print(f"run_dir: {run_dir}")
    for pth in build(run_dir, outdir):
        print(f"  wrote {pth}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
