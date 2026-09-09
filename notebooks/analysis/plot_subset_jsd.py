"""Training-subset selection record: the Jensen-Shannon divergence of every chosen
subset against the full DFS pool, the descriptor marginals it was measured on, and
a table of the subsets.

Inputs are the selection ledger (``subset_index_log.json``), the cached
per-species descriptors (``subset_descriptors/<name>_c<charge>_s<spin>_<formula>.npz``)
and the cached reference histogram (``dfs_pool_full_hist/reference.npz``). Every
divergence is recomputed from the caches with the package's own selection
primitives and must agree with the ledger to 1e-9; a disagreement is an error,
not a footnote.

Outputs, in ``--outdir``: ``subset_jsd_vs_full.png`` (panel a: the divergence
against the subset size, total and per marginal; panels b-d: the three reference
marginals with the chosen subsets overlaid) and ``subset_table.csv`` (one row per
subset size with the members, the species union and the divergence).

Usage::

    python notebooks/analysis/plot_subset_jsd.py \\
        --ledger notebooks/checkpoints_step7/alpha_on/subset_index_log.json \\
        --reference notebooks/checkpoints_step7/alpha_on/dfs_pool_full_hist/reference.npz \\
        --descriptors notebooks/checkpoints_step7/subset_descriptors \\
        --outdir notebooks/analysis/figures_dfs_step7_v7_subsets
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import textwrap
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")  # headless-safe; must precede pyplot import
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, MaxNLocator  # noqa: E402
import numpy as np  # noqa: E402

from xcquinox.alec import subset_selection as ss  # noqa: E402
from xcquinox.alec.training_points import build_dfs_pool_points  # noqa: E402

_REPO = Path(__file__).resolve().parents[2]
_CKPT = _REPO / "notebooks" / "checkpoints_step7"
DEFAULT_LEDGER = _CKPT / "alpha_on" / "subset_index_log.json"
DEFAULT_REFERENCE = _CKPT / "alpha_on" / "dfs_pool_full_hist" / "reference.npz"
DEFAULT_DESCRIPTORS = _CKPT / "subset_descriptors"
DEFAULT_OUTDIR = _REPO / "notebooks" / "analysis" / "figures_dfs_step7_v7_subsets"

_KEYS: Tuple[str, str, str] = ("rho_third", "s", "alpha")
_REF_NPZ = {"rho_third": ("h_ref_rho", "e_rho"), "s": ("h_ref_s", "e_s"),
            "alpha": ("h_ref_alpha", "e_alpha")}
_LABELS = {"rho_third": r"$\rho^{1/3}$", "s": r"$s$", "alpha": r"$\alpha$"}
_PART_COLS = {"rho_third": "jsd_rho", "s": "jsd_s", "alpha": "jsd_alpha"}
LEDGER_TOL = 1e-9
_TABLE_COLUMNS = ("r", "jsd_ledger", "jsd", "jsd_rho", "jsd_s", "jsd_alpha",
                  "point_names", "point_kinds", "species", "n_ae", "n_bh76",
                  "n_ip13", "chosen_indices")
_LIST_COLUMNS = ("point_names", "point_kinds", "species", "chosen_indices")


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------

def load_ledger(path, metric: str = "jsd") -> Dict[int, dict]:
    """The ledger entries of one metric keyed by the subset size r."""
    with open(path) as fh:
        raw = json.load(fh)
    out: Dict[int, dict] = {}
    prefix = f"{metric}/"
    for key, entry in raw.items():
        if not key.startswith(prefix):
            continue
        out[int(key[len(prefix):])] = entry
    if not out:
        raise ValueError(f"load_ledger: no '{prefix}<r>' entries in {path}")
    return out


def load_reference(path) -> Tuple[dict, dict]:
    """The cached reference histograms and their edges under the package's keys."""
    with np.load(path) as z:
        h_ref = {k: np.asarray(z[hk]) for k, (hk, _) in _REF_NPZ.items()}
        edges = {k: np.asarray(z[ek]) for k, (_, ek) in _REF_NPZ.items()}
    return h_ref, edges


def species_key(atoms) -> Tuple[str, int, int]:
    """(name, charge, spin) as the selection code keys a species."""
    name = (atoms.info.get("name") or atoms.info.get("dfs_hill")
            or atoms.get_chemical_formula())
    return str(name), int(atoms.info.get("charge", 0)), int(atoms.info.get("spin", 0))


def cache_prefix(key: Tuple[str, int, int]) -> str:
    """The stable part of a species' cache file name.

    The file is ``<prefix>_<formula>.npz``; the formula is not the name for the
    cations (``Liplus_c1_s0_Li.npz``), so only the prefix identifies the file.
    """
    name, charge, spin = key
    return f"{name.replace('+', 'plus').replace('/', '_')}_c{charge}_s{spin}"


def pool_blocks(cache_dir) -> Tuple[list, List[dict]]:
    """The pool points and, per point, the concatenation of its species' caches."""
    cache_dir = Path(cache_dir)
    points = build_dfs_pool_points()
    species: Dict[Tuple[str, int, int], dict] = {}
    for tp in points:
        for at in tp.species:
            key = species_key(at)
            if key in species:
                continue
            prefix = cache_prefix(key)
            hits = sorted(cache_dir.glob(f"{prefix}_*.npz"))
            if not hits:
                raise FileNotFoundError(
                    f"pool_blocks: no descriptor cache for species {key} "
                    f"(looked for {prefix}_*.npz in {cache_dir})")
            if len(hits) > 1:
                raise ValueError(
                    f"pool_blocks: {len(hits)} descriptor caches match species {key} "
                    f"({prefix}_*.npz in {cache_dir}): {[h.name for h in hits]}")
            with np.load(hits[0]) as d:
                arrs = {k: np.asarray(d[k]) for k in _KEYS}
                arrs["weights"] = (np.asarray(d["weights"]) if "weights" in d.files
                                   else np.ones_like(arrs["rho_third"]))
            species[key] = arrs
    blocks = ss.concatenate_point_descriptors(points, species)
    return points, blocks


# ---------------------------------------------------------------------------
# the metric
# ---------------------------------------------------------------------------

def _concat_blocks(blocks: Sequence[dict], indices: Sequence[int]) -> dict:
    sel = [blocks[i] for i in indices]
    out = {k: np.concatenate([b[k] for b in sel]) for k in _KEYS}
    out["weights"] = np.concatenate(
        [b.get("weights", np.ones_like(b["rho_third"])) for b in sel])
    return out


def subset_jsd(h_ref: dict, edges: dict, blocks: Sequence[dict],
               indices: Sequence[int]) -> Tuple[float, Dict[str, float]]:
    """(total, per-marginal parts) of a subset against the reference.

    The subset's concatenated block is binned on the REFERENCE edges; the total
    is the package's ``metric_jsd`` (equal marginal weights) and each part is
    ``0.5 [KL(P||M) + KL(Q||M)]`` of one PMF-normalized marginal, so the three
    parts sum to the total. A subset with no in-range mass in a marginal gives a
    non-finite total (``metric_jsd`` returns inf on a zero histogram and NaN on the
    all-NaN one ``density=True`` produces), which ``subset_rows`` refuses.
    """
    h_cand = ss._bin_with_edges(_concat_blocks(blocks, indices), edges)
    total = ss.metric_jsd(h_ref, h_cand)
    parts: Dict[str, float] = {}
    for k in _KEYS:
        p = ss._to_pmf(h_ref[k])
        q = ss._to_pmf(h_cand[k])
        m = 0.5 * (p + q)
        parts[k] = float(0.5 * (ss._kl(p, m) + ss._kl(q, m)))
    return float(total), parts


def subset_rows(ledger: Dict[int, dict], points, blocks: Sequence[dict],
                h_ref: dict, edges: dict) -> List[dict]:
    """One row per subset size, the ledger value checked against the recomputation."""
    rows: List[dict] = []
    for r in sorted(ledger):
        entry = ledger[r]
        indices = [int(i) for i in entry["chosen_indices"]]
        total, parts = subset_jsd(h_ref, edges, blocks, indices)
        ledger_value = float(entry["metric_value"])
        # written as a negated <= so that a NaN recomputation is refused too
        if not (np.isfinite(total) and abs(total - ledger_value) <= LEDGER_TOL):
            raise ValueError(
                f"subset_rows: the recomputed divergence for r = {r} is {total!r}, "
                f"the ledger holds {ledger_value!r} (tolerance {LEDGER_TOL:g})")
        names = [points[i].name for i in indices]
        kinds = [points[i].kind for i in indices]
        for field, mine in (("point_names", names), ("point_kinds", kinds)):
            if field in entry and list(entry[field]) != mine:
                raise ValueError(
                    f"subset_rows: the ledger's {field} for r = {r} are {list(entry[field])!r}, "
                    f"the pool gives {mine!r}")
        species: List[str] = []
        for i in indices:
            for at in points[i].species:
                name = species_key(at)[0]
                if name not in species:
                    species.append(name)
        rows.append({
            "r": int(r),
            "jsd_ledger": ledger_value,
            "jsd": total,
            "jsd_rho": parts["rho_third"],
            "jsd_s": parts["s"],
            "jsd_alpha": parts["alpha"],
            "point_names": names,
            "point_kinds": kinds,
            "species": species,
            "n_ae": kinds.count("ae"),
            "n_bh76": kinds.count("bh76"),
            "n_ip13": kinds.count("ip13"),
            "chosen_indices": indices,
        })
    return rows


def write_subset_table(rows: Sequence[dict], path) -> None:
    """The rows as CSV; list-valued columns joined by ';'."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(_TABLE_COLUMNS)
        for row in rows:
            writer.writerow([
                ";".join(str(v) for v in row[c]) if c in _LIST_COLUMNS else row[c]
                for c in _TABLE_COLUMNS])


# ---------------------------------------------------------------------------
# the figure
# ---------------------------------------------------------------------------

def _first_bin_share(h: np.ndarray) -> float:
    return float(ss._to_pmf(np.asarray(h))[0]) * 100.0


def _positive_or_nan(p: np.ndarray) -> np.ndarray:
    """Exact zeros are not drawable on a log axis (NaN); every positive value is."""
    p = np.asarray(p, dtype=float)
    return np.where(p > 0.0, p, np.nan)


def bin_left_edge(edges_k: np.ndarray, i):
    """The descriptor value at the left edge of bin ``i`` (linear edges)."""
    e = np.asarray(edges_k, dtype=float)
    return e[0] + np.asarray(i, dtype=float) * (e[1] - e[0])


def bin_index(edges_k: np.ndarray, value):
    """The (fractional) bin index of a descriptor value; the inverse of ``bin_left_edge``."""
    e = np.asarray(edges_k, dtype=float)
    return (np.asarray(value, dtype=float) - e[0]) / (e[1] - e[0])


def out_of_range_share(block: dict, edges_k: np.ndarray, k: str) -> float:
    """The percentage of a block's weight whose descriptor ``k`` lies outside the edges.

    ``np.histogram`` drops those samples; they enter no histogram and no divergence.
    """
    e = np.asarray(edges_k, dtype=float)
    x = np.asarray(block[k], dtype=float)
    w = np.asarray(block.get("weights", np.ones_like(x)), dtype=float)
    inside = (x >= e[0]) & (x <= e[-1])
    return float(100.0 * (1.0 - w[inside].sum() / w.sum()))


def plot_subset_jsd(rows: Sequence[dict], h_ref: dict, edges: dict,
                    blocks: Sequence[dict], out_png,
                    show_r: Sequence[int] = (1, 7, 26)) -> None:
    """Panel (a): JSD against r; panels (b)-(d): the reference marginals with subsets."""
    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    n_pool = len(blocks)
    by_r = {int(row["r"]): row for row in rows}
    shown = [r for r in show_r if r in by_r]

    fig, axes = plt.subplots(1, 4, figsize=(22, 6.2))
    ax = axes[0]
    rs = [int(row["r"]) for row in rows]
    ax.plot(rs, _positive_or_nan(np.array([row["jsd"] for row in rows])),
            "o-", color="black", lw=2, label="total (sum of the three marginals)")
    for k, color in zip(_KEYS, ("tab:blue", "tab:orange", "tab:green")):
        vals = np.array([row[_PART_COLS[k]] for row in rows])
        ax.plot(rs, _positive_or_nan(vals), "s--", color=color, label=_LABELS[k] + " marginal")
    ax.set_yscale("log")
    ax.set_xlabel("subset size r (training points)")
    ax.set_ylabel("Jensen-Shannon divergence to the full pool (nats)")
    ax.set_title("(a) divergence of the chosen subset against r")
    ax.text(0.98, 0.97,
            f"each marginal term lies in [0, ln 2];\nthe total in [0, 3 ln 2] = "
            f"[0, {3.0 * math.log(2.0):.3f}]\nr = {n_pool} is the pool itself: divergence 0,\n"
            f"not drawn on the log axis",
            transform=ax.transAxes, fontsize=8, ha="right", va="top")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.3)

    cmap = plt.get_cmap("viridis")
    pool_block = _concat_blocks(blocks, range(n_pool))
    for ax, k in zip(axes[1:], _KEYS):
        p_ref = ss._to_pmf(h_ref[k])
        idx = np.arange(p_ref.size)
        for j, r in enumerate(shown):
            members = by_r[r]["chosen_indices"]
            h_sub = ss._bin_with_edges(_concat_blocks(blocks, members), edges)
            p_sub = ss._to_pmf(h_sub[k])
            label = f"r = {r}"
            if len(members) == n_pool:
                label += " (the pool itself: under the reference line)"
            elif np.count_nonzero(p_sub > 0.0) <= 2:
                label += f" ({np.count_nonzero(p_sub > 0.0)} populated bins)"
            ax.plot(idx, _positive_or_nan(p_sub), lw=1.0, alpha=0.85,
                    color=cmap(0.15 + 0.7 * j / max(1, len(shown) - 1)),
                    label=label, zorder=2)
        # the reference on top: where a subset reproduces the pool the two coincide
        ax.plot(idx, _positive_or_nan(p_ref), color="black", lw=1.6, alpha=0.8,
                label="full pool (reference)", zorder=3)
        ax.set_yscale("log")
        ax.set_xlabel("bin index (200 linear bins over the 0.1-99.9 percentile range)")
        ax.set_ylabel("probability mass per bin")
        ax.set_title(f"({'bcd'[_KEYS.index(k)]}) {_LABELS[k]} marginal, reference and subsets")
        e = np.asarray(edges[k])
        lo, width = float(e[0]), float(e[1] - e[0])
        sec = ax.secondary_xaxis(
            "top", functions=(lambda i, e=e: bin_left_edge(e, i),
                              lambda v, e=e: bin_index(e, v)))
        sec.xaxis.set_major_locator(MaxNLocator(5))
        sec.xaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:.2g}"))
        sec.set_xlabel(f"{_LABELS[k]} value at the bin's left edge (bin width {width:.3g})",
                       fontsize=8)
        ax.text(0.98, 0.97,
                f"first bin: {_first_bin_share(h_ref[k]):.1f}% of the reference mass\n"
                f"edges {lo:.3g} to {float(e[-1]):.3g}; "
                f"{out_of_range_share(pool_block, e, k):.2f}% of the pool weight outside them, discarded",
                transform=ax.transAxes, fontsize=8, ha="right", va="top")
        ax.legend(fontsize=8, loc="center right")
        ax.grid(alpha=0.3)

    fig.suptitle("Training-subset selection: Jensen-Shannon divergence to the full DFS pool",
                 fontsize=13)
    footer = (
        "Descriptors (rho^1/3, s, alpha) from one PBE SCF per species at def2-SVP, grid level 1, "
        "every grid point weighted by its quadrature weight; a point's sample is the concatenation "
        "of its species' grid points. Each marginal is binned into 200 linear bins over the "
        "reference's 0.1-99.9 percentile range (samples outside the range are discarded, not "
        "clipped into the end bins) and normalized to a probability mass function; the divergence "
        "is the equally weighted sum over the three marginals of 0.5 [KL(P||M) + KL(Q||M)], "
        "M = (P + Q) / 2, P the pool and Q the subset. "
        f"For each r the subset is the exhaustive minimizer over all C({n_pool}, r) combinations.")
    fig.text(0.5, 0.005, textwrap.fill(footer, width=210),
             ha="center", va="bottom", fontsize=7.5)
    fig.tight_layout(rect=(0, 0.06, 1, 0.95))
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ledger", default=str(DEFAULT_LEDGER))
    ap.add_argument("--reference", default=str(DEFAULT_REFERENCE))
    ap.add_argument("--descriptors", default=str(DEFAULT_DESCRIPTORS))
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR))
    ap.add_argument("--show-r", type=int, nargs="+", default=[1, 7, 26],
                    help="subset sizes overlaid on the marginal panels")
    args = ap.parse_args(argv)

    ledger = load_ledger(args.ledger, metric="jsd")
    h_ref, edges = load_reference(args.reference)
    points, blocks = pool_blocks(args.descriptors)
    rows = subset_rows(ledger, points, blocks, h_ref, edges)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    write_subset_table(rows, outdir / "subset_table.csv")
    plot_subset_jsd(rows, h_ref, edges, blocks, outdir / "subset_jsd_vs_full.png",
                    show_r=tuple(args.show_r))
    for row in rows:
        print(f"r = {row['r']:2d}  jsd = {row['jsd']:.6g}  (rho {row['jsd_rho']:.3g}, "
              f"s {row['jsd_s']:.3g}, alpha {row['jsd_alpha']:.3g})  "
              f"{row['n_ae']} ae / {row['n_bh76']} bh76 / {row['n_ip13']} ip13")
    print(f"wrote {outdir / 'subset_jsd_vs_full.png'} and {outdir / 'subset_table.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
