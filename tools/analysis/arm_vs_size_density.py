#!/usr/bin/env python3
"""The 25-cycle question: each protocol arm against the size cell of the same stored
architecture and subset, species by species, on the held-out density.

A merged family view (``manifest.json`` and the ``eval_holdout_<channel>`` records of its
cells) is read through the figure suite's held-out density reader. Every arm cell (a
manifest cell carrying ``protocol``) is paired with the size cell of the same stored
architecture and subset. For every species the two cells share, the twin-collapsed
density RMSE of each cell and their ratio arm/size are written, with the per-pair count
of species below one and the median ratio, a two-panel figure, the caveat naming each
cell's SCF cycle budget and converged count, and the LaTeX rows of the deck table from
the figure set's two pool CSVs.

Usage::

    python tools/analysis/arm_vs_size_density.py \\
        --run-dir <results>/runs/dfs_step7/v7_family/runs/run_20260908T153908Z \\
        --eval-channel val_best \\
        --family-dir reports/v7/figures_dfs_step7_v7_family_val_best

``arm_vs_size_density.csv`` and ``arm_vs_size_density.png`` land in ``--out-dir`` (the
family directory unless given); the summaries, the caveat and the LaTeX rows are printed.

The comparison conflates the training protocol with the evaluation protocol: each arm
was evaluated with the cycle budget it was trained with and the size cells with theirs,
so the caveat states both from ``eval_metadata.json``; the converged-SCF re-evaluation of
the same networks is the direct test.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Set, Tuple

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_HERE = Path(__file__).resolve().parent
CSV_NAME = "arm_vs_size_density.csv"
PNG_NAME = "arm_vs_size_density.png"
CSV_COLUMNS = ("subset_size", "arm", "species", "rmse_size", "rmse_arm", "ratio",
               "eps_size", "eps_arm", "cycles_size", "cycles_arm",
               "converged_size", "converged_arm", "tail")
EPS_CSV = "holdout_by_pool_3x3_eps.csv"
GRID_CSV = "holdout_by_pool_3x3.csv"
EPS_LEG = "combined_wtmad2_eps_gamma_dfs"    # DFS units: WTMAD-2, eps_n, ED
GRID_LEG = "combined_wtmad2"                 # grid-weighted density RMSE


def _suite():
    """The figure suite, loaded from its path once (the loader ``report_tables._suite``
    uses): its held-out reader, species key and recurring-tail rule are reused here."""
    cached = sys.modules.get("make_ablation_arch_figure")
    if isinstance(cached, types.ModuleType) and \
            callable(getattr(cached, "collect_holdout_density_rows", None)):
        return cached
    path = _HERE / "make_ablation_arch_figure.py"
    spec = importlib.util.spec_from_file_location("make_ablation_arch_figure", path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules["make_ablation_arch_figure"] = mod
    try:
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
    except BaseException:
        sys.modules.pop("make_ablation_arch_figure", None)
        raise
    return mod


def _is_num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


# ---------------------------------------------------------------------------
# the pairs
# ---------------------------------------------------------------------------

class Pair(NamedTuple):
    arm_idx: int
    size_idx: int
    subset_size: int
    arch_stored: str
    protocol: str


def pairs_from_manifest(run_dir) -> List[Pair]:
    """Every arm cell of the manifest (a cell carrying ``protocol``) with the size cell of
    the same stored architecture and subset (the cell without a protocol), in
    (subset_size, protocol) order. An arm without a partner is skipped and named; two
    size cells for one (architecture, subset) are refused, since the pairing would be
    ambiguous."""
    manifest = json.loads((Path(run_dir) / "manifest.json").read_text())
    arms: List[Tuple[int, str, int, str]] = []
    sizes: Dict[Tuple[str, int], int] = {}
    for entry in manifest.get("specs", []):
        idx, cell = entry.get("index"), entry.get("cell") or {}
        if not isinstance(idx, int) or cell.get("arch") is None \
                or cell.get("subset_size") is None:
            continue
        arch, ss = str(cell["arch"]), int(cell["subset_size"])
        protocol = cell.get("protocol")
        if protocol:
            arms.append((idx, arch, ss, str(protocol)))
        elif (arch, ss) in sizes:
            raise ValueError(f"two size cells for {arch} at r = {ss} in {run_dir}: "
                             f"spec_{sizes[(arch, ss)]:04d} and spec_{idx:04d}")
        else:
            sizes[(arch, ss)] = idx
    pairs: List[Pair] = []
    for idx, arch, ss, protocol in arms:
        size_idx = sizes.get((arch, ss))
        if size_idx is None:
            print(f"  (arm_vs_size: no size cell for {protocol} at r = {ss} ({arch}); "
                  f"spec_{idx:04d} skipped)")
            continue
        pairs.append(Pair(idx, size_idx, ss, arch, protocol))
    pairs.sort(key=lambda p: (p.subset_size, p.protocol))
    return pairs


def evaluated_pairs(hd_rows: Sequence[Dict[str, Any]], pairs: Sequence[Pair]) -> List[Pair]:
    """The pairs whose two cells both have held-out density rows. A merged view lists every
    arm the run will evaluate, so an arm the channel has not evaluated yet (or whose pass was
    refused) is paired by the manifest and dropped here, one printed line per pair naming
    the cell without rows."""
    present = {r.get("idx") for r in hd_rows if _is_num(r.get("density_rmse"))}
    kept: List[Pair] = []
    for p in pairs:
        missing = [(role, idx) for role, idx in (("arm", p.arm_idx), ("size", p.size_idx))
                   if idx not in present]
        if missing:
            what = ", ".join(f"the {role} cell spec_{idx:04d}" for role, idx in missing)
            print(f"  (arm_vs_size: r = {p.subset_size}, {p.protocol}: no held-out density "
                  f"rows for {what}; pair skipped)")
            continue
        kept.append(p)
    return kept


# ---------------------------------------------------------------------------
# the species rows
# ---------------------------------------------------------------------------

def _cell_species(hd_rows: Sequence[Dict[str, Any]], idx: int) -> Dict[str, Dict[str, Any]]:
    """Per casefolded species of one cell (rows joined on the spec index): the NN density
    RMSE as the mean over the case twins (the suite's own reduction), the per-electron L1
    as the mean of the finite twins, the cycles as the maximum of the known twins and the
    convergence as the twins' conjunction; a value no twin carries stays None. Rows
    without a finite NN leg contribute nothing."""
    key = _suite()._mol_cf
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in hd_rows:
        if r.get("idx") != idx or not r.get("molecule") or not _is_num(r.get("density_rmse")):
            continue
        groups.setdefault(key(r["molecule"]), []).append(r)
    out: Dict[str, Dict[str, Any]] = {}
    for cf, rs in groups.items():
        eps = [float(r["density_eps_l1"]) for r in rs if _is_num(r.get("density_eps_l1"))]
        cycles = [int(r["cycles_run"]) for r in rs if _is_num(r.get("cycles_run"))]
        flags = [r.get("scf_converged") for r in rs]
        out[cf] = {
            "rmse": float(np.mean([float(r["density_rmse"]) for r in rs])),
            "eps": float(np.mean(eps)) if eps else None,
            "cycles": max(cycles) if cycles else None,
            "converged": None if any(f is None for f in flags) else all(bool(f) for f in flags),
        }
    return out


def species_rows(hd_rows: Sequence[Dict[str, Any]], pairs: Sequence[Pair],
                 tail_cf: Set[str]) -> List[Dict[str, Any]]:
    """One row per (pair, species the two cells share): the collapsed values of the size
    and the arm cell, the ratio arm/size, and the tail flag. A species present in one cell
    only, or whose size RMSE is zero, is dropped and named."""
    rows: List[Dict[str, Any]] = []
    for p in pairs:
        size, arm = _cell_species(hd_rows, p.size_idx), _cell_species(hd_rows, p.arm_idx)
        if not size or not arm:
            role, idx = ("arm", p.arm_idx) if not arm else ("size", p.size_idx)
            print(f"  (arm_vs_size: r = {p.subset_size}, {p.protocol}: no held-out density "
                  f"rows for the {role} cell spec_{idx:04d}; pair skipped)")
            continue
        shared = sorted(set(size) & set(arm))
        only_one = sorted(set(size) ^ set(arm))
        if only_one:
            print(f"  (arm_vs_size: r = {p.subset_size}, {p.protocol}: species in one cell "
                  f"only, dropped: {', '.join(only_one)})")
        for cf in shared:
            s, a = size[cf], arm[cf]
            if s["rmse"] <= 0.0:
                print(f"  (arm_vs_size: r = {p.subset_size}, {p.protocol}: {cf} has a zero "
                      f"size-cell RMSE, no ratio; dropped)")
                continue
            rows.append({
                "subset_size": p.subset_size,
                "arm": p.protocol,
                "species": cf,
                "rmse_size": s["rmse"],
                "rmse_arm": a["rmse"],
                "ratio": a["rmse"] / s["rmse"],
                "eps_size": s["eps"],
                "eps_arm": a["eps"],
                "cycles_size": s["cycles"],
                "cycles_arm": a["cycles"],
                "converged_size": s["converged"],
                "converged_arm": a["converged"],
                "tail": 1 if cf in tail_cf else 0,
            })
    return rows


def summarize(rows: Sequence[Dict[str, Any]]) -> Dict[Tuple[int, str], Dict[str, Any]]:
    """Per (subset_size, arm): the species count, the count of ratios strictly below one
    and the median ratio."""
    groups: Dict[Tuple[int, str], List[float]] = {}
    for r in rows:
        groups.setdefault((int(r["subset_size"]), str(r["arm"])), []).append(float(r["ratio"]))
    return {k: {"n": len(v), "n_below_1": sum(1 for x in v if x < 1.0),
                "median_ratio": float(np.median(v))}
            for k, v in groups.items()}


def _sort_key(r: Dict[str, Any]) -> Tuple[int, str, float]:
    return (int(r["subset_size"]), str(r["arm"]), float(r["ratio"]))


def write_csv(rows: Sequence[Dict[str, Any]], path) -> None:
    """The species rows, sorted by (subset_size, arm, ratio); an unknown value is an
    empty field."""
    with Path(path).open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(CSV_COLUMNS))
        w.writeheader()
        for r in sorted(rows, key=_sort_key):
            w.writerow({c: r.get(c) for c in CSV_COLUMNS})


# ---------------------------------------------------------------------------
# the caveat and the figure
# ---------------------------------------------------------------------------

def _max_cycles(run_dir, idx: int, eval_subdir: str) -> Optional[int]:
    path = Path(run_dir) / "checkpoints" / f"spec_{idx:04d}" / eval_subdir / "eval_metadata.json"
    try:
        meta = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    value = (meta.get("solver_config") or {}).get("max_cycles")
    return int(value) if _is_num(value) else None


def _converged_phrase(hd_rows: Sequence[Dict[str, Any]], idx: int) -> str:
    """The cell's converged count over its twin-collapsed species (the population of the
    ratios and the legend): a species converged when every twin did, unknown when a twin
    lacks the flag; the unknown species are counted apart, never as not converged."""
    flags = [s["converged"] for s in _cell_species(hd_rows, idx).values()]
    known = [f for f in flags if f is not None]
    if not known:
        return "convergence unknown"
    text = f"{sum(1 for f in known if f)} of {len(known)} converged"
    unknown = len(flags) - len(known)
    return text + (f", {unknown} unknown" if unknown else "")


def cycle_caveat(run_dir, pairs: Sequence[Pair], eval_subdir: str,
                 hd_rows: Optional[Sequence[Dict[str, Any]]] = None) -> str:
    """The evaluation-protocol caveat: the channel, then per cell (the size cells first,
    then the arms) its ``solver_config.max_cycles`` and its converged count over the
    twin-collapsed species the reader's rows yield for the cell (the population of the
    ratios and of the legend), the species with an unknown flag counted apart."""
    if hd_rows is None:
        hd_rows = _suite().collect_holdout_density_rows(Path(run_dir), eval_subdir)
    channel = eval_subdir[len("eval_holdout_"):] if eval_subdir.startswith("eval_holdout_") \
        else eval_subdir
    seen: Set[int] = set()
    cells: List[Tuple[str, int]] = []
    for p in pairs:
        if p.size_idx not in seen:
            cells.append((f"size cell at r = {p.subset_size}", p.size_idx))
            seen.add(p.size_idx)
    for p in pairs:
        cells.append((f"{p.protocol} at r = {p.subset_size}", p.arm_idx))
    parts = []
    for label, idx in cells:
        mc = _max_cycles(run_dir, idx, eval_subdir)
        parts.append(f"{label}: max_cycles {mc if mc is not None else 'unknown'}, "
                     f"{_converged_phrase(hd_rows, idx)}")
    return (f"{channel} channel; each cell evaluated with its own SCF cycle budget "
            f"(density species): " + "; ".join(parts) + ".")


def make_figure(rows: Sequence[Dict[str, Any]], summaries: Dict[Tuple[int, str], Dict[str, Any]],
                caveat: str, out_png):
    """One panel per subset: per arm the species ratios sorted ascending against their
    rank on a log axis, a line at one, the tail species as hollow markers, the legend
    carrying each arm's counts and median, the caveat as the footer. Returns the Figure."""
    subsets = sorted({int(r["subset_size"]) for r in rows})
    n = max(1, len(subsets))
    fig, axes = plt.subplots(1, n, figsize=(4.6 * n + 0.6, 4.6), squeeze=False)
    for ax, ss in zip(axes[0], subsets):
        arms = sorted({str(r["arm"]) for r in rows if int(r["subset_size"]) == ss})
        tail_drawn = False
        for arm in arms:
            arm_rows = sorted((r for r in rows
                               if int(r["subset_size"]) == ss and str(r["arm"]) == arm),
                              key=lambda r: float(r["ratio"]))
            s = summaries.get((ss, arm), {"n": len(arm_rows), "n_below_1": 0,
                                          "median_ratio": float("nan")})
            x = np.arange(1, len(arm_rows) + 1)
            y = np.array([float(r["ratio"]) for r in arm_rows])
            ax.plot(x, y, marker="o", ms=3, lw=1.0,
                    label=f"{arm}: {s['n_below_1']} of {s['n']} below 1, "
                          f"median {s['median_ratio']:.2f}")
            tail = [(i, float(r["ratio"])) for i, r in zip(x, arm_rows) if int(r["tail"]) == 1]
            if tail:
                ax.plot([i for i, _ in tail], [v for _, v in tail], "o", ms=8,
                        mfc="none", mec="black", lw=0,
                        label=None if tail_drawn else "recurring tail species")
                tail_drawn = True
        ax.axhline(1.0, color="0.4", lw=0.8, ls="--")
        ax.set_yscale("log")
        ax.set_title(f"r = {ss}")
        ax.set_xlabel("species, sorted by ratio")
        ax.set_ylabel("held-out density RMSE ratio, arm / size")
        ax.legend(fontsize=7, loc="upper left")
    if not subsets:
        axes[0][0].set_title("no paired cells")
    fig.text(0.01, 0.01, caveat, fontsize=6.5, ha="left", va="bottom", wrap=True)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(out_png, dpi=150)
    return fig


# ---------------------------------------------------------------------------
# the deck table
# ---------------------------------------------------------------------------

def _read_leg(path, leg: str) -> List[Dict[str, str]]:
    with Path(path).open(newline="") as f:
        return [r for r in csv.DictReader(f) if r.get("leg") == leg]


def _esc(text: str) -> str:
    return text.replace("_", "\\_")


def latex_rows(family_dir) -> List[str]:
    """The deck table rows from the figure set's two pool CSVs: for every cell whose stored
    architecture and subset carry both an untagged row (the size cell) and tagged rows (the
    arms), the size row then the arms by shown name --
    ``arch & r & WTMAD-2 & eps_n & grid RMSE & ED & ED cap`` (the energy, eps and ED columns
    from the combined DFS-units leg of the eps file, the grid RMSE from the combined leg of
    the grid file) -- then one PBE row from the files' own reference columns. A tagged row
    with no untagged partner is skipped and named; a selected cell missing from the grid
    file, two untagged rows for one cell, or a PBE reference that differs across the
    selected rows is refused."""
    fam = Path(family_dir)
    eps_rows = _read_leg(fam / EPS_CSV, EPS_LEG)
    grid_by = {(r["arch"], r["arch_stored"], int(r["subset_size"])): r
               for r in _read_leg(fam / GRID_CSV, GRID_LEG)}
    groups: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for r in eps_rows:
        key = (r["arch_stored"], int(r["subset_size"]))
        g = groups.setdefault(key, {"size": None, "arms": []})
        if "[" in r["arch"]:
            g["arms"].append(r)
        elif g["size"] is not None:
            raise ValueError(f"two untagged rows for {r['arch_stored']} at r = {key[1]} in "
                             f"{fam / EPS_CSV}")
        else:
            g["size"] = r
    out: List[str] = []
    refs: Set[Tuple[float, float, float]] = set()
    for stored, ss in sorted(groups, key=lambda k: (k[1], k[0])):
        g = groups[(stored, ss)]
        if not g["arms"]:
            continue
        if g["size"] is None:
            print(f"  (arm_vs_size: {', '.join(a['arch'] for a in g['arms'])} at r = {ss} "
                  f"has no untagged size row; skipped)")
            continue
        for r in [g["size"]] + sorted(g["arms"], key=lambda a: a["arch"]):
            gr = grid_by.get((r["arch"], r["arch_stored"], ss))
            if gr is None:
                raise ValueError(f"cell {r['arch']} at r = {ss} is in {EPS_CSV} but not on "
                                 f"the {GRID_LEG} leg of {GRID_CSV}")
            out.append(f"{_esc(r['arch'])} & {ss} & {float(r['E_kcalmol']):.2f} & "
                       f"{float(r['D_rmse']):.5f} & {float(gr['D_rmse']):.2e} & "
                       f"{float(r['ED_kcalmol']):.2f} & {float(r['ED_pbe_cell_kcalmol']):.2f}")
            refs.add((float(r["E_pbe_kcalmol"]), float(r["D_pbe_rmse"]),
                      float(gr["D_pbe_rmse"])))
    if not out:
        return out
    if len(refs) != 1:
        raise ValueError(f"the PBE reference differs across the selected rows: {sorted(refs)}")
    e_pbe, d_pbe, grid_pbe = next(iter(refs))
    out.append(f"PBE & -- & {e_pbe:.2f} & {d_pbe:.5f} & {grid_pbe:.2e} & -- & --")
    return out


# ---------------------------------------------------------------------------
# the command line
# ---------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run-dir", required=True, help="the merged family view")
    ap.add_argument("--eval-channel", default="val_best",
                    help="held-out channel: eval_holdout_<channel> (default val_best)")
    ap.add_argument("--family-dir", default=None,
                    help="the figure set holding the two pool CSVs (default "
                         "figures_dfs_step7_v7_family_<channel> beside this script)")
    ap.add_argument("--out-dir", default=None,
                    help="where the CSV and PNG land (default: the family directory)")
    args = ap.parse_args(argv)
    run = Path(args.run_dir)
    eval_subdir = f"eval_holdout_{args.eval_channel}"
    fam = Path(args.family_dir) if args.family_dir \
        else _HERE / f"figures_dfs_step7_v7_family_{args.eval_channel}"
    out_dir = Path(args.out_dir) if args.out_dir else fam
    out_dir.mkdir(parents=True, exist_ok=True)

    suite = _suite()
    pairs = pairs_from_manifest(run)
    if not pairs:
        print(f"no arm cell with a size partner in {run / 'manifest.json'}")
        return 1
    hd_rows = suite.collect_holdout_density_rows(run, eval_subdir)
    pairs = evaluated_pairs(hd_rows, pairs)
    if not pairs:
        print("no pair with held-out density rows on both sides")
        return 1
    tail, n_cells = suite.recurring_tail_species(hd_rows)
    print(f"recurring held-out tail over {n_cells} cells: {sorted(tail)}")
    rows = species_rows(hd_rows, pairs, set(tail))
    summaries = summarize(rows)
    for (ss, arm), s in sorted(summaries.items()):
        print(f"r = {ss}, {arm}: {s['n_below_1']} of {s['n']} species below 1, "
              f"median ratio {s['median_ratio']:.3f}")
    caveat = cycle_caveat(run, pairs, eval_subdir, hd_rows=hd_rows)
    print(caveat)
    write_csv(rows, out_dir / CSV_NAME)
    fig = make_figure(rows, summaries, caveat, out_dir / PNG_NAME)
    plt.close(fig)
    for line in latex_rows(fam):
        print(line)
    print(f"wrote {out_dir / CSV_NAME} and {out_dir / PNG_NAME}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
