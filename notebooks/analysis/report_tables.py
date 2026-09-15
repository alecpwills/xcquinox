"""The per-cell tables of the v7 documents, generated from the family figure CSVs.

The report, the summary and the slides print one table per leg over every finished cell
of the merged family view; the numbers are the ones the figure suite writes
(``holdout_by_pool_3x3_eps.csv`` and ``insample_by_pool_3x3_eps.csv`` of the family
directories), never retyped. Each markdown document carries marker blocks

    <!-- table:NAME -->
    <!-- /table:NAME -->

which :func:`splice` fills; a document without a marker for a table is left as it is for
that table, and a document is written only after every table it carries was spliced. The
tables: ``holdout_bh76``, ``holdout_w411``, ``holdout_combined`` (the validation-best family
set), ``holdout_combined_excl_tail`` (the tail-excluded sibling, whose PBE columns are that
file's own), and ``insample_bh76``, ``insample_w411``, ``insample_combined`` (the in-sample
twin). The slide decks take :func:`latex_rows`, one tabular row per cell.

Two further tables come from the cells' training logs rather than from a CSV (Sec. 4.6 of
the report; 2026-09-15): ``training_losses`` (the first and last five-epoch means of the
total and of every weighted channel, from ``aux_log.pkl`` through the figure suite's own
collector) and ``validation_checks`` (the validation MAE at every check in epoch order and
the validation-best epoch, the completed-epoch count at the check). They are generated from
the merged family view the figures were rendered from, one row per evaluated cell (the cells
the family CSV carries, the set every other table prints), named as the suite names them (a
cell whose run trains under another protocol shows its tag, ``deep_3x16 [25 cycles]``), in the
display order of the architectures and then the subset size, the run column the documents'
scope-table label of the cell's category; a document carrying either marker needs ``--view``.

Columns: Architecture (the shown name, in the file's order: the display order of the
architectures, then the subset size), Subset, n rxn (``n_reactions``), n species
(``n_density_species``), E NN, E PBE (pool), eps NN, eps PBE (pool), ED NN, ED PBE (cell),
dED = ED NN - ED PBE (cell) from the raw values with its sign (a value rounding to zero keeps
its sign), negative (the network below PBE) in bold, and beats (yes/no, the suite's
``beats_pbe`` flag, which must agree with the sign of dED: a disagreement is a defect of the
CSV and is refused).

Usage:
    python notebooks/analysis/report_tables.py --splice REPORT.md SUMMARY.md
    python notebooks/analysis/report_tables.py --view <family view dir> --splice REPORT.md
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, List

_HERE = Path(__file__).resolve().parent
DEFAULT_FAMILY_DIR = _HERE / "figures_dfs_step7_v7_family_val_best"
DEFAULT_EXCL_DIR = _HERE / "figures_dfs_step7_v7_family_val_best_excl_tail"
HOLDOUT_CSV = "holdout_by_pool_3x3_eps.csv"
INSAMPLE_CSV = "insample_by_pool_3x3_eps.csv"
LEGS = {"bh76": "bh76_wtmad2_eps_gamma_dfs", "w411": "w411_wtmad2_eps_gamma_dfs",
        "combined": "combined_wtmad2_eps_gamma_dfs"}
HEADER = ("| Architecture | Subset | n rxn | n species | E NN | E PBE (pool) | eps NN | "
          "eps PBE (pool) | ED NN | ED PBE (cell) | dED | beats |")
RULE = "|---|---|---|---|---|---|---|---|---|---|---|---|"
#: the two tables generated from the view's training logs rather than from a CSV
#: (Sec. 4.6 of the report): they need ``--view``
TRAINING_TABLE_NAMES = ("training_losses", "validation_checks")
TABLE_NAMES = ("holdout_bh76", "holdout_w411", "holdout_combined",
               "holdout_combined_excl_tail", "insample_bh76", "insample_w411",
               "insample_combined", *TRAINING_TABLE_NAMES)


def read_leg_rows(csv_path, leg: str) -> List[dict]:
    """The rows of one leg (a key of :data:`LEGS` or the full leg name), in the file's
    order; an unknown leg yields nothing."""
    name = LEGS.get(leg, leg)
    with open(csv_path, newline="") as fh:
        return [r for r in csv.DictReader(fh) if r["leg"] == name]


def _cell_values(r: dict) -> dict:
    ed_nn = float(r["ED_kcalmol"])
    ed_pbe = float(r["ED_pbe_cell_kcalmol"])
    ded = ed_nn - ed_pbe
    beats = str(r.get("beats_pbe", "")).strip().lower() == "true"
    if beats != (ded < 0):
        raise ValueError(
            f"{r['arch']} at {r['subset_size']} on {r['leg']}: beats_pbe={r.get('beats_pbe')} "
            f"disagrees with ED_NN - ED_PBE(cell) = {ded:+.4f}; the CSV's verdict and its "
            "cell values must be one comparison")
    return {"arch": r["arch"], "subset": int(float(r["subset_size"])),
            "n_rxn": r["n_reactions"], "n_species": r["n_density_species"],
            "e_nn": float(r["E_kcalmol"]), "e_pbe": float(r["E_pbe_kcalmol"]),
            "eps_nn": float(r["D_rmse"]), "eps_pbe": float(r["D_pbe_rmse"]),
            "ed_nn": ed_nn, "ed_pbe": ed_pbe, "ded": ded, "beats": beats}


def _fmt_row(r: dict) -> str:
    v = _cell_values(r)
    ded_text = f"{v['ded']:+.2f}"
    if v["ded"] < 0:
        ded_text = f"**{ded_text}**"
    return (f"| {v['arch']} | {v['subset']} | {v['n_rxn']} | {v['n_species']} | "
            f"{v['e_nn']:.2f} | {v['e_pbe']:.2f} | {v['eps_nn']:.5f} | {v['eps_pbe']:.5f} | "
            f"{v['ed_nn']:.2f} | {v['ed_pbe']:.2f} | {ded_text} | "
            f"{'yes' if v['beats'] else 'no'} |")


def _table(rows: List[dict], title: str) -> str:
    lines = []
    if title:
        lines.append(title)
        lines.append("")
    lines.append(HEADER)
    lines.append(RULE)
    lines.extend(_fmt_row(r) for r in rows)
    return "\n".join(lines)


def holdout_table(csv_path, leg: str, *, title: str = "") -> str:
    """The markdown table of one held-out leg over every cell of the file."""
    return _table(read_leg_rows(csv_path, leg), title)


def insample_table(csv_path, leg: str, *, title: str = "") -> str:
    """The in-sample twin: the same columns from the in-sample CSV."""
    return _table(read_leg_rows(csv_path, leg), title)


def latex_rows(csv_path, leg: str) -> List[str]:
    """One LaTeX tabular row per cell of the leg, ``Arch & r & E NN & E PBE & eps NN &
    eps PBE & ED NN & ED PBE & dED`` (underscores escaped, negative dED bold, no line
    terminator), for the slide decks."""
    out = []
    for r in read_leg_rows(csv_path, leg):
        v = _cell_values(r)
        ded_text = f"{v['ded']:+.2f}"
        if v["ded"] < 0:
            ded_text = f"\\textbf{{{ded_text}}}"
        arch = v["arch"].replace("_", "\\_")
        out.append(f"{arch} & {v['subset']} & {v['e_nn']:.2f} & {v['e_pbe']:.2f} & "
                   f"{v['eps_nn']:.5f} & {v['eps_pbe']:.5f} & {v['ed_nn']:.2f} & "
                   f"{v['ed_pbe']:.2f} & {ded_text}")
    return out


def splice(md_text: str, name: str, table: str) -> str:
    """``md_text`` with the block between ``<!-- table:NAME -->`` and
    ``<!-- /table:NAME -->`` replaced by ``table`` (the markers kept)."""
    start, end = f"<!-- table:{name} -->", f"<!-- /table:{name} -->"
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.S)
    if not pattern.search(md_text):
        raise ValueError(f"no marker block for table {name!r} ({start} ... {end})")
    return pattern.sub(lambda _m: f"{start}\n{table}\n{end}", md_text, count=1)


def has_marker(md_text: str, name: str) -> bool:
    return f"<!-- table:{name} -->" in md_text and f"<!-- /table:{name} -->" in md_text


# ---------------------------------------------------------------------------
# the training-log tables of Sec. 4.6 (2026-09-15): the cells' aux_log.pkl read
# through the figure suite's own collector on the merged family view
# ---------------------------------------------------------------------------
#: epochs averaged at each end of a cell's training: the report's "first and last
#: five-epoch means"
WINDOW = 5
#: the channel columns of the training table in the hand table's order: the
#: collector's series key and the column label
CHANNEL_COLUMNS = (("total", "total"), ("loss_AE", "AE"), ("loss_BH76", "reactions"),
                   ("loss_IP13", "IP13"), ("loss_vxc", "V_xc"), ("loss_rho", "rho"))
TRAINING_HEADER = ("| Run | Architecture | Subset | Epochs | "
                   + " | ".join(f"{label} first/last" for _k, label in CHANNEL_COLUMNS)
                   + " |")
TRAINING_RULE = "|" + "---|" * (4 + len(CHANNEL_COLUMNS))
VALIDATION_HEADER = ("| Run | Architecture | Subset | Epochs | "
                     "Validation MAE at each 25-epoch check (kcal/mol) | "
                     "Validation-best epoch |")
VALIDATION_RULE = "|---|---|---|---|---|---|"
#: the run labels of the documents' scope table, by the view manifest's category;
#: an unlisted category is printed as it is
RUN_LABELS = {
    "dfs6311_grid3_v7g1_size": "size",
    "dfs6311_grid3_v7g2a_families_core": "families",
    "dfs6311_grid3_v7g2_families_mgga": "meta-GGA families",
    "dfs6311_grid3_v7g1_c25": "25-cycle arm",
    "dfs6311_grid3_v7g1_dfsparity": "dpyscf-parity arm",
}
#: a value the cell does not have: no log on disk, no validation check
ABSENT = "--"


def _suite():
    """The figure suite, loaded from its path once: its collector reads the logs and
    its manifest boundary names the cells. An entry already in ``sys.modules`` is reused
    only when it is the suite (a stale or half-built entry is replaced), and a load that
    fails leaves no entry behind."""
    import importlib.util
    import types
    cached = sys.modules.get("make_ablation_arch_figure")
    if isinstance(cached, types.ModuleType) and \
            callable(getattr(cached, "collect_training_channel_losses", None)):
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


def _sci(x: float) -> str:
    """``%.2e`` without a zero-padded exponent (``4.47e-5``, ``0.00e+0``): the hand
    tables' form; a value that is not finite (a log whose rows carry no total) reads
    as absent."""
    x = float(x)
    if not math.isfinite(x):
        return ABSENT
    return re.sub(r"e([+-])0(?=\d)", r"e\1", f"{x:.2e}")


def _ends(values, window: int = WINDOW) -> str:
    """``first / last``: the means of the first and of the last ``window`` values
    (all of them when there are fewer)."""
    v = [float(x) for x in values]
    if not v:
        return ABSENT
    head, tail = v[:window], v[-window:]
    return f"{_sci(sum(head) / len(head))} / {_sci(sum(tail) / len(tail))}"


def family_cells(family_dir) -> set:
    """``{(shown arch, subset_size)}`` of the cells the family CSV carries: the
    evaluated cells every other table of the documents prints. A CSV without a
    cell row is refused: it would splice empty tables and say nothing."""
    path = Path(family_dir) / HOLDOUT_CSV
    with open(path, newline="") as fh:
        cells = {(r["arch"], int(float(r["subset_size"]))) for r in csv.DictReader(fh)}
    if not cells:
        raise ValueError(f"{path}: no cell rows; the family CSV names the evaluated "
                         "cells the tables print, and an empty one is not a view "
                         "without cells but a broken file")
    return cells


def view_cells(view_dir, family_dir=None) -> List[dict]:
    """``[{index, arch, subset_size, run}]`` of the view's cells the tables print:
    the manifest cells named as the suite names them (its manifest boundary,
    ``make_cluster_pulls_figure._read_manifest_cells``: the shown name with the
    cell's and the run's protocol tag), the run label of each cell's category, in
    the display order of the architectures (a tagged name after its untagged one)
    and then the subset size. With ``family_dir`` only the cells the family CSV
    carries are kept (the evaluated set); without it every manifest cell."""
    if str(_HERE) not in sys.path:
        sys.path.insert(0, str(_HERE))
    from arch_style import order_present
    view_dir = Path(view_dir)
    if not (view_dir / "manifest.json").is_file():
        raise FileNotFoundError(f"{view_dir / 'manifest.json'}: no manifest, no cells")
    named = _suite().ccp._read_manifest_cells(view_dir)
    manifest = json.loads((view_dir / "manifest.json").read_text())
    categories = {rec.get("index"): rec.get("category")
                  for rec in manifest.get("specs", []) if isinstance(rec, dict)}
    keep = family_cells(family_dir) if family_dir is not None else None
    cells: List[dict] = []
    for index in sorted(named):
        cell = named[index]
        arch, ss = cell.get("arch"), cell.get("subset_size")
        if arch is None or ss is None:
            print(f"[tables] view spec {index}: no architecture or subset size in the "
                  "manifest; not tabulated")
            continue
        if keep is not None and (str(arch), int(ss)) not in keep:
            continue
        category = categories.get(index)
        cells.append({"index": int(index), "arch": str(arch), "subset_size": int(ss),
                      "run": RUN_LABELS.get(category, category or ABSENT)})
    rank = {a: i for i, a in enumerate(order_present([c["arch"] for c in cells]))}
    cells.sort(key=lambda c: (rank[c["arch"]], c["subset_size"], c["index"]))
    return cells


def training_rows(view_dir) -> Dict[int, dict]:
    """The suite's per-cell training rows (``collect_training_channel_losses``: the
    per-epoch means of the total and of every weighted channel, the validation
    checks) by view index; a cell without a readable log has no row."""
    return {int(r["idx"]): r
            for r in _suite().collect_training_channel_losses(Path(view_dir))}


def _training_losses(cells: List[dict], rows: Dict[int, dict], title: str) -> str:
    lines = []
    if title:
        lines.append(title)
        lines.append("")
    lines.append(TRAINING_HEADER)
    lines.append(TRAINING_RULE)
    for c in cells:
        head = f"| {c['run']} | {c['arch']} | {c['subset_size']} | "
        r = rows.get(c["index"])
        if r is None:
            lines.append(head + " | ".join([ABSENT] * (1 + len(CHANNEL_COLUMNS))) + " |")
            continue
        series = {"total": r["total"], **r["channels"]}
        lines.append(head + f"{len(r['epochs'])} | "
                     + " | ".join(_ends(series[k]) for k, _label in CHANNEL_COLUMNS)
                     + " |")
    return "\n".join(lines)


def _validation_checks(cells: List[dict], rows: Dict[int, dict], title: str) -> str:
    lines = []
    if title:
        lines.append(title)
        lines.append("")
    lines.append(VALIDATION_HEADER)
    lines.append(VALIDATION_RULE)
    for c in cells:
        head = f"| {c['run']} | {c['arch']} | {c['subset_size']} | "
        r = rows.get(c["index"])
        if r is None:
            lines.append(head + f"{ABSENT} | {ABSENT} | {ABSENT} |")
            continue
        checks = sorted((int(e), float(m)) for e, m in r["validation"])
        if not checks:
            lines.append(head + f"{len(r['epochs'])} | {ABSENT} | {ABSENT} |")
            continue
        # the checkpoint the figures read: the smallest MAE, the earliest on a tie;
        # the log's epoch key is 0-based (train.py records the check at epoch + 1
        # being a multiple of validate_every), so the epoch printed is the
        # completed count at the check, as the Epochs column and the prose count
        best = min(checks, key=lambda t: (t[1], t[0]))[0] + 1
        lines.append(head + f"{len(r['epochs'])} | "
                     + ", ".join(f"{m:.1f}" for _e, m in checks) + f" | {best} |")
    return "\n".join(lines)


def _not_tabulated(cells: List[dict], rows: Dict[int, dict]) -> None:
    """One printed line for the logged specs the tables leave out (cells the family
    CSV does not carry, or specs without a manifest cell), so nothing vanishes."""
    left = sorted(set(rows) - {c["index"] for c in cells})
    if left:
        print(f"[tables] {len(left)} logged spec(s) not tabulated (not among the family "
              f"CSV's cells, or without a manifest cell): {left}")


def training_losses_table(view_dir, *, family_dir=None, title: str = "") -> str:
    """Run | Architecture | Subset | Epochs | total first/last | <channel> first/last
    ..., one row per cell (the family CSV's cells with ``family_dir``, every manifest
    cell without), from the first and last :data:`WINDOW` per-epoch means the
    suite's collector computes at the trained weights; a cell without a log reads
    ``--``."""
    cells, rows = view_cells(view_dir, family_dir), training_rows(view_dir)
    _not_tabulated(cells, rows)
    return _training_losses(cells, rows, title)


def validation_checks_table(view_dir, *, family_dir=None, title: str = "") -> str:
    """Run | Architecture | Subset | Epochs | the validation MAE at every check in
    epoch order | the validation-best epoch (the completed-epoch count at the check
    with the smallest MAE, the earliest on a tie), one row per cell as above; a cell
    without a log or without a check reads ``--``."""
    cells, rows = view_cells(view_dir, family_dir), training_rows(view_dir)
    _not_tabulated(cells, rows)
    return _validation_checks(cells, rows, title)


def training_tables(view_dir, family_dir=None) -> Dict[str, str]:
    """Both training-log tables by name, the logs read once."""
    cells, rows = view_cells(view_dir, family_dir), training_rows(view_dir)
    _not_tabulated(cells, rows)
    return {"training_losses": _training_losses(cells, rows, ""),
            "validation_checks": _validation_checks(cells, rows, "")}


def all_tables(family_dir, excl_dir) -> Dict[str, str]:
    """Every CSV table by name, from the two family directories."""
    family_dir, excl_dir = Path(family_dir), Path(excl_dir)
    out: Dict[str, str] = {}
    for leg in ("bh76", "w411", "combined"):
        out[f"holdout_{leg}"] = holdout_table(family_dir / HOLDOUT_CSV, leg)
        out[f"insample_{leg}"] = insample_table(family_dir / INSAMPLE_CSV, leg)
    out["holdout_combined_excl_tail"] = holdout_table(excl_dir / HOLDOUT_CSV, "combined")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--family-dir", default=str(DEFAULT_FAMILY_DIR))
    p.add_argument("--excl-dir", default=str(DEFAULT_EXCL_DIR))
    p.add_argument("--splice", nargs="+", required=True,
                   help="the markdown documents whose marker blocks are filled")
    p.add_argument("--view", default=None,
                   help="the merged family view directory (the run directory the "
                        "figures were rendered from); required when a document "
                        "carries a training_losses or validation_checks marker, "
                        "which are generated from the view's training logs")
    args = p.parse_args(argv)
    docs = [(Path(d), Path(d).read_text()) for d in args.splice]
    needs_view = [path.name for path, text in docs
                  if any(has_marker(text, n) for n in TRAINING_TABLE_NAMES)]
    if needs_view and args.view is None:
        p.error(f"--view is required: {', '.join(needs_view)} carries the "
                f"{' or '.join(TRAINING_TABLE_NAMES)} markers, which are generated "
                "from the family view's training logs")
    tables = all_tables(args.family_dir, args.excl_dir)
    if needs_view:
        tables.update(training_tables(args.view, args.family_dir))
    for path, text in docs:
        written = []
        for name in TABLE_NAMES:
            if has_marker(text, name):
                text = splice(text, name, tables[name])
                written.append(f"{name} ({tables[name].count(chr(10)) - 1} rows)")
        path.write_text(text)
        print(f"[tables] {path.name}: {len(written)} table(s): " + ", ".join(written))
    return 0


if __name__ == "__main__":
    sys.exit(main())
