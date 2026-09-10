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

Columns: Architecture (the shown name, in the file's order: the display order of the
architectures, then the subset size), Subset, n rxn (``n_reactions``), n species
(``n_density_species``), E NN, E PBE (pool), eps NN, eps PBE (pool), ED NN, ED PBE (cell),
dED = ED NN - ED PBE (cell) from the raw values with its sign (a value rounding to zero keeps
its sign), negative (the network below PBE) in bold, and beats (yes/no, the suite's
``beats_pbe`` flag, which must agree with the sign of dED: a disagreement is a defect of the
CSV and is refused).

Usage:
    python notebooks/analysis/report_tables.py --splice REPORT.md SUMMARY.md
"""
from __future__ import annotations

import argparse
import csv
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
TABLE_NAMES = ("holdout_bh76", "holdout_w411", "holdout_combined",
               "holdout_combined_excl_tail", "insample_bh76", "insample_w411",
               "insample_combined")


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


def all_tables(family_dir, excl_dir) -> Dict[str, str]:
    """Every table by name, from the two family directories."""
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
    args = p.parse_args(argv)
    tables = all_tables(args.family_dir, args.excl_dir)
    for doc in args.splice:
        path = Path(doc)
        text = path.read_text()
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
