"""The metrics table of a Slim16 evaluation run (hpcjobs/slim16_eval.py).

Per network: the converged species count, the total-energy MAE against PBE on
the run's own footing (the PBE-DF table beside the run; over every species and
over the converged ones), the same against the precompute's full-integral PBE,
the cloning paper's WTMAD-2 against PBE-DF (over the reactions whose species
all converged, the paper's filter, and over all), the reaction MAE against the
GMTKN55 references for the network, PBE-DF and full-integral PBE, and the
size of the density-fitting footing (the two PBE energies). Energies in
kcal/mol.

Usage::

    python tools/analysis/slim16_table.py <local run dir>

The table is printed and written as ``slim16_metrics.csv`` beside the run.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

from xcquinox.pipeline.eval_holdout import KCAL_PER_HA, paper_wtmad2
from xcquinox.pipeline.gmtkn55_sets import load_full_slim

HOLDOUT_SUBDIR = "eval_holdout_converged"
PBE_DF_FILE = "pbe_df.json"

COLUMNS = (
    "label", "arch_name", "kind", "certificate", "n_species", "n_converged",
    "n_species_not_df",
    "mae_te_df_all", "mae_te_df_converged", "mae_te_exact_all",
    "n_reactions", "n_reactions_converged",
    "wtmad2_paper_converged", "wtmad2_paper_all",
    "mae_rxn_ref_nn", "mae_rxn_ref_pbe_df", "mae_rxn_ref_pbe_exact",
    "df_footing_mean", "df_footing_max",
)


def _finite(value) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def _mean(values) -> float:
    values = [v for v in values if _finite(v)]
    return sum(values) / len(values) if values else float("nan")


def reaction_energy(record: dict, energies: dict) -> float:
    """The reaction energy of a per-reaction record from per-species total
    energies in Hartree, in kcal/mol; NaN when a species is missing."""
    names = list(record.get("reactants", [])) + list(record.get("products", []))
    coeffs = list(record.get("coeffs", []))
    if len(names) != len(coeffs):
        return float("nan")
    total = 0.0
    for name, coeff in zip(names, coeffs):
        energy = energies.get(name)
        if not _finite(energy):
            return float("nan")
        total += float(coeff) * float(energy)
    return total * KCAL_PER_HA


def network_metrics(molecules: list, reactions: list, pbe_df: dict,
                    subsets: dict) -> dict:
    """The metrics of one network from its channel outputs, the PBE-DF table
    (name -> energy in Hartree) and the reaction -> subset map."""
    e_nn = {m["molecule"]: m.get("E_total_nn") for m in molecules}
    e_pbe = {m["molecule"]: m.get("E_pbe") for m in molecules}
    converged = {m["molecule"] for m in molecules if m.get("scf_converged")}
    te_df = {n: (e_nn[n] - pbe_df[n]) * KCAL_PER_HA for n in e_nn
             if _finite(e_nn[n]) and _finite(pbe_df.get(n))}
    te_exact = {n: (e_nn[n] - e_pbe[n]) * KCAL_PER_HA for n in e_nn
                if _finite(e_nn[n]) and _finite(e_pbe.get(n))}
    footing = [(pbe_df[n] - e_pbe[n]) * KCAL_PER_HA for n in e_pbe
               if _finite(e_pbe[n]) and _finite(pbe_df.get(n))]
    rows_all, rows_converged, ref_rows = [], [], []
    for record in reactions:
        de_ref = reaction_energy(record, pbe_df)
        de_nn = record.get("de_nn_kcalmol")
        row = {"subset": subsets.get(record["name"], "?"), "de_nn": de_nn,
               "de_ref": de_ref}
        rows_all.append(row)
        names = list(record.get("reactants", [])) + list(record.get("products", []))
        if all(n in converged for n in names):
            rows_converged.append(row)
        ref = record.get("reaction_energy_ref_kcalmol")
        ref_rows.append((de_nn, de_ref, record.get("de_pbe_kcalmol"), ref))
    wt_conv, _ = paper_wtmad2(rows_converged)
    wt_all, _ = paper_wtmad2(rows_all)
    return {
        "n_species": len(molecules),
        "n_converged": len(converged),
        "mae_te_df_all": _mean(abs(v) for v in te_df.values()),
        "mae_te_df_converged": _mean(abs(v) for n, v in te_df.items() if n in converged),
        "mae_te_exact_all": _mean(abs(v) for v in te_exact.values()),
        "n_reactions": len(reactions),
        "n_reactions_converged": sum(
            1 for r in rows_converged if _finite(r["de_nn"]) and _finite(r["de_ref"])),
        "wtmad2_paper_converged": wt_conv,
        "wtmad2_paper_all": wt_all,
        "mae_rxn_ref_nn": _mean(abs(nn - ref) for nn, _df, _ex, ref in ref_rows
                                if _finite(nn) and _finite(ref)),
        "mae_rxn_ref_pbe_df": _mean(abs(df - ref) for _nn, df, _ex, ref in ref_rows
                                    if _finite(df) and _finite(ref)),
        "mae_rxn_ref_pbe_exact": _mean(abs(ex - ref) for _nn, _df, ex, ref in ref_rows
                                       if _finite(ex) and _finite(ref)),
        "df_footing_mean": _mean(abs(v) for v in footing),
        "df_footing_max": max((abs(v) for v in footing), default=float("nan")),
    }


def build_table(run_dir: Path) -> list:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    width = int(manifest["width"])
    pool = manifest.get("identity", {}).get("pool", "slim16")
    _species, pool_reactions = load_full_slim(pool)
    subsets = {r["name"]: r["subset"] for r in pool_reactions}
    table_path = run_dir / PBE_DF_FILE
    pbe_df = {}
    n_not_df = None
    if table_path.is_file():
        payload = json.loads(table_path.read_text(encoding="utf-8"))
        n_not_df = payload.get("n_species_not_df")
        pbe_df = {name: value.get("E_pbe_df")
                  for name, value in payload.get("species", {}).items()}
    rows = []
    for network in manifest["networks"]:
        idx = int(network["index"])
        channel = run_dir / "checkpoints" / f"spec_{idx:0{width}d}" / HOLDOUT_SUBDIR
        row = {"label": network.get("label"), "arch_name": network.get("arch_name"),
               "kind": network.get("kind"),
               "certificate": network.get("certificate_verdict")}
        if not (channel / "per_reaction.json").is_file():
            rows.append(row)
            continue
        molecules = json.loads((channel / "per_molecule.json").read_text(encoding="utf-8"))
        reactions = json.loads((channel / "per_reaction.json").read_text(encoding="utf-8"))
        row.update(network_metrics(molecules, reactions, pbe_df, subsets))
        row["n_species_not_df"] = n_not_df
        rows.append(row)
    return rows


def _fmt(value) -> str:
    if isinstance(value, float):
        return "nan" if math.isnan(value) else f"{value:.3f}"
    return "" if value is None else str(value)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir")
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir).resolve()
    rows = build_table(run_dir)
    out = run_dir / "slim16_metrics.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _fmt(row.get(key)) for key in COLUMNS})
    shown = ("label", "n_converged", "mae_te_df_all", "mae_te_df_converged",
             "wtmad2_paper_converged", "mae_rxn_ref_nn", "mae_rxn_ref_pbe_df",
             "df_footing_mean")
    print("  ".join(f"{key:>22s}" for key in shown))
    for row in rows:
        print("  ".join(f"{_fmt(row.get(key)):>22s}" for key in shown))
    print(f"written {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
