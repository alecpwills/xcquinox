"""The metrics table of a Slim16 evaluation run (hpcjobs/slim16_eval.py).

Per network: the converged species count, the total-energy MAE against PBE on
the run's own footing (the PBE-DF table beside the run; over every species and
over the converged ones), the same against the precompute's full-integral PBE,
the cloning paper's WTMAD-2 against PBE-DF (over the reactions whose species
all converged, the paper's filter, and over all), the reaction MAE against the
GMTKN55 references for the network, PBE-DF and full-integral PBE, the paper's
WTMAD-2 form against the GMTKN55 references with the full set's weights (the
network over every reaction it reports and over the converged ones alone,
PBE-DF over every reaction its table covers; the scale is the full set's own
mean absolute reference, not the GMTKN55 constant), and the size of the
density-fitting footing (the two PBE energies). Energies in kcal/mol.

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
    "wtmad2_ref_nn_all", "wtmad2_ref_nn_converged", "wtmad2_ref_pbe_df",
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


def reaction_rows(molecules: list, reactions: list, pbe_df: dict,
                  subsets: dict) -> tuple:
    """``(rows_all, rows_converged)``: the ``paper_wtmad2`` rows of one
    network, the reference leg PBE-DF (the paper's footing) and the
    converged-only filter the paper applies -- a reaction enters the
    converged set when every species it names reports ``scf_converged``.

    The rows behind the table's PBE-DF columns; a reaction whose subset the
    pool does not name reads ``'?'``.
    """
    converged = {m["molecule"] for m in molecules if m.get("scf_converged")}
    rows_all, rows_converged = [], []
    for record in reactions:
        de_ref = reaction_energy(record, pbe_df)
        de_nn = record.get("de_nn_kcalmol")
        row = {"subset": subsets.get(record["name"], "?"), "de_nn": de_nn,
               "de_ref": de_ref}
        rows_all.append(row)
        names = list(record.get("reactants", [])) + list(record.get("products", []))
        if all(n in converged for n in names):
            rows_converged.append(row)
    return rows_all, rows_converged


def _species_of(record: dict) -> list:
    return list(record.get("reactants", [])) + list(record.get("products", []))


def reference_rows(molecules: list, reactions: list, pbe_df: dict,
                   subsets: dict) -> tuple:
    """``(rows_nn, rows_pbe_df)``: one row per reaction of the channel, for
    :func:`wtmad2_full_weights`, against the GMTKN55 reference of the
    reaction (``reaction_energy_ref_kcalmol``): ``name``, ``subset``,
    ``de_ref``, ``de_nn`` (the network's reported reaction energy, or the
    PBE-DF table's) and ``converged`` (every species of the reaction reports
    ``scf_converged``; always true for PBE-DF, whose table carries no
    convergence). Nothing is filtered here: a reaction without a reference
    or without an energy carries a non-finite value, and the scorer names
    the second kind under ``unscored``.
    """
    converged = {m["molecule"] for m in molecules if m.get("scf_converged")}
    rows_nn, rows_pbe = [], []
    for record in reactions:
        name = record["name"]
        subset = subsets.get(name, "?")
        reference = record.get("reaction_energy_ref_kcalmol")
        rows_nn.append({"name": name, "subset": subset, "de_ref": reference,
                        "de_nn": record.get("de_nn_kcalmol"),
                        "converged": all(n in converged for n in _species_of(record))})
        rows_pbe.append({"name": name, "subset": subset, "de_ref": reference,
                         "de_nn": reaction_energy(record, pbe_df),
                         "converged": True})
    return rows_nn, rows_pbe


def wtmad2_full_weights(rows: list) -> dict:
    """The paper's WTMAD-2 form over the rows with the FULL set's weights,
    twice: ``all`` over every reaction as reported, converged or not, and
    ``converged`` over the converged reactions alone, the weights (N_i, m_i,
    N and the scale) staying those of the full set. The full set is the rows
    carrying a finite reference; a row without one is out of everything. A
    row with a reference but no finite energy (an evaluation that raised)
    keeps its weight, enters no mean and is named in ``unscored``; the
    unconverged reactions are named in ``unconverged``. A subset with no
    converged reaction is dropped from the converged sum, its weight is not
    given to the others, and it is named in ``dropped``; a subset with no
    scored reaction is out of both sums. A subset whose references average
    to zero carries no weight: its term is out of both sums, its reactions
    still count in N, and it is named in ``unweighted``. ``per_subset``
    carries ``N_i``, ``m_i`` (mean absolute reference), ``MAD_all`` and
    ``MAD_converged`` (None when the subset is out of that sum),
    ``n_scored`` and ``n_converged``. A total with no contributing subset is
    NaN; ``n_reactions`` is N.
    """
    referenced = [row for row in rows if _finite(row.get("de_ref"))]
    by_subset: dict = {}
    for row in referenced:
        by_subset.setdefault(str(row["subset"]), []).append(row)
    per_subset, dropped, unweighted = {}, [], []
    for subset in sorted(by_subset):
        group = by_subset[subset]
        n_i = len(group)
        scored = [row for row in group if _finite(row.get("de_nn"))]
        errors = [abs(row["de_ref"] - row["de_nn"]) for row in scored]
        converged = [abs(row["de_ref"] - row["de_nn"]) for row in scored
                     if row.get("converged")]
        m_i = sum(abs(row["de_ref"]) for row in group) / n_i
        per_subset[subset] = {
            "N_i": n_i,
            "m_i": m_i,
            "MAD_all": sum(errors) / len(errors) if errors else None,
            "MAD_converged": (sum(converged) / len(converged)
                              if converged else None),
            "n_scored": len(scored),
            "n_converged": len(converged),
        }
        if not converged:
            dropped.append(subset)
        if m_i <= 0:
            unweighted.append(subset)
    n_total = sum(entry["N_i"] for entry in per_subset.values())
    unconverged = sorted(row["name"] for row in referenced if not row.get("converged"))
    unscored = sorted(row["name"] for row in referenced if not _finite(row.get("de_nn")))
    out = {"all": float("nan"), "converged": float("nan"),
           "per_subset": per_subset, "dropped": dropped, "unconverged": unconverged,
           "unscored": unscored, "unweighted": unweighted, "n_reactions": n_total}
    if not n_total:
        return out
    scale = (sum(e["N_i"] * e["m_i"] for e in per_subset.values()) / n_total) / n_total

    def _total(key):
        terms = [e["N_i"] * e[key] / e["m_i"] for e in per_subset.values()
                 if e[key] is not None and e["m_i"] > 0]
        return scale * sum(terms) if terms else float("nan")

    out["all"] = float(_total("MAD_all"))
    out["converged"] = float(_total("MAD_converged"))
    return out


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
    rows_all, rows_converged = reaction_rows(molecules, reactions, pbe_df, subsets)
    ref_rows = []
    for record in reactions:
        ref_rows.append((record.get("de_nn_kcalmol"),
                         reaction_energy(record, pbe_df),
                         record.get("de_pbe_kcalmol"),
                         record.get("reaction_energy_ref_kcalmol")))
    wt_conv, _ = paper_wtmad2(rows_converged)
    wt_all, _ = paper_wtmad2(rows_all)
    ref_rows_nn, ref_rows_pbe = reference_rows(molecules, reactions, pbe_df, subsets)
    scored_nn = wtmad2_full_weights(ref_rows_nn)
    scored_pbe = wtmad2_full_weights(ref_rows_pbe)
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
        "wtmad2_ref_nn_all": scored_nn["all"],
        "wtmad2_ref_nn_converged": scored_nn["converged"],
        "wtmad2_ref_pbe_df": scored_pbe["all"],
    }


def run_context(run_dir: Path) -> tuple:
    """``(manifest, width, pbe_df, subsets, n_species_not_df)``: what every
    reader of the run's channels starts from -- the run's manifest and its
    index width, the PBE-DF footing table (name -> energy in Hartree, empty
    when the run wrote none), the pool's reaction -> subset map, and the
    not-density-fitted species count (None when the table is absent)."""
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
    return manifest, width, pbe_df, subsets, n_not_df


def channel_records(run_dir: Path, width: int, idx: int):
    """``(molecules, reactions)`` of network ``idx``'s holdout channel, or
    None when the channel was never evaluated (no ``per_reaction.json``)."""
    channel = run_dir / "checkpoints" / f"spec_{idx:0{width}d}" / HOLDOUT_SUBDIR
    if not (channel / "per_reaction.json").is_file():
        return None
    molecules = json.loads((channel / "per_molecule.json").read_text(encoding="utf-8"))
    reactions = json.loads((channel / "per_reaction.json").read_text(encoding="utf-8"))
    return molecules, reactions


def build_table(run_dir: Path) -> list:
    manifest, width, pbe_df, subsets, n_not_df = run_context(run_dir)
    rows = []
    for network in manifest["networks"]:
        idx = int(network["index"])
        row = {"label": network.get("label"), "arch_name": network.get("arch_name"),
               "kind": network.get("kind"),
               "certificate": network.get("certificate_verdict")}
        records = channel_records(run_dir, width, idx)
        if records is None:
            rows.append(row)
            continue
        molecules, reactions = records
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
             "wtmad2_paper_converged", "wtmad2_ref_nn_all",
             "wtmad2_ref_nn_converged", "wtmad2_ref_pbe_df",
             "mae_rxn_ref_nn", "df_footing_mean")
    print("  ".join(f"{key:>22s}" for key in shown))
    for row in rows:
        print("  ".join(f"{_fmt(row.get(key)):>22s}" for key in shown))
    print(f"written {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
