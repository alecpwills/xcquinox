"""Common-slice v4-vs-v6 tables and the depth-symmetric v6 rescore.

Two jobs, both from PULLED ARTIFACTS only (no SCF):

1. ``--tables`` (default): score every trained cell of two runs on ONE
   union-excluded reaction slice. The published per-run numbers are not
   comparable across generations because each run's strict eval removes its
   OWN supervised reactions and its OWN validation slice (v4's name-keyed
   49-entry record vs v6's identity-keyed 35-entry record share only part
   of the pool), so a cross-run table built from per-run CSVs compares
   different denominators. Here the kept slice excludes, by reaction
   identity, the UNION over both runs of (every cell's supervised-reaction
   identities + the run's validation identities); under the STRICT recipe
   every cell of both runs is then scored on exactly that slice (uniform
   row count per cell), identity-deduped (one term per physical reaction),
   with the slice size and exclusion recipe stamped into the output. The
   VALIDATION-ONLY recipe is like-for-like only across runs' validation
   slices: each cell's strict eval already removed its own supervised
   rows, so per-cell row counts vary there (120-134 measured) and its
   tables are the reproduction of the previously reported figure, not a
   uniform-slice comparison.

2. The same tables carry the v6 depth channels side by side: the
   validation-best channel (3 SCF cycles from the converged PBE seed) and
   the cold-start channel (25 cycles from a functional-free minao seed, the
   Letter's depth). The PBE comparator in both is the CONVERGED PBE energy
   (the stored reference SCF), which is symmetric for neither channel; a
   depth-matched PBE leg (25-cycle minao PBE energies per species) is a
   separate compute job (see ``--emit-pbe-depth-cmd``) and its output JSON
   is folded in when present.

Outputs under ``--out``: ``common_slice.json`` (provenance: slice size,
excluded identity counts per source, runs, channels), ``common_slice.csv``
(per cell x channel x pool), and ``common_slice_tables.md`` (the tables the
reports cite).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Pool + identity plumbing (the evaluation layer's own machinery)
# ---------------------------------------------------------------------------

_POOL_CACHE: Optional[Tuple[dict, list, dict]] = None


def _pool_and_key_map():
    """(pool_specs, pool_rxns, key_map): the canonical 216-reaction benchmark
    pool and the composition-level species key map, both from the library."""
    global _POOL_CACHE
    if _POOL_CACHE is None:
        from xcquinox.alec.full_benchmark_pools import load_full_held_out_pools
        from xcquinox.alec.species_matching import canonical_species_keys
        specs, rxns = load_full_held_out_pools(basis="def2-svp", grid_level=1)
        _POOL_CACHE = (specs, rxns, canonical_species_keys(specs))
    return _POOL_CACHE


def _row_identities(row: Dict[str, Any], key_map: Dict) -> Tuple:
    from xcquinox.alec.species_matching import reaction_identity_keys
    return reaction_identity_keys(row, key_map)


class _MetadataSpec:
    """Duck-typed spec over a train_metadata.json payload, the shape
    eval_holdout.trained_reaction_exclusion consumes."""

    def __init__(self, meta: Dict[str, Any]):
        self._meta = dict(meta or {})
        mols = self._meta.get("molecules") or []
        self.molecules = tuple(type("M", (), {"name": str(n)})()
                               for n in mols)
        lk = self._meta.get("loss_kwargs") or {}
        self.loss_kwargs = lk


def _val_record_entries(run_dir: Path) -> List[Dict[str, Any]]:
    """The run's validation-slice entries, from either layout:
    ``validation/val_reactions.json`` (v5+) or the run-root
    ``val_reactions.json`` (v4 era). Empty list when neither exists."""
    for cand in (run_dir / "validation" / "val_reactions.json",
                 run_dir / "val_reactions.json"):
        if cand.is_file():
            try:
                payload = json.loads(cand.read_text())
            except (json.JSONDecodeError, OSError):
                continue
            if isinstance(payload, dict):
                payload = payload.get("reactions", [])
            return list(payload)
    return []


def _spec_dirs(run_dir: Path) -> List[Tuple[int, Path]]:
    out = []
    ck = run_dir / "checkpoints"
    if not ck.is_dir():
        return out
    for d in sorted(ck.glob("spec_*")):
        try:
            out.append((int(d.name.split("_")[1]), d))
        except (IndexError, ValueError):
            continue
    return out


def _cells(run_dir: Path) -> Dict[int, Dict[str, Any]]:
    mf = run_dir / "manifest.json"
    try:
        manifest = json.loads(mf.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
    return {int(s["index"]): dict(s.get("cell", {}))
            for s in manifest.get("specs", [])}


def excluded_identities_for_run(run_dir: Path) -> Dict[str, set]:
    """{'validation': ids, 'supervised': ids} for one run: the validation
    record's reaction identities plus the union over every spec of its
    trained-reaction exclusion identities (the same builder the strict eval
    uses)."""
    from xcquinox.alec.eval_holdout import trained_reaction_exclusion
    pool_specs, _rxns, key_map = _pool_and_key_map()

    val_ids: set = set()
    for e in _val_record_entries(run_dir):
        val_ids.update(_row_identities(e, key_map))

    sup_ids: set = set()
    for _idx, sd in _spec_dirs(run_dir):
        tm = sd / "train_metadata.json"
        if not tm.is_file():
            continue
        try:
            meta = json.loads(tm.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        excl, _km = trained_reaction_exclusion(_MetadataSpec(meta),
                                               pool_specs)
        sup_ids.update(excl or set())
    return {"validation": val_ids, "supervised": sup_ids}


def common_slice_identities(run_dirs: List[Path],
                            include_supervised: bool = True
                            ) -> Dict[str, Any]:
    """The union-excluded common slice over ``run_dirs``: kept pool-reaction
    identities plus the provenance counts.

    ``include_supervised=True`` (the STRICT recipe) excludes each run's
    per-cell trained-reaction identities as well as its validation slice
    (v4gga + v6 G1: 119 identities / 120 rows). ``False`` (the
    VALIDATION-ONLY recipe) excludes only the two validation slices -- the
    recipe behind the previously reported "134-reaction common slice" (133
    identities / 134 rows): like-for-like across runs, but containing the
    24 supervised-reaction identities both runs trained on."""
    _specs, pool_rxns, key_map = _pool_and_key_map()
    excluded: set = set()
    prov: Dict[str, Any] = {"runs": [],
                            "recipe_name": ("strict" if include_supervised
                                            else "validation_only")}
    for rd in run_dirs:
        ex = excluded_identities_for_run(rd)
        excluded |= ex["validation"]
        if include_supervised:
            excluded |= ex["supervised"]
        prov["runs"].append({
            "run": str(rd),
            "n_validation_identities": len(ex["validation"]),
            "n_supervised_identities": len(ex["supervised"]),
        })
    kept: Dict[Tuple, str] = {}
    n_pool_ident = 0
    seen: set = set()
    for r in pool_rxns:
        ids = _row_identities(r, key_map)
        if not ids or ids in seen:
            continue
        seen.add(ids)
        n_pool_ident += 1
        if not (set(ids) & excluded):
            kept[ids] = r.get("source_pool") or r.get("pool") or "unknown"
    prov.update({
        "n_pool_reaction_identities": n_pool_ident,
        "n_excluded_identities": len(excluded),
        "n_common_slice": len(kept),
        "recipe": ("pool identities minus the union over runs of "
                   + ("(per-cell trained-reaction exclusions + validation "
                      "slice)" if include_supervised
                      else "(validation slices only; supervised reactions "
                           "REMAIN in the slice)")
                   + ", all at reaction-identity level"),
    })
    return {"kept": kept, "provenance": prov}


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _identity_dedup_mae(rows: List[Dict[str, Any]], err_key: str,
                        key_map: Dict) -> Tuple[float, int]:
    """(MAE, n_identities) over rows, one term per reaction identity
    (matching eval_holdout.reaction_mae_kcalmol's convention)."""
    groups: Dict[Tuple, List[float]] = {}
    for i, r in enumerate(rows):
        v = r.get(err_key)
        key = _row_identities(r, key_map) or ("__row__", i)
        groups.setdefault(key, []).append(
            float(v) if isinstance(v, (int, float)) else float("nan"))
    terms = []
    for vals in groups.values():
        finite = [v for v in vals if math.isfinite(v)]
        if finite:
            terms.append(sum(finite) / len(finite))
    if not terms:
        return float("nan"), 0
    return sum(terms) / len(terms), len(terms)


def score_run_on_slice(run_dir: Path, channel: str, kept: Dict[Tuple, str]
                       ) -> List[Dict[str, Any]]:
    """Per-cell rows: identity-deduped NN and PBE MAEs per pool + combined,
    over exactly the kept identities present in the cell's per_reaction."""
    _specs, _rxns, key_map = _pool_and_key_map()
    cells = _cells(run_dir)
    out: List[Dict[str, Any]] = []
    for idx, sd in _spec_dirs(run_dir):
        pr = sd / channel / "per_reaction.json"
        if not pr.is_file():
            continue
        try:
            rows = json.loads(pr.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        in_slice = [r for r in rows
                    if _row_identities(r, key_map) in kept]
        cell = cells.get(idx, {})
        rec: Dict[str, Any] = {
            "run": run_dir.name, "channel": channel, "idx": idx,
            "arch": cell.get("arch"), "subset_size": cell.get("subset_size"),
            "n_slice_rows": len(in_slice),
        }
        for pool_label, sub in (
                ("bh76", [r for r in in_slice
                          if (r.get("pool") or r.get("source_pool")) == "bh76"]),
                ("w411", [r for r in in_slice
                          if (r.get("pool") or r.get("source_pool")) == "w411"]),
                ("combined", in_slice)):
            mae_nn, n_nn = _identity_dedup_mae(
                sub, "abs_error_nn_kcalmol", key_map)
            mae_pbe, n_pbe = _identity_dedup_mae(
                sub, "abs_error_pbe_kcalmol", key_map)
            rec[f"{pool_label}_mae_nn"] = mae_nn
            rec[f"{pool_label}_mae_pbe"] = mae_pbe
            rec[f"{pool_label}_n"] = n_nn
            rec[f"{pool_label}_n_pbe"] = n_pbe
        out.append(rec)
    return out


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------

def _fmt(x: Any) -> str:
    if isinstance(x, float):
        return "nan" if not math.isfinite(x) else f"{x:.2f}"
    return str(x)


def write_tables(out_dir: Path, slice_info: Dict[str, Any],
                 scored: List[Dict[str, Any]],
                 pbe_depth: Optional[Dict[str, Any]]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    prov = slice_info["provenance"]
    (out_dir / "common_slice.json").write_text(json.dumps(
        {"provenance": prov,
         "cells": scored,
         "pbe_depth_leg": (pbe_depth or {}).get("provenance")},
        indent=1))

    cols = ["run", "channel", "idx", "arch", "subset_size", "n_slice_rows",
            "bh76_mae_nn", "bh76_mae_pbe", "bh76_n", "bh76_n_pbe",
            "w411_mae_nn", "w411_mae_pbe", "w411_n", "w411_n_pbe",
            "combined_mae_nn", "combined_mae_pbe", "combined_n",
            "combined_n_pbe"]
    with (out_dir / "common_slice.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for rec in scored:
            w.writerow(rec)

    lines: List[str] = []
    lines.append("# Common-slice comparison (union-excluded, "
                 "identity-deduped)\n")
    lines.append(f"Slice: {prov['n_common_slice']} of "
                 f"{prov['n_pool_reaction_identities']} pool reaction "
                 f"identities kept; {prov['n_excluded_identities']} excluded "
                 f"({prov['recipe']}).\n")
    for r in prov["runs"]:
        lines.append(f"- {r['run']}: {r['n_supervised_identities']} "
                     f"supervised + {r['n_validation_identities']} "
                     f"validation identities contributed to the union.")
    lines.append("")
    lines.append("PBE comparator: the stored CONVERGED reference-SCF energy "
                 "in every channel below. The 3-cycle NN channels are seeded "
                 "from that converged PBE; the cold-start channel runs 25 "
                 "cycles from minao -- depth-matched PBE energies "
                 + ("are folded in below." if pbe_depth else
                    "are NOT yet computed (see --emit-pbe-depth-cmd), so "
                    "the cold-start rows compare a 25-cycle NN against a "
                    "fully converged PBE.") + "\n")
    for (run, channel) in sorted({(r["run"], r["channel"]) for r in scored}):
        sub = [r for r in scored
               if r["run"] == run and r["channel"] == channel]
        if not sub:
            continue
        lines.append(f"## {run} -- {channel}\n")
        lines.append("| arch | ss | BH76 NN | BH76 PBE | W4-11 NN | "
                     "W4-11 PBE | comb NN | comb PBE | n | n_pbe |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for r in sorted(sub, key=lambda x: (str(x["arch"]),
                                            int(x["subset_size"] or 0))):
            lines.append(
                "| " + " | ".join(_fmt(r[k]) for k in (
                    "arch", "subset_size",
                    "bh76_mae_nn", "bh76_mae_pbe",
                    "w411_mae_nn", "w411_mae_pbe",
                    "combined_mae_nn", "combined_mae_pbe",
                    "combined_n", "combined_n_pbe")) + " |")
        lines.append("")
    (out_dir / "common_slice_tables.md").write_text("\n".join(lines) + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--v6-run", required=True)
    ap.add_argument("--v4-run", required=True)
    ap.add_argument("--out", default=str(Path(__file__).parent
                                         / "rescore_depth_symmetric_out"))
    ap.add_argument("--v6-channels", nargs="+",
                    default=["eval_holdout_val_best",
                             "eval_holdout_coldstart"])
    ap.add_argument("--v4-channels", nargs="+",
                    default=["eval_holdout_val_best"])
    ap.add_argument("--pbe-depth-json", default=None,
                    help="Output of the depth-matched PBE leg job; folded "
                         "into the tables when present.")
    ap.add_argument("--emit-pbe-depth-cmd", action="store_true",
                    help="Print the compute job for the 25-cycle minao PBE "
                         "leg (cluster-scale; not run here) and exit.")
    args = ap.parse_args(argv)

    if args.emit_pbe_depth_cmd:
        print("# Depth-matched PBE leg (214 species x 25-cycle minao PBE at"
              " the production basis): cluster job, not run locally.")
        print("# Submit via the harness eval override on any trained spec"
              " with a PBE-functional model, or as a standalone script;"
              " write per-species energies to pbe_depth25.json and pass"
              " --pbe-depth-json.")
        return 0

    v6 = Path(args.v6_run)
    v4 = Path(args.v4_run)
    pbe_depth = None
    if args.pbe_depth_json and Path(args.pbe_depth_json).is_file():
        pbe_depth = json.loads(Path(args.pbe_depth_json).read_text())
    for include_supervised, subdir in ((True, "strict"),
                                       (False, "validation_only")):
        sl = common_slice_identities([v4, v6],
                                     include_supervised=include_supervised)
        print(f"{sl['provenance']['recipe_name']} slice: "
              f"{sl['provenance']['n_common_slice']} of "
              f"{sl['provenance']['n_pool_reaction_identities']} identities")
        scored: List[Dict[str, Any]] = []
        for ch in args.v4_channels:
            scored += score_run_on_slice(v4, ch, sl["kept"])
        for ch in args.v6_channels:
            scored += score_run_on_slice(v6, ch, sl["kept"])
        write_tables(Path(args.out) / subdir, sl, scored, pbe_depth)
        print(f"wrote {args.out}/{subdir}/common_slice_tables.md "
              f"({len(scored)} cell x channel rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
