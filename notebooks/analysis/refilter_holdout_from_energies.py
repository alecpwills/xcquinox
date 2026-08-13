"""Re-derive corrected held-out ``eval_holdout/{per_reaction.json, test_set.csv}``
from the EXISTING per-molecule energies + the fixed molecule-level overlap.

The cluster-side held-out eval had two reaction-overlap bugs (atom-level overlap,
plus anion / cross-pool case-variant leakage; fixed in
``xcquinox.alec.eval_holdout``). Crucially the *per-molecule* energies it computed
are correct -- the bug was in which reactions were SELECTED into the held-out
metric, not in the SCF -- so the held-out per-reaction file and the test_set
summary can be regenerated WITHOUT re-running SCF: re-apply the fixed overlap to
the energies already in ``eval_holdout/per_molecule.json``.

This is the no-SCF analog of ``reeval_holdout_fixed.py`` for runs whose
per_molecule energies are already correct and only the overlap needs fixing, so
``make_ablation_arch_figure.py`` (which reads ``eval_holdout/per_reaction.json``)
renders the corrected parity / MAE figures.

Usage:
    python notebooks/analysis/refilter_holdout_from_energies.py --run-dir <run dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Set

from xcquinox.alec import eval_holdout as eh
from xcquinox.alec.eval_holdout import (
    filter_reactions, make_per_reaction_records, per_reaction_errors,
    reaction_mae_kcalmol, write_per_reaction_json, write_test_set_csv)
from xcquinox.alec.full_benchmark_pools import load_full_held_out_pools


def neutral_atom_names(held_out_specs: Dict) -> Set[str]:
    """Case-folded names of NEUTRAL monatomic anchors (held-out pool atoms via
    :func:`eval_holdout._spec_is_atom`) plus ``li`` -- a training anchor absent
    from the held-out pool. These are the ONLY training species excluded from the
    molecule-level overlap; anions (``f-``/``cl-``) and all multi-atom molecules
    are kept (see eval_holdout.training_molecule_names)."""
    names = {n.casefold() for n, s in held_out_specs.items() if eh._spec_is_atom(s)}
    names.add("li")
    return names


def training_molecule_names_from_meta(meta_names: List[str],
                                      neutral: Set[str]) -> Set[str]:
    """Molecule-level training names from ``train_metadata.json``'s species list:
    every species that is NOT a neutral monatomic anchor. Case-insensitive."""
    return {n for n in meta_names if n.casefold() not in neutral}


def refilter_spec(spec_dir: Path, full_rxns: List[Dict], neutral: Set[str]) -> Dict:
    """Regenerate one spec's corrected ``eval_holdout/{per_reaction.json,
    test_set.csv}`` (+ a provenance stamp); back up the cluster originals once."""
    eh_dir = spec_dir / "eval_holdout"
    pm = json.loads((eh_dir / "per_molecule.json").read_text())
    nn = {r["molecule"]: r["E_total_nn"] for r in pm
          if isinstance(r.get("E_total_nn"), (int, float))}
    pbe = {r["molecule"]: r["E_pbe"] for r in pm
           if isinstance(r.get("E_pbe"), (int, float))}
    meta_names = json.loads(
        (spec_dir / "train_metadata.json").read_text()).get("molecules", [])
    training_names = training_molecule_names_from_meta(meta_names, neutral)

    per_pool_mae: Dict[str, tuple] = {}
    all_kept: List[Dict] = []
    n_dropped_total = 0
    for pool in ("bh76", "w411"):
        prx = [r for r in full_rxns if r.get("source_pool") == pool]
        kept, dropped = filter_reactions(prx, training_names, strict=True)
        mae_nn, n_used, n_nan_nn = reaction_mae_kcalmol(nn, kept)
        mae_pbe, _, n_nan_pbe = reaction_mae_kcalmol(pbe, kept)
        per_pool_mae[pool] = (mae_nn, mae_pbe, n_used, len(dropped),
                              max(n_nan_nn, n_nan_pbe))
        all_kept.extend(kept)
        n_dropped_total += len(dropped)
    c_nn, c_used, c_nan_nn = reaction_mae_kcalmol(nn, all_kept)
    c_pbe, _, c_nan_pbe = reaction_mae_kcalmol(pbe, all_kept)
    combined = (c_nn, c_pbe, c_used, n_dropped_total, max(c_nan_nn, c_nan_pbe))

    nn_per = per_reaction_errors(nn, all_kept)
    pbe_per = per_reaction_errors(pbe, all_kept)
    records = make_per_reaction_records(all_kept, nn_per, pbe_per, training_names)

    # Back up the ORIGINAL cluster artifacts ONCE (never overwrite the backup).
    for name, stem, ext in (("per_reaction.json", "per_reaction", "json"),
                            ("test_set.csv", "test_set", "csv")):
        orig = eh_dir / name
        bak = eh_dir / f"{stem}.cluster_buggy.{ext}"
        if orig.is_file() and not bak.is_file():
            bak.write_bytes(orig.read_bytes())

    write_test_set_csv(eh_dir / "test_set.csv", per_pool_mae, combined, strict=True)
    write_per_reaction_json(eh_dir / "per_reaction.json", records)
    (eh_dir / "refilter_meta.json").write_text(json.dumps({
        "source": "refilter_holdout_from_energies (no SCF; reused per_molecule energies)",
        "overlap": "molecule-level, charge-aware, case-insensitive",
        "n_training_molecules": len(training_names),
        "n_held_out_reactions": len(all_kept),
        "n_dropped_overlap": n_dropped_total,
        "per_pool_mae": {k: list(v) for k, v in per_pool_mae.items()},
        "combined_mae": list(combined),
    }, indent=2) + "\n")
    return {"idx": spec_dir.name, "per_pool_mae": per_pool_mae, "combined": combined,
            "n_kept": len(all_kept), "training_names": sorted(training_names)}


def main(argv=None) -> int:
    print("REFUSING: this tool rewrites held-out artifacts under the "
          "RETIRED species-level strict-overlap rule. Held-out exclusion is "
          "now by VERBATIM supervised reaction (eval_holdout, 2026-08-13); "
          "running this on current pulls would clobber verbatim-rule "
          "artifacts with species-filtered ones. The figure layer "
          "reconstructs test slices from per-species energies directly -- "
          "no refilter step is needed. Pass --legacy-species-strict to run "
          "anyway on pre-2026-08 artifacts.")
    if "--legacy-species-strict" not in (argv if argv is not None
                                         else __import__("sys").argv[1:]):
        return 2
    argv = [a for a in (argv if argv is not None
                        else __import__("sys").argv[1:])
            if a != "--legacy-species-strict"]
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, help="pulled run dir")
    args = p.parse_args(argv)
    run_dir = Path(args.run_dir).expanduser().resolve()

    held_out_specs, full_rxns = load_full_held_out_pools()
    neutral = neutral_atom_names(held_out_specs)

    # Validate the name reconstruction against an authoritative .spec if present.
    specs_dir = run_dir / "specs"
    for sp in (sorted(specs_dir.glob("spec_*.spec")) if specs_dir.is_dir() else []):
        import pickle  # trusted local harness file (same as cluster/_eval_one_spec._load_spec)
        meta = run_dir / "checkpoints" / sp.stem / "train_metadata.json"
        if not meta.is_file():
            continue
        real = set(eh.training_molecule_names(pickle.loads(sp.read_bytes())))
        recon = training_molecule_names_from_meta(
            json.loads(meta.read_text()).get("molecules", []), neutral)
        if recon != real:
            print(f"FATAL: name reconstruction != real spec for {sp.stem}: "
                  f"{sorted(recon)} vs {sorted(real)}", file=sys.stderr)
            return 1
        print(f"validated reconstruction against {sp.name}: {sorted(real)}")
        break

    n = 0
    for sd in sorted((run_dir / "checkpoints").glob("spec_*")):
        if not (sd / "eval_holdout" / "per_molecule.json").is_file():
            continue
        r = refilter_spec(sd, full_rxns, neutral)
        b, w, c = r["per_pool_mae"]["bh76"], r["per_pool_mae"]["w411"], r["combined"]
        print(f"{r['idx']}: held-out {r['n_kept']:3d} rxns | "
              f"bh76 nn/pbe={b[0]:5.1f}/{b[1]:5.1f}  w411 nn/pbe={w[0]:5.1f}/{w[1]:5.1f}  "
              f"combined delta={c[0]-c[1]:+5.1f}")
        n += 1
    print(f"refiltered {n} specs in {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
