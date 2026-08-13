"""Refinalize completed held-out evals under the verbatim hold-out rule.

The per-species SCF energies a completed spec's eval wrote are correct and
rule-independent; the 2026-08-13 hold-out redefinition changed only which
reactions the reported test slice SELECTS (verbatim supervised reactions and
the recorded validation slice leave; species overlap stays). Specs evaluated
before the rule deployed therefore carry stale ``per_reaction.json`` /
``test_set.csv`` files that this module rewrites in place -- seconds per
spec, no SCF -- by re-running the finalize stage
(``eval_holdout._finalize_holdout_outputs``) on the energies already stored
in each channel's ``per_molecule.json``. Output is byte-equivalent to what a
fresh post-deployment eval writes for the same checkpoint.

Safety: the first rewrite of a channel backs up the previous artifacts as
``per_reaction.pre_verbatim.json`` / ``test_set.pre_verbatim.csv`` (never
overwritten once present); ``per_molecule.json`` is passed through unchanged.
Idempotent: a channel whose on-disk rows already equal the recomputed slice
is reported ``unchanged`` and not rewritten, so the run report doubles as
the ground-truth list of stale-rule specs. ``--dry-run`` computes every
report without writing anything.

Usage::

    python -m xcquinox.alec.refinalize_verbatim <run_dir> [<run_dir> ...] \
        [--channels eval_holdout eval_holdout_val_best] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

CHANNELS = ("eval_holdout", "eval_holdout_best", "eval_holdout_val_best")


class _MetadataSpec:
    """Duck-typed training-spec view over a pulled ``train_metadata.json``,
    exposing exactly what ``trained_reaction_exclusion`` consumes."""

    def __init__(self, meta: Dict[str, Any]):
        self._lk = dict(meta.get("loss_kwargs") or {})

    def loss_kwargs_dict(self) -> Dict[str, Any]:
        return self._lk


def _load_json(path: Path) -> Optional[Any]:
    if not path.is_file():
        return None
    try:
        with path.open() as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _annotation_names(meta: Dict[str, Any], pool_specs: Dict[str, Any]
                      ) -> Tuple[str, ...]:
    """Molecule-level training names + their pool aliases, for the
    informative ``in_sample_overlap`` annotation (matching what a fresh
    eval's serial path annotates; atoms are dropped as universal anchors)."""
    from xcquinox.alec.species_matching import (is_atomic,
                                                parse_formula_name,
                                                trained_pool_aliases)
    mol_level: List[str] = []
    for n in (meta.get("molecules") or []):
        parsed = parse_formula_name(str(n))
        if parsed is not None and is_atomic(parsed[0]):
            continue
        mol_level.append(str(n))
    if not mol_level:
        return ()
    aliases = trained_pool_aliases(mol_level, pool_specs, verbose=False)
    return tuple(sorted(set(mol_level) | aliases))


def _rows_equal(a: Optional[List[Dict[str, Any]]],
                b: Optional[List[Dict[str, Any]]],
                tol: float = 1e-12) -> bool:
    """Row-set equality of two per_reaction payloads: same name multiset and
    per-row numeric agreement within ``tol`` (non-numeric fields exact)."""
    if a is None or b is None:
        return a is b
    if len(a) != len(b):
        return False
    def _key(r):
        return (str(r.get("name")), str(r.get("pool")))
    sa = sorted(a, key=_key)
    sb = sorted(b, key=_key)
    for ra, rb in zip(sa, sb):
        if set(ra) != set(rb):
            return False
        for k in ra:
            va, vb = ra[k], rb[k]
            if isinstance(va, float) and isinstance(vb, float):
                if math.isnan(va) and math.isnan(vb):
                    continue
                if abs(va - vb) > tol:
                    return False
            elif va != vb:
                return False
    return True


def reactions_for_run(run_dir: Path,
                      pool_specs: Dict[str, Any],
                      pool_rxns: Sequence[Dict[str, Any]]
                      ) -> List[Dict[str, Any]]:
    """The run's reportable reaction list: the canonical pool minus the
    recorded validation slice, excluded by canonical identity (permuted-name
    twins leave with it). A missing ``validation/val_reactions.json`` means
    no exclusion -- mirroring a run that never validated. This is the
    file-presence form of the eval driver's spec-attribute gate."""
    from xcquinox.alec.species_matching import (canonical_species_keys,
                                                reaction_identity_keys)
    entries = _load_json(Path(run_dir) / "validation"
                         / "val_reactions.json") or []
    if not entries:
        return list(pool_rxns)
    key_map = canonical_species_keys(pool_specs)
    val_ids: set = set()
    for e in entries:
        val_ids.update(reaction_identity_keys(e, key_map))
    kept = []
    for r in pool_rxns:
        ids = set(reaction_identity_keys(r, key_map))
        if ids and ids & val_ids:
            continue
        kept.append(r)
    return kept


def refinalize_spec(spec_dir: Path,
                    pool_specs: Dict[str, Any],
                    reactions: Sequence[Dict[str, Any]], *,
                    channels: Sequence[str] = CHANNELS,
                    dry_run: bool = False) -> List[Dict[str, Any]]:
    """Refinalize one spec's channels; returns one report dict per channel:
    ``{spec, channel, status, n_old, n_new}`` with status ``rewritten``,
    ``unchanged``, ``would-rewrite`` (dry-run), or ``skipped-<reason>``."""
    from xcquinox.alec.eval_holdout import (_finalize_holdout_outputs,
                                            trained_reaction_exclusion)
    spec_dir = Path(spec_dir)
    meta = _load_json(spec_dir / "train_metadata.json")
    if meta is None:
        # Only worth a warning when there is something to refinalize: a
        # pending/untrained spec has neither metadata nor eval channels.
        if any((spec_dir / ch / "per_molecule.json").is_file()
               for ch in channels):
            print(f"[refinalize] WARNING: {spec_dir.name} has no readable "
                  "train_metadata.json -- no verbatim exclusion can be "
                  "built for it (validation exclusion still applies)",
                  flush=True)
        meta = {}
    excl, key_map = trained_reaction_exclusion(_MetadataSpec(meta),
                                               pool_specs)
    training_names = _annotation_names(meta, pool_specs)
    reports: List[Dict[str, Any]] = []
    for ch in channels:
        out_dir = spec_dir / ch
        pm = _load_json(out_dir / "per_molecule.json")
        rep = {"spec": spec_dir.name, "channel": ch,
               "n_old": None, "n_new": None}
        if pm is None:
            rep["status"] = "skipped-no-channel"
            reports.append(rep)
            continue
        e_nn = {str(r.get("molecule")): float(r["E_total_nn"]) for r in pm
                if isinstance(r.get("E_total_nn"), (int, float))}
        e_pbe = {str(r.get("molecule")): float(r["E_pbe"]) for r in pm
                 if isinstance(r.get("E_pbe"), (int, float))}
        if not e_nn or not e_pbe:
            rep["status"] = "skipped-no-energy-columns"
            reports.append(rep)
            continue
        old_rows = _load_json(out_dir / "per_reaction.json")
        rep["n_old"] = None if old_rows is None else len(old_rows)
        with tempfile.TemporaryDirectory() as td:
            _finalize_holdout_outputs(
                reactions, e_nn, e_pbe, mol_records=list(pm),
                training_names=training_names,
                n_species=len(pm), out_dir=Path(td), strict=True,
                excluded_identities=excl, species_key_map=key_map)
            new_rows = _load_json(Path(td) / "per_reaction.json") or []
            rep["n_new"] = len(new_rows)
            new_csv = (Path(td) / "test_set.csv").read_text()
            old_csv_path = out_dir / "test_set.csv"
            old_csv = (old_csv_path.read_text()
                       if old_csv_path.is_file() else None)
            # unchanged only when BOTH artifacts agree -- a crash between
            # the two writes must be healed by the next run, not reported
            # clean forever
            if _rows_equal(old_rows, new_rows) and old_csv == new_csv:
                rep["status"] = "unchanged"
                reports.append(rep)
                continue
            if dry_run:
                rep["status"] = "would-rewrite"
                reports.append(rep)
                continue
            for src, bak in (("per_reaction.json",
                              "per_reaction.pre_verbatim.json"),
                             ("test_set.csv", "test_set.pre_verbatim.csv")):
                s, b = out_dir / src, out_dir / bak
                if s.is_file() and not b.is_file():
                    shutil.copy2(s, b)
            # per_molecule.json is NOT rewritten (a proven byte-identical
            # pass-through: copying the energy record is pure risk); the two
            # derived artifacts land via same-directory temp + os.replace so
            # an interrupted run never leaves a truncated file.
            for name in ("per_reaction.json", "test_set.csv"):
                tmp = out_dir / f".tmp.{name}"
                shutil.copy2(Path(td) / name, tmp)
                os.replace(tmp, out_dir / name)
        rep["status"] = "rewritten"
        reports.append(rep)
    return reports


def refinalize_run(run_dir: Path, *,
                   channels: Sequence[str] = CHANNELS,
                   dry_run: bool = False,
                   allow_missing_validation: bool = False,
                   _pool: Optional[Tuple[Dict[str, Any],
                                         Sequence[Dict[str, Any]]]] = None
                   ) -> List[Dict[str, Any]]:
    """Refinalize every completed spec of ``run_dir``; prints one report
    line per non-skipped channel and a summary. ``_pool`` is a test seam."""
    run_dir = Path(run_dir)
    if not (run_dir / "validation" / "val_reactions.json").is_file() \
            and not allow_missing_validation:
        print(f"[refinalize] REFUSING {run_dir}: no validation/"
              "val_reactions.json -- refinalizing without it would report "
              "validation reactions as test rows. Pass "
              "--allow-missing-validation for a run that genuinely never "
              "validated.", flush=True)
        return [{"spec": None, "channel": None,
                 "status": "refused-no-validation-record",
                 "n_old": None, "n_new": None}]
    if _pool is not None:
        pool_specs, pool_rxns = _pool
    else:
        from xcquinox.alec.full_benchmark_pools import (
            load_full_held_out_pools)
        pool_specs, pool_rxns = load_full_held_out_pools()
    reactions = reactions_for_run(run_dir, pool_specs, list(pool_rxns))
    reports: List[Dict[str, Any]] = []
    for sd in sorted((run_dir / "checkpoints").glob("spec_*")):
        reports.extend(refinalize_spec(sd, pool_specs, reactions,
                                       channels=channels, dry_run=dry_run))
    n_re = sum(1 for r in reports
               if r["status"] in ("rewritten", "would-rewrite"))
    n_un = sum(1 for r in reports if r["status"] == "unchanged")
    for r in reports:
        print(f"[refinalize] {run_dir.name}/{r['spec']}/{r['channel']}: "
              f"{r['status']} ({r['n_old']} -> {r['n_new']} rows)")
    print(f"[refinalize] {run_dir}: {n_re} channel(s) "
          f"{'needing rewrite' if dry_run else 'rewritten'}, "
          f"{n_un} already verbatim-rule", flush=True)
    return reports


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("--channels", nargs="+", default=list(CHANNELS))
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--allow-missing-validation", action="store_true")
    args = p.parse_args(argv)
    rc = 0
    for rd in args.run_dirs:
        if not (Path(rd) / "checkpoints").is_dir():
            print(f"[refinalize] FATAL: {rd} has no checkpoints/ -- not a "
                  "run dir", flush=True)
            rc = 1
            continue
        reports = refinalize_run(
            Path(rd), channels=tuple(args.channels), dry_run=args.dry_run,
            allow_missing_validation=args.allow_missing_validation)
        if any(r["status"] == "refused-no-validation-record"
               for r in reports):
            rc = 1
    return rc


if __name__ == "__main__":
    sys.exit(main())
