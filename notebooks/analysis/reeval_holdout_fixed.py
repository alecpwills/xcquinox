#!/usr/bin/env python
"""Idempotent re-run of the FULL held-out eval (BH76 + W4-11) on pulled
checkpoints, using the corrected GMTKN55 geometries (2026-05-31 units fix).

WHY: `xcquinox/alec/full_benchmark_pools.py` previously mislabeled the GMTKN55
`struc.xyz` geometries as Bohr and divided them by 1.8897, shrinking every
held-out molecule ~1.89x. That made every held-out reaction energy garbage
(W4-11 atomizations came out negative; BH76 barriers ~20x too large) for BOTH
the NN and PBE channels. The library fix corrects the geometry; this script
regenerates each spec's `checkpoints/spec_<NNNN>/eval_holdout/{test_set.csv,
per_molecule.json,per_reaction.json}` — the same artifacts the cluster eval
writes — by calling `eval_holdout.run_full_holdout_eval` with the corrected
`full_benchmark_pools.load_full_held_out_pools`.

IDEMPOTENT / RE-RUNNABLE: a sidecar `eval_holdout/reeval_meta.json` records the
fix version. A spec is (re)processed iff it has a `model.eqx` AND lacks the
current stamp — so cluster-written / stale outputs are redone, already-fixed
specs are skipped, and specs that finish or download LATER are picked up on the
next run. Safe to run repeatedly as the sweep completes.

Usage:
    python notebooks/analysis/reeval_holdout_fixed.py \
        [--run-dir <pulled run dir>] [--specs 0,1,5] [--force]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

#: Bump when a future correctness fix / schema change should force re-processing
#: of every spec. v2 (2026-05-31): per-molecule per-SCF-step convergence trace
#: (scf_energy_step_<i> / scf_energy_residual_<i>) added to per_molecule.json.
REEVAL_VERSION = "geom_units_fix_v2"

_STAMP_NAME = "reeval_meta.json"
_DEFAULT_LOCAL_ROOT = Path.home() / "Documents/Research/xcquinox-results/runs"
_DEFAULT_CATEGORY = "ablation_notransform/polarized/runs"
_DEFAULT_BASIS = "def2-svp"
_DEFAULT_GRID_LEVEL = 1


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested without any heavy import)
# ---------------------------------------------------------------------------

def manifest_width(run_dir: Path) -> int:
    """Spec-index zero-pad width from manifest.json (default 4)."""
    mpath = run_dir / "manifest.json"
    if mpath.is_file():
        try:
            return int(json.loads(mpath.read_text()).get("width", 4))
        except (json.JSONDecodeError, OSError, ValueError):
            pass
    return 4


def discover_trained_specs(run_dir: Path) -> List[int]:
    """Sorted spec indices that have a materialized ``model.eqx``."""
    ck = run_dir / "checkpoints"
    if not ck.is_dir():
        return []
    out: List[int] = []
    for sd in sorted(ck.glob("spec_*")):
        if sd.is_dir() and (sd / "model.eqx").is_file():
            try:
                out.append(int(sd.name[len("spec_"):]))
            except ValueError:
                continue
    return sorted(set(out))


def read_stamp(spec_dir: Path) -> Optional[str]:
    """Return the recorded geom-fix version for a spec, or None."""
    p = spec_dir / "eval_holdout" / _STAMP_NAME
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text()).get("geom_units_fix")
    except (json.JSONDecodeError, OSError):
        return None


def needs_reeval(spec_dir: Path, *, version: str = REEVAL_VERSION,
                 force: bool = False) -> bool:
    """A spec needs re-eval iff it has a model.eqx AND (force OR its stamp does
    not match the current fix version)."""
    if not (spec_dir / "model.eqx").is_file():
        return False
    if force:
        return True
    return read_stamp(spec_dir) != version


def write_stamp(spec_dir: Path, summary: Dict[str, Any], *,
                version: str = REEVAL_VERSION) -> Path:
    """Record the fix version + a small eval summary next to the artifacts."""
    out = spec_dir / "eval_holdout"
    out.mkdir(parents=True, exist_ok=True)
    p = out / _STAMP_NAME
    payload = {
        "geom_units_fix": version,
        "n_reactions": summary.get("n_reactions"),
        "n_species": summary.get("n_species"),
        "n_dropped_nan": summary.get("n_dropped_nan"),
    }
    p.write_text(json.dumps(payload, indent=1))
    return p


def _fmt_eta(seconds: float) -> str:
    seconds = int(max(0.0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


# ---------------------------------------------------------------------------
# Run-dir resolution
# ---------------------------------------------------------------------------

def resolve_run_dir(run_dir: Optional[str]) -> Path:
    if run_dir:
        return Path(run_dir).expanduser().resolve()
    cand = _DEFAULT_LOCAL_ROOT / _DEFAULT_CATEGORY
    if cand.is_dir():
        runs = sorted(p for p in cand.glob("run_*") if p.is_dir())
        if runs:
            return runs[-1].resolve()
    raise SystemExit(
        f"No run dir found under {cand}; pass --run-dir explicitly.")


def _read_basis_grid(run_dir: Path) -> tuple[str, int]:
    """Read (basis, grid_level) from resolved_config.yaml; fall back to the
    cluster defaults. Tiny hand-parse to avoid a yaml dependency."""
    basis, grid = _DEFAULT_BASIS, _DEFAULT_GRID_LEVEL
    cfg = run_dir / "resolved_config.yaml"
    if cfg.is_file():
        for line in cfg.read_text().splitlines():
            t = line.strip()
            if t.startswith("basis:"):
                basis = t.split(":", 1)[1].strip().strip("'\"") or basis
            elif t.startswith("grid_level:"):
                try:
                    grid = int(t.split(":", 1)[1].strip())
                except ValueError:
                    pass
    return basis, grid


# ---------------------------------------------------------------------------
# Per-spec + driver (heavy deps injected for testability)
# ---------------------------------------------------------------------------

def _real_pools_loader(basis: str, grid_level: int):
    from xcquinox.alec.full_benchmark_pools import load_full_held_out_pools
    return load_full_held_out_pools(basis=basis, grid_level=grid_level)


def _real_spec_loader(spec_path: Path):
    from xcquinox.alec.eval_holdout import load_training_spec
    return load_training_spec(spec_path)


def _real_model_loader(training_spec, model_path: Path):
    from xcquinox.alec.eval_holdout import load_trained_model
    return load_trained_model(training_spec, model_path)


def _real_precompute_fn(training_spec, mol_specs):
    from xcquinox.alec.eval_holdout import precompute_holdout_for_spec
    return precompute_holdout_for_spec(training_spec, mol_specs)


def _real_eval_fn(training_spec, model, mol_specs, reactions, out_dir, mol_data):
    from xcquinox.alec.eval_holdout import run_full_holdout_eval
    return run_full_holdout_eval(
        training_spec, model, mol_specs, reactions, out_dir, mol_data=mol_data)


def descriptor_signature(training_spec) -> tuple:
    """Group key: ``(descriptor-type-names, solver-mode)``. Specs sharing this
    signature share an identical PBE/grid/eri precompute, so we precompute once
    per signature and reuse it across the group — the optimization that turns
    an N-spec re-eval from N precomputes into one-per-group."""
    try:
        descs = tuple(type(d).__name__
                      for d in training_spec.arch.materialize_descriptors())
    except AttributeError:
        descs = ()
    mode = getattr(getattr(training_spec, "solver_config", None), "mode", None)
    return (descs, getattr(mode, "name", str(mode)))


def run(
    run_dir: Path, *, force: bool = False,
    only_specs: Optional[Sequence[int]] = None,
    clock: Callable[[], float] = time.perf_counter,
    pools_loader: Callable = _real_pools_loader,
    spec_loader: Callable = _real_spec_loader,
    model_loader: Callable = _real_model_loader,
    precompute_fn: Callable = _real_precompute_fn,
    eval_fn: Callable = _real_eval_fn,
) -> Dict[str, Any]:
    """Idempotently re-eval every trained spec that lacks the current stamp.

    Specs are grouped by :func:`descriptor_signature`; the expensive PBE/grid/
    eri precompute runs ONCE per group and is reused across the group's specs
    (only the cheap per-model NN eval repeats). Heavy callables are injectable
    for tests. Returns ``{processed, skipped, failed}`` spec-index lists."""
    width = manifest_width(run_dir)
    basis, grid_level = _read_basis_grid(run_dir)
    trained = discover_trained_specs(run_dir)
    if only_specs is not None:
        wanted = set(only_specs)
        trained = [i for i in trained if i in wanted]

    def _spec_dir(idx: int) -> Path:
        return run_dir / "checkpoints" / f"spec_{idx:0{width}d}"

    todo = [i for i in trained if needs_reeval(_spec_dir(i), force=force)]
    skipped = [i for i in trained if i not in set(todo)]

    # Load each todo spec's training_spec once, and group by descriptor sig.
    specs_by_idx: Dict[int, Any] = {}
    groups: Dict[tuple, List[int]] = {}
    for idx in todo:
        ts = spec_loader(run_dir / "specs" / f"spec_{idx:0{width}d}.spec")
        specs_by_idx[idx] = ts
        groups.setdefault(descriptor_signature(ts), []).append(idx)

    print(f"[reeval] run_dir={run_dir.name}  basis={basis} grid={grid_level}  "
          f"trained={len(trained)}  to-process={len(todo)} in {len(groups)} "
          f"descriptor-group(s)  already-fixed={len(skipped)}  "
          f"(version {REEVAL_VERSION})", flush=True)

    mol_specs, reactions = pools_loader(basis, grid_level)

    processed: List[int] = []
    failed: List[int] = []
    t0 = clock()
    n_done = 0
    for sig, idxs in groups.items():
        # One precompute for the whole group (reused across its specs).
        rep_ts = specs_by_idx[idxs[0]]
        print(f"[reeval] group {sig} — {len(idxs)} spec(s); precomputing once "
              f"...", flush=True)
        try:
            mol_data = precompute_fn(rep_ts, mol_specs)
        except Exception as exc:  # noqa: BLE001 - whole group fails together
            failed.extend(idxs)
            print(f"[reeval]   group {sig} precompute FAILED: "
                  f"{type(exc).__name__}: {exc}  (skipping {len(idxs)} specs)",
                  flush=True)
            n_done += len(idxs)
            continue

        for idx in idxs:
            n_done += 1
            elapsed = clock() - t0
            eta = (elapsed / (n_done - 1) * (len(todo) - (n_done - 1))
                   if n_done > 1 else 0.0)
            print(f"[reeval] ({n_done}/{len(todo)}) spec {idx} "
                  f"[elapsed {_fmt_eta(elapsed)}, ETA {_fmt_eta(eta)}] ...",
                  flush=True)
            spec_dir = _spec_dir(idx)
            model_path = spec_dir / "model.eqx"
            try:
                model = model_loader(specs_by_idx[idx], model_path)
                summary = eval_fn(specs_by_idx[idx], model, mol_specs,
                                  reactions, spec_dir / "eval_holdout", mol_data)
                write_stamp(spec_dir, summary)
                processed.append(idx)
                comb = summary.get("combined")
                if comb and len(comb) >= 2:
                    print(f"[reeval]   spec {idx} done: combined NN MAE="
                          f"{comb[0]:.2f}  PBE MAE={comb[1]:.2f} kcal/mol  "
                          f"({summary.get('n_reactions')} rxn)", flush=True)
            except Exception as exc:  # noqa: BLE001 - report, continue
                failed.append(idx)
                print(f"[reeval]   spec {idx} FAILED: "
                      f"{type(exc).__name__}: {exc}", flush=True)

    print(f"[reeval] done: {len(processed)} processed, {len(skipped)} skipped, "
          f"{len(failed)} failed in {_fmt_eta(clock() - t0)}.", flush=True)
    if failed:
        print(f"[reeval] FAILED specs: {failed}", flush=True)
    return {"processed": processed, "skipped": skipped, "failed": failed}


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", default=None,
                   help="pulled run dir (default: latest ablation_notransform)")
    p.add_argument("--specs", default=None,
                   help="comma-separated spec indices to restrict to")
    p.add_argument("--force", action="store_true",
                   help="re-process even specs already stamped with the "
                        "current fix version")
    args = p.parse_args(argv)

    run_dir = resolve_run_dir(args.run_dir)
    only = None
    if args.specs:
        only = [int(t) for t in args.specs.split(",") if t.strip()]
    result = run(run_dir, force=args.force, only_specs=only)
    return 1 if result["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
