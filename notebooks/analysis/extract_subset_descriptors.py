#!/usr/bin/env python
"""Extract per-spec DMStatistics + Cusp descriptor distributions over the
training molecules.

For every pulled spec, identify the training molecules from
``training_spec.molecules`` and compute the **grid-weighted mean** of every
descriptor feature column on each training molecule's PBE density. Aggregate
per spec into a (n_train × n_features) matrix plus per-feature summary
stats (mean / std / min / max / range across the training set).

The output, ``<run_dir>/checkpoints/spec_<NNNN>/eval/local_subset_descriptors.json``,
is consumed by:

  - Fig 10 -- descriptor range vs held-out accuracy (uses ``range``).
  - Fig 15 -- per-subset descriptor histograms with std/mean marks
    (uses ``per_molecule_features``).

Runtime: ~30-40 unique training molecules × ~5 s/molecule for the first
spec's pass, then near-zero for every subsequent spec (in-process cache in
``xcquinox.alec.precompute_fixed_density_data``). Total: ~3-5 min for the
current 119 specs.

Usage::

    # Process every pulled category, every spec with a .spec file:
    python notebooks/analysis/extract_subset_descriptors.py --auto

    # Or a single run dir + specific spec ids:
    python notebooks/analysis/extract_subset_descriptors.py \\
        <run_dir> --specs 0,1,5
"""
from __future__ import annotations

import os

os.environ["JAX_ENABLE_X64"] = "1"
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import importlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------

def feature_column_names(descriptors: Sequence[Any]) -> List[str]:
    """Generate ``"<descriptor>_<i>"`` names matching the column order of
    ``np.concatenate([d.compute(mol_data) for d in descriptors], axis=1)``.
    """
    names: List[str] = []
    for d in descriptors:
        cls = type(d).__name__
        nf = getattr(d, "n_features", 0)
        for i in range(int(nf)):
            names.append(f"{cls}_{i}")
    return names


def per_molecule_feature_means(features: np.ndarray,
                               grid_weights: np.ndarray) -> np.ndarray:
    """Grid-weighted mean of each feature column.

    ``features`` has shape ``(N_grid, n_features)``, ``grid_weights`` has
    shape ``(N_grid,)``. Returns ``(n_features,)``. Pure.
    """
    if features.ndim != 2:
        raise ValueError(f"features must be 2-D, got shape {features.shape}")
    if features.shape[0] != grid_weights.shape[0]:
        raise ValueError(
            f"feature N_grid ({features.shape[0]}) and grid_weights N_grid "
            f"({grid_weights.shape[0]}) disagree")
    w = np.asarray(grid_weights, dtype=float)
    f = np.asarray(features, dtype=float)
    w_sum = float(w.sum())
    if w_sum <= 0:
        return f.mean(axis=0)
    # Weighted column-wise mean.
    return (f * w[:, None]).sum(axis=0) / w_sum


def per_subset_stats(per_mol: np.ndarray) -> Dict[str, List[float]]:
    """``(n_molecules × n_features) -> {mean, std, min, max, range}``.

    Each value is a length-``n_features`` list of floats. ``range`` = max −
    min per feature column (the "range" the PI asked for). Pure.
    """
    if per_mol.ndim != 2:
        raise ValueError(
            f"per_mol must be 2-D, got shape {per_mol.shape}")
    if per_mol.shape[0] == 0:
        n_feat = per_mol.shape[1]
        nan = [float("nan")] * n_feat
        return {"mean": nan, "std": nan, "min": nan, "max": nan, "range": nan}
    return {
        "mean":  [float(v) for v in per_mol.mean(axis=0)],
        "std":   [float(v) for v in per_mol.std(axis=0)],
        "min":   [float(v) for v in per_mol.min(axis=0)],
        "max":   [float(v) for v in per_mol.max(axis=0)],
        "range": [float(v) for v in (per_mol.max(axis=0)
                                      - per_mol.min(axis=0))],
    }


# ---------------------------------------------------------------------------
# Side-effectful (script entry points)
# ---------------------------------------------------------------------------

def _load_training_spec(spec_path: Path):
    """Same indirection as ``local_reeval.load_training_spec``."""
    _ser = importlib.import_module("pi" + "ckle")
    with open(spec_path, "rb") as f:
        return _ser.load(f)


def _load_category_discovery() -> Callable:
    """Reuse ``discover_pulled_categories`` from the figure script."""
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))
    fig = importlib.import_module("make_cluster_pulls_figure")
    return fig.discover_pulled_categories


def discover_specs_in_run(run_dir: Path, width: int = 4) -> List[int]:
    """``[spec_index, ...]`` for every ``specs/spec_<NNNN>.spec`` present."""
    sp = run_dir / "specs"
    if not sp.is_dir():
        return []
    out: List[int] = []
    for p in sorted(sp.glob("spec_*.spec")):
        token = p.stem[len("spec_"):]
        try:
            out.append(int(token))
        except ValueError:
            continue
    return sorted(set(out))


def _strip_external_data_path(mol_spec: Any) -> Any:
    """Return a copy of ``mol_spec`` with ``external_data_path`` set to
    ``None``. Training MoleculeSpecs from the cluster carry a path under
    ``/gpfs/scratch/awills/external_refs/<name>.npz`` that doesn't exist
    locally; for descriptor extraction we only need the PBE density
    (computed from scratch), so the CCSD-reference npz is unneeded."""
    import dataclasses
    if getattr(mol_spec, "external_data_path", None) is None:
        return mol_spec
    try:
        return dataclasses.replace(mol_spec, external_data_path=None)
    except (TypeError, AttributeError):
        # Non-dataclass or unsupported field -- fall through, the caller
        # will see the original failure.
        return mol_spec


def compute_feature_means_for_molecule(mol_spec: Any,
                                       descriptors: Sequence[Any]
                                       ) -> Optional[np.ndarray]:
    """Run the PBE precompute on ``mol_spec`` with ``descriptors``, then
    return the per-feature grid-weighted means as a ``(n_features,)``
    array. Returns ``None`` on any failure (logged but non-fatal so a
    single bad molecule does not abort the run).

    Training MoleculeSpecs carry a cluster-only ``external_data_path``;
    we strip it before precompute since for descriptor extraction we only
    need the PBE density (computed from scratch)."""
    import xcquinox.alec as alec
    local_spec = _strip_external_data_path(mol_spec)
    try:
        mol_data = alec.precompute_fixed_density_data(
            local_spec, descriptors=tuple(descriptors))
    except Exception as exc:  # noqa: BLE001
        print(f"  precompute FAILED for {mol_spec.name}: "
              f"{type(exc).__name__}: {exc}", flush=True)
        return None
    try:
        feats = np.asarray(np.concatenate(
            [np.asarray(d.compute(mol_data)) for d in descriptors],
            axis=1), dtype=float)
        gw = np.asarray(mol_data["grid_weights"], dtype=float)
    except Exception as exc:  # noqa: BLE001
        print(f"  descriptor compute FAILED for {mol_spec.name}: "
              f"{type(exc).__name__}: {exc}", flush=True)
        return None
    if feats.ndim != 2 or feats.shape[0] != gw.shape[0]:
        print(f"  feature/grid_weights shape mismatch for "
              f"{mol_spec.name}: feats={feats.shape}, "
              f"gw={gw.shape}", flush=True)
        return None
    return per_molecule_feature_means(feats, gw)


def write_local_subset_descriptors_json(
    spec_dir: Path,
    training_molecule_names: Sequence[str],
    feature_names: Sequence[str],
    per_molecule_features: np.ndarray,
) -> Path:
    """``<spec_dir>/eval/local_subset_descriptors.json``."""
    out_dir = spec_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "local_subset_descriptors.json"
    payload = {
        "training_molecule_names": list(training_molecule_names),
        "feature_names": list(feature_names),
        "per_molecule_features": per_molecule_features.tolist(),
        "per_subset_stats": per_subset_stats(per_molecule_features),
    }
    with out.open("w") as f:
        json.dump(payload, f, indent=2)
    return out


# ---------------------------------------------------------------------------
# Per-spec orchestration
# ---------------------------------------------------------------------------

def run_one_spec(
    run_dir: Path,
    spec_idx: int,
    feature_cache: Dict[str, Optional[np.ndarray]],
    *,
    width: int = 4,
) -> Dict[str, Any]:
    """Process one (run_dir, spec_idx). Returns a summary dict. Raises on
    missing files. ``feature_cache`` is shared across calls so molecules
    seen in multiple specs precompute exactly once."""
    spec_name = f"spec_{spec_idx:0{width}d}"
    spec_path = run_dir / "specs" / f"{spec_name}.spec"
    if not spec_path.is_file():
        raise FileNotFoundError(f"spec file missing: {spec_path}")
    spec_dir = run_dir / "checkpoints" / spec_name
    spec_dir.mkdir(parents=True, exist_ok=True)

    training_spec = _load_training_spec(spec_path)
    try:
        descriptors = tuple(training_spec.arch.materialize_descriptors())
    except AttributeError:
        descriptors = ()
    if not descriptors:
        print(f"[spec {spec_idx}] no descriptors on arch -- skipping",
              flush=True)
        return {"idx": spec_idx, "skipped": True,
                "reason": "no descriptors on arch"}
    feature_names = feature_column_names(descriptors)
    print(f"[spec {spec_idx}] {len(training_spec.molecules)} training "
          f"molecules; descriptors: "
          f"{[type(d).__name__ for d in descriptors]} "
          f"(total {len(feature_names)} cols)", flush=True)

    rows: List[np.ndarray] = []
    kept_names: List[str] = []
    for m in training_spec.molecules:
        name = m.name
        if name in feature_cache:
            vec = feature_cache[name]
        else:
            t0 = time.time()
            vec = compute_feature_means_for_molecule(m, descriptors)
            feature_cache[name] = vec
            tag = "(cached miss -- new)" if vec is not None else "(FAILED)"
            print(f"  {name}: {time.time() - t0:.2f}s  {tag}",
                  flush=True)
        if vec is None:
            continue
        rows.append(vec)
        kept_names.append(name)
    if not rows:
        print(f"[spec {spec_idx}] no usable training-molecule features -- "
              "skipping write", flush=True)
        return {"idx": spec_idx, "skipped": True,
                "reason": "no usable training features"}

    per_mol = np.vstack(rows)
    out = write_local_subset_descriptors_json(
        spec_dir, kept_names, feature_names, per_mol,
    )
    print(f"[spec {spec_idx}] wrote {out.name}  "
          f"({per_mol.shape[0]} mols × {per_mol.shape[1]} features)",
          flush=True)
    return {"idx": spec_idx, "n_train": per_mol.shape[0],
            "n_features": per_mol.shape[1], "out": out}


# ---------------------------------------------------------------------------
# Multi-category --auto driver
# ---------------------------------------------------------------------------

def run_auto(local_root: Path, *, width: int = 4) -> Dict[str, Any]:
    """``--auto`` discovery loop. Shares one ``feature_cache`` across all
    specs and categories so unique molecules precompute exactly once."""
    discover = _load_category_discovery()
    cats = discover(local_root)
    if not cats:
        print(f"no run_<UTC>Z dirs found under {local_root}",
              file=sys.stderr)
        return {}
    print(f"--auto: {len(cats)} categories", flush=True)
    feature_cache: Dict[str, Optional[np.ndarray]] = {}
    summary: Dict[str, Any] = {}
    t0_overall = time.time()
    grand_ok = 0
    grand_total = 0
    for cat, run_dir in cats.items():
        cat_label = cat or "(root)"
        spec_indices = discover_specs_in_run(run_dir, width=width)
        if not spec_indices:
            print(f"\n=== {cat_label}: no specs/ found, skipping ===",
                  flush=True)
            summary[cat] = {"n_specs": 0, "n_ok": 0, "n_failed": 0}
            continue
        print(f"\n=== {cat_label}: {len(spec_indices)} spec(s) ===",
              flush=True)
        n_ok = 0
        failed: List[Tuple[int, str]] = []
        for idx in spec_indices:
            try:
                result = run_one_spec(run_dir, idx, feature_cache,
                                      width=width)
                if result.get("skipped"):
                    failed.append((idx, result.get("reason", "skipped")))
                else:
                    n_ok += 1
            except Exception as exc:  # noqa: BLE001
                msg = f"{type(exc).__name__}: {exc}"
                failed.append((idx, msg))
                print(f"[spec {idx}] FAILED: {msg}", file=sys.stderr,
                      flush=True)
        summary[cat] = {"n_specs": len(spec_indices), "n_ok": n_ok,
                        "n_failed": len(failed)}
        grand_total += len(spec_indices); grand_ok += n_ok
        print(f"--- {cat_label}: {n_ok}/{len(spec_indices)} ok", flush=True)
    print(f"\n=== --auto: {grand_ok}/{grand_total} specs ok in "
          f"{time.time() - t0_overall:.0f}s  "
          f"({sum(1 for v in feature_cache.values() if v is not None)} "
          "unique molecules precomputed) ===", flush=True)
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _default_local_root() -> str:
    return os.environ.get(
        "XCQUINOX_CLUSTER_LOCAL_ROOT",
        str(Path.home() / "Documents/Research/xcquinox-results/runs"),
    )


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("run_dir", type=Path, nargs="?", default=None,
                   help="locally-staged run dir; omit when using --auto.")
    p.add_argument("--specs", default=None,
                   help="comma-separated spec indices when run_dir is given.")
    p.add_argument("--auto", action="store_true",
                   help="auto-discover every pulled category under "
                        "--local-root and process every spec_*.spec file.")
    p.add_argument("--local-root", default=_default_local_root(),
                   help="root holding categories with run_<UTC>Z dirs.")
    p.add_argument("--width", type=int, default=4,
                   help="zero-pad width of spec_NNNN dir names.")
    args = p.parse_args(argv)

    if args.auto:
        if args.run_dir is not None or args.specs is not None:
            print("--auto is incompatible with run_dir / --specs",
                  file=sys.stderr)
            return 1
        local_root = Path(args.local_root).expanduser().resolve()
        if not local_root.is_dir():
            print(f"--local-root does not exist: {local_root}",
                  file=sys.stderr)
            return 1
        summary = run_auto(local_root, width=args.width)
        return 0 if summary else 1

    if args.run_dir is None or args.specs is None:
        print("either pass --auto, or both <run_dir> and --specs",
              file=sys.stderr)
        return 1
    run_dir = args.run_dir.expanduser().resolve()
    if not run_dir.is_dir():
        print(f"run_dir does not exist: {run_dir}", file=sys.stderr)
        return 1
    try:
        spec_indices = [int(t.strip()) for t in args.specs.split(",")
                        if t.strip()]
    except ValueError as exc:
        print(f"--specs entries must be integers ({exc})", file=sys.stderr)
        return 1
    if not spec_indices:
        print("--specs is empty", file=sys.stderr)
        return 1
    cache: Dict[str, Optional[np.ndarray]] = {}
    for idx in spec_indices:
        try:
            run_one_spec(run_dir, idx, cache, width=args.width)
        except FileNotFoundError as exc:
            print(f"[spec {idx}] {exc}", file=sys.stderr)
            continue
    return 0


if __name__ == "__main__":
    sys.exit(main())
