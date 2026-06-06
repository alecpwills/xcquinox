#!/usr/bin/env python
"""Generate the step-7 histogram-matched subset ledgers for the cluster.

Standalone (no Jupyter) runner for the subset-SELECTION pre-process that the
``gga_subset_generation.ipynb`` notebook performs, sharing the same library
functions (`xcquinox.alec.subset_selection` + `training_points`). Produces, per
C4-03 alpha mode, the artifacts the SLURM harness consumes read-only:

    notebooks/checkpoints_step7/<alpha_on|alpha_off>/subset_index_log.json
    notebooks/checkpoints_step7/<alpha_on|alpha_off>/<metric>/bin<NN>/
        <ARCH>/<LOSS>/<solver>/subset.traj

Descriptors (ρ^{1/3}, s, α via one PBE SCF per unique species) and the
full-pool reference histogram are MODE-INDEPENDENT — the alpha mode only sets
``descriptor_weights`` inside ``select_subset`` — so both are computed ONCE
(shared cache) and the selection sweep is run for every requested mode.

Binning is LINEAR (2026-05-24 revision; see subset_selection.py). CPU-only
(pyscf SCF + numpy combinatorics) — does not use the GPU.

Usage:
    python scripts/generate_step7_subsets.py                 # both modes
    python scripts/generate_step7_subsets.py --modes alpha_off
    STEP7_SUBSET_BATCH=65536 python scripts/generate_step7_subsets.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from ase.io import write as ase_write

# Grid constants are the source of truth in the step-7 notebook builder; import
# them so this script can never drift from the notebook's grid definition.
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "notebooks"))
from _build_step7_notebook import (  # noqa: E402
    SUBSET_SIZES, METRICS, SOLVERS, ARCH_NAME, LOSS_NAME,
)

from xcquinox.alec import subset_selection as ss  # noqa: E402
from xcquinox.alec.training_points import (  # noqa: E402
    build_dfs_pool_points, species_union_from_points,
)

BASIS = "def2-svp"
GRID_LEVEL = 1
CHECKPOINTS = REPO / "notebooks" / "checkpoints_step7"
# Mode-independent shared caches (descriptors + reference are the same for every
# alpha mode; only select_subset's descriptor_weights differs).
SHARED_DESCRIPTOR_CACHE = CHECKPOINTS / "subset_descriptors"

_MODE_WEIGHTS = {
    "alpha_on": None,                 # default: alpha weighted equally
    "alpha_off": {"alpha": 0.0},      # C4-03: drop the meta-GGA alpha descriptor
}


def _resolve_sizes(base_sizes, n_pool: int, include_full: bool) -> list:
    """The subset sizes to generate. ``--include-full`` appends the full-pool
    size (r == n_pool, the complete training set; C(n_pool,n_pool)=1) when it is
    not already present. Order preserved; deduplicated."""
    sizes = list(base_sizes)
    if include_full and n_pool not in sizes:
        sizes.append(n_pool)
    return sizes


def _build_shared(points):
    """Extract per-species descriptors (cached) and the full-pool reference
    histogram + edges — all mode-independent."""
    species = species_union_from_points(points)
    print(f"Extracting descriptors for {len(species)} unique species "
          f"(PBE/{BASIS}/grid_level={GRID_LEVEL}; cached at "
          f"{SHARED_DESCRIPTOR_CACHE}) ...", flush=True)
    SHARED_DESCRIPTOR_CACHE.mkdir(parents=True, exist_ok=True)
    species_descriptors = ss.extract_descriptors_for_species(
        species, basis=BASIS, grid_level=GRID_LEVEL,
        cache_dir=SHARED_DESCRIPTOR_CACHE,
    )
    point_descriptors = ss.concatenate_point_descriptors(points, species_descriptors)
    h_ref, edges = ss.build_reference_histograms(point_descriptors)
    return point_descriptors, h_ref, edges


def _run_mode(mode, points, point_descriptors, h_ref, edges, sizes=SUBSET_SIZES):
    """Run the (metric, r) selection sweep for one alpha mode and write the
    ledger + subset.traj files under that mode's STEP7_ROOT. ``sizes`` defaults
    to the canonical ``SUBSET_SIZES`` (``--include-full`` extends it)."""
    weights = _MODE_WEIGHTS[mode]
    root = CHECKPOINTS / mode
    root.mkdir(parents=True, exist_ok=True)
    ref_cache = root / "dfs_pool_full_hist"
    ref_cache.mkdir(parents=True, exist_ok=True)
    np.savez(
        ref_cache / "reference.npz",
        h_ref_rho=h_ref["rho_third"], e_rho=edges["rho_third"],
        h_ref_s=h_ref["s"], e_s=edges["s"],
        h_ref_alpha=h_ref["alpha"], e_alpha=edges["alpha"],
    )
    ledger_path = root / "subset_index_log.json"

    # Resume: load any existing ledger entries.
    subset_index_log: dict = {}
    if ledger_path.exists():
        for slashkey, val in json.loads(ledger_path.read_text()).items():
            m, r = slashkey.split("/")
            subset_index_log[(m, int(r))] = val
        print(f"[{mode}] resumed ledger: {len(subset_index_log)} entries", flush=True)

    def _write_ledger():
        ledger_path.write_text(json.dumps(
            {f"{k[0]}/{k[1]}": v for k, v in subset_index_log.items()}, indent=2))

    n_pool = len(points)
    n_pairs = len(METRICS) * len(sizes)
    print(f"[{mode}] descriptor_weights={weights}; {n_pairs} (metric, r) pairs "
          f"over {n_pool} candidates -> {root}", flush=True)
    pair_idx = 0
    t0_all = time.time()
    for metric in METRICS:
        for r in sizes:
            pair_idx += 1
            all_present = (
                (metric, r) in subset_index_log
                and all(
                    (root / metric / f"bin{r:02d}" / ARCH_NAME / LOSS_NAME /
                     solver / "subset.traj").exists()
                    for solver in SOLVERS
                )
            )
            if all_present:
                print(f"[{mode}] [{pair_idx:>2d}/{n_pairs}] {metric} r={r:>2d} "
                      f"CACHED; skip.", flush=True)
                continue
            ncombo = math.comb(n_pool, r)
            t0 = time.time()
            print(f"[{mode}] [{pair_idx:>2d}/{n_pairs}] {metric} r={r:>2d} "
                  f"enumerating C({n_pool},{r})={ncombo:,} ...", flush=True)
            chosen, val = ss.select_subset(
                point_descriptors, edges, h_ref, r=r, metric=metric,
                descriptor_weights=weights, progress=False,
            )
            chosen_points = [points[i] for i in chosen]
            traj_atoms = species_union_from_points(chosen_points)
            tag = f"bin{r:02d}"
            for solver in SOLVERS:
                spec_dir = root / metric / tag / ARCH_NAME / LOSS_NAME / solver
                spec_dir.mkdir(parents=True, exist_ok=True)
                ase_write(str(spec_dir / "subset.traj"), traj_atoms)
            subset_index_log[(metric, r)] = {
                "chosen_indices": list(int(i) for i in chosen),
                "metric_value": float(val),
                "point_kinds": [tp.kind for tp in chosen_points],
                "point_names": [tp.name for tp in chosen_points],
                "tag": tag,
            }
            _write_ledger()
            dt = time.time() - t0
            print(f"[{mode}]     -> indices={list(chosen)} value={val:.6e} "
                  f"dt={dt:.1f}s", flush=True)
    _write_ledger()
    n_specs = len(subset_index_log) * len(SOLVERS)
    expected = len(sizes) * len(METRICS) * len(SOLVERS)
    print(f"[{mode}] wrote {len(subset_index_log)} (metric, r) entries; "
          f"{n_specs} subset.traj files; total {time.time()-t0_all:.1f}s", flush=True)
    assert n_specs == expected, f"[{mode}] expected {expected} specs, got {n_specs}"
    return ledger_path


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--modes", nargs="+", default=["alpha_on", "alpha_off"],
        choices=sorted(_MODE_WEIGHTS), help="alpha modes to generate")
    ap.add_argument(
        "--include-full", action="store_true",
        help="also generate the full-pool subset (r = pool size, the complete "
             "DFS training set) and add it to the ledger")
    args = ap.parse_args(argv)

    points = build_dfs_pool_points()
    by_kind = {k: sum(1 for p in points if p.kind == k) for k in ("ae", "bh76", "ip13")}
    print(f"Mixed pool: {len(points)} points ({by_kind['ae']} AE + "
          f"{by_kind['bh76']} BH76 + {by_kind['ip13']} IP13)", flush=True)
    sizes = _resolve_sizes(SUBSET_SIZES, len(points), args.include_full)
    if args.include_full:
        print(f"--include-full: sizes -> {sizes} (added r={len(points)} full pool)",
              flush=True)

    point_descriptors, h_ref, edges = _build_shared(points)

    written = []
    for mode in args.modes:
        written.append(_run_mode(mode, points, point_descriptors, h_ref, edges, sizes))

    print("\nDONE. Ledgers (stage these to the cluster's inputs.subset_ledger_path):")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
