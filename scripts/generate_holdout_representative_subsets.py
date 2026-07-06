#!/usr/bin/env python
"""Generate descriptor-representative subsets of the full BH76+W4-11 reaction set.

The ENTIRE BH76+W4-11 set (~216 reactions) is the candidate pool. For each
subset size r, exhaustively (parallel) select the r reactions whose pooled GGA
descriptor distribution minimizes the Jensen-Shannon divergence to the FULL-set
distribution -- i.e. the most representative size-r subset. Held-out = the
complement (the reactions NOT chosen). GGA XCDiff descriptors only
(rho^{1/3}, s, alpha); per-species histograms are precomputed once and cached;
the selector keeps only the running argmin (nothing per-combo to disk).

Outputs, per alpha mode (alpha_on = equal weights; alpha_off = drop the
meta-GGA alpha coordinate the GGA can't see):
  notebooks/analysis/bh76w411_representative_subsets/<mode>/subset_index_log.json
    { "jsd/<r>": {chosen_indices, metric_value, point_names, point_kinds,
                  held_out_names} }
  + summary.md

Usage:
  python scripts/generate_holdout_representative_subsets.py --sizes 1 2 3 4 5 6 7
  python scripts/generate_holdout_representative_subsets.py --sizes 1 2 3 --modes alpha_off
"""
import argparse
import json
import math
import time
from pathlib import Path

from xcquinox.alec.full_benchmark_pools import load_full_held_out_pools
from xcquinox.alec import subset_selection as ss
from xcquinox.alec.subset_selection_parallel import select_subset_parallel


BASIS = "def2-svp"
GRID_LEVEL = 1
OUT_ROOT = (Path(__file__).resolve().parents[1]
            / "notebooks" / "analysis" / "bh76w411_representative_subsets")
DESC_CACHE = OUT_ROOT / "_descriptor_cache"
# alpha_on: equal per-descriptor weights; alpha_off: zero the meta-GGA alpha
# coordinate (a GGA functional is structurally blind to tau).
_MODE_WEIGHTS = {"alpha_on": None, "alpha_off": {"alpha": 0.0}}


def _select(pool, edges, h_ref, r, weights, args):
    """Dispatch to the GPU selector (preferred) or the CPU parallel selector.

    ``--selector auto`` (default) tries the GPU JAX kernel and falls back to the
    CPU multiprocessing selector if no JAX device initializes; ``gpu``/``cpu``
    force one. Both return ``(chosen_indices, jsd_value)``."""
    if args.selector in ("auto", "gpu"):
        try:
            from xcquinox.alec.subset_selection_gpu import select_subset_gpu
            return select_subset_gpu(pool, edges, h_ref, r=r, metric="jsd",
                                     batch=args.gpu_batch,
                                     descriptor_weights=weights)
        except Exception as exc:  # noqa: BLE001
            if args.selector == "gpu":
                raise
            print(f"  [GPU selector unavailable ({type(exc).__name__}: {exc}); "
                  f"falling back to CPU]", flush=True)
    return select_subset_parallel(pool, edges, h_ref, r=r, metric="jsd",
                                  n_jobs=args.n_jobs, descriptor_weights=weights)


def _reaction_name(rxn, i):
    return rxn.get("name") or f"{rxn.get('source_pool', 'rxn')}_{i}"


def _write_summary(out_dir, ledger, n_rxn):
    lines = [
        "# BH76+W4-11 representative subsets (JSD, GGA descriptors)",
        "",
        f"Pool = {n_rxn} reactions. Held-out = the complement of each subset.",
        "",
        "| r | JSD | n_held_out | chosen (name:kind) |",
        "|---|-----|-----------|--------------------|",
    ]
    for r in sorted(int(k.split("/")[1]) for k in ledger):
        e = ledger[f"jsd/{r}"]
        names = ", ".join(f"{n}:{k}" for n, k in
                          zip(e["point_names"], e["point_kinds"]))
        lines.append(f"| {r} | {e['metric_value']:.4e} | "
                     f"{len(e['held_out_names'])} | {names} |")
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sizes", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6, 7],
                    help="subset sizes r to enumerate (default 1..7)")
    ap.add_argument("--modes", nargs="+", default=["alpha_on", "alpha_off"],
                    choices=sorted(_MODE_WEIGHTS), help="alpha weighting modes")
    ap.add_argument("--n-jobs", type=int, default=None,
                    help="CPU worker processes (default: auto-detect usable CPUs)")
    ap.add_argument("--selector", choices=["auto", "gpu", "cpu"], default="auto",
                    help="auto: GPU (JAX) then CPU fallback; gpu/cpu force one")
    ap.add_argument("--gpu-batch", type=int, default=None,
                    help="GPU selector: ranks per batch (default 2^20)")
    args = ap.parse_args(argv)

    full_specs, full_rxns = load_full_held_out_pools(
        basis=BASIS, grid_level=GRID_LEVEL)
    n_rxn = len(full_rxns)
    rxn_names = [_reaction_name(r, i) for i, r in enumerate(full_rxns)]
    rxn_kinds = [r.get("source_pool", "unknown") for r in full_rxns]
    n_bh76 = sum(k == "bh76" for k in rxn_kinds)
    n_w411 = sum(k == "w411" for k in rxn_kinds)
    print(f"Pool: {n_rxn} reactions ({n_bh76} bh76 + {n_w411} w411); "
          f"{len(full_specs)} unique species", flush=True)

    from xcquinox.alec.parallel import detect_available_cpus
    xjobs = int(args.n_jobs) if args.n_jobs else detect_available_cpus()
    DESC_CACHE.mkdir(parents=True, exist_ok=True)
    print(f"Extracting per-species GGA descriptors "
          f"(PBE/{BASIS}/grid_level={GRID_LEVEL}; {xjobs} workers; "
          f"cache {DESC_CACHE}) ...", flush=True)
    t0 = time.time()
    species_desc = ss.extract_descriptors_for_molspecs(
        full_specs.values(), basis=BASIS, grid_level=GRID_LEVEL,
        cache_dir=DESC_CACHE, n_jobs=xjobs)
    print(f"  {len(species_desc)} species in {time.time() - t0:.1f}s", flush=True)

    reaction_desc = ss.concatenate_reaction_descriptors(
        full_rxns, species_desc, full_specs)
    h_ref, edges = ss.build_reference_histograms(reaction_desc)
    print("Built full-pool reference histograms "
          f"(rho_third/s/alpha, {ss.NBINS} bins).", flush=True)

    for mode in args.modes:
        weights = _MODE_WEIGHTS[mode]
        out_dir = OUT_ROOT / mode
        out_dir.mkdir(parents=True, exist_ok=True)
        ledger_path = out_dir / "subset_index_log.json"
        ledger = {}
        if ledger_path.exists():
            ledger = json.loads(ledger_path.read_text())
            print(f"[{mode}] resumed ledger: {len(ledger)} entries", flush=True)

        for r in args.sizes:
            key = f"jsd/{r}"
            if key in ledger:
                print(f"[{mode}] {key} cached; skip.", flush=True)
                continue
            ncombo = math.comb(n_rxn, r)
            print(f"[{mode}] {key}: enumerating C({n_rxn},{r})={ncombo:,} ...",
                  flush=True)
            t1 = time.time()
            chosen, val = _select(reaction_desc, edges, h_ref, r, weights, args)
            chosen = list(chosen)
            chosen_set = set(chosen)
            held = [rxn_names[i] for i in range(n_rxn) if i not in chosen_set]
            ledger[key] = {
                "chosen_indices": [int(i) for i in chosen],
                "metric_value": float(val),
                "point_names": [rxn_names[i] for i in chosen],
                "point_kinds": [rxn_kinds[i] for i in chosen],
                "held_out_names": held,
            }
            ledger_path.write_text(json.dumps(ledger, indent=2))
            _write_summary(out_dir, ledger, n_rxn)
            print(f"[{mode}]   chosen={chosen} jsd={val:.6e} "
                  f"held_out={len(held)} dt={time.time() - t1:.1f}s", flush=True)

        print(f"[{mode}] DONE -> {ledger_path}", flush=True)


if __name__ == "__main__":
    main()
