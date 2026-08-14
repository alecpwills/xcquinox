"""Generate the SCAN seed cache for a run's TRAINING-side species.

Enumerates the species PROGRAMMATICALLY as the union over the run's FULL
training-point pool (``domain.pool_builder(cfg)`` -- every cell trains a
subset of these points, so the union covers every cell's molecule list)
and converges one SCAN KS-SCF per species into
``inputs.seed_cache_dir/_intermediates/`` via
``external_refs.run_scf_with_cache`` at the run identity: basis, grid
level, density fitting, and -- critically -- the orientation lock, which
is part of the cache filename identity (an unlocked SCAN seed would
re-enter the degenerate-component artifact through D0 for the 2-Pi
radicals). Cached species are skipped by the cache layer itself, so the
job is resumable; non-converged SCFs are escalated and never cached.

Eval/val (pool) species: pass ``--link-pool <scan_pool_dir>`` and the
existing scan-pool intermediates (all 214 at the production identity) are
re-keyed into the seed cache under geometry-qualified names computed from
the POOL geometries -- a bare species name cannot identify a seed, since
the training set and the pool both contain e.g. an H2O at different
geometries (the collision measured 2026-08-14: H2/H2O/CH4 pool files
answered training lookups; only the atom O was genuinely shared).

Usage::

    python -m xcquinox.alec.cluster.seed_cache <grid_config.yaml> \
        [--link-pool <scan_pool_dir>] [--dry-run]
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import List, Optional, Sequence


def _load_cfg(path: str):
    from xcquinox.alec.cluster.grid_config import load_grid_config
    return load_grid_config(path)


def _pool_species(cfg) -> List:
    """Deduplicated species union over the run's FULL training-point pool."""
    from xcquinox.alec.cluster.domain import get_domain_profile
    from xcquinox.alec.training_points import species_union_from_points
    domain = get_domain_profile(cfg.domain_profile)
    points = domain.pool_builder(cfg)
    return species_union_from_points(points)


def _run_scf_with_cache(entry, atoms, **kw):
    from xcquinox.alec.external_refs import run_scf_with_cache
    return run_scf_with_cache(entry, atoms, **kw)


def _held_out_pool_specs(cfg):
    from xcquinox.alec.full_benchmark_pools import load_full_held_out_pools
    pool_specs, _ = load_full_held_out_pools(
        basis=cfg.inputs.basis, grid_level=int(cfg.inputs.grid_level))
    return pool_specs


def _link_pool_intermediates(cfg, scan_pool_dir: str, cache_dir: str) -> int:
    """Re-key the scan-pool intermediates under geometry-qualified names.

    The pool cache was written with UNQUALIFIED species names; the seed
    loader qualifies every lookup with the geometry tag, so the link phase
    computes each pool spec's qualified target from the POOL geometry and
    symlinks the existing npz there. Idempotent; missing sources are
    counted and reported (a missing pool seed later fails loud at the
    val-coverage gate / loader, never silently).
    """
    import os

    from xcquinox.alec.data import seed_cache_file
    from xcquinox.alec.external_refs import _intermediate_cache_name
    pool_specs = _held_out_pool_specs(cfg)
    df = bool(cfg.inputs.density_fit)
    lock = float(cfg.inputs.orientation_lock_strength)
    n_linked = n_present = n_missing = 0
    for name, ps in sorted(pool_specs.items()):
        gl = ps.grid_level if ps.grid_level is not None else 3
        src = os.path.join(
            scan_pool_dir, "_intermediates", _intermediate_cache_name(
                name, grid_level=gl, basis=ps.basis, density_fit=df,
                kind="scf", orientation_lock_strength=lock, xc="scan"))
        dst = seed_cache_file(ps, seed_cache_dir=cache_dir, density_fit=df,
                              orientation_lock_strength=lock)
        if not os.path.isfile(src):
            n_missing += 1
            continue
        if os.path.exists(dst) or os.path.islink(dst):
            n_present += 1
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        os.symlink(src, dst)
        n_linked += 1
    print(f"[seed-cache] pool link phase: {n_linked} linked, "
          f"{n_present} already present, {n_missing} missing sources "
          f"(from {scan_pool_dir})", flush=True)
    return n_missing


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("config", help="grid config / resolved_config YAML")
    p.add_argument("--dry-run", action="store_true",
                   help="list the enumerated species and identity; run "
                        "nothing")
    p.add_argument("--link-pool", default=None, metavar="SCAN_POOL_DIR",
                   help="re-key this scan-pool cache's intermediates into "
                        "the seed cache under geometry-qualified names "
                        "(the val/eval species)")
    args = p.parse_args(argv)
    from xcquinox.alec.external_refs import SpeciesEntry

    cfg = _load_cfg(args.config)
    cache_dir = getattr(cfg.inputs, "seed_cache_dir", None)
    if not cache_dir:
        print("[seed-cache] FATAL: inputs.seed_cache_dir is unset in "
              f"{args.config} -- nowhere to write the cache", flush=True)
        return 1
    species = _pool_species(cfg)
    ident = (f"basis={cfg.inputs.basis} grid={cfg.inputs.grid_level} "
             f"df={cfg.inputs.density_fit} "
             f"lock={cfg.inputs.orientation_lock_strength:g} xc=scan")
    print(f"[seed-cache] {len(species)} training-side species -> "
          f"{cache_dir} ({ident})", flush=True)
    for at in species:
        print(f"[seed-cache]   {at.info['name']} "
              f"(charge={at.info.get('charge', 0)}, "
              f"spin={at.info.get('spin', 0)})", flush=True)
    if args.dry_run:
        return 0
    if args.link_pool:
        _link_pool_intermediates(cfg, args.link_pool, cache_dir)
    # geometry-qualified cache identity: shared canonicalization with the
    # loader (atoms -> pyscf string -> tag), so training and pool same-name
    # twins at different geometries occupy distinct files while identical
    # geometries deduplicate.
    from xcquinox.alec.cluster.spec_builder import atoms_to_pyscf_str
    from xcquinox.alec.data import seed_geometry_tag
    n_ok, failed = 0, []
    t0 = time.time()
    for i, at in enumerate(species, start=1):
        name = at.info["name"]
        charge = int(at.info.get("charge", 0))
        spin = int(at.info.get("spin", 0))
        tag = seed_geometry_tag(atoms_to_pyscf_str(at), charge, spin)
        entry = SpeciesEntry(name=f"{name}_gh{tag}", charge=charge,
                             spin=spin, source="seed")
        t1 = time.time()
        try:
            _run_scf_with_cache(
                entry, at, cache_dir=cache_dir, basis=cfg.inputs.basis,
                grid_level=int(cfg.inputs.grid_level),
                density_fit=bool(cfg.inputs.density_fit),
                auxbasis=cfg.inputs.auxbasis,
                orientation_lock_strength=float(
                    cfg.inputs.orientation_lock_strength),
                xc="scan")
        except Exception as exc:  # noqa: BLE001 -- collect, report, exit 1
            failed.append(name)
            print(f"[seed-cache] {i}/{len(species)} {name}: FAILED "
                  f"({type(exc).__name__}: {exc})", flush=True)
            continue
        n_ok += 1
        print(f"[seed-cache] {i}/{len(species)} {name}: cached "
              f"({time.time() - t1:.1f}s)", flush=True)
    print(f"[seed-cache] done in {time.time() - t0:.1f}s: {n_ok} cached, "
          f"{len(failed)} failed"
          + (f" ({', '.join(failed)})" if failed else ""), flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
