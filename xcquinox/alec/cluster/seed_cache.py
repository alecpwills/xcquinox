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

Eval/val (pool) species need nothing here: the scan-pool cache already
covers all 214 pool species at the production identity -- point
``inputs.seed_cache_dir`` at a root whose ``_intermediates/`` contains
or symlinks those files.

Usage::

    python -m xcquinox.alec.cluster.seed_cache <grid_config.yaml> [--dry-run]
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


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("config", help="grid config / resolved_config YAML")
    p.add_argument("--dry-run", action="store_true",
                   help="list the enumerated species and identity; run "
                        "nothing")
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
    n_ok, failed = 0, []
    t0 = time.time()
    for i, at in enumerate(species, start=1):
        name = at.info["name"]
        entry = SpeciesEntry(name=name, charge=int(at.info.get("charge", 0)),
                             spin=int(at.info.get("spin", 0)), source="seed")
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
