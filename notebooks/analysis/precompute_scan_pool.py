#!/usr/bin/env python
"""Precompute SCAN meta-GGA total energies over the held-out BH76+W4-11 pool.

Offline companion to ``make_ablation_arch_figure.scan_pool_baseline``: SCAN is
the meta-GGA the ``_mgga`` archs clone, so a SCAN full-pool reaction-energy MAE
line is the natural reference next to PBE on the rung figures. Computing it is a
real multi-hour job (one KS-SCF per unique species at ``xc="scan"``), so it is
run ONCE, offline, and cached; the figure code only ever reads the cache.

For every unique species in the requested pool this runs
``external_refs.run_scf_with_cache(xc="scan", ...)`` (individually cached +
atomic on disk, exactly as ``benchmark_refs.generate_one`` drives the PBE/CCSD
references) and records ``{molecule_name: E_scan_hartree}`` into
``<out-dir>/scan_pool_energies_<basis>.json``. Both layers are resumable: a kill
mid-run keeps the JSON written so far AND every finished species' SCF npz, so a
re-invocation skips completed work and continues.

Point the figure at the result by passing ``cache_dir=<out-dir>`` to
``scan_pool_baseline`` (or copy the JSON next to the run dir, whose basis label
resolves the same filename). Absent this cache the figures simply omit the SCAN
line -- this script is never required to render.

Usage (example -- run offline; NOT part of the fast figure/test path):
    python notebooks/analysis/precompute_scan_pool.py \
        --basis def2-svp --grid 2 --pool all \
        --out-dir notebooks/analysis/scan_cache
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional


def _fmt_hms(seconds: float) -> str:
    """``H:MM:SS`` (mirrors ``benchmark_refs._fmt_hms`` so the progress line
    reads the same as the reference-generation jobs)."""
    s = max(0, int(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{sec:02d}"


def _scan_cache_name(basis: str) -> str:
    """Output filename for the SCAN cache at ``basis``. Kept IDENTICAL to
    ``make_ablation_arch_figure._scan_cache_name`` so the figure finds the file:
    the ``+DF`` suffix is dropped and any path-unsafe char maps to ``_``."""
    b = (basis or "def2-svp").replace("+DF", "").strip() or "def2-svp"
    safe = "".join(c if (c.isalnum() or c in "-.+") else "_" for c in b)
    return f"scan_pool_energies_{safe}.json"


def _load_pool(pool: str, *, basis: str, grid_level: int) -> Dict[str, object]:
    """``{name: MoleculeSpec}`` for the requested pool, sorted by name so the
    resume order is stable. ``all`` = BH76 + W4-11 (the held-out union)."""
    from xcquinox.alec import full_benchmark_pools as fbp
    loader = {
        "all": fbp.load_full_held_out_pools,
        "bh76": fbp.load_full_bh76,
        "w411": fbp.load_full_w411,
    }[pool]
    mol_specs, _reactions = loader(basis=basis, grid_level=grid_level)
    return dict(sorted(mol_specs.items()))


def _atomic_write_json(path: Path, payload: Dict[str, float]) -> None:
    """Write ``payload`` to ``path`` via a tmp file + ``os.replace`` so a kill
    mid-write never leaves a truncated JSON (matches the atomic-write policy the
    reference caches use)."""
    import os
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, path)


def run(pool: str, *, basis: str, grid_level: int, out_dir: Path,
        density_fit: bool = False, auxbasis: Optional[str] = None,
        force: bool = False) -> int:
    """Compute + cache SCAN total energies for every species in ``pool``.

    Returns the number of species that FAILED (0 == clean). Progress is printed
    per species with a running index/total + ETA (a multi-hour job must not look
    like a hang)."""
    from xcquinox.alec.benchmark_refs import _mol_spec_to_atoms
    from xcquinox.alec.external_refs import SpeciesEntry, run_scf_with_cache

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_json = out_dir / _scan_cache_name(basis)

    energies: Dict[str, float] = {}
    if cache_json.is_file() and not force:
        try:
            energies = {k: float(v) for k, v in json.loads(
                cache_json.read_text()).items()}
        except (json.JSONDecodeError, OSError):
            energies = {}

    mol_specs = _load_pool(pool, basis=basis, grid_level=grid_level)
    names = list(mol_specs)
    n = len(names)
    print(f"SCAN pool precompute: pool={pool} basis={basis} grid={grid_level} "
          f"density_fit={density_fit}  species={n}  cache={cache_json}")
    print(f"  {len(energies)}/{n} already cached (resume); "
          f"{'FORCING recompute' if force else 'skipping cached'}.")

    t0 = time.monotonic()
    n_fail = 0
    for k, name in enumerate(names, 1):
        if name in energies and not force:
            print(f"  [{k}/{n}] {name}: SKIP (cached E={energies[name]:.8f} Ha)")
            continue
        ms = mol_specs[name]
        spec = SpeciesEntry(name=ms.name, charge=int(ms.charge),
                            spin=int(ms.spin), source="benchmark")
        atoms = _mol_spec_to_atoms(ms)
        t1 = time.monotonic()
        try:
            scf = run_scf_with_cache(
                spec, atoms, cache_dir=out_dir, basis=basis,
                grid_level=grid_level, density_fit=density_fit,
                auxbasis=auxbasis, xc="scan")
            e_tot = scf.get("e_tot")
            if e_tot is None:
                raise ValueError("run_scf_with_cache returned e_tot=None")
            energies[name] = float(e_tot)
            _atomic_write_json(cache_json, energies)     # persist after each
            status = f"E={float(e_tot):.8f} Ha"
        except Exception as exc:  # one hard species must not sink the shard
            n_fail += 1
            status = f"FAIL ({type(exc).__name__}: {exc})"
        wall = time.monotonic() - t1
        done = k
        elapsed = time.monotonic() - t0
        eta = (elapsed / done) * (n - done) if done else 0.0
        print(f"  [{k}/{n}] {name} (q={ms.charge}, 2s={ms.spin}): {status}  "
              f"[{wall:.1f}s | elapsed {_fmt_hms(elapsed)} | ETA {_fmt_hms(eta)}]")

    print(f"done: {len(energies)}/{n} species cached, {n_fail} failed -> "
          f"{cache_json}")
    return n_fail


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--basis", default="def2-svp",
                   help="SCF basis (default: def2-svp). The cache filename drops "
                        "any +DF suffix -- pair a DF run with --density-fit.")
    p.add_argument("--grid", type=int, default=2, dest="grid_level",
                   help="pyscf DFT grid level for the SCAN SCF (default: 2)")
    p.add_argument("--pool", default="all", choices=("all", "bh76", "w411"),
                   help="held-out pool to cover (default: all = BH76 + W4-11)")
    p.add_argument("--out-dir", default="scan_cache",
                   help="directory for the SCAN cache JSON + _intermediates/ "
                        "(default: ./scan_cache)")
    p.add_argument("--density-fit", action="store_true",
                   help="run the SCAN SCF with density fitting (match a DF run)")
    p.add_argument("--auxbasis", default=None,
                   help="auxiliary basis for --density-fit (default: auto)")
    p.add_argument("--force", action="store_true",
                   help="recompute every species even if already cached")
    args = p.parse_args(argv)

    n_fail = run(args.pool, basis=args.basis, grid_level=args.grid_level,
                 out_dir=Path(args.out_dir).expanduser(),
                 density_fit=args.density_fit, auxbasis=args.auxbasis,
                 force=args.force)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
