"""Density-only CCSD reference generation for the held-out benchmark pool.

Generates per-species CCSD reference densities for the full W4-11 + BH76
benchmark pool (~214 species) so the held-out evaluation can report
NN-vs-CCSD and PBE-vs-CCSD density errors. Unlike the training-pool
pipeline in :mod:`xcquinox.alec.external_refs` (``precompute_all``), this
generator runs ONLY stages 1+2 (PBE SCF + CCSD density on the SCF grid)
and skips the OEP inversion entirely: the eval-side loader
(:func:`xcquinox.alec.data._load_external_data`) gates every npz key
independently, so a density-only file ``{rho_ref_grid, ref_density_method,
grid_level_used, basis_used}`` loads cleanly with no ``vxc_ref`` /
``dm_target``. Those keys (and the fragile per-species OEP cascade behind
them) are a TRAINING-refs requirement only.

Reuses :func:`external_refs.run_scf_with_cache` and
:func:`external_refs.run_ccsd_with_cache` verbatim. Both already handle
the empty-spin-channel density-fitting fallback (non-DF CCSD when
``min(mol.nelec) == 0``, e.g. the H atom -- pyscf's DF-UCCSD outcore path
builds a zero-chunk HDF5 dataset there) and the UKS/UCCSD dispatch for
``spin > 0`` species, so open-shell radicals, anions and BH76 transition
structures need no special casing here. Do not bypass those guards.

Resumable + SLURM-array friendly: the per-species intermediates and the
final npz are written atomically (tempfile + fsync + ``os.replace``);
complete final files are skipped on re-run; ``--shard i/N`` slices the
sorted species list into disjoint contiguous chunks so concurrent array
tasks never duplicate work. A failing species is logged FAIL and the
shard continues (exit code 1 iff any species failed).

Usage (cluster, one shard of 16):
    python -m xcquinox.alec.benchmark_refs \
        --out-dir /gpfs/scratch/awills/external_refs_bench_svp_g2 \
        --pool all --basis def2-svp --grid-level 2 --shard 3/16

The tzvpd variant adds ``--basis def2-tzvpd --density-fit
--auxbasis def2-universal-jkfit`` (def2-tzvpd has no dedicated -jkfit).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from xcquinox.alec.config import MoleculeSpec
from xcquinox.alec.external_refs import (
    RunLog,
    SpeciesEntry,
    _fsync_dir,
    _fsync_file,
    run_ccsd_with_cache,
    run_scf_with_cache,
)
from xcquinox.alec import full_benchmark_pools

POOL_CHOICES: Tuple[str, ...] = ("bh76", "w411", "all")

# Keys every density-only reference npz must carry. All four are already in
# data._ALLOWED_EXTERNAL_KEYS; rho_ref_grid is the only one the density
# metrics consume, the others are identity guards (the loader raises on a
# grid_level mismatch, and basis_used keys the staleness check here).
_DENSITY_NPZ_KEYS: frozenset = frozenset(
    {"rho_ref_grid", "ref_density_method", "grid_level_used", "basis_used"})


def _mol_spec_to_atoms(ms: MoleculeSpec):
    """Inverse of ``spec_builder.atoms_to_pyscf_str``: parse the pyscf atom
    string (``"Sym x y z; ..."``, Angstrom) back into ``ase.Atoms`` with
    ``info`` carrying name/charge/spin."""
    from ase import Atoms

    syms: List[str] = []
    xyz: List[Tuple[float, float, float]] = []
    for tok in ms.atom.split(";"):
        parts = tok.split()
        if not parts:
            continue
        syms.append(parts[0])
        xyz.append((float(parts[1]), float(parts[2]), float(parts[3])))
    at = Atoms(symbols=syms, positions=xyz)
    at.info["name"] = ms.name
    at.info["charge"] = int(ms.charge)
    at.info["spin"] = int(ms.spin)
    return at


def _atomic_savez(path, **arrays) -> None:
    """Atomic + durable npz write: tempfile -> fsync file -> ``os.replace``
    -> fsync dir, mirroring ``run_scf_with_cache`` (EXTREF-04). An
    interrupted write can never leave a partial final npz that a later
    run would treat as a complete reference."""
    import tempfile

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(target.parent), suffix=".npz")
    os.close(fd)
    try:
        np.savez_compressed(tmp_name, **arrays)
        _fsync_file(tmp_name)
        os.replace(tmp_name, target)
        _fsync_dir(target.parent)
    except Exception:
        if os.path.exists(tmp_name):
            os.unlink(tmp_name)
        raise


def _benchmark_npz_is_complete(path, *, basis: str, grid_level: int) -> bool:
    """True iff ``path`` is a readable density-only reference matching this
    run's basis + grid_level. Deliberately NOT ``external_refs._npz_is_complete``
    (that demands the OEP keys vxc_ref/dm_target). A corrupt or partial file
    reads as incomplete -> regenerated."""
    p = Path(path)
    if not p.is_file():
        return False
    try:
        with np.load(p, allow_pickle=False) as z:
            files = set(z.files)
            if not {"rho_ref_grid", "ref_density_method"} <= files:
                return False
            if str(z["ref_density_method"]) != "ccsd":
                return False
            if "basis_used" in files and str(z["basis_used"]) != str(basis):
                return False
            if ("grid_level_used" in files
                    and int(z["grid_level_used"]) != int(grid_level)):
                return False
    except Exception:
        return False
    return True


def load_benchmark_species(pool: str = "all", *, basis: str = "def2-svp",
                           grid_level: int = 2) -> Dict[str, MoleculeSpec]:
    """Sorted ``{name: MoleculeSpec}`` for the requested benchmark pool.

    Sorting fixes the species order so ``--shard i/N`` slices are stable
    across invocations and disjoint across array tasks."""
    if pool not in POOL_CHOICES:
        raise ValueError(f"pool must be one of {POOL_CHOICES}, got {pool!r}")
    loader = {
        "bh76": full_benchmark_pools.load_full_bh76,
        "w411": full_benchmark_pools.load_full_w411,
        "all": full_benchmark_pools.load_full_held_out_pools,
    }[pool]
    mol_specs, _reactions = loader(basis=basis, grid_level=grid_level)
    return dict(sorted(mol_specs.items()))


def resolve_slice(n: int, *, shard: Optional[str] = None,
                  species_slice: Optional[str] = None) -> slice:
    """``--shard i/N`` (1-based) -> the i-th of N disjoint contiguous chunks
    covering 0..n; ``--species-slice A:B`` -> an explicit python slice."""
    if shard and species_slice:
        raise ValueError("--shard and --species-slice are mutually exclusive")
    if species_slice:
        a, _, b = species_slice.partition(":")
        return slice(int(a) if a else 0, int(b) if b else n)
    if shard:
        i_str, _, total_str = shard.partition("/")
        i, total = int(i_str), int(total_str)
        if total < 1 or not 1 <= i <= total:
            raise ValueError(f"--shard must be i/N with 1 <= i <= N, got {shard!r}")
        per = -(-n // total)  # ceil division: chunks cover the pool exactly
        return slice((i - 1) * per, min(i * per, n))
    return slice(0, n)


def generate_one(ms: MoleculeSpec, *, out_dir, basis: str, grid_level: int,
                 density_fit: bool = False,
                 auxbasis: Optional[str] = None) -> str:
    """Generate (or skip) one species' density-only reference npz.

    Returns ``"SKIP"`` when the final npz is already complete for this
    basis/grid, else runs SCF + CCSD (both stages individually cached and
    atomic, so a killed job resumes mid-species) and writes the final npz.
    ``source='benchmark'`` on the SpeciesEntry is a provenance tag only --
    geometry comes from the MoleculeSpec, never ``resolve_geometry``."""
    final = Path(out_dir) / f"{ms.name}.npz"
    if _benchmark_npz_is_complete(final, basis=basis, grid_level=grid_level):
        return "SKIP"
    spec = SpeciesEntry(name=ms.name, charge=int(ms.charge),
                        spin=int(ms.spin), source="benchmark")
    atoms = _mol_spec_to_atoms(ms)
    scf = run_scf_with_cache(
        spec, atoms, cache_dir=out_dir, basis=basis, grid_level=grid_level,
        density_fit=density_fit, auxbasis=auxbasis)
    cc = run_ccsd_with_cache(
        spec, atoms, scf_payload=scf, cache_dir=out_dir, basis=basis,
        grid_level=grid_level, density_fit=density_fit, auxbasis=auxbasis)
    _atomic_savez(
        final,
        rho_ref_grid=np.asarray(cc["rho_ref_grid"]),
        ref_density_method=np.array("ccsd"),
        grid_level_used=np.array(int(grid_level)),
        basis_used=np.array(str(basis)),
    )
    return "OK"


def _fmt_hms(seconds: float) -> str:
    s = max(0, int(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{sec:02d}"


def run_shard(names: List[str], mol_specs: Dict[str, MoleculeSpec], *,
              out_dir, basis: str, grid_level: int, density_fit: bool = False,
              auxbasis: Optional[str] = None, shard_label: str = "1/1",
              progress: bool = True) -> int:
    """Generate every species in ``names``; returns the FAIL count.

    Per-species outcomes go to an atomic RunLog ledger under
    ``<out_dir>/_runlogs/shard_<i>_of_<N>/`` (one dir per shard so
    concurrent array tasks never clobber each other's partial JSON), and a
    per-species progress line with ETA is printed (long jobs must not look
    like hangs)."""
    ledger_dir = (Path(out_dir) / "_runlogs"
                  / f"shard_{shard_label.replace('/', '_of_')}")
    log = RunLog(cache_dir=ledger_dir)
    log.start(names)
    n = len(names)
    t0 = time.monotonic()
    n_fail = 0
    for k, name in enumerate(names, 1):
        ms = mol_specs[name]
        t1 = time.monotonic()
        err: Optional[str] = None
        try:
            status = generate_one(ms, out_dir=out_dir, basis=basis,
                                  grid_level=grid_level,
                                  density_fit=density_fit, auxbasis=auxbasis)
        except Exception as exc:  # log + continue: one hard species must not
            status = "FAIL"      # sink the whole shard's remaining work
            err = f"{type(exc).__name__}: {exc}"
            n_fail += 1
        wall = time.monotonic() - t1
        log.record_result(name=name, charge=ms.charge, spin=ms.spin,
                          status=status, wall_clock_s=wall, error_msg=err)
        if progress:
            elapsed = time.monotonic() - t0
            eta = elapsed / k * (n - k)
            print(f"[gen {k}/{n} shard {shard_label}] {name} "
                  f"(q={ms.charge},2S={ms.spin}) status={status} "
                  f"wall={wall:.1f}s | elapsed {_fmt_hms(elapsed)} "
                  f"ETA {_fmt_hms(eta)}", flush=True)
            if err:
                print(f"    {err}", flush=True)
    log.finalize()
    return n_fail


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", required=True,
                   help="reference dir; final <name>.npz per species, "
                        "_intermediates/ + _runlogs/ underneath")
    p.add_argument("--pool", choices=POOL_CHOICES, default="all",
                   help="benchmark pool to cover (default all = BH76+W4-11)")
    p.add_argument("--basis", default="def2-svp")
    p.add_argument("--grid-level", type=int, default=2,
                   help="MUST match the eval run's resolved grid_level "
                        "(the loader raises on mismatch)")
    p.add_argument("--density-fit", action="store_true",
                   help="DF SCF/CCSD (use for def2-tzvpd; pass --auxbasis)")
    p.add_argument("--auxbasis", default=None,
                   help="DF auxiliary basis (e.g. def2-universal-jkfit)")
    p.add_argument("--shard", default=None,
                   help="i/N: run the i-th of N disjoint slices (1-based)")
    p.add_argument("--species-slice", default=None,
                   help="A:B explicit python slice over the sorted species "
                        "list (mutually exclusive with --shard)")
    p.add_argument("--no-progress", action="store_true")
    args = p.parse_args(argv)

    if args.auxbasis and not args.density_fit:
        p.error("--auxbasis only makes sense with --density-fit")

    mol_specs = load_benchmark_species(args.pool, basis=args.basis,
                                       grid_level=args.grid_level)
    all_names = list(mol_specs)
    sl = resolve_slice(len(all_names), shard=args.shard,
                       species_slice=args.species_slice)
    names = all_names[sl]
    shard_label = args.shard or "1/1"
    print(f"benchmark_refs: pool={args.pool} ({len(all_names)} species), "
          f"slice [{sl.start}:{sl.stop}] -> {len(names)} this task, "
          f"basis={args.basis} grid_level={args.grid_level} "
          f"density_fit={args.density_fit} auxbasis={args.auxbasis} "
          f"out_dir={args.out_dir}", flush=True)
    if not names:
        print("benchmark_refs: empty slice, nothing to do", flush=True)
        return 0
    n_fail = run_shard(names, mol_specs, out_dir=args.out_dir,
                       basis=args.basis, grid_level=args.grid_level,
                       density_fit=args.density_fit, auxbasis=args.auxbasis,
                       shard_label=shard_label,
                       progress=not args.no_progress)
    if n_fail:
        print(f"benchmark_refs: {n_fail}/{len(names)} species FAILED "
              "(see _runlogs ledger); rerun after triage -- complete species "
              "are skipped", flush=True)
        return 1
    print(f"benchmark_refs: all {len(names)} species complete", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
