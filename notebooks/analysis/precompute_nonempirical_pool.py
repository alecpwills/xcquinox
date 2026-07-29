#!/usr/bin/env python
"""Precompute the nonempirical-functional calibration pool (energies + eps).

Offline companion to the DFS Eq. 21 machinery: Dick & Fernandez-Serra (PRB
104, L161109 (2021)) fix the energy<->density conversion slope gamma as the
zero-intercept regression of WTMAD-2 on the per-electron L1 density error
eps (their Eq. 20) across six nonempirical functionals (PW91, PBE, TPSS,
revTPSS, SCAN, PBE0; gamma = 1084.87 kcal/mol on their axes). This script
produces the same calibration data ON OUR AXES: for each requested
functional it records, per held-out species,

    e_tot            converged KS-SCF total energy (Hartree)
    density_eps_l1   sum_i(w_i |rho_xc - rho_ref|_i) / N_e   (Letter Eq. 20)
    n_electrons      N_e = sum_i(w_i rho_ref_i)  (quadrature, reference)
    grid_weight_sum  sum_i(w_i)

against the SAME CCSD benchmark reference densities the NN eval uses
(``rho_ref_grid`` + ``grid_weights`` from the benchmark refs npz). The
figure code then fits gamma exactly the DFS way (zero-intercept slope over
the functional set) -- see ``make_ablation_arch_figure.gamma_zero_intercept``.
The fitted slope is "gamma on OUR basis/grid/eval set": it will NOT equal
1084.87, which belongs to the Letter's diet-GMTKN55/G2-97 axes.

Energies reuse ``external_refs.run_scf_with_cache`` (per-species npz cache
under ``<out-dir>/_intermediates``, ``xc``-tagged filenames; ``xc="pbe"``
follows the exact benchmark-refs recipe and cache identity). The out-dir
must be FRESH / job-owned, never the production refs dir -- legacy pre-e_tot
PBE intermediates there would load with ``e_tot=None`` and fail every pbe
entry loudly. Densities are evaluated on the STORED
first-attempt grid and shape-checked against the reference grid -- a
mismatch is a per-species failure, never a silent skip (the c2 grid-drift
lesson, HISTORY Phase 35). The PBE density takes the no-SCF fast path from
``rho_pbe_grid`` when the refs npz carries it.

Both cache layers are resumable (atomic JSON after every (species, xc);
per-species SCF npz). Like the SCAN pool cache, this is never required to
render -- figures omit the DFS-gamma variant when the cache is absent.

Usage (offline, multi-hour; submitted on the cluster after the live sweep
completes -- see hpcjobs/nonempirical_pool.sbatch):
    python notebooks/analysis/precompute_nonempirical_pool.py \
        --basis "6-311++G(3df,2pd)" --grid 3 --density-fit \
        --orientation-lock-strength 3e-05 \
        --density-refs /gpfs/scratch/awills/external_refs_bench_6311ppg3df2pd_g3 \
        --out-dir nonempirical_cache
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable, Dict, Optional, Sequence

NONEMPIRICAL_XCS = ("pw91", "pbe", "tpss", "revtpss", "scan", "pbe0")


def _fmt_hms(seconds: float) -> str:
    """``H:MM:SS`` (mirrors ``benchmark_refs._fmt_hms``)."""
    s = max(0, int(seconds))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:d}:{m:02d}:{sec:02d}"


def _pool_cache_name(basis: str) -> str:
    """Cache filename at ``basis``: same slug rule as the SCAN pool cache
    (``+DF`` dropped, path-unsafe chars -> ``_``) so both caches sit side by
    side and resolve from the same run basis label."""
    b = (basis or "def2-svp").replace("+DF", "").strip() or "def2-svp"
    safe = "".join(c if (c.isalnum() or c in "-.+") else "_" for c in b)
    return f"nonempirical_pool_{safe}.json"


def _atomic_write_json(path: Path, payload: Dict) -> None:
    """tmp + ``os.replace`` so a kill mid-write never truncates the cache."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
    os.replace(tmp, path)


def _load_pool(pool: str, *, basis: str, grid_level: int) -> Dict[str, object]:
    """``{name: MoleculeSpec}`` sorted by name (stable resume order)."""
    from xcquinox.alec import full_benchmark_pools as fbp
    loader = {
        "all": fbp.load_full_held_out_pools,
        "bh76": fbp.load_full_bh76,
        "w411": fbp.load_full_w411,
    }[pool]
    mol_specs, _reactions = loader(basis=basis, grid_level=grid_level)
    return dict(sorted(mol_specs.items()))


def _load_ref_arrays(ms, refs_dir: Optional[Path], *,
                     basis: Optional[str] = None,
                     grid_level: Optional[int] = None,
                     density_fit: bool = False,
                     orientation_lock_strength: float = 0.0):
    """``(rho_ref, weights, rho_pbe_or_None)`` for one species, from its
    resolved ``external_data_path`` or ``<refs_dir>/<name>.npz``. Raises
    KeyError/FileNotFoundError loudly when the reference is unusable.

    When ``basis``/``grid_level`` are given, the reference's identity stamps
    (``basis_used``, ``grid_level_used``, ``density_fit_used``,
    ``orientation_lock_strength``) are verified against the run parameters
    via ``benchmark_refs._benchmark_npz_is_complete`` -- the same predicate
    the reference generator treats as authoritative -- and a mismatch raises
    ValueError. The shape gate alone cannot catch a basis/DF/lock mismatch
    (grid size is set by (molecule, grid_level) only)."""
    import numpy as np
    p = getattr(ms, "external_data_path", None)
    if p is None and refs_dir is not None:
        p = Path(refs_dir) / f"{ms.name}.npz"
    if p is None or not Path(p).is_file():
        raise FileNotFoundError(f"no benchmark refs npz for {ms.name}")
    if basis is not None and grid_level is not None:
        from xcquinox.alec.benchmark_refs import _benchmark_npz_is_complete
        if not _benchmark_npz_is_complete(
                p, basis=basis, grid_level=grid_level,
                orientation_lock_strength=orientation_lock_strength,
                density_fit=density_fit):
            raise ValueError(
                f"refs identity mismatch for {ms.name}: {p} is not a "
                f"complete CCSD reference stamped (basis={basis!r}, "
                f"grid={grid_level}, DF={density_fit}, "
                f"lock={orientation_lock_strength}) -- wrong refs dir or "
                "wrong run parameters")
    with np.load(Path(p), allow_pickle=True) as z:
        rho_ref = np.asarray(z["rho_ref_grid"])
        weights = np.asarray(z["grid_weights"])
        rho_pbe = (np.asarray(z["rho_pbe_grid"])
                   if "rho_pbe_grid" in z.files else None)
    return rho_ref, weights, rho_pbe


def _density_on_grid(scf: Dict, ms, *, basis: str) -> "object":
    """Spin-summed density of the converged SCF on ITS OWN stored grid
    (the first-attempt minao-pruned grid, matching the reference grid for the
    same (mol, basis, grid_level) -- the recipe of benchmark_refs/external_refs).
    ``ms.atom`` is the pyscf-format geometry string in Angstrom (the
    MoleculeSpec contract), matching pyscf's default input unit and the
    ``unit="angstrom"`` used by ``run_scf_with_cache``."""
    import numpy as np
    from pyscf import gto
    from pyscf.dft import numint
    dm = np.asarray(scf["dm"])
    dm_tot = dm[0] + dm[1] if dm.ndim == 3 else dm
    mol = gto.M(atom=ms.atom, basis=basis,
                charge=int(ms.charge), spin=int(ms.spin), verbose=0)
    ao = numint.eval_ao(mol, np.asarray(scf["grid_coords"]), deriv=0)
    return np.einsum("ij,gj,gi->g", dm_tot, ao, ao)


def _default_scf(spec, atoms, **kw) -> Dict:
    from xcquinox.alec.external_refs import run_scf_with_cache
    return run_scf_with_cache(spec, atoms, **kw)


def _default_spec_atoms(ms):
    """``(SpeciesEntry, atoms)`` for one MoleculeSpec (real implementations;
    seam-replaceable in tests)."""
    from xcquinox.alec.external_refs import SpeciesEntry
    from xcquinox.alec.benchmark_refs import _mol_spec_to_atoms
    spec = SpeciesEntry(name=ms.name, charge=int(ms.charge),
                        spin=int(ms.spin), source="benchmark")
    return spec, _mol_spec_to_atoms(ms)


def run(pool: str, *, basis: str, grid_level: int, out_dir: Path,
        xcs: Sequence[str] = NONEMPIRICAL_XCS,
        density_refs: Optional[Path] = None,
        density_fit: bool = False, auxbasis: Optional[str] = None,
        orientation_lock_strength: float = 0.0,
        force: bool = False,
        _scf: Optional[Callable] = None,
        _pool_loader: Optional[Callable] = None,
        _refs_loader: Optional[Callable] = None,
        _density: Optional[Callable] = None,
        _spec_atoms: Optional[Callable] = None) -> int:
    """Fill ``<out-dir>/nonempirical_pool_<basis>.json`` with
    ``{name: {xc: {e_tot, density_eps_l1, n_electrons, grid_weight_sum}}}``.

    ``orientation_lock_strength`` must match the reference generation (the
    dfs6311 production refs are stamped 3e-05): the SCFs here use the same
    traceless-quadrupole bias so degenerate-state radicals (OH/NO 2-Pi class)
    converge in the reference orientation -- otherwise an orientation
    artifact enters eps as if it were functional error. The stamp check in
    ``_load_ref_arrays`` fails loudly on a mismatch. Returns the number of
    FAILED (species, xc) entries. The ``_scf`` / ``_pool_loader`` /
    ``_refs_loader`` / ``_density`` / ``_spec_atoms`` parameters are test
    seams (default: the real implementations)."""
    from xcquinox.alec.evaluation import density_eps_terms

    scf_fn = _scf or _default_scf
    pool_fn = _pool_loader or _load_pool
    if _refs_loader is not None:
        refs_fn = _refs_loader
    else:
        def refs_fn(ms, refs_dir):
            return _load_ref_arrays(
                ms, refs_dir, basis=basis, grid_level=grid_level,
                density_fit=density_fit,
                orientation_lock_strength=orientation_lock_strength)
    dens_fn = _density or _density_on_grid
    spec_atoms_fn = _spec_atoms or _default_spec_atoms

    if density_refs is not None:
        # make the pool loader resolve external_data_path transparently
        os.environ["XCQUINOX_BENCH_REFS_DIR"] = str(density_refs)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_json = out_dir / _pool_cache_name(basis)

    cache: Dict[str, Dict[str, Dict[str, float]]] = {}
    if cache_json.is_file() and not force:
        try:
            cache = json.loads(cache_json.read_text())
        except (json.JSONDecodeError, OSError):
            cache = {}

    mol_specs = pool_fn(pool, basis=basis, grid_level=grid_level)
    names = list(mol_specs)
    n_jobs = len(names) * len(xcs)
    print(f"nonempirical pool precompute: pool={pool} basis={basis} "
          f"grid={grid_level} density_fit={density_fit} "
          f"lock={orientation_lock_strength} xcs={','.join(xcs)} "
          f"species={len(names)} jobs={n_jobs} cache={cache_json}")
    n_done0 = sum(1 for nm in cache for _ in cache[nm])
    print(f"  {n_done0}/{n_jobs} already cached (resume); "
          f"{'FORCING recompute' if force else 'skipping cached'}.")

    t0 = time.monotonic()
    n_fail = 0
    k = 0
    for name in names:
        ms = mol_specs[name]
        try:
            rho_ref, weights, rho_pbe = refs_fn(ms, density_refs)
        except Exception as exc:
            n_fail += len(xcs)
            k += len(xcs)
            print(f"  [{k}/{n_jobs}] {name}: REFS FAIL "
                  f"({type(exc).__name__}: {exc}) -- all xcs skipped")
            continue
        for xc in xcs:
            k += 1
            if not force and cache.get(name, {}).get(xc):
                print(f"  [{k}/{n_jobs}] {name}/{xc}: SKIP (cached)")
                continue
            t1 = time.monotonic()
            try:
                spec, atoms = spec_atoms_fn(ms)
                scf = scf_fn(spec, atoms, cache_dir=out_dir, basis=basis,
                             grid_level=grid_level, density_fit=density_fit,
                             auxbasis=auxbasis,
                             orientation_lock_strength=orientation_lock_strength,
                             xc=xc)
                e_tot = scf.get("e_tot")
                if e_tot is None:
                    raise ValueError("run_scf_with_cache returned e_tot=None")
                if xc == "pbe" and rho_pbe is not None:
                    rho = rho_pbe          # no-SCF fast path for the density
                else:
                    rho = dens_fn(scf, ms, basis=basis)
                if rho.shape != rho_ref.shape:
                    raise ValueError(
                        f"grid mismatch: rho {rho.shape} vs "
                        f"rho_ref {rho_ref.shape} (c2-class drift)")
                eps, n_e, wsum = density_eps_terms(rho, rho_ref, weights)
                cache.setdefault(name, {})[xc] = {
                    "e_tot": float(e_tot),
                    "density_eps_l1": float(eps),
                    "n_electrons": float(n_e),
                    "grid_weight_sum": float(wsum),
                }
                _atomic_write_json(cache_json, cache)
                status = f"E={float(e_tot):.8f} Ha eps={float(eps):.3e}"
            except Exception as exc:   # one hard entry must not sink the job
                n_fail += 1
                status = f"FAIL ({type(exc).__name__}: {exc})"
            wall = time.monotonic() - t1
            elapsed = time.monotonic() - t0
            eta = (elapsed / k) * (n_jobs - k) if k else 0.0
            print(f"  [{k}/{n_jobs}] {name}/{xc} (q={ms.charge}, "
                  f"2s={ms.spin}): {status}  [{wall:.1f}s | elapsed "
                  f"{_fmt_hms(elapsed)} | ETA {_fmt_hms(eta)}]")

    n_done = sum(len(v) for v in cache.values())
    print(f"done: {n_done}/{n_jobs} (species, xc) entries cached, "
          f"{n_fail} failed -> {cache_json}")
    return n_fail


def main(argv: Optional[list] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--basis", default="def2-svp")
    p.add_argument("--grid", type=int, default=2, dest="grid_level",
                   help="pyscf DFT grid level (default: 2)")
    p.add_argument("--pool", default="all", choices=("all", "bh76", "w411"))
    p.add_argument("--out-dir", default="nonempirical_cache",
                   help="directory for the cache JSON + _intermediates/")
    p.add_argument("--xc", action="append", default=None,
                   help="functional to include (repeatable; default: the "
                        f"Letter's six: {', '.join(NONEMPIRICAL_XCS)})")
    p.add_argument("--density-refs", default=None,
                   help="benchmark refs dir (<name>.npz with rho_ref_grid + "
                        "grid_weights); also exported as "
                        "XCQUINOX_BENCH_REFS_DIR for the pool loader")
    p.add_argument("--density-fit", action="store_true")
    p.add_argument("--auxbasis", default=None)
    p.add_argument("--orientation-lock-strength", type=float, default=0.0,
                   help="traceless-quadrupole orientation-lock strength; "
                        "MUST match the reference generation (dfs6311 "
                        "production refs: 3e-05); a mismatch against the "
                        "reference stamps fails loudly per species")
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)

    n_fail = run(args.pool, basis=args.basis, grid_level=args.grid_level,
                 out_dir=Path(args.out_dir).expanduser(),
                 xcs=tuple(args.xc) if args.xc else NONEMPIRICAL_XCS,
                 density_refs=(Path(args.density_refs).expanduser()
                               if args.density_refs else None),
                 density_fit=args.density_fit, auxbasis=args.auxbasis,
                 orientation_lock_strength=args.orientation_lock_strength,
                 force=args.force)
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
