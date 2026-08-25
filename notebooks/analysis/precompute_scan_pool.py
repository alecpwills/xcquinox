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

DENSITY LEG (``--with-density``). ``run_scf_with_cache`` persists the AO-basis
density matrix beside ``e_tot``, so SCAN's density error against the CCSD
references costs no second SCF: the stored dm is contracted onto the reference
grid and scored with the SAME formulas the PBE columns use
(``evaluation.pbe_density_errors`` for the grid-weighted RMSE,
``evaluation.density_eps_terms`` for the Letter Eq. 20 per-electron eps). The
result lands in ``<out-dir>/scan_pool_density_<basis>.json``.

ORIENTATION LOCK. ``--orientation-lock`` must match the lock the references were
generated with. For a degenerate 2-Pi radical (CH, NO, OH) an unlocked SCAN SCF
settles on a different member of the degenerate manifold than the reference, and
the resulting density error is a component mismatch rather than a functional
error -- the failure documented in ``notebooks/analysis/DENSITY_DIAGNOSIS.md``.
The lock participates in the intermediate cache name, so locked and unlocked SCAN
caches cannot be confused for one another.

Point the figure at the result by passing ``cache_dir=<out-dir>`` to
``scan_pool_baseline`` (or copy the JSON next to the run dir, whose basis label
resolves the same filename). Absent this cache the figures simply omit the SCAN
line -- this script is never required to render.

Usage (example -- run offline; NOT part of the fast figure/test path):
    python notebooks/analysis/precompute_scan_pool.py \
        --basis def2-svp --grid 2 --pool all \
        --out-dir notebooks/analysis/scan_cache

    # production basis, with the density leg, locked to match the references
    python notebooks/analysis/precompute_scan_pool.py \
        --basis '6-311++G(3df,2pd)' --grid 3 --density-fit --pool all \
        --orientation-lock 3e-05 --with-density --refs-dir <density refs dir> \
        --out-dir <scan cache dir>
"""
from __future__ import annotations

import argparse
import json
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


def _scan_density_cache_name(basis: str) -> str:
    """Output filename for the SCAN density cache at ``basis``. Same slug rule as
    :func:`_scan_cache_name` so the two caches sit side by side and the figure
    resolves both from one basis label."""
    b = (basis or "def2-svp").replace("+DF", "").strip() or "def2-svp"
    safe = "".join(c if (c.isalnum() or c in "-.+") else "_" for c in b)
    return f"scan_pool_density_{safe}.json"


def _reference_ao(refs_dir: Path, name: str, *, basis: str, grid_level: int,
                  density_fit: bool, orientation_lock_strength: float):
    """``ao_grid`` on the grid the reference density lives on, read from the
    reference's OWN cached CCSD intermediate. ``None`` when it is not on disk.

    The grid CANNOT be rebuilt from the molecule and grid level: PySCF prunes
    the grid during ``kernel()``, so a freshly built ``Grids`` object has more
    points than the reference was written on (measured: 10128 vs 9264 for
    H2O/def2-svp/grid 1). ``external_refs.run_ccsd_with_cache`` says exactly
    this -- it takes coords/weights from the SCF payload "so the CCSD grid is
    identical to the SCF grid (PySCF prunes the grid during kernel(), so
    rebuilding from scratch ...)". Its intermediate stores the resulting
    ``ao_grid``, which is the only faithful source here; the SCAN SCF payload
    carries no grid of its own.
    """
    import numpy as np

    from xcquinox.alec.external_refs import _intermediate_cache_name

    fname = _intermediate_cache_name(
        name, grid_level=grid_level, basis=basis, density_fit=density_fit,
        kind="ccsd", orientation_lock_strength=orientation_lock_strength)
    path = Path(refs_dir) / "_intermediates" / fname
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as z:
            if "ao_grid" not in z.files:
                return None
            return np.asarray(z["ao_grid"])
    except (OSError, ValueError):
        return None


def _n_electrons(ms) -> float:
    """Electron count from pyscf's own bookkeeping -- charge-correct by
    construction, so the quadrature check needs no atomic-number table."""
    from pyscf import gto

    mol = gto.M(atom=ms.atom, basis=ms.basis, charge=int(ms.charge),
                spin=int(ms.spin), verbose=0)
    return float(mol.nelectron)


def scan_density_record(dm, ao, weights, rho_ref, *,
                        n_electrons_expected: Optional[float] = None,
                        n_electrons_tol: float = 1e-3) -> Dict[str, float]:
    """SCAN-vs-reference density errors for one species, from its stored dm.

    ``rho_scan = sum_ij dm_ij ao_gi ao_gj`` on the reference grid, then the SAME
    two metrics the PBE columns carry, computed by the SAME functions so the SCAN
    and PBE legs can never drift apart:
    ``evaluation.pbe_density_errors`` (grid-weight-averaged RMSE/L1) and
    ``evaluation.density_eps_terms`` (Letter Eq. 20 per-electron eps, normalized
    by the quadrature electron count of the REFERENCE density).

    GRID GUARD. ``sum(w * rho_scan)`` is the quadrature electron count of the
    SCAN density and must reproduce the species' true electron count. It cannot
    if ``ao`` was built on a different grid than ``weights`` came from, so this
    is the check that the recomputed grid really is the reference's grid --
    a mismatch raises rather than silently producing a plausible-looking error.
    """
    import numpy as np

    from xcquinox.alec.evaluation import density_eps_terms, pbe_density_errors

    dm = np.asarray(dm)
    dm_tot = dm[0] + dm[1] if dm.ndim == 3 else dm
    ao = np.asarray(ao)
    weights = np.asarray(weights)
    rho_ref = np.asarray(rho_ref)
    if not (ao.shape[0] == weights.shape[0] == rho_ref.shape[0]):
        raise ValueError(
            f"grid length mismatch: ao {ao.shape[0]}, weights "
            f"{weights.shape[0]}, rho_ref {rho_ref.shape[0]}")
    rho_scan = np.einsum("ij,gj,gi->g", dm_tot, ao, ao)
    n_e_scan = float(np.sum(weights * rho_scan))
    if n_electrons_expected is not None:
        if abs(n_e_scan - float(n_electrons_expected)) > n_electrons_tol:
            raise ValueError(
                f"SCAN density integrates to {n_e_scan:.6f} electrons, expected "
                f"{float(n_electrons_expected):.6f} -- the AO grid does not match "
                "the grid the reference weights were written for")
    md = {"rho_grid": rho_scan, "rho_ref_grid": rho_ref,
          "grid_weights": weights}
    rmse, l1 = pbe_density_errors(md)
    eps, n_e_ref, wsum = density_eps_terms(rho_scan, rho_ref, weights)
    return {"density_rmse_scan": float(rmse), "density_l1_scan": float(l1),
            "density_eps_l1_scan": float(eps), "n_electrons": float(n_e_ref),
            "n_electrons_scan": n_e_scan, "grid_weight_sum": float(wsum)}


def _reference_density(refs_dir: Path, name: str):
    """``(rho_ref_grid, grid_weights)`` from a benchmark density reference npz,
    or ``(None, None)`` when the species has no reference / the file predates the
    ``grid_weights`` key (``xcquinox.alec.benchmark_refs`` writes both)."""
    import numpy as np

    p = Path(refs_dir) / f"{name}.npz"
    if not p.is_file():
        return None, None
    try:
        with np.load(p, allow_pickle=False) as z:
            if not {"rho_ref_grid", "grid_weights"} <= set(z.files):
                return None, None
            return np.asarray(z["rho_ref_grid"]), np.asarray(z["grid_weights"])
    except (OSError, ValueError):
        return None, None


def _reference_lock(refs_dir: Path, name: str) -> Optional[float]:
    """``orientation_lock_strength`` stamped on a species' reference npz, or
    ``None`` when the file has no stamp.

    A MISSING stamp is not 0.0: pre-stamp references were written before the key
    existed and their lock is unknown, which is exactly the blindness that let
    the CH/NO references drift out of agreement with the training SCF. Callers
    must treat ``None`` as "cannot verify" and say so, not as "unlocked"."""
    import numpy as np

    p = Path(refs_dir) / f"{name}.npz"
    if not p.is_file():
        return None
    try:
        with np.load(p, allow_pickle=False) as z:
            if "orientation_lock_strength" not in z.files:
                return None
            return float(z["orientation_lock_strength"])
    except (OSError, ValueError):
        return None


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


def _density_leg(ms, scf, *, name: str, refs_dir: Optional[Path],
                 basis: str, grid_level: int, density_fit: bool,
                 orientation_lock_strength: float,
                 densities: Dict[str, Dict[str, float]],
                 density_json: Path) -> str:
    """One species' SCAN density error, appended to ``densities`` + persisted.

    Returns a short status fragment for the progress line. Species with no
    reference density (atoms, or anything outside the reference set) are
    recorded as ``None`` rather than dropped, so the consumer can tell "no
    reference" from "not yet computed"."""
    if refs_dir is None:
        raise ValueError("--with-density requires --refs-dir")
    rho_ref, weights = _reference_density(Path(refs_dir), name)
    if rho_ref is None:
        densities[name] = {"density_rmse_scan": None,
                           "density_eps_l1_scan": None}
        _atomic_write_json(density_json, densities)
        return "density: no reference"
    # The reference's own lock stamp is the authority. A MISMATCH means SCAN
    # locked a different degenerate component than the reference; a MISSING
    # stamp means the reference predates the key and its component is unknown.
    # Both make the density error uninterpretable, so both raise.
    stamped = _reference_lock(Path(refs_dir), name)
    if stamped is None:
        raise ValueError(
            f"reference for {name!r} carries no orientation_lock_strength "
            "stamp, so the component its density locked is unknown and a SCAN "
            "density error against it cannot be interpreted")
    if float(stamped) != float(orientation_lock_strength):
        raise ValueError(
            f"orientation lock mismatch for {name!r}: reference stamped "
            f"{stamped!r} but SCAN ran at {orientation_lock_strength!r}")
    ao = _reference_ao(Path(refs_dir), name, basis=basis,
                       grid_level=grid_level, density_fit=density_fit,
                       orientation_lock_strength=orientation_lock_strength)
    if ao is None:
        # Recorded, not raised: a missing intermediate is a cache-state fact
        # about the references, not a defect in this species' SCAN SCF, and it
        # must not sink the energy leg (which is already persisted).
        densities[name] = {"density_rmse_scan": None,
                           "density_eps_l1_scan": None}
        _atomic_write_json(density_json, densities)
        return "density: reference CCSD intermediate absent (no ao_grid)"
    rec = scan_density_record(scf["dm"], ao, weights, rho_ref,
                              n_electrons_expected=_n_electrons(ms))
    densities[name] = rec
    _atomic_write_json(density_json, densities)
    return (f"density RMSE={rec['density_rmse_scan']:.3e} "
            f"eps={rec['density_eps_l1_scan']:.3e}")


def run(pool: str, *, basis: str, grid_level: int, out_dir: Path,
        density_fit: bool = False, auxbasis: Optional[str] = None,
        force: bool = False, orientation_lock_strength: float = 0.0,
        with_density: bool = False, refs_dir: Optional[Path] = None) -> int:
    """Compute + cache SCAN total energies for every species in ``pool``, and --
    with ``with_density`` -- their density errors against the CCSD references.

    Returns the number of species that FAILED (0 == clean). Progress is printed
    per species with a running index/total + ETA (a multi-hour job must not look
    like a hang). The density leg reuses the dm the SCF already cached, so it
    adds no SCF; a species whose energy is cached but whose density is not is
    re-read from that cache rather than recomputed."""
    from xcquinox.alec.benchmark_refs import _mol_spec_to_atoms
    from xcquinox.alec.external_refs import SpeciesEntry, run_scf_with_cache

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_json = out_dir / _scan_cache_name(basis)
    density_json = out_dir / _scan_density_cache_name(basis)

    energies: Dict[str, float] = {}
    if cache_json.is_file() and not force:
        try:
            energies = {k: float(v) for k, v in json.loads(
                cache_json.read_text()).items()}
        except (json.JSONDecodeError, OSError):
            energies = {}
    densities: Dict[str, Dict[str, float]] = {}
    if with_density and density_json.is_file() and not force:
        try:
            densities = json.loads(density_json.read_text())
        except (json.JSONDecodeError, OSError):
            densities = {}

    mol_specs = _load_pool(pool, basis=basis, grid_level=grid_level)
    names = list(mol_specs)
    n = len(names)
    print(f"SCAN pool precompute: pool={pool} basis={basis} grid={grid_level} "
          f"density_fit={density_fit} lock={orientation_lock_strength}  "
          f"species={n}  cache={cache_json}")
    print(f"  {len(energies)}/{n} already cached (resume); "
          f"{'FORCING recompute' if force else 'skipping cached'}.")
    if with_density:
        print(f"  density leg ON -> {density_json}  refs={refs_dir}  "
              f"({len(densities)}/{n} already cached)")

    t0 = time.monotonic()
    n_fail = 0
    for k, name in enumerate(names, 1):
        ms = mol_specs[name]
        need_e = force or name not in energies
        need_d = with_density and (force or name not in densities)
        if not (need_e or need_d):
            print(f"  [{k}/{n}] {name}: SKIP (cached E={energies[name]:.8f} Ha)")
            continue
        spec = SpeciesEntry(name=ms.name, charge=int(ms.charge),
                            spin=int(ms.spin), source="benchmark")
        atoms = _mol_spec_to_atoms(ms)
        t1 = time.monotonic()
        try:
            # Cached after the first call, so the density leg re-reads the dm
            # rather than paying for a second SCF.
            scf = run_scf_with_cache(
                spec, atoms, cache_dir=out_dir, basis=basis,
                grid_level=grid_level, density_fit=density_fit,
                auxbasis=auxbasis, xc="scan",
                orientation_lock_strength=orientation_lock_strength)
            e_tot = scf.get("e_tot")
            if e_tot is None:
                raise ValueError("run_scf_with_cache returned e_tot=None")
            energies[name] = float(e_tot)
            _atomic_write_json(cache_json, energies)     # persist after each
            status = f"E={float(e_tot):.8f} Ha"
            if need_d:
                status += "  " + _density_leg(
                    ms, scf, name=name, refs_dir=refs_dir, basis=basis,
                    grid_level=grid_level, density_fit=density_fit,
                    orientation_lock_strength=orientation_lock_strength,
                    densities=densities, density_json=density_json)
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
    p.add_argument("--orientation-lock", type=float, default=0.0,
                   dest="orientation_lock_strength",
                   help="orientation-lock strength for the SCAN SCF; MUST match "
                        "the lock the references were generated with (the "
                        "dfs6311 references are locked at 3e-05). Default 0.0 "
                        "= unlocked, matching unlocked references.")
    p.add_argument("--with-density", action="store_true",
                   help="also score SCAN's density against the CCSD references "
                        "into scan_pool_density_<basis>.json (no extra SCF -- "
                        "reuses the density matrix the SCF already cached); "
                        "requires --refs-dir")
    p.add_argument("--refs-dir", default=None,
                   help="benchmark density-reference directory (<name>.npz from "
                        "xcquinox.alec.benchmark_refs) read by --with-density. "
                        "Read-only: SCAN artifacts go to --out-dir.")
    args = p.parse_args(argv)
    if args.with_density and not args.refs_dir:
        p.error("--with-density requires --refs-dir")

    n_fail = run(args.pool, basis=args.basis, grid_level=args.grid_level,
                 out_dir=Path(args.out_dir).expanduser(),
                 density_fit=args.density_fit, auxbasis=args.auxbasis,
                 force=args.force,
                 orientation_lock_strength=args.orientation_lock_strength,
                 with_density=args.with_density,
                 refs_dir=(Path(args.refs_dir).expanduser()
                           if args.refs_dir else None))
    return 1 if n_fail else 0


if __name__ == "__main__":
    # A scheduled job stage: its exit status is the scheduler's verdict on
    # the reference build, so it leaves through the shared hard exit (flush,
    # then os._exit) rather than through interpreter teardown, which aborted
    # on the cluster after a completed pretrain stage (job 2134455). Imported
    # here rather than in the module body, since the helper is needed only
    # when the module is RUN.
    from xcquinox.alec.cluster._exit import run_and_exit
    run_and_exit(main)
