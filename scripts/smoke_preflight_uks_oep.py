#!/usr/bin/env python
"""Smoke verification for xcquinox.alec.external_refs.preflight_uks_oep.

Runs the full SCF -> CCSD -> 2-tier OEP cascade for HO (2-Pi doublet) and
HN (3-Sigma- triplet) with verbose, real-time progress reporting:

  - Banner per stage with start time
  - Background heartbeat every 15 s showing current stage + elapsed + RSS
  - Per-stage wall-clock timings + cache file sizes
  - npz key + shape + dtype dump after each species
  - Explicit shape-contract assertions
  - Final summary table
  - Cache-hit re-run of preflight_uks_oep itself to prove the public
    function path works end-to-end (sub-minute on cache hits)

Usage:
    python scripts/smoke_preflight_uks_oep.py --cache-dir /tmp/smoke
    python scripts/smoke_preflight_uks_oep.py --cache-dir /tmp/smoke --dry-run

Exit codes:
    0  success
    1  species computation failure (full traceback printed)
    2  shape contract violation
    3  cache-hit re-run divergence
"""
from __future__ import annotations

import argparse
import platform
import resource
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np


HEARTBEAT_INTERVAL_SEC = 15.0
SHAPE_CONTRACT_RC = 2
CACHE_REPLAY_RC = 3


class ShapeContractError(Exception):
    """Raised when an npz output violates the UKS shape contract.

    Caught by main() so heartbeat.stop() and _print_summary() can run
    before exiting with SHAPE_CONTRACT_RC.
    """


def rss_mb() -> float:
    """Resident set size in MB (Linux ru_maxrss is in KB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def hms(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}h{m:02d}m{s:02d}s"


def banner(text: str) -> None:
    bar = "=" * 78
    print(f"\n{bar}\n  {text}\n{bar}", flush=True)


def subbanner(text: str) -> None:
    print(f"\n  --- {text} ---", flush=True)


class Heartbeat:
    """Daemon thread emitting a heartbeat every HEARTBEAT_INTERVAL_SEC.

    The current stage label is owned by the main thread; the heartbeat
    just reads it. Stop with .stop() before printing final summary lines
    so heartbeats do not interleave with the table.
    """

    def __init__(self, interval: float = HEARTBEAT_INTERVAL_SEC) -> None:
        self._interval = interval
        self._stage = "(idle)"
        self._stage_started = time.time()
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="smoke-heartbeat", daemon=True,
        )
        self._started = False

    def start_stage(self, stage: str) -> None:
        self._stage = stage
        self._stage_started = time.time()
        if not self._started:
            self._thread.start()
            self._started = True

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            elapsed = time.time() - self._stage_started
            print(
                f"  [heartbeat {hms(elapsed)}] {self._stage} "
                f"-- RSS={rss_mb():.0f} MB",
                flush=True,
            )

    def stop(self) -> None:
        self._stop.set()
        if self._started:
            self._thread.join(timeout=2 * self._interval)


def print_environment(args: argparse.Namespace) -> None:
    banner("Step-7 T6 smoke test -- preflight_uks_oep on HO + HN")
    print(f"  Date           : {time.strftime('%Y-%m-%d %H:%M:%S %Z')}", flush=True)
    print(f"  Host           : {platform.node()}", flush=True)
    print(f"  Platform       : {platform.platform()}", flush=True)
    print(f"  Python         : {sys.version.split()[0]}", flush=True)
    for libname in ("pyscf", "jax", "numpy"):
        try:
            mod = __import__(libname)
            print(f"  {libname:<14} : {getattr(mod, '__version__', '?')}",
                  flush=True)
        except Exception as e:
            print(f"  {libname:<14} : <import failed: {e}>", flush=True)
    print(f"  cache_dir      : {args.cache_dir.resolve()}", flush=True)
    print(f"  basis          : {args.basis}", flush=True)
    print(f"  grid_level     : {args.grid_level}", flush=True)
    print(
        "  Species        : HO (2-Pi doublet, 9e), HN (3-Sigma- triplet, 8e)",
        flush=True,
    )
    print(f"  Heartbeat int. : {HEARTBEAT_INTERVAL_SEC:.0f} s", flush=True)
    print(f"  RSS at start   : {rss_mb():.0f} MB", flush=True)


def dump_npz(npz_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Open npz with the safe-load contract, print every key + shape + dtype,
    return (vxc_ref, rho_ref_grid) for shape-contract assertions.
    """
    # Safe-load contract: verbatim from xcquinox/alec/external_refs.py:215
    # np.load(..., allow_pickle=False) -- pickle disabled to prevent
    # arbitrary code execution from untrusted npz files.
    with np.load(npz_path, allow_pickle=False) as z:
        keys = sorted(z.files)
        print(f"    npz keys: {keys}", flush=True)
        for k in keys:
            arr = z[k]
            try:
                if arr.ndim == 0:
                    print(
                        f"      {k:<22} shape={arr.shape!s:<22} "
                        f"dtype={arr.dtype} value={arr.item()!r}",
                        flush=True,
                    )
                else:
                    aabs = np.abs(arr) if np.issubdtype(arr.dtype, np.number) else None
                    summary = ""
                    if aabs is not None and arr.size:
                        summary = (
                            f" |min,max|={aabs.min():.3e},{aabs.max():.3e}"
                        )
                    print(
                        f"      {k:<22} shape={arr.shape!s:<22} "
                        f"dtype={arr.dtype}{summary}",
                        flush=True,
                    )
            except Exception as e:
                print(f"      {k:<22} <dump failed: {e}>", flush=True)
        vxc = np.asarray(z["vxc_ref"])
        rho = np.asarray(z["rho_ref_grid"])
    return vxc, rho


def check_shape_contract(name: str, vxc: np.ndarray, rho: np.ndarray) -> None:
    """Mirror the assertions inside preflight_uks_oep itself; fail with a
    distinct exit code so the user can tell shape violation apart from
    species computation failure."""
    if vxc.ndim != 3 or vxc.shape[0] != 2:
        raise ShapeContractError(
            f"{name}: vxc_ref.shape={vxc.shape} (expected (2, n_ao, n_ao))"
        )
    if rho.ndim != 1:
        raise ShapeContractError(
            f"{name}: rho_ref_grid.shape={rho.shape} "
            "(expected 1D spin-summed per data.py:296-299)"
        )
    print(
        f"    [OK] {name} shape contract: vxc_ref={vxc.shape}, "
        f"rho_ref_grid={rho.shape}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--cache-dir", required=True, type=Path,
                    help="Directory for npz outputs (will be created).")
    ap.add_argument("--basis", default="def2-svp",
                    help="AO basis (default: def2-svp).")
    ap.add_argument("--grid-level", type=int, default=1,
                    help="PySCF DFT grid level (default: 1).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Import only; do not execute SCF/CCSD/OEP.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    print_environment(args)

    if args.dry_run:
        banner("DRY-RUN: importing preflight_uks_oep and exiting")
        from xcquinox.alec.external_refs import preflight_uks_oep
        print(f"  Resolved symbol: {preflight_uks_oep!r}", flush=True)
        print("  (no SCF/CCSD/OEP executed; cache_dir untouched)", flush=True)
        return 0

    args.cache_dir.mkdir(parents=True, exist_ok=True)

    # Imports kept inside main() so --dry-run errors surface above without
    # the user seeing a benign "import succeeded" line they might mistake
    # for the smoke run starting.
    from xcquinox.alec.external_refs import (
        SpeciesEntry,
        preflight_uks_oep,
        resolve_geometry,
        run_scf_with_cache,
        run_ccsd_with_cache,
        run_oep_cascade,
    )

    smoke_specs = [
        SpeciesEntry("HO", 0, 1, "dfs_ae"),  # 2-Pi doublet, 9e
        SpeciesEntry("HN", 0, 2, "dfs_ae"),  # 3-Sigma- triplet, 8e
    ]

    heartbeat = Heartbeat()
    overall_start = time.time()
    timings: dict[str, dict[str, float]] = {}

    try:
        for idx, spec in enumerate(smoke_specs, start=1):
            banner(
                f"[{idx}/{len(smoke_specs)}] {spec.name}  "
                f"charge={spec.charge}  spin={spec.spin}  source={spec.source}"
            )
            species_start = time.time()
            stage_t: dict[str, float] = {}

            subbanner(f"{spec.name} | resolve_geometry")
            heartbeat.start_stage(f"{spec.name} resolve_geometry")
            t0 = time.time()
            atoms = resolve_geometry(spec)
            stage_t["resolve"] = time.time() - t0
            print(
                f"    n_atoms={len(atoms)}, "
                f"formula={atoms.get_chemical_formula()}, "
                f"elapsed={hms(stage_t['resolve'])}",
                flush=True,
            )

            subbanner(f"{spec.name} | SCF (PBE, basis={args.basis}, "
                      f"grid_level={args.grid_level})")
            heartbeat.start_stage(f"{spec.name} SCF")
            t0 = time.time()
            scf = run_scf_with_cache(
                spec, atoms, cache_dir=args.cache_dir,
                basis=args.basis, grid_level=args.grid_level,
            )
            stage_t["scf"] = time.time() - t0
            scf_path = args.cache_dir / "_intermediates" / f"{spec.name}_scf.npz"
            scf_size = scf_path.stat().st_size / 1e6 if scf_path.is_file() else float("nan")
            print(
                f"    spin_unrestricted={scf['spin_unrestricted']}, "
                f"n_ao={scf['n_ao']}, n_grid={scf['n_grid']}, "
                f"cache={scf_path.name} ({scf_size:.2f} MB), "
                f"elapsed={hms(stage_t['scf'])}",
                flush=True,
            )
            if not scf["spin_unrestricted"]:
                raise RuntimeError(
                    f"{spec.name} should be UKS (spin={spec.spin}) but "
                    "SCF dispatched RKS"
                )

            subbanner(f"{spec.name} | CCSD (spin-summed grid density)")
            heartbeat.start_stage(f"{spec.name} CCSD")
            t0 = time.time()
            cc = run_ccsd_with_cache(
                spec, atoms, scf_payload=scf, cache_dir=args.cache_dir,
                basis=args.basis, grid_level=args.grid_level,
            )
            stage_t["ccsd"] = time.time() - t0
            cc_path = args.cache_dir / "_intermediates" / f"{spec.name}_ccsd.npz"
            cc_size = cc_path.stat().st_size / 1e6 if cc_path.is_file() else float("nan")
            rho_grid = np.asarray(cc["rho_ref_grid"])
            print(
                f"    rho_ref_grid.shape={rho_grid.shape} (expect 1D), "
                f"sum={rho_grid.sum():.6f} (~ N_e), "
                f"cache={cc_path.name} ({cc_size:.2f} MB), "
                f"elapsed={hms(stage_t['ccsd'])}",
                flush=True,
            )

            subbanner(f"{spec.name} | OEP cascade (svp-jkfit -> tzvp-jkfit)")
            heartbeat.start_stage(
                f"{spec.name} OEP cascade (silent inside run_oep_inversion -- "
                "heartbeats prove progress)"
            )
            t0 = time.time()
            npz_path = run_oep_cascade(
                spec, atoms, ccsd_payload=cc, cache_dir=args.cache_dir,
                basis=args.basis, grid_level=args.grid_level,
            )
            stage_t["oep"] = time.time() - t0
            oep_size = Path(npz_path).stat().st_size / 1e6
            print(
                f"    OEP output: {Path(npz_path).name} "
                f"({oep_size:.2f} MB), elapsed={hms(stage_t['oep'])}",
                flush=True,
            )

            subbanner(f"{spec.name} | shape verification")
            heartbeat.start_stage(f"{spec.name} verify shapes")
            vxc, rho = dump_npz(Path(npz_path))
            check_shape_contract(spec.name, vxc, rho)

            stage_t["total"] = time.time() - species_start
            timings[spec.name] = stage_t
            print(
                f"\n  {spec.name} OK in {hms(stage_t['total'])}",
                flush=True,
            )

    except ShapeContractError as e:
        heartbeat.stop()
        print(f"\n  [SHAPE FAIL] {e}", flush=True)
        if timings:
            _print_summary(timings, time.time() - overall_start)
        return SHAPE_CONTRACT_RC
    except Exception:
        heartbeat.stop()
        print("\n!!! SMOKE RUN FAILED !!!", flush=True)
        traceback.print_exc()
        if timings:
            _print_summary(timings, time.time() - overall_start)
        return 1

    # Cache-hit re-run of the public function: should be sub-minute total.
    banner("Cache-hit re-run via preflight_uks_oep() public entry point")
    heartbeat.start_stage("preflight_uks_oep cache-hit re-run")
    t0 = time.time()
    try:
        preflight_uks_oep(
            cache_dir=args.cache_dir, basis=args.basis,
            grid_level=args.grid_level,
        )
    except Exception:
        heartbeat.stop()
        print("\n!!! CACHE-HIT RE-RUN FAILED !!!", flush=True)
        traceback.print_exc()
        _print_summary(timings, time.time() - overall_start)
        return CACHE_REPLAY_RC
    replay_elapsed = time.time() - t0
    print(
        f"  preflight_uks_oep cache-hit re-run: {hms(replay_elapsed)} "
        f"(expected sub-minute on cache hits)",
        flush=True,
    )

    heartbeat.stop()
    _print_summary(timings, time.time() - overall_start)
    print("\nALL CHECKS PASSED.", flush=True)
    return 0


def _print_summary(
    timings: dict[str, dict[str, float]],
    overall_seconds: float,
) -> None:
    banner("Summary")
    cols = ("species", "resolve", "scf", "ccsd", "oep", "total")
    widths = (8, 10, 10, 10, 10, 12)
    header = "  " + "".join(f"{c:<{w}}" for c, w in zip(cols, widths))
    print(header, flush=True)
    print("  " + "-" * sum(widths), flush=True)
    for name, t in timings.items():
        row = "  " + "".join(
            f"{val:<{w}}" for val, w in zip(
                (
                    name,
                    hms(t.get("resolve", 0.0)),
                    hms(t.get("scf", 0.0)),
                    hms(t.get("ccsd", 0.0)),
                    hms(t.get("oep", 0.0)),
                    hms(t.get("total", 0.0)),
                ),
                widths,
            )
        )
        print(row, flush=True)
    print(f"\n  Overall wall time: {hms(overall_seconds)}", flush=True)
    print(f"  Final RSS: {rss_mb():.0f} MB", flush=True)


if __name__ == "__main__":
    sys.exit(main())
