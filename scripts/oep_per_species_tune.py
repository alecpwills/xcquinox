#!/usr/bin/env python
"""Per-species OEP override harness.

Sweeps the literature-informed YAML knob grid for failed-species OEP
overrides (spec sec. 4 / sec. 6.1). Writes per-species JSONL trial
records to <out-dir>/<species>.jsonl plus a top-level summary.json.

PySCF imports are deferred to inside the trial loop so --dry-run
exits sub-second.

Usage:
  python scripts/oep_per_species_tune.py \\
      --species Be "C+" F2 F2O HF HS N2O O3 \\
      --grid scripts/oep_tune_grids.yaml \\
      --cache-dir notebooks/checkpoints_step7/external_refs \\
      --out-dir reports_local/oep_tune/2026-05-03 \\
      --wall-cap-min 5,8,15,30,5,8,15,15

Exit codes:
  0  every species hit its target_floor
  1  species computation crashed (full traceback printed)
  4  at least one species exhausted its grid without hitting target_floor
"""
from __future__ import annotations

import argparse
import json
import sys
from itertools import product
from pathlib import Path


def _parse_species_arg(arg_list: list[str]) -> list[tuple[str, int | None, int | None]]:
    """Parse --species args. Each is either a bare name (looked up in
    the YAML's charge/spin) or a "name,charge,spin" triple.
    Returns list of (name, charge_override, spin_override).
    """
    out = []
    for a in arg_list:
        if "," in a:
            parts = a.split(",")
            if len(parts) != 3:
                raise SystemExit(
                    f"--species value {a!r} must be 'name' "
                    f"or 'name,charge,spin'"
                )
            name, charge, spin = parts
            out.append((name.strip(), int(charge), int(spin)))
        else:
            out.append((a.strip(), None, None))
    return out


def _parse_wall_cap_min(arg: str, n_species: int) -> list[int]:
    """Parse comma-separated wall caps. Single value broadcasts; N
    values must match n_species exactly."""
    caps = [int(x.strip()) for x in arg.split(",")]
    if len(caps) == 1:
        return caps * n_species
    if len(caps) != n_species:
        raise SystemExit(
            f"--wall-cap-min has {len(caps)} values but --species "
            f"has {n_species}; must be 1 (broadcast) or {n_species}"
        )
    return caps


def _load_yaml_grid(path: Path) -> dict:
    """Load the YAML knob grid; validate top-level keys."""
    import yaml
    if not path.is_file():
        raise SystemExit(f"--grid {path}: no such file")
    with open(path) as f:
        grid = yaml.safe_load(f)
    if not isinstance(grid, dict):
        raise SystemExit(f"--grid {path}: top-level must be a mapping")
    return grid


def _validate_yaml_species_block(name: str, block: dict,
                                 allowlist: frozenset[str]) -> None:
    """Validate one species' YAML block."""
    required = {"charge", "spin", "target_floor", "sweep"}
    missing = required - set(block)
    if missing:
        raise SystemExit(
            f"YAML species {name!r}: missing required keys {sorted(missing)}"
        )
    if not isinstance(block["sweep"], dict) or not block["sweep"]:
        raise SystemExit(
            f"YAML species {name!r}: 'sweep' must be a non-empty mapping"
        )
    unknown = set(block["sweep"]) - allowlist
    if unknown:
        raise SystemExit(
            f"YAML species {name!r}: sweep contains unknown knobs "
            f"{sorted(unknown)}; allowed: {sorted(allowlist)}"
        )


def _enumerate_trials(species_name: str, block: dict) -> list[dict]:
    """Cartesian-product the sweep dict; apply the
    aux_basis -> regularization coupling-constraint filter (spec
    sec. 4 Be / F2 coupling rule). Return a list of trial-settings
    dicts."""
    sweep = block["sweep"]
    knobs = sorted(sweep.keys())
    values = [sweep[k] for k in knobs]
    trials = []
    for combo in product(*values):
        trial = dict(zip(knobs, combo))
        # Coupling rule: tzvp/qzvp aux requires reg >= 1e-3.
        aux = trial.get("aux_basis", "")
        reg = trial.get("regularization")
        needs_strong_reg = (
            "tzvp-jkfit" in aux or "qzvp-jkfit" in aux
        )
        if needs_strong_reg and reg is not None and reg < 1e-3:
            continue   # silently filter; warning printed by caller
        trials.append(trial)
    return trials


class _Heartbeat:
    """Background heartbeat thread mirroring
    scripts/smoke_preflight_uks_oep.py:73-110. Daemon thread prints a
    one-line status every `interval_sec` (default 15s) showing current
    species + trial_idx + elapsed wall-clock + RSS (Linux only).
    Spec §6.6 + user MEMORY directive on progress reporting."""

    def __init__(self, interval_sec: float = 15.0) -> None:
        import threading
        self.interval_sec = interval_sec
        self.stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._stage = "init"
        self._t0 = None

    def set_stage(self, stage: str) -> None:
        self._stage = stage

    def start(self) -> None:
        import threading, time
        self._t0 = time.time()
        self.stop_event.clear()
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="harness-heartbeat",
        )
        self._thread.start()

    def stop(self, timeout: float = 1.0) -> None:
        self.stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None

    def _run(self) -> None:
        import time, sys
        while not self.stop_event.wait(self.interval_sec):
            elapsed = time.time() - (self._t0 or time.time())
            try:
                import resource
                rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
            except (ImportError, AttributeError):
                rss_mb = -1.0
            print(
                f"[heartbeat] stage={self._stage}  "
                f"elapsed={elapsed:7.1f}s  rss={rss_mb:7.1f}MB",
                file=sys.stderr, flush=True,
            )


class _HarnessWallCap(Exception):
    """Raised by the SIGALRM handler when a trial exceeds its wall cap.
    Caught in the per-trial loop; partial trial record is finalized
    from the harness-owned history accumulator (spec sec. 6.1 Pass-8).
    """


def _wall_cap_handler(signum, frame):
    raise _HarnessWallCap()


def _install_wall_cap_handler():
    """Install SIGALRM handler ONCE at harness startup. Idempotent."""
    import signal
    try:
        signal.signal(signal.SIGALRM, _wall_cap_handler)
    except (ValueError, AttributeError):
        # Windows or test environment without SIGALRM
        pass


def _append_jsonl(path, record: dict) -> None:
    """Append a JSONL record to `path`. Spec sec. 6.1 (Pass 5):
    open(path, 'a') + write + fsync. Line-atomic on POSIX since
    typical record size << PIPE_BUF=4096."""
    import os
    import json
    line = json.dumps(record) + "\n"
    with open(path, "a") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def _compute_dm_observables(mol, dm, *, is_atomic: bool) -> dict:
    """Compute <r^2>, <3z^2 - r^2> (quadrupole anisotropy), and dipole
    magnitude on a DM. Per spec sec. 6.1 / sec. 7.1 Pass-7 fix:
    quadrupole anisotropy is the load-bearing observable for atomic
    Cartesian-bias detection; <r^2> is a coarser radial-extent
    diagnostic; dipole is null for atomic species (zero by parity)
    and computed for molecular species.

    Uses mol.intor('int1e_rr') -> shape (9, n_ao, n_ao) with diagonal
    indices 0=xx, 4=yy, 8=zz. Verified Pass 7/8 against
    pyscf/scf/hf.py::traceless_quadrupole_tensor.
    """
    import numpy as np
    rr = mol.intor("int1e_rr")           # (9, n_ao, n_ao)
    dm_total = dm.sum(axis=0) if dm.ndim == 3 else dm
    def _expect(op):
        return float(np.einsum("ij,ij->", op, dm_total))
    r2 = _expect(rr[0]) + _expect(rr[4]) + _expect(rr[8])
    quad_aniso = 2 * _expect(rr[8]) - _expect(rr[0]) - _expect(rr[4])
    if is_atomic:
        dipole = None
    else:
        r_op = mol.intor("int1e_r")      # (3, n_ao, n_ao)
        dip_xyz = [_expect(r_op[k]) for k in range(3)]
        dipole = float(np.linalg.norm(dip_xyz))
    return {"r_squared": r2,
            "quad_aniso": quad_aniso,
            "dipole": dipole}


def _is_stably_converged_inline(history: list[float], *,
                                  plateau_window: int,
                                  plateau_rtol: float,
                                  terminated_by: str) -> bool:
    """Spec §7.1 + carve-out: final plateau_window iters within
    plateau_rtol relative range. Carve-out: terminated_by ==
    'early_stop_conv_tol' AND len(history) < plateau_window → stable
    (the early-stop sentinel certifies cleanly hitting the goal).
    Plan-3-review fix: harness now populates `converged_stably` rather
    than always-False (was a JSONL schema contract drift)."""
    import statistics
    if not history:
        return False
    if (terminated_by == "early_stop_conv_tol"
            and len(history) < plateau_window):
        return True
    if len(history) < plateau_window:
        return False
    tail = history[-plateau_window:]
    tail_med = statistics.median(tail)
    if abs(tail_med) < 1e-30:
        return False
    rel_range = (max(tail) - min(tail)) / abs(tail_med)
    return rel_range < plateau_rtol


def _is_atomic_species(name: str) -> bool:
    """A species is 'atomic' when its name is a single chemical symbol
    (or symbol+'+' for cations). Mirrors the post-Pass-1 fix in
    xcquinox/alec/external_refs.py:resolve_geometry."""
    from ase.data import chemical_symbols
    sym = name.rstrip("+")
    return sym in chemical_symbols


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Per-species OEP override harness (spec sec. 6.1)."
    )
    parser.add_argument("--species", nargs="*", default=None,
                        help="Species to sweep; default: all in --grid YAML.")
    parser.add_argument("--grid", type=Path,
                        default=Path("scripts/oep_tune_grids.yaml"))
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--wall-cap-min", type=str, default="15",
                        help="Comma-separated minutes; one or len(species).")
    parser.add_argument("--max-trials-per-species", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate args + YAML; print plan; exit.")
    args = parser.parse_args()

    grid = _load_yaml_grid(args.grid)
    # Allowlist mirrors xcquinox/alec/external_refs.py:_OVERRIDE_TIER_KNOB_ALLOWLIST
    # but is duplicated here so --dry-run doesn't import external_refs/PySCF.
    allowlist = frozenset({
        "aux_basis", "regularization", "max_iter", "conv_tol",
        "grid_level", "level_shift", "inner_damp", "inner_diis_start_cycle",
    })
    for name, block in grid.items():
        _validate_yaml_species_block(name, block, allowlist)

    if args.species is None:
        species_arg = list(grid.keys())
    else:
        species_arg = args.species
    species_parsed = _parse_species_arg(species_arg)
    wall_caps_min = _parse_wall_cap_min(args.wall_cap_min, len(species_parsed))

    print("=" * 60)
    print("OEP per-species harness — trial-enumeration plan")
    print("=" * 60)
    total_trials = 0
    for (name, _, _), cap_min in zip(species_parsed, wall_caps_min):
        if name not in grid:
            print(f"  {name}: NOT in YAML grid — skipping")
            continue
        trials = _enumerate_trials(name, grid[name])
        n = min(len(trials), args.max_trials_per_species)
        worst_min = n * cap_min
        total_trials += n
        print(f"  {name:6s}  trials={n:>3d}  cap={cap_min} min  "
              f"worst-case={worst_min} min")
    print(f"  TOTAL trials (worst-case): {total_trials}")
    print("=" * 60)

    if args.dry_run:
        print("--dry-run: exiting before any OEP execution.")
        return 0

    # Defer heavy imports to inside main loop (spec sec. 6.1 dry-run
    # discipline; --dry-run must not drag PySCF in).
    return _run_harness(args, grid, species_parsed, wall_caps_min)


def _run_harness(args, grid, species_parsed, wall_caps_min) -> int:
    """Per-species, per-trial sweep loop. Spec sec. 6.1 / 6.2."""
    import datetime
    import json
    import resource
    import signal
    import time
    import traceback
    # Heavy imports deferred from --dry-run path:
    from xcquinox.alec.config import MoleculeSpec
    from xcquinox.alec.external_refs import (
        SpeciesEntry, build_species_union, resolve_geometry,
        run_scf_with_cache, run_ccsd_with_cache,
        _migrate_intermediates_to_grid_suffixed,
    )
    from xcquinox.alec.oep import run_oep_inversion

    # Migrate cache_dir defensively (idempotent if precompute_all has run):
    _migrate_intermediates_to_grid_suffixed(args.cache_dir)
    # Install SIGALRM handler ONCE at harness startup:
    _install_wall_cap_handler()
    heartbeat = _Heartbeat(interval_sec=15.0)
    heartbeat.start()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    union = {(s.name, s.charge, s.spin): s for s in build_species_union()}
    summary = {
        "started_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "ended_at_utc": None,
        "best_per_species": {},
        "n_trials_run": {},
        "short_circuited": [],
        "failed_target_floor": [],
    }
    overall_exit = 0

    for (name, charge_arg, spin_arg), cap_min in zip(species_parsed, wall_caps_min):
        if name not in grid:
            print(f"[skip] {name}: not in YAML grid", file=sys.stderr)
            continue
        block = grid[name]
        charge = charge_arg if charge_arg is not None else int(block["charge"])
        spin = spin_arg if spin_arg is not None else int(block["spin"])
        target_floor = float(block["target_floor"])
        trials = _enumerate_trials(name, block)[:args.max_trials_per_species]
        species_jsonl = args.out_dir / f"{name}.jsonl"
        heartbeat.set_stage(f"{name} setup")

        # Build SpeciesEntry + atoms ONCE per species:
        key = (name, charge, spin)
        if key in union:
            spec_entry = union[key]
        else:
            spec_entry = SpeciesEntry(name=name, charge=charge, spin=spin,
                                      source="harness")
        atoms = resolve_geometry(spec_entry)
        scf = run_scf_with_cache(spec_entry, atoms, cache_dir=args.cache_dir,
                                  basis="def2-svp", grid_level=1)
        cc = run_ccsd_with_cache(spec_entry, atoms, scf_payload=scf,
                                  cache_dir=args.cache_dir,
                                  basis="def2-svp", grid_level=1)
        is_atomic = _is_atomic_species(name)

        best_density_error_min = float("inf")
        best_record = None
        n_run = 0

        for trial_idx, trial_settings in enumerate(trials):
            n_run += 1
            heartbeat.set_stage(f"{name} trial={trial_idx}/{len(trials)}")
            density_error_history: list[float] = []
            def _trial_progress_cb(it, density_error_l2):
                density_error_history.append(float(density_error_l2))

            tier_grid_level = trial_settings.get("grid_level", 1)
            tier_mol_spec = MoleculeSpec.from_dict(
                name=name, atom="; ".join(
                    f"{s} {atoms.get_positions()[i,0]:.6f} "
                    f"{atoms.get_positions()[i,1]:.6f} "
                    f"{atoms.get_positions()[i,2]:.6f}"
                    for i, s in enumerate(atoms.get_chemical_symbols())
                ),
                basis="def2-svp", charge=charge, spin=spin,
                atom_composition={s: atoms.get_chemical_symbols().count(s)
                                  for s in set(atoms.get_chemical_symbols())},
                grid_level=tier_grid_level,
            )

            t0 = time.time()
            cap_sec = cap_min * 60
            signal.alarm(cap_sec)
            wall_capped = False
            exception_msg = None
            oep_result = None
            try:
                oep_result = run_oep_inversion(
                    tier_mol_spec, cc["dm_ao"],
                    aux_basis=trial_settings.get("aux_basis", "def2-svp-jkfit"),
                    regularization=trial_settings.get("regularization", 1e-4),
                    max_iter=trial_settings.get("max_iter", 500),
                    conv_tol=target_floor,
                    level_shift=trial_settings.get("level_shift",
                                                    0.5 if spin > 0 else 0.0),
                    inner_damp=trial_settings.get("inner_damp", 0.1),
                    inner_diis_start_cycle=trial_settings.get(
                        "inner_diis_start_cycle", 5),
                    progress_callback=_trial_progress_cb,
                )
            except _HarnessWallCap:
                wall_capped = True
            except Exception:
                exception_msg = traceback.format_exc()
            finally:
                signal.alarm(0)
            wall_clock_s = time.time() - t0

            # Compute observables only when oep_result is available:
            obs = {"r_squared": None, "quad_aniso": None, "dipole": None}
            target_obs = {"r_squared": None, "quad_aniso": None, "dipole": None}
            density_error_min = (
                float(min(density_error_history))
                if density_error_history else None
            )
            density_error_final = (
                float(density_error_history[-1])
                if density_error_history else None
            )
            converged = bool(oep_result.converged) if oep_result else False
            n_iter = len(density_error_history)
            if oep_result is not None and oep_result.dm_final is not None:
                from pyscf import gto
                _coords = atoms.get_positions()
                _syms = atoms.get_chemical_symbols()
                _atom_lines = [(s, tuple(_coords[i])) for i, s in enumerate(_syms)]
                mol = gto.M(atom=_atom_lines, basis="def2-svp",
                            charge=charge, spin=spin, verbose=0)
                obs = _compute_dm_observables(mol, oep_result.dm_final,
                                              is_atomic=is_atomic)
                target_obs = _compute_dm_observables(mol, cc["dm_ao"],
                                                    is_atomic=is_atomic)

            if exception_msg is not None:
                termination = "exception"
            elif wall_capped:
                termination = "wall_capped"
            elif oep_result is not None:
                terminated_by = getattr(oep_result, "terminated_by", "max_iter")
                termination = ("early_stop_conv_tol"
                               if terminated_by == "conv_tol"
                               else terminated_by)
            else:
                termination = "exception"

            record = {
                "trial_idx": trial_idx,
                "species": {"name": name, "charge": charge, "spin": spin},
                "settings": {
                    **trial_settings,
                    "conv_tol": target_floor,
                    "target_floor": target_floor,
                },
                "result": {
                    "density_error_history": density_error_history,
                    "F_val_history": [],   # v1: empty for wall-capped trials
                    "density_error_min": density_error_min,
                    "density_error_final": density_error_final,
                    "n_iter": n_iter,
                    "converged_stably": _is_stably_converged_inline(
                        density_error_history,
                        plateau_window=20,
                        plateau_rtol=0.02,
                        terminated_by=termination,
                    ),
                    "converged_to_target_floor": (
                        density_error_min is not None
                        and density_error_min <= target_floor
                    ),
                    "wall_clock_s": wall_clock_s,
                    "wall_capped": wall_capped,
                    "termination": termination,
                    "plateau_density_error": (
                        oep_result.density_error
                        if oep_result is not None
                        and getattr(oep_result, "terminated_by", None) == "plateau"
                        else None
                    ),
                    "plateau_window_iters": 20,
                    "inner_dm_r_squared": obs["r_squared"],
                    "target_dm_r_squared": target_obs["r_squared"],
                    "inner_dm_quad_aniso": obs["quad_aniso"],
                    "target_dm_quad_aniso": target_obs["quad_aniso"],
                    "inner_dm_dipole": obs["dipole"],
                    "target_dm_dipole": target_obs["dipole"],
                    "rss_mb_peak": (
                        float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0)
                        if hasattr(__import__("resource"), "getrusage") else None
                    ),
                    "error_msg": exception_msg,
                },
            }
            _append_jsonl(species_jsonl, record)

            if (record["result"]["converged_to_target_floor"]
                    and density_error_min is not None
                    and density_error_min < best_density_error_min):
                best_density_error_min = density_error_min
                best_record = record
                summary["short_circuited"].append(name)
                break    # short-circuit: hit target_floor

        summary["n_trials_run"][name] = n_run
        if best_record is not None:
            summary["best_per_species"][name] = {
                "trial_idx": best_record["trial_idx"],
                "settings": best_record["settings"],
                "density_error_min": best_record["result"]["density_error_min"],
                "wall_clock_s": best_record["result"]["wall_clock_s"],
            }
        else:
            summary["failed_target_floor"].append(name)
            overall_exit = 4

    heartbeat.stop()
    summary["ended_at_utc"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return overall_exit


if __name__ == "__main__":
    sys.exit(main())
