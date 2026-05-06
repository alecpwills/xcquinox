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
    """Real harness execution path — populated in Task 4 + 5."""
    raise NotImplementedError(
        "Harness execution not yet implemented (Plan 3 Task 4 + 5)."
    )


if __name__ == "__main__":
    sys.exit(main())
