#!/usr/bin/env python
"""Per-species OEP override verifier and snippet emitter.

Reads summary.json + per-species JSONL trial records produced by
scripts/oep_per_species_tune.py; applies the selection rule (spec sec. 7.1)
to pick a winner per species; emits a paste-ready Python snippet for
xcquinox/alec/external_refs.py:_PER_SPECIES_OEP_OVERRIDES.

Usage:
  python scripts/oep_per_species_emit_overrides.py \\
      --summary-path reports_local/oep_tune/2026-05-03/summary.json \\
      --out-dir      reports_local/oep_tune/2026-05-03 \\
      [--dry-run]

Exit codes:
  0 snippet emitted
  1 IO error / malformed inputs
  4 at least one species has no winner (failed target_floor)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _load_jsonl(path: Path) -> list[dict]:
    """Load one record per line."""
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def _round_2sigfig(value: float) -> float:
    """Round to 2 significant figures (spec sec. 7.2)."""
    import math
    if value <= 0 or not math.isfinite(value):
        return value
    magnitude = math.floor(math.log10(abs(value)))
    factor = 10 ** (magnitude - 1)
    return round(value / factor) * factor


def _is_stably_converged(history: list[float], plateau_window: int,
                          plateau_rtol: float) -> bool:
    """Spec sec. 7.1: final plateau_window iters within plateau_rtol relative
    range. Carve-out: history shorter than plateau_window AND last value
    < some conv_tol-equivalent → mark stably converged (handled by
    caller via terminated_by=='early_stop_conv_tol' check)."""
    if len(history) < plateau_window:
        return False
    tail = history[-plateau_window:]
    tail_max = max(tail)
    tail_min = min(tail)
    import statistics
    tail_med = statistics.median(tail)
    if abs(tail_med) < 1e-30:
        return False
    rel_range = (tail_max - tail_min) / abs(tail_med)
    return rel_range < plateau_rtol


def _passes_dm_bias_check(record: dict) -> bool:
    """Spec sec. 7.1 rule 3: only fires for level_shift > 0.5. Rejects if
    quad_aniso or r_squared differs by > 5% (normalized by target r^2).
    Dipole is null for atomic species, so dipole check is skipped there.
    """
    settings = record.get("settings", {})
    level_shift = settings.get("level_shift", 0.0)
    if level_shift <= 0.5:
        return True
    res = record["result"]
    target_r2 = res.get("target_dm_r_squared")
    if target_r2 is None or abs(target_r2) < 1e-30:
        return True   # cannot evaluate; let it through
    inner_q = res.get("inner_dm_quad_aniso") or 0.0
    target_q = res.get("target_dm_quad_aniso") or 0.0
    quad_diff_norm = abs(inner_q - target_q) / abs(target_r2)
    inner_r2 = res.get("inner_dm_r_squared") or 0.0
    r2_diff_norm = abs(inner_r2 - target_r2) / abs(target_r2)
    if quad_diff_norm > 0.05 or r2_diff_norm > 0.05:
        return False
    # Dipole check (molecular only):
    target_dip = res.get("target_dm_dipole")
    if target_dip is not None and abs(target_dip) > 1e-30:
        inner_dip = res.get("inner_dm_dipole") or 0.0
        if abs(inner_dip - target_dip) / abs(target_dip) > 0.05:
            return False
    return True


def _select_winner(records: list[dict], target_floor: float,
                    plateau_window: int = 20,
                    plateau_rtol: float = 0.02) -> dict | None:
    """Apply the spec sec. 7.1 selection rule. Returns winning record
    or None if no trial passes."""
    candidates = []
    for rec in records:
        res = rec["result"]
        if not res.get("converged_to_target_floor"):
            continue
        history = res.get("density_error_history", []) or []
        terminated_by = res.get("termination", "max_iter")
        # Stability via plateau-shared metric, with carve-out:
        if terminated_by == "early_stop_conv_tol" \
                and len(history) < plateau_window:
            stably = True
        else:
            stably = _is_stably_converged(history, plateau_window, plateau_rtol)
        if not stably:
            continue
        if not _passes_dm_bias_check(rec):
            continue
        candidates.append(rec)
    if not candidates:
        return None
    candidates.sort(key=lambda r: (
        r["result"]["density_error_min"],
        r["result"]["wall_clock_s"],
    ))
    return candidates[0]


def _emit_snippet(winner: dict) -> str:
    """Build the Python source snippet for one species' override entry."""
    spec = winner["species"]
    settings = winner["settings"]
    res = winner["result"]
    floor = res["density_error_min"]
    conv_tol = _round_2sigfig(1.7 * floor)
    # Build the tier dict:
    tier_keys = sorted(set(settings) - {"conv_tol", "target_floor"})
    tier_lines = []
    for k in tier_keys:
        v = settings[k]
        if isinstance(v, str):
            tier_lines.append(f'         "{k}": "{v}",')
        elif isinstance(v, float):
            tier_lines.append(f'         "{k}": {v:.4g},')
        else:
            tier_lines.append(f'         "{k}": {v!r},')
    tier_lines.append(f'         "conv_tol": {conv_tol:.2g},  # 1.7 * density_error_min')
    name = spec["name"]; charge = spec["charge"]; spin = spec["spin"]
    return (
        f'    # {name} winner: density_error_min={floor:.2e}, '
        f'n_iter={res["n_iter"]}, wall={res["wall_clock_s"]:.1f}s\n'
        f'    ("{name}", {charge}, {spin}): (\n'
        f'        {{\n'
        + "\n".join(tier_lines) + "\n"
        f'        }},\n'
        f'    ),\n'
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify harness JSONL and emit override snippet."
    )
    parser.add_argument("--summary-path", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    summary_path = args.summary_path
    if not summary_path.is_file():
        print(f"--summary-path {summary_path}: no such file", file=sys.stderr)
        return 1
    summary = json.loads(summary_path.read_text())
    out_dir = args.out_dir or summary_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    snippets = [
        "# AUTO-GENERATED snippet for xcquinox/alec/external_refs.py:_PER_SPECIES_OEP_OVERRIDES.\n"
        "# Source: %s\n" % summary_path
        + "# Citations: [oep-tdl-1..6] in reports_local/latex/references.bib —\n"
        "# AUTHOR-RECALLED, UNVERIFIED. Verify each via WebFetch + pdftotext\n"
        "# before paper write-up. REVIEW BEFORE PASTING.\n"
        "#\n"
    ]
    failed = []
    for name in summary.get("best_per_species", {}):
        jsonl_path = summary_path.parent / f"{name}.jsonl"
        if not jsonl_path.is_file():
            failed.append(name)
            continue
        records = _load_jsonl(jsonl_path)
        target_floor = float(records[0]["settings"].get(
            "target_floor", records[0]["settings"].get("conv_tol", 1e-3)
        ))
        winner = _select_winner(records, target_floor)
        if winner is None:
            failed.append(name)
            continue
        snippets.append(_emit_snippet(winner))
        snippets.append("# References: [oep-tdl-1..6] AUTHOR-RECALLED, UNVERIFIED.\n")

    snippet_text = "".join(snippets)
    (out_dir / "override_snippet.py").write_text(snippet_text)
    print(snippet_text)
    if failed:
        print(f"\nfailed_target_floor: {failed}", file=sys.stderr)
        return 4
    return 0


if __name__ == "__main__":
    sys.exit(main())
