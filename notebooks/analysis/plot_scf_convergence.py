#!/usr/bin/env python
"""Per-molecule SCF convergence figures from the held-out re-eval.

After ``reeval_holdout_fixed.py`` (v2) runs, each spec's
``checkpoints/spec_<NNNN>/eval_holdout/per_molecule.json`` carries, per
molecule, the NN self-consistent-field energy at every cycle of the training
solver (``full_3`` -> 3 cycles) as ``scf_energy_step_<i>`` plus the residual
``scf_energy_residual_<i> = |E_i - E_final|``. This script visualizes that
convergence: for each spec, residual-vs-SCF-step with one (faint) line per
molecule and the median overlaid, so you can see at a glance how the NN
functional's SCF converges across the held-out set.

Usage:
    python notebooks/analysis/plot_scf_convergence.py \
        [--run-dir <run dir>] [--specs 0,1] [--outdir <dir>]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from xcquinox.alec.eval_holdout import assert_channel_not_sliced

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_DEFAULT_LOCAL_ROOT = Path.home() / "Documents/Research/xcquinox-results/runs"
_DEFAULT_CATEGORY = "ablation_notransform/polarized/runs"


# ---------------------------------------------------------------------------
# Ingest (pure, unit-tested)
# ---------------------------------------------------------------------------

def molecule_scf_trace(record: Dict[str, Any]) -> Dict[str, Any]:
    """Extract the ordered per-cycle (step, energy, residual) from one
    per_molecule record. Returns ``{molecule, steps, energies, residuals,
    cycles_run, converged}`` -- empty ``steps`` when no trace is present."""
    steps: List[int] = []
    energies: List[float] = []
    residuals: List[float] = []
    i = 0
    while f"scf_energy_step_{i}" in record:
        e = record.get(f"scf_energy_step_{i}")
        r = record.get(f"scf_energy_residual_{i}")
        if isinstance(e, (int, float)) and math.isfinite(e):
            steps.append(i)
            energies.append(float(e))
            residuals.append(float(r) if isinstance(r, (int, float)) else float("nan"))
        i += 1
    return {
        "molecule": record.get("molecule"),
        "steps": steps,
        "energies": energies,
        "residuals": residuals,
        "cycles_run": record.get("cycles_run"),
        "converged": record.get("scf_converged"),
    }


def collect_spec_scf_traces(run_dir: Path, spec_idx: int,
                            width: int = 4,
                            eval_subdir: str = "eval_holdout"
                            ) -> List[Dict[str, Any]]:
    """All per-molecule SCF traces for one spec (only molecules with a trace).

    ``eval_subdir`` selects the channel; ``eval_holdout_coldstart`` carries
    the 25-cycle cold-start trajectories this figure exists to display."""
    # --specs names a spec directly, bypassing discovery, so this reader
    # carries its own refusal: an SCF trajectory over the handful of species
    # named for a workflow test is not the pool's.
    spec_dir = run_dir / "checkpoints" / f"spec_{spec_idx:0{width}d}"
    assert_channel_not_sliced(spec_dir, eval_subdir)
    pm = spec_dir / eval_subdir / "per_molecule.json"
    if not pm.is_file():
        return []
    try:
        records = json.loads(pm.read_text())
    except (json.JSONDecodeError, OSError):
        return []
    out = []
    for rec in records:
        tr = molecule_scf_trace(rec)
        if tr["steps"]:
            out.append(tr)
    return out


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_spec_convergence(traces: List[Dict[str, Any]], out_path: Path,
                          *, title: str) -> Path:
    """Residual-vs-SCF-step, one faint line per molecule + median overlay."""
    fig, ax = plt.subplots(figsize=(8, 5.5))
    if not traces:
        ax.text(0.5, 0.5, "no SCF traces (one-shot eval or no data)",
                ha="center", va="center", transform=ax.transAxes)
    else:
        max_step = max(max(t["steps"]) for t in traces)
        floor = 1e-8  # plot residuals on a log axis; clamp exact zeros
        for t in traces:
            ys = [max(r, floor) if math.isfinite(r) else floor
                  for r in t["residuals"]]
            ax.plot(t["steps"], ys, color="#4f81bd", alpha=0.18, linewidth=0.8,
                    marker="o", ms=2.5)
        # Median residual per step across molecules.
        med = []
        for s in range(max_step + 1):
            vals = [max(t["residuals"][t["steps"].index(s)], floor)
                    for t in traces if s in t["steps"]
                    and math.isfinite(t["residuals"][t["steps"].index(s)])]
            med.append(np.median(vals) if vals else np.nan)
        ax.plot(range(max_step + 1), med, color="#c0504d", linewidth=2.2,
                marker="s", ms=5, label="median |E_i - E_final|")
        ax.set_yscale("log")
        ax.set_xticks(range(max_step + 1))
        n_conv = sum(1 for t in traces if t["converged"])
        ax.legend(loc="upper right", fontsize=8)
        ax.text(0.02, 0.02,
                f"{len(traces)} molecules - {n_conv} converged - "
                f"{max_step + 1} SCF cycles",
                transform=ax.transAxes, fontsize=7, color="#555555")
    ax.set_xlabel("SCF cycle index  i")
    ax.set_ylabel(r"residual  $|E_i - E_{\rm final}|$  (Hartree, log)")
    ax.set_title(title, fontsize=10)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def _resolve_run_dir(run_dir: Optional[str]) -> Path:
    if run_dir:
        return Path(run_dir).expanduser().resolve()
    cand = _DEFAULT_LOCAL_ROOT / _DEFAULT_CATEGORY
    runs = sorted(p for p in cand.glob("run_*") if p.is_dir()) if cand.is_dir() else []
    if runs:
        return runs[-1].resolve()
    raise SystemExit(f"No run dir under {cand}; pass --run-dir.")


def _discover_specs_with_traces(run_dir: Path, width: int = 4,
                                eval_subdir: str = "eval_holdout"
                                ) -> List[int]:
    ck = run_dir / "checkpoints"
    out: List[int] = []
    if not ck.is_dir():
        return out
    for sd in sorted(ck.glob("spec_*")):
        # Before the existence probe, not after: discovery keys on
        # per_molecule.json alone, so an interrupted sliced channel (marker
        # written, energies never landed) would drop out of the list
        # silently instead of being refused.
        assert_channel_not_sliced(sd, eval_subdir)
        if (sd / eval_subdir / "per_molecule.json").is_file():
            try:
                out.append(int(sd.name[len("spec_"):]))
            except ValueError:
                continue
    return out


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", default=None)
    p.add_argument("--specs", default=None,
                   help="comma-separated spec indices (default: all with data)")
    p.add_argument("--eval-subdir", default="eval_holdout",
                   help="channel to read traces from (eval_holdout / "
                        "eval_holdout_coldstart)")
    p.add_argument("--outdir", default=str(
        Path(__file__).resolve().parent / "figures_ablation_notransform"
        / "scf_convergence"))
    args = p.parse_args(argv)

    run_dir = _resolve_run_dir(args.run_dir)
    outdir = Path(args.outdir).expanduser().resolve()
    specs = ([int(t) for t in args.specs.split(",") if t.strip()]
             if args.specs else _discover_specs_with_traces(
                 run_dir, eval_subdir=args.eval_subdir))
    print(f"run_dir: {run_dir}  specs: {specs}")
    n = 0
    for idx in specs:
        traces = collect_spec_scf_traces(run_dir, idx,
                                         eval_subdir=args.eval_subdir)
        out = plot_spec_convergence(
            traces, outdir / f"scf_convergence_spec_{idx:04d}.png",
            title=f"NN SCF convergence -- spec {idx} - {run_dir.name}")
        print(f"  wrote {out}  ({len(traces)} molecule traces)")
        n += 1
    if not n:
        print("  no specs with per_molecule.json found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
