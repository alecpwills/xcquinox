"""Parity check: cluster-written held-out rows vs the local reconstruction.

For each completed spec x channel of a pulled run, compares the
cluster-written ``per_reaction.json`` against the verbatim-holdout slice the
figure layer reconstructs from the same spec's per-species energies. Four
verdicts:

- ``parity``       -- same reaction set, values agreeing within tolerance
                      (a post-deployment or refinalized eval);
- ``stale-rule``   -- different reaction set (an eval written under the
                      retired species-strict rule; refinalize on the
                      cluster, or rely on the local reconstruction);
- ``value-mismatch`` -- same set but diverging values, including a finite
                      value on one side only (would indicate a real defect;
                      none expected);
- ``no-cluster-file`` -- the cluster never wrote rows for this spec/channel
                      (only the reconstruction exists).

Run after each pull; the closing state is all ``parity``.

Usage::

    python notebooks/analysis/verify_holdout_parity.py <run_dir> [...] \
        [--channels eval_holdout eval_holdout_val_best] [--tol 1e-9]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_HERE = Path(__file__).resolve().parent


def _load_fig_module():
    spec = importlib.util.spec_from_file_location(
        "make_ablation_arch_figure", _HERE / "make_ablation_arch_figure.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def compare_spec(cluster_rows: Optional[List[Dict[str, Any]]],
                 recon_rows: List[Dict[str, Any]],
                 tol: float = 1e-9) -> Dict[str, Any]:
    """Verdict dict for one spec x channel (pure; testable)."""
    if cluster_rows is None:
        return {"verdict": "no-cluster-file", "n_cluster": 0,
                "n_recon": len(recon_rows)}
    c_names = sorted(str(r.get("name")) for r in cluster_rows)
    r_names = sorted(str(r.get("name")) for r in recon_rows)
    out = {"n_cluster": len(c_names), "n_recon": len(r_names)}
    if c_names != r_names:
        only_c = sorted(set(c_names) - set(r_names))
        only_r = sorted(set(r_names) - set(c_names))
        out.update({"verdict": "stale-rule",
                    "only_cluster": only_c[:6], "only_recon": only_r[:6]})
        return out
    c_by = {}
    for r in cluster_rows:
        c_by.setdefault(str(r.get("name")), []).append(r)
    worst = 0.0
    for rr in recon_rows:
        cands = c_by.get(str(rr.get("name"))) or []
        best = None
        for cr in cands:
            ds = []
            for k in ("de_nn_kcalmol", "de_pbe_kcalmol",
                      "abs_error_nn_kcalmol", "abs_error_pbe_kcalmol"):
                a, b = cr.get(k), rr.get(k)
                fa = isinstance(a, (int, float)) and math.isfinite(a)
                fb = isinstance(b, (int, float)) and math.isfinite(b)
                if fa and fb:
                    ds.append(abs(float(a) - float(b)))
                elif fa != fb:
                    # finite on exactly one side is a divergence, not parity
                    ds.append(float("inf"))
            d = max(ds) if ds else 0.0
            if best is None or d < best:
                best = d
        worst = max(worst, best or 0.0)
    out["max_abs_delta"] = worst
    out["verdict"] = "parity" if worst <= tol else "value-mismatch"
    return out


def verify_run(run_dir: Path, *, channels: Sequence[str],
               tol: float = 1e-9, _fig=None) -> List[Dict[str, Any]]:
    fig = _fig if _fig is not None else _load_fig_module()
    run_dir = Path(run_dir)
    reports: List[Dict[str, Any]] = []
    for ch in channels:
        rows = fig.collect_holdout_reaction_rows(run_dir, eval_subdir=ch)
        by_idx: Dict[int, List[Dict[str, Any]]] = {}
        for r in rows:
            by_idx.setdefault(r["idx"], []).append(r)
        for idx in sorted(by_idx):
            sd = run_dir / "checkpoints" / f"spec_{idx:04d}"
            p = sd / ch / "per_reaction.json"
            cluster = None
            if p.is_file():
                try:
                    with p.open() as f:
                        cluster = json.load(f)
                except (json.JSONDecodeError, OSError):
                    cluster = None
            rep = compare_spec(cluster, by_idx[idx], tol=tol)
            rep.update({"spec": f"spec_{idx:04d}", "channel": ch})
            reports.append(rep)
            extra = ""
            if rep["verdict"] == "stale-rule":
                extra = (f" (cluster-only: {rep['only_cluster']}, "
                         f"recon-only: {rep['only_recon']})")
            elif rep["verdict"] == "parity":
                extra = f" (max |delta| {rep['max_abs_delta']:.2e})"
            print(f"[parity] {run_dir.name}/spec_{idx:04d}/{ch}: "
                  f"{rep['verdict']} [{rep['n_cluster']} cluster vs "
                  f"{rep['n_recon']} recon]{extra}")
    n_stale = sum(1 for r in reports if r["verdict"] == "stale-rule")
    n_par = sum(1 for r in reports if r["verdict"] == "parity")
    n_bad = sum(1 for r in reports if r["verdict"] == "value-mismatch")
    n_nof = sum(1 for r in reports if r["verdict"] == "no-cluster-file")
    print(f"[parity] {run_dir}: {n_par} parity, {n_stale} stale-rule, "
          f"{n_bad} value-mismatch, {n_nof} no-cluster-file", flush=True)
    return reports


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("--channels", nargs="+",
                   default=["eval_holdout", "eval_holdout_val_best"])
    p.add_argument("--tol", type=float, default=1e-9)
    args = p.parse_args(argv)
    bad = 0
    for rd in args.run_dirs:
        reports = verify_run(Path(rd), channels=args.channels, tol=args.tol)
        bad += sum(1 for r in reports if r["verdict"] == "value-mismatch")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
