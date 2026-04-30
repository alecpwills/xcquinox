"""Step-5 sweep data loader.

Step 5's outputs are a forest of per-spec ``aggregate.json`` +
``per_molecule.csv`` files (rather than the single tidy ``eval_df.csv``
that step 6 produces). This loader walks the directory tree and
returns a tidy DataFrame with the same schema family used by
``comparison_lib.py``:

  columns: arch, loss, solver, molecule, value_name, value

Coverage:
  - eval/{arch}/{loss}/{solver}/per_molecule.csv          -- training mols
  - test_new/{mol}/{arch}/{loss}/{solver}/per_molecule.csv -- transfer mols
  - eval_baseline/{arch}/{kind}/per_molecule.csv          -- random+pretrained baselines
  - eval_balancing/{loss}/{balancer}/per_molecule.csv     -- balancing study

Step 5 uses the same per_molecule.csv schema as step 6 (`molecule,
E_total_nn, E_pbe, E_error_hartree, E_error_kcalmol, AE_nn,
density_rmse, density_l1, skipped, skip_reason, ref_density_method,
AE_ref_kcalmol, AE_error_hartree, AE_error_kcalmol`) so the same
``mae_of`` helper from comparison_lib applies.
"""
from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

STEP5_ARCHS = (
    "deep", "deep_attn", "deep_dm", "deep_dm_attn",
    "deep_cusp", "deep_cusp_attn", "deep_combined", "deep_combined_attn",
)
STEP5_LOSSES = ("A_atomization", "B_atomization_plus_dm", "C_atomization_plus_grid")
STEP5_SOLVERS = ("oneshot", "fixed_j_3", "full_3")


# Atomic species are saved with skip_reason="atomic_system" and have no
# AE_error fields. Drop them at the molecule level.
def _per_molecule_rows_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Convert one wide per_molecule.csv into the long form used by the loader."""
    if "molecule" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    # Drop atomic-system rows (their numeric fields are NaN by design).
    if "skipped" in df.columns:
        skipped = df["skipped"].fillna(False)
        # Coerce to bool without invoking pandas FutureWarning on object-dtype.
        skipped_bool = skipped.map(
            lambda v: bool(v) if isinstance(v, (bool, int, float)) else
                      (str(v).lower() == "true")
        )
        df = df[~skipped_bool]
    # Pivot to long: each numeric column becomes a (value_name, value) pair.
    keep_cols = [c for c in df.columns if c not in ("skipped", "skip_reason",
                                                     "ref_density_method")]
    long = df[keep_cols].melt(id_vars=["molecule"], var_name="value_name",
                               value_name="value")
    long = long.dropna(subset=["value"]).copy()
    long.loc[:, "value"] = pd.to_numeric(long["value"], errors="coerce")
    long = long.dropna(subset=["value"])
    return long


def load_step5_eval_df(run_dir: Path) -> pd.DataFrame:
    """Load the trained-mol eval forest into a tidy long DataFrame.

    Each row carries (arch, loss, solver, molecule, value_name, value).
    """
    run_dir = Path(run_dir)
    rows = []
    for arch in STEP5_ARCHS:
        for loss in STEP5_LOSSES:
            for solver in STEP5_SOLVERS:
                p = run_dir / "eval" / arch / loss / solver / "per_molecule.csv"
                if not p.is_file():
                    continue
                df = pd.read_csv(p)
                long = _per_molecule_rows_to_long(df)
                if long.empty:
                    continue
                long["arch"] = arch
                long["loss"] = loss
                long["solver"] = solver
                rows.append(long)
    if not rows:
        return pd.DataFrame(columns=["arch", "loss", "solver", "molecule",
                                      "value_name", "value"])
    out = pd.concat(rows, ignore_index=True)
    return out[["arch", "loss", "solver", "molecule", "value_name", "value"]]


def load_step5_transfer_df(run_dir: Path) -> pd.DataFrame:
    """Load the test_new (transfer) forest into a tidy long DataFrame.

    Per-spec test_new outputs land at
        test_new/{mol}/{arch}/{loss}/{solver}/per_molecule.csv
    (each per_molecule.csv contains a single row for that molecule).

    Returns rows with columns (arch, loss, solver, molecule, value_name, value).
    Excludes the balancing/* and baseline/* subtrees (those are loaded by
    separate functions).
    """
    run_dir = Path(run_dir)
    rows = []
    test_new = run_dir / "test_new"
    if not test_new.is_dir():
        return pd.DataFrame(columns=["arch", "loss", "solver", "molecule",
                                      "value_name", "value"])
    for mol_dir in test_new.iterdir():
        if not mol_dir.is_dir():
            continue
        for arch in STEP5_ARCHS:
            arch_dir = mol_dir / arch
            if not arch_dir.is_dir():
                continue
            for loss in STEP5_LOSSES:
                for solver in STEP5_SOLVERS:
                    p = arch_dir / loss / solver / "per_molecule.csv"
                    if not p.is_file():
                        continue
                    df = pd.read_csv(p)
                    long = _per_molecule_rows_to_long(df)
                    if long.empty:
                        continue
                    long["arch"] = arch
                    long["loss"] = loss
                    long["solver"] = solver
                    rows.append(long)
    if not rows:
        return pd.DataFrame(columns=["arch", "loss", "solver", "molecule",
                                      "value_name", "value"])
    return pd.concat(rows, ignore_index=True)[
        ["arch", "loss", "solver", "molecule", "value_name", "value"]
    ]


def load_step5_baseline_df(run_dir: Path) -> pd.DataFrame:
    """Load random + pretrained baseline evals on training mols.

    Layout:
        eval_baseline/{arch}/{kind}/per_molecule.csv  with kind in
        {random, pretrained}.

    Returns rows with columns (baseline, arch, molecule, value_name, value).
    """
    run_dir = Path(run_dir)
    rows = []
    base = run_dir / "eval_baseline"
    if not base.is_dir():
        return pd.DataFrame(columns=["baseline", "arch", "molecule",
                                      "value_name", "value"])
    for arch in STEP5_ARCHS:
        for kind in ("random", "pretrained"):
            p = base / arch / kind / "per_molecule.csv"
            if not p.is_file():
                continue
            df = pd.read_csv(p)
            long = _per_molecule_rows_to_long(df)
            if long.empty:
                continue
            long["arch"] = arch
            long["baseline"] = kind
            rows.append(long)
    if not rows:
        return pd.DataFrame(columns=["baseline", "arch", "molecule",
                                      "value_name", "value"])
    return pd.concat(rows, ignore_index=True)[
        ["baseline", "arch", "molecule", "value_name", "value"]
    ]


def load_step5_pretrain_metadata(run_dir: Path) -> pd.DataFrame:
    """Load pretrain_metadata.json per arch into a tidy DataFrame.

    Each row: (arch, final_loss_x, final_loss_c, min_loss_x, min_loss_c,
              loss_weighting, pretrain_steps, duration_seconds).
    """
    run_dir = Path(run_dir)
    rows = []
    base = run_dir / "pretrain"
    if not base.is_dir():
        return pd.DataFrame()
    for arch in STEP5_ARCHS:
        p = base / arch / "pretrain_metadata.json"
        if not p.is_file():
            continue
        with p.open() as f:
            d = json.load(f)
        rows.append({
            "arch": arch,
            "final_loss_x": float(d.get("final_loss_x", float("nan"))),
            "final_loss_c": float(d.get("final_loss_c", float("nan"))),
            "min_loss_x": float(d.get("min_loss_x", float("nan"))),
            "min_loss_c": float(d.get("min_loss_c", float("nan"))),
            "loss_weighting": d.get("loss_weighting", "unknown"),
            "pretrain_steps": int(d.get("pretrain_steps", 0)),
            "duration_seconds": float(d.get("duration_seconds", float("nan"))),
        })
    return pd.DataFrame(rows)


def load_step5_run(run_dir: Path) -> dict:
    """One-call loader: returns dict with eval_df, transfer_df, baseline_df,
    pretrain_meta dataframes for one pretrain-loss-weighting variant."""
    return {
        "eval_df":        load_step5_eval_df(run_dir),
        "transfer_df":    load_step5_transfer_df(run_dir),
        "baseline_df":    load_step5_baseline_df(run_dir),
        "pretrain_meta":  load_step5_pretrain_metadata(run_dir),
    }
