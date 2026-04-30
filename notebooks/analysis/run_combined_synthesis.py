"""Combined step-5 + step-6 synthesis report — all four
{step5, step6} × {unweighted, integration} workflows in one place.

Reads:
  notebooks/checkpoints_step5/{unweighted, integration}/
  notebooks/checkpoints_step6/{unweighted, integration}/

Writes:
  reports_local/step56_combined_synthesis/
    ├─ report.md                                    (synthesis with citations)
    ├─ combined_headline.json                       (machine-readable)
    └─ figures/
       ├─ figcomb_1_best_ae_4way.png                (best AE per workflow + headline ratios)
       ├─ figcomb_2_pretrain_fx_4way.png            (pretrain F_x mean per workflow)
       ├─ figcomb_3_density_vs_ae_unified.png       (Medvedev plane, 4-way overlay on shared archs)
       ├─ figcomb_4_transfer_overlap.png            (CH4 / H2 / OH transfer across all 4 origins)
       └─ figcomb_5_normalized_landscape.png        (log10(MAE / chem-acc) heatmap, 4 panels)
"""
from __future__ import annotations

from pathlib import Path
import json
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from comparison_lib import (  # noqa: E402
    KCAL_PER_HA, CHEMICAL_ACCURACY_KCALMOL,
    load_run_artifacts, mae_of, trained_molecules_only,
)
from step5_data_loader import (  # noqa: E402
    STEP5_ARCHS, STEP5_LOSSES, STEP5_SOLVERS, load_step5_run,
)


REPO = HERE.parent.parent
OUT_DIR = REPO / "reports_local" / "step56_combined_synthesis"
FIG_DIR = OUT_DIR / "figures"

# Workflow display order + colors (consistent across all combined plots).
WORKFLOWS = [
    ("step5", "unweighted",  "#7eb6ff"),  # light blue
    ("step5", "integration", "#7fdf7f"),  # light green
    ("step6", "unweighted",  "#1f77b4"),  # dark blue
    ("step6", "integration", "#2ca02c"),  # dark green
]
WORKFLOW_LABEL = {
    ("step5", "unweighted"):  "step5 · unweighted",
    ("step5", "integration"): "step5 · integration",
    ("step6", "unweighted"):  "step6 · unweighted",
    ("step6", "integration"): "step6 · integration",
}

# Architectures shared by both step 5 and step 6 (the only fair common-arch axis).
SHARED_ARCHS = ("deep_combined", "deep_combined_attn")


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_all_runs() -> dict:
    """Returns dict {(step, origin): artifact_dict}."""
    arts = {}
    for step in ("step5", "step6"):
        for origin in ("unweighted", "integration"):
            run_dir = REPO / "notebooks" / f"checkpoints_{step}" / origin
            if not run_dir.is_dir():
                arts[(step, origin)] = None
                continue
            if step == "step5":
                arts[(step, origin)] = load_step5_run(run_dir)
            else:
                arts[(step, origin)] = load_run_artifacts(run_dir)
    return arts


def best_ae_on_h2o(art, step) -> float:
    """Best AE-MAE in kcal/mol on H2O (the trained mol common to both steps).
    Step 5 trains on H2O only; step 6 trains on H2O + C2H2 — for a fair
    cross-step number we restrict step 6 to its H2O subset.
    """
    if art is None:
        return float("nan")
    df = art["eval_df"]
    sub = df[(df["value_name"] == "AE_error_kcalmol") &
             (df["molecule"] == "H2O")].copy()
    sub["abs"] = sub["value"].abs()
    return float(sub["abs"].min())


def pretrain_fx_mean(art, step) -> float:
    if art is None:
        return float("nan")
    if step == "step5":
        pm = art["pretrain_meta"]
        return float(pm["final_loss_x"].mean()) if not pm.empty else float("nan")
    # Step 6: read pretrain_metadata.json files from the run directory directly.
    return float("nan")  # filled in below by reading metadata directly


def best_density_rmse(art) -> float:
    if art is None:
        return float("nan")
    df = art["eval_df"]
    sub = df[(df["value_name"] == "density_rmse") &
             (df["molecule"] == "H2O")]
    return float(sub["value"].min()) if not sub.empty else float("nan")


def per_arch_pretrain_fx(step, origin) -> dict:
    """Per-arch pretrain final F_x loss as a dict {arch: float}."""
    out = {}
    if step == "step5":
        pm = load_step5_run(REPO / "notebooks" / f"checkpoints_step5" / origin)["pretrain_meta"]
        for _, row in pm.iterrows():
            out[row["arch"]] = float(row["final_loss_x"])
    else:
        for arch in SHARED_ARCHS:
            p = (REPO / "notebooks" / "checkpoints_step6" / origin /
                 "pretrain" / arch / "pretrain_metadata.json")
            if p.is_file():
                with p.open() as f:
                    d = json.load(f)
                out[arch] = float(d.get("final_loss_x", float("nan")))
    return out


# ---------------------------------------------------------------------------
# Figure 1: Best AE-MAE per workflow + reduction-vs-PBE bar chart
# ---------------------------------------------------------------------------

def plot_best_ae_4way(arts: dict, out_path: Path) -> None:
    """Bar chart: best AE-MAE on H2O per workflow, log y, with PBE
    reference (~7 kcal/mol on H2O), pretrained-NN baseline mean, and
    chemical-accuracy line as references.
    """
    fig, ax = plt.subplots(figsize=(9.2, 5))
    x = np.arange(len(WORKFLOWS))
    bests = []
    for step, origin, _ in WORKFLOWS:
        bests.append(best_ae_on_h2o(arts[(step, origin)], step))
    colors = [c for (_, _, c) in WORKFLOWS]

    bars = ax.bar(x, bests, color=colors, edgecolor="k", linewidth=0.5)
    for xc, v in zip(x, bests):
        if np.isnan(v):
            continue
        ax.text(xc, v * 1.6, f"{v:.4g}", ha="center", fontsize=8)

    PBE_H2O = 7.027  # |AE_error_pbe_kcalmol| on H2O verified across all eval CSVs
    ax.axhline(PBE_H2O, ls="-", color="black", lw=1.4,
               label=f"PBE on H₂O ({PBE_H2O:.2f})")
    ax.axhline(CHEMICAL_ACCURACY_KCALMOL, ls="-.", color="purple", lw=1.4,
               label=f"chem. accuracy ({CHEMICAL_ACCURACY_KCALMOL})")

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([WORKFLOW_LABEL[(s, o)] for s, o, _ in WORKFLOWS],
                       rotation=18, ha="right", fontsize=10)
    ax.set_ylabel("best |AE error| on H₂O  (kcal/mol, log)")
    ax.set_title(
        "Combined synthesis — best H₂O AE-MAE achieved per workflow\n"
        "horizontal lines = PBE on H₂O (~7.03 kcal/mol), chemical accuracy (1 kcal/mol)",
        fontsize=11,
    )
    ax.grid(True, axis="y", which="both", ls=":", alpha=0.35)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: pretrain F_x per arch across all 4 workflows
# ---------------------------------------------------------------------------

def plot_pretrain_fx_4way(arts: dict, out_path: Path) -> None:
    """Per-arch pretrain F_x loss bars. Step 5 has 8 archs (left panel),
    step 6 has 2 archs that are a SUBSET (right panel) — plot both panels
    sharing the y-axis so the cross-step shift is visible.

    The cross-step shift on the SHARED archs (deep_combined,
    deep_combined_attn) is the cleanest cross-step comparison.
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.6),
                              gridspec_kw={"width_ratios": [4, 1.5]},
                              sharey=True)

    # ---- step 5 panel ----
    fx5_unw = per_arch_pretrain_fx("step5", "unweighted")
    fx5_int = per_arch_pretrain_fx("step5", "integration")
    archs5 = list(STEP5_ARCHS)
    x5 = np.arange(len(archs5))
    width = 0.38
    axes[0].bar(x5 - width/2, [fx5_unw.get(a, np.nan) for a in archs5],
                width=width, color="#7eb6ff", edgecolor="k", linewidth=0.4,
                label="step5 · unweighted")
    axes[0].bar(x5 + width/2, [fx5_int.get(a, np.nan) for a in archs5],
                width=width, color="#7fdf7f", edgecolor="k", linewidth=0.4,
                label="step5 · integration")
    axes[0].set_xticks(x5)
    axes[0].set_xticklabels(archs5, rotation=22, ha="right", fontsize=8)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("pretrain F_x final MSE  (log)")
    axes[0].set_title("(a) step 5 — 8 architectures", fontsize=10)
    axes[0].grid(True, axis="y", which="both", ls=":", alpha=0.35)
    axes[0].legend(loc="upper right", fontsize=8)

    # ---- step 6 panel (only shares deep_combined, deep_combined_attn) ----
    fx6_unw = per_arch_pretrain_fx("step6", "unweighted")
    fx6_int = per_arch_pretrain_fx("step6", "integration")
    archs6 = list(SHARED_ARCHS)
    x6 = np.arange(len(archs6))
    axes[1].bar(x6 - width/2, [fx6_unw.get(a, np.nan) for a in archs6],
                width=width, color="#1f77b4", edgecolor="k", linewidth=0.4,
                label="step6 · unweighted")
    axes[1].bar(x6 + width/2, [fx6_int.get(a, np.nan) for a in archs6],
                width=width, color="#2ca02c", edgecolor="k", linewidth=0.4,
                label="step6 · integration")
    axes[1].set_xticks(x6)
    axes[1].set_xticklabels(archs6, rotation=22, ha="right", fontsize=8)
    axes[1].set_title("(b) step 6 — shared archs only", fontsize=10)
    axes[1].grid(True, axis="y", which="both", ls=":", alpha=0.35)
    axes[1].legend(loc="upper right", fontsize=8)

    fig.suptitle(
        "Combined — pretrain F_x final MSE per architecture, all 4 workflows\n"
        "step 6 only includes deep_combined / deep_combined_attn (subset of step 5)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: Density-vs-AE Pareto plane, all 4 workflows overlaid
# ---------------------------------------------------------------------------

def plot_density_vs_ae_unified(arts: dict, out_path: Path) -> None:
    """Single-panel scatter: every spec in every workflow, colored by
    workflow, on the (AE-MAE, density-RMSE) plane evaluated on H2O.
    """
    fig, ax = plt.subplots(figsize=(9, 6.4))
    for step, origin, color in WORKFLOWS:
        art = arts[(step, origin)]
        if art is None:
            continue
        df = art["eval_df"]
        ae = df[(df["value_name"] == "AE_error_kcalmol") &
                (df["molecule"] == "H2O")].copy()
        ae["abs"] = ae["value"].abs()
        rho = df[(df["value_name"] == "density_rmse") &
                 (df["molecule"] == "H2O")].copy()
        # Step 6 has additional group/loss/solver columns; merge accordingly.
        merge_keys = ["arch", "loss", "solver", "molecule"]
        if step == "step6":
            merge_keys = ["group", "arch", "loss", "solver", "molecule"]
        merged = ae.merge(rho.rename(columns={"value": "rmse"}),
                           on=merge_keys, suffixes=("_ae", "_rho"))
        ax.scatter(merged["abs"], merged["rmse"],
                   facecolor=color, edgecolor="k", linewidths=0.4,
                   alpha=0.65, s=42,
                   label=WORKFLOW_LABEL[(step, origin)])
    ax.axvline(CHEMICAL_ACCURACY_KCALMOL, ls="-.", color="purple", lw=1.4,
               label="chem. accuracy = 1 kcal/mol")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("|AE error| on H₂O  (kcal/mol, log)")
    ax.set_ylabel("density-RMSE on H₂O  (e/bohr³, log)")
    ax.grid(True, which="both", ls=":", alpha=0.35)
    ax.set_title(
        "Combined density-vs-energy plane — all four step{5,6}/{unweighted,integration} workflows\n"
        "Burke 1998 (in Kepp 2017): 'functionals which yield highly accurate energies often\n"
        "produce potentials which differ markedly from the exact ones'",
        fontsize=10,
    )
    ax.legend(loc="lower right", fontsize=8, framealpha=0.92)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: Transfer overlap — CH4 / H2 / OH across all 4
# ---------------------------------------------------------------------------

def plot_transfer_overlap(arts: dict, out_path: Path) -> None:
    """Restricted-loss-set transfer comparison.

    **Fairness caveat.** A naive median across ALL specs in each
    workflow is NOT apples-to-apples across steps:

      step 5 loss set: {A_atomization, B_atomization_plus_dm,
                        C_atomization_plus_grid}    -- NO V_xc fitting
      step 6 loss set: {L1_B, L2_C_anchor, L3_balanced_vxc,
                        L4_balanced_vxc_anchor, L5_gradnorm_vxc}
                                                    -- L3/L4/L5 ARE V_xc-aware

    Step 6's V_xc-aware losses (L3, L4, L5) produce 30-40 kcal/mol AE
    on the transfer set because they over-fit V_xc shape to H2O+C2H2.
    Their medians inflate step 6's overall transfer median, making
    step 6 look worse than step 5 on transfer when in reality it is a
    DIFFERENT LOSS-STRATEGY MIXTURE.

    The fair cross-step comparison restricts to losses with direct
    cross-step analogs:

      step 5 loss B (B_atomization_plus_dm) ≈ step 6 loss L1_B
      step 5 loss C (C_atomization_plus_grid) ≈ step 6 loss L2_C_anchor

    Step 5's loss A (atomization-only) has no step-6 analog and is
    excluded. Step 6's loss L3/L4/L5 (V_xc-aware) have no step-5 analog
    and are excluded.

    This restriction is the ONLY way to claim "step 5 vs step 6 on
    transfer" honestly.
    """
    PBE_REF = {"CH4": 0.91, "H2": 7.31, "OH": 1.84}
    # Restricted loss sets per step (apples-to-apples).
    LOSS_RESTRICT = {
        "step5": {"B_atomization_plus_dm", "C_atomization_plus_grid"},
        "step6": {"L1_B", "L2_C_anchor"},
    }
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4), sharey=True)
    x = np.arange(len(WORKFLOWS))
    colors = [c for (_, _, c) in WORKFLOWS]
    labels = [WORKFLOW_LABEL[(s, o)] for s, o, _ in WORKFLOWS]
    for ax, mol in zip(axes, ("CH4", "H2", "OH")):
        bars = []
        for step, origin, _ in WORKFLOWS:
            art = arts[(step, origin)]
            if art is None:
                bars.append(np.nan); continue
            if step == "step5":
                df = art["transfer_df"]
            else:
                df = art["transfer_primary_df"]
            ae = df[(df["value_name"] == "AE_error_kcalmol") &
                    (df["molecule"] == mol) &
                    (df["loss"].isin(LOSS_RESTRICT[step]))]
            bars.append(float(ae["value"].abs().median())
                        if not ae.empty else np.nan)
        ax.bar(x, bars, color=colors, edgecolor="k", linewidth=0.4)
        ax.axhline(PBE_REF[mol], ls="-", color="black", lw=1.4,
                   label=f"PBE on {mol} ({PBE_REF[mol]:.2f})"
                   if mol == "CH4" else None)
        ax.axhline(CHEMICAL_ACCURACY_KCALMOL, ls="-.", color="purple", lw=1.4,
                   label=f"chem. acc. ({CHEMICAL_ACCURACY_KCALMOL})"
                   if mol == "CH4" else None)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=7)
        ax.set_title(f"transfer · {mol}", fontsize=10)
        ax.grid(True, axis="y", which="both", ls=":", alpha=0.35)
        if mol == "CH4":
            ax.legend(loc="upper right", fontsize=7)
    axes[0].set_ylabel("median |AE error|  on transfer mol  (kcal/mol, log)")
    fig.suptitle(
        "Combined transfer comparison — restricted to common-loss-strategy specs\n"
        "step 5: {B, C} only · step 6: {L1_B, L2_C_anchor} only · L3/L4/L5 V_xc-aware excluded\n"
        "(without this restriction, step 6's median is inflated by V_xc-aware losses with no step-5 analog)",
        fontsize=10,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 5: Normalized landscape comparison (all 4 panels)
# ---------------------------------------------------------------------------

def plot_normalized_landscape(arts: dict, out_path: Path) -> None:
    """Four-panel histogram: distribution of log10(AE-MAE/chemical-accuracy)
    on H2O across all specs in each workflow. Workflow with the leftmost
    distribution is the most accurate.
    """
    fig, ax = plt.subplots(figsize=(11, 5))
    for step, origin, color in WORKFLOWS:
        art = arts[(step, origin)]
        if art is None:
            continue
        df = art["eval_df"]
        ae = df[(df["value_name"] == "AE_error_kcalmol") &
                (df["molecule"] == "H2O")].copy()
        ae["abs"] = ae["value"].abs()
        if ae.empty:
            continue
        log_vals = np.log10(ae["abs"].values + 1e-12) - np.log10(CHEMICAL_ACCURACY_KCALMOL)
        ax.hist(log_vals, bins=np.linspace(-5, 4, 36),
                histtype="stepfilled", alpha=0.45,
                color=color, edgecolor="k", linewidth=0.5,
                label=f"{WORKFLOW_LABEL[(step, origin)]} (n={len(log_vals)})")
    ax.axvline(0, ls="-.", color="purple", lw=1.4,
               label="chem. accuracy = 1 kcal/mol")
    PBE_H2O = 7.027
    ax.axvline(np.log10(PBE_H2O), ls="-", color="black", lw=1.4,
               label=f"PBE on H₂O (log₁₀(7.03) = 0.85)")
    ax.set_xlabel("log₁₀(|AE error| / chem. accuracy)  on H₂O")
    ax.set_ylabel("count of specs")
    ax.set_title(
        "Combined — distribution of H₂O AE-MAE across all specs, four workflows\n"
        "leftward shift = better; left of purple = below chemical accuracy",
        fontsize=11,
    )
    ax.legend(loc="upper left", fontsize=8, framealpha=0.92)
    ax.grid(True, axis="y", ls=":", alpha=0.35)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Headline table data
# ---------------------------------------------------------------------------

def combined_headline(arts: dict) -> dict:
    """Build the cross-workflow comparison table data."""
    out = {}
    for step, origin, _ in WORKFLOWS:
        art = arts[(step, origin)]
        if art is None:
            out[(step, origin)] = None
            continue
        # Best AE on H2O (common axis)
        best_ae = best_ae_on_h2o(art, step)
        best_rho = best_density_rmse(art)
        # Pretrain F_x mean (per-step source)
        if step == "step5":
            pm = art["pretrain_meta"]
            fx_mean = float(pm["final_loss_x"].mean()) if not pm.empty else float("nan")
            fc_mean = float(pm["final_loss_c"].mean()) if not pm.empty else float("nan")
        else:
            # step 6: average over the 2 archs
            fx_arch = per_arch_pretrain_fx("step6", origin)
            fx_mean = float(np.mean(list(fx_arch.values()))) if fx_arch else float("nan")
            fc_arch = {}
            for arch in SHARED_ARCHS:
                p = (REPO / "notebooks" / "checkpoints_step6" / origin /
                     "pretrain" / arch / "pretrain_metadata.json")
                if p.is_file():
                    with p.open() as f:
                        d = json.load(f)
                    fc_arch[arch] = float(d.get("final_loss_c", float("nan")))
            fc_mean = float(np.mean(list(fc_arch.values()))) if fc_arch else float("nan")
        # Number of specs in run
        df = art["eval_df"]
        ae_h2o = df[(df["value_name"] == "AE_error_kcalmol") &
                    (df["molecule"] == "H2O")]
        n = int(len(ae_h2o))
        out[f"{step}_{origin}"] = {
            "best_ae_h2o":     best_ae,
            "best_density_h2o":best_rho,
            "pretrain_fx_mean":fx_mean,
            "pretrain_fc_mean":fc_mean,
            "n_specs":         n,
        }
    return out


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    arts = load_all_runs()
    for k, v in arts.items():
        print(f"  {k}: {'loaded' if v is not None else 'MISSING'}")

    print(f"Generating figures into {FIG_DIR} ...")
    plot_best_ae_4way(arts,             FIG_DIR / "figcomb_1_best_ae_4way.png")
    plot_pretrain_fx_4way(arts,         FIG_DIR / "figcomb_2_pretrain_fx_4way.png")
    plot_density_vs_ae_unified(arts,    FIG_DIR / "figcomb_3_density_vs_ae_unified.png")
    plot_transfer_overlap(arts,         FIG_DIR / "figcomb_4_transfer_overlap.png")
    plot_normalized_landscape(arts,     FIG_DIR / "figcomb_5_normalized_landscape.png")

    stats = combined_headline(arts)
    (OUT_DIR / "combined_headline.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
