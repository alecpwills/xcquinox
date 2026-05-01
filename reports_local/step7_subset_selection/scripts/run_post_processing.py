"""Step-7 post-processing: 6+1 figures + headline.json from 88 eval_df.csv files."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]  # scripts/ -> step7_subset_selection/ -> reports_local/ -> repo root
STEP7_ROOT = REPO / "notebooks" / "checkpoints_step7"
FIGS_DIR = REPO / "reports_local" / "step7_subset_selection" / "figures"
HEADLINE_PATH = REPO / "reports_local" / "step7_subset_selection" / "headline.json"
FIGS_DIR.mkdir(parents=True, exist_ok=True)

SUBSET_SIZES = (1, 2, 3, 4, 5, 6, 7, 12, 15, 18, 21)
METRICS = ("l2", "jsd")
SOLVERS = ("oneshot", "full_3")
AUGMENTATIONS = (False, True)
LOG_REGULARIZER = 1e-10


def _load_specs() -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        for r in SUBSET_SIZES:
            for aug in AUGMENTATIONS:
                tag = f"bin{r:02d}{'w' if aug else ''}"
                for solver in SOLVERS:
                    eval_csv = (STEP7_ROOT / metric / tag /
                                "deep_combined_attn" / "L5_gradnorm_vxc_step7" /
                                solver / "eval_df.csv")
                    if not eval_csv.exists():
                        continue
                    df = pd.read_csv(eval_csv)
                    df["metric"] = metric
                    df["r"] = r
                    df["aug"] = "w" if aug else "nw"
                    df["solver"] = solver
                    rows.append(df)
    if not rows:
        raise SystemExit("No eval_df.csv files found; run training first.")
    return pd.concat(rows, ignore_index=True)


def main():
    df = _load_specs()
    headline = {}

    # ---- Plot 1: subset-size sweep, AE-MAE on held-out diet150 ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for metric in METRICS:
        for solver in SOLVERS:
            for aug in ("nw", "w"):
                sub = df[(df["metric"] == metric) & (df["solver"] == solver)
                         & (df["aug"] == aug) & (df["set"] == "held_out_diet150")]
                if sub.empty:
                    continue
                grp = sub.groupby("r")["mae"].mean().sort_index()
                ax.plot(grp.index, grp.values, marker="o",
                        label=f"{metric}/{solver}/{aug}")
    ax.set_xlabel("subset size r")
    ax.set_ylabel("AE-MAE on held-out diet150 (kcal/mol)")
    ax.set_yscale("log")
    ax.set_title("Plot 1: Subset-size sweep (in-distribution AE)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "plot1_subset_sweep_ae.png", dpi=150)
    plt.close(fig)

    # ---- Plot 2: subset-size sweep, ρ-RMSE ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for metric in METRICS:
        for solver in SOLVERS:
            for aug in ("nw", "w"):
                sub = df[(df["metric"] == metric) & (df["solver"] == solver)
                         & (df["aug"] == aug) & (df["set"] == "held_out_diet150")]
                if sub.empty:
                    continue
                grp = sub.groupby("r")["rho_rmse"].mean().sort_index()
                ax.plot(grp.index, grp.values, marker="o",
                        label=f"{metric}/{solver}/{aug}")
    ax.set_xlabel("subset size r")
    ax.set_ylabel("ρ-RMSE on held-out diet150 (e/bohr³)")
    ax.set_yscale("log")
    ax.set_title("Plot 2: Subset-size sweep (in-distribution ρ-RMSE)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "plot2_subset_sweep_rho.png", dpi=150)
    plt.close(fig)

    # ---- Plot 3: subset overlap (Jaccard) heatmap of L2 vs JSD selections ----
    ledger_path = STEP7_ROOT / "subset_index_log.json"
    ledger = json.loads(ledger_path.read_text())
    overlap = np.zeros((len(SUBSET_SIZES), len(AUGMENTATIONS)))
    for i, r in enumerate(SUBSET_SIZES):
        for j, aug in enumerate(AUGMENTATIONS):
            if f"l2/{r}/{aug}" not in ledger or f"jsd/{r}/{aug}" not in ledger:
                continue
            l2_idx = set(ledger[f"l2/{r}/{aug}"]["chosen_indices"])
            jsd_idx = set(ledger[f"jsd/{r}/{aug}"]["chosen_indices"])
            jaccard = (len(l2_idx & jsd_idx) /
                       max(len(l2_idx | jsd_idx), 1))
            overlap[i, j] = jaccard
    fig, ax = plt.subplots(figsize=(5, 6))
    im = ax.imshow(overlap, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(AUGMENTATIONS)))
    ax.set_xticklabels(["no_w", "w"])
    ax.set_yticks(range(len(SUBSET_SIZES)))
    ax.set_yticklabels(SUBSET_SIZES)
    ax.set_ylabel("subset size r")
    ax.set_xlabel("augmentation")
    ax.set_title("Plot 3: Jaccard(L2-chosen, JSD-chosen) overlap")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "plot3_overlap_jaccard.png", dpi=150)
    plt.close(fig)

    # ---- Plot 4: histogram-fit quality vs r ----
    fig, ax = plt.subplots(figsize=(8, 5))
    for metric in METRICS:
        xs, ys = [], []
        for r in SUBSET_SIZES:
            key = f"{metric}/{r}/False"
            if key not in ledger:
                continue
            xs.append(r)
            ys.append(ledger[key]["metric_value"])
        ax.plot(xs, ys, marker="o", label=metric)
    ax.set_xlabel("subset size r")
    ax.set_ylabel("metric value (lower is closer to full pool)")
    ax.set_yscale("log")
    ax.set_title("Plot 4: Histogram-fit quality vs r")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "plot4_metric_vs_r.png", dpi=150)
    plt.close(fig)

    # ---- Plot 5: descriptor distribution overlay (full pool vs r=21 subsets) ----
    ref_npz = np.load(STEP7_ROOT / "dick_pool_full_hist" / "reference.npz")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    # Full-pool reference (black)
    axes[0].plot(ref_npz["e_rho"][:-1], ref_npz["h_ref_rho"], "k-", label="full pool")
    axes[1].plot(ref_npz["e_s"][:-1], ref_npz["h_ref_s"], "k-", label="full pool")
    axes[2].plot(ref_npz["e_alpha"][:-1], ref_npz["h_ref_alpha"], "k-", label="full pool")
    # r=21, aug=False overlay per metric (colored)
    plot5_colors = {"l2": "tab:blue", "jsd": "tab:orange"}
    desc_cache = STEP7_ROOT / "subset_descriptors"
    for metric in METRICS:
        key = f"{metric}/21/False"
        if key not in ledger:
            continue
        chosen = ledger[key]["chosen_indices"]
        # Load and concatenate descriptor arrays for the chosen pool entries
        arrs: dict = {"rho_third": [], "s": [], "alpha": [], "weights": []}
        for idx in chosen:
            matches = sorted(desc_cache.glob(f"{idx}_*.npz"))
            if not matches:
                continue
            z = np.load(matches[0])
            for k in ("rho_third", "s", "alpha", "weights"):
                arrs[k].append(z[k])
        if not arrs["rho_third"]:
            continue
        cat = {k: np.concatenate(arrs[k]) for k in ("rho_third", "s", "alpha", "weights")}
        # Rebin using the reference edges so histograms are aligned
        h_sub = {}
        for k, ek in (("rho_third", "e_rho"), ("s", "e_s"), ("alpha", "e_alpha")):
            log_x = np.log10(cat[k] + LOG_REGULARIZER)
            h, _ = np.histogram(log_x, bins=ref_npz[ek], weights=cat["weights"],
                                density=True)
            h_sub[k] = h
        color = plot5_colors[metric]
        axes[0].plot(ref_npz["e_rho"][:-1], h_sub["rho_third"],
                     color=color, label=f"{metric} r=21")
        axes[1].plot(ref_npz["e_s"][:-1], h_sub["s"],
                     color=color, label=f"{metric} r=21")
        axes[2].plot(ref_npz["e_alpha"][:-1], h_sub["alpha"],
                     color=color, label=f"{metric} r=21")
    for ax, lab in zip(axes, (r"$\rho^{1/3}$", "s", r"$\alpha$")):
        ax.set_xlabel(f"log10 {lab}")
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    axes[1].set_title("Plot 5: Descriptor histograms (full pool vs r=21 subsets)")
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "plot5_descriptor_overlay.png", dpi=150)
    plt.close(fig)

    # ---- Plot 6: W4-11 transfer MAE bar chart ----
    fig, ax = plt.subplots(figsize=(10, 5))
    bar_data = (
        df[df["set"] == "w4_11"]
        .groupby(["metric", "r", "solver", "aug"])["mae"]
        .mean()
        .reset_index()
    )
    if not bar_data.empty:
        labels = bar_data.apply(
            lambda x: f"{x['metric']}/r={x['r']}/{x['solver']}/{x['aug']}", axis=1
        )
        ax.bar(range(len(bar_data)), bar_data["mae"].values)
        ax.axhline(1.0, color="r", linestyle="--", label="chemical accuracy 1 kcal/mol")
        ax.set_xticks(range(len(bar_data)))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_ylabel("AE-MAE on W4-11 (kcal/mol)")
        ax.set_yscale("log")
        ax.set_title("Plot 6: W4-11 transfer MAE per spec")
        ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS_DIR / "plot6_w411_transfer.png", dpi=150)
    plt.close(fig)

    # ---- Headline table ----
    w411 = df[df["set"] == "w4_11"]
    if not w411.empty:
        headline["best_per_metric_solver_aug"] = (
            w411
            .loc[w411.groupby(["metric", "solver", "aug"])["mae"].idxmin()]
            [["metric", "solver", "aug", "r", "mae", "rho_rmse"]]
            .to_dict(orient="records")
        )
    else:
        headline["best_per_metric_solver_aug"] = []
    HEADLINE_PATH.write_text(json.dumps(headline, indent=2))
    print(f"Wrote 6 figures + headline to {HEADLINE_PATH}")


if __name__ == "__main__":
    main()
