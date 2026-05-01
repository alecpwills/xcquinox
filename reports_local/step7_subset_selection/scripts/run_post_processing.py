"""Step-7 post-processing: 8 figures + headline.json from 80 eval_df.csv files (10 sizes x 2 metrics x 2 augs x 2 solvers; r=21 excluded as full pool)."""
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

SUBSET_SIZES = (1, 2, 3, 4, 5, 6, 7, 12, 15, 18)  # r=21 excluded (full pool, no selection)
R_MAX = max(SUBSET_SIZES)  # largest *selected* subset size — used for "best subset"
                            # overlay (Plot 5) and probe-comparison reference (Plot 7).
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

    # ---- Plot 5: descriptor overlay (full pool vs largest-r subsets) -----------
    # Uses R_MAX (largest *selected* subset size), not the full pool size — the
    # full pool reference and an r=N_pool subset would be identical.
    ref_npz = np.load(STEP7_ROOT / "dfs_pool_full_hist" / "reference.npz")
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    # Full-pool reference (black)
    axes[0].plot(ref_npz["e_rho"][:-1], ref_npz["h_ref_rho"], "k-", label="full pool")
    axes[1].plot(ref_npz["e_s"][:-1], ref_npz["h_ref_s"], "k-", label="full pool")
    axes[2].plot(ref_npz["e_alpha"][:-1], ref_npz["h_ref_alpha"], "k-", label="full pool")
    # r=R_MAX, aug=False overlay per metric (colored)
    plot5_colors = {"l2": "tab:blue", "jsd": "tab:orange"}
    desc_cache = STEP7_ROOT / "subset_descriptors"
    for metric in METRICS:
        key = f"{metric}/{R_MAX}/False"
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
                     color=color, label=f"{metric} r={R_MAX}")
        axes[1].plot(ref_npz["e_s"][:-1], h_sub["s"],
                     color=color, label=f"{metric} r={R_MAX}")
        axes[2].plot(ref_npz["e_alpha"][:-1], h_sub["alpha"],
                     color=color, label=f"{metric} r={R_MAX}")
    for ax, lab in zip(axes, (r"$\rho^{1/3}$", "s", r"$\alpha$")):
        ax.set_xlabel(f"log10 {lab}")
        ax.set_ylabel("density")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    axes[1].set_title(f"Plot 5: Descriptor histograms (full pool vs r={R_MAX} subsets)")
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

    # ----------------------------------------------------------------------
    # T23 — Held-out probe-set comprehensive analysis
    #
    # Reads the per-spec eval_df.csv files in long-form schema:
    #   columns = metric / tag / solver / set / molecule / value_name / value
    # where `set` ∈ {training_subset, probe_a_chemical_similarity,
    #                probe_b_heteroatom, probe_c_bh76_transfer,
    #                probe_d_multireference}.
    #
    # Two figures + two tables + headline['probe_summary'].
    # ----------------------------------------------------------------------
    PROBE_NAMES = (
        "probe_a_chemical_similarity",
        "probe_b_heteroatom",
        "probe_c_bh76_transfer",
        "probe_d_multireference",
    )

    def _probe_signed_errors(_df_unused: pd.DataFrame, probe_name: str) -> pd.DataFrame:
        """Walk eval_probes/<probe>/per_molecule.json across all specs and
        extract per-molecule signed errors. Replaces the old long-form
        eval_df.csv read — eval_df.csv is now wide-form (one row per
        spec×set with aggregated mae), so per-molecule detail comes
        directly from the per_molecule.json files written by alec.run_test.

        Returns columns [metric, r, aug, solver, set, molecule, err_kcalmol].
        """
        rows = []
        err_keys = ("atomization_energy_error_kcalmol",
                    "ae_error_kcalmol",
                    "AE_error_kcalmol")
        for metric in METRICS:
            for r in SUBSET_SIZES:
                for aug in AUGMENTATIONS:
                    tag = f"bin{r:02d}{'w' if aug else ''}"
                    aug_label = "w" if aug else "nw"
                    for solver in SOLVERS:
                        pm_path = (STEP7_ROOT / metric / tag /
                                   "deep_combined_attn" /
                                   "L5_gradnorm_vxc_step7" / solver /
                                   "eval_probes" / probe_name /
                                   "per_molecule.json")
                        if not pm_path.exists():
                            continue
                        try:
                            pm = json.loads(pm_path.read_text())
                        except (json.JSONDecodeError, OSError):
                            continue
                        for entry in pm:
                            mol = entry.get("name") or entry.get("molecule")
                            err_val = next(
                                (entry[k] for k in err_keys if k in entry
                                 and isinstance(entry[k], (int, float))),
                                None)
                            if err_val is None:
                                continue
                            rows.append({
                                "metric": metric, "r": r, "aug": aug_label,
                                "solver": solver, "set": probe_name,
                                "molecule": mol,
                                "err_kcalmol": float(err_val),
                            })
        if not rows:
            return pd.DataFrame(
                columns=["metric", "r", "aug", "solver", "set",
                         "molecule", "err_kcalmol"])
        return pd.DataFrame(rows)

    # Build a tidy errors-frame across all probes.
    err_frames = [_probe_signed_errors(df, p) for p in PROBE_NAMES]
    df_err = pd.concat(err_frames, ignore_index=True) if err_frames else pd.DataFrame()

    # ---- Plot 7: probe-set MAE comparison at r=R_MAX (grouped bar) ----
    if not df_err.empty:
        # MAE per (set, metric, solver, aug) at the largest selected r.
        mae_rmax = (
            df_err[df_err["r"] == R_MAX]
            .assign(abs_err=lambda x: x["err_kcalmol"].abs())
            .groupby(["set", "metric", "solver", "aug"])["abs_err"].mean()
            .reset_index(name="mae")
        )
        if not mae_rmax.empty:
            fig, ax = plt.subplots(figsize=(12, 5))
            mae_rmax["spec_label"] = mae_rmax.apply(
                lambda x: f"{x['metric']}/{x['solver']}/{x['aug']}", axis=1)
            pivot = mae_rmax.pivot(index="set", columns="spec_label", values="mae")
            pivot = pivot.reindex(index=PROBE_NAMES)
            pivot.plot(kind="bar", ax=ax, width=0.85)
            ax.axhline(1.0, color="r", linestyle="--",
                       label="chemical accuracy 1 kcal/mol")
            ax.set_ylabel("MAE (kcal/mol)")
            ax.set_xlabel("probe set")
            ax.set_title(f"Plot 7: Held-out probe-set MAE comparison (r={R_MAX})")
            ax.set_yscale("log")
            ax.legend(fontsize=7, ncol=2, loc="upper right")
            ax.grid(alpha=0.3, axis="y")
            for tick in ax.get_xticklabels():
                tick.set_rotation(20)
                tick.set_ha("right")
            fig.tight_layout()
            fig.savefig(FIGS_DIR / "plot7_probe_comparison.png", dpi=150)
            plt.close(fig)

    # ---- Plot 8: cross-probe heatmap (probes × specs) ----
    if not df_err.empty:
        mae_all_r = (
            df_err
            .assign(abs_err=lambda x: x["err_kcalmol"].abs())
            .groupby(["set", "metric", "solver", "aug", "r"])["abs_err"].mean()
            .reset_index(name="mae")
        )
        # Pick the BEST r for each (probe, metric, solver, aug)
        best = (
            mae_all_r
            .loc[mae_all_r.groupby(["set", "metric", "solver", "aug"])["mae"].idxmin()]
        )
        if not best.empty:
            best["spec_label"] = best.apply(
                lambda x: f"{x['metric']}/{x['solver']}/{x['aug']}", axis=1)
            heat = best.pivot(index="set", columns="spec_label", values="mae")
            heat = heat.reindex(index=PROBE_NAMES)
            fig, ax = plt.subplots(figsize=(12, 5))
            log_heat = np.log10(heat.values + 1e-6)
            im = ax.imshow(log_heat, aspect="auto", cmap="viridis")
            ax.set_xticks(range(heat.shape[1]))
            ax.set_xticklabels(heat.columns, rotation=45, ha="right", fontsize=7)
            ax.set_yticks(range(heat.shape[0]))
            ax.set_yticklabels(heat.index, fontsize=8)
            for i in range(heat.shape[0]):
                for j in range(heat.shape[1]):
                    val = heat.values[i, j]
                    if not np.isnan(val):
                        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                                color="white" if log_heat[i, j] > log_heat.mean() else "black",
                                fontsize=6)
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("log10 MAE (kcal/mol)")
            ax.set_title("Plot 8: Cross-probe MAE heatmap (best r per spec)")
            fig.tight_layout()
            fig.savefig(FIGS_DIR / "plot8_probe_heatmap.png", dpi=150)
            plt.close(fig)

    # ---- Table 1: comprehensive markdown table (probe × metric × solver × aug) ----
    if not df_err.empty:
        agg = (
            df_err
            .assign(abs_err=lambda x: x["err_kcalmol"].abs(),
                    sq_err=lambda x: x["err_kcalmol"] ** 2)
            .groupby(["set", "metric", "solver", "aug", "r"])
            .agg(MAE=("abs_err", "mean"),
                 RMSE=("sq_err", lambda x: float(np.sqrt(x.mean()))),
                 N_eval=("err_kcalmol", "size"))
            .reset_index()
        )
        # For each (set, metric, solver, aug) pick the best r (lowest MAE).
        best_rows = (
            agg.loc[agg.groupby(["set", "metric", "solver", "aug"])["MAE"].idxmin()]
        )
        # Sort for readability: probe -> metric -> solver -> aug.
        best_rows = best_rows.sort_values(
            by=["set", "metric", "solver", "aug"]).reset_index(drop=True)
        md_lines = [
            "# Step-7 Probe Summary (T23)",
            "",
            "Comprehensive across-probe table.  For each (probe, metric, "
            "solver, aug) the row shows the best subset-size r and the "
            "corresponding MAE/RMSE on the held-out probe set.",
            "",
            "| probe | metric | solver | aug | r | MAE (kcal/mol) | RMSE (kcal/mol) | N_eval |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for _, row in best_rows.iterrows():
            md_lines.append(
                f"| {row['set']} | {row['metric']} | {row['solver']} | "
                f"{row['aug']} | {int(row['r'])} | "
                f"{row['MAE']:.3f} | {row['RMSE']:.3f} | {int(row['N_eval'])} |"
            )
        out_md = (REPO / "reports_local" / "step7_subset_selection" /
                  "probe_summary.md")
        out_md.write_text("\n".join(md_lines) + "\n")
        # headline['probe_summary'] = best (probe, metric) row records
        headline["probe_summary"] = best_rows.to_dict(orient="records")

    # ---- Table 2: per-molecule errors at r=R_MAX (CSV) ----
    if not df_err.empty:
        per_mol = df_err[df_err["r"] == R_MAX].copy()
        if not per_mol.empty:
            # Reference value: refer to eval_probes for AE/IP refs.
            # To keep the CSV self-contained, we attach probe-level
            # provenance by joining against ALL_PROBES from eval_probes.
            try:
                from xcquinox.alec import eval_probes
                ref_records: list = []
                for pn in PROBE_NAMES:
                    pp = eval_probes.ALL_PROBES[pn]
                    if eval_probes.PROBE_KIND[pn] == "ae":
                        for entry in pp:
                            ref_records.append({
                                "set": pn,
                                "molecule": entry["name"],
                                "E_ref_kcalmol": float(entry["ae_kcalmol"]),
                                "hill": entry["hill"],
                                "source": entry["source"],
                            })
                    else:  # bh76
                        for rxn in pp:
                            ref_records.append({
                                "set": pn,
                                "molecule": rxn["name"],
                                "E_ref_kcalmol": float(rxn["e_rxn_ref"]),
                                "hill": rxn["name"],
                                "source": rxn["source"],
                            })
                ref_df = pd.DataFrame(ref_records)
                per_mol = per_mol.merge(
                    ref_df, on=["set", "molecule"], how="left")
                per_mol["E_NN_kcalmol"] = (
                    per_mol["E_ref_kcalmol"] + per_mol["err_kcalmol"])
            except ImportError:
                # eval_probes not importable from this script context;
                # leave reference columns blank.
                per_mol["E_ref_kcalmol"] = np.nan
                per_mol["E_NN_kcalmol"] = np.nan
                per_mol["hill"] = ""
                per_mol["source"] = ""
            # Pick the best (metric, solver, aug) per probe (lowest MAE at r=R_MAX)
            mae_per_spec = (
                per_mol.assign(abs_err=lambda x: x["err_kcalmol"].abs())
                .groupby(["set", "metric", "solver", "aug"])["abs_err"].mean()
                .reset_index(name="mae")
            )
            best_spec_per_probe = (
                mae_per_spec
                .loc[mae_per_spec.groupby("set")["mae"].idxmin()]
                [["set", "metric", "solver", "aug"]]
            )
            best_per_mol = per_mol.merge(
                best_spec_per_probe,
                on=["set", "metric", "solver", "aug"], how="inner"
            )
            cols = ["set", "molecule", "hill", "E_ref_kcalmol",
                    "E_NN_kcalmol", "err_kcalmol", "metric", "solver", "aug"]
            cols = [c for c in cols if c in best_per_mol.columns]
            csv_path = (REPO / "reports_local" / "step7_subset_selection" /
                        f"per_molecule_errors_r{R_MAX}.csv")
            best_per_mol[cols].to_csv(csv_path, index=False)
            print(f"Wrote per-molecule errors CSV to {csv_path}")

    HEADLINE_PATH.write_text(json.dumps(headline, indent=2, default=str))
    print(f"Wrote 6 + 2 figures + 2 tables + headline to {HEADLINE_PATH}")


if __name__ == "__main__":
    main()
