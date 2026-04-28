"""Reusable analysis helpers for step-6 sweep evaluation.

Designed so the unweighted-pretrain sweep, the (in-progress)
integration-pretrain sweep, and any future pretrain-loss-weighting
variants can be analyzed by the same code path. Each function carries
the primary-source citation that justifies the physical/chemical
quantity it computes.

Quantities and their primary references
---------------------------------------

* **Atomization energy (AE)** ``AE_NN = sum_atoms E_atom_NN - E_mol_NN``.
  Convention follows pyscf (``mol.atom_coords()`` + ``mf.e_tot``) and
  the standard Pople-group definition used in W4-11.

  - Karton, Daon, Martin, *Chem. Phys. Lett.* **510**, 165 (2011)
    -- W4-11: 140 total atomization energies at sub-kcal/mol accuracy
    against post-CCSD(T) extrapolation.

* **Atomic reference totals** for ``E_atom_ref``: Chakravorty, Gwaltney,
  Davidson, Parpia, Fischer, *Phys. Rev. A* **47**, 3649 (1993) --
  non-relativistic ground-state HF + correlation totals for atoms
  H-Ar, basis-set complete.

* **Chemical accuracy threshold** = 1 kcal/mol.
  - Pople, *Rev. Mod. Phys.* **71**, 1267 (1999) (Nobel lecture):
    introduces "chemical accuracy" as the universal benchmark.

* **Density quality metrics**: density-RMSE and density-L1 are computed
  on the ``mf.grids.coords`` quadrature mesh and weighted by the
  Becke quadrature weights ``mf.grids.weights``. Becke quadrature is
  the standard for 3-D molecular numerical integration.

  - Becke, *J. Chem. Phys.* **88**, 2547 (1988) -- atom-centered
    multiexponential quadrature (the basis for pyscf's ``GRIDS``
    object).

* **Density-vs-energy trade-off** in trained DFT functionals:

  - Medvedev, Bushmarinov, Sun, Perdew, Lyssenko, *Science* **355**,
    49 (2017): "Density functional theory is straying from the path
    toward the exact functional." Demonstrates that empirical
    functionals trained against energy benchmarks give better
    energies but WORSE densities than physically-motivated functionals
    (PBE, SCAN). Frames the tradeoff a learned XC functional must
    navigate.

* **Transferability of fitted functionals**: in-distribution fit
  quality is necessary but not sufficient for out-of-distribution
  performance.

  - Behler, Parrinello, *Phys. Rev. Lett.* **98**, 146401 (2007):
    canonical reference for neural-network potentials and the role of
    holdout-set evaluation; Section II discusses transferability vs
    overfitting.

* **SCF cycles run** is the number of inner-loop iterations the
  ``full_3`` solver spent before its convergence criterion fired (or
  hitting ``max_cycles=3``). Lower values at the same final density
  indicate a smoother V_xc surface that does not require as many
  Roothaan iterations.

  - Roothaan, *Rev. Mod. Phys.* **23**, 69 (1951): Hartree-Fock SCF.
  - Pulay, *Chem. Phys. Lett.* **73**, 393 (1980): DIIS acceleration
    (used inside our SCF).

All other physics conventions (Hartree atomic units, kcal/mol = 627.5094740631 Ha,
PBE GGA exchange-correlation: Perdew-Burke-Ernzerhof, *Phys. Rev. Lett.*
**77**, 3865, 1996) follow the broader codebase.
"""
from __future__ import annotations

from pathlib import Path
import json
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

# Physical / unit constants
KCAL_PER_HA = 627.5094740631  # CODATA 2018; same constant used in alec.config.

# Chemical accuracy threshold (Pople 1999 Nobel lecture).
CHEMICAL_ACCURACY_KCALMOL = 1.0

# Loss-strategy display order. L5_gradnorm_vxc was added 2026-04-28 as the
# dynamic-balancing comparator to L3 (LossNorm step-0 balancing).
LOSS_DISPLAY_ORDER = (
    "L1_B",
    "L2_C_anchor",
    "L3_balanced_vxc",
    "L4_balanced_vxc_anchor",
    "L5_gradnorm_vxc",
)
LOSS_DISPLAY_LABELS = {
    "L1_B":                   "L1 · B + DM",
    "L2_C_anchor":            "L2 · C + grid + anchor",
    "L3_balanced_vxc":        "L3 · B + DM + V_xc · LossNorm-step0",
    "L4_balanced_vxc_anchor": "L4 · L3 + anchor",
    "L5_gradnorm_vxc":        "L5 · B + DM + V_xc · GradNorm",
}
LOSS_FAMILY = {
    "L1_B":                   "no-Vxc",
    "L2_C_anchor":            "no-Vxc",
    "L3_balanced_vxc":        "Vxc-LossNorm",
    "L4_balanced_vxc_anchor": "Vxc-LossNorm",
    "L5_gradnorm_vxc":        "Vxc-GradNorm",
}
LOSS_FAMILY_COLOR = {
    "no-Vxc":         "#1f77b4",  # blue
    "Vxc-LossNorm":   "#d62728",  # red
    "Vxc-GradNorm":   "#2ca02c",  # green
}
ARCH_DISPLAY_ORDER = ("deep_combined", "deep_combined_attn")
SOLVER_DISPLAY_ORDER = ("oneshot", "fixed_j_3", "full_3")
GROUP_DISPLAY_ORDER = ("group1", "group2", "group3")
GROUP_DISPLAY_LABELS = {
    "group1": "G1 · H₂O · 100 steps",
    "group2": "G2 · H₂O+C₂H₂ · 100 steps",
    "group3": "G3 · H₂O+C₂H₂ · 250 steps",
}


# ---------------------------------------------------------------------------
# Data-loading helpers
# ---------------------------------------------------------------------------

def load_run_artifacts(run_dir: Path) -> dict:
    """Load all step-6 evaluation dataframes from one pretrain-loss-weighting run.

    Parameters
    ----------
    run_dir : Path
        e.g. ``notebooks/checkpoints_step6/unweighted/``.

    Returns
    -------
    dict with keys ``eval_df``, ``baseline_df``, ``transfer_primary_df``,
    ``transfer_secondary_df`` (each a pandas DataFrame; ``None`` when the
    file is absent so callers can skip downstream sections cleanly).
    """
    run_dir = Path(run_dir)

    def _maybe_read(name: str):
        for ext in (".parquet", ".csv"):
            p = run_dir / f"{name}{ext}"
            if p.is_file():
                return pd.read_parquet(p) if ext == ".parquet" else pd.read_csv(p)
        return None

    return {
        "eval_df":              _maybe_read("eval_df"),
        "baseline_df":          _maybe_read("baseline_df"),
        "transfer_primary_df":  _maybe_read("transfer_primary_df"),
        "transfer_secondary_df":_maybe_read("transfer_secondary_df"),
    }


def trained_molecules_only(df: pd.DataFrame, mol_set=("H2O", "C2H2")) -> pd.DataFrame:
    """Restrict to atomization-energy rows for trained molecules.

    Atom rows (H, O, C) are present in ``eval_df`` because TrainingSpec
    requires per-molecule entries for every atom in the molecules tuple,
    but atom AE is undefined (sum_atoms - E_mol with mol == one atom is
    a degenerate identity). This helper drops those.
    """
    return df[df["molecule"].isin(mol_set)].copy()


def mae_of(df: pd.DataFrame, value_name: str, *, group_keys: list[str]) -> pd.DataFrame:
    """Mean absolute |value| of a metric, grouped on ``group_keys``.

    Implements the standard MAE convention from Karton et al. 2011 §III:
    MAE = mean over the test-set molecules of |predicted - reference|.
    """
    sub = df[df["value_name"] == value_name].copy()
    sub["abs"] = sub["value"].abs()
    return sub.groupby(group_keys)["abs"].mean().reset_index().rename(columns={"abs": "mae"})


# ---------------------------------------------------------------------------
# Plot 1 — multi-decade baseline reduction (log-y bar chart)
# ---------------------------------------------------------------------------

def plot_baseline_reduction(art: dict, out_path: Path, run_label: str = "unweighted") -> None:
    """Bar chart on log-y showing mean |AE error| (kcal/mol) per (loss, group),
    with horizontal reference lines for random NN, pretrained-only NN, and
    PBE vs W4-11 baselines.

    Why this plot
    -------------
    The single most important finding of any DFT-functional training run is
    "how many orders of magnitude does fine-tuning buy you over the
    starting point?". Random init for these networks gives AE errors of
    several hundred kcal/mol; PBE itself sits at ~7-8 kcal/mol on
    H₂O/C₂H₂; the W4-11 chemical-accuracy threshold is 1 kcal/mol.
    Plotting all three reference levels alongside the trained-model error
    on a log axis exposes whether the training reached chemical accuracy
    AND whether it pushed past PBE.

    References
    ----------
    - Karton et al. *CPL* 510, 165 (2011): W4-11 reference set.
    - Pople *RMP* 71, 1267 (1999): chemical-accuracy = 1 kcal/mol.
    - Perdew, Burke, Ernzerhof *PRL* 77, 3865 (1996): PBE GGA.
    """
    eval_df = art["eval_df"]
    baseline_df = art["baseline_df"]
    if eval_df is None:
        return

    ae = mae_of(
        trained_molecules_only(eval_df), "AE_error_kcalmol",
        group_keys=["group", "loss", "arch", "solver"],
    )
    ae_pbe_vs_w411 = float(np.abs(eval_df.loc[
        (eval_df.value_name == "AE_error_pbe_kcalmol") &
        (eval_df.molecule.isin(["H2O", "C2H2"])), "value"
    ].dropna()).mean())

    rand_b = baseline_df[(baseline_df.baseline == "random") &
                          (baseline_df.value_name == "AE_error_kcalmol")]["value"].abs().mean()
    pre_b  = baseline_df[(baseline_df.baseline == "pretrained") &
                          (baseline_df.value_name == "AE_error_kcalmol")]["value"].abs().mean()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6), sharey=True)
    losses = list(LOSS_DISPLAY_ORDER)
    # Bar layout: per-group panel; within panel, x=loss; per-loss, 6 bars
    # = 2 archs * 3 solvers (3 hatches per arch).
    width = 0.13
    arch_offsets = {"deep_combined": -0.18, "deep_combined_attn": +0.18}
    solver_marks = {"oneshot": "o", "fixed_j_3": "s", "full_3": "D"}
    arch_colors = {"deep_combined": "#1f77b4", "deep_combined_attn": "#ff7f0e"}

    for ax, group in zip(axes, GROUP_DISPLAY_ORDER):
        sub = ae[ae.group == group]
        for ai, arch in enumerate(ARCH_DISPLAY_ORDER):
            for si, solver in enumerate(SOLVER_DISPLAY_ORDER):
                row = sub[(sub.arch == arch) & (sub.solver == solver)]
                if row.empty:
                    continue
                xs = np.array([losses.index(l) for l in row["loss"]])
                xs = xs + arch_offsets[arch] + (si - 1) * width
                ax.bar(
                    xs, row["mae"].values, width=width,
                    color=arch_colors[arch],
                    edgecolor="k", linewidth=0.4,
                    alpha=0.55 + 0.18 * si,
                    label=f"{arch} · {solver}" if group == "group1" else None,
                )
        # Reference horizontal lines.
        ax.axhline(rand_b,           ls=":",  color="grey",   lw=1.4,
                   label=f"random NN ({rand_b:.0f})"      if group == "group1" else None)
        ax.axhline(pre_b,            ls="--", color="grey",   lw=1.4,
                   label=f"pretrained ({pre_b:.0f})"       if group == "group1" else None)
        ax.axhline(ae_pbe_vs_w411,   ls="-",  color="black",  lw=1.5,
                   label=f"PBE vs W4-11 ({ae_pbe_vs_w411:.1f})" if group == "group1" else None)
        ax.axhline(CHEMICAL_ACCURACY_KCALMOL, ls="-.", color="purple", lw=1.6,
                   label=f"chemical accuracy ({CHEMICAL_ACCURACY_KCALMOL})" if group == "group1" else None)

        ax.set_yscale("log")
        ax.set_xticks(range(len(losses)))
        ax.set_xticklabels([LOSS_DISPLAY_LABELS[l] for l in losses],
                           rotation=22, ha="right", fontsize=7)
        ax.set_title(GROUP_DISPLAY_LABELS[group], fontsize=10)
        ax.grid(True, axis="y", which="both", ls=":", alpha=0.35)

    axes[0].set_ylabel("mean |AE error|  vs W4-11 (kcal/mol, log)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5,
               fontsize=7, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(
        f"Baseline reduction — {run_label} pretrain-origin · trained mols (H₂O+C₂H₂)\n"
        "log-y bars vs reference levels: random NN > pretrained > PBE > chemical accuracy",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.95))
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 1b — multi-decade baseline reduction on TRANSFER sets
# ---------------------------------------------------------------------------

def plot_baseline_reduction_transfer(art: dict, out_path: Path, run_label: str = "unweighted") -> None:
    """Same layout as ``plot_baseline_reduction`` but evaluated on the
    held-out transfer molecules:

      - top row:   primary   = {CH₄, H₂, OH}        (W4-11 light hydrides)
      - bottom row: secondary = {CO₂, HF, NH₂, NH₃} (W4-11 mixed)

    Each row has 3 panels (one per training-data group), bars are the
    trained NN's MAE on the held-out set, and the horizontal references
    are the PBE-vs-W4-11 MAE computed on the same transfer set (so the
    "did training beat PBE on this molecule set?" comparison is fair).

    Why a separate plot
    -------------------
    The training-set version (``plot_baseline_reduction``) tells you
    whether the model fit the data it was trained on; the transfer-set
    version is the only way to tell whether the trained XC functional
    is a useful general-purpose functional. Behler & Parrinello (2007)
    §II make exactly this distinction for NN potentials. PBE itself
    sits at ~3.4 kcal/mol on the primary set and ~11.0 kcal/mol on the
    secondary set; a successfully transferable model must beat those
    horizontal lines.

    References
    ----------
    - Karton, Daon, Martin, *CPL* **510**, 165 (2011): primary set
      {CH₄, H₂, OH} and secondary set {CO₂, HF, NH₂, NH₃} are W4-11
      subsets.
    - Behler, Parrinello, *PRL* **98**, 146401 (2007) §II: in-distribution
      vs holdout MAE distinction for neural-network XC / potentials.
    - Pople, *RMP* **71**, 1267 (1999): chemical accuracy = 1 kcal/mol.
    - Perdew, Burke, Ernzerhof, *PRL* **77**, 3865 (1996): PBE GGA.
    """
    t1 = art["transfer_primary_df"]
    t2 = art["transfer_secondary_df"]
    if t1 is None and t2 is None:
        return

    sets = []
    if t1 is not None:
        sets.append(("primary {CH₄, H₂, OH}", t1))
    if t2 is not None:
        sets.append(("secondary {CO₂, HF, NH₂, NH₃}", t2))

    fig, axes = plt.subplots(
        len(sets), 3,
        figsize=(16, 4.6 * len(sets)),
        sharey="row",
        squeeze=False,
    )

    losses = list(LOSS_DISPLAY_ORDER)
    width = 0.13
    arch_offsets = {"deep_combined": -0.18, "deep_combined_attn": +0.18}
    arch_colors = {"deep_combined": "#1f77b4", "deep_combined_attn": "#ff7f0e"}

    for ri, (set_name, df_set) in enumerate(sets):
        # PBE reference MAE on this transfer set (over molecules):
        ae_pbe = float(np.abs(df_set.loc[
            df_set.value_name == "AE_error_pbe_kcalmol", "value"
        ]).mean())

        # Per-spec MAE (over the transfer set's molecules):
        spec_mae = mae_of(df_set, "AE_error_kcalmol",
                          group_keys=["group", "arch", "loss", "solver"])

        for ci, group in enumerate(GROUP_DISPLAY_ORDER):
            ax = axes[ri][ci]
            sub = spec_mae[spec_mae.group == group]
            for ai, arch in enumerate(ARCH_DISPLAY_ORDER):
                for si, solver in enumerate(SOLVER_DISPLAY_ORDER):
                    row = sub[(sub.arch == arch) & (sub.solver == solver)]
                    if row.empty:
                        continue
                    xs = np.array([losses.index(l) for l in row["loss"]])
                    xs = xs + arch_offsets[arch] + (si - 1) * width
                    ax.bar(
                        xs, row["mae"].values, width=width,
                        color=arch_colors[arch],
                        edgecolor="k", linewidth=0.4,
                        alpha=0.55 + 0.18 * si,
                        label=(f"{arch} · {solver}"
                               if ri == 0 and ci == 0 else None),
                    )
            ax.axhline(
                ae_pbe, ls="-", color="black", lw=1.5,
                label=(f"PBE vs W4-11 ({ae_pbe:.2f})"
                       if ri == 0 and ci == 0 else None),
            )
            ax.axhline(
                CHEMICAL_ACCURACY_KCALMOL, ls="-.", color="purple", lw=1.6,
                label=(f"chem. acc. ({CHEMICAL_ACCURACY_KCALMOL})"
                       if ri == 0 and ci == 0 else None),
            )
            ax.set_yscale("log")
            ax.set_xticks(range(len(losses)))
            ax.set_xticklabels([LOSS_DISPLAY_LABELS[l] for l in losses],
                               rotation=22, ha="right", fontsize=7)
            if ri == 0:
                ax.set_title(GROUP_DISPLAY_LABELS[group], fontsize=10)
            if ci == 0:
                ax.set_ylabel(
                    f"MAE on {set_name}\n(kcal/mol, log)",
                    fontsize=9,
                )
            ax.grid(True, axis="y", which="both", ls=":", alpha=0.35)

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5,
               fontsize=7, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(
        f"Baseline reduction (transfer sets) — {run_label} pretrain-origin\n"
        "log-y bars: trained NN MAE on held-out molecules; horizontal lines = PBE on the same set\n"
        "primary {CH₄, H₂, OH} · secondary {CO₂, HF, NH₂, NH₃}  (W4-11 subsets)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.94))
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 2 — density vs energy tradeoff scatter (Medvedev et al. 2017)
# ---------------------------------------------------------------------------

def plot_density_vs_energy_tradeoff(art: dict, out_path: Path, run_label: str = "unweighted") -> None:
    """Scatter: x=AE-MAE on trained mols; y=density-RMSE on same mols.
    Color = loss family (no-Vxc / Vxc-LossNorm / Vxc-GradNorm); marker = arch.
    Annotated chemical-accuracy + PBE-density-error reference lines.

    Why this plot
    -------------
    Medvedev et al. (Science 2017) showed that empirical XC functionals
    trained against energy benchmarks give BETTER energies but WORSE
    densities than physics-grounded functionals like PBE / SCAN. The
    same dichotomy MUST appear in our run if our V_xc-aware losses
    (L3, L4) are doing what we designed them to do: they should sit at
    LOWER density-RMSE but HIGHER AE-MAE than the energy-only losses
    (L1, L2). This plot is the direct test.

    References
    ----------
    - Medvedev, Bushmarinov, Sun, Perdew, Lyssenko, *Science* **355**,
      49 (2017): the canonical density-vs-energy tradeoff paper.
    """
    eval_df = art["eval_df"]
    if eval_df is None:
        return

    ae  = mae_of(trained_molecules_only(eval_df), "AE_error_kcalmol",
                 group_keys=["group","arch","loss","solver"])
    rho = mae_of(trained_molecules_only(eval_df), "density_rmse",
                 group_keys=["group","arch","loss","solver"]).rename(columns={"mae":"rmse"})
    merged = ae.merge(rho, on=["group","arch","loss","solver"])

    fig, ax = plt.subplots(figsize=(8, 6.2))
    arch_marker = {"deep_combined": "o", "deep_combined_attn": "^"}
    group_alpha = {"group1": 0.4, "group2": 0.65, "group3": 0.95}
    for _, r in merged.iterrows():
        family = LOSS_FAMILY[r.loss]
        ax.scatter(
            r.mae, r.rmse,
            s=64,
            facecolor=LOSS_FAMILY_COLOR[family],
            edgecolor="k",
            marker=arch_marker[r.arch],
            alpha=group_alpha[r.group],
            linewidths=0.6,
        )

    ax.axvline(CHEMICAL_ACCURACY_KCALMOL, ls="-.", color="purple", lw=1.4,
               label=f"chemical accuracy = {CHEMICAL_ACCURACY_KCALMOL} kcal/mol")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("mean |AE error|  (kcal/mol, log)")
    ax.set_ylabel("density-RMSE on Becke grid  (e/bohr³, log)")
    ax.set_title(
        f"Density vs energy tradeoff — {run_label} pretrain-origin\n"
        "after Medvedev et al. *Science* 355, 49 (2017): V_xc-aware losses (red/green)\n"
        "should give lower density error than energy-only losses (blue) at higher AE error",
        fontsize=10,
    )
    ax.grid(True, which="both", ls=":", alpha=0.35)

    # Family + arch + group legend.
    legend_elements = []
    from matplotlib.lines import Line2D
    for family, color in LOSS_FAMILY_COLOR.items():
        legend_elements.append(Line2D([0],[0], marker="s", color="w",
            markerfacecolor=color, markeredgecolor="k", markersize=10, label=family))
    for arch, marker in arch_marker.items():
        legend_elements.append(Line2D([0],[0], marker=marker, color="w",
            markerfacecolor="lightgrey", markeredgecolor="k", markersize=10, label=arch))
    for group, alpha in group_alpha.items():
        legend_elements.append(Line2D([0],[0], marker="o", color="w",
            markerfacecolor="lightgrey", markeredgecolor="k", markersize=10,
            alpha=alpha, label=GROUP_DISPLAY_LABELS[group]))
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 3 — in-distribution vs transfer generalization
# ---------------------------------------------------------------------------

def plot_in_dist_vs_transfer(art: dict, out_path: Path, run_label: str = "unweighted") -> None:
    """Scatter: x=AE-MAE on training mols; y=AE-MAE on holdout-set mols.
    The y=x line is the "no generalization gap" reference. Points above
    the line have higher transfer error than training error -- the
    expected direction of overfitting.

    Why this plot
    -------------
    A trained DFT functional that fits its training molecules tightly
    but gives high holdout error is overfit -- the underlying lesson
    from Behler & Parrinello (2007) on neural-network potentials.
    Plotting (train, transfer) per spec makes the generalization gap
    visible at a glance: the perpendicular distance from y=x is the
    overfit magnitude in MAE-space.

    References
    ----------
    - Behler, Parrinello, *PRL* **98**, 146401 (2007) §II:
      transferability vs in-distribution fit for neural-network XC /
      potential energy surfaces.
    - Karton, Daon, Martin, *CPL* **510**, 165 (2011): primary
      transfer set used here is the W4-11 subset {CH4, H2, OH}; the
      secondary set is {CO2, HF, NH2, NH3} (also W4-11).
    """
    eval_df = art["eval_df"]
    t1 = art["transfer_primary_df"]
    t2 = art["transfer_secondary_df"]
    if eval_df is None or (t1 is None and t2 is None):
        return

    train_mae = mae_of(
        trained_molecules_only(eval_df), "AE_error_kcalmol",
        group_keys=["group","arch","loss","solver"],
    ).rename(columns={"mae": "train_mae"})

    pieces = []
    if t1 is not None:
        m = mae_of(t1, "AE_error_kcalmol", group_keys=["group","arch","loss","solver"])
        m["set"] = "primary {CH4, H2, OH}"
        pieces.append(m)
    if t2 is not None:
        m = mae_of(t2, "AE_error_kcalmol", group_keys=["group","arch","loss","solver"])
        m["set"] = "secondary {CO2, HF, NH2, NH3}"
        pieces.append(m)
    transfer = pd.concat(pieces, ignore_index=True).rename(columns={"mae": "transfer_mae"})
    merged = train_mae.merge(transfer, on=["group","arch","loss","solver"])

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=True)
    set_to_ax = {
        "primary {CH4, H2, OH}":           axes[0],
        "secondary {CO2, HF, NH2, NH3}":   axes[1],
    }
    arch_marker = {"deep_combined": "o", "deep_combined_attn": "^"}
    group_alpha = {"group1": 0.4, "group2": 0.65, "group3": 0.95}

    for set_name, ax in set_to_ax.items():
        sub = merged[merged["set"] == set_name]
        for _, r in sub.iterrows():
            ax.scatter(
                r.train_mae, r.transfer_mae,
                facecolor=LOSS_FAMILY_COLOR[LOSS_FAMILY[r.loss]],
                edgecolor="k",
                marker=arch_marker[r.arch],
                alpha=group_alpha[r.group],
                s=72, linewidths=0.6,
            )
        # y=x reference.
        lo = max(1e-3, sub[["train_mae","transfer_mae"]].min().min() * 0.5)
        hi = max(sub[["train_mae","transfer_mae"]].max().max() * 2.0, 100)
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.0, alpha=0.5,
                label="train = transfer (no gap)")
        ax.axvline(CHEMICAL_ACCURACY_KCALMOL, ls=":", color="purple", lw=1.0)
        ax.axhline(CHEMICAL_ACCURACY_KCALMOL, ls=":", color="purple", lw=1.0,
                   label=f"chem. acc. = {CHEMICAL_ACCURACY_KCALMOL} kcal/mol")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("train MAE  H₂O+C₂H₂  (kcal/mol)")
        ax.set_title(f"transfer = {set_name}", fontsize=10)
        ax.grid(True, which="both", ls=":", alpha=0.35)
        ax.legend(loc="lower right", fontsize=7, framealpha=0.9)
    axes[0].set_ylabel("transfer MAE  (kcal/mol, log)")

    fig.suptitle(
        f"In-distribution vs transfer generalization — {run_label} pretrain-origin\n"
        "after Behler & Parrinello *PRL* 98, 146401 (2007): perpendicular distance from y=x is the overfit magnitude",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 4 — full-landscape AE-MAE heatmap (loss × group×solver)
# ---------------------------------------------------------------------------

def plot_loss_strategy_heatmap(art: dict, out_path: Path, run_label: str = "unweighted") -> None:
    """Single-pane heatmap, rows = loss, cols = (group, arch, solver).
    Color = log-MAE on trained mols. Designed as a "whole landscape at
    a glance" view that complements the per-group bar chart.
    """
    eval_df = art["eval_df"]
    if eval_df is None:
        return
    ae = mae_of(trained_molecules_only(eval_df), "AE_error_kcalmol",
                group_keys=["group","arch","loss","solver"])
    ae["col"] = ae["group"] + "·" + ae["arch"].str.replace("deep_combined","dc") + "·" + ae["solver"]
    pv = ae.pivot(index="loss", columns="col", values="mae")
    pv = pv.reindex(index=LOSS_DISPLAY_ORDER)
    cols_sorted = sorted(pv.columns)  # group first because group is leftmost token
    pv = pv[cols_sorted]
    log_pv = np.log10(pv.values)

    fig, ax = plt.subplots(figsize=(13, 4.6))
    im = ax.imshow(log_pv, aspect="auto", cmap="RdYlGn_r")
    ax.set_yticks(range(len(LOSS_DISPLAY_ORDER)))
    ax.set_yticklabels([LOSS_DISPLAY_LABELS[l] for l in LOSS_DISPLAY_ORDER], fontsize=8)
    ax.set_xticks(range(len(cols_sorted)))
    ax.set_xticklabels(cols_sorted, rotation=80, fontsize=6)
    cb = fig.colorbar(im, ax=ax, fraction=0.038, pad=0.02)
    cb.set_label("log₁₀ MAE  (kcal/mol)")
    # annotate each cell with the actual MAE value
    for i in range(log_pv.shape[0]):
        for j in range(log_pv.shape[1]):
            v = pv.values[i, j]
            ax.text(j, i, f"{v:.2g}", ha="center", va="center",
                    fontsize=5, color=("black" if log_pv[i,j] < 0.5 else "white"))
    ax.set_title(
        f"AE-MAE landscape — {run_label} pretrain-origin · trained molecules\n"
        f"green = below chemical accuracy (1 kcal/mol); red = above PBE (~7-8 kcal/mol)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Plot 5 — architecture comparison (deep_combined vs deep_combined_attn)
# ---------------------------------------------------------------------------

def plot_arch_comparison(art: dict, out_path: Path, run_label: str = "unweighted") -> None:
    """Per-loss bar chart comparing the two architectures' AE-MAE on:
    (a) trained mols, (b) primary transfer, (c) secondary transfer.
    Reveals where the new self-attention block earns its added capacity.

    The self-attention block (Vaswani et al. 2017, §3.2.1; Xiong et al.
    2020 Pre-LN variant) was rewritten in commit (2026-04-27) from a
    softmax channel-gate to canonical multi-head scaled-dot-product
    attention. This plot is the empirical test of whether the rewrite
    contributes to chemical accuracy.

    References
    ----------
    - Vaswani et al. *NeurIPS* 30 (2017): "Attention is All You Need",
      §3.2.1 (scaled dot-product) + §3.2.2 (multi-head).
    - Xiong et al. *ICML* (2020) eq. 3: Pre-LN transformer architecture.
    """
    eval_df = art["eval_df"]
    t1 = art["transfer_primary_df"]
    t2 = art["transfer_secondary_df"]
    if eval_df is None:
        return

    parts = []
    parts.append(
        mae_of(trained_molecules_only(eval_df), "AE_error_kcalmol",
               group_keys=["group","arch","loss","solver"]).assign(set="train"))
    if t1 is not None:
        parts.append(mae_of(t1, "AE_error_kcalmol",
                            group_keys=["group","arch","loss","solver"]).assign(set="primary"))
    if t2 is not None:
        parts.append(mae_of(t2, "AE_error_kcalmol",
                            group_keys=["group","arch","loss","solver"]).assign(set="secondary"))
    df = pd.concat(parts, ignore_index=True)
    pv = df.groupby(["set","loss","arch"])["mae"].mean().reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    width = 0.35
    losses = list(LOSS_DISPLAY_ORDER)
    for ax, set_name in zip(axes, ("train", "primary", "secondary")):
        sub = pv[pv["set"] == set_name]
        x = np.arange(len(losses))
        for i, arch in enumerate(ARCH_DISPLAY_ORDER):
            vals = []
            for l in losses:
                row = sub[(sub.loss == l) & (sub.arch == arch)]
                vals.append(float(row["mae"].iloc[0]) if not row.empty else np.nan)
            ax.bar(x + (i - 0.5) * width, vals, width=width,
                   label=arch, edgecolor="k", linewidth=0.4)
        ax.axhline(CHEMICAL_ACCURACY_KCALMOL, ls="-.", color="purple", lw=1.2,
                   label="chemical accuracy" if set_name == "train" else None)
        ax.set_yscale("log")
        ax.set_xticks(x)
        ax.set_xticklabels([LOSS_DISPLAY_LABELS[l] for l in losses],
                           rotation=22, ha="right", fontsize=7)
        ax.set_title({
            "train":     "(a) trained mols  (H₂O, C₂H₂)",
            "primary":   "(b) primary transfer  (CH₄, H₂, OH)",
            "secondary": "(c) secondary transfer  (CO₂, HF, NH₂, NH₃)",
        }[set_name], fontsize=10)
        ax.grid(True, axis="y", which="both", ls=":", alpha=0.35)
        if set_name == "train":
            ax.legend(loc="best", fontsize=8)
    axes[0].set_ylabel("mean |AE error|  (kcal/mol, log) -- mean over groups + solvers")
    fig.suptitle(
        f"Architecture comparison — {run_label} pretrain-origin\n"
        "deep_combined (no attn) vs deep_combined_attn (Vaswani 2017 SDPA + Xiong 2020 Pre-LN)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Summary statistics for the markdown report
# ---------------------------------------------------------------------------

def headline_stats(art: dict) -> dict:
    """Compute report-headline statistics from one run's artifacts.

    Returns a dict with keys ready for f-string substitution into the
    markdown report template.
    """
    eval_df = art["eval_df"]
    baseline_df = art["baseline_df"]
    if eval_df is None:
        return {}

    ae = mae_of(trained_molecules_only(eval_df), "AE_error_kcalmol",
                group_keys=["group","arch","loss","solver"])
    rho = mae_of(trained_molecules_only(eval_df), "density_rmse",
                 group_keys=["group","arch","loss","solver"]).rename(columns={"mae":"rmse"})

    best_ae = ae.loc[ae["mae"].idxmin()]
    best_rho = rho.loc[rho["rmse"].idxmin()]

    out = {
        "n_specs":                int(len(ae)),
        "best_ae_mae":            float(best_ae["mae"]),
        "best_ae_spec":           f"{best_ae.group}/{best_ae.arch}/{best_ae.loss}/{best_ae.solver}",
        "best_rmse":              float(best_rho["rmse"]),
        "best_rmse_spec":         f"{best_rho.group}/{best_rho.arch}/{best_rho.loss}/{best_rho.solver}",
    }
    if baseline_df is not None:
        out["random_ae"]      = float(baseline_df[(baseline_df.baseline=="random") &
                                                   (baseline_df.value_name=="AE_error_kcalmol")
                                                   ]["value"].abs().mean())
        out["pretrained_ae"]  = float(baseline_df[(baseline_df.baseline=="pretrained") &
                                                   (baseline_df.value_name=="AE_error_kcalmol")
                                                   ]["value"].abs().mean())
        out["pbe_ae"]         = float(np.abs(eval_df.loc[
            (eval_df.value_name=="AE_error_pbe_kcalmol") &
            (eval_df.molecule.isin(["H2O","C2H2"])), "value"]).mean())
        out["random_to_best_x"]      = out["random_ae"]      / out["best_ae_mae"]
        out["pretrained_to_best_x"]  = out["pretrained_ae"]  / out["best_ae_mae"]
        out["pbe_to_best_x"]         = out["pbe_ae"]         / out["best_ae_mae"]

    # Per-loss MAE on G3 (the longest-trained group; most informative)
    g3 = ae[ae.group == "group3"].groupby("loss")["mae"].mean().to_dict()
    out["g3_mae_by_loss"] = {l: float(v) for l, v in g3.items()}

    return out
