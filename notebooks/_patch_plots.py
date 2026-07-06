"""Patch cells 28, 33, 41, and 46 in step5 notebook to fix baselines in plots."""
import json

NB_PATH = "gga_training_example-step5.ipynb"

with open(NB_PATH) as f:
    nb = json.load(f)

# ── Cell 28 (id=5f70bc63): Fix pretrain_dst path (baseline_pretrain → baseline_pretrained) ──
CELL_28_SOURCE = r'''# Generate pretrained and random baseline model.eqx for all architectures
from xcquinox.alec.networks import create_network_pair

BASELINE_LABELS = []  # populated below

for arch_name in ARCH_NAMES:
    arch = alec.get_architecture(arch_name)

    # --- Pretrained baseline ---
    pretrain_src = f"{CHECKPOINT_BASE}/pretrain/{arch_name}"
    pretrain_dst = f"{CHECKPOINT_BASE}/baseline_pretrained/{arch_name}"
    pretrain_model_path = f"{pretrain_dst}/model.eqx"
    if (os.path.isfile(f"{pretrain_src}/xnet.eqx")
            and not os.path.isfile(pretrain_model_path)):
        os.makedirs(pretrain_dst, exist_ok=True)
        xnet_skel, cnet_skel = create_network_pair(arch, seed=42)
        loaded_xnet = eqx.tree_deserialise_leaves(
            f"{pretrain_src}/xnet.eqx", xnet_skel)
        loaded_cnet = eqx.tree_deserialise_leaves(
            f"{pretrain_src}/cnet.eqx", cnet_skel)
        model = alec.AlecGGAModel.from_arch(
            arch, xnet=loaded_xnet, cnet=loaded_cnet)
        eqx.tree_serialise_leaves(pretrain_model_path, model)

    # --- Random baseline ---
    random_dst = f"{CHECKPOINT_BASE}/baseline_random/{arch_name}"
    random_model_path = f"{random_dst}/model.eqx"
    if not os.path.isfile(random_model_path):
        os.makedirs(random_dst, exist_ok=True)
        model = alec.AlecGGAModel.from_arch(arch, seed=42)
        eqx.tree_serialise_leaves(random_model_path, model)

BASELINE_LABELS = ['pretrained', 'random']
baseline_colors = {'pretrained': '#888888', 'random': '#CCCCCC'}
print(f"Baselines ready for {len(ARCH_NAMES)} architectures: {BASELINE_LABELS}")'''

# ── Cell 33 (id=cell_26): H2O AE error bar chart ──
CELL_33_SOURCE = r'''# Reference lines: PBE and CCSD atomization energy errors vs experiment
ext_data_dir = f"{CHECKPOINT_BASE}/external_data"
_E_ref = {}
for _name in ("H", "O", "H2O"):
    with open(f"{ext_data_dir}/{_name}_metadata.json") as _f:
        _E_ref[_name] = json.load(_f)
_AE_expt_kcalmol = 233.016

_ae_pbe_Ha = 2 * _E_ref["H"]["E_pbe_total"] + _E_ref["O"]["E_pbe_total"] - _E_ref["H2O"]["E_pbe_total"]
PBE_AE_err_kcalmol = abs(_ae_pbe_Ha * 627.509 - _AE_expt_kcalmol)

_ae_ccsd_Ha = 2 * _E_ref["H"]["E_ccsd_total"] + _E_ref["O"]["E_ccsd_total"] - _E_ref["H2O"]["E_ccsd_total"]
CCSD_AE_err_kcalmol = abs(_ae_ccsd_Ha * 627.509 - _AE_expt_kcalmol)

# Color maps: solvers, balancing strategies, baselines
BAL_LABELS = [f'bal:{k}' for k in BALANCING_CONFIGS]
cmap_all = plt.get_cmap('tab10')
all_colors = {}
for i, sl in enumerate(SOLVER_LABELS):
    all_colors[sl] = solver_colors[sl]
for i, bl in enumerate(BAL_LABELS):
    all_colors[bl] = cmap_all(3 + i)
for bl in BASELINE_LABELS:
    all_colors[bl] = baseline_colors[bl]

fig, axes = plt.subplots(1, len(LOSS_NAMES), figsize=(6 * len(LOSS_NAMES), 7), squeeze=False)
for col_idx, loss_name in enumerate(LOSS_NAMES):
    ax = axes[0, col_idx]
    has_bal = loss_name in BAL_LOSS_NAMES
    _seen_labels = set()

    for arch_idx, arch_name in enumerate(ARCH_NAMES):
        # Treatments: solvers + (balancing if applicable) + baselines always
        labels_here = list(SOLVER_LABELS)
        if has_bal and arch_name == BAL_ARCH:
            labels_here += BAL_LABELS
        labels_here += BASELINE_LABELS
        n_bars = len(labels_here)
        bar_width = 0.8 / max(n_bars, 1)

        for s_idx, sl in enumerate(labels_here):
            # Baselines use loss="baseline" in the DataFrame
            if sl in BASELINE_LABELS:
                df_key = (arch_name, "baseline", sl)
            else:
                df_key = (arch_name, loss_name, sl)
            try:
                val = df.loc[df_key, "AE_error_kcalmol_mean"]
                h = abs(val)
            except KeyError:
                continue
            if not (np.isfinite(h) and h > 0):
                continue
            offset = (s_idx - (n_bars - 1) / 2) * bar_width
            _lbl = sl if sl not in _seen_labels else ''
            _seen_labels.add(sl)
            ax.bar(arch_idx + offset, h, width=bar_width,
                   color=all_colors.get(sl, 'gray'), label=_lbl)

    ax.set_xticks(range(len(ARCH_NAMES)))
    ax.set_xticklabels(ARCH_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("|AE error| (kcal/mol)")
    if ax.patches:
        ax.set_yscale("log")
    ax.set_title(f"Loss: {loss_name}", fontsize=11)
    ax.grid(True, which="both", axis="y", ls=":", alpha=0.4)

    ax.axhline(PBE_AE_err_kcalmol, linestyle=":", color="r", linewidth=1.5,
               label=f"PBE ({PBE_AE_err_kcalmol:.2f} kcal/mol)" if col_idx == len(LOSS_NAMES)-1 else "")
    ax.axhline(CCSD_AE_err_kcalmol, linestyle=":", color="b", linewidth=1.5,
               label=f"CCSD ({CCSD_AE_err_kcalmol:.2f} kcal/mol)" if col_idx == len(LOSS_NAMES)-1 else "")
    ax.axhline(1.0, linestyle="--", color="k", alpha=0.7,
               label="Chemical accuracy (1 kcal/mol)" if col_idx == len(LOSS_NAMES)-1 else "")

# Collect deduplicated legend
all_handles, all_labels_list = [], []
for ax in axes.flat:
    h, l = ax.get_legend_handles_labels()
    all_handles.extend(h)
    all_labels_list.extend(l)
by_label = dict(zip(all_labels_list, all_handles))
by_label = {k: v for k, v in by_label.items() if k}
axes[0, -1].legend(
    by_label.values(), by_label.keys(),
    loc="center left", bbox_to_anchor=(1.02, 0.5),
    fontsize="small", title="Treatment",
)

fig.suptitle(
    "H2O atomization-energy error by architecture\n"
    "(grouped by arch, colored by treatment -- includes pretrained & random baselines)",
    fontsize=13,
)
fig.tight_layout(rect=(0, 0, 1, 0.95))
os.makedirs(f"{CHECKPOINT_BASE}/figures", exist_ok=True)
fig.savefig(f"{CHECKPOINT_BASE}/figures/scf_comparison_ae.png", dpi=150, bbox_inches="tight")
plt.show()'''

# ── Cell 41 (id=cell_34): Feature impact chart -- fix ALL_SOLVER_LABELS ──
CELL_41_SOURCE = r'''_feature_archs = ["deep", "deep_cusp", "deep_dm", "deep_combined"]
_feature_archs = [a for a in _feature_archs if a in ARCH_NAMES]

if not _feature_archs:
    print("[Feature impact] no non-attention deep variants -- skipping")
else:
    fig, axes = plt.subplots(1, len(LOSS_NAMES), figsize=(6 * len(LOSS_NAMES), 6),
                             squeeze=False)

    for col_idx, loss_name in enumerate(LOSS_NAMES):
        ax = axes[0, col_idx]
        has_bal = loss_name in BAL_LOSS_NAMES
        _seen = set()

        for arch_idx, arch_name in enumerate(_feature_archs):
            # Treatments: solvers + (balancing if applicable) + baselines always
            labels_here = list(SOLVER_LABELS)
            if has_bal and arch_name == BAL_ARCH:
                labels_here += BAL_LABELS
            labels_here += BASELINE_LABELS
            n_bars = len(labels_here)
            bar_width = 0.8 / max(n_bars, 1)

            for s_idx, sl in enumerate(labels_here):
                if sl in BASELINE_LABELS:
                    df_key = (arch_name, "baseline", sl)
                else:
                    df_key = (arch_name, loss_name, sl)
                try:
                    val = df.loc[df_key, "AE_error_kcalmol_mean"]
                    h = abs(val)
                except KeyError:
                    continue
                if not (np.isfinite(h) and h > 0):
                    continue
                offset = (s_idx - (n_bars - 1) / 2) * bar_width
                _lbl = sl if sl not in _seen else ''
                _seen.add(sl)
                ax.bar(arch_idx + offset, h, width=bar_width,
                       color=all_colors.get(sl, 'gray'), label=_lbl)

        ax.set_xticks(range(len(_feature_archs)))
        ax.set_xticklabels(_feature_archs, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("|AE error| (kcal/mol)")
        if ax.patches:
            ax.set_yscale("log")
        ax.set_title(f"Loss: {loss_name}", fontsize=11)
        ax.grid(True, which="both", axis="y", ls=":", alpha=0.4)
        ax.axhline(PBE_AE_err_kcalmol, linestyle=":", color="r", linewidth=1.5,
                   label=f"PBE ({PBE_AE_err_kcalmol:.2f} kcal/mol)" if col_idx == len(LOSS_NAMES)-1 else "")
        ax.axhline(CCSD_AE_err_kcalmol, linestyle=":", color="b", linewidth=1.5,
                   label=f"CCSD ({CCSD_AE_err_kcalmol:.2f} kcal/mol)" if col_idx == len(LOSS_NAMES)-1 else "")
        ax.axhline(1.0, linestyle="--", color="k", alpha=0.7,
                   label="Chemical accuracy (1 kcal/mol)" if col_idx == len(LOSS_NAMES)-1 else "")

    all_handles, all_labels_list = [], []
    for ax in axes.flat:
        h, l = ax.get_legend_handles_labels()
        all_handles.extend(h)
        all_labels_list.extend(l)
    by_label = dict(zip(all_labels_list, all_handles))
    by_label = {k: v for k, v in by_label.items() if k}
    axes[0, -1].legend(
        by_label.values(), by_label.keys(),
        loc="center left", bbox_to_anchor=(1.02, 0.5),
        fontsize="small", title="solver / balancing / reference",
    )

    fig.suptitle(
        "Feature impact: non-attention deep variants x solver config\n"
        "(descriptor dim increases L-to-R: 2, 4, 5, 7; balancing at deep_combined)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    os.makedirs(f"{CHECKPOINT_BASE}/figures", exist_ok=True)
    fig.savefig(f"{CHECKPOINT_BASE}/figures/feature_impact_scf.png", dpi=150, bbox_inches="tight")
    plt.show()'''

# ── Cell 46 (id=1c427875): Transfer evaluation plots ──
# One figure per molecule: 2 rows (energy, density) x n_losses columns.
# x = architectures, bars = treatments. Large figures with thick visible bars.
CELL_46_SOURCE = r'''# ---- Transfer evaluation plots ----
n_mols = len(transfer_results)
if n_mols == 0:
    print("No transfer results to plot")
else:
    mol_items = list(transfer_results.items())
    _bl_set = set(BASELINE_LABELS)

    # Gather unique treatments across all molecules
    _solvers_set, _bal_set = set(), set()
    for tdf in transfer_results.values():
        for s in tdf['solver'].unique():
            if s in _bl_set:
                continue
            elif s.startswith('bal:'):
                _bal_set.add(s)
            else:
                _solvers_set.add(s)
    _main_slvrs = sorted(_solvers_set)
    _bal_slvrs = sorted(_bal_set)
    _base_slvrs = sorted(_bl_set)

    # Unique non-baseline losses
    _losses = sorted(set(
        l for tdf in transfer_results.values()
        for l in tdf['loss'].unique() if l != 'baseline'
    ))
    n_loss = len(_losses)
    _loss_abbrev = {l: l.split('_')[0] for l in _losses}

    # Treatment colors (strong, distinct)
    _tc = {}
    for sl in _main_slvrs:
        _tc[sl] = solver_colors.get(sl, 'gray')
    for sl in _bal_slvrs:
        _tc[sl] = all_colors.get(sl, 'gray')
    for sl in _base_slvrs:
        _tc[sl] = baseline_colors.get(sl, '#AAAAAA')

    os.makedirs(f"{CHECKPOINT_BASE}/figures", exist_ok=True)

    for mol_name, tdf in mol_items:
        tm = next(t for t in test_molecules if t['name'] == mol_name)
        _is_atom = (len(tm['spec'].atom_composition) == 1
                    and tm['spec'].atom_composition[0][1] == 1)
        e_col = "E_error_kcalmol" if _is_atom else "AE_error_kcalmol"
        e_lbl = "|E error| (kcal/mol)" if _is_atom else "|AE error| (kcal/mol)"

        fig, axes = plt.subplots(
            2, max(n_loss, 1),
            figsize=(6 * max(n_loss, 1), 12),
            squeeze=False,
        )

        for metric_row, (col_name, y_lbl, metric_tag) in enumerate([
            (e_col, e_lbl, "energy"),
            ("density_rmse", "Density RMSE vs HF", "density"),
        ]):
            for col, loss in enumerate(_losses):
                ax = axes[metric_row, col]
                _seen = set()

                for ai, arch in enumerate(ARCH_NAMES):
                    labels_here = list(_main_slvrs)
                    if arch == BAL_ARCH and loss in BAL_LOSS_NAMES:
                        labels_here += _bal_slvrs
                    labels_here += _base_slvrs
                    n_bars = len(labels_here)
                    bw = 0.8 / max(n_bars, 1)

                    for si, sl in enumerate(labels_here):
                        if sl in _bl_set:
                            sub = tdf[(tdf['arch'] == arch)
                                      & (tdf['loss'] == 'baseline')
                                      & (tdf['solver'] == sl)]
                        else:
                            sub = tdf[(tdf['arch'] == arch)
                                      & (tdf['loss'] == loss)
                                      & (tdf['solver'] == sl)]
                        if len(sub) == 0:
                            continue
                        val = sub.iloc[0][col_name]
                        if not (np.isfinite(val) and val > 0):
                            continue
                        off = (si - (n_bars - 1) / 2) * bw
                        lbl = sl if sl not in _seen else ''
                        _seen.add(sl)
                        ax.bar(ai + off, val, width=bw,
                               color=_tc.get(sl, 'gray'), label=lbl,
                               edgecolor='black', linewidth=0.4, alpha=0.9)

                ax.set_xticks(range(len(ARCH_NAMES)))
                ax.set_xticklabels(ARCH_NAMES, rotation=45, ha='right', fontsize=9)
                if ax.patches:
                    ax.set_yscale('log')
                else:
                    ax.text(0.5, 0.5, 'No data', transform=ax.transAxes,
                            ha='center', va='center', fontsize=12, color='gray')
                ax.grid(True, which='major', axis='y', ls='-', alpha=0.3)
                ax.set_axisbelow(True)

                if col == 0:
                    ax.set_ylabel(y_lbl, fontsize=10)
                if metric_row == 0:
                    ax.set_title(f"Loss {_loss_abbrev[loss]}", fontsize=11, fontweight='bold')

                # Reference lines (energy row only)
                _add_lbl = (col == n_loss - 1)
                if metric_tag == "energy":
                    refs = transfer_refs.get(mol_name, {})
                    if not _is_atom:
                        if 'pbe_ae_err' in refs:
                            ax.axhline(refs['pbe_ae_err'], ls=':', color='r', lw=1.5,
                                       label=f"PBE ({refs['pbe_ae_err']:.2f})" if _add_lbl else "")
                        if 'ccsd_ae_err' in refs:
                            ax.axhline(refs['ccsd_ae_err'], ls=':', color='b', lw=1.5,
                                       label=f"CCSD ({refs['ccsd_ae_err']:.2f})" if _add_lbl else "")
                        ax.axhline(1.0, ls='--', color='k', alpha=0.6, lw=1.2,
                                   label="Chem. accuracy (1 kcal/mol)" if _add_lbl else "")
                    else:
                        if 'pbe_E_err' in refs:
                            ax.axhline(refs['pbe_E_err'], ls=':', color='r', lw=1.5,
                                       label=f"PBE ({refs['pbe_E_err']:.1f})" if _add_lbl else "")
                        if 'ccsd_E_err' in refs:
                            ax.axhline(refs['ccsd_E_err'], ls=':', color='b', lw=1.5,
                                       label=f"CCSD ({refs['ccsd_E_err']:.1f})" if _add_lbl else "")

        # Deduplicated legend at bottom, multi-column
        all_h, all_l = [], []
        for ax in axes.flat:
            h, l = ax.get_legend_handles_labels()
            all_h.extend(h)
            all_l.extend(l)
        by_label = {k: v for k, v in dict(zip(all_l, all_h)).items() if k}
        fig.legend(
            by_label.values(), by_label.keys(),
            loc='lower center', bbox_to_anchor=(0.5, -0.02),
            ncol=min(len(by_label), 6), fontsize=9,
            title='Treatment', title_fontsize=10,
            frameon=True, fancybox=True, shadow=False,
        )

        fig.suptitle(
            f"{mol_name}: Transfer evaluation (energy + density)\n"
            "x = architecture, bars = treatment",
            fontsize=14, fontweight='bold',
        )
        fig.tight_layout(rect=(0, 0.06, 1, 0.95))
        fig.savefig(f"{CHECKPOINT_BASE}/figures/transfer_{mol_name}.png",
                    dpi=150, bbox_inches='tight')
        plt.show()
        print(f"  Saved transfer_{mol_name}.png")

    print(f"Transfer plots complete for {n_mols} molecules across {n_loss} losses")'''

# Apply patches
assert nb['cells'][28].get('id') == '5f70bc63', f"Cell 28 id mismatch: {nb['cells'][28].get('id')}"
assert nb['cells'][33].get('id') == 'cell_26', f"Cell 33 id mismatch: {nb['cells'][33].get('id')}"
assert nb['cells'][41].get('id') == 'cell_34', f"Cell 41 id mismatch: {nb['cells'][41].get('id')}"
assert nb['cells'][46].get('id') == '1c427875', f"Cell 46 id mismatch: {nb['cells'][46].get('id')}"

def _set_source(cell, source_str):
    lines = [line + '\n' for line in source_str.split('\n')]
    lines[-1] = lines[-1].rstrip('\n')
    cell['source'] = lines

_set_source(nb['cells'][28], CELL_28_SOURCE)
_set_source(nb['cells'][33], CELL_33_SOURCE)
_set_source(nb['cells'][41], CELL_41_SOURCE)
_set_source(nb['cells'][46], CELL_46_SOURCE)

with open(NB_PATH, 'w') as f:
    json.dump(nb, f, indent=1)

print("Patched cells 28, 33, 41, and 46 successfully")
