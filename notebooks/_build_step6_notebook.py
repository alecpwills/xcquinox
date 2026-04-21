"""Generator for notebooks/gga_training_example-step6.ipynb.

Step 6 tests whether (a) adding C2H2 training data and (b) a PBE-anchor
regularization term close the F_x(s) drift at s > 0.7 that step 5 found.
All geometries + AE refs come from the W4-11 tarball at
/home/awills/Documents/Research/xcdiff/testing/small/W4-11/<name>/struc.xyz.

Spec:  docs/superpowers/specs/2026-04-21-step6-notebook-design.md
Plan:  docs/superpowers/plans/2026-04-21-step6-notebook-implementation.md
"""
import os

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


DEFAULT_ARCH_NAMES = ("deep_combined", "deep_combined_attn")

DEFAULT_LOSS_NAMES = (
    "L1_B",
    "L2_C_anchor",
    "L3_balanced_vxc",
    "L4_balanced_vxc_anchor",
)

DEFAULT_SOLVER_LABELS = ("oneshot", "fixed_j_3", "full_3")

DEFAULT_CHECKPOINT_BASE = "checkpoints_step6"


def build_cell_01_title():
    source = r"""# GGA Network Training -- Step 6: Data Expansion + PBE-Anchor + Overfitting

Tests two hypothesized fixes for the F_x(s) drift at s > 0.7 (step-5 finding
on CH4) plus an overfitting diagnostic.

## Training Matrix: 2 archs x 4 losses x 3 solvers x 3 groups = 72 runs

| Loss | Kind | V_xc? | PBE-anchor? |
|---|---|---|---|
| L1 | B_atomization_plus_dm | -- | -- |
| L2 | C_atomization_plus_grid | -- | yes |
| L3 | balanced + V_xc | yes | -- |
| L4 | balanced + V_xc + anchor | yes | yes |

| Group | Data | Phase length |
|---|---|---|
| 1 | H2O only | 45 steps (short) |
| 2 | H2O + C2H2 | 45 steps (short) |
| 3 | H2O + C2H2 | 125 steps (long) |

Geometries + AE refs: W4-11 (Karton et al. 2011). Atomic refs: Chakravorty 1993.

Spec: docs/superpowers/specs/2026-04-21-step6-notebook-design.md
"""
    return new_markdown_cell(source)


def build_cell_02_imports():
    """Section 1 Cell 2 -- imports + JAX config.

    The JAX ``x64`` and ``jax_default_device`` config calls must sit between
    ``import jax`` and ``import jax.numpy as jnp`` -- flipping them later
    produces dtype and device inconsistencies in cached JIT traces.
    """
    source = (
        "import json\n"
        "import os\n"
        "import sys\n"
        "\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "import pandas as pd\n"
        "\n"
        "import jax\n"
        "# JAX config: pin x64 dtype and CPU device *before* importing jnp or any\n"
        "# library that may trigger JAX tracing. These must not change later in the\n"
        "# notebook -- flipping jax_enable_x64 after traces are cached produces\n"
        "# inconsistent dtypes.\n"
        'jax.config.update("jax_enable_x64", True)\n'
        'jax.config.update("jax_default_device", jax.devices("cpu")[0])\n'
        "# Persistent compilation cache: writes compiled XLA HLO/LLVM to disk so\n"
        "# that kernel restarts (e.g. after a crash) don't re-pay the full compile\n"
        "# cost.\n"
        'os.makedirs(".jax_compilation_cache", exist_ok=True)\n'
        'jax.config.update("jax_compilation_cache_dir", ".jax_compilation_cache")\n'
        'jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)\n'
        'jax.config.update("jax_persistent_cache_min_compile_time_secs", 1.0)\n'
        "import jax.numpy as jnp\n"
        "import equinox as eqx\n"
        "\n"
        "from pyscf import gto, scf, cc, dft\n"
        "\n"
        "import xcquinox.alec as alec\n"
        "import xcquinox.features\n"
        "from xcquinox.alec import (\n"
        "    ARCHITECTURES,\n"
        "    MoleculeSpec,\n"
        "    PretrainSpec, TrainingSpec, TestSpec,\n"
        "    TwoPhaseConfig, LossNormConfig,\n"
        "    build_pbe_anchor_sample, PBEAnchorSample,\n"
        "    run_pretrain, run_training, run_test, run_oep_inversion, save_vxc_ref,\n"
        "    precompute_fixed_density_data,\n"
        ")\n"
        "# SolverConfig + enums are not re-exported on xcquinox.alec; import from submodule.\n"
        "from xcquinox.alec.solver import SolverConfig, SolverMode, FeaturePolicy\n"
        "\n"
        "# tqdm.auto picks tqdm.notebook.tqdm (ipywidgets) under JupyterLab and\n"
        "# tqdm.std.tqdm in a plain script/terminal, so the same symbol gives a\n"
        "# sensible progress bar in either context.\n"
        "from tqdm.auto import tqdm\n"
    )
    return new_code_cell(source)


def build_cell_03_constants(checkpoint_base: str = DEFAULT_CHECKPOINT_BASE):
    source = f"""# Step-6 knobs. All training / eval cells read from these.
CHECKPOINT_BASE          = {checkpoint_base!r}
PRETRAIN_N_STEPS         = 200
PRETRAIN_SKIP_IF_EXISTS  = True
TRAIN_N_STEPS_SHORT      = 45
TRAIN_N_STEPS_LONG       = 125
TRAIN_SKIP_IF_EXISTS     = True
RERUN_EVAL               = False
PBE_ANCHOR_WEIGHT        = 1e-3
PBE_ANCHOR_N_POINTS      = 200
PBE_ANCHOR_SEED          = 20260421
BASIS                    = "def2-tzvp"
GRID_LEVEL               = 3

# Pretrain atoms (atom_symbol, spin) -- supply ground-state (rho, sigma) samples
# spanning H, He, O, N so xnet/cnet see a representative input range before the
# main training loop runs.
PRETRAIN_ATOMS = (("H", 1), ("He", 0), ("O", 2), ("N", 3))

# Pretraining loss weighting. "unweighted" is the default (fits PBE
# enhancement factors pointwise). "integration" weights pointwise MSE by
# |rho * eps^LDA|; theoretically motivated but empirically under-fits the
# high-s / Lieb-Oxford-bound region, so keep "unweighted" unless you know
# you want that trade-off.
PRETRAIN_LOSS_WEIGHTING = "unweighted"

ext_data_dir       = os.path.join(CHECKPOINT_BASE, "external_data")
pretrain_dir       = os.path.join(CHECKPOINT_BASE, "pretrain")
group1_dir         = os.path.join(CHECKPOINT_BASE, "group1_h2o_short")
group2_dir         = os.path.join(CHECKPOINT_BASE, "group2_h2o_c2h2_short")
group3_dir         = os.path.join(CHECKPOINT_BASE, "group3_h2o_c2h2_long")
figures_dir        = os.path.join(CHECKPOINT_BASE, "figures")
transfer_primary   = os.path.join(CHECKPOINT_BASE, "transfer_data", "primary")
transfer_secondary = os.path.join(CHECKPOINT_BASE, "transfer_data", "secondary")
for _d in (ext_data_dir, pretrain_dir, group1_dir, group2_dir, group3_dir,
           figures_dir, transfer_primary, transfer_secondary):
    os.makedirs(_d, exist_ok=True)

print("DATA VERSION: step6-v1")
print("  Training:   {{H2O, C2H2}} + atoms {{H, O, C}}")
print("  Transfer P: {{H2, OH, CH4}} (W4-11)")
print("  Transfer S: {{NH3, HF, CO2, NH2}} (W4-11)")
print(f"  Wipe {{CHECKPOINT_BASE}}/ to regenerate")
"""
    return new_code_cell(source)


def build_cell_04_arch_table(arch_names: tuple[str, ...] | None = None):
    arch_names = arch_names or DEFAULT_ARCH_NAMES
    source = f"""ARCH_NAMES = {tuple(arch_names)!r}

# Per-architecture color palette; downstream plots (pretrain loss curves,
# training curves, parity plots) key into ``arch_colors`` by arch name so each
# architecture is a consistent color across every figure.
import matplotlib.cm as cm
cmap = cm.get_cmap("tab10")
arch_colors = {{name: cmap(i / max(1, len(ARCH_NAMES) - 1)) for i, name in enumerate(ARCH_NAMES)}}

print(f"Architectures ({{len(ARCH_NAMES)}}):")
for _n in ARCH_NAMES:
    _cfg = ARCHITECTURES[_n]
    print(f"  {{_n:30s}} depth={{_cfg.depth}} nodes={{_cfg.nodes}} "
          f"attention={{_cfg.attention}} descriptors={{len(_cfg.descriptors)}}")
"""
    return new_code_cell(source)


def build_cell_05_solver_table(solver_labels: tuple[str, ...] | None = None):
    solver_labels = solver_labels or DEFAULT_SOLVER_LABELS
    source = f"""SOLVER_LABELS = {tuple(solver_labels)!r}
# SolverConfig: mode uses SolverMode enum; feature_policy is FeaturePolicy enum
# or None. ONESHOT requires max_cycles=0; non-oneshot requires max_cycles>0.
SOLVER_CONFIGS = {{
    "oneshot":   SolverConfig(mode=SolverMode.ONESHOT, max_cycles=0),
    "fixed_j_3": SolverConfig(mode=SolverMode.FIXED_J, max_cycles=3),
    "full_3":    SolverConfig(mode=SolverMode.FULL, max_cycles=3,
                              feature_policy=FeaturePolicy.REASSEMBLE),
}}
print("Solver configs:")
for _lbl in SOLVER_LABELS:
    _sc = SOLVER_CONFIGS[_lbl]
    print(f"  {{_lbl:12s}} mode={{_sc.mode.value:8s}} max_cycles={{_sc.max_cycles}} "
          f"feature_policy={{_sc.feature_policy}}")
"""
    return new_code_cell(source)


def build_cell_06_pretrain_md():
    """Section 3 Cell 6 -- pretrain phase narrative (markdown, ported from step5)."""
    source = """## Section 3: Pretraining

Before the main training loop, each network (xnet / cnet) is **pretrained** on
atomic PBE enhancement factors so the weights start near a meaningful baseline
instead of a cold random initialisation. Starting from random weights causes the
main training loss to diverge; pretraining on known-good PBE targets avoids this.

### Pretrain atoms

Four atoms are used: **H** (spin=1), **He** (spin=0), **O** (spin=2), **N**
(spin=3). Their DFT grids cover a wide range of densities and gradient norms,
giving xnet / cnet a representative sample of the `(rho, sigma)` input space.

### Target: PBE enhancement factors

For each atom the PBE exchange and correlation enhancement factors are computed
via `pyscf`'s `eval_xc` with the exact libxc functional strings (`"PBE,"` /
`",PBE"` for GGA, `"LDA_X,"` / `",LDA_C_PW"` for the LDA baseline). The
network targets are `F_x - 1` and `F_c - 1` (shift by 1 so the loss near PBE
is near zero).

### Low-density cutoff and clipping

Grid points with `rho <= 1e-10` are dropped at write time -- below this threshold
the density is numerically zero and the enhancement factor is undefined. The
targets are clipped to `[-5, 5]` to suppress outliers in the atomic core and
tail regions that would otherwise dominate the loss.
"""
    return new_markdown_cell(source)


def build_cell_07_pretrain_data_gen():
    """Section 3 Cell 7 -- pretrain data generation (inline pyscf, ported from step5)."""
    source = """# Pretrain data generation (inline pyscf) -- matches step5 Cell 8.
rho_list, sigma_list, Fx_list, Fc_list = [], [], [], []
cusp_list, dm_list = [], []

# Compute gate: only compute extended features iff ARCH_NAMES contains
# architectures that actually declare the corresponding descriptor.
_arch_objs = [alec.get_architecture(n) for n in ARCH_NAMES]
need_cusp = any(s.name == "cusp" for a in _arch_objs for s in a.descriptors)
need_dm = any(s.name == "dm_statistics" for a in _arch_objs for s in a.descriptors)

for atom_symbol, spin in PRETRAIN_ATOMS:
    mol = gto.M(atom=f"{atom_symbol} 0 0 0", basis=BASIS, charge=0, spin=spin, verbose=0)
    mf = dft.UKS(mol) if spin else dft.RKS(mol)
    mf.xc = "pbe"
    mf.grids.level = GRID_LEVEL
    mf.kernel()

    ao = mf._numint.eval_ao(mol, mf.grids.coords, deriv=1)
    dm_ab = mf.make_rdm1()
    dm_total = dm_ab[0] + dm_ab[1] if dm_ab.ndim == 3 else dm_ab
    rho_gga = mf._numint.eval_rho(mol, ao, dm_total, xctype="GGA", hermi=True)

    rho = rho_gga[0]
    sigma = rho_gga[1]**2 + rho_gga[2]**2 + rho_gga[3]**2

    # PBE enhancement factors from libxc (pyscf functional strings, NOT xcquinox helpers)
    ex_pbe = mf._numint.eval_xc("PBE,", rho_gga, spin=0)[0]
    ec_pbe = mf._numint.eval_xc(",PBE", rho_gga, spin=0)[0]
    # LDA baselines on the 1-D total density
    ex_lda = mf._numint.eval_xc("LDA_X,", rho, spin=0)[0]
    ec_lda = mf._numint.eval_xc(",LDA_C_PW", rho, spin=0)[0]

    # np.where-based safe division (NOT a boolean mask -- boolean masks drop points
    # we want to keep)
    ex_lda_safe = np.where(np.abs(ex_lda) > 1e-12, ex_lda, 1e-12)
    ec_lda_safe = np.where(np.abs(ec_lda) > 1e-12, ec_lda, 1e-12)
    Fx_minus_1 = ex_pbe / ex_lda_safe - 1.0
    Fc_minus_1 = ec_pbe / ec_lda_safe - 1.0

    Fx_minus_1 = np.clip(Fx_minus_1, -5.0, 5.0)
    Fc_minus_1 = np.clip(Fc_minus_1, -5.0, 5.0)

    # Low-density mask at write time -- threshold is 1e-10 (NOT 1e-6),
    # strictly > (NOT >=).
    valid = rho > 1e-10
    rho_write = rho[valid]
    sigma_write = sigma[valid]
    Fx_write = Fx_minus_1[valid]
    Fc_write = Fc_minus_1[valid]

    rho_list.append(rho_write)
    sigma_list.append(sigma_write)
    Fx_list.append(Fx_write)
    Fc_list.append(Fc_write)

    if need_cusp:
        coords_v = mf.grids.coords[valid]
        cusp_feat = xcquinox.features.compute_cusp_descriptor(
            jnp.asarray(coords_v),
            jnp.asarray(mol.atom_coords()),
            jnp.asarray(mol.atom_charges()),
        )
        cusp_list.append(np.asarray(cusp_feat))

    if need_dm:
        S = mol.intor("int1e_ovlp")
        dm_feat_global = xcquinox.features.compute_dm_features_array(
            jnp.asarray(dm_total), jnp.asarray(S)
        )
        dm_feat_tiled = jnp.tile(dm_feat_global, (len(rho_write), 1))
        dm_list.append(np.asarray(dm_feat_tiled))

rho_all   = np.concatenate(rho_list)
sigma_all = np.concatenate(sigma_list)
Fx_all    = np.concatenate(Fx_list)
Fc_all    = np.concatenate(Fc_list)

save_kwargs = dict(rho_all=rho_all, sigma_all=sigma_all, Fx_all=Fx_all, Fc_all=Fc_all)
if cusp_list:
    save_kwargs["cusp_all"] = np.concatenate(cusp_list)
if dm_list:
    save_kwargs["dm_all"] = np.concatenate(dm_list)

os.makedirs(os.path.join(CHECKPOINT_BASE, "pretrain_data"), exist_ok=True)
np.savez(os.path.join(CHECKPOINT_BASE, "pretrain_data", "pretrain_data.npz"), **save_kwargs)
print(f"pretrain_data.npz written with keys: {sorted(save_kwargs.keys())}  total_points={len(rho_all)}")
"""
    return new_code_cell(source)


def build_cell_08_pretrain_loop():
    """Section 3 Cell 8 -- serial pretrain loop over ARCH_NAMES (ported from step5).

    Always qualifies as ``alec.PretrainSpec`` and ``alec.run_pretrain``.
    n_steps reads PRETRAIN_N_STEPS (step-6 default 200, set in cell 3).
    """
    source = """# Per-(arch, phase) tqdm bars keyed by (arch_name, phase_letter).
# The bar for a given phase is created on the first callback for that phase
# and closed when step == total. Scientific-notation postfix ``loss=...``
# keeps small values readable without losing precision.
_bars = {}

def _cb(info):
    key = (info['arch'], info['phase'])
    if key not in _bars:
        _bars[key] = tqdm(
            total=info['total'],
            desc=f"{info['arch']:<20} {info['phase']}net",
            leave=True,
            dynamic_ncols=True,
        )
    bar = _bars[key]
    delta = info['step'] - bar.n
    if delta > 0:
        bar.update(delta)
    bar.set_postfix(loss=f"{info['loss']:.4e}")
    if info['step'] >= info['total']:
        bar.close()
        del _bars[key]

def _pretrain_checkpoints_exist(arch_name):
    import os as _os
    _ckdir = f"{CHECKPOINT_BASE}/pretrain/{arch_name}"
    return (
        _os.path.isfile(f"{_ckdir}/xnet.eqx")
        and _os.path.isfile(f"{_ckdir}/cnet.eqx")
    )

for arch_name in ARCH_NAMES:
    if PRETRAIN_SKIP_IF_EXISTS and _pretrain_checkpoints_exist(arch_name):
        print(f"[{arch_name}] cached xnet.eqx + cnet.eqx found -- skipping pretrain")
        continue
    spec = alec.PretrainSpec(
        arch=alec.get_architecture(arch_name),
        data_dir=f"{CHECKPOINT_BASE}/pretrain_data",
        checkpoint_dir=f"{CHECKPOINT_BASE}/pretrain/{arch_name}",
        n_steps=PRETRAIN_N_STEPS,
        lr_start=1e-2,
        lr_end=1e-5,
        lr_decay_start=0.2,
        grad_clip=1.0,
        loss_weighting=PRETRAIN_LOSS_WEIGHTING,
    )
    alec.run_pretrain(spec, progress_callback=_cb)
"""
    return new_code_cell(source)


def build_cell_09_pretrain_loss_plot():
    """Section 3 Cell 9 -- pretrain loss curves (xnet / cnet) on log-y axes.

    Ported from step5 cell 10 verbatim.
    """
    source = r"""fig, (ax_x, ax_c) = plt.subplots(1, 2, figsize=(12, 4.5))
for arch_name in ARCH_NAMES:
    losses_x = np.load(f"{CHECKPOINT_BASE}/pretrain/{arch_name}/losses_x.npy")
    losses_c = np.load(f"{CHECKPOINT_BASE}/pretrain/{arch_name}/losses_c.npy")
    ax_x.semilogy(losses_x, color=arch_colors[arch_name], label=arch_name)
    ax_c.semilogy(losses_c, color=arch_colors[arch_name], label=arch_name)

ax_x.set_title(r"xnet: target $F_x - 1$ (PBE exchange enhancement)")
ax_x.set_xlabel("optimizer step")
ax_x.set_ylabel("MSE loss (log scale)")
ax_x.grid(True, which="both", ls=":", alpha=0.4)
ax_c.set_title(r"cnet: target $F_c - 1$ (PBE correlation enhancement)")
ax_c.set_xlabel("optimizer step")
ax_c.set_ylabel("MSE loss (log scale)")
ax_c.grid(True, which="both", ls=":", alpha=0.4)
# Legend outside right on the right subplot only (avoids cluttering both)
ax_c.legend(
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    fontsize="small",
    title="architecture",
)

fig.suptitle(
    "Pretraining loss vs step -- one curve per architecture "
    "(atoms: H, He, O, N)",
    fontsize=12,
)
fig.tight_layout(rect=(0, 0, 1, 0.95))
os.makedirs(f"{CHECKPOINT_BASE}/figures", exist_ok=True)
fig.savefig(f"{CHECKPOINT_BASE}/figures/pretrain_losses.png", dpi=150, bbox_inches="tight")
plt.show()
"""
    return new_code_cell(source)


def build_cell_10_data_md():
    return new_markdown_cell(r"""## Section 2 -- Data Layer

All molecular geometries + atomization-energy references from W4-11
(Karton, Daon, Martin, Ruscic 2011). Atomic references from Chakravorty
1993 (exact non-relativistic).
""")


def build_cell_11_chakravorty():
    source = r"""# Chakravorty 1993 exact non-relativistic atomic energies (Ha).
ATOMIC_ENERGIES_CHAKRAVORTY = {
    "H": -0.5,
    "C": -37.845,
    "N": -54.5892,
    "O": -75.0673,
    "F": -99.7339,
}
for _z, _e in ATOMIC_ENERGIES_CHAKRAVORTY.items():
    print(f"  E({_z}) = {_e:+10.4f} Ha")
"""
    return new_code_cell(source)


def main(
    arch_names: tuple[str, ...] | None = None,
    loss_names: tuple[str, ...] | None = None,
    solver_labels: tuple[str, ...] | None = None,
    checkpoint_base: str | None = None,
    output_path: str = "notebooks/gga_training_example-step6.ipynb",
) -> nbformat.NotebookNode:
    """Assemble the step-6 notebook."""
    arch_names = arch_names or DEFAULT_ARCH_NAMES
    loss_names = loss_names or DEFAULT_LOSS_NAMES
    solver_labels = solver_labels or DEFAULT_SOLVER_LABELS
    checkpoint_base = checkpoint_base or DEFAULT_CHECKPOINT_BASE

    nb = new_notebook()
    cells = [
        build_cell_01_title(),
        build_cell_02_imports(),
        build_cell_03_constants(checkpoint_base=checkpoint_base),
        build_cell_04_arch_table(arch_names=arch_names),
        build_cell_05_solver_table(solver_labels=solver_labels),
        build_cell_06_pretrain_md(),
        build_cell_07_pretrain_data_gen(),
        build_cell_08_pretrain_loop(),
        build_cell_09_pretrain_loss_plot(),
        build_cell_10_data_md(),
        build_cell_11_chakravorty(),
    ]
    for idx, cell in enumerate(cells):
        cell.id = f"cell_{idx:02d}"
    nb.cells = cells

    nbformat.validate(nb)
    with open(output_path, "w") as fh:
        nbformat.write(nb, fh)
    return nb


if __name__ == "__main__":
    main()
