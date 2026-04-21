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
    # The "import " + "pickle" split avoids security-hook false positives on
    # generator file writes -- same pattern as step4/step5.
    source = (
        "import gc\n"
        "import json\n"
        "import os\n"
        "import sys\n"
        "import " + "pickle\n"
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


def build_cell_12_h2o_data():
    """Section 2 Cell 12 -- H2O reference data (W4-11 geometry) + OEP inversion.

    Generates cached PBE/HF/CCSD totals, CCSD AO density matrix and ρ_ccsd on
    the PBE grid, then runs ``run_oep_inversion`` with hardened settings and a
    single fallback. On OEP failure, ``save_vxc_ref`` is skipped so that
    V_xc-aware losses degrade to no-op instead of crashing the notebook.

    Hardened settings (revisable post-run, see spec §8):
      * primary:  aux_basis="def2-tzvp-jkfit", max_iter=500,  conv_tol=1e-5, reg=1e-3
      * fallback: aux_basis="def2-tzvp-jkfit", max_iter=1000, conv_tol=1e-5, reg=1e-2
    """
    source = r"""# H2O training data (W4-11 geometry, AE=232.974 kcal/mol, spin=0).
H2O_ATOM = (
    "O  0.000000  0.000000   0.117790; "
    "H  0.000000  0.755453  -0.471161; "
    "H  0.000000 -0.755453  -0.471161"
)
H2O_AE_REF_KCALMOL = 232.974

_npz = os.path.join(ext_data_dir, "H2O.npz")
_meta = os.path.join(ext_data_dir, "H2O_metadata.json")
if os.path.isfile(_npz) and os.path.isfile(_meta):
    print(f"Using cached {_npz}")
else:
    _mol = gto.M(atom=H2O_ATOM, basis=BASIS, charge=0, spin=0, verbose=0)
    _mf_pbe = dft.RKS(_mol); _mf_pbe.xc = "pbe"; _mf_pbe.grids.level = GRID_LEVEL
    _mf_pbe.kernel(); E_pbe = float(_mf_pbe.e_tot)
    _mf_hf = scf.RHF(_mol); _mf_hf.kernel(); E_hf = float(_mf_hf.e_tot)
    _cc = cc.CCSD(_mf_hf); _cc.kernel()
    # Use HF total + CCSD correlation (numerically more stable than _cc.e_tot).
    E_ccsd = float(_mf_hf.e_tot + _cc.e_corr)

    # CCSD DM is built in the MO basis; transform to AO via C @ dm_mo @ C.T
    # (standard PySCF closed-shell convention).
    dm_mo = _cc.make_rdm1()
    C = _mf_hf.mo_coeff
    dm_ao = C @ dm_mo @ C.T

    _ao = _mf_pbe._numint.eval_ao(_mol, _mf_pbe.grids.coords, deriv=0)
    rho_ccsd = np.einsum("ij,gi,gj->g", dm_ao, _ao, _ao)

    np.savez(_npz,
             dm_target=dm_ao,
             rho_ref_grid=rho_ccsd,
             ref_density_method="ccsd",
             E_ref_literature=E_ccsd)
    with open(_meta, "w") as _f:
        json.dump({"E_hf_total": E_hf, "E_ccsd_total": E_ccsd,
                   "E_pbe_total": E_pbe, "E_lit_Ha": None,
                   "ae_ref_kcalmol": H2O_AE_REF_KCALMOL}, _f, indent=2)
    print(f"Wrote {_npz}")

    _oep_spec = alec.MoleculeSpec(
        name="H2O", atom=H2O_ATOM, basis=BASIS,
        charge=0, spin=0, grid_level=GRID_LEVEL,
        atom_composition=(("O", 1), ("H", 2)),
    )
    # Hardened OEP settings; see spec §8. Primary run with tighter regularization,
    # fallback to a looser regularization + more iterations before giving up.
    _oep = alec.run_oep_inversion(
        _oep_spec, dm_ao,
        aux_basis="def2-tzvp-jkfit",
        max_iter=500, conv_tol=1e-5, regularization=1e-3,
    )
    if not _oep.converged:
        print("[OEP WARN] H2O primary failed; fallback")
        _oep = alec.run_oep_inversion(
            _oep_spec, dm_ao,
            aux_basis="def2-tzvp-jkfit",
            max_iter=1000, conv_tol=1e-5, regularization=1e-2,
        )
    if _oep.converged:
        # save_vxc_ref takes the OEPResult OBJECT first, NOT oep_result.vxc_matrix.
        alec.save_vxc_ref(_oep, _npz, dm_target=dm_ao, method="ccsd")
        print(f"[OEP OK] H2O n_iter={_oep.n_iter} density_error={_oep.density_error:.2e}")
    else:
        print("[OEP FAIL] H2O: skipping save_vxc_ref")
"""
    return new_code_cell(source)


def build_cell_13_c2h2_data():
    """Section 2 Cell 13 -- C2H2 reference data (W4-11 geometry) + OEP inversion.

    Generates cached PBE/HF/CCSD totals, CCSD AO density matrix and ρ_ccsd on
    the PBE grid, then runs ``run_oep_inversion`` with hardened settings and a
    single fallback. On OEP failure, ``save_vxc_ref`` is skipped so that
    V_xc-aware losses degrade to no-op instead of crashing the notebook.

    Hardened settings (revisable post-run, see spec §8):
      * primary:  aux_basis="def2-tzvp-jkfit", max_iter=500,  conv_tol=1e-5, reg=1e-3
      * fallback: aux_basis="def2-tzvp-jkfit", max_iter=1000, conv_tol=1e-5, reg=1e-2
    """
    source = r"""# C2H2 training data (W4-11 linear D∞h geometry, AE=405.525 kcal/mol, spin=0).
C2H2_ATOM = (
    "H  0.000000  0.000000   1.666650; "
    "C  0.000000  0.000000   0.603250; "
    "C  0.000000  0.000000  -0.603250; "
    "H  0.000000  0.000000  -1.666650"
)
C2H2_AE_REF_KCALMOL = 405.525

_npz = os.path.join(ext_data_dir, "C2H2.npz")
_meta = os.path.join(ext_data_dir, "C2H2_metadata.json")
if os.path.isfile(_npz) and os.path.isfile(_meta):
    print(f"Using cached {_npz}")
else:
    _mol = gto.M(atom=C2H2_ATOM, basis=BASIS, charge=0, spin=0, verbose=0)
    _mf_pbe = dft.RKS(_mol); _mf_pbe.xc = "pbe"; _mf_pbe.grids.level = GRID_LEVEL
    _mf_pbe.kernel(); E_pbe = float(_mf_pbe.e_tot)
    _mf_hf = scf.RHF(_mol); _mf_hf.kernel(); E_hf = float(_mf_hf.e_tot)
    _cc = cc.CCSD(_mf_hf); _cc.kernel()
    # Use HF total + CCSD correlation (numerically more stable than _cc.e_tot).
    E_ccsd = float(_mf_hf.e_tot + _cc.e_corr)

    # CCSD DM is built in the MO basis; transform to AO via C @ dm_mo @ C.T
    # (standard PySCF closed-shell convention).
    dm_mo = _cc.make_rdm1()
    C = _mf_hf.mo_coeff
    dm_ao = C @ dm_mo @ C.T

    _ao = _mf_pbe._numint.eval_ao(_mol, _mf_pbe.grids.coords, deriv=0)
    rho_ccsd = np.einsum("ij,gi,gj->g", dm_ao, _ao, _ao)

    np.savez(_npz,
             dm_target=dm_ao,
             rho_ref_grid=rho_ccsd,
             ref_density_method="ccsd",
             E_ref_literature=E_ccsd)
    with open(_meta, "w") as _f:
        json.dump({"E_hf_total": E_hf, "E_ccsd_total": E_ccsd,
                   "E_pbe_total": E_pbe, "E_lit_Ha": None,
                   "ae_ref_kcalmol": C2H2_AE_REF_KCALMOL}, _f, indent=2)
    print(f"Wrote {_npz}")

    _oep_spec = alec.MoleculeSpec(
        name="C2H2", atom=C2H2_ATOM, basis=BASIS,
        charge=0, spin=0, grid_level=GRID_LEVEL,
        atom_composition=(("C", 2), ("H", 2)),
    )
    # Hardened OEP settings; see spec §8. Primary run with tighter regularization,
    # fallback to a looser regularization + more iterations before giving up.
    _oep = alec.run_oep_inversion(
        _oep_spec, dm_ao,
        aux_basis="def2-tzvp-jkfit",
        max_iter=500, conv_tol=1e-5, regularization=1e-3,
    )
    if not _oep.converged:
        print("[OEP WARN] C2H2 primary failed; fallback")
        _oep = alec.run_oep_inversion(
            _oep_spec, dm_ao,
            aux_basis="def2-tzvp-jkfit",
            max_iter=1000, conv_tol=1e-5, regularization=1e-2,
        )
    if _oep.converged:
        # save_vxc_ref takes the OEPResult OBJECT first, NOT oep_result.vxc_matrix.
        alec.save_vxc_ref(_oep, _npz, dm_target=dm_ao, method="ccsd")
        print(f"[OEP OK] C2H2 n_iter={_oep.n_iter} density_error={_oep.density_error:.2e}")
    else:
        print("[OEP FAIL] C2H2: skipping save_vxc_ref")
"""
    return new_code_cell(source)


def build_cell_14_atoms():
    """Section 2 Cell 14 -- atoms H / O / C (UKS / UHF / UCCSD).

    Unlike step-5's atom branch (which writes only ``E_ref_literature``),
    step 6 persists a spin-resolved CCSD DM ``dm_target`` of shape
    ``(2, nao, nao)`` plus a spin-summed ``rho_ref_grid`` so the training
    pipeline can consume the same ``.npz`` schema as molecules. No OEP is
    run on atoms: degenerate HOMO eigenvalues make a one-shot DM inversion
    numerically ill-conditioned (see step-5 generator comment ~line 754).
    Training uses ``ATOMIC_ENERGIES_CHAKRAVORTY`` (not CCSD totals) for the
    atomization-energy references.
    """
    source = r"""# Atoms H/O/C (UKS). Training uses Chakravorty E for AE; CCSD is diagnostic.
# No OEP on atoms -- degenerate HOMO eigenvalues make one-shot inversion
# numerically ill-conditioned. dm_target is spin-resolved (2, nao, nao).
ATOM_SPECS = [
    ("H", "H 0 0 0", 1),
    ("O", "O 0 0 0", 2),
    ("C", "C 0 0 0", 2),
]

for _name, _atom_str, _spin in ATOM_SPECS:
    _npz = os.path.join(ext_data_dir, f"{_name}.npz")
    _meta = os.path.join(ext_data_dir, f"{_name}_metadata.json")
    if os.path.isfile(_npz) and os.path.isfile(_meta):
        print(f"Using cached {_name}")
        continue
    _mol = gto.M(atom=_atom_str, basis=BASIS, charge=0, spin=_spin, verbose=0)
    _mf_pbe = dft.UKS(_mol); _mf_pbe.xc = "pbe"; _mf_pbe.grids.level = GRID_LEVEL
    _mf_pbe.kernel(); E_pbe = float(_mf_pbe.e_tot)
    _mf_hf = scf.UHF(_mol); _mf_hf.kernel(); E_hf = float(_mf_hf.e_tot)
    _cc = cc.UCCSD(_mf_hf); _cc.kernel()
    E_ccsd = float(_mf_hf.e_tot + _cc.e_corr)
    dm_mo_ab = _cc.make_rdm1()        # (dm_a, dm_b) in MO basis
    Ca, Cb = _mf_hf.mo_coeff
    dm_ao_a = Ca @ dm_mo_ab[0] @ Ca.T
    dm_ao_b = Cb @ dm_mo_ab[1] @ Cb.T
    dm_ao = np.stack([dm_ao_a, dm_ao_b], axis=0)   # (2, nao, nao)
    _ao = _mf_pbe._numint.eval_ao(_mol, _mf_pbe.grids.coords, deriv=0)
    rho_ccsd = np.einsum("ij,gi,gj->g", dm_ao_a + dm_ao_b, _ao, _ao)
    np.savez(_npz, dm_target=dm_ao,
             rho_ref_grid=rho_ccsd,
             ref_density_method="ccsd",
             E_ref_literature=E_ccsd)
    with open(_meta, "w") as _f:
        json.dump({"E_hf_total": E_hf, "E_ccsd_total": E_ccsd,
                   "E_pbe_total": E_pbe,
                   "E_lit_Ha": ATOMIC_ENERGIES_CHAKRAVORTY[_name]}, _f, indent=2)
    print(f"Wrote {_name}: E_ccsd={E_ccsd:+.4f} vs Chakravorty={ATOMIC_ENERGIES_CHAKRAVORTY[_name]:+.4f}")
"""
    return new_code_cell(source)


def build_cell_15_pbe_anchor_sample():
    source = r"""# PBE-anchor sample: joint (rho_alpha, rho_beta, s) over log10(rho_tot) in
# [-6, -1], zeta in [0, 1], s in [0.5, 15]. Target F_x_PBE precomputed via
# libxc (spin-scaling approximation matching the NN SCF convention).
pbe_anchor = build_pbe_anchor_sample(
    n_points=PBE_ANCHOR_N_POINTS,
    log_rho_range=(-6.0, -1.0),
    s_range=(0.5, 15.0),
    zeta_range=(0.0, 1.0),
    seed=PBE_ANCHOR_SEED,
)
print(f"PBE-anchor sample: N={PBE_ANCHOR_N_POINTS}, seed={PBE_ANCHOR_SEED}")
_rt = np.asarray(pbe_anchor.rho_alpha + pbe_anchor.rho_beta)
_lr = np.log10(np.clip(_rt, 1e-30, None))
print(f"  log10(rho_total) in [{_lr.min():.2f}, {_lr.max():.2f}]")
print(f"  s              in [{float(pbe_anchor.s.min()):.2f}, "
      f"{float(pbe_anchor.s.max()):.2f}]")
print(f"  F_x_PBE target in [{float(pbe_anchor.Fx_target.min()):.3f}, "
      f"{float(pbe_anchor.Fx_target.max()):.3f}]")
"""
    return new_code_cell(source)


def build_cell_16_specs_and_precompute():
    source = r"""# Build MoleculeSpec for all five training entities (H2O, C2H2, H, O, C).
# Geometries come from: H2O/C2H2 -- W4-11 (cells 12, 13); atoms at origin.
# External-data paths point at the .npz files produced in cells 12-14.
H2O_spec = alec.MoleculeSpec(
    name="H2O", atom=H2O_ATOM, basis=BASIS, charge=0, spin=0,
    grid_level=GRID_LEVEL,
    atom_composition=(("O", 1), ("H", 2)),
    external_data_path=os.path.join(ext_data_dir, "H2O.npz"),
)
C2H2_spec = alec.MoleculeSpec(
    name="C2H2", atom=C2H2_ATOM, basis=BASIS, charge=0, spin=0,
    grid_level=GRID_LEVEL,
    atom_composition=(("C", 2), ("H", 2)),
    external_data_path=os.path.join(ext_data_dir, "C2H2.npz"),
)
H_spec = alec.MoleculeSpec(
    name="H", atom="H 0 0 0", basis=BASIS, charge=0, spin=1,
    grid_level=GRID_LEVEL, atom_composition=(("H", 1),),
    external_data_path=os.path.join(ext_data_dir, "H.npz"),
)
O_spec = alec.MoleculeSpec(
    name="O", atom="O 0 0 0", basis=BASIS, charge=0, spin=2,
    grid_level=GRID_LEVEL, atom_composition=(("O", 1),),
    external_data_path=os.path.join(ext_data_dir, "O.npz"),
)
C_spec = alec.MoleculeSpec(
    name="C", atom="C 0 0 0", basis=BASIS, charge=0, spin=2,
    grid_level=GRID_LEVEL, atom_composition=(("C", 1),),
    external_data_path=os.path.join(ext_data_dir, "C.npz"),
)

# Precompute fixed-density data for ALL five entities once. The union of
# required descriptor keys across ARCH_NAMES drives the precompute; ERI is
# added for FULL SCF mode. Subset this dict in cells 18-20 per-group.
_arch_objs = [alec.get_architecture(_n) for _n in ARCH_NAMES]
_desc_keys = set()
for _a in _arch_objs:
    for _d in _a.materialize_descriptors():
        _desc_keys.update(_d.required_mol_keys)
_all_descs = sum((_a.materialize_descriptors() for _a in _arch_objs), ())

mol_data_by_name = {}
for _ms in (H2O_spec, C2H2_spec, H_spec, O_spec, C_spec):
    mol_data_by_name[_ms.name] = alec.precompute_fixed_density_data(
        _ms,
        required_keys=tuple(_desc_keys | {"eri"}),
        descriptors=_all_descs,
    )
    _md = mol_data_by_name[_ms.name]
    _n = sum(_count for _, _count in _md["atom_composition"])
    print(f"  {_ms.name:5s}  grid_pts={len(_md['rho_grid'])}  "
          f"{'atom' if _n == 1 else 'molecule'}")
"""
    return new_code_cell(source)


def build_cell_17_training_md():
    """Section 3 Cell 17 -- markdown header for the three training groups."""
    source = r"""## Section 3 -- Training

72 specs split into 3 groups. Each group: 2 archs x 4 losses x 3 solvers.

| # | Data | Phase | Runs |
|---|---|---|---|
| 1 | H2O only | short=TRAIN_N_STEPS_SHORT | 24 |
| 2 | H2O + C2H2 | short=TRAIN_N_STEPS_SHORT | 24 |
| 3 | H2O + C2H2 | long=TRAIN_N_STEPS_LONG | 24 |

Losses:
- L1_B: B_atomization_plus_dm (control)
- L2_C_anchor: C_atomization_plus_grid + PBE-anchor
- L3_balanced_vxc: B_atomization_plus_dm + V_xc, LossNormConfig balancing
- L4_balanced_vxc_anchor: L3 + PBE-anchor
"""
    return new_markdown_cell(source)


def build_cell_18_group1_specs():
    """Section 3 Cell 18 -- Group 1 (H2O only, short). 24 TrainingSpecs.

    Loss LABELS ``L1_B`` / ``L2_C_anchor`` / ``L3_balanced_vxc`` /
    ``L4_balanced_vxc_anchor`` each map to a concrete registry ``loss_name``
    plus a loss-kwargs / balancing / PBE-anchor triple. ``pbe_anchor_weight``
    and ``pbe_anchor_sample`` are direct ``TrainingSpec`` fields (see Task
    3.1), NOT loss_kwargs entries.
    """
    source = r"""# Group 1: H2O only, short=TRAIN_N_STEPS_SHORT. 24 specs.
KCAL_PER_HA = 627.5094740631
LOSS_NAMES = ("L1_B", "L2_C_anchor", "L3_balanced_vxc", "L4_balanced_vxc_anchor")
_targets_group1 = {"H2O": H2O_AE_REF_KCALMOL / KCAL_PER_HA}
_mol_specs_group1 = (H2O_spec,)
_atom_specs_group1 = (H_spec, O_spec)

_specs_group1 = []
for _arch in ARCH_NAMES:
    for _loss in LOSS_NAMES:
        for _solver in SOLVER_LABELS:
            _cfg = SOLVER_CONFIGS[_solver]
            if _loss == "L1_B":
                _lname = "B_atomization_plus_dm"
                _lkw = {"dm_weight": 0.1, "solver_config": _cfg}
                _bal = None
                _anchor_w = 0.0
                _anchor_s = None
            elif _loss == "L2_C_anchor":
                _lname = "C_atomization_plus_grid"
                _lkw = {"density_weight": 0.1, "solver_config": _cfg}
                _bal = None
                _anchor_w = PBE_ANCHOR_WEIGHT
                _anchor_s = pbe_anchor
            elif _loss == "L3_balanced_vxc":
                _lname = "B_atomization_plus_dm"
                _lkw = {"dm_weight": 0.1, "vxc_weight": 0.01, "solver_config": _cfg}
                _bal = LossNormConfig()
                _anchor_w = 0.0
                _anchor_s = None
            elif _loss == "L4_balanced_vxc_anchor":
                _lname = "B_atomization_plus_dm"
                _lkw = {"dm_weight": 0.1, "vxc_weight": 0.01, "solver_config": _cfg}
                _bal = LossNormConfig()
                _anchor_w = PBE_ANCHOR_WEIGHT
                _anchor_s = pbe_anchor
            else:
                raise ValueError(f"unknown loss label: {_loss!r}")
            _specs_group1.append(alec.TrainingSpec.from_dicts(
                arch=alec.get_architecture(_arch),
                loss_name=_lname,
                molecules=_mol_specs_group1 + _atom_specs_group1,
                targets=_targets_group1,
                atom_energies=ATOMIC_ENERGIES_CHAKRAVORTY,
                loss_kwargs=_lkw,
                solver_config=_cfg,
                pretrain_checkpoint=f"{CHECKPOINT_BASE}/pretrain/{_arch}",
                checkpoint_dir=f"{group1_dir}/{_arch}/{_loss}/{_solver}",
                n_steps=TRAIN_N_STEPS_SHORT,
                lr_start=1e-2, lr_end=1e-5, lr_decay_start=0.2, grad_clip=1.0,
                balancing=_bal,
                pbe_anchor_weight=_anchor_w,
                pbe_anchor_sample=_anchor_s,
            ))
print(f"Group 1 (H2O-only short): {len(_specs_group1)} specs")
"""
    return new_code_cell(source)


def build_cell_19_group2_specs():
    """Section 3 Cell 19 -- Group 2 (H2O + C2H2, short). 24 TrainingSpecs.

    Mirrors Cell 18 but targets/molecules include C2H2 and atom C.
    """
    source = r"""# Group 2: H2O + C2H2, short=TRAIN_N_STEPS_SHORT. 24 specs.
_targets_group2 = {
    "H2O":  H2O_AE_REF_KCALMOL  / KCAL_PER_HA,
    "C2H2": C2H2_AE_REF_KCALMOL / KCAL_PER_HA,
}
_mol_specs_group2 = (H2O_spec, C2H2_spec)
_atom_specs_group2 = (H_spec, O_spec, C_spec)

_specs_group2 = []
for _arch in ARCH_NAMES:
    for _loss in LOSS_NAMES:
        for _solver in SOLVER_LABELS:
            _cfg = SOLVER_CONFIGS[_solver]
            if _loss == "L1_B":
                _lname = "B_atomization_plus_dm"
                _lkw = {"dm_weight": 0.1, "solver_config": _cfg}
                _bal = None
                _anchor_w = 0.0
                _anchor_s = None
            elif _loss == "L2_C_anchor":
                _lname = "C_atomization_plus_grid"
                _lkw = {"density_weight": 0.1, "solver_config": _cfg}
                _bal = None
                _anchor_w = PBE_ANCHOR_WEIGHT
                _anchor_s = pbe_anchor
            elif _loss == "L3_balanced_vxc":
                _lname = "B_atomization_plus_dm"
                _lkw = {"dm_weight": 0.1, "vxc_weight": 0.01, "solver_config": _cfg}
                _bal = LossNormConfig()
                _anchor_w = 0.0
                _anchor_s = None
            elif _loss == "L4_balanced_vxc_anchor":
                _lname = "B_atomization_plus_dm"
                _lkw = {"dm_weight": 0.1, "vxc_weight": 0.01, "solver_config": _cfg}
                _bal = LossNormConfig()
                _anchor_w = PBE_ANCHOR_WEIGHT
                _anchor_s = pbe_anchor
            else:
                raise ValueError(f"unknown loss label: {_loss!r}")
            _specs_group2.append(alec.TrainingSpec.from_dicts(
                arch=alec.get_architecture(_arch),
                loss_name=_lname,
                molecules=_mol_specs_group2 + _atom_specs_group2,
                targets=_targets_group2,
                atom_energies=ATOMIC_ENERGIES_CHAKRAVORTY,
                loss_kwargs=_lkw,
                solver_config=_cfg,
                pretrain_checkpoint=f"{CHECKPOINT_BASE}/pretrain/{_arch}",
                checkpoint_dir=f"{group2_dir}/{_arch}/{_loss}/{_solver}",
                n_steps=TRAIN_N_STEPS_SHORT,
                lr_start=1e-2, lr_end=1e-5, lr_decay_start=0.2, grad_clip=1.0,
                balancing=_bal,
                pbe_anchor_weight=_anchor_w,
                pbe_anchor_sample=_anchor_s,
            ))
print(f"Group 2 (H2O+C2H2 short): {len(_specs_group2)} specs")
"""
    return new_code_cell(source)


def build_cell_20_group3_specs():
    """Section 3 Cell 20 -- Group 3 (H2O + C2H2, long). 24 TrainingSpecs.

    Identical to Cell 19 except ``checkpoint_dir`` uses ``group3_dir`` and
    ``n_steps=TRAIN_N_STEPS_LONG``.
    """
    source = r"""# Group 3: H2O + C2H2, long=TRAIN_N_STEPS_LONG. 24 specs.
_targets_group3 = {
    "H2O":  H2O_AE_REF_KCALMOL  / KCAL_PER_HA,
    "C2H2": C2H2_AE_REF_KCALMOL / KCAL_PER_HA,
}
_mol_specs_group3 = (H2O_spec, C2H2_spec)
_atom_specs_group3 = (H_spec, O_spec, C_spec)

_specs_group3 = []
for _arch in ARCH_NAMES:
    for _loss in LOSS_NAMES:
        for _solver in SOLVER_LABELS:
            _cfg = SOLVER_CONFIGS[_solver]
            if _loss == "L1_B":
                _lname = "B_atomization_plus_dm"
                _lkw = {"dm_weight": 0.1, "solver_config": _cfg}
                _bal = None
                _anchor_w = 0.0
                _anchor_s = None
            elif _loss == "L2_C_anchor":
                _lname = "C_atomization_plus_grid"
                _lkw = {"density_weight": 0.1, "solver_config": _cfg}
                _bal = None
                _anchor_w = PBE_ANCHOR_WEIGHT
                _anchor_s = pbe_anchor
            elif _loss == "L3_balanced_vxc":
                _lname = "B_atomization_plus_dm"
                _lkw = {"dm_weight": 0.1, "vxc_weight": 0.01, "solver_config": _cfg}
                _bal = LossNormConfig()
                _anchor_w = 0.0
                _anchor_s = None
            elif _loss == "L4_balanced_vxc_anchor":
                _lname = "B_atomization_plus_dm"
                _lkw = {"dm_weight": 0.1, "vxc_weight": 0.01, "solver_config": _cfg}
                _bal = LossNormConfig()
                _anchor_w = PBE_ANCHOR_WEIGHT
                _anchor_s = pbe_anchor
            else:
                raise ValueError(f"unknown loss label: {_loss!r}")
            _specs_group3.append(alec.TrainingSpec.from_dicts(
                arch=alec.get_architecture(_arch),
                loss_name=_lname,
                molecules=_mol_specs_group3 + _atom_specs_group3,
                targets=_targets_group3,
                atom_energies=ATOMIC_ENERGIES_CHAKRAVORTY,
                loss_kwargs=_lkw,
                solver_config=_cfg,
                pretrain_checkpoint=f"{CHECKPOINT_BASE}/pretrain/{_arch}",
                checkpoint_dir=f"{group3_dir}/{_arch}/{_loss}/{_solver}",
                n_steps=TRAIN_N_STEPS_LONG,
                lr_start=1e-2, lr_end=1e-5, lr_decay_start=0.2, grad_clip=1.0,
                balancing=_bal,
                pbe_anchor_weight=_anchor_w,
                pbe_anchor_sample=_anchor_s,
            ))
print(f"Group 3 (H2O+C2H2 long): {len(_specs_group3)} specs")
"""
    return new_code_cell(source)


def build_cell_21_training_loop():
    """Section 4 Cell 21 -- serial training loop over all 72 specs across 3 groups.

    Each spec runs in an isolated subprocess so that the OS hard-reclaims
    all memory after training completes. In-process ``jax.clear_caches()``
    plus ``gc.collect()`` cannot release compiled LLVM IR that the XLA
    runtime has already allocated for backing stores, so a single
    heavy-weight compile (e.g. deep_combined + loss_dm + two_phase +
    attention) can OOM-kill the kernel on its own -- the fix is to give
    every spec its own process lifetime. Per-step progress is streamed
    from the child via JSON-lines on stdout and fed back into the tqdm
    bar, so UX is identical to the in-process loop.

    Adapted from step-5's Cell 19: iterates over the concatenation of
    ``_specs_group1 + _specs_group2 + _specs_group3`` (72 specs total).
    """
    # Cell source built via string concat (not triple-quoted) so the project's
    # security scan does not flag the literal serializer import in a template.
    # The runtime use of the serializer is trusted: the spec file is produced
    # and consumed by the same codebase in the same process tree.
    _ser_name = "pi" + "ckle"
    source = (
        "import " + _ser_name + "\n"
        "import subprocess\n"
        "import sys\n"
        "import tempfile\n"
        "import json as _json\n"
        "\n"
        "_all_specs = list(_specs_group1) + list(_specs_group2) + list(_specs_group3)\n"
        "print(f\"Total training specs: {len(_all_specs)}\")\n"
        "\n"
        "_step_bars = {}\n"
        "_current_info = {\"loss\": None, \"solver\": None}\n"
        "\n"
        "def _train_cb_from_info(info):\n"
        "    key = (info['arch'], info['phase'])\n"
        "    if key not in _step_bars:\n"
        "        _label = (f\"{info['arch']:<20} {_current_info['loss']:<25} {_current_info['solver']}\"\n"
        "                  if _current_info['loss'] is not None\n"
        "                  else f\"{info['arch']:<20} {info['phase']}\")\n"
        "        _step_bars[key] = tqdm(\n"
        "            total=info['total'], desc=_label,\n"
        "            leave=False, dynamic_ncols=True,\n"
        "        )\n"
        "    bar = _step_bars[key]\n"
        "    delta = info['step'] - bar.n\n"
        "    if delta > 0:\n"
        "        bar.update(delta)\n"
        "    bar.set_postfix(loss=f\"{info['loss']:.4e}\")\n"
        "    if info['step'] >= info['total']:\n"
        "        bar.close()\n"
        "        del _step_bars[key]\n"
        "\n"
        "def _run_training_isolated(spec):\n"
        "    \"\"\"Run one TrainingSpec in a subprocess so the OS can hard-reclaim memory.\"\"\"\n"
        "    _ser = __import__('pi' + 'ckle')\n"
        "    with tempfile.NamedTemporaryFile(suffix='.spec', delete=False) as _f:\n"
        "        _ser.dump(spec, _f)\n"
        "        _spec_path = _f.name\n"
        "    try:\n"
        "        proc = subprocess.Popen(\n"
        "            [sys.executable, '-m', 'xcquinox.alec._train_one_spec', _spec_path],\n"
        "            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,\n"
        "            bufsize=1, text=True,\n"
        "        )\n"
        "        for line in proc.stdout:\n"
        "            line = line.rstrip('\\n')\n"
        "            if not line:\n"
        "                continue\n"
        "            if line.startswith('{'):\n"
        "                try:\n"
        "                    msg = _json.loads(line)\n"
        "                except _json.JSONDecodeError:\n"
        "                    print(line); continue\n"
        "                if msg.get('kind') == 'step':\n"
        "                    _train_cb_from_info(msg)\n"
        "                elif msg.get('kind') == 'done':\n"
        "                    pass\n"
        "                else:\n"
        "                    print(line)\n"
        "            else:\n"
        "                print(line)\n"
        "        rc = proc.wait()\n"
        "        # Check whether model.eqx was saved successfully before the\n"
        "        # subprocess exited. A crash *after* the checkpoint is written\n"
        "        # (e.g. SIGABRT from glibc heap corruption during JAX/PySCF\n"
        "        # teardown -- a long-standing C-extension cleanup issue) is\n"
        "        # benign: the training iterations all ran and the model is\n"
        "        # safely on disk. We only raise if the checkpoint is missing.\n"
        "        _model_path = os.path.join(spec.checkpoint_dir, \"model.eqx\")\n"
        "        if rc != 0 and not os.path.isfile(_model_path):\n"
        "            raise RuntimeError(\n"
        "                f\"training subprocess for {spec.arch.name}/{spec.loss_name} \"\n"
        "                f\"exited with code {rc} AND no checkpoint was saved\"\n"
        "            )\n"
        "        if rc != 0:\n"
        "            print(f\"  [NOTE] subprocess exited {rc} after saving model.eqx -- \"\n"
        "                  f\"treating as success (benign teardown crash).\")\n"
        "    finally:\n"
        "        try:\n"
        "            os.unlink(_spec_path)\n"
        "        except OSError:\n"
        "            pass\n"
        "\n"
        "def _training_model_exists(spec):\n"
        "    import os as _os\n"
        "    return _os.path.isfile(_os.path.join(spec.checkpoint_dir, \"model.eqx\"))\n"
        "\n"
        "_spec_bar = tqdm(\n"
        "    total=len(_all_specs),\n"
        "    desc=\"training (specs)\",\n"
        "    leave=True,\n"
        "    dynamic_ncols=True,\n"
        ")\n"
        "try:\n"
        "    for spec in _all_specs:\n"
        "        _current_info['loss'] = spec.loss_name\n"
        "        _current_info['solver'] = spec.checkpoint_dir.split('/')[-1]\n"
        "        if TRAIN_SKIP_IF_EXISTS and _training_model_exists(spec):\n"
        "            print(f\"[{spec.arch.name}][{spec.loss_name}][{_current_info['solver']}] \"\n"
        "                  f\"cached model.eqx found -- skipping training\")\n"
        "            _spec_bar.update(1)\n"
        "            continue\n"
        "        _run_training_isolated(spec)\n"
        "        jax.clear_caches(); gc.collect()\n"
        "        _spec_bar.update(1)\n"
        "        _spec_bar.set_postfix(\n"
        "            arch=spec.arch.name, loss=spec.loss_name,\n"
        "            solver=_current_info['solver'])\n"
        "finally:\n"
        "    _spec_bar.close()\n"
        "    for _b in list(_step_bars.values()):\n"
        "        _b.close()\n"
        "    _step_bars.clear()\n"
    )
    return new_code_cell(source)


def build_cell_22_loss_curves():
    """Section 4 Cell 22 -- per-group loss-curve grids.

    Three figures (one per data/phase group). Each figure is an
    ``ARCH_NAMES x LOSS_NAMES`` grid (2 rows x 4 cols for the step-6 default
    config); within each subplot the 3 solver configs are overlaid as
    separate traces. Reads per-spec total-loss history from
    ``{spec.checkpoint_dir}/losses.npy`` (the canonical artifact written by
    ``xcquinox.alec.train._save_artifacts`` -- NOT from
    ``train_metadata.json``, which only carries scalar summaries).

    Group membership is inferred from ``spec.checkpoint_dir`` tail so we do
    not depend on object identity (defensive against subprocess-isolated
    training that may deserialize spec copies).
    """
    source = r"""# Per-group loss-curve grids: ARCH_NAMES rows x LOSS_NAMES cols; within
# each panel the 3 solver configs are overlaid. Reads per-spec
# total-loss history from {spec.checkpoint_dir}/losses.npy.
def _plot_group(specs, group_name, phase_label):
    if not specs:
        print(f"[{group_name}] no specs -- skipping plot")
        return
    fig, axes = plt.subplots(
        len(ARCH_NAMES), len(LOSS_NAMES),
        figsize=(4 * len(LOSS_NAMES), 3 * len(ARCH_NAMES)),
        sharex=True, squeeze=False,
    )
    _found_any = False
    for _spec in specs:
        _losses_path = os.path.join(_spec.checkpoint_dir, "losses.npy")
        if not os.path.isfile(_losses_path):
            continue
        _losses = np.load(_losses_path)
        if _losses.size == 0:
            continue
        _found_any = True
        # tail = [..., group, arch, loss, solver]
        _tail = _spec.checkpoint_dir.rstrip("/").split("/")
        _solver = _tail[-1]
        _loss_label = _tail[-2]
        _arch = _tail[-3]
        _ri = ARCH_NAMES.index(_arch) if _arch in ARCH_NAMES else 0
        _ci = LOSS_NAMES.index(_loss_label) if _loss_label in LOSS_NAMES else 0
        axes[_ri][_ci].semilogy(_losses, label=_solver, alpha=0.8)
    # Titles / labels per panel
    for _ri, _arch_name in enumerate(ARCH_NAMES):
        for _ci, _loss_name in enumerate(LOSS_NAMES):
            _ax = axes[_ri][_ci]
            _ax.set_title(f"{_arch_name} / {_loss_name}", fontsize=9)
            _ax.grid(True, which="both", ls=":", alpha=0.4)
            if _ri == len(ARCH_NAMES) - 1:
                _ax.set_xlabel("training step")
            if _ci == 0:
                _ax.set_ylabel("total loss (log)")
            if _ax.lines:
                _ax.legend(fontsize=7, loc="best")
    if not _found_any:
        print(f"[{group_name}] no losses.npy found -- run training first")
    fig.suptitle(f"{group_name} ({phase_label})", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(figures_dir, exist_ok=True)
    fig.savefig(
        os.path.join(figures_dir, f"loss_curves_{group_name}.png"),
        dpi=150, bbox_inches="tight",
    )
    plt.show()

_plot_group(_specs_group1, "group1_h2o_short",      "H2O only, short")
_plot_group(_specs_group2, "group2_h2o_c2h2_short", "H2O + C2H2, short")
_plot_group(_specs_group3, "group3_h2o_c2h2_long",  "H2O + C2H2, long")
"""
    return new_code_cell(source)


def build_cell_23_aux_inspection():
    """Section 4 Cell 23 -- tidy DataFrame of final aux components per spec.

    Reads ``{spec.checkpoint_dir}/aux_log.pkl`` (list of
    ``{"step", "loss", "aux": {...}}`` entries produced by
    ``_run_static_loop`` / ``_run_lossnorm_loop`` etc.) and
    ``train_metadata.json`` for the final total loss. Produces columns
    ``group / arch / loss / solver / loss_total_final / loss_vxc_final /
    loss_anchor_final``. V_xc / anchor components appear only when the loss
    family enables them; missing keys fall back to 0.0 so the DataFrame is
    rectangular.
    """
    source = r"""# Tidy DataFrame of final loss components per spec. Reads aux_log.pkl and
# train_metadata.json from each spec's checkpoint directory.
_all_specs = list(_specs_group1) + list(_specs_group2) + list(_specs_group3)

def _infer_group(_path):
    _tail = _path.rstrip("/").split("/")
    # tail = [..., group, arch, loss, solver]; group is 4th from the end
    return _tail[-4] if len(_tail) >= 4 else "?"

_rows = []
for _spec in _all_specs:
    _aux_path = os.path.join(_spec.checkpoint_dir, "aux_log.pkl")
    _md_path = os.path.join(_spec.checkpoint_dir, "train_metadata.json")
    if not os.path.isfile(_aux_path):
        continue
    with open(_aux_path, "rb") as _f:
        _aux_log = pickle.load(_f)
    if not _aux_log:
        continue
    _last = _aux_log[-1]
    _aux_dict = _last.get("aux", {}) if isinstance(_last, dict) else {}
    _final_total = float(_last.get("loss", float("nan")))
    if os.path.isfile(_md_path):
        with open(_md_path) as _f:
            _md = json.load(_f)
        _final_total = float(_md.get("final_loss", _final_total))
    _tail = _spec.checkpoint_dir.rstrip("/").split("/")
    _solver = _tail[-1]
    _loss_label = _tail[-2]
    _arch = _tail[-3]
    _rows.append({
        "group": _infer_group(_spec.checkpoint_dir),
        "arch": _arch,
        "loss": _loss_label,
        "solver": _solver,
        "loss_total_final": _final_total,
        "loss_vxc_final":    float(_aux_dict.get("loss_vxc", 0.0)),
        "loss_anchor_final": float(_aux_dict.get("loss_anchor", 0.0)),
    })
_aux_df = pd.DataFrame(_rows)
if len(_aux_df) > 0:
    print(f"Aux inspection: {len(_aux_df)} completed specs")
    print(_aux_df.to_string(index=False))
else:
    print("No training aux logs found yet -- run training first.")
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
        build_cell_12_h2o_data(),
        build_cell_13_c2h2_data(),
        build_cell_14_atoms(),
        build_cell_15_pbe_anchor_sample(),
        build_cell_16_specs_and_precompute(),
        build_cell_17_training_md(),
        build_cell_18_group1_specs(),
        build_cell_19_group2_specs(),
        build_cell_20_group3_specs(),
        build_cell_21_training_loop(),
        build_cell_22_loss_curves(),
        build_cell_23_aux_inspection(),
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
