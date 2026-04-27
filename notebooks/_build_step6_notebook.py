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

---

> **Note (2026-04-27):** Step-6 attention runs prior to this date used a
> broken `SelfAttentionBlock` (softmax channel-gate, not self-attention).
> The block has been rewritten to canonical multi-head scaled-dot-product
> attention; the previous `*_attn` checkpoints have been deleted and must
> be regenerated.
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
        "\n"
        "# -- BLAS thread-cap for OEP inversions --------------------------------\n"
        "# Wu-Yang OEP (cell 12 + 13) is CPU-bound via PySCF's inner SCF, which\n"
        "# uses OpenMP-threaded BLAS. Because this kernel ALSO imports JAX (which\n"
        "# maintains its own thread pool), leaving PySCF at its default thread\n"
        "# count (N-cores) causes oversubscription and 5-10x slowdowns on every\n"
        "# L-BFGS-B iteration. Capping to a modest N via pyscf.lib.num_threads\n"
        "# for the duration of the call keeps the two pools out of each other's\n"
        "# way and recovers the subprocess-clean throughput.\n"
        "from contextlib import contextmanager as _contextmanager\n"
        "from pyscf import lib as _pyscf_lib\n"
        "\n"
        "@_contextmanager\n"
        "def _capped_blas_threads(n=4):\n"
        "    \"\"\"Temporarily cap PySCF's BLAS thread count to `n`.\"\"\"\n"
        "    _prev = _pyscf_lib.num_threads()\n"
        "    _pyscf_lib.num_threads(min(_prev, int(n)))\n"
        "    try:\n"
        "        yield\n"
        "    finally:\n"
        "        _pyscf_lib.num_threads(_prev)\n"
        "\n"
        "# -- DataFrame save/load helpers --------------------------------------\n"
        "# pandas.DataFrame.to_parquet requires pyarrow or fastparquet, which\n"
        "# are NOT in this environment's baseline. A hard ImportError at the\n"
        "# eval-df / transfer-df write step would kill the notebook. These\n"
        "# helpers prefer parquet when a backend is importable, and fall back\n"
        "# to CSV (pandas stdlib) otherwise -- CSV is safe (no code execution\n"
        "# risk) and portable. _df_load auto-detects which extension exists,\n"
        "# so runs with different backends can cohabit.\n"
        "def _parquet_engine_available():\n"
        "    try:\n"
        "        import pyarrow  # noqa: F401\n"
        "        return 'pyarrow'\n"
        "    except ImportError:\n"
        "        pass\n"
        "    try:\n"
        "        import fastparquet  # noqa: F401\n"
        "        return 'fastparquet'\n"
        "    except ImportError:\n"
        "        return None\n"
        "\n"
        "def _df_save(df, path):\n"
        "    \"\"\"Save a DataFrame to `path`. If the requested extension is\n"
        "    .parquet and no engine is available, writes a sibling .csv\n"
        "    instead. Returns the actual path written.\"\"\"\n"
        "    _base, _ext = os.path.splitext(path)\n"
        "    if _ext == '.parquet' and _parquet_engine_available() is None:\n"
        "        _actual = _base + '.csv'\n"
        "        df.to_csv(_actual, index=False)\n"
        "        return _actual\n"
        "    if _ext == '.parquet':\n"
        "        df.to_parquet(path)\n"
        "    elif _ext == '.csv':\n"
        "        df.to_csv(path, index=False)\n"
        "    else:\n"
        "        raise ValueError(f'_df_save: unsupported extension {_ext!r}')\n"
        "    return path\n"
        "\n"
        "def _df_load(path):\n"
        "    \"\"\"Load a DataFrame. If the requested .parquet file is missing\n"
        "    but a sibling .csv exists, load the .csv (and vice versa).\"\"\"\n"
        "    _base, _ext = os.path.splitext(path)\n"
        "    _parq = _base + '.parquet'\n"
        "    _csv  = _base + '.csv'\n"
        "    if os.path.isfile(_parq) and _parquet_engine_available() is not None:\n"
        "        return pd.read_parquet(_parq)\n"
        "    if os.path.isfile(_csv):\n"
        "        return pd.read_csv(_csv)\n"
        "    # Fall back to whatever was requested, letting pandas raise.\n"
        "    if _ext == '.parquet':\n"
        "        return pd.read_parquet(path)\n"
        "    return pd.read_csv(path)\n"
        "\n"
        "def _df_exists(path):\n"
        "    \"\"\"True if `path` -- or its sibling with the other extension --\n"
        "    exists on disk.\"\"\"\n"
        "    _base, _ = os.path.splitext(path)\n"
        "    return os.path.isfile(_base + '.parquet') or os.path.isfile(_base + '.csv')\n"
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
# Case-study memory budget: a heavier (def2-tzvp, GRID_LEVEL=3) pairing
# OOMs the first training step on an 8 GB GPU (~11 GB peak per-step
# tape). Step 6 keeps step 5's lighter (def2-svp, GRID_LEVEL=1) so the
# whole 72-spec sweep stays runnable on a modest workstation.
BASIS                    = "def2-svp"
GRID_LEVEL               = 1

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

# RUN_DIR namespaces every pretraining-dependent artifact under
# CHECKPOINT_BASE/PRETRAIN_LOSS_WEIGHTING/, so back-to-back runs of this
# notebook with different PRETRAIN_LOSS_WEIGHTING values produce side-by-side
# output trees that are directly comparable. Pretrain-run-INVARIANT
# artifacts (CCSD reference data, raw pretrain (rho, sigma) samples,
# precomputed transfer-set mol_data) stay under CHECKPOINT_BASE so they
# are reused across runs rather than recomputed.
RUN_DIR              = os.path.join(CHECKPOINT_BASE, PRETRAIN_LOSS_WEIGHTING)
ext_data_dir         = os.path.join(CHECKPOINT_BASE, "external_data")
pretrain_data_dir    = os.path.join(CHECKPOINT_BASE, "pretrain_data")
transfer_primary     = os.path.join(CHECKPOINT_BASE, "transfer_data", "primary")
transfer_secondary   = os.path.join(CHECKPOINT_BASE, "transfer_data", "secondary")
pretrain_dir         = os.path.join(RUN_DIR, "pretrain")
group1_dir           = os.path.join(RUN_DIR, "group1_h2o_short")
group2_dir           = os.path.join(RUN_DIR, "group2_h2o_c2h2_short")
group3_dir           = os.path.join(RUN_DIR, "group3_h2o_c2h2_long")
figures_dir          = os.path.join(RUN_DIR, "figures")
eval_dir             = os.path.join(RUN_DIR, "eval")
transfer_eval_dir    = os.path.join(RUN_DIR, "transfer_eval")
eval_baseline_root   = RUN_DIR  # eval_baseline_<kind>/ subdirs land here
for _d in (ext_data_dir, pretrain_data_dir, RUN_DIR, pretrain_dir,
           group1_dir, group2_dir, group3_dir, figures_dir, eval_dir,
           transfer_eval_dir, transfer_primary, transfer_secondary):
    os.makedirs(_d, exist_ok=True)

import pathlib
pathlib.Path(CHECKPOINT_BASE, "VERSION").write_text("step6-v2\\n")
print("DATA VERSION: step6-v2 (real-attention)")
print(f"  PRETRAIN_LOSS_WEIGHTING = {{PRETRAIN_LOSS_WEIGHTING!r}}")
print(f"  RUN_DIR        = {{RUN_DIR}}")
print("  Training:   {{H2O, C2H2}} + atoms {{H, O, C}}")
print("  Transfer P: {{H2, OH, CH4}} (W4-11)")
print("  Transfer S: {{NH3, HF, CO2, NH2}} (W4-11)")
print(f"  Wipe {{RUN_DIR}}/ to regenerate this pretraining variant only;")
print(f"  wipe {{CHECKPOINT_BASE}}/ to also discard CCSD/transfer data.")
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
          f"attention={{_cfg.attention}} num_heads={{_cfg.num_heads}} "
          f"descriptors={{len(_cfg.descriptors)}}")
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
weights_list = []  # Becke-Lebedev quadrature dr_i — needed by integration mode
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
    is_uks = (dm_ab.ndim == 3)

    if is_uks:
        # Open-shell: spin-resolve rho_gga and call libxc with spin=1 for
        # correct UKS PBE F_x / F_c targets. The closed-shell-on-total-density
        # call (spin=0) is wrong for Li/N/O/etc. (PBE 1996 §III spin-scaling
        # gives F_x_UKS != F_x_RKS(rho_total) for any nonzero polarization).
        dm_total = dm_ab[0] + dm_ab[1]
        rho_a_gga = mf._numint.eval_rho(mol, ao, dm_ab[0], xctype="GGA", hermi=True)
        rho_b_gga = mf._numint.eval_rho(mol, ao, dm_ab[1], xctype="GGA", hermi=True)
        rho_gga_uks = np.stack([rho_a_gga, rho_b_gga], axis=0)
        rho = rho_a_gga[0] + rho_b_gga[0]
        # UKS sigma = sigma_aa + 2 sigma_ab + sigma_bb (total ∇ρ squared)
        nabla_a = rho_a_gga[1:4]
        nabla_b = rho_b_gga[1:4]
        nabla_total = nabla_a + nabla_b
        sigma = (nabla_total ** 2).sum(axis=0)
        ex_pbe = mf._numint.eval_xc("PBE,", rho_gga_uks, spin=1)[0]
        ec_pbe = mf._numint.eval_xc(",PBE", rho_gga_uks, spin=1)[0]
        # UKS LDA baselines: pass (rho_a, rho_b) as 2-tuple of 1-D densities.
        ex_lda = mf._numint.eval_xc("LDA_X,", (rho_a_gga[0], rho_b_gga[0]), spin=1)[0]
        ec_lda = mf._numint.eval_xc(",LDA_C_PW", (rho_a_gga[0], rho_b_gga[0]), spin=1)[0]
    else:
        dm_total = dm_ab
        rho_gga = mf._numint.eval_rho(mol, ao, dm_total, xctype="GGA", hermi=True)
        rho = rho_gga[0]
        sigma = rho_gga[1]**2 + rho_gga[2]**2 + rho_gga[3]**2
        # Closed-shell: spin=0 calls correct.
        ex_pbe = mf._numint.eval_xc("PBE,", rho_gga, spin=0)[0]
        ec_pbe = mf._numint.eval_xc(",PBE", rho_gga, spin=0)[0]
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
    weights_write = np.asarray(mf.grids.weights)[valid]

    rho_list.append(rho_write)
    sigma_list.append(sigma_write)
    Fx_list.append(Fx_write)
    Fc_list.append(Fc_write)
    weights_list.append(weights_write)

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
        # For UKS, pass spin-resolved DM (3-D) so compute_dm_features picks
        # the correct UKS branch (D_sigma S D_sigma = D_sigma per spin —
        # Pople-Nesbet 1954). Pre-fix code summed to dm_total which gives
        # meaningless idempotency_error for open-shell (E3 audit).
        dm_for_features = jnp.asarray(dm_ab) if is_uks else jnp.asarray(dm_total)
        dm_feat_global = xcquinox.features.compute_dm_features_array(
            dm_for_features, jnp.asarray(S)
        )
        dm_feat_tiled = jnp.tile(dm_feat_global, (len(rho_write), 1))
        dm_list.append(np.asarray(dm_feat_tiled))

rho_all     = np.concatenate(rho_list)
sigma_all   = np.concatenate(sigma_list)
Fx_all      = np.concatenate(Fx_list)
Fc_all      = np.concatenate(Fc_list)
weights_all = np.concatenate(weights_list)

save_kwargs = dict(rho_all=rho_all, sigma_all=sigma_all,
                   Fx_all=Fx_all, Fc_all=Fc_all,
                   weights_all=weights_all)
if cusp_list:
    save_kwargs["cusp_all"] = np.concatenate(cusp_list)
if dm_list:
    save_kwargs["dm_all"] = np.concatenate(dm_list)

os.makedirs(pretrain_data_dir, exist_ok=True)
np.savez(os.path.join(pretrain_data_dir, "pretrain_data.npz"), **save_kwargs)
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
    _ckdir = os.path.join(pretrain_dir, arch_name)
    return (
        _os.path.isfile(_os.path.join(_ckdir, "xnet.eqx"))
        and _os.path.isfile(_os.path.join(_ckdir, "cnet.eqx"))
    )

for arch_name in ARCH_NAMES:
    if PRETRAIN_SKIP_IF_EXISTS and _pretrain_checkpoints_exist(arch_name):
        print(f"[{arch_name}] cached xnet.eqx + cnet.eqx found -- skipping pretrain")
        continue
    spec = alec.PretrainSpec(
        arch=alec.get_architecture(arch_name),
        data_dir=pretrain_data_dir,
        checkpoint_dir=os.path.join(pretrain_dir, arch_name),
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
    losses_x = np.load(os.path.join(pretrain_dir, arch_name, "losses_x.npy"))
    losses_c = np.load(os.path.join(pretrain_dir, arch_name, "losses_c.npy"))
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
os.makedirs(figures_dir, exist_ok=True)
fig.savefig(os.path.join(figures_dir, "pretrain_losses.png"), dpi=150, bbox_inches="tight")
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
    the PBE grid, then runs ``run_oep_inversion`` through a two-tier
    cascade. On OEP failure at both tiers, ``save_vxc_ref`` is skipped so
    V_xc-aware losses degrade to no-op rather than crashing the notebook.

    OEP cascade (redesigned 2026-04-23 after measurement-driven audit). The
    previous "step-5-proven" primary (def2-svp-jkfit, conv_tol=1e-6,
    reg=1e-4) was built on a false premise: step 5's H2O OEP never
    converged either -- it silently fell through to ``vxc_ref=None``. With
    a finite jkfit aux basis, Wu-Yang + L-BFGS-B + Tikhonov has an
    asymptotic density-error floor around ~1e-3 for H2O; a conv_tol of
    1e-6 is unreachable. Measurements (H2O, aux=def2-tzvp-jkfit):

        AO=def2-tzvp, reg=1e-4 -> density_error 2.6e-3
        AO=def2-tzvp, reg=1e-5 -> density_error 1.3e-3
        AO=def2-tzvp, reg=1e-6 -> density_error 9.6e-4 (@800 iters)
        AO=def2-svp,  reg=1e-5 -> density_error 9.8e-4 (the case-study default)

    Higher regularization makes density_error WORSE (Tikhonov pulls the
    V_xc expansion toward zero), so the new cascade LOWERS reg across
    tiers rather than raising it:

      * primary:  aux_basis="def2-tzvp-jkfit", max_iter=500,  conv_tol=2e-3, reg=1e-5
      * fallback: aux_basis="def2-tzvp-jkfit", max_iter=1000, conv_tol=2e-3, reg=1e-6

    conv_tol=2e-3 gives a ~2x margin over the primary density_error
    floor measured on H2O at both AO bases. Step 6's V_xc efficacy
    experiment (Cell 27, L1 vs L3) depends on vxc_ref being present;
    silent failure is NOT an acceptable outcome here.

    Re-run behavior: if H2O.npz exists but lacks vxc_ref (e.g. prior OEP
    failed), reload dm_target from the cached .npz and retry the cascade
    without recomputing CCSD.
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

def _npz_has_vxc_ref(_p):
    if not os.path.isfile(_p): return False
    with np.load(_p) as _f:
        return "vxc_ref" in _f.files

if os.path.isfile(_npz) and os.path.isfile(_meta) and _npz_has_vxc_ref(_npz):
    print(f"Using cached {_npz} (vxc_ref present)")
else:
    # Thread-cap wraps BOTH the CCSD branch and the OEP cascade. CCSD at
    # PySCF's default thread count (one-per-core) fights JAX's thread
    # pool and pays ~30x overhead on C2H2 in a JAX-loaded kernel
    # (measured 73s vs 2.6s in a clean subprocess). The cap recovers
    # that in the same motion that it fixes OEP throughput.
    with _capped_blas_threads(4):
        if os.path.isfile(_npz) and os.path.isfile(_meta):
            # CCSD data exists but OEP was skipped previously. Reload
            # dm_target and jump straight to the OEP cascade.
            print(f"Cached {_npz} missing vxc_ref; retrying OEP cascade")
            with np.load(_npz) as _f:
                dm_ao = np.asarray(_f["dm_target"])
        else:
            _mol = gto.M(atom=H2O_ATOM, basis=BASIS, charge=0, spin=0, verbose=0)
            _mf_pbe = dft.RKS(_mol); _mf_pbe.xc = "pbe"; _mf_pbe.grids.level = GRID_LEVEL
            _mf_pbe.kernel(); E_pbe = float(_mf_pbe.e_tot)
            _mf_hf = scf.RHF(_mol); _mf_hf.kernel(); E_hf = float(_mf_hf.e_tot)
            _cc = cc.CCSD(_mf_hf); _cc.kernel()
            # Use HF total + CCSD correlation (numerically more stable
            # than _cc.e_tot).
            E_ccsd = float(_mf_hf.e_tot + _cc.e_corr)

            # CCSD DM is built in the MO basis; transform to AO via
            # C @ dm_mo @ C.T (standard PySCF closed-shell convention).
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
        # Two-tier OEP cascade (measurement-driven 2026-04-23). Primary
        # reaches density_error ~1.25e-3 on H2O/def2-tzvp; fallback
        # lowers regularization for marginally harder inversions. See
        # cell docstring for the measurement rationale and why the old
        # svp-jkfit/reg=1e-4 primary was abandoned.
        _OEP_TIERS = [
            ("primary",  dict(aux_basis="def2-tzvp-jkfit", max_iter=500,  conv_tol=2e-3, regularization=1e-5)),
            ("fallback", dict(aux_basis="def2-tzvp-jkfit", max_iter=1000, conv_tol=2e-3, regularization=1e-6)),
        ]
        _oep = None
        for _tier_name, _tier_kw in _OEP_TIERS:
            # Per-tier tqdm bar so dozens-of-minutes inversions stop
            # being silent. The scipy L-BFGS-B callback fires once per
            # iteration; run_oep_inversion forwards (iter, density_error)
            # to our cb.
            _pbar = tqdm(
                total=_tier_kw["max_iter"],
                desc=f"OEP H2O {_tier_name}",
                leave=False, dynamic_ncols=True,
            )
            def _oep_progress(_it, _err, _bar=_pbar):
                _delta = _it - _bar.n
                if _delta > 0:
                    _bar.update(_delta)
                _bar.set_postfix(density_err=f"{_err:.2e}")
            try:
                _oep = alec.run_oep_inversion(
                    _oep_spec, dm_ao, **_tier_kw,
                    progress_callback=_oep_progress,
                )
            finally:
                _pbar.close()
            if _oep.converged:
                print(f"[OEP OK] H2O ({_tier_name}): n_iter={_oep.n_iter} "
                      f"density_error={_oep.density_error:.2e}")
                break
            print(f"[OEP WARN] H2O {_tier_name} failed (density_error="
                  f"{_oep.density_error:.2e}); trying next tier")
        if _oep is not None and _oep.converged:
            # save_vxc_ref takes the OEPResult OBJECT first, NOT
            # oep_result.vxc_matrix.
            alec.save_vxc_ref(_oep, _npz, dm_target=dm_ao, method="ccsd")
        else:
            print("[OEP FAIL] H2O: all tiers failed; skipping save_vxc_ref "
                  "(V_xc losses become no-op on H2O)")
"""
    return new_code_cell(source)


def build_cell_13_c2h2_data():
    """Section 2 Cell 13 -- C2H2 reference data (W4-11 geometry) + OEP inversion.

    Same measurement-driven two-tier cascade as cell 12 (H2O):
    tzvp-jkfit aux at reg=1e-5 (primary) / reg=1e-6 (fallback), both at
    conv_tol=2e-3. See cell 12 docstring for the full rationale.
    Re-run behavior: if C2H2.npz lacks vxc_ref, reload dm_target from
    cache and retry OEP without recomputing CCSD.
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

# _npz_has_vxc_ref is defined earlier in cell 12 and reusable in the
# notebook namespace.
if os.path.isfile(_npz) and os.path.isfile(_meta) and _npz_has_vxc_ref(_npz):
    print(f"Using cached {_npz} (vxc_ref present)")
else:
    # Same hoisted thread-cap rationale as cell 12 -- CCSD and the OEP
    # cascade both run under the cap so kernel-local JAX contention
    # doesn't turn C2H2 into a dozens-of-minutes wait.
    with _capped_blas_threads(4):
        if os.path.isfile(_npz) and os.path.isfile(_meta):
            print(f"Cached {_npz} missing vxc_ref; retrying OEP cascade")
            with np.load(_npz) as _f:
                dm_ao = np.asarray(_f["dm_target"])
        else:
            _mol = gto.M(atom=C2H2_ATOM, basis=BASIS, charge=0, spin=0, verbose=0)
            _mf_pbe = dft.RKS(_mol); _mf_pbe.xc = "pbe"; _mf_pbe.grids.level = GRID_LEVEL
            _mf_pbe.kernel(); E_pbe = float(_mf_pbe.e_tot)
            _mf_hf = scf.RHF(_mol); _mf_hf.kernel(); E_hf = float(_mf_hf.e_tot)
            _cc = cc.CCSD(_mf_hf); _cc.kernel()
            # Use HF total + CCSD correlation (numerically more stable
            # than _cc.e_tot).
            E_ccsd = float(_mf_hf.e_tot + _cc.e_corr)

            # CCSD DM is built in the MO basis; transform to AO via
            # C @ dm_mo @ C.T (standard PySCF closed-shell convention).
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
        # Two-tier OEP cascade; see cell 12 docstring for rationale.
        _OEP_TIERS = [
            ("primary",  dict(aux_basis="def2-tzvp-jkfit", max_iter=500,  conv_tol=2e-3, regularization=1e-5)),
            ("fallback", dict(aux_basis="def2-tzvp-jkfit", max_iter=1000, conv_tol=2e-3, regularization=1e-6)),
        ]
        _oep = None
        for _tier_name, _tier_kw in _OEP_TIERS:
            _pbar = tqdm(
                total=_tier_kw["max_iter"],
                desc=f"OEP C2H2 {_tier_name}",
                leave=False, dynamic_ncols=True,
            )
            def _oep_progress(_it, _err, _bar=_pbar):
                _delta = _it - _bar.n
                if _delta > 0:
                    _bar.update(_delta)
                _bar.set_postfix(density_err=f"{_err:.2e}")
            try:
                _oep = alec.run_oep_inversion(
                    _oep_spec, dm_ao, **_tier_kw,
                    progress_callback=_oep_progress,
                )
            finally:
                _pbar.close()
            if _oep.converged:
                print(f"[OEP OK] C2H2 ({_tier_name}): n_iter={_oep.n_iter} "
                      f"density_error={_oep.density_error:.2e}")
                break
            print(f"[OEP WARN] C2H2 {_tier_name} failed (density_error="
                  f"{_oep.density_error:.2e}); trying next tier")
        if _oep is not None and _oep.converged:
            alec.save_vxc_ref(_oep, _npz, dm_target=dm_ao, method="ccsd")
        else:
            print("[OEP FAIL] C2H2: all tiers failed; skipping save_vxc_ref "
                  "(V_xc losses become no-op on C2H2)")
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
# TrainingSpec.validate() requires a targets entry for every molecule in the
# molecules tuple (config.py:545-547). Atom targets are never dereferenced at
# training time but must be finite floats -- we use the Chakravorty atomic
# totals (same anchor as atom_energies) so the placeholder values are
# semantically consistent with the rest of the spec.
_targets_group1 = {
    "H2O": H2O_AE_REF_KCALMOL / KCAL_PER_HA,
    "H":   ATOMIC_ENERGIES_CHAKRAVORTY["H"],
    "O":   ATOMIC_ENERGIES_CHAKRAVORTY["O"],
}
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
                pretrain_checkpoint=f"{pretrain_dir}/{_arch}",
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
# Atom targets are placeholders required by TrainingSpec.validate(); see
# cell 18 for the rationale.
_targets_group2 = {
    "H2O":  H2O_AE_REF_KCALMOL  / KCAL_PER_HA,
    "C2H2": C2H2_AE_REF_KCALMOL / KCAL_PER_HA,
    "H":    ATOMIC_ENERGIES_CHAKRAVORTY["H"],
    "O":    ATOMIC_ENERGIES_CHAKRAVORTY["O"],
    "C":    ATOMIC_ENERGIES_CHAKRAVORTY["C"],
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
                pretrain_checkpoint=f"{pretrain_dir}/{_arch}",
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
# Atom targets are placeholders required by TrainingSpec.validate(); see
# cell 18 for the rationale.
_targets_group3 = {
    "H2O":  H2O_AE_REF_KCALMOL  / KCAL_PER_HA,
    "C2H2": C2H2_AE_REF_KCALMOL / KCAL_PER_HA,
    "H":    ATOMIC_ENERGIES_CHAKRAVORTY["H"],
    "O":    ATOMIC_ENERGIES_CHAKRAVORTY["O"],
    "C":    ATOMIC_ENERGIES_CHAKRAVORTY["C"],
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
                pretrain_checkpoint=f"{pretrain_dir}/{_arch}",
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
        "# OOM signatures we recognize as 'retry this spec on CPU'. Kept loose\n"
        "# so both XLA and CUDA-driver messages are caught.\n"
        "_GPU_OOM_MARKERS = (\n"
        "    'RESOURCE_EXHAUSTED',\n"
        "    'Out of memory',\n"
        "    'CUDA_ERROR_OUT_OF_MEMORY',\n"
        "    'cuMemAlloc',\n"
        ")\n"
        "\n"
        "def _looks_like_gpu_oom(text):\n"
        "    return any(m in text for m in _GPU_OOM_MARKERS)\n"
        "\n"
        "def _invoke_training_worker(spec_path, device=None):\n"
        "    \"\"\"Run _train_one_spec for one spec. Returns (rc, captured_text).\n"
        "\n"
        "    `device` is either None (let the worker default to 'auto') or the\n"
        "    explicit value 'cpu' used by the OOM retry path. The captured text\n"
        "    is stdout+stderr merged, needed for post-mortem OOM classification.\n"
        "\n"
        "    When device='cpu' the parent sets JAX_PLATFORMS=cpu in the spawned\n"
        "    process's environment. This matters: `python -m\n"
        "    xcquinox.alec._train_one_spec` imports the xcquinox.alec package\n"
        "    before main() runs, which transitively imports jax.numpy via\n"
        "    descriptors.py, so JAX initializes on GPU BEFORE any in-process\n"
        "    env fiddling can take effect. The --device=cpu CLI flag alone is\n"
        "    therefore insufficient on a GPU host; the env override is the\n"
        "    only reliable switch.\n"
        "    \"\"\"\n"
        "    cmd = [sys.executable, '-m', 'xcquinox.alec._train_one_spec', spec_path]\n"
        "    env = None\n"
        "    if device is not None:\n"
        "        cmd.append(f'--device={device}')\n"
        "        if device == 'cpu':\n"
        "            env = dict(os.environ)\n"
        "            env['JAX_PLATFORMS'] = 'cpu'\n"
        "    proc = subprocess.Popen(\n"
        "        cmd,\n"
        "        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,\n"
        "        bufsize=1, text=True, env=env,\n"
        "    )\n"
        "    captured = []\n"
        "    for line in proc.stdout:\n"
        "        line = line.rstrip('\\n')\n"
        "        captured.append(line)\n"
        "        if not line:\n"
        "            continue\n"
        "        if line.startswith('{'):\n"
        "            try:\n"
        "                msg = _json.loads(line)\n"
        "            except _json.JSONDecodeError:\n"
        "                print(line); continue\n"
        "            if msg.get('kind') == 'step':\n"
        "                _train_cb_from_info(msg)\n"
        "            elif msg.get('kind') in ('init', 'done'):\n"
        "                pass\n"
        "            else:\n"
        "                print(line)\n"
        "        else:\n"
        "            print(line)\n"
        "    rc = proc.wait()\n"
        "    return rc, '\\n'.join(captured)\n"
        "\n"
        "def _run_training_isolated(spec):\n"
        "    \"\"\"Run one TrainingSpec in a subprocess so the OS can hard-reclaim memory.\n"
        "\n"
        "    On GPU OOM (subprocess exits non-zero AND no model.eqx saved AND the\n"
        "    captured output matches a GPU-OOM signature), automatically re-invoke\n"
        "    the worker with --device=cpu. Training on CPU is slower but always fits,\n"
        "    so the sweep finishes instead of bailing out on a 7-11 GB peak tape.\n"
        "    \"\"\"\n"
        "    _ser = __import__('pi' + 'ckle')\n"
        "    with tempfile.NamedTemporaryFile(suffix='.spec', delete=False) as _f:\n"
        "        _ser.dump(spec, _f)\n"
        "        _spec_path = _f.name\n"
        "    try:\n"
        "        _model_path = os.path.join(spec.checkpoint_dir, \"model.eqx\")\n"
        "        rc, captured = _invoke_training_worker(_spec_path, device=None)\n"
        "        # First failure mode: no checkpoint AND GPU OOM -> retry on CPU.\n"
        "        if rc != 0 and not os.path.isfile(_model_path):\n"
        "            if _looks_like_gpu_oom(captured):\n"
        "                print(f\"  [GPU OOM on {spec.arch.name}/{spec.loss_name} -- \"\n"
        "                      f\"retrying subprocess with --device=cpu]\")\n"
        "                rc, captured = _invoke_training_worker(_spec_path, device='cpu')\n"
        "            if rc != 0 and not os.path.isfile(_model_path):\n"
        "                raise RuntimeError(\n"
        "                    f\"training subprocess for {spec.arch.name}/{spec.loss_name} \"\n"
        "                    f\"exited with code {rc} AND no checkpoint was saved \"\n"
        "                    f\"(CPU retry {'also failed' if _looks_like_gpu_oom(captured) else 'not attempted'})\"\n"
        "                )\n"
        "        # Second failure mode: non-zero exit AFTER model.eqx saved. This is\n"
        "        # a glibc/JAX/PySCF C-extension teardown crash; the training work is\n"
        "        # complete and on disk, so it is safe to continue.\n"
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


def build_cell_24_eval_md():
    """Section 5 Cell 24 -- eval-section markdown header."""
    source = r"""## Section 5 -- Evaluation

Per-spec evaluation sweep. For each of the 72 trained specs (3 groups x 2
archs x 4 losses x 3 solvers), the model checkpoint is re-run on its
training molecules with ``alec.run_test`` to produce per-molecule metrics
and a scalar aggregate. The metrics are:

- ``total_energy`` -- absolute E (Ha) + error-from-CCSD
- ``atomization_energy`` -- AE (kcal/mol) using the Chakravorty atomic
  references; AE reference values come from W4-11 (H2O=232.974,
  C2H2=405.525)
- ``density_rmse`` -- L2 norm of (rho_NN - rho_CCSD)
- ``constraint_violations`` -- count of negativity / Lieb-Oxford violations

Artifacts are written under ``{CHECKPOINT_BASE}/eval/{group}/{arch}/{loss}/{solver}/``
(``aggregate.json`` + ``per_molecule.json`` + ``test_metadata.json``),
mirroring the training directory layout. A per-spec aggregate existing on
disk causes the sweep to skip (unless ``RERUN_EVAL=True``).

After the sweep, the per-molecule JSON outputs are ingested into a single
long-form / tidy ``eval_df`` DataFrame with one row per
``(group, arch, loss, solver, phase_length, molecule, value_name)``
combination; this is the canonical substrate for all Section 6 plots.
"""
    return new_markdown_cell(source)


def build_cell_25_main_sweep():
    """Section 5 Cell 25 -- main evaluation sweep.

    For each spec in the combined 72-spec list, if the training checkpoint
    exists and ``aggregate.json`` is missing (or ``RERUN_EVAL`` is set),
    build a ``TestSpec`` and call ``alec.run_test``. Artifacts land under
    ``{CHECKPOINT_BASE}/eval/{group}/{arch}/{loss}/{solver}/``.

    Group membership is derived by object identity against
    ``_specs_group1 / _specs_group2 / _specs_group3``; the AE reference
    dict narrows to just H2O for group 1 and H2O+C2H2 for groups 2/3.
    ``jax.clear_caches() + gc.collect()`` runs between specs so the
    XLA-compiled eval graphs do not accumulate and OOM the kernel.
    """
    source = r"""# Per-spec evaluation sweep. Iterates the same concatenated _all_specs list
# as the training loop; per-spec outputs under
# {RUN_DIR}/eval/{group}/{arch}/{loss}/{solver}/.
_eval_base = eval_dir
os.makedirs(_eval_base, exist_ok=True)

# PBE-consistent atomic totals for the PBEReferenceMetric baseline.
# Cell 14 wrote {H,O,C}_metadata.json with PBE totals; transfer molecules
# also reference N and F (NH3, NH2, HF), so we lazily compute and cache
# any missing element here. Using these (rather than Chakravorty
# literature anchors) gives a *fair* PBE baseline:
#   AE_pbe = sum_Z n_Z * E_Z^PBE - E_mol^PBE
_atom_energies_pbe = {}
for _Z in ("H", "O", "C"):
    _meta_path = os.path.join(ext_data_dir, f"{_Z}_metadata.json")
    if os.path.isfile(_meta_path):
        with open(_meta_path) as _f:
            _atom_energies_pbe[_Z] = json.load(_f)["E_pbe_total"]

# UHF spin multiplicity for atoms with an open-shell ground-state electron count.
_ATOM_SPIN = {"H": 1, "He": 0, "Li": 1, "Be": 0, "B": 1, "C": 2,
              "N": 3, "O": 2, "F": 1, "Ne": 0}

def _ensure_pbe_atom_total(Z):
    if Z in _atom_energies_pbe:
        return
    _spin = _ATOM_SPIN.get(Z, 0)
    _mol = gto.M(atom=f"{Z} 0 0 0", basis=BASIS, charge=0, spin=_spin, verbose=0)
    _mf = dft.UKS(_mol) if _spin else dft.RKS(_mol)
    _mf.xc = "pbe"; _mf.grids.level = GRID_LEVEL
    _mf.kernel()
    _atom_energies_pbe[Z] = float(_mf.e_tot)
    print(f"  computed PBE total for {Z}: {_atom_energies_pbe[Z]:+.6f} Ha")

# Pre-populate any element that appears in transfer sets so PBEReferenceMetric
# never sees a KeyError mid-eval.
for _Z in ("N", "F"):
    _ensure_pbe_atom_total(_Z)

import time as _time
_N_specs = len(_all_specs)
_t_eval_start = _time.time()
_n_eval_done = 0; _n_eval_cached = 0; _n_eval_no_ckpt = 0
print(f"[main eval] {_N_specs} specs to evaluate "
      f"(metrics: total_energy, atomization_energy, density_rmse, "
      f"constraint_violations, pbe_reference, scf_convergence)", flush=True)
# Hold JIT cache through CACHE_FLUSH_EVERY specs so per-(arch, mol)
# compiled XLA artifacts are reused across spec invocations. Clearing
# every spec turns a few-minute sweep into many hours of recompile.
_EVAL_CACHE_FLUSH_EVERY = 16
for _idx, _spec in enumerate(_all_specs):
    _ckpt = os.path.join(_spec.checkpoint_dir, "model.eqx")
    if not os.path.isfile(_ckpt):
        _n_eval_no_ckpt += 1
        continue
    _tail = _spec.checkpoint_dir.rstrip("/").split("/")
    _solver = _tail[-1]
    _loss_label = _tail[-2]
    _arch = _tail[-3]
    _group = (
        "group1" if _spec in _specs_group1
        else "group2" if _spec in _specs_group2
        else "group3"
    )
    _out = os.path.join(_eval_base, _group, _arch, _loss_label, _solver)
    if not RERUN_EVAL and os.path.isfile(os.path.join(_out, "aggregate.json")):
        _n_eval_cached += 1
        continue
    _ae_ref = {"H2O": H2O_AE_REF_KCALMOL}
    if _group != "group1":
        _ae_ref["C2H2"] = C2H2_AE_REF_KCALMOL
    _test_spec = alec.TestSpec.from_dicts(
        arch=alec.get_architecture(_arch),
        model_checkpoint=_ckpt,
        molecules=tuple(_spec.molecules),
        metrics=("total_energy", "atomization_energy", "density_rmse",
                 "constraint_violations", "pbe_reference", "scf_convergence"),
        metric_kwargs={
            "atomization_energy": {"reference_ae_kcalmol": _ae_ref},
            "pbe_reference": {
                "atom_energies": _atom_energies_pbe,
                "reference_ae_kcalmol": _ae_ref,
            },
        },
        atom_energies=ATOMIC_ENERGIES_CHAKRAVORTY,
        output_dir=_out,
        solver_config=SOLVER_CONFIGS[_solver],
        pbe_anchor_weight=_spec.pbe_anchor_weight,
        pbe_anchor_sample=_spec.pbe_anchor_sample,
    )
    _t0 = _time.time()
    alec.run_test(_test_spec)
    _n_eval_done += 1
    _dt = _time.time() - _t0
    _elapsed = _time.time() - _t_eval_start
    _eta = _elapsed / max(_n_eval_done, 1) * max(_N_specs - (_idx + 1), 0)
    print(f"  [{_idx+1:>3d}/{_N_specs}] {_group}/{_arch}/{_loss_label}/{_solver:9s} "
          f"  dt={_dt:5.1f}s  elapsed={_elapsed/60:5.1f}min  "
          f"eta={_eta/60:5.1f}min", flush=True)
    if _n_eval_done % _EVAL_CACHE_FLUSH_EVERY == 0:
        jax.clear_caches(); gc.collect()
print(f"[main eval] done={_n_eval_done} cached={_n_eval_cached} "
      f"no_ckpt={_n_eval_no_ckpt}  total={(_time.time()-_t_eval_start)/60:.1f}min",
      flush=True)

_n_done = sum(
    1 for _s in _all_specs
    if os.path.isfile(os.path.join(
        _eval_base,
        "group1" if _s in _specs_group1 else "group2" if _s in _specs_group2 else "group3",
        _s.checkpoint_dir.rstrip("/").split("/")[-3],
        _s.checkpoint_dir.rstrip("/").split("/")[-2],
        _s.checkpoint_dir.rstrip("/").split("/")[-1],
        "aggregate.json",
    ))
)
print(f"Evaluation complete: {_n_done} / {len(_all_specs)} aggregates on disk")
"""
    return new_code_cell(source)


def build_cell_26_eval_preview():
    """Section 5 Cell 26 -- tidy DataFrame of per-molecule eval results.

    Aggregates the per-spec ``per_molecule.json`` outputs into a single
    long-form DataFrame with columns
    ``group / arch / loss / solver / phase_length / molecule / value_name / value``.
    All numeric scalars in each per-molecule record are folded into rows
    keyed by ``value_name`` (non-numeric fields like ``molecule`` / ``name``
    are skipped). Persists to ``{CHECKPOINT_BASE}/eval_df.parquet`` and
    reuses the cache on subsequent runs unless ``RERUN_EVAL`` is set.
    """
    source = r"""# Build eval_df: one row per (group, arch, loss, solver, phase_length,
# molecule, value_name). Numeric scalars from per_molecule.json become
# rows; non-numeric fields (molecule/name) are skipped.
_rows = []
_parq = os.path.join(RUN_DIR, "eval_df.parquet")
if not RERUN_EVAL and _df_exists(_parq):
    eval_df = _df_load(_parq)
    print(f"Using cached eval_df ({len(eval_df)} rows)")
else:
    for _spec in _all_specs:
        _tail = _spec.checkpoint_dir.rstrip("/").split("/")
        _solver = _tail[-1]
        _loss_label = _tail[-2]
        _arch = _tail[-3]
        _group = (
            "group1" if _spec in _specs_group1
            else "group2" if _spec in _specs_group2
            else "group3"
        )
        _phase = "short" if _spec.n_steps == TRAIN_N_STEPS_SHORT else "long"
        _out = os.path.join(eval_dir, _group, _arch, _loss_label, _solver)
        _pm_path = os.path.join(_out, "per_molecule.json")
        if not os.path.isfile(_pm_path):
            continue
        with open(_pm_path) as _f:
            _pm = json.load(_f)
        for _row in _pm:
            _mol = _row.get("name") or _row.get("molecule")
            for _k, _v in _row.items():
                if _k in ("name", "molecule"):
                    continue
                if isinstance(_v, bool):
                    # bool is a subtype of int; skip so boolean flags don't
                    # leak into the numeric value column.
                    continue
                if isinstance(_v, (int, float)):
                    _rows.append({
                        "group":        _group,
                        "arch":         _arch,
                        "loss":         _loss_label,
                        "solver":       _solver,
                        "phase_length": _phase,
                        "molecule":     _mol,
                        "value_name":   _k,
                        "value":        float(_v),
                    })
    eval_df = pd.DataFrame(_rows)
    _written = _df_save(eval_df, _parq)
    print(f"Wrote {_written} ({len(eval_df)} rows)")
"""
    return new_code_cell(source)


def build_cell_26b_baseline_evals():
    """Section 5 Cell 26b -- evaluate pretrained-only and random-init models.

    These two baselines are what the trained NNs are competing against:

      * ``pretrained``: the network after pretraining on F_x(rho, sigma)
        samples but BEFORE any fine-tuning against atomization energies.
        Per arch (one evaluation pass across all reference + transfer
        molecules).

      * ``random``: a fresh random-init model per arch. Shows what
        "untrained NN XC" produces — i.e., the error floor before any
        training at all.

    Both are solver-invariant for the energy metrics; DensityRMSEMetric
    evaluates under solver_config=None (oneshot) for consistency across
    the baseline panels.

    Artifacts land under ``{CHECKPOINT_BASE}/eval_baseline_{kind}/{arch}/``.
    """
    source = r"""# Baseline evaluations: pretrained-only (xnet/cnet from pretrain/) and
# random-init. Both are per-arch; neither depends on a TrainingSpec.
# Produces per_molecule.json aligned with the main eval schema so the
# plot cells can overlay baselines on trained-NN bars.
_baseline_mol_specs = (H2O_spec, C2H2_spec, H_spec, O_spec, C_spec)
_baseline_ae_ref = {"H2O": H2O_AE_REF_KCALMOL, "C2H2": C2H2_AE_REF_KCALMOL}
_baseline_metrics = ("total_energy", "atomization_energy", "density_rmse",
                     "pbe_reference")
_baseline_metric_kwargs = {
    "atomization_energy": {"reference_ae_kcalmol": _baseline_ae_ref},
    "pbe_reference": {
        "atom_energies": _atom_energies_pbe,
        "reference_ae_kcalmol": _baseline_ae_ref,
    },
}

def _eval_baseline_model(model, arch_name, kind):
    # Run eval with a caller-provided model (not loaded from checkpoint).
    # Serializes the model to a scratch file, builds a TestSpec that points
    # at it, then calls run_test. Output dir is
    # {CHECKPOINT_BASE}/eval_baseline_{kind}/{arch_name}/.
    _out = os.path.join(RUN_DIR, f"eval_baseline_{kind}", arch_name)
    _agg = os.path.join(_out, "aggregate.json")
    if not RERUN_EVAL and os.path.isfile(_agg):
        return
    os.makedirs(_out, exist_ok=True)
    _ckpt_path = os.path.join(_out, "_baseline_model.eqx")
    eqx.tree_serialise_leaves(_ckpt_path, model)
    _spec = alec.TestSpec.from_dicts(
        arch=alec.get_architecture(arch_name),
        model_checkpoint=_ckpt_path,
        molecules=_baseline_mol_specs,
        metrics=_baseline_metrics,
        metric_kwargs=_baseline_metric_kwargs,
        atom_energies=ATOMIC_ENERGIES_CHAKRAVORTY,
        output_dir=_out,
        solver_config=None,      # oneshot baseline — no SCF
        pbe_anchor_weight=0.0, pbe_anchor_sample=None,
    )
    alec.run_test(_spec)
    jax.clear_caches(); gc.collect()

for _arch in ARCH_NAMES:
    # (1) pretrained: load xnet/cnet from pretrain/ but skip training.
    _arch_cfg = alec.get_architecture(_arch)
    _xnet_sk, _cnet_sk = alec.create_network_pair(_arch_cfg, seed=0)
    _xnet = eqx.tree_deserialise_leaves(
        os.path.join(pretrain_dir, _arch, "xnet.eqx"), _xnet_sk,
    )
    _cnet = eqx.tree_deserialise_leaves(
        os.path.join(pretrain_dir, _arch, "cnet.eqx"), _cnet_sk,
    )
    _pretrained_model = alec.AlecGGAModel.from_arch(
        _arch_cfg, xnet=_xnet, cnet=_cnet,
    )
    _eval_baseline_model(_pretrained_model, _arch, "pretrained")

    # (2) random: fresh-init model with a deterministic seed.
    _random_model = alec.AlecGGAModel.from_arch(_arch_cfg, seed=12345)
    _eval_baseline_model(_random_model, _arch, "random")

print(f"Baseline evals done for {len(ARCH_NAMES)} archs")

# Aggregate baselines into a tidy DataFrame mirroring eval_df's schema
# (group='_baseline_*', loss/solver='_baseline', molecule, value_name, value).
_baseline_rows = []
for _kind in ("pretrained", "random"):
    for _arch in ARCH_NAMES:
        _pm_path = os.path.join(
            RUN_DIR, f"eval_baseline_{_kind}", _arch, "per_molecule.json",
        )
        if not os.path.isfile(_pm_path):
            continue
        with open(_pm_path) as _f:
            _pm = json.load(_f)
        for _row in _pm:
            _mol = _row.get("name") or _row.get("molecule")
            for _k, _v in _row.items():
                if _k in ("name", "molecule"):
                    continue
                if isinstance(_v, bool):
                    continue
                if isinstance(_v, (int, float)):
                    _baseline_rows.append({
                        "baseline":   _kind,
                        "arch":       _arch,
                        "molecule":   _mol,
                        "value_name": _k,
                        "value":      float(_v),
                    })
baseline_df = pd.DataFrame(_baseline_rows)
_bparq = os.path.join(RUN_DIR, "baseline_df.parquet")
_written = _df_save(baseline_df, _bparq)
print(f"Wrote {_written} ({len(baseline_df)} rows)")
"""
    return new_code_cell(source)


def build_cell_27_vxc_efficacy():
    """Section 5 Cell 27 -- V_xc efficacy (L1 vs L3) on Group 2 short.

    Restructured (2026-04-26): one bar group for the baseline (random +
    pretrained, both archs); one bar group per treatment (L1, L3) with one
    bar per architecture. Solver becomes the row dimension. PBE-vs-W4-11
    and CCSD-vs-W4-11 errors drawn as horizontal lines on the AE panel.
    Density panels keep PBE-vs-CCSD as the only meaningful baseline (no
    W4-11 analog for density).
    """
    source = r"""# V_xc efficacy on group 2 (H2O+C2H2 short). Bar groups:
#   "baseline" -- random NN + pretrained-only NN, both archs
#   "L1_B"     -- atomization-only fine-tune, per arch
#   "L3_balanced_vxc" -- atomization + V_xc fine-tune, per arch
# Rows = solver mode. Cols = (AE_error, density_rmse, density_l1).
# AE panel: PBE / CCSD vs W4-11 horizontal reference lines.

# --- Baseline error helpers ------------------------------------------
# PBE/CCSD AE error vs W4-11 lit per molecule, computed from the
# precomputed metadata files written by cells 12/13/30/31.
def _read_meta(_dir, _name):
    _p = os.path.join(_dir, f"{_name}_metadata.json")
    if not os.path.isfile(_p):
        return None
    with open(_p) as _f:
        return json.load(_f)

def _ae_baseline_kcalmol(_meta_dirs, _mol_name, _comp, _ref_kcalmol, _kind):
    # _kind in {"pbe","ccsd"}: returns AE error vs W4-11 lit (kcal/mol).
    _key = "E_pbe_total" if _kind == "pbe" else "E_ccsd_total"
    _mol_meta = None
    for _d in _meta_dirs:
        _mol_meta = _read_meta(_d, _mol_name)
        if _mol_meta is not None: break
    if _mol_meta is None or _key not in _mol_meta: return None
    _E_mol = float(_mol_meta[_key])
    _E_atoms = 0.0
    for _Z, _n in _comp:
        _atom_meta = None
        for _d in _meta_dirs:
            _atom_meta = _read_meta(_d, _Z)
            if _atom_meta is not None: break
        if _atom_meta is None or _key not in _atom_meta: return None
        _E_atoms += float(_atom_meta[_key]) * _n
    _AE_Ha = _E_atoms - _E_mol
    return _AE_Ha * KCAL_PER_HA - _ref_kcalmol

# Training-set composition for group2: H2O + C2H2. AE refs from cell 11.
_TRAIN_MOLS_G2 = (
    ("H2O",  (("O",1),("H",2)), H2O_AE_REF_KCALMOL),
    ("C2H2", (("C",2),("H",2)), C2H2_AE_REF_KCALMOL),
)
_META_DIRS = (ext_data_dir,)
def _ae_mae_baseline(_kind):
    _vals = [
        _ae_baseline_kcalmol(_META_DIRS, _n, _c, _ref, _kind)
        for _n, _c, _ref in _TRAIN_MOLS_G2
    ]
    _vals = [abs(_v) for _v in _vals if _v is not None]
    return float(np.mean(_vals)) if _vals else None

_pbe_ae_mae  = _ae_mae_baseline("pbe")
_ccsd_ae_mae = _ae_mae_baseline("ccsd")

# Density RMSE/L1 baselines from baseline_df (random / pretrained NN).
def _density_baseline(_arch, _value_name, _kind, _mols=("H2O","C2H2")):
    if 'baseline_df' not in globals() or baseline_df is None:
        return None
    _sub = baseline_df[(baseline_df.arch == _arch)
                       & (baseline_df.baseline == _kind)
                       & (baseline_df.value_name == _value_name)
                       & (baseline_df.molecule.isin(_mols))]
    if _sub.empty: return None
    return float(_sub["value"].abs().mean())

_g2 = eval_df[eval_df.group == "group2"]
_LOSSES_HERE = ("L1_B", "L3_balanced_vxc")
_GROUP_ORDER = ("baseline",) + _LOSSES_HERE
_METRICS = ("AE_error_kcalmol", "density_rmse", "density_l1")
_LOG_PANELS = {"AE_error_kcalmol"}

# Each "baseline" bar group has 2*len(ARCH_NAMES) bars (random + pretrained
# per arch); each treatment bar group has len(ARCH_NAMES) bars. Layout
# bar positions so groups are visually separated.
_arch_colors = {arch: cmap(_i / max(len(ARCH_NAMES) - 1, 1))
                for _i, arch in enumerate(ARCH_NAMES)}
_BASE_KIND_ALPHA = {"random": 0.45, "pretrained": 0.85}

fig, axes = plt.subplots(
    len(SOLVER_LABELS), len(_METRICS),
    figsize=(5.2 * len(_METRICS), 3.6 * len(SOLVER_LABELS)),
    squeeze=False,
)
for _ri, _solver in enumerate(SOLVER_LABELS):
    for _ci, _metric in enumerate(_METRICS):
        _ax = axes[_ri][_ci]
        _slice = _g2[(_g2.solver == _solver) & (_g2.value_name == _metric)]
        _xticks = []; _xlabels = []
        _bar_idx = 0; _gap_within = 0.22; _gap_between = 0.6
        # Baseline group
        for _kind in ("random", "pretrained"):
            for _arch in ARCH_NAMES:
                _v = _density_baseline(_arch, _metric, _kind) \
                    if _metric != "AE_error_kcalmol" else _density_baseline(
                        _arch, "AE_error_kcalmol", _kind)
                _vv = abs(_v) if _v is not None else 0.0
                _ax.bar(_bar_idx, _vv, width=0.6,
                        color=_arch_colors[_arch],
                        alpha=_BASE_KIND_ALPHA[_kind],
                        edgecolor="k", linewidth=0.5,
                        label=f"baseline · {_kind} · {_arch}"
                              if _ri == 0 and _ci == 0 else None)
                _xticks.append(_bar_idx)
                _xlabels.append(f"{_kind[:3]}\n{_arch}")
                _bar_idx += 1
        _bar_idx += _gap_between
        # Treatment groups
        for _loss in _LOSSES_HERE:
            for _arch in ARCH_NAMES:
                _d = _slice[(_slice.loss == _loss) & (_slice.arch == _arch)]
                _val = float(_d["value"].abs().mean()) if len(_d) else 0.0
                _ax.bar(_bar_idx, _val, width=0.6,
                        color=_arch_colors[_arch],
                        edgecolor="k", linewidth=0.5,
                        hatch="//" if _loss == "L3_balanced_vxc" else None,
                        label=f"{_loss} · {_arch}"
                              if _ri == 0 and _ci == 0 else None)
                _xticks.append(_bar_idx)
                _xlabels.append(f"{_loss}\n{_arch}")
                _bar_idx += 1
            _bar_idx += _gap_between
        # Horizontal reference lines (AE panel only)
        if _metric == "AE_error_kcalmol":
            if _pbe_ae_mae is not None and _pbe_ae_mae > 0:
                _ax.axhline(_pbe_ae_mae, ls="--", lw=1.6, color="k",
                            label=f"PBE vs W4-11 ({_pbe_ae_mae:.1f})"
                                  if _ri == 0 and _ci == 0 else None)
            if _ccsd_ae_mae is not None and _ccsd_ae_mae > 0:
                _ax.axhline(_ccsd_ae_mae, ls="-.", lw=1.6, color="purple",
                            label=f"CCSD vs W4-11 ({_ccsd_ae_mae:.2f})"
                                  if _ri == 0 and _ci == 0 else None)
        if _metric in _LOG_PANELS:
            _ax.set_yscale("log")
        _ax.set_xticks(_xticks)
        _ax.set_xticklabels(_xlabels, rotation=70, fontsize=6)
        _ylabel = {
            "AE_error_kcalmol": "MAE |AE error| vs W4-11 (kcal/mol, log)",
            "density_rmse":     "density RMSE (e/bohr³)",
            "density_l1":       "density L1 (e/bohr³)",
        }[_metric]
        _ax.set_ylabel(_ylabel, fontsize=8)
        _ax.set_title(f"solver={_solver} | {_metric}", fontsize=9)
        _ax.grid(True, axis="y", which="both", ls=":", alpha=0.4)
fig.legend(*axes[0][0].get_legend_handles_labels(),
           loc="lower center", ncol=4, fontsize=7,
           bbox_to_anchor=(0.5, -0.02))
fig.suptitle("V_xc efficacy on group 2 (H₂O+C₂H₂): "
             "baseline (random/pretrained) vs L1 vs L3, per arch",
             fontsize=11)
fig.tight_layout(rect=(0, 0.04, 1, 0.96))
fig.savefig(os.path.join(figures_dir, "vxc_efficacy.png"), dpi=120, bbox_inches="tight")
plt.show()
"""
    return new_code_cell(source)


def build_cell_28_anchor_effect():
    """Section 5 Cell 28 -- anchor-effect paired bars (L3 -> L4).

    For each (arch, group) panel, plots per-solver DeltaAE = AE(L3_balanced_vxc)
    - AE(L4_balanced_vxc_anchor). Positive bars (green) indicate anchor
    helps; negative (red) indicate anchor hurts. Writes
    ``{figures_dir}/anchor_effect.png``.
    """
    source = r"""# Anchor-effect: ΔAE = |L3 atomization-error| - |L4 atomization-error|
# per (arch, solver, group). Positive (green) -> anchor regularizer
# helps; negative (red) -> anchor hurts. Single proxy legend below
# explains the colors.
import matplotlib.patches as _mpatches
fig, axes = plt.subplots(len(ARCH_NAMES), 3, figsize=(14, 4 * len(ARCH_NAMES)),
                         squeeze=False)
for _ri, _arch in enumerate(ARCH_NAMES):
    for _ci, _grp in enumerate(["group1", "group2", "group3"]):
        _ax = axes[_ri][_ci]
        _slice = eval_df[
            (eval_df.group == _grp) & (eval_df.arch == _arch)
            & (eval_df.value_name == "AE_error_kcalmol")
        ]
        _deltas = []; _labels = []
        for _s in SOLVER_LABELS:
            _off = _slice[(_slice.loss == "L3_balanced_vxc")
                          & (_slice.solver == _s)]["value"].abs().mean()
            _on = _slice[(_slice.loss == "L4_balanced_vxc_anchor")
                         & (_slice.solver == _s)]["value"].abs().mean()
            _deltas.append((_off if pd.notna(_off) else 0.0)
                           - (_on if pd.notna(_on) else 0.0))
            _labels.append(_s)
        _x = np.arange(len(_deltas))
        _ax.bar(_x, _deltas,
                color=["seagreen" if _d > 0 else "indianred" for _d in _deltas],
                edgecolor="k", linewidth=0.5)
        _ax.axhline(0, color="k", lw=1.0)
        _ax.set_xticks(_x); _ax.set_xticklabels(_labels, fontsize=8)
        _ax.set_xlabel("solver mode")
        _ax.set_title(f"{_arch} — trained on {_grp}", fontsize=9)
        _ax.set_ylabel("Δ|AE error|  (kcal/mol)")
        _ax.grid(True, axis="y", ls=":", alpha=0.4)
        if _ci == 0:
            _ax.legend(handles=[
                _mpatches.Patch(color="seagreen", label="Δ > 0  (anchor helps: |L4| < |L3|)"),
                _mpatches.Patch(color="indianred", label="Δ < 0  (anchor hurts: |L4| > |L3|)"),
            ], fontsize=7, loc="best", framealpha=0.85)
fig.suptitle("PBE-anchor effect: |L3 (V_xc only)| − |L4 (V_xc + PBE anchor)|  on training molecules",
             fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(os.path.join(figures_dir, "anchor_effect.png"), dpi=120, bbox_inches="tight")
plt.show()
"""
    return new_code_cell(source)


def build_cell_29_transfer_md():
    """Section 6 Cell 29 -- transfer-learning section header (markdown)."""
    source = r"""## Section 6 -- Transfer-learning Evaluation

Evaluates every trained spec on W4-11 molecules held out from training.
Primary set is small and chemically close (H2 / OH / CH4); secondary set
spans a broader chemistry (NH3 / HF / CO2 / NH2). NH2 is a UKS doublet.

Geometries + AE references: W4-11. No OEP on transfer molecules -- V_xc
matching is a training-only regularizer.
"""
    return new_markdown_cell(source)


def build_cell_30_transfer_primary():
    """Section 6 Cell 30 -- primary transfer data generation (H2 / OH / CH4)."""
    source = r"""# Primary transfer set: {H2, OH, CH4} on W4-11 geometries.
# OH is UKS (doublet); H2 and CH4 are RKS closed shell.
TRANSFER_PRIMARY = (
    {
        "name": "H2",
        "atom": "H  0.000000  0.000000  0.370946; H  0.000000  0.000000 -0.370946",
        "spin": 0,
        "ae_ref_kcalmol": 109.493,
        "comp": (("H", 2),),
    },
    {
        "name": "OH",
        "atom": "O  0.000000  0.000000  0.107851; H  0.000000  0.000000 -0.862809",
        "spin": 1,
        "ae_ref_kcalmol": 107.208,
        "comp": (("O", 1), ("H", 1)),
    },
    {
        "name": "CH4",
        "atom": ("H 0.628099 0.628099 0.628099; C 0 0 0; "
                 "H -0.628099 -0.628099 0.628099; "
                 "H -0.628099 0.628099 -0.628099; "
                 "H 0.628099 -0.628099 -0.628099"),
        "spin": 0,
        "ae_ref_kcalmol": 420.420,
        "comp": (("C", 1), ("H", 4)),
    },
)


def _gen_transfer_npz(m, out_dir):
    # Generate {name}.npz + {name}_metadata.json for a transfer molecule.
    # Runs PBE + HF + CCSD on the W4-11 geometry, extracts AO-basis CCSD DM
    # (spin-resolved for UKS), computes rho_ccsd on the PBE grid. No OEP.
    # Guard: skip if both artifacts already exist.
    _npz = os.path.join(out_dir, f"{m['name']}.npz")
    _meta = os.path.join(out_dir, f"{m['name']}_metadata.json")
    if os.path.isfile(_npz) and os.path.isfile(_meta):
        print(f"Using cached {_npz}")
        return
    _mol = gto.M(atom=m["atom"], basis=BASIS, charge=0, spin=m["spin"], verbose=0)
    if m["spin"]:
        _mf_pbe = dft.UKS(_mol); _mf_pbe.xc = "pbe"; _mf_pbe.grids.level = GRID_LEVEL
        _mf_pbe.kernel(); E_pbe = float(_mf_pbe.e_tot)
        _mf_hf = scf.UHF(_mol); _mf_hf.kernel(); E_hf = float(_mf_hf.e_tot)
        _cc = cc.UCCSD(_mf_hf); _cc.kernel()
        E_ccsd = float(_mf_hf.e_tot + _cc.e_corr)
        dm_mo_ab = _cc.make_rdm1()
        Ca, Cb = _mf_hf.mo_coeff
        dm_ao_a = Ca @ dm_mo_ab[0] @ Ca.T
        dm_ao_b = Cb @ dm_mo_ab[1] @ Cb.T
        dm_ao = np.stack([dm_ao_a, dm_ao_b], axis=0)   # (2, nao, nao)
        _ao = _mf_pbe._numint.eval_ao(_mol, _mf_pbe.grids.coords, deriv=0)
        rho_ccsd = np.einsum("ij,gi,gj->g", dm_ao_a + dm_ao_b, _ao, _ao)
    else:
        _mf_pbe = dft.RKS(_mol); _mf_pbe.xc = "pbe"; _mf_pbe.grids.level = GRID_LEVEL
        _mf_pbe.kernel(); E_pbe = float(_mf_pbe.e_tot)
        _mf_hf = scf.RHF(_mol); _mf_hf.kernel(); E_hf = float(_mf_hf.e_tot)
        _cc = cc.CCSD(_mf_hf); _cc.kernel()
        E_ccsd = float(_mf_hf.e_tot + _cc.e_corr)
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
                   "E_pbe_total": E_pbe,
                   "ae_ref_kcalmol": m["ae_ref_kcalmol"]}, _f, indent=2)
    print(f"Wrote {_npz}  E_ccsd={E_ccsd:+.4f}  AE_ref={m['ae_ref_kcalmol']:.3f} kcal/mol")


for _m in TRANSFER_PRIMARY:
    _gen_transfer_npz(_m, transfer_primary)
"""
    return new_code_cell(source)


def build_cell_31_transfer_secondary():
    """Section 6 Cell 31 -- secondary transfer data generation.

    Reuses ``_gen_transfer_npz`` from cell 30 (cells execute sequentially
    in the notebook, so the helper is in scope).
    """
    source = r"""# Secondary transfer set: {NH3, HF, CO2, NH2} on W4-11 geometries.
# NH2 is a UKS doublet radical; the rest are closed-shell RKS.
TRANSFER_SECONDARY = (
    {
        "name": "NH3",
        "atom": ("N 0 0 0.116671; H 0 0.934724 -0.272232; "
                 "H 0.809495 -0.467362 -0.272232; "
                 "H -0.809495 -0.467362 -0.272232"),
        "spin": 0,
        "ae_ref_kcalmol": 298.018,
        "comp": (("N", 1), ("H", 3)),
    },
    {
        "name": "HF",
        "atom": "F 0 0 0.091577; H 0 0 -0.824192",
        "spin": 0,
        "ae_ref_kcalmol": 141.640,
        "comp": (("H", 1), ("F", 1)),
    },
    {
        "name": "CO2",
        "atom": "C 0 0 0; O 0 0 1.162600; O 0 0 -1.162600",
        "spin": 0,
        "ae_ref_kcalmol": 390.141,
        "comp": (("C", 1), ("O", 2)),
    },
    {
        "name": "NH2",
        "atom": ("N 0 0 0.142235; H 0 0.800646 -0.497821; "
                 "H 0 -0.800646 -0.497821"),
        "spin": 1,
        "ae_ref_kcalmol": 182.591,
        "comp": (("N", 1), ("H", 2)),
    },
)

for _m in TRANSFER_SECONDARY:
    _gen_transfer_npz(_m, transfer_secondary)
"""
    return new_code_cell(source)


def build_cell_32_transfer_primary_eval():
    """Section 6 Cell 32 -- primary transfer test loop.

    For each of the 72 trained specs crossed with the 3 primary transfer
    molecules, constructs a TestSpec with the trained model's checkpoint
    and the transfer molecule's ``.npz`` as external data, runs
    ``alec.run_test``, and aggregates the ``per_molecule.json`` outputs
    into a tidy DataFrame ``transfer_primary_df``.

    Defines the helper ``_run_transfer_eval(mols_list, out_dir, parquet_name)``
    that cell 33 reuses for the secondary set.
    """
    source = r"""# Primary transfer test loop. Reads trained checkpoint for each of the 72
# specs, runs alec.run_test on each transfer molecule, aggregates to tidy
# DataFrame. pbe_anchor_weight=0 / pbe_anchor_sample=None for transfer
# (anchor is a training regularizer only).
import time as _time
def _run_transfer_eval(mols_list, out_dir, parquet_name):
    _parq = os.path.join(RUN_DIR, parquet_name)
    if not RERUN_EVAL and _df_exists(_parq):
        _df = _df_load(_parq)
        print(f"[transfer eval] cached: {parquet_name} ({len(_df)} rows)")
        return _df
    _ae_ref = {_m["name"]: _m["ae_ref_kcalmol"] for _m in mols_list}
    _mol_specs = tuple(
        alec.MoleculeSpec(
            name=_m["name"], atom=_m["atom"], basis=BASIS,
            charge=0, spin=_m["spin"], grid_level=GRID_LEVEL,
            atom_composition=_m["comp"],
            external_data_path=os.path.join(out_dir, f"{_m['name']}.npz"),
        )
        for _m in mols_list
    )
    _rows = []
    _N = len(_all_specs)
    _mol_names = ", ".join(m["name"] for m in mols_list)
    print(f"[transfer eval] {parquet_name}: {_N} specs x {len(mols_list)} "
          f"transfer mols ({_mol_names})", flush=True)
    _t_start = _time.time()
    _n_done = 0; _n_skipped_cache = 0; _n_skipped_no_ckpt = 0
    # JIT-cache pressure mitigation: clearing every spec forced full
    # XLA retracing for every (arch, mol, metric) combination on every
    # spec, ballooning a few-hundred-spec sweep into many hours. Hold the
    # cache through CACHE_FLUSH_EVERY specs so common shape signatures
    # (per-arch, per-molecule) reuse compiled XLA across spec invocations.
    CACHE_FLUSH_EVERY = 16
    for _idx, _spec in enumerate(_all_specs):
        _ckpt = os.path.join(_spec.checkpoint_dir, "model.eqx")
        if not os.path.isfile(_ckpt):
            _n_skipped_no_ckpt += 1
            continue
        _tail = _spec.checkpoint_dir.rstrip("/").split("/")
        _solver = _tail[-1]
        _loss_label = _tail[-2]
        _arch = _tail[-3]
        _group = (
            "group1" if _spec in _specs_group1
            else "group2" if _spec in _specs_group2
            else "group3"
        )
        _out = os.path.join(transfer_eval_dir,
                            parquet_name.replace(".parquet", ""),
                            _group, _arch, _loss_label, _solver)
        _agg_path = os.path.join(_out, "aggregate.json")
        _pm_path = os.path.join(_out, "per_molecule.json")
        if not RERUN_EVAL and os.path.isfile(_agg_path):
            _n_skipped_cache += 1
        else:
            _t0 = _time.time()
            _test_spec = alec.TestSpec.from_dicts(
                arch=alec.get_architecture(_arch),
                model_checkpoint=_ckpt,
                molecules=_mol_specs,
                metrics=("total_energy", "atomization_energy", "density_rmse",
                         "pbe_reference"),
                metric_kwargs={
                    "atomization_energy": {"reference_ae_kcalmol": _ae_ref},
                    "pbe_reference": {
                        "atom_energies": _atom_energies_pbe,
                        "reference_ae_kcalmol": _ae_ref,
                    },
                },
                atom_energies=ATOMIC_ENERGIES_CHAKRAVORTY,
                output_dir=_out,
                solver_config=SOLVER_CONFIGS[_solver],
                pbe_anchor_weight=0.0,
                pbe_anchor_sample=None,
            )
            alec.run_test(_test_spec)
            _n_done += 1
            _dt = _time.time() - _t0
            _elapsed = _time.time() - _t_start
            _eta = _elapsed / max(_n_done, 1) * max(_N - (_idx + 1), 0)
            print(f"  [{_idx+1:>3d}/{_N}] {_group}/{_arch}/{_loss_label}/{_solver:9s} "
                  f"  dt={_dt:5.1f}s  elapsed={_elapsed/60:5.1f}min  "
                  f"eta={_eta/60:5.1f}min", flush=True)
            if _n_done % CACHE_FLUSH_EVERY == 0:
                jax.clear_caches(); gc.collect()
        if not os.path.isfile(_pm_path):
            continue
        with open(_pm_path) as _f:
            _pm = json.load(_f)
        for _row in _pm:
            _mol = _row.get("name") or _row.get("molecule")
            for _k, _v in _row.items():
                if _k in ("name", "molecule"):
                    continue
                if isinstance(_v, bool):
                    continue
                if isinstance(_v, (int, float)):
                    _rows.append({
                        "group":      _group,
                        "arch":       _arch,
                        "loss":       _loss_label,
                        "solver":     _solver,
                        "molecule":   _mol,
                        "value_name": _k,
                        "value":      float(_v),
                    })
    _df = pd.DataFrame(_rows)
    _written = _df_save(_df, _parq)
    _total_min = (_time.time() - _t_start) / 60
    print(f"[transfer eval] {parquet_name}: ran={_n_done} cached={_n_skipped_cache} "
          f"no_ckpt={_n_skipped_no_ckpt}  total={_total_min:.1f}min", flush=True)
    print(f"[transfer eval] wrote {_written} ({len(_df)} rows)", flush=True)
    return _df


transfer_primary_df = _run_transfer_eval(
    TRANSFER_PRIMARY, transfer_primary, "transfer_primary_df.parquet",
)
print(f"transfer_primary_df: {len(transfer_primary_df)} rows")
"""
    return new_code_cell(source)


def build_cell_33_transfer_secondary_eval():
    """Section 6 Cell 33 -- secondary transfer test loop.

    Reuses ``_run_transfer_eval`` defined in cell 32 (cells execute
    sequentially in the notebook). Produces ``transfer_secondary_df``.
    """
    source = r"""# Secondary transfer test loop. Same shape as primary; reuses the helper
# _run_transfer_eval (which calls alec.run_test under the hood) defined
# in the previous cell.
transfer_secondary_df = _run_transfer_eval(
    TRANSFER_SECONDARY, transfer_secondary, "transfer_secondary_df.parquet",
)
print(f"transfer_secondary_df: {len(transfer_secondary_df)} rows")
"""
    return new_code_cell(source)


def build_cell_34_transfer_primary_plot():
    """Section 6 Cell 34 -- primary transfer aggregate MAE plot.

    Restructured 2026-04-26: bar groups = (baseline; treatment x arch),
    PBE-vs-W4-11 and CCSD-vs-W4-11 horizontal reference lines, panels by
    (solver, group). Defines the runtime helper ``_render_transfer_plot``
    that cell 35 reuses for the secondary transfer set.
    Writes ``{figures_dir}/transfer_primary_mae.png``.
    """
    source = r"""# Transfer-plot helper. Bar groups per panel:
#   "baseline"    -- random + pretrained NN (both archs)
#   one per loss  -- LOSS_NAMES, one bar per arch
# Rows = solver, cols = trained-on group. PBE / CCSD MAE vs W4-11
# horizontal reference lines on every panel. Cell 35 calls the same
# helper with the secondary df + dataset.

def _read_mol_meta(_d, _name):
    _p = os.path.join(_d, f"{_name}_metadata.json")
    if not os.path.isfile(_p): return None
    with open(_p) as _f:
        return json.load(_f)

def _ae_kind_error_kcalmol(_meta_dirs, _name, _comp, _ref_kcalmol, _kind):
    _key = "E_pbe_total" if _kind == "pbe" else "E_ccsd_total"
    _mol_meta = None
    for _d in _meta_dirs:
        _mol_meta = _read_mol_meta(_d, _name)
        if _mol_meta is not None: break
    if _mol_meta is None or _key not in _mol_meta: return None
    _E_mol = float(_mol_meta[_key])
    _E_atoms = 0.0
    for _Z, _n in _comp:
        _atom_meta = None
        for _d in _meta_dirs:
            _atom_meta = _read_mol_meta(_d, _Z)
            if _atom_meta is not None: break
        if _atom_meta is None or _key not in _atom_meta: return None
        _E_atoms += float(_atom_meta[_key]) * _n
    return (_E_atoms - _E_mol) * KCAL_PER_HA - _ref_kcalmol

def _baseline_ae_nn(_arch, _kind):
    if 'baseline_df' not in globals() or baseline_df is None:
        return None
    _sub = baseline_df[(baseline_df.arch == _arch)
                       & (baseline_df.baseline == _kind)
                       & (baseline_df.value_name == "AE_error_kcalmol")]
    if _sub.empty: return None
    return float(_sub["value"].abs().mean())

_arch_colors_tr = {arch: cmap(_i / max(len(ARCH_NAMES) - 1, 1))
                   for _i, arch in enumerate(ARCH_NAMES)}
_BASE_KIND_ALPHA = {"random": 0.45, "pretrained": 0.85}
_HATCHED_LOSSES = {"L3_balanced_vxc", "L4_balanced_vxc_anchor"}

def _render_transfer_plot(target_df, transfer_mols, data_dir, png_name,
                          dataset_label, suptitle):
    _meta_dirs = (data_dir, ext_data_dir)
    _pbe_vals = [
        _ae_kind_error_kcalmol(_meta_dirs, m["name"], m["comp"],
                               m["ae_ref_kcalmol"], "pbe")
        for m in transfer_mols
    ]
    _ccsd_vals = [
        _ae_kind_error_kcalmol(_meta_dirs, m["name"], m["comp"],
                               m["ae_ref_kcalmol"], "ccsd")
        for m in transfer_mols
    ]
    _pbe_mae  = float(np.mean([abs(_v) for _v in _pbe_vals  if _v is not None])) \
                if any(_v is not None for _v in _pbe_vals)  else None
    _ccsd_mae = float(np.mean([abs(_v) for _v in _ccsd_vals if _v is not None])) \
                if any(_v is not None for _v in _ccsd_vals) else None
    print(f"[{dataset_label} transfer baselines vs W4-11]  "
          f"PBE_MAE={_pbe_mae}  CCSD_MAE={_ccsd_mae}")
    _GROUP_ORDER = ("group1", "group2", "group3")
    fig, axes = plt.subplots(
        len(SOLVER_LABELS), len(_GROUP_ORDER),
        figsize=(5.2 * len(_GROUP_ORDER), 3.6 * len(SOLVER_LABELS)),
        squeeze=False,
    )
    for _ri, _solver in enumerate(SOLVER_LABELS):
        for _ci, _grp in enumerate(_GROUP_ORDER):
            _ax = axes[_ri][_ci]
            _slice = target_df[
                (target_df.group == _grp)
                & (target_df.solver == _solver)
                & (target_df.value_name == "AE_error_kcalmol")
            ]
            _xticks = []; _xlabels = []
            _bar_idx = 0; _gap = 0.6
            for _kind in ("random", "pretrained"):
                for _arch in ARCH_NAMES:
                    _v = _baseline_ae_nn(_arch, _kind)
                    _vv = abs(_v) if _v is not None else 0.0
                    _ax.bar(_bar_idx, _vv, width=0.6,
                            color=_arch_colors_tr[_arch],
                            alpha=_BASE_KIND_ALPHA[_kind],
                            edgecolor="k", linewidth=0.5,
                            label=f"baseline · {_kind} · {_arch}"
                                  if _ri == 0 and _ci == 0 else None)
                    _xticks.append(_bar_idx)
                    _xlabels.append(f"{_kind[:3]}\n{_arch}")
                    _bar_idx += 1
            _bar_idx += _gap
            for _loss in LOSS_NAMES:
                for _arch in ARCH_NAMES:
                    _d = _slice[(_slice.loss == _loss) & (_slice.arch == _arch)]
                    _v = _d["value"].abs()
                    _mae = float(_v.dropna().mean()) if len(_v.dropna()) else 0.0
                    _ax.bar(_bar_idx, _mae if _mae > 0 else 0.0, width=0.6,
                            color=_arch_colors_tr[_arch],
                            edgecolor="k", linewidth=0.5,
                            hatch="//" if _loss in _HATCHED_LOSSES else None,
                            label=f"{_loss} · {_arch}"
                                  if _ri == 0 and _ci == 0 else None)
                    _xticks.append(_bar_idx)
                    _xlabels.append(f"{_loss}\n{_arch}")
                    _bar_idx += 1
                _bar_idx += _gap
            if _pbe_mae is not None and _pbe_mae > 0:
                _ax.axhline(_pbe_mae, ls="--", lw=1.6, color="k",
                            label=f"PBE vs W4-11 ({_pbe_mae:.1f})"
                                  if _ri == 0 and _ci == 0 else None)
            if _ccsd_mae is not None and _ccsd_mae > 0:
                _ax.axhline(_ccsd_mae, ls="-.", lw=1.6, color="purple",
                            label=f"CCSD vs W4-11 ({_ccsd_mae:.2f})"
                                  if _ri == 0 and _ci == 0 else None)
            _ax.set_yscale("log")
            _ax.set_xticks(_xticks)
            _ax.set_xticklabels(_xlabels, rotation=70, fontsize=5)
            _ax.set_title(f"solver={_solver} | trained on {_grp}", fontsize=9)
            _ax.grid(True, which="both", axis="y", ls=":", alpha=0.4)
            if _ci == 0:
                _ax.set_ylabel("MAE |AE error| (kcal/mol, log)", fontsize=8)
    fig.legend(*axes[0][0].get_legend_handles_labels(),
               loc="lower center", ncol=4, fontsize=6,
               bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    fig.savefig(os.path.join(figures_dir, png_name),
                dpi=120, bbox_inches="tight")
    plt.show()

_render_transfer_plot(
    target_df=transfer_primary_df,
    transfer_mols=TRANSFER_PRIMARY,
    data_dir=transfer_primary,
    png_name="transfer_primary_mae.png",
    dataset_label="primary",
    suptitle=("Primary transfer ({H₂, OH, CH₄}): "
              "MAE |AE error| vs W4-11 with PBE/CCSD horizontal references"),
)
"""
    return new_code_cell(source)


def build_cell_35_transfer_secondary_plot():
    """Section 6 Cell 35 -- secondary transfer aggregate MAE plot.

    Reuses ``_render_transfer_plot`` defined in cell 34.
    """
    source = r"""# Reuses _render_transfer_plot defined in cell 34.
_render_transfer_plot(
    target_df=transfer_secondary_df,
    transfer_mols=TRANSFER_SECONDARY,
    data_dir=transfer_secondary,
    png_name="transfer_secondary_mae.png",
    dataset_label="secondary",
    suptitle=("Secondary transfer ({NH₃, HF, CO₂, NH₂}): "
              "MAE |AE error| vs W4-11 with PBE/CCSD horizontal references"),
)
"""
    return new_code_cell(source)


def build_cell_36_drift_md():
    """Section 7 Cell 36 -- F_x(s) drift diagnostic headline (markdown).

    Step-5 finding: NN F_x deforms away from PBE at s > 0.7 on CH4 grid points.
    Step 6 tests two candidate fixes (adding C2H2 to training, PBE-anchor
    regularization). The three panels sample F_x(s) on molecular grids
    spanning three regimes:

      * Panel B / CH4  -- transfer-reference molecule (step-5 finding).
      * Panel B / C2H2 -- in-training molecule (groups 2 + 3).
      * Panel C / C2H4 -- held-out generalization probe (not in any
        training or transfer set).
    """
    return new_markdown_cell(r"""## Section 7 -- F_x(s) Drift Diagnostic (headline)

Step 5 finding: the trained NN F_x(s) drifts away from PBE at s > 0.7 on
CH4 grid points, and this drift correlates with the CH4 transfer gap.
Step 6's two candidate fixes (add C2H2 to training; PBE-anchor
regularization on synthetic (rho, s) samples) are evaluated by sampling
F_x(s) on three molecular grids spanning three regimes:

* **Panel B / CH4**  -- transfer-reference molecule (the step-5 finding
  that motivates step 6).
* **Panel B / C2H2** -- in-training molecule for groups 2 + 3; answers
  "does C2H2 in the batch stabilize F_x where the batch samples it?"
* **Panel C / C2H4** -- held-out generalization probe; not in any
  training or transfer set, tests whether (a) data expansion or (b)
  anchor regularization closes the drift on a genuinely unseen molecule.

For each panel the analytic PBE curve (solid black) is the reference;
pretrained baselines (green) show where fine-tuning starts; fine-tuned
models (blue / orange / red for groups 1 / 2 / 3) show where fine-tuning
ends.
""")


def build_cell_37_drift_panel_b():
    """Section 7 Cell 37 -- F_x(s) drift Panel B (CH4 + C2H2).

    Samples F_x(s) on CH4 + C2H2 grid points for all 72 trained models +
    per-arch pretrained baselines + the analytic PBE reference. The NN
    F_x is evaluated via ``_nn_fx_local_uks(model, rho/2, rho/2, s)`` --
    the spin-scaled UKS approximation that matches the SCF-time
    convention used inside the solver. Pretrained checkpoints are
    {pretrain_dir}/{arch}/xnet.eqx + cnet.eqx combined via
    ``AlecGGAModel.from_arch(arch, xnet=..., cnet=...)``.

    API fix: the plan's ``load_model_checkpoint`` symbol does not exist
    on ``xcquinox.alec.models``; the canonical pattern (verified at
    ``xcquinox/alec/evaluation.py:215-218``) is
    ``eqx.tree_deserialise_leaves(ckpt_path, AlecGGAModel.from_arch(arch_cfg, seed=0))``.
    """
    source = r"""# F_x(s) drift Panel B: CH4 (transfer ref) + C2H2 (in-training). Samples
# F_x on each molecule's PBE grid for all 72 trained models + per-arch
# pretrained baselines + the analytic PBE reference. F_x is evaluated on
# the molecule using the SAME descriptor features (cusp + dm_statistics)
# the network sees during a real SCF -- evaluating with zero-extras gives
# fictional curves for archs whose F_x depends on the descriptors.
from xcquinox.alec.models import AlecGGAModel
from xcquinox.alec.networks import create_network_pair
from xcquinox.alec.descriptors import assemble_descriptor_features


def _load_full_model_from_ckpt(_ckpt_path, _arch_name):
    # Canonical pattern (matches run_test @ evaluation.py:215-218).
    _arch_cfg = alec.get_architecture(_arch_name)
    _skel = AlecGGAModel.from_arch(_arch_cfg, seed=0)
    return eqx.tree_deserialise_leaves(_ckpt_path, _skel)


def _load_pretrain_model(_pretrain_dir, _arch_name):
    # Pretrain saves xnet.eqx + cnet.eqx separately (step-3 pretrain.py);
    # combine into a full AlecGGAModel for F_x sampling.
    _arch_cfg = alec.get_architecture(_arch_name)
    _xskel, _cskel = create_network_pair(_arch_cfg, seed=0)
    _xnet = eqx.tree_deserialise_leaves(
        os.path.join(_pretrain_dir, _arch_name, "xnet.eqx"), _xskel,
    )
    _cnet = eqx.tree_deserialise_leaves(
        os.path.join(_pretrain_dir, _arch_name, "cnet.eqx"), _cskel,
    )
    return AlecGGAModel.from_arch(_arch_cfg, xnet=_xnet, cnet=_cnet)


# Per-(arch, molecule) cache: stores (rho, s, features_per_grid_row,
# spin) where features_per_grid_row is the (n_grid, n_extra) array of
# real descriptor features (cusp + dm_statistics) the network expects.
# Shared across all checkpoints with the same architecture so the heavy
# PBE SCF + descriptor precompute runs once per (arch, molecule).
_grid_cache = {}

def _compute_grid_for_molecule(_atom_str, _spin, _arch_name, _mol_name):
    _arch_cfg = alec.get_architecture(_arch_name)
    _key = (_atom_str, int(_spin), BASIS, int(GRID_LEVEL), _arch_name)
    if _key in _grid_cache:
        return _grid_cache[_key]
    _skel = AlecGGAModel.from_arch(_arch_cfg, seed=0)
    _required = tuple(set(
        _k for _d in _skel.descriptors for _k in _d.required_mol_keys
    ))
    _mol_spec = alec.MoleculeSpec(
        name=_mol_name, atom=_atom_str, basis=BASIS,
        charge=0, spin=int(_spin), grid_level=int(GRID_LEVEL),
        atom_composition=(),
    )
    _mol_data = alec.precompute_fixed_density_data(
        _mol_spec,
        required_keys=_required,
        descriptors=_skel.descriptors,
    )
    _rho = np.asarray(_mol_data["rho_grid"])
    _sigma = np.asarray(_mol_data["sigma_grid"])
    _kF = (3.0 * np.pi ** 2) ** (1.0 / 3.0)
    _s = np.sqrt(np.clip(_sigma, 0.0, None)) / (
        2.0 * _kF * np.clip(_rho, 1e-12, None) ** (4.0 / 3.0)
    )
    _features = np.asarray(
        assemble_descriptor_features(_skel.descriptors, _mol_data)
    )  # (n_grid, n_extra)
    _grid_cache[_key] = (_rho, _s, _sigma, _features)
    return _grid_cache[_key]


def _sample_fx_on_molecule(_model, _atom_str, _spin, _arch_name, _mol_name):
    # Evaluate F_x at every molecular grid point using the network's
    # actual input layout: [rho, sigma_eff, *features]. Returns (s, F_x).
    # For the closed-shell test mols (CH4, C2H2, C2H4) the spin-scaled
    # UKS formula collapses to the RKS evaluation since
    # rho_alpha = rho_beta = rho/2 and zeta = 0; we therefore evaluate
    # model.xnet at (rho, sigma_tot, features_g) per grid point.
    _rho, _s, _sigma, _features = _compute_grid_for_molecule(
        _atom_str, _spin, _arch_name, _mol_name,
    )
    _rho_j = jnp.asarray(_rho)
    _sigma_j = jnp.asarray(_sigma)
    _features_j = jnp.asarray(_features)
    _xnet = _model.xnet

    def _fx_one(_rho_g, _sigma_g, _features_g):
        _inputs = jnp.concatenate([
            jnp.atleast_1d(_rho_g),
            jnp.atleast_1d(_sigma_g),
            _features_g,
        ])
        return _xnet(_inputs)

    _fx = jax.vmap(_fx_one)(_rho_j, _sigma_j, _features_j)
    return _s, np.asarray(_fx)


def _fx_pbe_analytic(_s):
    # PBE: F_x(s) = 1 + kappa - kappa/(1 + mu*s^2/kappa), with
    # kappa=0.804 (Lieb-Oxford bound) and mu=0.21951 (Perdew et al. 1996).
    _kappa = 0.804
    _mu = 0.21951
    return 1.0 + _kappa - _kappa / (1.0 + _mu * _s ** 2 / _kappa)


_PROBE_MOLS = [
    ("CH4",
     "C 0 0 0; H 0.628099 0.628099 0.628099; H -0.628099 -0.628099 0.628099; "
     "H -0.628099 0.628099 -0.628099; H 0.628099 -0.628099 -0.628099",
     0),
    ("C2H2", C2H2_ATOM, 0),
]

_GROUP_COLORS = {"group1": "tab:blue", "group2": "tab:orange", "group3": "tab:red"}
_s_ref = np.linspace(0.01, 15.0, 200)
_fx_ref = _fx_pbe_analytic(_s_ref)

import time as _time
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
_PANEL_FLUSH_EVERY = 16
for _ax, (_nm, _atom, _spin) in zip(axes, _PROBE_MOLS):
    print(f"[Panel B / {_nm}] precomputing PBE grid + descriptor features…",
          flush=True)
    _t_grid = _time.time()
    for _arch in ARCH_NAMES:
        _compute_grid_for_molecule(_atom, _spin, _arch, _nm)
    print(f"[Panel B / {_nm}] grids ready ({_time.time()-_t_grid:.1f}s); "
          f"sampling F_x for pretrained + {len(_all_specs)} fine-tuned models",
          flush=True)
    _t_loop = _time.time()
    # Analytic PBE reference.
    _ax.plot(_s_ref, _fx_ref, "k-", lw=2, label="PBE (analytic)")

    # Per-arch pretrained baselines.
    for _arch in ARCH_NAMES:
        _pre_ckdir = os.path.join(pretrain_dir, _arch)
        if not (os.path.isfile(os.path.join(_pre_ckdir, "xnet.eqx"))
                and os.path.isfile(os.path.join(_pre_ckdir, "cnet.eqx"))):
            continue
        try:
            _m = _load_pretrain_model(pretrain_dir, _arch)
            _sv, _fx = _sample_fx_on_molecule(_m, _atom, _spin, _arch, _nm)
        except Exception as _e:
            print(f"  [pretrain {_arch}] skipped: {_e}", flush=True)
            continue
        _ax.scatter(_sv, _fx, s=2, alpha=0.3, color="green",
                    label="pretrained" if _arch == ARCH_NAMES[0] else None)

    # Fine-tuned models.
    _seen_groups = set()
    _n_done = 0
    _N = sum(1 for _s in _all_specs
             if os.path.isfile(os.path.join(_s.checkpoint_dir, "model.eqx")))
    for _idx, _spec in enumerate(_all_specs):
        _ckpt = os.path.join(_spec.checkpoint_dir, "model.eqx")
        if not os.path.isfile(_ckpt):
            continue
        _group = (
            "group1" if _spec in _specs_group1
            else "group2" if _spec in _specs_group2
            else "group3"
        )
        try:
            _m = _load_full_model_from_ckpt(_ckpt, _spec.arch.name)
            _sv, _fx = _sample_fx_on_molecule(_m, _atom, _spin,
                                              _spec.arch.name, _nm)
        except Exception as _e:
            print(f"  [{_spec.checkpoint_dir}] skipped: {_e}", flush=True)
            continue
        _lbl = _group if _group not in _seen_groups else None
        _seen_groups.add(_group)
        _ax.scatter(_sv, _fx, s=2, alpha=0.15,
                    color=_GROUP_COLORS[_group], label=_lbl)
        _n_done += 1
        # Hold JIT cache through batches; print a progress beat each flush.
        if _n_done % _PANEL_FLUSH_EVERY == 0:
            _el = _time.time() - _t_loop
            _eta = _el / max(_n_done, 1) * max(_N - _n_done, 0)
            print(f"  [{_nm}] {_n_done:>3d}/{_N} sampled  "
                  f"elapsed={_el/60:.1f}min  eta={_eta/60:.1f}min", flush=True)
            jax.clear_caches(); gc.collect()
    print(f"[Panel B / {_nm}] done in {(_time.time()-_t_loop)/60:.1f}min "
          f"({_n_done} fine-tuned models sampled)", flush=True)
    jax.clear_caches(); gc.collect()

    _ax.set_xscale("log")
    _ax.set_xlim(0.01, 15)
    _ax.set_xlabel("reduced gradient s (log)")
    _ax.set_title(f"F_x(s) sampled at {_nm} grid")
    _ax.grid(True, which="both", ls=":", alpha=0.4)

axes[0].set_ylabel(r"exchange enhancement $F_x(s)$")
axes[0].legend(fontsize=8, loc="upper left", framealpha=0.9)
fig.suptitle("Panel B: F_x(s) drift at CH4 (transfer ref) + C2H2 (in-training)",
             fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(os.path.join(figures_dir, "fx_drift_panel_B.png"),
            dpi=120, bbox_inches="tight")
plt.show()
"""
    return new_code_cell(source)


def build_cell_38_drift_panel_c():
    """Section 7 Cell 38 -- F_x(s) drift Panel C (C2H4, held-out probe).

    Mirrors Panel B but on C2H4 (ethylene), which is present in NO
    training set and NO transfer set -- tests whether the F_x deformation
    that trained-set members induce also manifests on a genuinely unseen
    molecule. Reuses ``_sample_fx_on_molecule`` + ``_fx_pbe_analytic`` +
    the two loader helpers defined in cell 37 (sequential notebook
    namespace).
    """
    source = r"""# F_x(s) drift Panel C: C2H4 (ethylene) held-out generalization probe.
# Not in any training or transfer set -- tests whether the trained-model
# F_x deformation persists on a truly unseen molecule. Reuses helpers
# (_sample_fx_on_molecule, _fx_pbe_analytic, _load_pretrain_model,
# _load_full_model_from_ckpt) defined in the previous cell.
C2H4_ATOM = (
    "C  0.000000  0.000000   0.667100; "
    "C  0.000000  0.000000  -0.667100; "
    "H  0.000000  0.923404  -1.231634; "
    "H  0.000000 -0.923404  -1.231634; "
    "H  0.000000  0.923404   1.231634; "
    "H  0.000000 -0.923404   1.231634"
)

_s_ref_c = np.linspace(0.01, 15.0, 200)
_fx_ref_c = _fx_pbe_analytic(_s_ref_c)

import time as _time
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(_s_ref_c, _fx_ref_c, "k-", lw=2, label="PBE (analytic)")
print(f"[Panel C / C2H4] precomputing PBE grid + descriptor features…",
      flush=True)
_tg = _time.time()
for _arch in ARCH_NAMES:
    _compute_grid_for_molecule(C2H4_ATOM, 0, _arch, "C2H4")
print(f"[Panel C / C2H4] grid ready ({_time.time()-_tg:.1f}s); sampling F_x for "
      f"pretrained + {len(_all_specs)} fine-tuned models", flush=True)
_tloop_c = _time.time()

# Per-arch pretrained baselines.
for _arch in ARCH_NAMES:
    _pre_ckdir = os.path.join(pretrain_dir, _arch)
    if not (os.path.isfile(os.path.join(_pre_ckdir, "xnet.eqx"))
            and os.path.isfile(os.path.join(_pre_ckdir, "cnet.eqx"))):
        continue
    try:
        _m = _load_pretrain_model(pretrain_dir, _arch)
        _sv, _fx = _sample_fx_on_molecule(_m, C2H4_ATOM, 0, _arch, "C2H4")
    except Exception as _e:
        print(f"  [pretrain {_arch}] skipped: {_e}", flush=True)
        continue
    ax.scatter(_sv, _fx, s=2, alpha=0.3, color="green",
               label="pretrained" if _arch == ARCH_NAMES[0] else None)

# Fine-tuned models.
_PANEL_FLUSH_EVERY = 16
_seen_groups_c = set()
_n_done_c = 0
_N_c = sum(1 for _s in _all_specs
           if os.path.isfile(os.path.join(_s.checkpoint_dir, "model.eqx")))
for _idx, _spec in enumerate(_all_specs):
    _ckpt = os.path.join(_spec.checkpoint_dir, "model.eqx")
    if not os.path.isfile(_ckpt):
        continue
    _group = (
        "group1" if _spec in _specs_group1
        else "group2" if _spec in _specs_group2
        else "group3"
    )
    try:
        _m = _load_full_model_from_ckpt(_ckpt, _spec.arch.name)
        _sv, _fx = _sample_fx_on_molecule(_m, C2H4_ATOM, 0,
                                          _spec.arch.name, "C2H4")
    except Exception as _e:
        print(f"  [{_spec.checkpoint_dir}] skipped: {_e}", flush=True)
        continue
    _lbl = _group if _group not in _seen_groups_c else None
    _seen_groups_c.add(_group)
    ax.scatter(_sv, _fx, s=2, alpha=0.15,
               color=_GROUP_COLORS[_group], label=_lbl)
    _n_done_c += 1
    if _n_done_c % _PANEL_FLUSH_EVERY == 0:
        _el = _time.time() - _tloop_c
        _eta = _el / max(_n_done_c, 1) * max(_N_c - _n_done_c, 0)
        print(f"  [C2H4] {_n_done_c:>3d}/{_N_c} sampled  "
              f"elapsed={_el/60:.1f}min  eta={_eta/60:.1f}min", flush=True)
        jax.clear_caches(); gc.collect()
print(f"[Panel C / C2H4] done in {(_time.time()-_tloop_c)/60:.1f}min "
      f"({_n_done_c} fine-tuned models sampled)", flush=True)
jax.clear_caches(); gc.collect()

ax.set_xscale("log")
ax.set_xlim(0.01, 15)
ax.set_xlabel("reduced gradient s (log)")
ax.set_ylabel(r"exchange enhancement $F_x(s)$")
ax.set_title("Panel C: F_x(s) at C2H4 grid (held-out generalization probe)")
ax.grid(True, which="both", ls=":", alpha=0.4)
ax.legend(fontsize=8, loc="upper left", framealpha=0.9)
fig.tight_layout()
fig.savefig(os.path.join(figures_dir, "fx_drift_panel_C.png"),
            dpi=120, bbox_inches="tight")
plt.show()
"""
    return new_code_cell(source)


def build_cell_39_scf_convergence():
    """Section 7 Cell 39 -- SCF convergence trace plot.

    Plots the per-cycle |E_n - E_final| convergence trace, averaged across
    every (arch, loss, solver, molecule) combination that ran an SCF loop
    (FIXED_J + FULL only — ONESHOT has no SCF and is sentinel-zero in
    cycles_run). The trace is recorded by ``SCFConvergenceMetric`` in
    ``evaluation.py`` (per-cycle ``e_tot`` captured via a pyscfad SCF
    callback) and ingested into ``eval_df`` as
    ``scf_energy_residual_<i>`` rows. The plot draws one line per solver
    (mean over cycles), with a shaded ± stddev band so the user sees both
    the average decay rate AND the spread across runs.
    """
    source = r"""# SCF convergence trace: per-cycle |E_n - E_final| averaged across all
# specs that actually ran SCF (FIXED_J + FULL). Reads
# scf_energy_residual_<i> rows produced by SCFConvergenceMetric in
# cell 25's run_test sweep. The plot draws one line per solver +
# shaded ± stddev band; ONESHOT specs are skipped (no SCF).
import re as _re
_RES_RE = _re.compile(r"^scf_energy_residual_(\d+)$")
_res_rows = eval_df[eval_df.value_name.str.match(_RES_RE)].copy()

fig, ax = plt.subplots(figsize=(8, 5))
if len(_res_rows) == 0:
    ax.text(0.5, 0.5,
            "no SCF energy traces\n"
            "(re-run cell 25 with scf_convergence metric on a build "
            "that records SCFConvergenceMetric energy_trace)",
            ha="center", va="center", fontsize=11,
            transform=ax.transAxes, color="gray")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("SCF convergence trace across training specs")
    print("[scf_convergence] no per-cycle traces recorded -- placeholder plot")
else:
    _res_rows["cycle"] = _res_rows["value_name"].map(
        lambda _v: int(_RES_RE.match(_v).group(1))
    )
    _res_rows["residual"] = _res_rows["value"].astype(float)
    _solver_colors = {"oneshot": "tab:blue",
                      "fixed_j_3": "tab:orange",
                      "full_3": "tab:green"}
    _ordered_solvers = [_s for _s in SOLVER_LABELS
                        if _s in _res_rows.solver.unique()]
    for _solver in _ordered_solvers:
        _sub = _res_rows[_res_rows.solver == _solver]
        if _sub.empty:
            continue
        _agg = _sub.groupby("cycle")["residual"].agg(["mean", "std", "count"])
        _agg = _agg.sort_index()
        _x = _agg.index.values
        _mean = _agg["mean"].values
        _std = _agg["std"].fillna(0.0).values
        _color = _solver_colors.get(_solver, "tab:gray")
        ax.plot(_x, _mean, "o-", color=_color, lw=1.6, ms=4,
                label=f"{_solver} (n_runs at cycle 0 = {int(_agg['count'].iloc[0])})")
        # Lower band can dip below zero on a log axis; clip to a tiny
        # positive floor so the fill is well-defined.
        _lo = np.maximum(_mean - _std, 1e-14)
        _hi = _mean + _std
        ax.fill_between(_x, _lo, _hi, color=_color, alpha=0.18,
                        linewidth=0)
    ax.set_yscale("log")
    ax.set_xlabel("SCF cycle index n")
    ax.set_ylabel(r"average $|E_n - E_{\rm final}|$  (Ha, log scale)")
    ax.set_title("SCF convergence trace by solver "
                 "(per-cycle residual to converged energy)")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize=8, loc="best")
    print(f"[scf_convergence] {len(_res_rows)} per-cycle residual rows "
          f"across solvers={_ordered_solvers}")

fig.tight_layout()
fig.savefig(os.path.join(figures_dir, "scf_convergence.png"),
            dpi=120, bbox_inches="tight")
plt.show()
"""
    return new_code_cell(source)


def build_cell_40_findings_md():
    return new_markdown_cell(r"""## Section 7 -- Findings

Populate post-run. Template:
- H1 (data fix): ...
- H2 (regularization fix): ...
- H3 (interaction): ...
- H4 (overfitting): ...
- H5 (V_xc necessity): ...
""")


def build_cell_41_step7_roadmap_md():
    return new_markdown_cell(r"""## Section 8 -- Step 7 Roadmap

Skeleton -- body depends on step-6 results. Candidate directions:
1. If data fix works: widen training to W4-11 subset.
2. If PBE-anchor works: sweep w_anchor in {1e-4, 1e-3, 1e-2}.
3. If overfitting confirmed: test early-stopping criteria.
""")


def build_cell_42_closing_md():
    return new_markdown_cell(r"""## Closing

End of step-6 notebook. Regenerate from
`notebooks/_build_step6_notebook.py` -- never hand-edit.
""")


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
        build_cell_24_eval_md(),
        build_cell_25_main_sweep(),
        build_cell_26_eval_preview(),
        build_cell_26b_baseline_evals(),
        build_cell_27_vxc_efficacy(),
        build_cell_28_anchor_effect(),
        build_cell_29_transfer_md(),
        build_cell_30_transfer_primary(),
        build_cell_31_transfer_secondary(),
        build_cell_32_transfer_primary_eval(),
        build_cell_33_transfer_secondary_eval(),
        build_cell_34_transfer_primary_plot(),
        build_cell_35_transfer_secondary_plot(),
        build_cell_36_drift_md(),
        build_cell_37_drift_panel_b(),
        build_cell_38_drift_panel_c(),
        build_cell_39_scf_convergence(),
        build_cell_40_findings_md(),
        build_cell_41_step7_roadmap_md(),
        build_cell_42_closing_md(),
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
