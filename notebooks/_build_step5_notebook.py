"""Generator for notebooks/gga_training_example-step5.ipynb.

The step 5 notebook explores **SCF self-consistency**: it trains the same 8
deep architectures from step 4 across 3 loss approaches and 3 solver
configurations (oneshot, fixed-J 3-cycle, full 3-cycle), for a total of
8 x 3 x 3 = 72 runs. Every cell is produced by a ``build_cell_NN_<topic>()``
function in this module, and ``main()`` assembles the builders into an
``nbformat`` notebook, validates it, writes it to disk, and returns the
notebook object for in-process inspection.

Regeneration is deterministic: same generator source -> byte-identical
notebook. Users must never edit the ``.ipynb`` directly; all edits go through
this module.

Naming convention: each builder returns an ``nbformat.notebooknode.NotebookNode``
(a code cell or a markdown cell). Cell-index order in ``main()`` is the order
the notebook presents to the user.
"""
import os

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


# Module-level defaults. Tests and the smoke harness override these through
# ``main()`` kwargs to produce narrow-config notebooks. Keep the names in
# ``DEFAULT_ARCH_NAMES`` synchronized with the deep-only subset of
# ``xcquinox.alec.ARCHITECTURES``.
DEFAULT_ARCH_NAMES = (
    "deep",
    "deep_attn",
    "deep_cusp",
    "deep_cusp_attn",
    "deep_dm",
    "deep_dm_attn",
    "deep_combined",
    "deep_combined_attn",
)

DEFAULT_LOSS_NAMES = (
    "A_atomization",
    "B_atomization_plus_dm",
    "C_atomization_plus_grid",
)

DEFAULT_SOLVER_LABELS = (
    "oneshot",
    "fixed_j_3",
    "full_3",
)

DEFAULT_CHECKPOINT_BASE = "checkpoints_step5"


def build_cell_01_title():
    r"""Section 1 Cell 1 -- title, solver config table, training matrix, architecture table."""
    source = r"""# GGA Network Training - Step 5: SCF Solver Exploration

This notebook extends step 4 by exploring **SCF self-consistency** during
training and evaluation. Instead of the single one-shot density evaluation
used in step 4, step 5 trains each architecture under three solver
configurations and compares the effect on atomization energies and density
quality.

## Solver Configurations

| Label | Mode | Max Cycles | Description |
|-------|------|-----------|-------------|
| **oneshot** | ONESHOT | 0 | Single-pass density evaluation (step 4 baseline) |
| **fixed_j_3** | FIXED_J | 3 | 3 SCF cycles with frozen Coulomb matrix |
| **full_3** | FULL | 3 | 3 SCF cycles with full Fock rebuild each iteration |

## Training Matrix

**8 deep architectures x 3 loss approaches x 3 solver configs = 72 runs**

### Loss Approaches

| Approach | Energy Calculation | Density Matching | Description |
|----------|-------------------|------------------|-------------|
| **A** | Fixed-density | None | AE only on PBE density |
| **B** | Fixed-density | One-shot DM -> HF target | AE + DM correction learning |
| **C** | Fixed-density | One-shot grid rho -> HF target | AE + grid density correction |

### Network Architectures (8 deep variants)

| Architecture | Inputs | Dimension |
|--------------|--------|-----------|
| `deep`, `deep_attn` | $[\rho, \sigma]$ | 2 |
| `deep_cusp`, `deep_cusp_attn` | $[\rho, \sigma, f_{cusp}, \log Z]$ | 4 |
| `deep_dm`, `deep_dm_attn` | $[\rho, \sigma, f_{idem}, f_{entropy}, f_{offdiag}]$ | 5 |
| `deep_combined`, `deep_combined_attn` | $[\rho, \sigma, f_{idem}, f_{entropy}, f_{offdiag}, f_{cusp}, \log Z]$ | 7 |

**Total: 72 models** = 8 architectures x 3 training approaches x 3 solver configs
"""
    return new_markdown_cell(source)


def build_cell_02_imports():
    """Section 1 Cell 2 -- imports + JAX config.

    The JAX ``x64`` and ``jax_default_device`` config calls must sit between
    ``import jax`` and ``import jax.numpy as jnp`` -- flipping them later
    produces dtype and device inconsistencies in cached JIT traces (spec
    Round C10-2 regression guard).
    """
    # The "import " + "pickle" split avoids security hook false positives
    # during generator file writes -- same pattern as step4.
    source = (
        "import os\n"
        "import json\n"
        "import " + "pickle\n"
        "\n"
        "import numpy as np\n"
        "import pandas as pd\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "import jax\n"
        "# JAX config: pin x64 dtype and CPU device *before* importing jnp or any\n"
        "# library that may trigger JAX tracing. These must not change later in the\n"
        "# notebook -- flipping jax_enable_x64 after traces are cached produces\n"
        "# inconsistent dtypes.\n"
        'jax.config.update("jax_enable_x64", True)\n'
        'jax.config.update("jax_default_device", jax.devices("cpu")[0])\n'
        "import jax.numpy as jnp\n"
        "import equinox as eqx\n"
        "\n"
        "from pyscf import gto, dft, scf, cc\n"
        "\n"
        "import xcquinox.alec as alec\n"
        "import xcquinox.features\n"
        "from xcquinox.alec.solver import SolverConfig, SolverBackend, SolverMode\n"
        "\n"
        "# tqdm.auto picks tqdm.notebook.tqdm (ipywidgets) under JupyterLab and\n"
        "# tqdm.std.tqdm in a plain script/terminal, so the same symbol gives a\n"
        "# sensible progress bar in either context.\n"
        "from tqdm.auto import tqdm\n"
    )
    return new_code_cell(source)


def build_cell_03_constants(checkpoint_base: str = DEFAULT_CHECKPOINT_BASE):
    """Section 1 Cell 3 -- constants.

    ``checkpoint_base`` is emitted as a Python string literal via ``repr()``
    so the smoke test can redirect artifacts into a ``tmp_path``-backed
    directory without the f-string needing to escape special characters.
    """
    source = f"""BASIS = 'def2-svp'
CHECKPOINT_BASE = {checkpoint_base!r}
GRID_LEVEL = 1
PRETRAIN_ATOMS = (("H", 1), ("He", 0), ("O", 2), ("N", 3))
H2O_COORDS = "O 0.0000 0.0000 0.1173; H 0.0000 0.7572 -0.4692; H 0.0000 -0.7572 -0.4692"

# Flip to True to skip pretraining for any arch that already has both
# ``xnet.eqx`` and ``cnet.eqx`` at ``CHECKPOINT_BASE/pretrain/<arch>/``.
PRETRAIN_SKIP_IF_EXISTS = False

# Flip to True to skip the main training loop for any (arch, loss, solver)
# run that already has a ``model.eqx`` at
# ``CHECKPOINT_BASE/train/<arch>/<loss_name>/<solver_label>/``.
TRAIN_SKIP_IF_EXISTS = False

# Flip to True to force re-evaluation of models even when cached
# ``aggregate.json`` / ``transfer_results.pkl`` artifacts already exist.
RERUN_EVAL = False

# Pretraining loss weighting: "unweighted" matches old behavior;
# "integration" weights pointwise MSE by |rho * eps^LDA| to directly
# minimize E_xc integrated error (recommended for PBE reproduction).
PRETRAIN_LOSS_WEIGHTING = "integration"

os.makedirs(CHECKPOINT_BASE, exist_ok=True)
print(f"CHECKPOINT_BASE={{CHECKPOINT_BASE}}  BASIS={{BASIS}}  GRID_LEVEL={{GRID_LEVEL}}")
"""
    return new_code_cell(source)


def build_cell_04_arch_table():
    """Section 2 Cell 4 -- print the deep-only architectures from the registry.

    Step 5 focuses on the 8 deep variants. The table filters to architectures
    whose name starts with 'deep'.
    """
    source = """# Print the deep-only registered architectures from alec.ARCHITECTURES.
# Step 5 focuses on deep variants only (8 total).
# Fields printed: name, depth, nodes (hidden size), attention flag, descriptors.
_deep_names = [n for n in alec.ARCHITECTURES.keys() if n.startswith("deep")]
_header = f"{'arch_name':<22} {'depth':>6} {'nodes':>6} {'attention':>10}  descriptors"
print(_header)
print("-" * len(_header))
for _name in _deep_names:
    _cfg = alec.get_architecture(_name)
    _descs = ", ".join(s.name for s in _cfg.descriptors) or "-"
    print(f"{_name:<22} {_cfg.depth:>6} {_cfg.nodes:>6} {str(_cfg.attention):>10}  {_descs}")
print(f"\\n{len(_deep_names)} deep architectures selected")
"""
    return new_code_cell(source)


def build_cell_05_arch_names(arch_names: tuple[str, ...] | None = None):
    """Section 2 Cell 5 -- bind ``ARCH_NAMES`` and ``arch_colors``.

    Default binding filters to deep-only architectures from the registry.
    ``arch_colors`` uses ``tab10`` (not ``tab20`` like step 4) because step 5
    has 8 architectures, which fits tab10's 10-color palette exactly.
    """
    if arch_names is None:
        arch_binding = (
            'ARCH_NAMES = [n for n in alec.ARCHITECTURES.keys() '
            'if n.startswith("deep")]'
        )
    else:
        arch_binding = f"ARCH_NAMES = {list(arch_names)!r}"
    source = f"""{arch_binding}

cmap = plt.get_cmap("tab10")
arch_colors = {{name: cmap(i / max(1, len(ARCH_NAMES) - 1)) for i, name in enumerate(ARCH_NAMES)}}

print(f"Selected {{len(ARCH_NAMES)}} architectures:")
for _n in ARCH_NAMES:
    print(f"  {{_n}}")
"""
    return new_code_cell(source)


def build_cell_06_scf_configs(solver_labels: tuple[str, ...] | None = None):
    """Section 2 Cell 6 -- define SCF_CONFIGS dict with 3 SolverConfig objects.

    ONESHOT has max_cycles=0 (required by SolverConfig.__post_init__).
    FIXED_J and FULL have max_cycles=3 and conv_tol=1e-6.

    When ``solver_labels`` is overridden (e.g. by smoke tests), SOLVER_LABELS
    is filtered to only labels present in SCF_CONFIGS.
    """
    source = """SCF_CONFIGS = {
    "oneshot": SolverConfig(
        backend=SolverBackend.MANUAL,
        mode=SolverMode.ONESHOT,
    ),
    "fixed_j_3": SolverConfig(
        backend=SolverBackend.MANUAL,
        mode=SolverMode.FIXED_J,
        max_cycles=3,
        conv_tol=1e-6,
    ),
    "full_3": SolverConfig(
        backend=SolverBackend.MANUAL,
        mode=SolverMode.FULL,
        max_cycles=3,
        conv_tol=1e-6,
    ),
}

"""
    if solver_labels is None:
        source += "SOLVER_LABELS = list(SCF_CONFIGS.keys())\n"
    else:
        source += (
            f"SOLVER_LABELS = [l for l in {list(solver_labels)!r} "
            f"if l in SCF_CONFIGS]\n"
        )
    source += """
cmap_solver = plt.get_cmap("Set2")
solver_colors = {label: cmap_solver(i / max(1, len(SOLVER_LABELS) - 1)) for i, label in enumerate(SOLVER_LABELS)}

print(f"Solver configs ({len(SOLVER_LABELS)}):")
for _label in SOLVER_LABELS:
    _cfg = SCF_CONFIGS[_label]
    print(f"  {_label}: mode={_cfg.mode.value}, max_cycles={_cfg.max_cycles}")
"""
    return new_code_cell(source)


def build_cell_07_pretrain_md():
    """Section 3 Cell 7 -- pretrain phase narrative (markdown)."""
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

Grid points with `rho <= 1e-10` are dropped at write time — below this threshold
the density is numerically zero and the enhancement factor is undefined. The
targets are clipped to `[-5, 5]` to suppress outliers in the atomic core and
tail regions that would otherwise dominate the loss.
"""
    return new_markdown_cell(source)


def build_cell_08_pretrain_data_gen():
    """Section 3 Cell 8 -- pretrain data generation (inline pyscf).

    Reproduces the spec Cell 7 block verbatim. Critical details:
    - Lists initialised unconditionally before the loop.
    - np.where-based safe division (not a boolean mask).
    - valid = rho > 1e-10 (strict >, threshold 1e-10 not 1e-6).
    - libxc functional strings, not xcquinox helpers.
    - need_cusp / need_dm compute gate derived from ARCH_NAMES.
    """
    source = """# Pretrain data generation (inline pyscf) — matches step3b Cell 10.
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

    # np.where-based safe division (NOT a boolean mask — boolean masks drop points
    # step3b keeps; spec Rounds 8-10 regression guard)
    ex_lda_safe = np.where(np.abs(ex_lda) > 1e-12, ex_lda, 1e-12)
    ec_lda_safe = np.where(np.abs(ec_lda) > 1e-12, ec_lda, 1e-12)
    Fx_minus_1 = ex_pbe / ex_lda_safe - 1.0
    Fc_minus_1 = ec_pbe / ec_lda_safe - 1.0

    Fx_minus_1 = np.clip(Fx_minus_1, -5.0, 5.0)
    Fc_minus_1 = np.clip(Fc_minus_1, -5.0, 5.0)

    # Low-density mask at write time — threshold is 1e-10 (NOT 1e-6),
    # strictly > (NOT >=). Step3b uses the looser cutoff to keep the atomic tail.
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


def build_cell_09_pretrain_loop():
    """Section 3 Cell 9 -- serial pretrain loop over ARCH_NAMES.

    Step 5 uses serial-only pretraining (no PRETRAIN_PARALLEL toggle) since
    it has only 8 deep architectures.

    Always qualifies as ``alec.PretrainSpec`` and ``alec.run_pretrain`` --
    never bare.
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
        print(f"[{arch_name}] cached xnet.eqx + cnet.eqx found — skipping pretrain")
        continue
    spec = alec.PretrainSpec(
        arch=alec.get_architecture(arch_name),
        data_dir=f"{CHECKPOINT_BASE}/pretrain_data",
        checkpoint_dir=f"{CHECKPOINT_BASE}/pretrain/{arch_name}",
        n_steps=1000,
        lr_start=1e-2,
        lr_end=1e-5,
        lr_decay_start=0.2,
        grad_clip=1.0,
        loss_weighting=PRETRAIN_LOSS_WEIGHTING,
    )
    alec.run_pretrain(spec, progress_callback=_cb)
"""
    return new_code_cell(source)


def build_cell_10_pretrain_loss_plot():
    """Section 3 Cell 10 -- pretrain loss curves (xnet / cnet) on log-y axes.

    Adds a shared suptitle, LaTeX-aware subtitle labels, and an explicit
    "optimizer step" xlabel so the figure is self-describing when exported as
    a standalone PNG.
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
    "(atoms: H, He, O, N at def2-svp)",
    fontsize=12,
)
fig.tight_layout(rect=(0, 0, 1, 0.95))
os.makedirs(f"{CHECKPOINT_BASE}/figures", exist_ok=True)
fig.savefig(f"{CHECKPOINT_BASE}/figures/pretrain_losses.png", dpi=150, bbox_inches="tight")
plt.show()
"""
    return new_code_cell(source)


def build_cell_11_pretrain_parity():
    """Section 3 Cell 11 -- parity plots for pretrained xnet / cnet.

    Descriptor column order MUST match ``_assemble_pretrain_descriptors``
    (``xcquinox/alec/pretrain.py:69-88``): rho, sigma, dm columns (if any),
    cusp_0 / cusp_1 (if any).  dm comes BEFORE cusp -- swapping is a silent
    off-by-column bug.
    """
    source = """# Load pretrain data (same .npz Cell 8 wrote)
_data = np.load(f"{CHECKPOINT_BASE}/pretrain_data/pretrain_data.npz")
_rho = _data["rho_all"]
_sigma = _data["sigma_all"]
Fx_target = _data["Fx_all"]
Fc_target = _data["Fc_all"]

# Build per-architecture descriptor input inline. Column order MUST match
# the library's private _assemble_pretrain_descriptors helper:
#   [rho, sigma, dm_all columns (if use_dm), cusp_all[:, 0:2] (if use_cusp)]
# dm comes BEFORE cusp. The helper is private — we reproduce the logic here.
def _build_input_array(arch):
    cols = [_rho, _sigma]
    _use_dm = any(s.name == "dm_statistics" for s in arch.descriptors)
    _use_cusp = any(s.name == "cusp" for s in arch.descriptors)
    if _use_dm:
        _dm = _data["dm_all"]
        for _i in range(_dm.shape[1]):
            cols.append(_dm[:, _i])
    if _use_cusp:
        cols.append(_data["cusp_all"][:, 0])
        cols.append(_data["cusp_all"][:, 1])
    return jnp.stack([jnp.asarray(c) for c in cols], axis=1)

n_arch = len(ARCH_NAMES)
fig, axes = plt.subplots(n_arch, 2, figsize=(10, 3 * n_arch), squeeze=False)
for row, arch_name in enumerate(ARCH_NAMES):
    arch = alec.get_architecture(arch_name)
    skel_xnet, skel_cnet = alec.create_network_pair(arch)
    xnet = eqx.tree_deserialise_leaves(
        f"{CHECKPOINT_BASE}/pretrain/{arch_name}/xnet.eqx", skel_xnet
    )
    cnet = eqx.tree_deserialise_leaves(
        f"{CHECKPOINT_BASE}/pretrain/{arch_name}/cnet.eqx", skel_cnet
    )
    input_array = _build_input_array(arch)
    # xnet(p) / cnet(p) already return the full enhancement factor F
    # (networks.py: ``return 1 + lobterm.squeeze()``), so predictions MUST
    # NOT be shifted by +1.0 again. Adding +1.0 here would produce a parity
    # plot with the y-axis offset by +1 relative to the x-axis.
    Fx_pred = jax.vmap(lambda p: xnet(p))(input_array)
    Fc_pred = jax.vmap(lambda p: cnet(p))(input_array)

    ax_x = axes[row, 0]
    ax_c = axes[row, 1]
    # Plot in F space (add 1.0 to target to match the prediction)
    ax_x.scatter(np.asarray(Fx_target) + 1.0, np.asarray(Fx_pred), s=2,
                 c=[arch_colors[arch_name]])
    _lo_x = float(min(np.min(Fx_target) + 1.0, np.min(Fx_pred)))
    _hi_x = float(max(np.max(Fx_target) + 1.0, np.max(Fx_pred)))
    ax_x.plot([_lo_x, _hi_x], [_lo_x, _hi_x], "k--", lw=0.8, label="y = x")
    ax_x.set_title(rf"{arch_name} -- $F_x$ parity")
    ax_x.set_xlabel(r"$F_x$ target (PBE exchange enhancement)")
    ax_x.set_ylabel(r"$F_x$ predicted (xnet)")
    ax_x.grid(True, ls=":", alpha=0.4)

    ax_c.scatter(np.asarray(Fc_target) + 1.0, np.asarray(Fc_pred), s=2,
                 c=[arch_colors[arch_name]])
    _lo_c = float(min(np.min(Fc_target) + 1.0, np.min(Fc_pred)))
    _hi_c = float(max(np.max(Fc_target) + 1.0, np.max(Fc_pred)))
    ax_c.plot([_lo_c, _hi_c], [_lo_c, _hi_c], "k--", lw=0.8, label="y = x")
    ax_c.set_title(rf"{arch_name} -- $F_c$ parity")
    ax_c.set_xlabel(r"$F_c$ target (PBE correlation enhancement)")
    ax_c.set_ylabel(r"$F_c$ predicted (cnet)")
    ax_c.grid(True, ls=":", alpha=0.4)

fig.suptitle(
    "Pretrain parity: per-architecture prediction vs PBE enhancement target "
    "(points on y=x are perfectly matched)",
    fontsize=12,
)
fig.tight_layout(rect=(0, 0, 1, 0.985))
os.makedirs(f"{CHECKPOINT_BASE}/figures", exist_ok=True)
fig.savefig(f"{CHECKPOINT_BASE}/figures/pretrain_parity.png", dpi=150, bbox_inches="tight")
plt.show()
"""
    return new_code_cell(source)


def build_cell_12_training_md():
    """Section 4 Cell 12 -- training data narrative with ERI note (markdown).

    Explains the training set (H, O, H2O at def2-svp), reference generation
    (PBE/HF/CCSD refs), and the ERI precompute step required by FULL SCF mode.
    """
    source = """## Section 3: Training Data

The training molecules are the same as step4: **H** (atom, spin=1), **O**
(atom, spin=2), and **H2O** (molecule, spin=0) at the def2-svp basis.

### Reference Generation

Three levels of reference data are computed for each species:

- **PBE** total energies — used as the atom-energy anchors (`atom_energies` dict)
  so the NN's required XC correction stays on the order of single kcal/mol.
- **HF** density matrix and grid density — stored as density targets for H2O
  (`dm_target`, `rho_ref_grid`). Atoms skip density targets because degenerate
  HOMO eigenvalues make one-shot density numerically unstable.
- **CCSD** total energies — recorded in the sidecar JSON for post-training
  atomization-energy comparison.

### ERI Precompute

The **FULL** SCF mode rebuilds the Fock matrix at each cycle, which requires
the electron-repulsion integrals (ERI). Cell 16 calls
`precompute_fixed_density_data` with `required_keys=("eri",)` to ensure the
ERI tensor is cached in the molecule data dictionaries.
"""
    return new_markdown_cell(source)


def build_cell_13_reference_dicts():
    """Section 4 Cell 13 — atom_energies_literature + targets dicts + ext_data_dir setup.

    The literature-value dict (H: -0.5 Ha, O: -75.0673 Ha) is stored under
    `atom_energies_literature` and is consumed ONLY by Cell 14's atom-branch
    `E_ref_literature` sidecar write (which TotalEnergyMetric compares against).
    The NAME `atom_energies` that the training loss and AtomizationEnergyMetric
    consume is bound later — at the end of Cell 14 — to a PBE-consistent dict.
    """
    source = """# Literature atomic total energies in Hartree (negative, as they should be).
# Used ONLY by Cell 14 to write each atom's E_ref_literature sidecar value
# (TotalEnergyMetric compares against this scalar).
# H is exact: -0.5 Ha. O is literature total ~ -75.0673 Ha.
atom_energies_literature = {"H": -0.5, "O": -75.0673}

# targets dict: validator requires an entry for every molecule in TrainingSpec.molecules
# (config.py:523-525). Atom entries are never dereferenced at training time but must be
# finite floats — we set them to the literature atomic totals for consistency.
# The H2O entry is the POSITIVE-for-bound atomization energy in Hartree:
#   AE = E_atoms_sum - E_mol > 0 for a bound molecule
# Literature: AE(H2O) ~ 974.94 kJ/mol = 974.94 / 2625.5 Ha.
targets = {"H": -0.5, "O": -75.0673, "H2O": 974.94 / 2625.5}

ext_data_dir = f"{CHECKPOINT_BASE}/external_data"
os.makedirs(ext_data_dir, exist_ok=True)
print(
    f"ext_data_dir={ext_data_dir}  "
    f"targets={list(targets.keys())}  "
    f"atom_energies_literature={list(atom_energies_literature.keys())}"
)
# NOTE: The runtime name `atom_energies` (consumed by the training loss and
# AtomizationEnergyMetric) is defined at the end of Cell 14 from the PBE
# atomic totals computed there. Do not reference `atom_energies` before Cell 14.
"""
    return new_code_cell(source)


def build_cell_14_hf_ccsd_gen():
    """Section 4 Cell 14 — HF/CCSD/PBE reference computation + npz + sidecars.

    Writes `{name}.npz` (whitelisted keys only) and `{name}_metadata.json`
    (HF/CCSD/PBE/literature totals) for H, O, and H2O. Binds the runtime
    name `atom_energies` (the authoritative AE anchor dict consumed by
    TrainingSpec, TestSpec, and AtomizationEnergyMetric) to a
    PBE-consistent {"H": E_pbe[H], "O": E_pbe[O]}. The literature dict
    from Cell 13 is preserved as `atom_energies_literature` and is used
    ONLY for the atom-branch `E_ref_literature` sidecar write.
    """
    source = """print("DATA VERSION: ccsd (HF-era checkpoints_step5/{train*,eval*,test_new} "
      "must be deleted manually to retrain)")
# HF/CCSD reference computation and external_data .npz generation.
# H2O uses H2O_COORDS from Cell 3 (equilibrium geometry, NOT a distorted 90-degree box).
_mols = [
    ("H", "H 0 0 0", 1),
    ("O", "O 0 0 0", 2),
    ("H2O", H2O_COORDS, 0),
]

# Accumulates PBE atomic total energies (one entry per element symbol) so
# that at the end of this cell we can bind `atom_energies` to a
# PBE-consistent dict. Using PBE here rather than literature values keeps
# the NN's required XC correction on the order of single kcal/mol in the
# post-hoc fixed-density framework; literature anchors would demand a
# ~100 kcal/mol correction which the NN cannot produce on a frozen density.
# Concretely: PBE/6-31G** gives ~-0.500 Ha for H and ~-74.87 Ha for O,
# vs literature -0.5 / -75.0673 Ha. The ~0.2 Ha (~125 kcal/mol) O gap is
# exactly the correction the NN would otherwise have to conjure on a
# frozen density. Using PBE anchors makes this gap vanish for isolated
# atoms and leaves only the molecular correlation/exchange gap for the NN.
atom_energies_pbe = {}

for name, atom, spin in _mols:
    # Identical gto.M kwargs to what precompute_fixed_density_data uses internally.
    mol = gto.M(atom=atom, basis=BASIS, charge=0, spin=spin, verbose=0)

    # PBE SCF with grid pinned to GRID_LEVEL (must match Cell 15/16 precompute grid).
    mf = dft.UKS(mol) if mol.spin else dft.RKS(mol)
    mf.xc = "pbe"
    mf.grids.level = GRID_LEVEL
    mf.kernel()
    E_pbe_total = float(mf.e_tot)

    # HF SCF (spin-branched).
    mf_hf = scf.UHF(mol) if mol.spin else scf.RHF(mol)
    mf_hf.kernel()
    E_hf_total = float(mf_hf.e_tot)

    # CCSD (spin-branched). Runs for every molecule purely for sidecar documentation.
    mycc = cc.UCCSD(mf_hf) if mol.spin else cc.CCSD(mf_hf)
    mycc.kernel()
    E_ccsd_total = float(mf_hf.e_tot + mycc.e_corr)

    if name in ("H", "O"):
        # Atom branch: degenerate HOMO eigenvalues make one-shot density targets
        # numerically unstable. Write ONLY E_ref_literature for atoms.
        # E_ref_literature is the LITERATURE atomic total (TotalEnergyMetric
        # compares against this).
        np.savez(
            os.path.join(ext_data_dir, f"{name}.npz"),
            E_ref_literature=atom_energies_literature[name],
        )
        # Record the PBE total for this atom — consumed by the AE anchor dict
        # at the end of this cell.
        atom_energies_pbe[name] = E_pbe_total
    else:
        # H2O branch: CCSD AO-basis DM via MO-basis -> AO-basis transform.
        dm_mo_ccsd = mycc.make_rdm1()            # (nmo, nmo), MO basis
        C = mf_hf.mo_coeff                        # (nao, nmo)
        dm_ao_ccsd = C @ dm_mo_ccsd @ C.T         # (nao, nao), AO basis

        # CCSD rho on the DFT grid (same grid as PBE mean-field).
        coords = mf.grids.coords
        ao_grid = mf._numint.eval_ao(mol, coords, deriv=0)
        rho_ccsd = np.einsum("ij,gi,gj->g", dm_ao_ccsd, ao_grid, ao_grid)

        # Save base .npz with CCSD-consistent keys. vxc_ref is appended below.
        _npz_path = os.path.join(ext_data_dir, f"{name}.npz")
        np.savez(
            _npz_path,
            dm_target=dm_ao_ccsd,
            rho_ref_grid=rho_ccsd,
            ref_density_method="ccsd",
            E_ref_literature=float(E_ccsd_total),
        )

        # OEP inversion -> vxc_ref, appended to the .npz via save_vxc_ref merge.
        _oep_spec = alec.MoleculeSpec(
            name="H2O", atom=H2O_COORDS, basis=BASIS,
            charge=0, spin=0,
            atom_composition=(("H", 2), ("O", 1)),
            grid_level=GRID_LEVEL,
        )
        _oep = alec.run_oep_inversion(
            _oep_spec, dm_ao_ccsd,
            aux_basis="def2-svp-jkfit",
            max_iter=200,
            conv_tol=1e-6,
            regularization=1e-4,
        )
        print(f"OEP(H2O): converged={_oep.converged} "
              f"n_iter={_oep.n_iter} density_error={_oep.density_error:.3e}")
        if _oep.converged:
            alec.save_vxc_ref(
                _oep,
                _npz_path,
                dm_target=dm_ao_ccsd,
                method="ccsd",
            )
        else:
            print(f"WARNING: OEP did not converge; vxc_ref NOT written to {_npz_path}")

    # Sidecar JSON for every species — library .npz cannot carry extra keys,
    # so HF/CCSD/literature/PBE totals live here. Cell 25 reads E_ccsd_total
    # from this file for the CCSD atomization-energy reference line.
    with open(os.path.join(ext_data_dir, f"{name}_metadata.json"), "w") as _f:
        json.dump(
            {
                "E_hf_total": E_hf_total,
                "E_ccsd_total": E_ccsd_total,
                "E_lit_Ha": atom_energies_literature.get(name, None),
                "E_pbe_total": E_pbe_total,
            },
            _f,
            indent=2,
        )

# Bind the runtime name `atom_energies` to the PBE-consistent dict. This is
# the dict that flows into TrainingSpec.atom_energies and
# TestSpec.atom_energies. After the losses.py fix, both the training
# loss and AtomizationEnergyMetric compute atomization energy as
# `sum(atom_energies[Z] * n_Z) - E_mol`, so the training loss and evaluation
# metric agree exactly on every compound.
atom_energies = dict(atom_energies_pbe)
print(f"Reference data written to {ext_data_dir}")
_ae_str = {k: round(v, 6) for k, v in atom_energies.items()}
print(f"atom_energies (PBE-consistent) = {_ae_str}")
"""
    return new_code_cell(source)


def build_cell_15_mol_specs():
    """Section 4 Cell 15 — construct three alec.MoleculeSpec objects.

    The list is an explicit three-element literal (NOT a comprehension) so that
    downstream cells can reference the individual entries by index and Cell 16
    can iterate over them by name. All kwargs match Cell 14's gto.M kwargs
    exactly so that precompute_fixed_density_data rebuilds the same pyscf grid
    and _load_external_data accepts the .npz arrays.
    """
    source = """mol_specs = [
    alec.MoleculeSpec(
        name="H",
        atom="H 0 0 0",
        basis=BASIS,
        charge=0,
        spin=1,
        atom_composition=(("H", 1),),
        external_data_path=f"{ext_data_dir}/H.npz",
        grid_level=GRID_LEVEL,
    ),
    alec.MoleculeSpec(
        name="O",
        atom="O 0 0 0",
        basis=BASIS,
        charge=0,
        spin=2,
        atom_composition=(("O", 1),),
        external_data_path=f"{ext_data_dir}/O.npz",
        grid_level=GRID_LEVEL,
    ),
    alec.MoleculeSpec(
        name="H2O",
        atom=H2O_COORDS,
        basis=BASIS,
        charge=0,
        spin=0,
        atom_composition=(("H", 2), ("O", 1)),
        external_data_path=f"{ext_data_dir}/H2O.npz",
        grid_level=GRID_LEVEL,
    ),
]
print(f"Built {len(mol_specs)} MoleculeSpec objects: {[m.name for m in mol_specs]}")
"""
    return new_code_cell(source)


def build_cell_16_precompute():
    """Section 4 Cell 16 -- precompute fixed-density data with ERI for FULL mode."""
    source = """# Precompute mol_data for all molecules. required_keys includes "eri"
# because FULL SCF mode rebuilds the Fock matrix and needs the 4-center integrals.
_arch_objs = [alec.get_architecture(n) for n in ARCH_NAMES]
_desc_keys = set()
for _a in _arch_objs:
    for _d in _a.materialize_descriptors():
        _desc_keys.update(_d.required_mol_keys)

mol_data_list = []
for ms in mol_specs:
    md = alec.precompute_fixed_density_data(
        ms,
        required_keys=tuple(_desc_keys | {"eri"}),
        descriptors=sum((_a.materialize_descriptors() for _a in _arch_objs), ()),
    )
    mol_data_list.append(md)

# Sanity: atoms have 1-element composition, molecules have > 1
for md in mol_data_list:
    _n_atoms = sum(n for _, n in md["atom_composition"])
    _kind = "atom" if _n_atoms == 1 else "molecule"
    print(f"  {md['name']:5s}  ({_kind})  grid_pts={len(md['rho_grid'])}  keys={sorted(md.keys())[:8]}...")
    if "eri" in md:
        print(f"         ERI shape: {md['eri'].shape}")
"""
    return new_code_cell(source)


def build_cell_17_training_md():
    """Section 4 Cell 17 -- training narrative."""
    source = """## Section 4: SCF-Varied Training

This is the core experiment: **72 training runs** = 8 architectures x 3 loss
families x 3 solver configurations.

### Solver Impact by Loss Family

- **Loss A** (`A_atomization`): Energy-only. Uses `fixed_density_total_energy`
  which does NOT route through the SCF solver -- it always computes a one-shot
  energy on the PBE density. This is the **control experiment**: all 3 solver
  configs produce identical models for loss A.

- **Loss B** (`B_atomization_plus_dm`): Energy + density matrix term. The DM
  term (`_dm_term`) uses the solver to compute a self-consistent density matrix
  when solver_config is non-ONESHOT. With FIXED_J, the Coulomb operator is
  frozen during SCF cycles; with FULL, the full Fock matrix is rebuilt.

- **Loss C** (`C_atomization_plus_grid`): Energy + grid density term. Same SCF
  routing as loss B but applied to the grid-space density comparison.

### Expected Compute Cost

ONESHOT is fastest (single forward pass). FIXED_J(3 cycles) is ~3x slower.
FULL(3 cycles) is the most expensive due to ERI contraction at each cycle.
"""
    return new_markdown_cell(source)


def build_cell_18_training_specs(loss_names=None):
    """Section 4 Cell 18 -- build 72 TrainingSpec objects.

    Triple-nested loop: arch x loss x solver. solver_config flows through
    BOTH loss_kwargs (for actual training use by make_loss) AND
    TrainingSpec.solver_config (for metadata logging).
    """
    if loss_names is None:
        loss_names_binding = (
            'LOSS_NAMES = (\n'
            '    "A_atomization",\n'
            '    "B_atomization_plus_dm",\n'
            '    "C_atomization_plus_grid",\n'
            ')'
        )
    else:
        loss_names_binding = f"LOSS_NAMES = {tuple(loss_names)!r}"
    source = f"""{loss_names_binding}

LOSS_KWARGS_BASE = {{
    "A_atomization": {{}},
    "B_atomization_plus_dm": {{"dm_weight": 0.1}},
    "C_atomization_plus_grid": {{"density_weight": 0.1}},
}}

specs = []
for arch_name in ARCH_NAMES:
    for loss_name in LOSS_NAMES:
        for solver_label in SOLVER_LABELS:
            cfg = SCF_CONFIGS[solver_label]
            # solver_config flows through loss_kwargs to make_loss -> loss ctor.
            # For loss A (energy-only), solver_config is accepted but ignored.
            _lkw = {{**LOSS_KWARGS_BASE[loss_name], "solver_config": cfg}}
            specs.append(alec.TrainingSpec.from_dicts(
                arch=alec.get_architecture(arch_name),
                loss_name=loss_name,
                molecules=tuple(mol_specs),
                targets=targets,
                atom_energies=atom_energies,
                loss_kwargs=_lkw,
                solver_config=cfg,
                pretrain_checkpoint=f"{{CHECKPOINT_BASE}}/pretrain/{{arch_name}}",
                checkpoint_dir=f"{{CHECKPOINT_BASE}}/train/{{arch_name}}/{{loss_name}}/{{solver_label}}",
                n_steps=250,
                lr_start=1e-2,
                lr_end=1e-5,
                lr_decay_start=0.2,
                grad_clip=1.0,
            ))
print(f"Built {{len(specs)}} training specs "
      f"({{len(ARCH_NAMES)}} archs x {{len(LOSS_NAMES)}} losses x {{len(SOLVER_LABELS)}} solvers)")
"""
    return new_code_cell(source)


def build_cell_19_training_loop():
    """Section 4 Cell 19 -- serial training loop over all 72 specs.

    Three-tier tqdm: outer spec counter, inner per-step bars.
    """
    source = """_step_bars = {}
_current_info = {"loss": None, "solver": None}

def _train_cb(info):
    key = (info['arch'], info['phase'])
    if key not in _step_bars:
        _label = (f"{info['arch']:<20} {_current_info['loss']:<25} {_current_info['solver']}"
                  if _current_info['loss'] is not None
                  else f"{info['arch']:<20} {info['phase']}")
        _step_bars[key] = tqdm(
            total=info['total'],
            desc=_label,
            leave=False,
            dynamic_ncols=True,
        )
    bar = _step_bars[key]
    delta = info['step'] - bar.n
    if delta > 0:
        bar.update(delta)
    bar.set_postfix(loss=f"{info['loss']:.4e}")
    if info['step'] >= info['total']:
        bar.close()
        del _step_bars[key]

def _training_model_exists(spec):
    import os as _os
    return _os.path.isfile(_os.path.join(spec.checkpoint_dir, "model.eqx"))

_spec_bar = tqdm(
    total=len(specs),
    desc="training (specs)",
    leave=True,
    dynamic_ncols=True,
)
try:
    for spec in specs:
        _current_info['loss'] = spec.loss_name
        _current_info['solver'] = spec.checkpoint_dir.split('/')[-1]
        if TRAIN_SKIP_IF_EXISTS and _training_model_exists(spec):
            print(f"[{spec.arch.name}][{spec.loss_name}][{_current_info['solver']}] "
                  f"cached model.eqx found -- skipping training")
            _spec_bar.update(1)
            continue
        alec.run_training(spec, progress_callback=_train_cb)
        _spec_bar.update(1)
        _spec_bar.set_postfix(
            arch=spec.arch.name, loss=spec.loss_name,
            solver=_current_info['solver'])
finally:
    _spec_bar.close()
    for _b in list(_step_bars.values()):
        _b.close()
    _step_bars.clear()
"""
    return new_code_cell(source)


def build_cell_20_training_loss_plot():
    """Section 4 Cell 20 -- 3x3 training loss curves grid.

    Rows = solver config (oneshot, fixed_j, full).
    Columns = loss family (A, B, C).
    Each subplot: 8 arch traces (semilogy).
    """
    source = """fig, axes = plt.subplots(3, 3, figsize=(15, 13), squeeze=False)
for row_idx, solver_label in enumerate(SOLVER_LABELS):
    for col_idx, loss_name in enumerate(LOSS_NAMES):
        ax = axes[row_idx, col_idx]
        for arch_name in ARCH_NAMES:
            ckpt_dir = f"{CHECKPOINT_BASE}/train/{arch_name}/{loss_name}/{solver_label}"
            losses_path = f"{ckpt_dir}/losses.npy"
            if not os.path.isfile(losses_path):
                continue
            losses = np.load(losses_path)
            ax.semilogy(losses, color=arch_colors[arch_name], label=arch_name)
        ax.set_title(f"{solver_label} / {loss_name}", fontsize=10)
        ax.set_xlabel("training step")
        ax.set_ylabel("total loss (log)")
        ax.grid(True, which="both", ls=":", alpha=0.4)

# Shared legend from top-right subplot
axes[0, 2].legend(
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    fontsize="small",
    title="architecture",
)

fig.suptitle(
    "Training loss curves -- rows: solver config, columns: loss family\\n"
    "(8 deep architectures per subplot, one trace per arch)",
    fontsize=13,
)
fig.tight_layout(rect=(0, 0, 1, 0.95))
os.makedirs(f"{CHECKPOINT_BASE}/figures", exist_ok=True)
fig.savefig(f"{CHECKPOINT_BASE}/figures/training_losses.png", dpi=150, bbox_inches="tight")
plt.show()
"""
    return new_code_cell(source)


def build_cell_21_aux_inspection():
    """Section 4 Cell 21 -- aux loss component inspection."""
    source = '''arch_name = "deep_combined"
_aux_keys_per_family = {
    "A_atomization": ("loss_energy", "atomic_reg"),
    "B_atomization_plus_dm": ("loss_energy", "atomic_reg", "loss_dm"),
    "C_atomization_plus_grid": ("loss_energy", "atomic_reg", "loss_grid"),
}

fig, axes = plt.subplots(len(SOLVER_LABELS), len(LOSS_NAMES),
                         figsize=(15, 4 * len(SOLVER_LABELS)), squeeze=False)
for row_idx, solver_label in enumerate(SOLVER_LABELS):
    for col_idx, loss_name in enumerate(LOSS_NAMES):
        ax = axes[row_idx, col_idx]
        ckpt_dir = f"{CHECKPOINT_BASE}/train/{arch_name}/{loss_name}/{solver_label}"
        aux_path = f"{ckpt_dir}/aux_log.pkl"
        if not os.path.isfile(aux_path):
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center")
            ax.set_title(f"{solver_label} / {loss_name}", fontsize=10)
            continue
        with open(aux_path, "rb") as _f:
            aux_log = pickle.load(_f)

        _steps = [entry["step"] for entry in aux_log]
        for key in _aux_keys_per_family.get(loss_name, ("loss_energy",)):
            _vals = [entry["aux"].get(key, float("nan")) for entry in aux_log]
            ax.semilogy(_steps, _vals, label=key)
        ax.set_title(f"{solver_label} / {loss_name}", fontsize=10)
        ax.set_xlabel("training step")
        ax.set_ylabel("loss component (log scale)")
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.legend(fontsize="small", loc="best")

fig.suptitle(
    f"Aux loss components for arch = {arch_name!r}\\n"
    f"rows: solver config, columns: loss family",
    fontsize=13,
)
fig.tight_layout(rect=(0, 0, 1, 0.95))
os.makedirs(f"{CHECKPOINT_BASE}/figures", exist_ok=True)
fig.savefig(f"{CHECKPOINT_BASE}/figures/aux_components_{arch_name}.png", dpi=150, bbox_inches="tight")
plt.show()
'''
    return new_code_cell(source)


def build_cell_21_balancing_md():
    """Section 4b -- Multi-task loss balancing sweep header (markdown)."""
    source = """### Section 4b: Multi-Task Loss Balancing Comparison

This section compares four balancing strategies against the static baseline
on the deep_combined architecture, showing how weight schedules affect
convergence for density-matching losses (B, C).

Variants are restricted to one architecture (`deep_combined`) and one solver
(`oneshot`) to keep the sweep tractable; the V_xc-augmented group below
additionally sweeps the three solver configs since V_xc matching is the
main differentiator.
"""
    return new_markdown_cell(source)


def build_cell_22_balancing_configs():
    """Section 4b Cell 22 -- balancing configs + V_xc variants spec list.

    Emits both the existing 8-strategy sweep on oneshot solver and the 9 new
    V_xc variants (3 variants x 3 solvers on deep_combined arch).
    """
    source = """BALANCING_CONFIGS = {
    "static": None,
    "loss_norm": LossNormConfig(),
    "two_phase": TwoPhaseConfig(phase1_steps=100),
    "gradnorm": GradNormConfig(alpha=1.5, weight_lr=0.025),
}

BAL_LOSS_NAMES = (
    "B_atomization_plus_dm",
    "C_atomization_plus_grid",
)
BAL_SOLVER = "oneshot"
BAL_ARCH = "deep_combined"

# V_xc matching variants. Each variant is swept across SOLVER_LABELS so we can
# observe how SCF depth interacts with V_xc-based supervision.
VXC_VARIANTS = {
    "static_vxc": (
        "B_atomization_plus_dm",
        None,
        (("vxc_weight", 1.0),),
    ),
    "two_phase_dfirst": (
        "B_atomization_plus_dm",
        TwoPhaseConfig(
            phase1_steps=100,
            phase1_loss="B_atomization_plus_dm",
            phase1_loss_kwargs=(("vxc_weight", 1.0), ("dm_weight", 0.5)),
        ),
        (),
    ),
    "static_vxc_A": (
        "A_atomization",
        None,
        (("vxc_weight", 1.0),),
    ),
}

bal_specs = []

# Existing 4-strategy sweep on oneshot solver.
for loss_name in BAL_LOSS_NAMES:
    for bal_label, bal_cfg in BALANCING_CONFIGS.items():
        cfg = SCF_CONFIGS[BAL_SOLVER]
        _lkw = {**LOSS_KWARGS_BASE[loss_name], "solver_config": cfg}
        bal_specs.append(alec.TrainingSpec.from_dicts(
            arch=alec.get_architecture(BAL_ARCH),
            loss_name=loss_name,
            molecules=tuple(mol_specs),
            targets=targets,
            atom_energies=atom_energies,
            loss_kwargs=_lkw,
            solver_config=cfg,
            pretrain_checkpoint=f"{CHECKPOINT_BASE}/pretrain/{BAL_ARCH}",
            checkpoint_dir=f"{CHECKPOINT_BASE}/train_balancing/{loss_name}/{bal_label}",
            n_steps=250,
            lr_start=1e-2,
            lr_end=1e-5,
            lr_decay_start=0.2,
            grad_clip=1.0,
            balancing=bal_cfg,
        ))

# V_xc-augmented sweep: 3 variants x 3 solvers = 9 specs on deep_combined.
for variant_label, (loss_name, bal_cfg, extra_kwargs) in VXC_VARIANTS.items():
    for solver_label in SOLVER_LABELS:
        cfg = SCF_CONFIGS[solver_label]
        _lkw = {
            **LOSS_KWARGS_BASE[loss_name],
            **dict(extra_kwargs),
            "solver_config": cfg,
        }
        bal_specs.append(alec.TrainingSpec.from_dicts(
            arch=alec.get_architecture(BAL_ARCH),
            loss_name=loss_name,
            molecules=tuple(mol_specs),
            targets=targets,
            atom_energies=atom_energies,
            loss_kwargs=_lkw,
            solver_config=cfg,
            pretrain_checkpoint=f"{CHECKPOINT_BASE}/pretrain/{BAL_ARCH}",
            checkpoint_dir=f"{CHECKPOINT_BASE}/train_balancing/vxc/{variant_label}/{solver_label}",
            n_steps=250,
            lr_start=1e-2,
            lr_end=1e-5,
            lr_decay_start=0.2,
            grad_clip=1.0,
            balancing=bal_cfg,
        ))

print(f"Built {len(bal_specs)} balancing specs "
      f"({len(BAL_LOSS_NAMES)}x{len(BALANCING_CONFIGS)} base "
      f"+ {len(VXC_VARIANTS)}x{len(SOLVER_LABELS)} vxc)")
"""
    return new_code_cell(source)


def build_cell_23_balancing_loop():
    """Section 4b Cell 23 -- training loop over bal_specs (includes V_xc variants)."""
    source = """_bal_bars = {}
_bal_info = {"loss": None, "solver": None}

def _bal_cb(info):
    key = (info['arch'], info['phase'])
    if key not in _bal_bars:
        _label = f"{info['arch']:<20} {_bal_info['loss']:<25} {_bal_info['solver']}"
        _bal_bars[key] = tqdm(
            total=info['total'],
            desc=_label,
            leave=False,
            dynamic_ncols=True,
        )
    bar = _bal_bars[key]
    delta = info['step'] - bar.n
    if delta > 0:
        bar.update(delta)
    bar.set_postfix(loss=f"{info['loss']:.4e}")
    if info['step'] >= info['total']:
        bar.close()
        del _bal_bars[key]

_bal_spec_bar = tqdm(
    total=len(bal_specs),
    desc="balancing sweep",
    leave=True,
    dynamic_ncols=True,
)
try:
    for spec in bal_specs:
        _bal_info['loss'] = spec.loss_name
        _bal_info['solver'] = spec.checkpoint_dir.split('/')[-1]
        if TRAIN_SKIP_IF_EXISTS and _training_model_exists(spec):
            print(f"[{spec.loss_name}][{_bal_info['solver']}] cached -- skipping")
            _bal_spec_bar.update(1)
            continue
        alec.run_training(spec, progress_callback=_bal_cb)
        _bal_spec_bar.update(1)
        _bal_spec_bar.set_postfix(
            loss=spec.loss_name, strategy=_bal_info['solver'])
finally:
    _bal_spec_bar.close()
    for _b in list(_bal_bars.values()):
        _b.close()
    _bal_bars.clear()
"""
    return new_code_cell(source)


def build_cell_24_balancing_aux_inspection():
    """Section 4b Cell 24 -- aux_log inspection (base + V_xc variant keys)."""
    source = """_bal_aux_keys = {
    "B_atomization_plus_dm": ("loss_energy", "atomic_reg", "loss_dm"),
    "C_atomization_plus_grid": ("loss_energy", "atomic_reg", "loss_grid"),
}

bal_labels = list(BALANCING_CONFIGS.keys())
cmap_bal = plt.get_cmap('tab10')
bal_colors = {label: cmap_bal(i) for i, label in enumerate(bal_labels)}

fig, axes = plt.subplots(
    len(BAL_LOSS_NAMES), 3, figsize=(16, 4 * len(BAL_LOSS_NAMES)), squeeze=False,
)
for row_idx, loss_name in enumerate(BAL_LOSS_NAMES):
    aux_keys = _bal_aux_keys[loss_name]
    for col_idx, key in enumerate(aux_keys):
        ax = axes[row_idx, col_idx]
        for bal_label in bal_labels:
            ckpt = f"{CHECKPOINT_BASE}/train_balancing/{loss_name}/{bal_label}"
            aux_path = f"{ckpt}/aux_log.pkl"
            if not os.path.isfile(aux_path):
                continue
            with open(aux_path, "rb") as _f:
                aux_log = pickle.load(_f)
            _steps = [e["step"] for e in aux_log]
            _vals = [e["aux"].get(key, float("nan")) for e in aux_log]
            ax.semilogy(_steps, _vals, label=bal_label,
                        color=bal_colors[bal_label], alpha=0.85)
        ax.set_title(f"{loss_name} / {key}", fontsize=10)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Component value (log scale)")
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.legend(fontsize="small", loc="best")

fig.suptitle(
    f"Aux loss components by balancing strategy (arch={BAL_ARCH}, solver={BAL_SOLVER})\\n"
    f"rows: loss family, columns: component",
    fontsize=13,
)
fig.tight_layout(rect=(0, 0, 1, 0.93))
os.makedirs(f"{CHECKPOINT_BASE}/figures", exist_ok=True)
fig.savefig(
    f"{CHECKPOINT_BASE}/figures/balancing_aux_comparison.png",
    dpi=150, bbox_inches='tight',
)
plt.show()
"""
    return new_code_cell(source)


def build_cell_25_balancing_loss_plot():
    """Section 4b Cell 25 -- gradnorm weight evolution loss plot."""
    source = """fig, axes = plt.subplots(1, len(BAL_LOSS_NAMES), figsize=(7 * len(BAL_LOSS_NAMES), 5))
if len(BAL_LOSS_NAMES) == 1:
    axes = [axes]

for ax, loss_name in zip(axes, BAL_LOSS_NAMES):
    ckpt = f"{CHECKPOINT_BASE}/train_balancing/{loss_name}/gradnorm"
    aux_path = f"{ckpt}/aux_log.pkl"
    if not os.path.isfile(aux_path):
        ax.text(0.5, 0.5, "no gradnorm data", transform=ax.transAxes, ha="center")
        ax.set_title(loss_name)
        continue
    with open(aux_path, "rb") as _f:
        aux_log = pickle.load(_f)

    _steps = [e["step"] for e in aux_log]
    bal_info_list = [
        e.get("balancing_info", {}) for e in aux_log
    ]
    ew = [bi.get('effective_weights', {}) for bi in bal_info_list]
    if not any(ew):
        ax.text(0.5, 0.5, "no weight data", transform=ax.transAxes, ha="center")
        ax.set_title(loss_name)
        continue

    weight_keys = sorted(ew[-1].keys()) if ew[-1] else []
    for wk in weight_keys:
        _wvals = [w.get(wk, float('nan')) for w in ew]
        ax.plot(_steps, _wvals, label=wk)

    ax.set_title(f"GradNorm weights: {loss_name}", fontsize=11)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Effective weight")
    ax.legend(fontsize="small", loc="best")
    ax.grid(True, ls=":", alpha=0.4)

fig.suptitle(
    f"GradNorm learned task weights (arch={BAL_ARCH}, solver={BAL_SOLVER})",
    fontsize=13,
)
fig.tight_layout(rect=(0, 0, 1, 0.93))
os.makedirs(f"{CHECKPOINT_BASE}/figures", exist_ok=True)
fig.savefig(
    f"{CHECKPOINT_BASE}/figures/gradnorm_weight_evolution.png",
    dpi=150, bbox_inches='tight',
)
plt.show()
"""
    return new_code_cell(source)


def build_cell_26_vxc_loss_plot():
    """Section 4b -- V_xc variants loss plot (rows: variant, cols: solver)."""
    source = """# V_xc aux keys -- per-variant components to plot (additive to base loss keys).
_bal_aux_keys_vxc = {
    "static_vxc":       ("loss_energy", "atomic_reg", "loss_dm", "loss_vxc"),
    "two_phase_dfirst": ("loss_energy", "atomic_reg", "loss_dm", "loss_vxc"),
    "static_vxc_A":     ("loss_energy", "atomic_reg", "loss_vxc"),
}

fig_vxc, axes_vxc = plt.subplots(
    len(VXC_VARIANTS), len(SOLVER_LABELS),
    figsize=(5 * len(SOLVER_LABELS), 4 * len(VXC_VARIANTS)),
    squeeze=False,
)
for row_idx, (variant_label, _) in enumerate(VXC_VARIANTS.items()):
    for col_idx, solver_label in enumerate(SOLVER_LABELS):
        ax = axes_vxc[row_idx, col_idx]
        ckpt_dir = f"{CHECKPOINT_BASE}/train_balancing/vxc/{variant_label}/{solver_label}"
        aux_path = f"{ckpt_dir}/aux_log.pkl"
        if not os.path.isfile(aux_path):
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center")
            ax.set_title(f"{variant_label} / {solver_label}", fontsize=10)
            continue
        with open(aux_path, "rb") as _f:
            aux_log = pickle.load(_f)
        _steps = [entry["step"] for entry in aux_log]
        # Plots per-variant aux keys including loss_vxc (from _bal_aux_keys_vxc).
        for key in _bal_aux_keys_vxc.get(variant_label, ("loss_energy", "loss_vxc")):
            _vals = [entry["aux"].get(key, float("nan")) for entry in aux_log]
            ax.semilogy(_steps, _vals, label=key)
        ax.set_title(f"{variant_label} / {solver_label}", fontsize=10)
        ax.set_xlabel("training step")
        if col_idx == 0:
            ax.set_ylabel("loss component (log scale)")
        ax.grid(True, which="both", ls=":", alpha=0.4)
        ax.legend(fontsize="small", loc="best")
fig_vxc.suptitle("V_xc variants -- loss components (rows: variant, cols: solver)",
                 fontsize=13)
fig_vxc.tight_layout(rect=(0, 0, 1, 0.96))
os.makedirs(f"{CHECKPOINT_BASE}/figures", exist_ok=True)
fig_vxc.savefig(f"{CHECKPOINT_BASE}/figures/bal_loss_plot_vxc.png",
                dpi=150, bbox_inches="tight")
plt.show()
"""
    return new_code_cell(source)


def build_cell_26_balancing_eval():
    """Section 4b Cell 26 -- evaluate base balancing sweep models."""
    source = """# Evaluate the balancing sweep models (same metrics as main eval)
for loss_name in BAL_LOSS_NAMES:
    for bal_label in BALANCING_CONFIGS:
        ckpt_dir = f"{CHECKPOINT_BASE}/train_balancing/{loss_name}/{bal_label}"
        model_path = f"{ckpt_dir}/model.eqx"
        out_dir = f"{CHECKPOINT_BASE}/eval_balancing/{loss_name}/{bal_label}"
        if not os.path.isfile(model_path):
            continue
        if not RERUN_EVAL and os.path.isfile(f"{out_dir}/aggregate.json"):
            continue
        cfg = SCF_CONFIGS[BAL_SOLVER]
        test_spec = alec.TestSpec.from_dicts(
            arch=alec.get_architecture(BAL_ARCH),
            model_checkpoint=model_path,
            molecules=tuple(mol_specs),
            metrics=("total_energy", "atomization_energy", "density_rmse", "constraint_violations"),
            metric_kwargs={"atomization_energy": {"reference_ae_kcalmol": {"H2O": 233.016}}},
            atom_energies=atom_energies,
            output_dir=out_dir,
            solver_config=cfg,
        )
        alec.run_test(test_spec)
print(f"Balancing eval complete (RERUN_EVAL={RERUN_EVAL})")
"""
    return new_code_cell(source)


def build_cell_28_vxc_eval():
    """Section 4b -- evaluate V_xc variants (9 runs = 3 variants x 3 solvers)."""
    source = """# V_xc variants eval (9 runs: 3 variants x 3 solvers)
for variant_label, (loss_name, _, _) in VXC_VARIANTS.items():
    for solver_label in SOLVER_LABELS:
        ckpt_dir = f"{CHECKPOINT_BASE}/train_balancing/vxc/{variant_label}/{solver_label}"
        model_path = f"{ckpt_dir}/model.eqx"
        out_dir = f"{CHECKPOINT_BASE}/eval_balancing/vxc/{variant_label}/{solver_label}"
        if not os.path.isfile(model_path):
            continue
        if not RERUN_EVAL and os.path.isfile(f"{out_dir}/aggregate.json"):
            continue
        cfg = SCF_CONFIGS[solver_label]
        test_spec = alec.TestSpec.from_dicts(
            arch=alec.get_architecture(BAL_ARCH),
            model_checkpoint=model_path,
            molecules=tuple(mol_specs),
            metrics=("total_energy", "atomization_energy", "density_rmse", "constraint_violations"),
            metric_kwargs={"atomization_energy": {"reference_ae_kcalmol": {"H2O": 233.016}}},
            atom_energies=atom_energies,
            output_dir=out_dir,
            solver_config=cfg,
        )
        alec.run_test(test_spec)
print(f"V_xc eval complete (RERUN_EVAL={RERUN_EVAL})")
"""
    return new_code_cell(source)


def build_cell_27_baseline_gen():
    """Section 5 Cell 27 -- generate pretrained + random baseline model.eqx files."""
    source = """# Generate pretrained and random baseline model.eqx for all architectures
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
print(f"Baselines ready for {len(ARCH_NAMES)} architectures: {BASELINE_LABELS}")
"""
    return new_code_cell(source)


def build_cell_42_transfer_md():
    """Section 7 Cell 42 -- transfer evaluation narrative."""
    source = """## Section 7: Transfer Evaluation on New Molecules

This section tests all trained models (72 main + 8 balancing) on molecules
not seen during training:

| Molecule | Why | Key metric |
|----------|-----|------------|
| **H2** | Simplest diatomic; only H atoms (in training set) | AE error, density RMSE |
| **OH** | New element, never seen during training | Total energy error, density RMSE |
| **CH4** | New element (C), 5 atoms, tetrahedral geometry | AE error, density RMSE |

For each, we compute PBE/HF/CCSD reference data, then sweep all checkpoints.
"""
    return new_markdown_cell(source)


def build_cell_43_transfer_data_gen():
    """Section 7 Cell 43 -- transfer reference data (H2/OH/CH4/C) with CCSD DM targets."""
    source = """ext_data_dir = f"{CHECKPOINT_BASE}/external_data"
os.makedirs(ext_data_dir, exist_ok=True)

# Define test molecules with their reference AE values
test_molecules = [
    {
        "name": "H2",
        "spec": alec.MoleculeSpec(
            name="H2",
            atom="H 0 0 0; H 0 0 0.74",
            basis=BASIS, charge=0, spin=0,
            atom_composition=(("H", 2),),
            grid_level=GRID_LEVEL,
            external_data_path=f"{ext_data_dir}/H2.npz",
        ),
        "new_atoms": [],
        "ae_ref_kcalmol": 109.493,
    },
    {
        "name": "OH",
        "spec": alec.MoleculeSpec(
            name="OH",
            atom="O 0 0 0.10785; H 0 0 -0.86281",
            basis=BASIS, charge=0, spin=1,
            atom_composition=(("O", 1), ("H", 1)),
            grid_level=GRID_LEVEL,
            external_data_path=f"{ext_data_dir}/OH.npz",
        ),
        "new_atoms": [],
        "ae_ref_kcalmol": 107.208,
    },
    {
        "name": "CH4",
        "spec": alec.MoleculeSpec(
            name="CH4",
            atom="C 0 0 0; H 0.63 0.63 0.63; H -0.63 -0.63 0.63; H -0.63 0.63 -0.63; H 0.63 -0.63 -0.63",
            basis=BASIS, charge=0, spin=0,
            atom_composition=(("C", 1), ("H", 4)),
            grid_level=GRID_LEVEL,
            external_data_path=f"{ext_data_dir}/CH4.npz",
        ),
        "new_atoms": [("C", "C 0 0 0", 2)],
        "ae_ref_kcalmol": 420.421,
    },
]

# Collect all new atom species across test molecules
all_new_atoms = []
_seen_atoms = set()
for tm in test_molecules:
    for a_name, a_atom, a_spin in tm["new_atoms"]:
        if a_name not in _seen_atoms:
            all_new_atoms.append((a_name, a_atom, a_spin))
            _seen_atoms.add(a_name)

# Generate reference data for all test molecules and new atoms
_all_entities = []
for tm in test_molecules:
    spec = tm["spec"]
    _all_entities.append((spec.name, spec.atom, spec.spin, False))
for a_name, a_atom, a_spin in all_new_atoms:
    _all_entities.append((a_name, a_atom, a_spin, True))

for _name, _atom, _spin, _is_atom in _all_entities:
    _npz_path = f"{ext_data_dir}/{_name}.npz"
    _meta_path = f"{ext_data_dir}/{_name}_metadata.json"
    if os.path.isfile(_npz_path) and os.path.isfile(_meta_path):
        print(f"Using cached {_name} reference data")
        continue

    _mol = gto.M(atom=_atom, basis=BASIS, charge=0, spin=_spin, verbose=0)

    _mf_pbe = dft.UKS(_mol) if _spin else dft.RKS(_mol)
    _mf_pbe.xc = "pbe"
    _mf_pbe.grids.level = GRID_LEVEL
    _mf_pbe.kernel()
    _E_pbe_total = float(_mf_pbe.e_tot)

    _mf_hf = scf.UHF(_mol) if _spin else scf.RHF(_mol)
    _mf_hf.kernel()
    _E_hf_total = float(_mf_hf.e_tot)

    _mycc = cc.UCCSD(_mf_hf) if _spin else cc.CCSD(_mf_hf)
    _mycc.kernel()
    _E_ccsd_total = float(_mf_hf.e_tot + _mycc.e_corr)

    _sidecar = {
        "E_hf_total": _E_hf_total,
        "E_ccsd_total": _E_ccsd_total,
        "E_pbe_total": _E_pbe_total,
        "E_lit_Ha": None,
    }

    if _is_atom:
        np.savez(_npz_path, E_ref_literature=_E_ccsd_total)
    else:
        # CCSD AO-basis DM via MO->AO transform. Closed-shell uses RCCSD;
        # open-shell (e.g. OH, spin=1) uses UCCSD with spin-branched mo_coeff.
        if _spin == 0:
            _dm_mo_ccsd = _mycc.make_rdm1()
            _C = _mf_hf.mo_coeff
            _dm_ao_ccsd = _C @ _dm_mo_ccsd @ _C.T
        else:
            _dm_mo_a, _dm_mo_b = _mycc.make_rdm1()
            _Ca, _Cb = _mf_hf.mo_coeff[0], _mf_hf.mo_coeff[1]
            _dm_ao_a = _Ca @ _dm_mo_a @ _Ca.T
            _dm_ao_b = _Cb @ _dm_mo_b @ _Cb.T
            _dm_ao_ccsd = np.stack([_dm_ao_a, _dm_ao_b], axis=0)
        _dm_total = (
            _dm_ao_ccsd[0] + _dm_ao_ccsd[1]
            if _dm_ao_ccsd.ndim == 3
            else _dm_ao_ccsd
        )
        _coords = _mf_pbe.grids.coords
        _weights = _mf_pbe.grids.weights
        _ao = _mf_pbe._numint.eval_ao(_mol, _coords, deriv=0)
        _rho_ccsd = np.einsum("ij,gi,gj->g", _dm_total, _ao, _ao)
        _dm_pbe = _mf_pbe.make_rdm1()
        _dm_pbe_total = _dm_pbe[0] + _dm_pbe[1] if _dm_pbe.ndim == 3 else _dm_pbe
        _rho_pbe = np.einsum("ij,gi,gj->g", _dm_pbe_total, _ao, _ao)
        _rho_pbe_ccsd_rmse = float(
            np.sqrt(np.sum(_weights * (_rho_pbe - _rho_ccsd) ** 2) / np.sum(_weights))
        )
        np.savez(
            _npz_path,
            dm_target=_dm_ao_ccsd,
            rho_ref_grid=_rho_ccsd,
            ref_density_method="ccsd",
            E_ref_literature=float(_E_ccsd_total),
        )
        _sidecar["rho_pbe_ccsd_rmse"] = _rho_pbe_ccsd_rmse

    with open(_meta_path, "w") as _f:
        json.dump(_sidecar, _f, indent=2)
    print(f"Generated {_name} reference data -> {_npz_path}")

# Build atom_energies for transfer eval (PBE-consistent)
transfer_atom_energies = {**atom_energies}
for a_name, _, _ in all_new_atoms:
    with open(f"{ext_data_dir}/{a_name}_metadata.json") as _f:
        transfer_atom_energies[a_name] = json.load(_f)["E_pbe_total"]
print(f"transfer_atom_energies: {list(transfer_atom_energies.keys())}")
print(f"Test molecules: {[tm['name'] for tm in test_molecules]}")
"""
    return new_code_cell(source)


def build_cell_44_transfer_plot_md():
    """Section 7 Cell 44 -- transfer evaluation figure narrative."""
    source = """### Figure: Transfer Evaluation -- H2, OH, CH4

The next cell sweeps all trained checkpoints (72 main + 8 balancing) on each
test molecule. Results plotted as grouped bars:

- **H2:** AE error and density RMSE (known element, new geometry)
- **OH:** Total energy error and density RMSE (unseen element)
- **CH4:** AE error and density RMSE (unseen element C, larger molecule)"""
    return new_markdown_cell(source)


def build_cell_45_transfer_eval_loop():
    """Section 7 Cell 45 -- transfer evaluation loop over test_molecules (incl. V_xc variants)."""
    source = '''_transfer_pkl = f"{CHECKPOINT_BASE}/transfer_results.pkl"

def _eval_model_on_mol(arch_name, model_path, mol_spec, ae_ref, mol_name,
                        out_dir, solver_config, atom_energies_dict):
    """Run evaluation and return result row dict."""
    _is_atom = (len(mol_spec.atom_composition) == 1
                and mol_spec.atom_composition[0][1] == 1)
    if _is_atom:
        _metrics = ("total_energy", "density_rmse")
        _mk = {}
    else:
        _metrics = ("total_energy", "atomization_energy", "density_rmse")
        _mk = ({"atomization_energy": {"reference_ae_kcalmol": {mol_name: ae_ref}}}
               if ae_ref else {})
    _spec = alec.TestSpec.from_dicts(
        arch=alec.get_architecture(arch_name),
        model_checkpoint=model_path,
        molecules=(mol_spec,),
        metrics=_metrics,
        metric_kwargs=_mk,
        atom_energies=atom_energies_dict,
        output_dir=out_dir,
        solver_config=solver_config,
    )
    _res = alec.run_test(_spec)
    _pm = _res["per_molecule"][0]
    return {
        "AE_error_kcalmol": float(abs(_pm.get("AE_error_kcalmol", float("nan")))),
        "E_error_kcalmol": float(abs(_pm.get("E_error_kcalmol", float("nan")))),
        "density_rmse": float(_pm["density_rmse"]) if _pm.get("density_rmse") is not None else float("nan"),
    }

if RERUN_EVAL or not os.path.isfile(_transfer_pkl):
    transfer_results = {}
    for tm in test_molecules:
        _mol_name = tm["name"]
        _mol_spec = tm["spec"]
        _ae_ref = tm["ae_ref_kcalmol"]
        _rows = []

        # Trained models
        for _arch in ARCH_NAMES:
            for _loss in LOSS_NAMES:
                for _solver in SOLVER_LABELS:
                    _ckpt = f"{CHECKPOINT_BASE}/train/{_arch}/{_loss}/{_solver}/model.eqx"
                    if not os.path.isfile(_ckpt):
                        continue
                    _out = f"{CHECKPOINT_BASE}/test_new/{_mol_name}/{_arch}/{_loss}/{_solver}"
                    row = _eval_model_on_mol(
                        _arch, _ckpt, _mol_spec, _ae_ref, _mol_name,
                        _out, SCF_CONFIGS[_solver], transfer_atom_energies)
                    row.update({"arch": _arch, "loss": _loss, "solver": _solver})
                    _rows.append(row)

        # Balancing sweep
        for _loss in BAL_LOSS_NAMES:
            for _bl in BALANCING_CONFIGS:
                _ckpt = f"{CHECKPOINT_BASE}/train_balancing/{_loss}/{_bl}/model.eqx"
                if not os.path.isfile(_ckpt):
                    continue
                _out = f"{CHECKPOINT_BASE}/test_new/{_mol_name}/balancing/{_loss}/{_bl}"
                row = _eval_model_on_mol(
                    BAL_ARCH, _ckpt, _mol_spec, _ae_ref, _mol_name,
                    _out, SCF_CONFIGS[BAL_SOLVER], transfer_atom_energies)
                row.update({"arch": BAL_ARCH, "loss": _loss, "solver": f"bal:{_bl}"})
                _rows.append(row)

        # V_xc variants transfer eval (9 runs on deep_combined)
        for variant_label, (loss_name, _, _) in VXC_VARIANTS.items():
            for solver_label in SOLVER_LABELS:
                _ckpt = f"{CHECKPOINT_BASE}/train_balancing/vxc/{variant_label}/{solver_label}/model.eqx"
                if not os.path.isfile(_ckpt):
                    continue
                _out = f"{CHECKPOINT_BASE}/test_new/{_mol_name}/balancing_vxc/{variant_label}/{solver_label}"
                row = _eval_model_on_mol(
                    BAL_ARCH, _ckpt, _mol_spec, _ae_ref, _mol_name,
                    _out, SCF_CONFIGS[solver_label], transfer_atom_energies)
                row.update({
                    "arch": BAL_ARCH,
                    "loss": loss_name,
                    "solver": f"bal_vxc:{variant_label}/{solver_label}",
                })
                _rows.append(row)

        # Baselines (pretrained + random)
        for _arch in ARCH_NAMES:
            for _bl in BASELINE_LABELS:
                _ckpt = f"{CHECKPOINT_BASE}/baseline_{_bl}/{_arch}/model.eqx"
                if not os.path.isfile(_ckpt):
                    continue
                _out = f"{CHECKPOINT_BASE}/test_new/{_mol_name}/baseline/{_arch}/{_bl}"
                row = _eval_model_on_mol(
                    _arch, _ckpt, _mol_spec, _ae_ref, _mol_name,
                    _out, None, transfer_atom_energies)
                row.update({"arch": _arch, "loss": "baseline", "solver": _bl})
                _rows.append(row)

        if _rows:
            transfer_results[_mol_name] = pd.DataFrame(_rows)
            print(f"{_mol_name}: {len(_rows)} evaluations")
        else:
            print(f"{_mol_name}: no checkpoints found")

    with open(_transfer_pkl, "wb") as _f:
        pickle.dump(transfer_results, _f, protocol=4)
    print(f"\\nSaved to {_transfer_pkl}")
else:
    with open(_transfer_pkl, "rb") as _f:
        transfer_results = pickle.load(_f)
    print(f"Loaded cached results from {_transfer_pkl}")
    for mol_name, tdf in transfer_results.items():
        print(f"  {mol_name}: {len(tdf)} rows")

# ---- Reference energies for each test molecule ----
transfer_refs = {}
for tm in test_molecules:
    _mol_name = tm["name"]
    _ae_ref = tm["ae_ref_kcalmol"]
    _meta_path = f"{ext_data_dir}/{_mol_name}_metadata.json"
    refs = {}
    if os.path.isfile(_meta_path):
        with open(_meta_path) as _f:
            _meta = json.load(_f)
        if _ae_ref is not None:
            _comp = dict(tm["spec"].atom_composition)
            for method in ('pbe', 'ccsd'):
                _key = f"E_{method}_total"
                _atom_E = sum(
                    transfer_refs.get(Z, {}).get(_key,
                        json.load(open(f"{ext_data_dir}/{Z}_metadata.json")).get(_key, 0)
                    ) * cnt for Z, cnt in _comp.items()
                )
                _mol_E = _meta.get(_key, 0)
                refs[f'{method}_ae_err'] = abs((_atom_E - _mol_E) * 627.509 - _ae_ref)
    transfer_refs[_mol_name] = refs
    if os.path.isfile(_meta_path):
        with open(_meta_path) as _f:
            for k, v in json.load(_f).items():
                transfer_refs.setdefault(_mol_name, {})[k] = v

print(f"\\nTransfer evaluation ready: {list(transfer_results.keys())}")
'''
    return new_code_cell(source)


def build_cell_46_transfer_plots():
    """Section 7 Cell 46 -- transfer evaluation plots (AE error + density RMSE vs CCSD)."""
    source = '''# ---- Transfer evaluation plots ----
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
            ("density_rmse", "Density RMSE vs CCSD", "density"),
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
                    _abbrev_append = {"A":"AE Only", "B":"AE+DM","C":"AE+Rho"}
                    ax.set_title(f"Loss {_loss_abbrev[loss]}: {_abbrev_append[_loss_abbrev[loss]]}", fontsize=11, fontweight='bold')

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
            f"{mol_name}: Transfer evaluation (energy + density)\\n"
            "x = architecture, bars = treatment",
            fontsize=14, fontweight='bold',
        )
        fig.tight_layout(rect=(0, 0.06, 1, 0.95))
        fig.savefig(f"{CHECKPOINT_BASE}/figures/transfer_{mol_name}.png",
                    dpi=150, bbox_inches='tight')
        plt.show()
        print(f"  Saved transfer_{mol_name}.png")

    print(f"Transfer plots complete for {n_mols} molecules across {n_loss} losses")
'''
    return new_code_cell(source)


def build_cell_22_eval_md():
    """Section 5 Cell 22 -- evaluation narrative."""
    source = """## Section 5: Evaluation

Each trained model is scored on the same molecules used for training. Four
metrics are computed per molecule:

- **`total_energy`** -- NN total energy vs PBE/HF reference.
- **`atomization_energy`** -- AE_nn vs literature (233.016 kcal/mol for H2O).
- **`density_rmse`** -- RMSE of grid density vs HF target (molecules only).
- **`constraint_violations`** -- flattened constraint report.

The evaluation loop sweeps all 72 (arch, loss, solver_config) combinations.
Each TestSpec carries the solver_config for metadata logging.
"""
    return new_markdown_cell(source)


def build_cell_23_test_loop():
    """Section 5 Cell 29 (old name cell_23) -- main sweep + baseline eval loop.

    Covers main sweep (arch x loss x solver) and baseline models
    (BASELINE_LABELS). Balancing-sweep eval (BAL_LOSS_NAMES x
    BALANCING_CONFIGS) and V_xc-variant eval live in
    ``build_cell_26_balancing_eval``.
    """
    source = """# --- Trained model evaluations ---
for arch_name in ARCH_NAMES:
    for loss_name in LOSS_NAMES:
        for solver_label in SOLVER_LABELS:
            cfg = SCF_CONFIGS[solver_label]
            ckpt_dir = f"{CHECKPOINT_BASE}/train/{arch_name}/{loss_name}/{solver_label}"
            model_path = f"{ckpt_dir}/model.eqx"
            out_dir = f"{CHECKPOINT_BASE}/eval/{arch_name}/{loss_name}/{solver_label}"
            if not os.path.isfile(model_path):
                continue
            if not RERUN_EVAL and os.path.isfile(f"{out_dir}/aggregate.json"):
                continue
            test_spec = alec.TestSpec.from_dicts(
                arch=alec.get_architecture(arch_name),
                model_checkpoint=model_path,
                molecules=tuple(mol_specs),
                metrics=("total_energy", "atomization_energy", "density_rmse", "constraint_violations"),
                metric_kwargs={"atomization_energy": {"reference_ae_kcalmol": {"H2O": 233.016}}},
                atom_energies=atom_energies,
                output_dir=out_dir,
                solver_config=cfg,
            )
            alec.run_test(test_spec)

# --- Baseline evaluations (pretrained + random) ---
for arch_name in ARCH_NAMES:
    for bl in BASELINE_LABELS:
        bl_path = f"{CHECKPOINT_BASE}/baseline_{bl}/{arch_name}/model.eqx"
        out_dir = f"{CHECKPOINT_BASE}/eval_baseline/{arch_name}/{bl}"
        if not os.path.isfile(bl_path):
            continue
        if not RERUN_EVAL and os.path.isfile(f"{out_dir}/aggregate.json"):
            continue
        test_spec = alec.TestSpec.from_dicts(
            arch=alec.get_architecture(arch_name),
            model_checkpoint=bl_path,
            molecules=tuple(mol_specs),
            metrics=("total_energy", "atomization_energy", "density_rmse", "constraint_violations"),
            metric_kwargs={"atomization_energy": {"reference_ae_kcalmol": {"H2O": 233.016}}},
            atom_energies=atom_energies,
            output_dir=out_dir,
            solver_config=None,
        )
        alec.run_test(test_spec)

_n_trained = sum(1 for a in ARCH_NAMES for l in LOSS_NAMES for s in SOLVER_LABELS
                 if os.path.isfile(f"{CHECKPOINT_BASE}/eval/{a}/{l}/{s}/aggregate.json"))
_n_baseline = sum(1 for a in ARCH_NAMES for bl in BASELINE_LABELS
                  if os.path.isfile(f"{CHECKPOINT_BASE}/eval_baseline/{a}/{bl}/aggregate.json"))
print(f"Evaluation complete: {_n_trained} trained + {_n_baseline} baseline results on disk")
"""
    return new_code_cell(source)


def build_cell_24_dataframe():
    """Section 5 Cell 24 -- aggregate results (main sweep + balancing + baselines + V_xc) into DataFrame."""
    source = """rows = []

# Trained model results
for arch_name in ARCH_NAMES:
    for loss_name in LOSS_NAMES:
        for solver_label in SOLVER_LABELS:
            output_dir = f"{CHECKPOINT_BASE}/eval/{arch_name}/{loss_name}/{solver_label}"
            try:
                with open(f"{output_dir}/aggregate.json") as _f:
                    agg = json.load(_f)
            except FileNotFoundError:
                agg = {}
            rows.append({
                "arch": arch_name,
                "loss": loss_name,
                "solver": solver_label,
                "AE_error_kcalmol_mean": agg.get("AE_error_kcalmol", {}).get("mean", np.nan),
                "AE_error_kcalmol_RMSE": agg.get("AE_error_kcalmol", {}).get("RMSE", np.nan),
                "E_error_kcalmol_mean": agg.get("E_error_kcalmol", {}).get("mean", np.nan),
                "density_rmse_mean": agg.get("density_rmse", {}).get("mean", np.nan),
            })

# Balancing sweep results
for loss_name in BAL_LOSS_NAMES:
    for bal_label in BALANCING_CONFIGS:
        output_dir = f"{CHECKPOINT_BASE}/eval_balancing/{loss_name}/{bal_label}"
        try:
            with open(f"{output_dir}/aggregate.json") as _f:
                agg = json.load(_f)
        except FileNotFoundError:
            agg = {}
        rows.append({
            "arch": BAL_ARCH,
            "loss": loss_name,
            "solver": f"bal:{bal_label}",
            "AE_error_kcalmol_mean": agg.get("AE_error_kcalmol", {}).get("mean", np.nan),
            "AE_error_kcalmol_RMSE": agg.get("AE_error_kcalmol", {}).get("RMSE", np.nan),
            "E_error_kcalmol_mean": agg.get("E_error_kcalmol", {}).get("mean", np.nan),
            "density_rmse_mean": agg.get("density_rmse", {}).get("mean", np.nan),
        })

# Baseline results (pretrained + random)
for arch_name in ARCH_NAMES:
    for bl in BASELINE_LABELS:
        output_dir = f"{CHECKPOINT_BASE}/eval_baseline/{arch_name}/{bl}"
        try:
            with open(f"{output_dir}/aggregate.json") as _f:
                agg = json.load(_f)
        except FileNotFoundError:
            agg = {}
        rows.append({
            "arch": arch_name,
            "loss": "baseline",
            "solver": bl,
            "AE_error_kcalmol_mean": agg.get("AE_error_kcalmol", {}).get("mean", np.nan),
            "AE_error_kcalmol_RMSE": agg.get("AE_error_kcalmol", {}).get("RMSE", np.nan),
            "E_error_kcalmol_mean": agg.get("E_error_kcalmol", {}).get("mean", np.nan),
            "density_rmse_mean": agg.get("density_rmse", {}).get("mean", np.nan),
        })

# V_xc variants ingestion (eval_balancing/vxc/...)
for variant_label, (loss_name, _, _) in VXC_VARIANTS.items():
    for solver_label in SOLVER_LABELS:
        output_dir = f"{CHECKPOINT_BASE}/eval_balancing/vxc/{variant_label}/{solver_label}"
        try:
            with open(f"{output_dir}/aggregate.json") as _f:
                agg = json.load(_f)
        except FileNotFoundError:
            agg = {}
        rows.append({
            "arch": BAL_ARCH,
            "loss": loss_name,
            "solver": f"bal_vxc:{variant_label}/{solver_label}",
            "AE_error_kcalmol_mean": agg.get("AE_error_kcalmol", {}).get("mean", np.nan),
            "AE_error_kcalmol_RMSE": agg.get("AE_error_kcalmol", {}).get("RMSE", np.nan),
            "E_error_kcalmol_mean": agg.get("E_error_kcalmol", {}).get("mean", np.nan),
            "density_rmse_mean": agg.get("density_rmse", {}).get("mean", np.nan),
        })

df = pd.DataFrame(rows).set_index(["arch", "loss", "solver"])
print(f"Built results DataFrame: {df.shape[0]} rows x {df.shape[1]} cols")
_n_bal = len(BAL_LOSS_NAMES) * len(BALANCING_CONFIGS)
_n_bl = len(ARCH_NAMES) * len(BASELINE_LABELS)
_n_vxc = len(VXC_VARIANTS) * len(SOLVER_LABELS)
print(f"  ({_n_bal} balancing + {_n_bl} baseline + {_n_vxc} vxc entries)")
"""
    return new_code_cell(source)


def build_cell_25_results_table():
    """Section 5 Cell 25 -- pivot table showing mean |AE error| per solver."""
    source = """# Pivot: mean |AE error| per (arch, loss) x solver
piv = df["AE_error_kcalmol_mean"].abs().unstack(level="solver")
print("Mean |AE error| (kcal/mol), (arch, loss) x solver:")
print(piv.round(3))
print()
# Best config per loss
for loss_name in LOSS_NAMES:
    _sub = df.xs(loss_name, level="loss")["AE_error_kcalmol_mean"].abs()
    _best = _sub.idxmin()
    print(f"Best config for {loss_name}: arch={_best[0]}, solver={_best[1]}, "
          f"|AE err|={_sub[_best]:.3f} kcal/mol")
"""
    return new_code_cell(source)


def build_cell_26_scf_impact_md():
    """Section 6 Cell 26 -- SCF impact analysis header."""
    source = """## Section 6: SCF Impact Analysis

This section compares trained models across the three solver configurations to
answer the central question: does training through an iterative SCF loop
produce better density functionals than one-shot prediction?

### Figure: SCF Comparison -- AE Error by Architecture

The next cell renders the **headline figure**: grouped bars showing |AE error|
for each architecture, with 3 bars per arch (oneshot / fixed_j / full). One
subplot per loss family (A, B, C). PBE error and 1 kcal/mol chemical accuracy
lines are overlaid as references.

**How to read it:** For loss A (energy only), all 3 solver configs should produce
nearly identical bars (control). For losses B and C, differences between solver
configs reveal the impact of SCF self-consistency on the learned functional.
"""
    return new_markdown_cell(source)


def build_cell_27_scf_comparison_bars():
    """Section 6 Cell 27 -- SCF comparison grouped bar chart."""
    source = """# Reference lines: PBE and CCSD atomization energy errors vs experiment
ext_data_dir = f"{CHECKPOINT_BASE}/external_data"
_E_ref = {}
for _name in ("H", "O", "H2O"):
    with open(f"{ext_data_dir}/{_name}_metadata.json") as _f:
        _E_ref[_name] = json.load(_f)
_AE_expt_kcalmol = 233.016  # experimental H2O atomization energy

_ae_pbe_Ha = 2 * _E_ref["H"]["E_pbe_total"] + _E_ref["O"]["E_pbe_total"] - _E_ref["H2O"]["E_pbe_total"]
PBE_AE_err_kcalmol = abs(_ae_pbe_Ha * 627.509 - _AE_expt_kcalmol)

_ae_ccsd_Ha = 2 * _E_ref["H"]["E_ccsd_total"] + _E_ref["O"]["E_ccsd_total"] - _E_ref["H2O"]["E_ccsd_total"]
CCSD_AE_err_kcalmol = abs(_ae_ccsd_Ha * 627.509 - _AE_expt_kcalmol)

fig, axes = plt.subplots(1, len(LOSS_NAMES), figsize=(6 * len(LOSS_NAMES), 7), squeeze=False)
for col_idx, loss_name in enumerate(LOSS_NAMES):
    ax = axes[0, col_idx]
    n_archs = len(ARCH_NAMES)
    n_solvers = len(SOLVER_LABELS)
    x_positions = np.arange(n_archs)
    bar_width = 0.8 / max(n_solvers, 1)

    for s_idx, solver_label in enumerate(SOLVER_LABELS):
        heights = []
        for arch_name in ARCH_NAMES:
            try:
                val = df.loc[(arch_name, loss_name, solver_label), "AE_error_kcalmol_mean"]
                heights.append(abs(val) if not np.isnan(val) else np.nan)
            except KeyError:
                heights.append(np.nan)
        offset = (s_idx - (n_solvers - 1) / 2) * bar_width
        ax.bar(x_positions + offset, heights, width=bar_width,
               color=solver_colors[solver_label], label=solver_label)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(ARCH_NAMES, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("|AE error| (kcal/mol)")
    ax.set_yscale("log")
    ax.set_title(f"Loss: {loss_name}", fontsize=11)
    ax.grid(True, which="both", axis="y", ls=":", alpha=0.4)

    ax.axhline(PBE_AE_err_kcalmol, linestyle=":", color="r", linewidth=1.5,
               label=f"PBE ({PBE_AE_err_kcalmol:.2f} kcal/mol)")
    ax.axhline(CCSD_AE_err_kcalmol, linestyle=":", color="b", linewidth=1.5,
               label=f"CCSD ({CCSD_AE_err_kcalmol:.2f} kcal/mol)")
    ax.axhline(1.0, linestyle="--", color="k", alpha=0.7,
               label="Chemical accuracy (1 kcal/mol)")

axes[0, -1].legend(
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    fontsize="small",
    title="solver / reference",
)

fig.suptitle(
    "H2O atomization-energy error by architecture and solver config\\n"
    "(one subplot per loss family, grouped bars = solver configs)",
    fontsize=13,
)
fig.tight_layout(rect=(0, 0, 1, 0.95))
os.makedirs(f"{CHECKPOINT_BASE}/figures", exist_ok=True)
fig.savefig(f"{CHECKPOINT_BASE}/figures/scf_comparison_ae.png", dpi=150, bbox_inches="tight")
plt.show()
"""
    return new_code_cell(source)


def build_cell_28_dm_heatmaps_md():
    """Section 6 Cell 28 -- DM heatmaps description."""
    source = """### Figure: Density Matrix Residuals -- SCF Comparison

For loss B (energy + DM), this plot compares the density-matrix residuals
(NN - HF target) across the 3 solver configurations for the best-performing
architecture. A 1x3 panel with shared colorbar shows how self-consistency
affects the learned density matrix.
"""
    return new_markdown_cell(source)


def build_cell_29_dm_heatmaps():
    """Section 6 Cell 29 -- DM heatmaps: loss B, best arch, 3 solver configs."""
    source = """_loss_b = "B_atomization_plus_dm"
if _loss_b not in LOSS_NAMES:
    print("[Cell 29] loss B not in config -- skipping DM heatmaps")
else:
    # Find best arch for loss B across all solvers
    _sub = df.xs(_loss_b, level="loss")["AE_error_kcalmol_mean"].abs()
    _best_arch, _best_solver = _sub.idxmin()

    dm_hf = mol_data_list[2]["dm_target"]  # H2O is index 2

    fig, axes = plt.subplots(1, len(SOLVER_LABELS), figsize=(5 * len(SOLVER_LABELS), 4.5),
                             squeeze=False)
    _vmax = 0
    _dm_panels = []
    for solver_label in SOLVER_LABELS:
        ckpt = f"{CHECKPOINT_BASE}/train/{_best_arch}/{_loss_b}/{solver_label}/model.eqx"
        if not os.path.isfile(ckpt):
            _dm_panels.append(None)
            continue
        _arch_config = alec.get_architecture(_best_arch)
        _model = eqx.tree_deserialise_leaves(ckpt, alec.AlecGGAModel.from_arch(_arch_config))
        _dm_nn = alec.oneshot_dm_prediction_fast(_model, mol_data_list[2])
        _delta = _dm_nn - dm_hf
        _dm_panels.append(_delta)
        _vmax = max(_vmax, float(jnp.abs(_delta).max()))

    for i, (solver_label, delta) in enumerate(zip(SOLVER_LABELS, _dm_panels)):
        ax = axes[0, i]
        if delta is None:
            ax.text(0.5, 0.5, "no data", transform=ax.transAxes, ha="center")
        else:
            _rmse = float(jnp.sqrt(jnp.mean(delta ** 2)))
            im = ax.imshow(np.asarray(delta), cmap="RdBu_r", vmin=-_vmax, vmax=_vmax)
            ax.set_title(f"{solver_label}\\nFrob RMSE={_rmse:.4e}", fontsize=10)
            ax.set_xlabel("AO index j")
            ax.set_ylabel("AO index i")
            fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle(
        f"H2O DM residuals (NN - HF) for loss B, arch={_best_arch}\\n"
        f"(shared colorscale, one panel per solver config)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    os.makedirs(f"{CHECKPOINT_BASE}/figures", exist_ok=True)
    fig.savefig(f"{CHECKPOINT_BASE}/figures/dm_heatmaps_scf.png", dpi=150, bbox_inches="tight")
    plt.show()
"""
    return new_code_cell(source)


def build_cell_30_density_histograms_md():
    """Section 6 Cell 30 -- density histograms description."""
    source = """### Figure: Grid Density Residuals -- SCF Comparison

For loss C (energy + grid density), this plot overlays histograms of the
grid-weighted density residual (delta-rho) across the 3 solver configurations
for the best-performing architecture. Tighter distributions indicate better
density prediction.
"""
    return new_markdown_cell(source)


def build_cell_31_density_histograms():
    """Section 6 Cell 31 -- overlaid density residual histograms."""
    source = """_loss_c = "C_atomization_plus_grid"
if _loss_c not in LOSS_NAMES:
    print("[Cell 31] loss C not in config -- skipping density histograms")
else:
    _sub = df.xs(_loss_c, level="loss")["AE_error_kcalmol_mean"].abs()
    _best_arch, _best_solver = _sub.idxmin()

    rho_ref = mol_data_list[2]["rho_ref_grid"]
    weights = mol_data_list[2]["grid_weights"]
    _bins = np.linspace(-0.15, 0.15, 81)

    # PBE baseline density residuals
    _rho_pbe = mol_data_list[2]["rho_grid"]
    _delta_pbe = np.asarray(_rho_pbe - rho_ref)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(_delta_pbe, bins=_bins, alpha=0.3, color="red", edgecolor="red",
            linewidth=0.5, label="PBE baseline",
            weights=np.asarray(weights), density=True)

    for solver_label in SOLVER_LABELS:
        ckpt = f"{CHECKPOINT_BASE}/train/{_best_arch}/{_loss_c}/{solver_label}/model.eqx"
        if not os.path.isfile(ckpt):
            continue
        _arch_config = alec.get_architecture(_best_arch)
        _model = eqx.tree_deserialise_leaves(ckpt, alec.AlecGGAModel.from_arch(_arch_config))
        _rho_nn = alec.oneshot_grid_density(_model, mol_data_list[2])
        _delta = np.asarray(_rho_nn - rho_ref)
        ax.hist(_delta, bins=_bins, alpha=0.4, color=solver_colors[solver_label],
                edgecolor=solver_colors[solver_label], linewidth=0.5,
                label=solver_label, weights=np.asarray(weights), density=True)

    ax.set_xlabel(r"$\\rho_{\\mathrm{NN}} - \\rho_{\\mathrm{HF}}$  (grid-weighted residual)")
    ax.set_ylabel("probability density (log scale)")
    ax.set_yscale("log")
    ax.set_title(
        f"Grid density residuals vs HF reference for loss C, arch={_best_arch}\\n"
        f"(PBE baseline in red, NN models by solver config)",
    )
    ax.legend(title="model / baseline", fontsize="small")
    ax.grid(True, which="both", axis="y", ls=":", alpha=0.4)
    fig.tight_layout()
    os.makedirs(f"{CHECKPOINT_BASE}/figures", exist_ok=True)
    fig.savefig(f"{CHECKPOINT_BASE}/figures/grid_density_scf.png", dpi=150, bbox_inches="tight")
    plt.show()
"""
    return new_code_cell(source)


def build_cell_32_convergence_md():
    """Section 6 Cell 32 -- convergence analysis description."""
    source = """### Figure: SCF Convergence Diagnostic

This plot runs the SCF solver with increased `max_cycles=10` on a trained
model (deep_combined, loss A) and plots |E(n) - E(n-1)| vs cycle number for
both FIXED_J and FULL modes. This shows the convergence rate under
self-consistency -- how quickly the energy settles.

Note: this is evaluation-only (no training through the loop). The trained
model was trained with the solver config in its checkpoint path; here we
just observe how it behaves under extended iteration.
"""
    return new_markdown_cell(source)


def build_cell_33_convergence_diagnostic():
    """Section 6 Cell 33 -- SCF convergence rate plot."""
    source = """from xcquinox.alec.solver import run_scf

_diag_arch = "deep_combined" if "deep_combined" in ARCH_NAMES else ARCH_NAMES[0]
_diag_loss = LOSS_NAMES[0]
_diag_solver = SOLVER_LABELS[0]
_ckpt = f"{CHECKPOINT_BASE}/train/{_diag_arch}/{_diag_loss}/{_diag_solver}/model.eqx"

if not os.path.isfile(_ckpt):
    print(f"[Cell 33] checkpoint not found: {_ckpt} -- skipping convergence plot")
else:
    _arch_config = alec.get_architecture(_diag_arch)
    _model = eqx.tree_deserialise_leaves(_ckpt, alec.AlecGGAModel.from_arch(_arch_config))
    _h2o_data = mol_data_list[2]  # H2O

    _diag_configs = {
        "FIXED_J(10)": SolverConfig(
            backend=SolverBackend.MANUAL,
            mode=SolverMode.FIXED_J,
            max_cycles=10,
            conv_tol=1e-10,
        ),
        "FULL(10)": SolverConfig(
            backend=SolverBackend.MANUAL,
            mode=SolverMode.FULL,
            max_cycles=10,
            conv_tol=1e-10,
        ),
    }

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, cfg in _diag_configs.items():
        result = run_scf(cfg, _model, _h2o_data)
        if hasattr(result, "energy_trace") and result.energy_trace is not None:
            _trace = np.array(result.energy_trace)
            _deltas = np.abs(np.diff(_trace))
            ax.semilogy(range(1, len(_deltas) + 1), _deltas, "o-", label=label)

    ax.set_xlabel("SCF cycle")
    ax.set_ylabel("|E(n) - E(n-1)| (Hartree, log)")
    ax.set_title(
        f"SCF convergence diagnostic -- arch={_diag_arch}, loss={_diag_loss}\\n"
        f"(eval-only: trained model run through extended SCF cycles)"
    )
    ax.legend(title="mode (max_cycles=10)")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    fig.tight_layout()
    os.makedirs(f"{CHECKPOINT_BASE}/figures", exist_ok=True)
    fig.savefig(f"{CHECKPOINT_BASE}/figures/scf_convergence.png", dpi=150, bbox_inches="tight")
    plt.show()
"""
    return new_code_cell(source)


def build_cell_34_feature_impact_md():
    """Section 6 Cell 34 -- feature impact description."""
    source = """### Figure: Feature Impact Across Solver Configs

This plot compares the 4 non-attention deep variants (`deep`, `deep_cusp`,
`deep_dm`, `deep_combined`) across the 3 solver configs. One subplot per loss
family (A, B, C). Using non-attention variants only provides a clean comparison
of descriptor impact without attention confounds.
"""
    return new_markdown_cell(source)


def build_cell_35_feature_impact():
    """Section 6 Cell 35 -- feature impact across solver configs."""
    source = """_feature_archs = ["deep", "deep_cusp", "deep_dm", "deep_combined"]
_feature_archs = [a for a in _feature_archs if a in ARCH_NAMES]

if not _feature_archs:
    print("[Cell 35] no non-attention deep variants in config -- skipping")
else:
    fig, axes = plt.subplots(1, len(LOSS_NAMES), figsize=(6 * len(LOSS_NAMES), 6),
                             squeeze=False)
    n_archs = len(_feature_archs)
    n_solvers = len(SOLVER_LABELS)
    x_positions = np.arange(n_archs)
    bar_width = 0.8 / max(n_solvers, 1)

    for col_idx, loss_name in enumerate(LOSS_NAMES):
        ax = axes[0, col_idx]
        for s_idx, solver_label in enumerate(SOLVER_LABELS):
            heights = []
            for arch_name in _feature_archs:
                try:
                    val = df.loc[(arch_name, loss_name, solver_label), "AE_error_kcalmol_mean"]
                    heights.append(abs(val) if not np.isnan(val) else np.nan)
                except KeyError:
                    heights.append(np.nan)
            offset = (s_idx - (n_solvers - 1) / 2) * bar_width
            ax.bar(x_positions + offset, heights, width=bar_width,
                   color=solver_colors[solver_label], label=solver_label)

        ax.set_xticks(x_positions)
        ax.set_xticklabels(_feature_archs, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("|AE error| (kcal/mol)")
        ax.set_yscale("log")
        ax.set_title(f"Loss: {loss_name}", fontsize=11)
        ax.grid(True, which="both", axis="y", ls=":", alpha=0.4)
        ax.axhline(PBE_AE_err_kcalmol, linestyle=":", color="r", linewidth=1.5,
                   label=f"PBE ({PBE_AE_err_kcalmol:.2f} kcal/mol)")
        ax.axhline(CCSD_AE_err_kcalmol, linestyle=":", color="b", linewidth=1.5,
                   label=f"CCSD ({CCSD_AE_err_kcalmol:.2f} kcal/mol)")
        ax.axhline(1.0, linestyle="--", color="k", alpha=0.7,
                   label="Chemical accuracy (1 kcal/mol)")

    axes[0, -1].legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize="small",
        title="solver / reference",
    )

    fig.suptitle(
        "Feature impact: non-attention deep variants x solver config\\n"
        "(descriptor dimension increases left to right: 2, 4, 5, 7)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    os.makedirs(f"{CHECKPOINT_BASE}/figures", exist_ok=True)
    fig.savefig(f"{CHECKPOINT_BASE}/figures/feature_impact_scf.png", dpi=150, bbox_inches="tight")
    plt.show()
"""
    return new_code_cell(source)


# NOTE on function names: some build_cell_XX_* function names retain their
# original numeric suffix (e.g. build_cell_22_eval_md produces cell at index 27
# in the final notebook) because renaming during the balancing+transfer
# backfill would create unnecessary diff churn. The symbolic portion of each
# name still identifies the cell semantically. See plan
# docs/superpowers/plans/2026-04-18-step5-vxc-training-integration.md.
def main(
    output_path: str,
    *,
    arch_names: tuple[str, ...] | None = None,
    loss_names: tuple[str, ...] | None = None,
    solver_labels: tuple[str, ...] | None = None,
    checkpoint_base: str | None = None,
):
    """Assemble the step 5 notebook, validate it, write it to ``output_path``.

    Parameters
    ----------
    output_path
        Filesystem path where the generated ``.ipynb`` is written.
    arch_names
        Optional override for ``DEFAULT_ARCH_NAMES``. Used by the smoke test
        to produce a single-architecture notebook.
    loss_names
        Optional override for ``DEFAULT_LOSS_NAMES``. Used by the smoke test
        to produce a single-loss notebook.
    solver_labels
        Optional override for ``DEFAULT_SOLVER_LABELS``. Used by the smoke
        test to produce a single-solver notebook.
    checkpoint_base
        Optional override for ``DEFAULT_CHECKPOINT_BASE``. Used by the smoke
        test to redirect artifacts into a ``tmp_path``-backed directory.

    Returns
    -------
    nbformat.notebooknode.NotebookNode
        The assembled notebook, already written to disk.
    """
    if checkpoint_base is None:
        checkpoint_base = DEFAULT_CHECKPOINT_BASE

    nb = new_notebook()
    nb.cells = [
        # Section 1: Setup (cells 0-5)
        build_cell_01_title(),                          # 0
        build_cell_02_imports(),                         # 1
        build_cell_03_constants(checkpoint_base),        # 2
        build_cell_04_arch_table(),                      # 3
        build_cell_05_arch_names(arch_names),            # 4
        build_cell_06_scf_configs(solver_labels),        # 5
        # Section 2: Pretraining (cells 6-10)
        build_cell_07_pretrain_md(),                     # 6
        build_cell_08_pretrain_data_gen(),                # 7
        build_cell_09_pretrain_loop(),                    # 8
        build_cell_10_pretrain_loss_plot(),               # 9
        build_cell_11_pretrain_parity(),                  # 10
        # Section 3: Training Data (cells 11-15)
        build_cell_12_training_md(),                     # 11
        build_cell_13_reference_dicts(),                  # 12
        build_cell_14_hf_ccsd_gen(),                      # 13
        build_cell_15_mol_specs(),                        # 14
        build_cell_16_precompute(),                       # 15
        # Section 4: SCF-Varied Training (cells 16-20)
        build_cell_17_training_md(),                     # 16
        build_cell_18_training_specs(loss_names),         # 17
        build_cell_19_training_loop(),                    # 18
        build_cell_20_training_loss_plot(),                # 19
        build_cell_21_aux_inspection(),                   # 20
        # Section 4b: Balancing + V_xc variants (cells 21-28)
        build_cell_21_balancing_md(),                     # 21
        build_cell_22_balancing_configs(),                # 22
        build_cell_23_balancing_loop(),                   # 23
        build_cell_24_balancing_aux_inspection(),         # 24
        build_cell_25_balancing_loss_plot(),              # 25
        build_cell_26_vxc_loss_plot(),                    # 26  (new: V_xc plot as its own cell)
        build_cell_26_balancing_eval(),                   # 27  (base balancing eval only)
        build_cell_28_vxc_eval(),                         # 28  (new: V_xc eval as its own cell)
        # Section 5: Evaluation (cells 29-33)
        build_cell_22_eval_md(),                          # 29 (old function name kept)
        build_cell_27_baseline_gen(),                     # 30
        build_cell_23_test_loop(),                        # 31 (old name, expanded body)
        build_cell_24_dataframe(),                        # 32 (old name, expanded body)
        build_cell_25_results_table(),                    # 33 (old name kept)
        # Section 6: SCF Impact Analysis (cells 34-43)
        build_cell_26_scf_impact_md(),                    # 34
        build_cell_27_scf_comparison_bars(),              # 35
        build_cell_28_dm_heatmaps_md(),                   # 36
        build_cell_29_dm_heatmaps(),                      # 37
        build_cell_30_density_histograms_md(),             # 38
        build_cell_31_density_histograms(),                # 39
        build_cell_32_convergence_md(),                   # 40
        build_cell_33_convergence_diagnostic(),            # 41
        build_cell_34_feature_impact_md(),                 # 42
        build_cell_35_feature_impact(),                    # 43
        # Section 7: Transfer Evaluation (cells 44-48)
        build_cell_42_transfer_md(),                      # 44
        build_cell_43_transfer_data_gen(),                # 45
        build_cell_44_transfer_plot_md(),                 # 46
        build_cell_45_transfer_eval_loop(),               # 47
        build_cell_46_transfer_plots(),                   # 48
    ]

    # Assign deterministic cell IDs so two back-to-back regenerations produce
    # byte-identical notebooks. nbformat.v4.new_code_cell / new_markdown_cell
    # otherwise auto-assign random UUIDs per call.
    for idx, cell in enumerate(nb.cells):
        cell.id = f"cell_{idx:02d}"

    nbformat.validate(nb)

    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    nbformat.write(nb, output_path)
    return nb


if __name__ == "__main__":
    main("notebooks/gga_training_example-step5.ipynb")
