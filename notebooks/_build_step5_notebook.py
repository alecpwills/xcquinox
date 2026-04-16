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
        build_cell_01_title(),
        build_cell_02_imports(),
        build_cell_03_constants(checkpoint_base),
        build_cell_04_arch_table(),
        build_cell_05_arch_names(arch_names),
        build_cell_06_scf_configs(solver_labels),
        build_cell_07_pretrain_md(),
        build_cell_08_pretrain_data_gen(),
        build_cell_09_pretrain_loop(),
        build_cell_10_pretrain_loss_plot(),
        build_cell_11_pretrain_parity(),
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
