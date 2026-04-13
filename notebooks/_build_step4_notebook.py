"""Generator for notebooks/gga_training_example-step4.ipynb.

The step 4 notebook is **not** hand-edited as ``.ipynb`` JSON. Every cell is
produced by a ``build_cell_NN_<topic>()`` function in this module, and
``main()`` assembles the builders into an ``nbformat`` notebook, validates it,
writes it to disk, and returns the notebook object for in-process inspection.

Regeneration is deterministic: same generator source -> byte-identical
notebook. Users must never edit the ``.ipynb`` directly; all edits go through
this module. See ``docs/superpowers/plans/2026-04-12-step4-notebook-implementation.md``
for the full contract.

Naming convention: each builder returns an ``nbformat.notebooknode.NotebookNode``
(a code cell or a markdown cell). Cell-index order in ``main()`` is the order
the notebook presents to the user.
"""
import os

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


# Module-level defaults. Tests and the smoke harness override these through
# ``main()`` kwargs to produce narrow-config notebooks. Keep the names in
# ``DEFAULT_ARCH_NAMES`` synchronized with ``xcquinox.alec.ARCHITECTURES``.
DEFAULT_ARCH_NAMES = (
    "shallow",
    "shallow_attn",
    "medium",
    "medium_attn",
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
    "D1_delta_ae",
    "D2_delta_ae_plus_dm",
    "D3_delta_ae_plus_grid",
)

DEFAULT_CHECKPOINT_BASE = "checkpoints_step4"


def build_cell_00_smoke_marker():
    """Placeholder cell used by Task 1's scaffolding test.

    This builder exists so the ``test_main_produces_valid_notebook`` test can
    round-trip ``main()`` end-to-end before any real cell builders are added.
    It is deleted entirely in Task 12.
    """
    return new_markdown_cell("# Step 4 Notebook (generated, do not edit)")


def build_cell_01_title():
    r"""Section 1 Cell 1 — title, methodology table, architecture list."""
    source = r"""# GGA Network Training - Step 4: Refactored Library-Driven Training

This notebook reproduces the Step 3b experiment (H / O / H2O at def2-svp,
12 architectures x 6 loss approaches = 72 models) using the refactored
`xcquinox.alec` subpackage. All loss, network, training, and evaluation
logic lives in the library -- this notebook is a thin orchestration layer
that builds `Spec` objects and calls `run_pretrain`, `run_training`, and
`run_test`.

## Training Methodology

| Approach | Energy Calculation | Density Matching | Description |
|----------|-------------------|------------------|-------------|
| **A** | Fixed-density | None | AE only on PBE density |
| **B** | Fixed-density | One-shot DM -> HF target | AE + DM correction learning |
| **C** | Fixed-density | One-shot grid rho -> HF target | AE + grid density correction |
| **D1** | Fixed-density | None | Delta-learning energy only |
| **D2** | Fixed-density | One-shot DM -> HF target | Delta-E + DM correction |
| **D3** | Fixed-density | One-shot grid rho -> HF target | Delta-E + grid density correction |

## Key Change from Step 3b

Step 3b inlined the loss, network, training loop, and evaluation code inside
the notebook. Step 4 delegates every step to `xcquinox.alec`:

- `alec.PretrainSpec` / `alec.run_pretrain` -- pretraining phase
- `alec.TrainingSpec.from_dicts` / `alec.run_training` -- main training phase
- `alec.TestSpec.from_dicts` / `alec.run_test` -- evaluation phase

The registry-driven composition means adding a new loss or architecture is a
single-line library change, not a notebook edit.

## Network Architectures (12 total)

**Standard (2 inputs: rho, sigma):**
`shallow`, `shallow_attn`, `medium`, `medium_attn`, `deep`, `deep_attn`

**Extended features (deep only):**

| Architecture | Inputs | Dimension |
|--------------|--------|-----------|
| `deep_cusp`, `deep_cusp_attn` | $[\rho, \sigma, f_{cusp}, \log Z]$ | 4 |
| `deep_dm`, `deep_dm_attn` | $[\rho, \sigma, f_{idem}, f_{entropy}, f_{offdiag}]$ | 5 |
| `deep_combined`, `deep_combined_attn` | $[\rho, \sigma, f_{idem}, f_{entropy}, f_{offdiag}, f_{cusp}, \log Z]$ | 7 |

**Total: 72 models** = 12 architectures x 6 training approaches
"""
    return new_markdown_cell(source)


def build_cell_02_imports():
    """Section 1 Cell 2 — imports + JAX config.

    The JAX ``x64`` and ``jax_default_device`` config calls must sit between
    ``import jax`` and ``import jax.numpy as jnp`` — flipping them later
    produces dtype and device inconsistencies in cached JIT traces (spec
    Round C10-2 regression guard).
    """
    source = """import os
import json
import """ + """pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import jax
# JAX config: pin x64 dtype and CPU device *before* importing jnp or any
# library that may trigger JAX tracing. These must not change later in the
# notebook -- flipping jax_enable_x64 after traces are cached produces
# inconsistent dtypes.
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_device", jax.devices("cpu")[0])
import jax.numpy as jnp
import equinox as eqx

from pyscf import gto, dft, scf, cc

import xcquinox.alec as alec
import xcquinox.features
"""
    return new_code_cell(source)


def build_cell_03_constants(checkpoint_base: str = DEFAULT_CHECKPOINT_BASE):
    """Section 1 Cell 3 — constants.

    ``checkpoint_base`` is emitted as a Python string literal via ``repr()``
    so the smoke test can redirect artifacts into a ``tmp_path``-backed
    directory without the f-string needing to escape special characters.
    """
    source = f"""BASIS = 'def2-svp'
CHECKPOINT_BASE = {checkpoint_base!r}
GRID_LEVEL = 1
PRETRAIN_ATOMS = (("H", 1), ("He", 0), ("O", 2), ("N", 3))
H2O_COORDS = "O 0.0000 0.0000 0.1173; H 0.0000 0.7572 -0.4692; H 0.0000 -0.7572 -0.4692"

os.makedirs(CHECKPOINT_BASE, exist_ok=True)
print(f"CHECKPOINT_BASE={{CHECKPOINT_BASE}}  BASIS={{BASIS}}  GRID_LEVEL={{GRID_LEVEL}}")
"""
    return new_code_cell(source)


def build_cell_04_arch_table():
    """Section 2 Cell 4 — print the 12 architectures from the registry.

    Uses ``print`` instead of pandas so the table renders before any plot
    cell runs (no cross-cell dependency on ``pd``).
    """
    source = """# Print all 12 registered architectures from alec.ARCHITECTURES.
# Fields printed: name, depth, nodes (hidden size), attention flag, descriptors.
_header = f"{'arch_name':<22} {'depth':>6} {'nodes':>6} {'attention':>10}  descriptors"
print(_header)
print("-" * len(_header))
for _name in alec.ARCHITECTURES.keys():
    _cfg = alec.get_architecture(_name)
    _descs = ", ".join(s.name for s in _cfg.descriptors) or "-"
    print(f"{_name:<22} {_cfg.depth:>6} {_cfg.nodes:>6} {str(_cfg.attention):>10}  {_descs}")
"""
    return new_code_cell(source)


def build_cell_05_arch_names(arch_names: tuple[str, ...] | None = None):
    """Section 2 Cell 5 — bind ``ARCH_NAMES`` and ``arch_colors``.

    ``arch_colors`` MUST be bound here (not in Section 7) because Cell 9's
    pretrain loss plot references ``arch_colors[arch_name]`` well before
    Section 7 executes. Leaving the binding in Cell 25 produces a forward
    reference that fires ``NameError`` on any fresh top-to-bottom run
    (spec Round B11-1 regression guard).
    """
    if arch_names is None:
        arch_binding = "ARCH_NAMES = list(alec.ARCHITECTURES.keys())"
    else:
        arch_binding = f"ARCH_NAMES = {list(arch_names)!r}"
    source = f"""{arch_binding}

cmap = plt.get_cmap("tab20")
arch_colors = {{name: cmap(i / max(1, len(ARCH_NAMES) - 1)) for i, name in enumerate(ARCH_NAMES)}}

print(f"Selected {{len(ARCH_NAMES)}} architectures:")
for _n in ARCH_NAMES:
    print(f"  {{_n}}")
"""
    return new_code_cell(source)


def build_cell_06_pretrain_md():
    """Section 3 Cell 6 — pretrain phase narrative (markdown)."""
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


def build_cell_07_pretrain_data_gen():
    """Section 3 Cell 7 — pretrain data generation (inline pyscf).

    Reproduces the spec §2 Cell 7 block verbatim. Critical details:
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


def build_cell_08_pretrain_loop():
    """Section 3 Cell 8 — pretrain loop over ARCH_NAMES.

    Always qualifies as alec.PretrainSpec and alec.run_pretrain — never bare.
    Includes an inline progress callback for per-arch feedback during long runs.
    """
    source = """def _cb(info):
    print(f"[{info['arch']}][{info['phase']}] step {info['step']}/{info['total']} loss={info['loss']:.4e}")

for arch_name in ARCH_NAMES:
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


def build_cell_09_pretrain_loss_plot():
    """Section 3 Cell 9 — pretrain loss curves (xnet / cnet) on log-y axes."""
    source = """fig, (ax_x, ax_c) = plt.subplots(1, 2, figsize=(12, 4))
for arch_name in ARCH_NAMES:
    losses_x = np.load(f"{CHECKPOINT_BASE}/pretrain/{arch_name}/losses_x.npy")
    losses_c = np.load(f"{CHECKPOINT_BASE}/pretrain/{arch_name}/losses_c.npy")
    ax_x.semilogy(losses_x, color=arch_colors[arch_name], label=arch_name)
    ax_c.semilogy(losses_c, color=arch_colors[arch_name], label=arch_name)

ax_x.set_title("xnet pretrain loss")
ax_x.set_xlabel("step")
ax_x.set_ylabel("MSE loss")
ax_c.set_title("cnet pretrain loss")
ax_c.set_xlabel("step")
ax_c.set_ylabel("MSE loss")
# Legend outside right on the right subplot only (avoids cluttering both)
ax_c.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize="small")

fig.tight_layout()
os.makedirs(f"{CHECKPOINT_BASE}/figures", exist_ok=True)
fig.savefig(f"{CHECKPOINT_BASE}/figures/pretrain_losses.png", dpi=150, bbox_inches="tight")
plt.show()
"""
    return new_code_cell(source)


def build_cell_10_pretrain_parity():
    """Section 3 Cell 10 — parity plots for pretrained xnet / cnet.

    Descriptor column order MUST match ``_assemble_pretrain_descriptors``
    (``xcquinox/alec/pretrain.py:69-88``): rho, sigma, dm columns (if any),
    cusp_0 / cusp_1 (if any).  dm comes BEFORE cusp — swapping is a silent
    off-by-column bug.
    """
    source = """# Load pretrain data (same .npz Cell 7 wrote)
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
    Fx_pred = jax.vmap(lambda p: xnet(p) + 1.0)(input_array)
    Fc_pred = jax.vmap(lambda p: cnet(p) + 1.0)(input_array)

    ax_x = axes[row, 0]
    ax_c = axes[row, 1]
    # Plot in F space (add 1.0 to target to match the prediction)
    ax_x.scatter(np.asarray(Fx_target) + 1.0, np.asarray(Fx_pred), s=2,
                 c=[arch_colors[arch_name]])
    _lo_x = float(min(np.min(Fx_target) + 1.0, np.min(Fx_pred)))
    _hi_x = float(max(np.max(Fx_target) + 1.0, np.max(Fx_pred)))
    ax_x.plot([_lo_x, _hi_x], [_lo_x, _hi_x], "k--", lw=0.8)
    ax_x.set_title(f"{arch_name} Fx parity")
    ax_x.set_xlabel("Fx target")
    ax_x.set_ylabel("Fx predicted")

    ax_c.scatter(np.asarray(Fc_target) + 1.0, np.asarray(Fc_pred), s=2,
                 c=[arch_colors[arch_name]])
    _lo_c = float(min(np.min(Fc_target) + 1.0, np.min(Fc_pred)))
    _hi_c = float(max(np.max(Fc_target) + 1.0, np.max(Fc_pred)))
    ax_c.plot([_lo_c, _hi_c], [_lo_c, _hi_c], "k--", lw=0.8)
    ax_c.set_title(f"{arch_name} Fc parity")
    ax_c.set_xlabel("Fc target")
    ax_c.set_ylabel("Fc predicted")

fig.tight_layout()
os.makedirs(f"{CHECKPOINT_BASE}/figures", exist_ok=True)
fig.savefig(f"{CHECKPOINT_BASE}/figures/pretrain_parity.png", dpi=150, bbox_inches="tight")
plt.show()
"""
    return new_code_cell(source)


def build_cell_11_training_md():
    """Section 4 Cell 11 — training data narrative (markdown).

    Explains the training set composition, reference data split, .npz sidecar
    convention, and the step3b HF-DM naming quirk.
    """
    source = """## Section 4: Training Data

### Training set

Three species are used: **H** (spin=1), **O** (spin=2), and **H2O** (spin=0),
all computed at the **def2-svp** basis (``BASIS`` from Cell 3).

### Reference data split

- **H / O (atoms):** Reference energies come from literature total energies
  (H: exact −0.5 Ha; O: ~−75.0673 Ha). Degenerate HOMO eigenvalues in
  open-shell atoms make one-shot density targets numerically unstable, so
  **no** ``dm_target`` or ``rho_ccsd_grid`` is stored for atoms.
- **H2O:** Uses the equilibrium geometry ``H2O_COORDS`` from Cell 3 (NOT a
  distorted 90-degree box). The HF density matrix is stored as the density
  target, and the HF grid density is stored as the grid-density target.

### .npz sidecar convention

Each species gets two files:
1. ``{name}.npz`` — holds **only** the three whitelisted keys that
   ``xcquinox.alec.data`` accepts: ``dm_target``, ``rho_ccsd_grid``,
   ``E_ref_literature``. Any extra key causes a ``ValueError`` at load time.
2. ``{name}_metadata.json`` — holds HF, CCSD, literature, and PBE total
   energies that cannot be stored in the whitelisted ``.npz``.

### Step 3b naming quirk

Despite the key name ``dm_target``, step3b uses the **HF** density matrix as
the target (not the CCSD 1-RDM). The same convention is reproduced here so
the trained models are numerically identical to step3b checkpoints.
"""
    return new_markdown_cell(source)


def build_cell_12_reference_dicts():
    """Section 4 Cell 12 — atom_energies and targets dicts + ext_data_dir setup."""
    source = """# Atom energies: literature total energies in Hartree (negative, as they should be).
# H is exact: -0.5 Ha. O is literature total ~ -75.0673 Ha.
atom_energies = {"H": -0.5, "O": -75.0673}

# targets dict: validator requires an entry for every molecule in TrainingSpec.molecules
# (config.py:523-525). Atom entries are never dereferenced at training time but must be
# finite floats — we set them to match atom_energies for future-refactor consistency.
# The H2O entry is the POSITIVE-for-bound atomization energy in Hartree:
#   AE = E_atoms_sum - E_mol > 0 for a bound molecule
# Literature: AE(H2O) ~ 974.94 kJ/mol = 974.94 / 2625.5 Ha.
targets = {"H": -0.5, "O": -75.0673, "H2O": 974.94 / 2625.5}

ext_data_dir = f"{CHECKPOINT_BASE}/external_data"
os.makedirs(ext_data_dir, exist_ok=True)
print(f"ext_data_dir={ext_data_dir}  targets={list(targets.keys())}  atom_energies={list(atom_energies.keys())}")
"""
    return new_code_cell(source)


def build_cell_13_hf_ccsd_gen():
    """Section 4 Cell 13 — HF/CCSD reference computation and .npz generation.

    Writes {name}.npz (whitelisted keys only) and {name}_metadata.json
    (HF/CCSD/PBE totals) for H, O, and H2O.
    """
    source = """# HF/CCSD reference computation and external_data .npz generation.
# H2O uses H2O_COORDS from Cell 3 (equilibrium geometry, NOT a distorted 90-degree box).
_mols = [
    ("H", "H 0 0 0", 1),
    ("O", "O 0 0 0", 2),
    ("H2O", H2O_COORDS, 0),
]

for name, atom, spin in _mols:
    # Identical gto.M kwargs to what precompute_fixed_density_data uses internally.
    mol = gto.M(atom=atom, basis=BASIS, charge=0, spin=spin, verbose=0)

    # PBE SCF with grid pinned to GRID_LEVEL (must match Cell 14/15 precompute grid).
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
        np.savez(
            os.path.join(ext_data_dir, f"{name}.npz"),
            E_ref_literature=atom_energies[name],
        )
    else:
        # H2O branch: write HF DM as density target (NOT CCSD DM — step3b uses HF).
        dm_hf = mf_hf.make_rdm1()
        dm_hf_total = dm_hf[0] + dm_hf[1] if dm_hf.ndim == 3 else dm_hf

        # Grid density from HF DM via einsum on the AO grid.
        coords = mf.grids.coords
        ao_grid = mf._numint.eval_ao(mol, coords, deriv=0)
        rho_hf = np.einsum("ij,gi,gj->g", dm_hf_total, ao_grid, ao_grid)

        # The three keys below are the ONLY keys _ALLOWED_EXTERNAL_KEYS accepts
        # (data.py:17-21). E_ref_literature is the HF total, not the CCSD total,
        # because TotalEnergyMetric.E_error_hartree gauges against this scalar and
        # the density-matching losses (B/C/D2/D3) optimize toward the HF density.
        np.savez(
            os.path.join(ext_data_dir, f"{name}.npz"),
            dm_target=dm_hf,
            rho_ccsd_grid=rho_hf,
            E_ref_literature=float(mf_hf.e_tot),
        )

    # Sidecar JSON for every species — library .npz cannot carry extra keys,
    # so HF/CCSD/literature/PBE totals live here. Cell 25 reads E_ccsd_total
    # from this file for the CCSD atomization-energy reference line.
    with open(os.path.join(ext_data_dir, f"{name}_metadata.json"), "w") as _f:
        json.dump(
            {
                "E_hf_total": E_hf_total,
                "E_ccsd_total": E_ccsd_total,
                "E_lit_Ha": atom_energies.get(name, None),
                "E_pbe_total": E_pbe_total,
            },
            _f,
            indent=2,
        )

print(f"Reference data written to {ext_data_dir}")
"""
    return new_code_cell(source)


def main(
    output_path: str,
    *,
    arch_names: tuple[str, ...] | None = None,
    loss_names: tuple[str, ...] | None = None,
    checkpoint_base: str | None = None,
):
    """Assemble the step 4 notebook, validate it, write it to ``output_path``.

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
    checkpoint_base
        Optional override for ``DEFAULT_CHECKPOINT_BASE``. Used by the smoke
        test to redirect artifacts into a ``tmp_path``-backed directory.

    Returns
    -------
    nbformat.notebooknode.NotebookNode
        The assembled notebook, already written to disk.
    """
    # Resolve defaults only for the values that need to be *injected into the
    # notebook source*. ``arch_names=None`` and ``loss_names=None`` deliberately
    # stay as ``None`` so downstream builders emit the dynamic forms
    # (``ARCH_NAMES = list(alec.ARCHITECTURES.keys())`` / ``LOSS_NAMES = (...)``
    # literal that traces the library registries at runtime).
    if checkpoint_base is None:
        checkpoint_base = DEFAULT_CHECKPOINT_BASE

    nb = new_notebook()
    nb.cells = [
        build_cell_01_title(),
        build_cell_02_imports(),
        build_cell_03_constants(checkpoint_base),
        build_cell_04_arch_table(),
        build_cell_05_arch_names(arch_names),
        build_cell_06_pretrain_md(),
        build_cell_07_pretrain_data_gen(),
        build_cell_08_pretrain_loop(),
        build_cell_09_pretrain_loss_plot(),
        build_cell_10_pretrain_parity(),
        build_cell_11_training_md(),
        build_cell_12_reference_dicts(),
        build_cell_13_hf_ccsd_gen(),
    ]

    nbformat.validate(nb)

    parent = os.path.dirname(output_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    nbformat.write(nb, output_path)
    return nb


if __name__ == "__main__":
    main("notebooks/gga_training_example-step4.ipynb")
