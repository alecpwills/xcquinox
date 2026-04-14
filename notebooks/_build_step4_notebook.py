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

# tqdm.auto picks tqdm.notebook.tqdm (ipywidgets) under JupyterLab and
# tqdm.std.tqdm in a plain script/terminal, so the same symbol gives a
# sensible progress bar in either context.
from tqdm.auto import tqdm
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

# Flip to True to run Cell 8's per-arch pretraining in parallel subprocesses.
# Each worker gets XLA_FLAGS=--xla_cpu_multi_thread_eigen=false and
# OMP_NUM_THREADS=1 so N parallel workers on one machine do not oversubscribe
# XLA's internal thread pool. max_workers = min(len(ARCH_NAMES), cpu_count()//2).
# Trade-off: per-step progress callbacks are lost (subprocesses do not stream
# back), so output collapses to one "[arch] pretrain complete" line per arch.
PRETRAIN_PARALLEL = False

# Flip to True to skip pretraining for any arch that already has both
# ``xnet.eqx`` and ``cnet.eqx`` at ``CHECKPOINT_BASE/pretrain/<arch>/``. Cell 8
# re-loads those checkpoints downstream (Cell 10 parity / Section 5 training),
# so skipping is safe as long as the existing files match your current
# architecture registry. Default is False (always re-pretrain) to match the
# step3b-era behaviour.
PRETRAIN_SKIP_IF_EXISTS = False

# Flip to True to skip the main training loop for any (arch, loss) run that
# already has a ``model.eqx`` at
# ``CHECKPOINT_BASE/train/<arch>/<loss_name>/``. Cell 18 re-uses those
# checkpoints via the Section 6+ evaluation cells, so skipping is safe as
# long as the existing files match your current ARCH_NAMES * LOSS_NAMES
# combo. Default is False (always re-train).
TRAIN_SKIP_IF_EXISTS = False

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

    Emits a single unified cell that contains BOTH a serial in-process loop
    and a parallel subprocess+ThreadPoolExecutor dispatch, branching on the
    runtime constant ``PRETRAIN_PARALLEL`` (bound in Cell 3). Users flip the
    Cell 3 constant and re-run Cell 8 to opt into parallel pretraining without
    regenerating the notebook.

    Always qualifies as ``alec.PretrainSpec`` and ``alec.run_pretrain`` —
    never bare.

    Parallel branch design
    ----------------------
    Each arch gets its own Python interpreter via ``subprocess.run``, so
    ``XLA_FLAGS=--xla_cpu_multi_thread_eigen=false`` and ``OMP_NUM_THREADS=1``
    are applied BEFORE JAX is imported in the child — otherwise N parallel
    workers on one machine oversubscribe XLA's internal thread pool.
    ``max_workers = min(len(ARCH_NAMES), cpu_count()//2)``. Failures re-raise
    with stdout+stderr context.

    Trade-off: per-step progress callbacks are lost (subprocesses do not
    stream back), so parallel output collapses to one "[arch] pretrain
    complete" line per arch instead of per-step loss.
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

if PRETRAIN_PARALLEL:
    import os
    import subprocess
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def _pretrain_one_subprocess(arch_name):
        # Each worker gets its own Python interpreter, so XLA_FLAGS and
        # OMP_NUM_THREADS are applied BEFORE JAX is imported in the child.
        # This is what prevents N parallel workers from each trying to use
        # all CPUs for XLA's matmul thread pool.
        child_env = dict(os.environ)
        child_env["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"
        child_env["OMP_NUM_THREADS"] = "1"
        child_code = f'''
import xcquinox.alec as alec
spec = alec.PretrainSpec(
    arch=alec.get_architecture({arch_name!r}),
    data_dir={CHECKPOINT_BASE + '/pretrain_data'!r},
    checkpoint_dir={CHECKPOINT_BASE + '/pretrain/' + arch_name!r},
    n_steps=1000,
    lr_start=1e-2,
    lr_end=1e-5,
    lr_decay_start=0.2,
    grad_clip=1.0,
)
alec.run_pretrain(spec)
'''
        try:
            subprocess.run(
                ["python", "-c", child_code],
                env=child_env,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"[{arch_name}] pretrain subprocess failed:\\n"
                f"stdout:\\n{exc.stdout}\\nstderr:\\n{exc.stderr}"
            ) from exc
        return arch_name

    # Filter out arches whose checkpoints already exist when the skip flag is on.
    _archs_to_run = [
        _a for _a in ARCH_NAMES
        if not (PRETRAIN_SKIP_IF_EXISTS and _pretrain_checkpoints_exist(_a))
    ]
    _skipped = [_a for _a in ARCH_NAMES if _a not in _archs_to_run]
    if _skipped:
        print(f"Skipping {len(_skipped)} cached arch(es): {_skipped}")

    _n_cpus = os.cpu_count() or 2
    _max_workers = max(1, min(len(_archs_to_run) or 1, _n_cpus // 2))
    print(f"Pretraining {len(_archs_to_run)} archs in parallel: "
          f"{_max_workers} workers on {_n_cpus} CPUs")

    # Per-step callbacks do not stream back through subprocesses, so the
    # best-available progress signal in parallel mode is an arch-completion
    # counter. The postfix shows the most recently completed arch name.
    _arch_bar = tqdm(
        total=len(ARCH_NAMES),
        desc="pretrain (parallel)",
        leave=True,
        dynamic_ncols=True,
    )
    # Count skipped arches against the bar so the total stays at len(ARCH_NAMES).
    if _skipped:
        _arch_bar.update(len(_skipped))
    try:
        if _archs_to_run:
            with ThreadPoolExecutor(max_workers=_max_workers) as _ex:
                _futures = {_ex.submit(_pretrain_one_subprocess, _a): _a for _a in _archs_to_run}
                for _future in as_completed(_futures):
                    _arch_done = _future.result()
                    _arch_bar.update(1)
                    _arch_bar.set_postfix(arch=_arch_done)
    finally:
        _arch_bar.close()
else:
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


def build_cell_09_pretrain_loss_plot():
    """Section 3 Cell 9 — pretrain loss curves (xnet / cnet) on log-y axes.

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
    """Section 4 Cell 12 — atom_energies_literature + targets dicts + ext_data_dir setup.

    The literature-value dict (H: -0.5 Ha, O: -75.0673 Ha) is stored under
    `atom_energies_literature` and is consumed ONLY by Cell 13's atom-branch
    `E_ref_literature` sidecar write (which TotalEnergyMetric compares against).
    The NAME `atom_energies` that the training loss and AtomizationEnergyMetric
    consume is bound later — at the end of Cell 13 — to a PBE-consistent dict.
    """
    source = """# Literature atomic total energies in Hartree (negative, as they should be).
# Used ONLY by Cell 13 to write each atom's E_ref_literature sidecar value
# (TotalEnergyMetric compares NN atomic totals against this scalar).
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
# AtomizationEnergyMetric) is defined at the end of Cell 13 from the PBE
# atomic totals computed there. Do not reference `atom_energies` before Cell 13.
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


def build_cell_14_mol_specs():
    """Section 4 Cell 14 — construct three alec.MoleculeSpec objects.

    The list is an explicit three-element literal (NOT a comprehension) so that
    Cell 17 can reference the individual entries by index and Cell 15 can iterate
    over them by name. All kwargs match Cell 13's gto.M kwargs exactly so that
    precompute_fixed_density_data rebuilds the same pyscf grid and
    _load_external_data accepts the .npz arrays.
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


def build_cell_15_precompute_sanity():
    """Section 4 Cell 15 — call precompute_fixed_density_data and assert invariants.

    Builds ``mol_data_list`` with the union of descriptor ``required_mol_keys``
    across every arch in ``ARCH_NAMES``, so the Section 7 visualization cells
    (26 dm_heatmaps, 27 density_histograms) can call descriptor-demanding
    APIs like ``oneshot_dm_prediction_fast`` / ``oneshot_grid_density`` on
    ``mol_data_list[2]`` regardless of which arch Cell 25's ``best_idx``
    selects as best-per-loss. A bare precompute call would leave
    ``cusp_features`` / ``dm_features`` as ``None`` and break descriptor archs
    (``deep_cusp``, ``deep_dm``, ``deep_combined``, and their ``_attn``
    variants) with ``TypeError: concatenate requires ndarray or scalar
    arguments, got <class 'NoneType'>`` inside
    ``assemble_descriptor_features``.

    Prints per-molecule shapes and energies, then asserts the atom-vs-compound
    invariants: atoms have no dm_target / rho_ccsd_grid; H2O has all three.
    The negative atom assertions guard against a Cell 13 regression that would
    accidentally write dm_target / rho_ccsd_grid into the atom .npz.
    """
    source = """# Union of descriptor required_mol_keys across every arch we will train.
# Ensures mol_data_list carries cusp_features / dm_features for any arch the
# downstream Section 7 visualization cells (26, 27) may later pass to
# oneshot_dm_prediction_fast / oneshot_grid_density. Derive from ARCH_NAMES
# (Cell 5) rather than hardcoding so a custom arch list stays correct.
_mol_data_required_keys = set()
for _arch_name in ARCH_NAMES:
    for _desc in alec.get_architecture(_arch_name).materialize_descriptors():
        _mol_data_required_keys.update(_desc.required_mol_keys)
_mol_data_required_keys = tuple(sorted(_mol_data_required_keys))
print(f"mol_data_list required_keys union: {_mol_data_required_keys}")

mol_data_list = [
    alec.precompute_fixed_density_data(m, required_keys=_mol_data_required_keys)
    for m in mol_specs
]

for mol_spec, mol_data in zip(mol_specs, mol_data_list):
    print(f"\\n=== {mol_spec.name} ===")
    print(f"  n_grid:            {mol_data['rho_grid'].shape[0]}")
    print(f"  n_ao:              {mol_data['ao_grid'].shape[1]}")
    print(f"  E_pbe (Ha):        {mol_data['E_pbe']:.6f}")
    print(f"  E_non_xc (Ha):     {mol_data['E_non_xc']:.6f}")
    print(f"  E_ref_literature:  {mol_data['E_ref_literature']}")
    print(f"  dm_target:         {None if mol_data['dm_target'] is None else mol_data['dm_target'].shape}")
    print(f"  rho_ccsd_grid:     {None if mol_data['rho_ccsd_grid'] is None else mol_data['rho_ccsd_grid'].shape}")

# Atom-vs-compound invariants: atoms skip density targets, H2O has all three.
# Negative atom assertions guard against a Cell 13 regression that would accidentally
# write dm_target / rho_ccsd_grid into the atom .npz.
assert mol_data_list[0]["E_ref_literature"] is not None  # H
assert mol_data_list[0]["dm_target"] is None             # H — atoms skip density
assert mol_data_list[0]["rho_ccsd_grid"] is None         # H — atoms skip density
assert mol_data_list[1]["E_ref_literature"] is not None  # O
assert mol_data_list[1]["dm_target"] is None             # O
assert mol_data_list[1]["rho_ccsd_grid"] is None         # O
assert mol_data_list[2]["E_ref_literature"] is not None  # H2O
assert mol_data_list[2]["dm_target"] is not None         # H2O — HF DM target
assert mol_data_list[2]["rho_ccsd_grid"] is not None     # H2O — HF grid density
print("\\nAll atom-vs-compound invariants satisfied.")
"""
    return new_code_cell(source)



def build_cell_16_training_md():
    """Section 5 Cell 16 -- training section overview narrative (markdown)."""
    source = """## Section 5: Main Training Loop

72 models are trained: **12 architectures x 6 loss approaches**.

### Loss Approaches

| Label | Loss family | Energy term | Density term | Weights |
|-------|------------|-------------|-------------|---------|
| A | `A_atomization` | AE (fixed-density PBE) | -- | defaults |
| B | `B_atomization_plus_dm` | AE | DM -> HF target | dm_weight=0.1 |
| C | `C_atomization_plus_grid` | AE | grid rho -> HF target | density_weight=0.1 |
| D1 | `D1_delta_ae` | delta-AE | -- | defaults |
| D2 | `D2_delta_ae_plus_dm` | delta-AE | DM -> HF target | dm_weight=0.1 |
| D3 | `D3_delta_ae_plus_grid` | delta-AE | grid rho -> HF target | density_weight=0.1 |

### Shared Hyperparameter Schedule

`n_steps=250`, `lr_start=1e-2`, `lr_end=1e-5`, `lr_decay_start=0.2`, `grad_clip=1.0`

### Artifact Layout

Each run writes four files to its own subdirectory:

```
{CHECKPOINT_BASE}/train/{arch}/{loss}/model.eqx
{CHECKPOINT_BASE}/train/{arch}/{loss}/losses.npy
{CHECKPOINT_BASE}/train/{arch}/{loss}/aux_log.pkl
{CHECKPOINT_BASE}/train/{arch}/{loss}/train_metadata.json
```
"""
    return new_markdown_cell(source)


def build_cell_17_training_specs(loss_names=None):
    """Section 5 Cell 17 -- build 72 alec.TrainingSpec objects.

    loss_names is an optional override for the loss-name tuple. When
    None, the cell emits the dynamic form
    LOSS_NAMES = ("A_atomization", ...). When a tuple is provided, a
    literal list form is emitted (mirroring build_cell_05_arch_names).

    Each spec carries a per-(arch, loss) checkpoint_dir so that all 72
    runs route their artifacts without overwriting each other.
    """
    if loss_names is None:
        loss_names_binding = (
            'LOSS_NAMES = (\n'
            '    "A_atomization",\n'
            '    "B_atomization_plus_dm",\n'
            '    "C_atomization_plus_grid",\n'
            '    "D1_delta_ae",\n'
            '    "D2_delta_ae_plus_dm",\n'
            '    "D3_delta_ae_plus_grid",\n'
            ')'
        )
    else:
        loss_names_binding = f"LOSS_NAMES = {tuple(loss_names)!r}"
    source = f"""{loss_names_binding}

LOSS_KWARGS = {{
    "A_atomization": {{}},
    "B_atomization_plus_dm": {{"dm_weight": 0.1}},
    "C_atomization_plus_grid": {{"density_weight": 0.1}},
    "D1_delta_ae": {{}},
    "D2_delta_ae_plus_dm": {{"dm_weight": 0.1}},
    "D3_delta_ae_plus_grid": {{"density_weight": 0.1}},
}}

specs = []
for arch_name in ARCH_NAMES:
    for loss_name in LOSS_NAMES:
        specs.append(alec.TrainingSpec.from_dicts(
            arch=alec.get_architecture(arch_name),
            loss_name=loss_name,
            molecules=tuple(mol_specs),
            targets=targets,
            atom_energies=atom_energies,
            loss_kwargs=LOSS_KWARGS[loss_name],
            pretrain_checkpoint=f"{{CHECKPOINT_BASE}}/pretrain/{{arch_name}}",
            checkpoint_dir=f"{{CHECKPOINT_BASE}}/train/{{arch_name}}/{{loss_name}}",
            n_steps=250,
            lr_start=1e-2,
            lr_end=1e-5,
            lr_decay_start=0.2,
            grad_clip=1.0,
        ))
print(f"Built {{len(specs)}} training specs")
"""
    return new_code_cell(source)


def build_cell_18_training_loop():
    """Section 5 Cell 18 -- serial training loop over all 72 specs.

    Implements the serial path only. Each spec already carries its per-(arch,
    loss) checkpoint_dir from Cell 17, so artifacts route automatically.
    For a parallel variant using alec.build_training_jobs +
    alec.run_workers, see spec line 88 -- not emitted here.

    Progress reporting uses a two-tier ``tqdm`` display:
      - Outer bar with ``total=len(specs)`` counts completed (arch, loss) runs.
      - Inner per-spec bar, recreated on the first callback of each spec and
        closed when ``step == total``, shows per-step progress with a
        scientific-notation ``loss=...`` postfix.
    """
    source = """# Serial training loop.  Each spec already carries its per-(arch, loss) checkpoint_dir
# from Cell 17, so artifacts route automatically.  For a parallel variant using
# alec.build_training_jobs + alec.run_workers, see spec line 88 -- not emitted here.
#
# Two-tier tqdm progress display:
#   - _spec_bar: outer bar counting completed specs out of len(specs)
#   - _step_bars: dict of per-(arch, phase) inner bars tracking step progress
#     within the current spec. An inner bar is created on the first callback
#     for a given key and closed (and removed) when step == total, so the next
#     spec with the same arch starts a fresh bar.
_step_bars = {}
_current_loss = {"name": None}

def _train_cb(info):
    key = (info['arch'], info['phase'])
    if key not in _step_bars:
        _label = (f"{info['arch']:<20} {_current_loss['name']:<25}"
                  if _current_loss['name'] is not None
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
        _current_loss['name'] = spec.loss_name
        if TRAIN_SKIP_IF_EXISTS and _training_model_exists(spec):
            print(f"[{spec.arch.name}][{spec.loss_name}] cached model.eqx found — skipping training")
            _spec_bar.update(1)
            _spec_bar.set_postfix(arch=spec.arch.name, loss=spec.loss_name, skipped=True)
            continue
        alec.run_training(spec, progress_callback=_train_cb)
        _spec_bar.update(1)
        _spec_bar.set_postfix(arch=spec.arch.name, loss=spec.loss_name)
finally:
    _spec_bar.close()
    for _b in list(_step_bars.values()):
        _b.close()
    _step_bars.clear()
"""
    return new_code_cell(source)


def build_cell_19_training_loss_plot():
    """Section 5 Cell 19 -- 2x3 training loss curves grid (log y, one per loss family).

    Loads losses.npy from each per-(arch, loss) checkpoint directory and
    plots 12 arch curves per subplot with a shared legend on the top-right.
    """
    source = """fig, axes = plt.subplots(2, 3, figsize=(15, 9), squeeze=False)
_axes_flat = axes.flatten()
for idx, loss_name in enumerate(LOSS_NAMES):
    ax = _axes_flat[idx]
    for arch_name in ARCH_NAMES:
        losses = np.load(f"{CHECKPOINT_BASE}/train/{arch_name}/{loss_name}/losses.npy")
        ax.semilogy(losses, color=arch_colors[arch_name], label=arch_name)
    ax.set_title(f"loss family: {loss_name}")
    ax.set_xlabel("training step")
    ax.set_ylabel("total loss (log scale)")
    ax.grid(True, which="both", ls=":", alpha=0.4)

# Shared legend outside right on the rightmost top subplot only
axes[0, 2].legend(
    loc="center left",
    bbox_to_anchor=(1.02, 0.5),
    fontsize="small",
    title="architecture",
)

fig.suptitle(
    "Main training loss curves -- 12 architectures x 6 loss families "
    "(one subplot per loss, one trace per architecture)",
    fontsize=13,
)
fig.tight_layout(rect=(0, 0, 1, 0.95))
os.makedirs(f"{CHECKPOINT_BASE}/figures", exist_ok=True)
fig.savefig(f"{CHECKPOINT_BASE}/figures/training_losses.png", dpi=150, bbox_inches="tight")
plt.show()
"""
    return new_code_cell(source)


def build_cell_20_aux_inspection():
    """Section 5 Cell 20 -- aux_log inspection for a sample arch.

    Loads aux_log.pkl for arch_name = "shallow" across all loss
    families and plots per-family component losses in a 2x3 grid.
    """
    source = '''arch_name = "shallow"
_aux_keys_per_family = {
    "A_atomization": ("loss_energy", "atomic_reg"),
    "B_atomization_plus_dm": ("loss_energy", "loss_dm"),
    "C_atomization_plus_grid": ("loss_energy", "loss_grid"),
    "D1_delta_ae": ("loss_delta", "atomic_reg"),
    "D2_delta_ae_plus_dm": ("loss_delta", "loss_dm"),
    "D3_delta_ae_plus_grid": ("loss_delta", "loss_grid"),
}

fig, axes = plt.subplots(2, 3, figsize=(15, 9), squeeze=False)
_ax_by_loss = dict(zip(LOSS_NAMES, axes.flatten()))
for loss_name in LOSS_NAMES:
    ax = _ax_by_loss[loss_name]
    ckpt_dir = f"{CHECKPOINT_BASE}/train/{arch_name}/{loss_name}"
    with open(f"{ckpt_dir}/aux_log.pkl", "rb") as _f:
        aux_log = pickle.load(_f)

    _steps = [entry["step"] for entry in aux_log]
    for key in _aux_keys_per_family[loss_name]:
        _vals = [entry["aux"][key] for entry in aux_log]
        ax.semilogy(_steps, _vals, label=key)
    ax.set_title(f"loss family: {loss_name}")
    ax.set_xlabel("training step")
    ax.set_ylabel("aux component value (log scale)")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(fontsize="small", title="aux key")

fig.suptitle(
    f"Aux loss components for arch = {arch_name!r} -- per-family breakdown "
    f"(single architecture, 6 loss families)",
    fontsize=13,
)
fig.tight_layout(rect=(0, 0, 1, 0.95))
os.makedirs(f"{CHECKPOINT_BASE}/figures", exist_ok=True)
fig.savefig(f"{CHECKPOINT_BASE}/figures/aux_components_{arch_name}.png", dpi=150, bbox_inches="tight")
plt.show()
'''
    return new_code_cell(source)


def build_cell_21_eval_md():
    """Section 6 Cell 21 -- evaluation metrics narrative."""
    source = """## Section 6: Evaluation

Each trained model is scored on the same `mol_specs` used for training via
`alec.TestSpec.from_dicts(...)` + `alec.run_test(spec)`. Four metrics are
computed for every molecule:

- **`total_energy`** -- NN total energy vs. PBE reference, reported as
  `E_error_kcalmol` (kcal/mol).
- **`atomization_energy`** -- `AE_nn = sum(E_atoms) - E_mol` (positive for
  a bound molecule), compared against the literature AE of 233.016 kcal/mol
  for H2O. Reported as `AE_error_kcalmol`.
- **`density_rmse`** -- RMSE of the NN grid density against the CCSD
  reference density. Atoms are auto-skipped (single-atom systems return
  `None`), so H2O is the sole contributor in this experiment.
- **`constraint_violations`** -- flattens `x_constraints` / `c_constraints`
  from the arch config. Every default architecture ships with empty
  constraint tuples, so the `aggregate.json` files contain **no**
  `constraint_violations` top-level key at all -- this is expected for
  step3b parity, not a bug.

Cell 22 below runs `alec.run_test` over the same `(arch, loss)` product as
Cell 17, so every trained model is paired with a `TestSpec` and writes its
own `aggregate.json` under `{CHECKPOINT_BASE}/test/{arch}/{loss}/`.
"""
    return new_markdown_cell(source)


def build_cell_22_test_loop():
    """Section 6 Cell 22 -- build TestSpec + run_test per trained model."""
    source = """for arch_name in ARCH_NAMES:
    for loss_name in LOSS_NAMES:
        ckpt_dir = f"{CHECKPOINT_BASE}/train/{arch_name}/{loss_name}"
        test_spec = alec.TestSpec.from_dicts(
            arch=alec.get_architecture(arch_name),
            model_checkpoint=f"{ckpt_dir}/model.eqx",
            molecules=tuple(mol_specs),
            metrics=("total_energy", "atomization_energy", "density_rmse", "constraint_violations"),
            metric_kwargs={"atomization_energy": {"reference_ae_kcalmol": {"H2O": 233.016}}},
            atom_energies=atom_energies,
            output_dir=f"{CHECKPOINT_BASE}/test/{arch_name}/{loss_name}",
        )
        alec.run_test(test_spec)
"""
    return new_code_cell(source)


def build_cell_23_dataframe():
    """Section 6 Cell 23 -- aggregate results into pandas DataFrame."""
    source = """rows = []
for arch_name in ARCH_NAMES:
    for loss_name in LOSS_NAMES:
        output_dir = f"{CHECKPOINT_BASE}/test/{arch_name}/{loss_name}"
        try:
            with open(f"{output_dir}/aggregate.json") as _f:
                agg = json.load(_f)
        except FileNotFoundError:
            agg = {}
        rows.append({
            "arch": arch_name,
            "loss": loss_name,
            "AE_error_kcalmol_mean": agg.get("AE_error_kcalmol", {}).get("mean", np.nan),
            "AE_error_kcalmol_RMSE": agg.get("AE_error_kcalmol", {}).get("RMSE", np.nan),
            "E_error_kcalmol_mean": agg.get("E_error_kcalmol", {}).get("mean", np.nan),
            "density_rmse_mean": agg.get("density_rmse", {}).get("mean", np.nan),
        })
df = pd.DataFrame(rows).set_index(["arch", "loss"])
print(f"Built results DataFrame: {df.shape[0]} rows x {df.shape[1]} cols")
"""
    return new_code_cell(source)


def build_cell_24_results_table():
    """Section 6 Cell 24 -- print results table + best arch per loss."""
    source = """piv = df["AE_error_kcalmol_mean"].unstack(level="loss")
print("AE error (kcal/mol), arch x loss:")
print(piv.round(3))
print()
print("Best architecture per loss family (minimum |AE error|):")
print(piv.abs().idxmin(axis=0))
"""
    return new_code_cell(source)


def build_cell_25_ae_bars():
    """Section 7 Cell 25 -- AE error grouped bar chart + shared Section 7 bindings.

    Also binds ``best_idx`` and ``pairs`` which are consumed by every
    downstream Section 7 cell (26-31). These MUST be bound here, not in
    the Section 7 preamble prose.
    """
    source = """# Shared Section 7 bindings (consumed by Cells 26-31).
best_idx = df["AE_error_kcalmol_mean"].unstack("loss").idxmin(axis=0)
pairs = [(n, f"{n}_attn") for n in ARCH_NAMES if not n.endswith("_attn") and f"{n}_attn" in ARCH_NAMES]

# Parallel bar-height / error-bar DataFrames.
bar_heights = df["AE_error_kcalmol_mean"].abs().unstack("loss")
bar_yerr = df["AE_error_kcalmol_RMSE"].unstack("loss")

# PBE reference line: compute from in-scope mol_data_list, fall back on
# sidecar metadata JSONs if the kernel was restarted between Cell 15
# and Cell 25 (requires Cells 2, 3, 12 to have been re-executed).
try:
    PBE_AE_Ha = 2 * mol_data_list[0]["E_pbe"] + mol_data_list[1]["E_pbe"] - mol_data_list[2]["E_pbe"]
except NameError:
    _E_pbe = {}
    for _name in ("H", "O", "H2O"):
        with open(f"{ext_data_dir}/{_name}_metadata.json") as _f:
            _E_pbe[_name] = json.load(_f)["E_pbe_total"]
    PBE_AE_Ha = 2 * _E_pbe["H"] + _E_pbe["O"] - _E_pbe["H2O"]
PBE_AE_kcalmol = PBE_AE_Ha * 627.509
PBE_AE_err_kcalmol = abs(PBE_AE_kcalmol - 233.016)

# CCSD reference line: load from Cell 13 sidecar JSONs.
ccsd_totals = {}
for _name in ("H", "O", "H2O"):
    with open(f"{ext_data_dir}/{_name}_metadata.json") as _f:
        ccsd_totals[_name] = json.load(_f)["E_ccsd_total"]
CCSD_AE_Ha = 2 * ccsd_totals["H"] + ccsd_totals["O"] - ccsd_totals["H2O"]
CCSD_AE_kcalmol = CCSD_AE_Ha * 627.509
CCSD_AE_err_kcalmol = abs(CCSD_AE_kcalmol - 233.016)

# Grouped bar chart: 6 loss groups, 12 arch bars per group.
fig, ax = plt.subplots(figsize=(14, 8))
n_archs = len(ARCH_NAMES)
x_positions = np.arange(len(LOSS_NAMES))
bar_width = 0.8 / n_archs
for i, arch_name in enumerate(ARCH_NAMES):
    heights = bar_heights.loc[arch_name, list(LOSS_NAMES)].values
    yerrs = bar_yerr.loc[arch_name, list(LOSS_NAMES)].values
    offset = (i - (n_archs - 1) / 2) * bar_width
    ax.bar(x_positions + offset, heights, width=bar_width, yerr=yerrs,
           color=arch_colors[arch_name], label=arch_name)
ax.set_xticks(x_positions)
ax.set_xticklabels(list(LOSS_NAMES), rotation=30, ha="right")
ax.set_xlabel("loss family (training objective)")
ax.set_ylabel("|AE error| vs literature (kcal/mol, log scale)")
ax.set_yscale("log")
ax.set_title(
    "H2O atomization-energy error by architecture x loss family\\n"
    "(literature AE = 233.016 kcal/mol; error bars = per-molecule RMSE)",
    fontsize=12,
)
ax.grid(True, which="both", axis="y", ls=":", alpha=0.4)

ax.axhline(PBE_AE_err_kcalmol, linestyle="dotted", color="r", alpha=0.7,
           label=f"PBE Error ({PBE_AE_err_kcalmol:.2f} kcal/mol)")
ax.axhline(CCSD_AE_err_kcalmol, linestyle="-.", color="r", alpha=0.7,
           label=f"CCSD Error ({CCSD_AE_err_kcalmol:.2f} kcal/mol)")
ax.axhline(1.0, linestyle="--", color="r", alpha=0.7,
           label="Chemical accuracy (1 kcal/mol)")

ax.legend(
    bbox_to_anchor=(1.02, 1),
    loc="upper left",
    fontsize=8,
    title="architecture / reference",
)
fig.tight_layout()
os.makedirs(f"{CHECKPOINT_BASE}/figures", exist_ok=True)
fig.savefig(f"{CHECKPOINT_BASE}/figures/ae_error_by_loss.png", dpi=150, bbox_inches="tight")
plt.show()
"""
    return new_code_cell(source)


def build_cell_26_dm_heatmaps():
    """Section 7 Cell 26 -- 2x2 DM heatmaps (PBE-HF, best-B, best-D1, best-D2)."""
    source = """# Per-loop template rebuild: different archs have different PyTree layouts,
# so the template must match the specific checkpoint being deserialised.
model_bindings = {}
for loss_name in ("B_atomization_plus_dm", "D1_delta_ae", "D2_delta_ae_plus_dm"):
    if loss_name not in best_idx.index:
        print(f"[Cell 26] skipping {loss_name}: not in narrow config")
        continue
    best_arch = best_idx[loss_name]
    arch_config = alec.get_architecture(best_arch)
    model_template = alec.AlecGGAModel.from_arch(arch_config)
    ckpt_path = f"{CHECKPOINT_BASE}/train/{best_arch}/{loss_name}/model.eqx"
    model_bindings[loss_name] = eqx.tree_deserialise_leaves(ckpt_path, model_template)
model_B = model_bindings["B_atomization_plus_dm"] if "B_atomization_plus_dm" in model_bindings else None
model_D1 = model_bindings["D1_delta_ae"] if "D1_delta_ae" in model_bindings else None
model_D2 = model_bindings["D2_delta_ae_plus_dm"] if "D2_delta_ae_plus_dm" in model_bindings else None

if not model_bindings:
    print("[Cell 26] no DM-loss models in this configuration; skipping heatmaps")
else:
    # DMs for H2O (index 2 in mol_data_list).
    dm_pbe = mol_data_list[2]["dm_pbe"]
    dm_hf = mol_data_list[2]["dm_target"]
    dm_nn_B = alec.oneshot_dm_prediction_fast(model_B, mol_data_list[2]) if model_B is not None else None
    dm_nn_D1 = alec.oneshot_dm_prediction_fast(model_D1, mol_data_list[2]) if model_D1 is not None else None
    dm_nn_D2 = alec.oneshot_dm_prediction_fast(model_D2, mol_data_list[2]) if model_D2 is not None else None

    # 2x2 panel assignment: top row = (PBE-HF, best-B NN-HF), bottom = (best-D1, best-D2).
    _panel_deltas = [("PBE \u2212 HF", dm_pbe - dm_hf)]
    if dm_nn_B is not None:
        _panel_deltas.append(("best-B NN \u2212 HF", dm_nn_B - dm_hf))
    if dm_nn_D1 is not None:
        _panel_deltas.append(("best-D1 NN \u2212 HF", dm_nn_D1 - dm_hf))
    if dm_nn_D2 is not None:
        _panel_deltas.append(("best-D2 NN \u2212 HF", dm_nn_D2 - dm_hf))
    vmax = max(float(jnp.abs(delta).max()) for _, delta in _panel_deltas)

    _n_panels = len(_panel_deltas)
    _ncols = min(_n_panels, 2)
    _nrows = (_n_panels + _ncols - 1) // _ncols
    fig, axes = plt.subplots(_nrows, _ncols, figsize=(11, 10), squeeze=False)
    _axes_flat = axes.flatten()
    for ax, (title, delta) in zip(_axes_flat, _panel_deltas):
        im = ax.imshow(np.asarray(delta), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("AO basis index $j$")
        ax.set_ylabel("AO basis index $i$")
        fig.colorbar(
            im,
            ax=ax,
            label=r"$\\Delta \\mathrm{DM}_{ij}$ (residual vs HF target)",
        )
    for ax in _axes_flat[_n_panels:]:
        ax.set_visible(False)

    fig.suptitle(
        "H2O density-matrix residuals vs HF target -- PBE baseline "
        "and best NN per DM-loss family\\n"
        "(RdBu_r diverging colormap, shared symmetric vmax across panels)",
        fontsize=12,
    )

    # Inline Frobenius RMSE per panel (display-only; not a library metric).
    dm_rmse_pbe_hf = float(jnp.linalg.norm(dm_pbe - dm_hf) / jnp.sqrt(dm_pbe.size))
    print(f"PBE|HF Frobenius DM RMSE: {dm_rmse_pbe_hf:.4e}")
    if dm_nn_B is not None:
        dm_rmse_B = float(jnp.linalg.norm(dm_nn_B - dm_hf) / jnp.sqrt(dm_nn_B.size))
        print(f"best-B NN|HF Frobenius DM RMSE: {dm_rmse_B:.4e}")
    if dm_nn_D1 is not None:
        dm_rmse_D1 = float(jnp.linalg.norm(dm_nn_D1 - dm_hf) / jnp.sqrt(dm_nn_D1.size))
        print(f"best-D1 NN|HF Frobenius DM RMSE: {dm_rmse_D1:.4e}")
    if dm_nn_D2 is not None:
        dm_rmse_D2 = float(jnp.linalg.norm(dm_nn_D2 - dm_hf) / jnp.sqrt(dm_nn_D2.size))
        print(f"best-D2 NN|HF Frobenius DM RMSE: {dm_rmse_D2:.4e}")

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    os.makedirs(f"{CHECKPOINT_BASE}/figures", exist_ok=True)
    fig.savefig(f"{CHECKPOINT_BASE}/figures/dm_heatmaps_h2o.png", dpi=150, bbox_inches="tight")
    plt.show()
"""
    return new_code_cell(source)


def build_cell_27_density_histograms():
    """Section 7 Cell 27 -- grid density difference histograms for C/D3 best models."""
    source = """_c_d3_losses = ("C_atomization_plus_grid", "D3_delta_ae_plus_grid")
_c_d3_losses_present = [ln for ln in _c_d3_losses if ln in best_idx.index]

if not _c_d3_losses_present:
    print("[Cell 27] no grid-density-loss models in this configuration; skipping density histograms")
else:
    fig, axes = plt.subplots(1, len(_c_d3_losses_present), figsize=(14, 5.5), squeeze=False)
    _axes_flat = axes.flatten()
    for ax, loss_name in zip(_axes_flat, _c_d3_losses_present):
        best_arch = best_idx[loss_name]
        arch_config = alec.get_architecture(best_arch)
        model_template = alec.AlecGGAModel.from_arch(arch_config)
        ckpt_path = f"{CHECKPOINT_BASE}/train/{best_arch}/{loss_name}/model.eqx"
        model = eqx.tree_deserialise_leaves(ckpt_path, model_template)

        rho_nn = alec.oneshot_grid_density(model, mol_data_list[2])
        rho_ref = mol_data_list[2]["rho_ccsd_grid"]
        w = mol_data_list[2]["grid_weights"]

        # Inline step3b Table 4 |delta rho|_1 metric (display-only, not a library metric).
        delta_rho_L1 = float(jnp.sum(w * jnp.abs(rho_nn - rho_ref)) / jnp.sum(w))
        print(f"H2O |drho|_1 ({loss_name}, best={best_arch}): {delta_rho_L1:.4e} e/bohr^3")

        _diff = np.asarray(rho_nn - rho_ref)
        _w = np.asarray(w)
        ax.hist(_diff, bins=60, weights=_w)
        ax.set_yscale("log")
        ax.axvline(0.0, color="k", linewidth=0.8, label=r"perfect match ($\\Delta\\rho = 0$)")
        _lib_rmse = df.loc[(best_arch, loss_name), "density_rmse_mean"]
        ax.set_title(
            f"loss family {loss_name}\\n"
            f"best arch = {best_arch}  |  lib density RMSE = {_lib_rmse:.2e}  |  "
            rf"$|\\Delta\\rho|_1$ = {delta_rho_L1:.2e}"
        )
        ax.set_xlabel(r"$\\rho_{\\mathrm{NN}} - \\rho_{\\mathrm{HF}}$  (a.u., electron/bohr$^3$)")
        ax.set_ylabel("grid-weighted point count (log scale)")
        ax.grid(True, which="both", axis="y", ls=":", alpha=0.4)
        ax.legend(loc="upper right", fontsize="small")

    fig.suptitle(
        "H2O grid-density residual histograms -- best architecture per "
        "density-aware loss family\\n"
        "(reference density is the HF target stored in mol_data_list[2][\\"rho_ccsd_grid\\"])",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    os.makedirs(f"{CHECKPOINT_BASE}/figures", exist_ok=True)
    fig.savefig(f"{CHECKPOINT_BASE}/figures/grid_density_diffs.png", dpi=150, bbox_inches="tight")
    plt.show()
"""
    return new_code_cell(source)


def build_cell_28_attn_comparison():
    """Section 7 Cell 28 -- attention vs non-attention paired bar comparison."""
    source = """if not pairs:
    print("[Cell 28] no attention pairs in this configuration")
else:
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), squeeze=False)
    _axes_flat = axes.flatten()
    for ax, loss_name in zip(_axes_flat, LOSS_NAMES):
        x_positions = np.arange(len(pairs))
        base_heights = [df.loc[(base, loss_name), "AE_error_kcalmol_mean"] for base, _attn in pairs]
        attn_heights = [df.loc[(attn, loss_name), "AE_error_kcalmol_mean"] for _base, attn in pairs]
        bar_width = 0.35
        ax.bar(x_positions - bar_width / 2, base_heights, width=bar_width, label="no attention")
        ax.bar(x_positions + bar_width / 2, attn_heights, width=bar_width, hatch="//", label="with attention")
        ax.set_xticks(x_positions)
        ax.set_xticklabels([base for base, _ in pairs], rotation=30, ha="right")
        ax.set_xlabel("base architecture")
        ax.set_ylabel("signed AE error (kcal/mol)")
        ax.set_title(f"loss family: {loss_name}")
        ax.axhline(0.0, color="k", lw=0.6, alpha=0.6)
        ax.grid(True, axis="y", ls=":", alpha=0.4)
        ax.legend(fontsize="small")

    fig.suptitle(
        "Attention vs non-attention comparison -- H2O signed AE error "
        "per loss family\\n"
        "(signed so positive = NN over-predicts AE; closer-to-zero bar = more accurate)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    os.makedirs(f"{CHECKPOINT_BASE}/figures", exist_ok=True)
    fig.savefig(f"{CHECKPOINT_BASE}/figures/attn_vs_no_attn.png", dpi=150, bbox_inches="tight")
    plt.show()
"""
    return new_code_cell(source)


def build_cell_29_feature_comparison():
    """Section 7 Cell 29 -- extended features impact (deep base variants only)."""
    source = """feature_variants = [n for n in ARCH_NAMES if n.startswith("deep") and not n.endswith("_attn")]

if not feature_variants:
    print("[Cell 29] no deep-feature variants in this configuration; skipping feature comparison plot")
else:
    fig, ax = plt.subplots(figsize=(12, 6.5))
    n_losses = len(LOSS_NAMES)
    x_positions = np.arange(len(feature_variants))
    bar_width = 0.8 / n_losses
    for j, loss_name in enumerate(LOSS_NAMES):
        heights = [df.loc[(variant, loss_name), "AE_error_kcalmol_mean"] for variant in feature_variants]
        offset = (j - (n_losses - 1) / 2) * bar_width
        ax.bar(x_positions + offset, heights, width=bar_width, label=loss_name)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(feature_variants, rotation=20, ha="right")
    ax.set_xlabel("architecture (deep base variants: rho/sigma + descriptor set)")
    ax.set_ylabel("signed AE error (kcal/mol)")
    ax.set_title(
        "Extended-feature impact on H2O AE error -- deep base variants x loss family\\n"
        "(signed error; bars closer to zero = more accurate)",
        fontsize=12,
    )
    ax.axhline(0.0, color="k", lw=0.6, alpha=0.6)
    ax.grid(True, axis="y", ls=":", alpha=0.4)
    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=8,
        title="loss family",
    )
    fig.tight_layout()
    os.makedirs(f"{CHECKPOINT_BASE}/figures", exist_ok=True)
    fig.savefig(f"{CHECKPOINT_BASE}/figures/extended_features_impact.png", dpi=150, bbox_inches="tight")
    plt.show()
"""
    return new_code_cell(source)


def build_cell_30_future_md():
    """Section 8 Cell 30 -- future testing markdown."""
    source = """## Section 8: Test on New Molecules

Cell 31 is a self-contained template for extending step 4 to a new molecule:
it generates its own PBE/HF/CCSD reference data (`.npz` + metadata sidecar)
on first run and caches it under `ext_data_dir` for subsequent runs. Cell 32
then sweeps every trained `(arch, loss)` checkpoint and renders a 3-panel
comparison against PBE / CCSD / HF / chemical-accuracy reference lines.

To re-parameterise this section for a different molecule:

1. **Update `new_mol_spec`** (Cell 31) with your target's `name`, `atom`,
   `basis=BASIS`, `charge`, `spin`, `atom_composition`, `grid_level=GRID_LEVEL`,
   and an `external_data_path` under `{ext_data_dir}`.

2. **Update `new_atom_specs`** (Cell 31) to list every element in the new
   composition that is NOT already in Cell 12's `atom_energies` dict
   (`"H"`, `"O"`). Each entry is `(name, atom, spin)`; Cell 31 runs
   PBE+HF+CCSD for each, writes a `.npz` (atom branch only `E_ref_literature`,
   molecule branch also `dm_target` + `rho_ccsd_grid` + `rho_pbe_hf_rmse`),
   and a `{name}_metadata.json` sidecar with all three reference totals.
   `new_atom_energies` is then built from Cell 12 plus each sidecar's HF total.

3. **Update the literature AE reference** in `metric_kwargs` (Cell 31's
   `new_test_spec` — also propagated to Cell 32's per-combo specs).

4. **Cell 31 step 6 is commented** so a bare re-run of the notebook does not
   train/test on the new molecule. Cell 32 is the end-to-end sweep: uncomment
   Cell 31 step 6 only if you want a single best-D2 model run outside the
   full sweep.
"""
    return new_markdown_cell(source)


def build_cell_31_new_molecule_template():
    """Section 8 Cell 31 -- new-molecule template with reference data generation.

    Step 2 runs PBE / HF / CCSD for the new molecule AND every atom in
    ``new_atom_specs``, writing per-species ``.npz`` files and
    ``{name}_metadata.json`` sidecars (PBE/HF/CCSD totals, plus
    ``rho_pbe_hf_rmse`` for the molecule). Cell 32 then consumes those sidecars
    to draw PBE/CCSD/HF reference lines on the comparison plot.

    ``new_atom_energies`` is built from the Cell 12 dict and updated from each
    new atom's sidecar JSON (HF total) so adding another element only requires
    appending to ``new_atom_specs`` -- no hardcoded literature value lookup.
    """
    source = """# 1. Define the new molecule and any new atomic species its composition
#    requires that are NOT already in Cell 12's atom_energies dict (H, O).
#    Carbon ground state is 3P triplet => spin=2 in pyscf.
new_mol_spec = alec.MoleculeSpec(
    name="CH4",
    atom="C 0 0 0; H 0.63 0.63 0.63; H -0.63 -0.63 0.63; H -0.63 0.63 -0.63; H 0.63 -0.63 -0.63",
    basis=BASIS,
    charge=0,
    spin=0,
    atom_composition=(("C", 1), ("H", 4)),
    grid_level=GRID_LEVEL,
    external_data_path=f"{ext_data_dir}/CH4.npz",
)
new_atom_specs = [("C", "C 0 0 0", 2)]  # (name, atom, spin) -- extend as needed

# 2. Generate PBE/HF/CCSD reference data + metadata sidecar for the molecule
#    and every atom in new_atom_specs. Each iteration is guarded on both the
#    .npz and the metadata JSON so reruns are cheap. Mirrors Cell 13's
#    H/O/H2O pattern (atom-branch writes only E_ref_literature; molecule-branch
#    writes dm_target + rho_ccsd_grid + E_ref_literature).
_atom_names = {s[0] for s in new_atom_specs}
_entities = [(new_mol_spec.name, new_mol_spec.atom, new_mol_spec.spin)] + new_atom_specs
os.makedirs(ext_data_dir, exist_ok=True)

for _name, _atom, _spin in _entities:
    _npz_path = f"{ext_data_dir}/{_name}.npz"
    _meta_path = f"{ext_data_dir}/{_name}_metadata.json"
    if os.path.isfile(_npz_path) and os.path.isfile(_meta_path):
        print(f"Using cached {_name} reference data")
        continue

    _mol = gto.M(atom=_atom, basis=BASIS, charge=0, spin=_spin, verbose=0)

    # PBE SCF (spin-branched, grid pinned to GRID_LEVEL to match Cell 14/15).
    _mf_pbe = dft.UKS(_mol) if _spin else dft.RKS(_mol)
    _mf_pbe.xc = "pbe"
    _mf_pbe.grids.level = GRID_LEVEL
    _mf_pbe.kernel()
    _E_pbe_total = float(_mf_pbe.e_tot)

    # HF SCF (spin-branched).
    _mf_hf = scf.UHF(_mol) if _spin else scf.RHF(_mol)
    _mf_hf.kernel()
    _E_hf_total = float(_mf_hf.e_tot)

    # CCSD (spin-branched).
    _mycc = cc.UCCSD(_mf_hf) if _spin else cc.CCSD(_mf_hf)
    _mycc.kernel()
    _E_ccsd_total = float(_mf_hf.e_tot + _mycc.e_corr)

    _sidecar = {
        "E_hf_total": _E_hf_total,
        "E_ccsd_total": _E_ccsd_total,
        "E_pbe_total": _E_pbe_total,
        "E_lit_Ha": None,
    }

    _is_atom = _name in _atom_names
    if _is_atom:
        # Atom branch: degenerate HOMO eigenvalues make density targets
        # unstable; write only a scalar reference energy (HF total).
        np.savez(_npz_path, E_ref_literature=_E_hf_total)
    else:
        # Molecule branch: HF DM as density target + HF density on the PBE grid.
        _dm_hf = _mf_hf.make_rdm1()
        _dm_hf_total = _dm_hf[0] + _dm_hf[1] if _dm_hf.ndim == 3 else _dm_hf
        _coords = _mf_pbe.grids.coords
        _weights = _mf_pbe.grids.weights
        _ao = _mf_pbe._numint.eval_ao(_mol, _coords, deriv=0)
        _rho_hf = np.einsum("ij,gi,gj->g", _dm_hf_total, _ao, _ao)

        # PBE density on the same grid (mf_pbe already converged) so we can
        # report the PBE|HF weighted-RMSE reference for Cell 32's density panel.
        _dm_pbe = _mf_pbe.make_rdm1()
        _dm_pbe_total = _dm_pbe[0] + _dm_pbe[1] if _dm_pbe.ndim == 3 else _dm_pbe
        _rho_pbe = np.einsum("ij,gi,gj->g", _dm_pbe_total, _ao, _ao)
        _rho_pbe_hf_rmse = float(
            np.sqrt(np.sum(_weights * (_rho_pbe - _rho_hf) ** 2) / np.sum(_weights))
        )

        np.savez(
            _npz_path,
            dm_target=_dm_hf,
            rho_ccsd_grid=_rho_hf,
            E_ref_literature=float(_mf_hf.e_tot),
        )
        _sidecar["rho_pbe_hf_rmse"] = _rho_pbe_hf_rmse

    with open(_meta_path, "w") as _f:
        json.dump(_sidecar, _f, indent=2)
    print(f"Generated {_name} reference data -> {_npz_path}")

# 3. Build new_atom_energies from the Cell 12 dict + each new atom's HF total
#    (read from its sidecar, not hardcoded). Using HF here keeps the molecule's
#    E_ref_literature (also HF) and new_atom_energies self-consistent so the
#    AtomizationEnergyMetric's HF-baseline AE error panel line is clean.
new_atom_energies = {**atom_energies}
for _name, _atom, _spin in new_atom_specs:
    with open(f"{ext_data_dir}/{_name}_metadata.json") as _f:
        new_atom_energies[_name] = json.load(_f)["E_hf_total"]
print(f"new_atom_energies: {new_atom_energies}")

# 4. Pick a trained model (best D2 for DM-aware prediction; fall back to first
#    available loss family when running a narrow-config smoke test).
_d2_key = "D2_delta_ae_plus_dm"
best_arch = best_idx[_d2_key] if _d2_key in best_idx.index else best_idx.iloc[0]
_chosen_loss = _d2_key if _d2_key in best_idx.index else best_idx.index[0]

# 5. Build the TestSpec.
new_test_spec = alec.TestSpec.from_dicts(
    arch=alec.get_architecture(best_arch),
    model_checkpoint=f"{CHECKPOINT_BASE}/train/{best_arch}/{_chosen_loss}/model.eqx",
    molecules=(new_mol_spec,),
    metrics=("total_energy", "atomization_energy", "density_rmse"),
    metric_kwargs={"atomization_energy": {"reference_ae_kcalmol": {"CH4": 420.0}}},
    atom_energies=new_atom_energies,
    output_dir=f"{CHECKPOINT_BASE}/test_new/CH4",
)

# 6. Run it (commented out -- uncomment once you trust the template). Cell 32
#    runs the full sweep over every trained (arch, loss) combination to
#    produce the comparison plot.
# alec.run_test(new_test_spec)
"""
    return new_code_cell(source)


def build_cell_32_new_mol_comparison():
    """Section 8 Cell 32 -- new-molecule comparison sweep + 3-panel plot.

    Sweeps every trained (arch, loss) checkpoint for the new molecule defined
    in Cell 31, runs ``alec.run_test`` per combination, and renders a 1x3
    panel comparison against PBE / CCSD / HF reference values (sourced from
    the sidecar JSONs written by Cell 13 and Cell 31).

    Panels:

    * |AE error| (kcal/mol, log scale) with PBE / CCSD / HF / 1-kcal chemical
      accuracy reference lines. AE references are computed from the per-atom
      HF/CCSD/PBE totals in each species' sidecar versus the literature AE.
    * |E error| (kcal/mol, log scale) with PBE / CCSD reference lines. The
      molecule's .npz uses HF as ``E_ref_literature`` so the HF error line is
      trivially zero and is suppressed.
    * Density RMSE with the PBE-vs-HF density RMSE reference line from the
      molecule sidecar.

    Narrow-config tolerant: skips ``(arch, loss)`` pairs whose checkpoint
    does not exist (the smoke test only trains a single arch x loss combo).
    """
    source = """# Sweep every (arch, loss) combination, run alec.run_test on the new molecule,
# and collect the per-molecule AE / E / density error triples for plotting.
_mol_name = new_mol_spec.name
_sweep_rows = []
for _arch in ARCH_NAMES:
    for _loss in LOSS_NAMES:
        _ckpt = f"{CHECKPOINT_BASE}/train/{_arch}/{_loss}/model.eqx"
        if not os.path.isfile(_ckpt):
            continue
        _out_dir = f"{CHECKPOINT_BASE}/test_new/{_mol_name}/{_arch}/{_loss}"
        _spec = alec.TestSpec.from_dicts(
            arch=alec.get_architecture(_arch),
            model_checkpoint=_ckpt,
            molecules=(new_mol_spec,),
            metrics=("total_energy", "atomization_energy", "density_rmse"),
            metric_kwargs={"atomization_energy": {"reference_ae_kcalmol": {_mol_name: 420.0}}},
            atom_energies=new_atom_energies,
            output_dir=_out_dir,
        )
        _res = alec.run_test(_spec)
        _pm = _res["per_molecule"][0]
        _sweep_rows.append({
            "arch": _arch,
            "loss": _loss,
            "AE_error_kcalmol": float(abs(_pm.get("AE_error_kcalmol", float("nan")))),
            "E_error_kcalmol": float(abs(_pm.get("E_error_kcalmol", float("nan")))),
            "density_rmse": float(_pm.get("density_rmse", float("nan"))),
        })

if not _sweep_rows:
    print(f"[Cell 32] no checkpoints found under {CHECKPOINT_BASE}/train/ -- skipping plot")
else:
    _sweep_df = pd.DataFrame(_sweep_rows).set_index(["arch", "loss"])

    # Load reference totals from Cell 13 (H/O sidecars) and Cell 31
    # (CH4 + C sidecars) for the PBE / CCSD / HF comparison lines.
    _ref_totals = {"PBE": {}, "HF": {}, "CCSD": {}}
    _ref_species = [_mol_name] + [s[0] for s in new_atom_specs] + ["H"]
    for _sp in _ref_species:
        with open(f"{ext_data_dir}/{_sp}_metadata.json") as _f:
            _meta = json.load(_f)
        _ref_totals["PBE"][_sp] = _meta["E_pbe_total"]
        _ref_totals["HF"][_sp] = _meta["E_hf_total"]
        _ref_totals["CCSD"][_sp] = _meta["E_ccsd_total"]

    # The PBE|HF grid density RMSE lives in the molecule sidecar -- read it
    # for the density panel's PBE reference line. HF is the density target,
    # so HF / CCSD reference RMSEs on this grid are 0 / unavailable.
    with open(f"{ext_data_dir}/{_mol_name}_metadata.json") as _f:
        _rho_pbe_hf_rmse = json.load(_f)["rho_pbe_hf_rmse"]

    # Compute the reference AE and reference total-energy errors per method.
    _HA_TO_KCAL = 627.509
    _CH4_AE_LIT_KCAL = 420.0
    _comp = dict(new_mol_spec.atom_composition)
    _ref_ae_err_kcal = {}
    _ref_e_err_kcal = {}
    _hf_mol_total = _ref_totals["HF"][_mol_name]
    for _method, _totals in _ref_totals.items():
        _E_mol = _totals[_mol_name]
        _E_atoms = sum(_totals[_sym] * _n for _sym, _n in _comp.items())
        _AE_kcal = (_E_atoms - _E_mol) * _HA_TO_KCAL
        _ref_ae_err_kcal[_method] = abs(_AE_kcal - _CH4_AE_LIT_KCAL)
        _ref_e_err_kcal[_method] = abs((_E_mol - _hf_mol_total) * _HA_TO_KCAL)

    # 1x3 subplot grid: AE error / E error / density RMSE.
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    _ax_ae, _ax_e, _ax_rho = axes

    _x = np.arange(len(_sweep_df))
    _labels = [f"{a}/{l}" for a, l in _sweep_df.index]

    _ax_ae.bar(_x, _sweep_df["AE_error_kcalmol"].values, color="C0")
    _ax_ae.set_xticks(_x)
    _ax_ae.set_xticklabels(_labels, rotation=75, ha="right", fontsize=7)
    _ax_ae.set_ylabel("|AE error| (kcal/mol)")
    _ax_ae.set_yscale("log")
    _ax_ae.set_title(f"{_mol_name} atomization energy error")
    _ax_ae.axhline(_ref_ae_err_kcal["PBE"], linestyle="dotted", color="r", alpha=0.7,
                   label=f"PBE Error ({_ref_ae_err_kcal['PBE']:.2f} kcal/mol)")
    _ax_ae.axhline(_ref_ae_err_kcal["CCSD"], linestyle="-.", color="g", alpha=0.7,
                   label=f"CCSD Error ({_ref_ae_err_kcal['CCSD']:.2f} kcal/mol)")
    _ax_ae.axhline(_ref_ae_err_kcal["HF"], linestyle=":", color="b", alpha=0.7,
                   label=f"HF Error ({_ref_ae_err_kcal['HF']:.2f} kcal/mol)")
    _ax_ae.axhline(1.0, linestyle="--", color="k", alpha=0.7,
                   label="Chemical accuracy (1 kcal/mol)")
    _ax_ae.legend(fontsize=7, loc="best")

    _ax_e.bar(_x, _sweep_df["E_error_kcalmol"].values, color="C1")
    _ax_e.set_xticks(_x)
    _ax_e.set_xticklabels(_labels, rotation=75, ha="right", fontsize=7)
    _ax_e.set_ylabel("|E error vs HF| (kcal/mol)")
    _ax_e.set_yscale("log")
    _ax_e.set_title(f"{_mol_name} total energy error")
    _ax_e.axhline(_ref_e_err_kcal["PBE"], linestyle="dotted", color="r", alpha=0.7,
                  label=f"PBE vs HF ({_ref_e_err_kcal['PBE']:.2f} kcal/mol)")
    _ax_e.axhline(_ref_e_err_kcal["CCSD"], linestyle="-.", color="g", alpha=0.7,
                  label=f"CCSD vs HF ({_ref_e_err_kcal['CCSD']:.2f} kcal/mol)")
    _ax_e.legend(fontsize=7, loc="best")

    _ax_rho.bar(_x, _sweep_df["density_rmse"].values, color="C2")
    _ax_rho.set_xticks(_x)
    _ax_rho.set_xticklabels(_labels, rotation=75, ha="right", fontsize=7)
    _ax_rho.set_xlabel("(architecture / loss family)")
    _ax_rho.set_ylabel(r"density RMSE vs HF (a.u., log scale)")
    _ax_rho.set_yscale("log")
    _ax_rho.set_title(f"{_mol_name} grid-density RMSE vs HF target")
    _ax_rho.axhline(_rho_pbe_hf_rmse, linestyle="dotted", color="r", alpha=0.7,
                    label=f"PBE vs HF ({_rho_pbe_hf_rmse:.2e})")
    _ax_rho.grid(True, which="both", axis="y", ls=":", alpha=0.4)
    _ax_rho.legend(fontsize=7, loc="best")

    _ax_ae.set_xlabel("(architecture / loss family)")
    _ax_e.set_xlabel("(architecture / loss family)")
    _ax_ae.grid(True, which="both", axis="y", ls=":", alpha=0.4)
    _ax_e.grid(True, which="both", axis="y", ls=":", alpha=0.4)

    fig.suptitle(
        f"Transfer evaluation -- {_mol_name} errors across every "
        f"(arch, loss) checkpoint\\n"
        f"(one bar per trained combination; red / green / blue lines = "
        f"PBE / CCSD / HF references)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    os.makedirs(f"{CHECKPOINT_BASE}/figures", exist_ok=True)
    fig.savefig(f"{CHECKPOINT_BASE}/figures/new_mol_{_mol_name}_comparison.png",
                dpi=150, bbox_inches="tight")
    plt.show()
    print(_sweep_df.round(4))
"""
    return new_code_cell(source)


# ---------------------------------------------------------------------------
# Figure-description markdown builders.
#
# These cells are inserted into ``main()`` immediately BEFORE each comparison
# plot so the reader sees a prose explanation of what is being compared above
# the figure. Every helper returns a ``new_markdown_cell`` so the notebook's
# narrative stays adjacent to the plot it describes.
# ---------------------------------------------------------------------------


def build_section7_overview_md():
    """Section 7 header + Cell 25 description.

    This is the only markdown cell that introduces a whole section (Section 7
    -- Visualization of Results). It also describes the grouped-bar AE error
    chart rendered by the next code cell, so readers have context both for the
    section and the headline plot before the figure renders.
    """
    source = """## Section 7: Visualization of Results

This section compares every trained `(arch, loss)` combination against
physical reference lines (PBE, CCSD, chemical accuracy) and against each
other. The plots below are the primary results figures for the step 4
experiment -- each visualization is preceded by a short description of
what is being compared and how to read it.

### Figure: H2O atomization-energy error by architecture x loss family

The next cell renders a **grouped bar chart** of `|AE error|` (kcal/mol, log
scale). Each of the 6 loss families (A, B, C, D1, D2, D3) forms an x-axis
group; each of the 12 architectures contributes one coloured bar inside
every group. Error bars are the per-molecule RMSE from the evaluation set
(currently a single H2O geometry, so they're small).

Three horizontal reference lines are overlaid:

- **PBE Error** -- AE error of plain PBE (no neural correction).
- **CCSD Error** -- AE error of the CCSD baseline (effectively the best any
  fit-to-HF-target can achieve on this basis).
- **Chemical accuracy (1 kcal/mol)** -- the standard threshold for
  benchmark-quality DFT.

**How to read it:** bars *below* the chemical-accuracy line have reached the
target accuracy on H2O. Bars below the PBE line show net improvement over
the PBE baseline. The y-axis is log-scale so even sub-kcal/mol differences
are visible.
"""
    return new_markdown_cell(source)


def build_cell_26_dm_heatmaps_md():
    """Cell 26 description -- H2O density-matrix residuals vs HF target."""
    source = """### Figure: H2O density-matrix residuals vs HF target

The next cell compares the **H2O density matrix error** (NN prediction
minus HF reference) against the PBE baseline error in a grid of heatmaps.
Only loss families that actually optimize the density matrix are shown:

- **PBE - HF** -- fixed-density baseline (what every NN model starts from).
- **best-B NN - HF** -- best architecture under loss B
  (atomization energy + DM matching).
- **best-D1 NN - HF** -- best under loss D1
  (delta-learning energy only -- no density term, shown as a control).
- **best-D2 NN - HF** -- best under loss D2
  (delta-learning energy + DM matching).

All panels use a shared symmetric diverging colormap (`RdBu_r`) with a
common `vmax`, so bar magnitudes across panels are directly comparable.
Panels also report the Frobenius density-matrix RMSE below the figure as a
quantitative summary of each residual.

**How to read it:** panels whose colours are mostly white (near-zero) match
HF better than the PBE baseline. Diverging blue/red patches show where the
network systematically over- or under-predicts a density-matrix element.
"""
    return new_markdown_cell(source)


def build_cell_27_density_histograms_md():
    """Cell 27 description -- grid-density residual histograms."""
    source = """### Figure: H2O grid-density residual histograms

The next cell compares the two **density-grid-matching** loss families (C
and D3) by selecting the best-performing architecture for each and plotting
a weighted histogram of per-grid-point density residuals
$\\rho_{\\mathrm{NN}} - \\rho_{\\mathrm{HF}}$.

- **x-axis:** density residual in atomic units (electron / bohr$^{3}$).
  The vertical line at 0 marks perfect density agreement.
- **y-axis:** grid-weighted point count on a log scale
  (weights come from the molecular integration grid).
- Each panel title reports the library density RMSE and the step3b
  $|\\Delta\\rho|_1$ metric as a quantitative summary.

**How to read it:** a tight, centred distribution means the NN grid density
matches HF closely on the grid used for evaluation. A distribution that
leans left or right indicates a systematic over- or under-prediction of
density across the molecule.
"""
    return new_markdown_cell(source)


def build_cell_28_attn_comparison_md():
    """Cell 28 description -- attention vs non-attention pairs."""
    source = """### Figure: Attention vs non-attention architecture comparison

For each of the 6 loss families, the next cell compares every base
architecture (`shallow`, `medium`, `deep`, and their extended-feature
variants) against its attention-augmented counterpart (`*_attn`). Each
subplot shows:

- **Blue (solid) bars:** non-attention base architectures.
- **Orange (hatched) bars:** attention-augmented counterparts.
- **x-axis:** base architecture name (the pair is implicit).
- **y-axis:** *signed* AE error in kcal/mol -- positive means the NN
  over-predicts atomization energy, negative means it under-predicts.

**How to read it:** attention improves accuracy whenever the attention bar
is closer to the zero line than its non-attention neighbour. If both bars
sit on the same side of zero the attention variant is merely a different
amount of the same bias; if they straddle zero the attention module is
changing the sign of the bias.
"""
    return new_markdown_cell(source)


def build_cell_29_feature_comparison_md():
    """Cell 29 description -- extended-feature impact on deep base variants."""
    source = """### Figure: Extended-feature impact on deep base variants

The next cell isolates the four **deep base variants** and compares them
across every loss family so the reader can see how much each extended
descriptor set helps:

- `deep` -- bare `(rho, sigma)` inputs (no extended features).
- `deep_cusp` -- adds the cusp descriptor $[f_{cusp}, \\log Z]$.
- `deep_dm` -- adds the DM-statistics descriptor
  $[f_{idem}, f_{entropy}, f_{offdiag}]$.
- `deep_combined` -- adds both extended descriptor sets.

Each subplot group is one architecture on the x-axis; coloured bars within
a group are the 6 loss families. The y-axis is **signed** AE error in
kcal/mol, so positive and negative biases are directly comparable. A
horizontal line at zero marks perfect agreement with the literature AE.

**How to read it:** the descriptor set helps a loss family when the `deep`
bar is farther from zero than the corresponding `deep_cusp` / `deep_dm` /
`deep_combined` bar. The attention-augmented variants are deliberately
excluded here so the descriptor effect is not confounded with attention.
"""
    return new_markdown_cell(source)


def build_cell_32_new_mol_comparison_md():
    """Cell 32 description -- transfer sweep comparison on a new molecule."""
    source = """### Figure: Transfer evaluation on the new molecule

The next cell sweeps **every trained `(arch, loss)` checkpoint** that
exists on disk, runs `alec.run_test` on the new molecule from Cell 31, and
renders a three-panel comparison so the reader can see which combination
generalises best.

- **Panel 1 -- `|AE error|` (kcal/mol, log scale):** atomization-energy
  error vs literature. Reference lines show the PBE / CCSD / HF method
  errors against the same literature value, plus the 1 kcal/mol chemical
  accuracy target.
- **Panel 2 -- `|E error vs HF|` (kcal/mol, log scale):** total-energy
  error against the HF reference. PBE and CCSD reference lines show each
  wavefunction method's error vs HF. The HF self-error is trivially zero
  and is suppressed.
- **Panel 3 -- density RMSE vs HF (log scale):** grid density residual
  against the HF density target. The reference line is the PBE-vs-HF grid
  RMSE stored in the molecule sidecar (there is no CCSD grid density on
  this grid, so PBE is the only reference).

**How to read it:** within each panel the x-axis enumerates `(arch, loss)`
combinations, so you can spot which checkpoint transfers best for AE,
total energy, and density simultaneously. Bars below the coloured
reference lines show genuine improvement over the corresponding reference
method on the new molecule.
"""
    return new_markdown_cell(source)


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

    Notes
    -----
    Parallel pretraining is an in-notebook runtime toggle, not a generator
    argument: flip ``PRETRAIN_PARALLEL = True`` in Cell 3 and re-run Cell 8
    to opt in. See ``build_cell_08_pretrain_loop`` for the trade-offs.
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
        build_cell_14_mol_specs(),
        build_cell_15_precompute_sanity(),
        build_cell_16_training_md(),
        build_cell_17_training_specs(loss_names),
        build_cell_18_training_loop(),
        build_cell_19_training_loss_plot(),
        build_cell_20_aux_inspection(),
        build_cell_21_eval_md(),
        build_cell_22_test_loop(),
        build_cell_23_dataframe(),
        build_cell_24_results_table(),
        # Section 7 header + Cell 25 (AE bars) description.
        build_section7_overview_md(),
        build_cell_25_ae_bars(),
        # Per-comparison description markdown precedes each comparison plot.
        build_cell_26_dm_heatmaps_md(),
        build_cell_26_dm_heatmaps(),
        build_cell_27_density_histograms_md(),
        build_cell_27_density_histograms(),
        build_cell_28_attn_comparison_md(),
        build_cell_28_attn_comparison(),
        build_cell_29_feature_comparison_md(),
        build_cell_29_feature_comparison(),
        build_cell_30_future_md(),
        build_cell_31_new_molecule_template(),
        build_cell_32_new_mol_comparison_md(),
        build_cell_32_new_mol_comparison(),
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
    main("notebooks/gga_training_example-step4.ipynb")
