"""Build train_dfs_density.ipynb from source cells.

Run:  python _build_notebook.py   ->  writes train_dfs_density.ipynb
"""
import os

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def code(text):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


# ---------------------------------------------------------------------------
md(r"""
# DFS self-consistent density training

Trains xcquinox exchange-correlation networks self-consistently -- differentiating through the
Kohn-Sham SCF loop -- to reproduce CCSD electron densities, with energies anchored to GMTKN55
atomization energies. The training configuration matches the repo's `dfs_step7` recipe, which
replicates Dick & Fernandez-Serra, Phys. Rev. B 104, L161109 (2021):

- loss: density term (weight 20, normalized by N_e^2) + atomization-energy term (weight 1), in the
  `per_molecule` update loop (one optimizer step per species-group per epoch);
- atomization energies as reactions (molecule -> constituent atoms), scored with the network's own
  self-consistent atom energies;
- `full_3` (3-cycle) and `full_25` (25-cycle) FULL differentiable SCF; decaying mixer
  `alpha = 0.3^step + 0.3`; tail-weighted energy loss;
- adamw with linear learning-rate decay;
- networks pretrained to PBE before density training (they zero-initialize to LDA).

Architectures: `deep_3x16` and `deep_rung35_3x16`. Change `ARCH_NAMES` in the setup cell to use others.

Deviations from PRB L161109: CCSD (not CCSD(T)) reference densities; GGA + rung-3.5 networks (not the
paper's meta-GGA); `grid_level=2` (paper 3); adamw + linear decay (paper Adam + ReduceLROnPlateau);
spin-summed `N_e^2` (paper per-spin `N_sigma^2`).

Set `STEP_SMOKE=1` for a small end-to-end run (2 systems, `full_3`, few epochs, well-conditioned
basis). The full run (4 models at `6-311++G(3df,2pd)`, `full_25` = 25 differentiated SCF cycles) is
compute-heavy.
""")

# ---------------------------------------------------------------------------
code(r"""
import os, sys, time, warnings
import numpy as np

import jax
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_device", jax.devices("cpu")[0])
os.makedirs(".jax_compilation_cache", exist_ok=True)
jax.config.update("jax_compilation_cache_dir", ".jax_compilation_cache")
import jax.numpy as jnp
import matplotlib.pyplot as plt
try:
    import pandas as pd
except ImportError:
    pd = None

# Density-only supervision: no OEP V_xc references, so L5's vxc channel is inert
# (contributes 0). The per_molecule loop forces vxc_weight=1.0, so it emits a
# per-step vxc_ref=None warning. The DFS paper's loss has no vxc term, so this is
# expected; silence the repeated warning.
warnings.filterwarnings("ignore", message=r".*vxc_ref=None.*", category=RuntimeWarning)

sys.path.insert(0, os.getcwd())
import dfs_demo
from xcquinox.alec import run_training, run_test

SMOKE = os.environ.get("STEP_SMOKE", "0") == "1"
if SMOKE:
    HILLS          = dfs_demo.SMOKE_MOLECULE_HILLS   # H2O + OH
    BASIS          = "6-311G(d,p)"                   # no diffuse functions -> well-conditioned
    GRID_LEVEL     = 1
    ARCH_NAMES     = ("deep_3x16", "deep_rung35_3x16")
    SOLVER_NAMES   = ("full_3",)
    N_EPOCHS       = {"full_3": 3, "full_25": 3}
    PRETRAIN_STEPS = 20
else:
    HILLS          = dfs_demo.DEFAULT_MOLECULE_HILLS  # H2O, LiH, OH, NH
    BASIS          = dfs_demo.DFS_BASIS               # 6-311++G(3df,2pd)
    GRID_LEVEL     = dfs_demo.DFS_GRID_LEVEL          # 2
    ARCH_NAMES     = dfs_demo.ARCH_NAMES              # deep_3x16, deep_rung35_3x16
    SOLVER_NAMES   = ("full_3", "full_25")
    N_EPOCHS       = dfs_demo.DFS_N_EPOCHS            # 150 / 100
    PRETRAIN_STEPS = dfs_demo.DFS_PRETRAIN_STEPS      # 2500

DO_PRETRAIN = True

OUT_DIR      = os.path.abspath("runs_smoke" if SMOKE else "runs")
# The full run reuses the top-level refs/ CCSD cache; the smoke isolates its refs
# under runs_smoke/ so a smoke (grid 1, small basis) never collides with a full
# run (grid 2, big basis) sharing one refs/ dir -- which risks a stale-ref
# grid_level/basis mismatch at consumption.
REFS_DIR     = os.path.join(OUT_DIR, "refs") if SMOKE else os.path.abspath("refs")
PRETRAIN_DIR = os.path.join(OUT_DIR, "pretrain")
for _d in (REFS_DIR, OUT_DIR, PRETRAIN_DIR):
    os.makedirs(_d, exist_ok=True)


def make_progress(label, updates=20):
    # Robust flushed-print progress that works in Jupyter, nbconvert, and a plain
    # terminal (unlike a tqdm bar, which can silently no-op headless). Handles the
    # run_training callback (phase "train") and the run_pretrain callbacks
    # (phases "X" then "C"), each with its own step/total.
    state = {}
    def cb(p):
        phase = p.get("phase", "")
        step, total, loss = int(p["step"]), int(p["total"]), float(p["loss"])
        key = (label, phase)
        if step <= 1 or key not in state:
            state[key] = time.time()
        every = max(1, total // updates)
        if step == 1 or step % every == 0 or step >= total:
            el = time.time() - state[key]
            eta = el / max(step, 1) * max(total - step, 0)
            tag = (label + " " + phase).strip()
            print(f"    [{tag}] {step}/{total}  loss={loss:.4g}  "
                  f"{el:.0f}s elapsed, ~{eta:.0f}s left", flush=True)
    return cb


print("SMOKE =", SMOKE, "| basis =", BASIS, "grid", GRID_LEVEL)
print("archs =", ARCH_NAMES, "| solvers =", SOLVER_NAMES, "| epochs =", {s: N_EPOCHS[s] for s in SOLVER_NAMES})
""")

# ---------------------------------------------------------------------------
md(r"""
## 1. Systems

Spin-diverse subset from `build_dfs_pool()` (Haunschild/GMTKN55 atomization energies, cited spins).
Closed-shell H2O, LiH; open-shell OH (doublet), NH (triplet). Atomization energies train as reactions,
so each molecule carries its constituent H/O/Li/N atoms; the density objective applies to the
molecules (atoms are skipped).
""")

code(r"""
chosen = dfs_demo.select_dfs_points(HILLS)
mol_specs = dfs_demo.build_mol_specs(chosen, basis=BASIS, grid_level=GRID_LEVEL, refs_dir=REFS_DIR)

rows = []
for ms in mol_specs:
    natoms = sum(dict(ms.atom_composition).values())
    rows.append({"name": ms.name, "spin(2S)": ms.spin, "charge": ms.charge,
                 "shell": "open" if ms.spin else "closed",
                 "kind": "molecule" if natoms > 1 else "atom"})
if pd is not None:
    try:
        from IPython.display import display
        display(pd.DataFrame(rows).sort_values(["kind", "name"]).reset_index(drop=True))
    except Exception:
        print(pd.DataFrame(rows))
else:
    for r in rows:
        print(r)
""")

# ---------------------------------------------------------------------------
md(r"""
## 2. CCSD reference densities

Per-molecule CCSD reference density `rho_ref_grid` (converged HF -> CCSD 1-RDM -> spin-summed density
on the PBE-SCF grid), via `benchmark_refs.generate_one`, at the training basis so the grids align.
Cached; atoms need no reference.
""")

code(r"""
dfs_demo.generate_ccsd_density_refs(
    mol_specs, refs_dir=REFS_DIR, basis=BASIS, grid_level=GRID_LEVEL)

mol_specs = dfs_demo.build_mol_specs(chosen, basis=BASIS, grid_level=GRID_LEVEL, refs_dir=REFS_DIR)
print("molecules with CCSD density:", [ms.name for ms in mol_specs if ms.external_data_path])

ex = next(ms for ms in mol_specs if ms.external_data_path)
_z = np.load(ex.external_data_path)
print(f"integral rho_ref[{ex.name}] = {float((_z['rho_ref_grid'] * _z['grid_weights']).sum()):.4f} electrons")
""")

# ---------------------------------------------------------------------------
md(r"""
## 3. Pretrain to PBE

The networks zero-initialize to LDA (`F_x = F_c = 1` multiply `lda_x` + PW92, the uniform-gas limit).
The DFS recipe fits `F` to PBE first (`Fx = F_x^PBE/F_x^LDA - 1`). This runs one PBE fit per
architecture and writes `xnet.eqx`/`cnet.eqx` used as the training warm-start.
""")

code(r"""
pretrained = {}
if DO_PRETRAIN:
    pretrain_atoms = dfs_demo.pretrain_atoms_for(mol_specs)
    print("pretrain atoms (derived from the systems):", [a[0] for a in pretrain_atoms], flush=True)
    for arch_name in ARCH_NAMES:
        ck = os.path.join(PRETRAIN_DIR, arch_name)
        print(f"pretrain {arch_name} (reuse cached checkpoint if present, else "
              f"{PRETRAIN_STEPS} steps per X/C net):", flush=True)
        dfs_demo.pretrain_to_pbe(
            dfs_demo.dfs_arch(arch_name), data_dir=PRETRAIN_DIR, checkpoint_dir=ck,
            basis=BASIS, grid_level=GRID_LEVEL, atoms=pretrain_atoms, n_steps=PRETRAIN_STEPS,
            progress_callback=make_progress(arch_name))
        pretrained[arch_name] = ck
        print("  wrote", ck, flush=True)
else:
    print("pretraining off; networks start from the LDA init")
""")

# ---------------------------------------------------------------------------
md(r"""
## 4. Architectures

`deep_3x16` (plain GGA) and `deep_rung35_3x16` (cusp + rung-3.5 localized-DM occupancy), both with
spin-polarized correlation. Change `ARCH_NAMES` (setup cell) or pass a custom `ArchitectureConfig` to
`build_dfs_training_spec` to use your own.
""")

code(r"""
for a in ARCH_NAMES:
    arch = dfs_demo.dfs_arch(a)
    descs = [d.__class__.__name__ for d in arch.materialize_descriptors()]
    print(f"{a}: depth={arch.depth} nodes={arch.nodes} descriptors={descs or '[]'} "
          f"polarized_correlation={arch.use_polarized_correlation}")
""")

# ---------------------------------------------------------------------------
md(r"""
## 5. Training spec

`build_dfs_training_spec` calls the same `spec_builder`/domain functions the cluster harness uses for
`dfs_step7`, so the configuration matches a production run except for the pool size.
""")

code(r"""
solvers = dfs_demo.solver_configs()
_ex = dfs_demo.build_dfs_training_spec(
    arch=dfs_demo.dfs_arch(ARCH_NAMES[0]), solver_cfg=solvers[SOLVER_NAMES[0]],
    chosen_points=chosen, mol_specs=mol_specs,
    checkpoint_dir=os.path.join(OUT_DIR, "_example"), n_steps=N_EPOCHS[SOLVER_NAMES[0]],
    pretrain_checkpoint=pretrained.get(ARCH_NAMES[0]))
lk = _ex.loss_kwargs_dict
print("update_scheme:", _ex.update_scheme)
print("loss:", _ex.loss_name)
print("density_per_electron:", lk["density_per_electron"])
print("channel_weights:", _ex.channel_weights)
print("ae_as_reactions:", [r["name"] for r in lk["bh76_reactions"]])
print("regularize_atom_syms:", lk["regularize_atom_syms"])
print("use_polarized_correlation:", _ex.arch.use_polarized_correlation)
print("optimizer: adamw lr %.0e->%.0e decay@%.1f clip=%.1f wd=%.0e"
      % (_ex.lr_start, _ex.lr_end, _ex.lr_decay_start, _ex.grad_clip, _ex.weight_decay))
print("solver:", _ex.solver_config.mode, "max_cycles", _ex.solver_config.max_cycles,
      _ex.solver_config.mixer_name)
print("pretrain_checkpoint:", _ex.pretrain_checkpoint)
_ex.validate()
""")

# ---------------------------------------------------------------------------
md(r"""
## 6. Train

Each architecture under each solver. Every optimizer step differentiates through the full KS SCF.
`full_25` (25 SCF cycles) is the slow path.
""")

code(r"""
# Ensure every molecule's CCSD reference exists before training: regenerate any
# missing one (cached refs are instant) and re-wire mol_specs, so a deleted or
# partially generated ref can never crash training mid-run.
dfs_demo.generate_ccsd_density_refs(
    mol_specs, refs_dir=REFS_DIR, basis=BASIS, grid_level=GRID_LEVEL, progress=False)
mol_specs = dfs_demo.build_mol_specs(chosen, basis=BASIS, grid_level=GRID_LEVEL, refs_dir=REFS_DIR)

trained = {}
for arch_name in ARCH_NAMES:
    for solver_name in SOLVER_NAMES:
        ckpt = os.path.join(OUT_DIR, f"{arch_name}__{solver_name}")
        spec = dfs_demo.build_dfs_training_spec(
            arch=dfs_demo.dfs_arch(arch_name), solver_cfg=solvers[solver_name],
            chosen_points=chosen, mol_specs=mol_specs,
            checkpoint_dir=ckpt, n_steps=N_EPOCHS[solver_name],
            pretrain_checkpoint=pretrained.get(arch_name))
        print(f"train {arch_name} / {solver_name} ({N_EPOCHS[solver_name]} epochs)")
        meta = run_training(spec, progress_callback=make_progress(f"{arch_name}/{solver_name}"))
        trained[(arch_name, solver_name)] = {"spec": spec, "ckpt": ckpt, "meta": meta}
        print(f"  final_loss = {meta['final_loss']:.5g}")
""")

# ---------------------------------------------------------------------------
md(r"""
## 7. Evaluate

`run_test` under the FULL solver: `density_rmse` is the model's self-consistent density error vs CCSD;
`density_rmse_pbe` is the PBE-vs-CCSD baseline on the same grid.
""")

code(r"""
KCAL = 627.5094740631
CHAK = dfs_demo.DOMAIN.atom_energies
ae_ref_kcal = {tp.name: tp.metadata.get("e_rxn_ref") for tp in chosen}
comp_by_name = {ms.name: dict(ms.atom_composition) for ms in mol_specs}

evals = {}
for key, info in trained.items():
    ts = dfs_demo.build_dfs_test_spec(
        training_spec=info["spec"],
        model_checkpoint=os.path.join(info["ckpt"], "model.eqx"),
        solver_cfg=solvers[key[1]],
        output_dir=os.path.join(info["ckpt"], "eval"))
    res = run_test(ts)
    evals[key] = res
    dens = dfs_demo.aggregate_density_diagnostics(res["per_molecule"])
    print(f"{key[0]} / {key[1]}:")
    for r in dens:
        print(f"  {r['name']}: NN {r['density_rmse']:.4e}  PBE {r['density_rmse_pbe']:.4e}")
""")

# ---------------------------------------------------------------------------
md(r"""
## 8. Figures
""")

code(r"""
# (a) training-loss curves
fig, ax = plt.subplots(figsize=(7, 4))
for key, info in trained.items():
    lp = os.path.join(info["ckpt"], "losses.npy")
    if os.path.exists(lp):
        L = np.asarray(np.load(lp)).ravel()
        ax.plot(np.arange(len(L)), L, label=f"{key[0]}/{key[1]}")
ax.set_xlabel("optimizer step"); ax.set_ylabel("training loss")
if ax.has_data():
    ax.set_yscale("log")
ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "fig_loss_curves.png"), dpi=110); plt.show()
""")

code(r"""
# (b) self-consistent density RMSE vs CCSD: NN per (arch, solver) vs PBE baseline
mols = [ms.name for ms in dfs_demo.molecule_specs(mol_specs)]
pbe_rmse, nn_rmse = {}, {key: {} for key in evals}
for key, res in evals.items():
    for rec in res["per_molecule"]:
        if rec.get("density_rmse") is None:
            continue
        nn_rmse[key][rec["molecule"]] = rec["density_rmse"]
        pbe_rmse[rec["molecule"]] = rec.get("density_rmse_pbe")

x = np.arange(len(mols)); width = 0.8 / (len(evals) + 1)
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(x, [pbe_rmse.get(m, np.nan) for m in mols], width, label="PBE", color="0.6")
for i, key in enumerate(evals):
    ax.bar(x + (i + 1) * width, [nn_rmse[key].get(m, np.nan) for m in mols], width,
           label=f"NN {key[0]}/{key[1]}")
ax.set_xticks(x + 0.4 - width / 2); ax.set_xticklabels(mols)
ax.set_ylabel("density RMSE vs CCSD"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "fig_density_rmse.png"), dpi=110); plt.show()
""")

code(r"""
# (c) atomization-energy error vs GMTKN55 (diagnostic), fixed Chakravorty anchors
def ae_err_kcal(E_mol_ha, comp, ref_kcal):
    ae_ha = sum(CHAK[s] * n for s, n in comp.items()) - float(E_mol_ha)
    return ae_ha * KCAL - float(ref_kcal)

pbe_ae, nn_ae = {}, {key: {} for key in evals}
for key, res in evals.items():
    for rec in res["per_molecule"]:
        name = rec["molecule"]
        if name not in comp_by_name or ae_ref_kcal.get(name) is None:
            continue
        if rec.get("E_total_nn") is not None:
            nn_ae[key][name] = ae_err_kcal(rec["E_total_nn"], comp_by_name[name], ae_ref_kcal[name])
        if rec.get("E_pbe") is not None:
            pbe_ae[name] = ae_err_kcal(rec["E_pbe"], comp_by_name[name], ae_ref_kcal[name])

ae_mols = [m for m in mols if m in pbe_ae]
x = np.arange(len(ae_mols)); width = 0.8 / (len(evals) + 1)
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.axhline(0, color="k", lw=0.8)
ax.bar(x, [pbe_ae.get(m, np.nan) for m in ae_mols], width, label="PBE", color="0.6")
for i, key in enumerate(evals):
    ax.bar(x + (i + 1) * width, [nn_ae[key].get(m, np.nan) for m in ae_mols], width,
           label=f"NN {key[0]}/{key[1]}")
ax.set_xticks(x + 0.4 - width / 2); ax.set_xticklabels(ae_mols)
ax.set_ylabel("AE error vs GMTKN55 (kcal/mol)"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "fig_ae_error.png"), dpi=110); plt.show()
""")

# ---------------------------------------------------------------------------
md(r"""
## 9. Notes

- Figure (b): self-consistent density RMSE vs CCSD per molecule, NN vs the PBE baseline. `full_25`
  reaches a more self-consistent fixed point than `full_3` (25 vs 3 differentiated cycles).
- Energies (figure c) are a diagnostic; the density channel carries 20x the weight of the energy
  channel.
- To adapt: change `ARCH_NAMES` or pass a custom `ArchitectureConfig` to `build_dfs_training_spec`;
  extend `HILLS` with any `build_dfs_pool()` Hill formula. For the full pool + BH76/IP13 channels +
  V_xc supervision, use the cluster harness (`xcquinox.alec.cluster`) with the `dfs_step7` config; the
  CCSD reference generator there is `xcquinox.alec.external_refs.precompute_all` (adds the OEP V_xc
  cascade).
- Deviations from PRB L161109: CCSD (not CCSD(T)); GGA + rung-3.5 (not meta-GGA); `grid_level=2`;
  adamw + linear decay; spin-summed `N_e^2`.
""")

# ---------------------------------------------------------------------------
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_dfs_density.ipynb")
with open(_out, "w") as f:
    nbf.write(nb, f)
print("wrote", _out, "with", len(cells), "cells")
