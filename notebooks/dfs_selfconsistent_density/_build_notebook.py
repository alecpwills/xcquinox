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

Architectures: two GGA-ladder networks (`deep_3x16`, `deep_rung35_3x16`) and two meta-GGA networks
(`deep_mgga_3x16`, `deep_rung35_mgga_3x16`, which pretrain to SCAN). Change `ARCH_NAMES` in the setup
cell to use others. Section 7 adds a **SCAN** self-consistent baseline alongside PBE, so the
meta-GGA networks can be judged against the meta-GGA (SCAN) they were warm-started from, not just PBE.

Deviations from PRB L161109: CCSD (not CCSD(T)) reference densities; `grid_level=2` (paper 3);
adamw + linear decay (paper Adam + ReduceLROnPlateau); spin-summed `N_e^2` (paper per-spin `N_sigma^2`).
The arch set spans GGA -> rung-3.5 -> meta-GGA, so the paper's meta-GGA rung IS included (the
`deep_mgga_*` nets).

**Orientation lock.** OH (and the held-out NO in section 9) is an X-2-Pi *orbitally degenerate*
radical: its singly-occupied pi hole can sit in any combination of the degenerate `(pi_x, pi_y)` pair,
so its single-determinant density on a fixed grid is orientation-arbitrary and *not reproducible*
across processes/machines (threaded BLAS tips the near-degenerate SCF differently each run) -- even
though the energy is degeneracy-invariant. An **orientation lock** fixes this: a small, fixed,
traceless anisotropic-quadrupole bias is added to `h_core` **identically** in the CCSD reference, the
PBE seed, training, and eval, so the reference and the functional always select the *same*
representative of the degenerate manifold. The density becomes deterministic; energies shift
negligibly (< 0.1 kcal/mol). See the companion README for the physics and `xcquinox.alec.orientation_lock`.

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
    ARCH_NAMES     = ("deep_3x16", "deep_rung35_3x16",
                      "deep_mgga_3x16", "deep_rung35_mgga_3x16")
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

> **One-time reset if you trained before the orientation lock:** the lock is now on, so a fresh run
> trains against the locked OH reference. CCSD references self-heal (they carry the lock strength and
> regenerate automatically). Training checkpoints do **not** auto-invalidate, so if you have
> `runs/*__*/model.eqx` from an earlier *unlocked* run, delete them (keep `runs/pretrain/`, which is
> orientation-invariant) to retrain consistently: `rm runs/*__*/model.eqx`.
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
        # Reuse a finished training checkpoint on rerun: if model.eqx already exists
        # skip (re)training and just evaluate it. Delete <ckpt>/model.eqx (or the
        # whole runs/ dir) to force a fresh train.
        if os.path.isfile(os.path.join(ckpt, "model.eqx")):
            print(f"reuse trained {arch_name} / {solver_name}: {ckpt}/model.eqx", flush=True)
            trained[(arch_name, solver_name)] = {"spec": spec, "ckpt": ckpt, "meta": None}
            continue
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
### SCAN meta-GGA baseline

PBE is a GGA. **SCAN** (Sun, Ruzsinszky, Perdew, PRL 115, 036402 (2015)) is a *meta-GGA*: a
strongly-constrained, higher-rung functional that is systematically more accurate than PBE for many
densities. It is the natural comparator for the demo's two meta-GGA networks (`deep_mgga_3x16`,
`deep_rung35_mgga_3x16`), which pretrain to SCAN. The scientific question this section poses is
therefore sharper than "does the network beat PBE?": **does a network trained self-consistently on
CCSD densities improve on SCAN itself** -- the functional it was warm-started from -- at reproducing
the CCSD density and the atomization energy?

SCAN is *model-independent* (a fixed functional), so like the PBE baseline it is computed once over
the species union and reused for every trained network. Each species (molecules **and** the
atomization-energy constituent atoms) gets a SCAN Kohn-Sham SCF via `run_scf_with_cache(xc="scan")`,
reusing the **same** basis / grid / charge / spin / orientation-lock as the PBE and CCSD references
(so the degenerate radicals OH and held-out NO lock the same density component the CCSD reference
does). The SCAN self-consistent density is scored against CCSD on the identical evaluation grid, and
SCAN's own self-consistent atomization energy is formed from its own atom energies -- exactly
mirroring how the network and PBE are scored. Sections 8-10 then report **NN vs PBE vs SCAN**.

> The outcome is empirical and reported as-is: a trained network may or may not beat SCAN. Beating PBE
> is the low bar; beating SCAN (a meta-GGA) is the demanding one, especially for the GGA-rung networks.
""")

code(r"""
from xcquinox.alec.data import precompute_fixed_density_data

# SCAN is a fixed meta-GGA -> model-independent, so compute it ONCE over the
# species union and reuse for every trained model (exactly like the PBE
# baseline). Rebuild each molecule's eval mol_data (reference-grid ao_grid +
# grid_weights + CCSD rho_ref_grid) via precompute -- same orientation lock as
# run_test uses -- then run a SCAN KS-SCF per species (molecules AND their AE
# atoms), scoring SCAN's self-consistent density on that exact grid.
scan_mol_data = {
    ms.name: precompute_fixed_density_data(
        ms, orientation_lock_strength=dfs_demo.ORIENTATION_LOCK_STRENGTH)
    for ms in dfs_demo.molecule_specs(mol_specs)
}
scan_by_name = dfs_demo.scan_baseline(
    mol_specs, scan_mol_data, refs_dir=REFS_DIR, basis=BASIS, grid_level=GRID_LEVEL)

# Attach the SCAN series to every model's records so sections 8-10 (density, AE,
# combined ED) report NN vs PBE vs SCAN. SCAN is identical across models.
for key in evals:
    evals[key]["per_molecule"] = dfs_demo.attach_scan_baseline(
        evals[key]["per_molecule"], scan_by_name)

print("SCAN baseline (model-independent):")
for _m in [ms.name for ms in dfs_demo.molecule_specs(mol_specs)]:
    _s = scan_by_name.get(_m, {})
    _d = _s.get("density_rmse_scan")
    print(f"  {_m}: E_scan={_s.get('E_scan'):.5f} Ha  "
          f"density_rmse_scan={'n/a' if _d is None else f'{_d:.4e}'}")
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
# (b) self-consistent density RMSE vs CCSD: NN per (arch, solver) vs PBE + SCAN
mols = [ms.name for ms in dfs_demo.molecule_specs(mol_specs)]
pbe_rmse, nn_rmse = {}, {key: {} for key in evals}
for key, res in evals.items():
    for rec in res["per_molecule"]:
        if rec.get("density_rmse") is None:
            continue
        nn_rmse[key][rec["molecule"]] = rec["density_rmse"]
        pbe_rmse[rec["molecule"]] = rec.get("density_rmse_pbe")

def _nan(v):
    return np.nan if v is None else v
scan_rmse = {m: _nan((scan_by_name.get(m) or {}).get("density_rmse_scan")) for m in mols}

# PBE + SCAN baselines then one bar per model.
nbar = len(evals) + 2; x = np.arange(len(mols)); width = 0.8 / nbar
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.bar(x, [pbe_rmse.get(m, np.nan) for m in mols], width, label="PBE", color="0.6")
ax.bar(x + width, [scan_rmse.get(m, np.nan) for m in mols], width, label="SCAN", color="0.35")
for i, key in enumerate(evals):
    ax.bar(x + (i + 2) * width, [nn_rmse[key].get(m, np.nan) for m in mols], width,
           label=f"NN {key[0]}/{key[1]}")
ax.set_xticks(x + 0.4 - width / 2); ax.set_xticklabels(mols)
ax.set_ylabel("density RMSE vs CCSD"); ax.legend(fontsize=8)
# log y-scale: the OH radical (~2.6e-3) is 40-250x the closed-shell systems, so a
# linear axis hides the H2O/NH wins and the full_3-vs-full_25 spread.
ax.set_yscale("log"); ax.set_ylim(bottom=1e-6)
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "fig_density_rmse.png"), dpi=110); plt.show()
""")

code(r"""
# (c) atomization-energy error vs reference, from each functional's OWN
#     self-consistent atom energies (the physically correct AE, and what
#     ae_as_reactions trains). NOT the anchored AE_nn field, which subtracts the
#     molecule energy from FIXED exact atoms and so reports the net's absolute-
#     energy offset (tens-to-hundreds of kcal/mol), not its atomization energy.
ae_rows = {key: dfs_demo.self_consistent_ae(res["per_molecule"], comp_by_name, ae_ref_kcal)
           for key, res in evals.items()}
ae_mols = [r["name"] for r in next(iter(ae_rows.values()))]
_first = next(iter(ae_rows.values()))
pbe_err = {r["name"]: r["err_pbe"] for r in _first}
# SCAN AE error is model-independent (err_scan is identical across models).
scan_err = {r["name"]: r.get("err_scan") for r in _first}

nbar = len(evals) + 2; x = np.arange(len(ae_mols)); width = 0.8 / nbar
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.axhline(0, color="k", lw=0.8)
ax.bar(x, [pbe_err.get(m, np.nan) for m in ae_mols], width, label="PBE", color="0.6")
ax.bar(x + width, [scan_err.get(m) if scan_err.get(m) is not None else np.nan
                   for m in ae_mols], width, label="SCAN", color="0.35")
for i, (key, rows) in enumerate(ae_rows.items()):
    nn_err = {r["name"]: r["err_nn"] for r in rows}
    ax.bar(x + (i + 2) * width, [nn_err.get(m, np.nan) for m in ae_mols], width,
           label=f"NN {key[0]}/{key[1]}")
ax.set_xticks(x + 0.4 - width / 2); ax.set_xticklabels(ae_mols)
ax.set_ylabel("AE error vs reference (kcal/mol)"); ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "fig_ae_error.png"), dpi=110); plt.show()
""")

code(r"""
# (d) DFS-Fig.2-style 3-panel, NN vs PBE vs SCAN: energy AE-MAE (top), mean density error
#     (middle), combined ED (bottom). The mean density error is dominated by the OH
#     radical (~2.6e-3, ~30-250x the closed-shell systems), so the aggregate density
#     win is modest even though H2O/NH improve ~40% -- see the "excl. OH" print below.
#     ED = harmonic-mean(E_MAE, gamma*D), gamma self-calibrated from PBE so
#     ED_pbe == E_MAE_pbe (dfs_demo.combined_energy_density; DFS PRB 104 L161109 Eq. 21).
ed = {}
for key, res in evals.items():
    rows = dfs_demo.self_consistent_ae(res["per_molecule"], comp_by_name, ae_ref_kcal)
    drows = dfs_demo.aggregate_density_diagnostics(res["per_molecule"])
    ed[key] = dfs_demo.combined_energy_density(rows, drows)

# PBE is model-independent, so its mean density error MUST match across models. A
# spread flags inconsistent CCSD references between eval batches (a refs/*.npz that
# changed between runs -- the open-shell OH radical reference is the usual culprit).
# Re-run the Evaluate cell cleanly (training/pretraining are reused) so every model
# scores against one stable refs/ set.
_dpbe = [ed[k]["D_pbe"] for k in ed]
if _dpbe and (max(_dpbe) - min(_dpbe)) / min(_dpbe) > 0.02:
    print(f"WARNING: PBE mean density RMSE varies across models "
          f"({min(_dpbe):.3e}..{max(_dpbe):.3e}) -- but PBE is model-independent, so the CCSD "
          f"references are INCONSISTENT across eval batches. Cross-model density bars below are "
          f"not comparable; re-run section 7 (checkpoints are reused) for one stable refs/ set.",
          flush=True)

keys = list(evals); xk = np.arange(len(keys)); w = 0.26

def _grouped(ax, pbe_vals, scan_vals, nn_vals, fmt):
    ax.bar(xk - w, pbe_vals, w, label="PBE", color="0.6")
    ax.bar(xk + 0.0, scan_vals, w, label="SCAN", color="0.35")
    ax.bar(xk + w, nn_vals, w, label="NN", color="C0")
    for xi, (p, s, n) in enumerate(zip(pbe_vals, scan_vals, nn_vals)):
        ax.annotate(fmt(p), (xi - w, p), ha="center", va="bottom", fontsize=6)
        ax.annotate(fmt(s), (xi, s), ha="center", va="bottom", fontsize=6)
        ax.annotate(fmt(n), (xi + w, n), ha="center", va="bottom", fontsize=6)
    ax.margins(y=0.20)

def _col(k, field):
    return [ed[kk].get(field, np.nan) for kk in k]

fig, (axE, axD, axED) = plt.subplots(3, 1, figsize=(8, 9), sharex=True)
_grouped(axE, _col(keys, "E_MAE_pbe"), _col(keys, "E_MAE_scan"), _col(keys, "E_MAE_nn"),
         lambda v: f"{v:.2f}")
axE.set_ylabel("AE-MAE (kcal/mol)"); axE.set_title("energy error"); axE.legend(fontsize=8)
_grouped(axD, _col(keys, "D_pbe"), _col(keys, "D_scan"), _col(keys, "D_nn"), lambda v: f"{v:.2e}")
axD.set_ylabel("mean density RMSE vs CCSD")
axD.set_title("density error (mean over molecules; OH radical dominates the mean)")
_grouped(axED, _col(keys, "ED_pbe"), _col(keys, "ED_scan"), _col(keys, "ED_nn"), lambda v: f"{v:.2f}")
axED.set_ylabel(r"$\mathcal{ED}$ (kcal/mol)")
axED.set_title("combined energy-density (DFS Eq. 21, gamma self-calibrated from PBE)")
axED.set_xticks(xk); axED.set_xticklabels([f"{k[0]}\n{k[1]}" for k in keys], fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "fig_combined_ed.png"), dpi=110); plt.show()

for k in keys:
    m = ed[k]
    no_oh = [r for r in dfs_demo.aggregate_density_diagnostics(evals[k]["per_molecule"])
             if r["name"] != "HO"]
    d_nn_x = sum(r["density_rmse"] for r in no_oh) / len(no_oh)
    d_pbe_x = sum(r["density_rmse_pbe"] for r in no_oh) / len(no_oh)
    scan_txt = ""
    if "ED_scan" in m:
        beats_scan = "beats SCAN" if m["beats_scan"] else "NO"
        scan_txt = (f" || SCAN: AE-MAE {m['E_MAE_scan']:.2f} densMAE {m['D_scan']:.3e} "
                    f"ED {m['ED_scan']:.2f} (NN {beats_scan})")
    print(f"{k[0]}/{k[1]}: AE-MAE NN {m['E_MAE_nn']:.2f} vs PBE {m['E_MAE_pbe']:.2f} kcal/mol | "
          f"densMAE NN {m['D_nn']:.3e} vs PBE {m['D_pbe']:.3e} "
          f"(excl. OH: {d_nn_x:.3e} vs {d_pbe_x:.3e}) | "
          f"ED NN {m['ED_nn']:.2f} vs PBE {m['ED_pbe']:.2f} "
          f"({'beats PBE' if m['beats_pbe'] else 'NO'}){scan_txt}")
""")

# ---------------------------------------------------------------------------
md(r"""
## 9. Held-out generalization -- does the tiny functional beat PBE off the training set?

The four molecules above were the *training* set. Here the already-trained models (no retraining) are
evaluated on systems they never saw: **N2** (closed-shell triple bond, a classic PBE density-error
case), **NO** (X-2-Pi degenerate radical), and **NO2** (bent doublet). All are real
`build_dfs_pool()` entries -- geometry, spin, and atomization energy come from the pool (no fabricated
values).

**NO is the acid test for the orientation lock:** it is a degenerate radical *outside* the training
set, so a reproducible, PBE-beating NO density shows the lock generalizes -- the CCSD reference and the
functional select the same representative of NO's degenerate 2-Pi manifold on any machine. Metrics
match sections 7-8: self-consistent density RMSE vs CCSD and own-atom atomization energy, NN vs PBE.
""")

code(r"""
# Evaluate the ALREADY-TRAINED models (no retraining) on held-out systems.
HELDOUT_HILLS = ("N2", "NO") if SMOKE else dfs_demo.HELDOUT_MOLECULE_HILLS
ho_points = dfs_demo.heldout_points(HELDOUT_HILLS)
ho_specs  = dfs_demo.build_mol_specs(ho_points, basis=BASIS, grid_level=GRID_LEVEL, refs_dir=REFS_DIR)

# Locked CCSD references for the held-out molecules (self-heals; NO's degenerate
# density is reproducible because ref + eval lock the same pi component).
dfs_demo.generate_ccsd_density_refs(ho_specs, refs_dir=REFS_DIR, basis=BASIS, grid_level=GRID_LEVEL)
ho_specs = dfs_demo.build_mol_specs(ho_points, basis=BASIS, grid_level=GRID_LEVEL, refs_dir=REFS_DIR)
ho_comp, ho_ae_ref = dfs_demo.heldout_comp_and_ae(ho_points, ho_specs)
print("held-out molecules:", [ms.name for ms in dfs_demo.molecule_specs(ho_specs)], flush=True)

# SCAN held-out baseline (model-independent): one SCAN SCF per held-out species,
# same lock/basis/grid as the CCSD refs, so NO's degenerate density is scored
# against the matching CCSD component -- the acid test that SCAN, like the NN,
# reproduces a held-out degenerate radical.
from xcquinox.alec.data import precompute_fixed_density_data
ho_scan_mol_data = {
    ms.name: precompute_fixed_density_data(
        ms, orientation_lock_strength=dfs_demo.ORIENTATION_LOCK_STRENGTH)
    for ms in dfs_demo.molecule_specs(ho_specs)
}
ho_scan_by_name = dfs_demo.scan_baseline(
    ho_specs, ho_scan_mol_data, refs_dir=REFS_DIR, basis=BASIS, grid_level=GRID_LEVEL)

HELDOUT_DIR = os.path.join(OUT_DIR, "heldout")
ho_evals, ho_combined = {}, {}
for key, info in trained.items():
    tag = f"{key[0]}__{key[1]}"
    ts = dfs_demo.build_heldout_test_spec(
        arch=info["spec"].arch, solver_cfg=solvers[key[1]], mol_specs=ho_specs,
        model_checkpoint=os.path.join(info["ckpt"], "model.eqx"),
        output_dir=os.path.join(HELDOUT_DIR, tag, "eval"))
    res = run_test(ts)
    res["per_molecule"] = dfs_demo.attach_scan_baseline(
        res["per_molecule"], ho_scan_by_name)
    ho_evals[key] = res
    arows = dfs_demo.self_consistent_ae(res["per_molecule"], ho_comp, ho_ae_ref)
    drows = dfs_demo.aggregate_density_diagnostics(res["per_molecule"])
    if arows and drows:
        ho_combined[tag] = dfs_demo.combined_energy_density(arows, drows)

summary = dfs_demo.heldout_summary(ho_combined)
N = summary["n_models"]
print(f"\nHeld-out generalization ({N} trained models on {HELDOUT_HILLS}):", flush=True)
print(f"  beat PBE on AE-MAE:  {summary['n_beat_ae']}/{N}")
print(f"  beat PBE on density: {summary['n_beat_density']}/{N}")
print(f"  beat PBE on ED:      {summary['n_beat_ed']}/{N}")
if "n_beat_ed_scan" in summary:
    print(f"  beat SCAN on AE-MAE:  {summary['n_beat_ae_scan']}/{N}")
    print(f"  beat SCAN on density: {summary['n_beat_density_scan']}/{N}")
    print(f"  beat SCAN on ED:      {summary['n_beat_ed_scan']}/{N}")
""")

code(r"""
# Figure: held-out generalization -- mean density RMSE (log) + AE-MAE, NN vs PBE
# vs SCAN, per model.
if ho_combined:
    tags = list(ho_combined)
    _has_scan = all("ED_scan" in ho_combined[t] for t in tags)
    xk = np.arange(len(tags)); w = 0.26 if _has_scan else 0.38
    def _pos(slot):  # slot in {-1,0,1} for PBE/SCAN/NN when scan present, else {-1,1}->{-.5,.5}
        return xk + slot * w if _has_scan else xk + slot * (w / 2)
    fig, (axD, axE) = plt.subplots(1, 2, figsize=(11, 4.5))
    axD.bar(_pos(-1), [ho_combined[t]["D_pbe"] for t in tags], w, label="PBE", color="0.6")
    if _has_scan:
        axD.bar(_pos(0), [ho_combined[t]["D_scan"] for t in tags], w, label="SCAN", color="0.35")
    axD.bar(_pos(1), [ho_combined[t]["D_nn"]  for t in tags], w, label="NN",  color="C2")
    axD.set_yscale("log"); axD.set_ylabel("held-out mean density RMSE vs CCSD")
    axD.set_title("density generalization"); axD.legend(fontsize=8)
    axE.bar(_pos(-1), [ho_combined[t]["E_MAE_pbe"] for t in tags], w, label="PBE", color="0.6")
    if _has_scan:
        axE.bar(_pos(0), [ho_combined[t]["E_MAE_scan"] for t in tags], w, label="SCAN", color="0.35")
    axE.bar(_pos(1), [ho_combined[t]["E_MAE_nn"]  for t in tags], w, label="NN",  color="C2")
    axE.set_ylabel("held-out AE-MAE (kcal/mol)"); axE.set_title("energy generalization"); axE.legend(fontsize=8)
    for ax in (axD, axE):
        ax.set_xticks(xk); ax.set_xticklabels([t.replace("__", "\n") for t in tags], fontsize=7)
    plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, "fig_heldout_generalization.png"), dpi=110); plt.show()
else:
    print("no held-out combined metrics (missing refs or eval); skipping figure", flush=True)

# Orientation-lock generalization check: NO is a degenerate 2-Pi radical NONE of the
# models trained on. Its PBE density RMSE is model-independent, so a spread across
# models would flag a non-reproducible (unlocked) NO density; with the lock on it is flat.
no_pbe = []
for key, res in ho_evals.items():
    for rec in res["per_molecule"]:
        if (rec.get("name") or rec.get("molecule")) == "NO" and rec.get("density_rmse_pbe") is not None:
            no_pbe.append(float(rec["density_rmse_pbe"]))
if len(no_pbe) > 1:
    spread = (max(no_pbe) - min(no_pbe)) / min(no_pbe)
    status = "reproducible (orientation lock holds)" if spread <= 0.02 else "VARYING -- lock may be off!"
    print(f"held-out NO PBE density RMSE across models: {min(no_pbe):.3e}..{max(no_pbe):.3e} -> {status}",
          flush=True)
""")

# ---------------------------------------------------------------------------
md(r"""
## 10. Notes

- Figure (b): self-consistent density RMSE vs CCSD per molecule (log scale), NN vs the PBE baseline.
  `full_25` reaches a more self-consistent fixed point than `full_3` (25 vs 3 differentiated cycles).
- Figure (c): atomization-energy error vs reference, from each functional's OWN self-consistent atom
  energies (the physically correct AE, matching `ae_as_reactions`) -- NOT the anchored `AE_nn` field
  (molecule energy minus FIXED exact atoms), which reports absolute-energy offset, not the AE.
- Figure (d): DFS combined energy-density error (Eq. 21), 3 panels (cf. DFS Fig. 2) -- energy AE-MAE,
  mean density RMSE, and the combined `ED`, NN vs PBE vs SCAN; `gamma` self-calibrated from PBE so
  `ED_pbe == E_MAE_pbe` and PBE/SCAN/NN share one kcal/mol scale. The mean density error is
  OH-radical-dominated (~2.6e-3 vs ~1e-4 elsewhere), so the aggregate density win is modest even
  though H2O/NH improve ~40% -- the printout's "excl. OH" mean makes this explicit.
- SCAN baseline (section 7): a meta-GGA self-consistent comparator to PBE, computed once per species
  (model-independent) via `run_scf_with_cache(xc="scan")` and scored identically (own-atom AE + CCSD
  density RMSE on the eval grid). Figures (b)/(c)/(d) and the section-9 held-out figure carry a SCAN
  series next to PBE. Whether a trained network beats SCAN -- the demanding, meta-GGA bar -- is
  reported as-is by the printouts and figures; it is NOT assumed.
- Section 9 (held-out): the trained models are evaluated on N2/NO/NO2 -- none in the training set --
  and the printout reports how many beat PBE on density, AE, and `ED` (`fig_heldout_generalization.png`).
  It also confirms the held-out degenerate NO radical's PBE density RMSE is model-independent, i.e. the
  orientation lock generalizes to an unseen 2-Pi system.
- To adapt: change `ARCH_NAMES` or pass a custom `ArchitectureConfig` to `build_dfs_training_spec`;
  extend `HILLS` with any `build_dfs_pool()` Hill formula. For the full pool + BH76/IP13 channels +
  V_xc supervision, use the cluster harness (`xcquinox.alec.cluster`) with the `dfs_step7` config; the
  CCSD reference generator there is `xcquinox.alec.external_refs.precompute_all` (adds the OEP V_xc
  cascade).
- Deviations from PRB L161109: CCSD (not CCSD(T)); `grid_level=2`; adamw + linear decay;
  spin-summed `N_e^2`. (The meta-GGA rung IS now included -- the `deep_mgga_*` nets.)
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
