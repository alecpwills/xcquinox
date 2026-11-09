"""Builder for notebooks/registry_pt_transfer.ipynb.

The notebook is the package tutorial AND the paper's miniature test case:
every architecture in the registry (44) is cloned by DFS-style pretraining
on the corpus atoms (H, Li, C, N, O, F), fine-tuned on the DFS-Letter
miniature (16 species: the atoms, the Li/C cations, H2O with its W4-11
atomization energy, the DFS barrier trio's reactants and transition
states) under the production Letter pathway -- 25 differentiable SCF
cycles per update, the Letter's loss weights, the per-update seed mixture,
relaxed CCSD(T) density references -- and read out on three proton-transfer
paths it never saw (the water-dimer cation hemibonded -> proton-transferred
pair, a Zundel H5O2+ scan, a neutral water-dimer scan), against the parent
functional of each architecture's footing on the same frozen densities and
the tracked CCSD(T) barrier from data/training_subsets/01wf.

Two routes are walked end to end on purpose: the harness CLI (yaml assembly,
submit dry-run, stage modules) on two demo architectures, and the library API
(run_pretrain / TrainingSpec / run_training / fidelity_certificate /
precompute_fixed_density_data / run_test) on the rest, both writing the same
<run_root>/pretrain/<arch> layout.

Sweep composition (recorded design change, 2026-11-03): no single campaign
model block can state the registry's gate spectrum -- deep_gea_3x16's
gea_mu='pbe' pins its gate to x2, every meta-GGA network pins its gate to
tanh2 (networks.py), and the block is run-wide. The campaign yaml therefore
sweeps the registry minus the GEA arm (43); the GEA arm trains in the
library route under its own block (false, paper, x2 -- production
arch_screen's form). Every model the notebook builds resolves through
apply_run_config (the harness stages' own resolution), so the two routes and
the readout share one identity per architecture and all 44 train.

Training-stage mechanism (DFS-parity upgrade, 2026-11-04): the training arc
replicates the production Letter pathway verbatim -- the solver block of
hpcjobs/configs/dfs_step8.v9_dfs_letter_allsc.yaml (FULL mode, 25 fixed
differentiable SCF cycles, REASSEMBLE features, the decaying linear mixer,
full-trajectory tail weights, seeded Fock noise, no convergence freeze), the
loss L6_letter_parity with the Letter's channel weights (reactions 1,
densities 20, atomic totals 0.01, no vxc), the per-update atomic/minao seed
mixture, and adam_plateau. The corpus is the DFS miniature: six atoms
(H, Li, C, N, O, F), the DFS ionization pair's cations (Li+, C+), H2O with
its W4-11 atomization energy as a predicted-atom reaction, the DFS barrier
trio's reactants and staged transition states, and RELAXED CCSD(T) density
references (pyscf's (T)-lambda + (T)-RDM modules) for every compound plus
the H and Li atoms. The FULL wall is projected from a measured steady
per-update cost and asserted under a 12 h ceiling BEFORE the sweep; a firing
ceiling is a stop for a user decision, never a silent knob.

The patched-record helper (patch_record / energy_on_dm) exists at readout
only, and always rebuilds J from a rebuilt Mole -- the stored j_matrix is
full-ERI J[dm_pbe] (data.py), and reusing it at a patched dm corrupts
E_non_xc by ~0.6 kcal/mol at a 1e-3 density perturbation.

Regenerating:  python notebooks/_build_registry_pt_transfer_notebook.py
The tracked .ipynb is the byte-exact product of this builder (no outputs,
deterministic cell ids cell_01..cell_36), enforced by
notebooks/test_notebook_hygiene.py.
"""
from __future__ import annotations

import inspect
import os

import nbformat as nbf


DEFAULT_WORK_ROOT = "notebooks/checkpoints_registry_pt"

#: the two architectures walked through the harness route; one GGA, one
#: meta-GGA, so both parent-density files the datagen builds are consumed
DEMO_ARCHS = ("deep_3x16", "deep_mgga_3x16")

#: smoke-tier slice: both demo architectures plus one further GGA family
SMOKE_ARCH_SUBSET = ("deep_3x16", "deep_mgga_3x16", "medium_attn")

H2O_COORDS = "O 0.0000 0.0000 0.1173; H 0.0000 0.7572 -0.4692; H 0.0000 -0.7572 -0.4692"


def md(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


# ---------------------------------------------------------------------------
# The campaign configuration (the unit test's entry point)
# ---------------------------------------------------------------------------


def make_grid_config_dict(work_root, archs=None, pretrain_steps=4000):
    """Assemble the campaign configuration as a plain dict.

    Every value below is stated, none inherited silently, so the returned
    dict is a complete description of the run it configures. Required keys
    follow cluster/grid_config.py: HyperParams requires n_steps, lr_start,
    lr_end, lr_decay_start, grad_clip, gradnorm_alpha, vxc_weight,
    density_weight; ClusterResources requires partition, time, mem,
    cpus_per_task, array_throttle, eval_array_throttle,
    max_concurrent_tasks; PretrainConfig.atoms takes FREE-ATOM PAIRS ONLY
    (molecules cannot be spelled in the yaml pretrain block -- H2O enters
    training through the library API instead).

    ``archs=None`` sweeps every registered architecture minus deep_gea_3x16
    (43, registry order -- the GEA arm's gate cannot share a run-wide model
    block with the meta rungs; see the exclusion note at the arch list); an
    explicit sequence is the smoke-tier slice (the exclusion applies to it
    too when the arm is present).

    ``pretrain_steps`` defaults to the production 4000; the notebook passes
    its tier's ``pretrain_steps_harness`` here so the campaign artifact
    carries the tier actually being run (the harness stages read the step
    count from the resolved config, so the knob must reach the dict -- a
    display-only tier table would contradict the run's own metadata).

    Identity: def2-svp, grid level 1, no density fitting, the calibrated
    3.0e-5 orientation lock, the polarized correlation network; model block
    (parent_anchor false, descriptor_coordinates paper, gate unstated -- the
    tanh2 default is a no-op on every swept architecture).

    Two waivers are stated with reasons, as the validators require:

    * inputs.allow_irreproducible_degenerate -- the 3P terms of the O and C
      atoms are not resolved by a grid-1 quadrature, and pretrain_data_gen's
      guard refuses the build at grid_level < 3 REGARDLESS of the orientation
      lock (``if not (coarse or unlocked): return False``); the waiver is
      recorded in the datagen sidecar manifests.
    * fidelity.enforce=false -- the certificate is still computed and
      written; only its gate is released. A miniature-corpus pretrain
      cannot reproduce the parent functional to the binding tolerances on
      the full oracle, so FAIL is the expected recorded verdict. A MISSING
      certificate is waived by nothing: the train gate refuses it.

    cluster.mail_user / mail_type stay blank on purpose: no script rendered
    from this configuration is ever queued (submit runs in its default
    dry-run; the stage modules are invoked directly), so there is no
    submission, completion or failure to report, and a blank mail block
    keeps a local-tutorial configuration from being mistaken for a
    production one.
    """
    import xcquinox.pipeline as pipeline

    if archs is None:
        arch_list = list(pipeline.ARCHITECTURES.keys())
    else:
        unknown = [a for a in archs if a not in pipeline.ARCHITECTURES]
        if unknown:
            raise ValueError(f"archs not in the registry: {unknown}")
        arch_list = list(archs)
    # One campaign model block cannot state the registry's gate spectrum:
    # deep_gea_3x16's gea_mu='pbe' pins its ueg gate to x2 (ArchitectureConfig
    # invariant) while every meta-GGA network pins its gate to tanh2
    # (networks.py), and the block is run-wide (ModelConfig defaults are
    # concrete strings, so even an absent block forces tanh2). Production
    # splits campaigns the same way -- arch_screen.yaml (x2) sweeps the GEA
    # arm, families_mgga (no gate key) sweeps the meta arms. The GEA arm
    # leaves the campaign here and joins in the notebook under its own block;
    # the notebook still trains all 44 architectures.
    arch_list = [a for a in arch_list if a != "deep_gea_3x16"]
    return {
        "sweep": {
            "arch": arch_list,
            "loss": ["L6_letter_parity"],
            "metric": ["jsd"],
            "subset_size": [1],
            "solver": ["full_25"],
        },
        "solvers": {
            # the v9 dfs_letter_allsc solver block verbatim (the production
            # Letter pathway; the notebook's library-route training builds
            # the identical SolverConfig)
            "full_25": {
                "mode": "FULL",
                "max_cycles": 25,
                "feature_policy": "REASSEMBLE",
                "scf_grad_checkpoint": True,
                "mixer_name": "decaying_linear",
                "mixer_kwargs": {"base": 0.3, "floor": 0.3,
                                 "step_offset": 1},
                "scf_loss_use_tail": True,
                "scf_loss_tail": 25,
                "scf_loss_weight_power": 2.0,
                "freeze_on_convergence": False,
                "fock_noise_std": 1.0e-5,
                "fock_noise_seed": 0,
            },
        },
        "hyperparams": {
            # the v9 dfs_letter_allsc hyperparameter set (n_steps stated at
            # the FULL value; the executed epochs are the library tier knob)
            "n_steps": 60,
            "lr_start": 1.0e-4,
            "lr_end": 1.0e-7,
            "lr_decay_start": 0.5,
            "grad_clip": 1.0,
            "gradnorm_alpha": 1.5,
            "vxc_weight": 0.01,
            "density_weight": 0.1,
            "density_per_electron": True,
            "weight_decay": 1.0e-6,
            "val_frac": 0.2,
            "validate_every": 0,
            "patience": 8,
            "early_stop_min_delta": 0.1,
            "checkpoint_every": 10,
            "pbe_anchor_weight": 0.0,
            "require_atom_anchors": False,
            "seed": 42,
            "update_scheme": "per_molecule",
            "pad_group_to_common_shape": True,
            "optimizer": "adam_plateau",
            "plateau_patience": 10,
            "plateau_factor": 0.1,
            "seed_mix_atomic": True,
            "respect_sc_flag": False,
            "nonsc_weight": 0.01,
            "validation_density_weight": 20.0,
            "channel_weights": {"loss_AE": 1.0, "loss_BH76": 1.0,
                                "loss_IP13": 1.0, "loss_rho": 20.0,
                                "loss_vxc": 0.0, "loss_Eatom": 0.01},
        },
        "inputs": {
            "external_refs_dir": f"{work_root}/external_refs",
            "subset_ledger_path": f"{work_root}/subset_index_log.json",
            "basis": "def2-svp",
            "grid_level": 1,
            # output_root is where the loader CREATES its runs/ directory;
            # naming the work root itself keeps the run dir at
            # <work>/runs/run_XXX (naming <work>/runs doubles it)
            "output_root": f"{work_root}",
            "density_fit": False,
            "orientation_lock_strength": 3.0e-5,
            # run-wide pbe seeding, a stated deviation from v9's per-rung
            # auto (SCAN seeds for the meta arms): auto on this footing
            # would trigger an uncached SCAN seed generation
            "seed_xc": "pbe",
            "allow_irreproducible_degenerate": True,
            "irreproducible_degenerate_reason": (
                "registry-pt miniature at grid level 1: the 3P terms of the "
                "O and C atoms are unresolved by this quadrature and the "
                "guard refuses the build at any lock; two draws of those "
                "rows differ, the waiver is recorded in the datagen sidecar "
                "manifests"
            ),
        },
        "pretrain": {
            "data_dir": f"{work_root}/pretrain_data",
            "n_steps": pretrain_steps,
            "lr_start": 1.0e-3,
            "lr_end": 1.0e-5,
            "lr_decay_start": 0.5,
            "lr_decay_end": 0.9,
            "grad_clip": 0.0,
            "seed": 42,
            "loss_weighting": "rho_w_sampled",
            # the six corpus elements' free-atom pairs (spin = 2S, the
            # production convention; v9's own pretrain atom set minus Na,
            # which this miniature's corpus never uses)
            "atoms": {"H": 1, "Li": 1, "C": 2, "N": 3, "O": 2, "F": 1},
            "dfs_set": False,
            "pool_atoms": False,
            "parent_density": "auto",
            "exchange_footing": "paper",
            "mesh_fraction": 0.3,
            "energy_term_weight": 0.0,
            "points_per_system": 800,
            "sampling_seed": 42,
            "validation_fraction": 0.0,
            "validation_seed": 0,
            "validate_every": 4000,
            "patience": 0,
        },
        "cluster": {
            "partition": "",
            "time": "00:30:00",
            "mem": "4G",
            "cpus_per_task": 4,
            "array_throttle": 1,
            "eval_array_throttle": 1,
            "max_concurrent_tasks": 2,
            "max_array_size": 1000,
            "device": "cpu",
            "conda_env": "",
            "conda_profile": "",
            "mail_user": "",
            "mail_type": "NONE",
            "account": "",
            "eval_workers": 1,
            "preflight_time": "00:30:00",
            "eval_time": "00:30:00",
            "pretrain_time": "00:30:00",
        },
        "domain_profile": "dfs_step7",
        "on_precompute_failure": "abort",
        "bh76_mode": "barrier_height",
        "ae_as_reactions": True,
        "use_polarized_correlation": True,
        "inline_eval": False,
        "defer_eval": False,
        "model": {
            # Unanchored, paper coordinates (gga_final / v9 letter's block
            # minus its gate). The gate is deliberately UNSTATED: the tanh2
            # default is a no-op on every swept architecture (all 43 carry
            # tanh2 in the registry), while stating x2 -- the letter's value
            # -- would be refused by every meta-GGA network at pretrain time
            # (networks.py: the meta rung's gate already carries x2).
            "parent_anchor": False,
            "descriptor_coordinates": "paper",
        },
        "fidelity": {
            "enforce": False,
            "override_reason": (
                "registry-pt miniature: 44-architecture tutorial corpus "
                "(H, O, H2O-CCSD, W4-11 AE); the certificate is computed and "
                "recorded, the gate is released because a miniature pretrain "
                "cannot meet the full-oracle tolerances"
            ),
        },
    }


def parallel_worker_cap(n_tasks, *, n_cpus=None, total_gib=None,
                        per_worker_gib):
    """Parallel worker count bounded by memory as well as CPUs.

    The 2026-11-03 full-tier run on a 20-CPU / 30 GiB host lost a training
    worker to the OOM killer at the fifth wave: cpu_count()//2 = 10 workers
    stacked more live children than the RAM held once several large
    attention architectures shared a wave (the killed child was at 4.77 GiB
    and still growing). The cap is

        min(n_tasks, n_cpus // 2, int(0.8 * total_gib / per_worker_gib))

    with per_worker_gib measured per stage: 3.0 GiB per pretrain worker
    (about 2 GiB observed each), and for training workers the budget is the
    SCF child's measured peak RSS plus headroom (the 2026-11-03 oneshot run
    measured 4.77 GiB growing at a 10-worker wave; the SCF-in-the-loop
    upgrade re-measures it at the smoke tier -- the training cell's
    TRAIN_WORKER_GIB carries the measured value). The 0.8 reserve leaves
    the OS and the parent kernel their share.
    """
    if n_cpus is None:
        n_cpus = os.cpu_count() or 2
    if total_gib is None:
        total_gib = (os.sysconf("SC_PAGE_SIZE")
                     * os.sysconf("SC_PHYS_PAGES") / 1024 ** 3)
    cpu_cap = max(1, n_cpus // 2)
    mem_cap = max(1, int(0.8 * total_gib / per_worker_gib))
    return max(1, min(n_tasks, cpu_cap, mem_cap))


# ---------------------------------------------------------------------------
# Cells
# ---------------------------------------------------------------------------


def build_cell_01_intro():
    source = """# Registry-wide miniature training and proton-transfer transfer

This notebook is two things at once.

1. **The package tutorial.** Every stage of the workflow is walked end to end:
   the campaign yaml (assembled key by key, both waivers explained), the
   `xcquinox.pipeline.cluster` harness CLI (`submit` in its default dry-run,
   then the stage modules invoked directly), and the library API
   (`run_pretrain`, `TrainingSpec` / `run_training`, `fidelity_certificate`,
   `precompute_fixed_density_data`, `run_test`).
2. **The paper's test case.** All 44 registered architectures are cloned by
   DFS-style pretraining, then fine-tuned on the DFS-Letter miniature corpus
   under the production Letter pathway -- 25 differentiable SCF cycles per
   update, the Letter's loss weights (reactions 1, densities 20, atomic
   totals 0.01, no vxc), the per-update seed mixture, CCSD(T) density
   references -- and read out on three proton-transfer paths they never saw.
   The question: does the published optimization pathway, run on a miniature
   of its own corpus, transfer to a proton-transfer barrier?

Three arcs, in order:

| arc | route | architectures |
|---|---|---|
| harness tutorial | yaml -> submit dry-run -> `_datagen` -> `_pretrain` -> `fidelity` | 2 demo archs (one GGA, one meta-GGA) |
| full-registry training | library API (`run_pretrain`, `run_training`) | the remaining 42 (the GEA arm among them, under its own model block) |
| transfer readout | frozen-CCSD(T)-density evaluation + parent baselines | every arch (all 44), 3 paths |

Budget: the training arc runs self-consistent SCF in the loop, so the FULL
tier is an overnight-scale run by design -- a measurement cell times the
steady per-update wall first, projects the whole sweep from it, and refuses
to start past a 12 h ceiling (the projection and the ceiling are printed
before the sweep); the SMOKE tier (next cell) completes in minutes.

Artifacts land under one run root: the materialized run directory from the
submit dry-run, per-arch pretrain checkpoints + fidelity certificates, per-arch
trained models, the CCSD reference cache, and `results/barriers.csv`.
"""
    return md(source)


def build_cell_02_map():
    source = """## The workflow map: harness route and library route

Two routes through the same package, deliberately both shown.

**The harness route** (2 demo architectures) is what a cluster campaign runs:
a campaign yaml is validated and resolved by `submit` (dry-run by default; it
creates the run directory, `resolved_config.yaml` and the rendered scripts),
then each stage module -- `_datagen`, `_pretrain`, `fidelity` -- is invoked on
that run directory, exactly as the array jobs would on a node. Nothing is
queued here: `--partition local` + dry-run keeps everything on this machine.

**The library route** (the remaining architectures) is what the Python API
runs: `PretrainSpec` + `run_pretrain`, then `TrainingSpec` + `run_training`,
with the fidelity certificate written by the same `fidelity_certificate`
function the harness stage calls.

Why the split: the yaml pretrain block takes **free-atom pairs only**
(`PretrainConfig.atoms`; a molecule cannot be spelled there), so every
corpus molecule -- H2O, the barrier reactants, the transition states, the
cations -- enters training through the library route's `MoleculeSpec` list.
Both routes write the same `<run_root>/pretrain/<arch>/` layout, and
`run_training` accepts either's checkpoints -- the certificate gate checks
the directory, not who wrote it.

Everything downstream (training, transfer readout, results) consumes the union:
per-arch pretrain checkpoints under `<run_root>/pretrain/<arch>/`, trained
models under `<run_root>/train/<arch>/`.
"""
    return md(source)


def build_cell_03_tier(pretrain_steps_harness, pretrain_steps_library,
                       train_epochs, pt_scan_points, arch_subset, work_root):
    arch_expr = "None  # every registered architecture (44)"
    if arch_subset is not None:
        arch_expr = repr(tuple(arch_subset)) + "  # smoke-tier slice"
    source = f"""import os

# Tier switch. The committed notebook runs the FULL tier; setting STEP_SMOKE=1
# in the environment (as the end-to-end test does) shrinks every knob to a
# minutes-scale wiring check of the same code paths.
STEP_SMOKE = bool(os.environ.get("STEP_SMOKE"))
TIERS = {{
    "full": {{
        "pretrain_steps_harness": {pretrain_steps_harness},
        "pretrain_steps_library": {pretrain_steps_library},
        "train_epochs": {train_epochs},
        "pt_scan_points": {pt_scan_points},
    }},
    "smoke": {{
        "pretrain_steps_harness": 25,
        "pretrain_steps_library": 5,
        # 2 epochs, not 1: the cost-measurement cell differences two runs
        # to remove the first-epoch compile, which needs a second epoch
        # to difference against
        "train_epochs": 2,
        "pt_scan_points": 2,
    }},
}}
TIER = TIERS["smoke" if STEP_SMOKE else "full"]

# The sweep slice. None = every registered architecture. The smoke slice used
# by the end-to-end test is baked in at build time (it must contain both demo
# architectures so the harness route is exercised); the FULL notebook ships
# with None.
SWEEP_ARCH_NAMES = {arch_expr}

WORK_ROOT = {work_root!r}
BASIS = "def2-svp"
GRID_LEVEL = 1
ORIENTATION_LOCK = 3.0e-5
DEMO_ARCHS = ("deep_3x16", "deep_mgga_3x16")
# The corpus H2O geometry (experimental gas-phase; the builder's module
# constant, baked here so the cells that read it are self-contained).
H2O_COORDS = {H2O_COORDS!r}

os.makedirs(WORK_ROOT, exist_ok=True)
os.makedirs(os.path.join(WORK_ROOT, "external_refs"), exist_ok=True)
os.makedirs(os.path.join(WORK_ROOT, "results"), exist_ok=True)

TIMINGS = {{}}  # every stage records (label, seconds) here; the close-out lists them

print(f"tier={{'smoke' if STEP_SMOKE else 'full'}}")
print(f"knobs: {{TIER}}")
print(f"sweep: {{'all registered' if SWEEP_ARCH_NAMES is None else len(SWEEP_ARCH_NAMES)}} archs, demos {{DEMO_ARCHS}}")
print(f"work root: {{os.path.abspath(WORK_ROOT)}}")
"""
    return code(source)


def build_cell_04_imports():
    source = """import json
import os
import pickle
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

import jax
# JAX config: pin x64 dtype and CPU device *before* importing jnp or any
# library that may trigger JAX tracing. These must not change later in the
# notebook -- flipping jax_enable_x64 after traces are cached produces
# inconsistent dtypes.
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_device", jax.devices("cpu")[0])
import jax.numpy as jnp

from pyscf import gto, dft, scf, cc

import xcquinox.pipeline
import xcquinox.pipeline as pipeline
from xcquinox.pipeline.checkpoint_class import load_trained_checkpoint
from xcquinox.pipeline.config import apply_run_config
from xcquinox.pipeline.solver import (
    FeaturePolicy, SolverConfig, SolverMode, _reassemble_features)
from xcquinox.pipeline.oneshot import fixed_density_total_energy
from xcquinox.pipeline.models import AlecGGAModel
from xcquinox.pipeline.evaluation import run_test
from xcquinox.pipeline.balancing import GradNormConfig
from xcquinox.pipeline.orientation_lock import orientation_lock_bias
from xcquinox.pipeline.cluster.grid_config import (
    ModelConfig,
    load_grid_config,
    validate_grid_semantics,
    require_explicit_bh76_mode,
    load_resolved_run_config,
    pretrain_checkpoint_dir,
    pretrain_stage_archs,
)
from xcquinox.pipeline.cluster.domain import (
    KCAL_PER_HA,
    ATOMIC_ENERGIES_CHAKRAVORTY,
    get_domain_profile,
)
from xcquinox.pipeline.cluster import fidelity as fidelity_lib
from xcquinox.pipeline.full_benchmark_pools import W411_JSON_PATH
from xcquinox.pipeline.subset_selection import _make_hb_atoms, _make_pt_atoms
from xcquinox.pipeline.external_refs import (
    _require_electron_count, _t1_diagnostic)

# tqdm.auto picks tqdm.notebook.tqdm (ipywidgets) under JupyterLab and
# tqdm.std.tqdm in a plain script/terminal, so the same symbol gives a
# sensible progress bar in either context.
from tqdm.auto import tqdm

print(f"jax x64: {{jax.config.jax_enable_x64}}, devices: {{jax.devices()}}")
print(f"registry: {{len(pipeline.ARCHITECTURES)}} architectures")
"""
    return code(source)


def build_cell_05_registry_md():
    source = """## Arc B -- the registry and the campaign yaml

The architecture registry (`pipeline.ARCHITECTURES`, `config.py`) is the
package's single source of truth for what can be trained: 44 entries, 36
run-2 GGA functionals and 8 meta-GGA ones. The `meta_gga` flag decides each
architecture's parent density (`parent_density: auto` below builds a PBE file
for a GGA arm and a SCAN file for a meta-GGA arm, so one datagen serves both),
and the descriptors decide which per-molecule record blocks the readout needs.

One registry entry cannot join the campaign below: `deep_gea_3x16` (the GEA
arm) pins its uniform-gas gate to `x2` (`gea_mu='pbe'` fixes the s^2
coefficient under that gate only), every meta-GGA network pins its gate to
`tanh2` (the meta rung's gate already carries x2), and a campaign model block
is run-wide. Production runs the same split as two campaigns
(`arch_screen.yaml` with `ueg_gate: x2` for the GEA arm; the mgga families
config with no gate key). This notebook keeps all 44 in training: the 43 go
through the campaign, and the GEA arm joins in the library route under its
own block -- production `arch_screen`'s form -- resolved everywhere through
the package's own `apply_run_config`.

The next cell pins the registry census the way the test suite does (44
architectures, the GGA/meta split, both demo architectures present), then
binds `ARCH_NAMES` -- the sweep this notebook trains (all 44; the campaign
yaml carries these minus the GEA arm).
"""
    return md(source)


def build_cell_06_registry():
    source = """ARCH_NAMES = list(SWEEP_ARCH_NAMES) if SWEEP_ARCH_NAMES is not None \\
    else list(pipeline.ARCHITECTURES.keys())

META_NAMES = [n for n, cfg in pipeline.ARCHITECTURES.items() if cfg.meta_gga]
GGA_NAMES = [n for n in pipeline.ARCHITECTURES if n not in set(META_NAMES)]
GEA_ARCH = "deep_gea_3x16"  # the one arm whose gate cannot share the campaign

# Registry census, pinned the way xcquinox/pipeline/tests/test_config.py does.
assert len(pipeline.ARCHITECTURES) == 44, len(pipeline.ARCHITECTURES)
assert len(META_NAMES) == 8 and len(GGA_NAMES) == 36
assert set(DEMO_ARCHS) <= set(ARCH_NAMES), "demo archs must be inside the sweep"

print(f"{'arch_name':<26} {'family':<6} {'depth':>5} {'nodes':>5} {'attn':>5}  descriptors")
print("-" * 88)
for _name in ARCH_NAMES:
    _cfg = pipeline.get_architecture(_name)
    _fam = "meta" if _cfg.meta_gga else "gga"
    _descs = ", ".join(d.name for d in _cfg.descriptors) or "-"
    print(f"{_name:<26} {_fam:<6} {_cfg.depth:>5} {_cfg.nodes:>5} "
          f"{str(bool(_cfg.attention)):>5}  {_descs}")
print(f"\\ntrained set: {len(ARCH_NAMES)} archs "
      f"({sum(1 for n in ARCH_NAMES if n in set(META_NAMES))} meta-GGA)")
print(f"campaign sweep: {len(ARCH_NAMES) - (1 if GEA_ARCH in ARCH_NAMES else 0)} "
      f"archs (the GEA arm trains in the library route under its own block)")
"""
    return code(source)


def build_cell_07_yaml_md():
    source = """### The campaign yaml, key by key

The dict assembled in the next cell is the same artifact a cluster campaign
submits -- dumped to `campaign.yaml`, loaded back through the same validators
`submit` runs (`load_grid_config`, `validate_grid_semantics`,
`require_explicit_bh76_mode`). The blocks:

* **sweep** -- the grid axes: every architecture x one loss
  (`L6_letter_parity`, the Letter-parity loss -- the only loss carrying the
  `loss_Eatom` channel, the Letter's lambda_E = 0.01 absolute atomic-total
  term) x the JSD metric x subset size 1 (H2O; the one-point ledger below is
  written to match) x the `full_25` solver.
* **solvers.full_25** -- the v9 dfs_letter_allsc solver block verbatim: FULL
  mode (self-consistent SCF in the loop), 25 fixed differentiable cycles,
  REASSEMBLE features, gradient checkpointing of the scan body, the DFS
  decaying linear mixer (`alpha = 0.3^(step+1) + 0.3`; the executed-script
  first mix 0.6), the full-trajectory tail weights (`scf_loss_tail: 25` =
  max_cycles keeps all 25 steps at weight `(i / 24)^2`), no convergence
  freeze (every cycle runs, exactly the published protocol), and the seeded
  symmetrized Fock degeneracy noise at 1e-5. This IS the production
  optimization pathway; the library training below builds the identical
  `SolverConfig`.
* **hyperparams** -- the v9 dfs_letter_allsc train-stage set: adam_plateau
  (constant lr_start, plateau factor 0.1 after 10 stagnant epochs, floored
  at lr_end -- the linear decay schedule is inert under this optimizer,
  stated verbatim anyway), coupled L2 weight decay 1e-6, the per-update
  atomic/minao seed mixture (`seed_mix_atomic`), every point self-consistent
  (`respect_sc_flag: false`), per_molecule updates with padded groups (one
  compile per spin type), and the Letter's channel weights spelled in full
  (`loss_vxc: 0`, `loss_Eatom: 0.01`). `validate_every: 0` keeps the
  in-loop validation slice out (a stated miniature deviation; v9 validates
  against a manifest set this corpus lacks).
* **inputs** -- the identity: def2-svp, grid level 1, no density fitting (a
  stated deviation: v9's own inputs run density fitting at
  6-311++G(3df,2pd) / grid 3), run-wide `seed_xc: pbe` (v9 resolves per-rung
  `auto`; a stated deviation -- auto here would trigger an uncached SCAN
  seed generation), the calibrated 3.0e-5 orientation lock (authoritative
  for the whole run: the training SCF, the CCSD(T) references and the
  pretraining rows all lock the same degenerate component of an open shell).
  `external_refs_dir` is where the corpus cell writes its CCSD(T) `.npz`
  references; `subset_ledger_path` is the one-point subset ledger written
  beside the yaml.
* **pretrain** -- the cloning protocol: 4000 steps (FULL tier) at
  lr 1e-3 -> 1e-5 (decay window 0.5 -> 0.9), `rho_w_sampled` weighting, 800
  points per system, `parent_density: auto` (PBE file for GGA arms, SCAN file
  for meta-GGA arms), `exchange_footing: paper`, mesh fraction 0.3, on the
  six corpus elements' free-atom pairs -- free atoms only, which is why every
  molecule (H2O, the barrier species, the transition states) enters through
  the library route. `dfs_set` / `pool_atoms` are off: the pretrain set is
  exactly the corpus atoms. (The exchange footing and mesh fraction shape the
  DATA BUILD the datagen performs; the library `PretrainSpec` below reads the
  file they produced.)
* **cluster** -- SLURM fields are rendered but never submitted (submit runs
  in its default dry-run; the stage modules are invoked directly).
  `mail_user`/`mail_type` stay blank: no script from this configuration is
  ever queued, so there is no submission, completion or failure to report,
  and a blank mail block keeps a local-tutorial config from being mistaken
  for a production one.
* **domain_profile / bh76_mode / model** -- the dfs_step7 profile; an
  explicit `bh76_mode` (`barrier_height`) because the validator refuses a
  config that leaves it to the default; the model block states
  `parent_anchor: false` and `descriptor_coordinates: paper` (the letter
  arms' form) and deliberately leaves `ueg_gate` unstated -- the tanh2
  default is a no-op on every swept architecture, while stating `x2` (the
  letter's value) would be refused by every meta-GGA network at pretrain
  time: the meta rung's gate already carries x2 (`networks.py`). The GEA
  arm, whose `gea_mu='pbe'` pins its gate to x2, is out of this campaign
  for exactly that reason (see the registry cell) and joins in the library
  route under its own block; the polarized correlation network
  (`use_polarized_correlation: true`), which is why datagen writes
  `pretrain_data_polarized*.npz`.

**The two waivers** (both must carry a non-empty reason, or the validators
refuse the config):

1. `inputs.allow_irreproducible_degenerate: true` -- the O atom's 3P term is
   not resolved by a grid-1 quadrature, and the datagen guard refuses the
   build at `grid_level < 3` *regardless of the orientation lock*
   (`pretrain_data_gen.py`: `coarse = grid_level < 3; ... if not (coarse or
   unlocked): return False`). Two independent draws of those rows differ;
   the waiver says so and is recorded in the sidecar manifests beside the
   data files. This was checked empirically for exactly this corpus and
   identity: both the PBE and the SCAN polarized files build in seconds.
2. `fidelity.enforce: false` -- the certificate is still **computed and
   written**; only its gate is released. A miniature-corpus pretrain cannot
   reproduce the parent functional to the binding tolerances on the full
   oracle, so FAIL is the expected recorded verdict. A *missing* certificate
   is waived by nothing: `run_training`'s gate refuses it. There is no
   environment-variable bypass.
"""
    return md(source)


def build_cell_08_yaml_code():
    src = inspect.getsource(make_grid_config_dict)
    source = src + f"""

# The self-generated one-point ledger: subset-selection's schema with H2O as
# the single size-1 point (jsd/1 = H2O). The harness consumes ledgers of this
# shape from the subset-selection pre-process; writing our own keeps the whole
# run inside the work root.
{json_dump_snippet()}

# Assemble at the runtime sweep, dump, and validate exactly as submit does.
# The tier's harness-pretrain step count flows into the campaign here: the
# harness stages read it from the resolved config, so a knob that only
# reached the tier-table printout would leave the run's own metadata
# (pretrain_steps_run) contradicting what the notebook printed.
campaign = make_grid_config_dict(WORK_ROOT, archs=SWEEP_ARCH_NAMES,
                                 pretrain_steps=TIER["pretrain_steps_harness"])
assert campaign["pretrain"]["n_steps"] == TIER["pretrain_steps_harness"], (
    "the tier's pretrain step count did not reach the campaign artifact")
_expected_sweep = tuple(a for a in ARCH_NAMES if a != GEA_ARCH)
assert tuple(campaign["sweep"]["arch"]) == _expected_sweep, (
    "the campaign carries the trained set minus the GEA arm")
assert campaign["sweep"]["loss"] == ["L6_letter_parity"]
_full25 = campaign["solvers"]["full_25"]
assert _full25["mode"] == "FULL" and _full25["max_cycles"] == 25
assert _full25["mixer_kwargs"] == {{"base": 0.3, "floor": 0.3,
                                   "step_offset": 1}}
assert _full25["scf_loss_tail"] == 25 and _full25["fock_noise_std"] == 1.0e-5
assert set(campaign["hyperparams"]["channel_weights"]) == {{
    "loss_AE", "loss_BH76", "loss_IP13", "loss_rho", "loss_vxc", "loss_Eatom"}}
assert campaign["inputs"]["allow_irreproducible_degenerate"] is True
assert campaign["inputs"]["irreproducible_degenerate_reason"]
assert campaign["fidelity"]["enforce"] is False
assert campaign["fidelity"]["override_reason"]

campaign_path = os.path.join(WORK_ROOT, "campaign.yaml")
with open(campaign_path, "w") as _f:
    yaml_dump(campaign, _f)

cfg = load_grid_config(campaign_path)
validate_grid_semantics(cfg, get_domain_profile(cfg.domain_profile))
require_explicit_bh76_mode(campaign_path)
assert cfg.sweep.arch == _expected_sweep, "sweep order must follow the registry"
print(f"campaign validated: {{len(cfg.sweep.arch)}} archs x {{cfg.sweep.loss}} "
      f"(trained set {{len(ARCH_NAMES)}}: the GEA arm trains under its own block)")
print(f"run identity: {{cfg.inputs.basis}} / grid {{cfg.inputs.grid_level}} / "
      f"lock {{cfg.inputs.orientation_lock_strength}}")
"""
    return code(source)


def json_dump_snippet():
    return """import yaml as _yaml_mod
def yaml_dump(obj, fh):
    _yaml_mod.safe_dump(obj, fh, sort_keys=False)

_ledger = {
    "jsd/1": {
        "chosen_indices": [0],
        "metric_value": 0.0,
        "point_kinds": ["ae"],
        "point_names": ["h2o"],
        "tag": "bin01",
    }
}
with open(os.path.join(WORK_ROOT, "subset_index_log.json"), "w") as _f:
    json.dump(_ledger, _f, indent=1)
"""


def build_cell_09_submit_md():
    source = """## Arc C -- the harness route

`python -m xcquinox.pipeline.cluster submit <yaml>` is the campaign front
door. Two behaviors matter here:

* it is a **dry-run by default**: `--submit` is the opt-in that queues
  anything. Without it, `submit` validates the yaml, creates a fresh run
  directory under `inputs.output_root`, resolves the sweep into it
  (`resolved_config.yaml`) and renders the array scripts (`scripts/`) plus
  the command list (`submit_commands.txt`) -- everything a submission would
  need, minus the queue.
* `--partition` is required (here `local`: the scripts are rendered, never
  submitted), and `--stages pretrain` limits the materialized pipeline to the
  stages this arc runs (datagen + pretrain; the certificate stage is invoked
  directly below).

The printed line `submit: run dir = <path>` is the run root every later cell
shares.
"""
    return md(source)


def build_cell_10_submit():
    source = """# A fresh work root runs the submit dry-run (validating the yaml and
# materializing the run directory); a work root that already holds one
# REUSES it -- every heavy artifact below is cached under the run dir and
# skipped on a rerun, so an interrupted run continues where it stopped
# instead of starting over in a new run_XXX.
_existing = sorted(
    Path(os.path.join(WORK_ROOT, "runs")).glob("run_*")
) if os.path.isdir(os.path.join(WORK_ROOT, "runs")) else []

if _existing:
    RUN_DIR = str(_existing[-1])
    print(f"reusing run dir: {RUN_DIR} (a rerun continues where it stopped)")
else:
    _t0 = time.time()
    _proc = subprocess.run(
        [sys.executable, "-m", "xcquinox.pipeline.cluster", "submit",
         os.path.join(WORK_ROOT, "campaign.yaml"),
         "--partition", "local", "--stages", "pretrain"],
        capture_output=True, text=True, check=True,
    )
    TIMINGS["submit (dry-run)"] = time.time() - _t0
    print(_proc.stdout[-2000:])

    _out = _proc.stdout + _proc.stderr
    _run_dirs = [ln.split("run dir = ", 1)[1].strip()
                 for ln in _out.splitlines()
                 if "run dir = " in ln]
    assert _run_dirs, "submit did not report a run dir"
    RUN_DIR = _run_dirs[-1]

assert os.path.isfile(os.path.join(RUN_DIR, "resolved_config.yaml"))
assert os.path.isdir(os.path.join(RUN_DIR, "scripts"))
assert os.path.isfile(os.path.join(RUN_DIR, "submit_commands.txt"))
print(f"\\nrun dir: {RUN_DIR}")
print("  resolved_config.yaml, scripts/, submit_commands.txt present")
"""
    return code(source)


def build_cell_11_datagen_md():
    source = """### Stage 1: `_datagen`

The datagen stage builds the pretraining data from the **resolved** run
config: one polarized file per parent density the sweep needs
(`parent_density: auto` -> `pretrain_data_polarized.npz` for the PBE footing
and `pretrain_data_polarized_scan.npz` for the SCAN footing), every descriptor
column the registry's architectures consume, plus the (r_s, s, alpha) mesh. A
sidecar `<file>.npz.manifest.json` records each file's build identity --
including the degeneracy waiver this identity runs under.

Invoking `xcquinox.pipeline.cluster._datagen.main([run_dir])` is exactly what
the datagen array job does on a node -- the same entry point, the same
resolved config, the same manifest.
"""
    return md(source)


def build_cell_12_datagen():
    source = """from xcquinox.pipeline.cluster import _datagen

_t0 = time.time()
_rc = _datagen.main([RUN_DIR])
TIMINGS["datagen"] = time.time() - _t0
assert _rc in (0, None), _rc

_pretrain_data_dir = cfg.pretrain.data_dir
_data_files = sorted(f for f in os.listdir(_pretrain_data_dir) if f.endswith(".npz"))
print(f"pretrain data files: {_data_files}")
assert any("polarized" in f for f in _data_files)

# The sidecar manifests record the degeneracy waiver this identity runs under.
_manifests = sorted(Path(_pretrain_data_dir).glob("*.manifest.json"))
assert _manifests, "no sidecar manifests beside the pretrain data"
_waiver_seen = False
for _m in _manifests:
    _txt = _m.read_text()
    if "irreproducible" in _txt or "degenerate" in _txt:
        _waiver_seen = True
assert _waiver_seen, "the recorded waiver should appear in a sidecar manifest"
print(f"{len(_manifests)} sidecar manifest(s); the degeneracy waiver is recorded")
"""
    return code(source)


def build_cell_13_pretrain_stage():
    source = """from xcquinox.pipeline.cluster import _pretrain

# The array indices the harness would use: pretrain_stage_archs(cfg) is the
# canonical sorted distinct-architecture list the pretrain array sizes itself
# with (donors removed -- none here), and every stage indexes into it.
_stage_archs = pretrain_stage_archs(cfg)

_t0 = time.time()
for _arch in tqdm(DEMO_ARCHS, desc="harness pretrain", dynamic_ncols=True):
    _idx = _stage_archs.index(_arch)
    _rc = _pretrain.main([RUN_DIR, str(_idx)])
    assert _rc in (0, None), (_arch, _rc)
    # The certificate stage on the same index: computed and written, verdict
    # recorded (FAIL is expected under the waiver), gate released.
    _rc = fidelity_lib.main([RUN_DIR, str(_idx)])
    assert _rc in (0, None), (_arch, _rc)
TIMINGS["harness pretrain+cert (2 demos)"] = time.time() - _t0

for _arch in DEMO_ARCHS:
    _ck = pretrain_checkpoint_dir(RUN_DIR, _arch)
    assert os.path.isfile(os.path.join(_ck, "xnet.eqx")), _ck
    _cert = os.path.join(_ck, "fidelity_certificate.json")
    assert os.path.isfile(_cert), _cert
    with open(_cert) as _f:
        _payload = json.load(_f)
    print(f"{_arch}: certificate written "
          f"(verdict {_payload.get('verdict', _payload.get('result', 'recorded'))})")
"""
    return code(source)


def build_cell_14_library_md():
    source = """### Arc D -- the library route: pretraining the remaining architectures

The harness route is now demonstrated end to end on the two demo
architectures. The remaining architectures -- including the GEA arm, whose
gate keeps it out of the campaign yaml -- go through the library API, which
is the same machinery with the config objects built in Python.

One rule makes the two routes interchangeable: **every model the notebook
builds resolves its architecture through `apply_run_config`** (the registry
entry + the run's polarization override + the model block) -- exactly what
the harness stages do internally. The GEA arm resolves under its own block
(`x2`), everyone else under the campaign's. A checkpoint written by either
route is then the same model class the readout deserialises.

`PretrainSpec` states the cloning protocol. The knobs below are the
production GGA-final cloning values: lr 1e-3 -> 1e-5 with the decay window
opening at 0.5 and closing at 0.9, no gradient clipping, seed 42,
`rho_w_sampled` point weighting with 800 points per system (sampling seed
42), zero energy term. The DATA the spec reads was shaped by the yaml block
above (`data_dir` points at the same datagen output the harness stage
consumed: `paper` exchange footing, mesh fraction 0.3, both parent files),
and `parent_density: "auto"` makes `run_pretrain` resolve the parent per
architecture -- PBE rows for a GGA arm, SCAN rows for a meta-GGA arm. The
step count is the tier knob (FULL: the library value, SMOKE: 5).

`checkpoint_dir` points at the same `<run_root>/pretrain/<arch>` layout, so
the certificate gate in the training stage cannot tell the routes apart.
"""
    return md(source)


def build_cell_15_pretrain_specs():
    source = """_LIBRARY_ARCHS = [a for a in ARCH_NAMES if a not in set(DEMO_ARCHS)]

# THE IDENTITY every model in this notebook is built at: the registry entry
# with the run's polarization override and model block applied -- the same
# apply_run_config the harness stages resolve through, so the two routes and
# the readout cannot disagree about what a checkpoint holds. The GEA arm
# resolves under its own block (x2, production arch_screen's form): a
# run-wide block cannot state its gate alongside the meta rungs' tanh2, which
# is why it is out of the campaign sweep but not out of the training set.
cfg_gea = replace(cfg, model=ModelConfig(parent_anchor=False,
                                         descriptor_coordinates="paper",
                                         ueg_gate="x2"))

def resolve_arch(name):
    return apply_run_config(pipeline.get_architecture(name),
                            cfg_gea if name == GEA_ARCH else cfg)

def make_pretrain_spec(arch_name, n_steps):
    return pipeline.PretrainSpec(
        arch=resolve_arch(arch_name),
        data_dir=cfg.pretrain.data_dir,
        checkpoint_dir=pretrain_checkpoint_dir(RUN_DIR, arch_name),
        n_steps=n_steps,
        lr_start=1.0e-3,
        lr_end=1.0e-5,
        lr_decay_start=0.5,
        lr_decay_end=0.9,
        grad_clip=0.0,
        seed=42,
        loss_weighting="rho_w_sampled",
        parent_density="auto",
        energy_term_weight=0.0,
        points_per_system=800,
        sampling_seed=42,
    )

PRETRAIN_STEPS_LIBRARY = TIER["pretrain_steps_library"]
print(f"library archs: {len(_LIBRARY_ARCHS)} at {PRETRAIN_STEPS_LIBRARY} steps "
      f"(harness demos ran {TIER['pretrain_steps_harness']} steps)")
print(f"identity: registry + polarized + paper coords"
      f"{' + x2 gate (own block)' if GEA_ARCH in ARCH_NAMES else ''}, "
      f"resolved via apply_run_config")
"""
    return code(source)


def build_cell_16_pretrain_run():
    # the worker cap's source is embedded verbatim (inspect.getsource), the
    # same single-source-of-truth move the campaign cell makes with
    # make_grid_config_dict: the notebook defines it once, here, and the
    # training executor below reuses the kernel-namespace definition
    source = inspect.getsource(parallel_worker_cap) + """

# One architecture runs in-process with the per-step callback (the tutorial
# view; note the two phases -- X and C networks -- each with its own bar);
# the rest run as parallel subprocesses, each pinned to one thread so N
# workers never oversubscribe XLA's internal thread pool.
_bars = {}
def _cb(info):
    key = (info.get("arch", "pretrain"), info.get("phase", "train"))
    if key not in _bars:
        _bars[key] = tqdm(total=info["total"], desc="/".join(key),
                          leave=False, dynamic_ncols=True)
    bar = _bars[key]
    _delta = info["step"] - bar.n
    if _delta > 0:
        bar.update(_delta)
    bar.set_postfix(loss=f"{info['loss']:.4e}")
    if info["step"] >= info["total"]:
        bar.close()
        del _bars[key]

_skip = lambda a: os.path.isfile(
    os.path.join(pretrain_checkpoint_dir(RUN_DIR, a), "xnet.eqx"))

_todo = [a for a in _LIBRARY_ARCHS if not _skip(a)]
_skipped = len(_LIBRARY_ARCHS) - len(_todo)
if _skipped:
    print(f"skipping {_skipped} cached arch(s) (xnet.eqx present)")

if _todo:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    _spec_dir = os.path.join(WORK_ROOT, "pretrain_specs")
    os.makedirs(_spec_dir, exist_ok=True)
    for _a in _todo:
        with open(os.path.join(_spec_dir, f"{_a}.pkl"), "wb") as _f:
            pickle.dump(make_pretrain_spec(_a, PRETRAIN_STEPS_LIBRARY), _f)

    _t0 = time.time()
    # tutorial arm, in-process
    _first = _todo[0]
    pipeline.run_pretrain(make_pretrain_spec(_first, PRETRAIN_STEPS_LIBRARY),
                          progress_callback=_cb)
    TIMINGS["library pretrain (first, in-process)"] = time.time() - _t0

    # the rest: the pickled spec goes to a worker subprocess, the same
    # hand-off the cluster's train array performs (single-quoted block: the
    # notebook source itself lives inside a triple-double-quoted string)
    _child_code = \'\'\'
import pickle, sys
import xcquinox.pipeline as pipeline
with open(sys.argv[1], "rb") as f:
    spec = pickle.load(f)
pipeline.run_pretrain(spec)
\'\'\'

    def _pretrain_one(arch_name):
        child_env = dict(os.environ)
        child_env["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"
        child_env["OMP_NUM_THREADS"] = "1"
        _p = subprocess.run(
            [sys.executable, "-c", _child_code,
             os.path.join(_spec_dir, f"{arch_name}.pkl")],
            env=child_env, capture_output=True, text=True)
        if _p.returncode != 0:
            raise RuntimeError(f"[{arch_name}] pretrain failed:\\n"
                               f"{_p.stdout[-800:]}\\n{_p.stderr[-2000:]}")
        return arch_name

    _rest = _todo[1:]
    if _rest:
        _max_workers = parallel_worker_cap(len(_rest), per_worker_gib=3.0)
        print(f"pretraining {len(_rest)} archs in parallel: "
              f"{_max_workers} workers (cpu- and memory-capped; "
              f"~3 GiB budgeted per pretrain worker)")
        _bar = tqdm(total=len(_rest), desc="pretrain (parallel)",
                    dynamic_ncols=True)
        _t0 = time.time()
        try:
            with ThreadPoolExecutor(max_workers=_max_workers) as _ex:
                _futs = {_ex.submit(_pretrain_one, a): a for a in _rest}
                for _fut in as_completed(_futs):
                    _bar.update(1)
                    _bar.set_postfix(arch=_fut.result())
        finally:
            _bar.close()
        TIMINGS["library pretrain (parallel)"] = time.time() - _t0

for _a in _LIBRARY_ARCHS:
    assert os.path.isfile(
        os.path.join(pretrain_checkpoint_dir(RUN_DIR, _a), "xnet.eqx")), _a
print(f"all {len(_LIBRARY_ARCHS)} library pretrains present")
"""
    return code(source)


def build_cell_17_pretrain_timing():
    source = """print("pretrain timing:")
for _k, _v in TIMINGS.items():
    if "pretrain" in _k or "datagen" in _k or "submit" in _k:
        print(f"  {_k}: {_v:.1f} s")

print("\\n(all pretrain timings above are measurements of THIS run, not estimates;")
print(" the close-out cell compares the total against the stated budget)")
"""
    return code(source)


def build_cell_18_corpus_md():
    source = """## Arc E -- the corpus: the DFS miniature at 16 species

The training corpus is the published training set in miniature -- its three
reaction channels and every element it touches, at one compound instead of
twenty-one:

* **six neutral atoms** H, Li, C, N, O, F (spins from the package table) --
  the pretraining footing and the `loss_Eatom` anchors (the Letter's
  lambda_E = 0.01 absolute atomic-total term);
* **the DFS ionization pair's cations** Li+ (singlet) and C+ (doublet) --
  energy-only records; the IP channel trains `E_nn[cation] - E_nn[neutral]`
  against the pair's NIST reference, and L6 excludes cations from both the
  anchor groups and the density records;
* **H2O** -- the W4-11 atomization energy, staged as a *predicted-atom
  reaction* (`E_nn[2H + O] - E_nn[h2o]` against the AE target): under the
  per-molecule update scheme a compound trains its AE only through its own
  group, and a compound-alone group holds no atom records -- the reaction
  form (which pulls the atoms into the group) is the only way the AE trains.
  H2O is therefore aux-excluded from the AE channel and trains through its
  reaction group, exactly how production stages `ae_as_reactions`;
* **the DFS barrier trio's reactants** OH, N2, CH3, HF (geometries from the
  tracked G2/97 trajectory the pool builder itself reads; spins/charges from
  the reaction table's species maps) **and its staged transition states**
  n2ohts, RKT11, hf2ts (geometries, spins and charges from the tracked BH76
  benchmark pool). In barrier mode the trained quantity is
  `E(TS) - sum(E_reactants)`: the chemical products (H, N2O, O, CH4, F2) are
  never staged -- production's barrier staging replaces the product side
  with the transition state, and so does the loss entry below.

Two names are load-bearing. **OH must be declared a radical explicitly**
(`radical_species=("HO",)`): production derives the radical set from
atomization-point metadata, and OH enters this corpus only through barrier
points, which carry no radical flag -- undeclared, its density would train
at 2000x the production weight. **h_species is (H, Li)**, the Dick
regularizer set: L6's density records are every compound PLUS the neutral
h_species atoms, so H and Li carry CCSD(T) density targets while O, C, N, F
stay energy-anchor-only.

The density references are **relaxed CCSD(T)** 1-RDMs: converged canonical
(locked) HF -> CCSD/UCCSD -> the (T)-lambda equations -> the (T)-relaxed
RDM (pyscf's opt-in `ccsd_t_lambda` / `ccsd_t_rdm` modules; the make_rdm1
call includes the HF occupancy, and the UHF variant returns the spin pair).
The npz mechanism is the package's own: `MoleculeSpec.external_data_path`
whitelists `dm_target` / `rho_ref_grid` onto the record, and the density
channel trains the model's *self-consistent* density against them at the
Letter's weight 20.

Units and anchors (the loss takes Hartree; a kcal-scale reference is refused
by the 10-Ha sanity guard at loss construction):

* every reference is computed from the package tables at build time --
  `DFS_BH76_REACTIONS` (barrier refs), `DFS_IP13_PAIRS` (NIST IPs),
  `W411_JSON_PATH` (the H2O AE), each divided by `KCAL_PER_HA`;
* the atom anchors are `ATOMIC_ENERGIES_CHAKRAVORTY` for all six elements
  (the production convention), and the AE convention
  `AE = sum_Z n_Z * atom_energies[Z] - E_mol` pins E(H2O) at the
  reference-consistent total.
"""
    return md(source)


def build_cell_18b_staging():
    source = '''# The 16-species corpus, staged from the package tables. Three swappable
# lists (the contingency's requirement: the fallback corpus -- H2O/O
# vertical IPs, the H2O dissociation barrier -- is a list edit here and
# nowhere else): the atom/cation set, the barrier trio (consumed by the
# training cell straight from DFS_BH76_REACTIONS), and the IP pair.
from ase.io import read as _ase_read
from xcquinox.pipeline.dfs_pool import (
    ATOMIC_GROUND_STATE_SPIN, DFS_BH76_REACTIONS, DFS_IP13_PAIRS,
    _g297_traj_path, make_atom_atoms)

CORPUS_ATOM_SYMS = ("H", "Li", "C", "N", "O", "F")
CORPUS_TS_NAMES = ("n2ohts", "RKT11", "hf2ts")

# spin/charge of the trio's molecular reactants from the reaction table's
# own species maps (HO 1, N2 0, CH3 1, HF 0 -- derived, never typed)
_mol_spec = {}
for _r in DFS_BH76_REACTIONS:
    for _n in (*_r["reactants"], *_r["products"]):
        if _n in ("HO", "N2", "CH3", "HF"):
            _mol_spec.setdefault(
                _n, (int(_r["species_charges"].get(_n, 0)),
                     int(_r["species_spins"][_n])))
assert set(_mol_spec) == {"HO", "N2", "CH3", "HF"}, sorted(_mol_spec)

# geometries: the tracked G2/97 trajectory the pool builder itself resolves
# through (keyed by ASE Hill formula, the same map build_dfs_pool_points
# builds; these four formulas are unambiguous in the trajectory)
_g297 = _ase_read(str(_g297_traj_path()), ":")
_traj = {a.get_chemical_formula(): a for a in _g297}
assert set(_traj) >= set(_mol_spec), "the G2/97 trajectory lacks a reactant"

# the staged transition states from the tracked BH76 benchmark pool
# (geometry string, spin, charge and composition all from the json entry)
with open(Path(pipeline.__file__).resolve().parent
          / "data" / "bh76_full_pool.json") as _f:
    ts_entries = {e["name"]: e for e in json.load(_f)["species"]
                  if e["name"] in CORPUS_TS_NAMES}
assert set(ts_entries) == set(CORPUS_TS_NAMES), sorted(ts_entries)

def _spec_from_atoms(a, name, charge, spin, external=None):
    syms, pos = list(a.symbols), np.asarray(a.positions)
    atom = "; ".join(f"{s} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}"
                     for s, p in zip(syms, pos))
    comp = {}
    for s in syms:
        comp[s] = comp.get(s, 0) + 1
    return pipeline.MoleculeSpec(
        name=name, atom=atom, basis=BASIS, charge=int(charge),
        spin=int(spin), atom_composition=tuple(sorted(comp.items())),
        external_data_path=external, grid_level=GRID_LEVEL)

def _spec_from_str(atom, name, charge, spin, external=None):
    # composition parsed from the geometry string itself, so the census
    # checks the geometry against the json entry rather than echoing it
    syms = [p.split()[0] for p in atom.split(";") if p.strip()]
    comp = {}
    for s in syms:
        comp[s] = comp.get(s, 0) + 1
    return pipeline.MoleculeSpec(
        name=name, atom=atom, basis=BASIS, charge=int(charge),
        spin=int(spin), atom_composition=tuple(sorted(comp.items())),
        external_data_path=external, grid_level=GRID_LEVEL)

def _npz(name):
    return os.path.join(ext_data_dir, f"{name}.npz")

ext_data_dir = os.path.join(WORK_ROOT, "external_refs")
os.makedirs(ext_data_dir, exist_ok=True)

corpus_specs = (
    # list 1: the six neutral atoms, spins from the package table
    *[_spec_from_atoms(make_atom_atoms(_s), _s, 0,
                       ATOMIC_GROUND_STATE_SPIN[_s], external=_npz(_s))
      for _s in CORPUS_ATOM_SYMS],
    # list 2: the DFS IP pair's cations (energy-only records)
    *[_spec_from_atoms(
        make_atom_atoms(_p["cation"].rstrip("+"),
                        charge=int(_p["cation_charge"]),
                        spin=int(_p["cation_spin"])),
        _p["cation"], int(_p["cation_charge"]), int(_p["cation_spin"]),
        external=_npz(_p["cation"]))
      for _p in DFS_IP13_PAIRS],
    # the AE compound (the builder's experimental-geometry constant)
    _spec_from_str(H2O_COORDS, "h2o", 0, 0, external=_npz("h2o")),
    # list 3: the barrier trio's molecular reactants
    *[_spec_from_atoms(_traj[_n], _n, _mol_spec[_n][0], _mol_spec[_n][1],
                       external=_npz(_n))
      for _n in sorted(_mol_spec)],
    # the staged transition states
    *[_spec_from_str(_e["atom"], _n, int(_e["charge"]), int(_e["spin"]),
                     external=_npz(_n))
      for _n, _e in sorted(ts_entries.items())],
)

# the census: 16 species, every (name, charge, spin, composition) derived
# above from the package tables
assert len(corpus_specs) == 16, len(corpus_specs)
assert len({s.name for s in corpus_specs}) == 16, "species names collide"
print(f"corpus: {len(corpus_specs)} species -- "
      f"{len(CORPUS_ATOM_SYMS)} atoms + 2 cations + h2o + "
      f"{len(_mol_spec)} reactants + {len(ts_entries)} transition states")
for _s in corpus_specs:
    print(f"  {_s.name:<7} q={_s.charge:+d} 2S={_s.spin}  "
          f"{dict(_s.atom_composition)}")
'''
    return code(source)


def build_cell_19_ccsd_refs():
    source = '''# References at the run identity. The W4-11 AE target and the Chakravorty
# anchors come from the package; the density references are RELAXED CCSD(T)
# 1-RDMs for the ten rho-target species (L6's rho-record rule: every
# compound plus the neutral h_species atoms H and Li; the O/C/N/F atoms and
# both cations stay energy-only).
with open(W411_JSON_PATH) as _f:
    _w411 = json.load(_f)
AE_W411_KCAL = float(next(r for r in _w411["reactions"]
                          if r["name"] == "w411_h2o_atomization")
                     ["reaction_energy_ref"])
AE_HA = AE_W411_KCAL / KCAL_PER_HA
# kcal left unconverted would put a 232.97 Ha "atomization energy" into the
# AE channel; the guard below fires on exactly that class of mistake.
assert 232.0 < AE_W411_KCAL < 234.0, AE_W411_KCAL
assert 0.3 < AE_HA < 0.4, AE_HA

ATOM_ANCHORS = {s: float(ATOMIC_ENERGIES_CHAKRAVORTY[s])
                for s in ("H", "Li", "C", "N", "O", "F")}
# The H2O total-energy target implied by the anchor convention.
E_REF_H2O = 2 * ATOM_ANCHORS["H"] + ATOM_ANCHORS["O"] - AE_HA

RHO_TARGET_NAMES = ("H", "Li", "h2o", "HO", "N2", "HF", "CH3",
                    "n2ohts", "RKT11", "hf2ts")
ENERGY_ONLY_NAMES = ("O", "C", "N", "F", "Li+", "C+")

print(f"W4-11 AE(H2O) = {AE_W411_KCAL:.3f} kcal/mol = {AE_HA:.8f} Ha")
print(f"atom anchors (Chakravorty): {ATOM_ANCHORS}")

def _ccsd_t_relaxed(mol, tag):
    """Converged canonical (locked) HF -> CCSD/UCCSD -> the (T)-lambda
    equations -> the RELAXED CCSD(T) 1-RDM (pyscf's opt-in ccsd_t_lambda /
    ccsd_t_rdm modules; with_mf, so the HF occupancy is included; the UHF
    variant returns the spin pair). The orientation lock biases the HF core
    exactly as the package's own CCSD reference generation does, so the
    reference and the functional's SCF lock the same degenerate component.

    Returns (dm_relaxed, dm_total, dm_ccsd_total, E_ccsd_t, t1): dm_relaxed
    keeps the shape the record's dm_pbe carries (the per-spin pair for open
    shells -- data.py refuses a dm_target of any other shape), and the grid
    projection below contracts the spin-summed dm_total.
    """
    from pyscf.cc import ccsd_t_lambda, ccsd_t_rdm
    from pyscf.cc import uccsd_t_lambda, uccsd_t_rdm
    from xcquinox.pipeline.external_refs import (
        _build_hf_meanfield, _converge_scf_tiered)
    # the package's own reference path: tiered HF convergence (plain DIIS
    # -> SOSCF/newton -> ...) with the orientation lock applied
    # idempotently on every tier, so the reference and the functional's
    # SCF lock the same degenerate component
    _is_uhf = mol.spin != 0
    _locked = None
    if ORIENTATION_LOCK:
        _bare = _build_hf_meanfield(mol, _is_uhf)
        _locked = (np.asarray(_bare.get_hcore())
                   + orientation_lock_bias(mol, ORIENTATION_LOCK))
    mf_hf = _converge_scf_tiered(
        lambda: _build_hf_meanfield(mol, _is_uhf),
        dm0=None, is_uks=_is_uhf, locked_hcore=_locked)
    assert mf_hf.converged, tag
    mycc = cc.UCCSD(mf_hf) if mol.spin else cc.CCSD(mf_hf)
    mycc.kernel()
    if not mycc.converged:
        # a stalled amplitude iteration (measured on the RKT11 triplet TS:
        # a tight limit cycle at fixed e_corr that the plain restart, damped
        # amplitudes and a looser tolerance all fail to break) -- a FRESH
        # solve with a wide DIIS space from the MP2 start escapes it; a
        # restart from the stalled amplitudes does not, even under the
        # wide DIIS (measured, smoke run 4)
        mycc = cc.UCCSD(mf_hf) if mol.spin else cc.CCSD(mf_hf)
        mycc.diis_space = 20
        mycc.max_cycle = 300
        mycc.kernel()
    assert mycc.converged, tag
    eris = mycc.ao2mo()
    _lam = uccsd_t_lambda if mol.spin else ccsd_t_lambda
    _rdm = uccsd_t_rdm if mol.spin else ccsd_t_rdm
    # The (T)-lambda solve reads its DIIS space off the CC object, so it is
    # set explicitly here. At pyscf's default space of 6 the iteration crawls
    # or stalls on open-shell points (measured at def2-svp: cation_pt 462
    # cycles; RKT11 not converged in 2000), while a space of 20 converges
    # every measured species in <= 72 cycles (cation_pt 28, RKT11 36, n2ohts
    # 30, hf2ts 72). Where both spaces converge, the relaxed 1-RDMs agree to
    # within 6e-8 (hf2ts 7e-9, n2ohts 1e-8, cation_pt 5e-8). RKT11 previously
    # converged only because the CCSD restart above builds a fresh object
    # with the wide space. max_cycle 300 remains the bound.
    mycc.diis_space = 20
    conv, l1, l2 = _lam.kernel(mycc, eris, mycc.t1, mycc.t2,
                                max_cycle=300)
    assert conv, (tag, "the (T)-lambda solve did not converge")
    dm_t = np.asarray(_rdm.make_rdm1(mycc, mycc.t1, mycc.t2, l1, l2,
                                     eris, ao_repr=True))
    # AO representation throughout: every consumer contracts the DM against
    # AO products, and pyscf's make_rdm1 defaults to MO space --
    # shape-compatible garbage in those contractions (external_refs.py makes
    # the same call, with the same electron-count refusal used below).
    dm_ccsd = np.asarray(mycc.make_rdm1(ao_repr=True))
    if mol.spin:
        dm_store, dm_total = dm_t, dm_t[0] + dm_t[1]
        dm_ccsd_total = dm_ccsd[0] + dm_ccsd[1]
    else:
        dm_store, dm_total, dm_ccsd_total = dm_t, dm_t, dm_ccsd
    E_ccsd_t = float(mf_hf.e_tot + mycc.e_corr + mycc.ccsd_t())
    # the package's T1 diagnostic (Lee-Taylor; for UCCSD the spin-orbital
    # norm normalized to the restricted value at closed shell)
    return dm_store, dm_total, dm_ccsd_total, E_ccsd_t, _t1_diagnostic(mycc)


_t0 = time.time()
for _spec in tqdm(corpus_specs, desc="corpus CCSD(T) refs", dynamic_ncols=True):
    _path = os.path.join(ext_data_dir, f"{_spec.name}.npz")
    # cached like every other artifact: a resumed run reuses the reference
    # npz on disk (delete a file to recompute that species)
    if os.path.isfile(_path):
        continue
    if _spec.name in ENERGY_ONLY_NAMES:
        # energy-only records: the atom anchors from the package table (the
        # cations record their neutral element's anchor -- an inert
        # placeholder; cations are neither compounds nor density targets)
        np.savez(_path, E_ref_literature=float(
            ATOM_ANCHORS[_spec.name.rstrip("+")]))
        continue
    assert _spec.name in RHO_TARGET_NAMES, _spec.name
    mol = gto.M(atom=_spec.atom, basis=BASIS, charge=_spec.charge,
                spin=_spec.spin, verbose=0)
    dm_store, dm_total, dm_ccsd_total, E_ccsd_t, _t1 = _ccsd_t_relaxed(
        mol, _spec.name)
    _require_electron_count(dm_store, mol, _path)
    # the (T)-vs-CCSD density delta: small and nonzero on every correlated
    # species (the mutation killer for a driver that silently returns the
    # CCSD density). One-electron species (H) are the physical exception --
    # CCSD is already exact there, the (T) correction vanishes and the delta
    # is exactly zero, recorded rather than asserted.
    _delta = float(np.max(np.abs(dm_total - dm_ccsd_total)))
    if mol.nelectron > 1:
        assert _delta > 0.0, (_spec.name, _delta)
    # grid projection on the record's own quadrature: two independent grid
    # builds disagree at the same level (pyscf prunes during kernel(); the
    # package pins its own pruning/cutoffs), so the projection consumes a
    # probe record's ao_grid/grid_weights rather than a hand-built grid
    _probe = pipeline.precompute_fixed_density_data(
        pipeline.MoleculeSpec(
            name=f"{_spec.name}_probe", atom=_spec.atom, basis=BASIS,
            charge=_spec.charge, spin=_spec.spin,
            atom_composition=_spec.atom_composition,
            grid_level=GRID_LEVEL),
        orientation_lock_strength=ORIENTATION_LOCK)
    _ao = np.asarray(_probe["ao_grid"])
    _w = np.asarray(_probe["grid_weights"])
    _rho = np.einsum("ij,pi,pj->p", dm_total, _ao, _ao)
    # the projected density integrates to the electron count on this
    # quadrature (loose 1%: grid level 1); the exact trace check above
    # catches a basis mix-up, this one catches a projection slip
    assert abs(float(np.dot(_rho, _w))
               - mol.nelectron) / mol.nelectron < 0.01, _spec.name
    _e_ref = {"H": ATOM_ANCHORS["H"], "Li": ATOM_ANCHORS["Li"],
              "h2o": float(E_REF_H2O)}.get(_spec.name, E_ccsd_t)
    np.savez(
        _path,
        dm_target=dm_store,
        rho_ref_grid=_rho,
        ref_density_method="ccsd(t)",
        E_ref_literature=float(_e_ref),
        grid_level_used=GRID_LEVEL,
        basis_used=BASIS,
        grid_weights=_w,
        t1_diagnostic=_t1,
        ccsd_converged=True,
        density_fit_used=False,
        orientation_lock_strength=ORIENTATION_LOCK,
    )
    print(f"  {_spec.name:<7} E_ccsd(T) = {E_ccsd_t:.6f} Ha  "
          f"(T)-delta = {_delta:.2e}  t1 = {_t1:.3f}")
TIMINGS["corpus CCSD(T) refs"] = time.time() - _t0
'''
    return code(source)


def build_cell_20_records():
    source = """# The union of per-descriptor record keys across the sweep: one record per
# species then serves every architecture's readout regardless of which
# descriptors it consumes (a bare precompute would leave e.g. cusp_features
# None and break descriptor architectures downstream). ARCHITECTURES stores
# FeatureSpec name+kwargs pairs; materialize_descriptors() (config.py) turns
# them into the live Descriptor objects, injecting the arch-level cusp
# log_transform (uniform across the registry: one kwargs variant per name).
# Keys are per descriptor CLASS (data.py), so the union is deduped by class
# -- and each key is guarded to a single (class, width) signature: a shared
# record cannot serve the sweep if two archs wrote one key at two widths.
_ALL_BY_CLASS = {}
_KEY_SIG = {}
for _a in ARCH_NAMES:
    for _d in resolve_arch(_a).materialize_descriptors():
        _ALL_BY_CLASS.setdefault(type(_d).__name__, _d)
        for _k in _d.required_mol_keys:
            _sig = (type(_d).__name__, _d.n_features)
            _prev = _KEY_SIG.setdefault(_k, _sig)
            assert _prev == _sig, (
                f"record key {_k!r} is written at two signatures "
                f"({_prev} vs {_sig}); one shared record could not "
                "serve every architecture")
_ALL_DESCRIPTORS = tuple(_ALL_BY_CLASS.values())
_REQUIRED_KEYS = set(_KEY_SIG)
print(f"record keys requested by the sweep's descriptors: {sorted(_REQUIRED_KEYS)}")
print(f"distinct descriptor classes: {len(_ALL_DESCRIPTORS)}")

# One record per corpus species (the staging cell's specs carry the
# external_data_path wiring). with_minao_seed: the seed mixture requires
# dm_minao on EVERY molecule of every group, and the group union covers all
# 16 species (train._require_minao_seed refuses the first epoch without it).
_t0 = time.time()
corpus_records = {}
for _spec in tqdm(corpus_specs, desc="precompute corpus", dynamic_ncols=True):
    corpus_records[_spec.name] = pipeline.precompute_fixed_density_data(
        _spec,
        required_keys=tuple(sorted(_REQUIRED_KEYS)),
        descriptors=_ALL_DESCRIPTORS,
        orientation_lock_strength=ORIENTATION_LOCK,
        with_minao_seed=True,
    )
TIMINGS["corpus precompute"] = time.time() - _t0

# The load-bearing asserts: the CCSD(T) reference actually reached the
# records the training loss will see, at the record's own dm_pbe shape (the
# per-spin pair for the seven open-shell targets -- data.py refuses any
# other shape at load time; asserted here so a driver slip dies at the source)
for _n in RHO_TARGET_NAMES:
    _rec = corpus_records[_n]
    assert _rec["dm_target"] is not None, (_n, "record carries no dm_target")
    assert _rec["rho_ref_grid"] is not None, _n
    assert _rec["ref_density_method"] == "ccsd(t)", _n
    assert (np.asarray(_rec["dm_target"]).shape
            == np.asarray(_rec["dm_pbe"]).shape), _n
    assert _rec["dm_minao"] is not None, (_n, "no minao seed on the record")
for _n in ENERGY_ONLY_NAMES:
    assert corpus_records[_n].get("dm_target") is None, _n
print("rho-target records: ccsd(t) dm_target at dm_pbe shape + minao seed; "
      "energy-only records carry no density target")
_h2o_rec = corpus_records["h2o"]
print(f"H2O E_pbe = {float(_h2o_rec['E_pbe']):.6f} Ha, "
      f"E_non_xc = {float(_h2o_rec['E_non_xc']):.6f} Ha")
"""
    return code(source)


def build_cell_21_certificates():
    source = """# Fidelity certificates for every architecture in the trained set, library
# route: the same fidelity_certificate the harness stage calls, with the
# oracle narrowed to the corpus (H, O, H2O) so each certificate is a
# measurement on the systems this run actually trains. The full 15-atom
# oracle is the production instrument; at this tier it would dominate the
# runtime for nothing the miniature can act on. Under the waiver the verdict
# is recorded, not enforced; a missing certificate would still block
# training. fidelity_certificate resolves the arch through the run config
# itself (apply_run_config), so the GEA arm is certified under its own
# block -- the same per-arch cfg rule every model build follows.
_cfg_resolved = load_resolved_run_config(RUN_DIR)
_full_oracle = fidelity_lib.build_oracle_set(_cfg_resolved, ARCH_NAMES[0])
# oracle atoms are named atom_H / atom_O / ...; keep whatever oracle members
# the corpus actually stages (atoms by symbol, molecules by name; both
# sides lowercased -- the corpus names are mixed case)
_corpus_names = {s.name.lower() for s in corpus_specs}
_corpus_oracle = tuple(
    ms for ms in _full_oracle
    if ms.name.lower().removeprefix("atom_") in _corpus_names)
assert len(_corpus_oracle) >= 3, [m.name for m in _full_oracle][:12]

_t0 = time.time()
for _arch in tqdm(ARCH_NAMES, desc="certificates", dynamic_ncols=True):
    _ck = pretrain_checkpoint_dir(RUN_DIR, _arch)
    if os.path.isfile(os.path.join(_ck, "fidelity_certificate.json")):
        continue  # the two harness demos already carry theirs
    fidelity_lib.fidelity_certificate(
        _cfg_resolved if _arch != GEA_ARCH else cfg_gea,
        RUN_DIR, _arch, oracle_set=_corpus_oracle)
TIMINGS["certificates"] = time.time() - _t0

for _arch in ARCH_NAMES:
    assert os.path.isfile(os.path.join(
        pretrain_checkpoint_dir(RUN_DIR, _arch), "fidelity_certificate.json")), _arch
print(f"certificates present for all {len(ARCH_NAMES)} archs")
"""
    return code(source)


def build_cell_22_deviations_md():
    source = """### Deviations from the Letter and from v9 (stated here, restated at the close)

The PATHWAY is production-verbatim (solver block, loss, weights, optimizer,
seed mixture, noise). What differs is the scale of the corpus and the
identity it runs at:

* **Corpus: 16 species vs the published 26 points.** One atomization
  compound (H2O) instead of twenty-one; the Letter's own barrier trio and
  ionization pair, staged from the same tracked tables production reads.
  The radical set is therefore `{HO}` where the pool's radical hills are
  `CH, HO` (CH is not in this corpus) -- stated explicitly because the
  per-update staging derives radicals from atomization metadata this
  corpus's OH never carries.
* **Identity: def2-svp / grid level 1 / no density fitting**, where the v9
  letter arms run 6-311++G(3df,2pd) / grid 3 / density fitting -- the
  workstation-footprint miniature. The 3P terms of O and C at grid 1 carry
  the recorded degeneracy waiver.
* **Tail weights over the full 25-step trajectory** (`w_i = (i / 24)^2`
  over every cycle), not the paper's last-10 window (its `j = 10..25`
  schedule): the package's tail machinery cannot express a skipped early
  window without changing every other campaign -- a recorded, analyzed
  difference (reports_local/dfs_training_review_2026-06-22.md).
* **Seed mixture beta = (r + 1) / 2**, drawing from U(0.5, 1): the executed
  dpyscf script's form, not the paper's U(0, 1) (train.py's mix_rng stream;
  documented in the seed-mixture analysis).
* **60 epochs vs the production arms' 100** steps (and the published
  unbounded loop), and **validate_every 0** vs v9's 10: the miniature keeps
  no held-out manifest slice to validate against. The measured FULL wall at
  60 epochs (~34 h floor, 44-60 h across the fleet at 3 workers) was
  accepted over an epoch cut when the 12 h gate fired (2026-11-04); the
  cost gate now guards the accepted wall at 72 h.
* **seed_xc pbe run-wide** vs v9's per-rung auto (SCAN seeds for the eight
  meta arms): auto on this footing would trigger an uncached SCAN seed
  generation, and the notebook states the simpler deviation instead.
* **CCSD(T) relaxed density references** for every rho target (the
  published protocol's reference quality at the miniature's basis);
  **lr_end / lr_decay_start are carried but inert** under adam_plateau
  (constant lr_start with the plateau controller flooring at lr_end) --
  stated because a reader comparing against the yaml should not expect the
  linear decay to act.
* **The campaign sweeps 43 of the 44 registered architectures** (the GEA
  arm's gate cannot share a run-wide model block; it trains under its own
  block through the library route, resolved with the rest through
  `apply_run_config`), the **grid-1 degeneracy waiver** and the **fidelity
  gate release**, both with recorded reasons.
* **Parent baselines are evaluated on the frozen CCSD(T) densities**, not
  as self-consistent parent barriers -- the comparison isolates the
  exchange-correlation prediction at one density, the quantity being
  trained.
"""
    return md(source)


def build_cell_23_train_md():
    source = """## Arc F -- training all 44 architectures on the Letter pathway

`TrainingSpec.from_dicts` assembles one training run. Everything below is
the v9 dfs_letter_allsc production form (that yaml is the anchor; the
library route states each field explicitly):

* **solver** -- `SolverConfig(mode=FULL, max_cycles=25, REASSEMBLE,
  decaying_linear mixer, scf_grad_checkpoint, tail 25 / power 2, no
  freeze, fock noise 1e-5)`: every training evaluation runs 25
  differentiable SCF cycles seeded from the record's parent-density
  `dm_seed`, and the loss scores the tail-weighted energy trajectory. The
  seed mixture randomizes the start per update:
  `D0 = (1 - beta) D_seed + beta D_minao` with `beta = (r + 1) / 2` drawn
  from the checkpointed mix RNG -- the basin-robustness start of the
  published protocol (which is why the corpus precompute requested the
  minao seed).
* **loss** -- `L6_letter_parity`, the only loss emitting `loss_Eatom` (the
  Letter's absolute atomic-total anchor at lambda_E = 0.01). Its
  `loss_kwargs` carry the reaction channels in the package's dict form:
  the barrier trio in BARRIER form (products = the transition state only;
  the chemical products are never staged) plus the H2O atomization as a
  predicted-atom reaction, and the Li/C ionization pair; `radical_species`
  is stated explicitly (see the corpus arc); `vxc_weight` / `density_weight`
  ride along verbatim from v9 but are inert under per_molecule (the loop
  forces them to 1.0 per group and the outer channel weights carry the
  structure).
* **update_scheme `per_molecule`** -- one optimizer step per group per
  epoch, fixed channel weights, `n_steps` counts EPOCHS. This corpus
  derives **12 groups**: 3 barrier reactions, 1 atomization reaction
  (h2o + its atoms), 2 ionization pairs, 6 atom anchors -- 720 updates per
  architecture at 60 epochs, 31,680 across the registry.
* **optimizer** -- `adam_plateau`: Adam at constant lr 1e-4 with coupled
  L2 (1e-6), the learning rate multiplied by 0.1 after 10 stagnant epochs
  and floored at 1e-7 (the linear decay fields are inert here; v9 verbatim).
* **channel_weights** -- all six emitted keys, spelled. A *partial* dict
  merges over the defaults and silently inherits what it omitted --
  `loss_Eatom` would stay at the default 1.0, a silent 100x on the
  Letter's 0.01 -- so the cell asserts the full spelling and the two
  deliberate differences from the package defaults (vxc 0, Eatom 0.01).
* **pretrain_checkpoint** -- `<run_root>/pretrain/<arch>`; with the
  certificate present there, the gate opens (under the recorded waiver).

**The cost gate.** Self-consistent training is expensive, so before the
sweep a measurement cell times one architecture at 1 and 2 epochs in a
subprocess and differences the walls (startup, imports and the first-epoch
compile cancel), giving the steady per-update cost; the same child reports
its peak RSS, which sets the worker memory budget. The train-run cell then
projects the FULL sweep (44 archs x 720 updates / workers) and asserts it
under a **72 h ceiling -- the accepted wall**: the original 12 h gate fired
on the measured numbers (11.5 s/update on the smallest architecture, 33
s/update on an attention arm, 7.02 GiB peak RSS) and the run was accepted
at the full 60 epochs rather than cut (2026-11-04). A firing assert is a
stop for a decision, never a silently turned knob.

Every architecture trains as a parallel `python -m xcquinox.pipeline.
_train_one_spec <spec.pkl> --device cpu` worker -- the same worker the
cluster's train array invokes, fed a pickled spec. (An in-process training
was tried first for the streamed-progress tutorial view and abandoned: the
25-cycle padded SCF graph fails to materialize its XLA dylib in the
long-lived kernel after the pretrain and certificate compiles, while a
fresh process trains the identical spec without issue -- the
progress-callback streaming stays demonstrated in the pretrain arc, which
runs in-process.) The worker count is capped by memory as well as CPUs
(`parallel_worker_cap`, defined in the pretrain arc; the budget is the SCF
child's measured peak).
"""
    return md(source)


def build_cell_24_train_spec():
    source = """# The six channels L6 emits, at the Letter weights (reactions 1, densities
# 20, atomic totals 0.01, no vxc). All six are spelled because a partial
# dict merges over defaults instead of replacing them: an omitted
# loss_Eatom would silently inherit the package default 1.0 (100x the
# Letter's value).
CHANNEL_WEIGHTS = {
    "loss_AE": 1.0,
    "loss_BH76": 1.0,
    "loss_IP13": 1.0,
    "loss_rho": 20.0,
    "loss_vxc": 0.0,
    "loss_Eatom": 0.01,
}
from xcquinox.pipeline.train import _DEFAULT_CHANNEL_WEIGHTS, _training_groups
assert set(CHANNEL_WEIGHTS) == set(_DEFAULT_CHANNEL_WEIGHTS)
for _k in ("loss_vxc", "loss_Eatom"):
    # the two DELIBERATE differences from the package defaults (the Letter's
    # values); every other key coincides with them
    assert CHANNEL_WEIGHTS[_k] != _DEFAULT_CHANNEL_WEIGHTS[_k], _k
assert 0.3 < AE_HA < 0.4  # Hartree, not kcal/mol

# every molecule needs a targets entry (spec validation): the atoms carry
# their anchors (loss_Eatom reads them), the cations the neutral element's
# anchor (inert -- cations are neither compounds nor density targets), h2o
# keeps its real AE value (it trains through the reaction channel, aux-forced
# out of the AE channel), and the barrier species carry 0.0 placeholders.
TARGETS = {
    **{s: ATOM_ANCHORS[s] for s in ATOM_ANCHORS},
    "Li+": ATOM_ANCHORS["Li"], "C+": ATOM_ANCHORS["C"],
    "h2o": AE_HA,
    **{n: 0.0 for n in ("HO", "N2", "HF", "CH3", "n2ohts", "RKT11", "hf2ts")},
}

# the v9 dfs_letter_allsc solver block, field for field
SOLVER = SolverConfig(
    mode=SolverMode.FULL,
    max_cycles=25,
    feature_policy=FeaturePolicy.REASSEMBLE,
    mixer_name="decaying_linear",
    mixer_kwargs=(("base", 0.3), ("floor", 0.3), ("step_offset", 1.0)),
    density_fit=False,
    scf_grad_checkpoint=True,
    scf_loss_use_tail=True,
    scf_loss_tail=25,
    scf_loss_weight_power=2.0,
    freeze_on_convergence=False,
    orientation_lock_strength=ORIENTATION_LOCK,
    seed_source="pbe",
    fock_noise_std=1.0e-5,
    fock_noise_seed=0,
)

def make_train_spec(arch_name):
    return pipeline.TrainingSpec.from_dicts(
        molecules=tuple(corpus_specs),
        targets=TARGETS,
        atom_energies=dict(ATOM_ANCHORS),
        loss_name="L6_letter_parity",
        loss_kwargs={
            # the barrier trio in BARRIER form: products = the staged TS
            # only (the chemical products are never staged), coeffs
            # (-1,)*n_reactants + (1,), the GMTKN55 forward barrier in Ha
            # (computed from the package table, never transcribed)
            "bh76_reactions": [
                *[{"name": _r["name"],
                   "reactants": tuple(_r["reactants"]),
                   "products": (_r["ts_species"],),
                   "coeffs": (-1.0,) * len(_r["reactants"]) + (1.0,),
                   "e_rxn_ref": _r["barrier_ref"] / KCAL_PER_HA}
                  for _r in DFS_BH76_REACTIONS],
                # the H2O atomization as a predicted-atom reaction (the only
                # form under which a compound's AE trains per-molecule)
                {"name": "h2o", "reactants": ("h2o",),
                 "products": ("H", "O"), "coeffs": (-1.0, 2.0, 1.0),
                 "e_rxn_ref": AE_HA},
            ],
            # the DFS ionization pair, NIST references in Ha
            "ip13_pairs": [
                {"name": _p["name"], "neutral": _p["neutral"],
                 "cation": _p["cation"], "ip_ref": _p["ip_ref"] / KCAL_PER_HA}
                for _p in DFS_IP13_PAIRS],
            # every compound is aux-excluded (h2o trains through its
            # reaction; the barrier species are channel species)
            "aux_only_names": ("CH3", "HF", "HO", "N2", "RKT11", "h2o",
                               "hf2ts", "n2ohts"),
            # the Letter's L6 threading: every neutral atom regularized, the
            # Dick h_species, and the radical set STATED (the pool derives
            # radicals from AE metadata this corpus's OH never carries)
            "regularize_atom_syms": ("C", "F", "H", "Li", "N", "O"),
            "h_species": ("H", "Li"),
            "radical_species": ("HO",),
            "solver_config": SOLVER,
            # v9 verbatim; both are inert under per_molecule (the loop
            # forces them to 1.0 per group -- the outer channel weights
            # carry the structure)
            "vxc_weight": 0.01,
            "density_weight": 0.1,
            "density_per_electron": True,
        },
        arch=resolve_arch(arch_name),  # the same identity pretrain wrote
        n_steps=TIER["train_epochs"],
        lr_start=1.0e-4,
        lr_end=1.0e-7,       # the plateau floor (the linear decay is inert)
        lr_decay_start=0.5,  # inert under adam_plateau; v9 verbatim
        grad_clip=1.0,
        weight_decay=1.0e-6,
        seed=42,
        solver_config=SOLVER,
        update_scheme="per_molecule",
        optimizer="adam_plateau",
        plateau_patience=10,
        plateau_factor=0.1,
        seed_mix_atomic=True,
        respect_sc_flag=False,   # every point self-consistent (the allsc arm)
        nonsc_weight=0.01,
        pad_group_to_common_shape=True,
        # Masked-superset assembly exists (default-off): one run-level loss
        # + per-group traced masks collapses the assembly to one module
        # (plateau 14k measured, 2026-11-04) but currently computes the
        # UNION's SCF energies every update (~15x per-update cost -- over
        # the accepted 72 h ceiling). Off until the subset-stack variant
        # lands; the scoped path with canonical field-presence holds 27.7k
        # (2.4x margin) at the original per-update cost.
        superset_assembly=False,
        channel_weights=dict(CHANNEL_WEIGHTS),
        val_frac=0.2,
        validate_every=0,
        patience=8,
        early_stop_min_delta=0.1,
        validation_density_weight=20.0,
        checkpoint_every=10,
        pbe_anchor_weight=0.0,
        require_atom_anchors=False,
        balancing=GradNormConfig(alpha=1.5),  # unused by per_molecule; v9 verbatim
        pretrain_checkpoint=pretrain_checkpoint_dir(RUN_DIR, arch_name),
        checkpoint_dir=os.path.join(RUN_DIR, "train", arch_name),
    )

_spec0 = make_train_spec(ARCH_NAMES[0])
assert _spec0.loss_kwargs_dict["radical_species"] == ("HO",)
# the group census, from the package's own group builder: 3 barriers + 1
# atomization reaction + 2 ionization pairs + 6 atom anchors
N_GROUPS = len(_training_groups(_spec0))
assert N_GROUPS == 12, N_GROUPS
print(f"spec for {_spec0.arch.name}: {_spec0.n_steps} epochs x {N_GROUPS} "
      f"groups, {len(_spec0.molecules)} molecules, L6 + FULL-25 SCF")
print(f"groups: {[g['label'] for g in _training_groups(_spec0)]}")
print(f"targets: {TARGETS}")
"""
    return code(source)


def build_cell_24b_measure():
    source = '''# The cost gate's measurement: time one architecture at 1 and 2 epochs in
# a subprocess (startup, imports and the first-epoch compile cancel in the
# difference -- the progress callback fires per EPOCH, so there is no
# per-update wall to read off a single run), and read the child's own peak
# RSS (ru_maxrss), which sets the training worker's memory budget.
def _measure_run(arch_name, epochs, tag):
    _spec = replace(make_train_spec(arch_name), n_steps=epochs,
                    checkpoint_dir=os.path.join(WORK_ROOT, "measure", tag))
    _pkl = os.path.join(WORK_ROOT, "measure", f"{tag}.pkl")
    os.makedirs(os.path.dirname(_pkl), exist_ok=True)
    with open(_pkl, "wb") as _f:
        pickle.dump(_spec, _f)
    _code = (
        "import pickle, resource, sys, time\\n"
        "import xcquinox.pipeline as pipeline\\n"
        "t0 = time.time()\\n"
        "with open(sys.argv[1], 'rb') as f:\\n"
        "    spec = pickle.load(f)\\n"
        "pipeline.run_training(spec)\\n"
        "peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 ** 2\\n"
        "print(f'MEASURE wall_s {time.time() - t0:.1f} peak_gib {peak:.2f}')\\n"
    )
    _env = dict(os.environ)
    _env["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"
    _env["OMP_NUM_THREADS"] = "1"
    _t0 = time.time()
    _p = subprocess.run([sys.executable, "-c", _code, _pkl],
                        env=_env, capture_output=True, text=True)
    if _p.returncode != 0:
        raise RuntimeError(f"measurement {tag} failed:\\n{_p.stderr[-2000:]}")
    _m = [ln for ln in _p.stdout.splitlines()
          if ln.startswith("MEASURE")][-1]
    _toks = _m.split()
    return float(_toks[2]), float(_toks[4]), time.time() - _t0

_meas_arch = ARCH_NAMES[0]
_w1, _pk1, _t1 = _measure_run(_meas_arch, 1, "t1")
_w2, _pk2, _t2 = _measure_run(_meas_arch, 2, "t2")
STEADY_UPDATE_S = (_w2 - _w1) / N_GROUPS
PEAK_RSS_GIB = max(_pk1, _pk2)
TIMINGS["cost measurement"] = _t1 + _t2
print(f"steady per-update wall (compile removed): {STEADY_UPDATE_S:.2f} s "
      f"[compile-inclusive bound {_w1 / N_GROUPS:.2f} s]")
print(f"training-child peak RSS: {PEAK_RSS_GIB:.2f} GiB -> suggested worker "
      f"budget {PEAK_RSS_GIB + 2.0:.1f} GiB (peak + headroom)")
'''
    return code(source)


def build_cell_25_train_run():
    source = """def _model_path(a):
    return os.path.join(RUN_DIR, "train", a, "model.eqx")

_todo = [a for a in ARCH_NAMES if not os.path.isfile(_model_path(a))]
if len(_todo) < len(ARCH_NAMES):
    print(f"skipping {len(ARCH_NAMES) - len(_todo)} cached model(s)")

# Decision's gate: project the FULL tier from the measured steady wall and
# refuse to start the sweep past the accepted ceiling. The original 12 h
# gate fired on the measured numbers (33.7 h floor at the smallest arch's
# 11.5 s/update; the attention arms measure ~33 s/update and the fleet
# estimate was 44-60 h; a 4th worker needs a 6.1 GiB budget, under the
# measured 7.02 GiB attention peak -- dead on this host), and the user
# accepted the wall at the full 60 epochs (2026-11-04). The 72 h ceiling
# admits the accepted run and still stops a runaway projection. A firing
# assert is a STOP for a user decision, never a knob turned silently. The
# projection targets the FULL tier's numbers (44 archs x 60 epochs x 12
# groups) whatever tier is running, so the smoke tier guards the real run
# too.
TRAIN_WORKER_GIB = 9.0  # 2 concurrent SCF workers (the user's call after
                        # the 3-wide compile-transient failure, smoke run
                        # 9): int(0.8 x 30.5 / 9.0) = 2, each child holding
                        # a 9 GiB budget over the measured 7.02 GiB peak
_workers_full = parallel_worker_cap(
    len(pipeline.ARCHITECTURES) - 1, per_worker_gib=TRAIN_WORKER_GIB)
_proj_s = (len(pipeline.ARCHITECTURES) * TIERS["full"]["train_epochs"]
           * N_GROUPS * STEADY_UPDATE_S / _workers_full)
print(f"FULL projection: {len(pipeline.ARCHITECTURES)} archs x "
      f"{TIERS['full']['train_epochs']} epochs x {N_GROUPS} groups x "
      f"{STEADY_UPDATE_S:.2f} s/update / {_workers_full} workers "
      f"= {_proj_s / 3600:.1f} h (ceiling 72 h, the accepted wall)")
assert _proj_s < 72 * 3600, (
    f"the measured FULL projection {_proj_s / 3600:.1f} h exceeds the "
    f"72 h accepted-wall ceiling -- STOP for a decision: this is past "
    f"even the 2026-11-04 acceptance")

if _todo:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    _spec_dir = os.path.join(WORK_ROOT, "train_specs")
    os.makedirs(_spec_dir, exist_ok=True)
    for _a in _todo:
        with open(os.path.join(_spec_dir, f"{_a}.pkl"), "wb") as _f:
            pickle.dump(make_train_spec(_a), _f)

    # Preflight: the per-process VMA ceiling, measured 2026-11-04. Before
    # the defused-step compile-leak fix, training children climbed past the
    # kernel's default vm.max_map_count=65530 and died at LLVM section
    # allocation. With every solver-carrying channel behind a stable-key
    # jit, a child's map count PLATEAUS at ~64.8k on this corpus -- under
    # the default ceiling, with little margin. Warn (do not refuse): a
    # bigger corpus or arch could tip over, and the sysctl
    # vm.max_map_count=262144 (administrator) is the remedy if one does.
    _max_maps = int(Path("/proc/sys/vm/max_map_count").read_text())
    if _max_maps <= 65530:
        print(f"NOTE vm.max_map_count={_max_maps} (kernel default): the "
              f"measured training-child plateau is ~64.8k maps on this "
              f"corpus -- it fits, with thin margin; ask the administrator "
              f"for 262144 if a larger sweep ever dies at an LLVM section "
              f"allocation")

    def _train_one(arch_name):
        child_env = dict(os.environ)
        child_env["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"
        child_env["OMP_NUM_THREADS"] = "1"
        # glibc's per-thread arenas add VMAs of their own; capped at two so
        # every mapping the child consumes below the ceiling is XLA's own
        child_env["MALLOC_ARENA_MAX"] = "2"
        _p = subprocess.run(
            [sys.executable, "-m", "xcquinox.pipeline._train_one_spec",
             os.path.join(_spec_dir, f"{arch_name}.pkl"), "--device", "cpu"],
            env=child_env, capture_output=True, text=True)
        if _p.returncode != 0:
            raise RuntimeError(f"[{arch_name}] training failed:\\n"
                               f"{_p.stdout[-800:]}\\n{_p.stderr[-2000:]}")
        return arch_name

    # Release the kernel's memory before the training wave. Every training
    # child rebuilds its corpus from the MoleculeSpecs, so the kernel's 16
    # precomputed records are dead weight past this point (only the h2o
    # record is still needed, for the readout identity checks), and the jax
    # executable caches from the pretrain/certificate stages hold several
    # GiB more. The 2026-11-04 LLVM section-memory failures (smoke runs 9
    # and 10, at 3 and 2 workers) were compile transients landing on top of
    # this arena; a slimmer kernel is what makes the wave fit.
    def _rss_gib():
        with open("/proc/self/status") as _st:
            for _ln in _st:
                if _ln.startswith("VmRSS"):
                    return int(_ln.split()[1]) / 1024 ** 2
    print(f"kernel RSS before release: {_rss_gib():.1f} GiB")
    _keep = {"h2o": corpus_records["h2o"]}
    corpus_records.clear()
    corpus_records.update(_keep)
    jax.clear_caches()
    import gc
    gc.collect()
    print(f"kernel RSS after release:  {_rss_gib():.1f} GiB")

    # EVERY architecture trains as a subprocess worker (the same worker the
    # cluster's train array invokes). An in-process training was tried and
    # abandoned: the 25-cycle padded SCF graph fails to materialize its XLA
    # dylib in the long-lived tutorial kernel after the pretrain/certificate
    # compiles, while a fresh process compiles and trains the identical
    # spec without issue (measured, smoke run 8). The progress-callback
    # streaming is demonstrated in the pretrain arc, which runs in-process.
    _max_workers = parallel_worker_cap(len(_todo),
                                       per_worker_gib=TRAIN_WORKER_GIB)
    print(f"training {len(_todo)} archs in parallel: "
          f"{_max_workers} workers (cpu- and memory-capped; "
          f"{TRAIN_WORKER_GIB} GiB budgeted per SCF training worker)")
    _bar = tqdm(total=len(_todo), desc="train (parallel)", dynamic_ncols=True)
    _t0 = time.time()
    try:
        with ThreadPoolExecutor(max_workers=_max_workers) as _ex:
            _futs = {}
            for _i, _a in enumerate(_todo):
                _futs[_ex.submit(_train_one, _a)] = _a
                # stagger the FIRST wave's compiles apart: the LLVM section
                # allocations are transient spikes, and two of them landing
                # together is what killed smoke runs 9 and 10
                if _i < _max_workers - 1:
                    time.sleep(90)
            for _fut in as_completed(_futs):
                _bar.update(1)
                _bar.set_postfix(arch=_fut.result())
    finally:
        _bar.close()
    TIMINGS["training (parallel)"] = time.time() - _t0

for _a in ARCH_NAMES:
    assert os.path.isfile(_model_path(_a)), _a
print(f"trained models present for all {len(ARCH_NAMES)} archs")
"""
    return code(source)


def build_cell_26_train_timing():
    source = """print("training timing:")
for _k, _v in TIMINGS.items():
    if "train" in _k:
        print(f"  {_k}: {_v:.1f} s")
print("\\n(again: measurements of THIS run; the close-out cell compares them to the budget)")
"""
    return code(source)


def build_cell_27_pt_md():
    source = """## Arc G -- proton-transfer transfer

Three paths the training corpus never saw:

1. **Water-dimer cation, hemibonded -> proton-transferred** (HBWD -> PTWD).
   The two geometries are packaged in the library
   (`subset_selection._make_hb_atoms` / `_make_pt_atoms`, charge +1, spin 1,
   def2-svp / grid-1 footing); the barrier is `E(PTWD) - E(HBWD)`, the
   convention of `data/training_subsets/01wf` (README), whose tracked
   CCSD(T) barrier is read back in the results cell as the cross-check.
2. **Zundel H5O2+ scan** -- O-O fixed at 2.4 A, the central proton displaced
   along the O-O axis across the tier's scan points (schematic, not a relaxed
   path).
3. **Neutral water-dimer scan** -- O-O fixed at 2.9 A, the donor's bridging
   proton moved between the two oxygens (schematic).

**Readout semantics.** Every energy on these paths is evaluated on the
**frozen CCSD(T) density** of that point: `E_tot = E_non_xc[dm_ccsd_t] +
E_xc^NN[rho_ccsd_t]`, with the parent functional of the architecture's footing
(PBE for a GGA arm, SCAN for a meta-GGA arm) evaluated on the *same* patched
record as its baseline. The reference densities come from the same relaxed
(T) driver the corpus used. This is what the patched-record helper below
does -- readout only; training ran self-consistent SCF from the seed mixture
(see the Deviations cell).

The helper always **rebuilds J from a rebuilt Mole**: the record's stored
`j_matrix` is full-ERI `J[dm_pbe]` (`data.py`), and reusing it at a different
density matrix corrupts `E_non_xc` -- measured 29 kcal/mol on the H2O record
at `dm' = 1.001 dm_pbe` -- while every dm_pbe-pinned check stays green.
Four identity checks pin the helper, asserted before any barrier is
reported:

1. patching with `dm_pbe` reproduces `fixed_density_total_energy` on the
   unpatched record exactly;
2. `E_non_xc(dm_pbe) + E_xc^parent(rho_pbe)` reproduces the record's `E_pbe`;
3. the helper's own `J[dm_pbe]` equals the stored `j_matrix`;
4. at a perturbed `dm' = 1.001 dm_pbe` the freshly rebuilt J **diverges**
   from the stored `j_matrix` -- checks 1-3 are all pinned at `dm_pbe`,
   where stored and rebuilt J are identical, so the reuse error above is
   invisible to them; this one transposes the comparison to the perturbed
   density, where a reuse collapses the divergence to exactly zero.
"""
    return md(source)


def build_cell_28_paths(pt_scan_points):
    n = pt_scan_points
    source = f"""# Cation pair: the packaged geometries (relaxed references, tracked; the
# makers carry charge/spin/name in Atoms.info).
def _atoms_to_spec(a, name):
    syms, pos = list(a.symbols), np.asarray(a.positions)
    atom = "; ".join(f"{{s}} {{p[0]:.6f}} {{p[1]:.6f}} {{p[2]:.6f}}"
                     for s, p in zip(syms, pos))
    comp = {{}}
    for s in syms:
        comp[s] = comp.get(s, 0) + 1
    return pipeline.MoleculeSpec(
        name=name, atom=atom, basis=BASIS,
        charge=int(a.info.get("charge", 0)), spin=int(a.info.get("spin", 0)),
        atom_composition=tuple(sorted(comp.items())), grid_level=GRID_LEVEL)

cation_points = [
    _atoms_to_spec(_make_hb_atoms(), "cation_hb"),
    _atoms_to_spec(_make_pt_atoms(), "cation_pt"),
]

# Zundel H5O2+ schematic scan: O-O 2.4 A along x, central H at midpoint + delta.
def _zundel_spec(delta):
    d = float(delta)
    atom = (
        f"O {{-1.2:.6f}} 0 0; O {{1.2:.6f}} 0 0; "
        f"H {{d:.6f}} 0.0 0.0; "
        "H -1.45 0.85 0.35; H -1.45 -0.85 -0.35; "
        "H 1.45 0.85 -0.35; H 1.45 -0.85 0.35"
    )
    return pipeline.MoleculeSpec(
        name=f"zundel_{{d:+.2f}}", atom=atom, basis=BASIS, charge=1, spin=0,
        atom_composition=(("H", 5), ("O", 2)), grid_level=GRID_LEVEL)

# Neutral dimer schematic scan: O-O 2.9 A, bridging H moved between them.
def _neutral_spec(delta):
    d = float(delta)
    atom = (
        "O 0 0 0; H 0.6 0.75 0; H -0.6 0.75 0; "
        f"O 2.9 0 0; H {{2.9 - 0.96 + d:.6f}} -0.35 0; H 3.5 -0.75 0"
    )
    return pipeline.MoleculeSpec(
        name=f"neutral_{{d:+.2f}}", atom=atom, basis=BASIS, charge=0, spin=0,
        atom_composition=(("H", 4), ("O", 2)), grid_level=GRID_LEVEL)

_deltas = np.linspace(-0.6, 0.6, {n})
zundel_points = [_zundel_spec(d) for d in _deltas]
neutral_points = [_neutral_spec(d) for d in _deltas]

all_points = cation_points + zundel_points + neutral_points
print(f"path points: 2 cation + {{len(zundel_points)}} Zundel + "
      f"{{len(neutral_points)}} neutral = {{len(all_points)}}")
for _s in cation_points:
    print(f"  {{_s.name}}: charge {{_s.charge}} spin {{_s.spin}}")
"""
    return code(source)


def build_cell_29_pt_ccsd():
    source = """# Relaxed CCSD(T) references per path point (the corpus driver's helper),
# cached; the DMs are the frozen densities every barrier is evaluated on.
_pt_cache = os.path.join(WORK_ROOT, "pt_refs")
os.makedirs(_pt_cache, exist_ok=True)

def _pt_ref_path(name):
    return os.path.join(_pt_cache, f"{name}.npz")

_t0 = time.time()
for _spec in tqdm(all_points, desc="PT CCSD(T) refs", dynamic_ncols=True):
    if os.path.isfile(_pt_ref_path(_spec.name)):
        continue
    mol = gto.M(atom=_spec.atom, basis=BASIS,
                charge=_spec.charge, spin=_spec.spin, verbose=0)
    dm_pt, _dm_total, _dm_ccsd, E_t, _t1 = _ccsd_t_relaxed(
        mol, _spec.name)
    # per-spin pair for the UKS cation points (the UHF (T)-RDM returns the
    # pair); the patch helper indexes exactly these shapes, and the
    # electron-count refusal rules out a basis mix-up
    _require_electron_count(dm_pt, mol, _pt_ref_path(_spec.name))
    np.savez(
        _pt_ref_path(_spec.name),
        E_ccsd_t=np.asarray(E_t),
        dm=dm_pt,
        is_unrestricted=np.asarray(mol.spin != 0),
        t1_diagnostic=np.asarray(_t1),
    )
TIMINGS["PT CCSD(T) refs"] = time.time() - _t0

_pt_refs = {}
for _spec in all_points:
    with np.load(_pt_ref_path(_spec.name)) as _z:
        _pt_refs[_spec.name] = {k: _z[k] for k in _z.files}
for _spec in all_points:
    print(f"  {_spec.name:<14} E_ccsd(T) = "
          f"{float(_pt_refs[_spec.name]['E_ccsd_t']):.6f} Ha"
          f"  ({'UKS' if bool(_pt_refs[_spec.name]['is_unrestricted']) else 'RKS'})")
"""
    return code(source)


def build_cell_30_energy_on_dm():
    source = """# Parent records for every path point (the footing the descriptors and
# E_non_xc pieces are built on), then the readout helper.
_t0 = time.time()
pt_records = {}
for _spec in tqdm(all_points, desc="precompute PT", dynamic_ncols=True):
    pt_records[_spec.name] = pipeline.precompute_fixed_density_data(
        _spec,
        required_keys=tuple(sorted(_REQUIRED_KEYS)),
        descriptors=_ALL_DESCRIPTORS,
        orientation_lock_strength=ORIENTATION_LOCK,
    )
TIMINGS["PT precompute"] = time.time() - _t0


def _load_trained(arch_name):
    # The package's trained-checkpoint reader: the class record beside the
    # checkpoint (model.eqx.class.json) is checked against the skeleton's
    # class BEFORE the leaves are read (checkpoint_class
    # .load_trained_checkpoint). A bare deserialise would fill this skeleton
    # with another class's leaves unnoticed -- the anchor, the descriptor
    # coordinates and the ueg gate change no parameter shape, and the
    # x2/tanh2 gate pair is exactly such a shape-identical pair, so a
    # disagreement between the readout's resolve_arch and the identity
    # training built at would load silently and every identity check below
    # would still pass.
    _skeleton = AlecGGAModel.from_arch(resolve_arch(arch_name), seed=0)
    return load_trained_checkpoint(_model_path(arch_name), _skeleton)


def _write_feature_blocks(rec, descriptors, feats, spin_channel=None):
    # Split one concatenated _reassemble_features output back into the
    # record's per-descriptor slots (total: required_mol_keys[0]; per-channel:
    # spin_mol_keys; geometry-only descriptors have no per-channel key).
    _off = 0
    for _d in descriptors:
        _w = _d.n_features
        if spin_channel is None:
            rec[_d.required_mol_keys[0]] = feats[:, _off:_off + _w]
        elif _d.spin_mol_keys:
            rec[_d.spin_mol_keys[spin_channel]] = feats[:, _off:_off + _w]
        _off += _w
    assert _off == feats.shape[1]


def rec_descriptors_of(rec):
    # The descriptor tuple whose blocks this record carries: the sweep's
    # deduped class union, filtered to the keys the record actually has
    # (records built in this notebook carry every architecture's blocks; the
    # filter keeps the helper honest about what it consumes).
    return tuple(_d for _d in _ALL_DESCRIPTORS
                 if rec.get(_d.required_mol_keys[0]) is not None)


def patch_record(record, dm):
    # Copy the record (never mutate the memoized one) and recompute every
    # density-contracted quantity at `dm`:
    #   - J is ALWAYS rebuilt from a rebuilt Mole (the stored j_matrix is
    #     J[dm_pbe] by construction; reusing it at a patched dm is the
    #     kcal-scale error the identity #4 check exists to kill);
    #   - per-spin rho/sigma for an unrestricted record, rho/sigma/nabla for
    #     a closed-shell one;
    #   - every DM-dependent descriptor block via the package's own
    #     reassembly (compute_from_dm per descriptor; the doubled-density
    #     per-channel blocks use rho = 2 rho_sigma, sigma = 4 sigma_ss).
    # Returns (patched record, E_non_xc at dm, J at dm) -- J as the stored
    # j_matrix stores it (per-spin pair for UKS, total for RKS).
    rec = dict(record)
    md_ = rec["mol_metadata"]
    dm_np = np.asarray(dm)
    unres = bool(rec["is_unrestricted"])
    mol = gto.M(atom=md_["atom"], basis=md_["basis"],
                charge=int(md_["charge"]), spin=int(md_["spin"]), verbose=0)

    if unres:
        _j_at_dm = np.asarray([scf.RHF(mol).get_j(mol, dm_np[s]) for s in (0, 1)])
        _j_total = _j_at_dm[0] + _j_at_dm[1]
        _dm_tot = dm_np[0] + dm_np[1]
    else:
        _j_at_dm = np.asarray(scf.RHF(mol).get_j(mol, dm_np))
        _j_total = _j_at_dm
        _dm_tot = dm_np
    _h = np.asarray(rec["h_core"])
    _e_non_xc = (float(rec["e_nuc"])
                 + float(np.einsum("ij,ji->", _dm_tot, _h))
                 + 0.5 * float(np.einsum("ij,ji->", _dm_tot, _j_total)))

    ao = np.asarray(rec["ao_grid"])              # (n_grid, nao)
    ao_deriv = np.asarray(rec["ao_grid_deriv"])  # (4, n_grid, nao)

    def _rho_sigma(d):
        r = np.einsum("pi,ij,pj->p", ao, d, ao)
        gx = 2 * np.einsum("pi,ij,pj->p", ao_deriv[1], d, ao)
        gy = 2 * np.einsum("pi,ij,pj->p", ao_deriv[2], d, ao)
        gz = 2 * np.einsum("pi,ij,pj->p", ao_deriv[3], d, ao)
        return r, gx ** 2 + gy ** 2 + gz ** 2, np.stack([gx, gy, gz])

    descriptors = rec_descriptors_of(rec)

    def _reassemble(d, rho, sigma, spin_channel=None):
        return _reassemble_features(
            descriptors,
            jnp.asarray(d),
            s_matrix=rec["s_matrix"],
            cusp_features=rec.get("cusp_features"),
            atom_context_features=rec.get("atom_context_features"),
            n_grid=ao.shape[0],
            rung35_proj_ao=rec.get("rung35_proj_ao"),
            rung35ms_proj_ao=rec.get("rung35ms_proj_ao"),
            ao_grad=jnp.asarray(ao_deriv[1:4]),
            rho=jnp.asarray(rho),
            sigma=jnp.asarray(sigma),
            spin_channel=spin_channel,
            ao=jnp.asarray(ao),
            ao_lapl=rec.get("ao_grid_lapl"),
        )

    if unres:
        # The UKS energy branch recomputes rho/sigma itself from dm_pbe, so
        # patching the matrix carries the grid quantities with it.
        rec["dm_pbe"] = jnp.asarray(dm_np)
        rho_t, sig_t, _ = _rho_sigma(_dm_tot)
        _write_feature_blocks(
            rec, descriptors, _reassemble(_dm_tot, rho_t, sig_t))
        for _s in (0, 1):
            r_s, g_s, _ = _rho_sigma(dm_np[_s])
            # the package's per-channel convention (solver._features_spin):
            # the dm argument is the FULL spin pair (2, nao, nao) -- the
            # descriptor dispatch slices the channel itself -- and only
            # rho/sigma are precomputed per spin (doubled: 2 rho_s, 4 sigma_ss)
            _write_feature_blocks(
                rec, descriptors,
                _reassemble(dm_np, 2.0 * r_s, 4.0 * g_s, spin_channel=_s),
                spin_channel=_s)
    else:
        # The RKS branch reads the precomputed grid quantities.
        rho_t, sig_t, nab_t = _rho_sigma(dm_np)
        rec["rho_grid"] = jnp.asarray(rho_t)
        rec["sigma_grid"] = jnp.asarray(sig_t)
        rec["nabla_rho_grid"] = jnp.asarray(nab_t).reshape(
            np.asarray(rec["nabla_rho_grid"]).shape)
        _write_feature_blocks(
            rec, descriptors, _reassemble(dm_np, rho_t, sig_t))

    rec["E_non_xc"] = _e_non_xc
    return rec, _e_non_xc, _j_at_dm


def energy_on_dm(model, record, dm):
    # Total energy of `model` on the frozen density matrix `dm`:
    # E_non_xc[dm] (fresh J) + E_xc^NN[rho_dm] through the package's own
    # fixed-density evaluation. Returns (E_total, E_non_xc, patched record).
    rec, _e_non_xc, _ = patch_record(record, dm)
    return float(fixed_density_total_energy(model, rec)), _e_non_xc, rec


# --- the four identity checks, on the H2O record and one cation record ---
def _identity_checks(model, record, tag):
    _dm_pbe = np.asarray(record["dm_pbe"])
    # (1) patch(dm_pbe) reproduces the unpatched oneshot energy exactly.
    _e_ref = float(fixed_density_total_energy(model, record))
    _e_patched, _e_non_xc, _ = energy_on_dm(model, record, _dm_pbe)
    assert abs(_e_patched - _e_ref) < 1e-9, (tag, _e_patched, _e_ref)
    # (2) E_non_xc(dm_pbe) + E_xc^parent(rho_pbe) == E_pbe (the record's own
    # split: E_pbe = E_non_xc + E_xc_pbe, both built from the reference SCF).
    _exc_p = fidelity_lib._parent_exc_on_stored_grid(record, "pbe")
    assert abs((_e_non_xc + _exc_p) - float(record["E_pbe"])) < 1e-8, tag
    # (3) the helper's J(dm_pbe) equals the stored j_matrix (the rebuilt-J
    # check; the J-reuse class of error is invisible to identities 1 and 2).
    _, _, _J = patch_record(record, _dm_pbe)
    _j_store = np.asarray(record["j_matrix"])
    assert np.asarray(_J).shape == _j_store.shape, (tag, _J.shape, _j_store.shape)
    assert np.max(np.abs(np.asarray(_J) - _j_store)) < 1e-9, tag
    # (4) the stored-J-reuse class, transposed to a PERTURBED dm: identities
    # 1-3 are pinned at dm_pbe, where the stored j_matrix equals the rebuilt
    # J exactly, so a helper that reuses J[dm_pbe] at any other density --
    # a silent kcal-scale corruption of every barrier -- is invisible to
    # them (under the reuse mutant every check compares stored against
    # stored). At dm' the freshly rebuilt J must DIVERGE from the stored
    # one; measured on this corpus: ~2e-2 in J, 29 kcal/mol in E_non_xc.
    _dm_prime = _dm_pbe * (1.0 + 1.0e-3)
    _, _, _J_prime = patch_record(record, _dm_prime)
    assert np.asarray(_J_prime).shape == _j_store.shape, tag
    _div = float(np.max(np.abs(np.asarray(_J_prime) - _j_store)))
    assert _div > 1.0e-6, (tag, _div)
    print(f"identity checks pass: {tag}")

model0 = _load_trained(ARCH_NAMES[0])
_identity_checks(model0, corpus_records["h2o"], "h2o")
_hb_rec = pt_records["cation_hb"]
_identity_checks(model0, _hb_rec, "cation_hb (UKS)")
"""
    return code(source)


def build_cell_31_barriers():
    source = """# Barriers: every arch x every point on the frozen CCSD(T) densities, plus
# the parent baseline of each arch's footing on the SAME patched records
# (E_non_xc[dm_ccsd_t] + E_xc^parent[rho_ccsd_t]: the parent E_xc is
# evaluated through the package's own helper, which reads dm_pbe off the
# patched copy).
def _resolve_parent(arch_name):
    return "scan" if resolve_arch(arch_name).meta_gga else "pbe"

_t0 = time.time()
rows = []
for _arch in tqdm(ARCH_NAMES, desc="barriers", dynamic_ncols=True):
    _model = _load_trained(_arch)
    _parent = _resolve_parent(_arch)
    _e_nn, _e_par, _e_ccsd = {}, {}, {}
    for _spec in all_points:
        _rec = pt_records[_spec.name]
        _dm = _pt_refs[_spec.name]["dm"]
        _e, _e_non_xc, _prec = energy_on_dm(_model, _rec, _dm)
        _e_nn[_spec.name] = _e
        _e_par[_spec.name] = (_e_non_xc
                              + fidelity_lib._parent_exc_on_stored_grid(_prec, _parent))
        _e_ccsd[_spec.name] = float(_pt_refs[_spec.name]["E_ccsd_t"])

    _cat_b = _e_nn["cation_pt"] - _e_nn["cation_hb"]
    _zundel_b = max(_e_nn[s.name] for s in zundel_points) - min(
        _e_nn[s.name] for s in zundel_points)
    _neutral_b = max(_e_nn[s.name] for s in neutral_points) - min(
        _e_nn[s.name] for s in neutral_points)
    _cat_b_par = _e_par["cation_pt"] - _e_par["cation_hb"]
    _cat_b_ccsd = _e_ccsd["cation_pt"] - _e_ccsd["cation_hb"]
    rows.append({
        "arch": _arch,
        "family": "meta" if resolve_arch(_arch).meta_gga else "gga",
        "parent": _parent,
        "barrier_cation_ha": _cat_b,
        "barrier_zundel_ha": _zundel_b,
        "barrier_neutral_ha": _neutral_b,
        "parent_barrier_cation_ha": _cat_b_par,
        "ccsd_barrier_cation_ha": _cat_b_ccsd,
    })
TIMINGS["barrier readout"] = time.time() - _t0
print(f"{len(rows)} arch rows computed")
"""
    return code(source)


def build_cell_32_run_test():
    source = """# The packaged evaluation API, demonstrated on the corpus (parent-footing
# records -- run_test rebuilds records like run_training, so the frozen-DM
# barriers above never route through it; see the path markdown).
_eval_spec = pipeline.TestSpec.from_dicts(
    model_checkpoint=_model_path(ARCH_NAMES[0]),
    arch=resolve_arch(ARCH_NAMES[0]),
    molecules=tuple(corpus_specs),
    metrics=("total_energy", "atomization_energy"),
    atom_energies=dict(ATOM_ANCHORS),
    solver_config=SOLVER,
    output_dir=os.path.join(RUN_DIR, "eval_demo"),
)
_eval_out = run_test(_eval_spec)
print(json.dumps(_eval_out, indent=1, default=str)[:1200])
"""
    return code(source)


def build_cell_33_results():
    source = """import pandas as pd
from ase.io import read as ase_read

# The tracked CCSD(T) cation barrier, read from the packaged reference (never
# hand-entered): the repo's data/training_subsets/01wf/subat_ref.traj, whose
# README defines the barrier as PTWD - HBWD with the totals under
# info['calc_energy'].
_repo_root = Path(xcquinox.pipeline.__file__).resolve().parents[2]
_subat = _repo_root / "data" / "training_subsets" / "01wf" / "subat_ref.traj"
if not _subat.is_file():
    _subat = Path.cwd() / "data" / "training_subsets" / "01wf" / "subat_ref.traj"
_frames = ase_read(str(_subat), index=":")
_e_hb = next(float(f.info["calc_energy"]) for f in _frames
             if f.info.get("name") == "HBWD")
_e_pt = next(float(f.info["calc_energy"]) for f in _frames
             if f.info.get("name") == "PTWD")
TRACKED_BARRIER_HA = _e_pt - _e_hb

df = pd.DataFrame(rows)
df["barrier_error_ha"] = df["barrier_cation_ha"] - TRACKED_BARRIER_HA
df["barrier_cation_kcal"] = df["barrier_cation_ha"] * KCAL_PER_HA
df["parent_barrier_kcal"] = df["parent_barrier_cation_ha"] * KCAL_PER_HA
df["tracked_ref_kcal"] = TRACKED_BARRIER_HA * KCAL_PER_HA

results_csv = os.path.join(WORK_ROOT, "results", "barriers.csv")
df.to_csv(results_csv, index=False)
with open(os.path.join(WORK_ROOT, "results", "barriers.json"), "w") as _f:
    json.dump({"tracked_barrier_ha": TRACKED_BARRIER_HA,
               "rows": rows}, _f, indent=1)

assert df["barrier_cation_ha"].notna().all()
assert df["barrier_error_ha"].abs().max() < 1.0, "a > 1 Ha barrier error means a broken readout"

print(f"tracked CCSD(T) cation barrier: {TRACKED_BARRIER_HA:.6f} Ha "
      f"({TRACKED_BARRIER_HA * KCAL_PER_HA:.2f} kcal/mol)")
_best = df.loc[df["barrier_error_ha"].abs().idxmin()]
_worst = df.loc[df["barrier_error_ha"].abs().idxmax()]
print(f"best  {_best['arch']:<24} err = {_best['barrier_error_ha'] * KCAL_PER_HA:+.2f} kcal/mol")
print(f"worst {_worst['arch']:<24} err = {_worst['barrier_error_ha'] * KCAL_PER_HA:+.2f} kcal/mol")

# The contingency comparison: the DFS-parity corpus is the first experiment,
# and the fallback additions (H2O/O vertical IPs, the H2O dissociation
# barrier) are the recorded next step if these statistics do not improve on
# the committed oneshot run. Baseline values are the committed run's own
# readout (commit edc7f93, FULL log scratch/2026-11-03_registry_pt).
ONESHOT_BASELINE = {"best_abs_err_ha": 0.00970,
                    "worst_abs_err_ha": 0.02422,
                    "beat_parent": "13 of 44"}
_parent_err = (df["parent_barrier_cation_ha"] - TRACKED_BARRIER_HA).abs()
_beat = int((df["barrier_error_ha"].abs() < _parent_err).sum())
_comparison = {
    "baseline_oneshot_run": ONESHOT_BASELINE,
    "this_run": {
        "best_abs_err_ha": float(df["barrier_error_ha"].abs().min()),
        "worst_abs_err_ha": float(df["barrier_error_ha"].abs().max()),
        "beat_parent": f"{_beat} of {len(df)}",
    },
}
with open(os.path.join(WORK_ROOT, "results", "baseline_comparison.json"),
          "w") as _f:
    json.dump(_comparison, _f, indent=1)
print(f"baseline (oneshot run): best {ONESHOT_BASELINE['best_abs_err_ha']:.5f} Ha, "
      f"worst {ONESHOT_BASELINE['worst_abs_err_ha']:.5f} Ha, "
      f"beat parent {ONESHOT_BASELINE['beat_parent']}")
print(f"this run:               best {_comparison['this_run']['best_abs_err_ha']:.5f} Ha, "
      f"worst {_comparison['this_run']['worst_abs_err_ha']:.5f} Ha, "
      f"beat parent {_comparison['this_run']['beat_parent']}")
df[["arch", "family", "barrier_cation_kcal", "parent_barrier_kcal",
    "tracked_ref_kcal"]].to_string(index=False)
"""
    return code(source)


def build_cell_34_manifest():
    source = """# Artifacts manifest + measured timings against the stated budget.
_manifest = {"run_dir": RUN_DIR, "work_root": WORK_ROOT}
for _root, _label in ((RUN_DIR, "run_dir"), (os.path.join(WORK_ROOT, "results"), "results")):
    _files = []
    for _dirpath, _dirnames, _filenames in os.walk(_root):
        for _fn in _filenames:
            _p = os.path.join(_dirpath, _fn)
            _files.append((os.path.relpath(_p, _root), os.path.getsize(_p)))
    _manifest[_label] = sorted(_files)

_total_s = sum(TIMINGS.values())
_manifest["timings_s"] = {k: round(v, 1) for k, v in TIMINGS.items()}
_manifest["total_s"] = round(_total_s, 1)
with open(os.path.join(WORK_ROOT, "results", "manifest.json"), "w") as _f:
    json.dump(_manifest, _f, indent=1)
with open(os.path.join(WORK_ROOT, "results", "timings.json"), "w") as _f:
    json.dump({k: round(v, 1) for k, v in TIMINGS.items()}, _f, indent=1)

print(f"total measured time: {_total_s:.0f} s ({_total_s / 60:.1f} min)")
for _k in sorted(TIMINGS, key=TIMINGS.get, reverse=True):
    print(f"  {_k}: {TIMINGS[_k]:.1f} s")
print(f"\\nartifacts: {len(_manifest['run_dir'])} files under the run dir, "
      f"{len(_manifest['results'])} under results/")
"""
    return code(source)


def build_cell_35_deviations_close_md():
    source = """### Caveats and what this notebook is not

* The training corpus is **16 species** (six atoms, two cations, one
  atomization compound, four barrier reactants, three transition states)
  against the published set's 26 points; the fine-tune asks whether the
  production pathway, run on a miniature of its own corpus, transfers. The
  parents are evaluated **on the frozen CCSD(T) densities** (not
  self-consistent parent barriers), so arch-vs-parent compares
  exchange-correlation predictions at one density; the contingency's
  fallback additions (H2O/O vertical IPs, the H2O dissociation barrier)
  are the recorded next step if the baseline comparison does not improve.
* The campaign yaml sweeps **43 of the 44** registered architectures: the
  GEA arm's `gea_mu='pbe'`-pinned `x2` gate cannot share a run-wide model
  block with the meta rungs' `tanh2` (production splits the same way into
  arch_screen and the mgga families campaign). The arm still trains -- in
  the library route under its own block, resolved with the rest through
  `apply_run_config` -- so the readout covers all 44.
* The Zundel and neutral-dimer scans are **schematic** (fixed O-O, displaced
  proton), not relaxed paths; only the cation pair uses relaxed geometries,
  and it is the one carrying the tracked reference.
* The fidelity certificates under the waiver record a verdict on the
  corpus-narrowed oracle; they are measurements, not gates, and a run
  rendered from this configuration can never become a quantitative
  cluster result (the gate release reaches the train stage and nothing
  else).
* `run_test` is demonstrated on the parent-footing corpus records only; the
  barrier numbers come from the notebook-side frozen-DM evaluation, because
  `run_test` rebuilds records from the MoleculeSpecs (there is no
  patched-record injection point -- readout-only is what the patched-record
  helper exists for).
"""
    return md(source)


def build_cell_36_next_md():
    source = """## Where to go next

* `notebooks/gga_training_pipeline.ipynb` -- the single-architecture deep
  dive this tutorial compresses (pretraining internals, solver modes,
  evaluation probes).
* `xcquinox/pipeline/cluster/examples/` -- production campaign yamls
  (the workflow-matrix template is the wiring check; the gga_final configs
  are the production arms this miniature borrows its cloning knobs from).
* `hpcjobs/SEAWULF_RUNBOOK.md` -- taking a campaign from a dry-run like the
  one above to queued array jobs (the same run directory, `--submit`).
* `tools/run_fast_suite.py` -- the fast test chunks covering everything this
  notebook touches (chunk A: notebooks; E1a: the end-to-end notebook tests).

The miniature scales by exactly one move: the same yaml with a production
identity (basis, grid, the full pools) and a real partition, submitted from
the cluster checkout. The notebook's run directory is already its layout.
"""
    return md(source)


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def build_cells(*, pretrain_steps_harness=4000, pretrain_steps_library=1500,
                train_epochs=60, pt_scan_points=5, arch_subset=None,
                work_root=DEFAULT_WORK_ROOT):
    cells = [
        build_cell_01_intro(),
        build_cell_02_map(),
        build_cell_03_tier(pretrain_steps_harness, pretrain_steps_library,
                           train_epochs, pt_scan_points, arch_subset,
                           work_root),
        build_cell_04_imports(),
        build_cell_05_registry_md(),
        build_cell_06_registry(),
        build_cell_07_yaml_md(),
        build_cell_08_yaml_code(),
        build_cell_09_submit_md(),
        build_cell_10_submit(),
        build_cell_11_datagen_md(),
        build_cell_12_datagen(),
        build_cell_13_pretrain_stage(),
        build_cell_14_library_md(),
        build_cell_15_pretrain_specs(),
        build_cell_16_pretrain_run(),
        build_cell_17_pretrain_timing(),
        build_cell_18_corpus_md(),
        build_cell_18b_staging(),
        build_cell_19_ccsd_refs(),
        build_cell_20_records(),
        build_cell_21_certificates(),
        build_cell_22_deviations_md(),
        build_cell_23_train_md(),
        build_cell_24_train_spec(),
        build_cell_24b_measure(),
        build_cell_25_train_run(),
        build_cell_26_train_timing(),
        build_cell_27_pt_md(),
        build_cell_28_paths(pt_scan_points),
        build_cell_29_pt_ccsd(),
        build_cell_30_energy_on_dm(),
        build_cell_31_barriers(),
        build_cell_32_run_test(),
        build_cell_33_results(),
        build_cell_34_manifest(),
        build_cell_35_deviations_close_md(),
        build_cell_36_next_md(),
    ]
    for idx, cell in enumerate(cells, start=1):
        cell["id"] = f"cell_{idx:02d}"
    return cells


def main(output_path=None, **knobs):
    if output_path is None:
        output_path = "notebooks/registry_pt_transfer.ipynb"
    nb = nbf.v4.new_notebook(
        cells=build_cells(**knobs),
        metadata={
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
    )
    nbf.validate(nb)
    with open(output_path, "w") as f:
        nbf.write(nb, f)
    print(f"wrote {output_path} ({len(nb.cells)} cells)")
    return output_path


if __name__ == "__main__":
    main()
