"""xcquinox.alec.cluster.grid_config: config layer for the HPC training harness.

The harness submits a grid of training jobs to a SLURM cluster as an array
job. The grid is the Cartesian product of a small set of swept axes, defined
declaratively in a YAML (or JSON) config file. This module provides:

  - Frozen dataclasses describing every section of that config.
  - ``load_grid_config``: parse a ``.yaml``/``.json`` file into a ``GridConfig``.
  - ``expand_grid``: the deterministic Cartesian product producing one
    ``GridCell`` per SLURM array task. A cell's index in the returned list IS
    its array task id, so the expansion MUST be byte-stable across runs and
    Python versions (achieved via ``sorted(set(...))`` per axis).
  - ``validate_grid_semantics``: login-node pre-submission sanity checks.

Design note, the ``domain`` dependency:
    ``validate_grid_semantics`` needs the size of the training-point pool to
    bound ``subset_size``. That pool lives in the not-yet-built ``domain.py``
    module. To avoid a hard import dependency on a module that does not exist,
    the domain object is received as a parameter; we depend only on it
    exposing an integer ``pool_size`` attribute. See the function docstring.
"""
from dataclasses import dataclass, field, fields
from itertools import product
import math
import os
import re
import warnings

# The ONE definition of the calibrated orientation-lock strength, in a module
# that imports nothing. It is bound in this module's BODY deliberately: the
# earlier form deferred ``from orientation_lock import DEFAULT_STRENGTH`` into
# a ``default_factory`` to keep numpy out of the certificate readers' closure,
# and did not achieve it -- the factory runs whenever the field is not
# supplied, which is every configuration that does not state the lock, i.e.
# exactly the case the default exists for. Measured with the package __init__
# modules stubbed, on such a configuration: 223 modules after a load with
# numpy present, against 121 before the harness had a lock default at all and
# 122 here.
from xcquinox.alec.orientation_lock_default import (
    DEFAULT_STRENGTH as DEFAULT_ORIENTATION_LOCK_STRENGTH)


# ---------------------------------------------------------------------------
# Module constants
# ---------------------------------------------------------------------------

# Harness-local set of supported subset-selection metrics. There is no metric
# registry in subset_selection.py, so the allowed names are hard-coded here.
VALID_METRICS = frozenset({"l2", "jsd"})

# Allowed values for the GridConfig string-enum fields.
VALID_ON_PRECOMPUTE_FAILURE = frozenset({"abort", "drop_failed_species"})
VALID_BH76_MODE = frozenset({"reaction_energy", "barrier_height"})


# ---------------------------------------------------------------------------
# Grid cell + swept axes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GridCell:
    """One grid point = one SLURM array task.

    A cell's position in the list returned by ``expand_grid`` IS its SLURM
    array task index, so the expansion order must be deterministic.
    """
    arch: str
    loss: str
    metric: str
    subset_size: int
    solver: str


@dataclass(frozen=True)
class SweepAxes:
    """The five swept axes. Each is a tuple of candidate values; the grid is
    their Cartesian product (see ``expand_grid``)."""
    arch: tuple[str, ...]
    loss: tuple[str, ...]
    metric: tuple[str, ...]
    subset_size: tuple[int, ...]
    solver: tuple[str, ...]


# ---------------------------------------------------------------------------
# Named solver config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SolverNamed:
    """A named solver configuration referenced by the ``solver`` sweep axis.

    Core fields: ``mode``, ``max_cycles``, optional ``feature_policy`` /
    ``scf_grad_checkpoint``. Plus the DFS self-consistency knobs (2026-06-24):
    an optional ``mixer_name`` / ``mixer_kwargs`` (to select the step-decaying
    mixer) and the tail-weighted-energy-loss toggles. All default to the prior
    linear/0.5 mixer + final-step-only loss so existing solvers are unchanged.
    Do NOT add conv_tol here.
    """
    mode: str
    max_cycles: int
    feature_policy: str | None = None
    # 2026-06-20: wrap the unrolled SCF scan in jax.checkpoint (O(sqrt(cycles))
    # backprop memory at ~1.5x recompute) so long-cycle FULL training (full_25)
    # stays memory-bounded. Default off keeps existing solvers byte-identical.
    scf_grad_checkpoint: bool = False
    # 2026-06-24: DFS step-decaying mixer + tail-weighted energy loss. None ->
    # SolverConfig keeps its linear/alpha-0.5 default; tail off -> final-step
    # only. mixer_kwargs is a hashable tuple-of-(name, value) pairs.
    mixer_name: str | None = None
    mixer_kwargs: tuple[tuple[str, float], ...] | None = None
    scf_loss_use_tail: bool = False
    scf_loss_tail: int = 10
    scf_loss_weight_power: float = 2.0
    # 2026-07-02: orientation lock. Coefficient on the traceless
    # anisotropic-quadrupole h_core bias (orientation_lock.py) that makes a
    # degenerate radical's density reproducible. 0.0 -> off -> byte-identical, so
    # existing sweep YAMLs (which do not set it) are unchanged.
    #
    # Deliberately NOT the run-level default below. A named solver is a library
    # primitive a caller may drive without a run around it, and
    # ``spec_builder._solver_config_from_named`` overrides this value with
    # ``inputs.orientation_lock_strength`` whenever a run supplies one, so the
    # run-level knob -- not this one -- is what a production sweep locks with.
    orientation_lock_strength: float = 0.0


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HyperParams:
    """Training hyperparameters shared by every grid cell."""
    n_steps: int
    lr_start: float
    lr_end: float
    # lr_decay_start is a FRACTION of n_steps, in [0, 1], matches the
    # PretrainSpec / TrainingSpec convention in xcquinox.alec.config.
    lr_decay_start: float
    grad_clip: float
    gradnorm_alpha: float
    vxc_weight: float
    density_weight: float
    # Per-electron^2 normalization of the density channel (dpyscf
    # losses.py:171 convention; see losses._grid_term). Default False keeps
    # existing sweeps byte-identical.
    density_per_electron: bool = False
    # Decoupled L2 weight decay passed to train.build_optimizer (adamw). Default
    # 0.0 keeps existing sweeps byte-identical. 2026-06-20.
    weight_decay: float = 0.0
    # Held-out VALIDATION slice (WS3, 2026-06-20). The held-out reactions are
    # split val/test by eval_holdout.split_held_out(val_frac); the val slice
    # drives in-loop early-stop + validation-best model selection, the test slice
    # is what the held-out eval REPORTS. ALL default to a NO-OP so decay-free
    # runs stay byte-identical: validate_every=0 => no in-loop validation;
    # patience=0 => no early-stop.
    val_frac: float = 0.2
    validate_every: int = 0
    patience: int = 0
    early_stop_min_delta: float = 0.0
    # Periodic-resume checkpoint cadence (WS5, 2026-06-20). Default 0 => no-op
    # (no resume_* writes) so existing sweeps stay byte-identical. Threaded onto
    # every TrainingSpec by spec_builder like weight_decay / validate_every.
    checkpoint_every: int = 0
    pbe_anchor_weight: float = 0.0
    require_atom_anchors: bool = False
    seed: int = 42
    # Optimizer update scheme (2026-06-01). Defaults to the DFS/dpyscf-style
    # per-molecule stochastic updates (one optimizer step per training group per
    # epoch, fixed channel weights), the recommended default. Set "batched" to
    # use the historical full-batch + GradNorm path. See
    # xcquinox.alec.train._run_per_molecule_loop / TrainingSpec.update_scheme.
    update_scheme: str = "per_molecule"
    # Opt-in (default OFF): pad every molecule in a training group up to one common
    # shape so the de-fused per-molecule kernels collapse to one compile per spin-type
    # (RKS + UKS), bounding the JIT mmap footprint for large deep_attn subsets.
    # Results-neutral; see xcquinox.alec.padding. Off => byte-identical training.
    pad_group_to_common_shape: bool = False
    # Fixed per-channel weights for per_molecule mode (e.g.
    # {"loss_rho": 20.0, "loss_AE": 1.0}); empty -> train._DEFAULT_CHANNEL_WEIGHTS
    # (density-dominant, dpyscf-style). Stored sorted for determinism.
    channel_weights: tuple = ()


# ---------------------------------------------------------------------------
# Input paths
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class InputPaths:
    """Filesystem inputs for the harness.

    IMPORTANT: every path field below must be an ABSOLUTE path that resolves
    identically on the SLURM login node and on every compute node (i.e. it
    must live on a shared filesystem). Relative paths or login-node-only
    scratch paths will break compute-node tasks.
    """
    # ``external_refs_dir`` holds the per-species CCSD ``external_refs`` .npz
    # files. These ARE (re)computable by the harness (skip-if-cached), so the
    # directory is both a consumed input and a possible harness output.
    external_refs_dir: str
    # ``subset_ledger_path`` points at the EXISTING ``subset_index_log.json``
    # produced by the (already-finished) subset-selection pre-process. The
    # harness CONSUMES this ledger (and the per-spec ``subset.traj`` files
    # alongside it), it does NOT run subset selection, descriptor extraction,
    # or reference-histogram building. The ledger schema is
    # ``{"<metric>/<r>": {"chosen_indices": [...], "metric_value": float,
    # "point_kinds": [...], "point_names": [...], "tag": "bin01"}}``.
    subset_ledger_path: str
    basis: str
    grid_level: int
    output_root: str
    # Density fitting: when True, the SCF Coulomb is built from a 3-index
    # cderi (RI) instead of the full ERI, making larger bases memory-feasible.
    # ``auxbasis`` None -> auto-select from the orbital basis. Default off.
    density_fit: bool = False
    auxbasis: str | None = None
    # 2026-07-02: run-level orientation-lock strength (orientation_lock.py).
    # Authoritative for the WHOLE run -- threaded to the training/eval SCF (via
    # SolverConfig, in spec_builder), the CCSD reference generation (training
    # refs via external_refs.precompute_all + the held-out benchmark_refs job)
    # AND the pretraining-data identity (cluster/_datagen.py, cluster/inputs.py),
    # so the references, the pretraining rows and the functional lock the SAME
    # degenerate component of a radical (OH/CH/NO) and of an open p-shell atom.
    #
    # 2026-08-23: the default is the calibrated lock, not 0.0. It is the value
    # the production configurations carry and the value the data generator
    # builds at, and an unlocked degenerate open shell is not reproducible
    # between processes at all -- independent draws of the free O atom at grid
    # level 3 keep different numbers of rows and disagree at the 3e-7 Ha level
    # in the total energy. A run may still state 0.0 deliberately.
    orientation_lock_strength: float = DEFAULT_ORIENTATION_LOCK_STRENGTH
    # The one waiver of the data generator's irreproducible-degenerate
    # refusal, which fires when a spatially degenerate free atom is asked for
    # below grid level 3 or with the lock off. Both builds write a file whose
    # degenerate rows are one arbitrary member of a manifold under a manifest
    # that records a definite identity, so the waiver is deliberate, carries a
    # written reason and is recorded in the manifest. The shipped templates
    # and the pre-2026-08 campaigns run at grid level 1 or 2 and state it; a
    # grid-level-3 campaign does not need it.
    #
    # A waived run that is ALSO unlocked disagrees with its own certificate on
    # the degenerate atoms: ``fidelity`` evaluates them at the calibrated lock
    # whatever the run states, so the certificate would bound E_xc on a
    # density the network was not pretrained against (see the comment at that
    # branch in ``cluster/fidelity.py``).
    allow_irreproducible_degenerate: bool = False
    irreproducible_degenerate_reason: str | None = None
    # Hold-out benchmark reference-density dir (W4-11+BH76 pool). When set,
    # ``submit`` ALSO submits one standalone benchmark_refs job (CCSD + PBE
    # densities, no OEP) that starts once the train array has begun, and the
    # eval tasks export XCQUINOX_BENCH_REFS_DIR so the held-out eval picks up
    # whatever references exist at eval time. None (default) = feature off.
    benchmark_refs_dir: str | None = None
    # WS3 (2026-06-20): density-only SCF inputs for the VAL slice of the held-out
    # pools, precomputed at preflight and scored in-loop for early-stop /
    # validation-best selection. Mirrors ``benchmark_refs_dir``. None (default)
    # = no in-loop validation precompute.
    val_refs_dir: str | None = None
    # Per-rung SCF seeding (2026-08-14). "pbe" (default) keeps every arch on
    # the converged-PBE seed -- byte-identical to the pre-seeding protocol,
    # and deliberately NOT "auto": an auto default would silently convert a
    # pending arm (e.g. the v4 mgga stacks) to the new protocol on resubmit.
    # "auto" = rung-derived per arch (rungs.seed_xc_for_arch: the meta-GGA
    # family seeds from converged SCAN, everything else from PBE); "scan"
    # forces SCAN for every arch (controlled experiments only).
    seed_xc: str = "pbe"
    # Root of the SCAN seed cache (run_scf_with_cache layout: the per-species
    # npz files live under ``<seed_cache_dir>/_intermediates/``). Required
    # when any cell resolves a "scan" seed.
    seed_cache_dir: str | None = None


# ---------------------------------------------------------------------------
# Pretraining-fidelity certificate config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FidelityConfig:
    """Tolerances for the per-architecture physics certificate.

    Every architecture's pretrained networks must reproduce their parent
    functional (PBE for a GGA-rung architecture, SCAN for a meta-GGA one) in
    energy units before the run may train: the certificate
    (``cluster/fidelity.py``) requires max |dE_xc| over free atoms <=
    ``tol_atom`` mHa AND, on the atomization energies, the aggregate selected
    by ``tol_AE_aggregate`` -- ``"max"`` (the original rule) gates
    max |dAE| <= ``tol_AE`` kcal/mol; ``"mae"`` (2026-09-03) gates
    mean |dAE| <= ``tol_AE`` AND max |dAE| <= ``tol_AE_max_backstop``, all on
    frozen parent densities at the run's identity. The two-tier form exists
    because the max over ~23 correlated atomizations is a high-variance order
    statistic: a single species a few hundredths over the bound held a whole
    group's training, while the set-level fidelity (mean 0.1-0.3 kcal/mol on
    the 2026-09-02 cloning fits) was far inside chemical accuracy. The
    backstop keeps a worst-case ceiling so a set measure can never green-stamp
    one badly wrong species (a lone 4.6 kcal/mol offset at n=23 has MAE 0.30),
    and every certificate records the per-species offsets plus the list of
    species above 1.0 kcal/mol regardless of the aggregate.

    The defaults are the program's binding decision (1.0 kcal/mol and 1.0
    mHa, aggregate ``"max"``, backstop 2.0 -- the same 2.0 that is this
    validator's own loosening ceiling). ``validate_grid_semantics`` refuses
    any of the three tolerances above 2.0 unless ``override_reason`` is
    non-empty, so a run can only be loosened deliberately and with the reason
    on the record: the string is copied into every certificate the run writes.
    """
    tol_AE: float = 1.0          # kcal/mol, atomization-energy offset
    tol_atom: float = 1.0        # mHa, free-atom E_xc offset
    # How tol_AE is applied over the atomization set: "max" (each species
    # individually) or "mae" (the set mean, with tol_AE_max_backstop as the
    # per-species ceiling). The backstop key may only be written when the
    # aggregate is "mae"; under "max" it is inert and its presence is refused.
    tol_AE_aggregate: str = "max"
    tol_AE_max_backstop: float = 2.0   # kcal/mol, per-species ceiling under "mae"
    override_reason: str | None = None
    # When False the certificate is still computed and written with its TRUE
    # verdict, but the ON-NODE gates (the pretrain worker's exit code, the
    # train task, the preflight sweep) log the verdict and continue instead of
    # refusing. Permitted only with a non-empty ``override_reason``. It exists
    # for the per-architecture workflow-verification matrix, whose short
    # pretraining runs cannot meet the tolerance yet must exercise the train
    # and eval wiring with the physics on record. The RECORD layers
    # (``validate_run``, ``merge_v4_arms``, the figure loaders) ignore this
    # field and require PASS regardless, so a non-enforcing run can never
    # become a quantitative result.
    enforce: bool = True


# ---------------------------------------------------------------------------
# Model class
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    """The model class every architecture of the run is built as.

    ``parent_anchor`` anchors each architecture's enhancement networks to its
    parent functional (PBE on the GGA rungs, SCAN on the meta-GGA rungs) in
    the pre-image of the networks' bounded map, so the model equals its
    parent at initialization and the pretraining-fidelity certificate holds
    by construction (SPEC_parent_anchor.md). Applied wherever the run
    resolves an architecture -- the training specs, the pretrain stage, the
    certificate, the run validator -- through ``config.apply_model_block``,
    and recorded in the manifest, the pretrain metadata and the certificate.
    Requires the polarized correlation network (``use_polarized_correlation``
    at the run level or on the architecture); both rungs are accepted, the
    meta-GGA parents being ``parents.scan_fx`` / ``scan_fc``. An anchored
    configuration states its
    ``pretrain.energy_term_weight`` (0.0 is exact) without a sweep: the
    weight-zero refusal of ``validate_grid_semantics`` applies to unanchored
    configurations only.

    ``descriptor_coordinates`` selects the coordinates the networks' MLPs read
    a row in: ``"legacy"`` (today's layout, byte for byte) or ``"dfs"`` (the
    coordinate set of Dick and Fernandez-Serra, PRB 104, L161109 (2021), as
    ``networks.py`` states them). Both default to the pre-anchor model class.
    """
    parent_anchor: bool = False
    descriptor_coordinates: str = "legacy"


# ---------------------------------------------------------------------------
# Pretrain stage config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PretrainConfig:
    """Config for the pretraining workflow stage.

    Pretraining is now a harness STAGE: one ``run_pretrain`` job per distinct
    architecture, submitted up front, feeding every downstream train task. The
    stage builds a :class:`xcquinox.alec.config.PretrainSpec` per architecture
    from these parameters, runs it, and writes the resulting checkpoint into the
    run directory at ``<run_dir>/pretrain/<arch>/`` (see
    ``pretrain_checkpoint_dir``). Each train task then references that directory
    as its ``pretrain_checkpoint``, so the checkpoint is a harness PRODUCT, not
    a pre-staged input. Keeping it under ``run_dir`` (already unique per
    submission) co-locates every artifact for a run in one folder and keeps
    concurrent runs that pretrain the same architecture from clobbering each
    other.

    Defaults below mirror what the step-7 notebook's pretraining cell uses
    (see ``notebooks/_build_step7_notebook.py`` / ``_build_step6_notebook.py``):
    1000 pretraining steps, lr 1e-2 -> 1e-5, decay starting at 0.2 of the
    schedule, grad-clip 1.0, ``integration`` loss weighting (step-7's only
    pretrain origin, ``PRETRAIN_ORIGIN = "integration"``).

    ``data_dir`` (the pretraining INPUT dataset) has no sensible cross-cluster
    default and MUST be supplied as an ABSOLUTE shared-filesystem path.
    """
    data_dir: str
    n_steps: int = 1000              # (E) step-7 pretrain schedule length
    lr_start: float = 1e-2           # (E) step-7 pretrain lr start
    lr_end: float = 1e-5             # (E) step-7 pretrain lr floor
    # lr_decay_start is a FRACTION of n_steps, in [0, 1], matches the
    # PretrainSpec convention in xcquinox.alec.config.
    lr_decay_start: float = 0.2      # (E) step-7 pretrain decay onset
    grad_clip: float = 1.0           # (E) step-7 pretrain grad-clip
    seed: int = 42
    # PretrainSpec.loss_weighting is a str validated to {"unweighted",
    # "integration"}. Step-7 uses "integration" exclusively.
    loss_weighting: str = "integration"
    # Pretraining atom set as ((symbol, 2S-spin), ...). Empty tuple -> the
    # historical default (H, He, O, N). Extending coverage to every element
    # of the training pool (e.g. +Li, C, F, Na for dfs_step7) forces a
    # pretrain-data regen via the data manifest's "atoms" key.
    atoms: tuple = ()
    # --- Pretraining protocol (spec Sections 3.2, 6, 7) -------------------
    # The set. ``dfs_set`` adds the DFS pretraining inventory in its entirety
    # (8 free atoms and 22 G2/97 molecules for the GGA rung, 20 for the
    # meta-GGA rung); ``pool_atoms`` adds every single-atom species of the
    # BH76 and W4-11 pools with its production charge and spin. Turning either
    # on REPLACES the historical four-atom default, which ``atoms`` can still
    # extend. Both default False, so an existing YAML is unchanged.
    dfs_set: bool = False
    pool_atoms: bool = False
    # The density the targets sit on: "pbe", "scan", or "auto" for the
    # architecture's rung baseline. "pbe" is every file written before this
    # change; "auto" splits a mixed-rung sweep across two data files.
    parent_density: str = "pbe"
    # How OPEN-SHELL exchange rows are posed. "spin_channel" is the exact
    # spin-scaling footing the production UKS exchange evaluates, per channel
    # at (2 rho_sigma, 4 sigma_sigma_sigma, features of diag(P_sigma,
    # P_sigma)); "total" is the historical total-density footing. The footing
    # is part of the data's identity, so a change regenerates the file.
    exchange_footing: str = "total"
    # Share of the total integration weight carried by the synthetic
    # (r_s, s, alpha) mesh, which is kept as a regularizer only. Must equal
    # pretrain_data_gen.MESH_WEIGHT_FRACTION's historical 0.3 to reproduce
    # existing data; written as a literal because this module deliberately
    # imports neither JAX nor PySCF.
    mesh_fraction: float = 0.3
    # The objective. energy_term_weight is the weight of the per-system energy
    # term in inverse Hartree^2; 0.0 is the point-wise objective alone.
    energy_term_weight: float = 0.0
    # Validation and the stop criterion.
    validation_fraction: float = 0.0
    validation_seed: int = 0
    validate_every: int = 50
    patience: int = 0


# ---------------------------------------------------------------------------
# Cluster resources
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClusterResources:
    """SLURM resource requests + submission knobs."""
    partition: str
    time: str
    mem: str
    cpus_per_task: int
    array_throttle: int
    eval_array_throttle: int
    max_concurrent_tasks: int
    # max_array_size is a TASK COUNT. SLURM rejects an array whose largest
    # index is >= the cluster's MaxArraySize; with 0-based indexing a count
    # of N uses indices 0..N-1, so the count must be <= the cluster limit.
    # 1000 is a safe default for most SeaWulf-class clusters.
    max_array_size: int = 1000
    device: str = "cpu"            # "cpu" | "gpu" | "auto"
    gpus_per_task: int = 0
    conda_profile: str = ""
    conda_env: str = ""
    mail_user: str = ""
    mail_type: str = ""
    account: str = ""
    preflight_partition: str = ""
    preflight_time: str = ""
    eval_partition: str = ""
    eval_time: str = ""
    # Held-out eval parallelism: number of molecule-shard worker processes for
    # the held-out eval. None (default) = auto-detect the usable CPUs at runtime
    # (queue-agnostic, via parallel.detect_available_cpus) and parallelize over
    # them, degrading to serial on failure; 1 = force serial; N = explicit cap.
    eval_workers: int | None = None
    # Pretrain-stage resources. The pretrain stage is a small up-front array
    # (one task per distinct architecture). Each knob is None-by-default and
    # falls back to the train-array resource when unset, the same None-
    # fallback pattern ``oom_retry_*`` / ``timeout_retry_*`` use, resolved at
    # render time in submit.render_sbatch.
    pretrain_partition: str | None = None      # None -> cluster.partition
    pretrain_time: str | None = None           # None -> cluster.time
    pretrain_mem: str | None = None            # None -> cluster.mem
    pretrain_cpus_per_task: int | None = None  # None -> cluster.cpus_per_task
    # (E) pretrain_throttle: None means "run every distinct architecture
    # concurrently": the pretrain array is a handful of jobs, so the default
    # is the arch count (resolved in submit.render_sbatch as ARRAY_MAX + 1).
    pretrain_throttle: int | None = None
    # Datagen-stage resources (the front stage that generates the pretrain-data
    # file(s) before pretrain consumes them). Single job; each knob is None-by-
    # default and falls back to the pretrain knob, then the train-array cluster
    # value, in submit.render_sbatch.
    datagen_partition: str | None = None       # None -> pretrain_partition -> partition
    datagen_time: str | None = None            # None -> pretrain_time -> time
    datagen_mem: str | None = None             # None -> pretrain_mem -> mem
    datagen_cpus_per_task: int | None = None   # None -> pretrain_cpus_per_task -> cpus
    # Optional retry knobs, used when re-submitting a task that died from
    # OOM or wall-clock timeout. None = no dedicated retry config.
    oom_retry_partition: str | None = None
    oom_retry_mem: str | None = None
    # When True, an OOM retry is forced onto the CPU (``--gres=gpu:0`` releases
    # the GPU and ``JAX_PLATFORMS=cpu`` makes JAX ignore any still-visible GPU).
    # Without this, the retry resubmits the SAME gpu-rendered train script and
    # re-runs on a GPU, so a GPU-memory-bound spec just re-OOMs (CW2-M1). CPU
    # has far more RAM, so this is the real recovery path for GPU OOM.
    oom_retry_force_cpu: bool = False
    timeout_retry_partition: str | None = None
    timeout_retry_time: str | None = None
    # Per-stage node-allocation mode: "exclusive" books a whole node per array
    # task (``#SBATCH --nodes=1 --exclusive`` plus ``--mem=0`` to claim all of
    # the node's RAM -- an exclusive job that omits --mem is still cgroup-capped
    # at DefMemPerCPU*cpus-per-task on a memory-tracking SelectType) and "shared"
    # requests a cpu/mem slice (``#SBATCH --mem``).
    # Training peaks near a full node's memory, so every stage defaults to
    # whole-node; flip a stage to "shared" only when its tasks are small enough
    # to co-tenant a node. See ``submit.render_sbatch``.
    train_allocation: str = "exclusive"
    eval_allocation: str = "exclusive"
    preflight_allocation: str = "exclusive"
    pretrain_allocation: str = "exclusive"
    datagen_allocation: str = "exclusive"
    # Benchmark hold-out refs job (single job; submitted only when
    # inputs.benchmark_refs_dir is set). Falls back preflight -> train.
    benchmark_refs_partition: str | None = None
    benchmark_refs_time: str | None = None
    benchmark_refs_allocation: str = "exclusive"
    # Opt-in compile-smoke gate. When True the preflight compiles the heaviest
    # attention cell once on its exclusive node before the array launches and
    # blocks the array on a host-OOM (one cheap failure instead of every
    # large-basis task OOMing at XLA/LLVM compile time). Default False ->
    # byte-identical (no probe, no extra subprocess).
    preflight_compile_smoke: bool = False


# ---------------------------------------------------------------------------
# Aggregate config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GridConfig:
    """The complete harness config, aggregate of every section above."""
    sweep: SweepAxes
    # Named solver configs, keyed by the names used in the ``solver`` axis.
    solvers: dict[str, SolverNamed]
    hyperparams: HyperParams
    inputs: InputPaths
    pretrain: PretrainConfig
    cluster: ClusterResources
    domain_profile: str
    on_precompute_failure: str = "abort"   # {"abort","drop_failed_species"}
    bh76_mode: str = "reaction_energy"     # {"reaction_energy","barrier_height"}
    # DFS-domain AE points in predicted-atom reaction form (the converged
    # bh76w411/dpyscf construction) instead of fixed Chakravorty anchors.
    # Point names are unchanged, so subset ledgers resolve identically.
    ae_as_reactions: bool = False
    # Run-level toggle: when True, every architecture in the run is built
    # spin-polarization-aware (cnet input +1), so the UKS energy path uses the
    # zeta-dependent (spin-polarized) PW92c correlation baseline with the real
    # zeta. Default False -> byte-identical unpolarized behavior. Set via the
    # ``submit --polarized`` flag (or ``use_polarized_correlation: true`` in YAML).
    use_polarized_correlation: bool = False
    # When True, the held-out eval drops any reaction that shares a species with
    # the training subset (strict overlap filtering) so held-out = the true
    # complement with no leakage. Required for the representative-subset
    # (BH76+W4-11) runs where training subset + held-out partition one benchmark.
    held_out_strict: bool = False
    # Run-level toggle: when True, the eval array is NOT submitted up front. The
    # initial ``submit`` queues pretrain+preflight+train plus a tiny launcher job
    # (afterany on train) that submits the eval array only after train terminates,
    # shrinking the per-run queued-job footprint (relevant under SLURM per-user
    # submit caps). Default False -> byte-identical (eval submitted with the rest).
    # Set via the ``submit --defer-eval`` flag (or ``defer_eval: true`` in YAML).
    defer_eval: bool = False
    # ``inline_eval`` (2026-05-29): each train array task runs its own eval
    # immediately after training in the SAME SLURM task, instead of submitting
    # a separate eval array. 3-stage graph: pretrain -> preflight -> 
    # train+eval inline. Eliminates the inter-stage queue gap. Mutually
    # exclusive with defer_eval (an inline eval is the OPPOSITE of a deferred
    # eval, there IS no separate eval array to defer). Default False ->
    # byte-identical (eval as a separate array).
    inline_eval: bool = False
    # ``eval_coldstart`` (2026-08-14): when True, each spec's held-out eval
    # additionally writes the ``eval_holdout_coldstart`` channel -- the FINAL
    # checkpoint re-evaluated under a cold-start trajectory diagnostic
    # (seed_source="minao", max_cycles=25, conv_tol=1e-12; mode stays FULL).
    # Default False -> byte-identical (three channels as before).
    eval_coldstart: bool = False
    # Pretraining-fidelity certificate tolerances. Optional in the YAML: a
    # config written before the certificate existed loads at the binding
    # 1.0 kcal/mol / 1.0 mHa defaults rather than at no tolerance.
    fidelity: FidelityConfig = field(default_factory=FidelityConfig)
    # The model class (the parent anchor, the descriptor coordinates).
    # Optional in the YAML: a config written before the anchor existed loads
    # as the unanchored, legacy-coordinate class it was written for.
    model: ModelConfig = field(default_factory=ModelConfig)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _require(d: dict, key: str, ctx: str):
    """Fetch ``d[key]`` or raise a clear KeyError-style ValueError naming the
    missing key and the config section it belongs to."""
    if not isinstance(d, dict):
        raise ValueError(
            f"grid config section {ctx!r} must be a mapping, got "
            f"{type(d).__name__}"
        )
    if key not in d:
        raise ValueError(
            f"grid config is missing required key {ctx + '.' + key!r}"
        )
    return d[key]


#: Keys a block once read and no longer does. Accepted with a warning rather
#: than refused: ``pretrain.pretrain_root`` names the retired
#: ``<pretrain_root>/<run_id>/<arch>`` checkpoint layout (the stage now writes
#: under the run directory, see :func:`pretrain_checkpoint_dir`) and four
#: shipped configurations still state it, one of them tracked. A load that
#: fails on a key the harness itself once wrote is a worse failure than the
#: silence the refusal below exists to end.
_RETIRED_PRETRAIN_KEYS = ("pretrain_root",)


def _reject_unknown_keys(d, dc_type, ctx: str, *, retired=()):
    """Refuse a key this loader does not read, naming it and the accepted set.

    An unknown key is otherwise SILENTLY IGNORED: every builder below reads
    the keys it knows and nothing enumerates the rest, so ``pretrain:
    {patiense: 10}`` loads with ``patience`` at its default of 0 and the file
    reads as though it had set it. These blocks carry the system set, the
    schedule and the stop criterion, so a typo of that kind is a run other
    than the one written, with nothing anywhere reporting a problem.

    The accepted keys are the target dataclass's own FIELDS, which is exactly
    what ``__main__._config_to_raw_dict`` writes with ``dataclasses.asdict``:
    a ``resolved_config.yaml`` round trip is therefore accepted by
    construction, and a field added later is accepted without a second edit
    here. A block that is not a mapping is left to the builder's own check,
    which names the section and the type it got.

    A key spelled ``x-<something>`` is accepted anywhere and read by nothing:
    the closed schema otherwise leaves a YAML author nowhere to define an
    ANCHOR for reuse, or to park a note beside the values it describes, and
    the ``x-`` prefix is the established spelling for a field a schema does
    not own. It is an explicit statement that the key is not a harness knob,
    which a misspelling is not.
    """
    if not isinstance(d, dict):
        return
    accepted = frozenset(f.name for f in fields(dc_type))
    unknown = sorted(str(k) for k in d
                     if str(k) not in accepted and str(k) not in retired
                     and not str(k).startswith("x-"))
    if unknown:
        raise ValueError(
            f"grid config section {ctx!r} carries unknown key(s) "
            f"{', '.join(repr(k) for k in unknown)}; this loader reads only "
            f"{', '.join(sorted(accepted))}. An unread key is not applied, so "
            "the run would take the default of whatever the key was meant to "
            "set while the file appears to state it")
    for key in sorted(str(k) for k in d if str(k) in retired):
        warnings.warn(
            f"grid config key {ctx + '.' + key!r} is retired: the harness no "
            "longer reads it and its value has no effect on the run",
            stacklevel=2)


# ---------------------------------------------------------------------------
# Cluster walltimes
#
# A wall clock written into a hand-edited YAML does not survive the load
# unaided. ``yaml.safe_load`` applies the YAML 1.1 implicit resolvers, and an
# unquoted ``8:00:00`` matches the sexagesimal INTEGER resolver: it arrives as
# the base-60 integer 28800, is carried through ``ClusterResources.time``, and
# is substituted into ``#SBATCH --time=${TIME}`` by ``submit.render_sbatch``.
# SLURM reads a bare integer as MINUTES, so an 8-hour request is submitted as a
# 20-day one, with nothing anywhere reporting a problem. ``00:30:00`` survives
# only by accident -- the resolver requires a leading 1-9 -- so the defect is
# invisible in exactly the short-wall configs that are cheapest to test with.
#
# Every walltime field is therefore restored (when the loader mangled it) and
# checked against the shapes this harness accepts, before ``ClusterResources``
# is built. The check is exposed as ``normalize_cluster_walltimes`` so the
# paths that do not go through ``load_grid_config`` share it rather than
# restate it: ``workflow_matrix._restore_clock_strings`` applies it to the
# matrix template, which is read before any GridConfig exists, and
# ``__main__._apply_time_overrides`` applies it to the CLI ``--time`` walls,
# which reach ``cfg.cluster`` by ``dataclasses.replace``.
# ---------------------------------------------------------------------------

#: ``ClusterResources`` fields holding a SLURM wall clock. Derived from the
#: dataclass, so a wall added there later is covered without a second edit.
WALLTIME_FIELDS = tuple(
    f.name for f in fields(ClusterResources)
    if f.name == "time" or f.name.endswith("_time")
)

#: PyYAML's implicit resolver for a sexagesimal ``tag:yaml.org,2002:int``,
#: unsigned branch. A signed literal resolves too but is refused downstream, a
#: negative wall being no wall at all.
_SEXAGESIMAL_INT_RE = re.compile(r"^[1-9][0-9_]*(?::[0-5]?[0-9])+$")

#: The two shapes accepted as a wall clock: ``H:MM:SS`` (the hours field is
#: unbounded, so ``48:00:00`` is two days) and ``D-HH:MM:SS``. ``sbatch --time``
#: also accepts ``minutes``, ``minutes:seconds``, ``days-hours`` and
#: ``days-hours:minutes``; each is legal SLURM and each means something other
#: than the HH:MM:SS every walltime field of this harness is documented as, so
#: they are refused rather than guessed at. Quoting is no protection and
#: therefore not the criterion: ``time: "30"`` loads as a string and reaches
#: ``#SBATCH --time=30`` exactly as the unquoted ``time: 30`` does.
#:
#: Matching the shape is not sufficient, and the two further rules are checked
#: against the captured fields rather than by more regex: the wall must be a
#: NON-ZERO duration (``--time=0`` is SLURM for no limit, the opposite of the
#: bound a ``0:00:00`` looks like), and the hours field of the days form is a
#: time of day, 0-23 (``1-99:00:00`` is 99 hours into a day; the hours belong
#: in the days field rather than left to SLURM's normalisation).
_WALLTIME_RE = re.compile(
    r"^(?:(?P<days>[0-9]+)-(?P<day_hours>[0-9]{1,2})|(?P<hours>[0-9]+))"
    r":(?P<minutes>[0-5][0-9]):(?P<seconds>[0-5][0-9])$")

#: Upper bound of the hours field in ``D-HH:MM:SS``: a time of day, not a count.
_MAX_DAY_HOURS = 23


def _walltime_seconds(match) -> int:
    """Duration of a matched walltime, in seconds."""
    days = int(match.group("days") or 0)
    hours = int(match.group("day_hours") or match.group("hours") or 0)
    return ((days * 24 + hours) * 60 + int(match.group("minutes"))) * 60 \
        + int(match.group("seconds"))


def _sexagesimal_seconds(token: str):
    """Base-60 value of a YAML sexagesimal integer literal, or None.

    Reproduces the colon branch of PyYAML's ``construct_yaml_int``. Every field
    after the first is bounded at 59 by the resolver, so the value is the
    literal read as SECONDS whichever shape it was written in: ``8:00:00`` and
    ``480:00`` both give 28800. That coincidence is why the loaded integer on
    its own cannot say which literal produced it (see :func:`_restored_clock`).
    """
    if not _SEXAGESIMAL_INT_RE.match(token):
        return None
    total = 0
    for part in token.replace("_", "").split(":"):
        total = total * 60 + int(part)
    return total


#: The top-level ``cluster:`` header, in the spellings a hand-written config
#: uses: bare, quoted either way, and with a YAML anchor and/or a trailing
#: comment. A header this does NOT match yields no block, and the walltime is
#: refused -- scanning the rest of the document instead would let an unrelated
#: section supply the literal, which is the misreading this module exists to
#: prevent rather than a lenient fallback.
_CLUSTER_HEADER_RE = re.compile(
    r"""^(?P<q>["']?)cluster(?P=q):[ \t]*(?:&[^\s#]+[ \t]*)?(?:\#.*)?$""",
    re.M)


def _cluster_block(text: str):
    """The body of the top-level ``cluster:`` mapping in ``text``, or None.

    Restricting the literal scan to that block keeps it from reaching an
    identically named key in another section: with a decoy ``time: 8:00:00``
    elsewhere in the document, a document-wide scan accepts it as the literal
    behind an authored ``time: 28800`` -- 28800 minutes rendered as an 8-hour
    wall. None (rather than the whole text) is therefore returned when no
    header is recognised.
    """
    match = _CLUSTER_HEADER_RE.search(text)
    if match is None:
        return None
    rest = text[match.end():]
    end = re.search(r"^\S", rest, re.M)
    return rest[:end.start()] if end else rest


def _block_indent(block: str):
    """Indent of the block's own keys: that of its first key line, or None."""
    for line in block.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        return len(line) - len(line.lstrip(" \t"))
    return None


def _literal_tokens(block: str, key: str) -> list:
    """Every scalar written for ``<key>:`` at the block's own indent level.

    Only reached for a value that did NOT load as a string, so the scalar was
    unquoted and a ``#`` at a word boundary opens a YAML comment. Two rules
    make the match the key that was actually loaded rather than any line that
    reads like it: the indent must equal the block's first-key indent, so a
    ``time:`` nested one level deeper (under a ``notes:`` sub-mapping, say)
    supplies nothing, and the leading indent keeps ``time`` from matching
    ``pretrain_time``. Duplicates are returned rather than collapsed: YAML
    keeps the LAST of two identical keys while a scan reads the first, and
    ``8:00:00`` and ``480:00`` share the base-60 value 28800, so the two cannot
    be told apart by the value they loaded as.
    """
    indent = _block_indent(block)
    if indent is None:
        return []
    pattern = re.compile(
        rf"^[ \t]{{{indent}}}{re.escape(key)}:[ \t]*(.*)$", re.M)
    return [re.split(r"(?:^|\s)#", raw, maxsplit=1)[0].strip()
            for raw in pattern.findall(block)]


def _walltime_defect(key: str, value, source: str, detail: str, *,
                     advise_quoting: bool = True) -> ValueError:
    """The refusal for one walltime field, naming the key and what was read.

    ``advise_quoting`` is dropped for the refusals where the FORM is already
    right and the duration is what is wrong (a zero wall, an out-of-range
    days-hours field): telling the author to quote a value that is quoted and
    correctly shaped points at the wrong thing.
    """
    where = f" in {source}" if source else ""
    advice = (' Write it QUOTED, as "HH:MM:SS" (or "D-HH:MM:SS").'
              if advise_quoting else "")
    return ValueError(
        f"cluster.{key}{where} is {value!r}, which is not a usable SLURM "
        f"walltime: {detail}.{advice}"
    )


def _restored_clock(value: int, key: str, text: str, source: str) -> str:
    """The literal that a sexagesimal-resolved integer was written as.

    The integer alone does not determine it. PyYAML reads every colon-separated
    field in base 60, so ``8:00:00`` and ``480:00`` both arrive as 28800, and a
    deliberately bare ``28800`` -- which SLURM would read as 28800 minutes -- is
    indistinguishable from either. The literal is therefore read back out of the
    document and accepted only when its own base-60 value reproduces the loaded
    integer, which makes the restoration exact rather than inferred; where there
    is no such literal (a JSON config, or a genuinely bare integer) the field is
    refused instead of guessed at.
    """
    block = _cluster_block(text) if text else None
    if text and block is None:
        raise _walltime_defect(
            key, value, source,
            "the top-level cluster: block could not be located, so the literal "
            "this number was written as cannot be recovered from the document "
            "(the rest of the document is deliberately NOT searched: another "
            "section's clock-shaped value would supply a wall nobody wrote)")
    tokens = _literal_tokens(block, key) if block is not None else []
    if len(tokens) > 1:
        raise _walltime_defect(
            key, value, source,
            f"it is written {len(tokens)} times in the cluster block "
            f"({', '.join(repr(t) for t in tokens)}); YAML keeps the last, "
            "and spellings of one wall that differ in shape can share a "
            "base-60 value, so which was meant cannot be established")
    if tokens and _sexagesimal_seconds(tokens[0]) == value:
        return tokens[0]
    raise _walltime_defect(
        key, value, source,
        "SLURM reads a bare number as MINUTES while this field is documented "
        "as HH:MM:SS (an unquoted 8:00:00 also arrives here as a number: "
        "YAML 1.1 resolves it in base 60, to 28800)")


def _walltime_string(value, key: str, text: str, source: str):
    """One walltime field: restored if the loader mangled it, then checked.

    ``None`` and ``""`` pass through untouched on the per-stage walls: they are
    the unset sentinels those fields fall back from (to ``cluster.time``,
    resolved in ``submit.render_sbatch``), and ``_config_to_raw_dict`` writes
    them into every ``resolved_config.yaml`` the later stages re-read. On
    ``cluster.time`` there is nothing to fall back to -- an empty base wall
    renders a bare ``#SBATCH --time=`` -- so it is refused instead.
    """
    if value is None or value == "":
        if key == "time":
            raise _walltime_defect(
                key, value, source,
                "the base wall has no fallback (every per-stage wall falls "
                "back TO it)")
        return value
    if isinstance(value, str):
        candidate = value.strip()
    elif isinstance(value, bool):
        # bool is an int subclass; ``time: yes`` must not reach the arithmetic.
        raise _walltime_defect(key, value, source,
                               "a boolean is not a wall clock")
    elif isinstance(value, int):
        candidate = _restored_clock(value, key, text, source)
    else:
        # A sexagesimal FLOAT (``8:00:00.5`` -> 28800.5) lands here alongside a
        # plain float; neither has an accepted shape to be restored to.
        raise _walltime_defect(
            key, value, source,
            f"a {type(value).__name__} is not a wall clock")
    match = _WALLTIME_RE.match(candidate)
    if match is None:
        detail = 'the accepted shapes are "H:MM:SS" and "D-HH:MM:SS"'
        if candidate != value:
            # A restored literal: name it, or the message reports only the
            # base-60 integer the loader made of it.
            detail = f"it was written as {candidate!r} and " + detail
        raise _walltime_defect(key, value, source, detail)
    day_hours = match.group("day_hours")
    if day_hours is not None and int(day_hours) > _MAX_DAY_HOURS:
        raise _walltime_defect(
            key, value, source,
            f"the hours field of D-HH:MM:SS is a time of day, "
            f"0-{_MAX_DAY_HOURS}, not {int(day_hours)}; carry the excess in "
            "the days field", advise_quoting=False)
    if _walltime_seconds(match) == 0:
        raise _walltime_defect(
            key, value, source,
            "a zero wall is not a short wall: SLURM reads --time=0 as NO "
            "LIMIT", advise_quoting=False)
    return candidate


def normalize_cluster_walltimes(cluster: dict, *, text: str = "",
                                source: str = "") -> dict:
    """Return a copy of ``cluster`` whose walltime fields are checked strings.

    Args:
        cluster: the raw ``cluster`` mapping as parsed.
        text: the document it was parsed from, when there is one. A wall the
            YAML loader resolved to a number is restored from its literal
            there; without the text such a value is refused.
        source: the file named in a refusal message.

    Raises:
        ValueError: naming the key and the value read, for any field that is
            neither unset nor one of the accepted walltime shapes.
    """
    if not isinstance(cluster, dict):
        raise ValueError(
            f"grid config section 'cluster' must be a mapping, got "
            f"{type(cluster).__name__}"
        )
    out = dict(cluster)
    for key in WALLTIME_FIELDS:
        if key in out:
            out[key] = _walltime_string(out[key], key, text, source)
    return out


def _build_sweep(d: dict) -> SweepAxes:
    """Build SweepAxes from a raw dict; list fields become tuples."""
    _reject_unknown_keys(d, SweepAxes, "sweep")
    return SweepAxes(
        arch=tuple(_require(d, "arch", "sweep")),
        loss=tuple(_require(d, "loss", "sweep")),
        metric=tuple(_require(d, "metric", "sweep")),
        subset_size=tuple(_require(d, "subset_size", "sweep")),
        solver=tuple(_require(d, "solver", "sweep")),
    )


def _build_solvers(d: dict) -> dict[str, SolverNamed]:
    """Build the {name: SolverNamed} mapping from a raw nested dict."""
    if not isinstance(d, dict):
        raise ValueError(
            f"grid config 'solvers' must be a mapping of name -> solver "
            f"config, got {type(d).__name__}"
        )
    out: dict[str, SolverNamed] = {}
    for name, sd in d.items():
        ctx = f"solvers.{name}"
        # The OUTER mapping is free-form -- its keys are the solver names the
        # ``solver`` sweep axis references -- so only each named solver's own
        # block is closed.
        _reject_unknown_keys(sd, SolverNamed, ctx)
        mixer_kwargs = _parse_mixer_kwargs(sd.get("mixer_kwargs"), ctx)
        out[str(name)] = SolverNamed(
            mode=_require(sd, "mode", ctx),
            max_cycles=_require(sd, "max_cycles", ctx),
            feature_policy=sd.get("feature_policy"),
            scf_grad_checkpoint=bool(sd.get("scf_grad_checkpoint", False)),
            mixer_name=sd.get("mixer_name"),
            mixer_kwargs=mixer_kwargs,
            scf_loss_use_tail=bool(sd.get("scf_loss_use_tail", False)),
            scf_loss_tail=int(sd.get("scf_loss_tail", 10)),
            scf_loss_weight_power=float(sd.get("scf_loss_weight_power", 2.0)),
            orientation_lock_strength=float(
                sd.get("orientation_lock_strength", 0.0)),
        )
    return out


def _build_hyperparams(d: dict) -> HyperParams:
    ctx = "hyperparams"
    _reject_unknown_keys(d, HyperParams, ctx)
    return HyperParams(
        n_steps=_require(d, "n_steps", ctx),
        lr_start=_require(d, "lr_start", ctx),
        lr_end=_require(d, "lr_end", ctx),
        lr_decay_start=_require(d, "lr_decay_start", ctx),
        grad_clip=_require(d, "grad_clip", ctx),
        gradnorm_alpha=_require(d, "gradnorm_alpha", ctx),
        vxc_weight=_require(d, "vxc_weight", ctx),
        density_weight=_require(d, "density_weight", ctx),
        density_per_electron=bool(d.get("density_per_electron", False)),
        weight_decay=float(d.get("weight_decay", 0.0)),
        val_frac=float(d.get("val_frac", 0.2)),
        validate_every=int(d.get("validate_every", 0)),
        patience=int(d.get("patience", 0)),
        early_stop_min_delta=float(d.get("early_stop_min_delta", 0.0)),
        checkpoint_every=int(d.get("checkpoint_every", 0)),
        pbe_anchor_weight=d.get("pbe_anchor_weight", 0.0),
        require_atom_anchors=d.get("require_atom_anchors", False),
        seed=d.get("seed", 42),
        update_scheme=d.get("update_scheme", "per_molecule"),
        pad_group_to_common_shape=bool(d.get("pad_group_to_common_shape", False)),
        channel_weights=_parse_channel_weights(d.get("channel_weights")),
    )


def _parse_channel_weights(raw) -> tuple:
    """Accept either a {channel: weight} dict (user YAML) or a list of
    [channel, weight] pairs (round-tripped resolved_config, where
    dataclasses.asdict turned the tuple into nested lists). Return a
    deterministic sorted tuple of (str, float) pairs."""
    if not raw:
        return ()
    items = raw.items() if isinstance(raw, dict) else raw
    return tuple(sorted((str(k), float(v)) for k, v in items))


def _orientation_lock(d, ctx: str) -> float:
    """The run-level orientation-lock strength, bounded at parse.

    An absent key and an explicit null both mean "not stated", which is the
    calibrated lock; only a written number is a statement, and a written 0.0
    is the statement that the run is unlocked.

    The value is read through the house number reader rather than with a bare
    ``float()`` because it is an identity-bearing physical quantity -- the
    coefficient on the traceless-quadrupole h_core bias that the CCSD
    references, the training SCF and the pretraining rows are all built at --
    and the coercions that reader exists to refuse all reach it: ``float(True)``
    is 1.0, four orders above the calibrated strength; ``float(None)`` and a
    list raise ``TypeError``, which passes every ``except ValueError`` handler
    in the load path; and a NaN escapes any bound written against it.

    Negative is refused HERE rather than downstream. It is not zero, so the
    generator's degeneracy refusal does not fire on it, and a degenerate
    atom's rows would be written under a sign-flipped bias while the manifest
    recorded a legitimate identity; ``SolverConfig`` catches it later, in the
    spec builder on a compute node, after the data is on disk. There is no
    upper bound: no measurement anchors one, and the lock is an energy
    coefficient whose only hard requirement is that it be finite and not
    reverse the operator.
    """
    if d.get("orientation_lock_strength") is None:
        return float(DEFAULT_ORIENTATION_LOCK_STRENGTH)
    return _config_number(d, "orientation_lock_strength",
                          DEFAULT_ORIENTATION_LOCK_STRENGTH, minimum=0,
                          ctx=ctx)


def _build_inputs(d: dict) -> InputPaths:
    ctx = "inputs"
    _reject_unknown_keys(d, InputPaths, ctx)
    # The waiver of the data generator's irreproducible-degenerate refusal
    # carries prose, for the reason ``fidelity.override_reason`` does: it
    # authorises a pretraining file whose degenerate-atom rows are one
    # arbitrary member of a manifold under a manifest that records a definite
    # identity, which is the defect the manifest exists to exclude. A
    # non-string is refused rather than coerced -- ``str(False)`` is the
    # non-empty string ``'False'``, so ``irreproducible_degenerate_reason:
    # false`` would otherwise read as a written reason -- and a blank string
    # is not a reason either.
    allow_irreproducible = _config_bool(
        d, "allow_irreproducible_degenerate", False, ctx=ctx)
    reason = d.get("irreproducible_degenerate_reason")
    if reason is not None and not isinstance(reason, str):
        raise ValueError(
            f"grid config key '{ctx}.irreproducible_degenerate_reason' must "
            f"be a string or null, got {type(reason).__name__} ({reason!r}); "
            "a boolean or a number is not a reason. The value states why this "
            "run's pretraining data may be built at an identity that does not "
            "reproduce between processes")
    if allow_irreproducible and not (reason or "").strip():
        raise ValueError(
            f"grid config key '{ctx}.allow_irreproducible_degenerate' is "
            f"true, so '{ctx}.irreproducible_degenerate_reason' must be a "
            f"non-empty string, got {reason!r}. The waiver permits a "
            "pretraining file whose spatially degenerate free atoms' rows are "
            "one arbitrary member of their manifold -- below grid level 3 the "
            "quadrature does not resolve the term, and with the lock off the "
            "SCF may land anywhere on it -- so the run record states why that "
            "was acceptable")
    seed_xc = str(d.get("seed_xc", "pbe"))
    if seed_xc not in ("pbe", "scan", "auto"):
        raise ValueError(
            f"{ctx}.seed_xc must be one of 'pbe'/'scan'/'auto', got "
            f"{seed_xc!r}"
        )
    return InputPaths(
        external_refs_dir=_require(d, "external_refs_dir", ctx),
        subset_ledger_path=_require(d, "subset_ledger_path", ctx),
        basis=_require(d, "basis", ctx),
        grid_level=_require(d, "grid_level", ctx),
        output_root=_require(d, "output_root", ctx),
        density_fit=bool(d.get("density_fit", False)),
        auxbasis=d.get("auxbasis"),
        orientation_lock_strength=_orientation_lock(d, ctx),
        allow_irreproducible_degenerate=allow_irreproducible,
        irreproducible_degenerate_reason=reason,
        benchmark_refs_dir=d.get("benchmark_refs_dir"),
        val_refs_dir=d.get("val_refs_dir"),
        seed_xc=seed_xc,
        seed_cache_dir=d.get("seed_cache_dir"),
    )


# The pretraining-protocol string knobs' allowed sets, named once for the
# parser and for the semantic check below. The library side states the parent
# densities as ``config.PARENT_DENSITIES`` and the seed ceiling as
# ``config.MAX_SEED`` rather than importing them from here, because
# ``xcquinox.alec.config`` pulls JAX and equinox and the harness parser runs on
# the login node; the two pairs are pinned equal by
# ``test_the_parent_density_set_is_stated_once`` /
# ``test_the_seed_range_is_stated_once``, so a value one layer admits and the
# other refuses cannot ship.
_PARENT_DENSITIES = ("pbe", "scan", "auto")
_EXCHANGE_FOOTINGS = ("total", "spin_channel")
_LOSS_WEIGHTINGS = ("unweighted", "integration")
# jax.random.PRNGKey wraps modulo 2**32 instead of raising, so a seed outside
# that range silently ALIASES another run's initialization (measured:
# PRNGKey(-1) == PRNGKey(2**32 - 1), PRNGKey(2**32) == PRNGKey(0)) while the
# metadata records the number that was written. create_network_pair keys cnet
# at seed + 1, so the top of the range is excluded too.
_MAX_SEED = 2 ** 32 - 2
# The grid level at and above which a spatially degenerate free atom's rows
# reproduce between processes, restated here from
# ``pretrain_data_gen.COARSE_DEGENERATE_MIN_GRID_LEVEL`` for the reason the
# parent-density set is restated: that module pulls JAX and PySCF, and this
# parser runs on the login node. The two are pinned equal by
# ``test_the_reproducible_grid_level_is_stated_once``, so a level one layer
# waives and the other refuses cannot ship.
_MIN_REPRODUCIBLE_GRID_LEVEL = 3


def _config_bool(d, key: str, default: bool, ctx: str = "pretrain") -> bool:
    """Read one switch out of a raw config section; a non-boolean is an error.

    Refused rather than coerced, as ``fidelity.enforce`` is: ``bool("false")``
    is True -- and so are the quoted forms of YAML 1.1's ``no`` and ``0`` --
    so coercion turns the switch ON in a config whose author wrote it OFF,
    while ``bool(None)`` (an empty ``dfs_set:``) reads as OFF without remark.
    The pretraining switches select the SYSTEM SET, so either misreading
    changes what the run fits, and ``inputs.allow_irreproducible_degenerate``
    grants permission to write a file whose identity is not reproducible.
    """
    v = d.get(key, default)
    if not isinstance(v, bool):
        raise ValueError(
            f"grid config key '{ctx}.{key}' must be a boolean "
            f"(true/false), got {type(v).__name__} ({v!r})")
    return v


def _config_number(d, key: str, default, whole: bool = False,
                   minimum=None, maximum=None, minimum_open: bool = False,
                   maximum_open: bool = False, ctx: str = "pretrain"):
    """Read one number out of a raw config section (``pretrain`` by default).

    The reasoning of ``_fidelity_tolerance``: a boolean or a container is a
    config error rather than something to coerce -- ``float(True)`` is 1.0 and
    ``int(True)`` is 1 (silently a weight of one, or one validation every
    step), and ``float(None)`` raises ``TypeError``, which passes every
    ``except ValueError`` handler in the load path and surfaces as a crash
    naming no key. Integers, floats and numeric strings remain valid.

    Non-finite values are refused for the reason the certificate tolerances
    are: NaN escapes an ordinary bound in whichever direction it is written,
    so the value would load with no complaint and every comparison against it
    downstream would be False as well.

    With ``whole=True`` the value must be an exact integer: ``int(2.5)``
    truncates to 2, which is a validation or early-stopping schedule other
    than the one written. An integer (or an integer-spelling string) is read
    WITHOUT a float round trip, because ``float(2**53 + 1)`` is ``2**53`` and
    the round trip would load a step count other than the one written.

    ``minimum`` / ``maximum`` (with ``minimum_open`` / ``maximum_open`` for a
    strict bound) carry the range the CONSUMER needs, so a value that would
    make the run a no-op is refused at load rather than at step 1 of a queued
    job. Every such bound is stated HERE as well as in
    ``validate_grid_semantics``: the semantic check runs on the login node at
    submit, while the datagen, pretrain and preflight workers reach their
    configuration through ``load_grid_config`` alone, so a bound that lives
    only in the semantic check does not exist for the process that runs the
    schedule.
    """
    v = d.get(key, default)
    kind = "a whole number" if whole else "a number"
    if isinstance(v, bool) or not isinstance(v, (int, float, str)):
        raise ValueError(
            f"grid config key '{ctx}.{key}' must be {kind}, got "
            f"{type(v).__name__} ({v!r})")
    out = None
    if whole and isinstance(v, int):
        out = v
    elif whole and isinstance(v, str):
        try:
            out = int(v.strip())
        except ValueError:
            out = None
    if out is None:
        try:
            out = float(v)
        except ValueError:
            raise ValueError(
                f"grid config key '{ctx}.{key}' must be {kind}, got "
                f"{type(v).__name__} ({v!r})") from None
        if not math.isfinite(out):
            raise ValueError(
                f"grid config key '{ctx}.{key}' must be a FINITE number, "
                f"got {v!r}; a NaN value satisfies neither side of the bound "
                "it is checked against and turns every comparison against it "
                "into the sense of that comparison rather than a measurement")
        if whole and out != int(out):
            raise ValueError(
                f"grid config key '{ctx}.{key}' must be a whole number, "
                f"got {v!r}; int() would truncate it to {int(out)} and run a "
                "schedule other than the one written")
    if minimum is not None and (out <= minimum if minimum_open
                                else out < minimum):
        raise ValueError(
            f"grid config key '{ctx}.{key}' must be "
            f"{'>' if minimum_open else '>='} {minimum}, got {v!r}")
    if maximum is not None and (out >= maximum if maximum_open
                                else out > maximum):
        raise ValueError(
            f"grid config key '{ctx}.{key}' must be "
            f"{'<' if maximum_open else '<='} {maximum}, got {v!r}")
    return int(out) if whole else out


def _pretrain_choice(d, key: str, default: str, allowed) -> str:
    """Read one pretraining string knob and test it against its allowed set.

    ``str(None)`` is the non-empty string ``'None'`` and ``str(True)`` is
    ``'True'``, so coercion carries an empty key or a typo past the parse. The
    member test lives here as well as in ``validate_grid_semantics`` so that
    ``load_grid_config`` refuses an unknown value whether or not the semantic
    check is reached.
    """
    v = d.get(key, default)
    if not isinstance(v, str) or v not in allowed:
        raise ValueError(
            f"grid config key 'pretrain.{key}' must be one of "
            f"{', '.join(repr(a) for a in allowed)}, got "
            f"{type(v).__name__} ({v!r})")
    return v


def _build_pretrain_from(raw: dict, *, source=None) -> PretrainConfig:
    """Build the pretrain block, naming ``source`` in any retired-key warning.

    A retired key warns rather than refuses, and the warning is only useful if
    it says WHICH file still carries the key; the block builder itself does
    not know the file, so the re-emission happens here."""
    d = _require(raw, "pretrain", "<root>")
    if source is None:
        return _build_pretrain(d)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = _build_pretrain(d)
    for w in caught:
        message = str(w.message)
        if "is retired" in message:
            message = f"{message} (in {source})"
        warnings.warn_explicit(message, w.category, str(source), 0)
    return cfg


def _build_pretrain(d: dict) -> PretrainConfig:
    ctx = "pretrain"
    _reject_unknown_keys(d, PretrainConfig, ctx,
                         retired=_RETIRED_PRETRAIN_KEYS)
    return PretrainConfig(
        data_dir=_require(d, "data_dir", ctx),
        n_steps=_config_number(d, "n_steps", 1000, whole=True, minimum=0,
                               minimum_open=True),
        # A non-positive Adam rate is a no-op (0) or an ascent (< 0), so the
        # start is strictly positive; lr_end may be 0, which is a linear
        # anneal to zero and a legitimate schedule.
        lr_start=_config_number(d, "lr_start", 1e-2, minimum=0,
                                minimum_open=True),
        lr_end=_config_number(d, "lr_end", 1e-5, minimum=0),
        # A FRACTION of n_steps, not a step count: decay_start_step =
        # int(lr_decay_start * n_steps).
        lr_decay_start=_config_number(d, "lr_decay_start", 0.2, minimum=0,
                                      maximum=1),
        # optax.clip_by_global_norm(0.0) zeroes every gradient and (-1.0)
        # reverses it; neither is a run, and the consumer has no None branch
        # (a None reaches the update and raises there, not at load).
        grad_clip=_config_number(d, "grad_clip", 1.0, minimum=0,
                                 minimum_open=True),
        seed=_config_number(d, "seed", 42, whole=True, minimum=0,
                            maximum=_MAX_SEED),
        loss_weighting=_pretrain_choice(
            d, "loss_weighting", "integration", _LOSS_WEIGHTINGS),
        atoms=_parse_pretrain_atoms(d.get("atoms")),
        dfs_set=_config_bool(d, "dfs_set", False),
        pool_atoms=_config_bool(d, "pool_atoms", False),
        parent_density=_pretrain_choice(
            d, "parent_density", "pbe", _PARENT_DENSITIES),
        exchange_footing=_pretrain_choice(
            d, "exchange_footing", "total", _EXCHANGE_FOOTINGS),
        # The mesh's SHARE of the total integration weight: strictly between
        # 0 and 1, the range pretrain_data_gen._check_generator_arguments (the
        # only consumer) requires. A share of 0 is not a mesh and a share of 1
        # leaves the atomic grids no weight at all.
        mesh_fraction=_config_number(d, "mesh_fraction", 0.3, minimum=0,
                                     minimum_open=True, maximum=1,
                                     maximum_open=True),
        # A loss weight, in inverse Hartree^2: 0 turns the energy term off, a
        # negative one rewards the network for getting the energy wrong.
        energy_term_weight=_config_number(d, "energy_term_weight", 0.0,
                                          minimum=0),
        # A FRACTION of the multi-nucleus systems: 0 = no split; 1 would hold
        # out the whole set and fit nothing.
        validation_fraction=_config_number(d, "validation_fraction", 0.0,
                                           minimum=0, maximum=1,
                                           maximum_open=True),
        # The held-out permutation's seed, bounded like the initialization
        # seed and for the same reason (see _MAX_SEED).
        validation_seed=_config_number(d, "validation_seed", 0, whole=True,
                                       minimum=0, maximum=_MAX_SEED),
        # Optimizer steps between validations: 0 or negative is not a period.
        validate_every=_config_number(d, "validate_every", 50, whole=True,
                                      minimum=0, minimum_open=True),
        # Validations without improvement before the stop; 0 = no early stop.
        patience=_config_number(d, "patience", 0, whole=True, minimum=0),
    )


def _fidelity_tolerance(d, key: str, default: float = 1.0) -> float:
    """Read one certificate tolerance out of a raw ``fidelity`` mapping.

    A tolerance is an energy bound, so a boolean or a container is a config
    error rather than something to coerce: ``float(True)`` is 1.0 (silently
    the binding tolerance) and ``float(None)`` raises ``TypeError``, which
    passes every ``except ValueError`` handler in the load path. Integers,
    floats and numeric strings remain valid.

    The value must also be FINITE. NaN escapes the bounds in
    ``validate_grid_semantics`` outright -- ``nan <= 0`` and ``nan > 2.0`` are
    both False, so a NaN tolerance loads with no override_reason and no
    complaint -- and every downstream comparison against it is False too, so
    the certificate verdict it produces is whatever the sense of that
    comparison happens to be rather than a measurement. The infinities are
    caught downstream (``-inf`` by the positivity floor, ``+inf`` by the 2.0
    ceiling) and are refused here for the same reason: a tolerance is a finite
    energy bound.
    """
    v = d.get(key, default)
    if isinstance(v, bool) or not isinstance(v, (int, float, str)):
        raise ValueError(
            f"grid config key 'fidelity.{key}' must be a number (kcal/mol for "
            f"tol_AE, mHa for tol_atom), got {type(v).__name__} ({v!r})")
    try:
        out = float(v)
    except ValueError:
        raise ValueError(
            f"grid config key 'fidelity.{key}' must be a number (kcal/mol for "
            f"tol_AE, mHa for tol_atom), got {type(v).__name__} "
            f"({v!r})") from None
    if not math.isfinite(out):
        raise ValueError(
            f"grid config key 'fidelity.{key}' must be a FINITE number "
            f"(kcal/mol for tol_AE, mHa for tol_atom), got {v!r}; a NaN "
            "tolerance satisfies neither the positivity floor nor the 2.0 "
            "ceiling and turns every certificate comparison against it into "
            "the sense of that comparison rather than a measurement")
    return out


#: The coordinate sets the networks' MLPs can read a row in. Stated here as
#: well as in ``config.DESCRIPTOR_COORDINATES`` for the reason
#: ``_PARENT_DENSITIES`` is: this parser runs on the login node without the
#: library; the two are pinned equal by the test suite.
_DESCRIPTOR_COORDINATES = ("legacy", "dfs")


def _build_model_block(d) -> ModelConfig:
    """Build ModelConfig from a raw dict; ``None`` -> the defaults.

    The ``model`` section is OPTIONAL so every YAML authored before the parent
    anchor existed still loads, as the unanchored legacy-coordinate class. Its
    keys are refused rather than coerced, like ``fidelity.enforce``:
    ``bool("false")`` is True and ``bool(None)`` is False, so a coerced
    ``parent_anchor`` would build a model class no configuration asked for.
    """
    if d is None:
        return ModelConfig()
    if not isinstance(d, dict):
        raise ValueError(
            f"grid config section 'model' must be a mapping, got "
            f"{type(d).__name__}")
    _reject_unknown_keys(d, ModelConfig, "model")
    anchor = d.get("parent_anchor", False)
    if not isinstance(anchor, bool):
        raise ValueError(
            f"grid config key 'model.parent_anchor' must be a boolean "
            f"(true/false), got {type(anchor).__name__} ({anchor!r}); the "
            "value selects the model class every architecture of the run is "
            "built as and is not coerced")
    coords = d.get("descriptor_coordinates", "legacy")
    if not isinstance(coords, str) or coords not in _DESCRIPTOR_COORDINATES:
        raise ValueError(
            f"grid config key 'model.descriptor_coordinates' must be one of "
            f"{', '.join(repr(v) for v in _DESCRIPTOR_COORDINATES)}, got "
            f"{coords!r}")
    return ModelConfig(parent_anchor=anchor, descriptor_coordinates=coords)


def _build_fidelity(d) -> FidelityConfig:
    """Build FidelityConfig from a raw dict; ``None`` -> the defaults.

    The ``fidelity`` section is OPTIONAL so every YAML authored before the
    certificate existed still loads, at the binding tolerances.
    """
    if d is None:
        return FidelityConfig()
    if not isinstance(d, dict):
        raise ValueError(
            f"grid config section 'fidelity' must be a mapping, got "
            f"{type(d).__name__}")
    _reject_unknown_keys(d, FidelityConfig, "fidelity")
    reason = d.get("override_reason")
    # A non-string reason is REFUSED, never coerced. str(False) is the
    # non-empty string 'False', so coercion would let `override_reason: false`
    # -- and its YAML 1.1 synonym `no`, and a bare `0` -- satisfy the
    # non-empty-reason test in validate_grid_semantics and authorise a
    # loosened tolerance, the opposite of what such an author wrote. The
    # reason is prose copied verbatim into every certificate the run writes.
    if reason is not None and not isinstance(reason, str):
        raise ValueError(
            f"grid config key 'fidelity.override_reason' must be a string or "
            f"null, got {type(reason).__name__} ({reason!r}); a boolean or a "
            "number is not a reason. The value authorises a loosened "
            "certificate tolerance (or disabled on-node gates) and is "
            "recorded in every certificate the run writes")
    # enforce is likewise refused rather than coerced: bool(None) is False, so
    # an empty `enforce:` (a YAML null) would DISABLE the on-node gates in a
    # config that never asked for it -- unremarked whenever an override_reason
    # is present for some other purpose -- and bool("false") is True, which
    # contradicts the author the other way.
    enforce = d.get("enforce", True)
    if not isinstance(enforce, bool):
        raise ValueError(
            f"grid config key 'fidelity.enforce' must be a boolean "
            f"(true/false), got {type(enforce).__name__} ({enforce!r})")
    # The aggregate is a CHOICE, not a number: refused unless it names one of
    # the two rules the certificate implements, so a typo cannot silently gate
    # on the default while the author believes the set rule is in force.
    aggregate = d.get("tol_AE_aggregate", "max")
    if not isinstance(aggregate, str) or aggregate not in ("max", "mae"):
        raise ValueError(
            f"grid config key 'fidelity.tol_AE_aggregate' must be 'max' or "
            f"'mae', got {aggregate!r}")
    # Under "max" the backstop gates nothing; a config that writes it anyway
    # believes it is bounding something, so the combination is refused rather
    # than carried inert.
    if "tol_AE_max_backstop" in d and aggregate == "max":
        raise ValueError(
            "grid config key 'fidelity.tol_AE_max_backstop' is only "
            "meaningful under fidelity.tol_AE_aggregate 'mae'; with the "
            "aggregate 'max' the per-species ceiling IS tol_AE and the "
            "backstop key gates nothing")
    return FidelityConfig(
        tol_AE=_fidelity_tolerance(d, "tol_AE"),
        tol_atom=_fidelity_tolerance(d, "tol_atom"),
        tol_AE_aggregate=aggregate,
        tol_AE_max_backstop=_fidelity_tolerance(
            d, "tol_AE_max_backstop", default=2.0),
        override_reason=reason,
        enforce=enforce,
    )


def _parse_mixer_kwargs(raw, ctx):
    """Normalize a named solver's ``mixer_kwargs`` to a hashable, sorted
    tuple-of-(str, float) pairs, or ``None`` when absent.

    Accepts either a ``{name: value}`` dict (the user-authored YAML form) or a
    list/tuple of ``[name, value]`` pairs -- the latter is the round-tripped
    ``resolved_config.yaml`` form: ``submit`` serializes ``SolverNamed`` via
    ``dataclasses.asdict`` (keeping the tuple-of-pairs) and ``yaml.safe_dump``
    writes tuples as YAML sequences, so a reload (datagen/pretrain/preflight/
    eval) parses them back as nested lists. Mirrors ``_parse_channel_weights``
    / ``_parse_pretrain_atoms``, which solve the same dict<->list round-trip;
    NOT handling the list form is what crashed datagen with
    ``mixer_kwargs must be a mapping, got list``.

    ``None``/empty -> ``None`` so ``SolverConfig`` keeps its default mixer
    kwargs unless a solver explicitly overrides them.
    """
    if not raw:
        return None
    if isinstance(raw, dict):
        items = raw.items()
    elif isinstance(raw, (list, tuple)):
        items = raw
    else:
        raise ValueError(
            f"{ctx}.mixer_kwargs must be a mapping or a list of "
            f"[name, value] pairs, got {type(raw).__name__}"
        )
    return tuple(sorted((str(k), float(v)) for k, v in items))


def _parse_pretrain_atoms(raw) -> tuple:
    """Normalize the YAML ``pretrain.atoms`` value to ((symbol, spin), ...).

    Accepts a {symbol: spin} dict or a list of [symbol, spin] pairs (the
    round-tripped resolved_config form). Empty/None -> () (the generator's
    DEFAULT_PRETRAIN_ATOMS applies)."""
    if not raw:
        return ()
    if isinstance(raw, dict):
        return tuple((str(sym), int(sp)) for sym, sp in raw.items())
    return tuple((str(pair[0]), int(pair[1])) for pair in raw)


def _build_cluster(d: dict, *, text: str = "",
                   source: str = "") -> ClusterResources:
    """Build ClusterResources from a raw ``cluster`` mapping.

    ``text``/``source`` are the document the mapping was parsed from and its
    path; they let :func:`normalize_cluster_walltimes` restore a wall clock the
    YAML loader resolved to a number. Both default to empty, so a caller
    holding only a mapping (a JSON config, a test) still gets the shape check,
    with any number refused rather than restored.
    """
    ctx = "cluster"
    _reject_unknown_keys(d, ClusterResources, ctx)
    d = normalize_cluster_walltimes(d, text=text, source=source)
    return ClusterResources(
        partition=_require(d, "partition", ctx),
        time=_require(d, "time", ctx),
        # mem is OPTIONAL, whole-node/exclusive stages emit no --mem at all,
        # and a shared stage that omits it lets SLURM apply the partition
        # default-mem-per-cpu. Absent -> "" (no directive rendered).
        mem=d.get("mem", ""),
        cpus_per_task=_require(d, "cpus_per_task", ctx),
        array_throttle=_require(d, "array_throttle", ctx),
        eval_array_throttle=_require(d, "eval_array_throttle", ctx),
        max_concurrent_tasks=_require(d, "max_concurrent_tasks", ctx),
        max_array_size=d.get("max_array_size", 1000),
        device=d.get("device", "cpu"),
        gpus_per_task=d.get("gpus_per_task", 0),
        conda_profile=d.get("conda_profile", ""),
        conda_env=d.get("conda_env", ""),
        mail_user=d.get("mail_user", ""),
        mail_type=d.get("mail_type", ""),
        account=d.get("account", ""),
        preflight_partition=d.get("preflight_partition", ""),
        preflight_time=d.get("preflight_time", ""),
        eval_partition=d.get("eval_partition", ""),
        eval_time=d.get("eval_time", ""),
        eval_workers=d.get("eval_workers"),
        pretrain_partition=d.get("pretrain_partition"),
        pretrain_time=d.get("pretrain_time"),
        pretrain_mem=d.get("pretrain_mem"),
        pretrain_cpus_per_task=d.get("pretrain_cpus_per_task"),
        pretrain_throttle=d.get("pretrain_throttle"),
        datagen_partition=d.get("datagen_partition"),
        datagen_time=d.get("datagen_time"),
        datagen_mem=d.get("datagen_mem"),
        datagen_cpus_per_task=d.get("datagen_cpus_per_task"),
        oom_retry_partition=d.get("oom_retry_partition"),
        oom_retry_mem=d.get("oom_retry_mem"),
        oom_retry_force_cpu=bool(d.get("oom_retry_force_cpu", False)),
        timeout_retry_partition=d.get("timeout_retry_partition"),
        timeout_retry_time=d.get("timeout_retry_time"),
        train_allocation=d.get("train_allocation", "exclusive"),
        eval_allocation=d.get("eval_allocation", "exclusive"),
        preflight_allocation=d.get("preflight_allocation", "exclusive"),
        pretrain_allocation=d.get("pretrain_allocation", "exclusive"),
        datagen_allocation=d.get("datagen_allocation", "exclusive"),
        benchmark_refs_partition=d.get("benchmark_refs_partition"),
        benchmark_refs_time=d.get("benchmark_refs_time"),
        benchmark_refs_allocation=d.get("benchmark_refs_allocation",
                                        "exclusive"),
        preflight_compile_smoke=bool(d.get("preflight_compile_smoke", False)),
    )


def _resolve_eval_workers(cl: ClusterResources, *, n_molecules: int) -> int:
    """Resolve the top of the held-out-eval worker ladder.

    ``cl.eval_workers`` (when set) is an explicit cap; otherwise auto-detect the
    CPUs this process may actually use at runtime (queue-agnostic). Capped at
    ``n_molecules`` (no point spawning more workers than molecules) and floored
    at 1. A returned value of 1 means serial (the ladder is empty)."""
    from xcquinox.alec import parallel
    base = cl.eval_workers if cl.eval_workers else parallel.detect_available_cpus()
    return max(1, min(int(base), max(1, n_molecules)))


def _top_level_bh76_mode_count(text: str, path: str) -> int:
    """How many times ``bh76_mode`` appears as a ROOT-mapping key in ``text``.

    Structural, not textual: ``yaml.compose`` (or a JSON ``object_pairs_hook``
    for ``.json`` paths) sees the parsed document's root mapping, so a quoted
    key, a uniformly indented document, and a JSON duplicate all count, while
    a commented line or a column-0 flow-sequence continuation that merely
    begins with the key's spelling does not. A column-0 regex failed all four
    of those cases. Raises nothing for a non-mapping root (returns 0); parse
    errors propagate to the caller.
    """
    if path.lower().endswith(".json"):
        import json
        seen: list = []
        json.loads(text, object_pairs_hook=lambda pairs: (seen.append(pairs),
                                                          dict(pairs))[1])
        top = seen[-1] if seen else []
        return sum(1 for k, _v in top if k == "bh76_mode")
    import yaml
    node = yaml.compose(text, Loader=yaml.SafeLoader)
    if not isinstance(node, yaml.MappingNode):
        return 0
    return sum(1 for k, _v in node.value
               if getattr(k, "value", None) == "bh76_mode")


def require_explicit_bh76_mode(path: str) -> None:
    """Refuse a DFS-domain config FILE whose text does not state ``bh76_mode``.

    The knob selects WHAT the three BH76 training points supervise: the
    staged transition states as true forward barrier heights
    (``barrier_height``, the treatment in the reference dpyscf training set,
    whose trajectory carries the HNNO / CH3OH / FHF transition states with
    ``reference_height`` values) or the historical reaction-energy
    substitution (``reaction_energy``). The dataclass default filled the key
    in silently, and every campaign through v6 trained the substitution that
    way; the two submission entry points (``prepare`` / ``submit``) call this
    so a submitted file states its objective. In-process construction (test
    fixtures, resolved-config round-trips, ``resubmit`` on an existing run
    directory) keeps the dataclass default and is not checked here.

    Scoped to ``domain_profile: dfs_step7`` -- the bh76w411 pool carries no
    transition states and its builder already rejects ``barrier_height``
    loudly, so only one value is legal there and explicitness adds nothing.

    Raises ``ValueError`` naming the file, the key, and both legal values.
    An UNPARSEABLE file is not this guard's concern: it returns silently for
    BOTH extensions (``json.JSONDecodeError`` subclasses ``ValueError``, so a
    naive re-raise would leak a parse error dressed as a mode refusal) and
    the caller's own ``load_grid_config`` failure handling produces the
    message (in the resubmission commands: corrupt means unrecoverable).
    """
    lower = path.lower()
    if not lower.endswith((".yaml", ".yml", ".json")):
        raise ValueError(
            f"unsupported grid config extension for {path!r}: "
            "expected .yaml, .yml, or .json")
    try:
        if lower.endswith(".json"):
            import json
            with open(path) as f:
                raw = json.load(f)
        else:
            import yaml
            with open(path) as f:
                raw = yaml.safe_load(f)
    except Exception:
        return
    if not isinstance(raw, dict) or raw.get("domain_profile") != "dfs_step7":
        return
    if "bh76_mode" not in raw:
        raise ValueError(
            f"{path}: domain_profile dfs_step7 requires an explicit "
            "bh76_mode ('barrier_height' trains the staged transition "
            "states as forward barrier heights; 'reaction_energy' trains "
            "the historical reaction-energy substitution). The silent "
            "default trained the substitution through every campaign to "
            "v6 -- state the objective in the file.")


def load_grid_config(path: str) -> GridConfig:
    """Load a ``.yaml`` or ``.json`` grid config and build the nested frozen
    dataclasses.

    YAML support uses a lazy ``import yaml`` so the dependency is only
    required when a YAML file is actually loaded (matches the pattern in
    ``scripts/oep_per_species_tune.py``). JSON uses the stdlib.

    Raises:
        ValueError: if a required key is missing (the message names the key),
            or if the file extension is unsupported.
        ImportError: if a ``.yaml`` file is loaded but PyYAML is not installed.
    """
    lower = path.lower()
    text = ""
    if lower.endswith((".yaml", ".yml")):
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise ImportError(
                "loading a YAML grid config requires PyYAML, "
                "install it with `pip install pyyaml`"
            ) from exc
        # The document text is kept: a walltime the YAML 1.1 resolvers turned
        # into a base-60 integer is restored from its literal (_build_cluster).
        with open(path) as f:
            text = f.read()
        raw = yaml.safe_load(text)
    elif lower.endswith(".json"):
        import json
        with open(path) as f:
            text = f.read()
        raw = json.loads(text)
    else:
        raise ValueError(
            f"unsupported grid config extension for {path!r}: "
            "expected .yaml, .yml, or .json"
        )

    if not isinstance(raw, dict):
        raise ValueError(
            f"grid config {path!r}: top-level must be a mapping, got "
            f"{type(raw).__name__}"
        )
    # Both formats keep only the LAST of two duplicated top-level keys, so a
    # second bh76_mode entry silently decides the trained objective while the
    # first is dead text an operator may edit to no effect (the last-wins
    # hazard the duplicated-walltime refusal covers). Counted STRUCTURALLY on
    # the root mapping (yaml.compose / a JSON pairs hook): a column-0 text
    # scan missed quoted keys, indented documents and JSON entirely.
    n_modes = _top_level_bh76_mode_count(text, path)
    if n_modes > 1:
        raise ValueError(
            f"{path}: bh76_mode appears {n_modes} times at the top level; "
            "the parser keeps only the last, so the others are dead text. "
            "State the objective exactly once."
        )
    _reject_unknown_keys(raw, GridConfig, "<root>")

    return GridConfig(
        sweep=_build_sweep(_require(raw, "sweep", "<root>")),
        solvers=_build_solvers(_require(raw, "solvers", "<root>")),
        hyperparams=_build_hyperparams(_require(raw, "hyperparams", "<root>")),
        inputs=_build_inputs(_require(raw, "inputs", "<root>")),
        pretrain=_build_pretrain_from(raw, source=path),
        cluster=_build_cluster(_require(raw, "cluster", "<root>"),
                               text=text, source=path),
        domain_profile=_require(raw, "domain_profile", "<root>"),
        on_precompute_failure=raw.get("on_precompute_failure", "abort"),
        bh76_mode=raw.get("bh76_mode", "reaction_energy"),
        ae_as_reactions=bool(raw.get("ae_as_reactions", False)),
        use_polarized_correlation=bool(raw.get("use_polarized_correlation", False)),
        held_out_strict=bool(raw.get("held_out_strict", False)),
        defer_eval=bool(raw.get("defer_eval", False)),
        inline_eval=bool(raw.get("inline_eval", False)),
        eval_coldstart=bool(raw.get("eval_coldstart", False)),
        fidelity=_build_fidelity(raw.get("fidelity")),
        model=_build_model_block(raw.get("model")),
    )


# ---------------------------------------------------------------------------
# Grid expansion
# ---------------------------------------------------------------------------

def _canon_axis(values):
    """Deduplicate and sort an axis so the index->GridCell map is byte-stable.

    ``sorted(set(...))`` gives lexical order for the string axes and numeric
    order for ``subset_size``: deterministic across runs and Python versions.
    """
    return sorted(set(values))


def expand_grid(cfg: GridConfig) -> list[GridCell]:
    """Deterministic Cartesian product over the 5 axes.

    The axis order is FIXED: ``(arch, loss, metric, subset_size, solver)``.
    Each axis is independently deduplicated and sorted (see ``_canon_axis``)
    so the result list is byte-stable. The index of a ``GridCell`` in the
    returned list IS its SLURM array task id.
    """
    s = cfg.sweep
    arch = _canon_axis(s.arch)
    loss = _canon_axis(s.loss)
    metric = _canon_axis(s.metric)
    subset = _canon_axis(s.subset_size)
    solver = _canon_axis(s.solver)
    return [
        GridCell(arch=a, loss=l, metric=m, subset_size=ss, solver=sv)
        for a, l, m, ss, sv in product(arch, loss, metric, subset, solver)
    ]


# ---------------------------------------------------------------------------
# Semantic validation
# ---------------------------------------------------------------------------

def _warn_axis_dedups(cfg: GridConfig) -> None:
    """Emit a warning for every duplicate value removed from a sweep axis."""
    s = cfg.sweep
    for axis_name in ("arch", "loss", "metric", "subset_size", "solver"):
        raw = list(getattr(s, axis_name))
        seen = set()
        for v in raw:
            if v in seen:
                warnings.warn(
                    f"sweep axis {axis_name!r} contains duplicate value "
                    f"{v!r}; expand_grid will keep only one copy",
                    stacklevel=2,
                )
            seen.add(v)


class ParentAnchorNotImplemented(NotImplementedError, ValueError):
    """The refusal the PBE-anchor commit raised for an anchored meta-GGA
    architecture while the SCAN parent had not landed: a ``ValueError``, so
    every submission surface reported it as a configuration refusal, and a
    ``NotImplementedError``, since it was the scope of that commit rather
    than a defect of the file. No longer raised: ``parents.scan_fx`` /
    ``scan_fc`` carry the SCAN parent and a meta-GGA architecture under
    ``model.parent_anchor`` is accepted. Kept so that the name stays
    importable."""


def validate_grid_semantics(cfg: GridConfig, domain) -> None:
    """Login-node pre-submission validation of a ``GridConfig``.

    ``domain`` interface: this function depends ONLY on ``domain`` exposing an
    integer ``pool_size`` attribute (the size of the training-point pool that
    subsets are drawn from). It is passed as a parameter rather than imported
    so this module does not depend on the not-yet-built ``domain.py``.

    Raises:
        ValueError: on any hard semantic error (bad metric, out-of-range
            subset size, empty/oversized grid, bad numeric bounds, etc.).

    Emits ``warnings.warn`` for soft issues (axis dedup, SeaWulf throttle
    etiquette, advisory path checks).
    """
    # --- eval-mode consistency ---------------------------------------------
    if cfg.defer_eval and cfg.inline_eval:
        raise ValueError(
            "defer_eval and inline_eval are mutually exclusive (inline eval runs "
            "inside each train task; deferred eval is a separate array submitted "
            "after train). Set at most one."
        )
    # --- grid cardinality ---------------------------------------------------
    cells = expand_grid(cfg)
    n = len(cells)
    max_n = cfg.cluster.max_array_size
    if n == 0:
        raise ValueError(
            "grid expands to 0 cells, at least one sweep axis is empty; "
            "every axis (arch, loss, metric, subset_size, solver) must have "
            "at least one value"
        )
    if n > max_n:
        raise ValueError(
            f"grid expands to {n} cells but cluster.max_array_size is "
            f"{max_n}; either shrink the sweep axes or raise max_array_size "
            "(must stay <= the cluster's SLURM MaxArraySize)"
        )

    # --- duplicate-axis warnings -------------------------------------------
    _warn_axis_dedups(cfg)

    # --- metric membership --------------------------------------------------
    for m in cfg.sweep.metric:
        if m not in VALID_METRICS:
            raise ValueError(
                f"sweep metric {m!r} is not a known harness metric; "
                f"valid metrics: {sorted(VALID_METRICS)}"
            )

    # --- string-enum membership (on_precompute_failure / bh76_mode) --------
    # These carry defaults, so they are always set; validate at submit so an
    # invalid YAML value fails on the login node rather than at a later stage.
    opf = getattr(cfg, "on_precompute_failure", None)
    if opf is not None and opf not in VALID_ON_PRECOMPUTE_FAILURE:
        raise ValueError(
            f"on_precompute_failure {opf!r} is not valid; must be one of "
            f"{sorted(VALID_ON_PRECOMPUTE_FAILURE)}"
        )
    # A stub config without the attribute is a construction-time artifact and
    # keeps the dataclass default; a REAL config whose loaded value is None
    # means the YAML stated ``bh76_mode:`` with no value -- presence without a
    # choice -- and must refuse rather than short-circuit past the enum check
    # (previously such a config staged a full run with ``bh76_mode: null`` in
    # its resolved_config.yaml).
    bh76_mode = getattr(cfg, "bh76_mode", "reaction_energy")
    if bh76_mode is None or bh76_mode not in VALID_BH76_MODE:
        raise ValueError(
            f"bh76_mode {bh76_mode!r} is not valid; must be one of "
            f"{sorted(VALID_BH76_MODE)} (a bare 'bh76_mode:' with no value "
            "states no objective)"
        )

    # --- arch-name resolvability -------------------------------------------
    # Every value on the arch axis must resolve via get_architecture. Catching
    # an unknown arch on the login node gives a clear error instead of letting
    # the pretrain worker (_pretrain.py) fail at runtime on a compute node.
    from xcquinox.alec.config import get_architecture, list_architectures
    for a in cfg.sweep.arch:
        try:
            get_architecture(a)
        except KeyError:
            raise ValueError(
                f"sweep arch {a!r} is not a known architecture; valid "
                f"architectures: {list_architectures()}"
            ) from None

    # --- the model block --------------------------------------------------
    # The parent anchor is a property of the model class the run builds, so
    # it is checked against every architecture the run resolves, at submit:
    # an anchored correlation network must be polarization-aware (the
    # parent's correlation is divided by the model's zeta-dependent baseline,
    # which the pretraining data's open-shell targets are formed against).
    # Both rungs have their parent (parents.pbe_* and parents.scan_*), so the
    # rung itself is no ground for refusal.
    model_block = getattr(cfg, "model", None)
    if model_block is not None and getattr(model_block, "parent_anchor", False):
        run_polarized = bool(getattr(cfg, "use_polarized_correlation", False))
        for a in _canon_axis(cfg.sweep.arch):
            arch = get_architecture(a)
            if not (run_polarized or arch.use_polarized_correlation):
                raise ValueError(
                    f"model.parent_anchor is true but architecture {a!r} "
                    "would be built with use_polarized_correlation=False. An "
                    "anchored correlation network must be polarization-aware "
                    "(SPEC_parent_anchor.md Section 3.1: the parent's "
                    "correlation is divided by the zeta-dependent PW92 "
                    "baseline the pretraining data's open-shell Fc targets are "
                    "formed against; a zeta-blind network disagrees with them "
                    "by 14.9 mHa on the N atom). Set "
                    "use_polarized_correlation: true at the run level.")
    if model_block is not None and (
            getattr(model_block, "descriptor_coordinates", "legacy") == "dfs"
            and not bool(getattr(cfg, "use_polarized_correlation", False))):
        for a in _canon_axis(cfg.sweep.arch):
            if not get_architecture(a).use_polarized_correlation:
                raise ValueError(
                    f"model.descriptor_coordinates is 'dfs' but architecture "
                    f"{a!r} would be built with use_polarized_correlation="
                    "False; the DFS correlation network reads x1 = "
                    "ln(spinscale), so the polarized correlation network is "
                    "required. Set use_polarized_correlation: true at the "
                    "run level.")

    # --- subset_size bounds -------------------------------------------------
    pool_size = getattr(domain, "pool_size", None)
    if pool_size is None:
        raise ValueError(
            "validate_grid_semantics: the `domain` argument must expose an "
            "integer `pool_size` attribute (training-point pool size)"
        )
    for ss in cfg.sweep.subset_size:
        if not (1 <= ss <= pool_size):
            raise ValueError(
                f"subset_size {ss} is out of range; must satisfy "
                f"1 <= subset_size <= pool_size ({pool_size})"
            )

    # --- cheap numeric hyperparameter bounds -------------------------------
    hp = cfg.hyperparams
    if hp.n_steps <= 0:
        raise ValueError(f"hyperparams.n_steps must be > 0, got {hp.n_steps}")
    if not (0.0 <= hp.lr_decay_start <= 1.0):
        raise ValueError(
            f"hyperparams.lr_decay_start must be in [0, 1], got "
            f"{hp.lr_decay_start}"
        )
    if hp.lr_start < hp.lr_end:
        raise ValueError(
            f"hyperparams.lr_start ({hp.lr_start}) must be >= lr_end "
            f"({hp.lr_end})"
        )
    if hp.grad_clip <= 0:
        raise ValueError(
            f"hyperparams.grad_clip must be > 0, got {hp.grad_clip}"
        )
    if hp.update_scheme not in ("batched", "per_molecule"):
        raise ValueError(
            f"hyperparams.update_scheme must be 'batched' or 'per_molecule', "
            f"got {hp.update_scheme!r}"
        )

    # --- WS3 validation-slice knobs (2026-06-20) ---------------------------
    # FIX 2 (WS3-CFG-2): bound the 4 new knobs at submit time.
    if not (0.0 < hp.val_frac < 1.0):
        raise ValueError(
            f"hyperparams.val_frac must be in (0, 1), got {hp.val_frac}"
        )
    if hp.validate_every < 0:
        raise ValueError(
            f"hyperparams.validate_every must be >= 0, got {hp.validate_every}"
        )
    if hp.patience < 0:
        raise ValueError(
            f"hyperparams.patience must be >= 0, got {hp.patience}"
        )
    if hp.early_stop_min_delta < 0:
        raise ValueError(
            f"hyperparams.early_stop_min_delta must be >= 0, got "
            f"{hp.early_stop_min_delta}"
        )
    # FIX 1 (asymmetric/dead-config guard): in-loop validation actually runs ONLY
    # in train._run_per_molecule_loop (the only loop with the validation hook) and
    # ONLY when the val slice was staged (inputs.val_refs_dir, which spec_builder
    # uses to attach validation_molecules + validation_reactions_path). Without
    # both, validate_every>0 would never validate yet the eval would still exclude
    # a val slice (silent, non-comparable metric shrink). Make that unreachable at
    # submit: validate_every>0 REQUIRES val_refs_dir AND update_scheme per_molecule.
    if hp.validate_every > 0:
        if not getattr(cfg.inputs, "val_refs_dir", None):
            raise ValueError(
                f"hyperparams.validate_every={hp.validate_every} > 0 but "
                f"inputs.val_refs_dir is unset; in-loop validation needs the "
                f"staged density-only val slice (val_refs_dir). Set val_refs_dir "
                f"or set validate_every=0 to disable validation."
            )
        if hp.update_scheme != "per_molecule":
            raise ValueError(
                f"hyperparams.validate_every={hp.validate_every} > 0 requires "
                f"update_scheme='per_molecule' (the only training loop with an "
                f"in-loop validation hook); got update_scheme="
                f"{hp.update_scheme!r}. The 'batched' loop never validates, so "
                f"the eval would silently report a non-comparable shrunken metric."
            )
        # Early-stop GEOMETRY guard: in-loop validation fires
        # floor(n_steps/validate_every) times, and train._BestValidationTracker.
        # should_stop can only build a no-improvement streak of n_checks-1 (the
        # FIRST finite check sets the baseline and never counts as non-improving --
        # the correct Keras semantics). So a patience of P fires anywhere but the
        # degenerate "val-min at the very first check" case ONLY when
        # n_checks >= P+2. patience >= n_checks-1 is effectively dead: the v3 runs
        # (n_steps=150, validate_every=25 -> 6 checks, patience=5) reported
        # early_stopped=False for EVERY spec. Reject at submit so a whole training
        # run is not silently wasted on an early-stop that can never trigger.
        if hp.patience > 0:
            n_checks = int(hp.n_steps) // hp.validate_every
            if hp.patience >= n_checks - 1:
                raise ValueError(
                    f"hyperparams.patience={hp.patience} cannot drive early-stop "
                    f"with n_steps={hp.n_steps} / validate_every={hp.validate_every}"
                    f": that is only n_checks={n_checks} validation checks, and the "
                    f"no-improvement streak maxes at n_checks-1={n_checks - 1} (the "
                    f"first check sets the baseline), so early-stop would fire only "
                    f"degenerately or never. Use patience <= {max(0, n_checks - 2)} "
                    f"(n_checks >= patience+2), or raise n_steps / lower "
                    f"validate_every."
                )
    # The harness NEVER builds a PBE-anchor sample (spec_builder hardcodes
    # pbe_anchor_sample=None), so a positive pbe_anchor_weight is a silent
    # no-op for the A/B/C/D losses and a hard error for L5_gradnorm_vxc_step7
    # (its round-3 fail-fast). Reject it at submit time for ALL losses rather
    # than let it mislead (CW2/CODE-2 round-4). Anchoring is a pretrain-stage
    # concern; step-7 freezes pretraining from step-6.
    if hp.pbe_anchor_weight and hp.pbe_anchor_weight > 0.0:
        raise ValueError(
            f"hyperparams.pbe_anchor_weight={hp.pbe_anchor_weight} > 0 but the "
            f"harness builds no pbe_anchor_sample, so the PBE anchor cannot be "
            f"applied (it would be silently dropped, or raise for "
            f"L5_gradnorm_vxc_step7). Set pbe_anchor_weight=0."
        )

    # --- pretrain stage bounds ---------------------------------------------
    pt = cfg.pretrain
    if pt.n_steps <= 0:
        raise ValueError(
            f"pretrain.n_steps must be > 0, got {pt.n_steps}"
        )
    if not pt.data_dir:
        raise ValueError("pretrain.data_dir must be a non-empty path")
    if not (0.0 <= pt.lr_decay_start <= 1.0):
        raise ValueError(
            f"pretrain.lr_decay_start must be in [0, 1], got "
            f"{pt.lr_decay_start}"
        )
    if pt.lr_start < pt.lr_end:
        raise ValueError(
            f"pretrain.lr_start ({pt.lr_start}) must be >= lr_end "
            f"({pt.lr_end})"
        )
    if pt.grad_clip <= 0:
        raise ValueError(
            f"pretrain.grad_clip must be > 0, got {pt.grad_clip}"
        )
    if pt.loss_weighting not in _LOSS_WEIGHTINGS:
        raise ValueError(
            f"pretrain.loss_weighting must be 'unweighted' or "
            f"'integration', got {pt.loss_weighting!r}"
        )

    # --- pretraining-protocol bounds ----------------------------------------
    # The numeric knobs are refused when NON-FINITE for the reason the
    # certificate tolerances are: NaN satisfies neither sense of an ordinary
    # bound (nan < 0 and nan >= 1.0 are both False), so it would load with no
    # complaint and every comparison against it downstream would be False as
    # well. mesh_fraction and validation_fraction are bounded on both sides,
    # which already catches the infinities; energy_term_weight is bounded from
    # below only, so its finiteness is checked outright.
    if pt.parent_density not in _PARENT_DENSITIES:
        raise ValueError(
            f"pretrain.parent_density must be 'pbe', 'scan' or 'auto', got "
            f"{pt.parent_density!r}"
        )
    if pt.exchange_footing not in _EXCHANGE_FOOTINGS:
        raise ValueError(
            f"pretrain.exchange_footing must be 'total' or 'spin_channel', "
            f"got {pt.exchange_footing!r}"
        )
    # The bound is the CONSUMER's: pretrain_data_gen._check_generator_arguments
    # requires 0 < mesh_fraction < 1, so a share of exactly zero loads here and
    # is refused in the queued generator. Stated open on both sides.
    if not (0.0 < pt.mesh_fraction < 1.0):
        raise ValueError(
            f"pretrain.mesh_fraction must be in (0, 1), got "
            f"{pt.mesh_fraction}"
        )
    if not math.isfinite(pt.energy_term_weight) or pt.energy_term_weight < 0:
        raise ValueError(
            f"pretrain.energy_term_weight must be a FINITE number >= 0, got "
            f"{pt.energy_term_weight}"
        )
    # The OBJECTIVE, read against the certificate it is fitted for. At
    # EXACTLY zero the per-system energy term is not small but absent:
    # ``run_pretrain`` short-circuits it (``pretrain.py``: ``if
    # self.energy_weight == 0.0 or self.energy_target is None``), leaving the
    # integration-weighted point-wise residual alone. That is the
    # pre-protocol objective, and it is the one
    # SPEC_pretrain_fidelity_program.md Section 2 measured 2.3 to
    # 56.1 kcal/mol of atomization offset under -- no architecture reached
    # its parent inside the certificate's tolerances at weight zero, the
    # descriptor-free ones included. Refused at submit because the cost is
    # paid before the failure is visible: the datagen job and every
    # architecture's pretraining run first, then a certificate FAIL on all of
    # them and a train array whose afterok dependency never releases.
    #
    # The CONJUNCTION is what is refused, not any of its three parts. Each is
    # legal and shipped on its own: the templates run the point-wise
    # objective on the historical four-atom set (``dfs_set`` off), and the
    # workflow-verification matrix runs the protocol set with the gate waived
    # (``enforce`` false, which already demands a written override_reason).
    #
    # An ANCHORED configuration (model.parent_anchor) is exempt: its networks
    # equal the parent at initialization, so the certificate holds by
    # construction and the weight is no longer the value that decides
    # whether it can be met (SPEC_parent_anchor.md Section 3.5). The
    # energy-weight sweep the refusal names measured that no weight brings
    # a point-wise fit of the parent to the certificate (Section 2); 0.0 is
    # exact for an anchored run and is stated without a sweep.
    anchored_run = bool(getattr(getattr(cfg, "model", None),
                                "parent_anchor", False))
    if (pt.dfs_set and cfg.fidelity.enforce and pt.energy_term_weight == 0.0
            and not anchored_run):
        raise ValueError(
            "pretrain.energy_term_weight is 0.0 with pretrain.dfs_set: true "
            "and fidelity.enforce: true. At exactly zero the per-system "
            "energy term is not small, it is NOT EVALUATED (pretrain.py "
            "short-circuits on `energy_weight == 0.0`), so this run would "
            "fit the protocol pretraining set with the integration-weighted "
            "point-wise objective ALONE -- the pre-protocol objective under "
            "which NO architecture reached its parent inside these "
            "tolerances: SPEC_pretrain_fidelity_program.md Section 2 records "
            "atomization-energy offsets of 2.3 to 56.1 kcal/mol against "
            f"fidelity.tol_AE = {cfg.fidelity.tol_AE} kcal/mol. The "
            "certificate would then FAIL on every architecture, after the "
            "datagen job and the whole pretrain array had run. The weight is "
            "dimensionful (inverse Hartree^2) and is measured, not derived: "
            "hpcjobs/probe_pretrain_energy_weight.py sweeps it and prints "
            "the chosen value on its `recommendation:` line. Land that "
            "number and this refusal clears, e.g. "
            r"sed -i 's/^  energy_term_weight: 0\.0$/  energy_term_weight: "
            "<W>/' <config>.yaml. To run this set WITHOUT the gate instead "
            "-- a workflow-verification run, never a quantitative result -- "
            "set fidelity.enforce: false with a fidelity.override_reason")
    if not (0.0 <= pt.validation_fraction < 1.0):
        raise ValueError(
            f"pretrain.validation_fraction must be in [0, 1), got "
            f"{pt.validation_fraction}"
        )
    if pt.validate_every <= 0:
        raise ValueError(
            f"pretrain.validate_every must be > 0, got {pt.validate_every}"
        )
    if pt.patience < 0:
        raise ValueError(
            f"pretrain.patience must be >= 0, got {pt.patience}"
        )
    if not (0 <= pt.validation_seed <= _MAX_SEED):
        raise ValueError(
            f"pretrain.validation_seed must be in [0, {_MAX_SEED}], got "
            f"{pt.validation_seed}"
        )

    # --- the irreproducible-degenerate waiver -------------------------------
    # The waiver authorises a pretraining file whose spatially degenerate free
    # atoms' rows are one arbitrary member of their manifold. Exactly two
    # identities produce such a file (pretrain_data_gen.
    # _check_irreproducible_degenerate): a grid below
    # _MIN_REPRODUCIBLE_GRID_LEVEL, and the lock at zero. At a production
    # identity -- grid level >= 3 with the lock on -- the generator refuses
    # nothing, so a stated waiver permits nothing and is instead a leftover of
    # the template it was copied from. Refused rather than warned, because the
    # flag is dormant only until the next edit of the identity: it would then
    # authorise the very build it was never meant to cover, silently.
    inp = cfg.inputs
    if getattr(inp, "allow_irreproducible_degenerate", False):
        _level = int(inp.grid_level)
        _lock = float(getattr(inp, "orientation_lock_strength", 0.0))
        if _level >= _MIN_REPRODUCIBLE_GRID_LEVEL and _lock > 0.0:
            raise ValueError(
                "inputs.allow_irreproducible_degenerate: waiver stated "
                f"without need at grid level {_level} with "
                f"orientation_lock_strength={_lock:g}. Every spatially "
                "degenerate free atom's rows reproduce at that identity "
                f"(grid level >= {_MIN_REPRODUCIBLE_GRID_LEVEL} with the lock "
                "on), so the data generator refuses nothing here and the "
                "waiver grants a permission the run never exercises -- the "
                "manifest records it as unexercised. The shipped templates "
                "state it because they run at grid level 1; a copy promoted "
                "to a production identity must drop both "
                "inputs.allow_irreproducible_degenerate and "
                "inputs.irreproducible_degenerate_reason, or the next change "
                "of basis or grid level silently authorises an irreproducible "
                "build")

    # --- certificate tolerance bounds --------------------------------------
    # The program's binding decision is tol_AE = 1.0 kcal/mol and tol_atom =
    # 1.0 mHa for every architecture. A looser run is possible but never
    # silent: above 2.0 / 2.0 the config must carry a non-empty
    # override_reason, which the certificate copies into its own record.
    fid = cfg.fidelity
    if fid.tol_AE <= 0:
        raise ValueError(f"fidelity.tol_AE must be > 0, got {fid.tol_AE}")
    if fid.tol_atom <= 0:
        raise ValueError(f"fidelity.tol_atom must be > 0, got {fid.tol_atom}")
    if fid.tol_AE_max_backstop <= 0:
        raise ValueError(
            f"fidelity.tol_AE_max_backstop must be > 0, got "
            f"{fid.tol_AE_max_backstop}")
    _override = (fid.override_reason or "").strip()
    if (fid.tol_AE > 2.0 or fid.tol_atom > 2.0
            or fid.tol_AE_max_backstop > 2.0) and not _override:
        raise ValueError(
            f"fidelity.tol_AE={fid.tol_AE} kcal/mol / "
            f"fidelity.tol_atom={fid.tol_atom} mHa / "
            f"fidelity.tol_AE_max_backstop={fid.tol_AE_max_backstop} kcal/mol "
            "exceed the 2.0 ceiling; a certificate tolerance above that "
            "ceiling requires a non-empty fidelity.override_reason, which is "
            "recorded in every certificate the run writes")
    if not fid.enforce and not _override:
        raise ValueError(
            "fidelity.enforce=false disables the on-node certificate gates, "
            "so it requires a non-empty fidelity.override_reason; the reason "
            "is recorded in every certificate the run writes. Such a run is "
            "still refused by validate_run, merge_v4_arms and the figure "
            "suite, so it can only be used for workflow verification")

    # --- resource bounds ----------------------------------------------------
    cl = cfg.cluster
    if cl.array_throttle < 1:
        raise ValueError(
            f"cluster.array_throttle must be >= 1, got {cl.array_throttle}"
        )
    if cl.eval_array_throttle < 1:
        raise ValueError(
            f"cluster.eval_array_throttle must be >= 1, got "
            f"{cl.eval_array_throttle}"
        )
    if cl.cpus_per_task < 1:
        raise ValueError(
            f"cluster.cpus_per_task must be >= 1, got {cl.cpus_per_task}"
        )
    if cl.device == "gpu" and cl.gpus_per_task < 1:
        raise ValueError(
            f"cluster.device is 'gpu' but gpus_per_task is "
            f"{cl.gpus_per_task}; must be >= 1"
        )
    # Pretrain-stage resource knobs (all None-defaulted; bound-check only the
    # numeric ones when explicitly set).
    if cl.pretrain_throttle is not None and cl.pretrain_throttle < 1:
        raise ValueError(
            f"cluster.pretrain_throttle must be >= 1 when set, got "
            f"{cl.pretrain_throttle}"
        )
    if cl.pretrain_cpus_per_task is not None and cl.pretrain_cpus_per_task < 1:
        raise ValueError(
            f"cluster.pretrain_cpus_per_task must be >= 1 when set, got "
            f"{cl.pretrain_cpus_per_task}"
        )

    # --- per-stage allocation mode -----------------------------------------
    # datagen MUST be validated too: render_sbatch(kind='datagen') only emits
    # the exclusive `--nodes=1 --exclusive` lines when the value is exactly
    # "exclusive" and otherwise falls through to a SHARED render, so an invalid
    # datagen_allocation would silently downgrade the memory-heavy datagen job
    # to a shared node.
    _ALLOC_MODES = ("exclusive", "shared")
    for _stage in ("train", "eval", "preflight", "pretrain", "datagen",
                   "benchmark_refs"):
        _mode = getattr(cl, f"{_stage}_allocation")
        if _mode not in _ALLOC_MODES:
            raise ValueError(
                f"cluster.{_stage}_allocation must be one of {_ALLOC_MODES}, "
                f"got {_mode!r}"
            )

    # --- SeaWulf throttle etiquette ----------------------------------------
    # The train array and the eval array each consume concurrent slots. If
    # they share a partition, their throttles compete for the same pool.
    train_part = cl.partition
    eval_part = cl.eval_partition or cl.partition
    if train_part == eval_part:
        if cl.array_throttle + cl.eval_array_throttle > cl.max_concurrent_tasks:
            warnings.warn(
                f"train and eval arrays share partition {train_part!r}: "
                f"array_throttle ({cl.array_throttle}) + eval_array_throttle "
                f"({cl.eval_array_throttle}) = "
                f"{cl.array_throttle + cl.eval_array_throttle} exceeds "
                f"max_concurrent_tasks ({cl.max_concurrent_tasks}), this "
                "may violate cluster fair-use policy",
                stacklevel=2,
            )
    else:
        if cl.array_throttle > cl.max_concurrent_tasks:
            warnings.warn(
                f"cluster.array_throttle ({cl.array_throttle}) exceeds "
                f"max_concurrent_tasks ({cl.max_concurrent_tasks})",
                stacklevel=2,
            )
        if cl.eval_array_throttle > cl.max_concurrent_tasks:
            warnings.warn(
                f"cluster.eval_array_throttle ({cl.eval_array_throttle}) "
                f"exceeds max_concurrent_tasks ({cl.max_concurrent_tasks})",
                stacklevel=2,
            )

    # --- advisory login-node path checks -----------------------------------
    # These are login-node-local and therefore only advisory; the preflight
    # job (a later task) running on a compute node is authoritative.
    if not os.path.isdir(cfg.pretrain.data_dir):
        warnings.warn(
            f"pretrain.data_dir {cfg.pretrain.data_dir!r} not found on the "
            "login node, this is advisory; the preflight job is "
            "authoritative for compute-node path resolution",
            stacklevel=2,
        )
    out_parent = os.path.dirname(cfg.inputs.output_root.rstrip("/")) or "."
    if not os.path.isdir(out_parent):
        warnings.warn(
            f"parent of inputs.output_root ({out_parent!r}) does not exist "
            "on the login node, advisory; the preflight job is authoritative",
            stacklevel=2,
        )


# ---------------------------------------------------------------------------
# Pretrain checkpoint path, job-scoped
# ---------------------------------------------------------------------------

def pretrain_checkpoint_dir(run_dir: str, arch: str) -> str:
    """Return the pretrain checkpoint dir for one architecture, under the run dir.

    Layout: ``<run_dir>/pretrain/<arch>``. Co-locating the pretrain checkpoint
    with every other artifact for the submission (``logs/``, ``specs/``,
    ``checkpoints/`` ...) keeps all work for a run in one folder. Because
    ``run_dir`` is already unique per submission (its timestamped basename), two
    runs that pretrain the SAME architecture write to DISTINCT directories
    instead of clobbering each other's ``xnet.eqx``/``cnet.eqx``: the same
    anti-clobber guarantee the former ``<pretrain_root>/<run_id>/<arch>`` layout
    provided, now intrinsic to the run dir.

    The pretrain worker (``cluster/_pretrain.py``), the spec-builder
    (``cluster/spec_builder.py``, which sets each TrainingSpec's
    ``pretrain_checkpoint``), and the status check (``cluster/__main__.py``)
    all derive the path through THIS function so they cannot drift.
    """
    return os.path.join(os.path.abspath(run_dir), "pretrain", arch)
