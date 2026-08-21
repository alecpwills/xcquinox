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
import os
import warnings


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
    # SolverConfig, in spec_builder) AND the CCSD reference generation (training
    # refs via external_refs.precompute_all + the held-out benchmark_refs job), so
    # the references and the functional lock the SAME degenerate component of a
    # radical (OH/CH/NO). 0.0 -> off -> byte-identical; existing YAMLs unaffected.
    orientation_lock_strength: float = 0.0
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
    ``tol_atom`` mHa AND max |dAE| over atomization energies <= ``tol_AE``
    kcal/mol on frozen parent densities at the run's identity.

    The defaults are the program's binding decision (1.0 kcal/mol and 1.0
    mHa). ``validate_grid_semantics`` refuses either tolerance above 2.0
    unless ``override_reason`` is non-empty, so a run can only be loosened
    deliberately and with the reason on the record: the string is copied into
    every certificate the run writes.
    """
    tol_AE: float = 1.0          # kcal/mol, atomization-energy offset
    tol_atom: float = 1.0        # mHa, free-atom E_xc offset
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


def _build_sweep(d: dict) -> SweepAxes:
    """Build SweepAxes from a raw dict; list fields become tuples."""
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


def _build_inputs(d: dict) -> InputPaths:
    ctx = "inputs"
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
        orientation_lock_strength=float(d.get("orientation_lock_strength", 0.0)),
        benchmark_refs_dir=d.get("benchmark_refs_dir"),
        val_refs_dir=d.get("val_refs_dir"),
        seed_xc=seed_xc,
        seed_cache_dir=d.get("seed_cache_dir"),
    )


def _build_pretrain(d: dict) -> PretrainConfig:
    ctx = "pretrain"
    return PretrainConfig(
        data_dir=_require(d, "data_dir", ctx),
        n_steps=d.get("n_steps", 1000),
        lr_start=d.get("lr_start", 1e-2),
        lr_end=d.get("lr_end", 1e-5),
        lr_decay_start=d.get("lr_decay_start", 0.2),
        grad_clip=d.get("grad_clip", 1.0),
        seed=d.get("seed", 42),
        loss_weighting=d.get("loss_weighting", "integration"),
        atoms=_parse_pretrain_atoms(d.get("atoms")),
    )


def _fidelity_tolerance(d, key: str) -> float:
    """Read one certificate tolerance out of a raw ``fidelity`` mapping.

    A tolerance is an energy bound, so a boolean or a container is a config
    error rather than something to coerce: ``float(True)`` is 1.0 (silently
    the binding tolerance) and ``float(None)`` raises ``TypeError``, which
    passes every ``except ValueError`` handler in the load path. Integers,
    floats and numeric strings remain valid.
    """
    v = d.get(key, 1.0)
    if isinstance(v, bool) or not isinstance(v, (int, float, str)):
        raise ValueError(
            f"grid config key 'fidelity.{key}' must be a number (kcal/mol for "
            f"tol_AE, mHa for tol_atom), got {type(v).__name__} ({v!r})")
    try:
        return float(v)
    except ValueError:
        raise ValueError(
            f"grid config key 'fidelity.{key}' must be a number (kcal/mol for "
            f"tol_AE, mHa for tol_atom), got {type(v).__name__} "
            f"({v!r})") from None


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
    return FidelityConfig(
        tol_AE=_fidelity_tolerance(d, "tol_AE"),
        tol_atom=_fidelity_tolerance(d, "tol_atom"),
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


def _build_cluster(d: dict) -> ClusterResources:
    ctx = "cluster"
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
    if lower.endswith((".yaml", ".yml")):
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - env-dependent
            raise ImportError(
                "loading a YAML grid config requires PyYAML, "
                "install it with `pip install pyyaml`"
            ) from exc
        with open(path) as f:
            raw = yaml.safe_load(f)
    elif lower.endswith(".json"):
        import json
        with open(path) as f:
            raw = json.load(f)
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

    return GridConfig(
        sweep=_build_sweep(_require(raw, "sweep", "<root>")),
        solvers=_build_solvers(_require(raw, "solvers", "<root>")),
        hyperparams=_build_hyperparams(_require(raw, "hyperparams", "<root>")),
        inputs=_build_inputs(_require(raw, "inputs", "<root>")),
        pretrain=_build_pretrain(_require(raw, "pretrain", "<root>")),
        cluster=_build_cluster(_require(raw, "cluster", "<root>")),
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
    bh76_mode = getattr(cfg, "bh76_mode", None)
    if bh76_mode is not None and bh76_mode not in VALID_BH76_MODE:
        raise ValueError(
            f"bh76_mode {bh76_mode!r} is not valid; must be one of "
            f"{sorted(VALID_BH76_MODE)}"
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
    if pt.loss_weighting not in ("unweighted", "integration"):
        raise ValueError(
            f"pretrain.loss_weighting must be 'unweighted' or "
            f"'integration', got {pt.loss_weighting!r}"
        )

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
    _override = (fid.override_reason or "").strip()
    if (fid.tol_AE > 2.0 or fid.tol_atom > 2.0) and not _override:
        raise ValueError(
            f"fidelity.tol_AE={fid.tol_AE} kcal/mol / "
            f"fidelity.tol_atom={fid.tol_atom} mHa exceed the 2.0 / 2.0 "
            "ceiling; a certificate tolerance above that ceiling requires a "
            "non-empty fidelity.override_reason, which is recorded in every "
            "certificate the run writes")
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
