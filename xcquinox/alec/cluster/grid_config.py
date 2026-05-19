"""xcquinox.alec.cluster.grid_config — config layer for the HPC training harness.

The harness submits a grid of training jobs to a SLURM cluster as an array
job. The grid is the Cartesian product of a small set of swept axes, defined
declaratively in a YAML (or JSON) config file. This module provides:

  - Frozen dataclasses describing every section of that config.
  - ``load_grid_config`` — parse a ``.yaml``/``.json`` file into a ``GridConfig``.
  - ``expand_grid`` — the deterministic Cartesian product producing one
    ``GridCell`` per SLURM array task. A cell's index in the returned list IS
    its array task id, so the expansion MUST be byte-stable across runs and
    Python versions (achieved via ``sorted(set(...))`` per axis).
  - ``validate_grid_semantics`` — login-node pre-submission sanity checks.

Design note — the ``domain`` dependency:
    ``validate_grid_semantics`` needs the size of the training-point pool to
    bound ``subset_size``. That pool lives in the not-yet-built ``domain.py``
    module. To avoid a hard import dependency on a module that does not exist,
    the domain object is received as a *parameter*; we depend only on it
    exposing an integer ``pool_size`` attribute. See the function docstring.
"""
from dataclasses import dataclass, fields
from itertools import product
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

    Deliberately limited: only ``mode``, ``max_cycles`` and an optional
    ``feature_policy``. Do NOT add conv_tol / mixer fields here — those belong
    to a richer solver config consumed downstream.
    """
    mode: str
    max_cycles: int
    feature_policy: str | None = None


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class HyperParams:
    """Training hyperparameters shared by every grid cell."""
    n_steps: int
    lr_start: float
    lr_end: float
    # lr_decay_start is a FRACTION of n_steps, in [0, 1] — matches the
    # PretrainSpec / TrainingSpec convention in xcquinox.alec.config.
    lr_decay_start: float
    grad_clip: float
    gradnorm_alpha: float
    vxc_weight: float
    density_weight: float
    pbe_anchor_weight: float = 0.0
    require_atom_anchors: bool = False
    seed: int = 42


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
    external_refs_dir: str
    descriptor_cache: str
    refhist_cache: str
    subset_ledger_path: str
    basis: str
    grid_level: int
    output_root: str
    pretrain_checkpoint: str | None = None


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
    # Optional retry knobs — used when re-submitting a task that died from
    # OOM or wall-clock timeout. None = no dedicated retry config.
    oom_retry_partition: str | None = None
    oom_retry_mem: str | None = None
    timeout_retry_partition: str | None = None
    timeout_retry_time: str | None = None


# ---------------------------------------------------------------------------
# Aggregate config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GridConfig:
    """The complete harness config — aggregate of every section above."""
    sweep: SweepAxes
    # Named solver configs, keyed by the names used in the ``solver`` axis.
    solvers: dict[str, SolverNamed]
    hyperparams: HyperParams
    inputs: InputPaths
    cluster: ClusterResources
    domain_profile: str
    on_precompute_failure: str = "abort"   # {"abort","drop_failed_species"}
    bh76_mode: str = "reaction_energy"     # {"reaction_energy","barrier_height"}


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
        out[str(name)] = SolverNamed(
            mode=_require(sd, "mode", ctx),
            max_cycles=_require(sd, "max_cycles", ctx),
            feature_policy=sd.get("feature_policy"),
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
        pbe_anchor_weight=d.get("pbe_anchor_weight", 0.0),
        require_atom_anchors=d.get("require_atom_anchors", False),
        seed=d.get("seed", 42),
    )


def _build_inputs(d: dict) -> InputPaths:
    ctx = "inputs"
    return InputPaths(
        external_refs_dir=_require(d, "external_refs_dir", ctx),
        descriptor_cache=_require(d, "descriptor_cache", ctx),
        refhist_cache=_require(d, "refhist_cache", ctx),
        subset_ledger_path=_require(d, "subset_ledger_path", ctx),
        basis=_require(d, "basis", ctx),
        grid_level=_require(d, "grid_level", ctx),
        output_root=_require(d, "output_root", ctx),
        pretrain_checkpoint=d.get("pretrain_checkpoint"),
    )


def _build_cluster(d: dict) -> ClusterResources:
    ctx = "cluster"
    return ClusterResources(
        partition=_require(d, "partition", ctx),
        time=_require(d, "time", ctx),
        mem=_require(d, "mem", ctx),
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
        oom_retry_partition=d.get("oom_retry_partition"),
        oom_retry_mem=d.get("oom_retry_mem"),
        timeout_retry_partition=d.get("timeout_retry_partition"),
        timeout_retry_time=d.get("timeout_retry_time"),
    )


def load_grid_config(path: str) -> GridConfig:
    """Load a ``.yaml`` or ``.json`` grid config and build the nested frozen
    dataclasses.

    YAML support uses a *lazy* ``import yaml`` so the dependency is only
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
                "loading a YAML grid config requires PyYAML — "
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
        cluster=_build_cluster(_require(raw, "cluster", "<root>")),
        domain_profile=_require(raw, "domain_profile", "<root>"),
        on_precompute_failure=raw.get("on_precompute_failure", "abort"),
        bh76_mode=raw.get("bh76_mode", "reaction_energy"),
    )


# ---------------------------------------------------------------------------
# Grid expansion
# ---------------------------------------------------------------------------

def _canon_axis(values):
    """Deduplicate and sort an axis so the index->GridCell map is byte-stable.

    ``sorted(set(...))`` gives lexical order for the string axes and numeric
    order for ``subset_size`` — deterministic across runs and Python versions.
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
    # --- grid cardinality ---------------------------------------------------
    cells = expand_grid(cfg)
    n = len(cells)
    max_n = cfg.cluster.max_array_size
    if n == 0:
        raise ValueError(
            "grid expands to 0 cells — at least one sweep axis is empty; "
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
                f"max_concurrent_tasks ({cl.max_concurrent_tasks}) — this "
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
    import os
    if cfg.inputs.pretrain_checkpoint is not None and not os.path.exists(
        cfg.inputs.pretrain_checkpoint
    ):
        warnings.warn(
            f"inputs.pretrain_checkpoint {cfg.inputs.pretrain_checkpoint!r} "
            "not found on the login node — this is advisory; the preflight "
            "job is authoritative for compute-node path resolution",
            stacklevel=2,
        )
    out_parent = os.path.dirname(cfg.inputs.output_root.rstrip("/")) or "."
    if not os.path.isdir(out_parent):
        warnings.warn(
            f"parent of inputs.output_root ({out_parent!r}) does not exist "
            "on the login node — advisory; the preflight job is authoritative",
            stacklevel=2,
        )
