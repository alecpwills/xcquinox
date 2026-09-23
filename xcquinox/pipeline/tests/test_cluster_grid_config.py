"""Tests for xcquinox.pipeline.cluster.grid_config: the HPC harness config layer."""
import json
import re

import pytest

from xcquinox.pipeline.cluster.grid_config import (
    GridConfig,
    GridCell,
    SweepAxes,
    SolverNamed,
    HyperParams,
    InputPaths,
    PretrainConfig,
    ClusterResources,
    load_grid_config,
    expand_grid,
    validate_grid_semantics,
    VALID_METRICS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _StubDomain:
    """Minimal stand-in for the not-yet-built DomainProfile, exposes only the
    `pool_size` attribute that validate_grid_semantics depends on."""
    def __init__(self, pool_size: int):
        self.pool_size = pool_size


def _base_config_dict():
    """A complete, valid raw config dict (the form a YAML/JSON file parses to)."""
    return {
        "sweep": {
            "arch": ["medium"],
            "loss": ["delta_ae"],
            "metric": ["l2", "jsd"],
            "subset_size": [4, 8, 12, 16, 20, 24, 28, 32, 36, 40],
            "solver": ["fast", "robust"],
        },
        "solvers": {
            "fast": {"mode": "fixed_density", "max_cycles": 1},
            "robust": {
                "mode": "scf",
                "max_cycles": 30,
                "feature_policy": "clamp",
            },
        },
        "hyperparams": {
            "n_steps": 200,
            "lr_start": 1e-3,
            "lr_end": 1e-5,
            "lr_decay_start": 0.2,
            "grad_clip": 1.0,
            "gradnorm_alpha": 1.5,
            "vxc_weight": 1.0,
            "density_weight": 0.5,
        },
        "inputs": {
            "external_refs_dir": "/shared/refs",
            "subset_ledger_path": "/shared/subset_index_log.json",
            "basis": "def2-tzvp",
            "grid_level": 3,
            "output_root": "/shared/runs",
        },
        "pretrain": {
            "data_dir": "/shared/pretrain_data",
            "n_steps": 1000,
            "lr_start": 1e-2,
            "lr_end": 1e-5,
            "lr_decay_start": 0.2,
            "grad_clip": 1.0,
            "loss_weighting": "integration",
        },
        "cluster": {
            "partition": "long-40core",
            "time": "12:00:00",
            "mem": "32G",
            "cpus_per_task": 4,
            "array_throttle": 10,
            "eval_array_throttle": 5,
            "max_concurrent_tasks": 40,
        },
        "domain_profile": "gmtkn55_subset",
    }


def _write(tmp_path, name, data):
    """Serialize `data` to a temp YAML or JSON file and return its path."""
    p = tmp_path / name
    if name.endswith((".yaml", ".yml")):
        yaml = pytest.importorskip("yaml")
        p.write_text(yaml.safe_dump(data))
    else:
        p.write_text(json.dumps(data))
    return str(p)


# ---------------------------------------------------------------------------
# load_grid_config: round-trips
# ---------------------------------------------------------------------------

def _assert_well_formed(cfg):
    assert isinstance(cfg, GridConfig)
    assert isinstance(cfg.sweep, SweepAxes)
    assert isinstance(cfg.hyperparams, HyperParams)
    assert isinstance(cfg.inputs, InputPaths)
    assert isinstance(cfg.pretrain, PretrainConfig)
    assert isinstance(cfg.cluster, ClusterResources)
    assert isinstance(cfg.solvers, dict)
    assert all(isinstance(v, SolverNamed) for v in cfg.solvers.values())
    # list fields became tuples
    assert isinstance(cfg.sweep.arch, tuple)
    assert isinstance(cfg.sweep.subset_size, tuple)
    # named solver fields preserved
    assert cfg.solvers["robust"].mode == "scf"
    assert cfg.solvers["robust"].max_cycles == 30
    assert cfg.solvers["robust"].feature_policy == "clamp"
    assert cfg.solvers["fast"].feature_policy is None
    # enum defaults
    assert cfg.on_precompute_failure == "abort"
    assert cfg.bh76_mode == "reaction_energy"
    # pretrain section round-trips
    assert cfg.pretrain.data_dir == "/shared/pretrain_data"
    assert cfg.pretrain.n_steps == 1000
    assert cfg.pretrain.loss_weighting == "integration"


def test_yaml_round_trip(tmp_path):
    path = _write(tmp_path, "grid.yaml", _base_config_dict())
    cfg = load_grid_config(path)
    _assert_well_formed(cfg)


def test_json_round_trip(tmp_path):
    path = _write(tmp_path, "grid.json", _base_config_dict())
    cfg = load_grid_config(path)
    _assert_well_formed(cfg)


# ---------------------------------------------------------------------------
# Solver mixer_kwargs: resolved_config.yaml round-trip (datagen regression)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Solver orientation_lock_strength: parse, default-off, resolved round-trip
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Run-level inputs.orientation_lock_strength (authoritative for the whole run)
# ---------------------------------------------------------------------------


def test_load_unsupported_extension(tmp_path):
    p = tmp_path / "grid.txt"
    p.write_text("nonsense")
    with pytest.raises(ValueError, match="unsupported grid config extension"):
        load_grid_config(str(p))


def test_load_missing_required_key(tmp_path):
    data = _base_config_dict()
    del data["hyperparams"]["n_steps"]
    path = _write(tmp_path, "grid.json", data)
    with pytest.raises(ValueError, match="hyperparams.n_steps"):
        load_grid_config(path)


def test_load_missing_top_level_section(tmp_path):
    data = _base_config_dict()
    del data["cluster"]
    path = _write(tmp_path, "grid.json", data)
    with pytest.raises(ValueError, match="cluster"):
        load_grid_config(path)


# ---------------------------------------------------------------------------
# expand_grid
# ---------------------------------------------------------------------------

def _cfg(**sweep_overrides):
    """Build a GridConfig directly with optional sweep-axis overrides."""
    base = dict(
        arch=("medium",),
        loss=("delta_ae",),
        metric=("l2", "jsd"),
        subset_size=tuple(range(4, 44, 4)),  # 10 values
        solver=("fast", "robust"),
    )
    base.update(sweep_overrides)
    return GridConfig(
        sweep=SweepAxes(**base),
        solvers={
            "fast": SolverNamed(mode="fixed_density", max_cycles=1),
            "robust": SolverNamed(mode="scf", max_cycles=30),
        },
        hyperparams=HyperParams(
            n_steps=200, lr_start=1e-3, lr_end=1e-5, lr_decay_start=0.2,
            grad_clip=1.0, gradnorm_alpha=1.5, vxc_weight=1.0,
            density_weight=0.5,
        ),
        inputs=InputPaths(
            external_refs_dir="/shared/refs",
            subset_ledger_path="/shared/subset_index_log.json",
            basis="def2-tzvp", grid_level=3, output_root="/shared/runs",
        ),
        pretrain=PretrainConfig(
            data_dir="/shared/pretrain_data",
        ),
        cluster=ClusterResources(
            partition="long-40core", time="12:00:00", mem="32G",
            cpus_per_task=4, array_throttle=10, eval_array_throttle=5,
            max_concurrent_tasks=40,
        ),
        domain_profile="gmtkn55_subset",
    )


def test_expand_grid_cardinality():
    # axes (1, 1, 2, 10, 2) -> 40 cells
    import itertools
    cfg = _cfg()
    cells = expand_grid(cfg)
    assert len(cells) == 40
    assert all(isinstance(c, GridCell) for c in cells)
    # No duplicate cells in the 40-way expansion.
    assert len(set(cells)) == 40
    # The expansion is exactly the Cartesian product of the five sweep axes.
    got = {(c.arch, c.loss, c.metric, c.subset_size, c.solver) for c in cells}
    expected = set(itertools.product(
        cfg.sweep.arch, cfg.sweep.loss, cfg.sweep.metric,
        cfg.sweep.subset_size, cfg.sweep.solver))
    assert got == expected


def test_expand_grid_axis_order_fixed():
    # The fixed axis order is (arch, loss, metric, subset_size, solver):
    # the slowest-varying is arch, the fastest is solver.
    cells = expand_grid(_cfg(arch=("a", "b")))
    # first len/2 cells all have arch 'a'
    half = len(cells) // 2
    assert all(c.arch == "a" for c in cells[:half])
    assert all(c.arch == "b" for c in cells[half:])
    # solver is fastest-varying: adjacent cells alternate solver
    assert cells[0].solver != cells[1].solver


def test_expand_grid_dedup():
    # A repeated axis value collapses to a single GridCell.
    cells = expand_grid(_cfg(arch=("medium", "medium")))
    arches = {c.arch for c in cells}
    assert arches == {"medium"}
    # same cardinality as the non-duplicated single-arch grid
    assert len(cells) == len(expand_grid(_cfg(arch=("medium",))))


# ---------------------------------------------------------------------------
# validate_grid_semantics
# ---------------------------------------------------------------------------

def test_validate_ok():
    # pool_size 40 covers subset sizes up to 40
    validate_grid_semantics(_cfg(), _StubDomain(pool_size=40))


def test_validate_bad_metric():
    with pytest.raises(ValueError, match="not a known harness metric"):
        validate_grid_semantics(
            _cfg(metric=("l2", "bogus")), _StubDomain(pool_size=40)
        )
    # the valid set is exactly {l2, jsd}
    assert VALID_METRICS == frozenset({"l2", "jsd"})


def test_validate_grid_too_large():
    cfg = _cfg()
    # shrink the cluster's max_array_size below the 40-cell grid
    small = ClusterResources(
        partition="p", time="1:00:00", mem="8G", cpus_per_task=1,
        array_throttle=1, eval_array_throttle=1, max_concurrent_tasks=40,
        max_array_size=10,
    )
    cfg = GridConfig(
        sweep=cfg.sweep, solvers=cfg.solvers, hyperparams=cfg.hyperparams,
        inputs=cfg.inputs, pretrain=cfg.pretrain, cluster=small,
        domain_profile=cfg.domain_profile,
    )
    with pytest.raises(ValueError, match="max_array_size"):
        validate_grid_semantics(cfg, _StubDomain(pool_size=40))


def test_validate_missing_pool_size():
    class _Empty:
        pass
    with pytest.raises(ValueError, match="pool_size"):
        validate_grid_semantics(_cfg(), _Empty())


def _cfg_with(hp_kwargs=None, inputs_kwargs=None):
    """Build a GridConfig (single sweep cell so cardinality is irrelevant) with
    HyperParams / InputPaths field overrides, for the WS3 validation guards."""
    base = _cfg(
        arch=("medium",), loss=("delta_ae",), metric=("l2",),
        subset_size=(4,), solver=("fast",),
    )
    hp_defaults = dict(
        n_steps=200, lr_start=1e-3, lr_end=1e-5, lr_decay_start=0.2,
        grad_clip=1.0, gradnorm_alpha=1.5, vxc_weight=1.0, density_weight=0.5,
    )
    hp_defaults.update(hp_kwargs or {})
    hp = HyperParams(**hp_defaults)
    in_defaults = dict(
        external_refs_dir="/shared/refs",
        subset_ledger_path="/shared/subset_index_log.json",
        basis="def2-tzvp", grid_level=3, output_root="/shared/runs",
    )
    in_defaults.update(inputs_kwargs or {})
    inputs = InputPaths(**in_defaults)
    return GridConfig(
        sweep=base.sweep, solvers=base.solvers, hyperparams=hp,
        inputs=inputs, pretrain=base.pretrain, cluster=base.cluster,
        domain_profile=base.domain_profile,
    )


# --- WS3 validation-slice cross-field + range guards (2026-06-20) -----------

def test_validate_rejects_validate_every_without_val_refs_dir():
    """FIX 1(b): validate_every>0 requires inputs.val_refs_dir (the only thing
    that stages the val slice). Without it training never validates yet the eval
    would still exclude a val slice -> asymmetric/dead config; reject at submit."""
    cfg = _cfg_with(hp_kwargs=dict(validate_every=2, update_scheme="per_molecule"),
                    inputs_kwargs=dict(val_refs_dir=None))
    with pytest.raises(ValueError, match="val_refs_dir"):
        validate_grid_semantics(cfg, _StubDomain(pool_size=40))


def test_validate_rejects_dead_early_stop_patience():
    # Early-stop GEOMETRY: n_steps=150 / validate_every=25 -> floor=6 validation
    # checks; should_stop's no-improvement streak maxes at n_checks-1=5 (the first
    # check sets the baseline), so patience=5 can fire only degenerately/never --
    # exactly the v3 config whose runs all reported early_stopped=False. Reject it
    # at submit so a whole training run is not wasted on a dead early-stop.
    cfg = _cfg_with(
        hp_kwargs=dict(n_steps=150, validate_every=25, patience=5,
                       update_scheme="per_molecule"),
        inputs_kwargs=dict(val_refs_dir="/shared/val_refs"))
    with pytest.raises(ValueError, match="early-stop"):
        validate_grid_semantics(cfg, _StubDomain(pool_size=40))


def test_validate_rejects_unknown_arch_name():
    """Every arch-axis value must resolve via get_architecture; an unknown
    name is rejected on the login node, not deferred to the pretrain worker."""
    with pytest.raises(ValueError, match="not a known architecture"):
        validate_grid_semantics(
            _cfg(arch=("medium", "no_such_arch")), _StubDomain(pool_size=40)
        )


# ---------------------------------------------------------------------------
# PretrainConfig
# ---------------------------------------------------------------------------

def test_pretrain_config_round_trip(tmp_path):
    """The pretrain section round-trips through load_grid_config."""
    path = _write(tmp_path, "grid.json", _base_config_dict())
    cfg = load_grid_config(path)
    pt = cfg.pretrain
    assert isinstance(pt, PretrainConfig)
    assert pt.data_dir == "/shared/pretrain_data"
    assert pt.n_steps == 1000
    assert pt.lr_start == 1e-2
    assert pt.lr_end == 1e-5
    assert pt.lr_decay_start == 0.2
    assert pt.grad_clip == 1.0
    assert pt.loss_weighting == "integration"


def test_pretrain_config_defaults(tmp_path):
    """Optional pretrain keys fall back to step-7 defaults."""
    data = _base_config_dict()
    data["pretrain"] = {
        "data_dir": "/shared/pretrain_data",
    }
    path = _write(tmp_path, "grid.json", data)
    cfg = load_grid_config(path)
    pt = cfg.pretrain
    assert pt.n_steps == 1000
    assert pt.lr_start == 1e-2
    assert pt.lr_end == 1e-5
    assert pt.lr_decay_start == 0.2
    assert pt.grad_clip == 1.0
    assert pt.seed == 42
    assert pt.loss_weighting == "integration"


def test_load_missing_pretrain_required_key(tmp_path):
    data = _base_config_dict()
    del data["pretrain"]["data_dir"]
    path = _write(tmp_path, "grid.json", data)
    with pytest.raises(ValueError, match="pretrain.data_dir"):
        load_grid_config(path)


# ---------------------------------------------------------------------------
# Per-stage allocation mode + optional mem
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# pretrain_checkpoint_dir: run-scoped pretrain output path
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# use_polarized_correlation (run-level spin-polarized correlation toggle)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# defer_eval (run-level deferred-eval submission toggle)
# ---------------------------------------------------------------------------


def test_validate_rejects_both_defer_and_inline_eval():
    """defer_eval and inline_eval are mutually exclusive. A config that sets
    BOTH (no CLI flags) must fail at login-node validation, not only later
    inside submit_jobs."""
    import dataclasses
    cfg = dataclasses.replace(_cfg(), defer_eval=True, inline_eval=True)
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_grid_semantics(cfg, _StubDomain(pool_size=40))
    # Each alone validates fine.
    validate_grid_semantics(
        dataclasses.replace(_cfg(), defer_eval=True), _StubDomain(pool_size=40))
    validate_grid_semantics(
        dataclasses.replace(_cfg(), inline_eval=True), _StubDomain(pool_size=40))


# 2026-06-20 (WS3): held-out validation slice knobs drive in-loop early-stop +
# validation-best selection. All MUST default to a NO-OP so decay-free runs stay
# byte-identical (validate_every=0 -> no in-loop validation; patience=0 -> no
# early-stop).


# WS5 (2026-06-20): periodic-resume checkpoint cadence; default 0 => no-op so
# existing sweeps stay byte-identical.


# 2026-06-20 (WS4): a named solver entry may opt into SCF gradient checkpointing
# (for full_25); the parser must read it, defaulting off.


# ---------------------------------------------------------------------------
# Per-rung SCF seeding knobs (inputs.seed_xc / seed_cache_dir) + eval_coldstart
# ---------------------------------------------------------------------------


def test_seed_and_coldstart_resolved_round_trip(tmp_path):
    """seed_xc / seed_cache_dir / eval_coldstart survive asdict + yaml +
    reload (the resolved_config.yaml path the preflight re-reads) -- the
    ae_as_reactions silent-drop incident class."""
    from xcquinox.pipeline.cluster.__main__ import _config_to_raw_dict
    d = _base_config_dict()
    d["inputs"]["seed_xc"] = "auto"
    d["inputs"]["seed_cache_dir"] = "/gpfs/scratch/x/seed_cache"
    d["eval_coldstart"] = True
    cfg = load_grid_config(_write(tmp_path, "grid.yaml", d))
    assert cfg.inputs.seed_xc == "auto"
    assert cfg.inputs.seed_cache_dir == "/gpfs/scratch/x/seed_cache"
    assert cfg.eval_coldstart is True
    cfg2 = load_grid_config(
        _write(tmp_path, "resolved.yaml", _config_to_raw_dict(cfg)))
    assert cfg2.inputs.seed_xc == "auto"
    assert cfg2.inputs.seed_cache_dir == "/gpfs/scratch/x/seed_cache"
    assert cfg2.eval_coldstart is True


# ---------------------------------------------------------------------------
# FidelityConfig: the per-architecture physics-certificate tolerances
# ---------------------------------------------------------------------------


def test_fidelity_block_parses(tmp_path):
    raw = _base_config_dict()
    raw["fidelity"] = {"tol_AE": 0.5, "tol_atom": 0.25,
                       "override_reason": None}
    cfg = load_grid_config(_write(tmp_path, "grid.yaml", raw))
    assert cfg.fidelity.tol_AE == 0.5
    assert cfg.fidelity.tol_atom == 0.25


def test_fidelity_resolved_round_trip(tmp_path):
    """The resolved config is re-read by the pretrain, preflight and eval
    stages; a dropped fidelity block would silently revert a documented
    override to the binding tolerances mid-run."""
    from xcquinox.pipeline.cluster.__main__ import _config_to_raw_dict
    raw = _base_config_dict()
    raw["fidelity"] = {"tol_AE": 2.5, "tol_atom": 2.5,
                       "override_reason": "rung-3.5 control arm"}
    cfg = load_grid_config(_write(tmp_path, "grid.yaml", raw))
    cfg2 = load_grid_config(
        _write(tmp_path, "resolved.yaml", _config_to_raw_dict(cfg)))
    assert cfg2.fidelity == cfg.fidelity


def test_validate_rejects_a_loose_tolerance_without_an_override_reason(tmp_path):
    raw = _base_config_dict()
    raw["fidelity"] = {"tol_AE": 3.0, "tol_atom": 1.0}
    cfg = load_grid_config(_write(tmp_path, "grid.yaml", raw))
    with pytest.raises(ValueError, match="override_reason"):
        validate_grid_semantics(cfg, _StubDomain(pool_size=100))


# ---------------------------------------------------------------------------
# cluster walltimes: sexagesimal restoration + SLURM shape validation
# ---------------------------------------------------------------------------

#: Walltime field -> the (render kind, array_max) whose ``#SBATCH --time``
#: directive that field feeds. ``preflight`` takes no array index.
_WALLTIME_RENDER = {
    "time": ("train", 1),
    "preflight_time": ("preflight", None),
    "eval_time": ("eval", 1),
    "pretrain_time": ("pretrain", 0),
}

#: Every walltime field of ClusterResources. The four in ``_WALLTIME_RENDER``
#: are the ones an sbatch stage renders directly; the remaining three are
#: fallback/retry walls and are exposed to the same YAML resolver.
_WALLTIME_KEYS = tuple(_WALLTIME_RENDER) + (
    "datagen_time", "timeout_retry_time", "benchmark_refs_time")

#: Base wall used for the non-``time`` fields, distinct from every literal
#: under test so a rendered directive identifies which field it came from
#: (every per-stage wall falls back to ``cluster.time`` when unset).
_OTHER_BASE_TIME = "01:00:00"


def _write_walltime_yaml(tmp_path, key, literal, name="grid.yaml"):
    """Write a config whose ``cluster.<key>`` carries the RAW YAML token
    ``literal``.

    The token is appended to a dumped ``cluster:`` block instead of being
    routed through ``yaml.safe_dump``, which quotes any string that would
    otherwise re-resolve to a non-string -- precisely the quoting a
    hand-written config may omit.
    """
    yaml = pytest.importorskip("yaml")
    raw = _base_config_dict()
    cluster = raw.pop("cluster")
    cluster.pop(key, None)
    if key != "time":
        cluster["time"] = _OTHER_BASE_TIME
    p = tmp_path / name
    p.write_text(yaml.safe_dump(raw) + yaml.safe_dump({"cluster": cluster})
                 + f"  {key}: {literal}\n")
    return str(p)


@pytest.mark.parametrize("literal,expected", [
    ('"8:00:00"', "8:00:00"),
    ("1-12:00:00", "1-12:00:00"),
])
@pytest.mark.parametrize("key", ["time"])
def test_walltime_accepted_shapes(tmp_path, key, literal, expected):
    """``H:MM:SS`` and ``D-HH:MM:SS`` are the accepted walltime shapes."""
    cfg = load_grid_config(_write_walltime_yaml(tmp_path, key, literal))
    assert getattr(cfg.cluster, key) == expected


@pytest.mark.parametrize("literal", ["30", "1.5", "true"])
@pytest.mark.parametrize("key", ["time"])
def test_walltime_bad_shapes_refused(tmp_path, key, literal):
    """Anything outside the two accepted shapes is refused, naming the key."""
    with pytest.raises(ValueError, match=re.escape(f"cluster.{key}")):
        load_grid_config(_write_walltime_yaml(tmp_path, key, literal))


#: The campaign configs under version control: the template, the two grid-2
#: campaigns the user guide walks through, the grid-3 lineage root the loss
#: primer cites line by line, and the six v7 files (the three group files, the
#: reaction-energy control and the two arms). ``hpcjobs/.gitignore`` excludes
#: ``configs/*.local.yaml`` (personal cluster-filled copies), so a fresh clone,
#: a git worktree and the cluster checkout carry only these; counting whatever
#: ``*.yaml`` happens to be on disk would make this file red wherever the
#: untracked copies are absent. The set is held equal to the index below.
_TRACKED_CONFIGS = (
    "bh76w411_repr.svp_grid2.yaml",
    "bh76w411_repr.tzvpd_grid2_df.yaml",
    "dfs_step7.dfs6311_grid3_v3.yaml",
    "dfs_step7.dfs6311_grid3_v7g1_c25.yaml",
    "dfs_step7.dfs6311_grid3_v7g1_dfsparity.yaml",
    "dfs_step7.dfs6311_grid3_v7g1_rxn.yaml",
    "dfs_step7.dfs6311_grid3_v7g1_size.yaml",
    "dfs_step7.dfs6311_grid3_v7g2_families_mgga.yaml",
    "dfs_step7.dfs6311_grid3_v7g2a_families_core.yaml",
    "step7.yaml",
)


def _config_tree():
    """(config dir, shipped example) of this checkout, or None when absent."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[3]
    cfg_dir = root / "hpcjobs" / "configs"
    example = root / "xcquinox" / "pipeline" / "cluster" / "examples" / \
        "grid_step7.yaml"
    if not cfg_dir.is_dir() or not example.is_file():
        return None
    return cfg_dir, example


def _assert_walltimes_are_strings(path):
    cfg = load_grid_config(str(path))
    for key in _WALLTIME_KEYS:
        value = getattr(cfg.cluster, key)
        assert value is None or isinstance(value, str), (
            f"{path}: cluster.{key} loaded as {type(value).__name__}"
        )
    return cfg


def test_tracked_configs_carry_valid_walltimes():
    """Every version-controlled campaign config and the shipped example load.

    The tracked set is listed by name rather than globbed: the count is then a
    property of the repository, not of which untracked ``*.local.yaml`` copies
    happen to sit in the working tree.
    """
    tree = _config_tree()
    if tree is None:
        pytest.skip("cluster config tree not present in this checkout")
    cfg_dir, example = tree
    for name in _TRACKED_CONFIGS:
        path = cfg_dir / name
        assert path.is_file(), f"tracked config missing: {path}"
        _assert_walltimes_are_strings(path)
    _assert_walltimes_are_strings(example)
    assert len(_TRACKED_CONFIGS) + 1 == 11, "tracked config count changed"


# ---------------------------------------------------------------------------
# Literal recovery: the cluster block has to be located, and only its own
# top-level keys may supply a literal
# ---------------------------------------------------------------------------

def _write_cluster_header_yaml(tmp_path, header, key_lines, decoy=True):
    """Write a config whose ``cluster:`` header is spelled ``header``.

    ``key_lines`` is appended inside the block. A decoy section carrying a
    clock-shaped ``time:`` is placed FIRST, so any scan that is not confined to
    the cluster block finds it before the authored value. It is spelled
    ``x-decoy`` because the loader's blocks are closed: an ``x-`` key is the
    one thing a document may carry that no builder reads.
    """
    yaml = pytest.importorskip("yaml")
    raw = _base_config_dict()
    cluster = raw.pop("cluster")
    cluster.pop("time", None)
    block = yaml.safe_dump({"cluster": cluster})
    body = block[len("cluster:\n"):]
    text = ("x-decoy:\n  time: 8:00:00\n" if decoy else "")
    text += yaml.safe_dump(raw) + header + "\n" + body + key_lines
    p = tmp_path / "grid.yaml"
    p.write_text(text)
    return str(p)


def test_duplicated_walltime_key_is_refused(tmp_path):
    """YAML keeps the LAST of two duplicated keys while a first-match scan takes
    the first. ``8:00:00`` and ``480:00`` share the base-60 value 28800, so the
    consistency check cannot separate them and the authored ``480:00`` would be
    accepted as ``8:00:00``. Two spellings of one wall is a config defect."""
    path = _write_cluster_header_yaml(
        tmp_path, "cluster:", "  time: 8:00:00\n  time: 480:00\n")
    with pytest.raises(ValueError, match=re.escape("cluster.time")):
        load_grid_config(path)


# ---------------------------------------------------------------------------
# Durations: zero is not a wall, and D-HH is a time of day
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# PretrainConfig: pretraining-protocol fields
# ---------------------------------------------------------------------------


def test_build_pretrain_parses_every_protocol_field():
    """A field missing from _build_pretrain silently reverts to its default on
    every stage that re-reads resolved_config.yaml."""
    from xcquinox.pipeline.cluster.grid_config import _build_pretrain
    pt = _build_pretrain({
        "data_dir": "/d", "dfs_set": True, "pool_atoms": True,
        "parent_density": "auto", "exchange_footing": "spin_channel",
        "mesh_fraction": 0.25, "energy_term_weight": 1.0,
        "validation_fraction": 0.2, "validation_seed": 11,
        "validate_every": 25, "patience": 8,
    })
    assert pt.dfs_set is True and pt.pool_atoms is True
    assert pt.parent_density == "auto"
    assert pt.exchange_footing == "spin_channel"
    assert pt.mesh_fraction == 0.25
    assert pt.energy_term_weight == 1.0
    assert pt.validation_fraction == 0.2
    assert pt.validation_seed == 11
    assert pt.validate_every == 25
    assert pt.patience == 8


def test_config_to_raw_dict_round_trips_every_protocol_field(tmp_path):
    """The resolved_config.yaml round trip is what datagen, pretrain, preflight
    and eval all read; a dropped field is a silently reverted run."""
    import dataclasses
    from xcquinox.pipeline.cluster.__main__ import _config_to_raw_dict
    from xcquinox.pipeline.cluster.grid_config import (_build_pretrain,
                                                   pretrain_to_raw_dict)
    protocol = {
        "dfs_set": True, "pool_atoms": True,
        "parent_density": "auto", "exchange_footing": "spin_channel",
        "mesh_fraction": 0.25, "energy_term_weight": 1.0,
        "validation_fraction": 0.2, "validation_seed": 11,
        "validate_every": 25, "patience": 8,
    }
    pt = _build_pretrain(dict(protocol, data_dir="/d"))
    # The round-trip dict is the WRITER's, not a bare asdict: under the
    # non-sampled weighting modes the loader refuses the inert sampling keys,
    # so the writer drops exactly those and nothing else.
    raw = pretrain_to_raw_dict(pt)
    assert _build_pretrain(raw) == pt
    dropped = {"points_per_system", "sampling_seed"}
    assert not (dropped & set(raw)), sorted(dropped & set(raw))
    for f in dataclasses.fields(pt):
        if f.name in dropped:
            continue
        assert f.name in raw, f.name
    # An unknown key is IGNORED by _build_pretrain, so the equality above holds
    # vacuously for a field the dataclass does not carry; name the protocol
    # keys outright.
    for key, value in protocol.items():
        assert raw.get(key) == value, (key, raw.get(key), value)
    # ... and the same dict is what the serializer puts under "pretrain". A
    # field is guarded here ONLY while the fixture carries a NON-DEFAULT value
    # for it: a field the parser drops reloads at its default, which equals the
    # value under test whenever the fixture leaves that field alone, and the
    # comparison then passes against a parser that never read it. So every
    # field is taken off its default below, and the guard is asserted.
    # ``rho_w_sampled`` is the off-default weighting so the sampling keys are
    # legitimate, carried, and themselves guarded off-default.
    every = _build_pretrain(dict(
        raw, n_steps=7, lr_start=3e-2, lr_end=3e-6, lr_decay_start=0.4,
        lr_decay_end=0.7, grad_clip=2.5, seed=1234,
        loss_weighting="rho_w_sampled", points_per_system=333,
        sampling_seed=9, atoms=[["Li", 1], ["C", 2]]))
    default = PretrainConfig(data_dir="/d")
    for f in dataclasses.fields(every):
        if f.name == "data_dir":
            continue
        assert getattr(every, f.name) != getattr(default, f.name), (
            f"PretrainConfig.{f.name} is at its default in this fixture, so "
            "the round trip is NOT guarded for it")
    cfg = dataclasses.replace(_cfg(), pretrain=every)
    serialized = _config_to_raw_dict(cfg)["pretrain"]
    # Under the sampled mode nothing is inert, so the writer's dict IS the
    # full asdict and every field -- the sampling keys included -- survives.
    assert serialized == dataclasses.asdict(every)
    assert _build_pretrain(serialized) == every


# ---------------------------------------------------------------------------
# The pre-protocol objective under an enforced certificate
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# _build_pretrain: coercion hardening (the fidelity block's house pattern)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", ["dfs_set"])
@pytest.mark.parametrize("value", ["false"])
def test_build_pretrain_refuses_a_non_boolean_switch(key, value):
    """``bool("false")`` is True, so a hand-quoted switch would turn the DFS
    set (or the pool atoms) ON in a config that wrote it OFF, and ``bool(None)``
    -- an empty ``dfs_set:`` -- would read as OFF without remark. The fidelity
    block refuses ``enforce`` on the same grounds rather than coercing it."""
    from xcquinox.pipeline.cluster.grid_config import _build_pretrain
    with pytest.raises(ValueError) as exc:
        _build_pretrain({"data_dir": "/d", key: value})
    assert f"pretrain.{key}" in str(exc.value)
    assert repr(value) in str(exc.value)


@pytest.mark.parametrize("key", ["mesh_fraction"])
@pytest.mark.parametrize("value", [True])
def test_build_pretrain_refuses_a_non_numeric_protocol_value(key, value):
    """``float(True)`` is 1.0 and ``int(True)`` is 1 -- silently a weight of
    one, or one validation every step -- while ``float(None)`` raises
    TypeError, which passes every ``except ValueError`` handler in the load
    path and surfaces as a crash naming no key."""
    from xcquinox.pipeline.cluster.grid_config import _build_pretrain
    with pytest.raises(ValueError) as exc:
        _build_pretrain({"data_dir": "/d", key: value})
    assert f"pretrain.{key}" in str(exc.value)
    assert repr(value) in str(exc.value)


# ---------------------------------------------------------------------------
# _build_pretrain: the pre-protocol keys, same typed parsing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["Integration"])
def test_build_pretrain_refuses_an_unknown_loss_weighting(value):
    from xcquinox.pipeline.cluster.grid_config import _build_pretrain
    with pytest.raises(ValueError) as exc:
        _build_pretrain({"data_dir": "/d", "loss_weighting": value})
    assert "pretrain.loss_weighting" in str(exc.value)
    assert repr(value) in str(exc.value)


# ---------------------------------------------------------------------------
# The orientation lock: ONE constant for the harness default
# ---------------------------------------------------------------------------


def test_the_default_lock_reaches_the_training_solver_config(tmp_path):
    """A config without the key renders a training SCF AT the lock: the
    run-level value is authoritative over the solver's own 0.0, so the
    functional and the references sit on the same component."""
    from xcquinox.pipeline.cluster.spec_builder import _solver_config_from_named
    from xcquinox.pipeline.orientation_lock import DEFAULT_STRENGTH
    cfg = load_grid_config(_write(tmp_path, "grid.yaml", _base_config_dict()))
    # The solver states no lock of its own, which is the case the harness
    # default has to cover: spec_builder passes the run-level value on every
    # cell (spec_builder.py, _solver_config_from_named call site).
    named = SolverNamed(mode="oneshot", max_cycles=0)
    assert named.orientation_lock_strength == 0.0
    sc = _solver_config_from_named(
        named, orientation_lock_strength=cfg.inputs.orientation_lock_strength)
    assert sc.orientation_lock_strength == DEFAULT_STRENGTH


# ---------------------------------------------------------------------------
# The protocol knobs are bounded where every loader passes, not only in the
# semantic check the worker paths skip
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key,value", [
    ("mesh_fraction", 0.0),        # the consumer requires a STRICT lower bound
    ("patience", -1),
    ("validation_seed", 2 ** 40),  # above the generator seed range
])
def test_build_pretrain_bounds_every_protocol_number(key, value):
    """``validate_grid_semantics`` is the LOGIN-node check; the datagen,
    pretrain and preflight workers call ``load_grid_config`` alone. A bound
    that lives only in the semantic check therefore does not exist for the
    process that runs the schedule, so each one is stated at parse as well."""
    from xcquinox.pipeline.cluster.grid_config import _build_pretrain
    with pytest.raises(ValueError) as exc:
        _build_pretrain({"data_dir": "/d", key: value})
    assert f"pretrain.{key}" in str(exc.value)


# ---------------------------------------------------------------------------
# The fourth SCF seed: the superposition-of-atomic-densities guess
# ---------------------------------------------------------------------------


def test_seed_xc_minao_round_trips_to_the_solver_seed(tmp_path):
    """``minao`` is a run-wide seed choice beside pbe and scan: the config
    carries it, the resolved config the workers re-read carries it, and the
    per-cell resolution hands it to the solver configuration, which has
    accepted the value since the coldstart channel was added
    (``solver.py``: ``seed_source in ('pbe', 'scan', 'minao')``).

    Oracle: the value at each of the three layers, with the solver in the FULL
    mode a non-pbe seed requires, and the refusal message of a value that is
    none of the four.
    """
    from xcquinox.pipeline.cluster.__main__ import _config_to_raw_dict
    from xcquinox.pipeline.cluster.spec_builder import (
        _solver_config_from_named,
        resolve_seed_xc,
    )
    d = _base_config_dict()
    d["inputs"]["seed_xc"] = "minao"
    cfg = load_grid_config(_write(tmp_path, "grid.yaml", d))
    assert cfg.inputs.seed_xc == "minao"
    cfg2 = load_grid_config(
        _write(tmp_path, "resolved.yaml", _config_to_raw_dict(cfg)))
    assert cfg2.inputs.seed_xc == "minao"

    # The spec builder's own call shape: the per-cell resolution supplies
    # seed_source, and a non-pbe seed is accepted only in FULL mode.
    sc = _solver_config_from_named(
        SolverNamed(mode="FULL", max_cycles=3),
        seed_source=resolve_seed_xc(cfg2.inputs, cfg2.sweep.arch[0]))
    assert sc.seed_source == "minao"

    # The refusal names every accepted value, so the reader of the failure
    # does not have to read the parser to learn what the fourth one is.
    d["inputs"]["seed_xc"] = "hf"
    with pytest.raises(ValueError) as exc:
        load_grid_config(_write(tmp_path, "bad.yaml", d))
    message = str(exc.value)
    for value in ("pbe", "scan", "auto", "minao"):
        assert value in message, (value, message)


# ---------------------------------------------------------------------------
# The published cloning protocol's configuration surface
# ---------------------------------------------------------------------------

def test_the_model_block_carries_the_gate(tmp_path):
    """``model.ueg_gate`` is parsed, defaults to the gate every model before
    the field carried, survives the resolved-config round trip the later
    stages re-read, and refuses an unknown value naming both.

    Oracle: the parsed configuration, and the refusal's own message.
    """
    from xcquinox.pipeline.cluster.__main__ import _config_to_raw_dict

    raw = _base_config_dict()
    raw["model"] = {"ueg_gate": "x2", "descriptor_coordinates": "paper"}
    cfg = load_grid_config(_write(tmp_path, "grid.yaml", raw))
    assert cfg.model.ueg_gate == "x2"
    assert cfg.model.descriptor_coordinates == "paper"
    cfg2 = load_grid_config(
        _write(tmp_path, "resolved.yaml", _config_to_raw_dict(cfg)))
    assert cfg2.model == cfg.model

    plain = load_grid_config(
        _write(tmp_path, "plain.yaml", _base_config_dict()))
    assert plain.model.ueg_gate == "tanh2"

    bad = _base_config_dict()
    bad["model"] = {"ueg_gate": "X2"}
    with pytest.raises(ValueError) as excinfo:
        load_grid_config(_write(tmp_path, "bad.yaml", bad))
    message = str(excinfo.value)
    assert "ueg_gate" in message and "'tanh2'" in message and "'x2'" in message


def _published_pretrain_block():
    """The ``pretrain:`` block of the published cloning protocol: no clip, the
    published targets, the sampled objective, the protocol set, no energy
    term."""
    return {"data_dir": "/shared/pretrain_data", "n_steps": 20000,
            "lr_start": 1e-3, "lr_end": 1e-5, "lr_decay_start": 0.5,
            "lr_decay_end": 0.9, "grad_clip": 0, "loss_weighting":
            "rho_w_sampled", "points_per_system": 800, "sampling_seed": 42,
            "exchange_footing": "paper", "dfs_set": True,
            "energy_term_weight": 0.0}


def test_the_pretrain_block_accepts_the_published_values(tmp_path):
    """The published protocol's three refused values load: no clip, the
    published targets, and a zero energy-term weight under the sampled
    objective the paper actually ran.

    The energy-weight refusal was measured under the integration-weighted
    objective, so it stays in force there; a negative clip is still refused,
    since only exactly zero means "no clip".

    Oracle: ``validate_grid_semantics`` against a stub domain, and the two
    refusals, each seen to fire.
    """
    raw = _base_config_dict()
    raw["pretrain"] = _published_pretrain_block()
    raw["fidelity"] = {"enforce": True}
    cfg = load_grid_config(_write(tmp_path, "grid.yaml", raw))
    assert cfg.pretrain.grad_clip == 0.0
    assert cfg.pretrain.exchange_footing == "paper"
    assert cfg.pretrain.loss_weighting == "rho_w_sampled"
    validate_grid_semantics(cfg, _StubDomain(pool_size=100))

    measured = _base_config_dict()
    measured["pretrain"] = {k: v for k, v in _published_pretrain_block().items()
                            if k not in ("points_per_system", "sampling_seed")}
    measured["pretrain"]["loss_weighting"] = "integration"
    measured["pretrain"]["grad_clip"] = 1.0
    measured["fidelity"] = {"enforce": True}
    cfg_measured = load_grid_config(
        _write(tmp_path, "measured.yaml", measured))
    with pytest.raises(ValueError, match="energy_term_weight"):
        validate_grid_semantics(cfg_measured, _StubDomain(pool_size=100))

    negative = _base_config_dict()
    negative["pretrain"] = _published_pretrain_block()
    negative["pretrain"]["grad_clip"] = -1.0
    path = _write(tmp_path, "negative.yaml", negative)
    with pytest.raises(ValueError, match="grad_clip"):
        validate_grid_semantics(load_grid_config(path),
                                _StubDomain(pool_size=100))


def test_the_paper_coordinates_require_the_polarized_network_at_the_parser(
        tmp_path):
    """The published coordinates read the spin coordinate in the correlation
    network exactly as the dfs set does, so a run whose architectures would
    be built zeta-blind is refused on the login node, before any job is
    queued, and not inside the pretrain array.

    Oracle: ``validate_grid_semantics`` on a run of the base architecture,
    which carries no polarized flag, under each of the two coordinate sets,
    seen to refuse both with the same requirement.
    """
    for coordinates in ("paper", "dfs"):
        raw = _base_config_dict()
        raw["model"] = {"descriptor_coordinates": coordinates}
        cfg = load_grid_config(_write(tmp_path, f"{coordinates}.yaml", raw))
        with pytest.raises(ValueError, match="polarized") as excinfo:
            validate_grid_semantics(cfg, _StubDomain(pool_size=100))
        assert repr(coordinates) in str(excinfo.value)


def test_the_paper_footing_refuses_an_energy_term(tmp_path):
    """Under the published footing an open shell's per-system exchange table
    integrates the total-density form rather than PBE's spin-scaled exchange,
    so a per-system energy term at any positive weight is refused with the
    footing named; the published protocol has no such term.

    Oracle: the published block with a positive weight, seen to be refused,
    and the same block at zero, which loads.
    """
    raw = _base_config_dict()
    raw["pretrain"] = _published_pretrain_block()
    raw["pretrain"]["energy_term_weight"] = 0.1
    raw["fidelity"] = {"enforce": True}
    cfg = load_grid_config(_write(tmp_path, "weighted.yaml", raw))
    with pytest.raises(ValueError, match="energy_term_weight") as excinfo:
        validate_grid_semantics(cfg, _StubDomain(pool_size=100))
    assert "paper" in str(excinfo.value)

    raw["pretrain"]["energy_term_weight"] = 0.0
    cfg = load_grid_config(_write(tmp_path, "unweighted.yaml", raw))
    validate_grid_semantics(cfg, _StubDomain(pool_size=100))
