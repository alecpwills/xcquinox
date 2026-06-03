"""Tests for xcquinox.alec.cluster.grid_config: the HPC harness config layer."""
import json

import pytest

from xcquinox.alec.cluster.grid_config import (
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
    cells = expand_grid(_cfg())
    assert len(cells) == 40
    assert all(isinstance(c, GridCell) for c in cells)
    # indices are 0..N-1 and the list IS the index map
    assert cells[0] is cells[0]
    assert cells[39] == cells[-1]


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


def test_expand_grid_order_invariance():
    # Permuting a YAML axis's value order yields an identical expansion.
    a = expand_grid(_cfg(metric=("l2", "jsd")))
    b = expand_grid(_cfg(metric=("jsd", "l2")))
    assert a == b


def test_expand_grid_dedup():
    # A repeated axis value collapses to a single GridCell.
    cells = expand_grid(_cfg(arch=("medium", "medium")))
    arches = {c.arch for c in cells}
    assert arches == {"medium"}
    # same cardinality as the non-duplicated single-arch grid
    assert len(cells) == len(expand_grid(_cfg(arch=("medium",))))


def test_expand_grid_subset_size_numeric_sort():
    cells = expand_grid(_cfg(
        arch=("a",), loss=("x",), metric=("l2",),
        subset_size=(40, 4, 12), solver=("fast",),
    ))
    assert [c.subset_size for c in cells] == [4, 12, 40]


# ---------------------------------------------------------------------------
# validate_grid_semantics
# ---------------------------------------------------------------------------

def test_validate_ok():
    # pool_size 40 covers subset sizes up to 40
    validate_grid_semantics(_cfg(), _StubDomain(pool_size=40))


def test_validate_rejects_positive_pbe_anchor_weight():
    """CW2/CODE-4 round-4: the harness builds no pbe_anchor_sample, so a
    positive hyperparams.pbe_anchor_weight is a no-op (A/B/C/D) or a hard error
    (L5): reject it at submit time for all losses."""
    import dataclasses
    cfg = _cfg()
    cfg = dataclasses.replace(
        cfg, hyperparams=dataclasses.replace(
            cfg.hyperparams, pbe_anchor_weight=0.5))
    with pytest.raises(ValueError, match="pbe_anchor_weight"):
        validate_grid_semantics(cfg, _StubDomain(pool_size=40))
    # weight 0.0 (the default) validates fine.
    validate_grid_semantics(_cfg(), _StubDomain(pool_size=40))


def test_validate_dedup_warns():
    with pytest.warns(UserWarning, match="duplicate"):
        validate_grid_semantics(
            _cfg(arch=("medium", "medium")), _StubDomain(pool_size=40)
        )


def test_validate_bad_metric():
    with pytest.raises(ValueError, match="not a known harness metric"):
        validate_grid_semantics(
            _cfg(metric=("l2", "bogus")), _StubDomain(pool_size=40)
        )
    # the valid set is exactly {l2, jsd}
    assert VALID_METRICS == frozenset({"l2", "jsd"})


def test_validate_subset_size_too_large():
    with pytest.raises(ValueError, match="out of range"):
        # subset sizes go up to 40 but pool only has 20 points
        validate_grid_semantics(_cfg(), _StubDomain(pool_size=20))


def test_validate_subset_size_zero():
    with pytest.raises(ValueError, match="out of range"):
        validate_grid_semantics(
            _cfg(subset_size=(0, 4)), _StubDomain(pool_size=40)
        )


def test_validate_empty_axis_rejected():
    with pytest.raises(ValueError, match="0 cells"):
        validate_grid_semantics(
            _cfg(solver=()), _StubDomain(pool_size=40)
        )


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


def test_validate_array_throttle_too_small():
    cfg = _cfg()
    bad = ClusterResources(
        partition="p", time="1:00:00", mem="8G", cpus_per_task=1,
        array_throttle=0, eval_array_throttle=1, max_concurrent_tasks=40,
    )
    cfg = GridConfig(
        sweep=cfg.sweep, solvers=cfg.solvers, hyperparams=cfg.hyperparams,
        inputs=cfg.inputs, pretrain=cfg.pretrain, cluster=bad,
        domain_profile=cfg.domain_profile,
    )
    with pytest.raises(ValueError, match="array_throttle must be >= 1"):
        validate_grid_semantics(cfg, _StubDomain(pool_size=40))


def test_validate_n_steps_nonpositive():
    cfg = _cfg()
    bad_hp = HyperParams(
        n_steps=0, lr_start=1e-3, lr_end=1e-5, lr_decay_start=0.2,
        grad_clip=1.0, gradnorm_alpha=1.5, vxc_weight=1.0, density_weight=0.5,
    )
    cfg = GridConfig(
        sweep=cfg.sweep, solvers=cfg.solvers, hyperparams=bad_hp,
        inputs=cfg.inputs, pretrain=cfg.pretrain, cluster=cfg.cluster,
        domain_profile=cfg.domain_profile,
    )
    with pytest.raises(ValueError, match="n_steps must be > 0"):
        validate_grid_semantics(cfg, _StubDomain(pool_size=40))


def test_validate_lr_decay_start_out_of_range():
    cfg = _cfg()
    bad_hp = HyperParams(
        n_steps=200, lr_start=1e-3, lr_end=1e-5, lr_decay_start=1.5,
        grad_clip=1.0, gradnorm_alpha=1.5, vxc_weight=1.0, density_weight=0.5,
    )
    cfg = GridConfig(
        sweep=cfg.sweep, solvers=cfg.solvers, hyperparams=bad_hp,
        inputs=cfg.inputs, pretrain=cfg.pretrain, cluster=cfg.cluster,
        domain_profile=cfg.domain_profile,
    )
    with pytest.raises(ValueError, match="lr_decay_start must be in"):
        validate_grid_semantics(cfg, _StubDomain(pool_size=40))


def test_validate_shared_partition_throttle_overflow():
    cfg = _cfg()
    # train + eval arrays both on the same partition, throttles sum > max
    bad = ClusterResources(
        partition="long-40core", time="1:00:00", mem="8G", cpus_per_task=1,
        array_throttle=30, eval_array_throttle=20, max_concurrent_tasks=40,
        eval_partition="long-40core",
    )
    cfg = GridConfig(
        sweep=cfg.sweep, solvers=cfg.solvers, hyperparams=cfg.hyperparams,
        inputs=cfg.inputs, pretrain=cfg.pretrain, cluster=bad,
        domain_profile=cfg.domain_profile,
    )
    with pytest.warns(UserWarning, match="exceeds max_concurrent_tasks"):
        validate_grid_semantics(cfg, _StubDomain(pool_size=40))


def test_validate_missing_pool_size():
    class _Empty:
        pass
    with pytest.raises(ValueError, match="pool_size"):
        validate_grid_semantics(_cfg(), _Empty())


def test_validate_rejects_unknown_arch_name():
    """Every arch-axis value must resolve via get_architecture; an unknown
    name is rejected on the login node, not deferred to the pretrain worker."""
    with pytest.raises(ValueError, match="not a known architecture"):
        validate_grid_semantics(
            _cfg(arch=("medium", "no_such_arch")), _StubDomain(pool_size=40)
        )


def test_validate_known_arch_names_accepted():
    """A grid using only registered architecture names validates cleanly."""
    validate_grid_semantics(
        _cfg(arch=("medium", "deep_combined_attn")),
        _StubDomain(pool_size=40),
    )


def test_validate_bad_pretrain_throttle():
    cfg = _cfg()
    bad = ClusterResources(
        partition="p", time="1:00:00", mem="8G", cpus_per_task=1,
        array_throttle=1, eval_array_throttle=1, max_concurrent_tasks=40,
        pretrain_throttle=0,
    )
    cfg = GridConfig(
        sweep=cfg.sweep, solvers=cfg.solvers, hyperparams=cfg.hyperparams,
        inputs=cfg.inputs, pretrain=cfg.pretrain, cluster=bad,
        domain_profile=cfg.domain_profile,
    )
    with pytest.raises(ValueError, match="pretrain_throttle must be >= 1"):
        validate_grid_semantics(cfg, _StubDomain(pool_size=40))


def test_validate_bad_pretrain_cpus_per_task():
    cfg = _cfg()
    bad = ClusterResources(
        partition="p", time="1:00:00", mem="8G", cpus_per_task=1,
        array_throttle=1, eval_array_throttle=1, max_concurrent_tasks=40,
        pretrain_cpus_per_task=0,
    )
    cfg = GridConfig(
        sweep=cfg.sweep, solvers=cfg.solvers, hyperparams=cfg.hyperparams,
        inputs=cfg.inputs, pretrain=cfg.pretrain, cluster=bad,
        domain_profile=cfg.domain_profile,
    )
    with pytest.raises(ValueError, match="pretrain_cpus_per_task must be >= 1"):
        validate_grid_semantics(cfg, _StubDomain(pool_size=40))


def test_validate_pretrain_resource_knobs_none_default_ok():
    """The pretrain resource knobs are None-by-default and fall back to the
    train-array values, an unset config validates cleanly."""
    cfg = _cfg()
    assert cfg.cluster.pretrain_partition is None
    assert cfg.cluster.pretrain_time is None
    assert cfg.cluster.pretrain_mem is None
    assert cfg.cluster.pretrain_cpus_per_task is None
    assert cfg.cluster.pretrain_throttle is None
    validate_grid_semantics(cfg, _StubDomain(pool_size=40))


def test_load_parses_pretrain_resource_knobs(tmp_path):
    """load_grid_config parses the optional cluster.pretrain_* resource knobs."""
    d = _base_config_dict()
    d["cluster"].update({
        "pretrain_partition": "long-96core",
        "pretrain_time": "08:00:00",
        "pretrain_mem": "64G",
        "pretrain_cpus_per_task": 12,
        "pretrain_throttle": 3,
    })
    cfg = load_grid_config(_write(tmp_path, "grid.json", d))
    cl = cfg.cluster
    assert cl.pretrain_partition == "long-96core"
    assert cl.pretrain_time == "08:00:00"
    assert cl.pretrain_mem == "64G"
    assert cl.pretrain_cpus_per_task == 12
    assert cl.pretrain_throttle == 3


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


def test_load_missing_pretrain_section(tmp_path):
    data = _base_config_dict()
    del data["pretrain"]
    path = _write(tmp_path, "grid.json", data)
    with pytest.raises(ValueError, match="pretrain"):
        load_grid_config(path)


def test_load_missing_pretrain_required_key(tmp_path):
    data = _base_config_dict()
    del data["pretrain"]["data_dir"]
    path = _write(tmp_path, "grid.json", data)
    with pytest.raises(ValueError, match="pretrain.data_dir"):
        load_grid_config(path)


def _cfg_with_pretrain(pt: PretrainConfig) -> GridConfig:
    """Build a GridConfig from the _cfg() base with a substituted pretrain."""
    base = _cfg()
    return GridConfig(
        sweep=base.sweep, solvers=base.solvers, hyperparams=base.hyperparams,
        inputs=base.inputs, pretrain=pt, cluster=base.cluster,
        domain_profile=base.domain_profile,
    )


def test_validate_pretrain_n_steps_nonpositive():
    cfg = _cfg_with_pretrain(PretrainConfig(
        data_dir="/shared/pretrain_data",
        n_steps=0,
    ))
    with pytest.raises(ValueError, match="pretrain.n_steps must be > 0"):
        validate_grid_semantics(cfg, _StubDomain(pool_size=40))


def test_validate_pretrain_empty_data_dir():
    cfg = _cfg_with_pretrain(PretrainConfig(
        data_dir="",
    ))
    with pytest.raises(ValueError, match="pretrain.data_dir"):
        validate_grid_semantics(cfg, _StubDomain(pool_size=40))


def test_validate_pretrain_bad_loss_weighting():
    cfg = _cfg_with_pretrain(PretrainConfig(
        data_dir="/shared/pretrain_data",
        loss_weighting="bogus",
    ))
    with pytest.raises(ValueError, match="pretrain.loss_weighting"):
        validate_grid_semantics(cfg, _StubDomain(pool_size=40))


def test_validate_pretrain_lr_decay_out_of_range():
    cfg = _cfg_with_pretrain(PretrainConfig(
        data_dir="/shared/pretrain_data",
        lr_decay_start=1.5,
    ))
    with pytest.raises(ValueError, match="pretrain.lr_decay_start"):
        validate_grid_semantics(cfg, _StubDomain(pool_size=40))


# ---------------------------------------------------------------------------
# Per-stage allocation mode + optional mem
# ---------------------------------------------------------------------------

def test_allocation_defaults_to_exclusive(tmp_path):
    """A config that omits the per-stage allocation knobs defaults every stage
    to whole-node 'exclusive' (training peaks near a full node's RAM, so
    whole-node is the safe default)."""
    cfg = load_grid_config(_write(tmp_path, "g.json", _base_config_dict()))
    assert cfg.cluster.train_allocation == "exclusive"
    assert cfg.cluster.eval_allocation == "exclusive"
    assert cfg.cluster.preflight_allocation == "exclusive"
    assert cfg.cluster.pretrain_allocation == "exclusive"


def test_load_parses_per_stage_allocation(tmp_path):
    """Per-stage allocation modes round-trip through load_grid_config."""
    d = _base_config_dict()
    d["cluster"]["train_allocation"] = "exclusive"
    d["cluster"]["eval_allocation"] = "shared"
    d["cluster"]["preflight_allocation"] = "shared"
    d["cluster"]["pretrain_allocation"] = "exclusive"
    cfg = load_grid_config(_write(tmp_path, "g.json", d))
    assert cfg.cluster.train_allocation == "exclusive"
    assert cfg.cluster.eval_allocation == "shared"
    assert cfg.cluster.preflight_allocation == "shared"
    assert cfg.cluster.pretrain_allocation == "exclusive"


def test_mem_is_optional(tmp_path):
    """mem is no longer required, a config that omits it loads with mem=''
    (whole-node/exclusive stages need no --mem; SLURM applies the partition
    default for any shared stage that also leaves mem unset)."""
    d = _base_config_dict()
    d["cluster"].pop("mem", None)
    cfg = load_grid_config(_write(tmp_path, "g.json", d))
    assert cfg.cluster.mem == ""


def test_validate_rejects_unknown_allocation():
    """An allocation mode other than 'exclusive'/'shared' is a hard error."""
    import dataclasses
    base = _cfg()
    bad = dataclasses.replace(base.cluster, train_allocation="whole-cluster")
    cfg = dataclasses.replace(base, cluster=bad)
    with pytest.raises(ValueError, match="train_allocation"):
        validate_grid_semantics(cfg, _StubDomain(pool_size=40))


def test_validate_rejects_unknown_datagen_allocation():
    """datagen_allocation is validated too, an invalid value must be rejected,
    not silently rendered as a SHARED datagen job (drops --nodes=1 --exclusive)."""
    import dataclasses
    base = _cfg()
    bad = dataclasses.replace(base.cluster, datagen_allocation="whole-cluster")
    cfg = dataclasses.replace(base, cluster=bad)
    with pytest.raises(ValueError, match="datagen_allocation"):
        validate_grid_semantics(cfg, _StubDomain(pool_size=40))


# ---------------------------------------------------------------------------
# pretrain_checkpoint_dir: run-scoped pretrain output path
# ---------------------------------------------------------------------------

def test_pretrain_checkpoint_dir_is_under_run_dir():
    """The pretrain checkpoint dir lives under the run dir at
    ``<run_dir>/pretrain/<arch>``, co-locating it with the run's other
    artifacts. Because run_dir is unique per submission, two runs that pretrain
    the SAME architecture still resolve to distinct dirs (no clobbering)."""
    from xcquinox.alec.cluster.grid_config import pretrain_checkpoint_dir

    p1 = pretrain_checkpoint_dir(
        "/scratch/runs/run_AAA", "deep_combined_attn"
    )
    assert p1 == "/scratch/runs/run_AAA/pretrain/deep_combined_attn"

    p2 = pretrain_checkpoint_dir(
        "/scratch/runs/run_BBB", "deep_combined_attn"
    )
    assert p2 == "/scratch/runs/run_BBB/pretrain/deep_combined_attn"

    # Same arch, different run -> distinct dirs (run_dir is the uniqueness key).
    assert p1 != p2


def test_pretrain_checkpoint_dir_normalizes_trailing_sep():
    """A trailing slash on run_dir does not produce a doubled separator."""
    from xcquinox.alec.cluster.grid_config import pretrain_checkpoint_dir
    assert pretrain_checkpoint_dir(
        "/scratch/runs/run_AAA/", "medium"
    ) == "/scratch/runs/run_AAA/pretrain/medium"


# ---------------------------------------------------------------------------
# use_polarized_correlation (run-level spin-polarized correlation toggle)
# ---------------------------------------------------------------------------

def test_use_polarized_correlation_default_and_parse(tmp_path):
    # Default: absent key -> False (byte-identical unpolarized behavior).
    cfg = load_grid_config(_write(tmp_path, "g.json", _base_config_dict()))
    assert cfg.use_polarized_correlation is False
    # Explicit true parses through.
    data = _base_config_dict()
    data["use_polarized_correlation"] = True
    cfg2 = load_grid_config(_write(tmp_path, "g2.json", data))
    assert cfg2.use_polarized_correlation is True


# ---------------------------------------------------------------------------
# defer_eval (run-level deferred-eval submission toggle)
# ---------------------------------------------------------------------------

def test_defer_eval_default_and_parse(tmp_path):
    # Default: absent key -> False (byte-identical: eval submitted up front).
    cfg = load_grid_config(_write(tmp_path, "g.json", _base_config_dict()))
    assert cfg.defer_eval is False
    # Explicit true parses through.
    data = _base_config_dict()
    data["defer_eval"] = True
    cfg2 = load_grid_config(_write(tmp_path, "g2.json", data))
    assert cfg2.defer_eval is True
