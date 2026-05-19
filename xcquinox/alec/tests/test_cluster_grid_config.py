"""Tests for xcquinox.alec.cluster.grid_config — the HPC harness config layer."""
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
    """Minimal stand-in for the not-yet-built DomainProfile — exposes only the
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
            "pretrain_root": "/shared/pretrain",
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
# load_grid_config — round-trips
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
    assert cfg.pretrain.pretrain_root == "/shared/pretrain"
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
            pretrain_root="/shared/pretrain",
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
    assert pt.pretrain_root == "/shared/pretrain"
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
        "pretrain_root": "/shared/pretrain",
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
        data_dir="/shared/pretrain_data", pretrain_root="/shared/pretrain",
        n_steps=0,
    ))
    with pytest.raises(ValueError, match="pretrain.n_steps must be > 0"):
        validate_grid_semantics(cfg, _StubDomain(pool_size=40))


def test_validate_pretrain_empty_data_dir():
    cfg = _cfg_with_pretrain(PretrainConfig(
        data_dir="", pretrain_root="/shared/pretrain",
    ))
    with pytest.raises(ValueError, match="pretrain.data_dir"):
        validate_grid_semantics(cfg, _StubDomain(pool_size=40))


def test_validate_pretrain_bad_loss_weighting():
    cfg = _cfg_with_pretrain(PretrainConfig(
        data_dir="/shared/pretrain_data", pretrain_root="/shared/pretrain",
        loss_weighting="bogus",
    ))
    with pytest.raises(ValueError, match="pretrain.loss_weighting"):
        validate_grid_semantics(cfg, _StubDomain(pool_size=40))


def test_validate_pretrain_lr_decay_out_of_range():
    cfg = _cfg_with_pretrain(PretrainConfig(
        data_dir="/shared/pretrain_data", pretrain_root="/shared/pretrain",
        lr_decay_start=1.5,
    ))
    with pytest.raises(ValueError, match="pretrain.lr_decay_start"):
        validate_grid_semantics(cfg, _StubDomain(pool_size=40))
