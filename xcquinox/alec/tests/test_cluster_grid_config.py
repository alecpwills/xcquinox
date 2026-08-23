"""Tests for xcquinox.alec.cluster.grid_config: the HPC harness config layer."""
import json
import re

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


# ---------------------------------------------------------------------------
# Solver mixer_kwargs: resolved_config.yaml round-trip (datagen regression)
# ---------------------------------------------------------------------------

def _mixer_config_dict():
    """A base config whose swept solver carries the DFS step-decaying mixer +
    tail-loss knobs -- the exact form the dfs_step7 v3 configs use."""
    d = _base_config_dict()
    d["solvers"]["robust"].update({
        "mixer_name": "decaying_linear",
        "mixer_kwargs": {"base": 0.3, "floor": 0.3},
        "scf_loss_use_tail": True,
        "scf_loss_tail": 10,
        "scf_loss_weight_power": 2.0,
    })
    return d


def test_solver_mixer_kwargs_dict_form_parses(tmp_path):
    """A user-authored {base: .., floor: ..} mapping parses to a sorted
    hashable tuple-of-pairs on SolverNamed (and the scalar tail knobs ride
    along)."""
    cfg = load_grid_config(_write(tmp_path, "grid.yaml", _mixer_config_dict()))
    sv = cfg.solvers["robust"]
    assert sv.mixer_name == "decaying_linear"
    assert sv.mixer_kwargs == (("base", 0.3), ("floor", 0.3))
    assert sv.scf_loss_use_tail is True
    assert sv.scf_loss_tail == 10
    assert sv.scf_loss_weight_power == 2.0


def test_solver_mixer_kwargs_resolved_round_trip(tmp_path):
    """REGRESSION (datagen crash): submit writes resolved_config.yaml by
    serializing each SolverNamed through dataclasses.asdict (keeps the
    mixer_kwargs tuple) then yaml.safe_dump (which writes tuples as YAML
    sequences); reloading parses them back as nested LISTS. Re-loading that
    resolved config -- which datagen/pretrain/preflight/eval all do -- must NOT
    raise 'mixer_kwargs must be a mapping, got list'. Uses the real production
    serializer so the test tracks any future serialization change."""
    from xcquinox.alec.cluster.__main__ import _config_to_raw_dict
    yaml = pytest.importorskip("yaml")
    cfg = load_grid_config(_write(tmp_path, "grid.yaml", _mixer_config_dict()))
    # exactly what `submit` writes to resolved_config.yaml:
    resolved = _config_to_raw_dict(cfg)
    resolved_path = _write(tmp_path, "resolved.yaml", resolved)
    # the on-disk form datagen actually re-reads: the yaml dump->load turned the
    # tuple-of-pairs into list-of-lists (else this test would not exercise the bug):
    with open(resolved_path) as fh:
        on_disk = yaml.safe_load(fh)
    assert isinstance(on_disk["solvers"]["robust"]["mixer_kwargs"], list)
    # re-load the resolved config, as every downstream stage does:
    cfg2 = load_grid_config(resolved_path)
    sv = cfg2.solvers["robust"]
    assert sv.mixer_kwargs == (("base", 0.3), ("floor", 0.3))
    assert sv.mixer_name == "decaying_linear"
    # the scalar knobs survive the round-trip too:
    assert sv.scf_loss_use_tail is True
    assert sv.scf_loss_tail == 10
    assert sv.scf_loss_weight_power == 2.0


# ---------------------------------------------------------------------------
# Solver orientation_lock_strength: parse, default-off, resolved round-trip
# ---------------------------------------------------------------------------

def test_solver_orientation_lock_default_off(tmp_path):
    """Solvers without the key parse to strength 0.0 (byte-identical / off)."""
    cfg = load_grid_config(_write(tmp_path, "grid.yaml", _base_config_dict()))
    assert cfg.solvers["robust"].orientation_lock_strength == 0.0


def test_solver_orientation_lock_parses_and_reaches_solver_config(tmp_path):
    from xcquinox.alec.cluster.spec_builder import _solver_config_from_named
    from xcquinox.alec.cluster.grid_config import SolverNamed
    # the scalar lands on the parsed SolverNamed
    d = _base_config_dict()
    d["solvers"]["robust"]["orientation_lock_strength"] = 3e-5
    cfg = load_grid_config(_write(tmp_path, "grid.yaml", d))
    assert cfg.solvers["robust"].orientation_lock_strength == 3e-5
    # and it materializes onto the runtime SolverConfig (valid oneshot solver)
    sv = SolverNamed(mode="oneshot", max_cycles=0, orientation_lock_strength=3e-5)
    sc = _solver_config_from_named(sv)
    assert sc.orientation_lock_strength == 3e-5


def test_solver_orientation_lock_resolved_round_trip(tmp_path):
    """The scalar survives asdict + yaml.safe_dump + reload (the datagen path),
    like the mixer/tail knobs -- guards against the full25 list/dict crash class."""
    from xcquinox.alec.cluster.__main__ import _config_to_raw_dict
    yaml = pytest.importorskip("yaml")
    d = _base_config_dict()
    d["solvers"]["robust"]["orientation_lock_strength"] = 3e-5
    cfg = load_grid_config(_write(tmp_path, "grid.yaml", d))
    resolved_path = _write(tmp_path, "resolved.yaml", _config_to_raw_dict(cfg))
    cfg2 = load_grid_config(resolved_path)
    assert cfg2.solvers["robust"].orientation_lock_strength == 3e-5


# ---------------------------------------------------------------------------
# Run-level inputs.orientation_lock_strength (authoritative for the whole run)
# ---------------------------------------------------------------------------

def test_inputs_orientation_lock_default_off(tmp_path):
    cfg = load_grid_config(_write(tmp_path, "grid.yaml", _base_config_dict()))
    assert cfg.inputs.orientation_lock_strength == 0.0


def test_inputs_orientation_lock_parses_roundtrips_and_is_authoritative(tmp_path):
    """The run-level inputs value parses, survives the resolved-config reload, and
    is threaded into the SolverConfig (overriding the per-solver value) so the
    training/eval SCF locks the same component as the references."""
    from xcquinox.alec.cluster.__main__ import _config_to_raw_dict
    from xcquinox.alec.cluster.spec_builder import _solver_config_from_named
    from xcquinox.alec.cluster.grid_config import SolverNamed
    d = _base_config_dict()
    d["inputs"]["orientation_lock_strength"] = 3e-5
    cfg = load_grid_config(_write(tmp_path, "grid.yaml", d))
    assert cfg.inputs.orientation_lock_strength == 3e-5
    cfg2 = load_grid_config(_write(tmp_path, "resolved.yaml", _config_to_raw_dict(cfg)))
    assert cfg2.inputs.orientation_lock_strength == 3e-5
    # run-level value wins over a per-solver 0.0 (the SCF must match the refs)
    sv = SolverNamed(mode="oneshot", max_cycles=0, orientation_lock_strength=0.0)
    sc = _solver_config_from_named(
        sv, orientation_lock_strength=cfg.inputs.orientation_lock_strength)
    assert sc.orientation_lock_strength == 3e-5


def test_parse_mixer_kwargs_accepts_dict_and_list():
    """_parse_mixer_kwargs accepts both the user {name: value} dict and the
    round-tripped [name, value]-pair list, returns None for empty/absent, and
    rejects a genuinely-malformed scalar with a contextual error."""
    from xcquinox.alec.cluster.grid_config import _parse_mixer_kwargs
    ctx = "solvers.full_25"
    want = (("base", 0.3), ("floor", 0.3))
    assert _parse_mixer_kwargs({"base": 0.3, "floor": 0.3}, ctx) == want
    assert _parse_mixer_kwargs([["base", 0.3], ["floor", 0.3]], ctx) == want
    # tuple-of-pairs (jit form) is accepted verbatim too
    assert _parse_mixer_kwargs((("base", 0.3), ("floor", 0.3)), ctx) == want
    assert _parse_mixer_kwargs(None, ctx) is None
    assert _parse_mixer_kwargs({}, ctx) is None
    with pytest.raises(ValueError, match=r"solvers\.full_25\.mixer_kwargs"):
        _parse_mixer_kwargs(0.5, ctx)


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


def test_validate_rejects_validate_every_with_batched_scheme():
    """FIX 1(b): only _run_per_molecule_loop has the validation hook; with
    update_scheme='batched' validate_every>0 would never validate. Reject it."""
    cfg = _cfg_with(
        hp_kwargs=dict(validate_every=2, update_scheme="batched"),
        inputs_kwargs=dict(val_refs_dir="/shared/val_refs"))
    with pytest.raises(ValueError, match="per_molecule"):
        validate_grid_semantics(cfg, _StubDomain(pool_size=40))


def test_validate_accepts_validate_every_when_fully_configured():
    """validate_every>0 + val_refs_dir set + per_molecule scheme validates."""
    cfg = _cfg_with(
        hp_kwargs=dict(validate_every=2, update_scheme="per_molecule"),
        inputs_kwargs=dict(val_refs_dir="/shared/val_refs"))
    validate_grid_semantics(cfg, _StubDomain(pool_size=40))


def test_validate_default_no_validation_is_clean():
    """validate_every=0 (default) needs neither val_refs_dir nor per_molecule;
    the cross-field guard must not fire for the no-op default."""
    cfg = _cfg_with(hp_kwargs=dict(validate_every=0, update_scheme="batched"),
                    inputs_kwargs=dict(val_refs_dir=None))
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


def test_validate_accepts_early_stop_with_enough_checks():
    # n_steps=500 / validate_every=25 -> 20 checks; patience=5 has ample slack
    # (streak can reach 5 long before the end), so the geometry guard must NOT fire.
    cfg = _cfg_with(
        hp_kwargs=dict(n_steps=500, validate_every=25, patience=5,
                       update_scheme="per_molecule"),
        inputs_kwargs=dict(val_refs_dir="/shared/val_refs"))
    validate_grid_semantics(cfg, _StubDomain(pool_size=40))


def test_validate_early_stop_guard_ignores_patience_zero():
    # patience=0 disables early-stop (val-best tracking only); the geometry guard
    # must not fire even when n_checks is tiny.
    cfg = _cfg_with(
        hp_kwargs=dict(n_steps=150, validate_every=25, patience=0,
                       update_scheme="per_molecule"),
        inputs_kwargs=dict(val_refs_dir="/shared/val_refs"))
    validate_grid_semantics(cfg, _StubDomain(pool_size=40))


def test_validate_early_stop_message_n_checks_is_int():
    # a float n_steps must not leak "n_checks=6.0" into the actionable message.
    cfg = _cfg_with(
        hp_kwargs=dict(n_steps=150.0, validate_every=25, patience=5,
                       update_scheme="per_molecule"),
        inputs_kwargs=dict(val_refs_dir="/shared/val_refs"))
    with pytest.raises(ValueError) as exc:
        validate_grid_semantics(cfg, _StubDomain(pool_size=40))
    assert "n_checks=6" in str(exc.value)
    assert "n_checks=6.0" not in str(exc.value)


def test_validate_rejects_bad_val_frac():
    """FIX 2 (WS3-CFG-2): val_frac must be in (0, 1)."""
    for bad in (0.0, 1.0, -0.1, 1.5):
        cfg = _cfg_with(hp_kwargs=dict(val_frac=bad))
        with pytest.raises(ValueError, match="val_frac"):
            validate_grid_semantics(cfg, _StubDomain(pool_size=40))


def test_validate_rejects_negative_validate_every():
    """FIX 2 (WS3-CFG-2): validate_every must be >= 0."""
    cfg = _cfg_with(hp_kwargs=dict(validate_every=-1))
    with pytest.raises(ValueError, match="validate_every"):
        validate_grid_semantics(cfg, _StubDomain(pool_size=40))


def test_validate_rejects_negative_patience():
    """FIX 2 (WS3-CFG-2): patience must be >= 0."""
    cfg = _cfg_with(hp_kwargs=dict(patience=-1))
    with pytest.raises(ValueError, match="patience"):
        validate_grid_semantics(cfg, _StubDomain(pool_size=40))


def test_validate_rejects_negative_early_stop_min_delta():
    """FIX 2 (WS3-CFG-2): early_stop_min_delta must be >= 0."""
    cfg = _cfg_with(hp_kwargs=dict(early_stop_min_delta=-0.1))
    with pytest.raises(ValueError, match="early_stop_min_delta"):
        validate_grid_semantics(cfg, _StubDomain(pool_size=40))


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


def test_hyperparams_density_per_electron_optional_default_false():
    from xcquinox.alec.cluster.grid_config import _build_hyperparams
    base = {"n_steps": 1, "lr_start": 1e-2, "lr_end": 1e-5,
            "lr_decay_start": 0.2, "grad_clip": 1.0, "gradnorm_alpha": 1.5,
            "vxc_weight": 0.01, "density_weight": 0.1}
    hp = _build_hyperparams(dict(base))
    assert hp.density_per_electron is False           # byte-identical default
    hp_on = _build_hyperparams(dict(base, density_per_electron=True))
    assert hp_on.density_per_electron is True


# 2026-06-20 (WS3): held-out validation slice knobs drive in-loop early-stop +
# validation-best selection. All MUST default to a NO-OP so decay-free runs stay
# byte-identical (validate_every=0 -> no in-loop validation; patience=0 -> no
# early-stop).
def test_hyperparams_validation_knobs_default_noop():
    from xcquinox.alec.cluster.grid_config import _build_hyperparams
    base = {"n_steps": 1, "lr_start": 1e-2, "lr_end": 1e-5,
            "lr_decay_start": 0.2, "grad_clip": 1.0, "gradnorm_alpha": 1.5,
            "vxc_weight": 0.01, "density_weight": 0.1}
    hp = _build_hyperparams(dict(base))
    assert hp.val_frac == 0.2                 # default split fraction
    assert hp.validate_every == 0             # no in-loop validation
    assert hp.patience == 0                   # no early-stop
    assert hp.early_stop_min_delta == 0.0


def test_hyperparams_validation_knobs_override():
    from xcquinox.alec.cluster.grid_config import _build_hyperparams
    base = {"n_steps": 1, "lr_start": 1e-2, "lr_end": 1e-5,
            "lr_decay_start": 0.2, "grad_clip": 1.0, "gradnorm_alpha": 1.5,
            "vxc_weight": 0.01, "density_weight": 0.1}
    hp = _build_hyperparams(dict(base, val_frac=0.25, validate_every=10,
                                  patience=5, early_stop_min_delta=0.01))
    assert hp.val_frac == 0.25
    assert hp.validate_every == 10
    assert hp.patience == 5
    assert hp.early_stop_min_delta == 0.01


# WS5 (2026-06-20): periodic-resume checkpoint cadence; default 0 => no-op so
# existing sweeps stay byte-identical.
def test_hyperparams_checkpoint_every_default_noop():
    from xcquinox.alec.cluster.grid_config import _build_hyperparams
    base = {"n_steps": 1, "lr_start": 1e-2, "lr_end": 1e-5,
            "lr_decay_start": 0.2, "grad_clip": 1.0, "gradnorm_alpha": 1.5,
            "vxc_weight": 0.01, "density_weight": 0.1}
    hp = _build_hyperparams(dict(base))
    assert hp.checkpoint_every == 0


def test_hyperparams_checkpoint_every_override():
    from xcquinox.alec.cluster.grid_config import _build_hyperparams
    base = {"n_steps": 1, "lr_start": 1e-2, "lr_end": 1e-5,
            "lr_decay_start": 0.2, "grad_clip": 1.0, "gradnorm_alpha": 1.5,
            "vxc_weight": 0.01, "density_weight": 0.1}
    hp = _build_hyperparams(dict(base, checkpoint_every=25))
    assert hp.checkpoint_every == 25


def test_inputs_val_refs_dir_optional_default_none():
    from xcquinox.alec.cluster.grid_config import _build_inputs
    base = {"external_refs_dir": "/refs", "subset_ledger_path": "/led.json",
            "basis": "def2-svp", "grid_level": 1, "output_root": "/out"}
    cfg_in = _build_inputs(dict(base))
    assert cfg_in.val_refs_dir is None                # byte-identical default
    cfg_on = _build_inputs(dict(base, val_refs_dir="/val_refs"))
    assert cfg_on.val_refs_dir == "/val_refs"


# 2026-06-20 (WS4): a named solver entry may opt into SCF gradient checkpointing
# (for full_25); the parser must read it, defaulting off.
def test_build_solvers_parses_scf_grad_checkpoint():
    from xcquinox.alec.cluster.grid_config import _build_solvers
    solvers = _build_solvers({
        "full_25": {"mode": "FULL", "max_cycles": 25, "scf_grad_checkpoint": True},
        "full_3": {"mode": "FULL", "max_cycles": 3},
    })
    assert solvers["full_25"].scf_grad_checkpoint is True
    assert solvers["full_3"].scf_grad_checkpoint is False


# ---------------------------------------------------------------------------
# Per-rung SCF seeding knobs (inputs.seed_xc / seed_cache_dir) + eval_coldstart
# ---------------------------------------------------------------------------

def test_inputs_seed_defaults_keep_every_run_on_pbe(tmp_path):
    """A config that never mentions the knobs parses to the pre-seeding
    protocol: seed_xc 'pbe', no cache dir, no coldstart channel."""
    cfg = load_grid_config(_write(tmp_path, "grid.yaml", _base_config_dict()))
    assert cfg.inputs.seed_xc == "pbe"
    assert cfg.inputs.seed_cache_dir is None
    assert cfg.eval_coldstart is False


def test_inputs_seed_xc_value_validated(tmp_path):
    d = _base_config_dict()
    d["inputs"]["seed_xc"] = "b3lyp"
    with pytest.raises(ValueError):
        load_grid_config(_write(tmp_path, "grid.yaml", d))


def test_seed_and_coldstart_resolved_round_trip(tmp_path):
    """seed_xc / seed_cache_dir / eval_coldstart survive asdict + yaml +
    reload (the resolved_config.yaml path the preflight re-reads) -- the
    ae_as_reactions silent-drop incident class."""
    from xcquinox.alec.cluster.__main__ import _config_to_raw_dict
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

def test_fidelity_defaults_to_the_binding_tolerances(tmp_path):
    """A config with no fidelity block carries tol_AE = 1.0 kcal/mol and
    tol_atom = 1.0 mHa, so every YAML written before the certificate existed
    loads at the binding tolerances rather than at no tolerance."""
    from xcquinox.alec.cluster.grid_config import FidelityConfig
    cfg = load_grid_config(_write(tmp_path, "grid.yaml", _base_config_dict()))
    assert isinstance(cfg.fidelity, FidelityConfig)
    assert cfg.fidelity.tol_AE == 1.0
    assert cfg.fidelity.tol_atom == 1.0
    assert cfg.fidelity.override_reason is None
    assert cfg.fidelity.enforce is True


def test_fidelity_block_parses(tmp_path):
    raw = _base_config_dict()
    raw["fidelity"] = {"tol_AE": 0.5, "tol_atom": 0.25,
                       "override_reason": None}
    cfg = load_grid_config(_write(tmp_path, "grid.yaml", raw))
    assert cfg.fidelity.tol_AE == 0.5
    assert cfg.fidelity.tol_atom == 0.25


def test_fidelity_block_must_be_a_mapping(tmp_path):
    raw = _base_config_dict()
    raw["fidelity"] = [1.0, 1.0]
    with pytest.raises(ValueError, match="fidelity"):
        load_grid_config(_write(tmp_path, "grid.yaml", raw))


def test_fidelity_resolved_round_trip(tmp_path):
    """The resolved config is re-read by the pretrain, preflight and eval
    stages; a dropped fidelity block would silently revert a documented
    override to the binding tolerances mid-run."""
    from xcquinox.alec.cluster.__main__ import _config_to_raw_dict
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


def test_validate_rejects_a_loose_atom_tolerance_without_an_override_reason(
        tmp_path):
    raw = _base_config_dict()
    raw["fidelity"] = {"tol_AE": 1.0, "tol_atom": 2.5,
                       "override_reason": "   "}
    cfg = load_grid_config(_write(tmp_path, "grid.yaml", raw))
    with pytest.raises(ValueError, match="override_reason"):
        validate_grid_semantics(cfg, _StubDomain(pool_size=100))


def test_validate_accepts_a_loose_tolerance_with_an_override_reason(tmp_path):
    raw = _base_config_dict()
    raw["fidelity"] = {"tol_AE": 3.0, "tol_atom": 3.0,
                       "override_reason": "descriptor-free control arm, "
                                          "documented in HISTORY 2026-08-21"}
    cfg = load_grid_config(_write(tmp_path, "grid.yaml", raw))
    validate_grid_semantics(cfg, _StubDomain(pool_size=100))


def test_validate_accepts_the_ceiling_without_an_override_reason(tmp_path):
    """2.0 / 2.0 is the ceiling, not past it."""
    raw = _base_config_dict()
    raw["fidelity"] = {"tol_AE": 2.0, "tol_atom": 2.0}
    cfg = load_grid_config(_write(tmp_path, "grid.yaml", raw))
    validate_grid_semantics(cfg, _StubDomain(pool_size=100))


def test_validate_rejects_disabled_enforcement_without_a_reason(tmp_path):
    """Turning the on-node gates off is a documented decision or it does not
    happen: the reason is copied into every certificate the run writes."""
    raw = _base_config_dict()
    raw["fidelity"] = {"tol_AE": 1.0, "tol_atom": 1.0, "enforce": False}
    cfg = load_grid_config(_write(tmp_path, "grid.yaml", raw))
    with pytest.raises(ValueError, match="override_reason"):
        validate_grid_semantics(cfg, _StubDomain(pool_size=100))


def test_validate_accepts_disabled_enforcement_with_a_reason(tmp_path):
    raw = _base_config_dict()
    raw["fidelity"] = {"tol_AE": 1.0, "tol_atom": 1.0, "enforce": False,
                       "override_reason": "workflow-verification matrix, "
                                          "50-step pretrain"}
    cfg = load_grid_config(_write(tmp_path, "grid.yaml", raw))
    validate_grid_semantics(cfg, _StubDomain(pool_size=100))
    assert cfg.fidelity.enforce is False


def test_fidelity_enforce_round_trips(tmp_path):
    from xcquinox.alec.cluster.__main__ import _config_to_raw_dict
    raw = _base_config_dict()
    raw["fidelity"] = {"tol_AE": 1.0, "tol_atom": 1.0, "enforce": False,
                       "override_reason": "workflow matrix"}
    cfg = load_grid_config(_write(tmp_path, "grid.yaml", raw))
    cfg2 = load_grid_config(
        _write(tmp_path, "resolved2.yaml", _config_to_raw_dict(cfg)))
    assert cfg2.fidelity.enforce is False
    assert cfg2.fidelity == cfg.fidelity


def test_validate_rejects_a_nonpositive_tolerance(tmp_path):
    raw = _base_config_dict()
    raw["fidelity"] = {"tol_AE": 0.0, "tol_atom": 1.0}
    cfg = load_grid_config(_write(tmp_path, "grid.yaml", raw))
    with pytest.raises(ValueError, match="tol_AE must be > 0"):
        validate_grid_semantics(cfg, _StubDomain(pool_size=100))


def test_validate_rejects_a_nonpositive_atom_tolerance(tmp_path):
    """Mirror of the tol_AE floor on the free-atom leg. Both legs are
    independent gates of the certificate, so a zero or negative tol_atom is a
    config error in its own right: no measurement satisfies |dE_xc| <= 0."""
    for bad in (0.0, -0.5):
        raw = _base_config_dict()
        raw["fidelity"] = {"tol_AE": 1.0, "tol_atom": bad}
        cfg = load_grid_config(_write(tmp_path, f"grid_{bad}.yaml", raw))
        with pytest.raises(ValueError, match="tol_atom must be > 0"):
            validate_grid_semantics(cfg, _StubDomain(pool_size=100))


def test_validate_rejects_a_negative_ae_tolerance(tmp_path):
    raw = _base_config_dict()
    raw["fidelity"] = {"tol_AE": -1.0, "tol_atom": 1.0}
    cfg = load_grid_config(_write(tmp_path, "grid.yaml", raw))
    with pytest.raises(ValueError, match="tol_AE must be > 0"):
        validate_grid_semantics(cfg, _StubDomain(pool_size=100))


@pytest.mark.parametrize("bad", [False, True, 0, 1, 2.5, [1.0], {"why": "x"}])
def test_fidelity_override_reason_must_be_a_string(tmp_path, bad):
    """A non-string override_reason is refused at load rather than coerced.

    ``str(False)`` is the non-empty string ``'False'``, so coercing whatever
    the YAML carried would let ``override_reason: false`` (and its YAML
    synonym ``no``, and a bare ``0``) authorise a loosened tolerance -- the
    opposite of what the author wrote. The reason is prose copied verbatim
    into every certificate the run writes, so only a string is one."""
    raw = _base_config_dict()
    raw["fidelity"] = {"tol_AE": 1.0, "tol_atom": 1.0, "override_reason": bad}
    with pytest.raises(ValueError, match="override_reason"):
        load_grid_config(_write(tmp_path, "grid.json", raw))


def test_fidelity_override_reason_false_does_not_authorise_a_loosening(tmp_path):
    """The exploit the string coercion opened, end to end: a YAML boolean
    ``false`` next to a 3.0 kcal/mol tolerance must not reach
    validate_grid_semantics as the reason 'False'."""
    yaml = pytest.importorskip("yaml")
    raw = _base_config_dict()
    path = tmp_path / "grid.yaml"
    path.write_text(
        yaml.safe_dump(raw)
        + "fidelity:\n  tol_AE: 3.0\n  tol_atom: 1.0\n"
          "  override_reason: false\n"
    )
    with pytest.raises(ValueError, match="override_reason"):
        load_grid_config(str(path))


def test_fidelity_override_reason_yaml_no_is_refused(tmp_path):
    """``no`` is YAML 1.1 for the boolean False, so an author writing
    ``override_reason: no`` (meaning 'no reason') must not have it coerced to
    the authorising string 'False'."""
    yaml = pytest.importorskip("yaml")
    raw = _base_config_dict()
    path = tmp_path / "grid.yaml"
    path.write_text(
        yaml.safe_dump(raw)
        + "fidelity:\n  tol_AE: 3.0\n  tol_atom: 1.0\n"
          "  override_reason: no\n"
    )
    with pytest.raises(ValueError, match="override_reason"):
        load_grid_config(str(path))


def test_fidelity_override_reason_accepts_a_string(tmp_path):
    """The guard refuses only non-strings; a real reason still loads."""
    raw = _base_config_dict()
    raw["fidelity"] = {"tol_AE": 3.0, "tol_atom": 3.0,
                       "override_reason": "descriptor-free control arm"}
    cfg = load_grid_config(_write(tmp_path, "grid.json", raw))
    assert cfg.fidelity.override_reason == "descriptor-free control arm"
    validate_grid_semantics(cfg, _StubDomain(pool_size=100))


@pytest.mark.parametrize("key", ["tol_AE", "tol_atom"])
@pytest.mark.parametrize("bad", [True, False, None, [1.0], {"v": 1.0}])
def test_fidelity_tolerances_must_be_numbers(tmp_path, key, bad):
    """A tolerance is a measured energy bound, so a boolean, a null or a
    container is a config error named at load. ``float(True)`` is 1.0 and
    ``float(None)`` raises TypeError past every ValueError handler, so neither
    may reach FidelityConfig."""
    raw = _base_config_dict()
    raw["fidelity"] = {"tol_AE": 1.0, "tol_atom": 1.0, key: bad}
    with pytest.raises(ValueError, match=f"fidelity.{key}"):
        load_grid_config(_write(tmp_path, "grid.json", raw))


@pytest.mark.parametrize("good,expected", [(2, 2.0), (0.5, 0.5), ("1.5", 1.5)])
def test_fidelity_tolerances_accept_numeric_yaml_scalars(tmp_path, good,
                                                         expected):
    """An integer, a float and a quoted number all remain valid tolerances."""
    raw = _base_config_dict()
    raw["fidelity"] = {"tol_AE": good, "tol_atom": good,
                       "override_reason": "control arm"}
    cfg = load_grid_config(_write(tmp_path, "grid.json", raw))
    assert cfg.fidelity.tol_AE == expected
    assert cfg.fidelity.tol_atom == expected


@pytest.mark.parametrize("bad", [None, 0, 1, "false", "true", [], {}])
def test_fidelity_enforce_must_be_a_boolean(tmp_path, bad):
    """A non-boolean enforce is refused rather than coerced.

    ``bool(None)`` is False, so an empty ``enforce:`` (a null in YAML) would
    otherwise DISABLE the on-node certificate gates in a config that never
    asked for it; ``bool("false")`` is True, which silently contradicts the
    author the other way. Only a YAML boolean sets this field."""
    raw = _base_config_dict()
    raw["fidelity"] = {"tol_AE": 1.0, "tol_atom": 1.0, "enforce": bad,
                       "override_reason": "control arm"}
    with pytest.raises(ValueError, match="fidelity.enforce"):
        load_grid_config(_write(tmp_path, "grid.json", raw))


def test_fidelity_enforce_null_does_not_disable_the_gates(tmp_path):
    """The path that made this reachable in a live config: a documented
    override_reason satisfies the enforce-needs-a-reason rule, so an empty
    ``enforce:`` next to it would have disabled the gates unremarked."""
    yaml = pytest.importorskip("yaml")
    raw = _base_config_dict()
    path = tmp_path / "grid.yaml"
    path.write_text(
        yaml.safe_dump(raw)
        + "fidelity:\n  tol_AE: 1.0\n  tol_atom: 1.0\n"
          "  override_reason: rung-3.5 control arm\n  enforce:\n"
    )
    with pytest.raises(ValueError, match="fidelity.enforce"):
        load_grid_config(str(path))


@pytest.mark.parametrize("good", [True, False])
def test_fidelity_enforce_accepts_booleans(tmp_path, good):
    raw = _base_config_dict()
    raw["fidelity"] = {"tol_AE": 1.0, "tol_atom": 1.0, "enforce": good,
                       "override_reason": "control arm"}
    cfg = load_grid_config(_write(tmp_path, "grid.json", raw))
    assert cfg.fidelity.enforce is good


@pytest.mark.parametrize("key", ["tol_AE", "tol_atom"])
@pytest.mark.parametrize("token", [".nan", ".NaN", "'nan'", ".inf", "-.inf",
                                   "'-inf'", "1e309"])
def test_fidelity_tolerances_must_be_finite(tmp_path, key, token):
    """A non-finite tolerance is refused at load.

    NaN escapes the bounds in ``validate_grid_semantics`` entirely: both
    ``nan <= 0`` and ``nan > 2.0`` are False, so a NaN tolerance loads with no
    override_reason and no complaint, and every downstream ``<= tol``
    comparison against it is False -- a certificate that can never fail. The
    infinities are caught downstream (``-.inf`` by the positivity floor,
    ``.inf`` by the 2.0 ceiling) but are refused here for the same reason: a
    tolerance is a finite energy bound.
    """
    yaml = pytest.importorskip("yaml")
    raw = _base_config_dict()
    other = "tol_atom" if key == "tol_AE" else "tol_AE"
    path = tmp_path / "grid.yaml"
    path.write_text(
        yaml.safe_dump(raw)
        + f"fidelity:\n  {key}: {token}\n  {other}: 1.0\n"
    )
    with pytest.raises(ValueError, match=f"fidelity.{key}"):
        load_grid_config(str(path))


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


def _rendered_time_lines(cfg, run_dir, key):
    """The ``#SBATCH --time=`` lines of the stage script ``key`` feeds."""
    from xcquinox.alec.cluster.submit import render_sbatch
    kind, array_max = _WALLTIME_RENDER[key]
    text = render_sbatch(kind, cfg, run_dir, array_max=array_max)
    return [ln for ln in text.splitlines() if ln.startswith("#SBATCH --time=")]


@pytest.mark.parametrize("literal", ["8:00:00", "12:00:00"])
@pytest.mark.parametrize("key", sorted(_WALLTIME_RENDER))
def test_unquoted_sexagesimal_walltime_restored_and_rendered(
        tmp_path, key, literal):
    """An unquoted ``H:MM:SS`` reaches SLURM as the wall that was written.

    YAML 1.1's implicit int resolver reads an unquoted sexagesimal literal in
    base 60, so ``8:00:00`` loads as 28800 and ``12:00:00`` as 43200. SLURM
    reads a bare integer as MINUTES, which turns an 8-hour request into 20
    days without any diagnostic. The literal is restored on load, so the
    rendered directive is the author's wall.
    """
    yaml = pytest.importorskip("yaml")
    path = _write_walltime_yaml(tmp_path, key, literal)
    with open(path) as fh:
        assert isinstance(yaml.safe_load(fh)["cluster"][key], int), (
            f"{literal} no longer resolves to an integer; the resolver this "
            "test pins has changed"
        )
    cfg = load_grid_config(path)
    assert getattr(cfg.cluster, key) == literal
    assert _rendered_time_lines(cfg, str(tmp_path / "run"), key) == [
        f"#SBATCH --time={literal}"]


@pytest.mark.parametrize("literal,expected", [
    ('"8:00:00"', "8:00:00"),
    ('"00:30:00"', "00:30:00"),
    ('"1-12:00:00"', "1-12:00:00"),
    ("1-12:00:00", "1-12:00:00"),
    ('"48:00:00"', "48:00:00"),
])
@pytest.mark.parametrize("key", _WALLTIME_KEYS)
def test_walltime_accepted_shapes(tmp_path, key, literal, expected):
    """``H:MM:SS`` and ``D-HH:MM:SS`` are the accepted walltime shapes."""
    cfg = load_grid_config(_write_walltime_yaml(tmp_path, key, literal))
    assert getattr(cfg.cluster, key) == expected


@pytest.mark.parametrize("literal", [
    "30",            # bare integer: SLURM minutes, the field is HH:MM:SS
    '"30"',          # same, quoted -- a string, still a bare-minutes request
    "30:00",         # minutes:seconds, resolves to the integer 1800
    '"30:00"',
    "480:00",        # 28800 as minutes:seconds; the same integer as 8:00:00
    '"1-12"',        # days-hours
    '"1-12:00"',     # days-hours:minutes
    '"8h"',
    '"soon"',
    '"8:60:00"',     # 60 minutes is out of range
    '"-8:00:00"',
    "-8:00:00",      # resolves to -28800
    "8:00:00.5",     # sexagesimal FLOAT, 28800.5
    "1.5",
    "true",
])
@pytest.mark.parametrize("key", _WALLTIME_KEYS)
def test_walltime_bad_shapes_refused(tmp_path, key, literal):
    """Anything outside the two accepted shapes is refused, naming the key."""
    with pytest.raises(ValueError, match=re.escape(f"cluster.{key}")):
        load_grid_config(_write_walltime_yaml(tmp_path, key, literal))


@pytest.mark.parametrize("value", [30, 28800, 1800])
@pytest.mark.parametrize("key", _WALLTIME_KEYS)
def test_walltime_integer_in_json_refused(tmp_path, key, value):
    """JSON has no sexagesimal resolver, so an integer there was written as an
    integer and is a bare-minutes request, not a mangled clock string."""
    raw = _base_config_dict()
    raw["cluster"][key] = value
    with pytest.raises(ValueError, match=re.escape(f"cluster.{key}")):
        load_grid_config(_write(tmp_path, "grid.json", raw))


def test_walltime_unset_fields_keep_their_fallback_sentinels(tmp_path):
    """An absent per-stage wall stays unset (falls back to ``cluster.time``);
    an explicit YAML null does the same rather than being refused."""
    yaml = pytest.importorskip("yaml")
    raw = _base_config_dict()
    cluster = raw.pop("cluster")
    p = tmp_path / "grid.yaml"
    p.write_text(yaml.safe_dump(raw) + yaml.safe_dump({"cluster": cluster})
                 + "  pretrain_time:\n  eval_time: ''\n")
    cfg = load_grid_config(str(p))
    assert cfg.cluster.pretrain_time is None
    assert cfg.cluster.eval_time == ""
    assert cfg.cluster.preflight_time == ""
    assert cfg.cluster.datagen_time is None


@pytest.mark.parametrize("literal", ['""', ""])
def test_empty_base_walltime_refused(tmp_path, literal):
    """``cluster.time`` is the wall every per-stage field falls back TO, so an
    empty one has nothing behind it and renders a bare ``#SBATCH --time=``.
    The per-stage fields keep the opposite reading (see the sentinel test):
    ``_config_to_raw_dict`` writes ``eval_time: ''`` into every
    ``resolved_config.yaml`` the later stages re-read."""
    with pytest.raises(ValueError, match=re.escape("cluster.time")):
        load_grid_config(_write_walltime_yaml(tmp_path, "time", literal))


def test_walltime_refusal_message_carries_the_offending_value(tmp_path):
    """The message names both the key and what was read, so a mangled
    sexagesimal is recognisable from the integer it became."""
    with pytest.raises(ValueError) as exc:
        load_grid_config(_write_walltime_yaml(tmp_path, "time", "30"))
    assert "cluster.time" in str(exc.value)
    assert "30" in str(exc.value)


def test_walltime_refusal_names_the_restored_literal(tmp_path):
    """A ``minutes:seconds`` literal is refused by shape, and the message
    carries the literal as well as the base-60 integer it loaded as -- the two
    are far apart (``480:00`` -> 28800) and only the literal is searchable in
    the config."""
    with pytest.raises(ValueError) as exc:
        load_grid_config(_write_walltime_yaml(tmp_path, "time", "480:00"))
    assert "480:00" in str(exc.value)
    assert "28800" in str(exc.value)


@pytest.mark.parametrize("suffix", ["", "   # bumped for the r=26 full-pool cell"])
def test_walltime_literal_recovered_past_a_trailing_comment(tmp_path, suffix):
    """Several campaign configs annotate the wall on the same line, so the
    literal scan has to stop at the YAML comment rather than miss the line."""
    cfg = load_grid_config(
        _write_walltime_yaml(tmp_path, "time", "16:00:00" + suffix))
    assert cfg.cluster.time == "16:00:00"


def test_walltime_literal_not_taken_from_another_section(tmp_path):
    """The scan is confined to the ``cluster:`` block: an identically named key
    elsewhere in the document must not supply the restoration."""
    yaml = pytest.importorskip("yaml")
    raw = _base_config_dict()
    cluster = raw.pop("cluster")
    cluster.pop("time")
    p = tmp_path / "grid.yaml"
    # An earlier section carrying a clock-shaped `time:` whose base-60 value
    # (28800) differs from the cluster block's (43200).
    p.write_text("decoy:\n  time: 8:00:00\n"
                 + yaml.safe_dump(raw) + yaml.safe_dump({"cluster": cluster})
                 + "  time: 12:00:00\n")
    cfg = load_grid_config(str(p))
    assert cfg.cluster.time == "12:00:00"


#: The campaign configs under version control. ``hpcjobs/.gitignore`` excludes
#: ``configs/*.local.yaml`` (personal cluster-filled copies), so a fresh clone,
#: a git worktree and the cluster checkout carry only these; counting whatever
#: ``*.yaml`` happens to be on disk would make this file red wherever the
#: untracked copies are absent.
_TRACKED_CONFIGS = (
    "bh76w411_repr.svp_grid2.yaml",
    "bh76w411_repr.tzvpd_grid2_df.yaml",
    "dfs_step7.dfs6311_grid3_v3.yaml",
    "dfs_step7.dfs6311_grid3_v4.yaml",
    "dfs_step7.dfs6311_grid3_v4gga.yaml",
    "dfs_step7.dfs6311_grid3_v4mgga2.yaml",
    "dfs_step7.dfs6311_grid3_v5.yaml",
    "dfs_step7.dfs6311_grid3_v5mgga2.yaml",
    "dfs_step7.svp_grid2.yaml",
    "dfs_step7.svp_grid2_v2.yaml",
    "dfs_step7.svp_grid2_v3.yaml",
    "dfs_step7.svp_grid2_v3_full25.yaml",
    "dfs_step7.svp_grid2_v3_rung35ab.yaml",
    "dfs_step7.tzvpd_grid2_df.yaml",
    "step7.yaml",
)


def _config_tree():
    """(config dir, shipped example) of this checkout, or None when absent."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[3]
    cfg_dir = root / "hpcjobs" / "configs"
    example = root / "xcquinox" / "alec" / "cluster" / "examples" / \
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
    assert len(_TRACKED_CONFIGS) + 1 == 16, "tracked config count changed"


def test_untracked_local_configs_carry_valid_walltimes():
    """The gitignored ``*.local.yaml`` copies are validated when present and
    skipped when they are not, so this checkout's extras are covered without
    the tracked set's coverage depending on them."""
    tree = _config_tree()
    if tree is None:
        pytest.skip("cluster config tree not present in this checkout")
    cfg_dir, _ = tree
    local = sorted(cfg_dir.glob("*.local.yaml"))
    if not local:
        pytest.skip("no *.local.yaml copies in this checkout")
    for path in local:
        _assert_walltimes_are_strings(path)


# ---------------------------------------------------------------------------
# Literal recovery: the cluster block has to be located, and only its own
# top-level keys may supply a literal
# ---------------------------------------------------------------------------

def _write_cluster_header_yaml(tmp_path, header, key_lines, decoy=True):
    """Write a config whose ``cluster:`` header is spelled ``header``.

    ``key_lines`` is appended inside the block. A decoy section carrying a
    clock-shaped ``time:`` is placed FIRST, so any scan that is not confined to
    the cluster block finds it before the authored value.
    """
    yaml = pytest.importorskip("yaml")
    raw = _base_config_dict()
    cluster = raw.pop("cluster")
    cluster.pop("time", None)
    block = yaml.safe_dump({"cluster": cluster})
    body = block[len("cluster:\n"):]
    text = ("decoy:\n  time: 8:00:00\n" if decoy else "")
    text += yaml.safe_dump(raw) + header + "\n" + body + key_lines
    p = tmp_path / "grid.yaml"
    p.write_text(text)
    return str(p)


@pytest.mark.parametrize("header", [
    "cluster:",
    "cluster: &cl",
    "cluster:  # per-stage SLURM resources",
    "cluster: &cl  # per-stage SLURM resources",
    '"cluster":',
    "'cluster':",
])
def test_cluster_block_located_for_every_header_spelling(tmp_path, header):
    """The literal comes from the cluster block whichever way its key is
    written. An anchored or quoted header that is not recognised sends the scan
    over the whole document, where the decoy's ``8:00:00`` (28800 s) would be
    accepted for an authored ``28800`` -- 28800 MINUTES, a 60-fold error in the
    direction this check exists to prevent."""
    path = _write_cluster_header_yaml(tmp_path, header, "  time: 12:00:00\n")
    assert load_grid_config(path).cluster.time == "12:00:00"


def test_unlocatable_cluster_block_refuses_rather_than_scanning_the_document(
        tmp_path):
    """With no recognisable header there is no block to recover a literal from,
    so the number is refused; falling back to the whole document would let an
    unrelated section supply it."""
    path = _write_cluster_header_yaml(tmp_path, "cluster: !!map", "  time: 28800\n")
    with pytest.raises(ValueError, match=re.escape("cluster.time")):
        load_grid_config(path)


def test_nested_mapping_inside_the_cluster_block_supplies_no_literal(tmp_path):
    """Only the block's own top-level keys are read. A nested ``time:`` one
    level deeper carries 28800 as well, so without the indent rule an authored
    ``480:00`` (minutes:seconds, refusable) is accepted as ``8:00:00``."""
    path = _write_cluster_header_yaml(
        tmp_path, "cluster:", "  notes:\n    time: 8:00:00\n  time: 480:00\n")
    with pytest.raises(ValueError, match=re.escape("cluster.time")):
        load_grid_config(path)


def test_nested_mapping_does_not_shadow_a_valid_top_level_wall(tmp_path):
    """The reverse ordering: a nested decoy must not make a correctly written
    top-level wall unrecoverable."""
    path = _write_cluster_header_yaml(
        tmp_path, "cluster:", "  notes:\n    time: 480:00\n  time: 8:00:00\n")
    assert load_grid_config(path).cluster.time == "8:00:00"


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

@pytest.mark.parametrize("literal", ['"0:00:00"', '"00:00:00"',
                                     '"0-00:00:00"', '"000:00:00"'])
@pytest.mark.parametrize("key", _WALLTIME_KEYS)
def test_zero_walltime_refused(tmp_path, key, literal):
    """``#SBATCH --time=0`` is NO LIMIT to SLURM, so a zero-duration wall is
    the opposite of the bound it looks like."""
    with pytest.raises(ValueError, match=re.escape(f"cluster.{key}")):
        load_grid_config(_write_walltime_yaml(tmp_path, key, literal))


@pytest.mark.parametrize("literal,expected", [
    ('"1-99:00:00"', None),
    ('"1-24:00:00"', None),
    ('"0-24:00:00"', None),
    ('"1-23:59:59"', "1-23:59:59"),
    ('"0-00:00:01"', "0-00:00:01"),
])
@pytest.mark.parametrize("key", _WALLTIME_KEYS)
def test_days_hours_field_is_a_time_of_day(tmp_path, key, literal, expected):
    """In ``D-HH:MM:SS`` the hours field is 0-23; hours beyond that belong in
    the days field, and SLURM's own normalisation of an out-of-range one is not
    something to rely on."""
    path = _write_walltime_yaml(tmp_path, key, literal)
    if expected is None:
        with pytest.raises(ValueError, match=re.escape(f"cluster.{key}")):
            load_grid_config(path)
    else:
        assert getattr(load_grid_config(path).cluster, key) == expected


# ---------------------------------------------------------------------------
# PretrainConfig: pretraining-protocol fields
# ---------------------------------------------------------------------------

def test_pretrain_config_protocol_defaults_are_todays_behavior():
    from xcquinox.alec.cluster.grid_config import PretrainConfig
    pt = PretrainConfig(data_dir="/d")
    assert pt.dfs_set is False
    assert pt.pool_atoms is False
    assert pt.parent_density == "pbe"
    assert pt.exchange_footing == "total"
    assert pt.mesh_fraction == 0.3
    assert pt.energy_term_weight == 0.0
    assert pt.validation_fraction == 0.0
    assert pt.validation_seed == 0
    assert pt.validate_every == 50
    assert pt.patience == 0


def test_pretrain_config_mesh_fraction_default_matches_the_generator():
    """grid_config imports neither JAX nor PySCF, so the default is written as
    a literal; this pins it against the constant it must equal."""
    from xcquinox.alec.cluster.grid_config import PretrainConfig
    from xcquinox.alec.pretrain_data_gen import MESH_WEIGHT_FRACTION
    assert PretrainConfig(data_dir="/d").mesh_fraction == MESH_WEIGHT_FRACTION


def test_build_pretrain_parses_every_protocol_field():
    """A field missing from _build_pretrain silently reverts to its default on
    every stage that re-reads resolved_config.yaml."""
    from xcquinox.alec.cluster.grid_config import _build_pretrain
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
    from xcquinox.alec.cluster.__main__ import _config_to_raw_dict
    from xcquinox.alec.cluster.grid_config import _build_pretrain
    protocol = {
        "dfs_set": True, "pool_atoms": True,
        "parent_density": "auto", "exchange_footing": "spin_channel",
        "mesh_fraction": 0.25, "energy_term_weight": 1.0,
        "validation_fraction": 0.2, "validation_seed": 11,
        "validate_every": 25, "patience": 8,
    }
    pt = _build_pretrain(dict(protocol, data_dir="/d"))
    raw = dataclasses.asdict(pt)
    assert _build_pretrain(raw) == pt
    for f in dataclasses.fields(pt):
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
    every = _build_pretrain(dict(
        raw, n_steps=7, lr_start=3e-2, lr_end=3e-6, lr_decay_start=0.4,
        grad_clip=2.5, seed=1234, loss_weighting="unweighted",
        atoms=[["Li", 1], ["C", 2]]))
    default = PretrainConfig(data_dir="/d")
    for f in dataclasses.fields(every):
        if f.name == "data_dir":
            continue
        assert getattr(every, f.name) != getattr(default, f.name), (
            f"PretrainConfig.{f.name} is at its default in this fixture, so "
            "the round trip is NOT guarded for it")
    cfg = dataclasses.replace(_cfg(), pretrain=every)
    serialized = _config_to_raw_dict(cfg)["pretrain"]
    assert serialized == dataclasses.asdict(every)
    assert _build_pretrain(serialized) == every


def test_validate_grid_semantics_bounds_the_protocol_fields():
    import dataclasses
    from xcquinox.alec.cluster.grid_config import validate_grid_semantics
    cfg = _cfg()                       # the module's GridConfig builder
    domain = _StubDomain(pool_size=40)  # as in test_validate_ok
    for field, value, message in (
            ("parent_density", "blyp", "parent_density"),
            ("exchange_footing", "per_orbital", "exchange_footing"),
            ("mesh_fraction", 1.0, "mesh_fraction"),
            ("energy_term_weight", -1.0, "energy_term_weight"),
            ("validation_fraction", 1.0, "validation_fraction"),
            ("validate_every", 0, "validate_every"),
            ("patience", -1, "patience"),
    ):
        bad = dataclasses.replace(
            cfg, pretrain=dataclasses.replace(cfg.pretrain, **{field: value}))
        with pytest.raises(ValueError, match=message):
            validate_grid_semantics(bad, domain)


@pytest.mark.parametrize("field", ["mesh_fraction", "energy_term_weight",
                                   "validation_fraction"])
@pytest.mark.parametrize("literal", ["nan", "inf", "-inf"])
def test_validate_grid_semantics_refuses_a_non_finite_protocol_weight(
        field, literal):
    """NaN escapes an ordinary bound in whichever direction the comparison is
    written -- ``nan < 0`` and ``nan >= 1.0`` are both False -- so a NaN weight
    would load with no complaint and every downstream comparison against it
    would be False as well. The certificate tolerances are refused on the same
    grounds."""
    import dataclasses
    from xcquinox.alec.cluster.grid_config import validate_grid_semantics
    bad = dataclasses.replace(
        _cfg(), pretrain=dataclasses.replace(
            _cfg().pretrain, **{field: float(literal)}))
    with pytest.raises(ValueError, match=field):
        validate_grid_semantics(bad, _StubDomain(pool_size=40))


# ---------------------------------------------------------------------------
# _build_pretrain: coercion hardening (the fidelity block's house pattern)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", ["dfs_set", "pool_atoms"])
@pytest.mark.parametrize("value", ["false", "no", "0", "", None, 0, 1])
def test_build_pretrain_refuses_a_non_boolean_switch(key, value):
    """``bool("false")`` is True, so a hand-quoted switch would turn the DFS
    set (or the pool atoms) ON in a config that wrote it OFF, and ``bool(None)``
    -- an empty ``dfs_set:`` -- would read as OFF without remark. The fidelity
    block refuses ``enforce`` on the same grounds rather than coercing it."""
    from xcquinox.alec.cluster.grid_config import _build_pretrain
    with pytest.raises(ValueError) as exc:
        _build_pretrain({"data_dir": "/d", key: value})
    assert f"pretrain.{key}" in str(exc.value)
    assert repr(value) in str(exc.value)


@pytest.mark.parametrize("key", ["mesh_fraction", "energy_term_weight",
                                 "validation_fraction", "validation_seed",
                                 "validate_every", "patience"])
@pytest.mark.parametrize("value", [True, False, None, "abc", "", [0.1],
                                   {"a": 1}])
def test_build_pretrain_refuses_a_non_numeric_protocol_value(key, value):
    """``float(True)`` is 1.0 and ``int(True)`` is 1 -- silently a weight of
    one, or one validation every step -- while ``float(None)`` raises
    TypeError, which passes every ``except ValueError`` handler in the load
    path and surfaces as a crash naming no key."""
    from xcquinox.alec.cluster.grid_config import _build_pretrain
    with pytest.raises(ValueError) as exc:
        _build_pretrain({"data_dir": "/d", key: value})
    assert f"pretrain.{key}" in str(exc.value)
    assert repr(value) in str(exc.value)


@pytest.mark.parametrize("key", ["validation_seed", "validate_every",
                                 "patience"])
def test_build_pretrain_refuses_a_fractional_step_count(key):
    """``int(2.5)`` truncates to 2: a schedule silently different from the one
    written. A count is a whole number or a config error."""
    from xcquinox.alec.cluster.grid_config import _build_pretrain
    with pytest.raises(ValueError) as exc:
        _build_pretrain({"data_dir": "/d", key: 2.5})
    assert f"pretrain.{key}" in str(exc.value)
    assert "2.5" in str(exc.value)


@pytest.mark.parametrize("key", ["mesh_fraction", "energy_term_weight",
                                 "validation_fraction"])
@pytest.mark.parametrize("literal", ["nan", "inf", "-inf"])
def test_build_pretrain_refuses_a_non_finite_protocol_number(key, literal):
    """Refused at the parse, as ``_fidelity_tolerance`` refuses a non-finite
    tolerance: a NaN weight escapes the bounds in validate_grid_semantics in
    whichever direction they are written, and turns every comparison against
    it into the sense of that comparison rather than a measurement."""
    from xcquinox.alec.cluster.grid_config import _build_pretrain
    with pytest.raises(ValueError) as exc:
        _build_pretrain({"data_dir": "/d", key: float(literal)})
    assert f"pretrain.{key}" in str(exc.value)


@pytest.mark.parametrize("key,value", [
    ("parent_density", "blyp"), ("parent_density", "PBE"),
    ("parent_density", None), ("parent_density", True),
    ("exchange_footing", "per_orbital"), ("exchange_footing", "Total"),
    ("exchange_footing", None), ("exchange_footing", ["total"]),
])
def test_build_pretrain_refuses_an_unknown_string_knob(key, value):
    """``str(None)`` is the non-empty string 'None' and ``str(True)`` is
    'True', so coercion carries a typo or an empty key past the parse; the
    member test lives here so ``load_grid_config`` refuses it whether or not
    ``validate_grid_semantics`` is reached."""
    from xcquinox.alec.cluster.grid_config import _build_pretrain
    with pytest.raises(ValueError) as exc:
        _build_pretrain({"data_dir": "/d", key: value})
    assert f"pretrain.{key}" in str(exc.value)
    assert repr(value) in str(exc.value)


def test_build_pretrain_still_accepts_numeric_strings_and_real_booleans():
    """The hardening refuses types, not the YAML idioms that already worked:
    an unquoted true/false and a quoted number both remain valid, as they do
    for the certificate tolerances."""
    from xcquinox.alec.cluster.grid_config import _build_pretrain
    pt = _build_pretrain({
        "data_dir": "/d", "dfs_set": True, "pool_atoms": False,
        "mesh_fraction": "0.25", "energy_term_weight": "1",
        "validation_fraction": "0.2", "validation_seed": "11",
        "validate_every": "25", "patience": "8",
    })
    assert pt.dfs_set is True and pt.pool_atoms is False
    assert pt.mesh_fraction == 0.25 and pt.energy_term_weight == 1.0
    assert pt.validation_fraction == 0.2
    assert (pt.validation_seed, pt.validate_every, pt.patience) == (11, 25, 8)
    assert isinstance(pt.validation_seed, int)
    assert isinstance(pt.validate_every, int)
    assert isinstance(pt.patience, int)


# ---------------------------------------------------------------------------
# _build_pretrain: the pre-protocol keys, same typed parsing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("key", ["n_steps", "lr_start", "lr_end",
                                 "lr_decay_start", "grad_clip", "seed"])
@pytest.mark.parametrize("value", [True, None, "abc", "", [1], {"a": 1}])
def test_build_pretrain_refuses_a_non_numeric_pre_protocol_value(key, value):
    """These keys were passed through with no coercion at all, so a mistyped
    value reached optax and jax.random -- ``clip_by_global_norm(None)`` raises
    a TypeError inside the update, thousands of steps into a cluster job,
    naming no key."""
    from xcquinox.alec.cluster.grid_config import _build_pretrain
    with pytest.raises(ValueError) as exc:
        _build_pretrain({"data_dir": "/d", key: value})
    assert f"pretrain.{key}" in str(exc.value)
    assert repr(value) in str(exc.value)


@pytest.mark.parametrize("key", ["n_steps", "seed"])
def test_build_pretrain_refuses_a_fractional_pre_protocol_count(key):
    from xcquinox.alec.cluster.grid_config import _build_pretrain
    with pytest.raises(ValueError) as exc:
        _build_pretrain({"data_dir": "/d", key: 2.5})
    assert f"pretrain.{key}" in str(exc.value)
    assert "2.5" in str(exc.value)


@pytest.mark.parametrize("key,value", [
    ("n_steps", 0), ("n_steps", -1),
    ("lr_start", 0.0), ("lr_start", -1e-3),
    ("lr_end", -1e-6),
    ("grad_clip", 0.0), ("grad_clip", -1.0),
])
def test_build_pretrain_refuses_a_non_positive_rate_or_clip(key, value):
    """Measured on the consumers: ``clip_by_global_norm(0.0)`` zeroes every
    gradient and ``clip_by_global_norm(-1.0)`` reverses its direction (the
    same |g| = 5 gradient comes back as [0.6, 0.8, 0] clipped at 1.0 and
    [-0.6, -0.8, -0] at -1.0), and a non-positive Adam rate is a no-op or an
    ascent. None of the three is a run."""
    from xcquinox.alec.cluster.grid_config import _build_pretrain
    with pytest.raises(ValueError) as exc:
        _build_pretrain({"data_dir": "/d", key: value})
    assert f"pretrain.{key}" in str(exc.value)


def test_build_pretrain_accepts_an_anneal_to_zero():
    """``lr_end: 0`` is a linear anneal to zero, a legitimate schedule, so the
    floor is >= 0 while ``lr_start`` must be strictly positive."""
    from xcquinox.alec.cluster.grid_config import _build_pretrain
    pt = _build_pretrain({"data_dir": "/d", "lr_end": 0.0})
    assert pt.lr_end == 0.0


@pytest.mark.parametrize("value", [-1, 2**32 - 1, 2**32, 2**40])
def test_build_pretrain_refuses_a_seed_outside_the_key_range(value):
    """Measured: ``jax.random.PRNGKey`` wraps modulo 2**32 rather than
    raising -- PRNGKey(-1) and PRNGKey(2**32 - 1) are the same key, and
    PRNGKey(2**32) is PRNGKey(0) -- so an out-of-range seed silently ALIASES
    another run's initialization while the metadata records the number
    written. ``create_network_pair`` keys cnet at ``seed + 1``, so the top of
    the range is excluded as well."""
    from xcquinox.alec.cluster.grid_config import _build_pretrain
    with pytest.raises(ValueError) as exc:
        _build_pretrain({"data_dir": "/d", "seed": value})
    assert "pretrain.seed" in str(exc.value)


@pytest.mark.parametrize("value", [-0.1, 1.5])
def test_build_pretrain_refuses_an_out_of_range_decay_onset(value):
    from xcquinox.alec.cluster.grid_config import _build_pretrain
    with pytest.raises(ValueError) as exc:
        _build_pretrain({"data_dir": "/d", "lr_decay_start": value})
    assert "pretrain.lr_decay_start" in str(exc.value)


def test_build_pretrain_keeps_lr_decay_start_a_fraction():
    """lr_decay_start is a FRACTION of n_steps (``decay_start_step =
    int(lr_decay_start * n_steps)``), not a step count: 0.2 -- the default and
    the value every shipped config writes -- must parse as 0.2."""
    from xcquinox.alec.cluster.grid_config import _build_pretrain
    assert _build_pretrain({"data_dir": "/d"}).lr_decay_start == 0.2
    assert _build_pretrain(
        {"data_dir": "/d", "lr_decay_start": 0.35}).lr_decay_start == 0.35


@pytest.mark.parametrize("value", ["Integration", "l2", "", None, True, 1])
def test_build_pretrain_refuses_an_unknown_loss_weighting(value):
    from xcquinox.alec.cluster.grid_config import _build_pretrain
    with pytest.raises(ValueError) as exc:
        _build_pretrain({"data_dir": "/d", "loss_weighting": value})
    assert "pretrain.loss_weighting" in str(exc.value)
    assert repr(value) in str(exc.value)


@pytest.mark.parametrize("value", ["unweighted", "integration"])
def test_build_pretrain_accepts_both_loss_weightings(value):
    from xcquinox.alec.cluster.grid_config import _build_pretrain
    assert _build_pretrain(
        {"data_dir": "/d", "loss_weighting": value}).loss_weighting == value


def test_build_pretrain_still_accepts_the_shipped_pre_protocol_values():
    """The values every shipped config writes, and the quoted forms, stay
    valid: the hardening refuses types and ranges, not the YAML in the tree."""
    from xcquinox.alec.cluster.grid_config import _build_pretrain
    pt = _build_pretrain({
        "data_dir": "/d", "n_steps": 2500, "lr_start": 0.01, "lr_end": 1e-05,
        "lr_decay_start": 0.2, "grad_clip": 1.0, "seed": 42,
        "loss_weighting": "integration",
    })
    assert pt.n_steps == 2500 and isinstance(pt.n_steps, int)
    assert (pt.lr_start, pt.lr_end) == (0.01, 1e-05)
    assert pt.lr_decay_start == 0.2 and pt.grad_clip == 1.0
    assert pt.seed == 42 and isinstance(pt.seed, int)
    quoted = _build_pretrain({
        "data_dir": "/d", "n_steps": "2500", "lr_start": "1e-2",
        "lr_end": "1e-05", "lr_decay_start": "0.2", "grad_clip": "1.0",
        "seed": "42",
    })
    assert quoted.n_steps == 2500 and quoted.seed == 42
    assert quoted.lr_start == 0.01 and quoted.grad_clip == 1.0
