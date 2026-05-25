"""Tests for xcquinox.alec.cluster.inputs — input-artifact staging.

``prepare_inputs`` is consume-only for subsets: it builds the training-point
pool, loads the EXISTING ``subset_index_log.json`` ledger, and ensures CCSD
external references are staged. These tests stub ``build_dfs_pool_points`` and
the ``_build_species_union`` / ``_precompute_all`` heavy seams (via the names
``inputs.py`` imported) so the orchestration runs without building the real
DFS pool or doing any SCF / CCSD work.
"""
import json

import pytest
from ase import Atoms

from xcquinox.alec.cluster import inputs as inputs_mod
from xcquinox.alec.cluster.inputs import (
    prepare_inputs,
    StagedInputs,
)
from xcquinox.alec.cluster.grid_config import (
    GridConfig,
    SweepAxes,
    SolverNamed,
    HyperParams,
    InputPaths,
    PretrainConfig,
    ClusterResources,
)
from xcquinox.alec.training_points import TrainingPoint


# ---------------------------------------------------------------------------
# Synthetic pool + config helpers
# ---------------------------------------------------------------------------

def _named_atoms(symbol, name, charge=0, spin=0):
    a = Atoms(symbol, positions=[(0.0, 0.0, 0.0)])
    a.info["name"] = name
    a.info["charge"] = charge
    a.info["spin"] = spin
    return a


def _ae_point(name):
    """A minimal AE TrainingPoint with one species carrying info['name']."""
    return TrainingPoint(
        kind="ae",
        name=name,
        species=(_named_atoms("H", name),),
        metadata={"ae_kcalmol": 100.0},
    )


def _make_pool():
    """A 4-point synthetic pool."""
    return [_ae_point(n) for n in ("P0", "P1", "P2", "P3")]


def _make_cfg(tmp_path, bh76_mode="reaction_energy", basis="def2-svp",
              grid_level=1):
    """A GridConfig whose grid is metric=l2 x subset_size={2,3} (2 cells)."""
    sweep = SweepAxes(
        arch=("shallow",),
        loss=("L5_gradnorm_vxc_step7",),
        metric=("l2",),
        subset_size=(2, 3),
        solver=("oneshot",),
    )
    solvers = {"oneshot": SolverNamed(mode="oneshot", max_cycles=0)}
    hp = HyperParams(
        n_steps=100,
        lr_start=1e-3,
        lr_end=1e-5,
        lr_decay_start=0.0,
        grad_clip=1.0,
        gradnorm_alpha=1.5,
        vxc_weight=0.01,
        density_weight=0.1,
    )
    inputs = InputPaths(
        external_refs_dir=str(tmp_path / "refs"),
        subset_ledger_path=str(tmp_path / "subset_index_log.json"),
        basis=basis,
        grid_level=grid_level,
        output_root=str(tmp_path / "out"),
    )
    pretrain = PretrainConfig(
        data_dir=str(tmp_path / "data"),
    )
    cluster = ClusterResources(
        partition="short",
        time="04:00:00",
        mem="16G",
        cpus_per_task=4,
        array_throttle=8,
        eval_array_throttle=4,
        max_concurrent_tasks=16,
    )
    return GridConfig(
        sweep=sweep,
        solvers=solvers,
        hyperparams=hp,
        inputs=inputs,
        pretrain=pretrain,
        cluster=cluster,
        domain_profile="dfs_step7",
        bh76_mode=bh76_mode,
    )


def _make_ledger(entries=None):
    """A stub subset_index_log.json covering ``_make_cfg``'s 2-cell grid.

    Schema matches the existing notebook ledger: ``"<metric>/<r>"`` keys with
    ``chosen_indices`` / ``metric_value`` / ``point_kinds`` / ``point_names``
    / ``tag`` fields.
    """
    if entries is None:
        entries = {
            "l2/2": {
                "chosen_indices": [0, 1],
                "metric_value": 26.2,
                "point_kinds": ["ae", "ae"],
                "point_names": ["P0", "P1"],
                "tag": "bin02",
            },
            "l2/3": {
                "chosen_indices": [0, 1, 2],
                "metric_value": 20.3,
                "point_kinds": ["ae", "ae", "ae"],
                "point_names": ["P0", "P1", "P2"],
                "tag": "bin03",
            },
        }
    return entries


def _write_ledger(path, ledger):
    with open(path, "w") as f:
        json.dump(ledger, f)


@pytest.fixture
def stub_pool(monkeypatch):
    """Stub ``build_dfs_pool_points`` inside inputs.py so tests need neither
    the real pool nor the traj files."""
    pool = _make_pool()
    monkeypatch.setattr(inputs_mod, "build_dfs_pool_points",
                        lambda bh76_mode="reaction_energy": pool)
    return pool


@pytest.fixture
def stub_refs(monkeypatch):
    """Stub the CCSD external-reference seams so no real SCF / CCSD runs.

    Returns a ``calls`` dict tests assert against."""
    calls = {"union": 0, "precompute": 0, "precompute_kwargs": None}

    def fake_build_species_union():
        calls["union"] += 1
        return ["species-union-sentinel"]

    def fake_precompute_all(species, *, cache_dir, basis, grid_level):
        calls["precompute"] += 1
        calls["precompute_kwargs"] = {
            "species": species,
            "cache_dir": cache_dir,
            "basis": basis,
            "grid_level": grid_level,
        }

    monkeypatch.setattr(inputs_mod, "_build_species_union",
                        fake_build_species_union)
    monkeypatch.setattr(inputs_mod, "_precompute_all", fake_precompute_all)
    return calls


# ---------------------------------------------------------------------------
# Happy path — ledger loaded + returned, refs ensured
# ---------------------------------------------------------------------------

def test_prepare_inputs_loads_ledger_and_returns_it(tmp_path, stub_pool,
                                                    stub_refs):
    cfg = _make_cfg(tmp_path)
    ledger = _make_ledger()
    _write_ledger(cfg.inputs.subset_ledger_path, ledger)

    staged = prepare_inputs(cfg)

    assert isinstance(staged, StagedInputs)
    assert staged.points is stub_pool
    # subset_ledger is the raw notebook-format dict, returned verbatim.
    assert staged.subset_ledger == ledger
    assert set(staged.subset_ledger) == {"l2/2", "l2/3"}
    assert staged.subset_ledger["l2/2"]["point_names"] == ["P0", "P1"]


def test_prepare_inputs_calls_precompute_all_for_external_refs(
    tmp_path, stub_pool, stub_refs
):
    cfg = _make_cfg(tmp_path)
    _write_ledger(cfg.inputs.subset_ledger_path, _make_ledger())

    prepare_inputs(cfg)

    # external refs ensured via build_species_union -> precompute_all
    assert stub_refs["union"] == 1
    assert stub_refs["precompute"] == 1
    kw = stub_refs["precompute_kwargs"]
    assert kw["species"] == ["species-union-sentinel"]
    assert kw["cache_dir"] == cfg.inputs.external_refs_dir
    assert kw["basis"] == cfg.inputs.basis
    assert kw["grid_level"] == cfg.inputs.grid_level


def test_prepare_inputs_precompute_is_skip_if_cached_noop_friendly(
    tmp_path, stub_pool, monkeypatch
):
    """precompute_all is skip-if-cached / idempotent: a no-op precompute (all
    refs already staged) lets prepare_inputs succeed cleanly."""
    cfg = _make_cfg(tmp_path)
    _write_ledger(cfg.inputs.subset_ledger_path, _make_ledger())

    monkeypatch.setattr(inputs_mod, "_build_species_union", lambda: ["u"])
    # a precompute that does nothing — every ref already cached
    monkeypatch.setattr(
        inputs_mod, "_precompute_all",
        lambda species, *, cache_dir, basis, grid_level: None,
    )

    staged = prepare_inputs(cfg)
    assert isinstance(staged, StagedInputs)
    assert staged.subset_ledger == _make_ledger()


def test_prepare_inputs_recompute_refs_false_skips_precompute(
    tmp_path, stub_pool, stub_refs
):
    cfg = _make_cfg(tmp_path)
    _write_ledger(cfg.inputs.subset_ledger_path, _make_ledger())

    staged = prepare_inputs(cfg, recompute_refs=False)

    # neither seam touched when refs are known-staged
    assert stub_refs["union"] == 0
    assert stub_refs["precompute"] == 0
    assert isinstance(staged, StagedInputs)
    assert staged.subset_ledger == _make_ledger()


# ---------------------------------------------------------------------------
# Fail-fast cases — ledger problems
# ---------------------------------------------------------------------------

def test_prepare_inputs_missing_ledger_fails(tmp_path, stub_pool, stub_refs):
    cfg = _make_cfg(tmp_path)  # ledger never written
    with pytest.raises(ValueError, match="not found"):
        prepare_inputs(cfg)


def test_prepare_inputs_unparseable_ledger_fails(tmp_path, stub_pool,
                                                 stub_refs):
    cfg = _make_cfg(tmp_path)
    with open(cfg.inputs.subset_ledger_path, "w") as f:
        f.write("{ this is not valid json")
    with pytest.raises(ValueError, match="unparseable"):
        prepare_inputs(cfg)


def test_prepare_inputs_non_object_ledger_fails(tmp_path, stub_pool,
                                                stub_refs):
    cfg = _make_cfg(tmp_path)
    with open(cfg.inputs.subset_ledger_path, "w") as f:
        json.dump(["not", "an", "object"], f)
    with pytest.raises(ValueError, match="not a JSON object"):
        prepare_inputs(cfg)


def test_prepare_inputs_missing_metric_size_cell_fails(tmp_path, stub_pool,
                                                       stub_refs):
    """The grid sweeps l2/2 AND l2/3; a ledger missing l2/3 fails fast."""
    cfg = _make_cfg(tmp_path)
    ledger = _make_ledger(entries={
        "l2/2": {
            "chosen_indices": [0, 1],
            "metric_value": 26.2,
            "point_kinds": ["ae", "ae"],
            "point_names": ["P0", "P1"],
            "tag": "bin02",
        },
    })
    _write_ledger(cfg.inputs.subset_ledger_path, ledger)
    with pytest.raises(ValueError, match="missing entries"):
        prepare_inputs(cfg)


def test_prepare_inputs_does_not_precompute_when_ledger_invalid(
    tmp_path, stub_pool, stub_refs
):
    """A missing ledger fails fast BEFORE any CCSD precompute is attempted."""
    cfg = _make_cfg(tmp_path)  # no ledger
    with pytest.raises(ValueError):
        prepare_inputs(cfg)
    assert stub_refs["precompute"] == 0
