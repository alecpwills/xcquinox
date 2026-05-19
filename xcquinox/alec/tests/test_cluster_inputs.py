"""Tests for xcquinox.alec.cluster.inputs — input-artifact staging.

The fast reuse-mode tests stub ``build_dfs_pool_points`` / ``pool_fingerprint``
(via the names ``inputs.py`` imported) so they exercise the ledger-validation
logic without building the real DFS pool or doing any SCF / CCSD work. The
slow regen-mode test monkeypatches every heavy seam so the orchestration runs
end-to-end without real compute.
"""
import json

import pytest
from ase import Atoms

from xcquinox.alec.cluster import inputs as inputs_mod
from xcquinox.alec.cluster.inputs import prepare_inputs, StagedInputs
from xcquinox.alec.cluster.grid_config import (
    GridConfig,
    SweepAxes,
    SolverNamed,
    HyperParams,
    InputPaths,
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
    """A 4-point synthetic pool — enough for subset sizes 2 and 3."""
    return [_ae_point(n) for n in ("P0", "P1", "P2", "P3")]


_FAKE_FP = "deadbeef" * 8  # fixed stub fingerprint


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
        descriptor_cache=str(tmp_path / "desc"),
        refhist_cache=str(tmp_path / "refhist"),
        subset_ledger_path=str(tmp_path / "ledger.json"),
        basis=basis,
        grid_level=grid_level,
        output_root=str(tmp_path / "out"),
        pretrain_checkpoint=None,
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
        cluster=cluster,
        domain_profile="dfs_step7",
        bh76_mode=bh76_mode,
    )


def _make_ledger(basis="def2-svp", grid_level=1, bh76_mode="reaction_energy",
                 fingerprint=_FAKE_FP, entries=None):
    """A consistent stub ledger matching ``_make_cfg``'s 2-cell grid."""
    if entries is None:
        entries = {
            "l2:2": {"metric": "l2", "subset_size": 2,
                     "point_names": ["P0", "P1"]},
            "l2:3": {"metric": "l2", "subset_size": 3,
                     "point_names": ["P0", "P1", "P2"]},
        }
    return {
        "pool_fingerprint": fingerprint,
        "basis": basis,
        "grid_level": grid_level,
        "bh76_mode": bh76_mode,
        "pool_was_shrunk": False,
        "dropped_species": [],
        "entries": entries,
    }


@pytest.fixture
def stub_pool(monkeypatch):
    """Stub ``build_dfs_pool_points`` / ``pool_fingerprint`` inside inputs.py
    so reuse-mode tests need neither the real pool nor the traj files."""
    pool = _make_pool()
    monkeypatch.setattr(inputs_mod, "build_dfs_pool_points",
                        lambda bh76_mode="reaction_energy": pool)
    monkeypatch.setattr(inputs_mod, "pool_fingerprint", lambda pts: _FAKE_FP)
    return pool


def _write_ledger(path, ledger):
    with open(path, "w") as f:
        json.dump(ledger, f)


# ---------------------------------------------------------------------------
# Reuse mode — happy path
# ---------------------------------------------------------------------------

def test_reuse_consistent_ledger_succeeds(tmp_path, stub_pool):
    cfg = _make_cfg(tmp_path)
    _write_ledger(cfg.inputs.subset_ledger_path, _make_ledger())
    staged = prepare_inputs(cfg, regenerate=False)
    assert isinstance(staged, StagedInputs)
    assert staged.points is stub_pool
    assert staged.subset_ledger["pool_fingerprint"] == _FAKE_FP
    assert set(staged.subset_ledger["entries"]) == {"l2:2", "l2:3"}


# ---------------------------------------------------------------------------
# Reuse mode — fail-fast cases
# ---------------------------------------------------------------------------

def test_reuse_missing_ledger_fails(tmp_path, stub_pool):
    cfg = _make_cfg(tmp_path)  # ledger never written
    with pytest.raises(ValueError, match="not found"):
        prepare_inputs(cfg, regenerate=False)


def test_reuse_unparseable_ledger_fails(tmp_path, stub_pool):
    cfg = _make_cfg(tmp_path)
    with open(cfg.inputs.subset_ledger_path, "w") as f:
        f.write("{ this is not valid json")
    with pytest.raises(ValueError, match="unparseable"):
        prepare_inputs(cfg, regenerate=False)


def test_reuse_basis_mismatch_fails(tmp_path, stub_pool):
    cfg = _make_cfg(tmp_path, basis="def2-svp")
    _write_ledger(cfg.inputs.subset_ledger_path,
                  _make_ledger(basis="def2-tzvp"))
    with pytest.raises(ValueError, match="basis mismatch"):
        prepare_inputs(cfg, regenerate=False)


def test_reuse_grid_level_mismatch_fails(tmp_path, stub_pool):
    cfg = _make_cfg(tmp_path, grid_level=1)
    _write_ledger(cfg.inputs.subset_ledger_path,
                  _make_ledger(grid_level=3))
    with pytest.raises(ValueError, match="grid_level mismatch"):
        prepare_inputs(cfg, regenerate=False)


def test_reuse_bh76_mode_mismatch_fails(tmp_path, stub_pool):
    cfg = _make_cfg(tmp_path, bh76_mode="reaction_energy")
    _write_ledger(cfg.inputs.subset_ledger_path,
                  _make_ledger(bh76_mode="barrier_height"))
    with pytest.raises(ValueError, match="bh76_mode mismatch"):
        prepare_inputs(cfg, regenerate=False)


def test_reuse_pool_fingerprint_mismatch_fails(tmp_path, stub_pool):
    cfg = _make_cfg(tmp_path)
    _write_ledger(cfg.inputs.subset_ledger_path,
                  _make_ledger(fingerprint="0" * 64))
    with pytest.raises(ValueError, match="pool_fingerprint mismatch"):
        prepare_inputs(cfg, regenerate=False)


def test_reuse_missing_metric_size_entry_fails(tmp_path, stub_pool):
    cfg = _make_cfg(tmp_path)  # grid needs l2:2 AND l2:3
    # Ledger only has l2:2 — l2:3 cell has no entry.
    ledger = _make_ledger(entries={
        "l2:2": {"metric": "l2", "subset_size": 2,
                 "point_names": ["P0", "P1"]},
    })
    _write_ledger(cfg.inputs.subset_ledger_path, ledger)
    with pytest.raises(ValueError, match="missing entries"):
        prepare_inputs(cfg, regenerate=False)


def test_reuse_no_entries_mapping_fails(tmp_path, stub_pool):
    """A ledger with no 'entries' mapping fails fast."""
    cfg = _make_cfg(tmp_path)
    ledger = _make_ledger()
    del ledger["entries"]
    _write_ledger(cfg.inputs.subset_ledger_path, ledger)
    with pytest.raises(ValueError, match="no 'entries'"):
        prepare_inputs(cfg, regenerate=False)


# ---------------------------------------------------------------------------
# Regenerate mode (slow — heavy seams monkeypatched, deselected by default)
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_regenerate_writes_ledger_with_monkeypatched_seams(
    tmp_path, monkeypatch
):
    cfg = _make_cfg(tmp_path)
    pool = _make_pool()

    calls = {"precompute": 0, "union": 0, "descriptors": 0,
             "histograms": 0, "select": []}

    def fake_build_pool(bh76_mode="reaction_energy"):
        assert bh76_mode == cfg.bh76_mode
        return pool

    def fake_build_species_union():
        calls["union"] += 1
        return ["species-union-sentinel"]

    def fake_precompute_all(species, *, cache_dir, basis, grid_level):
        calls["precompute"] += 1
        assert species == ["species-union-sentinel"]
        assert basis == cfg.inputs.basis
        assert grid_level == cfg.inputs.grid_level

    def fake_extract(species, *, basis, grid_level, cache_dir):
        calls["descriptors"] += 1
        return {"species-descriptors": True}

    def fake_concat(points, species_descriptors):
        assert points is pool
        # one descriptor dict per point
        return [{"i": i} for i in range(len(points))]

    def fake_build_histograms(pool_descriptors):
        calls["histograms"] += 1
        return ("h_ref-sentinel", "edges-sentinel")

    def fake_select_subset(pool_descriptors, edges, h_ref, *, r, metric):
        calls["select"].append((metric, r))
        assert h_ref == "h_ref-sentinel"
        assert edges == "edges-sentinel"
        # Return integer POSITIONS into the pool (first r positions).
        return tuple(range(r)), 0.123

    monkeypatch.setattr(inputs_mod, "build_dfs_pool_points", fake_build_pool)
    monkeypatch.setattr(inputs_mod, "_build_species_union",
                        fake_build_species_union)
    monkeypatch.setattr(inputs_mod, "_precompute_all", fake_precompute_all)
    monkeypatch.setattr(inputs_mod, "_extract_descriptors", fake_extract)
    monkeypatch.setattr(inputs_mod, "_concatenate_point_descriptors",
                        fake_concat)
    monkeypatch.setattr(inputs_mod, "_build_reference_histograms",
                        fake_build_histograms)
    monkeypatch.setattr(inputs_mod, "_select_subset", fake_select_subset)

    staged = prepare_inputs(cfg, regenerate=True)

    # --- pipeline ran the heavy seams the expected number of times ---------
    assert calls["precompute"] == 1
    assert calls["union"] == 1
    assert calls["descriptors"] == 1
    assert calls["histograms"] == 1
    # one select_subset per distinct (metric, subset_size) pair
    assert sorted(calls["select"]) == [("l2", 2), ("l2", 3)]

    # --- ledger written atomically in the exact schema ---------------------
    ledger_path = cfg.inputs.subset_ledger_path
    import os
    assert os.path.isfile(ledger_path)
    with open(ledger_path) as f:
        on_disk = json.load(f)

    from xcquinox.alec.cluster.spec_builder import pool_fingerprint
    assert on_disk["pool_fingerprint"] == pool_fingerprint(pool)
    assert on_disk["basis"] == cfg.inputs.basis
    assert on_disk["grid_level"] == cfg.inputs.grid_level
    assert on_disk["bh76_mode"] == cfg.bh76_mode
    assert on_disk["pool_was_shrunk"] is False
    assert on_disk["dropped_species"] == []

    entries = on_disk["entries"]
    assert set(entries) == {"l2:2", "l2:3"}
    # point_names stored — NOT positional indices.
    assert entries["l2:2"]["point_names"] == ["P0", "P1"]
    assert entries["l2:3"]["point_names"] == ["P0", "P1", "P2"]
    assert entries["l2:2"]["metric"] == "l2"
    assert entries["l2:2"]["subset_size"] == 2

    # --- returned StagedInputs matches the on-disk ledger ------------------
    assert isinstance(staged, StagedInputs)
    assert staged.points is pool
    assert staged.subset_ledger == on_disk
