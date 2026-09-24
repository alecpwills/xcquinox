"""Tests for xcquinox.pipeline.cluster.inputs: input-artifact staging.

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

from xcquinox.pipeline.cluster import inputs as inputs_mod
from xcquinox.pipeline.external_refs import SpeciesEntry
from xcquinox.pipeline.cluster.inputs import (
    prepare_inputs,
    StagedInputs,
)
from xcquinox.pipeline.cluster.grid_config import (
    GridConfig,
    SweepAxes,
    SolverNamed,
    HyperParams,
    InputPaths,
    PretrainConfig,
    ClusterResources,
)
from xcquinox.pipeline.training_points import TrainingPoint


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
              grid_level=1, density_fit=False, auxbasis=None,
              use_polarized_correlation=False, orientation_lock_strength=0.0):
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
        density_fit=density_fit,
        auxbasis=auxbasis,
        orientation_lock_strength=orientation_lock_strength,
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
        use_polarized_correlation=use_polarized_correlation,
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
    """Stub the domain's ``pool_builder`` (the seam ``prepare_inputs`` uses) so
    tests need neither the real pool nor the traj files; CCSD stays on the
    canonical (non-ledger-scoped) path."""
    from types import SimpleNamespace
    pool = _make_pool()
    fake_domain = SimpleNamespace(pool_builder=lambda cfg: pool,
                                  ccsd_species_from_ledger=False)
    monkeypatch.setattr(inputs_mod, "_get_domain_profile",
                        lambda name: fake_domain)
    return pool


@pytest.fixture
def stub_refs(monkeypatch):
    """Stub the CCSD external-reference seams so no real SCF / CCSD runs.

    Returns a ``calls`` dict tests assert against."""
    calls = {"union": 0, "precompute": 0, "precompute_kwargs": None,
             "pretrain": 0, "pretrain_kwargs": None}

    def fake_build_species_union():
        calls["union"] += 1
        # The canonical set: the pool's four species plus one (Q) no point
        # names, so the run-scoped selection is observable.
        return [SpeciesEntry(n, 0, 0, "dfs_ae")
                for n in ("P0", "P1", "P2", "P3", "Q")]

    def fake_precompute_all(species, *, cache_dir, basis, grid_level,
                            density_fit=False, auxbasis=None,
                            orientation_lock_strength=0.0):
        calls["precompute"] += 1
        calls["precompute_kwargs"] = {
            "species": species,
            "cache_dir": cache_dir,
            "basis": basis,
            "grid_level": grid_level,
            "density_fit": density_fit,
            "auxbasis": auxbasis,
            "orientation_lock_strength": orientation_lock_strength,
        }

    def fake_ensure_pretrain(data_dir, *, basis, grid_level, density_fit=False,
                             polarized=False, **_kw):
        calls["pretrain"] += 1
        calls["pretrain_kwargs"] = {
            "data_dir": data_dir,
            "basis": basis,
            "grid_level": grid_level,
            "density_fit": density_fit,
            "polarized": polarized,
        }

    monkeypatch.setattr(inputs_mod, "_build_species_union",
                        fake_build_species_union)
    monkeypatch.setattr(inputs_mod, "_precompute_all", fake_precompute_all)
    monkeypatch.setattr(inputs_mod, "_ensure_pretrain_data", fake_ensure_pretrain)
    return calls


# ---------------------------------------------------------------------------
# Happy path, ledger loaded + returned, refs ensured
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

    # external refs ensured via build_species_union -> precompute_all, for
    # the canonical species the run's two cells name (P0, P1, P2): P3 is in
    # the pool but in no cell, Q is canonical but in no point.
    assert stub_refs["union"] == 1
    assert stub_refs["precompute"] == 1
    kw = stub_refs["precompute_kwargs"]
    assert [s.name for s in kw["species"]] == ["P0", "P1", "P2"]
    assert kw["cache_dir"] == cfg.inputs.external_refs_dir
    assert kw["basis"] == cfg.inputs.basis
    assert kw["grid_level"] == cfg.inputs.grid_level


def test_prepare_inputs_locks_dfs_ccsd_refs_when_configured(
    tmp_path, stub_pool, stub_refs
):
    """The dfs_step7 (non-ledger) else-branch must thread the run-level
    orientation_lock_strength into the TRAINING CCSD references, so they lock the
    same degenerate density component as the (locked) functional and held-out
    refs. Without it the radical (OH/CH/NO) training densities are orientation-
    scrambled in the density-matching loss (the artifact the lock removes)."""
    cfg = _make_cfg(tmp_path, orientation_lock_strength=3e-5)
    _write_ledger(cfg.inputs.subset_ledger_path, _make_ledger())

    prepare_inputs(cfg)

    kw = stub_refs["precompute_kwargs"]
    assert kw["orientation_lock_strength"] == 3e-5


# ---------------------------------------------------------------------------
# The reference build is scoped to the run's own cells
# ---------------------------------------------------------------------------


def test_prepare_inputs_reports_cell_species_outside_the_canonical_set(
    tmp_path, stub_refs, monkeypatch
):
    """An AE point naming an atom the canonical set does not carry (N, 2S=3,
    as the AE-as-reactions pool does): no reference is built for it -- the
    canonical set is the only source of geometry and provenance for the DFS
    build -- and the staging names it, so the preflight log states that the
    species trains without a density target, as in every run before."""
    from types import SimpleNamespace
    pool = _make_pool()
    n_atom = _named_atoms("N", "N", charge=0, spin=3)
    pool[0] = TrainingPoint(kind="ae", name="P0",
                            species=(_named_atoms("H", "P0"), n_atom),
                            metadata={"ae_kcalmol": 100.0})
    fake_domain = SimpleNamespace(pool_builder=lambda cfg: pool,
                                  ccsd_species_from_ledger=False)
    monkeypatch.setattr(inputs_mod, "_get_domain_profile",
                        lambda name: fake_domain)
    cfg = _make_cfg(tmp_path)
    _write_ledger(cfg.inputs.subset_ledger_path, _make_ledger())

    staged = prepare_inputs(cfg)

    assert [s.name for s in stub_refs["precompute_kwargs"]["species"]] == \
        ["P0", "P1", "P2"]
    assert staged.cell_species_without_reference == (("N", 0, 3),)


# ---------------------------------------------------------------------------
# Fail-fast cases, ledger problems
# ---------------------------------------------------------------------------

def test_prepare_inputs_missing_ledger_fails(tmp_path, stub_pool, stub_refs):
    cfg = _make_cfg(tmp_path)  # ledger never written
    with pytest.raises(ValueError, match="not found"):
        prepare_inputs(cfg)


# ---------------------------------------------------------------------------
# WS3 (2026-06-20): held-out VALIDATION-slice staging
# ---------------------------------------------------------------------------

class _FakeMolSpec:
    """Minimal MoleculeSpec stand-in for held-out pool stubs."""
    def __init__(self, name, charge=0, spin=0):
        self.name = name
        self.charge = charge
        self.spin = spin
        self.atom = f"{name} 0 0 0"
        self.basis = "def2-svp"
        self.grid_level = 1
        self.atom_composition = ((name, 1),)


def _fake_held_out_pools():
    """A tiny held-out pool: 6 species, 4 reactions, returned in the
    (mols_by_name, reactions) shape of load_full_held_out_pools."""
    names = ["A", "B", "C", "D", "E", "F"]
    mols = {n: _FakeMolSpec(n) for n in names}
    reactions = [
        {"name": "rxn0", "source_pool": "bh76", "reactants": ["A"],
         "products": ["B"], "coeffs": [-1.0, 1.0], "reaction_energy_ref": 1.0},
        {"name": "rxn1", "source_pool": "bh76", "reactants": ["C"],
         "products": ["D"], "coeffs": [-1.0, 1.0], "reaction_energy_ref": 2.0},
        {"name": "rxn2", "source_pool": "w411", "reactants": ["E"],
         "products": ["F"], "coeffs": [-1.0, 1.0], "reaction_energy_ref": 3.0},
        {"name": "rxn3", "source_pool": "w411", "reactants": ["A"],
         "products": ["F"], "coeffs": [-1.0, 1.0], "reaction_energy_ref": 4.0},
    ]
    return mols, reactions


def test_the_validation_slice_loads_the_configured_pools(tmp_path, monkeypatch):
    """The in-loop validation slice is drawn from the pools the run evaluates, so a run
    that holds out a wider set validates against that set rather than against the pair.
    A configuration that states nothing gets the pair, and the reactions it stages are
    the ones it staged before the knob existed.

    Oracle: the seam's recorded arguments and the staged ``val_reactions.json``.
    """
    import dataclasses
    import json as _json
    import os
    from pathlib import Path

    from xcquinox.pipeline.cluster import inputs as inputs_mod

    seen = []

    def _capture(names=("bh76", "w411"), basis=None, grid_level=None,
                 refs_dir=None):
        seen.append((tuple(names), basis, grid_level))
        return _fake_held_out_pools()

    monkeypatch.setattr(inputs_mod, "_load_full_held_out_pools", _capture)

    cfg = _make_cfg(tmp_path, basis="def2-tzvp", grid_level=2)
    run_dir = str(tmp_path / "run_default")
    os.makedirs(run_dir, exist_ok=True)
    default_rxns = inputs_mod._stage_validation_slice(cfg, run_dir)
    assert seen[-1] == (("bh76", "w411"), "def2-tzvp", 2)

    wide = dataclasses.replace(
        cfg, inputs=dataclasses.replace(cfg.inputs,
                                        held_out_pools=("bh76", "w411", "diet150")))
    run_dir_wide = str(tmp_path / "run_wide")
    os.makedirs(run_dir_wide, exist_ok=True)
    wide_rxns = inputs_mod._stage_validation_slice(wide, run_dir_wide)
    assert seen[-1] == (("bh76", "w411", "diet150"), "def2-tzvp", 2)

    # The slice of an existing configuration is unchanged by the threading.
    assert [r["name"] for r in wide_rxns] == [r["name"] for r in default_rxns]
    staged = _json.loads(
        (Path(run_dir) / "validation" / "val_reactions.json").read_text())
    assert [r["name"] for r in staged] == [r["name"] for r in default_rxns]




# ---------------------------------------------------------------------------
# Pretrain-data staging: every required file, at the run's own identity
# ---------------------------------------------------------------------------


