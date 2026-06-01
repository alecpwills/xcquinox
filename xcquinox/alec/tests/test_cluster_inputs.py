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
              grid_level=1, density_fit=False, auxbasis=None,
              use_polarized_correlation=False):
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
        return ["species-union-sentinel"]

    def fake_precompute_all(species, *, cache_dir, basis, grid_level,
                            density_fit=False, auxbasis=None):
        calls["precompute"] += 1
        calls["precompute_kwargs"] = {
            "species": species,
            "cache_dir": cache_dir,
            "basis": basis,
            "grid_level": grid_level,
            "density_fit": density_fit,
            "auxbasis": auxbasis,
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
        lambda species, *, cache_dir, basis, grid_level,
        density_fit=False, auxbasis=None: None,
    )
    # pretrain data already current — ensure is a no-op
    monkeypatch.setattr(
        inputs_mod, "_ensure_pretrain_data",
        lambda data_dir, **_kw: None,
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
    assert stub_refs["pretrain"] == 0
    assert isinstance(staged, StagedInputs)
    assert staged.subset_ledger == _make_ledger()


def test_prepare_inputs_bh76w411_ledger_scoped_ccsd(tmp_path, monkeypatch):
    """An external (ccsd_species_from_ledger) domain: prepare_inputs builds the
    domain pool and restricts CCSD to the union of species across the LEDGER's
    chosen points, handing their geometries to precompute_all + skipping the
    DFS override check / preflight."""
    from types import SimpleNamespace
    from ase import Atoms
    from xcquinox.alec.training_points import TrainingPoint

    def _pt(name, spnames):
        sp = []
        for n in spnames:
            a = Atoms("He", positions=[(0.0, 0.0, 0.0)])
            a.info.update(name=n, charge=0, spin=0)
            sp.append(a)
        return TrainingPoint(kind="bh76", name=name, species=tuple(sp),
                             metadata={"e_rxn_ref": 1.0})

    # pool point names match _make_ledger's point_names (P0,P1,P2)
    pool = [_pt("P0", ["a", "b"]), _pt("P1", ["b", "c"]), _pt("P2", ["c", "d"])]
    fake_domain = SimpleNamespace(pool_builder=lambda cfg: pool,
                                  ccsd_species_from_ledger=True)
    monkeypatch.setattr(inputs_mod, "_get_domain_profile", lambda name: fake_domain)

    cap = {}

    def fake_precompute(species, *, cache_dir, basis, grid_level,
                        density_fit=False, auxbasis=None, atoms_by_key=None,
                        validate_overrides=True, run_preflight=True):
        cap["names"] = sorted(s.name for s in species)
        cap["keys"] = sorted(atoms_by_key) if atoms_by_key else None
        cap["validate_overrides"] = validate_overrides
        cap["run_preflight"] = run_preflight
    monkeypatch.setattr(inputs_mod, "_precompute_all", fake_precompute)
    monkeypatch.setattr(inputs_mod, "_ensure_pretrain_data", lambda *a, **k: None)

    cfg = _make_cfg(tmp_path)
    _write_ledger(cfg.inputs.subset_ledger_path, _make_ledger())
    prepare_inputs(cfg)

    # union of species across the chosen P0,P1,P2 = {a,b,c,d} (subset, geometries
    # passed directly), with DFS override-check + preflight skipped.
    assert cap["names"] == ["a", "b", "c", "d"]
    assert cap["keys"] == [("a", 0, 0), ("b", 0, 0), ("c", 0, 0), ("d", 0, 0)]
    assert cap["validate_overrides"] is False
    assert cap["run_preflight"] is False


def test_prepare_inputs_threads_density_fit_and_ensures_pretrain(
    tmp_path, stub_pool, stub_refs
):
    """density_fit/auxbasis from inputs reach BOTH the CCSD ref precompute and
    the pretrain-data ensure; pretrain ensure runs at the configured basis and
    the run's polarization."""
    cfg = _make_cfg(tmp_path, basis="def2-tzvp", grid_level=2,
                    density_fit=True, auxbasis="def2-tzvp-jkfit",
                    use_polarized_correlation=True)
    _write_ledger(cfg.inputs.subset_ledger_path, _make_ledger())

    prepare_inputs(cfg)

    kw = stub_refs["precompute_kwargs"]
    assert kw["density_fit"] is True
    assert kw["auxbasis"] == "def2-tzvp-jkfit"

    assert stub_refs["pretrain"] == 1
    pk = stub_refs["pretrain_kwargs"]
    assert pk["data_dir"] == cfg.pretrain.data_dir
    assert pk["basis"] == "def2-tzvp"
    assert pk["grid_level"] == 2
    assert pk["density_fit"] is True
    assert pk["polarized"] is True


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
