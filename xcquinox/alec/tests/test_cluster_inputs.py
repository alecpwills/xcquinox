"""Tests for xcquinox.alec.cluster.inputs: input-artifact staging.

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
from xcquinox.alec.external_refs import SpeciesEntry
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


def test_prepare_inputs_precompute_is_skip_if_cached_noop_friendly(
    tmp_path, stub_pool, monkeypatch
):
    """precompute_all is skip-if-cached / idempotent: a no-op precompute (all
    refs already staged) lets prepare_inputs succeed cleanly."""
    cfg = _make_cfg(tmp_path)
    _write_ledger(cfg.inputs.subset_ledger_path, _make_ledger())

    monkeypatch.setattr(inputs_mod, "_build_species_union",
                        lambda: [SpeciesEntry("P0", 0, 0, "dfs_ae")])
    # a precompute that does nothing, every ref already cached
    monkeypatch.setattr(
        inputs_mod, "_precompute_all",
        lambda species, *, cache_dir, basis, grid_level,
        density_fit=False, auxbasis=None,
        orientation_lock_strength=0.0: None,
    )
    # pretrain data already current, ensure is a no-op
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
                        validate_overrides=True, run_preflight=True,
                        orientation_lock_strength=0.0):
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


# ---------------------------------------------------------------------------
# The reference build is scoped to the run's own cells
# ---------------------------------------------------------------------------

def test_prepare_inputs_scopes_dfs_references_to_the_runs_cells(
    tmp_path, stub_pool, stub_refs
):
    """DFS domain: the build covers the canonical species the run's cells
    name -- l2/2 names P0, P1 and l2/3 names P0, P1, P2 -- and nothing else:
    P3 sits in the pool but in no cell, Q in the canonical set but in no
    point. The staging reports what it built and the canonical size."""
    cfg = _make_cfg(tmp_path)
    _write_ledger(cfg.inputs.subset_ledger_path, _make_ledger())

    staged = prepare_inputs(cfg)

    built = [s.name for s in stub_refs["precompute_kwargs"]["species"]]
    assert built == ["P0", "P1", "P2"]
    assert all(s.source == "dfs_ae" for s in
               stub_refs["precompute_kwargs"]["species"])
    assert staged.reference_species == ("P0", "P1", "P2")
    assert staged.canonical_species_count == 5
    assert staged.cell_species_without_reference == ()


def test_prepare_inputs_ignores_ledger_cells_outside_the_grid(
    tmp_path, stub_pool, stub_refs
):
    """The notebook ledger carries every cell ever selected; an entry for a
    subset size the run does not sweep (l2/4, naming P3) contributes no
    species to this run's build."""
    cfg = _make_cfg(tmp_path)
    ledger = _make_ledger()
    ledger["l2/4"] = {
        "chosen_indices": [0, 1, 2, 3], "metric_value": 15.0,
        "point_kinds": ["ae"] * 4, "point_names": ["P0", "P1", "P2", "P3"],
        "tag": "bin04",
    }
    _write_ledger(cfg.inputs.subset_ledger_path, ledger)

    staged = prepare_inputs(cfg)

    assert [s.name for s in stub_refs["precompute_kwargs"]["species"]] == \
        ["P0", "P1", "P2"]
    assert staged.reference_species == ("P0", "P1", "P2")
    assert staged.subset_ledger == ledger


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


def test_prepare_inputs_external_domain_scopes_to_the_runs_cells(
    tmp_path, monkeypatch
):
    """An external (ccsd_species_from_ledger) domain builds the species of the
    run's cells with their own geometries; a ledger entry outside the grid
    (l2/4 naming P3, whose species is e) adds nothing."""
    from types import SimpleNamespace

    def _pt(name, spnames):
        sp = []
        for n in spnames:
            a = Atoms("He", positions=[(0.0, 0.0, 0.0)])
            a.info.update(name=n, charge=0, spin=0)
            sp.append(a)
        return TrainingPoint(kind="bh76", name=name, species=tuple(sp),
                             metadata={"e_rxn_ref": 1.0})

    pool = [_pt("P0", ["a", "b"]), _pt("P1", ["b", "c"]), _pt("P2", ["c", "d"]),
            _pt("P3", ["e"])]
    fake_domain = SimpleNamespace(pool_builder=lambda cfg: pool,
                                  ccsd_species_from_ledger=True)
    monkeypatch.setattr(inputs_mod, "_get_domain_profile", lambda name: fake_domain)
    cap = {}

    def fake_precompute(species, *, cache_dir, basis, grid_level,
                        density_fit=False, auxbasis=None, atoms_by_key=None,
                        validate_overrides=True, run_preflight=True,
                        orientation_lock_strength=0.0):
        cap["names"] = sorted(s.name for s in species)
        cap["keys"] = sorted(atoms_by_key)
    monkeypatch.setattr(inputs_mod, "_precompute_all", fake_precompute)
    monkeypatch.setattr(inputs_mod, "_ensure_pretrain_data", lambda *a, **k: None)

    cfg = _make_cfg(tmp_path)
    ledger = _make_ledger()
    ledger["l2/4"] = {
        "chosen_indices": [0, 1, 2, 3], "metric_value": 15.0,
        "point_kinds": ["bh76"] * 4, "point_names": ["P0", "P1", "P2", "P3"],
        "tag": "bin04",
    }
    _write_ledger(cfg.inputs.subset_ledger_path, ledger)

    staged = prepare_inputs(cfg)

    assert cap["names"] == ["a", "b", "c", "d"]
    assert cap["keys"] == [("a", 0, 0), ("b", 0, 0), ("c", 0, 0), ("d", 0, 0)]
    assert staged.reference_species == ("a", "b", "c", "d")
    assert staged.canonical_species_count == 0
    assert staged.cell_species_without_reference == ()


def test_prepare_inputs_builds_the_regularizer_anchors_the_specs_carry(
    tmp_path, stub_refs, monkeypatch
):
    """spec_builder injects the neutral H and Li anchors into every spec whose
    subset lacks them as single atoms, so the references must cover them even
    when no chosen point names them: a single cell on a point without lithium
    still trains with the Li anchor in its spec. DFS domain: the anchors are
    taken from the canonical set with their own sources, in canonical order."""
    from types import SimpleNamespace
    pool = _make_pool()
    fake_domain = SimpleNamespace(pool_builder=lambda cfg: pool,
                                  ccsd_species_from_ledger=False,
                                  regularize_atom_syms=("H", "Li"))
    monkeypatch.setattr(inputs_mod, "_get_domain_profile",
                        lambda name: fake_domain)
    canonical = [SpeciesEntry("H", 0, 1, "dfs_atom"),
                 SpeciesEntry("Li", 0, 1, "dfs_atom")] + \
        [SpeciesEntry(n, 0, 0, "dfs_ae") for n in ("P0", "P1", "P2", "P3", "Q")]
    monkeypatch.setattr(inputs_mod, "_build_species_union", lambda: canonical)
    cfg = _make_cfg(tmp_path)
    _write_ledger(cfg.inputs.subset_ledger_path, _make_ledger())

    staged = prepare_inputs(cfg)

    built = stub_refs["precompute_kwargs"]["species"]
    assert [(s.name, s.charge, s.spin, s.source) for s in built] == [
        ("H", 0, 1, "dfs_atom"), ("Li", 0, 1, "dfs_atom"),
        ("P0", 0, 0, "dfs_ae"), ("P1", 0, 0, "dfs_ae"), ("P2", 0, 0, "dfs_ae")]
    assert staged.reference_species == ("H", "Li", "P0", "P1", "P2")
    assert staged.cell_species_without_reference == ()


def test_prepare_inputs_external_domain_builds_the_regularizer_anchors(
    tmp_path, monkeypatch
):
    """External pool: the anchors are built with the bare-atom geometries the
    specs carry (NIST ground-state spins: H 1, Li 1), beside the cells' own
    species; an anchor a point already names is not duplicated."""
    from types import SimpleNamespace

    def _pt(name, spnames):
        sp = []
        for n in spnames:
            a = Atoms("He", positions=[(0.0, 0.0, 0.0)])
            a.info.update(name=n, charge=0, spin=0)
            sp.append(a)
        return TrainingPoint(kind="bh76", name=name, species=tuple(sp),
                             metadata={"e_rxn_ref": 1.0})

    h_atom = _named_atoms("H", "H", charge=0, spin=1)
    pool = [_pt("P0", ["a"]), _pt("P1", ["b"]),
            TrainingPoint(kind="ae", name="P2", species=(_named_atoms("H", "c"),
                                                          h_atom),
                          metadata={"ae_kcalmol": 1.0})]
    fake_domain = SimpleNamespace(pool_builder=lambda cfg: pool,
                                  ccsd_species_from_ledger=True,
                                  regularize_atom_syms=("H", "Li"))
    monkeypatch.setattr(inputs_mod, "_get_domain_profile", lambda name: fake_domain)
    cap = {}

    def fake_precompute(species, *, cache_dir, basis, grid_level,
                        density_fit=False, auxbasis=None, atoms_by_key=None,
                        validate_overrides=True, run_preflight=True,
                        orientation_lock_strength=0.0):
        cap["keys"] = sorted(atoms_by_key)
        cap["names"] = sorted(s.name for s in species)
        cap["li"] = atoms_by_key[("Li", 0, 1)]
    monkeypatch.setattr(inputs_mod, "_precompute_all", fake_precompute)
    monkeypatch.setattr(inputs_mod, "_ensure_pretrain_data", lambda *a, **k: None)

    cfg = _make_cfg(tmp_path)
    _write_ledger(cfg.inputs.subset_ledger_path, _make_ledger())

    staged = prepare_inputs(cfg)

    assert cap["keys"] == [("H", 0, 1), ("Li", 0, 1), ("a", 0, 0), ("b", 0, 0),
                           ("c", 0, 0)]
    assert cap["names"] == ["H", "Li", "a", "b", "c"]
    assert cap["li"].get_chemical_symbols() == ["Li"]
    assert cap["li"].info["spin"] == 1 and cap["li"].info["charge"] == 0
    assert sorted(staged.reference_species) == ["H", "Li", "a", "b", "c"]


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
# Fail-fast cases, ledger problems
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


@pytest.fixture
def stub_val_slice(monkeypatch):
    """Stub the held-out-pool load + spy on the SCF primitive.

    FIX 4 (WS3-INPUTS-01): validation is energy-only and rebuilds its PBE density
    at train time (``data.precompute_fixed_density_data`` runs a fresh SCF and
    NEVER reads any ``val_refs_dir`` cache). The preflight val-slice SCF precompute
    was therefore pure wasted SCF whose artifact path no spec ever resolved -- it
    is removed. The spy patches the underlying SCF primitive
    (``external_refs.run_scf_with_cache``) so any regression that re-introduces a
    val-slice precompute FAILS the no-waste tests below (precompute == 0)."""
    import xcquinox.alec.external_refs as _ext
    calls = {"precompute": 0}

    def fake_load_pools(basis="def2-svp", grid_level=1, refs_dir=None):
        return _fake_held_out_pools()

    def fake_run_scf_with_cache(*a, **k):
        calls["precompute"] += 1
        return {}

    monkeypatch.setattr(inputs_mod, "_load_full_held_out_pools",
                        fake_load_pools)
    monkeypatch.setattr(_ext, "run_scf_with_cache", fake_run_scf_with_cache)
    return calls


def test_stage_validation_slice_writes_disjoint_val_reactions(
        tmp_path, stub_val_slice):
    """_stage_validation_slice splits the held-out pool and writes a
    val_reactions.json that is the val slice and DISJOINT from the test slice.

    FIX 4: it must NOT precompute any SCF (validation rebuilds the density at
    train time; the preflight precompute was dead + wasted). val_reactions.json
    staging IS still needed (the train loop scores these reactions)."""
    from xcquinox.alec.cluster.inputs import _stage_validation_slice
    from xcquinox.alec.eval_holdout import split_held_out

    cfg = _make_cfg(tmp_path)
    import dataclasses
    cfg = dataclasses.replace(
        cfg, inputs=dataclasses.replace(
            cfg.inputs, val_refs_dir=str(tmp_path / "val_refs")))
    run_dir = str(tmp_path / "run")

    val_rxns = _stage_validation_slice(cfg, run_dir)

    # val_reactions.json written under <run_dir>/validation/.
    val_json = tmp_path / "run" / "validation" / "val_reactions.json"
    assert val_json.is_file()
    with open(val_json) as f:
        on_disk = json.load(f)

    _mols, reactions = _fake_held_out_pools()
    exp_val, exp_test = split_held_out(reactions, val_frac=cfg.hyperparams.val_frac)
    assert {r["name"] for r in on_disk} == {r["name"] for r in exp_val}
    assert {r["name"] for r in val_rxns} == {r["name"] for r in exp_val}
    # disjoint from the test slice -> the reported eval can never see val.
    assert ({r["name"] for r in on_disk}
            .isdisjoint({r["name"] for r in exp_test}))

    # FIX 4: NO wasted SCF precompute (dead path removed).
    assert stub_val_slice["precompute"] == 0


def test_prepare_inputs_stages_val_slice_when_configured(
        tmp_path, stub_pool, stub_refs, stub_val_slice):
    """prepare_inputs(cfg, run_dir=...) stages the val slice when
    inputs.val_refs_dir is set; the val_reactions.json appears under run_dir and
    NO val-slice SCF precompute runs (FIX 4)."""
    import dataclasses
    cfg = _make_cfg(tmp_path)
    cfg = dataclasses.replace(
        cfg, inputs=dataclasses.replace(
            cfg.inputs, val_refs_dir=str(tmp_path / "val_refs")))
    _write_ledger(cfg.inputs.subset_ledger_path, _make_ledger())
    run_dir = str(tmp_path / "run")

    prepare_inputs(cfg, run_dir=run_dir)

    assert (tmp_path / "run" / "validation" / "val_reactions.json").is_file()
    assert stub_val_slice["precompute"] == 0


def test_prepare_inputs_skips_val_slice_when_not_configured(
        tmp_path, stub_pool, stub_refs, stub_val_slice):
    """No val_refs_dir (default) OR no run_dir -> the val-slice staging is a
    NO-OP, so existing runs are byte-identical (no file written)."""
    cfg = _make_cfg(tmp_path)            # val_refs_dir defaults to None
    _write_ledger(cfg.inputs.subset_ledger_path, _make_ledger())

    # run_dir given but val_refs_dir None -> skip.
    prepare_inputs(cfg, run_dir=str(tmp_path / "run"))
    assert stub_val_slice["precompute"] == 0
    assert not (tmp_path / "run" / "validation").exists()

    # val_refs_dir set but run_dir omitted -> skip (no place to write reactions).
    import dataclasses
    cfg2 = dataclasses.replace(
        cfg, inputs=dataclasses.replace(
            cfg.inputs, val_refs_dir=str(tmp_path / "val_refs")))
    prepare_inputs(cfg2)
    assert stub_val_slice["precompute"] == 0


# ---------------------------------------------------------------------------
# Pretrain-data staging: every required file, at the run's own identity
# ---------------------------------------------------------------------------

def _pretrain_calls(monkeypatch):
    """Capture EVERY ``_ensure_pretrain_data`` call, not just the last."""
    calls = []
    monkeypatch.setattr(
        inputs_mod, "_ensure_pretrain_data",
        lambda data_dir, **kw: calls.append((data_dir, kw)))
    return calls


def _protocol_cfg(tmp_path, **pretrain_kw):
    """A two-architecture, mixed-rung config carrying protocol knobs."""
    import dataclasses
    cfg = _make_cfg(tmp_path, use_polarized_correlation=True)
    pretrain = dataclasses.replace(cfg.pretrain, **pretrain_kw)
    return dataclasses.replace(
        cfg,
        sweep=dataclasses.replace(
            cfg.sweep, arch=("deep_3x16", "deep_mgga_3x16")),
        pretrain=pretrain)


def test_prepare_inputs_ensures_every_required_file_with_the_protocol_keywords(
        tmp_path, stub_pool, stub_refs, monkeypatch):
    """Under ``parent_density: auto`` a mixed-rung sweep needs the PBE-density
    and the SCAN-density file; the preflight must ensure BOTH, each carrying
    the protocol keywords, or a run whose datagen was skipped trains on one of
    them built at the wrong identity."""
    cfg = _protocol_cfg(tmp_path, parent_density="auto", dfs_set=True,
                        pool_atoms=True, exchange_footing="spin_channel",
                        mesh_fraction=0.25)
    _write_ledger(cfg.inputs.subset_ledger_path, _make_ledger())
    calls = _pretrain_calls(monkeypatch)

    prepare_inputs(cfg)

    assert [kw["reference_xc"] for _d, kw in calls] == ["pbe", "scan"]
    for _data_dir, kw in calls:
        assert kw["polarized"] is True
        assert kw["dfs_set"] is True
        assert kw["pool_atoms"] is True
        assert kw["exchange_footing"] == "spin_channel"
        assert kw["mesh_fraction"] == 0.25
        assert kw["basis"] == cfg.inputs.basis
        assert kw["grid_level"] == cfg.inputs.grid_level


def test_prepare_inputs_asks_the_currency_check_at_the_runs_own_lock(
        tmp_path, stub_pool, stub_refs, monkeypatch):
    """The orientation lock is part of the data's identity: a degenerate atom's
    rows are a different component of its manifold under a different lock. A
    run at a lock other than the generator's own must not be served the file
    built at 3e-5, so the lock reaches the currency check."""
    import dataclasses
    cfg = _protocol_cfg(tmp_path)
    cfg = dataclasses.replace(
        cfg, inputs=dataclasses.replace(cfg.inputs,
                                        orientation_lock_strength=1e-4))
    _write_ledger(cfg.inputs.subset_ledger_path, _make_ledger())
    calls = _pretrain_calls(monkeypatch)

    prepare_inputs(cfg)

    assert len(calls) == 1
    assert calls[0][1]["orientation_lock_strength"] == 1e-4


def test_prepare_inputs_states_an_unlocked_run_rather_than_defaulting(
        tmp_path, stub_pool, stub_refs, monkeypatch):
    """A run that states ``inputs.orientation_lock_strength: 0.0`` is unlocked,
    and the preflight must say so on every ``ensure_pretrain_data`` call:
    leaving the keyword out would ask the currency check at the GENERATOR's
    own default (3e-5) and serve a locked file to an unlocked run."""
    cfg = _protocol_cfg(tmp_path)
    _write_ledger(cfg.inputs.subset_ledger_path, _make_ledger())
    calls = _pretrain_calls(monkeypatch)

    prepare_inputs(cfg)

    assert calls[0][1]["orientation_lock_strength"] == 0.0


def test_prepare_inputs_carries_the_waiver_the_configuration_states(
        tmp_path, stub_pool, stub_refs, monkeypatch):
    """The preflight ensures the same files the datagen stage does, and the
    refusal is applied to the requested identity before the currency check, so
    a waived run must state the waiver here too. Otherwise a run whose datagen
    stage completed raises in the preflight over a file already on disk."""
    import dataclasses
    cfg = _protocol_cfg(tmp_path)
    cfg = dataclasses.replace(
        cfg, inputs=dataclasses.replace(
            cfg.inputs, orientation_lock_strength=3e-5,
            allow_irreproducible_degenerate=True,
            irreproducible_degenerate_reason="grid level 1 example"))
    _write_ledger(cfg.inputs.subset_ledger_path, _make_ledger())
    calls = _pretrain_calls(monkeypatch)

    prepare_inputs(cfg)

    assert len(calls) == 1
    assert calls[0][1]["allow_irreproducible_degenerate"] is True


def test_prepare_inputs_states_no_waiver_when_none_is_granted(
        tmp_path, stub_pool, stub_refs, monkeypatch):
    """False is the generator's own default, so a run that waives nothing
    reaches it with the keyword set it always did."""
    cfg = _protocol_cfg(tmp_path)
    _write_ledger(cfg.inputs.subset_ledger_path, _make_ledger())
    calls = _pretrain_calls(monkeypatch)

    prepare_inputs(cfg)

    assert "allow_irreproducible_degenerate" not in calls[0][1]
