"""Tests for xcquinox.pipeline.cluster.spec_builder: generic spec assembly.

These tests use a small synthetic ``TrainingPoint`` pool and a stub
``subset_ledger`` in the EXISTING ``subset_index_log.json`` format
(``"<metric>/<r>"`` keys carrying ``point_names``) so they stay focused on the
spec-assembly logic (name resolution, fail-fast on missing cells / unresolved
names, targets / aux-only classification, BH76 filtering, checkpoint-dir
padding, pretrain-checkpoint derivation, TestSpec wiring) without depending on
the heavy DFS pool builder.
"""
import dataclasses
import os

import pytest
from ase import Atoms

from xcquinox.pipeline.cluster.domain import get_domain_profile
from xcquinox.pipeline.cluster.grid_config import (
    GridConfig,
    SweepAxes,
    SolverNamed,
    HyperParams,
    InputPaths,
    PretrainConfig,
    ClusterResources,
    expand_grid,
)
from xcquinox.pipeline.cluster.spec_builder import (
    build_training_specs,
    build_test_spec,
    atoms_to_mol_spec,
)
from xcquinox.pipeline.losses import make_loss
from xcquinox.pipeline.training_points import (
    TrainingPoint,
)


# ---------------------------------------------------------------------------
# Synthetic pool helpers
# ---------------------------------------------------------------------------

def _named_atoms(symbol_positions, name, charge=0, spin=0):
    """Build an ASE Atoms with the info keys TrainingPoint / spec-builder need."""
    syms = [s for s, _ in symbol_positions]
    pos = [p for _, p in symbol_positions]
    a = Atoms(syms, positions=pos)
    a.info["name"] = name
    a.info["charge"] = charge
    a.info["spin"] = spin
    return a


def _ae_point(name, compound_atoms, ae_kcalmol):
    """AE TrainingPoint: compound + an H atom anchor."""
    h_anchor = _named_atoms([("H", (0.0, 0.0, 0.0))], "H")
    return TrainingPoint(
        kind="ae",
        name=name,
        species=(compound_atoms, h_anchor),
        metadata={"ae_kcalmol": ae_kcalmol},
    )


def _bh76_point(name, species_atoms, reactants, products, coeffs, e_rxn_ref):
    return TrainingPoint(
        kind="bh76",
        name=name,
        species=tuple(species_atoms),
        metadata={
            "reactants": tuple(reactants),
            "products": tuple(products),
            "coeffs": tuple(coeffs),
            "e_rxn_ref": e_rxn_ref,
        },
    )


def _ip13_point(name, neutral_atoms, cation_atoms, ip_ref):
    return TrainingPoint(
        kind="ip13",
        name=name,
        species=(neutral_atoms, cation_atoms),
        metadata={
            "neutral": neutral_atoms.info["name"],
            "cation": cation_atoms.info["name"],
            "ip_ref": ip_ref,
        },
    )


def _make_pool():
    """A 4-point synthetic pool: 2 AE + 1 BH76 + 1 IP13."""
    h2 = _named_atoms(
        [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.74))], "H2"
    )
    h2o = _named_atoms(
        [("O", (0.0, 0.0, 0.0)),
         ("H", (0.0, 0.757, 0.587)),
         ("H", (0.0, -0.757, 0.587))],
        "H2O",
    )
    # BH76 reaction species, a polyatomic NOT present as an AE point.
    n2 = _named_atoms(
        [("N", (0.0, 0.0, 0.0)), ("N", (0.0, 0.0, 1.10))], "N2"
    )
    no = _named_atoms(
        [("N", (0.0, 0.0, 0.0)), ("O", (0.0, 0.0, 1.15))], "NO"
    )
    li_neutral = _named_atoms([("Li", (0.0, 0.0, 0.0))], "Li", spin=1)
    li_cation = _named_atoms([("Li", (0.0, 0.0, 0.0))], "Li+", charge=1, spin=0)

    ae_h2 = _ae_point("H2", h2, ae_kcalmol=109.5)
    ae_h2o = _ae_point("H2O", h2o, ae_kcalmol=232.2)
    bh = _bh76_point(
        "N2_NO_rxn", [n2, no],
        reactants=("N2",), products=("NO",),
        coeffs=(-1.0, 1.0), e_rxn_ref=42.0,
    )
    ip = _ip13_point("Li_IP", li_neutral, li_cation, ip_ref=124.3)
    return [ae_h2, ae_h2o, bh, ip]


def _make_ledger():
    """Stub ledger in the EXISTING subset_index_log.json format.

    Top-level keys are ``"<metric>/<r>"``; each entry carries ``point_names``
    (the stable selection key) plus provenance-only fields.
    """
    return {
        # l2 / r=2 -> the two AE points only.
        "l2/2": {
            "chosen_indices": [0, 1],
            "metric_value": 12.5,
            "point_kinds": ["ae", "ae"],
            "point_names": ["H2", "H2O"],
            "tag": "bin02",
        },
        # l2 / r=3 -> AE + BH76 + IP13.
        "l2/3": {
            "chosen_indices": [1, 2, 3],
            "metric_value": 8.1,
            "point_kinds": ["ae", "bh76", "ip13"],
            "point_names": ["H2O", "N2_NO_rxn", "Li_IP"],
            "tag": "bin03",
        },
    }


def _make_cfg(tmp_path):
    """A GridConfig whose grid is metric=l2 x subset_size={2,3} (2 cells)."""
    sweep = SweepAxes(
        arch=("shallow",),
        loss=("L5_gradnorm_vxc_step7",),
        metric=("l2",),
        subset_size=(2, 3),
        solver=("oneshot",),
    )
    solvers = {
        "oneshot": SolverNamed(mode="oneshot", max_cycles=0),
    }
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
        subset_ledger_path=str(tmp_path / "ledger.json"),
        basis="def2-svp",
        grid_level=1,
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
    )


# ---------------------------------------------------------------------------
# atoms_to_pyscf_str / atoms_to_mol_spec
# ---------------------------------------------------------------------------


def test_atoms_to_mol_spec_wires_external_ref(tmp_path):
    refs = tmp_path / "refs"
    refs.mkdir()
    (refs / "H2.npz").write_bytes(b"stub")
    a = _named_atoms(
        [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.74))], "H2"
    )
    ms = atoms_to_mol_spec(
        a, basis="def2-svp", grid_level=1, external_refs_dir=str(refs)
    )
    assert ms.external_data_path == str(refs / "H2.npz")


# ---------------------------------------------------------------------------
# build_targets / classify_aux_only
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# build_training_specs
# ---------------------------------------------------------------------------

def test_build_training_specs_produces_one_spec_per_cell(tmp_path):
    domain = get_domain_profile("dfs_step7")
    pool = _make_pool()
    ledger = _make_ledger()
    cfg = _make_cfg(tmp_path)
    run_dir = str(tmp_path / "run")

    out = build_training_specs(pool, ledger, cfg, domain, run_dir)
    cells = expand_grid(cfg)
    assert len(out) == len(cells) == 2
    for (cell, _spec), expected_cell in zip(out, cells):
        assert cell == expected_cell


def test_build_training_specs_targets_and_aux_only(tmp_path):
    domain = get_domain_profile("dfs_step7")
    pool = _make_pool()
    ledger = _make_ledger()
    cfg = _make_cfg(tmp_path)
    out = build_training_specs(pool, ledger, cfg, domain, str(tmp_path / "run"))

    # Cell 0: (l2, 2) -> H2 + H2O AE points only. No BH76/IP13.
    cell0, spec0 = out[0]
    assert cell0.subset_size == 2
    t0 = spec0.targets_dict
    assert t0["H2O"] == pytest.approx(232.2 / domain.kcal_per_ha)
    assert t0["H2"] == pytest.approx(109.5 / domain.kcal_per_ha)
    lk0 = spec0.loss_kwargs_dict
    assert lk0["bh76_reactions"] == []
    assert lk0["ip13_pairs"] == []
    assert lk0["aux_only_names"] == ()

    # Cell 1: (l2, 3) -> H2O AE + N2_NO_rxn BH76 + Li_IP IP13.
    cell1, spec1 = out[1]
    assert cell1.subset_size == 3
    lk1 = spec1.loss_kwargs_dict
    assert len(lk1["bh76_reactions"]) == 1
    assert len(lk1["ip13_pairs"]) == 1
    # N2 and NO are BH76 species absent from any AE point -> aux-only.
    assert set(lk1["aux_only_names"]) == {"N2", "NO"}
    # BH76 e_rxn_ref converted kcal -> Ha.
    assert lk1["bh76_reactions"][0]["e_rxn_ref"] == pytest.approx(
        42.0 / domain.kcal_per_ha
    )
    assert lk1["ip13_pairs"][0]["ip_ref"] == pytest.approx(
        124.3 / domain.kcal_per_ha
    )


def test_build_training_specs_missing_cell_raises(tmp_path):
    """A grid cell with no ledger entry fails fast, naming the missing key."""
    domain = get_domain_profile("dfs_step7")
    pool = _make_pool()
    ledger = _make_ledger()
    del ledger["l2/3"]
    cfg = _make_cfg(tmp_path)
    with pytest.raises(ValueError, match=r"no entry for.*l2/3"):
        build_training_specs(pool, ledger, cfg, domain, str(tmp_path / "run"))


# ---------------------------------------------------------------------------
# build_test_spec
# ---------------------------------------------------------------------------

def test_build_test_spec_absolute_output_dir_and_ref_kcalmol(tmp_path):
    domain = get_domain_profile("dfs_step7")
    pool = _make_pool()
    ledger = _make_ledger()
    cfg = _make_cfg(tmp_path)
    run_dir = str(tmp_path / "run")
    out = build_training_specs(pool, ledger, cfg, domain, run_dir)

    _cell, training_spec = out[0]
    test_spec = build_test_spec(training_spec, run_dir, 0, domain)

    expected_dir = os.path.join(
        os.path.abspath(run_dir), "checkpoints", "spec_0000"
    )
    assert test_spec.output_dir == os.path.join(expected_dir, "eval")
    assert os.path.isabs(test_spec.output_dir)
    assert test_spec.model_checkpoint == os.path.join(expected_dir, "model.eqx")

    mk = test_spec.metric_kwargs_dict
    assert "atomization_energy" in mk
    ref = mk["atomization_energy"]["reference_ae_kcalmol"]
    # Compound molecules only; Ha -> kcal round-trip.
    assert ref["H2O"] == pytest.approx(232.2)
    assert ref["H2"] == pytest.approx(109.5)
    # Single-atom H must NOT appear in the AE reference dict.
    assert "H" not in ref


# ---------------------------------------------------------------------------
# _solver_config_from_named, accepts enum NAME or VALUE
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# build_test_spec: in-distribution transparency + optional holdout
# ---------------------------------------------------------------------------


def test_build_test_spec_holdout_ae_refs_from_holdout_targets(tmp_path):
    """Held-out AE references MUST come from the held-out set's own targets, not
    the training targets. Regression: building from training_spec.molecules left
    every held-out compound without an AE reference (silently unscored)."""
    import warnings
    domain = get_domain_profile("dfs_step7")
    pool = _make_pool()
    ledger = _make_ledger()
    cfg = _make_cfg(tmp_path)
    run_dir = str(tmp_path / "run")
    out = build_training_specs(pool, ledger, cfg, domain, run_dir)

    n2_atoms = pool[2].species[0]   # N2, a compound NOT in cell-0 training set
    from xcquinox.pipeline.cluster.spec_builder import atoms_to_mol_spec
    n2_ms = atoms_to_mol_spec(
        n2_atoms, basis="def2-svp", grid_level=1,
        external_refs_dir=str(tmp_path / "refs"),
    )
    holdout = (n2_ms,)
    _cell, training_spec = out[0]

    # (a) With holdout_targets, the held-out compound gets an AE reference.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ts_with = build_test_spec(
            training_spec, run_dir, 0, domain,
            holdout_molecules=holdout, holdout_targets={n2_ms.name: -0.40},
        )
    refs_with = ts_with.metric_kwargs_dict["atomization_energy"]["reference_ae_kcalmol"]
    assert n2_ms.name in refs_with, "held-out compound must get an AE reference"
    assert refs_with[n2_ms.name] == -0.40 * domain.kcal_per_ha
    # And no training-set compound leaked in (refs are from the eval set only).
    assert set(refs_with) == {n2_ms.name}

    # (b) Without holdout_targets, refs are empty (NOT silently from training)
    #     and a RuntimeWarning fires.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ts_without = build_test_spec(
            training_spec, run_dir, 0, domain, holdout_molecules=holdout,
        )
    refs_without = ts_without.metric_kwargs_dict["atomization_energy"]["reference_ae_kcalmol"]
    assert refs_without == {}, "must not build held-out refs from the training set"
    assert any("holdout_targets" in str(w.message) for w in caught), (
        "expected a RuntimeWarning that held-out AE refs are unavailable"
    )


# ---------------------------------------------------------------------------
# Dick atomic-regularizer anchor injection (size-1 / Li-less subset fix)
# ---------------------------------------------------------------------------

def _cfg_with_subset_sizes(tmp_path, sizes):
    """``_make_cfg`` with the sweep's ``subset_size`` axis overridden."""
    cfg = _make_cfg(tmp_path)
    sweep = dataclasses.replace(cfg.sweep, subset_size=tuple(sizes))
    return dataclasses.replace(cfg, sweep=sweep)


def test_build_training_specs_injects_missing_dick_anchor(tmp_path):
    """A size-1 subset whose only point is H-only (no Li-bearing species) must
    still carry a neutral Li single-atom anchor so the Dick regularizer
    (``regularize_atom_syms == ('H', 'Li')``) is satisfied and the L5 loss
    constructs without raising CFG-02.  Regression for the deterministic
    ``train_failed`` on ``jsd/1`` / ``l2/1`` cluster specs.
    """
    domain = get_domain_profile("dfs_step7")
    assert set(domain.regularize_atom_syms) == {"H", "Li"}

    pool = _make_pool()
    # Size-1 subset = the H2 AE point only (species: H2 compound + H anchor).
    ledger = {
        "l2/1": {
            "chosen_indices": [0],
            "metric_value": 1.0,
            "point_kinds": ["ae"],
            "point_names": ["H2"],
            "tag": "bin01",
        },
    }
    cfg = _cfg_with_subset_sizes(tmp_path, (1,))

    out = build_training_specs(pool, ledger, cfg, domain, str(tmp_path / "run"))
    assert len(out) == 1
    _cell, spec = out[0]

    # (1) The neutral single-atom anchors present must cover every Dick symbol.
    neutral_single_atom_syms = {
        next(iter(dict(ms.atom_composition)))
        for ms in spec.molecules
        if sum(dict(ms.atom_composition).values()) == 1 and int(ms.charge) == 0
    }
    assert "Li" in neutral_single_atom_syms, (
        "neutral Li anchor was not injected into the H-only size-1 subset"
    )
    assert set(domain.regularize_atom_syms) <= neutral_single_atom_syms

    # (2) The L5 loss must now construct (CFG-02 passes), replicates the
    # run_training call site (train.py).
    loss = make_loss(
        spec.loss_name,
        molecules=spec.molecules,
        pbe_anchor_weight=spec.pbe_anchor_weight,
        pbe_anchor_sample=spec.pbe_anchor_sample,
        **spec.loss_kwargs_dict,
    )
    assert loss is not None


# ---------------------------------------------------------------------------
# predicted-atom reaction-form AE (ae_as_reactions)
# ---------------------------------------------------------------------------


# 2026-06-20 (WS4): full_25 (25-cycle FULL SCF) needs gradient checkpointing to
# keep backprop memory bounded; the SolverNamed -> SolverConfig mapping must
# carry scf_grad_checkpoint through, defaulting off for existing solvers.


# 2026-06-24: full_X needs the DFS step-decaying mixer + tail-weighted energy
# loss; the SolverNamed -> SolverConfig mapping and the YAML -> SolverNamed
# parser must carry mixer_name/mixer_kwargs + scf_loss_* through, defaulting to
# the current linear/0.5 + tail-off behavior for existing solvers.


def test_build_solvers_to_config_roundtrip_builds_dfs_mixer():
    from xcquinox.pipeline.cluster.grid_config import _build_solvers
    from xcquinox.pipeline.cluster.spec_builder import _solver_config_from_named
    from xcquinox.pipeline.solver import MIXER_REGISTRY
    s = _build_solvers({
        "full_25": {
            "mode": "FULL", "max_cycles": 25, "feature_policy": "REASSEMBLE",
            "mixer_name": "decaying_linear",
            "mixer_kwargs": {"base": 0.3, "floor": 0.3},
            "scf_loss_use_tail": True,
        }
    })["full_25"]
    cfg = _solver_config_from_named(s)
    assert MIXER_REGISTRY[cfg.mixer_name].__name__ == "DecayingLinearMixer"
    assert cfg.scf_loss_use_tail is True




# ---------------------------------------------------------------------------
# The per-cell SCF seed resolution
# ---------------------------------------------------------------------------


def test_resolve_seed_xc_passes_minao_through(tmp_path):
    """The resolver passes an explicit seed through verbatim and derives only
    ``auto`` from the architecture's rung, so a run-wide superposition-of-
    atomic-densities seed reaches the solver configuration of every cell
    rather than being silently replaced by the pbe default.

    Oracle: the resolver's return, and the solver configuration of every spec
    built from a config whose inputs name the seed. The cells carry a FULL
    solver because a non-pbe seed is accepted in no other mode: ONESHOT
    evaluates at the stored PBE density and would ignore the seed.
    """
    from xcquinox.pipeline.cluster.spec_builder import resolve_seed_xc
    base = _make_cfg(tmp_path)
    inputs = dataclasses.replace(base.inputs, seed_xc="minao")
    assert resolve_seed_xc(inputs, base.sweep.arch[0]) == "minao"

    cfg = dataclasses.replace(
        base,
        inputs=inputs,
        sweep=dataclasses.replace(base.sweep, solver=("full_3",)),
        solvers={"full_3": SolverNamed(mode="FULL", max_cycles=3)},
    )
    out = build_training_specs(_make_pool(), _make_ledger(), cfg,
                               get_domain_profile("dfs_step7"),
                               str(tmp_path / "run"))
    assert len(out) == len(expand_grid(cfg)) == 2
    for cell, spec in out:
        assert spec.solver_config.seed_source == "minao", cell
