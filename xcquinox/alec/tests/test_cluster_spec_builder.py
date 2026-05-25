"""Tests for xcquinox.alec.cluster.spec_builder — generic spec assembly.

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

from xcquinox.alec.cluster.domain import get_domain_profile
from xcquinox.alec.cluster.grid_config import (
    GridConfig,
    SweepAxes,
    SolverNamed,
    HyperParams,
    InputPaths,
    PretrainConfig,
    ClusterResources,
    expand_grid,
)
from xcquinox.alec.cluster.spec_builder import (
    build_training_specs,
    build_test_spec,
    atoms_to_pyscf_str,
    atoms_to_mol_spec,
    build_targets,
    classify_aux_only,
    _solver_config_from_named,
)
from xcquinox.alec.losses import make_loss
from xcquinox.alec.training_points import (
    TrainingPoint,
    species_union_from_points,
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
    # BH76 reaction species — a polyatomic NOT present as an AE point.
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

def test_atoms_to_pyscf_str_format():
    a = _named_atoms(
        [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.74))], "H2"
    )
    s = atoms_to_pyscf_str(a)
    assert s == "H 0.000000 0.000000 0.000000; H 0.000000 0.000000 0.740000"


def test_atoms_to_mol_spec_no_external_ref(tmp_path):
    a = _named_atoms(
        [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 0.74))], "H2"
    )
    ms = atoms_to_mol_spec(
        a, basis="def2-svp", grid_level=1,
        external_refs_dir=str(tmp_path / "refs"),
    )
    assert ms.name == "H2"
    assert ms.basis == "def2-svp"
    assert ms.grid_level == 1
    assert ms.external_data_path is None
    assert dict(ms.atom_composition) == {"H": 2}


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

def test_build_targets_and_aux_only_classification(tmp_path):
    domain = get_domain_profile("dfs_step7")
    pool = _make_pool()
    # mol_specs spanning an AE compound (H2O), an aux polyatomic (N2),
    # and a single atom (H).
    h2o = atoms_to_mol_spec(
        pool[1].species[0], basis="def2-svp", grid_level=1,
        external_refs_dir=str(tmp_path),
    )
    n2 = atoms_to_mol_spec(
        pool[2].species[0], basis="def2-svp", grid_level=1,
        external_refs_dir=str(tmp_path),
    )
    h_atom = atoms_to_mol_spec(
        pool[0].species[1], basis="def2-svp", grid_level=1,
        external_refs_dir=str(tmp_path),
    )
    mol_specs = (h2o, n2, h_atom)
    ae_ref = {"H2O": 232.2}  # N2 absent -> aux-only

    aux = classify_aux_only(mol_specs, ae_ref)
    assert aux == ("N2",)

    targets = build_targets(mol_specs, ae_ref, domain)
    # AE compound: kcal -> Ha
    assert targets["H2O"] == pytest.approx(232.2 / domain.kcal_per_ha)
    # aux polyatomic: 0.0 placeholder
    assert targets["N2"] == 0.0
    # single atom: Chakravorty anchor
    assert targets["H"] == pytest.approx(domain.atom_energies["H"])


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


def test_build_training_specs_resolves_points_by_name(tmp_path):
    """The chosen training points come from ``point_names``, not indices."""
    domain = get_domain_profile("dfs_step7")
    pool = _make_pool()
    ledger = _make_ledger()
    cfg = _make_cfg(tmp_path)
    out = build_training_specs(pool, ledger, cfg, domain, str(tmp_path / "run"))

    # Cell 0 = (l2, 2): point_names ["H2", "H2O"] -> both AE compounds present.
    _cell0, spec0 = out[0]
    names0 = {m.name for m in spec0.molecules}
    assert {"H2", "H2O"} <= names0

    # Cell 1 = (l2, 3): point_names ["H2O", "N2_NO_rxn", "Li_IP"].
    _cell1, spec1 = out[1]
    names1 = {m.name for m in spec1.molecules}
    assert {"H2O", "N2", "NO", "Li", "Li+"} <= names1
    # H2 was NOT chosen for cell 1 -> absent from its molecules.
    assert "H2" not in names1


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


def test_build_training_specs_sets_require_atom_anchors_false(tmp_path):
    domain = get_domain_profile("dfs_step7")
    pool = _make_pool()
    ledger = _make_ledger()
    cfg = _make_cfg(tmp_path)
    out = build_training_specs(pool, ledger, cfg, domain, str(tmp_path / "run"))
    for _cell, spec in out:
        assert spec.require_atom_anchors is False


def test_build_training_specs_checkpoint_dir_is_absolute_padded(tmp_path):
    domain = get_domain_profile("dfs_step7")
    pool = _make_pool()
    ledger = _make_ledger()
    cfg = _make_cfg(tmp_path)
    run_dir = str(tmp_path / "run")
    out = build_training_specs(pool, ledger, cfg, domain, run_dir)
    for idx, (_cell, spec) in enumerate(out):
        expected = os.path.join(
            os.path.abspath(run_dir), "checkpoints", f"spec_{idx:04d}"
        )
        assert spec.checkpoint_dir == expected
        assert os.path.isabs(spec.checkpoint_dir)


def test_build_training_specs_pretrain_checkpoint_is_per_arch(tmp_path):
    """pretrain_checkpoint is ``<run_dir>/pretrain/<arch>/`` — the run-scoped dir
    the pretrain stage writes for that architecture, co-located with the run's
    other artifacts (run_dir is unique per submission, so two runs of the same
    arch don't clobber each other)."""
    domain = get_domain_profile("dfs_step7")
    pool = _make_pool()
    ledger = _make_ledger()
    cfg = _make_cfg(tmp_path)
    run_dir = str(tmp_path / "run")
    out = build_training_specs(pool, ledger, cfg, domain, run_dir)
    for cell, spec in out:
        expected = os.path.join(os.path.abspath(run_dir), "pretrain", cell.arch)
        assert spec.pretrain_checkpoint == expected
        # The synthetic grid sweeps only arch="shallow".
        assert spec.pretrain_checkpoint == os.path.join(
            os.path.abspath(run_dir), "pretrain", "shallow"
        )


def test_build_training_specs_hyperparams_wired(tmp_path):
    domain = get_domain_profile("dfs_step7")
    pool = _make_pool()
    ledger = _make_ledger()
    cfg = _make_cfg(tmp_path)
    out = build_training_specs(pool, ledger, cfg, domain, str(tmp_path / "run"))
    _cell, spec = out[0]
    assert spec.n_steps == 100
    assert spec.lr_start == pytest.approx(1e-3)
    assert spec.lr_end == pytest.approx(1e-5)
    assert spec.grad_clip == pytest.approx(1.0)
    lk = spec.loss_kwargs_dict
    assert lk["vxc_weight"] == pytest.approx(0.01)
    assert lk["density_weight"] == pytest.approx(0.1)
    assert lk["regularize_atom_syms"] == ("H", "Li")


def test_build_training_specs_missing_cell_raises(tmp_path):
    """A grid cell with no ledger entry fails fast, naming the missing key."""
    domain = get_domain_profile("dfs_step7")
    pool = _make_pool()
    ledger = _make_ledger()
    del ledger["l2/3"]
    cfg = _make_cfg(tmp_path)
    with pytest.raises(ValueError, match=r"no entry for.*l2/3"):
        build_training_specs(pool, ledger, cfg, domain, str(tmp_path / "run"))


def test_build_training_specs_unresolved_point_name_raises(tmp_path):
    """A ledger point_name absent from the pool fails fast, naming it."""
    domain = get_domain_profile("dfs_step7")
    pool = _make_pool()
    ledger = _make_ledger()
    # Replace a real point name with one not in the pool.
    ledger["l2/2"]["point_names"] = ["H2", "NOT_IN_POOL"]
    cfg = _make_cfg(tmp_path)
    with pytest.raises(ValueError, match="NOT_IN_POOL"):
        build_training_specs(pool, ledger, cfg, domain, str(tmp_path / "run"))


def test_build_training_specs_missing_point_names_key_raises(tmp_path):
    """A ledger entry with no 'point_names' key fails fast, naming the cell key."""
    domain = get_domain_profile("dfs_step7")
    pool = _make_pool()
    ledger = _make_ledger()
    # Remove point_names entirely from one entry.
    del ledger["l2/2"]["point_names"]
    cfg = _make_cfg(tmp_path)
    with pytest.raises(ValueError, match=r"l2/2.*malformed|malformed.*l2/2"):
        build_training_specs(pool, ledger, cfg, domain, str(tmp_path / "run"))


def test_build_training_specs_empty_point_names_raises(tmp_path):
    """A ledger entry with point_names=[] fails fast, naming the cell key."""
    domain = get_domain_profile("dfs_step7")
    pool = _make_pool()
    ledger = _make_ledger()
    # Set point_names to an empty list.
    ledger["l2/2"]["point_names"] = []
    cfg = _make_cfg(tmp_path)
    with pytest.raises(ValueError, match=r"l2/2.*malformed|malformed.*l2/2"):
        build_training_specs(pool, ledger, cfg, domain, str(tmp_path / "run"))


def test_build_training_specs_cells_subset(tmp_path):
    domain = get_domain_profile("dfs_step7")
    pool = _make_pool()
    ledger = _make_ledger()
    cfg = _make_cfg(tmp_path)
    cells = expand_grid(cfg)
    out = build_training_specs(
        pool, ledger, cfg, domain, str(tmp_path / "run"), cells=cells[:1]
    )
    assert len(out) == 1
    assert out[0][0] == cells[0]


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


def test_build_test_spec_excludes_aux_only_from_ae_reference(tmp_path):
    """Aux-only reaction species (BH76/IP13 polyatomics with no real AE target)
    must NOT appear in reference_ae_kcalmol. Otherwise eval scores their full
    atomization energy against a 0.0 reference — the CH4/HF ~+440 kcal/mol
    artifact. The training loss already excludes them via classify_aux_only; the
    eval reference must do the same."""
    domain = get_domain_profile("dfs_step7")
    pool = _make_pool()
    ledger = _make_ledger()
    cfg = _make_cfg(tmp_path)
    run_dir = str(tmp_path / "run")
    out = build_training_specs(pool, ledger, cfg, domain, run_dir)

    # ledger l2/3 -> [H2O (AE), N2_NO_rxn (bh76: species N2, NO), Li_IP (ip13)].
    spec_by_ss = {cell.subset_size: ts for cell, ts in out}
    ts = spec_by_ss[3]
    # The bh76 reaction species are recorded as aux-only (the data the fix uses).
    aux = set(ts.loss_kwargs_dict.get("aux_only_names", ()))
    assert "N2" in aux and "NO" in aux

    test_spec = build_test_spec(ts, run_dir, 1, domain)
    ref = test_spec.metric_kwargs_dict["atomization_energy"]["reference_ae_kcalmol"]
    assert "H2O" in ref                          # real AE compound kept
    assert "N2" not in ref and "NO" not in ref   # aux-only excluded


def test_build_test_spec_metrics_and_eval_molecules(tmp_path):
    domain = get_domain_profile("dfs_step7")
    pool = _make_pool()
    ledger = _make_ledger()
    cfg = _make_cfg(tmp_path)
    run_dir = str(tmp_path / "run")
    out = build_training_specs(pool, ledger, cfg, domain, run_dir)
    _cell, training_spec = out[0]
    test_spec = build_test_spec(training_spec, run_dir, 0, domain)

    assert test_spec.metrics == (
        "total_energy", "atomization_energy",
        "density_rmse", "scf_convergence",
    )
    # Eval molecules come straight from training_spec.molecules.
    assert test_spec.molecules == training_spec.molecules
    assert test_spec.atom_energies_dict == dict(domain.atom_energies)


# ---------------------------------------------------------------------------
# _solver_config_from_named — accepts enum NAME or VALUE
# ---------------------------------------------------------------------------

def test_solver_config_accepts_enum_name_and_value():
    """The step-7 configs spell the solver mode / feature policy as the
    uppercase enum NAME ('ONESHOT'/'FULL'/'REASSEMBLE'); the unit tests use the
    lowercase enum VALUE ('oneshot'/'full'/'reassemble'). Both must resolve, or
    spec-building dies at the preflight stage on the real config."""
    from xcquinox.alec.solver import SolverMode, FeaturePolicy

    by_value = _solver_config_from_named(SolverNamed(mode="oneshot", max_cycles=0))
    assert by_value.mode == SolverMode.ONESHOT

    by_name = _solver_config_from_named(SolverNamed(mode="ONESHOT", max_cycles=0))
    assert by_name.mode == SolverMode.ONESHOT

    full_named = _solver_config_from_named(
        SolverNamed(mode="FULL", max_cycles=3, feature_policy="REASSEMBLE")
    )
    assert full_named.mode == SolverMode.FULL
    assert full_named.feature_policy == FeaturePolicy.REASSEMBLE

    full_value = _solver_config_from_named(
        SolverNamed(mode="full", max_cycles=3, feature_policy="reassemble")
    )
    assert full_value.mode == SolverMode.FULL
    assert full_value.feature_policy == FeaturePolicy.REASSEMBLE


def test_solver_config_rejects_unknown_mode():
    """A mode that is neither a valid enum name nor value is a clear error."""
    with pytest.raises(ValueError, match="SolverMode"):
        _solver_config_from_named(SolverNamed(mode="bogus", max_cycles=0))


# ---------------------------------------------------------------------------
# build_test_spec — in-distribution transparency + optional holdout
# ---------------------------------------------------------------------------

def test_build_test_spec_default_molecules_unchanged(tmp_path):
    """Backward-compat: default call (no holdout) keeps molecules == training_spec.molecules."""
    import warnings
    domain = get_domain_profile("dfs_step7")
    pool = _make_pool()
    ledger = _make_ledger()
    cfg = _make_cfg(tmp_path)
    run_dir = str(tmp_path / "run")
    out = build_training_specs(pool, ledger, cfg, domain, run_dir)
    _cell, training_spec = out[0]

    with warnings.catch_warnings():
        warnings.simplefilter("always")
        test_spec = build_test_spec(training_spec, run_dir, 0, domain)

    assert test_spec.molecules == training_spec.molecules


def test_build_test_spec_default_emits_in_distribution_warning(tmp_path):
    """Default (no holdout_molecules) must emit a RuntimeWarning that names
    the in-distribution / not-held-out nature of the evaluation."""
    import warnings
    domain = get_domain_profile("dfs_step7")
    pool = _make_pool()
    ledger = _make_ledger()
    cfg = _make_cfg(tmp_path)
    run_dir = str(tmp_path / "run")
    out = build_training_specs(pool, ledger, cfg, domain, run_dir)
    _cell, training_spec = out[0]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build_test_spec(training_spec, run_dir, 0, domain)

    runtime_warnings = [w for w in caught if issubclass(w.category, RuntimeWarning)]
    assert runtime_warnings, "Expected at least one RuntimeWarning from build_test_spec default path"
    # The warning message must mention in-distribution (or training) and not held-out.
    msg = str(runtime_warnings[0].message).lower()
    assert "in-distribution" in msg or "in distribution" in msg or "training" in msg


def test_build_test_spec_holdout_molecules_used_when_provided(tmp_path):
    """When holdout_molecules is provided, the TestSpec evaluates on those
    molecules instead of the training set."""
    import warnings
    domain = get_domain_profile("dfs_step7")
    pool = _make_pool()
    ledger = _make_ledger()
    cfg = _make_cfg(tmp_path)
    run_dir = str(tmp_path / "run")
    out = build_training_specs(pool, ledger, cfg, domain, run_dir)

    # Build a held-out MoleculeSpec from a molecule NOT in the training set for cell 0.
    # Cell 0 uses subset_size=2: H2 + H2O only.
    # We'll use the N2 molecule spec (from the pool BH76 point) as the held-out set.
    n2_atoms = pool[2].species[0]   # N2 Atoms
    from xcquinox.alec.cluster.spec_builder import atoms_to_mol_spec
    n2_ms = atoms_to_mol_spec(
        n2_atoms, basis="def2-svp", grid_level=1,
        external_refs_dir=str(tmp_path / "refs"),
    )
    holdout = (n2_ms,)

    _cell, training_spec = out[0]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        test_spec = build_test_spec(
            training_spec, run_dir, 0, domain,
            holdout_molecules=holdout,
        )

    assert test_spec.molecules == holdout
    # The held-out molecules must NOT match training_spec.molecules.
    assert test_spec.molecules != training_spec.molecules


def test_build_test_spec_holdout_suppresses_in_distribution_warning(tmp_path):
    """When a held-out set is provided, no in-distribution RuntimeWarning is emitted."""
    import warnings
    domain = get_domain_profile("dfs_step7")
    pool = _make_pool()
    ledger = _make_ledger()
    cfg = _make_cfg(tmp_path)
    run_dir = str(tmp_path / "run")
    out = build_training_specs(pool, ledger, cfg, domain, run_dir)

    n2_atoms = pool[2].species[0]
    from xcquinox.alec.cluster.spec_builder import atoms_to_mol_spec
    n2_ms = atoms_to_mol_spec(
        n2_atoms, basis="def2-svp", grid_level=1,
        external_refs_dir=str(tmp_path / "refs"),
    )
    holdout = (n2_ms,)

    _cell, training_spec = out[0]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build_test_spec(
            training_spec, run_dir, 0, domain,
            holdout_molecules=holdout,
        )

    runtime_warnings = [
        w for w in caught
        if issubclass(w.category, RuntimeWarning)
        and ("in-distribution" in str(w.message).lower()
             or "in distribution" in str(w.message).lower()
             or "training" in str(w.message).lower())
    ]
    assert not runtime_warnings, (
        "No in-distribution RuntimeWarning expected when holdout_molecules is provided"
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

    # (2) The L5 loss must now construct (CFG-02 passes) — replicates the
    # run_training call site (train.py).
    loss = make_loss(
        spec.loss_name,
        molecules=spec.molecules,
        pbe_anchor_weight=spec.pbe_anchor_weight,
        pbe_anchor_sample=spec.pbe_anchor_sample,
        **spec.loss_kwargs_dict,
    )
    assert loss is not None


def test_build_training_specs_no_spurious_anchor_when_present(tmp_path):
    """A subset that already carries both Dick anchors (l2/3 includes the IP13
    neutral Li) must NOT gain an injected duplicate — the molecule set is
    byte-identical to the plain species union, so currently-passing specs are
    unchanged.
    """
    domain = get_domain_profile("dfs_step7")
    pool = _make_pool()
    ledger = _make_ledger()
    cfg = _cfg_with_subset_sizes(tmp_path, (3,))

    out = build_training_specs(pool, ledger, cfg, domain, str(tmp_path / "run"))
    assert len(out) == 1
    _cell, spec = out[0]

    points_by_name = {tp.name: tp for tp in pool}
    chosen = [points_by_name[pn] for pn in ledger["l2/3"]["point_names"]]
    expected_names = sorted(a.info["name"] for a in species_union_from_points(chosen))
    got_names = sorted(ms.name for ms in spec.molecules)
    assert got_names == expected_names, (
        "molecule set diverged from the plain species union — a spurious anchor "
        "was injected for an already-present symbol"
    )
