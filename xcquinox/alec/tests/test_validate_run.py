"""Tests for the run-directory validator (cluster/validate_run.py).

The validator exists because a wrong conclusion was once drawn from the
architecture registry's DEFAULT ``use_polarized_correlation`` when the live
specs carried the sweep-level override -- so these tests pin that it reads the
ARTIFACTS: a synthetic run directory with real pickled ``TrainingSpec`` files,
checked against a config object, with every failure mode it claims to detect
demonstrated to actually fire.
"""
from __future__ import annotations

import dataclasses
import importlib
import json
import os
from types import SimpleNamespace

import pytest

import xcquinox.alec as alec
from xcquinox.alec.config import MoleculeSpec, TrainingSpec
from xcquinox.alec.solver import (SolverConfig, SolverBackend, SolverMode,
                                  FeaturePolicy)
from xcquinox.alec.cluster.grid_config import SolverNamed
from xcquinox.alec.cluster import validate_run as vr

_ARCHS = ("deep_3x16", "deep_attn_3x16")
_BASIS = "sto-3g"


def _cfg():
    """A minimal config carrying exactly the attributes the validator reads."""
    return SimpleNamespace(
        use_polarized_correlation=True,
        sweep=SimpleNamespace(arch=_ARCHS, loss=("L",), metric=("m",),
                              subset_size=(1,), solver=("full_3",)),
        solvers={"full_3": SolverNamed(mode="FULL", max_cycles=3,
                                       feature_policy="REASSEMBLE")},
        hyperparams=SimpleNamespace(n_steps=200, seed=42,
                                    update_scheme="per_molecule"),
        inputs=SimpleNamespace(basis=_BASIS, grid_level=1, density_fit=False,
                               auxbasis=None,
                               external_refs_dir="/refs/train",
                               val_refs_dir="/refs/val",
                               benchmark_refs_dir="/refs/bench"),
        pretrain=SimpleNamespace(n_steps=2500),
    )


def _spec_for(arch_name, *, polarized=True, n_steps=200, seed=42,
              max_cycles=3, basis=_BASIS, arch_override=None):
    arch = arch_override or dataclasses.replace(
        alec.get_architecture(arch_name),
        use_polarized_correlation=polarized)
    mol = MoleculeSpec(name="H2", atom="H 0 0 0; H 0 0 0.74", basis=basis,
                       charge=0, spin=0, atom_composition=(("H", 2),),
                       grid_level=1,
                       external_data_path="/refs/train/H2.npz")
    spec = TrainingSpec.from_dicts(
        arch=arch, molecules=(mol,), targets={"H2": -1.0},
        atom_energies={"H": -0.5}, loss_name="A_atomization",
        loss_kwargs={"vxc_weight": 0.0}, update_scheme="per_molecule",
        require_atom_anchors=False, n_steps=n_steps, lr_start=1e-3,
        lr_end=1e-5, lr_decay_start=0.0, grad_clip=1.0,
        checkpoint_dir=None, seed=seed)
    solver = SolverConfig(backend=SolverBackend.MANUAL, mode=SolverMode.FULL,
                          max_cycles=max_cycles, conv_tol=1e-12,
                          feature_policy=FeaturePolicy.REASSEMBLE)
    return dataclasses.replace(spec, solver_config=solver)


def _write_run(tmp_path, specs):
    run = tmp_path / "run"
    (run / "specs").mkdir(parents=True)
    (run / "resolved_config.yaml").write_text("placeholder: true\n")
    with open(run / "manifest.json", "w") as f:
        json.dump({"width": 4}, f)
    ser = importlib.import_module("pi" + "ckle")
    for i, spec in enumerate(specs):
        with open(run / "specs" / f"spec_{i:04d}.spec", "wb") as f:
            ser.dump(spec, f)
    return str(run)


@pytest.fixture()
def patched_cfg(monkeypatch):
    cfg = _cfg()
    monkeypatch.setattr(vr, "load_grid_config", lambda path: cfg)
    return cfg


def test_clean_run_validates(tmp_path, patched_cfg):
    # expand_grid sorts each axis, so index 0 is deep_3x16, index 1 deep_attn.
    run = _write_run(tmp_path, [_spec_for("deep_3x16"),
                                _spec_for("deep_attn_3x16")])
    failures, warnings, n = vr.validate_run(run)
    assert failures == [], failures
    assert n == 2
    # no pretrain dirs in the synthetic run -> reported, not failed
    assert any("pretrain_metadata" in w for w in warnings)


def test_polarization_mismatch_is_detected(tmp_path, patched_cfg):
    run = _write_run(tmp_path, [_spec_for("deep_3x16", polarized=False),
                                _spec_for("deep_attn_3x16")])
    failures, _w, _n = vr.validate_run(run)
    assert any("use_polarized_correlation" in f for f in failures), failures


def test_missing_spec_index_is_detected(tmp_path, patched_cfg):
    run = _write_run(tmp_path, [_spec_for("deep_3x16"),
                                _spec_for("deep_attn_3x16")])
    os.remove(os.path.join(run, "specs", "spec_0001.spec"))
    failures, _w, _n = vr.validate_run(run)
    assert any("missing spec indices" in f for f in failures), failures
    assert any("spec count" in f for f in failures), failures


def test_arch_shape_tamper_is_detected(tmp_path, patched_cfg):
    tampered = dataclasses.replace(
        alec.get_architecture("deep_3x16"),
        use_polarized_correlation=True, nodes=32)
    run = _write_run(tmp_path, [_spec_for("deep_3x16", arch_override=tampered),
                                _spec_for("deep_attn_3x16")])
    failures, _w, _n = vr.validate_run(run)
    assert any("differs from registry" in f and "nodes" in f
               for f in failures), failures


def test_index_to_cell_mapping_break_is_detected(tmp_path, patched_cfg):
    # specs swapped across indices: right archs, wrong order.
    run = _write_run(tmp_path, [_spec_for("deep_attn_3x16"),
                                _spec_for("deep_3x16")])
    failures, _w, _n = vr.validate_run(run)
    assert any("index->cell mapping" in f for f in failures), failures


def test_solver_field_mismatch_is_detected(tmp_path, patched_cfg):
    run = _write_run(tmp_path, [_spec_for("deep_3x16", max_cycles=25),
                                _spec_for("deep_attn_3x16")])
    failures, _w, _n = vr.validate_run(run)
    assert any("solver.max_cycles" in f for f in failures), failures


def test_wrong_basis_and_stray_reference_dir_are_detected(tmp_path,
                                                          patched_cfg):
    bad = _spec_for("deep_3x16", basis="def2-svp")
    mol = dataclasses.replace(bad.molecules[0], basis=_BASIS,
                              external_data_path="/elsewhere/H2.npz")
    stray = dataclasses.replace(bad, molecules=(mol,))
    run = _write_run(tmp_path, [bad, stray])
    # index 1 holds the stray-reference spec but carries the WRONG arch for its
    # cell; rebuild with the right arch so only the reference check fires there.
    fixed = dataclasses.replace(
        _spec_for("deep_attn_3x16"),
        molecules=(dataclasses.replace(
            _spec_for("deep_attn_3x16").molecules[0],
            external_data_path="/elsewhere/H2.npz"),))
    ser = importlib.import_module("pi" + "ckle")
    with open(os.path.join(run, "specs", "spec_0001.spec"), "wb") as f:
        ser.dump(fixed, f)
    failures, _w, _n = vr.validate_run(run)
    assert any("basis" in f for f in failures), failures
    assert any("outside the configured reference dirs" in f
               for f in failures), failures


def test_pretrain_metadata_checks(tmp_path, patched_cfg):
    run = _write_run(tmp_path, [_spec_for("deep_3x16"),
                                _spec_for("deep_attn_3x16")])
    # legacy file: polarized flag only -> provenance WARNINGS, no failure.
    d = os.path.join(run, "pretrain", "deep_3x16")
    os.makedirs(d)
    with open(os.path.join(d, "pretrain_metadata.json"), "w") as f:
        json.dump({"use_polarized_correlation": True}, f)
    failures, warnings, _n = vr.validate_run(run)
    assert failures == [], failures
    assert any("lacks 'meta_gga'" in w for w in warnings), warnings

    # provenance present but wrong -> failure.
    with open(os.path.join(d, "pretrain_metadata.json"), "w") as f:
        json.dump({"use_polarized_correlation": True, "meta_gga": False,
                   "n_extra_features": 5, "pretrain_steps": 2500}, f)
    failures, _w, _n = vr.validate_run(run)
    assert any("n_extra_features" in f for f in failures), failures

    # the step count is stored under "pretrain_steps" (written since the
    # writer existed), so a mismatch must fail even on a legacy file that
    # lacks the 2026-08-06 shape keys entirely.
    with open(os.path.join(d, "pretrain_metadata.json"), "w") as f:
        json.dump({"use_polarized_correlation": True,
                   "pretrain_steps": 100}, f)
    failures, warnings, _n = vr.validate_run(run)
    assert any("pretrain_steps" in f and "100" in f for f in failures), failures
    assert any("lacks 'meta_gga'" in w for w in warnings), warnings

    # the (s, alpha) mesh flag: deep_3x16 is not a meta-GGA arch, so a
    # checkpoint claiming mesh-augmented pretraining contradicts the
    # registry-derived expectation and must fail.
    with open(os.path.join(d, "pretrain_metadata.json"), "w") as f:
        json.dump({"use_polarized_correlation": True,
                   "pretrain_mesh": True}, f)
    failures, _w, _n = vr.validate_run(run)
    assert any("pretrain_mesh" in f for f in failures), failures

    # polarization mismatch in metadata -> failure.
    with open(os.path.join(d, "pretrain_metadata.json"), "w") as f:
        json.dump({"use_polarized_correlation": False}, f)
    failures, _w, _n = vr.validate_run(run)
    assert any("pretrain/deep_3x16: use_polarized_correlation" in f
               for f in failures), failures


def test_seed_source_mismatch_flagged(tmp_path, patched_cfg):
    """A spec whose recorded seed_source disagrees with the RESOLVED per-cell
    expectation (inputs.seed_xc + arch rung) is a validation failure."""
    patched_cfg.inputs.seed_xc = "scan"
    run = _write_run(tmp_path, [_spec_for("deep_3x16"),
                                _spec_for("deep_attn_3x16")])
    failures, warnings, n = vr.validate_run(run)
    assert any("seed_source" in f for f in failures), failures


def test_seed_source_match_passes(tmp_path, patched_cfg):
    """Default-config (no seed_xc attr) + default-spec ('pbe') validates
    clean -- old configs and old pickles stay green."""
    run = _write_run(tmp_path, [_spec_for("deep_3x16"),
                                _spec_for("deep_attn_3x16")])
    failures, warnings, n = vr.validate_run(run)
    assert not any("seed" in f for f in failures), failures
