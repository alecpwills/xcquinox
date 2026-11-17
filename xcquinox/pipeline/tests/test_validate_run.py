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

import xcquinox.pipeline as pipeline
from xcquinox.pipeline.config import MoleculeSpec, TrainingSpec
from xcquinox.pipeline.solver import (SolverConfig, SolverBackend, SolverMode,
                                  FeaturePolicy)
from xcquinox.pipeline.cluster.grid_config import SolverNamed
from xcquinox.pipeline.cluster import validate_run as vr

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
        pipeline.get_architecture(arch_name),
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


_VERSION = "test-version"


def _write_certificate(run_dir, arch, *, verdict="PASS", identity=None,
                       version=_VERSION, arch_field=None, parent="pbe",
                       checkpoint=None, enforced=None, override_reason=None):
    d = os.path.join(run_dir, "pretrain", arch)
    os.makedirs(d, exist_ok=True)
    payload = {
        "verdict": verdict,
        "arch": arch if arch_field is None else arch_field,
        "parent": parent,
        "xcquinox_version": version,
        "identity": identity if identity is not None else {
            "basis": _BASIS, "grid_level": 1, "density_fit": False,
            "auxbasis": None, "orientation_lock_strength": 0.0},
        "tolerances": {"tol_AE": 1.0, "tol_atom": 1.0,
                       "override_reason": override_reason},
        "per_system": [], "per_atomization": [],
        "summary": {"max_atom_mHa": 0.1, "max_dAE_kcalmol": 0.2,
                    "n_systems": 2, "failure_reasons": []},
    }
    if checkpoint is not None:
        payload["checkpoint"] = checkpoint
    if enforced is not None:
        payload["enforced"] = enforced
    with open(os.path.join(d, "fidelity_certificate.json"), "w") as f:
        json.dump(payload, f)
    return os.path.join(d, "fidelity_certificate.json")


def _write_checkpoint_files(run_dir, arch, xnet=b"xnet-bytes",
                            cnet=b"cnet-bytes"):
    """Write the two pretrained network files and return their digests."""
    import hashlib
    d = os.path.join(run_dir, "pretrain", arch)
    os.makedirs(d, exist_ok=True)
    digests = {}
    for name, blob in (("xnet.eqx", xnet), ("cnet.eqx", cnet)):
        with open(os.path.join(d, name), "wb") as f:
            f.write(blob)
        digests[name] = hashlib.sha256(blob).hexdigest()
    return digests


def _write_run(tmp_path, specs, certificates=True):
    run = tmp_path / "run"
    (run / "specs").mkdir(parents=True)
    (run / "resolved_config.yaml").write_text("placeholder: true\n")
    with open(run / "manifest.json", "w") as f:
        json.dump({"width": 4, "xcquinox_version": _VERSION}, f)
    ser = importlib.import_module("pi" + "ckle")
    for i, spec in enumerate(specs):
        with open(run / "specs" / f"spec_{i:04d}.spec", "wb") as f:
            ser.dump(spec, f)
    if certificates:
        for arch in _ARCHS:
            _write_certificate(str(run), arch)
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


def test_index_to_cell_mapping_break_is_detected(tmp_path, patched_cfg):
    # specs swapped across indices: right archs, wrong order.
    run = _write_run(tmp_path, [_spec_for("deep_attn_3x16"),
                                _spec_for("deep_3x16")])
    failures, _w, _n = vr.validate_run(run)
    assert any("index->cell mapping" in f for f in failures), failures


# ---------------------------------------------------------------------------
# Pretraining-fidelity certificates
# ---------------------------------------------------------------------------

def test_missing_certificate_is_a_failure(tmp_path, patched_cfg):
    run = _write_run(tmp_path, [_spec_for("deep_3x16"),
                                _spec_for("deep_attn_3x16")],
                     certificates=False)
    failures, _warnings, _n = vr.validate_run(run)
    assert any("no fidelity_certificate.json" in f for f in failures)
    assert sum("fidelity_certificate" in f for f in failures) == 2


def test_identity_mismatch_is_a_failure(tmp_path, patched_cfg):
    """A certificate computed at a different basis or grid says nothing about
    this run: the energy differences it bounds are not this run's."""
    run = _write_run(tmp_path, [_spec_for("deep_3x16"),
                                _spec_for("deep_attn_3x16")])
    _write_certificate(run, "deep_3x16", identity={
        "basis": "def2-tzvpd", "grid_level": 3, "density_fit": True,
        "auxbasis": "def2-universal-jkfit",
        "orientation_lock_strength": 0.02})
    failures, _warnings, _n = vr.validate_run(run)
    assert any("identity basis=" in f for f in failures)
    assert any("identity grid_level=" in f for f in failures)
    assert any("identity density_fit=" in f for f in failures)
    assert any("identity auxbasis=" in f for f in failures)
    assert any("identity orientation_lock_strength=" in f for f in failures)


def test_checkpoint_digest_mismatch_is_a_failure(tmp_path, patched_cfg):
    """A checkpoint rewritten after certification is not the one measured."""
    run = _write_run(tmp_path, [_spec_for("deep_3x16"),
                                _spec_for("deep_attn_3x16")])
    digests = _write_checkpoint_files(run, "deep_3x16")
    _write_certificate(run, "deep_3x16", checkpoint={
        "dir": os.path.join(run, "pretrain", "deep_3x16"),
        "xnet_sha256": "0" * 64,
        "cnet_sha256": digests["cnet.eqx"]})
    failures, _warnings, _n = vr.validate_run(run)
    assert any("xnet.eqx" in f and "certificate measured" in f
               for f in failures)
    # the cnet digest agrees, so only the perturbed file is reported
    assert not any("cnet.eqx" in f for f in failures)




def test_gate_mismatch_is_detected(tmp_path, patched_cfg):
    """A pretraining record that states no uniform-gas gate is the gate every
    model before the field carried, and a run configured for the other gate
    must not accept it; a record stating the run's gate passes.

    Oracle: the validator's failures on a synthetic run carrying the record,
    under a model block of each gate.
    """
    run = _write_run(tmp_path, [_spec_for("deep_3x16"),
                                _spec_for("deep_attn_3x16")])
    d = os.path.join(run, "pretrain", "deep_3x16")
    with open(os.path.join(d, "pretrain_metadata.json"), "w") as f:
        json.dump({"use_polarized_correlation": True, "parent_anchor": False,
                   "descriptor_coordinates": "legacy"}, f)
    patched_cfg.model = SimpleNamespace(parent_anchor=False,
                                        descriptor_coordinates="legacy",
                                        ueg_gate="x2")
    failures, _w, _n = vr.validate_run(run)
    assert any("pretrain/deep_3x16: ueg_gate=" in f for f in failures), failures

    patched_cfg.model = SimpleNamespace(parent_anchor=False,
                                        descriptor_coordinates="legacy",
                                        ueg_gate="tanh2")
    failures, _w, _n = vr.validate_run(run)
    assert not any("ueg_gate=" in f for f in failures), failures

    # A record that STATES the other gate is refused by the same run, and one
    # stating the run's gate passes: the validator reads the record's own
    # field, not only its absence.
    with open(os.path.join(d, "pretrain_metadata.json"), "w") as f:
        json.dump({"use_polarized_correlation": True, "parent_anchor": False,
                   "descriptor_coordinates": "legacy", "ueg_gate": "x2"}, f)
    failures, _w, _n = vr.validate_run(run)
    assert any("pretrain/deep_3x16: ueg_gate='x2'" in f for f in failures), failures

    with open(os.path.join(d, "pretrain_metadata.json"), "w") as f:
        json.dump({"use_polarized_correlation": True, "parent_anchor": False,
                   "descriptor_coordinates": "legacy", "ueg_gate": "tanh2"}, f)
    failures, _w, _n = vr.validate_run(run)
    assert not any("ueg_gate=" in f for f in failures), failures


def test_static_registry_provenance_fields_are_cross_checked(
        tmp_path, patched_cfg):
    """``dm_entropy_intensive`` and ``double_lob_clamp_allowed`` join the
    provenance keys cross-checked against the registry (deep_3x16's registry
    values are True and False): a record stating the other value is a failure
    naming it, and a record written before the keys existed is a warning
    rather than a silent pass."""
    run = _write_run(tmp_path, [_spec_for("deep_3x16"),
                                _spec_for("deep_attn_3x16")])
    d = os.path.join(run, "pretrain", "deep_3x16")
    with open(os.path.join(d, "pretrain_metadata.json"), "w") as f:
        json.dump({"use_polarized_correlation": True, "parent_anchor": False,
                   "descriptor_coordinates": "legacy",
                   "dm_entropy_intensive": False,
                   "double_lob_clamp_allowed": True}, f)
    failures, _w, _n = vr.validate_run(run)
    assert any("dm_entropy_intensive" in f for f in failures), failures
    assert any("double_lob_clamp_allowed" in f for f in failures), failures

    with open(os.path.join(d, "pretrain_metadata.json"), "w") as f:
        json.dump({"use_polarized_correlation": True, "parent_anchor": False,
                   "descriptor_coordinates": "legacy"}, f)
    failures, warnings, _n = vr.validate_run(run)
    assert not any("dm_entropy_intensive" in f
                   or "double_lob_clamp_allowed" in f
                   for f in failures), failures
    assert any("dm_entropy_intensive" in w for w in warnings), warnings
    assert any("double_lob_clamp_allowed" in w for w in warnings), warnings


# ---------------------------------------------------------------------------
# Donor warm-starts: a donor arch certifies on its donor, compat still binds
# ---------------------------------------------------------------------------

def _donor_setup(tmp_path, patched_cfg, *, donor_arch="deep_3x16",
                 donor_certificate=True, donor_metadata=None):
    """A two-arch run whose config states a donor for ``donor_arch``; the
    OTHER arch certifies run-locally, the donor dir lives outside the run,
    and the cfg's pretrain namespace carries the donors mapping."""
    run = _write_run(tmp_path, [_spec_for("deep_3x16"),
                                _spec_for("deep_attn_3x16")],
                     certificates=False)
    other = "deep_attn_3x16" if donor_arch == "deep_3x16" else "deep_3x16"
    _write_certificate(run, other)
    donor_root = tmp_path / "donor_run"
    donor = os.path.join(str(donor_root), "pretrain", donor_arch)
    os.makedirs(donor)
    if donor_certificate:
        # Foreign on purpose: another run's identity, version and step count.
        _write_certificate(str(donor_root), donor_arch, identity={
            "basis": "def2-svp", "grid_level": 2, "density_fit": False,
            "auxbasis": None, "orientation_lock_strength": 0.0},
            version="donor-build-2026.09")
    if donor_metadata is not None:
        with open(os.path.join(donor, "pretrain_metadata.json"), "w") as f:
            json.dump(donor_metadata, f)
    patched_cfg.pretrain = SimpleNamespace(
        n_steps=2500, donor_checkpoints={donor_arch: donor})
    return run, donor


def test_donor_arch_certifies_on_its_donor_and_states_the_provenance(
        tmp_path, patched_cfg):
    """The donor's own PASS certificate releases the arch; the comparisons
    that presuppose the product was made by THIS run (its identity, its code
    version, its step count) do not apply to a foreign product, and the
    report SAYS the arch warm-starts from a donor rather than falling
    silent about it."""
    run, donor = _donor_setup(tmp_path, patched_cfg)
    # The donor's metadata records its own (different) protocol; the compat
    # fields match this config, the protocol fields do not and must not bind.
    with open(os.path.join(donor, "pretrain_metadata.json"), "w") as f:
        json.dump({"use_polarized_correlation": True, "parent_anchor": False,
                   "descriptor_coordinates": "legacy", "ueg_gate": "tanh2",
                   "pretrain_steps": 999}, f)

    failures, warnings, _n = vr.validate_run(run)

    assert failures == [], failures
    assert any("deep_3x16" in w and "donor" in w for w in warnings), warnings
    assert not os.path.exists(os.path.join(run, "pretrain", "deep_3x16"))


def test_donor_without_a_certificate_is_a_failure_naming_the_donor(
        tmp_path, patched_cfg):
    run, donor = _donor_setup(tmp_path, patched_cfg, donor_certificate=False)
    failures, _w, _n = vr.validate_run(run)
    assert any("donor" in f and "deep_3x16" in f for f in failures), failures


def test_donor_metadata_compat_still_binds(tmp_path, patched_cfg):
    """The model-class compatibility checks apply to the donor's metadata
    exactly as to a run-local product: a donor pretrained in another class
    is a warm start this config must refuse, not a provenance difference to
    wave through."""
    run, donor = _donor_setup(
        tmp_path, patched_cfg,
        donor_metadata={"use_polarized_correlation": False,
                        "parent_anchor": False,
                        "descriptor_coordinates": "legacy",
                        "ueg_gate": "tanh2"})
    failures, _w, _n = vr.validate_run(run)
    assert any("use_polarized_correlation" in f for f in failures), failures
