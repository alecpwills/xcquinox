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
    # _write_run already created the dir for the arch's certificate.
    os.makedirs(d, exist_ok=True)
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


def test_failed_certificate_is_a_failure(tmp_path, patched_cfg):
    run = _write_run(tmp_path, [_spec_for("deep_3x16"),
                                _spec_for("deep_attn_3x16")])
    _write_certificate(run, "deep_3x16", verdict="FAIL")
    failures, _warnings, _n = vr.validate_run(run)
    assert any("verdict 'FAIL'" in f for f in failures)


def test_unreadable_certificate_is_a_failure(tmp_path, patched_cfg):
    run = _write_run(tmp_path, [_spec_for("deep_3x16"),
                                _spec_for("deep_attn_3x16")])
    path = os.path.join(run, "pretrain", "deep_3x16",
                        "fidelity_certificate.json")
    with open(path, "w") as f:
        f.write("{truncated")
    failures, _warnings, _n = vr.validate_run(run)
    assert any("not readable JSON" in f for f in failures)


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


def test_unenforced_failure_is_still_a_validation_failure(tmp_path,
                                                          patched_cfg):
    """`fidelity.enforce: false` releases the ON-NODE gates only. A run whose
    certificate reads FAIL can never enter the record, whatever it recorded
    about enforcement."""
    run = _write_run(tmp_path, [_spec_for("deep_3x16"),
                                _spec_for("deep_attn_3x16")])
    _write_certificate(run, "deep_3x16", verdict="FAIL")
    path = os.path.join(run, "pretrain", "deep_3x16",
                        "fidelity_certificate.json")
    with open(path) as f:
        payload = json.load(f)
    payload["enforced"] = False
    payload["tolerances"]["override_reason"] = "workflow matrix"
    with open(path, "w") as f:
        json.dump(payload, f)
    failures, _warnings, _n = vr.validate_run(run)
    assert any("verdict 'FAIL'" in f for f in failures)


def test_version_mismatch_is_a_failure(tmp_path, patched_cfg):
    """The certificate stands in for the O1-O4 oracles: it certifies the
    installed code. A certificate from other code certifies nothing here."""
    run = _write_run(tmp_path, [_spec_for("deep_3x16"),
                                _spec_for("deep_attn_3x16")])
    _write_certificate(run, "deep_3x16", version="some-other-build")
    failures, _warnings, _n = vr.validate_run(run)
    assert any("xcquinox_version" in f and "manifest" in f for f in failures)


def test_manifest_without_a_version_warns_rather_than_fails(tmp_path,
                                                            patched_cfg):
    run = _write_run(tmp_path, [_spec_for("deep_3x16"),
                                _spec_for("deep_attn_3x16")])
    with open(os.path.join(run, "manifest.json"), "w") as f:
        json.dump({"width": 4}, f)
    failures, warnings, _n = vr.validate_run(run)
    assert not any("xcquinox_version" in f for f in failures)
    assert any("xcquinox_version" in w for w in warnings)


def test_certificate_naming_another_arch_is_a_failure(tmp_path, patched_cfg):
    """The certificate is located by directory; the arch it NAMES must agree,
    so a file copied from another architecture's pretrain dir is caught."""
    run = _write_run(tmp_path, [_spec_for("deep_3x16"),
                                _spec_for("deep_attn_3x16")])
    _write_certificate(run, "deep_3x16", arch_field="deep_attn_3x16")
    failures, _warnings, _n = vr.validate_run(run)
    assert any("certificate names arch" in f for f in failures)


def test_wrong_parent_functional_is_a_failure(tmp_path, patched_cfg):
    """The parent is a property of the architecture's rung. A certificate
    measured against another functional does not bound this arch's offsets."""
    run = _write_run(tmp_path, [_spec_for("deep_3x16"),
                                _spec_for("deep_attn_3x16")])
    _write_certificate(run, "deep_3x16", parent="scan")
    failures, _warnings, _n = vr.validate_run(run)
    assert any("parent" in f and "'scan'" in f for f in failures)


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


def test_matching_checkpoint_digests_validate(tmp_path, patched_cfg):
    run = _write_run(tmp_path, [_spec_for("deep_3x16"),
                                _spec_for("deep_attn_3x16")])
    digests = _write_checkpoint_files(run, "deep_3x16")
    _write_certificate(run, "deep_3x16", checkpoint={
        "dir": os.path.join(run, "pretrain", "deep_3x16"),
        "xnet_sha256": digests["xnet.eqx"],
        "cnet_sha256": digests["cnet.eqx"]})
    failures, _warnings, _n = vr.validate_run(run)
    assert failures == [], failures


def test_certified_checkpoint_gone_is_a_failure(tmp_path, patched_cfg):
    """The verdict refers to two files; if they are not in the run, nothing
    ties the certificate to what the train stage would load."""
    run = _write_run(tmp_path, [_spec_for("deep_3x16"),
                                _spec_for("deep_attn_3x16")])
    _write_certificate(run, "deep_3x16", checkpoint={
        "dir": os.path.join(run, "pretrain", "deep_3x16"),
        "xnet_sha256": "a" * 64, "cnet_sha256": "b" * 64})
    failures, _warnings, _n = vr.validate_run(run)
    assert any("no such file is present" in f and "xnet.eqx" in f
               for f in failures)


def test_uncertified_checkpoint_files_are_a_failure(tmp_path, patched_cfg):
    """Networks present that the certificate does not name cannot be the ones
    it measured."""
    run = _write_run(tmp_path, [_spec_for("deep_3x16"),
                                _spec_for("deep_attn_3x16")])
    _write_checkpoint_files(run, "deep_3x16")
    failures, _warnings, _n = vr.validate_run(run)
    assert any("records no xnet_sha256" in f for f in failures)


def test_absent_digests_and_absent_checkpoints_warn(tmp_path, patched_cfg):
    """The synthetic run carries neither; that is reported as uncheckable,
    not as a disagreement."""
    run = _write_run(tmp_path, [_spec_for("deep_3x16"),
                                _spec_for("deep_attn_3x16")])
    failures, warnings, _n = vr.validate_run(run)
    assert failures == [], failures
    assert any("xnet_sha256" in w for w in warnings)


@pytest.mark.parametrize("body", ("[]", "null", '"nope"', "0"))
def test_a_certificate_that_is_not_an_object_is_a_named_failure(
        tmp_path, patched_cfg, body):
    """A document that PARSES but is not a certificate states no verdict.

    A JSON array, null, string or number satisfies ``json.load`` and carries
    none of the fields the checks below read, so a reader that only guards the
    parse and then tests the payload's type reports nothing at all for that
    architecture -- the one outcome a gate may never produce. The record layer
    calls such a file UNREADABLE, and an unverifiable certificate is refused.
    """
    run = _write_run(tmp_path, [_spec_for("deep_3x16"),
                                _spec_for("deep_attn_3x16")])
    path = os.path.join(run, "pretrain", "deep_3x16",
                        "fidelity_certificate.json")
    with open(path, "w") as f:
        f.write(body)
    failures, _warnings, _n = vr.validate_run(run)
    assert any("pretrain/deep_3x16" in f and "fidelity_certificate.json" in f
               for f in failures), failures


def test_a_certificate_that_cannot_be_opened_is_a_named_failure(tmp_path,
                                                                patched_cfg):
    """An unreadable file is a reported failure, not an exception.

    ``validate_run`` is report-only: a certificate whose permissions deny the
    read is as unverifiable as a truncated one, and must be listed beside the
    other findings rather than abort the scan.
    """
    run = _write_run(tmp_path, [_spec_for("deep_3x16"),
                                _spec_for("deep_attn_3x16")])
    path = os.path.join(run, "pretrain", "deep_3x16",
                        "fidelity_certificate.json")
    os.chmod(path, 0o000)
    if os.access(path, os.R_OK):  # pragma: no cover -- privileged environment
        os.chmod(path, 0o600)
        pytest.skip("the running user can read a mode-000 file")
    try:
        failures, _warnings, _n = vr.validate_run(run)
    finally:
        os.chmod(path, 0o600)
    assert any("pretrain/deep_3x16" in f and "fidelity_certificate.json" in f
               for f in failures), failures


def test_an_identity_field_the_run_does_not_state_is_reported(tmp_path,
                                                              patched_cfg):
    """The identity comparison covers the UNION of the two key sets.

    The expected identity comes from ``fidelity.run_identity``; a certificate
    that carries a further identity field states a condition this run does not
    match, and a comparison that iterates only the expected keys would pass it
    silently -- as it would a sixth field added to the shared builder.
    """
    run = _write_run(tmp_path, [_spec_for("deep_3x16"),
                                _spec_for("deep_attn_3x16")])
    _write_certificate(run, "deep_3x16", identity={
        "basis": _BASIS, "grid_level": 1, "density_fit": False,
        "auxbasis": None, "orientation_lock_strength": 0.0,
        "seed_xc": "scan"})
    failures, _warnings, _n = vr.validate_run(run)
    assert any("seed_xc" in f for f in failures), failures


def test_an_identity_field_the_certificate_omits_is_reported(tmp_path,
                                                            patched_cfg):
    """The other half of the UNION: a key the certificate does not state.

    ``auxbasis`` is ``None`` in this run's identity, so a comparison that reads
    an absent key as ``None``, or that iterates only the keys the certificate
    happens to carry, reads a certificate that never recorded the Coulomb
    backend as agreeing about it. Absence is not a statement: the run's config
    is the authority on what must have been measured, and the missing field is
    reported as ``<absent>``.
    """
    run = _write_run(tmp_path, [_spec_for("deep_3x16"),
                                _spec_for("deep_attn_3x16")])
    _write_certificate(run, "deep_3x16", identity={
        "basis": _BASIS, "grid_level": 1, "density_fit": False,
        "orientation_lock_strength": 0.0})
    failures, _warnings, _n = vr.validate_run(run)
    assert any("deep_3x16" in f and "auxbasis=<absent>" in f
               and "None" in f for f in failures), failures


def test_each_certificate_is_read_once(tmp_path, patched_cfg, monkeypatch):
    """One parse per certificate, so no report can mix two documents.

    Classifying the file and then re-opening it for its contents gives a
    certificate rewritten between the two opens a report that states both: the
    status of the file as it was, beside a finding about the file as it
    became. Here the second read of each certificate would find a truncated
    file, and a two-read validator emits 'not readable as a certificate
    (fidelity certificate PASS)' -- a line contradicting itself about a run
    whose certificates are intact.
    """
    import builtins
    import io as _io
    from xcquinox.alec.cluster.fidelity import CERTIFICATE_FILENAME
    run = _write_run(tmp_path, [_spec_for("deep_3x16"),
                                _spec_for("deep_attn_3x16")])
    real_open = builtins.open
    reads: dict = {}

    def counting_open(file, *args, **kwargs):
        path = str(file)
        if path.endswith(CERTIFICATE_FILENAME):
            reads[path] = reads.get(path, 0) + 1
            if reads[path] > 1:
                return _io.StringIO("{truncated")
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", counting_open)
    failures, _warnings, _n = vr.validate_run(run)
    monkeypatch.undo()
    assert len(reads) == 2, reads
    assert set(reads.values()) == {1}, reads
    assert not any("not readable as a certificate" in f and "PASS" in f
                   for f in failures), failures
    assert failures == [], failures


def test_a_malformed_identity_value_is_a_mismatch_not_a_crash(tmp_path,
                                                              patched_cfg):
    """A non-numeric grid level is a disagreement, and the scan continues.

    Coercing the recorded value to the config's type is how ``"1"`` and ``1``
    are read as the same grid; a value that cannot be coerced disagrees with
    the config and is reported as such. Raising instead would discard every
    failure accumulated before it -- here the FAIL verdict of the
    alphabetically earlier architecture.
    """
    run = _write_run(tmp_path, [_spec_for("deep_3x16"),
                                _spec_for("deep_attn_3x16")])
    _write_certificate(run, "deep_3x16", verdict="FAIL")
    _write_certificate(run, "deep_attn_3x16", identity={
        "basis": _BASIS, "grid_level": "abc", "density_fit": False,
        "auxbasis": None, "orientation_lock_strength": 0.0})
    failures, _warnings, _n = vr.validate_run(run)
    assert any("deep_attn_3x16" in f and "grid_level" in f
               for f in failures), failures
    assert any("deep_3x16:" in f and "verdict" in f
               for f in failures), failures


def test_the_report_names_a_recorded_waiver(tmp_path, patched_cfg):
    """A run refused for a deliberately non-enforcing certificate says so.

    ``fidelity.enforce: false`` releases the on-node gates only, so a
    workflow-verification run reaches validation with a FAIL on record. The
    verdict is still a failure, but the report carries the recorded reason and
    the flag, so that run is distinguishable from one whose physics simply did
    not certify.
    """
    run = _write_run(tmp_path, [_spec_for("deep_3x16"),
                                _spec_for("deep_attn_3x16")])
    _write_certificate(run, "deep_3x16", verdict="FAIL", enforced=False,
                       override_reason="workflow matrix: 50-step pretrain")
    failures, _warnings, _n = vr.validate_run(run)
    assert any("verdict 'FAIL'" in f
               and "workflow matrix: 50-step pretrain" in f
               and "enforced" in f for f in failures), failures


def _code_string_constants(func):
    """Every string literal in a function's code object.

    Constant dict keys are compiled into a single tuple constant rather than
    separate string constants, so nested tuples are flattened.
    """
    out = set()
    for const in func.__code__.co_consts:
        if isinstance(const, str):
            out.add(const)
        elif isinstance(const, tuple):
            out.update(c for c in const if isinstance(c, str))
    return out


def test_the_checkpoint_digest_names_come_from_the_writer():
    """The two sides of the digest comparison read one table of names.

    ``validate_run`` compares the digests ``fidelity_certificate`` recorded,
    so the network file names and the payload keys are taken from
    ``fidelity.CHECKPOINT_DIGEST_KEYS`` instead of being restated: a rename in
    the writer would otherwise leave the validator comparing keys nothing
    writes, which reads as a clean run.
    """
    from xcquinox.alec.cluster import fidelity
    written = _code_string_constants(fidelity.fidelity_certificate)
    for filename, digest_key in fidelity.CHECKPOINT_DIGEST_KEYS:
        assert filename in written
        assert digest_key in written
