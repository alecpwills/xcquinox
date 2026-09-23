"""Tests for xcquinox.pipeline.cluster.fidelity -- the per-architecture physics
certificate.

The cheap layer (the certificate predicate, the parent resolution, the oracle
set) is tested directly. The certificate itself is tested three ways: with the
per-system evaluation monkeypatched at the ``evaluate`` seam, so the verdict
arithmetic and the JSON schema are exercised with no SCF at all; for REAL on H
and H2 at sto-3g with networks built in the test, so the energy path, the three
parent routes and the atomization fold are pinned against physics; and with the
exact parent functional (PBE, SCAN through libxc) presented behind the model
interface on O, H and H2O, so the whole path is shown to be an identity when the
network is its parent and to report a known per-electron offset exactly.
"""
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xcquinox.pipeline.cluster import fidelity as fid
from xcquinox.pipeline.pyscf_determinism import pin_small_rho_cutoff


# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

def _cfg(arch=("deep_3x16",), basis="sto-3g", grid_level=1,
         tol_AE=1.0, tol_atom=1.0, override_reason=None, enforce=True,
         polarized=False, pretrain_seed=42):
    """The attribute surface fidelity reads off a GridConfig."""
    return SimpleNamespace(
        sweep=SimpleNamespace(arch=tuple(arch)),
        inputs=SimpleNamespace(basis=basis, grid_level=grid_level,
                               density_fit=False, auxbasis=None,
                               orientation_lock_strength=0.0),
        pretrain=SimpleNamespace(seed=pretrain_seed),
        fidelity=SimpleNamespace(tol_AE=tol_AE, tol_atom=tol_atom,
                                 override_reason=override_reason,
                                 enforce=enforce),
        use_polarized_correlation=polarized,
    )


def _write_certificate(run_dir, arch, verdict="PASS", **extra):
    """Write a certificate; ``extra`` sets any schema key (``enforced``,
    ``summary``, ``tolerances``, ...)."""
    d = os.path.join(run_dir, "pretrain", arch)
    os.makedirs(d, exist_ok=True)
    payload = {"verdict": verdict, "arch": arch}
    payload.update(extra)
    with open(os.path.join(d, fid.CERTIFICATE_FILENAME), "w") as f:
        json.dump(payload, f)
    return d


# ---------------------------------------------------------------------------
# Import weight: the module body stays a pure reader
# ---------------------------------------------------------------------------

# The stacks a certificate reader must never pull in, and the only xcquinox
# modules cheap enough for the module body: grid_config supplies the run
# layout, domain the kcal/mol conversion, materialize the atomic JSON writer
# the certificate writer uses. Each is stdlib-only in its own body.
_HEAVY_IMPORT_ROOTS = frozenset({
    "jax", "jaxlib", "equinox", "optax", "numpy", "scipy", "pyscf", "pyscfad",
    "ase", "pandas", "matplotlib", "torch", "h5py",
})
_CHEAP_XCQ_MODULES = frozenset({
    "xcquinox.pipeline.cluster.grid_config",
    "xcquinox.pipeline.cluster.domain",
    "xcquinox.pipeline.cluster.materialize",
    # The calibrated orientation-lock strength, and nothing else: the harness
    # parser's default reads it in its module body, and it must not drag
    # ``orientation_lock``'s numpy in behind it.
    "xcquinox.pipeline.orientation_lock_default",
    # The shared hard exit, reached only from this module's
    # ``if __name__ == "__main__"`` block, which an import never runs. It is
    # stdlib-only by construction, and whitelisting it rather than exempting
    # the block keeps the walk transitive: a heavy import added to the helper
    # would still be caught here.
    "xcquinox.pipeline.cluster._exit",
})

#: Whitelisted modules the SOURCE walk sees but an import never binds, because
#: the only statement importing them is the ``if __name__ == "__main__"``
#: block. The two tests below therefore treat them oppositely on purpose: the
#: walk accepts the name and recurses into it, and the closure test requires
#: the name to be ABSENT from ``sys.modules`` after the body has run -- which
#: is the measurement that the entry block really is off the import path.
_ENTRY_BLOCK_ONLY_MODULES = frozenset({"xcquinox.pipeline.cluster._exit"})

# Upper bound on the modules present in sys.modules after the file is executed
# with the package __init__ modules stubbed (the closure test below). The
# committed tree measures 123, of which 78 are the interpreter's own startup
# set. A module-body ``import pyscf`` measures 799 (numpy, scipy, pyscf) and an
# ``import jax`` in grid_config's body measures 612 (numpy, jax, jaxlib), so any
# bound between 123 and 612 discriminates; 300 leaves the readers room to grow a
# stdlib import without a test edit.
_CLOSURE_MODULE_BOUND = 300


def test_fidelity_imports_in_a_fresh_interpreter():
    """No import cycle: _pretrain imports fidelity, fidelity must not import
    _pretrain (it derives the distinct-arch list from _canon_axis instead)."""
    out = subprocess.run(
        [sys.executable, "-c", "import xcquinox.pipeline.cluster.fidelity"],
        capture_output=True, text=True)
    assert out.returncode == 0, out.stderr


# ---------------------------------------------------------------------------
# The one predicate every enforcement site calls
# ---------------------------------------------------------------------------

def test_certificate_status_missing(tmp_path):
    status, reason = fid.certificate_status_in(str(tmp_path))
    assert status == "MISSING"
    assert fid.CERTIFICATE_FILENAME in reason


def test_certificate_status_pass(tmp_path):
    d = _write_certificate(str(tmp_path), "deep_3x16", verdict="PASS")
    assert fid.certificate_status_in(d) == ("PASS", "fidelity certificate PASS")


def test_certificate_status_fail_carries_the_summary(tmp_path):
    d = _write_certificate(str(tmp_path), "deep_3x16", verdict="FAIL",
                           summary={"max_atom_mHa": 13.7,
                                    "max_dAE_kcalmol": 25.7})
    status, reason = fid.certificate_status_in(d)
    assert status == "FAIL"
    assert "13.7" in reason and "25.7" in reason


def test_certificate_status_unreadable(tmp_path):
    d = tmp_path / "pretrain" / "deep_3x16"
    d.mkdir(parents=True)
    (d / fid.CERTIFICATE_FILENAME).write_text("{not json")
    status, reason = fid.certificate_status_in(str(d))
    assert status == "UNREADABLE"
    assert "JSON" in reason


_UNACTIONABLE_VERDICTS = [
    pytest.param({"arch": "deep_3x16"}, id="no-verdict-key"),
    pytest.param({"arch": "deep_3x16", "verdict": "pass"}, id="wrong-case"),
    pytest.param({"arch": "deep_3x16", "verdict": None}, id="null"),
    pytest.param({"arch": "deep_3x16", "verdict": 1}, id="integer"),
]


# ---------------------------------------------------------------------------
# The ON-NODE gate: fidelity.enforce = False records the verdict and continues
# ---------------------------------------------------------------------------


def test_gate_allows_a_passing_certificate(tmp_path):
    _write_certificate(str(tmp_path), "deep_3x16", verdict="PASS")
    allowed, message = fid.gate_certificate(str(tmp_path), "deep_3x16")
    assert allowed is True
    assert "PASS" in message


def test_gate_refuses_an_enforced_failure(tmp_path):
    _write_certificate(str(tmp_path), "deep_3x16", verdict="FAIL",
                       enforced=True,
                       summary={"max_atom_mHa": 13.7,
                                "max_dAE_kcalmol": 25.7})
    allowed, message = fid.gate_certificate(str(tmp_path), "deep_3x16")
    assert allowed is False
    assert "13.7" in message


def test_gate_allows_a_recorded_failure_when_enforcement_is_off(tmp_path):
    """The Section 3.4 workflow matrix: a 50-step pretrain cannot meet the
    tolerance, but train and eval must still be exercised end to end with the
    real verdict on the record."""
    _write_certificate(str(tmp_path), "deep_3x16", verdict="FAIL",
                       enforced=False,
                       tolerances={"tol_AE": 1.0, "tol_atom": 1.0,
                                   "override_reason": "workflow matrix"},
                       summary={"max_atom_mHa": 13.7,
                                "max_dAE_kcalmol": 25.7})
    allowed, message = fid.gate_certificate(str(tmp_path), "deep_3x16")
    assert allowed is True
    assert "enforcement is OFF" in message
    assert "workflow matrix" in message


@pytest.mark.parametrize("tolerances", [
    pytest.param(None, id="no-tolerances-block"),
    pytest.param({"tol_AE": 1.0, "tol_atom": 1.0}, id="no-reason-field"),
    pytest.param({"override_reason": None}, id="null"),
    pytest.param({"override_reason": ""}, id="empty"),
    pytest.param({"override_reason": "   "}, id="whitespace"),
])
def test_the_gate_refuses_a_waiver_that_records_no_reason(tmp_path,
                                                          tolerances):
    """``validate_grid_semantics`` refuses ``fidelity.enforce: false`` without
    a non-empty ``fidelity.override_reason``. The on-node gate imposes the
    same invariant on the certificate itself, so a hand-edited certificate or
    resolved_config.yaml on a node cannot release a FAIL with no reason on the
    record."""
    extra = {} if tolerances is None else {"tolerances": tolerances}
    _write_certificate(str(tmp_path), "deep_3x16", verdict="FAIL",
                       enforced=False,
                       summary={"max_atom_mHa": 13.7,
                                "max_dAE_kcalmol": 25.7},
                       **extra)
    allowed, message = fid.gate_certificate(str(tmp_path), "deep_3x16")
    assert allowed is False, tolerances
    assert "override_reason" in message


def test_gate_never_allows_a_missing_certificate(tmp_path):
    """Enforcement can only be waived by a certificate that exists to record
    the waiver; an absent one waives nothing."""
    allowed, message = fid.gate_certificate(str(tmp_path), "deep_3x16")
    assert allowed is False
    assert "MISSING" in message or "was never checked" in message


# ---------------------------------------------------------------------------
# One document per decision
# ---------------------------------------------------------------------------

def _serve_documents(monkeypatch, path, documents):
    """Serve ``documents`` to successive READ opens of ``path``.

    The list returned collects one entry per read served, so a caller can
    state how many parses a decision rested on. Writes and every other path
    are passed through; once the list is exhausted its last entry repeats, so
    a caller that reads more often than the sequence is long is handed a
    complete document rather than an empty file.
    """
    import builtins
    import io
    real_open = builtins.open
    served: list = []

    def fake_open(file, *args, **kwargs):
        mode = kwargs.get("mode", args[0] if args else "r")
        if str(file) == str(path) and "r" in mode:
            doc = documents[min(len(served), len(documents) - 1)]
            served.append(doc)
            return io.StringIO(doc if isinstance(doc, str)
                               else json.dumps(doc))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open)
    return served


# Three FAIL documents, each refused on its own: D1 records no waiver, D2
# records one that states no reason, D3 states a reason beside enforcement
# that is ON. A gate that classifies one document and reads the waiver off
# another releases on D1 -> D2 -> D3, which is a release granted to a
# certificate that never existed.
_D1 = {"verdict": "FAIL",
       "summary": {"max_atom_mHa": 13.7, "max_dAE_kcalmol": 25.7}}
_D2 = {"verdict": "FAIL", "enforced": False}
_D3 = {"verdict": "FAIL", "enforced": True,
       "tolerances": {"override_reason": "workflow matrix"}}


# ---------------------------------------------------------------------------
# Parent resolution: the arch's RUNG picks the parent, not inputs.seed_xc
# ---------------------------------------------------------------------------

def test_parent_is_pbe_for_gga_rung_and_scan_for_meta_gga():
    assert fid.resolve_parent("deep_3x16") == "pbe"
    assert fid.resolve_parent("deep_cusp_3x16") == "pbe"
    assert fid.resolve_parent("deep_rung35_3x16") == "pbe"
    assert fid.resolve_parent("deep_mgga_3x16") == "scan"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Run identity
# ---------------------------------------------------------------------------

def test_run_identity_carries_the_five_scf_identity_fields():
    cfg = _cfg(basis="6-311++G(3df,2pd)", grid_level=3)
    cfg.inputs.density_fit = True
    cfg.inputs.auxbasis = "def2-universal-jkfit"
    cfg.inputs.orientation_lock_strength = 0.02
    assert fid.run_identity(cfg) == {
        "basis": "6-311++G(3df,2pd)", "grid_level": 3, "density_fit": True,
        "auxbasis": "def2-universal-jkfit",
        "orientation_lock_strength": 0.02}


# ---------------------------------------------------------------------------
# Oracle set
# ---------------------------------------------------------------------------


def test_oracle_set_carries_every_pool_free_atom_with_its_pool_spin():
    from xcquinox.pipeline.full_benchmark_pools import load_full_held_out_pools
    pool, _ = load_full_held_out_pools(basis="sto-3g", grid_level=1)
    systems = {ms.name: ms for ms in fid.build_oracle_set(_cfg(), "deep_3x16")}
    seen = 0
    for ms in pool.values():
        comp = tuple(ms.atom_composition)
        if len(comp) != 1 or int(comp[0][1]) != 1:
            continue
        seen += 1
        name = fid.atom_system_name(comp[0][0], ms.charge)
        assert name in systems, name
        assert systems[name].spin == ms.spin
        assert systems[name].charge == ms.charge
    assert seen >= 14


def test_oracle_set_supplies_a_free_atom_for_every_element_it_dissociates():
    systems = fid.build_oracle_set(_cfg(), "deep_3x16")
    names = {ms.name for ms in systems}
    for ms in systems:
        if fid.is_atom_system(ms):
            continue
        for sym, _n in ms.atom_composition:
            assert fid.atom_system_name(sym, 0) in names, (ms.name, sym)


def test_ground_state_spin_table_agrees_with_the_pool_spins():
    """The certificate's Hund ground-state table is the atomization reference;
    it must agree species by species with the spins the BH76 / W4-11 pools
    carry, or a molecule would be folded against a different atom than the
    benchmark uses."""
    from xcquinox.pipeline.full_benchmark_pools import load_full_held_out_pools
    pool, _ = load_full_held_out_pools(basis="sto-3g", grid_level=1)
    for ms in pool.values():
        comp = tuple(ms.atom_composition)
        if len(comp) != 1 or int(comp[0][1]) != 1 or ms.charge != 0:
            continue
        sym = comp[0][0]
        assert fid._ATOM_GROUND_SPIN[sym] == ms.spin, sym


# ---------------------------------------------------------------------------
# The three unconditional molecules are ONE molecule for every rung
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Anti-fork guard: no second construction of a precompute quantity
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Model construction and the parent-density request
# ---------------------------------------------------------------------------

def test_build_certified_model_loads_the_checkpoint_not_the_skeleton(tmp_path):
    """The skeleton's seed fixes the tree SHAPE only; every array leaf comes
    from the checkpoint. A builder that returned the skeleton would certify a
    randomly initialised network -- exactly the state the gate exists to
    catch."""
    import equinox as eqx
    import jax
    import jax.numpy as jnp
    from xcquinox.pipeline.config import get_architecture
    from xcquinox.pipeline.networks import create_network_pair

    run_dir = str(tmp_path / "run")
    _stub_checkpoint(run_dir, "deep_3x16", seed=7)
    arch = get_architecture("deep_3x16")
    _built, model = fid.build_certified_model(_cfg(pretrain_seed=99), run_dir,
                                              "deep_3x16")
    from_checkpoint, _ = create_network_pair(arch, seed=7)
    from_skeleton, _ = create_network_pair(arch, seed=99)
    got = jax.tree_util.tree_leaves(eqx.filter(model.xnet, eqx.is_array))
    want = jax.tree_util.tree_leaves(eqx.filter(from_checkpoint, eqx.is_array))
    other = jax.tree_util.tree_leaves(eqx.filter(from_skeleton, eqx.is_array))
    assert len(got) == len(want) == len(other)
    assert all(bool(jnp.allclose(a, b)) for a, b in zip(got, want))
    # The two seeds really do differ, so the assertion above has content.
    assert any(not bool(jnp.allclose(a, b)) for a, b in zip(want, other))


def test_evaluate_system_refuses_a_record_built_on_another_functional(
        monkeypatch):
    """A record whose reference_xc is not the parent would measure the network
    against the wrong density; that raises rather than entering the table."""
    import xcquinox.pipeline.data as data_mod
    from xcquinox.pipeline.config import get_architecture
    from xcquinox.pipeline.models import AlecGGAModel

    original = data_mod.precompute_fixed_density_data

    def _mislabel(mol_spec, **kwargs):
        md = dict(original(mol_spec, **kwargs))
        md["reference_xc"] = "lda,vwn"
        return md

    monkeypatch.setattr(data_mod, "precompute_fixed_density_data", _mislabel)
    arch = get_architecture("deep_3x16")
    model = AlecGGAModel.from_arch(arch, seed=0)
    with pytest.raises(ValueError, match="reference_xc"):
        fid.evaluate_system(model, arch.materialize_descriptors(),
                            _tiny_oracle_set()[0], parent="pbe")


def test_meta_gga_architecture_is_certified_against_scan(tmp_path, monkeypatch):
    """End to end for the rung that motivated reference_xc: a meta-GGA
    architecture's certificate must be computed against SCAN, on SCAN's own
    density."""
    import xcquinox.pipeline.data as data_mod
    seen = []
    original = data_mod.precompute_fixed_density_data

    def _spy(mol_spec, **kwargs):
        seen.append(kwargs.get("reference_xc"))
        return original(mol_spec, **kwargs)

    monkeypatch.setattr(data_mod, "precompute_fixed_density_data", _spy)
    run_dir = str(tmp_path / "run")
    _stub_checkpoint(run_dir, "deep_mgga_3x16", seed=0)
    payload = fid.fidelity_certificate(
        _cfg(arch=("deep_mgga_3x16",), pretrain_seed=0), run_dir,
        "deep_mgga_3x16", oracle_set=_tiny_oracle_set())
    assert payload["parent"] == "scan"
    assert set(seen) == {"scan"}
    assert all(r["reference_xc"] == "scan" for r in payload["per_system"])


# ---------------------------------------------------------------------------
# The certificate, with the per-system evaluation mocked at the seam
# ---------------------------------------------------------------------------

def _fake_evaluate(table):
    """Build an ``evaluate`` seam returning canned dE_xc (mHa) per name."""
    def _evaluate(model, descriptors, mol_spec, *, parent,
                  auxbasis=None, orientation_lock_strength=0.0):
        d = table[mol_spec.name]
        return {"name": mol_spec.name, "spin": int(mol_spec.spin),
                "charge": int(mol_spec.charge),
                "is_atom": fid.is_atom_system(mol_spec),
                "n_grid": 10, "reference_xc": parent,
                "E_xc_nn": -1.0 + d / fid.HA_TO_MHA, "E_xc_parent": -1.0,
                "E_xc_parent_numint": -1.0, "E_xc_parent_record": -1.0,
                "parent_grid_diff_Ha": 0.0, "parent_record_diff_Ha": 0.0,
                "dE_xc_mHa": d, "duration_s": 0.0}
    return _evaluate


def _tiny_oracle_set(basis="sto-3g", grid_level=1):
    from xcquinox.pipeline.config import MoleculeSpec
    return (
        MoleculeSpec(name="atom_H", atom="H 0.0 0.0 0.0", basis=basis, spin=1,
                     atom_composition=(("H", 1),), grid_level=grid_level),
        MoleculeSpec(name="H2", atom="H 0 0 0.371395; H 0 0 -0.371395",
                     basis=basis, spin=0, atom_composition=(("H", 2),),
                     grid_level=grid_level),
    )


def _stub_checkpoint(run_dir, arch_name="deep_3x16", seed=42):
    """Write a real xnet.eqx + cnet.eqx pair for ``arch_name``."""
    import equinox as eqx
    from xcquinox.pipeline.config import get_architecture
    from xcquinox.pipeline.networks import create_network_pair
    from xcquinox.pipeline.cluster.grid_config import pretrain_checkpoint_dir
    arch = get_architecture(arch_name)
    xnet, cnet = create_network_pair(arch, seed=seed)
    d = pretrain_checkpoint_dir(run_dir, arch_name)
    os.makedirs(d, exist_ok=True)
    eqx.tree_serialise_leaves(os.path.join(d, "xnet.eqx"), xnet)
    eqx.tree_serialise_leaves(os.path.join(d, "cnet.eqx"), cnet)
    return d


def test_certificate_passes_within_tolerance_and_writes_the_schema(tmp_path):
    run_dir = str(tmp_path / "run")
    _stub_checkpoint(run_dir)
    cfg = _cfg()
    payload = fid.fidelity_certificate(
        cfg, run_dir, "deep_3x16",
        oracle_set=_tiny_oracle_set(),
        evaluate=_fake_evaluate({"atom_H": 0.5, "H2": 1.0}))

    assert payload["verdict"] == "PASS"
    assert payload["arch"] == "deep_3x16"
    assert payload["parent"] == "pbe"
    assert payload["identity"] == fid.run_identity(cfg)
    assert payload["tolerances"] == {"tol_AE": 1.0, "tol_atom": 1.0,
                                     "tol_AE_aggregate": "max",
                                     "tol_AE_max_backstop": None,
                                     "override_reason": None}
    assert payload["enforced"] is True
    assert isinstance(payload["xcquinox_version"], str)
    assert payload["timestamp"].endswith("Z")
    assert payload["duration_s"] >= 0.0
    assert [r["name"] for r in payload["per_system"]] == ["atom_H", "H2"]
    assert [r["name"] for r in payload["per_atomization"]] == ["H2"]
    s = payload["summary"]
    assert s["n_systems"] == 2 and s["n_atoms"] == 1
    assert s["n_atomizations"] == 1 and s["n_failed_systems"] == 0
    assert s["max_parent_grid_diff_Ha"] == pytest.approx(0.0)
    assert s["max_parent_record_diff_Ha"] == pytest.approx(0.0)
    assert s["max_atom_mHa"] == pytest.approx(0.5)
    # dAE = dE_xc(H2) - 2 dE_xc(H) = 1.0 - 1.0 = 0 mHa.
    assert s["max_dAE_kcalmol"] == pytest.approx(0.0, abs=1e-12)
    assert s["failure_reasons"] == []

    on_disk = json.loads(
        open(fid.certificate_path(run_dir, "deep_3x16")).read())
    assert on_disk == payload
    assert fid.certificate_status(run_dir, "deep_3x16")[0] == "PASS"


def test_certificate_fails_on_the_atom_tolerance(tmp_path):
    run_dir = str(tmp_path / "run")
    _stub_checkpoint(run_dir)
    payload = fid.fidelity_certificate(
        _cfg(), run_dir, "deep_3x16", oracle_set=_tiny_oracle_set(),
        evaluate=_fake_evaluate({"atom_H": 13.7, "H2": 27.4}))
    assert payload["verdict"] == "FAIL"
    assert payload["summary"]["max_atom_mHa"] == pytest.approx(13.7)
    assert any("tol_atom" in r for r in payload["summary"]["failure_reasons"])
    assert fid.certificate_status(run_dir, "deep_3x16")[0] == "FAIL"


def test_certificate_honours_configured_tolerances(tmp_path):
    run_dir = str(tmp_path / "run")
    _stub_checkpoint(run_dir)
    cfg = _cfg(tol_AE=2.0, tol_atom=2.0,
               override_reason=None)
    payload = fid.fidelity_certificate(
        cfg, run_dir, "deep_3x16", oracle_set=_tiny_oracle_set(),
        evaluate=_fake_evaluate({"atom_H": 1.5, "H2": 3.0}))
    assert payload["verdict"] == "PASS"
    assert payload["tolerances"]["tol_atom"] == 2.0


def test_certificate_records_the_checkpoint_digests(tmp_path):
    """The certificate names the exact networks it measured: the SHA-256 of
    xnet.eqx and cnet.eqx, so a checkpoint rewritten after certification can
    be told apart from the one the verdict refers to."""
    from xcquinox.pipeline.cluster.materialize import _sha256_file
    run_dir = str(tmp_path / "run")
    d = _stub_checkpoint(run_dir)
    payload = fid.fidelity_certificate(
        _cfg(), run_dir, "deep_3x16", oracle_set=_tiny_oracle_set(),
        evaluate=_fake_evaluate({"atom_H": 0.5, "H2": 1.0}))
    assert payload["checkpoint"] == {
        "dir": d,
        "xnet_sha256": _sha256_file(os.path.join(d, "xnet.eqx")),
        "cnet_sha256": _sha256_file(os.path.join(d, "cnet.eqx")),
    }
    assert len(payload["checkpoint"]["xnet_sha256"]) == 64


# ---------------------------------------------------------------------------
# The descriptor log transform: recorded, and compared where it is stated
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# REAL physics: H and H2 at sto-3g, networks built in the test (seconds)
# ---------------------------------------------------------------------------

def test_certificate_real_physics_on_h_and_h2_at_sto3g(tmp_path):
    """The whole energy path, for real, on two tiny systems.

    ``deep_3x16`` is built with ``zero_init_final_layer=True``, so a freshly
    seeded network has Fx = Fc = 1 exactly and its E_xc is the LDA exchange
    plus PW92 correlation. Against PBE on the same frozen PBE density that is
    a large, definite offset, so this pins the sign, the magnitude, the
    atomization fold and the FAIL branch at once. Every number the certificate
    reports is re-derived in the test from an independent PySCF route.
    """
    from pyscf import dft, gto
    from pyscf.dft import numint

    run_dir = str(tmp_path / "run")
    _stub_checkpoint(run_dir, "deep_3x16", seed=0)
    cfg = _cfg(pretrain_seed=0)
    systems = _tiny_oracle_set()

    payload = fid.fidelity_certificate(cfg, run_dir, "deep_3x16",
                                       oracle_set=systems)

    by_name = {r["name"]: r for r in payload["per_system"]}
    assert set(by_name) == {"atom_H", "H2"}
    assert by_name["atom_H"]["is_atom"] is True
    assert by_name["H2"]["is_atom"] is False
    assert by_name["atom_H"]["spin"] == 1 and by_name["H2"]["spin"] == 0

    # (1) Every record was built on the PARENT's own self-consistent density,
    #     and the parent energy is that functional on that density, on the
    #     SAME grid PySCF's own nr_rks / nr_uks uses.
    assert all(r["reference_xc"] == "pbe" for r in by_name.values())
    for ms in systems:
        rec = by_name[ms.name]
        mol = gto.M(atom=ms.atom, basis=ms.basis, charge=ms.charge,
                    spin=ms.spin, verbose=0)
        mf = dft.UKS(mol) if ms.spin else dft.RKS(mol)
        mf.xc = "pbe"
        mf.grids.level = ms.grid_level
        # The oracle's SCF prunes its grid at the density cutoff the record's
        # reference SCF pins (pyscf 2.14's class default keeps every point),
        # so its density is the record's; the energy below is then taken on
        # the full Becke-Lebedev grid built explicitly.
        pin_small_rho_cutoff(mf)
        mf.kernel()
        grids = dft.Grids(mol)
        grids.level = ms.grid_level
        grids.build()
        ni = numint.NumInt()
        dm = mf.make_rdm1()
        if ms.spin:
            _v, exc, _ = ni.nr_uks(mol, grids, "PBE", dm)
        else:
            _v, exc, _ = ni.nr_rks(mol, grids, "PBE", dm)
        assert rec["E_xc_parent"] == pytest.approx(float(exc), abs=1e-8)
        assert rec["E_xc_parent_numint"] == pytest.approx(float(exc), abs=1e-8)
        assert abs(rec["parent_grid_diff_Ha"]) < fid.PARENT_GRID_TOL_HA
        # Third independent route: the XC energy PySCF accumulated during the
        # reference SCF itself, carried on the record as E_xc_pbe.
        assert abs(rec["parent_record_diff_Ha"]) < fid.PARENT_GRID_TOL_HA

    # (2) dE_xc is exactly the difference the record carries, in mHa.
    for rec in by_name.values():
        assert rec["dE_xc_mHa"] == pytest.approx(
            (rec["E_xc_nn"] - rec["E_xc_parent"]) * fid.HA_TO_MHA, rel=1e-12)

    # (3) The atomization offset is the molecule minus its atoms, in kcal/mol.
    dae = {r["name"]: r["dAE_kcalmol"] for r in payload["per_atomization"]}
    expected = ((by_name["H2"]["dE_xc_mHa"] - 2 * by_name["atom_H"]["dE_xc_mHa"])
                / fid.HA_TO_MHA * fid.HA_TO_KCAL)
    assert dae["H2"] == pytest.approx(expected, rel=1e-12)

    # (4) An LDA-limit network is nowhere near PBE, so the certificate FAILS
    #     at the binding 1.0 mHa / 1.0 kcal/mol tolerances.
    assert payload["verdict"] == "FAIL"
    assert payload["summary"]["max_atom_mHa"] > 1.0
    assert abs(dae["H2"]) > 1.0
    assert fid.certificate_status(run_dir, "deep_3x16")[0] == "FAIL"


# ---------------------------------------------------------------------------
# The parent functional presented as the model: exactness of the energy path
# ---------------------------------------------------------------------------

class _ParentCNet(eqx.Module):
    """The one cnet attribute the UKS energy path reads."""
    use_spin_polarization: bool = eqx.field(static=True)


class _LibxcParentModel(eqx.Module):
    """The exact parent functional behind the model interface.

    ``fixed_density_total_energy`` reads ``descriptors``,
    ``cnet.use_spin_polarization``, ``eval_exc`` (closed shell) and
    ``eval_ex`` / ``eval_ec`` (open shell: exchange spin-scaled on the doubled
    channel densities, correlation on the total density with the production
    ``uks_zeta``). Each evaluation hands libxc exactly the rows that interface
    carries -- the density, the gradient invariant, the descriptor block and,
    through a polarization-aware cnet, zeta -- so the number that comes back
    is the parent's own energy density on the same points the network is
    integrated on. libxc runs inside ``jax.pure_callback`` because
    ``compute_exc_nn`` is jitted.

    GGA parent (``alpha_column < 0``): rows (rho, |grad rho|, 0, 0). PBE
    correlation depends on the spin densities and the TOTAL gradient only
    (Perdew, Burke, Ernzerhof, PRL 77, 3865 (1996), Eq. 7-8), so the
    per-spin gradient split proportional to the spin densities is exact.

    Meta-GGA parent (``alpha_column >= 0``): the interface carries no tau,
    only the descriptor's clamped iso-orbital alpha, so tau is recovered as
    ``alpha tau_unif + sigma / (8 rho)`` (``metagga.compute_alpha`` inverted;
    exact wherever the [0, 100] clamp was inactive). The doubled exchange
    channels invert to ``2 tau_sigma`` directly. SCAN's correlation depends on
    the total density, total gradient, total tau and zeta only (Sun,
    Ruzsinszky, Perdew, PRL 115, 036402 (2015): alpha with d_s(zeta)), so the
    proportional per-spin split of tau is exact on physical data -- measured
    2.1e-18 Ha on the O-atom SCAN record against the true per-spin rows.

    ``offset_per_electron`` adds a constant ``c`` to the energy per electron,
    ``E_xc -> E_xc + c N``: the known offset the O-B oracle measures.
    """
    xc: str = eqx.field(static=True)
    descriptors: tuple = eqx.field(static=True)
    polarized: bool = eqx.field(static=True)
    offset_per_electron: float = eqx.field(static=True)
    alpha_column: int = eqx.field(static=True, default=-1)

    @property
    def cnet(self):
        return _ParentCNet(use_spin_polarization=self.polarized)

    def _tau(self, rho, sigma, alpha):
        rho_safe = np.maximum(rho, 1e-30)
        tau_unif = (0.3 * (3.0 * np.pi ** 2) ** (2.0 / 3.0)
                    * rho_safe ** (5.0 / 3.0))
        return alpha * tau_unif + sigma / (8.0 * rho_safe)

    def _rows(self, rho, sigma, alpha):
        zero = np.zeros_like(rho)
        rows = [rho, np.sqrt(np.maximum(sigma, 0.0)), zero, zero]
        if self.alpha_column >= 0:
            rows.append(self._tau(rho, sigma, alpha))
        return np.vstack(rows)

    def _x(self, rho, sigma, alpha):
        from pyscf.dft import numint
        exc = numint.NumInt().eval_xc(f"{self.xc},",
                                      self._rows(rho, sigma, alpha), spin=0)[0]
        return rho * exc

    def _c(self, rho, sigma, alpha, zeta):
        from pyscf.dft import numint
        ni = numint.NumInt()
        if self.polarized:
            g = np.sqrt(np.maximum(sigma, 0.0))
            zero = np.zeros_like(rho)
            wa, wb = 0.5 * (1.0 + zeta), 0.5 * (1.0 - zeta)
            rows_a = [rho * wa, g * wa, zero, zero]
            rows_b = [rho * wb, g * wb, zero, zero]
            if self.alpha_column >= 0:
                tau = self._tau(rho, sigma, alpha)
                rows_a.append(tau * wa)
                rows_b.append(tau * wb)
            exc = ni.eval_xc(f",{self.xc}",
                             (np.vstack(rows_a), np.vstack(rows_b)), spin=1)[0]
        else:
            exc = ni.eval_xc(f",{self.xc}", self._rows(rho, sigma, alpha),
                             spin=0)[0]
        return rho * (exc + self.offset_per_electron)

    def _alpha(self, rho, features):
        if self.alpha_column >= 0:
            return features[:, self.alpha_column]
        return jnp.zeros_like(rho)

    def _callback(self, fn, rho, *args):
        out = jax.ShapeDtypeStruct(rho.shape, rho.dtype)
        return jax.pure_callback(fn, out, rho, *args)

    def eval_ex(self, rho, sigma, features):
        return self._callback(lambda r, s, a: np.asarray(self._x(r, s, a)),
                              rho, sigma, self._alpha(rho, features))

    def eval_ec(self, rho, sigma, features, zeta=0.0):
        zeta = jnp.broadcast_to(jnp.asarray(zeta, dtype=rho.dtype), rho.shape)
        return self._callback(
            lambda r, s, a, z: np.asarray(self._c(r, s, a, z)),
            rho, sigma, self._alpha(rho, features), zeta)

    def eval_exc(self, rho, sigma, features, zeta=0.0):
        return self.eval_ex(rho, sigma, features) + self.eval_ec(
            rho, sigma, features, zeta=zeta)


def _parent_oracle_set(basis="sto-3g", grid_level=1):
    """O, H and H2O: the two free atoms H2O dissociates into, so the
    atomization fold is exercised on a real molecule."""
    from xcquinox.pipeline.config import MoleculeSpec
    return (
        MoleculeSpec(name="atom_H", atom="H 0.0 0.0 0.0", basis=basis, spin=1,
                     atom_composition=(("H", 1),), grid_level=grid_level),
        MoleculeSpec(name="atom_O", atom="O 0.0 0.0 0.0", basis=basis, spin=2,
                     atom_composition=(("O", 1),), grid_level=grid_level),
        MoleculeSpec(name="H2O",
                     atom="O 0.0 0.0 0.1173; H 0.0 0.7572 -0.4692; "
                          "H 0.0 -0.7572 -0.4692",
                     basis=basis, spin=0,
                     atom_composition=(("H", 2), ("O", 1)),
                     grid_level=grid_level),
    )


# (arch, libxc parent code, alpha column of the descriptor block)
_PARENT_DOUBLES = {
    "pbe": ("deep_3x16", "PBE", -1),
    "scan": ("deep_mgga_3x16", "SCAN", 0),
}


def _certify_with_parent_as_model(tmp_path, monkeypatch, parent, offset):
    from xcquinox.pipeline.config import get_architecture
    arch_name, xc, alpha_column = _PARENT_DOUBLES[parent]
    run_dir = str(tmp_path / f"run_{parent}_{offset:g}")
    _stub_checkpoint(run_dir, arch_name, seed=0)
    arch = get_architecture(arch_name)
    double = _LibxcParentModel(xc=xc, descriptors=arch.materialize_descriptors(),
                               polarized=True, offset_per_electron=offset,
                               alpha_column=alpha_column)
    monkeypatch.setattr(fid, "build_certified_model",
                        lambda cfg, rd, name: (arch, double))
    payload = fid.fidelity_certificate(
        _cfg(arch=(arch_name,), pretrain_seed=0), run_dir, arch_name,
        oracle_set=_parent_oracle_set())
    assert payload["parent"] == parent
    return payload, {r["name"]: r for r in payload["per_system"]}


# Bounds on |E_xc_nn - E_xc_parent| with the parent behind the model
# interface, anchored to the sto-3g / grid 1 measurements quoted in the test
# below: (O and H2O, H atom), in Ha.
_PARENT_AS_MODEL_BOUNDS = {
    "pbe": (1e-10, 5e-6),
    "scan": (1e-8, 5e-6),
}


@pytest.mark.parametrize("parent", ["pbe", "scan"])
def test_certificate_is_exact_when_the_model_is_the_parent_functional(
        tmp_path, monkeypatch, parent):
    """O-A. With the parent itself behind the model interface, the
    certificate's E_xc_nn - E_xc_parent on the O atom and on H2O is round-off
    and the verdict is PASS: the model path and the point-wise parent route
    reduce the same density, on the same points and weights, through the same
    libxc, the one in JAX and the other in numpy.

    Measured at sto-3g / grid 1. PBE: 3.6e-15 Ha (O) and 7.1e-15 Ha (H2O),
    a few ulps of an E_xc of order 8 Ha. SCAN: 2.4e-9 Ha (O), 3.4e-10 Ha
    (H2O) -- the meta-GGA interface carries the smoothed, clamped alpha, not
    tau, so the 572 (O) / 627 (H2O) tail points clamped at alpha = 100 carry
    1.5e-4 / 5.6e-4 electrons whose tau the double cannot recover, and the
    indicator's own smoothing floor (metagga._ALPHA_SMOOTHING_WIDTH / 2,
    which the certificate's inversion does not undo) enters its tau at the
    1e-9-Ha level. The SCAN figures replace a superseded record of 2.0e-10
    (O) and 1.6e-9 Ha (H2O) taken under the hard clip; the O atom moved 12x,
    so the 1e-8 bound now clears it by 4.2x rather than 50x -- the tightest
    margin in this test, and what a further move of the indicator's floor
    would trip. Both figures stay more than five orders inside tol_atom. The
    per-spin meta-GGA blocks are on trial here as much as the energy path: a
    wrong doubled-channel alpha would move the O atom by mHa.

    The H atom is pinned separately: the production path clips zeta to
    1 - 1e-6 (oneshot._ZETA_BOUNDARY_EPS) where the parent sees zeta = 1
    exactly, and the one-electron atom's correlation is not zero
    (self-correlation), so the double carries the zeta-derivative of
    rho eps_c across that clip -- 8.0e-7 Ha for PBE and 4.4e-8 Ha for SCAN,
    more than two orders inside tol_atom."""
    bound_heavy, bound_h = _PARENT_AS_MODEL_BOUNDS[parent]
    payload, by_name = _certify_with_parent_as_model(tmp_path, monkeypatch,
                                                     parent, 0.0)
    for name in ("atom_O", "H2O"):
        rec = by_name[name]
        assert abs(rec["E_xc_nn"] - rec["E_xc_parent"]) < bound_heavy, (
            name, rec)
        assert abs(rec["parent_grid_diff_Ha"]) < fid.PARENT_GRID_TOL_HA
        assert abs(rec["parent_record_diff_Ha"]) < fid.PARENT_GRID_TOL_HA
    assert abs(by_name["atom_H"]["E_xc_nn"]
               - by_name["atom_H"]["E_xc_parent"]) < bound_h
    assert payload["verdict"] == "PASS"
    assert payload["summary"]["failure_reasons"] == []
    assert payload["summary"]["max_atom_mHa"] < 1e-2
    assert payload["summary"]["max_dAE_kcalmol"] < 1e-2


def test_certificate_measures_a_known_per_electron_offset(tmp_path,
                                                          monkeypatch):
    """O-B. The parent plus a constant c = 0.5 mHa per electron must move
    every dE_xc by exactly c N_e_grid, N_e_grid the ELECTRON count the
    record's quadrature carries -- named to keep it apart from the payload's
    n_grid, the number of grid points -- (the shift against the c = 0
    certificate is compared, so the H atom's zeta-clip residual of the
    previous test drops out); the O atom (N_e = 8) then exceeds tol_atom by
    the predicted 4.0 mHa while the atomization fold cancels the offset to
    c (N_e_grid(H2O) - N_e_grid(O) - 2 N_e_grid(H)), the quadrature residual.
    That is why the certificate carries an atomic tolerance beside the
    atomization one: a per-electron offset is invisible to dAE. Measured
    shift minus prediction: 6.8e-12 mHa (O), 1.4e-12 mHa (H2O) at sto-3g /
    grid 1."""
    import numpy as np
    from xcquinox.pipeline.data import precompute_fixed_density_data
    c_mha = 0.5
    _payload0, base = _certify_with_parent_as_model(tmp_path, monkeypatch,
                                                    "pbe", 0.0)
    payload, by_name = _certify_with_parent_as_model(
        tmp_path, monkeypatch, "pbe", c_mha / fid.HA_TO_MHA)
    n_e_grid = {}
    for ms in _parent_oracle_set():
        # The record the certificate measured: the O atom is degenerate and
        # carries the certificate's orientation lock, the others none.
        md = precompute_fixed_density_data(
            ms, reference_xc="pbe",
            orientation_lock_strength=by_name[ms.name][
                "orientation_lock_strength"])
        n_e_grid[ms.name] = float(np.sum(np.asarray(md["grid_weights"])
                                         * np.asarray(md["rho_grid"])))
    assert by_name["atom_O"]["orientation_lock_strength"] == (
        fid.atom_orientation_lock_strength())
    assert by_name["atom_H"]["orientation_lock_strength"] == 0.0
    assert n_e_grid["atom_O"] == pytest.approx(8.0, abs=1e-3)
    for name, rec in by_name.items():
        shift = rec["dE_xc_mHa"] - base[name]["dE_xc_mHa"]
        assert shift == pytest.approx(c_mha * n_e_grid[name], abs=1e-7), name
    for name in ("atom_O", "H2O"):
        assert by_name[name]["dE_xc_mHa"] == pytest.approx(
            c_mha * n_e_grid[name], abs=1e-6), name
    assert payload["verdict"] == "FAIL"
    assert payload["summary"]["max_atom_mHa"] == pytest.approx(
        c_mha * n_e_grid["atom_O"], abs=1e-6)
    assert any("tol_atom" in r for r in payload["summary"]["failure_reasons"])
    predicted_dae = (c_mha * (n_e_grid["H2O"] - n_e_grid["atom_O"]
                              - 2 * n_e_grid["atom_H"])
                     / fid.HA_TO_MHA * fid.HA_TO_KCAL)
    dae = {r["name"]: r["dAE_kcalmol"] for r in payload["per_atomization"]}
    dae0 = {r["name"]: r["dAE_kcalmol"] for r in _payload0["per_atomization"]}
    assert dae["H2O"] - dae0["H2O"] == pytest.approx(predicted_dae, abs=1e-7)
    assert abs(dae["H2O"]) < 1.0
    assert not any("tol_AE" in r for r in payload["summary"]["failure_reasons"])


# ---------------------------------------------------------------------------
# The reference SCF must have converged
# ---------------------------------------------------------------------------


def test_certificate_fails_by_name_when_a_reference_scf_did_not_converge(
        tmp_path):
    """An unconverged reference is a named consistency failure of its own:
    the per-system entry carries the stamp, the summary counts it, and the
    reason names the system -- it is never folded into the generic
    'could not be evaluated' bucket and can never PASS."""
    run_dir = str(tmp_path / "run")
    _stub_checkpoint(run_dir)

    def _evaluate(model, descriptors, mol_spec, *, parent,
                  auxbasis=None, orientation_lock_strength=0.0):
        if mol_spec.name == "H2":
            raise fid.ReferenceNotConverged(
                "the reference PBE SCF for 'H2' did not converge", cycles=50)
        return _fake_evaluate({"atom_H": 0.1})(
            model, descriptors, mol_spec, parent=parent)

    payload = fid.fidelity_certificate(
        _cfg(), run_dir, "deep_3x16", oracle_set=_tiny_oracle_set(),
        evaluate=_evaluate)
    assert payload["verdict"] == "FAIL"
    entry = [r for r in payload["per_system"] if r["name"] == "H2"][0]
    assert entry["reference_scf_converged"] is False
    assert entry["reference_scf_cycles"] == 50
    assert "did not converge" in entry["error"]
    s = payload["summary"]
    assert s["n_reference_unconverged"] == 1
    assert s["n_failed_systems"] == 1
    assert any("did not converge" in r and "H2" in r
               for r in s["failure_reasons"])
    assert not any("could not be evaluated" in r for r in s["failure_reasons"])
    assert fid.certificate_status(run_dir, "deep_3x16")[0] == "FAIL"


# ---------------------------------------------------------------------------
# Degenerate free atoms are evaluated on an orientation-locked density
# ---------------------------------------------------------------------------

def _free_atom(symbol, charge=0, spin=0):
    from xcquinox.pipeline.config import MoleculeSpec
    return MoleculeSpec(name=fid.atom_system_name(symbol, charge),
                        atom=f"{symbol} 0 0 0", basis="sto-3g", charge=charge,
                        spin=spin, atom_composition=((symbol, 1),))


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# A non-finite measurement can satisfy no tolerance
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The producer's own non-convergence refusal is counted by name
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Per-system containment, diagnostics and config hygiene
# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# The uniform-gas gate in the certificate's model class
# ---------------------------------------------------------------------------

def test_the_certificate_class_check_reports_the_gate(tmp_path):
    """The certificate records the uniform-gas gate the certified networks
    carry, and a run of the other gate does not accept it.

    Oracle: the payload the certificate writes (with the per-system
    evaluation stubbed, so no SCF is paid for) and ``model_class_mismatches``
    on hand-built certificates. A certificate written before the field
    existed states nothing, which reads as the gate every model before it
    carried, so the files of the earlier campaigns are read exactly as they
    were.
    """
    run_dir = str(tmp_path / "run")
    _stub_checkpoint(run_dir)
    cfg = _cfg()
    cfg.model = SimpleNamespace(parent_anchor=False,
                                descriptor_coordinates="legacy",
                                ueg_gate="x2")
    payload = fid.fidelity_certificate(
        cfg, run_dir, "deep_3x16",
        oracle_set=_tiny_oracle_set(),
        evaluate=_fake_evaluate({"atom_H": 0.5, "H2": 1.0}))
    assert payload["ueg_gate"] == "x2"

    assert fid.model_class_mismatches(cfg, payload) == []

    tanh_cfg = _cfg()
    tanh_cfg.model = SimpleNamespace(parent_anchor=False,
                                     descriptor_coordinates="legacy",
                                     ueg_gate="tanh2")
    assert fid.model_class_mismatches(tanh_cfg, payload) == [
        ("ueg_gate", "x2", "tanh2")]

    legacy_cert = {"parent_anchor": False, "descriptor_coordinates": "legacy"}
    assert fid.model_class_mismatches(tanh_cfg, legacy_cert) == []
    assert fid.model_class_mismatches(cfg, legacy_cert) == [
        ("ueg_gate", "tanh2", "x2")]
