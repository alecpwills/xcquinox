"""Tests for xcquinox.alec.cluster.fidelity -- the per-architecture physics
certificate.

The cheap layer (the certificate predicate, the parent resolution, the oracle
set) is tested directly. The certificate itself is tested twice: once with the
per-system evaluation monkeypatched at the ``evaluate`` seam, so the verdict
arithmetic and the JSON schema are exercised with no SCF at all, and once for
REAL on H and H2 at sto-3g with networks built in the test, so the energy path,
the libxc parent route and the atomization fold are pinned against physics.
"""
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from xcquinox.alec.cluster import fidelity as fid


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

def test_fidelity_module_body_imports_only_cheap_modules():
    """`cluster status`, validate_run, merge and the train task's PARENT
    process all read certificates. fidelity.py's module BODY must therefore
    stay a pure reader -- path helpers, the certificate predicates, constants
    -- with every jax / pyscf / xcquinox.alec.data import inside a function,
    so a reader never triggers a model import or an SCF-capable stack it does
    not use. Checked on the source: importing any cluster module already
    executes the package's own jax-carrying __init__, so sys.modules cannot
    distinguish this file's cost from the package's."""
    import ast
    import inspect
    tree = ast.parse(inspect.getsource(fid))
    top = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            top += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            top.append(node.module or "")
    assert sorted(top) == [
        "__future__", "argparse", "json", "os", "sys", "time",
        "xcquinox.alec.cluster.grid_config",
        "xcquinox.alec.cluster.materialize",
    ], sorted(top)


def test_fidelity_imports_in_a_fresh_interpreter():
    """No import cycle: _pretrain imports fidelity, fidelity must not import
    _pretrain (it derives the distinct-arch list from _canon_axis instead)."""
    out = subprocess.run(
        [sys.executable, "-c", "import xcquinox.alec.cluster.fidelity"],
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


def test_certificate_status_by_run_dir_and_arch_uses_the_harness_layout(tmp_path):
    from xcquinox.alec.cluster.grid_config import pretrain_checkpoint_dir
    _write_certificate(str(tmp_path), "deep_3x16")
    assert fid.certificate_path(str(tmp_path), "deep_3x16") == os.path.join(
        pretrain_checkpoint_dir(str(tmp_path), "deep_3x16"),
        fid.CERTIFICATE_FILENAME)
    assert fid.certificate_status(str(tmp_path), "deep_3x16")[0] == "PASS"


def test_read_certificate_returns_none_when_absent(tmp_path):
    assert fid.read_certificate(str(tmp_path)) is None


# ---------------------------------------------------------------------------
# The ON-NODE gate: fidelity.enforce = False records the verdict and continues
# ---------------------------------------------------------------------------

def test_certificate_enforced_defaults_to_true_when_the_field_is_absent(
        tmp_path):
    """A certificate that does not say otherwise is enforcing; so is an
    absent one, which cannot say anything at all."""
    d = _write_certificate(str(tmp_path), "deep_3x16")
    assert fid.certificate_enforced_in(d) is True
    assert fid.certificate_enforced_in(str(tmp_path / "nowhere")) is True


def test_certificate_enforced_reads_the_recorded_flag(tmp_path):
    d = _write_certificate(str(tmp_path), "deep_3x16", verdict="FAIL",
                           enforced=False)
    assert fid.certificate_enforced_in(d) is False


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


def test_gate_never_allows_a_missing_certificate(tmp_path):
    """Enforcement can only be waived by a certificate that exists to record
    the waiver; an absent one waives nothing."""
    allowed, message = fid.gate_certificate(str(tmp_path), "deep_3x16")
    assert allowed is False
    assert "MISSING" in message or "was never checked" in message


def test_gate_never_allows_an_unreadable_certificate(tmp_path):
    d = tmp_path / "pretrain" / "deep_3x16"
    d.mkdir(parents=True)
    (d / fid.CERTIFICATE_FILENAME).write_text("{truncated")
    allowed, _message = fid.gate_certificate(str(tmp_path), "deep_3x16")
    assert allowed is False


# ---------------------------------------------------------------------------
# Parent resolution: the arch's RUNG picks the parent, not inputs.seed_xc
# ---------------------------------------------------------------------------

def test_parent_is_pbe_for_gga_rung_and_scan_for_meta_gga():
    assert fid.resolve_parent("deep_3x16") == "pbe"
    assert fid.resolve_parent("deep_cusp_3x16") == "pbe"
    assert fid.resolve_parent("deep_rung35_3x16") == "pbe"
    assert fid.resolve_parent("deep_mgga_3x16") == "scan"


def test_parent_agrees_with_the_rung_seed_policy():
    from xcquinox.alec.rungs import seed_xc_for_arch
    from xcquinox.alec.config import list_architectures
    for name in list_architectures():
        assert fid.resolve_parent(name) == seed_xc_for_arch(name)


def test_dfs_level_follows_the_parent():
    assert fid.dfs_level_for_parent("pbe") == "gga"
    assert fid.dfs_level_for_parent("scan") == "mgga"


def test_distinct_archs_matches_the_pretrain_workers_selector():
    from xcquinox.alec.cluster import _pretrain as pt
    cfg = _cfg(arch=("medium", "deep", "medium", "shallow"))
    assert fid._distinct_archs(cfg) == ["deep", "medium", "shallow"]
    assert fid._distinct_archs(cfg) == pt._distinct_archs(cfg)


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

def test_atom_system_names_are_canonical():
    assert fid.atom_system_name("O", 0) == "atom_O"
    assert fid.atom_system_name("F", -1) == "atom_F-"
    assert fid.atom_system_name("Na", 1) == "atom_Na+"
    assert fid.atom_system_name("O", -2) == "atom_O-2"


def test_oracle_set_puts_atoms_first_then_molecules_each_sorted():
    systems = fid.build_oracle_set(_cfg(), "deep_3x16")
    names = [ms.name for ms in systems]
    atoms = [n for n in names if n.startswith("atom_")]
    mols = [n for n in names if not n.startswith("atom_")]
    assert names == atoms + mols
    assert atoms == sorted(atoms)
    assert mols == sorted(mols)


def test_oracle_set_carries_every_pool_free_atom_with_its_pool_spin():
    from xcquinox.alec.full_benchmark_pools import load_full_held_out_pools
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


def test_oracle_set_carries_the_dfs_molecules_and_the_fixed_three():
    systems = {ms.name for ms in fid.build_oracle_set(_cfg(), "deep_3x16")}
    for name in ("H2", "LiF", "AlCl3", "C4H6", "SiH4"):
        assert name in systems
    for name in ("H2O", "N2", "CH4"):
        assert name in systems


def test_meta_gga_oracle_set_drops_h2_but_keeps_n2_from_the_fixed_three():
    """The meta-GGA DFS variant omits H2 and N2; the fixed molecule set
    restores N2, so every architecture is measured on a common N2 / H2O / CH4
    core whatever its rung."""
    gga = {ms.name for ms in fid.build_oracle_set(_cfg(), "deep_3x16")}
    mgga = {ms.name for ms in fid.build_oracle_set(_cfg(), "deep_mgga_3x16")}
    assert "H2" in gga and "H2" not in mgga
    assert "N2" in gga and "N2" in mgga


def test_oracle_set_supplies_a_free_atom_for_every_element_it_dissociates():
    systems = fid.build_oracle_set(_cfg(), "deep_3x16")
    names = {ms.name for ms in systems}
    for ms in systems:
        if fid.is_atom_system(ms):
            continue
        for sym, _n in ms.atom_composition:
            assert fid.atom_system_name(sym, 0) in names, (ms.name, sym)


def test_oracle_set_adds_lithium_and_sodium_which_no_pool_carries():
    names = {ms.name for ms in fid.build_oracle_set(_cfg(), "deep_3x16")}
    assert "atom_Li" in names and "atom_Na" in names


def test_a_dfs_record_wins_over_the_fixed_molecule_of_the_same_name():
    """A name the DFS set carries is certified at the DFS geometry, which is
    the geometry the networks were pretrained on. N2 therefore comes from the
    DFS record for a GGA rung and from the fixed set only for a meta-GGA one,
    whose DFS variant drops it; CH4's two geometries are identical; H2O is in
    neither DFS variant and always comes from the fixed set."""
    from xcquinox.alec.dfs_pretrain_set import dfs_pretrain_records
    dfs = {r["name"]: r for r in dfs_pretrain_records("gga")}
    gga = {ms.name: ms for ms in fid.build_oracle_set(_cfg(), "deep_3x16")}
    mgga = {ms.name: ms
            for ms in fid.build_oracle_set(_cfg(), "deep_mgga_3x16")}
    fixed = {name: atom for name, atom, _c, _s in fid._FIXED_MOLECULES}
    assert gga["N2"].atom == dfs["N2"]["atom"] != fixed["N2"]
    assert gga["CH4"].atom == dfs["CH4"]["atom"] == fixed["CH4"]
    assert mgga["N2"].atom == fixed["N2"]
    assert gga["H2O"].atom == mgga["H2O"].atom == fixed["H2O"]


def test_a_fixed_molecule_composition_counts_every_nucleus():
    """The atomization fold weights each free atom's offset by its count, so
    a fixed molecule's composition carries counts, not just the element set."""
    gga = {ms.name: ms for ms in fid.build_oracle_set(_cfg(), "deep_3x16")}
    mgga = {ms.name: ms
            for ms in fid.build_oracle_set(_cfg(), "deep_mgga_3x16")}
    assert gga["H2O"].atom_composition == (("H", 2), ("O", 1))
    assert mgga["N2"].atom_composition == (("N", 2),)


def test_an_element_with_no_ground_state_spin_is_refused(tmp_path,
                                                         monkeypatch):
    """A molecule whose element has no recorded ground-state spin has no
    atomization reference; the oracle set refuses it instead of silently
    dropping the free atom the fold needs."""
    monkeypatch.setattr(fid, "_FIXED_MOLECULES",
                        (("KrH", "Kr 0 0 0; H 0 0 1.4", 0, 1),))
    with pytest.raises(ValueError, match="ground-state spin"):
        fid.build_oracle_set(_cfg(), "deep_3x16")


def test_sources_that_disagree_on_a_free_atom_spin_are_refused(monkeypatch):
    """The pools and the DFS set must declare one spin per free atom: folding
    a molecule against an atom evaluated at a different multiplicity would
    change the atomization offset with nothing in the certificate to show it."""
    import xcquinox.alec.dfs_pretrain_set as dfs_set
    records = dfs_set.dfs_pretrain_records("gga")
    for record in records:
        if record["name"] == "H":
            record["spin"] = 3          # the pools carry the H atom at 2S = 1
    monkeypatch.setattr(dfs_set, "dfs_pretrain_records",
                        lambda level: records)
    with pytest.raises(ValueError, match="exactly one spin"):
        fid.build_oracle_set(_cfg(), "deep_3x16")


def test_ground_state_spin_table_agrees_with_the_pool_spins():
    """The certificate's Hund ground-state table is the atomization reference;
    it must agree species by species with the spins the BH76 / W4-11 pools
    carry, or a molecule would be folded against a different atom than the
    benchmark uses."""
    from xcquinox.alec.full_benchmark_pools import load_full_held_out_pools
    pool, _ = load_full_held_out_pools(basis="sto-3g", grid_level=1)
    for ms in pool.values():
        comp = tuple(ms.atom_composition)
        if len(comp) != 1 or int(comp[0][1]) != 1 or ms.charge != 0:
            continue
        sym = comp[0][0]
        assert fid._ATOM_GROUND_SPIN[sym] == ms.spin, sym


def test_oracle_set_specs_carry_the_run_identity():
    cfg = _cfg(basis="def2-tzvpd", grid_level=2)
    for ms in fid.build_oracle_set(cfg, "deep_3x16"):
        assert ms.basis == "def2-tzvpd"
        assert ms.grid_level == 2
        assert ms.external_data_path is None


def test_is_atom_system():
    from xcquinox.alec.config import MoleculeSpec
    atom = MoleculeSpec(name="atom_O", atom="O 0 0 0", basis="sto-3g", spin=2,
                        atom_composition=(("O", 1),))
    mol = MoleculeSpec(name="OH", atom="O 0 0 0; H 0 0 1", basis="sto-3g",
                       spin=1, atom_composition=(("H", 1), ("O", 1)))
    assert fid.is_atom_system(atom)
    assert not fid.is_atom_system(mol)
