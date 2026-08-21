"""The committed DFS pretraining set (spec Section 6).

The set is the pretraining protocol of the DFS code (Dick and
Fernandez-Serra, Phys. Rev. B 104, L161109 (2021)): eight free atoms with
explicit spins plus 22 G2/97 molecules from the Haunschild and Klopper
trajectory (Theor. Chem. Acc. 131, 1112 (2012)), all molecules run closed
shell. These tests pin the count, the names, the spins and two geometries so
a regenerated JSON that silently changes the set is caught.
"""
import importlib.util
import json
import os

import pytest

from xcquinox.alec.dfs_pretrain_set import (
    LEVELS, MGGA_EXCLUDED, dfs_pretrain_records, dfs_pretrain_systems,
)

_MOLECULE_NAMES = (
    "H2", "N2", "LiF", "HCN", "CO2", "Cl2", "F2", "O2", "C2H2", "CO",
    "HCl", "LiH", "Na2", "AlCl3", "PH3", "Si2", "C4H6", "CH4", "SiCH6",
    "C3H8", "CH2", "SiH4",
)
_ATOM_SPINS = {"P": 3, "N": 3, "H": 1, "Li": 1, "O": 2, "Cl": 1,
               "Al": 1, "S": 2}


def test_levels_and_exclusions_are_declared():
    assert LEVELS == ("gga", "mgga")
    assert MGGA_EXCLUDED == ("H2", "N2")


def test_gga_level_is_thirty_systems_eight_atoms_twentytwo_molecules():
    recs = dfs_pretrain_records("gga")
    atoms = [r for r in recs if r["kind"] == "atom"]
    mols = [r for r in recs if r["kind"] == "molecule"]
    assert len(recs) == 30
    assert len(atoms) == 8
    assert len(mols) == 22


def test_mgga_level_drops_h2_and_n2_only():
    gga = {r["name"] for r in dfs_pretrain_records("gga")}
    mgga = {r["name"] for r in dfs_pretrain_records("mgga")}
    assert gga - mgga == {"H2", "N2"}
    assert len(dfs_pretrain_records("mgga")) == 28


def test_molecule_names_and_order_are_the_spec_order():
    mols = [r for r in dfs_pretrain_records("gga") if r["kind"] == "molecule"]
    assert tuple(r["name"] for r in mols) == _MOLECULE_NAMES


def test_every_molecule_is_closed_shell_and_neutral():
    for r in dfs_pretrain_records("gga"):
        if r["kind"] != "molecule":
            continue
        assert r["spin"] == 0, r["name"]
        assert r["charge"] == 0, r["name"]


def test_atom_spins_are_the_hund_ground_states_the_protocol_declares():
    atoms = {r["name"]: r for r in dfs_pretrain_records("gga")
             if r["kind"] == "atom"}
    assert set(atoms) == set(_ATOM_SPINS)
    for name, spin in _ATOM_SPINS.items():
        assert atoms[name]["spin"] == spin
        assert atoms[name]["charge"] == 0
        assert atoms[name]["atom_composition"] == [[name, 1]]


def test_h2_geometry_is_the_g2_97_entry():
    mols = {r["name"]: r for r in dfs_pretrain_records("gga")}
    h2 = mols["H2"]
    assert h2["g2_97_index"] == 2
    lines = [ln.strip() for ln in h2["atom"].split(";")]
    assert lines == ["H 0.0000000000 0.0000000000 0.3713950000",
                     "H 0.0000000000 0.0000000000 -0.3713950000"]


def test_ch4_geometry_is_the_g2_97_entry():
    mols = {r["name"]: r for r in dfs_pretrain_records("gga")}
    ch4 = mols["CH4"]
    assert ch4["g2_97_index"] == 10
    lines = [ln.strip() for ln in ch4["atom"].split(";")]
    assert lines[0] == "C 0.0000000000 0.0000000000 0.0000000000"
    assert lines[1] == "H 0.6303820000 0.6303820000 0.6303820000"
    assert len(lines) == 5


def test_atom_composition_matches_the_geometry_for_every_record():
    for r in dfs_pretrain_records("gga"):
        symbols = [ln.strip().split()[0] for ln in r["atom"].split(";")]
        counts = {}
        for s in symbols:
            counts[s] = counts.get(s, 0) + 1
        assert sorted(tuple(x) for x in r["atom_composition"]) == \
            sorted(counts.items()), r["name"]


def test_systems_are_molecule_specs_carrying_the_requested_identity():
    systems = dfs_pretrain_systems("gga", basis="sto-3g", grid_level=1)
    assert len(systems) == 30
    assert all(ms.basis == "sto-3g" for ms in systems)
    assert all(ms.grid_level == 1 for ms in systems)
    by_name = {ms.name: ms for ms in systems}
    assert by_name["O"].spin == 2
    assert by_name["C4H6"].atom_composition == (("C", 4), ("H", 6))


def test_systems_default_to_the_production_identity():
    systems = dfs_pretrain_systems("gga")
    assert systems[0].basis == "6-311++G(3df,2pd)"
    assert systems[0].grid_level == 3


def test_unknown_level_is_rejected():
    with pytest.raises(ValueError, match="level"):
        dfs_pretrain_records("lda")


def test_records_are_copies_the_caller_cannot_poison():
    a = dfs_pretrain_records("gga")
    a[0]["name"] = "MUTATED"
    b = dfs_pretrain_records("gga")
    assert b[0]["name"] != "MUTATED"


def test_committed_json_declares_its_provenance():
    from xcquinox.alec.dfs_pretrain_set import _DATA_PATH
    with open(_DATA_PATH) as f:
        raw = json.load(f)
    assert "source" in raw
    assert "g2_97" in raw["source"]["trajectory"]
    assert raw["source"]["indices"][:3] == [2, 113, 25]


def _load_exporter():
    """Import ``scripts/generate_dfs_pretrain_set.py`` from the source tree.

    The exporter is a repository script rather than package code, so it is
    loaded by path relative to the package data directory.
    """
    from xcquinox.alec.dfs_pretrain_set import _DATA_PATH
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.abspath(_DATA_PATH)))))
    script = os.path.join(repo_root, "scripts", "generate_dfs_pretrain_set.py")
    if not os.path.exists(script):
        return None
    spec = importlib.util.spec_from_file_location(
        "_generate_dfs_pretrain_set_under_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_committed_json_equals_a_fresh_export_from_the_trajectory():
    """The committed data is byte-equivalent to a regeneration.

    Skipped where the trajectory or ASE is unavailable (compute nodes carry
    neither); the remaining tests then pin the loader against the committed
    JSON alone.
    """
    exporter = _load_exporter()
    if exporter is None:
        pytest.skip("exporter script not present alongside the package")
    if not os.path.exists(exporter.DEFAULT_TRAJ):
        pytest.skip(f"trajectory absent: {exporter.DEFAULT_TRAJ}")
    pytest.importorskip("ase.io")
    from xcquinox.alec.dfs_pretrain_set import _DATA_PATH
    with open(_DATA_PATH) as f:
        committed = json.load(f)
    assert exporter.build(exporter.DEFAULT_TRAJ) == committed
