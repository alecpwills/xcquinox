"""The committed DFS pretraining set (spec Section 6).

The set is the pretraining protocol of the DFS code (Dick and
Fernandez-Serra, Phys. Rev. B 104, L161109 (2021)): eight free atoms with
explicit spins plus 22 G2/97 molecules from the Haunschild and Klopper
trajectory (Theor. Chem. Acc. 131, 1112 (2012)), all molecules run closed
shell. These tests pin the count, the names, the spins and two geometries so
a regenerated JSON that silently changes the set is caught.
"""
import hashlib
import importlib.util
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from xcquinox.alec.dfs_pretrain_set import (
    LEVELS, MGGA_EXCLUDED, dfs_pretrain_records, dfs_pretrain_systems,
    formula_from_name,
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


def _exporter_path():
    """Path of ``scripts/generate_dfs_pretrain_set.py`` in the source tree.

    The exporter is a repository script rather than package code, so it is
    located relative to the loader module, not imported by name. Returns None
    where the package is installed without the repository beside it.
    """
    import xcquinox.alec.dfs_pretrain_set as module
    repo_root = Path(module.__file__).resolve().parents[2]
    script = repo_root / "scripts" / "generate_dfs_pretrain_set.py"
    return script if script.exists() else None


def _load_exporter():
    """Import the exporter script by path, or None where it is absent."""
    script = _exporter_path()
    if script is None:
        return None
    spec = importlib.util.spec_from_file_location(
        "_generate_dfs_pretrain_set_under_test", str(script))
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


_COMMITTED_SHA256 = \
    "d1599b796ea344e25b4f6cad5dd628115bc01c643e2662a1066bb818bd9b6900"


def test_committed_json_bytes_carry_the_pinned_digest():
    """The committed data file hashes to the digest pinned above.

    The pin is a deliberate change detector, not a checksum of convenience:
    it fires on any edit of the bytes -- a single coordinate moved by
    0.05 A included -- and it fires everywhere, whereas
    ``test_committed_json_equals_a_fresh_export_from_the_trajectory`` skips
    wherever the exporter script, the trajectory or ASE is absent (installed
    wheels, compute nodes). Any regeneration of the set must therefore update
    the literal above in the same change; the regeneration test, where it
    runs, is what establishes that the new bytes are a faithful export of the
    trajectory rather than an accepted corruption.
    """
    from xcquinox.alec.dfs_pretrain_set import _DATA_PATH
    digest = hashlib.sha256(Path(_DATA_PATH).read_bytes()).hexdigest()
    assert digest == _COMMITTED_SHA256, _DATA_PATH


# ---------------------------------------------------------------------------
# Name against geometry
#
# Names and trajectory indices are two hand-written parallel lists, and the
# composition of a record is derived from the geometry the index selected, so
# the two are self-consistent for any pairing. Only the formula implied by the
# NAME is an independent statement about what the record is supposed to be:
# swapping two indices, or shifting one, is caught here and nowhere else.
# ---------------------------------------------------------------------------

def _pairs(composition):
    """A record's atom_composition as sorted (symbol, count) tuples."""
    return tuple(sorted((str(s), int(n)) for s, n in composition))


def test_name_parser_reads_concatenated_element_symbols():
    """Oracle for the parser the record check depends on."""
    assert formula_from_name("H2") == (("H", 2),)
    assert formula_from_name("CO") == (("C", 1), ("O", 1))      # not cobalt
    assert formula_from_name("Cl2") == (("Cl", 2),)
    assert formula_from_name("CH4") == (("C", 1), ("H", 4))
    assert formula_from_name("AlCl3") == (("Al", 1), ("Cl", 3))
    assert formula_from_name("SiCH6") == (("C", 1), ("H", 6), ("Si", 1))
    assert formula_from_name("C4H6") == (("C", 4), ("H", 6))
    assert formula_from_name("Si") == (("Si", 1),)
    # A repeated element is summed rather than overwritten.
    assert formula_from_name("CH3CH3") == (("C", 2), ("H", 6))
    for bad in ("", "h2", "2H", "C-H", "C0", "CH 4", "1"):
        with pytest.raises(ValueError):
            formula_from_name(bad)


def test_parser_reproduces_every_committed_name():
    """Every name in the set parses, and to the length the geometry has."""
    for r in dfs_pretrain_records("gga"):
        parsed = formula_from_name(r["name"])
        n_atoms = sum(n for _, n in parsed)
        assert n_atoms == len(r["atom"].split(";")), r["name"]


def test_every_record_name_matches_its_geometry():
    """The formula the name declares equals the record's stored composition.

    This is the check that fails when a trajectory index is swapped between
    two records or shifted by one: the record keeps its name while carrying
    another species' coordinates, and the composition exported alongside them
    spells that other species. The stored composition is tied to the geometry
    string itself by
    ``test_atom_composition_matches_the_geometry_for_every_record`` and, by
    atom count, by ``test_parser_reproduces_every_committed_name``; the three
    together close the loop from name to coordinates, which no one of them
    closes alone.
    """
    for r in dfs_pretrain_records("gga"):
        assert formula_from_name(r["name"]) == _pairs(r["atom_composition"]), \
            r["name"]


def test_systems_canonicalize_an_unsorted_composition(monkeypatch):
    """A hand-edited record with an out-of-order composition is canonicalized.

    MoleculeSpec is frozen and hashed on its fields, so an unsorted
    composition would silently produce a spec that fails to match an
    otherwise identical one.
    """
    import xcquinox.alec.dfs_pretrain_set as module
    raw = json.loads(json.dumps(module._load()))
    for m in raw["molecules"]:
        if m["name"] == "C4H6":
            m["atom_composition"] = [["H", 6], ["C", 4]]
    monkeypatch.setattr(module, "_CACHE", raw)
    spec = {ms.name: ms for ms in dfs_pretrain_systems("gga")}["C4H6"]
    assert spec.atom_composition == (("C", 4), ("H", 6))
    assert isinstance(hash(spec), int)


def test_regenerated_file_stays_group_readable(tmp_path):
    """mkstemp creates at 0600; the exported data must not inherit that.

    The committed JSON is read from a group-shared cluster tree, so a
    regeneration that left it owner-only would make the package data
    unreadable for every other member of the group.
    """
    exporter = _load_exporter()
    if exporter is None:
        pytest.skip("exporter script not present alongside the package")
    out = tmp_path / "written.json"
    exporter._write_json_atomic({"a": 1}, str(out))
    assert stat.S_IMODE(out.stat().st_mode) == 0o644
    assert json.load(open(out)) == {"a": 1}


def test_missing_trajectory_is_a_one_line_usage_error(tmp_path):
    """An absent trajectory is a usage error, not an ASE traceback."""
    script = _exporter_path()
    if script is None:
        pytest.skip("exporter script not present alongside the package")
    out = tmp_path / "unwritten.json"
    missing = tmp_path / "absent" / "g2_97.traj"
    proc = subprocess.run(
        [sys.executable, str(script), "--traj", str(missing),
         "--out", str(out)],
        capture_output=True, text=True)
    assert proc.returncode == 2, proc.stderr
    assert "Traceback" not in proc.stderr
    last = proc.stderr.strip().splitlines()[-1]
    assert "error:" in last and str(missing) in last
    assert not out.exists()
