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
from pathlib import Path

import pytest

from xcquinox.pipeline.dfs_pretrain_set import (
    dfs_pretrain_records, dfs_pretrain_systems,
)

_MOLECULE_NAMES = (
    "H2", "N2", "LiF", "HCN", "CO2", "Cl2", "F2", "O2", "C2H2", "CO",
    "HCl", "LiH", "Na2", "AlCl3", "PH3", "Si2", "C4H6", "CH4", "SiCH6",
    "C3H8", "CH2", "SiH4",
)
_ATOM_SPINS = {"P": 3, "N": 3, "H": 1, "Li": 1, "O": 2, "Cl": 1,
               "Al": 1, "S": 2}


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


def _exporter_path():
    """Path of ``tools/generate_dfs_pretrain_set.py`` in the source tree.

    The exporter is a repository script rather than package code, so it is
    located relative to the loader module, not imported by name. Returns None
    where the package is installed without the repository beside it.
    """
    import xcquinox.pipeline.dfs_pretrain_set as module
    repo_root = Path(module.__file__).resolve().parents[2]
    script = repo_root / "tools" / "generate_dfs_pretrain_set.py"
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
    from xcquinox.pipeline.dfs_pretrain_set import _DATA_PATH
    with open(_DATA_PATH) as f:
        committed = json.load(f)
    assert exporter.build(exporter.DEFAULT_TRAJ) == committed


_COMMITTED_SHA256 = \
    "d1599b796ea344e25b4f6cad5dd628115bc01c643e2662a1066bb818bd9b6900"


# ---------------------------------------------------------------------------
# Name against geometry
#
# Names and trajectory indices are two hand-written parallel lists, and the
# composition of a record is derived from the geometry the index selected, so
# the two are self-consistent for any pairing. Only the formula implied by the
# NAME is an independent statement about what the record is supposed to be:
# swapping two indices, or shifting one, is caught here and nowhere else.
# ---------------------------------------------------------------------------


