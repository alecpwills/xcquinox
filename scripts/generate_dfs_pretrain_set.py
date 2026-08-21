"""Export the DFS pretraining set to xcquinox/alec/data/dfs_pretrain_set.json.

The pretraining protocol of the DFS code (Dick and Fernandez-Serra,
Phys. Rev. B 104, L161109 (2021)) trains on eight free atoms with explicit
spins -- P (2S=3), N (3), H (1), Li (1), O (2), Cl (1), Al (1), S (2) -- plus
22 molecules taken from the Haunschild and Klopper G2/97 trajectory
(Theor. Chem. Acc. 131, 1112 (2012)) at the indices below, all run as closed
shells. The meta-GGA variant of the protocol drops H2 and N2.

The trajectory is an ASE file outside this repository, so the geometries are
exported ONCE into package data: the cluster nodes carry only this package,
and the certificate and the pretraining data generator must resolve the same
geometries with no ASE dependency at run time.

Usage:
    python scripts/generate_dfs_pretrain_set.py [--traj PATH] [--out PATH]
"""
import argparse
import json
import os
import sys
import tempfile

# G2/97 trajectory indices, in the order the DFS notebook lists them.
G2_97_INDICES = (2, 113, 25, 18, 11, 17, 114, 121, 101, 0, 20, 26, 29, 67,
                 28, 110, 125, 10, 115, 89, 105, 50)
# Names in the same order (the trajectory carries formulas, not these names).
MOLECULE_NAMES = ("H2", "N2", "LiF", "HCN", "CO2", "Cl2", "F2", "O2", "C2H2",
                  "CO", "HCl", "LiH", "Na2", "AlCl3", "PH3", "Si2", "C4H6",
                  "CH4", "SiCH6", "C3H8", "CH2", "SiH4")
# The eight free atoms and their 2S values, as the protocol declares them.
ATOM_SPINS = (("P", 3), ("N", 3), ("H", 1), ("Li", 1), ("O", 2), ("Cl", 1),
              ("Al", 1), ("S", 2))

DEFAULT_TRAJ = os.path.expanduser(
    "~/Documents/Research/xcdiff/data/haunschild_g2/g2_97.traj")


def _atom_string(symbols, positions):
    """PySCF geometry string in Angstrom, ten decimals (the trajectory's
    precision), one atom per ';'-separated field."""
    return "; ".join(
        f"{s} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}"
        for s, p in zip(symbols, positions))


def _composition(symbols):
    """Sorted (symbol, count) pairs, the MoleculeSpec.atom_composition form."""
    counts = {}
    for s in symbols:
        counts[s] = counts.get(s, 0) + 1
    return [[s, counts[s]] for s in sorted(counts)]


def _write_json_atomic(payload, path):
    """Write ``payload`` as pretty JSON through mkstemp + os.replace.

    A reader can never observe a half-written file. mkstemp creates at mode
    0600, so the result is set to 0644: the data is read from a group-shared
    project tree and an owner-only file would be unreadable to the rest of
    the group.
    """
    out_dir = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp-dfs-pretrain-", dir=out_dir)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp_name, path)
        tmp_name = None
    finally:
        if tmp_name is not None and os.path.exists(tmp_name):
            os.unlink(tmp_name)
    os.chmod(path, 0o644)


def _check_name_matches_geometry(record):
    """Reject a record whose name and geometry disagree.

    The names and the trajectory indices are two parallel hand-written lists
    and the composition is derived from the geometry the index selected, so a
    swapped or shifted index produces a fully self-consistent record under
    the wrong name. The formula the name spells out is the independent
    statement, and it is checked before anything is written.
    """
    from xcquinox.alec.dfs_pretrain_set import formula_from_name
    declared = formula_from_name(record["name"])
    found = tuple(sorted((str(s), int(n))
                         for s, n in record["atom_composition"]))
    if declared != found:
        raise ValueError(
            f"record {record['name']!r} (g2_97 index "
            f"{record['g2_97_index']}) has geometry {found}, but its name "
            f"spells {declared}")


def build(traj_path):
    from ase.io import read
    frames = read(traj_path, ":")
    atoms = [{"kind": "atom", "name": sym,
              "atom": f"{sym} 0.0000000000 0.0000000000 0.0000000000",
              "charge": 0, "spin": spin,
              "atom_composition": [[sym, 1]], "g2_97_index": None}
             for sym, spin in ATOM_SPINS]
    molecules = []
    for name, idx in zip(MOLECULE_NAMES, G2_97_INDICES):
        frame = frames[idx]
        symbols = list(frame.get_chemical_symbols())
        molecules.append({
            "kind": "molecule", "name": name,
            "atom": _atom_string(symbols, frame.get_positions()),
            "charge": 0, "spin": 0,
            "atom_composition": _composition(symbols),
            "g2_97_index": int(idx),
        })
    for record in atoms + molecules:
        _check_name_matches_geometry(record)
    return {
        "source": {
            "protocol": "Dick and Fernandez-Serra, Phys. Rev. B 104, "
                        "L161109 (2021), pretraining notebook",
            "trajectory": "haunschild_g2/g2_97.traj (Haunschild and Klopper, "
                          "Theor. Chem. Acc. 131, 1112 (2012))",
            "indices": [int(i) for i in G2_97_INDICES],
            "units": "angstrom",
        },
        "atoms": atoms,
        "molecules": molecules,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traj", default=DEFAULT_TRAJ)
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)
    if not os.path.exists(args.traj):
        parser.error(f"trajectory not found: {args.traj}")
    out = args.out
    if out is None:
        import xcquinox.alec
        out = os.path.join(os.path.dirname(os.path.abspath(
            xcquinox.alec.__file__)), "data", "dfs_pretrain_set.json")
    payload = build(args.traj)
    _write_json_atomic(payload, out)
    sys.stdout.write(
        f"wrote {len(payload['atoms'])} atom(s) + "
        f"{len(payload['molecules'])} molecule(s) to {out}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
