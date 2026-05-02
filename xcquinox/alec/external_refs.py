"""Step-7 CCSD reference pre-compute pipeline.

Generates per-species external_data_path .npz files containing CCSD
reference density matrix, grid density, and OEP-inverted V_xc for the
union of training + held-out probe + HBPT species.

Pipeline stages (each individually cached via np.savez_compressed):
  1. SCF  → _intermediates/<name>_scf.npz   (MO coeffs, DM, S)
  2. CCSD → _intermediates/<name>_ccsd.npz  (CC density matrix + rho)
  3. OEP  → <name>.npz                       (vxc_ref + dm_target +
                                              rho_ref_grid + provenance)

Reuses step-6 cells 12-13 OEP-cascade pattern verbatim
(_build_step6_notebook.py:728-768, 843-877). 2-tier: svp-jkfit primary
(reg=1e-4, max_iter=500), def2-tzvp-jkfit fallback (reg=1e-4,
max_iter=1000). On both-tier failure raises RuntimeError.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SpeciesEntry:
    """Canonical pre-compute species record.

    Dedup key is (name, charge, spin) — `Li` and `Li+` are distinct
    entries with different charges.
    """
    name: str
    charge: int
    spin: int  # PySCF 2S = N_α − N_β convention
    source: str  # one of "dfs_ae", "dfs_atom", "bh76", "ip13",
                 # "probe_a", "probe_b", "probe_c", "probe_d",
                 # "probe_atom_ref", "hbpt"


def build_species_union() -> list[SpeciesEntry]:
    """Assemble the canonical species set requiring CCSD references.

    Iterates DFS pool, BH76 reactions, IP13 pairs, atom refs (training +
    probe-induced), Probe A/B/C/D, and HBPT pairs.  De-duplicates on
    (name, charge, spin).  Returns a deterministic list (sorted by name
    then charge then spin) so the iteration order is reproducible across
    runs.
    """
    from xcquinox.alec import dfs_pool, eval_probes
    from xcquinox.alec.subset_selection import _make_hb_atoms, _make_pt_atoms
    seen: dict[tuple[str, int, int], SpeciesEntry] = {}

    def _add(name: str, charge: int, spin: int, source: str) -> None:
        key = (name, charge, spin)
        if key not in seen:
            seen[key] = SpeciesEntry(
                name=name, charge=charge, spin=spin, source=source,
            )

    # DFS AE molecules
    pool = dfs_pool.build_dfs_pool()
    for at in pool["ae_molecules"]:
        _add(
            at.info["dfs_hill"],
            int(at.info.get("charge", 0)),
            int(at.info["spin"]),
            "dfs_ae",
        )
    # DFS atom refs (H, Li)
    for at in pool["atom_refs"]:
        sym = at.info["name"]
        _add(sym, int(at.info.get("charge", 0)), int(at.info["spin"]),
             "dfs_atom")

    # BH76 species (need atom-level dispatch)
    for rxn in pool["bh76_reactions"]:
        spins = rxn.get("species_spins", {})
        charges = rxn.get("species_charges", {})
        for sp in (*rxn["reactants"], *rxn["products"]):
            _add(sp, int(charges.get(sp, 0)), int(spins.get(sp, 0)), "bh76")

    # IP13 neutrals + cations
    for pair in pool["ip13_pairs"]:
        _add(pair["neutral"], int(pair["neutral_charge"]),
             int(pair["neutral_spin"]), "ip13")
        _add(pair["cation"], int(pair["cation_charge"]),
             int(pair["cation_spin"]), "ip13")

    # Probe sets — read entries from eval_probes
    for probe_name in eval_probes.ALL_PROBES:
        kind = eval_probes.PROBE_KIND[probe_name]
        entries = eval_probes.ALL_PROBES[probe_name]
        if kind == "ae":
            for entry in entries:
                _add(entry["hill"], int(entry.get("charge", 0)),
                     int(entry["spin"]), f"probe_{probe_name.split('_')[1]}")
        else:  # bh76
            for rxn in entries:
                spins = rxn.get("species_spins", {})
                charges = rxn.get("species_charges", {})
                for sp in (*rxn["reactants"], *rxn["products"]):
                    _add(sp, int(charges.get(sp, 0)),
                         int(spins.get(sp, 0)), f"probe_{probe_name.split('_')[1]}")

    # Probe-induced atom refs (S, Cl, P, Si, Be) for atom_energies anchor
    from xcquinox.alec.dfs_pool import ATOMIC_GROUND_STATE_SPIN
    for sym in ("S", "Cl", "P", "Si", "Be"):
        _add(sym, 0, ATOMIC_GROUND_STATE_SPIN[sym], "probe_atom_ref")

    # HBPT water-dimer pairs
    hb = _make_hb_atoms()
    pt = _make_pt_atoms()
    _add(hb.info["dfs_hill"], int(hb.info["charge"]),
         int(hb.info["spin"]), "hbpt")
    _add(pt.info["dfs_hill"], int(pt.info["charge"]),
         int(pt.info["spin"]), "hbpt")

    return sorted(seen.values(), key=lambda s: (s.name, s.charge, s.spin))


def resolve_geometry(spec: SpeciesEntry):
    """Build an ASE Atoms object for a SpeciesEntry.

    Strategy by source:
      - dfs_ae: lookup by Hill formula in g2_97.traj
      - dfs_atom / bh76 (single-letter or two-letter symbol):
        bare atom at origin
      - ip13: bare atom (cation = bare atom with charge+1)
      - probe_a / probe_b / probe_d (compound):
        lookup by Hill formula in g2_97.traj OR pull from
        eval_probes.build_probe_pool's output entries
      - probe_c (BH76 species): same dispatch as bh76
      - hbpt: call _make_hb_atoms / _make_pt_atoms
    """
    from ase import Atoms
    from xcquinox.alec.dfs_pool import _g297_traj_path
    from xcquinox.alec.subset_selection import _make_hb_atoms, _make_pt_atoms
    from ase.io import read as ase_read

    if spec.source == "hbpt":
        atoms = _make_hb_atoms() if spec.name == "HBWD" else _make_pt_atoms()
        return atoms

    # Atomic species: name is a 1-2 letter symbol (or symbol+"+" for cations)
    sym = spec.name.rstrip("+")
    if len(sym) <= 2 and sym.isalpha() and spec.source in (
        "dfs_atom", "bh76", "ip13", "probe_atom_ref", "probe_c",
    ):
        atoms = Atoms(sym, positions=[(0.0, 0.0, 0.0)])
        atoms.info["name"] = spec.name
        atoms.info["charge"] = spec.charge
        atoms.info["spin"] = spec.spin
        return atoms

    # Compound species: try g2_97.traj first
    traj = ase_read(str(_g297_traj_path()), ":")
    by_hill = {a.get_chemical_formula(): a for a in traj}
    if spec.name in by_hill:
        atoms = by_hill[spec.name].copy()
        atoms.info["dfs_hill"] = spec.name
        atoms.info["charge"] = spec.charge
        atoms.info["spin"] = spec.spin
        return atoms

    # Probe species not in g2_97: pull from eval_probes.build_probe_pool.
    # ``pool["entries"]`` is list[dict] (raw PROBE_* entries); ``pool["molecules"]``
    # is the corresponding list[ASE Atoms] with at.info["name"] set by
    # eval_probes._attach_info.  Match by the dict's "name" against
    # at.info["name"] (T2 spec-review fix — earlier draft iterated entries
    # as if they were Atoms, which is unreachable today but would crash
    # if a probe AE molecule were ever absent from g2_97.traj).
    from xcquinox.alec import eval_probes
    for probe_name in eval_probes.ALL_PROBES:
        if eval_probes.PROBE_KIND[probe_name] != "ae":
            continue
        for entry in eval_probes.ALL_PROBES[probe_name]:
            if entry["hill"] == spec.name:
                pool = eval_probes.build_probe_pool(probe_name)
                for at in pool["molecules"]:
                    if at.info.get("name") == entry["name"]:
                        a = at.copy()
                        a.info["charge"] = spec.charge
                        a.info["spin"] = spec.spin
                        return a
    raise KeyError(
        f"Could not resolve geometry for SpeciesEntry(name={spec.name!r}, "
        f"charge={spec.charge}, spin={spec.spin}, source={spec.source!r})"
    )


def run_scf_with_cache(
    spec: SpeciesEntry,
    atoms,
    *,
    cache_dir,
    basis: str = "def2-svp",
    grid_level: int = 1,
) -> dict:
    """Stage 1: PBE SCF with on-disk cache (np.savez_compressed).

    Returns dict with keys: dm, mo_coeff, mo_occ, mo_energy, S,
    spin_unrestricted, n_ao, n_grid.

    Cache layout:
      <cache_dir>/_intermediates/<name>_scf.npz
    """
    import numpy as np
    from pathlib import Path
    from pyscf import dft, gto

    inter = Path(cache_dir) / "_intermediates"
    inter.mkdir(parents=True, exist_ok=True)
    cache_path = inter / f"{spec.name}_scf.npz"

    if cache_path.is_file():
        with np.load(cache_path, allow_pickle=False) as z:
            return {
                "dm": np.asarray(z["dm"]),
                "mo_coeff": np.asarray(z["mo_coeff"]),
                "mo_occ": np.asarray(z["mo_occ"]),
                "mo_energy": np.asarray(z["mo_energy"]),
                "S": np.asarray(z["S"]),
                "spin_unrestricted": bool(z["spin_unrestricted"]),
                "n_ao": int(z["n_ao"]),
                "n_grid": int(z["n_grid"]),
            }

    coords = atoms.get_positions()
    syms = atoms.get_chemical_symbols()
    atom_lines = [(s, tuple(coords[i])) for i, s in enumerate(syms)]
    mol = gto.M(atom=atom_lines, basis=basis, charge=spec.charge,
                spin=spec.spin, unit="angstrom", verbose=0)

    is_uks = spec.spin > 0
    mf = dft.UKS(mol) if is_uks else dft.RKS(mol)
    mf.xc = "pbe"
    mf.grids.level = grid_level
    mf.kernel()

    np.savez_compressed(
        cache_path,
        dm=np.asarray(mf.make_rdm1()),
        mo_coeff=np.asarray(mf.mo_coeff),
        mo_occ=np.asarray(mf.mo_occ),
        mo_energy=np.asarray(mf.mo_energy),
        S=np.asarray(mf.get_ovlp()),
        spin_unrestricted=np.array(is_uks),
        n_ao=np.array(mol.nao),
        n_grid=np.array(mf.grids.weights.size),
    )
    return {
        "dm": np.asarray(mf.make_rdm1()),
        "mo_coeff": np.asarray(mf.mo_coeff),
        "mo_occ": np.asarray(mf.mo_occ),
        "mo_energy": np.asarray(mf.mo_energy),
        "S": np.asarray(mf.get_ovlp()),
        "spin_unrestricted": bool(is_uks),
        "n_ao": int(mol.nao),
        "n_grid": int(mf.grids.weights.size),
    }
