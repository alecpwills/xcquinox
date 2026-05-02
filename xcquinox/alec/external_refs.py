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
