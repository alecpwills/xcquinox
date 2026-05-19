"""Mixed training-point pool for Dick 2021-style subset selection.

A :class:`TrainingPoint` represents one unit of training signal — either
an atomization-energy compound, a BH76 reaction barrier, or an IP13
ionization potential. ``build_dfs_pool_points()`` returns the flat list
of 26 = 21 + 3 + 2 points. ``select_subset`` then picks ``r`` points from
this mixed pool, and the resulting ``spec.molecules`` is the deduplicated
union of all participating species across the chosen points (no implicit
augmentation; every species in the spec comes from a chosen point).

Per spec design (2026-05-07):
- Each TrainingPoint carries its own atom anchors EXPLICITLY in
  ``species`` (design choice "b"). Spec builder dedupes by
  ``(name, charge, spin)``.
- Descriptor for a multi-species point = concatenation of all
  participating species' grid descriptors (design choice "a").
- ``regularize_atom_syms`` for the loss is Dick's list, ``("H", "Li")``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ase import Atoms

from .dfs_pool import (
    BH76_MODES,
    DFS_BH76_REACTIONS,
    DFS_IP13_PAIRS,
    ATOMIC_GROUND_STATE_SPIN,
    build_dfs_pool,
    make_atom_atoms,
)


# Dick & Fernandez-Serra 2021 SI §II atom regularizer set.
# DFS_ATOM_REFS in dfs_pool.py:394 declares H + Li as the canonical
# atomic-density references; this is the same list. Used by the loss to
# scope `_atomic_reg` (which by default would regularize EVERY atom anchor
# in spec.molecules — that gets out of hand once Na, Cl, etc. enter via
# chosen compounds).
DICK_ATOM_REGULARIZER_SYMS: tuple[str, ...] = ("H", "Li")


@dataclass
class TrainingPoint:
    """One unit of Dick-pool training signal.

    Attributes
    ----------
    kind : 'ae' | 'bh76' | 'ip13'
    name : unique identifier (e.g. 'CH4', 'OH+N2_to_H+N2O', 'Li_IP').
    species : tuple of ASE Atoms — every molecule this point requires for
        the loss to evaluate. For 'ae': (compound, atom_anchors...).
        For 'bh76': (reactants..., products..., atom_anchors...).
        For 'ip13': (neutral, cation).
        Each Atoms carries info['name'], info['spin'], info['charge'].
    metadata : kind-specific data (ae_kcalmol / e_rxn_ref / ip_ref / coeffs / ...).
    """

    kind: str
    name: str
    species: tuple
    metadata: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in ("ae", "bh76", "ip13"):
            raise ValueError(
                f"TrainingPoint.kind must be 'ae'|'bh76'|'ip13', got {self.kind!r}"
            )
        if not self.species:
            raise ValueError(
                f"TrainingPoint(name={self.name!r}): species must be non-empty"
            )
        for s in self.species:
            if not isinstance(s, Atoms):
                raise TypeError(
                    f"TrainingPoint(name={self.name!r}): species entries "
                    f"must be ASE Atoms, got {type(s).__name__}"
                )
            if "name" not in s.info:
                raise ValueError(
                    f"TrainingPoint(name={self.name!r}): species without "
                    f"info['name'] — every species must be uniquely named "
                    f"so the spec-builder can dedupe across points."
                )


def _atom_anchor_atoms(sym: str) -> Atoms:
    """Build a single-atom ASE Atoms with NIST ground-state spin attached.

    The result has info['name']=sym, info['spin']=NIST 2S, info['charge']=0.
    Used as the atom-anchor entry inside a TrainingPoint's species tuple.
    """
    a = make_atom_atoms(sym)
    a.info["dfs_hill"] = sym
    return a


def _ae_point_from_atoms(compound: Atoms) -> TrainingPoint:
    """Build an AE TrainingPoint: compound + atom anchors ONLY for the
    Dick-regularized elements (``DICK_ATOM_REGULARIZER_SYMS = ('H', 'Li')``)
    that appear in the compound. Other element symbols (C, N, O, F, ...)
    are NOT given separate MoleculeSpec entries — the AE channel's
    ``_ae_from_atoms`` formula reads from the fixed ``atom_energies`` dict
    and does not require those atoms in the spec.
    """
    name = compound.info.get("dfs_hill", compound.get_chemical_formula())
    cmp = compound.copy()
    cmp.info["name"] = name
    elements = sorted(set(compound.get_chemical_symbols()))
    species: list = [cmp]
    for sym in elements:
        if sym in DICK_ATOM_REGULARIZER_SYMS and sym != name:
            species.append(_atom_anchor_atoms(sym))
    return TrainingPoint(
        kind="ae",
        name=name,
        species=tuple(species),
        metadata={
            "ae_kcalmol": compound.info.get("ae_kcalmol"),
            "ae_source": compound.info.get("ae_source"),
            "ae_name": compound.info.get("ae_name"),
        },
    )


def _bh76_point_from_dict(
    rxn: dict,
    *,
    atoms_by_name: dict,
    bh76_mode: str = "reaction_energy",
) -> TrainingPoint:
    """Build a BH76 TrainingPoint from one DFS_BH76_REACTIONS dict.

    species = (every reactant + every product, deduped by name) +
              atom anchors for every element appearing in any species.

    ``bh76_mode`` selects which reference value is attached as the
    point's ``e_rxn_ref`` metadata (consumed mode-agnostically by
    ``losses._rxn_residual_term`` as ``Σ coeffs·E``):

    - ``"reaction_energy"`` (DEFAULT) — the true reaction energy ΔE
      (GMTKN55-BH76RC). The loss is trained against E(products) −
      E(reactants), matching Dick & Fernandez-Serra 2021 (their
      training set had no transition-state geometries). This is the
      bug fix: the historical ``e_rxn_ref`` values were barrier
      heights, which the reactant→product stoichiometry cannot
      reproduce.
    - ``"barrier_height"`` (opt-in) — the forward barrier height. A
      true forward barrier is ``E(TS) − E(reactants)``, so each
      reaction must additionally supply a transition-state geometry
      (``rxn["ts_species"]``). Those geometries are not yet staged in
      the repo; this path raises a clear, actionable error until they
      are (the toggle is fully wired — only the data is missing).
    """
    if bh76_mode not in BH76_MODES:
        raise ValueError(
            f"Unknown bh76_mode {bh76_mode!r}; expected one of {BH76_MODES}."
        )
    if bh76_mode == "barrier_height":
        if rxn.get("ts_species") is None:
            raise NotImplementedError(
                "bh76_mode='barrier_height' requires transition-state "
                "geometries for the 3 BH76 reactions, which are not yet "
                "staged in dfs_pool.py (every DFS_BH76_REACTIONS entry has "
                "ts_species=None). Supply the transition-state geometries "
                "(populate the 'ts_species' slot) or use the default "
                "bh76_mode='reaction_energy'."
            )
    species_names: list[str] = list(rxn["reactants"]) + list(rxn["products"])
    species: list = []
    seen_names: set[str] = set()
    elements: set[str] = set()
    species_spins = rxn.get("species_spins", {})
    species_charges = rxn.get("species_charges", {})
    for sp_name in species_names:
        if sp_name in seen_names:
            continue
        seen_names.add(sp_name)
        if sp_name in atoms_by_name:
            a = atoms_by_name[sp_name].copy()
        elif len(sp_name) <= 2 and sp_name.isalpha():
            # Single-atom reactant/product (H, F, O, ...).
            a = make_atom_atoms(
                sp_name,
                charge=int(species_charges.get(sp_name, 0)),
                spin=int(species_spins.get(sp_name, ATOMIC_GROUND_STATE_SPIN[sp_name])),
            )
        else:
            raise RuntimeError(
                f"BH76 reaction {rxn['name']!r}: species {sp_name!r} not in "
                f"g2_97 atoms_by_name and not recognized as an atomic symbol."
            )
        a.info["name"] = sp_name
        a.info["spin"] = int(species_spins.get(sp_name, a.info.get("spin", 0)))
        a.info["charge"] = int(species_charges.get(sp_name, 0))
        species.append(a)
        elements.update(a.get_chemical_symbols())
    # Atom anchors ONLY for Dick-regularized elements (H, Li) that
    # appear in any species but aren't already a single-atom reactant
    # or product. C, N, O, F, ... do NOT get separate MoleculeSpecs.
    for sym in sorted(elements):
        if sym in DICK_ATOM_REGULARIZER_SYMS and sym not in seen_names:
            species.append(_atom_anchor_atoms(sym))
            seen_names.add(sym)
    # Mode selects which reference the loss is trained against. The
    # loss reads only metadata["e_rxn_ref"]; barrier_ref /
    # reaction_energy_ref are kept alongside for provenance.
    if bh76_mode == "reaction_energy":
        e_rxn_ref = rxn.get("reaction_energy_ref", rxn.get("e_rxn_ref"))
    else:  # "barrier_height" — already validated above (TS present)
        e_rxn_ref = rxn.get("barrier_ref", rxn.get("e_rxn_ref"))
    return TrainingPoint(
        kind="bh76",
        name=rxn["name"],
        species=tuple(species),
        metadata={
            "reactants": tuple(rxn["reactants"]),
            "products": tuple(rxn["products"]),
            "coeffs": tuple(rxn["coeffs"]),
            "e_rxn_ref": e_rxn_ref,
            "bh76_mode": bh76_mode,
            "barrier_ref": rxn.get("barrier_ref"),
            "reaction_energy_ref": rxn.get("reaction_energy_ref"),
            "source": rxn.get("source"),
        },
    )


def _ip13_point_from_dict(pair: dict) -> TrainingPoint:
    """Build an IP13 TrainingPoint: neutral atom + cation atom."""
    neutral = make_atom_atoms(
        pair["neutral"].rstrip("+"),
        charge=int(pair.get("neutral_charge", 0)),
        spin=int(pair["neutral_spin"]),
    )
    neutral.info["name"] = pair["neutral"]
    cation = make_atom_atoms(
        pair["cation"].rstrip("+"),
        charge=int(pair.get("cation_charge", 1)),
        spin=int(pair["cation_spin"]),
    )
    cation.info["name"] = pair["cation"]
    return TrainingPoint(
        kind="ip13",
        name=pair["name"],
        species=(neutral, cation),
        metadata={
            "neutral": pair["neutral"],
            "cation": pair["cation"],
            "ip_ref": pair.get("ip_ref"),
            "source": pair.get("source"),
        },
    )


def build_dfs_pool_points(
    bh76_mode: str = "reaction_energy",
) -> list[TrainingPoint]:
    """Return the flat list of 26 = 21 AE + 3 BH76 + 2 IP13 training points
    that ``select_subset`` should pick from.

    Each point carries (a) its participating species INCLUDING atom anchors
    needed for that point (design choice b), and (b) its kind-specific
    metadata for the corresponding loss channel.

    Parameters
    ----------
    bh76_mode : {'reaction_energy', 'barrier_height'}, default 'reaction_energy'
        Selects what the 3 BH76 training points are trained against.

        - ``'reaction_energy'`` (DEFAULT, "dick default") — BH76 points
          carry the true reaction energy ΔE (GMTKN55-BH76RC) as
          ``e_rxn_ref``. The BH76 loss term computes
          ``Σ coeffs·E = E(products) − E(reactants)``, so the reference
          MUST be a reaction energy — this is the correct, bug-fixed
          behaviour and matches Dick & Fernandez-Serra 2021.
        - ``'barrier_height'`` (opt-in) — BH76 points carry the forward
          barrier height as ``e_rxn_ref``. A true forward barrier is
          ``E(TS) − E(reactants)``, which requires a transition-state
          geometry per reaction. Those geometries are NOT yet staged in
          dfs_pool.py, so selecting this mode raises NotImplementedError
          until they are supplied.

    Raises
    ------
    ValueError
        If ``bh76_mode`` is not a recognized value.
    NotImplementedError
        If ``bh76_mode='barrier_height'`` is selected while the BH76
        transition-state geometries are not yet staged.
    """
    if bh76_mode not in BH76_MODES:
        raise ValueError(
            f"Unknown bh76_mode {bh76_mode!r}; expected one of {BH76_MODES}."
        )
    from ase.io import read as _ase_read
    from .dfs_pool import _g297_traj_path
    pool = build_dfs_pool()
    # BH76 reactants/products may be molecules NOT in the AE pool (CH3, CH4,
    # HF, etc.); load the full g2_97 trajectory so we can resolve them.
    g297 = _ase_read(str(_g297_traj_path()), ":")
    atoms_by_name: dict = {a.get_chemical_formula(): a for a in g297}
    # AE pool entries take precedence (they have richer info attached).
    for a in pool["ae_molecules"]:
        atoms_by_name[a.info.get("dfs_hill", a.get_chemical_formula())] = a
    points: list[TrainingPoint] = []
    for compound in pool["ae_molecules"]:
        points.append(_ae_point_from_atoms(compound))
    for rxn in DFS_BH76_REACTIONS:
        points.append(
            _bh76_point_from_dict(
                rxn, atoms_by_name=atoms_by_name, bh76_mode=bh76_mode
            )
        )
    for pair in DFS_IP13_PAIRS:
        points.append(_ip13_point_from_dict(pair))
    return points


def species_union_from_points(
    points: Sequence[TrainingPoint],
) -> list[Atoms]:
    """Return the deduplicated union of species across the chosen points.

    Dedupe key = (name, charge, spin). This is the ``spec.molecules``
    list — every entry comes from one of the chosen points; nothing is
    forcibly added.
    """
    seen: dict[tuple, Atoms] = {}
    for tp in points:
        for s in tp.species:
            key = (
                s.info["name"],
                int(s.info.get("charge", 0)),
                int(s.info.get("spin", 0)),
            )
            if key not in seen:
                seen[key] = s
    return list(seen.values())
