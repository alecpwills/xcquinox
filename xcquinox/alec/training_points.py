"""Mixed training-point pool for Dick 2021-style subset selection.

A :class:`TrainingPoint` represents one unit of training signal, either
an atomization-energy compound, a BH76 reaction barrier, or an IP13
ionization potential. ``build_dfs_pool_points()`` returns the flat list
of 26 = 21 + 3 + 2 points. ``select_subset`` then picks ``r`` points from
this mixed pool, and the resulting ``spec.molecules`` is the deduplicated
union of all participating species across the chosen points (no implicit
augmentation; every species in the spec comes from a chosen point).

Per spec design:
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
# in spec.molecules: that gets out of hand once Na, Cl, etc. enter via
# chosen compounds).
DICK_ATOM_REGULARIZER_SYMS: tuple[str, ...] = ("H", "Li")


@dataclass
class TrainingPoint:
    """One unit of Dick-pool training signal.

    Attributes
    ----------
    kind : 'ae' | 'bh76' | 'ip13'
    name : unique identifier (e.g. 'CH4', 'OH+N2_to_H+N2O', 'Li_IP').
    species : tuple of ASE Atoms, every molecule this point requires for
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
                    f"info['name']: every species must be uniquely named "
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
    are NOT given separate MoleculeSpec entries, the AE channel's
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


def _ae_reaction_point_from_atoms(compound: Atoms) -> TrainingPoint:
    """AE point in PREDICTED-ATOM reaction form (``bh76``-kind).

    Instead of the fixed-anchor AE channel (``_ae_from_atoms`` against the
    Chakravorty table), the atomization energy becomes a reaction the L5
    BH76 channel trains with the NETWORK'S OWN atom energies:
    ``AE = Σ n_Z·E_NN(Z) - E_NN(mol)`` via ``reactants=(mol,)``,
    ``products=(elements...)``, ``coeffs=(-1, n_Z...)``. This is the form
    both dpyscf (Dick & Fernandez-Serra 2021: atoms enter the AE assembly
    as predicted energies, their fixed totals never enter the loss,
    E_weight=0) and our converged bh76w411_step7 runs use; the dfs_step7
    forensics traced the Na2 blowup to the fixed-anchor relative AE form.

    The point NAME equals ``_ae_point_from_atoms``'s, so name-keyed subset
    ledgers (subset_index_log.json) resolve identically -- the same JSD
    subsets train under either form. Species gain one neutral ground-state
    atom per ELEMENT (multiplicity lives in the coeffs). ``spec_builder``
    injects this point's ``e_rxn_ref`` into ``ae_ref_kcalmol`` (so
    ``build_targets`` gives the compound a REAL target for eval scoring) AND
    forces its name into ``aux_only_names`` -- so ``_ae_losses`` zeroes its
    fixed-anchor AE channel and it trains via the BH76 reaction channel, while
    ``_training_groups`` SKIPS its redundant ``ae:<name>`` group so it is
    density/vxc-supervised exactly once (via its bh76 group). ``e_rxn_ref`` is
    kept in kcal/mol (``bh76_meta_to_loss_dict`` converts to Ha).
    """
    name = compound.info.get("dfs_hill", compound.get_chemical_formula())
    cmp = compound.copy()
    cmp.info["name"] = name
    counts: dict = {}
    for sym in compound.get_chemical_symbols():
        counts[sym] = counts.get(sym, 0) + 1
    elements = sorted(counts)
    species: list = [cmp]
    for sym in elements:
        species.append(_atom_anchor_atoms(sym))
    return TrainingPoint(
        kind="bh76",
        name=name,
        species=tuple(species),
        metadata={
            "reactants": (name,),
            "products": tuple(elements),
            "coeffs": (-1.0, *(float(counts[s]) for s in elements)),
            "e_rxn_ref": compound.info.get("ae_kcalmol"),
            "ae_form": "predicted_atom_reaction",
            "ae_source": compound.info.get("ae_source"),
            "ae_name": compound.info.get("ae_name"),
            "source": compound.info.get("ae_source"),
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

    - ``"reaction_energy"`` (DEFAULT), the true reaction energy ΔE
      (GMTKN55-BH76RC). The loss is trained against E(products) -
      E(reactants), matching Dick & Fernandez-Serra 2021 (their
      training set had no transition-state geometries). The reference
      MUST be a reaction energy, since barrier heights cannot be
      reproduced by the reactant -> product stoichiometry.
    - ``"barrier_height"`` (opt-in), the forward barrier height. A
      true forward barrier is ``E(TS) - E(reactants)``, so each
      reaction must additionally supply a transition-state geometry
      (``rxn["ts_species"]``). Those geometries are not yet staged in
      the repo; this path raises a clear, actionable error until they
      are (the toggle is fully wired, only the data is missing).
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
    # reaction_energy_ref are kept alongside for provenance. A missing
    # source key must KeyError loudly rather than silently fall back.
    if bh76_mode == "reaction_energy":
        e_rxn_ref = rxn["reaction_energy_ref"]
    else:  # "barrier_height": already validated above (TS present)
        e_rxn_ref = rxn["barrier_ref"]
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
            "barrier_ref": rxn["barrier_ref"],
            "reaction_energy_ref": rxn["reaction_energy_ref"],
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
    ae_as_reactions: bool = False,
) -> list[TrainingPoint]:
    """Return the flat list of 26 = 21 AE + 3 BH76 + 2 IP13 training points
    that ``select_subset`` should pick from.

    Each point carries (a) its participating species INCLUDING atom anchors
    needed for that point (design choice b), and (b) its kind-specific
    metadata for the corresponding loss channel.

    Parameters
    ----------
    ae_as_reactions : bool, default False
        When True, the 21 AE points are built in PREDICTED-ATOM reaction
        form (:func:`_ae_reaction_point_from_atoms`, ``bh76``-kind) instead
        of the fixed-anchor AE form; point names (and thus subset-ledger
        resolution) are identical either way.
    bh76_mode : {'reaction_energy', 'barrier_height'}, default 'reaction_energy'
        Selects what the 3 BH76 training points are trained against.

        - ``'reaction_energy'`` (DEFAULT, "dick default"), BH76 points
          carry the true reaction energy ΔE (GMTKN55-BH76RC) as
          ``e_rxn_ref``. The BH76 loss term computes
          ``Σ coeffs·E = E(products) - E(reactants)``, so the reference
          MUST be a reaction energy, this is the correct behaviour and
          matches Dick & Fernandez-Serra 2021.
        - ``'barrier_height'`` (opt-in), BH76 points carry the forward
          barrier height as ``e_rxn_ref``. A true forward barrier is
          ``E(TS) - E(reactants)``, which requires a transition-state
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
        points.append(_ae_reaction_point_from_atoms(compound)
                      if ae_as_reactions else _ae_point_from_atoms(compound))
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
    list: every entry comes from one of the chosen points; nothing is
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


def _molspec_to_atoms(spec) -> Atoms:
    """Build an ASE ``Atoms`` (carrying ``info`` name/charge/spin) from a
    ``full_benchmark_pools`` ``MoleculeSpec``. ``spec.atom`` is a pyscf-format
    ``"Sym x y z; ..."`` string in ANGSTROM."""
    symbols: list[str] = []
    positions: list[list[float]] = []
    for tok in spec.atom.replace("\n", ";").split(";"):
        tok = tok.strip()
        if not tok:
            continue
        parts = tok.split()
        symbols.append(parts[0])
        positions.append([float(parts[1]), float(parts[2]), float(parts[3])])
    a = Atoms(symbols=symbols, positions=positions)
    a.info["name"] = spec.name
    a.info["charge"] = int(getattr(spec, "charge", 0) or 0)
    a.info["spin"] = int(getattr(spec, "spin", 0) or 0)
    return a


def build_reaction_pool_points(reactions, atoms_by_name) -> list[TrainingPoint]:
    """Build reaction-energy ``bh76``-kind TrainingPoints from an ARBITRARY set
    of reactions + a ``name -> ASE Atoms`` map, generalizable to any benchmark
    or custom training set, not just BH76+W4-11.

    Each ``reaction`` dict must carry ``name``, ``reactants``/``products``
    (species-name lists), ``coeffs``, and ``reaction_energy_ref`` (kcal/mol,
    ``bh76_meta_to_loss_dict`` divides by ``KCAL_PER_HA``). ``atoms_by_name``
    resolves every species name to an ASE ``Atoms`` carrying ``info``
    name/charge/spin (e.g. via :func:`_molspec_to_atoms`). Each reaction becomes
    one point whose ``name`` IS the reaction name (so a name-keyed subset ledger
    resolves directly), and whose ``species`` are its participating molecules;
    the L5 BH76 channel trains ``Σ coeffs·E`` against the reference while the
    vxc/rho channels train each species against its CCSD reference density (the
    harness preflight generates those for the training-subset species).

    Reactions are deduplicated by name, identical-name entries collapse to
    one point (the harness resolves training points by name, which must be
    unique). Raises ``KeyError`` if a species name is absent from
    ``atoms_by_name`` (fail loud rather than train on a missing molecule)."""
    points: list[TrainingPoint] = []
    seen_names: set[str] = set()
    for rxn in reactions:
        name = rxn["name"]
        if name in seen_names:
            continue
        seen_names.add(name)
        species: list[Atoms] = []
        seen_sp: set[str] = set()
        for sp_name in list(rxn["reactants"]) + list(rxn["products"]):
            if sp_name in seen_sp:
                continue
            seen_sp.add(sp_name)
            species.append(atoms_by_name[sp_name].copy())
        points.append(TrainingPoint(
            kind="bh76",
            name=name,
            species=tuple(species),
            metadata={
                "reactants": tuple(rxn["reactants"]),
                "products": tuple(rxn["products"]),
                "coeffs": tuple(rxn["coeffs"]),
                "e_rxn_ref": rxn["reaction_energy_ref"],
                "source": rxn.get("source"),
                "source_pool": rxn.get("source_pool"),
            },
        ))
    return points


def build_bh76w411_pool_points() -> list[TrainingPoint]:
    """Trainable pool = the full BH76+W4-11 reaction set (212 unique reactions;
    4 identical-name duplicates in the benchmark collapse).

    A thin wrapper over the generalizable :func:`build_reaction_pool_points`:
    pulls the reactions + geometries from ``full_benchmark_pools`` and resolves
    species via :func:`_molspec_to_atoms`. Reaction names match the
    representative-subset ledger's ``point_names``."""
    from xcquinox.alec.full_benchmark_pools import load_full_held_out_pools

    full_specs, full_rxns = load_full_held_out_pools(basis="def2-svp",
                                                     grid_level=1)
    atoms_by_name = {name: _molspec_to_atoms(spec)
                     for name, spec in full_specs.items()}
    return build_reaction_pool_points(full_rxns, atoms_by_name)
