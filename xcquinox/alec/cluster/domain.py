"""xcquinox.alec.cluster.domain: physics tables for the HPC training harness.

The cluster harness extracts step-7 spec-building from the notebook
``notebooks/_build_step7_notebook.py`` into a generic ``cluster/`` package.
This module isolates the parts of that extraction that are physics, not
workflow: the atomic-energy reference table, the kcal/mol -> Ha conversion
constant, the Dick atom-regularizer element set, and the two BH76/IP13
metadata-to-loss-dict extractor functions.

Keeping these here lets the generic ``spec_builder.py`` (a later task) stay
domain-agnostic: it receives a :class:`DomainProfile` bundling these tables
rather than hard-coding them.

Citation policy (hard project requirement): every physical constant in this
module carries either an ``(L)`` literature-citation comment or an ``(E)``
"empirically tuned / no literature value" comment.
"""
from dataclasses import dataclass, field
from typing import Callable


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

# (L) CODATA-2018 value of one Hartree in kcal/mol
KCAL_PER_HA = 627.5094740631


# Chakravorty 1993 exact non-relativistic atomic totals (Ha).
#
# (L) Chakravorty, Gwaltney, Davidson, Parpia, Froese Fischer,
#     Phys. Rev. A 47, 3649 (1993), exact non-relativistic atomic totals.
#
# Used as the atom_energies anchor AND as placeholder target values for
# single-atom MoleculeSpecs (TrainingSpec.validate requires a targets entry
# for every molecule, including atoms).
ATOMIC_ENERGIES_CHAKRAVORTY = {
    'H':  -0.5,        # (L) exact hydrogenic -1/2 Ha (not a Chakravorty datum)
    'C':  -37.845,     # (L) Chakravorty 1993 Table XI (neutral C,  -E(6))
    'N':  -54.5892,    # (L) Chakravorty 1993 Table XI (neutral N,  -E(7))
    'O':  -75.0673,    # (L) Chakravorty 1993 Table XI (neutral O,  -E(8))
    'F':  -99.7339,    # (L) Chakravorty 1993 Table XI (neutral F,  -E(9))
    'Li': -7.4781,     # (L) Chakravorty 1993 Table XI (neutral Li, -E(3) = -7.47806)
                       #     (corrected from the HF limit -7.4327 in a prior fix)
    'Na': -162.2546,   # (L) Chakravorty 1993 Table XI (neutral Na, -E(11))
    'S':  -398.1095,   # (L) Chakravorty 1993 (PRA 47, 3649) Table XI exact
                       #     non-relativistic total (neutral S, -E(16) = -398.110)
    # Heavier elements required by the BH76+W4-11 reaction pool (HCl, PH3,
    # SiH4, and the B/Be/Al W4-11 species). Transcribed verbatim from
    # Chakravorty, Gwaltney, Davidson, Parpia, Froese Fischer, Phys. Rev. A
    # 47, 3649 (1993), Table XI, "Estimated nonrelativistic stationary-
    # point-nucleus atomic energies" (hartrees), neutral-atom diagonal
    # E(Z, Z). Printed precision: 5 dp for Be/B, 3 dp for Z >= 13.
    'Be': -14.66736,   # (L) Chakravorty 1993 Table XI (neutral Be, -E(4))
    'B':  -24.65391,   # (L) Chakravorty 1993 Table XI (neutral B,  -E(5))
    'Al': -242.346,    # (L) Chakravorty 1993 Table XI (neutral Al, -E(13))
    'Si': -289.359,    # (L) Chakravorty 1993 Table XI (neutral Si, -E(14))
    'P':  -341.259,    # (L) Chakravorty 1993 Table XI (neutral P,  -E(15))
    'Cl': -460.148,    # (L) Chakravorty 1993 Table XI (neutral Cl, -E(17))
}


# Dick atom-regularizer element set. The DFS training-point pool only ever
# anchors single-atom MoleculeSpecs for these elements (see
# ``training_points.DICK_ATOM_REGULARIZER_SYMS``); imported below so the value
# stays in lockstep with the canonical definition.
from xcquinox.alec.training_points import (
    DICK_ATOM_REGULARIZER_SYMS as DICK_ATOM_REGULARIZER_SYMS,
)


# Size of the canonical DFS training-point pool: 26 = 21 AE + 3 BH76 + 2 IP13
# (see ``training_points.build_dfs_pool_points``). Hard-coded here as a fixed
# integer so importing this module does NOT pull in the heavy ASE/pyscf
# dependency chain that ``build_dfs_pool_points()`` triggers; the value is
# verified against ``len(build_dfs_pool_points())`` in the test suite.
DFS_POOL_SIZE = 26


# ---------------------------------------------------------------------------
# BH76 / IP13 metadata-to-loss-dict extractors
# ---------------------------------------------------------------------------
# Ported verbatim (logic-preserving) from the notebook helpers
# ``_bh76_meta_to_loss_dict`` / ``_ip13_meta_to_loss_dict`` in
# ``notebooks/_build_step7_notebook.py`` (~lines 557-581). The only change is
# dropping the leading underscore to make them part of the public API.

def bh76_meta_to_loss_dict(tp):
    """BH76 TrainingPoint -> loss-input dict (kcal/mol -> Ha).

    Reads ``metadata['e_rxn_ref']``: the mode-selected reaction energy the
    DFS pool sets (kcal/mol), and converts it to Hartree via
    :data:`KCAL_PER_HA`.

    Parameters
    ----------
    tp : TrainingPoint
        A ``bh76``-kind training point.

    Returns
    -------
    dict
        Loss-input dict with ``name``, ``reactants``, ``products``,
        ``coeffs``, and (when present) the Ha-converted ``e_rxn_ref``.
    """
    md = dict(tp.metadata)
    out = {
        'name':      tp.name,
        'reactants': md.get('reactants', ()),
        'products':  md.get('products', ()),
        'coeffs':    md.get('coeffs', ()),
    }
    eref = md.get('e_rxn_ref')
    if eref is not None:
        out['e_rxn_ref'] = float(eref) / KCAL_PER_HA
    return out


def ip13_meta_to_loss_dict(tp):
    """IP13 TrainingPoint -> loss-input dict (kcal/mol -> Ha).

    Reads ``metadata['ip_ref']`` (kcal/mol) and converts it to Hartree via
    :data:`KCAL_PER_HA`.

    Parameters
    ----------
    tp : TrainingPoint
        An ``ip13``-kind training point.

    Returns
    -------
    dict
        Loss-input dict with ``name``, ``neutral``, ``cation``, and (when
        present) the Ha-converted ``ip_ref``.
    """
    md = dict(tp.metadata)
    out = {
        'name':    tp.name,
        'neutral': md.get('neutral'),
        'cation':  md.get('cation'),
    }
    ipref = md.get('ip_ref')
    if ipref is not None:
        out['ip_ref'] = float(ipref) / KCAL_PER_HA
    return out


# ---------------------------------------------------------------------------
# DomainProfile + registry
# ---------------------------------------------------------------------------

BH76W411_POOL_SIZE = 212   # 216 reactions, 4 identical-name duplicates collapse


def _dfs_pool_builder(cfg):
    """Default pool builder: the 26-point DFS pool (lazy import keeps domain.py
    free of the heavy ASE/pyscf chain until a pool is actually requested).
    ``cfg.ae_as_reactions`` selects the predicted-atom reaction form for the
    21 AE points (same point names, so subset ledgers resolve unchanged)."""
    from xcquinox.alec.training_points import build_dfs_pool_points
    return build_dfs_pool_points(
        bh76_mode=cfg.bh76_mode,
        ae_as_reactions=bool(getattr(cfg, "ae_as_reactions", False)))


def _bh76w411_pool_builder(cfg):
    """Pool builder for the full BH76+W4-11 reaction set (representative-subset
    training).

    The BH76+W4-11 pool attaches reaction energies only (it has no
    transition-state geometries / barrier-height references), so it supports
    only ``bh76_mode='reaction_energy'``. A ``barrier_height`` request is
    rejected loudly rather than silently trained as reaction energy (unlike the
    DFS builder, which honors both modes).
    """
    mode = getattr(cfg, "bh76_mode", "reaction_energy")
    if mode != "reaction_energy":
        raise ValueError(
            f"bh76w411_step7 supports only bh76_mode='reaction_energy', got "
            f"{mode!r}: the BH76+W4-11 reaction pool carries reaction energies, "
            f"not transition-state barrier heights."
        )
    from xcquinox.alec.training_points import build_bh76w411_pool_points
    return build_bh76w411_pool_points()


@dataclass(frozen=True)
class DomainProfile:
    """Bundle of physics tables a generic spec-builder needs.

    The generic ``spec_builder.py`` receives one of these so it stays
    domain-agnostic. ``grid_config.validate_grid_semantics`` reads
    :attr:`pool_size` to bound ``subset_size``.

    Attributes
    ----------
    name : str
        Registry key for this profile.
    atom_energies : dict[str, float]
        Atomic-energy reference table (Ha), the Chakravorty 1993 dict.
    regularize_atom_syms : tuple[str, ...]
        Element symbols whose single-atom MoleculeSpecs are anchored
        (the Dick H/Li set).
    kcal_per_ha : float
        kcal/mol -> Ha conversion constant.
    pool_size : int
        Size of the training-point pool the harness selects subsets from.
    bh76_meta_to_loss_dict : Callable
        BH76 TrainingPoint -> loss-input dict extractor.
    ip13_meta_to_loss_dict : Callable
        IP13 TrainingPoint -> loss-input dict extractor.
    """
    name: str
    atom_energies: dict
    regularize_atom_syms: tuple
    kcal_per_ha: float
    pool_size: int
    bh76_meta_to_loss_dict: Callable = field(default=bh76_meta_to_loss_dict)
    ip13_meta_to_loss_dict: Callable = field(default=ip13_meta_to_loss_dict)
    # pool_builder(cfg) -> list[TrainingPoint]: the trainable pool for this
    # domain. Default = the 26-point DFS pool; alternate domains (e.g.
    # bh76w411_step7) supply their own. Generalizable to any training set.
    pool_builder: Callable = field(default=_dfs_pool_builder)
    # When True, the harness restricts CCSD-reference generation to the species
    # actually present in the loaded subset ledger (training-subset species),
    # passing their geometries directly, instead of the canonical DFS species
    # union. Keeps the preflight feasible for large external pools.
    ccsd_species_from_ledger: bool = False


# Registry of named domain profiles. ``dfs_step7`` is the step-7 DFS profile:
# Chakravorty atomic anchors, Dick H/Li regularizer set, the 26-point DFS pool.
DOMAIN_PROFILES: dict = {
    "dfs_step7": DomainProfile(
        name="dfs_step7",
        atom_energies=dict(ATOMIC_ENERGIES_CHAKRAVORTY),
        regularize_atom_syms=tuple(DICK_ATOM_REGULARIZER_SYMS),
        kcal_per_ha=KCAL_PER_HA,
        pool_size=DFS_POOL_SIZE,
        bh76_meta_to_loss_dict=bh76_meta_to_loss_dict,
        ip13_meta_to_loss_dict=ip13_meta_to_loss_dict,
    ),
    # Representative-subset training on the full BH76+W4-11 reaction set: same
    # Chakravorty anchors / Dick regularizer / loss extractors, but the pool is
    # the 212-reaction BH76+W4-11 set and CCSD refs are scoped to the ledger's
    # training-subset species.
    "bh76w411_step7": DomainProfile(
        name="bh76w411_step7",
        atom_energies=dict(ATOMIC_ENERGIES_CHAKRAVORTY),
        regularize_atom_syms=tuple(DICK_ATOM_REGULARIZER_SYMS),
        kcal_per_ha=KCAL_PER_HA,
        pool_size=BH76W411_POOL_SIZE,
        bh76_meta_to_loss_dict=bh76_meta_to_loss_dict,
        ip13_meta_to_loss_dict=ip13_meta_to_loss_dict,
        pool_builder=_bh76w411_pool_builder,
        ccsd_species_from_ledger=True,
    ),
}


def get_domain_profile(name: str) -> DomainProfile:
    """Look up a :class:`DomainProfile` by registry name.

    Parameters
    ----------
    name : str
        Registry key, e.g. ``'dfs_step7'``.

    Returns
    -------
    DomainProfile

    Raises
    ------
    ValueError
        If ``name`` is not a registered profile.
    """
    try:
        return DOMAIN_PROFILES[name]
    except KeyError:
        raise ValueError(
            f"Unknown domain profile {name!r}; "
            f"registered profiles: {sorted(DOMAIN_PROFILES)}."
        ) from None
