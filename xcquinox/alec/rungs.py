"""Jacob's-ladder rung taxonomy derived from the architecture registry.

Library-side single source of truth for classifying a registered
architecture by its physics ingredients: the meta-GGA iso-orbital
descriptor (``metagga`` descriptor / ``meta_gga`` flag) and the rung-3.5
localized-DM occupancy descriptors (any ``rung35*`` descriptor name).
Consumers hold registry NAMES (sweep cells, run validation, figure
labels), so :func:`rung_of` takes the name and resolves it through
``get_architecture``; an unregistered name raises ``KeyError`` rather
than guessing from name tokens. The analysis-side ``arch_style`` module
imports this taxonomy and adds a token fallback only for legacy
unregistered display names, plus the styling.

The per-rung SCF seed policy lives here as well
(:func:`seed_xc_for_arch`) so the training-spec builder, run validation,
and the figure layer agree on one predicate by construction.
"""
from __future__ import annotations

from typing import Dict, Tuple

RUNG_GGA = "GGA"
RUNG_MGGA = "meta-GGA"
RUNG_R35 = "rung-3.5"
RUNG_R35_MGGA = "rung-3.5+meta-GGA"
RUNG_ORDER: Tuple[str, ...] = (RUNG_GGA, RUNG_MGGA, RUNG_R35, RUNG_R35_MGGA)
RUNG_RANK: Dict[str, int] = {r: i for i, r in enumerate(RUNG_ORDER)}

# Seed policies: which registry ingredients pull an arch onto the SCAN
# seed. "mgga_scan" is the production phase-1 policy (rung-3.5-only archs
# keep the PBE seed, so their v4 results carry over); "beyond_gga_scan"
# extends SCAN seeding to any beyond-GGA ingredient (the rung-3.5
# control-arm policy).
SEED_POLICIES: Tuple[str, ...] = ("mgga_scan", "beyond_gga_scan")


def arch_ingredients(name: str) -> Tuple[bool, bool]:
    """``(has_meta_gga, has_rung35)`` for registry arch ``name``.

    Strictly registry-derived; raises ``KeyError`` for unregistered names.
    """
    from xcquinox.alec.config import get_architecture
    cfg = get_architecture(name)
    desc = {getattr(d, "name", None) for d in getattr(cfg, "descriptors", ())}
    has_meta = bool(getattr(cfg, "meta_gga", False)) or "metagga" in desc
    # by prefix: the registry names both rung35 and rung35_multishell
    has_r35 = any(n and n.startswith("rung35") for n in desc)
    return has_meta, has_r35


def rung_from_ingredients(has_meta: bool, has_r35: bool) -> str:
    """Map the two ingredient flags onto a rung label."""
    if has_meta and has_r35:
        return RUNG_R35_MGGA
    if has_meta:
        return RUNG_MGGA
    if has_r35:
        return RUNG_R35
    return RUNG_GGA


def rung_of(name: str) -> str:
    """Jacob's-ladder rung of registry arch ``name`` (one of RUNG_ORDER)."""
    return rung_from_ingredients(*arch_ingredients(name))


def rung_rank(name: str) -> int:
    """Ladder rank of ``name`` (0=GGA .. 3=combined), for sorting."""
    return RUNG_RANK[rung_of(name)]


def seed_xc_for_arch(name: str, policy: str = "mgga_scan") -> str:
    """Rung-baseline SCF seed functional ("pbe" or "scan") for ``name``.

    ``"mgga_scan"`` (production phase 1): SCAN iff the arch carries the
    meta-GGA ingredient. ``"beyond_gga_scan"`` (control-arm follow-up):
    SCAN for any beyond-GGA ingredient.
    """
    if policy not in SEED_POLICIES:
        raise ValueError(f"unknown seed policy: {policy!r} "
                         f"(expected one of {SEED_POLICIES})")
    has_meta, has_r35 = arch_ingredients(name)
    if policy == "mgga_scan":
        return "scan" if has_meta else "pbe"
    return "scan" if (has_meta or has_r35) else "pbe"
