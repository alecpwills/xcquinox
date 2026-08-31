#!/usr/bin/env python
"""Shared architecture styling -- Jacob's-ladder rung taxonomy, palette, grouping.

Single source of truth for how xcquinox architectures are ORDERED, COLORED, and
GROUPED BY RUNG across every figure script (``make_ablation_arch_figure.py``, the
DFS self-consistent-density demo notebook, ``plot_pretraining_curves.py``). Keeping
it in one importable module means the meta-GGA-vs-SCAN / "does climbing Jacob's
ladder help?" story reads consistently everywhere, instead of each script inventing
its own colors.

The RUNG of an arch is DERIVED from the architecture registry, via the
library taxonomy in ``xcquinox.alec.rungs`` (a ``metagga`` descriptor /
``meta_gga`` flag -> meta-GGA, a ``rung35*`` descriptor -> rung-3.5, both ->
the combined top arch, neither -> GGA). So a newly registered arch classifies
automatically instead of needing a hand-maintained map that silently rots.
This module adds only a name-token fallback covering the legacy 4x32 base
names that appear in ``ARCH_ORDER`` but are not separately registered, plus
the styling (colors, bands, ordering).
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import matplotlib
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# Canonical display order. Moved VERBATIM from make_ablation_arch_figure.py so
# there is ONE definition; that module re-imports these names for back-compat.
# The 8 base (4x32) archs MUST stay first -- they own the tab10 colors, and every
# depth-3/width-16 twin reuses its 4x32 sibling's color (same architecture).
# --------------------------------------------------------------------------- #
ARCH_ORDER: Tuple[str, ...] = (
    "deep", "deep_attn", "deep_cusp", "deep_dm",
    "deep_combined", "deep_combined_attn",
    "deep_notransform", "deep_notransform_attn",
    # v6 G1 ladder archs (registered base names, no width-twin suffix).
    # shallow/shallow_attn are the reduced size (depth 2 x 8 nodes);
    # medium/medium_attn are the PRODUCTION 3x16 size differing from
    # deep_3x16/deep_attn_3x16 only in descriptor_log_transform and
    # zero_init_final_layer -- a transform/initialization ablation at fixed
    # capacity, not a size step. Attention twin after its base.
    "shallow", "shallow_attn", "medium", "medium_attn",
    "deep_3x16", "deep_attn_3x16", "deep_cusp_3x16", "deep_dm_3x16",
    "deep_combined_3x16", "deep_combined_attn_3x16",
    "deep_notransform_3x16", "deep_notransform_attn_3x16",
    "deep_rung35_3x16", "deep_rung35_attn_3x16", "deep_rung35only_3x16",
    "deep_rung35ms_3x16",
    "deep_mgga_3x16", "deep_mgga_attn_3x16", "deep_rung35_mgga_3x16",
    "deep_cusp_mgga_3x16", "deep_rung35ms_mgga_3x16",
)

_ARCH_TAB = plt.get_cmap("tab10")
ARCH_COLOR: Dict[str, str] = {
    a: matplotlib.colors.to_hex(_ARCH_TAB(i)) for i, a in enumerate(ARCH_ORDER[:8])
}
# rung-3.5 base names (no 4x32 sibling to inherit from): the 2 unused tab10 slots
# + a distinct extra, so the _3x16-strip heuristic resolves them.
ARCH_COLOR["deep_rung35"] = matplotlib.colors.to_hex(_ARCH_TAB(8))
ARCH_COLOR["deep_rung35_attn"] = matplotlib.colors.to_hex(_ARCH_TAB(9))
ARCH_COLOR["deep_rung35only"] = "#393b79"
# meta-GGA base names (tab10 + #393b79 taken): distinct tab20b-family hexes.
ARCH_COLOR["deep_mgga"] = "#8c6d31"
ARCH_COLOR["deep_mgga_attn"] = "#843c39"
ARCH_COLOR["deep_rung35_mgga"] = "#7b4173"
# v4-sweep base names (rung-3.5 multishell + the mgga stacking completions):
# further tab20b-family hexes, each near its closest kin above (rung35only
# blues / mgga golds / rung35_mgga purples).
ARCH_COLOR["deep_rung35ms"] = "#6b6ecf"
ARCH_COLOR["deep_cusp_mgga"] = "#bd9e39"
ARCH_COLOR["deep_rung35ms_mgga"] = "#a55194"
# v6 G1 ladder archs: green-family entries chosen by worst-case CIEDE2000
# distance against every colour a same-figure architecture or rung accent can
# carry -- tab10[2] #2ca02c (deep_cusp, its 3x16 twin, and the rung-3.5
# accent) is itself a green, so proximity to it is the binding constraint.
# shallow pair (tab20b[4]/[5]): worst cross-family separation 15.91.
# medium pair (tab20[5]/tab20c[11]): worst 16.90 (the tab20c[8] value first
# chosen here sat at 5.79 from #2ca02c, half the palette's own twin gap).
# Each attention twin is the lighter shade of its base.
ARCH_COLOR["shallow"] = "#637939"
ARCH_COLOR["shallow_attn"] = "#8ca252"
ARCH_COLOR["medium"] = "#98df8a"
ARCH_COLOR["medium_attn"] = "#c7e9c0"
for _small in ARCH_ORDER[8:]:
    # Only width-twin names inherit by suffix-strip; a base name in the tail
    # (the size ladder) keeps its explicit entry above -- the unguarded strip
    # would resolve "medium" via ARCH_COLOR.get("m") and clobber it with the
    # unknown-base default.
    if not _small.endswith("_3x16"):
        continue
    ARCH_COLOR[_small] = ARCH_COLOR.get(_small[: -len("_3x16")], "#333333")

SUBSET_SIZES: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 12, 15, 18)

# --------------------------------------------------------------------------- #
# Jacob's-ladder rung taxonomy -- imported from the library single source
# (xcquinox.alec.rungs). Order: GGA (rung 2) < meta-GGA (rung 3) < rung-3.5
# (Janesko, "between meta-GGA and hybrid") < the combined rung-3.5+meta-GGA
# arch (the most ingredients: the meta-GGA iso-orbital alpha AND the rung-3.5
# localized-DM occupancy).
# --------------------------------------------------------------------------- #
from xcquinox.alec.rungs import (  # noqa: E402
    RUNG_GGA, RUNG_MGGA, RUNG_R35, RUNG_R35_MGGA, RUNG_ORDER,
    RUNG_RANK as _RUNG_RANK,
    arch_ingredients as _registry_ingredients,
    rung_from_ingredients as _rung_from_ingredients,
)

# Saturated accent per rung (per-rung summary bars, legend section swatches) and a
# light background tint (axvspan rung bands behind per-arch bars).
RUNG_ACCENT: Dict[str, str] = {
    RUNG_GGA: "#1f77b4",       # blue
    RUNG_MGGA: "#8c564b",      # brown  (meta-GGA family)
    RUNG_R35: "#2ca02c",       # green
    RUNG_R35_MGGA: "#9467bd",  # purple (both ingredients)
}
RUNG_BAND: Dict[str, str] = {
    RUNG_GGA: "#e8f0f7",
    RUNG_MGGA: "#f3ebe8",
    RUNG_R35: "#e9f5e9",
    RUNG_R35_MGGA: "#f0eaf5",
}


def _arch_ingredients(arch: str) -> Tuple[bool, bool]:
    """``(has_meta_gga, has_rung35)`` for ``arch``, from the registry if possible.

    Falls back to name tokens for archs not in the registry (e.g. the legacy 4x32
    base names ``deep_rung35`` / ``deep_mgga`` that only exist as ARCH_COLOR keys).
    """
    try:
        return _registry_ingredients(arch)
    except Exception:
        return (("mgga" in arch) or ("metagga" in arch)), ("rung35" in arch)


def rung_of(arch: str) -> str:
    """Jacob's-ladder rung of ``arch`` (one of :data:`RUNG_ORDER`)."""
    return _rung_from_ingredients(*_arch_ingredients(arch))


def rung_rank(arch: str) -> int:
    """Ladder rank of ``arch`` (0=GGA .. 3=combined), for sorting."""
    return _RUNG_RANK[rung_of(arch)]


def arch_color(arch: str) -> str:
    """Per-arch hex color; falls back to the _3x16 base, then the rung accent."""
    if arch in ARCH_COLOR:
        return ARCH_COLOR[arch]
    if arch.endswith("_3x16") and arch[: -len("_3x16")] in ARCH_COLOR:
        return ARCH_COLOR[arch[: -len("_3x16")]]
    return RUNG_ACCENT.get(rung_of(arch), "#333333")


def rung_color(rung: str) -> str:
    """Saturated accent for a rung (one of :data:`RUNG_ORDER`)."""
    return RUNG_ACCENT.get(rung, "#333333")


def sort_by_rung(archs) -> List[str]:
    """Archs reordered so rungs are contiguous (ascending ladder), stable within a
    rung by canonical ``ARCH_ORDER`` position (unknown archs sort last)."""
    def _key(a):
        try:
            pos = ARCH_ORDER.index(a)
        except ValueError:
            pos = len(ARCH_ORDER)
        return (rung_rank(a), pos, a)
    return sorted(archs, key=_key)


def by_rung(archs) -> Dict[str, List[str]]:
    """``{rung: [archs]}`` in ladder order, each list in :func:`sort_by_rung` order.
    Only rungs actually present among ``archs`` appear."""
    out: Dict[str, List[str]] = {}
    for a in sort_by_rung(archs):
        out.setdefault(rung_of(a), []).append(a)
    return out


def rung_bands(archs) -> List[Tuple[str, int, int]]:
    """Contiguous ``(rung, i_start, i_end)`` half-open index spans over ``archs`` AS
    GIVEN, for drawing ``axvspan`` rung backgrounds. Pass a :func:`sort_by_rung`
    -ordered list so each rung is a single span."""
    archs = list(archs)
    spans: List[Tuple[str, int, int]] = []
    i = 0
    while i < len(archs):
        r = rung_of(archs[i])
        j = i + 1
        while j < len(archs) and rung_of(archs[j]) == r:
            j += 1
        spans.append((r, i, j))
        i = j
    return spans
