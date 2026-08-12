#!/usr/bin/env python
"""Tests for arch_style.py -- the shared rung taxonomy + palette + grouping.

The load-bearing property is that ``rung_of`` is DERIVED from the architecture
registry (not a hand-maintained map), so these tests cross-check the derivation
against ``xcquinox.alec.config`` for every registered arch.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import arch_style as A  # noqa: E402


# --------------------------------------------------------------------------- #
# rung_of: explicit expectations for the dfs6311 sweep archs
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("arch,rung", [
    ("deep_3x16", A.RUNG_GGA),
    ("deep_attn_3x16", A.RUNG_GGA),
    ("deep_cusp_3x16", A.RUNG_GGA),          # cusp is a GGA-level add-on, not a rung
    ("deep_dm_3x16", A.RUNG_GGA),            # dm_statistics likewise
    ("deep_rung35_3x16", A.RUNG_R35),        # localized DM occupancy -> rung-3.5
    ("deep_rung35only_3x16", A.RUNG_R35),
    ("deep_mgga_3x16", A.RUNG_MGGA),         # iso-orbital alpha -> meta-GGA
    ("deep_mgga_attn_3x16", A.RUNG_MGGA),
    ("deep_rung35_mgga_3x16", A.RUNG_R35_MGGA),  # both ingredients -> combined top
    # v4 additions; the multishell descriptor is registered as rung35_multishell,
    # so these two pin the prefix (not exact-name) rung-3.5 detection
    ("deep_rung35ms_3x16", A.RUNG_R35),
    ("deep_cusp_mgga_3x16", A.RUNG_MGGA),
    ("deep_rung35ms_mgga_3x16", A.RUNG_R35_MGGA),
    # legacy 4x32 base names (present in ARCH_ORDER, not separately registered)
    ("deep", A.RUNG_GGA),
    ("deep_combined", A.RUNG_GGA),
    ("deep_notransform", A.RUNG_GGA),
    ("deep_rung35", A.RUNG_R35),
    ("deep_mgga", A.RUNG_MGGA),
    # v4 base names (ARCH_COLOR keys resolved via the name-token fallback);
    # each must agree with its registered _3x16 twin above
    ("deep_rung35ms", A.RUNG_R35),
    ("deep_cusp_mgga", A.RUNG_MGGA),
    ("deep_rung35ms_mgga", A.RUNG_R35_MGGA),
])
def test_rung_of_expected(arch, rung):
    assert A.rung_of(arch) == rung


def test_rung_order_and_ranks():
    assert A.RUNG_ORDER == (A.RUNG_GGA, A.RUNG_MGGA, A.RUNG_R35, A.RUNG_R35_MGGA)
    # ascending Jacob's-ladder rank
    assert A.rung_rank("deep_3x16") < A.rung_rank("deep_mgga_3x16")
    assert A.rung_rank("deep_mgga_3x16") < A.rung_rank("deep_rung35_3x16")
    assert A.rung_rank("deep_rung35_3x16") < A.rung_rank("deep_rung35_mgga_3x16")


# --------------------------------------------------------------------------- #
# Derivation matches the registry for EVERY registered arch (no stale map)
# --------------------------------------------------------------------------- #
def test_rung_of_matches_registry_for_all_registered_archs():
    # Drift guard only: this recomputation mirrors the derivation, so the
    # non-circular rung anchors are the explicit expectations in
    # test_rung_of_expected above. rung-3.5 detection is by name PREFIX --
    # the registry carries both `rung35` and `rung35_multishell`.
    from xcquinox.alec.config import get_architecture, list_architectures
    for name in list_architectures():
        cfg = get_architecture(name)
        desc = {getattr(d, "name", None) for d in getattr(cfg, "descriptors", ())}
        has_meta = bool(getattr(cfg, "meta_gga", False)) or "metagga" in desc
        has_r35 = any(n and n.startswith("rung35") for n in desc)
        expected = (A.RUNG_R35_MGGA if (has_meta and has_r35)
                    else A.RUNG_MGGA if has_meta
                    else A.RUNG_R35 if has_r35
                    else A.RUNG_GGA)
        assert A.rung_of(name) == expected, name
        assert A.rung_of(name) in A.RUNG_ORDER


def test_meta_gga_flag_requires_metagga_descriptor_is_consistent():
    # meta_gga=True archs must classify as a meta-GGA family rung (sanity vs config)
    from xcquinox.alec.config import get_architecture, list_architectures
    for name in list_architectures():
        if getattr(get_architecture(name), "meta_gga", False):
            assert A.rung_of(name) in (A.RUNG_MGGA, A.RUNG_R35_MGGA), name


# --------------------------------------------------------------------------- #
# Palette back-compat + distinctness
# --------------------------------------------------------------------------- #
def test_arch_color_covers_every_arch_order_entry():
    for a in A.ARCH_ORDER:
        assert a in A.ARCH_COLOR
        assert A.arch_color(a).startswith("#")


def test_base8_take_tab10():
    import matplotlib
    tab = matplotlib.cm.get_cmap("tab10")
    for i, a in enumerate(A.ARCH_ORDER[:8]):
        assert A.ARCH_COLOR[a] == matplotlib.colors.to_hex(tab(i))


def test_meta_gga_archs_have_mutually_distinct_colors():
    mgga = ["deep_mgga_3x16", "deep_mgga_attn_3x16", "deep_rung35_mgga_3x16"]
    cols = [A.arch_color(a) for a in mgga]
    assert len(set(cols)) == len(cols), cols


def test_3x16_twin_inherits_base_color():
    assert A.ARCH_COLOR["deep_cusp_3x16"] == A.ARCH_COLOR["deep_cusp"]


def test_arch_color_unknown_falls_back_to_rung_accent():
    # an unregistered, non-ARCH_ORDER meta-GGA-looking name -> meta-GGA accent
    assert A.arch_color("deep_mgga_experimental") == A.RUNG_ACCENT[A.RUNG_MGGA]


def test_rung_accent_and_band_cover_every_rung():
    for r in A.RUNG_ORDER:
        assert A.RUNG_ACCENT[r].startswith("#")
        assert A.RUNG_BAND[r].startswith("#")


# --------------------------------------------------------------------------- #
# grouping helpers
# --------------------------------------------------------------------------- #
def test_sort_by_rung_groups_ladder_ascending():
    archs = ["deep_rung35_mgga_3x16", "deep_mgga_3x16", "deep_3x16",
             "deep_rung35_3x16", "deep_cusp_3x16"]
    ordered = A.sort_by_rung(archs)
    ranks = [A.rung_rank(a) for a in ordered]
    assert ranks == sorted(ranks)  # non-decreasing
    # GGA block first, combined last
    assert A.rung_of(ordered[0]) == A.RUNG_GGA
    assert A.rung_of(ordered[-1]) == A.RUNG_R35_MGGA


def test_sort_by_rung_stable_within_rung_by_arch_order():
    # deep_3x16 precedes deep_cusp_3x16 in ARCH_ORDER; both GGA -> order preserved
    got = A.sort_by_rung(["deep_cusp_3x16", "deep_3x16"])
    assert got == ["deep_3x16", "deep_cusp_3x16"]


def test_by_rung_partitions_input():
    archs = ["deep_3x16", "deep_mgga_3x16", "deep_rung35_3x16",
             "deep_rung35_mgga_3x16", "deep_cusp_3x16"]
    groups = A.by_rung(archs)
    # keys in ladder order, only present rungs
    assert list(groups) == [A.RUNG_GGA, A.RUNG_MGGA, A.RUNG_R35, A.RUNG_R35_MGGA]
    flat = [a for r in groups for a in groups[r]]
    assert sorted(flat) == sorted(archs)
    assert groups[A.RUNG_GGA] == ["deep_3x16", "deep_cusp_3x16"]


def test_v4_campaign_archs_all_in_arch_order_with_distinct_colors():
    """Every arch of the three v4 sweep arms must be figure-renderable.

    The suite guard (make_ablation_arch_figure.build_bh76w411_suite) raises on
    any eval'd arch outside ARCH_ORDER, so each arm's archs must be listed with
    a deliberate palette entry BEFORE its cells land. Arch axes quoted from
    hpcjobs/configs/dfs_step7.dfs6311_grid3_v4{,gga,mgga2}.yaml.
    """
    expected_rung = {
        # arm 1 (meta-GGA)
        "deep_mgga_3x16": A.RUNG_MGGA,
        "deep_mgga_attn_3x16": A.RUNG_MGGA,
        "deep_rung35_mgga_3x16": A.RUNG_R35_MGGA,
        # arm 2 (GGA-based)
        "deep_3x16": A.RUNG_GGA,
        "deep_attn_3x16": A.RUNG_GGA,
        "deep_cusp_3x16": A.RUNG_GGA,
        "deep_rung35_3x16": A.RUNG_R35,
        "deep_rung35_attn_3x16": A.RUNG_R35,
        "deep_rung35ms_3x16": A.RUNG_R35,
        # arm 3 (mgga stacking completions)
        "deep_cusp_mgga_3x16": A.RUNG_MGGA,
        "deep_rung35ms_mgga_3x16": A.RUNG_R35_MGGA,
    }
    v4_archs = list(expected_rung)
    assert len(v4_archs) == 11
    for a in v4_archs:
        assert a in A.ARCH_ORDER, a
        assert a in A.ARCH_COLOR, a
        # rung placement drives every rung-banded figure (gutters, spans,
        # by_rung summaries), so the roster pins it explicitly
        assert A.rung_of(a) == expected_rung[a], a
    cols = [A.ARCH_COLOR[a] for a in v4_archs]
    assert len(set(cols)) == len(cols), cols
    assert "#333333" not in cols  # nothing fell through to the unknown-base default


def test_rung_bands_contiguous_and_cover_all_indices():
    archs = A.sort_by_rung(["deep_3x16", "deep_cusp_3x16", "deep_mgga_3x16",
                            "deep_rung35_3x16", "deep_rung35_mgga_3x16"])
    bands = A.rung_bands(archs)
    # spans tile [0, len) with no gaps/overlaps
    assert bands[0][1] == 0 and bands[-1][2] == len(archs)
    for (_, s, e), (_, s2, _e2) in zip(bands, bands[1:]):
        assert e == s2 and e > s
    # each span is a single rung
    for r, s, e in bands:
        assert all(A.rung_of(archs[k]) == r for k in range(s, e))
    # the two GGA archs form one band of width 2
    assert bands[0] == (A.RUNG_GGA, 0, 2)
