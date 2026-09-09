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
    """The eight 4x32 legacy bases own tab10 in their palette order; the
    display order moved them to the tail, the palette did not move."""
    import matplotlib
    tab = matplotlib.cm.get_cmap("tab10")
    for i, a in enumerate(_LEGACY_4X32):
        assert A.ARCH_COLOR[a] == matplotlib.colors.to_hex(tab(i))


def test_meta_gga_archs_have_mutually_distinct_colors():
    mgga = ["deep_mgga_3x16", "deep_mgga_attn_3x16", "deep_rung35_mgga_3x16"]
    cols = [A.arch_color(a) for a in mgga]
    assert len(set(cols)) == len(cols), cols


def test_3x16_twin_inherits_base_color():
    # the palette is keyed by the shown names; a stored key maps through
    assert A.ARCH_COLOR["deep0_cusp_3x16"] == A.ARCH_COLOR["deep0_cusp_4x32"]
    assert A.arch_color("deep_cusp_3x16") == A.arch_color("deep_cusp")


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
    # the YAML axes hold STORED keys; the order and the palette hold shown
    # names, so membership is checked through the display map (a stored key
    # that is also a shown name -- deep_3x16 -- cannot be mapped by inspection)
    for a in v4_archs:
        shown = A.display_name(a)
        assert shown in A.ARCH_ORDER, (a, shown)
        assert shown in A.ARCH_COLOR, (a, shown)
        # rung placement drives every rung-banded figure (gutters, spans,
        # by_rung summaries), so the roster pins it explicitly
        assert A.rung_of(shown) == expected_rung[a], a
    cols = [A.arch_color(A.display_name(a)) for a in v4_archs]
    assert len(set(cols)) == len(cols), cols
    assert "#333333" not in cols  # nothing fell through to the unknown-base default


def test_v6_campaign_archs_all_in_arch_order_with_distinct_colors():
    """Every arch of the six v6 group files must be figure-renderable.

    Same guard as the v4 roster test above, but read from the group YAMLs
    themselves (hpcjobs/configs/dfs_step7.dfs6311_grid3_v6g*.yaml) so an
    edit to a group's arch axis cannot drift past the palette. The union is
    the campaign's 20 architectures; the four G1 size-ladder base names
    (shallow/shallow_attn/medium/medium_attn) are the entries the palette
    gained for v6 -- registered GGA archs with no width-twin suffix, shown
    as the Glorot ladder deep_2x8 / deep_attn_2x8 / deep_3x16 /
    deep_attn_3x16.
    """
    import glob

    import yaml

    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    files = sorted(glob.glob(os.path.join(
        repo, "hpcjobs", "configs", "dfs_step7.dfs6311_grid3_v6g*.yaml")))
    assert len(files) == 6, files
    union: set = set()
    for path in files:
        with open(path) as fh:
            cfg = yaml.safe_load(fh)
        archs = cfg["sweep"]["arch"]
        assert archs, path
        union.update(archs)
        # Each group renders its own figures, so distinctness is required
        # within a group's axis (across groups the palette deliberately
        # repeats: a width twin reuses its 4x32 base's color, and G4's
        # deep/deep_attn are the bases of G2's deep_3x16/deep_attn_3x16).
        # Membership and collision are separate defects with separate
        # messages: a missing palette key must not read as a collision. The
        # axes hold stored keys; the palette holds shown names.
        missing = [a for a in archs if A.display_name(a) not in A.ARCH_COLOR]
        assert not missing, (path, missing)
        group_cols = [A.arch_color(A.display_name(a)) for a in archs]
        assert len(set(group_cols)) == len(group_cols), (path, group_cols)
    assert len(union) == 20, sorted(union)
    for a in sorted(union):
        assert A.display_name(a) in A.ARCH_ORDER, a
        assert A.display_name(a) in A.ARCH_COLOR, a
    cols = [A.arch_color(A.display_name(a)) for a in sorted(union)]
    assert "#333333" not in cols  # nothing fell through to the unknown-base default
    # the only union-level repeats are the two by-design base/twin pairs
    assert len(set(cols)) == len(cols) - 2, cols
    assert A.ARCH_COLOR["deep0_4x32"] == A.ARCH_COLOR["deep0_3x16"]
    assert A.ARCH_COLOR["deep0_attn_4x32"] == A.ARCH_COLOR["deep0_attn_3x16"]
    # the size ladder is registered and classifies GGA off the registry
    for a in ("deep_2x8", "deep_attn_2x8", "deep_3x16", "deep_attn_3x16"):
        assert A.rung_of(a) == A.RUNG_GGA, a


def test_size_ladder_colors_explicit_not_suffix_stripped():
    """Base names in the ARCH_ORDER tail keep their explicit palette entries.

    The width-twin inheritance strips the last five characters of every tail
    entry; unguarded, it resolved "medium" through ``ARCH_COLOR.get("m")`` and
    replaced the explicit entry with the unknown-base default. The guard skips
    names without the ``_3x16`` suffix while leaving the twin inheritance
    intact.
    """
    ladder = ("deep_2x8", "deep_attn_2x8", "deep_3x16", "deep_attn_3x16")
    cols = [A.ARCH_COLOR[a] for a in ladder]
    assert len(set(cols)) == len(cols), cols
    assert "#333333" not in cols
    # width-twin inheritance unchanged; the colour-only base names
    # (deep_rung35, deep_mgga: never registry keys) keep their stored spelling
    assert A.ARCH_COLOR["deep0_3x16"] == A.ARCH_COLOR["deep0_4x32"]
    assert A.ARCH_COLOR["deep0_rung35_3x16"] == A.ARCH_COLOR["deep_rung35"]
    assert A.ARCH_COLOR["deep0_mgga_3x16"] == A.ARCH_COLOR["deep_mgga"]


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


# --------------------------------------------------------------------------- #
# Taxonomy is the LIBRARY's (xcquinox.alec.rungs); arch_style only adds the
# name-token fallback for legacy unregistered display names + the styling.
# --------------------------------------------------------------------------- #
def test_taxonomy_delegates_to_library_rungs():
    from xcquinox.alec import rungs
    assert A.RUNG_GGA is rungs.RUNG_GGA
    assert A.RUNG_MGGA is rungs.RUNG_MGGA
    assert A.RUNG_R35 is rungs.RUNG_R35
    assert A.RUNG_R35_MGGA is rungs.RUNG_R35_MGGA
    assert A.RUNG_ORDER == rungs.RUNG_ORDER
    # registered names agree between the strict library predicate and the
    # fallback-carrying analysis one
    for arch in ("deep_3x16", "deep_rung35_3x16", "deep_mgga_3x16",
                 "deep_rung35ms_mgga_3x16"):
        assert A.rung_of(arch) == rungs.rung_of(arch)
    # the fallback remains: a legacy unregistered name resolves in
    # arch_style but raises in the library
    assert A.rung_of("deep_mgga") == A.RUNG_MGGA
    import pytest as _pytest
    with _pytest.raises(KeyError):
        rungs.rung_of("deep_mgga")


# --------------------------------------------------------------------------- #
# T3: display names (2026-09-09)
#
# ARCH_ORDER and ARCH_COLOR move to the SHOWN names -- the names that say what
# each network is -- and every style function resolves a name in the shown
# sense, mapping a stored key through ``display_name`` first. The colour of a
# stored key moves with it, so the shown ``deep_3x16`` (the registry's
# ``medium``) carries medium's green and ``deep0_3x16`` (the registry's
# ``deep_3x16``) carries the tab10 blue of the old deep family.
# --------------------------------------------------------------------------- #

_MEDIUM_GREEN = "#98df8a"
_MEDIUM_ATTN_GREEN = "#c7e9c0"
_DEEP_BLUE = "#1f77b4"        # tab10[0], the old `deep` / `deep_3x16` colour
_DEEP_CUSP_GREEN = "#2ca02c"  # tab10[2]
_ZEROED_TEXT = "last layer zeroed (pre-training starts at the LDA)"
#: the eight 4x32 legacy bases, in their tab10 palette order
_LEGACY_4X32 = (
    "deep0_4x32", "deep0_attn_4x32", "deep0_cusp_4x32", "deep0_dm_4x32",
    "deep0_combined_4x32", "deep0_combined_attn_4x32",
    "deep0_notransform_4x32", "deep0_notransform_attn_4x32")


def test_arch_order_holds_shown_names_only():
    """No stored-only key survives in the ordering the figures draw by.

    RED: ARCH_ORDER carries `medium`, `deep_cusp_3x16` and the rest of the
    stored keys today.
    """
    for a in A.ARCH_ORDER:
        assert A.display_name(A.stored_key(a)) == a, a
    # the order runs along the axes: the Glorot ladder, the 3x16 zero-init
    # family (plain, attention, cusp, dm, combined, notransform, rung-3.5,
    # meta-GGA), the 4x32 legacy family last
    assert A.ARCH_ORDER[:12] == (
        "deep_2x8", "deep_attn_2x8", "deep_3x16", "deep_attn_3x16",
        "deep0_3x16", "deep0_attn_3x16", "deep0_cusp_3x16", "deep0_dm_3x16",
        "deep0_combined_3x16", "deep0_combined_attn_3x16",
        "deep0_notransform_3x16", "deep0_notransform_attn_3x16")
    assert A.ARCH_ORDER[-8:] == _LEGACY_4X32
    for stored_only in ("medium", "medium_attn", "shallow", "shallow_attn",
                        "deep", "deep_cusp_3x16", "deep_mgga_3x16"):
        assert stored_only not in A.ARCH_ORDER, stored_only
    for shown in ("deep_3x16", "deep0_3x16", "deep0_cusp_3x16",
                  "deep0_cusp_mgga_3x16", "deep_2x8"):
        assert shown in A.ARCH_ORDER, shown


def test_arch_color_keyed_by_shown_names():
    """Each stored key's colour travels with it under the rename.

    Kills m1/m3 at the palette: a rename that left the palette keyed on the
    stored keys would give the shown `deep_3x16` the old deep blue -- the
    colour of a DIFFERENT network -- in every figure.
    """
    assert A.ARCH_COLOR["deep_3x16"] == _MEDIUM_GREEN
    assert A.arch_color("deep_3x16") == _MEDIUM_GREEN
    assert A.ARCH_COLOR["deep_attn_3x16"] == _MEDIUM_ATTN_GREEN
    assert A.ARCH_COLOR["deep0_4x32"] == _DEEP_BLUE
    assert A.arch_color("deep0_3x16") == _DEEP_BLUE
    assert A.ARCH_COLOR["deep0_cusp_4x32"] == _DEEP_CUSP_GREEN
    # the width-twin inheritance, restated in shown names
    assert A.ARCH_COLOR["deep0_cusp_3x16"] == A.ARCH_COLOR["deep0_cusp_4x32"]
    assert A.ARCH_COLOR["deep0_3x16"] == A.ARCH_COLOR["deep0_4x32"]
    # nothing fell through to the unknown-base default
    assert "#333333" not in [A.ARCH_COLOR[a] for a in A.ARCH_ORDER]


def test_a_stored_key_is_accepted_and_mapped_through_display_name():
    """A caller holding a stored key (a pulled manifest read outside the
    boundary, a fixture) still gets the right colour and rung.

    Kills m2 at the colour: `arch_color("medium")` must be the colour of the
    shown `deep_3x16`, not of the stored key `deep_3x16`.
    """
    assert A.arch_color("medium") == A.arch_color("deep_3x16") == _MEDIUM_GREEN
    assert A.arch_color("deep_cusp") == A.arch_color("deep0_cusp_4x32")
    assert A.arch_color("deep_cusp_3x16") == _DEEP_CUSP_GREEN
    assert A.rung_of("medium") == A.RUNG_GGA
    assert A.rung_of("deep_cusp_mgga_3x16") == A.RUNG_MGGA


def test_display_name_and_expanded_key_are_reexported():
    """The figure scripts import the layer from here, so the re-exports are
    part of this module's interface. Kills m2: with ``stored_key`` returning
    its input, ``expanded_key("deep_3x16")`` reads the zero-init text of the
    other network."""
    assert A.display_name("medium") == "deep_3x16"
    assert A.display_name("medium", protocol="25 cycles") == \
        "deep_3x16 [25 cycles]"
    assert A.stored_key("deep_3x16") == "medium"
    assert A.expanded_key("deep_3x16") == "3 x 16, Glorot initialization"
    assert A.expanded_key("deep0_3x16") == f"3 x 16, {_ZEROED_TEXT}"
    assert A.key_line(["deep_3x16", "deep0_3x16"]) == (
        "deep_3x16: 3 x 16, Glorot initialization; "
        f"deep0_3x16: 3 x 16, {_ZEROED_TEXT}")


def test_rung_of_a_shown_name_delegates_to_the_registry():
    """The rung of a shown name is the rung of the configuration it names.

    The VALUES below do not discriminate on their own -- arch_style's
    name-token fallback already reads `mgga` / `rung35` out of a shown name --
    so the pin is the delegation itself: the rung must come from the registry
    entry ``stored_key`` resolves, not from the characters of the label.
    """
    from xcquinox.alec import rungs
    for shown in ("deep_3x16", "deep0_3x16", "deep0_cusp_mgga_3x16",
                  "deep0_rung35ms_mgga_3x16", "deep0_mgga_3x16"):
        assert A.rung_of(shown) == rungs.rung_of(A.stored_key(shown)), shown
    assert A.rung_of("deep0_cusp_mgga_3x16") == A.RUNG_MGGA
    assert A.rung_of("deep0_rung35ms_mgga_3x16") == A.RUNG_R35_MGGA


def test_sort_by_rung_orders_shown_names_by_arch_order():
    """Within a rung the order is ARCH_ORDER's, which now holds shown names:
    the Glorot ladder leads, the zero-init 3x16 family follows, the 4x32
    legacy family closes.

    RED: today `deep_3x16` is an ARCH_ORDER member and `deep0_4x32` is not,
    so the unknown name sorts last by the fallback rather than by position.
    """
    assert A.sort_by_rung(["deep0_4x32", "deep_3x16"]) == \
        ["deep_3x16", "deep0_4x32"]
    assert A.sort_by_rung(["deep0_3x16", "deep_3x16", "deep_2x8"]) == \
        ["deep_2x8", "deep_3x16", "deep0_3x16"]


# --------------------------------------------------------------------------- #
# Protocol-tagged names: a tag rides on the shown name and changes neither the
# colour nor the position of the architecture (2026-09-09, review findings)
# --------------------------------------------------------------------------- #

def test_tagged_names_keep_their_architectures_colour():
    """RED before the fix: the tagged branch of arch_color looked the stored
    key up in the SHOWN-keyed palette, so every tagged name of an anchored
    run fell through to the rung accent (four architectures in one colour)
    and ``deep0_3x16 [anchored]`` took medium's green."""
    assert A.arch_color("deep_3x16 [anchored]") == A.arch_color("deep_3x16") \
        == _MEDIUM_GREEN
    assert A.arch_color("deep0_3x16 [anchored]") == A.arch_color("deep0_3x16") \
        == _DEEP_BLUE
    assert A.arch_color("deep_2x8 [anchored]") == A.ARCH_COLOR["deep_2x8"]
    assert A.arch_color("deep_attn_3x16 [25 cycles]") == _MEDIUM_ATTN_GREEN
    # a tagged STORED-only key maps through as well
    assert A.arch_color("medium [anchored]") == _MEDIUM_GREEN
    assert A.as_shown("medium [anchored]") == "deep_3x16 [anchored]"
    assert A.base_name("medium [anchored]") == "deep_3x16"


def test_tagged_names_are_order_members_and_sort_after_their_base():
    """A tagged name is an ARCH_ORDER member through its base (the per-arch
    figures can draw it) and sorts right after the untagged name of the same
    architecture; unknown names stay last."""
    assert A.in_order("deep_3x16 [anchored]")
    assert A.in_order("medium [anchored]")
    assert not A.in_order("no_such_arch")
    got = A.order_present(["deep0_3x16", "deep_3x16 [anchored]", "zzz",
                           "deep_3x16", "deep_3x16 [25 cycles]", "deep_2x8"])
    assert got == ["deep_2x8", "deep_3x16", "deep_3x16 [25 cycles]",
                   "deep_3x16 [anchored]", "deep0_3x16", "zzz"]
    assert A.order_known(got) == got[:-1]
    assert A.order_present(["deep_3x16", "deep_3x16", None]) == ["deep_3x16"]
    assert A.sort_by_rung(["deep_3x16 [anchored]", "deep0_mgga_3x16",
                           "deep_3x16"]) == \
        ["deep_3x16", "deep_3x16 [anchored]", "deep0_mgga_3x16"]
    assert A.rung_of("deep0_mgga_3x16 [anchored]") == A.RUNG_MGGA
