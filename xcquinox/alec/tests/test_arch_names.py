"""Tests for the architecture display-name layer (``xcquinox.alec.arch_names``).

The registry keys are storage identifiers: ``medium`` and ``deep_3x16`` name
the same network (depth 3, width 16, the same inputs), separated only by
``zero_init_final_layer``. The display layer derives, from the registry
configuration alone, a name that states what the network is, and inverts that
map so a shown name can still reach the registry.

Two properties carry the layer. The derived map must be ONE-TO-ONE over the
registry -- a derivation that ignores the zero-init flag sends ``medium`` and
``deep_3x16`` to the same shown name and destroys the inverse -- and the two
shown names that collide with OTHER stored keys (``deep_3x16``,
``deep_attn_3x16``) must resolve in the SHOWN sense everywhere, since every
consumer downstream of the display boundary holds shown names.

The expected map below is written out by hand from the rule and the registry
rather than recomputed from it, so a change to either the rule or a registry
entry fails here instead of silently redefining what the figures are called.
"""
import pytest

from xcquinox.alec.config import ARCHITECTURES


class _Lazy:
    """``xcquinox.alec.arch_names``, resolved at first use.

    The module is imported inside each test rather than at module scope so a
    missing module fails the tests that need it instead of erroring the whole
    collection (which would hide every other test in the same run).
    """

    def __getattr__(self, name):
        from xcquinox.alec import arch_names
        return getattr(arch_names, name)


AN = _Lazy()


# ---------------------------------------------------------------------------
# The map, by hand: stored key -> shown name.
#
# Rule (page of 2026-09-09, amendment 2): the family word carries ONE axis,
# the initialization -- ``deep0`` for a zero-initialized final layer, ``deep``
# otherwise, at every size; tokens are the stored key without its leading
# family word and without a trailing size; the size suffix is
# ``_{depth}x{nodes}`` and carries the depth.
# ---------------------------------------------------------------------------
EXPECTED_DISPLAY = {
    # the four Glorot entries (zero_init_final_layer=False)
    "shallow":                     "deep_2x8",
    "shallow_attn":                "deep_attn_2x8",
    "medium":                      "deep_3x16",
    "medium_attn":                 "deep_attn_3x16",
    # the 4x32 family (zero_init_final_layer=True)
    "deep":                        "deep0_4x32",
    "deep_attn":                   "deep0_attn_4x32",
    "deep_cusp":                   "deep0_cusp_4x32",
    "deep_cusp_attn":              "deep0_cusp_attn_4x32",
    "deep_dm":                     "deep0_dm_4x32",
    "deep_dm_attn":                "deep0_dm_attn_4x32",
    "deep_combined":               "deep0_combined_4x32",
    "deep_combined_attn":          "deep0_combined_attn_4x32",
    "deep_notransform":            "deep0_notransform_4x32",
    "deep_notransform_attn":       "deep0_notransform_attn_4x32",
    # the 3x16 twins (zero_init_final_layer=True)
    "deep_3x16":                   "deep0_3x16",
    "deep_attn_3x16":              "deep0_attn_3x16",
    "deep_cusp_3x16":              "deep0_cusp_3x16",
    "deep_dm_3x16":                "deep0_dm_3x16",
    "deep_combined_3x16":          "deep0_combined_3x16",
    "deep_combined_attn_3x16":     "deep0_combined_attn_3x16",
    "deep_notransform_3x16":       "deep0_notransform_3x16",
    "deep_notransform_attn_3x16":  "deep0_notransform_attn_3x16",
    "deep_rung35_3x16":            "deep0_rung35_3x16",
    "deep_rung35_attn_3x16":       "deep0_rung35_attn_3x16",
    "deep_rung35ms_3x16":          "deep0_rung35ms_3x16",
    "deep_rung35only_3x16":        "deep0_rung35only_3x16",
    "deep_mgga_3x16":              "deep0_mgga_3x16",
    "deep_mgga_attn_3x16":         "deep0_mgga_attn_3x16",
    "deep_rung35_mgga_3x16":       "deep0_rung35_mgga_3x16",
    "deep_cusp_mgga_3x16":         "deep0_cusp_mgga_3x16",
    "deep_rung35ms_mgga_3x16":     "deep0_rung35ms_mgga_3x16",
}

#: The two shown names that are also stored keys of OTHER configurations.
COLLIDING = ("deep_3x16", "deep_attn_3x16")

_ZEROED = "last layer zeroed (pre-training starts at the LDA)"
_GLOROT = "Glorot initialization"


# ---------------------------------------------------------------------------
# T1: the derived map
# ---------------------------------------------------------------------------
def test_display_map_covers_every_registry_key_exactly():
    """Every stored key is named, and nothing else is.

    Kills m1 (a derivation that ignores ``zero_init_final_layer``): with the
    flag dropped, ``medium`` and ``deep_3x16`` both derive ``deep_3x16`` and
    the table below no longer holds.
    """
    assert set(AN.DISPLAY_NAME) == set(ARCHITECTURES), (
        set(AN.DISPLAY_NAME).symmetric_difference(ARCHITECTURES))
    assert set(EXPECTED_DISPLAY) == set(ARCHITECTURES), (
        "the pinned table and the registry disagree: "
        f"{set(EXPECTED_DISPLAY).symmetric_difference(ARCHITECTURES)}")
    assert AN.DISPLAY_NAME == EXPECTED_DISPLAY


def test_display_map_is_one_to_one():
    """A collision in the inverse would make a shown name ambiguous. Kills m1."""
    shown = list(AN.DISPLAY_NAME.values())
    assert len(set(shown)) == len(shown), sorted(
        n for n in shown if shown.count(n) > 1)
    assert AN.STORED_KEY == {v: k for k, v in EXPECTED_DISPLAY.items()}


def test_derive_display_name_matches_the_map_for_every_key():
    """The map is the rule applied to the registry, not a hand-kept table."""
    for key, cfg in ARCHITECTURES.items():
        assert AN.derive_display_name(key, cfg) == EXPECTED_DISPLAY[key], key


def test_stored_key_inverts_display_name_for_every_registry_key():
    for key in ARCHITECTURES:
        assert AN.stored_key(AN.display_name(key)) == key, key


def test_colliding_names_resolve_in_the_shown_sense():
    """``deep_3x16`` is medium's shown name AND another registry key. Every
    style function resolves it in the SHOWN sense, so the stored configuration
    of that key is reached only through its own shown name ``deep0_3x16``.
    Kills m1 and m2 (a ``stored_key`` that returns its input answers
    ``deep_3x16`` here).
    """
    assert AN.stored_key("deep_3x16") == "medium"
    assert AN.stored_key("deep_attn_3x16") == "medium_attn"
    assert AN.stored_key("deep0_3x16") == "deep_3x16"
    assert AN.stored_key("deep0_attn_3x16") == "deep_attn_3x16"
    assert AN.display_name("deep_3x16") == "deep0_3x16"
    assert AN.display_name("medium") == "deep_3x16"


def test_unknown_names_pass_through_unchanged():
    """A legacy display name (``deep_mgga``, an ARCH_COLOR key that was never
    registered) and a fixture name must not raise or be rewritten."""
    assert AN.display_name("deep_mgga") == "deep_mgga"
    assert AN.stored_key("not_an_arch") == "not_an_arch"
    assert AN.expanded_key("not_an_arch") == ""


# ---------------------------------------------------------------------------
# T1: the protocol tag
# ---------------------------------------------------------------------------
def test_protocol_tag_round_trip():
    """A run's protocol tag rides on the shown name and survives the inverse.
    Kills m8 (``display_name`` dropping the tag)."""
    tagged = AN.display_name("medium", protocol="25 cycles")
    assert tagged == "deep_3x16 [25 cycles]"
    assert AN.stored_key(tagged) == "medium"
    assert AN.expanded_key(tagged) == f"3 x 16, {_GLOROT}; 25 cycles"
    parity = AN.display_name("medium", protocol="dfsparity")
    assert parity == "deep_3x16 [dfsparity]"
    assert AN.stored_key(parity) == "medium"
    # a known tag is spelled out; the anchor is the v6 runs' tag, whose
    # run-time zero-init is what the text states
    anchored = AN.display_name("medium", protocol="anchored")
    assert anchored == "deep_3x16 [anchored]"
    assert AN.stored_key(anchored) == "medium"
    assert AN.expanded_key(anchored) == (
        f"3 x 16, {_GLOROT}; anchored on the parent "
        "(last layer zeroed at run time)")


def test_display_name_without_protocol_is_the_bare_name():
    assert AN.display_name("medium", protocol=None) == "deep_3x16"


# ---------------------------------------------------------------------------
# T1: the expanded key
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name,text", [
    # the two names of the SAME 3x16 network, separated by the init only
    ("deep_3x16", f"3 x 16, {_GLOROT}"),
    ("deep0_3x16", f"3 x 16, {_ZEROED}"),
    ("deep_attn_3x16", f"3 x 16, {_GLOROT}, 4 attention heads"),
    ("deep0_cusp_3x16", f"3 x 16, {_ZEROED}, cusp descriptors"),
    ("deep0_cusp_mgga_3x16",
     f"3 x 16, {_ZEROED}, cusp descriptors, meta-GGA (SCAN parent)"),
    ("deep_2x8", f"2 x 8, {_GLOROT}"),
    ("deep_attn_2x8", f"2 x 8, {_GLOROT}, 2 attention heads"),
])
def test_expanded_key_texts(name, text):
    """Hand-written from the rule; ``deep_3x16`` vs ``deep0_3x16`` is the pair
    the whole rename exists for, so both are pinned. Kills m2 (with
    ``stored_key`` returning its input, ``deep_3x16`` reads the zero-init
    text)."""
    assert AN.expanded_key(name) == text


def test_expanded_key_of_the_four_by_thirty_two_family():
    """The size clause is the configuration's, not the name's suffix."""
    assert AN.expanded_key("deep0_4x32") == f"4 x 32, {_ZEROED}"
    assert AN.expanded_key("deep0_attn_4x32") == (
        f"4 x 32, {_ZEROED}, 4 attention heads")


def test_expanded_key_states_size_and_initialization_for_every_key():
    """Shape guard over the whole registry: every architecture's key opens
    with its size and states one of the two initializations."""
    for key, cfg in ARCHITECTURES.items():
        text = AN.expanded_key(EXPECTED_DISPLAY[key])
        assert text.startswith(f"{cfg.depth} x {cfg.nodes}, "), (key, text)
        assert (_ZEROED in text) == bool(cfg.zero_init_final_layer), (key, text)
        assert (_GLOROT in text) != bool(cfg.zero_init_final_layer), (key, text)


# ---------------------------------------------------------------------------
# T1: the key line
# ---------------------------------------------------------------------------
def test_key_line_keeps_order_and_drops_repeats():
    line = AN.key_line(["deep_3x16", "deep0_3x16", "deep_3x16"])
    assert line == (f"deep_3x16: 3 x 16, {_GLOROT}; "
                    f"deep0_3x16: 3 x 16, {_ZEROED}")
    assert AN.key_line([]) == ""


def test_key_line_carries_a_protocol_tag_once_per_name():
    line = AN.key_line(["deep_3x16", "deep_3x16 [25 cycles]"])
    assert line == (f"deep_3x16: 3 x 16, {_GLOROT}; "
                    f"deep_3x16 [25 cycles]: 3 x 16, {_GLOROT}; 25 cycles")


# ---------------------------------------------------------------------------
# T1: the aliases
# ---------------------------------------------------------------------------
def test_aliases_are_the_shown_names_that_are_not_stored_keys():
    """``get_architecture`` resolves a shown name through these. The two
    colliding names are NOT aliases: they are stored keys of other
    configurations and must keep resolving to those."""
    expected = {shown: stored for stored, shown in EXPECTED_DISPLAY.items()
                if shown not in ARCHITECTURES}
    assert AN.ALIASES == expected
    assert AN.ALIASES["deep0_3x16"] == "deep_3x16"
    assert AN.ALIASES["deep0_attn_3x16"] == "deep_attn_3x16"
    assert AN.ALIASES["deep_2x8"] == "shallow"
    assert AN.ALIASES["deep_attn_2x8"] == "shallow_attn"
    assert AN.ALIASES["deep0_4x32"] == "deep"
    for name in COLLIDING:
        assert name not in AN.ALIASES, name
    assert not (set(AN.ALIASES) & set(ARCHITECTURES)), (
        set(AN.ALIASES) & set(ARCHITECTURES))
    for stored in AN.ALIASES.values():
        assert stored in ARCHITECTURES, stored


# ---------------------------------------------------------------------------
# The rule itself, on a configuration outside the registry
# ---------------------------------------------------------------------------
def test_family_word_is_the_initialization_not_the_depth():
    """The family slot is the initialization alone: ``deep0`` is zero-init at
    ANY size and ``deep`` is Glorot at ANY size, the depth living in the size
    suffix -- so a 2x8 zero-init network is ``deep0_2x8`` and a 4x32 Glorot
    one ``deep_4x32``. Pinned because a reader of the figures will take
    ``deep`` for a statement about depth, and the suffix is where that is.
    """
    from dataclasses import replace
    shallow_zeroed = replace(ARCHITECTURES["shallow"],
                             zero_init_final_layer=True)
    assert AN.derive_display_name("shallow", shallow_zeroed) == "deep0_2x8"
    wide_glorot = replace(ARCHITECTURES["medium"], depth=4, nodes=32)
    assert AN.derive_display_name("medium", wide_glorot) == "deep_4x32"
