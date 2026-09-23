"""Tests for the architecture display-name layer (``xcquinox.pipeline.arch_names``).

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

from xcquinox.pipeline.config import ARCHITECTURES


class _Lazy:
    """``xcquinox.pipeline.arch_names``, resolved at first use.

    The module is imported inside each test rather than at module scope so a
    missing module fails the tests that need it instead of erroring the whole
    collection (which would hide every other test in the same run).
    """

    def __getattr__(self, name):
        from xcquinox.pipeline import arch_names
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
    # 2026-09-11: the width and depth completions of the pure DFS meta-GGA
    "deep_mgga_3x32":              "deep0_mgga_3x32",
    "deep_mgga_4x16":              "deep0_mgga_4x16",
    "deep_mgga_4x32":              "deep0_mgga_4x32",
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


# ---------------------------------------------------------------------------
# T1: the protocol tag
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# T1: the expanded key
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# T1: the key line
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# T1: the aliases
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The rule itself, on a configuration outside the registry
# ---------------------------------------------------------------------------
