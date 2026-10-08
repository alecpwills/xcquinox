"""Tests for the architecture display-name layer (``xcquinox.pipeline.arch_names``).

The registry keys are storage identifiers: run directories, ``train_metadata.json``,
the cluster configurations and every pulled result are filed under them, so they
do not change. The name a figure, a table or a document shows is derived here
from the key and its configuration by one rule: the key's own leading family
word (``deep``, ``medium`` or ``shallow``), then the key's remaining tokens,
then the size ``_<depth>x<nodes>``. So ``medium`` is shown as ``medium_3x16``,
``shallow`` as ``shallow_2x8``, ``deep`` as ``deep_4x32``, and a key that
already states its size (``deep_3x16``, ``deep_geom_attn_3x16``) is shown as
itself.

Two properties carry the layer. The derived map must be ONE-TO-ONE over the
registry -- a family word taken from anywhere but the key sends ``medium`` and
``deep_3x16`` to the same shown name, which ``_build`` refuses at import -- and
no shown name may be another entry's stored key, so ``display_name`` and
``stored_key`` are plain inverses and no name can resolve into a different
network.

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
# Rule: the family word is the leading word of the STORED KEY itself; the
# tokens are the key without that word and without a trailing size; the size
# suffix is ``_{depth}x{nodes}`` and carries the depth. The name states no
# initialization -- the zeroed final layer is a run-time setting of the parent
# anchor, recorded beside the checkpoint, and no registry entry carries it.
# ---------------------------------------------------------------------------
EXPECTED_DISPLAY = {
    # the 2x8 pair
    "shallow":                     "shallow_2x8",
    "shallow_attn":                "shallow_attn_2x8",
    # the 3x16 pair whose keys state no size
    "medium":                      "medium_3x16",
    "medium_attn":                 "medium_attn_3x16",
    # the 4x32 family, whose keys state no size either
    "deep":                        "deep_4x32",
    "deep_attn":                   "deep_attn_4x32",
    "deep_cusp":                   "deep_cusp_4x32",
    "deep_cusp_attn":              "deep_cusp_attn_4x32",
    "deep_dm":                     "deep_dm_4x32",
    "deep_dm_attn":                "deep_dm_attn_4x32",
    "deep_combined":               "deep_combined_4x32",
    "deep_combined_attn":          "deep_combined_attn_4x32",
    "deep_notransform":            "deep_notransform_4x32",
    "deep_notransform_attn":       "deep_notransform_attn_4x32",
    # the 3x16 twins, each shown under its own key
    "deep_3x16":                   "deep_3x16",
    "deep_attn_3x16":              "deep_attn_3x16",
    "deep_cusp_3x16":              "deep_cusp_3x16",
    "deep_geom_3x16":              "deep_geom_3x16",
    "deep_geom_attn_3x16":         "deep_geom_attn_3x16",
    "deep_gea_3x16":               "deep_gea_3x16",
    "deep_ff_3x16":                "deep_ff_3x16",
    "deep_sine_3x16":              "deep_sine_3x16",
    "deep_dm_3x16":                "deep_dm_3x16",
    "deep_combined_3x16":          "deep_combined_3x16",
    "deep_combined_attn_3x16":     "deep_combined_attn_3x16",
    "deep_notransform_3x16":       "deep_notransform_3x16",
    "deep_notransform_attn_3x16":  "deep_notransform_attn_3x16",
    "deep_rung35_3x16":            "deep_rung35_3x16",
    "deep_rung35_attn_3x16":       "deep_rung35_attn_3x16",
    "deep_rung35ms_3x16":          "deep_rung35ms_3x16",
    "deep_rung35only_3x16":        "deep_rung35only_3x16",
    "deep_mgga_3x16":              "deep_mgga_3x16",
    "deep_mgga_attn_3x16":         "deep_mgga_attn_3x16",
    "deep_rung35_mgga_3x16":       "deep_rung35_mgga_3x16",
    "deep_cusp_mgga_3x16":         "deep_cusp_mgga_3x16",
    "deep_rung35ms_mgga_3x16":     "deep_rung35ms_mgga_3x16",
    # 2026-09-11: the width and depth completions of the pure DFS meta-GGA
    "deep_mgga_3x32":              "deep_mgga_3x32",
    "deep_mgga_4x16":              "deep_mgga_4x16",
    "deep_mgga_4x32":              "deep_mgga_4x32",
}

#: The shown names that are not themselves registry keys: the fourteen keys
#: that state no size, named under the size their configuration carries.
ALIAS_NAMES = (
    "deep_4x32", "deep_attn_4x32", "deep_cusp_4x32", "deep_cusp_attn_4x32",
    "deep_dm_4x32", "deep_dm_attn_4x32", "deep_combined_4x32",
    "deep_combined_attn_4x32", "deep_notransform_4x32",
    "deep_notransform_attn_4x32",
    "medium_3x16", "medium_attn_3x16", "shallow_2x8", "shallow_attn_2x8",
)

#: The initialization phrases neither a shown name nor an expanded key states.
#: The ``[anchored]`` protocol tag still spells out the run-time zeroing, which
#: is a property of that run and not of the architecture.
_INITIALIZATION_WORDS = ("zeroed", "Glorot")


# ---------------------------------------------------------------------------
# T1: the derived map
# ---------------------------------------------------------------------------
def test_display_map_covers_every_registry_key_exactly():
    """Every stored key is named, and nothing else is.

    Kills m2 (a family word not read off the key): derived as ``deep`` for
    every key, ``medium`` and ``deep_3x16`` take the same shown name and the
    table below no longer holds -- and, since the map refuses a collision,
    ``_build`` raises while this module is imported.
    """
    assert set(AN.DISPLAY_NAME) == set(ARCHITECTURES), (
        set(AN.DISPLAY_NAME).symmetric_difference(ARCHITECTURES))
    assert set(EXPECTED_DISPLAY) == set(ARCHITECTURES), (
        "the pinned table and the registry disagree: "
        f"{set(EXPECTED_DISPLAY).symmetric_difference(ARCHITECTURES)}")
    assert AN.DISPLAY_NAME == EXPECTED_DISPLAY


def test_display_map_is_one_to_one():
    """A collision in the inverse would make a shown name ambiguous. Kills m2."""
    shown = list(AN.DISPLAY_NAME.values())
    assert len(set(shown)) == len(shown), sorted(
        n for n in shown if shown.count(n) > 1)
    assert AN.STORED_KEY == {v: k for k, v in EXPECTED_DISPLAY.items()}


def test_stored_key_inverts_display_name_for_every_registry_key():
    for key in ARCHITECTURES:
        assert AN.stored_key(AN.display_name(key)) == key, key


def test_every_shown_name_is_its_own_key_or_one_of_the_aliases():
    """A shown name is either the stored key itself or one of the fourteen
    aliases, and never another entry's stored key.

    That is what makes the two directions plain inverses: a manifest cell, a
    pretrain directory or a figure axis may hold either spelling of the
    twenty-two keys that state their own size without the name changing
    architecture underneath it. Kills m2, which puts two keys on one name.
    """
    for key, shown in AN.DISPLAY_NAME.items():
        assert shown == key or shown in ALIAS_NAMES, (key, shown)
        if shown != key:
            assert shown not in ARCHITECTURES, (key, shown)
    assert AN.stored_key("deep_3x16") == "deep_3x16"
    assert AN.display_name("deep_3x16") == "deep_3x16"
    assert AN.display_name("medium") == "medium_3x16"
    assert AN.stored_key("medium_3x16") == "medium"
    # a name of neither kind is returned as it stands, in both directions
    assert AN.stored_key("deep0_3x16") == "deep0_3x16"
    assert AN.display_name("deep0_3x16") == "deep0_3x16"


# ---------------------------------------------------------------------------
# T1: the protocol tag
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# T1: the expanded key
# ---------------------------------------------------------------------------
def test_the_expanded_key_states_the_architecture_and_no_initialization():
    """The expanded key spells out the size, the attention heads, the
    descriptors and the rung, and states no initialization.

    Every network of the registry starts from the library's initialization
    but the sine network, whose SIREN draw the expanded key names with its
    activation, so a legend that named the initialization elsewhere would
    state a difference that does not exist; the zeroed final layer is a
    run-time setting and is spelled out by the protocol tag that carries it,
    where it belongs.

    Kills m3 (the initialization clause kept in ``expanded_key``): the sweep
    over the registry finds the phrase on every entry.
    """
    for key, cfg in ARCHITECTURES.items():
        text = AN.expanded_key(AN.display_name(key))
        assert text, key
        assert f"{int(cfg.depth)} x {int(cfg.nodes)}" in text, (key, text)
        for word in _INITIALIZATION_WORDS:
            assert word not in text, (key, word, text)

    # the settings that ARE the architecture's own are still stated
    assert "4 attention heads" in AN.expanded_key("deep_attn_3x16")
    assert "attention heads" not in AN.expanded_key("deep_3x16")
    assert "cusp descriptors" in AN.expanded_key("deep_cusp_3x16")
    assert "rung-3.5" in AN.expanded_key("deep_rung35_3x16")
    assert "meta-GGA" in AN.expanded_key("deep_mgga_3x16")

    # a protocol tag is echoed after the architecture's own settings, and a
    # known tag is spelled out: the anchor zeroes the last layer at run time
    tagged = AN.expanded_key("deep_3x16 [anchored]")
    assert tagged.startswith(AN.expanded_key("deep_3x16")), tagged
    assert "anchored on the parent" in tagged, tagged
    assert AN.expanded_key("not_an_arch") == ""


# ---------------------------------------------------------------------------
# T1: the key line
# ---------------------------------------------------------------------------
def test_the_key_line_joins_the_expanded_keys_in_order_each_once():
    """The footer of a figure: ``name: expanded`` for each name, in the order
    given, each name once, and no initialization stated.

    Kills m3 where a reader meets it -- the footer is where the expanded key is
    printed -- and pins that a name repeated across panels is not spelled out
    twice.
    """
    line = AN.key_line(["deep_3x16", "medium_3x16", "deep_3x16"])
    names = [chunk.split(":")[0] for chunk in line.split("; ")]
    assert names == ["deep_3x16", "medium_3x16"], line
    assert "3 x 16" in line, line
    for word in _INITIALIZATION_WORDS:
        assert word not in line, (word, line)
    # a name outside the registry contributes nothing
    assert AN.key_line(["not_an_arch"]) == ""


# ---------------------------------------------------------------------------
# T1: the aliases
# ---------------------------------------------------------------------------
def test_the_aliases_are_the_shown_names_that_are_not_registry_keys():
    """``ALIASES`` maps each shown name that is not itself a key to the key it
    stands for, so a configuration file may name an architecture the way a
    figure does (``config.get_architecture`` reads this map).

    Kills m4 (an empty ``ALIASES``): the fourteen names reach nothing, and a
    configuration written in the shown spelling fails at submit instead of
    resolving.
    """
    assert set(AN.ALIASES) == set(ALIAS_NAMES), (
        set(AN.ALIASES).symmetric_difference(ALIAS_NAMES))
    assert AN.ALIASES == {name: key for key, name in EXPECTED_DISPLAY.items()
                          if name in ALIAS_NAMES}
    # an alias never shadows a stored key: the registry lookup comes first, and
    # there is nothing for it to disagree with
    assert not set(AN.ALIASES) & set(ARCHITECTURES)


# ---------------------------------------------------------------------------
# The rule itself, on a configuration outside the registry
# ---------------------------------------------------------------------------
def test_the_family_word_comes_from_the_key_and_the_size_from_the_configuration():
    """``derive_display_name`` reads the family word off the stored key and the
    size off the configuration.

    Stated on configurations outside the registry, so it is the rule that is
    tested and not the entries: a key whose family word is ``medium`` keeps it
    whatever ``zero_init_final_layer`` says, which is what keeps the map
    one-to-one. Kills m2 (the family word taken from the flag) and m5 (the
    tokens read before the family word, which would leave the family word among
    them).
    """
    from types import SimpleNamespace

    zeroed = SimpleNamespace(depth=3, nodes=16, zero_init_final_layer=True)
    live = SimpleNamespace(depth=3, nodes=16, zero_init_final_layer=False)
    assert AN.derive_display_name("medium_something", zeroed) == \
        "medium_something_3x16"
    assert AN.derive_display_name("medium_something", live) == \
        "medium_something_3x16"
    assert AN.derive_display_name("medium", zeroed) == "medium_3x16"
    assert AN.derive_display_name(
        "shallow_thing_2x8",
        SimpleNamespace(depth=2, nodes=8,
                        zero_init_final_layer=True)) == "shallow_thing_2x8"
    # the size stated is the configuration's, replacing the key's own suffix
    assert AN.derive_display_name(
        "deep_thing_3x16",
        SimpleNamespace(depth=4, nodes=32,
                        zero_init_final_layer=False)) == "deep_thing_4x32"
