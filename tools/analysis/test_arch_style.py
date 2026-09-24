"""Figure styling of the architectures (``tools/analysis/arch_style.py``).

The module is the single source of the order, the colour and the rung every
per-architecture figure draws with, and it is keyed by SHOWN names while the
manifest cells and pretrain directories hold STORED keys. What is asserted
here for the v8 geometric pair: it sits in the canonical order directly after
the cusp architecture it twins, both spellings of each name reach that order,
each carries a colour of its own that no other architecture carries and that
is not the unknown-architecture default, the stored key resolves to the shown
name's colour, and the rung derived from the registry is the GGA rung.

Oracles: ``ARCH_ORDER``, ``ARCH_COLOR`` and the module's own style functions,
read against the display names the registry derives.

The module is loaded from its path rather than imported as a package: there is
no ``__init__.py`` under ``tools``, and every other consumer of ``arch_style``
(the figure scripts, the notebook builder) loads it the same way.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib

# A file-backed backend: importing the module pulls in ``matplotlib.pyplot``
# for its colour maps, and a test run has no display.
matplotlib.use("Agg")

_HERE = Path(__file__).resolve().parent


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


AS = _load("arch_style")

#: The pair in the spelling the figures hold (shown names).
_SHOWN_PAIR = ("deep0_geom_3x16", "deep0_geom_attn_3x16")

#: The same pair in the spelling the manifest cells hold (registry keys).
_STORED_PAIR = ("deep_geom_3x16", "deep_geom_attn_3x16")


def test_the_geometric_pair_is_ordered_and_coloured_in_the_figures():
    """The pair follows its cusp twin in the order and carries its own colours.

    Position: directly after ``deep0_cusp_3x16``, the architecture the pair
    differs from in name and attention alone, so the three read as one family
    on every axis.

    Colour: distinct from each other, distinct from every colour a
    non-geometric name carries, and not ``#333333`` -- the value the
    ``_3x16``-suffix strip falls back to when the base name has no entry, so a
    pair added to the orders but not to the colour table would be drawn in the
    same grey as any unknown architecture. The comparison skips the entries of
    the pair itself: a base name and the width twin that inherits from it
    share one colour by construction (``deep_rung35`` and
    ``deep0_rung35_3x16`` already do).

    Coverage: every ordered name has a colour entry, which a pair inserted
    among the eight tab10 base names of ``_STORED_ORDER`` would break by
    pushing one of those names out of the table.
    """
    cusp = AS.ARCH_ORDER.index("deep0_cusp_3x16")
    assert AS.ARCH_ORDER.index("deep0_geom_3x16") == cusp + 1, AS.ARCH_ORDER
    assert AS.ARCH_ORDER.index("deep0_geom_attn_3x16") == cusp + 2, AS.ARCH_ORDER

    for name in _SHOWN_PAIR + _STORED_PAIR:
        assert AS.in_order(name), name

    colours = {name: AS.ARCH_COLOR[name] for name in _SHOWN_PAIR}
    assert (colours["deep0_geom_3x16"].lower()
            != colours["deep0_geom_attn_3x16"].lower()), colours
    others = {value.lower() for key, value in AS.ARCH_COLOR.items()
              if "geom" not in key}
    for name, value in colours.items():
        assert value.lower() != "#333333", name
        assert value.lower() not in others, (name, value)

    for stored, shown in zip(_STORED_PAIR, _SHOWN_PAIR):
        assert AS.arch_color(stored) == AS.ARCH_COLOR[shown], stored

    for name in _SHOWN_PAIR:
        assert AS.rung_of(name) == AS.RUNG_GGA, name

    uncoloured = [name for name in AS.ARCH_ORDER if name not in AS.ARCH_COLOR]
    assert uncoloured == [], uncoloured
