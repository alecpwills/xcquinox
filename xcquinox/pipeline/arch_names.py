"""Shown names for the registry architectures, derived from the key and the size.

The registry keys are storage identifiers: run directories, ``train_metadata.json``, the
cluster configurations and every pulled result are filed under them, so they do not
change. The name a figure, a table or a document shows is derived here from the key and
its configuration by one rule:

* the key's family word (``deep``, ``medium`` or ``shallow``), as the key states it;
* then the descriptor, attention and network tokens of the stored key (``attn``, ``cusp``,
  ``dm``, ``combined``, ``notransform``, ``rung35``, ``mgga``, ``kan``, ...) as they are;
* then the size, ``_<depth>x<nodes>``, which is where the depth is stated.

So ``deep_3x16`` is shown as itself, ``deep`` as ``deep_4x32``, ``deep_cusp_attn`` as
``deep_cusp_attn_4x32``, ``medium`` as ``medium_3x16`` and ``shallow_attn`` as
``shallow_attn_2x8``. A key whose name already carries its size is its own shown name; the
keys without a size suffix are reached through the aliases (:data:`ALIASES`). The
initialization is no part of the name: no registry entry zeroes its final layer (the sine
network's SIREN initialization is a token of its key, ``sine``), and the parent anchor, which
zeroes it, is a run protocol marked by a tag. Files written before this
rule (the v7 documents and their CSVs) carry the earlier names, ``deep_3x16`` for
``medium`` and ``deep0_*`` for the entries that were then zero-initialized; their
``arch_stored`` column is the key.

The expanded key (:func:`expanded_key`) spells the settings out, for legends and captions,
and :func:`key_line` joins them for a figure footer. A protocol tag ``[tag]`` appended to a
shown name (``deep_3x16 [25 cycles]``, ``deep_3x16 [anchored]``) marks a run trained under
a different protocol; it passes through :func:`stored_key` and is echoed by
:func:`expanded_key`, spelled out when the tag is a known one.

This module imports nothing from matplotlib or numpy so that ``config.py`` can use it for
the aliases.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Tuple

_FAMILY_WORDS = ("deep", "medium", "shallow")
_SIZE_SUFFIX = re.compile(r"_(\d+)x(\d+)$")
_TAG = re.compile(r"^(.*?)\s*\[([^\]]+)\]$")

_DESCRIPTOR_TEXT = {
    "cusp": "cusp descriptors",
    "dm_statistics": "density-matrix descriptors",
    "rung35": "rung-3.5 occupancy",
    "rung35_multishell": "rung-3.5 multishell occupancy",
    "metagga": "meta-GGA (SCAN parent)",
}

#: the protocol tags with a spelled-out form; an unknown tag is echoed as written
_TAG_TEXT = {
    "anchored": "anchored on the parent (last layer zeroed at run time)",
}


def _family_word(stored: str) -> str:
    """The leading family word of a stored key (``deep_cusp_3x16`` -> ``deep``,
    ``medium`` -> ``medium``). A key outside the three families is refused: the
    rule would have nothing to name it by."""
    for word in _FAMILY_WORDS:
        if stored == word or stored.startswith(word + "_"):
            return word
    raise ValueError(
        f"arch_names: the stored key {stored!r} starts with none of the family "
        f"words {_FAMILY_WORDS}, so no shown name can be derived for it")


def _tokens(stored: str) -> str:
    """The descriptor and attention tokens of a stored key: the trailing size
    suffix is removed first (``deep_3x16`` -> ``deep``), then the leading
    family word (``deep`` -> ``""``, ``deep_cusp`` -> ``cusp``)."""
    name = _SIZE_SUFFIX.sub("", stored)
    for word in _FAMILY_WORDS:
        if name == word:
            return ""
        if name.startswith(word + "_"):
            return name[len(word) + 1:]
    return name


def derive_display_name(stored: str, cfg) -> str:
    """The shown name of the registry key ``stored`` with configuration ``cfg``:
    the key's family word, its tokens and ``_<depth>x<nodes>``. The
    initialization is read from nothing: no registry entry zeroes its final
    layer, and a configuration that does derives the same name."""
    depth, nodes = int(cfg.depth), int(cfg.nodes)
    family = _family_word(stored)
    tokens = _tokens(stored)
    return f"{family}_{tokens}_{depth}x{nodes}" if tokens else f"{family}_{depth}x{nodes}"


def split_tag(name: str) -> Tuple[str, str]:
    """``(base, tag)`` of a name with an optional trailing `` [tag]``; the tag
    is ``""`` when absent."""
    m = _TAG.match(name)
    return (m.group(1), m.group(2)) if m else (name, "")


_split_tag = split_tag


def _registry():
    from xcquinox.pipeline.config import ARCHITECTURES
    return ARCHITECTURES


def _build() -> Tuple[Dict[str, str], Dict[str, str]]:
    shown: Dict[str, str] = {}
    inverse: Dict[str, str] = {}
    for key, cfg in _registry().items():
        name = derive_display_name(key, cfg)
        if name in inverse:
            raise ValueError(
                f"arch_names: the shown name {name!r} is derived for both "
                f"{inverse[name]!r} and {key!r}; the rule does not separate them")
        shown[key] = name
        inverse[name] = key
    return shown, inverse


DISPLAY_NAME, STORED_KEY = _build()

#: the shown names that are not themselves registry keys, as configuration-file aliases
ALIASES: Dict[str, str] = {
    name: key for name, key in STORED_KEY.items() if name not in DISPLAY_NAME}


def display_name(stored: str, protocol: str | None = None) -> str:
    """The shown name of a stored key (a name outside the registry is returned as is),
    with `` [protocol]`` appended when a protocol tag is given."""
    name = DISPLAY_NAME.get(stored, stored)
    return f"{name} [{protocol}]" if protocol else name


def stored_key(name: str) -> str:
    """The registry key of a shown name (a `` [tag]`` suffix is dropped); a name that is
    not a shown name is returned as is. A key that states its size is its own shown name
    and resolves to itself."""
    base, _tag = _split_tag(name)
    return STORED_KEY.get(base, base)


def expanded_key(name: str) -> str:
    """The settings the shown name stands for, spelled out; ``""`` for an unknown name.

    The size, the attention heads, the descriptors and the rung are the architecture's
    own and are stated; the descriptor coordinates and the flags that are inert under
    them are run settings and are not."""
    base, tag = _split_tag(name)
    key = STORED_KEY.get(base)
    if key is None:
        if base in DISPLAY_NAME:            # a stored-only key, mapped through
            key = base
        else:
            return ""
    cfg = _registry()[key]
    parts = [f"{int(cfg.depth)} x {int(cfg.nodes)}"]
    if getattr(cfg, "attention", False):
        parts.append(f"{int(cfg.num_heads)} attention heads")
    for d in getattr(cfg, "descriptors", ()) or ():
        dname = getattr(d, "name", None) or str(d)
        if dname in _DESCRIPTOR_TEXT and dname != "metagga":
            parts.append(_DESCRIPTOR_TEXT[dname])
    from xcquinox.pipeline.config import ArchitectureConfig
    if ArchitectureConfig.is_meta_gga(cfg):
        parts.append(_DESCRIPTOR_TEXT["metagga"])
    coefficient = getattr(cfg, "gea_mu", None)
    if coefficient is not None:
        named = {"pbe": "PBE's gradient expansion (mu = beta pi^2/3)",
                 "gea": "the exact gradient expansion (mu = 10/81)"}
        parts.append("exchange curvature fixed at " + (
            named.get(coefficient, f"the gradient expansion mu = {coefficient}")
            if isinstance(coefficient, str)
            else f"the gradient expansion mu = {float(coefficient):.6g}"))
    m = int(getattr(cfg, "fourier_features", 0) or 0)
    if m:
        parts.append(f"Fourier features (m = {m}, sigma = "
                     f"{float(getattr(cfg, 'fourier_scale', 1.0)):g} per "
                     "coordinate range)")
    if getattr(cfg, "activation", "gelu") == "sine":
        parts.append("sine activation (SIREN, omega_0 = "
                     f"{float(getattr(cfg, 'omega_0', 1.0)):g})")
    if getattr(cfg, "network", "mlp") == "kan":
        order = int(getattr(cfg, "kan_order", 0))
        degree = {1: "linear", 2: "quadratic", 3: "cubic"}.get(
            order, f"degree-{order}")
        parts.append(f"Kolmogorov-Arnold network, {degree} B-splines on "
                     f"{int(getattr(cfg, 'kan_grid', 0))} intervals per edge, "
                     "SiLU base")
    text = ", ".join(parts)
    return f"{text}; {_TAG_TEXT.get(tag, tag)}" if tag else text


def key_line(names: Iterable[str]) -> str:
    """``"name: expanded; name: expanded"`` over the given names, each once, in order."""
    seen: List[str] = []
    for n in names:
        if n not in seen:
            seen.append(n)
    return "; ".join(f"{n}: {expanded_key(n)}" for n in seen if expanded_key(n))
