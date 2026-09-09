"""Shown names for the registry architectures, derived from what each network is.

The registry keys are storage identifiers: run directories, ``train_metadata.json``, the
cluster configurations and every pulled result are filed under them, so they do not
change. The name a figure, a table or a document shows is derived here from the
configuration by one rule, so that two keys that build the same network (``medium`` and
``deep_3x16`` are both depth 3, width 16, the same inputs) are told apart by the one
setting that differs, and the name says which setting that is:

* family ``deep``: the default, Glorot initialization of every layer;
* family ``deep0``: the last layer's weight and bias zeroed at construction
  (``zero_init_final_layer``), so the untrained network is the LDA and pre-training
  starts from it;
* then the descriptor and attention tokens of the stored key (``attn``, ``cusp``, ``dm``,
  ``combined``, ``notransform``, ``rung35``, ``mgga``, ...) as they are;
* then the size, ``_<depth>x<nodes>``, which is where the depth is stated.

So ``medium`` is shown as ``deep_3x16``, ``deep_3x16`` as ``deep0_3x16``, ``medium_attn``
as ``deep_attn_3x16``, ``deep_cusp_mgga_3x16`` as ``deep0_cusp_mgga_3x16``, ``shallow`` as
``deep_2x8``. Two shown names coincide with stored keys of other configurations
(``deep_3x16``, ``deep_attn_3x16``); every function here resolves a name in the SHOWN
sense, and the places that hold a stored key (a manifest cell, a pretrain directory) call
:func:`display_name` before the name reaches anything that draws or prints.

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

_ZEROED_TEXT = "last layer zeroed (pre-training starts at the LDA)"
_GLOROT_TEXT = "Glorot initialization"


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
    """The shown name of the registry key ``stored`` with configuration ``cfg``."""
    depth, nodes = int(cfg.depth), int(cfg.nodes)
    family = "deep0" if bool(getattr(cfg, "zero_init_final_layer", False)) else "deep"
    tokens = _tokens(stored)
    return f"{family}_{tokens}_{depth}x{nodes}" if tokens else f"{family}_{depth}x{nodes}"


def split_tag(name: str) -> Tuple[str, str]:
    """``(base, tag)`` of a name with an optional trailing `` [tag]``; the tag
    is ``""`` when absent."""
    m = _TAG.match(name)
    return (m.group(1), m.group(2)) if m else (name, "")


_split_tag = split_tag


def _registry():
    from xcquinox.alec.config import ARCHITECTURES
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
    not a shown name is returned as is. A name that is both a shown name and a stored key
    resolves in the shown sense."""
    base, _tag = _split_tag(name)
    return STORED_KEY.get(base, base)


def expanded_key(name: str) -> str:
    """The settings the shown name stands for, spelled out; ``""`` for an unknown name.

    The size, the initialization, the attention heads, the descriptors and the rung are
    the architecture's own and are stated; the descriptor coordinates and the flags
    that are inert under them are run settings and are not."""
    base, tag = _split_tag(name)
    key = STORED_KEY.get(base)
    if key is None:
        if base in DISPLAY_NAME:            # a stored-only key, mapped through
            key = base
        else:
            return ""
    cfg = _registry()[key]
    parts = [f"{int(cfg.depth)} x {int(cfg.nodes)}"]
    parts.append(_ZEROED_TEXT if getattr(cfg, "zero_init_final_layer", False)
                 else _GLOROT_TEXT)
    if getattr(cfg, "attention", False):
        parts.append(f"{int(cfg.num_heads)} attention heads")
    for d in getattr(cfg, "descriptors", ()) or ():
        dname = getattr(d, "name", None) or str(d)
        if dname in _DESCRIPTOR_TEXT and dname != "metagga":
            parts.append(_DESCRIPTOR_TEXT[dname])
    from xcquinox.alec.config import ArchitectureConfig
    if ArchitectureConfig.is_meta_gga(cfg):
        parts.append(_DESCRIPTOR_TEXT["metagga"])
    text = ", ".join(parts)
    return f"{text}; {_TAG_TEXT.get(tag, tag)}" if tag else text


def key_line(names: Iterable[str]) -> str:
    """``"name: expanded; name: expanded"`` over the given names, each once, in order."""
    seen: List[str] = []
    for n in names:
        if n not in seen:
            seen.append(n)
    return "; ".join(f"{n}: {expanded_key(n)}" for n in seen if expanded_key(n))
