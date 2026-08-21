"""The DFS pretraining set: eight free atoms and 22 G2/97 molecules.

Source protocol: the pretraining notebook of the DFS code (Dick and
Fernandez-Serra, Phys. Rev. B 104, L161109 (2021)). Eight free atoms with
explicit spins -- P (2S=3), N (3), H (1), Li (1), O (2), Cl (1), Al (1),
S (2) -- plus 22 molecules of the Haunschild and Klopper G2/97 set
(Theor. Chem. Acc. 131, 1112 (2012)), every molecule run as a closed shell
(including O2 and CH2, which are open-shell species physically: the protocol
poses them at 2S = 0 and the pretraining targets follow). The meta-GGA
variant of the protocol drops H2 and N2, giving 28 systems against 30.

Geometries are committed package data (``data/dfs_pretrain_set.json``,
regenerated with ``scripts/generate_dfs_pretrain_set.py``) rather than read
from the ASE trajectory they came from: the compute nodes carry only this
package, and the fidelity certificate and the pretraining data generator must
resolve byte-identical geometries.
"""
from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:                      # import cost stays out of the run path
    from xcquinox.alec.config import MoleculeSpec

LEVELS: tuple[str, ...] = ("gga", "mgga")
# The meta-GGA variant of the DFS pretraining notebook omits these two.
MGGA_EXCLUDED: tuple[str, ...] = ("H2", "N2")

_DATA_DIR = Path(__file__).parent / "data"
_DATA_PATH = _DATA_DIR / "dfs_pretrain_set.json"

_CACHE: dict | None = None

# One element symbol with an optional count: an upper-case letter, any
# lower-case letters, then digits. Anchored matching of the whole name rejects
# anything that is not a concatenated formula.
_ELEMENT_TERM = re.compile(r"([A-Z][a-z]*)(\d*)")
_FORMULA = re.compile(r"(?:[A-Z][a-z]*\d*)+\Z")


def formula_from_name(name: str) -> tuple[tuple[str, int], ...]:
    """Element counts implied by a species name written as a formula.

    The set's names are concatenated element symbols with optional counts
    ("CH4", "AlCl3", "SiCH6"), which is the only statement of what a record
    is supposed to be that does not come from the geometry itself: a record's
    composition is derived from the coordinates the trajectory index selected,
    so name and composition disagree exactly when the index is wrong. Symbols
    are read syntactically -- "CO" is carbon and oxygen, not cobalt, which is
    the reading the set requires. Returns sorted (symbol, count) pairs, the
    ``MoleculeSpec.atom_composition`` form. Raises ValueError on a name that
    is not such a formula, or that carries a zero count.
    """
    if not isinstance(name, str) or not _FORMULA.match(name):
        raise ValueError(f"not a chemical formula: {name!r}")
    counts: dict[str, int] = {}
    for symbol, digits in _ELEMENT_TERM.findall(name):
        count = int(digits) if digits else 1
        if count < 1:
            raise ValueError(f"zero count in formula: {name!r}")
        counts[symbol] = counts.get(symbol, 0) + count
    return tuple(sorted(counts.items()))


def _load() -> dict:
    """Read and memoize the committed JSON."""
    global _CACHE
    if _CACHE is None:
        with open(_DATA_PATH) as f:
            _CACHE = json.load(f)
    return _CACHE


def dfs_pretrain_records(level: str = "gga") -> list[dict]:
    """The set's raw records for ``level``, atoms first then molecules.

    Each record is ``{"kind": "atom"|"molecule", "name", "atom", "charge",
    "spin", "atom_composition", "g2_97_index"}`` with ``atom`` a PySCF
    geometry string in Angstrom. Returns fresh copies so a caller cannot
    poison the module cache.
    """
    if level not in LEVELS:
        raise ValueError(
            f"dfs_pretrain_set level must be one of {LEVELS}, got {level!r}")
    raw = _load()
    excluded = set(MGGA_EXCLUDED) if level == "mgga" else set()
    out = [copy.deepcopy(r) for r in raw["atoms"]]
    out += [copy.deepcopy(r) for r in raw["molecules"]
            if r["name"] not in excluded]
    return out


def dfs_pretrain_systems(level: str = "gga", *,
                         basis: str = "6-311++G(3df,2pd)",
                         grid_level: int | None = 3) -> list["MoleculeSpec"]:
    """The set as :class:`~xcquinox.alec.config.MoleculeSpec` objects.

    ``basis`` / ``grid_level`` default to the production identity of the
    campaign (6-311++G(3df,2pd), grid level 3); pass a smaller pair for a
    local probe.

    The composition is sorted here rather than trusted from the file:
    MoleculeSpec is frozen and hashes every field, so an out-of-order
    composition would produce a spec that compares unequal to an otherwise
    identical one and misses in a jit cache keyed on it.
    """
    from xcquinox.alec.config import MoleculeSpec
    return [
        MoleculeSpec(
            name=r["name"], atom=r["atom"], basis=basis,
            charge=int(r["charge"]), spin=int(r["spin"]),
            atom_composition=tuple(sorted((str(s), int(n))
                                          for s, n in r["atom_composition"])),
            grid_level=grid_level,
        )
        for r in dfs_pretrain_records(level)
    ]
