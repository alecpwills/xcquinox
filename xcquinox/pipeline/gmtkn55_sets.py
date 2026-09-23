"""The Slim and diet sets of GMTKN55 as pools of this repository.

Three tracked pool caches are built here from the GMTKN55 checkout
(``data/gmtkn55``, see ``data/gmtkn55/PROVENANCE.md``):

* ``slim05`` and ``slim16``: the Slim100M sets of Gould and Vuckovic (J. Chem.
  Theory Comput. 21, 6517 (2025)), as the composition files of the published
  cloning study list them (``data/slim/``, see ``data/slim/PROVENANCE.md``):
  one line per GMTKN55 subset with the 1-based reaction indices of that subset.
  Slim05 also carries the study's pretraining molecules: the de-duplicated
  molecule list in the study's order and the 25 it draws from that list.
* ``diet150``: the Diet GMTKN55 set at 150 reactions (``data/dietgmtkn55-150``),
  defined by its element list and its subset list -- reactions, stoichiometric
  counts, reference energies and subset weights -- with the geometries, charges
  and spins of the checkout.

Every pool cache has the shape of the BH76 cache
(:mod:`xcquinox.pipeline.full_benchmark_pools`): species records and reaction
dicts in the probe schema, so the loaders below hand back the pair the held-out
evaluation consumes. Reaction and species records also carry their GMTKN55
identity (``subset``, ``index``, ``system``) and each species its Hill formula,
which the overlap reports (:func:`overlap_rows`) are written from.

The reaction lines of a subset's ``.res`` file share one grammar across the
collection: a line whose first token is ``$tmer`` or ``tmer2++`` names its
species as ``<name>/$f`` tokens, one of which may carry a brace group in the
shell's notation (``1{,A,B}``, ``{ed,ts}1``, ``01{A,B,}``), then the marker
``x``, one integer coefficient per species, the marker ``$w`` and the reference
energy in kcal/mol; anything after the reference is ignored. The reaction
index of a line is its 1-based position among the reaction lines of the file.
"""
from __future__ import annotations

import copy
import functools
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from xcquinox.pipeline.config import MoleculeSpec
from xcquinox.pipeline.full_benchmark_pools import (
    BH76_JSON_PATH,
    W411_JSON_PATH,
    _DATA_DIR,
    _atoms_to_pyscf_str,
    _build_species_dict,
    _load_pool_from_json,
    _resolve_refs_dir,
    gmtkn55_subset_dir,
)

#: The subsets the published study de-duplicates by system name rather than by
#: formula: their systems are conformers of one molecule, all of which it keeps.
CONFORMER_SUBSETS: Tuple[str, ...] = (
    "IDISP", "ICONF", "ACONF", "Amino20x4", "PCONF21", "MCONF", "SCONF", "UPU23",
    "BUT14DIOL")

#: The Slim sets whose composition files are copied under ``data/slim/``.
SLIM_NAMES: Tuple[str, ...] = ("slim05", "slim16", "slim20")
#: The Slim sets built into pool caches.
POOL_SETS: Tuple[str, ...] = ("slim05", "slim16")
DIET150 = "diet150"

#: The subsets whose species keep their bare GMTKN55 system name: they are the
#: species of the two tracked pools, and BH76RC's reactions are written over
#: the BH76 directory.
BARE_SUBSETS: Tuple[str, ...] = ("BH76", "BH76RC", "W4-11")

_REACTION_LINE_TOKENS = ("$tmer", "tmer2++")
_SPECIES_SUFFIX = "/$f"

_REPO_ROOT = Path(__file__).resolve().parents[2]
SLIM_DATA_DIR = _REPO_ROOT / "data" / "slim"
DIET_DATA_DIR = _REPO_ROOT / "data" / "dietgmtkn55-150"
_SLIM_COMPOSITION_FILES = {
    "slim05": "Slim100M_05_composition.txt",
    "slim16": "Slim100M_16_composition.txt",
    "slim20": "Slim100M_20_composition.txt",
}
DIET_ELEMENTS_FILE = DIET_DATA_DIR / "AllElements-150.yaml"
DIET_SUBSETS_FILE = DIET_DATA_DIR / "SubsetGMTKN55_150.yaml"

SLIM_JSON_PATHS: Dict[str, Path] = {
    name: _DATA_DIR / f"{name}_pool.json" for name in POOL_SETS}
DIET150_JSON_PATH = _DATA_DIR / "diet150_pool.json"
OVERLAP_JSON_PATHS: Dict[str, Path] = {
    "diet150": _DATA_DIR / "diet150_overlap.json",
    "bh76": _DATA_DIR / "bh76_overlap.json",
    "w411": _DATA_DIR / "w411_overlap.json",
}

#: The published study's pretraining draw: the first repetition of its cloning
#: script, which seeds the legacy global generator at 42 + i for repetition i
#: and draws 25 molecules without replacement.
PRETRAIN_DRAW_SEED = 42
PRETRAIN_DRAW_SIZE = 25

#: The unpaired-electron table the published study applies to a single neutral
#: atom; every other system gets the parity of its electron count.
_STUDY_ATOM_SPINS = {
    "Al": 1, "B": 1, "Li": 1, "Na": 1, "Si": 2, "Be": 0, "C": 2, "Cl": 1,
    "F": 1, "H": 1, "N": 3, "O": 2, "P": 3, "S": 2, "Ar": 0, "Br": 1, "Ne": 0,
    "Sb": 3, "Bi": 3, "Te": 2, "I": 1,
}

_SOURCES = {
    "slim05": ("GMTKN55 (Goerigk, Hansen, Bauer, Ehrlich, Najibi, Grimme, PCCP "
               "19 32184 (2017)); the Slim05 composition of Gould and Vuckovic "
               "(J. Chem. Theory Comput. 21, 6517 (2025)) as listed by "
               "data/slim/Slim100M_05_composition.txt; the reaction lines of the "
               "checkout, data/gmtkn55/PROVENANCE.md"),
    "slim16": ("GMTKN55 (Goerigk, Hansen, Bauer, Ehrlich, Najibi, Grimme, PCCP "
               "19 32184 (2017)); the Slim16 composition of Gould and Vuckovic "
               "(J. Chem. Theory Comput. 21, 6517 (2025)) as listed by "
               "data/slim/Slim100M_16_composition.txt; the reaction lines of the "
               "checkout, data/gmtkn55/PROVENANCE.md"),
    "diet150": ("GMTKN55 (Goerigk, Hansen, Bauer, Ehrlich, Najibi, Grimme, PCCP "
                "19 32184 (2017)); the Diet GMTKN55 set of 150 systems (Gould, "
                "PCCP 20, 27735 (2018)) as data/dietgmtkn55-150/AllElements-150.yaml "
                "and SubsetGMTKN55_150.yaml define it; geometries from the "
                "checkout, data/gmtkn55/PROVENANCE.md"),
}


# ---------------------------------------------------------------------------
# The composition files
# ---------------------------------------------------------------------------

def composition_path(name: str) -> Path:
    """The composition file of a Slim set under ``data/slim/``."""
    if name not in _SLIM_COMPOSITION_FILES:
        raise ValueError(f"unknown Slim set {name!r}; the sets are {SLIM_NAMES}")
    return SLIM_DATA_DIR / _SLIM_COMPOSITION_FILES[name]


def parse_composition(text: str) -> Tuple[Tuple[str, int], ...]:
    """``((subset, index), ...)`` in the file's order.

    A line is a subset name followed by the 1-based reaction indices of that
    subset in the set; a blank line or a line starting with ``#`` is skipped. A
    line repeating an index, or a second line for a subset, is refused: either
    would enter a reaction twice or place a subset's reactions at two positions
    and shift every later position of the study's molecule list."""
    out: List[Tuple[str, int]] = []
    seen_subsets: set = set()
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            raise ValueError(f"composition line names no reaction index: {raw!r}")
        subset = parts[0]
        if subset in seen_subsets:
            raise ValueError(f"composition names a subset twice: {raw!r}")
        seen_subsets.add(subset)
        try:
            indices = [int(p) for p in parts[1:]]
        except ValueError:
            raise ValueError(
                f"composition line carries a non-integer index: {raw!r}") from None
        if any(i < 1 for i in indices):
            raise ValueError(f"composition indices are 1-based: {raw!r}")
        if len(set(indices)) != len(indices):
            raise ValueError(f"composition line repeats an index: {raw!r}")
        out.extend((subset, i) for i in indices)
    return tuple(out)


# ---------------------------------------------------------------------------
# The reaction-line grammar
# ---------------------------------------------------------------------------

def expand_braces(token: str) -> List[str]:
    """The names a species token stands for.

    One brace group, in the shell's notation, anywhere in the token: ``1{,A,B}``
    is ``1``, ``1A`` and ``1B``; ``{ed,ts}1`` is ``ed1`` and ``ts1``; an empty
    alternative is the bare prefix and suffix. A token without a group is
    itself. A nested, unbalanced or second group is refused rather than
    expanded into a name no directory carries."""
    n_open, n_close = token.count("{"), token.count("}")
    if n_open == 0 and n_close == 0:
        return [token]
    if n_open != 1 or n_close != 1:
        raise ValueError(
            f"species token {token!r} must carry exactly one balanced brace group")
    i, j = token.index("{"), token.index("}")
    if j < i:
        raise ValueError(f"species token {token!r} closes its brace group before "
                         "opening it")
    prefix, body, suffix = token[:i], token[i + 1:j], token[j + 1:]
    return [prefix + alt + suffix for alt in body.split(",")]


@dataclass(frozen=True)
class ResLine:
    """One reaction line: its 1-based index among the reaction lines of the
    file, its species in the line's order, the integer coefficient of each
    and the reference energy in kcal/mol."""
    index: int
    systems: List[str]
    coeffs: List[int]
    ref: float


def parse_res(text: str) -> List[ResLine]:
    """The reaction lines of a ``.res`` (or ``.resRC``) file.

    Comments after ``#`` are dropped and fields split on any whitespace. The
    species tokens stand before the marker ``x`` with their ``/$f`` suffix
    stripped and brace groups expanded; the coefficients follow, one per
    species, until ``$w``; the token after ``$w`` is the reference and
    anything after it is ignored. A line whose coefficient count is not its
    species count is refused."""
    out: List[ResLine] = []
    index = 0
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        tokens = line.split()
        if tokens[0] not in _REACTION_LINE_TOKENS:
            continue
        index += 1
        if "x" not in tokens or "$w" not in tokens:
            raise ValueError(
                f"reaction line {index} lacks the x or $w marker: {raw!r}")
        ix, iw = tokens.index("x"), tokens.index("$w")
        if not 0 < ix < iw < len(tokens) - 1:
            raise ValueError(f"reaction line {index} is not species, x, "
                             f"coefficients, $w, reference: {raw!r}")
        systems: List[str] = []
        for token in tokens[1:ix]:
            base = (token[:-len(_SPECIES_SUFFIX)]
                    if token.endswith(_SPECIES_SUFFIX) else token)
            systems.extend(expand_braces(base))
        try:
            coeffs = [int(t) for t in tokens[ix + 1:iw]]
            ref = float(tokens[iw + 1])
        except ValueError:
            raise ValueError(
                f"reaction line {index} carries a non-numeric coefficient or "
                f"reference: {raw!r}") from None
        if len(coeffs) != len(systems):
            raise ValueError(
                f"reaction line {index}: {len(systems)} species but "
                f"{len(coeffs)} coefficients in {raw!r}")
        if any(c == 0 for c in coeffs):
            raise ValueError(
                f"reaction line {index} carries a zero coefficient: {raw!r}")
        out.append(ResLine(index=index, systems=systems, coeffs=coeffs, ref=ref))
    return out


def species_directory(subset: str) -> Path:
    """The directory holding a subset's species; BH76RC's is BH76's."""
    return gmtkn55_subset_dir("BH76" if subset == "BH76RC" else subset)


def subset_res_lines(subset: str) -> List[ResLine]:
    """The reaction lines of a subset: ``BH76/.resRC`` for BH76RC, the
    subset's own ``.res`` otherwise."""
    directory = species_directory(subset)
    res = directory / (".resRC" if subset == "BH76RC" else ".res")
    if not res.is_file():
        raise FileNotFoundError(f"no reaction file for {subset!r} at {res}")
    return parse_res(res.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Names, formulas and records
# ---------------------------------------------------------------------------

def subset_tag(subset: str) -> str:
    """The lower-cased subset name with hyphens as underscores."""
    return subset.lower().replace("-", "_")


def _tracked_pool_species(path: Path) -> Dict[str, Tuple[str, int, int]]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {sd["name"]: (sd["atom"], int(sd["charge"]), int(sd["spin"]))
            for sd in data["species"]}


@functools.lru_cache(maxsize=None)
def w411_qualified_names() -> frozenset:
    """The W4-11 system names that BH76 also carries with another geometry,
    charge or spin. The union of the two tracked pools keeps BH76's species
    for such a name, so a W4-11 species named bare would be scored on BH76's
    molecule; those names are prefixed instead."""
    bh76 = _tracked_pool_species(BH76_JSON_PATH)
    w411 = _tracked_pool_species(W411_JSON_PATH)
    return frozenset(name for name, key in w411.items()
                     if name in bh76 and bh76[name] != key)


def species_name(subset: str, system: str) -> str:
    """The pool name of a GMTKN55 system.

    BH76, BH76RC and W4-11 systems keep their bare name: they are the species
    of the tracked pools, and one name is one geometry across every pool. The
    exception is a W4-11 name BH76 carries with another geometry
    (:func:`w411_qualified_names`), which is prefixed ``w411_``. Every other
    subset's system is prefixed with the subset's tag, so a name two subsets
    share is two pool species."""
    if subset in ("BH76", "BH76RC"):
        return system
    if subset == "W4-11":
        return f"w411_{system}" if system in w411_qualified_names() else system
    return f"{subset_tag(subset)}_{system}"


def reaction_name(subset: str, index: int) -> str:
    """The pool name of a reaction: the subset's tag and its 1-based index."""
    return f"{subset_tag(subset)}_{int(index):03d}"


def hill_formula(symbols: Iterable[str]) -> str:
    """The Hill formula of a list of element symbols: C first and H second
    when carbon is present, every other element (and H without carbon) in
    alphabetical order, a count above one written after the symbol."""
    counts: Dict[str, int] = {}
    for symbol in symbols:
        counts[symbol] = counts.get(symbol, 0) + 1
    order: List[str] = []
    if "C" in counts:
        order.append("C")
        if "H" in counts:
            order.append("H")
    order.extend(sorted(e for e in counts if e not in order))
    return "".join(f"{e}{counts[e] if counts[e] > 1 else ''}" for e in order)


def _symbols_of(atom: str) -> List[str]:
    return [tok.split()[0] for tok in atom.split(";") if tok.split()]


def species_record(subset: str, system: str) -> Dict[str, Any]:
    """The species record of a GMTKN55 system: the tracked pools' fields
    (name, atom, atom_composition, charge, spin) under the pool name, then
    the subset, the system name and the Hill formula of the geometry."""
    sd = _build_species_dict(system, species_directory(subset))
    return {
        "name": species_name(subset, system),
        "atom": sd["atom"],
        "atom_composition": sd["atom_composition"],
        "charge": sd["charge"],
        "spin": sd["spin"],
        "subset": subset,
        "system": system,
        "formula": hill_formula(_symbols_of(sd["atom"])),
    }


def reaction_record(subset: str, line: ResLine, *, source_pool: str,
                    species: Mapping[str, Dict[str, Any]],
                    weight: Optional[float] = None) -> Dict[str, Any]:
    """The reaction dict of a reaction line in the probe schema, with the
    GMTKN55 identity beside it.

    ``reactants`` and ``products`` are the pool names of the species with a
    negative and a positive coefficient, in the line's relative order;
    ``coeffs`` are the coefficients in that reactants-then-products order and
    ``systems`` the GMTKN55 system names in the same order, so the three lists
    align. ``species`` supplies the charge and spin maps."""
    pairs = list(zip(line.systems, line.coeffs))
    reactant_pairs = [(s, c) for s, c in pairs if c < 0]
    product_pairs = [(s, c) for s, c in pairs if c > 0]
    ordered = reactant_pairs + product_pairs
    names = [species_name(subset, s) for s, _ in ordered]
    reactants = names[:len(reactant_pairs)]
    products = names[len(reactant_pairs):]
    rec: Dict[str, Any] = {
        "name": reaction_name(subset, line.index),
        "source_pool": source_pool,
        "reactants": reactants,
        "products": products,
        "coeffs": [float(c) for _, c in ordered],
        "reaction_energy_ref": float(line.ref),
        "source": _SOURCES[source_pool],
        "species_spins": {n: int(species[n]["spin"]) for n in names},
        "species_charges": {n: int(species[n]["charge"]) for n in names},
        "subset": subset,
        "index": int(line.index),
        "systems": [s for s, _ in ordered],
    }
    if weight is not None:
        rec["weight"] = float(weight)
    return rec


# ---------------------------------------------------------------------------
# The published study's molecule list and draw
# ---------------------------------------------------------------------------

def paper_molecules(entries: Iterable[Tuple[str, int, Sequence[str]]]
                    ) -> List[Dict[str, Any]]:
    """The study's molecule list over ``entries`` = ``(subset, index,
    systems)`` in composition order, each reaction's systems in the line's
    order.

    Per subset, with the seen-set reset at every subset: a standard subset
    keeps the first system of each Hill formula, a conformer subset the first
    of each system name. Each molecule is ``{"subset", "index", "system",
    "name", "formula"}``."""
    out: List[Dict[str, Any]] = []
    seen_by_subset: Dict[str, set] = {}
    cache: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for subset, index, systems in entries:
        seen = seen_by_subset.setdefault(subset, set())
        for system in systems:
            key = (subset, system)
            if key not in cache:
                cache[key] = species_record(subset, system)
            rec = cache[key]
            marker = system if subset in CONFORMER_SUBSETS else rec["formula"]
            if marker in seen:
                continue
            seen.add(marker)
            out.append({"subset": subset, "index": int(index), "system": system,
                        "name": rec["name"], "formula": rec["formula"]})
    return out


def paper_draw(n_total: int, n: int = PRETRAIN_DRAW_SIZE,
               seed: int = PRETRAIN_DRAW_SEED) -> Tuple[int, ...]:
    """The positions the study's cloning script draws from a molecule list of
    ``n_total`` entries: the legacy generator seeded at ``seed``, ``n`` draws
    without replacement, sorted."""
    import numpy as np
    state = np.random.RandomState(seed)
    drawn = state.choice(range(int(n_total)), int(n), replace=False)
    return tuple(sorted(int(i) for i in drawn))


def _study_spin(symbols: Sequence[str], charge: int) -> int:
    """The unpaired-electron count the study's own runs used: the table for a
    single neutral atom, the parity of the electron count otherwise."""
    from ase.data import atomic_numbers
    if len(symbols) == 1 and charge == 0 and symbols[0] in _STUDY_ATOM_SPINS:
        return _STUDY_ATOM_SPINS[symbols[0]]
    n_elec = sum(atomic_numbers[s] for s in symbols) - int(charge)
    return n_elec % 2


# ---------------------------------------------------------------------------
# The pool builders
# ---------------------------------------------------------------------------

def build_slim_pool_dict(name: str) -> Dict[str, Any]:
    """The pool cache of a Slim set: ``species``, ``reactions``, the study's
    ``molecules`` and, for slim05, the study's pretraining draw
    (``pretrain_draw``) with the drawn molecules' records
    (``pretrain_molecules``)."""
    if name not in POOL_SETS:
        raise ValueError(f"no pool is built for {name!r}; the sets are {POOL_SETS}")
    entries = parse_composition(composition_path(name).read_text(encoding="utf-8"))
    lines_by_subset: Dict[str, List[ResLine]] = {}
    species: Dict[str, Dict[str, Any]] = {}
    reactions: List[Dict[str, Any]] = []
    molecule_entries: List[Tuple[str, int, List[str]]] = []
    for subset, index in entries:
        lines = lines_by_subset.setdefault(subset, subset_res_lines(subset))
        if not 1 <= index <= len(lines):
            raise ValueError(
                f"{name}: {subset} has {len(lines)} reactions, index {index} "
                "is outside them")
        line = lines[index - 1]
        for system in line.systems:
            pool_name = species_name(subset, system)
            if pool_name not in species:
                species[pool_name] = species_record(subset, system)
        reactions.append(reaction_record(subset, line, source_pool=name,
                                         species=species))
        molecule_entries.append((subset, index, list(line.systems)))
    pool: Dict[str, Any] = {
        "species": list(species.values()),
        "reactions": reactions,
        "molecules": paper_molecules(molecule_entries),
    }
    if name == "slim05":
        indices = paper_draw(len(pool["molecules"]))
        pool["pretrain_draw"] = {"seed": PRETRAIN_DRAW_SEED,
                                 "n": PRETRAIN_DRAW_SIZE,
                                 "indices": list(indices)}
        pool["pretrain_molecules"] = [
            _pretrain_molecule(pool["molecules"][i], species) for i in indices]
    return pool


def _pretrain_molecule(molecule: Dict[str, Any],
                       species: Mapping[str, Dict[str, Any]]) -> Dict[str, Any]:
    """The pretraining record of a drawn molecule: the record form the
    pretraining-set resolver consumes, the GMTKN55 identity, and the study's
    own spin rule beside the checkout's spin."""
    rec = species[molecule["name"]]
    return {
        "kind": "molecule",
        "name": rec["name"],
        "atom": rec["atom"],
        "atom_composition": rec["atom_composition"],
        "charge": rec["charge"],
        "spin": rec["spin"],
        "subset": molecule["subset"],
        "index": molecule["index"],
        "system": molecule["system"],
        "formula": rec["formula"],
        "spin_parity_rule": _study_spin(_symbols_of(rec["atom"]), rec["charge"]),
    }


def _diet_definition() -> Tuple[Dict[str, Any], Dict[str, Any]]:
    import yaml
    elements = yaml.safe_load(DIET_ELEMENTS_FILE.read_text(encoding="utf-8"))
    subsets = yaml.safe_load(DIET_SUBSETS_FILE.read_text(encoding="utf-8"))
    return elements, subsets["Systems"]


def _species_signature(subset: str, system: str, coeff: int
                       ) -> Tuple[int, int, int, Tuple[str, ...]]:
    sd = _build_species_dict(system, species_directory(subset))
    return (int(coeff), int(sd["charge"]), int(sd["spin"]),
            tuple(_symbols_of(sd["atom"])))


#: The largest difference, in Angstrom, between an internal distance of a
#: species in the checkout and the same distance in the element list under
#: which the two are one geometry. The list states positions to five
#: decimals, so rounding alone moves a distance by less than 2e-5 A, and
#: 322 of the 332 diet species sit under that; one (the S66 monomer 42A)
#: sits at 9e-5 A, and the nine species whose checkout geometry the list
#: does not describe sit at 2e-4 A and above.
ELEMENT_LIST_DISTANCE_TOL = 1e-4


def _positions_of_atom_string(atom: str) -> List[Tuple[float, float, float]]:
    out = []
    for tok in atom.split(";"):
        parts = tok.split()
        if parts:
            out.append((float(parts[1]), float(parts[2]), float(parts[3])))
    return out


def _internal_distances(positions: Sequence[Sequence[float]]) -> List[float]:
    """The pairwise distances of ``positions`` in row-major pair order."""
    pos = [tuple(float(v) for v in p) for p in positions]
    return [math.dist(pos[i], pos[j])
            for i in range(len(pos)) for j in range(i + 1, len(pos))]


def geometry_deviation(atom: str, positions: Sequence[Sequence[float]]) -> float:
    """The largest difference between an internal distance of the PySCF
    ``atom`` string and the same distance of ``positions`` (Angstrom, the
    same atom order), a frame-independent comparison; zero for a single
    atom."""
    a = _internal_distances(_positions_of_atom_string(atom))
    b = _internal_distances(positions)
    if len(a) != len(b):
        raise ValueError("geometry_deviation: the two geometries differ in "
                         "their atom count")
    return max((abs(x - y) for x, y in zip(a, b)), default=0.0)


def _list_signature(entry: Mapping[str, Any]) -> Tuple[int, int, int, Tuple[str, ...]]:
    return (int(entry["Count"]), int(entry["Charge"]), int(entry["UHF"]),
            tuple(str(e) for e in entry["Elements"]))


def _diet_species_entries(subset: str, line: ResLine, block: Mapping[str, Any]
                          ) -> Dict[str, Tuple[str, Mapping[str, Any]]]:
    """``{system: (list name, list entry)}`` for the systems of ``line``:
    each system is paired with the element list's entry of equal
    stoichiometric count, charge, unpaired electrons and element sequence,
    the closest geometry deciding among several; every entry is used once.
    Entries equally close to a system are accepted when they are one
    geometry themselves (the two monomers of a homodimer), the first name
    taken."""
    remaining: Dict[str, Mapping[str, Any]] = dict(block["Species"])
    out: Dict[str, Tuple[str, Mapping[str, Any]]] = {}
    for system, coeff in zip(line.systems, line.coeffs):
        sig = _species_signature(subset, system, coeff)
        atom = _build_species_dict(system, species_directory(subset))["atom"]
        scored = sorted(
            (geometry_deviation(atom, entry["Positions"]), name)
            for name, entry in remaining.items() if _list_signature(entry) == sig)
        if not scored:
            raise ValueError(
                f"diet150: {subset} {line.index}: the element list carries no "
                f"entry for the checkout's {system!r}")
        name = scored[0][1]
        for deviation, other in scored[1:]:
            if deviation != scored[0][0]:
                break
            same = _internal_distances(remaining[name]["Positions"])
            tied = _internal_distances(remaining[other]["Positions"])
            if max((abs(x - y) for x, y in zip(same, tied)), default=0.0) \
                    > ELEMENT_LIST_DISTANCE_TOL:
                raise ValueError(
                    f"diet150: {subset} {line.index}: the element list's "
                    f"{name!r} and {other!r} are equally close to the "
                    f"checkout's {system!r} and are not one geometry")
        out[system] = (name, remaining.pop(name))
    return out


def _diet_species_record(subset: str, system: str, list_name: str,
                         entry: Mapping[str, Any]) -> Dict[str, Any]:
    """The species record of a diet system: the checkout's record where its
    internal distances agree with the element list's positions within
    :data:`ELEMENT_LIST_DISTANCE_TOL`, and otherwise the list's positions,
    since the list's reference energies belong to the list's geometries.
    The record names the list's entry, the source of its geometry and the
    deviation of the checkout's geometry from the list's. A species named as
    a tracked pool's is refused where the two disagree: its name reuses
    that pool's reference file and geometry."""
    rec = species_record(subset, system)
    if (int(rec["charge"]), int(rec["spin"])) != (int(entry["Charge"]),
                                                  int(entry["UHF"])):
        raise ValueError(
            f"diet150: {subset} {system!r} carries charge {rec['charge']} and "
            f"{rec['spin']} unpaired electrons in the checkout and "
            f"{entry['Charge']} and {entry['UHF']} in the element list")
    deviation = geometry_deviation(rec["atom"], entry["Positions"])
    rec["element_list_name"] = str(list_name)
    rec["element_list_deviation"] = float(deviation)
    if deviation <= ELEMENT_LIST_DISTANCE_TOL:
        rec["geometry_source"] = "checkout"
        return rec
    if subset in BARE_SUBSETS:
        raise ValueError(
            f"diet150: {subset} {system!r} is a tracked pool's species and the "
            f"element list places it at another geometry (an internal distance "
            f"differs by {deviation:.2e} A)")
    atoms_ang = [(str(e), float(p[0]), float(p[1]), float(p[2]))
                 for e, p in zip(entry["Elements"], entry["Positions"])]
    rec["atom"] = _atoms_to_pyscf_str(atoms_ang)
    rec["geometry_source"] = "element_list"
    return rec


def _diet_line(subset: str, index: int, block: Mapping[str, Any]) -> ResLine:
    """The reaction line of the checkout that the element list's block for
    ``(subset, index)`` describes: the line at that index of the subset's own
    reaction file or, for BH76RC, of the BH76 file, whichever carries the
    block's species (stoichiometric count, charge, unpaired electrons and
    element sequence per species). The reference energy is the block's."""
    want = sorted((int(v["Count"]), int(v["Charge"]), int(v["UHF"]),
                   tuple(str(e) for e in v["Elements"]))
                  for v in block["Species"].values())
    candidates = [subset]
    if subset == "BH76RC":
        candidates.append("BH76")
    for source in candidates:
        lines = subset_res_lines(source)
        if not 1 <= index <= len(lines):
            continue
        line = lines[index - 1]
        got = sorted(_species_signature(subset, s, c)
                     for s, c in zip(line.systems, line.coeffs))
        if got == want:
            return ResLine(index=int(index), systems=list(line.systems),
                           coeffs=list(line.coeffs), ref=float(block["Energy"]))
    raise ValueError(
        f"diet150: no reaction line of {' or '.join(candidates)} at index "
        f"{index} carries the species the element list names for "
        f"{subset} {index}: {list(block['Species'])}")


def build_diet150_pool_dict() -> Dict[str, Any]:
    """The pool cache of the diet set: every reaction of the subset list with
    its subset weight; species from the checkout where the checkout's
    geometry is the element list's, and from the list where it is not
    (:func:`_diet_species_record`)."""
    elements, subsets = _diet_definition()
    species: Dict[str, Dict[str, Any]] = {}
    reactions: List[Dict[str, Any]] = []
    for subset, (weight, indices) in subsets.items():
        for index in indices:
            block = elements[subset][int(index)]
            if abs(float(block["Weight"]) - float(weight)) > 1e-9:
                raise ValueError(
                    f"diet150: {subset} {index} carries weight {block['Weight']} "
                    f"in the element list and {weight} in the subset list")
            line = _diet_line(subset, int(index), block)
            entries = _diet_species_entries(subset, line, block)
            for system in line.systems:
                pool_name = species_name(subset, system)
                list_name, entry = entries[system]
                if pool_name not in species:
                    species[pool_name] = _diet_species_record(
                        subset, system, list_name, entry)
                    continue
                again = geometry_deviation(species[pool_name]["atom"],
                                           entry["Positions"])
                if again > ELEMENT_LIST_DISTANCE_TOL:
                    raise ValueError(
                        f"diet150: the element list places {subset} {system!r} "
                        f"at two geometries ({subset} {index} differs from the "
                        f"recorded one by {again:.2e} A)")
            reactions.append(reaction_record(subset, line, source_pool=DIET150,
                                             species=species,
                                             weight=float(weight)))
    return {"species": list(species.values()), "reactions": reactions}


# ---------------------------------------------------------------------------
# The loaders
# ---------------------------------------------------------------------------

_SLIM_CACHE: Dict[Tuple[str, str, Any, Any], Tuple[Dict[str, MoleculeSpec],
                                                    List[Dict[str, Any]]]] = {}
_DIET_CACHE: Dict[Tuple[str, Any, Any], Tuple[Dict[str, MoleculeSpec],
                                               List[Dict[str, Any]]]] = {}


def load_full_slim(name: str, basis: str = "def2-svp",
                   grid_level: int | None = 1,
                   refs_dir: str | os.PathLike | None = None
                   ) -> Tuple[Dict[str, MoleculeSpec], List[Dict[str, Any]]]:
    """``({species_name: MoleculeSpec}, [reaction_dict, ...])`` of a Slim set
    from its tracked cache; the loader semantics of ``load_full_bh76``."""
    if name not in POOL_SETS:
        raise ValueError(f"no pool is built for {name!r}; the sets are {POOL_SETS}")
    resolved_refs = _resolve_refs_dir(refs_dir)
    key = (name, basis, grid_level, resolved_refs)
    if key not in _SLIM_CACHE:
        _SLIM_CACHE[key] = _load_pool_from_json(
            SLIM_JSON_PATHS[name], lambda: build_slim_pool_dict(name), basis,
            grid_level, resolved_refs)
    return _SLIM_CACHE[key]


def load_full_diet150(basis: str = "def2-svp", grid_level: int | None = 1,
                      refs_dir: str | os.PathLike | None = None
                      ) -> Tuple[Dict[str, MoleculeSpec], List[Dict[str, Any]]]:
    """``({species_name: MoleculeSpec}, [reaction_dict, ...])`` of the diet
    set from its tracked cache, each reaction carrying its subset ``weight``."""
    resolved_refs = _resolve_refs_dir(refs_dir)
    key = (basis, grid_level, resolved_refs)
    if key not in _DIET_CACHE:
        _DIET_CACHE[key] = _load_pool_from_json(
            DIET150_JSON_PATH, build_diet150_pool_dict, basis, grid_level,
            resolved_refs)
    return _DIET_CACHE[key]


def slim_pretrain_records(name: str = "slim05") -> List[Dict[str, Any]]:
    """The study's drawn pretraining molecules of a Slim set, in the drawn
    order, as records the pretraining-set resolver consumes (fresh copies)."""
    if name not in POOL_SETS:
        raise ValueError(f"no pool is built for {name!r}; the sets are {POOL_SETS}")
    path = SLIM_JSON_PATHS[name]
    data = (json.loads(path.read_text(encoding="utf-8")) if path.is_file()
            else build_slim_pool_dict(name))
    if "pretrain_molecules" not in data:
        raise ValueError(f"{name} carries no pretraining draw; slim05 does")
    return [copy.deepcopy(r) for r in data["pretrain_molecules"]]


# ---------------------------------------------------------------------------
# The overlap reports
# ---------------------------------------------------------------------------

def overlap_rows(pool: Mapping[str, Any],
                 training: Mapping[str, Sequence[Mapping[str, Any]]]
                 ) -> List[Dict[str, Any]]:
    """Per reaction of ``pool``, per species, the training sets that carry the
    species exactly (same subset and system) and by formula (same Hill formula,
    charge and spin). ``training`` maps a set's name to its species records
    (``subset``, ``system``, ``formula``, ``charge``, ``spin``). A report: the
    pool is not modified."""
    exact: Dict[Tuple[str, str], set] = {}
    by_formula: Dict[Tuple[str, int, int], set] = {}
    for set_name, records in training.items():
        for rec in records:
            exact.setdefault((rec["subset"], rec["system"]), set()).add(set_name)
            by_formula.setdefault(
                (rec["formula"], int(rec["charge"]), int(rec["spin"])),
                set()).add(set_name)
    species = {sd["name"]: sd for sd in pool["species"]}
    rows: List[Dict[str, Any]] = []
    for rxn in pool["reactions"]:
        row_species: Dict[str, Dict[str, List[str]]] = {}
        for name in list(rxn["reactants"]) + list(rxn["products"]):
            if name in row_species:
                continue
            sd = species[name]
            row_species[name] = {
                "exact": sorted(exact.get((sd["subset"], sd["system"]), ())),
                "formula": sorted(by_formula.get(
                    (sd["formula"], int(sd["charge"]), int(sd["spin"])), ())),
            }
        rows.append({"name": rxn["name"], "subset": rxn["subset"],
                     "index": int(rxn["index"]), "species": row_species})
    return rows


def _tracked_pool_with_identity(path: Path, subset: str) -> Dict[str, Any]:
    """A tracked BH76 or W4-11 cache with the subset, system and formula of
    every species added, so it can be read like the pools built here."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    species = []
    for sd in data["species"]:
        rec = dict(sd)
        rec["subset"] = subset
        rec["system"] = sd["name"]
        rec["formula"] = hill_formula(_symbols_of(sd["atom"]))
        species.append(rec)
    reactions = []
    for i, rxn in enumerate(data["reactions"], start=1):
        rec = dict(rxn)
        rec.setdefault("subset", subset)
        rec.setdefault("index", i)
        reactions.append(rec)
    return {"species": species, "reactions": reactions}


def dfs_training_species() -> List[Dict[str, Any]]:
    """The species of the DFS training set as overlap records: every species
    of the set's training points, named as the points name them, with the
    BH76 pool's species carrying the BH76 identity and the rest the set's own."""
    from xcquinox.pipeline.training_points import (build_dfs_pool_points,
                                                  species_union_from_points)
    bh76 = _tracked_pool_species(BH76_JSON_PATH)
    out: List[Dict[str, Any]] = []
    for atoms in species_union_from_points(build_dfs_pool_points()):
        name = str(atoms.info.get("name") or atoms.get_chemical_formula())
        subset = "BH76" if name in bh76 else "DFS"
        out.append({"name": name, "subset": subset, "system": name,
                    "formula": hill_formula(list(atoms.get_chemical_symbols())),
                    "charge": int(atoms.info.get("charge", 0)),
                    "spin": int(atoms.info.get("spin", 0))})
    return out


def build_overlap_reports() -> Dict[str, List[Dict[str, Any]]]:
    """The overlap reports of the three held-out pools against the three
    training sets (slim05, slim16 and the DFS set), keyed as
    :data:`OVERLAP_JSON_PATHS`."""
    slim = {name: build_slim_pool_dict(name) for name in POOL_SETS}
    training = {name: pool["species"] for name, pool in slim.items()}
    training["dfs"] = dfs_training_species()
    pools = {
        "diet150": build_diet150_pool_dict(),
        "bh76": _tracked_pool_with_identity(BH76_JSON_PATH, "BH76"),
        "w411": _tracked_pool_with_identity(W411_JSON_PATH, "W4-11"),
    }
    return {key: overlap_rows(pool, training) for key, pool in pools.items()}
