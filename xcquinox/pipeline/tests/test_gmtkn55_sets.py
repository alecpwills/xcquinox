"""The Slim and diet150 pools of GMTKN55 (``xcquinox.pipeline.gmtkn55_sets``).

The module builds three tracked pool caches -- Slim05, Slim16 and the 150-reaction
diet set -- out of the GMTKN55 checkout, the three composition files copied under
``data/slim/`` and ``data/dietgmtkn55-150/``. Every count asserted below was measured
from that checkout with the grammar the module implements; the oracles are named per
test. Tests that read the checkout skip where it is absent (``_missing_subsets``); the
rules that do not need the real data are exercised against synthetic trees under
``tmp_path``.
"""
from __future__ import annotations

import ast
import inspect
import json
import os
import sys
from pathlib import Path

import pytest

from xcquinox.pipeline.full_benchmark_pools import (
    BH76_JSON_PATH,
    W411_JSON_PATH,
)

#: The repository root: this file sits at ``xcquinox/pipeline/tests/``.
_REPO_ROOT = Path(__file__).resolve().parents[3]

#: The composition files the Slim pools are built from.
_COMPOSITION_NAMES = ("Slim100M_05_composition.txt", "Slim100M_16_composition.txt",
                      "Slim100M_20_composition.txt")

#: The composition file of each Slim set, by the name the builders take.
_SLIM_COMPOSITION_FILES = {f"slim{n}": f"Slim100M_{n}_composition.txt"
                           for n in ("05", "16", "20")}

#: The nine subsets the paper de-duplicates by system name rather than by formula.
_CONFORMER_SUBSETS = ("IDISP", "ICONF", "ACONF", "Amino20x4", "PCONF21", "MCONF",
                      "SCONF", "UPU23", "BUT14DIOL")


def _sets():
    """The module under test, imported at call time so a test's fixtures are built
    before the import and a fixture fault is distinguishable from the absent module."""
    import xcquinox.pipeline.gmtkn55_sets as gs
    return gs


def _missing_subsets(names) -> list[str]:
    """The named subsets whose ``.res`` is on disk under neither layout of the GMTKN55
    clone. Resolved without the module's own resolver, so a skip is never decided by
    the code under test."""
    env = os.environ.get("XCQUINOX_GMTKN55_DIR")
    root = Path(env) if env else _REPO_ROOT / "data" / "gmtkn55"
    return [n for n in names
            if not (root / n / ".res").is_file()
            and not (root / "gmtkn55" / n / ".res").is_file()]


def _missing_compositions() -> list[str]:
    """The composition files absent from ``data/slim/``."""
    base = _REPO_ROOT / "data" / "slim"
    return [n for n in _COMPOSITION_NAMES if not (base / n).is_file()]


def _missing_slim_subsets(name, base=None) -> list[str]:
    """The subsets a Slim set's composition file names whose ``.res`` is absent
    from the checkout, in the file's order and without repeats. The composition
    is read here rather than through the module under test, as the two guards
    above are; a set whose composition file is itself absent reports nothing,
    since :func:`_missing_compositions` is the guard that covers it. ``base``
    is the directory holding the composition files, the tracked one by
    default, so the reading itself can be exercised on a synthetic file. A name
    that is not a Slim set is refused rather than reported complete, since an
    empty list reads as "every subset is present"."""
    return _missing_subsets(_named_slim_subsets(name, base=base))


def _named_slim_subsets(name, base=None) -> tuple:
    """The checkout directories a Slim set's composition file names, in the
    file's order and without repeats."""
    if name not in _SLIM_COMPOSITION_FILES:
        raise ValueError(f"not a Slim set: {name!r}")
    base = Path(base) if base is not None else _REPO_ROOT / "data" / "slim"
    path = base / _SLIM_COMPOSITION_FILES[name]
    if not path.is_file():
        return ()
    named = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        subset = line.split()[0]
        # BH76RC has no directory of its own: its reactions are written over
        # the BH76 species, in that directory's .resRC.
        named.append("BH76" if subset == "BH76RC" else subset)
    return tuple(dict.fromkeys(named))


def _skip_without_slim_sources(*names) -> None:
    """Skip when a Slim set cannot be built here: the composition files are
    tracked but the GMTKN55 checkout is not (``.gitignore`` keeps all of it but
    the provenance file), so a fresh clone has the one and not the other."""
    absent = _missing_compositions()
    if absent:
        pytest.skip(f"the composition files are not on disk: {absent}")
    named = {s for n in names for s in _named_slim_subsets(n)}
    missing = sorted({s for n in names for s in _missing_slim_subsets(n)})
    if missing:
        pytest.skip(f"the GMTKN55 checkout is short {len(missing)} of the "
                    f"{len(named)} subsets {', '.join(names)} name: "
                    f"{missing[:5]}")


#: The entry points of the module under test that read the checkout for a Slim
#: set: a test naming one of them builds a Slim pool and needs the guard.
_SLIM_BUILDER_CALLS = ("build_slim_pool_dict", "slim_pretrain_records",
                       "build_overlap_reports")

#: The guard's name as a test's syntax tree spells it.
_SLIM_GUARD = "_skip_without_slim_sources"

#: The Slim set names the builders take.
_SLIM_SET_NAMES = frozenset(_SLIM_COMPOSITION_FILES)


def _slim_builder_tests() -> list:
    """``(function, argument tuples)`` for every test of this module that builds
    a Slim pool, found by the builders each one names rather than listed here,
    so a test added later is covered without being added to a list. A
    parameterized test carries one argument tuple per case, read from its own
    mark; a test the sweep cannot call is refused rather than passed over."""
    out = []
    for name, fn in sorted(vars(sys.modules[__name__]).items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        source = inspect.getsource(fn)
        if not any(call in source for call in _SLIM_BUILDER_CALLS):
            continue
        argnames, argsets = (), [()]
        for mark in getattr(fn, "pytestmark", []):
            if mark.name != "parametrize":
                continue
            argnames = tuple(a.strip() for a in mark.args[0].split(","))
            argsets = [v if isinstance(v, tuple) else (v,) for v in mark.args[1]]
        takes = tuple(inspect.signature(fn).parameters)
        assert takes == argnames, (
            f"{name} takes {takes}, which the sweep cannot supply")
        out.append((fn, argsets))
    return out


def _tracked_pool(path) -> dict:
    """One tracked benchmark pool cache as a dict."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _mini_checkout(tmp_path, subsets) -> Path:
    """A synthetic GMTKN55 tree: ``subsets`` maps a subset name to ``(res_text,
    {system: (xyz_lines, charge, unpaired)})``. Returns the root to point
    ``XCQUINOX_GMTKN55_DIR`` at."""
    root = tmp_path / "mini"
    for subset, (res_text, systems) in subsets.items():
        sdir = root / subset
        sdir.mkdir(parents=True, exist_ok=True)
        (sdir / ".res").write_text(res_text, encoding="utf-8")
        for system, (atoms, charge, unpaired) in systems.items():
            wdir = sdir / system
            wdir.mkdir(parents=True, exist_ok=True)
            body = "\n".join(f"{e} {x:.10f} {y:.10f} {z:.10f}" for e, x, y, z in atoms)
            (wdir / "struc.xyz").write_text(f"{len(atoms)}\n\n{body}\n", encoding="utf-8")
            (wdir / "coord").write_text(
                f"$coord\n$eht charge={charge} unpaired={unpaired}\n$end\n",
                encoding="utf-8")
    return root


# ---------------------------------------------------------------------------
# 1. The line grammar
# ---------------------------------------------------------------------------

def test_brace_expansion_covers_every_placement_in_the_checkout():
    """The brace group appears as a prefix, as a suffix, with an empty alternative in
    any position, and not at all. Nested and unbalanced groups are refused rather than
    expanded into a name no directory carries.

    Oracle: the placements measured in the checkout -- AHB21 ``1{,A,B}``, S66
    ``01{A,B,}``, BHDIV10 ``{ed,ts}1``, DIPCS10 ``c4h4{,_2+}``, W4-11 ``{h2,h}``,
    WATER27 ``{OHmH2O,OHm,H2O}`` and the bare token of BH76.
    """
    gs = _sets()
    assert gs.expand_braces("1{,A,B}") == ["1", "1A", "1B"]
    assert gs.expand_braces("01{A,B,}") == ["01A", "01B", "01"]
    assert gs.expand_braces("{ed,ts}1") == ["ed1", "ts1"]
    assert gs.expand_braces("c4h4{,_2+}") == ["c4h4", "c4h4_2+"]
    assert gs.expand_braces("{h2,h}") == ["h2", "h"]
    assert gs.expand_braces("{OHmH2O,OHm,H2O}") == ["OHmH2O", "OHm", "H2O"]
    assert gs.expand_braces("n2ohts") == ["n2ohts"]
    for bad in ("{a{b,c}}", "{a,b", "a,b}", "{a}{b}"):
        with pytest.raises(ValueError):
            gs.expand_braces(bad)


def test_the_reaction_index_counts_only_reaction_lines(tmp_path, monkeypatch):
    """The ``.res`` files open with a shell header and carry comments between the
    reaction lines, so the index of a reaction is its 1-based position among the lines
    whose first token is ``$tmer`` or ``tmer2++``, not its line number.

    Oracle: a synthetic ``.res`` carrying the header of the checkout, a comment line
    and a blank line between two reactions.
    """
    res = (
        'if [ "$TMER" == "" ]\nthen\n  tmer=tmer2++\nfi\nf=$1\nw=$2\n'
        "# a comment\n"
        "$tmer  a/$f  b/$f  x  -1  1  $w  1.5\n"
        "\n"
        "tmer2++\tc/$f\td/$f\tx\t-1\t1\t$w\t-2.5  # trailing note\n"
        "$tmer  {e,f}/$f  x  -1  2  $w  3.5 0 1.0\n"
    )
    root = _mini_checkout(tmp_path, {"FAKE": (res, {})})
    monkeypatch.setenv("XCQUINOX_GMTKN55_DIR", str(root))
    gs = _sets()
    lines = gs.parse_res(res)
    assert [ln.index for ln in lines] == [1, 2, 3]
    assert [ln.systems for ln in lines] == [["a", "b"], ["c", "d"], ["e", "f"]]
    assert [ln.coeffs for ln in lines] == [[-1, 1], [-1, 1], [-1, 2]]
    assert [ln.ref for ln in lines] == [1.5, -2.5, 3.5]


def test_a_line_whose_coefficients_do_not_match_its_species_is_refused():
    """A miscounted line would silently drop or duplicate a term of the reaction
    energy, so the parser refuses it."""
    gs = _sets()
    with pytest.raises(ValueError):
        gs.parse_res("$tmer  a/$f  b/$f  c/$f  x  -1  1  $w  1.0\n")


# ---------------------------------------------------------------------------
# 2. The general parser against the two tracked pools
# ---------------------------------------------------------------------------

def test_the_general_parser_reproduces_the_tracked_pools():
    """The general parser reads the two subsets the tracked caches were built from and
    gives, reaction by reaction, the same reactants, products, coefficients and
    reference. A parser that agreed with the caches only on one of them would be a
    second grammar, not a generalization of the one in use.

    Oracle: ``xcquinox/pipeline/data/{bh76,w411}_full_pool.json`` as tracked.
    """
    missing = _missing_subsets(("BH76", "W4-11"))
    if missing:
        pytest.skip(f"the GMTKN55 clone carries no .res for {', '.join(missing)}")
    tracked = {"BH76": _tracked_pool(BH76_JSON_PATH)["reactions"],
               "W4-11": _tracked_pool(W411_JSON_PATH)["reactions"]}
    gs = _sets()
    for subset, reactions in tracked.items():
        lines = gs.subset_res_lines(subset)
        assert len(lines) == len(reactions), subset
        for line, rxn in zip(lines, reactions):
            reactants = [s for s, c in zip(line.systems, line.coeffs) if c < 0]
            products = [s for s, c in zip(line.systems, line.coeffs) if c > 0]
            coeffs = [float(c) for c in line.coeffs if c < 0] + \
                     [float(c) for c in line.coeffs if c > 0]
            assert reactants == rxn["reactants"], (subset, line.index)
            assert products == rxn["products"], (subset, line.index)
            assert coeffs == rxn["coeffs"], (subset, line.index)
            assert line.ref == pytest.approx(rxn["reaction_energy_ref"], abs=1e-9)


def test_the_reaction_coupling_subset_reads_its_own_reaction_file():
    """BH76RC is not a directory of its own: its reactions live in ``BH76/.resRC`` and
    its species in the BH76 directory.

    Oracle: the 30 ``tmer2++`` lines of ``BH76/.resRC``, whose first is
    ``h + n2o -> oh + n2`` at -64.91 kcal/mol.
    """
    missing = _missing_subsets(("BH76",))
    if missing:
        pytest.skip("the GMTKN55 clone carries no BH76/.res")
    gs = _sets()
    lines = gs.subset_res_lines("BH76RC")
    assert len(lines) == 30
    first = lines[0]
    assert first.systems == ["h", "n2o", "oh", "n2"]
    assert first.coeffs == [-1, -1, 1, 1]
    assert first.ref == pytest.approx(-64.91, abs=1e-9)
    assert gs.species_directory("BH76RC") == gs.species_directory("BH76")


# ---------------------------------------------------------------------------
# 3. The composition files
# ---------------------------------------------------------------------------

def test_the_composition_files_parse_in_file_order():
    """Each composition file names 100 reactions as ``SUBSET index index ...`` lines,
    and the parsed order is the file's: the paper's molecule list is built by walking
    the subsets in that order and the indices as listed.

    Oracle: the three files as copied under ``data/slim/``; Slim05's W4-11 line.
    """
    gs = _sets()
    absent = _missing_compositions()
    assert not absent, f"the composition files are not copied yet: {absent}"
    for name in ("slim05", "slim16", "slim20"):
        entries = gs.parse_composition(gs.composition_path(name).read_text())
        assert len(entries) == 100, name
        subsets = list(dict.fromkeys(s for s, _ in entries))
        assert subsets == sorted(subsets), name
    slim05 = gs.parse_composition(gs.composition_path("slim05").read_text())
    assert [i for s, i in slim05 if s == "W4-11"] == [
        23, 42, 60, 62, 65, 84, 97, 105, 109, 118, 124, 132, 134]
    assert [s for s, _ in slim05][:3] == ["AHB21", "AHB21", "AHB21"]


def test_a_composition_line_repeating_an_index_is_refused():
    """A repeated index would enter the same reaction twice and shift every later
    position of the paper's molecule list."""
    gs = _sets()
    with pytest.raises(ValueError):
        gs.parse_composition("AHB21 1 2 1\n")


# ---------------------------------------------------------------------------
# 4. The paper's de-duplication
# ---------------------------------------------------------------------------

def test_the_paper_rule_keeps_the_first_of_each_formula_per_subset_and_each_conformer(
        tmp_path, monkeypatch):
    """A standard subset contributes the first system of each Hill formula and a
    conformer subset the first of each system name, with the seen-set reset at every
    subset -- so a formula repeated in a later subset is kept again, and two conformers
    of one formula are both kept.

    Oracle: a synthetic tree of two standard subsets sharing a formula, plus a
    conformer subset carrying that formula under two system names.
    """
    h2 = (("H", 0.0, 0.0, 0.0), ("H", 0.0, 0.0, 0.74))
    hf = (("H", 0.0, 0.0, 0.0), ("F", 0.0, 0.0, 0.92))
    root = _mini_checkout(tmp_path, {
        "AAA": ("$tmer  a/$f  b/$f  x  -1  1  $w  1.0\n"
                "$tmer  c/$f  b/$f  x  -1  1  $w  2.0\n",
                {"a": (h2, 0, 0), "b": (hf, 0, 0), "c": (h2, 0, 0)}),
        "BBB": ("$tmer  d/$f  x  -1  $w  3.0\n", {"d": (h2, 0, 0)}),
        "ACONF": ("$tmer  p/$f  q/$f  x  -1  1  $w  4.0\n",
                  {"p": (h2, 0, 0), "q": (h2, 0, 0)}),
    })
    monkeypatch.setenv("XCQUINOX_GMTKN55_DIR", str(root))
    gs = _sets()
    entries = (("AAA", 1, ["a", "b"]), ("AAA", 2, ["c", "b"]),
               ("BBB", 1, ["d"]), ("ACONF", 1, ["p", "q"]))
    molecules = gs.paper_molecules(entries)
    assert [(m["subset"], m["index"], m["system"]) for m in molecules] == [
        ("AAA", 1, "a"), ("AAA", 1, "b"),
        ("BBB", 1, "d"),
        ("ACONF", 1, "p"), ("ACONF", 1, "q"),
    ]
    assert [m["formula"] for m in molecules] == ["H2", "FH", "H2", "H2", "H2"]
    assert "ACONF" in gs.CONFORMER_SUBSETS and "AAA" not in gs.CONFORMER_SUBSETS


# ---------------------------------------------------------------------------
# 5. The paper's draw
# ---------------------------------------------------------------------------

def test_the_paper_draw_is_numpys_legacy_choice():
    """The 25 pretraining molecules are drawn with the legacy global generator at seed
    42 and sorted, which is what the published cloning script does for its first
    repetition. A different seed draws a different set, and passing the population as a
    range rather than a list does not move the draw.

    Oracle: ``numpy.random.seed`` / ``numpy.random.choice`` recomputed here.
    """
    import numpy as np
    gs = _sets()
    n_total = 169
    np.random.seed(42)
    expected = tuple(sorted(int(i) for i in
                            np.random.choice(range(n_total), 25, replace=False)))
    assert gs.paper_draw(n_total) == expected
    np.random.seed(42)
    as_list = tuple(sorted(int(i) for i in
                           np.random.choice(list(range(n_total)), 25, replace=False)))
    assert as_list == expected
    np.random.seed(0)
    other = tuple(sorted(int(i) for i in
                         np.random.choice(range(n_total), 25, replace=False)))
    assert gs.paper_draw(n_total, seed=0) == other
    assert other != expected
    assert len(set(gs.paper_draw(n_total))) == 25


# ---------------------------------------------------------------------------
# 6. The Slim pools
# ---------------------------------------------------------------------------

#: Per Slim set: reactions, distinct (subset, system) identities over those reactions,
#: distinct pool species names, and molecules under the paper's per-subset rule.
#: Measured from the checkout with the module's grammar.
#: Every distinct (subset, system) pair of a Slim set is one species of its
#: pool, since a species name carries its subset's tag: the identity count and
#: the species count agree, where the bare names of three subsets once
#: collapsed four pairs of slim05 and five of slim16 onto one species each.
_SLIM_COUNTS = {
    "slim05": {"reactions": 100, "identities": 211, "species": 211, "molecules": 169},
    "slim16": {"reactions": 100, "identities": 239, "species": 239, "molecules": 216},
}

#: The molecules the paper's first repetition draws from Slim05, as
#: ``subset:index:system``, at the drawn positions of the molecule list.
_SLIM05_DRAWN = (
    (12, "AHB21", 13, "13"), (15, "ALKBDE10", 3, "ca"), (16, "ALKBDE10", 3, "o"),
    (18, "ALKBDE10", 5, "k"), (19, "ALKBDE10", 5, "f"), (24, "BH76", 4, "h"),
    (29, "BH76", 42, "H2O"), (30, "BH76", 42, "RKT02"), (31, "BH76", 66, "oh"),
    (45, "CHB6", 3, "24A"), (51, "G21EA", 14, "EA_14n"), (60, "G21IP", 21, "IP_64"),
    (68, "G2RC", 2, "1"), (75, "G2RC", 14, "18"), (85, "G2RC", 23, "14"),
    (105, "HEAVYSB11", 10, "cl2"), (108, "PA26", 5, "nh3"), (109, "PA26", 5, "nh3p"),
    (119, "RG18", 2, "ar"), (133, "RG18", 14, "c2h2Ar"), (138, "SIE4x4", 2, "h"),
    (143, "W4-11", 23, "ch3f"), (157, "W4-11", 105, "hnnn"),
    (162, "W4-11", 132, "oclo"), (165, "WATER27", 20, "OHmH2O"),
)


@pytest.mark.parametrize("name", ("slim05", "slim16"))
def test_the_slim_pools_have_the_counts_of_the_checkout(name):
    """Each Slim set carries 100 reactions; the species identities its reactions touch,
    the pool species names and the molecules the paper's rule keeps are the measured
    counts of the checkout.

    Oracle: the composition file walked over the checkout's ``.res`` and ``struc.xyz``.
    """
    _skip_without_slim_sources(name)
    gs = _sets()
    pool = gs.build_slim_pool_dict(name)
    want = _SLIM_COUNTS[name]
    assert len(pool["reactions"]) == want["reactions"]
    identities = {(r["subset"], s) for r in pool["reactions"] for s in r["systems"]}
    assert len(identities) == want["identities"]
    names = [s["name"] for s in pool["species"]]
    assert len(names) == len(set(names)), "a species name is recorded twice"
    assert len(names) == want["species"]
    assert len(pool["molecules"]) == want["molecules"]


def test_the_slim05_draw_names_the_paper_s_twenty_five_molecules():
    """The drawn positions index the molecule list, and the pinned molecules are the
    ones at those positions: a draw that moved, or a molecule list whose order changed,
    changes which 25 systems are pretrained on.

    Oracle: the molecule list built from the checkout, indexed by the legacy draw.
    """
    _skip_without_slim_sources("slim05")
    gs = _sets()
    pool = gs.build_slim_pool_dict("slim05")
    draw = pool["pretrain_draw"]
    assert draw["seed"] == 42 and draw["n"] == 25
    indices = list(draw["indices"])
    assert indices == sorted(set(indices))
    assert all(0 <= i < len(pool["molecules"]) for i in indices)
    assert indices == [i for i, _s, _x, _y in _SLIM05_DRAWN]
    drawn = [pool["molecules"][i] for i in indices]
    assert [(m["subset"], m["index"], m["system"]) for m in drawn] == [
        (s, x, y) for _i, s, x, y in _SLIM05_DRAWN]
    assert [(m["subset"], m["index"], m["system"]) for m in pool["pretrain_molecules"]] \
        == [(s, x, y) for _i, s, x, y in _SLIM05_DRAWN]


def test_every_slim_species_name_carries_one_geometry():
    """A pool species name is an identity: the evaluation computes one energy per name
    and the references directory holds one file per name, so a name that resolves to two
    geometries would score one geometry's reaction with the other's energy.

    Oracle: the species records of both Slim pools, compared by name.
    """
    _skip_without_slim_sources("slim05", "slim16")
    gs = _sets()
    for name in ("slim05", "slim16"):
        pool = gs.build_slim_pool_dict(name)
        by_name = {}
        for sd in pool["species"]:
            key = (sd["atom"], sd["charge"], sd["spin"])
            if sd["name"] in by_name:
                assert by_name[sd["name"]] == key, (name, sd["name"])
            by_name[sd["name"]] = key


def test_an_open_shell_species_of_the_pools_is_not_recorded_closed_shell():
    """Charge and 2S come from the checkout, whose ``coord`` files do not all carry the
    ``$eht`` block; a species defaulted to (0, 0) on an odd electron count is not a
    molecule PySCF can build.

    Oracle: every species record of the three pools, against the electron count of its
    own geometry.
    """
    _skip_without_slim_sources("slim05", "slim16")
    numbers = {"H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8,
               "F": 9, "Ne": 10, "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15,
               "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20, "Se": 34, "Br": 35,
               "Kr": 36, "Sb": 51, "Te": 52, "I": 53, "Pb": 82, "Bi": 83}
    gs = _sets()
    for name in ("slim05", "slim16", "diet150"):
        pool = (gs.build_diet150_pool_dict() if name == "diet150"
                else gs.build_slim_pool_dict(name))
        for sd in pool["species"]:
            symbols = [tok.split()[0] for tok in sd["atom"].split(";") if tok.split()]
            unlisted = sorted({s for s in symbols if s.capitalize() not in numbers})
            assert not unlisted, (name, sd["name"], unlisted)
            n_elec = sum(numbers[s.capitalize()] for s in symbols) - int(sd["charge"])
            assert (n_elec - int(sd["spin"])) % 2 == 0, (name, sd["name"], n_elec,
                                                         sd["spin"])


# ---------------------------------------------------------------------------
# 7. The diet150 pool
# ---------------------------------------------------------------------------

def _diet_yaml(name):
    yaml = pytest.importorskip("yaml")
    path = _REPO_ROOT / "data" / "dietgmtkn55-150" / name
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_the_diet150_pool_matches_the_element_list():
    """The 150 reactions, their species' stoichiometric counts, charges, unpaired
    electron counts and element sequences, their reference energies and their subset
    weights are the tracked element list's. The comparison is by the multiset of
    per-species records, since the element list names several systems by their published
    name rather than by the checkout's directory and orders two reactions' species
    differently.

    Oracle: ``data/dietgmtkn55-150/AllElements-150.yaml`` and
    ``SubsetGMTKN55_150.yaml``.
    """
    elements = _diet_yaml("AllElements-150.yaml")
    subsets = _diet_yaml("SubsetGMTKN55_150.yaml")["Systems"]
    gs = _sets()
    pool = gs.build_diet150_pool_dict()
    assert len(pool["reactions"]) == 150
    species = {sd["name"]: sd for sd in pool["species"]}
    seen = set()
    for rxn in pool["reactions"]:
        subset, index = rxn["subset"], int(rxn["index"])
        seen.add((subset, index))
        block = elements[subset][index]
        assert rxn["reaction_energy_ref"] == pytest.approx(
            float(block["Energy"]), abs=1e-4), (subset, index)
        assert rxn["weight"] == pytest.approx(float(block["Weight"]), abs=1e-6)
        assert rxn["weight"] == pytest.approx(float(subsets[subset][0]), abs=1e-6)
        coeff_by_system = dict(zip(rxn["systems"], rxn["coeffs"]))
        got = sorted(
            (int(coeff_by_system[system]),
             int(species[gs.species_name(subset, system)]["charge"]),
             int(species[gs.species_name(subset, system)]["spin"]),
             tuple(tok.split()[0] for tok
                   in species[gs.species_name(subset, system)]["atom"].split(";")
                   if tok.split()))
            for system in rxn["systems"])
        want = sorted((int(v["Count"]), int(v["Charge"]), int(v["UHF"]),
                       tuple(v["Elements"]))
                      for v in block["Species"].values())
        assert got == want, (subset, index)
    expected = {(s, int(i)) for s, (_w, ids) in subsets.items() for i in ids}
    assert seen == expected


def test_the_diet150_pool_keeps_every_subset_and_excludes_nothing():
    """The set is the 49 subsets of the tracked subset list with the weights that list
    carries; no reaction is dropped for overlapping a training set.

    Oracle: ``data/dietgmtkn55-150/SubsetGMTKN55_150.yaml``.
    """
    subsets = _diet_yaml("SubsetGMTKN55_150.yaml")["Systems"]
    gs = _sets()
    pool = gs.build_diet150_pool_dict()
    assert len({r["subset"] for r in pool["reactions"]}) == len(subsets) == 49
    assert sum(len(ids) for _w, ids in subsets.values()) == len(pool["reactions"])
    for rxn in pool["reactions"]:
        assert rxn["source_pool"] == "diet150"
        assert rxn["weight"] > 0.0


# ---------------------------------------------------------------------------
# 10. The overlap report
# ---------------------------------------------------------------------------

def test_the_overlap_report_names_matches_and_excludes_nothing():
    """The report names, per reaction and per species, the training species it matches
    exactly (same subset and system) and by formula (same Hill formula, charge and 2S).
    It is a report: the pool's reactions are unchanged by building it.

    Oracle: a synthetic pool and two synthetic training sets.
    """
    gs = _sets()
    pool = {
        "species": [
            {"name": "aaa_x", "subset": "AAA", "system": "x", "formula": "H2",
             "charge": 0, "spin": 0, "atom": "H 0 0 0; H 0 0 0.74",
             "atom_composition": [["H", 2]]},
            {"name": "bbb_y", "subset": "BBB", "system": "y", "formula": "FH",
             "charge": 0, "spin": 0, "atom": "H 0 0 0; F 0 0 0.92",
             "atom_composition": [["F", 1], ["H", 1]]},
        ],
        "reactions": [
            {"name": "aaa_001", "subset": "AAA", "index": 1,
             "systems": ["x", "y"], "reactants": ["aaa_x"], "products": ["bbb_y"],
             "coeffs": [-1.0, 1.0], "reaction_energy_ref": 1.0,
             "source_pool": "diet150", "source": "s",
             "species_spins": {"aaa_x": 0, "bbb_y": 0},
             "species_charges": {"aaa_x": 0, "bbb_y": 0}},
        ],
    }
    training = {
        "slim05": [{"name": "aaa_x", "subset": "AAA", "system": "x", "formula": "H2",
                    "charge": 0, "spin": 0}],
        "dfs": [{"name": "h2", "subset": "DFS", "system": "h2", "formula": "H2",
                 "charge": 0, "spin": 0}],
    }
    before = json.dumps(pool["reactions"], sort_keys=True)
    rows = gs.overlap_rows(pool, training)
    assert json.dumps(pool["reactions"], sort_keys=True) == before
    assert len(rows) == 1
    row = rows[0]
    assert row["name"] == "aaa_001" and row["subset"] == "AAA" and row["index"] == 1
    assert row["species"]["aaa_x"]["exact"] == ["slim05"]
    assert sorted(row["species"]["aaa_x"]["formula"]) == ["dfs", "slim05"]
    assert row["species"]["bbb_y"]["exact"] == []
    assert row["species"]["bbb_y"]["formula"] == []


# ---------------------------------------------------------------------------
# 11. The loaders and the pool union
# ---------------------------------------------------------------------------

def test_the_loaders_return_the_probe_schema():
    """The new loaders hand back the same pair the held-out evaluation already consumes
    -- a name-to-MoleculeSpec map and reaction dicts carrying every key the reaction MAE
    reads -- with the diet set's subset weight beside them.

    Oracle: the reaction-dict keys of the tracked BH76 cache.
    """
    from xcquinox.pipeline.config import MoleculeSpec
    required = ("name", "source_pool", "reactants", "products", "coeffs",
                "reaction_energy_ref", "species_spins", "species_charges", "source")
    gs = _sets()
    specs, reactions = gs.load_full_diet150(basis="def2-svp", grid_level=1,
                                            refs_dir=None)
    assert len(reactions) == 150
    for spec in specs.values():
        assert isinstance(spec, MoleculeSpec)
        assert spec.basis == "def2-svp" and spec.grid_level == 1
    for rxn in reactions:
        for key in required:
            assert key in rxn, (rxn.get("name"), key)
        assert "weight" in rxn
        assert set(rxn["reactants"]) | set(rxn["products"]) <= set(specs)
        assert len(rxn["coeffs"]) == len(rxn["reactants"]) + len(rxn["products"])


def test_an_unknown_pool_name_is_refused():
    """A misspelt pool would evaluate a smaller set than the configuration names, and
    the channel would be read as the full one."""
    from xcquinox.pipeline.full_benchmark_pools import (
        POOL_NAMES, load_held_out_pools)
    assert POOL_NAMES == ("bh76", "w411", "diet150", "slim05", "slim16")
    with pytest.raises(ValueError):
        load_held_out_pools(("bh76", "bh77"), basis="def2-svp", grid_level=1)
    with pytest.raises(ValueError):
        load_held_out_pools((), basis="def2-svp", grid_level=1)


# ---------------------------------------------------------------------------
# 12. The pretraining records
# ---------------------------------------------------------------------------

def test_slim_pretrain_records_are_the_drawn_molecules():
    """The pretraining records served for a Slim set are exactly the drawn molecules, in
    the drawn order, in the record form the pretraining-set resolver consumes.

    Oracle: the drawn molecules of the built Slim05 pool.
    """
    _skip_without_slim_sources("slim05")
    gs = _sets()
    records = gs.slim_pretrain_records("slim05")
    assert len(records) == 25
    pool = gs.build_slim_pool_dict("slim05")
    drawn = [pool["molecules"][i] for i in pool["pretrain_draw"]["indices"]]
    assert [(r["subset"], r["index"], r["system"]) for r in records] == [
        (m["subset"], m["index"], m["system"]) for m in drawn]
    for record in records:
        assert record["kind"] == "molecule"
        for key in ("name", "atom", "charge", "spin", "atom_composition"):
            assert key in record, (record.get("name"), key)
        assert isinstance(record["atom"], str) and record["atom"]


def test_the_drawn_molecules_record_the_paper_s_spin_rule_beside_the_checkout_s():
    """The paper's own runs set the spin by electron parity rather than from the
    checkout's metadata -- a single neutral atom from its table, every other system
    the parity of its electron count -- and the drawn molecules carry that value beside
    the checkout's, so wherever the two differ the difference is on record rather than
    silently resolved.

    Oracle: the rule of the published script (``handle_mols.xyz_to_mol_wspin`` and its
    atom table), recomputed here over every drawn molecule's elements and charge.
    """
    _skip_without_slim_sources("slim05")
    from ase.data import atomic_numbers
    study_atom_spins = {
        "Al": 1, "B": 1, "Li": 1, "Na": 1, "Si": 2, "Be": 0, "C": 2, "Cl": 1,
        "F": 1, "H": 1, "N": 3, "O": 2, "P": 3, "S": 2, "Ar": 0, "Br": 1, "Ne": 0,
        "Sb": 3, "Bi": 3, "Te": 2, "I": 1}
    gs = _sets()
    pool = gs.build_slim_pool_dict("slim05")
    molecules = pool["pretrain_molecules"]
    assert len(molecules) == 25
    for molecule in molecules:
        symbols = [tok.split()[0] for tok in molecule["atom"].split(";") if tok.split()]
        charge = int(molecule["charge"])
        if len(symbols) == 1 and charge == 0 and symbols[0] in study_atom_spins:
            want = study_atom_spins[symbols[0]]
        else:
            want = (sum(atomic_numbers[s] for s in symbols) - charge) % 2
        assert int(molecule["spin_parity_rule"]) == want, molecule["name"]
        assert "spin" in molecule and int(molecule["spin"]) >= 0
    # the two rules are recorded as two fields, whatever they agree on
    assert all("spin_parity_rule" in m and "spin" in m for m in molecules)


# ---------------------------------------------------------------------------
# 13. The coord reader's fallbacks
# ---------------------------------------------------------------------------

def test_the_coord_reader_falls_back_to_the_metadata_files_and_then_to_parity(
        tmp_path):
    """A species directory without the ``$eht`` block is read from its
    ``.CHRG`` and ``.UHF`` files, and without those from the parity of its
    electron count, so a one-electron system is never recorded as a closed
    shell; the block wins over the files where both exist.

    Oracle: three synthetic hydrogen directories built here.
    """
    from xcquinox.pipeline.full_benchmark_pools import _read_coord_meta
    xyz = "1\n\nH 0.0000000000 0.0000000000 0.0000000000\n"

    bare = tmp_path / "bare"
    bare.mkdir()
    (bare / "struc.xyz").write_text(xyz, encoding="utf-8")
    (bare / "coord").write_text("$coord\n0.0 0.0 0.0 h\n$end\n", encoding="utf-8")
    assert _read_coord_meta(bare) == (0, 1)

    filed = tmp_path / "filed"
    filed.mkdir()
    (filed / "struc.xyz").write_text(xyz, encoding="utf-8")
    (filed / "coord").write_text("$coord\n0.0 0.0 0.0 h\n$end\n", encoding="utf-8")
    (filed / ".CHRG").write_text("1\n", encoding="utf-8")
    (filed / ".UHF").write_text("0\n", encoding="utf-8")
    assert _read_coord_meta(filed) == (1, 0)

    blocked = tmp_path / "blocked"
    blocked.mkdir()
    (blocked / "struc.xyz").write_text(xyz, encoding="utf-8")
    (blocked / "coord").write_text(
        "$coord\n0.0 0.0 0.0 h\n$eht charge=0 unpaired=1\n$end\n", encoding="utf-8")
    (blocked / ".UHF").write_text("3\n", encoding="utf-8")
    assert _read_coord_meta(blocked) == (0, 1)

    charged_only = tmp_path / "charged"
    charged_only.mkdir()
    (charged_only / "struc.xyz").write_text(xyz, encoding="utf-8")
    (charged_only / ".CHRG").write_text("-1\n", encoding="utf-8")
    assert _read_coord_meta(charged_only) == (-1, 0)


# ---------------------------------------------------------------------------
# 11. The composition's subsets and the element list's geometries
# ---------------------------------------------------------------------------

def test_a_composition_naming_a_subset_twice_is_refused():
    """A subset's reactions sit at one position of the study's molecule list; a
    second line for the subset would place them at two."""
    gs = _sets()
    with pytest.raises(ValueError):
        gs.parse_composition("AHB21 1 2\nBH76 4\nAHB21 3\n")



# ---------------------------------------------------------------------------
# 12. One naming rule for every subset
# ---------------------------------------------------------------------------

def test_every_species_name_carries_its_subset_tag():
    """Every GMTKN55 system is named with its subset's tag, BH76, BH76RC and
    W4-11 included. No subset keeps a bare system name and no set renames a
    selection of its own systems to dodge another set's: a set is evaluated
    under the names its own definition gives it.

    Oracle: the naming rule against subsets whose system names collide in the
    checkout, and against the system names the two tracked pools share.
    """
    gs = _sets()
    assert gs.species_name("BH76", "h") == "bh76_h"
    assert gs.species_name("BH76", "CH4") == "bh76_CH4"
    assert gs.species_name("BH76RC", "ch3f") == "bh76rc_ch3f"
    assert gs.species_name("W4-11", "h2") == "w4_11_h2"
    assert gs.species_name("W4-11", "ch4") == "w4_11_ch4"
    assert gs.species_name("G21IP", "h") == "g21ip_h"
    assert gs.species_name("SIE4x4", "h") == "sie4x4_h"
    assert gs.species_name("MB16-43", "H2") == "mb16_43_H2"
    assert gs.species_name("W4-11", "h") != gs.species_name("SIE4x4", "h")
    assert gs.species_name("W4-11", "h") != gs.species_name("BH76", "h")
    assert gs.reaction_name("W4-11", 23) == "w4_11_023"
    assert gs.reaction_name("BH76RC", 5) == "bh76rc_005"
    assert not hasattr(gs, "BARE_SUBSETS")
    assert not hasattr(gs, "w411_qualified_names")


# ---------------------------------------------------------------------------
# 13. The diet set from its own list
# ---------------------------------------------------------------------------

def _diet_entries():
    """``{(subset, list name): entry}`` over the blocks the subset list
    selects, and the two loaded documents beside it."""
    elements = _diet_yaml("AllElements-150.yaml")
    subsets = _diet_yaml("SubsetGMTKN55_150.yaml")["Systems"]
    entries = {}
    for subset, (_weight, indices) in subsets.items():
        for index in indices:
            for name, entry in elements[subset][int(index)]["Species"].items():
                entries.setdefault((subset, name), entry)
    return elements, subsets, entries


def _positions_of(atom):
    """The coordinates of a PySCF atom string, flattened."""
    out = []
    for token in atom.split(";"):
        parts = token.split()
        if parts:
            out.extend(float(v) for v in parts[1:4])
    return out


def test_the_diet_pool_is_built_from_its_own_list_alone(tmp_path, monkeypatch):
    """The set is defined by its own two files, so it builds with no GMTKN55
    checkout in reach at all: the reactions, the species and their coordinates
    come from the element list and the subset list and from nothing else.

    Oracle: ``data/dietgmtkn55-150/AllElements-150.yaml`` and
    ``SubsetGMTKN55_150.yaml`` read here, against a build pointed at an empty
    directory.
    """
    empty = tmp_path / "no_checkout"
    empty.mkdir()
    monkeypatch.setenv("XCQUINOX_GMTKN55_DIR", str(empty))
    _elements, _subsets, entries = _diet_entries()
    gs = _sets()
    pool = gs.build_diet150_pool_dict()
    assert len(pool["reactions"]) == 150
    assert len(entries) == 335
    assert len(pool["species"]) == len(entries)
    species = {sd["name"]: sd for sd in pool["species"]}
    assert set(species) == {gs.species_name(subset, name)
                            for subset, name in entries}
    for (subset, name), entry in sorted(
            entries.items(), key=lambda kv: (kv[0][0], str(kv[0][1]))):
        sd = species[gs.species_name(subset, name)]
        want = [float(v) for position in entry["Positions"] for v in position]
        assert _positions_of(sd["atom"]) == pytest.approx(want, abs=1e-12), (
            subset, name)


def test_every_diet_species_carries_the_list_geometry_and_identity():
    """Over the built set, every species carries the element list's element
    sequence, coordinates, charge and unpaired-electron count, and every
    reaction carries the list's reference energy and its subset's weight. A
    species whose coordinates came from anywhere else is a species whose energy
    does not belong to the reference energy it is scored against.

    Oracle: ``data/dietgmtkn55-150/AllElements-150.yaml`` and
    ``SubsetGMTKN55_150.yaml``.
    """
    elements, subsets, _entries = _diet_entries()
    gs = _sets()
    pool = gs.build_diet150_pool_dict()
    species = {sd["name"]: sd for sd in pool["species"]}
    n_checked = 0
    for rxn in pool["reactions"]:
        subset, index = rxn["subset"], int(rxn["index"])
        block = elements[subset][index]
        assert rxn["reaction_energy_ref"] == pytest.approx(
            float(block["Energy"]), abs=1e-9), (subset, index)
        assert rxn["weight"] == pytest.approx(float(block["Weight"]), abs=1e-9)
        assert rxn["weight"] == pytest.approx(float(subsets[subset][0]),
                                              abs=1e-9)
        for name, entry in block["Species"].items():
            key = gs.species_name(subset, name)
            assert key in species, (subset, name)
            sd = species[key]
            symbols = [token.split()[0] for token in sd["atom"].split(";")
                       if token.split()]
            assert symbols == [str(e) for e in entry["Elements"]]
            want = [float(v) for position in entry["Positions"]
                    for v in position]
            assert _positions_of(sd["atom"]) == pytest.approx(want, abs=1e-12)
            assert int(sd["charge"]) == int(entry["Charge"])
            assert int(sd["spin"]) == int(entry["UHF"])
            assert "geometry_source" not in sd
            assert "element_list_deviation" not in sd
            n_checked += 1
    assert n_checked > 0


# ---------------------------------------------------------------------------
# 14. The tracked caches
# ---------------------------------------------------------------------------

def test_the_tracked_caches_regenerate_byte_for_byte():
    """Every tracked cache and report the rebuild tool writes is what the
    sources produce now, byte for byte, and every species of a set built from
    the collection is named with its subset's tag. A cache that no longer
    regenerates is a cache whose provenance has been lost.

    Oracle: the tracked JSON files under ``xcquinox/pipeline/data/``.
    """
    # Both Slim pools are regenerated below and the overlap reports need every
    # subset either set names, BH76 and W4-11 among them.
    _skip_without_slim_sources("slim05", "slim16")
    from xcquinox.pipeline.full_benchmark_pools import (build_bh76_pool_dict,
                                                        build_w411_pool_dict)
    gs = _sets()
    builders = [
        (build_bh76_pool_dict, BH76_JSON_PATH),
        (build_w411_pool_dict, W411_JSON_PATH),
        (lambda: gs.build_slim_pool_dict("slim05"), gs.SLIM_JSON_PATHS["slim05"]),
        (lambda: gs.build_slim_pool_dict("slim16"), gs.SLIM_JSON_PATHS["slim16"]),
        (gs.build_diet150_pool_dict, gs.DIET150_JSON_PATH),
    ]
    reports = gs.build_overlap_reports()
    for key, path in gs.OVERLAP_JSON_PATHS.items():
        builders.append((lambda k=key: reports[k], path))
    for builder, json_path in builders:
        regenerated = (json.dumps(builder(), indent=2, sort_keys=False,
                                  ensure_ascii=False) + "\n").encode("utf-8")
        tracked = Path(json_path).read_bytes()
        assert regenerated == tracked, (
            f"{Path(json_path).name}: {len(regenerated)} bytes regenerated "
            f"against {len(tracked)} tracked")
    for name in ("slim05", "slim16", "diet150"):
        path = (gs.DIET150_JSON_PATH if name == "diet150"
                else gs.SLIM_JSON_PATHS[name])
        cache = json.loads(Path(path).read_text(encoding="utf-8"))
        for sd in cache["species"]:
            assert sd["name"] == gs.species_name(sd["subset"], sd["system"]), (
                name, sd["name"])
            assert sd["name"].startswith(gs.subset_tag(sd["subset"]) + "_"), (
                name, sd["name"])


# ---------------------------------------------------------------------------
# 12. What a checkout-free clone can run
# ---------------------------------------------------------------------------

def test_the_slim_subset_guard_names_the_absent_subsets(tmp_path, monkeypatch):
    """The guard over a Slim set's own subsets reads that set's composition file
    and reports which of the subsets it names have no reaction file under the
    checkout, under either of its two layouts.

    Oracle: an empty checkout directory, against the composition file on disk.
    """
    absent = _missing_compositions()
    if absent:
        pytest.skip(f"the composition files are not on disk: {absent}")
    monkeypatch.setenv("XCQUINOX_GMTKN55_DIR", str(tmp_path))
    missing = _missing_slim_subsets("slim05")
    assert missing, "an empty checkout has none of the set's subsets"
    assert "BH76" in missing and "W4-11" in missing
    assert "#" not in "".join(missing)
    assert len(missing) == len(set(missing))

    flat = tmp_path / "BH76"
    flat.mkdir()
    (flat / ".res").write_text("$tmer h/$f x -1 $w 1.0\n", encoding="utf-8")
    assert "BH76" not in _missing_slim_subsets("slim05")
    nested = tmp_path / "gmtkn55" / "W4-11"
    nested.mkdir(parents=True)
    (nested / ".res").write_text("$tmer h/$f x -1 $w 1.0\n", encoding="utf-8")
    assert "W4-11" not in _missing_slim_subsets("slim05")

    monkeypatch.delenv("XCQUINOX_GMTKN55_DIR")
    # Against the checkout as it stands, whatever state that is: the subsets
    # reported absent are exactly those of the set's own composition file that
    # are absent, read here line by line rather than through the guard.
    independent = []
    for raw in (_REPO_ROOT / "data" / "slim"
                / "Slim100M_05_composition.txt").read_text(
                    encoding="utf-8").splitlines():
        head = raw.split()[:1]
        if not head or head[0].startswith("#"):
            continue
        subset = "BH76" if head[0] == "BH76RC" else head[0]
        if subset not in independent:
            independent.append(subset)
    assert _named_slim_subsets("slim05") == tuple(independent)
    assert _missing_slim_subsets("slim05") == _missing_subsets(independent)


def test_the_slim_subset_guard_refuses_a_name_that_is_not_a_slim_set():
    """A name with no composition file is refused rather than reported
    complete: an empty list reads as "every subset is present", which would
    turn a misspelled set name into a silent pass.

    Oracle: the three Slim names of the composition table, against two others.
    """
    assert set(_SLIM_COMPOSITION_FILES) == {"slim05", "slim16", "slim20"}
    for name in ("diet150", "slim5", "", "slim"):
        with pytest.raises(ValueError):
            _missing_slim_subsets(name)


def test_the_slim_subset_guard_reads_the_composition_grammar(tmp_path,
                                                              monkeypatch):
    """The guard takes a composition file's subset names off the head of each
    line: a blank line and a line opened by a hash are not subsets, and BH76RC
    is asked for under BH76, the directory whose reaction file carries it. It
    is deliberately more permissive than the set's own parser, which refuses a
    file naming a subset twice, so that a composition the builders reject is
    still skipped rather than failed; a repeated subset is reported once.

    Oracle: a synthetic composition file, since no tracked one carries a
    comment or a repeat, with the parser run on the same text.
    """
    monkeypatch.setenv("XCQUINOX_GMTKN55_DIR", str(tmp_path / "clone"))
    text = ("# a note about the set\n"
            "\n"
            "AHB21 1 2\n"
            "BH76RC 2 30\n"
            "AHB21 4\n"
            "   \n"
            "#BH76 1\n")
    base = tmp_path / "slim"
    base.mkdir()
    (base / "Slim100M_05_composition.txt").write_text(text, encoding="utf-8")
    assert _named_slim_subsets("slim05", base=base) == ("AHB21", "BH76")
    assert _missing_slim_subsets("slim05", base=base) == ["AHB21", "BH76"]
    assert _missing_slim_subsets("slim16", base=base) == []
    with pytest.raises(ValueError):
        _sets().parse_composition(text)


def test_every_slim_builder_skips_without_the_checkout(tmp_path, monkeypatch):
    """Every test that builds a Slim pool skips when the GMTKN55 checkout is
    absent, rather than failing. The checkout is not in the repository
    (``.gitignore`` keeps all of it but the provenance file) while the
    composition files are, so a clone without it -- which is what CI runs --
    would otherwise fail these on a missing reaction file.

    Oracle: the test functions themselves, called against an empty checkout.
    """
    from _pytest.outcomes import Skipped
    absent = _missing_compositions()
    if absent:
        pytest.skip(f"the composition files are not on disk: {absent}")
    cases = _slim_builder_tests()
    runs = sum(len(argsets) for _fn, argsets in cases)
    assert runs >= 8, f"only {runs} builder runs over {len(cases)} tests"
    monkeypatch.setenv("XCQUINOX_GMTKN55_DIR", str(tmp_path))
    for fn, argsets in cases:
        for args in argsets:
            with pytest.raises(Skipped):
                fn(*args)


def test_every_slim_builder_guards_the_sets_it_builds():
    """A builder's skip names every Slim set the builder goes on to read. A
    guard narrowed to one of the two sets a test builds is invisible against a
    checkout that is wholly absent and fails against one that is merely
    incomplete, so the sets are compared rather than the skipping.

    Oracle: each test's own syntax tree -- the set names it passes to the
    guard, against the set names it passes to anything else.
    """
    cases = _slim_builder_tests()
    runs = sum(len(argsets) for _fn, argsets in cases)
    assert runs >= 8, f"only {runs} builder runs over {len(cases)} tests"
    for fn, argsets in cases:
        body = ast.parse(inspect.getsource(fn)).body[0].body
        nodes = [n for stmt in body for n in ast.walk(stmt)]
        guarded, named, parametrized = set(), set(), False
        for call in [n for n in nodes if isinstance(n, ast.Call)]:
            if getattr(call.func, "id", None) != _SLIM_GUARD:
                continue
            for arg in call.args:
                if isinstance(arg, ast.Constant):
                    guarded.add(arg.value)
                else:
                    parametrized = True
        for node in nodes:
            if isinstance(node, ast.Constant) and node.value in _SLIM_SET_NAMES:
                named.add(node.value)
        built = named - guarded
        assert guarded or parametrized, f"{fn.__name__} guards on nothing"
        if parametrized:
            # The guard is passed the case parameter, so the set it names is
            # the case: every case value must be a Slim set.
            values = {v for argset in argsets for v in argset}
            assert values and values <= _SLIM_SET_NAMES, (fn.__name__, values)
            guarded |= values
        assert built <= guarded, (
            f"{fn.__name__} builds {sorted(built)} but guards on "
            f"{sorted(guarded)}")
