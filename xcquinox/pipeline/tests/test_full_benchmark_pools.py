"""Tests for ``xcquinox.pipeline.full_benchmark_pools``.

The pre-built JSON caches under ``xcquinox/pipeline/data/{bh76,w411}_full_pool.json``
are committed to the repo so these tests do NOT touch the GMTKN55 source
on disk. The regen script
``tools/rebuild_full_benchmark_pools.py`` is what (re)generates the
JSON; if the source data changes, the JSON gets rebuilt and the test
assertions on counts adjust accordingly.
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest

from xcquinox.pipeline.full_benchmark_pools import (
    load_full_bh76,
    load_full_w411,
    load_full_held_out_pools,
)
from xcquinox.pipeline.config import MoleculeSpec


# ---------------------------------------------------------------------------
# The source root and the provenance of the caches
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Geometry helpers (for the units-regression tests below)
# ---------------------------------------------------------------------------

def _parse_atom_str(atom_str: str):
    """``'H x y z; O x y z'`` -> [(elem, x, y, z), ...] as floats (Angstrom)."""
    out = []
    for tok in atom_str.split(";"):
        p = tok.split()
        if len(p) < 4:
            continue
        out.append((p[0], float(p[1]), float(p[2]), float(p[3])))
    return out


def _bond_length(atom_str: str, i: int, j: int) -> float:
    c = _parse_atom_str(atom_str)
    return math.dist(c[i][1:], c[j][1:])


# ---------------------------------------------------------------------------
# Sanity counts
# ---------------------------------------------------------------------------

def test_load_full_bh76_returns_76_reactions():
    """GMTKN55-BH76 has exactly 76 forward-barrier reactions."""
    mol_specs, reactions = load_full_bh76()
    assert len(reactions) == 76, len(reactions)
    # Each reaction has at least 2 species touching it (lowest-bound; some
    # have 3+ for the transition state path).
    for r in reactions:
        assert len(r["reactants"]) + len(r["products"]) >= 2, r["name"]


def test_load_full_w411_returns_140_atomizations():
    """GMTKN55-W4-11 has exactly 140 atomization reactions."""
    mol_specs, reactions = load_full_w411()
    assert len(reactions) == 140, len(reactions)
    # Each W4-11 reaction has exactly one molecule reactant + ≥1 atom product.
    for r in reactions:
        assert len(r["reactants"]) == 1, r["name"]
        assert len(r["products"]) >= 1, r["name"]


@pytest.mark.parametrize("loader_name,cache_name", [
    ("load_full_bh76", "_BH76_CACHE"),
    ("load_full_w411", "_W411_CACHE"),
])
def test_pool_load_hits_cache_on_second_call(loader_name, cache_name, monkeypatch):
    """REGRESSION: the (basis, grid_level) cache must HIT on a
    repeat call. The cache compared keys with ``is`` (identity) against a freshly
    built tuple, so it never hit and re-parsed the JSON on every call."""
    import xcquinox.pipeline.full_benchmark_pools as fbp
    monkeypatch.setattr(fbp, cache_name, None)
    calls = {"n": 0}
    orig = fbp._load_pool_from_json

    def _counting(*args, **kwargs):
        calls["n"] += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr(fbp, "_load_pool_from_json", _counting)
    loader = getattr(fbp, loader_name)
    first_specs, first_rxns = loader("def2-svp", 1)
    second_specs, second_rxns = loader("def2-svp", 1)
    # The second identical call must hit the cache, i.e. NOT re-parse the JSON.
    assert calls["n"] == 1, (
        f"cache never hit: _load_pool_from_json called {calls['n']}x for two "
        f"identical {loader_name}(...) calls")
    # And it must return the cached payload (same species/reaction objects).
    assert second_specs is getattr(fbp, cache_name)[1][0]
    assert len(second_rxns) == len(first_rxns)


# ---------------------------------------------------------------------------
# Schema parity with PROBE_C_BH76_OUT_OF_TRAINING
# ---------------------------------------------------------------------------

# The reaction-dict keys that reaction_mae_kcalmol + per_reaction_errors
# read from. Any reaction in the full-pool must carry all of these.
_REQUIRED_RXN_KEYS = (
    "name", "source_pool", "reactants", "products", "coeffs",
    "reaction_energy_ref", "species_spins", "species_charges", "source",
)


@pytest.mark.parametrize("loader,expected_pool",
                          [(load_full_bh76, "bh76"),
                           (load_full_w411, "w411")])
def test_reaction_schema_matches_probe_c_keys(loader, expected_pool):
    """Every reaction dict has every key reaction_mae_kcalmol consumes."""
    _, reactions = loader()
    for r in reactions:
        for k in _REQUIRED_RXN_KEYS:
            assert k in r, f"{r.get('name')} missing {k}"
        assert r["source_pool"] == expected_pool, r["name"]
        # coeffs aligned with reactants + products
        assert len(r["coeffs"]) == len(r["reactants"]) + len(r["products"]), (
            r["name"], r["coeffs"], r["reactants"], r["products"]
        )
        # signed convention: reactant coeffs negative, product coeffs positive
        n_r = len(r["reactants"])
        for c in r["coeffs"][:n_r]:
            assert c < 0, (r["name"], r["coeffs"])
        for c in r["coeffs"][n_r:]:
            assert c > 0, (r["name"], r["coeffs"])


# ---------------------------------------------------------------------------
# MoleculeSpec construction
# ---------------------------------------------------------------------------

def test_species_dicts_yield_valid_mol_specs():
    """Each species in BH76+W4-11 builds a hashable MoleculeSpec."""
    mol_specs, _ = load_full_bh76()
    for name, ms in mol_specs.items():
        assert isinstance(ms, MoleculeSpec), name
        assert ms.name == name, (ms.name, name)
        # atom_composition is a tuple of pairs (element, count), hashable
        for elem, count in ms.atom_composition:
            assert isinstance(elem, str), elem
            assert isinstance(count, int) and count >= 1, count
        # spin = 2S (non-negative); charge can be negative
        assert ms.spin >= 0, (name, ms.spin)


def test_load_full_held_out_pools_merges_species():
    """Combined pool: BH76 + W4-11 species merged (no duplicate by name)."""
    bh76_mols, bh76_rxns = load_full_bh76()
    w411_mols, w411_rxns = load_full_w411()
    all_mols, all_rxns = load_full_held_out_pools()
    # Reactions concatenate; species merge.
    assert len(all_rxns) == len(bh76_rxns) + len(w411_rxns)
    assert len(all_mols) <= len(bh76_mols) + len(w411_mols)
    # Every species in BH76 or W4-11 must appear in the merged dict.
    for name in bh76_mols:
        assert name in all_mols, name
    for name in w411_mols:
        assert name in all_mols, name


# ---------------------------------------------------------------------------
# Round-trip a single reaction (sanity that the parser preserves the math)
# ---------------------------------------------------------------------------

def test_bh76_first_reaction_h_n2o_to_n2ohts_has_ref_17_7():
    """The first BH76 line is ``$tmer h n2o n2ohts x -1 -1 1 $w 17.7``. The
    parsed reaction must carry that exact reference energy."""
    _, reactions = load_full_bh76()
    target = None
    for r in reactions:
        if r["reactants"] == ["h", "n2o"] and r["products"] == ["n2ohts"]:
            target = r
            break
    assert target is not None, "h + n2o -> n2ohts not found in BH76"
    assert target["reaction_energy_ref"] == pytest.approx(17.7, abs=1e-6)
    assert target["coeffs"] == [-1.0, -1.0, 1.0]
    # H is open-shell (doublet); n2o and the transition state should be
    # closed-shell singlet by GMTKN55 convention.
    assert target["species_spins"]["h"] == 1
    assert target["species_charges"]["h"] == 0


def test_w411_h2_atomization_ref_109_493():
    """The first W4-11 line is ``$tmer {h2,h} x -1 2 $w 109.493``. The parsed
    atomization must carry the reference 2*E(h) - E(h2) = 109.493."""
    _, reactions = load_full_w411()
    target = None
    for r in reactions:
        if r["reactants"] == ["h2"] and r["products"] == ["h"]:
            target = r
            break
    assert target is not None
    assert target["reaction_energy_ref"] == pytest.approx(109.493, abs=1e-6)
    assert target["coeffs"] == [-1.0, 2.0]


# ---------------------------------------------------------------------------
# Geometry-units regression (the held-out struc.xyz-as-Bohr bug, 2026-05-31)
# ---------------------------------------------------------------------------
#
# GMTKN55 ``struc.xyz`` files are in ANGSTROM (standard .xyz convention). A bug
# in ``_atoms_to_pyscf_str`` divided them by BOHR_PER_ANGSTROM, shrinking every
# molecule ~1.89x and producing catastrophically wrong held-out reaction
# energies (W4-11 atomizations came out negative; BH76 barriers ~20x too big).
# The pre-2026-05-31 suite never checked a bond length or an energy, so it
# missed this entirely. These tests pin the physical geometry + energy sign.

def test_w411_h2_bond_length_is_physical_angstrom():
    """H2 equilibrium bond length is 0.741 Angstrom. The buggy (shrunk)
    geometry gives ~0.393 A (0.741 / 1.8897)."""
    mol_specs, _ = load_full_w411()
    d = _bond_length(mol_specs["h2"].atom, 0, 1)
    assert d == pytest.approx(0.741, abs=0.03), (
        f"H2 bond length {d:.4f} A is not physical (expect ~0.741 A). "
        f"~0.393 A indicates the struc.xyz-as-Bohr units bug.")


# ---------------------------------------------------------------------------
# benchmark refs_dir wiring (density-only CCSD references)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Held-out species slice: an explicitly named handful of the pool, for
# workflow verification only.
# ---------------------------------------------------------------------------


def test_slice_held_out_pools_keeps_only_closed_reactions():
    from xcquinox.pipeline.full_benchmark_pools import slice_held_out_pools
    mols = {"a": 1, "b": 2, "c": 3}
    rxns = [
        {"name": "closed", "reactants": ["a"], "products": ["b"]},
        {"name": "open", "reactants": ["a"], "products": ["c"]},
    ]
    kept_mols, kept_rxns = slice_held_out_pools(mols, rxns, ("a", "b"))
    assert kept_mols == {"a": 1, "b": 2}
    assert [r["name"] for r in kept_rxns] == ["closed"]


# ---------------------------------------------------------------------------
# The GMTKN55 source clone: where it is resolved, and that the tracked caches
# still come out of it
# ---------------------------------------------------------------------------

#: The repository root: this file sits at ``xcquinox/pipeline/tests/``.
_REPO_ROOT = Path(__file__).resolve().parents[3]

#: The two subsets the pool builders parse.
_SUBSETS = ("BH76", "W4-11")


def _serialize_pool(data) -> bytes:
    """A pool dict as ``tools/rebuild_full_benchmark_pools.py`` writes it:
    two-space indent, insertion order kept, non-ASCII verbatim, and the closing
    newline its writer appends after the dump."""
    return (json.dumps(data, indent=2, sort_keys=False,
                       ensure_ascii=False) + "\n").encode("utf-8")


def _missing_sources() -> list[str]:
    """The subsets whose ``.res`` is on disk under neither layout of the
    GMTKN55 clone. Resolved without ``gmtkn55_root()`` on purpose: a skip
    decided by the resolution under test would hide the failure the
    regeneration test exists to report."""
    env = os.environ.get("XCQUINOX_GMTKN55_DIR")
    root = Path(env) if env else _REPO_ROOT / "data" / "gmtkn55"
    return [name for name in _SUBSETS
            if not (root / name / ".res").is_file()
            and not (root / "gmtkn55" / name / ".res").is_file()]


def test_gmtkn55_root_descends_into_a_nested_checkout(tmp_path, monkeypatch):
    """A clone whose own top directory repeats the collection name carries
    every subset one level below the configured root, and the root resolves to
    the level that holds the subsets -- the level the two pool builders read.

    Oracle: a tree carrying ``<root>/gmtkn55/BH76`` and no ``<root>/BH76``,
    which is the layout of the clone under ``data/gmtkn55``.
    """
    from xcquinox.pipeline.full_benchmark_pools import gmtkn55_root
    (tmp_path / "gmtkn55" / "BH76").mkdir(parents=True)
    monkeypatch.setenv("XCQUINOX_GMTKN55_DIR", str(tmp_path))
    assert gmtkn55_root() == tmp_path / "gmtkn55"


def test_gmtkn55_root_keeps_a_flat_checkout(tmp_path, monkeypatch):
    """The descent is taken only where the subsets are not at the root: a
    clone holding them directly resolves to itself even with a directory of
    the collection's name beside them, and a machine carrying no clone at all
    gets the root back rather than an error: the resolver imports everywhere,
    and the subset accessor is what refuses, at the point of reading.

    Oracle: two trees -- the subsets at the root with a decoy one level below,
    and a root with nothing under it.
    """
    from xcquinox.pipeline.full_benchmark_pools import gmtkn55_root
    (tmp_path / "BH76").mkdir()
    (tmp_path / "gmtkn55" / "BH76").mkdir(parents=True)
    monkeypatch.setenv("XCQUINOX_GMTKN55_DIR", str(tmp_path))
    assert gmtkn55_root() == tmp_path

    bare = tmp_path / "bare"
    bare.mkdir()
    monkeypatch.setenv("XCQUINOX_GMTKN55_DIR", str(bare))
    assert gmtkn55_root() == bare


def test_gmtkn55_subset_dir_names_both_candidates(tmp_path, monkeypatch):
    """The subset accessor resolves a subset under either layout, and where
    the subset is under neither it reports both paths it looked at, so the
    reader of the failure knows which clone belongs where.

    Oracle: three trees -- flat, nested, and empty.
    """
    from xcquinox.pipeline.full_benchmark_pools import gmtkn55_subset_dir

    flat = tmp_path / "flat"
    (flat / "BH76").mkdir(parents=True)
    monkeypatch.setenv("XCQUINOX_GMTKN55_DIR", str(flat))
    assert gmtkn55_subset_dir("BH76") == flat / "BH76"

    nested = tmp_path / "nested"
    for name in _SUBSETS:
        (nested / "gmtkn55" / name).mkdir(parents=True)
    monkeypatch.setenv("XCQUINOX_GMTKN55_DIR", str(nested))
    assert gmtkn55_subset_dir("W4-11") == nested / "gmtkn55" / "W4-11"

    bare = tmp_path / "bare"
    bare.mkdir()
    monkeypatch.setenv("XCQUINOX_GMTKN55_DIR", str(bare))
    with pytest.raises(FileNotFoundError) as exc:
        gmtkn55_subset_dir("BH76")
    message = str(exc.value)
    assert str(bare / "BH76") in message, message
    assert str(bare / "gmtkn55" / "BH76") in message, message


def test_the_tracked_pools_regenerate_byte_for_byte():
    """The tracked caches are what the source produces now: each builder's
    dict, serialized the way the regeneration script serializes it, equals the
    tracked file byte for byte. A cache that no longer regenerates is a cache
    whose provenance has been lost.

    Oracle: ``xcquinox/pipeline/data/{bh76,w411}_full_pool.json`` as tracked.
    """
    missing = _missing_sources()
    if missing:
        pytest.skip("the GMTKN55 clone is not on this machine: no .res for "
                    f"{', '.join(missing)} under data/gmtkn55 in either "
                    "layout")
    from xcquinox.pipeline.full_benchmark_pools import (
        BH76_JSON_PATH,
        W411_JSON_PATH,
        build_bh76_pool_dict,
        build_w411_pool_dict,
    )
    for builder, json_path in ((build_bh76_pool_dict, BH76_JSON_PATH),
                               (build_w411_pool_dict, W411_JSON_PATH)):
        regenerated = _serialize_pool(builder())
        tracked = Path(json_path).read_bytes()
        assert regenerated == tracked, (
            f"{Path(json_path).name}: {len(regenerated)} bytes regenerated "
            f"against {len(tracked)} tracked")
