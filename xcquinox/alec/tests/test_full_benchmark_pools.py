"""Tests for ``xcquinox.alec.full_benchmark_pools``.

The pre-built JSON caches under ``xcquinox/alec/data/{bh76,w411}_full_pool.json``
are committed to the repo so these tests do NOT touch the GMTKN55 source
on disk. The regen script
``scripts/rebuild_full_benchmark_pools.py`` is what (re)generates the
JSON; if the source data changes, the JSON gets rebuilt and the test
assertions on counts adjust accordingly.
"""
from __future__ import annotations

import math

import pytest

from xcquinox.alec.full_benchmark_pools import (
    load_full_bh76,
    load_full_w411,
    load_full_held_out_pools,
)
from xcquinox.alec.config import MoleculeSpec


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
    import xcquinox.alec.full_benchmark_pools as fbp
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


def test_full_bh76_species_count_at_least_50():
    """BH76 covers at least 50 distinct species (including transition states)."""
    mol_specs, _ = load_full_bh76()
    assert len(mol_specs) >= 50, len(mol_specs)


def test_full_w411_species_count_at_least_100():
    """W4-11 covers ~99 molecules + ~10 atoms; at least 100 species."""
    mol_specs, _ = load_full_w411()
    assert len(mol_specs) >= 100, len(mol_specs)


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


def test_h_atom_appears_in_both_pools():
    """The H atom is in both BH76 and W4-11. Confirms the merge logic in
    ``load_full_held_out_pools`` deduplicates correctly."""
    bh76_mols, _ = load_full_bh76()
    w411_mols, _ = load_full_w411()
    assert "h" in bh76_mols, list(bh76_mols)[:10]
    assert "h" in w411_mols, list(w411_mols)[:10]
    # H is a doublet (1 unpaired electron, 2S = 1)
    assert bh76_mols["h"].spin == 1
    # Same atom regardless of pool
    assert bh76_mols["h"].atom_composition == w411_mols["h"].atom_composition


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


def test_w411_n2o_bond_lengths_are_physical_angstrom():
    """N2O is linear with N-N ~ 1.128 A and N-O ~ 1.184 A. The buggy geometry
    shrinks these to ~0.60 / ~0.63 A."""
    mol_specs, _ = load_full_w411()
    coords = _parse_atom_str(mol_specs["n2o"].atom)
    # Distances between consecutive atoms along the molecular axis; the two
    # nearest-neighbour bonds should be ~1.13 and ~1.18 A (order-independent).
    import itertools
    dists = sorted(math.dist(a[1:], b[1:])
                   for a, b in itertools.combinations(coords, 2))
    assert dists[0] == pytest.approx(1.13, abs=0.05), dists
    assert dists[1] == pytest.approx(1.18, abs=0.05), dists


@pytest.mark.slow
def test_w411_h2_atomization_pbe_sign_and_magnitude():
    """End-to-end: PBE/def2-svp atomization 2*E(H) - E(H2) must be ~+105
    kcal/mol (positive). The shrunk geometry gives ~-46 kcal/mol (wrong sign)."""
    from pyscf import gto, dft
    KCAL = 627.5094740631
    mol_specs, _ = load_full_w411()

    def _e(name, spin):
        ms = mol_specs[name]
        mol = gto.M(atom=ms.atom, basis="def2-svp", charge=ms.charge,
                    spin=spin, unit="angstrom", verbose=0)
        mf = (dft.UKS(mol) if spin else dft.RKS(mol))
        mf.xc = "pbe"
        return float(mf.kernel())

    e_h2 = _e("h2", 0)
    e_h = _e("h", 1)
    atomization = (2.0 * e_h - e_h2) * KCAL
    assert atomization == pytest.approx(105.0, abs=15.0), (
        f"PBE H2 atomization {atomization:.1f} kcal/mol (expect ~+105; "
        f"~-46 indicates the geometry units bug).")


# ---------------------------------------------------------------------------
# benchmark refs_dir wiring (density-only CCSD references)
# ---------------------------------------------------------------------------

def test_refs_dir_wires_external_data_path(tmp_path, monkeypatch):
    import numpy as np
    from xcquinox.alec import full_benchmark_pools as fbp

    monkeypatch.delenv("XCQUINOX_BENCH_REFS_DIR", raising=False)
    np.savez_compressed(tmp_path / "h.npz", rho_ref_grid=np.ones(3))
    mols, _ = fbp.load_full_bh76(basis="def2-svp", grid_level=2,
                                 refs_dir=tmp_path)
    assert mols["h"].external_data_path == str(tmp_path / "h.npz")
    # species without a reference file stay None
    others = [m for n, m in mols.items() if n != "h"]
    assert all(m.external_data_path is None for m in others)
    # refs_dir is part of the cache key: a refs-free call must NOT return the
    # refs-wired cached pool
    mols_plain, _ = fbp.load_full_bh76(basis="def2-svp", grid_level=2)
    assert mols_plain["h"].external_data_path is None


def test_refs_dir_env_var_fallback(tmp_path, monkeypatch):
    import numpy as np
    from xcquinox.alec import full_benchmark_pools as fbp

    np.savez_compressed(tmp_path / "h.npz", rho_ref_grid=np.ones(3))
    monkeypatch.setenv("XCQUINOX_BENCH_REFS_DIR", str(tmp_path))
    mols, _ = fbp.load_full_held_out_pools(basis="def2-svp", grid_level=2)
    assert mols["h"].external_data_path == str(tmp_path / "h.npz")
    # explicit kwarg wins over the env var
    other = tmp_path / "other"
    other.mkdir()
    mols2, _ = fbp.load_full_bh76(basis="def2-svp", grid_level=2,
                                  refs_dir=other)
    assert mols2["h"].external_data_path is None   # no h.npz under other/


# ---------------------------------------------------------------------------
# Held-out species slice: an explicitly named handful of the pool, for
# workflow verification only (SPEC_pretrain_fidelity_program.md 3.4).
# ---------------------------------------------------------------------------

def test_resolve_species_slice_is_none_without_the_variable():
    from xcquinox.alec.full_benchmark_pools import resolve_species_slice
    assert resolve_species_slice({}) is None


def test_resolve_species_slice_is_none_for_an_empty_variable():
    from xcquinox.alec.full_benchmark_pools import (
        HELDOUT_SPECIES_SLICE_ENV, resolve_species_slice)
    assert resolve_species_slice({HELDOUT_SPECIES_SLICE_ENV: ""}) is None
    assert resolve_species_slice({HELDOUT_SPECIES_SLICE_ENV: "   "}) is None


def test_resolve_species_slice_parses_and_strips():
    from xcquinox.alec.full_benchmark_pools import (
        HELDOUT_SPECIES_SLICE_ENV, resolve_species_slice)
    got = resolve_species_slice(
        {HELDOUT_SPECIES_SLICE_ENV: " h , h2,o ,oh,n2o,n2ohts "})
    assert got == ("h", "h2", "o", "oh", "n2o", "n2ohts")


def test_resolve_species_slice_refuses_a_repeated_name():
    from xcquinox.alec.full_benchmark_pools import (
        HELDOUT_SPECIES_SLICE_ENV, resolve_species_slice)
    with pytest.raises(ValueError, match="repeats"):
        resolve_species_slice({HELDOUT_SPECIES_SLICE_ENV: "h,h2,h"})


def test_resolve_species_slice_reads_the_process_environment(monkeypatch):
    from xcquinox.alec.full_benchmark_pools import (
        HELDOUT_SPECIES_SLICE_ENV, resolve_species_slice)
    monkeypatch.setenv(HELDOUT_SPECIES_SLICE_ENV, "h,h2")
    assert resolve_species_slice() == ("h", "h2")


def test_slice_held_out_pools_keeps_only_closed_reactions():
    from xcquinox.alec.full_benchmark_pools import slice_held_out_pools
    mols = {"a": 1, "b": 2, "c": 3}
    rxns = [
        {"name": "closed", "reactants": ["a"], "products": ["b"]},
        {"name": "open", "reactants": ["a"], "products": ["c"]},
    ]
    kept_mols, kept_rxns = slice_held_out_pools(mols, rxns, ("a", "b"))
    assert kept_mols == {"a": 1, "b": 2}
    assert [r["name"] for r in kept_rxns] == ["closed"]


def test_slice_held_out_pools_refuses_a_species_absent_from_the_pool():
    from xcquinox.alec.full_benchmark_pools import slice_held_out_pools
    with pytest.raises(ValueError, match="nosuchspecies"):
        slice_held_out_pools({"a": 1}, [], ("a", "nosuchspecies"))


def test_slice_held_out_pools_preserves_the_requested_order():
    from xcquinox.alec.full_benchmark_pools import slice_held_out_pools
    kept, _ = slice_held_out_pools({"a": 1, "b": 2, "c": 3}, [], ("c", "a"))
    assert list(kept) == ["c", "a"]


def test_the_matrix_six_species_slice_spans_both_pools():
    """The workflow matrix's slice: six species of the real pool that close
    three reactions, one BH76 barrier and two W4-11 atomizations, over both
    spin types (RKS h2/n2o, UKS h/o/oh/n2ohts). The W4-11 leg is atomization
    energies, so every one of its reactions carries single-atom legs: a slice
    holding no atom closes only BH76 barriers (41 of the 216 reactions need no
    atom, all of them BH76) and would not span both pools."""
    from xcquinox.alec.full_benchmark_pools import (
        load_full_held_out_pools, slice_held_out_pools)
    mols, rxns = load_full_held_out_pools(basis="def2-svp", grid_level=1)
    names = ("h", "h2", "o", "oh", "n2o", "n2ohts")
    kept_mols, kept_rxns = slice_held_out_pools(mols, rxns, names)
    assert len(kept_mols) == 6
    assert sorted(r["name"] for r in kept_rxns) == [
        "bh76_h_n2o_to_n2ohts", "w411_h2_atomization", "w411_oh_atomization"]
    assert {r["source_pool"] for r in kept_rxns} == {"bh76", "w411"}


def test_resolve_species_slice_refuses_separators_naming_no_species():
    """A variable holding only separators is a typo, not the full pool: the
    full pool is requested by unsetting the variable, so a value that survives
    the blank check yet names nothing is refused rather than silently widened.
    """
    from xcquinox.alec.full_benchmark_pools import (
        HELDOUT_SPECIES_SLICE_ENV, resolve_species_slice)
    with pytest.raises(ValueError, match="names no species"):
        resolve_species_slice({HELDOUT_SPECIES_SLICE_ENV: " , , "})


def test_slice_held_out_pools_error_names_the_case_sensitivity():
    """Pool keys are case-sensitive, so the refusal message must say so. The
    W4-11 leg is lower case throughout (0 of its 152 names carry a capital);
    BH76 capitalises 33 of its 79, and 11 species -- 'H2', 'H2O', 'CH4',
    'NH3', 'NH2', 'C2H6', 'O', 'PH3', 'H2S', 'HS', 'NH' -- exist under both
    forms as separate pool entries closing different reactions. A message
    calling the names lower case invites a mis-cased slice that resolves
    silently to the wrong species.
    """
    from xcquinox.alec.full_benchmark_pools import slice_held_out_pools
    with pytest.raises(ValueError) as excinfo:
        slice_held_out_pools({"h2": 1, "H2": 2}, [], ("h2", "hydrogen"))
    assert "case-sensitive" in str(excinfo.value)


def test_slice_held_out_pools_refuses_a_slice_closing_no_reaction():
    """A slice closing no reaction would leave the sliced channel averaging an
    empty reaction set, which reads on disk as a completed evaluation carrying
    no reaction data. An empty reaction pool is the separate, legitimate case:
    there is no reaction to lose, and it stays accepted (exercised by the
    order-preserving and absent-species tests, which pass no reactions).
    """
    from xcquinox.alec.full_benchmark_pools import slice_held_out_pools
    rxns = [{"name": "open", "reactants": ["a"], "products": ["c"]}]
    with pytest.raises(ValueError, match="closes no reaction"):
        slice_held_out_pools({"a": 1, "b": 2, "c": 3}, rxns, ("a", "b"))


def test_slice_held_out_pools_refuses_a_repeated_name():
    """A repeated name collapses in the species dict: ``("a", "a")`` yields one
    species for a two-name request, so the slice evaluated is not the slice
    asked for. The repeat is refused, as it is in the variable parser.
    """
    from xcquinox.alec.full_benchmark_pools import slice_held_out_pools
    with pytest.raises(ValueError, match="repeats"):
        slice_held_out_pools({"a": 1, "b": 2}, [], ("a", "a"))


def test_resolve_species_slice_prefers_an_explicit_mapping(monkeypatch):
    """An explicit ``env`` is the whole environment for that call. Merging the
    process environment underneath it would let a variable set in the shell
    slice a caller that passed its own mapping, so the empty mapping resolves
    to the full pool even while the variable is set.
    """
    from xcquinox.alec.full_benchmark_pools import (
        HELDOUT_SPECIES_SLICE_ENV, resolve_species_slice)
    monkeypatch.setenv(HELDOUT_SPECIES_SLICE_ENV, "h,h2")
    assert resolve_species_slice({}) is None
    assert resolve_species_slice(
        {HELDOUT_SPECIES_SLICE_ENV: "o,oh"}) == ("o", "oh")
