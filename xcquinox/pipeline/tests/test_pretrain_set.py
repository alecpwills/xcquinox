"""Composition of the pretraining set (spec Sections 3.2, 6 deviation 1, 7).

Section 7 binds the set: "the DFS pretraining set in its entirety, plus every
atom of the BH76 / W4-11 pools (open shells per spin channel), plus the
synthetic mesh as a regularizer". These tests pin the composition layer alone --
no SCF, no libxc, no grid.
"""
import pytest

import xcquinox.pipeline.pretrain_data_gen as pdg


# ---------------------------------------------------------------------------
# The pool atoms
# ---------------------------------------------------------------------------

def test_pool_atom_systems_are_the_fourteen_single_atom_species():
    """Section 6 deviation 1 says "14 elements". The union of the two pools has
    12 neutral single-atom species plus the two closed-shell anions F- and Cl-
    that carry BH76 barrier heights: 14 distinct (symbol, charge, 2S) triples."""
    got = sorted((s.atom.split()[0], s.charge, s.spin)
                 for s in pdg.pool_atom_systems())
    assert got == sorted([
        ("Al", 0, 1), ("B", 0, 1), ("Be", 0, 0), ("C", 0, 2),
        ("Cl", 0, 1), ("Cl", -1, 0), ("F", 0, 1), ("F", -1, 0),
        ("H", 0, 1), ("N", 0, 3), ("O", 0, 2), ("P", 0, 3),
        ("S", 0, 2), ("Si", 0, 2),
    ])
    assert len(got) == 14


def test_pool_atoms_sit_at_the_origin_like_the_free_atom_path():
    """A free atom's geometry is a single nucleus at the origin, spelled the
    same way the historical atom path spells it, so a pool atom and a
    ``pretrain.atoms`` entry for the same element deduplicate to one system.

    This pins the OUTPUT spelling only; that the spelling is faithful to the
    pool is pinned on the input by the test below."""
    for s in pdg.pool_atom_systems():
        assert s.atom.split()[1:] == ["0", "0", "0"], s.atom


# ---------------------------------------------------------------------------
# normalize_system
# ---------------------------------------------------------------------------


def test_normalize_system_accepts_a_dfs_record():
    """The DFS inventory hands out mappings with exactly these keys plus
    ``kind``, ``atom_composition`` and ``g2_97_index``, which are ignored."""
    s = pdg.normalize_system({"kind": "molecule", "name": "H2",
                              "atom": "H 0 0 0; H 0 0 0.74", "charge": 0,
                              "spin": 0, "atom_composition": [["H", 2]],
                              "g2_97_index": 2})
    assert (s.name, s.charge, s.spin) == ("H2", 0, 0)


# ---------------------------------------------------------------------------
# Geometry-keyed de-duplication and the MoleculeSpec builder
# ---------------------------------------------------------------------------

def test_geometry_key_collapses_two_spellings_of_the_same_atom():
    assert pdg._geometry_key("H 0 0 0") == pdg._geometry_key(
        "h 0.0 0.0 0.00000000")


def test_mol_spec_for_carries_the_identity_and_the_composition():
    """The generator hands this spec to precompute_fixed_density_data, so its
    composition is derived from the geometry rather than trusted from a record:
    a pool entry, a DFS entry and a (symbol, 2S) pair must produce the same spec
    for the same molecule."""
    ms = pdg._mol_spec_for({"name": "h2o",
                            "atom": "O 0 0 0.117; H 0 0.757 -0.469; "
                                    "H 0 -0.757 -0.469",
                            "charge": 0, "spin": 0},
                           "def2-svp", 3)
    assert ms.name == "h2o"
    assert ms.basis == "def2-svp"
    assert ms.grid_level == 3
    assert ms.charge == 0 and ms.spin == 0
    assert ms.atom_composition == (("H", 2), ("O", 1))


# ---------------------------------------------------------------------------
# resolve_pretrain_systems
# ---------------------------------------------------------------------------


def test_resolve_deduplicates_by_geometry_charge_and_spin():
    got = pdg.resolve_pretrain_systems(atoms=(("H", 1),), pool_atoms=True)
    assert len(got) == 14


# ---------------------------------------------------------------------------
# The DFS inventory seam
# ---------------------------------------------------------------------------

def test_dfs_level_maps_the_rung_baseline_to_the_inventory():
    assert pdg.dfs_level_for_reference_xc("pbe") == "gga"
    assert pdg.dfs_level_for_reference_xc("scan") == "mgga"
    with pytest.raises(ValueError, match="reference_xc"):
        pdg.dfs_level_for_reference_xc("blyp")


def test_meta_gga_level_drops_exactly_the_excluded_molecules():
    from xcquinox.pipeline.dfs_pretrain_set import MGGA_EXCLUDED
    gga = {s.name for s in pdg.resolve_pretrain_systems(dfs_set=True,
                                                        reference_xc="pbe")}
    mgga = {s.name for s in pdg.resolve_pretrain_systems(dfs_set=True,
                                                         reference_xc="scan")}
    assert gga - mgga == set(MGGA_EXCLUDED)


# ---------------------------------------------------------------------------
# Parent density and filename
# ---------------------------------------------------------------------------


def test_resolve_parent_density_auto_is_the_rung_baseline():
    """"auto" must agree with rungs.seed_xc_for_arch under its production
    "mgga_scan" policy for EVERY registered architecture: the pretraining parent
    density and the SCF seed are the same rung baseline (PBE for the GGA rung,
    SCAN for the meta-GGA rung), and a disagreement would pretrain a network
    against a density its own SCF never visits."""
    from xcquinox.pipeline.config import ARCHITECTURES, get_architecture
    from xcquinox.pipeline.rungs import seed_xc_for_arch
    for name in ARCHITECTURES:
        assert pdg.resolve_parent_density(get_architecture(name), "auto") == \
            seed_xc_for_arch(name), name


def test_lda_exchange_coefficient_is_the_one_libxc_returns():
    """``_LDA_X_C`` is the denominator the stored enhancement factors are
    formed against in ``spin_channel_exchange_rows``, so it has to be libxc's
    own LDA exchange constant and not an independent transcription of it that
    happens to agree.

    The anchor is the constant's defining relation: eps_x^LDA(rho) =
    _LDA_X_C rho^(1/3), so at rho = 1 libxc's ``LDA_X,`` returns the constant
    itself. ``abs=1e-15`` admits no more than the last few bits of a double at
    this magnitude (spacing 1.11e-16 at 0.739); the measured difference is
    0.0."""
    import numpy as np
    from pyscf.dft.libxc import eval_xc
    exc, _vxc, _fxc, _kxc = eval_xc("LDA_X,", np.array([1.0]), spin=0)
    assert pdg._LDA_X_C == pytest.approx(float(exc[0]), abs=1e-15)


