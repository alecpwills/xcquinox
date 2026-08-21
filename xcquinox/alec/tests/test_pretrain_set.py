"""Composition of the pretraining set (spec Sections 3.2, 6 deviation 1, 7).

Section 7 binds the set: "the DFS pretraining set in its entirety, plus every
atom of the BH76 / W4-11 pools (open shells per spin channel), plus the
synthetic mesh as a regularizer". These tests pin the composition layer alone --
no SCF, no libxc, no grid.
"""
import pytest

import xcquinox.alec.pretrain_data_gen as pdg


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


def test_pool_atom_names_are_unique_and_mark_the_anions():
    names = [s.name for s in pdg.pool_atom_systems()]
    assert len(set(names)) == len(names)
    assert {"F-", "Cl-", "Be", "H"} <= set(names)


def test_pool_atoms_sit_at_the_origin_like_the_free_atom_path():
    """A free atom's geometry is a single nucleus at the origin, spelled the
    same way the historical atom path spells it, so a pool atom and a
    ``pretrain.atoms`` entry for the same element deduplicate to one system.

    This pins the OUTPUT spelling only; that the spelling is faithful to the
    pool is pinned on the input by the test below."""
    for s in pdg.pool_atom_systems():
        assert s.atom.split()[1:] == ["0", "0", "0"], s.atom


def test_the_pools_own_single_atom_species_sit_at_the_origin():
    """``pool_atom_systems`` writes the free-atom geometry itself
    (``"<Sym> 0 0 0"``) rather than carrying the pool's coordinates through, so
    "the geometry is the pool's own" is a claim about the INPUT: a single-atom
    species displaced from the origin in the committed pool JSON -- a nucleus
    left at a fragment position by a regeneration from the GMTKN55 source --
    would be silently moved to the origin, and the pretraining atom would then
    be a different system from the pool species the certificate bounds.

    Fifteen species by name, not fourteen: the union spells atomic oxygen both
    ``O`` (BH76) and ``o`` (W4-11), and the two collapse only downstream, on
    (symbol, charge, 2S)."""
    from xcquinox.alec.full_benchmark_pools import load_full_held_out_pools
    mol_specs, _reactions = load_full_held_out_pools()
    singles = {name: ms for name, ms in mol_specs.items()
               if sum(n for _sym, n in ms.atom_composition) == 1}
    assert len(singles) == 15, sorted(singles)
    for name, ms in sorted(singles.items()):
        chunks = [c for c in str(ms.atom).replace("\n", ";").split(";")
                  if c.split()]
        assert len(chunks) == 1, (name, ms.atom)
        fields = chunks[0].split()
        assert len(fields) == 4, (name, ms.atom)
        assert [float(v) for v in fields[1:]] == [0.0, 0.0, 0.0], (name,
                                                                   ms.atom)


# ---------------------------------------------------------------------------
# normalize_system
# ---------------------------------------------------------------------------

def test_normalize_system_accepts_the_historical_symbol_spin_pair():
    assert pdg.normalize_system(("O", 2)) == pdg.PretrainSystem(
        name="O", atom="O 0 0 0", charge=0, spin=2)


def test_normalize_system_accepts_a_dfs_record():
    """The DFS inventory hands out mappings with exactly these keys plus
    ``kind``, ``atom_composition`` and ``g2_97_index``, which are ignored."""
    s = pdg.normalize_system({"kind": "molecule", "name": "H2",
                              "atom": "H 0 0 0; H 0 0 0.74", "charge": 0,
                              "spin": 0, "atom_composition": [["H", 2]],
                              "g2_97_index": 2})
    assert (s.name, s.charge, s.spin) == ("H2", 0, 0)


def test_normalize_system_accepts_a_mol_spec():
    from xcquinox.alec.config import MoleculeSpec
    ms = MoleculeSpec(name="he", atom="He 0 0 0", basis="sto-3g", charge=0,
                      spin=0, atom_composition=(("He", 1),))
    assert pdg.normalize_system(ms).atom == "He 0 0 0"


def test_normalize_system_is_idempotent():
    s = pdg.PretrainSystem("x", "H 0 0 0", 0, 1)
    assert pdg.normalize_system(s) is s


def test_normalize_system_refuses_an_unusable_object():
    with pytest.raises(TypeError, match="pretraining system"):
        pdg.normalize_system(42)


# ---------------------------------------------------------------------------
# Geometry-keyed de-duplication and the MoleculeSpec builder
# ---------------------------------------------------------------------------

def test_geometry_key_collapses_two_spellings_of_the_same_atom():
    assert pdg._geometry_key("H 0 0 0") == pdg._geometry_key(
        "h 0.0 0.0 0.00000000")


def test_geometry_key_separates_two_geometries():
    assert pdg._geometry_key("H 0 0 0; H 0 0 0.74") != pdg._geometry_key(
        "H 0 0 0; H 0 0 1.40")


def test_geometry_key_rejects_a_malformed_geometry():
    with pytest.raises(ValueError, match="geometry"):
        pdg._geometry_key("H 0 0")


def test_composition_and_atom_count_come_from_the_geometry():
    assert pdg._composition_from_atom("C 0 0 0; H 0 0 1.1; H 0 1.1 0") == \
        (("C", 1), ("H", 2))
    assert pdg._n_atoms("C 0 0 0; H 0 0 1.1; H 0 1.1 0") == 3


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

def test_resolve_defaults_to_the_historical_four_atoms():
    got = pdg.resolve_pretrain_systems()
    assert tuple((s.name, s.spin) for s in got) == pdg.DEFAULT_PRETRAIN_ATOMS


def test_resolve_honors_an_explicit_atom_list():
    got = pdg.resolve_pretrain_systems(atoms=(("Li", 1), ("C", 2)))
    assert [s.name for s in got] == ["Li", "C"]


def test_resolve_pool_atoms_drops_the_historical_default():
    """Turning an inventory on replaces the four-atom default rather than adding
    to it: He is in neither pool nor the DFS set, and the set the spec binds is
    stated exactly."""
    got = pdg.resolve_pretrain_systems(pool_atoms=True)
    assert len(got) == 14
    assert "He" not in [s.name for s in got]


def test_resolve_keeps_an_explicit_atom_alongside_the_pool():
    got = pdg.resolve_pretrain_systems(atoms=(("He", 0),), pool_atoms=True)
    assert len(got) == 15
    assert got[-1].name == "He"


def test_resolve_deduplicates_by_geometry_charge_and_spin():
    got = pdg.resolve_pretrain_systems(atoms=(("H", 1),), pool_atoms=True)
    assert len(got) == 14


def test_resolve_deduplicates_two_spellings_of_one_molecule():
    """The de-duplication key is the GEOMETRY, not the name. The three
    inventories name the same molecule independently -- the DFS records carry
    G2/97 labels, the pool JSON carries GMTKN55 directory names, and
    ``pretrain.atoms`` carries whatever the YAML wrote -- so a name-keyed
    collapse would pretrain the same water twice under two labels, doubling its
    weight in the loss. Here the two records differ in name, in nucleus
    ordering, in the ``";"``/newline separator and in the number of trailing
    zeros, and agree only in the structure."""
    a = {"name": "water_a",
         "atom": "O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469",
         "charge": 0, "spin": 0}
    b = {"name": "water_b",
         "atom": "H 0 -0.757 -0.469\nH 0 0.7570000 -0.469\nO 0.0 0.0 0.1170",
         "charge": 0, "spin": 0}
    assert len(pdg.resolve_pretrain_systems(atoms=(a, b))) == 1


def test_resolve_deduplicates_a_renamed_pool_atom():
    """The cross-inventory case of the same rule: an explicit ``pretrain.atoms``
    entry naming a pool atom under a different label is still that pool atom.
    The pool spells hydrogen ``H``; a YAML that spells it ``hydrogen`` must not
    add a fifteenth system."""
    got = pdg.resolve_pretrain_systems(
        atoms=(pdg.PretrainSystem("hydrogen", "H 0 0 0", 0, 1),),
        pool_atoms=True)
    assert len(got) == 14, [s.name for s in got]


def test_resolve_keeps_an_ion_beside_its_neutral_atom():
    """Same geometry, different charge: two physical systems, both kept."""
    got = pdg.resolve_pretrain_systems(pool_atoms=True)
    fluorines = [s for s in got if s.atom.startswith("F ")]
    assert sorted((s.charge, s.spin) for s in fluorines) == [(-1, 0), (0, 1)]


# ---------------------------------------------------------------------------
# The DFS inventory seam
# ---------------------------------------------------------------------------

def test_dfs_level_maps_the_rung_baseline_to_the_inventory():
    assert pdg.dfs_level_for_reference_xc("pbe") == "gga"
    assert pdg.dfs_level_for_reference_xc("scan") == "mgga"
    with pytest.raises(ValueError, match="reference_xc"):
        pdg.dfs_level_for_reference_xc("blyp")


def test_dfs_levels_agree_with_the_inventory_module():
    from xcquinox.alec.dfs_pretrain_set import LEVELS
    assert set(LEVELS) == {pdg.dfs_level_for_reference_xc("pbe"),
                           pdg.dfs_level_for_reference_xc("scan")}


def test_resolve_dfs_set_asks_the_inventory_for_the_rungs_level(monkeypatch):
    calls = []

    def _fake(level):
        calls.append(level)
        return [{"name": "H2", "atom": "H 0 0 0; H 0 0 0.74", "charge": 0,
                 "spin": 0},
                {"name": "Li", "atom": "Li 0 0 0", "charge": 0, "spin": 1}]

    monkeypatch.setattr(pdg, "_dfs_pretrain_records", _fake)
    got = pdg.resolve_pretrain_systems(dfs_set=True, reference_xc="scan")
    assert calls == ["mgga"]
    assert [s.name for s in got] == ["H2", "Li"]


def test_resolve_dfs_set_reads_the_committed_inventory():
    gga = pdg.resolve_pretrain_systems(dfs_set=True, reference_xc="pbe")
    assert len(gga) == 30, [s.name for s in gga]
    mgga = pdg.resolve_pretrain_systems(dfs_set=True, reference_xc="scan")
    assert len(mgga) == 28, [s.name for s in mgga]


def test_meta_gga_level_drops_exactly_the_excluded_molecules():
    from xcquinox.alec.dfs_pretrain_set import MGGA_EXCLUDED
    gga = {s.name for s in pdg.resolve_pretrain_systems(dfs_set=True,
                                                        reference_xc="pbe")}
    mgga = {s.name for s in pdg.resolve_pretrain_systems(dfs_set=True,
                                                         reference_xc="scan")}
    assert gga - mgga == set(MGGA_EXCLUDED)


def test_the_v6_set_is_the_dfs_set_plus_every_pool_atom():
    """DFS contributes 8 free atoms, 7 of which (H, N, O, P, S, Cl, Al) are also
    pool atoms; the pools add B, Be, C, F, Si and the two anions. 30 + 7 for the
    GGA rung, 28 + 7 for the meta-GGA rung."""
    gga = pdg.resolve_pretrain_systems(dfs_set=True, pool_atoms=True,
                                       reference_xc="pbe")
    names = [s.name for s in gga]
    assert len(set(names)) == len(names)
    assert len(names) == 37, names
    mgga = pdg.resolve_pretrain_systems(dfs_set=True, pool_atoms=True,
                                        reference_xc="scan")
    assert len(mgga) == 35, [s.name for s in mgga]


# ---------------------------------------------------------------------------
# Parent density and filename
# ---------------------------------------------------------------------------

def test_resolve_parent_density_passes_an_explicit_choice_through():
    from xcquinox.alec.config import get_architecture
    arch = get_architecture("deep_3x16")
    assert pdg.resolve_parent_density(arch, "pbe") == "pbe"
    assert pdg.resolve_parent_density(arch, "scan") == "scan"
    with pytest.raises(ValueError, match="parent_density"):
        pdg.resolve_parent_density(arch, "blyp")


def test_resolve_parent_density_auto_is_the_rung_baseline():
    """"auto" must agree with rungs.seed_xc_for_arch under its production
    "mgga_scan" policy for EVERY registered architecture: the pretraining parent
    density and the SCF seed are the same rung baseline (PBE for the GGA rung,
    SCAN for the meta-GGA rung), and a disagreement would pretrain a network
    against a density its own SCF never visits."""
    from xcquinox.alec.config import ARCHITECTURES, get_architecture
    from xcquinox.alec.rungs import seed_xc_for_arch
    for name in ARCHITECTURES:
        assert pdg.resolve_parent_density(get_architecture(name), "auto") == \
            seed_xc_for_arch(name), name


def test_resolve_parent_density_auto_reads_the_meta_gga_ingredient():
    """The rung is carried by the meta-GGA INGREDIENT, which
    ``rungs.arch_ingredients`` reads as the ``meta_gga`` flag OR a "metagga"
    descriptor. The pairing of the two is enforced in
    ``ArchitectureConfig.from_spec`` alone (``config.py``, "meta_gga=True
    requires a 'metagga' descriptor"), and only in that direction; the
    dataclass constructor used below enforces neither, so an architecture
    assembled outside ``from_spec`` can carry the descriptor alone -- a case
    the registry sweep above cannot reach, and one that would pretrain a
    meta-GGA network on a PBE density its own SCF never visits."""
    from xcquinox.alec.config import ArchitectureConfig, FeatureSpec
    arch = ArchitectureConfig(name="ad_hoc_metagga", depth=3, nodes=16,
                              descriptors=(FeatureSpec(name="metagga"),))
    assert arch.meta_gga is False
    assert pdg.resolve_parent_density(arch, "auto") == "scan"


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


def test_pretrain_data_filename_keeps_the_two_historical_names():
    assert pdg.pretrain_data_filename(False) == "pretrain_data.npz"
    assert pdg.pretrain_data_filename(True) == "pretrain_data_polarized.npz"
    assert pdg.pretrain_data_filename(True, "scan") == \
        "pretrain_data_polarized_scan.npz"
    assert pdg.pretrain_data_filename(False, "scan") == "pretrain_data_scan.npz"
