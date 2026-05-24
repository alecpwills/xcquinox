"""Tests for xcquinox.alec.cluster.domain — the HPC harness physics tables."""
import pytest

from xcquinox.alec.cluster.domain import (
    ATOMIC_ENERGIES_CHAKRAVORTY,
    KCAL_PER_HA,
    DFS_POOL_SIZE,
    DICK_ATOM_REGULARIZER_SYMS,
    DomainProfile,
    DOMAIN_PROFILES,
    get_domain_profile,
    bh76_meta_to_loss_dict,
    ip13_meta_to_loss_dict,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _StubPoint:
    """Minimal stand-in for a TrainingPoint: the meta-to-loss-dict extractors
    only ever read `.name` and `.metadata`, never `.species`."""
    def __init__(self, name, metadata):
        self.name = name
        self.metadata = metadata


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

def test_kcal_per_ha_value():
    """KCAL_PER_HA is the CODATA-2018 Hartree-in-kcal/mol constant."""
    assert KCAL_PER_HA == 627.5094740631


@pytest.mark.parametrize(
    "sym, expected",
    [
        ("H", -0.5),
        ("C", -37.845),
        ("N", -54.5892),
        ("O", -75.0673),
        ("F", -99.7339),
        ("Li", -7.4781),
        ("Na", -162.2546),
        ("S", -398.1095),
    ],
)
def test_atomic_energies_values(sym, expected):
    """Every ATOMIC_ENERGIES_CHAKRAVORTY entry equals its expected number."""
    assert ATOMIC_ENERGIES_CHAKRAVORTY[sym] == expected


def test_atomic_energies_exact_keys():
    """The dict carries exactly the 8 expected element symbols."""
    assert set(ATOMIC_ENERGIES_CHAKRAVORTY) == {
        "H", "C", "N", "O", "F", "Li", "Na", "S"
    }


def test_lithium_is_corrected_value():
    """Li MUST be the exact non-relativistic total -7.4781, NOT the HF
    limit -7.4327 (the value was corrected in a prior fix)."""
    assert ATOMIC_ENERGIES_CHAKRAVORTY["Li"] == -7.4781
    assert ATOMIC_ENERGIES_CHAKRAVORTY["Li"] != -7.4327


def test_sulfur_is_chakravorty_value():
    """S MUST be the genuine Chakravorty 1993 (PRA 47, 3649) Table-I exact
    non-relativistic total -398.1095, NOT the prior -398.0 placeholder
    (the value was corrected in the Round-3 GMTKN55 realignment)."""
    assert ATOMIC_ENERGIES_CHAKRAVORTY["S"] == -398.1095
    assert ATOMIC_ENERGIES_CHAKRAVORTY["S"] != -398.0


def test_dick_atom_regularizer_syms():
    """The Dick atom-regularizer set is the H/Li pair."""
    assert DICK_ATOM_REGULARIZER_SYMS == ("H", "Li")


# ---------------------------------------------------------------------------
# DomainProfile + registry
# ---------------------------------------------------------------------------

def test_get_domain_profile_returns_profile():
    """get_domain_profile resolves a registered name to a DomainProfile."""
    prof = get_domain_profile("dfs_step7")
    assert isinstance(prof, DomainProfile)
    assert prof.name == "dfs_step7"
    assert prof in DOMAIN_PROFILES.values()


def test_get_domain_profile_unknown_name_raises():
    """An unregistered profile name raises a clear ValueError."""
    with pytest.raises(ValueError, match="Unknown domain profile"):
        get_domain_profile("not_a_real_profile")


def test_domain_profile_exposes_pool_size():
    """The profile exposes an integer pool_size (read by
    grid_config.validate_grid_semantics to bound subset_size)."""
    prof = get_domain_profile("dfs_step7")
    assert isinstance(prof.pool_size, int)
    assert prof.pool_size == DFS_POOL_SIZE == 26


def test_domain_profile_carries_physics_tables():
    """The profile bundles the Chakravorty table, the regularizer set, the
    conversion constant, and the two extractor callables."""
    prof = get_domain_profile("dfs_step7")
    assert prof.atom_energies == ATOMIC_ENERGIES_CHAKRAVORTY
    assert prof.regularize_atom_syms == ("H", "Li")
    assert prof.kcal_per_ha == KCAL_PER_HA
    assert prof.bh76_meta_to_loss_dict is bh76_meta_to_loss_dict
    assert prof.ip13_meta_to_loss_dict is ip13_meta_to_loss_dict


def test_domain_profile_is_frozen():
    """DomainProfile is a frozen dataclass — assignment raises."""
    prof = get_domain_profile("dfs_step7")
    with pytest.raises(Exception):
        prof.pool_size = 99


def test_pool_size_matches_canonical_pool():
    """DFS_POOL_SIZE matches the actual length of the canonical DFS pool."""
    from xcquinox.alec.training_points import build_dfs_pool_points
    assert len(build_dfs_pool_points()) == DFS_POOL_SIZE


# ---------------------------------------------------------------------------
# bh76 / ip13 meta-to-loss-dict extractors
# ---------------------------------------------------------------------------

def test_bh76_meta_to_loss_dict_converts_to_ha():
    """bh76_meta_to_loss_dict carries reactants/products/coeffs through and
    converts e_rxn_ref from kcal/mol to Ha."""
    tp = _StubPoint(
        name="rxnA",
        metadata={
            "reactants": ("OH", "N2"),
            "products": ("H", "N2O"),
            "coeffs": (-1, -1, 1, 1),
            "e_rxn_ref": 10.0,  # kcal/mol
        },
    )
    out = bh76_meta_to_loss_dict(tp)
    assert out["name"] == "rxnA"
    assert out["reactants"] == ("OH", "N2")
    assert out["products"] == ("H", "N2O")
    assert out["coeffs"] == (-1, -1, 1, 1)
    assert out["e_rxn_ref"] == pytest.approx(10.0 / KCAL_PER_HA)


def test_bh76_meta_to_loss_dict_missing_eref_omitted():
    """When e_rxn_ref is absent the key is omitted from the loss dict."""
    tp = _StubPoint(name="rxnB", metadata={"reactants": (), "products": ()})
    out = bh76_meta_to_loss_dict(tp)
    assert "e_rxn_ref" not in out
    assert out["reactants"] == ()
    assert out["products"] == ()
    assert out["coeffs"] == ()


def test_ip13_meta_to_loss_dict_converts_to_ha():
    """ip13_meta_to_loss_dict carries neutral/cation through and converts
    ip_ref from kcal/mol to Ha."""
    tp = _StubPoint(
        name="Li_IP",
        metadata={"neutral": "Li", "cation": "Li+", "ip_ref": 124.3},
    )
    out = ip13_meta_to_loss_dict(tp)
    assert out["name"] == "Li_IP"
    assert out["neutral"] == "Li"
    assert out["cation"] == "Li+"
    assert out["ip_ref"] == pytest.approx(124.3 / KCAL_PER_HA)


def test_ip13_meta_to_loss_dict_missing_ipref_omitted():
    """When ip_ref is absent the key is omitted from the loss dict."""
    tp = _StubPoint(name="X_IP", metadata={"neutral": "X", "cation": "X+"})
    out = ip13_meta_to_loss_dict(tp)
    assert "ip_ref" not in out
    assert out["neutral"] == "X"
    assert out["cation"] == "X+"


def test_extractors_on_real_pool_points():
    """The extractors work on real bh76/ip13 TrainingPoints from the DFS
    pool and produce Ha-scaled reference values (|Ha| < |kcal/mol|)."""
    from xcquinox.alec.training_points import build_dfs_pool_points
    pool = build_dfs_pool_points()

    bh76_pts = [p for p in pool if p.kind == "bh76"]
    ip13_pts = [p for p in pool if p.kind == "ip13"]
    assert bh76_pts, "expected bh76 points in the DFS pool"
    assert ip13_pts, "expected ip13 points in the DFS pool"

    for tp in bh76_pts:
        out = bh76_meta_to_loss_dict(tp)
        assert out["name"] == tp.name
        assert out["reactants"] == tp.metadata.get("reactants", ())
        assert out["products"] == tp.metadata.get("products", ())
        assert out["coeffs"] == tp.metadata.get("coeffs", ())
        eref = tp.metadata.get("e_rxn_ref")
        if eref is not None:
            assert out["e_rxn_ref"] == pytest.approx(float(eref) / KCAL_PER_HA)

    for tp in ip13_pts:
        out = ip13_meta_to_loss_dict(tp)
        assert out["name"] == tp.name
        assert out["neutral"] == tp.metadata.get("neutral")
        assert out["cation"] == tp.metadata.get("cation")
        ipref = tp.metadata.get("ip_ref")
        if ipref is not None:
            assert out["ip_ref"] == pytest.approx(float(ipref) / KCAL_PER_HA)
