"""Tests for xcquinox.pipeline.cluster.domain: the HPC harness physics tables."""
import pytest

from xcquinox.pipeline.cluster.domain import (
    ATOMIC_ENERGIES_CHAKRAVORTY,
    KCAL_PER_HA,
    DomainProfile,
    DOMAIN_PROFILES,
    get_domain_profile,
    bh76_meta_to_loss_dict,
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


def test_atomic_energies_exact_keys():
    """The dict carries exactly the 14 expected element symbols (the original
    DFS 8 plus the 6 heavier elements required by the BH76+W4-11 pool)."""
    assert set(ATOMIC_ENERGIES_CHAKRAVORTY) == {
        "H", "C", "N", "O", "F", "Li", "Na", "S",
        "Be", "B", "Al", "Si", "P", "Cl",
    }


def test_lithium_is_corrected_value():
    """Li MUST be the exact non-relativistic total -7.4781, NOT the HF
    limit -7.4327 (the value was corrected in a prior fix)."""
    assert ATOMIC_ENERGIES_CHAKRAVORTY["Li"] == -7.4781
    assert ATOMIC_ENERGIES_CHAKRAVORTY["Li"] != -7.4327


def test_sulfur_is_chakravorty_value():
    """S MUST be the genuine Chakravorty 1993 (PRA 47, 3649) Table-XI exact
    non-relativistic total -398.1095, NOT the prior -398.0 placeholder
    (the value was corrected in the Round-3 GMTKN55 realignment)."""
    assert ATOMIC_ENERGIES_CHAKRAVORTY["S"] == -398.1095
    assert ATOMIC_ENERGIES_CHAKRAVORTY["S"] != -398.0


# ---------------------------------------------------------------------------
# DomainProfile + registry
# ---------------------------------------------------------------------------

def test_get_domain_profile_returns_profile():
    """get_domain_profile resolves a registered name to a DomainProfile."""
    prof = get_domain_profile("dfs_step7")
    assert isinstance(prof, DomainProfile)
    assert prof.name == "dfs_step7"
    assert prof in DOMAIN_PROFILES.values()


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


# ---------------------------------------------------------------------------
# Atom-energy coverage of training pools (CFG-01 regression guard)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Import weight: the physics tables are read on a login node
# ---------------------------------------------------------------------------


