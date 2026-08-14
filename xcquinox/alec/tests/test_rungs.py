"""Library-side rung taxonomy + per-rung seed policy (xcquinox.alec.rungs).

The load-bearing properties: classification is derived strictly from the
architecture registry (unregistered names raise instead of guessing from
name tokens -- the analysis-side ``arch_style`` keeps the token fallback
for legacy display names), and the SCF seed policy is a single predicate
shared by the spec builder, run validation, and the figure layer.
"""
import pytest

from xcquinox.alec import rungs

# The 11 v4-campaign archs and their rungs, pinned explicitly.
_V4_RUNGS = {
    "deep_3x16": rungs.RUNG_GGA,
    "deep_attn_3x16": rungs.RUNG_GGA,
    "deep_cusp_3x16": rungs.RUNG_GGA,
    "deep_rung35_3x16": rungs.RUNG_R35,
    "deep_rung35_attn_3x16": rungs.RUNG_R35,
    "deep_rung35ms_3x16": rungs.RUNG_R35,
    "deep_mgga_3x16": rungs.RUNG_MGGA,
    "deep_mgga_attn_3x16": rungs.RUNG_MGGA,
    "deep_cusp_mgga_3x16": rungs.RUNG_MGGA,
    "deep_rung35_mgga_3x16": rungs.RUNG_R35_MGGA,
    "deep_rung35ms_mgga_3x16": rungs.RUNG_R35_MGGA,
}

# Phase-1 seed policy: SCAN iff the meta-GGA ingredient is present
# (pure rung-3.5 archs keep the PBE seed and carry over from v4).
_MGGA_FAMILY = (
    "deep_mgga_3x16", "deep_mgga_attn_3x16", "deep_rung35_mgga_3x16",
    "deep_cusp_mgga_3x16", "deep_rung35ms_mgga_3x16",
)


@pytest.mark.parametrize("arch,rung", sorted(_V4_RUNGS.items()))
def test_rung_of_pins_all_v4_archs(arch, rung):
    assert rungs.rung_of(arch) == rung


def test_rung_order_and_rank_map():
    assert rungs.RUNG_ORDER == (rungs.RUNG_GGA, rungs.RUNG_MGGA,
                                rungs.RUNG_R35, rungs.RUNG_R35_MGGA)
    assert [rungs.RUNG_RANK[r] for r in rungs.RUNG_ORDER] == [0, 1, 2, 3]
    assert rungs.rung_rank("deep_3x16") < rungs.rung_rank("deep_mgga_3x16") \
        < rungs.rung_rank("deep_rung35_3x16") \
        < rungs.rung_rank("deep_rung35_mgga_3x16")


def test_rung_from_ingredients_mapping():
    assert rungs.rung_from_ingredients(False, False) == rungs.RUNG_GGA
    assert rungs.rung_from_ingredients(True, False) == rungs.RUNG_MGGA
    assert rungs.rung_from_ingredients(False, True) == rungs.RUNG_R35
    assert rungs.rung_from_ingredients(True, True) == rungs.RUNG_R35_MGGA


def test_seed_xc_phase1_mgga_scan_policy():
    for arch in _V4_RUNGS:
        expected = "scan" if arch in _MGGA_FAMILY else "pbe"
        assert rungs.seed_xc_for_arch(arch) == expected, arch
    # default policy is the production phase-1 policy
    assert rungs.seed_xc_for_arch("deep_mgga_3x16",
                                  policy="mgga_scan") == "scan"


def test_seed_xc_beyond_gga_policy_covers_rung35():
    scan = {a for a in _V4_RUNGS
            if rungs.seed_xc_for_arch(a, policy="beyond_gga_scan") == "scan"}
    assert scan == set(_MGGA_FAMILY) | {
        "deep_rung35_3x16", "deep_rung35_attn_3x16", "deep_rung35ms_3x16"}


def test_unregistered_arch_raises_not_guesses():
    # 'deep_mgga' is a legacy display name with an obvious token reading;
    # the LIBRARY predicate must refuse it rather than name-parse.
    with pytest.raises(KeyError):
        rungs.rung_of("deep_mgga")
    with pytest.raises(KeyError):
        rungs.seed_xc_for_arch("deep_mgga")


def test_unknown_seed_policy_raises():
    with pytest.raises(ValueError):
        rungs.seed_xc_for_arch("deep_3x16", policy="always_scan")
