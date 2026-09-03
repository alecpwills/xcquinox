"""Two-tier atomization gate and the in-place certificate re-verdict.

The gate (``fidelity._ae_gate_terms``) and the re-verdict
(``fidelity.regate_certificate_payload``) share one implementation, so these
tests pin the gate arithmetic on hand-computed values, the exact reason text
of each tier, and every refusal class of the regate -- each tier and each
refusal is exercised by an input that makes it FIRE, not only by inputs that
pass. The certificate payloads are synthetic but carry the exact keys the
real writer records (verdict, tolerances, per_system, per_atomization,
summary), so a schema drift in either direction breaks here.
"""
import copy
import math

from xcquinox.alec.cluster import fidelity as fid
from xcquinox.alec.cluster.grid_config import FidelityConfig

MAE_CFG = FidelityConfig(tol_AE=1.0, tol_atom=1.0, tol_AE_aggregate="mae",
                         tol_AE_max_backstop=2.0)
MAX_CFG = FidelityConfig()  # the original single-tier gate, all defaults


def _rows(*vals, names=None):
    names = names or [f"m{i}" for i in range(len(vals))]
    return [{"name": n, "dAE_kcalmol": v} for n, v in zip(names, vals)]


# ---------------------------------------------------------------------------
# _ae_gate_terms: arithmetic and tier texts
# ---------------------------------------------------------------------------

def test_gate_terms_hand_values_pass_the_mae_gate_with_the_species_flagged():
    rows = _rows(0.2, -0.4, 1.42, names=["a", "b", "c"])
    t = fid._ae_gate_terms(rows, MAE_CFG)
    assert t["max"] == 1.42
    assert abs(t["mean"] - (0.2 + 0.4 + 1.42) / 3) < 1e-15
    assert abs(t["rmse"]
               - math.sqrt((0.04 + 0.16 + 1.42 ** 2) / 3)) < 1e-15
    assert t["species_over_1_kcalmol"] == ["c"]
    assert t["reasons"] == []


def test_max_gate_reason_text_is_byte_identical_to_the_original():
    rows = _rows(0.2, -0.4, 1.42)
    t = fid._ae_gate_terms(rows, MAX_CFG)
    assert t["reasons"] == [
        f"max |dAE| {1.42!r} kcal/mol exceeds tol_AE {1.0!r} kcal/mol"]


def test_mae_tier_and_backstop_both_fire_on_a_gross_outlier():
    rows = _rows(4.6, 0.1, 0.1)
    t = fid._ae_gate_terms(rows, MAE_CFG)
    mean = (4.6 + 0.1 + 0.1) / 3
    assert abs(t["mean"] - mean) < 1e-15 and mean > 1.0
    assert len(t["reasons"]) == 2
    assert any("mean |dAE|" in r and "(aggregate 'mae')" in r
               for r in t["reasons"])
    assert any("tol_AE_max_backstop" in r for r in t["reasons"])


def test_backstop_fires_alone_when_the_mean_is_clean():
    rows = _rows(2.5, 0.1, 0.1)
    t = fid._ae_gate_terms(rows, MAE_CFG)
    assert t["mean"] < 1.0
    assert t["reasons"] == [
        f"max |dAE| {2.5!r} kcal/mol exceeds tol_AE_max_backstop "
        f"{2.0!r} kcal/mol"]


def test_the_gates_are_strict_inequalities():
    # mean exactly at tol_AE and max exactly at the backstop both PASS.
    t = fid._ae_gate_terms(_rows(1.0, 1.0, 1.0), MAE_CFG)
    assert t["mean"] == 1.0 and t["reasons"] == []
    t = fid._ae_gate_terms(_rows(2.0, 0.05, 0.05), MAE_CFG)
    assert t["max"] == 2.0 and t["reasons"] == []


def test_none_rows_are_excluded_from_the_statistics():
    rows = _rows(0.2, 0.3) + [{"name": "bad", "dAE_kcalmol": None,
                               "error": "not finite"}]
    t = fid._ae_gate_terms(rows, MAE_CFG)
    assert t["max"] == 0.3
    assert abs(t["mean"] - 0.25) < 1e-15


def test_no_usable_rows_is_the_untested_reason():
    t = fid._ae_gate_terms([], MAE_CFG)
    assert t["max"] is None and t["mean"] is None and t["rmse"] is None
    assert t["reasons"] == [
        "no atomization offset could be formed, so tol_AE is untested"]
    t2 = fid._ae_gate_terms([{"name": "x", "dAE_kcalmol": None}], MAX_CFG)
    assert t2["reasons"] == t["reasons"]


def test_species_flag_is_recorded_under_both_aggregates():
    rows = _rows(1.42, 0.1)
    for cfg in (MAE_CFG, MAX_CFG):
        t = fid._ae_gate_terms(rows, cfg)
        assert t["species_over_1_kcalmol"] == ["m0"]
    # ... at the module's stated constant.
    assert fid.SPECIES_FLAG_KCALMOL == 1.0


def test_tolerances_record_drops_the_backstop_under_max():
    r = fid._tolerances_record(MAX_CFG)
    assert r["tol_AE_aggregate"] == "max"
    assert r["tol_AE_max_backstop"] is None
    r = fid._tolerances_record(MAE_CFG)
    assert r["tol_AE_max_backstop"] == 2.0


# ---------------------------------------------------------------------------
# regate_certificate_payload
# ---------------------------------------------------------------------------

def _payload(mol_dae=1.42, *, atom_mha=0.5, converged=True, grid_diff=0.0,
             record_diff=0.0, error_row=False, verdict="FAIL",
             tolerances=None):
    """A synthetic certificate with the exact keys the real writer records."""
    per_system = [
        {"name": "atom_H", "dE_xc_mHa": atom_mha, "is_atom": True,
         "parent_grid_diff_Ha": grid_diff, "parent_record_diff_Ha": 0.0,
         "reference_scf_converged": True},
        {"name": "H2", "dE_xc_mHa": 1.5, "is_atom": False,
         "parent_grid_diff_Ha": 0.0, "parent_record_diff_Ha": record_diff,
         "reference_scf_converged": converged},
    ]
    if error_row:
        per_system.append({"name": "broken", "error": "could not evaluate"})
    tol = tolerances if tolerances is not None else {
        "tol_AE": 1.0, "tol_atom": 1.0, "override_reason": None}
    return {
        "verdict": verdict,
        "arch": "deep_cusp_3x16",
        "per_system": per_system,
        "per_atomization": [{"name": "H2", "dAE_kcalmol": mol_dae},
                            {"name": "H2O", "dAE_kcalmol": 0.2}],
        "tolerances": tol,
        "summary": {"max_atom_mHa": atom_mha, "max_dAE_kcalmol": mol_dae,
                    "failure_reasons": ["max |dAE| ..."]},
    }


def test_regate_flips_a_single_species_fail_to_pass_with_provenance():
    p = _payload(1.42)
    before = copy.deepcopy(p)
    new, report = fid.regate_certificate_payload(
        p, MAE_CFG, config_source="configs/x.yaml")
    assert p == before, "the input payload was mutated"
    assert new is not None and new["verdict"] == fid.VERDICT_PASS
    assert "FAIL -> PASS" in report
    assert new["tolerances"] == {
        "tol_AE": 1.0, "tol_atom": 1.0, "tol_AE_aggregate": "mae",
        "tol_AE_max_backstop": 2.0, "override_reason": None}
    s = new["summary"]
    assert s["failure_reasons"] == []
    assert s["max_dAE_kcalmol"] == 1.42
    assert abs(s["mean_dAE_kcalmol"] - (1.42 + 0.2) / 2) < 1e-15
    assert s["species_over_1_kcalmol"] == ["H2"]
    r = new["regate"]
    assert r["original_verdict"] == "FAIL"
    assert r["original_tolerances"] == before["tolerances"]
    assert r["original_failure_reasons"] == ["max |dAE| ..."]
    assert isinstance(r["regated_at"], str) and r["regated_at"].endswith("Z")
    assert r["config_source"] == "configs/x.yaml"


def test_regate_keeps_a_backstop_fail_failing():
    new, report = fid.regate_certificate_payload(
        _payload(4.6), MAE_CFG, config_source="x")
    assert new is not None and new["verdict"] == fid.VERDICT_FAIL
    assert any("tol_AE_max_backstop" in r
               for r in new["summary"]["failure_reasons"])
    assert "still failing" in report


def test_regate_refuses_an_error_row():
    new, report = fid.regate_certificate_payload(
        _payload(error_row=True), MAE_CFG, config_source="x")
    assert new is None and "evaluation error" in report and "broken" in report


def test_regate_refuses_an_unconverged_reference():
    new, report = fid.regate_certificate_payload(
        _payload(converged=False), MAE_CFG, config_source="x")
    assert new is None and "did not converge" in report and "H2" in report


def test_regate_refuses_a_missing_converged_key_conservatively():
    p = _payload()
    del p["per_system"][1]["reference_scf_converged"]
    new, report = fid.regate_certificate_payload(
        p, MAE_CFG, config_source="x")
    assert new is None and "did not converge" in report


def test_regate_refuses_an_atom_over_tol_atom():
    new, report = fid.regate_certificate_payload(
        _payload(atom_mha=1.3), MAE_CFG, config_source="x")
    assert new is None and "tol_atom" in report
    assert "unchanged by the" in report  # the atom gate is not the aggregate


def test_regate_refuses_a_parent_route_disagreement():
    over = 2.0 * fid.PARENT_GRID_TOL_HA
    new, report = fid.regate_certificate_payload(
        _payload(grid_diff=over), MAE_CFG, config_source="x")
    assert new is None and "disagree" in report
    new, report = fid.regate_certificate_payload(
        _payload(record_diff=over), MAE_CFG, config_source="x")
    assert new is None and "accumulated" in report


def test_regate_at_exactly_the_route_tolerance_is_not_refused():
    new, _ = fid.regate_certificate_payload(
        _payload(grid_diff=fid.PARENT_GRID_TOL_HA), MAE_CFG,
        config_source="x")
    assert new is not None


def test_regate_refuses_an_unrecognized_verdict():
    new, report = fid.regate_certificate_payload(
        _payload(verdict="pass"), MAE_CFG, config_source="x")
    assert new is None and "recognised verdict" in report.replace(
        "recognized", "recognised") or "verdict" in report


def test_regate_of_an_identically_gated_pass_is_a_noop():
    already = fid._tolerances_record(MAE_CFG)
    p = _payload(0.5, verdict="PASS", tolerances=already)
    new, report = fid.regate_certificate_payload(
        p, MAE_CFG, config_source="x")
    assert new is None and "already" in report


def test_regate_under_the_max_gate_rewrites_a_stale_tolerance_block():
    # A PASS written under mae regates under max: same verdict is possible,
    # but a different tolerance block still rewrites so the run states ONE
    # gate.
    p = _payload(0.5, verdict="PASS",
                 tolerances=fid._tolerances_record(MAE_CFG))
    new, _ = fid.regate_certificate_payload(p, MAX_CFG, config_source="x")
    assert new is not None
    assert new["verdict"] == fid.VERDICT_PASS
    assert new["tolerances"]["tol_AE_aggregate"] == "max"
    assert new["tolerances"]["tol_AE_max_backstop"] is None
