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

from xcquinox.pipeline.cluster import fidelity as fid
from xcquinox.pipeline.cluster.grid_config import FidelityConfig

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


def test_regate_refuses_an_unconverged_reference():
    new, report = fid.regate_certificate_payload(
        _payload(converged=False), MAE_CFG, config_source="x")
    assert new is None and "did not converge" in report and "H2" in report


# ---------------------------------------------------------------------------
# Non-finite handling and provenance across repeated regates (review round)
# ---------------------------------------------------------------------------

def test_gate_terms_fail_loud_on_a_raw_nan_row():
    """NaN passes an ``is not None`` filter and every ``>`` comparison is
    False, so an unguarded gate returns nan statistics with NO reasons -- a
    silent PASS input. The gate must instead name the non-finite row."""
    rows = [{"name": "a", "dAE_kcalmol": float("nan")},
            {"name": "b", "dAE_kcalmol": 0.1}]
    t = fid._ae_gate_terms(rows, MAE_CFG)
    assert any("non-finite" in r and "a" in r for r in t["reasons"]), \
        t["reasons"]
    # Statistics come from the finite rows alone, never nan.
    assert t["max"] == 0.1 and t["mean"] == 0.1
    assert t["rmse"] == 0.1
    inf_rows = [{"name": "c", "dAE_kcalmol": float("inf")}]
    t = fid._ae_gate_terms(inf_rows, MAE_CFG)
    assert any("non-finite" in r for r in t["reasons"])


def test_second_regate_preserves_the_first_ever_verdict():
    """A regate of an already-regated payload must keep the TRUE original
    verdict/tolerances/reasons and record the chain of rewrites -- not
    overwrite history with the intermediate state."""
    p = _payload(1.63)  # FAIL under max; PASS under mae (mean 0.915, max<2)
    step1, _ = fid.regate_certificate_payload(
        p, MAE_CFG, config_source="one.yaml")
    assert step1["verdict"] == fid.VERDICT_PASS
    step2, _ = fid.regate_certificate_payload(
        step1, MAX_CFG, config_source="two.yaml")
    assert step2 is not None and step2["verdict"] == fid.VERDICT_FAIL
    r = step2["regate"]
    assert r["original_verdict"] == "FAIL"
    assert r["original_tolerances"] == {"tol_AE": 1.0, "tol_atom": 1.0,
                                        "override_reason": None}
    assert r["original_failure_reasons"] == ["max |dAE| ..."]
    assert r["config_source"] == "two.yaml"
    chain = r["chain"]
    assert [c["to_verdict"] for c in chain] == ["PASS", "FAIL"]
    assert chain[0]["config_source"] == "one.yaml"
    assert chain[1]["config_source"] == "two.yaml"
