"""Tests for plot_certificate_summary.py (the certificate quick-look).

Fast + headless (matplotlib Agg): builds synthetic run/pretrain trees with
fabricated certificates -- one FAIL with a flagged species, one legacy-style
certificate lacking the 2026-09-03 summary/tolerance keys -- exercises the
collector, the CSV writer and the plotter, and checks the numbers against
hand-computed values. No cluster data.
"""
import csv
import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plot_certificate_summary as pcs  # noqa: E402


def _cert(arch, daes, *, verdict, two_tier=True, names=None):
    names = names or [f"m{i}" for i in range(len(daes))]
    payload = {
        "arch": arch,
        "verdict": verdict,
        "per_atomization": [{"name": n, "dAE_kcalmol": v}
                            for n, v in zip(names, daes)],
        "tolerances": {"tol_AE": 1.0, "tol_atom": 1.0,
                       "override_reason": None},
        "summary": {"max_dAE_kcalmol": max(abs(v) for v in daes),
                    "failure_reasons": []},
    }
    if two_tier:
        payload["tolerances"]["tol_AE_aggregate"] = "mae"
        payload["tolerances"]["tol_AE_max_backstop"] = 2.0
        devs = [abs(v) for v in daes]
        payload["summary"]["mean_dAE_kcalmol"] = sum(devs) / len(devs)
        payload["summary"]["rmse_dAE_kcalmol"] = math.sqrt(
            sum(v * v for v in devs) / len(devs))
        payload["summary"]["species_over_1_kcalmol"] = [
            n for n, v in zip(names, daes) if abs(v) > 1.0]
        payload["regate"] = {"original_verdict": "FAIL"}
    return payload


def _write_run(tmp_path, name, certs):
    run = tmp_path / name
    for arch, payload in certs.items():
        d = run / "pretrain" / arch
        d.mkdir(parents=True)
        with open(d / "fidelity_certificate.json", "w") as f:
            json.dump(payload, f)
    return str(run)


@pytest.fixture()
def two_label_tree(tmp_path):
    """Two labels x two archs; one two-tier FAIL with a flagged species and
    one legacy-style certificate without any of the new keys."""
    v7 = _write_run(tmp_path, "run_v7", {
        "medium": _cert("medium", [0.2, -0.4, 0.3], verdict="PASS"),
        "shallow": _cert("shallow", [0.2, -0.4, 1.42], verdict="FAIL",
                         names=["a", "b", "C3H8"]),
    })
    legacy = _write_run(tmp_path, "run_legacy", {
        "medium": _cert("medium", [2.0, -4.0], verdict="FAIL",
                        two_tier=False, names=["x", "y"]),
        "shallow": _cert("shallow", [0.1, 0.2], verdict="FAIL",
                         two_tier=False),
    })
    return v7, legacy


def test_collector_recomputes_and_reads_both_schemas(two_label_tree):
    v7, legacy = two_label_tree
    records = pcs.collect_certificates([("v7", v7), ("legacy", legacy)])
    by = {(label, arch): r for label, arch, r in records}
    r = by[("v7", "shallow")]
    assert r["mean"] == pytest.approx((0.2 + 0.4 + 1.42) / 3, abs=1e-12)
    assert r["rmse"] == pytest.approx(
        math.sqrt((0.04 + 0.16 + 1.42 ** 2) / 3), abs=1e-12)
    assert r["max"] == pytest.approx(1.42, abs=1e-12)
    assert r["species_over"] == ["C3H8"]
    assert r["aggregate"] == "mae" and r["backstop"] == 2.0
    # The legacy certificate lacks the new keys: stats recomputed, the flag
    # list derived from per_atomization, the aggregate defaulting to max.
    r = by[("legacy", "medium")]
    assert r["mean"] == pytest.approx(3.0, abs=1e-12)
    assert r["max"] == pytest.approx(4.0, abs=1e-12)
    assert r["species_over"] == ["x", "y"]
    assert r["aggregate"] == "max" and r["backstop"] is None


def test_png_and_csv_written_with_hand_checked_numbers(two_label_tree,
                                                       tmp_path):
    v7, legacy = two_label_tree
    out = tmp_path / "figs" / "certificate_summary.png"
    rc = pcs.main(["--runs", f"v7={v7}", "--runs", f"legacy={legacy}",
                   "--out", str(out)])
    assert rc == 0
    assert out.is_file() and out.stat().st_size > 0
    csv_path = tmp_path / "figs" / "certificate_summary.csv"
    assert csv_path.is_file()
    with open(csv_path) as f:
        rows = {(r["label"], r["arch"]): r for r in csv.DictReader(f)}
    assert len(rows) == 4
    r = rows[("v7", "shallow")]
    assert float(r["mean_abs_dAE_kcalmol"]) == pytest.approx(
        (0.2 + 0.4 + 1.42) / 3, abs=1e-12)
    assert float(r["max_abs_dAE_kcalmol"]) == pytest.approx(1.42, abs=1e-12)
    assert r["species_over_1_kcalmol"] == "C3H8"
    assert r["verdict"] == "FAIL"
    assert r["tol_AE_aggregate"] == "mae"
    r = rows[("legacy", "medium")]
    assert float(r["rmse_dAE_kcalmol"]) == pytest.approx(
        math.sqrt((4.0 + 16.0) / 2), abs=1e-12)
    assert r["tol_AE_aggregate"] == "max"


def test_a_repeated_label_merges_disjoint_arch_sets(tmp_path):
    a = _write_run(tmp_path, "g1", {
        "medium": _cert("medium", [0.2], verdict="PASS")})
    b = _write_run(tmp_path, "g2a", {
        "deep_3x16": _cert("deep_3x16", [0.3], verdict="PASS")})
    records = pcs.collect_certificates([("v7", a), ("v7", b)])
    assert {(l, ar) for l, ar, _ in records} == {("v7", "medium"),
                                                 ("v7", "deep_3x16")}


def test_duplicate_label_arch_pair_is_refused(tmp_path):
    a = _write_run(tmp_path, "g1", {
        "medium": _cert("medium", [0.2], verdict="PASS")})
    b = _write_run(tmp_path, "g1b", {
        "medium": _cert("medium", [0.3], verdict="PASS")})
    with pytest.raises(ValueError, match="duplicate certificate"):
        pcs.collect_certificates([("v7", a), ("v7", b)])


def test_a_run_dir_without_certificates_is_refused(tmp_path):
    empty = tmp_path / "empty_run"
    (empty / "pretrain").mkdir(parents=True)
    with pytest.raises(ValueError, match=str(empty)):
        pcs.collect_certificates([("v7", str(empty))])


def test_malformed_runs_argument_is_refused():
    with pytest.raises(ValueError, match="LABEL=RUN_DIR"):
        pcs._parse_runs(["no-equals-sign"])


def test_clipped_outliers_still_render(two_label_tree, tmp_path):
    """A max far above the cap draws at the axis edge with its number; the
    figure still writes (the legacy medium max of 4.0 sits under the default
    cap, so raise one to force the clip path)."""
    v7, _ = two_label_tree
    big = _write_run(tmp_path, "run_big", {
        "shallow": _cert("shallow", [11.3, 0.5], verdict="FAIL",
                         two_tier=False)})
    out = tmp_path / "clip.png"
    rc = pcs.main(["--runs", f"v7={v7}", "--runs", f"legacy={big}",
                   "--out", str(out)])
    assert rc == 0 and out.stat().st_size > 0


def test_render_manifest_pins_every_drawn_behavior(two_label_tree, tmp_path):
    """Arithmetic-only pins let every graphical behaviour be deleted
    silently (a build without gate lines, hatching or the clip branch still
    writes a non-empty PNG); the manifest, built at the draw sites, pins
    them: gate lines with the RIGHT rule text, FAIL hatching, the clipped
    marker and its value, distinct label colors, species notes with
    truncation, and a cap that covers every bar."""
    v7, _ = two_label_tree
    big = _write_run(tmp_path, "run_big", {
        "shallow": _cert("shallow", [11.3, 0.5], verdict="FAIL",
                         two_tier=False),
        "medium": _cert("medium", [0.6, 0.7, 1.1, 1.2, 1.3, 1.4],
                        verdict="FAIL", two_tier=False,
                        names=["s1", "s2", "s3", "s4", "s5", "s6"]),
    })
    records = pcs.collect_certificates([("v7", v7), ("legacy", big)])
    manifest = pcs.plot_certificate_summary(records,
                                            str(tmp_path / "m.png"))
    # Gate lines exist, and the merged caption names BOTH rules at the
    # shared tol_AE (mixed schemas: v7 certs record mae, legacy max).
    tol_lines = {v: text for v, text in manifest["gate_lines"]}
    assert 1.0 in tol_lines and 2.0 in tol_lines
    assert "mae: gates the set mean" in tol_lines[1.0]
    assert "max: gates every species" in tol_lines[1.0]
    assert sum(1 for v, _t in manifest["gate_lines"] if v == 1.0) == 1
    assert "per-species ceiling" in tol_lines[2.0]
    # FAIL bars hatched; PASS bars not.
    assert ("v7", "shallow") in manifest["hatched"]
    assert ("legacy", "shallow") in manifest["hatched"]
    assert ("v7", "medium") not in manifest["hatched"]
    # The 11.3 outlier clips with its value; the cap still covers every bar.
    assert ("legacy", "shallow", 11.3) in manifest["clipped"]
    means = [r["mean"] for _l, _a, r in records if r["mean"] is not None]
    assert manifest["y_cap"] >= max(means)
    # Distinct colors per label.
    assert len(set(manifest["colors"].values())) == 2
    # Species notes: flagged species inline, truncated at three with the
    # +N tail.
    assert "C3H8" in manifest["notes"][("v7", "shallow")]
    assert "+1" in manifest["notes"][("legacy", "medium")]


def test_a_certificate_with_no_usable_rows_is_annotated_not_dropped(
        tmp_path):
    run = _write_run(tmp_path, "run_empty_rows", {
        "medium": {"arch": "medium", "verdict": "FAIL",
                   "per_atomization": [],
                   "tolerances": {"tol_AE": 1.0, "tol_atom": 1.0,
                                  "override_reason": None},
                   "summary": {"failure_reasons": ["untested"]}}})
    records = pcs.collect_certificates([("v7", run)])
    manifest = pcs.plot_certificate_summary(records,
                                            str(tmp_path / "e.png"))
    note = manifest["notes"][("v7", "medium")]
    assert "no atomization data" in note and "FAIL" in note


def test_an_arch_directory_mismatch_is_refused(tmp_path):
    run = _write_run(tmp_path, "run_mismatch", {
        "medium": _cert("shallow", [0.2], verdict="PASS")})
    with pytest.raises(ValueError, match="mislabeled"):
        pcs.collect_certificates([("v7", run)])
