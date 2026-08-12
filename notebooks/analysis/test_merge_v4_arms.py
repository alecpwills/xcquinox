"""Tests for the v4 cross-arm merged view builder."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import merge_v4_arms as mv


def _mk_arm(root, base, run_name, n_specs, payload="x"):
    import json
    run = root / base / "runs" / run_name
    ck = run / "checkpoints"
    ck.mkdir(parents=True)
    for i in range(n_specs):
        d = ck / f"spec_{i:04d}"
        d.mkdir()
        (d / "completion.json").write_text(payload)
    (run / "manifest.json").write_text(json.dumps({
        "n_specs": n_specs,
        "specs": [{"index": i,
                   "cell": {"arch": f"{payload}_arch", "subset_size": i + 1}}
                  for i in range(n_specs)]}))
    return ck


def test_merged_view_renumbers_across_arms(tmp_path):
    _mk_arm(tmp_path, "dfs6311_grid3_v4", "run_20260810T193206Z", 3, "arm1")
    _mk_arm(tmp_path, "dfs6311_grid3_v4gga", "run_20260810T202813Z", 2, "arm2")
    out = tmp_path / "merged"
    report = mv.build_view(tmp_path, out)
    assert report["dfs6311_grid3_v4"] == ("run_20260810T193206Z", 3)
    assert report["dfs6311_grid3_v4gga"] == ("run_20260810T202813Z", 2)
    assert report["dfs6311_grid3_v4mgga2"] == (None, 0)      # not pulled yet
    specs = sorted(os.listdir(out / "checkpoints"))
    assert specs == [f"spec_{i:04d}" for i in range(5)]
    # Renumbered links resolve to the ORIGINAL spec dirs, arm 1 first.
    assert (out / "checkpoints/spec_0000/completion.json").read_text() == "arm1"
    assert (out / "checkpoints/spec_0003/completion.json").read_text() == "arm2"
    # Provenance breadcrumb names both arms.
    txt = (out / "MERGED_ARMS.txt").read_text()
    assert "run_20260810T193206Z" in txt and "run_20260810T202813Z" in txt
    # The merged manifest carries renumbered cells the collectors join on.
    import json
    m = json.loads((out / "manifest.json").read_text())
    assert m["n_specs"] == 5
    assert m["specs"][0]["cell"]["arch"] == "arm1_arch"
    assert m["specs"][3]["cell"]["arch"] == "arm2_arch"
    assert m["specs"][3]["arm"] == "dfs6311_grid3_v4gga"
    assert m["specs"][3]["arm_index"] == 0


def test_view_rebuild_is_idempotent_and_tracks_growth(tmp_path):
    _mk_arm(tmp_path, "dfs6311_grid3_v4", "run_20260810T193206Z", 2)
    out = tmp_path / "merged"
    mv.build_view(tmp_path, out)
    assert len(os.listdir(out / "checkpoints")) == 2
    # A second arm lands; rebuild picks it up with no stale leftovers.
    _mk_arm(tmp_path, "dfs6311_grid3_v4gga", "run_20260810T202813Z", 4)
    mv.build_view(tmp_path, out)
    assert len(os.listdir(out / "checkpoints")) == 6
    assert (out / "MERGED_ARMS.txt").read_text().count("\n") == 2


def test_newest_run_wins(tmp_path):
    _mk_arm(tmp_path, "dfs6311_grid3_v4", "run_20260801T000000Z", 1, "old")
    _mk_arm(tmp_path, "dfs6311_grid3_v4", "run_20260810T193206Z", 1, "new")
    out = tmp_path / "merged"
    mv.build_view(tmp_path, out)
    assert (out / "checkpoints/spec_0000/completion.json").read_text() == "new"


def test_wrapper_suite_call_uses_full_arm_basis_token():
    """The per-arm suite call must pass the FULL arm name as its --bases token.

    ``_newest_run_per_basis`` joins ``<results_root>/<domain>/<basis>/runs``
    literally, and the wrapper's own pull target is ``$RESULTS_ROOT/$arm``
    (full name), so a stripped token (``v4``) can never resolve -- and the
    resulting FileNotFoundError is masked by the ``|| echo WARNING`` guard,
    leaving every per-arm suite silently unrendered.
    """
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "pull_and_plot_v4.sh")
    with open(script) as fh:
        text = fh.read()
    suite_lines = [ln for ln in text.splitlines() if "--bases" in ln]
    assert len(suite_lines) == 1, suite_lines
    assert '--bases "$arm"' in suite_lines[0], suite_lines[0]


def test_view_propagates_config_and_scan_cache(tmp_path):
    ck = _mk_arm(tmp_path, "dfs6311_grid3_v4", "run_20260810T193206Z", 1, "arm1")
    run = ck.parent
    (run / "resolved_config.yaml").write_text(
        "basis: 6-311++G(3df,2pd)\ndensity_fit: true\n")
    (run / "scan_pool_energies_X.json").write_text('{"H2": -1.0}')
    _mk_arm(tmp_path, "dfs6311_grid3_v4gga", "run_20260810T202813Z", 1, "arm2")
    out = tmp_path / "merged"
    mv.build_view(tmp_path, out)
    # basis-label + SCAN-cache resolution both key off the view-dir root
    assert (out / "resolved_config.yaml").read_text().startswith("basis:")
    assert (out / "scan_pool_energies_X.json").read_text() == '{"H2": -1.0}'


def test_view_warns_on_arm_identity_mismatch(tmp_path, capsys):
    ck1 = _mk_arm(tmp_path, "dfs6311_grid3_v4", "run_20260810T193206Z", 1, "a1")
    (ck1.parent / "resolved_config.yaml").write_text(
        "basis: 6-311++G(3df,2pd)\ndensity_fit: true\ngrid_level: 3\n")
    ck2 = _mk_arm(tmp_path, "dfs6311_grid3_v4gga", "run_20260810T202813Z", 1, "a2")
    (ck2.parent / "resolved_config.yaml").write_text(
        "basis: def2-svp\ndensity_fit: true\ngrid_level: 2\n")
    mv.build_view(tmp_path, tmp_path / "merged")
    assert "identity" in capsys.readouterr().out.lower()


def test_view_builds_without_config_or_cache(tmp_path):
    _mk_arm(tmp_path, "dfs6311_grid3_v4", "run_20260810T193206Z", 1)
    out = tmp_path / "merged"
    report = mv.build_view(tmp_path, out)
    assert report["dfs6311_grid3_v4"][1] == 1
    assert not (out / "resolved_config.yaml").exists()


def test_wrapper_merged_step_full_families_and_cache_copy():
    """The merged view must render the FULL figure families (incl. the
    SCAN-line set) in final AND val-best variants, and the wrapper must seed
    each arm's newest run dir with the local SCAN caches so the reference
    lines resolve."""
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "pull_and_plot_v4.sh")
    with open(script) as fh:
        text = fh.read()
    assert "build_density_energy_figures" in text
    assert "eval_holdout_val_best" in text
    assert "figures_dfs6311_v4_merged_val_best" in text
    assert "scan_pool_" in text  # cache seeding step
