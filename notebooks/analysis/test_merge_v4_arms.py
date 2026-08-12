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
