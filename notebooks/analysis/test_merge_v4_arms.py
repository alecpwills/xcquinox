"""Tests for the v4 cross-arm merged view builder."""
import os
import sys

import pytest

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
    _mk_arm(tmp_path, "dfs6311_grid3_v5", "run_20260810T193206Z", 3, "arm1")
    _mk_arm(tmp_path, "dfs6311_grid3_v4gga", "run_20260810T202813Z", 2, "arm2")
    out = tmp_path / "merged"
    report = mv.build_view(tmp_path, out)
    assert report["dfs6311_grid3_v5"] == ("run_20260810T193206Z", 3)
    assert report["dfs6311_grid3_v4gga"] == ("run_20260810T202813Z", 2)
    assert report["dfs6311_grid3_v5mgga2"] == (None, 0)      # not pulled yet
    specs = sorted(os.listdir(out / "checkpoints"))
    assert specs == [f"spec_{i:04d}" for i in range(5)]
    # Renumbered links resolve to the ORIGINAL spec dirs, in ARM_BASES
    # order (v4gga first, then the v5 arms).
    assert (out / "checkpoints/spec_0000/completion.json").read_text() == "arm2"
    assert (out / "checkpoints/spec_0002/completion.json").read_text() == "arm1"
    # Provenance breadcrumb names both arms.
    txt = (out / "MERGED_ARMS.txt").read_text()
    assert "run_20260810T193206Z" in txt and "run_20260810T202813Z" in txt
    # The merged manifest carries renumbered cells the collectors join on.
    import json
    m = json.loads((out / "manifest.json").read_text())
    assert m["n_specs"] == 5
    assert m["specs"][0]["cell"]["arch"] == "arm2_arch"
    assert m["specs"][2]["cell"]["arch"] == "arm1_arch"
    assert m["specs"][2]["arm"] == "dfs6311_grid3_v5"
    assert m["specs"][2]["arm_index"] == 0


def test_view_rebuild_is_idempotent_and_tracks_growth(tmp_path):
    _mk_arm(tmp_path, "dfs6311_grid3_v5", "run_20260810T193206Z", 2, "a")
    out = tmp_path / "merged"
    mv.build_view(tmp_path, out)
    assert len(os.listdir(out / "checkpoints")) == 2
    # A second arm lands (distinct archs; identical cells across arms are
    # refused by design); rebuild picks it up with no stale leftovers.
    _mk_arm(tmp_path, "dfs6311_grid3_v4gga", "run_20260810T202813Z", 4, "b")
    mv.build_view(tmp_path, out)
    assert len(os.listdir(out / "checkpoints")) == 6
    assert (out / "MERGED_ARMS.txt").read_text().count("\n") == 2


def test_newest_run_wins(tmp_path):
    _mk_arm(tmp_path, "dfs6311_grid3_v5", "run_20260801T000000Z", 1, "old")
    _mk_arm(tmp_path, "dfs6311_grid3_v5", "run_20260810T193206Z", 1, "new")
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
    ck = _mk_arm(tmp_path, "dfs6311_grid3_v5", "run_20260810T193206Z", 1, "arm1")
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
    ck1 = _mk_arm(tmp_path, "dfs6311_grid3_v5", "run_20260810T193206Z", 1, "a1")
    (ck1.parent / "resolved_config.yaml").write_text(
        "basis: 6-311++G(3df,2pd)\ndensity_fit: true\ngrid_level: 3\n")
    ck2 = _mk_arm(tmp_path, "dfs6311_grid3_v4gga", "run_20260810T202813Z", 1, "a2")
    (ck2.parent / "resolved_config.yaml").write_text(
        "basis: def2-svp\ndensity_fit: true\ngrid_level: 2\n")
    mv.build_view(tmp_path, tmp_path / "merged")
    assert "identity" in capsys.readouterr().out.lower()


def test_view_builds_without_config_or_cache(tmp_path):
    _mk_arm(tmp_path, "dfs6311_grid3_v5", "run_20260810T193206Z", 1)
    out = tmp_path / "merged"
    report = mv.build_view(tmp_path, out)
    assert report["dfs6311_grid3_v5"][1] == 1
    assert not (out / "resolved_config.yaml").exists()


def test_missing_manifest_warns_guard_skipped(tmp_path, capsys):
    """An arm whose newest run lacks manifest.json merges its specs with no
    arch/subset labels AND silently skips the seed-provenance guard -- that
    state must be loud, never silent."""
    run = tmp_path / "dfs6311_grid3_v5" / "runs" / "run_20260815T000000Z"
    ck = run / "checkpoints"
    (ck / "spec_0000").mkdir(parents=True)
    (ck / "spec_0000" / "completion.json").write_text("x")
    mv.build_view(tmp_path, tmp_path / "merged")
    out = capsys.readouterr().out
    assert "WARNING" in out and "seed-provenance" in out
    assert "dfs6311_grid3_v5" in out


def test_merged_manifest_carries_spec_provenance(tmp_path):
    """spec_file + sha256 + the source run name survive into the merged
    manifest: the expected-hash record plus enough addressing
    (<arm>/runs/<arm_run>/specs/<spec_file>) to verify a spec against the
    arm run without guessing which run is newest."""
    import json
    ck = _mk_arm(tmp_path, "dfs6311_grid3_v4gga", "run_20260810T202813Z", 1,
                 "arm2")
    run = ck.parent
    m = json.loads((run / "manifest.json").read_text())
    m["specs"][0]["spec_file"] = "spec_0000.spec"
    m["specs"][0]["sha256"] = "a" * 64
    (run / "manifest.json").write_text(json.dumps(m))
    out = tmp_path / "merged"
    mv.build_view(tmp_path, out)
    merged = json.loads((out / "manifest.json").read_text())
    assert merged["specs"][0]["spec_file"] == "spec_0000.spec"
    assert merged["specs"][0]["sha256"] == "a" * 64
    assert merged["specs"][0]["arm_run"] == "run_20260810T202813Z"


def test_partial_manifest_warns_unlabeled_specs(tmp_path, capsys):
    """A readable manifest that covers only part of the on-disk spec dirs is
    the same hazard as a missing one for the uncovered rest: those specs
    merge with no labels, no duplicate-cell key, and no seed validation."""
    import json
    ck = _mk_arm(tmp_path, "dfs6311_grid3_v4gga", "run_20260810T202813Z", 1,
                 "arm2")
    for i in (1, 2):
        d = ck / f"spec_{i:04d}"
        d.mkdir()
        (d / "completion.json").write_text("x")
    mv.build_view(tmp_path, tmp_path / "merged")
    out = capsys.readouterr().out
    assert "WARNING" in out and "lacks" in out
    assert "2 on-disk spec dir(s)" in out


def test_no_manifest_no_specs_is_note_not_warning(tmp_path, capsys):
    """An arm that has not materialized anything yet (no manifest, no spec
    dirs -- the state of a freshly submitted arm) gets a low-key note, not
    the WARNING, so the loud message keeps its signal."""
    run = tmp_path / "dfs6311_grid3_v5" / "runs" / "run_20260815T000000Z"
    (run / "checkpoints").mkdir(parents=True)
    _mk_arm(tmp_path, "dfs6311_grid3_v4gga", "run_20260810T202813Z", 1, "a2")
    mv.build_view(tmp_path, tmp_path / "merged")
    out = capsys.readouterr().out
    assert "no manifest yet" in out
    assert not any("WARNING" in ln and "dfs6311_grid3_v5" in ln
                   for ln in out.splitlines())


def test_merged_arms_txt_counts_eval_coverage(tmp_path):
    """A spec-dir count is not an eval-cell count (empty and mid-training
    dirs inflate it); the breadcrumb carries both."""
    ck = _mk_arm(tmp_path, "dfs6311_grid3_v4gga", "run_20260810T202813Z", 2,
                 "arm2")
    eh = ck / "spec_0000" / "eval_holdout"
    eh.mkdir()
    (eh / "per_molecule.json").write_text("[]")
    mv.build_view(tmp_path, tmp_path / "merged")
    txt = (tmp_path / "merged" / "MERGED_ARMS.txt").read_text()
    assert "2 specs" in txt and "1 eval_holdout" in txt


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


# --------------------------------------------------------------------------- #
# v5 era: arm roster, seed-policy validation, duplicate-cell refusal
# --------------------------------------------------------------------------- #
def _full_config_yaml(seed_xc=None):
    """A minimal-but-loadable grid config (all required sections)."""
    import yaml
    cfg = {
        "sweep": {"arch": ["deep_mgga_3x16"], "loss": ["l2"],
                  "metric": ["jsd"], "subset_size": [1],
                  "solver": ["full_3"]},
        "solvers": {"full_3": {"mode": "FULL", "max_cycles": 3}},
        "hyperparams": {"n_steps": 200, "lr_start": 1e-3, "lr_end": 1e-5,
                        "lr_decay_start": 0.5, "grad_clip": 1.0,
                        "gradnorm_alpha": 1.5, "vxc_weight": 1.0,
                        "density_weight": 1.0},
        "inputs": {"external_refs_dir": "/refs",
                   "subset_ledger_path": "/ledger.json",
                   "basis": "6-311++G(3df,2pd)", "grid_level": 3,
                   "output_root": "/out", "density_fit": True},
        "pretrain": {"data_dir": "/pre"},
        "cluster": {"partition": "p", "time": "01:00:00", "mem": "8G",
                    "cpus_per_task": 1, "array_throttle": 1,
                    "eval_array_throttle": 1, "max_concurrent_tasks": 1},
        "domain_profile": "dfs_step7",
    }
    if seed_xc is not None:
        cfg["inputs"]["seed_xc"] = seed_xc
    return yaml.safe_dump(cfg)


def _mk_registry_arm(root, base, run_name, arch, seed_xc=None,
                     subset_sizes=(1,)):
    import json
    run = root / base / "runs" / run_name
    ck = run / "checkpoints"
    ck.mkdir(parents=True)
    for i, ss in enumerate(subset_sizes):
        d = ck / f"spec_{i:04d}"
        d.mkdir()
        (d / "completion.json").write_text(arch)
    (run / "manifest.json").write_text(json.dumps({
        "n_specs": len(subset_sizes),
        "specs": [{"index": i, "cell": {"arch": arch, "subset_size": ss}}
                  for i, ss in enumerate(subset_sizes)]}))
    (run / "resolved_config.yaml").write_text(_full_config_yaml(seed_xc))
    return run


def test_arm_roster_is_v4gga_plus_v5():
    """The retired v4 mgga arms are OUT of the merged view; the roster is
    the still-valid GGA/rung-3.5 arm plus the two SCAN-seeded v5 arms."""
    assert mv.ARM_BASES == ("dfs6311_grid3_v4gga", "dfs6311_grid3_v5",
                            "dfs6311_grid3_v5mgga2")


def test_view_refuses_mis_seeded_registry_arch(tmp_path):
    """A PBE-seeded mgga arm (the retired v4 protocol: no seed_xc in the
    config resolves 'pbe', but the rung-baseline policy demands 'scan')
    must be REFUSED, not silently merged into the grouped figures."""
    _mk_registry_arm(tmp_path, "dfs6311_grid3_v5", "run_20260814T000000Z",
                     "deep_mgga_3x16", seed_xc=None)
    with pytest.raises(SystemExit, match="seed"):
        mv.build_view(tmp_path, tmp_path / "merged")


def test_view_accepts_policy_consistent_seeds(tmp_path):
    """seed_xc: auto resolves the rung baseline for every arch -- a
    correctly seeded v5 mgga arm passes validation."""
    _mk_registry_arm(tmp_path, "dfs6311_grid3_v5", "run_20260814T000000Z",
                     "deep_mgga_3x16", seed_xc="auto")
    report = mv.build_view(tmp_path, tmp_path / "merged")
    assert report["dfs6311_grid3_v5"] == ("run_20260814T000000Z", 1)


def test_view_refuses_duplicate_cells_across_arms(tmp_path):
    """The same (arch, subset_size) cell arriving from two arms is a
    double-count, never a merge."""
    _mk_registry_arm(tmp_path, "dfs6311_grid3_v5", "run_20260814T000000Z",
                     "deep_mgga_3x16", seed_xc="auto", subset_sizes=(1, 2))
    _mk_registry_arm(tmp_path, "dfs6311_grid3_v5mgga2",
                     "run_20260814T000001Z",
                     "deep_mgga_3x16", seed_xc="auto", subset_sizes=(2,))
    with pytest.raises(SystemExit, match="duplicate"):
        mv.build_view(tmp_path, tmp_path / "merged")


def _mark_sliced(ck, spec="spec_0000", chan="eval_holdout",
                 names=("h", "h2", "o", "oh", "n2o", "n2ohts")):
    """Write the pre-eval slice marker into one arm spec's channel."""
    import json
    d = ck / spec / chan
    d.mkdir(parents=True, exist_ok=True)
    (d / "sliced_eval.json").write_text(json.dumps(
        {"species_slice": list(names), "n_species": len(names),
         "n_reactions": 1,
         "env_var": "XCQUINOX_HELDOUT_SPECIES_SLICE"}))
    return d


def test_merged_view_refuses_a_sliced_arm_channel(tmp_path):
    """A workflow-verification slice must never be merged into a cross-arm
    view: every figure built on the view would average its handful of species
    into a cell as though it were the full pool."""
    from xcquinox.alec.eval_holdout import SlicedChannelError
    ck = _mk_arm(tmp_path, "dfs6311_grid3_v4gga", "run_20260810T202813Z", 2,
                 "arm2")
    _mark_sliced(ck)
    with pytest.raises(SlicedChannelError) as exc:
        mv.build_view(tmp_path, tmp_path / "merged")
    msg = str(exc.value)
    assert "run_20260810T202813Z" in msg
    assert "spec_0000" in msg
    assert "eval_holdout" in msg
    assert "'n2ohts'" in msg


def test_merged_view_refuses_a_sliced_non_default_channel(tmp_path):
    """build_view symlinks the WHOLE spec dir, so every held-out channel it
    carries enters the view -- not just the final-step one. A slice in the
    validation-best channel is as unusable there as one in eval_holdout."""
    from xcquinox.alec.eval_holdout import SlicedChannelError
    ck = _mk_arm(tmp_path, "dfs6311_grid3_v4gga", "run_20260810T202813Z", 2,
                 "arm2")
    _mark_sliced(ck, chan="eval_holdout_val_best")
    with pytest.raises(SlicedChannelError) as exc:
        mv.build_view(tmp_path, tmp_path / "merged")
    msg = str(exc.value)
    assert "run_20260810T202813Z" in msg
    assert "spec_0000" in msg
    assert "eval_holdout_val_best" in msg
    assert "'n2ohts'" in msg


def test_merged_view_refuses_a_sliced_coldstart_channel(tmp_path):
    """Same for the cold-start channel: the guard covers every eval_holdout*
    directory the spec dir carries, not a fixed pair."""
    from xcquinox.alec.eval_holdout import SlicedChannelError
    ck = _mk_arm(tmp_path, "dfs6311_grid3_v4gga", "run_20260810T202813Z", 2,
                 "arm2")
    _mark_sliced(ck, spec="spec_0001", chan="eval_holdout_coldstart")
    with pytest.raises(SlicedChannelError) as exc:
        mv.build_view(tmp_path, tmp_path / "merged")
    assert "eval_holdout_coldstart" in str(exc.value)
    assert "spec_0001" in str(exc.value)


def test_merged_view_still_builds_with_unmarked_channels(tmp_path):
    """The widened guard is a no-op on unmarked channels: a spec carrying
    every channel with no mark merges exactly as before."""
    ck = _mk_arm(tmp_path, "dfs6311_grid3_v4gga", "run_20260810T202813Z", 2,
                 "arm2")
    for ch in ("eval_holdout", "eval_holdout_best", "eval_holdout_val_best",
               "eval_holdout_coldstart"):
        d = ck / "spec_0000" / ch
        d.mkdir(parents=True)
        (d / "per_reaction.json").write_text("[]")
    out = tmp_path / "merged"
    report = mv.build_view(tmp_path, out)
    assert report["dfs6311_grid3_v4gga"] == ("run_20260810T202813Z", 2)
    assert (out / "checkpoints/spec_0000/eval_holdout/per_reaction.json"
            ).is_file()


# ---------------------------------------------------------------------------
# The guard's import must stay lazy in the analysis scripts
# ---------------------------------------------------------------------------

#: Analysis scripts that call ``eval_holdout.assert_channel_not_sliced``.
#: None of them imported the training package before the guard landed, and
#: importing it drags jax / pyscf / equinox in (measured: ~1 s and ~1575
#: extra modules), which a merge or a plot has no use for. The import
#: therefore lives at the guard, inside the function. Covered here in one
#: place because the property and its fix are identical across the six.
_GUARDED_ANALYSIS_SCRIPTS = (
    "merge_v4_arms",
    "verify_holdout_parity",
    "regen_dfs_step7_basis_comparison",
    "density_diagnosis_evidence",
    "mgga_diagnosis_evidence",
    "plot_scf_convergence",
)


@pytest.mark.parametrize("name", _GUARDED_ANALYSIS_SCRIPTS)
def test_guard_import_stays_lazy_in_the_analysis_scripts(name):
    import subprocess
    import sys
    import textwrap

    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        f"{name}.py")
    code = textwrap.dedent(f"""
        import importlib.util, sys
        spec = importlib.util.spec_from_file_location("_probe", {path!r})
        mod = importlib.util.module_from_spec(spec)
        sys.modules["_probe"] = mod
        spec.loader.exec_module(mod)
        print(",".join(sorted(
            n for n in ("xcquinox", "jax", "pyscf", "equinox")
            if n in sys.modules)))
    """)
    env = dict(os.environ, MPLBACKEND="Agg", JAX_PLATFORMS="cpu")
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True,
                          text=True, env=env, timeout=180)
    assert proc.returncode == 0, proc.stderr
    loaded = [t for t in proc.stdout.strip().split(",") if t]
    assert loaded == [], (
        f"{name}.py imports {loaded} at module scope; the guard's import "
        "belongs inside the function that calls it")
