"""Tests for the v4 cross-arm merged view builder."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import merge_v4_arms as mv


def _mk_arm(root, base, run_name, n_specs, payload="x", arch=None,
            certified=True, verdict="PASS", identity=None, cert_text=None,
            cert_extra=None):
    """An arm on disk. ``arch`` names the manifest architecture (default: a
    non-registry fixture name); ``certified`` writes a pretraining-fidelity
    certificate for it under ``<run>/pretrain/<arch>``.

    A named ``arch`` also gets a resolved_config.yaml: the seed-provenance
    guard refuses a REGISTRY arch whose config cannot be loaded, and that
    refusal would pre-empt the certificate guard under test. ``seed_xc: auto``
    resolves each arch's own rung baseline, so the seed guard passes and the
    certificate is what decides the arm's fate.
    """
    import json
    run = root / base / "runs" / run_name
    ck = run / "checkpoints"
    ck.mkdir(parents=True)
    arch_name = arch or f"{payload}_arch"
    for i in range(n_specs):
        d = ck / f"spec_{i:04d}"
        d.mkdir()
        (d / "completion.json").write_text(payload)
    (run / "manifest.json").write_text(json.dumps({
        "n_specs": n_specs,
        "specs": [{"index": i,
                   "cell": {"arch": arch_name, "subset_size": i + 1}}
                  for i in range(n_specs)]}))
    if arch is not None:
        (run / "resolved_config.yaml").write_text(_full_config_yaml("auto"))
    if certified or cert_text is not None:
        d = run / "pretrain" / arch_name
        d.mkdir(parents=True)
        payload_json = {"verdict": verdict, "arch": arch_name,
                        "summary": {"max_atom_mHa": 0.1,
                                    "max_dAE_kcalmol": 0.2}}
        if identity is not None:
            payload_json["identity"] = identity
        if cert_extra:
            payload_json.update(cert_extra)
        (d / "fidelity_certificate.json").write_text(
            cert_text if cert_text is not None else json.dumps(payload_json))
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


def test_missing_manifest_refuses_the_arm(tmp_path):
    """An arm whose newest run holds spec dirs but no usable manifest entries
    (missing, unreadable, or empty manifest.json -- the state of a mid-rsync
    pull) has no architecture names, so neither the seed-provenance nor the
    fidelity gate can be applied to a single one of its specs. Merging them
    unlabelled would put ungated cells in a view whose own record asserts
    universal certificate coverage."""
    run = tmp_path / "dfs6311_grid3_v5" / "runs" / "run_20260815T000000Z"
    ck = run / "checkpoints"
    (ck / "spec_0000").mkdir(parents=True)
    (ck / "spec_0000" / "completion.json").write_text("x")
    with pytest.raises(SystemExit) as exc:
        mv.build_view(tmp_path, tmp_path / "merged")
    msg = str(exc.value)
    assert "dfs6311_grid3_v5" in msg and "run_20260815T000000Z" in msg
    assert "1 spec dir" in msg
    assert "manifest" in msg


def test_unreadable_manifest_refuses_the_arm(tmp_path):
    """Same state reached through a truncated manifest rather than a missing
    one: an arm mid-transfer must not merge ungated."""
    ck = _mk_arm(tmp_path, "dfs6311_grid3_v5", "run_20260815T000000Z", 1, "a")
    (ck.parent / "manifest.json").write_text('{"specs": [{"index"')
    with pytest.raises(SystemExit) as exc:
        mv.build_view(tmp_path, tmp_path / "merged")
    assert "manifest" in str(exc.value)


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
        # The orientation lock is STATED rather than inherited: the harness
        # default is the calibrated 3e-5, while the synthetic certificates
        # below record an unlocked identity, and the identity check compares
        # the two. Stating it keeps the fixtures self-consistent, so a test
        # that means to mismatch on the basis mismatches on the basis alone.
        "inputs": {"external_refs_dir": "/refs",
                   "subset_ledger_path": "/ledger.json",
                   "basis": "6-311++G(3df,2pd)", "grid_level": 3,
                   "output_root": "/out", "density_fit": True,
                   "orientation_lock_strength": 0.0},
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
                     subset_sizes=(1,), certified=True):
    """An arm carrying a REGISTRY architecture.

    ``certified`` writes the arch's PASS fidelity certificate. It defaults on
    because an uncertified registry arch is refused before the seed-policy and
    duplicate-cell behaviour under test can be reached.
    """
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
    if certified:
        d = run / "pretrain" / arch
        d.mkdir(parents=True)
        (d / "fidelity_certificate.json").write_text(json.dumps(
            {"verdict": "PASS", "arch": arch,
             "summary": {"max_atom_mHa": 0.1, "max_dAE_kcalmol": 0.2}}))
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
    with pytest.raises(SystemExit, match="seed") as exc:
        mv.build_view(tmp_path, tmp_path / "merged")
    # The tmp_path basename carries the test name, so the bare pattern above
    # can also match the path any refusal embeds; pin the seed guard's own
    # wording, and the arch it resolved for.
    msg = str(exc.value)
    assert "rung-baseline policy demands" in msg
    assert "'scan'" in msg and "'pbe'" in msg


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
    with pytest.raises(SystemExit, match="duplicate") as exc:
        mv.build_view(tmp_path, tmp_path / "merged")
    assert "double-count" in str(exc.value)


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


# --------------------------------------------------------------------------- #
# Pretraining-fidelity certificates: an uncertified arm never enters the view
# --------------------------------------------------------------------------- #
def _cert_path(root, base, run_name, arch):
    return (root / base / "runs" / run_name / "pretrain" / arch
            / "fidelity_certificate.json")


def test_merge_refuses_a_registry_arch_with_no_certificate(tmp_path):
    """A registry architecture that was never certified cannot enter the
    grouped figures: its pretrained networks may be arbitrarily far from the
    parent functional every number on the figure is compared against."""
    _mk_arm(tmp_path, "dfs6311_grid3_v5", "run_20260810T193206Z", 1,
            "a", arch="deep_3x16", certified=False)
    with pytest.raises(SystemExit, match="fidelity") as exc:
        mv.build_view(tmp_path, tmp_path / "merged")
    assert "-- MISSING at" in str(exc.value)


def test_merge_refuses_a_registry_arch_whose_certificate_failed(tmp_path):
    _mk_arm(tmp_path, "dfs6311_grid3_v5", "run_20260810T193206Z", 1,
            "a", arch="deep_3x16", certified=True, verdict="FAIL")
    with pytest.raises(SystemExit, match="fidelity") as exc:
        mv.build_view(tmp_path, tmp_path / "merged")
    assert "has no PASS pretraining-fidelity certificate -- FAIL" in str(
        exc.value)


def test_merge_refuses_an_unenforced_failure(tmp_path):
    """``fidelity.enforce: false`` releases the ON-NODE gates only; the merge
    is a record layer and refuses a FAIL regardless."""
    import json
    _mk_arm(tmp_path, "dfs6311_grid3_v5", "run_20260810T193206Z", 1,
            "a", arch="deep_3x16", certified=True, verdict="FAIL")
    cert = _cert_path(tmp_path, "dfs6311_grid3_v5", "run_20260810T193206Z",
                      "deep_3x16")
    payload = json.loads(cert.read_text())
    payload["enforced"] = False
    payload["tolerances"] = {"override_reason": "workflow verification"}
    cert.write_text(json.dumps(payload))
    # The same certificate releases the on-node gate it was written for, so
    # the refusal below is the record layer's own rule and not a side effect
    # of an incomplete waiver.
    from xcquinox.alec.cluster.fidelity import gate_certificate
    run = tmp_path / "dfs6311_grid3_v5" / "runs" / "run_20260810T193206Z"
    allowed, _msg = gate_certificate(str(run), "deep_3x16")
    assert allowed is True
    with pytest.raises(SystemExit, match="fidelity") as exc:
        mv.build_view(tmp_path, tmp_path / "merged")
    # Named as the waiver it is, in the figure layer's vocabulary: a run that
    # was never meant to certify is a different thing from an architecture
    # whose physics did not, and "FAIL" alone states neither.
    assert "has no PASS pretraining-fidelity certificate -- waived FAIL" in \
        str(exc.value)


def test_the_waived_label_matches_the_figure_layers(tmp_path):
    """One vocabulary for the four states across the two record layers.

    ``make_ablation_arch_figure`` names MISSING, UNREADABLE, FAIL and waived
    FAIL; a reader who meets the same run in a merge refusal and on a figure
    footer must not be told two different things about it.
    """
    import json
    import sys as _sys
    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import make_ablation_arch_figure as _fig
    waived = {"verdict": "FAIL", "arch": "deep_3x16", "enforced": False,
              "tolerances": {"override_reason": "workflow verification"},
              "summary": {"max_atom_mHa": 13.7, "max_dAE_kcalmol": 25.7}}
    for status, payload in (("MISSING", None),
                            ("UNREADABLE", {"arch": "deep_3x16"}),
                            ("FAIL", {"verdict": "FAIL"}),
                            ("FAIL", waived)):
        assert (mv._certificate_status_label(status, payload)
                == _fig._certificate_status_label(status, payload))
    assert mv._certificate_status_label("FAIL", waived) == "waived FAIL"
    assert json.loads(json.dumps(waived))["enforced"] is False


# --------------------------------------------------------------------------- #
# One document per decision
# --------------------------------------------------------------------------- #
def _serve_after_the_first_read(monkeypatch, document):
    """Serve ``document`` to every certificate READ after the first.

    The list returned collects one entry per read, so a caller can state how
    many parses its decision rested on; writes and every other path are
    passed through.
    """
    import builtins
    import io as _io
    import json
    real_open = builtins.open
    reads: list = []

    def fake_open(file, *args, **kwargs):
        path = str(file)
        mode = kwargs.get("mode", args[0] if args else "r")
        if path.endswith("fidelity_certificate.json") and "r" in mode:
            reads.append(path)
            if len(reads) > 1:
                return _io.StringIO(json.dumps(document))
        return real_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", fake_open)
    return reads


def test_each_arm_certificate_is_read_once(tmp_path, monkeypatch):
    """One parse per certificate, so no refusal can mix two documents.

    The guard classified the file and then opened it again for the records it
    re-checks, so a certificate rewritten between the two opens was judged on
    one document and reported on another.
    """
    _mk_arm(tmp_path, "dfs6311_grid3_v5", "run_20260810T193206Z", 1,
            "a", arch="deep_3x16", certified=True)
    run = tmp_path / "dfs6311_grid3_v5" / "runs" / "run_20260810T193206Z"
    reads = _serve_after_the_first_read(monkeypatch, {"verdict": "PASS"})
    mv._validate_arm_fidelity_certificates(run, {"deep_3x16": [0]}, arm="v5")
    monkeypatch.undo()
    assert len(reads) == 1, reads


def test_a_second_document_cannot_refuse_a_certified_arm(tmp_path,
                                                         monkeypatch):
    """A refusal describes the document that was classified, or no document.

    Here the certificate on disk is a self-consistent PASS and every read
    after the first would find one naming another architecture. A guard that
    classifies one parse and re-reads for the record checks refuses the arm
    over a file that never existed as a whole.
    """
    _mk_arm(tmp_path, "dfs6311_grid3_v5", "run_20260810T193206Z", 1,
            "a", arch="deep_3x16", certified=True)
    run = tmp_path / "dfs6311_grid3_v5" / "runs" / "run_20260810T193206Z"
    foreign = {"verdict": "PASS", "arch": "somebody_elses_arch",
               "parent": "scan",
               "summary": {"max_atom_mHa": 0.1, "max_dAE_kcalmol": 0.2}}
    reads = _serve_after_the_first_read(monkeypatch, foreign)
    mv._validate_arm_fidelity_certificates(run, {"deep_3x16": [0]}, arm="v5")
    monkeypatch.undo()
    assert len(reads) == 1, reads


def test_the_guard_returns_the_statuses_it_validated(tmp_path):
    """The status recorded per spec is the one the guard acted on.

    Reading the certificates a second time to record them would let the view
    state a status the validation never saw.
    """
    _mk_arm(tmp_path, "dfs6311_grid3_v5", "run_20260810T193206Z", 1,
            "a", arch="deep_3x16", certified=True)
    run = tmp_path / "dfs6311_grid3_v5" / "runs" / "run_20260810T193206Z"
    statuses = mv._validate_arm_fidelity_certificates(
        run, {"deep_3x16": [0]}, arm="v5")
    assert {a: st for a, (st, *_rest) in statuses.items()} == {
        "deep_3x16": "PASS"}


def test_merge_refuses_an_unreadable_certificate(tmp_path):
    """A certificate that records no verdict this module recognises states no
    outcome that can be acted on, and an unverifiable arm is refused."""
    _mk_arm(tmp_path, "dfs6311_grid3_v5", "run_20260810T193206Z", 1,
            "a", arch="deep_3x16", cert_text='{"arch": "deep_3x16"}')
    with pytest.raises(SystemExit, match="fidelity") as exc:
        mv.build_view(tmp_path, tmp_path / "merged")
    assert "-- UNREADABLE at" in str(exc.value)


def test_merge_refuses_a_certificate_from_another_run_identity(tmp_path):
    """A PASS measured at a different basis/grid/Coulomb identity does not
    certify THIS arm's networks: the certificate's energies were computed on
    another SCF than the one the arm's held-out numbers come from."""
    _mk_arm(tmp_path, "dfs6311_grid3_v5", "run_20260810T193206Z", 1,
            "a", arch="deep_3x16",
            identity={"basis": "def2-svp", "grid_level": 1,
                      "density_fit": True, "auxbasis": None,
                      "orientation_lock_strength": 0.0})
    with pytest.raises(SystemExit, match="identity") as exc:
        mv.build_view(tmp_path, tmp_path / "merged")
    msg = str(exc.value)
    assert "fidelity" in msg
    assert "basis" in msg and "def2-svp" in msg


def test_merge_accepts_a_certificate_matching_the_run_identity(tmp_path):
    """The identity check compares the recorded identity against the arm's
    own resolved_config.yaml, so a certificate measured at the arm's identity
    passes it."""
    _mk_arm(tmp_path, "dfs6311_grid3_v5", "run_20260810T193206Z", 1,
            "a", arch="deep_3x16",
            identity={"basis": "6-311++G(3df,2pd)", "grid_level": 3,
                      "density_fit": True, "auxbasis": None,
                      "orientation_lock_strength": 0.0})
    report = mv.build_view(tmp_path, tmp_path / "merged")
    assert report["dfs6311_grid3_v5"] == ("run_20260810T193206Z", 1)


def test_merge_refusal_names_arm_spec_arch_certificate_and_status(tmp_path):
    """The refusal has to be actionable without a second search: which arm,
    which spec dirs, which architecture, which file, and what it said."""
    _mk_arm(tmp_path, "dfs6311_grid3_v5", "run_20260810T193206Z", 2,
            "a", arch="deep_3x16", certified=False)
    with pytest.raises(SystemExit) as exc:
        mv.build_view(tmp_path, tmp_path / "merged")
    msg = str(exc.value)
    assert "dfs6311_grid3_v5" in msg
    assert "run_20260810T193206Z" in msg
    assert "deep_3x16" in msg
    assert "MISSING" in msg
    assert str(_cert_path(tmp_path, "dfs6311_grid3_v5",
                          "run_20260810T193206Z", "deep_3x16")) in msg
    assert "spec_0000" in msg and "spec_0001" in msg


def test_merge_accepts_a_certified_registry_arch(tmp_path):
    _mk_arm(tmp_path, "dfs6311_grid3_v5", "run_20260810T193206Z", 1,
            "a", arch="deep_3x16", certified=True)
    report = mv.build_view(tmp_path, tmp_path / "merged")
    assert report["dfs6311_grid3_v5"] == ("run_20260810T193206Z", 1)


def test_merge_skips_non_registry_archs(tmp_path):
    """Test fixtures and legacy display names carry no certificate
    expectation, matching the seed-policy guard."""
    _mk_arm(tmp_path, "dfs6311_grid3_v5", "run_20260810T193206Z", 1,
            "legacy", arch="not_a_registry_arch", certified=False)
    report = mv.build_view(tmp_path, tmp_path / "merged")
    assert report["dfs6311_grid3_v5"] == ("run_20260810T193206Z", 1)


def test_merged_view_carries_the_arms_pretrain_certificates(tmp_path):
    """The merged directory has no pretrain stage of its own, so the figure
    layer would read every arch as uncertified unless the arms' certificates
    travel with the merge."""
    _mk_arm(tmp_path, "dfs6311_grid3_v5", "run_20260810T193206Z", 1,
            "a", arch="deep_3x16")
    out = tmp_path / "merged"
    mv.build_view(tmp_path, out)
    cert = out / "pretrain" / "deep_3x16" / "fidelity_certificate.json"
    assert cert.is_file()
    import json
    assert json.loads(cert.read_text())["verdict"] == "PASS"


def test_merged_view_records_the_certificate_status_per_spec(tmp_path):
    """The merged manifest is the view's own record: each spec carries the
    status its architecture's certificate was admitted under, and archs with
    no certificate expectation are labelled as such rather than left blank."""
    import json
    _mk_arm(tmp_path, "dfs6311_grid3_v5", "run_20260810T193206Z", 1,
            "a", arch="deep_3x16")
    _mk_arm(tmp_path, "dfs6311_grid3_v4gga", "run_20260810T202813Z", 1,
            "legacy", arch="not_a_registry_arch", certified=False)
    out = tmp_path / "merged"
    mv.build_view(tmp_path, out)
    m = json.loads((out / "manifest.json").read_text())
    by_arch = {s["cell"]["arch"]: s["fidelity_status"] for s in m["specs"]}
    assert by_arch["deep_3x16"] == "PASS"
    assert by_arch["not_a_registry_arch"] == "NOT_IN_REGISTRY"
    fid = m["fidelity"]
    assert fid["by_arm"]["dfs6311_grid3_v5"] == {"deep_3x16": "PASS"}
    assert fid["by_arm"]["dfs6311_grid3_v4gga"] == {}
    # No status other than PASS can reach a built view, so the waiver count
    # is a recorded zero rather than an unstated assumption.
    assert fid["n_waived"] == 0
    txt = (out / "MERGED_ARMS.txt").read_text()
    assert "deep_3x16=PASS" in txt


def test_unlabeled_specs_record_no_certificate_status(tmp_path):
    """A spec dir with no manifest entry has no architecture to certify; the
    record says so instead of implying a PASS."""
    import json
    ck = _mk_arm(tmp_path, "dfs6311_grid3_v4gga", "run_20260810T202813Z", 1,
                 "arm2")
    d = ck / "spec_0001"
    d.mkdir()
    (d / "completion.json").write_text("x")
    out = tmp_path / "merged"
    mv.build_view(tmp_path, out)
    m = json.loads((out / "manifest.json").read_text())
    assert m["specs"][1]["fidelity_status"] == "UNLABELED"


def test_view_warns_when_two_arms_carry_the_same_arch_certificate(tmp_path):
    """One arch name, two arms, one pretrain slot in the view: the figure
    layer will read the first arm's certificate for both, so the collision is
    named rather than resolved silently."""
    import json
    _mk_arm(tmp_path, "dfs6311_grid3_v4gga", "run_20260810T202813Z", 1,
            "a", arch="deep_3x16")
    ck = _mk_arm(tmp_path, "dfs6311_grid3_v5", "run_20260810T193206Z", 2,
                 "b", arch="deep_3x16")
    # Distinct subset sizes: the same arch in two arms is the case under
    # test, the same CELL in two arms is a double-count refused elsewhere.
    run2 = ck.parent
    m = json.loads((run2 / "manifest.json").read_text())
    for k, e in enumerate(m["specs"]):
        e["cell"]["subset_size"] = 10 + k
    (run2 / "manifest.json").write_text(json.dumps(m))
    out = tmp_path / "merged"
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        mv.build_view(tmp_path, out)
    printed = buf.getvalue()
    assert "WARNING" in printed and "deep_3x16" in printed
    assert (out / "pretrain" / "deep_3x16").resolve() == (
        tmp_path / "dfs6311_grid3_v4gga" / "runs" / "run_20260810T202813Z"
        / "pretrain" / "deep_3x16").resolve()


# --------------------------------------------------------------------------- #
# Certificate <-> checkpoint binding, multi-arm coverage, view durability
# --------------------------------------------------------------------------- #
def _renumber_subsets(run, start):
    """Move an arm's cells to distinct subset sizes (the duplicate-cell guard
    is a separate rule and would otherwise pre-empt the case under test)."""
    import json
    m = json.loads((run / "manifest.json").read_text())
    for k, e in enumerate(m["specs"]):
        e["cell"]["subset_size"] = start + k
    (run / "manifest.json").write_text(json.dumps(m))


def _add_checkpoint_digests(root, base, run_name, arch,
                            xnet=b"xnet-weights", cnet=b"cnet-weights"):
    """Write the two checkpoint files and record their digests in the arch's
    certificate, as the certificate writer does."""
    import json
    from xcquinox.alec.cluster.materialize import _sha256_file
    d = root / base / "runs" / run_name / "pretrain" / arch
    (d / "xnet.eqx").write_bytes(xnet)
    (d / "cnet.eqx").write_bytes(cnet)
    cert = d / "fidelity_certificate.json"
    payload = json.loads(cert.read_text())
    payload["checkpoint"] = {
        "dir": str(d),
        "xnet_sha256": _sha256_file(str(d / "xnet.eqx")),
        "cnet_sha256": _sha256_file(str(d / "cnet.eqx"))}
    cert.write_text(json.dumps(payload))
    return d


def test_merge_refuses_an_uncertified_arm_after_a_certified_one(tmp_path):
    """The gate is per ARM: a certified first arm must not license the rest.
    Only a check that runs on every arm can catch the arm that lands later."""
    _mk_arm(tmp_path, "dfs6311_grid3_v4gga", "run_20260810T202813Z", 1,
            "a", arch="deep_3x16")
    _mk_arm(tmp_path, "dfs6311_grid3_v5", "run_20260815T034818Z", 1,
            "b", arch="deep_mgga_3x16", certified=False)
    with pytest.raises(SystemExit) as exc:
        mv.build_view(tmp_path, tmp_path / "merged")
    msg = str(exc.value)
    assert "dfs6311_grid3_v5" in msg and "run_20260815T034818Z" in msg
    assert "deep_mgga_3x16" in msg and "-- MISSING at" in msg


def test_view_links_only_the_gated_archs_pretrain_dirs(tmp_path):
    """<run>/pretrain can hold directories no cell of the run references (an
    arch from an earlier submission, an arch whose specs were dropped). Those
    were never gated, so linking one would put an ungated -- here FAILED --
    certificate in the view under a name the figure layer reads, and would
    take the slot of the arm that really ran that arch."""
    import json
    _mk_arm(tmp_path, "dfs6311_grid3_v4gga", "run_20260810T202813Z", 1,
            "a", arch="deep_3x16")
    stale = (tmp_path / "dfs6311_grid3_v4gga" / "runs"
             / "run_20260810T202813Z" / "pretrain" / "deep_mgga_3x16")
    stale.mkdir(parents=True)
    (stale / "fidelity_certificate.json").write_text(json.dumps(
        {"verdict": "FAIL", "arch": "deep_mgga_3x16",
         "summary": {"max_atom_mHa": 99.0, "max_dAE_kcalmol": 99.0}}))
    _mk_arm(tmp_path, "dfs6311_grid3_v5", "run_20260815T034818Z", 1,
            "b", arch="deep_mgga_3x16")
    out = tmp_path / "merged"
    mv.build_view(tmp_path, out)
    linked = out / "pretrain" / "deep_mgga_3x16"
    assert linked.resolve() == (
        tmp_path / "dfs6311_grid3_v5" / "runs" / "run_20260815T034818Z"
        / "pretrain" / "deep_mgga_3x16").resolve()
    assert json.loads(
        (linked / "fidelity_certificate.json").read_text())["verdict"] == "PASS"


def test_view_refuses_the_same_arch_certified_at_two_identities(tmp_path):
    """One arch, two arms, one pretrain slot in the view. Certificates that
    agree can be represented by either; certificates measured at DIFFERENT
    SCF identities cannot, so the collision is refused rather than resolved
    by arm order."""
    import yaml
    ident_a = {"basis": "6-311++G(3df,2pd)", "grid_level": 3,
               "density_fit": True, "auxbasis": None,
               "orientation_lock_strength": 0.0}
    ident_b = dict(ident_a, basis="def2-svp")
    _mk_arm(tmp_path, "dfs6311_grid3_v4gga", "run_20260810T202813Z", 1,
            "a", arch="deep_3x16", identity=ident_a)
    ck = _mk_arm(tmp_path, "dfs6311_grid3_v5", "run_20260815T034818Z", 1,
                 "b", arch="deep_3x16", identity=ident_b)
    run_b = ck.parent
    cfg = yaml.safe_load(_full_config_yaml("auto"))
    cfg["inputs"]["basis"] = "def2-svp"
    (run_b / "resolved_config.yaml").write_text(yaml.safe_dump(cfg))
    _renumber_subsets(run_b, 10)
    with pytest.raises(SystemExit) as exc:
        mv.build_view(tmp_path, tmp_path / "merged")
    msg = str(exc.value)
    assert "deep_3x16" in msg
    assert "def2-svp" in msg
    assert "identit" in msg


def test_merge_refuses_a_certificate_naming_another_arch(tmp_path):
    """The certificate is located by DIRECTORY; a file copied from another
    arch's pretrain dir would otherwise certify this one."""
    _mk_arm(tmp_path, "dfs6311_grid3_v5", "run_20260810T193206Z", 1,
            "a", arch="deep_3x16",
            cert_extra={"arch": "deep_mgga_3x16"})
    with pytest.raises(SystemExit) as exc:
        mv.build_view(tmp_path, tmp_path / "merged")
    msg = str(exc.value)
    assert "names arch" in msg
    assert "deep_mgga_3x16" in msg and "deep_3x16" in msg


def test_merge_refuses_a_certificate_measured_against_the_wrong_parent(
        tmp_path):
    """The parent functional follows the arch's RUNG (PBE for a GGA arch,
    SCAN for a meta-GGA one). A certificate measured against the other one
    bounds nothing about the distance this arch was pretrained to close."""
    _mk_arm(tmp_path, "dfs6311_grid3_v5", "run_20260810T193206Z", 1,
            "a", arch="deep_3x16", cert_extra={"parent": "scan"})
    with pytest.raises(SystemExit) as exc:
        mv.build_view(tmp_path, tmp_path / "merged")
    msg = str(exc.value)
    assert "parent" in msg
    assert "'scan'" in msg and "'pbe'" in msg


def test_merge_refuses_a_certificate_whose_checkpoint_moved(tmp_path):
    """The verdict refers to two specific files. A checkpoint rewritten or
    re-pretrained after certification is not the one that was measured, and
    the digests are what tie the verdict to the networks on disk."""
    _mk_arm(tmp_path, "dfs6311_grid3_v5", "run_20260810T193206Z", 1,
            "a", arch="deep_3x16")
    d = _add_checkpoint_digests(tmp_path, "dfs6311_grid3_v5",
                                "run_20260810T193206Z", "deep_3x16")
    (d / "xnet.eqx").write_bytes(b"xnet-weightt")   # one byte changed
    with pytest.raises(SystemExit) as exc:
        mv.build_view(tmp_path, tmp_path / "merged")
    msg = str(exc.value)
    assert "xnet.eqx" in msg
    assert "sha256" in msg or "digest" in msg


def test_merge_refuses_a_certificate_whose_checkpoint_is_gone(tmp_path):
    """A digest recorded for a file that is not there certifies nothing."""
    _mk_arm(tmp_path, "dfs6311_grid3_v5", "run_20260810T193206Z", 1,
            "a", arch="deep_3x16")
    d = _add_checkpoint_digests(tmp_path, "dfs6311_grid3_v5",
                                "run_20260810T193206Z", "deep_3x16")
    (d / "cnet.eqx").unlink()
    with pytest.raises(SystemExit) as exc:
        mv.build_view(tmp_path, tmp_path / "merged")
    assert "cnet.eqx" in str(exc.value)


def test_merge_accepts_matching_checkpoint_digests(tmp_path):
    """The binding is a comparison, not a requirement that the field be
    absent: an untouched checkpoint pair passes."""
    _mk_arm(tmp_path, "dfs6311_grid3_v5", "run_20260810T193206Z", 1,
            "a", arch="deep_3x16", cert_extra={"parent": "pbe"})
    _add_checkpoint_digests(tmp_path, "dfs6311_grid3_v5",
                            "run_20260810T193206Z", "deep_3x16")
    report = mv.build_view(tmp_path, tmp_path / "merged")
    assert report["dfs6311_grid3_v5"] == ("run_20260810T193206Z", 1)


def test_a_refused_rebuild_leaves_the_previous_view_intact(tmp_path):
    """The view is rebuilt from scratch on every invocation, and every guard
    can refuse mid-build. Wiping the directory first would make one refusal
    destroy the last good view, so the rebuild is staged and swapped in only
    once every arm has passed."""
    import json
    _mk_arm(tmp_path, "dfs6311_grid3_v4gga", "run_20260810T202813Z", 2,
            "a", arch="deep_3x16")
    out = tmp_path / "merged"
    mv.build_view(tmp_path, out)
    specs_before = sorted(os.listdir(out / "checkpoints"))
    manifest_before = (out / "manifest.json").read_text()
    assert specs_before == ["spec_0000", "spec_0001"]
    # An uncertified arm lands and the rebuild is refused.
    _mk_arm(tmp_path, "dfs6311_grid3_v5", "run_20260815T034818Z", 1,
            "b", arch="deep_mgga_3x16", certified=False)
    with pytest.raises(SystemExit):
        mv.build_view(tmp_path, out)
    assert sorted(os.listdir(out / "checkpoints")) == specs_before
    assert (out / "manifest.json").read_text() == manifest_before
    assert (out / "checkpoints" / "spec_0000" / "completion.json"
            ).read_text() == "a"
    assert json.loads(manifest_before)["n_specs"] == 2
    # and no half-built directory is left behind
    assert not (tmp_path / "merged.building").exists()
    assert not (tmp_path / "merged.previous").exists()


def test_a_refused_first_build_leaves_no_view(tmp_path):
    """With no previous view to protect, a refused build must not leave a
    half-populated directory that a later figure run would read as complete."""
    _mk_arm(tmp_path, "dfs6311_grid3_v4gga", "run_20260810T202813Z", 1,
            "a", arch="deep_3x16")
    _mk_arm(tmp_path, "dfs6311_grid3_v5", "run_20260815T034818Z", 1,
            "b", arch="deep_mgga_3x16", certified=False)
    out = tmp_path / "merged"
    with pytest.raises(SystemExit):
        mv.build_view(tmp_path, out)
    assert not out.exists()
    assert not (tmp_path / "merged.building").exists()


def test_view_swaps_through_a_symlinked_out_dir(tmp_path):
    """The view path itself can be a SYMLINK (a view parked on another
    filesystem and reached through a link). ``shutil.rmtree`` refuses a
    symbolic link, so the swap has to unlink one: otherwise a build that
    SUCCEEDED raises after it has already swapped, and the leftover
    ``<name>.previous`` link makes every later build raise before the swap,
    accumulating ``<name>.building`` directories."""
    _mk_arm(tmp_path, "dfs6311_grid3_v4gga", "run_20260810T202813Z", 2,
            "a", arch="deep_3x16")
    target = tmp_path / "view_on_another_disk"
    target.mkdir()
    out = tmp_path / "merged"
    out.symlink_to(target)
    rc = mv.main(["--results-root", str(tmp_path), "--out", str(out)])
    assert rc == 0
    assert sorted(os.listdir(out / "checkpoints")) == ["spec_0000",
                                                       "spec_0001"]
    for litter in ("merged.previous", "merged.building"):
        assert not (tmp_path / litter).exists()
        assert not (tmp_path / litter).is_symlink()
    # The link's target is left where it was, never deleted.
    assert target.is_dir()
    # A second build over the swapped-in view still succeeds and still
    # leaves nothing behind.
    report = mv.build_view(tmp_path, out)
    assert report["dfs6311_grid3_v4gga"] == ("run_20260810T202813Z", 2)
    for litter in ("merged.previous", "merged.building"):
        assert not (tmp_path / litter).exists()
        assert not (tmp_path / litter).is_symlink()


def test_a_refused_rebuild_through_a_symlink_leaves_the_target_intact(
        tmp_path):
    """Refusing through a symlinked view path must not touch what the link
    points at, nor replace the link."""
    _mk_arm(tmp_path, "dfs6311_grid3_v4gga", "run_20260810T202813Z", 2,
            "a", arch="deep_3x16")
    target = tmp_path / "view_on_another_disk"
    mv.build_view(tmp_path, target)
    specs_before = sorted(os.listdir(target / "checkpoints"))
    manifest_before = (target / "manifest.json").read_text()
    out = tmp_path / "merged"
    out.symlink_to(target)
    _mk_arm(tmp_path, "dfs6311_grid3_v5", "run_20260815T034818Z", 1,
            "b", arch="deep_mgga_3x16", certified=False)
    with pytest.raises(SystemExit):
        mv.build_view(tmp_path, out)
    assert out.is_symlink() and out.resolve() == target.resolve()
    assert sorted(os.listdir(target / "checkpoints")) == specs_before
    assert (target / "manifest.json").read_text() == manifest_before
    assert not (tmp_path / "merged.building").exists()
    assert not (tmp_path / "merged.previous").is_symlink()
