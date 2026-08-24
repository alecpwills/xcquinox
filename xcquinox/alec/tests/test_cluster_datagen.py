"""Tests for the datagen front stage (``xcquinox.alec.cluster._datagen``).

The datagen stage runs FIRST in the job graph and generates the pretrain-data
file(s) every swept arch needs, before the pretrain stage consumes them. These
tests pin: (1) which files are required for polarized / unpolarized / mixed
sweeps, and (2) that ``main`` calls the idempotent generator once per distinct
required file with the basis/grid/density_fit taken from the resolved config,
without running any real PBE SCF (the generator seam is monkeypatched).
"""
from __future__ import annotations

import os
import types

import pytest

from xcquinox.alec.cluster import _datagen


def _ns(**kw):
    return types.SimpleNamespace(**kw)


def _arch(**kw):
    """An architecture-LIKE double, carrying the attribute the rung is read
    from.

    ``resolve_parent_density`` resolves the pretraining parent through
    ``ArchitectureConfig.is_meta_gga``, which reads ``descriptors``; a double
    without it is not an architecture and is refused by type rather than
    answered with the GGA-rung parent. Empty means the GGA rung, which is what
    these polarization fixtures intend.
    """
    kw.setdefault("descriptors", ())
    return types.SimpleNamespace(**kw)


def _cfg(archs, polarized, *, basis="def2-svp", grid=2, df=False, aux=None,
         data_dir="/data/pt", lock=0.0):
    return _ns(
        sweep=_ns(arch=list(archs)),
        use_polarized_correlation=polarized,
        pretrain=_ns(data_dir=data_dir),
        inputs=_ns(basis=basis, grid_level=grid, density_fit=df, auxbasis=aux,
                   orientation_lock_strength=lock),
    )


# ---------------------------------------------------------------------------
# _required_polarized_flags, which files the sweep's archs consume
# ---------------------------------------------------------------------------

def test_required_flags_polarized_run():
    # Real registry archs (base, attn, cusp, dm, combined, notransform) under a
    # run-level polarized flag all resolve to the single polarized file.
    cfg = _cfg(["deep", "deep_attn", "deep_cusp", "deep_dm", "deep_combined",
                "deep_combined_attn", "deep_notransform",
                "deep_notransform_attn"], True)
    assert _datagen._required_polarized_flags(cfg) == [True]


def test_required_flags_unpolarized_run():
    cfg = _cfg(["deep", "deep_attn"], False)
    assert _datagen._required_polarized_flags(cfg) == [False]


def test_required_flags_mixed(monkeypatch):
    # Synthetic per-arch polarization with the run-level flag OFF: the worker
    # must request BOTH the unpolarized and the polarized file.
    monkeypatch.setattr(_datagen, "get_architecture",
                        lambda name: _arch(use_polarized_correlation=(name == "pol")))
    cfg = _cfg(["plain", "pol"], False)
    assert _datagen._required_polarized_flags(cfg) == [False, True]


# ---------------------------------------------------------------------------
# main: generation calls (generator seam monkeypatched)
# ---------------------------------------------------------------------------

def _run_main(monkeypatch, tmp_path, cfg):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "resolved_config.yaml").write_text("dummy: 1\n")
    monkeypatch.setattr(_datagen, "load_grid_config", lambda p: cfg)
    calls = []
    monkeypatch.setattr(
        _datagen, "_ensure_pretrain_data",
        lambda data_dir, **kw: (calls.append((data_dir, kw))
                                or f"{data_dir}/x.npz"))
    rc = _datagen.main([str(run_dir)])
    return rc, calls


def test_main_polarized_svp_covers_all_archs(monkeypatch, tmp_path):
    cfg = _cfg(
        ["deep", "deep_attn", "deep_cusp", "deep_dm", "deep_combined",
         "deep_combined_attn", "deep_notransform", "deep_notransform_attn"],
        True, basis="def2-svp", grid=2, df=False, data_dir="/d/svp")
    rc, calls = _run_main(monkeypatch, tmp_path, cfg)
    assert rc == 0
    # ONE call covers all 8 archs: one polarized file with descriptors on (so
    # cusp/dm archs get cusp_all/dm_all).
    assert len(calls) == 1
    data_dir, kw = calls[0]
    assert data_dir == "/d/svp"
    assert kw == {"basis": "def2-svp", "grid_level": 2, "density_fit": False,
                  "auxbasis": None, "polarized": True, "descriptors": True,
                  # The run's own orientation lock is part of the data's
                  # identity and is always stated (see the lock tests below).
                  "orientation_lock_strength": 0.0}


def test_main_density_fit_tzvpd(monkeypatch, tmp_path):
    cfg = _cfg(["deep", "deep_combined"], True, basis="def2-tzvpd", grid=2,
               df=True, aux="def2-universal-jkfit", data_dir="/d/tz")
    rc, calls = _run_main(monkeypatch, tmp_path, cfg)
    assert rc == 0
    assert len(calls) == 1
    assert calls[0][1]["density_fit"] is True
    assert calls[0][1]["basis"] == "def2-tzvpd"
    # GAP-4: the configured auxbasis must reach the pretrain-data generator.
    assert calls[0][1]["auxbasis"] == "def2-universal-jkfit"
    assert calls[0][1]["polarized"] is True


def test_main_mixed_generates_both_files(monkeypatch, tmp_path):
    monkeypatch.setattr(_datagen, "get_architecture",
                        lambda name: _arch(use_polarized_correlation=(name == "pol")))
    cfg = _cfg(["plain", "pol"], False, data_dir="/d/m")
    rc, calls = _run_main(monkeypatch, tmp_path, cfg)
    assert rc == 0
    assert sorted(kw["polarized"] for _, kw in calls) == [False, True]


def test_main_missing_config_returns_1(tmp_path):
    assert _datagen.main([str(tmp_path / "nope")]) == 1


def test_main_no_argv_returns_1():
    assert _datagen.main([]) == 1


def test_main_generation_failure_returns_1(monkeypatch, tmp_path):
    # A generator failure must exit non-zero so the pretrain afterok:datagen
    # dependency blocks rather than running against missing data.
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "resolved_config.yaml").write_text("dummy: 1\n")
    monkeypatch.setattr(_datagen, "load_grid_config",
                        lambda p: _cfg(["deep"], True))

    def _boom(*a, **k):
        raise RuntimeError("SCF blew up")

    monkeypatch.setattr(_datagen, "_ensure_pretrain_data", _boom)
    assert _datagen.main([str(run_dir)]) == 1


# ---------------------------------------------------------------------------
# JAX precision routing: the datagen node must compute in float64
# ---------------------------------------------------------------------------

def test_datagen_routes_jax_to_double_precision(monkeypatch):
    """The generator's descriptor, tau and alpha columns are JAX computations,
    and the parent density arrives as a jnp array. Without this the datagen node
    computes them in single precision while every test computes them in double,
    so the file the cluster writes is not the file the tests describe."""
    import os
    monkeypatch.delenv("JAX_ENABLE_X64", raising=False)
    _datagen._route_jax_env()
    assert os.environ["JAX_ENABLE_X64"] == "1"


def test_datagen_routes_before_anything_that_imports_jax():
    """The switch is only honored before the first jax import, so the routing
    call must precede every other statement of main() and the module must not
    import the generator at module scope."""
    import inspect
    src = inspect.getsource(_datagen.main)
    assert src.index("_route_jax_env()") < src.index("argv = sys.argv")
    assert src.index("_route_jax_env()") < src.index("load_grid_config")
    module_src = inspect.getsource(_datagen)
    head = module_src.split("def _route_jax_env")[0]
    assert "import pretrain_data_gen" not in head, (
        "importing the generator at module scope pulls in jax.numpy before "
        "_route_jax_env can set the precision flag")


def test_datagen_routing_switches_a_live_jax_to_double_precision():
    """``python -m xcquinox.alec.cluster._datagen`` runs the package
    initializers first, and ``import xcquinox`` already imports jax, so by the
    time this module's body runs the environment variable is read too late.
    The routing must therefore also flip the live configuration; the
    observable is the dtype a float64 host array keeps on entering JAX."""
    import numpy as np
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", False)
    try:
        assert jnp.asarray(np.ones(1)).dtype == np.float32
        _datagen._route_jax_env()
        assert bool(jax.config.jax_enable_x64) is True
        assert jnp.asarray(np.ones(1)).dtype == np.float64
    finally:
        jax.config.update("jax_enable_x64", True)


def test_require_x64_reports_single_precision():
    """The guarantee must not rest on a third-party import side effect
    (pyscfad enables x64 when imported); the worker checks the live dtype and
    names the defect when it is absent."""
    import jax
    jax.config.update("jax_enable_x64", False)
    try:
        problem = _datagen._require_x64()
    finally:
        jax.config.update("jax_enable_x64", True)
    assert problem is not None
    assert "float64" in problem
    assert _datagen._require_x64() is None


def test_main_refuses_to_generate_without_double_precision(monkeypatch, tmp_path):
    """A refused precision check exits non-zero BEFORE any generation call, so
    the pretrain ``afterok:datagen`` dependency blocks instead of consuming a
    single-precision file."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "resolved_config.yaml").write_text("dummy: 1\n")
    monkeypatch.setattr(_datagen, "load_grid_config",
                        lambda p: _cfg(["deep"], True))
    calls = []
    monkeypatch.setattr(_datagen, "_ensure_pretrain_data",
                        lambda *a, **k: calls.append((a, k)) or "x.npz")
    monkeypatch.setattr(_datagen, "_require_x64",
                        lambda: "JAX is computing in float32")
    assert _datagen.main([str(run_dir)]) == 1
    assert calls == []


def test_main_binds_the_generator_seam_when_unbound(monkeypatch):
    """The seam is bound lazily in ``main`` (importing the generator pulls in
    jax.numpy); an unbound seam must resolve to the real generator, and a
    patched one must be left alone."""
    from xcquinox.alec import pretrain_data_gen
    monkeypatch.setattr(_datagen, "_ensure_pretrain_data", None)
    assert _datagen.main([]) == 1  # no argv: exits before any generation
    assert _datagen._ensure_pretrain_data is pretrain_data_gen.ensure_pretrain_data
    sentinel = object()
    monkeypatch.setattr(_datagen, "_ensure_pretrain_data", sentinel)
    assert _datagen.main([]) == 1
    assert _datagen._ensure_pretrain_data is sentinel


# ---------------------------------------------------------------------------
# Pretraining-protocol plumbing
# ---------------------------------------------------------------------------

def _cfg2(archs, polarized, *, lock=0.0, **pretrain_kw):
    pt = dict(data_dir="/d/pt", atoms=(), dfs_set=False, pool_atoms=False,
              parent_density="pbe", exchange_footing="total",
              mesh_fraction=0.3)
    pt.update(pretrain_kw)
    return _ns(
        sweep=_ns(arch=list(archs)),
        use_polarized_correlation=polarized,
        pretrain=_ns(**pt),
        inputs=_ns(basis="def2-svp", grid_level=3, density_fit=False,
                   auxbasis=None, orientation_lock_strength=lock),
    )


def test_required_data_specs_single_parent():
    cfg = _cfg2(["deep_3x16", "deep_mgga_3x16"], True)
    assert _datagen._required_data_specs(cfg) == [(True, "pbe")]


def test_required_data_specs_auto_splits_a_mixed_rung_sweep():
    """With parent_density: auto a GGA-rung arch wants the PBE-density file and
    a meta-GGA-rung arch wants the SCAN-density file, so datagen builds both."""
    cfg = _cfg2(["deep_3x16", "deep_mgga_3x16"], True, parent_density="auto")
    assert _datagen._required_data_specs(cfg) == [(True, "pbe"),
                                                  (True, "scan")]


def test_required_data_specs_auto_single_rung():
    cfg = _cfg2(["deep_mgga_3x16"], True, parent_density="auto")
    assert _datagen._required_data_specs(cfg) == [(True, "scan")]


def test_main_threads_every_protocol_knob(monkeypatch, tmp_path):
    cfg = _cfg2(["deep_3x16"], True, dfs_set=True, pool_atoms=True,
                exchange_footing="spin_channel", mesh_fraction=0.25)
    rc, calls = _run_main(monkeypatch, tmp_path, cfg)
    assert rc == 0
    assert len(calls) == 1
    _dd, kw = calls[0]
    assert kw["dfs_set"] is True
    assert kw["pool_atoms"] is True
    assert kw["reference_xc"] == "pbe"
    assert kw["exchange_footing"] == "spin_channel"
    assert kw["mesh_fraction"] == 0.25


def test_main_default_call_is_unchanged(monkeypatch, tmp_path):
    """A YAML written before the protocol change must reach the generator with
    exactly the keyword set it always did -- plus the run's own orientation
    lock, which the manifest keys on and the call had been omitting -- so
    nothing but a lock change can regenerate its data file."""
    cfg = _cfg(["deep", "deep_attn"], True, basis="def2-svp", grid=2,
               df=False, data_dir="/d/svp")
    rc, calls = _run_main(monkeypatch, tmp_path, cfg)
    assert rc == 0
    assert calls[0][1] == {"basis": "def2-svp", "grid_level": 2,
                           "density_fit": False, "auxbasis": None,
                           "polarized": True, "descriptors": True,
                           "orientation_lock_strength": 0.0}


def test_main_names_an_unconverged_reference_scf_and_exits_nonzero(
        monkeypatch, tmp_path):
    """The reference SCF behind a SCAN parent can stall. The stage must report
    the refusal by name and exit non-zero, so the pretrain array's
    ``afterok:datagen`` dependency blocks rather than the traceback being
    swallowed into a successful-looking job."""
    from xcquinox.alec.data import ReferenceSCFNotConverged
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "resolved_config.yaml").write_text("dummy: 1\n")
    monkeypatch.setattr(
        _datagen, "load_grid_config",
        lambda p: _cfg2(["deep_mgga_3x16"], True, parent_density="auto"))

    def _stalled(*a, **k):
        raise ReferenceSCFNotConverged("SCAN reference SCF did not converge",
                                       cycles=150)

    monkeypatch.setattr(_datagen, "_ensure_pretrain_data", _stalled)
    printed = []
    monkeypatch.setattr(_datagen, "_log", printed.append)
    assert _datagen.main([str(run_dir)]) == 1
    failure = [line for line in printed if line.startswith("ERROR:")]
    assert len(failure) == 1
    assert "ReferenceSCFNotConverged" in failure[0]
    assert "150 cycle(s)" in failure[0]


def test_main_logs_the_protocol_knobs_it_generates_with(monkeypatch, tmp_path):
    """The run record must state which knobs produced the data file, so a file
    on disk can be attributed to the configuration that built it."""
    cfg = _cfg2(["deep_3x16"], True, dfs_set=True,
                exchange_footing="spin_channel", mesh_fraction=0.25)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "resolved_config.yaml").write_text("dummy: 1\n")
    monkeypatch.setattr(_datagen, "load_grid_config", lambda p: cfg)
    monkeypatch.setattr(_datagen, "_ensure_pretrain_data",
                        lambda data_dir, **kw: f"{data_dir}/x.npz")
    printed = []
    monkeypatch.setattr(_datagen, "_log", printed.append)
    assert _datagen.main([str(run_dir)]) == 0
    text = "\n".join(printed)
    assert "'dfs_set': True" in text
    assert "'exchange_footing': 'spin_channel'" in text
    assert "'mesh_fraction': 0.25" in text
    assert "pretrain_data_polarized.npz" in text


def test_main_asks_the_currency_check_at_the_runs_own_lock(monkeypatch,
                                                           tmp_path):
    """A degenerate atom's rows are a different component of its manifold
    under a different orientation lock, and the manifest keys on the lock. A
    run at a lock other than the generator's own 3e-5 must therefore state it,
    or the currency check declares the 3e-5 file current and the run trains on
    rows from another Hamiltonian."""
    cfg = _cfg2(["deep_3x16"], True, lock=1e-4)
    rc, calls = _run_main(monkeypatch, tmp_path, cfg)
    assert rc == 0
    assert calls[0][1]["orientation_lock_strength"] == 1e-4


def test_main_logs_the_lock_it_generates_at(monkeypatch, tmp_path):
    cfg = _cfg2(["deep_3x16"], True, lock=1e-4)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "resolved_config.yaml").write_text("dummy: 1\n")
    monkeypatch.setattr(_datagen, "load_grid_config", lambda p: cfg)
    monkeypatch.setattr(_datagen, "_ensure_pretrain_data",
                        lambda data_dir, **kw: f"{data_dir}/x.npz")
    printed = []
    monkeypatch.setattr(_datagen, "_log", printed.append)
    assert _datagen.main([str(run_dir)]) == 0
    assert "orientation_lock_strength=0.0001" in "\n".join(printed)


# ---------------------------------------------------------------------------
# The irreproducible-degenerate waiver reaches the generator from the YAML
# ---------------------------------------------------------------------------

def _cfg_waived(waived, reason="a stated reason", *, lock=3e-5, grid=1):
    cfg = _cfg2(["deep_3x16"], True, lock=lock)
    cfg.inputs.grid_level = grid
    cfg.inputs.allow_irreproducible_degenerate = waived
    cfg.inputs.irreproducible_degenerate_reason = reason
    return cfg


def test_main_carries_the_waiver_the_configuration_states(monkeypatch,
                                                          tmp_path):
    """The refusal is applied to the requested identity inside
    ``ensure_pretrain_data``, so a run whose YAML waives it must say so at the
    call: without the keyword the stage returns 1 and the pretrain array's
    ``afterok`` chain goes ``DependencyNeverSatisfied``, even where the file
    is already on disk and current."""
    rc, calls = _run_main(monkeypatch, tmp_path, _cfg_waived(True))
    assert rc == 0
    assert calls[0][1]["allow_irreproducible_degenerate"] is True


def test_main_states_no_waiver_when_the_configuration_grants_none(monkeypatch,
                                                                  tmp_path):
    """False is the generator's own default, so a configuration that waives
    nothing reaches it with exactly the keyword set it always did and its
    existing data file stays current."""
    rc, calls = _run_main(monkeypatch, tmp_path, _cfg_waived(False))
    assert rc == 0
    assert "allow_irreproducible_degenerate" not in calls[0][1]


def test_main_logs_whether_the_waiver_was_granted(monkeypatch, tmp_path):
    cfg = _cfg_waived(True, "the run record must carry this")
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "resolved_config.yaml").write_text("dummy: 1\n")
    monkeypatch.setattr(_datagen, "load_grid_config", lambda p: cfg)
    monkeypatch.setattr(_datagen, "_ensure_pretrain_data",
                        lambda data_dir, **kw: f"{data_dir}/x.npz")
    printed = []
    monkeypatch.setattr(_datagen, "_log", printed.append)
    assert _datagen.main([str(run_dir)]) == 0
    text = "\n".join(printed)
    assert "allow_irreproducible_degenerate=True" in text
    assert "the run record must carry this" in text


def test_the_generators_waiver_default_is_off():
    """The harness states the flag only when it is granted, which is only
    equivalent to stating False while the generator's own default IS False.
    Pinned so a changed generator default cannot silently waive every run."""
    import inspect

    from xcquinox.alec import pretrain_data_gen as pdg
    for fn in (pdg.ensure_pretrain_data, pdg.generate_pretrain_data_npz):
        param = inspect.signature(fn).parameters[
            "allow_irreproducible_degenerate"]
        assert param.default is False, fn.__name__


def test_a_pre_protocol_namespace_grants_no_waiver(monkeypatch, tmp_path):
    """A ``resolved_config.yaml`` written before the key existed reloads
    without it; the stage must read a refusal, not raise ``AttributeError``
    inside the try and report a generation failure that never happened."""
    cfg = _cfg2(["deep_3x16"], True, lock=3e-5)
    assert not hasattr(cfg.inputs, "allow_irreproducible_degenerate")
    rc, calls = _run_main(monkeypatch, tmp_path, cfg)
    assert rc == 0
    assert "allow_irreproducible_degenerate" not in calls[0][1]


# ---------------------------------------------------------------------------
# Every shipped configuration clears the refusal gate
# ---------------------------------------------------------------------------

def _shipped_config_paths():
    """``hpcjobs/configs/*.yaml`` plus the two shipped example templates.

    Empty outside a source checkout (an installed wheel carries neither), in
    which case the test below skips rather than asserting on an empty glob.
    """
    import glob

    from xcquinox.alec.cluster import _datagen as dg
    cluster_dir = os.path.dirname(os.path.abspath(dg.__file__))
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(cluster_dir)))
    paths = sorted(glob.glob(os.path.join(repo_root, "hpcjobs", "configs",
                                          "*.yaml")))
    examples = os.path.join(cluster_dir, "examples")
    paths += sorted(glob.glob(os.path.join(examples, "*.yaml")))
    return [p for p in paths if os.path.isfile(p)]


def test_every_shipped_configuration_clears_the_datagen_refusal(monkeypatch,
                                                                tmp_path):
    """The real ``_datagen.main`` is run over every deployment configuration
    and every shipped template, with the generator stubbed at its boundary so
    only the refusal decides.

    The gate refuses a spatially degenerate free atom below grid level 3 or at
    an unlocked SCF, BEFORE the currency check, so a configuration that trips
    it cannot run its datagen stage at all -- not even where the required
    ``.npz`` is already on disk and current. Fifteen deployment
    configurations and both templates sit below grid level 3, so each of them
    must state the waiver and its reason; the six grid-level-3 campaigns must
    not need it. No unit test drove a shipped configuration through this stage
    before, which is how a tree in which two thirds of them could not
    generate their data passed green."""
    import yaml as _yaml

    from xcquinox.alec import pretrain_data_gen as pdg
    from xcquinox.alec.cluster.grid_config import load_grid_config
    paths = _shipped_config_paths()
    if not paths:
        pytest.skip("no shipped configurations in this checkout")
    assert len(paths) >= 17, paths

    # The generator's own boundary: the real ensure_pretrain_data runs, so the
    # real refusal, the real system resolution and the real degeneracy rule
    # decide, and nothing is built.
    monkeypatch.setattr(pdg, "pretrain_data_is_current",
                        lambda *a, **k: False)
    monkeypatch.setattr(
        pdg, "generate_pretrain_data_npz",
        lambda out_dir, **kw: os.path.join(out_dir, "built.npz"))
    monkeypatch.setattr(_datagen, "_ensure_pretrain_data",
                        pdg.ensure_pretrain_data)

    refused, built = [], []
    for path in paths:
        run_dir = tmp_path / os.path.basename(path)
        run_dir.mkdir()
        # One TOKEN is substituted, the way ``workflow_matrix.write_matrix_yaml``
        # substitutes it: the matrix template names its architecture
        # CHANGE_ME_ARCH, and every other placeholder in these files is a path
        # the stubbed generator never opens.
        text = open(path).read().replace("CHANGE_ME_ARCH", "deep_3x16")
        (run_dir / "resolved_config.yaml").write_text(text)
        printed = []
        monkeypatch.setattr(_datagen, "_log", printed.append)
        rc = _datagen.main([str(run_dir)])
        (built if rc == 0 else refused).append(
            (os.path.basename(path), "\n".join(printed)))
    assert refused == [], [(name, log.splitlines()[-1])
                           for name, log in refused]
    assert len(built) == len(paths)

    # ... and the waiver is stated exactly where the identity needs it.
    for path in paths:
        raw = (_yaml.safe_load(open(path).read()) or {}).get("inputs") or {}
        cfg = load_grid_config(path)
        waived = cfg.inputs.allow_irreproducible_degenerate
        if cfg.inputs.grid_level >= pdg.COARSE_DEGENERATE_MIN_GRID_LEVEL:
            assert waived is False, os.path.basename(path)
            assert "allow_irreproducible_degenerate" not in raw, path
        else:
            assert waived is True, os.path.basename(path)
            assert cfg.inputs.irreproducible_degenerate_reason.strip(), path
        # The lock is WRITTEN DOWN in every deployment configuration rather
        # than inherited: the harness default is the calibrated lock, so an
        # omitted key would re-identify the fifteen pre-2026-08 campaigns,
        # which ran unlocked and whose pretraining files and cached CCSD
        # intermediates all carry the unlocked identity. Its VALUE is pinned
        # only where the identity depends on it -- a grid-3 campaign builds
        # its degenerate atoms reproducibly only with the lock on. Below that
        # level the waiver covers either lock, and the two shipped templates
        # are not campaigns (grid_step7.yaml states 3e-5; the matrix template
        # inherits it deliberately, with the reason in the file).
        if "hpcjobs" in path:
            assert "orientation_lock_strength" in raw, path
        if cfg.inputs.grid_level >= pdg.COARSE_DEGENERATE_MIN_GRID_LEVEL:
            assert (cfg.inputs.orientation_lock_strength
                    == pdg.PRETRAIN_ORIENTATION_LOCK_STRENGTH), path
