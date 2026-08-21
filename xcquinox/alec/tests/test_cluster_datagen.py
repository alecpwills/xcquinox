"""Tests for the datagen front stage (``xcquinox.alec.cluster._datagen``).

The datagen stage runs FIRST in the job graph and generates the pretrain-data
file(s) every swept arch needs, before the pretrain stage consumes them. These
tests pin: (1) which files are required for polarized / unpolarized / mixed
sweeps, and (2) that ``main`` calls the idempotent generator once per distinct
required file with the basis/grid/density_fit taken from the resolved config,
without running any real PBE SCF (the generator seam is monkeypatched).
"""
from __future__ import annotations

import types

from xcquinox.alec.cluster import _datagen


def _ns(**kw):
    return types.SimpleNamespace(**kw)


def _cfg(archs, polarized, *, basis="def2-svp", grid=2, df=False, aux=None,
         data_dir="/data/pt"):
    return _ns(
        sweep=_ns(arch=list(archs)),
        use_polarized_correlation=polarized,
        pretrain=_ns(data_dir=data_dir),
        inputs=_ns(basis=basis, grid_level=grid, density_fit=df, auxbasis=aux),
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
                        lambda name: _ns(use_polarized_correlation=(name == "pol")))
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
                  "auxbasis": None, "polarized": True, "descriptors": True}


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
                        lambda name: _ns(use_polarized_correlation=(name == "pol")))
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
