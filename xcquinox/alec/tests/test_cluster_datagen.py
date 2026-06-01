"""Tests for the datagen front stage (``xcquinox.alec.cluster._datagen``).

The datagen stage runs FIRST in the job graph and generates the pretrain-data
file(s) every swept arch needs, before the pretrain stage consumes them. These
tests pin: (1) which files are required for polarized / unpolarized / mixed
sweeps, and (2) that ``main`` calls the idempotent generator once per distinct
required file with the basis/grid/density_fit taken from the resolved config —
without running any real PBE SCF (the generator seam is monkeypatched).
"""
from __future__ import annotations

import types

from xcquinox.alec.cluster import _datagen


def _ns(**kw):
    return types.SimpleNamespace(**kw)


def _cfg(archs, polarized, *, basis="def2-svp", grid=2, df=False,
         data_dir="/data/pt"):
    return _ns(
        sweep=_ns(arch=list(archs)),
        use_polarized_correlation=polarized,
        pretrain=_ns(data_dir=data_dir),
        inputs=_ns(basis=basis, grid_level=grid, density_fit=df),
    )


# ---------------------------------------------------------------------------
# _required_polarized_flags — which files the sweep's archs consume
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
# main — generation calls (generator seam monkeypatched)
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
                  "polarized": True, "descriptors": True}


def test_main_density_fit_tzvpd(monkeypatch, tmp_path):
    cfg = _cfg(["deep", "deep_combined"], True, basis="def2-tzvpd", grid=2,
               df=True, data_dir="/d/tz")
    rc, calls = _run_main(monkeypatch, tmp_path, cfg)
    assert rc == 0
    assert len(calls) == 1
    assert calls[0][1]["density_fit"] is True
    assert calls[0][1]["basis"] == "def2-tzvpd"
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
