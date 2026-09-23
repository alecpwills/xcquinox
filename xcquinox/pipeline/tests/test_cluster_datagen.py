"""Tests for the datagen front stage (``xcquinox.pipeline.cluster._datagen``).

The datagen stage runs FIRST in the job graph and generates the pretrain-data
file(s) every swept arch needs, before the pretrain stage consumes them. These
tests pin: (1) which files are required for polarized / unpolarized / mixed
sweeps, and (2) that ``main`` calls the idempotent generator once per distinct
required file with the basis/grid/density_fit taken from the resolved config,
without running any real PBE SCF (the generator seam is monkeypatched).
"""
from __future__ import annotations

import types
from pathlib import Path

import pytest

from xcquinox.pipeline.cluster import _datagen

#: The campaign configuration the factored call is read against: the v7
#: meta-GGA families group, whose five architectures all sit on one parent
#: density, so the stage builds exactly one file.
_MGGA_CONFIG = (Path(__file__).resolve().parents[3] / "hpcjobs" / "configs"
                / "dfs_step7.dfs6311_grid3_v7g2_families_mgga.yaml")


def _mgga_config():
    """The campaign configuration file, skipped where the checkout has none."""
    pytest.importorskip("yaml")
    if not _MGGA_CONFIG.is_file():
        pytest.skip(f"no {_MGGA_CONFIG.name} in this checkout")
    return _MGGA_CONFIG


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


def test_main_missing_config_returns_1(tmp_path):
    assert _datagen.main([str(tmp_path / "nope")]) == 1


# ---------------------------------------------------------------------------
# JAX precision routing: the datagen node must compute in float64
# ---------------------------------------------------------------------------


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


def test_main_names_an_unconverged_reference_scf_and_exits_nonzero(
        monkeypatch, tmp_path):
    """The reference SCF behind a SCAN parent can stall. The stage must report
    the refusal by name and exit non-zero, so the pretrain array's
    ``afterok:datagen`` dependency blocks rather than the traceback being
    swallowed into a successful-looking job."""
    from xcquinox.pipeline.data import ReferenceSCFNotConverged
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


# ---------------------------------------------------------------------------
# The irreproducible-degenerate waiver reaches the generator from the YAML
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Every shipped configuration clears the refusal gate
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The generation call as a module-level function: the stage and any tool that
# opens the file it wrote ask for ONE identity
# ---------------------------------------------------------------------------

def test_datagen_call_is_the_stage_call(monkeypatch, tmp_path):
    """The call the stage makes IS the factored one, argument for argument.

    A tool that opens the file this stage wrote must derive the file's
    identity through the same code. A keyword present on one side alone names
    another file, or -- worse -- rebuilds the run's own under a manifest the
    run never asked for, at the cost of the reference SCFs and of the
    comparison the file exists for.
    """
    from xcquinox.pipeline.cluster._datagen import datagen_call
    from xcquinox.pipeline.cluster.grid_config import load_grid_config

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "resolved_config.yaml").write_text(_mgga_config().read_text())
    calls = []
    monkeypatch.setattr(
        _datagen, "_ensure_pretrain_data",
        lambda data_dir, **kw: (calls.append((data_dir, kw))
                                or f"{data_dir}/x.npz"))
    assert _datagen.main([str(run_dir)]) == 0

    cfg = load_grid_config(str(run_dir / "resolved_config.yaml"))
    # Every architecture on this axis carries the meta-GGA ingredient and the
    # run is polarized, so the sweep requires exactly one file.
    assert len(calls) == 1
    assert calls[0] == datagen_call(cfg, True, "scan")


