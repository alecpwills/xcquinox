"""Tests for ``notebooks/analysis/extract_subset_descriptors.py``.

Pure unit tests for the stat-aggregation helpers + a schema round-trip
test for ``write_local_subset_descriptors_json``. The expensive PBE
precompute path is not exercised in pytest (same separation as
``test_local_reeval.py``).
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
import types
from pathlib import Path

import numpy as np
import pytest

_PATH = Path(__file__).resolve().parent / "extract_subset_descriptors.py"
_spec = importlib.util.spec_from_file_location("extract_subset_descriptors",
                                                _PATH)
ext = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
sys.modules[_spec.name] = ext  # type: ignore[union-attr]
_spec.loader.exec_module(ext)  # type: ignore[union-attr]


# ---------------------------------------------------------------------------
# feature_column_names
# ---------------------------------------------------------------------------

def test_feature_column_names_pairs_class_and_index():
    # Build classes with the right names so type(d).__name__ resolves to
    # what the figure ingest will pick up at runtime.
    DMStatisticsDescriptor = type(
        "DMStatisticsDescriptor", (), {"__init__": lambda self: None})
    CuspDescriptor = type(
        "CuspDescriptor", (), {"__init__": lambda self: None})
    DM = DMStatisticsDescriptor(); DM.n_features = 3
    Cusp = CuspDescriptor(); Cusp.n_features = 2
    names = ext.feature_column_names([DM, Cusp])
    assert names == ["DMStatisticsDescriptor_0", "DMStatisticsDescriptor_1",
                     "DMStatisticsDescriptor_2",
                     "CuspDescriptor_0", "CuspDescriptor_1"]


def test_feature_column_names_empty_list():
    assert ext.feature_column_names([]) == []


# ---------------------------------------------------------------------------
# per_molecule_feature_means
# ---------------------------------------------------------------------------

def test_per_molecule_feature_means_uniform_weights():
    # 4 grid points, 3 features. Uniform weights → ordinary column mean.
    features = np.array([
        [1.0, 2.0, 3.0],
        [2.0, 4.0, 6.0],
        [3.0, 6.0, 9.0],
        [4.0, 8.0, 12.0],
    ])
    weights = np.ones(4)
    out = ext.per_molecule_feature_means(features, weights)
    assert out.shape == (3,)
    assert np.allclose(out, [2.5, 5.0, 7.5])


def test_per_molecule_feature_means_weighted():
    # Weights concentrate on the first row → result should be ~row 0.
    features = np.array([
        [10.0, 100.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ])
    weights = np.array([1.0, 1e-12, 1e-12])
    out = ext.per_molecule_feature_means(features, weights)
    assert np.allclose(out, [10.0, 100.0], rtol=1e-6)


def test_per_molecule_feature_means_zero_weight_falls_back_to_unweighted():
    features = np.array([[1.0, 2.0], [3.0, 4.0]])
    weights = np.zeros(2)
    out = ext.per_molecule_feature_means(features, weights)
    assert np.allclose(out, [2.0, 3.0])  # unweighted mean


def test_per_molecule_feature_means_shape_mismatch_raises():
    with pytest.raises(ValueError, match="N_grid"):
        ext.per_molecule_feature_means(
            np.zeros((3, 2)), np.zeros(5),
        )


def test_per_molecule_feature_means_rejects_1d_features():
    with pytest.raises(ValueError, match="2-D"):
        ext.per_molecule_feature_means(
            np.zeros(3), np.ones(3),
        )


# ---------------------------------------------------------------------------
# per_subset_stats
# ---------------------------------------------------------------------------

def test_per_subset_stats_known_values():
    # 3 molecules × 2 features.
    per_mol = np.array([
        [1.0, 10.0],
        [3.0, 30.0],
        [5.0, 50.0],
    ])
    stats = ext.per_subset_stats(per_mol)
    assert stats["mean"] == pytest.approx([3.0, 30.0])
    assert stats["min"] == pytest.approx([1.0, 10.0])
    assert stats["max"] == pytest.approx([5.0, 50.0])
    assert stats["range"] == pytest.approx([4.0, 40.0])
    # Population std (np default).
    assert stats["std"] == pytest.approx(
        [np.std([1, 3, 5]), np.std([10, 30, 50])])


def test_per_subset_stats_empty_returns_nan_per_feature():
    per_mol = np.zeros((0, 4))
    stats = ext.per_subset_stats(per_mol)
    for k, v in stats.items():
        assert len(v) == 4
        for x in v:
            assert math.isnan(x), f"{k} expected NaN, got {x}"


def test_per_subset_stats_rejects_1d():
    with pytest.raises(ValueError, match="2-D"):
        ext.per_subset_stats(np.zeros(5))


# ---------------------------------------------------------------------------
# JSON round-trip
# ---------------------------------------------------------------------------

def test_write_local_subset_descriptors_json_round_trip(tmp_path):
    spec_dir = tmp_path / "checkpoints" / "spec_0000"
    spec_dir.mkdir(parents=True)
    per_mol = np.array([[1.0, 10.0], [2.0, 20.0]])
    out = ext.write_local_subset_descriptors_json(
        spec_dir,
        training_molecule_names=["H2O", "CH4"],
        feature_names=["dm_0", "cusp_0"],
        per_molecule_features=per_mol,
    )
    assert out.is_file()
    loaded = json.loads(out.read_text())
    assert loaded["training_molecule_names"] == ["H2O", "CH4"]
    assert loaded["feature_names"] == ["dm_0", "cusp_0"]
    assert loaded["per_molecule_features"] == [[1.0, 10.0], [2.0, 20.0]]
    assert loaded["per_subset_stats"]["mean"] == pytest.approx([1.5, 15.0])
    assert loaded["per_subset_stats"]["range"] == pytest.approx([1.0, 10.0])


# ---------------------------------------------------------------------------
# discover_specs_in_run
# ---------------------------------------------------------------------------

def test_discover_specs_in_run_lists_present_spec_files(tmp_path):
    sp = tmp_path / "specs"
    sp.mkdir()
    for name in ("spec_0000.spec", "spec_0005.spec", "spec_0036.spec",
                 "manifest.json", "spec_xx.spec"):
        (sp / name).write_bytes(b"x")
    out = ext.discover_specs_in_run(tmp_path)
    assert out == [0, 5, 36]
