"""Tests for xcquinox.pipeline.cluster.materialize -- on-disk spec serialization.

The round-trip tests deliberately load spec files through the actual
``xcquinox.pipeline._train_one_spec._load_spec`` so the test verifies the real
worker loader, not a reconstruction of it.
"""
import hashlib
import json
import os
from dataclasses import dataclass


from xcquinox.pipeline._train_one_spec import _load_spec
from xcquinox.pipeline.cluster.grid_config import GridCell
from xcquinox.pipeline.cluster.materialize import (
    materialize_specs,
    write_manifest,
    write_spec_atomic,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@dataclass
class _StubSpec:
    """Minimal serializable stand-in for a TrainingSpec.

    The harness serializer only ever reads ``pbe_anchor_sample`` for its
    defensive guard; everything else is opaque payload that must round-trip.
    """
    name: str
    payload: tuple
    pbe_anchor_sample: object = None


def _make_cells(n):
    """n distinct GridCells."""
    return [
        GridCell(
            arch=f"arch{i}",
            loss="l5",
            metric="l2",
            subset_size=8 + i,
            solver="default",
        )
        for i in range(n)
    ]


# ---------------------------------------------------------------------------
# write_spec_atomic
# ---------------------------------------------------------------------------

def test_write_spec_atomic_round_trips_via_worker_loader(tmp_path):
    obj = _StubSpec(name="ae_h2o", payload=(1, 2, "three"))
    path = str(tmp_path / "spec_0000.spec")
    write_spec_atomic(obj, path)

    # Loaded through the real worker loader.
    loaded = _load_spec(path)
    assert loaded == obj


# ---------------------------------------------------------------------------
# materialize_specs
# ---------------------------------------------------------------------------

def test_materialize_specs_writes_padded_paths_in_order(tmp_path):
    cells = _make_cells(3)
    specs = [(c, _StubSpec(name=c.arch, payload=(i,))) for i, c in enumerate(cells)]
    out_dir = str(tmp_path / "specs")

    paths = materialize_specs(specs, out_dir)

    assert paths == [
        os.path.join(out_dir, "spec_0000.spec"),
        os.path.join(out_dir, "spec_0001.spec"),
        os.path.join(out_dir, "spec_0002.spec"),
    ]
    for i, p in enumerate(paths):
        assert os.path.isfile(p)
        assert _load_spec(p) == specs[i][1]


# ---------------------------------------------------------------------------
# write_manifest
# ---------------------------------------------------------------------------

def test_write_manifest_records_cells_files_hashes_and_top_level(tmp_path):
    import xcquinox

    cells = _make_cells(3)
    specs = [(c, _StubSpec(name=c.arch, payload=(i,))) for i, c in enumerate(cells)]
    out_dir = str(tmp_path / "specs")
    paths = materialize_specs(specs, out_dir)

    manifest_path = write_manifest(cells, paths, out_dir)
    assert manifest_path == os.path.join(out_dir, "manifest.json")

    with open(manifest_path) as f:
        manifest = json.load(f)

    # Top-level metadata.
    assert manifest["xcquinox_version"] == xcquinox.__version__
    assert manifest["n_specs"] == 3
    assert manifest["width"] == 4
    assert "python_version" in manifest

    # Per-index entries.
    assert len(manifest["specs"]) == 3
    for idx, entry in enumerate(manifest["specs"]):
        assert entry["index"] == idx
        assert entry["spec_file"] == f"spec_{idx:04d}.spec"
        assert entry["cell"] == {
            "arch": cells[idx].arch,
            "loss": cells[idx].loss,
            "metric": cells[idx].metric,
            "subset_size": cells[idx].subset_size,
            "solver": cells[idx].solver,
        }
        with open(paths[idx], "rb") as fh:
            expected_hash = hashlib.sha256(fh.read()).hexdigest()
        assert entry["sha256"] == expected_hash


