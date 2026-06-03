"""Tests for xcquinox.alec.cluster.materialize -- on-disk spec serialization.

The round-trip tests deliberately load spec files through the actual
``xcquinox.alec._train_one_spec._load_spec`` so the test verifies the real
worker loader, not a reconstruction of it.
"""
import hashlib
import json
import os
from dataclasses import dataclass

import pytest

from xcquinox.alec._train_one_spec import _load_spec
from xcquinox.alec.cluster.grid_config import GridCell
from xcquinox.alec.cluster.materialize import (
    _spec_filename,
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


def test_write_spec_atomic_uses_protocol_4(tmp_path):
    obj = _StubSpec(name="ae", payload=())
    path = str(tmp_path / "spec_0000.spec")
    write_spec_atomic(obj, path)

    with open(path, "rb") as f:
        head = f.read(2)
    # A protocol-N serialized stream begins with the PROTO opcode (b"\x80")
    # followed by a single byte holding the protocol number.
    assert head[0:1] == b"\x80"
    assert head[1] == 4


def test_write_spec_atomic_rejects_non_none_pbe_anchor(tmp_path):
    obj = _StubSpec(name="ae", payload=(), pbe_anchor_sample=[0.1, 0.2])
    path = str(tmp_path / "spec_0000.spec")
    with pytest.raises(ValueError, match="pbe_anchor_sample"):
        write_spec_atomic(obj, path)
    # Nothing (not even an orphan temp) should be left behind.
    assert os.listdir(tmp_path) == []


def test_write_spec_atomic_leaves_no_orphan_temp(tmp_path):
    obj = _StubSpec(name="ae", payload=(7,))
    path = str(tmp_path / "spec_0000.spec")
    write_spec_atomic(obj, path)
    names = sorted(os.listdir(tmp_path))
    assert names == ["spec_0000.spec"]
    assert not any(n.startswith(".mktmp_") for n in names)


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


def test_materialize_specs_purges_crash_orphaned_temp(tmp_path):
    out_dir = tmp_path / "specs"
    out_dir.mkdir()
    # Simulate a crash-orphaned temp file from a prior interrupted write.
    orphan = out_dir / ".mktmp_deadbeef"
    orphan.write_bytes(b"garbage")

    cells = _make_cells(2)
    specs = [(c, _StubSpec(name=c.arch, payload=())) for c in cells]
    paths = materialize_specs(specs, str(out_dir))

    names = sorted(os.listdir(out_dir))
    # Orphan purged; only the two real spec files remain.
    assert names == ["spec_0000.spec", "spec_0001.spec"]
    assert not orphan.exists()
    assert len(paths) == 2


def test_materialize_specs_purges_stale_higher_index_spec(tmp_path):
    out_dir = tmp_path / "specs"
    out_dir.mkdir()
    # Leftover from a prior, larger grid -- index 30 >= new N (10).
    stale = out_dir / "spec_0030.spec"
    stale.write_bytes(b"stale")
    # An in-range stale file should be overwritten, not left as-is.
    keep_overwritten = out_dir / "spec_0005.spec"
    keep_overwritten.write_bytes(b"old")

    cells = _make_cells(10)
    specs = [(c, _StubSpec(name=c.arch, payload=(i,))) for i, c in enumerate(cells)]
    paths = materialize_specs(specs, str(out_dir))

    assert not stale.exists()
    names = sorted(os.listdir(out_dir))
    assert names == [f"spec_{i:04d}.spec" for i in range(10)]
    assert len(paths) == 10
    # The in-range slot was overwritten with the fresh spec.
    assert _load_spec(str(keep_overwritten)) == specs[5][1]


def test_materialize_specs_pad_width_widens_for_large_grids():
    # N = 10001 -> largest index 10000 -> width 5.
    n = 10001
    width = max(4, len(str(n - 1)))
    assert width == 5
    assert _spec_filename(0, width) == "spec_00000.spec"
    assert _spec_filename(10000, width) == "spec_10000.spec"


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


def test_write_manifest_is_atomic_no_partial_file(tmp_path):
    cells = _make_cells(2)
    specs = [(c, _StubSpec(name=c.arch, payload=())) for c in cells]
    out_dir = str(tmp_path / "specs")
    paths = materialize_specs(specs, out_dir)

    write_manifest(cells, paths, out_dir)

    # No leftover temp file; the manifest is complete, valid JSON.
    names = sorted(os.listdir(out_dir))
    assert not any(n.startswith(".mktmp_") for n in names)
    assert "manifest.json" in names
    with open(os.path.join(out_dir, "manifest.json")) as f:
        json.load(f)  # raises if partial/corrupt


def test_write_manifest_rejects_mismatched_lengths(tmp_path):
    cells = _make_cells(2)
    with pytest.raises(ValueError, match="same length"):
        write_manifest(cells, ["only_one_path"], str(tmp_path))
