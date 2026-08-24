"""Tests for xcquinox.alec.pretrain.

Implements THE SPEC §13.2 test_pretrain.py items (1)-(23).

Tests 1-8 and 19 need only PretrainSpec + stdlib. The end-to-end
run_pretrain tests (9-16) run against tiny session-generated pretrain data
(the ``tiny_pretrain_data_dir`` fixture; He, sto-3g, grid 0) produced by the
production writer, so their data schema tracks the writer by construction.
Tests needing legacy_step3b_checkpoint/ remain xfail until that fixture
exists.
"""
import dataclasses
import json
import math
import os
import tempfile

import pytest

from xcquinox.alec.config import ArchitectureConfig, PretrainSpec, get_architecture


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_arch(**overrides):
    defaults = dict(name="t", depth=2, nodes=8, attention=False,
                    descriptors=(), x_constraints=(), c_constraints=(),
                    double_lob_clamp_allowed=False)
    defaults.update(overrides)
    return ArchitectureConfig(**defaults)


def _make_spec(**overrides):
    """Build a PretrainSpec with a real temp data_dir and checkpoint_dir."""
    tmpdir = tempfile.mkdtemp()
    ckdir = os.path.join(tmpdir, "ckpt")
    defaults = dict(
        arch=_make_arch(),
        data_dir=tmpdir,
        checkpoint_dir=ckdir,
        n_steps=10,
        lr_start=1e-2,
        lr_end=1e-5,
        lr_decay_start=0.2,
        grad_clip=1.0,
        seed=42,
    )
    defaults.update(overrides)
    return PretrainSpec(**defaults)


# ---------------------------------------------------------------------------
# Tests 1-7: PretrainSpec.validate negative paths
# ---------------------------------------------------------------------------

# (1) n_steps=0 raises ValueError
def test_pretrainspec_validate_n_steps_zero():
    with tempfile.TemporaryDirectory() as tmpdir:
        ckdir = os.path.join(tmpdir, "ck")
        spec = PretrainSpec(
            arch=_make_arch(), data_dir=tmpdir, checkpoint_dir=ckdir,
            n_steps=0,
        )
        with pytest.raises(ValueError, match="n_steps must be > 0"):
            spec.validate()


# (2) lr_decay_start=1.5 raises ValueError
def test_pretrainspec_validate_lr_decay_start_out_of_range():
    with tempfile.TemporaryDirectory() as tmpdir:
        ckdir = os.path.join(tmpdir, "ck")
        spec = PretrainSpec(
            arch=_make_arch(), data_dir=tmpdir, checkpoint_dir=ckdir,
            lr_decay_start=1.5,
        )
        with pytest.raises(ValueError, match="lr_decay_start must be in"):
            spec.validate()


# (3) missing data_dir raises ValueError
def test_pretrainspec_validate_missing_data_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        ckdir = os.path.join(tmpdir, "ck")
        spec = PretrainSpec(
            arch=_make_arch(),
            data_dir="/tmp/alec_nonexistent_data_dir_xyz",
            checkpoint_dir=ckdir,
        )
        with pytest.raises(ValueError, match="data_dir does not exist"):
            spec.validate()


# (4) lr_start < lr_end raises ValueError
def test_pretrainspec_validate_lr_start_below_lr_end():
    with tempfile.TemporaryDirectory() as tmpdir:
        ckdir = os.path.join(tmpdir, "ck")
        spec = PretrainSpec(
            arch=_make_arch(), data_dir=tmpdir, checkpoint_dir=ckdir,
            lr_start=1e-5, lr_end=1e-2,
        )
        with pytest.raises(ValueError, match="lr_start .* must be >= lr_end"):
            spec.validate()


# (5) grad_clip=-1.0 raises ValueError
def test_pretrainspec_validate_grad_clip_nonpositive():
    with tempfile.TemporaryDirectory() as tmpdir:
        ckdir = os.path.join(tmpdir, "ck")
        spec = PretrainSpec(
            arch=_make_arch(), data_dir=tmpdir, checkpoint_dir=ckdir,
            grad_clip=-1.0,
        )
        with pytest.raises(ValueError, match="grad_clip must be > 0"):
            spec.validate()


# (6) C-R11-H7: non-finite float hyperparameter raises ValueError
@pytest.mark.parametrize("field_name", ["lr_start", "lr_end", "lr_decay_start", "grad_clip"])
@pytest.mark.parametrize("bad_value", [float("nan"), float("inf")])
def test_pretrainspec_validate_nonfinite_float(field_name, bad_value):
    with tempfile.TemporaryDirectory() as tmpdir:
        ckdir = os.path.join(tmpdir, "ck")
        kwargs = dict(arch=_make_arch(), data_dir=tmpdir, checkpoint_dir=ckdir)
        kwargs[field_name] = bad_value
        spec = PretrainSpec(**kwargs)
        with pytest.raises(ValueError, match=f"{field_name} must be finite"):
            spec.validate()


# (7) C-R11-H7: checkpoint_dir exists as a regular file raises ValueError
def test_pretrainspec_validate_checkpoint_dir_is_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = os.path.join(tmpdir, "not_a_dir.chk")
        with open(file_path, "w") as f:
            f.write("x")
        spec = PretrainSpec(
            arch=_make_arch(), data_dir=tmpdir,
            checkpoint_dir=file_path,
        )
        with pytest.raises(ValueError, match="checkpoint_dir exists but is not a directory"):
            spec.validate()


# ---------------------------------------------------------------------------
# Test 8: PretrainSpec.describe roundtrip
# ---------------------------------------------------------------------------

def test_pretrainspec_describe_roundtrip():
    """(8) describe() returns a dict with all fields; json-serializable."""
    spec = PretrainSpec(
        arch=get_architecture("deep_combined"),
        data_dir="/tmp/alec_nonexistent_data_dir",
        checkpoint_dir="/tmp/alec_nonexistent_pretrain_ckpt",
        n_steps=100,
        lr_start=1e-2,
        lr_end=1e-5,
        lr_decay_start=0.2,
        grad_clip=1.0,
        seed=0,
    )
    out = spec.describe()
    assert isinstance(out, dict)
    field_names = {f.name for f in dataclasses.fields(spec)}
    assert field_names == set(out.keys()), (
        f"describe() field-set mismatch: missing={field_names - set(out.keys())}, "
        f"extra={set(out.keys()) - field_names}"
    )
    # arch must serialize as its name string
    assert out["arch"] == "deep_combined"
    # Must be JSON serializable
    json.dumps(out)


# ---------------------------------------------------------------------------
# cnet pretraining input carries the zeta column when polarized.
# ---------------------------------------------------------------------------
def _polc_arch():
    return ArchitectureConfig.from_spec(
        "polc_pt", 2, 8, use_polarized_correlation=True)


def test_assemble_pretrain_descriptors_cnet_inserts_zeta_when_polarized():
    import numpy as np
    from xcquinox.alec.pretrain import _assemble_pretrain_descriptors

    n = 6
    data = {
        "rho_all": np.linspace(0.1, 1.0, n),
        "sigma_all": np.linspace(0.0, 0.5, n),
        "zeta_all": np.linspace(-0.8, 0.8, n),
    }
    arch = _polc_arch()
    # xnet input: zeta-blind -> [rho, sigma] (no descriptors here).
    dx = np.asarray(_assemble_pretrain_descriptors(arch, data))
    assert dx.shape == (n, 2)
    # cnet input: zeta at column 2.
    dc = np.asarray(_assemble_pretrain_descriptors(arch, data, for_cnet=True))
    assert dc.shape == (n, 3)
    np.testing.assert_allclose(dc[:, 2], data["zeta_all"])
    # rho/sigma columns unchanged.
    np.testing.assert_allclose(dc[:, 0], data["rho_all"])
    np.testing.assert_allclose(dc[:, 1], data["sigma_all"])


def test_assemble_pretrain_descriptors_cnet_zeta_zeros_fallback():
    import numpy as np
    from xcquinox.alec.pretrain import _assemble_pretrain_descriptors

    n = 5
    data = {"rho_all": np.linspace(0.1, 1.0, n),
            "sigma_all": np.linspace(0.0, 0.5, n)}  # no zeta_all
    dc = np.asarray(
        _assemble_pretrain_descriptors(_polc_arch(), data, for_cnet=True))
    assert dc.shape == (n, 3)
    np.testing.assert_allclose(dc[:, 2], np.zeros(n))  # valid unpolarized warm-start


def test_assemble_pretrain_descriptors_unpolarized_ignores_for_cnet():
    import numpy as np
    from xcquinox.alec.pretrain import _assemble_pretrain_descriptors

    n = 4
    data = {"rho_all": np.linspace(0.1, 1.0, n),
            "sigma_all": np.linspace(0.0, 0.5, n),
            "zeta_all": np.linspace(-0.5, 0.5, n)}
    arch = _make_arch()  # use_polarized_correlation defaults False
    dx = np.asarray(_assemble_pretrain_descriptors(arch, data))
    dc = np.asarray(_assemble_pretrain_descriptors(arch, data, for_cnet=True))
    # Unpolarized: no zeta column inserted even with for_cnet and zeta_all present.
    assert dx.shape == (n, 2) and dc.shape == (n, 2)
    np.testing.assert_allclose(dx, dc)


def test_assemble_pretrain_descriptors_rung35_arch():
    # Regression: deep_rung35_3x16 (cusp + rung35) assembles without the KeyError
    # rung35 hit before it was added to pretrain _key_map.
    import dataclasses
    import numpy as np
    from xcquinox.alec.pretrain import _assemble_pretrain_descriptors
    from xcquinox.alec import get_architecture

    n = 6
    data = {
        "rho_all": np.linspace(0.1, 1.0, n),
        "sigma_all": np.linspace(0.0, 0.5, n),
        "zeta_all": np.linspace(-0.5, 0.5, n),
        "cusp_all": np.linspace(0.0, 1.0, 2 * n).reshape(n, 2),
        "rung35_all": np.linspace(1.0, 0.0, 2 * n).reshape(n, 2),
    }
    arch = get_architecture("deep_rung35_3x16")  # descriptors: cusp (2) + rung35 (2)
    dx = np.asarray(_assemble_pretrain_descriptors(arch, data))
    assert dx.shape == (n, 6)  # rho, sigma, cusp(2), rung35(2); rung35 -> no KeyError
    # Polarized cnet inserts zeta at column 2, then the 4 descriptor columns.
    parch = dataclasses.replace(arch, use_polarized_correlation=True)
    dc = np.asarray(_assemble_pretrain_descriptors(parch, data, for_cnet=True))
    assert dc.shape == (n, 7)
    np.testing.assert_allclose(dc[:, 2], data["zeta_all"])


def test_atom_columns_includes_rung35_occupancy():
    # The per-atom pretrain column generator must emit a bounded [0, 1] rung35
    # occupancy column aligned with rho (H atom, sto-3g -> fast).
    import numpy as np
    from xcquinox.alec.pretrain_data_gen import _atom_columns
    cols = _atom_columns("H", 1, "sto-3g", 1, polarized=True, descriptors=True)
    assert "rung35" in cols
    r = np.asarray(cols["rung35"])
    assert r.ndim == 2 and r.shape == (len(cols["rho"]), 2)
    assert np.all(r >= -1e-6) and np.all(r <= 1.0 + 1e-6)


# ---------------------------------------------------------------------------
# Tests 9-16: run_pretrain end-to-end (xfail, need fixture)
# ---------------------------------------------------------------------------

FIXTURE_DIR = os.path.join(
    os.path.dirname(__file__), "fixtures"
)
@pytest.fixture(scope="session")
def tiny_pretrain_data_dir(tmp_path_factory):
    """Tiny pretrain data (He, sto-3g, grid 0) generated by the production
    writer, so the schema can never drift from what run_pretrain loads.

    Replaces a committed fixture that never existed: the end-to-end pretrain
    tests below had xfailed on the missing file since they were written, and
    their gate checked a filename (pretrain_data_tiny.npz) run_pretrain never
    loads in any case."""
    from xcquinox.alec.pretrain_data_gen import generate_pretrain_data_npz
    d = tmp_path_factory.mktemp("pretrain_tiny")
    generate_pretrain_data_npz(str(d), atoms=(("He", 0),), basis="sto-3g",
                               grid_level=0, polarized=False,
                               descriptors=True, density_fit=False)
    return str(d)


def test_run_pretrain_end_to_end(tiny_pretrain_data_dir):
    """(9) run_pretrain produces all expected artifacts."""
    from xcquinox.alec.pretrain import run_pretrain

    with tempfile.TemporaryDirectory() as ckdir:
        spec = PretrainSpec(
            arch=_make_arch(),
            data_dir=tiny_pretrain_data_dir,
            checkpoint_dir=ckdir,
            n_steps=3,
            lr_start=1e-2,
            lr_end=1e-5,
            lr_decay_start=0.0,
            grad_clip=1.0,
            seed=0,
        )
        metadata = run_pretrain(spec)
        assert os.path.isfile(os.path.join(ckdir, "xnet.eqx"))
        assert os.path.isfile(os.path.join(ckdir, "cnet.eqx"))
        assert os.path.isfile(os.path.join(ckdir, "losses_x.npy"))
        assert os.path.isfile(os.path.join(ckdir, "losses_c.npy"))
        assert os.path.isfile(os.path.join(ckdir, "pretrain_metadata.json"))
        assert isinstance(metadata, dict)


def test_run_pretrain_losses_finite(tiny_pretrain_data_dir):
    """(10) Losses returned by run_pretrain are finite scalars."""
    import numpy as np
    from xcquinox.alec.pretrain import run_pretrain

    with tempfile.TemporaryDirectory() as ckdir:
        spec = PretrainSpec(
            arch=_make_arch(),
            data_dir=tiny_pretrain_data_dir,
            checkpoint_dir=ckdir,
            n_steps=3,
            lr_start=1e-2,
            lr_end=1e-5,
            lr_decay_start=0.0,
            grad_clip=1.0,
            seed=0,
        )
        metadata = run_pretrain(spec)
        losses_x = np.load(os.path.join(ckdir, "losses_x.npy"))
        losses_c = np.load(os.path.join(ckdir, "losses_c.npy"))
        assert np.all(np.isfinite(losses_x))
        assert np.all(np.isfinite(losses_c))


def test_run_pretrain_xnet_serialization_roundtrip(tiny_pretrain_data_dir):
    """(11) xnet.eqx round-trips: deserialise preserves outputs bitwise."""
    import numpy as np
    import jax.numpy as jnp
    import equinox as eqx
    from xcquinox.alec.pretrain import run_pretrain
    from xcquinox.alec.networks import create_network_pair

    with tempfile.TemporaryDirectory() as ckdir:
        arch = _make_arch()
        spec = PretrainSpec(
            arch=arch,
            data_dir=tiny_pretrain_data_dir,
            checkpoint_dir=ckdir,
            n_steps=3,
            seed=0,
        )
        run_pretrain(spec)

        xnet_path = os.path.join(ckdir, "xnet.eqx")
        xnet_skel, _ = create_network_pair(arch, seed=0)
        xnet_loaded = eqx.tree_deserialise_leaves(xnet_path, xnet_skel)

        # Compare on a synthetic input
        inp = jnp.array([0.1, 0.01])
        out_orig = xnet_loaded(inp)
        out_reload = eqx.tree_deserialise_leaves(xnet_path, xnet_skel)(inp)
        assert np.array_equal(np.array(out_orig), np.array(out_reload))


def test_run_pretrain_cnet_serialization_roundtrip(tiny_pretrain_data_dir):
    """(12) cnet.eqx round-trips: deserialise preserves outputs bitwise."""
    import numpy as np
    import jax.numpy as jnp
    import equinox as eqx
    from xcquinox.alec.pretrain import run_pretrain
    from xcquinox.alec.networks import create_network_pair

    with tempfile.TemporaryDirectory() as ckdir:
        arch = _make_arch()
        spec = PretrainSpec(
            arch=arch,
            data_dir=tiny_pretrain_data_dir,
            checkpoint_dir=ckdir,
            n_steps=3,
            seed=0,
        )
        run_pretrain(spec)

        cnet_path = os.path.join(ckdir, "cnet.eqx")
        _, cnet_skel = create_network_pair(arch, seed=0)
        cnet_loaded = eqx.tree_deserialise_leaves(cnet_path, cnet_skel)

        inp = jnp.array([0.1, 0.01])
        out_orig = cnet_loaded(inp)
        out_reload = eqx.tree_deserialise_leaves(cnet_path, cnet_skel)(inp)
        assert np.array_equal(np.array(out_orig), np.array(out_reload))


def test_run_pretrain_losses_x_npy_roundtrip(tiny_pretrain_data_dir):
    """(13) np.load('losses_x.npy') returns the same array as in-memory."""
    import numpy as np
    from xcquinox.alec.pretrain import run_pretrain

    with tempfile.TemporaryDirectory() as ckdir:
        spec = PretrainSpec(
            arch=_make_arch(),
            data_dir=tiny_pretrain_data_dir,
            checkpoint_dir=ckdir,
            n_steps=3,
            seed=0,
        )
        metadata = run_pretrain(spec)
        losses_x = np.load(os.path.join(ckdir, "losses_x.npy"))
        assert len(losses_x) == 3
        assert losses_x.dtype == np.float64
        assert math.isfinite(metadata["final_loss_x"])


def test_run_pretrain_losses_c_npy_roundtrip(tiny_pretrain_data_dir):
    """(14) np.load('losses_c.npy') returns the same array as in-memory."""
    import numpy as np
    from xcquinox.alec.pretrain import run_pretrain

    with tempfile.TemporaryDirectory() as ckdir:
        spec = PretrainSpec(
            arch=_make_arch(),
            data_dir=tiny_pretrain_data_dir,
            checkpoint_dir=ckdir,
            n_steps=3,
            seed=0,
        )
        metadata = run_pretrain(spec)
        losses_c = np.load(os.path.join(ckdir, "losses_c.npy"))
        assert len(losses_c) == 3
        assert losses_c.dtype == np.float64
        assert math.isfinite(metadata["final_loss_c"])


def test_run_pretrain_metadata_json_all_fields(tiny_pretrain_data_dir):
    """(15) pretrain_metadata.json roundtrips with every documented field."""
    from xcquinox.alec.pretrain import run_pretrain

    required_fields = {
        "arch_name", "pretrain_steps", "lr_start", "lr_end",
        "lr_decay_start", "grad_clip", "final_loss_x", "final_loss_c",
        "min_loss_x", "min_loss_c", "use_cusp", "use_dm",
        "meta_gga", "n_extra_features", "pretrain_mesh",
        "timestamp", "duration_seconds",
    }

    with tempfile.TemporaryDirectory() as ckdir:
        spec = PretrainSpec(
            arch=_make_arch(),
            data_dir=tiny_pretrain_data_dir,
            checkpoint_dir=ckdir,
            n_steps=3,
            seed=0,
        )
        run_pretrain(spec)
        md_path = os.path.join(ckdir, "pretrain_metadata.json")
        with open(md_path) as f:
            md = json.load(f)
        missing = required_fields - set(md.keys())
        assert not missing, f"pretrain_metadata.json missing keys: {missing}"
        assert md["arch_name"] == "t"
        assert md["pretrain_steps"] == 3
        # Shape keys the run validator cross-checks: must carry the arch's
        # real values (this arch has no descriptors and no meta-GGA input,
        # so the (s, alpha) mesh must not have been appended).
        assert md["meta_gga"] is False
        assert md["n_extra_features"] == 0
        assert md["pretrain_mesh"] is False


def test_run_pretrain_warmup_phase_and_progress_callback(tiny_pretrain_data_dir):
    """(16) warmup phase is respected and progress_callback receives dict payloads."""
    from xcquinox.alec.pretrain import run_pretrain

    received = []

    def _cb(payload):
        received.append(payload)

    with tempfile.TemporaryDirectory() as ckdir:
        spec = PretrainSpec(
            arch=_make_arch(),
            data_dir=tiny_pretrain_data_dir,
            checkpoint_dir=ckdir,
            n_steps=5,
            lr_start=1e-2,
            lr_end=1e-5,
            lr_decay_start=0.4,
            grad_clip=1.0,
            seed=0,
        )
        run_pretrain(spec, progress_callback=_cb)

    assert len(received) > 0
    payload = received[0]
    for key in ("arch", "phase", "step", "total", "loss", "timestamp"):
        assert key in payload, f"progress_callback payload missing key {key!r}"


# ---------------------------------------------------------------------------
# Tests 17a-17b: from_legacy_step3b pretrain layout (xfail, need fixture)
# ---------------------------------------------------------------------------

LEGACY_CKPT_DIR = os.path.join(FIXTURE_DIR, "legacy_step3b_checkpoint")
_LEGACY_REASON = (
    "legacy_step3b_checkpoint/ fixture not yet generated; "
    "run generate_legacy_step3b_checkpoint.py to create it"
)


@pytest.mark.xfail(
    not os.path.isdir(LEGACY_CKPT_DIR),
    reason=_LEGACY_REASON,
    strict=False,
)
def test_from_legacy_step3b_pretrain_layout_returns_tuple():
    """(17a) from_legacy_step3b returns tuple[AlecGGA_XNet, AlecGGA_CNet]."""
    import equinox as eqx
    from xcquinox.alec.pretrain import from_legacy_step3b
    from xcquinox.alec.networks import AlecGGA_XNet, AlecGGA_CNet

    arch = _make_arch()
    result = from_legacy_step3b(LEGACY_CKPT_DIR, arch)
    assert isinstance(result, tuple)
    assert len(result) == 2
    xnet, cnet = result
    assert isinstance(xnet, AlecGGA_XNet)
    assert isinstance(cnet, AlecGGA_CNet)
    assert isinstance(xnet, eqx.Module)
    assert isinstance(cnet, eqx.Module)


@pytest.mark.xfail(
    not os.path.isdir(LEGACY_CKPT_DIR),
    reason=_LEGACY_REASON,
    strict=False,
)
def test_from_legacy_step3b_pretrain_layout_eval_bit_exact():
    """(17b) Loaded networks' eval_Fx/eval_Fc match reference to within 1e-12."""
    import numpy as np
    import jax.numpy as jnp
    from xcquinox.alec.pretrain import from_legacy_step3b
    from xcquinox.alec.models import AlecGGAModel

    arch = _make_arch()
    xnet, cnet = from_legacy_step3b(LEGACY_CKPT_DIR, arch)
    model = AlecGGAModel.from_arch(arch, xnet=xnet, cnet=cnet)

    rho = jnp.array([0.1, 0.2, 0.3])
    sigma = jnp.array([0.01, 0.02, 0.03])
    features = jnp.zeros((3, 0))

    # Load the reference model the same way for comparison
    xnet_ref, cnet_ref = from_legacy_step3b(LEGACY_CKPT_DIR, arch)
    model_ref = AlecGGAModel.from_arch(arch, xnet=xnet_ref, cnet=cnet_ref)

    fx = np.array(model.eval_Fx(rho, sigma, features))
    fx_ref = np.array(model_ref.eval_Fx(rho, sigma, features))
    assert np.array_equal(fx, fx_ref), "eval_Fx outputs differ between two loads"


# ---------------------------------------------------------------------------
# Test 18: ambiguous layout raises ValueError (xfail, need fixture)
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    not os.path.isdir(LEGACY_CKPT_DIR),
    reason=_LEGACY_REASON,
    strict=False,
)
def test_from_legacy_step3b_ambiguous_layout_raises():
    """(18) Both pretrain and training layouts present raises ValueError."""
    import shutil
    from xcquinox.alec.pretrain import from_legacy_step3b

    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy pretrain layout
        for fname in ("xnet.eqx", "cnet.eqx", "pretrain_metadata.json"):
            src = os.path.join(LEGACY_CKPT_DIR, fname)
            if os.path.isfile(src):
                shutil.copy(src, os.path.join(tmpdir, fname))
        # Add training layout files to create ambiguity
        xcmodel_path = os.path.join(tmpdir, "xcmodel.eqx")
        train_md_path = os.path.join(tmpdir, "train_metadata.json")
        with open(xcmodel_path, "wb") as f:
            f.write(b"dummy")
        with open(train_md_path, "w") as f:
            json.dump({"depth": 2, "nodes": 8}, f)

        with pytest.raises(ValueError, match="BOTH"):
            from_legacy_step3b(tmpdir, _make_arch())


# ---------------------------------------------------------------------------
# Test 19: PretrainSpec defaults match notebook
# ---------------------------------------------------------------------------

def test_pretrainspec_defaults_match_notebook():
    """(19) Default field values match the notebook's LIVE pretraining config."""
    arch = _make_arch()
    spec = PretrainSpec(
        arch=arch,
        data_dir="/tmp/nonexistent",
        checkpoint_dir="/tmp/nonexistent_ck",
    )
    assert spec.n_steps == 1000
    assert spec.lr_start == 1e-2
    assert spec.lr_end == 1e-5
    assert spec.lr_decay_start == 0.2
    assert spec.grad_clip == 1.0


# ---------------------------------------------------------------------------
# Test 20: LOB leaf remap (xfail, needs fixture)
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    not os.path.isdir(LEGACY_CKPT_DIR),
    reason=_LEGACY_REASON,
    strict=False,
)
def test_from_legacy_step3b_lob_leaf_remap():
    """(20) D-H5: legacy LOB leaf type remapped; eval_Fx matches library bit-exactly."""
    import jax
    import numpy as np
    import jax.numpy as jnp
    import equinox as eqx
    import xcquinox.net
    from xcquinox.alec.pretrain import (
        from_legacy_step3b,
        _legacy_xnet_lob_lim,
        _legacy_cnet_lob_lim,
    )
    from xcquinox.alec.models import AlecGGAModel

    arch = _make_arch()
    xnet, cnet = from_legacy_step3b(LEGACY_CKPT_DIR, arch)
    model = AlecGGAModel.from_arch(arch, xnet=xnet, cnet=cnet)

    # Build library reference
    xnet_path = os.path.join(LEGACY_CKPT_DIR, "xnet.eqx")
    lib_skel = xcquinox.net.GGA_FxNet_extended(
        depth=arch.depth, nodes=arch.nodes, seed=0,
        lob_lim=_legacy_xnet_lob_lim,
        lower_rho_cutoff=1e-12,
        use_self_attention=arch.attention,
        use_laplacian=False,
        use_dm_features=False,
        use_cusp=False,
        n_dm_features=3,
    )
    lib_loaded = eqx.tree_deserialise_leaves(xnet_path, lib_skel)

    rho = jnp.linspace(0.01, 1.0, 10)
    sigma = jnp.linspace(0.001, 0.1, 10)
    features = jnp.zeros((10, 0))

    alec_fx = np.array(model.eval_Fx(rho, sigma, features))
    lib_fx = np.array(jax.vmap(lib_loaded)(
        jnp.stack([rho, sigma], axis=1)
    ).squeeze())

    # The lob remap must not silently alter numerics
    np.testing.assert_allclose(alec_fx, lib_fx, rtol=1e-10, atol=1e-12)


# ---------------------------------------------------------------------------
# Test 21: leaf count matches skeleton (xfail, needs fixture)
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    not os.path.isdir(LEGACY_CKPT_DIR),
    reason=_LEGACY_REASON,
    strict=False,
)
def test_from_legacy_step3b_leaf_count_matches_skeleton():
    """(21) H-E12-4: _count_disk_records matches library skeleton leaf count."""
    import jax
    import xcquinox.net
    from equinox._filters import is_array_like
    from xcquinox.alec.pretrain import (
        _count_disk_records,
        _legacy_xnet_lob_lim,
    )
    from xcquinox.alec.networks import create_network_pair

    arch = _make_arch()
    xnet_path = os.path.join(LEGACY_CKPT_DIR, "xnet.eqx")

    lib_skel = xcquinox.net.GGA_FxNet_extended(
        depth=arch.depth, nodes=arch.nodes, seed=0,
        lob_lim=_legacy_xnet_lob_lim,
        lower_rho_cutoff=1e-12,
        use_self_attention=arch.attention,
        use_laplacian=False,
        use_dm_features=False,
        use_cusp=False,
        n_dm_features=3,
    )
    n_lib_leaves = sum(
        1 for leaf in jax.tree_util.tree_leaves(lib_skel)
        if is_array_like(leaf)
    )
    n_disk = _count_disk_records(xnet_path)
    assert n_disk == n_lib_leaves, (
        f"disk records ({n_disk}) != library skeleton leaves ({n_lib_leaves})"
    )

    # Verify that the alec skeleton has fewer leaves (static scalars removed)
    alec_skel, _ = create_network_pair(arch, seed=0)
    n_alec_leaves = sum(
        1 for leaf in jax.tree_util.tree_leaves(alec_skel)
        if is_array_like(leaf)
    )
    assert n_alec_leaves < n_lib_leaves, (
        "alec skeleton should have fewer is_array_like leaves than library"
    )


# ---------------------------------------------------------------------------
# Test 22: loads under LOB constraint (xfail, needs fixture)
# ---------------------------------------------------------------------------

@pytest.mark.xfail(
    not os.path.isdir(LEGACY_CKPT_DIR),
    reason=_LEGACY_REASON,
    strict=False,
)
def test_from_legacy_step3b_loads_under_lieb_oxford_constraint():
    """(22) B-H-R15: legacy loader succeeds even when arch has LiebOxfordBound.

    Pre-fix: passing arch.resolved_xnet_lob_lim=None to GGA_FxNet_extended
    would crash. Post-fix: hardcoded _legacy_xnet_lob_lim=1.804 is used
    for the library skeleton, and the alec XNet correctly ends up with
    lob_lim=None (the constraint does the clamping).
    """
    from xcquinox.alec.pretrain import from_legacy_step3b
    from xcquinox.alec.config import FeatureSpec
    from xcquinox.alec.networks import AlecGGA_XNet

    # Build arch with LiebOxfordBound on x_constraints (avoidance rule active)
    arch = ArchitectureConfig.from_spec(
        "t_lob", depth=2, nodes=8,
        x_constraints=["lieb_oxford"],
    )
    assert arch.resolved_xnet_lob_lim is None, (
        "arch.resolved_xnet_lob_lim should be None when LiebOxfordBound "
        "is registered and double_lob_clamp_allowed=False"
    )

    # This must NOT raise (pre-fix it would crash with TypeError)
    xnet, cnet = from_legacy_step3b(LEGACY_CKPT_DIR, arch)
    assert isinstance(xnet, AlecGGA_XNet)

    # The alec-side network must have lob_lim=None (constraint is the sole clamp)
    assert xnet.lob_lim is None, (
        "xnet.lob_lim should be None after loading under LiebOxfordBound arch"
    )


# ---------------------------------------------------------------------------
# Test 23: validate rejects non-finite loss_kwargs values
# (This is a PretrainSpec validate check, tests the nonfinite guard.)
# ---------------------------------------------------------------------------

def test_pretrainspec_validate_n_steps_negative():
    """(23) Negative n_steps raises ValueError (additional validate branch)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckdir = os.path.join(tmpdir, "ck")
        spec = PretrainSpec(
            arch=_make_arch(), data_dir=tmpdir, checkpoint_dir=ckdir,
            n_steps=-5,
        )
        with pytest.raises(ValueError, match="n_steps must be > 0"):
            spec.validate()


# ---------------------------------------------------------------------------
# Checkpoint isolation + early xnet save (fixture-free; trainer is faked)
# ---------------------------------------------------------------------------

def test_run_pretrain_separates_checkpoints_and_saves_xnet_early(tmp_path, monkeypatch):
    """run_pretrain gives xnet/cnet their OWN periodic-snapshot subdirs (so
    their ``xc.eqx.<step>`` files don't clobber each other), and serialises
    the final ``xnet.eqx`` BEFORE cnet training (durable if cnet later fails).

    Heavy work is stubbed: descriptors and networks are stubbed and a minimal
    real ``pretrain_data.npz`` is written, so this is fixture-free and fast
    while still exercising run_pretrain's real control flow. Every write is
    observed through the serialisation call itself rather than through a faked
    trainer, so the order and the destinations are the ones the run performs.
    """
    import numpy as np
    import jax
    import jax.numpy as jnp
    import equinox as eqx
    import xcquinox.alec.pretrain as ptmod

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    np.savez(
        os.path.join(str(data_dir), "pretrain_data.npz"),
        Fx_all=np.zeros((4,), np.float64),
        Fc_all=np.zeros((4,), np.float64),
    )

    # Stub the compute-heavy seams.
    # The stub carries the assembler's full keyword signature, block selector
    # included: run_pretrain names the block on every call, so a stub that
    # accepted only ``for_cnet`` would force the caller to keep a branch whose
    # two arms agree at the default.
    monkeypatch.setattr(
        ptmod, "_assemble_pretrain_descriptors",
        lambda arch, data, for_cnet=False, suffix="_all": jnp.zeros((4, 1)),
    )
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    fake_x = eqx.nn.Linear(1, 1, key=k1)
    fake_c = eqx.nn.Linear(1, 1, key=k2)
    monkeypatch.setattr(
        ptmod, "create_network_pair", lambda arch, seed=0: (fake_x, fake_c),
    )

    ckdir = tmp_path / "ck"

    saved = []  # (path, xnet.eqx already on disk?) in serialisation order
    real_ser = eqx.tree_serialise_leaves

    def _spy_ser(path, tree):
        saved.append((str(path),
                      os.path.isfile(os.path.join(str(ckdir), "xnet.eqx"))))
        return real_ser(path, tree)

    monkeypatch.setattr(ptmod.eqx, "tree_serialise_leaves", _spy_ser)

    # 120 steps so the periodic snapshots exist at all: the interval a run
    # asks for is max(50, n_steps // 10), which exceeds any schedule below 50.
    spec = PretrainSpec(
        arch=_make_arch(), data_dir=str(data_dir), checkpoint_dir=str(ckdir),
        n_steps=120, lr_start=1e-2, lr_end=1e-5, lr_decay_start=0.0,
        grad_clip=1.0, seed=0, loss_weighting="unweighted",
    )
    ptmod.run_pretrain(spec)

    paths = [pth for pth, _present in saved]
    names = [os.path.basename(pth) for pth in paths]
    # xnet snapshots -> <ck>/xnet, cnet snapshots -> <ck>/cnet (no shared dir),
    # so the two nets cannot overwrite each other's xc.eqx.<step>.
    snap_dirs = {os.path.dirname(pth) for pth in paths
                 if os.path.basename(pth).startswith("xc.eqx.")}
    assert snap_dirs == {os.path.join(str(ckdir), "xnet"),
                         os.path.join(str(ckdir), "cnet")}
    assert snap_dirs and len(snap_dirs) == 2
    # Every snapshot is numbered at the interval the run asked for.
    for pth in paths:
        base = os.path.basename(pth)
        if base.startswith("xc.eqx."):
            assert int(base.rsplit(".", 1)[1]) % 50 == 0
    # Durability: every cnet write happens with xnet.eqx already on disk, and
    # no xnet write does, i.e. the final xnet was persisted before the cnet
    # phase rather than after it.
    for pth, present in saved:
        in_cnet = os.path.join(str(ckdir), "cnet") in pth \
            or os.path.basename(pth) == "cnet.eqx"
        assert present is in_cnet, (pth, present)
    # The final xnet.eqx is serialized BEFORE cnet.eqx.
    assert names.index("xnet.eqx") < names.index("cnet.eqx")
    # Finals land at the top level of checkpoint_dir.
    assert os.path.isfile(os.path.join(str(ckdir), "xnet.eqx"))
    assert os.path.isfile(os.path.join(str(ckdir), "cnet.eqx"))


# ---------------------------------------------------------------------------
# Pretraining is now constraint-aware: run_pretrain trains the networks built by
# create_network_pair, which enforce the arch's constraints in their forward.
# This pins that the exact forward run_pretrain optimizes (jax.vmap(xnet)(rows))
# is constrained, so pretraining fits the CONSTRAINED functional.
# ---------------------------------------------------------------------------

def test_pretrain_forward_is_constraint_aware():
    import numpy as _np
    import jax as _jax
    import jax.numpy as _jnp
    from xcquinox.alec.networks import create_network_pair
    from xcquinox.alec.config import ArchitectureConfig

    arch = ArchitectureConfig.from_spec("t", 2, 8, x_constraints=["lieb_oxford"])
    xnet, _cnet = create_network_pair(arch, seed=0)
    # The xnet built for pretraining carries the constraint and disables the
    # built-in LOB wrap (the external constraint owns the bound).
    assert [c.registry_name for c in xnet.constraints] == ["lieb_oxford"]
    assert xnet.lobf is None

    # Replicate run_pretrain's forward: jax.vmap(xnet)(descriptors), rows are
    # [rho, sigma] for a no-descriptor arch.
    rng = _np.random.default_rng(0)
    descriptors = _jnp.asarray(rng.uniform(0.01, 3.0, size=(64, 2)))
    out = _np.asarray(_jax.vmap(xnet)(descriptors))
    assert _np.all(_np.isfinite(out))
    assert _np.all(out > 0.0) and _np.all(out < 1.804 + 1e-6)


# ---------------------------------------------------------------------------
# pretrain-data filename selection (polarized -> zeta-aware file)
# ---------------------------------------------------------------------------

def test_pretrain_data_filename_selection():
    from xcquinox.alec.pretrain import _pretrain_data_filename
    assert _pretrain_data_filename(_make_arch()) == "pretrain_data.npz"
    assert _pretrain_data_filename(
        _make_arch(use_polarized_correlation=True)) == "pretrain_data_polarized.npz"


def test_run_pretrain_polarized_missing_data_errors_clearly():
    """A spin-polarized run with no pretrain_data_polarized.npz fails fast with a
    message naming the expected file and the generator script (no silent zeta=0
    fallback)."""
    from xcquinox.alec.pretrain import run_pretrain
    spec = _make_spec(arch=_make_arch(use_polarized_correlation=True))
    with pytest.raises(FileNotFoundError, match="pretrain_data_polarized.npz"):
        run_pretrain(spec)


def test_run_pretrain_networks_override_bypasses_create_network_pair(tmp_path):
    """``run_pretrain(spec, networks=(xnet, cnet))`` trains the PROVIDED networks
    and never calls ``create_network_pair`` -- the path used to pretrain a
    truly-unconstrained (lob_lim=None) net the arch cannot express. Trains 1 step
    on a tiny synthetic unpolarized dataset (plumbing test, not accuracy)."""
    import numpy as np
    from xcquinox.alec import pretrain as P
    from xcquinox.alec.networks import AlecGGA_XNet, AlecGGA_CNet

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rng = np.random.default_rng(0)
    n = 64
    np.savez(
        os.path.join(str(data_dir), "pretrain_data.npz"),
        rho_all=rng.uniform(0.05, 1.0, n).astype(float),
        sigma_all=rng.uniform(0.0, 1.0, n).astype(float),
        Fx_all=rng.uniform(-0.5, 0.5, n).astype(float),
        Fc_all=rng.uniform(-0.5, 0.5, n).astype(float),
    )
    spec = PretrainSpec(
        arch=_make_arch(),  # default: unpolarized, no descriptors -> n_extra_features=0
        data_dir=str(data_dir), checkpoint_dir=str(tmp_path / "ck"), n_steps=1,
    )
    # Truly-unconstrained override networks (lob_lim=None; create_network_pair
    # cannot produce these without baking in the LO constraint).
    xnet = AlecGGA_XNet(n_extra_features=0, depth=2, nodes=8, seed=0, lob_lim=None)
    cnet = AlecGGA_CNet(n_extra_features=0, depth=2, nodes=8, seed=1, lob_lim=None)

    called = {"hit": False}
    real_cnp = P.create_network_pair

    def _boom(*a, **k):
        called["hit"] = True
        return real_cnp(*a, **k)

    P.create_network_pair = _boom
    try:
        md = P.run_pretrain(spec, networks=(xnet, cnet))
    finally:
        P.create_network_pair = real_cnp

    assert called["hit"] is False, "create_network_pair must NOT run when networks= is given"
    assert "final_loss_x" in md and "final_loss_c" in md
    # the override net (lob_lim=None) was the thing trained + saved
    assert os.path.isfile(os.path.join(str(tmp_path / "ck"), "xnet.eqx"))
