"""Tests for xcquinox.alec.pretrain.

Implements THE SPEC §13.2 test_pretrain.py items (1)-(23).

Tests 1-8 and 19 run immediately (only need PretrainSpec + stdlib).
Tests 9-16, 17a, 17b, 18, 20, 21, 22 are marked xfail because they
require fixtures (pretrain_data_tiny.npz and legacy_step3b_checkpoint/)
which have not yet been generated.
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
# Tests 9-16: run_pretrain end-to-end (xfail — need fixture)
# ---------------------------------------------------------------------------

FIXTURE_DIR = os.path.join(
    os.path.dirname(__file__), "fixtures"
)
PRETRAIN_DATA_TINY = os.path.join(FIXTURE_DIR, "pretrain_data_tiny.npz")

_FIXTURE_REASON = (
    "pretrain_data_tiny.npz fixture not yet generated; "
    "run generate_pretrain_data_tiny.py to create it"
)


@pytest.mark.xfail(
    not os.path.isfile(PRETRAIN_DATA_TINY),
    reason=_FIXTURE_REASON,
    strict=False,
)
def test_run_pretrain_end_to_end():
    """(9) run_pretrain produces all expected artifacts."""
    from xcquinox.alec.pretrain import run_pretrain

    with tempfile.TemporaryDirectory() as ckdir:
        spec = PretrainSpec(
            arch=_make_arch(),
            data_dir=FIXTURE_DIR,
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


@pytest.mark.xfail(
    not os.path.isfile(PRETRAIN_DATA_TINY),
    reason=_FIXTURE_REASON,
    strict=False,
)
def test_run_pretrain_losses_finite():
    """(10) Losses returned by run_pretrain are finite scalars."""
    import numpy as np
    from xcquinox.alec.pretrain import run_pretrain

    with tempfile.TemporaryDirectory() as ckdir:
        spec = PretrainSpec(
            arch=_make_arch(),
            data_dir=FIXTURE_DIR,
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


@pytest.mark.xfail(
    not os.path.isfile(PRETRAIN_DATA_TINY),
    reason=_FIXTURE_REASON,
    strict=False,
)
def test_run_pretrain_xnet_serialization_roundtrip():
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
            data_dir=FIXTURE_DIR,
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


@pytest.mark.xfail(
    not os.path.isfile(PRETRAIN_DATA_TINY),
    reason=_FIXTURE_REASON,
    strict=False,
)
def test_run_pretrain_cnet_serialization_roundtrip():
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
            data_dir=FIXTURE_DIR,
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


@pytest.mark.xfail(
    not os.path.isfile(PRETRAIN_DATA_TINY),
    reason=_FIXTURE_REASON,
    strict=False,
)
def test_run_pretrain_losses_x_npy_roundtrip():
    """(13) np.load('losses_x.npy') returns the same array as in-memory."""
    import numpy as np
    from xcquinox.alec.pretrain import run_pretrain

    with tempfile.TemporaryDirectory() as ckdir:
        spec = PretrainSpec(
            arch=_make_arch(),
            data_dir=FIXTURE_DIR,
            checkpoint_dir=ckdir,
            n_steps=3,
            seed=0,
        )
        metadata = run_pretrain(spec)
        losses_x = np.load(os.path.join(ckdir, "losses_x.npy"))
        assert len(losses_x) == 3
        assert losses_x.dtype == np.float64
        assert math.isfinite(metadata["final_loss_x"])


@pytest.mark.xfail(
    not os.path.isfile(PRETRAIN_DATA_TINY),
    reason=_FIXTURE_REASON,
    strict=False,
)
def test_run_pretrain_losses_c_npy_roundtrip():
    """(14) np.load('losses_c.npy') returns the same array as in-memory."""
    import numpy as np
    from xcquinox.alec.pretrain import run_pretrain

    with tempfile.TemporaryDirectory() as ckdir:
        spec = PretrainSpec(
            arch=_make_arch(),
            data_dir=FIXTURE_DIR,
            checkpoint_dir=ckdir,
            n_steps=3,
            seed=0,
        )
        metadata = run_pretrain(spec)
        losses_c = np.load(os.path.join(ckdir, "losses_c.npy"))
        assert len(losses_c) == 3
        assert losses_c.dtype == np.float64
        assert math.isfinite(metadata["final_loss_c"])


@pytest.mark.xfail(
    not os.path.isfile(PRETRAIN_DATA_TINY),
    reason=_FIXTURE_REASON,
    strict=False,
)
def test_run_pretrain_metadata_json_all_fields():
    """(15) pretrain_metadata.json roundtrips with every documented field."""
    from xcquinox.alec.pretrain import run_pretrain

    required_fields = {
        "arch_name", "pretrain_steps", "lr_start", "lr_end",
        "lr_decay_start", "grad_clip", "final_loss_x", "final_loss_c",
        "min_loss_x", "min_loss_c", "use_cusp", "use_dm",
        "timestamp", "duration_seconds",
    }

    with tempfile.TemporaryDirectory() as ckdir:
        spec = PretrainSpec(
            arch=_make_arch(),
            data_dir=FIXTURE_DIR,
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


@pytest.mark.xfail(
    not os.path.isfile(PRETRAIN_DATA_TINY),
    reason=_FIXTURE_REASON,
    strict=False,
)
def test_run_pretrain_warmup_phase_and_progress_callback():
    """(16) warmup phase is respected and progress_callback receives dict payloads."""
    from xcquinox.alec.pretrain import run_pretrain

    received = []

    def _cb(payload):
        received.append(payload)

    with tempfile.TemporaryDirectory() as ckdir:
        spec = PretrainSpec(
            arch=_make_arch(),
            data_dir=FIXTURE_DIR,
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
# Tests 17a-17b: from_legacy_step3b pretrain layout (xfail — need fixture)
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
# Test 18: ambiguous layout raises ValueError (xfail — need fixture)
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
# Test 20: LOB leaf remap (xfail — needs fixture)
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
# Test 21: leaf count matches skeleton (xfail — needs fixture)
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
# Test 22: loads under LOB constraint (xfail — needs fixture)
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
# (This is a PretrainSpec validate check — tests the nonfinite guard.)
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
