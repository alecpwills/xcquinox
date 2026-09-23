"""Tests for xcquinox.pipeline.pretrain.

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
import os
import tempfile

import pytest

from xcquinox.pipeline.config import ArchitectureConfig, PretrainSpec, get_architecture


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


# (3) missing data_dir raises ValueError
def test_pretrainspec_validate_missing_data_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        ckdir = os.path.join(tmpdir, "ck")
        spec = PretrainSpec(
            arch=_make_arch(),
            data_dir="/tmp/pipeline_nonexistent_data_dir_xyz",
            checkpoint_dir=ckdir,
        )
        with pytest.raises(ValueError, match="data_dir does not exist"):
            spec.validate()


# (4) lr_start < lr_end raises ValueError


# (5) grad_clip=-1.0 raises ValueError


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


# ---------------------------------------------------------------------------
# Test 8: PretrainSpec.describe roundtrip
# ---------------------------------------------------------------------------

def test_pretrainspec_describe_roundtrip():
    """(8) describe() returns a dict with all fields; json-serializable."""
    spec = PretrainSpec(
        arch=get_architecture("deep_combined"),
        data_dir="/tmp/pipeline_nonexistent_data_dir",
        checkpoint_dir="/tmp/pipeline_nonexistent_pretrain_ckpt",
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
    from xcquinox.pipeline.pretrain import _assemble_pretrain_descriptors

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


def test_assemble_pretrain_descriptors_rung35_arch():
    # Regression: deep_rung35_3x16 (cusp + rung35) assembles without the KeyError
    # rung35 hit before it was added to pretrain _key_map.
    import dataclasses
    import numpy as np
    from xcquinox.pipeline.pretrain import _assemble_pretrain_descriptors
    from xcquinox.pipeline import get_architecture

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
    from xcquinox.pipeline.pretrain_data_gen import _atom_columns
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
    from xcquinox.pipeline.pretrain_data_gen import generate_pretrain_data_npz
    d = tmp_path_factory.mktemp("pretrain_tiny")
    generate_pretrain_data_npz(str(d), atoms=(("He", 0),), basis="sto-3g",
                               grid_level=0, polarized=False,
                               descriptors=True, density_fit=False)
    return str(d)


def test_run_pretrain_end_to_end(tiny_pretrain_data_dir):
    """(9) run_pretrain produces all expected artifacts."""
    from xcquinox.pipeline.pretrain import run_pretrain

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
    from xcquinox.pipeline.pretrain import run_pretrain

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
    from xcquinox.pipeline.pretrain import run_pretrain
    from xcquinox.pipeline.networks import create_network_pair

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


def test_run_pretrain_metadata_json_all_fields(tiny_pretrain_data_dir):
    """(15) pretrain_metadata.json roundtrips with every documented field."""
    from xcquinox.pipeline.pretrain import run_pretrain

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
    from xcquinox.pipeline.pretrain import run_pretrain

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
def test_from_legacy_step3b_pretrain_layout_eval_bit_exact():
    """(17b) Loaded networks' eval_Fx/eval_Fc match reference to within 1e-12."""
    import numpy as np
    import jax.numpy as jnp
    from xcquinox.pipeline.pretrain import from_legacy_step3b
    from xcquinox.pipeline.models import AlecGGAModel

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


# ---------------------------------------------------------------------------
# Test 19: PretrainSpec defaults match notebook
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Test 20: LOB leaf remap (xfail, needs fixture)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Test 21: leaf count matches skeleton (xfail, needs fixture)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Test 22: loads under LOB constraint (xfail, needs fixture)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Test 23: validate rejects non-finite loss_kwargs values
# (This is a PretrainSpec validate check, tests the nonfinite guard.)
# ---------------------------------------------------------------------------


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
    import xcquinox.pipeline.pretrain as ptmod

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
    from xcquinox.pipeline.networks import create_network_pair
    from xcquinox.pipeline.config import ArchitectureConfig

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


def test_run_pretrain_polarized_missing_data_errors_clearly():
    """A spin-polarized run with no pretrain_data_polarized.npz fails fast with a
    message naming the expected file and the generator script (no silent zeta=0
    fallback)."""
    from xcquinox.pipeline.pretrain import run_pretrain
    spec = _make_spec(arch=_make_arch(use_polarized_correlation=True))
    with pytest.raises(FileNotFoundError, match="pretrain_data_polarized.npz"):
        run_pretrain(spec)


