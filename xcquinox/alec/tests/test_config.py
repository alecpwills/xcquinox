import pytest
from xcquinox.alec.config import ArchitectureConfig, FeatureSpec, _FrozenDict


# §13.2 item (2)
def test_feature_spec_of_from_string():
    fs = FeatureSpec.of("cusp")
    assert fs.name == "cusp"
    assert dict(fs.kwargs) == {}


# §13.2 item (3)
def test_feature_spec_of_from_tuple():
    fs = FeatureSpec.of(("lieb_oxford", {"limit": 1.804}))
    assert fs.name == "lieb_oxford"
    assert dict(fs.kwargs) == {"limit": 1.804}


# §13.2 item (4)
def test_feature_spec_of_from_feature_spec():
    original = FeatureSpec(name="x", kwargs=_FrozenDict([("k", 1)]))
    round_tripped = FeatureSpec.of(original)
    assert round_tripped is original


# §13.2 item (5)
def test_feature_spec_of_rejects_unknown_type():
    with pytest.raises(TypeError):
        FeatureSpec.of(42)


# §13.2 item (6)
def test_feature_spec_of_rejects_bad_tuple():
    with pytest.raises(TypeError):
        FeatureSpec.of(("x", [1, 2, 3]))


# §13.2 item (7)
def test_feature_spec_dict_kwarg_roundtrip():
    fs = FeatureSpec.of(("x", {"a": 1, "b": 2, "nested": {"c": 3}}))
    out = fs.as_kwargs()
    assert isinstance(out, dict)
    assert out == {"a": 1, "b": 2, "nested": {"c": 3}}
    assert isinstance(out["nested"], dict)


# §13.2 item (8)
def test_feature_spec_list_pairs_kwarg_roundtrip():
    fs = FeatureSpec.of(("x", {"pairs": [("k", 1), ("v", 2)]}))
    out = fs.as_kwargs()
    assert isinstance(out["pairs"], list)
    assert out["pairs"] == [("k", 1), ("v", 2)]


# §13.2 item (9)
def test_feature_spec_equal_hash_insensitive_to_order():
    fs_a = FeatureSpec.of(("x", {"a": 1, "b": 2}))
    fs_b = FeatureSpec.of(("x", {"b": 2, "a": 1}))
    assert fs_a == fs_b
    assert hash(fs_a) == hash(fs_b)


# --- §13.2 item (1) ---------------------------------------------------------

def _valid_base_kwargs():
    return dict(
        name="test_arch",
        depth=2,
        nodes=8,
        attention=False,
        descriptors=(),
        x_constraints=(),
        c_constraints=(),
        double_lob_clamp_allowed=False,
    )


@pytest.mark.parametrize(
    "field, value, exc",
    [
        # name
        ("name", "", ValueError),
        ("name", 123, TypeError),
        # depth
        ("depth", 0, ValueError),
        ("depth", -1, ValueError),
        ("depth", True, TypeError),
        ("depth", 1.0, TypeError),
        # nodes
        ("nodes", 0, ValueError),
        ("nodes", -5, ValueError),
        ("nodes", True, TypeError),
        ("nodes", 2.0, TypeError),
        # attention
        ("attention", "yes", TypeError),
        ("attention", 1, TypeError),
        # double_lob_clamp_allowed
        ("double_lob_clamp_allowed", "yes", TypeError),
        # descriptors
        ("descriptors", ("cusp",), TypeError),
        ("descriptors", ("cusp", FeatureSpec(name="cusp", kwargs=_FrozenDict(()))), TypeError),
        ("descriptors", [FeatureSpec(name="cusp", kwargs=_FrozenDict(()))], TypeError),
        # x_constraints
        ("x_constraints", ("ueg_limit",), TypeError),
        ("x_constraints", [FeatureSpec(name="ueg_limit", kwargs=_FrozenDict(()))], TypeError),
        # c_constraints
        ("c_constraints", ("ueg_limit",), TypeError),
        ("c_constraints", [FeatureSpec(name="ueg_limit", kwargs=_FrozenDict(()))], TypeError),
        # positive path
        (None, None, None),
    ],
)
def test_architecture_config_field_validation(field, value, exc):
    """§13.2 item (1): parametrized over every __post_init__ branch."""
    if field is None:
        cfg = ArchitectureConfig(
            name="deep_combined_attn",
            depth=4,
            nodes=32,
            attention=True,
            descriptors=(
                FeatureSpec(name="dm_statistics", kwargs=_FrozenDict(())),
                FeatureSpec(name="cusp", kwargs=_FrozenDict(())),
            ),
            x_constraints=(),
            c_constraints=(),
            double_lob_clamp_allowed=False,
        )
        assert cfg.name == "deep_combined_attn"
        assert cfg.attention is True
        assert len(cfg.descriptors) == 2
        return
    kwargs = _valid_base_kwargs()
    kwargs[field] = value
    with pytest.raises(exc):
        ArchitectureConfig(**kwargs)


# --- §13.2 items (12)-(15), (17), Task 1.3 --------------------------------

# §13.2 item (12), 2026-05-29: bumped from 12 to 14 by adding
# deep_notransform + deep_notransform_attn for the descriptor ablation sweep.
def test_architectures_has_14_keys():
    from xcquinox.alec.config import ARCHITECTURES
    assert len(ARCHITECTURES) == 14
    expected_keys = {
        "shallow", "shallow_attn", "medium", "medium_attn",
        "deep", "deep_attn", "deep_cusp", "deep_cusp_attn",
        "deep_dm", "deep_dm_attn", "deep_combined", "deep_combined_attn",
        # New 2026-05-29 entries, no DM/Cusp descriptors, Dick log-transform
        # explicitly disabled, for ablation against the 6 standard archs.
        "deep_notransform", "deep_notransform_attn",
    }
    assert set(ARCHITECTURES.keys()) == expected_keys


# §13.2 item (13)
def test_get_architecture_raises_for_unknown():
    from xcquinox.alec.config import get_architecture
    with pytest.raises(KeyError):
        get_architecture("nonexistent")


# §13.2 item (14)
def test_list_architectures_returns_sorted():
    from xcquinox.alec.config import list_architectures
    names = list_architectures()
    assert names == sorted(names)
    assert len(names) == 14


# §13.2 item (15)
def test_architectures_match_notebook_reference():
    from xcquinox.alec.config import ARCHITECTURES
    from xcquinox.alec.tests.fixtures.notebook_reference import NOTEBOOK_ARCHITECTURES
    assert set(ARCHITECTURES.keys()) == set(NOTEBOOK_ARCHITECTURES.keys())
    for name, expected in NOTEBOOK_ARCHITECTURES.items():
        actual = ARCHITECTURES[name]
        assert actual.name == expected["name"]
        assert actual.depth == expected["depth"]
        assert actual.nodes == expected["nodes"]
        assert actual.attention is expected["attention"]
        actual_descr_names = [d.name for d in actual.descriptors]
        assert actual_descr_names == expected["descriptors"]


# §13.2 item (17)
def test_architecture_config_from_spec_equals_direct_construction():
    via_factory = ArchitectureConfig.from_spec(
        "deep_combined",
        4, 32,
        descriptors=["dm_statistics", "cusp"],
    )
    via_direct = ArchitectureConfig(
        name="deep_combined",
        depth=4,
        nodes=32,
        attention=False,
        descriptors=(
            FeatureSpec(name="dm_statistics", kwargs=_FrozenDict(())),
            FeatureSpec(name="cusp", kwargs=_FrozenDict(())),
        ),
        x_constraints=(),
        c_constraints=(),
        double_lob_clamp_allowed=False,
    )
    assert via_factory == via_direct
    assert via_factory.descriptors == via_direct.descriptors


# --- §13.2 items (10)-(11), Task 1.5 step 6 --------------------------------

# §13.2 item (10)
def test_architecture_n_input_features_arithmetic():
    from xcquinox.alec.config import get_architecture
    zero = get_architecture("shallow")
    one_cusp = get_architecture("deep_cusp")
    one_dm = get_architecture("deep_dm")
    two = get_architecture("deep_combined")
    assert zero.n_input_features == 2
    assert one_cusp.n_input_features == 2 + 2
    assert one_dm.n_input_features == 2 + 3
    assert two.n_input_features == 2 + 3 + 2


# §13.2 item (11)
def test_architecture_materialize_roundtrip_returns_registry_instances():
    from xcquinox.alec.config import get_architecture
    from xcquinox.alec.descriptors import CuspDescriptor, DMStatisticsDescriptor
    from xcquinox.alec.constraints import LiebOxfordBound, UEGLimit

    deep_combined = get_architecture("deep_combined")
    descr = deep_combined.materialize_descriptors()
    assert isinstance(descr, tuple)
    assert len(descr) == 2
    assert isinstance(descr[0], DMStatisticsDescriptor)
    assert isinstance(descr[1], CuspDescriptor)

    arch_with_constraints = ArchitectureConfig.from_spec(
        "deep_lob_ueg",
        4, 32,
        descriptors=["cusp"],
        x_constraints=["lieb_oxford"],
        c_constraints=["ueg_limit"],
    )
    xcs = arch_with_constraints.materialize_x_constraints()
    ccs = arch_with_constraints.materialize_c_constraints()
    assert isinstance(xcs, tuple) and len(xcs) == 1
    assert isinstance(ccs, tuple) and len(ccs) == 1
    assert isinstance(xcs[0], LiebOxfordBound)
    assert isinstance(ccs[0], UEGLimit)


# --- §13.2 items (18)-(19), Task 1.6 ----------------------------------------

# §13.2 item (18)
def test_trainingspec_describe_json_serializes_with_all_fields():
    import json
    import dataclasses
    from xcquinox.alec.config import TrainingSpec, MoleculeSpec, get_architecture

    mols = (
        MoleculeSpec.from_dict(
            name="H", atom="H 0 0 0",
            basis="sto-3g", charge=0, spin=1,
            atom_composition={"H": 1},
        ),
        MoleculeSpec.from_dict(
            name="O", atom="O 0 0 0",
            basis="sto-3g", charge=0, spin=2,
            atom_composition={"O": 1},
        ),
        MoleculeSpec.from_dict(
            name="H2O", atom="O 0 0 0; H 0 0 0.96; H 0.93 0 -0.24",
            basis="sto-3g", charge=0, spin=0,
            atom_composition={"H": 2, "O": 1},
        ),
    )
    spec = TrainingSpec(
        arch=get_architecture("deep_combined"),
        molecules=mols,
        targets=(("H", 0.0), ("H2O", 232.0), ("O", 0.0)),
        atom_energies=(("H", -0.5), ("O", -75.0)),
        loss_name="A_atomization",
        loss_kwargs=(),
        n_steps=10,
        lr_start=1e-3,
        lr_end=1e-5,
        lr_decay_start=0.0,
        grad_clip=1.0,
        pretrain_checkpoint=None,
        checkpoint_dir="/tmp/alec_nonexistent_ckpt_dir",
        seed=0,
    )
    out = spec.describe()
    assert isinstance(out, dict)
    field_names = {f.name for f in dataclasses.fields(spec)}
    assert field_names == set(out.keys()), (
        f"describe() field-set mismatch: missing={field_names - set(out.keys())}, "
        f"extra={set(out.keys()) - field_names}"
    )
    assert out["arch"] == "deep_combined"
    assert out["molecules"] == ["H", "O", "H2O"]
    json.dumps(out)


# §13.2 item (19)
def test_pretrainspec_describe_json_serializes_with_all_fields():
    import json
    import dataclasses
    from xcquinox.alec.config import PretrainSpec, get_architecture

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
    assert out["arch"] == "deep_combined"
    json.dumps(out)


# --- §13.2 item (16), Task 2.2 step 6 ----------------------------------------

# §13.2 item (16)
def test_architectures_all_materialize_via_from_arch():
    from xcquinox.alec.config import ARCHITECTURES
    from xcquinox.alec.models import AlecGGAModel
    assert len(ARCHITECTURES) == 14  # 2026-05-29: +deep_notransform, +deep_notransform_attn
    for arch_name, arch in ARCHITECTURES.items():
        try:
            model = AlecGGAModel.from_arch(arch, seed=0)
        except Exception as exc:
            raise AssertionError(
                f"AlecGGAModel.from_arch failed for {arch_name!r}: {exc}"
            ) from exc
        assert model is not None
        assert len(model.descriptors) == len(arch.descriptors), (
            f"{arch_name!r} descriptor tuple arity drift"
        )
        assert len(model.x_constraints) == len(arch.x_constraints), (
            f"{arch_name!r} x_constraints tuple arity drift"
        )
        assert len(model.c_constraints) == len(arch.c_constraints), (
            f"{arch_name!r} c_constraints tuple arity drift"
        )


def test_training_spec_default_solver_config_is_none():
    from xcquinox.alec.config import TrainingSpec, ArchitectureConfig, MoleculeSpec

    arch = ArchitectureConfig(
        name="t", depth=2, nodes=8, attention=False,
        descriptors=(), x_constraints=(), c_constraints=(),
        double_lob_clamp_allowed=False,
    )
    mols = (MoleculeSpec(
        name="H", atom="H 0 0 0", basis="sto-3g",
        charge=0, spin=1, atom_composition=(("H", 1),),
    ),)
    spec = TrainingSpec(
        arch=arch, molecules=mols,
        targets=(("H", 0.0),), atom_energies=(("H", -0.5),),
        loss_name="A_atomization",
    )
    assert spec.solver_config is None


def test_test_spec_default_solver_config_is_none():
    from xcquinox.alec.config import TestSpec, ArchitectureConfig, MoleculeSpec

    arch = ArchitectureConfig(
        name="t", depth=2, nodes=8, attention=False,
        descriptors=(), x_constraints=(), c_constraints=(),
        double_lob_clamp_allowed=False,
    )
    mols = (MoleculeSpec(
        name="H", atom="H 0 0 0", basis="sto-3g",
        charge=0, spin=1, atom_composition=(("H", 1),),
    ),)
    import tempfile, os
    with tempfile.NamedTemporaryFile(delete=False, suffix=".eqx") as f:
        f.write(b"stub")
        path = f.name
    try:
        spec = TestSpec(
            model_checkpoint=path, arch=arch, molecules=mols,
        )
        assert spec.solver_config is None
    finally:
        os.unlink(path)


# --- PretrainSpec.loss_weighting (physics-fixes Task 1) -----------------------

def test_pretrain_spec_loss_weighting_default_unweighted():
    from xcquinox.alec.config import PretrainSpec
    import xcquinox.alec as alec
    spec = PretrainSpec(
        arch=alec.get_architecture("deep"),
        data_dir="/tmp/x",
        checkpoint_dir="/tmp/y",
        n_steps=10,
    )
    assert spec.loss_weighting == "unweighted"


def test_pretrain_spec_loss_weighting_integration_accepted():
    from xcquinox.alec.config import PretrainSpec
    import xcquinox.alec as alec
    spec = PretrainSpec(
        arch=alec.get_architecture("deep"),
        data_dir="/tmp/x",
        checkpoint_dir="/tmp/y",
        n_steps=10,
        loss_weighting="integration",
    )
    assert spec.loss_weighting == "integration"


def test_pretrain_spec_loss_weighting_invalid_raises():
    from xcquinox.alec.config import PretrainSpec
    import xcquinox.alec as alec
    with pytest.raises((ValueError, TypeError)):
        PretrainSpec(
            arch=alec.get_architecture("deep"),
            data_dir="/tmp/x",
            checkpoint_dir="/tmp/y",
            n_steps=10,
            loss_weighting="foo",
        )


def test_pretrain_spec_loss_weighting_in_describe():
    from xcquinox.alec.config import PretrainSpec
    import xcquinox.alec as alec
    spec = PretrainSpec(
        arch=alec.get_architecture("deep"),
        data_dir="/tmp/x",
        checkpoint_dir="/tmp/y",
        n_steps=10,
        loss_weighting="integration",
    )
    out = spec.describe()
    assert out["loss_weighting"] == "integration"


# ---------------------------------------------------------------------------
# Step-6 Task 3.1: PBE-anchor pass-through fields on TrainingSpec / TestSpec
# ---------------------------------------------------------------------------

def test_training_spec_accepts_pbe_anchor_fields():
    """TrainingSpec.pbe_anchor_weight + pbe_anchor_sample are declared fields."""
    from xcquinox.alec.config import TrainingSpec
    import dataclasses
    field_names = {f.name for f in dataclasses.fields(TrainingSpec)}
    assert "pbe_anchor_weight" in field_names
    assert "pbe_anchor_sample" in field_names


def test_training_spec_defaults_anchor_to_zero_and_none():
    """Constructing a TrainingSpec without anchor kwargs yields (0.0, None)."""
    from xcquinox.alec.config import TrainingSpec, ArchitectureConfig
    from xcquinox.alec.tests.fixtures.molecules import h_atom
    arch = ArchitectureConfig(
        name="tiny", depth=2, nodes=8, attention=False,
        descriptors=(), x_constraints=(), c_constraints=(),
        double_lob_clamp_allowed=False,
    )
    spec = TrainingSpec(
        arch=arch, molecules=(h_atom(),),
        targets=(("H", 0.0),),
        atom_energies=(("H", -0.5),),
        loss_name="A_atomization",
    )
    assert spec.pbe_anchor_weight == 0.0
    assert spec.pbe_anchor_sample is None


def test_test_spec_accepts_pbe_anchor_fields():
    from xcquinox.alec.config import TestSpec
    import dataclasses
    field_names = {f.name for f in dataclasses.fields(TestSpec)}
    assert "pbe_anchor_weight" in field_names
    assert "pbe_anchor_sample" in field_names


# ---------------------------------------------------------------------------
# Self-attention registry tests (spec §Tests 20-21)
# ---------------------------------------------------------------------------

def test_attn_registry_entries_have_valid_num_heads():
    """Test 20: each *_attn arch satisfies divisibility + head_dim >= 4."""
    from xcquinox.alec.config import ARCHITECTURES
    attn_keys = [k for k in ARCHITECTURES if k.endswith("_attn")]
    # 2026-05-29: bumped 6 -> 7 with `deep_notransform_attn`.
    assert len(attn_keys) == 7, f"expected 7 attn archs, got {len(attn_keys)}"
    for k in attn_keys:
        arch = ARCHITECTURES[k]
        assert arch.attention is True, k
        assert arch.num_heads >= 1, k
        assert arch.nodes % arch.num_heads == 0, (
            f"{k}: nodes={arch.nodes} not divisible by num_heads="
            f"{arch.num_heads}"
        )
        head_dim = arch.nodes // arch.num_heads
        assert head_dim >= 4, (
            f"{k}: head_dim={head_dim} < 4 (registry value violates spec)"
        )


def test_registry_smoke_forward_each_attn_arch():
    """Test 21: every *_attn arch builds and runs a forward pass."""
    import jax.numpy as jnp
    from xcquinox.alec.config import ARCHITECTURES
    from xcquinox.alec.networks import create_network_pair

    attn_keys = [k for k in ARCHITECTURES if k.endswith("_attn")]
    for k in attn_keys:
        arch = ARCHITECTURES[k]
        xnet, cnet = create_network_pair(arch, seed=0)
        n_extra = sum(d.n_features for d in arch.materialize_descriptors())
        # input layout: rho, sigma, then n_extra zeros
        inputs = jnp.array([1.0, 1.0] + [0.0] * n_extra)
        out_x = xnet(inputs)
        out_c = cnet(inputs)
        assert jnp.isfinite(out_x), f"{k}: xnet produced non-finite"
        assert jnp.isfinite(out_c), f"{k}: cnet produced non-finite"


# ---------------------------------------------------------------------------
# validate() must catch elements missing from atom_energies even when
# require_atom_anchors=False (the 2026-05-07 mixed-pool path).
# ---------------------------------------------------------------------------

def _tiny_arch():
    from xcquinox.alec.config import ArchitectureConfig
    return ArchitectureConfig(
        name="tiny", depth=2, nodes=8, attention=False,
        descriptors=(), x_constraints=(), c_constraints=(),
        double_lob_clamp_allowed=False,
    )


def test_validate_missing_element_in_atom_energies_require_anchors_false():
    """compound references C which is absent from atom_energies;
    validate() must raise ValueError naming C, even with require_atom_anchors=False."""
    import tempfile
    from xcquinox.alec.config import TrainingSpec, MoleculeSpec

    h_atom = MoleculeSpec(
        name="H", atom="H 0 0 0", basis="sto-3g",
        charge=0, spin=1, atom_composition=(("H", 1),),
    )
    # Compound with C and H; atom_energies only covers H.
    ch4 = MoleculeSpec(
        name="CH4", atom="C 0 0 0; H 0.63 0.63 0.63; H -0.63 -0.63 0.63; "
                        "H -0.63 0.63 -0.63; H 0.63 -0.63 -0.63",
        basis="sto-3g", charge=0, spin=0,
        atom_composition=(("C", 1), ("H", 4)),
    )
    with tempfile.TemporaryDirectory() as ckpt_dir:
        spec = TrainingSpec(
            arch=_tiny_arch(),
            molecules=(h_atom, ch4),
            targets=(("H", 0.0), ("CH4", -100.0)),
            # atom_energies covers H but NOT C
            atom_energies=(("H", -0.5),),
            loss_name="A_atomization",
            checkpoint_dir=ckpt_dir,
            require_atom_anchors=False,
        )
        with pytest.raises(ValueError, match="C"):
            spec.validate()


def test_validate_cl_compound_passes_with_bh76w411_anchors():
    """CFG-01 positive counterpart / regression for preflight 54403:
    a Cl-containing compound (HCl), which aborted the BH76+W4-11 cluster
    preflight when the bh76w411_step7 anchor table lacked Cl, must now
    validate cleanly against the extended Chakravorty anchors.
    """
    import tempfile
    from xcquinox.alec.config import TrainingSpec, MoleculeSpec
    from xcquinox.alec.cluster.domain import get_domain_profile

    prof = get_domain_profile("bh76w411_step7")
    hcl = MoleculeSpec(
        name="HCl", atom="H 0 0 0; Cl 0 0 1.275", basis="def2-svp",
        charge=0, spin=0, atom_composition=(("H", 1), ("Cl", 1)),
    )
    h2 = MoleculeSpec(
        name="H2", atom="H 0 0 0; H 0 0 0.74", basis="def2-svp",
        charge=0, spin=0, atom_composition=(("H", 2),),
    )
    with tempfile.TemporaryDirectory() as ckpt_dir:
        spec = TrainingSpec.from_dicts(
            arch=_tiny_arch(),
            molecules=(hcl, h2),
            targets={"HCl": -0.17, "H2": -0.17},
            atom_energies=prof.atom_energies,  # now carries Cl (and Be/B/Al/Si/P)
            loss_name="A_atomization",
            checkpoint_dir=ckpt_dir,
            require_atom_anchors=False,
        )
        spec.validate()  # must NOT raise (CFG-01 cleared for Cl)


# ---------------------------------------------------------------------------
# bool values must be rejected from targets and atom_energies even
# though math.isfinite(True) is True.
# ---------------------------------------------------------------------------

def test_validate_bool_in_targets_rejected():
    """True passed as a target energy must raise ValueError."""
    import tempfile
    from xcquinox.alec.config import TrainingSpec, MoleculeSpec

    h_atom = MoleculeSpec(
        name="H", atom="H 0 0 0", basis="sto-3g",
        charge=0, spin=1, atom_composition=(("H", 1),),
    )
    h2 = MoleculeSpec(
        name="H2", atom="H 0 0 0; H 0 0 0.74", basis="sto-3g",
        charge=0, spin=0, atom_composition=(("H", 2),),
    )
    with tempfile.TemporaryDirectory() as ckpt_dir:
        spec = TrainingSpec(
            arch=_tiny_arch(),
            molecules=(h_atom, h2),
            # True instead of a float for H2 target
            targets=(("H", 0.0), ("H2", True)),
            atom_energies=(("H", -0.5),),
            loss_name="A_atomization",
            checkpoint_dir=ckpt_dir,
        )
        with pytest.raises((ValueError, TypeError)):
            spec.validate()


def test_validate_bool_in_atom_energies_rejected():
    """True passed as an atom energy must raise ValueError."""
    import tempfile
    from xcquinox.alec.config import TrainingSpec, MoleculeSpec

    h_atom = MoleculeSpec(
        name="H", atom="H 0 0 0", basis="sto-3g",
        charge=0, spin=1, atom_composition=(("H", 1),),
    )
    h2 = MoleculeSpec(
        name="H2", atom="H 0 0 0; H 0 0 0.74", basis="sto-3g",
        charge=0, spin=0, atom_composition=(("H", 2),),
    )
    with tempfile.TemporaryDirectory() as ckpt_dir:
        spec = TrainingSpec(
            arch=_tiny_arch(),
            molecules=(h_atom, h2),
            targets=(("H", 0.0), ("H2", -1.0)),
            # True instead of a float for atom energy
            atom_energies=(("H", True),),
            loss_name="A_atomization",
            checkpoint_dir=ckpt_dir,
        )
        with pytest.raises((ValueError, TypeError)):
            spec.validate()
