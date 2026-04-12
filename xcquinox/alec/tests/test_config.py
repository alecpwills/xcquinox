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


# --- §13.2 items (12)-(15), (17) — Task 1.3 --------------------------------

# §13.2 item (12)
def test_architectures_has_12_keys():
    from xcquinox.alec.config import ARCHITECTURES
    assert len(ARCHITECTURES) == 12
    expected_keys = {
        "shallow", "shallow_attn", "medium", "medium_attn",
        "deep", "deep_attn", "deep_cusp", "deep_cusp_attn",
        "deep_dm", "deep_dm_attn", "deep_combined", "deep_combined_attn",
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
    assert len(names) == 12


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


# --- §13.2 items (10)-(11) — Task 1.5 step 6 --------------------------------

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


# --- §13.2 items (18)-(19) — Task 1.6 ----------------------------------------

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
