import pytest
from xcquinox.alec.config import FeatureSpec, _FrozenDict


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
