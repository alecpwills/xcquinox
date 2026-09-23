import pytest
from xcquinox.pipeline.config import ArchitectureConfig, FeatureSpec, _FrozenDict


# §13.2 item (2)
def test_feature_spec_of_from_string():
    fs = FeatureSpec.of("cusp")
    assert fs.name == "cusp"
    assert dict(fs.kwargs) == {}


# §13.2 item (3)


# §13.2 item (4)


# §13.2 item (5)
def test_feature_spec_of_rejects_unknown_type():
    with pytest.raises(TypeError):
        FeatureSpec.of(42)


# §13.2 item (6)


# §13.2 item (7)


# §13.2 item (8)


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
# 2026-06-20: bumped to 22 by adding the 8 depth-3/width-16 dfs_step7 twins.
# 2026-06-28: bumped to 25 by adding the 3 rung-3.5 localized-DM archs.
# 2026-09-11: bumped to 34 by the width and depth completions of the pure DFS
# meta-GGA (deep_mgga_3x32, deep_mgga_4x16, deep_mgga_4x32). The 3x16 clone of
# SCAN plateaus short of the atom certificate; the three completions ask
# whether that miss is capacity, and they are probes only until it is
# measured -- so they sit on no group's axis and carry no campaign cell, and
# differ from deep_mgga_3x16 in shape alone.
def test_architectures_registry_key_set():
    from xcquinox.pipeline.config import ARCHITECTURES
    assert len(ARCHITECTURES) == 36
    expected_keys = {
        "shallow", "shallow_attn", "medium", "medium_attn",
        "deep", "deep_attn", "deep_cusp", "deep_cusp_attn",
        "deep_dm", "deep_dm_attn", "deep_combined", "deep_combined_attn",
        # New 2026-05-29 entries, no DM/Cusp descriptors, Dick log-transform
        # explicitly disabled, for ablation against the 6 standard archs.
        "deep_notransform", "deep_notransform_attn",
        # New 2026-06-20: depth-3/width-16 capacity-reduction twins.
        "deep_3x16", "deep_attn_3x16", "deep_cusp_3x16", "deep_dm_3x16",
        "deep_combined_3x16", "deep_combined_attn_3x16",
        "deep_notransform_3x16", "deep_notransform_attn_3x16",
        # New 2026-06-28: rung-3.5 localized-DM archs (additive; deep_rung35_3x16
        # = cusp+rung35 replaces deep_combined in the sweep, deep_rung35only_3x16
        # = rung35 alone replaces deep_dm; the leaky entries are kept for in-flight).
        "deep_rung35_3x16", "deep_rung35_attn_3x16", "deep_rung35only_3x16",
        # 2026-08-06: multi-width rung-3.5 (radial NeuralXC-style projection).
        "deep_rung35ms_3x16",
        # New 2026-07-02: DFS-faithful meta-GGA archs (meta_gga=True; iso-orbital
        # alpha descriptor + DFS (x2+tanh^2(x3)) gate + 1.174 LOB; pretrain to SCAN).
        # deep_rung35_mgga_3x16 (cusp+rung35+metagga) replaces deep_rung35only in
        # the dfs6311 sweep; deep_mgga_3x16 is the pure DFS meta-GGA.
        "deep_mgga_3x16", "deep_mgga_attn_3x16", "deep_rung35_mgga_3x16",
        # 2026-08-10: the mgga stacking completions (third sweep arm):
        # cusp+metagga, and cusp+multishell+metagga (SCAN pretrain, no mesh
        # -- geometry-free mesh nodes cannot define their extra columns).
        "deep_cusp_mgga_3x16", "deep_rung35ms_mgga_3x16",
        # 2026-09-11: the width and depth completions of the pure DFS meta-GGA
        # (deep_mgga_3x16 with depth and nodes changed and nothing else), the
        # capacity question of its plateau against the atom certificate.
        # Probe-only until measured: excluded from the v6 campaign.
        "deep_mgga_3x32", "deep_mgga_4x16", "deep_mgga_4x32",
        # The v8 geometric pair: the cusp descriptor's two columns on the 3x16
        # GGA network, with and without attention.
        "deep_geom_3x16", "deep_geom_attn_3x16",
    }
    assert set(ARCHITECTURES.keys()) == expected_keys


# 2026-09-11: the three completions are a capacity experiment, which they are
# only if capacity is all that separates them from deep_mgga_3x16. The
# comparison is over every dataclass field rather than the keywords the
# registry entry spells out, so a flag the entry forgot -- or one added to
# ArchitectureConfig later with a default the three take and their parent does
# not -- turns this red instead of being read as capacity.


def test_the_geometric_pair_is_the_cusp_twin_with_and_without_attention():
    """The two geometric entries are deep_cusp_3x16 under another name, the
    attention twin adding only the attention block.

    Oracle: the registry entry compared field by field against
    ``dataclasses.replace`` of deep_cusp_3x16, so a keyword the entry forgot
    -- or one added to ArchitectureConfig later whose default the pair takes
    and deep_cusp_3x16 does not -- fails here instead of being read as the
    geometric architecture. The three flags and the descriptor list are stated
    again on their own so the failure names which of them moved.
    """
    import dataclasses
    from xcquinox.pipeline.config import get_architecture

    cusp = get_architecture("deep_cusp_3x16")
    plain = get_architecture("deep_geom_3x16")
    attn = get_architecture("deep_geom_attn_3x16")

    want_plain = dataclasses.replace(cusp, name="deep_geom_3x16")
    want_attn = dataclasses.replace(cusp, name="deep_geom_attn_3x16",
                                    attention=True, num_heads=4)
    for got, want in ((plain, want_plain), (attn, want_attn)):
        differing = {f.name: (getattr(got, f.name), getattr(want, f.name))
                     for f in dataclasses.fields(got)
                     if getattr(got, f.name) != getattr(want, f.name)}
        assert not differing, (got.name, differing)
        assert got == want, got.name

    for cfg in (plain, attn):
        assert cfg.depth == 3 and cfg.nodes == 16, cfg.name
        assert cfg.zero_init_final_layer is True, cfg.name
        assert cfg.descriptor_log_transform is True, cfg.name
        assert cfg.dm_entropy_intensive is True, cfg.name
        assert [d.name for d in cfg.descriptors] == ["cusp"], cfg.name
    assert plain.attention is False
    assert attn.attention is True and attn.num_heads == 4


# §13.2 item (13)
def test_get_architecture_raises_for_unknown():
    from xcquinox.pipeline.config import get_architecture
    with pytest.raises(KeyError):
        get_architecture("nonexistent")


# ---------------------------------------------------------------------------
# The seed mixture against the cold start: the pair the mixture cannot express
# ---------------------------------------------------------------------------

def _seed_spec(tmp_path, *, seed_source, in_loss_kwargs=True, **extra):
    """An H / O / H2O spec whose only free variable is where the SCF starts.

    The solver config is placed either in ``loss_kwargs`` or in the
    ``solver_config`` field, the pair the training loop reads as
    ``loss_kwargs_dict.get("solver_config") or solver_config``; both routes
    must reach the same rule.
    """
    from xcquinox.pipeline.config import (
        MoleculeSpec, TrainingSpec, get_architecture)
    from xcquinox.pipeline.solver import SolverConfig, SolverMode

    mols = (
        MoleculeSpec.from_dict(
            name="H", atom="H 0 0 0", basis="sto-3g", charge=0, spin=1,
            atom_composition={"H": 1},
        ),
        MoleculeSpec.from_dict(
            name="O", atom="O 0 0 0", basis="sto-3g", charge=0, spin=2,
            atom_composition={"O": 1},
        ),
        MoleculeSpec.from_dict(
            name="H2O", atom="O 0 0 0; H 0 0 0.96; H 0.93 0 -0.24",
            basis="sto-3g", charge=0, spin=0,
            atom_composition={"H": 2, "O": 1},
        ),
    )
    sc = SolverConfig(mode=SolverMode.FULL, max_cycles=3,
                      seed_source=seed_source)
    kwargs = dict(
        arch=get_architecture("deep_combined"),
        molecules=mols,
        targets=(("H", 0.0), ("H2O", 232.0), ("O", 0.0)),
        atom_energies=(("H", -0.5), ("O", -75.0)),
        loss_name="A_atomization",
        loss_kwargs=(("solver_config", sc),) if in_loss_kwargs else (),
        solver_config=None if in_loss_kwargs else sc,
        n_steps=10,
        lr_start=1e-3,
        lr_end=1e-5,
        lr_decay_start=0.0,
        grad_clip=1.0,
        pretrain_checkpoint=None,
        checkpoint_dir=str(tmp_path / "ckpt"),
        seed=0,
        update_scheme="per_molecule",
        seed_mix_atomic=True,
    )
    kwargs.update(extra)
    return TrainingSpec(**kwargs)


def test_seed_mixture_is_refused_on_a_cold_start_seed(tmp_path):
    """``seed_mix_atomic`` with a ``minao`` SCF seed is refused by
    ``TrainingSpec.validate``, by both routes the loop reads the solver config.

    The mixture forms ``(1 - beta) D_seed + beta D_minao``. Where the seed IS
    the atomic guess the two endpoints coincide and the combination is the cold
    start itself with a coefficient that changes nothing: an arm that reads as
    a third protocol while executing the second. The oracle is the pair of
    specs differing in ``seed_source`` alone -- the ``pbe`` form validates, the
    ``minao`` form raises with both field names in the message. A third spec
    keeps the cold-start seed and drops the mixture, which separates a rule
    that refuses the seed by itself from one that refuses the pair.
    """
    import dataclasses

    # Control: the mixture over a converged parent density is the protocol,
    # and validates both before and after the rule lands.
    _seed_spec(tmp_path, seed_source="pbe").validate()

    for in_loss_kwargs in (True, False):
        spec = _seed_spec(tmp_path, seed_source="minao",
                          in_loss_kwargs=in_loss_kwargs)
        with pytest.raises(ValueError) as excinfo:
            spec.validate()
        message = str(excinfo.value)
        assert "seed_mix_atomic" in message, message
        assert "seed_source" in message, message

    # Control: the cold start WITHOUT the mixture is a legal arm (it is the
    # protocol of the cold-start held-out channel), so the rule must name the
    # pair and not the seed by itself.
    dataclasses.replace(_seed_spec(tmp_path, seed_source="minao"),
                        seed_mix_atomic=False).validate()


# §13.2 item (14)


# §13.2 item (15)


# 2026-06-20: 3x16 (depth-3, width-16) twins of the 8 dfs_step7 sweep archs,
# matching DFS's published net size (Dick & Fernandez-Serra 2021, 3 hidden
# layers x 16 nodes), for the capacity-reduction experiment. Each twin must
# differ from its 4x32 sibling ONLY in depth/nodes.
_DFS_3X16_TWINS = {
    "deep_3x16": "deep",
    "deep_attn_3x16": "deep_attn",
    "deep_cusp_3x16": "deep_cusp",
    "deep_dm_3x16": "deep_dm",
    "deep_combined_3x16": "deep_combined",
    "deep_combined_attn_3x16": "deep_combined_attn",
    "deep_notransform_3x16": "deep_notransform",
    "deep_notransform_attn_3x16": "deep_notransform_attn",
}


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
    from xcquinox.pipeline.config import get_architecture
    zero = get_architecture("shallow")
    one_cusp = get_architecture("deep_cusp")
    one_dm = get_architecture("deep_dm")
    two = get_architecture("deep_combined")
    assert zero.n_input_features == 2
    assert one_cusp.n_input_features == 2 + 2
    # dm_statistics is 2 wide since dm_entropy was removed 2026-08-06.
    assert one_dm.n_input_features == 2 + 2
    assert two.n_input_features == 2 + 2 + 2


# §13.2 item (11)
def test_architecture_materialize_roundtrip_returns_registry_instances():
    from xcquinox.pipeline.config import get_architecture
    from xcquinox.pipeline.descriptors import CuspDescriptor, DMStatisticsDescriptor
    from xcquinox.pipeline.constraints import LiebOxfordBound, UEGLimit

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
    from xcquinox.pipeline.config import TrainingSpec, MoleculeSpec, get_architecture

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
        checkpoint_dir="/tmp/pipeline_nonexistent_ckpt_dir",
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
    from xcquinox.pipeline.config import PretrainSpec, get_architecture

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
    assert out["arch"] == "deep_combined"
    json.dumps(out)


# --- §13.2 item (16), Task 2.2 step 6 ----------------------------------------

# §13.2 item (16)
def test_architectures_all_materialize_via_from_arch():
    from xcquinox.pipeline.config import ARCHITECTURES
    from xcquinox.pipeline.models import AlecGGAModel
    assert len(ARCHITECTURES) == 36
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


# --- PretrainSpec.loss_weighting (physics-fixes Task 1) -----------------------


# ---------------------------------------------------------------------------
# Step-6 Task 3.1: PBE-anchor pass-through fields on TrainingSpec / TestSpec
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Self-attention registry tests (spec §Tests 20-21)
# ---------------------------------------------------------------------------


def test_registry_smoke_forward_each_attn_arch():
    """Test 21: every *_attn arch builds and runs a forward pass."""
    import jax.numpy as jnp
    from xcquinox.pipeline.config import ARCHITECTURES
    from xcquinox.pipeline.networks import create_network_pair

    # Filter on the attention flag so the *_attn_3x16 archs (which end in
    # `_3x16`, not `_attn`) also get a forward-pass smoke.
    attn_keys = [k for k, a in ARCHITECTURES.items() if a.attention]
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
    from xcquinox.pipeline.config import ArchitectureConfig
    return ArchitectureConfig(
        name="tiny", depth=2, nodes=8, attention=False,
        descriptors=(), x_constraints=(), c_constraints=(),
        double_lob_clamp_allowed=False,
    )


def test_validate_missing_element_in_atom_energies_require_anchors_false():
    """compound references C which is absent from atom_energies;
    validate() must raise ValueError naming C, even with require_atom_anchors=False."""
    import tempfile
    from xcquinox.pipeline.config import TrainingSpec, MoleculeSpec

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


# ---------------------------------------------------------------------------
# bool values must be rejected from targets and atom_energies even
# though math.isfinite(True) is True.
# ---------------------------------------------------------------------------

def test_validate_bool_in_targets_rejected():
    """True passed as a target energy must raise ValueError."""
    import tempfile
    from xcquinox.pipeline.config import TrainingSpec, MoleculeSpec

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


# ---------------------------------------------------------------------------
# PretrainSpec: pretraining-protocol fields (spec Sections 3.2, 6, 7)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# T2: the display-name layer reaches the registry (2026-09-09)
#
# The figures are relabelled by what each network IS, and the shown names that
# are not themselves registry keys resolve to the stored configuration, so a
# later configuration file may use them directly. The stored keys are
# untouched: the running arms and every pulled result are filed under them.
# ---------------------------------------------------------------------------

def test_get_architecture_resolves_display_name_aliases():
    """A shown name that is not a stored key resolves to its configuration,
    and the two shown names that ARE stored keys of other configurations keep
    resolving to those.

    Kills m4 (the alias lookup removed): without it every assertion in the
    first block raises KeyError.
    """
    from xcquinox.pipeline.config import ARCHITECTURES, get_architecture
    assert get_architecture("deep0_3x16") is ARCHITECTURES["deep_3x16"]
    assert get_architecture("deep0_attn_3x16") is ARCHITECTURES["deep_attn_3x16"]
    assert get_architecture("deep0_cusp_mgga_3x16") is \
        ARCHITECTURES["deep_cusp_mgga_3x16"]
    assert get_architecture("deep_2x8") is ARCHITECTURES["shallow"]
    assert get_architecture("deep_attn_2x8") is ARCHITECTURES["shallow_attn"]
    assert get_architecture("deep0_4x32") is ARCHITECTURES["deep"]
    # the collision: `deep_3x16` is medium's SHOWN name and another entry's
    # stored key. The registry lookup is the storage sense and does not move.
    assert get_architecture("deep_3x16") is ARCHITECTURES["deep_3x16"]
    assert get_architecture("deep_attn_3x16") is ARCHITECTURES["deep_attn_3x16"]
    assert get_architecture("medium") is ARCHITECTURES["medium"]
    # an alias never shadows a stored key, and an unknown name still raises
    assert not set(_alias_map()) & set(ARCHITECTURES)
    with pytest.raises(KeyError):
        get_architecture("nonexistent")
    with pytest.raises(KeyError):
        get_architecture("deep0_not_an_arch")


def _alias_map():
    from xcquinox.pipeline.arch_names import ALIASES
    return ALIASES


