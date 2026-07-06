"""Tests for xcquinox.alec.solver: SolverConfig, Mixer, ConvergenceCriterion."""
import json
import pytest
import jax.numpy as jnp

from xcquinox.alec.solver import (
    SolverBackend,
    SolverMode,
    FeaturePolicy,
)


def test_solver_backend_values():
    assert SolverBackend.MANUAL.value == "manual"
    assert SolverBackend.PYSCFAD.value == "pyscfad"


def test_solver_mode_values():
    assert SolverMode.ONESHOT.value == "oneshot"
    assert SolverMode.FIXED_J.value == "fixed_j"
    assert SolverMode.FULL.value == "full"


def test_feature_policy_values():
    assert FeaturePolicy.FROZEN.value == "frozen"
    assert FeaturePolicy.REASSEMBLE.value == "reassemble"


def test_enums_are_json_serializable():
    assert json.dumps(SolverBackend.MANUAL.value) == '"manual"'
    assert json.dumps(SolverMode.FIXED_J.value) == '"fixed_j"'
    assert json.dumps(FeaturePolicy.REASSEMBLE.value) == '"reassemble"'


from xcquinox.alec.solver import SolverConfig


def test_solver_config_default_is_oneshot_manual():
    cfg = SolverConfig()
    assert cfg.backend == SolverBackend.MANUAL
    assert cfg.mode == SolverMode.ONESHOT
    assert cfg.max_cycles == 0
    assert cfg.conv_tol == 1e-6


def test_solver_config_rejects_negative_cycles():
    with pytest.raises(ValueError, match="max_cycles must be >= 0"):
        SolverConfig(max_cycles=-1)


def test_solver_config_rejects_oneshot_with_cycles():
    with pytest.raises(ValueError, match="oneshot mode requires max_cycles=0"):
        SolverConfig(mode=SolverMode.ONESHOT, max_cycles=3)


def test_solver_config_rejects_nononeshot_with_zero_cycles():
    with pytest.raises(ValueError, match="non-oneshot modes require"):
        SolverConfig(mode=SolverMode.FIXED_J, max_cycles=0)


def test_solver_config_rejects_nonpositive_tol():
    with pytest.raises(ValueError, match="conv_tol must be > 0"):
        SolverConfig(conv_tol=0)


def test_solver_config_is_hashable():
    cfg = SolverConfig()
    hash(cfg)
    d = {cfg: "value"}
    assert d[SolverConfig()] == "value"


def test_solver_config_describe_is_json_serializable():
    cfg = SolverConfig(mode=SolverMode.FIXED_J, max_cycles=5)
    described = cfg.describe()
    assert json.dumps(described)


def test_effective_feature_policy_fixed_j_is_frozen():
    cfg = SolverConfig(mode=SolverMode.FIXED_J, max_cycles=5)
    assert cfg.effective_feature_policy == FeaturePolicy.FROZEN


def test_effective_feature_policy_full_is_reassemble():
    cfg = SolverConfig(mode=SolverMode.FULL, max_cycles=5)
    assert cfg.effective_feature_policy == FeaturePolicy.REASSEMBLE


from xcquinox.alec.solver import Mixer, LinearMixer, MixerState


def test_mixer_abc_is_abstract():
    with pytest.raises(TypeError):
        Mixer()


def test_linear_mixer_alpha_1_is_identity():
    mixer = LinearMixer(alpha=1.0)
    state = mixer.init_state(nao=3)
    D_in = jnp.eye(3) * 2.0
    D_out = jnp.eye(3) * 3.0
    _, D_mixed = mixer.step(state, D_in, D_out)
    assert jnp.allclose(D_mixed, D_out)


def test_linear_mixer_alpha_0_pins_D_in():
    mixer = LinearMixer(alpha=0.0)
    state = mixer.init_state(nao=3)
    D_in = jnp.eye(3) * 2.0
    D_out = jnp.eye(3) * 3.0
    _, D_mixed = mixer.step(state, D_in, D_out)
    assert jnp.allclose(D_mixed, D_in)


def test_linear_mixer_alpha_half_averages():
    mixer = LinearMixer(alpha=0.5)
    state = mixer.init_state(nao=3)
    D_in = jnp.eye(3) * 2.0
    D_out = jnp.eye(3) * 4.0
    _, D_mixed = mixer.step(state, D_in, D_out)
    expected = 0.5 * (D_in + D_out)
    assert jnp.allclose(D_mixed, expected)


def test_linear_mixer_rejects_out_of_range_alpha():
    with pytest.raises(ValueError, match="alpha must be in"):
        LinearMixer(alpha=-0.1)
    with pytest.raises(ValueError, match="alpha must be in"):
        LinearMixer(alpha=1.5)


def test_linear_mixer_step_increments_state():
    mixer = LinearMixer(alpha=0.5)
    state = mixer.init_state(nao=3)
    assert int(state.step_index) == 0
    D = jnp.eye(3)
    new_state, _ = mixer.step(state, D, D)
    assert int(new_state.step_index) == 1


def test_linear_mixer_registry_name():
    assert LinearMixer.registry_name == "linear"


from xcquinox.alec.solver import ConvergenceCriterion, EnergyConvergence


def test_convergence_criterion_abc_is_abstract():
    with pytest.raises(TypeError):
        ConvergenceCriterion()


def test_energy_convergence_small_delta_converges():
    crit = EnergyConvergence(tol=1e-6)
    e_prev = jnp.float64(1.0)
    e_curr = jnp.float64(1.0 + 1e-8)
    assert bool(crit.is_converged_from_energies(e_prev, e_curr))


def test_energy_convergence_large_delta_not_converged():
    crit = EnergyConvergence(tol=1e-6)
    e_prev = jnp.float64(1.0)
    e_curr = jnp.float64(1.0 + 1e-3)
    assert not bool(crit.is_converged_from_energies(e_prev, e_curr))


def test_energy_convergence_rejects_nonpositive_tol():
    with pytest.raises(ValueError, match="tol must be > 0"):
        EnergyConvergence(tol=-1.0)
    with pytest.raises(ValueError, match="tol must be > 0"):
        EnergyConvergence(tol=0.0)


def test_energy_convergence_registry_name():
    assert EnergyConvergence.registry_name == "energy"


from xcquinox.alec.solver import SCFResult, run_scf


def test_scf_result_is_dataclass():
    import dataclasses as dc
    assert dc.is_dataclass(SCFResult)


def test_run_scf_unknown_backend_raises():
    from unittest.mock import MagicMock
    cfg = SolverConfig.__new__(SolverConfig)
    object.__setattr__(cfg, "backend", "bogus")
    object.__setattr__(cfg, "mode", SolverMode.ONESHOT)
    object.__setattr__(cfg, "max_cycles", 0)
    object.__setattr__(cfg, "conv_tol", 1e-6)
    object.__setattr__(cfg, "feature_policy", None)
    object.__setattr__(cfg, "mixer_name", "linear")
    object.__setattr__(cfg, "mixer_kwargs", (("alpha", 0.5),))
    object.__setattr__(cfg, "convergence_name", "energy")
    with pytest.raises(ValueError, match="unknown solver backend"):
        run_scf(cfg, MagicMock(), {})


def test_effective_feature_policy_honors_explicit_override():
    cfg = SolverConfig(
        mode=SolverMode.FIXED_J, max_cycles=5,
        feature_policy=FeaturePolicy.REASSEMBLE,
    )
    assert cfg.effective_feature_policy == FeaturePolicy.REASSEMBLE


def test_contract_dm_to_grid_matches_precompute():
    """_contract_dm_to_grid(D_PBE, ao_deriv) should reproduce the (rho, sigma)
    stored by precompute_fixed_density_data for the same DM."""
    import numpy as np
    from xcquinox.alec.solver import _contract_dm_to_grid
    from xcquinox.alec.data import precompute_fixed_density_data
    from xcquinox.alec.tests.fixtures.molecules import h2_molecule

    data = precompute_fixed_density_data(h2_molecule())
    rho, sigma = _contract_dm_to_grid(
        data["dm_pbe"], data["ao_grid_deriv"],
    )
    np.testing.assert_allclose(
        np.asarray(rho), np.asarray(data["rho_grid"]),
        atol=1e-10, rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(sigma), np.asarray(data["sigma_grid"]),
        atol=1e-10, rtol=0.0,
    )


def test_reassemble_features_matches_precompute_for_cusp_and_dm():
    """_reassemble_features run at D=D_PBE should match the frozen features
    produced by assemble_descriptor_features(mol_data)."""
    import numpy as np
    from xcquinox.alec.solver import _reassemble_features
    from xcquinox.alec.descriptors import (
        CuspDescriptor, DMStatisticsDescriptor, assemble_descriptor_features,
    )
    from xcquinox.alec.data import precompute_fixed_density_data
    from xcquinox.alec.tests.fixtures.molecules import h2_molecule

    descriptors = (CuspDescriptor(), DMStatisticsDescriptor())
    data = precompute_fixed_density_data(h2_molecule(), descriptors=descriptors)

    features_frozen = assemble_descriptor_features(descriptors, data)
    features_reassembled = _reassemble_features(
        descriptors=descriptors,
        dm=data["dm_pbe"],
        s_matrix=data["s_matrix"],
        cusp_features=data["cusp_features"],
    )
    assert features_reassembled.shape == features_frozen.shape
    np.testing.assert_allclose(
        np.asarray(features_reassembled),
        np.asarray(features_frozen),
        atol=1e-10, rtol=0.0,
    )


def test_reassemble_features_dm_only_uses_n_grid():
    """_reassemble_features with DMStatisticsDescriptor only (no cusp) uses
    the n_grid parameter instead of cusp_features for the grid-size hint."""
    import numpy as np
    from xcquinox.alec.solver import _reassemble_features
    from xcquinox.alec.descriptors import (
        DMStatisticsDescriptor, assemble_descriptor_features,
    )
    from xcquinox.alec.data import precompute_fixed_density_data
    from xcquinox.alec.tests.fixtures.molecules import h2_molecule

    descriptors = (DMStatisticsDescriptor(),)
    data = precompute_fixed_density_data(h2_molecule(), descriptors=descriptors)
    n_grid = data["grid_weights"].shape[0]

    features_frozen = assemble_descriptor_features(descriptors, data)
    features_reassembled = _reassemble_features(
        descriptors=descriptors,
        dm=data["dm_pbe"],
        s_matrix=data["s_matrix"],
        cusp_features=None,
        n_grid=n_grid,
    )
    assert features_reassembled.shape == features_frozen.shape
    np.testing.assert_allclose(
        np.asarray(features_reassembled),
        np.asarray(features_frozen),
        atol=1e-10, rtol=0.0,
    )


def test_oneshot_result_matches_legacy_total_energy():
    """_oneshot_result(model, mol_data) should produce total_energy identical
    to fixed_density_total_energy(model, mol_data)."""
    import numpy as np
    from xcquinox.alec.solver import _oneshot_result
    from xcquinox.alec.oneshot import fixed_density_total_energy
    from xcquinox.alec.config import ArchitectureConfig
    from xcquinox.alec.models import AlecGGAModel
    from xcquinox.alec.data import precompute_fixed_density_data
    from xcquinox.alec.tests.fixtures.molecules import h2_molecule

    arch = ArchitectureConfig(
        name="t", depth=2, nodes=8, attention=False,
        descriptors=(), x_constraints=(), c_constraints=(),
        double_lob_clamp_allowed=False,
    )
    model = AlecGGAModel.from_arch(arch, seed=0)
    data = precompute_fixed_density_data(h2_molecule())
    result = _oneshot_result(model, data)
    e_legacy = float(fixed_density_total_energy(model, data))
    assert float(result.total_energy) == pytest.approx(e_legacy, abs=1e-12)
    assert int(result.cycles_run) == 0
    assert bool(result.converged) is True


# ---------------------------------------------------------------------------
# Fix: mixer registry
# ---------------------------------------------------------------------------

def test_mixer_registry_resolves_linear_via_class_lookup():
    """_build_mixer must resolve config.mixer_name through MIXER_REGISTRY,
    not a hard-coded 'linear' branch (fix). The default
    'linear' name maps to LinearMixer with the kwargs from
    config.mixer_kwargs."""
    from xcquinox.alec.solver import (
        SolverConfig, SolverBackend, SolverMode, LinearMixer, MIXER_REGISTRY
    )
    from xcquinox.alec.solver_manual import _build_mixer
    assert "linear" in MIXER_REGISTRY
    assert MIXER_REGISTRY["linear"] is LinearMixer
    cfg = SolverConfig(
        backend=SolverBackend.MANUAL, mode=SolverMode.FIXED_J,
        max_cycles=2,
        mixer_name="linear", mixer_kwargs=(("alpha", 0.3),),
    )
    m = _build_mixer(cfg)
    assert isinstance(m, LinearMixer)
    assert abs(m.alpha - 0.3) < 1e-12


def test_mixer_registry_unknown_name_raises_with_available_list():
    """Unknown mixer name must raise NotImplementedError listing
    available mixers (so users see what they can pick)."""
    import pytest
    from xcquinox.alec.solver import SolverConfig, SolverBackend, SolverMode
    from xcquinox.alec.solver_manual import _build_mixer
    cfg = SolverConfig(
        backend=SolverBackend.MANUAL, mode=SolverMode.FIXED_J,
        max_cycles=2, mixer_name="bogus_mixer_xyz",
    )
    with pytest.raises(NotImplementedError, match="available:"):
        _build_mixer(cfg)


def test_full_mode_rejects_frozen_feature_policy():
    """Fix: (FULL, FROZEN) is incoherent; constructor
    must reject it with a clear message."""
    import pytest
    from xcquinox.alec.solver import (
        SolverConfig, SolverBackend, SolverMode, FeaturePolicy
    )
    with pytest.raises(ValueError, match="incoherent"):
        SolverConfig(
            backend=SolverBackend.MANUAL, mode=SolverMode.FULL,
            max_cycles=3,
            feature_policy=FeaturePolicy.FROZEN,
        )
    # FULL with REASSEMBLE explicit is fine.
    cfg = SolverConfig(
        backend=SolverBackend.MANUAL, mode=SolverMode.FULL, max_cycles=3,
        feature_policy=FeaturePolicy.REASSEMBLE,
    )
    assert cfg.feature_policy == FeaturePolicy.REASSEMBLE
    # FULL with feature_policy=None auto-resolves to REASSEMBLE.
    cfg2 = SolverConfig(
        backend=SolverBackend.MANUAL, mode=SolverMode.FULL, max_cycles=3,
    )
    assert cfg2.effective_feature_policy == FeaturePolicy.REASSEMBLE


# ---------------------------------------------------------------------------
# DecayingLinearMixer: DFS step-decaying damping alpha = base**step + floor.
# DFS (Dick & Fernandez-Serra 2021) torch_routines.py:174-178 uses
# alpha = (0.3)**step + 0.3 -> step0=1.3 (over-relaxation), step1=0.6,
# step2=0.39 -> asymptote 0.3. Unlike LinearMixer it must NOT clamp alpha.
# ---------------------------------------------------------------------------
from xcquinox.alec.solver import DecayingLinearMixer, MIXER_REGISTRY


def test_decaying_linear_mixer_registry_name():
    assert DecayingLinearMixer.registry_name == "decaying_linear"


def test_decaying_linear_mixer_in_registry():
    assert MIXER_REGISTRY["decaying_linear"] is DecayingLinearMixer


def test_decaying_linear_mixer_alpha_schedule():
    # D_in = 0, D_out = I  =>  D_mixed = alpha * I, isolating alpha per step.
    mixer = DecayingLinearMixer(base=0.3, floor=0.3)
    state = mixer.init_state(nao=3)
    D_in = jnp.zeros((3, 3))
    D_out = jnp.eye(3)
    expected_alpha = [1.3, 0.6, 0.39]  # 0.3**step + 0.3 for step 0,1,2
    for step, exp in enumerate(expected_alpha):
        new_state, D_mixed = mixer.step(state, D_in, D_out)
        assert jnp.allclose(jnp.diag(D_mixed), exp), (
            f"step {step}: alpha {float(D_mixed[0, 0])} != {exp}"
        )
        state = new_state


def test_decaying_linear_mixer_step0_over_relaxes():
    # step 0 alpha = 1.3 > 1: D_mixed extrapolates beyond D_out (DFS-faithful,
    # not clamped). D_mixed = 1.3*D_out - 0.3*D_in.
    mixer = DecayingLinearMixer(base=0.3, floor=0.3)
    state = mixer.init_state(nao=2)
    D_in = jnp.eye(2) * 1.0
    D_out = jnp.eye(2) * 2.0
    _, D_mixed = mixer.step(state, D_in, D_out)
    expected = 1.3 * D_out - 0.3 * D_in
    assert jnp.allclose(D_mixed, expected)


def test_decaying_linear_mixer_step_increments_state():
    mixer = DecayingLinearMixer()
    state = mixer.init_state(nao=3)
    assert int(state.step_index) == 0
    D = jnp.eye(3)
    new_state, _ = mixer.step(state, D, D)
    assert int(new_state.step_index) == 1


def test_decaying_linear_mixer_rejects_bad_params():
    with pytest.raises(ValueError, match="base must be in"):
        DecayingLinearMixer(base=0.0)
    with pytest.raises(ValueError, match="base must be in"):
        DecayingLinearMixer(base=1.0)
    with pytest.raises(ValueError, match="floor must be in"):
        DecayingLinearMixer(floor=-0.1)
    with pytest.raises(ValueError, match="floor must be in"):
        DecayingLinearMixer(floor=1.0)


# ---------------------------------------------------------------------------
# SolverConfig: DFS tail-weighted-loss knobs (opt-in; defaults inert).
# ---------------------------------------------------------------------------

def test_solver_config_tail_defaults_are_inert():
    cfg = SolverConfig()
    assert cfg.scf_loss_use_tail is False
    assert cfg.scf_loss_tail == 10
    assert cfg.scf_loss_weight_power == 2.0


def test_solver_config_rejects_bad_tail():
    with pytest.raises(ValueError, match="scf_loss_tail must be >= 1"):
        SolverConfig(scf_loss_tail=0)


def test_solver_config_rejects_negative_weight_power():
    with pytest.raises(ValueError, match="scf_loss_weight_power must be >= 0"):
        SolverConfig(scf_loss_weight_power=-1.0)


def test_solver_config_describe_includes_tail_knobs():
    cfg = SolverConfig(
        mode=SolverMode.FULL, max_cycles=25,
        scf_loss_use_tail=True, scf_loss_tail=10, scf_loss_weight_power=2.0,
    )
    d = cfg.describe()
    assert d["scf_loss_use_tail"] is True
    assert d["scf_loss_tail"] == 10
    assert d["scf_loss_weight_power"] == 2.0
    assert json.dumps(d)  # still JSON-serializable
