"""Tests for xcquinox.alec.solver — SolverConfig, Mixer, ConvergenceCriterion."""
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
