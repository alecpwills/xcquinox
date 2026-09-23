"""Tests for xcquinox.pipeline.solver: SolverConfig, Mixer, ConvergenceCriterion."""
import json
import pytest
import jax.numpy as jnp

from xcquinox.pipeline.solver import (
    SolverBackend,
    SolverMode,
    FeaturePolicy,
)


def test_enums_are_json_serializable():
    assert json.dumps(SolverBackend.MANUAL.value) == '"manual"'
    assert json.dumps(SolverMode.FIXED_J.value) == '"fixed_j"'
    assert json.dumps(FeaturePolicy.REASSEMBLE.value) == '"reassemble"'


from xcquinox.pipeline.solver import SolverConfig


def test_solver_config_default_is_oneshot_manual():
    cfg = SolverConfig()
    assert cfg.backend == SolverBackend.MANUAL
    assert cfg.mode == SolverMode.ONESHOT
    assert cfg.max_cycles == 0
    assert cfg.conv_tol == 1e-6


def test_solver_config_rejects_oneshot_with_cycles():
    with pytest.raises(ValueError, match="oneshot mode requires max_cycles=0"):
        SolverConfig(mode=SolverMode.ONESHOT, max_cycles=3)


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


from xcquinox.pipeline.solver import LinearMixer


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


from xcquinox.pipeline.solver import EnergyConvergence


def test_energy_convergence_small_delta_converges():
    crit = EnergyConvergence(tol=1e-6)
    e_prev = jnp.float64(1.0)
    e_curr = jnp.float64(1.0 + 1e-8)
    assert bool(crit.is_converged_from_energies(e_prev, e_curr))




def test_contract_dm_to_grid_matches_precompute():
    """_contract_dm_to_grid(D_PBE, ao_deriv) should reproduce the (rho, sigma)
    stored by precompute_fixed_density_data for the same DM."""
    import numpy as np
    from xcquinox.pipeline.solver import _contract_dm_to_grid
    from xcquinox.pipeline.data import precompute_fixed_density_data
    from xcquinox.pipeline.tests.fixtures.molecules import h2_molecule

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
    from xcquinox.pipeline.solver import _reassemble_features
    from xcquinox.pipeline.descriptors import (
        CuspDescriptor, DMStatisticsDescriptor, assemble_descriptor_features,
    )
    from xcquinox.pipeline.data import precompute_fixed_density_data
    from xcquinox.pipeline.tests.fixtures.molecules import h2_molecule

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


# ---------------------------------------------------------------------------
# Fix: mixer registry
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# DecayingLinearMixer: DFS step-decaying damping alpha = base**step + floor.
# DFS (Dick & Fernandez-Serra 2021) torch_routines.py:174-178 uses
# alpha = (0.3)**step + 0.3 -> step0=1.3 (over-relaxation), step1=0.6,
# step2=0.39 -> asymptote 0.3. Unlike LinearMixer it must NOT clamp alpha.
# ---------------------------------------------------------------------------
from xcquinox.pipeline.solver import DecayingLinearMixer


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


# ---------------------------------------------------------------------------
# SolverConfig: DFS tail-weighted-loss knobs (opt-in; defaults inert).
# ---------------------------------------------------------------------------


# --------------------------------------------------------------------------- #
# SCF seed source (per-rung seeding protocol)
# --------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------
# DecayingLinearMixer step_offset: the SI equation (offset 0) against the
# schedule dpyscf's loop actually executes (offset 1). og_dpyscf places the mix
# at the top of its SCF loop, where dm == dm_old on step 0, so its first
# EFFECTIVE mix uses alpha = 0.3**1 + 0.3 = 0.6, then 0.39, then 0.327 -- the
# equation's schedule shifted by one step.
# ---------------------------------------------------------------------------

def _alpha_trace(mixer, n_steps):
    """The mixer's alpha at steps 0..n_steps-1, isolated by D_in = 0, D_out = I
    (D_mixed = alpha * I), the construction of
    test_decaying_linear_mixer_alpha_schedule."""
    state = mixer.init_state(nao=3)
    D_in = jnp.zeros((3, 3))
    D_out = jnp.eye(3)
    alphas = []
    for _ in range(n_steps):
        state, D_mixed = mixer.step(state, D_in, D_out)
        alphas.append(D_mixed[0, 0])
    return alphas


def test_decaying_linear_mixer_step_offset_reproduces_the_executed_dpyscf_schedule():
    """``alpha = base ** (step_index + step_offset) + floor``. Offset 1 is the
    schedule og_dpyscf's loop executes (0.6, 0.39, 0.327); offset 0 is the SI
    equation and must leave the current schedule (1.3, 0.6, 0.39) bit-identical
    to the no-offset mixer. The YAML mixer-kwarg parser coerces every value to
    float, so an integral float (1.0) is accepted and stored as an int; a
    non-integral value and a negative one are refused by name."""
    offset1 = _alpha_trace(DecayingLinearMixer(base=0.3, floor=0.3, step_offset=1), 3)
    for got, exp in zip(offset1, [0.6, 0.39, 0.327]):
        assert jnp.allclose(got, exp), (float(got), exp)
    # offset 0 == the pre-offset mixer, bit for bit (no schedule change for
    # every existing solver, whose mixer_kwargs carry base and floor only).
    old = _alpha_trace(DecayingLinearMixer(base=0.3, floor=0.3), 3)
    new = _alpha_trace(DecayingLinearMixer(base=0.3, floor=0.3, step_offset=0), 3)
    for a, b in zip(old, new):
        assert jnp.array_equal(a, b), (float(a), float(b))
    assert jnp.allclose(jnp.asarray(old), jnp.asarray([1.3, 0.6, 0.39]))
    # A float from _parse_mixer_kwargs is accepted when integral, and stored as
    # an int so the schedule exponent stays exact.
    m = DecayingLinearMixer(base=0.3, floor=0.3, step_offset=1.0)
    assert isinstance(m.step_offset, int) and m.step_offset == 1
    assert jnp.allclose(_alpha_trace(m, 1)[0], 0.6)
    with pytest.raises(ValueError, match="step_offset") as half:
        DecayingLinearMixer(step_offset=0.5)
    assert "0.5" in str(half.value)
    with pytest.raises(ValueError, match="step_offset") as neg:
        DecayingLinearMixer(step_offset=-1)
    assert "-1" in str(neg.value)


