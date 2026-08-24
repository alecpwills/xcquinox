"""The per-system energy term of the pretraining objective.

Spec Section 6 deviation 3: "the point-wise residual is integration-weighted
(as today) AND a per-system energy term E_xc^NN - E_xc^parent in Hartree is
added, so the H atom and every molecule carry an energy of their own". These
tests pin the term's algebra against closed forms and its plumbing against a
real tiny .npz.
"""
import json
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import xcquinox.alec.pretrain_data_gen as pdg
from xcquinox.alec.config import ArchitectureConfig, PretrainSpec
from xcquinox.alec.pretrain import (
    _PretrainLoss, _assemble_pretrain_descriptors, _energy_term_inputs,
    run_pretrain)


class _EchoModel(eqx.Module):
    """A stand-in network whose enhancement factor is the row's first column
    plus a constant, so a test can make it reproduce a target exactly or miss
    it by a stated amount."""
    offset: float = 0.0

    def __call__(self, row):
        return 1.0 + row[0] + self.offset


def _loss_arrays():
    """Two systems, three rows each, with a mesh row belonging to neither."""
    ref = jnp.asarray([0.1, -0.2, 0.3, 0.0, 0.5, -0.4, 0.0])
    descriptors = jnp.stack([ref, jnp.ones(7)], axis=1)
    row_weight = jnp.asarray([1.0, 2.0, 0.5, 3.0, 1.5, 1.0, 0.0])
    segment = jnp.asarray([0, 0, 0, 1, 1, 1, 2], dtype=jnp.int32)
    return ref, descriptors, row_weight, segment


def _parent_energy(ref, row_weight, segment, n_systems):
    """The parent's own value of the same quadrature: sum w (1 + F_ref)."""
    contrib = np.asarray(row_weight) * (1.0 + np.asarray(ref))
    seg = np.asarray(segment)
    return jnp.asarray([contrib[seg == s].sum() for s in range(n_systems)])


# ---------------------------------------------------------------------------
# The term's algebra
# ---------------------------------------------------------------------------

def test_energy_term_vanishes_for_a_network_that_reproduces_the_target():
    """The stored per-system target is the quadrature of the stored
    enhancement factors, so a network that reproduces them exactly carries no
    energy error. That is what makes the term measure the fit and nothing
    else."""
    ref, descriptors, row_weight, segment = _loss_arrays()
    target = _parent_energy(ref, row_weight, segment, 2)
    loss = _PretrainLoss(weights=jnp.ones(7), energy_row_weight=row_weight,
                         energy_segment=segment, energy_target=target,
                         energy_weight=1.0, n_systems=2)
    pointwise, energy = loss.parts(_EchoModel(0.0), descriptors, ref)
    assert float(pointwise) == pytest.approx(0.0, abs=1e-28)
    assert float(energy) == pytest.approx(0.0, abs=1e-24)
    assert float(loss(_EchoModel(0.0), descriptors, ref)) == \
        pytest.approx(0.0, abs=1e-24)


def test_constant_offset_gives_the_analytic_energy_term():
    """A network uniformly off by c gives per-system energy error c * R_s with
    R_s the system's total row weight, so the term is
    mean_s (c R_s)^2 exactly."""
    ref, descriptors, row_weight, segment = _loss_arrays()
    target = _parent_energy(ref, row_weight, segment, 2)
    loss = _PretrainLoss(weights=jnp.ones(7), energy_row_weight=row_weight,
                         energy_segment=segment, energy_target=target,
                         energy_weight=1.0, n_systems=2)
    c = 0.25
    _pw, energy = loss.parts(_EchoModel(c), descriptors, ref)
    rw = np.asarray(row_weight)
    seg = np.asarray(segment)
    expect = float(np.mean([(c * rw[seg == s].sum()) ** 2 for s in range(2)]))
    assert float(energy) == pytest.approx(expect, rel=1e-12)


def test_mesh_rows_carry_no_energy():
    """A synthetic (r_s, s, alpha) node belongs to no system: its sink segment
    index is asked of segment_sum and dropped, so its enhancement factor can
    never move a system's energy."""
    ref, descriptors, row_weight, segment = _loss_arrays()
    target = _parent_energy(ref, row_weight, segment, 2)
    loss = _PretrainLoss(weights=jnp.ones(7), energy_row_weight=row_weight,
                         energy_segment=segment, energy_target=target,
                         energy_weight=1.0, n_systems=2)
    _pw, base = loss.parts(_EchoModel(0.0), descriptors, ref)
    bumped = descriptors.at[6, 0].add(10.0)
    _pw2, moved = loss.parts(_EchoModel(0.0), bumped, ref)
    assert float(base) == pytest.approx(float(moved), abs=1e-24)


def test_total_loss_is_pointwise_plus_the_weighted_energy_term():
    ref, descriptors, row_weight, segment = _loss_arrays()
    target = _parent_energy(ref, row_weight, segment, 2)
    for w_e in (0.5, 2.0):
        loss = _PretrainLoss(weights=jnp.ones(7),
                             energy_row_weight=row_weight,
                             energy_segment=segment, energy_target=target,
                             energy_weight=w_e, n_systems=2)
        pw, en = loss.parts(_EchoModel(0.3), descriptors, ref)
        assert float(loss(_EchoModel(0.3), descriptors, ref)) == \
            pytest.approx(float(pw) + w_e * float(en), rel=1e-12)


def test_zero_weight_returns_the_pre_existing_loss_bit_for_bit():
    """Default configuration: the energy term is not merely zero, it is not
    evaluated, so an existing run's loss value does not move."""
    ref, descriptors, row_weight, segment = _loss_arrays()
    target = _parent_energy(ref, row_weight, segment, 2)
    w = jnp.asarray([1.0, 2.0, 0.5, 3.0, 1.5, 1.0, 0.25])
    plain = _PretrainLoss(weights=w)
    armed = _PretrainLoss(weights=w, energy_row_weight=row_weight,
                          energy_segment=segment, energy_target=target,
                          energy_weight=0.0, n_systems=2)
    model = _EchoModel(0.4)
    a = float(plain(model, descriptors, ref))
    b = float(armed(model, descriptors, ref))
    assert a == b
    resid = (np.asarray(descriptors)[:, 0] + 0.4 - np.asarray(ref)) ** 2
    expect = float(np.sum(np.asarray(w) * resid) / (np.sum(np.asarray(w))
                                                    + 1e-12))
    assert a == pytest.approx(expect, rel=1e-12)


def test_energy_term_is_differentiable():
    """The term must reach the optimizer: a zero gradient would make it
    decorative. The offset leaf is a JAX scalar because ``eqx.filter_grad``
    differentiates inexact-array leaves only (a Python float rides along as
    static), exactly as a real network's weights are arrays."""
    ref, descriptors, row_weight, segment = _loss_arrays()
    target = _parent_energy(ref, row_weight, segment, 2)
    loss = _PretrainLoss(weights=jnp.ones(7), energy_row_weight=row_weight,
                         energy_segment=segment, energy_target=target,
                         energy_weight=1.0, n_systems=2)
    grad = eqx.filter_grad(loss)(_EchoModel(jnp.asarray(0.3)), descriptors,
                                 ref)
    assert abs(float(grad.offset)) > 1e-6


# ---------------------------------------------------------------------------
# _energy_term_inputs
# ---------------------------------------------------------------------------

def test_energy_term_inputs_pad_the_mesh_with_the_sink_segment():
    data = {"weights_all": jnp.asarray([1.0, 2.0, 4.0]),
            "e_lda_c_all": jnp.asarray([-1.0, -2.0, -0.5]),
            "system_all": jnp.asarray([0, 0, 1], dtype=jnp.int32),
            "e_c_parent_sys": jnp.asarray([-5.0, -2.0])}
    rw, seg, tgt, ns = _energy_term_inputs(
        data, weight_key="weights_all", lda_key="e_lda_c_all",
        segment_key="system_all", target_key="e_c_parent_sys", n_mesh=2)
    assert ns == 2
    np.testing.assert_allclose(np.asarray(rw), [-1.0, -4.0, -2.0, 0.0, 0.0])
    assert np.asarray(seg).tolist() == [0, 0, 1, 2, 2]
    np.testing.assert_allclose(np.asarray(tgt), [-5.0, -2.0])


# ---------------------------------------------------------------------------
# Row-block selection
# ---------------------------------------------------------------------------

def test_assemble_reads_the_exchange_block_on_request():
    arch = ArchitectureConfig.from_spec("t_plain", 2, 8)
    data = {"rho_all": jnp.ones(3), "sigma_all": jnp.zeros(3),
            "rho_x": jnp.full(5, 2.0), "sigma_x": jnp.full(5, 3.0)}
    assert _assemble_pretrain_descriptors(arch, data).shape == (3, 2)
    got = _assemble_pretrain_descriptors(arch, data, suffix="_x")
    assert got.shape == (5, 2)
    assert float(got[0, 0]) == 2.0


def test_assemble_refuses_a_correlation_row_set_that_is_not_the_total_density():
    """Correlation is spin-interpolated rather than spin-scaled and stays on the
    total density (von Barth and Hedin, J. Phys. C 5, 1629 (1972); Perdew and
    Wang, Phys. Rev. B 45, 13244 (1992)), so the cnet never reads the
    per-channel exchange block."""
    arch = ArchitectureConfig.from_spec("t_plain", 2, 8)
    with pytest.raises(ValueError, match="total density"):
        _assemble_pretrain_descriptors(arch, {"rho_x": jnp.ones(3),
                                              "sigma_x": jnp.ones(3)},
                                       for_cnet=True, suffix="_x")


# ---------------------------------------------------------------------------
# run_pretrain plumbing, on a real tiny .npz
# ---------------------------------------------------------------------------

_TINY = (("He", 0), ("H", 1))


@pytest.fixture(scope="module")
def tiny_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("energy_term")
    pdg.generate_pretrain_data_npz(
        str(d), atoms=_TINY, basis="sto-3g", grid_level=0, polarized=False,
        descriptors=True, exchange_footing="spin_channel")
    return str(d)


def _spec(tmp_path, data_dir, **kw):
    arch = ArchitectureConfig.from_spec("t_energy", 2, 8)
    return PretrainSpec(arch=arch, data_dir=data_dir,
                        checkpoint_dir=str(tmp_path / "ck"), n_steps=2,
                        seed=0, loss_weighting="integration", **kw)


def test_run_pretrain_records_the_energy_term(tiny_dir, tmp_path):
    md = run_pretrain(_spec(tmp_path, tiny_dir, energy_term_weight=1.0))
    assert md["energy_term_weight"] == 1.0
    assert md["n_systems"] == 2
    assert np.isfinite(md["energy_term_x_final"])
    assert np.isfinite(md["energy_term_c_final"])
    assert md["exchange_footing"] == "spin_channel"
    on_disk = json.load(open(os.path.join(tmp_path / "ck",
                                          "pretrain_metadata.json")))
    assert on_disk["energy_term_weight"] == 1.0


def test_run_pretrain_default_records_a_zero_weight(tiny_dir, tmp_path):
    md = run_pretrain(_spec(tmp_path, tiny_dir))
    assert md["energy_term_weight"] == 0.0


def test_run_pretrain_refuses_the_energy_term_without_a_system_index(tmp_path):
    d = tmp_path / "legacy"
    d.mkdir()
    np.savez(d / "pretrain_data.npz", rho_all=np.ones(4),
             sigma_all=np.zeros(4), Fx_all=np.zeros(4), Fc_all=np.zeros(4),
             Fx_scan_all=np.zeros(4), Fc_scan_all=np.zeros(4),
             metagga_all=np.zeros((4, 1)), weights_all=np.ones(4))
    with pytest.raises(ValueError, match="system_all"):
        run_pretrain(_spec(tmp_path, str(d), energy_term_weight=1.0))


def test_run_pretrain_refuses_a_file_built_on_the_wrong_parent_density(
        tiny_dir, tmp_path):
    """A meta-GGA architecture pretraining on a PBE-density file would be fit
    to a density its SCF never sees; the mismatch fails loudly instead."""
    arch = ArchitectureConfig.from_spec("t_mgga_parent", 2, 8,
                                        descriptors=["metagga"],
                                        meta_gga=True)
    spec = PretrainSpec(arch=arch, data_dir=tiny_dir,
                        checkpoint_dir=str(tmp_path / "ck_p"), n_steps=2,
                        seed=0, parent_density="auto")
    with pytest.raises(ValueError, match="parent"):
        run_pretrain(spec)
