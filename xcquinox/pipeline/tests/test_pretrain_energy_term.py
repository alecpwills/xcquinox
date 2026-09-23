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

import xcquinox.pipeline.pretrain_data_gen as pdg
from xcquinox.pipeline.config import ArchitectureConfig, PretrainSpec
from xcquinox.pipeline.pretrain import (
    _PretrainLoss, _assemble_pretrain_descriptors, _energy_term_inputs,
    run_pretrain)


class _EchoModel(eqx.Module):
    """A stand-in network whose enhancement factor is the row's first column
    plus a constant, so a test can make it reproduce a target exactly or miss
    it by a stated amount."""
    offset: float = 0.0

    def __call__(self, row):
        return 1.0 + row[0] + self.offset


class _TableModel(eqx.Module):
    """A stand-in network that returns a TABULATED enhancement factor.

    The descriptor row carries the row's own index, so the table can be
    libxc's enhancement factor at that row and the loss's reconstruction is
    then the parent's own energy rather than a network's guess at it.
    """
    table: jnp.ndarray

    def __call__(self, row):
        return self.table[jnp.asarray(row[0], dtype=jnp.int32)]


def _index_rows(n):
    """``(descriptors, ref_F)`` for a table model over ``n`` rows."""
    return (jnp.arange(n, dtype=jnp.float64).reshape(-1, 1), jnp.zeros(n))


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
    decorative. What is differentiated is the ENERGY PART ALONE -- the total
    loss carries a point-wise gradient that is non-zero at this offset
    whatever the energy term does, so differentiating the total would pass
    against a term that returned zero. The offset leaf is a JAX scalar
    because ``eqx.filter_grad`` differentiates inexact-array leaves only (a
    Python float rides along as static), exactly as a real network's weights
    are arrays. The closed form is d/dc mean_s (c R_s)^2 = 2 c mean_s R_s^2
    with row-weight sums R_0 = 3.5 and R_1 = 5.5, i.e. 12.75 at c = 0.3."""
    ref, descriptors, row_weight, segment = _loss_arrays()
    target = _parent_energy(ref, row_weight, segment, 2)
    loss = _PretrainLoss(weights=jnp.ones(7), energy_row_weight=row_weight,
                         energy_segment=segment, energy_target=target,
                         energy_weight=1.0, n_systems=2)
    grad = eqx.filter_grad(lambda m, d, r: loss.parts(m, d, r)[1])(
        _EchoModel(jnp.asarray(0.3)), descriptors, ref)
    assert float(grad.offset) == pytest.approx(12.75, rel=1e-12)


# ---------------------------------------------------------------------------
# _energy_term_inputs
# ---------------------------------------------------------------------------


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
    # Strictly positive, not merely finite: a term that returned zero would
    # be finite. Two optimizer steps from a random initialization leave the
    # network far from the parent's energies (measured 2e-2 / 5e-3 Ha^2).
    assert md["energy_term_x_final"] > 0.0
    assert md["energy_term_c_final"] > 0.0
    assert md["exchange_footing"] == "spin_channel"
    on_disk = json.load(open(os.path.join(tmp_path / "ck",
                                          "pretrain_metadata.json")))
    assert on_disk["energy_term_weight"] == 1.0


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


@pytest.fixture(scope="module")
def tiny_scan_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("energy_term_scan")
    pdg.generate_pretrain_data_npz(
        str(d), atoms=_TINY, basis="sto-3g", grid_level=0, polarized=False,
        descriptors=True, exchange_footing="spin_channel", reference_xc="scan")
    return str(d)


def test_run_pretrain_opens_the_scan_file_for_a_meta_gga_parent(
        tiny_scan_dir, tmp_path):
    """A meta-GGA run under the rung baseline pretrains on the SCAN-density
    file: the run opens ``pretrain_data_scan.npz`` (the only file in the
    directory) and records the SCAN parent. Before the parent was resolved
    ahead of the file name, the run opened the PBE name and failed for every
    non-PBE parent, the configuration every meta-GGA campaign uses."""
    # The SCAN file and its sidecar manifest (``<npz>.manifest.json``) are the
    # only things in the directory: no PBE file exists for the run to fall
    # back to, so the SCAN file is the one it opened.
    assert sorted(os.listdir(tiny_scan_dir)) == [
        "pretrain_data_scan.npz", "pretrain_data_scan.npz.manifest.json"]
    arch = ArchitectureConfig.from_spec("t_mgga_scan_file", 2, 8,
                                        descriptors=["metagga"], meta_gga=True)
    spec = PretrainSpec(arch=arch, data_dir=tiny_scan_dir,
                        checkpoint_dir=str(tmp_path / "ck_scan"), n_steps=2,
                        seed=0, parent_density="auto")
    md = run_pretrain(spec)
    assert md["reference_xc"] == "scan"


# ---------------------------------------------------------------------------
# The block the run actually read, and libxc's own enhancement factors
# ---------------------------------------------------------------------------

_SPIN_CHANNEL_SYSTEMS = (
    pdg.PretrainSystem("He", "He 0 0 0", 0, 0),
    pdg.PretrainSystem("Li", "Li 0 0 0", 0, 1),
)


@pytest.fixture(scope="module")
def spin_channel_dir(tmp_path_factory):
    """A real two-system file on the per-channel exchange footing.

    Li is what makes the two blocks DIFFERENT lengths: both of its spin
    channels are occupied, so its exchange rows are the doubled density of
    each channel while its correlation rows stay on the total density. He and
    H would not do -- a closed shell's exchange block IS its total-density
    block, and H's empty beta channel contributes no rows, so a He/H file has
    equal block lengths and cannot tell the two footings apart. Written
    spin-polarized so the stored zeta column can rebuild libxc's open-shell
    correlation call, and without the geometry descriptors, which nothing
    here reads.
    """
    d = tmp_path_factory.mktemp("energy_term_spin_channel")
    pdg.generate_pretrain_data_npz(
        str(d), systems=_SPIN_CHANNEL_SYSTEMS, basis="sto-3g", grid_level=0,
        polarized=True, descriptors=False, exchange_footing="spin_channel")
    return str(d)


def _spin_channel_arrays(spin_channel_dir):
    path = os.path.join(spin_channel_dir, "pretrain_data_polarized.npz")
    return {k: np.asarray(v) for k, v in np.load(path).items()}


def test_integration_weight_completeness_covers_the_exchange_block(tmp_path):
    """The recorded completeness flag was decided from ``weights_all`` alone
    while the exchange loss is built from the exchange block's own quadrature
    column, so a file carrying one and not the other recorded a complete
    weighting for a run that had none on the exchange side."""
    d = tmp_path / "no_weights_x"
    d.mkdir()
    np.savez(d / "pretrain_data.npz", rho_all=np.ones(4),
             sigma_all=np.zeros(4), Fx_all=np.zeros(4), Fc_all=np.zeros(4),
             Fx_scan_all=np.zeros(4), Fc_scan_all=np.zeros(4),
             metagga_all=np.zeros((4, 1)), weights_all=np.ones(4),
             rho_x=np.full(6, 2.0), sigma_x=np.zeros(6), Fx_x=np.zeros(6),
             Fx_scan_x=np.zeros(6), metagga_x=np.zeros((6, 1)))
    md = run_pretrain(_spec(tmp_path, str(d)))
    assert md["exchange_footing"] == "spin_channel"
    assert md["n_rows_x"] == 6 and md["n_rows_c"] == 4
    assert md["integration_weights_complete"] is False


def _pbe_eps(spec, rho, sigma, zeta=None):
    """libxc's own PBE energy per particle at the stored rows.

    ``spec`` is ``"PBE,"`` (exchange) or ``",PBE"`` (correlation). The
    gradient is rebuilt as one Cartesian component of length ``sqrt(sigma)``:
    both PBE pieces read the gradient only through the invariant ``sigma``, so
    any vector of that norm is the same input. With ``zeta`` supplied the call
    is the spin-polarized one, the channels split as
    ``rho_s = rho (1 +- zeta) / 2`` and the gradient split in the same ratio
    so the channel gradients sum to the stored total.
    """
    from pyscf.dft import libxc
    rho = np.asarray(rho, dtype=np.float64)
    grad = np.zeros((3, rho.shape[0]))
    grad[0] = np.sqrt(np.maximum(np.asarray(sigma, dtype=np.float64), 0.0))
    if zeta is None:
        return np.asarray(libxc.eval_xc(spec, np.vstack([rho, grad]),
                                        spin=0)[0], dtype=np.float64)
    frac_a = 0.5 * (1.0 + np.asarray(zeta, dtype=np.float64))
    rows_a = np.vstack([rho * frac_a, grad * frac_a])
    rows_b = np.vstack([rho * (1.0 - frac_a), grad * (1.0 - frac_a)])
    return np.asarray(libxc.eval_xc(spec, (rows_a, rows_b), spin=1)[0],
                      dtype=np.float64)


def _reconstruct(data, ratio, *, weight_key, lda_key, segment_key,
                 target_key, shift=None):
    """``(term, n_systems)`` from ``_PretrainLoss.parts`` with ``ratio`` in
    place of the network's enhancement factor, optionally after shifting the
    first system's target by ``shift`` Hartree."""
    jnp_data = {k: jnp.asarray(v) for k, v in data.items()}
    row_weight, segment, target, n_systems = _energy_term_inputs(
        jnp_data, weight_key=weight_key, lda_key=lda_key,
        segment_key=segment_key, target_key=target_key, n_mesh=0)
    if shift is not None:
        shifted = np.asarray(target).copy()
        shifted[0] += shift
        target = jnp.asarray(shifted)
    loss = _PretrainLoss(energy_row_weight=row_weight,
                         energy_segment=segment, energy_target=target,
                         energy_weight=1.0, n_systems=n_systems)
    descriptors, ref = _index_rows(int(np.asarray(ratio).shape[0]))
    _pointwise, energy = loss.parts(_TableModel(jnp.asarray(ratio)),
                                    descriptors, ref)
    return float(energy), n_systems


def test_libxc_exchange_factors_reconstruct_the_stored_system_energies(
        spin_channel_dir):
    """The parent's own enhancement factor, recomputed from libxc rather than
    read from the file, must reproduce the stored per-system exchange energies
    through the loss's own reconstruction.

    That is what makes the term an energy: the quadrature the loss performs is
    the one the target was built from, so a network reproducing the parent
    carries no energy error, and any residual is round-off. The LDA
    denominator is typed out here -- ``-(3/4)(3/pi)^(1/3)`` -- rather than
    imported, and is checked against the stored column before it is used.
    """
    d = _spin_channel_arrays(spin_channel_dir)
    lda_constant = -(3.0 / 4.0) * (3.0 / np.pi) ** (1.0 / 3.0)
    rho_x = d["rho_x"]
    np.testing.assert_allclose(rho_x * (lda_constant * np.cbrt(rho_x)),
                               d["e_lda_x_x"], rtol=1e-15, atol=0.0)
    # The doubled-density rows are posed for the SPIN-UNPOLARIZED call: that
    # is the exact-spin-scaling relation the footing implements.
    ratio = rho_x * _pbe_eps("PBE,", rho_x, d["sigma_x"]) / d["e_lda_x_x"]
    assert np.max(np.abs(d["Fx_x"] - (ratio - 1.0))) < 1e-13  # measured 4.4e-16
    assert not np.any(np.abs(d["Fx_x"]) >= 5.0)               # no clipped row
    term, n_systems = _reconstruct(
        d, ratio, weight_key="weights_x", lda_key="e_lda_x_x",
        segment_key="system_x", target_key="e_x_parent_sys")
    # The term is the MEAN of the squared per-system errors, so the largest
    # single error is at most sqrt(n_systems * term). Measured 4.4e-15 Ha on
    # this set; the bound is two orders above that and nine orders below the
    # Section 3.3 tolerance tol_atom = 1.0 mHa.
    assert float(np.sqrt(n_systems * term)) < 1e-12
    # A term that returned zero would satisfy the line above. Shifting one
    # system's target by +1 mHa must move the term by exactly delta^2 /
    # n_systems, because the reconstruction sits ON the target.
    shifted, _n = _reconstruct(
        d, ratio, weight_key="weights_x", lda_key="e_lda_x_x",
        segment_key="system_x", target_key="e_x_parent_sys", shift=1e-3)
    assert shifted == pytest.approx(1e-6 / n_systems, rel=1e-9)


def test_libxc_correlation_factors_reconstruct_the_stored_system_energies(
        spin_channel_dir):
    """The same closure on the correlation channel, which stays on the TOTAL
    density with zeta (correlation is spin-interpolated, not spin-scaled), so
    the enhancement factor comes from the spin-polarized libxc call rebuilt
    from the stored rho, zeta and sigma columns."""
    d = _spin_channel_arrays(spin_channel_dir)
    rho = d["rho_all"]
    ratio = rho * _pbe_eps(",PBE", rho, d["sigma_all"],
                           zeta=d["zeta_all"]) / d["e_lda_c_all"]
    assert np.max(np.abs(d["Fc_all"] - (ratio - 1.0))) < 1e-13  # 8.9e-16
    assert not np.any(np.abs(d["Fc_all"]) >= 5.0)
    term, n_systems = _reconstruct(
        d, ratio, weight_key="weights_all", lda_key="e_lda_c_all",
        segment_key="system_all", target_key="e_c_parent_sys")
    # Measured 2.5e-17 Ha on this set; the bound is four orders above.
    assert float(np.sqrt(n_systems * term)) < 1e-13
    shifted, _n = _reconstruct(
        d, ratio, weight_key="weights_all", lda_key="e_lda_c_all",
        segment_key="system_all", target_key="e_c_parent_sys", shift=1e-3)
    assert shifted == pytest.approx(1e-6 / n_systems, rel=1e-9)


# ---------------------------------------------------------------------------
# Held-out-system validation
# ---------------------------------------------------------------------------

def test_validation_holds_out_molecules_and_never_an_atom():
    """Every pool atom is a system the Section 3.3 certificate bounds at
    tol_atom = 1.0 mHa, and every atomization energy is anchored on atoms. A
    held-out atom would be an atom the fit never saw, so the split draws from
    the MOLECULES only."""
    from xcquinox.pipeline.pretrain import _validation_systems
    natoms = np.array([1, 1, 1, 2, 3, 5, 4, 2, 3, 10], dtype=np.int32)
    held = _validation_systems(natoms, 0.3, seed=0)
    assert held
    assert all(int(natoms[i]) > 1 for i in held)
    assert len(held) == 2  # round(0.3 * 7)


def test_validation_split_is_seeded_and_reproducible():
    from xcquinox.pipeline.pretrain import _validation_systems
    natoms = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.int32)
    a = _validation_systems(natoms, 0.5, seed=7)
    b = _validation_systems(natoms, 0.5, seed=7)
    c = _validation_systems(natoms, 0.5, seed=8)
    assert a == b
    assert a != c
    assert tuple(sorted(a)) == a


def test_training_loop_stops_on_patience_and_returns_the_best_weights():
    """The stop criterion replaces the DFS protocol's hand interruption (spec
    Section 6): training halts when the monitored validation quantity has not
    improved for ``patience`` validations, and the weights that are kept are
    the best ones seen, not the last ones."""
    import optax
    from xcquinox.pipeline.pretrain import _train_pretrain_network
    ref = jnp.asarray([0.0, 0.0])
    desc = jnp.stack([ref, jnp.ones(2)], axis=1)
    loss = _PretrainLoss(weights=jnp.ones(2))
    model, losses, record = _train_pretrain_network(
        _EchoModel(1.0), optax.sgd(1e-9), loss, desc, ref, loss, desc, ref,
        n_steps=100, validate_every=1, patience=3, monitor="pointwise")
    assert len(losses) < 100
    assert record["stopped_early"] is True
    # Step 0 is scored; on this flat trajectory the initialization ties and
    # the earlier candidate wins.
    assert record["best_step"] == 0
    assert len(record["history"]) == len(losses) + 1
    assert float(record["best_value"]) <= float(record["history"][0][1])


def test_training_loop_returns_the_best_model_not_the_last(tmp_path):
    """Training rows push the offset up; validation rows want it at zero, so
    every validation is worse than the one before. The model handed back is
    the one at the first validation, and the best-so-far checkpoint on disk
    is that same model."""
    import optax
    from xcquinox.pipeline.pretrain import _train_pretrain_network
    desc = jnp.stack([jnp.zeros(2), jnp.ones(2)], axis=1)
    ref_train = jnp.asarray([1.0, 1.0])
    ref_val = jnp.asarray([0.0, 0.0])
    loss = _PretrainLoss(weights=jnp.ones(2))
    ck = str(tmp_path / "best.eqx")
    start = _EchoModel(jnp.asarray(0.0))
    first, losses_a, rec_a = _train_pretrain_network(
        start, optax.sgd(0.1), loss, desc, ref_train, loss, desc, ref_val,
        n_steps=5, validate_every=1, patience=0, monitor="pointwise",
        checkpoint_path=ck)
    last, losses_b, rec_b = _train_pretrain_network(
        start, optax.sgd(0.1), loss, desc, ref_train, loss, desc, ref_val,
        n_steps=5, validate_every=5, patience=0, monitor="pointwise")
    assert losses_a == losses_b
    # The zero-offset start scores exactly 0 on the validation rows, so the
    # initialization (step 0) is the best candidate in BOTH schedules.
    assert rec_a["best_step"] == 0 and rec_b["best_step"] == 0
    assert float(first.offset) == 0.0 and float(last.offset) == 0.0
    vals = [h[3] for h in rec_a["history"]]
    assert vals == sorted(vals) and rec_a["best_value"] == vals[0]
    on_disk = eqx.tree_deserialise_leaves(ck, start)
    assert float(on_disk.offset) == 0.0


def test_training_loop_reproduces_the_trainer_on_identical_rows(tmp_path):
    """With nothing held out the validated loop is the same arithmetic as the
    pre-existing trainer: one full-batch Adam step per iteration on the same
    loss, so the loss trajectories and the final weights agree bit for bit."""
    import xcquinox.train
    from xcquinox.pipeline.networks import create_network_pair
    from xcquinox.pipeline.pretrain import _build_optimizer, _train_pretrain_network
    arch = ArchitectureConfig.from_spec("t_loop", 2, 8)
    xnet, _cnet = create_network_pair(arch, seed=0)
    rho = jnp.linspace(0.05, 2.0, 48)
    sigma = jnp.linspace(0.0, 1.5, 48)
    desc = jnp.stack([rho, sigma], axis=1)
    ref = 0.2 * jnp.tanh(sigma)
    loss = _PretrainLoss(weights=jnp.linspace(1.0, 3.0, 48))
    kw = dict(lr_start=1e-2, lr_end=1e-5, n_steps=6, lr_decay_start=0.5,
              grad_clip=1.0)
    trainer = xcquinox.train.xcTrainer(
        model=xnet, optim=_build_optimizer(**kw), loss=loss, steps=6,
        do_jit=True, serialize_every=0, checkpoint_dir=str(tmp_path))
    m_trainer, losses_trainer = trainer(1, [desc], [ref])
    m_loop, losses_loop, record = _train_pretrain_network(
        xnet, _build_optimizer(**kw), loss, desc, ref, loss, desc, ref,
        n_steps=6, validate_every=6, patience=0, monitor="pointwise")
    assert losses_loop == losses_trainer
    assert record["best_step"] == 6
    leaves_t = jax.tree_util.tree_leaves(eqx.filter(m_trainer, eqx.is_array))
    leaves_l = jax.tree_util.tree_leaves(eqx.filter(m_loop, eqx.is_array))
    assert len(leaves_t) == len(leaves_l) > 0
    for a, b in zip(leaves_t, leaves_l):
        assert np.array_equal(np.asarray(a), np.asarray(b))


_MOLECULES = (
    pdg.PretrainSystem("H", "H 0 0 0", 0, 1),
    pdg.PretrainSystem("Li", "Li 0 0 0", 0, 1),
    pdg.PretrainSystem("H2", "H 0 0 0; H 0 0 0.74", 0, 0),
    pdg.PretrainSystem("LiH", "Li 0 0 0; H 0 0 1.6", 0, 0),
)


@pytest.fixture(scope="module")
def molecule_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("energy_term_molecules")
    pdg.generate_pretrain_data_npz(
        str(d), systems=_MOLECULES, basis="sto-3g", grid_level=0,
        polarized=False, descriptors=False, exchange_footing="spin_channel")
    return str(d)


def test_run_pretrain_validation_holds_out_a_molecule_and_keeps_the_best(
        molecule_dir, tmp_path):
    """Two atoms and two molecules: the split holds out one MOLECULE by name,
    both networks are scored on it at the run's energy weight, the record
    carries the step of the best value, and the network written to disk is
    the best-validation one (the same bytes as the best-so-far snapshot)."""
    spec = PretrainSpec(arch=ArchitectureConfig.from_spec("t_energy", 2, 8),
                        data_dir=molecule_dir,
                        checkpoint_dir=str(tmp_path / "ck"), n_steps=6,
                        seed=0, loss_weighting="integration",
                        energy_term_weight=1.0, validation_fraction=0.5,
                        validation_seed=3, validate_every=1, patience=2)
    md = run_pretrain(spec)
    v = md["validation"]
    assert v["active"] is True
    assert (v["fraction"], v["seed"], v["validate_every"], v["patience"]) \
        == (0.5, 3, 1, 2)
    assert v["monitor"] == "loss"
    # The permutation is keyed on validation_seed, NOT on the network seed:
    # over this four-system set (H, Li, H2, LiH) seed 3 draws LiH while the
    # run's network seed 0 would draw H2, so the recorded name says which
    # seed was used.
    assert v["systems"] == ["LiH"]
    for key in ("x", "c"):
        rec = v[key]
        assert 0 <= rec["best_step"] <= rec["steps_run"] <= 6
        assert len(rec["history"]) == rec["steps_run"] + 1
        assert rec["best_value"] == min(h[3] for h in rec["history"])
        assert rec["n_rows_train"] > 0 and rec["n_rows_val"] > 0
        assert np.isfinite(rec["best_value"])
    ck = tmp_path / "ck"
    on_disk = json.load(open(os.path.join(ck, "pretrain_metadata.json")))
    assert on_disk["validation"]["systems"] == v["systems"]
    assert (ck / "xnet.eqx").read_bytes() == \
        (ck / "xnet" / "xnet_val_best.eqx").read_bytes()
    assert (ck / "cnet.eqx").read_bytes() == \
        (ck / "cnet" / "cnet_val_best.eqx").read_bytes()
    assert len(np.load(ck / "losses_x.npy")) == v["x"]["steps_run"]


# ---------------------------------------------------------------------------
# The stop point, the refusals, and what the record says about the artifact
# ---------------------------------------------------------------------------


def _write_system_npz(directory, *, natoms, segment=None, n_systems=None,
                      weights=None, fx=None, fc=None):
    """A GGA pretrain file with a stated system table.

    ``natoms`` is written verbatim, so a table of a length the energy tables
    cannot serve is expressible; ``n_systems`` fixes the length of the energy
    tables independently of the row index, so a system owning no row is too.
    """
    seg = (np.array([0, 0, 0, 1, 1, 1], dtype=np.int32) if segment is None
           else np.asarray(segment, dtype=np.int32))
    n = int(seg.shape[0])
    n_sys = int(n_systems) if n_systems is not None else int(seg.max()) + 1
    w = np.linspace(0.5, 1.5, n) if weights is None else np.asarray(weights,
                                                                    float)
    e_lda_x = -np.linspace(0.5, 1.5, n)
    e_lda_c = -np.linspace(0.05, 0.15, n)
    fx = np.full(n, 0.1) if fx is None else np.asarray(fx, float)
    fc = np.full(n, -0.2) if fc is None else np.asarray(fc, float)
    fx_s, fc_s = np.full(n, 0.3), np.full(n, -0.4)

    def targets(lda, factor):
        return np.array([float(np.sum(w[seg == s] * lda[seg == s]
                                      * (1.0 + factor[seg == s])))
                         for s in range(n_sys)])

    np.savez(os.path.join(directory, "pretrain_data.npz"),
             rho_all=np.linspace(0.1, 2.0, n),
             sigma_all=np.linspace(0.0, 1.0, n),
             # The iso-orbital alpha column, so a meta-GGA-rung architecture
             # can read this file too; a GGA-rung one never asks for it.
             metagga_all=np.linspace(0.0, 2.0, n).reshape(-1, 1),
             Fx_all=fx, Fc_all=fc, Fx_scan_all=fx_s, Fc_scan_all=fc_s,
             weights_all=w, e_lda_x_all=e_lda_x, e_lda_c_all=e_lda_c,
             system_all=seg,
             system_natoms=np.asarray(natoms, dtype=np.int32),
             e_x_parent_sys=targets(e_lda_x, fx),
             e_c_parent_sys=targets(e_lda_c, fc),
             e_x_parent_scan_sys=targets(e_lda_x, fx_s),
             e_c_parent_scan_sys=targets(e_lda_c, fc_s))
    return n


def test_run_pretrain_refuses_a_diverged_fit_and_writes_no_checkpoint(
        tmp_path):
    """Every validation non-finite means the best-model bookkeeping never
    improved on its ``inf`` seed, so the network the loop hands back is the
    untrained initialization. Writing that as ``xnet.eqx`` would put a random
    network behind the training stage with nothing but the Section 3.3
    certificate left to catch it, so the run fails by name instead."""
    from xcquinox.pipeline.pretrain import PretrainDiverged
    d = tmp_path / "diverged_x"
    d.mkdir()
    n = _write_system_npz(str(d), natoms=(2, 3))
    # A non-finite exchange target on every row: the loss is nan on both
    # sides of any split, whichever system is held out.
    with np.load(os.path.join(str(d), "pretrain_data.npz")) as raw:
        cols = {k: np.array(raw[k]) for k in raw.files}
    cols["Fx_all"] = np.full(n, np.nan)
    np.savez(os.path.join(str(d), "pretrain_data.npz"), **cols)
    ck = tmp_path / "ck"
    with pytest.raises(PretrainDiverged,
                       match="no finite validation value was recorded"):
        run_pretrain(_spec(tmp_path, str(d), validation_fraction=0.5,
                           validation_seed=3, validate_every=1, patience=2))
    assert not (ck / "xnet.eqx").exists()
    assert not (ck / "cnet.eqx").exists()
    # Step-0 scoring writes the initialization to the val-best path before
    # the refusal fires; a refused run must not leave that untrained
    # network behind (the summaries pull carries xnet/xnet_val_best.eqx).
    assert not (ck / "xnet" / "xnet_val_best.eqx").exists()
    failure = json.loads((ck / "pretrain_failed.json").read_text())
    assert failure["network"] == "xnet"
    assert failure["n_validations"] == 2
    assert failure["arch_name"] == "t_energy"
    # The record of a diverged run is itself strict JSON: its history is all
    # non-finite, which is exactly the case a bare json.dump would spell NaN.
    assert "NaN" not in (ck / "pretrain_failed.json").read_text()
    assert failure["history"][0][1] is None


# ---------------------------------------------------------------------------
# What the metadata says: strict JSON, the saved network, the run length
# ---------------------------------------------------------------------------

def _refuse_json_constants(token):
    raise AssertionError(f"non-RFC-8259 token {token!r} in the metadata")


def test_metadata_is_written_as_strict_json_with_null_for_non_finite(
        tmp_path):
    """RFC 8259 has no NaN or Infinity token. Python writes them anyway
    unless told not to, and a file carrying them is refused by every strict
    parser; the documented encoding here is ``null``."""
    from xcquinox.pipeline.pretrain import _json_safe, _write_metadata
    record = {"pos_inf": float("inf"), "neg_inf": float("-inf"),
              "nan": float("nan"), "finite": 1.5, "flag": True,
              "count": np.int64(3), "name": "x", "absent": None,
              "nested": {"h": [1.0, float("nan"), (2, float("inf"))]}}
    path = tmp_path / "md.json"
    _write_metadata(str(path), record)
    text = path.read_text()
    assert "NaN" not in text and "Infinity" not in text
    got = json.loads(text, parse_constant=_refuse_json_constants)
    assert got["pos_inf"] is None and got["neg_inf"] is None
    assert got["nan"] is None
    assert got["nested"]["h"] == [1.0, None, [2, None]]
    # Finite values are untouched, including the types json has no float for.
    assert got["finite"] == 1.5 and got["flag"] is True
    assert got["count"] == 3 and got["name"] == "x" and got["absent"] is None
    assert _json_safe(0.0) == 0.0 and _json_safe(-1.25) == -1.25


# ---------------------------------------------------------------------------
# The energy fidelity of the SAVED network, at any weight
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The unvalidated path leaves the periodic snapshots the trajectory scripts read
# ---------------------------------------------------------------------------

def _expected_snapshots(losses, serialize_every):
    """The ``xc.eqx.<step>`` names the trainer's rule produces for a curve.

    Written out here from ``xcquinox/train.py:148-152`` rather than called
    from the module under test: positive interval no larger than the
    schedule, 0-based index a multiple of it, and the loss strictly better
    than the last snapshot's (watermark seeded at 1e10).
    """
    n = len(losses)
    if serialize_every <= 0 or serialize_every > n:
        return []
    out, best = [], 1e10
    for k, value in enumerate(losses):
        if k % serialize_every == 0 and value < best:
            out.append(f"xc.eqx.{k}")
            best = value
    return out


def test_the_unvalidated_path_writes_the_periodic_snapshots(tmp_path):
    """With nothing held out the run keeps the periodic ``xc.eqx.<step>``
    family: two legacy trajectory scripts list that directory and select from
    it by step number, so a run that stopped leaving them would silently give
    those scripts nothing to restart from. The cadence is the trainer's --
    interval ``max(50, n_steps // 10)``, 0-based indices, and only where the
    loss improves on the last snapshot -- and the file at index ``k`` holds
    the network whose loss is ``losses[k]``, not the one the step produced."""
    from xcquinox.pipeline.networks import create_network_pair
    from xcquinox.pipeline.pretrain import _PRETRAIN_SERIALIZE_EVERY
    d = tmp_path / "snap"
    d.mkdir()
    _write_system_npz(str(d), natoms=(2, 3))
    arch = ArchitectureConfig.from_spec("t_snap", 2, 8)
    ck = tmp_path / "ck_snap"
    n_steps = 120
    md = run_pretrain(PretrainSpec(
        arch=arch, data_dir=str(d), checkpoint_dir=str(ck), n_steps=n_steps,
        seed=0, loss_weighting="unweighted"))
    assert md["validation"]["active"] is False
    every = _PRETRAIN_SERIALIZE_EVERY(n_steps)
    assert every == 50 and every <= n_steps
    for sub, curve in (("xnet", "losses_x.npy"), ("cnet", "losses_c.npy")):
        losses = [float(v) for v in np.load(ck / curve)]
        assert len(losses) == n_steps
        want = _expected_snapshots(losses, every)
        # A curve that improves throughout puts a snapshot at every due index;
        # the test is only meaningful if there is more than the first one.
        assert len(want) >= 2, want
        assert sorted(p.name for p in (ck / sub).iterdir()) == sorted(want)
    # The snapshot at index k is the network whose loss the run recorded as
    # losses[k] -- the model as it ENTERED step k+1, not the one that step
    # produced. Checked on the first snapshot, whose model is the untrained
    # initialization the run started from.
    x_skel, _c_skel = create_network_pair(arch, seed=0)
    first = eqx.tree_deserialise_leaves(str(ck / "xnet" / "xc.eqx.0"), x_skel)
    with np.load(os.path.join(str(d), "pretrain_data.npz")) as raw:
        data = {k: jnp.asarray(raw[k]) for k in raw.files}
    desc = _assemble_pretrain_descriptors(arch, data)
    got = float(jnp.mean(
        (jax.vmap(first)(desc).squeeze() - 1.0 - data["Fx_all"]) ** 2))
    assert got == pytest.approx(float(np.load(ck / "losses_x.npy")[0]),
                                rel=1e-12)
    initial = float(jnp.mean(
        (jax.vmap(x_skel)(desc).squeeze() - 1.0 - data["Fx_all"]) ** 2))
    assert got == pytest.approx(initial, rel=1e-12)


# ---------------------------------------------------------------------------
# A fit that overflowed is not a functional, validated or not
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The snapshot cadence against the trainer it was transcribed from
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The meta-GGA rung is one predicate, and four readers share it
# ---------------------------------------------------------------------------


def test_every_registered_architecture_states_its_rung_once():
    """Every architecture in the registry carries the flag and the descriptor
    together or carries neither, so the predicate and the flag answer the same
    for all of them and no registry entry was in the state the refusal now
    closes."""
    from xcquinox.pipeline.config import ARCHITECTURES, get_architecture
    for name in sorted(ARCHITECTURES):
        arch = get_architecture(name)
        assert ArchitectureConfig.is_meta_gga(arch) is bool(arch.meta_gga), name


