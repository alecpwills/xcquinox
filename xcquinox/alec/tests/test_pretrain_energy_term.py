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
    # Strictly positive, not merely finite: a term that returned zero would
    # be finite. Two optimizer steps from a random initialization leave the
    # network far from the parent's energies (measured 2e-2 / 5e-3 Ha^2).
    assert md["energy_term_x_final"] > 0.0
    assert md["energy_term_c_final"] > 0.0
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


def test_run_pretrain_records_the_exchange_block_it_actually_read(
        spin_channel_dir, tmp_path):
    """The recorded footing and row counts must come from the rows the loss
    was built on, not from the manifest.

    Forcing the exchange rows back to the total-density block leaves the
    manifest saying ``spin_channel`` while the fit uses the historical
    footing -- the Section 3.2 correction undone, with the checkpoint's own
    provenance asserting it was applied. The two blocks differ in length on
    this set (2502 exchange rows against 1576 correlation rows, measured), so
    the row counts alone identify the block.
    """
    arch = ArchitectureConfig.from_spec("t_footing", 2, 8,
                                        use_polarized_correlation=True)
    md = run_pretrain(PretrainSpec(
        arch=arch, data_dir=spin_channel_dir,
        checkpoint_dir=str(tmp_path / "ck_footing"), n_steps=2, seed=0,
        loss_weighting="integration", energy_term_weight=1.0))
    d = _spin_channel_arrays(spin_channel_dir)
    assert md["n_rows_x"] == int(d["rho_x"].shape[0])
    assert md["n_rows_c"] == int(d["rho_all"].shape[0])
    assert md["n_rows_x"] != md["n_rows_c"]
    assert md["exchange_footing"] == "spin_channel"
    assert md["n_systems"] == int(d["system_natoms"].shape[0]) == 2
    # Strictly positive, not merely finite: a term that returned zero would be
    # finite too. Measured 8.9e-3 / 5.2e-3 Ha^2 after two steps from a random
    # initialization.
    assert md["energy_term_x_final"] > 1e-6
    assert md["energy_term_c_final"] > 1e-6


def test_run_pretrain_refuses_a_declared_footing_the_file_cannot_serve(
        tmp_path):
    """A manifest claiming the per-channel footing beside a file with no
    exchange block would be pretrained at the historical footing in silence,
    which is exactly the state Section 3.2 exists to leave behind."""
    d = tmp_path / "footing_declared"
    d.mkdir()
    path = d / "pretrain_data.npz"
    np.savez(path, rho_all=np.ones(4), sigma_all=np.zeros(4),
             Fx_all=np.zeros(4), Fc_all=np.zeros(4), Fx_scan_all=np.zeros(4),
             Fc_scan_all=np.zeros(4), metagga_all=np.zeros((4, 1)),
             weights_all=np.ones(4))
    with open(str(path) + ".manifest.json", "w") as f:
        json.dump({"reference_xc": "pbe", "exchange_footing": "spin_channel"},
                  f)
    with pytest.raises(ValueError, match="spin_channel"):
        run_pretrain(_spec(tmp_path, str(d)))


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


def _write_mesh_npz(directory, share, *, n_atomic=6, n_mesh=4):
    """A synthetic meta-GGA file whose stored mesh share is ``share``."""
    np.savez(os.path.join(directory, "pretrain_data.npz"),
             rho_all=np.linspace(0.1, 2.0, n_atomic),
             sigma_all=np.linspace(0.0, 1.0, n_atomic),
             metagga_all=np.linspace(0.0, 2.0, n_atomic).reshape(-1, 1),
             Fx_all=np.zeros(n_atomic), Fc_all=np.zeros(n_atomic),
             Fx_scan_all=np.full(n_atomic, 0.3),
             Fc_scan_all=np.full(n_atomic, -0.4),
             weights_all=np.ones(n_atomic),
             rho_mesh=np.linspace(0.2, 1.0, n_mesh),
             sigma_mesh=np.linspace(0.1, 0.6, n_mesh),
             metagga_mesh=np.linspace(0.0, 3.0, n_mesh).reshape(-1, 1),
             Fx_scan_mesh=np.full(n_mesh, 0.7),
             Fc_scan_mesh=np.full(n_mesh, -0.7),
             weights_mesh=np.full(n_mesh, 0.25),
             mesh_weight_fraction=np.asarray(float(share)))


@pytest.mark.parametrize("share", (0.3, 0.5, 0.05))
def test_mesh_share_of_the_loss_follows_the_data(tmp_path, share):
    """``pretrain.mesh_fraction`` is a live YAML knob and the generator stores
    the value it built the file at. The mesh block's share of each channel's
    loss weight must be that stored value, not the generator's default: at
    0.3 -- the default -- a hard-coded constant is indistinguishable from
    reading the file, which is why 0.5 and 0.05 are measured beside it."""
    d = tmp_path / f"mesh_{share}"
    d.mkdir()
    _write_mesh_npz(str(d), share)
    arch = ArchitectureConfig.from_spec("t_mesh_share", 2, 8,
                                        descriptors=["metagga"],
                                        meta_gga=True)
    md = run_pretrain(PretrainSpec(
        arch=arch, data_dir=str(d), checkpoint_dir=str(d / "ck"), n_steps=2,
        seed=0, loss_weighting="integration"))
    assert md["pretrain_mesh"] is True
    assert md["mesh_weight_fraction"] == pytest.approx(share, rel=1e-12)
    assert md["mesh_loss_share_x"] == pytest.approx(share, rel=1e-12)
    assert md["mesh_loss_share_c"] == pytest.approx(share, rel=1e-12)


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
    from xcquinox.alec.pretrain import _validation_systems
    natoms = np.array([1, 1, 1, 2, 3, 5, 4, 2, 3, 10], dtype=np.int32)
    held = _validation_systems(natoms, 0.3, seed=0)
    assert held
    assert all(int(natoms[i]) > 1 for i in held)
    assert len(held) == 2  # round(0.3 * 7)


def test_validation_split_is_seeded_and_reproducible():
    from xcquinox.alec.pretrain import _validation_systems
    natoms = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.int32)
    a = _validation_systems(natoms, 0.5, seed=7)
    b = _validation_systems(natoms, 0.5, seed=7)
    c = _validation_systems(natoms, 0.5, seed=8)
    assert a == b
    assert a != c
    assert tuple(sorted(a)) == a


def test_validation_split_is_empty_at_zero_fraction():
    from xcquinox.alec.pretrain import _validation_systems
    natoms = np.array([1, 2, 3], dtype=np.int32)
    assert _validation_systems(natoms, 0.0, seed=0) == ()


def test_validation_split_never_takes_every_molecule():
    """A split that held out all the molecules would leave the fit with atoms
    only, which is the coverage failure the set change exists to remove."""
    from xcquinox.alec.pretrain import _validation_systems
    natoms = np.array([1, 2, 3], dtype=np.int32)
    held = _validation_systems(natoms, 1.0, seed=0)
    assert len(held) == 1


def test_validation_split_with_no_molecules_is_empty():
    from xcquinox.alec.pretrain import _validation_systems
    assert _validation_systems(np.ones(4, dtype=np.int32), 0.5, seed=0) == ()


def test_validation_split_rounds_half_up_and_floors_at_one_molecule():
    """A non-zero fraction always holds out at least one molecule (an empty
    split would leave the stop criterion with nothing to score), and a tie
    rounds up rather than to the even integer."""
    from xcquinox.alec.pretrain import _validation_systems
    two = np.array([1, 2, 2], dtype=np.int32)
    assert len(_validation_systems(two, 0.01, seed=0)) == 1
    five = np.array([1, 2, 2, 2, 2, 2], dtype=np.int32)
    assert len(_validation_systems(five, 0.5, seed=0)) == 3   # 2.5 -> 3
    assert len(_validation_systems(five, 0.3, seed=0)) == 2   # 1.5 -> 2


def test_split_arrays_keep_the_mesh_in_training():
    """The synthetic mesh regularizes the functional form; it is not a system
    whose energy is predicted, so holding it out would measure nothing."""
    from xcquinox.alec.pretrain import _system_split_arrays
    seg = np.array([0, 0, 1, 1, 2, 2, 3, 3], dtype=np.int32)  # 3 = sink
    train_mask, val_mask, train_remap, val_remap, train_ids, val_ids = \
        _system_split_arrays(seg, 3, (1,))
    assert train_mask.tolist() == [True, True, False, False, True, True,
                                   True, True]
    assert val_mask.tolist() == [False, False, True, True, False, False,
                                 False, False]
    assert train_ids.tolist() == [0, 2]
    assert val_ids.tolist() == [1]
    # Renumbering maps kept systems onto 0..n-1 and everything else onto the
    # sink index.
    assert train_remap[np.array([0, 2, 3])].tolist() == [0, 1, 2]
    assert int(train_remap[1]) == 2
    assert int(val_remap[1]) == 0


def test_restrict_loss_reindexes_the_energy_term():
    from xcquinox.alec.pretrain import _restrict_loss, _system_split_arrays
    ref = jnp.asarray([0.1, 0.2, 0.3, 0.4])
    desc = jnp.stack([ref, jnp.ones(4)], axis=1)
    seg = jnp.asarray([0, 0, 1, 1], dtype=jnp.int32)
    rw = jnp.asarray([1.0, 1.0, 2.0, 2.0])
    tgt = jnp.asarray([10.0, 20.0])
    full = _PretrainLoss(weights=jnp.ones(4), energy_row_weight=rw,
                         energy_segment=seg, energy_target=tgt,
                         energy_weight=1.0, n_systems=2)
    tm, vm, trm, vrm, tid, vid = _system_split_arrays(np.asarray(seg), 2, (1,))
    tr_loss, tr_desc, tr_ref = _restrict_loss(full, desc, ref, tm, trm, tid)
    assert tr_desc.shape == (2, 2)
    assert tr_loss.n_systems == 1
    assert np.asarray(tr_loss.energy_segment).tolist() == [0, 0]
    np.testing.assert_allclose(np.asarray(tr_loss.energy_target), [10.0])
    va_loss, va_desc, _va_ref = _restrict_loss(full, desc, ref, vm, vrm, vid)
    assert va_desc.shape == (2, 2)
    assert np.asarray(va_loss.energy_target).tolist() == [20.0]


def test_validation_split_keeps_the_mesh_share_on_the_training_side():
    """Under integration weighting the mesh block is a flat weight normalized
    to a stated share of the channel's total. Removing the held-out rows
    lowers the atomic total, so an untouched mesh block would pull harder on
    a validated fit than on an unvalidated one; the training restriction
    resets the block so the share is the one the data was built at, and the
    validation side carries no mesh row at all."""
    from xcquinox.alec.pretrain import _validation_split
    share = 0.3
    ref = jnp.asarray([0.1, -0.2, 0.3, 0.0, 0.5, -0.4, 0.0, 0.0])
    desc = jnp.stack([ref, jnp.ones(8)], axis=1)
    atomic = jnp.asarray([1.0, 2.0, 0.5, 3.0, 1.5, 1.0])
    mesh = jnp.full(2, float(jnp.sum(atomic)) * share / (1.0 - share) / 2)
    w = jnp.concatenate([atomic, mesh])
    seg = jnp.asarray([0, 0, 0, 1, 1, 1, 2, 2], dtype=jnp.int32)
    full = _PretrainLoss(weights=w)
    (tr_loss, tr_desc, _), (va_loss, va_desc, _) = _validation_split(
        full, desc, ref, np.asarray(seg), 2, (1,), n_mesh=2, mesh_share=share)
    wt = np.asarray(tr_loss.weights)
    assert tr_desc.shape == (5, 2)
    np.testing.assert_allclose(wt[:3], np.asarray(atomic)[:3])
    np.testing.assert_allclose(wt[3:].sum() / wt.sum(), share, rtol=1e-12)
    assert np.allclose(wt[3:], wt[3])
    assert va_desc.shape == (3, 2)
    assert np.asarray(va_loss.weights).shape == (3,)


def test_training_loop_stops_on_patience_and_returns_the_best_weights():
    """The stop criterion replaces the DFS protocol's hand interruption (spec
    Section 6): training halts when the monitored validation quantity has not
    improved for ``patience`` validations, and the weights that are kept are
    the best ones seen, not the last ones."""
    import optax
    from xcquinox.alec.pretrain import _train_pretrain_network
    ref = jnp.asarray([0.0, 0.0])
    desc = jnp.stack([ref, jnp.ones(2)], axis=1)
    loss = _PretrainLoss(weights=jnp.ones(2))
    model, losses, record = _train_pretrain_network(
        _EchoModel(1.0), optax.sgd(1e-9), loss, desc, ref, loss, desc, ref,
        n_steps=100, validate_every=1, patience=3, monitor="pointwise")
    assert len(losses) < 100
    assert record["stopped_early"] is True
    assert record["best_step"] >= 1
    assert len(record["history"]) == len(losses) // 1
    assert float(record["best_value"]) <= float(record["history"][0][1])


def test_training_loop_runs_to_the_end_without_patience():
    import optax
    from xcquinox.alec.pretrain import _train_pretrain_network
    ref = jnp.asarray([0.0, 0.0])
    desc = jnp.stack([ref, jnp.ones(2)], axis=1)
    loss = _PretrainLoss(weights=jnp.ones(2))
    _m, losses, record = _train_pretrain_network(
        _EchoModel(1.0), optax.sgd(1e-3), loss, desc, ref, loss, desc, ref,
        n_steps=10, validate_every=5, patience=0, monitor="pointwise")
    assert len(losses) == 10
    assert record["stopped_early"] is False


def test_training_loop_returns_the_best_model_not_the_last(tmp_path):
    """Training rows push the offset up; validation rows want it at zero, so
    every validation is worse than the one before. The model handed back is
    the one at the first validation, and the best-so-far checkpoint on disk
    is that same model."""
    import optax
    from xcquinox.alec.pretrain import _train_pretrain_network
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
    assert rec_a["best_step"] == 1 and rec_b["best_step"] == 5
    assert 0.0 < float(first.offset) < float(last.offset)
    vals = [h[3] for h in rec_a["history"]]
    assert vals == sorted(vals) and rec_a["best_value"] == vals[0]
    on_disk = eqx.tree_deserialise_leaves(ck, start)
    assert float(on_disk.offset) == float(first.offset)


def test_training_loop_monitors_the_validation_loss_at_the_run_weight():
    """What is scored between steps is the objective itself on the held-out
    systems -- point-wise term plus the energy term at the run's weight -- so
    the checkpoint kept is the one that generalizes on what was optimized."""
    import optax
    from xcquinox.alec.pretrain import _train_pretrain_network
    ref, descriptors, row_weight, segment = _loss_arrays()
    target = _parent_energy(ref, row_weight, segment, 2)
    w_e = 2.5
    loss = _PretrainLoss(weights=jnp.ones(7), energy_row_weight=row_weight,
                         energy_segment=segment, energy_target=target,
                         energy_weight=w_e, n_systems=2)
    _m, _l, record = _train_pretrain_network(
        _EchoModel(jnp.asarray(0.3)), optax.sgd(1e-3), loss, descriptors,
        ref, loss, descriptors, ref, n_steps=4, validate_every=2,
        patience=0, monitor="loss")
    assert record["monitor"] == "loss"
    assert len(record["history"]) == 2
    for _step, pointwise, energy, monitored in record["history"]:
        assert energy > 0.0
        assert monitored == pytest.approx(pointwise + w_e * energy, rel=1e-12)
    assert record["best_value"] == min(h[3] for h in record["history"])
    with pytest.raises(ValueError, match="monitor"):
        _train_pretrain_network(
            _EchoModel(jnp.asarray(0.3)), optax.sgd(1e-3), loss, descriptors,
            ref, loss, descriptors, ref, n_steps=1, validate_every=1,
            patience=0, monitor="energy")


def test_training_loop_reproduces_the_trainer_on_identical_rows(tmp_path):
    """With nothing held out the validated loop is the same arithmetic as the
    pre-existing trainer: one full-batch Adam step per iteration on the same
    loss, so the loss trajectories and the final weights agree bit for bit."""
    import xcquinox.train
    from xcquinox.alec.networks import create_network_pair
    from xcquinox.alec.pretrain import _build_optimizer, _train_pretrain_network
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


def test_run_pretrain_validation_records_the_held_out_systems(tiny_dir,
                                                              tmp_path):
    """The tiny file is two free atoms, so there is nothing to hold out: the
    split must be empty and the run must still complete."""
    md = run_pretrain(_spec(tmp_path, tiny_dir, validation_fraction=0.5,
                            patience=2, validate_every=1))
    assert md["validation"]["fraction"] == 0.5
    assert md["validation"]["systems"] == []
    assert md["validation"]["monitor"] == "pointwise"
    assert md["validation"]["active"] is False
    # No per-network record: with nothing held out the run goes through the
    # pre-existing trainer, which is what keeps it byte-identical.
    assert "x" not in md["validation"] and "c" not in md["validation"]


def test_run_pretrain_refuses_validation_without_a_system_table(tmp_path):
    """A file written before the system table exists cannot say which rows
    belong to which molecule; asking it for a held-out split is refused by
    name rather than silently trained without one."""
    d = tmp_path / "legacy_split"
    d.mkdir()
    np.savez(d / "pretrain_data.npz", rho_all=np.ones(4),
             sigma_all=np.zeros(4), Fx_all=np.zeros(4), Fc_all=np.zeros(4),
             Fx_scan_all=np.zeros(4), Fc_scan_all=np.zeros(4),
             metagga_all=np.zeros((4, 1)), weights_all=np.ones(4))
    with pytest.raises(ValueError, match="system_natoms"):
        run_pretrain(_spec(tmp_path, str(d), validation_fraction=0.5))


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
        assert 1 <= rec["best_step"] <= rec["steps_run"] <= 6
        assert len(rec["history"]) == rec["steps_run"]
        assert rec["best_value"] == min(h[3] for h in rec["history"])
        assert rec["n_rows_train"] > 0 and rec["n_rows_val"] > 0
        assert np.isfinite(rec["best_value"])
    ck = tmp_path / "ck"
    on_disk = json.load(open(os.path.join(ck, "pretrain_metadata.json")))
    assert on_disk["validation"]["systems"] == v["systems"]
    assert (ck / "xnet.eqx").read_bytes() == \
        (ck / "xnet" / "xc.eqx.best").read_bytes()
    assert (ck / "cnet.eqx").read_bytes() == \
        (ck / "cnet" / "xc.eqx.best").read_bytes()
    assert len(np.load(ck / "losses_x.npy")) == v["x"]["steps_run"]


def test_run_pretrain_validates_without_the_energy_term(molecule_dir,
                                                        tmp_path):
    """The split and the stop criterion do not depend on the energy term. At
    weight zero the monitored quantity is the point-wise residual on the
    held-out molecule -- the same objective the fit minimizes -- and the
    restricted loss carries no energy term to reindex."""
    spec = PretrainSpec(arch=ArchitectureConfig.from_spec("t_val_pw", 2, 8),
                        data_dir=molecule_dir,
                        checkpoint_dir=str(tmp_path / "ck_pw"), n_steps=4,
                        seed=0, loss_weighting="integration",
                        validation_fraction=0.5, validation_seed=3,
                        validate_every=2, patience=0)
    v = run_pretrain(spec)["validation"]
    assert v["active"] is True and v["monitor"] == "pointwise"
    # The same seed holds out the same molecule as the weighted run above:
    # the split is a property of the set and the seed, not of the objective.
    assert v["systems"] == ["LiH"]
    for key in ("x", "c"):
        rec = v[key]
        assert rec["steps_run"] == 4 and len(rec["history"]) == 2
        for _step, pointwise, energy, monitored in rec["history"]:
            assert energy == 0.0
            assert monitored == pointwise
        assert rec["best_value"] == min(h[3] for h in rec["history"])


def _write_mesh_system_npz(directory, share, *, n_mesh=4):
    """Synthetic meta-GGA rows carrying a two-molecule system table.

    Three rows per system, the per-system targets built from the stored
    columns by the generator's own expression ``sum_i w_i e_LDA_i (1 + F_i)``,
    so a network that reproduced the stored enhancement factors would carry
    no energy error.
    """
    seg = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    n_atomic = int(seg.shape[0])
    weights = np.linspace(0.5, 1.5, n_atomic)
    e_lda_x = -np.linspace(0.5, 1.5, n_atomic)
    e_lda_c = -np.linspace(0.05, 0.15, n_atomic)
    fx, fc = np.full(n_atomic, 0.1), np.full(n_atomic, -0.2)
    fx_scan, fc_scan = np.full(n_atomic, 0.3), np.full(n_atomic, -0.4)

    def targets(lda, factor):
        return np.array([float(np.sum(weights[seg == s] * lda[seg == s]
                                      * (1.0 + factor[seg == s])))
                         for s in (0, 1)])

    np.savez(os.path.join(directory, "pretrain_data.npz"),
             rho_all=np.linspace(0.1, 2.0, n_atomic),
             sigma_all=np.linspace(0.0, 1.0, n_atomic),
             metagga_all=np.linspace(0.0, 2.0, n_atomic).reshape(-1, 1),
             Fx_all=fx, Fc_all=fc, Fx_scan_all=fx_scan, Fc_scan_all=fc_scan,
             weights_all=weights, e_lda_x_all=e_lda_x, e_lda_c_all=e_lda_c,
             system_all=seg, system_natoms=np.array([2, 3], dtype=np.int32),
             e_x_parent_sys=targets(e_lda_x, fx),
             e_c_parent_sys=targets(e_lda_c, fc),
             e_x_parent_scan_sys=targets(e_lda_x, fx_scan),
             e_c_parent_scan_sys=targets(e_lda_c, fc_scan),
             rho_mesh=np.linspace(0.2, 1.0, n_mesh),
             sigma_mesh=np.linspace(0.1, 0.6, n_mesh),
             metagga_mesh=np.linspace(0.0, 3.0, n_mesh).reshape(-1, 1),
             Fx_scan_mesh=np.full(n_mesh, 0.7),
             Fc_scan_mesh=np.full(n_mesh, -0.7),
             weights_mesh=np.full(n_mesh, 0.25),
             mesh_weight_fraction=np.asarray(float(share)))
    return n_atomic, n_mesh


def test_run_pretrain_validates_a_mesh_carrying_meta_gga_run(tmp_path):
    """The mesh and the split together: the synthetic mesh rows belong to no
    system, so they stay on the TRAINING side whichever molecule is held out,
    and the row set the loss is restricted to must still line up with the
    descriptor tensor the mesh was concatenated onto (a padded segment array
    one row short would silently fold the mesh onto a molecule's energy)."""
    d = tmp_path / "mesh_split"
    d.mkdir()
    share = 0.4
    n_atomic, n_mesh = _write_mesh_system_npz(str(d), share)
    arch = ArchitectureConfig.from_spec("t_mesh_val", 2, 8,
                                        descriptors=["metagga"],
                                        meta_gga=True)
    md = run_pretrain(PretrainSpec(
        arch=arch, data_dir=str(d), checkpoint_dir=str(d / "ck"), n_steps=3,
        seed=0, loss_weighting="integration", energy_term_weight=1.0,
        validation_fraction=0.5, validation_seed=0, validate_every=1,
        patience=0))
    v = md["validation"]
    assert v["active"] is True and v["monitor"] == "loss"
    # No manifest beside a hand-written file, so the record names the held-out
    # system by its index rather than inventing one.
    assert v["systems"] in (["sys0"], ["sys1"])
    for key in ("x", "c"):
        rec = v[key]
        assert rec["n_rows_train"] == n_atomic // 2 + n_mesh
        assert rec["n_rows_val"] == n_atomic // 2
        assert rec["steps_run"] == 3 and 1 <= rec["best_step"] <= 3
    assert md["pretrain_mesh"] is True
    assert md["mesh_loss_share_x"] == pytest.approx(share, rel=1e-12)
