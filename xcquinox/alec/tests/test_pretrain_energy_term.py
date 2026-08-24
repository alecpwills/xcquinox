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


def test_pretrain_data_filename_follows_the_resolved_parent():
    """The file a run opens carries the parent's suffix: a meta-GGA architecture
    under the rung baseline ("auto") resolves to SCAN and therefore to the
    ``_scan`` file the datagen stage writes for it; a GGA architecture keeps
    the PBE name; an explicit parent wins over the rung."""
    from xcquinox.alec.pretrain import _pretrain_data_filename
    gga = ArchitectureConfig.from_spec("t_name_gga", 2, 8)
    mgga = ArchitectureConfig.from_spec("t_name_mgga", 2, 8,
                                        descriptors=["metagga"], meta_gga=True)
    assert _pretrain_data_filename(gga) == "pretrain_data.npz"
    assert _pretrain_data_filename(gga, "auto") == "pretrain_data.npz"
    assert _pretrain_data_filename(mgga, "auto") == "pretrain_data_scan.npz"
    assert _pretrain_data_filename(mgga, "scan") == "pretrain_data_scan.npz"
    assert _pretrain_data_filename(mgga, "pbe") == "pretrain_data.npz"
    assert _pretrain_data_filename(gga, "scan") == "pretrain_data_scan.npz"


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


def test_run_pretrain_names_the_missing_scan_file(tiny_dir, tmp_path):
    """With only the PBE-density file present, a meta-GGA run is refused with
    a message naming the parent it resolves to and the file it needs."""
    arch = ArchitectureConfig.from_spec("t_mgga_missing", 2, 8,
                                        descriptors=["metagga"], meta_gga=True)
    spec = PretrainSpec(arch=arch, data_dir=tiny_dir,
                        checkpoint_dir=str(tmp_path / "ck_m"), n_steps=2,
                        seed=0, parent_density="scan")
    with pytest.raises(ValueError) as excinfo:
        run_pretrain(spec)
    text = str(excinfo.value)
    assert "'scan' parent density" in text
    assert "pretrain_data_scan.npz" in text
    assert "only the PBE-density file" in text


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
        (ck / "xnet" / "xnet_val_best.eqx").read_bytes()
    assert (ck / "cnet.eqx").read_bytes() == \
        (ck / "cnet" / "cnet_val_best.eqx").read_bytes()
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


def _write_mesh_system_npz(directory, share, *, n_mesh=4, weights=None):
    """Synthetic meta-GGA rows carrying a two-molecule system table.

    Three rows per system, the per-system targets built from the stored
    columns by the generator's own expression ``sum_i w_i e_LDA_i (1 + F_i)``,
    so a network that reproduced the stored enhancement factors would carry
    no energy error. ``weights`` overrides the quadrature column; a column of
    zeros is the degenerate file whose integration weights sum to zero.
    """
    seg = np.array([0, 0, 0, 1, 1, 1], dtype=np.int32)
    n_atomic = int(seg.shape[0])
    weights = (np.linspace(0.5, 1.5, n_atomic) if weights is None
               else np.asarray(weights, dtype=np.float64))
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


# ---------------------------------------------------------------------------
# The stop point, the refusals, and what the record says about the artifact
# ---------------------------------------------------------------------------

def _worsening_run(*, every, patience, n_steps=40, lr=0.1, offset=0.0):
    """A fit whose held-out score worsens at every validation.

    The training rows want the offset at 1 and the held-out rows want it at 0,
    so plain gradient descent walks the offset monotonically from 0 towards 1
    and the held-out squared residual, ``offset^2``, rises at every score. The
    best validation is therefore the FIRST one and every later one is stale,
    which is the configuration the stop rule is stated for.
    """
    import optax
    from xcquinox.alec.pretrain import _train_pretrain_network
    desc = jnp.stack([jnp.zeros(2), jnp.ones(2)], axis=1)
    loss = _PretrainLoss(weights=jnp.ones(2))
    return _train_pretrain_network(
        _EchoModel(jnp.asarray(offset)), optax.sgd(lr), loss, desc,
        jnp.asarray([1.0, 1.0]), loss, desc, jnp.asarray([0.0, 0.0]),
        n_steps=n_steps, validate_every=every, patience=patience,
        monitor="pointwise")


@pytest.mark.parametrize("every,patience", [(1, 1), (1, 2), (2, 3), (3, 2)])
def test_the_stop_lands_exactly_at_best_step_plus_patience_intervals(
        every, patience):
    """``patience`` validations WITHOUT an improvement stop the run, so the
    last optimizer step taken is ``best_step + patience * validate_every``.
    The arithmetic is the whole content of the criterion: one validation of
    drift is ``validate_every`` optimizer steps per network per architecture,
    50 of them at the production interval, and ``steps_run`` would then
    disagree with the rule the record is read by."""
    _m, losses, rec = _worsening_run(every=every, patience=patience)
    assert rec["stopped_early"] is True
    # The first validation is the best one, so the rule is checkable against
    # a step this test knows independently of the record.
    assert rec["best_step"] == every
    assert rec["steps_run"] == rec["best_step"] + patience * every
    assert len(losses) == rec["steps_run"]
    assert len(rec["history"]) == 1 + patience
    scores = [h[3] for h in rec["history"]]
    assert scores == sorted(scores) and scores[0] == rec["best_value"]


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


def test_run_pretrain_refuses_a_system_table_the_energy_tables_cannot_serve(
        tmp_path):
    """The split renumbers the energy term's segment array through a table of
    its own length, and JAX CLAMPS an out-of-range gather instead of raising,
    so a table longer than the energy tables would silently fold one system's
    rows onto another system's energy rather than fail."""
    remap = jnp.asarray([0, 1, 2], dtype=jnp.int32)
    assert list(np.asarray(remap[jnp.asarray([0, 5, 99])])) == [0, 2, 2]
    d = tmp_path / "wrong_table"
    d.mkdir()
    _write_system_npz(str(d), natoms=(2, 3, 2))  # 3 systems listed, 2 energies
    with pytest.raises(ValueError, match="lists 3 systems"):
        run_pretrain(_spec(tmp_path, str(d), validation_fraction=0.5))


def test_run_pretrain_refuses_a_split_whose_side_owns_no_row(tmp_path):
    """A held-out system that contributes no row leaves the monitor a
    constant: the integration-weighted loss of an empty row set is 0.0 (the
    empty sum over the +1e-12 floor) and the unweighted one is nan, so no
    validation ever improves and the run keeps its first network. No
    generated file can reach that state, which is why it fails loudly."""
    d = tmp_path / "empty_side"
    d.mkdir()
    # Two systems in both tables; every row belongs to system 0, so holding
    # out system 1 -- which seed 3 does over two eligible systems -- leaves
    # the held-out side without a row.
    _write_system_npz(str(d), natoms=(2, 3), n_systems=2,
                      segment=[0, 0, 0, 0, 0, 0])
    with pytest.raises(ValueError, match="owns no row"):
        run_pretrain(_spec(tmp_path, str(d), validation_fraction=0.5,
                           validation_seed=3))


def test_run_pretrain_refuses_a_diverged_fit_and_writes_no_checkpoint(
        tmp_path):
    """Every validation non-finite means the best-model bookkeeping never
    improved on its ``inf`` seed, so the network the loop hands back is the
    untrained initialization. Writing that as ``xnet.eqx`` would put a random
    network behind the training stage with nothing but the Section 3.3
    certificate left to catch it, so the run fails by name instead."""
    from xcquinox.alec.pretrain import PretrainDiverged
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
    failure = json.loads((ck / "pretrain_failed.json").read_text())
    assert failure["network"] == "xnet"
    assert failure["n_validations"] == 2
    assert failure["arch_name"] == "t_energy"
    # The record of a diverged run is itself strict JSON: its history is all
    # non-finite, which is exactly the case a bare json.dump would spell NaN.
    assert "NaN" not in (ck / "pretrain_failed.json").read_text()
    assert failure["history"][0][1] is None


def test_a_diverged_correlation_fit_takes_the_exchange_checkpoint_with_it(
        tmp_path):
    """``xnet.eqx`` is written before the cnet phase so a job that dies there
    keeps it. A DIVERGED cnet is not that case -- the run has no product --
    so the half-pair is removed with the refusal."""
    from xcquinox.alec.pretrain import PretrainDiverged
    d = tmp_path / "diverged_c"
    d.mkdir()
    n = _write_system_npz(str(d), natoms=(2, 3))
    with np.load(os.path.join(str(d), "pretrain_data.npz")) as raw:
        cols = {k: np.array(raw[k]) for k in raw.files}
    cols["Fc_all"] = np.full(n, np.nan)
    np.savez(os.path.join(str(d), "pretrain_data.npz"), **cols)
    ck = tmp_path / "ck"
    with pytest.raises(PretrainDiverged, match="cnet"):
        run_pretrain(_spec(tmp_path, str(d), validation_fraction=0.5,
                           validation_seed=3, validate_every=1, patience=2))
    assert not (ck / "xnet.eqx").exists()
    assert not (ck / "cnet.eqx").exists()
    assert json.loads((ck / "pretrain_failed.json").read_text())["network"] \
        == "cnet"


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
    from xcquinox.alec.pretrain import _json_safe, _write_metadata
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


def test_a_non_finite_metadata_value_reaches_the_file_as_null(tmp_path):
    """A quadrature column of zeros leaves the integration weights summing to
    zero, so the mesh's share of them is undefined and the record says so."""
    d = tmp_path / "zero_weights"
    d.mkdir()
    n_atomic, n_mesh = _write_mesh_system_npz(str(d), 0.3,
                                              weights=np.zeros(6))
    assert (n_atomic, n_mesh) == (6, 4)
    arch = ArchitectureConfig.from_spec("t_zero_w", 2, 8,
                                        descriptors=["metagga"],
                                        meta_gga=True)
    md = run_pretrain(PretrainSpec(
        arch=arch, data_dir=str(d), checkpoint_dir=str(d / "ck"), n_steps=2,
        seed=0, loss_weighting="integration"))
    assert not np.isfinite(md["mesh_loss_share_x"])
    text = (d / "ck" / "pretrain_metadata.json").read_text()
    assert "NaN" not in text and "Infinity" not in text
    on_disk = json.loads(text, parse_constant=_refuse_json_constants)
    assert on_disk["mesh_loss_share_x"] is None
    assert on_disk["mesh_loss_share_c"] is None
    assert on_disk["mesh_weight_fraction"] == pytest.approx(0.3)


def test_final_loss_describes_the_saved_network_not_the_last_step(tmp_path):
    """``final_loss_x`` is the objective of the network on disk. The last
    entry of the training trajectory is the loss BEFORE the final update, so
    it describes a different network even when no early stop occurred; it is
    recorded beside it as ``last_step_loss_x``."""
    from xcquinox.alec.networks import create_network_pair
    d = tmp_path / "saved_loss"
    d.mkdir()
    n = _write_system_npz(str(d), natoms=(2, 3))
    arch = ArchitectureConfig.from_spec("t_saved", 2, 8)
    ck = tmp_path / "ck_saved"
    md = run_pretrain(PretrainSpec(
        arch=arch, data_dir=str(d), checkpoint_dir=str(ck), n_steps=4, seed=0,
        loss_weighting="unweighted"))
    losses_x = np.load(ck / "losses_x.npy")
    losses_c = np.load(ck / "losses_c.npy")
    assert md["last_step_loss_x"] == float(losses_x[-1])
    assert md["last_step_loss_c"] == float(losses_c[-1])
    # Reconstructed independently: the plain mean of squared residuals of the
    # DESERIALIZED network on the same rows.
    with np.load(os.path.join(str(d), "pretrain_data.npz")) as raw:
        data = {k: jnp.asarray(raw[k]) for k in raw.files}
    x_skel, c_skel = create_network_pair(arch, seed=0)
    xnet = eqx.tree_deserialise_leaves(str(ck / "xnet.eqx"), x_skel)
    cnet = eqx.tree_deserialise_leaves(str(ck / "cnet.eqx"), c_skel)
    desc = _assemble_pretrain_descriptors(arch, data)
    desc_c = _assemble_pretrain_descriptors(arch, data, for_cnet=True)
    got_x = float(jnp.mean(
        (jax.vmap(xnet)(desc).squeeze() - 1.0 - data["Fx_all"]) ** 2))
    got_c = float(jnp.mean(
        (jax.vmap(cnet)(desc_c).squeeze() - 1.0 - data["Fc_all"]) ** 2))
    assert md["final_loss_x"] == pytest.approx(got_x, rel=1e-12)
    assert md["final_loss_c"] == pytest.approx(got_c, rel=1e-12)
    # The two keys are different numbers: the fit moved on the last step.
    assert md["final_loss_x"] != md["last_step_loss_x"]
    assert md["final_loss_c"] != md["last_step_loss_c"]
    assert n == 6


def test_the_record_states_the_steps_requested_and_the_steps_run(
        molecule_dir, tmp_path):
    """An early stop makes the requested schedule and the run length differ.
    ``pretrain_steps`` keeps the requested value, which is what the run
    validator compares against the configuration, and the length the loss
    curves actually hold is recorded beside it."""
    spec = PretrainSpec(arch=ArchitectureConfig.from_spec("t_steps", 2, 8),
                        data_dir=molecule_dir,
                        checkpoint_dir=str(tmp_path / "ck_steps"), n_steps=12,
                        seed=0, loss_weighting="integration",
                        lr_start=3e-1, lr_end=3e-1,
                        energy_term_weight=1.0, validation_fraction=0.5,
                        validation_seed=3, validate_every=1, patience=2)
    md = run_pretrain(spec)
    assert md["pretrain_steps"] == 12
    assert md["pretrain_steps_requested"] == 12
    ck = tmp_path / "ck_steps"
    n_x = len(np.load(ck / "losses_x.npy"))
    n_c = len(np.load(ck / "losses_c.npy"))
    assert md["pretrain_steps_run"] == max(n_x, n_c)
    assert md["validation"]["x"]["stopped_early"] is True
    assert md["pretrain_steps_run"] < md["pretrain_steps_requested"]
    # The saved network is the best-validation one, several steps behind the
    # last one stepped to, so the two loss keys cannot be the same number.
    assert md["validation"]["x"]["best_step"] < md["validation"]["x"]["steps_run"]
    assert md["final_loss_x"] != md["last_step_loss_x"]
    # The transient best-so-far snapshot carries a name no ``xc.eqx.<step>``
    # sort sees: two legacy trajectory scripts list this directory and key on
    # ``int(name.split('.')[-1])`` over names containing 'xc.eqx'.
    assert sorted(p.name for p in (ck / "xnet").iterdir()) \
        == ["xnet_val_best.eqx"]
    assert sorted(p.name for p in (ck / "cnet").iterdir()) \
        == ["cnet_val_best.eqx"]
    for sub in ("xnet", "cnet"):
        assert [p.name for p in (ck / sub).iterdir() if "xc.eqx" in p.name] \
            == []


def test_the_mesh_banner_prints_the_measured_share(tmp_path, capsys):
    """Under the unweighted reduction every row counts once, so the mesh's
    share of the loss is a ROW COUNT and not the share the data was built at.
    The banner prints the quantity the metadata records, off the same weight
    vector, so the log and the record cannot disagree."""
    d = tmp_path / "mesh_banner"
    d.mkdir()
    n_atomic, n_mesh = _write_mesh_system_npz(str(d), 0.3)
    arch = ArchitectureConfig.from_spec("t_banner", 2, 8,
                                        descriptors=["metagga"],
                                        meta_gga=True)
    md = run_pretrain(PretrainSpec(
        arch=arch, data_dir=str(d), checkpoint_dir=str(d / "ck"), n_steps=2,
        seed=0, loss_weighting="unweighted"))
    row_share = n_mesh / float(n_atomic + n_mesh)
    assert md["mesh_weight_fraction"] == pytest.approx(0.3)
    assert md["mesh_loss_share_x"] == pytest.approx(row_share, rel=1e-12)
    printed = capsys.readouterr().out
    banner = [ln for ln in printed.splitlines()
              if "mesh share of each channel" in ln]
    assert len(banner) == 1, printed
    assert f"x {md['mesh_loss_share_x']:.4f}" in banner[0]
    assert f"c {md['mesh_loss_share_c']:.4f}" in banner[0]
    # The share the data was built at is not the share this run felt.
    assert f"x {0.3:.4f}" not in banner[0]


# ---------------------------------------------------------------------------
# The energy fidelity of the SAVED network, at any weight
# ---------------------------------------------------------------------------

def _reconstruct_saved_energy(arch, data_dir, checkpoint_dir, *, seed=0):
    """``(term_x, term_c, delta_x, delta_c)`` rebuilt from the file and the
    checkpoints, through ``_PretrainLoss.parts`` on the deserialized nets."""
    from xcquinox.alec.networks import create_network_pair
    with np.load(os.path.join(data_dir, "pretrain_data.npz")) as raw:
        data = {k: jnp.asarray(raw[k]) for k in raw.files}
    suffix = "_x" if "rho_x" in data else "_all"
    x_skel, c_skel = create_network_pair(arch, seed=seed)
    xnet = eqx.tree_deserialise_leaves(
        os.path.join(checkpoint_dir, "xnet.eqx"), x_skel)
    cnet = eqx.tree_deserialise_leaves(
        os.path.join(checkpoint_dir, "cnet.eqx"), c_skel)
    desc_x = _assemble_pretrain_descriptors(arch, data, suffix=suffix)
    desc_c = _assemble_pretrain_descriptors(arch, data, for_cnet=True)
    out = []
    for net, desc, wk, lk, sk, tk, ref in (
            (xnet, desc_x, "weights" + suffix, "e_lda_x" + suffix,
             "system" + suffix, "e_x_parent_sys", data["Fx" + suffix]),
            (cnet, desc_c, "weights_all", "e_lda_c_all", "system_all",
             "e_c_parent_sys", data["Fc_all"])):
        rw, seg, tgt, ns = _energy_term_inputs(
            data, weight_key=wk, lda_key=lk, segment_key=sk, target_key=tk,
            n_mesh=0)
        loss = _PretrainLoss(energy_row_weight=rw, energy_segment=seg,
                             energy_target=tgt, n_systems=ns,
                             energy_weight=0.0)
        out.append((float(loss.parts(net, desc, ref)[1]),
                    np.asarray(loss.system_energy_errors(net, desc))))
    (term_x, delta_x), (term_c, delta_c) = out
    return term_x, term_c, delta_x, delta_c


@pytest.mark.parametrize("weight", (0.0, 1.0))
def test_the_energy_term_recorded_is_the_saved_network_at_any_weight(
        molecule_dir, tmp_path, weight):
    """At weight zero the objective SHORT-CIRCUITS the energy term -- that is
    what keeps its trajectory byte-identical to the pre-protocol one -- so a
    metadata value taken from the fitted loss is identically 0.0 there and
    reads as a perfect fit against a network whose per-system error is not
    zero. The record therefore measures the term on the SAVED network
    whatever the weight was, and states the certificate's own quantity, the
    per-system maximum, in mHa."""
    ck = tmp_path / f"ck_w{weight}"
    arch = ArchitectureConfig.from_spec("t_energy_any", 2, 8)
    md = run_pretrain(PretrainSpec(
        arch=arch, data_dir=molecule_dir, checkpoint_dir=str(ck), n_steps=2,
        seed=0, loss_weighting="integration", energy_term_weight=weight))
    assert md["energy_term_weight"] == weight
    assert md["energy_term_x_final"] > 0.0
    assert md["energy_term_c_final"] > 0.0
    term_x, term_c, delta_x, delta_c = _reconstruct_saved_energy(
        arch, molecule_dir, str(ck))
    assert md["energy_term_x_final"] == pytest.approx(term_x, rel=1e-12)
    assert md["energy_term_c_final"] == pytest.approx(term_c, rel=1e-12)
    assert md["energy_term_max_abs_dE_mHa"] == pytest.approx(
        1000.0 * float(np.max(np.abs(delta_x + delta_c))), rel=1e-12)
    assert md["energy_term_rms_dE_mHa"] == pytest.approx(
        1000.0 * float(np.sqrt(term_x + term_c)), rel=1e-12)
    # The maximum is a real per-system number, not a mean dressed up as one.
    assert md["energy_term_max_abs_dE_mHa"] > 0.0
    on_disk = json.loads((ck / "pretrain_metadata.json").read_text(),
                         parse_constant=_refuse_json_constants)
    assert on_disk["energy_term_max_abs_dE_mHa"] == \
        md["energy_term_max_abs_dE_mHa"]


def test_a_file_without_an_energy_table_records_null_not_zero(tmp_path):
    """A file predating the per-system energy tables cannot say what the
    saved network's energy error is; the record says so rather than
    reporting a zero it did not measure."""
    d = tmp_path / "no_energy_table"
    d.mkdir()
    np.savez(os.path.join(str(d), "pretrain_data.npz"),
             rho_all=np.linspace(0.1, 2.0, 4),
             sigma_all=np.linspace(0.0, 1.0, 4),
             Fx_all=np.zeros(4), Fc_all=np.zeros(4),
             Fx_scan_all=np.zeros(4), Fc_scan_all=np.zeros(4),
             weights_all=np.ones(4))
    md = run_pretrain(_spec(tmp_path, str(d)))
    assert md["energy_term_x_final"] is None
    assert md["energy_term_c_final"] is None
    assert md["energy_term_max_abs_dE_mHa"] is None
    assert md["energy_term_rms_dE_mHa"] is None


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
    from xcquinox.alec.networks import create_network_pair
    from xcquinox.alec.pretrain import _PRETRAIN_SERIALIZE_EVERY
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


def test_a_short_unvalidated_run_writes_no_snapshot_at_all(tmp_path):
    """The interval a run asks for, ``max(50, n_steps // 10)``, exceeds any
    schedule below 50 steps, and the trainer's rule declines to serialise at
    all when the interval is longer than the run. Short runs therefore leave
    the two net subdirectories empty, which is what they did before."""
    d = tmp_path / "short"
    d.mkdir()
    _write_system_npz(str(d), natoms=(2, 3))
    run_pretrain(_spec(tmp_path, str(d)))  # n_steps = 2
    ck = tmp_path / "ck"  # the checkpoint_dir _spec builds
    assert list((ck / "xnet").iterdir()) == []
    assert list((ck / "cnet").iterdir()) == []


def test_the_snapshot_rule_is_the_trainers(tmp_path):
    """``_snapshot_due`` transcribed against the same rule written out
    independently, over the interval bounds, the 0-based multiples and the
    improvement gate."""
    from xcquinox.alec.pretrain import _snapshot_due
    losses = [0.5, 0.4, 0.9, 0.3, 0.35, 0.2, 0.1, 0.05]
    for every in (0, 1, 2, 3, 9):
        best, got = 1e10, []
        for k, value in enumerate(losses):
            if _snapshot_due(k, len(losses), every, value, best):
                got.append(f"xc.eqx.{k}")
                best = value
        assert got == _expected_snapshots(losses, every), (every, got)
    # An interval longer than the schedule writes nothing, and a due step
    # whose loss did not improve is skipped.
    assert _snapshot_due(0, 4, 5, 0.1, 1e10) is False
    assert _snapshot_due(2, 8, 2, 0.9, 0.4) is False
    assert _snapshot_due(2, 8, 2, 0.3, 0.4) is True


# ---------------------------------------------------------------------------
# A fit that overflowed is not a functional, validated or not
# ---------------------------------------------------------------------------

def _nan_target_dir(directory, column):
    """A pretrain file whose ``column`` is non-finite on every row."""
    _write_system_npz(directory, natoms=(2, 3))
    path = os.path.join(directory, "pretrain_data.npz")
    with np.load(path) as raw:
        cols = {k: np.array(raw[k]) for k in raw.files}
    cols[column] = np.full(cols[column].shape, np.nan)
    np.savez(path, **cols)
    return path


def test_an_unvalidated_run_refuses_a_non_finite_trajectory(tmp_path):
    """With nothing held out there is no monitor to catch an overflow, and the
    network the fit hands back is one of NaNs that serialises and loads like
    any other. The trajectory is the criterion there: one non-finite entry and
    the run has no product."""
    from xcquinox.alec.pretrain import PretrainDiverged
    d = tmp_path / "nan_x"
    d.mkdir()
    _nan_target_dir(str(d), "Fx_all")
    ck = tmp_path / "ck"
    with pytest.raises(PretrainDiverged, match="non-finite training loss"):
        run_pretrain(_spec(tmp_path, str(d)))  # validation_fraction 0
    assert not (ck / "xnet.eqx").exists()
    assert not (ck / "cnet.eqx").exists()
    failure = json.loads((ck / "pretrain_failed.json").read_text())
    assert failure["network"] == "xnet"
    assert failure["reason"] == "a recorded training loss is non-finite"
    assert failure["first_non_finite_step"] == 1
    assert failure["n_non_finite"] == failure["steps_run"] == 2
    # Strict JSON: the trajectory it records is entirely non-finite.
    assert "NaN" not in (ck / "pretrain_failed.json").read_text()
    assert failure["losses"] == [None, None]


def test_the_refusal_keeps_the_periodic_snapshots_and_drops_the_finals(
        tmp_path):
    """The periodic ``xc.eqx.<step>`` files predate the overflow -- the
    improvement gate cannot pass a non-finite value, so every snapshot on disk
    was written from a finite loss -- and they are the last good state a
    restart would want. What must not survive is the pair downstream loads."""
    from xcquinox.alec.pretrain import PretrainDiverged
    d = tmp_path / "nan_c"
    d.mkdir()
    _nan_target_dir(str(d), "Fc_all")  # the xnet fits, the cnet overflows
    ck = tmp_path / "ck_nan_c"
    n_steps = 120
    with pytest.raises(PretrainDiverged, match="cnet"):
        run_pretrain(PretrainSpec(
            arch=ArchitectureConfig.from_spec("t_nan_c", 2, 8),
            data_dir=str(d), checkpoint_dir=str(ck), n_steps=n_steps, seed=0,
            loss_weighting="unweighted"))
    # The xnet's own snapshots were written from finite losses and are kept,
    # while the final pair -- including the xnet.eqx this run had already
    # persisted before the cnet phase -- is gone.
    kept = sorted(p.name for p in (ck / "xnet").iterdir())
    assert kept and all(n.startswith("xc.eqx.") for n in kept)
    assert not (ck / "xnet.eqx").exists()
    assert not (ck / "cnet.eqx").exists()
    # The cnet's trajectory is non-finite from its first step, so the gate
    # never passed and it left no snapshot of its own.
    assert list((ck / "cnet").iterdir()) == []
    failure = json.loads((ck / "pretrain_failed.json").read_text())
    assert failure["network"] == "cnet"
    assert failure["first_non_finite_step"] == 1


def test_a_saturating_network_keeps_its_loss_finite_under_a_huge_rate(
        tmp_path):
    """Why the refusal is written against the TRAJECTORY and not against the
    learning rate: the pretrained networks are bounded by construction (the
    enhancement factor is clamped), so even at a rate of 1e14 the loss stays
    finite and there is nothing to refuse. An overflow reaches these runs
    through the DATA, which is what the two tests above drive."""
    d = tmp_path / "huge_lr"
    d.mkdir()
    _write_system_npz(str(d), natoms=(2, 3))
    ck = tmp_path / "ck_huge"
    md = run_pretrain(PretrainSpec(
        arch=ArchitectureConfig.from_spec("t_huge", 2, 8), data_dir=str(d),
        checkpoint_dir=str(ck), n_steps=6, seed=0,
        loss_weighting="unweighted", lr_start=1e14, lr_end=1e14,
        grad_clip=1e30))
    assert np.all(np.isfinite(np.load(ck / "losses_x.npy")))
    assert np.isfinite(md["final_loss_x"])
    assert (ck / "xnet.eqx").is_file() and (ck / "cnet.eqx").is_file()


# ---------------------------------------------------------------------------
# The snapshot cadence against the trainer it was transcribed from
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("lr,expect_gap", ((0.05, False), (1.1, True)))
def test_the_snapshot_cadence_matches_the_library_trainer(tmp_path, lr,
                                                          expect_gap):
    """The cadence rule lives in two places -- ``xcquinox/train.py`` and, so
    the pretraining loop leaves the same files behind, transcribed into
    ``_PRETRAIN_SERIALIZE_EVERY`` / ``_snapshot_due``. This runs the LIBRARY
    trainer for the same schedule into its own directory and requires the set
    of ``xc.eqx.<k>`` indices it writes to be the set the transcription
    predicts from the trainer's own loss curve, so a change upstream turns
    this red rather than silently splitting the two families.

    Both branches of the rule are exercised: a converging rate writes at every
    due index, a diverging one improves only at the first and skips the rest.
    """
    import optax
    import xcquinox.train
    from xcquinox.alec.pretrain import (_PRETRAIN_SERIALIZE_EVERY,
                                        _snapshot_due)
    n_steps = 120
    every = _PRETRAIN_SERIALIZE_EVERY(n_steps)
    assert every == 50 and every <= n_steps
    d = tmp_path / f"traj_{lr}"
    d.mkdir()
    desc = jnp.stack([jnp.zeros(2), jnp.ones(2)], axis=1)
    ref = jnp.asarray([1.0, 1.0])
    trainer = xcquinox.train.xcTrainer(
        model=_EchoModel(jnp.asarray(0.0)), optim=optax.sgd(lr),
        loss=_PretrainLoss(weights=jnp.ones(2)), steps=n_steps, do_jit=True,
        serialize_every=every, checkpoint_dir=str(d))
    _model, losses = trainer(1, [desc], [ref])
    assert len(losses) == n_steps
    written = sorted(int(p.name.rsplit(".", 1)[1]) for p in d.iterdir()
                     if p.name.startswith("xc.eqx."))
    best, predicted = 1e10, []
    for k, value in enumerate(losses):
        if _snapshot_due(k, n_steps, every, value, best):
            predicted.append(k)
            best = value
    assert written == predicted, (written, predicted)
    assert written[0] == 0
    # The two cells differ in whether the improvement gate skips a due index.
    assert (written != [0, 50, 100]) is expect_gap, (lr, written)


# ---------------------------------------------------------------------------
# The meta-GGA rung is one predicate, and four readers share it
# ---------------------------------------------------------------------------

def _mgga_and_gga():
    """A meta-GGA-rung and a GGA-rung architecture from the registry."""
    from xcquinox.alec.config import get_architecture
    return get_architecture("deep_mgga_3x16"), get_architecture("deep_3x16")


def test_an_architecture_whose_rung_statements_disagree_is_refused():
    """The rung is one fact stated twice -- the ``meta_gga`` flag switches the
    DFS UEG gate and the Lieb-Oxford ceiling, the ``metagga`` descriptor
    supplies the iso-orbital alpha that gate reads -- so the two must agree at
    CONSTRUCTION rather than be reconciled by whichever reader gets there
    first. Both directions are refused, and the message names both."""
    from xcquinox.alec.config import FeatureSpec
    with pytest.raises(ValueError) as descriptor_only:
        ArchitectureConfig(name="descriptor_only", depth=3, nodes=16,
                           descriptors=(FeatureSpec(name="metagga"),))
    text = str(descriptor_only.value)
    assert "meta_gga=False" in text and "'metagga'" in text
    assert "descriptor_only" in text
    with pytest.raises(ValueError) as flag_only:
        ArchitectureConfig(name="flag_only", depth=3, nodes=16,
                           descriptors=(FeatureSpec(name="cusp"),),
                           meta_gga=True)
    text = str(flag_only.value)
    assert "meta_gga=True" in text and "does not carry" in text
    # Both together, and neither, are architectures.
    both = ArchitectureConfig.from_spec("both", 3, 16,
                                        descriptors=["metagga"],
                                        meta_gga=True)
    neither = ArchitectureConfig.from_spec("neither", 3, 16)
    assert ArchitectureConfig.is_meta_gga(both) is True
    assert ArchitectureConfig.is_meta_gga(neither) is False


def test_the_predicate_answers_for_an_architecture_like_object():
    """The predicate is a static method rather than a property because its
    callers include code that receives arch-LIKE objects -- the energy-weight
    sweep's own consistency check builds them, and so do test doubles. It has
    to answer for those instead of raising AttributeError."""
    import types
    like = types.SimpleNamespace(
        descriptors=(types.SimpleNamespace(name="metagga"),))
    assert ArchitectureConfig.is_meta_gga(like) is True
    assert ArchitectureConfig.is_meta_gga(types.SimpleNamespace()) is False


@pytest.mark.parametrize("name", sorted(__import__(
    "xcquinox.alec.config", fromlist=["ARCHITECTURES"]).ARCHITECTURES))
def test_every_registered_architecture_states_its_rung_once(name):
    """Every architecture in the registry carries the flag and the descriptor
    together or carries neither, so the predicate and the flag answer the same
    for all of them and no registry entry was in the state the refusal now
    closes."""
    from xcquinox.alec.config import get_architecture
    arch = get_architecture(name)
    assert ArchitectureConfig.is_meta_gga(arch) is bool(arch.meta_gga), name


def test_reader_1_the_pretraining_parent_density_reads_the_predicate():
    """``pretrain_data_gen.resolve_parent_density`` under ``"auto"``."""
    mgga, gga = _mgga_and_gga()
    assert pdg.resolve_parent_density(mgga, "auto") == "scan"
    assert pdg.resolve_parent_density(gga, "auto") == "pbe"


def test_reader_2_the_scf_seed_functional_reads_the_predicate():
    """``rungs.arch_ingredients``, which ``seed_xc_for_arch`` drives. The seed
    the SCF starts from and the density the pretraining fits must be the same
    rung baseline, which is why they share the predicate rather than each
    carrying a rule."""
    from xcquinox.alec.rungs import arch_ingredients, seed_xc_for_arch
    assert arch_ingredients("deep_mgga_3x16")[0] is True
    assert arch_ingredients("deep_3x16")[0] is False
    assert seed_xc_for_arch("deep_mgga_3x16") == "scan"
    assert seed_xc_for_arch("deep_3x16") == "pbe"


def test_reader_3_the_datagen_stage_reads_the_predicate():
    """``cluster/_datagen._required_data_specs``: which pretrain-data files a
    sweep has to build. A mixed-rung sweep needs BOTH densities, and the file
    it writes has to be the file the pretrain worker opens."""
    import types
    from xcquinox.alec.cluster import _datagen

    def _cfg(archs):
        return types.SimpleNamespace(
            sweep=types.SimpleNamespace(arch=list(archs)),
            use_polarized_correlation=False,
            pretrain=types.SimpleNamespace(parent_density="auto"))

    assert _datagen._required_data_specs(_cfg(["deep_mgga_3x16"])) == \
        [(False, "scan")]
    assert _datagen._required_data_specs(_cfg(["deep_3x16"])) == \
        [(False, "pbe")]
    assert _datagen._required_data_specs(
        _cfg(["deep_3x16", "deep_mgga_3x16"])) == [(False, "pbe"),
                                                   (False, "scan")]


def test_reader_4_run_pretrain_selects_its_targets_from_the_predicate(
        tmp_path, monkeypatch):
    """``run_pretrain``'s enhancement-factor targets, its per-system
    parent-energy keys and its (s, alpha) mesh. This is the reader that used
    to ask the question differently: it took the FLAG while the parent density
    was resolved from the flag OR the descriptor, so an architecture carrying
    the descriptor alone was fitted to the PBE targets on the SCAN
    self-consistent density."""
    import xcquinox.alec.pretrain as ptmod
    d = tmp_path / "rung_targets"
    d.mkdir()
    # Fx_all is 0.1 and Fx_scan_all is 0.3 in this file, so the captured
    # target array says which parent's rows the fit was given.
    _write_system_npz(str(d), natoms=(2, 3))
    captured = []

    def _stub_train(model, _optimizer, _loss, _desc, ref_train, *_a, **_kw):
        captured.append(np.asarray(ref_train).reshape(-1))
        return model, [0.0], {"history": [], "steps_run": 1}

    monkeypatch.setattr(ptmod, "_train_pretrain_network", _stub_train)
    mgga = ArchitectureConfig.from_spec("t_rung_mgga", 2, 8,
                                        descriptors=["metagga"],
                                        meta_gga=True)
    gga = ArchitectureConfig.from_spec("t_rung_gga", 2, 8)
    for arch, want_fx, want_ref in ((mgga, 0.3, "scan"), (gga, 0.1, "pbe")):
        captured.clear()
        md = ptmod.run_pretrain(PretrainSpec(
            arch=arch, data_dir=str(d),
            checkpoint_dir=str(tmp_path / f"ck_{arch.name}"), n_steps=2,
            seed=0, loss_weighting="unweighted", parent_density="pbe"))
        np.testing.assert_allclose(captured[0], want_fx)
        assert md["meta_gga"] is (want_ref == "scan")
        # The per-system energy keys follow the same reading: the metadata's
        # measured term is built on the parent the rung names.
        assert pdg.resolve_parent_density(arch, "auto") == want_ref
