"""The pyscfad backend's UKS callback on per-channel feature blocks.

Each spin-scaled exchange term of the UKS callback is evaluated at the
descriptor block of its own doubled density diag(P_sigma, P_sigma) (Oliver and
Perdew, Phys. Rev. A 20, 397 (1979)); correlation keeps the total block. The
blocks live on pyscfad's own (pruned) grid, so the on-grid reassembly gains a
``spin_channel`` and the feature holder carries three full-grid arrays.
"""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

pyscfad = pytest.importorskip("pyscfad")

import xcquinox.alec as alec
import xcquinox.alec.solver_pyscfad as solver_pyscfad
from xcquinox.alec.config import MoleculeSpec
from xcquinox.alec.data import precompute_fixed_density_data
from xcquinox.alec.descriptors import (assemble_descriptor_features,
                                       doubled_spin_dm)
from xcquinox.alec.oneshot import split_exc_energy_uks
from xcquinox.alec.solver import (
    FeaturePolicy, SolverBackend, SolverConfig, SolverMode, run_scf)
from xcquinox.alec.solver_pyscfad import (
    _build_pyscfad_mf, _make_alec_eval_xc, _maybe_metagga_ao_grad,
    _maybe_rung35_proj_ao, _maybe_rung35ms_proj_ao,
    _reassemble_features_on_grid)


def _model(arch_name, seed=0, polarized=True):
    arch = dataclasses.replace(alec.get_architecture(arch_name),
                               use_polarized_correlation=polarized,
                               zero_init_final_layer=False)
    xnet, cnet = alec.create_network_pair(arch, seed=seed)
    return alec.AlecGGAModel.from_arch(arch, xnet=xnet, cnet=cnet)


def _md(model, name, atom, spin, composition, basis="def2-svp", grid_level=1):
    keys = tuple(sorted({k for d in model.descriptors
                         for k in d.required_mol_keys}))
    return precompute_fixed_density_data(
        MoleculeSpec(name=name, atom=atom, basis=basis, charge=0, spin=spin,
                     atom_composition=composition, grid_level=grid_level),
        required_keys=keys, descriptors=model.descriptors)


def _mol(atom, spin, basis="def2-svp"):
    from pyscf import gto
    return gto.M(atom=atom, basis=basis, spin=spin, verbose=0)


def _uks_grid(mol, grid_level=1):
    """The parent's own grid for this molecule, built independently of the
    precompute so the on-grid reassembly is exercised on real coordinates."""
    from pyscf import dft
    mf = dft.UKS(mol)
    mf.xc = "pbe"
    mf.grids.level = grid_level
    mf.grids.build()
    return jnp.asarray(mf.grids.coords)


def _grid_blocks(model, md, mf, dm, mol):
    """The three full-grid blocks on pyscfad's grid, as the backend builds
    them: total, alpha channel, beta channel."""
    common = dict(
        descriptors=model.descriptors,
        s_matrix=jnp.asarray(md["s_matrix"]),
        grid_coords=jnp.asarray(mf.grids.coords),
        mol=mol,
        rung35_proj_ao=_maybe_rung35_proj_ao(model.descriptors, mol,
                                             mf.grids.coords),
        rung35ms_proj_ao=_maybe_rung35ms_proj_ao(model.descriptors, mol,
                                                 mf.grids.coords),
        metagga_ao=_maybe_metagga_ao_grad(model.descriptors, mol,
                                          mf.grids.coords),
    )
    return (_reassemble_features_on_grid(dm=dm, **common),
            _reassemble_features_on_grid(dm=dm, spin_channel=0, **common),
            _reassemble_features_on_grid(dm=dm, spin_channel=1, **common))


def _split_energy_on_grid(model, mf, mol, dm, blocks):
    """``split_exc_energy_uks`` on pyscfad's grid from the AOs at its
    coordinates; the reference the callback's integrated XC energy must
    reproduce."""
    ao = np.asarray(mol.eval_gto("GTOval_sph_deriv1",
                                 np.asarray(mf.grids.coords)))
    ao0, ag = jnp.asarray(ao[0]), jnp.asarray(ao[1:4])
    w = jnp.asarray(mf.grids.weights)

    def grid(D):
        rho = jnp.einsum("ij,gi,gj->g", D, ao0, ao0)
        nab = 2.0 * jnp.einsum("ij,dgi,gj->gd", D, ag, ao0)
        return rho, nab, jnp.sum(nab * nab, axis=1)

    P = jnp.asarray(dm)
    rho_a, nab_a, sig_aa = grid(P[0])
    rho_b, nab_b, sig_bb = grid(P[1])
    nab_t = nab_a + nab_b
    tot, a, b = blocks
    return float(split_exc_energy_uks(
        model, rho_a, rho_b, sig_aa, sig_bb, jnp.sum(nab_t * nab_t, axis=1),
        a, b, tot, w))


def _callback_exc(model, md, mf, mol, dm, holder):
    """The integrated XC energy pyscfad's numint obtains from the callback
    with the given holder, at density ``dm``."""
    cb = _make_alec_eval_xc(model, model.descriptors, md, FeaturePolicy.FROZEN,
                            feature_holder=holder)
    mf.define_xc_(cb, "GGA")
    holder["offset"] = 0
    return float(mf.get_veff(mol, np.asarray(dm)).exc)


@pytest.mark.parametrize("arch_name", ["deep_rung35_mgga_3x16",
                                       "deep_rung35ms_3x16", "deep_dm_3x16"])
def test_on_grid_reassembly_with_a_spin_channel_doubles_the_density_matrix(
        arch_name):
    model = _model(arch_name)
    md = _md(model, "Li", "Li 0 0 0", 1, (("Li", 1),))
    mol = _mol("Li 0 0 0", 1)
    P = jnp.asarray(md["dm_pbe"])
    kw = dict(descriptors=model.descriptors, s_matrix=jnp.asarray(md["s_matrix"]),
              grid_coords=_uks_grid(mol), mol=mol)
    channelled = _reassemble_features_on_grid(dm=P, spin_channel=0, **kw)
    explicit = _reassemble_features_on_grid(dm=doubled_spin_dm(P, 0), **kw)
    np.testing.assert_allclose(np.asarray(channelled), np.asarray(explicit),
                               rtol=0, atol=0)
    total = _reassemble_features_on_grid(dm=P, **kw)
    assert float(np.max(np.abs(np.asarray(channelled) - np.asarray(total)))) > 1e-6
    beta = _reassemble_features_on_grid(dm=P, spin_channel=1, **kw)
    np.testing.assert_allclose(
        np.asarray(beta),
        np.asarray(_reassemble_features_on_grid(dm=doubled_spin_dm(P, 1), **kw)),
        rtol=0, atol=0)
    assert float(np.max(np.abs(np.asarray(beta) - np.asarray(channelled)))) > 1e-6


def test_on_grid_reassembly_collapses_at_a_closed_shell_density():
    model = _model("deep_rung35_mgga_3x16")
    from pyscf import dft
    mol = _mol("O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469", 0)
    mf = dft.RKS(mol)
    mf.xc = "pbe"
    mf.grids.level = 1
    mf.kernel()
    half = 0.5 * jnp.asarray(mf.make_rdm1())
    P = jnp.stack([half, half], axis=0)
    kw = dict(descriptors=model.descriptors,
              s_matrix=jnp.asarray(mol.intor("int1e_ovlp")),
              grid_coords=jnp.asarray(mf.grids.coords), mol=mol)
    a = np.asarray(_reassemble_features_on_grid(dm=P, spin_channel=0, **kw))
    b = np.asarray(_reassemble_features_on_grid(dm=P, spin_channel=1, **kw))
    t = np.asarray(_reassemble_features_on_grid(dm=P, **kw))
    np.testing.assert_allclose(a, t, rtol=0, atol=0)
    np.testing.assert_allclose(b, t, rtol=0, atol=0)


@pytest.mark.parametrize("arch_name", ["deep_mgga_3x16", "deep_rung35_3x16"])
def test_pyscfad_uks_callback_integrates_the_three_block_energy(arch_name):
    """pyscfad's numint must integrate the SAME functional the manual path
    evaluates: with the three holder blocks of pyscfad's grid, the callback's
    XC energy equals ``split_exc_energy_uks`` on that grid with those blocks;
    a holder with the channel blocks swapped, or with the total block in every
    slot (the superseded two-block evaluation), integrates a different number.

    The polarized callback forms zeta with its own clip (``[-1, 1]``, floor
    1e-300) where ``split_exc_energy_uks`` uses ``uks_zeta`` (``+-(1 - 1e-6)``,
    floor 1e-12); the two differ only where ``|zeta| > 1 - 1e-6``, which on Li
    is the density tail: measured 9e-15 Ha between the two conventions here,
    the same as with an unpolarized correlation network.
    """
    model = _model(arch_name)
    md = _md(model, "Li", "Li 0 0 0", 1, (("Li", 1),))
    mol = md["_pyscfad_mol"]
    mf = _build_pyscfad_mf(mol, md)
    dm = jnp.asarray(md["dm_pbe"])
    tot, a, b = _grid_blocks(model, md, mf, dm, mol)
    assert float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) > 1e-6

    e_three = _split_energy_on_grid(model, mf, mol, dm, (tot, a, b))
    e_two = _split_energy_on_grid(model, mf, mol, dm, (tot, tot, tot))
    e_swap = _split_energy_on_grid(model, mf, mol, dm, (tot, b, a))
    holder = {"features_full": tot, "features_full_a": a,
              "features_full_b": b, "offset": 0}
    exc = _callback_exc(model, md, mf, mol, dm, holder)
    holder_swap = {"features_full": tot, "features_full_a": b,
                   "features_full_b": a, "offset": 0}
    exc_swap = _callback_exc(model, md, mf, mol, dm, holder_swap)
    # Same grid, same blocks, same scalar energy densities; the callback
    # divides by rho and numint multiplies it back, zeroing rho <= 1e-12
    # points whose contribution is below 1e-16 Ha. Measured residuals on Li:
    # 2.0e-14 (deep_mgga_3x16) and 7.3e-15 (deep_rung35_3x16), with the
    # swapped holders at 9e-15 / 1.3e-14; the bound is 500x the worst. The
    # two-block value sits 1.2e-4 / 1.8e-4 Ha away and the swapped one
    # 5.6e-5 / 2.3e-5 Ha away, so the separation bound has a 23x margin.
    assert abs(exc - e_three) < 1e-11, (exc, e_three, exc - e_three)
    assert abs(exc_swap - e_swap) < 1e-11, (exc_swap, e_swap)
    assert abs(exc - e_two) > 1e-6, (exc, e_two)
    assert abs(exc - exc_swap) > 1e-6, (exc, exc_swap)


@pytest.mark.parametrize("arch_name", ["deep_rung35_3x16", "deep_mgga_3x16"])
def test_pyscfad_uks_scf_matches_the_manual_backend_energy(arch_name):
    """Both backends must converge the same functional on an open shell: the
    same three blocks reach the same exchange and correlation terms.

    The pyscfad backend refuses a density-matrix-dependent descriptor under
    REASSEMBLE (its per-point callback cannot carry the feature-response
    term), so the shared configuration is FIXED_J with FROZEN blocks: the
    manual loop takes the stored per-channel precompute blocks, the pyscfad
    loop the per-channel blocks of the same seed density on its own grid.
    The energies are compared at convergence (the loops mix differently, so
    a one-cycle comparison would measure the mixers, not the functional).
    """
    model = _model(arch_name)
    md = _md(model, "Li", "Li 0 0 0", 1, (("Li", 1),))
    common = dict(mode=SolverMode.FIXED_J, max_cycles=80, conv_tol=1e-10,
                  feature_policy=FeaturePolicy.FROZEN)
    r_m = run_scf(SolverConfig(backend=SolverBackend.MANUAL, **common),
                  model, md, forward_only=True)
    r_p = run_scf(SolverConfig(backend=SolverBackend.PYSCFAD, **common),
                  model, md)
    assert bool(r_m.converged) and bool(r_p.converged)
    e_manual, e_pyscfad = float(r_m.total_energy), float(r_p.total_energy)
    # The two loops take their frozen blocks from two implementations (the
    # stored precompute blocks against the on-grid reassembly, which differ by
    # the 1e-9 to 5e-8 rounding of the iso-orbital indicator) and stop on
    # different convergence criteria, so the bound is not an identity; measured
    # 3.7e-7 Ha (deep_rung35_3x16) and 3.9e-7 Ha (deep_mgga_3x16), against the
    # 1.8e-4 / 1.2e-4 Ha a two-block callback would read. The bound is far
    # below the 1.0 mHa atomic tolerance of the fidelity certificate.
    assert abs(e_manual - e_pyscfad) < 1e-4, (e_manual, e_pyscfad)
    d_m, d_p = np.asarray(r_m.density_matrix), np.asarray(r_p.density_matrix)
    assert d_m.shape == d_p.shape == (2,) + d_m.shape[1:]
    # Measured 4.1e-6 / 4.3e-6 on the same runs.
    assert float(np.max(np.abs(d_m - d_p))) < 1e-3, float(np.max(np.abs(d_m - d_p)))


def test_pyscfad_reassemble_refreshes_all_three_blocks_from_the_live_dm(
        monkeypatch):
    """Under REASSEMBLE the get_veff wrapper must rebuild the total AND the two
    per-channel blocks from the density matrix pyscfad hands it, every cycle.
    The production refusal of DM-dependent descriptors on this backend is
    bypassed here to reach the wiring; only the calls are inspected."""
    model = _model("deep_mgga_3x16")
    md = _md(model, "Li", "Li 0 0 0", 1, (("Li", 1),))
    monkeypatch.setattr(solver_pyscfad, "_reject_dm_dependent_descriptors",
                        lambda model, policy: None)
    calls = []
    original = solver_pyscfad._reassemble_features_on_grid

    def recorder(*args, **kwargs):
        dm = kwargs.get("dm", args[1] if len(args) > 1 else None)
        calls.append((kwargs.get("spin_channel"), np.array(np.asarray(dm))))
        return original(*args, **kwargs)

    monkeypatch.setattr(solver_pyscfad, "_reassemble_features_on_grid",
                        recorder)
    cfg = SolverConfig(backend=SolverBackend.PYSCFAD, mode=SolverMode.FIXED_J,
                       feature_policy=FeaturePolicy.REASSEMBLE, max_cycles=2,
                       conv_tol=1e-14)
    run_scf(cfg, model, md)
    channels = [c for c, _ in calls]
    # Initial holder (3 calls) plus three per get_veff call.
    assert len(calls) >= 6 and len(calls) % 3 == 0, channels
    for k in range(0, len(calls), 3):
        assert channels[k:k + 3] == [None, 0, 1], channels
        np.testing.assert_array_equal(calls[k][1], calls[k + 1][1])
        np.testing.assert_array_equal(calls[k][1], calls[k + 2][1])
    dm_seed = np.asarray(md["dm_pbe"])
    np.testing.assert_array_equal(calls[0][1], dm_seed)
    # At least one refresh happened at a density other than the seed.
    assert any(not np.array_equal(calls[k][1], dm_seed)
               for k in range(3, len(calls), 3)), "no refresh at a moved DM"


def _uks_rho_block(md, n_points=8):
    """A ``(2, 4, n)`` spin-resolved rho block in pyscfad's numint layout,
    taken from the precompute's own density at its densest grid points, so the
    callback is exercised on real numbers rather than on a synthetic array."""
    ao_deriv = jnp.asarray(md["ao_grid_deriv"])
    ao, ao_grad = ao_deriv[0], ao_deriv[1:4]
    P = jnp.asarray(md["dm_pbe"])
    channels = []
    for spin in (0, 1):
        rho = jnp.einsum("ij,gi,gj->g", P[spin], ao, ao)
        nabla = 2.0 * jnp.einsum("ij,dgi,gj->gd", P[spin], ao_grad, ao)
        channels.append(jnp.concatenate([rho[None, :], nabla.T], axis=0))
    full = np.asarray(jnp.stack(channels, axis=0))
    densest = np.argsort(full[0, 0] + full[1, 0])[::-1][:n_points]
    return full[:, :, np.sort(densest)]


def test_pyscfad_uks_callback_refuses_a_holder_without_per_channel_blocks():
    """The UKS branch of the callback needs the descriptor blocks of
    diag(P_a, P_a) and diag(P_b, P_b) for its two spin-scaled exchange terms.
    Where they are absent it must REFUSE: the reachable alternative -- the
    total block in both channels -- is the superseded two-block evaluation,
    which is silent, wrong on every open shell by tens of mHa, and
    indistinguishable from the correct path in the returned arrays.

    Two entries reach the missing-block state: a holder carrying only
    ``features_full`` (the shape a caller written against the previous
    contract would build) and the holder-less legacy path, which returns the
    precompute block as the total block and has no per-channel block to give.
    Both are checked, together with the positive control that the same
    callback evaluates when all three blocks are present -- without it the
    refusal test would pass on any error at all.
    """
    model = _model("deep_mgga_3x16")
    md = _md(model, "Li", "Li 0 0 0", 1, (("Li", 1),))
    total = assemble_descriptor_features(model.descriptors, md)
    rho = _uks_rho_block(md)
    n_points = rho.shape[-1]

    def callback(holder):
        return _make_alec_eval_xc(model, model.descriptors, md,
                                  FeaturePolicy.FROZEN, feature_holder=holder)

    def full_holder():
        return {"features_full": total, "features_full_a": total,
                "features_full_b": total, "offset": 0}

    exc, vxc, _fxc, _kxc = callback(full_holder())("", rho, spin=1, deriv=1)
    assert np.asarray(exc).shape == (n_points,)
    assert bool(np.all(np.isfinite(np.asarray(exc))))
    assert np.asarray(vxc[0]).shape == (n_points, 2)

    for missing in ("features_full_a", "features_full_b"):
        holder = full_holder()
        holder.pop(missing)
        with pytest.raises(ValueError,
                           match="without per-channel feature blocks"):
            callback(holder)("", rho, spin=1, deriv=1)
        holder = full_holder()
        holder[missing] = None
        with pytest.raises(ValueError,
                           match="without per-channel feature blocks"):
            callback(holder)("", rho, spin=1, deriv=1)

    whole_grid = _uks_rho_block(md, n_points=int(np.asarray(total).shape[0]))
    with pytest.raises(ValueError, match="without per-channel feature blocks"):
        callback(None)("", whole_grid, spin=1, deriv=1)
