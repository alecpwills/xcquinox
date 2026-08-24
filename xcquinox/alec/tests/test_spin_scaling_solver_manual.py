"""The manual UKS SCF on per-channel feature blocks.

The exchange term of the UKS energy is evaluated on the symmetric doubled
density diag(P_sigma, P_sigma) for each channel (Oliver and Perdew, Phys. Rev. A
20, 397 (1979)); the correlation term stays on the total density. These tests
pin that the SCF energy is that functional, that the blocks the loop consumes
are the per-channel blocks of the CURRENT density matrix (the stored precompute
blocks only under the FROZEN policy), and that the Fock matrices the loop builds
are the exact derivative of that energy.
"""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

import xcquinox.alec as alec
import xcquinox.alec.solver_manual as solver_manual
from xcquinox.alec.config import MoleculeSpec
from xcquinox.alec.data import precompute_fixed_density_data
from xcquinox.alec.descriptors import assemble_descriptor_features
from xcquinox.alec.solver import (
    FeaturePolicy, SolverBackend, SolverConfig, SolverMode, run_scf,
    _contract_dm_to_grid_with_nabla, make_uks_feature_fns)
from xcquinox.alec.solver_manual import _compute_total_energy_uks

_FD_EPS = 1e-6
# Relative residual of the Fock/energy finite-difference check. Measured on the
# O atom (def2-svp, grid level 1) at the 1e-6 step with the solver's own Fock
# pair: 4.1e-11 (deep_rung35_3x16), 4.6e-10 (deep_mgga_3x16), 3.9e-10
# (deep_dm_3x16). The superseded two-block potential (total block in both
# exchange channels, one accumulated feature-response contraction) against the
# same three-block energy reads 1.4e-4, 3.6e-5 and 7.1e-5 respectively. The
# bound sits three orders above the measured floor and 36x below the smallest
# defect signal.
_FD_TOL = 1e-6


def _model(arch_name, seed=0):
    arch = dataclasses.replace(alec.get_architecture(arch_name),
                               use_polarized_correlation=True,
                               zero_init_final_layer=False)
    xnet, cnet = alec.create_network_pair(arch, seed=seed)
    return alec.AlecGGAModel.from_arch(arch, xnet=xnet, cnet=cnet)


def _md(model, name, atom, spin, composition, basis="def2-svp", grid_level=1):
    keys = tuple(sorted({k for d in model.descriptors
                         for k in d.required_mol_keys} | {"eri"}))
    return precompute_fixed_density_data(
        MoleculeSpec(name=name, atom=atom, basis=basis, charge=0, spin=spin,
                     atom_composition=composition, grid_level=grid_level),
        required_keys=keys, descriptors=model.descriptors)


def _config(policy, mode=SolverMode.FULL, max_cycles=2, conv_tol=1e-6):
    return SolverConfig(mode=mode, backend=SolverBackend.MANUAL,
                        max_cycles=max_cycles, conv_tol=conv_tol,
                        feature_policy=policy)


def _live_fns(model, md):
    """The three ``P_ab -> block`` closures the solver consumes."""
    return make_uks_feature_fns(
        descriptors=model.descriptors,
        ao_deriv=jnp.asarray(md["ao_grid_deriv"]),
        s_matrix=jnp.asarray(md["s_matrix"]),
        n_grid=int(np.asarray(md["grid_weights"]).shape[0]),
        cusp_features=md.get("cusp_features"),
        rung35_proj_ao=md.get("rung35_proj_ao"),
        rung35ms_proj_ao=md.get("rung35ms_proj_ao"))


def _precompute_blocks(model, md):
    return (assemble_descriptor_features(model.descriptors, md, spin_channel=0),
            assemble_descriptor_features(model.descriptors, md, spin_channel=1),
            assemble_descriptor_features(model.descriptors, md))


def _record_energy_calls(monkeypatch):
    """Record ``(D_a, D_b, features_a, features_b, features_tot, rho_a, rho_b,
    sigma_aa, sigma_bb, sigma_tot)`` of every ``_compute_total_energy_uks``
    call the UKS loop makes. The loop must be driven with
    ``forward_only=True`` so the arguments are concrete."""
    calls = []
    original = solver_manual._compute_total_energy_uks

    def recorder(model, D_a, D_b, rho_a, rho_b, sigma_aa, sigma_bb, sigma_tot,
                 features_a, features_b, features_tot, grid_weights, h_core,
                 J_total, e_nuc):
        calls.append(tuple(np.array(np.asarray(x)) for x in
                           (D_a, D_b, features_a, features_b, features_tot,
                            rho_a, rho_b, sigma_aa, sigma_bb, sigma_tot)))
        return original(model, D_a, D_b, rho_a, rho_b, sigma_aa, sigma_bb,
                        sigma_tot, features_a, features_b, features_tot,
                        grid_weights, h_core, J_total, e_nuc)

    monkeypatch.setattr(solver_manual, "_compute_total_energy_uks", recorder)
    return calls


def _record_fock_matrices(monkeypatch):
    """Record every Fock matrix the UKS loop hands to its eigensolver, in call
    order (alpha, beta, alpha, beta, ...)."""
    focks = []
    original = solver_manual._diagonalize_roothaan_unrestricted

    def recorder(F, S, nocc):
        focks.append(np.array(np.asarray(F)))
        return original(F, S, nocc)

    monkeypatch.setattr(solver_manual, "_diagonalize_roothaan_unrestricted",
                        recorder)
    return focks


def _spin_grid(D, ao, ao_xyz):
    """``(rho, nabla_rho, sigma)`` of one spin block, reduced exactly as
    ``solver._contract_dm_to_grid_with_nabla`` reduces it. The gradient
    invariant must come off the kernel's ``einsum("gd,gd->g")``: the finite
    differences below are references for the loop's own functional, and a
    ``jnp.sum(nabla * nabla, axis=1)`` reduction is a different array
    (2-3e-16 relative on a fifth of the grid). See
    ``test_the_finite_difference_reference_reduces_sigma_the_kernel_way``."""
    rho = jnp.einsum("ij,gi,gj->g", D, ao, ao)
    nabla = 2.0 * jnp.einsum("ij,dgi,gj->gd", D, ao_xyz, ao)
    return rho, nabla, jnp.einsum("gd,gd->g", nabla, nabla)


@pytest.mark.parametrize("arch_name", ["deep_rung35_mgga_3x16",
                                       "deep_rung35ms_3x16", "deep_dm_3x16"])
def test_manual_uks_energy_is_the_three_block_split_energy(arch_name):
    """The SCF's own energy helper must equal oneshot.split_exc_energy_uks fed
    the three live blocks, term for term."""
    from xcquinox.alec.oneshot import split_exc_energy_uks
    model = _model(arch_name)
    md = _md(model, "Li", "Li 0 0 0", 1, (("Li", 1),))
    ao = jnp.asarray(md["ao_grid"])
    ao_xyz = jnp.asarray(md["ao_grid_deriv"])[1:4]
    w = jnp.asarray(md["grid_weights"])
    P = jnp.asarray(md["dm_pbe"])
    fa, fb, ft = _live_fns(model, md)

    rho_a, nab_a, sig_aa = _spin_grid(P[0], ao, ao_xyz)
    rho_b, nab_b, sig_bb = _spin_grid(P[1], ao, ao_xyz)
    nab_t = nab_a + nab_b
    sig_tot = jnp.einsum("gd,gd->g", nab_t, nab_t)
    h = jnp.asarray(md["h_core"])
    j_tot = jnp.asarray(md["j_matrix"])[0] + jnp.asarray(md["j_matrix"])[1]
    e_nuc = jnp.asarray(md["e_nuc"])
    got = float(_compute_total_energy_uks(
        model, P[0], P[1], rho_a, rho_b, sig_aa, sig_bb, sig_tot,
        fa(P), fb(P), ft(P), w, h, j_tot, e_nuc))
    xc = float(split_exc_energy_uks(model, rho_a, rho_b, sig_aa, sig_bb,
                                    sig_tot, fa(P), fb(P), ft(P), w))
    one = float(jnp.einsum("ij,ij->", h, P[0] + P[1]))
    coul = float(0.5 * jnp.einsum("ij,ij->", j_tot, P[0] + P[1]))
    # Same four terms summed in the same order; measured difference 0.0 on
    # all three architectures (1.4e-14 would be one ulp of a 7 Ha energy).
    assert abs(got - (float(e_nuc) + one + coul + xc)) < 1e-12


@pytest.mark.parametrize("arch_name", ["deep_rung35_mgga_3x16",
                                       "deep_rung35ms_3x16", "deep_dm_3x16"])
def test_manual_uks_scf_runs_and_stays_finite_under_reassemble(arch_name):
    model = _model(arch_name)
    md = _md(model, "Li", "Li 0 0 0", 1, (("Li", 1),))
    result = run_scf(_config(FeaturePolicy.REASSEMBLE), model, md)
    assert bool(jnp.isfinite(result.total_energy))
    assert bool(jnp.all(jnp.isfinite(result.density_matrix)))
    n_grid = int(np.asarray(md["grid_weights"]).shape[0])
    n_feat = sum(d.n_features for d in model.descriptors)
    assert result.features_used.shape == (n_grid, n_feat)


@pytest.mark.parametrize("arch_name", ["deep_rung35_mgga_3x16", "deep_dm_3x16"])
def test_manual_uks_frozen_policy_uses_the_precomputed_channel_blocks(
        arch_name, monkeypatch):
    """FROZEN must freeze the three blocks separately, not freeze one block and
    reuse it for all three. FROZEN is only coherent with FIXED_J (FULL refuses
    it at construction), so the pinned loop is the FIXED_J one."""
    model = _model(arch_name)
    md = _md(model, "Li", "Li 0 0 0", 1, (("Li", 1),))
    block_a, block_b, block_tot = [np.asarray(b) for b in
                                   _precompute_blocks(model, md)]
    assert float(np.max(np.abs(block_a - block_tot))) > 1e-6, (
        "Li must have distinguishable per-channel and total blocks, otherwise "
        "this test cannot see the difference it is checking")
    calls = _record_energy_calls(monkeypatch)
    result = run_scf(_config(FeaturePolicy.FROZEN, mode=SolverMode.FIXED_J,
                             max_cycles=2, conv_tol=1e-14),
                     model, md, forward_only=True)
    assert bool(jnp.isfinite(result.total_energy))
    # One call for the initial energy, one per cycle.
    assert len(calls) == 3
    for k, call in enumerate(calls):
        _da, _db, fa, fb, ft = call[:5]
        np.testing.assert_array_equal(fa, block_a, err_msg=f"call {k}: alpha")
        np.testing.assert_array_equal(fb, block_b, err_msg=f"call {k}: beta")
        np.testing.assert_array_equal(ft, block_tot, err_msg=f"call {k}: total")
    # The density matrix moved, so a REASSEMBLE loop would have changed the
    # blocks: the equality above is not vacuous.
    assert float(np.max(np.abs(calls[-1][0] - np.asarray(md["dm_pbe"])[0]))) \
        > 1e-6
    np.testing.assert_array_equal(np.asarray(result.features_used), block_tot)


@pytest.mark.parametrize("arch_name", ["deep_rung35_mgga_3x16",
                                       "deep_rung35ms_3x16", "deep_dm_3x16"])
def test_manual_uks_reassemble_feeds_the_live_blocks_of_the_current_density(
        arch_name, monkeypatch):
    """Under REASSEMBLE every energy evaluation must carry the three blocks of
    the density matrix it is evaluated at -- the per-channel blocks of
    diag(P_sigma, P_sigma) and the total block -- and never the stored
    precompute blocks once the density has moved."""
    model = _model(arch_name)
    md = _md(model, "Li", "Li 0 0 0", 1, (("Li", 1),))
    fa_of, fb_of, ft_of = _live_fns(model, md)
    block_a, block_b, block_tot = [np.asarray(b) for b in
                                   _precompute_blocks(model, md)]
    calls = _record_energy_calls(monkeypatch)
    result = run_scf(_config(FeaturePolicy.REASSEMBLE, max_cycles=2,
                             conv_tol=1e-14), model, md, forward_only=True)
    assert bool(jnp.isfinite(result.total_energy))
    assert len(calls) == 3
    for k, call in enumerate(calls):
        da, db, fa, fb, ft = call[:5]
        P = jnp.stack([jnp.asarray(da), jnp.asarray(db)], axis=0)
        np.testing.assert_array_equal(fa, np.asarray(fa_of(P)),
                                      err_msg=f"call {k}: alpha block")
        np.testing.assert_array_equal(fb, np.asarray(fb_of(P)),
                                      err_msg=f"call {k}: beta block")
        np.testing.assert_array_equal(ft, np.asarray(ft_of(P)),
                                      err_msg=f"call {k}: total block")
        assert float(np.max(np.abs(fa - fb))) > 1e-6, (
            f"call {k}: Li's alpha and beta blocks must differ")
    # Call 0 is the seed density, evaluated with the LIVE closures there as
    # well. Live and stored blocks agree to rounding on every column that is
    # linear in the density matrix; the iso-orbital indicator differs by up to
    # 1.4e-9 in the alpha/total blocks (the einsum-vs-PySCF tail floor of the
    # closures) and by up to 4.7e-8 in the beta block, whose one electron makes
    # tau - tau_W vanish identically so both columns are rounding residues.
    # The later calls are at moved densities whose blocks must differ from the
    # stored ones by far more than those residues.
    dm_seed = np.asarray(md["dm_seed"])
    np.testing.assert_array_equal(calls[0][0], dm_seed[0])
    np.testing.assert_array_equal(calls[0][1], dm_seed[1])
    np.testing.assert_allclose(calls[0][2], block_a, rtol=0, atol=1e-6)
    np.testing.assert_allclose(calls[0][3], block_b, rtol=0, atol=1e-6)
    np.testing.assert_allclose(calls[0][4], block_tot, rtol=0, atol=1e-6)
    _da, _db, fa2, fb2, ft2 = calls[-1][:5]
    assert float(np.max(np.abs(fa2 - block_a))) > 1e-6
    assert float(np.max(np.abs(fb2 - block_b))) > 1e-6
    assert float(np.max(np.abs(ft2 - block_tot))) > 1e-6
    # SCFResult.features_used is the PHYSICAL block of the final density.
    P_final = jnp.asarray(result.density_matrix)
    np.testing.assert_array_equal(np.asarray(result.features_used),
                                  np.asarray(ft_of(P_final)))


@pytest.mark.parametrize("arch_name", ["deep_rung35_mgga_3x16", "deep_3x16"])
def test_manual_uks_energy_ingredients_are_the_kernel_contractions(
        arch_name, monkeypatch):
    """The per-channel rho_s and sigma_ss the loop feeds to the energy (and,
    doubled, to the exchange terms) must be the kernel contractions the live
    closures use for the same channel -- the same arrays, not the same numbers
    up to rounding. Under JAX_PLATFORMS=cpu (the production evaluation
    configuration, and pytest's) a sigma reduced with
    ``jnp.sum(nabla * nabla, axis=1)`` differs from the kernel's
    ``einsum("gd,gd->g")`` by 2.2-2.8e-16 relative on 13-23% of the grid
    points of O and Li, which the meta-GGA indicator amplifies to 5.3e-11 on
    O's beta channel (resolved region) and 4.5e-10 in Li's beta tail; with
    the kernel's reduction the two agree bitwise.
    """
    model = _model(arch_name)
    md = _md(model, "O", "O 0 0 0", 2, (("O", 1),))
    ao_deriv = jnp.asarray(md["ao_grid_deriv"])
    calls = _record_energy_calls(monkeypatch)
    run_scf(_config(FeaturePolicy.REASSEMBLE, max_cycles=1, conv_tol=1e-14),
            model, md, forward_only=True)
    assert len(calls) == 2
    for k, call in enumerate(calls):
        da, db = call[0], call[1]
        rho_a, rho_b, sig_aa, sig_bb, sig_tot = call[5:]
        nablas = []
        for lab, D, rho, sig in (("alpha", da, rho_a, sig_aa),
                                 ("beta", db, rho_b, sig_bb)):
            rho_k, nabla_k, sig_k = _contract_dm_to_grid_with_nabla(
                jnp.asarray(D), ao_deriv)
            np.testing.assert_array_equal(rho, np.asarray(rho_k),
                                          err_msg=f"call {k}: {lab} rho")
            np.testing.assert_array_equal(sig, np.asarray(sig_k),
                                          err_msg=f"call {k}: {lab} sigma")
            nablas.append(nabla_k)
        # The total-density gradient invariant is |nabla_a + nabla_b|^2, the
        # quantity whose derivative the shared v_sigma term contracts with,
        # reduced the same way.
        nab_t = nablas[0] + nablas[1]
        np.testing.assert_array_equal(
            sig_tot, np.asarray(jnp.einsum("gd,gd->g", nab_t, nab_t)),
            err_msg=f"call {k}: sigma_tot")


def test_manual_uks_closed_shell_density_gives_three_identical_blocks():
    """A UKS run at rho_a = rho_b must reduce to the RKS functional exactly."""
    model = _model("deep_rung35_mgga_3x16")
    md = _md(model, "H2O", "O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469",
             0, (("O", 1), ("H", 2)))
    fa, fb, ft = _live_fns(model, md)
    half = 0.5 * jnp.asarray(md["dm_pbe"])
    P = jnp.stack([half, half], axis=0)
    np.testing.assert_allclose(np.asarray(fa(P)), np.asarray(ft(P)),
                               rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(fb(P)), np.asarray(ft(P)),
                               rtol=0, atol=0)


@pytest.mark.parametrize("name,atom,composition", [
    ("Li", "Li 0 0 0", (("Li", 1),)), ("H", "H 0 0 0", (("H", 1),))])
def test_manual_uks_one_electron_block_fock_is_rounding_stable(
        name, atom, composition, monkeypatch):
    """A one-electron density is a single orbital, for which tau = tau_W
    pointwise, so the iso-orbital indicator of the block built on it (the
    doubled density of a one-electron CHANNEL -- Li beta, H alpha -- or the
    TOTAL density of the H atom) is a 0/0 sitting on the lower clip of
    ``metagga.compute_alpha`` wherever that block is idempotent, which the
    seed density and the fixed point both are. Autodiff there returns the
    rounding-selected side: without the gate, the beta-channel
    feature-response term of Li/deep_mgga_3x16 is 1.13 Ha and MOVES BY
    0.93 Ha under a 1e-14 relative change of the density matrix (H: 3.4e-3
    alpha channel, 7.6e-4 total block; every multi-electron block is stable
    to 4e-16). Free H and Li are atomization anchors; the Fock the loop
    diagonalizes there must be a function of the density, not of the
    rounding. With the indicator columns of a one-electron block's de/df
    zeroed, the measured movement is (6.1e-16, 4.8e-11) on Li and
    (1.5e-15, 1.9e-15) on H: the 4.8e-11 residual is the VALUE of the beta
    block's indicator column -- a cancellation residue bounded by 4.7e-8 in
    the tail -- wobbling through the fixed-feature JVPs, not a derivative
    term. The bound is 100x the measured worst and 1.8e7 below the ungated
    signal.

    The seed probed here is idempotent, so what the gate drops IS the exact
    response. That is not true of the mixed densities the loop visits between
    cycles, where the block is rank 2 and the indicator is an ordinary O(1)
    column; the scope of the gate, and what it costs off that manifold, is in
    the docstring of ``solver_manual._drop_one_orbital_indicator_response``
    and pinned by ``test_manual_uks_gated_fock_at_the_li_fixed_point``."""
    model = _model("deep_mgga_3x16")
    md = _md(model, name, atom, 1, composition)
    P0 = np.asarray(md["dm_pbe"])
    rng = np.random.default_rng(7)
    N = rng.standard_normal(P0.shape)
    N = 0.5 * (N + np.swapaxes(N, -1, -2))
    P1 = P0 + 1e-14 * np.linalg.norm(P0) / np.linalg.norm(N) * N

    focks = _record_fock_matrices(monkeypatch)
    for seed in (P0, P1):
        md_run = dict(md)
        md_run["dm_seed"] = jnp.asarray(seed)
        run_scf(_config(FeaturePolicy.REASSEMBLE, max_cycles=1,
                        conv_tol=1e-14), model, md_run, forward_only=True)
    assert len(focks) == 4
    move_a = float(np.max(np.abs(focks[2] - focks[0])))
    move_b = float(np.max(np.abs(focks[3] - focks[1])))
    assert max(move_a, move_b) < 5e-9, (
        f"{name}: the Fock pair moved by ({move_a:.3e}, {move_b:.3e}) under a "
        f"1e-14 relative density change; a rounding-selected clip-kink "
        f"response is reaching the eigensolver")


@pytest.mark.parametrize("arch_name", ["deep_rung35_3x16", "deep_mgga_3x16",
                                       "deep_dm_3x16"])
def test_manual_uks_fock_is_the_derivative_of_the_three_block_energy(
        arch_name, monkeypatch):
    """The Fock matrices the loop diagonalizes must be dE/dP_sigma of the
    energy the loop reports, with the three live feature maps differentiated
    -- the solver's own matrices, captured on their way to the eigensolver,
    against a central difference of the solver's own energy helper.

    Probe: the O atom (5 alpha, 3 beta electrons). A one-electron channel
    (Li beta) is a single orbital whose doubled-block iso-orbital indicator
    sits on the lower clip of ``metagga.compute_alpha``, where a central
    difference is not a derivative; on O every block stays >= 6.6e-4 above the
    clip at the 1e-6 step (see test_solv01_split_xc).
    """
    model = _model(arch_name)
    md = _md(model, "O", "O 0 0 0", 2, (("O", 1),))
    ao = jnp.asarray(md["ao_grid"])
    ao_xyz = jnp.asarray(md["ao_grid_deriv"])[1:4]
    w = jnp.asarray(md["grid_weights"])
    h = jnp.asarray(md["h_core"])
    e_nuc = jnp.asarray(md["e_nuc"])
    eri = jnp.asarray(md["eri"])
    fa_of, fb_of, ft_of = _live_fns(model, md)
    P0 = jnp.asarray(md["dm_pbe"])

    focks = _record_fock_matrices(monkeypatch)
    run_scf(_config(FeaturePolicy.REASSEMBLE, max_cycles=1, conv_tol=1e-14),
            model, md, forward_only=True)
    assert len(focks) == 2
    F_a, F_b = focks

    def energy(P):
        rho_a, nab_a, sig_aa = _spin_grid(P[0], ao, ao_xyz)
        rho_b, nab_b, sig_bb = _spin_grid(P[1], ao, ao_xyz)
        nab_t = nab_a + nab_b
        J = solver_manual._compute_j_matrix(P[0] + P[1], eri)
        return _compute_total_energy_uks(
            model, P[0], P[1], rho_a, rho_b, sig_aa, sig_bb,
            jnp.einsum("gd,gd->g", nab_t, nab_t), fa_of(P), fb_of(P),
            ft_of(P),
            w, h, J, e_nuc)

    rng = np.random.default_rng(20260821)
    W = rng.standard_normal(P0.shape)
    W = jnp.asarray(0.5 * (W + np.swapaxes(W, -1, -2)))
    analytic = float(np.sum(F_a * np.asarray(W[0]))
                     + np.sum(F_b * np.asarray(W[1])))
    fd = float((energy(P0 + _FD_EPS * W) - energy(P0 - _FD_EPS * W))
               / (2.0 * _FD_EPS))
    rel = abs(fd - analytic) / max(abs(fd), abs(analytic), 1e-30)
    assert rel < _FD_TOL, (
        f"{arch_name}: the UKS Fock pair is not dE/dP of the three-block "
        f"energy (FD={fd:.10e} analytic={analytic:.10e} rel={rel:.3e})")


def test_the_finite_difference_reference_reduces_sigma_the_kernel_way():
    """``_spin_grid`` is the grid contraction the finite-difference reference
    energies in this file are built from, so it must be the loop's own
    contraction -- ``solver._contract_dm_to_grid_with_nabla`` -- and not a
    numerically different reduction of the same expression.

    Under JAX_PLATFORMS=cpu a ``jnp.sum(nabla * nabla, axis=1)`` reduction of
    the gradient invariant differs from the kernel's ``einsum("gd,gd->g")`` by
    2.2-2.8e-16 relative on 13-23 per cent of the grid points of O and Li,
    which the meta-GGA indicator amplifies by up to five orders (5.3e-11 in the
    resolved region, 4.5e-10 in the tail). That is the invariant
    ``test_manual_uks_energy_ingredients_are_the_kernel_contractions`` pins on
    the loop; a reference energy built the other way contradicts it.
    """
    model = _model("deep_mgga_3x16")
    md = _md(model, "O", "O 0 0 0", 2, (("O", 1),))
    ao_deriv = jnp.asarray(md["ao_grid_deriv"])
    ao = jnp.asarray(md["ao_grid"])
    ao_xyz = ao_deriv[1:4]
    P = jnp.asarray(md["dm_pbe"])
    for lab, D in (("alpha", P[0]), ("beta", P[1])):
        rho, nabla, sigma = _spin_grid(D, ao, ao_xyz)
        rho_k, nabla_k, sigma_k = _contract_dm_to_grid_with_nabla(D, ao_deriv)
        np.testing.assert_array_equal(np.asarray(rho), np.asarray(rho_k),
                                      err_msg=f"{lab} rho")
        np.testing.assert_array_equal(np.asarray(nabla), np.asarray(nabla_k),
                                      err_msg=f"{lab} nabla_rho")
        np.testing.assert_array_equal(np.asarray(sigma), np.asarray(sigma_k),
                                      err_msg=f"{lab} sigma")


def _energy_helper(model, md):
    """The loop's own energy helper as a function of the DM pair, with the
    three live blocks and ``J`` rebuilt from the ERI -- the function whose
    derivative the Fock pair must be."""
    ao = jnp.asarray(md["ao_grid"])
    ao_xyz = jnp.asarray(md["ao_grid_deriv"])[1:4]
    w = jnp.asarray(md["grid_weights"])
    h = jnp.asarray(md["h_core"])
    e_nuc = jnp.asarray(md["e_nuc"])
    eri = jnp.asarray(md["eri"])
    fa_of, fb_of, ft_of = _live_fns(model, md)

    def energy(P):
        P = jnp.asarray(P)
        rho_a, nab_a, sig_aa = _spin_grid(P[0], ao, ao_xyz)
        rho_b, nab_b, sig_bb = _spin_grid(P[1], ao, ao_xyz)
        nab_t = nab_a + nab_b
        return float(_compute_total_energy_uks(
            model, P[0], P[1], rho_a, rho_b, sig_aa, sig_bb,
            jnp.einsum("gd,gd->g", nab_t, nab_t),
            fa_of(P), fb_of(P), ft_of(P), w, h,
            solver_manual._compute_j_matrix(P[0] + P[1], eri), e_nuc))

    return energy


def _fock_pair_at(model, md, seed_dm, focks):
    """The Fock pair the loop hands to its eigensolver when seeded at
    ``seed_dm`` -- the solver's own matrices, captured on their way to the
    eigensolver, not a reconstruction. ``focks`` is the list returned by
    ``_record_fock_matrices``; several seeds may be probed in one test."""
    first = len(focks)
    md_run = dict(md)
    md_run["dm_seed"] = jnp.asarray(seed_dm)
    run_scf(_config(FeaturePolicy.REASSEMBLE, max_cycles=1, conv_tol=1e-14),
            model, md_run, forward_only=True)
    assert len(focks) == first + 2
    return focks[first], focks[first + 1]


def _one_orbital_rotation(s_matrix, P_s, seed=7):
    """``theta -> P_s(theta)``, a RANK-PRESERVING rotation of the single
    occupied orbital of a one-electron spin block.

    ``P_s(theta) = c(theta) c(theta)^T`` with ``c^T S c = 1`` at every theta,
    so the block is a single orbital -- rank 1, idempotent in the metric S --
    all along the path, and ``tau = tau_W`` there identically. Returns the map
    and the descending occupations of ``S^1/2 P_s S^1/2``, which the caller
    must check are (1, 0, ...): the path is the physical one-orbital manifold
    only if the block it starts from lies on it.
    """
    ev, evec = np.linalg.eigh(np.asarray(s_matrix))
    s_half = evec @ np.diag(np.sqrt(ev)) @ evec.T
    s_inv_half = evec @ np.diag(1.0 / np.sqrt(ev)) @ evec.T
    occ, U = np.linalg.eigh(s_half @ np.asarray(P_s) @ s_half)
    c_orth = U[:, int(np.argmax(occ))]
    step = np.random.default_rng(seed).standard_normal(c_orth.shape)
    step = step - (step @ c_orth) * c_orth

    def P_of(theta):
        c = c_orth + theta * step
        c = s_inv_half @ (c / np.linalg.norm(c))
        return np.outer(c, c)

    return P_of, np.sort(occ)[::-1]


@pytest.mark.parametrize("name,atom,composition,channel,tol", [
    ("H", "H 0 0 0", (("H", 1),), 0, 1e-7),
    ("Li", "Li 0 0 0", (("Li", 1),), 1, 1e-5)])
def test_manual_uks_gated_fock_is_the_derivative_along_a_one_orbital_rotation(
        name, atom, composition, channel, tol, monkeypatch):
    """On a ONE-ELECTRON spin channel -- where the iso-orbital-indicator gate
    of ``solver_manual`` fires -- the gated Fock pair must still be the
    derivative of the three-block energy along the manifold of single-orbital
    densities.

    The probe is a rank-preserving rotation of that channel's occupied orbital
    (H alpha, Li beta), taken at the PBE seed rather than at the fixed point,
    where every orbital rotation is stationary and the check would be vacuous.
    ``tau = tau_W`` holds identically along such a path, so the indicator is a
    rounding residue (8.3e-11 on H's channel block) and the response the gate
    drops is the clip's 0/0; the gradient of every other ingredient is
    untouched, so the gated Fock must reproduce dE/dtheta exactly. Measured
    (``deep_mgga_3x16``, def2-svp, grid level 1, 1e-5 step): dE/dtheta =
    -6.869811e-3 with a relative residual 7.8e-10 (H) and -6.807988e-3 with
    5.9e-10 absolute, 8.7e-8 relative (Li, whose beta indicator column carries
    a larger cancellation residue). The residual falls as h^2 from the 1e-3
    step (5.5e-6 / 2.5e-5), so the quantity being reproduced is the
    derivative and not a coincidence of one step. The bounds are 128x and
    115x the measured floors.

    What this check does NOT do, measured rather than assumed: with the gate
    disabled the same probe reads 7.5e-11 (H) and 8.6e-8 (Li), so it does not
    discriminate against removing the gate. At an exactly idempotent
    one-orbital block the response the gate drops is numerically orthogonal to
    a rank-preserving rotation, which is the same statement as the check
    passing. Gate removal is caught by
    ``test_manual_uks_one_electron_block_fock_is_rounding_stable`` (the
    ungated Fock moves by 0.93 Ha under a 1e-14 relative density change) and
    by ``test_manual_uks_gated_fock_at_the_li_fixed_point`` (the unconstrained
    direction reads 0.88 relative ungated against 5.5e-2 gated). What this
    check adds is the spec's energy/potential oracle on the species where the
    gate fires: the shipped O-atom check runs only where it does not.
    """
    model = _model("deep_mgga_3x16")
    md = _md(model, name, atom, 1, composition)
    P_pbe = np.asarray(md["dm_pbe"])
    rotate, occ = _one_orbital_rotation(md["s_matrix"], P_pbe[channel])
    assert abs(occ[0] - 1.0) < 1e-9 and abs(occ[1]) < 1e-8, (
        f"{name}: the probed channel is not a single orbital at the seed "
        f"(occupations {occ[:3]})")

    def P_of(theta):
        out = np.array(P_pbe)
        out[channel] = rotate(theta)
        return out

    # theta = 0 must BE the seed density: the reference is built from occupied
    # orbitals, so the one-electron channel is already a single orbital there.
    assert float(np.max(np.abs(P_of(0.0) - P_pbe))) < 1e-12

    focks = _record_fock_matrices(monkeypatch)
    F_a, F_b = _fock_pair_at(model, md, P_of(0.0), focks)
    energy = _energy_helper(model, md)
    step = 1e-5
    dP = (P_of(step) - P_of(-step)) / (2.0 * step)
    analytic = float(np.sum(F_a * dP[0]) + np.sum(F_b * dP[1]))
    fd = (energy(P_of(step)) - energy(P_of(-step))) / (2.0 * step)
    assert abs(analytic) > 1e-3, (
        f"{name}: the probe direction is stationary ({analytic:.3e}); a "
        f"vanishing analytic derivative cannot discriminate")
    rel = abs(fd - analytic) / max(abs(fd), abs(analytic), 1e-30)
    assert rel < tol, (
        f"{name}: the gated Fock pair is not dE/dtheta along the one-orbital "
        f"manifold (FD={fd:.10e} analytic={analytic:.10e} rel={rel:.3e})")


def test_manual_uks_gated_fock_at_the_li_fixed_point(monkeypatch):
    """Li at its own fixed point, with the gate live on the beta channel.

    Directions that keep the beta channel a SINGLE ORBITAL are the ones the
    gate's justification covers, and there the converged Fock pair is the
    derivative of the three-block energy: an alpha-channel perturbation (the
    beta block untouched, hence still rank 1) reproduces dE/dP to 5.0e-10
    relative on a derivative of 1.183, and a rank-preserving rotation of the
    beta orbital reproduces the stationarity of the fixed point (both sides
    6.6e-9, agreeing to 9.2e-10).

    An UNCONSTRAINED direction does not, and by design: it takes the beta
    block off the one-orbital manifold, where the indicator is a smooth
    O(1) column of the mixed density rather than a 0/0, so the response the
    gate drops is a real term of the energy's derivative. Measured 5.55e-2
    relative (h = 1e-5 and 1e-6 agree to three digits, so this is a genuine
    gap and not finite-difference noise). The fixed point itself is unchanged
    -- gate-live and gate-disabled loops converge Li to within 8e-15 Ha -- and
    the gap is the scope statement of
    ``solver_manual._drop_one_orbital_indicator_response``. Closing
    DEFERRED_WORK #27 (a smooth positive part in the ENERGY of
    ``metagga.compute_alpha``, which retires the gate) is expected to close
    this gap; the lower bound below is what will report it.
    """
    model = _model("deep_mgga_3x16")
    md = _md(model, "Li", "Li 0 0 0", 1, (("Li", 1),))
    # Convergence is reached in 21 cycles; the cap is 40 because the loop
    # carries its full trip count.
    converged = run_scf(_config(FeaturePolicy.REASSEMBLE, max_cycles=40,
                                conv_tol=1e-10), model, md, forward_only=True)
    assert bool(converged.converged)
    P_c = np.asarray(converged.density_matrix)
    rotate, occ = _one_orbital_rotation(md["s_matrix"], P_c[1])
    assert abs(occ[0] - 1.0) < 1e-9 and abs(occ[1]) < 1e-8, occ[:3]

    rng = np.random.default_rng(20260823)
    W_a = rng.standard_normal(P_c[0].shape)
    W_a = 0.5 * (W_a + W_a.T)
    W_full = rng.standard_normal(P_c.shape)
    W_full = 0.5 * (W_full + np.swapaxes(W_full, -1, -2))

    def alpha_only(theta):
        out = np.array(P_c)
        out[0] = P_c[0] + theta * W_a
        return out

    def beta_rotation(theta):
        out = np.array(P_c)
        out[1] = rotate(theta)
        return out

    def unconstrained(theta):
        return P_c + theta * W_full

    focks = _record_fock_matrices(monkeypatch)
    F_a, F_b = _fock_pair_at(model, md, P_c, focks)
    energy = _energy_helper(model, md)
    step = 1e-5

    def probe(P_of):
        dP = (P_of(step) - P_of(-step)) / (2.0 * step)
        analytic = float(np.sum(F_a * dP[0]) + np.sum(F_b * dP[1]))
        fd = (energy(P_of(step)) - energy(P_of(-step))) / (2.0 * step)
        return fd, analytic

    fd_a, an_a = probe(alpha_only)
    assert abs(an_a) > 1e-2, an_a
    rel_a = abs(fd_a - an_a) / max(abs(fd_a), abs(an_a), 1e-30)
    assert rel_a < 1e-6, (
        f"alpha direction (beta block still one orbital): FD={fd_a:.10e} "
        f"analytic={an_a:.10e} rel={rel_a:.3e}")

    fd_b, an_b = probe(beta_rotation)
    # An occupied-virtual rotation at a stationary point has zero gradient;
    # the pin is that BOTH sides report it, to 1e-7 absolute (measured
    # 6.6e-9 analytic against 7.6e-9, difference 9.2e-10).
    assert abs(an_b) < 1e-6 and abs(fd_b) < 1e-6, (fd_b, an_b)
    assert abs(fd_b - an_b) < 1e-7, (fd_b, an_b)

    fd_u, an_u = probe(unconstrained)
    rel_u = abs(fd_u - an_u) / max(abs(fd_u), abs(an_u), 1e-30)
    assert 1e-3 < rel_u < 5e-1, (
        f"the unconstrained direction reads rel={rel_u:.3e} (FD={fd_u:.10e} "
        f"analytic={an_u:.10e}); the measured 5.55e-2 is the indicator "
        f"response the one-electron gate drops off the single-orbital "
        f"manifold. Above the upper bound the beta Fock is rounding-selected "
        f"again (the gate disabled reads 0.88, analytic -2.61 against an FD "
        f"of -0.32). Below the lower bound the gate no longer drops a real "
        f"term -- re-anchor this test and the scope paragraph of "
        f"_drop_one_orbital_indicator_response")


@pytest.mark.parametrize("arch_name", ["deep_rung35_3x16", "deep_mgga_3x16",
                                       "deep_dm_3x16"])
def test_manual_uks_fock_is_the_derivative_on_a_spherical_open_shell(
        arch_name, monkeypatch):
    """The same energy/potential check as on the O atom, on the N atom (5
    alpha, 2 beta electrons; a half-filled p shell, so the reference density
    is spherical and carries none of O's 2p-hole orientation degeneracy), and
    resolved element by element rather than along one random direction.

    Both channels are multi-electron, so the one-electron indicator gate does
    NOT fire: this is the derivative check of the ungated three-block
    potential on the spec's remaining open-shell species. Directions: the
    diagonal element P[s][0,0] and two off-diagonal element pairs of EACH
    channel (a direction whose analytic derivative vanishes by symmetry
    proves nothing, so each is required to carry a derivative above 1).
    Measured relative residuals at the 1e-5 step, over the three
    architectures: 3.8e-13 to 6.6e-11; the bound is 1.5e3 times the worst and
    16x below the smallest defect signal, the superseded two-block evaluation
    (the total block in both exchange channels), which reads 1.6e-6
    (deep_rung35_3x16), 1.0e-4 (deep_mgga_3x16) and 1.6e-6 (deep_dm_3x16) on
    the alpha diagonal of this same probe.

    A random symmetric direction, which the O-atom test uses, is NOT usable
    here. N's beta channel holds 1s and 2s only, so tau - tau_W is small over
    most of space: the raw iso-orbital indicator of its doubled block has
    median 3.2e-3 and minimum 7.7e-9 against O's median 0.58 and minimum
    6.6e-4. A random symmetric step then crosses the lower clip of
    ``metagga.compute_alpha`` on 431 (h = 1e-6) to 710 (h = 1e-5) of the 4098
    resolved beta points and the central difference reads 6.0e-5 relative on
    ``deep_mgga_3x16``, a one-sided-derivative artefact of the clip
    (DEFERRED_WORK #27), not a potential defect. The element directions used
    here cross the clip on zero points at both steps.
    """
    model = _model(arch_name)
    md = _md(model, "N", "N 0 0 0", 3, (("N", 1),))
    P0 = np.asarray(md["dm_pbe"])
    focks = _record_fock_matrices(monkeypatch)
    F_a, F_b = _fock_pair_at(model, md, P0, focks)
    energy = _energy_helper(model, md)
    step = 1e-5

    directions = []
    for channel, label in ((0, "alpha"), (1, "beta")):
        for i, j in ((0, 0), (1, 0), (0, 2)):
            W = np.zeros_like(P0)
            W[channel, i, j] = 1.0
            W[channel, j, i] = 1.0
            directions.append((f"P[{label}][{i},{j}]"
                               + ("" if i == j else "+sym"), W))

    for label, W in directions:
        analytic = float(np.sum(F_a * W[0]) + np.sum(F_b * W[1]))
        assert abs(analytic) > 1.0, (
            f"{arch_name} {label}: derivative {analytic:.3e} is too small to "
            f"discriminate")
        fd = float((energy(P0 + step * W) - energy(P0 - step * W))
                   / (2.0 * step))
        rel = abs(fd - analytic) / max(abs(fd), abs(analytic), 1e-30)
        assert rel < 1e-7, (
            f"{arch_name} {label}: the UKS Fock pair is not dE/dP of the "
            f"three-block energy (FD={fd:.10e} analytic={analytic:.10e} "
            f"rel={rel:.3e})")


@pytest.mark.parametrize("name,atom,spin,composition", [
    ("O", "O 0 0 0", 2, (("O", 1),)), ("Li", "Li 0 0 0", 1, (("Li", 1),))])
def test_manual_uks_total_block_ingredients_match_the_kernel_contraction(
        name, atom, spin, composition):
    """The per-channel invariant of
    ``test_manual_uks_energy_ingredients_are_the_kernel_contractions`` is
    exact; the TOTAL-density one is not, and the residual is recorded here
    rather than claimed away.

    The loop forms the correlation term's ingredients as sums,
    ``rho_a + rho_b`` and ``|nabla_rho_a + nabla_rho_b|^2``, while the total
    descriptor block is built from the kernel contraction of ``P_a + P_b``.
    Floating-point addition is not associative, so the two differ on 68-71 per
    cent of the grid points of O and Li at 1.4e-15 relative or below, which
    moves the total block's iso-orbital indicator by 8.1e-12 (O) and 1.9e-13
    (Li) in the resolved region. Closing it costs one more grid contraction
    per energy evaluation and buys 1e-15; the residual is bounded here
    instead, at 1e-14 relative and 1e-10 on the indicator -- seven and two
    orders below any energy tolerance in the program (1.0 mHa per atom).
    Bitwise equality is NOT asserted: it does not hold today, and a change
    that made it hold would still satisfy this bound.
    """
    from xcquinox.alec.metagga import compute_alpha, compute_tau_from_dm
    model = _model("deep_mgga_3x16")
    md = _md(model, name, atom, spin, composition)
    ao_deriv = jnp.asarray(md["ao_grid_deriv"])
    P = jnp.asarray(md["dm_pbe"])
    rho_a, nabla_a, _ = _contract_dm_to_grid_with_nabla(P[0], ao_deriv)
    rho_b, nabla_b, _ = _contract_dm_to_grid_with_nabla(P[1], ao_deriv)
    nabla_sum = nabla_a + nabla_b
    rho_sum = rho_a + rho_b
    sigma_sum = jnp.einsum("gd,gd->g", nabla_sum, nabla_sum)
    rho_k, _nabla_k, sigma_k = _contract_dm_to_grid_with_nabla(P[0] + P[1],
                                                               ao_deriv)
    for label, summed, kernel in (("rho_tot", rho_sum, rho_k),
                                  ("sigma_tot", sigma_sum, sigma_k)):
        summed = np.asarray(summed)
        kernel = np.asarray(kernel)
        rel = np.abs(summed - kernel) / np.maximum(np.abs(kernel), 1e-300)
        assert float(rel.max()) < 1e-14, (
            f"{name} {label}: max relative difference {float(rel.max()):.3e}")
    tau = compute_tau_from_dm(ao_deriv[1:4], P)
    moved = np.abs(np.asarray(compute_alpha(rho_sum, sigma_sum, tau))
                   - np.asarray(compute_alpha(rho_k, sigma_k, tau)))
    resolved = np.asarray(rho_k) > 1e-6
    assert float(moved[resolved].max()) < 1e-10, (
        f"{name}: the total block's indicator moves by "
        f"{float(moved[resolved].max()):.3e} between the two forms")


def test_manual_uks_frozen_refuses_a_record_without_per_spin_blocks():
    """Under FROZEN the loop reads three stored blocks. A record written
    before the per-channel convention carries only the total block, and must
    be refused BY NAME: the alternative -- falling back to the total block in
    both exchange channels -- silently reinstates the superseded two-block
    evaluation, which moves the self-consistent O-atom energy by 48.5 mHa
    (30.5 kcal/mol). REASSEMBLE has no such requirement, because the live
    closures build all three blocks from the density matrix, so the same
    record still runs self-consistently; that asymmetry is the reason the
    refusal is worth pinning at the solver level rather than only at the
    descriptor.
    """
    model = _model("deep_mgga_3x16")
    md = _md(model, "Li", "Li 0 0 0", 1, (("Li", 1),))
    assert "metagga_features_a" in md and "metagga_features_b" in md
    legacy = {k: v for k, v in md.items()
              if k not in ("metagga_features_a", "metagga_features_b")}
    with pytest.raises(KeyError, match="metagga_features_a"):
        run_scf(_config(FeaturePolicy.FROZEN, mode=SolverMode.FIXED_J,
                        max_cycles=1, conv_tol=1e-14), model, legacy,
                forward_only=True)
    result = run_scf(_config(FeaturePolicy.REASSEMBLE, max_cycles=1,
                             conv_tol=1e-14), model, legacy,
                     forward_only=True)
    assert bool(jnp.isfinite(result.total_energy))
