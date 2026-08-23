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
    rho = jnp.einsum("ij,gi,gj->g", D, ao, ao)
    nabla = 2.0 * jnp.einsum("ij,dgi,gj->gd", D, ao_xyz, ao)
    return rho, nabla, jnp.sum(nabla * nabla, axis=1)


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
    sig_tot = jnp.sum(nab_t * nab_t, axis=1)
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
    TOTAL density of the H atom) is identically zero along the loop's own
    idempotent-density manifold and sits on the lower clip of
    ``metagga.compute_alpha``. Autodiff there returns the rounding-selected
    side of a 0/0: without the gate, the beta-channel feature-response term of
    Li/deep_mgga_3x16 is 1.13 Ha and MOVES BY 0.93 Ha under a 1e-14 relative
    change of the density matrix (H: 3.4e-3 alpha channel, 7.6e-4 total
    block; every multi-electron block is stable to 4e-16). Free H and Li are
    atomization anchors; the Fock the loop diagonalizes there must be a
    function of the density, not of the rounding. With the indicator columns
    of a one-electron block's de/df zeroed (the exact manifold value), the
    measured movement is (6.1e-16, 4.8e-11) on Li and (1.5e-15, 1.9e-15) on
    H: the 4.8e-11 residual is the VALUE of the beta block's indicator
    column -- a cancellation residue bounded by 4.7e-8 in the tail --
    wobbling through the fixed-feature JVPs, not a derivative term. The
    bound is 100x the measured worst and 1.8e7 below the ungated signal."""
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
            jnp.sum(nab_t * nab_t, axis=1), fa_of(P), fb_of(P), ft_of(P),
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
