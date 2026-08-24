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
    # tau - tau_W vanish identically so both columns are the smooth positive
    # part of a rounding residue.
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


def _perturbed_seed(P0, scale=1e-14, seed=7):
    """``P0`` moved by ``scale`` in relative Frobenius norm along a fixed
    random symmetric direction."""
    rng = np.random.default_rng(seed)
    N = rng.standard_normal(P0.shape)
    N = 0.5 * (N + np.swapaxes(N, -1, -2))
    return P0 + scale * np.linalg.norm(P0) / np.linalg.norm(N) * N


def test_manual_uks_h_atom_fock_is_rounding_stable_without_the_gate(
        monkeypatch):
    """The H atom's density is one orbital everywhere, so tau = tau_W
    pointwise and the raw iso-orbital indicator of its alpha block and of its
    total block is the rounding residue of that cancellation. With the hard
    clip at zero, autodiff returned the rounding-selected side of the kink and
    the Fock pair the loop diagonalizes moved by 4.2e-3 (alpha) and 7.1e-4
    (total) Ha under a 1e-14 relative change of the density matrix -- the
    reason the loop once zeroed the indicator's response on a one-electron
    block. With the smooth positive part of ``metagga.compute_alpha`` (width
    1e-5) the same probe, every column live, moves the pair by 3.6e-12 and
    1.3e-12 Ha (measured on deep_mgga_3x16, def2-svp, grid 1; 3.6e-11 at
    width 1e-6 and 3.5e-10 at 1e-7 -- the derivative's rounding sensitivity
    falls as 1/width). The bound is the one the gated loop was held to and
    clears the measurement by 1.4e3.
    """
    model = _model("deep_mgga_3x16")
    md = _md(model, "H", "H 0 0 0", 1, (("H", 1),))
    P0 = np.asarray(md["dm_pbe"])
    focks = _record_fock_matrices(monkeypatch)
    for seed in (P0, _perturbed_seed(P0)):
        md_run = dict(md)
        md_run["dm_seed"] = jnp.asarray(seed)
        run_scf(_config(FeaturePolicy.REASSEMBLE, max_cycles=1,
                        conv_tol=1e-14), model, md_run, forward_only=True)
    assert len(focks) == 4
    move_a = float(np.max(np.abs(focks[2] - focks[0])))
    move_b = float(np.max(np.abs(focks[3] - focks[1])))
    assert max(move_a, move_b) < 5e-9, (
        f"H: the Fock pair moved by ({move_a:.3e}, {move_b:.3e}) under a "
        f"1e-14 relative density change; a rounding-selected response is "
        f"reaching the eigensolver")


def test_manual_uks_li_beta_response_annihilates_the_occupied_orbital(
        monkeypatch):
    """Li's beta channel is one orbital c (rank one, tau = tau_W). The
    indicator response of that channel's block is a continuous function of
    the density matrix now, but its size is set by the descriptor's tail
    amplification, which is PEAKED ON A SHELL rather than a power law in the
    density: max |d alpha_raw / dP| reads 8.0e4 above 2 rho_beta = 1e-4,
    2.2e6 on 1e-6 to 1e-4, 1.3e8 on 1e-8 to 1e-6, 4.1e11 on 1e-9 to 1e-8 and
    1.2e1 below 1e-9 (log-log slope -0.43 against 2 rho, not -5/3). On this
    record the peak shell (rho_beta = 1.0e-9, 898 points) contributes 2.9e-3
    Ha per point to F_beta[1, 2], so a 1e-14 relative change of the density
    matrix moves the raw indicator by 4e-4 --
    40 widths -- and the beta Fock by 0.37 Ha (measured; 0.93 with the hard
    clip, 0.2-0.5 at every width from 1e-9 to 1e-5; DEFERRED_WORK.md entry
    30). What that movement can and cannot do is pinned here:

    * it never touches the occupied orbital. The raw indicator is stationary
      along every rank-preserving rotation of a one-orbital block, so the
      response annihilates c exactly, V_resp c = 0, and F_beta c is the same
      vector to rounding under the perturbation (measured 2e-13 against
      F_beta c of order 1 and the 0.37 Ha movement of the matrix). With
      dF c = 0 and dF symmetric, every matrix element of the movement that
      pairs with c -- the occupied-occupied element c^T dF c and the
      occupied-virtual row c^T dF d -- is bounded by |dF c|, so the movement
      lives entirely in the virtual-virtual block of the fixed channel;
    * the alpha channel, a two-orbital block, is stable to 1e-15.

    The fixed point of the loop therefore does not depend on the rounding
    (``test_manual_uks_fixed_points_are_reproducible_without_the_gate``).
    """
    model = _model("deep_mgga_3x16")
    md = _md(model, "Li", "Li 0 0 0", 1, (("Li", 1),))
    P0 = np.asarray(md["dm_pbe"])
    S = np.asarray(md["s_matrix"])
    c = _rank_one_orbital(P0[1], S)
    focks = _record_fock_matrices(monkeypatch)
    for seed in (P0, _perturbed_seed(P0)):
        md_run = dict(md)
        md_run["dm_seed"] = jnp.asarray(seed)
        run_scf(_config(FeaturePolicy.REASSEMBLE, max_cycles=1,
                        conv_tol=1e-14), model, md_run, forward_only=True)
    F_a0, F_b0, F_a1, F_b1 = focks
    assert float(np.max(np.abs(F_a1 - F_a0))) < 1e-12
    dF = F_b1 - F_b0
    move = float(np.max(np.abs(dF)))
    # The virtual-block movement is the documented tail response; the pin is
    # that it exists (a vanishing movement would mean the response term is no
    # longer live) and that it stays in the class measured.
    assert 1e-3 < move < 5.0, move
    action = np.abs(dF @ c).max()
    assert action < 1e-10, (action, move)
    # dF is symmetric, so |c^T dF d| <= |dF c| |d| for every d: the movement
    # carries no occupied-occupied and no occupied-virtual component beyond
    # the rounding of dF c, i.e. it sits in the virtual-virtual block.
    assert float(np.abs(c @ dF @ c)) < 1e-10
    assert float(np.max(np.abs(c @ dF))) < 1e-10


@pytest.mark.parametrize("name,atom,composition", [
    ("Li", "Li 0 0 0", (("Li", 1),)), ("H", "H 0 0 0", (("H", 1),))])
def test_manual_uks_fixed_points_are_reproducible_without_the_gate(
        name, atom, composition):
    """H and Li converge to the same energy from two seeds 1e-14 apart, with
    every column live: the tail response of the one-orbital channel sits in
    the virtual-virtual block of its Fock and cannot move the occupied
    orbital, so the fixed point is a continuous function of the seed with no
    rounding selection. Measured differences 0.0 (H) and 1.9e-14 Ha (Li) at
    conv_tol 1e-10 (20 cycles on Li) -- the Li figure is of the order of the
    seed separation itself (1e-14 relative of a 7.3 Ha energy), i.e. the
    response of a continuous map to the moved input, not a draw. The bound
    is 5e-14 Ha, 2.7x the measured worse.
    """
    model = _model("deep_mgga_3x16")
    md = _md(model, name, atom, 1, composition)
    P0 = np.asarray(md["dm_pbe"])
    energies = []
    for seed in (P0, _perturbed_seed(P0)):
        md_run = dict(md)
        md_run["dm_seed"] = jnp.asarray(seed)
        result = run_scf(_config(FeaturePolicy.REASSEMBLE, max_cycles=40,
                                 conv_tol=1e-10), model, md_run,
                         forward_only=True)
        assert bool(result.converged), name
        energies.append(float(result.total_energy))
    assert abs(energies[0] - energies[1]) < 5e-14, energies


def _rank_one_orbital(P_s, s_matrix):
    """The S-orthonormal orbital c of a one-electron spin density matrix
    P_s = c c^T."""
    k = int(np.argmax(np.diag(P_s)))
    c = P_s[:, k] / np.sqrt(P_s[k, k])
    assert float(np.max(np.abs(P_s - np.outer(c, c)))) < 1e-10, "not rank one"
    assert abs(float(c @ s_matrix @ c) - 1.0) < 1e-8, "not S-normalized"
    return c


@pytest.mark.parametrize("arch_name", ["deep_rung35_3x16", "deep_mgga_3x16",
                                       "deep_dm_3x16"])
def test_manual_uks_fock_is_the_derivative_of_the_three_block_energy(
        arch_name, monkeypatch):
    """The Fock matrices the loop diagonalizes must be dE/dP_sigma of the
    energy the loop reports, with the three live feature maps differentiated
    -- the solver's own matrices, captured on their way to the eigensolver,
    against a central difference of the solver's own energy helper.

    Probe: the O atom (5 alpha, 3 beta electrons) along an unrestricted
    random symmetric direction in both channels. Every block of O keeps a
    raw indicator >= 6.6e-4 on the resolved grid, so the 1e-6 step moves it
    by a small fraction of its value everywhere that carries integrand mass
    and the linear displacement is a valid probe; on the one-orbital
    channels of H and Li, and on N's beta channel, it is not (the tail
    response of the indicator; see ``_rotation_path`` and the tests below).
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
def test_manual_uks_fock_is_the_derivative_along_a_one_orbital_rotation(
        name, atom, composition, channel, tol, monkeypatch):
    """On a ONE-ELECTRON spin channel the Fock pair, every column live, is
    the derivative of the three-block energy along the manifold of
    single-orbital densities.

    The probe is a rank-preserving rotation of that channel's occupied
    orbital (H alpha, Li beta), taken at the PBE seed rather than at the
    fixed point, where every orbital rotation is stationary and the check
    would be vacuous. ``tau = tau_W`` holds identically along such a path,
    so the raw indicator is a rounding residue (8.3e-11 on H's channel
    block) sitting at the smoothing's floor, and its response along the
    path vanishes at first order; the gradient of every other ingredient is
    untouched. Measured (``deep_mgga_3x16``, def2-svp, grid level 1, 1e-5
    step): dE/dtheta = -6.869811e-3 with a relative residual 7.5e-11 (H) and
    -6.807988e-3 with 8.6e-8 relative (Li, whose beta indicator column
    carries a larger cancellation residue), the same figures the gated loop
    gave (7.8e-10 and 8.7e-8), as they must be: at an exactly idempotent
    one-orbital block the response is numerically orthogonal to a
    rank-preserving rotation. The residual falls as h^2 from the 1e-3 step
    (5.5e-6 / 2.5e-5), so the quantity being reproduced is the derivative
    and not a coincidence of one step.
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
        f"{name}: the Fock pair is not dE/dtheta along the one-orbital "
        f"manifold (FD={fd:.10e} analytic={analytic:.10e} rel={rel:.3e})")


def _rotation_path(P0, s_matrix, seed=20260824):
    """``eps -> P(eps)``: every populated channel of ``P0`` rotated by
    ``expm(eps K_s)`` in the S-metric, ``K_s`` a random antisymmetric
    generator of unit Frobenius norm -- the manifold of aufbau density
    matrices the Roothaan step returns (rank, idempotency and positive
    semidefiniteness preserved at every eps). The tangent at eps = 0 is a
    random symmetric matrix in both channels. The iterating loop does not
    stay on that manifold: it assembles its features and its Fock at the
    MIXER output, a convex combination of two rank-nocc projectors, which is
    rank two for a one-electron block. This path is therefore the geometry
    that keeps a probe off the rank-one boundary, not the only density the
    loop visits (DEFERRED_WORK.md entry 30)."""
    from scipy.linalg import expm
    ev, evec = np.linalg.eigh(np.asarray(s_matrix))
    s_half = evec @ np.diag(np.sqrt(ev)) @ evec.T
    s_inv_half = evec @ np.diag(1.0 / np.sqrt(ev)) @ evec.T
    rng = np.random.default_rng(seed)
    generators = []
    for s in range(P0.shape[0]):
        K = rng.standard_normal(P0[s].shape)
        K = K - K.T
        generators.append(K / np.linalg.norm(K))

    def P_of(eps):
        out = np.array(P0)
        for s, K in enumerate(generators):
            if float(np.max(np.abs(P0[s]))) == 0.0:
                continue
            U = s_inv_half @ expm(eps * K) @ s_half
            out[s] = U @ np.asarray(P0[s]) @ U.T
        return out

    return P_of


_SCF_MANIFOLD_CASES = [
    ("H", "H 0 0 0", 1, (("H", 1),)), ("Li", "Li 0 0 0", 1, (("Li", 1),)),
    ("N", "N 0 0 0", 3, (("N", 1),)), ("O", "O 0 0 0", 2, (("O", 1),))]


@pytest.mark.parametrize("name,atom,spin,composition", _SCF_MANIFOLD_CASES,
                         ids=[c[0] for c in _SCF_MANIFOLD_CASES])
def test_manual_uks_fock_is_the_derivative_along_the_scf_manifold(
        name, atom, spin, composition, monkeypatch):
    """The Fock pair, every column live, is the derivative of the three-block
    energy along a RANDOM direction tangent to the SCF's own manifold in
    BOTH channels, on all four open-shell atoms -- no gate, no mask.

    The direction is the tangent of a random orbital rotation of every
    populated channel (``_rotation_path``), i.e. a random symmetric matrix
    per channel restricted to the directions along which the density matrix
    stays an aufbau matrix. The NET derivative along such a direction can be
    small -- the reference density is a PBE fixed point, near-stationary
    under rotations for any functional close to it: measured nets of 8e-3
    (H), 2e-3 (Li), 9e-4 (N) and of order 1e-3 on O (draw-dependent, the O
    record carrying the 2p-hole orientation) against absolute contributions
    summing to 0.13 (H), 0.57 (Li), 6.9 (N) and 9.6 (O) -- so the residual
    is taken relative to the sum of ABSOLUTE contributions sum |F . W| per
    channel, the scale the finite difference and the contraction are
    actually computed at, with that scale required to be above 0.1.
    Measured residuals on that footing with the solver's own Fock pair
    (``deep_mgga_3x16``, def2-svp, grid level 1, 1e-5 step): 2.9e-11 (H),
    2.4e-10 (Li), 9.8e-11 (N), 5.5e-11 (O); the bound of 5e-9 clears the
    worst by 21x.

    What an UNRESTRICTED random symmetric direction reads on the same Fock
    pair, for the record: 3.7e-8 on H at the 1e-6 step, falling to 3.8e-10
    at 1e-7 (the H atom is where the clip's kink used to give 7.4e-4 flat
    in the step, the closure of DEFERRED_WORK.md entry 27), but 5.2e-2 on
    Li and 6.0e-6 on N, flat between the 1e-5 and 1e-7 steps. What such a
    direction hits is a RANK-ONE channel, not the boundary of the positive
    semidefinite cone: on Li, displacing the alpha channel alone takes it
    out of the cone (minimum eigenvalue -2.2e-7) and still reads 1.2e-10,
    while displacing the rank-one beta channel alone reads 0.74 flat in the
    step, its block indicator saturating the ceiling at every step. On such
    a channel the tail response -- peaked on the 2 rho_beta ~ 1e-9 to 1e-8
    shell, where max |d alpha_raw / dP| = 4.1e11 -- moves the raw indicator
    by 1e3-1e5 at a 1e-6 step, beyond any linear regime of the energy at any
    width of the smooth positive part (entry 30). The manifold direction is
    the tangent the Roothaan step needs, and on it the potential is the
    derivative; an unrestricted direction is a valid probe as well once the
    base point is off the rank-one boundary, which entry 30 records at
    5.4e-11 on the density the mixer produces.
    """
    model = _model("deep_mgga_3x16")
    md = _md(model, name, atom, spin, composition)
    P0 = np.asarray(md["dm_pbe"])
    P_of = _rotation_path(P0, md["s_matrix"])
    focks = _record_fock_matrices(monkeypatch)
    F_a, F_b = _fock_pair_at(model, md, P0, focks)
    energy = _energy_helper(model, md)
    step = 1e-5
    dP = (P_of(step) - P_of(-step)) / (2.0 * step)
    analytic = float(np.sum(F_a * dP[0]) + np.sum(F_b * dP[1]))
    scale = float(np.sum(np.abs(F_a * dP[0])) + np.sum(np.abs(F_b * dP[1])))
    assert scale > 0.1, (name, scale)
    fd = (energy(P_of(step)) - energy(P_of(-step))) / (2.0 * step)
    rel = abs(fd - analytic) / scale
    assert rel < 5e-9, (
        f"{name}: the UKS Fock pair is not dE/dP along the SCF manifold "
        f"(FD={fd:.10e} analytic={analytic:.10e} scale={scale:.3e} "
        f"rel={rel:.3e})")


def test_manual_uks_h_atom_fock_is_the_derivative_along_an_unrestricted_direction(
        monkeypatch):
    """The clip's kink, closed: on the H atom -- one orbital everywhere, so
    every grid point sat on the indicator's lower clip -- a central
    difference along an UNRESTRICTED random symmetric direction read 7.4e-4
    relative at every step from 1e-5 to 1e-7 with the hard clip (the average
    of two one-sided slopes). With the smooth positive part the same probe
    reads 9.3e-7, 3.7e-8 and 3.8e-10 to 6.2e-10 at the 1e-5, 1e-6 and 1e-7
    steps (three reference solutions), i.e. it falls with the step as a
    derivative must. The 1e-7 step is used; the bound clears the worst
    measurement by 16x and refuses the clip by 1e5.
    """
    model = _model("deep_mgga_3x16")
    md = _md(model, "H", "H 0 0 0", 1, (("H", 1),))
    P0 = np.asarray(md["dm_pbe"])
    rng = np.random.default_rng(20260821)
    W = rng.standard_normal(P0.shape)
    W = 0.5 * (W + np.swapaxes(W, -1, -2))
    W[1] = 0.0                                  # the beta channel is empty
    focks = _record_fock_matrices(monkeypatch)
    F_a, F_b = _fock_pair_at(model, md, P0, focks)
    energy = _energy_helper(model, md)
    analytic = float(np.sum(F_a * W[0]) + np.sum(F_b * W[1]))
    step = 1e-7
    fd = (energy(P0 + step * W) - energy(P0 - step * W)) / (2.0 * step)
    rel = abs(fd - analytic) / max(abs(fd), abs(analytic), 1e-30)
    assert rel < 1e-8, (
        f"H: FD={fd:.10e} analytic={analytic:.10e} rel={rel:.3e}")


def test_manual_uks_fock_at_the_li_fixed_point(monkeypatch):
    """Li at its own fixed point, every column live.

    Four directions at the converged density. An alpha-channel perturbation
    (the beta block untouched) reproduces dE/dP to 2e-10 relative on a
    derivative of 1.183; a rank-preserving rotation of the beta orbital
    reproduces the stationarity of the fixed point (both sides below 1e-6,
    agreeing to 1e-7); a random rotation of BOTH channels -- the SCF's own
    manifold -- is likewise stationary at the model's own fixed point (both
    sides at -7e-9, agreeing to 3e-10: the occupied-virtual blocks of the
    converged Fock vanish, which is itself the SCF condition and holds with
    every column live); and the direction the gated loop could not be
    checked on -- BOTH channels moving, the alpha block linearly and the
    beta block along its rotation manifold, so the probe is nonstationary
    while the beta block stays rank one -- reproduces dE/dP at the same
    floor as the constrained direction (measured 3.8e-10 relative on a
    derivative of 1.183 at the 1e-5 step). That last statement is what
    closing DEFERRED_WORK.md entry 27 changed: with the occupancy gate live
    the two-channel probes read 5.5e-2, the response the gate dropped off
    the single-orbital manifold.

    What is deliberately NOT asserted to be at the floor: an unrestricted
    random symmetric direction, which reads 0.88 relative here (analytic
    -2.58 against a finite difference of -0.316, flat between the 1e-5 and
    1e-6 steps) with the clip and with the smoothing alike. The beta block
    is one orbital, and a linear displacement of a RANK-ONE block drives
    the raw indicator of its peak response shell (2 rho_beta ~ 1e-9 to 1e-8,
    max |d alpha_raw / dP| = 4.1e11) by 1e4-1e5 per step, so the energy is
    not linearizable over any usable step there and the probe measures the
    descriptor's tail response, not the potential (entry 30). Cone departure
    is not what does it: this same displacement applied to Li's ALPHA
    channel alone leaves the cone by -2.2e-7 and still reproduces dE/dP to
    1.2e-10. That figure is pinned as a lower bound so a change of the
    descriptor's tail behaviour is noticed.
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
    manifold = _rotation_path(P_c, md["s_matrix"])

    def alpha_only(theta):
        out = np.array(P_c)
        out[0] = P_c[0] + theta * W_a
        return out

    def beta_rotation(theta):
        out = np.array(P_c)
        out[1] = rotate(theta)
        return out

    def unrestricted(theta):
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
    # 6.7e-9 analytic against 7.5e-9, difference 8e-10).
    assert abs(an_b) < 1e-6 and abs(fd_b) < 1e-6, (fd_b, an_b)
    assert abs(fd_b - an_b) < 1e-7, (fd_b, an_b)

    fd_m, an_m = probe(manifold)
    # Stationarity along the SCF manifold at the model's own fixed point:
    # the converged Fock's occupied-virtual blocks vanish, so both sides are
    # zero to the convergence floor and agree.
    assert abs(an_m) < 1e-6 and abs(fd_m) < 1e-6, (fd_m, an_m)
    assert abs(fd_m - an_m) < 1e-7, (fd_m, an_m)

    def combined(theta):
        out = np.array(P_c)
        out[0] = P_c[0] + theta * W_a
        out[1] = rotate(theta)
        return out

    fd_c, an_c = probe(combined)
    assert abs(an_c) > 1e-2, an_c
    rel_c = abs(fd_c - an_c) / max(abs(fd_c), abs(an_c), 1e-30)
    assert rel_c < 1e-6, (
        f"both channels moving, beta on its rank-one manifold: "
        f"FD={fd_c:.10e} analytic={an_c:.10e} rel={rel_c:.3e}; the Fock "
        f"pair is not the derivative along the SCF's own directions at the "
        f"fixed point")

    fd_u, an_u = probe(unrestricted)
    rel_u = abs(fd_u - an_u) / max(abs(fd_u), abs(an_u), 1e-30)
    assert rel_u > 1e-2, (
        f"the unrestricted direction reads rel={rel_u:.3e} (FD={fd_u:.10e} "
        f"analytic={an_u:.10e}); the measured 0.88 is the descriptor's tail "
        f"response off the positive semidefinite cone (DEFERRED_WORK.md "
        f"entry 30). Below this bound the tail behaviour of the indicator "
        f"has changed -- re-anchor this test and entry 30")


@pytest.mark.parametrize("arch_name", ["deep_rung35_3x16", "deep_mgga_3x16",
                                       "deep_dm_3x16"])
def test_manual_uks_fock_is_the_derivative_on_a_spherical_open_shell(
        arch_name, monkeypatch):
    """The same energy/potential check as on the O atom, on the N atom (5
    alpha, 2 beta electrons; a half-filled p shell, so the reference density
    is spherical and carries none of O's 2p-hole orientation degeneracy), and
    resolved element by element rather than along one random direction.

    Directions: the diagonal element P[s][0,0] and two off-diagonal element
    pairs of EACH channel (a direction whose analytic derivative vanishes by
    symmetry proves nothing, so each is required to carry a derivative above
    1), and a random rotation of both channels along the SCF's manifold
    (``_rotation_path``; the net derivative there is small, the reference
    being a PBE fixed point, so its residual is taken against the sum of
    absolute contributions). Measured residuals at the 1e-5 step, over the
    three architectures: 3.8e-13 to 6.6e-11 on the element directions and
    5.6e-11 to 1.1e-10 of the contribution scale (6.9) on the manifold
    direction, bounded at 1e-9; the element bound is
    1.5e3 times the worst element residual and 16x below the smallest defect
    signal, the superseded two-block evaluation (the total block in both
    exchange channels), which reads 1.6e-6 (deep_rung35_3x16), 1.0e-4
    (deep_mgga_3x16) and 1.6e-6 (deep_dm_3x16) on the alpha diagonal of this
    same probe.

    An UNRESTRICTED random symmetric direction, which the O-atom test uses,
    is not a valid probe here. N's beta channel holds 1s and 2s only, so
    tau - tau_W is small over most of space: the raw iso-orbital indicator
    of its doubled block has median 3.2e-3 and minimum 7.7e-9 against O's
    median 0.58 and minimum 6.6e-4, and the descriptor's tail response --
    peaked on the low-density shell rather than a power law in the density,
    max |d alpha_raw / dP| reaching 4.1e11 on Li's 2 rho ~ 1e-9 to 1e-8 band
    against 1.2e1 below it -- lets a 1e-6 step move it by orders of
    magnitude on hundreds of resolved points (once read as 431 to 710 clip
    crossings at the 1e-6 and 1e-5 steps). The central difference then reads
    6.0e-6 relative on ``deep_mgga_3x16`` at the 1e-6 step with the smooth
    positive part in place, exactly as with the hard clip, and flat from
    1e-5 to 1e-7 -- the descriptor's tail response along an unrestricted
    linear displacement (DEFERRED_WORK.md entry 30), not a potential defect
    and not a consequence of leaving the positive semidefinite cone, which
    on Li is measured harmless on its own. The element directions and the
    manifold direction do not reach that regime.
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

    P_of = _rotation_path(P0, md["s_matrix"])
    dP = (P_of(step) - P_of(-step)) / (2.0 * step)
    analytic = float(np.sum(F_a * dP[0]) + np.sum(F_b * dP[1]))
    scale = float(np.sum(np.abs(F_a * dP[0])) + np.sum(np.abs(F_b * dP[1])))
    assert scale > 1.0, (arch_name, scale)
    fd = (energy(P_of(step)) - energy(P_of(-step))) / (2.0 * step)
    rel = abs(fd - analytic) / scale
    assert rel < 1e-9, (
        f"{arch_name} SCF-manifold direction: FD={fd:.10e} "
        f"analytic={analytic:.10e} scale={scale:.3e} rel={rel:.3e}")


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
