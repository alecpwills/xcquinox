"""Fractional occupations in the manual solver and the diagnostics module built on them.

The two Roothaan density builders fill the orbitals in order through the mask
``clip(nocc - i, 0, 1)``: at an integer count the density's occupations
against the overlap are ones and zeros, at a fractional count the fraction
sits in the orbital after the last full one, and the empty channel of a
one-electron atom skips the eigensolver. On that mask the diagnostics module
runs the H atom at a fractional electron number: with the PBE parent as the
model (``parents.pbe_fx`` and ``pbe_fc`` in the model's shell) the curve is
pyscf's UKS PBE at the same occupations, to the solver's convergence above
one electron and to the correlation offset of the clamped spin polarization
below it (``oneshot.uks_zeta`` holds zeta at 1 - 1e-6 where libxc evaluates
1: 7.1e-7 Ha on the fully polarized atom). The references are pyscf's own:
UHF for the one-electron atom, CCSD converged tightly for the two-electron
systems (full CI in the basis), the straight lines between them.

Oracles: the eigenvalues of ``S_r^1/2 D S_r^1/2`` (``S_r`` the overlap the
builders orthonormalize against) and ``tr(D S_r)``; pyscf's UKS, RKS, UHF
and CCSD built here at the item's identity (def2-TZVP, grid level 4, the
pinned density cutoff, which makes pyscf's grid of the H atom the record's
point for point); ``run_scf`` on a record precomputed here (the H2 point).
Tolerances: 1e-12 on the occupations (measured 2.4e-15); 1.5e-6 Ha on the
fully polarized atom (measured 7.11e-7 at N = 1, 2.96e-7 at N = 0.5) and
5e-8 above one electron (measured 8.95e-9 at N = 1.5), 2.1 and 5.6 times the
measurements; 1e-8 on the CCSD reference (pyscf's default convergence sits
4.9e-8 above full CI at 1.4 bohr and is refused); 1e-7 on the restricted
comparator. The fractional SCF runs at 40 cycles, about two minutes.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import types

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from pyscf import cc, dft, gto, scf

import xcquinox.pipeline as pipeline
import xcquinox.pipeline.solver_manual as solver_manual
from xcquinox.pipeline import diagnostics
from xcquinox.pipeline.data import clear_precompute_cache, precompute_fixed_density_data
from xcquinox.pipeline.models import AlecGGAModel
from xcquinox.pipeline.parents import pbe_fc, pbe_fx
from xcquinox.pipeline.pyscf_determinism import pin_small_rho_cutoff
from xcquinox.pipeline.solver import (
    DEGENERACY_REG, FeaturePolicy, SolverBackend, SolverConfig, SolverMode, run_scf,
)

jax.config.update("jax_enable_x64", True)

_BASIS = "def2-TZVP"
_GRID = 4
_KCAL = 627.5094740631
_FRACTIONAL_CYCLES = 40
_TOL_POLARIZED = 1.5e-6
_TOL_OCCUPIED_BETA = 5e-8


class _PBEExchange(eqx.Module):
    def __call__(self, row):
        return pbe_fx(row[0], row[1])


class _PBECorrelation(eqx.Module):
    use_spin_polarization: bool = eqx.field(static=True, default=True)

    def __call__(self, row):
        return pbe_fc(row[0], row[1], row[2])


def _pbe_parent_model():
    """The PBE parent in the model's shell: exchange on the doubled-density
    row, correlation relative to the PW92 baseline on the polarized row."""
    return AlecGGAModel(xnet=_PBEExchange(), cnet=_PBECorrelation(), descriptors=())


class _NoDescriptorArch:
    name = "pbe_parent"

    def materialize_descriptors(self):
        return ()


def _spec(arch):
    """A stand-in for a Slim16 network's training spec: its architecture
    and a density-fitted training solver; the diagnostics run with full
    integrals whatever the spec trained with."""
    sc = SolverConfig(backend=SolverBackend.MANUAL, mode=SolverMode.FULL, max_cycles=25,
                      conv_tol=1e-6, feature_policy=FeaturePolicy.REASSEMBLE,
                      density_fit=True, auxbasis="def2-tzvp-jkfit")
    return types.SimpleNamespace(arch=arch, solver_config=sc)


def _clone(name, seed=0):
    arch = dataclasses.replace(pipeline.get_architecture(name),
                               use_polarized_correlation=True, zero_init_final_layer=False)
    xnet, cnet = pipeline.create_network_pair(arch, seed=seed)
    return arch, pipeline.AlecGGAModel.from_arch(arch, xnet=xnet, cnet=cnet)


def _h_atom():
    return gto.M(atom="H 0 0 0", basis=_BASIS, spin=1, verbose=0)


def _fractional_uks(n, xc="pbe"):
    """pyscf's UKS energy of the H atom at N electrons through the occupation
    override, the record's own grid, conv_tol 1e-11."""
    mol = _h_atom()
    mf = dft.UKS(mol)
    mf.xc = xc
    mf.grids.level = _GRID
    pin_small_rho_cutoff(mf)
    mf.conv_tol = 1e-11
    nao = mol.nao
    occ = np.stack([np.clip(min(n, 1.0) - np.arange(nao), 0.0, 1.0),
                    np.clip(max(n - 1.0, 0.0) - np.arange(nao), 0.0, 1.0)])
    mf.get_occ = lambda mo_energy=None, mo_coeff=None: occ
    energy = float(mf.kernel())
    assert mf.converged
    return energy


def _rks(r_bohr, xc):
    mol = gto.M(atom=f"H 0 0 0; H 0 0 {r_bohr}", unit="bohr", basis=_BASIS, verbose=0)
    mf = dft.RKS(mol)
    mf.xc = xc
    mf.grids.level = _GRID
    pin_small_rho_cutoff(mf)
    mf.conv_tol = 1e-11
    energy = float(mf.kernel())
    assert mf.converged
    return energy


def _ccsd(mol):
    """RHF then CCSD converged far below the comparison (1e-12 on the
    energy, 1e-10 on the amplitudes); exact for two electrons."""
    hf = scf.RHF(mol)
    hf.conv_tol = 1e-12
    hf.kernel()
    assert hf.converged
    ccsd = cc.CCSD(hf)
    ccsd.conv_tol = 1e-12
    ccsd.conv_tol_normt = 1e-10
    ccsd.kernel()
    assert ccsd.converged
    return float(ccsd.e_tot)


def _uhf_h():
    mf = scf.UHF(_h_atom())
    mf.conv_tol = 1e-12
    energy = float(mf.kernel())
    assert mf.converged
    return energy


def test_the_occupation_mask_fills_the_orbitals_in_order(monkeypatch):
    """On random symmetric Fock and overlap matrices (nao 6): at the integer
    counts 0 to 3 the density's occupations against S_r are ones then zeros
    (twice that restricted) and tr(D S_r) the count, on the default int path
    and on the padded path (a traced float64 0-d array under jit); at 0.5 and
    1.5 they are {0.5} and {1, 0.5}; the unrestricted builder skips the
    eigensolver for a concrete empty channel and calls it once otherwise."""
    rng = np.random.default_rng(20261008)
    nao = 6
    a = rng.standard_normal((nao, nao))
    s = a @ a.T / nao + np.eye(nao)
    d = 1.0 / np.sqrt(np.diag(s))
    S = jnp.asarray(s * d[:, None] * d[None, :])
    f = rng.standard_normal((nao, nao))
    F = jnp.asarray(0.5 * (f + f.T))
    s_reg = np.asarray(S) + DEGENERACY_REG * np.eye(nao)
    w, u = np.linalg.eigh(s_reg)
    half = u @ np.diag(np.sqrt(w)) @ u.T
    builders = (("_diagonalize_roothaan_unrestricted", 1.0), ("_diagonalize_roothaan", 2.0))
    for nocc, occupied in ((0, []), (1, [1.0]), (2, [1.0, 1.0]), (3, [1.0, 1.0, 1.0]),
                           (0.5, [0.5]), (1.5, [1.0, 0.5])):
        for name, factor in builders:
            builder = getattr(solver_manual, name)
            for label, D in (("eager", builder(F, S, nocc)),
                             ("traced", jax.jit(builder)(F, S, jnp.asarray(float(nocc))))):
                D = np.asarray(D)
                occ = np.sort(np.linalg.eigvalsh(half @ D @ half))[::-1]
                expected = factor * np.array(occupied + [0.0] * (nao - len(occupied)))
                assert occ == pytest.approx(expected, abs=1e-12), (name, nocc, label, occ)
                assert float(np.trace(D @ s_reg)) == pytest.approx(factor * nocc, abs=1e-12)

    calls = []
    eigh = jnp.linalg.eigh

    def counting_eigh(*args, **kwargs):
        calls.append(1)
        return eigh(*args, **kwargs)

    monkeypatch.setattr(jnp.linalg, "eigh", counting_eigh)
    empty = solver_manual._diagonalize_roothaan_unrestricted(F, S, 0)
    assert calls == [] and not np.any(np.asarray(empty))
    solver_manual._diagonalize_roothaan_unrestricted(F, S, 1)
    assert calls == [1]


@pytest.mark.slow
def test_the_fractional_charge_curve_of_the_pbe_parent_is_pyscfs():
    """With the PBE parent as the model, ``fractional_charge_curve`` on the H
    atom at N in {0.5, 1, 1.5} is pyscf's UKS PBE at the same occupations,
    the point at N = 1 is ``run_scf`` on the unmodified record (the same
    number), the precomputed record keeps its integer counts and its seed,
    and ``comparator_fractional`` is pyscf's override run here (1e-8, two
    pyscf runs on one grid)."""
    model = _pbe_parent_model()
    config = dataclasses.replace(diagnostics.diagnostic_solver_config(),
                                 max_cycles=_FRACTIONAL_CYCLES)
    n_values = [0.5, 1.0, 1.5]
    curves = diagnostics.fractional_charge_curve(
        {"PBE parent": (model, _spec(_NoDescriptorArch()))}, n_values, config)
    curve = curves["PBE parent"]
    assert all(curve["converged"]), curve
    for n, energy in zip(n_values, curve["E"]):
        tol = _TOL_POLARIZED if n <= 1.0 else _TOL_OCCUPIED_BETA
        assert energy == pytest.approx(_fractional_uks(n), abs=tol), n
    record = precompute_fixed_density_data(diagnostics.h_atom_spec(),
                                           required_keys=("eri",), descriptors=())
    direct = run_scf(config, model, record, forward_only=True)
    assert curve["E"][1] == float(direct.total_energy)
    assert (record["nocc_a"], record["nocc_b"]) == (1, 0)
    assert type(record["nocc_a"]) is int
    assert float(np.max(np.abs(np.asarray(record["dm_seed"])
                               - np.asarray(record["dm_pbe"])))) == 0.0
    comparator = diagnostics.comparator_fractional([0.5, 1.5], ["pbe"])["PBE"]
    assert comparator["E"] == pytest.approx([_fractional_uks(0.5), _fractional_uks(1.5)],
                                            abs=1e-8)
    assert all(comparator["converged"])


def test_h2_at_1p4_bohr_is_run_scf_and_the_references_are_pyscfs():
    """``dissociation_curve`` on a seed-0 deep_3x16 clone at 1.4 bohr under
    the manual configuration is ``run_scf`` on a record precomputed here
    from ``h2_spec(1.4)`` with full integrals, exactly; ``ccsd_dissociation``
    at 1.4 bohr is pyscf's CCSD converged tightly; ``comparator_dissociation``
    at 6.0 bohr is pyscf's restricted PBE. The geometry is 1.4 bohr in
    angstrom (relative 1e-9)."""
    ms = diagnostics.h2_spec(1.4)
    atoms = [tok.split() for tok in ms.atom.split(";") if tok.strip()]
    xyz = np.array([[float(v) for v in a[1:4]] for a in atoms])
    assert float(np.linalg.norm(xyz[1] - xyz[0])) == pytest.approx(
        1.4 * diagnostics.BOHR_TO_ANGSTROM, rel=1e-9)
    assert ms.spin == 0 and ms.grid_level == _GRID and ms.basis == _BASIS
    arch, model = _clone("deep_3x16")
    config = dataclasses.replace(diagnostics.diagnostic_solver_config(), max_cycles=4)
    curve = diagnostics.dissociation_curve({"S_deep_3x16": (model, _spec(arch))}, [1.4],
                                           config)["S_deep_3x16"]
    clear_precompute_cache()
    record = precompute_fixed_density_data(ms, required_keys=("eri",),
                                           descriptors=model.descriptors)
    direct = run_scf(config, model, record, forward_only=True)
    assert math.isfinite(float(direct.total_energy))
    assert curve["E"] == [float(direct.total_energy)]
    assert curve["converged"] == [bool(direct.converged)]
    assert curve["cycles"] == [int(direct.cycles_run)]
    mol = gto.M(atom="H 0 0 0; H 0 0 1.4", unit="bohr", basis=_BASIS, verbose=0)
    assert diagnostics.ccsd_dissociation([1.4]) == pytest.approx([_ccsd(mol)], abs=1e-8)
    comparator = diagnostics.comparator_dissociation([6.0], ["pbe"])["PBE"]
    assert comparator["E"] == pytest.approx([_rks(6.0, "pbe")], abs=1e-7)


def test_the_fractional_reference_is_the_straight_line_between_exact_integers():
    """On the default N grid: E(0) = 0, E(1) the H atom's UHF energy, E(2)
    the anion's CCSD energy, the straight lines between them (the midpoints
    at 0.5 and 1.5 among them); the default grids are the decimal grids."""
    n_grid = [float(v) for v in diagnostics.grid(*diagnostics.N_DEFAULT)]
    assert n_grid == [k / 10 for k in range(0, 21)]
    assert [float(v) for v in diagnostics.grid(*diagnostics.R_BOHR_DEFAULT)] == [
        k / 10 for k in range(5, 61)]
    reference = diagnostics.exact_fractional_reference(n_grid)
    e1 = _uhf_h()
    e2 = _ccsd(gto.M(atom="H 0 0 0", basis=_BASIS, charge=-1, spin=0, verbose=0))
    integers = reference["integer_energies"]
    assert float(integers["0"]) == 0.0
    assert float(integers["1"]) == pytest.approx(e1, abs=1e-10)
    assert float(integers["2"]) == pytest.approx(e2, abs=1e-8)
    expected = [n * e1 if n <= 1.0 else e1 + (n - 1.0) * (e2 - e1) for n in n_grid]
    assert [float(v) for v in reference["reference_linear"]] == pytest.approx(expected, abs=1e-8)


def test_a_fractional_curve_refuses_a_configuration_without_fractional_occupation():
    """The pyscfad backend and the one-shot mode give the one-electron energy
    whatever the count: the curve refuses them before any run; the
    comparator resolver takes names or labels and refuses anything else."""
    pyscfad = dataclasses.replace(diagnostics.diagnostic_solver_config(),
                                  backend=SolverBackend.PYSCFAD)
    with pytest.raises(ValueError, match="manual"):
        diagnostics.fractional_charge_curve({}, [0.5], pyscfad)
    oneshot = SolverConfig(backend=SolverBackend.MANUAL, mode=SolverMode.ONESHOT)
    with pytest.raises(ValueError, match="FULL"):
        diagnostics.fractional_charge_curve({}, [0.5], oneshot)
    with pytest.raises(KeyError, match="scan"):
        diagnostics.comparator_dissociation([1.4], ["scan"])
    assert diagnostics.h2_solver_config().backend == SolverBackend.PYSCFAD
    assert diagnostics.fractional_solver_config().max_cycles == diagnostics.FRACTIONAL_CYCLES


def test_the_grid_ends_within_the_range_and_the_values_are_checked_first(tmp_path):
    """grid(): the last multiple of the step within the end (2.1 is not a
    point of (0, 2, 0.3); 1.95 ends (0, 2, 0.15)); a step that is not positive
    or exceeds the range, or an end or step that is not finite, is refused
    by name. compute_diagnostics refuses an electron number beyond 2 and a
    bond length that is not positive or not finite before any run."""
    nan = float("nan")
    assert diagnostics.grid(0, 2, 0.3) == [0.0, 0.3, 0.6, 0.9, 1.2, 1.5, 1.8]
    assert diagnostics.grid(0, 2, 0.15)[-1] == 1.95
    assert diagnostics.grid(0, 2, 0.5) == [0.0, 0.5, 1.0, 1.5, 2.0]
    with pytest.raises(ValueError):
        diagnostics.grid(0, 2, 0.0)
    with pytest.raises(ValueError):
        diagnostics.grid(0, 2, 3.0)
    for lo, hi, step in ((0.0, 2.0, nan), (nan, 2.0, 0.5), (0.0, float("inf"), 0.5)):
        with pytest.raises(ValueError, match="finite"):
            diagnostics.grid(lo, hi, step)
    run = tmp_path / "run"
    run.mkdir()
    (run / "manifest.json").write_text(json.dumps({
        "width": 4, "networks": [{"label": "S_deep_3x16", "index": 0, "arch_name": "deep_3x16"}]}))
    with pytest.raises(ValueError, match="2 electrons"):
        diagnostics.compute_diagnostics(run, n_values=[0.0, 2.5], r_values=[1.4],
                                        models={"S_deep_3x16": (None, None)})
    with pytest.raises(ValueError, match="positive"):
        diagnostics.compute_diagnostics(run, n_values=[1.0], r_values=[0.0],
                                        models={"S_deep_3x16": (None, None)})
    with pytest.raises(ValueError, match="finite"):
        diagnostics.compute_diagnostics(run, n_values=[1.0], r_values=[nan],
                                        models={"S_deep_3x16": (None, None)})
    manifest = {"networks": [{"label": "S_a", "index": 0, "arch_name": "deep_3x16"},
                             {"label": "S_b", "index": 1}]}
    with pytest.raises(ValueError, match="S_b"):
        diagnostics.select_networks(manifest)


@pytest.mark.slow
def test_the_h2_curve_through_the_pyscfad_configuration_is_pyscfs_pbe():
    """The PBE parent through dissociation_curve's default configuration
    (the pyscfad backend with full integrals) at 1.4 and 6.0 bohr equals
    pyscf's restricted PBE run here (the pinned cutoff, conv_tol 1e-11) to
    1e-7 Ha; measured 1.8e-12 at 1.4 bohr and 1.1e-8 at 6.0."""
    model = _pbe_parent_model()
    curve = diagnostics.dissociation_curve(
        {"PBE parent": (model, _spec(_NoDescriptorArch()))}, [1.4, 6.0])
    assert all(curve["PBE parent"]["converged"])
    for r, energy in zip((1.4, 6.0), curve["PBE parent"]["E"]):
        assert energy == pytest.approx(_rks(r, "pbe"), abs=1e-7), (r, energy)


def test_load_networks_reads_a_checkpoint_written_the_way_the_job_writes_one(tmp_path):
    """A run directory with a spec and a model checkpoint written through
    the training stage's writer (as slim16_eval.py prepare does): the model
    load_networks returns gives the one-shot energy of the original on the
    H atom's record."""
    from xcquinox.pipeline.cluster.materialize import write_spec_atomic
    from xcquinox.pipeline.config import TrainingSpec
    from xcquinox.pipeline.oneshot import fixed_density_total_energy
    from xcquinox.pipeline.train import save_trained_checkpoint
    arch, model = _clone("deep_3x16", seed=3)
    run = tmp_path / "run"
    (run / "specs").mkdir(parents=True)
    checkpoint_dir = run / "checkpoints" / "spec_0000"
    checkpoint_dir.mkdir(parents=True)
    save_trained_checkpoint(str(checkpoint_dir / "model.eqx"), model, arch)
    spec = TrainingSpec(arch=arch, molecules=(), targets=(), atom_energies=(),
                        loss_name="slim16_eval", checkpoint_dir=str(checkpoint_dir),
                        solver_config=diagnostics.diagnostic_solver_config())
    write_spec_atomic(spec, str(run / "specs" / "spec_0000.spec"))
    manifest = {"width": 4, "n_specs": 1,
                "networks": [{"label": "S_deep_3x16", "index": 0, "arch_name": "deep_3x16"}]}
    (run / "manifest.json").write_text(json.dumps(manifest))
    loaded = diagnostics.load_networks(run, diagnostics.select_networks(manifest))
    assert list(loaded) == ["S_deep_3x16"]
    model_loaded, spec_loaded = loaded["S_deep_3x16"]
    assert spec_loaded.arch.name == "deep_3x16"
    record = precompute_fixed_density_data(diagnostics.h_atom_spec(), required_keys=(),
                                           descriptors=model.descriptors)
    assert float(fixed_density_total_energy(model_loaded, record)) == \
        float(fixed_density_total_energy(model, record))
