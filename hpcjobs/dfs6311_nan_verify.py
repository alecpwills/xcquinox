"""Production-scale verification of the corrected V_xc on the meta-GGA path.

Three legs, all at the DFS-parity basis 6-311++G(3df,2pd) with PRETRAINED
checkpoints from the live run -- the configuration the 2026-07-04 alpha-tail
NaN appeared in, and the one a workstation cannot fully reach (grid 3, full
species set). An earlier version of this file ran a CO/C/O padding check on an
architecture with NO descriptors, so ``compute_alpha`` was never called and the
job verified nothing about the meta-GGA path; this version loads the actual
``deep_mgga_3x16`` / ``deep_rung35_mgga_3x16`` pretrain checkpoints.

LEG 1 (assertion): the FULL-SCF training gradient at 25 cycles with
``scf_grad_checkpoint=True`` is finite at every leaf and nonzero, per
architecture x species. This is the alpha-tail regression at production scale:
the tail-gradient freeze was removed from ``compute_alpha`` (HISTORY
2026-08-06), the clip bounds the VALUE only, and the counter-evidence on record
is a ``max|d alpha/d sigma|`` of 2.20e31 on Li at this basis -- finite training
gradients here settle that the removal stands; a non-finite leaf means the
fallback (a smooth damping applied to the ENERGY, never a gradient freeze)
must be designed instead.

LEG 2 (measurement): converged-energy shift between the corrected potential
and the prior assembly (feature-response term absent), per species, open-shell
included. Locally, with pretrained weights at def2-svp, the shift was 0.03-0.26
kcal/mol against the ~22 kcal/mol meta-GGA gap; this measures the same number
at the production basis and grid. The prior assembly is emulated by disabling
the descriptor-response predicate, which removes exactly the term the fix
added.

LEG 3 (measurement): SCF convergence behaviour, corrected vs prior -- cycles
run, converged flag, and the energy at increasing cycle caps (a coarse
oscillation profile). A more correct potential that converges worse would be
worth knowing before any of the 55 affected cells is re-run.

BLAS NOTE: the sbatch wrapper pins OMP/MKL/OPENBLAS to ONE thread. Measured on
the workstation: at 25 unrolled cycles an UNCONVERGED diffuse-basis atom's
energy and gradient magnitude wander between identical multithreaded runs
(60 mHa / eight orders in |g|max across four repeats), while single-threaded
runs are bit-identical. A threaded single run is therefore not evidence;
single-threaded runs are reproducible and comparable.

Usage:
    python hpcjobs/dfs6311_nan_verify.py <run_dir> [--grid-level 3]
        [--cycles 25] [--out <json path>]

``<run_dir>`` must carry ``pretrain/<arch>/{xnet.eqx,cnet.eqx}``.
"""
import os

os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import dataclasses
import json
import sys
import time

import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from xcquinox.alec.config import ARCHITECTURES, MoleculeSpec
from xcquinox.alec.data import precompute_fixed_density_data
from xcquinox.alec.models import AlecGGAModel
from xcquinox.alec.solver import (FeaturePolicy, SolverBackend, SolverConfig,
                                  SolverMode, run_scf)
import xcquinox.alec.oneshot as oneshot

BASIS = "6-311++G(3df,2pd)"
HARTREE_TO_KCAL = 627.5094740631

ARCHS = ("deep_mgga_3x16", "deep_rung35_mgga_3x16")

# The bh76:HLi failure species plus one closed-shell molecule for contrast.
SPECIES = (
    ("H", "H 0 0 0", 1, (("H", 1),)),
    ("Li", "Li 0 0 0", 1, (("Li", 1),)),
    ("LiH", "Li 0 0 0; H 0 0 1.5949", 0, (("Li", 1), ("H", 1))),
    ("H2O", "O 0 0 0.117; H 0 0.757 -0.469; H 0 -0.757 -0.469", 0,
     (("O", 1), ("H", 2))),
)


def load_pretrained(run_dir, arch_name):
    """The registry architecture with the run's pretrain weights loaded.

    Mirrors the sweep's own construction (registry entry + the sweep-level
    polarized override; every live spec and pretrain manifest carries
    ``use_polarized_correlation=True``). Raises rather than silently
    evaluating a random initialization."""
    arch = dataclasses.replace(ARCHITECTURES[arch_name],
                               use_polarized_correlation=True)
    model = AlecGGAModel.from_arch(arch, seed=0)
    d = os.path.join(run_dir, "pretrain", arch_name)
    for f in ("xnet.eqx", "cnet.eqx"):
        if not os.path.isfile(os.path.join(d, f)):
            raise FileNotFoundError(f"pretrain checkpoint missing: {d}/{f}")
    model = eqx.tree_at(
        lambda m: m.xnet, model,
        eqx.tree_deserialise_leaves(os.path.join(d, "xnet.eqx"), model.xnet))
    model = eqx.tree_at(
        lambda m: m.cnet, model,
        eqx.tree_deserialise_leaves(os.path.join(d, "cnet.eqx"), model.cnet))
    return model


def mol_data_for(model, name, atom, spin, comp, grid_level):
    # cderi, not eri: the production sweep runs density-fitted Coulomb
    # (resolved_config inputs.density_fit: true), so the shift this job
    # measures must be measured on the same backend -- and the exact 4-index
    # ERI would also be the dominant memory term at 6-311++G(3df,2pd).
    keys = tuple(sorted({k for d in model.descriptors
                         for k in d.required_mol_keys} | {"cderi"}))
    spec = MoleculeSpec(name=name, atom=atom, basis=BASIS, charge=0,
                        spin=spin, atom_composition=comp,
                        grid_level=grid_level)
    return precompute_fixed_density_data(spec, descriptors=model.descriptors,
                                         required_keys=keys)


def cfg_for(cycles, checkpoint=True):
    # density_fit=True with auxbasis auto-selected: production parity (the 55
    # affected cells train under DF, so the corrected-vs-prior shift is
    # measured on the Coulomb backend they actually use).
    return SolverConfig(backend=SolverBackend.MANUAL, mode=SolverMode.FULL,
                        max_cycles=cycles, conv_tol=1e-12,
                        feature_policy=FeaturePolicy.REASSEMBLE,
                        scf_grad_checkpoint=checkpoint,
                        density_fit=True)


class prior_potential:
    """Context manager emulating the assembly WITHOUT the feature-response
    term: the descriptor-response predicate is disabled, which removes exactly
    the ``sum_g w_g (de/df)_g . df_g/dP`` contribution the fix added. The
    manual solver re-imports the predicate from ``oneshot`` on every call, so
    swapping the module attribute takes effect immediately and reversibly."""

    def __enter__(self):
        self._orig = oneshot.has_dm_dependent_descriptor
        oneshot.has_dm_dependent_descriptor = lambda model: False
        return self

    def __exit__(self, *exc):
        oneshot.has_dm_dependent_descriptor = self._orig
        return False


def leg1_gradient_finiteness(run_dir, grid_level, cycles, report):
    print(f"[leg1] training-gradient finiteness, {cycles} cycles, "
          f"grid {grid_level}", flush=True)
    ok = True
    for arch_name in ARCHS:
        model = load_pretrained(run_dir, arch_name)
        for name, atom, spin, comp in SPECIES:
            t0 = time.time()
            md = mol_data_for(model, name, atom, spin, comp, grid_level)
            rho_min = float(np.min(np.asarray(md["rho_grid"])))
            cfg = cfg_for(cycles)
            val, grads = eqx.filter_value_and_grad(
                lambda m: run_scf(cfg, m, md).total_energy)(model)
            leaves = jax.tree_util.tree_leaves(
                eqx.filter(grads, eqx.is_inexact_array))
            finite = all(bool(jnp.all(jnp.isfinite(l))) for l in leaves)
            nonzero = any(bool(jnp.any(l != 0.0)) for l in leaves)
            gmax = max(float(jnp.max(jnp.abs(l))) for l in leaves)
            row = dict(arch=arch_name, species=name, rho_min=rho_min,
                       energy=float(val), grad_finite=finite,
                       grad_nonzero=nonzero, gmax=gmax,
                       seconds=round(time.time() - t0, 1))
            report["leg1"].append(row)
            ok = ok and finite and nonzero
            print(f"[leg1]   {arch_name:<24}{name:<5} rho_min={rho_min:.3e} "
                  f"E={float(val):+.8f} finite={finite} nonzero={nonzero} "
                  f"|g|max={gmax:.3e} ({row['seconds']}s)", flush=True)
    print(f"[leg1] VERDICT: {'ALL FINITE AND NONZERO' if ok else 'FAILURE'}",
          flush=True)
    return ok


def leg2_potential_effect(run_dir, grid_level, cycles, report):
    print("[leg2] corrected-vs-prior converged-energy shift", flush=True)
    for arch_name in ARCHS:
        model = load_pretrained(run_dir, arch_name)
        for name, atom, spin, comp in SPECIES:
            md = mol_data_for(model, name, atom, spin, comp, grid_level)
            cfg = cfg_for(cycles)
            e_corr = float(run_scf(cfg, model, md).total_energy)
            with prior_potential():
                e_prior = float(run_scf(cfg, model, md).total_energy)
            shift = (e_corr - e_prior) * HARTREE_TO_KCAL
            report["leg2"].append(dict(
                arch=arch_name, species=name, e_corrected=e_corr,
                e_prior=e_prior, shift_kcalmol=shift))
            print(f"[leg2]   {arch_name:<24}{name:<5} "
                  f"E_corr={e_corr:+.8f} E_prior={e_prior:+.8f} "
                  f"shift={shift:+.4e} kcal/mol", flush=True)


def leg3_convergence_behaviour(run_dir, grid_level, report,
                               caps=(5, 10, 15, 20, 25)):
    print("[leg3] convergence behaviour, corrected vs prior", flush=True)
    for arch_name in ARCHS:
        model = load_pretrained(run_dir, arch_name)
        for name, atom, spin, comp in SPECIES:
            md = mol_data_for(model, name, atom, spin, comp, grid_level)
            for tag in ("corrected", "prior"):
                energies, cycles_run, converged = [], [], []

                def _sweep():
                    for c in caps:
                        r = run_scf(cfg_for(c), model, md)
                        energies.append(float(r.total_energy))
                        cycles_run.append(int(r.cycles_run))
                        converged.append(bool(r.converged))

                if tag == "corrected":
                    _sweep()
                else:
                    with prior_potential():
                        _sweep()
                steps = [abs(energies[i + 1] - energies[i])
                         for i in range(len(energies) - 1)]
                report["leg3"].append(dict(
                    arch=arch_name, species=name, variant=tag, caps=list(caps),
                    energies=energies, cycles_run=cycles_run,
                    converged=converged, max_late_step=max(steps[-2:])))
                print(f"[leg3]   {arch_name:<24}{name:<5} {tag:<10} "
                      f"E(caps)={['%+.6f' % e for e in energies]} "
                      f"conv={converged[-1]} "
                      f"late-step={max(steps[-2:]):.3e}", flush=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("run_dir")
    p.add_argument("--grid-level", type=int, default=3)
    p.add_argument("--cycles", type=int, default=25)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    report = dict(basis=BASIS, grid_level=args.grid_level, cycles=args.cycles,
                  archs=list(ARCHS), leg1=[], leg2=[], leg3=[])
    ok = leg1_gradient_finiteness(args.run_dir, args.grid_level, args.cycles,
                                  report)
    leg2_potential_effect(args.run_dir, args.grid_level, args.cycles, report)
    leg3_convergence_behaviour(args.run_dir, args.grid_level, report)

    out = args.out or os.path.join(
        args.run_dir, f"nan_verify_report_grid{args.grid_level}.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=1)
    print(f"[done] report -> {out}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
