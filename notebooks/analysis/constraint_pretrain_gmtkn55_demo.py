#!/usr/bin/env python
"""Demo: physical constraints + pretraining for xcquinox.alec XC networks.

Shows how physical constraints and pretraining each pull a randomly-initialized
exchange-correlation network toward correct physics, evaluated against GMTKN55
references — and why the choice of metric matters.

Scenarios per constraint level (x-axis):
  - "unconstrained": a truly raw network (built-in Lieb-Oxford squash disabled,
    lob_lim=None, no constraints) — F_x can run to ~17.
  - "+LO(x)" -> "+LO+UEG(x)" -> "+LO+UEG+NNc(c)": increasing physical constraints,
    enforced INTRINSICALLY by the networks (so the same constrained functional is
    used in pretraining, optimization, and eval).

Two series: random-init (multi-seed: mean bar + worst-case whisker) and
constraint-aware-pretrained (single seed; library ``alec.run_pretrain``).

All archs use ``use_polarized_correlation=True``, so the UKS energy path uses
the spin-polarized PW92c correlation baseline with the real zeta for open-shell
species (atoms, radicals) — not the zeta=0 unpolarized baseline. Pretrain data
is spin-resolved (libxc spin=1 targets + per-point ``zeta_all``).

THREE metrics (three panels), because the constraint benefit is metric-dependent:
  1. BH76 reaction-energy MAE vs GMTKN55-BH76RC (Probe-C, 6 reactions).
     Reaction energies are balanced (sum of coeffs = 0), so they CANCEL the large
     systematic per-species XC error that constraints reduce — the effect looks
     small here.
  2. Per-species |E_nn - E_pbe| MAE (deviation from the PBE total energy). This is
     where the constraint benefit is sizable: constraints cut the worst-case and
     variance of a random network dramatically. (No GMTKN55 absolute-energy
     reference exists — PBE is the baseline.)
  3. Atomization-energy MAE vs GMTKN55 W4-11 (parsed from the local clone). W4-11
     writes each atomization as a molecule->atoms reaction, so it reuses the same
     reaction-energy scorer; it partially cancels, so the effect is intermediate.

This is a DEMONSTRATION (small basis, coarse grid, standard 1000 pretrain steps),
not a benchmark. Requires local reference data: g2_97.traj (via eval_probes) and the
GMTKN55 clone at scripts/script_data/gmtkn55/ (W4-11 subset).

Run::

    python notebooks/analysis/constraint_pretrain_gmtkn55_demo.py

Prints three result tables and writes a 3-panel figure next to this file
(``constraint_pretrain_gmtkn55_demo.png``).
"""
from __future__ import annotations

import os
import re
import time

import numpy as np
import equinox as eqx

import matplotlib
matplotlib.use("Agg")  # headless: write a PNG, never open a window
import matplotlib.pyplot as plt  # noqa: E402

from ase.io import read as ase_read  # noqa: E402
from pyscf import gto, dft  # noqa: E402

import xcquinox.alec as alec  # noqa: E402
from xcquinox.alec.config import ArchitectureConfig  # noqa: E402
from xcquinox.alec.models import AlecGGAModel  # noqa: E402
from xcquinox.alec.networks import (  # noqa: E402
    create_network_pair, AlecGGA_XNet, AlecGGA_CNet,
)
from xcquinox.alec.eval_probes import build_probe_pool  # noqa: E402
from xcquinox.alec.dfs_pool import make_atom_atoms  # noqa: E402
from xcquinox.alec.cluster.spec_builder import atoms_to_mol_spec  # noqa: E402


# --- demo knobs (kept small for a quick run) -------------------------------
KCAL_PER_HA = 627.5094740631
BASIS = "def2-svp"
GRID_LEVEL = 1
SEED = 0
N_SEEDS = 16                 # random-init seeds (cheap: mol_data is cached)
DEPTH, NODES = 3, 16
PRETRAIN_N_STEPS = 1000       # standard pretrain schedule
# (symbol, PySCF 2S spin) — atoms whose PBE/LDA grid enhancement factors seed
# the pretraining targets.
PRETRAIN_ATOMS = (("H", 1), ("He", 0), ("N", 3), ("O", 2))
PROBE = "probe_c_bh76_transfer"
# Small, closed-shell W4-11 molecules for the atomization-energy metric.
W411_MOLECULES = ("h2", "h2o", "ch4", "nh3", "co", "n2", "co2", "hf", "c2h2", "c2h4")

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))
GMTKN55_DIR = os.path.join(_REPO, "scripts", "script_data", "gmtkn55")
W411_DIR = os.path.join(GMTKN55_DIR, "W4-11")
OUTDIR = os.path.join(_HERE, "_constraint_demo_work")
# Step- and baseline-tagged so runs never clobber each other: the original
# 150-step figure (``constraint_pretrain_gmtkn55_demo.png``) and the unpolarized
# 1000-step figure (``..._pretrain1000step.png``) are both preserved; this
# spin-polarized run writes ``..._pretrain{N}step_polc.png``.
PLOT_PATH = os.path.join(
    _HERE, f"constraint_pretrain_gmtkn55_demo_pretrain{PRETRAIN_N_STEPS}step_polc.png")


# ---------------------------------------------------------------------------
# constraint progression + model construction
# ---------------------------------------------------------------------------

def make_constraint_levels():
    """Increasing-constraint progression as ``(label, spec)`` where ``spec`` is
    ``None`` for the truly-unconstrained baseline, else ``(x_constraints,
    c_constraints)``. The unconstrained level disables the built-in Lieb-Oxford
    squash too — otherwise "no constraints" would still be LO-bounded."""
    return [
        ("unconstrained", None),
        ("+LO(x)", (("lieb_oxford",), ())),
        ("+LO+UEG(x)", (("lieb_oxford", "ueg_limit"), ())),
        ("+LO+UEG+NNc(c)", (("lieb_oxford", "ueg_limit"), ("non_negative_correlation",))),
    ]


def build_arch(label, x_constraints, c_constraints):
    """Build a no-descriptor GGA arch with the given constraint set (constraints
    are baked into the networks by create_network_pair).

    ``use_polarized_correlation=True`` makes the cnet spin-polarization-aware, so
    the UKS energy path (``split_exc_energy_uks``) uses the spin-polarized PW92c
    correlation baseline with the real zeta for open-shell species."""
    safe = (
        label.replace("+", "").replace("(", "").replace(")", "").strip().lower()
        or "base"
    )
    return ArchitectureConfig.from_spec(
        safe, DEPTH, NODES,
        x_constraints=list(x_constraints), c_constraints=list(c_constraints),
        use_polarized_correlation=True)


def build_random_model(spec, seed):
    """Random-init model for a level. ``spec is None`` -> TRULY unconstrained
    (lob_lim=None, constraints=()); else a constrained arch. Both use the
    spin-polarization-aware cnet (polarized PW92c baseline)."""
    if spec is None:
        xnet = AlecGGA_XNet(n_extra_features=0, depth=DEPTH, nodes=NODES,
                            seed=seed, lob_lim=None)
        cnet = AlecGGA_CNet(n_extra_features=0, depth=DEPTH, nodes=NODES,
                            seed=seed + 1, lob_lim=None, use_spin_polarization=True)
        base = ArchitectureConfig.from_spec(
            "base", DEPTH, NODES, use_polarized_correlation=True)
        return AlecGGAModel.from_arch(base, xnet=xnet, cnet=cnet)
    x_constraints, c_constraints = spec
    return AlecGGAModel.from_arch(
        build_arch("lvl", x_constraints, c_constraints), seed=seed)


# ---------------------------------------------------------------------------
# evaluation pools
# ---------------------------------------------------------------------------

def _mol_spec_from_atoms(atoms, name):
    refs_dir = os.path.join(OUTDIR, "_no_external_refs")  # intentionally absent
    return atoms_to_mol_spec(atoms, basis=BASIS, grid_level=GRID_LEVEL,
                             external_refs_dir=refs_dir, name=name)


def build_bh76_pool():
    """``(mol_specs, reactions)`` for GMTKN55-BH76RC Probe-C (keyed by Hill formula)."""
    pool = build_probe_pool(PROBE)
    mol_specs = {a.get_chemical_formula(): _mol_spec_from_atoms(a, a.get_chemical_formula())
                 for a in pool["molecules"]}
    needed = set()
    for rxn in pool["reactions"]:
        needed.update(rxn["reactants"]); needed.update(rxn["products"])
    missing = needed - set(mol_specs)
    if missing:
        raise RuntimeError(f"BH76 pool missing species: {sorted(missing)}")
    return mol_specs, list(pool["reactions"])


def build_w411_ae_pool():
    """``(mol_specs, reactions)`` for GMTKN55 W4-11 atomization energies.

    Parses the W4-11 ``.res`` tmer file (e.g. ``$tmer {h2,h}/$f x -1 2 $w 109.493``
    => 2*E(H) - E(H2) = 109.493 kcal/mol) for the curated closed-shell molecules.
    Each atomization is a molecule->atoms reaction, scored by ``reaction_energy_mae``.
    Molecule geometries come from the clone's struc.xyz; atom species (with NIST
    ground-state spins) from make_atom_atoms. References are read straight from the
    GMTKN55 clone — no transcription/fabrication."""
    res_path = os.path.join(W411_DIR, ".res")
    if not os.path.isfile(res_path):
        raise RuntimeError(
            f"GMTKN55 W4-11 not found at {W411_DIR} (.res missing). The demo needs "
            f"the local GMTKN55 clone at {GMTKN55_DIR}.")
    with open(res_path) as f:
        text = f.read()
    # $tmer {csv}/$f x <int coeffs...> $w <ref>
    pat = re.compile(r"\$tmer\s+\{([^}]+)\}/\$f\s+x\s+([-\d\s]+?)\s+\$w\s+(-?[\d.]+)")
    reactions, mol_specs = [], {}
    wanted = set(W411_MOLECULES)
    for m in pat.finditer(text):
        species = [s.strip() for s in m.group(1).split(",")]
        coeffs = [int(c) for c in m.group(2).split()]
        ref = float(m.group(3))
        mol = species[0]
        if mol not in wanted or len(coeffs) != len(species):
            continue
        reactions.append({
            "name": f"AE_{mol}", "reactants": list(species), "products": [],
            "coeffs": coeffs, "reaction_energy_ref": ref,
        })
        # molecule geometry from the clone (closed-shell -> spin 0)
        if mol not in mol_specs:
            xyz = os.path.join(W411_DIR, mol, "struc.xyz")
            if not os.path.isfile(xyz):
                raise RuntimeError(f"W4-11 geometry missing: {xyz}")
            a = ase_read(xyz)
            a.info.update(name=mol, charge=0, spin=0)
            mol_specs[mol] = _mol_spec_from_atoms(a, mol)
        # atom species via NIST ground-state spins
        for sym_tok in species[1:]:
            if sym_tok not in mol_specs:
                at = make_atom_atoms(sym_tok.capitalize())
                at.info.update(name=sym_tok, charge=0)
                mol_specs[sym_tok] = _mol_spec_from_atoms(at, sym_tok)
    if not reactions:
        raise RuntimeError("No W4-11 reactions parsed for the curated molecule set.")
    return mol_specs, reactions


def precompute_all(mol_specs):
    """Run the (memoized) PBE SCF + grid precompute once per unique species."""
    out = {}
    for i, (name, spec) in enumerate(mol_specs.items(), 1):
        print(f"  [precompute {i}/{len(mol_specs)}] {name} ...", flush=True)
        out[name] = alec.precompute_fixed_density_data(spec, descriptors=())
    return out


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def species_energies(model, mol_data_by_name):
    """Per-species total energy (Ha). ``model is None`` -> PBE baseline."""
    out = {}
    for name, md in mol_data_by_name.items():
        out[name] = (float(md["E_pbe"]) if model is None
                     else float(alec.fixed_density_total_energy(model, md)))
    return out


def reaction_energy_mae(energies_by_name, reactions):
    """MAE (kcal/mol) of predicted reaction energy vs reference. Used for BOTH
    BH76 reaction energies and W4-11 atomization energies (same dict shape).

    ``dE = sum_i coeff_i * E(species_i)`` (Ha) over reactants-then-products in
    ``coeffs`` order, to kcal/mol, vs ``reaction_energy_ref``. Pure — unit-tested."""
    errs = []
    for rxn in reactions:
        names = list(rxn["reactants"]) + list(rxn["products"])
        e_ha = sum(c * energies_by_name[n] for c, n in zip(rxn["coeffs"], names))
        errs.append(abs(e_ha * KCAL_PER_HA - rxn["reaction_energy_ref"]))
    return float(np.mean(errs))


def pbe_total_energy_dev_mae(energies_by_name, mol_data_by_name):
    """Mean over species of |E_nn - E_pbe| (kcal/mol) — direct XC-functional
    quality, the metric on which the constraint benefit is sizable. Pure —
    unit-tested. (E_pbe read from each species' precomputed mol_data.)"""
    errs = [abs(energies_by_name[n] - float(mol_data_by_name[n]["E_pbe"])) * KCAL_PER_HA
            for n in energies_by_name]
    return float(np.mean(errs))


# metric registry: name -> (callable(energies, ctx), pretty label, has_pbe_baseline)
def compute_metrics(model, mol_data, bh76_rxns, w411_rxns):
    e = species_energies(model, mol_data)
    return {
        "bh76": reaction_energy_mae(e, bh76_rxns),
        "pbe_dev": pbe_total_energy_dev_mae(e, mol_data),
        "w411_ae": reaction_energy_mae(e, w411_rxns),
    }


# ---------------------------------------------------------------------------
# pretraining (library; constraint-aware via the networks)
# ---------------------------------------------------------------------------

def generate_pretrain_data(data_dir):
    """Write ``<data_dir>/pretrain_data.npz`` with rho_all/sigma_all/Fx_all/Fc_all
    AND zeta_all, from a few atoms on a coarse grid. Fx/Fc stored as F-1.

    SPIN-RESOLVED targets (ported from notebooks/_build_step6_notebook.py): for
    open-shell atoms the PBE/LDA enhancement factors are evaluated with libxc
    ``spin=1`` on the spin-resolved density (the spin=0 total-density call is
    wrong for open-shell — PBE 1996 §III spin-scaling), and ``zeta_all`` carries
    the per-grid-point spin polarization so the polarized cnet is pretrained on
    the real zeta rather than a zeta=0 warm-start."""
    rho_l, sig_l, fx_l, fc_l, zeta_l, w_l = [], [], [], [], [], []
    for symbol, spin in PRETRAIN_ATOMS:
        mol = gto.M(atom=f"{symbol} 0 0 0", basis=BASIS, charge=0, spin=spin, verbose=0)
        mf = dft.UKS(mol) if spin else dft.RKS(mol)
        mf.xc = "pbe"
        mf.grids.level = GRID_LEVEL
        mf.kernel()
        ao = mf._numint.eval_ao(mol, mf.grids.coords, deriv=1)
        dm_ab = mf.make_rdm1()
        if dm_ab.ndim == 3:  # open-shell (UKS): spin-resolve, libxc spin=1
            rho_a_gga = mf._numint.eval_rho(mol, ao, dm_ab[0], xctype="GGA", hermi=True)
            rho_b_gga = mf._numint.eval_rho(mol, ao, dm_ab[1], xctype="GGA", hermi=True)
            rho_gga_uks = np.stack([rho_a_gga, rho_b_gga], axis=0)
            rho_a, rho_b = rho_a_gga[0], rho_b_gga[0]
            rho = rho_a + rho_b
            nabla_total = rho_a_gga[1:4] + rho_b_gga[1:4]
            sigma = (nabla_total ** 2).sum(axis=0)
            zeta = (rho_a - rho_b) / np.maximum(rho, 1e-300)
            ex_pbe = mf._numint.eval_xc("PBE,", rho_gga_uks, spin=1)[0]
            ec_pbe = mf._numint.eval_xc(",PBE", rho_gga_uks, spin=1)[0]
            ex_lda = mf._numint.eval_xc("LDA_X,", (rho_a, rho_b), spin=1)[0]
            ec_lda = mf._numint.eval_xc(",LDA_C_PW", (rho_a, rho_b), spin=1)[0]
        else:  # closed-shell (RKS): zeta = 0, spin=0 calls
            rho_gga = mf._numint.eval_rho(mol, ao, dm_ab, xctype="GGA", hermi=True)
            rho = rho_gga[0]
            sigma = rho_gga[1] ** 2 + rho_gga[2] ** 2 + rho_gga[3] ** 2
            zeta = np.zeros_like(rho)
            ex_pbe = mf._numint.eval_xc("PBE,", rho_gga, spin=0)[0]
            ec_pbe = mf._numint.eval_xc(",PBE", rho_gga, spin=0)[0]
            ex_lda = mf._numint.eval_xc("LDA_X,", rho, spin=0)[0]
            ec_lda = mf._numint.eval_xc(",LDA_C_PW", rho, spin=0)[0]
        ex_safe = np.where(np.abs(ex_lda) > 1e-12, ex_lda, 1e-12)
        ec_safe = np.where(np.abs(ec_lda) > 1e-12, ec_lda, 1e-12)
        fx = np.clip(ex_pbe / ex_safe - 1.0, -5.0, 5.0)
        fc = np.clip(ec_pbe / ec_safe - 1.0, -5.0, 5.0)
        valid = rho > 1e-10
        rho_l.append(rho[valid]); sig_l.append(sigma[valid])
        fx_l.append(fx[valid]); fc_l.append(fc[valid]); zeta_l.append(zeta[valid])
        # Becke-Lebedev quadrature weights dr_i, so loss_weighting="integration"
        # is the TRUE quadrature-weighted loss (not the magnitude-only fallback).
        w_l.append(np.asarray(mf.grids.weights)[valid])
    os.makedirs(data_dir, exist_ok=True)
    # Polarized filename: the demo's archs set use_polarized_correlation=True, so
    # run_pretrain selects pretrain_data_polarized.npz (carrying zeta_all).
    # weights_all is ADDITIVE — existing consumers ignore extra npz keys.
    np.savez(os.path.join(data_dir, "pretrain_data_polarized.npz"),
             rho_all=np.concatenate(rho_l), sigma_all=np.concatenate(sig_l),
             Fx_all=np.concatenate(fx_l), Fc_all=np.concatenate(fc_l),
             zeta_all=np.concatenate(zeta_l), weights_all=np.concatenate(w_l))


def pretrain_and_load(spec, data_dir, ckpt_dir):
    """Library ``run_pretrain`` on a constrained arch (constraint-aware), then load
    the trained networks. Only valid for constrained levels (spec is not None)."""
    x_constraints, c_constraints = spec
    arch = build_arch("lvl", x_constraints, c_constraints)
    alec.run_pretrain(alec.PretrainSpec(
        arch=arch, data_dir=data_dir, checkpoint_dir=ckpt_dir,
        n_steps=PRETRAIN_N_STEPS, loss_weighting="unweighted", seed=SEED))
    xnet_skel, cnet_skel = create_network_pair(arch, seed=SEED)
    xnet = eqx.tree_deserialise_leaves(os.path.join(ckpt_dir, "xnet.eqx"), xnet_skel)
    cnet = eqx.tree_deserialise_leaves(os.path.join(ckpt_dir, "cnet.eqx"), cnet_skel)
    return AlecGGAModel.from_arch(arch, xnet=xnet, cnet=cnet, seed=SEED)


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------

_METRICS = [
    ("bh76", "BH76 reaction-energy MAE\nvs GMTKN55-BH76RC (kcal/mol)", True),
    ("pbe_dev", "per-species |E_nn - E_pbe| MAE\n(kcal/mol; deviation from PBE)", False),
    ("w411_ae", "atomization-energy MAE\nvs GMTKN55 W4-11 (kcal/mol)", True),
]


def main():
    t0 = time.time()
    os.makedirs(OUTDIR, exist_ok=True)

    print("[1/4] Building eval pools (GMTKN55 BH76-RC + W4-11 atomization) ...", flush=True)
    bh76_specs, bh76_rxns = build_bh76_pool()
    w411_specs, w411_rxns = build_w411_ae_pool()
    mol_specs = {**bh76_specs, **w411_specs}
    print(f"      {len(mol_specs)} species; {len(bh76_rxns)} BH76 reactions, "
          f"{len(w411_rxns)} W4-11 atomizations", flush=True)

    print("[2/4] Precomputing PBE SCF + grids (once per species) ...", flush=True)
    mol_data = precompute_all(mol_specs)

    print("[3/4] Generating pretrain data ...", flush=True)
    data_dir = os.path.join(OUTDIR, "pretrain_data")
    generate_pretrain_data(data_dir)

    levels = make_constraint_levels()
    pbe = compute_metrics(None, mol_data, bh76_rxns, w411_rxns)

    print(f"[4/4] Random ({N_SEEDS} seeds) + pretrained per level ...", flush=True)
    # rand[level_label][metric] = dict(mean, max, std); pre[level_label][metric] = val|None
    rand, pre = {}, {}
    for label, spec in levels:
        per_metric = {k: [] for k, _, _ in _METRICS}
        for sd in range(N_SEEDS):
            m = compute_metrics(build_random_model(spec, sd), mol_data, bh76_rxns, w411_rxns)
            for k in per_metric:
                per_metric[k].append(m[k])
        rand[label] = {k: dict(mean=float(np.mean(v)), max=float(np.max(v)),
                               std=float(np.std(v))) for k, v in per_metric.items()}
        if spec is None:
            pre[label] = {k: None for k, _, _ in _METRICS}  # can't library-pretrain a raw net
        else:
            ckpt = os.path.join(OUTDIR, "pretrain", build_arch("lvl", *spec).name)
            pm = compute_metrics(pretrain_and_load(spec, data_dir, ckpt),
                                 mol_data, bh76_rxns, w411_rxns)
            pre[label] = pm
        print(f"      {label:16s} "
              f"bh76[r~{rand[label]['bh76']['mean']:.1f}] "
              f"pbe_dev[r~{rand[label]['pbe_dev']['mean']:.0f} max{rand[label]['pbe_dev']['max']:.0f}] "
              f"w411[r~{rand[label]['w411_ae']['mean']:.1f}]", flush=True)

    _print_tables(levels, pbe, rand, pre)
    _plot(levels, pbe, rand, pre)
    print(f"\nDone in {time.time() - t0:.1f}s. Plot -> {PLOT_PATH}", flush=True)


def _print_tables(levels, pbe, rand, pre):
    for key, label, has_pbe in _METRICS:
        print("\n" + "=" * 78)
        print(label.replace("\n", " ") + "  (lower is better)")
        print("=" * 78)
        print(f"{'level':18s} {'random mean':>12s} {'random max':>12s} "
              f"{'random std':>11s} {'pretrained':>12s}")
        print("-" * 78)
        if has_pbe:
            print(f"{'PBE baseline':18s} {pbe[key]:>12.2f} {'':>12s} {'':>11s} {'(n/a)':>12s}")
        for lbl, _spec in levels:
            r = rand[lbl][key]
            pv = pre[lbl][key]
            ps = "(n/a)" if pv is None else f"{pv:.2f}"
            print(f"{lbl:18s} {r['mean']:>12.2f} {r['max']:>12.2f} "
                  f"{r['std']:>11.2f} {ps:>12s}")
        print("=" * 78)


_PLOT_STYLE = {
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7,
    "axes.axisbelow": True, "figure.dpi": 120, "savefig.dpi": 150,
    "savefig.bbox": "tight",
}


def _plot(levels, pbe, rand, pre):
    labels = [lbl for lbl, _ in levels]
    x = np.arange(len(labels))
    w = 0.38
    with plt.rc_context(_PLOT_STYLE):
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
        for i, (ax, (key, ylab, has_pbe)) in enumerate(zip(axes, _METRICS)):
            first = (i == 0)
            means = np.array([rand[l][key]["mean"] for l in labels])
            maxes = np.array([rand[l][key]["max"] for l in labels])
            stds = np.array([rand[l][key]["std"] for l in labels])
            ax.bar(x - w / 2, means, w, color="#c0504d", zorder=2,
                   label="random init (mean)" if first else None)
            # faint upper whisker to the worst seed ...
            ax.errorbar(x - w / 2, means,
                        yerr=[np.zeros_like(means), np.maximum(maxes - means, 0)],
                        fmt="none", ecolor="#9a9a9a", elinewidth=1.0, capsize=6,
                        capthick=1.0, zorder=4,
                        label=f"worst of {N_SEEDS} seeds" if first else None)
            # ... and a bold ± std whisker across seeds.
            ax.errorbar(x - w / 2, means, yerr=stds, fmt="none", ecolor="#5a1714",
                        elinewidth=1.8, capsize=3, capthick=1.6, zorder=5,
                        label="± std (seeds)" if first else None)
            pre_vals = [pre[l][key] for l in labels]
            xp = [xi for xi, v in zip(x, pre_vals) if v is not None]
            yp = [v for v in pre_vals if v is not None]
            ax.bar(np.array(xp) + w / 2, yp, w, color="#4f81bd", zorder=3,
                   label="pretrained (constraint-aware)" if first else None)
            if has_pbe:
                ax.axhline(pbe[key], ls="--", color="k", lw=1.1,
                           label=f"PBE baseline ({pbe[key]:.1f})" if first else None)
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=18, ha="right")
            ax.grid(axis="y", alpha=0.3)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.margins(y=0.08)
            ax.set_title(ylab)
            if first:
                ax.legend(loc="upper left", framealpha=0.9)
        axes[0].set_ylabel("MAE (kcal/mol)")
        fig.suptitle("Physical constraints + pretraining vs GMTKN55 — "
                     "reaction energies cancel the per-species error that constraints reduce  "
                     f"(pretrain: {PRETRAIN_N_STEPS} steps, seed {SEED})",
                     fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.96))
        fig.savefig(PLOT_PATH)
        plt.close(fig)


if __name__ == "__main__":
    main()
