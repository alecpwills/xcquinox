"""Dissociation and fractional-charge diagnostics of a functional on the evaluation path.

Two curves where a semilocal functional fails by construction: the restricted
dissociation of H2, whose energy at large separation sits above twice the
atom's (the static-correlation error), and the energy of the H atom at a
fractional electron number, which bows below the straight line between the
integers (the delocalization error; the dissociation limit of H2+ is twice
the atom at half an electron). Each is computed for trained networks through
the evaluation path's solver, for the libxc comparators through pyscf, and
against the exact reference in the basis: CCSD converged tightly for the
two-electron systems (H2 and the H anion, where it is full CI), UHF for the
one-electron atom, and the straight lines between the integers for the
fractional curve.

The dissociation curve runs on the pyscfad backend with full integrals, the
converged channel's own backend: the manual solver's linear mixing does not
converge the stretched molecule (at 6 bohr it runs its whole budget and ends
40 kcal/mol above the converged energy with a density that is not
idempotent), while DIIS converges it in a few cycles. The fractional curve
runs on the manual backend, the one that takes a fractional occupation: the
two density builders of the manual solver fill the orbitals in order through
the mask ``clip(nocc - i, 0, 1)`` and put the fraction in the orbital after
the last full one, so a fractional record is the atom's record with the
counts replaced. The energy at N = 0 is zero for every functional and is not
computed. In def2-TZVP the anion is unbound (E(2) above E(1) by 7.6
kcal/mol), so a functional's deviation from the exact line between 1 and 2
electrons carries its anion error with its curvature.

The curves are read by ``tools/analysis/make_diagnostics_figure.py``, which
keeps them beside the run as ``diagnostics_curves.json``.
"""
from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np

BOHR_TO_ANGSTROM = 0.529177210903
DIAGNOSTIC_BASIS = "def2-TZVP"
DIAGNOSTIC_GRID_LEVEL = 4
#: the comparators, (libxc name, label): the parent and the three of the Slim16 tables
COMPARATORS = (("pbe", "PBE"), ("r2scan", "r2SCAN"), ("b3lyp", "B3LYP"), ("wb97m-v", "wB97M-V"))
#: the inclusive arithmetic grids, (lo, hi, step): the bond length in bohr, the electron number
R_BOHR_DEFAULT = (0.5, 6.0, 0.1)
N_DEFAULT = (0.0, 2.0, 0.1)
CURVES_FILE = "diagnostics_curves.json"
#: the cycle budget of the fractional curve: the forward loop runs the whole
#: budget, and the H atom converges from its seed in at most 26 cycles over
#: the N grid at 1e-8 (the PBE parent and the trained networks alike)
FRACTIONAL_CYCLES = 40
#: CCSD thresholds at which the two-electron energies equal full CI to 4e-12 Ha
CCSD_CONV_TOL = 1e-12
CCSD_CONV_TOL_NORMT = 1e-10


def grid(lo: float, hi: float, step: float) -> list:
    """The arithmetic grid from ``lo`` in steps of ``step`` up to the last
    multiple within ``hi`` (``hi`` itself when the step divides the range),
    each value rounded to six decimals; an end or a step that is not finite,
    a step that is not positive or one that exceeds the range is refused."""
    lo, hi, step = float(lo), float(hi), float(step)
    if not all(math.isfinite(v) for v in (lo, hi, step)) or step <= 0.0 or step > hi - lo:
        raise ValueError(f"the grid ({lo}, {hi}, {step}) must have finite ends and a finite, "
                         f"positive step of at most {hi - lo}")
    n = int(math.floor((hi - lo) / step + 1e-9))
    return [round(lo + i * step, 6) for i in range(n + 1)]


def diagnostic_solver_config():
    """The converged channel's cycles and tolerance on the manual backend
    with full integrals: FULL mode, 100 cycles, 1e-8, the features
    reassembled from the live density, the PBE seed."""
    from xcquinox.pipeline.solver import (
        FeaturePolicy, SolverBackend, SolverConfig, SolverMode,
    )
    return SolverConfig(backend=SolverBackend.MANUAL, mode=SolverMode.FULL,
                        max_cycles=100, conv_tol=1e-8,
                        feature_policy=FeaturePolicy.REASSEMBLE,
                        density_fit=False, seed_source="pbe")


def fractional_solver_config():
    """The fractional curve's configuration: the manual one at
    ``FRACTIONAL_CYCLES`` cycles."""
    return dataclasses.replace(diagnostic_solver_config(), max_cycles=FRACTIONAL_CYCLES)


def h2_solver_config():
    """The dissociation curve's configuration: the converged channel's
    pyscfad backend with full integrals, FULL mode, 100 cycles, 1e-8, the
    features reassembled from the live density, the PBE seed."""
    from xcquinox.pipeline.solver import (
        FeaturePolicy, SolverBackend, SolverConfig, SolverMode,
    )
    return SolverConfig(backend=SolverBackend.PYSCFAD, mode=SolverMode.FULL,
                        max_cycles=100, conv_tol=1e-8,
                        feature_policy=FeaturePolicy.REASSEMBLE,
                        density_fit=False, seed_source="pbe")


def h2_spec(r_bohr: float):
    """H2 with the bond along z, the length in bohr."""
    from xcquinox.pipeline.config import MoleculeSpec
    return MoleculeSpec(name=f"h2_r{float(r_bohr):.2f}",
                        atom=f"H 0 0 0; H 0 0 {float(r_bohr) * BOHR_TO_ANGSTROM:.10f}",
                        basis=DIAGNOSTIC_BASIS, charge=0, spin=0,
                        atom_composition=(("H", 2),), grid_level=DIAGNOSTIC_GRID_LEVEL)


def h_atom_spec():
    """The H atom, one unpaired electron."""
    from xcquinox.pipeline.config import MoleculeSpec
    return MoleculeSpec(name="h_atom", atom="H 0 0 0", basis=DIAGNOSTIC_BASIS,
                        charge=0, spin=1, atom_composition=(("H", 1),),
                        grid_level=DIAGNOSTIC_GRID_LEVEL)


def occupation_counts(n: float) -> tuple:
    """``(nocc_a, nocc_b)`` of the H atom at ``n`` electrons: the alpha
    channel up to one, the rest beta."""
    n = float(n)
    if not 0.0 <= n <= 2.0:
        raise ValueError(f"the fractional-charge curve runs from 0 to 2 electrons, not {n}")
    return min(n, 1.0), max(n - 1.0, 0.0)


def fractional_record(md: dict, n: float) -> dict:
    """The H atom's record at ``n`` electrons: a shallow copy with the counts
    of :func:`occupation_counts`; at one electron the record itself. The
    seed density is left as the atom's: the SCF converges from it in 20 to 26
    cycles over the N grid, against 17 to 23 from a seed scaled to the
    count."""
    n_a, n_b = occupation_counts(n)
    if n_a == 1.0 and n_b == 0.0:
        return md
    if not md.get("is_unrestricted", False):
        raise ValueError("a fractional record is built from an unrestricted record")
    out = dict(md)
    out["nocc_a"], out["nocc_b"] = n_a, n_b
    return out


def _require_fractional_config(config) -> None:
    """A fractional occupation reaches the SCF through the manual backend's
    density builders alone; under another backend or a one-shot mode the
    record would silently give the one-electron energy."""
    from xcquinox.pipeline.solver import SolverBackend, SolverMode
    if config.backend != SolverBackend.MANUAL or config.mode != SolverMode.FULL:
        raise ValueError("the fractional-charge curve runs on the manual backend in FULL "
                         f"mode, not {config.backend.value}/{config.mode.value}")


def network_energy(model, md: dict, config) -> tuple:
    """``(E, converged, cycles)`` of one self-consistent run of ``model`` on
    the record, the evaluation path's own call."""
    from xcquinox.pipeline.solver import run_scf
    result = run_scf(config, model, md, forward_only=True)
    return float(result.total_energy), bool(result.converged), int(result.cycles_run)


def _signature(spec) -> tuple:
    """The descriptors a network's record carries and the precompute keys of
    a full-integral SCF, whatever the spec trained with."""
    return tuple(spec.arch.materialize_descriptors()), ("eri",)


def precompute_records(mol_spec, models: dict) -> dict:
    """One record per network label, the precompute done once per
    descriptor signature and shared."""
    from xcquinox.pipeline.data import precompute_fixed_density_data
    cache: dict = {}
    out = {}
    for label, (_model, spec) in models.items():
        descriptors, required_keys = _signature(spec)
        key = (tuple(repr(d) for d in descriptors), required_keys)
        if key not in cache:
            cache[key] = precompute_fixed_density_data(
                mol_spec, required_keys=required_keys, descriptors=descriptors)
        out[label] = cache[key]
    return out


def _empty_curve() -> dict:
    return {"E": [], "converged": [], "cycles": []}


def dissociation_curve(models: dict, r_values, config=None) -> dict:
    """``{label: {"E", "converged", "cycles"}}`` of the restricted H2 energy
    at each bond length (bohr) for ``models`` (``{label: (model, spec)}``),
    under ``config`` (the pyscfad configuration by default)."""
    config = config or h2_solver_config()
    out = {label: _empty_curve() for label in models}
    for r in r_values:
        records = precompute_records(h2_spec(r), models)
        for label, (model, _spec) in models.items():
            e, converged, cycles = network_energy(model, records[label], config)
            out[label]["E"].append(e)
            out[label]["converged"].append(converged)
            out[label]["cycles"].append(cycles)
    return out


def fractional_charge_curve(models: dict, n_values, config=None) -> dict:
    """``{label: {"E", "converged", "cycles"}}`` of the H atom's energy at
    each electron number for ``models`` under ``config`` (the manual
    configuration by default; another is refused); zero at N = 0 without a
    run."""
    config = config or fractional_solver_config()
    _require_fractional_config(config)
    base = precompute_records(h_atom_spec(), models)
    out = {label: _empty_curve() for label in models}
    for n in n_values:
        for label, (model, _spec) in models.items():
            if float(n) == 0.0:
                e, converged, cycles = 0.0, True, 0
            else:
                e, converged, cycles = network_energy(
                    model, fractional_record(base[label], n), config)
            out[label]["E"].append(e)
            out[label]["converged"].append(converged)
            out[label]["cycles"].append(cycles)
    return out


def _resolve_comparators(xcs) -> list:
    """``[(libxc name, label), ...]`` for names or labels of ``COMPARATORS``,
    all of them when ``xcs`` is None."""
    if xcs is None:
        return list(COMPARATORS)
    by_name = dict(COMPARATORS)
    by_label = {label: xc for xc, label in COMPARATORS}
    out = []
    for item in xcs:
        if item in by_name:
            out.append((item, by_name[item]))
        elif item in by_label:
            out.append((by_label[item], item))
        else:
            raise KeyError(f"not a comparator of the diagnostics: {item!r} "
                           f"(the comparators: {', '.join(by_name)})")
    return out


def _pyscf_mol(mol_spec):
    from pyscf import gto
    from xcquinox.pipeline.config import mole_ecp
    return gto.M(atom=mol_spec.atom, basis=mol_spec.basis, charge=mol_spec.charge,
                 spin=mol_spec.spin, unit="angstrom", verbose=0,
                 ecp=mole_ecp(mol_spec.basis, mol_spec.atom))


def _pyscf_ks(mol, xc: str, unrestricted: bool):
    from pyscf import dft
    from xcquinox.pipeline.pyscf_determinism import pin_small_rho_cutoff
    mf = dft.UKS(mol) if unrestricted else dft.RKS(mol)
    mf.xc = xc
    mf.grids.level = DIAGNOSTIC_GRID_LEVEL
    pin_small_rho_cutoff(mf)
    return mf


def comparator_dissociation(r_values, xcs=None) -> dict:
    """``{label: {"E", "converged"}}``: the restricted Kohn-Sham energy of H2
    under each comparator at each bond length, through pyscf at the
    diagnostics' basis and grid (the restricted curve, the networks'
    closed-shell treatment)."""
    out = {label: {"E": [], "converged": []} for _xc, label in _resolve_comparators(xcs)}
    for r in r_values:
        mol = _pyscf_mol(h2_spec(r))
        for xc, label in _resolve_comparators(xcs):
            mf = _pyscf_ks(mol, xc, unrestricted=False)
            e = mf.kernel()
            out[label]["E"].append(float(e))
            out[label]["converged"].append(bool(mf.converged))
    return out


def fractional_occupations(n: float, nao: int) -> np.ndarray:
    """pyscf's occupation arrays of the H atom at ``n`` electrons: the
    orbitals filled in order per channel, the fraction in the first partly
    filled one, ``(2, nao)``."""
    n_a, n_b = occupation_counts(n)
    i = np.arange(nao)
    return np.stack([np.clip(n_a - i, 0.0, 1.0), np.clip(n_b - i, 0.0, 1.0)])


def comparator_fractional(n_values, xcs=None) -> dict:
    """``{label: {"E", "converged"}}``: the unrestricted Kohn-Sham energy of
    the H atom at each electron number under each comparator, pyscf's
    occupations overridden by :func:`fractional_occupations`; zero at N = 0."""
    mol = _pyscf_mol(h_atom_spec())
    nao = int(mol.nao)
    out = {label: {"E": [], "converged": []} for _xc, label in _resolve_comparators(xcs)}
    for n in n_values:
        for xc, label in _resolve_comparators(xcs):
            if float(n) == 0.0:
                out[label]["E"].append(0.0)
                out[label]["converged"].append(True)
                continue
            mf = _pyscf_ks(mol, xc, unrestricted=True)
            occupations = fractional_occupations(n, nao)
            mf.get_occ = lambda mo_energy=None, mo_coeff=None, occ=occupations: occ
            e = mf.kernel()
            out[label]["E"].append(float(e))
            out[label]["converged"].append(bool(mf.converged))
    return out


def _tight_ccsd(mol) -> float:
    """The CCSD energy of a two-electron molecule converged to
    ``CCSD_CONV_TOL``, full CI in the basis; a CCSD that does not converge
    raises."""
    from pyscf import cc, scf
    hf = scf.RHF(mol).run()
    mycc = cc.CCSD(hf)
    mycc.conv_tol = CCSD_CONV_TOL
    mycc.conv_tol_normt = CCSD_CONV_TOL_NORMT
    mycc.run()
    if not mycc.converged:
        raise RuntimeError(f"CCSD of {mol.atom} did not converge")
    return float(mycc.e_tot)


def ccsd_dissociation(r_values) -> list:
    """The CCSD energy of H2 at each bond length, exact for two electrons in
    the basis."""
    return [_tight_ccsd(_pyscf_mol(h2_spec(r))) for r in r_values]


def exact_fractional_reference(n_values) -> dict:
    """The straight lines between the exact energies at the integers: zero
    at N = 0, the H atom's UHF energy at N = 1 (exact for one electron in the
    basis), the anion's CCSD energy at N = 2 (exact for two);
    ``{"integer_energies": {"0", "1", "2"}, "reference_linear": [...]}``."""
    from pyscf import gto, scf
    from xcquinox.pipeline.config import mole_ecp
    e1 = float(scf.UHF(_pyscf_mol(h_atom_spec())).run().e_tot)
    e2 = _tight_ccsd(gto.M(atom="H 0 0 0", basis=DIAGNOSTIC_BASIS, charge=-1, spin=0,
                           verbose=0, ecp=mole_ecp(DIAGNOSTIC_BASIS, "H 0 0 0")))
    integers = {0: 0.0, 1: e1, 2: e2}
    linear = []
    for n in n_values:
        n = float(n)
        m = min(int(math.floor(n)), 1)
        f = n - m
        linear.append((1.0 - f) * integers[m] + f * integers[m + 1])
    return {"integer_energies": {"0": 0.0, "1": e1, "2": e2}, "reference_linear": linear}


def select_networks(manifest: dict, labels=None) -> dict:
    """One network per architecture, the first in manifest order, or the
    networks named by ``labels``: ``{label: {"index", "arch_name"}}``. A
    network without an architecture name is refused: the figure colours by
    it and the selection groups by it."""
    networks = list(manifest["networks"])
    nameless = [n.get("label") for n in networks if not n.get("arch_name")]
    if nameless:
        raise ValueError(f"networks without an architecture name: {nameless}")
    if labels:
        by_label = {n.get("label"): n for n in networks}
        missing = [label for label in labels if label not in by_label]
        if missing:
            raise KeyError(f"the run carries no network {missing}")
        chosen = [by_label[label] for label in labels]
    else:
        seen, chosen = set(), []
        for n in networks:
            arch = n.get("arch_name")
            if arch in seen:
                continue
            seen.add(arch)
            chosen.append(n)
    return {n["label"]: {"index": int(n["index"]), "arch_name": n.get("arch_name")}
            for n in chosen}


def load_networks(run_dir, selected: dict) -> dict:
    """``{label: (model, spec)}`` of the selected networks of a Slim16
    evaluation run, through the loaders the job and the evaluation use."""
    from xcquinox.pipeline.cluster._eval_one_spec import (
        _checkpoint_dir, _load_spec, _read_width, _spec_path,
    )
    from xcquinox.pipeline.eval_holdout import load_trained_model
    run_dir = str(run_dir)
    width = _read_width(run_dir)
    models = {}
    for label, info in selected.items():
        spec = _load_spec(_spec_path(run_dir, info["index"], width))
        checkpoint = Path(_checkpoint_dir(run_dir, info["index"], width)) / "model.eqx"
        models[label] = (load_trained_model(spec, checkpoint), spec)
    return models


def compute_diagnostics(run_dir, labels=None, r_values=None, n_values=None,
                        models=None) -> dict:
    """The payload of both curves for one network per architecture of a
    Slim16 evaluation run (or the networks named; or ``models``, ``{label:
    (model, spec)}``, handed in with their labels in the manifest), the
    comparators and the references::

        {"identity": {"basis", "grid_level", "solver": {"h2", "h"}, "run_dir",
                      "r_bohr", "n", "comparators"},
         "networks": {label: {"index", "arch_name"}},
         "h2": {"r_bohr", "reference_ccsd", "curves": {label: {"E", "converged", "cycles"}}},
         "h": {"n", "reference_linear", "integer_energies", "curves": {...}}}

    The grids are checked before any run: a bond length must be positive and
    an electron number within 0 and 2."""
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    selected = select_networks(manifest, labels if labels else
                               (list(models) if models else None))
    r_values = list(r_values) if r_values is not None else grid(*R_BOHR_DEFAULT)
    n_values = list(n_values) if n_values is not None else grid(*N_DEFAULT)
    for n in n_values:
        occupation_counts(n)
    if not r_values or any(not math.isfinite(float(r)) or float(r) <= 0.0 for r in r_values):
        raise ValueError(f"the bond lengths must be finite and positive: {r_values}")
    if models is None:
        models = load_networks(run_dir, selected)
    h2_config, h_config = h2_solver_config(), fractional_solver_config()
    h2_curves = dissociation_curve(models, r_values, h2_config)
    h2_curves.update(comparator_dissociation(r_values))
    h_curves = fractional_charge_curve(models, n_values, h_config)
    h_curves.update(comparator_fractional(n_values))
    reference = exact_fractional_reference(n_values)
    return {
        "identity": {"basis": DIAGNOSTIC_BASIS, "grid_level": DIAGNOSTIC_GRID_LEVEL,
                     "solver": {"h2": h2_config.describe(), "h": h_config.describe()},
                     "run_dir": str(run_dir), "r_bohr": r_values, "n": n_values,
                     "comparators": [label for _xc, label in COMPARATORS]},
        "networks": selected,
        "h2": {"r_bohr": r_values, "reference_ccsd": ccsd_dissociation(r_values),
               "curves": h2_curves},
        "h": {"n": n_values, "reference_linear": reference["reference_linear"],
              "integer_energies": reference["integer_energies"], "curves": h_curves},
    }


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def write_payload(path, payload: dict) -> None:
    """The payload as JSON, a non-finite energy written as null."""
    Path(path).write_text(json.dumps(json_safe(payload), indent=1, sort_keys=True) + "\n",
                          encoding="utf-8")
