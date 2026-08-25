"""xcquinox.alec.evaluation -- Metric ABC, 4 built-in metrics, run_test."""
import abc
import os
import csv
import json
import math
import struct
import time
from typing import ClassVar

import jax.numpy as jnp
import equinox as eqx

from xcquinox.alec.checkpoint_class import (model_class_of_arch,
                                            require_matching_class)
from xcquinox.alec.config import TestSpec, ArchitectureConfig
from xcquinox.alec.models import AlecGGAModel
from xcquinox.alec.data import precompute_fixed_density_data
from xcquinox.alec.oneshot import (
    total_energy_for_solver,
    oneshot_grid_density,
)
from xcquinox.alec.descriptors import assemble_descriptor_features


# ---------------------------------------------------------------------------
# Conversion factor
# ---------------------------------------------------------------------------

HA_TO_KCAL = 627.5094740631


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

METRIC_REGISTRY: dict[str, type["Metric"]] = {}


def register_metric(name: str):
    def deco(cls):
        METRIC_REGISTRY[name] = cls
        cls.registry_name = name
        return cls
    return deco


def make_metric(name: str, **kwargs) -> "Metric":
    if name not in METRIC_REGISTRY:
        raise KeyError(f"unknown metric {name!r}; known: {list(METRIC_REGISTRY)}")
    return METRIC_REGISTRY[name](**kwargs)


def list_metrics() -> list[str]:
    return sorted(METRIC_REGISTRY)


# ---------------------------------------------------------------------------
# Metric ABC
# ---------------------------------------------------------------------------

class Metric(abc.ABC):
    registry_name: ClassVar[str] = ""
    required_mol_keys: ClassVar[tuple[str, ...]] = ()

    @abc.abstractmethod
    def compute(
        self,
        model: "AlecGGAModel",
        mol_data: dict,
        solver_config: object | None = None,
    ) -> dict[str, float | None | bool | str]:
        """Compute this metric on one molecule.

        ``solver_config`` lets the metric evaluate the model under the
        same SCF protocol that trained it. Solver-invariant metrics
        (e.g. ``TotalEnergyMetric`` / ``AtomizationEnergyMetric``
        evaluate the post-hoc ``E[rho_PBE, V_xc^NN]`` functional) ignore
        it; metrics that depend on the predicted density
        (``DensityRMSEMetric``, ``SCFConvergenceMetric``) consume it.
        Default ``None`` is back-compat with callers that ran metrics
        without an SCF context.
        """
        ...


# ---------------------------------------------------------------------------
# TotalEnergyMetric
# ---------------------------------------------------------------------------

@register_metric("total_energy")
class TotalEnergyMetric(Metric):
    required_mol_keys: ClassVar[tuple[str, ...]] = (
        "rho_grid", "sigma_grid", "grid_weights", "E_non_xc", "E_pbe", "E_ref_literature",
    )

    def compute(self, model, mol_data, solver_config=None):
        # Solver-mode-aware energy: FULL evaluates the SELF-CONSISTENT
        # run_scf(...).total_energy (the energy a deployed functional actually
        # produces, and what FULL-mode training now optimizes); ONESHOT/FIXED_J/
        # None evaluate the one-shot fixed-density functional on rho_PBE. This
        # keeps training and evaluation measuring the same quantity.
        E_nn = float(total_energy_for_solver(model, mol_data, solver_config,
                                             forward_only=True))
        E_pbe = float(mol_data["E_pbe"])
        result = {"E_total_nn": E_nn, "E_pbe": E_pbe}
        E_ref = mol_data.get("E_ref_literature")
        if E_ref is not None:
            err_ha = E_nn - float(E_ref)
            result["E_error_hartree"] = err_ha
            result["E_error_kcalmol"] = err_ha * HA_TO_KCAL
        return result


# ---------------------------------------------------------------------------
# AtomizationEnergyMetric
# ---------------------------------------------------------------------------

@register_metric("atomization_energy")
class AtomizationEnergyMetric(Metric):
    required_mol_keys: ClassVar[tuple[str, ...]] = (
        "rho_grid", "sigma_grid", "grid_weights", "E_non_xc", "atom_composition",
    )

    def __init__(self, atom_energies: dict[str, float],
                 reference_ae_kcalmol: dict[str, float] | None = None):
        self.atom_energies = atom_energies
        self.reference_ae_kcalmol = reference_ae_kcalmol or {}

    def compute(self, model, mol_data, solver_config=None):
        # Same solver-mode-aware rule as TotalEnergyMetric: FULL uses the
        # self-consistent run_scf energy, ONESHOT/FIXED_J/None the one-shot
        # fixed-density functional. AE = sum(atom_energies) - E_mol; the
        # atom-energy anchors are fixed references.
        E_mol = float(total_energy_for_solver(model, mol_data, solver_config,
                                              forward_only=True))
        comp = mol_data["atom_composition"]
        E_atoms_sum = sum(self.atom_energies[sym] * n for sym, n in comp)
        AE_nn = E_atoms_sum - E_mol  # positive for bound molecule
        result = {"AE_nn": AE_nn}
        mol_name = mol_data.get("name", "")
        if mol_name in self.reference_ae_kcalmol:
            ae_ref = self.reference_ae_kcalmol[mol_name]
            result["AE_ref_kcalmol"] = ae_ref
            result["AE_error_hartree"] = AE_nn - ae_ref / HA_TO_KCAL
            result["AE_error_kcalmol"] = AE_nn * HA_TO_KCAL - ae_ref
        return result


# ---------------------------------------------------------------------------
# DensityRMSEMetric
# ---------------------------------------------------------------------------

def pbe_density_errors(mol_data) -> tuple:
    """Model-free PBE-vs-reference weighted grid density errors.

    ``mol_data['rho_grid']`` IS the PBE density evaluated on the same pruned
    grid the external reference density (``rho_ref_grid``, e.g. CCSD) was
    written for, so the PBE baseline needs no model and no extra SCF: it is
    the :class:`DensityRMSEMetric` formula with rho_pbe in place of rho_nn.
    Returns ``(rmse, l1)``, or ``(None, None)`` when no reference density is
    loaded.

    The returned errors are grid-weight-AVERAGED (see
    :class:`DensityRMSEMetric`), distinct from the DFS per-electron density
    error eps_{|n|} = (1/N_e) * integral|rho - rho_ref| (Letter Eq.20) and from
    the N_e^2-normalized training-loss form. The Eq. 20 form is emitted
    alongside via :func:`density_eps_terms` / :func:`pbe_density_eps`.
    """
    rho_ref = mol_data.get("rho_ref_grid")
    if rho_ref is None:
        return None, None
    rho_pbe = jnp.asarray(mol_data["rho_grid"])
    rho_ref = jnp.asarray(rho_ref)
    if rho_pbe.shape != rho_ref.shape:
        raise ValueError(
            f"density shape mismatch: rho_pbe {rho_pbe.shape} vs "
            f"rho_ref {rho_ref.shape}"
        )
    w = jnp.asarray(mol_data["grid_weights"])
    diff = rho_pbe - rho_ref
    rmse = float(jnp.sqrt(jnp.sum(w * diff ** 2) / jnp.sum(w)))
    l1 = float(jnp.sum(w * jnp.abs(diff)) / jnp.sum(w))
    return rmse, l1


def density_eps_terms(rho, rho_ref, w):
    """DFS Letter Eq. 20 per-species density-error ingredients.

    ``eps = sum_i(w_i |rho - rho_ref|_i) / N_e`` with ``N_e = sum_i(w_i *
    rho_ref_i)`` -- the quadrature electron count of the REFERENCE density,
    the same N_e convention as this package's density loss
    (``losses._grid_term`` with ``per_electron=True``, which realizes the
    Letter's Eq. 17 normalization; the vendored dpyscf instead counts
    neutral-atom Z, which is wrong for ions -- the quadrature count is
    charge-correct by construction). Using the identical quadrature for
    numerator and N_e makes eps a pure ratio of two integrals on the same
    grid, so grid-truncation error partially cancels.
    Returns ``(eps, n_electrons, grid_weight_sum)`` as floats; eps degrades
    to NaN when the quadrature N_e is non-positive (unphysical input)."""
    rho = jnp.asarray(rho)
    rho_ref = jnp.asarray(rho_ref)
    w = jnp.asarray(w)
    n_e = float(jnp.sum(w * rho_ref))
    wsum = float(jnp.sum(w))
    if n_e <= 0.0:
        return float("nan"), n_e, wsum
    eps = float(jnp.sum(w * jnp.abs(rho - rho_ref)) / n_e)
    return eps, n_e, wsum


def pbe_density_eps(mol_data) -> tuple:
    """Model-free DFS Eq. 20 terms for the stored PBE density vs the loaded
    reference: ``(density_eps_l1_pbe, n_electrons, grid_weight_sum)``, or
    ``(None, None, None)`` when no reference density is present (the
    historical skip semantics of :func:`pbe_density_errors`)."""
    rho_ref = mol_data.get("rho_ref_grid")
    if rho_ref is None:
        return None, None, None
    return density_eps_terms(mol_data["rho_grid"], rho_ref,
                             mol_data["grid_weights"])


@register_metric("density_rmse")
class DensityRMSEMetric(Metric):
    required_mol_keys: ClassVar[tuple[str, ...]] = (
        "rho_grid", "sigma_grid", "ao_grid", "grid_weights", "rho_ref_grid",
        "ref_density_method",
        "s_matrix", "h_core", "j_matrix", "nocc", "nocc_a", "nocc_b",
        "is_unrestricted", "atom_composition",
    )

    def compute(self, model, mol_data, solver_config=None):
        comp = mol_data["atom_composition"]
        total_atoms = sum(n for _, n in comp)
        if total_atoms == 1:
            return {
                "density_rmse": None,
                "density_l1": None,
                "density_rmse_pbe": None,
                "density_l1_pbe": None,
                "density_eps_l1": None,
                "density_eps_l1_pbe": None,
                "n_electrons": None,
                "grid_weight_sum": None,
                "skipped": True,
                "skip_reason": "atomic_system",
                "ref_density_method": mol_data.get("ref_density_method"),
            }
        # Solver-aware: when ``solver_config`` is supplied, the density is
        # the SCF-iterated density (FIXED_J / FULL) rather than the
        # 1-Roothaan-step oneshot density. ``oneshot_grid_density`` already
        # accepts ``solver_config=None`` for the back-compat oneshot path.
        rho_nn = oneshot_grid_density(model, mol_data, solver_config=solver_config,
                                      forward_only=True)
        rho_ref = mol_data["rho_ref_grid"]
        if rho_ref is None:
            # External CCSD reference density not loaded for this species
            # (e.g. spec.external_data_path was None, or the .npz file
            # didn't carry rho_ref_grid).  Skip gracefully, matches the
            # existing pattern in losses._grid_term which skips when
            # rho_ref is None.
            return {
                "density_rmse": None,
                "density_l1": None,
                "density_rmse_pbe": None,
                "density_l1_pbe": None,
                "density_eps_l1": None,
                "density_eps_l1_pbe": None,
                "n_electrons": None,
                "grid_weight_sum": None,
                "skipped": True,
                "skip_reason": "no_rho_ref_grid",
                "ref_density_method": mol_data.get("ref_density_method"),
            }
        if rho_nn.shape != rho_ref.shape:
            raise ValueError(
                f"density shape mismatch: rho_nn {rho_nn.shape} vs "
                f"rho_ref {rho_ref.shape}"
            )
        w = mol_data["grid_weights"]
        diff = rho_nn - rho_ref
        # Grid-weight-AVERAGED RMSE / L1: sqrt(sum w*(dn)^2 / sum w) and
        # sum w*|dn| / sum w -- the figure suite's self-calibrated scale.
        # The DFS per-electron density error eps_{|n|} = (1/N_e) *
        # integral|rho_nn - rho_ref| (Letter Eq.20) is emitted ALONGSIDE as
        # density_eps_l1 (+ the model-free PBE twin), with N_e the quadrature
        # integral of the reference density, so the Letter's gamma applies to
        # it dimensionally. n_electrons / grid_weight_sum make any other
        # normalization reconstructible offline.
        rmse = float(jnp.sqrt(jnp.sum(w * diff ** 2) / jnp.sum(w)))
        l1 = float(jnp.sum(w * jnp.abs(diff)) / jnp.sum(w))
        rmse_pbe, l1_pbe = pbe_density_errors(mol_data)
        eps_nn, n_e, wsum = density_eps_terms(rho_nn, rho_ref, w)
        eps_pbe, _, _ = pbe_density_eps(mol_data)
        return {
            "density_rmse": rmse,
            "density_l1": l1,
            # model-free PBE-vs-CCSD baseline on the same grid/weights, so
            # in-sample per_molecule.json carries the comparison directly
            "density_rmse_pbe": rmse_pbe,
            "density_l1_pbe": l1_pbe,
            "density_eps_l1": eps_nn,
            "density_eps_l1_pbe": eps_pbe,
            "n_electrons": n_e,
            "grid_weight_sum": wsum,
            "ref_density_method": mol_data.get("ref_density_method"),
        }


# ---------------------------------------------------------------------------
# PBEReferenceMetric
# ---------------------------------------------------------------------------

@register_metric("pbe_reference")
class PBEReferenceMetric(Metric):
    """Emit PBE's predicted total / atomization energy and its error vs
    literature reference.

    Model-independent baseline: reads ``mol_data['E_pbe']`` (stashed by
    ``precompute_fixed_density_data``) and combines with the
    caller-supplied ``atom_energies`` dict (PBE-consistent atomic totals)
    to produce ``AE_pbe = sum n_Z * atom_energies[Z] - E_pbe_total``.
    Bookkeeping layer that gives comparison plots a "what if we just used
    PBE" line next to trained-NN bars.

    The ``model`` argument is accepted for API uniformity and IGNORED.
    """
    required_mol_keys: ClassVar[tuple[str, ...]] = ("E_pbe", "atom_composition")

    def __init__(self, atom_energies: dict[str, float],
                 reference_ae_kcalmol: dict[str, float] | None = None):
        self.atom_energies = atom_energies
        self.reference_ae_kcalmol = reference_ae_kcalmol or {}

    def compute(self, model, mol_data, solver_config=None):
        del model, solver_config  # baseline is model- and solver-independent.
        E_pbe = float(mol_data["E_pbe"])
        comp = mol_data["atom_composition"]
        E_atoms_sum = sum(self.atom_energies[sym] * n for sym, n in comp)
        AE_pbe = E_atoms_sum - E_pbe  # positive for bound molecule
        result = {"E_pbe_total": E_pbe, "AE_pbe": AE_pbe}
        mol_name = mol_data.get("name", "")
        if mol_name in self.reference_ae_kcalmol:
            ae_ref = self.reference_ae_kcalmol[mol_name]
            result["AE_ref_kcalmol"] = ae_ref
            result["AE_error_pbe_hartree"] = AE_pbe - ae_ref / HA_TO_KCAL
            result["AE_error_pbe_kcalmol"] = AE_pbe * HA_TO_KCAL - ae_ref
        return result


# ---------------------------------------------------------------------------
# SCFConvergenceMetric
# ---------------------------------------------------------------------------

@register_metric("scf_convergence")
class SCFConvergenceMetric(Metric):
    """Run one SCF solve and report the cycles taken + converged flag.

    Solver-aware by construction. When ``solver_config`` is None or
    ONESHOT, the metric emits sentinel values (cycles_run=0,
    converged=True) since there is no SCF loop. Useful for the notebook's
    SCF convergence aggregate plot.

    Per-cycle |E_n - E_final| residuals (when the backend records an
    energy_trace) are emitted as ``scf_energy_residual_<i>`` keys for
    cycles that actually ran.
    """
    required_mol_keys: ClassVar[tuple[str, ...]] = (
        "rho_grid", "sigma_grid", "ao_grid", "grid_weights",
        "s_matrix", "h_core", "j_matrix", "nocc", "nocc_a", "nocc_b",
        "is_unrestricted",
    )

    def compute(self, model, mol_data, solver_config=None):
        if solver_config is None:
            return {"cycles_run": 0, "scf_converged": True}
        from xcquinox.alec.solver import SolverMode, run_scf
        if getattr(solver_config, "mode", None) == SolverMode.ONESHOT:
            return {"cycles_run": 0, "scf_converged": True}
        result = run_scf(solver_config, model, mol_data, forward_only=True)
        out = {
            "cycles_run": int(result.cycles_run),
            "scf_converged": bool(result.converged),
            "scf_total_energy": float(result.total_energy),
        }
        if getattr(result, "energy_trace", None) is not None:
            import numpy as _np
            trace = _np.asarray(result.energy_trace)
            e_final = float(result.total_energy)
            for i, e_step in enumerate(trace):
                if math.isnan(float(e_step)):
                    continue
                out[f"scf_energy_residual_{i}"] = abs(float(e_step) - e_final)
        return out


# ---------------------------------------------------------------------------
# ConstraintViolationsMetric
# ---------------------------------------------------------------------------

@register_metric("constraint_violations")
class ConstraintViolationsMetric(Metric):
    required_mol_keys: ClassVar[tuple[str, ...]] = (
        "rho_grid", "sigma_grid",
    )

    def compute(self, model, mol_data, solver_config=None):
        # Constraints are intrinsic to the model's F_x(rho, sigma) surface,
        # independent of how the model is run within an SCF loop.
        del solver_config
        features = assemble_descriptor_features(model.descriptors, mol_data)
        rho = mol_data["rho_grid"]
        sigma = mol_data["sigma_grid"]
        report = model.constraint_report(rho, sigma, features)
        return _flatten_constraint_report(report)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _flatten_constraint_report(report: dict) -> dict[str, float]:
    flat = {}
    for side, per_name in report.items():
        for name, stats in per_name.items():
            for stat, value in stats.items():
                flat[f"{side}_{name}_{stat}"] = float(value)
    return flat


# ---------------------------------------------------------------------------
# run_test
# ---------------------------------------------------------------------------

def run_test(spec: TestSpec, progress_callback=None) -> dict:
    """Evaluate a trained model on a set of molecules with registered metrics.

    Parameters
    ----------
    spec : TestSpec
        Full test/evaluation configuration.
    progress_callback : callable, optional
        Called with a dict payload for each molecule evaluated.

    Returns
    -------
    dict
        {"per_molecule": [...], "aggregate": {...}}
    """
    t0 = time.time()

    # 1. Validate
    spec.validate()

    # 2. Build model skeleton
    skeleton = AlecGGAModel.from_arch(spec.arch, seed=0)

    # The model class the checkpoint was written as, against the class of the
    # skeleton about to be filled. The parent anchor and the descriptor
    # coordinates change no parameter shape, so a checkpoint of another class
    # deserialises into this skeleton without complaint and evaluates as a
    # model that is neither -- the same hazard the polarization check below
    # covers for the one property that DOES change a width. Refused before
    # the leaves are read.
    #
    # The record is first held to the leaves it claims to describe, by the
    # SHA-256 it carries (checkpoint_class.require_matching_digest). Record
    # and checkpoint are two files with one rename each, so a training write
    # interrupted between them leaves the new record over the previous run's
    # complete .eqx; without the digest this reader would take the record's
    # word for leaves another run wrote, which is the silent cross-class load
    # in the form the class comparison alone cannot see.
    require_matching_class(spec.model_checkpoint, model_class_of_arch(spec.arch))

    # 3. Deserialize trained weights
    try:
        model = eqx.tree_deserialise_leaves(spec.model_checkpoint, skeleton)
    except (ValueError, EOFError, struct.error) as e:
        _path_for_hint = spec.model_checkpoint
        if "_attn" in _path_for_hint or "/attention" in _path_for_hint:
            raise ValueError(
                f"Failed to deserialise {_path_for_hint}: {e}\n\n"
                "This path includes an attention checkpoint. The "
                "self-attention block was rewritten 2026-04-27 to real "
                "multi-head scaled-dot-product attention; old `_attn` "
                "checkpoints are NOT loadable under the new schema. "
                "Delete the old checkpoint and retrain."
            ) from e
        raise

    # The cnet's static use_spin_polarization flag MUST match the arch's
    # flag. Mismatch would indicate a checkpoint built outside
    # create_network_pair OR a round-trip bug; either way, polarized vs
    # unpolarized comparisons would become degenerate at eval (zeta silently
    # dropped).
    if hasattr(model.cnet, "use_spin_polarization"):
        if model.cnet.use_spin_polarization != spec.arch.use_polarized_correlation:
            raise ValueError(
                f"Polarization-flag mismatch at load time: "
                f"model.cnet.use_spin_polarization="
                f"{model.cnet.use_spin_polarization} but "
                f"spec.arch.use_polarized_correlation="
                f"{spec.arch.use_polarized_correlation}. The cnet must be "
                f"built via create_network_pair(arch), the single site that "
                f"derives the cnet flag from arch.use_polarized_correlation."
            )

    # 4. Instantiate metrics
    mk_dict = spec.metric_kwargs_dict
    ae_dict = spec.atom_energies_dict
    metrics = []
    for name in spec.metrics:
        kwargs = dict(mk_dict.get(name, {}))
        if name == "atomization_energy":
            kwargs.setdefault("atom_energies", ae_dict)
        metrics.append(make_metric(name, **kwargs))

    # 5. Determine required mol_data keys
    metric_keys = set().union(*(m.required_mol_keys for m in metrics))
    descriptor_keys = set().union(*(d.required_mol_keys for d in spec.arch.materialize_descriptors()))
    # When the solver runs with FULL Fock rebuilds, run_scf needs the
    # 4-index ERI tensor in mol_data; precompute_fixed_density_data must
    # be told to include it via required_keys. ONESHOT and FIXED_J reuse
    # mol_data["j_matrix"] (already in DensityRMSEMetric.required_mol_keys),
    # so they don't need this branch.
    solver_keys: set[str] = set()
    if spec.solver_config is not None:
        from xcquinox.alec.solver import SolverMode
        if getattr(spec.solver_config, "mode", None) == SolverMode.FULL:
            solver_keys.add(
                "cderi" if getattr(spec.solver_config, "density_fit", False)
                else "eri")
            solver_keys.add("ao_grid_deriv")
    required_keys = tuple(metric_keys | descriptor_keys | solver_keys)

    # 6. Evaluate each molecule
    per_molecule = []
    for i, mol_spec in enumerate(spec.molecules):
        # Forward the spec's DF auxbasis exactly as the held-out path does
        # (eval_holdout.py): without it a CONFIGURED fitting basis silently
        # differs between training and the inline eval (auto-select on this
        # side). None when DF is off or the auxbasis is unset -> auto, which
        # matches training's own auto-select.
        _sc = spec.solver_config
        mol_data = precompute_fixed_density_data(
            mol_spec,
            required_keys=required_keys,
            descriptors=spec.arch.materialize_descriptors(),
            auxbasis=(getattr(_sc, "auxbasis", None)
                      if getattr(_sc, "density_fit", False) else None),
            orientation_lock_strength=getattr(
                _sc, "orientation_lock_strength", 0.0),
            seed_source=getattr(_sc, "seed_source", "pbe"),
            seed_cache_dir=getattr(_sc, "seed_cache_dir", None),
            seed_density_fit=bool(getattr(_sc, "density_fit", False)),
        )
        mol_result = {"molecule": mol_spec.name}
        for metric in metrics:
            metric_out = metric.compute(model, mol_data,
                                       solver_config=spec.solver_config)
            mol_result.update(metric_out)
        per_molecule.append(mol_result)
        if progress_callback is not None:
            progress_callback({
                "phase": "test",
                "molecule": mol_spec.name,
                "index": i + 1,
                "total": len(spec.molecules),
                "timestamp": time.time(),
            })

    # 7. Aggregate numeric scalars
    aggregate = _aggregate_results(per_molecule)

    duration = time.time() - t0

    # 8. Save artifacts
    os.makedirs(spec.output_dir, exist_ok=True)

    if spec.save_per_molecule:
        pm_json_path = os.path.join(spec.output_dir, "per_molecule.json")
        with open(pm_json_path, "w") as f:
            json.dump(per_molecule, f, indent=2, default=_json_default)

        pm_csv_path = os.path.join(spec.output_dir, "per_molecule.csv")
        if per_molecule:
            all_keys = []
            seen = set()
            for row in per_molecule:
                for k in row:
                    if k not in seen:
                        all_keys.append(k)
                        seen.add(k)
            with open(pm_csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=all_keys)
                writer.writeheader()
                writer.writerows(per_molecule)

    if spec.save_aggregate:
        agg_path = os.path.join(spec.output_dir, "aggregate.json")
        with open(agg_path, "w") as f:
            json.dump(aggregate, f, indent=2, default=_json_default)

    # test_metadata.json is always saved
    test_metadata = {
        "arch_name": spec.arch.name,
        "model_checkpoint": spec.model_checkpoint,
        "metrics": list(spec.metrics),
        "molecules": [m.name for m in spec.molecules],
        "metric_kwargs": spec.metric_kwargs_dict,
        "atom_energies": spec.atom_energies_dict,
        "output_dir": spec.output_dir,
        "save_per_molecule": spec.save_per_molecule,
        "save_aggregate": spec.save_aggregate,
        "solver_config": (
            spec.solver_config.describe()
            if spec.solver_config is not None
            else None
        ),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "duration_seconds": round(duration, 1),
    }
    md_path = os.path.join(spec.output_dir, "test_metadata.json")
    with open(md_path, "w") as f:
        json.dump(test_metadata, f, indent=2)

    # 9. Return
    return {"per_molecule": per_molecule, "aggregate": aggregate}


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _json_default(obj):
    """JSON serialization fallback for non-standard types."""
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (jnp.ndarray,)):
        return float(obj)
    return str(obj)


#: Quadrature bookkeeping columns (not error metrics): excluded from
#: aggregate.json -- their mean/MAE/RMSE would read as pseudo-metrics
#: (grid_weight_sum is ~1e5-scale integration volume, n_electrons a count).
_NON_METRIC_KEYS = frozenset({"n_electrons", "grid_weight_sum"})


def _aggregate_results(per_molecule: list[dict]) -> dict:
    """For each numeric key across molecules, compute mean/MAE/RMSE/max/count.

    Each metric entry now also records:
      ``n_total``   -- number of molecules considered for the key (i.e. the
                       length of per_molecule, regardless of skips).
      ``n_skipped`` -- molecules that had None/NaN/non-numeric values for the
                       key and therefore did not contribute to the statistics.

    ``count`` (== n_total - n_skipped) is the number that did contribute.
    All three fields are present so consumers can detect partial-population
    aggregates (e.g. 2 of 26 molecules having density_rmse != None).
    """
    if not per_molecule:
        return {}

    n_total = len(per_molecule)

    # Collect all keys that appear in any result
    all_keys = set()
    for row in per_molecule:
        all_keys.update(row.keys())

    aggregate = {}
    for key in sorted(all_keys):
        if key == "molecule" or key in _NON_METRIC_KEYS:
            continue
        values = []
        for row in per_molecule:
            v = row.get(key)
            if v is None:
                continue
            if isinstance(v, bool):
                continue
            if isinstance(v, str):
                continue
            if isinstance(v, (int, float)):
                if math.isfinite(v):
                    values.append(v)
        if not values:
            continue
        arr = [float(v) for v in values]
        n = len(arr)
        mean_val = sum(arr) / n
        abs_arr = [abs(v) for v in arr]
        mae_val = sum(abs_arr) / n
        rmse_val = math.sqrt(sum(v ** 2 for v in arr) / n)
        max_val = max(abs_arr)
        aggregate[key] = {
            "mean": mean_val,
            "MAE": mae_val,
            "RMSE": rmse_val,
            "max": max_val,
            "count": n,
            "n_total": n_total,
            "n_skipped": n_total - n,
        }
    return aggregate
