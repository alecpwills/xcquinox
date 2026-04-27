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

from xcquinox.alec.config import TestSpec, ArchitectureConfig
from xcquinox.alec.models import AlecGGAModel
from xcquinox.alec.data import precompute_fixed_density_data
from xcquinox.alec.oneshot import fixed_density_total_energy, oneshot_grid_density
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
        # Solver-invariant by framework design: the post-hoc fixed-density
        # total energy is a functional of rho_PBE with NN's V_xc, not of
        # the SCF-iterated density.
        del solver_config
        E_nn = float(fixed_density_total_energy(model, mol_data))
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
        # Same rationale as TotalEnergyMetric: AE is a difference of
        # fixed-density functionals, so it is solver-invariant by design.
        del solver_config
        E_mol = float(fixed_density_total_energy(model, mol_data))
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
                "skipped": True,
                "skip_reason": "atomic_system",
                "ref_density_method": mol_data.get("ref_density_method"),
            }
        # Solver-aware: when ``solver_config`` is supplied, the density is
        # the SCF-iterated density (FIXED_J / FULL) rather than the
        # 1-Roothaan-step oneshot density. ``oneshot_grid_density`` already
        # accepts ``solver_config=None`` for the back-compat oneshot path.
        rho_nn = oneshot_grid_density(model, mol_data, solver_config=solver_config)
        rho_ref = mol_data["rho_ref_grid"]
        if rho_nn.shape != rho_ref.shape:
            raise ValueError(
                f"density shape mismatch: rho_nn {rho_nn.shape} vs "
                f"rho_ref {rho_ref.shape}"
            )
        w = mol_data["grid_weights"]
        diff = rho_nn - rho_ref
        rmse = float(jnp.sqrt(jnp.sum(w * diff ** 2) / jnp.sum(w)))
        l1 = float(jnp.sum(w * jnp.abs(diff)) / jnp.sum(w))
        return {
            "density_rmse": rmse,
            "density_l1": l1,
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
        result = run_scf(solver_config, model, mol_data)
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
            solver_keys.add("eri")
            solver_keys.add("ao_grid_deriv")
    required_keys = tuple(metric_keys | descriptor_keys | solver_keys)

    # 6. Evaluate each molecule
    per_molecule = []
    for i, mol_spec in enumerate(spec.molecules):
        mol_data = precompute_fixed_density_data(
            mol_spec,
            required_keys=required_keys,
            descriptors=spec.arch.materialize_descriptors(),
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


def _aggregate_results(per_molecule: list[dict]) -> dict:
    """For each numeric key across molecules, compute mean/MAE/RMSE/max/count."""
    if not per_molecule:
        return {}

    # Collect all keys that appear in any result
    all_keys = set()
    for row in per_molecule:
        all_keys.update(row.keys())

    aggregate = {}
    for key in sorted(all_keys):
        if key == "molecule":
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
        }
    return aggregate
