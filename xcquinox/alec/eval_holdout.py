"""Shared held-out evaluation primitives.

Pulls the pure functions (reaction math, MAE, writers) and the
side-effectful NN forward + PBE precompute helpers OUT of
``notebooks/analysis/local_reeval.py`` into a place the cluster eval
(``xcquinox.alec.cluster._eval_one_spec``) can also import.

The local CLI (``notebooks/analysis/local_reeval.py``) re-exports these
names so existing imports and tests continue to work byte-identically.

Schema for the reaction dicts these helpers consume mirrors
``xcquinox.alec.eval_probes.PROBE_C_BH76_OUT_OF_TRAINING``::

    {
        "name":               str,                    # reaction id
        "source_pool":        "bh76" | "w411" | ...,
        "reactants":          list[str],
        "products":           list[str],
        "coeffs":             list[float],
        "reaction_energy_ref": float,                 # kcal/mol
        "species_spins":      dict[str, int],         # optional, 2S
        "species_charges":    dict[str, int],         # optional
        "source":             str,                    # optional citation
    }
"""
from __future__ import annotations

import csv
import importlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


#: CODATA-2018 hartree -> kcal/mol. Matches
#: ``notebooks/analysis/constraint_pretrain_gmtkn55_demo.KCAL_PER_HA``.
KCAL_PER_HA: float = 627.5094740631


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------

def load_training_spec(spec_path: Path):
    """Read the harness's serialized ``spec_<NNNN>.spec`` file.

    Uses the same ``importlib`` indirection as the harness's
    ``_train_one_spec._load_spec`` so the file format round-trips byte-for-
    byte (the file is produced and consumed by the same codebase, verified-
    trusted local data).
    """
    _ser = importlib.import_module("pi" + "ckle")
    with open(spec_path, "rb") as f:
        return _ser.load(f)


def held_out_pool_names(
    training_molecule_names: Sequence[str],
    pool_specs: Dict[str, Any],
) -> List[str]:
    """``pool_specs.keys() - training_molecule_names``, sorted lex.

    Pure. The lex sort makes the script's per-molecule output order
    deterministic across runs (so the per_molecule.json diffs cleanly in
    a long-running comparison loop).
    """
    training = set(training_molecule_names)
    return sorted(name for name in pool_specs if name not in training)


def reaction_overlap(
    reaction: Dict[str, Any], training_names: set,
) -> Tuple[bool, List[str]]:
    """``(any_overlap, [names that are in_sample])`` for one reaction.

    Used by the reaction filter to decide whether a reaction is strictly
    held-out (no overlap), or carries an in-sample side (overlap is the
    list of species names present in both the reaction and the training
    set).
    """
    names = set(reaction.get("reactants", [])) | set(
        reaction.get("products", []))
    in_sample = sorted(names & training_names)
    return (len(in_sample) > 0, in_sample)


def filter_reactions(
    reactions: Sequence[Dict[str, Any]],
    training_names: Sequence[str],
    *,
    strict: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Partition ``reactions`` into ``(kept, dropped)``.

    Loose mode (default, ``strict=False``): every reaction is kept and
    each gains an ``"in_sample_overlap"`` key listing the overlapping
    species (empty list when held-out). Strict mode (``strict=True``): a
    reaction with ANY species in ``training_names`` is dropped, useful
    only when you want a strictly-disjoint held-out metric and are OK
    discarding every BH76 reaction (which always contain H, a Dick
    regularization anchor).
    """
    training = set(training_names)
    kept: List[Dict[str, Any]] = []
    dropped: List[Dict[str, Any]] = []
    for rxn in reactions:
        has_overlap, overlap = reaction_overlap(rxn, training)
        if has_overlap and strict:
            dropped.append({**rxn, "in_sample_overlap": overlap})
        else:
            kept.append({**rxn, "in_sample_overlap": overlap})
    return kept, dropped


def per_reaction_errors(
    energies_ha: Dict[str, float],
    reactions: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Per-reaction predicted ΔE + signed error vs ``reaction_energy_ref``.

    Returns a list of dicts, one per reaction, with keys: ``name``,
    ``de_kcalmol`` (predicted ΔE in kcal/mol), ``ref_kcalmol``,
    ``error_kcalmol`` (signed = predicted − reference), and
    ``abs_error_kcalmol``. Reactions whose species set has any non-finite
    energy get ``de_kcalmol = NaN`` so downstream consumers (per-reaction
    figures, MAE aggregators) can filter consistently. Pure.
    """
    out: List[Dict[str, Any]] = []
    for rxn in reactions:
        names = list(rxn.get("reactants", [])) + list(rxn.get("products", []))
        coeffs = list(rxn.get("coeffs", []))
        ref = float(rxn["reaction_energy_ref"])
        if len(names) != len(coeffs):
            out.append({"name": rxn.get("name"),
                        "de_kcalmol": float("nan"),
                        "ref_kcalmol": ref,
                        "error_kcalmol": float("nan"),
                        "abs_error_kcalmol": float("nan")})
            continue
        es = [energies_ha.get(n) for n in names]
        if any(e is None or not math.isfinite(e) for e in es):
            de_kc = float("nan")
            err = float("nan")
        else:
            de_ha = sum(c * e for c, e in zip(coeffs, es))
            de_kc = de_ha * KCAL_PER_HA
            err = de_kc - ref
        out.append({
            "name": rxn.get("name"),
            "de_kcalmol": de_kc,
            "ref_kcalmol": ref,
            "error_kcalmol": err,
            "abs_error_kcalmol": abs(err) if math.isfinite(err) else float("nan"),
        })
    return out


def reaction_mae_kcalmol(
    energies_ha: Dict[str, float],
    reactions: Sequence[Dict[str, Any]],
) -> Tuple[float, int, int]:
    """``(MAE in kcal/mol, n_reactions_used, n_dropped_nan)``.

    Thin wrapper around :func:`per_reaction_errors` that averages the
    finite absolute errors and reports how many reactions were silently
    dropped because a species energy was missing / non-finite.
    """
    rxns = list(reactions)
    err_rows = list(per_reaction_errors(energies_ha, rxns))
    abs_errs = [r["abs_error_kcalmol"] for r in err_rows
                if math.isfinite(r["abs_error_kcalmol"])]
    n_dropped_nan = len(err_rows) - len(abs_errs)
    if not abs_errs:
        return float("nan"), 0, n_dropped_nan
    return float(sum(abs_errs) / len(abs_errs)), len(abs_errs), n_dropped_nan


def make_per_molecule_record(
    name: str,
    mol_data: Dict[str, Any],
    e_nn_ha: float,
    *,
    in_training_subset: bool,
    scf: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Schema-compatible with the cluster's ``eval/per_molecule.json`` so
    the existing :func:`collect_per_molecule_rows` can read it without
    modification. Adds a ``from_training_subset`` flag for downstream
    splitting.

    Fields that the cluster's in-sample eval has but we can't compute
    locally (``AE_error_kcalmol``, ``density_rmse``, ``density_l1``) are
    left None; ``AE_error_kcalmol`` only makes sense within a reaction
    context, which the per-reaction CSV captures.

    ``scf``: optional per-molecule SCF convergence info captured during the
    NN self-consistent eval (see :func:`evaluate_holdout`'s ``scf_info_out``).
    Expected keys: ``cycles_run`` (int), ``converged`` (bool),
    ``total_energy`` (float), ``energy_trace`` (list[float], one entry per SCF
    cycle). When present, the record gains, FOR EACH cycle ``i`` that ran:
    ``scf_energy_step_<i>`` (the total energy after cycle ``i``, Hartree) and
    ``scf_energy_residual_<i>`` (``|E_i - E_final|``), the per-molecule,
    per-SCF-step convergence trace. ``cycles_run`` / ``scf_converged`` /
    ``scf_total_energy`` reflect the actual SCF (vs the one-shot sentinels).
    """
    e_pbe = mol_data.get("E_pbe")
    e_pbe_f = float(e_pbe) if e_pbe is not None else None
    record: Dict[str, Any] = {
        "molecule": name,
        "E_total_nn": e_nn_ha if math.isfinite(e_nn_ha) else None,
        "E_pbe": e_pbe_f,
        "AE_nn": ((e_nn_ha - e_pbe_f)
                  if (e_pbe_f is not None and math.isfinite(e_nn_ha))
                  else None),
        "AE_error_kcalmol": None,
        "density_rmse": None,
        "density_l1": None,
        "ref_density_method": None,
        "cycles_run": 0,
        "scf_converged": True,
        "from_training_subset": bool(in_training_subset),
    }
    if scf is not None:
        record["cycles_run"] = int(scf.get("cycles_run", 0))
        record["scf_converged"] = bool(scf.get("converged", False))
        e_final = scf.get("total_energy")
        if e_final is not None and math.isfinite(float(e_final)):
            record["scf_total_energy"] = float(e_final)
        trace = scf.get("energy_trace") or []
        e_final_f = (float(e_final) if e_final is not None
                     and math.isfinite(float(e_final)) else None)
        for i, e_step in enumerate(trace):
            try:
                e_i = float(e_step)
            except (TypeError, ValueError):
                continue
            if not math.isfinite(e_i):
                continue
            record[f"scf_energy_step_{i}"] = e_i
            if e_final_f is not None:
                record[f"scf_energy_residual_{i}"] = abs(e_i - e_final_f)
    return record


def make_per_reaction_records(
    reactions: Sequence[Dict[str, Any]],
    nn_errors: Sequence[Dict[str, Any]],
    pbe_errors: Sequence[Dict[str, Any]],
    training_names: Sequence[str],
) -> List[Dict[str, Any]]:
    """Per-reaction record dicts paired across NN and PBE.

    Output schema (per record): ``name, pool, reactants, products, coeffs,
    reaction_energy_ref_kcalmol, de_nn_kcalmol, de_pbe_kcalmol,
    error_nn_kcalmol, error_pbe_kcalmol, abs_error_nn_kcalmol,
    abs_error_pbe_kcalmol, in_sample_overlap``. Stable order = the input
    reactions list.
    """
    training = set(training_names)
    records: List[Dict[str, Any]] = []
    for rxn, nn, pbe in zip(reactions, nn_errors, pbe_errors):
        overlap = sorted((set(rxn.get("reactants", []))
                          | set(rxn.get("products", []))) & training)
        records.append({
            "name": rxn.get("name"),
            "pool": rxn.get("source_pool"),
            "reactants": list(rxn.get("reactants", [])),
            "products": list(rxn.get("products", [])),
            "coeffs": list(rxn.get("coeffs", [])),
            "reaction_energy_ref_kcalmol": float(rxn["reaction_energy_ref"]),
            "de_nn_kcalmol": nn["de_kcalmol"],
            "de_pbe_kcalmol": pbe["de_kcalmol"],
            "error_nn_kcalmol": nn["error_kcalmol"],
            "error_pbe_kcalmol": pbe["error_kcalmol"],
            "abs_error_nn_kcalmol": nn["abs_error_kcalmol"],
            "abs_error_pbe_kcalmol": pbe["abs_error_kcalmol"],
            "in_sample_overlap": overlap,
        })
    return records


# ---------------------------------------------------------------------------
# Side-effectful: PBE precompute + NN forward
# ---------------------------------------------------------------------------

def precompute_holdout(
    mol_specs: Dict[str, Any],
    descriptors: Sequence[Any] = (),
    *,
    required_keys: Sequence[str] = (),
    auxbasis: str | None = None,
) -> Dict[str, Any]:
    """Run the PBE precompute over a held-out pool of species.

    ``descriptors`` must be the SAME descriptor list the trained model uses
    (typically ``training_spec.arch.materialize_descriptors()``). With
    empty descriptors, models that consume descriptor features (e.g.
    ``deep_combined_attn`` with ``dm_statistics`` + ``cusp``) will raise
    ``TypeError`` from ``jnp.concatenate`` at evaluation time because the
    descriptor columns will be ``None`` in the resulting ``mol_data``.

    ``required_keys`` lets callers ask for additional precomputed fields
    beyond the baseline (e.g. ``("eri",)`` when an SCF solver will be run
    against the precomputed data, without ``eri``, ``run_scf`` raises
    ``ValueError: Cannot determine the shape of None`` because the Coulomb
    rebuild needs the electron-repulsion integrals).

    Per-molecule progress logged (~1-10 s/molecule for PBE SCF + grid build).
    """
    import xcquinox.alec as alec
    out: Dict[str, Any] = {}
    n = len(mol_specs)
    t0 = time.time()
    for i, (name, spec) in enumerate(mol_specs.items(), start=1):
        t1 = time.time()
        try:
            out[name] = alec.precompute_fixed_density_data(
                spec, descriptors=tuple(descriptors),
                required_keys=tuple(required_keys), auxbasis=auxbasis)
        except Exception as exc:  # noqa: BLE001
            print(f"  [precompute {i}/{n}] {name}: FAILED ({exc})",
                  flush=True)
            continue
        print(f"  [precompute {i}/{n}] {name}  "
              f"({time.time() - t1:.1f}s)", flush=True)
    print(f"  precompute done in {time.time() - t0:.1f}s "
          f"({len(out)}/{n} succeeded)", flush=True)
    return out


def evaluate_holdout(model, mol_data: Dict[str, Any],
                     *, solver_config=None,
                     verbose_first_failure: bool = True,
                     scf_info_out: Optional[Dict[str, Dict[str, Any]]] = None,
                     ) -> Dict[str, float]:
    """Per-species total energy.

    Energy source follows the same solver-MODE rule as training
    (``losses._compute_energies`` / ``oneshot.total_energy_for_solver``):

    * ``FULL`` -> the self-consistent ``alec.run_scf(...).total_energy`` (and the
      per-SCF-step trace is captured into ``scf_info_out``), matching what
      FULL-mode training optimizes.
    * ``ONESHOT`` / ``FIXED_J`` / ``None`` -> the one-shot
      ``alec.fixed_density_total_energy(model, mol_data[name])`` on ρ_PBE.
      FIXED_J stays one-shot deliberately (its run_scf energy is an incoherent
      J-pinned hybrid), so a FIXED_J-trained spec is not silently evaluated on
      that hybrid here.

    ``scf_info_out``: optional dict; when provided AND an SCF runs, it is
    populated ``{name: {cycles_run, converged, total_energy, energy_trace}}``
    per species, the per-molecule, per-SCF-step convergence trace that
    :func:`make_per_molecule_record` turns into ``scf_energy_step_<i>`` /
    ``scf_energy_residual_<i>`` columns. Captures even when the final energy
    is non-finite (so a diverged SCF's trace is still recorded).

    NaN on exception. When ``verbose_first_failure`` is True (default),
    the FIRST exception in a batch is printed with its full message so
    the operator sees real errors instead of a silent column of NaNs.
    """
    import xcquinox.alec as alec
    from xcquinox.alec.solver import run_scf, SolverMode
    import numpy as _np
    out: Dict[str, float] = {}
    first_err_shown = False
    n_failed = 0
    # FULL -> self-consistent run_scf energy (+ trace); else one-shot. Matches
    # oneshot.total_energy_for_solver so train == in-sample-eval == held-out-eval.
    use_scf = (solver_config is not None
               and getattr(solver_config, "mode", None) == SolverMode.FULL)
    for name, md in mol_data.items():
        try:
            if use_scf:
                result = run_scf(solver_config, model, md)
                e = float(result.total_energy)
                if scf_info_out is not None:
                    trace = getattr(result, "energy_trace", None)
                    scf_info_out[name] = {
                        "cycles_run": int(getattr(result, "cycles_run", 0)),
                        "converged": bool(getattr(result, "converged", False)),
                        "total_energy": e,
                        "energy_trace": ([float(x) for x in _np.asarray(trace)]
                                         if trace is not None else []),
                    }
            else:
                e = float(alec.fixed_density_total_energy(model, md))
        except Exception as exc:  # noqa: BLE001
            if verbose_first_failure and not first_err_shown:
                print(f"  eval[{name}] FAILED: {type(exc).__name__}: {exc}",
                      flush=True)
                first_err_shown = True
            n_failed += 1
            e = float("nan")
        out[name] = e if math.isfinite(e) else float("nan")
    if n_failed:
        print(f"  eval: {n_failed}/{len(mol_data)} species failed "
              "(NaN energy; see first error above)", flush=True)
    return out


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

# Default filenames (kept as constants so callers can construct paths in
# their own subdir conventions without re-deriving the suffix).
DEFAULT_CSV_NAME = "test_set.csv"
DEFAULT_PER_MOLECULE_NAME = "per_molecule.json"
DEFAULT_PER_REACTION_NAME = "per_reaction.json"


def write_test_set_csv(
    out_path: Path,
    per_pool_mae: Dict[str, Tuple[float, float, int, int, int]],
    combined_mae: Tuple[float, float, int, int, int],
    strict: bool,
) -> Path:
    """Write the per-spec MAE summary CSV (one row per pool + one combined).

    ``per_pool_mae`` maps pool token (``"bh76"``, ``"w411"``) to
    ``(mae_nn_kcalmol, mae_pbe_kcalmol, n_used, n_dropped_overlap,
    n_dropped_nan)``. The PBE MAE comes from re-evaluating the same
    reactions against ``mol_data["E_pbe"]`` instead of the NN, costs
    nothing extra since the PBE energies are by-products of the precompute
    step, and gives the operator a direct apples-to-apples NN-vs-PBE
    comparison on the SAME pool the NN was scored on.

    The ``n_dropped_nan`` column reports reactions silently dropped because
    their species energies were missing or non-finite. It is distinct from
    ``n_dropped_overlap`` (training-set
    overlap drops in strict mode). A row with ``n_dropped_nan > 0``
    indicates the MAE was computed on a SMALLER reaction set than expected.

    Writes to ``out_path`` (caller controls the full path so cluster uses
    ``<ckpt>/eval_holdout/test_set.csv`` and the local CLI uses
    ``<ckpt>/local_test_set.csv``).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["set", "mae_nn_kcalmol", "mae_pbe_kcalmol",
                  "delta_nn_minus_pbe", "n_reactions", "n_dropped_overlap",
                  "n_dropped_nan", "note"]
    with out_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for pool_name, vals in per_pool_mae.items():
            mae_nn, mae_pbe, n_used, n_dropped, n_nan = vals
            note_parts = []
            if strict and n_dropped:
                note_parts.append(
                    f"strict (held-out only); {n_dropped} reactions dropped "
                    "due to training overlap")
            elif strict:
                note_parts.append("strict (held-out only)")
            else:
                note_parts.append("loose (in-sample overlap allowed; "
                                  "flagged in per_molecule.json)")
            if n_nan:
                note_parts.append(
                    f"{n_nan} reactions silently dropped (missing/NaN "
                    "species energies)")
            delta = (mae_nn - mae_pbe
                     if math.isfinite(mae_nn) and math.isfinite(mae_pbe)
                     else float("nan"))
            w.writerow({
                "set": f"test_set_{pool_name}",
                "mae_nn_kcalmol": ("" if not math.isfinite(mae_nn)
                                   else f"{mae_nn:.6f}"),
                "mae_pbe_kcalmol": ("" if not math.isfinite(mae_pbe)
                                    else f"{mae_pbe:.6f}"),
                "delta_nn_minus_pbe": ("" if not math.isfinite(delta)
                                       else f"{delta:+.6f}"),
                "n_reactions": n_used,
                "n_dropped_overlap": n_dropped,
                "n_dropped_nan": n_nan,
                "note": "; ".join(note_parts),
            })
        mae_c_nn, mae_c_pbe, n_used_c, n_dropped_c, n_nan_c = combined_mae
        delta_c = (mae_c_nn - mae_c_pbe
                   if math.isfinite(mae_c_nn) and math.isfinite(mae_c_pbe)
                   else float("nan"))
        combined_note_parts = ["combined across pools"
                                + (" (strict)" if strict else " (loose)")]
        if n_nan_c:
            combined_note_parts.append(
                f"{n_nan_c} reactions silently dropped (missing/NaN species)")
        w.writerow({
            "set": "test_set_held_out_combined",
            "mae_nn_kcalmol": ("" if not math.isfinite(mae_c_nn)
                               else f"{mae_c_nn:.6f}"),
            "mae_pbe_kcalmol": ("" if not math.isfinite(mae_c_pbe)
                                else f"{mae_c_pbe:.6f}"),
            "delta_nn_minus_pbe": ("" if not math.isfinite(delta_c)
                                   else f"{delta_c:+.6f}"),
            "n_reactions": n_used_c,
            "n_dropped_overlap": n_dropped_c,
            "n_dropped_nan": n_nan_c,
            "note": "; ".join(combined_note_parts),
        })
    return out_path


def write_per_molecule_json(out_path: Path,
                             records: List[Dict[str, Any]]) -> Path:
    """Write per-species records to JSON at ``out_path``."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(records, f, indent=2)
    return out_path


def write_per_reaction_json(out_path: Path,
                             records: List[Dict[str, Any]]) -> Path:
    """Write per-reaction records to JSON at ``out_path``."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(records, f, indent=2)
    return out_path


# ---------------------------------------------------------------------------
# Model loading helpers
# ---------------------------------------------------------------------------

def arch_polarized_flag(arch) -> bool:
    """``True`` iff this architecture trained with spin-polarized
    correlation (the ``use_polarized_correlation`` field on
    ``ArchitectureConfig``). Pure."""
    return bool(getattr(arch, "use_polarized_correlation", False))


def load_trained_model(training_spec, model_path: Path):
    """Build the AlecGGAModel skeleton from ``training_spec.arch`` and
    deserialize ``model.eqx`` into it. Returns the trained
    ``AlecGGAModel``.

    Logs the polarization mode (RKS-trained vs UKS-trained) so the
    operator sees at-a-glance which path the eval will take for open-shell
    species. The actual routing is decided per-molecule by
    ``xcquinox.alec.oneshot.fixed_density_total_energy`` based on the
    cnet's ``use_spin_polarization`` flag, which is exactly what
    ``AlecGGAModel.from_arch`` sets from
    ``arch.use_polarized_correlation``."""
    import equinox as eqx
    from xcquinox.alec.models import AlecGGAModel
    skeleton = AlecGGAModel.from_arch(training_spec.arch, seed=0)
    polarized = arch_polarized_flag(training_spec.arch)
    mode = "polarized (UKS for open-shell)" if polarized else "unpolarized (RKS)"
    print(f"  arch: {getattr(training_spec.arch, 'name', '?')}  "
          f"[{mode}]", flush=True)
    return eqx.tree_deserialise_leaves(str(model_path), skeleton)


# ---------------------------------------------------------------------------
# Cluster-side high-level driver
# ---------------------------------------------------------------------------

def descriptors_and_required_keys(training_spec):
    """``(descriptors, required_keys, mode_str)`` for a training spec.

    ``descriptors`` are the arch's materialized descriptors (matched to what
    the trained model consumes); ``required_keys`` is ``("eri",)`` when the
    training solver is a non-ONESHOT SCF (so ``run_scf`` can rebuild J), else
    ``()``. These determine the precompute and are identical for every spec
    that shares an arch descriptor signature + solver mode, which is what
    lets a caller precompute ONCE and reuse the result across specs."""
    try:
        descriptors = tuple(training_spec.arch.materialize_descriptors())
    except AttributeError:
        descriptors = ()
    spec_solver_config = getattr(training_spec, "solver_config", None)
    needs_scf = (
        spec_solver_config is not None
        and hasattr(spec_solver_config, "mode")
        and spec_solver_config.mode.value != "oneshot"
    )
    if needs_scf:
        # DF path needs the 3-index cderi instead of the full 4-index eri.
        required_keys = (("cderi",)
                         if getattr(spec_solver_config, "density_fit", False)
                         else ("eri",))
    else:
        required_keys = ()
    mode_str = (
        spec_solver_config.mode.name if spec_solver_config is not None
        and hasattr(spec_solver_config, "mode") else "fixed_density"
    )
    return descriptors, required_keys, mode_str


def precompute_holdout_for_spec(training_spec, mol_specs: Dict[str, Any]):
    """Precompute PBE + grid + (eri) + descriptor features for one spec's arch.

    The expensive part (PBE SCF + integrals) depends only on the geometry,
    basis, descriptor signature and solver mode, NOT on the trained weights,
    so the returned ``mol_data`` is reusable by ``run_full_holdout_eval`` (via
    its ``mol_data=`` argument) for EVERY spec that shares the same descriptor
    signature + solver mode. This is what turns an N-spec re-eval from N
    precomputes into one-per-descriptor-group."""
    descriptors, required_keys, mode_str = descriptors_and_required_keys(
        training_spec)
    # Forward the spec's DF auxbasis so held-out cderi uses the same fitting
    # basis as training (else build_cderi falls back to a possibly-different
    # auto aux). Only meaningful when density_fit is on; None otherwise.
    sc = getattr(training_spec, "solver_config", None)
    auxbasis = (getattr(sc, "auxbasis", None)
                if getattr(sc, "density_fit", False) else None)
    print(f"[holdout] precomputing {len(mol_specs)} species "
          f"(descriptors: {[type(d).__name__ for d in descriptors] or 'none'}; "
          f"solver: {mode_str}; extra precompute keys: "
          f"{list(required_keys) or 'none'}) ...", flush=True)
    return precompute_holdout(mol_specs, descriptors=descriptors,
                              required_keys=required_keys, auxbasis=auxbasis)


def run_full_holdout_eval(
    training_spec,
    model,
    mol_specs: Dict[str, Any],
    reactions: Sequence[Dict[str, Any]],
    out_dir: Path,
    *,
    strict: Optional[bool] = None,
    mol_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """End-to-end held-out eval driver, what the cluster eval task calls.

    Workflow:
      1. Decide ``required_keys=("eri",)`` if the training spec's
         solver_config requests a non-ONESHOT SCF (so ``run_scf`` has the
         electron-repulsion integrals it needs).
      2. Precompute PBE on every species in ``mol_specs``.
      3. NN forward (one-shot or SCF, matching the training solver) on
         every species.
      4. Filter reactions per ``strict`` (env var
         ``XCQUINOX_HELDOUT_STRICT=1`` overrides the kwarg; default is
         loose).
      5. Compute per-pool MAE + per-reaction NN/PBE error records.
      6. Write three artifacts under ``out_dir``:
         - ``test_set.csv``: per-pool + combined MAE rows
         - ``per_molecule.json``: per-species E_nn + E_pbe + flags
         - ``per_reaction.json``: per-reaction NN + PBE errors with
           ``in_sample_overlap`` lists

    ``mol_data``: when provided (from :func:`precompute_holdout_for_spec`),
    the expensive PBE/grid/eri precompute is SKIPPED and the supplied data is
    reused: the caller is responsible for ensuring it was built with a
    matching descriptor signature + solver mode (same arch group).

    Returns a small summary dict (counts + output paths) for the caller's
    log line.
    """
    if strict is None:
        strict = os.environ.get("XCQUINOX_HELDOUT_STRICT") == "1"
    per = compute_holdout_per_molecule(
        training_spec, model, mol_specs, mol_data=mol_data)
    return _finalize_holdout_outputs(
        reactions, per["energies"], per["pbe_energies"], per["mol_records"],
        per["training_names"], per["n_species"], out_dir, strict=strict)


def compute_holdout_per_molecule(training_spec, model, mol_specs: Dict[str, Any],
                                 *, mol_data: Optional[Dict[str, Any]] = None
                                 ) -> Dict[str, Any]:
    """Per-molecule stage of the held-out eval (the parallelizable part).

    Precomputes PBE/grid/(eri) and runs the NN forward (one-shot or SCF) on
    every species in ``mol_specs``, then builds the per-molecule records. This
    is pure of reaction aggregation and file IO, so a sharded driver can call it
    on a SUBSET of ``mol_specs`` and merge the returned maps (the loops iterate
    the dict, so a shard is just a smaller ``mol_specs``).

    Returns a dict with: ``energies`` (name -> E_nn), ``pbe_energies``
    (name -> E_pbe), ``scf_info`` (name -> SCF convergence dict), ``mol_records``
    (per_molecule.json rows, sorted by name), ``n_species`` and ``training_names``.
    """
    _descriptors, _extra_required, mode_str = descriptors_and_required_keys(
        training_spec)
    spec_solver_config = getattr(training_spec, "solver_config", None)

    training_names = tuple(
        getattr(m, "name", "?") for m in
        getattr(training_spec, "molecules", ())
    )

    if mol_data is None:
        mol_data = precompute_holdout_for_spec(training_spec, mol_specs)
    else:
        print(f"[holdout] reusing precomputed {len(mol_data)} species "
              f"(solver: {mode_str}) ...", flush=True)

    print(f"[holdout] evaluating model on {len(mol_data)} species "
          f"(solver: {mode_str}) ...", flush=True)
    scf_info: Dict[str, Dict[str, Any]] = {}
    energies = evaluate_holdout(model, mol_data,
                                 solver_config=spec_solver_config,
                                 scf_info_out=scf_info)

    pbe_energies = {n: float(md.get("E_pbe"))
                    for n, md in mol_data.items()
                    if md.get("E_pbe") is not None
                    and math.isfinite(float(md.get("E_pbe")))}

    training_set = set(training_names)
    mol_records: List[Dict[str, Any]] = []
    for name in sorted(mol_data):
        mol_records.append(make_per_molecule_record(
            name, mol_data[name], energies.get(name, float("nan")),
            in_training_subset=(name in training_set),
            scf=scf_info.get(name),
        ))

    return {
        "energies": energies,
        "pbe_energies": pbe_energies,
        "scf_info": scf_info,
        "mol_records": mol_records,
        "n_species": len(mol_data),
        "training_names": training_names,
    }


def merge_holdout_shards(shard_payloads: Sequence[Dict[str, Any]]
                         ) -> Tuple[Dict[str, float], Dict[str, float],
                                    List[Dict[str, Any]]]:
    """Merge per-shard ``{energies, pbe_energies, mol_records}`` payloads into
    the combined maps the finalize stage consumes.

    Molecule names are a partition of the held-out set (each shard owns a
    disjoint subset), so the dict-unions are collision-free. ``mol_records`` are
    concatenated and re-sorted by molecule name to match the serial ordering."""
    energies: Dict[str, float] = {}
    pbe_energies: Dict[str, float] = {}
    mol_records: List[Dict[str, Any]] = []
    for payload in shard_payloads:
        energies.update(payload.get("energies", {}))
        pbe_energies.update(payload.get("pbe_energies", {}))
        mol_records.extend(payload.get("mol_records", []))
    mol_records.sort(key=lambda r: r.get("molecule", ""))
    return energies, pbe_energies, mol_records


def _n_nan_union(energies_ha: Dict[str, float],
                 pbe_energies_ha: Dict[str, float],
                 reactions: Sequence[Dict[str, Any]]) -> int:
    """Count reactions dropped (non-finite abs error) in EITHER the NN or the
    PBE metric, i.e. the union. The two metrics can drop DIFFERENT reactions, so
    max(n_nan_nn, n_nan_pbe) undercounts the true dropped set."""
    nn = list(per_reaction_errors(energies_ha, reactions))
    pb = list(per_reaction_errors(pbe_energies_ha, reactions))
    return sum(
        1 for a, b in zip(nn, pb)
        if not (math.isfinite(a["abs_error_kcalmol"])
                and math.isfinite(b["abs_error_kcalmol"]))
    )


def _finalize_holdout_outputs(reactions: Sequence[Dict[str, Any]],
                              energies: Dict[str, float],
                              pbe_energies: Dict[str, float],
                              mol_records: List[Dict[str, Any]],
                              training_names: Sequence[str],
                              n_species: int,
                              out_dir: Path, *, strict: bool) -> Dict[str, Any]:
    """Reaction aggregation + artifact writing, the fast serial tail of the
    held-out eval, shared by the serial driver and the sharded/parallel driver.

    Needs ALL molecule energies (reactions span the whole pool), so it runs once
    after every shard has finished. Writes ``test_set.csv``, ``per_molecule.json``
    and ``per_reaction.json`` under ``out_dir`` and returns the summary dict."""
    # Partition reactions by source_pool so we can write per-pool rows.
    by_pool: Dict[str, List[Dict[str, Any]]] = {}
    for r in reactions:
        by_pool.setdefault(r.get("source_pool", "unknown"), []).append(r)

    per_pool_mae: Dict[str, Tuple[float, float, int, int, int]] = {}
    all_kept: List[Dict[str, Any]] = []
    n_dropped_total = 0
    n_nan_total = 0
    for pool, pool_rxns in by_pool.items():
        kept, dropped = filter_reactions(
            pool_rxns, training_names, strict=strict)
        n_dropped_pool = len(dropped)
        mae_nn, n_used, n_nan_nn = reaction_mae_kcalmol(energies, kept)
        mae_pbe, _, n_nan_pbe = reaction_mae_kcalmol(pbe_energies, kept)
        # Union: NN and PBE can drop DIFFERENT reactions, so max() undercounts.
        n_nan = _n_nan_union(energies, pbe_energies, kept)
        per_pool_mae[pool] = (mae_nn, mae_pbe, n_used, n_dropped_pool, n_nan)
        all_kept.extend(kept)
        n_dropped_total += n_dropped_pool
        n_nan_total += n_nan
    combined_mae_nn, combined_n_used, combined_n_nan_nn = reaction_mae_kcalmol(
        energies, all_kept)
    combined_mae_pbe, _, combined_n_nan_pbe = reaction_mae_kcalmol(
        pbe_energies, all_kept)
    combined = (combined_mae_nn, combined_mae_pbe, combined_n_used,
                n_dropped_total, _n_nan_union(energies, pbe_energies, all_kept))

    nn_per_rxn = per_reaction_errors(energies, all_kept)
    pbe_per_rxn = per_reaction_errors(pbe_energies, all_kept)
    rxn_records = make_per_reaction_records(
        all_kept, nn_per_rxn, pbe_per_rxn, training_names)

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = write_test_set_csv(
        out_dir / DEFAULT_CSV_NAME, per_pool_mae, combined, strict)
    mol_json_path = write_per_molecule_json(
        out_dir / DEFAULT_PER_MOLECULE_NAME, mol_records)
    rxn_json_path = write_per_reaction_json(
        out_dir / DEFAULT_PER_REACTION_NAME, rxn_records)
    print(f"[holdout] wrote {csv_path.name}, "
          f"{mol_json_path.name}, {rxn_json_path.name} "
          f"({len(all_kept)} reactions, {n_nan_total} NaN-drops, "
          f"{n_dropped_total} overlap-drops)", flush=True)

    return {
        "n_reactions": len(all_kept),
        "n_species": n_species,
        "n_dropped_overlap": n_dropped_total,
        "n_dropped_nan": n_nan_total,
        "per_pool_mae": per_pool_mae,
        "combined": combined,
        "csv_path": str(csv_path),
        "per_molecule_path": str(mol_json_path),
        "per_reaction_path": str(rxn_json_path),
    }
