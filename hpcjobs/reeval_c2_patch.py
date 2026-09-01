"""Patch C2 wrong-branch reference values in pulled held-out eval artifacts.

Seven completed specs of run_20260827T163330Z (dfs6311_grid3_v6g1_size)
carry held-out evaluations computed while the reference SCF could land
C2/PBE on the internally unstable branch: their per_molecule records show
E_pbe(c2) = -75.7368945257551 where the stable solution -- and the other
18 completed specs, bit-identical across 72 clean channels -- reads
-75.81674071208121 (+50.10 kcal/mol; density_rmse_pbe 0.000221 vs
0.00278). The reference code is fixed (data.py branch-checked rescue,
bfde6316a); training never used c2, so the checkpoints are valid and only
the c2-derived entries of the eval artifacts need repair.

What one contaminated channel needs, and why
--------------------------------------------
The NN SCF of the standard channels is seeded from the reference PBE
density: data.py:1480-1496 supplies ``dm_seed`` as the SAME array as
``dm_pbe`` for ``seed_source="pbe"``, and the manual solver starts from
``mol_data["dm_seed"]`` unconditionally (solver_manual.py:311, 487;
solver.py:89-90). A wrong-branch reference therefore contaminates the c2
NN energy, SCF trace and NN density metrics of ``eval_holdout`` /
``eval_holdout_best`` / ``eval_holdout_val_best`` along with every
model-free PBE column. The cold-start channel instead seeds minao
(eval_holdout.coldstart_solver_config, applied in
cluster/_eval_one_spec.py:661-673 BEFORE the precompute), and FULL-mode
energies are rebuilt from the live density (h_core + J[D] + E_xc[D] +
e_nuc; solver_manual._compute_total_energy) with no PBE-derived scalar
consumed, so its NN rows are independent of the reference branch: they are
verified against a recompute and left in place, and only the PBE columns
are patched there.

Per patched channel the tool rewrites, in place and nothing else:

* ``per_molecule.json`` -- the c2 record's ``E_pbe``, ``AE_nn``, the
  model-free density sextet (``density_rmse_pbe``, ``density_l1_pbe``,
  ``density_eps_l1_pbe``, ``n_electrons``, ``grid_weight_sum``), and for
  the PBE-seeded channels also ``E_total_nn``, ``density_rmse``,
  ``density_l1``, ``density_eps_l1``, ``ref_density_method``,
  ``cycles_run``, ``scf_converged``, ``scf_total_energy`` and the
  ``scf_energy_step_<i>`` / ``scf_energy_residual_<i>`` trace.
* ``per_reaction.json`` -- the ``w411_c2_atomization`` row's
  ``de_*``/``error_*``/``abs_error_*`` columns, recomputed from the
  patched c2 energies and the RECORDED C-atom energies of the same
  channel (gated: E_pbe(c) must agree across every audited channel).
* ``test_set.csv`` -- the MAE/delta cells of the rows whose reaction set
  contains the c2 reaction (``test_set_w411`` and the combined row),
  recomputed with eval_holdout's exact semantics (``reaction_mae_kcalmol``
  averages finite absolute errors; ``n_reactions`` counts the NN-finite
  rows; ``n_dropped_nan`` is the NN/PBE union); counts, drops and notes
  must reproduce, and the untouched pool rows must reproduce byte-for-byte
  or the patch is refused.
* ``eval_metadata.json`` -- gains a ``reference_patch`` stamp (species,
  date, from/to E_pbe, patched fields); every existing key is preserved.

The repaired reference is recomputed ONCE per seed through the repo's own
path (full_benchmark_pools.load_full_held_out_pools at the run's
basis/grid + data.precompute_fixed_density_data, whose process memo
deduplicates repeat calls) and must land on the stable branch within 1e-6
Ha. The clean pool is NOT one bit-identical value set: it spans several
evaluation generations (measured 2026-08-31: the 72 pre-fix channels
share one c2 sextet, while the post-fix specs 0027/0028 each carry their
own values -- per-evaluation SCF reconvergence slack, all converged). The
four SCF-dependent model-free columns (E_pbe + the PBE density trio) are
therefore written from the LOCAL RECOMPUTE, gated to lie within the
clean-channel envelope widened by 10x each field's measured spread
(MODEL_FREE_SPREAD); the two pure grid quantities (n_electrons,
grid_weight_sum), bit-identical over all 80 clean channels, keep the
exact consensus and refuse on ANY disagreement. All gates run before the
first write: a refusal (exit 2) leaves every file byte-identical; a
write-phase integrity failure is the separate partial-write path (exit 3,
committed channels named -- see the Exit codes section below).

Usage::

    python hpcjobs/reeval_c2_patch.py --run-dir <pulled run dir> \
        [--specs 19,20] [--dry-run] [--bench-refs-dir DIR]
        [--allow-unknown-skip] [--coldstart-tol-e HA]
        [--coldstart-tol-dens X]

The audit is only as current as the local pull: refresh first and re-audit
after the training array drains (the banner prints the exact pull
command). ``--dry-run`` stops after the audit. The benchmark CCSD
reference ``c2.npz`` (cluster
``/gpfs/scratch/awills/external_refs_bench_6311ppg3df2pd_g3``) must be
available locally for the density recompute; the tool prints the fetch
command when it is missing.

Exit codes
----------
* ``0`` -- audit-only completion (``--dry-run``), nothing patchable, or
  every patchable channel patched and integrity-verified.
* ``2`` -- refusal BEFORE any write (a gate fired, a precondition is
  missing, or an artifact is unreadable): every file is byte-identical to
  its pre-run state.
* ``3`` -- write-phase failure: the channels named as committed were
  patched and verified before the failure and are valid; the failing
  channel may be partially rewritten and must be re-pulled from the
  cluster.
* ``4`` -- the audit found channels in an unknown / no-c2 state and
  ``--allow-unknown-skip`` was not given; nothing was written.
"""
from __future__ import annotations

import argparse
import copy
import csv
import dataclasses
import hashlib
import io
import json
import math
import os
import re
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Anchors and identities (measured on run_20260827T163330Z, 2026-08-31)
# ---------------------------------------------------------------------------

SPECIES = "c2"
REACTION_NAME = "w411_c2_atomization"
C_ATOM = "c"

#: Stable-branch E_pbe(c2): bit-identical across the 72 clean channels of
#: the pulled run (18 specs x 4 channels).
GOOD_E_PBE = -75.81674071208121
#: Unstable-branch E_pbe(c2): the value in 27 of 27 contaminated channels.
BAD_E_PBE = -75.7368945257551
#: Classification window around either branch value. The inter-branch gap
#: is 7.98e-2 Ha, so the windows cannot overlap.
CLASSIFY_TOL = 1e-4
#: The recomputed reference must land on the stable branch within this.
REF_GATE_TOL = 1e-6
#: Recorded C-atom E_pbe must agree across channels within this before its
#: energies are reused in the reaction recompute.
C_ATOM_TOL = 1e-6
#: Recorded per-reaction rows must reproduce from the recorded
#: per-molecule energies within this (kcal/mol) before being patched.
REACTION_CONSISTENCY_TOL = 1e-6
#: Measured clean-pool reproducibility of the SCF-dependent model-free
#: fields (fresh pull, 2026-08-31): the clean pool held 80 channels from
#: THREE evaluation generations -- the 72 pre-fix channels share one
#: bit-identical c2 sextet, while the two post-fix specs (0027, 0028)
#: each carry their own micro-different values. Per-evaluation SCF
#: reconvergence slack, not a code split: all fully converged; the fixed
#: rescue's extra macro-iterations wander within the orientation lock's
#: flat direction where the old endpoint happened to be deterministic.
#: Measured max spreads (with the three cluster values where quoted):
#: Constants are the measured spreads rounded UP at 3 s.f. (measured
#: 2.611955e-11 / 2.808558e-9 / 3.401316e-10 / 3.783110e-7), so the band
#: covers the measurement at any BAND_FACTOR >= 1:
#:   E_pbe               2.62e-11 Ha  (-75.81674071208121 / ...207661 /
#:                                     ...210273)
#:   density_rmse_pbe    2.81e-9      (0.00022149606464626117 /
#:                                     ...643534261995 / ...362678433278)
#:   density_l1_pbe      3.41e-10
#:   density_eps_l1_pbe  3.79e-7      (0.01149699623628779 /
#:                                     0.011497374547239803 /
#:                                     0.011497281245159167)
#: n_electrons and grid_weight_sum were EXACTLY single-valued over all 80
#: channels (pure grid quantities).
MODEL_FREE_SPREAD = {
    "E_pbe": 2.62e-11,
    "density_rmse_pbe": 2.81e-9,
    "density_l1_pbe": 3.41e-10,
    "density_eps_l1_pbe": 3.79e-7,
}
#: The local recompute must lie within the clean-channel envelope widened
#: by this factor times the field's measured max spread.
BAND_FACTOR = 10.0

#: CODATA-2018 hartree -> kcal/mol; pinned to eval_holdout.KCAL_PER_HA by a
#: source-text test so the two cannot drift.
KCAL_PER_HA = 627.5094740631

CHANNELS = ("eval_holdout", "eval_holdout_best", "eval_holdout_val_best",
            "eval_holdout_coldstart")
#: Checkpoint evaluated by each channel (cluster/_eval_one_spec.py:539,
#: 631-649, 661-673).
CHANNEL_MODEL = {
    "eval_holdout": "model.eqx",
    "eval_holdout_best": "model_best.eqx",
    "eval_holdout_val_best": "model_val_best.eqx",
    "eval_holdout_coldstart": "model.eqx",
}
PATCH_ARTIFACTS = ("per_molecule.json", "per_reaction.json", "test_set.csv",
                   "eval_metadata.json")

#: Model-free columns of the c2 per-molecule record: functions of the
#: reference PBE density and the fixed CCSD reference only.
MODEL_FREE_FIELDS = ("E_pbe", "density_rmse_pbe", "density_l1_pbe",
                     "density_eps_l1_pbe", "n_electrons", "grid_weight_sum")
#: The four SCF-dependent members: micro-different per evaluation
#: generation (see MODEL_FREE_SPREAD), so the patch writes the LOCAL
#: RECOMPUTE's values, band-gated against the clean-channel envelope.
SCF_DEPENDENT_MODEL_FREE = ("E_pbe", "density_rmse_pbe", "density_l1_pbe",
                            "density_eps_l1_pbe")
#: The two pure grid quantities: bit-identical across every clean channel
#: (80/80 measured), patched from the exact consensus; any disagreement
#: -- in the pool or in the recompute -- is a grid-identity problem.
EXACT_MODEL_FREE = ("n_electrons", "grid_weight_sum")
#: NN scalar columns recomputed for the PBE-seeded channels.
NN_SCALAR_FIELDS = ("E_total_nn", "density_rmse", "density_l1",
                    "density_eps_l1", "ref_density_method", "cycles_run",
                    "scf_converged", "scf_total_energy")
#: Cold-start verification: recomputed-vs-recorded NN values.
COLDSTART_VERIFY_E_FIELDS = ("E_total_nn", "scf_total_energy")
COLDSTART_VERIFY_DENS_FIELDS = ("density_rmse", "density_l1",
                                "density_eps_l1")

_TRACE_KEY_RE = re.compile(r"^scf_energy_(?:step|residual)_(\d+)$")

CSV_FIELDNAMES = ["set", "mae_nn_kcalmol", "mae_pbe_kcalmol",
                  "delta_nn_minus_pbe", "n_reactions", "n_dropped_overlap",
                  "n_dropped_nan", "note"]


class PatchRefused(RuntimeError):
    """A gate failed; nothing has been written."""


class PartialWriteError(RuntimeError):
    """A write-phase failure AFTER one or more channels were committed.

    Distinct from :class:`PatchRefused` because the byte-identical claim
    no longer holds: ``committed`` lists the :class:`ChannelAudit` entries
    whose four artifacts were written and integrity-verified before the
    failure, ``failed`` names the channel whose write did not verify (its
    artifacts may be partially rewritten and must be re-pulled)."""

    def __init__(self, message, committed, failed):
        super().__init__(message)
        self.committed = committed
        self.failed = failed


# ---------------------------------------------------------------------------
# Small IO helpers
# ---------------------------------------------------------------------------

def _read_json(path: Path):
    with open(path) as f:
        return json.load(f)


def _read_json_refusing(path: Path, what: str):
    """Parsed JSON, or a :class:`PatchRefused` naming the file -- an
    unreadable artifact is an operator-visible refusal, not a traceback."""
    try:
        return _read_json(path)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        raise PatchRefused(
            f"unreadable {what}: {path} ({exc}); re-pull it from the "
            "cluster before running this tool.") from exc


def _write_json_atomic(path: Path, obj, *, sort_keys: bool = False) -> None:
    """Serialize exactly as the eval writers do (indent=2, no trailing
    newline; eval_metadata.json additionally sort_keys=True) via a same-dir
    temporary and an atomic replace."""
    data = json.dumps(obj, indent=2, sort_keys=sort_keys)
    _write_text_atomic(path, data)


def _write_text_atomic(path: Path, text: str) -> None:
    import tempfile
    fd, tmp = tempfile.mkstemp(dir=str(path.parent),
                               prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", newline="") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot_channel(channel_dir: Path) -> dict:
    """``{relative path: sha256}`` for every file under the channel dir."""
    out = {}
    for p in sorted(Path(channel_dir).rglob("*")):
        if p.is_file():
            out[str(p.relative_to(channel_dir))] = _sha256(p)
    return out


def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _finite(x) -> bool:
    return _is_num(x) and math.isfinite(float(x))


# ---------------------------------------------------------------------------
# Classification + audit
# ---------------------------------------------------------------------------

def _row_by_molecule(pm_rows, name):
    for r in pm_rows:
        if r.get("molecule") == name:
            return r
    return None


def classify_rows(pm_rows):
    """``(state, e_pbe)`` for one channel's per_molecule rows.

    States: ``clean`` (within :data:`CLASSIFY_TOL` of the stable branch),
    ``wrong`` (within it of the unstable branch), ``no-c2`` (no c2 row),
    ``unknown`` (neither branch; not patchable -- an unknown state cannot
    be repaired by construction).
    """
    row = _row_by_molecule(pm_rows, SPECIES)
    if row is None:
        return "no-c2", None
    e = row.get("E_pbe")
    if not _finite(e):
        return "unknown", e
    e = float(e)
    if abs(e - GOOD_E_PBE) <= CLASSIFY_TOL:
        return "clean", e
    if abs(e - BAD_E_PBE) <= CLASSIFY_TOL:
        return "wrong", e
    return "unknown", e


@dataclasses.dataclass
class ChannelAudit:
    spec: int
    channel: str
    path: Path
    state: str                 # clean / wrong / unknown / no-c2 / no-artifacts
    e_pbe: float | None
    model_file: str
    model_present: bool
    failed: bool               # failure.json present in the channel dir
    stamped: bool              # eval_metadata.json carries reference_patch
    pending_fetch: bool        # wrong but checkpoint absent locally
    patchable: bool


def _spec_index(name: str) -> int | None:
    m = re.fullmatch(r"spec_(\d+)", name)
    return int(m.group(1)) if m else None


def audit_run(run_dir, specs=None):
    """One :class:`ChannelAudit` per (spec dir, existing channel dir).

    Every spec dir present under ``<run_dir>/checkpoints`` that has at
    least one eval channel is audited; ``specs`` optionally restricts to
    the named indices. The contaminated set is a moving target -- the
    train array kept finishing specs after the last pull -- so no index
    list is baked in.
    """
    run_dir = Path(run_dir)
    rows = []
    ckpt = run_dir / "checkpoints"
    for sd in sorted(ckpt.glob("spec_*")):
        idx = _spec_index(sd.name)
        if idx is None or (specs is not None and idx not in set(specs)):
            continue
        for channel in CHANNELS:
            ch = sd / channel
            if not ch.is_dir():
                continue
            model_file = CHANNEL_MODEL[channel]
            model_present = (sd / model_file).is_file()
            failed = (ch / "failure.json").is_file()
            pm_path = ch / "per_molecule.json"
            artifacts = all((ch / a).is_file() for a in PATCH_ARTIFACTS)
            stamped = False
            if (ch / "eval_metadata.json").is_file():
                try:
                    meta = _read_json(ch / "eval_metadata.json")
                    stamped = isinstance(meta, dict) and \
                        "reference_patch" in meta
                except (json.JSONDecodeError, OSError):
                    stamped = False
            if not pm_path.is_file() or not artifacts:
                rows.append(ChannelAudit(
                    spec=idx, channel=channel, path=ch, state="no-artifacts",
                    e_pbe=None, model_file=model_file,
                    model_present=model_present, failed=failed,
                    stamped=stamped, pending_fetch=False, patchable=False))
                continue
            state, e = classify_rows(_read_json_refusing(
                pm_path, f"per_molecule.json for spec {idx} {channel}"))
            pending = (state == "wrong") and not model_present
            patchable = (state == "wrong") and model_present and not stamped
            rows.append(ChannelAudit(
                spec=idx, channel=channel, path=ch, state=state, e_pbe=e,
                model_file=model_file, model_present=model_present,
                failed=failed, stamped=stamped, pending_fetch=pending,
                patchable=patchable))
    return rows


def _read_output_root(run_dir) -> str | None:
    """``inputs.output_root`` from resolved_config.yaml by a plain text
    read (no config import, so the audit stays light)."""
    cfg = Path(run_dir) / "resolved_config.yaml"
    if not cfg.is_file():
        return None
    m = re.search(r"^\s*output_root:\s*(\S+)\s*$", cfg.read_text(),
                  flags=re.MULTILINE)
    return m.group(1) if m else None


def _category_of(run_dir) -> str | None:
    """The pull ``--category`` for this run, derived from the recorded
    ``output_root`` (``.../xcquinox_runs/<category head>`` + ``/runs``)."""
    root = _read_output_root(run_dir)
    if not root:
        return None
    parts = Path(root).parts
    if "xcquinox_runs" in parts:
        tail = parts[parts.index("xcquinox_runs") + 1:]
        if tail:
            return "/".join(tail) + "/runs"
    return None


def _pull_refresh_command(run_dir) -> str:
    cat = _category_of(run_dir) or "<category>"
    return (f"python -m xcquinox.alec.cluster pull {Path(run_dir).name} "
            f"--category {cat}")


def _fetch_command(run_dir, pending_specs) -> str:
    cat = _category_of(run_dir) or "<category>"
    speclist = ",".join(str(s) for s in sorted(set(pending_specs)))
    return (f"python -m xcquinox.alec.cluster pull {Path(run_dir).name} "
            f"--category {cat} --profile full --specs {speclist}")


def format_audit_table(rows, run_dir) -> str:
    lines = []
    lines.append(f"{'spec':>5}  {'channel':<24} {'state':<13} "
                 f"{'E_pbe(c2)':>18}  notes")
    for r in sorted(rows, key=lambda x: (x.spec, CHANNELS.index(x.channel))):
        notes = []
        if r.patchable:
            notes.append("PATCHABLE")
        if r.pending_fetch:
            notes.append(f"PENDING-FETCH ({r.model_file} not pulled)")
        if r.failed:
            notes.append("failure.json present")
        if r.stamped:
            notes.append("already stamped")
        if r.state == "wrong" and not r.patchable and not r.pending_fetch \
                and not r.stamped:
            notes.append("UNPATCHABLE-LOCALLY")
        e = f"{r.e_pbe:.10f}" if r.e_pbe is not None else "-"
        lines.append(f"{r.spec:>5}  {r.channel:<24} {r.state:<13} "
                     f"{e:>18}  {'; '.join(notes)}")
    pending = sorted({r.spec for r in rows if r.pending_fetch})
    if pending:
        lines.append("")
        lines.append("Channels marked PENDING-FETCH need the best-loss "
                     "checkpoint, which the default pull omits.")
        lines.append("Ordering: PUSH any locally patched channels to the "
                     "cluster FIRST -- this fetch mirrors the "
                     "eval_holdout*/ dirs and would overwrite locally "
                     "patched channels with the cluster's wrong copies. "
                     "Then fetch:")
        lines.append("  " + _fetch_command(run_dir, pending))
        lines.append("then rerun this tool for those channels.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Reaction math (mirrors eval_holdout.per_reaction_errors semantics)
# ---------------------------------------------------------------------------

def _reaction_names(row):
    return list(row.get("reactants", [])) + list(row.get("products", []))


def _reaction_de_kcalmol(row, energies):
    names = _reaction_names(row)
    coeffs = list(row.get("coeffs", []))
    es = [energies.get(n) for n in names]
    if len(names) != len(coeffs) or any(e is None or not _finite(e)
                                        for e in es):
        return float("nan")
    return sum(c * e for c, e in zip(coeffs, es)) * KCAL_PER_HA


def _reaction_contains_species(row, species=SPECIES) -> bool:
    return any(str(n).casefold() == species
               for n in _reaction_names(row))


def _energy_maps(pm_rows):
    e_nn = {r["molecule"]: r.get("E_total_nn") for r in pm_rows}
    e_pbe = {r["molecule"]: r.get("E_pbe") for r in pm_rows}
    return e_nn, e_pbe


# ---------------------------------------------------------------------------
# Patched-artifact builders (pure)
# ---------------------------------------------------------------------------

def _trace_pairs(record) -> dict:
    """The ``scf_energy_step_<i>`` / ``scf_energy_residual_<i>`` keys of a
    record, in the canonical interleaved order make_per_molecule_record
    emits."""
    idx = sorted({int(m.group(1)) for k in record
                  if (m := _TRACE_KEY_RE.match(str(k)))})
    out = {}
    for i in idx:
        for kind in ("step", "residual"):
            k = f"scf_energy_{kind}_{i}"
            if k in record:
                out[k] = record[k]
    return out


def build_patched_per_molecule(pm_rows, new_fields, new_trace=None):
    """Rows with the c2 record's ``new_fields`` replaced; when
    ``new_trace`` is given the old trace keys are dropped and the new ones
    appended (they sit last in the writer's key order). Non-c2 rows pass
    through by identity."""
    out = []
    for r in pm_rows:
        if r.get("molecule") != SPECIES:
            out.append(r)
            continue
        row = {}
        for k, v in r.items():
            if new_trace is not None and _TRACE_KEY_RE.match(str(k)):
                continue
            row[k] = new_fields[k] if k in new_fields else v
        for k, v in new_fields.items():
            if k not in row and not _TRACE_KEY_RE.match(str(k)):
                row[k] = v
        if new_trace is not None:
            row.update(new_trace)
        out.append(row)
    return out


def build_patched_per_reaction(pr_rows, pm_rows_patched, *, patch_nn):
    """Rows with every c2-containing reaction's de/error/abs columns
    recomputed from the PATCHED per-molecule energies (the C-atom energies
    are the recorded ones -- only c2's entries changed). ``patch_nn=False``
    (cold-start) leaves the NN trio untouched."""
    e_nn, e_pbe = _energy_maps(pm_rows_patched)
    out = []
    for r in pr_rows:
        if not _reaction_contains_species(r):
            out.append(r)
            continue
        row = dict(r)
        ref = float(r["reaction_energy_ref_kcalmol"])
        de_pbe = _reaction_de_kcalmol(r, e_pbe)
        err_pbe = de_pbe - ref
        row["de_pbe_kcalmol"] = de_pbe
        row["error_pbe_kcalmol"] = err_pbe
        row["abs_error_pbe_kcalmol"] = (abs(err_pbe)
                                        if math.isfinite(err_pbe)
                                        else float("nan"))
        if patch_nn:
            de_nn = _reaction_de_kcalmol(r, e_nn)
            err_nn = de_nn - ref
            row["de_nn_kcalmol"] = de_nn
            row["error_nn_kcalmol"] = err_nn
            row["abs_error_nn_kcalmol"] = (abs(err_nn)
                                           if math.isfinite(err_nn)
                                           else float("nan"))
        out.append(row)
    return out


def _fmt_mae(x) -> str:
    return "" if not math.isfinite(x) else f"{x:.6f}"


def _fmt_delta(x) -> str:
    return "" if not math.isfinite(x) else f"{x:+.6f}"


def _pool_stats(rows):
    """(mae_nn, mae_pbe, n_used_nn, n_nan_union) with
    eval_holdout.reaction_mae_kcalmol / _n_nan_union semantics on stored
    per-reaction rows."""
    nn = [float(r["abs_error_nn_kcalmol"]) for r in rows
          if _finite(r.get("abs_error_nn_kcalmol"))]
    pbe = [float(r["abs_error_pbe_kcalmol"]) for r in rows
           if _finite(r.get("abs_error_pbe_kcalmol"))]
    mae_nn = sum(nn) / len(nn) if nn else float("nan")
    mae_pbe = sum(pbe) / len(pbe) if pbe else float("nan")
    n_nan = sum(1 for r in rows
                if not (_finite(r.get("abs_error_nn_kcalmol"))
                        and _finite(r.get("abs_error_pbe_kcalmol"))))
    return mae_nn, mae_pbe, len(nn), n_nan


def recompute_test_set_csv(old_text: str, pr_rows_patched) -> str:
    """The CSV with the MAE/delta cells of the affected rows recomputed
    from the patched per-reaction rows.

    Column semantics reproduce eval_holdout.write_test_set_csv /
    _finalize_holdout_outputs: per-pool rows are ``test_set_<pool>`` over
    the rows of that pool, the combined row is over all rows;
    ``n_reactions`` is the count of NN-finite rows, ``n_dropped_nan`` the
    NN/PBE union of non-finite rows; ``n_dropped_overlap`` and ``note``
    are not recomputable from the kept rows and are copied through.
    """
    reader = csv.DictReader(io.StringIO(old_text))
    if reader.fieldnames != CSV_FIELDNAMES:
        raise PatchRefused(
            f"test_set.csv header {reader.fieldnames} does not match the "
            f"writer's fieldnames {CSV_FIELDNAMES}; refusing to rewrite a "
            "CSV whose schema is not the one the recomputation reproduces.")
    old_rows = list(reader)
    out = io.StringIO()
    w = csv.DictWriter(out, fieldnames=CSV_FIELDNAMES)
    w.writeheader()
    for old in old_rows:
        set_name = old["set"]
        if set_name == "test_set_held_out_combined":
            subset = list(pr_rows_patched)
        elif set_name.startswith("test_set_"):
            pool = set_name[len("test_set_"):]
            subset = [r for r in pr_rows_patched if r.get("pool") == pool]
        else:
            raise PatchRefused(
                f"unrecognized test_set.csv row {set_name!r}; the "
                "recomputation only reproduces test_set_<pool> and "
                "test_set_held_out_combined rows.")
        mae_nn, mae_pbe, n_used, n_nan = _pool_stats(subset)
        delta = (mae_nn - mae_pbe
                 if math.isfinite(mae_nn) and math.isfinite(mae_pbe)
                 else float("nan"))
        new = dict(old)
        new["mae_nn_kcalmol"] = _fmt_mae(mae_nn)
        new["mae_pbe_kcalmol"] = _fmt_mae(mae_pbe)
        new["delta_nn_minus_pbe"] = _fmt_delta(delta)
        new["n_reactions"] = str(n_used)
        new["n_dropped_nan"] = str(n_nan)
        # The patch changes values, never row membership or finiteness, so
        # the recomputed counts must reproduce the recorded ones exactly; a
        # mismatch means the recomputation does not carry the writer's
        # semantics and nothing may be rewritten from it.
        for count_col in ("n_reactions", "n_dropped_nan"):
            if new[count_col] != old[count_col]:
                raise PatchRefused(
                    f"test_set.csv row {set_name!r}: recomputed "
                    f"{count_col}={new[count_col]} but the recorded row "
                    f"says {old[count_col]}; the aggregate semantics do "
                    "not reproduce, so the CSV is not rewritten.")
        # Rows whose reaction subset does not contain the patched species
        # must reproduce the recorded cells bit-for-bit -- the internal
        # oracle that the recomputation equals the writer that produced
        # the file.
        if not any(_reaction_contains_species(r) for r in subset):
            for col in ("mae_nn_kcalmol", "mae_pbe_kcalmol",
                        "delta_nn_minus_pbe"):
                if new[col] != old[col]:
                    raise PatchRefused(
                        f"test_set.csv row {set_name!r} contains no "
                        f"{SPECIES!r} reaction, yet the recomputed {col} "
                        f"({new[col]}) differs from the recorded value "
                        f"({old[col]}); the aggregate recomputation does "
                        "not reproduce the writer and nothing is "
                        "rewritten.")
        w.writerow(new)
    return out.getvalue()


def stamp_metadata(meta: dict, *, from_e_pbe, to_e_pbe, fields) -> dict:
    out = copy.deepcopy(meta)
    out["reference_patch"] = {
        "species": SPECIES,
        "reaction": REACTION_NAME,
        "date": time.strftime("%Y-%m-%d"),
        "from_E_pbe": from_e_pbe,
        "to_E_pbe": to_e_pbe,
        "fields": sorted(fields),
        "tool": "hpcjobs/reeval_c2_patch.py",
    }
    return out


# ---------------------------------------------------------------------------
# Compute seams (replaced by stubs in tests; real SCF only in production)
# ---------------------------------------------------------------------------

def _route_jax_env():
    """fp64 + CPU + shard-worker parity env BEFORE any jax import.

    Mirrors cluster/_eval_one_spec._route_jax_env (JAX_ENABLE_X64=1,
    JAX_PLATFORMS=cpu via setdefault) plus the shard workers' thread/XLA
    pinning (workers/eval_holdout_worker.py:96-108: XLA_FLAGS setdefault
    with the llvm/optimization trims, one BLAS thread per process), so the
    recompute runs on the same numerical footing the recorded values came
    from. Must be the FIRST statement of main(): _load_cfg and
    _solver_config_for_channel import xcquinox modules that pull JAX in
    transitively, and a JAX backend initialized before this routing would
    ignore it (measured: default backend gpu with JAX_PLATFORMS unset at
    that point). The single BLAS thread makes the reference SCF slower
    than a free-threaded run; that is the worker footing the recorded
    values were produced under.
    """
    os.environ["JAX_ENABLE_X64"] = "1"
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault(
        "XLA_FLAGS",
        "--xla_llvm_disable_expensive_passes=true "
        "--xla_backend_optimization_level=1")
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"


def _load_cfg(run_dir):
    from xcquinox.alec.cluster.grid_config import load_grid_config
    return load_grid_config(os.path.join(str(run_dir),
                                         "resolved_config.yaml"))


def _arch_for_cell(cfg, cell):
    """The cell's architecture, rebuilt exactly as
    cluster/spec_builder.build_training_specs does (registry arch +
    run-level polarized-correlation toggle + model block). The class
    record beside each checkpoint verifies the rebuild on load."""
    from xcquinox.alec.config import apply_model_block, get_architecture
    arch = get_architecture(cell["arch"])
    if getattr(cfg, "use_polarized_correlation", False):
        arch = dataclasses.replace(arch, use_polarized_correlation=True)
    model_block = getattr(cfg, "model", None)
    if model_block is not None:
        arch = apply_model_block(arch, model_block)
    return arch


def _solver_config_for_channel(cfg, cell, channel):
    """The channel's SolverConfig, rebuilt exactly as
    cluster/spec_builder.build_training_specs does, with the cold-start
    channel transformed by eval_holdout.coldstart_solver_config -- the
    same single source of truth the eval task applied. The rebuilt
    ``describe()`` is gated against the channel's recorded
    eval_metadata.json before any recompute."""
    from xcquinox.alec.cluster.spec_builder import (_solver_config_from_named,
                                                    resolve_seed_xc)
    sc = _solver_config_from_named(
        cfg.solvers[cell["solver"]],
        density_fit=cfg.inputs.density_fit,
        auxbasis=cfg.inputs.auxbasis,
        orientation_lock_strength=cfg.inputs.orientation_lock_strength,
        seed_source=resolve_seed_xc(cfg.inputs, cell["arch"]),
        seed_cache_dir=getattr(cfg.inputs, "seed_cache_dir", None))
    if channel == "eval_holdout_coldstart":
        from xcquinox.alec.eval_holdout import coldstart_solver_config
        sc = coldstart_solver_config(sc)
    return sc


def _load_model_for_channel(cfg, cell, model_path):
    """The channel's trained model through the class-record loader the
    evals use (eval_holdout.load_trained_model ->
    checkpoint_class.require_matching_class + tree_deserialise_leaves)."""
    from types import SimpleNamespace
    from xcquinox.alec.eval_holdout import load_trained_model
    shim = SimpleNamespace(arch=_arch_for_cell(cfg, cell))
    return load_trained_model(shim, Path(model_path))


def _recompute_reference(cfg, cell, sc, bench_refs_dir):
    """c2's MoleculeData at the exact eval identity, through the repo's own
    path: the pool loader at the run's basis/grid with the benchmark refs
    wired, then data.precompute_fixed_density_data (the branch-checked
    code) with the same arguments eval_holdout.precompute_holdout_for_spec
    passes. The process memo inside precompute_fixed_density_data
    deduplicates repeat calls with the same identity, so calling this per
    channel costs one SCF per distinct (descriptors, seed) pair.

    ``seed_cache_dir`` is forwarded verbatim; data.py:1487-1496 reads the
    cache only for seed_source='scan', so the cluster-only path is inert
    for the pbe/minao seeds this campaign uses."""
    _route_jax_env()
    from types import SimpleNamespace
    import xcquinox.alec as alec
    from xcquinox.alec.eval_holdout import descriptors_and_required_keys
    from xcquinox.alec.full_benchmark_pools import load_full_held_out_pools
    inputs = getattr(cfg, "inputs", None)
    basis = getattr(inputs, "basis", None) or "def2-svp"
    grid_level = getattr(inputs, "grid_level", None)
    grid_level = 1 if grid_level is None else int(grid_level)
    mol_specs, _rxns = load_full_held_out_pools(
        basis=basis, grid_level=grid_level, refs_dir=str(bench_refs_dir))
    if SPECIES not in mol_specs:
        raise PatchRefused(
            f"the held-out pool at basis={basis!r} grid_level={grid_level} "
            f"names no {SPECIES!r} species; the eval identity cannot be "
            "reproduced.")
    spec = mol_specs[SPECIES]
    if not getattr(spec, "external_data_path", None):
        raise PatchRefused(
            f"no benchmark CCSD reference wired for {SPECIES!r} "
            f"({bench_refs_dir}/{SPECIES}.npz): the density metrics cannot "
            "be recomputed at the eval identity.")
    shim = SimpleNamespace(arch=_arch_for_cell(cfg, cell), solver_config=sc)
    descriptors, required_keys, _mode = descriptors_and_required_keys(shim)
    return alec.precompute_fixed_density_data(
        spec,
        descriptors=tuple(descriptors),
        required_keys=tuple(required_keys),
        auxbasis=(getattr(sc, "auxbasis", None)
                  if getattr(sc, "density_fit", False) else None),
        orientation_lock_strength=getattr(sc, "orientation_lock_strength",
                                          0.0),
        seed_source=getattr(sc, "seed_source", "pbe"),
        seed_cache_dir=getattr(sc, "seed_cache_dir", None),
        seed_density_fit=bool(getattr(sc, "density_fit", False)))


def _nn_record_for_channel(model, md, sc):
    """A fresh c2 per-molecule record through the eval's own helpers
    (evaluate_holdout -> run_scf; density_errors_for_record;
    make_per_molecule_record), so the recomputed fields carry exactly the
    semantics the channel's original evaluation used."""
    from xcquinox.alec.eval_holdout import (density_errors_for_record,
                                            evaluate_holdout,
                                            make_per_molecule_record)
    scf_info = {}
    energies = evaluate_holdout(model, {SPECIES: md}, solver_config=sc,
                                scf_info_out=scf_info)
    e_nn = energies.get(SPECIES, float("nan"))
    if not math.isfinite(e_nn):
        raise PatchRefused(
            f"the NN recompute for {SPECIES!r} produced a non-finite "
            f"energy ({scf_info.get(SPECIES)}); nothing is patched from a "
            "failed evaluation.")
    dens = density_errors_for_record(model, md, solver_config=sc)
    return make_per_molecule_record(
        SPECIES, md, e_nn, in_training_subset=False,
        scf=scf_info.get(SPECIES), density=dens)


# ---------------------------------------------------------------------------
# Patch planning
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class ChannelPlan:
    audit: ChannelAudit
    pm_rows: list
    pr_rows: list
    csv_text: str
    meta: dict
    old_c2: dict
    new_pm_rows: list = None
    new_pr_rows: list = None
    new_csv_text: str = None
    new_meta: dict = None
    patched_fields: list = None
    coldstart: bool = False


def _load_channel(audit: ChannelAudit) -> ChannelPlan:
    ch = audit.path
    what = f"for spec {audit.spec} {audit.channel}"
    pm_rows = _read_json_refusing(ch / "per_molecule.json",
                                  f"per_molecule.json {what}")
    pr_rows = _read_json_refusing(ch / "per_reaction.json",
                                  f"per_reaction.json {what}")
    try:
        csv_text = (ch / "test_set.csv").read_text()
    except (OSError, UnicodeDecodeError) as exc:
        raise PatchRefused(
            f"unreadable test_set.csv {what}: {ch / 'test_set.csv'} "
            f"({exc}); re-pull it from the cluster before running this "
            "tool.") from exc
    meta = _read_json_refusing(ch / "eval_metadata.json",
                               f"eval_metadata.json {what}")
    old_c2 = _row_by_molecule(pm_rows, SPECIES)
    coldstart = bool(meta.get("coldstart"))
    return ChannelPlan(audit=audit, pm_rows=pm_rows, pr_rows=pr_rows,
                       csv_text=csv_text, meta=meta, old_c2=old_c2,
                       coldstart=coldstart)


def _plan_patch(plan: ChannelPlan, fresh: dict, pool: dict) -> None:
    """Fill the plan's new_* artifacts. ``fresh`` is the recomputed c2
    record; ``pool`` is the clean-channel evidence from
    :func:`_collect_consensus`. The SCF-dependent model-free fields are
    written from the LOCAL RECOMPUTE (already band-gated against the
    clean envelope); the grid pair is written from the exact consensus.
    A field with no clean-channel evidence, or no recomputed value, must
    not overwrite a finite recorded one."""
    old = plan.old_c2
    new_fields = {}
    for k in MODEL_FREE_FIELDS:
        if k in EXACT_MODEL_FREE:
            v = pool["exact"].get(k)
        else:
            v = (fresh.get(k)
                 if pool["envelope"].get(k) is not None else None)
        if v is None and _finite(old.get(k)):
            raise PatchRefused(
                f"spec {plan.audit.spec} {plan.audit.channel}: no clean-"
                f"channel evidence (or no recomputed value) for {k} while "
                f"the recorded value ({old.get(k)!r}) is finite; a null "
                "cannot overwrite a finite recorded value.")
        new_fields[k] = v
    new_trace = None
    if plan.coldstart:
        # Recorded NN values stand (verified against the recompute by the
        # caller); only the PBE columns and the NN-PBE gap are patched.
        e_nn = old.get("E_total_nn")
        new_fields["AE_nn"] = (float(e_nn) - float(new_fields["E_pbe"])
                               if _finite(e_nn) else old.get("AE_nn"))
    else:
        for k in NN_SCALAR_FIELDS:
            new_fields[k] = fresh.get(k)
        new_fields["AE_nn"] = fresh.get("AE_nn")
        new_trace = _trace_pairs(fresh)
    plan.new_pm_rows = build_patched_per_molecule(
        plan.pm_rows, new_fields, new_trace=new_trace)
    plan.new_pr_rows = build_patched_per_reaction(
        plan.pr_rows, plan.new_pm_rows, patch_nn=not plan.coldstart)
    plan.new_csv_text = recompute_test_set_csv(plan.csv_text,
                                               plan.new_pr_rows)
    fields = sorted(set(new_fields) | ({"scf_energy_trace"}
                                       if new_trace else set()))
    plan.patched_fields = fields
    plan.new_meta = stamp_metadata(
        plan.meta, from_e_pbe=old.get("E_pbe"),
        to_e_pbe=new_fields["E_pbe"], fields=fields)


def _write_plan(plan: ChannelPlan) -> dict:
    """Write the four artifacts atomically; return the change table
    ``{relpath: 'changed'|'unchanged'}`` from the pre/post snapshots."""
    ch = plan.audit.path
    before = snapshot_channel(ch)
    _write_json_atomic(ch / "per_molecule.json", plan.new_pm_rows)
    _write_json_atomic(ch / "per_reaction.json", plan.new_pr_rows)
    _write_text_atomic(ch / "test_set.csv", plan.new_csv_text)
    _write_json_atomic(ch / "eval_metadata.json", plan.new_meta,
                       sort_keys=True)
    after = snapshot_channel(ch)
    allowed = set(PATCH_ARTIFACTS)
    changed = {p for p in before if before[p] != after.get(p)}
    added = set(after) - set(before)
    removed = set(before) - set(after)
    if added or removed or not changed <= allowed:
        raise PatchRefused(
            f"integrity violation in {ch}: changed={sorted(changed)} "
            f"added={sorted(added)} removed={sorted(removed)}; only "
            f"{sorted(allowed)} may change. The channel must be re-pulled "
            "from the cluster before any further patching.")
    return {p: ("changed" if p in changed else "unchanged")
            for p in sorted(before)}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def _parse_specs(text):
    if text is None:
        return None
    out = []
    for tok in text.split(","):
        tok = tok.strip()
        if tok:
            out.append(int(tok))
    return out or None


def _print(msg=""):
    print(msg, flush=True)


def _collect_consensus(run_dir, audit_rows):
    """Model-free evidence from the CLEAN channels' c2 rows.

    Returns ``(pool, n_channels)`` with ``pool = {"envelope": {field:
    (lo, hi) | None}, "exact": {field: value | None}}``, or ``(None, 0)``
    when no clean channel exists. The SCF-dependent fields
    (:data:`SCF_DEPENDENT_MODEL_FREE`) span an envelope because the clean
    pool holds several evaluation generations with per-evaluation SCF
    reconvergence slack (see :data:`MODEL_FREE_SPREAD`); the pure grid
    pair (:data:`EXACT_MODEL_FREE`) must be single-valued (bit-identical
    over all 80 measured clean channels) and refuses otherwise."""
    values = {k: [] for k in MODEL_FREE_FIELDS}
    n = 0
    for r in audit_rows:
        if r.state != "clean":
            continue
        row = _row_by_molecule(
            _read_json_refusing(
                r.path / "per_molecule.json",
                f"per_molecule.json for spec {r.spec} {r.channel}"),
            SPECIES)
        if row is None:
            continue
        n += 1
        for k in MODEL_FREE_FIELDS:
            values[k].append(row.get(k))
    if n == 0:
        return None, 0
    pool = {"envelope": {}, "exact": {}}
    for k, vals in values.items():
        finite = [float(v) for v in vals if _finite(v)]
        if k in EXACT_MODEL_FREE:
            # Pure grid quantities are bit-identical across every clean
            # channel (80/80 measured); any pool-internal disagreement is
            # a grid-identity problem and refuses.
            if not finite:
                pool["exact"][k] = None
                continue
            if max(finite) != min(finite):
                raise PatchRefused(
                    f"clean channels disagree on the grid quantity {k}: "
                    f"spread {max(finite) - min(finite):g} over {n} "
                    "channels where bit-identity is the measured norm "
                    "(80/80); a grid-identity problem must be resolved "
                    "before any patch.")
            pool["exact"][k] = finite[0]
        else:
            # SCF-dependent fields legitimately differ per evaluation
            # generation (see MODEL_FREE_SPREAD); the clean pool defines
            # an ENVELOPE, not a single value.
            pool["envelope"][k] = ((min(finite), max(finite))
                                   if finite else None)
    return pool, n


def _push_sheet(run_dir, patched) -> str:
    """The exact push commands for the user (never run here): one rsync
    per patched channel dir, artifacts only (_shards excluded)."""
    root = _read_output_root(run_dir)
    run_name = Path(run_dir).name
    dest = (f"{root}/runs/{run_name}/checkpoints" if root
            else "<cluster run dir>/checkpoints")
    lines = []
    lines.append("# Push the patched channels back to the cluster run dir:")
    lines.append(f'RUNC={dest}')
    lines.append(f'LOC={Path(run_dir)}/checkpoints')
    for audit in patched:
        rel = f"spec_{audit.spec:04d}/{audit.channel}"
        lines.append(f'rsync -av --exclude=_shards "$LOC/{rel}/" '
                     f'"$swpath":"$RUNC/{rel}/"')
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Patch C2 wrong-branch reference values in pulled "
                    "held-out eval artifacts.")
    parser.add_argument("--run-dir", required=True,
                        help="The pulled run directory (contains "
                             "checkpoints/, manifest.json, "
                             "resolved_config.yaml).")
    parser.add_argument("--specs", default=None,
                        help="Optional comma-separated spec indices to "
                             "restrict the audit/patch to.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Audit and report only; write nothing.")
    parser.add_argument("--bench-refs-dir",
                        default=os.environ.get("XCQUINOX_BENCH_REFS_DIR"),
                        help="Local dir of benchmark CCSD reference .npz "
                             "files at the eval identity (default: "
                             "$XCQUINOX_BENCH_REFS_DIR).")
    parser.add_argument("--allow-unknown-skip", action="store_true",
                        help="Skip channels whose c2 E_pbe matches neither "
                             "branch instead of refusing the run.")
    parser.add_argument("--coldstart-tol-e", type=float, default=1e-6,
                        help="Max |recomputed - recorded| (Ha) for the "
                             "cold-start channel's NN energies; the "
                             "channel seeds minao so its NN rows must "
                             "reproduce (default 1e-6, the reference-gate "
                             "scale).")
    parser.add_argument("--coldstart-tol-dens", type=float, default=3.78e-6,
                        help="Max |recomputed - recorded| for the "
                             "cold-start channel's NN density metrics. "
                             "Default 3.78e-6 = 10 x the largest measured "
                             "cross-evaluation spread of the density "
                             "metrics' noise class (density_eps 3.78e-7, "
                             "2026-08-31 three-generation measurement) -- "
                             "the same 10x-band footing as the model-free "
                             "gates; override for a tighter or looser "
                             "verification.")
    # JAX routing + shard-worker parity env FIRST: _load_cfg /
    # _solver_config_for_channel below import xcquinox modules that pull
    # JAX in transitively, and a backend initialized before the routing
    # would ignore it (mirrors cluster/_eval_one_spec.main).
    _route_jax_env()

    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    if not (run_dir / "checkpoints").is_dir():
        _print(f"FATAL: {run_dir} has no checkpoints/ dir")
        return 2
    specs = _parse_specs(args.specs)

    _print("=" * 72)
    _print("C2 wrong-branch reference patch -- audit")
    _print(f"run: {run_dir}")
    _print("NOTE: this audit is only as current as the local pull. Refresh")
    _print("first, and re-audit after the training array drains:")
    _print("  " + _pull_refresh_command(run_dir))
    _print("=" * 72)

    try:
        audit_rows = audit_run(run_dir, specs=specs)
    except PatchRefused as exc:
        _print(f"REFUSED: {exc}")
        _print("nothing was written")
        return 2
    if not audit_rows:
        _print("no spec dirs with eval channels found; nothing to do")
        return 0
    _print(format_audit_table(audit_rows, run_dir))
    _print()

    unknown = [r for r in audit_rows if r.state in ("unknown", "no-c2")]
    if unknown:
        for r in unknown:
            if r.state == "no-c2":
                _print(f"NO-C2: spec {r.spec} {r.channel} has no c2 row "
                       "in per_molecule.json; nothing to classify")
            else:
                _print(f"UNKNOWN state: spec {r.spec} {r.channel} "
                       f"E_pbe(c2)={r.e_pbe!r} matches neither branch "
                       f"within {CLASSIFY_TOL:g}")
        if not args.allow_unknown_skip:
            _print("unknown/no-c2 states are not patchable; rerun with "
                   "--allow-unknown-skip to skip them explicitly")
            return 4

    patchable = [r for r in audit_rows if r.patchable]
    pending = [r for r in audit_rows if r.pending_fetch]
    n_wrong = sum(1 for r in audit_rows if r.state == "wrong")
    _print(f"audit: {len(audit_rows)} channels; {n_wrong} wrong, "
           f"{len(patchable)} patchable now, {len(pending)} pending fetch")

    if args.dry_run:
        _print("--dry-run: stopping after the audit")
        return 0
    if not patchable:
        _print("nothing patchable; done")
        return 0

    try:
        return _run_patch(run_dir, args, audit_rows, patchable, pending)
    except PartialWriteError as exc:
        _print(f"WRITE-PHASE FAILURE: {exc}")
        if exc.committed:
            _print(f"{len(exc.committed)} channels were PATCHED AND ON "
                   "DISK before the failure (each integrity-verified):")
            for a in exc.committed:
                _print(f"  spec {a.spec} {a.channel}")
        else:
            _print("no channel had been committed before the failure")
        a = exc.failed
        _print(f"the failing channel spec {a.spec} {a.channel} may be "
               "partially rewritten; re-pull it from the cluster before "
               "re-running")
        if exc.committed:
            _print()
            _print(_push_sheet(run_dir, exc.committed))
        return 3
    except PatchRefused as exc:
        _print(f"REFUSED: {exc}")
        _print("nothing was written")
        return 2


def _cluster_bench_refs_dir(run_dir) -> str | None:
    """``inputs.benchmark_refs_dir`` from resolved_config.yaml by text."""
    cfg = Path(run_dir) / "resolved_config.yaml"
    if not cfg.is_file():
        return None
    m = re.search(r"^\s*benchmark_refs_dir:\s*(\S+)\s*$", cfg.read_text(),
                  flags=re.MULTILINE)
    return m.group(1) if m else None


def _gate_bench_refs(run_dir, bench_refs_dir) -> Path:
    """The local benchmark-refs dir with ``c2.npz`` present, or a refusal.

    The density metrics are NN/PBE-vs-CCSD on the reference grid, so the
    recompute needs the same ``c2.npz`` the evals loaded; the default pull
    does not carry it, and a patch that silently skipped the density
    columns would leave wrong-branch values standing."""
    remote = _cluster_bench_refs_dir(run_dir) or \
        "/gpfs/scratch/awills/external_refs_bench_6311ppg3df2pd_g3"
    hint_local = os.path.expanduser(
        "~/Documents/Research/xcquinox-results/external_refs/"
        + os.path.basename(remote))
    hint = (f"fetch it with:\n"
            f'  mkdir -p {hint_local}\n'
            f'  rsync -av "$swpath":{remote}/{SPECIES}.npz {hint_local}/\n'
            f"then rerun with --bench-refs-dir {hint_local}")
    if not bench_refs_dir:
        raise PatchRefused(
            "no benchmark CCSD reference dir: pass --bench-refs-dir or set "
            "XCQUINOX_BENCH_REFS_DIR. The c2 density metrics are "
            f"CCSD-referenced and cannot be recomputed without it; {hint}")
    refs = Path(bench_refs_dir)
    if not (refs / f"{SPECIES}.npz").is_file():
        raise PatchRefused(
            f"{refs}/{SPECIES}.npz not found: the density metrics cannot "
            f"be recomputed at the eval identity; {hint}")
    return refs


def _gate_stamped_wrong(audit_rows) -> None:
    """A channel stamped as patched whose values are still wrong is an
    inconsistent state no automatic rewrite can be trusted over."""
    bad = [r for r in audit_rows if r.state == "wrong" and r.stamped]
    if bad:
        names = ", ".join(f"spec {r.spec} {r.channel}" for r in bad)
        raise PatchRefused(
            f"{names}: eval_metadata.json already carries a "
            "reference_patch stamp, yet the c2 record still shows the "
            "wrong-branch E_pbe. A stamped channel must be clean; re-pull "
            "the channel from the cluster and re-audit before patching.")


def _gate_c_atom(audit_rows, plans) -> float:
    """The recorded C-atom E_pbe must agree across every audited channel
    (the reaction recompute REUSES those recorded energies)."""
    values = []
    plan_paths = {p.audit.path for p in plans}
    plan_by_path = {p.audit.path: p for p in plans}
    for r in audit_rows:
        if r.state not in ("clean", "wrong"):
            continue
        if r.path in plan_paths:
            pm_rows = plan_by_path[r.path].pm_rows
        else:
            pm_rows = _read_json(r.path / "per_molecule.json")
        row = _row_by_molecule(pm_rows, C_ATOM)
        if row is None or not _finite(row.get("E_pbe")):
            raise PatchRefused(
                f"spec {r.spec} {r.channel}: no finite E_pbe for the "
                f"{C_ATOM!r} atom; the reaction recompute reuses the "
                "recorded C-atom energies and cannot proceed without them.")
        values.append((r.spec, r.channel, float(row["E_pbe"])))
    es = [v for _, _, v in values]
    spread = max(es) - min(es)
    if spread > C_ATOM_TOL:
        detail = "; ".join(f"spec {s} {c}: {v!r}" for s, c, v in values)
        raise PatchRefused(
            f"recorded C-atom E_pbe disagrees across channels by "
            f"{spread:g} Ha (> {C_ATOM_TOL:g}): {detail}. The reaction "
            "recompute reuses the recorded C-atom energies, which is only "
            "valid while they are one value.")
    return es[0]


def _gate_reaction_consistency(plan: ChannelPlan) -> None:
    """The recorded c2 reaction rows must reproduce from the recorded
    per-molecule energies before they are rewritten from patched ones."""
    rows = [r for r in plan.pr_rows if _reaction_contains_species(r)]
    if not any(r.get("name") == REACTION_NAME for r in rows):
        raise PatchRefused(
            f"spec {plan.audit.spec} {plan.audit.channel}: "
            f"per_reaction.json has no {REACTION_NAME!r} row where one is "
            "expected; the artifact set is not the shape this tool "
            "repairs.")
    e_nn, e_pbe = _energy_maps(plan.pm_rows)
    for r in rows:
        for tag, emap in (("nn", e_nn), ("pbe", e_pbe)):
            recorded = r.get(f"de_{tag}_kcalmol")
            recomputed = _reaction_de_kcalmol(r, emap)
            both_nan = (not _finite(recorded)
                        and not math.isfinite(recomputed))
            if both_nan:
                continue
            if not _finite(recorded) or \
                    abs(float(recorded) - recomputed) > \
                    REACTION_CONSISTENCY_TOL:
                raise PatchRefused(
                    f"spec {plan.audit.spec} {plan.audit.channel}: "
                    f"recorded de_{tag}_kcalmol={recorded!r} of "
                    f"{r.get('name')!r} does not reproduce from the "
                    f"recorded per-molecule energies ({recomputed!r}); "
                    "the artifacts disagree internally and are not "
                    "patched.")


def _gate_reference(md, where: str) -> None:
    e = md.get("E_pbe")
    if not _finite(e) or abs(float(e) - GOOD_E_PBE) > REF_GATE_TOL:
        raise PatchRefused(
            f"the recomputed {SPECIES!r} reference for {where} landed at "
            f"E_pbe={e!r}, not the stable-branch {GOOD_E_PBE!r} (tol "
            f"{REF_GATE_TOL:g}); no patch is written from an unverified "
            "reference.")


def _gate_solver_describe(plan: ChannelPlan, sc) -> None:
    recorded = plan.meta.get("solver_config")
    rebuilt = sc.describe()
    if recorded != rebuilt:
        diff = {k: (recorded.get(k) if isinstance(recorded, dict) else None,
                    rebuilt.get(k))
                for k in set(rebuilt) | set(recorded or {})
                if (recorded.get(k) if isinstance(recorded, dict)
                    else None) != rebuilt.get(k)}
        raise PatchRefused(
            f"spec {plan.audit.spec} {plan.audit.channel}: the rebuilt "
            f"solver config does not match the channel's recorded "
            f"eval_metadata.json solver_config (differing keys: {diff}); "
            "the recompute would not run the protocol the channel ran.")


def _gate_coldstart_verify(plan: ChannelPlan, fresh: dict, *, tol_e: float,
                           tol_dens: float) -> dict:
    """The cold-start channel seeds minao, so its recorded NN values must
    reproduce; a mismatch means the recompute is NOT running the channel's
    protocol and nothing may be patched from it. Returns the measured
    deltas for the report."""
    old = plan.old_c2
    deltas = {}
    checks = [(f, tol_e) for f in COLDSTART_VERIFY_E_FIELDS]
    checks += [(f, tol_dens) for f in COLDSTART_VERIFY_DENS_FIELDS]
    for field, tol in checks:
        a, b = old.get(field), fresh.get(field)
        if a is None and b is None:
            continue
        if not (_is_num(a) and _is_num(b)):
            raise PatchRefused(
                f"spec {plan.audit.spec} {plan.audit.channel}: cold-start "
                f"verification cannot compare {field!r} "
                f"(recorded {a!r}, recomputed {b!r}).")
        d = abs(float(a) - float(b))
        deltas[field] = d
        if d > tol:
            raise PatchRefused(
                f"spec {plan.audit.spec} {plan.audit.channel}: the "
                f"minao-seeded recompute does not reproduce the recorded "
                f"{field} (|delta|={d:g} > {tol:g}; recorded {a!r}, "
                f"recomputed {b!r}). The cold-start NN rows should be "
                "independent of the reference branch, so this mismatch "
                "means the recompute protocol differs from the channel's; "
                "nothing is patched.")
    for field in ("cycles_run", "scf_converged"):
        if old.get(field) != fresh.get(field):
            raise PatchRefused(
                f"spec {plan.audit.spec} {plan.audit.channel}: cold-start "
                f"{field} differs (recorded {old.get(field)!r}, recomputed "
                f"{fresh.get(field)!r}); the recompute protocol differs "
                "from the channel's and nothing is patched.")
    return deltas


def _gate_model_free_recompute(plan: ChannelPlan, fresh: dict,
                               pool: dict | None) -> None:
    """The locally recomputed model-free values must sit inside the
    clean-channel evidence.

    SCF-dependent fields: the recompute (the corrected code, the same
    generation as the post-fix cluster evals) must lie within the clean
    envelope widened by :data:`BAND_FACTOR` x the field's measured max
    spread (:data:`MODEL_FREE_SPREAD`). Grid quantities: the recompute
    must reproduce the exact consensus value bit-for-bit -- a mismatch
    means the local grid identity differs from the eval's and nothing may
    be patched."""
    if pool is None:
        return
    for k in SCF_DEPENDENT_MODEL_FREE:
        env = pool["envelope"].get(k)
        f = fresh.get(k)
        if env is None or f is None:
            continue
        lo, hi = env
        band = BAND_FACTOR * MODEL_FREE_SPREAD[k]
        if not (lo - band <= float(f) <= hi + band):
            raise PatchRefused(
                f"spec {plan.audit.spec} {plan.audit.channel}: the local "
                f"recompute of {k} ({f!r}) lies outside the clean-channel "
                f"envelope [{lo!r}, {hi!r}] widened by "
                f"{BAND_FACTOR:g} x the measured {MODEL_FREE_SPREAD[k]:g} "
                "reproducibility spread (2026-08-31 measurement, three "
                "evaluation generations); the recompute does not "
                "reproduce the reference within the known reconvergence "
                "class and nothing is patched.")
    for k in EXACT_MODEL_FREE:
        v = pool["exact"].get(k)
        f = fresh.get(k)
        if v is None or f is None:
            continue
        if float(f) != float(v):
            raise PatchRefused(
                f"spec {plan.audit.spec} {plan.audit.channel}: the local "
                f"recompute of the grid quantity {k} ({f!r}) does not "
                f"equal the clean-channel value ({v!r}) exactly; grid "
                "quantities are bit-identical across evaluations (80/80 "
                "measured), so this mismatch is a grid-identity problem "
                "and nothing is patched.")


def _run_patch(run_dir, args, audit_rows, patchable, pending) -> int:
    t_start = time.time()

    # ---- preconditions (no reads beyond the audit, no writes) -----------
    refs_dir = _gate_bench_refs(run_dir, args.bench_refs_dir)
    _gate_stamped_wrong(audit_rows)

    # ---- read + pre-validate every patchable channel (no writes yet) ----
    manifest = _read_json_refusing(run_dir / "manifest.json",
                                   "manifest.json")
    cells = {s["index"]: s["cell"] for s in manifest["specs"]}

    plans = []
    for audit in patchable:
        plan = _load_channel(audit)
        if plan.old_c2 is None:
            raise PatchRefused(
                f"spec {audit.spec} {audit.channel}: no {SPECIES!r} row in "
                "per_molecule.json where one is expected.")
        _gate_reaction_consistency(plan)
        plans.append(plan)

    # The consensus and C-atom pools are ALWAYS run-wide: a --specs
    # restriction narrows what is patched, never the evidence base -- the
    # restricted set is exactly the contaminated specs, which carry no
    # clean channel of their own.
    specs = _parse_specs(args.specs)
    full_rows = (audit_rows if specs is None
                 else audit_run(run_dir, specs=None))

    c_atom_e_pbe = _gate_c_atom(full_rows, plans)
    _print(f"C-atom gate: recorded E_pbe({C_ATOM}) = {c_atom_e_pbe!r} "
           f"agrees across all audited channels (tol {C_ATOM_TOL:g})")

    pool, n_cons = _collect_consensus(run_dir, full_rows)
    if pool is None:
        raise PatchRefused(
            "no clean channel anywhere in the run supplies the model-free "
            "evidence; the six model-free c2 columns are patched only "
            "against a clean-channel envelope (SCF-dependent fields) and "
            "exact consensus (grid quantities). Re-pull after more specs "
            "complete, or repair one spec on the cluster first.")
    _env = pool["envelope"].get("E_pbe")
    _print(f"model-free consensus from {n_cons} clean channels: "
           f"E_pbe envelope={_env!r}, "
           f"n_electrons={pool['exact'].get('n_electrons')!r}")

    # ---- recompute + plan (all gates; still no writes) ----
    cfg = _load_cfg(run_dir)
    coldstart_deltas = {}
    n = len(plans)
    for i, plan in enumerate(plans, start=1):
        t0 = time.time()
        audit = plan.audit
        cell = cells.get(audit.spec)
        if cell is None:
            raise PatchRefused(
                f"spec {audit.spec} is absent from manifest.json's specs "
                f"list ({run_dir / 'manifest.json'}); the run dir is "
                "inconsistent and no cell identity exists to rebuild the "
                "arch/solver from.")
        _print(f"[{i}/{n}] spec {audit.spec} {audit.channel}: "
               f"recomputing (arch={cell['arch']}) ...")
        sc = _solver_config_for_channel(cfg, cell, audit.channel)
        _gate_solver_describe(plan, sc)
        model_path = (run_dir / "checkpoints" / f"spec_{audit.spec:04d}"
                      / audit.model_file)
        md = _recompute_reference(cfg, cell, sc, refs_dir)
        _gate_reference(md, f"spec {audit.spec} {audit.channel}")
        model = _load_model_for_channel(cfg, cell, model_path)
        fresh = _nn_record_for_channel(model, md, sc)
        _gate_model_free_recompute(plan, fresh, pool)
        if plan.coldstart:
            deltas = _gate_coldstart_verify(
                plan, fresh, tol_e=args.coldstart_tol_e,
                tol_dens=args.coldstart_tol_dens)
            coldstart_deltas[(audit.spec, audit.channel)] = deltas
            _print(f"[{i}/{n}] cold-start NN rows reproduce: " + ", ".join(
                f"{k} |d|={v:.3g}" for k, v in sorted(deltas.items())))
        _plan_patch(plan, fresh, pool)
        el = time.time() - t0
        eta = (n - i) * (time.time() - t_start) / i
        _print(f"[{i}/{n}] spec {audit.spec} {audit.channel}: planned "
               f"({el:.1f}s; ETA {eta:.0f}s)")

    # ---- write phase -----------------------------------------------------
    # Every gate has passed; from the first write on, a failure is a
    # PARTIAL-write state and is reported as such (never as "nothing was
    # written").
    committed = []
    for plan in plans:
        try:
            table = _write_plan(plan)
        except PatchRefused as exc:
            raise PartialWriteError(str(exc), committed=list(committed),
                                    failed=plan.audit) from exc
        committed.append(plan.audit)
        audit = plan.audit
        changed = sorted(p for p, s in table.items() if s == "changed")
        unchanged = sum(1 for s in table.values() if s == "unchanged")
        _print(f"patched spec {audit.spec} {audit.channel}: "
               f"changed {changed}; {unchanged} other files byte-identical")

    _print()
    _print(f"patched {len(plans)} channels in {time.time() - t_start:.1f}s")
    if pending:
        _print(f"{len(pending)} channels remain PENDING-FETCH "
               f"(best-loss checkpoints not in the local pull):")
        _print("  ordering: PUSH the patched channels first (sheet "
               "below), THEN fetch -- the fetch mirrors the "
               "eval_holdout*/ dirs and would overwrite locally patched "
               "channels with the cluster's wrong copies:")
        _print("  " + _fetch_command(run_dir,
                                     sorted({r.spec for r in pending})))
    _print()
    _print(_push_sheet(run_dir, [p.audit for p in plans]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
