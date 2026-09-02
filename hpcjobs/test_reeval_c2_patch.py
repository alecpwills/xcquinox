"""Fixture-based tests for the C2 wrong-branch reference patch tool.

The tool repairs held-out eval artifacts whose reference SCF landed C2/PBE
on the internally unstable branch (E_pbe(c2) = -75.7368945257551 instead of
-75.81674071208121, +50.10 kcal/mol; see data.py's branch-checked rescue).
These tests exercise the audit classification, the guarded patch, and the
aggregate recomputation on synthetic channel directories with known-wrong
c2 rows. No real SCF and no model deserialization run here: the compute
seams (_recompute_reference / _load_model_for_channel /
_solver_config_for_channel / _nn_record_for_channel / _load_cfg) are
replaced with stubs, so the suite pins the file semantics -- which fields
change, which bytes must not -- and the refusal gates.

Every guard test was first run against a tool version without its guard
(patch machinery present, gates absent) and observed to fail there, so each
gate is known to be load-bearing rather than vacuously green.
"""
from __future__ import annotations

import copy
import csv
import hashlib
import io
import json
import math
import os
import re
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import reeval_c2_patch as rcp  # noqa: E402


# ---------------------------------------------------------------------------
# Anchors (measured on run_20260827T163330Z, 2026-08-31)
# ---------------------------------------------------------------------------

GOOD = -75.81674071208121          # 72-channel bit-identical clean consensus
BAD = -75.7368945257551            # 27/27 contaminated channels
C_PBE = -37.794047545998325        # single distinct value across all channels
C_NN = -37.8001                    # fixture NN energy for the C atom
KC = 627.5094740631

# Clean-consensus model-free density sextet (fixture values shaped like the
# real ones; the tool treats them as opaque consensus numbers).
CONS = {
    "E_pbe": GOOD,
    "density_rmse_pbe": 0.00022149606464626117,
    "density_l1_pbe": 1.0336711804368342e-05,
    "density_eps_l1_pbe": 0.01149699623628779,
    "n_electrons": 12.00000029028857,
    "grid_weight_sum": 13346.987009408145,
}

# Wrong-branch density values carried by contaminated channels.
WRONG_DENS = {
    "density_rmse_pbe": 0.0027844891097218924,
    "density_l1_pbe": 0.00016849657902364242,
    "density_eps_l1_pbe": 0.18741013307960408,
}

# Three-generation clean-pool structure (fresh pull, 2026-08-31): the
# E_pbe and density_eps_l1_pbe triples are the measured cluster values
# verbatim (generation 0 = the 72-channel bit-identical set = CONS); the
# rmse/l1 companions span the measured spread classes (2.81e-9 / 3.40e-10)
# around generation 0 -- the fixture needs the structure, not the
# cluster's exact low digits.
GEN1 = {
    "E_pbe": -75.81674071207661,
    "density_rmse_pbe": CONS["density_rmse_pbe"] + 1.4e-9,
    "density_l1_pbe": CONS["density_l1_pbe"] + 1.7e-10,
    "density_eps_l1_pbe": 0.011497374547239803,
}
GEN2 = {
    "E_pbe": -75.81674071210273,
    "density_rmse_pbe": CONS["density_rmse_pbe"] - 1.4e-9,
    "density_l1_pbe": CONS["density_l1_pbe"] - 1.7e-10,
    "density_eps_l1_pbe": 0.011497281245159167,
}
# A fourth-generation local recompute, inside every clean envelope.
LOCAL_SCF = {
    "E_pbe": -75.81674071209,
    "density_rmse_pbe": CONS["density_rmse_pbe"] + 0.5e-9,
    "density_l1_pbe": CONS["density_l1_pbe"] + 0.5e-10,
    "density_eps_l1_pbe": 0.0114971,
}

NEW_NN = -75.8300                  # stubbed recomputed NN energy (standard)
NEW_NN_DENS = {"density_rmse": 0.00031, "density_l1": 1.6e-05,
               "density_eps_l1": 0.0171}

SOLVER_DESCRIBE = {
    "auxbasis": None,
    "backend": "manual",
    "conv_tol": 1e-06,
    "convergence_name": "energy",
    "density_fit": True,
    "feature_policy": "reassemble",
    "max_cycles": 3,
    "mixer_kwargs": {"base": 0.3, "floor": 0.3},
    "mixer_name": "decaying_linear",
    "mode": "full",
    "orientation_lock_strength": 3e-05,
    "scf_grad_checkpoint": False,
    "scf_loss_tail": 10,
    "scf_loss_use_tail": True,
    "scf_loss_weight_power": 2.0,
    "seed_cache_dir": "/gpfs/scratch/awills/seed_cache_scan",
    "seed_source": "pbe",
}

COLD_DESCRIBE = dict(SOLVER_DESCRIBE, conv_tol=1e-12, max_cycles=25,
                     seed_cache_dir=None, seed_source="minao")

OUTPUT_ROOT = "/gpfs/scratch/awills/xcquinox_runs/dfs_step7/dfs6311_grid3_v6g1_size"


# ---------------------------------------------------------------------------
# Fixture builders
# ---------------------------------------------------------------------------

def _atom_row(name, e_nn, e_pbe):
    return {
        "molecule": name, "E_total_nn": e_nn, "E_pbe": e_pbe,
        "AE_nn": e_nn - e_pbe, "AE_error_kcalmol": None,
        "density_rmse": None, "density_l1": None, "density_rmse_pbe": None,
        "density_l1_pbe": None, "density_eps_l1": None,
        "density_eps_l1_pbe": None, "n_electrons": None,
        "grid_weight_sum": None, "ref_density_method": None,
        "cycles_run": 3, "scf_converged": True,
        "from_training_subset": False, "scf_total_energy": e_nn,
    }


def _mol_row(name, e_nn, e_pbe, dens, n_steps=3):
    row = {
        "molecule": name, "E_total_nn": e_nn, "E_pbe": e_pbe,
        "AE_nn": e_nn - e_pbe, "AE_error_kcalmol": None,
        "density_rmse": dens.get("density_rmse", 3e-4),
        "density_l1": dens.get("density_l1", 2e-5),
        "density_rmse_pbe": dens.get("density_rmse_pbe", 2e-4),
        "density_l1_pbe": dens.get("density_l1_pbe", 1e-5),
        "density_eps_l1": dens.get("density_eps_l1", 1.5e-2),
        "density_eps_l1_pbe": dens.get("density_eps_l1_pbe", 1.1e-2),
        "n_electrons": dens.get("n_electrons", 12.00000029028857),
        "grid_weight_sum": dens.get("grid_weight_sum", 13346.987009408145),
        "ref_density_method": "ccsd",
        "cycles_run": n_steps, "scf_converged": False,
        "from_training_subset": False,
        "scf_total_energy": e_nn + 1e-4,
    }
    for i in range(n_steps):
        row[f"scf_energy_step_{i}"] = e_nn + (n_steps - 1 - i) * 1e-3
        row[f"scf_energy_residual_{i}"] = abs(
            row[f"scf_energy_step_{i}"] - row["scf_total_energy"])
    return row


def _per_molecule(c2_e_pbe, c2_e_nn=-75.8175485704837, c2_dens=None,
                  c_e_pbe=C_PBE, n_steps=3):
    dens = dict(n_electrons=CONS["n_electrons"],
                grid_weight_sum=CONS["grid_weight_sum"])
    dens.update(c2_dens or WRONG_DENS)
    return [
        _atom_row("c", C_NN, c_e_pbe),
        _mol_row("c2", c2_e_nn, c2_e_pbe, dens, n_steps=n_steps),
        _atom_row("h", -0.4980, -0.4995),
        _atom_row("o", -74.99, -75.01),
        _mol_row("o2", -150.31, -150.30, {}),
        _mol_row("oh", -75.71, -75.70, {}),
    ]


REACTION_DEFS = [
    ("bh76_r1", "bh76", ["oh"], ["o2", "h"], [-1.0, 1.0, 1.0], 10.0),
    ("w411_c2_atomization", "w411", ["c2"], ["c"], [-1.0, 2.0], 147.023),
    ("w411_o2_atomization", "w411", ["o2"], ["o"], [-1.0, 2.0], 120.0),
]


def _energy_maps(pm_rows):
    e_nn = {r["molecule"]: r["E_total_nn"] for r in pm_rows}
    e_pbe = {r["molecule"]: r["E_pbe"] for r in pm_rows}
    return e_nn, e_pbe


def _per_reaction(pm_rows):
    e_nn, e_pbe = _energy_maps(pm_rows)
    rows = []
    for name, pool, reac, prod, coeffs, ref in REACTION_DEFS:
        names = reac + prod
        de_nn = sum(c * e_nn[n] for c, n in zip(coeffs, names)) * KC
        de_pbe = sum(c * e_pbe[n] for c, n in zip(coeffs, names)) * KC
        rows.append({
            "name": name, "pool": pool, "reactants": reac, "products": prod,
            "coeffs": coeffs, "reaction_energy_ref_kcalmol": ref,
            "de_nn_kcalmol": de_nn, "de_pbe_kcalmol": de_pbe,
            "error_nn_kcalmol": de_nn - ref, "error_pbe_kcalmol": de_pbe - ref,
            "abs_error_nn_kcalmol": abs(de_nn - ref),
            "abs_error_pbe_kcalmol": abs(de_pbe - ref),
            "in_sample_overlap": [],
        })
    return rows


def _csv_text(pr_rows):
    """test_set.csv content with the production writer's exact semantics."""
    pools = []
    for r in pr_rows:
        if r["pool"] not in pools:
            pools.append(r["pool"])
    out = io.StringIO()
    fieldnames = ["set", "mae_nn_kcalmol", "mae_pbe_kcalmol",
                  "delta_nn_minus_pbe", "n_reactions", "n_dropped_overlap",
                  "n_dropped_nan", "note"]
    w = csv.DictWriter(out, fieldnames=fieldnames)
    w.writeheader()

    def _row(set_name, rows, note, n_dropped):
        nn = [r["abs_error_nn_kcalmol"] for r in rows
              if math.isfinite(r["abs_error_nn_kcalmol"])]
        pbe = [r["abs_error_pbe_kcalmol"] for r in rows
               if math.isfinite(r["abs_error_pbe_kcalmol"])]
        mae_nn = sum(nn) / len(nn)
        mae_pbe = sum(pbe) / len(pbe)
        w.writerow({
            "set": set_name, "mae_nn_kcalmol": f"{mae_nn:.6f}",
            "mae_pbe_kcalmol": f"{mae_pbe:.6f}",
            "delta_nn_minus_pbe": f"{mae_nn - mae_pbe:+.6f}",
            "n_reactions": len(nn), "n_dropped_overlap": n_dropped,
            "n_dropped_nan": 0, "note": note,
        })

    for pool in pools:
        rows = [r for r in pr_rows if r["pool"] == pool]
        note = "strict (held-out only)"
        n_dropped = 0
        if pool == "w411":
            note = ("strict (held-out only); 1 verbatim-supervised "
                    "reactions dropped")
            n_dropped = 1
        _row(f"test_set_{pool}", rows, note, n_dropped)
    _row("test_set_held_out_combined", pr_rows,
         "combined across pools (strict)", 1)
    return out.getvalue()


def _metadata(channel, describe, model="model.eqx", coldstart=False):
    return {
        "channel": channel, "coldstart": coldstart, "model": model,
        "n_reactions": 181, "n_species": 214,
        "solver_config": copy.deepcopy(describe), "species_slice": None,
    }


def _write_channel(spec_dir, channel, pm_rows, describe, model="model.eqx",
                   coldstart=False):
    ch = spec_dir / channel
    ch.mkdir(parents=True)
    pr_rows = _per_reaction(pm_rows)
    (ch / "per_molecule.json").write_text(json.dumps(pm_rows, indent=2))
    (ch / "per_reaction.json").write_text(json.dumps(pr_rows, indent=2))
    (ch / "test_set.csv").write_text(_csv_text(pr_rows), newline="")
    (ch / "eval_metadata.json").write_text(
        json.dumps(_metadata(channel, describe, model=model,
                             coldstart=coldstart), indent=2, sort_keys=True))
    shards = ch / "_shards"
    shards.mkdir()
    (shards / "shard_000.json").write_text("{}")
    return ch


def make_run(tmp_path, *, specs=(19,), c2_e_pbe=BAD, with_val_best_clean=True,
             with_best_dir=True, with_coldstart=True, n_specs=44):
    run = tmp_path / "run_20260827T163330Z"
    (run / "checkpoints").mkdir(parents=True)
    manifest = {
        "width": 4, "n_specs": n_specs,
        "specs": [{"cell": {"arch": "medium", "loss": "L", "metric": "jsd",
                            "solver": "full_3", "subset_size": 1},
                   "index": i, "spec_file": f"spec_{i:04d}.spec"}
                  for i in range(n_specs)],
    }
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2))
    (run / "resolved_config.yaml").write_text(
        "inputs:\n  basis: 6-311++G(3df,2pd)\n"
        f"  output_root: {OUTPUT_ROOT}\n")
    for idx in specs:
        sd = run / "checkpoints" / f"spec_{idx:04d}"
        sd.mkdir()
        (sd / "model.eqx").write_bytes(b"weights-final")
        (sd / "model_val_best.eqx").write_bytes(b"weights-valbest")
        _write_channel(sd, "eval_holdout", _per_molecule(c2_e_pbe),
                       SOLVER_DESCRIBE)
        if with_best_dir:
            _write_channel(sd, "eval_holdout_best", _per_molecule(c2_e_pbe),
                           SOLVER_DESCRIBE, model="model_best.eqx")
        if with_val_best_clean:
            _write_channel(sd, "eval_holdout_val_best",
                           _per_molecule(GOOD, c2_dens=CONS),
                           SOLVER_DESCRIBE, model="model_val_best.eqx")
        else:
            _write_channel(sd, "eval_holdout_val_best",
                           _per_molecule(c2_e_pbe),
                           SOLVER_DESCRIBE, model="model_val_best.eqx")
        if with_coldstart:
            _write_channel(sd, "eval_holdout_coldstart",
                           _per_molecule(c2_e_pbe, c2_e_nn=-75.7332397826433),
                           COLD_DESCRIBE, coldstart=True)
    return run


def _tree_hash(root):
    out = {}
    for p in sorted(Path(root).rglob("*")):
        if p.is_file():
            out[str(p.relative_to(root))] = hashlib.sha256(
                p.read_bytes()).hexdigest()
    return out


def _add_clean_valbest_spec(run, idx, gen):
    """A spec dir carrying ONLY a clean eval_holdout_val_best channel with
    the given generation's model-free values."""
    sd = run / "checkpoints" / f"spec_{idx:04d}"
    sd.mkdir()
    (sd / "model_val_best.eqx").write_bytes(b"weights-valbest")
    trio = {k: gen[k] for k in ("density_rmse_pbe", "density_l1_pbe",
                                "density_eps_l1_pbe")}
    _write_channel(sd, "eval_holdout_val_best",
                   _per_molecule(gen["E_pbe"], c2_dens=trio),
                   SOLVER_DESCRIBE, model="model_val_best.eqx")
    return sd


def _drift_c_atom_self_consistently(ch, delta=2e-5):
    """Drift the channel's recorded C-atom E_pbe and rebuild per_reaction
    + test_set.csv from the drifted energies, so the channel stays
    internally consistent and only the cross-channel C-atom gate can
    object to it."""
    ch = Path(ch)
    pm = json.loads((ch / "per_molecule.json").read_text())
    for r in pm:
        if r["molecule"] == "c":
            r["E_pbe"] = r["E_pbe"] + delta
    pr = _per_reaction(pm)
    (ch / "per_molecule.json").write_text(json.dumps(pm, indent=2))
    (ch / "per_reaction.json").write_text(json.dumps(pr, indent=2))
    (ch / "test_set.csv").write_text(_csv_text(pr), newline="")


# ---------------------------------------------------------------------------
# Stub seams
# ---------------------------------------------------------------------------

def _stub_cfg():
    return SimpleNamespace(
        inputs=SimpleNamespace(
            output_root=OUTPUT_ROOT, basis="6-311++G(3df,2pd)", grid_level=3,
            density_fit=True, auxbasis=None, orientation_lock_strength=3e-05,
            seed_cache_dir=SOLVER_DESCRIBE["seed_cache_dir"], seed_xc="auto",
            benchmark_refs_dir="/gpfs/unused"),
        model=SimpleNamespace(parent_anchor=True,
                              descriptor_coordinates="dfs"),
        solvers={}, use_polarized_correlation=True)


class _StubSolver:
    def __init__(self, describe):
        self._d = copy.deepcopy(describe)

    def describe(self):
        return copy.deepcopy(self._d)


def _fresh_record(seed_source, recorded_cold_nn=-75.7332397826433):
    """What the stubbed NN recompute hands back for c2.

    The minao branch mirrors the coldstart channel's RECORDED NN values
    (the fixture's ``_mol_row`` defaults), modelling the verified fact
    that the minao-seeded recompute reproduces the recorded rows; the
    pbe branch returns the repaired NN values.
    """
    if seed_source == "minao":
        e_nn = recorded_cold_nn
        dens = {"density_rmse": 3e-4, "density_l1": 2e-5,
                "density_eps_l1": 1.5e-2}
        scf_total = e_nn + 1e-4
        n_steps = 3
    else:
        e_nn = NEW_NN
        dens = dict(NEW_NN_DENS)
        scf_total = e_nn + 2e-4
        n_steps = 3
    rec = {
        "molecule": "c2", "E_total_nn": e_nn, "E_pbe": GOOD,
        "AE_nn": e_nn - GOOD, "AE_error_kcalmol": None,
        "density_rmse": dens["density_rmse"],
        "density_l1": dens["density_l1"],
        "density_rmse_pbe": CONS["density_rmse_pbe"],
        "density_l1_pbe": CONS["density_l1_pbe"],
        "density_eps_l1": dens["density_eps_l1"],
        "density_eps_l1_pbe": CONS["density_eps_l1_pbe"],
        "n_electrons": CONS["n_electrons"],
        "grid_weight_sum": CONS["grid_weight_sum"],
        "ref_density_method": "ccsd",
        "cycles_run": n_steps, "scf_converged": False,
        "from_training_subset": False, "scf_total_energy": scf_total,
    }
    for i in range(n_steps):
        rec[f"scf_energy_step_{i}"] = e_nn + (n_steps - 1 - i) * 9e-4
        rec[f"scf_energy_residual_{i}"] = abs(
            rec[f"scf_energy_step_{i}"] - rec["scf_total_energy"])
    return rec


@pytest.fixture
def stubbed(monkeypatch, tmp_path):
    """Patch every compute seam; return the bench-refs dir used."""
    refs = tmp_path / "bench_refs"
    refs.mkdir()
    (refs / "c2.npz").write_bytes(b"npz")
    monkeypatch.setattr(rcp, "_load_cfg", lambda run_dir: _stub_cfg())
    monkeypatch.setattr(
        rcp, "_recompute_reference",
        lambda cfg, cell, sc, bench_refs_dir:
        {"E_pbe": GOOD, "_seed": sc.describe()["seed_source"]})
    monkeypatch.setattr(
        rcp, "_solver_config_for_channel",
        lambda cfg, cell, channel:
        _StubSolver(COLD_DESCRIBE if channel == "eval_holdout_coldstart"
                    else SOLVER_DESCRIBE))
    monkeypatch.setattr(
        rcp, "_load_model_for_channel",
        lambda cfg, cell, model_path: object())
    monkeypatch.setattr(
        rcp, "_nn_record_for_channel",
        lambda model, md, sc: _fresh_record(sc.describe()["seed_source"]))
    return refs


def _main(run, refs, *extra):
    argv = ["--run-dir", str(run), "--bench-refs-dir", str(refs)]
    argv += list(extra)
    return rcp.main(argv)


# ---------------------------------------------------------------------------
# Constant pin
# ---------------------------------------------------------------------------

def test_kcal_constant_pinned_to_repo_source():
    src = (HERE.parent / "xcquinox" / "alec" / "eval_holdout.py").read_text()
    m = re.search(r"KCAL_PER_HA: float = ([0-9.]+)", src)
    assert m, "KCAL_PER_HA not found in eval_holdout.py"
    assert float(m.group(1)) == rcp.KCAL_PER_HA


def test_anchor_constants():
    assert rcp.GOOD_E_PBE == GOOD
    assert rcp.BAD_E_PBE == BAD


# ---------------------------------------------------------------------------
# Classification + audit
# ---------------------------------------------------------------------------

def test_classify_channel_states():
    assert rcp.classify_rows(_per_molecule(BAD))[0] == "wrong"
    assert rcp.classify_rows(_per_molecule(GOOD, c2_dens=CONS))[0] == "clean"
    assert rcp.classify_rows(_per_molecule(-75.60))[0] == "unknown"
    rows = [r for r in _per_molecule(BAD) if r["molecule"] != "c2"]
    assert rcp.classify_rows(rows)[0] == "no-c2"


def test_audit_states_flags_and_pending_fetch(tmp_path, capsys):
    run = make_run(tmp_path)
    rows = rcp.audit_run(run)
    by = {(r.spec, r.channel): r for r in rows}
    assert by[(19, "eval_holdout")].state == "wrong"
    assert by[(19, "eval_holdout")].patchable
    assert by[(19, "eval_holdout_val_best")].state == "clean"
    assert not by[(19, "eval_holdout_val_best")].patchable
    best = by[(19, "eval_holdout_best")]
    assert best.state == "wrong"
    assert best.pending_fetch and not best.patchable
    table = rcp.format_audit_table(rows, run)
    assert "PENDING-FETCH" in table
    assert ("pull run_20260827T163330Z --category "
            "dfs_step7/dfs6311_grid3_v6g1_size/runs --profile full "
            "--specs 19") in table


def test_audit_notes_failure_json_and_missing_artifacts(tmp_path):
    run = make_run(tmp_path)
    ch = run / "checkpoints" / "spec_0019" / "eval_holdout"
    (ch / "failure.json").write_text("{}")
    cold = run / "checkpoints" / "spec_0019" / "eval_holdout_coldstart"
    for name in ("per_molecule.json", "per_reaction.json", "test_set.csv",
                 "eval_metadata.json"):
        (cold / name).unlink()
    rows = rcp.audit_run(run)
    by = {(r.spec, r.channel): r for r in rows}
    assert by[(19, "eval_holdout")].failed
    assert by[(19, "eval_holdout")].patchable
    assert by[(19, "eval_holdout_coldstart")].state == "no-artifacts"
    assert not by[(19, "eval_holdout_coldstart")].patchable


def test_unknown_state_exits_4_unless_allowed(tmp_path, stubbed):
    run = make_run(tmp_path, c2_e_pbe=-75.60)
    assert _main(run, stubbed, "--dry-run") == 4
    assert _main(run, stubbed, "--dry-run", "--allow-unknown-skip") == 0


def test_no_c2_state_distinct_message(tmp_path, stubbed, capsys):
    """no-c2 is reported as an absent row, not as an off-branch value."""
    run = make_run(tmp_path)
    ch = run / "checkpoints" / "spec_0019" / "eval_holdout"
    pm = json.loads((ch / "per_molecule.json").read_text())
    pm = [r for r in pm if r["molecule"] != "c2"]
    (ch / "per_molecule.json").write_text(json.dumps(pm, indent=2))
    assert _main(run, stubbed, "--dry-run") == 4
    out = capsys.readouterr().out
    assert "no c2 row" in out
    assert "eval_holdout" in out.split("no c2 row")[0].rsplit("\n", 2)[-1] \
        or "spec 19 eval_holdout" in out


def test_specs_restriction_limits_audit(tmp_path):
    run = make_run(tmp_path, specs=(19, 20))
    rows = rcp.audit_run(run, specs=[20])
    assert {r.spec for r in rows} == {20}


def test_dry_run_writes_nothing(tmp_path, stubbed):
    run = make_run(tmp_path)
    before = _tree_hash(run)
    assert _main(run, stubbed, "--dry-run") == 0
    assert _tree_hash(run) == before


# ---------------------------------------------------------------------------
# The patch: fields + checksums
# ---------------------------------------------------------------------------

STD_PATCH_KEYS = {
    "E_pbe", "E_total_nn", "AE_nn", "density_rmse", "density_l1",
    "density_rmse_pbe", "density_l1_pbe", "density_eps_l1",
    "density_eps_l1_pbe", "n_electrons", "grid_weight_sum",
    "ref_density_method", "cycles_run", "scf_converged", "scf_total_energy",
}


def _changed_keys(old_row, new_row):
    keys = set(old_row) | set(new_row)
    return {k for k in keys if old_row.get(k, "\0") != new_row.get(k, "\0")}


def test_patch_standard_channel_fields_and_checksums(tmp_path, stubbed):
    run = make_run(tmp_path)
    ch = run / "checkpoints" / "spec_0019" / "eval_holdout"
    before = _tree_hash(run)
    old_pm = json.loads((ch / "per_molecule.json").read_text())
    assert _main(run, stubbed) == 0
    after = _tree_hash(run)
    changed = {p for p in before if before[p] != after.get(p)}
    expect_channels = [
        ("spec_0019", "eval_holdout"),
        ("spec_0019", "eval_holdout_coldstart"),
    ]
    expected = set()
    for spec, channel in expect_channels:
        for f in ("per_molecule.json", "per_reaction.json", "test_set.csv",
                  "eval_metadata.json"):
            expected.add(f"checkpoints/{spec}/{channel}/{f}")
    assert changed == expected
    assert set(before) == set(after), "files were added or removed"

    new_pm = json.loads((ch / "per_molecule.json").read_text())
    old_by = {r["molecule"]: r for r in old_pm}
    new_by = {r["molecule"]: r for r in new_pm}
    assert set(old_by) == set(new_by)
    for name in old_by:
        if name == "c2":
            continue
        assert old_by[name] == new_by[name], f"non-c2 row {name} changed"
    changed_keys = _changed_keys(old_by["c2"], new_by["c2"])
    trace_keys = {k for k in changed_keys
                  if k.startswith(("scf_energy_step_", "scf_energy_residual_"))}
    assert changed_keys - trace_keys <= STD_PATCH_KEYS
    c2 = new_by["c2"]
    assert c2["E_pbe"] == GOOD
    assert c2["E_total_nn"] == NEW_NN
    assert c2["AE_nn"] == NEW_NN - GOOD
    assert c2["density_rmse_pbe"] == CONS["density_rmse_pbe"]
    assert c2["density_rmse"] == NEW_NN_DENS["density_rmse"]

    old_pr = _per_reaction(old_pm)
    new_pr = json.loads((ch / "per_reaction.json").read_text())
    for old_r, new_r in zip(old_pr, new_pr):
        if new_r["name"] != "w411_c2_atomization":
            assert old_r == new_r
    c2r = [r for r in new_pr if r["name"] == "w411_c2_atomization"][0]
    de_nn = (-NEW_NN + 2 * C_NN) * KC
    de_pbe = (-GOOD + 2 * C_PBE) * KC
    assert c2r["de_nn_kcalmol"] == pytest.approx(de_nn, abs=1e-9)
    assert c2r["de_pbe_kcalmol"] == pytest.approx(de_pbe, abs=1e-9)
    assert c2r["error_nn_kcalmol"] == pytest.approx(de_nn - 147.023, abs=1e-9)
    assert c2r["abs_error_pbe_kcalmol"] == pytest.approx(
        abs(de_pbe - 147.023), abs=1e-9)


def test_patch_stamps_eval_metadata_without_clobbering(tmp_path, stubbed):
    run = make_run(tmp_path)
    ch = run / "checkpoints" / "spec_0019" / "eval_holdout"
    old_meta = json.loads((ch / "eval_metadata.json").read_text())
    assert _main(run, stubbed) == 0
    meta = json.loads((ch / "eval_metadata.json").read_text())
    stamp = meta.pop("reference_patch")
    assert meta == old_meta, "existing metadata keys were altered"
    assert stamp["species"] == "c2"
    assert stamp["from_E_pbe"] == BAD
    assert stamp["to_E_pbe"] == GOOD
    assert "date" in stamp
    assert "E_pbe" in stamp["fields"] and "E_total_nn" in stamp["fields"]


def test_aggregate_mae_matches_hand_computed(tmp_path, stubbed):
    run = make_run(tmp_path)
    ch = run / "checkpoints" / "spec_0019" / "eval_holdout"
    old_lines = (ch / "test_set.csv").read_bytes().split(b"\r\n")
    assert _main(run, stubbed) == 0
    raw = (ch / "test_set.csv").read_bytes()
    assert b"\r\n" in raw
    lines = raw.decode().split("\r\n")
    assert lines[0] == "set,mae_nn_kcalmol,mae_pbe_kcalmol,delta_nn_minus_pbe,n_reactions,n_dropped_overlap,n_dropped_nan,note"
    # bh76 row byte-identical (no c2 in its reactions)
    assert lines[1].encode() == old_lines[1]

    # hand-computed w411 row
    e_c2_nn, e_c2_pbe = NEW_NN, GOOD
    err_c2_nn = (-e_c2_nn + 2 * C_NN) * KC - 147.023
    err_c2_pbe = (-e_c2_pbe + 2 * C_PBE) * KC - 147.023
    err_o2_nn = (-(-150.31) + 2 * (-74.99)) * KC - 120.0
    err_o2_pbe = (-(-150.30) + 2 * (-75.01)) * KC - 120.0
    mae_nn = (abs(err_c2_nn) + abs(err_o2_nn)) / 2
    mae_pbe = (abs(err_c2_pbe) + abs(err_o2_pbe)) / 2
    w411 = lines[2].split(",")
    assert w411[0] == "test_set_w411"
    assert w411[1] == f"{mae_nn:.6f}"
    assert w411[2] == f"{mae_pbe:.6f}"
    assert w411[3] == f"{mae_nn - mae_pbe:+.6f}"
    assert w411[4:7] == ["2", "1", "0"], "counts must be preserved"
    comb = lines[3].split(",")
    assert comb[0] == "test_set_held_out_combined"
    assert comb[4:7] == ["3", "1", "0"]


def test_coldstart_patches_only_pbe_columns(tmp_path, stubbed):
    run = make_run(tmp_path)
    ch = run / "checkpoints" / "spec_0019" / "eval_holdout_coldstart"
    old_pm = json.loads((ch / "per_molecule.json").read_text())
    assert _main(run, stubbed) == 0
    new_pm = json.loads((ch / "per_molecule.json").read_text())
    old_c2 = [r for r in old_pm if r["molecule"] == "c2"][0]
    new_c2 = [r for r in new_pm if r["molecule"] == "c2"][0]
    changed = _changed_keys(old_c2, new_c2)
    assert changed == {"E_pbe", "AE_nn", "density_rmse_pbe",
                       "density_l1_pbe", "density_eps_l1_pbe"}
    assert new_c2["E_total_nn"] == old_c2["E_total_nn"]
    assert new_c2["AE_nn"] == old_c2["E_total_nn"] - GOOD
    assert new_c2["density_rmse"] == old_c2["density_rmse"]
    new_pr = json.loads((ch / "per_reaction.json").read_text())
    c2r = [r for r in new_pr if r["name"] == "w411_c2_atomization"][0]
    assert c2r["de_nn_kcalmol"] == pytest.approx(
        (-old_c2["E_total_nn"] + 2 * C_NN) * KC, abs=1e-9)
    meta = json.loads((ch / "eval_metadata.json").read_text())
    assert "E_total_nn" not in meta["reference_patch"]["fields"]


def test_failure_json_untouched_and_patch_proceeds(tmp_path, stubbed):
    run = make_run(tmp_path)
    ch = run / "checkpoints" / "spec_0019" / "eval_holdout"
    (ch / "failure.json").write_text('{"kind": "held_out_eval_failure"}')
    before = hashlib.sha256((ch / "failure.json").read_bytes()).hexdigest()
    assert _main(run, stubbed) == 0
    after = hashlib.sha256((ch / "failure.json").read_bytes()).hexdigest()
    assert before == after
    pm = json.loads((ch / "per_molecule.json").read_text())
    assert [r for r in pm if r["molecule"] == "c2"][0]["E_pbe"] == GOOD


def test_push_sheet_lists_patched_channels_only(tmp_path, stubbed, capsys):
    run = make_run(tmp_path)
    assert _main(run, stubbed) == 0
    out = capsys.readouterr().out
    assert '"$swpath"' in out
    dest = (OUTPUT_ROOT + "/runs/run_20260827T163330Z/checkpoints")
    assert dest in out
    assert re.search(r"rsync -av --exclude=_shards .*spec_0019/eval_holdout/",
                     out)
    assert "spec_0019/eval_holdout_coldstart/" in out
    assert "spec_0019/eval_holdout_best/" not in out
    assert "spec_0019/eval_holdout_val_best/" not in out


# ---------------------------------------------------------------------------
# Refusal gates (each observed RED against the gate-free tool)
# ---------------------------------------------------------------------------

def test_reference_gate_refuses_wrong_recompute(tmp_path, stubbed,
                                                monkeypatch):
    run = make_run(tmp_path)
    monkeypatch.setattr(
        rcp, "_recompute_reference",
        lambda cfg, cell, sc, bench_refs_dir: {"E_pbe": BAD})
    before = _tree_hash(run)
    assert _main(run, stubbed) == 2
    assert _tree_hash(run) == before, "patched despite failed reference gate"


def test_c_atom_drift_gate_refuses(tmp_path, stubbed):
    """ISOLATED fixture: the drifted channel's per_reaction + CSV are
    rebuilt self-consistently from the drifted energies, so of all the
    gates only the cross-channel C-atom agreement can object."""
    run = make_run(tmp_path, specs=(19, 20))
    _drift_c_atom_self_consistently(
        run / "checkpoints" / "spec_0020" / "eval_holdout")
    before = _tree_hash(run)
    assert _main(run, stubbed) == 2
    assert _tree_hash(run) == before, "patched despite C-atom drift"


def test_reaction_selfconsistency_gate_refuses(tmp_path, stubbed):
    run = make_run(tmp_path)
    ch = run / "checkpoints" / "spec_0019" / "eval_holdout"
    pr = json.loads((ch / "per_reaction.json").read_text())
    for r in pr:
        if r["name"] == "w411_c2_atomization":
            r["de_pbe_kcalmol"] += 0.5
    (ch / "per_reaction.json").write_text(json.dumps(pr, indent=2))
    before = _tree_hash(run)
    assert _main(run, stubbed) == 2
    assert _tree_hash(run) == before


def test_untouched_pool_row_reproduction_gate(tmp_path, stubbed):
    run = make_run(tmp_path)
    ch = run / "checkpoints" / "spec_0019" / "eval_holdout"
    text = (ch / "test_set.csv").read_bytes().decode()
    lines = text.split("\r\n")
    cells = lines[1].split(",")
    cells[1] = "99.999999"          # corrupt the bh76 MAE cell
    lines[1] = ",".join(cells)
    (ch / "test_set.csv").write_text("\r\n".join(lines), newline="")
    before = _tree_hash(run)
    assert _main(run, stubbed) == 2
    assert _tree_hash(run) == before


def test_solver_describe_mismatch_refuses(tmp_path, stubbed):
    run = make_run(tmp_path)
    ch = run / "checkpoints" / "spec_0019" / "eval_holdout"
    meta = json.loads((ch / "eval_metadata.json").read_text())
    meta["solver_config"]["max_cycles"] = 4
    (ch / "eval_metadata.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True))
    before = _tree_hash(run)
    assert _main(run, stubbed) == 2
    assert _tree_hash(run) == before


def test_coldstart_nn_mismatch_refuses(tmp_path, stubbed, monkeypatch):
    run = make_run(tmp_path)

    def _bad_record(model, md, sc):
        rec = _fresh_record(sc.describe()["seed_source"])
        if sc.describe()["seed_source"] == "minao":
            rec["E_total_nn"] += 1e-3
        return rec

    monkeypatch.setattr(rcp, "_nn_record_for_channel", _bad_record)
    before = _tree_hash(run)
    assert _main(run, stubbed) == 2
    assert _tree_hash(run) == before


def test_missing_reaction_row_refuses(tmp_path, stubbed):
    """ISOLATED fixture: the c2 reaction row is RENAMED, not deleted --
    counts, MAEs and self-consistency all still reproduce, so only the
    named-row-presence check can refuse."""
    run = make_run(tmp_path)
    ch = run / "checkpoints" / "spec_0019" / "eval_holdout"
    pr = json.loads((ch / "per_reaction.json").read_text())
    for r in pr:
        if r["name"] == "w411_c2_atomization":
            r["name"] = "w411_c2_atomization_renamed"
    (ch / "per_reaction.json").write_text(json.dumps(pr, indent=2))
    before = _tree_hash(run)
    assert _main(run, stubbed) == 2
    assert _tree_hash(run) == before


def test_existing_stamp_refuses_repatch(tmp_path, stubbed):
    run = make_run(tmp_path)
    ch = run / "checkpoints" / "spec_0019" / "eval_holdout"
    meta = json.loads((ch / "eval_metadata.json").read_text())
    meta["reference_patch"] = {"species": "c2"}
    (ch / "eval_metadata.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True))
    before = _tree_hash(run)
    assert _main(run, stubbed) == 2
    assert _tree_hash(run) == before


def test_bench_refs_missing_refused(tmp_path, stubbed):
    run = make_run(tmp_path)
    empty = tmp_path / "empty_refs"
    empty.mkdir()
    before = _tree_hash(run)
    assert _main(run, empty) == 2
    assert _tree_hash(run) == before


def test_three_generation_pool_patches_recompute_in_band(tmp_path, stubbed,
                                                         monkeypatch,
                                                         capsys):
    """A clean pool spanning three evaluation generations (per-evaluation
    SCF reconvergence slack) must PATCH when the local recompute lies
    within the clean envelope widened by the measured band -- and the
    written SCF-dependent values are the RECOMPUTE'S OWN, while the grid
    pair keeps the exact consensus."""
    run = make_run(tmp_path)              # spec 19 val_best = generation 0
    _add_clean_valbest_spec(run, 27, GEN1)
    _add_clean_valbest_spec(run, 28, GEN2)

    def local_record(model, md, sc):
        rec = _fresh_record(sc.describe()["seed_source"])
        rec.update(LOCAL_SCF)
        rec["AE_nn"] = rec["E_total_nn"] - LOCAL_SCF["E_pbe"]
        return rec

    monkeypatch.setattr(rcp, "_nn_record_for_channel", local_record)
    assert _main(run, stubbed) == 0
    out = capsys.readouterr().out
    assert "model-free consensus from 3 clean channels" in out
    ch = run / "checkpoints" / "spec_0019" / "eval_holdout"
    pm = json.loads((ch / "per_molecule.json").read_text())
    c2 = [r for r in pm if r["molecule"] == "c2"][0]
    for k, v in LOCAL_SCF.items():
        assert c2[k] == v, f"{k} must carry the local recompute's value"
    assert c2["n_electrons"] == CONS["n_electrons"]
    assert c2["grid_weight_sum"] == CONS["grid_weight_sum"]


def test_recompute_outside_band_refuses(tmp_path, stubbed, monkeypatch):
    """A recompute outside the clean envelope + 10x measured-spread band
    refuses, even though it passes the stable-branch 1e-6 gate. The
    offset (1e-9 Ha) sits far above the 2.61e-10 E_pbe band of a
    single-generation pool."""
    run = make_run(tmp_path)

    def off_record(model, md, sc):
        rec = _fresh_record(sc.describe()["seed_source"])
        rec["E_pbe"] = GOOD + 1e-9
        rec["AE_nn"] = rec["E_total_nn"] - rec["E_pbe"]
        return rec

    monkeypatch.setattr(rcp, "_nn_record_for_channel", off_record)
    before = _tree_hash(run)
    assert _main(run, stubbed) == 2
    assert _tree_hash(run) == before


def test_exact_field_recompute_disagreement_refuses(tmp_path, stubbed,
                                                    monkeypatch):
    """n_electrons / grid_weight_sum are pure grid quantities: the local
    recompute must reproduce the single consensus value EXACTLY, else the
    grid identity differs and nothing may be patched."""
    run = make_run(tmp_path)

    def off_record(model, md, sc):
        rec = _fresh_record(sc.describe()["seed_source"])
        rec["n_electrons"] = CONS["n_electrons"] + 1e-9
        return rec

    monkeypatch.setattr(rcp, "_nn_record_for_channel", off_record)
    before = _tree_hash(run)
    assert _main(run, stubbed) == 2
    assert _tree_hash(run) == before


def test_consensus_recompute_disagreement_refuses(tmp_path, stubbed,
                                                  monkeypatch):
    """A recompute far outside the band (1e-5 on density_rmse_pbe, band
    2.81e-8) must refuse."""
    run = make_run(tmp_path)

    def _off_record(model, md, sc):
        rec = _fresh_record(sc.describe()["seed_source"])
        rec["density_rmse_pbe"] = CONS["density_rmse_pbe"] + 1e-5
        return rec

    monkeypatch.setattr(rcp, "_nn_record_for_channel", _off_record)
    before = _tree_hash(run)
    assert _main(run, stubbed) == 2
    assert _tree_hash(run) == before


# ---------------------------------------------------------------------------
# Review round (D1-D9)
# ---------------------------------------------------------------------------

def test_route_jax_env_set_before_first_cluster_import(tmp_path, stubbed,
                                                       monkeypatch):
    """JAX routing + shard-worker parity env must be in place BEFORE the
    first xcquinox/cluster import (which pulls JAX in transitively)."""
    run = make_run(tmp_path)
    varnames = ("JAX_PLATFORMS", "JAX_ENABLE_X64", "OMP_NUM_THREADS",
                "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "XLA_FLAGS")
    for v in varnames:
        monkeypatch.delenv(v, raising=False)
    seen = {}

    def probe(run_dir):
        for v in varnames:
            seen[v] = os.environ.get(v)
        return _stub_cfg()

    monkeypatch.setattr(rcp, "_load_cfg", probe)
    assert _main(run, stubbed) == 0
    assert seen["JAX_ENABLE_X64"] == "1"
    assert seen["JAX_PLATFORMS"] == "cpu"
    assert seen["OMP_NUM_THREADS"] == "1"
    assert seen["MKL_NUM_THREADS"] == "1"
    assert seen["OPENBLAS_NUM_THREADS"] == "1"
    assert "--xla_backend_optimization_level=1" in (seen["XLA_FLAGS"] or "")


def test_specs_restriction_uses_runwide_consensus(tmp_path, stubbed, capsys):
    """--specs must not shrink the consensus pool: a clean channel in an
    unrestricted spec still supplies the model-free consensus."""
    run = make_run(tmp_path, with_val_best_clean=False)   # spec 19 all wrong
    sd = run / "checkpoints" / "spec_0020"
    sd.mkdir()
    (sd / "model_val_best.eqx").write_bytes(b"weights-valbest")
    _write_channel(sd, "eval_holdout_val_best",
                   _per_molecule(GOOD, c2_dens=CONS), SOLVER_DESCRIBE,
                   model="model_val_best.eqx")
    assert _main(run, stubbed, "--specs", "19") == 0
    out = capsys.readouterr().out
    assert "model-free consensus from 1 clean channels" in out
    ch = run / "checkpoints" / "spec_0019" / "eval_holdout"
    pm = json.loads((ch / "per_molecule.json").read_text())
    assert [r for r in pm if r["molecule"] == "c2"][0]["E_pbe"] == GOOD


def test_no_clean_channel_anywhere_refuses(tmp_path, stubbed):
    """No clean channel in the whole run -> refusal, never an ungated
    write of locally recomputed model-free values."""
    run = make_run(tmp_path, with_val_best_clean=False)
    before = _tree_hash(run)
    assert _main(run, stubbed) == 2
    assert _tree_hash(run) == before


def test_c_atom_pool_is_runwide_under_specs_restriction(tmp_path, stubbed):
    """A C-atom drift in an unrestricted spec still blocks a --specs
    patch: the gate's pool is the whole run."""
    run = make_run(tmp_path, specs=(19, 20))
    _drift_c_atom_self_consistently(
        run / "checkpoints" / "spec_0020" / "eval_holdout")
    before = _tree_hash(run)
    assert _main(run, stubbed, "--specs", "19") == 2
    assert _tree_hash(run) == before


def test_partial_write_reports_committed_channels(tmp_path, stubbed,
                                                  monkeypatch, capsys):
    """A write-phase failure must name what IS on disk, never claim
    nothing was written, and exit with the partial-write code."""
    run = make_run(tmp_path)
    real = rcp._write_plan
    done = []

    def fail_second(plan):
        if done:
            raise rcp.PatchRefused("integrity violation simulated")
        done.append(plan)
        return real(plan)

    monkeypatch.setattr(rcp, "_write_plan", fail_second)
    rc = _main(run, stubbed)
    out = capsys.readouterr().out
    assert rc == 3
    assert "nothing was written" not in out
    assert "spec 19 eval_holdout" in out          # committed, on disk
    assert "eval_holdout_coldstart" in out        # the failed channel named


def test_null_consensus_over_finite_recorded_refuses(tmp_path, stubbed):
    """A None consensus value must never overwrite a finite recorded
    one."""
    run = make_run(tmp_path)
    ch = run / "checkpoints" / "spec_0019" / "eval_holdout_val_best"
    pm = json.loads((ch / "per_molecule.json").read_text())
    for r in pm:
        if r["molecule"] == "c2":
            r["density_eps_l1_pbe"] = None
    (ch / "per_molecule.json").write_text(json.dumps(pm, indent=2))
    before = _tree_hash(run)
    assert _main(run, stubbed) == 2
    assert _tree_hash(run) == before


def test_push_before_fetch_ordering_in_audit_table(tmp_path, stubbed,
                                                   capsys):
    run = make_run(tmp_path)
    assert _main(run, stubbed, "--dry-run") == 0
    out = capsys.readouterr().out
    assert "PUSH any locally patched channels to the cluster FIRST" in out


def test_push_before_fetch_ordering_after_patch(tmp_path, stubbed, capsys):
    run = make_run(tmp_path)
    assert _main(run, stubbed) == 0
    out = capsys.readouterr().out
    # The post-patch report repeats the fetch command; the ordering rule
    # must stand immediately before that LAST occurrence (the audit table
    # earlier in the output carries its own copy of both).
    rule = out.rindex("PUSH the patched channels first")
    fetch = out.rindex("--profile full --specs")
    assert rule < fetch, "ordering rule must precede the fetch command"


def test_corrupt_per_molecule_refuses_naming_file(tmp_path, stubbed,
                                                  capsys):
    run = make_run(tmp_path)
    bad = run / "checkpoints" / "spec_0019" / "eval_holdout" \
        / "per_molecule.json"
    bad.write_text("{not json")
    assert _main(run, stubbed, "--dry-run") == 2
    out = capsys.readouterr().out
    assert str(bad) in out


def test_spec_missing_from_manifest_refuses(tmp_path, stubbed):
    run = make_run(tmp_path)
    manifest = json.loads((run / "manifest.json").read_text())
    manifest["specs"] = [s for s in manifest["specs"] if s["index"] != 19]
    (run / "manifest.json").write_text(json.dumps(manifest, indent=2))
    before = _tree_hash(run)
    assert _main(run, stubbed) == 2
    assert _tree_hash(run) == before


# ---------------------------------------------------------------------------
# Gate attribution battery (each proven load-bearing by weakest mutation)
# ---------------------------------------------------------------------------

def test_csv_count_mismatch_refuses(tmp_path, stubbed):
    """A recorded count that the recomputation cannot reproduce refuses
    the patch. The corrupted cell sits on the w411 row (which contains
    the c2 reaction), so the untouched-row reproduction gate never sees
    it -- only the count check can object."""
    run = make_run(tmp_path)
    ch = run / "checkpoints" / "spec_0019" / "eval_holdout"
    text = (ch / "test_set.csv").read_bytes().decode()
    lines = text.split("\r\n")
    cells = lines[2].split(",")
    assert cells[0] == "test_set_w411" and cells[4] == "2"
    cells[4] = "3"                      # corrupt n_reactions
    lines[2] = ",".join(cells)
    (ch / "test_set.csv").write_text("\r\n".join(lines), newline="")
    before = _tree_hash(run)
    assert _main(run, stubbed) == 2
    assert _tree_hash(run) == before


def test_coldstart_cycles_mismatch_refuses(tmp_path, stubbed, monkeypatch):
    """cycles_run must reproduce exactly on the cold-start channel; the
    energies and densities are left matching so only the cycles/converged
    equality check can refuse."""
    run = make_run(tmp_path)

    def bad_cycles(model, md, sc):
        rec = _fresh_record(sc.describe()["seed_source"])
        if sc.describe()["seed_source"] == "minao":
            rec["cycles_run"] = rec["cycles_run"] + 1
        return rec

    monkeypatch.setattr(rcp, "_nn_record_for_channel", bad_cycles)
    before = _tree_hash(run)
    assert _main(run, stubbed) == 2
    assert _tree_hash(run) == before


def test_clean_consensus_spread_refuses(tmp_path, stubbed):
    """Clean channels disagreeing on a GRID quantity (n_electrons) leave
    no exact consensus: pure grid quantities are bit-identical across all
    clean channels (measured 80/80), so any disagreement is a
    grid-identity problem and refuses. (The SCF-dependent fields are
    band-gated instead -- see the three-generation tests.)"""
    run = make_run(tmp_path, specs=(19, 20))
    ch = run / "checkpoints" / "spec_0020" / "eval_holdout_val_best"
    pm = json.loads((ch / "per_molecule.json").read_text())
    for r in pm:
        if r["molecule"] == "c2":
            r["n_electrons"] = CONS["n_electrons"] + 1e-9
    (ch / "per_molecule.json").write_text(json.dumps(pm, indent=2))
    before = _tree_hash(run)
    assert _main(run, stubbed) == 2
    assert _tree_hash(run) == before


def test_csv_header_schema_refusal(tmp_path, stubbed):
    run = make_run(tmp_path)
    ch = run / "checkpoints" / "spec_0019" / "eval_holdout"
    text = (ch / "test_set.csv").read_bytes().decode()
    lines = text.split("\r\n")
    lines[0] = lines[0].replace("mae_nn_kcalmol", "mae_nn")
    (ch / "test_set.csv").write_text("\r\n".join(lines), newline="")
    before = _tree_hash(run)
    assert _main(run, stubbed) == 2
    assert _tree_hash(run) == before


def test_csv_unrecognized_row_refusal(tmp_path, stubbed):
    """A row whose empty MAE cells and zero counts would reproduce from
    an empty reaction subset -- only the unrecognized-set refusal can
    object to it."""
    run = make_run(tmp_path)
    ch = run / "checkpoints" / "spec_0019" / "eval_holdout"
    text = (ch / "test_set.csv").read_bytes().decode()
    text += "extra_set,,,,0,0,0,stray row\r\n"
    (ch / "test_set.csv").write_text(text, newline="")
    before = _tree_hash(run)
    assert _main(run, stubbed) == 2
    assert _tree_hash(run) == before


def test_nn_record_nonfinite_refuses(monkeypatch):
    """A non-finite recomputed c2 energy refuses instead of flowing into
    a record. The eval helpers are faked through sys.modules so no heavy
    stack loads."""
    fake_eh = types.ModuleType("xcquinox.alec.eval_holdout")

    def fake_eval(model, mol_data, solver_config=None, scf_info_out=None):
        if scf_info_out is not None:
            scf_info_out["c2"] = {"eval_error": "ValueError: boom"}
        return {"c2": float("nan")}

    fake_eh.evaluate_holdout = fake_eval
    fake_eh.density_errors_for_record = lambda *a, **k: {}
    fake_eh.make_per_molecule_record = lambda *a, **k: {"molecule": "c2"}
    fake_pkg = types.ModuleType("xcquinox")
    fake_alec = types.ModuleType("xcquinox.alec")
    fake_pkg.alec = fake_alec
    fake_alec.eval_holdout = fake_eh
    monkeypatch.setitem(sys.modules, "xcquinox", fake_pkg)
    monkeypatch.setitem(sys.modules, "xcquinox.alec", fake_alec)
    monkeypatch.setitem(sys.modules, "xcquinox.alec.eval_holdout", fake_eh)
    with pytest.raises(rcp.PatchRefused, match="non-finite"):
        rcp._nn_record_for_channel(object(), {},
                                   _StubSolver(SOLVER_DESCRIBE))


def test_write_integrity_violation_detected(tmp_path, stubbed, monkeypatch,
                                            capsys):
    """A write that leaves anything beyond the four artifacts changed is
    detected and reported as a write-phase failure."""
    run = make_run(tmp_path)
    real = rcp._write_text_atomic

    def leaky(path, text):
        real(path, text)
        stray = Path(path).parent / "stray.bin"
        if not stray.exists():
            stray.write_bytes(b"x")

    monkeypatch.setattr(rcp, "_write_text_atomic", leaky)
    rc = _main(run, stubbed)
    out = capsys.readouterr().out
    assert rc == 3
    assert "integrity violation" in out


def test_pool_stats_matches_writer_on_identity_twins():
    """_pool_stats must reproduce eval_holdout's identity-deduped counts on
    a slice WITH twins (every earlier fixture was twin-free, so the row/
    identity semantic split was invisible): 3 rows over 2 identities."""
    from xcquinox.alec.eval_holdout import reaction_mae_kcalmol
    rows = [
        {"name": "fwd", "reactants": ["a", "b"], "products": ["ts"],
         "coeffs": [-1.0, -1.0, 1.0],
         "abs_error_nn_kcalmol": 2.0, "abs_error_pbe_kcalmol": 3.0},
        {"name": "fwd_perm", "reactants": ["b", "a"], "products": ["ts"],
         "coeffs": [-1.0, -1.0, 1.0],
         "abs_error_nn_kcalmol": 2.0, "abs_error_pbe_kcalmol": 3.0},
        {"name": "other", "reactants": ["a"], "products": ["b"],
         "coeffs": [-1.0, 1.0],
         "abs_error_nn_kcalmol": 8.0, "abs_error_pbe_kcalmol": 5.0},
    ]
    mae_nn, mae_pbe, n_used, n_nan = rcp._pool_stats(rows)
    assert n_used == 2
    assert mae_nn == pytest.approx(5.0)
    assert mae_pbe == pytest.approx(4.0)
    assert n_nan == 0
    # Cross-check against the writer-side reduction on the same rows.
    e = {"a": -1.0, "b": -2.0, "ts": -2.9}
    w_rxns = [dict(r, reaction_energy_ref=1.0) for r in rows]
    _, n_writer, _ = reaction_mae_kcalmol(e, w_rxns)
    assert n_writer == n_used
