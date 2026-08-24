"""Tests for the energy-term-weight measurement job.

Three things are pinned here. The COMMAND SURFACE: what the sweep runs when
nothing is said, and what ``--smoke`` substitutes, because a sweep launched at
the wrong basis or grid measures a different question and costs a node-day to
find out. The RECOMMENDATION RULE: exercised on synthetic tables, including
every edge the real table can present -- no weight clearing the gate, a weight
clearing it only on part of the architecture set, a weight buying the energy by
destroying the point-wise fit, and a diverged cell. The JOB SCRIPT: the house
shell idiom, the standing mail directives and the exact invocation, since a
script that activates the wrong environment or drops a flag produces a table
that looks valid and is not.

The ``--smoke`` end-to-end leg runs the real sweep at a two-system STO-3G
identity in a SUBPROCESS -- 30 s, measured -- so it is neither slow-marked nor
able to take the test session down with it if JAX aborts at interpreter exit.
"""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve().parent
_SCRIPT = _HERE / "probe_pretrain_energy_weight.py"
_SBATCH = _HERE / "probe_pretrain_energy_weight.sbatch"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


pw = _load("probe_pretrain_energy_weight")


# --------------------------------------------------------------------------- #
# Argument parsing and defaults
# --------------------------------------------------------------------------- #

def _args(*extra):
    return pw.parse_args(["--data-dir", "d", "--out", "o", *extra])


def test_production_defaults_are_the_swept_identity():
    a = _args()
    assert a.archs == ("deep_3x16", "deep_cusp_3x16", "deep_rung35_3x16",
                       "deep_mgga_3x16")
    assert a.weights == (0.0, 0.1, 1.0, 10.0, 100.0)
    assert a.n_steps == 1000
    assert a.basis == "def2-svp"
    # Level 3 is a floor, not a taste: below it the generator refuses the
    # degenerate open-shell atoms this set is full of.
    assert a.grid_level == 3
    assert a.seed == 42
    assert a.loss_weighting == "integration"
    assert a.smoke is False
    assert a.tol_atom_mha == 1.0        # the certificate's tol_atom
    assert a.margin_fraction == 0.5
    assert a.pointwise_factor == 3.0


def test_smoke_substitutes_a_seconds_long_identity():
    a = _args("--smoke")
    assert a.smoke is True
    assert a.basis == "sto-3g"
    assert a.grid_level == 1
    assert a.n_steps == 5
    assert a.archs == ("deep_3x16", "deep_mgga_3x16")
    assert a.weights == (0.0, 1.0)
    # Both rungs, so the smoke exercises both parent densities.
    assert len(set(a.archs)) == 2


def test_explicit_flags_win_over_smoke():
    a = _args("--smoke", "--basis", "def2-svp", "--grid-level", "3",
              "--n-steps", "11", "--archs", "deep_3x16", "--weights", "2")
    assert (a.basis, a.grid_level, a.n_steps) == ("def2-svp", 3, 11)
    assert a.archs == ("deep_3x16",)
    assert a.weights == (2.0,)


def test_weights_are_deduplicated_and_ascending():
    # The rule takes the SMALLEST clearing weight, so the sweep order and the
    # read order have to be the same one.
    assert _args("--weights", "10, 1,1 0").weights == (0.0, 1.0, 10.0)


def test_archs_accept_commas_and_spaces():
    assert _args("--archs", "a, b c").archs == ("a", "b", "c")


@pytest.mark.parametrize("bad", [
    ["--weights", "-1"],            # a negative weight is not an objective
    ["--weights", "nan"],
    ["--weights", "inf"],
    ["--weights", "banana"],
    ["--weights", ""],
    ["--archs", ""],
    ["--n-steps", "0"],
    ["--grid-level", "-1"],
    ["--recon-rtol", "0"],
    ["--tol-atom-mha", "0"],
    ["--margin-fraction", "-0.5"],
    ["--pointwise-factor", "nan"],
    ["--loss-weighting", "sortof"],
])
def test_bad_arguments_are_refused(bad):
    with pytest.raises(SystemExit):
        _args(*bad)


@pytest.mark.parametrize("missing", [
    ["--out", "o"],                  # no --data-dir
    ["--data-dir", "d"],             # no --out
])
def test_data_dir_and_out_are_required(missing):
    with pytest.raises(SystemExit):
        pw.parse_args(missing)


# --------------------------------------------------------------------------- #
# The recommendation rule
# --------------------------------------------------------------------------- #

def _row(arch, weight, max_mha, loss_x=1.0e-3, loss_c=1.0e-3):
    return {"arch": arch, "weight": float(weight), "final_loss_x": loss_x,
            "final_loss_c": loss_c, "max_dE_xc_mHa": max_mha,
            "rms_dE_xc_mHa": (None if max_mha is None else 0.5 * max_mha)}


def test_smallest_clearing_weight_wins():
    rows = [_row("a", 0.0, 9.0), _row("b", 0.0, 8.0),
            _row("a", 1.0, 0.4), _row("b", 1.0, 0.3),
            _row("a", 10.0, 0.1), _row("b", 10.0, 0.1)]
    out = pw.recommend(rows)
    assert out["cleared"] is True
    # Both 1 and 10 clear; the rule says the smallest, not the best.
    assert out["weight"] == 1.0


def test_the_gate_is_the_max_over_every_architecture():
    # Weight 1 clears on 'a' and misses on 'b'; the rule is "every
    # architecture", so the choice moves up to 10.
    rows = [_row("a", 0.0, 9.0), _row("b", 0.0, 9.0),
            _row("a", 1.0, 0.1), _row("b", 1.0, 0.9),
            _row("a", 10.0, 0.1), _row("b", 10.0, 0.2)]
    out = pw.recommend(rows)
    assert (out["cleared"], out["weight"]) == (True, 10.0)


def test_margin_is_half_the_tolerance_and_the_boundary_clears():
    assert pw.recommend([_row("a", 0.0, 9.0),
                         _row("a", 1.0, 0.5)])["weight"] == 1.0
    out = pw.recommend([_row("a", 0.0, 9.0), _row("a", 1.0, 0.500001)])
    assert out["cleared"] is False


def test_a_weight_that_destroys_the_pointwise_fit_is_refused():
    # Weight 10 clears the gate but its exchange loss is 10x the weight-0
    # value; weight 100 clears it inside the cap.
    rows = [_row("a", 0.0, 9.0, loss_x=1.0e-3),
            _row("a", 10.0, 0.1, loss_x=1.0e-2),
            _row("a", 100.0, 0.2, loss_x=2.0e-3)]
    out = pw.recommend(rows)
    assert (out["cleared"], out["weight"]) == (True, 100.0)


def test_correlation_loss_counts_toward_the_cap_too():
    rows = [_row("a", 0.0, 9.0, loss_c=1.0e-3),
            _row("a", 1.0, 0.1, loss_c=9.0e-3)]
    out = pw.recommend(rows)
    assert out["cleared"] is False


def test_nothing_clears_reports_the_tradeoff_and_the_best_weight():
    rows = [_row("a", 0.0, 9.0), _row("a", 1.0, 4.0), _row("a", 10.0, 2.0),
            _row("b", 0.0, 9.0), _row("b", 1.0, 5.0), _row("b", 10.0, 3.0)]
    out = pw.recommend(rows)
    assert out["cleared"] is False
    assert out["weight"] == 10.0                    # minimizes the worst max
    assert "NO swept weight clears" in out["reason"]
    assert out["margin_mHa"] == 0.5


def test_a_tie_on_the_worst_error_goes_to_the_smaller_weight():
    rows = [_row("a", 0.0, 9.0), _row("a", 1.0, 3.0), _row("a", 10.0, 3.0)]
    out = pw.recommend(rows)
    assert (out["cleared"], out["weight"]) == (False, 1.0)


def test_a_weight_measured_on_only_part_of_the_set_is_not_eligible():
    rows = [_row("a", 0.0, 9.0), _row("b", 0.0, 9.0),
            _row("a", 1.0, 0.1),                     # 'b' at w=1 is missing
            _row("a", 10.0, 0.1), _row("b", 10.0, 0.1)]
    out = pw.recommend(rows)
    assert (out["cleared"], out["weight"]) == (True, 10.0)
    entry = next(e for e in out["per_weight"] if e["weight"] == 1.0)
    assert entry["missing_archs"] == ["b"]
    assert entry["gate_ok"] is False


def test_a_diverged_cell_is_a_failure_not_a_gap():
    rows = [_row("a", 0.0, 9.0), _row("a", 1.0, float("nan")),
            _row("a", 10.0, 0.1)]
    out = pw.recommend(rows)
    assert (out["cleared"], out["weight"]) == (True, 10.0)
    entry = next(e for e in out["per_weight"] if e["weight"] == 1.0)
    assert entry["gate_ok"] is False


def test_without_a_weight_zero_baseline_nothing_can_be_certified():
    # The cap is a ratio against weight 0; with no such cell the rise cannot
    # be measured and the sweep must say so rather than certify on the gate
    # alone.
    out = pw.recommend([_row("a", 1.0, 0.1), _row("a", 10.0, 0.1)])
    assert out["cleared"] is False
    assert out["per_weight"][0]["archs_without_baseline"] == ["a"]


def test_an_empty_table_chooses_nothing():
    out = pw.recommend([])
    assert out["cleared"] is False
    assert out["weight"] is None
    assert "nothing to choose between" in out["reason"]


def test_the_tolerance_and_margin_are_configurable():
    rows = [_row("a", 0.0, 9.0), _row("a", 1.0, 1.5)]
    assert pw.recommend(rows)["cleared"] is False
    out = pw.recommend(rows, tol_atom_mha=2.0, margin_fraction=1.0)
    assert (out["cleared"], out["weight"]) == (True, 1.0)


# --------------------------------------------------------------------------- #
# The table
# --------------------------------------------------------------------------- #

def test_write_table_creates_its_directory_and_round_trips(tmp_path):
    target = tmp_path / "deep" / "nested" / "table.json"
    payload = {"rows": [_row("a", 1.0, 0.25)], "recommendation": {"x": 1}}
    assert pw.write_table(str(target), payload) == str(target)
    assert json.loads(target.read_text()) == payload


def test_format_table_renders_every_row_and_the_verdict():
    rows = [_row("a", 0.0, 9.0), _row("a", 1.0, 0.25)]
    text = pw.format_table(rows, pw.recommend(rows))
    lines = text.splitlines()
    assert lines[0].split()[:3] == ["arch", "parent", "w_E"]
    # The table body: header, rule, then one line per measured cell.
    body = lines[2:lines.index("")]
    assert len(body) == 2
    assert len([line for line in body if " 9.0000" in line]) == 1
    assert len([line for line in body if " 0.2500" in line]) == 1
    assert "rule:" in text
    assert "recommendation: energy_term_weight = 1  [CLEARS]" in text


def test_format_table_renders_an_absent_cell_as_a_dash():
    row = _row("a", 1.0, 0.25)
    row["reference_xc"] = None
    row["wall_seconds"] = None
    text = pw.format_table([row])
    assert " - " in text or text.rstrip().endswith("-")


# --------------------------------------------------------------------------- #
# Exit-code contract (the sweep loop with the measurement stubbed out)
# --------------------------------------------------------------------------- #

def _stub_sweep(monkeypatch, maxima):
    """Run main() with the data generation and the pretraining replaced by a
    table the caller dictates. ``maxima`` maps weight -> max |dE_xc| in mHa."""
    monkeypatch.setattr(pw, "ensure_data",
                        lambda *a, **k: "/nonexistent/pretrain_data.npz")

    def _cell(arch, arch_name, data_path, work_dir, *, weight, **kwargs):
        return _row(arch_name, weight, maxima[weight])

    monkeypatch.setattr(pw, "run_cell", _cell)


def test_exit_zero_when_a_weight_clears(tmp_path, monkeypatch):
    _stub_sweep(monkeypatch, {0.0: 9.0, 1.0: 0.2})
    rc = pw.main(["--data-dir", str(tmp_path), "--out",
                  str(tmp_path / "t.json"), "--archs", "deep_3x16",
                  "--weights", "0,1"])
    assert rc == 0
    assert json.loads((tmp_path / "t.json").read_text())[
        "recommendation"]["weight"] == 1.0


def test_exit_nonzero_when_nothing_clears_but_the_table_is_still_written(
        tmp_path, monkeypatch):
    _stub_sweep(monkeypatch, {0.0: 9.0, 1.0: 4.0})
    rc = pw.main(["--data-dir", str(tmp_path), "--out",
                  str(tmp_path / "t.json"), "--archs", "deep_3x16",
                  "--weights", "0,1"])
    assert rc == 2
    payload = json.loads((tmp_path / "t.json").read_text())
    assert payload["recommendation"]["cleared"] is False
    assert len(payload["rows"]) == 2


def test_an_unknown_architecture_is_refused_by_name(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        pw.main(["--data-dir", str(tmp_path), "--out",
                 str(tmp_path / "t.json"), "--archs", "deep_3x16,not_an_arch"])
    assert "not_an_arch" in str(excinfo.value)
    assert not (tmp_path / "t.json").exists()


def test_a_failed_cell_is_recorded_and_exits_one(tmp_path, monkeypatch):
    monkeypatch.setattr(pw, "ensure_data", lambda *a, **k: "/nonexistent.npz")

    def _boom(*a, **k):
        raise RuntimeError("segment table disagrees")

    monkeypatch.setattr(pw, "run_cell", _boom)
    rc = pw.main(["--data-dir", str(tmp_path), "--out",
                  str(tmp_path / "t.json"), "--archs", "deep_3x16",
                  "--weights", "1"])
    assert rc == 1
    payload = json.loads((tmp_path / "t.json").read_text())
    assert payload["failures"][0]["arch"] == "deep_3x16"
    assert "segment table disagrees" in payload["failures"][0]["error"]


# --------------------------------------------------------------------------- #
# End to end at the smoke identity
# --------------------------------------------------------------------------- #

def test_smoke_sweep_end_to_end(tmp_path):
    """The real sweep at two systems / STO-3G / grid 1 / five steps.

    Measured 29 s wall including both parent-density generations, so it is not
    slow-marked. Run in a subprocess: JAX can abort at interpreter exit on this
    backend, and a measurement probe must not be able to take the session with
    it.
    """
    env = dict(os.environ)
    env.update(OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4",
               MKL_NUM_THREADS="4", JAX_PLATFORMS="cpu",
               XLA_FLAGS="--xla_cpu_multi_thread_eigen=false")
    out = tmp_path / "table.json"
    proc = subprocess.run(
        [sys.executable, str(_SCRIPT), "--smoke",
         "--data-dir", str(tmp_path / "data"), "--out", str(out)],
        env=env, capture_output=True, text=True, timeout=900)
    # 0 = a weight cleared, 2 = none did. Five steps from a random
    # initialization will not clear; both are completions, and 1 is not.
    assert proc.returncode in (0, 2), proc.stdout[-4000:] + proc.stderr[-4000:]

    payload = json.loads(out.read_text())
    assert payload["failures"] == []
    assert payload["identity"]["basis"] == "sto-3g"
    assert payload["identity"]["grid_level"] == 1
    assert payload["identity"]["exchange_footing"] == "spin_channel"
    rows = payload["rows"]
    assert len(rows) == 4
    # Both rungs ran, each against its own parent density.
    assert {r["reference_xc"] for r in rows} == {"pbe", "scan"}
    assert {r["exchange_footing"] for r in rows} == {"spin_channel"}
    for row in rows:
        assert row["n_systems"] == 2
        assert row["max_dE_xc_mHa"] > 0.0
        assert row["rms_dE_xc_mHa"] > 0.0
        # A max over systems can never exceed the sum of the two channels'
        # RMS times sqrt(N); what it must never do is come back as the mean.
        assert row["max_dE_x_mHa"] > 0.0 and row["max_dE_c_mHa"] > 0.0
        assert row["worst_system"] in ("He", "Li")

    by_weight = {(r["arch"], r["weight"]): r for r in rows}
    for (arch, weight), row in by_weight.items():
        if weight == 0.0:
            # THE reason the table is reconstructed rather than read off the
            # metadata: at weight 0 the loss short-circuits before the energy
            # term, so the recorded value is 0 while the error is not.
            assert row["energy_term_x_final"] == 0.0
            assert row["energy_term_c_final"] == 0.0
            assert row["energy_term_x_recon"] > 0.0
            assert row["energy_term_c_recon"] > 0.0
            assert row["recon_max_rel_dev"] is None
        else:
            # Where the recorded value is real, the reconstruction is it.
            assert row["recon_max_rel_dev"] is not None
            assert row["recon_max_rel_dev"] <= 1.0e-6
            assert row["energy_term_x_recon"] == pytest.approx(
                row["energy_term_x_final"], rel=1e-12)
    # The meta-GGA cell is the one that carries the synthetic mesh.
    assert by_weight[("deep_mgga_3x16", 0.0)]["pretrain_mesh"] is True
    assert by_weight[("deep_3x16", 0.0)]["pretrain_mesh"] is False


# --------------------------------------------------------------------------- #
# The job script
# --------------------------------------------------------------------------- #

def _sbatch_text() -> str:
    return _SBATCH.read_text()


def test_mail_directives_present():
    t = _sbatch_text()
    assert "#SBATCH --mail-user=alec.wills@stonybrook.edu" in t
    assert "#SBATCH --mail-type=BEGIN,END,FAIL" in t


def test_house_shell_idiom():
    t = _sbatch_text()
    assert "set -uo pipefail" in t
    for line in t.splitlines():
        assert not line.strip().startswith("set -e"), line
        assert "errexit" not in line, line


def test_single_node_one_task_with_a_thread_cap_from_slurm():
    t = _sbatch_text()
    assert "#SBATCH --nodes=1" in t
    assert "#SBATCH --ntasks=1" in t
    assert "#SBATCH --cpus-per-task=40" in t
    assert 'THREADS="${SLURM_CPUS_PER_TASK:-40}"' in t
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        assert f'export {var}="$THREADS"' in t


def test_walltime_exceeds_the_short_queue_cap():
    # The estimate is ~6 h; the short-* queues cap at 4 h, so the request and
    # the partition have to agree on a long queue.
    t = _sbatch_text()
    assert "#SBATCH --time=12:00:00" in t
    assert "#SBATCH --partition=long-" in t


def test_x64_is_on_and_the_platform_is_cpu():
    t = _sbatch_text()
    assert "export JAX_ENABLE_X64=1" in t
    assert "export JAX_PLATFORMS=cpu" in t


def test_activation_by_effect_and_an_import_probe():
    t = _sbatch_text()
    assert 'conda activate "$ENV_PREFIX" || true' in t
    assert '"$ENV_PREFIX"/*) : ;;' in t
    assert 'python -c "import xcquinox.alec.pretrain"' in t
    assert "FATAL: repo import failed" in t


def test_the_invocation_carries_the_swept_identity_and_a_log():
    t = _sbatch_text()
    assert "python -u hpcjobs/probe_pretrain_energy_weight.py" in t
    for flag in ('--data-dir "$DATA_DIR"', '--out      "$OUT"',
                 "--basis def2-svp", "--grid-level 3", "--n-steps 1000"):
        assert flag in t, flag
    # The log is a file, and the exit code read is python's, not tee's.
    assert 'tee "$LOG"' in t
    assert 'RC="${PIPESTATUS[0]}"' in t
    assert "#SBATCH --output=" in t


def test_the_script_the_job_runs_exists():
    assert _SCRIPT.is_file()


def test_exit_code_two_is_documented_as_a_finding():
    # SLURM mails FAIL on any non-zero code; the log has to say that a 2 is a
    # completed sweep, or the mail reads as a crash.
    t = _sbatch_text()
    assert "gate NOT cleared" in t
    assert "Not a crash" in t
