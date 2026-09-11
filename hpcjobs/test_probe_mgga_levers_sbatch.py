"""The meta-GGA lever job: its request, its two invocations, and its epilogue.

``probe_mgga_levers.sbatch`` runs ``probe_pretrain_energy_weight.py`` twice in
configuration mode -- the energy-term weight over four network widths under the
run's own objective, then the failed run's integration objective on the
narrowest width -- and reports each leg beside the output that leg captured.
Three properties are pinned here.

THE REQUEST AND THE IDENTITY. The directives that place one exclusive 96-core
node for the 48 h every long-* QOS caps at; the house shell idiom (no
``set -e``, since activation of a prefix environment returns non-zero on
success); the repository on ``PYTHONPATH``; a default configuration that
exists; and the refusal to start without the pretraining file the run's own
data generation wrote, which is what the study's parity claim rests on. Each
leg states the configuration and the swept axes and nothing else: a restated
basis, grid level, seed or schedule flag fits a different protocol from the run
the cells are compared against, at a node-day apiece to discover.

THE SWEPT CELLS. The step count, the four architectures in width order, the
three weights and the two-weight integration arm are read out of the
assignments rather than transcribed, and the second leg's table name is derived
from the first's -- two legs sharing one table would overwrite each other's
rows, and a resumed submission would then read one leg's cells as the other's.

THE EPILOGUE. Exit code 2 is two outcomes -- a completed leg whose cells all
missed the fidelity certificate (a finding, with a table written) and an
argparse refusal in which no cell ran -- and a leg whose captured output is
missing is a third, since the two cannot then be told apart. The shipped
``leg_code`` function and the code arithmetic that follows it are therefore cut
out of the script and executed for every combination that decides the job's own
exit code, rather than restated in a copy that would pass while the script kept
other text.
"""
from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

_HPCJOBS = Path(__file__).resolve().parent
_REPO = _HPCJOBS.parent
_SCRIPT = _HPCJOBS / "probe_mgga_levers.sbatch"

#: The thread block, verbatim. The PySCF-serving pools are capped at
#: min(allocation, 8); sized to a 96-core allocation they spin-wait and an SCF
#: loop that takes 8 s at four threads takes minutes. The block's values are
#: executed against ``parallel.pyscf_pool_threads`` in ``test_thread_caps.py``;
#: the text is pinned here so a rewrite that agrees only on the sampled
#: allocations is still visible.
_THREAD_BLOCK = '''THREADS="${SLURM_CPUS_PER_TASK:-8}"
case "$THREADS" in ''|*[!0-9]*) THREADS=8 ;; esac
THREADS=$(( 10#$THREADS ))
[ "$THREADS" -ge 1 ] || THREADS=1
[ "$THREADS" -le 8 ] || THREADS=8
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS" OPENBLAS_NUM_THREADS="$THREADS"'''

#: XLA's own pool is sized to the allocation, as the harness's pretrain
#: template sizes it; the ``:-8`` fallback is what keeps ``set -u`` from killing
#: the job where the variable is unset.
_XLA_LINE = ('export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true '
             'intra_op_parallelism_threads=${SLURM_CPUS_PER_TASK:-8} '
             '--xla_force_host_platform_device_count=1"')

#: The two openings a leg's captured output can have under exit code 2:
#: argparse's usage line (no cell ran, no table) and a completed sweep's own
#: summary (every cell measured, the table written).
_USAGE_CAPTURE = ("usage: probe_pretrain_energy_weight [-h] [--config CONFIG]\n"
                  "probe_pretrain_energy_weight: error: unrecognized arguments\n")
_VERDICT_CAPTURE = ("[probe] cells: 14 requested, 0 already measured\n"
                    "recommendation: fidelity_certificate  [DOES NOT CLEAR]\n")

#: The two invocation blocks, each from its banner comment through the line
#: that reads the probe's own exit code out of the pipeline.
_LEG1 = ("# --- leg 1", 'RC1="${PIPESTATUS[0]}"')
_LEG2 = ("# --- leg 2", 'RC2="${PIPESTATUS[0]}"')


def _script_text() -> str:
    """The job script as text; its absence is stated as the missing path."""
    assert _SCRIPT.is_file(), f"no such file: {_SCRIPT}"
    return _SCRIPT.read_text()


def _block(text: str, opener: str, closer: str) -> str:
    """The slice of ``text`` from ``opener`` through the end of ``closer``."""
    start = text.index(opener)
    end = text.index(closer, start) + len(closer)
    return text[start:end]


def _default(text: str, name: str) -> str:
    """The default of ``NAME="${MGGA_LEVERS_<X>:-default}"``, as written."""
    pattern = r'^' + name + r'="\$\{MGGA_LEVERS_[A-Z_]+:-([^}"]+)\}"$'
    found = re.search(pattern, text, re.M)
    assert found is not None, f"{name}: no defaulted assignment"
    return found.group(1)


def _epilogue_block() -> str:
    """The shipped epilogue, verbatim: ``leg_code()`` and everything after it.

    Cut rather than paraphrased, so what is executed is what the job runs. The
    cut runs to the end of the file because the code arithmetic over the two
    legs, the table report and the final ``exit`` are all part of what a leg's
    code decides; there is no ``case "$RC" in`` block to cut at, as there is in
    the energy-weight job.
    """
    text = _script_text()
    block = text[text.index("leg_code() {"):]
    assert block.rstrip().endswith('exit "$RC"'), block[-200:]
    return block


def _run_epilogue(tmp_path, rc1, rc2, *, capture1="verdict", capture2="verdict",
                  table=True, table_arm=True, skipped2=False):
    """Execute the shipped epilogue for one pair of leg exit codes.

    ``capture1`` and ``capture2`` are what each leg captured -- ``"verdict"`` a
    completed sweep's summary, ``"usage"`` argparse's refusal, ``"absent"`` no
    file at all -- ``table`` and ``table_arm`` whether that leg's table is on
    disk, and ``skipped2`` whether the wall guard skipped the second leg. The
    exit code is the quantity under test, so ``(stdout, returncode)`` is
    returned rather than asserted here.
    """
    captures = {}
    for var, kind in (("RUN_OUT1", capture1), ("RUN_OUT2", capture2)):
        path = tmp_path / (var.lower() + ".last")
        if kind == "usage":
            path.write_text(_USAGE_CAPTURE)
        elif kind == "verdict":
            path.write_text(_VERDICT_CAPTURE)
        else:
            assert kind == "absent", kind
            if path.exists():
                path.unlink()
        captures[var] = path
    out = tmp_path / "mgga_levers_table.json"
    out_arm = tmp_path / "mgga_levers_table_integration.json"
    for path, present in ((out, table), (out_arm, table_arm)):
        if present:
            path.write_text("{}\n")
        elif path.exists():
            path.unlink()
    run_out1, run_out2 = captures["RUN_OUT1"], captures["RUN_OUT2"]
    stub = (f"RC1={rc1}\nRC2={rc2}\n"
            f'RUN_OUT1="{run_out1}"\nRUN_OUT2="{run_out2}"\n'
            f'OUT="{out}"\nOUT_ARM="{out_arm}"\n'
            f'SKIPPED2="{"yes" if skipped2 else ""}"\n'
            'T_START="$(date +%s)"\n')
    done = subprocess.run(
        ["bash", "-uo", "pipefail", "-c", stub + _epilogue_block()],
        env={"PATH": os.environ.get("PATH", "")},
        capture_output=True, text=True)
    assert done.stderr == "", done.stderr
    return done.stdout, done.returncode


# --------------------------------------------------------------------------- #
# The request and the identity
# --------------------------------------------------------------------------- #

def test_the_job_script_exists():
    """Every other test reads this file; its absence is one line rather than a
    traceback out of a read on nothing."""
    assert _SCRIPT.is_file(), f"no such file: {_SCRIPT}"


def test_the_request_places_one_exclusive_node_for_the_queue_cap():
    """Fourteen cells at 1 h 18 to 1 h 28 apiece is 18.2 to 20.5 h, and the
    three wider networks are unmeasured against the 3x16 calibration, so the
    request is the 48 h every long-* QOS caps at. The node is taken whole by
    ``--exclusive --mem=0``: the fits are one process sharing one XLA pool."""
    t = _script_text()
    for directive in ("#SBATCH --partition=long-96core",
                      "#SBATCH --time=48:00:00",
                      "#SBATCH --nodes=1",
                      "#SBATCH --ntasks=1",
                      "#SBATCH --cpus-per-task=24",
                      "#SBATCH --exclusive",
                      "#SBATCH --mem=0",
                      "#SBATCH --output=/gpfs/scratch/awills/mgga_levers_%j.out"):
        assert directive in t, directive


def test_the_allocation_is_the_one_the_cloned_runs_fits_were_rendered_at():
    """The allocation is not a throttle here: it is what sizes XLA's intra-op
    pool, and the run this study claims parity with rendered its pretrain task
    at ``cluster.cpus_per_task`` of the families configuration. A request above
    that fits every cell at a pool the cloned run never used, and the per-cell
    cost figures in the header are calibrated at that width too.

    Read from the configuration rather than transcribed, so a campaign that
    re-renders at another width turns this red instead of leaving the study
    fitting at a silently different one.
    """
    yaml = pytest.importorskip("yaml")
    t = _script_text()
    cfg = yaml.safe_load((_REPO / _default(t, "CFG")).read_text())
    rendered = (cfg["cluster"].get("pretrain_cpus_per_task")
                or cfg["cluster"]["cpus_per_task"])
    assert f"#SBATCH --cpus-per-task={rendered}" in t, rendered
    assert "--exclusive" in t          # the whole node is still taken


def test_mail_directives_present():
    """Submission, completion and failure each reach the address the HPC mail
    for this project goes to."""
    t = _script_text()
    assert "#SBATCH --mail-user=alec.wills@stonybrook.edu" in t
    assert "#SBATCH --mail-type=BEGIN,END,FAIL" in t


def test_house_shell_idiom():
    """``set -e`` would kill the job on the activation call, which returns
    non-zero on a successful prefix activation, so the failure modes are read
    explicitly instead. The comment that states this must not satisfy the test,
    which is why the form is line by line rather than a substring search."""
    t = _script_text()
    assert "set -uo pipefail" in t
    for line in t.splitlines():
        assert not line.strip().startswith("set -e"), line
        assert "errexit" not in line, line


def test_the_repository_is_exported_on_the_path_the_probe_imports_from():
    """The job runs the probe by path from the checkout, so the package it
    imports has to be the checkout's rather than whatever the environment
    happens to have installed."""
    t = _script_text()
    assert 'export PYTHONPATH="${REPO}' in t


def test_the_default_configuration_is_a_file_in_the_checkout():
    """Every cell's identity is read from this file. A default naming a path
    that does not exist would be found on the node, after the allocation
    started."""
    t = _script_text()
    default = _default(t, "CFG")
    assert default.startswith("hpcjobs/configs/"), default
    assert (_REPO / default).is_file(), default


def test_the_runs_own_pretraining_file_is_required_before_any_cell_runs():
    """The parity claim is that the cells open the file the run's own data
    generation wrote. Generating one inside this job would put an unbudgeted
    production-identity generation inside the wall and fit against a file the
    run never saw, so an absent file stops the job before the first fit."""
    t = _script_text()
    assert "pretrain_data_polarized_scan.npz" in t
    guard = [line for line in t.splitlines() if '[ -f "$DATA_FILE" ]' in line]
    assert len(guard) == 1, guard
    assert "exit 1" in guard[0], guard[0]
    # The path is composed from the directory the configuration states, and an
    # unparsed directory is refused as that rather than left to compose a path
    # rooted at the filesystem root and reported as a missing file.
    composed = [line for line in t.splitlines() if line.startswith("DATA_FILE=")]
    assert composed == ['DATA_FILE="${DATA_DIR}/pretrain_data_polarized_scan.npz"'], composed
    parsed = [line for line in t.splitlines() if line.startswith("DATA_DIR=")]
    assert len(parsed) == 1 and '"$CFG"' in parsed[0], parsed
    empty = [line for line in t.splitlines() if '[ -n "$DATA_DIR" ]' in line]
    assert len(empty) == 1 and "exit 1" in empty[0], empty


def test_the_data_directory_is_read_from_the_configuration_the_job_runs():
    """The shipped configurations indent the pretrain block by two spaces, which
    is what the extraction reads; the value it takes is asserted against the
    loader's own, so a configuration whose layout the pattern cannot read is a
    refusal rather than a path built from an empty string."""
    yaml = pytest.importorskip("yaml")
    t = _script_text()
    cfg_path = _REPO / _default(t, "CFG")
    stated = yaml.safe_load(cfg_path.read_text())["pretrain"]["data_dir"]
    script = _script_text()
    line = next(l for l in script.splitlines() if l.startswith("DATA_DIR="))
    extracted = subprocess.run(
        ["bash", "-uo", "pipefail", "-c", f'CFG="{cfg_path}"\n{line}\nprintf %s "$DATA_DIR"'],
        capture_output=True, text=True, env={"PATH": os.environ.get("PATH", "")})
    assert extracted.returncode == 0, extracted.stderr
    assert extracted.stdout == stated, (extracted.stdout, stated)


# --------------------------------------------------------------------------- #
# The two invocations and the cells they sweep
# --------------------------------------------------------------------------- #

def test_the_weights_leg_states_the_configuration_and_the_axes_and_nothing_else():
    """Configuration mode reads the identity, the polarization and model block,
    the pretrain block and the data directory from the run's own YAML. A flag
    that restated any of them on the command line would be refused by the probe
    or, worse, would fit a protocol the run never used; the swept axes and the
    step count are the only departures the study makes."""
    t = _script_text()
    leg = _block(t, *_LEG1)
    for flag in ('--config "$CFG"', '--archs "$ARCHS"', '--weights "$WEIGHTS"',
                 '--n-steps "$STEPS"', '--out "$OUT"', "--resume"):
        assert flag in leg, flag
    for refused in ("--basis", "--grid-level", "--polarized", "--seed", "--lr-",
                    "--data-dir", "--smoke", "--objective-arm"):
        assert refused not in leg, refused


def test_the_integration_leg_carries_the_arm_and_its_own_table():
    """The 2026-09-02 meta-GGA run fitted under integration, where the
    (r_s, s, alpha) mesh carries its 0.3 share; under the live rho_w_sampled
    objective those rows carry no weight. The arm is the second leg's whole
    reason for existing, and it writes its own table."""
    t = _script_text()
    leg = _block(t, *_LEG2)
    assert "--objective-arm integration" in leg
    assert '--out "$OUT_ARM"' in leg
    assert '--out "$OUT"' not in leg
    # Flag for flag, as the weights leg is: the arm's axes, the study's step
    # count and the resume. A leg that lost --n-steps would fit the
    # configuration's 20000 and double the budget off the study's identity; one
    # that lost --archs or --weights would sweep the weights leg's twelve cells
    # again; one that lost --resume would rewrite its table from scratch on a
    # resubmission; one that lost --config would be refused only at run time,
    # after the first leg had spent its eighteen hours.
    for flag in ('--config "$CFG"', '--archs "$ARM_ARCHS"',
                 '--weights "$ARM_WEIGHTS"', '--n-steps "$STEPS"',
                 '--out "$OUT_ARM"', "--resume"):
        assert flag in leg, flag
    for refused in ("--basis", "--grid-level", "--polarized", "--seed", "--lr-",
                    "--data-dir", "--smoke"):
        assert refused not in leg, refused


def test_the_defaults_are_the_fourteen_cells_the_study_measures():
    """Twelve cells of the weights leg (four widths at three weights) and two
    of the integration arm. The widths run narrowest first, which is the order
    a wall-clock kill leaves the table in, so the order is asserted and not
    only the membership."""
    t = _script_text()
    assert _default(t, "STEPS") == "10000"
    archs = _default(t, "ARCHS").split(",")
    weights = _default(t, "WEIGHTS").split(",")
    arm_archs = _default(t, "ARM_ARCHS").split(",")
    arm_weights = _default(t, "ARM_WEIGHTS").split(",")
    assert archs == ["deep_mgga_3x16", "deep_mgga_3x32", "deep_mgga_4x16",
                     "deep_mgga_4x32"], archs
    assert weights == ["0.1", "10", "100"], weights
    assert arm_archs == ["deep_mgga_3x16"], arm_archs
    assert arm_weights == ["0.1", "100"], arm_weights
    n_cells = len(archs) * len(weights) + len(arm_archs) * len(arm_weights)
    assert n_cells == 14, n_cells
    # The header's cost estimate is quoted for this many cells; an axis widened
    # without it turns this red rather than leaving a stale wall in the header.
    assert "fourteen cells" in t.lower()


def test_the_two_legs_write_two_tables():
    """One ``cells/`` directory is shared between the legs and separated by the
    cell-directory suffix; the tables are separated here. A second leg writing
    the first's table would overwrite its rows, and a resumed submission would
    read one leg's cells as the other's."""
    t = _script_text()
    assert 'OUT_ARM="${OUT%.json}_integration.json"' in t


def test_the_thread_block_and_the_xla_line_are_the_shipped_text():
    """The SCF-serving pools at min(allocation, 8) and XLA's pool at the
    allocation, both as text: the values are executed against the module rule
    in ``test_thread_caps.py``, and a form that agreed with it only on the
    sampled allocations would pass there and not here."""
    t = _script_text()
    assert _THREAD_BLOCK in t
    assert _XLA_LINE in t


def test_each_leg_reads_its_own_exit_code_past_the_tees():
    """``tee`` is the last command of each pipeline and succeeds whatever the
    probe did, so the code carried forward is ``PIPESTATUS[0]``. Exactly two,
    one per leg: a third would mean a pipeline whose code is read twice or a
    leg whose code is not read at all."""
    t = _script_text()
    assert t.count("PIPESTATUS[0]") == 2, t.count("PIPESTATUS[0]")
    assert 'RC1="${PIPESTATUS[0]}"' in t
    assert 'RC2="${PIPESTATUS[0]}"' in t
    assert '2>&1 | tee -a "$LOG" | tee "$RUN_OUT1"' in t
    assert '2>&1 | tee -a "$LOG" | tee "$RUN_OUT2"' in t
    # And IMMEDIATELY after its own pipeline. PIPESTATUS holds the last
    # pipeline's codes only: one command between the two -- an echo, a test, a
    # mkdir -- replaces them, and every leg then reads the code of that command
    # instead. A leg at rc 2 would be reported as a passing cell and the job
    # would exit 0 on a study that cleared nothing.
    lines = [line.strip() for line in t.splitlines()]
    for tee, capture in ((f'2>&1 | tee -a "$LOG" | tee "$RUN_OUT{n}"',
                          f'RC{n}="${{PIPESTATUS[0]}}"') for n in (1, 2)):
        index = lines.index(tee)
        assert lines[index + 1] == capture, (tee, lines[index + 1])


def test_the_usage_marker_is_read_from_this_submissions_own_capture():
    """``$LOG`` is opened with ``tee -a`` and accumulates across
    resubmissions, so a usage line left in it by an earlier submission would
    be read as this one's. The classification greps the per-leg capture, which
    this submission writes fresh, for argparse's own anchored line."""
    t = _script_text()
    assert 'grep -q "^usage: probe_pretrain_energy_weight" "$run_out"' in t


def test_the_job_leaves_with_the_code_the_epilogue_computed():
    t = _script_text()
    assert 'exit "$RC"' in t


def test_the_script_parses_under_bash():
    """A syntax error in a job script is otherwise found by the queue, after
    the allocation started."""
    assert _SCRIPT.is_file(), f"no such file: {_SCRIPT}"
    done = subprocess.run(["bash", "-n", str(_SCRIPT)],
                          capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    assert done.stderr == "", done.stderr


# --------------------------------------------------------------------------- #
# The epilogue, executed
# --------------------------------------------------------------------------- #

def test_a_passing_cell_in_either_leg_is_exit_zero(tmp_path):
    """The study's answer is a lever that clears the certificate, so one cell
    of either leg clearing it is the job's success whatever the other leg
    found."""
    out, code = _run_epilogue(tmp_path, 0, 0)
    assert code == 0, out
    assert out.count("a cell PASSED the fidelity certificate") == 2, out
    out, code = _run_epilogue(tmp_path, 2, 0)
    assert code == 0, out
    assert "COMPLETED, no cell passed the certificate" in out, out
    assert "a cell PASSED the fidelity certificate" in out, out


def test_two_completed_legs_without_a_passing_cell_are_exit_two(tmp_path):
    """A finding, not a crash: every cell was measured and the tables were
    written. SLURM mails FAIL on any non-zero code, so the line that says
    which of the two a 2 was has to be in the log."""
    out, code = _run_epilogue(tmp_path, 2, 2)
    assert code == 2, out
    assert out.count("COMPLETED, no cell passed the certificate") == 2, out
    assert "REFUSED" not in out, out


def test_a_usage_refusal_is_not_reported_as_a_finding(tmp_path):
    """Argparse exits 2 as well, and writes no table. Read as the sweep's own
    2, a mistyped ``MGGA_LEVERS_*`` value would mail "no cell passed the
    certificate" for a job in which no cell ran -- a statement about the
    objective invented out of a shell typo."""
    out, code = _run_epilogue(tmp_path, 2, 0, capture1="usage")
    assert code == 1, out
    assert "REFUSED" in out, out
    assert "argparse rejected the command line" in out, out
    assert "COMPLETED, no cell passed the certificate" not in out, out


def test_a_lost_capture_is_not_reported_as_a_finding(tmp_path):
    """Without the leg's captured output the two meanings of 2 cannot be told
    apart, so the leg is reported as a failure rather than guessed at."""
    out, code = _run_epilogue(tmp_path, 2, 0, capture1="absent")
    assert code == 1, out
    assert "NO captured output" in out, out
    assert "COMPLETED, no cell passed the certificate" not in out, out


def test_a_failed_cell_and_an_escaped_exception_keep_their_own_codes(tmp_path):
    """1 and 3 are the sweep's codes for a failed cell and for an exception
    that escaped it. Either, in either leg, is the job's code; the first leg's
    is reported ahead of the second's, so a refusal in the integration leg
    cannot mask an escape in the weights leg."""
    out, code = _run_epilogue(tmp_path, 1, 0)
    assert code == 1, out
    assert "FAILED (rc=1)" in out, out
    out, code = _run_epilogue(tmp_path, 0, 3)
    assert code == 3, out
    assert "an exception ESCAPED the sweep" in out, out
    out, code = _run_epilogue(tmp_path, 3, 2, capture2="usage")
    assert code == 3, out
    assert "an exception ESCAPED the sweep" in out, out
    assert "REFUSED" in out, out


def test_the_epilogue_names_the_tables_that_exist_and_the_ones_that_do_not(tmp_path):
    """One leg can finish with its table written and the other leave none, and
    a resubmission resumes from what is on disk, so the log states which of
    the two tables it found."""
    out, code = _run_epilogue(tmp_path, 0, 0, table=False, table_arm=False)
    assert code == 0, out
    assert out.count("NO TABLE at") == 2, out
    assert "] table: " not in out, out
    out, code = _run_epilogue(tmp_path, 0, 0)
    assert code == 0, out
    assert out.count("] table: ") == 2, out
    assert "NO TABLE at" not in out, out


def test_exit_two_without_a_table_is_not_the_studys_finding(tmp_path):
    """Three different things exit 2: the sweep's own verdict, argparse's
    refusal, and an interpreter that could not open the probe at all -- a
    renamed script, a wrong ``XCQ_REPO``, a partial sync. The last writes
    neither a table nor a usage line, so a code test alone files a broken
    checkout as the physics result that no lever cleared the certificate, on a
    job whose header tells the operator to read a 2 as a finding. The finding
    is claimed only against the table the sweep writes after every cell."""
    out, code = _run_epilogue(tmp_path, 2, 0, table=False)
    assert code == 1, out
    assert "NO table at" in out, out
    assert "NOT a finding" in out, out
    assert "COMPLETED, no cell passed" not in out.split("integration leg")[0], out
    # With the table on disk the same code IS the finding.
    out, code = _run_epilogue(tmp_path, 2, 2)
    assert code == 2, out
    assert out.count("COMPLETED, no cell passed") == 2, out


def test_a_skipped_integration_leg_is_not_reported_as_a_passing_cell(tmp_path):
    """The wall guard leaves ``RC2`` at 0 when it skips the arm, which the code
    reader would otherwise pronounce a passing cell and exit 0 on. The arm is
    reported as unmeasured instead, and the job's code comes from the leg that
    ran."""
    out, code = _run_epilogue(tmp_path, 2, 0, skipped2=True)
    assert code == 2, out
    assert "NOT MEASURED" in out, out
    assert "a cell PASSED" not in out, out
    out, code = _run_epilogue(tmp_path, 0, 0, skipped2=True)
    assert code == 0, out
    assert out.count("a cell PASSED") == 1, out
    assert "NOT MEASURED" in out, out


def test_the_second_leg_is_skipped_when_a_cells_wall_does_not_remain():
    """A leg killed at the wall mid-cell writes no row, never reaches the
    epilogue and is mailed as a bare failure. The guard reads the wall SLURM
    leaves on the allocation and compares it with one cell's estimate, the
    upper end of the header's own figure."""
    t = _script_text()
    assert 'CELL_WALL_S="${MGGA_LEVERS_CELL_WALL_S:-5400}"' in t
    assert "SLURM_JOB_END_TIME" in t
    assert '[ "$LEFT_S" -lt "$CELL_WALL_S" ]' in t
    assert 'SKIPPED2="yes"' in t
    # 5400 s is the 1 h 28 upper end of the per-cell estimate the header states.
    assert "1 h 18 to 1 h 28" in t
