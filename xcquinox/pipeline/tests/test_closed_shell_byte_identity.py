"""Oracle O3: closed-shell results are unchanged, within a stated tolerance.

rho_a = rho_b makes the three per-channel feature blocks identical -- doubling
either channel of [D/2, D/2] reproduces the matrix, and 2 rho_a / 4 sigma_aa
are then rho_tot / sigma_tot -- so the exact spin scaling of the pretraining
fidelity program has no
closed-shell content at all: RKS and closed-shell UKS energies and potentials
must reproduce the archived tree to within the last digits the machine itself
contributes, which :data:`REL_TOL` bounds.

Three archived records are compared against, all produced by
``record_closed_shell_reference.py`` with the same pins, and the live record
is produced the same way: by that script, run by path in a subprocess,
under the pins its ``__main__`` block sets before any numeric library loads
(the process confined to one CPU, the BLAS pools at one thread, XLA's Eigen
contractions single-threaded). Computed in the test process instead, the
record inherits that process: with Eigen multi-threaded the potential of an
attention architecture differs by one ulp (the three that failed the whole
suite on the recording workstation), and with Eigen single-threaded it still
differs with the CPU count XLA partitions its other parallel operations
over (20, 4 and 2 CPUs: three values of medium_attn's ``V_rks_trace``).
Both are properties of the process, not of the tree, and the pins hold them
fixed so that the comparison cannot read them as a moved code path:

* ``closed_shell_reference_ae204537e.json`` -- the tree at ae204537e, the last
  commit before the program's first code change. The spin-scaling change
  reproduced it on all 31 architectures bitwise (commit 96fb36fc3).
* ``closed_shell_reference_smooth_alpha.json`` -- the tree in which the lower
  bound of the iso-orbital indicator became a smooth positive part of width
  1e-5 (``metagga.compute_alpha``; docs/open_items.md entry 27). That change
  has closed-shell content by construction -- the indicator of every
  meta-GGA block moves by ``width^2 / (4 alpha_raw)`` away from zero and by
  up to ``width / 2`` at a one-orbital point -- so against the ae204537e
  fixture the 26 architectures without an indicator column are bitwise and
  the five meta-GGA architectures move by the measured amounts in
  :data:`_SMOOTH_ALPHA_DELTA` (H2O carries no one-orbital region, so the
  footprint is the ``width^2 / (4 alpha_raw)`` term integrated: 4.2e-11 Ha on
  the energies). Both were recorded on the previous stack (jax 0.7.0, pyscf
  2.11.0, numpy 2.3.4) and on the previous workstation (an i7-12700K).
* ``closed_shell_reference_jax0102.json`` -- the same closed-shell path on the
  current stack (jax 0.10.2, pyscf 2.14.0, numpy 2.5.3), recorded on the
  i7-5930K workstation with the reference SCF's density cutoff pinned at
  1e-7 (``pyscf_determinism.REFERENCE_SMALL_RHO_CUTOFF``: pyscf 2.14 prunes
  no grid point by default where 2.11 pruned at 1e-7, and with the pin the
  pruned grids of the two releases are identical bit for bit). The live
  record is compared against THIS fixture, which carries every registered
  architecture; against the smooth-alpha fixture every number moves by at
  most the stack's and the workstation's footprint together,
  :data:`_PREVIOUS_STACK_DELTA` (8.2e-14 absolute, 3.2e-15 relative, at
  worst), and against the ae204537e fixture by that plus the smoothing's on
  the five architectures with an indicator column.

Each record carries, beside the six closed-shell numbers, the two pins of the
record they were computed on (``E_non_xc`` and a digest of the reference
density matrix), so a moved INPUT is reported as a moved input rather than as
a moved code path.

THE TOLERANCE. The last digits of a record belong to the machine as much as to
the code. The recorder's pins (one PySCF thread, a fixed memory ceiling) make
a record reproducible across PROCESSES on the machine it was taken on; nothing
in them reaches the BLAS kernels the CPU selects or the compiled libraries
doing the arithmetic. Measured: the workflow-matrix smoke on an AMD Milan node
of SeaWulf read ``deep_3x16.E_non_xc`` = -67.00327081852355 against the
recording workstation's -67.0032708185235 -- three ulps of the double, 4.3e-14
Ha or 6.4e-16 relative. One comparison is therefore made, the same on every
machine:

* every number of the live leg within :data:`REL_TOL` relative, ``E_non_xc``
  among them, so a moved input is separated from a moved code path;
* the historical legs within the larger of :data:`REL_TOL` and their measured
  footprints, plus the smoothing's own footprint on the ae204537e leg of an
  architecture with an indicator column;
* the reference density matrix's digest reported rather than asserted: its
  last bits move with the machine exactly as the energies do (measured: the
  digest differs on a hosted runner) and a digest has no tolerance to be read
  at, so ``E_non_xc`` is what holds the input fixed.

The worst movement measured is carried in the report and in every failure
message.

The per-architecture comparison is also reached from
``test_spin_scaling_oracles`` (the module the workflow matrix selects with
``oracle_selector``); both entry points call
:func:`assert_closed_shell_record_matches` and print its report through
:func:`announce`, and the record of one architecture is computed once per
process.
"""
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import jax
import pytest

jax.config.update("jax_enable_x64", True)

import xcquinox.pipeline as pipeline
from xcquinox.pipeline.tests.record_closed_shell_reference import (
    PINS, RECORD_KEYS)

_FIXTURE_DIR = Path(__file__).parent / "fixtures"
#: The fixture the live tree is compared against: the current stack (jax
#: 0.10.2, pyscf 2.14.0, numpy 2.5.3) on the i7-5930K workstation, with the
#: reference SCF's density cutoff pinned (pyscf_determinism).
_FIXTURE = _FIXTURE_DIR / "closed_shell_reference_jax0102.json"
#: The previous live fixture: the same closed-shell path on the previous
#: stack (jax 0.7.0, pyscf 2.11.0, numpy 2.3.4) and workstation (an
#: i7-12700K), reproduced within :data:`_PREVIOUS_STACK_DELTA` by every
#: architecture it carries.
_FIXTURE_SMOOTH_ALPHA = _FIXTURE_DIR / "closed_shell_reference_smooth_alpha.json"
#: The pre-program fixture, on the previous stack and workstation: the
#: smooth-alpha fixture moved by :data:`_SMOOTH_ALPHA_DELTA` on the five
#: architectures with an indicator column and equal to it elsewhere.
_FIXTURE_AE204537E = _FIXTURE_DIR / "closed_shell_reference_ae204537e.json"
#: The previous stack's records, held to the larger of :data:`REL_TOL` and
#: their measured footprints.
_HISTORICAL_FIXTURES = (_FIXTURE_SMOOTH_ALPHA, _FIXTURE_AE204537E)
_FIXTURES = (_FIXTURE,) + _HISTORICAL_FIXTURES
_FIXTURE_IDS = ["jax0102", "smooth_alpha", "ae204537e"]

#: Relative tolerance of the comparison, per key, the same on every machine.
#: Anchored on both sides by measurement:
#:
#: * ABOVE the machine it has to absorb. The one discrepancy measured for this
#:   record between two machines is three ulps on ``E_non_xc``
#:   (-67.00327081852355 on the cluster's AMD Milan node against
#:   -67.0032708185235 on the recording workstation): 4.263e-14 Ha, 6.36e-16
#:   relative -- four orders below the tolerance.
#: * BELOW the code-path changes this oracle exists to catch. The closed shell
#:   either carries a change or it does not: the superseded total-density
#:   contract this program replaced sits 25.4 kcal/mol (4.0e-2 Ha) away on an
#:   open shell and exactly 0 on the closed shell. The smallest perturbation
#:   this module pins is 1e-9 Ha, which is 1.314e-11 relative on the largest
#:   key of the record (shallow's E_rks, -76.111) and 1.134e-10 on the
#:   smallest (deep_mgga_3x16's E_uks_closed, -8.8166) -- above the tolerance
#:   on every key of every architecture.
#:
#: What the tolerance does NOT resolve is the smallest DELIBERATE physics
#: change recorded in this program: the smooth positive part of the iso-orbital
#: indicator moved the meta-GGA architectures' closed-shell numbers by at most
#: 1.712e-10 absolute, which is 6.7e-12 relative on ``V_rks_sq`` and 5.5e-13
#: on ``E_rks`` -- below this tolerance. A sub-nanohartree movement of the
#: closed-shell path is not separable from the machine that computed it, so
#: this module reports the movement it measures rather than an equality it
#: cannot test; the smoothing's footprint enters the ae204537e leg's bound as
#: an explicit term instead (:data:`_SMOOTH_ALPHA_DELTA`).
REL_TOL = 1e-11

#: Largest movement of each closed-shell number between the ae204537e tree
#: and the smooth-alpha tree, measured on the recorder's H2O record over the
#: five meta-GGA architectures (worst architecture in each case
#: deep_mgga_attn_3x16): E_rks 4.165e-11, V_rks_trace 2.169e-11,
#: V_rks_sq 1.712e-10, E_uks_closed 4.165e-11, V_uks_a_trace 2.170e-11,
#: V_uks_a_sq 1.712e-10; the two input pins did not move. The bounds are
#: 2.4x the measured maxima, so a change of the shared code path that moved
#: a closed-shell number by more than the smoothing's own footprint would be
#: reported here as well as against the live fixture.
_SMOOTH_ALPHA_DELTA = {
    "E_rks": 1e-10, "V_rks_trace": 5.2e-11, "V_rks_sq": 4.1e-10,
    "E_uks_closed": 1e-10, "V_uks_a_trace": 5.2e-11, "V_uks_a_sq": 4.1e-10,
}

#: Largest movement of each closed-shell number between the smooth-alpha
#: fixture (jax 0.7.0, pyscf 2.11.0, numpy 2.3.4 on the i7-12700K) and the
#: jax0102 fixture (jax 0.10.2, pyscf 2.14.0, numpy 2.5.3 on the i7-5930K),
#: over the 31 architectures both carry, with the reference SCF's density
#: cutoff pinned so both are quadratures of the same grid: E_rks 4.263e-14
#: (deep), V_rks_trace 3.197e-14 (deep_mgga_attn_3x16), V_rks_sq 8.171e-14
#: (deep_3x16), E_uks_closed 1.954e-14 (deep), V_uks_a_trace 3.197e-14
#: (deep_attn_3x16), V_uks_a_sq 4.619e-14 (deep_rung35_mgga_3x16); 3.2e-15
#: relative at worst, three orders below the tolerance. The two
#: shares were measured apart on 2026-09-22 by recording the same tree on
#: the i7-5930K under both stacks: the stack alone moves the six numbers by
#: at most 4.263e-14 (E_non_xc bit-identical between the two releases, the
#: density matrix's digest not: numpy 2.5.3 bundles OpenBLAS 0.3.34 where
#: 2.3.4 bundled 0.3.30), the workstation alone by at most 7.105e-14. The
#: bounds are 2.4x the measured maxima, rounded up. The historical legs are
#: held to the larger of them and the machine's tolerance, which is three
#: orders above them on every key, so this table is the record of the
#: footprint and the tolerance is the bound.
_PREVIOUS_STACK_DELTA = {
    "E_rks": 1.1e-13, "V_rks_trace": 7.7e-14, "V_rks_sq": 2e-13,
    "E_uks_closed": 4.7e-14, "V_uks_a_trace": 7.7e-14, "V_uks_a_sq": 1.2e-13,
}

#: The archived records and the pins each was recorded under, read on first
#: use rather than at import. The oracle module of the workflow matrix imports
#: the comparison below, and a missing or unreadable fixture must fail O3
#: rather than stop O1, O2 and O4 from being collected at all. Two dicts
#: rather than one so that a test can substitute one fixture's records with
#: ``monkeypatch.setitem`` and leave its pins alone.
_RECORDS_BY_FIXTURE = {}
_PINS_BY_FIXTURE = {}


def _load(fixture):
    if fixture in _RECORDS_BY_FIXTURE:
        return
    document = json.loads(Path(fixture).read_text())
    absent = sorted({"pins", "records"} - set(document))
    if absent:
        raise ValueError(
            f"{Path(fixture).name} is not a closed-shell fixture: the blocks "
            f"{absent} are missing, found {sorted(document)}. Regenerate it "
            "with record_closed_shell_reference.py -- a record that does not "
            "state its pins follows the process it happened to run in rather "
            "than the tree, and cannot be compared with one that does.")
    stated = {name: document["pins"].get(name) for name in PINS}
    if stated != PINS:
        raise ValueError(
            f"{Path(fixture).name} was recorded under {document['pins']}, not "
            f"the recorder's pins {PINS}, so its last bits follow the process "
            "it happened to run in. Regenerate it with "
            "record_closed_shell_reference.py run by path, which pins itself.")
    _PINS_BY_FIXTURE[fixture] = document["pins"]
    _RECORDS_BY_FIXTURE[fixture] = document["records"]


def _reference(fixture=_FIXTURE):
    _load(fixture)
    return _RECORDS_BY_FIXTURE[fixture]


def _fixture_pins(fixture=_FIXTURE):
    _load(fixture)
    return _PINS_BY_FIXTURE[fixture]


def _bound(expected):
    """The tolerance on one key. Relative to the expected value, with 1.0 as
    the smallest divisor so a reference of zero would still carry a bound; no
    key of this record is below 8.8 in magnitude, so the guard never binds."""
    return REL_TOL * max(abs(float(expected)), 1.0)


def _has_indicator_column(arch_name):
    from xcquinox.pipeline.descriptors import MetaGGAAlphaDescriptor
    return any(isinstance(d, MetaGGAAlphaDescriptor)
               for d in pipeline.get_architecture(arch_name).materialize_descriptors())


#: The six numbers of the code path under test, and the two pins of the record
#: they were computed on. A mismatch in a pin is a moved input, which the
#: comparison reports separately: no statement about the code path can be made
#: on a record the two trees do not share.
_INPUT_PINS = ("E_non_xc", "dm_pbe_sha1")
_CODE_PATH_KEYS = tuple(k for k in RECORD_KEYS if k not in _INPUT_PINS)

_RECORDS = {}
#: The pins the recorder reported for each live record, by architecture.
_LIVE_PINS = {}


#: The recorder, invoked by PATH: run as a module (``-m``) the package
#: ``__init__`` imports precede its pin block, and XLA's thread pool, sized
#: when the backend first initializes, would already have been sized to
#: every CPU (measured: E_non_xc read the 20-CPU value under ``-m`` and the
#: one-CPU value by path).
_RECORDER = Path(__file__).with_name("record_closed_shell_reference.py")


def _recorder_environment():
    """The environment the recorder subprocess starts from: this process's,
    with the pinned variables set to the recorder's own values, so an
    inherited value (a worker environment carries XLA_FLAGS of its own) can
    neither leak into a record nor differ from what the recorder sets."""
    env = dict(os.environ)
    for name, value in PINS.items():
        if isinstance(value, str):      # the environment pins; the affinity
            env[name] = value           # and the import order are the
    return env                          # recorder's own to state


def _pinned_cpu():
    """The CPU the live recorder is confined to: the CPU the live fixture was
    recorded on when this process may run on it, so the record is taken on
    that core wherever it exists, and otherwise the lowest CPU of this
    process's set, which is what the recorder pins itself to. ``None`` where
    affinity cannot be read."""
    if not hasattr(os, "sched_getaffinity"):
        return None
    allowed = os.sched_getaffinity(0)
    recorded = _fixture_pins().get("cpu_index")
    if shutil.which("taskset") is not None and recorded in allowed:
        return recorded
    # Without taskset the recorder confines itself, to the lowest CPU of the
    # set it inherits, which is this one.
    return min(allowed)


def _one_cpu_prefix():
    """``taskset`` confining the recorder to :func:`_pinned_cpu`, applied from
    outside so the confinement precedes the interpreter itself; empty where
    ``taskset`` is not available (the recorder then confines itself to the
    lowest CPU of its set, before JAX initializes)."""
    cpu = _pinned_cpu()
    if shutil.which("taskset") is None or cpu is None:
        return ()
    return ("taskset", "-c", str(cpu))


def record_live(arch_name, *, prefix=()):
    """The live tree's closed-shell record for ``arch_name``, computed by the
    recorder script in a subprocess under its own pins.

    The archive was written by ``record_closed_shell_reference.py`` running
    by path, whose ``__main__`` block confines the process to one CPU, sets
    the BLAS pools to one thread and turns XLA's multi-threaded Eigen off
    before any numeric library loads. A record computed in this process
    inherits whatever this process runs with -- multi-threaded Eigen moves
    the potential of an attention architecture by one ulp, and the CPU count
    moves it again through XLA's other parallel operations
    (:data:`record_closed_shell_reference.PINS` carries the measurements) --
    so the live side is produced exactly as the archive was. ``prefix`` runs
    the recorder under a wrapper instead of the one-CPU ``taskset``.
    Returns ``(record, pins)``; a failed recorder raises with its stderr.
    """
    out = subprocess.run(
        [*(prefix or _one_cpu_prefix()), sys.executable, str(_RECORDER),
         "--arch", arch_name],
        env=_recorder_environment(), capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(
            f"the recorder failed for {arch_name} (exit {out.returncode}); "
            f"its stderr:\n{out.stderr}")
    document = json.loads(out.stdout)
    return document["records"][arch_name], document["pins"]


def _record(arch_name):
    if arch_name not in _RECORDS:
        record, pins = record_live(arch_name)
        assert {name: pins.get(name) for name in PINS} == PINS, (
            f"the recorder ran without its pins: {pins} != {PINS}")
        expected_cpu = _pinned_cpu()
        if expected_cpu is not None:
            assert pins.get("cpu_index") == expected_cpu, (
                f"the recorder was pinned to CPU {pins.get('cpu_index')}, "
                f"not the CPU {expected_cpu} this comparison selected")
        _RECORDS[arch_name] = record
        _LIVE_PINS[arch_name] = pins
    return _RECORDS[arch_name]


def assert_closed_shell_record_matches(arch_name, record=None):
    """The architecture's closed-shell record reproduces the live fixture
    within :data:`REL_TOL` relative per key, and the two historical fixtures
    within the larger of that and their measured footprints.

    The inputs are compared first: ``E_non_xc`` and the digest of the
    reference density matrix are properties of the precompute, not of the
    energy or potential path, and a difference there makes every downstream
    comparison meaningless rather than merely failing.

    ``record`` supplies the record instead of computing it, which is how the
    comparison is exercised without an SCF. Returns the one-line report,
    naming the worst movement measured.
    """
    if _record_pending(arch_name):
        pytest.skip(
            f"{arch_name}: no record in {_FIXTURE.name}; run "
            f"xcquinox/pipeline/tests/record_closed_shell_reference.py --arch "
            f"{arch_name} and merge its record")
    reference = _reference()[arch_name]
    # An architecture registered after a historical tree has no record in
    # that fixture; the comparison then runs without that leg and the report
    # says so. The lack of a HISTORICAL record never skips; only a name
    # listed in ``_AWAITING_RECORD`` and absent from the LIVE fixture is
    # skipped, by name, until its record is merged.
    historical = {fixture: _reference(fixture).get(arch_name)
                  for fixture in _HISTORICAL_FIXTURES}
    got = _record(arch_name) if record is None else record
    assert set(got) == set(reference) == set(RECORD_KEYS)
    for archived in historical.values():
        if archived is not None:
            assert set(archived) == set(RECORD_KEYS)
    return _assert_within_the_tolerance(arch_name, got, reference, historical)


def _historical_note(historical):
    """The report's statement of the historical legs: the fixtures absent for
    an architecture registered after their trees, if any."""
    absent = [Path(fixture).name for fixture, archived in historical.items()
              if archived is None]
    if not absent:
        return ""
    trees = "that tree" if len(absent) == 1 else "those trees"
    legs = "that leg" if len(absent) == 1 else "those legs"
    return (f"; {', '.join(absent)} absent for this architecture (registered "
            f"after {trees}), compared without {legs}")


def _historical_bound(fixture, arch_name, key, expected):
    """What a record of the current tree may differ from a historical
    fixture's by: the larger of the previous stack's measured footprint and
    the machine's tolerance, plus the smoothing's footprint on the ae204537e
    leg of an architecture with an indicator column (both are present in the
    same number)."""
    bound = max(_PREVIOUS_STACK_DELTA[key], _bound(expected))
    if fixture == _FIXTURE_AE204537E and _has_indicator_column(arch_name):
        bound += _SMOOTH_ALPHA_DELTA[key]
    return bound


def _assert_historical_legs(arch_name, got, historical, note):
    """The record against each historical fixture that carries this
    architecture: the non-XC energy within the tolerance (a larger difference
    is a moved input, not a moved code path) and the six numbers within
    :func:`_historical_bound`. Returns the worst movement measured as
    ``(relative, gap, key, fixture name)``, or None when no historical
    fixture carries the architecture."""
    measured = []
    for fixture, archived in historical.items():
        if archived is None:
            continue
        name = Path(fixture).name
        expected = archived["E_non_xc"]
        gap = abs(got["E_non_xc"] - expected)
        assert gap <= _bound(expected), (
            f"{arch_name}.E_non_xc: {got['E_non_xc']!r} is {gap:.3e} from "
            f"{name}'s {expected!r}, beyond the tolerance. The trees did not "
            "compute on the same record -- this is more than the machine -- "
            "so nothing can be concluded about the closed-shell code path "
            f"from this run. {note}")
        smoothed = (fixture == _FIXTURE_AE204537E
                    and _has_indicator_column(arch_name))
        for key in _CODE_PATH_KEYS:
            expected = archived[key]
            gap = abs(got[key] - expected)
            bound = _historical_bound(fixture, arch_name, key, expected)
            measured.append((gap / max(abs(expected), 1.0), gap, key, name))
            footprint = f"{_PREVIOUS_STACK_DELTA[key]:.1e}"
            if smoothed:
                footprint += (" plus the smooth positive part's "
                              f"{_SMOOTH_ALPHA_DELTA[key]:.1e}")
            footprint += ", or the machine's tolerance, whichever is larger"
            assert gap <= bound, (
                f"{arch_name}.{key}: {got[key]!r} is {gap:.3e} from {name}'s "
                f"{expected!r}, beyond the bound {bound:.3e} (the previous "
                f"stack's measured footprint {footprint}). Closed-shell "
                "results carry no per-channel content -- rho_a = rho_b makes "
                "the three feature blocks the same array -- so movement "
                "beyond the stack's and the machine's own is a change of the "
                f"shared code path. {note}")
    return max(measured) if measured else None


def _worst_historical(worst):
    """The report's statement of the historical legs' largest movement."""
    if worst is None:
        return ""
    relative, gap, key, name = worst
    return (f"; against the previous stack's records at worst {relative:.2e} "
            f"relative ({gap:.3e} on {key} against {name})")


def _assert_within_the_tolerance(arch_name, got, reference, historical):
    """Every key of the live leg within :data:`REL_TOL` relative, the
    historical legs within :func:`_historical_bound`, and the reference
    density matrix's digest reported instead of asserted."""
    note = (
        f"The comparison is held to {REL_TOL:.0e} relative per key, which is "
        "the machine's own last digits and no more: the arithmetic is done by "
        "the BLAS kernels the CPU selects and by the compiled libraries "
        "around them, which the recorder's pins do not reach.")
    measured = []

    def compare(key, expected, bound, reason):
        gap = abs(got[key] - expected)
        relative = gap / max(abs(expected), 1.0)
        measured.append((relative, gap, key, "live fixture"))
        assert gap <= bound, (
            f"{arch_name}.{key}: {got[key]!r} is {gap:.3e} ({relative:.2e} "
            f"relative) from the live fixture's {expected!r}, beyond the "
            f"bound {bound:.3e}. {reason} {note}")

    compare("E_non_xc", reference["E_non_xc"],
            _bound(reference["E_non_xc"]),
            "The trees did not compute on the same record -- this is more "
            "than the machine -- so nothing can be concluded about the "
            "closed-shell code path from this run.")
    # The digest is reported, not asserted: the reference SCF's own last bits
    # move with the machine exactly as the energies do, and a digest has no
    # tolerance to be read at. E_non_xc above is what holds the input fixed.
    digest = ("reference density digest matches"
              if got["dm_pbe_sha1"] == reference["dm_pbe_sha1"] else
              "reference density digest differs (expected on a machine other "
              "than the one that recorded the fixture)")
    for key in _CODE_PATH_KEYS:
        compare(key, reference[key], _bound(reference[key]),
                "Closed-shell results carry no per-channel content -- rho_a = "
                "rho_b makes the three feature blocks the same array -- so "
                "movement beyond the machine's own is an unintended change to "
                "the shared code path.")
    worst_historical = _assert_historical_legs(arch_name, got, historical,
                                               note)
    if worst_historical is not None:
        measured.append(worst_historical)
    relative, gap, key, against = max(measured)
    return (f"[O3] {arch_name}: within {REL_TOL:.0e} relative per key; worst "
            f"{relative:.2e} relative ({gap:.3e} on {key} against the "
            f"{against}); {digest}{_historical_note(historical)}")


#: The reports printed in this process. The workflow matrix runs one
#: architecture, so its oracle log carries one line; a full local run carries
#: what the comparison measured for each architecture it ran.
_ANNOUNCED = []


def announce(report, capsys):
    """Put the comparison's report on the real stdout.

    pytest swallows a passing test's output under ``-q``, which is how the
    workflow matrix runs the oracles, so the movement measured would otherwise
    be absent from the one log the matrix keeps. ``capsys.disabled()``
    suspends the capture for the write; the line is prefixed ``[O3]`` so a
    reader (or the matrix's own summary) can find it, and preceded by a
    newline so that it starts a line of its own rather than continuing
    pytest's progress dots -- which is what lets the log be searched for the
    marker at the start of a line.
    """
    _ANNOUNCED.append(report)
    with capsys.disabled():
        print(f"\n{report}")


@pytest.mark.parametrize("fixture", _FIXTURES, ids=_FIXTURE_IDS)
def test_the_reference_covers_every_architecture(fixture):
    """The live fixture carries every registered architecture except the
    names still awaiting their record (``_AWAITING_RECORD``, empty since the
    jax0102 recording), and a name listed there that the fixture DOES carry
    fails here, so the list expires on its own. The two historical fixtures
    carry every architecture that existed in their trees: the registry minus
    the ones registered afterwards (``_POST_ARCHIVE_ARCHS``), which those
    trees cannot have computed and which the comparison holds to the live
    fixture alone."""
    expected = set(pipeline.ARCHITECTURES)
    if fixture in _HISTORICAL_FIXTURES:
        expected -= set(_POST_ARCHIVE_ARCHS)
    else:
        recorded = set(_reference(fixture))
        stale = sorted(name for name in _AWAITING_RECORD if name in recorded)
        assert not stale, (
            f"recorded now; remove from _AWAITING_RECORD: {stale}")
        expected -= set(_AWAITING_RECORD) - recorded
    assert set(_reference(fixture)) == expected, (
        "the archived reference and the live architecture registry disagree; "
        "regenerate the fixture with record_closed_shell_reference.py"
    )


@pytest.mark.parametrize("fixture", _FIXTURES, ids=_FIXTURE_IDS)
def test_the_reference_carries_the_recorded_keys(fixture):
    """The fixture was written by the recorder this module imports; a record
    with fewer keys would make the comparison below silently partial."""
    for arch_name, record in sorted(_reference(fixture).items()):
        assert set(record) == set(RECORD_KEYS), arch_name
        assert all(isinstance(record[k], float) for k in _CODE_PATH_KEYS), (
            arch_name)


#: One configuration under two registry names: the v8 geometric entry and the
#: cusp twin it is registered from (every field but the name equal).
_SAME_CONFIGURATION = ("deep_geom_3x16", "deep_cusp_3x16")


def test_two_names_of_one_configuration_share_a_record():
    """The live fixture's records of ``deep_geom_3x16`` and ``deep_cusp_3x16``
    are equal key by key: the two names build the same network from the same
    seed, so a record that depended on the name (an initialization seeded
    from it, a feature block keyed by it) would separate them here."""
    records = _reference(_FIXTURE)
    geom, cusp = (records[name] for name in _SAME_CONFIGURATION)
    assert geom == cusp, {k: (geom.get(k), cusp[k]) for k in cusp
                          if geom.get(k) != cusp[k]}


# ---------------------------------------------------------------------------
# What the comparison binds on
# ---------------------------------------------------------------------------

#: The probe of the tolerance test: an architecture with no indicator column,
#: so its two fixture records are equal and both historical legs are exact
#: statements about the same numbers. Pinned by an assertion in the test.
_PROBE_ARCH = "deep_3x16"
#: Second probe: an architecture that DOES carry an indicator column, so the
#: ae204537e leg runs against a record the smoothing moved, whose bound
#: carries the smoothing's footprint as well as the tolerance.
_PROBE_MGGA_ARCH = "deep_mgga_3x16"


def _fixture_record(arch_name=_PROBE_ARCH):
    """What a re-run produces when nothing moved: the archived record."""
    return dict(_reference()[arch_name])


def test_the_tolerance_binds(monkeypatch):
    """The comparison is a bound and not a formality: the cluster's own
    three-ulp discrepancy passes, a one-ulp movement of any key passes, a
    1e-9 Ha movement of any key fails, and a historical record moved beyond
    its own bound fails that leg while the live leg still passes.

    Every case hands the record in rather than computing it, so the whole
    comparison is exercised without an SCF. The reference density digest is
    reported and not asserted -- a digest has no tolerance, and its last bits
    move with the machine exactly as the energies do -- so a changed digest
    appears in the report rather than raising; ``E_non_xc``, held at the
    tolerance, is the assertion that pins the input.
    """
    assert not _has_indicator_column(_PROBE_ARCH)

    # The record the cluster smoke actually produced: E_non_xc three ulps
    # away and a reference density matrix that no longer digests to the same
    # value.
    cluster = _fixture_record()
    cluster["E_non_xc"] = -67.00327081852355
    cluster["dm_pbe_sha1"] = "0" * 40
    assert cluster["E_non_xc"] != _fixture_record()["E_non_xc"]
    report = assert_closed_shell_record_matches(_PROBE_ARCH, record=cluster)
    assert "digest differs" in report
    assert f"within {REL_TOL:.0e} relative per key" in report

    for key in _CODE_PATH_KEYS + ("E_non_xc",):
        one_ulp = _fixture_record()
        one_ulp[key] = math.nextafter(one_ulp[key], math.inf)
        assert one_ulp[key] != _fixture_record()[key]
        assert abs(one_ulp[key] - _fixture_record()[key]) < 1e-13
        assert_closed_shell_record_matches(_PROBE_ARCH, record=one_ulp)
        nudged = _fixture_record()
        nudged[key] += 1e-9
        with pytest.raises(AssertionError, match=f"{_PROBE_ARCH}.{key}"):
            assert_closed_shell_record_matches(_PROBE_ARCH, record=nudged)

    # The historical legs bind as well: the smooth-alpha record of the probe
    # moved by twice its own bound on one key fails that leg, while the live
    # leg -- which the same handed-in record reproduces exactly -- passes.
    original = _reference(_FIXTURE_SMOOTH_ALPHA)
    archived = original[_PROBE_ARCH]["V_rks_sq"]
    moved = dict(original)
    moved[_PROBE_ARCH] = dict(original[_PROBE_ARCH])
    moved[_PROBE_ARCH]["V_rks_sq"] = archived + 2.0 * _historical_bound(
        _FIXTURE_SMOOTH_ALPHA, _PROBE_ARCH, "V_rks_sq", archived)
    monkeypatch.setitem(_RECORDS_BY_FIXTURE, _FIXTURE_SMOOTH_ALPHA, moved)
    with pytest.raises(AssertionError,
                       match=f"{_PROBE_ARCH}.V_rks_sq.*"
                             f"{_FIXTURE_SMOOTH_ALPHA.name}"):
        assert_closed_shell_record_matches(_PROBE_ARCH,
                                           record=_fixture_record())
    monkeypatch.setitem(_RECORDS_BY_FIXTURE, _FIXTURE_SMOOTH_ALPHA, original)
    assert "digest matches" in assert_closed_shell_record_matches(
        _PROBE_ARCH, record=_fixture_record())

    # The same on an architecture whose ae204537e record the smoothing moved,
    # so the leg whose bound carries that footprint runs too.
    assert _has_indicator_column(_PROBE_MGGA_ARCH)
    report = assert_closed_shell_record_matches(
        _PROBE_MGGA_ARCH, record=_fixture_record(_PROBE_MGGA_ARCH))
    assert "digest matches" in report
    for key in _CODE_PATH_KEYS:
        nudged = _fixture_record(_PROBE_MGGA_ARCH)
        nudged[key] += 1e-9
        with pytest.raises(AssertionError, match=f"{_PROBE_MGGA_ARCH}.{key}"):
            assert_closed_shell_record_matches(_PROBE_MGGA_ARCH, record=nudged)


#: An architecture registered after the ae204537e tree was recorded, so that
#: fixture carries no record for it: one of the width and depth completions of
#: the pure DFS meta-GGA. It carries an indicator column, which is
#: the archived leg most likely to read a record that is not there.
_POST_ARCHIVE_ARCH = "deep_mgga_3x32"

#: Every architecture registered after the two historical trees were
#: recorded: absent from both historical fixtures by construction, present in
#: the live one, and compared against the live fixture alone. A name added to
#: the registry joins this tuple or the historical coverage test names it.
_POST_ARCHIVE_ARCHS = ("deep_mgga_3x32", "deep_mgga_4x16", "deep_mgga_4x32",
                       "deep_geom_3x16", "deep_geom_attn_3x16")

#: Architectures registered after the live fixture was last recorded, awaiting
#: their records (the recorder command is in the skip message). Empty since
#: the jax0102 recording, which carries every registered architecture. A name
#: listed here that the fixture DOES carry fails the coverage test: remove it
#: from this tuple once its record is merged.
_AWAITING_RECORD = ()


def _record_pending(arch_name):
    """Whether ``arch_name`` still waits for its record: listed in
    ``_AWAITING_RECORD`` and absent from the current fixture."""
    return arch_name in _AWAITING_RECORD and arch_name not in _reference()


def test_the_tolerance_sits_between_the_machine_and_a_change():
    """The tolerance's two anchors, as numbers rather than as prose: four
    orders above the discrepancy measured between the recording workstation
    and the cluster's AMD node, and below a 1e-9 Ha movement of any key of any
    architecture."""
    cluster_relative = abs(-67.00327081852355 + 67.0032708185235) / 67.0
    assert 6.3e-16 < cluster_relative < 6.4e-16
    assert cluster_relative * 1e3 < REL_TOL
    biggest = max(abs(record[key]) for record in _reference().values()
                  for key in _CODE_PATH_KEYS + ("E_non_xc",))
    assert 1e-9 / biggest > REL_TOL
    # Nor is the previous stack's footprint: 8.171e-14 on V_rks_sq of
    # deep_3x16 (its largest), 3.2e-15 relative, so the historical legs are
    # tolerance comparisons too, with the footprint below the tolerance on
    # every key.
    assert (8.171e-14 / abs(_reference()["deep_3x16"]["V_rks_sq"])
            < REL_TOL / 1e3)
    # The smoothing's own footprint is NOT resolved at this tolerance: a
    # deliberate change of that size is separated from the machine only by the
    # ae204537e leg's bound, which carries it as an explicit term.
    worst = _reference()["deep_mgga_attn_3x16"]["V_rks_sq"]
    assert 1.712e-10 / abs(worst) < REL_TOL


@pytest.mark.parametrize("arch_name", ("deep_3x16", "deep_attn_3x16", "deep_mgga_3x16"))
def test_closed_shell_results_are_byte_identical_to_the_archived_tree(
        arch_name, capsys):
    """Within :data:`REL_TOL` of the live (jax0102) fixture per key, within
    the larger of that and :data:`_PREVIOUS_STACK_DELTA` of the smooth-alpha
    fixture, and within that plus :data:`_SMOOTH_ALPHA_DELTA` of the
    ae204537e fixture where an indicator column exists."""
    announce(assert_closed_shell_record_matches(arch_name), capsys)
