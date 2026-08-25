"""Oracle O3: closed-shell results are unchanged, digit for digit.

rho_a = rho_b makes the three per-channel feature blocks identical -- doubling
either channel of [D/2, D/2] reproduces the matrix, and 2 rho_a / 4 sigma_aa
are then rho_tot / sigma_tot -- so the exact spin scaling of the pretraining
fidelity program (SPEC_pretrain_fidelity_program.md, Section 3.1) has no
closed-shell content at all: RKS and closed-shell UKS energies and potentials
must reproduce the archived tree exactly, not merely within a tolerance.

Two archived records are compared against, both produced by
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
Both are properties of the process, not of the tree, and the bitwise
comparison must not read them as one:

* ``closed_shell_reference_ae204537e.json`` -- the tree at ae204537e, the last
  commit before the program's first code change. The spin-scaling change
  reproduced it on all 31 architectures bitwise (commit 96fb36fc3).
* ``closed_shell_reference_smooth_alpha.json`` -- the tree in which the lower
  bound of the iso-orbital indicator became a smooth positive part of width
  1e-5 (``metagga.compute_alpha``; DEFERRED_WORK.md entry 27). That change
  has closed-shell content by construction -- the indicator of every
  meta-GGA block moves by ``width^2 / (4 alpha_raw)`` away from zero and by
  up to ``width / 2`` at a one-orbital point -- and the record is bitwise
  against THIS fixture, while against the ae204537e fixture the 26
  architectures without an indicator column are still bitwise and the five
  meta-GGA architectures move by the measured amounts in
  :data:`_SMOOTH_ALPHA_DELTA` (H2O carries no one-orbital region, so the
  footprint is the ``width^2 / (4 alpha_raw)`` term integrated: 4.2e-11 Ha on
  the energies).

Each record carries, beside the six closed-shell numbers, the two pins of the
record they were computed on (``E_non_xc`` and a digest of the reference
density matrix), so a moved INPUT is reported as a moved input rather than as
a moved code path.

WHICH MACHINE. Bitwise equality is a statement about one machine. The
recorder's pins (one PySCF thread, a fixed memory ceiling) make a record
reproducible across PROCESSES on the machine it was taken on; nothing in them
reaches the BLAS kernels the CPU selects or the compiled libraries doing the
arithmetic, so the last digits belong to the machine as much as to the code.
Measured: the workflow-matrix smoke of 2026-08-24 (job 2134455, AMD Milan node
dn024 of SeaWulf) read ``deep_3x16.E_non_xc`` = -67.00327081852355 against this
workstation's -67.0032708185235 -- three ulps of the double, 4.3e-14 Ha or
6.4e-16 relative -- and the three-way bitwise assertion reported a physics
claim it was not testing. Each
fixture therefore states the platform it was recorded on
(``record_closed_shell_reference.PLATFORM_KEYS``) and the comparison has two
branches:

* :data:`BITWISE` where the running platform reproduces that block -- the
  three-way equality above, unchanged, and the branch the recording
  workstation takes by itself;
* :data:`CROSS_PLATFORM` where it does not -- every number held to
  :data:`CROSS_PLATFORM_REL_TOL` relative, the reference density matrix's
  digest reported rather than asserted (its last bits move with the machine
  exactly as the energies do), and the measured discrepancy and both
  fingerprints carried in the report and in every failure message.

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

import xcquinox.alec as alec
from xcquinox.alec.tests.record_closed_shell_reference import (
    PINS, PLATFORM_KEYS, RECORD_KEYS, platform_fingerprint)

_FIXTURE_DIR = Path(__file__).parent / "fixtures"
#: The fixture the live tree must reproduce bitwise.
_FIXTURE = _FIXTURE_DIR / "closed_shell_reference_smooth_alpha.json"
#: The pre-program fixture, reproduced bitwise by every architecture without
#: an indicator column and within :data:`_SMOOTH_ALPHA_DELTA` by the five
#: meta-GGA architectures.
_FIXTURE_AE204537E = _FIXTURE_DIR / "closed_shell_reference_ae204537e.json"
_FIXTURES = (_FIXTURE, _FIXTURE_AE204537E)

#: The two branches of the comparison, by the platform the record is read on.
BITWISE = "bitwise"
CROSS_PLATFORM = "cross-platform tolerance"

#: Relative floor of the :data:`CROSS_PLATFORM` branch, per key. Anchored on
#: both sides by measurement:
#:
#: * ABOVE the machine it has to absorb. The one cross-platform discrepancy
#:   measured for this record is three ulps on ``E_non_xc``
#:   (-67.00327081852355 on the cluster's AMD Milan node against
#:   -67.0032708185235 here): 4.263e-14 Ha, 6.36e-16 relative -- four orders
#:   below the floor.
#: * BELOW the code-path changes this oracle exists to catch. The closed shell
#:   either carries a change or it does not: the superseded total-density
#:   contract this program replaced sits 25.4 kcal/mol (4.0e-2 Ha) away on an
#:   open shell and exactly 0 on the closed shell. The smallest perturbation
#:   this module pins is 1e-9 Ha, which is 1.314e-11 relative on the largest
#:   key of the record (shallow's E_rks, -76.111) and 1.134e-10 on the
#:   smallest (deep_mgga_3x16's E_uks_closed, -8.8166) -- above the floor on
#:   every key of every architecture.
#:
#: What the floor does NOT resolve is the smallest DELIBERATE physics change
#: recorded in this program: the smooth positive part of the iso-orbital
#: indicator moved the meta-GGA architectures' closed-shell numbers by at most
#: 1.712e-10 absolute, which is 6.7e-12 relative on ``V_rks_sq`` and 5.5e-13
#: on ``E_rks`` -- below this floor. A change of that size is resolved by the
#: bitwise branch, which re-engages by itself on the recording platform; off
#: that platform a sub-nanohartree movement of the closed-shell path is not
#: separable from the machine, and this module says so rather than reporting
#: an equality it cannot test.
CROSS_PLATFORM_REL_TOL = 1e-11

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

#: The archived records and the platform each was recorded on, read on first
#: use rather than at import. The oracle module of the workflow matrix imports
#: the comparison below, and a missing or unreadable fixture must fail O3
#: rather than stop O1, O2 and O4 from being collected at all. Two dicts
#: rather than one so that a test can substitute one fixture's platform with
#: ``monkeypatch.setitem`` and leave its records alone.
_RECORDS_BY_FIXTURE = {}
_PLATFORM_BY_FIXTURE = {}
_PINS_BY_FIXTURE = {}


def _load(fixture):
    if fixture in _RECORDS_BY_FIXTURE:
        return
    document = json.loads(Path(fixture).read_text())
    if set(document) != {"platform", "pins", "records"}:
        raise ValueError(
            f"{Path(fixture).name} is not a closed-shell fixture: expected "
            f"the blocks ['pins', 'platform', 'records'], found "
            f"{sorted(document)}. Regenerate it with "
            "record_closed_shell_reference.py -- a record that does not state "
            "the platform it was taken on can be read neither bitwise nor at "
            "a tolerance, because which of the two applies is exactly what "
            "the platform decides, and one that does not state its pins "
            "cannot be read bitwise at all.")
    missing = sorted(set(PLATFORM_KEYS) - set(document["platform"]))
    if missing:
        raise ValueError(
            f"{Path(fixture).name} states an incomplete platform: {missing} "
            "missing. Regenerate it with record_closed_shell_reference.py.")
    stated = {name: document["pins"].get(name) for name in PINS}
    if stated != PINS:
        raise ValueError(
            f"{Path(fixture).name} was recorded under {document['pins']}, not "
            f"the recorder's pins {PINS}, so its last bits follow the process "
            "it happened to run in. Regenerate it with "
            "record_closed_shell_reference.py run by path, which pins itself.")
    _PLATFORM_BY_FIXTURE[fixture] = document["platform"]
    _PINS_BY_FIXTURE[fixture] = document["pins"]
    _RECORDS_BY_FIXTURE[fixture] = document["records"]


def _reference(fixture=_FIXTURE):
    _load(fixture)
    return _RECORDS_BY_FIXTURE[fixture]


def _fixture_platform(fixture=_FIXTURE):
    _load(fixture)
    return _PLATFORM_BY_FIXTURE[fixture]


def _fixture_pins(fixture=_FIXTURE):
    _load(fixture)
    return _PINS_BY_FIXTURE[fixture]


#: The running platform, measured once per process. A dict rather than a
#: module-level value for the same reason as the caches above.
_LIVE_PLATFORM = {}


def _live_platform():
    if "fingerprint" not in _LIVE_PLATFORM:
        _LIVE_PLATFORM["fingerprint"] = platform_fingerprint()
    return _LIVE_PLATFORM["fingerprint"]


def _platform_summary(fingerprint):
    """One line naming a platform, for a report or a failure message."""
    return (f"{fingerprint['cpu_model']}, numpy "
            f"{fingerprint['numpy_version']}, jax "
            f"{fingerprint['jax_version']}/{fingerprint['jaxlib_version']}, "
            f"pyscf {fingerprint['pyscf_version']} at "
            f"{fingerprint['pyscf_threads']} thread, blas "
            f"{fingerprint['blas']}, memory ceiling "
            f"{fingerprint['pinned_max_memory_mb']} MB")


def platform_differences():
    """The fields in which the running platform differs from a fixture's.

    Either fixture differing is enough: a record read on a platform that did
    not produce it is a cross-platform comparison whichever of the two it is.
    """
    live = _live_platform()
    found = {}
    for fixture in _FIXTURES:
        recorded = _fixture_platform(fixture)
        for key in PLATFORM_KEYS:
            if recorded[key] != live[key]:
                found.setdefault((key, repr(recorded[key])), []).append(
                    Path(fixture).name)
    return [f"{key}: {recorded} in {', '.join(names)} against "
            f"{live[key]!r} here"
            for (key, recorded), names in sorted(found.items())]


def comparison_mode():
    """:data:`BITWISE` on the platform and CPU the fixtures were recorded on,
    :data:`CROSS_PLATFORM` anywhere else. On the recording platform but
    another CPU the bitwise comparison is still attempted first and this
    names the branch a failure of it falls to."""
    if platform_differences() or pin_differences():
        return CROSS_PLATFORM
    return BITWISE


def _cross_platform_bound(expected):
    """The floor on one key. Relative to the expected value, with 1.0 as the
    smallest divisor so a reference of zero would still carry a bound; no key
    of this record is below 8.8 in magnitude, so the guard never binds."""
    return CROSS_PLATFORM_REL_TOL * max(abs(float(expected)), 1.0)


def _has_indicator_column(arch_name):
    from xcquinox.alec.descriptors import MetaGGAAlphaDescriptor
    return any(isinstance(d, MetaGGAAlphaDescriptor)
               for d in alec.get_architecture(arch_name).materialize_descriptors())


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
    recorded on when this process may run on it -- so a run whose lowest CPU
    is not the recording one still compares bitwise -- and otherwise the
    lowest CPU of this process's set, which is what the recorder pins itself
    to. ``None`` where affinity cannot be read."""
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


def pin_differences():
    """The ways the live recorder's pins differ from a fixture's beyond the
    pinned values themselves (which both are held to): the CPU it is confined
    to. On a hybrid part the potentials' last bits follow the core class
    (CPUs 0, 5 and 12 of the recording workstation agree; CPU 19, an
    efficiency core, does not), so a record from another CPU that does not
    match bitwise is held to the cross-platform floor instead. Empty on the
    recording CPU, which :func:`_pinned_cpu` selects whenever this process
    may run on it."""
    live = _pinned_cpu()
    found = []
    for fixture in _FIXTURES:
        recorded = _fixture_pins(fixture).get("cpu_index")
        if recorded != live:
            found.append(f"confined to CPU {live} here, {recorded} when "
                         f"{Path(fixture).name} was recorded")
    return found


def assert_closed_shell_record_matches(arch_name, record=None):
    """The architecture's closed-shell record equals the live fixture and the
    ae204537e fixture, bitwise on the platform they were recorded on and
    within :data:`CROSS_PLATFORM_REL_TOL` elsewhere.

    The inputs are compared first: ``E_non_xc`` and the digest of the
    reference density matrix are properties of the precompute, not of the
    energy or potential path, and a difference there makes every downstream
    comparison meaningless rather than merely failing.

    ``record`` supplies the record instead of computing it, which is how the
    branches below are exercised without an SCF. Returns the one-line report
    of the comparison, naming the branch that ran.
    """
    reference = _reference()[arch_name]
    archived = _reference(_FIXTURE_AE204537E)[arch_name]
    got = _record(arch_name) if record is None else record
    assert set(got) == set(reference) == set(archived) == set(RECORD_KEYS)
    differences = platform_differences()
    if differences:
        return _assert_within_the_cross_platform_floor(
            arch_name, got, reference, archived, differences)
    pinned_elsewhere = pin_differences() if record is None else []
    if not pinned_elsewhere:
        return _assert_bitwise(arch_name, got, reference, archived)
    # The recording platform, but the recorder confined to another CPU: the
    # strongest comparison is still tried, and only its failure is held to
    # the floor, since a record bit-identical to the fixtures says more than
    # the floor does whichever CPU produced it.
    try:
        return _assert_bitwise(arch_name, got, reference, archived)
    except AssertionError as failure:
        report = _assert_within_the_cross_platform_floor(
            arch_name, got, reference, archived, pinned_elsewhere)
        return (f"{report} (not bitwise on this CPU: "
                f"{str(failure).splitlines()[0]})")


def _assert_bitwise(arch_name, got, reference, archived):
    """The three-way equality, on the platform that recorded the fixtures."""
    for key in _INPUT_PINS:
        assert got[key] == reference[key] == archived[key], (
            f"{arch_name}.{key}: {got[key]!r} != archived {reference[key]!r} "
            f"/ {archived[key]!r}. The trees did not compute on the same "
            "record, so nothing can be concluded about the closed-shell code "
            "path from this run; the reference density matrix or the non-XC "
            "energy of the precompute has moved."
        )
    for key in _CODE_PATH_KEYS:
        assert got[key] == reference[key], (
            f"{arch_name}.{key}: {got[key]!r} != archived {reference[key]!r}. "
            "Closed-shell results carry no per-channel content -- rho_a = "
            "rho_b makes the three feature blocks the same array -- so any "
            "movement here is an unintended change to the shared code path."
        )
    if _has_indicator_column(arch_name):
        for key in _CODE_PATH_KEYS:
            assert abs(got[key] - archived[key]) <= _SMOOTH_ALPHA_DELTA[key], (
                f"{arch_name}.{key}: {got[key]!r} is "
                f"{got[key] - archived[key]:.3e} from the ae204537e tree's "
                f"{archived[key]!r}, beyond the smooth positive part's "
                f"measured footprint {_SMOOTH_ALPHA_DELTA[key]:.1e}"
            )
    else:
        for key in _CODE_PATH_KEYS:
            assert got[key] == archived[key], (
                f"{arch_name}.{key}: {got[key]!r} != ae204537e {archived[key]!r}. "
                "This architecture carries no iso-orbital indicator, so the "
                "smoothing of the indicator's lower bound cannot reach it; "
                "the shared code path moved."
            )
    return (f"[O3] {arch_name}: {BITWISE}; the running platform reproduces "
            f"the fixtures' own ({_platform_summary(_live_platform())})")


def _assert_within_the_cross_platform_floor(arch_name, got, reference,
                                            archived, differences):
    """The same comparison off the recording platform -- or on it, with the
    recorder confined to another CPU -- at the documented floor, with the
    density matrix's digest reported instead of asserted."""
    if platform_differences():
        cause = (f"This is not the platform the fixture was recorded on "
                 f"({'; '.join(differences)}), so the record's last digits "
                 "are a different machine's")
        label = "platform"
    else:
        cause = (f"This is the recording platform, but the recorder was "
                 f"confined to another CPU ({'; '.join(differences)}), so "
                 "the record's last digits may be another core's")
        label = "recorder"
    note = (
        f"[{CROSS_PLATFORM}] {cause} and the comparison is held to "
        f"{CROSS_PLATFORM_REL_TOL:.0e} relative per key rather than bitwise. "
        f"Recorded on: {_platform_summary(_fixture_platform())}. Running on: "
        f"{_platform_summary(_live_platform())}.")
    measured = []

    def compare(key, expected, bound, against, reason):
        gap = abs(got[key] - expected)
        relative = gap / max(abs(expected), 1.0)
        measured.append((relative, gap, key, against))
        assert gap <= bound, (
            f"{arch_name}.{key}: {got[key]!r} is {gap:.3e} ({relative:.2e} "
            f"relative) from the {against}'s {expected!r}, beyond the bound "
            f"{bound:.3e}. {reason} {note}")

    for against, expected in (("live fixture", reference["E_non_xc"]),
                              ("ae204537e fixture", archived["E_non_xc"])):
        compare("E_non_xc", expected, _cross_platform_bound(expected), against,
                "The trees did not compute on the same record -- this is more "
                "than the machine -- so nothing can be concluded about the "
                "closed-shell code path from this run.")
    # The digest is reported, not asserted: the reference SCF's own last bits
    # move with the machine exactly as the energies do, and a digest has no
    # tolerance to be read at. E_non_xc above is what holds the input fixed
    # on this branch.
    digest = ("reference density digest matches"
              if got["dm_pbe_sha1"] == reference["dm_pbe_sha1"] else
              "reference density digest differs (expected off the recording "
              "platform)")
    for key in _CODE_PATH_KEYS:
        compare(key, reference[key], _cross_platform_bound(reference[key]),
                "live fixture",
                "Closed-shell results carry no per-channel content -- rho_a = "
                "rho_b makes the three feature blocks the same array -- so "
                "movement beyond the machine's own is an unintended change to "
                "the shared code path.")
    indicator = _has_indicator_column(arch_name)
    for key in _CODE_PATH_KEYS:
        bound = _cross_platform_bound(archived[key])
        if indicator:
            # The smoothing's own footprint, or the machine's, whichever is
            # larger: this architecture carries an indicator column, so both
            # are present in the same number.
            bound = max(bound, _SMOOTH_ALPHA_DELTA[key])
            reason = ("Beyond the smooth positive part's measured footprint "
                      f"{_SMOOTH_ALPHA_DELTA[key]:.1e} and beyond the "
                      "machine's own floor.")
        else:
            reason = ("This architecture carries no iso-orbital indicator, so "
                      "the smoothing of the indicator's lower bound cannot "
                      "reach it; the shared code path moved.")
        compare(key, archived[key], bound, "ae204537e fixture", reason)
    relative, gap, key, against = max(measured)
    return (f"[O3] {arch_name}: {CROSS_PLATFORM} at "
            f"{CROSS_PLATFORM_REL_TOL:.0e} relative per key; worst "
            f"{relative:.2e} relative ({gap:.3e} on {key} against the "
            f"{against}); {digest}; {label} {'; '.join(differences)}")


#: Reports already printed in this process. The workflow matrix runs one
#: architecture, so its oracle log carries one line either way; a full local
#: run prints the branch once and then only what the cross-platform branch
#: measures per architecture.
_ANNOUNCED = []


def announce(report, capsys):
    """Put the comparison's report on the real stdout.

    pytest swallows a passing test's output under ``-q``, which is how the
    workflow matrix runs the oracles, so the branch that ran would otherwise
    be absent from the one log the matrix keeps. ``capsys.disabled()``
    suspends the capture for the write; the line is prefixed ``[O3]`` so a
    reader (or the matrix's own summary) can find it, and preceded by a
    newline so that it starts a line of its own rather than continuing
    pytest's progress dots -- which is what lets the log be searched for the
    marker at the start of a line.
    """
    if _ANNOUNCED and comparison_mode() == BITWISE:
        return
    _ANNOUNCED.append(report)
    with capsys.disabled():
        print(f"\n{report}")


@pytest.mark.parametrize("fixture", [_FIXTURE, _FIXTURE_AE204537E],
                         ids=["smooth_alpha", "ae204537e"])
def test_the_reference_covers_every_architecture(fixture):
    assert set(_reference(fixture)) == set(alec.ARCHITECTURES), (
        "the archived reference and the live architecture registry disagree; "
        "regenerate the fixture with record_closed_shell_reference.py"
    )


@pytest.mark.parametrize("fixture", [_FIXTURE, _FIXTURE_AE204537E],
                         ids=["smooth_alpha", "ae204537e"])
def test_the_reference_carries_the_recorded_keys(fixture):
    """The fixture was written by the recorder this module imports; a record
    with fewer keys would make the comparison below silently partial."""
    for arch_name, record in sorted(_reference(fixture).items()):
        assert set(record) == set(RECORD_KEYS), arch_name
        assert all(isinstance(record[k], float) for k in _CODE_PATH_KEYS), (
            arch_name)


@pytest.mark.parametrize("fixture", [_FIXTURE, _FIXTURE_AE204537E],
                         ids=["smooth_alpha", "ae204537e"])
def test_the_fixture_states_the_platform_it_was_recorded_on(fixture):
    """Without the platform block the comparison cannot know which of its two
    branches applies, so it is part of the fixture, not an annotation."""
    recorded = _fixture_platform(fixture)
    assert set(recorded) == set(PLATFORM_KEYS)
    assert recorded["cpu_model"] and isinstance(recorded["cpu_model"], str)
    assert recorded["pyscf_threads"] == 1, (
        "the record is reproducible only at one PySCF thread")
    assert recorded["pinned_max_memory_mb"] > 0.0
    # And the pins: one CPU, one thread per BLAS pool, single-threaded Eigen,
    # applied before any numeric library loaded; the CPU pinned to is stated
    # beside them (informative, not one of the pins).
    pins = _fixture_pins(fixture)
    assert {name: pins.get(name) for name in PINS} == PINS
    assert pins["cpu_affinity_count"] == 1
    assert pins["applied_before_imports"] is True
    assert isinstance(pins["cpu_index"], int)


def test_the_two_fixtures_were_recorded_on_the_same_platform():
    """Both records are read against one live record, so a difference between
    the two fixtures' machines would be charged to the code path."""
    assert _fixture_platform(_FIXTURE) == _fixture_platform(_FIXTURE_AE204537E)


def test_a_fixture_that_does_not_state_its_platform_is_refused(tmp_path):
    """The pre-fingerprint layout was a bare mapping of architectures. Read as
    if it were current it would take whichever branch the running platform
    happened to imply, which is the defect this module closes."""
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({"deep": dict(_reference()["deep"])}))
    with pytest.raises(ValueError, match="platform"):
        _reference(legacy)
    assert legacy not in _RECORDS_BY_FIXTURE
    incomplete = tmp_path / "incomplete.json"
    platform = dict(_fixture_platform())
    platform.pop("cpu_model")
    incomplete.write_text(json.dumps(
        {"platform": platform, "pins": PINS, "records": _reference()}))
    with pytest.raises(ValueError, match="cpu_model"):
        _reference(incomplete)


def test_a_fixture_recorded_without_the_pins_is_refused(tmp_path):
    """A record without its pins block, or one taken under other pins,
    carries last bits that followed the CPU set it ran on, and is refused
    rather than compared bitwise to a properly pinned live record."""
    unpinned = tmp_path / "unpinned.json"
    unpinned.write_text(json.dumps(
        {"platform": _fixture_platform(), "records": _reference()}))
    with pytest.raises(ValueError, match="pins"):
        _reference(unpinned)
    other = tmp_path / "other_pins.json"
    other.write_text(json.dumps(
        {"platform": _fixture_platform(),
         "pins": {**PINS, "cpu_affinity_count": 20},
         "records": _reference()}))
    with pytest.raises(ValueError, match="recorded under"):
        _reference(other)
    assert other not in _RECORDS_BY_FIXTURE


def test_the_two_fixtures_differ_only_where_the_indicator_is_present():
    """The live fixture is the ae204537e fixture moved by the smoothing of
    the indicator's lower bound and nothing else: identical on the 26
    architectures without an indicator column (pins included), and on the
    five with one moved by at most the measured footprint, in every key."""
    live, archived = _reference(), _reference(_FIXTURE_AE204537E)
    moved = []
    for arch_name in sorted(alec.ARCHITECTURES):
        for key in _INPUT_PINS:
            assert live[arch_name][key] == archived[arch_name][key], (
                arch_name, key)
        if _has_indicator_column(arch_name):
            moved.append(arch_name)
            for key in _CODE_PATH_KEYS:
                gap = abs(live[arch_name][key] - archived[arch_name][key])
                assert 0.0 < gap <= _SMOOTH_ALPHA_DELTA[key], (
                    arch_name, key, gap)
        else:
            assert live[arch_name] == archived[arch_name], arch_name
    assert len(moved) == 5, moved


# ---------------------------------------------------------------------------
# The two branches of the comparison
# ---------------------------------------------------------------------------

#: The probe of the branch tests: an architecture with no indicator column, so
#: its two fixture records are equal and both legs of either branch are exact
#: statements about the same numbers. Pinned by an assertion in each test.
_PROBE_ARCH = "deep_3x16"
#: Second probe: an architecture that DOES carry an indicator column, so the
#: ae204537e leg of either branch runs against a record the smoothing moved
#: (bitwise branch: the footprint; cross-platform branch: the larger of the
#: footprint and the floor).
_PROBE_MGGA_ARCH = "deep_mgga_3x16"


def _fixture_record(arch_name=_PROBE_ARCH):
    """What a re-run on the recording platform produces when nothing moved."""
    return dict(_reference()[arch_name])


def _pin_platform(monkeypatch, **overrides):
    """Read both fixtures as if recorded on the running platform, with
    ``overrides`` applied. Restored by ``monkeypatch`` at teardown."""
    for fixture in _FIXTURES:
        _fixture_platform(fixture)
        monkeypatch.setitem(_PLATFORM_BY_FIXTURE, fixture,
                            dict(_live_platform(), **overrides))


def test_a_matching_fingerprint_takes_the_bitwise_branch(monkeypatch):
    """On the recording platform the assertion is the three-way equality it
    always was: a record one ulp away from the fixture fails."""
    assert not _has_indicator_column(_PROBE_ARCH)
    _pin_platform(monkeypatch)
    assert comparison_mode() == BITWISE, platform_differences()
    report = assert_closed_shell_record_matches(_PROBE_ARCH,
                                                record=_fixture_record())
    assert BITWISE in report and CROSS_PLATFORM not in report
    for key in _CODE_PATH_KEYS + ("E_non_xc",):
        nudged = _fixture_record()
        nudged[key] = math.nextafter(nudged[key], math.inf)
        assert nudged[key] != _fixture_record()[key]
        assert abs(nudged[key] - _fixture_record()[key]) < 1e-13
        with pytest.raises(AssertionError, match=f"{_PROBE_ARCH}.{key}"):
            assert_closed_shell_record_matches(_PROBE_ARCH, record=nudged)
    digested = _fixture_record()
    digested["dm_pbe_sha1"] = "0" * 40
    with pytest.raises(AssertionError, match="dm_pbe_sha1"):
        assert_closed_shell_record_matches(_PROBE_ARCH, record=digested)


def test_an_unmatched_fingerprint_takes_the_cross_platform_branch(monkeypatch):
    """Off the recording platform the cluster's own three-ulp discrepancy
    passes and a 1e-9 Ha change of any key still fails."""
    assert not _has_indicator_column(_PROBE_ARCH)
    _pin_platform(monkeypatch)
    monkeypatch.setitem(
        _PLATFORM_BY_FIXTURE, _FIXTURE,
        dict(_live_platform(), cpu_model="AMD EPYC 7763 64-Core Processor"))
    assert comparison_mode() == CROSS_PLATFORM
    differences = platform_differences()
    assert len(differences) == 1 and differences[0].startswith("cpu_model")

    # The record the cluster smoke actually produced: E_non_xc three ulps
    # away and a reference density matrix that no longer digests to the same
    # value.
    cluster = _fixture_record()
    cluster["E_non_xc"] = -67.00327081852355
    cluster["dm_pbe_sha1"] = "0" * 40
    assert cluster["E_non_xc"] != _fixture_record()["E_non_xc"]
    report = assert_closed_shell_record_matches(_PROBE_ARCH, record=cluster)
    assert CROSS_PLATFORM in report
    assert "digest differs" in report
    assert "AMD EPYC 7763 64-Core Processor" in report
    assert "cpu_model" in report

    for key in _CODE_PATH_KEYS + ("E_non_xc",):
        one_ulp = _fixture_record()
        one_ulp[key] = math.nextafter(one_ulp[key], math.inf)
        assert CROSS_PLATFORM in assert_closed_shell_record_matches(
            _PROBE_ARCH, record=one_ulp)
        nudged = _fixture_record()
        nudged[key] += 1e-9
        with pytest.raises(AssertionError, match=f"{_PROBE_ARCH}.{key}"):
            assert_closed_shell_record_matches(_PROBE_ARCH, record=nudged)

    # The same on an architecture whose ae204537e record the smoothing moved,
    # so the leg that carries max(floor, footprint) runs as well.
    assert _has_indicator_column(_PROBE_MGGA_ARCH)
    report = assert_closed_shell_record_matches(
        _PROBE_MGGA_ARCH, record=_fixture_record(_PROBE_MGGA_ARCH))
    assert CROSS_PLATFORM in report and "digest matches" in report
    for key in _CODE_PATH_KEYS:
        nudged = _fixture_record(_PROBE_MGGA_ARCH)
        nudged[key] += 1e-9
        with pytest.raises(AssertionError, match=f"{_PROBE_MGGA_ARCH}.{key}"):
            assert_closed_shell_record_matches(_PROBE_MGGA_ARCH, record=nudged)


def test_the_cross_platform_floor_sits_between_the_machine_and_a_change():
    """The floor's two anchors, as numbers rather than as prose: four orders
    above the discrepancy measured between this workstation and the cluster's
    AMD node, and below a 1e-9 Ha movement of any key of any architecture."""
    cluster_relative = abs(-67.00327081852355 + 67.0032708185235) / 67.0
    assert 6.3e-16 < cluster_relative < 6.4e-16
    assert cluster_relative * 1e3 < CROSS_PLATFORM_REL_TOL
    biggest = max(abs(record[key]) for record in _reference().values()
                  for key in _CODE_PATH_KEYS + ("E_non_xc",))
    assert 1e-9 / biggest > CROSS_PLATFORM_REL_TOL
    # And the smoothing's own footprint is NOT resolved at this floor, which
    # is why the bitwise branch has to re-engage by itself where it can.
    worst = _reference()["deep_mgga_attn_3x16"]["V_rks_sq"]
    assert 1.712e-10 / abs(worst) < CROSS_PLATFORM_REL_TOL


@pytest.mark.parametrize("arch_name", sorted(alec.ARCHITECTURES))
def test_closed_shell_results_are_byte_identical_to_the_archived_tree(
        arch_name, capsys):
    """Bitwise against the live (smooth-alpha) fixture on the platform that
    recorded it, and against the ae204537e fixture bitwise where no indicator
    column exists and within :data:`_SMOOTH_ALPHA_DELTA` where one does; on
    any other platform the same comparison at
    :data:`CROSS_PLATFORM_REL_TOL`."""
    announce(assert_closed_shell_record_matches(arch_name), capsys)


@pytest.mark.skipif(shutil.which("taskset") is None,
                    reason="taskset is not available to change the CPU set")
def test_the_live_record_is_computed_under_the_recorders_pins():
    """Started on a four-CPU set, the recorder confines itself to one CPU
    before any numeric library loads and reports so; the record it returns
    equals the one the comparison uses, which is pinned the same way."""
    arch_name = "medium_attn"
    on_four, pins = record_live(arch_name, prefix=("taskset", "-c", "0-3"))
    assert {name: pins.get(name) for name in PINS} == PINS
    assert pins["cpu_index"] == 0
    assert on_four == _record(arch_name), {
        key: (on_four[key], _record(arch_name)[key]) for key in on_four
        if on_four[key] != _record(arch_name)[key]}


def test_a_recorder_run_as_a_module_states_that_its_pins_came_late(tmp_path):
    """``python -m`` executes the package imports before the pin block, so
    the numeric libraries are already loaded when the pins are set: with the
    BLAS pins absent from the environment every energy and the density
    digest move (measured), and even with them exported, as this test's
    environment does, the affinity pin comes after XLA's pool is sized and
    the potentials move. The record states that its pins came late, the
    recorder warns naming the libraries, and the loader refuses such a
    document rather than reading it bitwise."""
    out = subprocess.run(
        [sys.executable, "-m", "xcquinox.alec.tests.record_closed_shell_reference",
         "--arch", "deep"],
        env=_recorder_environment(), capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    document = json.loads(out.stdout)
    assert document["pins"]["applied_before_imports"] is False
    assert "WARNING" in out.stderr and "numpy" in out.stderr, out.stderr
    late = tmp_path / "late.json"
    late.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="recorded under"):
        _reference(late)
