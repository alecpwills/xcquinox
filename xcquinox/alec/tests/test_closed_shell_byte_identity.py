"""Oracle O3: closed-shell results are unchanged, digit for digit.

rho_a = rho_b makes the three per-channel feature blocks identical -- doubling
either channel of [D/2, D/2] reproduces the matrix, and 2 rho_a / 4 sigma_aa
are then rho_tot / sigma_tot -- so the exact spin scaling of the pretraining
fidelity program (SPEC_pretrain_fidelity_program.md, Section 3.1) has no
closed-shell content at all: RKS and closed-shell UKS energies and potentials
must reproduce the archived tree exactly, not merely within a tolerance.

Two archived records are compared against, both produced by
``record_closed_shell_reference.py`` with the same pins:

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

The per-architecture comparison is also reached from
``test_spin_scaling_oracles`` (the module the workflow matrix selects with
``oracle_selector``); both entry points call
:func:`assert_closed_shell_record_matches`, and the record of one architecture
is computed once per process.
"""
import json
from pathlib import Path

import jax
import pytest

jax.config.update("jax_enable_x64", True)

import xcquinox.alec as alec
from xcquinox.alec.tests.record_closed_shell_reference import (
    RECORD_KEYS, closed_shell_record)

_FIXTURE_DIR = Path(__file__).parent / "fixtures"
#: The fixture the live tree must reproduce bitwise.
_FIXTURE = _FIXTURE_DIR / "closed_shell_reference_smooth_alpha.json"
#: The pre-program fixture, reproduced bitwise by every architecture without
#: an indicator column and within :data:`_SMOOTH_ALPHA_DELTA` by the five
#: meta-GGA architectures.
_FIXTURE_AE204537E = _FIXTURE_DIR / "closed_shell_reference_ae204537e.json"

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

#: The archived records, read on first use rather than at import. The oracle
#: module of the workflow matrix imports the comparison below, and a missing or
#: unreadable fixture must fail O3 rather than stop O1, O2 and O4 from being
#: collected at all.
_REFERENCE_CACHE = {}


def _reference(fixture=_FIXTURE):
    if fixture not in _REFERENCE_CACHE:
        _REFERENCE_CACHE[fixture] = json.loads(fixture.read_text())
    return _REFERENCE_CACHE[fixture]


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


def _record(arch_name):
    if arch_name not in _RECORDS:
        _RECORDS[arch_name] = closed_shell_record(arch_name)
    return _RECORDS[arch_name]


def assert_closed_shell_record_matches(arch_name):
    """The architecture's closed-shell record equals the live fixture exactly
    and the ae204537e fixture within the smoothing's measured footprint.

    The inputs are compared first: ``E_non_xc`` and the digest of the
    reference density matrix are properties of the precompute, not of the
    energy or potential path, and a difference there makes every downstream
    comparison meaningless rather than merely failing.
    """
    reference = _reference()[arch_name]
    archived = _reference(_FIXTURE_AE204537E)[arch_name]
    got = _record(arch_name)
    assert set(got) == set(reference) == set(archived) == set(RECORD_KEYS)
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


@pytest.mark.parametrize("arch_name", sorted(alec.ARCHITECTURES))
def test_closed_shell_results_are_byte_identical_to_the_archived_tree(
        arch_name):
    """Bitwise against the live (smooth-alpha) fixture; against the ae204537e
    fixture bitwise where no indicator column exists and within
    :data:`_SMOOTH_ALPHA_DELTA` where one does."""
    assert_closed_shell_record_matches(arch_name)
