"""Oracle O3: closed-shell results are unchanged, digit for digit.

rho_a = rho_b makes the three per-channel feature blocks identical -- doubling
either channel of [D/2, D/2] reproduces the matrix, and 2 rho_a / 4 sigma_aa
are then rho_tot / sigma_tot -- so the exact spin scaling of the pretraining
fidelity program (SPEC_pretrain_fidelity_program.md, Section 3.1) has no
closed-shell content at all: RKS and closed-shell UKS energies and potentials
must reproduce the archived tree exactly, not merely within a tolerance.

The reference numbers were produced by ``record_closed_shell_reference.py``
run against the tree at ae204537e, the last commit before the program's first
code change. Each record carries, beside the six closed-shell numbers, the two
pins of the record they were computed on (``E_non_xc`` and a digest of the
reference density matrix), so a moved INPUT is reported as a moved input
rather than as a moved code path.

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

_FIXTURE = (Path(__file__).parent / "fixtures"
            / "closed_shell_reference_ae204537e.json")

#: The archived record, read on first use rather than at import. The oracle
#: module of the workflow matrix imports the comparison below, and a missing or
#: unreadable fixture must fail O3 rather than stop O1, O2 and O4 from being
#: collected at all.
_REFERENCE_CACHE = {}


def _reference():
    if not _REFERENCE_CACHE:
        _REFERENCE_CACHE.update(json.loads(_FIXTURE.read_text()))
    return _REFERENCE_CACHE


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
    """The architecture's closed-shell record equals the archived one exactly.

    The inputs are compared first: ``E_non_xc`` and the digest of the
    reference density matrix are properties of the precompute, not of the
    energy or potential path, and a difference there makes every downstream
    comparison meaningless rather than merely failing.
    """
    reference = _reference()[arch_name]
    got = _record(arch_name)
    assert set(got) == set(reference) == set(RECORD_KEYS)
    for key in _INPUT_PINS:
        assert got[key] == reference[key], (
            f"{arch_name}.{key}: {got[key]!r} != archived {reference[key]!r}. "
            "The two trees did not compute on the same record, so nothing can "
            "be concluded about the closed-shell code path from this run; the "
            "reference density matrix or the non-XC energy of the precompute "
            "has moved."
        )
    for key in _CODE_PATH_KEYS:
        assert got[key] == reference[key], (
            f"{arch_name}.{key}: {got[key]!r} != archived {reference[key]!r}. "
            "Closed-shell results carry no per-channel content -- rho_a = "
            "rho_b makes the three feature blocks the same array -- so any "
            "movement here is an unintended change to the shared code path."
        )


def test_the_reference_covers_every_architecture():
    assert set(_reference()) == set(alec.ARCHITECTURES), (
        "the archived reference and the live architecture registry disagree; "
        "regenerate the fixture with record_closed_shell_reference.py"
    )


def test_the_reference_carries_the_recorded_keys():
    """The fixture was written by the recorder this module imports; a record
    with fewer keys would make the comparison below silently partial."""
    for arch_name, record in sorted(_reference().items()):
        assert set(record) == set(RECORD_KEYS), arch_name
        assert all(isinstance(record[k], float) for k in _CODE_PATH_KEYS), (
            arch_name)


@pytest.mark.parametrize("arch_name", sorted(alec.ARCHITECTURES))
def test_closed_shell_results_are_byte_identical_to_the_archived_tree(
        arch_name):
    assert_closed_shell_record_matches(arch_name)
