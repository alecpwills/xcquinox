"""xcquinox.alec.cluster.fidelity -- the per-architecture physics certificate.

Pretrained networks are accepted only when they reproduce their parent
functional in ENERGY units. For one architecture the certificate evaluates

    dE_xc = E_xc^NN[rho_parent] - E_xc^parent[rho_parent]

through the production energy path, on the parent's own self-consistent
density, at the run's SCF identity, for every free atom of the BH76 / W4-11
pools, the DFS pretraining molecules and three common molecules taken from the
pools; the molecular differences are folded against the free atoms into
atomization-energy offsets

    dAE(mol) = dE_xc(mol) - sum_atoms n_atom * dE_xc(atom).

PASS requires max |dE_xc| over the free atoms <= tol_atom (mHa) AND max |dAE|
<= tol_AE (kcal/mol) AND the oracle tests O1-O4 passing on the installed code
(SPEC_pretrain_fidelity_program.md Section 3.3, item 4). The parent is PBE for
a GGA-rung architecture and SCAN for a meta-GGA one (rungs.seed_xc_for_arch),
which is what each rung was pretrained against. The parent's E_xc on each
record is computed three independent ways (point-wise libxc on the stored
grid, PySCF numint on a fresh grid, the reference SCF's own accumulated
value) and a disagreement above PARENT_GRID_TOL_HA, a reference SCF that did
not converge, a non-finite measurement, or a system that could not be
evaluated each FAIL the certificate by name. Degenerate free atoms (open p shell) are evaluated on an
orientation-locked reference density when the run carries no lock of its own,
so the atomic numbers do not depend on the orientation the SCF converged to.

The verdict, every number, the run identity, the SHA-256 digests of the two
checkpoint files measured and the installed code version go to
``<run_dir>/pretrain/<arch>/fidelity_certificate.json``. The pretrain
worker, the train task, the preflight, the in-process model builder, the run
validator, the cross-arm merge and the figure suite all read that file through
:func:`certificate_status`, so the gate cannot drift between sites.

Invocation on a node::

    python -m xcquinox.alec.cluster.fidelity <RUN_DIR> <ARCH_IDX>

ENFORCEMENT HAS TWO LAYERS. The ON-NODE gates (the pretrain worker's exit
code, the train task, the preflight sweep) call :func:`gate_certificate`,
which honours the certificate's recorded ``enforced`` flag: a run configured
with ``fidelity.enforce: false`` (permitted only with a non-empty
``fidelity.override_reason``) still computes and writes the certificate with
its TRUE verdict, and the gates log it and continue. That exists for the
workflow-verification matrix, whose short pretraining runs cannot meet the
tolerance but must exercise the train and eval wiring with the physics on
record. The RECORD layers -- ``validate_run``, ``merge_v4_arms`` and the
figure loaders -- call :func:`certificate_status` and require PASS
unconditionally, so a non-enforcing run can never become a quantitative
result.

IMPORT WEIGHT (a contract, pinned by an AST test on this file's source): the
MODULE BODY carries no jax / equinox / pyscf / numpy import and no
``xcquinox`` import outside the cheap cluster readers (``grid_config``,
``domain``, ``materialize``). The test walks those readers' own module bodies
TRANSITIVELY, so a name is admitted only while its body stays under the same
prohibition: a whitelist checked one level deep forbids nothing, since a
reader that grew a heavy import would load it through this file unremarked.
Every jax / equinox / pyscf / ``xcquinox.alec.data`` import happens INSIDE a
function, so the login node CLI, the run validator, the train task's parent
process and the analysis layer read a certificate without this file pulling a
model or an SCF stack.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback

from xcquinox.alec.cluster.domain import KCAL_PER_HA
from xcquinox.alec.cluster.grid_config import (
    _canon_axis, load_grid_config, pretrain_checkpoint_dir,
)
from xcquinox.alec.cluster.materialize import (
    _sha256_file, _write_json_atomic, running_xcquinox_version)


CERTIFICATE_FILENAME = "fidelity_certificate.json"
VERDICT_PASS = "PASS"
VERDICT_FAIL = "FAIL"

# The two pretrained network files a verdict refers to, paired with the keys
# :func:`fidelity_certificate` records their SHA-256 digests under in the
# certificate's ``checkpoint`` block. One table for the writer and for
# ``validate_run``'s cross-check of the files present in a run: a rename on one
# side alone would leave the validator comparing a key nothing writes, which
# reads as a clean run rather than as a check that stopped working.
CHECKPOINT_DIGEST_KEYS: tuple[tuple[str, str], ...] = (
    ("xnet.eqx", "xnet_sha256"),
    ("cnet.eqx", "cnet_sha256"),
)

# One Hartree in kcal/mol, taken from the harness domain table rather than
# restated here (domain.KCAL_PER_HA, CODATA-2018, cited at its definition).
# The certificate's atomization offsets and the campaign's benchmark errors
# are read against the same kcal/mol tolerances, so a locally truncated copy
# would put the two on slightly different scales.
HA_TO_KCAL = KCAL_PER_HA
HA_TO_MHA = 1000.0

# The parent XC energy is computed three independent ways per system:
# point-wise on the stored precompute grid (the grid the network is integrated
# on, so the comparison is grid-exact), through PySCF's own nr_rks / nr_uks on
# a freshly built grid of the same level, and as the XC energy the reference
# SCF itself accumulated (the record's E_xc field). At sto-3g grid level 1 the
# routes agree to 2.6e-11 Ha on OH and 6.2e-11 Ha on H2O, for PBE and SCAN
# alike, and at the production identity to 2.0e-10 Ha (recorded in
# notebooks/analysis/NOTES_v5_mgga_vs_scan.md, Section 5). The bound below is
# more than three orders of magnitude above that spread and three below
# tol_atom = 1.0 mHa, so it fires when the stored grid and the molecule no
# longer describe the same system and never on integration noise. A
# disagreement is a FAIL of the certificate's own consistency, reported as
# such; the routes are never averaged.
PARENT_GRID_TOL_HA = 1e-6

# libxc names of the two parents.
_PARENT_XC = {"pbe": "PBE", "scan": "SCAN"}

# 2S for the Hund ground state of every neutral element the oracle set can
# dissociate into. The twelve the BH76 / W4-11 pools carry as free atoms
# (H, Be, B, C, N, O, F, Al, Si, P, S, Cl) hold the spins the pools themselves
# declare, asserted species by species in the tests; Li repeats the spin the
# DFS pretraining protocol declares for its own free Li atom; Na (3s^1) is
# reached through Na2 in the DFS molecules, and Mg (3s^2) completes the
# third-row pair. The table is the ATOMIZATION reference: a molecule's offset
# is folded against these atoms.
_ATOM_GROUND_SPIN: dict[str, int] = {
    "H": 1, "Li": 1, "Be": 0, "B": 1, "C": 2, "N": 3, "O": 2, "F": 1,
    "Na": 1, "Mg": 0, "Al": 1, "Si": 2, "P": 3, "S": 2, "Cl": 1,
}

# The three molecules the pre-certificate offsets were measured on
# (SPEC_pretrain_fidelity_program.md Section 2). Every certificate carries all
# three, whatever the architecture's rung, at the BH76 / W4-11 POOL geometry --
# mapped here from the pool's own species key to the certificate's canonical
# name. Resolving them from the pool rather than from a literal table is what
# makes dAE(H2O), dAE(N2) and dAE(CH4) one physical quantity across rungs: the
# DFS pretraining set carries N2 only at its GGA level and at a different bond
# length (1.0987920 A against the pool's 1.0971114 A) and CH4 at a different
# r(CH) (1.0918537 A against 1.0874456 A), so a DFS record of one of these
# three names must never win. The three are therefore certified at the pool
# geometry, not at the DFS pretraining geometry; the pool species are also the
# ones the held-out atomization energies are scored on. All three keys resolve
# to BH76 entries -- "H2O" and "CH4" exist only there, and "n2" is in both
# sets, where the merge keeps BH76 (load_full_held_out_pools) -- so the three
# come from one benchmark's geometries and the certificate does not invent a
# second merge policy. The lower-case W4-11 twins ("h2o", "ch4") carry the
# same molecules at other geometries and are deliberately not used. Pool H2O
# is r = 0.9569131 A with a 104.5169 degree bond angle.
_FIXED_MOLECULE_POOL_NAMES: tuple[tuple[str, str], ...] = (
    ("H2O", "H2O"),
    ("N2", "n2"),
    ("CH4", "CH4"),
)


# ---------------------------------------------------------------------------
# The certificate file: one path helper and one predicate, shared by every
# enforcement site so the gate cannot drift between them.
# ---------------------------------------------------------------------------

def certificate_path_in(pretrain_dir: str) -> str:
    """Certificate path inside a pretrain checkpoint directory."""
    return os.path.join(pretrain_dir, CERTIFICATE_FILENAME)


def certificate_path(run_dir: str, arch: str) -> str:
    """Certificate path for one architecture of a run."""
    return certificate_path_in(pretrain_checkpoint_dir(run_dir, arch))


def read_certificate(pretrain_dir: str) -> dict | None:
    """The parsed certificate, or ``None`` when absent or unparseable."""
    try:
        with open(certificate_path_in(pretrain_dir)) as f:
            payload = json.load(f)
    except (OSError, ValueError):
        return None
    return payload if isinstance(payload, dict) else None


def read_certificate_status_in(
        pretrain_dir: str) -> tuple[str, str, dict | None]:
    """``(status, reason, payload)`` from a SINGLE read of the certificate.

    :func:`certificate_status_in` and :func:`read_certificate` each open the
    file themselves, so a caller that wants both the classification and the
    numbers classifies one document and reads another. A certificate rewritten
    between the two opens then produces a mixed statement -- a status taken
    from the file as it was and numbers taken from the file as it became -- and
    such a pair describes no document that ever existed. A caller that reports
    and acts on the same certificate takes both from here.

    ``payload`` is the parsed object, or ``None`` when the file is absent or is
    not a JSON object, which is the rule :func:`read_certificate` applies.
    """
    path = certificate_path_in(pretrain_dir)
    if not os.path.isfile(path):
        return "MISSING", (
            f"no {CERTIFICATE_FILENAME} in {pretrain_dir}: the architecture "
            "was never checked against its parent functional"), None
    try:
        with open(path) as f:
            payload = json.load(f)
    except (OSError, ValueError) as exc:
        return "UNREADABLE", (
            f"{path} is not readable JSON ({type(exc).__name__}: {exc})"), None
    if not isinstance(payload, dict):
        return "UNREADABLE", f"{path} is not a JSON object", None
    verdict = payload.get("verdict")
    if verdict == VERDICT_PASS:
        return VERDICT_PASS, "fidelity certificate PASS", payload
    if verdict != VERDICT_FAIL:
        return "UNREADABLE", (
            f"{path} records verdict {verdict!r}, which is neither "
            f"{VERDICT_PASS!r} nor {VERDICT_FAIL!r}: the file states no "
            "outcome that can be acted on"), payload
    summary = payload.get("summary") or {}
    return VERDICT_FAIL, (
        f"fidelity certificate verdict {verdict!r} at {path} "
        f"(max_atom_mHa={summary.get('max_atom_mHa')}, "
        f"max_dAE_kcalmol={summary.get('max_dAE_kcalmol')}, "
        f"reasons={summary.get('failure_reasons')})"), payload


def certificate_status_in(pretrain_dir: str) -> tuple[str, str]:
    """``(status, reason)`` for the certificate in ``pretrain_dir``.

    ``status`` is ``"PASS"``, ``"FAIL"``, ``"MISSING"`` (no file) or
    ``"UNREADABLE"`` (file present but not a JSON object, or recording no
    verdict this module recognises). Only ``"PASS"`` releases a gate: an
    unreadable certificate is unverifiable, and an unverifiable certificate is
    refused.

    ``"FAIL"`` is returned only for a certificate that literally records
    ``"FAIL"``. An absent, misspelt or mis-cased verdict is UNREADABLE and not
    FAIL, because FAIL is the one status a run can waive through
    ``enforced: false``: reading an unrecognised verdict as FAIL would let a
    truncated or schema-less file be waived through an on-node gate.

    The classification is :func:`read_certificate_status_in`'s, so a caller
    that needs the document too reads it once rather than twice.
    """
    status, reason, _payload = read_certificate_status_in(pretrain_dir)
    return status, reason


def certificate_status(run_dir: str, arch: str) -> tuple[str, str]:
    """``(status, reason)`` for one architecture of a run.

    This is the RECORD-layer predicate: ``validate_run``, ``merge_v4_arms``
    and the figure loaders require ``VERDICT_PASS`` from it unconditionally.
    On-node gates call :func:`gate_certificate` instead.
    """
    return certificate_status_in(pretrain_checkpoint_dir(run_dir, arch))


def certificate_enforced_in(pretrain_dir: str) -> bool:
    """Whether the certificate in ``pretrain_dir`` says its verdict is acted on.

    A certificate written by a run with ``fidelity.enforce: false`` records
    ``"enforced": false``. Absent, unreadable, or written before the field
    existed -> ``True``: enforcement can only be waived by a certificate that
    exists to record the waiver.

    The waiver is the JSON literal ``false`` and nothing else. A truthiness
    test would read ``"enforced": null`` -- the value a writer that left the
    field unpopulated emits -- as a waiver, so a certificate no run ever asked
    to be non-enforcing would release a FAIL. A value of any other type is
    likewise not a waiver.
    """
    payload = read_certificate(pretrain_dir)
    if not payload:
        return True
    return payload.get("enforced", True) is not False


def gate_certificate(run_dir: str, arch: str) -> tuple[bool, str]:
    """``(allowed, message)`` for an ON-NODE gate.

    ``allowed`` is True when the certificate PASSes, and also when it exists,
    FAILs, records ``enforced: false`` AND names a non-empty STRING
    ``tolerances.override_reason`` -- the workflow-verification matrix, whose
    short pretraining runs cannot meet the tolerance yet must exercise the
    train and eval wiring with the real verdict written down. A MISSING or
    UNREADABLE certificate is never allowed: there is then no record of what
    was measured or of any waiver.

    The reason is required here and not only in the configuration.
    ``validate_grid_semantics`` refuses ``fidelity.enforce: false`` without a
    non-empty ``fidelity.override_reason`` and ``_build_fidelity`` refuses a
    non-string one -- ``str(False)`` is the non-empty string 'False', so a
    coerced boolean or number would pass a strip-only test and authorise
    disabled gates no author asked for. This gate re-imposes both rules on the
    certificate it reads, so a hand-edited certificate or
    ``resolved_config.yaml`` on a compute node cannot release a stage with no
    reason on the record.

    The record layers do NOT call this. ``validate_run``, ``merge_v4_arms``
    and the figure loaders require PASS through :func:`certificate_status`, so
    a non-enforcing run can never enter the record.

    The document is parsed ONCE, through :func:`gate_certificate_from_read`.
    """
    return gate_certificate_from_read(
        *read_certificate_status_in(pretrain_checkpoint_dir(run_dir, arch)))


def gate_certificate_from_read(status: str, reason: str,
                               payload: dict | None) -> tuple[bool, str]:
    """:func:`gate_certificate`'s rule applied to ONE already-read document.

    The three questions a waiver release turns on -- what the verdict is,
    whether enforcement is recorded as off, and what reason is recorded --
    are answered from a single :func:`read_certificate_status_in`. Asking the
    file each of them separately lets a certificate rewritten between the
    opens assemble a release out of three documents: a FAIL verdict from the
    first, ``enforced: false`` from the second and a reason from the third,
    where none of the three states all of it and none would be released on
    its own. Measured on that sequence, such a gate releases.

    A caller that also REPORTS the certificate takes the same triple, so the
    numbers it prints and the decision it acts on describe one file
    (``cluster._pretrain``).
    """
    if status == VERDICT_PASS:
        return True, reason
    if status != VERDICT_FAIL:
        return False, reason
    # A FAIL classification implies a parsed JSON object, so the payload is
    # the document the status was read from. The default is enforcement ON:
    # a waiver is the JSON literal false and nothing else, matching
    # certificate_enforced_in, which applies the same rule to its own read.
    payload = payload or {}
    if payload.get("enforced", True) is not False:
        return False, reason
    tolerances = payload.get("tolerances")
    if not isinstance(tolerances, dict):
        tolerances = {}
    recorded = tolerances.get("override_reason")
    # Prose or nothing: a non-string is refused rather than coerced, matching
    # grid_config._build_fidelity. str(False) is the non-empty string 'False',
    # so coercion would let `override_reason: false` -- or a bare 0 -- state a
    # reason it does not state.
    override = recorded.strip() if isinstance(recorded, str) else ""
    if not override:
        return False, (
            f"{reason}; the certificate records enforced=false but its "
            f"tolerances.override_reason ({recorded!r}) is not a non-empty "
            "string, so the waiver states no reason. Disabling the on-node "
            "gates requires a non-empty prose fidelity.override_reason, which "
            "validate_grid_semantics and _build_fidelity impose on the "
            "configuration and this gate re-checks on the certificate.")
    return True, (
        f"{reason}; enforcement is OFF for this run "
        f"(fidelity.enforce=false, override_reason: {override}) so the "
        "verdict is recorded and the stage continues. This run cannot enter "
        "validate_run, merge_v4_arms or the figure suite.")


# ---------------------------------------------------------------------------
# Parent functional and run identity
# ---------------------------------------------------------------------------

def resolve_parent(arch_name: str) -> str:
    """The parent functional an architecture must reproduce: ``"pbe"`` for a
    GGA-rung architecture, ``"scan"`` for a meta-GGA one.

    Derived from the architecture's RUNG (``rungs.seed_xc_for_arch``), not
    from ``inputs.seed_xc``: the parent is what the networks were pretrained
    against, a property of the architecture, while ``inputs.seed_xc`` selects
    the SCF starting density for training and may be pinned to "pbe" for a
    controlled experiment.
    """
    from xcquinox.alec.rungs import seed_xc_for_arch
    return seed_xc_for_arch(arch_name)


def dfs_level_for_parent(parent: str) -> str:
    """The DFS pretraining-set level matching a parent functional.

    The meta-GGA variant of the DFS protocol drops H2 and N2, so a SCAN-parent
    architecture is certified on the same 28 systems it was pretrained on.
    """
    return "mgga" if parent == "scan" else "gga"


def run_identity(cfg) -> dict:
    """The SCF / grid identity every certificate number is computed at.

    The five fields are exactly the run-level inputs that change an energy:
    basis, grid level, the Coulomb backend (density fitting plus its auxiliary
    basis) and the orientation-lock strength. ``validate_run`` refuses a
    certificate whose identity differs from the config's.
    """
    inp = cfg.inputs
    return {
        "basis": inp.basis,
        "grid_level": int(inp.grid_level),
        "density_fit": bool(getattr(inp, "density_fit", False)),
        "auxbasis": getattr(inp, "auxbasis", None),
        "orientation_lock_strength": float(
            getattr(inp, "orientation_lock_strength", 0.0)),
    }


# ---------------------------------------------------------------------------
# Does a certificate describe THESE networks, at THIS run's identity, against
# THIS architecture's parent?
# ---------------------------------------------------------------------------
# The verdict says an architecture reproduced its parent. It does not say
# which networks were measured, at which basis and grid, or against which
# parent -- those are recorded in the certificate beside the verdict, and a
# document that disagrees with the run on any of them bounds nothing in it.
# The comparison lives here, with the writer of those fields, and is applied
# by the record layer (``validate_run``) and by the pretrain stage's
# keep-a-completed-pretraining check alike: two implementations would drift,
# and the cost of the drift is a train and eval graph run against a
# certificate the record layer refuses at the end of it.
#
# Distinguishes "the key is not there" from "the key is there and null", which
# are different statements about a run's identity.
_ABSENT = object()


def show_identity(value) -> str:
    """Readable form of an identity value, naming an absent key as absent."""
    return "<absent>" if value is _ABSENT else repr(value)


def coerce_identity(key, value):
    """Read a recorded identity value in the type the config states it in.

    ``grid_level`` and ``orientation_lock_strength`` are numbers that a YAML
    or JSON round trip may present as ``"1"`` or ``1``; coercing them is what
    makes those the same identity rather than a spurious disagreement. A value
    that cannot be coerced is returned unchanged, so it compares unequal and is
    reported as a disagreement: a malformed field is a disagreement, and one
    of them may not abort a scan that has findings to report.
    """
    if value is _ABSENT or value is None:
        return value
    try:
        if key == "grid_level":
            return int(value)
        if key == "orientation_lock_strength":
            return float(value)
    except (TypeError, ValueError):
        return value
    return value


def identity_mismatches(cfg, cert) -> list:
    """``(field, recorded, expected)`` per identity field that disagrees.

    The expected identity comes from :func:`run_identity`, so a field added
    there is compared without a second edit, and the scan covers the UNION of
    the two key sets: a field the certificate states and the run does not is
    as much a disagreement as the reverse. An absent key on either side is
    reported through the :func:`show_identity` sentinel rather than as
    ``None``, since a run that states ``auxbasis: null`` and a certificate
    that omits the key entirely say different things.
    """
    recorded = cert.get("identity")
    if not isinstance(recorded, dict):
        recorded = {}
    expected = run_identity(cfg)
    out = []
    for key in sorted(set(expected) | set(recorded)):
        want = expected.get(key, _ABSENT)
        got = coerce_identity(key, recorded.get(key, _ABSENT))
        if got != want:
            out.append((key, got, want))
    return out


def model_class_mismatches(cfg, cert, arch_name=None) -> list:
    """``[(key, recorded, wanted), ...]`` for the model-class fields a
    certificate records -- ``parent_anchor``, ``descriptor_coordinates`` and
    ``descriptor_log_transform`` -- that differ from the class this run
    builds. A certificate written before the first two fields existed records
    neither, which reads as the unanchored legacy class it certified; a run of
    any other class must not accept it.

    The first two are read from the run's ``model`` block, which is what sets
    them for every architecture of the run. The third is a property of the
    ARCHITECTURE -- no run-level switch states it, and neither the polarized
    override nor ``config.apply_model_block`` touches it -- so its expected
    value is the registry entry's, for ``arch_name``: the architecture the
    CALLER is asking about, as :func:`parent_mismatch` is asked. With no name
    given the certificate's own is used, which is the run's architecture only
    where the caller has separately held it to that (both do, reporting a
    disagreement as a finding of its own); since 23 of the 31 registered
    architectures set the transform, a certificate from another architecture's
    directory agrees on this field more often than not, so the name is worth
    passing.

    ``descriptor_log_transform`` is compared ONLY WHERE THE CERTIFICATE STATES
    IT: every certificate written before that key carries the two class fields
    alone, and is read exactly as it was. An architecture that is not in the
    registry, or a certificate whose ``arch`` is not even a string, has no
    expected value at all -- the unresolvable architecture is already reported
    by both callers through :func:`parent_mismatch`.
    """
    model_block = getattr(cfg, "model", None)
    want_anchor = bool(getattr(model_block, "parent_anchor", False))
    want_coords = str(getattr(model_block, "descriptor_coordinates", "legacy"))
    got_anchor = bool(cert.get("parent_anchor", False))
    got_coords = str(cert.get("descriptor_coordinates", "legacy"))
    out = []
    if got_anchor != want_anchor:
        out.append(("parent_anchor", got_anchor, want_anchor))
    if got_coords != want_coords:
        out.append(("descriptor_coordinates", got_coords, want_coords))
    got_transform = cert.get("descriptor_log_transform")
    name = arch_name if arch_name is not None else cert.get("arch")
    if got_transform is not None and isinstance(name, str):
        # Imported here, as :func:`resolve_parent` imports
        # ``rungs.seed_xc_for_arch`` for the same kind of registry answer:
        # the module body stays a pure reader, and the registry is consulted
        # only for a certificate that states the field.
        from xcquinox.alec.config import get_architecture
        try:
            arch = get_architecture(name)
        except KeyError:
            arch = None
        if arch is not None:
            want_transform = bool(
                getattr(arch, "descriptor_log_transform", False))
            if bool(got_transform) != want_transform:
                out.append(("descriptor_log_transform", bool(got_transform),
                            want_transform))
    return out


def parent_mismatch(arch_name: str, cert):
    """``(recorded, expected)`` when the certificate names another parent.

    The parent functional is a property of the architecture's RUNG (PBE for
    GGA, SCAN for meta-GGA), so a certificate measured against the other one
    bounds nothing for this architecture. ``None`` when they agree.

    ``KeyError`` propagates from :func:`resolve_parent` for an architecture
    that is not in the registry: the parent is then not resolvable at all,
    which is a finding of its own and is stated by the caller.
    """
    expected = resolve_parent(arch_name)
    recorded = cert.get("parent")
    return None if recorded == expected else (recorded, expected)


def checkpoint_digest_findings(pretrain_dir: str, cert) -> list:
    """``(kind, filename, digest_key, recorded, measured)`` per network file.

    The verdict refers to two specific files, and comparing their digests is
    what ties it to the networks the train stage loads: a checkpoint rewritten
    (or re-pretrained) after certification is not the one that was measured.
    The file names and the payload keys come from
    :data:`CHECKPOINT_DIGEST_KEYS`, so the two sides of the comparison cannot
    drift apart.

    ``kind`` is one of:

    * ``"unmeasured"`` -- neither a digest nor a file, so there is nothing to
      cross-check. The only kind that is not a disagreement.
    * ``"unrecorded"`` -- the file is present and the certificate records no
      digest for it, so it cannot be tied to the verdict.
    * ``"no_file"`` -- a digest is recorded and no such file is present.
    * ``"unreadable"`` -- the file could not be read; ``measured`` carries the
      ``OSError``. Reported, never raised: a scan with findings to report may
      not be aborted by one unreadable file.
    * ``"mismatch"`` -- both are there and they differ.

    A pair that agrees produces no entry.
    """
    recorded = cert.get("checkpoint")
    if not isinstance(recorded, dict):
        recorded = {}
    out = []
    for fname, key in CHECKPOINT_DIGEST_KEYS:
        path = os.path.join(pretrain_dir, fname)
        want = recorded.get(key)
        on_disk = os.path.isfile(path)
        if want is None:
            out.append(("unmeasured" if not on_disk else "unrecorded",
                        fname, key, want, None))
            continue
        if not on_disk:
            out.append(("no_file", fname, key, want, None))
            continue
        try:
            got = _sha256_file(path)
        except OSError as exc:
            out.append(("unreadable", fname, key, want, exc))
            continue
        if got != want:
            out.append(("mismatch", fname, key, want, got))
    return out


def certificate_describes_run(cfg, pretrain_dir: str, arch_name: str,
                              cert) -> list:
    """Every way ``cert`` fails to describe this run, as short statements.

    An empty list means the certificate names this architecture, was written
    by the running code, records this run's identity and this architecture's
    parent, and carries the digests of the two networks now on disk -- the
    five facts the record layer re-checks (``validate_run``) beside the
    verdict, which the caller judges first by the release rule; an empty list
    is the statement that on those five facts the stages after this one
    would accept what is on disk. The version is compared with the running
    code's rather than with
    the manifest's, because the recovery that reaches this check re-runs the
    preflight, which stamps the manifest from the running code: a
    certificate written by an earlier deployment would then disagree with the
    manifest the record layer compares it with, after the whole train and
    eval graph has been spent.

    ``"unmeasured"`` is the one digest finding not reported: a certificate
    that records no digest for a file that is not there states nothing about a
    network, and the caller that cares about the missing network has already
    refused it on that ground.
    """
    out = []
    recorded_arch = cert.get("arch")
    if recorded_arch != arch_name:
        out.append(f"certificate names arch {recorded_arch!r}, not "
                   f"{arch_name!r} -- a file from another architecture's "
                   "directory certifies nothing here")
    recorded_version = cert.get("xcquinox_version")
    running = running_xcquinox_version()
    if recorded_version != running:
        out.append(f"certificate xcquinox_version {recorded_version!r} is not "
                   f"the running code's {running!r}, the version the "
                   "recovered preflight stamps into the manifest the record "
                   "layer compares it with")
    for key, got, want in identity_mismatches(cfg, cert):
        out.append(f"certificate identity {key}={show_identity(got)} but the "
                   f"config states {show_identity(want)}")
    try:
        mismatch = parent_mismatch(arch_name, cert)
    except KeyError:
        out.append(f"architecture {arch_name!r} is not in the registry, so "
                   "the parent functional its certificate must be measured "
                   "against cannot be resolved")
    else:
        if mismatch is not None:
            recorded, expected = mismatch
            out.append(f"certificate parent {recorded!r}, but this "
                       f"architecture's rung is pretrained against "
                       f"{expected!r}")
    for key, got, want in model_class_mismatches(cfg, cert, arch_name):
        out.append(f"certificate records {key}={got!r}, but this run builds "
                   f"{key}={want!r} -- the certified networks are not the "
                   "model class this run trains")
    for kind, fname, key, want, measured in checkpoint_digest_findings(
            pretrain_dir, cert):
        if kind == "unmeasured":
            continue
        if kind == "unrecorded":
            out.append(f"{fname} is present but the certificate records no "
                       f"{key}, so the file cannot be tied to the verdict")
        elif kind == "no_file":
            out.append(f"the certificate measured {fname} (sha256 "
                       f"{str(want)[:12]}...) but no such file is present")
        elif kind == "unreadable":
            out.append(f"{fname} could not be read to check it against the "
                       f"certificate ({measured})")
        else:
            out.append(f"{fname} sha256 {str(measured)[:12]}... is not the "
                       f"file the certificate measured "
                       f"({str(want)[:12]}...)")
    return out


def _distinct_archs(cfg):
    """The de-duplicated, sorted architecture list of the sweep.

    Uses ``grid_config._canon_axis`` -- the EXACT de-dup + sort ``expand_grid``
    applies to the arch axis -- so ``<arch_idx>`` selects the same
    architecture here as in ``cluster._pretrain``. Deliberately NOT imported
    from ``_pretrain``: that module imports this one for its gate.
    """
    return _canon_axis(cfg.sweep.arch)


# ---------------------------------------------------------------------------
# Oracle set
# ---------------------------------------------------------------------------

def atom_system_name(symbol: str, charge: int) -> str:
    """Canonical oracle-set name for a free atom: ``atom_O``, ``atom_F-``.

    The merged BH76 / W4-11 species dict carries the same oxygen atom under
    both ``O`` and ``o``, so the oracle set renames every free atom by element
    symbol and charge instead of carrying the pool key through.
    """
    if charge == 0:
        return f"atom_{symbol}"
    sign = "-" if charge < 0 else "+"
    magnitude = "" if abs(charge) == 1 else str(abs(charge))
    return f"atom_{symbol}{sign}{magnitude}"


def is_atom_system(mol_spec) -> bool:
    """True for a single free atom (one element, one nucleus)."""
    comp = tuple(mol_spec.atom_composition)
    return len(comp) == 1 and int(comp[0][1]) == 1


def is_degenerate_atom(mol_spec) -> bool:
    """True for a free atom whose ground state is spatially degenerate.

    An open p shell holding 1, 2, 4 or 5 electrons is a P term, and a
    self-consistent field can converge to any orientation of its hole: B, C,
    O, F, Al, Si, S, Cl and their ions with those p counts. The energy of the
    exact functional is orientation-invariant, but the quadrature on a fixed
    real-space grid is not, and the meta-GGA parent feels it most: the SCAN
    E_xc of the free O atom spreads by of order 0.1 mHa between independent
    unconstrained SCFs at def2-svp / grid 3 (0.26 mHa over one triple of
    runs, 0.084 mHa over another; 0.21 mHa for F at sto-3g / grid 1),
    against 1.6e-3 mHa for PBE, so an unlocked certificate number
    for such an atom depends on which orientation the SCF happened to
    reach. Half-filled (N, P), closed (Be, Mg, F-, Cl-) and s-shell (H, Li,
    Na) atoms are spherical and excluded. Elements beyond argon are refused:
    the rule covers the shells the oracle set can contain.
    """
    if not is_atom_system(mol_spec):
        return False
    from pyscf import gto
    symbol = str(mol_spec.atom_composition[0][0])
    n_electrons = int(gto.charge(symbol)) - int(mol_spec.charge)
    if n_electrons <= 4:            # 1s and 2s only
        n_p = 0
    elif n_electrons <= 10:         # 2p^1 .. 2p^6
        n_p = n_electrons - 4
    elif n_electrons <= 12:         # 3s
        n_p = 0
    elif n_electrons <= 18:         # 3p^1 .. 3p^6
        n_p = n_electrons - 12
    else:
        raise ValueError(
            f"free atom {symbol} (charge {mol_spec.charge}) carries "
            f"{n_electrons} electrons, beyond argon; the certificate's "
            "degeneracy rule covers the first three rows only")
    return n_p in (1, 2, 4, 5)


def atom_orientation_lock_strength() -> float:
    """The orientation-lock strength applied to degenerate free atoms when
    the run's own lock is off.

    ``orientation_lock.DEFAULT_STRENGTH`` (3e-5, the value the production
    configurations carry), imported at call time because that module's body
    carries numpy. Two independent locked SCFs of O and F agreed to 3.4e-11
    mHa in E_xc at def2-svp / grid 3, against an unlocked spread of order
    0.1 mHa. That locked agreement is identity-specific: at def2-svp /
    grid 1 the locked PBE O atom still spread 2.5e-3 mHa between
    second-order SCF landings.
    """
    from xcquinox.alec.orientation_lock import DEFAULT_STRENGTH
    return float(DEFAULT_STRENGTH)


def build_oracle_set(cfg, arch_name: str) -> tuple:
    """The certificate's systems for ``arch_name``, in a byte-stable order.

    Free atoms first (sorted by canonical name), then molecules (sorted by
    name), so the per-system table of two certificates is directly diffable.

    Composition:
      * every free atom of the BH76 / W4-11 pools, at the pool's own charge
        and spin (the species the held-out atomization energies are built from);
      * the DFS pretraining set at the level matching the architecture's
        parent (30 systems for a GGA rung, 28 for a meta-GGA one);
      * H2O, N2 and CH4 -- the three molecules the pre-certificate offsets were
        measured on -- at the POOL geometry, for every rung, overriding any DFS
        record of the same name so the three headline atomization offsets are
        the same physical quantity across architectures;
      * one neutral ground-state free atom for every element any molecule
        dissociates into (Li and Na appear in the DFS molecules but in neither
        pool), so every atomization offset can be formed.

    ELEMENT COVERAGE. Spec Section 3.3 item 2 describes the common molecules
    as "spanning the pool's elements"; the program's decision (Section 2) is
    the three molecules whose offsets were measured, and those span only H, C,
    N and O. B, Be and S are therefore free atoms of the oracle set that no
    oracle molecule contains: for those three elements the networks are
    certified as free atoms and never in a molecular environment, so no dAE
    constrains them. The divergence is deliberate -- the atomization tolerance
    is stated on the three molecules Section 2 tabulates offsets for, and a B-,
    Be- or S-bearing molecule would enter the verdict with no measured
    baseline behind it. Widening the molecular set is a change to the program,
    not to this function.
    """
    from xcquinox.alec.config import MoleculeSpec
    from xcquinox.alec.full_benchmark_pools import load_full_held_out_pools
    from xcquinox.alec.dfs_pretrain_set import dfs_pretrain_records

    basis = cfg.inputs.basis
    grid_level = cfg.inputs.grid_level
    level = dfs_level_for_parent(resolve_parent(arch_name))

    atom_spin: dict[tuple[str, int], int] = {}

    def _add_atom(symbol, charge, spin, source):
        key = (str(symbol), int(charge))
        previous = atom_spin.get(key)
        if previous is not None and previous != int(spin):
            raise ValueError(
                f"free atom {symbol} charge {charge} carries 2S={previous} "
                f"and 2S={spin} in different oracle sources ({source}); the "
                "certificate needs exactly one spin per free atom")
        atom_spin[key] = int(spin)

    pool_specs, _pool_reactions = load_full_held_out_pools(
        basis=basis, grid_level=grid_level)
    for ms in pool_specs.values():
        if is_atom_system(ms):
            _add_atom(ms.atom_composition[0][0], ms.charge, ms.spin,
                      "BH76/W4-11 pools")

    molecules: dict[str, object] = {}
    for record in dfs_pretrain_records(level):
        if record["kind"] == "atom":
            _add_atom(record["atom_composition"][0][0], record["charge"],
                      record["spin"], "DFS pretraining set")
            continue
        # The composition is sorted rather than trusted from the record, as
        # dfs_pretrain_systems sorts it: MoleculeSpec is frozen and hashes
        # every field, so two orderings of one molecule are two unequal specs
        # and two precompute-cache entries.
        molecules[record["name"]] = MoleculeSpec(
            name=record["name"], atom=record["atom"], basis=basis,
            charge=int(record["charge"]), spin=int(record["spin"]),
            atom_composition=tuple(sorted((str(s), int(n))
                                          for s, n in
                                          record["atom_composition"])),
            grid_level=grid_level)

    # The three common molecules override any DFS record of the same name, so
    # every rung's certificate measures the same H2O, N2 and CH4. The DFS
    # molecules that are not among the three keep their own geometries, which
    # are the geometries their pretraining rows were generated at.
    for name, pool_key in _FIXED_MOLECULE_POOL_NAMES:
        source = pool_specs.get(pool_key)
        if source is None:
            raise ValueError(
                f"the BH76 / W4-11 pools carry no species {pool_key!r}, so "
                f"the certificate's unconditional molecule {name!r} cannot be "
                "resolved to a pool geometry")
        molecules[name] = MoleculeSpec(
            name=name, atom=source.atom, basis=basis,
            charge=int(source.charge), spin=int(source.spin),
            atom_composition=tuple(sorted((str(s), int(n))
                                          for s, n in
                                          source.atom_composition)),
            grid_level=grid_level)

    for ms in molecules.values():
        for symbol, _count in ms.atom_composition:
            if (symbol, 0) in atom_spin:
                continue
            if symbol not in _ATOM_GROUND_SPIN:
                raise ValueError(
                    f"no ground-state spin recorded for element {symbol!r}; "
                    "add it to fidelity._ATOM_GROUND_SPIN before certifying "
                    "an architecture on a molecule that contains it")
            _add_atom(symbol, 0, _ATOM_GROUND_SPIN[symbol],
                      "ground-state table")

    atoms: dict[str, object] = {}
    for (symbol, charge), spin in atom_spin.items():
        name = atom_system_name(symbol, charge)
        atoms[name] = MoleculeSpec(
            name=name,
            atom=f"{symbol} 0.0000000000 0.0000000000 0.0000000000",
            basis=basis, charge=int(charge), spin=int(spin),
            atom_composition=((symbol, 1),), grid_level=grid_level)

    return (tuple(atoms[k] for k in sorted(atoms))
            + tuple(molecules[k] for k in sorted(molecules)))


# ---------------------------------------------------------------------------
# Model construction -- the production builder, not a second one
# ---------------------------------------------------------------------------

def _build_model(arch, pretrain_dir: str, *, seed: int):
    """Load a pretrained xnet/cnet pair through the production model builder.

    Mirrors ``train._build_model``: ``create_network_pair`` supplies a
    skeleton whose every array leaf ``eqx.tree_deserialise_leaves`` overwrites
    from the checkpoint, so the skeleton's seed never reaches the certified
    model; only the architecture (depth, width, attention, descriptor count)
    has to match, and it does by construction.
    """
    import equinox as eqx
    from xcquinox.alec.models import AlecGGAModel
    from xcquinox.alec.networks import create_network_pair
    xnet_skeleton, cnet_skeleton = create_network_pair(arch, seed=seed)
    xnet = eqx.tree_deserialise_leaves(
        os.path.join(pretrain_dir, "xnet.eqx"), xnet_skeleton)
    cnet = eqx.tree_deserialise_leaves(
        os.path.join(pretrain_dir, "cnet.eqx"), cnet_skeleton)
    return AlecGGAModel.from_arch(arch, xnet=xnet, cnet=cnet)


def build_certified_model(cfg, run_dir: str, arch_name: str):
    """``(arch, model)`` for ``arch_name`` as the run itself would build them.

    The registry entry is patched with the run-level polarized-correlation
    override exactly as ``cluster._pretrain`` patches it before pretraining,
    so the cnet input width matches the checkpoint on disk.
    """
    import dataclasses
    from xcquinox.alec.config import apply_model_block, get_architecture
    arch = get_architecture(arch_name)
    if getattr(cfg, "use_polarized_correlation", False):
        arch = dataclasses.replace(arch, use_polarized_correlation=True)
    # The run's model block (the parent anchor, the descriptor coordinates),
    # so the certified model is the class the checkpoint was pretrained as.
    model_block = getattr(cfg, "model", None)
    if model_block is not None:
        arch = apply_model_block(arch, model_block)
    pretrain_dir = pretrain_checkpoint_dir(run_dir, arch_name)
    return arch, _build_model(arch, pretrain_dir, seed=cfg.pretrain.seed)


# ---------------------------------------------------------------------------
# The parent's exchange-correlation energy on the record's own density
# ---------------------------------------------------------------------------

def _parent_xc_code(parent: str) -> str:
    """The libxc code of a parent, refusing an unknown one loudly."""
    try:
        return _PARENT_XC[parent]
    except KeyError:
        raise ValueError(
            f"unknown parent functional {parent!r}; the certificate knows "
            f"{sorted(_PARENT_XC)}") from None


def _parent_exc_on_stored_grid(mol_data, parent: str) -> float:
    """E_xc^parent on the SAME grid and density the network is evaluated on.

    Built from ``mol_data``'s stored AO derivative table, density matrix and
    grid weights, so the parent and the network see byte-identical quadrature
    with no Grids object in between. Assembling libxc's input rows here is an
    EVALUATION of the parent functional, not a second construction of a
    ``mol_data`` field: nothing computed in this function is ever stored back.

    The row set is the one libxc demands for the parent's rung, as in the
    precompute's own closed-shell reference route: the density alone for an
    LDA, the density and its gradient for a GGA, and for a meta-GGA the
    positive kinetic-energy density ``tau = 1/2 sum_ij P_ij grad chi_i . grad
    chi_j`` besides (PySCF passes no Laplacian to libxc; SCAN needs none).
    For an open shell the rows are per spin channel and libxc is called with
    ``spin=1``; the exchange-correlation energy density it returns is per
    electron of the TOTAL density, so the integral is
    ``sum_g w_g rho_g eps_g``.
    """
    import numpy as np
    from pyscf.dft import libxc, numint
    xc = _parent_xc_code(parent)
    xctype = libxc.xc_type(xc)
    if xctype not in ("LDA", "GGA", "MGGA"):
        raise ValueError(
            f"parent {parent!r} ({xc}) is of libxc type {xctype!r}; the "
            "certificate evaluates semilocal parents only")
    ao = np.asarray(mol_data["ao_grid_deriv"])        # (4, n_grid, nao)
    dm = np.asarray(mol_data["dm_pbe"])
    weights = np.asarray(mol_data["grid_weights"])
    unrestricted = bool(mol_data["is_unrestricted"])
    if unrestricted != (dm.ndim == 3):
        raise ValueError(
            f"record {mol_data['name']!r} is_unrestricted={unrestricted} but "
            f"carries a density matrix of shape {dm.shape}")

    def _rows(d):
        out = [np.einsum("gi,ij,gj->g", ao[0], d, ao[0])]
        if xctype in ("GGA", "MGGA"):
            out += [2.0 * np.einsum("gi,ij,gj->g", ao[k], d, ao[0])
                    for k in (1, 2, 3)]
        if xctype == "MGGA":
            out.append(0.5 * np.einsum("dgi,ij,dgj->g", ao[1:4], d, ao[1:4]))
        return np.vstack(out)

    ni = numint.NumInt()
    if unrestricted:
        rows_a, rows_b = _rows(dm[0]), _rows(dm[1])
        rho_total = rows_a[0] + rows_b[0]
        exc = ni.eval_xc(xc, (rows_a, rows_b), spin=1)[0]
    else:
        rows = _rows(dm)
        rho_total = rows[0]
        exc = ni.eval_xc(xc, rows, spin=0)[0]
    return float(np.sum(weights * rho_total * exc))


def _parent_exc_numint(mol_spec, parent: str, dm) -> float:
    """Independent E_xc^parent through PySCF's own ``nr_rks`` / ``nr_uks``.

    Cross-check of :func:`_parent_exc_on_stored_grid` on a freshly built grid
    of the same level, from a freshly built molecule and the record's density
    matrix. The reference SCF prunes its grid on the initial density
    (``small_rho_cutoff``, a bound of 1e-7 electrons on what is dropped), so
    the two point counts differ; the integrals agree to 2.6e-11 Ha on OH at
    sto-3g and 2.0e-10 Ha at the production identity. The difference is
    recorded per system and bounded by :data:`PARENT_GRID_TOL_HA`.
    """
    import numpy as np
    from pyscf import dft, gto
    from pyscf.dft import numint
    xc = _parent_xc_code(parent)
    mol = gto.M(atom=mol_spec.atom, basis=mol_spec.basis,
                charge=mol_spec.charge, spin=mol_spec.spin, verbose=0)
    grids = dft.Grids(mol)
    if mol_spec.grid_level is not None:
        grids.level = int(mol_spec.grid_level)
    grids.build()
    ni = numint.NumInt()
    dm = np.asarray(dm)
    if dm.ndim == 3:
        _nelec, exc, _vmat = ni.nr_uks(mol, grids, xc, dm)
    else:
        _nelec, exc, _vmat = ni.nr_rks(mol, grids, xc, dm)
    return float(exc)


# ---------------------------------------------------------------------------
# Per-system evaluation -- the seam the mocked tests replace
# ---------------------------------------------------------------------------

class ReferenceNotConverged(ValueError):
    """The record's reference SCF did not converge, so the network is not
    measured on it.

    Raised by :func:`evaluate_system` when the record's metadata reports
    ``reference_scf_converged: False``; :func:`fidelity_certificate` turns it
    into a named consistency failure rather than a generic evaluation error.
    ``cycles`` carries the recorded ``reference_scf_cycles`` (or ``None``).

    The producer's own refusal -- ``data.ReferenceSCFNotConverged``, a
    RuntimeError raised by ``precompute_fixed_density_data`` before any
    record exists -- is caught on the same certificate branch; this class
    remains for a record that arrives already stamped unconverged (a cache or
    another producer).
    """

    def __init__(self, message, *, cycles=None):
        super().__init__(message)
        self.cycles = cycles


def evaluate_system(model, descriptors, mol_spec, *, parent: str,
                    auxbasis=None, orientation_lock_strength: float = 0.0
                    ) -> dict:
    """dE_xc for one system, on the parent's own density at the run identity.

    The record comes from the library's ONE construction path with
    ``reference_xc=parent``, so its density matrix, grid quantities and every
    descriptor block -- total-density and per-spin-channel -- are the parent
    functional's, built by exactly the code the training pipeline uses. This
    module constructs none of them. A record whose ``reference_xc`` is not
    the parent is refused rather than measured, and so is one whose metadata
    reports an unconverged reference SCF (:class:`ReferenceNotConverged`);
    the convergence stamp and cycle count are copied into the record.

    ``E_xc_nn`` is ``oneshot.fixed_density_total_energy(model, mol_data)
    - mol_data["E_non_xc"]``: the production energy path, minus a term that
    cancels identically (``fixed_density_total_energy`` returns
    ``E_non_xc + E_xc^NN``, so whatever ``E_non_xc`` holds drops out).
    ``E_xc_parent`` is libxc on the same stored grid and density, cross-checked
    against a fresh-grid ``nr_rks``/``nr_uks`` (``parent_grid_diff_Ha``) and
    against the XC energy the reference SCF itself accumulated
    (``parent_record_diff_Ha``). Neither difference is averaged into the
    parent energy: both are recorded, and the certificate FAILS when either
    exceeds :data:`PARENT_GRID_TOL_HA`.

    ``seed_source`` is deliberately left at its default: ``dm_seed`` is the SCF
    starting guess, which the fixed-density energy path never reads, and
    requesting the SCAN seed would demand a seed cache the certificate does not
    need -- the parent density here comes from ``reference_xc``, not from the
    seed axis.
    """
    import numpy as np
    from xcquinox.alec.data import precompute_fixed_density_data
    from xcquinox.alec.oneshot import fixed_density_total_energy

    t0 = time.time()
    required = tuple(sorted({k for d in descriptors
                             for k in d.required_mol_keys}))
    mol_data = precompute_fixed_density_data(
        mol_spec, required_keys=required, descriptors=descriptors,
        auxbasis=auxbasis,
        orientation_lock_strength=orientation_lock_strength,
        reference_xc=parent)
    got_reference = mol_data["reference_xc"]
    if got_reference != parent:
        raise ValueError(
            f"the precompute returned a record with reference_xc="
            f"{got_reference!r} for {mol_spec.name!r} but the certificate "
            f"asked for {parent!r}; the network would be measured against a "
            "density its parent functional did not produce")

    # The reference SCF's convergence stamp, written by the precompute into
    # the record's metadata. A record that reports an unconverged reference is
    # refused outright: an SCF stopped short of self-consistency is not the
    # parent's density (measured on H2O / SCAN at max_cycle=1: +7.2e-2 Ha in
    # the total energy, 0.315 in the density matrix), and the certificate
    # would otherwise compare the network against a density no functional
    # produced. The precompute itself raises data.ReferenceSCFNotConverged
    # before returning such a record, so this check guards records that
    # arrive stamped from elsewhere; a record carrying no stamp records
    # ``None`` and is not refused here.
    meta = mol_data.get("mol_metadata") or {}
    scf_converged = meta.get("reference_scf_converged")
    scf_cycles = meta.get("reference_scf_cycles")
    if scf_converged is False:
        raise ReferenceNotConverged(
            f"the reference {parent.upper()} SCF for {mol_spec.name!r} did "
            f"not converge (reference_scf_cycles={scf_cycles}); the network "
            "is not measured on an unconverged density", cycles=scf_cycles)

    dm = np.asarray(mol_data["dm_pbe"])
    e_xc_nn = (float(fixed_density_total_energy(model, mol_data))
               - float(mol_data["E_non_xc"]))
    e_xc_parent = _parent_exc_on_stored_grid(mol_data, parent)
    e_xc_parent_numint = _parent_exc_numint(mol_spec, parent, dm)
    # The XC energy PySCF accumulated during the reference SCF itself. Free,
    # and a third independent route to the same number.
    e_xc_parent_record = float(mol_data["E_xc_pbe"])
    n_grid = int(np.asarray(mol_data["grid_weights"]).shape[0])
    del mol_data

    return {
        "name": mol_spec.name,
        "spin": int(mol_spec.spin),
        "charge": int(mol_spec.charge),
        "is_atom": is_atom_system(mol_spec),
        "n_grid": n_grid,
        "reference_xc": got_reference,
        "reference_scf_converged": scf_converged,
        "reference_scf_cycles": scf_cycles,
        "orientation_lock_strength": float(orientation_lock_strength),
        "E_xc_nn": e_xc_nn,
        "E_xc_parent": e_xc_parent,
        "E_xc_parent_numint": e_xc_parent_numint,
        "E_xc_parent_record": e_xc_parent_record,
        "parent_grid_diff_Ha": e_xc_parent - e_xc_parent_numint,
        "parent_record_diff_Ha": e_xc_parent - e_xc_parent_record,
        "dE_xc_mHa": (e_xc_nn - e_xc_parent) * HA_TO_MHA,
        "duration_s": time.time() - t0,
    }


# ---------------------------------------------------------------------------
# The certificate
# ---------------------------------------------------------------------------

def _fmt_secs(seconds) -> str:
    """Compact h:mm:ss / m:ss formatting for elapsed time and ETA."""
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


# The per-record quantities that must be finite for the certificate to act on
# the record. E_xc_parent and its two cross-checks are included beside the
# network-side numbers: a non-finite parent makes every derived difference
# meaningless, and nulling it keeps the written file strict JSON.
_FINITE_RECORD_KEYS = ("E_xc_nn", "E_xc_parent", "E_xc_parent_numint",
                       "E_xc_parent_record", "dE_xc_mHa",
                       "parent_grid_diff_Ha", "parent_record_diff_Ha")


def _null_non_finite(rec) -> list:
    """Null every non-finite measurement in ``rec``; return the key names.

    ``nan > tol`` is False for every tolerance and ``max()`` either returns
    NaN or swallows it depending on its position in the sequence, so a
    non-finite number must never reach a comparison: the value is recorded as
    ``None`` (JSON null -- the bare ``NaN`` token json.dump would emit is not
    RFC 8259 JSON), the affected keys are listed under ``non_finite`` and the
    certificate fails by name before any tolerance is consulted. A
    non-numeric value is treated the same way.
    """
    bad = []
    for key in _FINITE_RECORD_KEYS:
        value = rec.get(key)
        if value is None:
            continue
        try:
            finite = math.isfinite(float(value))
        except (TypeError, ValueError):
            finite = False
        if not finite:
            bad.append(key)
    for key in bad:
        rec[key] = None
    if bad:
        rec["non_finite"] = bad
    return bad


def _write_certificate_payload(payload: dict, path: str) -> None:
    """Atomically write the certificate, refusing any non-finite float.

    ``json.dump``'s default serializer emits bare ``NaN`` / ``Infinity``
    tokens, which RFC 8259 does not define: a strict reader refuses the file
    and a lenient one round-trips a value no tolerance can act on. The
    payload is serialized with ``allow_nan=False`` first -- a ``ValueError``
    here means a non-finite number escaped the per-record nulling, a bug to
    surface, not a verdict to write -- and only then handed to the shared
    atomic writer, which serializes the identical value set.
    """
    json.dumps(payload, allow_nan=False)
    _write_json_atomic(payload, path)


def fidelity_certificate(cfg, run_dir: str, arch_name: str, *,
                         oracle_set=None, evaluate=None, log=None) -> dict:
    """Certify one architecture and write its certificate; return the payload.

    ``oracle_set`` overrides :func:`build_oracle_set` (a short list for a
    probe or a test); ``evaluate`` overrides :func:`evaluate_system` (the seam
    the schema tests replace so no SCF runs); ``log`` is an optional callable
    given one progress line per system, so a node log shows the sweep moving
    through dozens of production-basis SCFs rather than falling silent.

    The verdict is PASS only when every system was evaluated on a converged
    reference SCF, every recorded measurement is finite, the free-atom and
    atomization tolerances of ``cfg.fidelity`` hold, and the three parent
    routes agree within :data:`PARENT_GRID_TOL_HA` on every system, each
    offending system named in the reason. An unconverged reference (the
    producer's ``data.ReferenceSCFNotConverged`` or a record stamped
    unconverged) and a non-finite measurement are each a named failure of
    their own, never folded into the generic evaluation errors; a non-finite
    value is nulled before any maximum or comparison is formed, so a NaN can
    neither pass a gate nor mask a finite excess elsewhere in the set.

    Degenerate free atoms (:func:`is_degenerate_atom`) are evaluated on an
    orientation-locked reference density at
    :func:`atom_orientation_lock_strength` whenever the run's own
    ``inputs.orientation_lock_strength`` is zero, and the payload names them
    under ``atom_orientation_lock``; a run that carries its own lock applies
    it to every system unchanged. Without the lock the SCAN E_xc of the free O
    atom moves of order 0.1 mHa between independent SCFs at def2-svp /
    grid 3 (0.26 mHa over one triple of runs, 0.084 mHa over another) -- a
    meaningful fraction of ``tol_atom`` decided by BLAS scheduling -- and
    with it two independent runs agreed to 3.4e-11 mHa at that identity (at
    def2-svp / grid 1 the locked PBE O atom still spread 2.5e-3 mHa between
    second-order SCF landings).

    The
    tolerances actually applied, the override reason (copied verbatim) and
    the enforcement flag are recorded beside the verdict, as are the run
    identity and the SHA-256 digests of the two checkpoint files the verdict
    refers to.
    """

    t0 = time.time()
    fid_cfg = cfg.fidelity
    # The one flag that decides whether a recorded FAIL releases an on-node
    # gate. bool("false") is True and bool(None) is False, so a coerced value
    # would record an enforcement state no configuration asked for; only a
    # real boolean is recorded (grid_config._build_fidelity imposes the same
    # rule on the YAML; this re-imposes it on any hand-built cfg), refused
    # before any system is evaluated so no certificate is written.
    enforce = getattr(fid_cfg, "enforce", True)
    if not isinstance(enforce, bool):
        raise ValueError(
            f"cfg.fidelity.enforce must be a real boolean, got "
            f"{type(enforce).__name__} ({enforce!r})")
    parent = resolve_parent(arch_name)
    arch, model = build_certified_model(cfg, run_dir, arch_name)
    pretrain_dir = pretrain_checkpoint_dir(run_dir, arch_name)
    checkpoint = {
        "dir": pretrain_dir,
        "xnet_sha256": _sha256_file(os.path.join(pretrain_dir, "xnet.eqx")),
        "cnet_sha256": _sha256_file(os.path.join(pretrain_dir, "cnet.eqx")),
    }
    descriptors = arch.materialize_descriptors()
    systems = tuple(oracle_set) if oracle_set is not None \
        else build_oracle_set(cfg, arch_name)
    run = evaluate if evaluate is not None else evaluate_system
    say = log if log is not None else (lambda message: None)

    # The precompute memoizes on (spec, keys, descriptors, ...); dozens of
    # production-basis grids would exhaust a node's memory long before the
    # sweep finished, and each system is visited exactly once. The caller's
    # setting is restored afterwards, whichever it was.
    import xcquinox.alec.data as data_mod
    cache_was_enabled = bool(getattr(data_mod, "_PRECOMPUTE_CACHE_ENABLED",
                                     True))
    data_mod.set_precompute_cache_enabled(False)
    data_mod.clear_precompute_cache()

    inputs = cfg.inputs
    # The run's own orientation lock governs every system when it is on. When
    # it is off, a degenerate free atom (open p shell) is still evaluated on
    # an orientation-LOCKED reference density, so the certificate's atomic
    # E_xc does not depend on which orientation of the hole the SCF happened
    # to converge to; molecules and spherical atoms keep the run's setting.
    # Consequence for a run that waives the generator's
    # irreproducible-degenerate refusal at lock 0.0
    # (inputs.allow_irreproducible_degenerate): its pretraining rows sit at
    # 0.0 while this fallback measures the degenerate atoms at the calibrated
    # lock, so the certificate bounds those atoms on a density the network was
    # not pretrained against.
    run_lock = float(getattr(inputs, "orientation_lock_strength", 0.0))
    atom_lock = atom_orientation_lock_strength() if run_lock == 0.0 else 0.0
    locked_atoms = []
    per_system = []
    unconverged = []
    non_finite = []
    n_systems = len(systems)
    try:
        for index, mol_spec in enumerate(systems, start=1):
            t_sys = time.time()
            try:
                # Inside the try: the degeneracy rule can itself refuse (an
                # element beyond argon), and that is a per-system failure
                # like any other, not an abort that leaves no certificate.
                lock = run_lock
                if run_lock == 0.0 and is_degenerate_atom(mol_spec):
                    lock = atom_lock
                    locked_atoms.append(mol_spec.name)
                rec = run(
                    model, descriptors, mol_spec, parent=parent,
                    auxbasis=getattr(inputs, "auxbasis", None),
                    orientation_lock_strength=lock)
            except (ReferenceNotConverged,
                    data_mod.ReferenceSCFNotConverged) as exc:
                # One branch for both spellings of the same physics: the
                # producer refused to build the record
                # (data.ReferenceSCFNotConverged), or a record arrived
                # already stamped unconverged (ReferenceNotConverged). Both
                # carry the cycle count.
                per_system.append({
                    "name": mol_spec.name,
                    "error": f"{type(exc).__name__}: {exc}",
                    "reference_scf_converged": False,
                    "reference_scf_cycles": getattr(exc, "cycles", None)})
                unconverged.append(mol_spec.name)
                say(f"[{index}/{n_systems}] {mol_spec.name}: REFUSED {exc}")
                continue
            except Exception as exc:  # noqa: BLE001 -- recorded, not raised
                per_system.append({
                    "name": mol_spec.name,
                    "error": f"{type(exc).__name__}: {exc}"})
                say(f"[{index}/{n_systems}] {mol_spec.name}: FAILED "
                    f"{type(exc).__name__}: {exc}")
                continue
            bad = _null_non_finite(rec)
            per_system.append(rec)
            if bad:
                non_finite.append((mol_spec.name, bad))
                say(f"[{index}/{n_systems}] {mol_spec.name}: NON-FINITE "
                    f"measurement ({', '.join(bad)})")
                continue
            elapsed = time.time() - t0
            say(f"[{index}/{n_systems}] {mol_spec.name}: "
                f"dE_xc={rec['dE_xc_mHa']:+.4f} mHa "
                f"(parent routes {rec['parent_grid_diff_Ha']:.1e}/"
                f"{rec['parent_record_diff_Ha']:.1e} Ha; "
                f"{time.time() - t_sys:.1f} s; elapsed {_fmt_secs(elapsed)}; "
                f"ETA {_fmt_secs(elapsed / index * (n_systems - index))})")
    finally:
        data_mod.set_precompute_cache_enabled(cache_was_enabled)
        data_mod.clear_precompute_cache()

    ok = {r["name"]: r for r in per_system if "error" not in r}

    per_atomization = []
    for mol_spec in systems:
        if is_atom_system(mol_spec) or mol_spec.name not in ok:
            continue
        atom_terms = []
        missing = None
        unusable = (mol_spec.name
                    if ok[mol_spec.name]["dE_xc_mHa"] is None else None)
        for symbol, count in mol_spec.atom_composition:
            atom_name = atom_system_name(symbol, 0)
            if atom_name not in ok:
                missing = atom_name
                break
            if ok[atom_name]["dE_xc_mHa"] is None:
                unusable = unusable or atom_name
                continue
            atom_terms.append(int(count) * ok[atom_name]["dE_xc_mHa"])
        if missing is not None:
            per_atomization.append({
                "name": mol_spec.name, "dAE_kcalmol": None,
                "error": f"free atom {missing} is missing from the oracle set"})
            continue
        if unusable is not None:
            # A nulled (non-finite) dE_xc cannot enter the fold; the
            # non-finite reason already names the offending system.
            per_atomization.append({
                "name": mol_spec.name, "dAE_kcalmol": None,
                "error": f"dE_xc of {unusable} is not finite"})
            continue
        d_ae_mha = ok[mol_spec.name]["dE_xc_mHa"] - sum(atom_terms)
        d_ae_kcal = d_ae_mha / HA_TO_MHA * HA_TO_KCAL
        if not math.isfinite(d_ae_kcal):
            non_finite.append((mol_spec.name, ["dAE_kcalmol"]))
            per_atomization.append({
                "name": mol_spec.name, "dAE_kcalmol": None,
                "error": "the atomization offset is not finite"})
            continue
        per_atomization.append({
            "name": mol_spec.name,
            "dAE_kcalmol": d_ae_kcal})

    # A nulled non-finite value never enters a max() or a comparison.
    atom_dev = [abs(r["dE_xc_mHa"]) for r in ok.values()
                if r["is_atom"] and r["dE_xc_mHa"] is not None]
    ae_dev = [abs(r["dAE_kcalmol"]) for r in per_atomization
              if r.get("dAE_kcalmol") is not None]
    grid_dev = [abs(r["parent_grid_diff_Ha"]) for r in ok.values()
                if r["parent_grid_diff_Ha"] is not None]
    record_dev = [abs(r["parent_record_diff_Ha"]) for r in ok.values()
                  if r["parent_record_diff_Ha"] is not None]
    n_failed = sum(1 for r in per_system if "error" in r)

    tol_atom = float(fid_cfg.tol_atom)
    tol_ae = float(fid_cfg.tol_AE)
    max_atom = max(atom_dev) if atom_dev else None
    max_ae = max(ae_dev) if ae_dev else None
    max_grid = max(grid_dev) if grid_dev else None
    max_record = max(record_dev) if record_dev else None

    reasons = []
    if non_finite:
        reasons.append(
            f"a non-finite measurement was recorded for {len(non_finite)} "
            "system(s): "
            + "; ".join(f"{name} ({', '.join(keys)})"
                        for name, keys in non_finite)
            + " -- NaN satisfies no tolerance, so the verdict fails before "
            "any comparison")
    if unconverged:
        reasons.append(
            f"the reference SCF did not converge for {len(unconverged)} "
            f"system(s): " + ", ".join(unconverged)
            + "; the certificate never measures a network on an unconverged "
            "density")
    other_failed = [r["name"] for r in per_system
                    if "error" in r and r["name"] not in unconverged]
    if other_failed:
        reasons.append(
            f"{len(other_failed)} system(s) could not be evaluated: "
            + ", ".join(other_failed))
    if not atom_dev:
        reasons.append("no free atom was evaluated, so tol_atom is untested")
    elif max_atom > tol_atom:
        # repr: the shortest digit string that round-trips, so a one-ulp
        # excess never prints as equal to the tolerance it exceeds.
        reasons.append(
            f"max |dE_xc| over free atoms {max_atom!r} mHa exceeds "
            f"tol_atom {tol_atom!r} mHa")
    if not ae_dev:
        reasons.append(
            "no atomization offset could be formed, so tol_AE is untested")
    elif max_ae > tol_ae:
        reasons.append(
            f"max |dAE| {max_ae!r} kcal/mol exceeds tol_AE "
            f"{tol_ae!r} kcal/mol")
    grid_offenders = [
        (r["name"], r["parent_grid_diff_Ha"]) for r in ok.values()
        if r["parent_grid_diff_Ha"] is not None
        and abs(r["parent_grid_diff_Ha"]) > PARENT_GRID_TOL_HA]
    if grid_offenders:
        reasons.append(
            "the point-wise and fresh-grid parent routes disagree above the "
            f"{PARENT_GRID_TOL_HA:.0e} Ha bound on: "
            + ", ".join(f"{name} ({value!r} Ha)"
                        for name, value in grid_offenders))
    record_offenders = [
        (r["name"], r["parent_record_diff_Ha"]) for r in ok.values()
        if r["parent_record_diff_Ha"] is not None
        and abs(r["parent_record_diff_Ha"]) > PARENT_GRID_TOL_HA]
    if record_offenders:
        reasons.append(
            "the point-wise parent energy and the reference SCF's own "
            f"accumulated E_xc disagree above the {PARENT_GRID_TOL_HA:.0e} "
            "Ha bound on: "
            + ", ".join(f"{name} ({value!r} Ha)"
                        for name, value in record_offenders))

    payload = {
        "verdict": VERDICT_FAIL if reasons else VERDICT_PASS,
        "arch": arch_name,
        "parent": parent,
        # The model class the certified networks were built as: whether
        # they are anchored to the parent, which coordinates their MLPs
        # read, and whether their inputs are log-compressed (each a static
        # property of the networks, absent from the checkpoint's leaf stream,
        # so recorded here beside the arch). The third is compared where a
        # certificate states it (:func:`model_class_mismatches`); files
        # written before it carry the first two alone.
        "parent_anchor": bool(getattr(arch, "parent_anchor", False)),
        "descriptor_coordinates": str(
            getattr(arch, "descriptor_coordinates", "legacy")),
        "descriptor_log_transform": bool(
            getattr(arch, "descriptor_log_transform", False)),
        "xcquinox_version": running_xcquinox_version(),
        "identity": run_identity(cfg),
        "checkpoint": checkpoint,
        "atom_orientation_lock": {
            "run_orientation_lock_strength": run_lock,
            "strength": atom_lock,
            "applied_to": locked_atoms,
            "note": (
                "degenerate free atoms (open p shell) are evaluated on an "
                "orientation-locked reference density when the run's own "
                "lock is off, so their E_xc does not depend on the "
                "orientation the SCF converged to; the lock is "
                "orientation_lock.DEFAULT_STRENGTH and the run identity is "
                "otherwise unchanged"),
        },
        "tolerances": {"tol_AE": tol_ae, "tol_atom": tol_atom,
                       "override_reason": fid_cfg.override_reason},
        # Whether this run's ON-NODE gates act on the verdict. False belongs
        # to the workflow-verification matrix only; the record layers ignore
        # it and require PASS regardless. Validated a real boolean above.
        "enforced": enforce,
        "per_system": per_system,
        "per_atomization": per_atomization,
        "summary": {
            "max_atom_mHa": max_atom,
            "max_dAE_kcalmol": max_ae,
            "n_systems": len(per_system),
            "n_atoms": len(atom_dev),
            "n_atomizations": len(ae_dev),
            "n_failed_systems": n_failed,
            "n_reference_unconverged": len(unconverged),
            "n_non_finite_systems": len(non_finite),
            "max_parent_grid_diff_Ha": max_grid,
            "max_parent_record_diff_Ha": max_record,
            "failure_reasons": reasons,
        },
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "duration_s": time.time() - t0,
    }
    _write_certificate_payload(payload, certificate_path(run_dir, arch_name))
    return payload


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def _route_jax_env():
    """Pin JAX to fp64 for the certificate process.

    The ``xcquinox`` package imports jax while this module is being imported,
    so the ``JAX_ENABLE_X64`` environment variable alone cannot switch the
    already-initialised library; the runtime config update is the effective
    switch and runs before any network or grid array exists. The variable is
    set as well so any child process inherits it. ``JAX_PLATFORMS`` is left
    untouched so the certificate runs on whichever device the sbatch script
    requested (mirrors ``cluster._pretrain``).
    """
    os.environ["JAX_ENABLE_X64"] = "1"
    import jax
    jax.config.update("jax_enable_x64", True)


def _log(arch, message):
    """One tagged harness log line to stdout -- the SLURM log."""
    sys.stdout.write(f"[harness fidelity arch={arch}] {message}\n")
    sys.stdout.flush()


def main(argv=None) -> int:
    """Certificate entrypoint. Returns 0 on PASS, non-zero otherwise."""
    _route_jax_env()
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_dir", help="The materialized run directory.")
    parser.add_argument("arch_idx", type=int,
                        help="Index into the sorted distinct-architecture "
                             "list (the same selector the pretrain array "
                             "uses).")
    args = parser.parse_args(argv)
    run_dir = os.path.abspath(args.run_dir)

    cfg_path = os.path.join(run_dir, "resolved_config.yaml")
    if not os.path.isfile(cfg_path):
        json_path = os.path.join(run_dir, "resolved_config.json")
        if not os.path.isfile(json_path):
            sys.stdout.write(
                f"[harness fidelity] ERROR: no resolved config at "
                f"{cfg_path}\n")
            sys.stdout.flush()
            return 1
        cfg_path = json_path
    try:
        cfg = load_grid_config(cfg_path)
    except (ValueError, ImportError, OSError) as exc:
        sys.stdout.write(
            f"[harness fidelity] ERROR: failed to load resolved config: "
            f"{exc}\n")
        sys.stdout.flush()
        return 1

    archs = _distinct_archs(cfg)
    if not (0 <= args.arch_idx < len(archs)):
        sys.stdout.write(
            f"[harness fidelity] ERROR: arch_idx {args.arch_idx} is out of "
            f"range; the config has {len(archs)} distinct architecture(s) "
            f"(valid indices 0..{len(archs) - 1}): {archs}\n")
        sys.stdout.flush()
        return 1
    arch_name = archs[args.arch_idx]

    _log(arch_name, f"certifying against parent "
                    f"{resolve_parent(arch_name).upper()} at "
                    f"{run_identity(cfg)}")
    try:
        payload = fidelity_certificate(
            cfg, run_dir, arch_name,
            log=lambda message: _log(arch_name, message))
    except Exception as exc:  # noqa: BLE001 -- the node log carries it
        _log(arch_name, f"ERROR: the certificate could not be computed: "
                        f"{type(exc).__name__}: {exc}")
        sys.stdout.write(traceback.format_exc())
        sys.stdout.flush()
        return 1
    summary = payload["summary"]
    _log(arch_name,
         f"verdict={payload['verdict']} "
         f"max_atom={summary['max_atom_mHa']} mHa "
         f"max_dAE={summary['max_dAE_kcalmol']} kcal/mol over "
         f"{summary['n_systems']} system(s) "
         f"({summary['n_atoms']} atom(s), {summary['n_atomizations']} "
         f"atomization(s))")
    if payload["verdict"] != VERDICT_PASS:
        for reason in summary["failure_reasons"]:
            _log(arch_name, f"FAIL: {reason}")
        if not payload["enforced"]:
            _log(arch_name,
                 "enforcement is OFF for this run (fidelity.enforce=false, "
                 f"override_reason: "
                 f"{payload['tolerances']['override_reason']!r}); the verdict "
                 "is on record and the stage continues. This run cannot enter "
                 "validate_run, merge_v4_arms or the figure suite.")
            return 0
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess
    # The stage's verdict is the status this process hands SLURM, and
    # JAX's atexit teardown can abort the interpreter AFTER main() has
    # returned it (cluster job 2134455: the pretrain worker logged
    # "pretrain SUCCEEDED" and then died in glibc's "corrupted size vs.
    # prev_size", rc -6, so the stage read as FAILED and the dependent
    # array never ran). run_and_exit flushes and leaves through os._exit,
    # so the status is the verdict. See xcquinox/alec/cluster/_exit.py.
    # Imported HERE rather than in the module body: several of these
    # modules pin what their import pulls in (``fidelity`` is held to a
    # whitelist of cheap readers so the on-node gates can read a
    # certificate without the training stack), and the helper is needed
    # only when the module is RUN.
    from xcquinox.alec.cluster._exit import run_and_exit
    run_and_exit(main)
