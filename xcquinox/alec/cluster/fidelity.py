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
which is what each rung was pretrained against.

The verdict, every number, the run identity and the installed code version go
to ``<run_dir>/pretrain/<arch>/fidelity_certificate.json``. The pretrain
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
``domain``, ``materialize``, each stdlib-only in its own body). Every jax /
equinox / pyscf / ``xcquinox.alec.data`` import happens INSIDE a function, so
the login node CLI, the run validator, the train task's parent process and the
analysis layer read a certificate without this file pulling a model or an SCF
stack.
"""
from __future__ import annotations

import json
import os

from xcquinox.alec.cluster.domain import KCAL_PER_HA
from xcquinox.alec.cluster.grid_config import (
    _canon_axis, pretrain_checkpoint_dir,
)


CERTIFICATE_FILENAME = "fidelity_certificate.json"
VERDICT_PASS = "PASS"
VERDICT_FAIL = "FAIL"

# One Hartree in kcal/mol, taken from the harness domain table rather than
# restated here (domain.KCAL_PER_HA, CODATA-2018, cited at its definition).
# The certificate's atomization offsets and the campaign's benchmark errors
# are read against the same kcal/mol tolerances, so a locally truncated copy
# would put the two on slightly different scales.
HA_TO_KCAL = KCAL_PER_HA
HA_TO_MHA = 1000.0

# The parent XC energy is computed two independent ways per system: point-wise
# on the stored precompute grid (the grid the network is integrated on, so the
# comparison is grid-exact) and through PySCF's own nr_rks / nr_uks on a
# freshly built grid of the same level. At sto-3g grid level 1 the two routes
# agree to 2.6e-11 Ha on OH and 6.2e-11 Ha on H2O, for PBE and SCAN alike, and
# at the production identity to 2.0e-10 Ha (recorded in
# notebooks/analysis/NOTES_v5_mgga_vs_scan.md, Section 5). The bound below is
# more than three orders of magnitude above that spread and three below
# tol_atom = 1.0 mHa, so it fires when the stored grid and the molecule no
# longer describe the same system and never on integration noise.
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
    """
    path = certificate_path_in(pretrain_dir)
    if not os.path.isfile(path):
        return "MISSING", (
            f"no {CERTIFICATE_FILENAME} in {pretrain_dir}: the architecture "
            "was never checked against its parent functional")
    try:
        with open(path) as f:
            payload = json.load(f)
    except (OSError, ValueError) as exc:
        return "UNREADABLE", (
            f"{path} is not readable JSON ({type(exc).__name__}: {exc})")
    if not isinstance(payload, dict):
        return "UNREADABLE", f"{path} is not a JSON object"
    verdict = payload.get("verdict")
    if verdict == VERDICT_PASS:
        return VERDICT_PASS, "fidelity certificate PASS"
    if verdict != VERDICT_FAIL:
        return "UNREADABLE", (
            f"{path} records verdict {verdict!r}, which is neither "
            f"{VERDICT_PASS!r} nor {VERDICT_FAIL!r}: the file states no "
            "outcome that can be acted on")
    summary = payload.get("summary") or {}
    return VERDICT_FAIL, (
        f"fidelity certificate verdict {verdict!r} at {path} "
        f"(max_atom_mHa={summary.get('max_atom_mHa')}, "
        f"max_dAE_kcalmol={summary.get('max_dAE_kcalmol')}, "
        f"reasons={summary.get('failure_reasons')})")


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
    FAILs, records ``enforced: false`` AND names a non-empty
    ``tolerances.override_reason`` -- the workflow-verification matrix, whose
    short pretraining runs cannot meet the tolerance yet must exercise the
    train and eval wiring with the real verdict written down. A MISSING or
    UNREADABLE certificate is never allowed: there is then no record of what
    was measured or of any waiver.

    The reason is required here and not only in the configuration.
    ``validate_grid_semantics`` refuses ``fidelity.enforce: false`` without a
    non-empty ``fidelity.override_reason``, and this gate re-imposes the same
    invariant on the certificate it reads, so a hand-edited certificate or
    ``resolved_config.yaml`` on a compute node cannot release a stage with no
    reason on the record.

    The record layers do NOT call this. ``validate_run``, ``merge_v4_arms``
    and the figure loaders require PASS through :func:`certificate_status`, so
    a non-enforcing run can never enter the record.
    """
    pretrain_dir = pretrain_checkpoint_dir(run_dir, arch)
    status, reason = certificate_status_in(pretrain_dir)
    if status == VERDICT_PASS:
        return True, reason
    if status != VERDICT_FAIL:
        return False, reason
    if certificate_enforced_in(pretrain_dir):
        return False, reason
    payload = read_certificate(pretrain_dir) or {}
    tolerances = payload.get("tolerances")
    if not isinstance(tolerances, dict):
        tolerances = {}
    recorded = tolerances.get("override_reason")
    override = "" if recorded is None else str(recorded).strip()
    if not override:
        return False, (
            f"{reason}; the certificate records enforced=false but its "
            "tolerances.override_reason is empty, so the waiver states no "
            "reason. Disabling the on-node gates requires a non-empty "
            "fidelity.override_reason, which validate_grid_semantics imposes "
            "on the configuration and this gate re-checks on the certificate.")
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
