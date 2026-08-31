"""xcquinox.alec.data: MoleculeData TypedDict and precompute.

Implements THE SPEC §6.1 (MoleculeData), §6.2 (precompute_fixed_density_data).
"""
import os
from typing import TypedDict

import numpy as np
import jax.numpy as jnp

from xcquinox.alec.config import MoleculeSpec
from xcquinox.alec.descriptors import Descriptor
from xcquinox.alec.orientation_lock import orientation_lock_bias
from xcquinox.alec.pyscf_determinism import pin_reference_scf


# Keys allowed in MoleculeSpec.external_data_path .npz files. Kept as a
# module-level constant so tests and documentation can share it.
# OEP provenance keys (oep_*) are written by save_vxc_ref so downstream
# loaders can validate baseline / aux_basis / regularization / convergence
# consistency against what produced the V_xc.
_ALLOWED_EXTERNAL_KEYS = frozenset({
    "dm_target",
    "rho_ref_grid",
    "ref_density_method",
    "E_ref_literature",
    "vxc_ref",
    # OEP provenance (informational only, not validated against the
    # consumer's runtime config in this loader, but available for callers
    # that want to assert agreement).
    "oep_baseline_xc",
    "oep_aux_basis",
    "oep_regularization",
    "oep_density_error",
    "oep_converged",
    "oep_lbfgs_status",
    "oep_n_electrons",
    # grid_level the reference was generated on. When present,
    # _load_external_data asserts it equals the consumer's resolved
    # grid_level so a reference built on a different grid cannot load
    # silently against a mismatched density/V_xc grid.
    "grid_level_used",
    # basis the reference was generated for; lets the OEP cache-hit reject a
    # stale .npz built for a different basis in the same cache_dir.
    "basis_used",
    # Benchmark density-only refs (xcquinox.alec.benchmark_refs) also carry
    # the generator-side PBE density + grid weights so the model-free
    # PBE-vs-CCSD baseline is pure npz arithmetic. Shape-validated here but
    # NOT returned into MoleculeData (the precompute computes its own PBE
    # quantities on the identical grid).
    "rho_pbe_grid",
    "grid_weights",
    # Orientation-lock strength the reference density was generated with
    # (0.0 = unlocked). Informational: the consumer applies its own lock from
    # SolverConfig; the demo threads one shared constant to ref-gen + eval so
    # they match. Tolerated here so the loader does not reject a locked ref.
    "orientation_lock_strength",
    # Whether the reference was generated with density fitting. Identity
    # guard consumed by benchmark_refs._benchmark_npz_is_complete (a stamped
    # DF reference is never silently reused by a non-DF run or vice versa);
    # informational in this loader.
    "density_fit_used",
})


def _load_external_data(
    path: str,
    *,
    dm_pbe_shape: tuple[int, ...],
    rho_pbe_shape: tuple[int, ...],
    vxc_pbe_shape: tuple[int, ...],
    mol_name: str,
    grid_level: int | None = None,
    orientation_lock_strength: float | None = None,
) -> tuple[jnp.ndarray | None, jnp.ndarray | None, str | None, float | None, jnp.ndarray | None]:
    """Load and validate a MoleculeSpec.external_data_path .npz.

    The .npz may contain any subset of ``dm_target``, ``rho_ref_grid``,
    ``ref_density_method``, ``E_ref_literature``, ``vxc_ref``; unknown
    keys trigger ``ValueError``. Shape validation matches freshly computed
    PBE quantities so callers cannot silently mismatch densities/DMs/V_xc
    against the PBE grid or basis.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"MoleculeSpec.external_data_path does not exist for "
            f"{mol_name!r}: {path}"
        )

    with np.load(path) as npz:
        present = set(npz.files)
        unknown = present - _ALLOWED_EXTERNAL_KEYS
        if unknown:
            raise ValueError(
                f"external_data .npz for {mol_name!r} contains unknown "
                f"keys {sorted(unknown)}; allowed keys: "
                f"{sorted(_ALLOWED_EXTERNAL_KEYS)}"
            )

        # If the reference records the grid_level it was generated on,
        # assert it equals the consumer's resolved grid_level. This is the
        # primary consistency gate; the per-array shape checks below remain
        # as a fallback for references that do not carry this key.
        if "grid_level_used" in present and grid_level is not None:
            grid_level_used = int(np.asarray(npz["grid_level_used"]).item())
            if grid_level_used != int(grid_level):
                raise ValueError(
                    f"external reference for {mol_name!r} was generated at "
                    f"grid_level={grid_level_used} but the consumer resolves "
                    f"grid_level={int(grid_level)}; the reference density / "
                    f"V_xc grid does not match. Regenerate the reference at "
                    f"grid_level={int(grid_level)} or pin the MoleculeSpec "
                    f"grid_level to {grid_level_used}."
                )

        # Orientation-lock guard: if the reference RECORDS the lock it was
        # generated with, assert it matches the consumer's configured lock. A
        # mismatch means the training density (esp. the degenerate radicals
        # OH/CH/NO) was built for a different orientation than the functional's
        # locked SCF -- the silent cache-key gap that let unlocked refs be reused
        # for a locked run. Fires only when BOTH the ref carries the key and the
        # caller passes a lock (None consumer -> skip); a legacy ref without the
        # key is caught upstream by run_oep_cascade's skip-if-cached predicate,
        # which regenerates it with the key.
        if ("orientation_lock_strength" in present
                and orientation_lock_strength is not None):
            ref_ol = float(np.asarray(npz["orientation_lock_strength"]).item())
            if f"{ref_ol:g}" != f"{orientation_lock_strength:g}":
                raise ValueError(
                    f"external reference for {mol_name!r} was generated with "
                    f"orientation_lock_strength={ref_ol:g} but the consumer "
                    f"resolves {orientation_lock_strength:g}; the reference "
                    f"density does not match the functional's locked SCF. "
                    f"Regenerate the reference at the configured lock."
                )

        dm_target = None
        if "dm_target" in present:
            dm_arr = np.asarray(npz["dm_target"])
            if tuple(dm_arr.shape) != tuple(dm_pbe_shape):
                raise ValueError(
                    f"external dm_target shape {tuple(dm_arr.shape)} does "
                    f"not match dm_pbe shape {tuple(dm_pbe_shape)} for "
                    f"{mol_name!r}"
                )
            dm_target = jnp.array(dm_arr)

        rho_ref_grid = None
        ref_density_method = None
        if "rho_ref_grid" in present:
            rho_arr = np.asarray(npz["rho_ref_grid"])
            if tuple(rho_arr.shape) != tuple(rho_pbe_shape):
                raise ValueError(
                    f"external rho_ref_grid shape {tuple(rho_arr.shape)} "
                    f"does not match rho_grid shape {tuple(rho_pbe_shape)} "
                    f"for {mol_name!r}"
                )
            rho_ref_grid = jnp.array(rho_arr)
        if "ref_density_method" in present:
            method_arr = np.asarray(npz["ref_density_method"])
            ref_density_method = str(method_arr.item())
        # informational benchmark-refs arrays: validate shape, do not return
        for grid_key in ("rho_pbe_grid", "grid_weights"):
            if grid_key in present:
                arr = np.asarray(npz[grid_key])
                if tuple(arr.shape) != tuple(rho_pbe_shape):
                    raise ValueError(
                        f"external {grid_key} shape {tuple(arr.shape)} does "
                        f"not match rho_grid shape {tuple(rho_pbe_shape)} "
                        f"for {mol_name!r}"
                    )

        E_ref_literature = None
        if "E_ref_literature" in present:
            val = np.asarray(npz["E_ref_literature"])
            if val.ndim == 0 or (val.ndim == 1 and val.size == 1):
                E_ref_literature = float(val.reshape(()).item())
            else:
                raise ValueError(
                    f"external E_ref_literature for {mol_name!r} must be "
                    f"scalar, got shape {tuple(val.shape)}"
                )

        vxc_ref = None
        if "vxc_ref" in present:
            vxc_arr = np.asarray(npz["vxc_ref"])
            if tuple(vxc_arr.shape) != tuple(vxc_pbe_shape):
                raise ValueError(
                    f"external vxc_ref shape {tuple(vxc_arr.shape)} does not "
                    f"match vxc_pbe shape {tuple(vxc_pbe_shape)} for "
                    f"{mol_name!r}"
                )
            vxc_ref = jnp.array(vxc_arr)

    return dm_target, rho_ref_grid, ref_density_method, E_ref_literature, vxc_ref


class MoleculeData(TypedDict, total=True):
    """Pre-computed training/test data for one molecule.
    Every key is always present; unused keys are None.

    REFERENCE-SCF FIELDS. ``dm_pbe``, ``rho_grid``, ``sigma_grid``,
    ``nabla_rho_grid``, ``vxc_pbe``, ``E_pbe``, ``E_xc_pbe``, ``E_non_xc`` and
    every descriptor block hold quantities of the REFERENCE self-consistent
    field, whose functional is recorded in ``reference_xc``. The names carry
    ``pbe`` because PBE was the only reference the precompute could produce
    when they were introduced; they are kept so no consumer has to be rewritten
    for a naming change that carries no physics. A consumer that depends on the
    reference being a particular functional asserts ``reference_xc`` rather than
    reading the name.
    """
    name: str
    is_unrestricted: bool
    nocc: int | None
    nocc_a: int | None
    nocc_b: int | None
    dm_pbe: jnp.ndarray
    # The SCF seed D0 (per-rung seeding). For seed_source="pbe" this is the
    # SAME array object as dm_pbe (alias, byte-identical protocol); "scan"
    # loads a converged SCAN dm from the seed cache; "minao" is the
    # functional-free superposition guess. The solver consumes this key
    # unconditionally; only the supply here dispatches on seed_source.
    dm_seed: jnp.ndarray
    s_matrix: jnp.ndarray
    h_core: jnp.ndarray
    j_matrix: jnp.ndarray
    vxc_pbe: jnp.ndarray
    e_nuc: float
    E_pbe: float
    E_xc_pbe: float
    E_non_xc: float
    E_ref_literature: float | None
    dm_target: jnp.ndarray | None
    rho_ref_grid: jnp.ndarray | None
    ref_density_method: str | None
    vxc_ref: jnp.ndarray | None
    rho_grid: jnp.ndarray
    sigma_grid: jnp.ndarray
    nabla_rho_grid: jnp.ndarray
    grid_weights: jnp.ndarray
    ao_grid: jnp.ndarray
    ao_grid_deriv: jnp.ndarray
    cusp_features: jnp.ndarray | None
    dm_features: jnp.ndarray | None
    # Rung-3.5 localized DM descriptor (gated on a DMRung35Descriptor being
    # present). rung35_proj_ao is the constant projected-AO matrix A (N, nao);
    # rung35_features is the one-shot per-spin occupancy A^T P_pbe A (N, 2).
    rung35_proj_ao: jnp.ndarray | None
    rung35_features: jnp.ndarray | None
    # Multi-width rung-3.5 (DMRung35MultishellDescriptor). rung35ms_proj_ao is
    # the constant projected-AO STACK (n_alpha, N, nao); rung35ms_features is
    # the one-shot per-spin, per-width occupancy (N, 2 * n_alpha).
    rung35ms_proj_ao: jnp.ndarray | None
    rung35ms_features: jnp.ndarray | None
    # Meta-GGA iso-orbital alpha (metagga.py). One-shot alpha from the PBE DM (N, 1);
    # the FULL/REASSEMBLE SCF recomputes it self-consistently each cycle from the
    # live DM + the stored AO gradients (ao_grid_deriv).
    metagga_features: jnp.ndarray | None
    # Per-spin-channel descriptor blocks: this descriptor's features for the
    # symmetric doubled density diag(P_sigma, P_sigma), the spin-unpolarized
    # system the exact exchange spin-scaling relation refers to (Oliver and
    # Perdew, Phys. Rev. A 20, 397 (1979)). Layout matches the total-density
    # twin above column for column. None for a closed-shell molecule, whose
    # rho_a = rho_b makes the per-channel block identical to the total one, and
    # None for a descriptor the architecture does not carry.
    dm_features_a: jnp.ndarray | None
    dm_features_b: jnp.ndarray | None
    rung35_features_a: jnp.ndarray | None
    rung35_features_b: jnp.ndarray | None
    rung35ms_features_a: jnp.ndarray | None
    rung35ms_features_b: jnp.ndarray | None
    metagga_features_a: jnp.ndarray | None
    metagga_features_b: jnp.ndarray | None
    # Per-spin positive kinetic-energy density tau_sigma on the grid, (n_grid,).
    # The doubled system's tau is 2 tau_sigma. Stored alongside the meta-GGA
    # blocks so the open-shell exchange ingredients are inspectable without
    # recontracting the density matrix.
    tau_spin_a: jnp.ndarray | None
    tau_spin_b: jnp.ndarray | None
    # The functional of the reference SCF that produced every quantity above:
    # the density matrix, the grid quantities, the total and per-spin
    # descriptor blocks, and E_pbe / E_xc_pbe / E_non_xc. "pbe" for the whole
    # training and evaluation pipeline; the pretraining-fidelity certificate
    # requests "scan" for a meta-GGA architecture, whose parent functional is
    # SCAN, so the network is measured on the density it must reproduce.
    # Always the canonical spelling (canonical_reference_xc): stripped,
    # lower-cased, and "pbe" / "scan" for any spelling libxc parses to those
    # functionals. mol_metadata repeats it beside reference_scf_converged,
    # reference_scf_cycles and reference_scf_solver, the reference SCF's
    # convergence stamp.
    reference_xc: str
    eri: jnp.ndarray | None
    cderi: jnp.ndarray | None
    atom_composition: tuple[tuple[str, int], ...]
    mol_metadata: dict
    # Cached pyscfad.gto.Mole built once at precompute time so that hot-path
    # training (pyscfad backend, filter_jit'd) does not call Mole.build()
    # inside the traced region, Mole.build() invokes numpy.__array__ and
    # raises TracerArrayConversionError under jit. Always present; may be
    # None if pyscfad is unavailable or Mole construction failed.
    _pyscfad_mol: object | None


_PRECOMPUTE_CACHE: dict = {}
_PRECOMPUTE_CACHE_ENABLED: bool = True


class ReferenceSCFNotConverged(RuntimeError):
    """The reference SCF of a fixed-density record did not converge.

    Raised by :func:`precompute_fixed_density_data` instead of returning a
    record: every field of the record is a property of the reference
    functional's SELF-CONSISTENT density, and an SCF stopped short of it is
    not that functional's density (measured on H2O / sto-3g / grid 1 with
    SCAN stopped after one cycle: +7.2e-2 Ha in the total energy and 0.315 in
    the density matrix against the converged values, with ``mf.converged``
    False). ``cycles`` carries the number of SCF cycles pyscf ran.
    """

    def __init__(self, message: str, *, cycles: int | None = None):
        super().__init__(message)
        self.cycles = cycles


# The spellings under which the program's two reference functionals are
# recorded, memoized and compared (cluster.fidelity and
# pretrain_data_gen.dfs_level_for_reference_xc test ``reference_xc`` with
# ``==``). A request is canonicalized to one of these exactly when libxc
# parses it to the same functional -- the same (hybrid, alpha, omega) triple
# and the same (functional id, factor) list. Measured with pyscf 2.11:
# "PBE", "pbe,pbe" and "gga_x_pbe,gga_c_pbe" parse identically to "pbe"
# (((0, 0, 0), ((101, 1), (130, 1)))); "SCAN", " scan" and "scan,scan"
# identically to "scan" (((0, 0, 0), ((263, 1), (267, 1)))); "blyp" parses
# to ((106, 1), (131, 1)) and is not PBE.
_CANONICAL_REFERENCE_XC = ("pbe", "scan")


def canonical_reference_xc(reference_xc) -> str:
    """The one spelling a reference functional is keyed, run and recorded under.

    ``reference_xc`` is stripped and lower-cased (libxc's parser is case- and
    whitespace-insensitive, so these spellings are one functional and must
    not own separate SCFs or memo entries), validated against libxc -- an
    unknown name raises ``ValueError`` naming the argument here, rather than
    libxc's ``KeyError`` out of a later coefficient lookup -- and mapped onto
    one of :data:`_CANONICAL_REFERENCE_XC` when libxc parses it to that
    functional. Any other recognized functional is returned in its stripped,
    lower-cased form.
    """
    if not isinstance(reference_xc, str) or not reference_xc.strip():
        raise ValueError(
            f"reference_xc must be a non-empty pyscf/libxc functional string, "
            f"got {reference_xc!r}")
    from pyscf.dft import libxc
    name = reference_xc.strip().lower()
    try:
        parsed = libxc.parse_xc(name)
    except (KeyError, ValueError) as exc:
        # libxc: "LibXCFunctional: name 'X' not found."
        detail = exc.args[0] if exc.args else exc
        raise ValueError(
            f"reference_xc={reference_xc!r} is not a functional libxc "
            f"recognizes ({detail})") from exc
    for canonical in _CANONICAL_REFERENCE_XC:
        if name != canonical and parsed == libxc.parse_xc(canonical):
            return canonical
    return name


# Cycle cap of the reference SCF's DIIS stage (pyscf's default is 50).
# Measured on the O atom (3P) at def2-SVP / grid level 1 under the
# orientation lock, the one recipe that stalls on this machine: PBE DIIS runs
# from the minao guess that did converge took 38, 48 and 53 cycles -- one of
# them past pyscf's cap -- while a run that has stalled does not recover with
# more cycles (|g| 3.2e-4 to 6.2e-4 at 50 cycles, 3.8e-4 at 100, 8.3e-5 at
# 200, against the 3.2e-5 criterion). 100 covers the slowest converging run
# with margin and bounds what a stalled run spends before the second-order
# stage takes over. A run that converges within pyscf's 50 cycles is
# unaffected: the DIIS loop breaks at convergence and the cap never enters the
# iterates.
_REFERENCE_SCF_MAX_CYCLE = 100
# Macro-iteration cap of the second-order stage (pyscf's own SOSCF default).
# Measured: 7 macro-iterations on the locked PBE O-atom stall, 2 to 4 on
# H2O / SCAN started from a one- to three-cycle DIIS density.
_REFERENCE_SCF_NEWTON_MAX_CYCLE = 50
# Convergence threshold of the reference SCF, both stages: pyscf's default,
# stated here and stamped into every record beside the orbital gradient of the
# density the SCF returns (reference_scf_gradient). pyscf's loop converges on
# the gradient of h1e + vhf (no DIIS) under its bar sqrt(conv_tol), then runs
# one extra diagonalization and returns THAT density, accepting it when its
# energy moved by less than 10 conv_tol or its gradient is under 3 times the
# bar (scf/hf.py, the extra cycle). The energy branch bounds nothing on the
# gradient by itself, and the second-order kernel has no extra cycle, so 3
# times the bar is the DIIS kernel's stated ceiling rather than a bound on
# every returned density; no converged case above it was found in some 250
# small-basis configurations, the largest 2.26 times, and five forced
# second-order records sit at 0.03 to 0.35 times. Measured 1.024 times
# for singlet CH2 / SCAN / def2-svp / grid level 3 under the 3e-5 lock (7
# cycles) and 2.258 times for the same molecule bent to 1.44 A / 102 degrees
# under PBE. It is held at 1e-9 because the
# orientation lock's reproducibility was calibrated at it and does not survive
# a decade tighter: the locked O atom (PBE, def2-svp, grid level 3, the
# pretraining identity) converges in 7 cycles at 1e-9 with two processes
# landing on one density (second moments 3.369559 / 3.803699 / 3.749466 in
# both), while at 1e-10 the two processes wander 37 and 42 cycles along the
# lock's weakly broken flat direction and land on different densities
# (moments 3.371581 / 3.771174 / 3.779969 against 3.371541 / 3.770797 /
# 3.780388, 1.4e-3 relative in rho on 94 percent of the grid, energies 9e-8
# Ha above the 1e-9 point). The consumers that rebuild the orbital gradient
# from a record's Fock pieces hold it to twice pyscf's gradient criterion for
# the reason stated at pretrain_data_gen._GRADIENT_CHECK_MARGIN.
_REFERENCE_SCF_CONV_TOL = 1e-9


def _converge_reference_scf(mf, label="the reference SCF"):
    """Run ``mf`` to pyscf's convergence criterion: DIIS first, then the
    second-order solver (SOSCF) from the best-gradient point of the DIIS
    trajectory if DIIS stalls.

    Returns ``(mf_out, cycles, solver)``: the object whose ``e_tot``,
    ``make_rdm1()`` and ``get_veff`` hold the solution -- ``mf`` itself after a
    DIIS convergence, the SOSCF wrapper (which shares ``mf``'s grids, numint
    and instance-level ``get_hcore``, so an orientation lock stays applied)
    after the second stage -- the SCF cycles run in total (DIIS cycles plus
    second-order macro-iterations), and ``"diis"`` or ``"diis+newton"``. The
    caller reads ``converged`` from the returned object. ``label`` names the
    SCF in refusal messages. A second stage that cannot run (the empty
    rotation space below) or that raises inside pyscf is reported as
    :class:`ReferenceSCFNotConverged` carrying the DIIS cycle count -- never
    as the bare pyscf exception.

    Both stages test the same criterion, |g| < sqrt(conv_tol) and
    dE < conv_tol (``scf.hf.kernel``; ``soscf.newton_ah.kernel`` derives its
    ``conv_tol_grad`` from ``conv_tol`` the same way), so the second stage
    changes the minimizer, not the bar; the bar is
    :data:`_REFERENCE_SCF_CONV_TOL`, set here on both stages (see the
    constant for the measurement behind it). The reference SCF always starts from
    the minao guess: a start from a converged PBE density of the same system
    was measured and rejected, not because the minao start cannot stall -- it
    can, on both functionals (unlocked SCAN O atom at def2-SVP / grid level
    1: 2 of 12 processes stalled, one at a final gradient of 8.98e-5, with
    another run converging at 46 of 50 cycles; the stalled draw is exactly
    what the second stage rescues, measured 106 total cycles, diis+newton) --
    but because the seed adds hazards with no return: under the orientation
    lock it carried the PBE orientation into the SCAN run and converged to a
    stationary point 1.7e-4 Ha above the lock's minimum in 2 of 3 processes
    (the minao start reached the minimum in 9 of 9), unlocked it stalled in 1
    of 3 processes itself, and it changes the pruned grid (9088 -> 9080
    points on H2O / sto-3g) unless the grid is pre-initialized with the minao
    guess. The second stage starts from the lowest-gradient density of the
    DIIS trajectory, not from the guess, for the same reason and one more.
    Against the guess the case was measured on the O atom: SOSCF from the
    minao guess converged the locked SCAN O atom to a point 8e-5 Ha above
    the DIIS solution, whereas from the DIIS end point -- which for a stall
    that stays in the solution basin is itself near-best -- it reproduces
    the DIIS energy to 2e-10 Ha (SCAN, converged case) and lands within
    1e-6 Ha of converged DIIS attempts (PBE stall: 4e-9 Ha on one draw,
    2.3e-8 to 9.8e-7 Ha over five further rescued draws -- the
    flat-direction slack the 3.2e-5 gradient criterion leaves). Against the
    end point the case was measured on the Li atom at SCAN /
    6-311++G(3df,2pd) / grid level 3, where DIIS leaves the basin after
    reaching it: the best point comes at cycle 5 (E=-7.478697644723,
    |g|=7.5e-4), then the extrapolation throws the density into an
    unphysical state and wanders chaotically to the cycle cap (the
    post-explosion trajectory is draw-dependent: end points E -4.07 and
    -3.845 and SOSCF-stall gradients 1.2e-3 to 4e-3 were observed across
    draws; the pre-explosion basin numbers above reproduce); SOSCF from
    that final point runs its 50 macro-iterations unconverged, while from
    the best point it converges in 2 macro-iterations to E=-7.4786979415.
    A level shift (0.25 / 0.5), damping (0.5), a finer grid and removing
    the orientation lock were each measured on that case and none
    converges it. The best point is recorded by an ``mf.callback`` on the
    first stage -- a plain numpy copy of the cycle's density, kept with
    its |g| whenever the cycle's |g| is strictly below the best seen; a
    callback the caller had already installed keeps firing (the recorder
    chains to it) and is restored before any return, so the returned
    object carries the caller's callback and no trace of the recorder.
    The best-by-|g| selection is measured at the reference path's own
    solver settings; under caller-modified settings it is not universally
    the better start (measured: with a level shift, damping, or an all-ones
    guess, 7 of 27 differing rescues favoured the end point by up to
    3.09e-5 Ha), and this path sets its own settings above. For a system
    this rescue previously started from the end point, the second stage's
    start -- and with it the converged endpoint -- may move within the
    flat-direction slack quantified above (2.3e-8 to 9.8e-7 Ha).
    """
    mf.max_cycle = _REFERENCE_SCF_MAX_CYCLE
    mf.conv_tol = _REFERENCE_SCF_CONV_TOL
    # Trajectory-best density of the DIIS stage, for the rescue below: DIIS
    # can reach the solution basin and then leave it (the Li case in the
    # docstring -- best at cycle 5, unphysical from cycle 7, chaotic to the
    # cycle cap), so the density handed to the second stage is the
    # lowest-gradient one seen, not the last. scf.hf.kernel calls the
    # callback with its locals once per cycle (exactly ``mf.cycles`` times
    # in pyscf 2.11.0, converged or not); norm_gorb there is the plain-Fock orbital
    # gradient of that cycle's density, the quantity both stages converge
    # on, and make_rdm1(mo_coeff, mo_occ) holds for RKS and UKS alike.
    best = {"dm": None, "gorb": np.inf}
    caller_callback = getattr(mf, "callback", None)

    def _record_best(envs):
        gorb = float(envs.get("norm_gorb", np.inf))
        if gorb < best["gorb"]:
            best["gorb"] = gorb
            best["dm"] = np.array(
                envs["mf"].make_rdm1(envs["mo_coeff"], envs["mo_occ"]))
        if callable(caller_callback):
            caller_callback(envs)

    mf.callback = _record_best
    mf.kernel()
    # Restored (not nulled) before every return so neither the returned
    # object nor the SOSCF wrapper (whose constructor copies mf's instance
    # attributes) carries the recording closure and its density copy, and a
    # callback the caller installed survives the call.
    mf.callback = caller_callback
    cycles = int(mf.cycles)
    if mf.converged:
        return mf, cycles, "diis"
    # The measured crash class of the second stage: an UNRESTRICTED system
    # whose orbital-rotation space is empty in both spin channels -- the H
    # atom in a minimal basis (alpha 1 occupied, 0 virtual; beta 0 occupied,
    # 1 virtual; zero occupied-virtual pairs) -- makes pyscf's SOSCF pack a
    # scalar in place of the rotation vector and die in newton_ah.rotate_mo
    # with TypeError: 'float' object is not subscriptable. The electron count
    # is not the trigger: H at def2-svp (same nelec (1, 0), four alpha
    # pairs) runs the stage, and the restricted zero-pair case (He at
    # sto-3g) converges in it. Refused by name here so the certificate sees
    # a convergence refusal, not a pyscf internal.
    occ = np.asarray(mf.mo_occ)
    channels = occ if occ.ndim == 2 else occ[None]
    rotation_pairs = sum(
        int(np.count_nonzero(o > 0)) * int(np.count_nonzero(o == 0))
        for o in channels)
    if occ.ndim == 2 and rotation_pairs == 0:
        n_occ = [int(np.count_nonzero(o > 0)) for o in channels]
        raise ReferenceSCFNotConverged(
            f"{label} did not converge in the DIIS stage (converged=False, "
            f"cycles={cycles}, max_cycle={_REFERENCE_SCF_MAX_CYCLE}), and "
            "the second-order stage cannot run it: the unrestricted "
            "orbital-rotation space is empty in both spin channels "
            f"(occupied per channel {n_occ} of {int(channels.shape[1])} "
            "orbitals, no occupied-virtual pairs), which pyscf's SOSCF "
            "cannot represent.",
            cycles=cycles)
    try:
        so = mf.newton()
        so.max_cycle = _REFERENCE_SCF_NEWTON_MAX_CYCLE
        so.conv_tol = _REFERENCE_SCF_CONV_TOL
        macro = []
        # newton_ah.kernel calls back with its locals after every
        # macro-iteration and once more after the loop; the last imacro is
        # the count minus one.
        so.callback = lambda envs: macro.append(int(envs["imacro"]))
        # The trajectory-best density; the DIIS end point only if the
        # recorder never fired (defensive -- the kernel invokes the
        # callback on every cycle it runs).
        so.kernel(dm0=best["dm"] if best["dm"] is not None
                  else mf.make_rdm1())
    except Exception as exc:
        raise ReferenceSCFNotConverged(
            f"{label}: the second-order stage raised "
            f"{type(exc).__name__}: {exc} after the DIIS stage "
            f"(cycles={cycles}, max_cycle={_REFERENCE_SCF_MAX_CYCLE}).",
            cycles=cycles) from exc
    newton_cycles = (macro[-1] + 1) if macro else 0
    return so, cycles + newton_cycles, "diis+newton"


def _precompute_cache_key(
    mol_spec: MoleculeSpec,
    required_keys: tuple[str, ...],
    descriptors: tuple[Descriptor, ...],
    auxbasis: str | None = None,
    orientation_lock_strength: float = 0.0,
    seed_source: str = "pbe",
    seed_cache_dir: str | None = None,
    seed_density_fit: bool = False,
    reference_xc: str = "pbe",
) -> tuple:
    # MoleculeSpec is a frozen dataclass and hashes by structural identity.
    # required_keys are sorted to canonicalize set-equivalence.
    # Descriptors are keyed by class name + n_features so different
    # parameterizations of the same descriptor type don't collide.
    # The external_data_path file's (mtime_ns, size) is part of the key
    # so that re-running a notebook after vxc_ref regeneration (e.g.
    # step6's mid-notebook OEP rerun) invalidates stale cache entries.
    desc_key = tuple(
        (type(d).__name__, getattr(d, "n_features", None),
         # include settings that affect descriptor compute so e.g.
         # CuspDescriptor(log_transform=True) does not collide with the
         # untransformed variant in the cache. (The dm_statistics `intensive`
         # flag left this tuple 2026-08-06 with the dm_entropy removal.)
         getattr(d, "log_transform", False),
         # rung-3.5 projector width: distinct alpha -> distinct projected-AO A,
         # so DMRung35Descriptor(alpha=...) variants must not collide in cache.
         getattr(d, "alpha", None),
         getattr(d, "alphas", None))
        for d in descriptors
    )
    ext_path = getattr(mol_spec, "external_data_path", None)
    if ext_path and os.path.isfile(ext_path):
        st = os.stat(ext_path)
        ext_key = (ext_path, int(st.st_mtime_ns), int(st.st_size))
    else:
        ext_key = (ext_path, None, None)
    # auxbasis is part of the key: the DF auxiliary basis lives on SolverConfig,
    # not MoleculeSpec, so two runs with the same molecule but different auxbasis
    # would otherwise collide on the cached cderi.
    # orientation_lock_strength is likewise part of the key: it perturbs h_core
    # (and thus the PBE seed), so a locked run must not reuse an unlocked cache
    # entry (or one locked at a different strength).
    # The seed axis: a seed-blind key would hand a "pbe"-seeded record to a
    # "scan"/"minao" caller (or vice versa). seed_cache_dir and the DF flag
    # are part of the loaded file's identity, so they key too.
    # reference_xc keys for the same reason and a stronger one: it selects the
    # SCF that produced the density EVERY field is built from, so a blind key
    # would hand a PBE record to a SCAN caller and the fidelity certificate
    # would silently measure a meta-GGA network against the wrong density.
    # The slot holds the CANONICAL spelling (canonical_reference_xc), so
    # "SCAN", " scan" and "scan,scan" are one entry and one SCF, not three.
    return (mol_spec, tuple(sorted(required_keys)), desc_key, ext_key, auxbasis,
            float(orientation_lock_strength),
            (str(seed_source), seed_cache_dir, bool(seed_density_fit)),
            canonical_reference_xc(reference_xc))


def seed_geometry_tag(atom: str, charge: int, spin: int) -> str:
    """8-hex geometry/identity tag for seed cache filenames.

    A species NAME alone cannot identify a seed: the training set and the
    held-out pool both contain e.g. an ``H2O`` -- at DIFFERENT geometries --
    and a filename-only cache would hand one the other's converged dm (the
    overlap fingerprint then fails loud, but the cache could never hold
    both). The tag hashes the CANONICALIZED atom string (symbols +
    coordinates rounded to the ``atoms_to_pyscf_str`` 1e-6 Angstrom
    precision, so formatting differences between producers hash
    identically) together with charge and spin. Identical geometries
    deduplicate to one file; twins separate.
    """
    import hashlib
    parts = []
    for tok in str(atom).split(";"):
        p = tok.split()
        if not p:
            continue
        if len(p) != 4:
            raise ValueError(
                f"seed_geometry_tag: malformed atom token {tok!r} (expected "
                f"'Sym x y z'); a silently-partial hash could alias distinct "
                f"geometries")
        parts.append(f"{p[0]} {float(p[1]):.6f} {float(p[2]):.6f} "
                     f"{float(p[3]):.6f}")
    key = "; ".join(parts) + f"|q{int(charge)}|s{int(spin)}"
    return hashlib.sha1(key.encode()).hexdigest()[:8]


def seed_qualified_name(mol_spec: MoleculeSpec) -> str:
    """The geometry-qualified cache name for ``mol_spec``'s seed."""
    return (f"{mol_spec.name}_gh"
            f"{seed_geometry_tag(mol_spec.atom, mol_spec.charge, mol_spec.spin)}")


def seed_cache_file(mol_spec: MoleculeSpec, *, seed_cache_dir: str,
                    density_fit: bool = False,
                    orientation_lock_strength: float = 0.0) -> str:
    """Path of the cached SCAN seed npz for ``mol_spec`` (may not exist).

    Mirrors ``external_refs._intermediate_cache_name`` at the seed identity:
    (geometry-qualified name, basis, grid level, DF tag, orientation lock,
    xc="scan"). A ``grid_level`` of None on the spec normalizes to 3 (the
    PySCF default the precompute SCF then runs at).
    """
    from xcquinox.alec.external_refs import _intermediate_cache_name
    gl = mol_spec.grid_level if mol_spec.grid_level is not None else 3
    fname = _intermediate_cache_name(
        seed_qualified_name(mol_spec), grid_level=gl, basis=mol_spec.basis,
        density_fit=bool(density_fit), kind="scf",
        orientation_lock_strength=float(orientation_lock_strength),
        xc="scan")
    return os.path.join(seed_cache_dir, "_intermediates", fname)


def missing_seed_cache_files(mol_specs, *, seed_cache_dir: str,
                             density_fit: bool = False,
                             orientation_lock_strength: float = 0.0
                             ) -> list[str]:
    """Names of the specs in ``mol_specs`` with no cached SCAN seed on disk.

    The cheap coverage gate run before a val/eval precompute loop, so a
    wrong or incomplete cache dir fails loud up front instead of mid-run.
    """
    missing = []
    for ms in mol_specs:
        if not os.path.isfile(seed_cache_file(
                ms, seed_cache_dir=seed_cache_dir, density_fit=density_fit,
                orientation_lock_strength=orientation_lock_strength)):
            missing.append(ms.name)
    return missing


def _load_scan_seed_dm(mol_spec: MoleculeSpec, *, s_live,
                       seed_cache_dir: str | None,
                       density_fit: bool, auxbasis: str | None,
                       orientation_lock_strength: float,
                       allow_generate: bool):
    """Converged SCAN dm for ``mol_spec`` from the seed cache.

    The dm comes from a SEPARATE mean-field (``run_scf_with_cache``), never
    from this precompute's grid-owning kernel, so the integration grid is
    untouched by the seed choice (grid-identity rule). Cache identity is
    filename-only, so a loaded dm must pass the overlap-matrix fingerprint
    against the live molecule (shape checks alone pass for isomers).
    Generation on a cache miss is double-gated: the ``allow_generate``
    kwarg (True only for training-side call sites) AND the
    ``XCQUINOX_SEED_ALLOW_GENERATE=1`` environment flag (exported only by
    cluster task scripts) -- local runs fail loud instead of silently
    starting a production-basis SCAN SCF.
    """
    cache_dir = seed_cache_dir or os.environ.get("XCQUINOX_SEED_CACHE_DIR")
    if not cache_dir:
        raise RuntimeError(
            f"seed_source='scan' for {mol_spec.name!r} but no seed cache "
            "dir is configured: set SolverConfig.seed_cache_dir (via "
            "inputs.seed_cache_dir) or the XCQUINOX_SEED_CACHE_DIR "
            "environment variable"
        )
    path = seed_cache_file(
        mol_spec, seed_cache_dir=cache_dir, density_fit=density_fit,
        orientation_lock_strength=orientation_lock_strength)
    if not os.path.isfile(path):
        env_ok = os.environ.get("XCQUINOX_SEED_ALLOW_GENERATE") == "1"
        if not (allow_generate and env_ok):
            raise RuntimeError(
                f"no cached SCAN seed for {mol_spec.name!r} at {path} -- "
                "run the seed-cache job over the training species, or point "
                "seed_cache_dir at the scan-pool cache for pool species; "
                "on-cluster training-side generation requires BOTH "
                "seed_allow_generate and XCQUINOX_SEED_ALLOW_GENERATE=1"
            )
    from xcquinox.alec.benchmark_refs import _mol_spec_to_atoms
    from xcquinox.alec.external_refs import SpeciesEntry, run_scf_with_cache
    gl = mol_spec.grid_level if mol_spec.grid_level is not None else 3
    # geometry-qualified cache identity: same-name species at different
    # geometries (training vs pool twins) must resolve to distinct files
    entry = SpeciesEntry(name=seed_qualified_name(mol_spec),
                         charge=int(mol_spec.charge),
                         spin=int(mol_spec.spin), source="seed")
    rec = run_scf_with_cache(
        entry, _mol_spec_to_atoms(mol_spec), cache_dir=cache_dir,
        basis=mol_spec.basis, grid_level=gl, density_fit=bool(density_fit),
        auxbasis=auxbasis,
        orientation_lock_strength=float(orientation_lock_strength),
        xc="scan")
    dm = np.asarray(rec["dm"])
    s_npz = rec.get("S")
    s_live = np.asarray(s_live)
    if (s_npz is None or np.asarray(s_npz).shape != s_live.shape
            or not np.allclose(np.asarray(s_npz), s_live,
                               rtol=1e-6, atol=1e-8)):
        raise RuntimeError(
            f"SCAN seed cache for {mol_spec.name!r} at {path} fails the "
            "overlap-matrix fingerprint: the cached S does not match the "
            "live molecule (geometry/basis mismatch behind a same-name "
            "cache file)"
        )
    return dm


def clear_precompute_cache() -> None:
    """Wipe the in-memory precompute cache. Tests use this to isolate runs."""
    _PRECOMPUTE_CACHE.clear()


def set_precompute_cache_enabled(enabled: bool) -> None:
    """Toggle the in-memory precompute cache (default: enabled).

    Disable when calling precompute on streaming / changing inputs where the
    same MoleculeSpec object is reused with mutated external_data on disk.
    """
    global _PRECOMPUTE_CACHE_ENABLED
    _PRECOMPUTE_CACHE_ENABLED = bool(enabled)


def precompute_fixed_density_data(
    mol_spec: MoleculeSpec,
    required_keys: tuple[str, ...] = (),
    descriptors: tuple[Descriptor, ...] = (),
    auxbasis: str | None = None,
    orientation_lock_strength: float = 0.0,
    seed_source: str = "pbe",
    seed_cache_dir: str | None = None,
    seed_density_fit: bool = False,
    seed_allow_generate: bool = False,
    reference_xc: str = "pbe",
) -> MoleculeData:
    """Run the reference SCF, extract grid data, return a MoleculeData dict.

    ``reference_xc`` selects the functional of that SCF, and therefore the
    density every grid quantity, every descriptor block (total-density and
    per-spin-channel) and ``E_pbe`` / ``E_xc_pbe`` / ``E_non_xc`` are built
    from. It is recorded in the result as ``reference_xc``. The default
    ``"pbe"`` is the whole training and evaluation pipeline; the
    pretraining-fidelity certificate requests ``"scan"`` for a meta-GGA
    architecture, because SCAN is the parent functional those networks were
    pretrained against and the certificate must measure them on the density
    they have to reproduce. This is deliberately a parameter of this one
    construction rather than a second construction elsewhere: the density
    determines eighteen separate fields, and two code paths building them
    would have to be kept identical by hand.

    The name is canonicalized first (:func:`canonical_reference_xc`: stripped,
    lower-cased, and resolved to ``"pbe"`` / ``"scan"`` when libxc parses it
    to that functional), so every spelling of one functional is one memo
    entry, one SCF and one recorded name. Pure semilocal functionals only: a
    hybrid or a non-local-correlation (VV10) functional is refused, see
    below. The reference SCF must converge: DIIS from the minao guess first,
    the second-order solver from the best-gradient point of the DIIS
    trajectory if DIIS stalls
    (:func:`_converge_reference_scf`), and an SCF unconverged after both
    raises :class:`ReferenceSCFNotConverged` instead of producing a record.
    The record's ``mol_metadata`` carries ``reference_xc``,
    ``reference_scf_converged``, ``reference_scf_cycles``,
    ``reference_scf_solver``, and the two reproducibility stamps
    ``reference_xc_blksize`` (grid points per block of the reference SCF's
    XC quadrature, pinned by :mod:`xcquinox.alec.pyscf_determinism` so the
    summation order does not follow the process's memory history) and
    ``reference_blas_threads`` (pyscf's OpenMP worker count, which the
    record's last digits depend on above one thread); ``reference_eri_path``
    records whether the two-electron integrals were held in memory
    ("incore") or built directly ("direct"), a choice pinned to the system
    size by the same module.

    Baseline keys are always populated. Reference/descriptor keys are computed
    on-demand based on required_keys and descriptor.required_mol_keys.
    Unused keys are set to None for treedef homogeneity.

    Results are memoized in a process-level dict keyed on
    ``(mol_spec, sorted(required_keys), descriptor_classes)``. The
    precompute is pure (the reference SCF on a frozen geometry), so caching is
    correctness-preserving and gives O(N_specs) speedup when the notebook
    sweep evaluates the same molecule under many trained models.
    Disable via :func:`set_precompute_cache_enabled` if external_data on
    disk changes between calls.
    """
    if seed_source not in ("pbe", "scan", "minao"):
        raise ValueError(
            f"seed_source must be one of 'pbe'/'scan'/'minao', got "
            f"{seed_source!r}")
    # Canonical spelling before the key, the SCF and the record: "SCAN",
    # " scan" and "scan,scan" are one functional to libxc and are one record
    # here. Validates the name (ValueError) before anything is paid for.
    reference_xc = canonical_reference_xc(reference_xc)
    cache_key = None
    if _PRECOMPUTE_CACHE_ENABLED:
        try:
            cache_key = _precompute_cache_key(
                mol_spec, required_keys, descriptors, auxbasis,
                orientation_lock_strength, seed_source, seed_cache_dir,
                seed_density_fit, reference_xc)
        except TypeError:
            cache_key = None  # mol_spec or descriptors not hashable
        if cache_key is not None and cache_key in _PRECOMPUTE_CACHE:
            return _PRECOMPUTE_CACHE[cache_key]

    from pyscf import dft, gto
    from pyscf.dft import libxc

    # A hybrid reference would break the E_xc / E_non_xc split this record is
    # built on: pyscf reports only the SEMILOCAL part of the XC energy in
    # veff.exc and books the exact-exchange piece with the Coulomb term, so
    # E_non_xc = E_tot - E_xc would silently absorb it and every trained
    # functional would sit on top of a hidden exact-exchange term. Measured:
    # libxc.hybrid_coeff is 0.2 for b3lyp, 0.25 for pbe0 and 1.0 for the
    # range-separated wb97x, against 0.0 for pbe and scan.
    _hyb = float(libxc.hybrid_coeff(reference_xc))
    _omega = float(libxc.rsh_coeff(reference_xc)[0])
    if _hyb != 0.0 or _omega != 0.0:
        raise ValueError(
            f"reference_xc={reference_xc!r} is a hybrid functional "
            f"(hybrid_coeff={_hyb}, rsh omega={_omega}); the reference SCF "
            "must be a pure (semilocal) functional, because E_xc_pbe / "
            "E_non_xc split the total energy at the semilocal XC term and an "
            "exact-exchange contribution would be booked as non-XC.")
    # A non-local-correlation reference breaks the same split from the other
    # side: pyscf evaluates the VV10 kernel inside get_veff and books its
    # energy in veff.exc, where no point-wise semilocal consumer of the record
    # (the functional being trained, the certificate's numint route) can see
    # it. Measured on H2O / sto-3g / grid 1: veff.exc for b97m-v sits 4.3e-2
    # Ha from the semilocal numint value of the same functional on the same
    # density, 1.3e-2 Ha for scan_vv10. pyscf also reads the hyphen in
    # "scan-vv10" as a subtraction (SCAN minus the VV10 semilocal part; total
    # energy -65.83 Ha against SCAN's -75.29 Ha), and libxc.is_nlc flags that
    # string too.
    if bool(libxc.is_nlc(reference_xc)):
        raise ValueError(
            f"reference_xc={reference_xc!r} carries a non-local correlation "
            "(VV10) term; the reference SCF must be a semilocal functional, "
            "because pyscf books the non-local energy inside veff.exc where "
            "no point-wise consumer of this record can evaluate it.")

    # Build pyscf molecule
    mol = gto.M(
        atom=mol_spec.atom,
        basis=mol_spec.basis,
        charge=mol_spec.charge,
        spin=mol_spec.spin,
        verbose=0,
    )

    # Run the reference SCF (reference_xc; "pbe" for the training pipeline)
    is_unrestricted = mol_spec.spin != 0
    if is_unrestricted:
        mf = dft.UKS(mol)
    else:
        mf = dft.RKS(mol)
    mf.xc = reference_xc
    # Pin grid level when the spec requires it (e.g., external rho_ref_grid
    # was generated on a non-default grid). Setting .level must happen before
    # the first kernel call so .build() picks it up.
    if mol_spec.grid_level is not None:
        mf.grids.level = mol_spec.grid_level
    # Orientation lock: bias h_core with a small fixed anisotropic quadrupole
    # BEFORE kernel(), so the PBE seed (dm_pbe) already picks the locked
    # degenerate component and the stored h_core the manual/oneshot SCF consumes
    # is the biased one. Applied identically in the CCSD reference generation so
    # ref and functional lock the same pi component. strength=0 -> no-op.
    orientation_lock_bias_mat = None
    if orientation_lock_strength:
        orientation_lock_bias_mat = orientation_lock_bias(
            mol, orientation_lock_strength)
        _base_hcore = np.asarray(mf.get_hcore())
        _locked_hcore = _base_hcore + orientation_lock_bias_mat
        mf.get_hcore = lambda *a, **k: _locked_hcore
    # Fixed quadrature blocking: pyscf sizes the XC grid loop from the memory
    # the process has left, so the summation order -- and the converged
    # energy and density at the 1e-13 level -- would otherwise follow the
    # process's memory history (measured: E_pbe of the locked O atom at
    # def2-svp / grid 3 is -74.91469870612937 in a clean process and
    # ...939 in one above pyscf's 4000 MB ceiling, at 2.0x the wall time).
    # Pinned before the first get_veff; the second-order stage below shares
    # this integrator. The incore/direct choice of the two-electron
    # integrals follows process memory the same way and is pinned to the
    # system size. The thread count is the caller's; all three go into
    # mol_metadata (reference_xc_blksize, reference_blas_threads,
    # reference_eri_path) and into no cache identity (see
    # pyscf_determinism).
    reference_pins = pin_reference_scf(mf)
    # The record is a set of properties of the reference functional's
    # SELF-CONSISTENT density. pyscf returns from kernel() with mf.converged
    # False when max_cycle runs out, and nothing downstream can tell such a
    # record from a converged one: measured on H2O / sto-3g / grid 1 with
    # SCAN stopped after one cycle, the total energy is +7.2e-2 Ha off the
    # converged value and the density matrix 0.315 off at its maximum. A DIIS
    # stall is handed to the second-order solver first (see
    # _converge_reference_scf); what is still unconverged after that is
    # refused. No caller runs a deliberately short reference SCF, so the
    # refusal is unconditional; the stamps go into mol_metadata for the
    # consumers that check provenance (cluster.fidelity reads them).
    mf, reference_scf_cycles, reference_scf_solver = _converge_reference_scf(
        mf, label=f"the reference {reference_xc} SCF for {mol_spec.name!r}")
    reference_scf_converged = bool(mf.converged)
    if not reference_scf_converged:
        raise ReferenceSCFNotConverged(
            f"the reference {reference_xc} SCF for {mol_spec.name!r} did not "
            f"converge: converged={reference_scf_converged}, "
            f"cycles={reference_scf_cycles} (DIIS max_cycle="
            f"{_REFERENCE_SCF_MAX_CYCLE}, second-order max_cycle="
            f"{_REFERENCE_SCF_NEWTON_MAX_CYCLE}, conv_tol={mf.conv_tol:g}); "
            f"geometry {mol_spec.atom!r}, charge {mol_spec.charge}, 2S "
            f"{mol_spec.spin}, basis {mol_spec.basis!r}, grid level "
            f"{mol_spec.grid_level}. A fixed-density record is a set of "
            "properties of the self-consistent density, so none is written "
            "for an unconverged reference SCF.",
            cycles=reference_scf_cycles)

    # Overlap conditioning gate
    s_matrix = mf.get_ovlp()
    cond_s = float(np.linalg.cond(s_matrix))
    if cond_s > 1e10:
        raise ValueError(
            f"Overlap matrix for {mol_spec.name!r} is ill-conditioned: "
            f"cond(S) = {cond_s:.2e} > 1e10. This typically indicates "
            f"near-linear-dependent basis functions."
        )

    # Extract SCF quantities
    dm_pbe = mf.make_rdm1()
    h_core = mf.get_hcore()
    # The orbital gradient of the density this SCF returns, pyscf's own
    # get_grad on the plain Fock of that density (the quantity pyscf's extra
    # cycle accepted): stamped so a consumer rebuilding the gradient from the
    # stored Fock pieces can hold the record to what its precompute measured.
    # The Fock and the gradient are read off the DIIS object whichever object
    # the driver returned, so the stamp is one definition; at pyscf 2.11.0
    # the second-order wrapper overrides neither get_fock nor get_grad and
    # the two readings agree to 1e-15 on a forced second-order record.
    _base = getattr(mf, "_scf", mf)
    reference_scf_gradient = float(np.linalg.norm(_base.get_grad(
        mf.mo_coeff, mf.mo_occ, _base.get_fock(dm=dm_pbe))))
    # NOTE (density-fitting): j_matrix / E_pbe are deliberately computed with the
    # FULL ERI even when SolverConfig.density_fit is on. The PBE result is a
    # fixed, reference-quality anchor (it seeds E_non_xc and the FIXED_J pin);
    # the DF approximation is applied ONLY to the NN-functional SCF Coulomb that
    # is being trained, not to this baseline. Keeping PBE full-ERI also makes
    # E_pbe byte-identical to the pre-DF pipeline.
    j_matrix = mf.get_j(mol, dm_pbe)
    e_nuc = float(mf.energy_nuc())
    E_pbe = float(mf.e_tot)

    # V_xc^PBE = V_eff - J
    # For UKS, mf.get_j returns per-spin J[dm_s]; veff[s] = V_xc[s] + J_total.
    # For RKS, mf.get_j returns J[dm_total].
    veff = mf.get_veff(mol, dm_pbe)
    if np.asarray(j_matrix).ndim == 3:  # UKS
        j_total = np.asarray(j_matrix).sum(axis=0)  # (nao, nao)
        vxc_pbe = np.asarray(veff) - j_total[np.newaxis, ...]  # (2, nao, nao)
    else:  # RKS
        vxc_pbe = np.asarray(veff) - np.asarray(j_matrix)

    # Grid quantities
    coords = mf.grids.coords
    weights = mf.grids.weights
    ao = mf._numint.eval_ao(mol, coords, deriv=1)
    ao_no_deriv = ao[0]

    # Total DM for density/sigma computation (always 2D)
    if dm_pbe.ndim == 2:
        dm_pbe_tot = dm_pbe
    else:
        dm_pbe_tot = dm_pbe[0] + dm_pbe[1]

    # PBE density and gradient on grid
    rho_pbe = np.einsum("pi,ij,pj->p", ao[0], dm_pbe_tot, ao[0])
    drho_x = 2 * np.einsum("pi,ij,pj->p", ao[1], dm_pbe_tot, ao[0])
    drho_y = 2 * np.einsum("pi,ij,pj->p", ao[2], dm_pbe_tot, ao[0])
    drho_z = 2 * np.einsum("pi,ij,pj->p", ao[3], dm_pbe_tot, ao[0])
    sigma_pbe = drho_x ** 2 + drho_y ** 2 + drho_z ** 2
    # Store nabla_rho as (n_grid, 3) so compute_vxc_nn can assemble the GGA
    # v_sigma term V_xc_ij += 2 * integral v_sigma nabla_rho . nabla(phi_i phi_j) dr.
    nabla_rho_pbe = np.stack([drho_x, drho_y, drho_z], axis=-1)

    # Reference XC energy and E_non_xc
    _xctype = libxc.xc_type(reference_xc)
    if dm_pbe.ndim == 3:  # UKS
        # Use pyscf's veff.exc which already has the correct spin-resolved
        # evaluation. The `veff` object was computed above
        # (mf.get_veff(mol, dm_pbe)); reuse its .exc.
        E_xc_pbe = float(veff.exc)
    elif _xctype == "MGGA":
        # A meta-GGA needs the kinetic-energy density, which the GGA row set
        # below cannot carry (eval_xc refuses the 4-row GGA set for a meta-GGA
        # outright: "cannot reshape array of size 4N into shape (1,5,N)"), so
        # the closed-shell meta-GGA reference reads the XC energy pyscf already
        # accumulated on this grid. The two routes are one quantity, not two:
        # on H2O/sto-3g/grid 1 veff.exc agrees with the point-wise route to
        # round-off -- 1.8e-15 Ha for the PBE reference, and 1.8e-15 Ha for
        # the SCAN reference against a point-wise meta-GGA evaluation: one ulp
        # of an E_xc of magnitude 9 Ha, with repeated measurements spanning 0
        # to 2 ulp.
        E_xc_pbe = float(veff.exc)
    else:  # RKS, LDA or GGA reference
        # The row set is the one libxc demands for the reference functional's
        # rung: 4 rows (value + gradient) for a GGA, 1 for an LDA, and the AO
        # table to match -- eval_rho takes the deriv-1 stack for a GGA and the
        # bare table for an LDA. A fixed "GGA" row set is refused for an LDA
        # reference (measured on H2O/sto-3g/grid 1: ValueError, cannot reshape
        # array of size 36352 into shape (1,1,9088)), and the deriv-1 stack
        # with xctype="LDA" is refused in turn (ValueError, too many values to
        # unpack). xc_type("pbe") is "GGA", so the training pipeline's default
        # passes exactly the arguments it always did.
        ao_for_xc = ao if _xctype == "GGA" else ao_no_deriv
        rho_for_xc = mf._numint.eval_rho(mol, ao_for_xc, dm_pbe_tot,
                                         xctype=_xctype)
        exc_pbe, _, _, _ = mf._numint.eval_xc(reference_xc, rho_for_xc, spin=0)
        E_xc_pbe = float(np.sum(rho_pbe * exc_pbe * weights))
    E_non_xc = E_pbe - E_xc_pbe

    # Occupancies
    if is_unrestricted:
        nocc = None
        nocc_a = (mol.nelectron + mol.spin) // 2
        nocc_b = (mol.nelectron - mol.spin) // 2
    else:
        nocc = mol.nelectron // 2
        nocc_a = None
        nocc_b = None

    # Collect all needed keys from descriptors
    all_needed = set(required_keys)
    for d in descriptors:
        all_needed.update(d.required_mol_keys)

    # Descriptor features (on-demand)
    cusp_features = None
    dm_features = None
    rung35_proj_ao = None
    rung35_features = None
    rung35ms_proj_ao = None
    rung35ms_features = None
    metagga_features = None

    if "cusp_features" in all_needed:
        from xcquinox.features import compute_cusp_descriptor
        nuclear_coords = jnp.array(mol.atom_coords())
        nuclear_charges = jnp.array([mol.atom_charge(i) for i in range(mol.natm)])
        # pull the log_transform flag from the CuspDescriptor instance so
        # precompute matches what the descriptor's consumer expects.
        cusp_log_transform = False
        for d in descriptors:
            if type(d).__name__ == "CuspDescriptor":
                cusp_log_transform = bool(getattr(d, "log_transform", False))
                break
        cusp_features = compute_cusp_descriptor(
            jnp.array(coords), nuclear_coords, nuclear_charges,
            log_transform=cusp_log_transform,
        )

    if "dm_features" in all_needed:
        from xcquinox.features import compute_dm_features_array
        # Pass the SPIN-RESOLVED 3-D DM for UKS molecules so
        # compute_dm_features picks the per-spin idempotency-projector
        # branch (Pople-Nesbet 1954: D_sigma S D_sigma = D_sigma per spin).
        # Passing dm_pbe_tot (the spin-summed total) would force the RKS
        # branch and produce a non-zero, physically-meaningless
        # idempotency_error on every open-shell molecule because
        # (D_a + D_b)/2 · S · (D_a + D_b)/2 != (D_a + D_b)/2 (the cross
        # terms D_a S D_b survive).
        dm_for_features = jnp.array(dm_pbe) if dm_pbe.ndim == 3 \
                         else jnp.array(dm_pbe_tot)
        # No descriptor flag to thread since dm_entropy (and with it the
        # `intensive` normalization) was removed 2026-08-06; the remaining two
        # features have no configurable form.
        dm_feat_global = compute_dm_features_array(
            dm_for_features, jnp.array(s_matrix),
        )
        dm_features = jnp.tile(dm_feat_global, (len(rho_pbe), 1))

    if "rung35_features" in all_needed:
        from xcquinox.alec.rung35 import (
            compute_projected_ao, compute_rung35_occupancy, DEFAULT_RUNG35_ALPHA)
        # Pull the projector width from the DMRung35Descriptor instance so the
        # precompute matches the descriptor's consumer (and the cache key, which
        # includes alpha). First instance wins if several are present.
        rung35_alpha = DEFAULT_RUNG35_ALPHA
        for d in descriptors:
            if type(d).__name__ == "DMRung35Descriptor":
                rung35_alpha = float(getattr(d, "alpha", DEFAULT_RUNG35_ALPHA))
                break
        # A_mu(r) = <chi_mu | normalized Gaussian projector at r> -- a constant
        # (DM/density-independent) precompute; coords are in Bohr (mf.grids.coords).
        rung35_proj_ao = jnp.array(compute_projected_ao(mol, coords, rung35_alpha))
        # One-shot per-spin occupancy A^T P_pbe A from the PBE DM. Pass the
        # SPIN-RESOLVED 3-D DM for UKS so the alpha/beta channels are correct;
        # for RKS the 2-D total DM is split evenly inside compute_rung35_occupancy.
        rung35_features = compute_rung35_occupancy(rung35_proj_ao, jnp.array(dm_pbe))

    if "rung35ms_features" in all_needed:
        from xcquinox.alec.rung35 import (
            compute_projected_ao_multishell,
            compute_rung35_multishell_occupancy,
            DEFAULT_RUNG35_MULTISHELL_ALPHAS)
        # Widths come from the descriptor instance so the precompute matches the
        # consumer and the cache key (which includes `alphas`). First instance
        # wins if several are present, mirroring the single-width branch.
        ms_alphas = DEFAULT_RUNG35_MULTISHELL_ALPHAS
        for d in descriptors:
            if type(d).__name__ == "DMRung35MultishellDescriptor":
                ms_alphas = tuple(getattr(d, "alphas",
                                          DEFAULT_RUNG35_MULTISHELL_ALPHAS))
                break
        rung35ms_proj_ao = jnp.array(
            compute_projected_ao_multishell(mol, coords, ms_alphas))
        rung35ms_features = compute_rung35_multishell_occupancy(
            rung35ms_proj_ao, jnp.array(dm_pbe))

    if "metagga_features" in all_needed:
        from xcquinox.alec.metagga import compute_tau_from_dm, compute_alpha
        # One-shot meta-GGA alpha from the PBE DM: total tau from the deriv=1 AO
        # gradients (ao[1:4]) contracted with dm_pbe, then SCAN alpha from the PBE
        # rho/sigma. The FULL SCF recomputes this each cycle from the live DM.
        _tau_pbe = compute_tau_from_dm(jnp.array(ao[1:4]), jnp.array(dm_pbe))
        metagga_features = compute_alpha(
            jnp.array(rho_pbe), jnp.array(sigma_pbe), _tau_pbe).reshape(-1, 1)

    # --- Per-spin-channel descriptor blocks (open shells only) --------------
    # Every UKS exchange evaluation is posed on the symmetric doubled density
    # diag(P_sigma, P_sigma) (Oliver and Perdew, Phys. Rev. A 20, 397 (1979)):
    # density 2 rho_sigma, gradient invariant 4 sigma_sigma_sigma,
    # kinetic-energy density 2 tau_sigma. The blocks below are that system's
    # descriptor features, one per channel; they are what the exchange term
    # consumes, while correlation keeps the total density and the total block.
    # A closed-shell molecule has rho_a = rho_b, so its per-channel block IS the
    # total block and these keys stay None.
    dm_features_a = None
    dm_features_b = None
    rung35_features_a = None
    rung35_features_b = None
    rung35ms_features_a = None
    rung35ms_features_b = None
    metagga_features_a = None
    metagga_features_b = None
    tau_spin_a = None
    tau_spin_b = None
    if is_unrestricted:
        from xcquinox.alec.descriptors import doubled_spin_dm
        dm_pbe_spin = jnp.array(dm_pbe)
        doubled = [doubled_spin_dm(dm_pbe_spin, s) for s in (0, 1)]
        if dm_features is not None:
            from xcquinox.features import compute_dm_features_array
            dm_features_a, dm_features_b = [
                jnp.tile(compute_dm_features_array(d, jnp.array(s_matrix)),
                         (len(rho_pbe), 1))
                for d in doubled
            ]
        if rung35_features is not None:
            from xcquinox.alec.rung35 import compute_rung35_occupancy
            # [n_sigma, n_sigma]: the channel's occupancy in BOTH spin slots,
            # each still inside the Bessel bound [0, 1].
            rung35_features_a, rung35_features_b = [
                compute_rung35_occupancy(rung35_proj_ao, d) for d in doubled
            ]
        if rung35ms_features is not None:
            from xcquinox.alec.rung35 import compute_rung35_multishell_occupancy
            # Column order stays ALPHA-MAJOR then spin, as in the total block.
            rung35ms_features_a, rung35ms_features_b = [
                compute_rung35_multishell_occupancy(rung35ms_proj_ao, d)
                for d in doubled
            ]
        if metagga_features is not None:
            from xcquinox.alec.metagga import compute_tau_from_dm, compute_alpha
            # Doubled-system density 2 rho_sigma and gradient invariant
            # 4 sigma_sigma_sigma, from the contraction the total-density
            # branch uses for rho_pbe / sigma_pbe. Only the iso-orbital
            # indicator consumes them, so they are built only here; the
            # matrix-linear blocks above take the doubled density matrix
            # directly. The contraction is kept unoptimized on purpose:
            # optimize=True routes it through a matmul whose different
            # summation order moves rho_sigma by 1e-15 relative on roughly
            # half of the grid points (47 percent on Li/grid 3, the beta tail
            # reaching 3e-13 relative), and alpha's tail amplification
            # tau/tau_unif (up to 9e7 on Li's beta channel) turns that into
            # O(1e-7) changes of the stored indicator (measured maxima
            # 1.6e-7 at grid level 1 and 2.7e-7 at grid level 3).
            rho_doubled = []
            sigma_doubled = []
            for s in (0, 1):
                d_s = np.asarray(dm_pbe[s])
                r_s = np.einsum("pi,ij,pj->p", ao[0], d_s, ao[0])
                gx_s = 2 * np.einsum("pi,ij,pj->p", ao[1], d_s, ao[0])
                gy_s = 2 * np.einsum("pi,ij,pj->p", ao[2], d_s, ao[0])
                gz_s = 2 * np.einsum("pi,ij,pj->p", ao[3], d_s, ao[0])
                rho_doubled.append(2.0 * r_s)
                sigma_doubled.append(4.0 * (gx_s ** 2 + gy_s ** 2 + gz_s ** 2))
            ao_grad_j = jnp.array(ao[1:4])
            tau_spin_a, tau_spin_b = [
                compute_tau_from_dm(ao_grad_j, jnp.array(dm_pbe[s]))
                for s in (0, 1)
            ]
            # compute_tau_from_dm sums the two spin slots of a 3-D density
            # matrix, so the doubled matrix supplies tau = 2 tau_sigma directly.
            metagga_features_a, metagga_features_b = [
                compute_alpha(jnp.array(rho_doubled[s]),
                              jnp.array(sigma_doubled[s]),
                              compute_tau_from_dm(ao_grad_j, doubled[s])
                              ).reshape(-1, 1)
                for s in (0, 1)
            ]

    eri = None
    if "eri" in all_needed:
        eri = jnp.array(mol.intor("int2e", aosym="s1"))

    # Density-fitted 3-index Coulomb tensor (naux, nao, nao). geometry+basis
    # only (NOT NN-dependent), so it is precomputed here and contracted in JAX
    # by the manual solver when SolverConfig.density_fit is on. Far smaller than
    # the full s1 ERI (naux*nao^2 vs nao^4) -> larger bases stay in memory.
    cderi = None
    if "cderi" in all_needed:
        from xcquinox.alec.df_jk import build_cderi
        # Forward the configured auxbasis so DF uses the intended fitting basis
        # (e.g. def2-universal-jkfit for def2-tzvpd) consistently with the CCSD
        # references / pretrain data. auxbasis=None -> df_jk.default_auxbasis.
        cderi = build_cderi(mol, auxbasis=auxbasis)

    # External reference data (dm_target / rho_ref_grid / E_ref_literature)
    # come from an optional .npz pointed to by mol_spec.external_data_path.
    # precompute only handles SCF-level quantities; CCSD/HF post-SCF
    # computations are the caller's responsibility and are injected through
    # this path so run_training / run_test pick them up automatically.
    dm_target = None
    rho_ref_grid = None
    ref_density_method = None
    E_ref_literature = None
    vxc_ref = None
    if mol_spec.external_data_path is not None:
        dm_target, rho_ref_grid, ref_density_method, E_ref_literature, vxc_ref = _load_external_data(
            mol_spec.external_data_path,
            dm_pbe_shape=tuple(np.asarray(dm_pbe).shape),
            rho_pbe_shape=tuple(np.asarray(rho_pbe).shape),
            vxc_pbe_shape=tuple(np.asarray(vxc_pbe).shape),
            mol_name=mol_spec.name,
            grid_level=mol_spec.grid_level,
            orientation_lock_strength=orientation_lock_strength,
        )

    # Cache pyscfad Mole for hot-path training (avoids Mole.build() inside
    # jit; see MoleculeData._pyscfad_mol docstring). pyscfad is optional,
    # so swallow any import/build failure and leave the slot as None, the
    # pyscfad backend's _build_pyscfad_mf will fall back to rebuilding.
    pyscfad_mol: object | None = None
    try:
        import pyscfad.gto as pyscfad_gto
        mol_ad = pyscfad_gto.Mole()
        mol_ad.atom = mol_spec.atom
        mol_ad.basis = mol_spec.basis
        mol_ad.charge = mol_spec.charge
        mol_ad.spin = mol_spec.spin
        mol_ad.verbose = 0
        mol_ad.build()
        pyscfad_mol = mol_ad
    except Exception:
        pyscfad_mol = None

    # --- SCF seed supply (per-rung seeding). Single dispatch point; the
    # solver consumes dm_seed unconditionally. "pbe" aliases the SAME array
    # object as dm_pbe (not a copy) so the default protocol is byte- and
    # identity-equal to the pre-seeding pipeline. The scan/minao seeds come
    # from OUTSIDE the grid-owning kernel above, so the integration grid is
    # identical across seed choices.
    dm_pbe_arr = jnp.array(dm_pbe)
    if seed_source == "minao":
        dm_seed_arr = jnp.array(mf.get_init_guess())
    elif seed_source == "scan":
        dm_seed_arr = jnp.array(_load_scan_seed_dm(
            mol_spec, s_live=s_matrix, seed_cache_dir=seed_cache_dir,
            density_fit=seed_density_fit, auxbasis=auxbasis,
            orientation_lock_strength=orientation_lock_strength,
            allow_generate=seed_allow_generate))
    else:
        dm_seed_arr = dm_pbe_arr

    result = MoleculeData(
        name=mol_spec.name,
        is_unrestricted=is_unrestricted,
        nocc=nocc,
        nocc_a=nocc_a,
        nocc_b=nocc_b,
        dm_pbe=dm_pbe_arr,
        dm_seed=dm_seed_arr,
        s_matrix=jnp.array(s_matrix),
        h_core=jnp.array(h_core),
        j_matrix=jnp.array(j_matrix),
        vxc_pbe=jnp.array(vxc_pbe),
        e_nuc=e_nuc,
        E_pbe=E_pbe,
        E_xc_pbe=E_xc_pbe,
        E_non_xc=E_non_xc,
        E_ref_literature=E_ref_literature,
        dm_target=dm_target,
        rho_ref_grid=rho_ref_grid,
        ref_density_method=ref_density_method,
        vxc_ref=vxc_ref,
        rho_grid=jnp.array(rho_pbe),
        sigma_grid=jnp.array(sigma_pbe),
        nabla_rho_grid=jnp.array(nabla_rho_pbe),
        grid_weights=jnp.array(weights),
        ao_grid=jnp.array(ao_no_deriv),
        ao_grid_deriv=jnp.array(ao),
        cusp_features=cusp_features,
        dm_features=dm_features,
        rung35_proj_ao=rung35_proj_ao,
        rung35_features=rung35_features,
        rung35ms_proj_ao=rung35ms_proj_ao,
        rung35ms_features=rung35ms_features,
        metagga_features=metagga_features,
        dm_features_a=dm_features_a,
        dm_features_b=dm_features_b,
        rung35_features_a=rung35_features_a,
        rung35_features_b=rung35_features_b,
        rung35ms_features_a=rung35ms_features_a,
        rung35ms_features_b=rung35ms_features_b,
        metagga_features_a=metagga_features_a,
        metagga_features_b=metagga_features_b,
        tau_spin_a=tau_spin_a,
        tau_spin_b=tau_spin_b,
        reference_xc=reference_xc,
        eri=eri,
        cderi=cderi,
        atom_composition=mol_spec.atom_composition,
        mol_metadata={
            "atom": mol_spec.atom,
            "basis": mol_spec.basis,
            "charge": mol_spec.charge,
            "spin": mol_spec.spin,
            "grid_level": mol_spec.grid_level,
            "auxbasis": auxbasis,
            # Precomputed orientation-lock bias (numpy, AO basis) so the pyscfad
            # backend can add it to its internally-built get_hcore without
            # recomputing intor on a traced pyscfad Mole. None when off.
            "orientation_lock_bias": orientation_lock_bias_mat,
            # Provenance of the reference SCF, beside the record-level
            # reference_xc: the canonical functional name, pyscf's convergence
            # flag (True in every record this function returns -- an
            # unconverged reference raises instead), the cycle count (DIIS
            # cycles plus any second-order macro-iterations) and which stages
            # ran ("diis", or "diis+newton" when DIIS stalled).
            "reference_xc": reference_xc,
            "reference_scf_converged": reference_scf_converged,
            "reference_scf_cycles": reference_scf_cycles,
            "reference_scf_solver": reference_scf_solver,
            # The threshold both stages converged to (_REFERENCE_SCF_CONV_TOL,
            # pyscf's default) and the orbital gradient of the density
            # returned, pyscf's own get_grad on its plain Fock: what a
            # consumer rebuilding the gradient from the stored pieces holds
            # the record to, and what pyscf's extra cycle accepted (under 3
            # times sqrt(conv_tol), or with an energy change under 10 conv_tol).
            "reference_scf_conv_tol": float(_REFERENCE_SCF_CONV_TOL),
            "reference_scf_gradient": reference_scf_gradient,
            # Grid points per block of the reference SCF's XC quadrature
            # (fixed; the summation order no longer follows process memory)
            # and pyscf's OpenMP worker count at the time. Metadata, not
            # identity: a consumer that compares records bitwise checks
            # them; every tolerance in use holds either order.
            "reference_xc_blksize": reference_pins.xc_blksize,
            "reference_blas_threads": reference_pins.threads,
            # And how its two-electron integrals were built ("incore" or
            # "direct"): pyscf's own choice follows process memory too.
            "reference_eri_path": reference_pins.eri_path,
        },
        _pyscfad_mol=pyscfad_mol,
    )
    if cache_key is not None and _PRECOMPUTE_CACHE_ENABLED:
        _PRECOMPUTE_CACHE[cache_key] = result
    return result
