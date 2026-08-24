"""Memory-independent numerics for the reference SCFs.

Two decisions inside a PySCF SCF are taken from the memory the process has
left, ``mf.max_memory - lib.current_memory()``, so the last digits of a
reference SCF followed the memory history of the process that ran it.

1. The block loop of the exchange-correlation quadrature.
   ``pyscf.dft.rks.get_veff`` (and ``uks``) hands the memory left to
   ``numint.nr_rks`` / ``nr_uks``, and ``NumInt.block_loop`` turns it into a
   block of ``max(4, min(int(mem / ((comp + 1) nao 8 BLKSIZE)),
   ngrids // BLKSIZE + 1, 1200)) * BLKSIZE`` grid points. The per-block Fock
   contributions and energies are summed block by block, so the ORDER of the
   quadrature summation changes with the block size.
2. Whether the two-electron integrals are held in memory or rebuilt
   directly. ``SCF.get_jk`` keeps the packed tensor when
   ``SCF._is_mem_enough()``, i.e. ``nao**4 / 1e6 + lib.current_memory() <
   0.95 * max_memory``, and otherwise builds J and K from screened
   integrals, incrementally over the DIIS iterates; the two paths differ at
   round-off.
3. The Cholesky-vector loops of a density-fitted object.
   ``pyscf.df.df_jk.get_jk`` sizes its Coulomb and exchange auxiliary
   blocks from ``dfobj.max_memory - lib.current_memory()``
   (``max(4, int(min(blockdim, mem * .3e6 / 8 / nao**2)))`` for J), so the
   number of fitted-tensor blocks the J and K sums run over follows live
   memory once naux exceeds one block. The def2-svp fitting bases stay
   under one block (O: naux 77, H2O: 113, against blockdim 240), so the
   dependence cannot bind there; at the production basis it does (CH4:
   naux 288, two blocks at normal headroom; C5H8: 888, four blocks, or 222
   blocks of 4 once the process passes the ceiling; 156 of the 214 BH76 and
   W4-11 species exceed one block at that basis, the smallest at naux 242),
   and every production campaign runs density fitting.

Measured on the O atom (def2-svp, grid level 3, orientation lock on, one
thread): a clean process integrates the 11904-point grid in one block with
the tensor in memory and lands on E_pbe = -74.91469870612937; the same
process holding 3.6 GiB sits above PySCF's 4000 MB default ceiling,
integrates in 54 blocks of 224 points with direct integrals, and lands on
-74.91469870612939 with a different density matrix, at 2.0x the wall time
(H2O, def2-svp, grid level 3: 137 blocks, 2.5x). With the block size fixed
but the integral path free, the two processes still differed in the last
digit (-74.91469870612937 against ...936); with both pinned they agree bit
for bit on every stored quantity. Every tolerance in use holds either value;
the effect is a floor under any bitwise comparison of records across
processes, and a 2-2.5x slowdown of every reference SCF once a process
passes the ceiling.

:func:`pin_xc_block_size` fixes the block size of one mean-field object's
numerical integrator at :data:`REFERENCE_XC_BLKSIZE` points, whatever
``max_memory`` and the process's memory are. It replaces
``mf._numint.block_loop`` on the INSTANCE (PySCF's own ``nr_rks``,
``nr_uks``, ``get_rho`` and ``cache_xc_kernel`` all resolve ``ni.block_loop``
through the instance), so it also reaches the second-order (``newton``) and
density-fitting wrappers PySCF builds from the object -- both copy the
object's ``__dict__`` and therefore share the same integrator.
:func:`pin_eri_path` replaces ``mf._is_mem_enough`` on the instance by the
fixed rule of :func:`eri_path_for_nao` (the same tensor estimate against
:data:`REFERENCE_ERI_INCORE_MB`), so the integral path is a function of the
system alone; the pinned predicate also refuses, loudly, to answer for a
molecule of a different size than it was pinned for (``mf.reset`` to
another system must re-pin a fresh object, not inherit a stale decision).
On a density-fitted object it instead holds ``with_df.blockdim`` at
:data:`REFERENCE_DF_AUX_BLOCKDIM` (240, PySCF's own default) and
``with_df.max_memory`` at a sentinel, so the auxiliary loops run at the
blockdim whatever the live memory, and the fitted tensor is built in
memory in one pass. That build is a real memory request: ``cholesky_eri``
holds the tensor (``naux * nao (nao + 1) / 2 * 8`` bytes; 11.4 MB for CH4,
353.6 MB for C5H8, the largest pool species) plus two scratch buffers of
the same size in the single pass the sentinel forces -- a 1060.7 MB
allocation with a measured peak of +700 MB resident at C5H8 and +231 MB at
acetic, where an unpinned build under a starved budget would have spilled
to disk at +25 MB. Forced incore is kept deliberately: one code path with
one bitwise proof, paid on the reference-generation stage whose CCSD step
dwarfs it. The decision is stamped as ``"df-aux240"``, and the pin is
independent of ordering against ``density_fit()``: pinning first and
wrapping afterwards leaves an inherited non-DF stamp that the next pin
supersedes rather than refuses. :func:`pin_reference_scf` applies the
applicable pins and reports them with the thread count.

The pins hold their owner only through weak references, so a pinned
object is freed by refcounting exactly as an unpinned one is; a closure
that held the mean-field strongly turned every build-and-discard loop --
one inner SCF per OEP objective evaluation -- into an accumulator that
only the cycle collector drained, measured and removed.

A pinned object is no longer picklable: the pins are instance-level
closures, so ``pickle.dumps`` fails loudly rather than silently shedding
them (``copy.deepcopy`` and ``mf.copy()`` preserve the pins; no reference
path pickles a mean-field).

Block-size bound. One block holds ``comp * blksize * nao`` doubles of AO
values (``comp`` = 1 for values, 4 with the gradient the GGA and meta-GGA
paths use, 10 with second derivatives, which no production path requests)
plus one ``blksize * nao`` scratch table, i.e. ``5 * blksize * nao * 8``
bytes on the production (GGA / meta-GGA) path. With ``blksize`` = 12544 that
is 0.50 MB per basis function: 35 MB at nao = 69 (H2O, 6-311++G(3df,2pd)),
39 MB at nao = 78 (N2), 50 MB at nao = 99 (CH4), and 158 MB at nao = 315,
the largest species of the BH76 and W4-11 pools at that basis (C5H8 and the
RKT22 transition state, 13 atoms), so the block stays under 200 MB per
worker for every species of the training and benchmark pools. Every small
system is one block -- the pruned level-3 grid of the O atom at def2-svp
has 11904 points, the level-1 grid of H2O 9304 -- so its summation order is
the one a clean process already had (the closed-shell fixture of the
spin-scaling oracles was recorded that way); H2O at def2-svp / level 3
(30632 points) takes 3 blocks. The one-block statement is a def2-svp
statement: at the production basis, 6-311++G(3df,2pd), even a bare atom
exceeds one block (the O atom's pruned level-3 grid is 13504 points, two
blocks), so every production reference -- the atoms included -- shifts
once, at the 1e-13 level, when the pin first lands, and is held fixed
thereafter; the v6 campaign regenerates every production reference under
the pin. The other pruned level-3 production grids run from 26616 points
(N2, 3 blocks) through 49408 (CH4, 4 blocks) to 131584 (C5H8, 11 blocks),
where PySCF's own loop, given the memory, takes 1 to 2 blocks of up to
67200 points. Measured wall of the reference SCF (31 alternated repeats,
medians, four threads; the pruning pass runs on the UNPRUNED grid, so the
O atom's 14088 unpruned points split 12544 + 1544 under the pin where the
post-pruning SCF loop is one block either way): 0.055 s to 0.056 s on the
O atom at def2-svp / grid 3, with an independent measurement on a loaded
box reaching +15 percent on the fastest repeats; 0.127 s to 0.134 s on
H2O at def2-svp / grid 3 (three blocks against one, per-block call
overhead on a 24-function system); faster at the production identity,
0.342 s to 0.291 s on H2O and 0.555 s to 0.474 s on CH4, where the
smaller blocks fit the caches better than PySCF's whole-grid block.

What the pins do not fix. PySCF's threaded reductions are not associative:
at more than one OpenMP thread the reference SCF is not bit-reproducible
even run to run in one process (measured: two consecutive records of the O
atom in one process at four threads differ in the density matrix, the grid
columns and E_pbe; at one thread, and at one PySCF thread with a four-thread
BLAS, they agree bit for bit). The pins hold the summation ORDER fixed; the
thread count is the caller's (the cluster job scripts export
``OMP_NUM_THREADS``), and every record carries the count it was produced at
(``reference_blas_threads``) beside the block size
(``reference_xc_blksize``) and the integral path (``reference_eri_path``)
so a mismatch is visible. (Before the auxiliary loops were pinned, the
density-fitted exchange loop's memory dependence moved the HF density of
the O atom by 4.2e-15 and the CCSD density on it by 4.8e-15 between a
clean process and one above the ceiling -- the scale that keeps the
stamps out of the CCSD cache identity below.)

Cache identities. None of the stamps enters a cache identity: not the
reference cache of :mod:`external_refs` (a CCSD reference is hours per
species at the production identity, and a 1e-13 change in the Hartree-Fock
seed does not move a CCSD density above its own convergence floor), not
the pretraining data manifest, not the precompute memo. They are metadata:
a consumer that needs bitwise agreement compares them; every other consumer
is inside the tolerances that already held both orders.
"""
from __future__ import annotations

import inspect
import weakref
from typing import NamedTuple

from pyscf import lib
from pyscf.dft.gen_grid import BLKSIZE

__all__ = [
    "BLKSIZE",
    "REFERENCE_DF_AUX_BLOCKDIM",
    "REFERENCE_ERI_INCORE_MB",
    "REFERENCE_XC_BLKSIZE",
    "ReferencePins",
    "eri_path_for_nao",
    "pin_eri_path",
    "pin_xc_block_size",
    "pin_reference_scf",
    "pinned_xc_block_size",
    "reference_thread_count",
]

#: Grid points per block of the reference SCFs' exchange-correlation
#: quadrature: 224 * BLKSIZE (56) = 12544. A multiple of BLKSIZE, as
#: ``NumInt.block_loop`` asserts (the non-zero AO screening table is indexed
#: per BLKSIZE-aligned sub-block). See the module docstring for the bound.
REFERENCE_XC_BLKSIZE: int = 224 * BLKSIZE

# The parameters of pyscf.dft.numint.NumInt.block_loop (pyscf 2.11) the
# wrapper below forwards by keyword. A block_loop that takes neither these
# names nor arbitrary keywords (an instrumenting wrapper of the form
# ``(self, *args, **kwargs)`` passes them through) is refused rather than
# silently left unpinned.
_BLOCK_LOOP_PARAMETERS = frozenset(("mol", "grids", "nao", "deriv",
                                    "max_memory", "non0tab", "blksize",
                                    "buf"))

#: Attribute the pin leaves on the integrator, so a second pin of the same
#: object is a no-op and the applied value can be read back.
_PIN_ATTR = "_xcquinox_xc_blksize"

#: Budget (MB) below which the two-electron integrals of a reference SCF are
#: held in memory; above it every Coulomb and exchange build is direct
#: (integral-screened, incremental). PySCF's own rule is
#: ``nao**4 / 1e6 + lib.current_memory() < 0.95 * max_memory`` (the packed
#: integral tensor in MB against the memory the process has left), so the
#: choice, and with it the J/K summation order at the 1e-13 level, follows
#: the process's memory history. Here the same tensor estimate is compared
#: against a fixed budget: 2000 MB, PySCF's default in-memory budget for an
#: integral tensor (``pyscf.df.incore.MAX_MEMORY``) and half its 4000 MB
#: process ceiling. Incore up to nao = 211 (a 1.98 GB packed tensor per
#: worker), direct above; at the production basis the pools' species reach
#: nao = 315 (a 9.8 GB tensor, direct here and under PySCF's rule).
REFERENCE_ERI_INCORE_MB: float = 2000.0

#: Attribute the ERI pin leaves on the mean-field.
_ERI_PIN_ATTR = "_xcquinox_eri_path"

#: Cholesky-vector (auxiliary-function) block of a density-fitted reference
#: object's J/K loops: PySCF's own default blockdim
#: (``__config__.df_df_DF_blockdim``, 240), enforced explicitly so the
#: stamp names the value that ran.
REFERENCE_DF_AUX_BLOCKDIM: int = 240

#: Sentinel ceiling (MB) held on a density-fitted reference object so the
#: memory-derived bound of its auxiliary loops never undercuts the blockdim
#: and its fitted-tensor build is in-memory and single-pass. The forced
#: in-memory build is a real resource request -- the fitted tensor plus two
#: same-size scratch buffers, 1060.7 MB allocated and +700 MB peak resident
#: at the largest pool species (module docstring) -- accepted for one code
#: path with one bitwise proof.
_DF_PINNED_MAX_MEMORY_MB: float = 1e9


class ReferencePins(NamedTuple):
    """What a reference SCF was pinned at, for the record's metadata."""
    #: Grid points per quadrature block, or None for a mean-field with no
    #: numerical integrator (a Hartree-Fock object has no grid quadrature).
    xc_blksize: int | None
    #: PySCF's OpenMP worker count at pin time (``lib.num_threads()``).
    threads: int
    #: How the two-electron integrals are built: "incore" (held in memory),
    #: "direct" (screened, rebuilt each cycle), or "df" (density fitted; the
    #: incore/direct predicate is never consulted).
    eri_path: str


def reference_thread_count() -> int:
    """PySCF's OpenMP worker count, the count its compiled kernels reduce
    over (what ``OMP_NUM_THREADS`` sets; ``lib.num_threads(n)`` changes it)."""
    return int(lib.num_threads())


def pinned_xc_block_size(mf) -> int | None:
    """The block size ``mf``'s integrator was pinned at, or None."""
    ni = getattr(mf, "_numint", None)
    if ni is None:
        return None
    value = getattr(ni, _PIN_ATTR, None)
    return None if value is None else int(value)


def pin_xc_block_size(mf, blksize: int = REFERENCE_XC_BLKSIZE) -> int | None:
    """Fix the grid block size of ``mf``'s XC quadrature at ``blksize`` points.

    Every ``block_loop`` call on ``mf._numint`` that does not name a block
    size explicitly (no caller on the SCF path does; PySCF's callers pass
    ``max_memory``) runs at ``blksize`` regardless of ``max_memory`` and of
    the process's memory. A caller that passes ``blksize`` itself is
    honoured, because such a caller has sized its own ``buf`` for it.

    Returns the block size applied, or None when ``mf`` has no ``_numint``
    (Hartree-Fock: no grid quadrature, nothing to pin). Pinning an already
    pinned integrator at the same value is a no-op; at a different value it
    is refused (``ValueError``), since two pins on one object would make the
    recorded value wrong for one of them. ``blksize`` must be a positive
    multiple of ``BLKSIZE``.
    """
    ni = getattr(mf, "_numint", None)
    if ni is None:
        return None
    pinned = int(blksize)
    if pinned <= 0 or pinned % BLKSIZE:
        raise ValueError(
            f"blksize must be a positive multiple of pyscf's BLKSIZE "
            f"({BLKSIZE}), got {blksize!r}")
    already = getattr(ni, _PIN_ATTR, None)
    if already is not None:
        if int(already) != pinned:
            raise ValueError(
                f"this integrator is already pinned at {int(already)} grid "
                f"points per block; refusing to re-pin it at {pinned}")
        return pinned
    unpinned = type(ni).block_loop
    parameters = inspect.signature(unpinned).parameters
    passes_keywords = any(p.kind is inspect.Parameter.VAR_KEYWORD
                          for p in parameters.values())
    if not passes_keywords and not _BLOCK_LOOP_PARAMETERS.issubset(parameters):
        raise RuntimeError(
            "pyscf's NumInt.block_loop does not take the parameters this pin "
            f"forwards: expected {sorted(_BLOCK_LOOP_PARAMETERS)}, found "
            f"{list(parameters)}; update pin_xc_block_size before relying "
            "on it")

    owner = weakref.ref(ni)

    def block_loop(mol, grids, nao=None, deriv=0, max_memory=2000,
                   non0tab=None, blksize=None, buf=None):
        integrator = owner()
        if integrator is None:
            raise RuntimeError(
                "this pinned block_loop outlived the integrator it was "
                "installed on; pin a live mean-field instead of keeping "
                "the bare closure")
        if blksize is None:
            blksize = pinned
        return unpinned(integrator, mol=mol, grids=grids, nao=nao,
                        deriv=deriv, max_memory=max_memory, non0tab=non0tab,
                        blksize=blksize, buf=buf)

    block_loop.__doc__ = (
        f"NumInt.block_loop pinned at {pinned} grid points per block "
        f"(xcquinox.alec.pyscf_determinism.pin_xc_block_size).")
    ni.block_loop = block_loop
    setattr(ni, _PIN_ATTR, pinned)
    return pinned


def eri_path_for_nao(nao: int,
                     incore_budget_mb: float = REFERENCE_ERI_INCORE_MB) -> str:
    """Return "incore" when PySCF's packed-tensor estimate ``nao**4 / 1e6``
    MB fits the budget, else "direct": the estimate ``SCF._is_mem_enough``
    uses, without its ``lib.current_memory()`` term."""
    return "incore" if int(nao) ** 4 / 1e6 < float(incore_budget_mb) \
        else "direct"


def pin_eri_path(mf, incore_budget_mb: float = REFERENCE_ERI_INCORE_MB) -> str:
    """Fix how ``mf`` builds its two-electron integrals, independently of
    the process's memory.

    PySCF's ``get_jk`` keeps the packed integral tensor in memory when
    ``mf._is_mem_enough()`` -- ``nao**4 / 1e6 + lib.current_memory() <
    0.95 * max_memory`` -- and otherwise builds J and K directly from
    screened integrals, incrementally over the DIIS iterates; the two
    paths agree to round-off and differ at the 1e-13 level, and the choice
    follows the process's memory history. The predicate is replaced on the
    instance by the fixed rule of :func:`eri_path_for_nao`, so the choice
    is a function of the system alone; because the decision is derived
    from the molecule at pin time, the pinned predicate refuses (loudly,
    ``RuntimeError``) to answer after ``mf.reset`` to a molecule of a
    different size -- re-pin a fresh object instead. A density-fitted
    object builds J and K from its fitted tensor and never consults the
    predicate; there the auxiliary loops are pinned instead:
    ``with_df.blockdim`` is held at :data:`REFERENCE_DF_AUX_BLOCKDIM` and
    ``with_df.max_memory`` at a sentinel, so the Coulomb and exchange
    sums run over fixed 240-vector blocks and the fitted-tensor build is
    in-memory and single-pass whatever the live memory; recorded as
    ``"df-aux240"``. (``with_df.reset`` restores neither, but no reference
    path resets a mean-field; the non-DF branch's guard is the loud
    tripwire for that class of misuse.)

    Returns the path applied. The second-order and density-fitting
    wrappers copy the object's ``__dict__``, so they share the pin.
    """
    with_df = getattr(mf, "with_df", None)
    already = getattr(mf, _ERI_PIN_ATTR, None)
    if with_df is not None:
        path = f"df-aux{REFERENCE_DF_AUX_BLOCKDIM}"
        if (already is not None and already != path
                and already.startswith("df-")):
            raise ValueError(
                f"this mean-field's integral path is already pinned to "
                f"{already!r}; refusing to re-pin it to {path!r}")
        # A non-DF stamp here was inherited from a pin taken BEFORE
        # density_fit() (the wrapper copies __dict__): superseded, since
        # the object is density-fitted now and never consults the
        # inherited predicate. The assignments below also run on an
        # idempotent re-pin, restoring blockdim and the sentinel if
        # something changed them in between.
        with_df.blockdim = REFERENCE_DF_AUX_BLOCKDIM
        with_df.max_memory = _DF_PINNED_MAX_MEMORY_MB
        setattr(mf, _ERI_PIN_ATTR, path)
        return path
    path = eri_path_for_nao(mf.mol.nao, incore_budget_mb)
    if already is not None:
        if already != path:
            raise ValueError(
                f"this mean-field's integral path is already pinned to "
                f"{already!r}; refusing to re-pin it to {path!r}")
        return path
    incore = path == "incore"
    nao_at_pin = int(mf.mol.nao)
    owner = weakref.ref(mf)

    def _pinned_is_mem_enough():
        obj = owner()
        if obj is None:
            # The pinned owner is gone; a surviving __dict__ copy (a
            # second-order or density-fitting wrapper) still gets the
            # pinned decision, which is a pure function of the nao it was
            # derived from.
            return incore
        if int(obj.mol.nao) != nao_at_pin:
            raise RuntimeError(
                f"the integral path of this mean-field was pinned for "
                f"nao={nao_at_pin} but its molecule now has "
                f"nao={int(obj.mol.nao)} (reset to a different system?); "
                "build and pin a fresh mean-field instead of reusing this "
                "one")
        return incore

    mf._is_mem_enough = _pinned_is_mem_enough
    setattr(mf, _ERI_PIN_ATTR, path)
    return path


def pin_reference_scf(mf) -> ReferencePins:
    """Pin a reference mean-field object and report what it runs at.

    Applies :func:`pin_xc_block_size` (a no-op returning None on a
    Hartree-Fock object) and :func:`pin_eri_path`, and reads the thread
    count, so the caller can put all three into the record's metadata.
    """
    return ReferencePins(xc_blksize=pin_xc_block_size(mf),
                         threads=reference_thread_count(),
                         eri_path=pin_eri_path(mf))
