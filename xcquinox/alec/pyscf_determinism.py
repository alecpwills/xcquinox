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
system alone. :func:`pin_reference_scf` applies both and reports them with
the thread count.

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
(30632 points) takes 3 blocks. At the production identity the pruned
level-3 grids run from 26616 points (N2, 3 blocks) through 49408 (CH4, 4
blocks) to 131584 (C5H8, 11 blocks), where PySCF's own loop, given the
memory, takes 1 to 2 blocks of up to 67200 points. Measured wall of the
reference SCF (fifteen repeats, medians, four threads): unchanged on the O
atom at def2-svp / grid 3 (one block either way, 0.059 s to 0.058 s);
0.131 s to 0.145 s on H2O at def2-svp / grid 3 (three blocks against one;
the fastest repeats equal at 0.122 s, the cost is per-block call overhead
on a 24-function system); faster at the production identity, 0.357 s to
0.339 s on H2O and 0.564 s to 0.460 s on CH4, where the smaller blocks fit
the caches better than PySCF's whole-grid block.

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
so a mismatch is visible. The exchange loop of a density-fitted
Hartree-Fock object (the HF-for-CCSD with ``density_fit``) is sized from
process memory as well and is left as PySCF builds it: it moved the HF
density of the O atom by 4.2e-15 and the CCSD density on it by 4.8e-15
between a clean process and one above the ceiling, below any consumer's
tolerance and far below the CCSD convergence floor.

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
from typing import NamedTuple

from pyscf import lib
from pyscf.dft.gen_grid import BLKSIZE

__all__ = [
    "BLKSIZE",
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

    def block_loop(mol, grids, nao=None, deriv=0, max_memory=2000,
                   non0tab=None, blksize=None, buf=None):
        if blksize is None:
            blksize = pinned
        return unpinned(ni, mol=mol, grids=grids, nao=nao, deriv=deriv,
                        max_memory=max_memory, non0tab=non0tab,
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
    is a function of the system alone. A density-fitted object builds J
    and K from its fitted tensor and never consults the predicate; it is
    left alone and reported as "df".

    Returns the path applied. The second-order and density-fitting
    wrappers copy the object's ``__dict__``, so they share the pin.
    """
    if getattr(mf, "with_df", None) is not None:
        return "df"
    already = getattr(mf, _ERI_PIN_ATTR, None)
    path = eri_path_for_nao(mf.mol.nao, incore_budget_mb)
    if already is not None:
        if already != path:
            raise ValueError(
                f"this mean-field's integral path is already pinned to "
                f"{already!r}; refusing to re-pin it to {path!r}")
        return path
    incore = path == "incore"
    mf._is_mem_enough = lambda: incore
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
