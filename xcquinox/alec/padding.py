"""Self-contained shape-padding pass for the de-fused per-molecule training step.

Purpose
-------
The de-fused training step compiles one XLA kernel per distinct per-molecule JIT
signature and retains them all in a never-evicted process-global cache. For the
DFS-parity basis/grid (6-311++G(3df,2pd), grid 3) with an attention network, each
such kernel is enormous, and a full training subset holds ~26 of them resident,
exhausting the process ``mmap`` ceiling (``vm.max_map_count``) at COMPILE time
(see HISTORY). This module makes molecules that share a spin-type and occupation
compile ONE kernel instead of one-per-molecule.

Design
------
This is a pure, results-neutral INPUT pass -- ``pad_batch(batch, target)`` returns
a transformed batch; nothing in the solver / energy / loss code changes. The
per-molecule JIT keys on the whole ``mol_data`` pytree, so a molecule recompiles
for any of three reasons; the pass neutralizes the first two and leaves the third:

1. **Array shape** (n_ao / n_grid / naux differ) -> pad every shape-carrying array
   up to one common ``PadTarget`` shape, with masks that keep results identical:
   padded grid points carry integration weight 0, and the padded AO block is
   decoupled (overlap = I, Fock diagonal huge) so its orbitals sort last, never
   occupy (occupation is the stored scalar ``nocc``, not an array shape), and its
   density block is exactly zero every SCF cycle.
2. **Molecule-identifying static leaves** the energy kernel never reads
   (``_pyscfad_mol`` -- a pyscfad ``Mole`` whose basis-shaped array leaves would
   otherwise force a per-molecule recompile -- plus ``name`` / ``atom_composition``
   / identifying ``mol_metadata`` scalars) -> strip them. Verified energy-neutral.
   Precomputed scalar PBE energies (Python floats -> distinct static leaves) are
   converted to traced 0-d arrays so their VALUES stop keying the compile.
3. **Occupation** (``nocc`` / ``nocc_a`` / ``nocc_b`` -- Python ints baked into the
   kernel) -> traced to 0-d arrays so their VALUES stop keying the compile. This is
   the one field whose consumption needs a paired change: the manual SCF drops its
   ``int()`` casts so the ``arange(nao) < nocc`` occupation mask accepts a traced
   value (``solver_manual.py``; ``oneshot.py`` already reads it raw). Molecules then
   collapse to one kernel per SPIN-TYPE -> two total (RKS + UKS, genuinely different
   code paths).

Every padded field is analytically invariant under the masks above. The one
historical exception, ``dm_statistics.dm_entropy`` (a clip artifact on padded
zero-occupation eigenvalues), was removed 2026-08-06; both surviving
``dm_statistics`` features are exactly invariant.
"""
from typing import NamedTuple, Optional

import jax.numpy as jnp

_PAD_ORBITAL_ENERGY = 1.0e6  # padded-orbital Fock diagonal; >> real virtuals (~1e2)


class PadTarget(NamedTuple):
    n_ao: int
    n_grid: int
    naux: Optional[int]


def common_pad_target(mol_data_iterable) -> PadTarget:
    """Element-wise max ``(n_ao, n_grid, naux)`` over a set of ``mol_data`` dicts
    -- the single common shape every molecule is padded up to so they share one
    JIT signature. ``naux`` is ``None`` when no molecule carries a DF ``cderi``."""
    mds = list(mol_data_iterable)
    n_ao = max(int(md["s_matrix"].shape[-1]) for md in mds)
    n_grid = max(int(md["grid_weights"].shape[0]) for md in mds)
    nauxs = [int(md["cderi"].shape[0]) for md in mds
             if md.get("cderi") is not None]
    return PadTarget(n_ao=n_ao, n_grid=n_grid,
                     naux=(max(nauxs) if nauxs else None))


# --- array padders ---------------------------------------------------------
def _pad_ao_block(mat, n_ao_t, diag):
    """Pad a trailing ``(n_ao, n_ao)`` block (optionally spin-leading) to
    ``(n_ao_t, n_ao_t)``: real block kept, cross-blocks 0, padded diagonal = diag."""
    n = mat.shape[-1]
    p = n_ao_t - n
    if p <= 0:
        return mat
    mat = jnp.pad(mat, [(0, 0)] * (mat.ndim - 2) + [(0, p), (0, p)])
    if diag != 0.0:
        idx = jnp.arange(n, n_ao_t)
        eye_pad = jnp.zeros((n_ao_t, n_ao_t), dtype=mat.dtype).at[idx, idx].set(diag)
        mat = mat + eye_pad  # broadcasts over any spin-leading axis
    return mat


def _pad_grid(x, n_grid_t, mode):
    """Pad a leading grid axis to ``n_grid_t``. ``mode='zero'`` for the weight
    mask (padded weight 0); ``mode='edge'`` replicates a real row so padded-point
    network inputs stay finite (they are weight-0, hence neutral)."""
    p = n_grid_t - x.shape[0]
    if p <= 0:
        return x
    width = [(0, p)] + [(0, 0)] * (x.ndim - 1)
    return jnp.pad(x, width) if mode == "zero" else jnp.pad(x, width, mode="edge")


def _pad_ao_on_grid(x, n_grid_t, n_ao_t, grid_axis, ao_axis):
    """Zero-pad the AO axis (padded AO columns 0 -> zero V_xc/density contribution
    even where unweighted) and edge-pad the grid axis (finite, weight-0 rows)."""
    pao = n_ao_t - x.shape[ao_axis]
    if pao > 0:
        w = [(0, 0)] * x.ndim
        w[ao_axis] = (0, pao)
        x = jnp.pad(x, w)
    pg = n_grid_t - x.shape[grid_axis]
    if pg > 0:
        w = [(0, 0)] * x.ndim
        w[grid_axis] = (0, pg)
        x = jnp.pad(x, w, mode="edge")
    return x


# (spin?, n_ao, n_ao) matrices padded with a zero block
_PAD_AO_ZERO_BLOCK = ("dm_pbe", "j_matrix", "vxc_pbe", "dm_target", "vxc_ref")
# grid-only fields holding FINITE per-point data (edge-padded, weight-0 rows)
_PAD_GRID_EDGE = ("rho_grid", "sigma_grid", "nabla_rho_grid", "rho_ref_grid",
                  "cusp_features", "dm_features", "rung35_features",
                  "rung35ms_features", "metagga_features")
# (n_grid, n_ao): edge-pad grid axis 0, zero-pad AO axis 1
_PAD_AO_ON_GRID = ("ao_grid", "rung35_proj_ao")

# Molecule-identifying leaves the manual-backend energy never reads (verified
# energy-neutral); stripping them stops them keying the per-molecule compile.
_STRIP_KEYS = ("_pyscfad_mol", "name", "atom_composition")
# Precomputed scalar PBE energies stored as Python floats -> distinct static
# leaves; traced so their values stop keying the compile.
_TRACE_SCALARS = ("E_pbe", "E_xc_pbe", "E_non_xc", "e_nuc")
# Occupation counts (Python ints -> the kernel is keyed by electron count). Traced
# so molecules of one spin-type share a kernel; paired with dropping int(nocc) in
# solver_manual.py (the arange<nocc mask already accepts a traced value).
_TRACE_OCCUPATION = ("nocc", "nocc_a", "nocc_b")
# The only mol_metadata entry the SCF consumes as data (an AO matrix, padded).
_META_KEEP = ("orientation_lock_bias",)


def _pad_mol_data(mol_data, target: PadTarget):
    """Pad every shape-carrying array in ``mol_data`` to ``target``'s common shape
    with the results-neutral masks (see module docstring)."""
    n_ao_t, n_grid_t, naux_t = target.n_ao, target.n_grid, target.naux
    out = dict(mol_data)

    def present(k):
        return mol_data.get(k) is not None

    # AO matrices: overlap (padded diag 1) and core H (padded diag huge) build the
    # decoupled padded block; every other AO matrix pads with a zero block.
    if present("s_matrix"):
        out["s_matrix"] = _pad_ao_block(mol_data["s_matrix"], n_ao_t, 1.0)
    if present("h_core"):
        out["h_core"] = _pad_ao_block(mol_data["h_core"], n_ao_t,
                                      _PAD_ORBITAL_ENERGY)
    for k in _PAD_AO_ZERO_BLOCK:
        if present(k):
            out[k] = _pad_ao_block(mol_data[k], n_ao_t, 0.0)

    # Grid-only fields: the weight mask is zero-padded; finite per-point data is
    # edge-padded.
    if present("grid_weights"):
        out["grid_weights"] = _pad_grid(mol_data["grid_weights"], n_grid_t, "zero")
    for k in _PAD_GRID_EDGE:
        if present(k):
            out[k] = _pad_grid(mol_data[k], n_grid_t, "edge")

    # AO-on-grid tensors.
    for k in _PAD_AO_ON_GRID:
        if present(k):
            out[k] = _pad_ao_on_grid(mol_data[k], n_grid_t, n_ao_t,
                                     grid_axis=0, ao_axis=1)
    # (n_alpha, n_grid, n_ao): the multi-width projected-AO STACK. It must NOT
    # go in _PAD_AO_ON_GRID, which is consumed with grid_axis=0, ao_axis=1 --
    # wrong for a 3-D tensor, and measured to return (640, 500, 7) instead of
    # (3, 640, 13), an ~90x element blow-up that becomes an OOM at production
    # grid size rather than a clean error.
    if present("rung35ms_proj_ao"):
        out["rung35ms_proj_ao"] = _pad_ao_on_grid(
            mol_data["rung35ms_proj_ao"], n_grid_t, n_ao_t,
            grid_axis=1, ao_axis=2)
    if present("ao_grid_deriv"):  # (4, n_grid, n_ao)
        out["ao_grid_deriv"] = _pad_ao_on_grid(
            mol_data["ao_grid_deriv"], n_grid_t, n_ao_t, grid_axis=1, ao_axis=2)

    # Two-electron tensors: zero-pad every padded AO index (and DF aux rows), so
    # the assembled J's real block is exact and its padded block stays zero.
    if present("eri"):  # (n_ao,)*4
        e = mol_data["eri"]
        p = n_ao_t - e.shape[-1]
        if p > 0:
            out["eri"] = jnp.pad(e, [(0, p)] * 4)
    if present("cderi"):  # (naux, n_ao, n_ao)
        c = mol_data["cderi"]
        pa = n_ao_t - c.shape[-1]
        if pa > 0:
            c = jnp.pad(c, [(0, 0), (0, pa), (0, pa)])
        if naux_t is not None:
            pn = naux_t - c.shape[0]
            if pn > 0:
                c = jnp.pad(c, [(0, pn), (0, 0), (0, 0)])
        out["cderi"] = c

    # Orientation-lock h_core bias (array inside the otherwise-static metadata).
    meta = mol_data.get("mol_metadata")
    if isinstance(meta, dict) and meta.get("orientation_lock_bias") is not None:
        out["mol_metadata"] = {
            **meta,
            "orientation_lock_bias": _pad_ao_block(
                jnp.asarray(meta["orientation_lock_bias"]), n_ao_t, 0.0),
        }

    return out


def canonicalize_mol_data(mol_data, target: PadTarget):
    """The full pass for one molecule: pad arrays, strip the identifying leaves the
    energy kernel does not read, reduce ``mol_metadata`` to the AO matrix the SCF
    uses, and trace the static scalar energies. Results-neutral; see module docstring."""
    out = _pad_mol_data(mol_data, target)

    for k in _STRIP_KEYS:
        out.pop(k, None)

    meta = out.get("mol_metadata")
    if isinstance(meta, dict):
        kept = {k: meta[k] for k in _META_KEEP if meta.get(k) is not None}
        if kept:
            out["mol_metadata"] = kept
        else:
            out.pop("mol_metadata", None)

    # Static Python scalars (precomputed energies + occupation counts) -> traced
    # 0-d arrays, so their values stop keying the per-molecule compile. Values
    # already stored as arrays (numpy/jax) are dynamic already and left untouched.
    for k in _TRACE_SCALARS + _TRACE_OCCUPATION:
        v = out.get(k)
        if v is not None and not hasattr(v, "shape"):
            out[k] = jnp.asarray(float(v))

    return out


def pad_batch(batch, target: PadTarget):
    """The independent pass-through: return ``batch`` with every molecule in
    ``batch['mol_data']`` canonicalized to ``target``. Everything else (targets,
    atom_energies, ...) is carried through unchanged."""
    mol_data = tuple(canonicalize_mol_data(md, target) for md in batch["mol_data"])
    return {**batch, "mol_data": mol_data}
