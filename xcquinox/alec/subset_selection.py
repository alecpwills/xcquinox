"""Step-7 histogram-matched subset selection from Dick 2021 training pool.

This module ports the legacy data_binning2.ipynb cell-17 algorithm into
the alec subpackage and extends it with a Jensen-Shannon divergence
metric in addition to the original Euclidean L2-on-bins metric.

Three-descriptor objective: histograms over (ρ^{1/3}, s, α) where
- s is the PBE-1996 reduced gradient (Perdew, Burke, Ernzerhof, PRL 77, 3865, 1996)
- α is the SCAN-2015 iso-orbital indicator (Sun, Ruzsinszky, Perdew,
  PRL 115, 036402, 2015, eq. 4); used for subset selection only,
  NOT consumed by the trained GGA network.

Candidate pool is Dick & Fernandez-Serra 2021 SI §II training data:
21 G2/97 atomization-energy entries + 3 BH76 reactions + 2 IP13 IPs
+ 2 atomic-density references = 28 distinct training points. Selection
varies the 21 AE entries; auxiliaries are fixed per Dick's protocol.

Public API:
- extract_descriptors(atoms_obj, *, basis="def2-svp", grid_level=1, cache_dir)
- build_reference_histograms(descriptor_arrays, *, nbins=200)
- metric_l2(h_ref, h_cand) -> float       # 3-histogram sum
- metric_jsd(h_ref, h_cand) -> float      # 3-histogram sum, nats
- select_subset(pool, r, metric, fixed_indices=())
- compute_atom_set(ae_subset_atoms_list)
- augment_with_hbpt(ae_subset_atoms_list, atom_refs, *, with_hbpt: bool)
"""
from __future__ import annotations

import os
import json
from pathlib import Path
from itertools import combinations
from typing import Callable, Iterable

import numpy as np
import ase
from ase import Atoms
from ase.io import read, write

# Constants ---------------------------------------------------------------
NBINS = 200
LOG_REGULARIZER = 1e-10
KL_PROB_CLIP = 1e-12

# HB and PT water-dimer geometries verbatim from
# /home/awills/Documents/Research/Python/jup/data_binning2.ipynb cell 20.
# Original at.info: basis='6-311++G(3df,2pd)', grid_level=4. Step-7
# overrides these to def2-svp / grid_level=1 to keep histograms commensurate
# with the rest of the candidate pool.
_HB_POSITIONS = (
    (1.317021, -0.128356, 0.006258),
    (1.527437, 0.387478, -0.795622),
    (1.505382, 0.474880, 0.750724),
    (-1.017021, 0.128356, 0.006258),
    (-1.227437, -0.387478, -0.795622),
    (-1.205382, -0.474880, 0.750724),
)
_PT_POSITIONS = (
    (1.310944, -0.092374, 0.053983),
    (1.955110, 0.571413, -0.263648),
    (-0.101366, 0.045774, -0.012031),
    (-1.149037, 0.029559, -0.084434),
    (-1.608104, 0.722348, 0.414070),
    (-1.540923, -0.836961, 0.105186),
)
_HB_SYMBOLS = "OHHOHH"
_PT_SYMBOLS = "OHHOHH"


def compute_descriptor_triple(
    rho: np.ndarray,
    sigma: np.ndarray,
    tau: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compute (ρ^{1/3}, s, α) from SCF outputs on the integration grid.

    Parameters
    ----------
    rho : (N,) electron density on grid points
    sigma : (N,) |∇ρ|² (gradient squared)
    tau : (N,) kinetic-energy density

    Returns
    -------
    dict with keys "rho_third", "s", "alpha", each (N,) ndarray. α is
    clipped at 0 to handle grid noise in low-density tails.

    Formulas:
    - s = |∇ρ| / [2 (3π²)^{1/3} ρ^{4/3}]   (PBE 1996, before eq. 12)
    - τ_W = |∇ρ|²/(8ρ),  τ_unif = (3/10)(3π²)^{2/3} ρ^{5/3}
    - α = (τ - τ_W) / τ_unif               (SCAN 2015, eq. 4)
    """
    rho_safe = np.maximum(rho, 1e-30)
    grad_rho = np.sqrt(np.maximum(sigma, 0.0))
    rho_third = rho_safe ** (1.0 / 3.0)
    kf_factor = 2.0 * (3.0 * np.pi**2) ** (1.0 / 3.0)
    s = grad_rho / (kf_factor * rho_safe ** (4.0 / 3.0))
    tau_w = sigma / (8.0 * rho_safe)
    tau_unif = (
        (3.0 / 10.0) * (3.0 * np.pi**2) ** (2.0 / 3.0) * rho_safe ** (5.0 / 3.0)
    )
    alpha = np.maximum((tau - tau_w) / np.maximum(tau_unif, 1e-30), 0.0)
    return {"rho_third": rho_third, "s": s, "alpha": alpha}


_DESCRIPTOR_KEYS = ("rho_third", "s", "alpha")


def metric_l2(h_ref: dict, h_cand: dict) -> float:
    """Per-bin Euclidean distance summed across the 3 marginals.

    err = sum_b sqrt( (h^ref_rho - h^cand_rho)^2_b
                    + (h^ref_s   - h^cand_s)^2_b
                    + (h^ref_a   - h^cand_a)^2_b )

    This is the verbatim form from data_binning2.ipynb cell 17.
    """
    diffs_sq = np.zeros(NBINS)
    for k in _DESCRIPTOR_KEYS:
        diffs_sq += (h_ref[k] - h_cand[k]) ** 2
    return float(np.sum(np.sqrt(diffs_sq)))


def _kl(p: np.ndarray, q: np.ndarray) -> float:
    """Kullback-Leibler divergence in nats. Probabilities clipped at KL_PROB_CLIP."""
    p_c = np.clip(p, KL_PROB_CLIP, 1.0)
    q_c = np.clip(q, KL_PROB_CLIP, 1.0)
    return float(np.sum(p_c * (np.log(p_c) - np.log(q_c))))


def metric_jsd(h_ref: dict, h_cand: dict) -> float:
    """Jensen-Shannon divergence summed across the 3 marginals (nats).

    JSD(P||Q) = 0.5 [ KL(P||M) + KL(Q||M) ],   M = (P+Q)/2.

    Reference: Lin, IEEE Trans. Inf. Theory 37 (1991) eq. (4.1).

    NOTE: do NOT use scipy.spatial.distance.jensenshannon — that returns
    the JS distance (sqrt of the divergence), not the divergence itself.
    """
    total = 0.0
    for k in _DESCRIPTOR_KEYS:
        p = h_ref[k]
        q = h_cand[k]
        m = 0.5 * (p + q)
        total += 0.5 * (_kl(p, m) + _kl(q, m))
    return float(total)


def _bin_with_edges(arrs: dict, edges: dict) -> dict:
    """Bin descriptors using fixed pre-computed log10 edges; returns 3 histograms."""
    out = {}
    w = arrs.get("weights")
    for k in _DESCRIPTOR_KEYS:
        log_x = np.log10(arrs[k] + LOG_REGULARIZER)
        h, _ = np.histogram(log_x, bins=edges[k], weights=w, density=True)
        out[k] = h
    return out


def bin_descriptors(arrs: dict) -> dict:
    """Bin a single descriptor-array set with auto-computed log10 edges."""
    edges = {}
    for k in _DESCRIPTOR_KEYS:
        log_x = np.log10(arrs[k] + LOG_REGULARIZER)
        lo, hi = np.percentile(log_x, [0.1, 99.9])
        edges[k] = np.linspace(lo, hi, NBINS + 1)
    return _bin_with_edges(arrs, edges)


def build_reference_histograms(pool):
    """Concatenate descriptor arrays across the full candidate pool, build
    the 3 reference 200-bin log10 density-normalized histograms, and return
    the edges used so that candidate-subset histograms align."""
    cat: dict = {k: [] for k in _DESCRIPTOR_KEYS}
    cat_w: list = []
    for arrs in pool:
        for k in _DESCRIPTOR_KEYS:
            cat[k].append(arrs[k])
        cat_w.append(arrs.get("weights", np.ones_like(arrs["rho_third"])))
    full = {k: np.concatenate(cat[k]) for k in _DESCRIPTOR_KEYS}
    full["weights"] = np.concatenate(cat_w)
    edges = {}
    for k in _DESCRIPTOR_KEYS:
        log_x = np.log10(full[k] + LOG_REGULARIZER)
        lo, hi = np.percentile(log_x, [0.1, 99.9])
        edges[k] = np.linspace(lo, hi, NBINS + 1)
    h_ref = _bin_with_edges(full, edges)
    return h_ref, edges


def select_subset(
    pool,
    edges: dict,
    h_ref: dict,
    *,
    r: int,
    metric: str,
    fixed_indices: tuple = (),
):
    """Exhaustively enumerate all C(npool, r) subsets and return the
    indices of the size-r combination that minimizes the chosen metric.

    Parameters
    ----------
    pool : list of per-species descriptor-array dicts
    edges : pre-built bin edges from build_reference_histograms
    h_ref : reference 3-histogram tuple from build_reference_histograms
    r : target subset size
    metric : "l2" or "jsd"
    fixed_indices : pool indices that must be present in every candidate
        subset. The chosen subset has exactly r entries TOTAL including
        the fixed ones.
    """
    if metric == "l2":
        m = metric_l2
    elif metric == "jsd":
        m = metric_jsd
    else:
        raise ValueError(f"unknown metric: {metric!r}")

    npool = len(pool)
    if r > npool:
        raise ValueError(f"r={r} exceeds pool size {npool}")
    if r < len(fixed_indices):
        raise ValueError(f"r={r} smaller than fixed_indices count {len(fixed_indices)}")

    fixed_set = set(fixed_indices)
    free_indices = [i for i in range(npool) if i not in fixed_set]
    free_r = r - len(fixed_indices)

    best_val = float("inf")
    best_combo: tuple = ()
    for combo in combinations(free_indices, free_r):
        full = tuple(sorted(set(combo) | fixed_set))
        cat = {k: np.concatenate([pool[i][k] for i in full]) for k in _DESCRIPTOR_KEYS}
        cat["weights"] = np.concatenate(
            [pool[i].get("weights", np.ones_like(pool[i]["rho_third"])) for i in full]
        )
        h_cand = _bin_with_edges(cat, edges)
        v = m(h_ref, h_cand)
        if v < best_val:
            best_val = v
            best_combo = full
    return best_combo, best_val


def compute_atom_set(ae_subset) -> set:
    """Return the union of chemical-element symbols across the given AE
    molecules. Used to determine which atomic-energy references must
    appear in `subset.traj` for total-energy regularization (§5c)."""
    out: set = set()
    for a in ae_subset:
        out.update(a.get_chemical_symbols())
    return out


def _make_hb_atoms() -> Atoms:
    """H-bonded water-dimer reference. Geometry from data_binning2.ipynb cell 20.
    Basis/grid-level overrides applied here: def2-svp / grid_level=1, NOT the
    legacy 6-311++G(3df,2pd) / level=4."""
    a = Atoms(_HB_SYMBOLS, positions=list(_HB_POSITIONS))
    a.info.update({
        "charge": 1, "spin": 1, "name": "HBWD", "openshell": True,
        "sc": False, "sym": False, "reaction": "reactant",
        "grid_level": 1, "basis": "def2-svp", "pol": True,
    })
    return a


def _make_pt_atoms() -> Atoms:
    """Proton-transfer water-dimer reference. Geometry from data_binning2.ipynb cell 20.
    Basis/grid override: def2-svp / grid_level=1."""
    a = Atoms(_PT_SYMBOLS, positions=list(_PT_POSITIONS))
    a.info.update({
        "charge": 1, "spin": 1, "name": "PTWD", "openshell": True,
        "sc": False, "sym": False, "reaction": 1,
        "grid_level": 1, "basis": "def2-svp", "pol": True,
    })
    return a


def augment_with_hbpt(
    ae_subset,
    atom_refs,
    *,
    with_hbpt: bool,
):
    """Build the final list of Atoms objects to write to subset.traj.

    Composition: AE-subset entries + atomic-reference entries + (optionally)
    the HB and PT water-dimer pair. Atom-refs are added unconditionally and
    are determined by `compute_atom_set` upstream. HBPT geometries are
    overridden to def2-svp / grid_level=1 so descriptor histograms remain
    commensurate with the rest of the candidate pool."""
    out = list(ae_subset) + list(atom_refs)
    if with_hbpt:
        out.append(_make_hb_atoms())
        out.append(_make_pt_atoms())
    return out


def _ase_atoms_to_pyscf_mol(at: Atoms, *, basis: str, charge: int = 0, spin: int = 0):
    """Build a PySCF gto.M from an ASE Atoms object. Mirrors the inline
    builder in xcquinox/alec/data.py:precompute_fixed_density_data line 237."""
    from pyscf import gto

    coords = at.get_positions()
    atom_lines = [
        (sym, tuple(coords[i])) for i, sym in enumerate(at.get_chemical_symbols())
    ]
    return gto.M(
        atom=atom_lines,
        basis=basis,
        charge=int(at.info.get("charge", charge)),
        spin=int(at.info.get("spin", spin)),
        unit="angstrom",
        verbose=0,
    )


def extract_descriptors(
    at: Atoms,
    *,
    idx: int,
    cache_dir,
    basis: str = "def2-svp",
    grid_level: int = 1,
) -> dict:
    """Run a single PBE SCF and extract (ρ^{1/3}, s, α) on the molecular grid.

    Returns dict with keys rho_third, s, alpha, weights. Caches as
    <cache_dir>/<idx>_<species>.npz; second call hits the cache.

    Conventions match step-5/step-6 (def2-svp, grid_level=1) per
    _build_step6_notebook.py:528-532.
    """
    from pyscf import dft

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    species = at.info.get("species", at.get_chemical_formula())
    safe = species.replace("/", "_").replace(" ", "_")
    cache_path = cache_dir / f"{idx}_{safe}.npz"
    if cache_path.exists():
        z = np.load(cache_path)
        return {k: z[k] for k in ("rho_third", "s", "alpha", "weights")}

    mol = _ase_atoms_to_pyscf_mol(at, basis=basis)
    is_uhf = (mol.spin or 0) != 0
    if is_uhf:
        mf = dft.UKS(mol, xc="PBE,PBE")
    else:
        mf = dft.RKS(mol, xc="PBE,PBE")
    mf.grids.level = grid_level
    mf.grids.build()
    mf.kernel()
    dm = mf.make_rdm1()
    ao = mf._numint.eval_ao(mol, mf.grids.coords, deriv=2)
    if is_uhf:
        dm_total = dm[0] + dm[1]
    else:
        dm_total = dm
    rho_full = mf._numint.eval_rho(mol, ao, dm_total, xctype="MGGA")
    # rho_full shape (6, ngrid): [rho, rho_x, rho_y, rho_z, lapl, tau]
    rho = rho_full[0]
    sigma = rho_full[1] ** 2 + rho_full[2] ** 2 + rho_full[3] ** 2
    tau = rho_full[5]
    descriptors = compute_descriptor_triple(rho, sigma, tau)
    weights = mf.grids.weights
    out = {
        "rho_third": descriptors["rho_third"],
        "s": descriptors["s"],
        "alpha": descriptors["alpha"],
        "weights": weights,
    }
    np.savez(cache_path, **out)
    return out
