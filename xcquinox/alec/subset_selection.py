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
