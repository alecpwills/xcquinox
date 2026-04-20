"""
Feature extraction utilities for extended GGA neural network functionals.

This module provides functions for computing additional features beyond
the standard GGA descriptors (rho, sigma), including:
- Density matrix features (correlation indicators)
- Cusp-related quantities (nuclear position information)
- Extended local descriptors (Laplacian-based)

These features can be used to improve the accuracy of neural network
XC functionals, particularly for capturing correlation effects that
are difficult to represent with local/semi-local descriptors alone.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Tuple, Optional, Union
from functools import partial


# =============================================================================
# Density Matrix Features
# =============================================================================

def compute_dm_features(dm: jnp.ndarray, S: jnp.ndarray) -> Dict[str, float]:
    """
    Extract correlation-sensitive features from the density matrix.

    These features capture information about electron correlation that is
    not available from the density alone. They can serve as global modifiers
    or be broadcast to grid points as additional network inputs.

    :param dm: Density matrix in AO basis, shape (nao, nao)
    :type dm: jnp.ndarray
    :param S: Overlap matrix in AO basis, shape (nao, nao)
    :type S: jnp.ndarray
    :return: Dictionary of density matrix features:
        - 'idempotency_error': Tr(DM - DM @ S @ DM) / Tr(DM), measures deviation from HF
        - 'dm_entropy': -Tr(DM/N * log(DM/N + eps)), von Neumann-like entropy
        - 'off_diag_norm': Frobenius norm of off-diagonal elements / trace
        - 'trace': Tr(DM @ S) = number of electrons
    :rtype: Dict[str, float]
    """
    # Handle UKS case: sum alpha and beta density matrices
    if dm.ndim == 3:
        dm = dm[0] + dm[1]

    # Number of electrons
    n_elec = jnp.trace(dm @ S)

    # Idempotency error: for HF, DM @ S @ DM = DM
    # Deviation indicates correlation
    dm_s_dm = dm @ S @ dm
    idempotency_error = jnp.trace(dm - dm_s_dm) / (n_elec + 1e-12)

    # Compute natural orbital occupations via generalized eigenvalue problem
    # DM @ S @ C = C @ diag(n_i) where n_i are occupations
    # Use Lowdin orthogonalization: S^{-1/2} @ DM @ S^{1/2}
    S_eigvals, S_eigvecs = jnp.linalg.eigh(S)
    S_sqrt = S_eigvecs @ jnp.diag(jnp.sqrt(jnp.maximum(S_eigvals, 1e-12))) @ S_eigvecs.T
    S_inv_sqrt = S_eigvecs @ jnp.diag(1.0 / jnp.sqrt(jnp.maximum(S_eigvals, 1e-12))) @ S_eigvecs.T

    dm_ortho = S_inv_sqrt @ dm @ S_inv_sqrt.T
    occupations = jnp.linalg.eigvalsh(dm_ortho)
    occupations = jnp.clip(occupations, 1e-12, 2.0)  # Physical bounds

    # Occupation entropy: -sum(n_i * log(n_i)) normalized
    # High entropy indicates strong correlation (fractional occupations)
    occ_normalized = occupations / (jnp.sum(occupations) + 1e-12)
    dm_entropy = -jnp.sum(occ_normalized * jnp.log(occ_normalized + 1e-12))

    # Off-diagonal norm (correlation indicator)
    diag_dm = jnp.diag(jnp.diag(dm))
    off_diag = dm - diag_dm
    off_diag_norm = jnp.linalg.norm(off_diag, 'fro') / (jnp.trace(dm) + 1e-12)

    return {
        'idempotency_error': idempotency_error,
        'dm_entropy': dm_entropy,
        'off_diag_norm': off_diag_norm,
        'trace': n_elec,
    }


def compute_dm_features_array(dm: jnp.ndarray, S: jnp.ndarray) -> jnp.ndarray:
    """
    Compute density matrix features as a JAX array for use in networks.

    Returns features in a fixed order suitable for concatenation with
    grid-point descriptors.

    :param dm: Density matrix in AO basis
    :type dm: jnp.ndarray
    :param S: Overlap matrix
    :type S: jnp.ndarray
    :return: Array of shape (3,) containing [idempotency_error, dm_entropy, off_diag_norm]
    :rtype: jnp.ndarray
    """
    features = compute_dm_features(dm, S)
    return jnp.array([
        features['idempotency_error'],
        features['dm_entropy'],
        features['off_diag_norm'],
    ])


# =============================================================================
# Cusp-Related Features
# =============================================================================

def compute_cusp_distances(grid_coords: jnp.ndarray,
                           nuclear_coords: jnp.ndarray,
                           nuclear_charges: jnp.ndarray) -> Dict[str, jnp.ndarray]:
    """
    Compute cusp-related features at each grid point.

    The electron-nuclear cusp condition requires special behavior of the
    density and XC functional near nuclear positions. These features
    provide the network with information about nuclear proximity.

    :param grid_coords: Grid point coordinates, shape (N, 3)
    :type grid_coords: jnp.ndarray
    :param nuclear_coords: Nuclear positions, shape (M, 3)
    :type nuclear_coords: jnp.ndarray
    :param nuclear_charges: Nuclear charges Z, shape (M,)
    :type nuclear_charges: jnp.ndarray
    :return: Dictionary containing:
        - 'r_min': Distance to nearest nucleus, shape (N,)
        - 'Z_nearest': Charge of nearest nucleus, shape (N,)
        - 'cusp_factor': exp(-2 * Z_nearest * r_min), cusp decay factor, shape (N,)
        - 'weighted_Z_sum': sum_A Z_A / r_A, Coulomb-like weighting, shape (N,)
    :rtype: Dict[str, jnp.ndarray]
    """
    # Compute distances from each grid point to each nucleus
    # grid_coords: (N, 3), nuclear_coords: (M, 3)
    # distances: (N, M)
    diff = grid_coords[:, None, :] - nuclear_coords[None, :, :]  # (N, M, 3)
    distances = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-12)  # (N, M)

    # Distance to nearest nucleus
    r_min = jnp.min(distances, axis=1)  # (N,)
    nearest_idx = jnp.argmin(distances, axis=1)  # (N,)
    Z_nearest = nuclear_charges[nearest_idx]  # (N,)

    # Cusp decay factor: exponential decay from nearest nucleus
    # This is related to the Slater-type cusp: psi ~ exp(-Z*r)
    cusp_factor = jnp.exp(-2 * Z_nearest * r_min)

    # Weighted sum of nuclear contributions (Coulomb-like)
    weighted_Z_sum = jnp.sum(nuclear_charges[None, :] / distances, axis=1)

    return {
        'r_min': r_min,
        'Z_nearest': Z_nearest,
        'cusp_factor': cusp_factor,
        'weighted_Z_sum': weighted_Z_sum,
    }


def compute_cusp_descriptor(grid_coords: jnp.ndarray,
                            nuclear_coords: jnp.ndarray,
                            nuclear_charges: jnp.ndarray) -> jnp.ndarray:
    """
    Compute a compact cusp descriptor for each grid point.

    Returns a single descriptor combining nuclear proximity information
    in a form suitable for network input.

    Each column is bounded in a network-friendly range:
    * column 0 ``cusp_factor = exp(-2 Z r)`` lives in [0, 1] natively.
    * column 1 ``tanh(log_weighted_Z / 5)`` lives in (-1, 1). The raw
      ``log_weighted_Z = log(sum_A Z_A / r_A)`` has a dynamic range of
      ~14 units on physical grids (tail values ~ -2, near-nucleus values
      ~ 12), which previously dominated the MLP's first-layer activation
      for the exchange network and caused F_x predictions to saturate at
      ~1.4 on architectures using this descriptor (deep_cusp,
      deep_cusp_attn, deep_combined, deep_combined_attn). Dividing by 5
      before tanh keeps the transform ~linear for the typical
      ``log_weighted_Z`` range (~0..3) while smoothly compressing extreme
      near-nucleus values.

    :param grid_coords: Grid point coordinates, shape (N, 3)
    :type grid_coords: jnp.ndarray
    :param nuclear_coords: Nuclear positions, shape (M, 3)
    :type nuclear_coords: jnp.ndarray
    :param nuclear_charges: Nuclear charges, shape (M,)
    :type nuclear_charges: jnp.ndarray
    :return: Cusp descriptors, shape (N, 2) containing
        [cusp_factor, tanh(log_weighted_Z / 5)] — both in a bounded
        range suitable for direct MLP input.
    :rtype: jnp.ndarray
    """
    features = compute_cusp_distances(grid_coords, nuclear_coords, nuclear_charges)

    # Bounded form of the Coulomb-like weighted_Z feature. Division by 5
    # chosen so that typical mid-range log_weighted_Z values (~0-3) remain
    # in the ~linear region of tanh, while near-nucleus outliers
    # (log_weighted_Z >> 5) smoothly saturate at +1 rather than entering
    # the MLP as large unnormalized features.
    log_weighted_Z = jnp.log(features['weighted_Z_sum'] + 1e-12)
    log_weighted_Z_bounded = jnp.tanh(log_weighted_Z / 5.0)

    return jnp.stack([features['cusp_factor'], log_weighted_Z_bounded], axis=1)


# =============================================================================
# Extended Local Descriptors
# =============================================================================

def compute_reduced_laplacian(rho: jnp.ndarray,
                               laplacian: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the reduced Laplacian descriptor q.

    The reduced Laplacian is defined as:
        q = nabla^2 rho / (4 * k_F^2 * rho)

    where k_F = (3 * pi^2 * rho)^(1/3) is the Fermi wavevector.

    This dimensionless quantity characterizes the curvature of the density
    and is useful for distinguishing bonding regions from lone pairs.

    :param rho: Electron density, shape (N,)
    :type rho: jnp.ndarray
    :param laplacian: Laplacian of density nabla^2 rho, shape (N,)
    :type laplacian: jnp.ndarray
    :return: Reduced Laplacian q, shape (N,)
    :rtype: jnp.ndarray
    """
    # Fermi wavevector
    k_F = (3 * jnp.pi**2 * jnp.maximum(rho, 1e-12))**(1/3)

    # Reduced Laplacian
    q = laplacian / (4 * k_F**2 * jnp.maximum(rho, 1e-12))

    return q


def compute_elf_descriptor(rho: jnp.ndarray,
                           sigma: jnp.ndarray,
                           tau: jnp.ndarray) -> jnp.ndarray:
    """
    Compute the Electron Localization Function (ELF) descriptor.

    ELF measures the probability of finding an electron near a reference
    electron with the same spin. Values near 1 indicate localized electrons
    (bonds, lone pairs), values near 0.5 indicate electron gas-like behavior.

    ELF = 1 / (1 + (D/D_h)^2)

    where D = tau - tau_W is the Pauli kinetic energy density and
    D_h = (3/5) * (6*pi^2)^(2/3) * rho^(5/3) is the uniform electron gas value.

    :param rho: Electron density, shape (N,)
    :type rho: jnp.ndarray
    :param sigma: Squared gradient |nabla rho|^2, shape (N,)
    :type sigma: jnp.ndarray
    :param tau: Kinetic energy density, shape (N,)
    :type tau: jnp.ndarray
    :return: ELF values, shape (N,)
    :rtype: jnp.ndarray
    """
    # Von Weizsacker kinetic energy density
    tau_W = sigma / (8 * jnp.maximum(rho, 1e-12))

    # Pauli kinetic energy density
    D = jnp.maximum(tau - tau_W, 0.0)

    # Uniform electron gas kinetic energy density
    D_h = (3/5) * (6 * jnp.pi**2)**(2/3) * jnp.maximum(rho, 1e-12)**(5/3)

    # ELF
    chi = D / (D_h + 1e-12)
    elf = 1.0 / (1.0 + chi**2)

    return elf


def compute_extended_descriptors(rho: jnp.ndarray,
                                  sigma: jnp.ndarray,
                                  laplacian: Optional[jnp.ndarray] = None,
                                  tau: Optional[jnp.ndarray] = None,
                                  grid_coords: Optional[jnp.ndarray] = None,
                                  nuclear_coords: Optional[jnp.ndarray] = None,
                                  nuclear_charges: Optional[jnp.ndarray] = None,
                                  include_laplacian: bool = False,
                                  include_cusp: bool = False) -> jnp.ndarray:
    """
    Compute extended descriptors beyond standard GGA.

    This function assembles additional descriptors that can be appended
    to the standard GGA inputs [rho, sigma] based on what information
    is available and requested.

    :param rho: Electron density, shape (N,)
    :param sigma: Squared gradient, shape (N,)
    :param laplacian: Laplacian of density, shape (N,), optional
    :param tau: Kinetic energy density, shape (N,), optional
    :param grid_coords: Grid coordinates, shape (N, 3), optional
    :param nuclear_coords: Nuclear positions, shape (M, 3), optional
    :param nuclear_charges: Nuclear charges, shape (M,), optional
    :param include_laplacian: Whether to include reduced Laplacian
    :param include_cusp: Whether to include cusp descriptors
    :return: Extended descriptors array
    :rtype: jnp.ndarray
    """
    descriptors = []

    # Reduced Laplacian
    if include_laplacian and laplacian is not None:
        q = compute_reduced_laplacian(rho, laplacian)
        descriptors.append(q[:, None])

    # Cusp descriptors
    if include_cusp and grid_coords is not None and nuclear_coords is not None:
        cusp_desc = compute_cusp_descriptor(grid_coords, nuclear_coords, nuclear_charges)
        descriptors.append(cusp_desc)

    if len(descriptors) > 0:
        return jnp.concatenate(descriptors, axis=1)
    else:
        return jnp.zeros((rho.shape[0], 0))


# =============================================================================
# Cusp Correction Layer
# =============================================================================

class CuspCorrection:
    """
    Cusp correction module for XC energy density.

    This class implements a multiplicative correction to the XC energy
    density that enforces proper cusp behavior near nuclear positions.

    The correction has the form:
        exc_corrected = exc * (1 + amplitude * cusp_factor)

    where cusp_factor = exp(-2 * Z * r) decays away from nuclei.
    """

    def __init__(self, amplitude: float = 0.1):
        """
        Initialize cusp correction.

        :param amplitude: Strength of the cusp correction, defaults to 0.1
        :type amplitude: float
        """
        self.amplitude = amplitude

    def __call__(self,
                 exc: jnp.ndarray,
                 grid_coords: jnp.ndarray,
                 nuclear_coords: jnp.ndarray,
                 nuclear_charges: jnp.ndarray) -> jnp.ndarray:
        """
        Apply cusp correction to XC energy density.

        :param exc: XC energy density, shape (N,)
        :param grid_coords: Grid coordinates, shape (N, 3)
        :param nuclear_coords: Nuclear positions, shape (M, 3)
        :param nuclear_charges: Nuclear charges, shape (M,)
        :return: Corrected XC energy density
        """
        cusp_features = compute_cusp_distances(grid_coords, nuclear_coords, nuclear_charges)
        correction = 1.0 + self.amplitude * cusp_features['cusp_factor']
        return exc * correction


# =============================================================================
# Feature Normalization Utilities
# =============================================================================

def normalize_dm_features(features: jnp.ndarray,
                          means: Optional[jnp.ndarray] = None,
                          stds: Optional[jnp.ndarray] = None) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Normalize density matrix features for network input.

    :param features: DM features array
    :param means: Pre-computed means for normalization, or None to compute
    :param stds: Pre-computed stds for normalization, or None to compute
    :return: (normalized_features, means, stds)
    """
    if means is None:
        means = jnp.mean(features, axis=0)
    if stds is None:
        stds = jnp.std(features, axis=0) + 1e-8

    normalized = (features - means) / stds
    return normalized, means, stds


def safe_log_transform(x: jnp.ndarray, eps: float = 1e-12) -> jnp.ndarray:
    """
    Safe logarithmic transform for positive quantities.

    :param x: Input array (should be positive)
    :param eps: Small constant to prevent log(0)
    :return: log(x + eps)
    """
    return jnp.log(jnp.maximum(x, eps))


def inverse_transform(y: jnp.ndarray, transform: str = 'log') -> jnp.ndarray:
    """
    Inverse of common transforms.

    :param y: Transformed values
    :param transform: Transform type ('log', 'tanh', 'none')
    :return: Original values
    """
    if transform == 'log':
        return jnp.exp(y)
    elif transform == 'tanh':
        return jnp.arctanh(jnp.clip(y, -0.999, 0.999))
    else:
        return y
