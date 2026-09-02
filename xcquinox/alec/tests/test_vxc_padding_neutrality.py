"""Padding neutrality of the V_xc channel.

The shape-padding pass advertises itself as results-neutral (padding.py module
docstring), and it is for the energy and density channels. The V_xc channel's
absolute normalization divides by the AO count read off the (padded) reference
matrix, so padding a molecule from n_ao to n_ao_t rescales its loss_vxc by
(n_ao / n_ao_t)^2. At the production basis 6-311++G(3df,2pd) the DEFAULT
(reaction-energy) pool's AO counts span 15 (H, which carries both vxc_ref
and dm_target) to 117 (CO2/N2O/NO2/O3), a worst case of (15/117)^2 =
60.8x; the wired barrier mode adds the transition states (RKT11 at 138
AOs), raising it to (15/138)^2 = 84.6x. A 54-AO species padding to 117
(4.7x) is a mid-range example. These tests pin the neutrality contract:
the same molecule must contribute the same loss_vxc padded or not.

The NN-side V_xc is stubbed with the exact quadrature V_mn = sum_g w_g v_g
phi_m phi_n, which is itself padding-neutral (padded AO columns are zero,
padded grid rows carry zero weight), so any padded-vs-unpadded difference
isolates the denominator.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from xcquinox.alec import losses as losses_mod
from xcquinox.alec import padding


def _quadrature_vxc(model, rho, sigma, features, ao_grid, grid_weights,
                    nabla_rho=None, ao_grad=None):
    """Exact-quadrature stand-in for compute_vxc_nn: V = ao^T (w*v) ao with a
    fixed per-point potential v = rho. Padding-neutral by construction."""
    v = rho * grid_weights
    return jnp.einsum("g,gm,gn->mn", v, ao_grid, ao_grid)


def _mol_data(n_ao: int, n_grid: int, seed: int = 0) -> dict:
    rng = np.random.RandomState(seed)
    ao = rng.rand(n_grid, n_ao)
    return {
        "s_matrix": jnp.asarray(np.eye(n_ao)),
        "rho_grid": jnp.asarray(rng.rand(n_grid)),
        "sigma_grid": jnp.asarray(rng.rand(n_grid)),
        "ao_grid": jnp.asarray(ao),
        "grid_weights": jnp.asarray(rng.rand(n_grid)),
        "vxc_ref": jnp.asarray(rng.rand(n_ao, n_ao)),
    }


class _StubModel:
    descriptors = ()


@pytest.fixture()
def stubbed_vxc(monkeypatch):
    monkeypatch.setattr(losses_mod, "compute_vxc_nn", _quadrature_vxc)
    monkeypatch.setattr(losses_mod, "assemble_descriptor_features",
                        lambda descriptors, md, **kw: None)
    return _StubModel()


def _vxc_loss(model, md) -> float:
    return float(losses_mod._vxc_term(model, [md], iter_idx=(0,)))


def test_vxc_term_is_padding_neutral_rks(stubbed_vxc):
    """The defect: n_ao 2 padded to 5 rescaled loss_vxc by (2/5)^2 = 0.16."""
    md = _mol_data(n_ao=2, n_grid=7)
    target = padding.PadTarget(n_ao=5, n_grid=11, naux=None)
    padded = padding._pad_mol_data(md, target)

    unpadded_loss = _vxc_loss(stubbed_vxc, md)
    padded_loss = _vxc_loss(stubbed_vxc, padded)

    assert unpadded_loss > 0.0
    assert padded_loss == pytest.approx(unpadded_loss, rel=1e-12), (
        f"padding rescaled loss_vxc: {padded_loss} vs {unpadded_loss} "
        f"(ratio {padded_loss / unpadded_loss:.6f}; the defective "
        f"denominator gives (2/5)^2 = 0.16)")


def test_vxc_term_is_padding_neutral_uks(stubbed_vxc, monkeypatch):
    """Same contract on the UKS branch (denominator 2*n_ao^2)."""
    monkeypatch.setattr(
        losses_mod, "_uks_spin_resolved_vxc",
        lambda model, md, fa, fb, f: (
            _quadrature_vxc(model, md["rho_grid"], None, None,
                            md["ao_grid"], md["grid_weights"]),
            _quadrature_vxc(model, md["rho_grid"], None, None,
                            md["ao_grid"], md["grid_weights"]),
        ))
    md = _mol_data(n_ao=2, n_grid=7, seed=1)
    md["vxc_ref"] = jnp.asarray(
        np.random.RandomState(2).rand(2, 2, 2))
    target = padding.PadTarget(n_ao=5, n_grid=11, naux=None)
    padded = padding._pad_mol_data(md, target)

    unpadded_loss = _vxc_loss(stubbed_vxc, md)
    padded_loss = _vxc_loss(stubbed_vxc, padded)

    assert unpadded_loss > 0.0
    assert padded_loss == pytest.approx(unpadded_loss, rel=1e-12)


def test_pad_mol_data_records_true_ao_count():
    """The pad pass must record the pre-pad AO count so consumers can
    normalize by the physical size, not the padded one."""
    md = _mol_data(n_ao=3, n_grid=5)
    target = padding.PadTarget(n_ao=8, n_grid=9, naux=None)
    padded = padding._pad_mol_data(md, target)
    assert int(padded["n_ao_unpadded"]) == 3


def test_dm_term_is_padding_neutral(monkeypatch):
    """_dm_term normalizes by the element count of the (padded) dm_target;
    same defect class as _vxc_term, same neutrality contract."""
    monkeypatch.setattr(losses_mod, "dm_prediction_for_loss",
                        lambda model, md, solver_config=None:
                        0.5 * md["dm_target"])
    md = _mol_data(n_ao=2, n_grid=7, seed=3)
    md["dm_target"] = jnp.asarray(np.random.RandomState(4).rand(2, 2))
    target = padding.PadTarget(n_ao=5, n_grid=11, naux=None)
    padded = padding._pad_mol_data(md, target)

    model = _StubModel()
    unpadded_loss = float(losses_mod._dm_term(model, [md], iter_idx=(0,)))
    padded_loss = float(losses_mod._dm_term(model, [padded], iter_idx=(0,)))

    assert unpadded_loss > 0.0
    assert padded_loss == pytest.approx(unpadded_loss, rel=1e-12), (
        f"padding rescaled loss_dm: {padded_loss} vs {unpadded_loss} "
        f"(ratio {padded_loss / unpadded_loss:.6f})")


# ---------------------------------------------------------------------------
# Grid density channel: padding-neutral BY CONSTRUCTION (zero-padded
# quadrature weights), pinned here so the claim has an in-tree oracle.
# ---------------------------------------------------------------------------

def _grid_mol(n_grid, seed):
    rng = np.random.default_rng(seed)
    return {
        "rho_ref_grid": jnp.asarray(rng.uniform(0.1, 1.0, n_grid)),
        "grid_weights": jnp.asarray(rng.uniform(0.01, 0.1, n_grid)),
    }


@pytest.mark.parametrize("kwargs", [{}, {"relative": True},
                                    {"per_electron": True}])
def test_grid_term_is_padding_neutral(monkeypatch, kwargs):
    """_grid_term's error and normalizations are quadrature-weighted sums and
    the padder zero-pads grid_weights, so padding the grid axis must leave
    the loss bit-identical in all three normalization modes. The mutation
    control at the end proves the comparison can fail: edge-padding the
    WEIGHTS (nonzero rows in the padded tail) separates the two values."""
    monkeypatch.setattr(
        losses_mod, "grid_density_for_loss",
        lambda model, md, solver_config: md["rho_ref_grid"] * 1.1)
    mols = [_grid_mol(40, 0), _grid_mol(64, 1)]
    unpadded = float(losses_mod._grid_term(None, mols, range(2), **kwargs))

    target = padding.PadTarget(n_ao=1, n_grid=64, naux=None)
    padded = []
    for m in mols:
        out = dict(m)
        out["grid_weights"] = padding._pad_grid(
            m["grid_weights"], target.n_grid, "zero")
        out["rho_ref_grid"] = padding._pad_grid(
            m["rho_ref_grid"], target.n_grid, "edge")
        padded.append(out)
    padded_val = float(losses_mod._grid_term(None, padded, range(2), **kwargs))
    assert unpadded > 0.0
    assert padded_val == unpadded, (padded_val, unpadded)

    # Mutation control: edge-padded weights carry nonzero padded rows; the
    # padded tail then contributes and the values must separate.
    mutated = []
    for m in mols:
        out = dict(m)
        out["grid_weights"] = padding._pad_grid(
            m["grid_weights"], target.n_grid, "edge")
        out["rho_ref_grid"] = padding._pad_grid(
            m["rho_ref_grid"], target.n_grid, "edge")
        mutated.append(out)
    mutated_val = float(losses_mod._grid_term(None, mutated, range(2), **kwargs))
    assert mutated_val != unpadded, "mutation control failed to separate"
