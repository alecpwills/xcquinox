"""Pretrain/eval cusp must use the SAME log_transform as training.

Every cusp-using architecture in the registry (deep_cusp, deep_cusp_attn,
deep_combined, deep_combined_attn) sets descriptor_log_transform=True, and the
training pipeline (data.py) computes the cusp with that flag. Two other code
paths computed the cusp with the default (False) — the saturating
``tanh(weighted_Z/5)`` form — silently feeding a DIFFERENT feature distribution:

  * pretrain_data_gen.py  (pretraining input)
  * solver_pyscfad.py     (pyscfad-backend held-out eval input)

These tests pin that both now produce the bounded ``tanh(log(weighted_Z)/5)``
form, matching training.
"""
import numpy as np
import jax.numpy as jnp
import pytest
from pyscf import gto

from xcquinox.alec import pretrain_data_gen as pdg
from xcquinox.features import compute_cusp_descriptor


# A near-nucleus grid point makes the WRONG (default-False) path saturate to
# exactly 1.0, so these tests fail loudly if the skew regresses.
_NEAR = np.array([[0.001, 0.0, 0.0], [0.5, 0.0, 0.0], [2.0, 0.0, 0.0]])


def test_pretrain_atom_cusp_uses_bounded_log_form():
    """The per-atom pretrain cusp column must be the bounded log form (<1),
    not the saturating raw form (=1), so it matches the cusp-using archs."""
    cols = pdg._atom_columns("O", 2, "def2-svp", 1, polarized=False,
                             descriptors=True)
    cusp = np.asarray(cols["cusp"])
    assert cusp.shape[1] == 2
    # log form stays strictly < 1 even with O's near-nucleus grid points;
    # the raw default-False form would hit exactly 1.0 there.
    assert cusp[:, 1].max() < 1.0


def test_pretrain_cusp_log_transform_flag_threads():
    """cusp_log_transform=False reproduces the old saturating behavior, so the
    flag genuinely controls the path (and defaults to the bounded form)."""
    on = pdg._atom_columns("O", 2, "def2-svp", 1, polarized=False,
                           descriptors=True)
    off = pdg._atom_columns("O", 2, "def2-svp", 1, polarized=False,
                            descriptors=True, cusp_log_transform=False)
    assert np.asarray(on["cusp"])[:, 1].max() < 1.0
    assert np.asarray(off["cusp"])[:, 1].max() >= 1.0 - 1e-9


def test_pyscfad_reassemble_cusp_honors_log_transform():
    """_reassemble_features_on_grid must read the CuspDescriptor's log_transform
    (like data.py) rather than the saturating default."""
    from xcquinox.alec.solver_pyscfad import _reassemble_features_on_grid
    from xcquinox.alec.descriptors import CuspDescriptor

    mol = gto.M(atom="O 0 0 0", basis="sto-3g", spin=2, verbose=0)
    gc = jnp.asarray(_NEAR)
    nao = mol.nao_nr()
    dm = jnp.zeros((2, nao, nao))           # cusp ignores dm; shape only
    s = jnp.asarray(mol.intor("int1e_ovlp"))

    cols = _reassemble_features_on_grid(
        (CuspDescriptor(log_transform=True),), dm, s, gc, mol)

    expected = compute_cusp_descriptor(
        gc, jnp.asarray(mol.atom_coords()), jnp.asarray(mol.atom_charges()),
        log_transform=True)
    np.testing.assert_allclose(np.asarray(cols), np.asarray(expected),
                               rtol=0, atol=1e-6)
    assert np.asarray(cols)[:, 1].max() < 1.0    # bounded, not saturated
