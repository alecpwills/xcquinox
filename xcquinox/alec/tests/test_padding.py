"""Shape-padding key table (:mod:`xcquinox.alec.padding`): which mol_data arrays
the pass pads, and how.

The neutrality of the pass end to end (energy, per-channel loss, model gradient)
is pinned in test_shape_padding.py against the unpadded de-fuse. What is pinned
here is narrower and structural: ``_PAD_AO_ZERO_BLOCK`` -- the (spin?, n_ao,
n_ao) matrices padded with a zero block -- and the two places it is consumed,
the ``n_ao_unpadded`` scan (which takes the FIRST present key) and the pad loop
itself. A key absent from that tuple is silently carried through unpadded, which
at a common pad target is a shape mismatch inside the per-molecule kernel rather
than a clean error, so the membership is worth asserting directly.
"""
import numpy as np
import jax.numpy as jnp

from xcquinox.alec.padding import _PAD_AO_ZERO_BLOCK, _pad_mol_data, PadTarget


def test_pad_mol_data_pads_the_dm_minao_seed_like_the_pbe_seed():
    """``dm_minao`` (the atomic seed carried beside ``dm_seed`` for the
    per-update seed mixture) is an AO matrix of exactly ``dm_seed``'s shape, so
    it pads with the same zero block: real block kept, cross-blocks and padded
    diagonal zero. A molecule that does not carry it holds None, and the pass
    must leave the None alone rather than pad it -- ``present()`` is
    ``mol_data.get(k) is not None``, so a None value is skipped both by the pad
    loop and by the ``n_ao_unpadded`` scan that walks the same tuple.
    """
    assert "dm_minao" in _PAD_AO_ZERO_BLOCK
    target = PadTarget(n_ao=4, n_grid=3, naux=None)
    dm = jnp.arange(4.0).reshape(2, 2) + 1.0
    md = {"s_matrix": jnp.eye(2), "dm_pbe": dm, "dm_seed": dm,
          "dm_minao": 2.0 * dm, "grid_weights": jnp.ones(3)}
    out = _pad_mol_data(md, target)
    assert out["dm_minao"].shape == out["dm_seed"].shape == (4, 4)
    padded = np.asarray(out["dm_minao"])
    np.testing.assert_array_equal(padded[:2, :2], np.asarray(2.0 * dm))
    assert np.all(padded[2:, :] == 0.0) and np.all(padded[:, 2:] == 0.0)
    # The AO count recorded for the per-element loss normalizations is the
    # physical one, and is unchanged by the new key.
    assert float(out["n_ao_unpadded"]) == 2.0

    # A record built without the minao seed: the key is present and None, and
    # stays None through the pass.
    md_none = dict(md, dm_minao=None)
    out_none = _pad_mol_data(md_none, target)
    assert out_none["dm_minao"] is None
    assert float(out_none["n_ao_unpadded"]) == 2.0


def test_n_ao_unpadded_scan_tolerates_a_none_dm_minao():
    """The ``n_ao_unpadded`` scan takes the FIRST PRESENT key of
    ``("s_matrix", "h_core") + _PAD_AO_ZERO_BLOCK``, so where ``dm_minao`` sits
    in the tuple cannot change the recorded count: a None entry is skipped, and
    every present entry carries the same trailing AO dimension.
    """
    target = PadTarget(n_ao=5, n_grid=2, naux=None)
    # dm_minao alone, no overlap or core-H: the scan must still find an AO key.
    out = _pad_mol_data({"dm_minao": jnp.eye(3)}, target)
    assert float(out["n_ao_unpadded"]) == 3.0
    assert out["dm_minao"].shape == (5, 5)
    # A None dm_minao ahead of a present dm_seed must not be taken as the
    # AO-carrying key (it would make float(None.shape[-1]) raise).
    out2 = _pad_mol_data({"dm_minao": None, "dm_seed": jnp.eye(3)}, target)
    assert float(out2["n_ao_unpadded"]) == 3.0
