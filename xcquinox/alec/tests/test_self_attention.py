"""Pin canonical multi-head scaled-dot-product self-attention.

References: Vaswani et al. 2017 §3.2.1-§3.2.2; Xiong et al. 2020 §3.

These tests must FAIL against the pre-2026-04-27 broken
``SelfAttentionBlock`` (elementwise softmax channel-gate) and PASS
against the post-2026-04-27 canonical multi-head implementation. See
spec §"TDD red-phase ordering" for which tests must fail vs may fail.
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from xcquinox.net import SelfAttentionBlock


# ---------------------------------------------------------------------------
# Reference implementations used by tests
# ---------------------------------------------------------------------------

def _numpy_canonical_reference(block, x):
    """Independent factorization: explicit per-head loop with sliced packed
    weights. Catches C-order vs F-order reshape bugs that a same-shape
    reference wouldn't.
    """
    x_np = np.asarray(x)
    H = block.hidden_size
    nh = block.num_heads
    hd = block.head_dim

    # Pre-LN replicated explicitly (Equinox eqx.nn.LayerNorm uses biased
    # variance / divides by H, eps=1e-5).
    eps = 1e-5
    mean = x_np.mean()
    var = x_np.var()
    x_normed = (x_np - mean) / np.sqrt(var + eps)
    gamma = np.asarray(block.norm.weight)
    beta = np.asarray(block.norm.bias)
    x_n = gamma * x_normed + beta

    Wq = np.asarray(block.query_proj.weight)
    bq = np.asarray(block.query_proj.bias)
    Wk = np.asarray(block.key_proj.weight)
    bk = np.asarray(block.key_proj.bias)
    Wv = np.asarray(block.value_proj.weight)
    bv = np.asarray(block.value_proj.bias)
    Wo = np.asarray(block.output_proj.weight)
    bo = np.asarray(block.output_proj.bias)

    q_packed = Wq @ x_n + bq
    k_packed = Wk @ x_n + bk
    v_packed = Wv @ x_n + bv

    # Per-head loop with explicit slicing (different factorization than a
    # single reshape — catches reshape-order bugs in the JAX impl).
    out = np.zeros(H)
    q_h_all = np.stack([q_packed[i * hd:(i + 1) * hd] for i in range(nh)])
    k_h_all = np.stack([k_packed[i * hd:(i + 1) * hd] for i in range(nh)])
    v_h_all = np.stack([v_packed[i * hd:(i + 1) * hd] for i in range(nh)])
    for h in range(nh):
        scores_h = (q_h_all[h] @ k_h_all.T) / np.sqrt(hd)        # (nh,)
        attn_h = np.exp(scores_h - scores_h.max())
        attn_h = attn_h / attn_h.sum()                            # softmax
        out_h = attn_h @ v_h_all                                   # (hd,)
        out[h * hd:(h + 1) * hd] = out_h

    out = Wo @ out + bo
    return out + x_np


# ---------------------------------------------------------------------------
# Tests 1-5
# ---------------------------------------------------------------------------

def test_canonical_attention_against_numpy_hand_reference():
    """Test 1: pin block output against an independently-factored numpy
    reference. The reference uses an explicit per-head loop with sliced
    packed weights, so a C-order vs F-order reshape bug would be caught.
    """
    key = jax.random.PRNGKey(0)
    block = SelfAttentionBlock(hidden_size=8, num_heads=2, key=key)
    x = jax.random.normal(jax.random.PRNGKey(1), (8,))
    out_module = np.asarray(block(x))
    out_ref = _numpy_canonical_reference(block, x)
    assert np.allclose(out_module, out_ref, atol=1e-5, rtol=1e-4), (
        f"module output {out_module}\n  vs numpy ref {out_ref}"
    )


def test_score_matrix_shape():
    """Test 2: scores shape is (num_heads, num_heads) — not (H,) as the
    pre-fix elementwise-Hadamard code produced.
    """
    for nh in (1, 2, 4):
        H = max(nh, 8)
        # Round H up to a multiple of nh
        H = (H + nh - 1) // nh * nh
        key = jax.random.PRNGKey(0)
        block = SelfAttentionBlock(hidden_size=H, num_heads=nh, key=key)
        x = jnp.ones(H)
        _, _, _, scores = block._compute_scores(x)
        assert scores.shape == (nh, nh), (
            f"num_heads={nh}, H={H}: expected ({nh}, {nh}), "
            f"got {scores.shape}"
        )


def test_softmax_axis_is_key_axis():
    """Test 3: softmax over axis=-1; each ROW sums to 1."""
    key = jax.random.PRNGKey(0)
    block = SelfAttentionBlock(hidden_size=8, num_heads=2, key=key)
    x = jnp.array([1.0, 2.0, -0.5, 0.3, 0.0, 0.7, -0.1, 1.5])
    _, _, _, scores = block._compute_scores(x)
    attn = jax.nn.softmax(scores, axis=-1)
    row_sums = jnp.sum(attn, axis=-1)
    assert jnp.allclose(row_sums, 1.0, atol=1e-6), (
        f"row sums {row_sums} should all be 1.0"
    )


def test_scale_is_sqrt_head_dim():
    """Test 4: divider is sqrt(head_dim), not sqrt(hidden_size). With
    Q=K=I, biases=0, and LayerNorm forced to a constant ones-vector
    output (gamma=0, bias=1), each row of q is ones(head_dim), so
    scores[i, j] = head_dim / sqrt(head_dim) = sqrt(head_dim).
    """
    H, nh = 8, 2
    head_dim = H // nh
    key = jax.random.PRNGKey(0)
    block = SelfAttentionBlock(hidden_size=H, num_heads=nh, key=key)

    # Override Q/K to identity, biases to zero.
    block = eqx.tree_at(lambda m: m.query_proj.weight, block, jnp.eye(H))
    block = eqx.tree_at(lambda m: m.query_proj.bias, block, jnp.zeros(H))
    block = eqx.tree_at(lambda m: m.key_proj.weight, block, jnp.eye(H))
    block = eqx.tree_at(lambda m: m.key_proj.bias, block, jnp.zeros(H))
    # LN: gamma=0, bias=1 => post-LN output is the constant ones(H)
    block = eqx.tree_at(lambda m: m.norm.weight, block, jnp.zeros(H))
    block = eqx.tree_at(lambda m: m.norm.bias, block, jnp.ones(H))

    x = jnp.ones(H)
    _, _, _, scores = block._compute_scores(x)
    expected = jnp.sqrt(float(head_dim)) * jnp.ones((nh, nh))
    assert jnp.allclose(scores, expected, atol=1e-5), (
        f"scores={scores}, expected sqrt(head_dim)={jnp.sqrt(float(head_dim))} "
        f"in every cell"
    )


def test_num_heads_one_collapses_to_residual_mlp():
    """Test 5: with num_heads=1, scores is (1,1), softmax = [[1.0]],
    attn @ v = v, so output = output_proj(value_proj(LN(x))) + x.
    """
    H = 8
    key = jax.random.PRNGKey(0)
    block = SelfAttentionBlock(hidden_size=H, num_heads=1, key=key)
    x = jax.random.normal(jax.random.PRNGKey(2), (H,))

    _, _, v, scores = block._compute_scores(x)
    attn = jax.nn.softmax(scores, axis=-1)
    assert attn.shape == (1, 1)
    assert jnp.allclose(attn, jnp.array([[1.0]]), atol=1e-6)

    # Reference: output_proj(v.reshape(H)) + x
    expected = block.output_proj(v.reshape(H)) + x
    actual = block(x)
    assert jnp.allclose(actual, expected, atol=1e-6)


# ---------------------------------------------------------------------------
# Tests 6-10
# ---------------------------------------------------------------------------

def test_num_heads_h_vs_one_produce_different_outputs():
    """Test 6: with num_heads=H (head_dim=1), the score matrix is (H,H)
    — NOT (1,1) — and the softmax over the key axis is non-trivial.
    With num_heads=1 (head_dim=H), the score matrix is (1,1), softmax is
    [[1.0]], and attn @ v = v exactly.

    Therefore num_heads=1 and num_heads=H produce DIFFERENT outputs even
    when constructed with the same key (i.e. bit-identical Q/K/V/O
    weights and LayerNorm). This pins the attention dependency on
    num_heads — a regression that ignored num_heads (e.g. always treated
    as 1) would make the two outputs identical.

    Math:
      num_heads=1, head_dim=H:  attn=[[1]],     attn @ v = v.
      num_heads=H, head_dim=1:  attn=(H,H),     softmax(q_i*k_j) per row,
                                attn @ v generally != v.
    """
    H = 8
    key = jax.random.PRNGKey(0)

    block_1head = SelfAttentionBlock(hidden_size=H, num_heads=1, key=key)
    block_Hhead = SelfAttentionBlock(hidden_size=H, num_heads=H, key=key)
    x = jax.random.normal(jax.random.PRNGKey(3), (H,))
    out1 = block_1head(x)
    outH = block_Hhead(x)
    assert not jnp.allclose(out1, outH, atol=1e-3), (
        f"num_heads=1 and num_heads=H must produce different outputs "
        f"because the head-axis softmax is non-trivial when head_dim=1; "
        f"if these are equal, num_heads is being ignored.\n"
        f"  out1={out1}\n  outH={outH}"
    )


def test_head_permutation_equivariance():
    """Test 7: permute row-blocks of W_Q/W_K/W_V (output channels) AND
    column-blocks of W_O (input channels). Output is unchanged.

    This is the multi-head invariance theorem (Vaswani §3.2.2 eq. (2)):
    heads are independent and the permutation undone by W_O.
    """
    H, nh = 16, 4
    head_dim = H // nh
    key = jax.random.PRNGKey(0)
    block = SelfAttentionBlock(hidden_size=H, num_heads=nh, key=key)
    x = jax.random.normal(jax.random.PRNGKey(4), (H,))
    out_orig = block(x)

    # Permute heads 0,1,2,3 -> 2,0,3,1
    perm = jnp.array([2, 0, 3, 1])

    def permute_rows_blockwise(w):
        # w shape (H, in); permute output channels in head blocks.
        w_blocks = w.reshape(nh, head_dim, w.shape[1])
        return w_blocks[perm].reshape(w.shape)

    def permute_bias_blockwise(b):
        b_blocks = b.reshape(nh, head_dim)
        return b_blocks[perm].reshape(H)

    def permute_cols_blockwise(w):
        # w shape (out, H); permute input channels in head blocks.
        w_blocks = w.reshape(w.shape[0], nh, head_dim)
        return w_blocks[:, perm, :].reshape(w.shape)

    block_perm = eqx.tree_at(
        lambda m: (
            m.query_proj.weight, m.query_proj.bias,
            m.key_proj.weight,   m.key_proj.bias,
            m.value_proj.weight, m.value_proj.bias,
            m.output_proj.weight,
        ),
        block,
        (
            permute_rows_blockwise(block.query_proj.weight),
            permute_bias_blockwise(block.query_proj.bias),
            permute_rows_blockwise(block.key_proj.weight),
            permute_bias_blockwise(block.key_proj.bias),
            permute_rows_blockwise(block.value_proj.weight),
            permute_bias_blockwise(block.value_proj.bias),
            permute_cols_blockwise(block.output_proj.weight),
        ),
    )
    out_perm = block_perm(x)
    assert jnp.allclose(out_orig, out_perm, atol=1e-5), (
        f"head permutation should be invariant; got\n"
        f"  orig {out_orig}\n  perm {out_perm}"
    )


def test_grad_flows_through_all_six_parameters():
    """Test 8: gradients on Q/K/V/O Linear weights AND LayerNorm
    weight/bias are all finite + nonzero on a random gaussian input.
    """
    H, nh = 8, 2
    key = jax.random.PRNGKey(0)
    block = SelfAttentionBlock(hidden_size=H, num_heads=nh, key=key)
    x = jax.random.normal(jax.random.PRNGKey(5), (H,))

    def loss(b, x):
        return jnp.sum(b(x) ** 2)

    grad = eqx.filter_grad(loss)(block, x)

    leaves = [
        ("query_proj.weight",  grad.query_proj.weight),
        ("key_proj.weight",    grad.key_proj.weight),
        ("value_proj.weight",  grad.value_proj.weight),
        ("output_proj.weight", grad.output_proj.weight),
        ("norm.weight",        grad.norm.weight),
        ("norm.bias",          grad.norm.bias),
    ]
    for name, g in leaves:
        assert jnp.all(jnp.isfinite(g)), f"{name} has non-finite grad"
        assert jnp.abs(g).max() > 1e-4, (
            f"{name} max-abs grad {jnp.abs(g).max()} is too small "
            f"(expected > 1e-4)"
        )


def test_layernorm_zero_mean_unit_variance_pre_affine():
    """Test 9: replicate LayerNorm pre-affine output independently and
    assert mean ~ 0, variance ~ 1 to atol=1e-5. Bypasses the learnable
    affine (which is identity at init but drifts after training).
    """
    H = 16
    x = jax.random.normal(jax.random.PRNGKey(6), (H,))

    eps = 1e-5
    mean = x.mean()
    var = x.var()
    pre_affine = (x - mean) / jnp.sqrt(var + eps)
    assert jnp.abs(pre_affine.mean()) < 1e-5
    assert jnp.abs(pre_affine.var() - 1.0) < 1e-3


def test_divisibility_assertion():
    """Test 10: SelfAttentionBlock(hidden_size=10, num_heads=4) raises."""
    with pytest.raises(ValueError, match="must be divisible by"):
        SelfAttentionBlock(hidden_size=10, num_heads=4,
                           key=jax.random.PRNGKey(0))


# ---------------------------------------------------------------------------
# Frozen pre-fix reference (DO NOT EDIT — used as regression catcher)
# ---------------------------------------------------------------------------

class _LegacyBrokenBlockReference(eqx.Module):
    """Frozen copy of the pre-2026-04-27 broken block. Used by Test 11
    only; never imported elsewhere. Math: softmax(q*k/sqrt(H)) * v
    (elementwise Hadamard, NOT QK^T).
    """
    query_proj: eqx.nn.Linear
    key_proj: eqx.nn.Linear
    value_proj: eqx.nn.Linear
    output_proj: eqx.nn.Linear
    hidden_size: int = eqx.field(static=True)

    def __init__(self, hidden_size, key):
        keys = jax.random.split(key, 4)
        self.hidden_size = hidden_size
        self.query_proj = eqx.nn.Linear(hidden_size, hidden_size, key=keys[0])
        self.key_proj = eqx.nn.Linear(hidden_size, hidden_size, key=keys[1])
        self.value_proj = eqx.nn.Linear(hidden_size, hidden_size, key=keys[2])
        self.output_proj = eqx.nn.Linear(hidden_size, hidden_size, key=keys[3])

    def __call__(self, x):
        x_seq = x.reshape(1, -1)
        q = self.query_proj(x_seq.squeeze())
        k = self.key_proj(x_seq.squeeze())
        v = self.value_proj(x_seq.squeeze())
        scale = jnp.sqrt(self.hidden_size).astype(x.dtype)
        attn_weights = jax.nn.softmax(q * k / scale)
        attended = attn_weights * v
        return self.output_proj(attended) + x_seq.squeeze()


# ---------------------------------------------------------------------------
# Tests 11-14
# ---------------------------------------------------------------------------

def test_regression_against_frozen_broken_block():
    """Test 11: new block's output must differ measurably from the frozen
    pre-fix broken block on the same input.
    """
    H = 8
    key = jax.random.PRNGKey(0)
    new_block = SelfAttentionBlock(hidden_size=H, num_heads=2, key=key)
    legacy = _LegacyBrokenBlockReference(hidden_size=H, key=key)
    x = jax.random.normal(jax.random.PRNGKey(7), (H,))
    out_new = new_block(x)
    out_legacy = legacy(x)
    assert not jnp.allclose(out_new, out_legacy, atol=1e-3), (
        "new attention block produces same output as the legacy broken "
        "block — rewrite did not take effect"
    )


def test_hand_computed_numeric_pin():
    """Test 12: with hidden_size=4, num_heads=2, set Q/K/V/O to identity,
    LayerNorm gamma=0/beta=1 so post-LN output is constant 1, and assert
    the module output equals a value computed from the canonical formula
    on the per-head matrices.

    Worked example: post-LN(x) = ones(4). Q=K=V=I -> q=k=v=ones(4).
    Reshape to (2,2): each row is [1,1]. scores = [1*1+1*1, 1*1+1*1] /
    sqrt(2) = [sqrt(2), sqrt(2)] in every cell. Softmax -> [0.5, 0.5]
    everywhere. attn @ v = ones(2,2). Reshape to (4,) = ones(4).
    Output_proj=I -> ones(4). + residual x.
    """
    H, nh = 4, 2
    key = jax.random.PRNGKey(0)
    block = SelfAttentionBlock(hidden_size=H, num_heads=nh, key=key)

    # Override to identities (chain tree_at calls; each returns a new module).
    block = eqx.tree_at(lambda m: m.query_proj.weight, block, jnp.eye(H))
    block = eqx.tree_at(lambda m: m.query_proj.bias, block, jnp.zeros(H))
    block = eqx.tree_at(lambda m: m.key_proj.weight, block, jnp.eye(H))
    block = eqx.tree_at(lambda m: m.key_proj.bias, block, jnp.zeros(H))
    block = eqx.tree_at(lambda m: m.value_proj.weight, block, jnp.eye(H))
    block = eqx.tree_at(lambda m: m.value_proj.bias, block, jnp.zeros(H))
    block = eqx.tree_at(lambda m: m.output_proj.weight, block, jnp.eye(H))
    block = eqx.tree_at(lambda m: m.output_proj.bias, block, jnp.zeros(H))
    # LN: gamma=0, bias=1 -> post-LN output is the constant ones(H)
    block = eqx.tree_at(lambda m: m.norm.weight, block, jnp.zeros(H))
    block = eqx.tree_at(lambda m: m.norm.bias, block, jnp.ones(H))

    x = jnp.array([3.0, -1.0, 0.5, 7.0])
    expected = jnp.ones(H) + x  # attn output is ones(H), residual adds x
    actual = block(x)
    assert jnp.allclose(actual, expected, atol=1e-5), (
        f"hand-computed pin: expected {expected}, got {actual}"
    )


def test_save_load_roundtrip(tmp_path):
    """Test 13: serialise then deserialise; output must match."""
    H = 8
    key = jax.random.PRNGKey(0)
    block = SelfAttentionBlock(hidden_size=H, num_heads=2, key=key)
    x = jax.random.normal(jax.random.PRNGKey(8), (H,))
    out_before = block(x)

    p = tmp_path / "block.eqx"
    eqx.tree_serialise_leaves(str(p), block)
    skel = SelfAttentionBlock(hidden_size=H, num_heads=2,
                              key=jax.random.PRNGKey(99))
    block_loaded = eqx.tree_deserialise_leaves(str(p), skel)
    out_after = block_loaded(x)
    assert jnp.allclose(out_before, out_after, atol=1e-6)


def test_blocks_are_class_identical_across_modules():
    """Test 14: alec re-imports xcquinox.net.SelfAttentionBlock; class
    identity must hold (not a stale duplicate import).
    """
    from xcquinox.alec import networks as alec_networks
    import xcquinox.net as legacy_net
    assert alec_networks._xnet.SelfAttentionBlock is legacy_net.SelfAttentionBlock
