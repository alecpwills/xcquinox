"""xcquinox.alec.pretrain — Pretraining and legacy checkpoint loading.

Implements THE SPEC §8.1 (run_pretrain), §8.3 (from_legacy_step3b,
_load_one_network, _count_disk_records, _metadata_preflight,
_disk_record_preflight), and the B-H-R15 Round 15 legacy lob_lim constants.

Plan deviation (§13.6 fixture format): pretrain data is stored as .npz
(numpy compressed archive) rather than .pkl. The loader falls back to .pkl
for legacy files.
"""
import json
import os
import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
from equinox._filters import is_array_like

import xcquinox.net
import xcquinox.train

from xcquinox.alec.config import ArchitectureConfig, PretrainSpec
from xcquinox.alec.networks import AlecGGA_XNet, AlecGGA_CNet, create_network_pair
from xcquinox.utils import lda_x, pw92c_unpolarized_scalar


_RHO_FLOOR_INTEGRATION = 1e-18


def _compute_integration_weights(rho, grid_weights=None):
    """Return ``(w_x, w_c)`` integration weights for pretraining.

    **Weight convention (PRE-01, option b):** the per-point weight is the
    FIRST power (linear) of ``|ρ_i · ε_LDA_i|``, optionally multiplied by
    the Becke-Lebedev quadrature weight ``w_grid_i``.  The resulting loss
    is a ``|ρ · ε_LDA|``-magnitude-weighted mean of the squared per-point
    enhancement-factor residual::

        L = Σ_i w_i · (F_nn_i - F_ref_i)²  /  Σ_i w_i
            where w_i = |ρ_i · ε_LDA_i|   (or  |ρ_i · ε_LDA_i| · w_grid_i
                                             when grid_weights is supplied).

    This is NOT the squared integrated XC-energy residual.  A true
    integrated-energy L2 loss would require the square of the energy-density
    weight, i.e.  ``w_i = (ρ_i · ε_LDA_i · w_grid_i)²``.  The linear form
    was retained because it is the established convention in the codebase
    (all prior pretrain runs used it) and existing tests pin this behavior.
    The ``|ρ · ε_LDA|`` factor still steers gradient attention toward
    energetically important regions without the large dynamic range that the
    squared form would introduce.

    When ``grid_weights`` is supplied the weights incorporate the quadrature
    measure ``dr_i``, improving the energy-density calibration relative to the
    unweighted form.  When ``grid_weights`` is ``None`` the loss reverts to
    the legacy ``|ρ ε_LDA|``-weighted mean per sample; ``run_pretrain`` emits
    a ``RuntimeWarning`` in that case and records
    ``integration_weights_complete=False`` in the pretrain metadata.

    Parameters
    ----------
    rho : jnp.ndarray
        Electron density at grid points, shape ``(N,)``.
    grid_weights : jnp.ndarray | None
        Becke-Lebedev quadrature weights ``dr_i`` per grid point, shape
        ``(N,)``.  When supplied the per-point weight becomes
        ``|ρ ε_LDA| · w_grid``.  When ``None`` (legacy / pre-fix behavior)
        the ``dr_i`` factor is omitted; callers should expect the
        ``RuntimeWarning`` from ``run_pretrain``.

    Returns
    -------
    (w_x, w_c) : tuple of jnp.ndarray, each shape ``(N,)``
        Non-negative integration weights for the exchange and correlation
        networks respectively.
    """
    rho_safe = jnp.maximum(rho, _RHO_FLOOR_INTEGRATION)
    eps_x_lda = lda_x(rho_safe)
    eps_c_lda = pw92c_unpolarized_scalar(rho_safe)
    w_x = jnp.abs(rho_safe * eps_x_lda)
    w_c = jnp.abs(rho_safe * eps_c_lda)
    if grid_weights is not None:
        gw = jnp.asarray(grid_weights)
        w_x = w_x * gw
        w_c = w_c * gw
    return jnp.broadcast_to(w_x, rho_safe.shape), \
           jnp.broadcast_to(w_c, rho_safe.shape)


# === B-H-R15 Round 15 fix: legacy lob_lim constants ===
#
# The library-shaped skeletons constructed below are used ONLY to
# deserialize the on-disk byte stream into a pytree whose leaves we
# then graft onto the alec skeleton. Those skeletons therefore MUST
# receive the exact `lob_lim` values that the legacy checkpoints were
# written with — which are the hardcoded library defaults `1.804` (xnet,
# per `xcquinox/net.py:2049` `GGA_FxNet_extended.__init__` default) and
# `2.0` (cnet, per `xcquinox/net.py:2228` `GGA_FcNet_extended.__init__`
# default). Earlier drafts passed `arch.resolved_xnet_lob_lim` /
# `arch.resolved_cnet_lob_lim` here, which was a bug: when a
# `LiebOxfordBound` constraint is registered on `arch.x_constraints`
# AND `arch.double_lob_clamp_allowed=False`, the `resolved_xnet_lob_lim`
# property returns `None` (see §11.1 lines 6016-6024) so that
# `create_network_pair` can build an alec XNet that skips its built-in
# LOB wrap (the constraint becomes the sole clamp). But `None` is
# invalid for the LIBRARY class — `GGA_FxNet_extended(lob_lim=None, ...)`
# flows into `self.lobf = LOB(limit=None)` which crashes with TypeError
# the instant `LOB.__call__` dereferences `self.limit`, and the
# post-load `abs(loaded_lim - expected_lob_lim)` check would also crash
# with `TypeError: unsupported operand type(s) for -: 'float' and
# 'NoneType'`. The legacy checkpoints were *all* trained with the
# library defaults (the step3b notebook never configured
# `double_lob_clamp_allowed` because that field does not exist on the
# notebook's architecture), so hardcoding the legacy values is
# semantically correct — we are loading bytes that were *written* with
# these values, not configuring a fresh network. The alec-side
# `create_network_pair` still honors `arch.resolved_xnet_lob_lim` (which
# may be `None` under the C-H1 avoidance rule); the library→alec
# leaf graft is oblivious to `lob_lim` because `_AlecLOB.limit` is
# `eqx.field(static=True)` (§5.1 D-H2) and therefore not in the pytree
# leaf stream.
_legacy_xnet_lob_lim: float = 1.804  # `xcquinox/net.py:2049` default
_legacy_cnet_lob_lim: float = 2.0    # `xcquinox/net.py:2228` default


# ---------------------------------------------------------------------------
# Pretrain data assembly helper
# ---------------------------------------------------------------------------

def _assemble_pretrain_descriptors(arch: ArchitectureConfig, pretrain_data: dict,
                                   *, for_cnet: bool = False) -> jnp.ndarray:
    """Assemble the (N, F) input array for pretraining from pretrain_data.

    Column order: [rho_all, sigma_all, *(per-descriptor columns)] where
    the per-descriptor columns follow ``arch.descriptors`` declaration
    order (matching `descriptors.assemble_descriptor_features`'s
    runtime contract). Each descriptor's columns are pulled from a
    pretrain_data key derived by stripping ``_statistics`` and
    appending ``_all`` (e.g. ``dm_statistics`` → ``dm_all``,
    ``cusp`` → ``cusp_all``).

    P2-03: when ``for_cnet`` and ``arch.use_polarized_correlation``, a spin
    polarization column ``zeta_all`` is inserted at index 2 (right after
    sigma, BEFORE the descriptor extras) to match the polarized cnet's
    expected ``[rho, sigma, zeta, *extras]`` input layout (see
    ``AlecGGA_CNet.__call__``). The xnet input (``for_cnet=False``) NEVER
    carries zeta — exchange is zeta-independent (Oliver & Perdew, PRA 20,
    397 (1979)). If ``zeta_all`` is absent, zeta defaults to zeros: a valid
    unpolarized warm-start (zeta=0 -> x1=1, recovering the unpolarized cnet
    input and Fc target), to be refined by zeta-resolved training data.

    Raises KeyError if any declared descriptor's pretrain key is
    absent from pretrain_data — there is NO zero-array fallback
    (L-B14-2 Round 14).
    """
    cols = [pretrain_data["rho_all"], pretrain_data["sigma_all"]]
    if for_cnet and arch.use_polarized_correlation:
        zeta_all = pretrain_data.get("zeta_all")
        if zeta_all is None:
            zeta_all = jnp.zeros_like(pretrain_data["rho_all"])
        cols.append(zeta_all)
    # Map descriptor.name -> key in pretrain_data.
    _key_map = {"dm_statistics": "dm_all", "cusp": "cusp_all"}
    for spec in arch.descriptors:
        key = _key_map.get(spec.name)
        if key is None:
            raise KeyError(
                f"_assemble_pretrain_descriptors: no pretrain_data key "
                f"mapping registered for descriptor {spec.name!r}; update "
                f"_key_map in pretrain.py"
            )
        arr = pretrain_data[key]
        if arr.ndim == 1:
            cols.append(arr)
        else:
            for i in range(arr.shape[1]):
                cols.append(arr[:, i])
    return jnp.stack(cols, axis=1)


# ---------------------------------------------------------------------------
# Optimizer builder
# ---------------------------------------------------------------------------

def _build_optimizer(
    *,
    lr_start: float,
    lr_end: float,
    n_steps: int,
    lr_decay_start: float,
    grad_clip: float,
) -> optax.GradientTransformation:
    """Build canonical optimizer chain for pretraining.

    Chain order: clip_by_global_norm → adam(lr_schedule).
    LR schedule: optional constant warmup then linear decay.
    """
    decay_start_step = int(lr_decay_start * n_steps)
    decay_steps = n_steps - decay_start_step

    if lr_decay_start > 0 and decay_steps > 0:
        lr_schedule = optax.join_schedules(
            schedules=[
                optax.constant_schedule(lr_start),
                optax.linear_schedule(
                    init_value=lr_start,
                    end_value=lr_end,
                    transition_steps=decay_steps,
                ),
            ],
            boundaries=[decay_start_step],
        )
    else:
        lr_schedule = optax.linear_schedule(
            init_value=lr_start,
            end_value=lr_end,
            transition_steps=n_steps,
        )

    return optax.chain(
        optax.clip_by_global_norm(grad_clip),
        optax.adam(learning_rate=lr_schedule),
    )


# ---------------------------------------------------------------------------
# run_pretrain
# ---------------------------------------------------------------------------

def run_pretrain(spec: PretrainSpec, progress_callback=None) -> dict:
    """Pretrain xnet and cnet on synthetic grid data.

    Steps:
    1. spec.validate()
    2. Load pretrain_data.npz (or .pkl fallback) from spec.data_dir
    3. Assemble descriptor tensor
    4. Create network pair
    5. Train xnet on Fx_target, then cnet on Fc_target (both MSE)
    6. Build optimizer (optax chain)
    7. Save artifacts under spec.checkpoint_dir
    8. Return metadata dict

    Constraint awareness: the networks built by ``create_network_pair`` enforce
    ``spec.arch``'s physical constraints INTRINSICALLY in their forward pass, so
    the MSE here fits the CONSTRAINED enhancement to the PBE/LDA targets — the
    same constrained functional that training and evaluation use. (Constraints
    are static, so the saved ``xnet.eqx``/``cnet.eqx`` leaf streams are unchanged
    and remain compatible with existing checkpoints.)

    Returns:
        dict with pretrain_metadata.json fields.
    """
    spec.validate()

    # --- Load pretrain data (plan deviation: npz instead of pkl) ---
    npz_path = os.path.join(spec.data_dir, "pretrain_data.npz")
    pkl_path = os.path.join(spec.data_dir, "pretrain_data.pkl")

    if os.path.isfile(npz_path):
        raw = np.load(npz_path)
        pretrain_data_np = {k: np.array(raw[k]) for k in raw.files}
    elif os.path.isfile(pkl_path):
        # pkl fallback for legacy files — safe because only array data
        import pickle  # noqa: S403 — loading trusted local fixture files only
        with open(pkl_path, "rb") as _f:
            pretrain_data_np = _f.read()
        pretrain_data_np = __import__("pickle").loads(pretrain_data_np)
    else:
        raise FileNotFoundError(
            f"run_pretrain: neither pretrain_data.npz nor pretrain_data.pkl "
            f"found in data_dir={spec.data_dir!r}"
        )

    # Lift every array into JAX
    pretrain_data = {k: jnp.array(v) for k, v in pretrain_data_np.items()}

    # --- Assemble descriptor tensors ---
    # The xnet input is zeta-blind; the cnet input carries the zeta column
    # when the architecture uses polarized correlation (P2-03). They are
    # identical for the unpolarized (default) architecture.
    descriptors = _assemble_pretrain_descriptors(spec.arch, pretrain_data)
    descriptors_c = _assemble_pretrain_descriptors(
        spec.arch, pretrain_data, for_cnet=True)

    # Targets are stored as (F - 1), not F; PretrainLoss subtracts 1 from
    # network output (networks return 1 + enhancement).
    Fx_target = pretrain_data["Fx_all"]
    Fc_target = pretrain_data["Fc_all"]

    # --- Create network pair ---
    xnet, cnet = create_network_pair(spec.arch, seed=spec.seed)

    # --- PretrainLoss (scalar MSE, compatible with xcTrainer) ---
    #
    # Loss weighting is controlled by `spec.loss_weighting`:
    #   "unweighted" (default): plain mean of squared residuals.
    #   "integration":          integration-weighted sum-of-squared residuals,
    #                           with per-component weights |rho * eps^LDA|
    #                           computed once from pretrain_data["rho_all"]
    #                           (see _compute_integration_weights).
    #
    # The xnet and cnet are trained in separate trainer calls using different
    # weight arrays (w_x vs w_c). Each is baked into its own closure-like
    # eqx.Module instance so that xcTrainer can call `loss(model, descriptors,
    # ref_F)` with the familiar 3-arg API. Weights are carried as a static
    # array field on the module (eqx handles the pytree correctly).
    class _PretrainLoss(eqx.Module):
        """Scalar MSE loss for pretraining enhancement networks.

        Networks return 1 + F_enhancement; targets are stored as (F - 1),
        so pred - 1.0 aligns with ref_F. When ``weights`` is None (the
        ``"unweighted"`` branch) the reduction is a plain mean of squared
        residuals — preserving the exact prior behavior byte-for-byte. When
        ``weights`` is a 1-D array aligned with the descriptor rows
        (``"integration"`` branch) the reduction is
        ``sum(w * residual ** 2) / (sum(w) + 1e-12)``.
        """
        weights: jnp.ndarray | None = None

        def __call__(self, model, descriptors, ref_F):
            pred = jax.vmap(model)(descriptors).squeeze()
            pred = pred - 1.0
            residual_sq = (pred - ref_F) ** 2
            if self.weights is None:
                return jnp.mean(residual_sq)
            w = self.weights
            return jnp.sum(w * residual_sq) / (jnp.sum(w) + 1e-12)

    integration_weights_complete: bool | None = None  # set below for "integration" mode
    if spec.loss_weighting == "integration":
        rho_all = pretrain_data["rho_all"]
        # Becke-Lebedev quadrature weights ``dr_i`` improve energy-density
        # calibration; older pretrain_data files don't carry them — fall back
        # with a warning and record the degradation in metadata (PRE-02).
        grid_weights = pretrain_data.get("weights_all")
        if grid_weights is None:
            integration_weights_complete = False
            import warnings as _warn
            _msg = (
                "pretrain_data.npz lacks 'weights_all'; integration-mode "
                "loss is missing Becke quadrature weights and approximates "
                "a |rho*eps_LDA|-weighted mean rather than the integrated "
                "XC-energy residual. Regenerate pretrain_data.npz from a "
                "post-2026-04-27 notebook generator to get correct weights."
            )
            _warn.warn(_msg, RuntimeWarning, stacklevel=2)
            # RuntimeWarnings are easy to miss in a SLURM .out log; also emit a
            # flushed banner so the degradation is unmissable there (CW1-M1).
            print(f"\n{'!' * 72}\n[PRETRAIN WARNING] {_msg}\n{'!' * 72}\n",
                  flush=True)
        else:
            integration_weights_complete = True
        w_x, w_c = _compute_integration_weights(rho_all, grid_weights)
        loss_fn_x = _PretrainLoss(weights=w_x)
        loss_fn_c = _PretrainLoss(weights=w_c)
    else:  # "unweighted" — validated at construction
        loss_fn_x = _PretrainLoss()
        loss_fn_c = _PretrainLoss()

    checkpoint_dir = spec.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Helper: build phase progress callbacks for xcTrainer 3-arg form
    def _x_callback(step, total, loss):
        if progress_callback is not None:
            progress_callback({
                "arch": spec.arch.name,
                "phase": "X",
                "step": step,
                "total": total,
                "loss": float(loss),
                "timestamp": time.time(),
            })

    def _c_callback(step, total, loss):
        if progress_callback is not None:
            progress_callback({
                "arch": spec.arch.name,
                "phase": "C",
                "step": step,
                "total": total,
                "loss": float(loss),
                "timestamp": time.time(),
            })

    # Per-network checkpoint subdirs. xcTrainer serialises its periodic
    # best-loss snapshots as ``<checkpoint_dir>/xc.eqx.<step>`` — if both the
    # xnet and cnet trainers share one checkpoint_dir they clobber each other's
    # snapshots. Give each its own subdir; the FINAL xnet.eqx/cnet.eqx still
    # land at the top level (what downstream consumes).
    xnet_ckpt_dir = os.path.join(checkpoint_dir, "xnet")
    cnet_ckpt_dir = os.path.join(checkpoint_dir, "cnet")
    os.makedirs(xnet_ckpt_dir, exist_ok=True)
    os.makedirs(cnet_ckpt_dir, exist_ok=True)
    xnet_path = os.path.join(checkpoint_dir, "xnet.eqx")
    cnet_path = os.path.join(checkpoint_dir, "cnet.eqx")

    # --- Train xnet ---
    t0 = time.time()
    optimizer_x = _build_optimizer(
        lr_start=spec.lr_start,
        lr_end=spec.lr_end,
        n_steps=spec.n_steps,
        lr_decay_start=spec.lr_decay_start,
        grad_clip=spec.grad_clip,
    )
    trainer_x = xcquinox.train.xcTrainer(
        model=xnet,
        optim=optimizer_x,
        loss=loss_fn_x,
        steps=spec.n_steps,
        do_jit=True,
        serialize_every=max(50, spec.n_steps // 10),
        checkpoint_dir=xnet_ckpt_dir,
        progress_callback=_x_callback,
    )
    xnet_trained, losses_x = trainer_x(1, [descriptors], [Fx_target])
    # Persist the final xnet immediately, BEFORE cnet training starts — so a
    # job that dies or times out during the (separate) cnet phase does not lose
    # the already-completed xnet result.
    eqx.tree_serialise_leaves(xnet_path, xnet_trained)

    # --- Train cnet (fresh optimizer) ---
    optimizer_c = _build_optimizer(
        lr_start=spec.lr_start,
        lr_end=spec.lr_end,
        n_steps=spec.n_steps,
        lr_decay_start=spec.lr_decay_start,
        grad_clip=spec.grad_clip,
    )
    trainer_c = xcquinox.train.xcTrainer(
        model=cnet,
        optim=optimizer_c,
        loss=loss_fn_c,
        steps=spec.n_steps,
        do_jit=True,
        serialize_every=max(50, spec.n_steps // 10),
        checkpoint_dir=cnet_ckpt_dir,
        progress_callback=_c_callback,
    )
    cnet_trained, losses_c = trainer_c(1, [descriptors_c], [Fc_target])
    duration = time.time() - t0

    # --- Save artifacts ---
    eqx.tree_serialise_leaves(cnet_path, cnet_trained)

    losses_x_np = np.array(losses_x, dtype=np.float64)
    losses_c_np = np.array(losses_c, dtype=np.float64)
    np.save(os.path.join(checkpoint_dir, "losses_x.npy"), losses_x_np)
    np.save(os.path.join(checkpoint_dir, "losses_c.npy"), losses_c_np)

    use_cusp = any(s.name == "cusp" for s in spec.arch.descriptors)
    use_dm = any(s.name == "dm_statistics" for s in spec.arch.descriptors)

    metadata = {
        "arch_name": spec.arch.name,
        "depth": spec.arch.depth,
        "nodes": spec.arch.nodes,
        "pretrain_steps": spec.n_steps,
        "lr_start": spec.lr_start,
        "lr_end": spec.lr_end,
        "lr_decay_start": spec.lr_decay_start,
        "grad_clip": spec.grad_clip,
        "loss_weighting": spec.loss_weighting,
        "final_loss_x": float(losses_x_np[-1]) if len(losses_x_np) > 0 else float("nan"),
        "final_loss_c": float(losses_c_np[-1]) if len(losses_c_np) > 0 else float("nan"),
        "min_loss_x": float(np.min(losses_x_np)) if len(losses_x_np) > 0 else float("nan"),
        "min_loss_c": float(np.min(losses_c_np)) if len(losses_c_np) > 0 else float("nan"),
        "use_cusp": use_cusp,
        "use_dm": use_dm,
        # Shape-changing flag (CODE-5 round-4): polarized cnet input width +1.
        "use_polarized_correlation": bool(spec.arch.use_polarized_correlation),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
        "duration_seconds": round(duration, 1),
    }
    # PRE-02: record whether Becke quadrature weights were available for
    # integration mode.  None means the run did not use integration weighting.
    if integration_weights_complete is not None:
        metadata["integration_weights_complete"] = integration_weights_complete
    md_path = os.path.join(checkpoint_dir, "pretrain_metadata.json")
    with open(md_path, "w") as f:
        json.dump(metadata, f, indent=2)

    return metadata


# ---------------------------------------------------------------------------
# Legacy checkpoint helpers
# ---------------------------------------------------------------------------

def _count_disk_records(path: str) -> int:
    """Count numpy magic-marker occurrences in a .eqx file.

    L-A13-1 note: byte-level heuristic, not a true numpy header parser.
    Safe for the dtypes this loader sees; see spec §8.3 for the
    over-count failure mode and when to replace with a real parser.

    M-E13-2 Round 13: stream the file in chunks with a small overlap
    to avoid slurping a multi-hundred-MB checkpoint into memory.
    """
    marker = b'\x93NUMPY'
    count = 0
    chunk_size = 65536
    overlap = len(marker) - 1
    with open(path, 'rb') as f:
        tail = b''
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            count += (tail + chunk).count(marker)
            tail = chunk[-overlap:]
    return count


def _metadata_preflight(
    *,
    metadata_path: str,
    arch: ArchitectureConfig,
) -> dict:
    """M-E13-1 / H-E14-2 / M-E14-1: validate checkpoint metadata against arch."""
    try:
        with open(metadata_path, "r") as f:
            md = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(
            f"legacy checkpoint metadata unreadable: {metadata_path}: {e}"
        ) from e
    for key in ("depth", "nodes"):
        if key not in md:
            raise ValueError(
                f"legacy checkpoint metadata missing required key "
                f"{key!r}: {metadata_path}"
            )
    if md["depth"] != arch.depth or md["nodes"] != arch.nodes:
        raise ValueError(
            f"legacy checkpoint metadata mismatch: file has "
            f"depth={md['depth']}, nodes={md['nodes']}; "
            f"arch expects depth={arch.depth}, nodes={arch.nodes}"
        )
    return md


def _disk_record_preflight(
    *,
    path: str,
    library_skeleton: eqx.Module,
    arch: ArchitectureConfig,
) -> None:
    """L-A13-1 / M-E13-2: byte-level leaf-count sanity on the library skeleton."""
    n_disk_records = _count_disk_records(path)
    n_expected = sum(
        1 for leaf in jax.tree_util.tree_leaves(library_skeleton)
        if is_array_like(leaf)
    )
    if n_disk_records != n_expected:
        raise ValueError(
            f"legacy checkpoint {path} has {n_disk_records} on-disk "
            f"records, expected {n_expected} for library skeleton "
            f"(depth={arch.depth}, nodes={arch.nodes}, "
            f"use_self_attention={arch.attention})"
        )


def _load_one_network(
    *,
    path: str,
    library_skeleton: eqx.Module,
    alec_skeleton: eqx.Module,
    expected_lob_lim: float,
) -> eqx.Module:
    """Load a single network from a legacy-layout checkpoint and graft its
    trainable array leaves onto a fresh alec skeleton.

    Steps:
    1. Deserialize into library_skeleton (B-H4 / H-E14-1: capture return value).
    2. Check lobf.limit against expected_lob_lim (H-E13-2 / C-C14-1 Round 14).
    3. Graft eqx.is_array leaves from library onto alec via tree_flatten/unflatten
       (C-E14-1 Round 14 pattern — NOT eqx.tree_at).
    """
    # Step 2: deserialize — MUST capture the return value
    library_skeleton = eqx.tree_deserialise_leaves(path, library_skeleton)

    # Parameterised lob_lim check (C-C14-1 / H-E13-2 Round 14)
    loaded_lim = library_skeleton.lobf.limit
    if abs(loaded_lim - expected_lob_lim) >= 1e-12:
        raise ValueError(
            f"legacy checkpoint {path}: lobf.limit={loaded_lim} does not "
            f"match expected {expected_lob_lim} for this architecture"
        )

    # Step 3: structural tree walk — NOT eqx.tree_at
    src_leaves_all, _src_treedef = jax.tree_util.tree_flatten(library_skeleton)
    src_array_leaves = [l for l in src_leaves_all if eqx.is_array(l)]

    dst_leaves_all, dst_treedef = jax.tree_util.tree_flatten(alec_skeleton)

    dst_array_count = sum(1 for l in dst_leaves_all if eqx.is_array(l))
    if len(src_array_leaves) != dst_array_count:
        schema_hint = ""
        if "_attn" in path or "/attention" in path:
            schema_hint = (
                "\n\nThis path includes an attention checkpoint. The "
                "self-attention block was rewritten 2026-04-27 to real "
                "multi-head scaled-dot-product attention; old `_attn` "
                "checkpoints are NOT loadable under the new schema. "
                "Delete the old checkpoint and retrain."
            )
        raise ValueError(
            f"legacy->alec graft leaf mismatch at {path}: library has "
            f"{len(src_array_leaves)} eqx.is_array leaves, alec skeleton "
            f"expects {dst_array_count}. Likely causes: arch.depth or "
            f"arch.nodes differs from the pretrained checkpoint, or the "
            f"checkpoint predates the 2026-04-27 attention rewrite."
            f"{schema_hint}"
        )

    # Per-leaf shape/dtype sanity before the graft
    dst_array_positions = [i for i, l in enumerate(dst_leaves_all) if eqx.is_array(l)]
    for pair_idx, dst_pos in enumerate(dst_array_positions):
        s = src_array_leaves[pair_idx]
        d = dst_leaves_all[dst_pos]
        if s.shape != d.shape or s.dtype != d.dtype:
            raise ValueError(
                f"legacy→alec graft leaf #{pair_idx} shape/dtype mismatch "
                f"at {path}: library {s.shape}/{s.dtype} vs alec "
                f"{d.shape}/{d.dtype}"
            )

    # Final graft: build new destination leaf list and unflatten
    src_iter = iter(src_array_leaves)
    new_leaves = [
        next(src_iter) if eqx.is_array(l) else l
        for l in dst_leaves_all
    ]
    return jax.tree_util.tree_unflatten(dst_treedef, new_leaves)


def from_legacy_step3b(
    legacy_dir: str,
    arch: ArchitectureConfig,
) -> tuple:
    """Load a step3b-era checkpoint into fresh AlecGGA_XNet / AlecGGA_CNet.

    Handles two on-disk layouts:
      Pretrain: legacy_dir/xnet.eqx + cnet.eqx + pretrain_metadata.json
      Training: legacy_dir/xcmodel.eqx + train_metadata.json

    Raises:
        ValueError: ambiguous layout (both present), or metadata mismatch.
        FileNotFoundError: no recognised layout found.
        OSError: file exists but cannot be read.
    """
    pretrain_x_path = os.path.join(legacy_dir, "xnet.eqx")
    pretrain_c_path = os.path.join(legacy_dir, "cnet.eqx")
    pretrain_md_path = os.path.join(legacy_dir, "pretrain_metadata.json")
    training_model_path = os.path.join(legacy_dir, "xcmodel.eqx")
    training_md_path = os.path.join(legacy_dir, "train_metadata.json")

    has_pretrain = (
        os.path.isfile(pretrain_x_path)
        and os.path.isfile(pretrain_c_path)
        and os.path.isfile(pretrain_md_path)
    )
    has_training = (
        os.path.isfile(training_model_path)
        and os.path.isfile(training_md_path)
    )

    if has_pretrain and has_training:
        raise ValueError(
            f"legacy checkpoint directory {legacy_dir} contains BOTH a "
            f"pretrain layout (xnet.eqx + cnet.eqx + pretrain_metadata.json) "
            f"AND a training layout (xcmodel.eqx + train_metadata.json); "
            f"cannot disambiguate. Delete one set before retrying."
        )
    if not has_pretrain and not has_training:
        raise FileNotFoundError(
            f"legacy checkpoint directory {legacy_dir} contains neither a "
            f"pretrain layout nor a training layout. Expected one of: "
            f"(xnet.eqx, cnet.eqx, pretrain_metadata.json) OR "
            f"(xcmodel.eqx, train_metadata.json)."
        )

    if has_pretrain:
        # --- Pretrain-layout branch ---
        _metadata_preflight(metadata_path=pretrain_md_path, arch=arch)

        # B-H-R15 Round 15: library skeletons use hardcoded legacy values
        lib_xnet_skel = xcquinox.net.GGA_FxNet_extended(
            depth=arch.depth, nodes=arch.nodes, seed=0,
            lob_lim=_legacy_xnet_lob_lim,
            lower_rho_cutoff=1e-12,
            use_self_attention=arch.attention,
            use_laplacian=False,
            use_dm_features=any(d.name == "dm_statistics" for d in arch.descriptors),
            use_cusp=any(d.name == "cusp" for d in arch.descriptors),
            n_dm_features=3,
        )
        lib_cnet_skel = xcquinox.net.GGA_FcNet_extended(
            depth=arch.depth, nodes=arch.nodes, seed=0,
            lob_lim=_legacy_cnet_lob_lim,
            lower_rho_cutoff=1e-12,
            use_self_attention=arch.attention,
            use_laplacian=False,
            use_dm_features=any(d.name == "dm_statistics" for d in arch.descriptors),
            use_cusp=any(d.name == "cusp" for d in arch.descriptors),
            n_dm_features=3,
        )
        alec_xnet_skel, alec_cnet_skel = create_network_pair(arch, seed=0)

        _disk_record_preflight(
            path=pretrain_x_path, library_skeleton=lib_xnet_skel, arch=arch,
        )
        _disk_record_preflight(
            path=pretrain_c_path, library_skeleton=lib_cnet_skel, arch=arch,
        )

        xnet_loaded = _load_one_network(
            path=pretrain_x_path,
            library_skeleton=lib_xnet_skel,
            alec_skeleton=alec_xnet_skel,
            expected_lob_lim=_legacy_xnet_lob_lim,
        )
        cnet_loaded = _load_one_network(
            path=pretrain_c_path,
            library_skeleton=lib_cnet_skel,
            alec_skeleton=alec_cnet_skel,
            expected_lob_lim=_legacy_cnet_lob_lim,
        )
        return xnet_loaded, cnet_loaded

    else:  # has_training
        # --- Training-layout branch (H-E14-3 Round 14) ---
        _metadata_preflight(metadata_path=training_md_path, arch=arch)

        class _RXCModelWrapper(eqx.Module):
            """Minimal inline replica of the notebook's RXCModel_GGA_extended.

            Two fields, no methods — just a pytree container so
            `tree_deserialise_leaves` can consume the on-disk records in
            the same order the notebook wrote them.
            """
            xnet: eqx.Module
            cnet: eqx.Module

        # B-H-R15 Round 15: same hardcoded legacy values here
        lib_xnet_skel = xcquinox.net.GGA_FxNet_extended(
            depth=arch.depth, nodes=arch.nodes, seed=0,
            lob_lim=_legacy_xnet_lob_lim,
            lower_rho_cutoff=1e-12,
            use_self_attention=arch.attention,
            use_laplacian=False,
            use_dm_features=any(d.name == "dm_statistics" for d in arch.descriptors),
            use_cusp=any(d.name == "cusp" for d in arch.descriptors),
            n_dm_features=3,
        )
        lib_cnet_skel = xcquinox.net.GGA_FcNet_extended(
            depth=arch.depth, nodes=arch.nodes, seed=0,
            lob_lim=_legacy_cnet_lob_lim,
            lower_rho_cutoff=1e-12,
            use_self_attention=arch.attention,
            use_laplacian=False,
            use_dm_features=any(d.name == "dm_statistics" for d in arch.descriptors),
            use_cusp=any(d.name == "cusp" for d in arch.descriptors),
            n_dm_features=3,
        )
        wrapper_skel = _RXCModelWrapper(xnet=lib_xnet_skel, cnet=lib_cnet_skel)

        # H-E14-1 Round 14: MUST capture the return value
        wrapper_loaded = eqx.tree_deserialise_leaves(training_model_path, wrapper_skel)
        lib_xnet_loaded = wrapper_loaded.xnet
        lib_cnet_loaded = wrapper_loaded.cnet

        # Parameterised lob_lim check (B-H-R15 constant fix)
        for loaded, expected, label in [
            (lib_xnet_loaded, _legacy_xnet_lob_lim, "xnet"),
            (lib_cnet_loaded, _legacy_cnet_lob_lim, "cnet"),
        ]:
            if abs(loaded.lobf.limit - expected) >= 1e-12:
                raise ValueError(
                    f"legacy training checkpoint {training_model_path}: "
                    f"{label}.lobf.limit={loaded.lobf.limit} does not match "
                    f"expected {expected} for this architecture"
                )

        alec_xnet_skel, alec_cnet_skel = create_network_pair(arch, seed=0)

        def _graft_arrays(src: eqx.Module, dst: eqx.Module) -> eqx.Module:
            src_leaves_all, _ = jax.tree_util.tree_flatten(src)
            src_array_leaves = [l for l in src_leaves_all if eqx.is_array(l)]
            dst_leaves_all, dst_treedef = jax.tree_util.tree_flatten(dst)
            dst_array_count = sum(1 for l in dst_leaves_all if eqx.is_array(l))
            if len(src_array_leaves) != dst_array_count:
                raise ValueError(
                    f"legacy→alec training-layout graft leaf mismatch: "
                    f"library has {len(src_array_leaves)} eqx.is_array "
                    f"leaves, alec expects {dst_array_count}."
                )
            src_iter = iter(src_array_leaves)
            new_leaves = [
                next(src_iter) if eqx.is_array(l) else l
                for l in dst_leaves_all
            ]
            return jax.tree_util.tree_unflatten(dst_treedef, new_leaves)

        xnet_loaded = _graft_arrays(lib_xnet_loaded, alec_xnet_skel)
        cnet_loaded = _graft_arrays(lib_cnet_loaded, alec_cnet_skel)
        return xnet_loaded, cnet_loaded
