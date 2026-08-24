"""Isolation instrument for the dfs6311 training NaN on bh76:OH+N2_to_H+N2O.

Cluster job 2091616 (padded spec_0021 smoke, gradient-sweep guard live) aborted
at per-molecule step 4, group ``bh76:OH+N2_to_H+N2O``, with a FINITE loss
(1.0856e-2, every component finite) and ALL 36/36 gradient leaves NaN. That
exonerates ``ae:CO`` (the step-5 abort of run 2090653 was the one-step-late
symptom of this step's silently applied NaN gradient) and localizes the defect
to the BACKWARD pass of a path shared by every parameter -- the SCF/solver or a
loss-channel gradient -- on one or more of HO / N2 / H / N2O.

This script supersedes ``dfs6311_nan_verify.py`` (which targeted CO/C/O, the
wrong group). Three roles:

MODE --local (Stage 0, structural reproduction, no cluster needed)
    Rebuild the ONE failing group standalone at the production configuration
    (6-311++G(3df,2pd), grid 3, DF, full_3 tail solver, polarized correlation,
    orientation lock 3e-5, deep_attn_3x16, seed 42) with pool geometries from
    ``scripts/script_data/haunschild_g2/g2_97.traj`` and run the DECOMPOSITION
    below on a fresh-init model (production weights are pretrained -- available
    only on the cluster -- so a non-repro here does NOT clear the group; it
    sends the hunt to --spec).

MODE --spec SPECPATH (Stage 1, exact replay on the cluster)
    Load the real spec (real pretrained checkpoint, real subset, real refs),
    replay epoch 0 exactly as ``_run_per_molecule_loop`` does (same seeded
    shuffle, same optimizer updates), and run the decomposition at the first
    step whose full gradient goes non-finite (or at --at-step). ``--pad``
    forces ``pad_group_to_common_shape`` the way ``_train_one_spec
    --pad-group`` does -- REQUIRED to mirror job 2091616, whose spec predates
    the padding field and got padding only from that CLI override.
    ``--horizon-epochs 1`` mirrors the smoke's optimizer schedule length
    (LR-neutral before the decay onset, exact with it).

DECOMPOSITION (names the molecule AND the channel)
    ``defused_value_and_grad`` computes the group gradient in three stages
    (pass-1 per-molecule energy trajectories; an assembly value_and_grad over
    the injected energy stack, whose model gradient carries only the
    vxc/density channels; pass-2 per-molecule VJPs seeded by the energy
    cotangents). Driving those stages separately localizes a non-finite
    gradient to (molecule x {energy-VJP, rho-channel, vxc-channel}).
    Channel isolation EXCLUDES a channel from the assembled total instead of
    zero-weighting it: a zero weight still traces the channel's backward and
    ``0 * NaN = NaN`` would defeat the ablation.

ABLATIONS (--ablate, one per process -- module constants bake at trace time)
    sym_break       SYM_BREAK_SHIFT 1e-8 -> 1e-6 (eigh degeneracy splitting)
    degeneracy_reg  DEGENERACY_REG 1e-10 -> 1e-8 (overlap conditioning)
    both_eigh       both of the above
    ec_tail         freeze the ec_base (PW92) gradient below
                    _NN_TAIL_THRESHOLD via the double-where stop_gradient
                    pattern (the metagga.py alpha tail freeze, applied to
                    correlation); forward values unchanged
    zeta_eps        _ZETA_BOUNDARY_EPS 1e-6 -> 1e-4 (spin-polarization
                    boundary, targets the H atom path)

A NaN that vanishes under exactly one ablation names the mechanism.
Exit codes: 0 = every evaluated gradient and every decomposition stage
finite; 3 = a non-finite value was found and decomposed (the informative
outcome) -- including a forced ``--at-step`` decomposition any of whose
stages is non-finite.
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import dataclasses
import faulthandler
import sys
import tempfile
import time

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

jax.config.update("jax_enable_x64", True)
faulthandler.enable()

from xcquinox.alec.config import ARCHITECTURES, TrainingSpec
from xcquinox.alec.solver import SolverConfig, SolverBackend, SolverMode
from xcquinox.alec import solver as solver_mod
from xcquinox.alec import solver_manual as solver_manual_mod
from xcquinox.alec import oneshot as oneshot_mod
from xcquinox.alec import models as alec_models
from xcquinox.alec.models import AlecGGAModel
from xcquinox.alec.train import (
    _build_model, _build_batch, make_loss, _training_groups,
    _build_group_loss_and_batch, _effective_channel_weights,
    build_optimizer, _trainable_params)
from xcquinox.alec.data import precompute_fixed_density_data
from xcquinox.alec.defused_grad import (
    defused_value_and_grad, _energy_trajectory_jit, _energy_trajectory_vjp_jit)
from xcquinox.alec.energy_override import set_energy_override, reset_energy_override
from xcquinox.alec.oneshot import scf_loss_tail_weights
from xcquinox.alec.padding import common_pad_target
from xcquinox.alec.training_points import build_dfs_pool_points
from xcquinox.alec.cluster.domain import bh76_meta_to_loss_dict
from xcquinox.alec.cluster.spec_builder import atoms_to_mol_spec
from xcquinox.alec._train_one_spec import _load_spec

BASIS = "6-311++G(3df,2pd)"
GRID_LEVEL = 3
LOCK = 3e-5
TARGET_RXN = "OH+N2_to_H+N2O"
EXPECTED_SPINS = {"HO": 1, "N2": 0, "H": 1, "N2O": 0}


def log(msg):
    print(f"[isolate +{time.time() - T0:8.1f}s] {msg}", flush=True)


T0 = time.time()


# ---------------------------------------------------------------------------
# Finiteness reporting
# ---------------------------------------------------------------------------

def all_finite(tree):
    return all(bool(jnp.all(jnp.isfinite(x)))
               for x in jax.tree_util.tree_leaves(tree)
               if eqx.is_inexact_array(x))


def leaf_report(tree):
    """(n_bad, n_total, first_bad_path, n_nan, n_inf) over inexact leaves."""
    n_bad = n_tot = 0
    first = None
    for path, leaf in jax.tree_util.tree_flatten_with_path(tree)[0]:
        if not eqx.is_inexact_array(leaf):
            continue
        n_tot += 1
        if bool(jnp.all(jnp.isfinite(leaf))):
            continue
        n_bad += 1
        if first is None:
            first = (jax.tree_util.keystr(path),
                     int(jnp.isnan(leaf).sum()), int(jnp.isinf(leaf).sum()))
    return n_bad, n_tot, first


def describe_grad(tag, tree):
    n_bad, n_tot, first = leaf_report(tree)
    if n_bad == 0:
        log(f"  {tag}: FINITE ({n_tot} leaves)")
        return True
    log(f"  {tag}: NON-FINITE {n_bad}/{n_tot} leaves, first {first[0]} "
        f"(n_nan={first[1]}, n_inf={first[2]})")
    return False


# ---------------------------------------------------------------------------
# Ablations (applied before any trace so patched constants bake in)
# ---------------------------------------------------------------------------

def apply_ablation(name):
    if name in (None, "none"):
        return
    if name in ("sym_break", "both_eigh"):
        for mod in (solver_mod, solver_manual_mod, oneshot_mod):
            if hasattr(mod, "SYM_BREAK_SHIFT"):
                setattr(mod, "SYM_BREAK_SHIFT", 1e-6)
        log("ablation: SYM_BREAK_SHIFT -> 1e-6")
    if name in ("degeneracy_reg", "both_eigh"):
        for mod in (solver_mod, solver_manual_mod, oneshot_mod):
            if hasattr(mod, "DEGENERACY_REG"):
                setattr(mod, "DEGENERACY_REG", 1e-8)
        log("ablation: DEGENERACY_REG -> 1e-8")
    if name == "ec_tail":
        orig = AlecGGAModel._ec_baseline
        thr = alec_models._NN_TAIL_THRESHOLD

        def _frozen_tail_ec_baseline(self, rho_safe, zeta):
            live = rho_safe > thr
            rho_in = jnp.where(live, rho_safe, jnp.ones_like(rho_safe))
            if zeta is None:
                z_in = None
            else:
                z_arr = jnp.asarray(zeta)
                z_in = jnp.where(live, z_arr, jnp.zeros_like(z_arr))
            ec_live = orig(self, rho_in, z_in)
            ec_full = orig(self, rho_safe, zeta)
            return jnp.where(live, ec_live, jax.lax.stop_gradient(ec_full))

        AlecGGAModel._ec_baseline = _frozen_tail_ec_baseline
        log(f"ablation: ec_base gradient frozen below rho={thr:g} "
            "(double-where stop_gradient; forward values unchanged)")
    if name == "zeta_eps":
        if not hasattr(oneshot_mod, "_ZETA_BOUNDARY_EPS"):
            raise SystemExit("zeta_eps ablation: oneshot._ZETA_BOUNDARY_EPS not found")
        oneshot_mod._ZETA_BOUNDARY_EPS = 1e-4
        log("ablation: _ZETA_BOUNDARY_EPS -> 1e-4")


# ---------------------------------------------------------------------------
# Decomposition instrument
# ---------------------------------------------------------------------------

def decompose(model, gloss, gbatch, cw, relative, mol_names, label):
    """Localize a non-finite group gradient to (molecule x stage x channel)."""
    sc = gloss.solver_config
    if scf_loss_tail_weights(sc) is None:
        raise SystemExit("decompose expects the tail-trajectory energy form "
                         "(full_3); scalar form not wired here")
    mol_data = gbatch["mol_data"]
    n_mol = len(mol_data)
    log(f"DECOMPOSITION of {label} ({n_mol} molecules: {', '.join(mol_names)})")

    # Stage A -- full de-fused gradient (production-equivalent reference).
    (loss_val, comps), grads = defused_value_and_grad(
        gloss, model, gbatch, cw, relative)
    log(f"  loss = {float(loss_val):.10e}; components: "
        + ", ".join(f"{k}={float(v):.3e}" for k, v in comps.items()))
    baseline_ok = describe_grad("full de-fused grads", grads)

    # Stage B -- pass-1 forward trajectories per molecule.
    E_list = []
    for i in range(n_mol):
        E = _energy_trajectory_jit(model, mol_data[i], sc)
        E_list.append(E)
        finite = bool(jnp.all(jnp.isfinite(E)))
        log(f"  pass-1 forward E[{mol_names[i]}]: "
            f"{'FINITE' if finite else 'NON-FINITE'} "
            f"(last={float(E[-1]):.8f})")
    E_stack = jnp.stack(E_list)

    # Stage C -- assembly gradient with true channel excision. A channel left
    # out of the total receives NO cotangent (it is aux only), unlike a zero
    # weight, which would still trace its backward (0 * NaN = NaN).
    arrays, static = eqx.partition(model, eqx.is_inexact_array)

    def assemble_subset(energy_stack, arrays_, include):
        m = eqx.combine(arrays_, static)
        token = set_energy_override({"trajectory": energy_stack})
        try:
            comps_ = gloss.compute_components(m, gbatch, relative=relative)
            total = jnp.zeros(())
            for key, value in comps_.items():
                if include is None or key in include:
                    total = total + cw.get(key, 1.0) * value
        finally:
            reset_energy_override(token)
        return total, comps_

    channel_verdicts = {}
    ct_energy = None
    for tag, include in (
            ("all channels", None),
            ("loss_BH76 only", ("loss_BH76",)),
            ("loss_AE only", ("loss_AE",)),
            ("loss_rho only", ("loss_rho",)),
            ("loss_vxc only", ("loss_vxc",))):
        (tot, _c), (ct_e, g_ne) = jax.value_and_grad(
            assemble_subset, argnums=(0, 1), has_aux=True)(
                E_stack, arrays, include)
        if include is None:
            ct_energy = ct_e
        ct_ok = bool(jnp.all(jnp.isfinite(ct_e)))
        ne_ok = all_finite(g_ne)
        channel_verdicts[tag] = (ct_ok, ne_ok)
        log(f"  assembly [{tag}]: total={float(tot):.6e} "
            f"ct_energy {'FINITE' if ct_ok else 'NON-FINITE'}, "
            f"grad_nonenergy {'FINITE' if ne_ok else 'NON-FINITE'}")
        if not ne_ok:
            describe_grad(f"    grad_nonenergy [{tag}]", g_ne)

    # Stage D -- pass-2 per-molecule energy VJPs (real seeds from the
    # all-channels assembly above, then unit seeds).
    mol_verdicts = {}
    for i in range(n_mol):
        gi = _energy_trajectory_vjp_jit(model, mol_data[i], sc, ct_energy[i])
        ok_real = describe_grad(f"pass-2 VJP [{mol_names[i]}] real seed", gi)
        gi_unit = _energy_trajectory_vjp_jit(
            model, mol_data[i], sc, jnp.ones_like(E_stack[i]))
        ok_unit = describe_grad(f"pass-2 VJP [{mol_names[i]}] unit seed", gi_unit)
        mol_verdicts[mol_names[i]] = (ok_real, ok_unit)

    log("VERDICT TABLE")
    log(f"  full grads finite: {baseline_ok}")
    for tag, (ct_ok, ne_ok) in channel_verdicts.items():
        log(f"  {tag:16s}: ct_energy={'ok' if ct_ok else 'BAD'} "
            f"grad_nonenergy={'ok' if ne_ok else 'BAD'}")
    for nm, (ok_real, ok_unit) in mol_verdicts.items():
        log(f"  molecule {nm:4s}: energy-VJP real={'ok' if ok_real else 'BAD'} "
            f"unit={'ok' if ok_unit else 'BAD'}")
    overall_ok = (baseline_ok
                  and all(ct and ne for ct, ne in channel_verdicts.values())
                  and all(r and u for r, u in mol_verdicts.values()))
    return overall_ok


# ---------------------------------------------------------------------------
# --local build (Stage 0)
# ---------------------------------------------------------------------------

def build_local(ref_scale_vxc, ref_scale_rho):
    log(f"building {TARGET_RXN} standalone at {BASIS}/grid{GRID_LEVEL} "
        f"(DF, lock={LOCK:g}, polarized correlation)")
    points = build_dfs_pool_points(bh76_mode="reaction_energy")
    tp = next(p for p in points if p.name == TARGET_RXN)
    wanted = list(dict.fromkeys(
        (*tp.metadata["reactants"], *tp.metadata["products"])))
    refs_dir = tempfile.mkdtemp(prefix="xcq_isolate_refs_")
    mol_specs = []
    for want in wanted:
        at = next(a for a in tp.species if a.info.get("name") == want)
        ms = atoms_to_mol_spec(at, basis=BASIS, grid_level=GRID_LEVEL,
                               external_refs_dir=refs_dir, name=want)
        if EXPECTED_SPINS.get(want) is not None and ms.spin != EXPECTED_SPINS[want]:
            raise SystemExit(f"spin mismatch for {want}: MoleculeSpec.spin="
                             f"{ms.spin}, expected {EXPECTED_SPINS[want]}")
        mol_specs.append(ms)
        log(f"  {want}: spin={ms.spin} charge={ms.charge} atoms={ms.atom!r}")
    rxn = bh76_meta_to_loss_dict(tp)
    log(f"  reaction: coeffs={rxn['coeffs']} e_rxn_ref={rxn['e_rxn_ref']:.6f} Ha")

    arch = dataclasses.replace(ARCHITECTURES["deep_attn_3x16"],
                               use_polarized_correlation=True,
                               zero_init_final_layer=False)
    sc = SolverConfig(
        backend=SolverBackend.MANUAL, mode=SolverMode.FULL, max_cycles=3,
        mixer_name="decaying_linear", mixer_kwargs=(("base", 0.3), ("floor", 0.3)),
        density_fit=True,
        scf_loss_use_tail=True, scf_loss_tail=10, scf_loss_weight_power=2.0,
        orientation_lock_strength=LOCK)
    spec = TrainingSpec.from_dicts(
        arch=arch, molecules=tuple(mol_specs),
        targets={m.name: 0.0 for m in mol_specs},
        atom_energies={"H": -0.5, "N": -54.6, "O": -75.1},
        loss_name="L5_gradnorm_vxc_step7",
        # aux_only_names: production forces the bh76 reaction species aux-only
        # (spec_builder.py:527-555) so the compound-AE channel is zeroed inside
        # the group; density_per_electron mirrors the production hyperparams.
        loss_kwargs={"bh76_reactions": [rxn], "solver_config": sc,
                     "aux_only_names": tuple(sorted(m.name for m in mol_specs)),
                     "density_per_electron": True},
        update_scheme="per_molecule", require_atom_anchors=False, n_steps=1,
        lr_start=1e-3, lr_end=1e-5, lr_decay_start=0.5, grad_clip=1.0,
        checkpoint_dir=None, seed=42, weight_decay=1e-4)
    model = _build_model(spec)

    req = {"cderi"}
    for d in spec.arch.materialize_descriptors():
        req |= set(d.required_mol_keys)
    md = []
    for m in spec.molecules:
        t_mol = time.time()
        d = dict(precompute_fixed_density_data(
            m, required_keys=tuple(sorted(req)),
            descriptors=spec.arch.materialize_descriptors(),
            orientation_lock_strength=LOCK))
        if d.get("vxc_pbe") is not None:
            d["vxc_ref"] = ref_scale_vxc * jnp.asarray(d["vxc_pbe"])
        if d.get("rho_grid") is not None:
            d["rho_ref_grid"] = ref_scale_rho * jnp.asarray(d["rho_grid"])
        md.append(d)
        log(f"  precompute {m.name}: {time.time() - t_mol:.1f}s "
            f"(n_grid={np.asarray(d['rho_grid']).shape[-1] if d.get('rho_grid') is not None else '?'})")
    batch = {"mol_data": tuple(md), "targets": spec.targets_dict,
             "atom_energies": spec.atom_energies_dict}
    cw = _effective_channel_weights(spec.channel_weights_dict)
    relative = (spec.loss_metric == "relative")
    groups = _training_groups(spec)
    prepared = []
    for g in groups:
        gloss, gbatch = _build_group_loss_and_batch(spec, g, batch)
        names = tuple(ms.name for ms in g["species"])
        prepared.append((g["label"], gloss, gbatch, names))
    return spec, model, batch, cw, relative, prepared


def run_local(args):
    spec, model, batch, cw, relative, prepared = build_local(
        args.ref_scale_vxc, args.ref_scale_rho)
    bh76 = [p for p in prepared if p[0].startswith("bh76:")]
    if not bh76:
        raise SystemExit("no bh76 group built -- check loss_kwargs wiring")
    label, gloss, gbatch, names = bh76[0]

    found_bad = not decompose(model, gloss, gbatch, cw, relative, names, label)

    if args.padded:
        pad_target = common_pad_target(batch["mol_data"])
        (lp, _cp), gp = defused_value_and_grad(
            gloss, model, gbatch, cw, relative, pad_target=pad_target)
        (lu, _cu), gu = defused_value_and_grad(
            gloss, model, gbatch, cw, relative, pad_target=None)
        log(f"padded cross-check: unpadded finite={all_finite(gu)} "
            f"padded finite={all_finite(gp)} "
            f"(loss u={float(lu):.8e} p={float(lp):.8e})")

    if not found_bad and args.steps > 0:
        log(f"fresh-init group is finite; drifting {args.steps} steps over "
            f"all {len(prepared)} groups to probe weight dependence")
        opt = build_optimizer(
            lr_start=spec.lr_start, lr_end=spec.lr_end,
            n_steps=max(1, args.steps * len(prepared)),
            lr_decay_start=spec.lr_decay_start, grad_clip=spec.grad_clip,
            weight_decay=spec.weight_decay)
        opt_state = opt.init(eqx.filter(model, eqx.is_array))
        for step in range(args.steps):
            for glabel, gl, gb, gnames in prepared:
                (lv, _c), grads = defused_value_and_grad(gl, model, gb, cw, relative)
                if (not bool(jnp.isfinite(lv))) or (not all_finite(grads)):
                    log(f">>> non-finite at drift step {step} group {glabel}")
                    decompose(model, gl, gb, cw, relative, gnames, glabel)
                    return 3
                updates, opt_state = opt.update(
                    grads, opt_state, _trainable_params(model))
                model = eqx.apply_updates(model, updates)
            log(f"  drift step {step}: loss={float(lv):.6e} (all groups finite)")
        log("no non-finite gradient in --local mode; the singularity likely "
            "needs the PRETRAINED weights -- run --spec on the cluster")
        return 0
    return 3 if found_bad else 0


# ---------------------------------------------------------------------------
# --spec exact replay (Stage 1)
# ---------------------------------------------------------------------------

def run_spec_replay(args):
    spec = _load_spec(args.spec)
    log(f"loaded spec: arch={spec.arch.name} n_molecules={len(spec.molecules)} "
        f"seed={spec.seed} pretrain={spec.pretrain_checkpoint!r}")
    model = _build_model(spec)
    loss = make_loss(
        spec.loss_name, molecules=spec.molecules,
        pbe_anchor_weight=spec.pbe_anchor_weight,
        pbe_anchor_sample=spec.pbe_anchor_sample,
        **spec.loss_kwargs_dict)
    log("building batch (full precompute; this is the long step)")
    batch = _build_batch(spec, loss)
    cw = _effective_channel_weights(spec.channel_weights_dict)
    relative = (spec.loss_metric == "relative")
    # --pad mirrors _train_one_spec's --pad-group CLI override: the July-15
    # spec files predate the pad_group_to_common_shape field, and job 2091616
    # was padded only through that override, never through the stored spec.
    want_pad = args.pad or getattr(spec, "pad_group_to_common_shape", False)
    pad_target = common_pad_target(batch["mol_data"]) if want_pad else None
    log(f"pad_target={'set' if pad_target is not None else 'None'} "
        f"(--pad={args.pad})")
    groups = _training_groups(spec)
    prepared = []
    for g in groups:
        gloss, gbatch = _build_group_loss_and_batch(spec, g, batch)
        names = tuple(ms.name for ms in g["species"])
        prepared.append((g["label"], gloss, gbatch, names))
    n_groups = len(groups)
    n_epochs = (args.horizon_epochs if args.horizon_epochs is not None
                else spec.n_steps)
    log(f"optimizer horizon: {n_epochs} epoch(s) x {n_groups} groups")
    optimizer = build_optimizer(
        lr_start=spec.lr_start, lr_end=spec.lr_end,
        n_steps=max(1, n_epochs * n_groups),
        lr_decay_start=spec.lr_decay_start, grad_clip=spec.grad_clip,
        weight_decay=spec.weight_decay)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    rng = np.random.RandomState(spec.seed)
    order = np.arange(n_groups)
    rng.shuffle(order)
    log("epoch-0 group order: "
        + ", ".join(f"{u}:{prepared[gi][0]}" for u, gi in enumerate(order[:8]))
        + (" ..." if n_groups > 8 else ""))

    for update, gi in enumerate(order):
        label, gloss, gbatch, names = prepared[gi]
        (loss_val, comps), grads = defused_value_and_grad(
            gloss, model, gbatch, cw, relative, pad_target=pad_target)
        bad = (not bool(jnp.isfinite(loss_val))) or (not all_finite(grads))
        log(f"step {update} {label}: loss={float(loss_val):.6e} "
            f"grads {'NON-FINITE' if bad else 'finite'}")
        if bad or (args.at_step is not None and update == args.at_step):
            # The decomposition re-runs this group's gradient WITHOUT padding.
            # Padding neutrality is proven only for ae:CO; on this group (job
            # 2091734) the padded gradient is NaN while the unpadded re-run is
            # finite, with a 0.219% forward-loss shift at identical parameters.
            # An all-ok verdict below therefore localizes the defect to the
            # padded computation path rather than exonerating the group.
            stages_ok = decompose(model, gloss, gbatch, cw, relative, names,
                                  label)
            return 3 if (bad or not stages_ok) else 0
        updates, opt_state = optimizer.update(
            grads, opt_state, _trainable_params(model))
        model = eqx.apply_updates(model, updates)
    log("epoch 0 completed with no non-finite gradient")
    return 0


# ---------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--local", action="store_true",
                      help="Stage 0: standalone group, fresh-init model")
    mode.add_argument("--spec", type=str, default=None,
                      help="Stage 1: exact replay of a real spec file")
    ap.add_argument("--ablate", default="none",
                    choices=["none", "sym_break", "degeneracy_reg", "both_eigh",
                             "ec_tail", "zeta_eps"])
    ap.add_argument("--at-step", type=int, default=None,
                    help="replay: force the decomposition at this step")
    ap.add_argument("--pad", action="store_true",
                    help="replay: force pad_group_to_common_shape (mirrors "
                         "_train_one_spec --pad-group; the July-15 specs "
                         "predate the field)")
    ap.add_argument("--horizon-epochs", type=int, default=None,
                    help="replay: optimizer schedule length in epochs "
                         "(default: the spec's n_steps; use 1 to mirror the "
                         "--smoke jobs)")
    ap.add_argument("--steps", type=int, default=25,
                    help="local: drift steps if the fresh-init group is finite")
    ap.add_argument("--ref-scale-vxc", type=float, default=1.0)
    ap.add_argument("--ref-scale-rho", type=float, default=1.0)
    ap.add_argument("--padded", action="store_true",
                    help="local: run the padded finiteness cross-check")
    args = ap.parse_args()

    log(f"mode={'local' if args.local else 'spec'} ablate={args.ablate}")
    apply_ablation(args.ablate)
    rc = run_local(args) if args.local else run_spec_replay(args)
    log(f"exit {rc} ({'non-finite decomposed' if rc == 3 else 'all finite'})")
    return rc


if __name__ == "__main__":
    # A scheduled job stage: its exit status is the scheduler's verdict on
    # the finiteness decomposition, so it leaves through the shared hard exit
    # (flush, then os._exit) rather than through interpreter teardown, which
    # aborted on the cluster after a completed pretrain stage (job 2134455).
    # Imported here rather than in the module body, since the helper is
    # needed only when the module is RUN.
    from xcquinox.alec.cluster._exit import run_and_exit
    run_and_exit(main)
