#!/usr/bin/env python
"""Held-out eval of the PRETRAINED (zero task training) network.

WHY. The SCAN baseline resolved at 4.89 kcal/mol combined over the full
216-reaction held-out pool -- better than PBE (11.78) and better than every
trained cell in the sweep. The meta-GGA architecture whose whole purpose is to
clone SCAN scores 26.62 at subset_size 2, i.e. 5.4x worse than its own target.
So the net has diverged from what it was pretrained to reproduce, and the open
question is WHERE:

  * the PRETRAINED net scores near SCAN  -> the clone is faithful and TASK
    TRAINING on 1-4 molecules destroys it. The large-alpha distortion measured
    in notebooks/analysis/mgga_diagnosis_evidence.py (F_x at s=4, alpha=100:
    SCAN 0.851, pretrained 0.662, trained 0.252) and the raw-alpha MLP input
    become the prime suspects, and the fix is training-side.
  * the PRETRAINED net scores near the trained cell -> the clone was never
    faithful in DEPLOYMENT despite matching SCAN's F_x to ~0.02 on the alpha<=1
    slice, so the problem is in the SCF/descriptor path, not training. F_c, the
    3-cycle solver, and the self-consistently recomputed alpha become suspects.

This is a re-eval, not a training run: it loads the pretrain checkpoints the
sweep already produced.

PROTOCOL IDENTITY. The eval is not reimplemented here. The spec, the resolved
config, the basis/grid, the held-out pool, the val/test slicing and the parallel
sharding all come from ``_eval_one_spec._run_held_out_eval`` -- the exact
function the sweep's own eval task calls. The ONLY difference from the trained
cell is which weights are in the model, which is what makes the two numbers
directly comparable.

Deliberately reusing the TRAINED spec (default spec 0034) rather than
synthesizing one: its ``molecules`` list drives the in-sample-overlap reaction
filtering, so the pretrained net is scored on exactly the reaction set the
trained cell was scored on. "In-sample" is meaningless for a net that never
trained, but matching the filter is what makes the comparison like-for-like.

Usage:
    python hpcjobs/dfs6311_pretrained_holdout.py <run_dir> [--spec 34]
        [--arch deep_mgga_3x16] [--subdir eval_holdout_pretrained]
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


def build_pretrained_model(training_spec, pretrain_dir: Path):
    """The spec's architecture with the PRETRAIN xnet/cnet weights loaded.

    Built from ``AlecGGAModel.from_arch`` exactly as ``load_trained_model``
    builds the trained skeleton, then each subnet is deserialized from the
    pretrain checkpoints the harness wrote. Raises if either file is missing --
    silently evaluating a randomly-initialized net would produce a number that
    looks like a result.
    """
    import equinox as eqx

    from xcquinox.alec.models import AlecGGAModel

    xnet_path = pretrain_dir / "xnet.eqx"
    cnet_path = pretrain_dir / "cnet.eqx"
    for p in (xnet_path, cnet_path):
        if not p.is_file():
            raise FileNotFoundError(f"pretrain checkpoint missing: {p}")
    model = AlecGGAModel.from_arch(training_spec.arch, seed=0)
    model = eqx.tree_at(
        lambda m: m.xnet, model,
        eqx.tree_deserialise_leaves(str(xnet_path), model.xnet))
    model = eqx.tree_at(
        lambda m: m.cnet, model,
        eqx.tree_deserialise_leaves(str(cnet_path), model.cnet))
    return model


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run_dir", help="materialized run directory")
    p.add_argument("--spec", type=int, default=34,
                   help="spec index supplying the arch/solver/holdout "
                        "definition (default: 34 = deep_mgga_3x16 ss=2, the "
                        "cell whose 26.62 this is compared against)")
    p.add_argument("--arch", default=None,
                   help="pretrain subdirectory to load (default: the spec's "
                        "own arch name)")
    p.add_argument("--subdir", default="eval_holdout_pretrained",
                   help="output subdir under the spec's checkpoint dir")
    args = p.parse_args(argv)

    # Route JAX threading/env exactly as the sweep's eval task does, BEFORE any
    # jax import -- the harness sets backend defaults at first import.
    from xcquinox.alec.cluster._eval_one_spec import (
        _checkpoint_dir, _load_spec, _log, _read_width, _route_jax_env,
        _run_held_out_eval, _spec_path,
    )
    _route_jax_env()

    from xcquinox.alec.cluster.grid_config import load_grid_config
    from xcquinox.alec.train import save_trained_checkpoint

    run_dir = os.path.abspath(args.run_dir)
    idx = args.spec
    width = _read_width(run_dir)
    spec_path = _spec_path(run_dir, idx, width)
    if not os.path.exists(spec_path):
        print(f"FATAL: spec file not found: {spec_path}")
        return 1
    training_spec = _load_spec(spec_path)
    cfg = load_grid_config(os.path.join(run_dir, "resolved_config.yaml"))
    checkpoint_dir = _checkpoint_dir(run_dir, idx, width)
    os.makedirs(checkpoint_dir, exist_ok=True)

    arch_name = args.arch or getattr(training_spec.arch, "name", None)
    if not arch_name:
        print("FATAL: could not resolve an architecture name")
        return 1
    pretrain_dir = Path(run_dir) / "pretrain" / arch_name
    _log(idx, f"pretrained held-out eval: arch={arch_name} "
              f"pretrain_dir={pretrain_dir}")

    try:
        model = build_pretrained_model(training_spec, pretrain_dir)
    except Exception as exc:  # noqa: BLE001 - report, do not half-run
        print(f"FATAL: could not build the pretrained model: "
              f"{type(exc).__name__}: {exc}")
        return 1

    # Serialize to a DISTINCT filename: the shard workers reload the checkpoint
    # by basename, and this must never be mistaken for (or overwrite) the
    # trained model.eqx sitting beside it.
    #
    # Through the training stage's own writer, so the model-class record is
    # written beside it: the shard workers read this file with
    # ``eval_holdout.load_trained_model``, which compares the record's class
    # with the spec's arch and refuses a checkpoint that carries none for any
    # class but the legacy one.
    model_path = os.path.join(checkpoint_dir, "model_pretrained.eqx")
    save_trained_checkpoint(model_path, model, training_spec.arch)
    _log(idx, f"wrote {model_path}")

    _run_held_out_eval(run_dir, idx, cfg, checkpoint_dir, model_path,
                       training_spec, holdout_subdir=args.subdir)

    out = Path(checkpoint_dir) / args.subdir / "per_reaction.json"
    if not out.is_file():
        fail = Path(checkpoint_dir) / args.subdir / "failure.json"
        print(f"FAILED: no per_reaction.json at {out}"
              + (f" (see {fail})" if fail.is_file() else ""))
        return 1

    # Report the headline number here so the log carries it without a pull.
    import json
    rows = json.loads(out.read_text())
    errs = [abs(r["error_nn_kcalmol"]) for r in rows
            if isinstance(r.get("error_nn_kcalmol"), (int, float))]
    pbe = [abs(r["error_pbe_kcalmol"]) for r in rows
           if isinstance(r.get("error_pbe_kcalmol"), (int, float))]
    if errs:
        print(f"[pretrained] held-out MAE = {sum(errs)/len(errs):.3f} kcal/mol "
              f"over {len(errs)} reactions "
              f"(PBE on the same set: {sum(pbe)/len(pbe):.3f})")
    return 0


if __name__ == "__main__":
    # A scheduled job stage: its exit status is the scheduler's verdict on
    # the pretrained held-out evaluation, so it leaves through the shared
    # hard exit (flush, then os._exit) rather than through interpreter
    # teardown, which aborted on the cluster after a completed pretrain stage
    # (job 2134455). Imported here rather than in the module body, since the
    # helper is needed only when the module is RUN.
    from xcquinox.alec.cluster._exit import run_and_exit
    run_and_exit(main)
