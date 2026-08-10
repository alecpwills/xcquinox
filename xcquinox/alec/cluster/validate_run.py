"""Validate a materialized sweep run directory against its resolved config.

Reads the artifacts a run actually executes -- every ``specs/spec_NNNN.spec``,
the per-architecture ``pretrain/<arch>/pretrain_metadata.json`` -- and asserts
them against ``resolved_config.yaml`` and the architecture registry. The need
is recorded in HISTORY (2026-08-06): a scoping error was made by reading the
registry DEFAULT of ``use_polarized_correlation`` instead of the built specs,
which carry the sweep-level override; conclusions about a run must come from
the run's own artifacts, not from defaults.

Report-only by design: disagreements are listed and the exit status is
nonzero, but nothing is ever modified.

Checks
------
* spec-file completeness: one file per grid cell, indices contiguous, count
  equal to the sweep-axes product, enumerated by the SAME ``expand_grid`` the
  submit path uses (a cell's list position IS its array task index).
* per spec: the architecture resolves in the registry and equals the registry
  entry with the sweep's ``use_polarized_correlation`` override applied -- so
  name, depth/nodes, descriptor set, ``meta_gga`` and the derived
  ``n_extra_features`` all match; the spec's arch name matches the cell the
  index maps to; hyperparameters (``n_steps``, ``seed``, ``update_scheme``)
  match the config; every molecule carries the configured basis and grid
  level; external reference paths live under the configured reference
  directories; the solver config matches the named solver block field by
  field, plus ``density_fit`` / ``auxbasis`` from the inputs.
* pretrain metadata: ``use_polarized_correlation`` must match the config.
  ``meta_gga`` / ``n_extra_features`` / ``pretrain_steps``, where present,
  must match the registry and config; absence is reported as a legacy warning
  (files written before those keys existed cannot be cross-checked and should
  be regenerated when convenient). ``pretrain_steps`` has been written since
  the writer existed, so it is checked even on legacy files.

Usage::

    python -m xcquinox.alec.cluster.validate_run <run_dir> [--config PATH]

``--config`` defaults to ``<run_dir>/resolved_config.yaml``.
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys

from xcquinox.alec.cluster.grid_config import expand_grid, load_grid_config
from xcquinox.alec.cluster._eval_one_spec import (_load_spec, _read_width,
                                                  _spec_path)


def _enum_name(value) -> str:
    """Canonical comparable form for enum-or-string mode/policy values."""
    return str(value).split(".")[-1].upper()


def _check_solver(spec, named, inputs, idx, failures):
    sc = spec.solver_config
    if sc is None:
        failures.append(f"spec {idx}: solver_config is None")
        return
    checks = [
        ("mode", _enum_name(sc.mode), _enum_name(named.mode)),
        ("max_cycles", int(sc.max_cycles), int(named.max_cycles)),
        ("scf_grad_checkpoint", bool(getattr(sc, "scf_grad_checkpoint", False)),
         bool(named.scf_grad_checkpoint)),
        ("density_fit", bool(getattr(sc, "density_fit", False)),
         bool(inputs.density_fit)),
        ("auxbasis", getattr(sc, "auxbasis", None), inputs.auxbasis),
    ]
    if named.feature_policy is not None:
        checks.append(("feature_policy",
                       _enum_name(sc.effective_feature_policy),
                       _enum_name(named.feature_policy)))
    if named.mixer_name is not None:
        checks.append(("mixer_name", getattr(sc, "mixer_name", None),
                       named.mixer_name))
    for field, got, want in checks:
        if got != want:
            failures.append(
                f"spec {idx}: solver.{field} = {got!r}, config says {want!r}")


def validate_run(run_dir: str, config_path: str | None = None):
    """Return ``(failures, warnings, n_specs)`` for the run at ``run_dir``."""
    from xcquinox.alec.config import get_architecture

    failures: list[str] = []
    warnings: list[str] = []

    config_path = config_path or os.path.join(run_dir, "resolved_config.yaml")
    cfg = load_grid_config(config_path)
    cells = expand_grid(cfg)
    width = _read_width(run_dir)

    # --- spec-file completeness --------------------------------------------
    spec_dir = os.path.join(run_dir, "specs")
    present = sorted(f for f in os.listdir(spec_dir) if f.endswith(".spec")) \
        if os.path.isdir(spec_dir) else []
    if len(present) != len(cells):
        failures.append(
            f"spec count {len(present)} != sweep-axes product {len(cells)}")
    missing = [i for i in range(len(cells))
               if not os.path.isfile(_spec_path(run_dir, i, width))]
    if missing:
        failures.append(f"missing spec indices: {missing[:10]}"
                        + (" ..." if len(missing) > 10 else ""))

    # --- reference directories the specs may point into --------------------
    ref_dirs = tuple(d for d in (
        getattr(cfg.inputs, "external_refs_dir", None),
        getattr(cfg.inputs, "val_refs_dir", None),
        getattr(cfg.inputs, "benchmark_refs_dir", None),
    ) if d)

    # --- per-spec checks ----------------------------------------------------
    n_checked = 0
    for idx, cell in enumerate(cells):
        path = _spec_path(run_dir, idx, width)
        if not os.path.isfile(path):
            continue
        try:
            spec = _load_spec(path)
        except Exception as exc:  # noqa: BLE001 -- report, never crash the scan
            failures.append(f"spec {idx}: failed to load ({exc!r})")
            continue
        n_checked += 1
        arch = spec.arch

        if arch.name != cell.arch:
            failures.append(
                f"spec {idx}: arch {arch.name!r} but grid cell {idx} is "
                f"{cell.arch!r} -- the index->cell mapping is broken")
        try:
            expected = get_architecture(arch.name)
        except KeyError:
            failures.append(f"spec {idx}: arch {arch.name!r} not in registry")
            continue
        # One-directional, matching spec_builder: the sweep-level flag OVERRIDES
        # to True but never forces False onto an arch that defaults polarized.
        if cfg.use_polarized_correlation:
            expected = dataclasses.replace(
                expected, use_polarized_correlation=True)
        if arch != expected:
            diffs = [f.name for f in dataclasses.fields(arch)
                     if getattr(arch, f.name) != getattr(expected, f.name)]
            failures.append(
                f"spec {idx}: arch differs from registry+override in {diffs}")

        if cfg.use_polarized_correlation and not arch.use_polarized_correlation:
            failures.append(
                f"spec {idx}: use_polarized_correlation="
                f"{arch.use_polarized_correlation}, config says "
                f"{cfg.use_polarized_correlation}")

        hp = cfg.hyperparams
        for field, got, want in (
                ("n_steps", int(spec.n_steps), int(hp.n_steps)),
                ("seed", int(spec.seed), int(hp.seed)),
                ("update_scheme", spec.update_scheme,
                 getattr(hp, "update_scheme", spec.update_scheme))):
            if got != want:
                failures.append(
                    f"spec {idx}: {field} = {got!r}, config says {want!r}")

        for mol in spec.molecules:
            if mol.basis != cfg.inputs.basis:
                failures.append(
                    f"spec {idx}: molecule {mol.name!r} basis {mol.basis!r} "
                    f"!= configured {cfg.inputs.basis!r}")
            got_gl = getattr(mol, "grid_level", None)
            if got_gl is not None and int(got_gl) != int(cfg.inputs.grid_level):
                failures.append(
                    f"spec {idx}: molecule {mol.name!r} grid_level {got_gl} "
                    f"!= configured {cfg.inputs.grid_level}")
            ext = getattr(mol, "external_data_path", None)
            if ext and ref_dirs and not any(
                    str(ext).startswith(d) for d in ref_dirs):
                failures.append(
                    f"spec {idx}: molecule {mol.name!r} reference "
                    f"{ext!r} is outside the configured reference dirs")

        named = cfg.solvers.get(cell.solver)
        if named is None:
            failures.append(
                f"spec {idx}: cell solver {cell.solver!r} not in config "
                f"solvers {sorted(cfg.solvers)}")
        else:
            _check_solver(spec, named, cfg.inputs, idx, failures)

    # --- pretrain metadata --------------------------------------------------
    for arch_name in sorted(set(cfg.sweep.arch)):
        meta_path = os.path.join(run_dir, "pretrain", arch_name,
                                 "pretrain_metadata.json")
        if not os.path.isfile(meta_path):
            warnings.append(f"pretrain/{arch_name}: no pretrain_metadata.json")
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        if bool(meta.get("use_polarized_correlation")) != bool(
                cfg.use_polarized_correlation):
            failures.append(
                f"pretrain/{arch_name}: use_polarized_correlation="
                f"{meta.get('use_polarized_correlation')}, config says "
                f"{cfg.use_polarized_correlation}")
        try:
            reg = get_architecture(arch_name)
        except KeyError:
            failures.append(f"pretrain/{arch_name}: arch not in registry")
            continue
        provenance = (("meta_gga", bool(reg.meta_gga)),
                      ("n_extra_features", int(reg.n_extra_features)),
                      # The (s, alpha) mesh is appended exactly for meta-GGA
                      # archs whose descriptor set is (metagga,) -- derivable
                      # from the registry, so a meta-GGA checkpoint trained
                      # without it (the underdetermined-alpha clone) fails
                      # here instead of surfacing as bad held-out numbers.
                      ("pretrain_mesh", bool(
                          reg.meta_gga
                          and tuple(d.name for d in reg.descriptors)
                          == ("metagga",))),
                      # The step count has always been written as
                      # "pretrain_steps", so it is checkable on legacy files
                      # that predate the shape keys.
                      ("pretrain_steps", int(cfg.pretrain.n_steps)
                       if getattr(cfg.pretrain, "n_steps", None) is not None
                       else None))
        for key, want in provenance:
            got = meta.get(key)
            if got is None:
                warnings.append(
                    f"pretrain/{arch_name}: metadata lacks {key!r} (written "
                    f"before the provenance keys existed; the checkpoint's "
                    f"architecture shape cannot be cross-checked)")
            elif want is not None and got != want:
                failures.append(
                    f"pretrain/{arch_name}: metadata {key} = {got!r}, "
                    f"expected {want!r}")

    return failures, warnings, n_checked


def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("run_dir")
    p.add_argument("--config", default=None,
                   help="resolved config path (default: "
                        "<run_dir>/resolved_config.yaml)")
    args = p.parse_args(argv)
    failures, warnings, n = validate_run(args.run_dir, args.config)
    print(f"[validate_run] checked {n} spec(s) under {args.run_dir}")
    for w in warnings:
        print(f"[validate_run] WARNING: {w}")
    for f in failures:
        print(f"[validate_run] FAIL: {f}")
    if failures:
        print(f"[validate_run] {len(failures)} failure(s), "
              f"{len(warnings)} warning(s)")
        return 1
    print(f"[validate_run] clean ({len(warnings)} warning(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
