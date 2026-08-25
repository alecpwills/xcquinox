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
* fidelity certificate: every swept architecture must carry
  ``pretrain/<arch>/fidelity_certificate.json`` with ``verdict == "PASS"``.
  The file is read through ``fidelity``'s own reader, the one every gate
  uses, so a document that parses but states no certificate (``[]``, ``null``,
  a string, a number) and one the process may not open are reported here
  exactly as they are refused there. The certificate's ``enforced`` field
  releases the ON-NODE gates only and never releases this layer: a
  workflow-verification run must not be mistaken for a result, and the reason
  it recorded is copied into the failure text so such a run is
  distinguishable from an architecture whose physics did not certify.
  Required with the verdict: the ``arch`` and ``parent`` the certificate
  names, an ``identity`` block equal to ``fidelity.run_identity`` of the
  config -- compared over the UNION of the two key sets, so a field either
  side states alone is a disagreement -- an ``xcquinox_version`` equal to the
  manifest's, and ``checkpoint`` SHA-256 digests equal to the ``xnet.eqx`` /
  ``cnet.eqx`` present in the run. The certificate is what stands in for the
  spin-scaling oracles on the installed code, so a certificate from a
  different build, a different identity, a different architecture or a
  different pair of network files certifies nothing about this run. A
  manifest with no recorded version, and a certificate with neither recorded
  digests nor checkpoint files to compare them against, are warnings (they
  cannot be cross-checked); everything else is a failure.

Usage::

    python -m xcquinox.alec.cluster.validate_run <run_dir> [--config PATH]

``--config`` defaults to ``<run_dir>/resolved_config.yaml``.
"""
from __future__ import annotations

import dataclasses
import json
import os

from xcquinox.alec.cluster.grid_config import expand_grid, load_grid_config
from xcquinox.alec.cluster.fidelity import (CERTIFICATE_FILENAME,
                                            VERDICT_PASS,
                                            checkpoint_digest_findings,
                                            identity_mismatches,
                                            model_class_mismatches,
                                            parent_mismatch,
                                            read_certificate_status_in,
                                            show_identity)
from xcquinox.alec.cluster._eval_one_spec import (_load_spec, _read_width,
                                                  _spec_path)


def _enum_name(value) -> str:
    """Canonical comparable form for enum-or-string mode/policy values."""
    return str(value).split(".")[-1].upper()


def _check_solver(spec, named, inputs, arch_name, idx, failures):
    from xcquinox.alec.cluster.spec_builder import resolve_seed_xc
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
        # seed compared against the RESOLVED per-cell expectation
        # (inputs.seed_xc + arch rung), not the named solver, which carries
        # no seed field; getattr defaults keep pre-seeding pickles green.
        ("seed_source", getattr(sc, "seed_source", "pbe"),
         resolve_seed_xc(inputs, arch_name)),
        ("seed_cache_dir", getattr(sc, "seed_cache_dir", None),
         getattr(inputs, "seed_cache_dir", None)),
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
    from xcquinox.alec.config import apply_model_block, get_architecture

    failures: list[str] = []
    warnings: list[str] = []

    config_path = config_path or os.path.join(run_dir, "resolved_config.yaml")
    cfg = load_grid_config(config_path)
    cells = expand_grid(cfg)
    width = _read_width(run_dir)

    # The manifest's version is the run's code identity; a certificate must
    # have been produced by the same build, since the certificate is what
    # stands in for the spin-scaling oracles on the installed code.
    manifest_version = None
    try:
        with open(os.path.join(run_dir, "manifest.json")) as f:
            manifest_version = json.load(f).get("xcquinox_version")
    except (OSError, ValueError):
        manifest_version = None

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
        # The run's model block, as spec_builder applied it.
        model_block = getattr(cfg, "model", None)
        if model_block is not None:
            expected = apply_model_block(expected, model_block)
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
            _check_solver(spec, named, cfg.inputs, cell.arch, idx, failures)

    # --- pretrain metadata --------------------------------------------------
    for arch_name in sorted(set(cfg.sweep.arch)):
        # --- fidelity certificate ------------------------------------------
        # First in the loop body: the pretrain-metadata checks below ``continue``
        # on a missing metadata file, and a run may not skip the certificate
        # because an unrelated file is absent.
        #
        # The file is read through the certificate module's OWN reader rather
        # than a second one here, so this layer cannot disagree with the gates
        # about what counts as a certificate. A document that parses but is not
        # an object (``[]``, ``null``, a string, a number) and one the process
        # may not open are both UNREADABLE there, and neither may pass through
        # this loop producing no finding at all.
        #
        # ONE read: the status, the reason quoted in the report and the
        # document every check below is made against all come from the same
        # parse. Classifying the file and then re-opening it for its contents
        # would let a certificate rewritten between the two opens produce a
        # report that mixes them -- a reason taken from the file as it was
        # beside a finding taken from the file as it became -- describing no
        # document that ever existed on disk.
        pretrain_dir = os.path.join(run_dir, "pretrain", arch_name)
        status, status_reason, cert = read_certificate_status_in(pretrain_dir)
        if status == "MISSING":
            failures.append(
                f"pretrain/{arch_name}: no {CERTIFICATE_FILENAME} -- the "
                "architecture was never shown to reproduce its parent "
                "functional")
        elif cert is None:
            failures.append(
                f"pretrain/{arch_name}: {CERTIFICATE_FILENAME} is not "
                f"readable as a certificate ({status_reason})")
        else:
            if cert.get("verdict") != VERDICT_PASS:
                # ``enforced: false`` releases the ON-NODE gates only, so a
                # workflow-verification run reaches this layer with its FAIL
                # on record. The verdict is a failure regardless, but the
                # recorded waiver is carried into the report: a run refused
                # for a certificate that was deliberately not enforced is
                # otherwise indistinguishable from one whose physics did not
                # certify.
                waiver = ""
                if cert.get("enforced") is False:
                    tolerances = cert.get("tolerances")
                    waived_reason = (tolerances.get("override_reason")
                                     if isinstance(tolerances, dict) else None)
                    waiver = (
                        " -- the certificate records enforced=false "
                        f"(override_reason: {waived_reason!r}), which releases "
                        "the on-node gates only; a non-enforcing run is never "
                        "a result")
                failures.append(
                    f"pretrain/{arch_name}: fidelity certificate verdict "
                    f"{cert.get('verdict')!r}, expected {VERDICT_PASS!r} "
                    f"(summary: {cert.get('summary')}){waiver}")
            # The certificate is located by DIRECTORY; the architecture it
            # names must agree, or a file copied from another arch's pretrain
            # dir would certify this one.
            if cert.get("arch") != arch_name:
                failures.append(
                    f"pretrain/{arch_name}: certificate names arch "
                    f"{cert.get('arch')!r} -- it does not certify this "
                    "architecture")
            # The parent functional is a property of the arch's rung (PBE for
            # GGA, SCAN for meta-GGA); a certificate measured against the
            # other one bounds nothing here. The comparison is the certificate
            # module's own, so this layer and the pretrain stage's keep check
            # cannot come to different conclusions about one file.
            try:
                mismatch = parent_mismatch(arch_name, cert)
            except KeyError:
                failures.append(
                    f"pretrain/{arch_name}: arch is not in the registry, so "
                    "the parent functional its certificate must be measured "
                    "against cannot be resolved")
            else:
                if mismatch is not None:
                    recorded_parent, expected_parent = mismatch
                    failures.append(
                        f"pretrain/{arch_name}: certificate parent "
                        f"{recorded_parent!r}, but this architecture's rung "
                        f"is pretrained against {expected_parent!r}")
            for key, got, want in identity_mismatches(cfg, cert):
                failures.append(
                    f"pretrain/{arch_name}: certificate identity "
                    f"{key}={show_identity(got)} but the config says "
                    f"{show_identity(want)} -- the certificate was not "
                    "computed at this run's identity")
            # The model class the certified networks were built as (the
            # parent anchor, the descriptor coordinates): a static property
            # the checkpoint's leaves do not reveal, so the certificate's
            # record of it is compared with the run's model block. The
            # comparison is the certificate module's own, as the parent's is.
            for key, got, want in model_class_mismatches(cfg, cert):
                failures.append(
                    f"pretrain/{arch_name}: certificate records {key}="
                    f"{got!r} but the config builds {key}={want!r} -- the "
                    "certified networks are not the model class this run "
                    "trains")
            cert_version = cert.get("xcquinox_version")
            if manifest_version is None:
                warnings.append(
                    f"pretrain/{arch_name}: manifest.json records no "
                    "xcquinox_version, so the certificate's code version "
                    f"({cert_version!r}) cannot be cross-checked")
            elif cert_version != manifest_version:
                failures.append(
                    f"pretrain/{arch_name}: certificate xcquinox_version "
                    f"{cert_version!r} != manifest {manifest_version!r} "
                    "-- the certificate was produced by different code "
                    "than the run")
            # The verdict refers to two specific files. Comparing their
            # digests is what ties it to the networks the train stage loads:
            # a checkpoint rewritten (or re-pretrained) after certification is
            # not the one that was measured. The comparison is the certificate
            # module's own -- file names, payload keys and the five outcomes
            # -- so the two sides of it cannot drift apart and the pretrain
            # stage's keep check reaches the same verdict on the same pair.
            for kind, fname, key, want, measured in (
                    checkpoint_digest_findings(pretrain_dir, cert)):
                if kind == "unmeasured":
                    warnings.append(
                        f"pretrain/{arch_name}: the certificate records "
                        f"no {key} and no {fname} is present, so the "
                        "certified networks cannot be cross-checked "
                        "against the ones the run trains from")
                elif kind == "unrecorded":
                    failures.append(
                        f"pretrain/{arch_name}: {fname} is present but "
                        f"the certificate records no {key}, so the file "
                        "cannot be tied to the verdict")
                elif kind == "no_file":
                    failures.append(
                        f"pretrain/{arch_name}: the certificate measured "
                        f"{fname} (sha256 {str(want)[:12]}...) but no "
                        "such file is present in the run")
                elif kind == "unreadable":  # report, never crash the scan
                    failures.append(
                        f"pretrain/{arch_name}: {fname} could not be read "
                        f"to check it against the certificate ({measured})")
                else:
                    failures.append(
                        f"pretrain/{arch_name}: {fname} sha256 "
                        f"{str(measured)[:12]}... is not the file the "
                        f"certificate measured ({str(want)[:12]}...) -- "
                        "the checkpoint changed after it was certified")

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
        # The model class the networks were pretrained as. A metadata file
        # written before the fields existed records neither, which is the
        # unanchored legacy class; an anchored (or dfs-coordinate) run must
        # not accept such a checkpoint, since the anchor state is not visible
        # in the checkpoint's leaves.
        model_block = getattr(cfg, "model", None)
        want_anchor = bool(getattr(model_block, "parent_anchor", False))
        want_coords = str(getattr(model_block, "descriptor_coordinates",
                                  "legacy"))
        if bool(meta.get("parent_anchor", False)) != want_anchor:
            failures.append(
                f"pretrain/{arch_name}: parent_anchor="
                f"{meta.get('parent_anchor', False)}, config says "
                f"{want_anchor}")
        if str(meta.get("descriptor_coordinates", "legacy")) != want_coords:
            failures.append(
                f"pretrain/{arch_name}: descriptor_coordinates="
                f"{meta.get('descriptor_coordinates', 'legacy')!r}, config "
                f"says {want_coords!r}")
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
    # The stage's verdict is the status this process hands SLURM, and
    # JAX's atexit teardown can abort the interpreter AFTER main() has
    # returned it (cluster job 2134455: the pretrain worker logged
    # "pretrain SUCCEEDED" and then died in glibc's "corrupted size vs.
    # prev_size", rc -6, so the stage read as FAILED and the dependent
    # array never ran). run_and_exit flushes and leaves through os._exit,
    # so the status is the verdict. See xcquinox/alec/cluster/_exit.py.
    # Imported HERE rather than in the module body: several of these
    # modules pin what their import pulls in (``fidelity`` is held to a
    # whitelist of cheap readers so the on-node gates can read a
    # certificate without the training stack), and the helper is needed
    # only when the module is RUN.
    from xcquinox.alec.cluster._exit import run_and_exit
    run_and_exit(main)
