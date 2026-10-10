#!/usr/bin/env python
"""The Slim16 evaluation of a set of networks through the harness's converged channel.

The pretrained clones of the campaign arms and the trained cells of the v7 campaign
are evaluated on the Slim16 set at the identity of the cloning paper (def2-TZVP, grid
level 4) with that paper's metrics: the total-energy error against PBE and its WTMAD-2
(``eval_holdout.paper_wtmad2``). The evaluation itself is the harness's own,
``_eval_one_spec._run_held_out_eval`` under the converged channel (pyscfad, density
fitting, the PBE seed, DIIS to 1e-8 Ha), on a run directory this driver writes for the
purpose with one spec per network, so the shard workers reload the spec and the
checkpoint the way they do for a trained cell.

Density fitting is the one deviation from the paper's identity. Thirteen species of the
set exceed 300 basis functions at def2-TZVP, and the full-integral SCF of the harness
materializes the four-index tensor (222 GB at 408 functions), while the paper's own
engine cannot evaluate the two clones whose cusp descriptor the pyscfad backend
reassembles each cycle. The PBE reference energy of the precompute is computed with
full integrals, so a PBE-DF total energy per species is written beside the run
(``pbe_df.json``, through ``external_refs.run_scf_with_cache`` at the same auxiliary
basis) and the metrics compare the networks with PBE on one footing; the difference
between the two PBE energies measures the footing.

The comparators of the paper's tables, r2SCAN, B3LYP and wB97M-V, are computed on the
same footing through the same call and written beside the run as
``functionals_df.json``: libxc's r2SCAN (the meta-GGA rung), libxc's B3LYP (functional
402, the VWN-RPA correlation of the Gaussian form; ``b3lyp5`` is the VWN5 form) and
wB97M-V with its VV10 term, which PySCF applies by itself on a grid of its own (level
3, recorded in the file's identity block); the exchange of the two hybrids is density
fitted like the Coulomb term. Each entry records libxc's classification of the
functional (``vv10``, ``hybrid``, ``meta_gga``) and the libxc component names. A shard
whose mean field would not apply the VV10 term writes an error per species, never an
energy without the term. PBE is admitted to the comparator mode for the sanity pin
against ``pbe_df.json``; the readers never draw it as a comparator. A refused record stays
in the cache and is refused again at every rerun; the error names the cache file to remove.
The comparator file's ``species_slice`` is the union of the slices run; a functional's
coverage is its own species table.

Modes::

    python hpcjobs/slim16_eval.py prepare [--networks LEDGER] [--run-root ROOT]
    python hpcjobs/slim16_eval.py evaluate RUN_DIR IDX [--workers N]
    python hpcjobs/slim16_eval.py pbe-df RUN_DIR [--workers N]
    python hpcjobs/slim16_eval.py comparators-df RUN_DIR [--workers N] [--functionals LIST]

``prepare`` runs on the login node once every source checkpoint exists and prints the
run directory; the job script runs ``evaluate`` per array task and ``pbe-df`` then
``comparators-df`` after task 0, and ``slim16_tables.sbatch`` runs the two tables alone
on a run already evaluated. ``df-shard RUN_DIR NAMES_FILE OUT_FILE XC CACHE KEY`` is the
worker both tables launch; ``pbe-df-shard RUN_DIR NAMES_FILE OUT_FILE`` is its PBE
spelling.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from types import SimpleNamespace

POOL = "slim16"
BASIS = "def2-TZVP"
GRID_LEVEL = 4
ORIENTATION_LOCK_STRENGTH = 0.0
CHANNEL = "converged"
HOLDOUT_SUBDIR = "eval_holdout_converged"
WIDTH = 4
LOSS_NAME = "slim16_eval"
PBE_DF_FILE = "pbe_df.json"
PBE_DF_CACHE = "pbe_df_cache"
FUNCTIONALS_DF_FILE = "functionals_df.json"
FUNCTIONALS_DF_CACHE = "functionals_df_cache"
#: the comparators of the paper's tables, (libxc name, label), in the readers'
#: display order; b3lyp is libxc's functional 402, the VWN-RPA (Gaussian) form
COMPARATORS = (("r2scan", "r2SCAN"), ("b3lyp", "B3LYP"), ("wb97m-v", "wB97M-V"))
PBE_XC = "pbe"
HERE = Path(__file__).resolve().parent
DEFAULT_LEDGER = HERE / "ledgers" / "slim16_eval_networks.json"
DEFAULT_RUN_ROOT = "/gpfs/scratch/awills/xcquinox_runs/dfs_step8/slim16_eval"


def _log(message: str) -> None:
    print(f"[slim16-eval] {message}", flush=True)


def identity() -> dict:
    """The evaluation identity every output of the run records."""
    from xcquinox.pipeline.df_jk import default_auxbasis
    from xcquinox.pipeline.pyscf_determinism import REFERENCE_SMALL_RHO_CUTOFF
    return {
        "pool": POOL,
        "basis": BASIS,
        "grid_level": GRID_LEVEL,
        "density_fit": True,
        "auxbasis": default_auxbasis(BASIS),
        "orientation_lock_strength": ORIENTATION_LOCK_STRENGTH,
        "small_rho_cutoff": REFERENCE_SMALL_RHO_CUTOFF,
        "channel": CHANNEL,
        "holdout_subdir": HOLDOUT_SUBDIR,
    }


def base_solver_config():
    """The FULL-mode solver the specs carry. The converged override
    (``eval_holdout.converged_solver_config``) is applied at evaluation time
    by this driver and by the shard workers alike, from the channel name."""
    from xcquinox.pipeline.df_jk import default_auxbasis
    from xcquinox.pipeline.solver import FeaturePolicy, SolverConfig, SolverMode
    return SolverConfig(
        mode=SolverMode.FULL, max_cycles=100,
        feature_policy=FeaturePolicy.REASSEMBLE, density_fit=True,
        auxbasis=default_auxbasis(BASIS),
        orientation_lock_strength=ORIENTATION_LOCK_STRENGTH, seed_source="pbe")


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_ledger(path) -> list:
    """The network table: ``label``, ``kind`` (``pretrain`` with ``arch``, or
    ``trained`` with ``spec`` and ``checkpoint``) and ``run_dir`` per entry."""
    entries = json.loads(Path(path).read_text(encoding="utf-8"))
    for entry in entries:
        label = entry.get("label")
        missing = [key for key in ("label", "kind", "run_dir") if key not in entry]
        kind = entry.get("kind")
        if kind == "pretrain":
            missing += [key for key in ("arch",) if key not in entry]
        elif kind == "trained":
            missing += [key for key in ("spec", "checkpoint") if key not in entry]
        else:
            raise ValueError(f"network {label!r}: kind must be pretrain or trained")
        if missing:
            raise ValueError(f"network {label!r} lacks {missing}")
    labels = [entry["label"] for entry in entries]
    if len(set(labels)) != len(labels):
        raise ValueError(f"the ledger repeats a label: {labels}")
    return entries


def load_network(entry: dict):
    """``(arch, model, provenance)`` of one ledger entry, through the loaders
    the training and evaluation stages use for the same files.

    A pretrained clone is the run's architecture (``resolve_run_architecture``,
    the resolution the pretrain stage, the certificate and the spec builder
    share) with the pretrain directory's xnet/cnet leaves, after the class
    record check the training stage makes; the certificate's verdict is
    recorded, not gated. A trained cell is its own spec's architecture with
    the named checkpoint, through the evaluation's loader.
    """
    import equinox as eqx
    from xcquinox.pipeline.cluster._eval_one_spec import (
        _checkpoint_dir, _load_spec, _read_width, _spec_path,
    )
    from xcquinox.pipeline.cluster._pretrain import resolve_run_architecture
    from xcquinox.pipeline.cluster.fidelity import CERTIFICATE_FILENAME
    from xcquinox.pipeline.cluster.grid_config import load_grid_config
    from xcquinox.pipeline.config import get_architecture
    from xcquinox.pipeline.eval_holdout import load_trained_model
    from xcquinox.pipeline.models import AlecGGAModel
    from xcquinox.pipeline.networks import create_network_pair
    from xcquinox.pipeline.train import _require_matching_model_class

    run_dir = Path(entry["run_dir"])
    if entry["kind"] == "pretrain":
        cfg = load_grid_config(str(run_dir / "resolved_config.yaml"))
        arch = resolve_run_architecture(cfg, get_architecture(entry["arch"]))
        pretrain_dir = run_dir / "pretrain" / entry["arch"]
        xnet_path = pretrain_dir / "xnet.eqx"
        cnet_path = pretrain_dir / "cnet.eqx"
        for path in (xnet_path, cnet_path):
            if not path.is_file():
                raise FileNotFoundError(f"{entry['label']}: no {path}")
        _require_matching_model_class(str(pretrain_dir), arch)
        xnet_skeleton, cnet_skeleton = create_network_pair(arch, seed=0)
        xnet = eqx.tree_deserialise_leaves(str(xnet_path), xnet_skeleton)
        cnet = eqx.tree_deserialise_leaves(str(cnet_path), cnet_skeleton)
        model = AlecGGAModel.from_arch(arch, xnet=xnet, cnet=cnet)
        verdict = None
        certificate = pretrain_dir / CERTIFICATE_FILENAME
        if certificate.is_file():
            verdict = json.loads(certificate.read_text(encoding="utf-8")).get("verdict")
        provenance = {
            "sources": {"xnet.eqx": _sha256(xnet_path), "cnet.eqx": _sha256(cnet_path)},
            "certificate_verdict": verdict,
        }
    else:
        idx = int(entry["spec"])
        width = _read_width(str(run_dir))
        spec_path = Path(_spec_path(str(run_dir), idx, width))
        checkpoint = Path(_checkpoint_dir(str(run_dir), idx, width)) / entry["checkpoint"]
        for path in (spec_path, checkpoint):
            if not path.is_file():
                raise FileNotFoundError(f"{entry['label']}: no {path}")
        spec = _load_spec(str(spec_path))
        arch = spec.arch
        model = load_trained_model(spec, checkpoint)
        provenance = {
            "sources": {entry["checkpoint"]: _sha256(checkpoint)},
            "n_training_molecules": len(getattr(spec, "molecules", ()) or ()),
            "training_orientation_lock_strength": getattr(
                getattr(spec, "solver_config", None), "orientation_lock_strength", None),
        }
    provenance.update({
        "label": entry["label"], "kind": entry["kind"], "run_dir": str(run_dir),
        "arch_name": getattr(arch, "name", None),
    })
    return arch, model, provenance


def prepare(args) -> int:
    from xcquinox.pipeline.cluster.materialize import write_spec_atomic
    from xcquinox.pipeline.config import TrainingSpec
    from xcquinox.pipeline.train import save_trained_checkpoint

    entries = load_ledger(args.networks)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("run_%Y%m%dT%H%M%SZ")
    run_dir = Path(args.run_root) / "runs" / stamp
    if run_dir.exists():
        raise FileExistsError(f"{run_dir} exists")
    (run_dir / "specs").mkdir(parents=True)
    base = base_solver_config()
    networks = []
    for idx, entry in enumerate(entries):
        _log(f"network {idx}: {entry['label']}")
        arch, model, provenance = load_network(entry)
        checkpoint_dir = run_dir / "checkpoints" / f"spec_{idx:0{WIDTH}d}"
        checkpoint_dir.mkdir(parents=True)
        model_path = checkpoint_dir / "model.eqx"
        # through the training stage's writer, so the model-class record the
        # shard workers require stands beside the checkpoint
        save_trained_checkpoint(str(model_path), model, arch)
        spec = TrainingSpec(arch=arch, molecules=(), targets=(), atom_energies=(),
                            loss_name=LOSS_NAME, checkpoint_dir=str(checkpoint_dir),
                            solver_config=base)
        write_spec_atomic(spec, str(run_dir / "specs" / f"spec_{idx:0{WIDTH}d}.spec"))
        networks.append({"index": idx, "checkpoint": "model.eqx",
                         "checkpoint_sha256": _sha256(model_path), **provenance})
    manifest = {
        "width": WIDTH, "n_specs": len(entries), "kind": "slim16_eval",
        "created_utc": stamp, "identity": identity(),
        "solver_base": base.describe(), "networks": networks,
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _log(f"prepared {len(entries)} networks under {run_dir}")
    _log(f"submit with: sbatch --array=0-{len(entries) - 1} hpcjobs/slim16_eval.sbatch {run_dir}")
    print(run_dir, flush=True)
    return 0


def _network_label(run_dir: str, idx: int) -> str:
    try:
        manifest = json.loads(Path(run_dir, "manifest.json").read_text(encoding="utf-8"))
        return str(manifest["networks"][idx]["label"])
    except (OSError, ValueError, KeyError, IndexError, TypeError):
        return "?"


def _mean_abs(values) -> float:
    finite = [abs(v) for v in values if isinstance(v, (int, float)) and math.isfinite(v)]
    return sum(finite) / len(finite) if finite else float("nan")


def evaluate(args) -> int:
    import dataclasses
    from xcquinox.pipeline.cluster._eval_one_spec import (
        _checkpoint_dir, _load_spec, _read_width, _run_held_out_eval, _spec_path,
    )
    from xcquinox.pipeline.eval_holdout import converged_solver_config

    run_dir = os.path.abspath(args.run_dir)
    idx = int(args.idx)
    manifest = json.loads(Path(run_dir, "manifest.json").read_text(encoding="utf-8"))
    n_specs = int(manifest.get("n_specs", 0))
    if not 0 <= idx < n_specs:
        _log(f"FATAL: network index {idx} is outside the run's {n_specs} networks")
        return 1
    width = _read_width(run_dir)
    spec = _load_spec(_spec_path(run_dir, idx, width))
    # the same override the shard workers apply from the channel name
    spec = dataclasses.replace(spec, solver_config=converged_solver_config(spec.solver_config))
    checkpoint_dir = _checkpoint_dir(run_dir, idx, width)
    model_path = os.path.join(checkpoint_dir, "model.eqx")
    if not os.path.isfile(model_path):
        _log(f"FATAL: no {model_path}")
        return 1
    # the four attributes _run_held_out_eval reads of a configuration
    cfg = SimpleNamespace(
        inputs=SimpleNamespace(basis=BASIS, grid_level=GRID_LEVEL, held_out_pools=(POOL,)),
        cluster=SimpleNamespace(eval_workers=args.workers))
    _log(f"network {idx} ({_network_label(run_dir, idx)}): {POOL} at {BASIS} / "
         f"grid {GRID_LEVEL}, channel {CHANNEL}")
    _run_held_out_eval(run_dir, idx, cfg, checkpoint_dir, model_path, spec,
                       holdout_subdir=HOLDOUT_SUBDIR, channel=CHANNEL)
    out = Path(checkpoint_dir) / HOLDOUT_SUBDIR
    if not (out / "per_reaction.json").is_file():
        failure = out / "failure.json"
        _log(f"FAILED: no per_reaction.json under {out}"
             + (f" (see {failure})" if failure.is_file() else ""))
        return 1
    molecules = json.loads((out / "per_molecule.json").read_text(encoding="utf-8"))
    reactions = json.loads((out / "per_reaction.json").read_text(encoding="utf-8"))
    n_converged = sum(1 for row in molecules if row.get("scf_converged"))
    _log(f"done: {n_converged}/{len(molecules)} species converged; reaction MAE "
         f"against the references {_mean_abs(r.get('error_nn_kcalmol') for r in reactions):.3f} "
         f"kcal/mol (PBE {_mean_abs(r.get('error_pbe_kcalmol') for r in reactions):.3f}) "
         f"over {len(reactions)} reactions")
    return 0


def _df_table(run_dir: str, xc: str, workers, cache_name: str, energy_key: str,
              nlc_grid_level=None) -> tuple:
    """The density-fitted total energy under ``xc`` of every species of the
    pool (or of the harness's species slice) at the run's footing, sharded
    over the node: ``(species, names, species_slice, not_df, missing)``,
    ``species`` mapping each name to ``{energy_key, n_ao, reference_eri_path}``
    or ``{error}``. The shard files of the PBE table keep their unprefixed
    names under its own cache; the comparators share one cache and prefix
    theirs with the functional. ``nlc_grid_level``, when given, is the level
    the shards hold every record's VV10 stamp to."""
    from xcquinox.pipeline import parallel
    from xcquinox.pipeline.full_benchmark_pools import (
        load_held_out_pools, resolve_species_slice,
    )

    specs, _reactions = load_held_out_pools((POOL,), basis=BASIS, grid_level=GRID_LEVEL)
    # the harness's species-slice variable restricts the table the way it
    # restricts an evaluation (a smoke, or a rerun of a few species)
    species_slice = resolve_species_slice()
    names = sorted(species_slice) if species_slice else sorted(specs)
    unknown = sorted(n for n in names if n not in specs)
    if unknown:
        raise KeyError(f"the {POOL} pool carries none of {unknown}")
    total_cpus = parallel.detect_available_cpus()
    ladder = parallel.eval_worker_ladder(total_cpus, top=workers)
    n_workers, threads = ladder[0] if ladder else (1, max(1, total_cpus))
    n_workers = max(1, min(n_workers, len(names)))
    # the module rule the job script applies: a PySCF pool at min(allocation, 8)
    threads = min(threads, parallel.PYSCF_POOL_THREADS_MAX)
    shard_dir = Path(run_dir) / cache_name / "_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    prefix = "" if cache_name == PBE_DF_CACHE else f"{xc}_"
    jobs, out_files = [], []
    for si in range(n_workers):
        names_file = shard_dir / f"{prefix}names_s{si}.json"
        out_file = shard_dir / f"{prefix}shard_s{si}.json"
        names_file.write_text(json.dumps(names[si::n_workers]), encoding="utf-8")
        jobs.append(parallel.WorkerJob(
            name=f"{xc}_df_s{si}",
            cmd=[sys.executable, str(Path(__file__).resolve()), "df-shard",
                 run_dir, str(names_file), str(out_file), xc, cache_name, energy_key]
            + ([] if nlc_grid_level is None else ["--nlc-grid-level", str(int(nlc_grid_level))]),
            progress_file=None,
            thread_env=parallel._thread_env(threads, bound_worker=False),
            log_file=str(shard_dir / f"{prefix}worker_s{si}.log")))
        out_files.append(out_file)
    _log(f"{xc} (DF) over {len(names)} species: {n_workers} workers x {threads} threads")
    results = parallel.run_workers(jobs, max_parallel=n_workers)
    species = {}
    for result, out_file in zip(results, out_files):
        if getattr(result, "status", "failed") != "success" or not out_file.is_file():
            _log(f"shard {result.job.name} failed: {result.payload}")
            continue
        species.update(json.loads(out_file.read_text(encoding="utf-8")))
    missing = sorted(set(names) - set(species))
    not_df = sorted(name for name, value in species.items() if energy_key in value
                    and not str(value.get("reference_eri_path", "")).startswith("df"))
    if not_df:
        _log(f"WARNING: {len(not_df)} species built without density fitting: "
             f"{not_df[:5]}{' ...' if len(not_df) > 5 else ''}")
    for name in missing:
        species[name] = {"error": "the shard did not return"}
    return species, names, species_slice, not_df, missing


def pbe_df(args) -> int:
    """The PBE total energy of every species at the run's own footing (density
    fitting, the same auxiliary basis), sharded over the node."""
    run_dir = os.path.abspath(args.run_dir)
    species, names, species_slice, not_df, missing = _df_table(
        run_dir, PBE_XC, args.workers, PBE_DF_CACHE, "E_pbe_df")
    n_converged = sum(1 for value in species.values() if "E_pbe_df" in value)
    payload = {"identity": identity(), "n_species": len(names),
               "species_slice": list(species_slice) if species_slice else None,
               "n_converged": n_converged, "n_species_not_df": len(not_df),
               "species": species}
    Path(run_dir, PBE_DF_FILE).write_text(
        json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    _log(f"wrote {PBE_DF_FILE}: {n_converged}/{len(names)} species converged, "
         f"{len(missing)} not returned")
    return 1 if missing else 0


def _default_nlc_grid_level() -> int:
    """The level of the VV10 grid a pyscf Kohn-Sham mean field carries by
    default, which the cached SCF helper's mean fields inherit; the shard
    holds the stamp of every record to it, so the level the comparator file
    states is the level the SCFs ran with."""
    from pyscf import dft, gto
    return int(dft.RKS(gto.M(atom="He 0 0 0", basis="sto-3g", verbose=0)).nlcgrids.level)


def _libxc_components(xc: str) -> list:
    """The libxc functionals behind ``xc``, ``[{"id", "name"}, ...]``:
    ``parse_xc``'s ids, each with libxc's own identifier for it (the longest
    family-prefixed name of libxc's table, its aliases being the same name
    with an underscore dropped): ``HYB_GGA_XC_B3LYP`` (402) for b3lyp,
    ``HYB_MGGA_XC_WB97M_V`` (531) for wb97m-v. The table maps the names to
    numpy integers, hence the integral match."""
    import numbers

    from pyscf.dft import libxc
    _hyb, facs = libxc.parse_xc(xc)
    out = []
    for fid, _fac in facs:
        fid = int(fid)
        full = [name for name, value in libxc.XC.items()
                if isinstance(value, numbers.Integral) and int(value) == fid
                and name.startswith(("LDA_", "GGA_", "MGGA_", "HYB_"))]
        out.append({"id": fid, "name": max(full, key=lambda n: (len(n), n)) if full else None})
    return out


def _read_earlier_tables(path: Path):
    """The comparator file already beside the run, or None when there is
    none, when it cannot be read or when it is not of the form written
    here; the log says which."""
    if not path.is_file():
        return None
    try:
        earlier = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        _log(f"replaced {FUNCTIONALS_DF_FILE}: the earlier file is unreadable "
             f"({type(exc).__name__}: {exc})")
        return None
    functionals = earlier.get("functionals") if isinstance(earlier, dict) else None
    if not isinstance(functionals, dict) or not all(
            isinstance(entry, dict) and isinstance(entry.get("species"), dict)
            and all(isinstance(value, dict) for value in entry["species"].values())
            for entry in functionals.values()):
        _log(f"replaced {FUNCTIONALS_DF_FILE}: the earlier file is not of the form "
             "written here")
        return None
    return earlier


def _merge_species(kept: dict, fresh: dict) -> dict:
    """The functional entry ``kept`` with the species of ``fresh`` added or
    refreshed, the counts recomputed over the union. A species whose rerun
    failed keeps the energy on record: a failure of the rerun (a lost shard,
    an SCF that did not converge this time) erases nothing."""
    species = dict(kept.get("species", {}))
    for name, value in fresh["species"].items():
        if "E_df" in value or "E_df" not in species.get(name, {}):
            species[name] = value
    with_energy = [value for value in species.values() if "E_df" in value]
    return {**fresh, "species": species, "n_species": len(species),
            "n_converged": len(with_energy),
            "n_species_not_df": sum(
                1 for value in with_energy
                if not str(value.get("reference_eri_path", "")).startswith("df"))}


def _pool_digest(names) -> str:
    """The sha256 of the pool's sorted species names: a pool that changes
    its members, with or without its size, changes the identity block."""
    return hashlib.sha256("\n".join(sorted(names)).encode("utf-8")).hexdigest()


def comparators_df(args) -> int:
    """The comparator tables beside the run: one density-fitted total energy
    per species under each functional of ``COMPARATORS`` (or of
    ``--functionals``, which may name pbe for the sanity pin), through the
    call the PBE-DF table runs through, written as ``FUNCTIONALS_DF_FILE``
    with an identity block: the run's identity, the level of the VV10 grid
    and the size of the pool. An earlier file of the same identity block is
    merged into at the species level: a functional not run again is kept,
    a functional run again has the species of this run added or refreshed
    (a repair of a few species through the harness's slice), and the file's
    slice is the union, or None once either side covers the pool. An earlier
    file of another identity block, or one that cannot be read, is replaced.
    The file is written through a temporary name, so an interrupted run
    leaves the earlier file in place."""
    from pyscf.dft import libxc

    from xcquinox.pipeline.full_benchmark_pools import load_held_out_pools

    run_dir = os.path.abspath(args.run_dir)
    labels = dict(COMPARATORS)
    labels[PBE_XC] = "PBE"
    asked = ([k.strip() for k in args.functionals.split(",") if k.strip()]
             if args.functionals else [k for k, _label in COMPARATORS])
    asked = list(dict.fromkeys(asked))
    if not asked:
        _log(f"FATAL: no functional named (the job's: {', '.join(labels)})")
        return 2
    unknown = [k for k in asked if k not in labels]
    if unknown:
        _log(f"FATAL: not a functional of this job: {', '.join(unknown)} "
             f"(the job's: {', '.join(labels)})")
        return 2
    nlc_level = _default_nlc_grid_level()
    pool, _reactions = load_held_out_pools((POOL,), basis=BASIS, grid_level=GRID_LEVEL)
    identity_block = {**identity(), "nlc_grid_level": nlc_level,
                      "n_pool_species": len(pool), "pool_species_sha256": _pool_digest(pool)}
    functionals, failed, species_slice = {}, False, None
    for xc in asked:
        species, names, species_slice, not_df, missing = _df_table(
            run_dir, xc, args.workers, FUNCTIONALS_DF_CACHE, "E_df",
            nlc_grid_level=nlc_level)
        failed = failed or bool(missing)
        n_converged = sum(1 for value in species.values() if "E_df" in value)
        functionals[xc] = {
            "label": labels[xc], "xc": xc, "libxc": _libxc_components(xc),
            "vv10": bool(libxc.is_nlc(xc)), "hybrid": bool(libxc.is_hybrid_xc(xc)),
            "meta_gga": bool(libxc.is_meta_gga(xc)),
            "n_species": len(names), "n_converged": n_converged,
            "n_species_not_df": len(not_df), "species": species}
        _log(f"{labels[xc]}: {n_converged}/{len(names)} species converged, "
             f"{len(missing)} not returned")
    slice_list = sorted(species_slice) if species_slice else None
    path = Path(run_dir, FUNCTIONALS_DF_FILE)
    earlier = _read_earlier_tables(path)
    merged = dict(functionals)
    if earlier is not None:
        if earlier.get("identity") == identity_block:
            kept = {k: v for k, v in earlier["functionals"].items() if k not in functionals}
            merged = {**kept, **{xc: (_merge_species(earlier["functionals"][xc], entry)
                                      if xc in earlier["functionals"] else entry)
                                 for xc, entry in functionals.items()}}
            earlier_slice = earlier.get("species_slice")
            slice_list = (None if earlier_slice is None or slice_list is None
                          else sorted(set(earlier_slice) | set(slice_list)))
            _log(f"merged into {FUNCTIONALS_DF_FILE}: kept {', '.join(sorted(kept)) or 'nothing'}"
                 f"; added or refreshed {', '.join(asked)} on {len(names)} species")
        else:
            _log(f"replaced {FUNCTIONALS_DF_FILE}: its identity block differs")
    payload = {"identity": identity_block, "species_slice": slice_list,
               "functionals": merged}
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    _log(f"wrote {FUNCTIONALS_DF_FILE}: {', '.join(asked)}")
    return 1 if failed else 0


def df_shard(args) -> int:
    """The worker of both tables: the density-fitted total energy under
    ``xc`` of each species of ``names_file`` through the cached SCF helper
    (the cache under ``cache`` beside the run), written to ``out_file`` as
    ``{name: {key: E, n_ao, reference_eri_path}}`` or ``{name: {error}}``.
    The helper stamps each record with whether its mean field applied a
    VV10 term and with the level of its VV10 grid: a functional with a VV10
    term whose record is not stamped as applied gets an error, never an
    energy without the term, and so does a record whose level is not the
    one the table states (``--nlc-grid-level``)."""
    from pyscf.dft import libxc

    from xcquinox.pipeline.benchmark_refs import _mol_spec_to_atoms
    from xcquinox.pipeline.df_jk import default_auxbasis
    from xcquinox.pipeline.external_refs import (
        SpeciesEntry, _intermediate_cache_name, run_scf_with_cache,
    )
    from xcquinox.pipeline.full_benchmark_pools import load_held_out_pools

    run_dir = os.path.abspath(args.run_dir)
    names = json.loads(Path(args.names_file).read_text(encoding="utf-8"))
    xc, key = args.xc, args.key
    needs_vv10 = bool(libxc.is_nlc(xc))
    level_stated = getattr(args, "nlc_grid_level", None)

    def _cached_record(name):
        """The cache file a refused record lives in: removing it recomputes
        the species at the next run, the only way past a refusal."""
        return os.path.join(cache_dir, "_intermediates", _intermediate_cache_name(
            name, grid_level=GRID_LEVEL, basis=BASIS, density_fit=True, kind="scf",
            orientation_lock_strength=ORIENTATION_LOCK_STRENGTH, xc=xc))
    specs, _reactions = load_held_out_pools((POOL,), basis=BASIS, grid_level=GRID_LEVEL)
    cache_dir = os.path.join(run_dir, args.cache)
    auxbasis = default_auxbasis(BASIS)
    out = {}
    for i, name in enumerate(names, start=1):
        ms = specs[name]
        entry = SpeciesEntry(name=name, charge=int(ms.charge), spin=int(ms.spin), source=POOL)
        try:
            record = run_scf_with_cache(
                entry, _mol_spec_to_atoms(ms), cache_dir=cache_dir, basis=BASIS,
                grid_level=GRID_LEVEL, density_fit=True, auxbasis=auxbasis,
                orientation_lock_strength=ORIENTATION_LOCK_STRENGTH, xc=xc,
                # the network SCF applies density fitting to every species
                density_fit_empty_channel=True)
        except Exception as exc:  # noqa: BLE001 -- recorded per species, not fatal
            out[name] = {"error": f"{type(exc).__name__}: {exc}"}
            _log(f"[{i}/{len(names)}] {name}: FAILED {type(exc).__name__}: {exc}")
            continue
        applied = record.get("reference_nlc_applied")
        level = record.get("reference_nlc_grid_level")
        if needs_vv10 and applied is not True:
            out[name] = {"error": f"the mean field did not apply the VV10 (nlc) term of {xc}"
                         + (" (the record carries no VV10 stamp)" if applied is None else "")
                         + f"; remove {_cached_record(name)} to recompute"}
            _log(f"[{i}/{len(names)}] {name}: REFUSED {out[name]['error']}")
            continue
        # the level matters to a functional with a VV10 term alone
        if (needs_vv10 and level_stated is not None and level is not None
                and int(level) != int(level_stated)):
            out[name] = {"error": f"the VV10 grid level of the record ({int(level)}) is "
                                  f"not the table's ({int(level_stated)}); remove "
                                  f"{_cached_record(name)} to recompute"}
            _log(f"[{i}/{len(names)}] {name}: REFUSED {out[name]['error']}")
            continue
        out[name] = {key: float(record["e_tot"]), "n_ao": int(record["n_ao"]),
                     "reference_eri_path": record.get("reference_eri_path")}
        _log(f"[{i}/{len(names)}] {name}: {key} = {record['e_tot']:.8f} Ha ({xc})")
    Path(args.out_file).write_text(json.dumps(out, indent=1, sort_keys=True),
                                   encoding="utf-8")
    # the last stdout line is the result run_workers reads
    print(json.dumps({"status": "success", "n": len(out), "out_file": args.out_file}),
          flush=True)
    return 0


def pbe_df_shard(args) -> int:
    """``df-shard`` with PBE, the PBE-DF cache and the PBE-DF energy key."""
    args.xc, args.cache, args.key = PBE_XC, PBE_DF_CACHE, "E_pbe_df"
    return df_shard(args)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    modes = parser.add_subparsers(dest="mode", required=True)
    p = modes.add_parser("prepare", help="write the evaluation run directory")
    p.add_argument("--networks", default=str(DEFAULT_LEDGER))
    p.add_argument("--run-root", default=DEFAULT_RUN_ROOT)
    e = modes.add_parser("evaluate", help="one network through the converged channel")
    e.add_argument("run_dir")
    e.add_argument("idx", type=int)
    e.add_argument("--workers", type=int, default=None)
    d = modes.add_parser("pbe-df", help="the PBE-DF table of the set")
    d.add_argument("run_dir")
    d.add_argument("--workers", type=int, default=24,
                   help="shard workers; 24 keeps the density-fitted SCFs of the "
                        "largest species inside the node memory")
    c = modes.add_parser("comparators-df",
                         help="the comparator tables of the set (r2SCAN, B3LYP, wB97M-V)")
    c.add_argument("run_dir")
    c.add_argument("--workers", type=int, default=24,
                   help="shard workers, the PBE-DF table's rule")
    c.add_argument("--functionals", default=None,
                   help="comma-separated functionals of the job's list to run "
                        "(default: all three); pbe is admitted for the sanity pin")
    s = modes.add_parser("pbe-df-shard")
    s.add_argument("run_dir")
    s.add_argument("names_file")
    s.add_argument("out_file")
    g = modes.add_parser("df-shard")
    for name in ("run_dir", "names_file", "out_file", "xc", "cache", "key"):
        g.add_argument(name)
    g.add_argument("--nlc-grid-level", type=int, default=None,
                   help="the VV10 grid level the table states; a record stamped "
                        "with another level is refused")
    args = parser.parse_args(argv)
    # before any import that pulls jax in
    from xcquinox.pipeline.cluster._eval_one_spec import _route_jax_env
    _route_jax_env()
    handlers = {"prepare": prepare, "evaluate": evaluate,
                "pbe-df": pbe_df, "comparators-df": comparators_df,
                "pbe-df-shard": pbe_df_shard, "df-shard": df_shard}
    return handlers[args.mode](args)


if __name__ == "__main__":
    # A scheduled job stage: its exit status is the scheduler's verdict, so it
    # leaves through the shared hard exit rather than interpreter teardown.
    from xcquinox.pipeline.cluster._exit import run_and_exit
    run_and_exit(main)
