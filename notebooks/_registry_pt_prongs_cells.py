"""The cells of the two registry proton-transfer prong notebooks.

``build_cells(prong)`` returns the cell list of ``registry_pt_prong_A.ipynb``
(per-group updates, twelve optimizer steps per epoch, the DFS scheme) or
``registry_pt_prong_B.ipynb`` (gradient accumulation, the twelve group
gradients averaged into one step per epoch); ``write_notebook`` writes it.
The thin builders ``_build_registry_pt_prong_{A,B}_notebook.py`` call these,
one tracked product each (``notebooks/test_notebook_hygiene.py``).

The cells that are unchanged from the DFS-parity tutorial (imports, corpus
staging, the relaxed CCSD(T) reference driver, the proton-transfer path
geometries and their references) are taken from
``_build_registry_pt_transfer_notebook.py`` itself, so the two notebooks and
the tutorial cannot drift apart on them.

The notebook runs in one of two stages (``REGISTRY_PT_STAGE``: ``prep`` or
``train``), or ``all`` for both in sequence (the workstation smoke,
``STEP_SMOKE=1``); the cells that only one stage needs are gated on it. The
result tables are assembled at the end of every train stage; the cluster
driver assembles them alone as well (``registry_pt_prongs.assemble``).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import nbformat as nbf

_HERE = Path(__file__).resolve().parent
CLUSTER_WORK_ROOT = "/gpfs/scratch/awills/xcquinox_runs/registry_pt_prongs"
SMOKE_WORK_ROOT = "~/.cache/xcquinox/registry_pt_prongs_smoke"


def _tutorial_builder():
    path = _HERE / "_build_registry_pt_transfer_notebook.py"
    spec = importlib.util.spec_from_file_location("_registry_pt_tutorial_builder",
                                                  str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def md(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


_PRONG_TEXT = {
    "A": ("per-group updates", "one optimizer step per training group, twelve "
          "per epoch: the DFS scheme (`_train_one_spec`, the package's "
          "`per_molecule` loop)"),
    "B": ("gradient accumulation", "the twelve group gradients averaged into "
          "one optimizer step per epoch (`registry_pt_prongs.py train-accum`, "
          "built from the package's own loop elements)"),
}


def cell_intro(prong):
    short, long = _PRONG_TEXT[prong]
    return md(f"""
# Registry proton-transfer case study, prong {prong}: {short}

Twenty-six architectures -- the MLPs three layers of sixteen nodes (the DFS
network), the three Kolmogorov-Arnold networks two layers of eight, each
KAN trained alone on its node (one KAN training reached 26 GB at three of sixteen
and 21.5 GB at three of eight at the workstation identity) -- are cloned from their parent functional
on the corpus atoms and fine-tuned on
the DFS-Letter miniature: six atoms, the Li and C cations, H2O with its W4-11
atomization energy, the DFS barrier trio's reactants and transition states,
relaxed CCSD(T) density references. Training runs the production Letter
pathway at the production identity: 6-311++G(3df,2pd), grid level 3, density
fitting, 25 differentiable SCF cycles per update, the Letter's channel weights,
the per-update seed mixture, `adam_plateau` at lr 1e-4, 100 epochs, SCAN seeds
on the meta-GGA rung.

This notebook is prong {prong}: {long}. Prong {'B' if prong == 'A' else 'A'} trains
the same architectures from the same clones and references with the other
update rule; the two share every prepared artifact.

As soon as a network finishes training it is validated self-consistently with
its own training solver on the 16 corpus species and the proton-transfer path
points (the water-dimer radical cation pair, a Zundel scan, a neutral dimer
scan): training-reaction errors, the cation barrier against the tracked
CCSD(T) value, density errors against relaxed CCSD(T) (the DFS Letter's
Eq. 20) and the combined energy-density score (Eq. 21), beside the pretrained
clone's and PBE's.

**Stages.** The notebook runs in one of two stages (`REGISTRY_PT_STAGE`):

* `prep` -- the campaign and run directory, the pretraining data, the clones
  and their certificates, the CCSD(T) references, the training specs (once,
  for both prongs);
* `train` -- workers claim architectures from this prong's work queue, train,
  validate, record, and claim the next; a failure is recorded and the worker
  moves on; the result tables are assembled at the end.

`all` runs both in order (the workstation smoke, `STEP_SMOKE=1`, at
def2-svp / grid level 1 / two epochs on four architectures).
""")


def cell_setup(prong):
    return code(f"""
import json
import os
import socket
import sys
import time
from pathlib import Path

PRONG = {prong!r}
STAGE = os.environ.get("REGISTRY_PT_STAGE", "all")
assert STAGE in ("prep", "train", "all"), STAGE
SMOKE = bool(os.environ.get("STEP_SMOKE"))


def stage_is(*stages):
    return STAGE == "all" or STAGE in stages


import xcquinox
_REPO = Path(xcquinox.__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO / "notebooks"))
import registry_pt_prongs as rpp

IDENT = dict(rpp.SMOKE_IDENTITY if SMOKE else rpp.PRODUCTION_IDENTITY)
WORK_ROOT_TOP = os.path.expanduser(os.environ.get(
    "REGISTRY_PT_WORK_ROOT", {SMOKE_WORK_ROOT!r} if SMOKE else {CLUSTER_WORK_ROOT!r}))
SHARED = os.path.join(WORK_ROOT_TOP, "shared")
PRONG_ROOT = os.path.join(WORK_ROOT_TOP, f"prong_{{PRONG}}")
# the shared cells (references, pretraining data, run directory) write under
# WORK_ROOT, which is the shared tree both prongs read
WORK_ROOT = SHARED
SEED_CACHE_DIR = os.path.join(SHARED, "seed_cache")
for _d in (SHARED, PRONG_ROOT, SEED_CACHE_DIR, os.path.join(SHARED, "external_refs")):
    os.makedirs(_d, exist_ok=True)
# SCAN seeds for the meta-GGA arms are generated on a cache miss (double gate)
os.environ.setdefault("XCQUINOX_SEED_ALLOW_GENERATE", "1")

BASIS = IDENT["basis"]
GRID_LEVEL = IDENT["grid_level"]
ORIENTATION_LOCK = IDENT["orientation_lock"]
DEMO_ARCHS = rpp.DEMO_ARCHS
GEA_ARCH = rpp.GEA_ARCH
H2O_COORDS = "O 0.0000 0.0000 0.1173; H 0.0000 0.7572 -0.4692; H 0.0000 -0.7572 -0.4692"
TIMINGS = {{}}
JOB_ID = rpp.current_job_id()
HOST = socket.gethostname()

# preparation failures persist across jobs: an architecture whose clone or
# certificate could not be made is recorded here and enters the work queue
# as failed (stage "prep") instead of stopping the run
PREP_STATUS_PATH = os.path.join(SHARED, "prep_status.json")
PREP_FAILED = (json.loads(Path(PREP_STATUS_PATH).read_text())
               if os.path.isfile(PREP_STATUS_PATH) else {{}})


def prep_fail(arch, stage, error):
    PREP_FAILED[arch] = {{"stage": stage, "error": str(error)[-2000:],
                         "job": JOB_ID, "t": time.time()}}
    rpp.write_json(PREP_STATUS_PATH, PREP_FAILED)
    print(f"PREP FAILED {{arch}} ({{stage}}): {{str(error)[-300:]}}")


def prep_ok(arch):
    if PREP_FAILED.pop(arch, None) is not None:
        rpp.write_json(PREP_STATUS_PATH, PREP_FAILED)


print(f"prong {{PRONG}}, stage {{STAGE}}, identity {{IDENT['name']}}: {{BASIS}} / grid "
      f"{{GRID_LEVEL}} / density fitting {{IDENT['density_fit']}} / {{IDENT['epochs']}} epochs")
print(f"work root: {{WORK_ROOT_TOP}} (job {{JOB_ID}} on {{HOST}})")
""")


def cell_architectures():
    return code("""
# The 26 architectures: 22 registry entries and four built in
# registry_pt_prongs.unlisted_configs (the 2x8 KAN family and the meta-GGA
# atom-context arm). register_unlisted makes the four resolvable by name in
# THIS kernel only (the fidelity certificate resolves architectures through
# the registry); the package source is unchanged.
rpp.register_unlisted()
ARCH_CFGS = rpp.arch_set(rpp.SMOKE_ARCHS if SMOKE else rpp.ALL_ARCHS)
ARCH_NAMES = list(ARCH_CFGS)
CENSUS = rpp.census(ARCH_CFGS)
if not SMOKE:
    assert CENSUS == {"total": 26, "gga_mlp": 17, "meta_gga": 6, "kan": 3}, CENSUS
REGISTERED_SWEEP = [a for a in ARCH_NAMES if a in rpp.REGISTERED_ARCHS]
assert set(DEMO_ARCHS) <= set(REGISTERED_SWEEP)

print(f"{'architecture':<26} {'family':<8} {'shape':>5}  descriptors")
for _name, _c in ARCH_CFGS.items():
    _fam = ("kan" if getattr(_c, "network", "mlp") == "kan"
            else "meta" if _c.meta_gga else "gga")
    _descs = ", ".join(d.name for d in _c.descriptors) or "-"
    print(f"{_name:<26} {_fam:<8} {_c.depth}x{_c.nodes:<3}  {_descs}")
print(f"census: {CENSUS}")
""")


def cell_campaign():
    return code("""
# The campaign configuration for the harness stages the preparation runs
# (dry-run submit, datagen, the demos' pretraining and certificate): registry
# names only, the GEA arm outside the sweep (its x2 gate cannot share the
# run-wide model block with the meta rungs; it trains in the library route
# under its own block). Written once by the preparation; every later job
# rebuilds the dictionary and requires the file on disk to equal it, so no
# job runs against a configuration it did not describe.
import yaml

campaign_path = os.path.join(SHARED, "campaign.yaml")
campaign = rpp.make_campaign_dict(SHARED, REGISTERED_SWEEP, IDENT,
                                  seed_cache_dir=SEED_CACHE_DIR)
if not os.path.isfile(campaign_path):
    if not stage_is("prep"):
        raise RuntimeError("no campaign.yaml: the prep stage has not run")
    # the one-point subset ledger the harness reads (H2O as the size-1 point)
    rpp.write_json(os.path.join(SHARED, "subset_index_log.json"),
                   {"jsd/1": {"chosen_indices": [0], "metric_value": 0.0,
                              "point_kinds": ["ae"], "point_names": ["h2o"],
                              "tag": "bin01"}})
    with open(campaign_path, "w") as _f:
        yaml.safe_dump(campaign, _f, sort_keys=False)
with open(campaign_path) as _f:
    _on_disk = yaml.safe_load(_f)
assert _on_disk == campaign, "campaign.yaml is not the configuration this notebook builds"
cfg = load_grid_config(campaign_path)
validate_grid_semantics(cfg, get_domain_profile(cfg.domain_profile))
require_explicit_bh76_mode(campaign_path)
print(f"campaign: {len(cfg.sweep.arch)} swept archs, {cfg.inputs.basis} / grid "
      f"{cfg.inputs.grid_level} / density fitting {cfg.inputs.density_fit} / "
      f"seed {cfg.inputs.seed_xc}")
""")


def cell_run_dir():
    return code("""
# The run directory: made once by the preparation's dry-run submit, reused by
# every later job (the harness layout <run>/pretrain/<arch> is what the
# training gate reads the certificate waiver from).
import subprocess

_runs = (sorted(Path(SHARED, "runs").glob("run_*"))
         if os.path.isdir(os.path.join(SHARED, "runs")) else [])
if _runs:
    RUN_DIR = str(_runs[-1])
elif stage_is("prep"):
    _proc = subprocess.run(
        [sys.executable, "-m", "xcquinox.pipeline.cluster", "submit",
         campaign_path, "--partition", "local", "--stages", "pretrain"],
        capture_output=True, text=True, check=True)
    RUN_DIR = [ln.split("run dir = ", 1)[1].strip()
               for ln in (_proc.stdout + _proc.stderr).splitlines()
               if "run dir = " in ln][-1]
else:
    raise RuntimeError("no run directory: the prep stage has not run")
assert os.path.isfile(os.path.join(RUN_DIR, "resolved_config.yaml"))
print(f"run dir: {RUN_DIR}")
""")


def cell_datagen():
    return code("""
# Pretraining data at the run identity (the harness's datagen stage: one file
# per parent density, PBE for the GGA arms and SCAN for the meta-GGA arms).
from xcquinox.pipeline.cluster import _datagen

if stage_is("prep"):
    _t0 = time.time()
    _rc = _datagen.main([RUN_DIR])
    TIMINGS["datagen"] = time.time() - _t0
    assert _rc in (0, None), _rc
    for _m in sorted(Path(cfg.pretrain.data_dir).glob("*.manifest.json")):
        _man = json.loads(_m.read_text())
        print(f"{_m.name}: basis {_man.get('basis')}, grid {_man.get('grid_level')}, "
              f"density fitting {_man.get('density_fit')}")
else:
    print("skipped: not the prep stage")
""")


def cell_harness_pretrain():
    return code("""
# The two demonstration architectures through the harness's own pretrain and
# certificate stages; a failure is recorded, not raised.
from xcquinox.pipeline.cluster import _pretrain

if stage_is("prep"):
    _stage_archs = pretrain_stage_archs(cfg)
    for _arch in DEMO_ARCHS:
        _ck = pretrain_checkpoint_dir(RUN_DIR, _arch)
        if os.path.isfile(os.path.join(_ck, "fidelity_certificate.json")):
            prep_ok(_arch)
            continue
        try:
            _idx = _stage_archs.index(_arch)
            _t0 = time.time()
            _rc = _pretrain.main([RUN_DIR, str(_idx)])
            assert _rc in (0, None), f"pretrain rc {_rc}"
            _rc = fidelity_lib.main([RUN_DIR, str(_idx)])
            assert _rc in (0, None), f"certificate rc {_rc}"
            TIMINGS[f"harness {_arch}"] = time.time() - _t0
            prep_ok(_arch)
            print(f"{_arch}: harness pretrain + certificate done")
        except Exception as _exc:  # noqa: BLE001 -- recorded, the run goes on
            prep_fail(_arch, "harness_pretrain", f"{type(_exc).__name__}: {_exc}")
else:
    print("skipped: not the prep stage")
""")


def cell_resolve():
    return code("""
# THE identity every model is built at: the architecture with the run's
# polarization override and model block applied (apply_run_config, the
# harness's own resolution). The GEA arm resolves under its own block (x2).
cfg_gea = replace(cfg, model=ModelConfig(parent_anchor=False,
                                         descriptor_coordinates="paper",
                                         ueg_gate="x2"))


def resolve_arch(name):
    return apply_run_config(pipeline.get_architecture(name),
                            cfg_gea if name == GEA_ARCH else cfg)


def make_pretrain_spec(arch_name):
    return pipeline.PretrainSpec(
        arch=resolve_arch(arch_name),
        data_dir=cfg.pretrain.data_dir,
        checkpoint_dir=pretrain_checkpoint_dir(RUN_DIR, arch_name),
        n_steps=IDENT["pretrain_steps_library"],
        lr_start=1.0e-3, lr_end=1.0e-5, lr_decay_start=0.5, lr_decay_end=0.9,
        grad_clip=0.0, seed=42, loss_weighting="rho_w_sampled",
        parent_density="auto", energy_term_weight=0.0,
        points_per_system=800, sampling_seed=42)


for _a in ARCH_NAMES:
    resolve_arch(_a)   # every architecture resolves before any work starts
print(f"{len(ARCH_NAMES)} architectures resolve under the run configuration")
""")


def cell_library_pretrain():
    return code("""
# The remaining architectures through the library route, one child process
# each (PREP_WORKERS at a time, default 2); a failed child is recorded and the
# next architecture starts.
from concurrent.futures import ThreadPoolExecutor
import pickle

if stage_is("prep"):
    _library = [a for a in ARCH_NAMES if a not in DEMO_ARCHS]
    _todo = [a for a in _library if not os.path.isfile(
        os.path.join(pretrain_checkpoint_dir(RUN_DIR, a), "xnet.eqx"))]
    _spec_dir = os.path.join(SHARED, "pretrain_specs")
    os.makedirs(_spec_dir, exist_ok=True)
    _child = (
        "import pickle, sys\\n"
        "import xcquinox.pipeline as pipeline\\n"
        "with open(sys.argv[1], 'rb') as f:\\n"
        "    spec = pickle.load(f)\\n"
        "pipeline.run_pretrain(spec)\\n")

    def _pretrain_one(arch):
        _p = os.path.join(_spec_dir, f"{arch}.pkl")
        with open(_p, "wb") as _f:
            pickle.dump(make_pretrain_spec(arch), _f)
        _rc = rpp.run_child([sys.executable, "-c", _child, _p],
                            log_path=os.path.join(SHARED, "logs", f"{arch}.pretrain.log"))
        if _rc == 0:
            prep_ok(arch)
        else:
            prep_fail(arch, "pretrain", f"child exit {_rc}; log "
                      f"{os.path.join(SHARED, 'logs', arch + '.pretrain.log')}")
        return arch, _rc

    _workers = int(os.environ.get("PREP_WORKERS", "2"))
    print(f"library pretraining: {len(_todo)} to run, {len(_library) - len(_todo)} "
          f"cached, {_workers} workers")
    _t0 = time.time()
    with ThreadPoolExecutor(max_workers=_workers) as _ex:
        for _arch, _rc in _ex.map(_pretrain_one, _todo):
            print(f"  {_arch}: {'done' if _rc == 0 else f'FAILED (rc {_rc})'}")
    TIMINGS["library pretrain"] = time.time() - _t0
else:
    print("skipped: not the prep stage")
""")


def cell_refs_gate(what):
    return code(f"""
# The {what} references are made by the preparation only; a later job
# requires them on disk (the driver below skips every reference it finds).
REFS_MAY_COMPUTE = stage_is("prep")
print(f"{what} references: {{'computed where missing' if REFS_MAY_COMPUTE else 'read from disk'}}")
""")


def cell_corpus_refs_guard():
    return code("""
if not REFS_MAY_COMPUTE:
    _missing = [s.name for s in corpus_specs
                if not os.path.isfile(os.path.join(ext_data_dir, f"{s.name}.npz"))]
    if _missing:
        raise RuntimeError(f"corpus references missing {_missing}: the prep stage has not finished")
""")


def cell_pt_refs_guard():
    return code("""
if not REFS_MAY_COMPUTE:
    _missing = [s.name for s in all_points
                if not os.path.isfile(os.path.join(SHARED, "pt_refs", f"{s.name}.npz"))]
    if _missing:
        raise RuntimeError(f"PT references missing {_missing}: the prep stage has not finished")
""")


def cell_pt_specs_save():
    return code("""
# The path points travel to the validation children as a pickle of their
# MoleculeSpecs (written by the preparation, read by every readout).
import pickle

PT_SPECS_PATH = os.path.join(SHARED, "pt_specs.pkl")
if stage_is("prep") and not os.path.isfile(PT_SPECS_PATH):
    _tmp = PT_SPECS_PATH + ".tmp"
    with open(_tmp, "wb") as _f:
        pickle.dump(list(all_points), _f)
    os.replace(_tmp, PT_SPECS_PATH)
print(f"path points: {len(all_points)} ({PT_SPECS_PATH})")
""")


def cell_certificates():
    return code("""
# Fidelity certificates, library route, on the corpus oracle; a failure is
# recorded. The gate is released (fidelity.enforce false), the certificate's
# presence is required by the training gate.
if stage_is("prep"):
    _cfg_resolved = load_resolved_run_config(RUN_DIR)
    _full_oracle = fidelity_lib.build_oracle_set(_cfg_resolved, ARCH_NAMES[0])
    _corpus_names = {s.name.lower() for s in corpus_specs}
    _corpus_oracle = tuple(ms for ms in _full_oracle
                           if ms.name.lower().removeprefix("atom_") in _corpus_names)
    assert len(_corpus_oracle) >= 3, [m.name for m in _full_oracle][:12]
    _t0 = time.time()
    for _arch in ARCH_NAMES:
        _ck = pretrain_checkpoint_dir(RUN_DIR, _arch)
        if os.path.isfile(os.path.join(_ck, "fidelity_certificate.json")):
            continue
        if not os.path.isfile(os.path.join(_ck, "xnet.eqx")):
            continue   # its pretraining failure is already recorded
        try:
            fidelity_lib.fidelity_certificate(
                _cfg_resolved if _arch != GEA_ARCH else cfg_gea,
                RUN_DIR, _arch, oracle_set=_corpus_oracle)
            prep_ok(_arch)
            print(f"{_arch}: certificate written")
        except Exception as _exc:  # noqa: BLE001 -- recorded, the run goes on
            prep_fail(_arch, "certificate", f"{type(_exc).__name__}: {_exc}")
    TIMINGS["certificates"] = time.time() - _t0
else:
    print("skipped: not the prep stage")
""")


def cell_train_specs():
    return code("""
# The training specs of both prongs (identical but for the checkpoint
# directory): the Letter pathway at the run identity, the per-rung seed,
# 100 epochs, a resume checkpoint after every epoch.
import pickle
from xcquinox.pipeline.train import _DEFAULT_CHANNEL_WEIGHTS, _training_groups

CHANNEL_WEIGHTS = {"loss_AE": 1.0, "loss_BH76": 1.0, "loss_IP13": 1.0,
                   "loss_rho": 20.0, "loss_vxc": 0.0, "loss_Eatom": 0.01}
assert set(CHANNEL_WEIGHTS) == set(_DEFAULT_CHANNEL_WEIGHTS)
TARGETS = {
    **{s: ATOM_ANCHORS[s] for s in ATOM_ANCHORS},
    "Li+": ATOM_ANCHORS["Li"], "C+": ATOM_ANCHORS["C"],
    "h2o": AE_HA,
    **{n: 0.0 for n in ("HO", "N2", "HF", "CH3", "n2ohts", "RKT11", "hf2ts")},
}


def make_train_spec(arch_name, prong):
    arch = resolve_arch(arch_name)
    solver = rpp.make_solver(arch, IDENT, seed_cache_dir=SEED_CACHE_DIR)
    return pipeline.TrainingSpec.from_dicts(
        molecules=tuple(corpus_specs),
        targets=TARGETS,
        atom_energies=dict(ATOM_ANCHORS),
        loss_name="L6_letter_parity",
        loss_kwargs={
            "bh76_reactions": [
                *[{"name": _r["name"], "reactants": tuple(_r["reactants"]),
                   "products": (_r["ts_species"],),
                   "coeffs": (-1.0,) * len(_r["reactants"]) + (1.0,),
                   "e_rxn_ref": _r["barrier_ref"] / KCAL_PER_HA}
                  for _r in DFS_BH76_REACTIONS],
                {"name": "h2o", "reactants": ("h2o",), "products": ("H", "O"),
                 "coeffs": (-1.0, 2.0, 1.0), "e_rxn_ref": AE_HA},
            ],
            "ip13_pairs": [
                {"name": _p["name"], "neutral": _p["neutral"],
                 "cation": _p["cation"], "ip_ref": _p["ip_ref"] / KCAL_PER_HA}
                for _p in DFS_IP13_PAIRS],
            "aux_only_names": ("CH3", "HF", "HO", "N2", "RKT11", "h2o",
                               "hf2ts", "n2ohts"),
            "regularize_atom_syms": ("C", "F", "H", "Li", "N", "O"),
            "h_species": ("H", "Li"),
            "radical_species": ("HO",),
            "solver_config": solver,
            "vxc_weight": 0.01,
            "density_weight": 0.1,
            "density_per_electron": True,
        },
        arch=arch,
        n_steps=IDENT["epochs"],
        lr_start=1.0e-4, lr_end=1.0e-7, lr_decay_start=0.5,
        grad_clip=1.0, weight_decay=1.0e-6, seed=42,
        solver_config=solver,
        update_scheme="per_molecule",   # prong B's child averages the same groups
        optimizer="adam_plateau", plateau_patience=10, plateau_factor=0.1,
        seed_mix_atomic=True, respect_sc_flag=False, nonsc_weight=0.01,
        pad_group_to_common_shape=True, superset_assembly=False,
        channel_weights=dict(CHANNEL_WEIGHTS),
        val_frac=0.2, validate_every=0, patience=8, early_stop_min_delta=0.1,
        validation_density_weight=20.0,
        checkpoint_every=1,
        pbe_anchor_weight=0.0, require_atom_anchors=False,
        balancing=GradNormConfig(alpha=1.5),
        pretrain_checkpoint=pretrain_checkpoint_dir(RUN_DIR, arch_name),
        checkpoint_dir=os.path.join(WORK_ROOT_TOP, f"prong_{prong}", "train", arch_name),
    )


def spec_path(arch_name, prong):
    return os.path.join(SHARED, "specs", prong, f"{arch_name}.pkl")


_written = 0
for _arch in ARCH_NAMES:
    _ck = pretrain_checkpoint_dir(RUN_DIR, _arch)
    if not os.path.isfile(os.path.join(_ck, "fidelity_certificate.json")):
        if _arch not in PREP_FAILED and stage_is("train"):
            prep_fail(_arch, "certificate", "no certificate on disk")
        continue
    for _prong in ("A", "B"):
        _p = spec_path(_arch, _prong)
        if os.path.isfile(_p):
            continue
        os.makedirs(os.path.dirname(_p), exist_ok=True)
        with open(_p + ".tmp", "wb") as _f:
            pickle.dump(make_train_spec(_arch, _prong), _f)
        os.replace(_p + ".tmp", _p)
        _written += 1
_spec0 = make_train_spec(ARCH_NAMES[0], PRONG)
N_GROUPS = len(_training_groups(_spec0))
assert N_GROUPS == 12, N_GROUPS
print(f"specs written: {_written}; {N_GROUPS} groups x {_spec0.n_steps} epochs; "
      f"preparation failures: {sorted(PREP_FAILED)}")
""")


def cell_train():
    return code("""
# The work queue. TRAIN_WORKERS workers (default 2) each claim the next
# pending architecture, train it in a child process, validate it in another
# (then the clone baseline, once per architecture across both prongs), record
# the result and claim the next. A failed child is recorded with its stage,
# exit code and log tail, and the worker moves on at once; a wall-clock
# interruption returns the architecture to pending for the next job. Each
# worker's children are pinned to its own share of the node's cores.
import threading

LEDGER = rpp.Ledger(os.path.join(PRONG_ROOT, "ledger.json"))
if stage_is("train"):
    # the job's SIGTERM (forwarded by the driver to this kernel's process
    # group, children included) stops further claims instead of ending the
    # kernel, so every interrupted architecture is recorded before the cell
    # returns
    rpp.install_term_handler()
    LEDGER.ensure(ARCH_NAMES, failed={a: v["stage"] for a, v in PREP_FAILED.items()})
    if os.environ.get("REGISTRY_PT_RETRY_FAILED") == "1":
        print(f"retrying failed: {LEDGER.reset_failed()}")
    _back = LEDGER.reclaim_stale(1800.0, live_jobs=rpp.live_slurm_jobs(), own_job=JOB_ID)
    if _back:
        print(f"reclaimed from jobs no longer running: {_back}")

    _pt_refs_dir = os.path.join(SHARED, "pt_refs")

    def stages_for(arch):
        _spec = spec_path(arch, PRONG)
        _train_dir = os.path.join(PRONG_ROOT, "train", arch)
        _readout = os.path.join(PRONG_ROOT, "readout", arch)
        _clone = os.path.join(SHARED, "clone_readout", arch)
        if PRONG == "A":
            _train = [sys.executable, "-m", "xcquinox.pipeline._train_one_spec",
                      _spec, "--device", "cpu", "--no-progress"]
        else:
            _train = [sys.executable, rpp.__file__, "train-accum", _spec]
        _ro = [sys.executable, rpp.__file__, "readout", "--spec", _spec,
               "--pt-specs", PT_SPECS_PATH, "--pt-refs", _pt_refs_dir]
        return [
            ("train", _train, None, os.path.join(_train_dir, "completion.json"), True),
            ("readout", _ro + ["--model", os.path.join(_train_dir, "model.eqx"),
                               "--out", _readout, "--label", f"prong_{PRONG}:{arch}"],
             None, os.path.join(_readout, "readout.json"), True),
            ("clone_readout", _ro + ["--model", "clone", "--out", _clone,
                                     "--label", f"clone:{arch}"],
             None, os.path.join(_clone, "readout.json"), False),
        ]

    # Two training classes: the standard architectures TRAIN_WORKERS at a
    # time, then the KANs ONE at a time with the whole node (each KAN trains
    # alone). REGISTRY_PT_ARCH_CLASS restricts a job to one class (a KAN-only
    # job on its own node, beside standard jobs).
    _classes = ([os.environ["REGISTRY_PT_ARCH_CLASS"]]
                if os.environ.get("REGISTRY_PT_ARCH_CLASS") else list(rpp.ARCH_CLASSES))
    assert set(_classes) <= set(rpp.ARCH_CLASSES), _classes
    _cores = sorted(os.sched_getaffinity(0))
    _t0 = time.time()
    for _cls in _classes:
        if rpp.STOP.is_set():
            print(f"the job is ending; class {_cls} not started")
            break
        _names = [a for a in ARCH_NAMES if rpp.arch_class(ARCH_CFGS[a]) == _cls]
        _n_workers = 1 if _cls == "kan" else int(os.environ.get("TRAIN_WORKERS", "2"))
        _share = max(1, len(_cores) // _n_workers)
        print(f"class {_cls}: {len(_names)} architectures, {_n_workers} worker(s), "
              f"{_share} cores each")
        _threads = []
        for _i in range(_n_workers):
            _cpus = _cores[_i * _share:(_i + 1) * _share] or _cores
            _th = threading.Thread(
                target=rpp.run_worker, args=(LEDGER, stages_for),
                kwargs=dict(job=JOB_ID, worker=f"{JOB_ID}:{_cls}{_i}", host=HOST,
                            log_dir=os.path.join(PRONG_ROOT, "logs"), cpus=_cpus,
                            accept=_names, stop=rpp.STOP),
                daemon=True)
            _th.start()
            _threads.append(_th)
        for _th in _threads:
            _th.join()
    TIMINGS["train stage"] = time.time() - _t0
    print(json.dumps(LEDGER.summary()["counts"]))
else:
    print("skipped: not the train stage")
""")


def cell_assemble():
    return code("""
# Result tables from every validation present (idempotent; every train job
# ends here, so the tables always describe the queue as it stands).
if stage_is("train"):
    _res = rpp.assemble(PRONG_ROOT, clone_root=os.path.join(SHARED, "clone_readout"),
                        ledger=LEDGER)
    print(f"readouts: {_res['n_readouts']}; clone readouts: {_res['n_clone_readouts']}")
    _cols = ("arch", "state", "mae_train_kcal", "pt_err_kcal", "eps_mean_corpus",
             "ed_published", "clone_mae_train_kcal", "pbe_mae_train_kcal")

    def _fmt(v):
        return f"{v:.4g}" if isinstance(v, float) else str(v)

    print(" | ".join(_cols))
    for _row in _res["summary"]:
        print(" | ".join(_fmt(_row.get(c)) for c in _cols))
    print(f"tables: {os.path.join(PRONG_ROOT, 'results')}")
    print(f"timings: {TIMINGS}")
else:
    print("skipped: not the train stage")
""")


def cell_close(prong):
    return md(f"""
## Outputs and remaining deviations

Prong {prong} writes under its own tree (`prong_{prong}/`): `train/<arch>/`
(the final and best-loss networks, losses, the per-update log, the resume set
while running), `readout/<arch>/` (`species/<name>.json` and `.npz` per
species as soon as each SCF finishes: energies, the SCF trace and convergence,
the density matrix and the density on the grid, the CCSD(T) density and the
density error; `readout.json` assembled from them), `ledger.json` (the queue
and every architecture's history), `logs/`, and `results/` (`summary.csv`,
`reactions.csv`, `density.csv`, `status.json`). The clone baselines are under
`shared/clone_readout/`, shared with the other prong.

Deviations from the DFS Letter that remain at this identity: the tail weights
run over all 25 cycles (the package's tail machinery cannot express the
paper's skipped early window); the seed mixture draws beta from U(0.5, 1), the
executed dpyscf script's form; no validation manifest exists for this corpus,
so no in-loop validation or validation-best checkpoint is taken; the corpus
is the DFS miniature, not the DFS training set; the readout's energy term for
the combined score is the paper-form WTMAD-2 over the six training reactions,
not GMTKN55.
""")


def build_cells(prong):
    if prong not in ("A", "B"):
        raise ValueError(f"prong must be 'A' or 'B', got {prong!r}")
    tb = _tutorial_builder()
    cells = [
        cell_intro(prong),
        cell_setup(prong),
        tb.build_cell_04_imports(),
        cell_architectures(),
        cell_campaign(),
        cell_run_dir(),
        cell_datagen(),
        cell_harness_pretrain(),
        cell_resolve(),
        cell_library_pretrain(),
        tb.build_cell_18b_staging(),
        cell_refs_gate("corpus"),
        cell_corpus_refs_guard(),
        tb.build_cell_19_ccsd_refs(),
        tb.build_cell_28_paths(5),
        cell_refs_gate("path-point"),
        cell_pt_refs_guard(),
        tb.build_cell_29_pt_ccsd(),
        cell_pt_specs_save(),
        cell_certificates(),
        cell_train_specs(),
        cell_train(),
        cell_assemble(),
        cell_close(prong),
    ]
    for idx, cell in enumerate(cells, start=1):
        cell["id"] = f"cell_{idx:02d}"
    return cells


def write_notebook(output_path, prong):
    nb = nbf.v4.new_notebook(
        cells=build_cells(prong),
        metadata={"kernelspec": {"display_name": "Python 3", "language": "python",
                                 "name": "python3"},
                  "language_info": {"name": "python"}})
    nbf.validate(nb)
    with open(output_path, "w") as f:
        nbf.write(nb, f)
    print(f"wrote {output_path} ({len(nb.cells)} cells)")
    return output_path
