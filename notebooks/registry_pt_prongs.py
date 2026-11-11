"""Runtime support for the registry proton-transfer prong notebooks.

Two notebooks (``registry_pt_prong_A.ipynb``, ``registry_pt_prong_B.ipynb``)
train the same 26 architectures (the MLPs 3x16, the three KANs 2x8, each KAN
alone on its node) at the production identity
(6-311++G(3df,2pd), grid level 3, density fitting, SCAN seeds on the meta-GGA
rung) and differ in one thing:

* prong A takes one optimizer step per training group, twelve per epoch (the
  DFS per-molecule scheme, ``xcquinox.pipeline._train_one_spec``);
* prong B averages the twelve group gradients and takes one step per epoch
  (gradient accumulation, the ``train-accum`` entry point below).

This module holds what the notebooks and their child processes share, so no
code lives twice:

* the architecture set: the 22 registry entries and the four built here
  (``unlisted_configs``), with ``register_unlisted`` making the four
  resolvable by name inside one process (the fidelity certificate resolves
  architectures through the registry);
* the identities (production, smoke), the campaign configuration and the
  per-architecture solver;
* the work queue (``Ledger``) and the worker loop (``run_worker``): each
  worker claims the next architecture, trains it in a child process,
  validates it in another, records the result and claims the next; a failed
  child is recorded and the slot moves on;
* the two child entry points (``train-accum``, ``readout``) and the result
  assembly (``assemble``).

Run as a script it dispatches to the child entry points:
``python registry_pt_prongs.py train-accum <spec.pkl>`` and
``python registry_pt_prongs.py readout --spec ... --model ... --out ...``.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import fcntl
import json
import math
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

NOTEBOOK_DIR = Path(__file__).resolve().parent
REPO = NOTEBOOK_DIR.parent
TRACKED_PT_TRAJ = REPO / "data" / "training_subsets" / "01wf" / "subat_ref.traj"

KCAL_PER_HA = 627.5094740631
GAMMA_PUBLISHED = 1084.87   # DFS Letter, Eq. 21 (eval_holdout.DFS_LETTER_GAMMA)
GAMMA_OWN_AXES = 1158.34    # this project's own-axes fit (nonempirical_gamma)

# ---------------------------------------------------------------------------
# Architectures
# ---------------------------------------------------------------------------

#: the registry entries of the set, registry order (GGA, then meta-GGA)
REGISTERED_ARCHS = (
    "deep_3x16", "deep_attn_3x16", "deep_dm_3x16", "deep_combined_3x16",
    "deep_combined_attn_3x16", "deep_geom_3x16", "deep_geom_attn_3x16",
    "deep_atomctx_3x16", "deep_gea_3x16", "deep_ff_3x16", "deep_sine_3x16",
    "deep_lap_3x16", "deep_lap_geom_3x16", "deep_rung35_3x16",
    "deep_rung35_attn_3x16", "deep_rung35ms_3x16", "deep_rung35only_3x16",
    "deep_mgga_3x16", "deep_mgga_attn_3x16", "deep_rung35_mgga_3x16",
    "deep_cusp_mgga_3x16", "deep_rung35ms_mgga_3x16",
)
#: the four built here: the meta-GGA atom-context arm and the KAN family. The
#: KAN family is two layers of EIGHT: one KAN training reached 26 GB at 3x16 and
#: 21.5 GB at 3x8 before its first epoch, at def2-svp / grid level 1 (workstation
#: smoke, 2026-11-06), and every KAN trains alone on its node (class "kan" below)
UNLISTED_ARCHS = ("deep_atomctx_mgga_3x16", "deep_kan_2x8",
                  "deep_kan_geom_2x8", "deep_kan_atomctx_2x8")
ALL_ARCHS = REGISTERED_ARCHS + UNLISTED_ARCHS
#: the harness-route demonstration pair (one GGA, one meta-GGA)
DEMO_ARCHS = ("deep_3x16", "deep_mgga_3x16")
#: the arm whose x2 gate keeps it out of the run-wide campaign model block
GEA_ARCH = "deep_gea_3x16"
#: the smoke slice: both demos and two of the unregistered entries
SMOKE_ARCHS = ("deep_3x16", "deep_mgga_3x16", "deep_kan_atomctx_2x8",
               "deep_atomctx_mgga_3x16")
#: the shape of each family: the MLPs at the DFS network's 3x16, the KANs 2x8
SHAPE_MLP, SHAPE_KAN = (3, 16), (2, 8)
#: training classes: "standard" trains TRAIN_WORKERS at a time on a node,
#: "kan" trains one at a time, alone on its node
ARCH_CLASSES = ("standard", "kan")


def arch_class(cfg) -> str:
    """The training class of an architecture: "kan" for a KAN, else "standard"."""
    return "kan" if getattr(cfg, "network", "mlp") == "kan" else "standard"


def unlisted_configs() -> dict:
    """The four architectures that are not registry entries, built field for
    field as the registry builds its neighbours: the KAN fields of the
    registry's KAN entries (cubic B-splines on five intervals), the meta-GGA
    fields of ``deep_mgga_3x16``, and the log transform and intensive DM
    entropy every deep entry carries."""
    from xcquinox.pipeline.config import ArchitectureConfig

    common = dict(dm_entropy_intensive=True, descriptor_log_transform=True)
    kan = dict(network="kan", kan_grid=5, kan_order=3)
    return {
        "deep_atomctx_mgga_3x16": ArchitectureConfig.from_spec(
            "deep_atomctx_mgga_3x16", 3, 16,
            descriptors=["atom_context", "metagga"], meta_gga=True, **common),
        "deep_kan_2x8": ArchitectureConfig.from_spec(
            "deep_kan_2x8", *SHAPE_KAN, **common, **kan),
        "deep_kan_geom_2x8": ArchitectureConfig.from_spec(
            "deep_kan_geom_2x8", *SHAPE_KAN, descriptors=["cusp"], **common, **kan),
        "deep_kan_atomctx_2x8": ArchitectureConfig.from_spec(
            "deep_kan_atomctx_2x8", *SHAPE_KAN, descriptors=["atom_context"],
            **common, **kan),
    }


def register_unlisted() -> list:
    """Add the four unlisted architectures to the registry of THIS process
    only (``pipeline.config.ARCHITECTURES``), so name-based resolution -- the
    fidelity certificate's parent and model build -- finds them. Idempotent;
    returns the names added. The package source is not touched."""
    from xcquinox.pipeline import config as _config

    added = []
    for name, cfg in unlisted_configs().items():
        if name not in _config.ARCHITECTURES:
            _config.ARCHITECTURES[name] = cfg
            added.append(name)
    return added


def arch_set(names=ALL_ARCHS) -> dict:
    """name -> raw ArchitectureConfig for ``names`` (registry entries from the
    registry, the unlisted four from ``unlisted_configs``)."""
    from xcquinox.pipeline.config import get_architecture

    extra = unlisted_configs()
    return {n: (extra[n] if n in extra else get_architecture(n)) for n in names}


def census(cfgs: dict) -> dict:
    """Family counts of an architecture set, and its shape and transform
    invariants: every MLP 3x16, every KAN 2x8, every entry log-transformed."""
    def _shape_ok(c):
        want = SHAPE_KAN if arch_class(c) == "kan" else SHAPE_MLP
        return (c.depth, c.nodes) == want
    bad = {n: (c.depth, c.nodes, c.descriptor_log_transform)
           for n, c in cfgs.items()
           if not _shape_ok(c) or not c.descriptor_log_transform}
    if bad:
        raise ValueError(f"entries off the shape (MLP 3x16, KAN 2x8) or "
                         f"log-transform rule: {bad}")
    kan = [n for n, c in cfgs.items() if getattr(c, "network", "mlp") == "kan"]
    meta = [n for n, c in cfgs.items() if c.meta_gga]
    gga = [n for n in cfgs if n not in set(kan) | set(meta)]
    return {"total": len(cfgs), "gga_mlp": len(gga), "meta_gga": len(meta),
            "kan": len(kan)}


# ---------------------------------------------------------------------------
# Identities, campaign, solver
# ---------------------------------------------------------------------------

PRODUCTION_IDENTITY = {
    "name": "production",
    "basis": "6-311++G(3df,2pd)",
    "grid_level": 3,
    "density_fit": True,
    "auxbasis": None,          # pyscf auto-selects for a Pople basis
    "orientation_lock": 3.0e-5,
    "epochs": 100,
    "pretrain_steps_harness": 4000,
    "pretrain_steps_library": 1500,
    "pt_scan_points": 5,
}
#: the workstation smoke: the same code paths at a minutes-scale identity
#: (grid level 1 requires the degeneracy waiver; density fitting stays on so
#: the fitted path is exercised)
SMOKE_IDENTITY = {
    "name": "smoke",
    "basis": "def2-svp",
    "grid_level": 1,
    "density_fit": True,
    "auxbasis": None,
    "orientation_lock": 3.0e-5,
    "epochs": 2,
    "pretrain_steps_harness": 25,
    "pretrain_steps_library": 5,
    "pt_scan_points": 5,
}
#: grid level from which the validator refuses the degeneracy waiver
#: (grid_config._MIN_REPRODUCIBLE_GRID_LEVEL with the lock on)
_WAIVER_FREE_GRID_LEVEL = 3


def make_campaign_dict(work_root, archs, ident, *, seed_cache_dir):
    """The campaign configuration as a plain dict, for the harness stages the
    notebook runs (dry-run submit, datagen, the demos' pretrain and
    certificate). ``archs`` are REGISTRY names only (every harness stage
    resolves names); the GEA arm is left out of the sweep because its x2 gate
    cannot share the run-wide model block with the meta rungs (tanh2) and
    trains in the library route under its own block. Every value is stated.
    The degeneracy waiver is stated only below grid level 3, where the
    datagen guard needs it; at grid level 3 with the lock on the validator
    refuses it (grid_config.validate_grid_semantics)."""
    sweep = [a for a in archs if a != GEA_ARCH]
    inputs = {
        "external_refs_dir": f"{work_root}/external_refs",
        "subset_ledger_path": f"{work_root}/subset_index_log.json",
        "basis": ident["basis"],
        "grid_level": int(ident["grid_level"]),
        "output_root": f"{work_root}",
        "density_fit": bool(ident["density_fit"]),
        "orientation_lock_strength": float(ident["orientation_lock"]),
        # per-rung seeds (SCAN for the meta-GGA arms), as the v9 letter arms
        "seed_xc": "auto",
        "seed_cache_dir": seed_cache_dir,
    }
    if ident["auxbasis"]:
        inputs["auxbasis"] = ident["auxbasis"]
    if int(ident["grid_level"]) < _WAIVER_FREE_GRID_LEVEL:
        inputs["allow_irreproducible_degenerate"] = True
        inputs["irreproducible_degenerate_reason"] = (
            "registry-pt prong smoke at grid level 1: the 3P terms of the O "
            "and C atoms are unresolved by this quadrature; the waiver is "
            "recorded in the datagen sidecar manifests")
    return {
        "sweep": {"arch": sweep, "loss": ["L6_letter_parity"],
                  "metric": ["jsd"], "subset_size": [1],
                  "solver": ["full_25"]},
        "solvers": {"full_25": {
            "mode": "FULL", "max_cycles": 25, "feature_policy": "REASSEMBLE",
            "scf_grad_checkpoint": True, "mixer_name": "decaying_linear",
            "mixer_kwargs": {"base": 0.3, "floor": 0.3, "step_offset": 1},
            "scf_loss_use_tail": True, "scf_loss_tail": 25,
            "scf_loss_weight_power": 2.0, "freeze_on_convergence": False,
            "fock_noise_std": 1.0e-5, "fock_noise_seed": 0}},
        "hyperparams": {
            "n_steps": int(ident["epochs"]), "lr_start": 1.0e-4,
            "lr_end": 1.0e-7, "lr_decay_start": 0.5, "grad_clip": 1.0,
            "gradnorm_alpha": 1.5, "vxc_weight": 0.01, "density_weight": 0.1,
            "density_per_electron": True, "weight_decay": 1.0e-6,
            "val_frac": 0.2, "validate_every": 0, "patience": 8,
            "early_stop_min_delta": 0.1, "checkpoint_every": 1,
            "pbe_anchor_weight": 0.0, "require_atom_anchors": False,
            "seed": 42, "update_scheme": "per_molecule",
            "pad_group_to_common_shape": True, "optimizer": "adam_plateau",
            "plateau_patience": 10, "plateau_factor": 0.1,
            "seed_mix_atomic": True, "respect_sc_flag": False,
            "nonsc_weight": 0.01, "validation_density_weight": 20.0,
            "channel_weights": {"loss_AE": 1.0, "loss_BH76": 1.0,
                                "loss_IP13": 1.0, "loss_rho": 20.0,
                                "loss_vxc": 0.0, "loss_Eatom": 0.01}},
        "inputs": inputs,
        "pretrain": {
            "data_dir": f"{work_root}/pretrain_data",
            "n_steps": int(ident["pretrain_steps_harness"]),
            "lr_start": 1.0e-3, "lr_end": 1.0e-5, "lr_decay_start": 0.5,
            "lr_decay_end": 0.9, "grad_clip": 0.0, "seed": 42,
            "loss_weighting": "rho_w_sampled",
            "atoms": {"H": 1, "Li": 1, "C": 2, "N": 3, "O": 2, "F": 1},
            "dfs_set": False, "pool_atoms": False, "parent_density": "auto",
            "exchange_footing": "paper", "mesh_fraction": 0.3,
            "energy_term_weight": 0.0, "points_per_system": 800,
            "sampling_seed": 42, "validation_fraction": 0.0,
            "validation_seed": 0, "validate_every": 4000, "patience": 0},
        "cluster": {
            "partition": "", "time": "00:30:00", "mem": "4G",
            "cpus_per_task": 4, "array_throttle": 1,
            "eval_array_throttle": 1, "max_concurrent_tasks": 2,
            "max_array_size": 1000, "device": "cpu", "conda_env": "",
            "conda_profile": "", "mail_user": "", "mail_type": "NONE",
            "account": "", "eval_workers": 1, "preflight_time": "00:30:00",
            "eval_time": "00:30:00", "pretrain_time": "00:30:00"},
        "domain_profile": "dfs_step7",
        "on_precompute_failure": "abort",
        "bh76_mode": "barrier_height",
        "ae_as_reactions": True,
        "use_polarized_correlation": True,
        "inline_eval": False,
        "defer_eval": False,
        "model": {"parent_anchor": False, "descriptor_coordinates": "paper"},
        "fidelity": {
            "enforce": False,
            "override_reason": (
                "registry-pt prongs: the miniature corpus cannot meet the "
                "full-oracle tolerances; the certificate is computed on the "
                "corpus oracle and recorded, the gate is released")},
    }


def make_solver(arch, ident, *, seed_cache_dir):
    """The v9 letter FULL-25 solver at ``ident``, seeded per rung: SCAN for a
    meta-GGA architecture, PBE otherwise (the harness's ``seed_xc: auto``)."""
    from xcquinox.pipeline.solver import FeaturePolicy, SolverConfig, SolverMode

    return SolverConfig(
        mode=SolverMode.FULL,
        max_cycles=25,
        feature_policy=FeaturePolicy.REASSEMBLE,
        mixer_name="decaying_linear",
        mixer_kwargs=(("base", 0.3), ("floor", 0.3), ("step_offset", 1.0)),
        density_fit=bool(ident["density_fit"]),
        auxbasis=ident["auxbasis"],
        scf_grad_checkpoint=True,
        scf_loss_use_tail=True,
        scf_loss_tail=25,
        scf_loss_weight_power=2.0,
        freeze_on_convergence=False,
        orientation_lock_strength=float(ident["orientation_lock"]),
        seed_source="scan" if arch.meta_gga else "pbe",
        seed_cache_dir=seed_cache_dir,
        fock_noise_std=1.0e-5,
        fock_noise_seed=0,
    )


# ---------------------------------------------------------------------------
# Atomic writes
# ---------------------------------------------------------------------------

def write_json(path, payload) -> None:
    """Write JSON through a sibling temporary file and an atomic rename."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=1, default=_json_default)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _json_default(obj):
    try:
        import numpy as np
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except ImportError:  # pragma: no cover
        pass
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


# ---------------------------------------------------------------------------
# Work queue
# ---------------------------------------------------------------------------

#: ledger states: pending -> running -> trained -> validated | failed
PENDING, RUNNING, TRAINED, VALIDATED, FAILED = (
    "pending", "running", "trained", "validated", "failed")
#: child exit codes of an interruption (SIGTERM through the job's wall-clock
#: pre-kill): the architecture goes back to pending, not to failed
INTERRUPTED_CODES = (143, -15)


class Ledger:
    """The prong's work queue: one JSON file under an fcntl lock, shared by
    every job of the prong. Each architecture carries its state, the stage it
    is in, the claiming job and worker, a heartbeat, and its history."""

    def __init__(self, path):
        self.path = Path(path)
        self.lock_path = self.path.with_name(self.path.name + ".lock")
        self.path.parent.mkdir(parents=True, exist_ok=True)

    @contextlib.contextmanager
    def _locked(self):
        with open(self.lock_path, "a+") as lockf:
            fcntl.flock(lockf, fcntl.LOCK_EX)
            try:
                data = (json.loads(self.path.read_text())
                        if self.path.is_file() else {"order": [], "archs": {}})
                yield data
                write_json(self.path, data)
            finally:
                fcntl.flock(lockf, fcntl.LOCK_UN)

    def read(self) -> dict:
        with self._locked() as data:
            return json.loads(json.dumps(data))

    @staticmethod
    def _event(entry, state, **info):
        entry["state"] = state
        entry.update(info)
        entry.setdefault("history", []).append(
            {"t": time.time(), "state": state,
             **{k: v for k, v in info.items() if k != "stderr_tail"}})

    def ensure(self, names, failed=None) -> None:
        """Add every name not yet present as pending (idempotent: a second job
        of the prong changes nothing it finds). ``failed`` is the preparation's
        current record (name -> reason): an entry still pending whose
        preparation failed is recorded failed at stage "prep", and an entry
        failed at stage "prep" that the record no longer names (a later
        preparation repaired it) returns to pending. ``None`` means no record
        is known: prep failures already on the ledger are kept."""
        with self._locked() as data:
            for n in names:
                if n not in data["archs"]:
                    data["order"].append(n)
                    data["archs"][n] = {"state": PENDING, "history": []}
            if failed is None:
                return
            for n, reason in failed.items():
                e = data["archs"].get(n)
                if e is not None and e["state"] == PENDING:
                    self._event(e, FAILED, stage="prep", reason=str(reason))
            for n, e in data["archs"].items():
                if (e["state"] == FAILED and e.get("stage") == "prep"
                        and n not in failed):
                    self._event(e, PENDING, stage="prep",
                                reason="the preparation now succeeds")

    def claim_next(self, job, worker, host, accept=None):
        """Claim the first pending architecture in order (among ``accept``,
        when given: a training class's names); None when none is."""
        accept = None if accept is None else set(accept)
        with self._locked() as data:
            for n in data["order"]:
                e = data["archs"][n]
                if e["state"] == PENDING and (accept is None or n in accept):
                    self._event(e, RUNNING, stage="claimed", job=str(job),
                                worker=str(worker), host=str(host),
                                heartbeat=time.time())
                    return n
        return None

    def heartbeat(self, name, worker) -> None:
        with self._locked() as data:
            e = data["archs"].get(name)
            if e is not None and e.get("worker") == str(worker) \
                    and e["state"] in (RUNNING, TRAINED):
                e["heartbeat"] = time.time()

    def mark(self, name, state, *, owner=None, **info) -> bool:
        """Record ``state`` on ``name``. With ``owner`` (a worker id) the
        record is made only while that worker still holds the claim: a worker
        whose claim was reclaimed and taken by another job writes nothing.
        Returns whether the record was made."""
        with self._locked() as data:
            e = data["archs"][name]
            if owner is not None and e.get("worker") != str(owner):
                return False
            info.setdefault("heartbeat", time.time())
            self._event(e, state, **info)
            return True

    def reclaim_stale(self, max_age_s, live_jobs=None, own_job=None) -> list:
        """Return to pending every running or trained-but-unvalidated claim
        whose job is no longer alive or whose heartbeat is older than
        ``max_age_s``. A job counts as gone only when the scheduler's list is
        known (``live_jobs``) and trustworthy: when ``own_job`` is given it
        must appear in the list itself (an empty or partial reply judges no
        job dead), and the calling job's own claims are never judged by it."""
        now = time.time()
        back = []
        live = None if live_jobs is None else set(map(str, live_jobs))
        if live is not None and own_job is not None and str(own_job) not in live:
            live = None
        with self._locked() as data:
            for n in data["order"]:
                e = data["archs"][n]
                if e["state"] not in (RUNNING, TRAINED):
                    continue
                job = e.get("job")
                dead = (live is not None and job != str(own_job)
                        and job not in live)
                stale = now - float(e.get("heartbeat", 0.0)) > max_age_s
                if dead or stale:
                    self._event(e, PENDING, stage="reclaimed",
                                reason="job gone" if dead else "stale heartbeat")
                    back.append(n)
        return back

    def reset_failed(self) -> list:
        """Failed entries back to pending (an explicit retry of a prong)."""
        back = []
        with self._locked() as data:
            for n in data["order"]:
                e = data["archs"][n]
                if e["state"] == FAILED:
                    self._event(e, PENDING, stage="retry")
                    back.append(n)
        return back

    def summary(self) -> dict:
        data = self.read()
        counts = {}
        for e in data["archs"].values():
            counts[e["state"]] = counts.get(e["state"], 0) + 1
        return {"counts": counts,
                "archs": {n: {k: data["archs"][n].get(k) for k in
                              ("state", "stage", "job", "host", "returncode",
                               "reason")}
                          for n in data["order"]}}


def live_slurm_jobs():
    """The user's job ids SLURM reports alive, or None where squeue is absent
    (a workstation): liveness is then judged by heartbeat alone."""
    try:
        out = subprocess.run(
            ["squeue", "-h", "-u", os.environ.get("USER", ""), "-o", "%i"],
            capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    jobs = set()
    for line in out.stdout.split():
        jobs.add(line.strip())
        jobs.add(line.strip().split("_")[0])
    return jobs


def current_job_id() -> str:
    return os.environ.get("SLURM_JOB_ID") or f"local-{os.getpid()}"


# ---------------------------------------------------------------------------
# Worker loop
# ---------------------------------------------------------------------------

def _tail(path, n_lines=40) -> str:
    try:
        lines = Path(path).read_text(errors="replace").splitlines()
    except OSError:
        return ""
    return "\n".join(lines[-n_lines:])


#: set by the train stage's SIGTERM handler (:func:`install_term_handler`):
#: the job is ending, so the workers claim nothing more and start no stage
STOP = threading.Event()

_PR_SET_PDEATHSIG = 1


def install_term_handler() -> None:
    """Make SIGTERM to this process set :data:`STOP` instead of ending it.
    The job's TERM reaches the process group: the children (in the same
    group) flush and exit 143, and the workers, still alive, record each
    interrupted architecture as pending before the stage returns. A Python
    handler is reset to the default by exec, so the children keep their own
    handling. Called from the main thread."""
    signal.signal(signal.SIGTERM, lambda signum, frame: STOP.set())


def _libc_prctl():
    """libc's prctl, or None where there is none (non-Linux). Resolved in the
    parent: the child between fork and exec of a threaded process must not
    import or load anything (another thread may have held the import lock at
    the fork)."""
    try:
        import ctypes
        return ctypes.CDLL(None, use_errno=True).prctl
    except (OSError, AttributeError):
        return None


#: libc's prctl, resolved once at import (None off Linux)
_PRCTL = _libc_prctl() if sys.platform.startswith("linux") else None


def _child_preexec(cpu_list, parent_pid):
    """Runs in the child between fork and exec: pin the cores, and ask the
    kernel for SIGTERM when the parent thread dies, so a child is never
    orphaned by a dead kernel (it flushes its resume set and exits). Only
    system calls run here; the prctl handle is resolved at import."""
    prctl = _PRCTL
    sigterm = int(signal.SIGTERM)

    def preexec():
        if cpu_list:
            os.sched_setaffinity(0, cpu_list)
        if prctl is None:
            return
        prctl(_PR_SET_PDEATHSIG, sigterm, 0, 0, 0)
        if os.getppid() != parent_pid:   # the parent died before prctl
            os.kill(os.getpid(), sigterm)
    return preexec


def run_child(argv, *, env=None, log_path, heartbeat=None, heartbeat_s=60.0,
              cpus=None) -> int:
    """Run one child to completion with its output in ``log_path``; call
    ``heartbeat`` every ``heartbeat_s`` while it runs. ``cpus`` pins the child
    to a core set (the node is shared by the job's workers). The child
    receives SIGTERM if this process dies. A child that cannot even start
    returns -1000; the return code is never raised."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    preexec = _child_preexec(sorted(cpus) if cpus else None, os.getpid())
    with open(log_path, "a") as logf:
        logf.write(f"\n# {time.strftime('%Y-%m-%dT%H:%M:%S')} {' '.join(map(str, argv))}\n")
        logf.flush()
        try:
            proc = subprocess.Popen([str(a) for a in argv], env=env, stdout=logf,
                                    stderr=subprocess.STDOUT,
                                    preexec_fn=preexec)
        except OSError as exc:
            logf.write(f"# could not start: {exc}\n")
            return -1000
        while True:
            try:
                return proc.wait(timeout=heartbeat_s)
            except subprocess.TimeoutExpired:
                if heartbeat is not None:
                    try:
                        heartbeat()
                    except Exception as exc:  # noqa: BLE001 -- never kill the child for it
                        logf.write(f"# heartbeat failed: {exc}\n")
                        logf.flush()


def run_worker(ledger, stages_for, *, job, worker, host, log_dir,
               heartbeat_s=60.0, cpus=None, accept=None, say=print,
               stop=None) -> list:
    """Claim, run, record, repeat until nothing is pending. ``stages_for(name)``
    returns ``[(stage, argv, env, done_path, required), ...]``; a stage whose
    ``done_path`` exists is skipped (resume). The first stage is training:
    after it the entry is ``trained``. A non-zero exit of a required stage
    records ``failed`` with the stage, the code, the signal and the log tail,
    and the worker moves on at once; an optional stage's failure (the clone
    baseline) is recorded on the entry without failing it; an interruption
    code returns the entry to pending. Once ``stop`` (an Event) is set the
    worker claims nothing more. Returns the names this worker handled."""
    handled = []
    while True:
        if stop is not None and stop.is_set():
            say(f"[{worker}] the job is ending; no further claims")
            return handled
        name = ledger.claim_next(job, worker, host, accept=accept)
        if name is None:
            return handled
        handled.append(name)
        try:
            ending = _run_stages(ledger, stages_for, name, job=job, worker=worker,
                                 host=host, log_dir=log_dir, heartbeat_s=heartbeat_s,
                                 cpus=cpus, say=say, stop=stop)
        except _ClaimLost:
            say(f"[{worker}] {name}: the claim passed to another job; abandoned")
            continue
        except Exception as exc:  # noqa: BLE001 -- recorded; the worker goes on
            ledger.mark(name, FAILED, owner=worker, stage="worker",
                        reason=f"{type(exc).__name__}: {exc}",
                        stderr_tail=traceback.format_exc()[-4000:])
            say(f"[{worker}] {name}: worker error recorded ({type(exc).__name__}: {exc})")
            continue
        if ending:
            return handled


class _ClaimLost(Exception):
    """The worker's claim was reclaimed and taken by another job."""


def _run_stages(ledger, stages_for, name, *, job, worker, host, log_dir,
                heartbeat_s, cpus, say, stop=None) -> bool:
    """Run one architecture's stages; record the outcome. Returns True when
    the job is being interrupted (the worker stops claiming). Once ``stop``
    is set no further stage starts: an architecture with a required stage
    still to run returns to pending, one with only the optional baseline
    left is validated with the baseline noted as not run. Every record is
    made only while this worker holds the claim; a lost claim raises
    :class:`_ClaimLost` and no further stage of it starts here."""
    def _rec(state, **info):
        if not ledger.mark(name, state, owner=worker, **info):
            raise _ClaimLost(name)

    outcome = VALIDATED
    optional_failures = []
    for i, (stage, argv, env, done_path, required) in enumerate(stages_for(name)):
        if done_path is not None and Path(done_path).exists():
            say(f"[{worker}] {name}: {stage} already done")
            if i == 0:
                _rec(TRAINED, stage=stage, job=str(job),
                     worker=str(worker), host=str(host))
            continue
        if stop is not None and stop.is_set():
            if required:
                _rec(PENDING, stage=stage,
                     reason="interrupted before the stage started")
                say(f"[{worker}] {name}: job ending before {stage}; back to pending")
            else:
                optional_failures.append({"stage": stage, "returncode": None,
                                          "reason": "not run: the job was ending"})
                _rec(VALIDATED, stage="done",
                     optional_failures=optional_failures)
                say(f"[{worker}] {name}: job ending before optional {stage}; validated")
            return True
        _rec(RUNNING if i == 0 else TRAINED, stage=stage,
             job=str(job), worker=str(worker), host=str(host))
        log_path = Path(log_dir) / f"{name}.{stage}.log"
        t0 = time.time()
        say(f"[{worker}] {name}: {stage} started")
        rc = run_child(argv, env=env, log_path=log_path, cpus=cpus,
                       heartbeat=lambda n=name: ledger.heartbeat(n, worker),
                       heartbeat_s=heartbeat_s)
        dt = time.time() - t0
        if rc in INTERRUPTED_CODES and not required:
            # the required stages are done: an interrupted baseline does not
            # take a trained and validated architecture back to pending
            optional_failures.append({"stage": stage, "returncode": rc,
                                      "reason": "interrupted", "log": str(log_path)})
            _rec(VALIDATED, stage="done",
                 optional_failures=optional_failures)
            say(f"[{worker}] {name}: optional {stage} interrupted (rc {rc}); validated")
            return True
        if rc in INTERRUPTED_CODES:
            _rec(PENDING, stage=stage, reason="interrupted",
                 returncode=rc)
            say(f"[{worker}] {name}: {stage} interrupted (rc {rc}); back to pending")
            return True
        if rc != 0 and not required:
            optional_failures.append({"stage": stage, "returncode": rc,
                                      "log": str(log_path)})
            say(f"[{worker}] {name}: optional {stage} failed (rc {rc}); log {log_path}")
            continue
        if rc != 0:
            _rec(FAILED, stage=stage, returncode=rc,
                 signal=(-rc if rc < 0 else None), seconds=dt,
                 log=str(log_path), stderr_tail=_tail(log_path))
            say(f"[{worker}] {name}: {stage} FAILED (rc {rc}) after {dt:.0f} s; "
                f"log {log_path}")
            outcome = FAILED
            break
        say(f"[{worker}] {name}: {stage} done in {dt:.0f} s")
        if i == 0:
            _rec(TRAINED, stage=stage, seconds=dt)
    if outcome == VALIDATED:
        _rec(VALIDATED, stage="done",
             optional_failures=optional_failures)
    return False


# ---------------------------------------------------------------------------
# Child: gradient-accumulation training (prong B)
# ---------------------------------------------------------------------------

def train_accum_main(spec_path) -> int:
    """Train one spec with the twelve group gradients averaged into one
    optimizer step per epoch. Every other element is the per-molecule loop's
    own: the model build and its gates, the loss and batch, the groups, the
    seed mixture and its stream, the optimizer and the plateau controller on
    the same epoch metric, the best-loss snapshot, and the artifacts and the
    resume set in the loop's formats. The resume set is written after every
    epoch and flushed on SIGTERM."""
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    import jax
    jax.config.update("jax_enable_x64", True)
    import equinox as eqx
    import jax.numpy as jnp
    import numpy as np

    from xcquinox.pipeline import train as T
    from xcquinox.pipeline.checkpoint_class import load_pickle
    from xcquinox.pipeline.defused_grad import defused_value_and_grad
    from xcquinox.pipeline.losses import make_loss
    from xcquinox.pipeline.padding import common_pad_target
    from xcquinox.pipeline.procmem import read_rss_gb

    with open(spec_path, "rb") as fh:
        spec = load_pickle(fh)   # the run's own spec, written by the notebook
    spec.validate()
    # This loop builds its own groups and losses from the same helpers, so
    # the package entry's preflight (run_training) is bypassed here: the
    # same refusals apply, read from the module that defines them.
    T._preflight_refusals(spec)
    if getattr(spec, "respect_sc_flag", False):
        # the per-molecule loop scales a non-self-consistent group's energy
        # channels by nonsc_weight; this loop does not implement that, so a
        # spec that asks for it is refused rather than trained differently
        raise ValueError("train-accum does not implement respect_sc_flag; "
                         "the spec sets it")
    ckpt = spec.checkpoint_dir
    os.makedirs(ckpt, exist_ok=True)
    if os.path.isfile(os.path.join(ckpt, "completion.json")):
        print(json.dumps({"kind": "done", "already_complete": True}), flush=True)
        return 0

    t0 = time.time()
    model = T._build_model(spec)
    loss = make_loss(spec.loss_name, molecules=spec.molecules,
                     pbe_anchor_weight=spec.pbe_anchor_weight,
                     pbe_anchor_sample=spec.pbe_anchor_sample,
                     **spec.loss_kwargs_dict)
    batch = T._build_batch(spec, loss)
    cw = T._effective_channel_weights(spec.channel_weights_dict)
    relative = spec.loss_metric == "relative"
    pad = (common_pad_target(batch["mol_data"])
           if getattr(spec, "pad_group_to_common_shape", False) else None)
    groups = T._training_groups(spec)
    prepared = [(g["label"], *T._build_group_loss_and_batch(spec, g, batch), cw)
                for g in groups]
    n_groups = len(prepared)
    mix_seed = bool(getattr(spec, "seed_mix_atomic", False))
    if mix_seed:
        T._require_minao_seed(prepared)
    rng = np.random.RandomState(spec.seed)
    mix_rng = (np.random.RandomState(T._mixer_seed(spec.seed)) if mix_seed
               else None)
    order = np.arange(n_groups)
    n_epochs = int(spec.n_steps)

    optimizer = T.build_optimizer(
        lr_start=spec.lr_start, lr_end=spec.lr_end, n_steps=n_epochs,
        lr_decay_start=spec.lr_decay_start, grad_clip=spec.grad_clip,
        weight_decay=spec.weight_decay, optimizer=spec.optimizer)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    plateau = None
    if spec.optimizer == "adam_plateau":
        plateau = T._PlateauController(
            patience=spec.plateau_patience, factor=spec.plateau_factor,
            min_lr=spec.lr_end, lr=spec.lr_start)
    tracker = T._BestModelTracker(window=1)   # one entry per epoch
    losses, aux_log = [], []
    start_epoch = 0

    if T._has_resume_checkpoint(ckpt):
        try:
            restored = T._load_resume_checkpoint(
                ckpt, model_skeleton=model, opt_state_skeleton=opt_state)
            if mix_seed and restored.get("mix_rng_state") is None:
                raise ValueError("the resume set carries no mix_rng_state")
        except Exception as exc:  # noqa: BLE001 -- a corrupt set starts fresh
            print(json.dumps({"kind": "resume_refused",
                              "error": f"{type(exc).__name__}: {exc}"}), flush=True)
        else:
            model = restored["model"]
            opt_state = restored["opt_state"]
            if plateau is not None and restored.get("plateau_state") is not None:
                plateau.from_state(restored["plateau_state"])
            rng.set_state(restored["rng_state"])
            if mix_rng is not None:
                mix_rng.set_state(restored["mix_rng_state"])
            order[:] = restored["order"]
            tracker = T._rebuild_epoch_tracker(restored["train_tracker"])
            losses = restored["losses"]
            aux_log = restored["aux_log"]
            start_epoch = restored["epoch"]
            print(json.dumps({"kind": "resumed", "epoch": start_epoch}), flush=True)

    live = {}

    def capture(epochs_done):
        live.update(
            model=model, opt_state=opt_state, rng_state=rng.get_state(),
            mix_rng_state=mix_rng.get_state() if mix_rng is not None else None,
            order=list(order), epoch=epochs_done, update=epochs_done,
            losses=list(losses), aux_log=list(aux_log),
            plateau_state=plateau.state() if plateau is not None else None,
            train_best_loss=tracker.best_loss, train_recent=list(tracker._recent),
            train_window=tracker.window, train_best_model=tracker.best_model)

    def flush():
        if not live:
            return
        T._write_resume_checkpoint(
            ckpt, model=live["model"], opt_state=live["opt_state"],
            rng_state=live["rng_state"], order=live["order"],
            mix_rng_state=live["mix_rng_state"],
            train_best_loss=live["train_best_loss"],
            train_recent=live["train_recent"], train_window=live["train_window"],
            train_best_model=live["train_best_model"], val_present=False,
            val_best_mae=None, val_finite_metrics=None, val_best_model=None,
            epoch=live["epoch"], update=live["update"], losses=live["losses"],
            aux_log=live["aux_log"], early_stopped=False, arch=spec.arch,
            plateau_state=live["plateau_state"])

    def on_term(signum, frame):  # noqa: ARG001 -- signal-handler signature
        try:
            flush()
        finally:
            os._exit(143)

    capture(start_epoch)
    signal.signal(signal.SIGTERM, on_term)

    for epoch in range(start_epoch, n_epochs):
        rng.shuffle(order)
        acc = None
        group_losses = []
        for position, gi in enumerate(order):
            label, gloss, gbatch, gcw = prepared[gi]
            if mix_seed:
                gbatch = T._mix_seed_batch(gbatch, mix_rng)
            # One optimizer step per epoch but one SCF per group evaluation:
            # the Fock-noise counter advances per evaluation so no two of them
            # share a draw (the molecule id was stamped by the batch build).
            gbatch = T._stamp_fock_noise_update(
                gbatch, epoch * n_groups + position)
            (loss_val, comps), grads = defused_value_and_grad(
                gloss, model, gbatch, gcw, relative, pad_target=pad)
            T._abort_if_nonfinite(loss_val, comps, loop="grad_accum", step=epoch,
                                  group=label, grads=grads)
            acc = grads if acc is None else jax.tree_util.tree_map(jnp.add, acc, grads)
            loss_py = float(loss_val)
            group_losses.append(loss_py)
            rss_gb, hwm_gb = read_rss_gb()
            aux_log.append({"step": epoch, "epoch": epoch, "group": label,
                            "loss": loss_py,
                            "aux": {k: float(v) for k, v in comps.items()},
                            "update_scheme": "grad_accum_mean",
                            "rss_gb": rss_gb, "hwm_gb": hwm_gb})
        mean_grads = jax.tree_util.tree_map(lambda g: g / n_groups, acc)
        updates, opt_state = optimizer.update(mean_grads, opt_state,
                                              T._trainable_params(model))
        model = eqx.apply_updates(model, updates)
        epoch_loss = float(np.mean(group_losses))
        losses.append(epoch_loss)
        # The best-loss snapshot: the parameters this epoch ends with, scored
        # on every group at fixed inputs (the group losses above were taken at
        # the pre-step parameters on mixed seeds); the plateau controller
        # below keeps the epoch mean of the group losses.
        epoch_score = T._epoch_end_score(prepared, model, relative, pad, False)
        tracker.update(epoch_score, model)
        aux_log.append({"step": epoch, "epoch": epoch, "group": "__epoch_score__",
                        "epoch_score": epoch_score,
                        "update_scheme": "grad_accum_mean"})
        new_lr = None
        if plateau is not None:
            metric = float(np.sqrt(max(0.0, epoch_loss)) * 1000.0)
            old_lr = plateau.lr
            new_lr = plateau.update(metric)
            if new_lr != old_lr:
                opt_state = T._set_learning_rate(opt_state, new_lr)
            aux_log.append({"step": epoch, "epoch": epoch, "group": "__plateau__",
                            "epoch_loss": epoch_loss, "plateau_metric": metric,
                            "lr": new_lr, "update_scheme": "grad_accum_mean"})
        capture(epoch + 1)
        flush()
        print(json.dumps({"kind": "step", "epoch": epoch + 1, "total": n_epochs,
                          "loss": epoch_loss, "lr": new_lr,
                          "elapsed_s": time.time() - t0}), flush=True)

    T._save_artifacts(spec, model, losses, aux_log, time.time() - t0,
                      best_model=tracker.best_model,
                      extra_metadata={"update_scheme_effective": "grad_accum_mean",
                                      "n_groups_averaged": n_groups,
                                      "epochs_run": n_epochs,
                                      # one optimizer step per epoch: the aux
                                      # log's "step" is the optimizer step,
                                      # equal to the epoch index here (the
                                      # per-molecule loop counts one per group)
                                      "aux_log_step_unit": "optimizer step = epoch"})
    T._finalize_completion(ckpt, early_stopped=False, epochs_run=n_epochs)
    print(json.dumps({"kind": "done", "elapsed_s": time.time() - t0}), flush=True)
    return 0


# ---------------------------------------------------------------------------
# Child: validation readout (both prongs)
# ---------------------------------------------------------------------------

def tracked_pt_barrier_ha() -> float:
    """The tracked CCSD(T) cation barrier, E(proton-transferred) -
    E(hemibonded), from 01wf subat_ref.traj; the frame names are taken from the
    geometry builders the path points are made from (cation_pt from
    _make_pt_atoms, cation_hb from _make_hb_atoms), so the readout's
    cation_pt - cation_hb and this reference cannot pair the wrong frames."""
    from ase.io import read as ase_read
    from xcquinox.pipeline.subset_selection import _make_hb_atoms, _make_pt_atoms
    hb_name = _make_hb_atoms().info["name"]
    pt_name = _make_pt_atoms().info["name"]
    frames = ase_read(str(TRACKED_PT_TRAJ), index=":")
    e = {f.info.get("name"): float(f.info["calc_energy"]) for f in frames}
    return e[pt_name] - e[hb_name]


def _reaction_family(rxn) -> str:
    """WTMAD-2 subset of a training reaction: an atomization (every product a
    bare element symbol), a barrier otherwise; ionization pairs are their own
    family."""
    import re
    if all(re.fullmatch(r"[A-Z][a-z]?", p) for p in rxn["products"]):
        return "ae"
    return "barrier"


def _rho_on_grid(dm, ao):
    import numpy as np
    dm = np.asarray(dm)
    if dm.ndim == 3:
        dm = dm[0] + dm[1]
    return np.einsum("ij,gi,gj->g", dm, ao, ao)


def _reaction_errors(energies, reactions, ip_pairs):
    rows = {}
    for r in reactions:
        names = [*r["reactants"], *r["products"]]
        if not all(n in energies for n in names):
            continue
        pred = sum(c * energies[n] for n, c in zip(names, r["coeffs"]))
        rows[r["name"]] = {"pred_ha": pred, "ref_ha": float(r["e_rxn_ref"]),
                           "err_kcal": (pred - float(r["e_rxn_ref"])) * KCAL_PER_HA}
    for p in ip_pairs:
        if p["cation"] in energies and p["neutral"] in energies:
            pred = energies[p["cation"]] - energies[p["neutral"]]
            rows[p["name"]] = {"pred_ha": pred, "ref_ha": float(p["ip_ref"]),
                               "err_kcal": (pred - float(p["ip_ref"])) * KCAL_PER_HA}
    return rows


def _ed(e_kcal, eps, gamma):
    if not (math.isfinite(e_kcal) and e_kcal > 0 and math.isfinite(eps) and eps > 0):
        return float("nan")
    return 2.0 / (1.0 / e_kcal + 1.0 / (gamma * eps))


def summarize_readout(species, spec_loss_kwargs, pt_names, tracked_ha, *,
                      energy_key="E_final", eps_key="eps"):
    """Reaction errors, PT barrier, density errors and the DFS ED metric from
    per-species results (``species``: name -> dict with ``energy_key`` and,
    where a reference density exists, ``eps_key``)."""
    from xcquinox.pipeline.eval_holdout import paper_wtmad2

    energies = {n: float(s[energy_key]) for n, s in species.items()
                if s.get(energy_key) is not None}
    rx = _reaction_errors(energies, spec_loss_kwargs["bh76_reactions"],
                          spec_loss_kwargs["ip13_pairs"])
    # WTMAD-2 (paper form) over the training reactions, one subset per family
    # (barriers, atomization, ionization): with a single subset the formula
    # reduces to the plain MAE
    family = {r["name"]: _reaction_family(r) for r in spec_loss_kwargs["bh76_reactions"]}
    family.update({p["name"]: "ip" for p in spec_loss_kwargs["ip13_pairs"]})
    w2_rows = [{"subset": family[k], "de_ref": v["ref_ha"] * KCAL_PER_HA,
                "de_nn": v["pred_ha"] * KCAL_PER_HA} for k, v in rx.items()]
    wtmad2, _ = paper_wtmad2(w2_rows) if w2_rows else (float("nan"), {})
    corpus_eps = [float(s[eps_key]) for n, s in species.items()
                  if n not in pt_names and s.get(eps_key) is not None]
    pt_eps = [float(s[eps_key]) for n, s in species.items()
              if n in pt_names and s.get(eps_key) is not None]
    eps_c = float(sum(corpus_eps) / len(corpus_eps)) if corpus_eps else float("nan")
    eps_p = float(sum(pt_eps) / len(pt_eps)) if pt_eps else float("nan")
    pt = {}
    if "cation_pt" in energies and "cation_hb" in energies:
        b = energies["cation_pt"] - energies["cation_hb"]
        pt = {"cation_barrier_ha": b, "tracked_ha": tracked_ha,
              "err_kcal": (b - tracked_ha) * KCAL_PER_HA}
    for path in ("zundel", "neutral"):
        vals = [energies[n] for n in pt_names if n.startswith(path) and n in energies]
        if vals:
            pt[f"{path}_span_ha"] = max(vals) - min(vals)
    pt_abs = abs(pt["err_kcal"]) if "err_kcal" in pt else float("nan")
    errs = [abs(v["err_kcal"]) for v in rx.values()]
    return {
        "reactions": rx,
        "mae_train_kcal": float(sum(errs) / len(errs)) if errs else float("nan"),
        "wtmad2_corpus_kcal": float(wtmad2),
        "pt": pt,
        "eps_mean_corpus": eps_c,
        "eps_mean_pt": eps_p,
        "ed_published": _ed(float(wtmad2), eps_c, GAMMA_PUBLISHED),
        "ed_own_axes": _ed(float(wtmad2), eps_c, GAMMA_OWN_AXES),
        "ed_pt_published": _ed(pt_abs, eps_p, GAMMA_PUBLISHED),
        "ed_pt_own_axes": _ed(pt_abs, eps_p, GAMMA_OWN_AXES),
    }


def readout_main(spec_path, model_path, pt_specs_path, pt_refs_dir, out_dir,
                 label) -> int:
    """Self-consistent validation of one network: one SCF per species (the 16
    corpus species and the PT path points) with the network's training
    solver, energy and density from the same SCF. Each species' result is
    written as soon as it exists (``species/<name>.json`` and ``.npz``) and is
    skipped on a rerun; ``readout.json`` assembles them, with the PBE and the
    reference-density baselines of the same records."""
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    import jax
    jax.config.update("jax_enable_x64", True)
    import numpy as np

    import xcquinox.pipeline as pipeline
    from xcquinox.pipeline import train as T
    from xcquinox.pipeline.checkpoint_class import load_pickle, load_trained_checkpoint
    from xcquinox.pipeline.evaluation import density_eps_terms
    from xcquinox.pipeline.models import AlecGGAModel
    from xcquinox.pipeline.oneshot import tail_weighted_mean_energy
    from xcquinox.pipeline.solver import run_scf

    with open(spec_path, "rb") as fh:
        spec = load_pickle(fh)       # the run's own spec
    with open(pt_specs_path, "rb") as fh:
        pt_specs = load_pickle(fh)   # the run's own PT path points
    sc = spec.solver_config
    if model_path == "clone":
        model = T._build_model(spec)
    else:
        model = load_trained_checkpoint(str(model_path),
                                        AlecGGAModel.from_arch(spec.arch, seed=0))
    out = Path(out_dir)
    (out / "species").mkdir(parents=True, exist_ok=True)
    descriptors = spec.arch.materialize_descriptors()
    keys = set()
    for d in descriptors:
        keys |= set(d.required_mol_keys)
    keys |= {"cderi" if sc.density_fit else "eri", "ao_grid_deriv", "ao_grid",
             "dm_pbe", "grid_weights"}
    keys = tuple(sorted(keys))
    pt_names = [s.name for s in pt_specs]
    t_all = time.time()
    species = {}
    for kind, ms in [("corpus", m) for m in spec.molecules] + [("pt", m) for m in pt_specs]:
        jpath = out / "species" / f"{ms.name}.json"
        if jpath.is_file():
            species[ms.name] = json.loads(jpath.read_text())
            continue
        t0 = time.time()
        rec = pipeline.precompute_fixed_density_data(
            ms, required_keys=keys, descriptors=descriptors,
            auxbasis=sc.auxbasis if sc.density_fit else None,
            orientation_lock_strength=sc.orientation_lock_strength,
            seed_source=sc.seed_source, seed_cache_dir=sc.seed_cache_dir,
            seed_density_fit=bool(sc.density_fit), seed_allow_generate=True)
        res = run_scf(sc, model, rec, forward_only=True)
        trace = (np.asarray(res.energy_trace) if res.energy_trace is not None
                 else None)
        e_final = float(res.total_energy)
        e_tail = (float(tail_weighted_mean_energy(res.energy_trace, sc.scf_loss_tail,
                                                  sc.scf_loss_weight_power))
                  if trace is not None else e_final)
        ao = np.asarray(rec["ao_grid"])
        w = np.asarray(rec["grid_weights"])
        dm = np.asarray(res.density_matrix)
        rho = _rho_on_grid(dm, ao)
        dm_ref = None
        if kind == "corpus" and rec.get("dm_target") is not None:
            dm_ref = np.asarray(rec["dm_target"])
        elif kind == "pt":
            ref_npz = Path(pt_refs_dir) / f"{ms.name}.npz"
            if ref_npz.is_file():
                with np.load(ref_npz) as z:
                    dm_ref = np.asarray(z["dm"])
        row = {"name": ms.name, "kind": kind, "label": label,
               "E_final": e_final, "E_tail": e_tail, "E_pbe": float(rec["E_pbe"]),
               "converged": bool(res.converged), "cycles": int(res.cycles_run),
               "last_dE": (float(abs(trace[-1] - trace[-2]))
                           if trace is not None and len(trace) > 1 else None),
               "energy_trace": trace.tolist() if trace is not None else None,
               "eps": None, "eps_pbe": None, "n_electrons_ref": None,
               "seconds": None}
        arrays = {"dm_scf": dm, "rho_scf": rho, "grid_weights": w}
        if dm_ref is not None:
            rho_ref = _rho_on_grid(dm_ref, ao)
            eps, n_e, _ = density_eps_terms(rho, rho_ref, w)
            eps_pbe, _, _ = density_eps_terms(_rho_on_grid(rec["dm_pbe"], ao), rho_ref, w)
            row.update(eps=float(eps), eps_pbe=float(eps_pbe), n_electrons_ref=float(n_e),
                       dm_ref_maxdiff=float(np.max(np.abs(dm - dm_ref))))
            arrays["rho_ref"] = rho_ref
        row["seconds"] = time.time() - t0
        tmp = out / "species" / f"{ms.name}.npz.tmp.npz"
        np.savez(tmp, **arrays)
        os.replace(tmp, out / "species" / f"{ms.name}.npz")
        write_json(jpath, row)
        species[ms.name] = row
        print(json.dumps({"kind": "species", "name": ms.name, "E": e_final,
                          "converged": row["converged"], "eps": row["eps"],
                          "seconds": row["seconds"]}), flush=True)
        del rec

    tracked = tracked_pt_barrier_ha()
    lk = spec.loss_kwargs_dict
    payload = {
        "arch": spec.arch.name, "label": label, "model": str(model_path),
        "solver": {"max_cycles": sc.max_cycles, "density_fit": sc.density_fit,
                   "seed_source": sc.seed_source},
        "network": summarize_readout(species, lk, pt_names, tracked),
        "pbe": summarize_readout(species, lk, pt_names, tracked,
                                 energy_key="E_pbe", eps_key="eps_pbe"),
        "scf": {"n_species": len(species),
                "n_converged": sum(1 for s in species.values() if s["converged"]),
                "max_last_dE": max((s["last_dE"] or 0.0) for s in species.values())},
        "eps": {n: s["eps"] for n, s in species.items() if s["eps"] is not None},
        "elapsed_s": time.time() - t_all,
    }
    write_json(out / "readout.json", payload)
    print(json.dumps({"kind": "done", "elapsed_s": payload["elapsed_s"]}), flush=True)
    return 0


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def assemble(prong_root, clone_root=None, ledger=None) -> dict:
    """Collect every readout present into ``results/`` tables (idempotent):
    ``summary.csv`` (one row per architecture: status, the network's and the
    clone's and PBE's training-reaction MAE, PT error, mean density errors and
    ED), ``reactions.csv`` and ``density.csv`` (long form), ``status.json``."""
    prong_root = Path(prong_root)
    res_dir = prong_root / "results"
    res_dir.mkdir(parents=True, exist_ok=True)
    status = ledger.summary() if ledger is not None else {"archs": {}}
    readouts = {p.parent.name: json.loads(p.read_text())
                for p in sorted((prong_root / "readout").glob("*/readout.json"))}
    clones = {}
    if clone_root is not None:
        clones = {p.parent.name: json.loads(p.read_text())
                  for p in sorted(Path(clone_root).glob("*/readout.json"))}
    names = list(status["archs"]) or sorted(readouts)
    summary, reactions, density = [], [], []

    def _pick(block, key):
        # an architecture without a readout (failed, pending) has empty
        # cells, not NaN: absent is not a numerical result
        v = (block or {}).get(key)
        return "" if v is None or (isinstance(v, float) and math.isnan(v)) else v

    for n in names:
        r = readouts.get(n)
        c = clones.get(n)
        net = r["network"] if r else None
        cl = c["network"] if c else None
        pbe = (r or c or {}).get("pbe")
        summary.append({
            "arch": n, "state": status["archs"].get(n, {}).get("state"),
            "stage": status["archs"].get(n, {}).get("stage"),
            "mae_train_kcal": _pick(net, "mae_train_kcal"),
            "pt_err_kcal": _pick((net or {}).get("pt"), "err_kcal"),
            "eps_mean_corpus": _pick(net, "eps_mean_corpus"),
            "eps_mean_pt": _pick(net, "eps_mean_pt"),
            "ed_published": _pick(net, "ed_published"),
            "ed_own_axes": _pick(net, "ed_own_axes"),
            "clone_mae_train_kcal": _pick(cl, "mae_train_kcal"),
            "clone_pt_err_kcal": _pick((cl or {}).get("pt"), "err_kcal"),
            "clone_ed_published": _pick(cl, "ed_published"),
            "pbe_mae_train_kcal": _pick(pbe, "mae_train_kcal"),
            "pbe_pt_err_kcal": _pick((pbe or {}).get("pt"), "err_kcal"),
            "pbe_ed_published": _pick(pbe, "ed_published"),
            "scf_converged": (f"{r['scf']['n_converged']}/{r['scf']['n_species']}"
                              if r else ""),
        })
        for who, block in (("network", net), ("clone", cl), ("pbe", pbe)):
            for rx, v in ((block or {}).get("reactions") or {}).items():
                reactions.append({"arch": n, "functional": who, "reaction": rx,
                                  "err_kcal": v["err_kcal"]})
        for who, src in (("network", r), ("clone", c)):
            for sp, eps in ((src or {}).get("eps") or {}).items():
                density.append({"arch": n, "functional": who, "species": sp,
                                "eps": eps})
    for fname, rows in (("summary.csv", summary), ("reactions.csv", reactions),
                        ("density.csv", density)):
        if not rows:
            continue
        tmp = res_dir / (fname + ".tmp")
        with open(tmp, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
        os.replace(tmp, res_dir / fname)
    write_json(res_dir / "status.json", status)
    return {"n_readouts": len(readouts), "n_clone_readouts": len(clones),
            "summary": summary}


# ---------------------------------------------------------------------------
# Command line (the child entry points)
# ---------------------------------------------------------------------------

def _cli(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("train-accum")
    p.add_argument("spec_path")
    p = sub.add_parser("readout")
    p.add_argument("--spec", required=True)
    p.add_argument("--model", required=True, help="model.eqx, or 'clone'")
    p.add_argument("--pt-specs", required=True)
    p.add_argument("--pt-refs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--label", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "train-accum":
        return train_accum_main(args.spec_path)
    return readout_main(args.spec, args.model, args.pt_specs, args.pt_refs,
                        args.out, args.label)


if __name__ == "__main__":
    # leave through the package's shared hard exit, as the per-molecule
    # worker does (_train_one_spec): the interpreter's own teardown can abort
    # after the outputs are written (cluster/_exit.py), which would record a
    # finished stage as failed
    from xcquinox.pipeline.cluster._exit import run_and_exit
    run_and_exit(_cli)
