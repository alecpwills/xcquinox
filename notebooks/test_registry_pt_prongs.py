"""The registry proton-transfer prongs' runtime support (notebooks/registry_pt_prongs.py):
the work queue, the worker loop's failure handling, the architecture set and
its identities, the campaign configuration the harness validates, and the
readout summary and assembly."""
from __future__ import annotations

import json
import math
import multiprocessing as mp
import sys
import time
from pathlib import Path

import pytest

_NOTEBOOKS = Path(__file__).resolve().parent
sys.path.insert(0, str(_NOTEBOOKS))
import registry_pt_prongs as rpp  # noqa: E402


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------

def test_ledger_claims_in_order_and_ensure_is_idempotent(tmp_path):
    led = rpp.Ledger(tmp_path / "ledger.json")
    led.ensure(["a", "b", "c"])
    led.ensure(["a", "b", "c", "d"])           # a second job adds only what is new
    assert led.read()["order"] == ["a", "b", "c", "d"]
    assert led.claim_next("j1", "j1:w0", "h") == "a"
    assert led.claim_next("j1", "j1:w1", "h") == "b"
    led.mark("a", rpp.VALIDATED)
    assert led.summary()["counts"] == {"validated": 1, "running": 1, "pending": 2}


def test_ledger_prep_failure_enters_as_failed_and_is_never_claimed(tmp_path):
    led = rpp.Ledger(tmp_path / "ledger.json")
    led.ensure(["a", "b"], failed={"a": "pretrain"})
    assert led.read()["archs"]["a"]["state"] == rpp.FAILED
    assert led.claim_next("j", "j:w0", "h") == "b"
    assert led.claim_next("j", "j:w0", "h") is None
    assert led.reset_failed() == ["a"]
    assert led.claim_next("j", "j:w0", "h") == "a"


def test_a_repaired_preparation_returns_the_entry_to_pending(tmp_path):
    """A later job whose preparation record no longer names an architecture
    (a prep rerun repaired it) returns its prep failure to pending; a
    training failure is never touched by the preparation record."""
    led = rpp.Ledger(tmp_path / "ledger.json")
    led.ensure(["a", "b", "c"], failed={"a": "certificate", "b": "pretrain"})
    assert led.claim_next("j", "j:w0", "h") == "c"
    led.mark("c", rpp.FAILED, stage="train", returncode=1)
    led.ensure(["a", "b", "c"], failed={"b": "pretrain"})
    e = led.read()["archs"]
    assert e["a"]["state"] == rpp.PENDING
    assert (e["b"]["state"], e["b"]["stage"]) == (rpp.FAILED, "prep")
    assert (e["c"]["state"], e["c"]["stage"]) == (rpp.FAILED, "train")
    # no record known (None) keeps the prep failures; an empty record clears them
    led.ensure(["a", "b", "c"], failed=None)
    assert led.read()["archs"]["b"]["state"] == rpp.FAILED
    assert led.claim_next("j", "j:w0", "h") == "a"
    led.ensure(["a", "b", "c"], failed={})
    assert led.read()["archs"]["b"]["state"] == rpp.PENDING


def test_a_worker_that_lost_its_claim_records_nothing_and_moves_on(tmp_path):
    """A claim reclaimed and taken by another job while this worker's child
    runs: the worker's records are refused (mark's owner check), it starts
    no further stage of that architecture, and it claims the next."""
    led = rpp.Ledger(tmp_path / "ledger.json")
    led.ensure(["taken", "next"])
    assert led.mark("next", rpp.PENDING, owner="nobody") is False

    def say(msg):
        if "taken: train started" in msg:
            led.reclaim_stale(0.0)                       # the stale-heartbeat path
            assert led.claim_next("other", "other:w0", "h2") == "taken"

    def stages_for(name):
        return [("train", _py("pass"), None, None, True),
                ("readout", _py("import sys; sys.exit(9)"), None, None, True)]

    handled = rpp.run_worker(led, stages_for, job="j", worker="j:w0", host="h",
                             log_dir=tmp_path / "logs", say=say)
    e = led.read()["archs"]
    assert handled == ["taken", "next"]
    assert (e["taken"]["state"], e["taken"]["worker"]) == ("running", "other:w0")
    assert not (tmp_path / "logs" / "taken.readout.log").exists()
    assert e["next"]["state"] == "failed"               # its readout exits 9


def _claim_all(path, worker, out_q):
    led = rpp.Ledger(path)
    got = []
    while True:
        n = led.claim_next("job", worker, "host")
        if n is None:
            break
        got.append(n)
    out_q.put(got)


def test_ledger_never_double_claims_across_processes(tmp_path):
    path = tmp_path / "ledger.json"
    names = [f"arch{i:02d}" for i in range(40)]
    rpp.Ledger(path).ensure(names)
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    procs = [ctx.Process(target=_claim_all, args=(path, f"w{i}", q)) for i in range(4)]
    for p in procs:
        p.start()
    claimed = [n for _ in procs for n in q.get(timeout=60)]
    for p in procs:
        p.join(timeout=60)
    assert sorted(claimed) == names          # every name once, none twice


def test_reclaim_returns_dead_and_stale_claims_but_not_the_callers_own(tmp_path):
    led = rpp.Ledger(tmp_path / "ledger.json")
    led.ensure(["a", "b", "c", "d"])
    led.claim_next("111", "111:w0", "h")      # a: job gone
    led.claim_next("222", "222:w0", "h")      # b: job alive, fresh
    led.claim_next("333", "333:w0", "h")      # c: the caller's own job
    led.claim_next("222", "222:w1", "h")      # d: job alive, heartbeat stale
    with led._locked() as data:
        data["archs"]["d"]["heartbeat"] = time.time() - 7200
    back = led.reclaim_stale(1800.0, live_jobs={"222", "333"}, own_job="333")
    assert back == ["a", "d"]
    states = {n: e["state"] for n, e in led.read()["archs"].items()}
    assert states == {"a": "pending", "b": "running", "c": "running", "d": "pending"}
    # a scheduler reply that omits the caller's own (running) job is not
    # trusted: no claim is judged dead from it, an empty reply included
    led3 = rpp.Ledger(tmp_path / "l3.json")
    led3.ensure(["p", "q"])
    led3.claim_next("444", "444:w0", "h")
    for reply in (set(), {"555"}):
        assert led3.reclaim_stale(1800.0, live_jobs=reply, own_job="666") == []
    assert led3.reclaim_stale(1800.0, live_jobs={"666"}, own_job="666") == ["p"]
    # without a scheduler list only the heartbeat decides
    led2 = rpp.Ledger(tmp_path / "l2.json")
    led2.ensure(["x"])
    led2.claim_next("999", "999:w0", "h")
    assert led2.reclaim_stale(1800.0, live_jobs=None) == []


# ---------------------------------------------------------------------------
# Worker loop
# ---------------------------------------------------------------------------

def _py(code):
    return [sys.executable, "-c", code]


def test_worker_records_failures_and_moves_on(tmp_path):
    led = rpp.Ledger(tmp_path / "ledger.json")
    led.ensure(["ok", "train_fails", "readout_fails", "killed", "clone_fails"])

    def stages_for(name):
        train = _py("import sys; sys.exit(3)") if name == "train_fails" else (
            _py("import os, signal; os.kill(os.getpid(), signal.SIGKILL)")
            if name == "killed" else _py("print('trained')"))
        readout = _py("import sys; sys.exit(5)") if name == "readout_fails" else _py("pass")
        clone = _py("import sys; sys.exit(7)") if name == "clone_fails" else _py("pass")
        return [("train", train, None, None, True),
                ("readout", readout, None, None, True),
                ("clone_readout", clone, None, None, False)]

    handled = rpp.run_worker(led, stages_for, job="j", worker="j:w0", host="h",
                             log_dir=tmp_path / "logs", heartbeat_s=0.2,
                             say=lambda *_: None)
    assert handled == ["ok", "train_fails", "readout_fails", "killed", "clone_fails"]
    e = led.read()["archs"]
    assert e["ok"]["state"] == "validated"
    assert (e["train_fails"]["state"], e["train_fails"]["stage"],
            e["train_fails"]["returncode"]) == ("failed", "train", 3)
    assert (e["readout_fails"]["state"], e["readout_fails"]["stage"]) == ("failed", "readout")
    assert (e["killed"]["state"], e["killed"]["signal"]) == ("failed", 9)
    assert e["clone_fails"]["state"] == "validated"
    assert e["clone_fails"]["optional_failures"][0]["returncode"] == 7
    assert "trained" in Path(tmp_path / "logs" / "ok.train.log").read_text()


def test_worker_returns_an_interrupted_architecture_to_pending(tmp_path):
    led = rpp.Ledger(tmp_path / "ledger.json")
    led.ensure(["a", "b"])

    def stages_for(name):
        return [("train", _py("import sys; sys.exit(143)"), None, None, True)]

    handled = rpp.run_worker(led, stages_for, job="j", worker="j:w0", host="h",
                             log_dir=tmp_path / "logs", say=lambda *_: None)
    assert handled == ["a"]                   # the worker stops with its job
    assert led.read()["archs"]["a"]["state"] == "pending"
    assert led.read()["archs"]["b"]["state"] == "pending"


def _wait_for(path, timeout=30.0):
    t0 = time.time()
    while not Path(path).exists():
        if time.time() - t0 > timeout:
            raise AssertionError(f"{path} never appeared")
        time.sleep(0.05)


# a stage child: marks itself started, exits 143 on SIGTERM (a training run
# flushing its resume set), otherwise sleeps far past the test
_TERM_CHILD = (
    "import os, signal, sys, time\n"
    "def _term(*_):\n"
    "    open({mark!r} + '.term', 'w').write(str(os.getpid()))\n"
    "    os._exit(143)\n"
    "signal.signal(signal.SIGTERM, _term)\n"
    "open({mark!r}, 'w').write(str(os.getpid()))\n"
    "time.sleep(120)\n")


def test_sigterm_to_the_group_drains_the_worker_and_records_pending(tmp_path):
    """The train stage's chain with real signals: a kernel-like process that
    installed the handler runs a worker thread; SIGTERM to its process group
    reaches the child (exit 143) and the kernel, which survives, records the
    architecture pending, claims nothing more and exits normally."""
    import os
    import signal
    import subprocess
    mark = str(tmp_path / "child_started")
    child = _TERM_CHILD.format(mark=mark)
    kernel = (
        "import sys, threading\n"
        f"sys.path.insert(0, {str(_NOTEBOOKS)!r})\n"
        "import registry_pt_prongs as rpp\n"
        "rpp.install_term_handler()\n"
        f"led = rpp.Ledger({str(tmp_path / 'ledger.json')!r})\n"
        "led.ensure(['a', 'b'])\n"
        f"stages = lambda n: [('train', [sys.executable, '-c', {child!r}], None, None, True),\n"
        f"                    ('readout', [sys.executable, '-c', 'pass'], None, None, True)]\n"
        "th = threading.Thread(target=rpp.run_worker, args=(led, stages), daemon=True,\n"
        "    kwargs=dict(job='j', worker='j:w0', host='h',\n"
        f"                log_dir={str(tmp_path / 'logs')!r}, heartbeat_s=0.2, stop=rpp.STOP))\n"
        "th.start(); th.join()\n"
        "print('kernel finished', rpp.STOP.is_set())\n")
    proc = subprocess.Popen([sys.executable, "-c", kernel], start_new_session=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        _wait_for(mark)
        os.killpg(proc.pid, signal.SIGTERM)
        out, _ = proc.communicate(timeout=60)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
    assert proc.returncode == 0, out
    assert "kernel finished True" in out
    assert Path(mark + ".term").exists()              # the child got the TERM
    e = json.loads((tmp_path / "ledger.json").read_text())["archs"]
    assert (e["a"]["state"], e["a"]["returncode"]) == ("pending", 143)
    assert e["b"]["state"] == "pending" and e["b"].get("worker") is None


def test_a_child_gets_sigterm_when_its_parent_dies(tmp_path):
    """run_child's parent-death signal: SIGKILL the process that started the
    child, and the child still receives SIGTERM (it flushes and exits)
    instead of running on as an orphan."""
    import os
    import signal
    import subprocess
    mark = str(tmp_path / "child_started")
    child = _TERM_CHILD.format(mark=mark)
    parent = (
        "import sys\n"
        f"sys.path.insert(0, {str(_NOTEBOOKS)!r})\n"
        "import registry_pt_prongs as rpp\n"
        f"rpp.run_child([sys.executable, '-c', {child!r}],\n"
        f"              log_path={str(tmp_path / 'child.log')!r})\n")
    proc = subprocess.Popen([sys.executable, "-c", parent], start_new_session=True)
    child_pid = None
    try:
        _wait_for(mark)
        child_pid = int(Path(mark).read_text())
        os.kill(proc.pid, signal.SIGKILL)
        proc.wait(timeout=30)
        _wait_for(mark + ".term", timeout=15)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
        if child_pid is not None:
            try:
                os.kill(child_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
    assert int(Path(mark + ".term").read_text()) == child_pid


def test_a_stop_before_a_stage_records_pending_or_validated(tmp_path):
    """Once the job is ending no stage starts: an architecture still owing a
    required stage returns to pending; one owing only the optional baseline
    is validated with the baseline noted as not run; nothing more is claimed."""
    import threading
    stop = threading.Event()
    led = rpp.Ledger(tmp_path / "ledger.json")
    led.ensure(["owes_readout", "owes_clone", "untouched"])

    def stages_for(name):
        return [("train", _py("pass"), None, None, True),
                ("readout", _py("pass"), None,
                 str(tmp_path / "done") if name == "owes_clone" else None, True),
                ("clone_readout", _py("pass"), None, None, False)]

    def say(msg):
        if "train started" in msg:
            stop.set()                       # the TERM lands during training

    (tmp_path / "done").write_text("")
    handled = rpp.run_worker(led, stages_for, job="j", worker="j:w0", host="h",
                             log_dir=tmp_path / "logs", say=say, stop=stop)
    e = led.read()["archs"]
    assert handled == ["owes_readout"]
    assert (e["owes_readout"]["state"], e["owes_readout"]["stage"]) == ("pending", "readout")
    stop.clear()
    led.mark("owes_readout", rpp.VALIDATED)
    handled = rpp.run_worker(led, stages_for, job="j", worker="j:w0", host="h",
                             log_dir=tmp_path / "logs", say=say, stop=stop)
    e = led.read()["archs"]
    assert handled == ["owes_clone"]
    assert e["owes_clone"]["state"] == "validated"
    assert e["owes_clone"]["optional_failures"][0]["reason"].startswith("not run")
    assert e["untouched"]["state"] == "pending"
    assert rpp.run_worker(led, stages_for, job="j", worker="j:w0", host="h",
                          log_dir=tmp_path / "logs", say=say,
                          stop=stop) == []      # set again by the last training


def test_interrupted_optional_stage_keeps_the_architecture_validated(tmp_path):
    led = rpp.Ledger(tmp_path / "ledger.json")
    led.ensure(["a", "b"])

    def stages_for(name):
        return [("train", _py("pass"), None, None, True),
                ("readout", _py("pass"), None, None, True),
                ("clone_readout", _py("import sys; sys.exit(143)"), None, None, False)]

    handled = rpp.run_worker(led, stages_for, job="j", worker="j:w0", host="h",
                             log_dir=tmp_path / "logs", say=lambda *_: None)
    e = led.read()["archs"]
    assert handled == ["a"]                     # the job is going down: stop claiming
    assert e["a"]["state"] == "validated"
    assert e["a"]["optional_failures"][0]["reason"] == "interrupted"
    assert e["b"]["state"] == "pending"


def test_a_worker_error_is_recorded_and_the_worker_goes_on(tmp_path):
    led = rpp.Ledger(tmp_path / "ledger.json")
    led.ensure(["broken", "ok"])

    def stages_for(name):
        if name == "broken":
            raise RuntimeError("cannot build the stage list")
        return [("train", _py("pass"), None, None, True)]

    handled = rpp.run_worker(led, stages_for, job="j", worker="j:w0", host="h",
                             log_dir=tmp_path / "logs", say=lambda *_: None)
    e = led.read()["archs"]
    assert handled == ["broken", "ok"]
    assert (e["broken"]["state"], e["broken"]["stage"]) == ("failed", "worker")
    assert "cannot build the stage list" in e["broken"]["reason"]
    assert e["ok"]["state"] == "validated"


def test_tracked_barrier_reads_the_frames_the_builders_name():
    from ase.io import read as ase_read
    frames = {f.info.get("name"): float(f.info["calc_energy"])
              for f in ase_read(str(rpp.TRACKED_PT_TRAJ), index=":")}
    assert rpp.tracked_pt_barrier_ha() == pytest.approx(frames["PTWD"] - frames["HBWD"])


def test_worker_skips_stages_already_done(tmp_path):
    led = rpp.Ledger(tmp_path / "ledger.json")
    led.ensure(["a"])
    done = tmp_path / "completion.json"
    done.write_text("{}")
    marker = tmp_path / "ran"

    def stages_for(name):
        return [("train", _py(f"open({str(marker)!r}, 'w')"), None, done, True),
                ("readout", _py("pass"), None, None, True)]

    rpp.run_worker(led, stages_for, job="j", worker="j:w0", host="h",
                   log_dir=tmp_path / "logs", say=lambda *_: None)
    assert not marker.exists()
    assert led.read()["archs"]["a"]["state"] == "validated"


def test_heartbeat_refreshes_a_long_child(tmp_path):
    led = rpp.Ledger(tmp_path / "ledger.json")
    led.ensure(["a"])
    led.claim_next("j", "j:w0", "h")
    t_claim = led.read()["archs"]["a"]["heartbeat"]
    rc = rpp.run_child(_py("import time; time.sleep(1.0)"), log_path=tmp_path / "x.log",
                       heartbeat=lambda: led.heartbeat("a", "j:w0"), heartbeat_s=0.2)
    assert rc == 0
    assert led.read()["archs"]["a"]["heartbeat"] > t_claim


# ---------------------------------------------------------------------------
# Architectures, identities, campaign
# ---------------------------------------------------------------------------

def test_architecture_set_mlps_3x16_kans_2x8_log_transformed():
    from xcquinox.pipeline.config import get_architecture
    cfgs = rpp.arch_set()
    assert list(cfgs) == list(rpp.ALL_ARCHS)
    assert rpp.census(cfgs) == {"total": 26, "gga_mlp": 17, "meta_gga": 6, "kan": 3}
    assert set(rpp.SMOKE_ARCHS) <= set(cfgs) and set(rpp.DEMO_ARCHS) <= set(cfgs)
    kan = cfgs["deep_kan_atomctx_2x8"]
    assert (kan.network, kan.kan_grid, kan.kan_order, kan.depth, kan.nodes) == (
        "kan", 5, 3, 2, 8)
    assert [d.name for d in kan.descriptors] == ["atom_context"]
    assert {rpp.arch_class(c) for n, c in cfgs.items() if "kan" in n} == {"kan"}
    assert {rpp.arch_class(c) for n, c in cfgs.items() if "kan" not in n} == {"standard"}
    meta = cfgs["deep_atomctx_mgga_3x16"]
    assert meta.meta_gga and sorted(d.name for d in meta.descriptors) == [
        "atom_context", "metagga"]
    # a KAN at the MLP shape, and an MLP off 3x16, are both refused
    with pytest.raises(ValueError, match="shape"):
        rpp.census({"deep_kan_2x6": get_architecture("deep_kan_2x6")})
    with pytest.raises(ValueError, match="shape"):
        rpp.census({"deep": get_architecture("deep")})


def test_claim_respects_the_training_class(tmp_path):
    led = rpp.Ledger(tmp_path / "ledger.json")
    led.ensure(["mlp1", "kan1", "mlp2"])
    assert led.claim_next("j", "j:k0", "h", accept=["kan1"]) == "kan1"
    assert led.claim_next("j", "j:k0", "h", accept=["kan1"]) is None
    assert led.claim_next("j", "j:s0", "h", accept=["mlp1", "mlp2"]) == "mlp1"


def test_unlisted_architectures_register_in_this_process_only():
    from xcquinox.pipeline import config as _config
    added = rpp.register_unlisted()
    try:
        for n in rpp.UNLISTED_ARCHS:
            assert _config.get_architecture(n).name == n
        assert rpp.register_unlisted() == []        # idempotent
    finally:
        for n in added:
            _config.ARCHITECTURES.pop(n, None)


def test_solver_seeds_by_rung_and_carries_the_identity():
    cfgs = rpp.arch_set(["deep_3x16", "deep_mgga_3x16", "deep_atomctx_mgga_3x16"])
    for name, cfg in cfgs.items():
        sc = rpp.make_solver(cfg, rpp.PRODUCTION_IDENTITY, seed_cache_dir="/x")
        assert sc.seed_source == ("scan" if cfg.meta_gga else "pbe"), name
        assert sc.density_fit and sc.max_cycles == 25 and sc.seed_cache_dir == "/x"
        assert sc.orientation_lock_strength == 3.0e-5


@pytest.mark.parametrize("ident", [rpp.PRODUCTION_IDENTITY, rpp.SMOKE_IDENTITY],
                         ids=["production", "smoke"])
def test_campaign_validates_at_both_identities(ident, tmp_path):
    import yaml
    from xcquinox.pipeline.cluster.domain import get_domain_profile
    from xcquinox.pipeline.cluster.grid_config import (load_grid_config,
                                                       validate_grid_semantics)
    camp = rpp.make_campaign_dict(str(tmp_path), list(rpp.REGISTERED_ARCHS), ident,
                                  seed_cache_dir=str(tmp_path / "seed"))
    assert rpp.GEA_ARCH not in camp["sweep"]["arch"]
    assert len(camp["sweep"]["arch"]) == len(rpp.REGISTERED_ARCHS) - 1
    assert camp["hyperparams"]["n_steps"] == ident["epochs"]
    waived = "allow_irreproducible_degenerate" in camp["inputs"]
    assert waived == (ident["grid_level"] < 3)
    path = tmp_path / "campaign.yaml"
    path.write_text(yaml.safe_dump(camp, sort_keys=False))
    cfg = load_grid_config(str(path))
    validate_grid_semantics(cfg, get_domain_profile(cfg.domain_profile))
    assert (cfg.inputs.basis, cfg.inputs.grid_level, cfg.inputs.density_fit) == (
        ident["basis"], ident["grid_level"], ident["density_fit"])


# ---------------------------------------------------------------------------
# Readout summary and assembly
# ---------------------------------------------------------------------------

_LK = {"bh76_reactions": [{"name": "r1", "reactants": ("A", "B"), "products": ("TS",),
                           "coeffs": (-1.0, -1.0, 1.0), "e_rxn_ref": 0.010}],
       "ip13_pairs": [{"name": "ip", "neutral": "A", "cation": "A+", "ip_ref": 0.200}]}


def _species():
    return {
        "A": {"E_final": -1.0, "eps": 0.010}, "B": {"E_final": -2.0, "eps": 0.020},
        "TS": {"E_final": -2.985, "eps": None}, "A+": {"E_final": -0.79, "eps": None},
        "cation_hb": {"E_final": -10.0, "eps": 0.009},
        "cation_pt": {"E_final": -10.02, "eps": 0.011},
        "zundel_+0.00": {"E_final": -5.0, "eps": 0.008},
        "zundel_+0.30": {"E_final": -4.9, "eps": 0.008},
    }


def test_summary_reaction_errors_and_ed_by_hand():
    pt_names = ["cation_hb", "cation_pt", "zundel_+0.00", "zundel_+0.30"]
    s = rpp.summarize_readout(_species(), _LK, pt_names, tracked_ha=-0.03)
    k = rpp.KCAL_PER_HA
    assert s["reactions"]["r1"]["err_kcal"] == pytest.approx((0.015 - 0.010) * k)
    assert s["reactions"]["ip"]["err_kcal"] == pytest.approx((0.21 - 0.200) * k)
    assert s["mae_train_kcal"] == pytest.approx((0.005 * k + 0.010 * k) / 2)
    assert s["pt"]["err_kcal"] == pytest.approx((-0.02 + 0.03) * k)
    assert s["pt"]["zundel_span_ha"] == pytest.approx(0.1)
    assert s["eps_mean_corpus"] == pytest.approx(0.015)
    assert s["eps_mean_pt"] == pytest.approx((0.009 + 0.011 + 0.008 + 0.008) / 4)
    from xcquinox.pipeline.eval_holdout import dfs_letter_ed
    assert s["ed_published"] == pytest.approx(
        dfs_letter_ed(s["wtmad2_corpus_kcal"], s["eps_mean_corpus"]))
    w, e = s["wtmad2_corpus_kcal"], s["eps_mean_corpus"]
    assert s["ed_own_axes"] == pytest.approx(2.0 / (1.0 / w + 1.0 / (1158.34 * e)))
    # WTMAD-2 by family (r1 a barrier, ip an ionization), by hand: subsets of
    # one reaction each; mean |ref| over all = (0.010 + 0.200)/2 Ha; the sum of
    # MAD_i/m_i = 0.005/0.010 + 0.010/0.200; total = (0.105/2) * 0.55 Ha
    assert w == pytest.approx(0.105 / 2 * 0.55 * k)
    assert w != pytest.approx(s["mae_train_kcal"])
    assert math.isnan(rpp._ed(0.0, 0.01, 1084.87))


def test_assemble_writes_tables_from_the_readouts_present(tmp_path):
    prong = tmp_path / "prong_A"
    pt_names = ["cation_hb", "cation_pt"]
    net = rpp.summarize_readout(_species(), _LK, pt_names, tracked_ha=-0.03)
    rpp.write_json(prong / "readout" / "arch1" / "readout.json",
                   {"network": net, "pbe": net, "scf": {"n_species": 8, "n_converged": 8},
                    "eps": {"A": 0.01}})
    led = rpp.Ledger(prong / "ledger.json")
    led.ensure(["arch1", "arch2"], failed={"arch2": "pretrain"})
    led.mark("arch1", rpp.VALIDATED)
    res = rpp.assemble(prong, clone_root=tmp_path / "clones", ledger=led)
    assert res["n_readouts"] == 1
    rows = {r["arch"]: r for r in res["summary"]}
    assert rows["arch1"]["state"] == "validated" and rows["arch2"]["state"] == "failed"
    # an architecture without a readout has empty cells, never NaN
    assert rows["arch2"]["mae_train_kcal"] == "" and rows["arch2"]["ed_published"] == ""
    assert "nan" not in (prong / "results" / "summary.csv").read_text().lower()
    assert (prong / "results" / "summary.csv").is_file()
    assert json.loads((prong / "results" / "status.json").read_text())["counts"] == {
        "validated": 1, "failed": 1}


def test_the_driver_reports_143_only_for_a_sigterm_shaped_failure():
    """The stage driver's exit status: a raising cell or a dead kernel after
    a SIGTERM is an interruption (143); the same without a SIGTERM, or any
    other error after one, is a failure (1)."""
    import importlib.util
    path = _NOTEBOOKS.parent / "tools" / "run_registry_pt_stage.py"
    spec = importlib.util.spec_from_file_location("run_registry_pt_stage", path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)

    class CellExecutionError(Exception):
        pass

    class DeadKernelError(Exception):
        pass

    assert driver.exit_code(CellExecutionError(), True) == 143
    assert driver.exit_code(DeadKernelError(), True) == 143
    assert driver.exit_code(CellExecutionError(), False) == 1
    assert driver.exit_code(AttributeError("cleanup_resources"), True) == 1
    assert driver.exit_code(ValueError(), False) == 1
