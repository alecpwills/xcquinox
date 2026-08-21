# Per-Architecture Workflow Matrix -- Implementation Plan

**Goal:** Drive the whole harness stage sequence -- `submit` (dry-run) -> `_datagen` -> `_pretrain` -> certificate -> `_preflight` -> `_train_task` (two cells) -> `_eval_one_spec` (two cells) -> `validate_run`, plus the architecture's spin-scaling oracles -- as plain Python at a tiny def2-svp / grid-level-1 identity for EVERY architecture in `xcquinox.alec.config.ARCHITECTURES`, and record a per-architecture result table that becomes the HISTORY baseline every later change is measured against.

**Architecture:** One new module, `xcquinox/alec/cluster/workflow_matrix.py`, owns the whole runner: it renders a one-architecture tiny YAML from a checked-in template, executes a fixed stage table through a `runner=subprocess.run` seam with one log file per stage, stops at the first non-zero exit, collects artefact paths and the certificate verdict, and runs the architecture's slice of the oracle tests. Nothing about the harness stages changes: the matrix is a caller, not a wrapper. The single harness change this plan makes is the mechanism that makes the held-out evaluation affordable -- an explicitly-named species slice (`XCQUINOX_HELDOUT_SPECIES_SLICE`), applied inside `_eval_one_spec._run_held_out_eval` (never inside the pool loader, which the TRAINING pool also goes through), marked on disk twice so a sliced channel cannot be read as a full-pool one, and refused by the figure loaders.

**Tech Stack:** Python 3.12, PyYAML, pytest 9 (`-m "not slow"` by default, see `setup.cfg`), `subprocess`, `concurrent.futures.ThreadPoolExecutor`; the harness modules under `xcquinox/alec/cluster/`; PySCF + JAX only inside the stage subprocesses, never in the runner.

**Spec:** `xcquinox/alec/SPEC_pretrain_fidelity_program.md` (this plan implements Section 3.4 and the HISTORY baseline of Section 5 step 4; Sections 3.1, 3.2, 3.3 and 3.5 are other plans)


> **Controller ruling (2026-08-21), supersedes the plan's resolution of ambiguity 1:** the
> certificate is made non-blocking for the matrix through the spec 3.3 configuration knob
> `fidelity.enforce: false` (valid only with a non-empty `fidelity.override_reason`; the
> certificate is still computed and written with its true verdict; `validate_run`, the merge
> and the figure loaders keep refusing a non-PASS run), NOT through an environment variable.
> The matrix template sets `fidelity: {enforce: false, override_reason: "workflow matrix:
> wiring check at a 50-step pretrain, never a campaign"}`; `FIDELITY_OVERRIDE_REASON` and the
> `XCQUINOX_FIDELITY_OVERRIDE_REASON` export in Tasks 4 and 6 are replaced by that template
> block, and the certificate stage keeps `allow_nonzero=False` (with `enforce: false` the
> stage exits zero and the verdict is read from the certificate file).

## Global Constraints

Every task's requirements implicitly include this section.

- Section 3.4 of the spec, copied verbatim, is the acceptance list: "Assertions per architecture: every stage exits zero, the expected artefacts exist, the certificate verdict is recorded, the in-sample `eval_df.csv` and the sliced held-out channel are written, and the oracle tests O1-O4 pass for the architecture. Results are recorded in HISTORY as the baseline matrix every later change is measured against."
- Section 3.4 of the spec, copied verbatim, fixes the identity: "at def2-svp / grid level 1 with the repository's cached subset ledger and CCSD references (`notebooks/checkpoints_step7/`), the certificate step at the production identity; the held-out evaluation is exercised on a six-species slice of the pool (the full pool is hours per cell and not narrowable from the configuration)."
- Section 3.4 of the spec, copied verbatim, fixes the population: "For every architecture in `ARCHITECTURES` (30 entries; the figure layer's `ARCH_ORDER` renders 25 of them -- the matrix exercises all 30, the figure layer is not one of its assertions)."
- Section 4 of the spec, copied verbatim: "Every code change: tests first (RED shown against the archived tree), oracle tests executed, `py_compile`, and TWO independent reviews per commit, each required to execute the oracles themselves (one on the physics, one on the workflow end to end)." Every task below is written test-first with the failing run shown; the two reviews are the controller's, not the implementer's.
- The spec says `ARCHITECTURES` has 30 entries. The registry is the authority, not this plan and not the spec: enumerate it at run time (`sorted(ARCHITECTURES)`), never hard-code a count, and record the count MEASURED on the day in the HISTORY entry. Measured on 2026-08-21: `sorted(ARCHITECTURES)` holds 31 names, one more than the spec's count, and the figure layer's `ARCH_ORDER` holds 25 of them, every one of which is registered; the six the figure layer does not render are `deep_cusp_attn`, `deep_dm_attn`, `medium`, `medium_attn`, `shallow`, `shallow_attn`. The matrix runs all 31. `ARCH_ORDER` is not consulted anywhere in this plan.
- The certificate's own identity (6-311++G(3df,2pd), grid level 3, density fitting, parent functional per rung) is chosen by the certificate, spec 3.3: the matrix passes it a run directory and an architecture index and nothing else. Do not add an identity knob for it here.
- `XCQUINOX_FIDELITY_OVERRIDE_REASON` exists for THIS matrix and for nothing else. It never appears in a campaign YAML, a campaign sbatch script or a shell profile: Section 1 of the spec makes the certificate a hard gate on every campaign stage, and an override left in the environment would silently open it.
- Comments and docstrings are ASCII only, in scientific voice. They state physics, measurements and rationale. They never mention the process by which the code was produced, never mention an assistant or a model, never say "we", "I", "now", "previously", "as requested", "TODO" or "FIXME". Reference literature the way the surrounding code does (author, journal, volume, page, year).
- Run `python -m py_compile <file>` on every Python file immediately after editing it. A task is not finished while any edited file fails to compile.
- Every test run redirects to a log file and the log is read with `Read`. Never pipe a test run through `tail`, `head`, `less`, `grep -m`, or any other truncating filter: the log must be complete. Create the log directory once with `mkdir -p /tmp/xcq-testlogs`.
- Implementers run no git commands: no `git add`, `git commit`, `git push`, `git checkout`, `git branch`, `git stash`, `git rebase`. Committing is the controller's job.
- Nothing this plan runs may write into a tracked repository directory. In particular `notebooks/checkpoints_step7/external_refs/` is tracked and `external_refs.precompute_all` writes a `_run_log_<UTC>.json` into its `cache_dir` on EVERY call (`xcquinox/alec/external_refs.py:1494-1506`) and `run_oep_cascade` may rewrite a species npz; the matrix therefore COPIES that directory into its work root and never points a config at the tracked path. The work root lives outside the repo (default `/tmp/xcq-workflow-matrix`), and `--work-root` must never be given a path inside the repository.
- `xcquinox/alec/HISTORY.md` gets an entry for this change (Task 10). It is the canonical development record for the paper.
- Every number quoted in a comment or a docstring must have been measured by the implementer on this machine. Do not copy a number from this plan into a comment without re-measuring it; the plan's timings are expectations, not measurements.

## Interfaces owned by the sibling plans (given, not re-implemented here)

Two sibling plans are landing concurrently. This plan CONSUMES their interfaces and must not re-implement them.

- **Spec 3.3, the certificate.** Module `xcquinox/alec/cluster/fidelity.py`, function `fidelity_certificate(cfg, run_dir, arch_name) -> dict`, output `<run_dir>/pretrain/<arch>/fidelity_certificate.json` with a top-level `verdict` of `"PASS"` / `"FAIL"`, CLI `python -m xcquinox.alec.cluster.fidelity <run_dir> <arch_idx>` where `arch_idx` indexes `sorted(set(cfg.sweep.arch))` exactly as `cluster/_pretrain.py:_distinct_archs` does.
- **Spec 3.1, the oracles.** Test module `xcquinox/alec/tests/test_spin_scaling_oracles.py`, whose architecture-carrying oracles are parametrized over `sorted(alec.ARCHITECTURES)` so that a pytest `-k` expression can select exactly one architecture's oracles.
- **Agreement this plan requires from spec 3.3, stated here because the matrix cannot run without it.** Spec 3.2 puts the certificate's acceptance INSIDE the pretrain stage ("failure exits non-zero and blocks the campaign") and spec 3.3 puts it in front of the train task ("the train task refuses to start without a PASS certificate"). The matrix's pretraining is 50 steps on two atoms at def2-svp, which cannot meet `tol_AE = 1.0 kcal/mol` by construction, while spec 3.4 requires every stage to exit zero and the verdict merely to be RECORDED. The sanctioned bypass is a non-empty `XCQUINOX_FIDELITY_OVERRIDE_REASON` in the process environment: the enforcement points log it verbatim and proceed; the certificate itself still computes and records its true verdict, which the override never touches. The matrix exports it for every stage, with the reason string in the module constant `FIDELITY_OVERRIDE_REASON` (Task 6). If spec 3.3 lands a different bypass name, one constant in `workflow_matrix.py` changes; if it lands no bypass at all, `run_arch` records the refusing stage's non-zero rc and stops there, which the matrix report shows as a `FAIL @pretrain` or `FAIL @train[0]` row rather than a silent pass.

## File Structure

| File | Responsibility after this plan |
|---|---|
| `xcquinox/alec/full_benchmark_pools.py` | Adds the two pure primitives of the held-out species slice: `HELDOUT_SPECIES_SLICE_ENV`, `resolve_species_slice(env=None)`, `slice_held_out_pools(mol_specs, reactions, species)`. `load_full_held_out_pools` itself is UNCHANGED: the training pool (`cluster/inputs.py:232`, `cluster/spec_builder.py:355`, `training_points.py:486`) goes through it, and a slice applied there would silently shrink training. |
| `xcquinox/alec/cluster/_eval_one_spec.py` | Applies the slice inside `_run_held_out_eval` only; writes `sliced_eval.json` BEFORE the evaluation and a `species_slice` entry in `eval_metadata.json` after it. |
| `notebooks/analysis/make_ablation_arch_figure.py` | `assert_channel_not_sliced(spec_dir, eval_subdir)`; both held-out ingest loaders call it per spec directory so a sliced channel can never enter a figure. |
| `xcquinox/alec/cluster/examples/workflow_matrix_template.yaml` | The checked-in one-architecture tiny grid config (package data; `pyproject.toml` already ships `examples/*.yaml`). |
| `xcquinox/alec/cluster/workflow_matrix.py` | The runner: `repo_root_path`, `template_path`, `stage_cached_inputs`, `write_matrix_yaml`, `oracle_selector`, `stage_plan`, `run_arch`, `run_matrix`, `write_matrix_report`, `main`. |
| `xcquinox/alec/tests/test_cluster_workflow_matrix.py` | Seam-mocked tests of the whole module with a fake runner, plus the one real single-architecture smoke marked `@pytest.mark.slow`. |
| `xcquinox/alec/tests/test_full_benchmark_pools.py` | Tests of the slice primitives. |
| `xcquinox/alec/tests/test_cluster_eval_worker.py` | Tests of the sliced held-out channel and its two disk marks. |
| `notebooks/analysis/test_make_ablation_arch_figure.py` | Test that the figure loaders refuse a sliced channel. |
| `xcquinox/alec/HISTORY.md` | The baseline matrix table. |

---

## Task 1: The held-out species slice, as two pure functions

**Files:**
- Modify: `xcquinox/alec/full_benchmark_pools.py:543` (append after `load_full_held_out_pools`, which ends the file)
- Test: `xcquinox/alec/tests/test_full_benchmark_pools.py` (append)

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces:
  - `full_benchmark_pools.HELDOUT_SPECIES_SLICE_ENV: str = "XCQUINOX_HELDOUT_SPECIES_SLICE"`
  - `full_benchmark_pools.resolve_species_slice(env: dict | None = None) -> tuple[str, ...] | None`
  - `full_benchmark_pools.slice_held_out_pools(mol_specs: dict, reactions: list, species) -> tuple[dict, list]`

- [ ] **Step 1: Write the failing tests**

Append to `xcquinox/alec/tests/test_full_benchmark_pools.py`:

```python
# ---------------------------------------------------------------------------
# Held-out species slice: an explicitly named handful of the pool, for
# workflow verification only (SPEC_pretrain_fidelity_program.md 3.4).
# ---------------------------------------------------------------------------

def test_resolve_species_slice_is_none_without_the_variable():
    from xcquinox.alec.full_benchmark_pools import resolve_species_slice
    assert resolve_species_slice({}) is None


def test_resolve_species_slice_is_none_for_an_empty_variable():
    from xcquinox.alec.full_benchmark_pools import (
        HELDOUT_SPECIES_SLICE_ENV, resolve_species_slice)
    assert resolve_species_slice({HELDOUT_SPECIES_SLICE_ENV: ""}) is None
    assert resolve_species_slice({HELDOUT_SPECIES_SLICE_ENV: "   "}) is None


def test_resolve_species_slice_parses_and_strips():
    from xcquinox.alec.full_benchmark_pools import (
        HELDOUT_SPECIES_SLICE_ENV, resolve_species_slice)
    got = resolve_species_slice(
        {HELDOUT_SPECIES_SLICE_ENV: " h , h2,o ,oh,n2o,n2ohts "})
    assert got == ("h", "h2", "o", "oh", "n2o", "n2ohts")


def test_resolve_species_slice_refuses_a_repeated_name():
    from xcquinox.alec.full_benchmark_pools import (
        HELDOUT_SPECIES_SLICE_ENV, resolve_species_slice)
    with pytest.raises(ValueError, match="repeats"):
        resolve_species_slice({HELDOUT_SPECIES_SLICE_ENV: "h,h2,h"})


def test_resolve_species_slice_reads_the_process_environment(monkeypatch):
    from xcquinox.alec.full_benchmark_pools import (
        HELDOUT_SPECIES_SLICE_ENV, resolve_species_slice)
    monkeypatch.setenv(HELDOUT_SPECIES_SLICE_ENV, "h,h2")
    assert resolve_species_slice() == ("h", "h2")


def test_slice_held_out_pools_keeps_only_closed_reactions():
    from xcquinox.alec.full_benchmark_pools import slice_held_out_pools
    mols = {"a": 1, "b": 2, "c": 3}
    rxns = [
        {"name": "closed", "reactants": ["a"], "products": ["b"]},
        {"name": "open", "reactants": ["a"], "products": ["c"]},
    ]
    kept_mols, kept_rxns = slice_held_out_pools(mols, rxns, ("a", "b"))
    assert kept_mols == {"a": 1, "b": 2}
    assert [r["name"] for r in kept_rxns] == ["closed"]


def test_slice_held_out_pools_refuses_a_species_absent_from_the_pool():
    from xcquinox.alec.full_benchmark_pools import slice_held_out_pools
    with pytest.raises(ValueError, match="nosuchspecies"):
        slice_held_out_pools({"a": 1}, [], ("a", "nosuchspecies"))


def test_slice_held_out_pools_preserves_the_requested_order():
    from xcquinox.alec.full_benchmark_pools import slice_held_out_pools
    kept, _ = slice_held_out_pools({"a": 1, "b": 2, "c": 3}, [], ("c", "a"))
    assert list(kept) == ["c", "a"]


def test_the_matrix_six_species_slice_spans_both_pools():
    """The workflow matrix's slice: six species of the real pool that close
    three reactions, one BH76 barrier and two W4-11 atomizations, over both
    spin types (RKS h2/n2o, UKS h/o/oh/n2ohts). A slice of six molecules with
    no atoms closes NO reaction and would leave the reaction math untested."""
    from xcquinox.alec.full_benchmark_pools import (
        load_full_held_out_pools, slice_held_out_pools)
    mols, rxns = load_full_held_out_pools(basis="def2-svp", grid_level=1)
    names = ("h", "h2", "o", "oh", "n2o", "n2ohts")
    kept_mols, kept_rxns = slice_held_out_pools(mols, rxns, names)
    assert len(kept_mols) == 6
    assert sorted(r["name"] for r in kept_rxns) == [
        "bh76_h_n2o_to_n2ohts", "w411_h2_atomization", "w411_oh_atomization"]
    assert {r["source_pool"] for r in kept_rxns} == {"bh76", "w411"}
```

- [ ] **Step 2: Run the tests and confirm they fail**

```bash
mkdir -p /tmp/xcq-testlogs
python -m pytest xcquinox/alec/tests/test_full_benchmark_pools.py -v > /tmp/xcq-testlogs/task01_red.log 2>&1; echo "exit=$?"
```
Expected: the new tests fail with `ImportError: cannot import name 'resolve_species_slice' from 'xcquinox.alec.full_benchmark_pools'`. Read the log with `Read`.

- [ ] **Step 3: Append the two primitives**

Append to `xcquinox/alec/full_benchmark_pools.py` (the file currently ends at line 543 with the `return` of `load_full_held_out_pools`):

```python


# ---------------------------------------------------------------------------
# Held-out species slice
# ---------------------------------------------------------------------------

#: Environment variable naming a comma-separated species slice of the held-out
#: pool. The slice exists for workflow verification
#: (SPEC_pretrain_fidelity_program.md 3.4): the full pool is 216 reactions over
#: 214 species and hours of SCF per grid cell, which is not narrowable from the
#: grid config. It is NEVER applied by default and NEVER applied to the pool
#: loaders themselves -- the training-point pool and the spec builder read the
#: same loaders (cluster/inputs.py, cluster/spec_builder.py,
#: training_points.py), and a slice reaching them would silently shrink
#: training. Only the held-out evaluation channel honours it
#: (cluster/_eval_one_spec._run_held_out_eval), and a channel evaluated under a
#: slice is marked on disk so it cannot be read as a full-pool channel.
HELDOUT_SPECIES_SLICE_ENV = "XCQUINOX_HELDOUT_SPECIES_SLICE"


def resolve_species_slice(env: Dict[str, str] | None = None
                          ) -> Tuple[str, ...] | None:
    """The species slice named by ``env``, or ``None`` for the full pool.

    ``env`` defaults to ``os.environ``. An absent or blank variable is the full
    pool: a slice applies only when it is asked for by name.
    """
    source = os.environ if env is None else env
    raw = (source.get(HELDOUT_SPECIES_SLICE_ENV) or "").strip()
    if not raw:
        return None
    names = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not names:
        raise ValueError(
            f"{HELDOUT_SPECIES_SLICE_ENV}={raw!r} names no species; unset the "
            "variable to evaluate the full held-out pool."
        )
    if len(set(names)) != len(names):
        raise ValueError(
            f"{HELDOUT_SPECIES_SLICE_ENV}={raw!r} repeats a species name: "
            f"{names}"
        )
    return names


def slice_held_out_pools(
    mol_specs: Dict[str, MoleculeSpec],
    reactions: Sequence[Dict[str, Any]],
    species: Sequence[str],
) -> Tuple[Dict[str, MoleculeSpec], List[Dict[str, Any]]]:
    """Restrict a held-out pool to ``species`` and the reactions closed under it.

    A reaction survives only when every reactant and product is inside the
    slice; a reaction with a missing leg has no defined energy, so keeping it
    would put an undefined term in the pool MAE. The returned species dict
    preserves the order of ``species`` so the slice reads the same way in the
    log line and in the provenance stamp.
    """
    wanted = tuple(species)
    missing = [n for n in wanted if n not in mol_specs]
    if missing:
        raise ValueError(
            f"held-out species slice names {missing}, absent from the pool of "
            f"{len(mol_specs)} species. Pool names are lower case Hill-like "
            "strings, e.g. 'h', 'h2', 'oh', 'n2o', 'n2ohts'."
        )
    kept_names = set(wanted)
    kept_specs = {n: mol_specs[n] for n in wanted}
    kept_rxns = [
        r for r in reactions
        if set(r["reactants"]) | set(r["products"]) <= kept_names
    ]
    return kept_specs, kept_rxns
```

- [ ] **Step 4: Compile and run the tests**

```bash
cd /home/awills/Documents/Research/xcquinox && python -m py_compile xcquinox/alec/full_benchmark_pools.py && echo compiled
cd /home/awills/Documents/Research/xcquinox && python -m pytest xcquinox/alec/tests/test_full_benchmark_pools.py -v > /tmp/xcq-testlogs/task01_green.log 2>&1; echo "exit=$?"
```
Expected: PASS, every pre-existing test in the file still green.

**Covering test command:** `python -m pytest xcquinox/alec/tests/test_full_benchmark_pools.py -v > /tmp/xcq-testlogs/task01_green.log 2>&1`

---

## Task 2: Apply the slice in the held-out eval worker, marked twice on disk

**Files:**
- Modify: `xcquinox/alec/cluster/_eval_one_spec.py:248-250` (new helper inserted before `_test_slice_reactions`), `:350-352` (the pool load), `:398-410` (the provenance stamp)
- Test: `xcquinox/alec/tests/test_cluster_eval_worker.py` (append)

**Interfaces:**
- Consumes: `full_benchmark_pools.HELDOUT_SPECIES_SLICE_ENV`, `resolve_species_slice`, `slice_held_out_pools` (Task 1).
- Produces:
  - `_eval_one_spec._apply_species_slice(idx, full_specs, full_rxns, holdout_dir) -> tuple[dict, list, tuple[str, ...] | None]`
  - `<checkpoint_dir>/<channel>/sliced_eval.json` -- written BEFORE the evaluation, keys `species_slice`, `n_species`, `n_reactions`, `env_var`.
  - `<checkpoint_dir>/<channel>/eval_metadata.json` gains `species_slice` (list or `None`), `n_species`, `n_reactions`.

- [ ] **Step 1: Write the failing tests**

Append to `xcquinox/alec/tests/test_cluster_eval_worker.py`:

```python
# ---------------------------------------------------------------------------
# Held-out species slice (SPEC_pretrain_fidelity_program.md 3.4)
# ---------------------------------------------------------------------------

def _slice_fixture(monkeypatch, run_dir):
    """Wire the held-out seams so _run_held_out_eval runs no SCF.

    The pool stub is the six-species matrix slice plus one species outside it,
    so a slice that is applied is distinguishable from one that is not.
    """
    from types import SimpleNamespace
    import xcquinox.alec.eval_holdout as eh
    import xcquinox.alec.full_benchmark_pools as fbp
    spec = _full_mode_spec()
    ckpt_dir = _write_model(run_dir, 0)
    cfg = SimpleNamespace(cluster=SimpleNamespace(eval_workers=1),
                          held_out_strict=False)
    pool = {n: f"spec_{n}" for n in
            ("h", "h2", "o", "oh", "n2o", "n2ohts", "c2h6")}
    rxns = [
        {"name": "w411_h2_atomization", "reactants": ["h2"], "products": ["h"]},
        {"name": "w411_c2h6_atomization", "reactants": ["c2h6"],
         "products": ["h", "c"]},
    ]
    seen = {}
    monkeypatch.setattr(eh, "load_trained_model", lambda ts, mp: "MODEL")
    monkeypatch.setattr(fbp, "load_full_held_out_pools",
                        lambda basis=None, grid_level=None: (dict(pool),
                                                             list(rxns)))
    monkeypatch.setattr(ev, "_held_out_basis_grid", lambda cfg: ("def2-svp", 1))

    def _capture(**kw):
        seen["mol_specs"] = dict(kw["mol_specs"])
        seen["reactions"] = list(kw["reactions"])
        return {"n_reactions": len(kw["reactions"]),
                "n_species": len(kw["mol_specs"]),
                "n_dropped_nan": 0, "n_dropped_overlap": 0}

    monkeypatch.setattr(eh, "run_full_holdout_eval", _capture)
    return spec, cfg, ckpt_dir, seen


def test_held_out_eval_is_the_full_pool_without_the_slice_variable(
        run_dir, monkeypatch):
    """No variable, no slice: the channel carries every pool species and no
    slice mark. The full pool must stay the default on the cluster."""
    from xcquinox.alec.full_benchmark_pools import HELDOUT_SPECIES_SLICE_ENV
    monkeypatch.delenv(HELDOUT_SPECIES_SLICE_ENV, raising=False)
    spec, cfg, ckpt_dir, seen = _slice_fixture(monkeypatch, run_dir)
    ev._run_held_out_eval(run_dir, 0, cfg, ckpt_dir,
                          os.path.join(ckpt_dir, "model.eqx"), spec)
    assert len(seen["mol_specs"]) == 7
    assert len(seen["reactions"]) == 2
    chan = os.path.join(ckpt_dir, "eval_holdout")
    assert not os.path.exists(os.path.join(chan, "sliced_eval.json"))
    with open(os.path.join(chan, "eval_metadata.json")) as f:
        assert json.load(f)["species_slice"] is None


def test_held_out_eval_applies_the_named_species_slice(run_dir, monkeypatch):
    from xcquinox.alec.full_benchmark_pools import HELDOUT_SPECIES_SLICE_ENV
    monkeypatch.setenv(HELDOUT_SPECIES_SLICE_ENV, "h,h2,o,oh,n2o,n2ohts")
    spec, cfg, ckpt_dir, seen = _slice_fixture(monkeypatch, run_dir)
    ev._run_held_out_eval(run_dir, 0, cfg, ckpt_dir,
                          os.path.join(ckpt_dir, "model.eqx"), spec)
    assert sorted(seen["mol_specs"]) == ["h", "h2", "n2o", "n2ohts", "o", "oh"]
    # c2h6's atomization leaves the slice, so its reaction is not scored.
    assert [r["name"] for r in seen["reactions"]] == ["w411_h2_atomization"]


def test_sliced_channel_is_marked_before_the_evaluation_runs(run_dir,
                                                             monkeypatch):
    """The mark must survive an evaluation that dies: it is written before the
    energies, so an interrupted sliced channel is still unmistakable."""
    from xcquinox.alec.full_benchmark_pools import HELDOUT_SPECIES_SLICE_ENV
    import xcquinox.alec.eval_holdout as eh
    monkeypatch.setenv(HELDOUT_SPECIES_SLICE_ENV, "h,h2")
    spec, cfg, ckpt_dir, _seen = _slice_fixture(monkeypatch, run_dir)

    def _boom(**kw):
        raise RuntimeError("synthetic eval failure")

    monkeypatch.setattr(eh, "run_full_holdout_eval", _boom)
    ev._run_held_out_eval(run_dir, 0, cfg, ckpt_dir,
                          os.path.join(ckpt_dir, "model.eqx"), spec)
    chan = os.path.join(ckpt_dir, "eval_holdout")
    assert os.path.isfile(os.path.join(chan, "failure.json"))
    assert not os.path.exists(os.path.join(chan, "eval_metadata.json"))
    with open(os.path.join(chan, "sliced_eval.json")) as f:
        mark = json.load(f)
    assert mark["species_slice"] == ["h", "h2"]
    assert mark["env_var"] == HELDOUT_SPECIES_SLICE_ENV


def test_sliced_channel_stamp_records_the_slice_and_the_counts(run_dir,
                                                               monkeypatch):
    from xcquinox.alec.full_benchmark_pools import HELDOUT_SPECIES_SLICE_ENV
    monkeypatch.setenv(HELDOUT_SPECIES_SLICE_ENV, "h,h2,o,oh,n2o,n2ohts")
    spec, cfg, ckpt_dir, _seen = _slice_fixture(monkeypatch, run_dir)
    ev._run_held_out_eval(run_dir, 0, cfg, ckpt_dir,
                          os.path.join(ckpt_dir, "model.eqx"), spec)
    with open(os.path.join(ckpt_dir, "eval_holdout",
                           "eval_metadata.json")) as f:
        stamp = json.load(f)
    assert stamp["species_slice"] == ["h", "h2", "o", "oh", "n2o", "n2ohts"]
    assert stamp["n_species"] == 6
    assert stamp["n_reactions"] == 1
    assert stamp["channel"] == "eval_holdout"


def test_unknown_sliced_species_fails_the_channel_not_the_task(run_dir,
                                                               monkeypatch):
    """A misspelt slice must not silently evaluate a different set: the channel
    records failure.json (held-out failure is non-fatal by contract)."""
    from xcquinox.alec.full_benchmark_pools import HELDOUT_SPECIES_SLICE_ENV
    monkeypatch.setenv(HELDOUT_SPECIES_SLICE_ENV, "h,nosuchspecies")
    spec, cfg, ckpt_dir, _seen = _slice_fixture(monkeypatch, run_dir)
    ev._run_held_out_eval(run_dir, 0, cfg, ckpt_dir,
                          os.path.join(ckpt_dir, "model.eqx"), spec)
    with open(os.path.join(ckpt_dir, "eval_holdout", "failure.json")) as f:
        payload = json.load(f)
    assert payload["exception_type"] == "ValueError"
    assert "nosuchspecies" in payload["exception_message"]
```

- [ ] **Step 2: Run the tests and confirm they fail**

```bash
cd /home/awills/Documents/Research/xcquinox && python -m pytest xcquinox/alec/tests/test_cluster_eval_worker.py -v > /tmp/xcq-testlogs/task02_red.log 2>&1; echo "exit=$?"
```
Expected: the four slice tests fail -- `KeyError: 'species_slice'` on the stamp, and `assert not os.path.exists(...sliced_eval.json)` passing only in the no-variable case while the sliced cases find no mark. Read the log with `Read`.

- [ ] **Step 3: Add the slice helper**

Insert into `xcquinox/alec/cluster/_eval_one_spec.py` immediately after the `# main` banner comment block (currently lines 248-250) and BEFORE `def _test_slice_reactions`:

```python
def _apply_species_slice(idx, full_specs, full_rxns, holdout_dir):
    """Restrict the held-out pool to the environment's species slice.

    Returns ``(mol_specs, reactions, slice_names)``; ``slice_names`` is None
    when no slice is named, in which case the pool is returned untouched and
    the channel carries no mark -- the full 216-reaction BH76 + W4-11 pool
    stays the default.

    A sliced channel is marked TWICE. ``sliced_eval.json`` is written here,
    before any energy is computed, so an interrupted or failed sliced
    evaluation is still unmistakable; ``eval_metadata.json`` carries the same
    slice after the evaluation. The figure layer refuses a channel bearing
    either mark, because a slice covers a handful of species chosen for a
    workflow test and its MAE is not the pool MAE the architectures are
    compared on.
    """
    from xcquinox.alec.full_benchmark_pools import (
        HELDOUT_SPECIES_SLICE_ENV, resolve_species_slice,
        slice_held_out_pools)
    names = resolve_species_slice()
    if names is None:
        return full_specs, full_rxns, None
    sliced_specs, sliced_rxns = slice_held_out_pools(
        full_specs, full_rxns, names)
    holdout_dir.mkdir(parents=True, exist_ok=True)
    with open(holdout_dir / "sliced_eval.json", "w") as f:
        json.dump({
            "species_slice": list(names),
            "n_species": len(sliced_specs),
            "n_reactions": len(sliced_rxns),
            "env_var": HELDOUT_SPECIES_SLICE_ENV,
        }, f, indent=2, sort_keys=True)
        f.write("\n")
    _log(idx, f"held-out eval SLICED by {HELDOUT_SPECIES_SLICE_ENV} to "
              f"{len(sliced_specs)} species / {len(sliced_rxns)} reactions "
              f"({', '.join(names)}) -- this channel is NOT the full pool")
    return sliced_specs, sliced_rxns, names
```

- [ ] **Step 4: Call it and stamp the slice**

In `_run_held_out_eval`, replace the pool load at lines 350-352:

```python
        full_specs, full_rxns = load_full_held_out_pools(
            basis=_hb, grid_level=_hg,
        )
```

with:

```python
        full_specs, full_rxns = load_full_held_out_pools(
            basis=_hb, grid_level=_hg,
        )
        full_specs, full_rxns, _slice_names = _apply_species_slice(
            idx, full_specs, full_rxns, holdout_dir)
```

and replace the stamp payload at lines 401-408:

```python
            with open(holdout_dir / "eval_metadata.json", "w") as f:
                json.dump({
                    "channel": holdout_subdir,
                    "model": model_name,
                    "coldstart": bool(coldstart),
                    "solver_config": (_sc.describe()
                                      if _sc is not None else None),
                }, f, indent=2, sort_keys=True)
```

with:

```python
            with open(holdout_dir / "eval_metadata.json", "w") as f:
                json.dump({
                    "channel": holdout_subdir,
                    "model": model_name,
                    "coldstart": bool(coldstart),
                    "solver_config": (_sc.describe()
                                      if _sc is not None else None),
                    # None for the full pool. A list names the species the
                    # channel actually covers, so a sliced channel cannot be
                    # read as a full-pool one.
                    "species_slice": (list(_slice_names)
                                      if _slice_names else None),
                    "n_species": len(full_specs),
                    "n_reactions": len(full_rxns),
                }, f, indent=2, sort_keys=True)
```

- [ ] **Step 5: Compile and run the tests**

```bash
cd /home/awills/Documents/Research/xcquinox && python -m py_compile xcquinox/alec/cluster/_eval_one_spec.py && echo compiled
cd /home/awills/Documents/Research/xcquinox && python -m pytest xcquinox/alec/tests/test_cluster_eval_worker.py xcquinox/alec/tests/test_eval_holdout.py -v > /tmp/xcq-testlogs/task02_green.log 2>&1; echo "exit=$?"
```
Expected: PASS, including the pre-existing `test_run_held_out_eval_writes_provenance_stamp` (which asserts specific stamp keys and must not have moved).

**Covering test command:** `python -m pytest xcquinox/alec/tests/test_cluster_eval_worker.py xcquinox/alec/tests/test_eval_holdout.py -v > /tmp/xcq-testlogs/task02_green.log 2>&1`

---

## Task 3: The figure loaders refuse a sliced held-out channel

**Files:**
- Modify: `notebooks/analysis/make_ablation_arch_figure.py:246-247` (new function inserted before `collect_holdout_reaction_rows`), `:282` (loop guard), `:3518` (loop guard)
- Test: `notebooks/analysis/test_make_ablation_arch_figure.py` (append)

**Interfaces:**
- Consumes: the `sliced_eval.json` / `eval_metadata.json["species_slice"]` marks (Task 2).
- Produces: `make_ablation_arch_figure.assert_channel_not_sliced(spec_dir: Path, eval_subdir: str) -> None`, raising `RuntimeError`.

- [ ] **Step 1: Write the failing tests**

Append to `notebooks/analysis/test_make_ablation_arch_figure.py`:

```python
# ---------------------------------------------------------------------------
# A sliced held-out channel is not a pool channel and never enters a figure
# ---------------------------------------------------------------------------

def test_assert_channel_not_sliced_passes_an_unmarked_channel(tmp_path):
    spec_dir = tmp_path / "spec_0000"
    (spec_dir / "eval_holdout").mkdir(parents=True)
    fig.assert_channel_not_sliced(spec_dir, "eval_holdout")


def test_assert_channel_not_sliced_passes_a_full_pool_stamp(tmp_path):
    spec_dir = tmp_path / "spec_0000"
    chan = spec_dir / "eval_holdout"
    chan.mkdir(parents=True)
    (chan / "eval_metadata.json").write_text(json.dumps(
        {"channel": "eval_holdout", "species_slice": None}))
    fig.assert_channel_not_sliced(spec_dir, "eval_holdout")


def test_assert_channel_not_sliced_refuses_the_pre_eval_marker(tmp_path):
    spec_dir = tmp_path / "spec_0000"
    chan = spec_dir / "eval_holdout"
    chan.mkdir(parents=True)
    (chan / "sliced_eval.json").write_text(json.dumps(
        {"species_slice": ["h", "h2"], "n_species": 2, "n_reactions": 1}))
    with pytest.raises(RuntimeError, match="sliced_eval.json"):
        fig.assert_channel_not_sliced(spec_dir, "eval_holdout")


def test_assert_channel_not_sliced_refuses_a_sliced_stamp(tmp_path):
    spec_dir = tmp_path / "spec_0000"
    chan = spec_dir / "eval_holdout"
    chan.mkdir(parents=True)
    (chan / "eval_metadata.json").write_text(json.dumps(
        {"channel": "eval_holdout", "species_slice": ["h", "h2"]}))
    with pytest.raises(RuntimeError, match="species_slice"):
        fig.assert_channel_not_sliced(spec_dir, "eval_holdout")


def test_collect_holdout_reaction_rows_refuses_a_sliced_run(tmp_path):
    run = _make_run_dir(tmp_path)
    marked = run / "checkpoints" / "spec_0000" / "eval_holdout"
    (marked / "sliced_eval.json").write_text(json.dumps(
        {"species_slice": ["h", "h2"], "n_species": 2, "n_reactions": 1}))
    with pytest.raises(RuntimeError, match="spec_0000"):
        fig.collect_holdout_reaction_rows(run)


def test_collect_holdout_density_rows_refuses_a_sliced_run(tmp_path):
    run = _make_run_dir(tmp_path)
    marked = run / "checkpoints" / "spec_0000" / "eval_holdout"
    (marked / "per_molecule.json").write_text(json.dumps(
        [{"molecule": "h2", "density_rmse": 1e-3, "density_rmse_pbe": 2e-3}]))
    (marked / "eval_metadata.json").write_text(json.dumps(
        {"channel": "eval_holdout", "species_slice": ["h", "h2"]}))
    with pytest.raises(RuntimeError, match="species_slice"):
        fig.collect_holdout_density_rows(run)
```

- [ ] **Step 2: Run the tests and confirm they fail**

```bash
python -m pytest notebooks/analysis/test_make_ablation_arch_figure.py -v > /tmp/xcq-testlogs/task03_red.log 2>&1; echo "exit=$?"
```
Expected: `AttributeError: module 'make_ablation_arch_figure' has no attribute 'assert_channel_not_sliced'` on the first four, and `Failed: DID NOT RAISE` on the last two. Read the log with `Read`.

- [ ] **Step 3: Add the guard**

Insert into `notebooks/analysis/make_ablation_arch_figure.py` immediately before `def collect_holdout_reaction_rows` (line 248):

```python
def assert_channel_not_sliced(spec_dir: Path, eval_subdir: str) -> None:
    """Refuse a held-out channel evaluated on a species slice.

    A slice covers a handful of species named for a workflow test
    (``XCQUINOX_HELDOUT_SPECIES_SLICE``, SPEC_pretrain_fidelity_program.md
    3.4), not the 216-reaction BH76 + W4-11 pool the architectures are compared
    on; its MAE is a different quantity and averaging one into a figure would
    redefine the metric silently. ``cluster/_eval_one_spec`` marks a sliced
    channel twice -- ``sliced_eval.json`` written before the energies and a
    ``species_slice`` entry in ``eval_metadata.json`` written after them -- and
    either mark is fatal here, so an interrupted sliced evaluation is caught as
    surely as a complete one. An unparseable stamp is not a slice signal and is
    left to the readers below.
    """
    channel = spec_dir / eval_subdir
    marker = channel / "sliced_eval.json"
    if marker.is_file():
        raise RuntimeError(
            f"{marker} marks {spec_dir.name}/{eval_subdir} as a SLICED "
            "held-out channel (a workflow-verification slice of the pool). "
            "The figure layer reports full-pool channels only; drop the run or "
            "re-evaluate the channel without "
            "XCQUINOX_HELDOUT_SPECIES_SLICE."
        )
    stamp = channel / "eval_metadata.json"
    if not stamp.is_file():
        return
    try:
        with stamp.open() as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError):
        return
    if payload.get("species_slice"):
        raise RuntimeError(
            f"{stamp} records species_slice={payload['species_slice']!r} for "
            f"{spec_dir.name}/{eval_subdir}: a SLICED held-out channel, not "
            "the full BH76 + W4-11 pool. The figure layer reports full-pool "
            "channels only."
        )
```

- [ ] **Step 4: Call the guard from both ingest loaders**

In `collect_holdout_reaction_rows`, make the guard the first statement of the loop body at line 282:

```python
    for idx, spec_dir in ccp._spec_dirs(run_dir):
        assert_channel_not_sliced(spec_dir, eval_subdir)
        got = _reconstruct_spec_rows(run_dir, idx, spec_dir, cells,
                                     eval_subdir, recon_stats)
```

In `collect_holdout_density_rows`, do the same at line 3518:

```python
    for idx, spec_dir in ccp._spec_dirs(run_dir):
        assert_channel_not_sliced(spec_dir, eval_subdir)
        pm_path = spec_dir / eval_subdir / "per_molecule.json"
```

- [ ] **Step 5: Compile and run the tests**

```bash
cd /home/awills/Documents/Research/xcquinox && python -m py_compile notebooks/analysis/make_ablation_arch_figure.py && echo compiled
cd /home/awills/Documents/Research/xcquinox && python -m pytest notebooks/analysis/test_make_ablation_arch_figure.py -v > /tmp/xcq-testlogs/task03_green.log 2>&1; echo "exit=$?"
```
Expected: PASS, every pre-existing figure test still green (the guard is a no-op on an unmarked channel, which is what the fixture writes).

**Covering test command:** `python -m pytest notebooks/analysis/test_make_ablation_arch_figure.py -v > /tmp/xcq-testlogs/task03_green.log 2>&1`

---

## Task 4: The tiny one-architecture grid config -- template plus renderer

**Files:**
- Create: `xcquinox/alec/cluster/examples/workflow_matrix_template.yaml`
- Create: `xcquinox/alec/cluster/workflow_matrix.py`
- Test: `xcquinox/alec/tests/test_cluster_workflow_matrix.py` (create)

**Interfaces:**
- Consumes: `grid_config.load_grid_config`, `grid_config.expand_grid`, `grid_config.validate_grid_semantics`, `domain.get_domain_profile`.
- Produces:
  - `workflow_matrix.repo_root_path() -> Path`
  - `workflow_matrix.template_path() -> Path`
  - `workflow_matrix.CACHED_LEDGER_RELPATH: str`, `CACHED_REFS_RELPATH: str`
  - `workflow_matrix.stage_cached_inputs(dest_root, *, repo_root) -> dict` with keys `external_refs_dir`, `subset_ledger_path`
  - `workflow_matrix.write_matrix_yaml(arch, out_dir, *, repo_root, external_refs_dir=None, pretrain_data_dir=None) -> Path`

- [ ] **Step 1: Write the failing tests**

Create `xcquinox/alec/tests/test_cluster_workflow_matrix.py`:

```python
"""Tests for the per-architecture workflow matrix
(``xcquinox.alec.cluster.workflow_matrix``).

The matrix drives the harness stage sequence -- submit (dry-run), datagen,
pretrain, certificate, preflight, train, eval, validate_run -- as plain Python
at a tiny def2-svp / grid-level-1 identity for every registered architecture
(SPEC_pretrain_fidelity_program.md 3.4). These tests exercise the runner with a
FAKE runner in place of ``subprocess.run``: no stage subprocess is started, so
the whole file runs in seconds. The single real end-to-end pass is the
``slow``-marked smoke at the bottom.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from xcquinox.alec.cluster import workflow_matrix as wm
from xcquinox.alec.config import ARCHITECTURES


# ---------------------------------------------------------------------------
# Template + renderer
# ---------------------------------------------------------------------------

def test_template_exists_and_is_package_data():
    path = wm.template_path()
    assert path.is_file(), path
    assert not (path.parent / "__init__.py").exists(), (
        "cluster/examples/ ships as package DATA, not as a subpackage")


def test_template_carries_no_address_and_no_account():
    """The template is rendered in dry-run only and never submitted, so it
    carries no mail address and no allocation; a shipped example with a real
    address would mail a person from anybody's copy."""
    text = wm.template_path().read_text()
    assert "@" not in text.split("cluster:", 1)[1]
    assert '\n  mail_user: ""\n' in text
    assert '\n  account: ""\n' in text


def test_template_is_the_tiny_identity_the_spec_fixes():
    import yaml
    raw = yaml.safe_load(wm.template_path().read_text())
    assert raw["inputs"]["basis"] == "def2-svp"
    assert raw["inputs"]["grid_level"] == 1
    assert raw["sweep"]["solver"] == ["oneshot"]
    assert raw["sweep"]["subset_size"] == [1, 2]
    assert raw["hyperparams"]["n_steps"] == 3
    assert raw["hyperparams"]["validate_every"] == 0
    assert raw["hyperparams"]["checkpoint_every"] == 0
    assert raw["pretrain"]["n_steps"] == 50
    assert raw["pretrain"]["atoms"] == {"H": 1, "O": 2}
    assert raw["cluster"]["eval_workers"] == 1
    assert raw["cluster"]["device"] == "cpu"
    assert "benchmark_refs_dir" not in raw["inputs"]
    assert "val_refs_dir" not in raw["inputs"]


def test_stage_cached_inputs_copies_the_refs_out_of_the_repository(tmp_path):
    staged = wm.stage_cached_inputs(tmp_path, repo_root=wm.repo_root_path())
    refs = Path(staged["external_refs_dir"])
    assert refs.is_dir()
    assert not refs.is_symlink()
    assert (refs / "H2O.npz").is_file()
    assert (refs / "_intermediates" / "HO_g1_scf.npz").is_file()
    assert str(refs).startswith(str(tmp_path))
    # The run log precompute_all writes on every call must land here, never in
    # the tracked tree.
    assert not any(p.name.startswith("_run_log_") for p in refs.iterdir())
    assert Path(staged["subset_ledger_path"]).is_file()


def test_stage_cached_inputs_is_idempotent(tmp_path):
    first = wm.stage_cached_inputs(tmp_path, repo_root=wm.repo_root_path())
    probe = Path(first["external_refs_dir"]) / "_matrix_probe.txt"
    probe.write_text("kept")
    second = wm.stage_cached_inputs(tmp_path, repo_root=wm.repo_root_path())
    assert second == first
    assert probe.read_text() == "kept"


def test_write_matrix_yaml_renders_one_arch_and_two_cells(tmp_path):
    from xcquinox.alec.cluster.grid_config import expand_grid, load_grid_config
    out = tmp_path / "deep_3x16"
    path = wm.write_matrix_yaml("deep_3x16", out, repo_root=wm.repo_root_path())
    assert path == out.resolve() / "grid.yaml"
    cfg = load_grid_config(str(path))
    assert list(cfg.sweep.arch) == ["deep_3x16"]
    cells = expand_grid(cfg)
    assert len(cells) == 2
    assert sorted(c.subset_size for c in cells) == [1, 2]
    assert cfg.hyperparams.n_steps == 3
    assert cfg.pretrain.n_steps == 50
    assert cfg.pretrain.atoms == (("H", 1), ("O", 2))
    assert cfg.cluster.eval_workers == 1
    assert cfg.inputs.benchmark_refs_dir is None
    assert cfg.inputs.val_refs_dir is None


def test_write_matrix_yaml_paths_are_absolute_and_outside_the_repository(
        tmp_path):
    from xcquinox.alec.cluster.grid_config import load_grid_config
    out = tmp_path / "deep"
    cfg = load_grid_config(str(wm.write_matrix_yaml(
        "deep", out, repo_root=wm.repo_root_path())))
    repo = str(wm.repo_root_path())
    for value in (cfg.inputs.external_refs_dir, cfg.inputs.output_root,
                  cfg.pretrain.data_dir):
        assert os.path.isabs(value), value
        assert not value.startswith(repo), value
    # The ledger is READ-ONLY (only the JSON is read; no subset.traj is
    # opened), so it is consumed in place from the repository.
    assert cfg.inputs.subset_ledger_path.startswith(repo)
    assert os.path.isfile(cfg.inputs.subset_ledger_path)


def test_write_matrix_yaml_honours_shared_directories(tmp_path):
    from xcquinox.alec.cluster.grid_config import load_grid_config
    shared_refs = tmp_path / "shared_refs"
    shared_data = tmp_path / "shared_pretrain_data"
    cfg = load_grid_config(str(wm.write_matrix_yaml(
        "deep", tmp_path / "deep", repo_root=wm.repo_root_path(),
        external_refs_dir=shared_refs, pretrain_data_dir=shared_data)))
    assert cfg.inputs.external_refs_dir == str(shared_refs)
    assert cfg.pretrain.data_dir == str(shared_data)
    assert shared_data.is_dir(), "pretrain.data_dir must exist before datagen"


def test_write_matrix_yaml_refuses_an_unregistered_architecture(tmp_path):
    with pytest.raises(ValueError, match="not a registered architecture"):
        wm.write_matrix_yaml("no_such_arch", tmp_path / "x",
                             repo_root=wm.repo_root_path())


@pytest.mark.parametrize("arch", sorted(ARCHITECTURES))
def test_every_registered_architecture_renders_a_valid_grid(arch, tmp_path):
    """All 30-odd registry entries, not the 25 the figure layer renders."""
    from xcquinox.alec.cluster.domain import get_domain_profile
    from xcquinox.alec.cluster.grid_config import (load_grid_config,
                                                   validate_grid_semantics)
    cfg = load_grid_config(str(wm.write_matrix_yaml(
        arch, tmp_path / arch, repo_root=wm.repo_root_path(),
        external_refs_dir=tmp_path / "refs")))
    validate_grid_semantics(cfg, get_domain_profile(cfg.domain_profile))
```

- [ ] **Step 2: Run the tests and confirm they fail**

```bash
cd /home/awills/Documents/Research/xcquinox && python -m pytest xcquinox/alec/tests/test_cluster_workflow_matrix.py -v > /tmp/xcq-testlogs/task04_red.log 2>&1; echo "exit=$?"
```
Expected: collection fails with `ModuleNotFoundError: No module named 'xcquinox.alec.cluster.workflow_matrix'`. Read the log with `Read`.

- [ ] **Step 3: Create the template**

Create `xcquinox/alec/cluster/examples/workflow_matrix_template.yaml`:

```yaml
# =============================================================================
# workflow_matrix_template.yaml -- the one-architecture tiny identity of the
# per-architecture workflow matrix (SPEC_pretrain_fidelity_program.md 3.4)
# =============================================================================
#
# NOT a campaign config. This grid exists to run the harness stage sequence
# (submit dry-run -> datagen -> pretrain -> certificate -> preflight -> train
# -> eval -> validate_run) end to end, per architecture, in minutes, so a
# wiring defect is found before any YAML of a real campaign is rendered.
# Nothing it produces is a physics result: 3 training steps and 50 pretraining
# steps on two atoms fit nothing.
#
# Rendered by ``cluster.workflow_matrix.write_matrix_yaml``, which replaces the
# CHANGE_ME values programmatically (arch, the three path fields). As shipped,
# the file loads with ``load_grid_config`` and fails ``validate_grid_semantics``
# on the placeholder architecture, so an unedited copy cannot be submitted.
#
# Identity, fixed by the spec: def2-svp, grid level 1, the repository's cached
# subset ledger (notebooks/checkpoints_step7/alpha_on/subset_index_log.json)
# and its cached CCSD external references, which the renderer COPIES into the
# work root -- ``external_refs.precompute_all`` writes a run log into its cache
# directory on every call, and the repository copy is tracked.
# =============================================================================

# One architecture, two grid cells: subset sizes 1 and 2 of the JSD ledger
# (jsd/1 = H2O; jsd/2 = HLi + the OH+N2 -> H+N2O barrier), so the matrix covers
# a pure atomization cell and a mixed atomization/barrier cell. Solver oneshot:
# a single non-self-consistent pass, the cheapest path that still exercises the
# whole spec/eval plumbing.
sweep:
  arch: [CHANGE_ME_ARCH]
  loss: [L5_gradnorm_vxc_step7]
  metric: [jsd]
  subset_size: [1, 2]
  solver: [oneshot]

solvers:
  oneshot:
    mode: ONESHOT
    max_cycles: 0

# 3 optimization steps. validate_every 0 keeps the held-out validation slice
# out of the run (it would need inputs.val_refs_dir and a staged density-only
# slice); checkpoint_every 0 keeps the resume machinery out of it.
hyperparams:
  n_steps: 3
  lr_start: 0.01
  lr_end: 0.00001
  lr_decay_start: 0.2
  grad_clip: 1.0
  gradnorm_alpha: 1.5
  vxc_weight: 0.01
  density_weight: 0.1
  pbe_anchor_weight: 0.0
  require_atom_anchors: false
  validate_every: 0
  checkpoint_every: 0
  seed: 42

# benchmark_refs_dir and val_refs_dir stay UNSET: the first would submit a
# standalone CCSD reference job for the held-out pool, the second would stage a
# validation slice. Neither is part of the workflow assertion.
inputs:
  external_refs_dir: /nonexistent/CHANGE_ME/external_refs
  subset_ledger_path: /nonexistent/CHANGE_ME/subset_index_log.json
  basis: def2-svp
  grid_level: 1
  output_root: /nonexistent/CHANGE_ME/runs

# 50 pretraining steps on H (2S = 1) and O (2S = 2): one spin-1 and one spin-2
# open shell, enough for the generator to write every descriptor column
# (cusp, dm, rung35, rung35 multishell, meta-GGA alpha) plus the (s, alpha)
# mesh, so one data file serves every architecture in the registry.
pretrain:
  data_dir: /nonexistent/CHANGE_ME/pretrain_data
  n_steps: 50
  lr_start: 0.01
  lr_end: 0.00001
  lr_decay_start: 0.2
  grad_clip: 1.0
  seed: 42
  loss_weighting: integration
  atoms:
    H: 1
    O: 2

# SLURM fields are rendered but never submitted: the matrix runs `submit` in
# its default dry-run to create the run directory and resolved config, then
# invokes each stage module directly. No mail address is carried, because a
# script this file renders is never queued.
cluster:
  partition: ""
  time: "00:30:00"
  cpus_per_task: 4
  array_throttle: 1
  eval_array_throttle: 1
  max_concurrent_tasks: 2
  max_array_size: 1000
  device: cpu
  conda_env: ""
  conda_profile: ""
  mail_user: ""
  mail_type: NONE
  account: ""
  # Serial held-out eval: the sliced pool is six species, and the matrix runs
  # several architectures at once, so a worker ladder would oversubscribe.
  eval_workers: 1
  preflight_time: "00:30:00"
  eval_time: "00:30:00"
  pretrain_time: "00:30:00"

domain_profile: dfs_step7
on_precompute_failure: abort
bh76_mode: reaction_energy
# Matches the production arms: the polarized cnet, and therefore the
# zeta-carrying pretrain_data_polarized.npz that datagen generates.
use_polarized_correlation: true
# The separate eval array is rendered rather than folded into the train script,
# so `submit` produces the same five scripts the matrix executes as five
# stages.
inline_eval: false
defer_eval: false
# The certificate's tolerances and the enforcement bypass are NOT configured
# here: the certificate (SPEC 3.4 / 3.3) runs at the production identity and
# records its own verdict, and the workflow matrix passes
# XCQUINOX_FIDELITY_OVERRIDE_REASON in the environment because a 50-step
# pretrain on two atoms cannot meet tol_AE = 1.0 kcal/mol by construction.
```

- [ ] **Step 4: Create the module with the renderer**

Create `xcquinox/alec/cluster/workflow_matrix.py`:

```python
"""Per-architecture workflow matrix: the harness stage sequence at a tiny
identity, once per registered architecture.

SPEC_pretrain_fidelity_program.md 3.4 requires that, before any campaign YAML
is rendered, every architecture in the registry be driven through
``submit`` (dry-run) -> ``_datagen`` -> ``_pretrain`` -> the fidelity
certificate -> ``_preflight`` -> ``_train_task`` (two cells) ->
``_eval_one_spec`` (two cells) -> ``validate_run``, plus its spin-scaling
oracles, at def2-svp / grid level 1 against the repository's cached subset
ledger and CCSD references, with the held-out evaluation on a six-species
slice of the BH76 + W4-11 pool. The assertions are: every stage exits zero,
the expected artefacts exist, the certificate verdict is recorded, the
in-sample ``eval_df.csv`` and the sliced held-out channel are written, and the
architecture's oracles pass.

This module is a CALLER of the stage entry points, never a wrapper around
them: every stage runs as its own ``python -m`` subprocess with its own log,
exactly as SLURM would run it, so what the matrix verifies is the code the
cluster executes. ``runner`` is the single test seam (default
``subprocess.run``); with a fake runner the whole module is testable without
starting a process.
"""
from __future__ import annotations

import dataclasses
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from xcquinox.alec.config import ARCHITECTURES

#: Cached inputs of the tiny identity, relative to the repository root.
CACHED_REFS_RELPATH = "notebooks/checkpoints_step7/external_refs"
CACHED_LEDGER_RELPATH = "notebooks/checkpoints_step7/alpha_on/subset_index_log.json"

#: Rendered grid config filename inside an architecture's work directory.
GRID_FILENAME = "grid.yaml"


def repo_root_path() -> Path:
    """The repository root, four parents up from this file.

    ``<root>/xcquinox/alec/cluster/workflow_matrix.py`` -> ``<root>``.
    """
    return Path(__file__).resolve().parents[3]


def template_path() -> Path:
    """The checked-in one-architecture template (package data)."""
    return Path(__file__).resolve().parent / "examples" / \
        "workflow_matrix_template.yaml"


def stage_cached_inputs(dest_root, *, repo_root) -> dict:
    """Copy the cached CCSD references into ``dest_root`` and locate the ledger.

    ``external_refs.precompute_all`` creates its cache directory, migrates
    legacy filenames inside it and writes a ``_run_log_<UTC>.json`` on EVERY
    call, and ``run_oep_cascade`` may rewrite a species npz; the repository
    copy of these references is tracked, so the matrix works on a copy (74 MB,
    one per work root, shared by every architecture) rather than a symlink
    farm, which would carry those writes back into the tree. Existing run logs
    are not copied.

    The subset ledger is read-only for the harness (only the JSON is read; no
    ``subset.traj`` is opened, see ``spec_builder``), so it is consumed in
    place.
    """
    dest_root = Path(dest_root)
    refs_src = Path(repo_root) / CACHED_REFS_RELPATH
    ledger = Path(repo_root) / CACHED_LEDGER_RELPATH
    if not refs_src.is_dir():
        raise FileNotFoundError(
            f"cached CCSD references not found at {refs_src}; the workflow "
            "matrix consumes the repository's step-7 cache."
        )
    if not ledger.is_file():
        raise FileNotFoundError(
            f"cached subset ledger not found at {ledger}; the workflow matrix "
            "consumes the repository's step-7 ledger."
        )
    refs_dst = dest_root / "_inputs" / "external_refs"
    if not refs_dst.exists():
        refs_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(
            refs_src, refs_dst,
            ignore=shutil.ignore_patterns("_run_log_*.json"))
    return {"external_refs_dir": str(refs_dst),
            "subset_ledger_path": str(ledger)}


def write_matrix_yaml(arch, out_dir, *, repo_root,
                      external_refs_dir=None, pretrain_data_dir=None) -> Path:
    """Render the one-architecture tiny grid config into ``<out_dir>/grid.yaml``.

    The template is parsed and its four CHANGE_ME values are replaced as data,
    not as text, so a malformed substitution cannot produce a syntactically
    valid but semantically wrong config. ``external_refs_dir`` and
    ``pretrain_data_dir`` default to per-architecture directories under
    ``out_dir``; the matrix passes shared ones so the 74 MB reference copy and
    the pretrain-data generation are paid once per shard instead of once per
    architecture.
    """
    import yaml

    if arch not in ARCHITECTURES:
        raise ValueError(
            f"{arch!r} is not a registered architecture; "
            f"valid names: {sorted(ARCHITECTURES)}"
        )
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    with template_path().open() as f:
        raw = yaml.safe_load(f)

    if external_refs_dir is None:
        staged = stage_cached_inputs(out_dir, repo_root=repo_root)
        refs = staged["external_refs_dir"]
        ledger = staged["subset_ledger_path"]
    else:
        refs = str(Path(external_refs_dir).resolve())
        ledger = str((Path(repo_root) / CACHED_LEDGER_RELPATH).resolve())
    data_dir = Path(pretrain_data_dir).resolve() if pretrain_data_dir \
        else out_dir / "pretrain_data"
    # datagen writes into it and validate_grid_semantics warns when it is
    # absent on the submitting node.
    data_dir.mkdir(parents=True, exist_ok=True)

    raw["sweep"]["arch"] = [arch]
    raw["inputs"]["external_refs_dir"] = refs
    raw["inputs"]["subset_ledger_path"] = ledger
    raw["inputs"]["output_root"] = str(out_dir)
    raw["pretrain"]["data_dir"] = str(data_dir)

    path = out_dir / GRID_FILENAME
    with path.open("w") as f:
        yaml.safe_dump(raw, f, default_flow_style=False, sort_keys=True)
    return path
```

- [ ] **Step 5: Compile and run the tests**

```bash
cd /home/awills/Documents/Research/xcquinox && python -m py_compile xcquinox/alec/cluster/workflow_matrix.py && echo compiled
cd /home/awills/Documents/Research/xcquinox && python -m pytest xcquinox/alec/tests/test_cluster_workflow_matrix.py -v > /tmp/xcq-testlogs/task04_green.log 2>&1; echo "exit=$?"
```
Expected: PASS. The parametrized `test_every_registered_architecture_renders_a_valid_grid` reports one case per registry entry; read the count off the log and keep it -- it is the matrix population and goes into the HISTORY entry (Task 10).

**Covering test command:** `python -m pytest xcquinox/alec/tests/test_cluster_workflow_matrix.py -v > /tmp/xcq-testlogs/task04_green.log 2>&1`

---

## Task 5: The per-architecture oracle selector

**Files:**
- Modify: `xcquinox/alec/cluster/workflow_matrix.py` (append after `write_matrix_yaml`)
- Test: `xcquinox/alec/tests/test_cluster_workflow_matrix.py` (append)

**Interfaces:**
- Consumes: `xcquinox.alec.config.ARCHITECTURES`; the spec-3.1 test module `xcquinox/alec/tests/test_spin_scaling_oracles.py`, whose architecture-carrying oracles are parametrized over `sorted(alec.ARCHITECTURES)`.
- Produces:
  - `workflow_matrix.ORACLE_MODULE: str = "test_spin_scaling_oracles"`
  - `workflow_matrix.ORACLE_TEST_TARGET: str = "xcquinox/alec/tests"`
  - `workflow_matrix.oracle_selector(arch, archs=None) -> str`

- [ ] **Step 1: Write the failing tests**

Append to `xcquinox/alec/tests/test_cluster_workflow_matrix.py`:

```python
# ---------------------------------------------------------------------------
# Oracle selector
# ---------------------------------------------------------------------------

def test_oracle_selector_names_the_module_and_the_architecture():
    got = wm.oracle_selector("deep_rung35_mgga_3x16")
    assert got.startswith("test_spin_scaling_oracles and ")
    assert " and deep_rung35_mgga_3x16" in got


def test_oracle_selector_excludes_names_that_contain_this_one():
    """pytest -k matches SUBSTRINGS of the node id, so a bare 'deep_cusp'
    selects deep_cusp_3x16 and deep_cusp_mgga_3x16 as well. Every longer
    registry name containing this one is excluded explicitly."""
    got = wm.oracle_selector("deep_cusp",
                             archs=["deep_cusp", "deep_cusp_3x16",
                                    "deep_cusp_mgga_3x16", "deep_dm"])
    assert got == ("test_spin_scaling_oracles and deep_cusp "
                   "and not deep_cusp_3x16 and not deep_cusp_mgga_3x16")


def test_oracle_selector_adds_no_exclusion_when_the_name_is_unique():
    got = wm.oracle_selector("deep_dm_3x16",
                             archs=["deep_dm", "deep_dm_3x16"])
    assert got == "test_spin_scaling_oracles and deep_dm_3x16"


def test_oracle_selector_refuses_an_unregistered_architecture():
    with pytest.raises(ValueError, match="not a registered architecture"):
        wm.oracle_selector("no_such_arch")


@pytest.mark.parametrize("arch", sorted(ARCHITECTURES))
def test_oracle_selector_is_a_valid_k_expression(arch):
    """Every term must be a bare Python identifier: pytest parses -k as an
    expression, so a name with a hyphen or a dot would be a syntax error."""
    import keyword
    selector = wm.oracle_selector(arch)
    for token in selector.split():
        if token in ("and", "not"):
            continue
        assert token.isidentifier() and not keyword.iskeyword(token), token


def test_oracle_selector_selects_this_architecture_only(tmp_path):
    """Contract with the spec-3.1 oracle module: the selector must resolve to a
    non-empty set of collected tests, all of them this architecture's. Skipped
    until that module is installed, so this plan is executable on its own."""
    import subprocess
    module = (wm.repo_root_path() / "xcquinox" / "alec" / "tests"
              / f"{wm.ORACLE_MODULE}.py")
    if not module.is_file():
        pytest.skip(f"{module} not installed yet (spec 3.1)")
    arch = "deep_cusp" if "deep_cusp" in ARCHITECTURES else sorted(ARCHITECTURES)[0]
    log = tmp_path / "collect.log"
    with log.open("w") as fh:
        rc = subprocess.run(
            [sys.executable, "-m", "pytest", wm.ORACLE_TEST_TARGET,
             "--collect-only", "-q", "-p", "no:randomly",
             "-k", wm.oracle_selector(arch)],
            cwd=str(wm.repo_root_path()), stdout=fh,
            stderr=subprocess.STDOUT, check=False).returncode
    text = log.read_text()
    assert rc == 0, text
    node_ids = [ln for ln in text.splitlines() if "::" in ln]
    assert node_ids, text
    for node in node_ids:
        assert f"[{arch}]" in node or f"-{arch}]" in node, node
```

- [ ] **Step 2: Run the tests and confirm they fail**

```bash
python -m pytest xcquinox/alec/tests/test_cluster_workflow_matrix.py -v > /tmp/xcq-testlogs/task05_red.log 2>&1; echo "exit=$?"
```
Expected: `AttributeError: module 'xcquinox.alec.cluster.workflow_matrix' has no attribute 'oracle_selector'`. Read the log with `Read`.

- [ ] **Step 3: Add the selector**

Append to `xcquinox/alec/cluster/workflow_matrix.py`:

```python
# ---------------------------------------------------------------------------
# Oracle selection
# ---------------------------------------------------------------------------

#: Test module of the spin-scaling oracles O1-O4
#: (SPEC_pretrain_fidelity_program.md 3.1). Its architecture-carrying oracles
#: are parametrized over ``sorted(ARCHITECTURES)``, so a node id ends in
#: ``[<arch>]`` (or ``[<species>-<arch>]``) and one architecture's oracles are
#: selectable with ``-k``.
ORACLE_MODULE = "test_spin_scaling_oracles"

#: Collection target for the oracle run. A directory rather than the module
#: path so the module name in the selector is what pins the module; pytest
#: matches ``-k`` against the module name as well as the test name.
ORACLE_TEST_TARGET = "xcquinox/alec/tests"


def oracle_selector(arch, archs=None) -> str:
    """A pytest ``-k`` expression selecting one architecture's oracles.

    ``-k`` matches SUBSTRINGS of the node id, and the registry contains names
    that are prefixes of others (``deep`` of ``deep_attn``, ``deep_cusp`` of
    ``deep_cusp_mgga_3x16``, ``shallow`` of ``shallow_attn``), so a bare name
    would silently pull in a sibling architecture's cases and report them as
    this one's. Every longer registry name containing this one is therefore
    excluded explicitly. Every registry name is a Python identifier, which is
    what pytest's expression parser accepts as a term.
    """
    names = sorted(ARCHITECTURES) if archs is None else sorted(archs)
    if arch not in names:
        raise ValueError(
            f"{arch!r} is not a registered architecture; "
            f"valid names: {names}"
        )
    terms = [ORACLE_MODULE, arch]
    terms += [f"not {other}" for other in names
              if other != arch and arch in other]
    return " and ".join(terms)
```

- [ ] **Step 4: Compile and run the tests**

```bash
cd /home/awills/Documents/Research/xcquinox && python -m py_compile xcquinox/alec/cluster/workflow_matrix.py && echo compiled
cd /home/awills/Documents/Research/xcquinox && python -m pytest xcquinox/alec/tests/test_cluster_workflow_matrix.py -v > /tmp/xcq-testlogs/task05_green.log 2>&1; echo "exit=$?"
```
Expected: PASS (the contract test SKIPs until the spec-3.1 module lands; re-run this file once it does and confirm it turns green).

**Covering test command:** `python -m pytest xcquinox/alec/tests/test_cluster_workflow_matrix.py -v > /tmp/xcq-testlogs/task05_green.log 2>&1`

---

## Task 6: `run_arch` -- the stage sequence for one architecture

**Files:**
- Modify: `xcquinox/alec/cluster/workflow_matrix.py` (append after `oracle_selector`)
- Test: `xcquinox/alec/tests/test_cluster_workflow_matrix.py` (append)

**Interfaces:**
- Consumes: `write_matrix_yaml`, `stage_cached_inputs`, `oracle_selector` (Tasks 4, 5); `full_benchmark_pools.HELDOUT_SPECIES_SLICE_ENV` (Task 1); the certificate CLI and its `fidelity_certificate.json` (spec 3.3).
- Produces:
  - `workflow_matrix.Stage` (frozen dataclass: `name`, `argv`, `allow_nonzero=False`, `env_extra=()`)
  - `workflow_matrix.STAGE_ORDER: tuple[str, ...]`
  - `workflow_matrix.HELDOUT_SPECIES_SLICE: str = "h,h2,o,oh,n2o,n2ohts"`
  - `workflow_matrix.FIDELITY_OVERRIDE_ENV`, `FIDELITY_OVERRIDE_REASON`
  - `workflow_matrix.DEFAULT_STAGE_TIMEOUT_S: int`, `TIMEOUT_RC: int = 124`
  - `workflow_matrix.stage_plan(run_dir, *, species_slice=HELDOUT_SPECIES_SLICE, device="cpu") -> tuple[Stage, ...]`
  - `workflow_matrix.run_arch(arch, work_root, *, runner=subprocess.run, timeout_s=DEFAULT_STAGE_TIMEOUT_S, repo_root=None, external_refs_dir=None, pretrain_data_dir=None, species_slice=HELDOUT_SPECIES_SLICE, threads=4, run_oracles=True) -> dict` returning
    `{"arch": str, "run_dir": str | None, "seconds": float, "stages": [{"name","rc","seconds","log"}], "artefacts": {label: {"path","exists"}}, "certificate_verdict": str | None, "oracle_tests": {"rc","summary_line","log","selector"}}`

- [ ] **Step 1: Write the failing tests**

Append to `xcquinox/alec/tests/test_cluster_workflow_matrix.py`:

```python
# ---------------------------------------------------------------------------
# run_arch, driven by a fake runner (no subprocess is started)
# ---------------------------------------------------------------------------

class FakeRunner:
    """Stand-in for ``subprocess.run``.

    Records ``(argv, env)`` per call, echoes the submit stage's run-dir line
    into the stage log, materializes whatever artefacts the caller asked for,
    and returns the return code scheduled for that stage name.
    """

    def __init__(self, run_dir, *, rc_by_stage=None, artefacts=(),
                 verdict="PASS", oracle_summary="12 passed in 3.4s"):
        self.run_dir = Path(run_dir)
        self.rc_by_stage = dict(rc_by_stage or {})
        self.artefacts = tuple(artefacts)
        self.verdict = verdict
        self.oracle_summary = oracle_summary
        self.calls = []

    def __call__(self, argv, **kwargs):
        self.calls.append((list(argv), dict(kwargs.get("env") or {})))
        stream = kwargs.get("stdout")
        stage = self._stage_of(argv)
        # The tag goes FIRST: run_arch reads the LAST non-empty line of the
        # oracle log as its summary line.
        stream.write(f"[fake] {stage}\n")
        if stage == "submit":
            self.run_dir.mkdir(parents=True, exist_ok=True)
            stream.write(f"submit: created run dir {self.run_dir}\n")
            stream.write(f"submit: run dir = {self.run_dir}\n")
        elif stage == "certificate" and self.verdict is not None:
            cert = self.run_dir / "pretrain" / "deep" / \
                "fidelity_certificate.json"
            cert.parent.mkdir(parents=True, exist_ok=True)
            cert.write_text(json.dumps({"verdict": self.verdict}))
        elif stage == "oracles":
            stream.write("......\n")
            stream.write(f"{self.oracle_summary}\n")
        if stage == "validate_run":
            for rel in self.artefacts:
                target = self.run_dir / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("x")

        class _Completed:
            returncode = self.rc_by_stage.get(stage, 0)

        return _Completed()

    @staticmethod
    def _stage_of(argv):
        joined = " ".join(argv)
        if " -m pytest " in f" {joined} ":
            return "oracles"
        for token, name in (
                ("xcquinox.alec.cluster._datagen", "datagen"),
                ("xcquinox.alec.cluster._pretrain", "pretrain"),
                ("xcquinox.alec.cluster.fidelity", "certificate"),
                ("xcquinox.alec.cluster._preflight", "preflight"),
                ("xcquinox.alec.cluster._train_task", "train"),
                ("xcquinox.alec.cluster._eval_one_spec", "eval"),
                ("xcquinox.alec.cluster.validate_run", "validate_run"),
        ):
            if token in joined:
                return name
        return "submit"


def _run_arch(tmp_path, arch="deep", **kw):
    fake = kw.pop("fake", None)
    run_dir = tmp_path / arch / "runs" / "run_20260821T000000Z"
    if fake is None:
        fake = FakeRunner(run_dir)
    result = wm.run_arch(arch, tmp_path, runner=fake,
                         repo_root=wm.repo_root_path(),
                         external_refs_dir=tmp_path / "refs", **kw)
    return result, fake


def test_run_arch_runs_the_ten_stages_in_order(tmp_path):
    result, fake = _run_arch(tmp_path)
    assert [s["name"] for s in result["stages"]] == list(wm.STAGE_ORDER)
    assert [s["rc"] for s in result["stages"]] == [0] * len(wm.STAGE_ORDER)
    assert result["arch"] == "deep"
    assert result["run_dir"].endswith("run_20260821T000000Z")
    assert result["seconds"] >= 0.0


def test_run_arch_issues_the_exact_stage_command_lines(tmp_path):
    result, fake = _run_arch(tmp_path)
    run_dir = result["run_dir"]
    argvs = [argv for argv, _env in fake.calls]
    assert argvs[0][1:5] == [
        "-m", "xcquinox.alec.cluster", "submit",
        str(Path(tmp_path).resolve() / "deep" / "grid.yaml")]
    assert argvs[0][5:] == ["--partition", "local"]
    assert argvs[1][1:] == ["-m", "xcquinox.alec.cluster._datagen", run_dir]
    assert argvs[2][1:] == ["-m", "xcquinox.alec.cluster._pretrain", run_dir,
                            "0"]
    assert argvs[3][1:] == ["-m", "xcquinox.alec.cluster.fidelity", run_dir,
                            "0"]
    assert argvs[4][1:] == ["-m", "xcquinox.alec.cluster._preflight", run_dir]
    assert argvs[5][1:] == ["-m", "xcquinox.alec.cluster._train_task", run_dir,
                            "0", "--device", "cpu"]
    assert argvs[6][1:] == ["-m", "xcquinox.alec.cluster._train_task", run_dir,
                            "1", "--device", "cpu"]
    assert argvs[7][1:] == ["-m", "xcquinox.alec.cluster._eval_one_spec",
                            run_dir, "0"]
    assert argvs[8][1:] == ["-m", "xcquinox.alec.cluster._eval_one_spec",
                            run_dir, "1"]
    assert argvs[9][1:] == ["-m", "xcquinox.alec.cluster.validate_run",
                            run_dir]


def test_run_arch_passes_the_species_slice_to_the_eval_stages_only(tmp_path):
    from xcquinox.alec.full_benchmark_pools import HELDOUT_SPECIES_SLICE_ENV
    _result, fake = _run_arch(tmp_path)
    by_stage = {FakeRunner._stage_of(argv): env for argv, env in fake.calls}
    assert by_stage["eval"][HELDOUT_SPECIES_SLICE_ENV] == \
        "h,h2,o,oh,n2o,n2ohts"
    for stage in ("datagen", "pretrain", "preflight", "train",
                  "validate_run"):
        assert HELDOUT_SPECIES_SLICE_ENV not in by_stage[stage]


def test_run_arch_exports_the_fidelity_override_reason_to_every_stage(
        tmp_path):
    _result, fake = _run_arch(tmp_path)
    for argv, env in fake.calls:
        if " -m pytest " in " " + " ".join(argv) + " ":
            continue
        assert env[wm.FIDELITY_OVERRIDE_ENV] == wm.FIDELITY_OVERRIDE_REASON


def test_run_arch_pins_cpu_and_float64_for_every_stage(tmp_path):
    _result, fake = _run_arch(tmp_path)
    for _argv, env in fake.calls:
        assert env["JAX_PLATFORMS"] == "cpu"
        assert env["JAX_ENABLE_X64"] == "1"
        assert env["OMP_NUM_THREADS"] == "4"


def test_run_arch_stops_at_the_first_non_zero_stage(tmp_path):
    run_dir = tmp_path / "deep" / "runs" / "run_20260821T000000Z"
    fake = FakeRunner(run_dir, rc_by_stage={"preflight": 1})
    result, fake = _run_arch(tmp_path, fake=fake)
    assert [s["name"] for s in result["stages"]] == [
        "submit", "datagen", "pretrain", "certificate", "preflight"]
    assert result["stages"][-1]["rc"] == 1
    # The oracles still run: they are a property of the installed code, not of
    # this run directory.
    assert result["oracle_tests"]["rc"] == 0


def test_run_arch_does_not_stop_on_a_failing_certificate(tmp_path):
    """The certificate's VERDICT is recorded, not required: a 50-step pretrain
    on two atoms cannot meet tol_AE = 1.0 kcal/mol, and spec 3.4 asks the
    matrix to record the verdict while every stage exits zero."""
    run_dir = tmp_path / "deep" / "runs" / "run_20260821T000000Z"
    fake = FakeRunner(run_dir, rc_by_stage={"certificate": 1}, verdict="FAIL")
    result, _fake = _run_arch(tmp_path, fake=fake)
    assert [s["name"] for s in result["stages"]] == list(wm.STAGE_ORDER)
    assert result["certificate_verdict"] == "FAIL"
    cert_stage = [s for s in result["stages"] if s["name"] == "certificate"][0]
    assert cert_stage["rc"] == 1


def test_run_arch_records_no_verdict_when_the_certificate_is_absent(tmp_path):
    run_dir = tmp_path / "deep" / "runs" / "run_20260821T000000Z"
    fake = FakeRunner(run_dir, verdict=None)
    result, _fake = _run_arch(tmp_path, fake=fake)
    assert result["certificate_verdict"] is None


def test_run_arch_writes_one_log_per_stage(tmp_path):
    result, _fake = _run_arch(tmp_path)
    for stage in result["stages"]:
        path = Path(stage["log"])
        assert path.is_file(), path
        assert path.parent == Path(tmp_path).resolve() / "deep" / "logs"
        assert "[fake]" in path.read_text()
    assert Path(result["oracle_tests"]["log"]).is_file()


def test_run_arch_reports_missing_artefacts(tmp_path):
    result, _fake = _run_arch(tmp_path)
    art = result["artefacts"]
    assert art["manifest"]["exists"] is False
    assert art["eval_df[0]"]["path"].endswith(
        "checkpoints/spec_0000/eval_df.csv")
    assert art["holdout_sliced[1]"]["path"].endswith(
        "checkpoints/spec_0001/eval_holdout/sliced_eval.json")


def test_run_arch_marks_the_artefacts_the_stages_produced(tmp_path):
    run_dir = tmp_path / "deep" / "runs" / "run_20260821T000000Z"
    produced = (
        "resolved_config.yaml",
        "manifest.json",
        "scripts/datagen.sbatch", "scripts/pretrain.sbatch",
        "scripts/preflight.sbatch", "scripts/train_array.sbatch",
        "scripts/eval_array.sbatch",
        "pretrain/deep/xnet.eqx", "pretrain/deep/cnet.eqx",
        "pretrain/deep/pretrain_metadata.json",
        "specs/spec_0000.spec", "specs/spec_0001.spec",
        "checkpoints/spec_0000/model.eqx", "checkpoints/spec_0001/model.eqx",
        "checkpoints/spec_0000/eval_df.csv",
        "checkpoints/spec_0001/eval_df.csv",
        "checkpoints/spec_0000/eval_holdout/test_set.csv",
        "checkpoints/spec_0001/eval_holdout/test_set.csv",
        "checkpoints/spec_0000/eval_holdout/eval_metadata.json",
        "checkpoints/spec_0001/eval_holdout/eval_metadata.json",
        "checkpoints/spec_0000/eval_holdout/sliced_eval.json",
        "checkpoints/spec_0001/eval_holdout/sliced_eval.json",
    )
    fake = FakeRunner(run_dir, artefacts=produced)
    result, _fake = _run_arch(tmp_path, fake=fake)
    missing = [k for k, v in result["artefacts"].items() if not v["exists"]]
    assert missing == ["pretrain_data"], missing


def test_run_arch_records_the_oracle_summary_line(tmp_path):
    result, _fake = _run_arch(tmp_path)
    oracles = result["oracle_tests"]
    assert oracles["rc"] == 0
    assert oracles["summary_line"] == "12 passed in 3.4s"
    assert oracles["selector"] == wm.oracle_selector("deep")


def test_run_arch_can_skip_the_oracles(tmp_path):
    result, fake = _run_arch(tmp_path, run_oracles=False)
    assert result["oracle_tests"]["rc"] is None
    assert not any(" -m pytest " in " " + " ".join(argv) + " "
                   for argv, _env in fake.calls)


def test_run_arch_records_a_timeout_as_rc_124(tmp_path):
    import subprocess as sp
    run_dir = tmp_path / "deep" / "runs" / "run_20260821T000000Z"

    class _Timeout(FakeRunner):
        def __call__(self, argv, **kwargs):
            if "_preflight" in " ".join(argv):
                self.calls.append((list(argv), dict(kwargs.get("env") or {})))
                raise sp.TimeoutExpired(cmd=argv, timeout=1)
            return super().__call__(argv, **kwargs)

    result, _fake = _run_arch(tmp_path, fake=_Timeout(run_dir))
    assert result["stages"][-1]["name"] == "preflight"
    assert result["stages"][-1]["rc"] == wm.TIMEOUT_RC
    assert "exceeded" in Path(result["stages"][-1]["log"]).read_text()


def test_run_arch_reports_a_submit_that_printed_no_run_dir(tmp_path):
    run_dir = tmp_path / "deep" / "runs" / "run_20260821T000000Z"

    class _Silent(FakeRunner):
        def __call__(self, argv, **kwargs):
            if FakeRunner._stage_of(argv) == "submit":
                self.calls.append((list(argv), dict(kwargs.get("env") or {})))
                kwargs["stdout"].write("submit: DRY-RUN\n")

                class _C:
                    returncode = 0
                return _C()
            return super().__call__(argv, **kwargs)

    result, _fake = _run_arch(tmp_path, fake=_Silent(run_dir))
    assert result["run_dir"] is None
    assert [s["name"] for s in result["stages"]] == ["submit"]
    assert result["stages"][0]["rc"] != 0
```

- [ ] **Step 2: Run the tests and confirm they fail**

```bash
cd /home/awills/Documents/Research/xcquinox && python -m pytest xcquinox/alec/tests/test_cluster_workflow_matrix.py -v > /tmp/xcq-testlogs/task06_red.log 2>&1; echo "exit=$?"
```
Expected: `AttributeError: module 'xcquinox.alec.cluster.workflow_matrix' has no attribute 'STAGE_ORDER'` / `... 'run_arch'`. Read the log with `Read`.

- [ ] **Step 3: Add the stage table and the runner**

Append to `xcquinox/alec/cluster/workflow_matrix.py`:

```python
# ---------------------------------------------------------------------------
# Stage table
# ---------------------------------------------------------------------------

#: Six species of the BH76 + W4-11 pool closing three reactions -- one BH76
#: barrier (h + n2o -> n2ohts) and two W4-11 atomizations (h2, oh) -- over both
#: spin types (RKS h2 / n2o, UKS h / o / oh / n2ohts). The full pool is 216
#: reactions over 214 species and hours of SCF per grid cell
#: (SPEC_pretrain_fidelity_program.md 3.4). A slice of six MOLECULES with no
#: atoms closes no reaction at all and would leave the reaction math untested,
#: which is why the atoms are in it.
HELDOUT_SPECIES_SLICE = "h,h2,o,oh,n2o,n2ohts"

#: Environment knob the certificate's enforcement points honour. The
#: certificate still computes and records its own verdict; only the
#: enforcement is bypassed, because a 50-step pretrain on two atoms at
#: def2-svp cannot meet tol_AE = 1.0 kcal/mol and spec 3.4 requires every
#: stage to exit zero while the verdict is merely recorded.
FIDELITY_OVERRIDE_ENV = "XCQUINOX_FIDELITY_OVERRIDE_REASON"
FIDELITY_OVERRIDE_REASON = (
    "workflow matrix (SPEC_pretrain_fidelity_program.md 3.4): the matrix "
    "pretrains 50 steps on two atoms at def2-svp, which cannot reproduce the "
    "parent to tol_AE = 1.0 kcal/mol; the matrix verifies stage wiring and "
    "RECORDS the certificate verdict rather than requiring it"
)

#: Per-stage wall-clock cap. The expectation is 1-6 minutes per stage at this
#: identity; an hour is the point at which a stage is hung rather than slow.
DEFAULT_STAGE_TIMEOUT_S = 3600

#: Return code recorded for a stage killed by the timeout (the shell's
#: convention for a command terminated by ``timeout``).
TIMEOUT_RC = 124

#: Stage names in execution order; the report's column legend.
STAGE_ORDER = ("submit", "datagen", "pretrain", "certificate", "preflight",
               "train[0]", "train[1]", "eval[0]", "eval[1]", "validate_run")

_RUN_DIR_LINE = re.compile(r"^submit: run dir = (?P<path>\S.*)$")


@dataclasses.dataclass(frozen=True)
class Stage:
    """One stage invocation: its name, its argv, and its failure policy."""

    name: str
    argv: tuple
    #: The certificate is the one stage whose non-zero exit does not stop the
    #: sequence: spec 3.4 records its verdict, it does not require a PASS.
    allow_nonzero: bool = False
    #: Extra environment for this stage only, as ``((key, value), ...)``.
    env_extra: tuple = ()


def stage_plan(run_dir, *, species_slice=HELDOUT_SPECIES_SLICE,
               device="cpu") -> tuple:
    """The nine stages after ``submit``, in the order the job graph runs them.

    Each stage is the module SLURM would invoke, with the same argument vector,
    so the matrix verifies the code the cluster executes rather than an
    in-process re-implementation of it. The species slice reaches the eval
    stages ONLY: no other stage reads it, and confining it here keeps the
    training pool provably untouched.
    """
    py = sys.executable
    run_dir = str(run_dir)
    slice_env = ((_HELDOUT_SLICE_ENV, species_slice),) if species_slice else ()
    return (
        Stage("datagen",
              (py, "-m", "xcquinox.alec.cluster._datagen", run_dir)),
        Stage("pretrain",
              (py, "-m", "xcquinox.alec.cluster._pretrain", run_dir, "0")),
        Stage("certificate",
              (py, "-m", "xcquinox.alec.cluster.fidelity", run_dir, "0"),
              allow_nonzero=True),
        Stage("preflight",
              (py, "-m", "xcquinox.alec.cluster._preflight", run_dir)),
        Stage("train[0]",
              (py, "-m", "xcquinox.alec.cluster._train_task", run_dir, "0",
               "--device", device)),
        Stage("train[1]",
              (py, "-m", "xcquinox.alec.cluster._train_task", run_dir, "1",
               "--device", device)),
        Stage("eval[0]",
              (py, "-m", "xcquinox.alec.cluster._eval_one_spec", run_dir, "0"),
              env_extra=slice_env),
        Stage("eval[1]",
              (py, "-m", "xcquinox.alec.cluster._eval_one_spec", run_dir, "1"),
              env_extra=slice_env),
        Stage("validate_run",
              (py, "-m", "xcquinox.alec.cluster.validate_run", run_dir)),
    )


def _base_env(threads):
    """Process environment shared by every stage.

    fp32 versus fp64 silently changes every energy, and the matrix runs several
    architectures at once on one box, so the JAX backend and the BLAS thread
    caps are pinned here rather than inherited. Any inherited species slice is
    dropped: only the eval stages get one, and only from ``stage_plan``.
    """
    env = dict(os.environ)
    env["JAX_PLATFORMS"] = "cpu"
    env["JAX_ENABLE_X64"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[key] = str(threads)
    env[FIDELITY_OVERRIDE_ENV] = FIDELITY_OVERRIDE_REASON
    env.pop(_HELDOUT_SLICE_ENV, None)
    return env


def _run_stage(name, argv, log_path, *, runner, env, timeout_s, cwd):
    """Run one stage into its own log; return the stage record."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    with log_path.open("w") as fh:
        fh.write(f"$ {' '.join(str(a) for a in argv)}\n")
        fh.flush()
        try:
            completed = runner(list(argv), stdout=fh,
                               stderr=subprocess.STDOUT, cwd=str(cwd),
                               env=env, timeout=timeout_s, check=False)
            rc = int(completed.returncode)
        except subprocess.TimeoutExpired:
            fh.write(f"\n[workflow_matrix] {name} exceeded {timeout_s} s and "
                     "was killed\n")
            rc = TIMEOUT_RC
    return {"name": name, "rc": rc,
            "seconds": round(time.monotonic() - t0, 1), "log": str(log_path)}


def _parse_run_dir(log_path):
    """The run directory ``submit`` created, from its own log.

    ``cmd_submit`` ends with ``submit: run dir = <path>``; reading it is exact,
    where globbing ``<output_root>/runs`` would race a concurrent shard.
    """
    run_dir = None
    with Path(log_path).open(errors="replace") as fh:
        for line in fh:
            match = _RUN_DIR_LINE.match(line.strip())
            if match:
                run_dir = match.group("path").strip()
    return run_dir


def _manifest_width(run_dir):
    """Zero-pad width of the spec indices, from the manifest (default 4)."""
    path = Path(run_dir) / "manifest.json"
    try:
        with path.open() as fh:
            return int(json.load(fh)["width"])
    except (OSError, ValueError, KeyError):
        return 4


def _artefact_paths(run_dir, arch, data_dir, polarized=True):
    """Every artefact the stage sequence is expected to leave behind."""
    run = Path(run_dir)
    width = _manifest_width(run)
    pre = run / "pretrain" / arch
    scripts = run / "scripts"
    npz = ("pretrain_data_polarized.npz" if polarized
           else "pretrain_data.npz")
    labels = {
        "resolved_config": run / "resolved_config.yaml",
        "script_datagen": scripts / "datagen.sbatch",
        "script_pretrain": scripts / "pretrain.sbatch",
        "script_preflight": scripts / "preflight.sbatch",
        "script_train": scripts / "train_array.sbatch",
        "script_eval": scripts / "eval_array.sbatch",
        "pretrain_data": Path(data_dir) / npz,
        "pretrain_xnet": pre / "xnet.eqx",
        "pretrain_cnet": pre / "cnet.eqx",
        "pretrain_metadata": pre / "pretrain_metadata.json",
        "certificate": pre / "fidelity_certificate.json",
        "manifest": run / "manifest.json",
    }
    for idx in (0, 1):
        ckpt = run / "checkpoints" / f"spec_{idx:0{width}d}"
        labels[f"spec[{idx}]"] = run / "specs" / f"spec_{idx:0{width}d}.spec"
        labels[f"model[{idx}]"] = ckpt / "model.eqx"
        labels[f"eval_df[{idx}]"] = ckpt / "eval_df.csv"
        labels[f"holdout_test_set[{idx}]"] = \
            ckpt / "eval_holdout" / "test_set.csv"
        labels[f"holdout_metadata[{idx}]"] = \
            ckpt / "eval_holdout" / "eval_metadata.json"
        labels[f"holdout_sliced[{idx}]"] = \
            ckpt / "eval_holdout" / "sliced_eval.json"
    return {name: {"path": str(path), "exists": path.exists()}
            for name, path in labels.items()}


def _read_certificate_verdict(run_dir, arch):
    """The certificate's top-level verdict, or None when it wrote nothing."""
    path = Path(run_dir) / "pretrain" / arch / "fidelity_certificate.json"
    try:
        with path.open() as fh:
            return json.load(fh).get("verdict")
    except (OSError, ValueError):
        return None


def _summary_line(log_path):
    """The last non-empty line of a pytest log -- its one-line result."""
    try:
        lines = [ln.strip() for ln in
                 Path(log_path).read_text(errors="replace").splitlines()
                 if ln.strip()]
    except OSError:
        return ""
    return lines[-1] if lines else ""


def _run_oracles(arch, log_path, *, runner, env, timeout_s, cwd):
    """Run this architecture's slice of the spin-scaling oracles O1-O4."""
    selector = oracle_selector(arch)
    argv = (sys.executable, "-m", "pytest", ORACLE_TEST_TARGET,
            "-k", selector, "-q", "-p", "no:randomly")
    record = _run_stage("oracles", argv, log_path, runner=runner, env=env,
                        timeout_s=timeout_s, cwd=cwd)
    return {"rc": record["rc"], "summary_line": _summary_line(log_path),
            "log": record["log"], "selector": selector,
            "seconds": record["seconds"]}


def run_arch(arch, work_root, *, runner=subprocess.run,
             timeout_s=DEFAULT_STAGE_TIMEOUT_S, repo_root=None,
             external_refs_dir=None, pretrain_data_dir=None,
             species_slice=HELDOUT_SPECIES_SLICE, threads=4,
             run_oracles=True) -> dict:
    """Drive one architecture through the whole stage sequence.

    ``submit`` runs in its default DRY-RUN, which creates the run directory,
    writes ``resolved_config.yaml`` and renders every sbatch script without
    calling SLURM; the matrix then invokes each stage module itself. The
    sequence stops at the first non-zero exit -- a stage's inputs are the
    previous stage's outputs, so continuing past a failure measures nothing --
    except for the certificate, whose verdict is recorded rather than
    required. The oracles run regardless: they are a property of the installed
    code, not of this run directory.
    """
    root = Path(repo_root) if repo_root is not None else repo_root_path()
    arch_root = Path(work_root).resolve() / arch
    logs_dir = arch_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    grid_path = write_matrix_yaml(
        arch, arch_root, repo_root=root,
        external_refs_dir=external_refs_dir,
        pretrain_data_dir=pretrain_data_dir)
    data_dir = (Path(pretrain_data_dir) if pretrain_data_dir
                else arch_root / "pretrain_data")
    env = _base_env(threads)
    t_all = time.monotonic()

    stages = [_run_stage(
        "submit",
        (sys.executable, "-m", "xcquinox.alec.cluster", "submit",
         str(grid_path), "--partition", "local"),
        logs_dir / "submit.log", runner=runner, env=env, timeout_s=timeout_s,
        cwd=root)]
    run_dir = _parse_run_dir(stages[0]["log"]) if stages[0]["rc"] == 0 else None
    if stages[0]["rc"] == 0 and run_dir is None:
        # A zero exit with no run-dir line means the dry-run changed its
        # contract; every later stage takes the run dir as argv[0], so there is
        # nothing to run.
        stages[0]["rc"] = 2
        with (logs_dir / "submit.log").open("a") as fh:
            fh.write("\n[workflow_matrix] submit printed no "
                     "'submit: run dir = ' line\n")

    if run_dir is not None and stages[0]["rc"] == 0:
        for stage in stage_plan(run_dir, species_slice=species_slice):
            stage_env = dict(env)
            stage_env.update(dict(stage.env_extra))
            log_name = stage.name.replace("[", "_").replace("]", "")
            stages.append(_run_stage(
                stage.name, stage.argv, logs_dir / f"{log_name}.log",
                runner=runner, env=stage_env, timeout_s=timeout_s, cwd=root))
            if stages[-1]["rc"] != 0 and not stage.allow_nonzero:
                break

    oracle_tests = {"rc": None, "summary_line": "", "log": None,
                    "selector": oracle_selector(arch), "seconds": 0.0}
    if run_oracles:
        oracle_tests = _run_oracles(
            arch, logs_dir / "oracles.log", runner=runner, env=env,
            timeout_s=timeout_s, cwd=root)

    artefacts = (_artefact_paths(run_dir, arch, data_dir) if run_dir
                 else {})
    return {
        "arch": arch,
        "run_dir": run_dir,
        "seconds": round(time.monotonic() - t_all, 1),
        "stages": stages,
        "artefacts": artefacts,
        "certificate_verdict": (_read_certificate_verdict(run_dir, arch)
                                if run_dir else None),
        "oracle_tests": oracle_tests,
    }
```

Add the slice-variable import to the module's import block (top of the file), so `stage_plan` and `_base_env` name it from its owner rather than repeating the string:

```python
from xcquinox.alec.full_benchmark_pools import (
    HELDOUT_SPECIES_SLICE_ENV as _HELDOUT_SLICE_ENV,
)
```

- [ ] **Step 4: Compile and run the tests**

```bash
cd /home/awills/Documents/Research/xcquinox && python -m py_compile xcquinox/alec/cluster/workflow_matrix.py && echo compiled
cd /home/awills/Documents/Research/xcquinox && python -m pytest xcquinox/alec/tests/test_cluster_workflow_matrix.py -v > /tmp/xcq-testlogs/task06_green.log 2>&1; echo "exit=$?"
```
Expected: PASS.

**Covering test command:** `python -m pytest xcquinox/alec/tests/test_cluster_workflow_matrix.py -v > /tmp/xcq-testlogs/task06_green.log 2>&1`

---

## Task 7: `run_matrix` and the report writer

**Files:**
- Modify: `xcquinox/alec/cluster/workflow_matrix.py` (append after `run_arch`)
- Test: `xcquinox/alec/tests/test_cluster_workflow_matrix.py` (append)

**Interfaces:**
- Consumes: `run_arch`, `stage_cached_inputs`, `STAGE_ORDER` (Tasks 4, 6).
- Produces:
  - `workflow_matrix.MAX_SHARDS: int = 4`
  - `workflow_matrix.run_matrix(archs, work_root, *, shards=1, runner=subprocess.run, timeout_s=DEFAULT_STAGE_TIMEOUT_S, repo_root=None, external_refs_dir=None, species_slice=HELDOUT_SPECIES_SLICE, threads=None, run_oracles=True, progress=None) -> list[dict]`
  - `workflow_matrix.arch_row(result) -> dict` with keys `arch`, `stages_rc`, `certificate`, `oracles`, `wall`
  - `workflow_matrix.write_matrix_report(results, path) -> Path` (markdown at `path`, JSON at `path.with_suffix(".json")`)

- [ ] **Step 1: Write the failing tests**

Append to `xcquinox/alec/tests/test_cluster_workflow_matrix.py`:

```python
# ---------------------------------------------------------------------------
# run_matrix + report
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def shared_refs(tmp_path_factory):
    """One real copy of the 74 MB reference cache for the whole module.

    ``stage_cached_inputs`` copies rather than symlinks (the tracked cache
    receives a run log from every preflight), and every ``run_matrix`` test
    would otherwise pay that copy again.
    """
    root = tmp_path_factory.mktemp("shared_inputs")
    return wm.stage_cached_inputs(root, repo_root=wm.repo_root_path())[
        "external_refs_dir"]


class MatrixFakeRunner:
    """Fake runner for several architectures: derives each run directory from
    the grid path the submit stage is handed, so one instance serves the whole
    matrix. Thread-safe enough for the shard test: appends only."""

    def __init__(self, *, rc_by_arch=None, verdict="PASS"):
        self.rc_by_arch = dict(rc_by_arch or {})
        self.verdict = verdict
        self.calls = []

    def __call__(self, argv, **kwargs):
        argv = [str(a) for a in argv]
        self.calls.append((argv, dict(kwargs.get("env") or {})))
        stream = kwargs.get("stdout")
        stage = FakeRunner._stage_of(argv)
        arch = self._arch_of(argv)
        # The tag goes FIRST: run_arch reads the LAST non-empty line of the
        # oracle log as its summary line.
        stream.write(f"[fake] {stage} {arch}\n")
        if stage == "submit":
            run_dir = Path(argv[4]).parent / "runs" / f"run_{arch}"
            run_dir.mkdir(parents=True, exist_ok=True)
            stream.write(f"submit: run dir = {run_dir}\n")
        elif stage == "certificate":
            cert = Path(argv[3]) / "pretrain" / arch / \
                "fidelity_certificate.json"
            cert.parent.mkdir(parents=True, exist_ok=True)
            cert.write_text(json.dumps({"verdict": self.verdict}))
        elif stage == "oracles":
            stream.write("7 passed in 2.0s\n")

        class _Completed:
            returncode = (self.rc_by_arch.get(arch, 0)
                          if stage == "preflight" else 0)

        return _Completed()

    @staticmethod
    def _arch_of(argv):
        for token in argv:
            if token.endswith("grid.yaml"):
                return Path(token).parent.name
            if "/runs/run_" in token:
                return Path(token).name[len("run_"):]
            if token.startswith("test_spin_scaling_oracles and "):
                return token.split(" and ")[1]
        return "?"


def test_run_matrix_returns_one_result_per_arch_in_input_order(tmp_path,
                                                               shared_refs):
    archs = ["shallow", "deep", "medium"]
    results = wm.run_matrix(archs, tmp_path, runner=MatrixFakeRunner(),
                            repo_root=wm.repo_root_path(),
                            external_refs_dir=shared_refs)
    assert [r["arch"] for r in results] == archs
    for r in results:
        assert [s["rc"] for s in r["stages"]] == [0] * len(wm.STAGE_ORDER)
        assert r["certificate_verdict"] == "PASS"


def test_run_matrix_stages_the_reference_copy_once(tmp_path):
    """Default path: one copy under the work root, shared by every arch."""
    import yaml
    wm.run_matrix(["shallow", "deep"], tmp_path, runner=MatrixFakeRunner(),
                  repo_root=wm.repo_root_path())
    shared = tmp_path / "_inputs" / "external_refs"
    assert shared.is_dir()
    for arch in ("shallow", "deep"):
        raw = yaml.safe_load((tmp_path / arch / "grid.yaml").read_text())
        assert raw["inputs"]["external_refs_dir"] == str(shared)


def test_run_matrix_gives_each_shard_its_own_pretrain_data_dir(tmp_path,
                                                               shared_refs):
    """Two shards generating pretrain_data_polarized.npz into one directory
    would race on a fixed filename; within a shard the architectures run
    serially, so the second one's datagen is a skip-if-current no-op."""
    import yaml
    archs = ["shallow", "deep", "medium", "shallow_attn"]
    results = wm.run_matrix(archs, tmp_path, shards=2,
                            runner=MatrixFakeRunner(),
                            repo_root=wm.repo_root_path(),
                            external_refs_dir=shared_refs)
    assert [r["arch"] for r in results] == archs
    dirs = {}
    for arch in archs:
        raw = yaml.safe_load((tmp_path / arch / "grid.yaml").read_text())
        dirs[arch] = raw["pretrain"]["data_dir"]
    assert dirs["shallow"] == dirs["medium"]        # shard 0: archs[0::2]
    assert dirs["deep"] == dirs["shallow_attn"]     # shard 1: archs[1::2]
    assert dirs["shallow"] != dirs["deep"]
    assert dirs["shallow"].endswith("pretrain_data_shard0")
    assert dirs["deep"].endswith("pretrain_data_shard1")


def test_run_matrix_refuses_an_out_of_range_shard_count(tmp_path, shared_refs):
    for bad in (0, -1, wm.MAX_SHARDS + 1):
        with pytest.raises(ValueError, match="shards"):
            wm.run_matrix(["deep"], tmp_path, shards=bad,
                          runner=MatrixFakeRunner(),
                          repo_root=wm.repo_root_path(),
                          external_refs_dir=shared_refs)


def test_run_matrix_refuses_an_unregistered_architecture(tmp_path,
                                                         shared_refs):
    with pytest.raises(ValueError, match="no_such_arch"):
        wm.run_matrix(["deep", "no_such_arch"], tmp_path,
                      runner=MatrixFakeRunner(),
                      repo_root=wm.repo_root_path(),
                      external_refs_dir=shared_refs)


def test_run_matrix_calls_the_progress_hook_once_per_arch(tmp_path,
                                                          shared_refs):
    seen = []
    wm.run_matrix(["deep", "shallow"], tmp_path, runner=MatrixFakeRunner(),
                  repo_root=wm.repo_root_path(),
                  external_refs_dir=shared_refs, progress=seen.append)
    assert sorted(r["arch"] for r in seen) == ["deep", "shallow"]


def test_arch_row_renders_a_complete_run():
    result = {
        "arch": "deep", "seconds": 702.0, "certificate_verdict": "PASS",
        "stages": [{"name": n, "rc": 0} for n in wm.STAGE_ORDER],
        "oracle_tests": {"rc": 0, "summary_line": "12 passed in 3.4s"},
    }
    row = wm.arch_row(result)
    assert row["arch"] == "deep"
    assert row["stages_rc"] == ".".join(["0"] * len(wm.STAGE_ORDER))
    assert row["certificate"] == "PASS"
    assert row["oracles"] == "0 (12 passed in 3.4s)"
    assert row["wall"] == "11m42s"


def test_arch_row_marks_the_stages_a_failure_never_reached():
    result = {
        "arch": "deep_dm", "seconds": 61.0, "certificate_verdict": "FAIL",
        "stages": [{"name": "submit", "rc": 0}, {"name": "datagen", "rc": 0},
                   {"name": "pretrain", "rc": 1}],
        "oracle_tests": {"rc": 1, "summary_line": "1 failed, 11 passed"},
    }
    row = wm.arch_row(result)
    assert row["stages_rc"] == "0.0.1.-.-.-.-.-.-.-"
    assert row["certificate"] == "FAIL"
    assert row["oracles"] == "1 (1 failed, 11 passed)"
    assert row["wall"] == "1m01s"


def test_arch_row_renders_skipped_oracles():
    result = {"arch": "deep", "seconds": 0.0, "certificate_verdict": None,
              "stages": [], "oracle_tests": {"rc": None, "summary_line": ""}}
    row = wm.arch_row(result)
    assert row["oracles"] == "skipped"
    assert row["certificate"] == "-"


def test_write_matrix_report_writes_markdown_and_json(tmp_path):
    results = [
        {"arch": "deep", "seconds": 702.0, "certificate_verdict": "PASS",
         "run_dir": "/w/deep/runs/run_x",
         "stages": [{"name": n, "rc": 0} for n in wm.STAGE_ORDER],
         "artefacts": {"manifest": {"path": "/w/m.json", "exists": True}},
         "oracle_tests": {"rc": 0, "summary_line": "12 passed in 3.4s"}},
        {"arch": "deep_dm", "seconds": 61.0, "certificate_verdict": "FAIL",
         "run_dir": "/w/deep_dm/runs/run_y",
         "stages": [{"name": "submit", "rc": 0},
                    {"name": "datagen", "rc": 2}],
         "artefacts": {"manifest": {"path": "/w/n.json", "exists": False}},
         "oracle_tests": {"rc": 0, "summary_line": "12 passed in 3.1s"}},
    ]
    path = wm.write_matrix_report(results, tmp_path / "matrix.md")
    text = path.read_text()
    assert "| arch | stages rc | certificate | oracles | wall |" in text
    assert "| deep | 0.0.0.0.0.0.0.0.0.0 | PASS |" in text
    assert "| deep_dm | 0.2.-.-.-.-.-.-.-.- | FAIL |" in text
    assert ", ".join(wm.STAGE_ORDER) in text
    assert "1 of 2" in text
    sidecar = json.loads((tmp_path / "matrix.json").read_text())
    assert [r["arch"] for r in sidecar["results"]] == ["deep", "deep_dm"]
    assert sidecar["species_slice"] == wm.HELDOUT_SPECIES_SLICE
    assert sidecar["stage_order"] == list(wm.STAGE_ORDER)
```

- [ ] **Step 2: Run the tests and confirm they fail**

```bash
python -m pytest xcquinox/alec/tests/test_cluster_workflow_matrix.py -v > /tmp/xcq-testlogs/task07_red.log 2>&1; echo "exit=$?"
```
Expected: `AttributeError: module 'xcquinox.alec.cluster.workflow_matrix' has no attribute 'run_matrix'`. Read the log with `Read`.

- [ ] **Step 3: Add `run_matrix`, `arch_row` and `write_matrix_report`**

Append to `xcquinox/alec/cluster/workflow_matrix.py`:

```python
# ---------------------------------------------------------------------------
# The matrix
# ---------------------------------------------------------------------------

#: Concurrency ceiling. Each shard runs one SCF-heavy stage subprocess at a
#: time; beyond four on a 20-core box the stages contend for memory bandwidth
#: and the per-architecture wall stops being comparable across shards.
MAX_SHARDS = 4


def run_matrix(archs, work_root, *, shards=1, runner=subprocess.run,
               timeout_s=DEFAULT_STAGE_TIMEOUT_S, repo_root=None,
               external_refs_dir=None,
               species_slice=HELDOUT_SPECIES_SLICE, threads=None,
               run_oracles=True, progress=None) -> list:
    """Run the stage sequence for every architecture in ``archs``.

    Architectures are dealt round-robin into ``shards`` groups, each group run
    serially by one thread; the threads only wait on subprocesses, so the work
    is in the stages, not in this process. Every shard gets its OWN
    pretrain-data directory because the generator writes a fixed filename and
    two concurrent datagen stages would race on it; inside a shard the second
    architecture's datagen is a skip-if-current no-op.

    The reference copy is staged ONCE here, before any thread starts, so the
    copy itself cannot race; ``external_refs_dir`` re-uses an already staged
    copy instead. ``progress`` is called with each finished result and may be
    called from several threads.
    """
    archs = list(archs)
    if not archs:
        raise ValueError("run_matrix: no architectures given")
    unknown = [a for a in archs if a not in ARCHITECTURES]
    if unknown:
        raise ValueError(
            f"run_matrix: {unknown} are not registered architectures; "
            f"valid names: {sorted(ARCHITECTURES)}"
        )
    shards = int(shards)
    if not 1 <= shards <= MAX_SHARDS:
        raise ValueError(
            f"run_matrix: shards must satisfy 1 <= shards <= {MAX_SHARDS}, "
            f"got {shards}. Each shard runs one SCF-heavy stage at a time."
        )
    root = Path(repo_root) if repo_root is not None else repo_root_path()
    work_root = Path(work_root).resolve()
    work_root.mkdir(parents=True, exist_ok=True)
    refs = (str(Path(external_refs_dir).resolve()) if external_refs_dir
            else stage_cached_inputs(work_root,
                                     repo_root=root)["external_refs_dir"])
    if threads is None:
        threads = max(1, (os.cpu_count() or 4) // shards)
    groups = [archs[k::shards] for k in range(shards)]
    data_dirs = [work_root / "_inputs" / f"pretrain_data_shard{k}"
                 for k in range(shards)]

    def _run_group(k):
        out = []
        for arch in groups[k]:
            out.append(run_arch(
                arch, work_root, runner=runner, timeout_s=timeout_s,
                repo_root=root, external_refs_dir=refs,
                pretrain_data_dir=data_dirs[k],
                species_slice=species_slice, threads=threads,
                run_oracles=run_oracles))
            if progress is not None:
                progress(out[-1])
        return out

    if shards == 1:
        collected = _run_group(0)
    else:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=shards) as pool:
            collected = [record
                         for group in pool.map(_run_group, range(shards))
                         for record in group]
    order = {name: i for i, name in enumerate(archs)}
    return sorted(collected, key=lambda record: order[record["arch"]])


def _fmt_wall(seconds):
    """Compact wall-clock: ``11m42s`` under an hour, ``1h02m`` above."""
    total = int(round(float(seconds)))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m{secs:02d}s"


def arch_row(result) -> dict:
    """One report row: architecture, stage return codes, certificate, oracles,
    wall.

    ``stages_rc`` is one field per entry of :data:`STAGE_ORDER`, ``-`` for a
    stage the sequence never reached, so the column is a fixed-width
    fingerprint of the run and two matrices diff line by line.
    """
    by_name = {s["name"]: s["rc"] for s in result.get("stages", ())}
    stages_rc = ".".join(
        "-" if name not in by_name else str(by_name[name])
        for name in STAGE_ORDER)
    oracles = result.get("oracle_tests") or {}
    if oracles.get("rc") is None:
        oracle_cell = "skipped"
    else:
        oracle_cell = f"{oracles['rc']} ({oracles.get('summary_line', '')})"
    return {
        "arch": result["arch"],
        "stages_rc": stages_rc,
        "certificate": result.get("certificate_verdict") or "-",
        "oracles": oracle_cell,
        "wall": _fmt_wall(result.get("seconds", 0.0)),
    }


def _is_clean(result) -> bool:
    """True iff every stage ran and exited zero and the oracles passed.

    The certificate is exempt from the exit-code test: spec 3.4 records its
    verdict, it does not require a PASS from a 50-step pretrain.
    """
    if len(result.get("stages", ())) != len(STAGE_ORDER):
        return False
    for stage in result["stages"]:
        if stage["name"] != "certificate" and stage["rc"] != 0:
            return False
    rc = (result.get("oracle_tests") or {}).get("rc")
    return rc in (0, None)


def write_matrix_report(results, path) -> Path:
    """Write the matrix table as markdown, and the full records as JSON.

    The markdown table is the HISTORY baseline entry (columns: architecture,
    stage return codes, certificate verdict, oracle result, wall). The JSON
    sidecar beside it keeps every stage log path and artefact record, which the
    table cannot hold and a later comparison needs.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    results = list(results)
    rows = [arch_row(r) for r in results]
    clean = sum(1 for r in results if _is_clean(r))
    lines = [
        "# Per-architecture workflow matrix",
        "",
        "Identity: def2-svp, grid level 1, solver oneshot, 2 cells "
        "(subset sizes 1 and 2), 3 training steps, 50 pretraining steps on "
        "H and O; certificate at the production identity; held-out eval on "
        f"the species slice {HELDOUT_SPECIES_SLICE}.",
        "",
        "Stage order of the `stages rc` column (`-` = never reached; the "
        "certificate's non-zero exit does not stop the sequence, its verdict "
        "is recorded): " + ", ".join(STAGE_ORDER) + ".",
        "",
        f"{clean} of {len(results)} architectures completed every stage with "
        "exit 0 and passing oracles.",
        "",
        "| arch | stages rc | certificate | oracles | wall |",
        "|---|---|---|---|---|",
    ]
    lines += [f"| {r['arch']} | {r['stages_rc']} | {r['certificate']} | "
              f"{r['oracles']} | {r['wall']} |" for r in rows]
    lines.append("")
    path.write_text("\n".join(lines))
    sidecar = path.with_suffix(".json")
    with sidecar.open("w") as fh:
        json.dump({"stage_order": list(STAGE_ORDER),
                   "species_slice": HELDOUT_SPECIES_SLICE,
                   "fidelity_override_reason": FIDELITY_OVERRIDE_REASON,
                   "n_clean": clean,
                   "results": results}, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return path
```

- [ ] **Step 4: Compile and run the tests**

```bash
python -m py_compile xcquinox/alec/cluster/workflow_matrix.py && echo compiled
cd /home/awills/Documents/Research/xcquinox && python -m pytest xcquinox/alec/tests/test_cluster_workflow_matrix.py -v > /tmp/xcq-testlogs/task07_green.log 2>&1; echo "exit=$?"
```
Expected: PASS.

**Covering test command:** `python -m pytest xcquinox/alec/tests/test_cluster_workflow_matrix.py -v > /tmp/xcq-testlogs/task07_green.log 2>&1`

---

## Task 8: The command-line entry point

**Files:**
- Modify: `xcquinox/alec/cluster/workflow_matrix.py` (append after `write_matrix_report`)
- Test: `xcquinox/alec/tests/test_cluster_workflow_matrix.py` (append)

**Interfaces:**
- Consumes: `run_matrix`, `write_matrix_report`, `arch_row`, `_is_clean` (Task 7).
- Produces: `workflow_matrix.main(argv=None, *, runner=subprocess.run) -> int`, CLI `python -m xcquinox.alec.cluster.workflow_matrix --archs all|a,b,c --work-root DIR [--shards N] [--report PATH] [--timeout-s S] [--species-slice CSV] [--external-refs-dir DIR] [--no-oracles]`.

- [ ] **Step 1: Write the failing tests**

Append to `xcquinox/alec/tests/test_cluster_workflow_matrix.py`:

```python
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_main_runs_the_named_architectures_and_writes_the_report(
        tmp_path, shared_refs, capsys):
    rc = wm.main(["--archs", "deep,shallow", "--work-root", str(tmp_path),
                  "--external-refs-dir", str(shared_refs),
                  "--report", str(tmp_path / "matrix.md")],
                 runner=MatrixFakeRunner())
    assert rc == 0
    text = (tmp_path / "matrix.md").read_text()
    assert "| deep |" in text and "| shallow |" in text
    assert (tmp_path / "matrix.json").is_file()
    out = capsys.readouterr().out
    assert "deep" in out and "shallow" in out


def test_main_defaults_the_report_into_the_work_root(tmp_path, shared_refs):
    assert wm.main(["--archs", "deep", "--work-root", str(tmp_path),
                    "--external-refs-dir", str(shared_refs)],
                   runner=MatrixFakeRunner()) == 0
    assert (tmp_path / "workflow_matrix.md").is_file()
    assert (tmp_path / "workflow_matrix.json").is_file()


def test_main_all_selects_every_registered_architecture(tmp_path,
                                                        shared_refs):
    wm.main(["--archs", "all", "--work-root", str(tmp_path),
             "--external-refs-dir", str(shared_refs), "--no-oracles"],
            runner=MatrixFakeRunner())
    sidecar = json.loads((tmp_path / "workflow_matrix.json").read_text())
    assert [r["arch"] for r in sidecar["results"]] == sorted(ARCHITECTURES)


def test_main_returns_non_zero_when_a_stage_failed(tmp_path, shared_refs):
    rc = wm.main(["--archs", "deep,shallow", "--work-root", str(tmp_path),
                  "--external-refs-dir", str(shared_refs)],
                 runner=MatrixFakeRunner(rc_by_arch={"shallow": 1}))
    assert rc == 1
    text = (tmp_path / "workflow_matrix.md").read_text()
    assert "| shallow | 0.0.0.0.1.-.-.-.-.- |" in text


def test_main_refuses_a_work_root_inside_the_repository():
    inside = wm.repo_root_path() / "notebooks" / "matrix_scratch"
    with pytest.raises(ValueError, match="inside the repository"):
        wm.main(["--archs", "deep", "--work-root", str(inside)],
                runner=MatrixFakeRunner())


def test_main_refuses_an_unknown_architecture(tmp_path):
    with pytest.raises(ValueError, match="no_such_arch"):
        wm.main(["--archs", "no_such_arch", "--work-root", str(tmp_path)],
                runner=MatrixFakeRunner())


def test_main_passes_the_slice_and_the_oracle_switch_through(tmp_path,
                                                             shared_refs):
    from xcquinox.alec.full_benchmark_pools import HELDOUT_SPECIES_SLICE_ENV
    fake = MatrixFakeRunner()
    wm.main(["--archs", "deep", "--work-root", str(tmp_path),
             "--external-refs-dir", str(shared_refs),
             "--species-slice", "h,h2", "--no-oracles"], runner=fake)
    eval_envs = [env for argv, env in fake.calls
                 if "_eval_one_spec" in " ".join(argv)]
    assert eval_envs and all(env[HELDOUT_SPECIES_SLICE_ENV] == "h,h2"
                             for env in eval_envs)
    assert not any("-m pytest" in " ".join(argv) for argv, _e in fake.calls)
```

- [ ] **Step 2: Run the tests and confirm they fail**

```bash
cd /home/awills/Documents/Research/xcquinox && python -m pytest xcquinox/alec/tests/test_cluster_workflow_matrix.py -v > /tmp/xcq-testlogs/task08_red.log 2>&1; echo "exit=$?"
```
Expected: `AttributeError: module 'xcquinox.alec.cluster.workflow_matrix' has no attribute 'main'`. Read the log with `Read`.

- [ ] **Step 3: Add the CLI**

Append to `xcquinox/alec/cluster/workflow_matrix.py`:

```python
# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _resolve_archs(spec):
    """``all`` or a comma-separated list -> a validated architecture list."""
    if spec.strip() == "all":
        return sorted(ARCHITECTURES)
    names = [part.strip() for part in spec.split(",") if part.strip()]
    if not names:
        raise ValueError("--archs names no architecture")
    unknown = [n for n in names if n not in ARCHITECTURES]
    if unknown:
        raise ValueError(
            f"--archs names {unknown}, which are not registered "
            f"architectures; valid names: {sorted(ARCHITECTURES)}"
        )
    return names


def _refuse_repo_work_root(work_root, repo_root):
    """Keep every byte the matrix writes out of the tracked tree.

    The stages write run directories, a copy of the CCSD reference cache and
    pretrain data; the reference copy in particular receives a
    ``_run_log_<UTC>.json`` from every preflight.
    """
    work = Path(work_root).resolve()
    repo = Path(repo_root).resolve()
    if work == repo or repo in work.parents:
        raise ValueError(
            f"--work-root {work} is inside the repository {repo}; the matrix "
            "writes run directories, a copy of the CCSD reference cache and "
            "pretrain data, none of which belong in the tree. Use a path "
            "outside it (e.g. /tmp/xcq-workflow-matrix)."
        )


def main(argv=None, *, runner=subprocess.run) -> int:
    """Run the matrix and write its report. Returns a process exit code.

    Zero iff every architecture completed every stage with exit 0 and its
    oracles passed. The certificate's verdict does not enter the exit code: it
    is recorded per architecture (SPEC_pretrain_fidelity_program.md 3.4).
    """
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m xcquinox.alec.cluster.workflow_matrix",
        description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--archs", default="all",
        help="'all' (every registered architecture) or a comma-separated list")
    parser.add_argument(
        "--work-root", required=True,
        help="directory for run dirs, logs and staged inputs; must be outside "
             "the repository")
    parser.add_argument(
        "--shards", type=int, default=1,
        help=f"architectures run concurrently (1..{MAX_SHARDS}); each shard "
             "runs one SCF-heavy stage at a time")
    parser.add_argument(
        "--report", default=None,
        help="markdown report path (default <work-root>/workflow_matrix.md; "
             "the JSON sidecar sits beside it)")
    parser.add_argument(
        "--timeout-s", type=int, default=DEFAULT_STAGE_TIMEOUT_S,
        help="per-stage wall-clock cap in seconds")
    parser.add_argument(
        "--species-slice", default=HELDOUT_SPECIES_SLICE,
        help="comma-separated held-out species slice for the eval stages")
    parser.add_argument(
        "--external-refs-dir", default=None,
        help="an already staged copy of the CCSD reference cache (default: "
             "copy the repository cache into <work-root>/_inputs/external_refs)")
    parser.add_argument(
        "--no-oracles", action="store_true",
        help="skip the per-architecture spin-scaling oracle run")
    args = parser.parse_args(argv)

    root = repo_root_path()
    archs = _resolve_archs(args.archs)
    _refuse_repo_work_root(args.work_root, root)
    work_root = Path(args.work_root).resolve()
    report_path = Path(args.report) if args.report \
        else work_root / "workflow_matrix.md"

    print(f"[workflow-matrix] {len(archs)} architectures, "
          f"{args.shards} shard(s), work root {work_root}", flush=True)
    done = [0]

    def _progress(result):
        done[0] += 1
        row = arch_row(result)
        print(f"[workflow-matrix] {done[0]}/{len(archs)} {row['arch']}: "
              f"stages {row['stages_rc']} certificate {row['certificate']} "
              f"oracles {row['oracles']} wall {row['wall']}", flush=True)

    results = run_matrix(
        archs, work_root, shards=args.shards, runner=runner,
        timeout_s=args.timeout_s, repo_root=root,
        external_refs_dir=args.external_refs_dir,
        species_slice=args.species_slice, run_oracles=not args.no_oracles,
        progress=_progress)
    write_matrix_report(results, report_path)
    clean = sum(1 for r in results if _is_clean(r))
    print(f"[workflow-matrix] {clean}/{len(results)} clean; report "
          f"{report_path}", flush=True)
    return 0 if clean == len(results) else 1


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess
    sys.exit(main())
```

- [ ] **Step 4: Compile and run the tests**

```bash
cd /home/awills/Documents/Research/xcquinox && python -m py_compile xcquinox/alec/cluster/workflow_matrix.py && echo compiled
cd /home/awills/Documents/Research/xcquinox && python -m pytest xcquinox/alec/tests/test_cluster_workflow_matrix.py -v > /tmp/xcq-testlogs/task08_green.log 2>&1; echo "exit=$?"
```
Expected: PASS.

- [ ] **Step 5: Check the CLI help renders**

```bash
cd /home/awills/Documents/Research/xcquinox && python -m xcquinox.alec.cluster.workflow_matrix --help > /tmp/xcq-testlogs/task08_help.log 2>&1; echo "exit=$?"
```
Expected: exit 0 and the option list. Read the log with `Read`.

**Covering test command:** `python -m pytest xcquinox/alec/tests/test_cluster_workflow_matrix.py -v > /tmp/xcq-testlogs/task08_green.log 2>&1`

---

## Task 9: The real single-architecture smoke

**Files:**
- Test: `xcquinox/alec/tests/test_cluster_workflow_matrix.py` (append)

**Interfaces:**
- Consumes: `run_arch` (Task 6) with the DEFAULT runner (`subprocess.run`), the tiny template (Task 4), the slice mechanism (Tasks 1-2).
- Produces: nothing importable; this is the one test that proves the fake-runner tests describe the real stages.

- [ ] **Step 1: Write the smoke**

Append to `xcquinox/alec/tests/test_cluster_workflow_matrix.py`:

```python
# ---------------------------------------------------------------------------
# The one real pass: every stage as a real subprocess, cheapest architecture
# ---------------------------------------------------------------------------

_SMOKE_REQUIRED_ARTEFACTS = (
    "resolved_config", "script_datagen", "script_pretrain", "script_preflight",
    "script_train", "script_eval", "pretrain_data", "pretrain_xnet",
    "pretrain_cnet", "pretrain_metadata", "manifest",
    "spec[0]", "spec[1]", "model[0]", "model[1]",
    "eval_df[0]", "eval_df[1]",
    "holdout_test_set[0]", "holdout_test_set[1]",
    "holdout_metadata[0]", "holdout_metadata[1]",
    "holdout_sliced[0]", "holdout_sliced[1]",
)


@pytest.mark.slow
def test_workflow_matrix_smoke_runs_every_stage_for_shallow(tmp_path):
    """One architecture, every stage as the subprocess SLURM would run.

    ``shallow`` is the cheapest registry entry (depth 2, 8 nodes, no
    descriptors). Expect 10-15 minutes: datagen ~1 min, pretrain ~1 min,
    preflight ~2 min against the cached CCSD references, 2 training cells of
    3 steps, 2 sliced held-out evaluations of 6 species. The oracles are not
    run here -- they are covered per architecture by the matrix itself and by
    the selector contract test; running them would double the wall for no new
    information about the stage wiring.
    """
    result = wm.run_arch("shallow", tmp_path, run_oracles=False,
                         timeout_s=1800)
    ran = [s["name"] for s in result["stages"]]
    assert ran == list(wm.STAGE_ORDER), (
        f"stopped at {ran[-1]}: see {result['stages'][-1]['log']}")
    for stage in result["stages"]:
        if stage["name"] == "certificate":
            continue  # verdict recorded, exit code not required (spec 3.4)
        assert stage["rc"] == 0, (stage["name"], stage["log"])

    missing = [label for label in _SMOKE_REQUIRED_ARTEFACTS
               if not result["artefacts"][label]["exists"]]
    assert not missing, missing

    # The certificate module is spec 3.3's; when it is installed the matrix
    # must record its verdict.
    cert_installed = (wm.repo_root_path() / "xcquinox" / "alec" / "cluster"
                      / "fidelity.py").is_file()
    if cert_installed:
        assert result["certificate_verdict"] in ("PASS", "FAIL")

    # The held-out channel is the six-species slice, marked twice.
    for idx in (0, 1):
        with open(result["artefacts"][f"holdout_metadata[{idx}]"]["path"]) as f:
            stamp = json.load(f)
        assert stamp["species_slice"] == wm.HELDOUT_SPECIES_SLICE.split(",")
        assert stamp["n_species"] == 6
        assert stamp["n_reactions"] == 3
        with open(result["artefacts"][f"holdout_sliced[{idx}]"]["path"]) as f:
            assert json.load(f)["n_species"] == 6
        rows = Path(result["artefacts"][f"eval_df[{idx}]"]["path"]).read_text()
        assert len(rows.strip().splitlines()) >= 2, rows
```

- [ ] **Step 2: Run the smoke**

```bash
mkdir -p /tmp/xcq-testlogs
python -m pytest xcquinox/alec/tests/test_cluster_workflow_matrix.py -m slow -v > /tmp/xcq-testlogs/task09_smoke.log 2>&1; echo "exit=$?"
```
Expected: PASS in 10-15 minutes (the file's other tests are deselected by `-m slow`). Read the whole log with `Read`.

If a stage exits non-zero, the assertion message names its log; read THAT log with `Read` and fix the cause. Do not weaken the assertion: a stage that cannot run at this identity is exactly the finding the matrix exists to produce. Two failures are expected to be common and are NOT test bugs:
- `certificate`, when `xcquinox/alec/cluster/fidelity.py` is not installed yet (spec 3.3). The stage exits non-zero, `allow_nonzero` keeps the sequence going, and `certificate_verdict` stays `None`.
- `pretrain` or `train[0]`, if spec 3.3's enforcement does not honour `XCQUINOX_FIDELITY_OVERRIDE_REASON`. Read the stage log; if it names the fidelity gate, that is the interface disagreement recorded in the header of this plan -- settle the knob's name with the 3.3 implementation and change `FIDELITY_OVERRIDE_ENV`.

- [ ] **Step 3: Record the measured wall**

Read the per-stage seconds out of the smoke result (they are in the log's assertion output only on failure, so re-derive them from the stage logs' mtimes if needed) and note the total. The number goes into the HISTORY entry (Task 10) as the measured per-architecture wall at this identity; do not quote the plan's 10-15 minute expectation as a measurement.

- [ ] **Step 4: Confirm the default selection still excludes the smoke**

```bash
cd /home/awills/Documents/Research/xcquinox && python -m pytest xcquinox/alec/tests/test_cluster_workflow_matrix.py -v > /tmp/xcq-testlogs/task09_default.log 2>&1; echo "exit=$?"
```
Expected: PASS with the smoke reported as deselected (`setup.cfg` addopts `-m "not slow"`), and every fake-runner test green.

**Covering test command:** `python -m pytest xcquinox/alec/tests/test_cluster_workflow_matrix.py -m slow -v > /tmp/xcq-testlogs/task09_smoke.log 2>&1`

---

## Task 10: Run the matrix and record the HISTORY baseline

**Files:**
- Modify: `xcquinox/alec/HISTORY.md` (append a new phase section at the end of the dated sections, before `## Open / in-progress`)

**Interfaces:**
- Consumes: the CLI (Task 8), the report writer (Task 7).
- Produces: the baseline matrix table in HISTORY, and `/tmp/xcq-workflow-matrix/workflow_matrix.{md,json}` as its provenance.

- [ ] **Step 1: Confirm the whole affected test surface is green first**

```bash
cd /home/awills/Documents/Research/xcquinox && python -m pytest xcquinox/alec/tests/test_cluster_workflow_matrix.py xcquinox/alec/tests/test_full_benchmark_pools.py xcquinox/alec/tests/test_cluster_eval_worker.py xcquinox/alec/tests/test_eval_holdout.py notebooks/analysis/test_make_ablation_arch_figure.py -v > /tmp/xcq-testlogs/task10_pre.log 2>&1; echo "exit=$?"
```
Expected: PASS. Read the log with `Read`.

- [ ] **Step 2: Run the matrix over every architecture**

Start it in the BACKGROUND (5-7 hours serially, 2-3 hours at three shards) and poll the log; a foreground call will hit the tool's command timeout.

```bash
mkdir -p /tmp/xcq-workflow-matrix /tmp/xcq-testlogs
cd /home/awills/Documents/Research/xcquinox && python -m xcquinox.alec.cluster.workflow_matrix --archs all --work-root /tmp/xcq-workflow-matrix --shards 3 --report /tmp/xcq-workflow-matrix/workflow_matrix.md > /tmp/xcq-testlogs/task10_matrix.log 2>&1; echo "exit=$?"
```

Three shards, not four: each shard's stage subprocess is SCF-heavy and the box has 20 cores, so three leaves headroom for the nested worker subprocess `_train_task` spawns. Poll with

```bash
grep -c "^\[workflow-matrix\] " /tmp/xcq-testlogs/task10_matrix.log
```

and read the completed log with `Read` when the process exits.

- [ ] **Step 3: Read the report**

```bash
cat /tmp/xcq-workflow-matrix/workflow_matrix.md > /tmp/xcq-testlogs/task10_report.log 2>&1; echo "exit=$?"
```
Read `/tmp/xcq-testlogs/task10_report.log` with `Read`. For every row that is not all-zero, read the named stage log with `Read` and record the cause in one clause; a workflow matrix whose failures are unexplained is not a baseline.

- [ ] **Step 4: Append the HISTORY entry**

Append to `xcquinox/alec/HISTORY.md`, immediately before the `## Open / in-progress (as of 2026-06-20)` section, with the table pasted VERBATIM from the report (architecture count, return codes, verdicts, oracle results and walls all as measured -- no number in this entry may come from the plan):

```markdown
## Phase 40 -- Per-architecture workflow matrix: the harness stage sequence exercised for every architecture (2026-08-21)

- 2026-08-21 -- **Every registered architecture driven through the whole harness stage sequence at a tiny identity, as the baseline every later change is measured against.** `xcquinox/alec/cluster/workflow_matrix.py` renders a one-architecture grid config from `cluster/examples/workflow_matrix_template.yaml` (def2-svp, grid level 1, solver oneshot, two cells at subset sizes 1 and 2 of the cached JSD ledger, 3 training steps, 50 pretraining steps on H and O) and runs, as the same `python -m` subprocesses SLURM would run, `submit` in its default dry-run, `_datagen`, `_pretrain`, the fidelity certificate, `_preflight`, `_train_task` on both cells, `_eval_one_spec` on both cells and `validate_run`, then the architecture's slice of the spin-scaling oracles, stopping at the first non-zero exit. The cached step-7 CCSD references are COPIED into the work root rather than read in place, because `external_refs.precompute_all` writes a run log into its cache directory on every call and that directory is tracked. The held-out evaluation, hours per cell over the 216-reaction BH76 + W4-11 pool and not narrowable from the grid config, runs on a named six-species slice (`XCQUINOX_HELDOUT_SPECIES_SLICE=h,h2,o,oh,n2o,n2ohts`), which closes three reactions -- one BH76 barrier and two W4-11 atomizations -- across both spin types; the slice is honoured only by `_eval_one_spec._run_held_out_eval` (never by the pool loaders, which the training pool also goes through), a sliced channel is marked before the energies in `sliced_eval.json` and after them in `eval_metadata.json`, and the figure loaders refuse either mark. The certificate's verdict is RECORDED rather than required: a 50-step pretrain on two atoms cannot meet tol_AE = 1.0 kcal/mol, so the matrix carries `XCQUINOX_FIDELITY_OVERRIDE_REASON` and reports the verdict per architecture.

Stage order of the `stages rc` column (`-` = never reached): submit, datagen, pretrain, certificate, preflight, train[0], train[1], eval[0], eval[1], validate_run.

| arch | stages rc | certificate | oracles | wall |
|---|---|---|---|---|
| ... paste every row of /tmp/xcq-workflow-matrix/workflow_matrix.md ... |

  **Why:** the campaign practice this program replaces submitted architectures whose stage wiring had never been exercised end to end at any identity, so a defect in a stage's inputs surfaced as a dead SLURM array hours after submission and after the cluster had been paid for it; running the identical stage modules at a def2-svp identity that fits in minutes converts that class of failure into a table, and recording the table makes the next change's matrix a diff rather than an opinion. The certificate's verdict travels in the same table because the workflow question ("does the stage run?") and the physics question ("does the pretrained network reproduce its parent?") have to stay separable -- Section 3.4 asserts the first and records the second.
```

- [ ] **Step 5: Confirm HISTORY renders and nothing else moved**

```bash
cd /home/awills/Documents/Research/xcquinox && python - <<'PY' > /tmp/xcq-testlogs/task10_history.log 2>&1
from pathlib import Path
text = Path("xcquinox/alec/HISTORY.md").read_text()
rows = [ln for ln in text.splitlines()
        if ln.startswith("| ") and ln.count("|") == 6]
print(f"table rows: {len(rows)}")
print(f"phase 40 present: {'Phase 40' in text}")
print(f"open section still last: "
      f"{text.rindex('## Open / in-progress') > text.rindex('Phase 40')}")
PY
echo "exit=$?"
```
Expected: the row count equals the architecture count plus the header/separator rows, `Phase 40 present: True`, `open section still last: True`. Read the log with `Read`.

**Covering test command:** `python -m pytest xcquinox/alec/tests/test_cluster_workflow_matrix.py xcquinox/alec/tests/test_full_benchmark_pools.py xcquinox/alec/tests/test_cluster_eval_worker.py notebooks/analysis/test_make_ablation_arch_figure.py -v > /tmp/xcq-testlogs/task10_post.log 2>&1`

---

## Notes for the executor

- **Task order.** Tasks 1-3 (the slice mechanism) must land before Task 9 and Task 10, because the matrix's eval stage depends on them; Tasks 4-8 are independent of 1-3 and can be done first if the certificate module (spec 3.3) has not landed. Task 9 depends on everything before it. Task 10 is last.
- **What "every stage exits zero" does not cover.** Two paths of the production configuration are deliberately outside this matrix, and neither is one of Section 3.4's assertions: `seed_xc` stays `pbe` (the `scan` seed needs a populated `seed_cache_dir`, which does not exist locally), and `orientation_lock_strength` stays `0.0` (a non-zero value invalidates the cached CCSD references and would trigger hours of recomputation). Say so in the HISTORY entry if either becomes a campaign default.
- **Timing expectations, not measurements.** Per architecture: datagen ~1 min on the first architecture of a shard and a skip-if-current no-op afterwards, pretrain ~1 min, certificate ~10 s, preflight ~2 min warm, two training cells 2-6 min, two sliced evaluations 2-3 min; 30-odd architectures serially 5-7 h, about a third of that at three shards. Every one of these is an expectation to be replaced by the measured value in the report.
