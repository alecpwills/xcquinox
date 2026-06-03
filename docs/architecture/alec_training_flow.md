# `xcquinox.alec` training pipeline — code-flow diagrams

Flowcharts of the end-to-end machine-learned-XC training workflow: the SLURM **cluster
harness** (`xcquinox/alec/cluster/`) that orchestrates the job graph, and the **core compute**
(`xcquinox/alec/`) that runs inside each job — model build, the training-loop family, the
differentiable SCF, the loss channels, pretraining, and evaluation.

Each diagram appears twice: an inline **Mermaid** version (renders on GitHub / VS Code /
JupyterLab) and a link to a higher-resolution **Graphviz**-rendered image under
[`diagrams/`](diagrams/). Node labels are `file.py::function`, so the diagrams double as a
code index.

> **Regenerate the images** after editing a `.dot`:
> ```bash
> cd docs/architecture/diagrams
> for f in *.dot; do dot -Tsvg "$f" -o "${f%.dot}.svg"; dot -Tpng -Gdpi=150 "$f" -o "${f%.dot}.png"; done
> ```
> (Mermaid blocks have no build step — they render in any Mermaid-aware viewer.)

---

## 1 · Cluster harness — the 4-stage SLURM job graph

`python -m xcquinox.alec.cluster submit config.yaml` → `__main__.py::cmd_submit` →
`submit.py::submit_jobs`, which creates a fresh `run_<UTC>Z/` directory and submits five
`sbatch --parsable` jobs wired by SLURM dependencies:

- **datagen** (single, no dependency) → `pretrain_data[_polarized].npz` (the per-grid-point
  Fx/Fc enhancement targets every swept architecture needs).
- **pretrain** `[0..A-1]` (`afterok:datagen`) — one task per architecture → `pretrain/<arch>/{xnet,cnet}.eqx`.
- **preflight** (`afterok:pretrain`) — materializes one `TrainingSpec` per grid cell, ensures
  CCSD/external references, and writes `manifest.json` as its **atomic, final** step.
- **train** `[0..N-1]` (`afterok:pretrain:preflight` — gated on **both**) — one task per spec.
- **eval** `[0..N-1]` — `aftercorr:train` in the default mode (eval`[i]` after train`[i]`), or
  folded into the train task (`inline_eval`), or submitted by a deferred `eval_launcher`
  (`afterany:train`).

Outcomes are read back with `status` / `results` / `pull` and recovered with
`resubmit` / `resubmit-preflight` / `repair-manifest`.

```mermaid
flowchart TB
  cli(["$ python -m xcquinox.alec.cluster submit config.yaml"]) --> submit["__main__::cmd_submit<br/>submit.py::submit_jobs<br/>create run_&lt;UTC&gt;Z/ dir"]
  subgraph stages["SLURM job graph — sbatch --parsable, dependency-gated"]
    direction TB
    datagen["1 · datagen (single)<br/>_datagen::main<br/>→ pretrain_data[_polarized].npz"]
    pretrain["2 · pretrain [0..A-1, per arch]<br/>_pretrain::main → run_pretrain<br/>→ pretrain/&lt;arch&gt;/{xnet,cnet}.eqx"]
    preflight["3 · preflight (single)<br/>_preflight::main<br/>materialize specs + CCSD refs<br/>→ manifest.json (atomic, FINAL)"]
    train["4 · train [0..N-1, per spec]<br/>_train_task::main → run_training<br/>→ checkpoints/spec_NNNN/"]
    evalmode{"eval mode?"}
    evalstd["5 · eval [0..N-1]<br/>_eval_one_spec::main<br/>in-sample + held-out MAE"]
    launcher["eval_launcher<br/>_submit_eval::submit_deferred_eval"]
    inlinenote["inline: train_eval_inline template;<br/>eval runs in the SAME task"]
    datagen -->|afterok:datagen| pretrain
    pretrain -->|afterok:pretrain| preflight
    preflight -->|afterok:pretrain:preflight| train
    train --> evalmode
    evalmode -->|standard: aftercorr:train| evalstd
    evalmode -->|deferred: afterany:train| launcher
    launcher -.->|aftercorr:train| evalstd
    evalmode -.->|inline_eval| inlinenote
  end
  submit --> datagen
  subgraph art["run_&lt;UTC&gt;Z/ artifacts"]
    direction TB
    a_meta["manifest.json · jobs.json<br/>attempts.json · resolved_config.yaml"]
    a_pre["pretrain/&lt;arch&gt;/{xnet,cnet}.eqx · losses_*.npy"]
    a_ckpt["checkpoints/spec_NNNN/<br/>model.eqx · losses.npy · aux_log.pkl<br/>eval/ · eval_df.csv · eval_holdout/ · failure.json"]
  end
  pretrain -.->|writes| a_pre
  preflight -.->|writes| a_meta
  train -.->|writes| a_ckpt
  evalstd -.-> a_ckpt
  subgraph ops["read / recover (login node)"]
    direction TB
    o_status["status → job_tracking.reduce_outcomes"]
    o_results["results → aggregate eval_df.csv"]
    o_pull["pull → sync.py::build_rsync_command"]
    o_resub["resubmit / resubmit-preflight / repair-manifest"]
  end
  a_ckpt -.-> o_status
  a_ckpt -.-> o_results
  a_meta -.-> o_pull
  a_ckpt -.-> o_resub
```

[Full-resolution image → `diagrams/01_harness_pipeline.svg`](diagrams/01_harness_pipeline.svg)

![Harness SLURM pipeline](diagrams/01_harness_pipeline.svg)

---

## 2 · Per-spec training flow — `run_training`

Each train array task (`_train_task.py::main`) spawns a worker that calls
`train.py::run_training(spec)`. It builds the model (fresh `AlecGGAModel.from_arch`, or loaded
from the arch's `pretrain/` checkpoint), builds the batch
(`_build_batch → precompute_fixed_density_data`; the 4-index `eri` / DF `cderi` is added only
when the solver mode is `FULL`), then **dispatches** to one of five loops on `update_scheme`
and `balancing`. Every loop funnels through a single JIT-compiled step
(`filter_value_and_grad → optimizer.update → apply_updates`), guarded by `_abort_if_nonfinite`
— which **raises immediately** on a non-finite loss/gradient, naming the loop/step/group/channel
(the safeguard added after the polarized-correlation NaN). Final state is written by
`_save_artifacts`.

```mermaid
flowchart TB
  task(["_train_task::main (array task i)<br/>load specs/spec_i.spec"]) --> worker["worker: xcquinox.alec._train_one_spec<br/>→ run_training(spec)"]
  worker --> validate["spec.validate()"]
  subgraph build["run_training: build"]
    direction TB
    validate --> dpre{"pretrain_checkpoint?"}
    dpre -->|no| mscratch["_build_model:<br/>AlecGGAModel.from_arch"]
    dpre -->|yes| mload["_build_model: load<br/>pretrain/&lt;arch&gt;/{xnet,cnet}.eqx"]
    mscratch --> mkloss["make_loss(loss_name, ...)<br/>LOSS_REGISTRY[loss_name]"]
    mload --> mkloss
    mkloss --> batch["_build_batch →<br/>precompute_fixed_density_data"]
    batch --> dfull{"solver mode == FULL?"}
    dfull -->|yes| addEri["required_keys += eri / cderi"]
  end
  dfull -->|no| dscheme{"update_scheme ==<br/>per_molecule?"}
  addEri --> dscheme
  dscheme -->|yes| lperm["_run_per_molecule_loop<br/>1 step per target group / epoch"]
  dscheme -->|no, batched| dbal{"balancing?"}
  dbal -->|None| lstatic["_run_static_loop"]
  dbal -->|LossNorm| llnorm["_run_lossnorm_loop"]
  dbal -->|TwoPhase| l2phase["_run_twophase_loop"]
  dbal -->|GradNorm| lgnorm["_run_gradnorm_loop"]
  subgraph step["per optimizer step (every loop)"]
    direction TB
    sstep["_step / _train_step  [filter_jit]<br/>(loss, comps), grads = filter_value_and_grad(compute_components)<br/>optimizer.update (clip_by_global_norm + adam) → apply_updates"]
    guard{"_abort_if_nonfinite<br/>loss + channels finite?"}
    raise["raise FloatingPointError<br/>names loop/step/group/channel"]
    more{"more steps?"}
    sstep --> guard
    guard -->|no: NaN/Inf| raise
    guard -->|yes| more
    more -->|next step| sstep
  end
  lperm --> sstep
  lstatic --> sstep
  llnorm --> sstep
  l2phase --> sstep
  lgnorm --> sstep
  sstep -.->|forward| cc["compute_components → SCF + 5 channels<br/>(diagram 3)"]
  more -->|done| save["_save_artifacts<br/>model.eqx · losses.npy · aux_log.pkl · train_metadata.json"]
```

[Full-resolution image → `diagrams/02_training_flow.svg`](diagrams/02_training_flow.svg)

![Per-spec training flow](diagrams/02_training_flow.svg)

---

## 3 · Loss channels + the differentiable SCF

The L5 loss (`losses.py::L5GradnormVxcStep7.compute_components`) returns five GradNorm
channels. The three energy channels (`loss_AE`, `loss_BH76`, `loss_IP13`) consume per-molecule
energies from `_compute_energies → total_energy_for_solver`, which branches on solver mode:
`ONESHOT`/`FIXED_J` use `fixed_density_total_energy` (frozen PBE density), while `FULL` runs
`run_scf → run_manual_scf`, a differentiable `jax.lax.scan` over `max_cycles` self-consistent
cycles. `loss_vxc` matches the NN potential (`compute_vxc_nn` / `compute_vc_polarized_per_spin`)
against `vxc_ref`; `loss_rho` matches the NN grid density (`oneshot_grid_density`, which itself
runs the SCF density under `FULL`) against `rho_ref_grid`.

> The polarized-correlation NaN fixed earlier this session lives in the `compute_vxc_nn` /
> `_diagonalize_roothaan` part of this loop — the `eigh` cycle differentiated at full spin
> polarization (ζ=±1).

```mermaid
flowchart TB
  subgraph loss["compute_components → 5 GradNorm channels"]
    direction TB
    cc["compute_components(model, batch, relative)"]
    ae["loss_AE<br/>_ae_losses + _atomic_reg"]
    bh["loss_BH76<br/>_bh76_channel"]
    ip["loss_IP13<br/>_ip13_channel"]
    vxc["loss_vxc<br/>_vxc_term → compute_vxc_nn /<br/>compute_vc_polarized_per_spin (vs vxc_ref)"]
    rho["loss_rho<br/>_grid_term → oneshot_grid_density (vs rho_ref_grid)"]
    cc --> ae
    cc --> bh
    cc --> ip
    cc --> vxc
    cc --> rho
  end
  ae --> ce["_compute_energies<br/>E_nn[i] per molecule"]
  bh --> ce
  ip --> ce
  ce --> tes["total_energy_for_solver(model, mol_data[i], solver_config)"]
  rho -.->|rho path runs SCF density when FULL| tes
  tes --> dmode{"solver mode?"}
  dmode -->|ONESHOT / FIXED_J| fixed["fixed_density_total_energy<br/>split_exc_energy_uks / compute_exc_nn"]
  dmode -->|FULL| runscf["run_scf → run_manual_scf<br/>_run_manual_scf_rks / _uks"]
  subgraph scf["jax.lax.scan over max_cycles — differentiable SCF"]
    direction TB
    cfeat["spin-resolve density; assemble features<br/>(FROZEN reuse | REASSEMBLE from D)"]
    cvxc["compute_vxc_nn (+ polarized vc)<br/>→ V_xc^NN (per-point JVP)"]
    cfock["F = h_core + J + V_xc^NN<br/>J: eri/cderi [FULL] | pinned [FIXED_J]"]
    cdiag["_diagonalize_roothaan<br/>Cholesky(S+reg) → sym-break → eigh → D_new"]
    cmix["mix DM; recompute energy at D_mixed"]
    cconv{"converged or last cycle?"}
    cfeat --> cvxc --> cfock --> cdiag --> cmix --> cconv
    cconv -->|next cycle| cfeat
  end
  runscf --> cfeat
  cconv -->|yes| result(["SCFResult.total_energy → E_nn"])
  fixed --> result
  result -.-> back["channels summed → scalar loss → grads<br/>(diagram 2 _step)"]
  vxc -.-> back
  rho -.-> back
```

[Full-resolution image → `diagrams/03_scf_and_loss.svg`](diagrams/03_scf_and_loss.svg)

![SCF cycle and loss channels](diagrams/03_scf_and_loss.svg)

---

## 4 · Pretrain (stage 2) and Eval (stage 5)

**Pretrain** (`pretrain.py::run_pretrain`, once per architecture) loads the datagen `.npz`,
assembles descriptor inputs (inserting the spin-polarization ζ column for polarized cnets),
builds the `xnet`/`cnet` pair, and fits them — exchange first, then correlation with the xnet
frozen — to the per-point Fx/Fc enhancement targets under `|ρ·ε_LDA|` integration weights,
saving `{xnet,cnet}.eqx`. **Eval** (`_eval_one_spec.py::main`, once per spec) skips specs with
no `model.eqx`, else runs the in-sample metrics (`evaluation.py::run_test`) **and** the held-out
BH76+W4-11 pools (`run_holdout_with_escalation → run_full_holdout_eval`), both reporting
NN-vs-PBE error.

```mermaid
flowchart TB
  subgraph pre["PRETRAIN — run_pretrain (per arch)"]
    direction TB
    pdata["pretrain_data[_polarized].npz<br/>(Fx, Fc targets per grid point)"]
    pasm["_assemble_pretrain_descriptors<br/>[rho, sigma, (zeta if polarized), *descriptors]"]
    pnet["create_network_pair(arch) → xnet, cnet"]
    pw["_compute_integration_weights<br/>w = |rho · eps_LDA|"]
    px["Phase 1: fit xnet → Fx target<br/>(weighted MSE; _train_step)"]
    pc["Phase 2: fit cnet → Fc target (xnet frozen)"]
    psave["save pretrain/&lt;arch&gt;/{xnet,cnet}.eqx<br/>losses_*.npy · pretrain_metadata.json"]
    pdata --> pasm --> pnet --> pw --> px --> pc --> psave
  end
  subgraph ev["EVAL — _eval_one_spec::main (per spec)"]
    direction TB
    echk{"model.eqx present?"}
    eskip["write eval/skipped.json; exit 0"]
    eload["load model.eqx + spec → TestSpec"]
    ein["IN-SAMPLE: run_test<br/>per-molecule NN vs PBE<br/>(AE / barrier / IP / density_rmse / vxc_mae)"]
    eindf["→ eval/per_molecule.json · eval_df.csv"]
    eho["HELD-OUT: run_holdout_with_escalation<br/>→ run_full_holdout_eval (BH76 + W4-11)"]
    ehodf["→ eval_holdout/per_molecule.json<br/>(NN-vs-PBE reaction MAE, kcal/mol)"]
    echk -->|absent| eskip
    echk -->|present| eload
    eload --> ein --> eindf
    eload --> eho --> ehodf
  end
  psave -.-> consumed["consumed by _build_model<br/>at train time (diagram 2)"]
  ckpt["checkpoints/spec_i/model.eqx<br/>(from training, diagram 2)"] -.-> echk
```

[Full-resolution image → `diagrams/04_pretrain_and_eval.svg`](diagrams/04_pretrain_and_eval.svg)

![Pretrain and eval](diagrams/04_pretrain_and_eval.svg)

---

### Cross-references

- The harness stages (diagram 1) invoke the per-stage Python entrypoints under
  `xcquinox/alec/cluster/` (`_datagen`, `_pretrain`, `_preflight`, `_train_task`,
  `_eval_one_spec`).
- Diagram 2's `compute_components` box expands into diagram 3.
- Diagram 4's pretrain checkpoints feed diagram 2's `_build_model`; diagram 2's
  `model.eqx` feeds diagram 4's eval.
