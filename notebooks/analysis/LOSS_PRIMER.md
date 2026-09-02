# Training-loss primer -- what is optimized, and how the relative weights actually flow

This note is the canonical description of the dfs_step7 training objective and of every
weighting decision between the sweep YAML and the optimizer step. It exists because the
weight story had been mis-stated twice in this repo's own documentation (Sec. 1.1) and
because the run artifacts under-reported it (Sec. 7). Line anchors were taken at the
revision that introduced this file; `grep -n` for the quoted text if drift is suspected.
Every numeric claim in Sec. 1-3 was re-verified against completed production specs of
`run_20260728T140018Z` (dfs6311_grid3_v3), not only against unit fixtures.

Paths below are relative to the repository root; the sweep YAML is
`hpcjobs/configs/dfs_step7.dfs6311_grid3_v3.yaml` (cited as `yaml:<line>`).

## 1. The effective objective on the production runs

The sweep selects the loss by name (`yaml:64`, `loss: [L5_gradnorm_vxc_step7]`), a
five-channel loss class (`xcquinox/alec/losses.py:1063`):

    target_kinds: ClassVar[tuple[str, ...]] = ("AE", "BH76", "IP13", "vxc", "rho")

The per-update objective actually optimized is

    L  =  1 * loss_AE  +  1 * loss_BH76  +  1 * loss_IP13  +  1 * loss_vxc  +  20 * loss_rho

with FIXED channel weights, from `_DEFAULT_CHANNEL_WEIGHTS`
(`xcquinox/alec/train.py:1737-1743`):

    # Density-dominant fixed channel weights (dpyscf: density L_n weight ~20,
    # atomization/reaction L_RE ~1, total-energy L_E ~0.01). Energy channels at 1.0,
    # density at 20.0, vxc at 1.0. Used when update_scheme="per_molecule" and the
    # spec sets no explicit channel_weights.
    _DEFAULT_CHANNEL_WEIGHTS = {
        "loss_AE": 1.0, "loss_BH76": 1.0, "loss_IP13": 1.0,
        "loss_vxc": 1.0, "loss_rho": 20.0,
    }

resolved by `_effective_channel_weights` (`train.py:1746`, spec overrides merged over
these defaults) and applied in `xcquinox/alec/defused_grad.py:156`:

    total = total + channel_weights.get(key, 1.0) * value

This is the Dick & Fernandez-Serra weight STRUCTURE: the Letter (PRB 104, L161109
(2021), after Eq. 18) states "We set the weights to lambda_RE = 1, lambda_n = 20, and
lambda_E = 0.01" -- the reaction-energy channels here carry weight 1 and the density
channel 20, exactly as in the Letter. There is no separate total-energy channel, so
lambda_E = 0.01 has no direct analog; its closest relative is the `w_atomic = 0.01`
anchor regularizer inside loss_AE (Sec. 1.2).

Executed proof on production data (`run_20260728T140018Z/checkpoints/spec_0012`,
`aux_log.pkl`): over all 1400 optimizer updates the recorded `loss` equals
`1*(loss_AE + loss_BH76 + loss_IP13 + loss_vxc) + 20*loss_rho` at machine precision
(max relative deviation 2.2e-16 in one summation order, exactly 0 in another); every
entry carries `update_scheme: "per_molecule"`; `loss_vxc` is nonzero in 1000/1400
updates (the OEP V_xc channel is live in production, zero only for groups whose
species carry no V_xc reference). The identity was reproduced independently on
spec_0007 (3800 updates) and spec_0016 (2400 updates).

### 1.1 GradNorm is present in the codebase but DORMANT on every dfs_step7 run

The loss class name (`L5_gradnorm_vxc_step7`) and the YAML knobs
(`gradnorm_alpha: 1.5`, `vxc_weight: 0.01`, `density_weight: 0.1`, `yaml:105-107`)
suggest adaptive task weighting. At runtime none of them act:

1. The YAML sets no `update_scheme`, so the cluster default applies
   (`xcquinox/alec/cluster/grid_config.py:148-153`):

       # Optimizer update scheme (2026-06-01). Defaults to the DFS/dpyscf-style
       # per-molecule stochastic updates (one optimizer step per training group per
       # epoch, fixed channel weights), the recommended default. Set "batched" to
       # use the historical full-batch + GradNorm path.
       update_scheme: str = "per_molecule"

   (dict resolution at `grid_config.py:487`; `channel_weights` likewise absent ->
   empty at `:489`). The sweep could not choose otherwise: `validate_every: 25`
   (`yaml:114`) REQUIRES the per-molecule scheme -- `grid_config.py:909-910` rejects
   any other with "hyperparams.validate_every=... > 0 requires
   update_scheme='per_molecule' (the only training loop with an in-loop validation
   hook)".
2. `run_training` dispatches on the scheme BEFORE consulting the balancer
   (`train.py:2194-2197`): "DFS/dpyscf-style per-molecule stochastic updates: one
   optimizer step per target-group per epoch with fixed channel weights (ignores
   `balancing`, whose GradNorm rebalancing is a full-batch construct)". The GradNorm
   loop (`_run_gradnorm_loop`, `train.py:1566-1567`, "GradNorm (Chen et al. 2018):
   learned per-task weights via gradient norm equalization"; alpha = 1.5,
   weight_lr = 0.025 at `balancing.py:53-56`; softmax reparameterization keeping
   sum(w) = n_tasks at `train.py:1587`) runs only under `update_scheme="batched"`,
   which no dfs_step7 YAML sets.
3. The per-molecule loop REBUILDS the loss per group with the pre-scales forced to one
   (`train.py:1353-1355`, `:1361-1362`): "Channels are RAW (vxc/density pre-weights
   forced to 1.0), the outer fixed ``channel_weights`` are the sole weighting control
   in per-molecule mode." The YAML's `vxc_weight: 0.01` and `density_weight: 0.1` are
   therefore inert on these runs.

The repo's documentation had recorded both a correct and an incorrect version of this:
the 2026-06-05 methods-box entry in `xcquinox/alec/HISTORY.md` states "the loss is
*not* GradNorm-balanced at run time (... GradNorm(alpha=1.5) is dormant; effective
weights {AE:1,BH76:1,IP13:1,vxc:1,rho:20})", while the 2026-07-29 Phase-36 follow-up
analysis and the `losses.py` docstrings claimed GradNorm "discover[s] task weights
adaptively". The former is correct; the latter described the batched path as if it
were the production path (corrected in the same change that adds this file -- see the
2026-08-03 HISTORY erratum).

### 1.2 What each channel computes

All residuals are Hartree-derived; references arrive in kcal/mol and are converted
once (`KCAL_PER_HA = 627.5094740631`, `xcquinox/alec/cluster/domain.py:27`;
`bh76_meta_to_loss_dict`, `cluster/domain.py:89`) and sanity-gated
(`_HA_REF_SANITY_MAX = 10.0`, `losses.py:955`). `loss_metric` defaults to `"absolute"`
(`xcquinox/alec/config.py:740`) and no harness path overrides it, so the
BH76/IP13/vxc/rho residuals below are absolute.

- **loss_AE** (`losses.py:1326`, via `_ae_losses` at `:246`) = AE fitting term +
  atomic-anchor regularizer.
  - The fitting term compares network atomization energies (network molecule energy
    against TABULATED atomic anchors, `_ae_from_atoms`, `losses.py:208`) to reference
    AEs, RELATIVE-normalized with a floor (`losses.py:272`):
    `sq / jnp.maximum(tgt ** 2, _DELTA_TGT_FLOOR_HA2)`, floor = (1 kcal/mol)^2 in Ha^2
    (`losses.py:32`). The floor exists because "Na2's 0.0273 Ha target inflated its
    channel ~1340x vs CO2-class targets" (`losses.py:257`).
  - THE FITTING TERM'S REALIZED FORM DIFFERS BY RUN GENERATION. The source YAMLs
    request `ae_as_reactions: true` (`yaml:209`), which converts every AE point into
    a BH76-channel REACTION trained against the network's OWN atom energies
    (`xcquinox/alec/training_points.py:133`), with the names force-added to
    `aux_only_names` (`cluster/spec_builder.py:527`) and stripped from the compound
    index (`losses.py:1162-1165`). BUT the resolved-config serializer dropped the
    flag until 2026-08-10 (`cluster/__main__.py`, `_config_to_raw_dict`), and the
    preflight re-reads `resolved_config.yaml` before building specs -- so the
    v2/v3/v3_full25 runs REALIZED the fixed-anchor form above (verified on the v3
    artifacts: `resolved_config.yaml` lacks the key, 21 `ae:<name>` groups are live
    in `aux_log.pkl`, and `spec_0010` carries 3 BH76 reactions where the reaction
    form yields 24). Runs submitted after the fix (v4 on) realize the reaction form
    the YAML asks for.
  - What remains is `w_atomic * _atomic_reg` (`losses.py:223`): the relative squared
    deviation of the network's FREE-ATOM energies from tabulated anchors, restricted
    to H and Li (`domain.regularize_atom_syms`, threaded at
    `cluster/spec_builder.py:508`, `:554`), at the constructor default
    `w_atomic = 0.01` (`losses.py:1092`; the harness never overrides it). This is the
    only 0.01 factor surviving at runtime.
- **loss_BH76** (`_bh76_channel`, `losses.py:1225`; `_rxn_residual_term`, `:473`):
  mean over reactions of `(sum_k c_k E_k - E_rxn_ref)^2` in Ha^2 (absolute mode).
  Under `ae_as_reactions` this channel carries BOTH the genuine BH76 reaction
  energies (`bh76_mode: reaction_energy`, `yaml:208` -- reaction energies, NOT the
  Letter's barrier heights; the deviation is documented at
  `training_points.py:337`) AND the reaction-form AE points.
- **loss_IP13** (`_ip13_channel`, `losses.py:1255`; `_ip_residual_term`, `:524`):
  `(E_cation - E_neutral - IP_ref)^2`, Ha^2.
- **loss_vxc** (`losses.py:1346`; `_vxc_term`, `:399`): squared Frobenius deviation of
  the network's V_xc matrix from the OEP reference, normalized by n_ao^2 (2*n_ao^2 for
  UKS); species without a V_xc reference are skipped with a RuntimeWarning
  (`losses.py:456`). Runtime weight 1 (the 0.01 pre-scale is forced to 1, Sec. 1.1).
- **loss_rho** (`losses.py:1349`; `_grid_term`, `:357`): the dpyscf density residual
  `sum_i w_i (rho_i - rho_ref,i)^2 / N_e^2` (`losses.py:382`), with
  `density_per_electron: true` (`yaml:108`) selecting the N_e^2 normalization
  (N_e = sum w*rho_ref). Provenance and the one deviation are documented at
  `losses.py:364`: dpyscf normalizes per spin channel by N_sigma^2; this code carries
  a spin-summed density and uses the total N_e^2. Runtime weight 20. This is the
  Letter's L_n (Eqs. 17-18, `l_n = (1/N_e^2) int |n - n_ref|^2`) up to that detail.

The vxc and rho channels supervise molecules only -- compounds plus aux species; free
atoms carry no density/V_xc references (`_iter_idx_for_aux_channels`,
`losses.py:1362`).

## 2. How the weights travel through the pipeline

    YAML (hpcjobs/configs/dfs_step7.<basis>.yaml)
      | loss: [L5_gradnorm_vxc_step7]                           (yaml:64)
      | hyperparams: gradnorm_alpha, vxc_weight, density_weight,
      |   density_per_electron (yaml:105-108); validate_every,
      |   patience, early_stop_min_delta (yaml:114-116)
      | solvers.full_3: scf_loss_use_tail/tail/weight_power     (yaml:73-81)
      | bh76_mode, ae_as_reactions                              (yaml:208-209)
      v
    cluster/grid_config.py  (config resolution; defaults fill the gaps)
      | update_scheme  <- ABSENT in YAML -> "per_molecule"      (:148-153, :487)
      | channel_weights <- ABSENT -> ()                          (:489)
      | validate_every>0 enforces per_molecule                   (:909-910)
      | round-tripped to <run_dir>/resolved_config.yaml
      v
    cluster/spec_builder.py  (per-cell TrainingSpec, pickled spec_NNNN.spec)
      | loss_kwargs = {bh76_reactions, ip13_pairs, aux_only_names,
      |   regularize_atom_syms, solver_config, vxc_weight, density_weight,
      |   density_per_electron}                                  (:551-562)
      | balancing = GradNormConfig(alpha=hp.gradnorm_alpha)      (:595)  [dormant]
      | update_scheme = hp.update_scheme                         (:603)
      | channel_weights = hp.channel_weights  (empty)            (:605)
      v
    train worker (workers/train_worker.py:43-46; cluster path _train_one_spec.py)
      | pickle.load(spec) -> run_training(spec)
      v
    train.run_training
      | dispatch: update_scheme == "per_molecule"                (:1707-1710)
      |   -> _run_per_molecule_loop   (balancing NEVER consulted)
      v
    _run_per_molecule_loop                                       (:1393)
      | cw = _effective_channel_weights(spec.channel_weights_dict)   (:1404)
      |   empty spec weights -> _DEFAULT_CHANNEL_WEIGHTS {1,1,1,1,20} (:1258-1271)
      | per group: scoped loss rebuilt with vxc_weight=density_weight=1.0
      |   (:1353-1362), one optimizer step per group per epoch
      | defused_value_and_grad(gloss, model, gbatch, cw, relative, ...)  (:1438)
      |   -> total = sum_k cw[k] * component_k   (defused_grad.py:156)
      v
    artifacts
      | aux_log.pkl: per-update {step, epoch, group, loss, aux={raw components},
      |   update_scheme, rss_gb, hwm_gb}       (:1591-1597)  [weights NOT logged]
      | losses.npy, model.eqx / model_best.eqx / model_val_best.eqx
      | train_metadata.json (Sec. 7)

Update granularity: each epoch shuffles the per-target groups and takes ONE optimizer
step per group (`_run_per_molecule_loop` docstring, `train.py:1394`; groups built at
`_training_groups`, `:1280` -- one per BH76 reaction, one per IP13 pair, one per
non-aux AE compound, one per regularized neutral atom anchor). Two weighting
consequences worth naming: (i) a molecule appearing in many groups is supervised
proportionally more often, and its density term (weight 20) fires once per containing
group per epoch; (ii) under `ae_as_reactions` the would-be `ae:<name>` groups are
skipped so those molecules are not double-supervised (`train.py:1323`).

## 3. Per-SCF-step (trajectory) weighting -- the energy channels only

With `scf_loss_use_tail: true` (`yaml:79-81`), each energy residual is scored on a
weighted WINDOW of the SCF energy trajectory instead of the final cycle alone
(`losses.py:1303-1312`). The window comes from `scf_tail_window`
(`xcquinox/alec/oneshot.py:632`): keep the last `min(tail, N)` steps
(`skip = max(0, N - tail)`, generalizing dpyscf's `max(5, N-10)` which underflows for
small N) with weights `linspace(0, 1, N)**power` restricted to the kept steps; the
residual uses the SQUARED weights (`step_w2 = step_w ** 2`, `losses.py:1312`).

For the production solver `full_3` (max_cycles 3, tail 10, power 2.0) the call
returns all three steps with weights [0, 0.25, 1.0], i.e. step_w2 = [0, 0.0625, 1] --
the first SCF cycle carries exactly zero weight and each energy residual is
`mean(step_w2 * r^2)` over the trajectory. The tail applies to loss_AE, the anchor
regularizer, loss_BH76, and loss_IP13 (the `step_w2` paths in `_ae_losses`,
`_atomic_reg`, `_rxn_residual_term`, `_ip_residual_term`); loss_vxc and loss_rho use
the final converged density only ("DFS weights energy only").

Paper vs vendored code, adjudicated against both sources:

- The Letter (Eqs. 15-16 and the following text) writes the trajectory weights as
  `w_j = ((j-10)/15)^2` over SCF iterations j = 10..25, "employing w_j = ((j-10)/15)^2
  that penalize solutions which lead to slowly converging SCF calculations".
- The vendored dpyscf implementation does something different
  (`~/Documents/Research/og_dpyscf/og_dpyscf/ogdpyscf/losses.py:13-18`):
  `weights = torch.linspace(0,1,N)**2` over the WHOLE trajectory, then
  `dE = dE[skip_steps:]` with `skip_steps = max(5, args.scf_steps - 10)`
  (`og_dpyscf/scripts/train.py:283`). At scf_steps = 25 that keeps the LAST 10 steps
  with weights (15/24)^2..1 ~ 0.39..1.0 -- not the paper's 16 steps with weights
  0..1. (Verified by calling `scf_tail_window(25, 10, 2.0)` -> skip 15, weights
  0.390625..1.0.)
- `scf_tail_window`'s docstring claim of reproducing "DFS exactly" at
  (N=25, tail=10, power=2) therefore refers to the dpyscf CODE, not the paper's
  formula; this primer is where that distinction is recorded.

The same solver block pairs the tail loss with the DFS step-decaying mixer
(`decaying_linear`, `alpha = 0.3**step + 0.3`, `xcquinox/alec/solver.py:269`,
mirroring og_dpyscf `torch_routines.py:175`): the tail loss penalizes a
non-converging trajectory, the mixer damps it.

## 4. Optimizer and schedule (vs the Letter)

- Here: `optax.chain(clip_by_global_norm(grad_clip), adamw(lr_schedule,
  weight_decay))` (`train.py:165`) with linear LR decay over `n_epochs * n_groups`
  updates; production `weight_decay = 1e-4` (`yaml:112`), DECOUPLED AdamW. The
  deviation is documented in-code (`train.py:117`): the production weight decay "is
  two orders of magnitude larger than DFS's L2 regularization of 1e-6 (SI), and
  `adamw` applies DECOUPLED weight decay whereas DFS regularizes with coupled
  Adam-L2."
- The Letter: "The functional parameters are optimized using Adam with an initial
  learning rate 10^-4 which is decayed by a factor of 0.1 after every ten consecutive
  epochs without a decrease in training loss. We employ an l2 regularization of 10^-6
  and a batch size of one reaction." The per-group update scheme here is the analog of
  that batch-of-one-reaction training.

## 5. Validation and checkpoint selection -- where the weights DIFFER from the Letter

In-loop validation scores REACTION-ENERGY MAE ONLY, in kcal/mol: no density term, no
V_xc term, no anchor term (`_validation_reaction_mae`, `train.py:515-556`; the
validation MoleculeData is built with "NO external reference (no dm_target / rho_ref /
E_ref)", `cluster/spec_builder.py:309-312`). Every `validate_every = 25` epochs the
model is scored on the held-out validation reactions (`train.py:1622`);
`model_val_best.eqx` snapshots the best MAE (`train.py:443-445`) and early stopping
fires after `patience = 5` consecutive non-improving checks at
`early_stop_min_delta = 0.1` kcal/mol (`train.py:447`, `:1632`; `yaml:114-116`).
`model_best.eqx` separately tracks the best TRAINING loss (`train.py:384`, `:398`).

The Letter's validation loss, by contrast, is "identical to the training loss
presented in the main text except for lambda_E = 0 and w_j = delta_j,25" (SI) -- the
density term at weight 20 participates in every checkpoint decision. This selection
asymmetry is a structural deviation, but on the current run it is NOT costly: over the
14 dfs6311 specs with all three evals the median-of-medians NN/PBE eps ratio is 1.0503
(final), 1.0520 (train-best), 1.0488 (val-best) -- a selection cost of -0.1%
(`DENSITY_DIAGNOSIS.md` Sec. 5). An earlier figure quoted in HISTORY (1.066 -> 1.112)
came from the RMSE channel on a different pull and should not be carried forward as
this run's selection cost.

So the accurate one-line summary of "how do our weights differ from DFS" is NOT the
training lambdas (those match: the 1 / 20 structure) but: (i) the density channel exerts
no effective optimization pressure -- it is flat across training, and the open-shell
radicals CH and NO occupy 68-98% of it with a degenerate-component mismatch no functional
can close, so the runs are effectively energy-only training with a decorative density
term; this is the dominant effect and is diagnosed in `DENSITY_DIAGNOSIS.md` (the Letter
met the same wall on the same species class and down-weighted lambda_n by 0.01 for it);
(ii) the SCF trajectory is 3 cycles with a
[0, 0.0625, 1] tail rather than 25 cycles with the long dpyscf tail; (iii) checkpoint
SELECTION ignores the density channel, where the Letter selects density-inclusively
(structural, but measured at -0.1% here); (iv) the nominal pre-scales (0.01 vxc, 0.1
density) and the GradNorm knobs in the YAML are inert.

## 6. Pretraining (one paragraph)

Pretraining is a separate per-architecture stage that fits the exchange and
correlation ENHANCEMENT FACTORS pointwise on the stored grid rows of the pretraining
set. Three properties the earlier one-paragraph description got wrong, corrected here
for v6 (`grep -n` the quoted text on drift): the set is NOT per-atom -- the v6
resolution is 38 systems, 16 free atoms plus 22 molecules (`resolve_pretrain_systems`
with `dfs_set: true` + `pool_atoms: true`; `pretrain_metadata.json` records
`n_systems: 38`); every row's density comes from a CONVERGED reference SCF
(`pretrain_data_gen._system_columns` via `precompute_fixed_density_data`, refused
unless stamped converged by `_require_sane_density`); and the objective CARRIES an
energy term (`_PretrainLoss`, per-system Hartree^2), at weight 0.0 in v6, so the
fitted quantity is the point-wise residual but the objective's form is not
energy-free. Targets are stored as F - 1; GGA architectures pretrain to PBE targets
and the meta-GGA architectures to SCAN ("DFS-faithful meta_gga archs pretrain to SCAN
(a GGA structurally cannot fit SCAN's alpha-dependence)",
`xcquinox/alec/pretrain.py:1299`). The objective is an integration-weighted MSE
(`loss_weighting: integration`): `jnp.sum(w * residual_sq) / (jnp.sum(w) + 1e-12)`
(`_PretrainLoss.parts`, `pretrain.py:226-249`) with per-point weights
`w_i = |rho_i * eps_LDA,i| * w_grid,i` (`_compute_integration_weights`,
`pretrain.py:100-115`) -- a DEVIATION from the Letter's pretraining, which fits a
plain unweighted MSE over all rows (vendored `pretrain.ipynb`); x-net and c-net are
fitted in separate trainer calls. The convention caveat is documented in-code
(`pretrain.py:113`): "This is NOT the squared integrated XC-energy residual." The
initialization is scored as validation step 0 and is a full best-model candidate
(under a parent anchor it is the exact optimum of this objective). The step-7 loss
hard-rejects a PBE anchor at train time (`losses.py:513-521`), matching
`pbe_anchor_weight: 0.0`.

## 7. Observability -- what the artifacts record

- `aux_log.pkl` (per update, `train.py:1591-1597`): `step`, `epoch`, `group`, total
  `loss`, `aux = {loss_AE, loss_BH76, loss_IP13, loss_vxc, loss_rho}` as RAW
  (unweighted) components, `update_scheme`, RSS/HWM. The x20 density weight is
  recoverable only as `loss - sum(aux) = 19 * loss_rho`. Validation records use the
  same file with `group: "__validation__"` and a `val_mae_kcalmol` field.
- `losses.npy`: the weighted total per update.
- `train_metadata.json`: previously recorded the NOMINAL `loss_kwargs` pre-scales
  (0.01 / 0.1) and `balancing: {strategy: gradnorm, alpha: 1.5, weight_lr: 0.025}`
  while omitting `update_scheme` and the effective channel weights -- the
  mis-reporting that seeded the wrong premise twice. As of the change introducing
  this file the metadata additionally records `update_scheme`,
  `balancing_active` (False on every per-molecule run), and
  `effective_channel_weights`, so the runtime weighting is readable from the artifact
  alone (`train.py`, `_save_artifacts` metadata block).
- Nothing at evaluation time consumes the training weights; the eval pipeline scores
  energies and densities directly.

## 8. Deviation table vs Dick & Fernandez-Serra (PRB 104, L161109 (2021))

| Aspect | Letter | This pipeline |
|---|---|---|
| Channel weights (training) | lambda_RE=1, lambda_n=20, lambda_E=0.01 (after Eq. 18) | Fixed {AE 1, BH76 1, IP13 1, vxc 1, rho 20}; no total-energy channel; w_atomic=0.01 anchor regularizer; GradNorm dormant |
| Density residual | (1/N_e^2) int (n-n_ref)^2, per spin channel (dpyscf N_sigma^2) | Same form, spin-summed N_e^2 (`losses.py:364`) |
| SCF depth + trajectory weights | 25 cycles; paper: w_j=((j-10)/15)^2, j=10..25; vendored code: last 10 steps of linspace(0,1,25)^2 | 3 cycles (`full_3`); tail weights [0, 0.25, 1.0] squared to [0, 0.0625, 1] |
| BH76 targets | Barrier heights (SI Sec. I) | Reaction energies (`training_points.py:337`) |
| AE supervision | AE targets vs tabulated atoms, weight 1 in L_RE | v2/v3 runs: fixed-anchor relative AE (the resolved-config serializer dropped `ae_as_reactions` until 2026-08-10); v4 on: reaction-form with the network's own atoms; H/Li anchors regularized at 0.01 |
| Validation / selection | Training loss with lambda_E=0, w_j=delta_j,25 (density-inclusive) | Reaction-energy MAE only (no density) -- measured density cost at val-best selection |
| Optimizer | Adam 1e-4, plateau decay 0.1, coupled l2 1e-6, batch = one reaction | AdamW (decoupled wd 1e-4), grad clip, linear decay, one step per group per epoch |
| References | CCSD(T)/6-311++G(3df,2pd) | CCSD, density fitting, same basis/grid (`yaml:27-35`) |
| CH/OH degenerate radicals | lambda_n scaled by 0.01 for CH and OH | Orientation lock in the reference and training SCFs (`yaml:13-19`) |
| SCF convergence freeze | None: all 25 cycles run unconditionally | `full_3` freezes the SCF state once the per-cycle |dE| < 1e-6 Ha (a theta-dependent branch; fires on 22.9% of v3 molecule-instances, 9.4% before the last cycle) |
| SCF initialization | Randomized every optimization step: (1-beta) rho_atomic + beta rho_DFT, beta = (r+1)/2, r ~ U(0,1) (SI Sec. III A) | Converged PBE density matrix, deterministic |
| Degenerate-eigenvalue guard | Random half-normal V_xc noise, std 1e-8, symmetrized (SI Sec. IV) | Deterministic SYM_BREAK_SHIFT 1e-6 diagonal (HISTORY Phase 33) |
| Mixer schedule index | SI equation alpha_i = 0.3^i + 0.3; the vendored code no-ops the step-0 mix, so its first effective alpha is 0.6 | The SI equation verbatim: the first mix uses alpha_0 = 1.3 (one-step offset against the paper's own code) |
| Pretrain mesh | SI: 2-D (s, alpha) at fixed rho = 1, exchange only, ~10100 nodes, equal weight per point | 3-D (r_s, s, alpha), exchange AND correlation, 560 nodes, flat 30% loss-weight share per channel |
| Atomic densities | H and Li electron densities supervised (SI Sec. II) | Free-atom anchor groups carry no density/V_xc references (their density channels iterate over nothing) |
| V_xc channel | No such channel exists (the vendored loss vocabulary is {ip, energy, econv, ae, dm, rho, rho_alt, moe, gap}) | ADDED: loss_vxc at runtime weight 1.0, squared Frobenius deviation from a Wu-Yang OEP reference normalized by the physical n_ao^2 |
| Self-consistency per point class | BH76 + IP entries staged non-self-consistently ('sc': False, one pass, loss x 0.5 nonsc_weight) | Every point class fully self-consistent under one solver_config; no 0.5 multiplier |
| Learning rate | 1e-4 (vendored --lr default; SI) | lr_start 1e-3 (10x), constant for 50% then linear to 1e-5 |
| Gradient clipping | None | clip_by_global_norm(1.0) (load-bearing: an unclipped Na2 channel once pegged training) |
| NaN handling | 3-strike rollback to rotating checkpoints on RuntimeError | Fixed epoch budget, validation early stop, no rollback (spec_0010's unhandled NaN failure is the measured consequence) |
| Pretraining objective | Plain unweighted MSE over all rows, Adam 1e-3, no stopping rule (vendored pretrain.ipynb) | Integration-weighted MSE (Sec. 6), lr 0.01 -> 1e-5 with decay, 20% held-out validation split with step-0 (initialization) scored |
| AE tail weights at N=25 | ae_loss re-derives linspace(0,1)^2 on the SLICED trajectory (first kept step weighted 0), unlike energy_loss | One step_w2 applied uniformly to every energy channel (identical at the production N=3, where the conventions coincide) |
| Evaluation suites | BH76 + G2/97 + IP13 + S22 (vendored test_*.py) | BH76 + W4-11 held-out pool; S22 and IP13 not evaluated |
