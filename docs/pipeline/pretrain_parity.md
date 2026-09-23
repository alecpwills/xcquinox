# Pretraining parity with the published cloning protocol

The pretraining of the v8 campaigns follows the functional-cloning protocol of
arXiv:2605.10331 (Navarro and co-workers), read from its public code at
gitlab.com/saru1799/xcquinox-clone (branch `public`): `xcquinox_clone/reference_functionals.py`,
`models.py`, `train.py`, `preprocessing.py` and
`do_cloning/scripts_clone/clone_using_slim_densities.py`. Each row below names the element,
what the published code does, what this repository does under the published settings, and the
test or command that establishes the correspondence.

## The published settings

```yaml
pretrain:
  loss_weighting: rho_w_sampled
  points_per_system: 800
  n_steps: 20000
  lr_start: 1.0e-3
  lr_end: 1.0e-5
  lr_decay_start: 0.5
  lr_decay_end: 0.9
  grad_clip: 0
  energy_term_weight: 0.0
  validation_fraction: 0.0
  patience: 0
  exchange_footing: paper
model:
  parent_anchor: false
  descriptor_coordinates: paper
  ueg_gate: x2
```

The step count, the molecule count and the point count are the paper's text (arXiv:2605.10331,
Sect. II.3: 25 molecules, 20000 points, 20000 steps); the published script takes the first two
from its command line (`--mols_train`, `-N`, with defaults of 10 and 30000) and the points per
molecule are their quotient, 800 at the paper's values.

## Element by element

| element | the published code | this repository | check |
|---|---|---|---|
| exchange target | `Fx_PBE`: the spin densities and gradients summed, `F_x = 1 + kappa - kappa / (1 + mu s^2 / kappa)` with `s = |grad rho| / (2 k_F rho + 1e-30)`, `k_F = (3 pi^2 rho)^(1/3)`, kappa 0.804, mu 0.2195149727645171 | `parents.paper_pbe_fx`; under `exchange_footing: paper` every system's rows, open shells included, carry it on the total density | `test_the_paper_targets_equal_the_published_formulas`, `test_a_paper_file_carries_the_published_targets_for_every_system` |
| correlation target | `Fc_PBE`: PBE's `H` on the code's own PW92, `F_c = 1 + H / eps_c`, with `A = (0.031090690869654895, 0.015545, 0.016887)`, beta 0.06672455060314922, gamma (1 - ln 2) / pi^2, zeta = (rho_up - rho_down) / (rho + 1e-30) | `parents.paper_pw92_eps_c`, `parents.paper_pbe_fc` | the same two tests |
| LDA energy densities | `-3/4 (3/pi)^(1/3) rho^(1/3)` and the PW92 above | `e_lda_x`, `e_lda_c` of the file | the file test |
| where the two parameter sets differ | libxc evaluates PBE's numerator on PW_MOD (`A = 0.0310907, 0.01554535, 0.0168869`) and divides the stored ratio of the `total` footing by PW (`A = 0.031091, 0.015545, 0.016887`), which is also this repository's model baseline | the `total` footing's correlation rows and the model's baseline keep libxc's sets; the `paper` rows carry the published one; on the two-system test file the closed-shell correlation targets of the two footings differ by at most 3.8e-6 (the relative difference of the amplitude sets being 4.0e-6 on those rows), and over a grid of `r_s` in [0.05, 50] and `t` in [0, 4] at zero polarization by at most 5.8e-6 | `test_the_paper_footing_changes_only_what_the_paper_changes` prints the largest closed-shell difference of the correlation targets and its first-order bound |
| network inputs | `x0 = log(rho^(1/3) + 1e-5)`, `x1 = log(zeta' + 1e-5)` with `zeta' = ((1 + zeta)^(4/3) + (1 - zeta)^(4/3)) / 2`, `x2 = (1 - exp(-s^2)) log(1 + s)`; the exchange network reads `x2` alone, the correlation network all three | `descriptor_coordinates: paper` (the dfs coordinates with the epsilon inside `x1`), on a density floored at 1e-12 and a polarization clipped into [-1, 1] (stated below; inactive on physical rows) | `test_paper_coordinates_add_the_epsilon_to_x1`, `test_paper_coordinates_for_exchange_are_the_dfs_coordinates` |
| network form | MLP of depth 3 and width 16, GELU, `F = 1 + LOB_a(x2 net(...))`, `LOB_a(x) = a sigmoid(x - ln(a - 1)) - 1`, a = 1.804 (exchange) and 2.0 (correlation) | the same bound map (`networks._AlecLOB`); `ueg_gate: x2` puts `x2` in place of `tanh(s)^2` before the map | `test_the_x2_gate_multiplies_the_network_by_the_transformed_reduced_gradient` |
| initialization | both networks from `PRNGKey(42)`; no zero initialization of the final layer | the exchange network from `seed`, the correlation network from `seed + 1`; the registry entries set `zero_init_final_layer` | stated; not changed |
| use at SCF time | exchange per spin channel on the doubled densities, correlation on the total density with zeta | the same | read against `models.py` of the published code (`RXCModel.__call__`), not pinned by a test |
| loss | the plain mean of the squared residual of `F` over the sample | the masked loss over the 0/1 sample of `F - 1` against `F - 1`, the same residual | `test_masked_loss_is_the_plain_mse_over_the_sampled_rows` |
| optimizer | Adam alone | `grad_clip: 0` removes the clip | `test_grad_clip_zero_disables_the_clip` |
| schedule | constant 1e-3 to 50 percent of the steps, linear to 1e-5 at 90 percent, constant after | `lr_decay_start: 0.5`, `lr_decay_end: 0.9` | `test_published_shape_holds_the_floor_from_ninety_percent` |
| steps | 20000 full-batch steps | `n_steps: 20000` | the configuration |
| point sampling | `N_POINTS // MOLS_TRAIN` points per molecule, drawn without replacement with probability `w rho / sum(w rho)`; the script's defaults are 30000 points over 10 molecules, and the paper's text (arXiv:2605.10331, Sect. II.3) states 20000 points over 25 molecules, 800 each, which the script reaches through its command line | `_rho_w_sampling_mask`: the same size and measure; the random stream is the repository's own | `test_the_draw_is_without_replacement`, `test_the_draw_is_biased_toward_the_w_rho_measure` |
| molecules | `MOLS_TRAIN` indices drawn from the script's molecule list (the de-duplicated set its `superdict.json` holds) with `np.random.seed(42 + i)` for repetition `i`, seed 42 for the single default repetition; 25 by the paper's text | the Slim05 pool and the draw (a later change) | later |
| energy term, validation, early stop | none | `energy_term_weight: 0.0` (the term is not evaluated), `validation_fraction: 0.0`, `patience: 0` | `test_the_pretrain_block_accepts_the_published_values` |
| mesh rows | none | the synthetic mesh rides at weight zero under `rho_w_sampled` | `test_mesh_rows_ride_at_zero_weight_and_are_never_sampled` |
| fidelity certificate | none | kept as the gate on training | unchanged |
| geometric architectures | none | the exchange input `[x2, cusp_0, cusp_1]`, the correlation input `[x0, x1, x2, cusp_0, cusp_1]` | `test_the_geometric_architectures_pretrain_on_the_published_inputs_plus_the_cusp_pair` |

## Stated differences

- The random streams differ: the published script draws molecules and points from one numpy
  stream seeded 42; this repository draws points with its own seeded sampler. The measure and
  the sample size are the same; the sampled points are not.
- The published networks are both initialized from key 42; here the two networks use `seed`
  and `seed + 1`, and the registry's 3x16 entries zero the final layer.
- The correlation targets of the `paper` footing are formed on the published PW92 parameter
  set, while the model multiplies its enhancement factor by the repository's PW baseline, and
  the fidelity certificate evaluates the parent with libxc rather than from the file's
  targets. The published amplitudes move the PW92 correlation energy per electron by at most
  1.6e-7 relative at zero polarization and 1.3e-5 at full polarization, which on a system with
  0.30 Ha of correlation energy is 0.004 mHa against the certificate's tolerance of 1.0 mHa per
  atom, so the certificate remains the gate it was.
- The published exchange constant `mu = 0.2195149727645171` and the amplitude
  `A_0 = 0.031090690869654895` are literals in the published code, two ulp from the derived
  values `beta pi^2 / 3` and `(1 - ln 2) / pi^2` (5.6e-17 and 6.9e-18 apart, the spacing of a
  double being 2.8e-17 and 3.5e-18 there); the port carries the literals.
- Two regularizations of the network inputs are this tree's and not the published code's, both
  inactive on physical rows: the density is floored at 1e-12 before the coordinates are formed
  (the published transform forms `s` on the unfloored density with `1e-30` in the denominator),
  and the polarization is clipped into [-1, 1] before `zeta'` (the published transform does
  not clip). No pretraining row carries a density under the floor or a polarization outside
  the interval, so no number moves.
- The per-system exchange table of a `paper` file integrates the published total-density form,
  which for an open shell is not PBE's spin-scaled exchange energy; a per-system energy term is
  therefore refused at any positive weight under that footing, at the parser and in the run,
  and the table serves the pretraining record's fit diagnostics alone.
- The synthetic `(r_s, s, alpha)` mesh rows keep libxc's targets under every footing; they ride
  at weight zero under the published objective and reach no GGA architecture.
- A `paper` file's key set is the `total` footing's, so the manifest beside it is what names the
  footing; the pretraining run records the manifest's footing.
- Every existing checkpoint is unchanged: the defaults `ueg_gate: tanh2` and its recorded
  coordinates reproduce the recorded outputs of the two v7 fixtures exactly
  (`test_the_v7_checkpoints_load_and_reproduce_their_recorded_outputs`).
