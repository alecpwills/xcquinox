#!/usr/bin/env python3
"""Code notes of the annotated results deck: verbatim excerpts traced from the repository.

The annotated deck (``SLIDES_v7_2026-09-15_annotated.tex``) follows every slide with a notes
frame. For the slides that state equations, descriptors, architectures, the training pool, the
subset metric, the pre-training objective, the fidelity certificate, the training objective, the
solver and the held-out metrics, the notes are followed by the code that implements each claim:
the code appendix of the annotated deck (``SLIDES_v7_2026-09-15_code_appendix.tex``, one
``allowframebreaks`` frame per manifest row, labelled ``code:<slug>``), which the notes point
into by page reference; the plain deck inputs neither the notes nor the appendix.

``MANIFEST`` is the trace: for each row the slide page (of the plain deck), the order on that
page, a title, the repository file, the 1-based inclusive line range, the tokens the claim
rests on and the anchor, the verbatim first line of the range. A row is written only when its
range lies inside the file, starts on its anchor and every token occurs on some line of the
excerpt; a claim that its block does not carry is a defect and is refused, and so is a range the
source has shifted from under (a row written one line late keeps every
token inside the range while its def line is lost: the anchor is what refuses that). The excerpt
files are tracked; a later code edit dates them, and ``--write`` regenerates them against the
current source (the anchor refuses a range that no longer starts on its first line).

Usage, from the repository root, after a change to any excerpted file::

    python3 tools/analysis/slide_code_notes.py --write

writes ``reports/v7/slides_code/*.txt`` and the appendix file, and prints the pointer
line of every manifest page (the sentence that closes that page's notes in
``SLIDES_v7_2026-09-15_frames.tex``); without ``--write`` it only prints.
Standard library only.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable, List, NamedTuple, Optional, Sequence, Tuple


class Entry(NamedTuple):
    page: int          # page of the plain deck (the N-th frame of the frames file)
    order: int         # position among the page's code notes, from 1
    title: str         # what the block implements, as the frame title
    path: str          # repository-relative file
    first: int         # first line, 1-based inclusive
    last: int          # last line, inclusive
    tokens: Tuple[str, ...]   # each must occur on one line of the excerpt
    head: Optional[str] = None   # the anchor: the verbatim first line of the excerpt; a range
                                 # that no longer starts on it is refused, so a source that
                                 # gained or lost a line above the range cannot be written one
                                 # line off (None only for the synthetic rows of the tests)


# The frames file holds this many \begin{frame} lines; page N of the plain deck is the N-th
# (37 main and 8 backup frames).
FRAME_COUNT = 45

# The rows: page, order, title, path, first, last, tokens, head.
MANIFEST: Tuple[Entry, ...] = (
    Entry(5, 1, "k_F, s and the density floor (exchange network)",
          "xcquinox/pipeline/networks.py", 319, 324,
          ("lower_rho_cutoff", "k_F = (3 * jnp.pi**2 * rho) ** (1 / 3)", "s = jnp.sqrt(sigma) / (2 * k_F * rho)", ),
          head="        rho = jnp.maximum(rho, self.lower_rho_cutoff)"),
    Entry(5, 2, "r_s, k_F, s (correlation network)",
          "xcquinox/pipeline/networks.py", 589, 597,
          ("rs = (3 / (4 * jnp.pi * rho)) ** (1 / 3)", ),
          head="        rho = jnp.maximum(rho, self.lower_rho_cutoff)"),
    Entry(5, 3, "the DFS coordinates x_0, x_1, x_s (correlation network)",
          "xcquinox/pipeline/networks.py", 626, 654,
          ("spinscale", "_DFS_LOG_EPS", "_dfs_log_transform(s)", "_dfs_indicator_coordinate", ),
          head="        if self.descriptor_coordinates == \"dfs\":"),
    Entry(5, 4, "the exchange coordinate x_s",
          "xcquinox/pipeline/networks.py", 341, 352,
          ("_dfs_log_transform(s)", ),
          head="        if self.descriptor_coordinates == \"dfs\":"),
    Entry(5, 5, "the log transform and the 1e-5 constant",
          "xcquinox/pipeline/networks.py", 27, 34,
          ("_DFS_LOG_EPS = 1e-5", "jnp.log(x + 1.0)", ),
          head="_DFS_LOG_EPS = 1e-5"),
    Entry(5, 6, "the meta-GGA coordinate ln((alpha + 1)/2)",
          "xcquinox/pipeline/networks.py", 48, 80,
          ("(alpha + 1)", ),
          head="def _dfs_indicator_coordinate(alpha_raw):"),
    Entry(5, 7, "the exchange network on the doubled spin channel",
          "xcquinox/pipeline/oneshot.py", 465, 537,
          ("2.0 * rho_a, 4.0 * sigma_aa", "2.0 * rho_b, 4.0 * sigma_bb", ),
          head="def split_exc_energy_uks(model, rho_a, rho_b, sigma_aa, sigma_bb,"),
    Entry(6, 1, "tau from the live density matrix",
          "xcquinox/pipeline/metagga.py", 115, 139,
          ("einsum", ),
          head="def compute_tau_from_dm(ao_grad, dm) -> jnp.ndarray:"),
    Entry(6, 2, "the smooth positive part",
          "xcquinox/pipeline/metagga.py", 142, 152,
          ("jnp.sqrt(x * x + width * width)", ),
          head="def smooth_positive_part(x, width):"),
    Entry(6, 3, "alpha = (tau - tau_W) / tau_unif",
          "xcquinox/pipeline/metagga.py", 163, 274,
          ("tau_w", "tau_unif", "smooth_positive_part", ),
          head="def compute_alpha(rho, sigma, tau) -> jnp.ndarray:"),
    Entry(6, 4, "the alpha descriptor of the network input",
          "xcquinox/pipeline/descriptors.py", 432, 472,
          ("compute_alpha", ),
          head="@register_descriptor(\"metagga\")"),
    Entry(6, 5, "the cusp pair x_4, x_5",
          "xcquinox/pipeline/descriptors.py", 225, 258,
          ("exp(-2 Z_nearest r_min)", "tanh", ),
          head="@register_descriptor(\"cusp\")"),
    Entry(8, 1, "the bounded map L_lambda",
          "xcquinox/pipeline/networks.py", 129, 169,
          ("jax.nn.sigmoid(x - jnp.log(self.limit - 1.0)) - 1.0", ),
          head="class _AlecLOB(eqx.Module):"),
    Entry(7, 1, "the MLP, GELU, the attention block after the first hidden layer, the gate and the output",
          "xcquinox/pipeline/networks.py", 366, 405,
          ("jnp.tanh(s) ** 2", "jax.nn.gelu", "if i == 0", "self.attention(x)", "lobterm = self.lobf(gated)", ),
          head="        if self.meta_gga:"),
    Entry(7, 2, "the network construction and the zero-initialized last layer",
          "xcquinox/pipeline/networks.py", 283, 300,
          ("zero_init_final_layer", "jnp.zeros_like(self.net.layers[-1].weight)", ),
          head="        self.net = eqx.nn.MLP("),
    Entry(8, 2, "the exchange and correlation energy densities (LDA and PW92 factors)",
          "xcquinox/pipeline/models.py", 175, 215,
          ("lda_x", "pw92", ),
          head="    def eval_exc(self, rho, sigma, features, zeta=0.0):"),
    Entry(8, 3, "the spin-scaled exchange energy (Oliver and Perdew)",
          "xcquinox/pipeline/oneshot.py", 465, 537,
          ("2.0 * rho_a, 4.0 * sigma_aa", ),
          head="def split_exc_energy_uks(model, rho_a, rho_b, sigma_aa, sigma_bb,"),
    Entry(7, 3, "the self-attention block",
          "xcquinox/net.py", 50, 136,
          ("softmax", "sqrt", "LayerNorm", "return out + residual", ),
          head="class SelfAttentionBlock(eqx.Module):"),
    Entry(9, 1, "the architecture record",
          "xcquinox/pipeline/config.py", 103, 159,
          ("depth: int", "nodes: int", "num_heads: int = 1", "zero_init_final_layer: bool = False", ),
          head="@dataclass(frozen=True)"),
    Entry(9, 2, "the registry entries shallow, shallow_attn, medium and medium_attn (shown "
                "deep_2x8, deep_attn_2x8, deep_3x16, deep_attn_3x16)",
          "xcquinox/pipeline/config.py", 512, 515,
          ("\"shallow\"", "\"medium\"", "num_heads=2", "num_heads=4", ),
          head="    \"shallow\":             ArchitectureConfig(name=\"shallow\",      depth=2, nodes=8),"),
    Entry(9, 3, "the registry entries deep_3x16, deep_attn_3x16, deep_cusp_3x16 (shown deep0_*)",
          "xcquinox/pipeline/config.py", 579, 592,
          ("zero_init_final_layer=True", "num_heads=4", ),
          head="    \"deep_3x16\":                ArchitectureConfig.from_spec(\"deep_3x16\",               3, 16,"),
    Entry(9, 4, "the registry entry deep_cusp_mgga_3x16",
          "xcquinox/pipeline/config.py", 690, 694,
          ("deep_cusp_mgga_3x16", ),
          head="    \"deep_cusp_mgga_3x16\":      ArchitectureConfig.from_spec(\"deep_cusp_mgga_3x16\",     3, 16,"),
    Entry(9, 5, "from_spec: the fields an entry sets",
          "xcquinox/pipeline/config.py", 385, 470,
          ("zero_init_final_layer", ),
          head="    @classmethod"),
    Entry(10, 1, "the 21 atomization points (names, spins, charges, references)",
          "xcquinox/pipeline/dfs_pool.py", 106, 225,
          ("\"H2O\"", "\"spin\": 2", ),
          head="DFS_AE_DATA = ["),
    Entry(10, 2, "the three BH76 reaction points",
          "xcquinox/pipeline/dfs_pool.py", 308, 397,
          ("OH+N2_to_H+N2O", "OH+CH3_to_O+CH4", "HF+F_to_H+F2", ),
          head="DFS_BH76_REACTIONS = ["),
    Entry(10, 3, "the two ionization points",
          "xcquinox/pipeline/dfs_pool.py", 427, 466,
          ("Li_IP", "C_IP", ),
          head="DFS_IP13_PAIRS = ["),
    Entry(10, 4, "the H and Li anchors",
          "xcquinox/pipeline/dfs_pool.py", 472, 477,
          ("\"H\"", "\"Li\"", ),
          head="DFS_ATOM_REFS = ["),
    Entry(10, 5, "the pool assembly",
          "xcquinox/pipeline/dfs_pool.py", 547, 639,
          ("DFS_AE_DATA", "DFS_BH76_REACTIONS", "DFS_IP13_PAIRS", ),
          head="def build_dfs_pool() -> dict:"),
    Entry(10, 6, "the 26 points and their species",
          "xcquinox/pipeline/training_points.py", 347, 436,
          ("21 AE + 3 BH76 + 2 IP13", ),
          head="def build_dfs_pool_points("),
    Entry(11, 1, "the descriptor triple (rho^(1/3), s, alpha) and the clip to [0, 100]",
          "xcquinox/pipeline/subset_selection.py", 81, 119,
          ("rho_third", "kf_factor", "np.clip", "100.0", ),
          head="def compute_descriptor_triple("),
    Entry(11, 2, "one PBE SCF per species at def2-svp, grid level 1",
          "xcquinox/pipeline/subset_selection.py", 375, 406,
          ("def2-svp", "grid_level: int = 1", ),
          head="def extract_descriptors_for_species("),
    Entry(11, 3, "a point's sample is the concatenation over its species",
          "xcquinox/pipeline/subset_selection.py", 409, 445,
          ("concatenate", ),
          head="def concatenate_point_descriptors(points, species_descriptors: dict[tuple, dict]) -> list[dict]:"),
    Entry(11, 4, "the reference histograms: 200 bins between the 0.1 and 99.9 percentiles",
          "xcquinox/pipeline/subset_selection.py", 448, 469,
          ("np.percentile(full[k], [0.1, 99.9])", "200", ),
          head="def build_reference_histograms(pool):"),
    Entry(11, 5, "binning with the shared edges and the mass function",
          "xcquinox/pipeline/subset_selection.py", 175, 193,
          ("def _to_pmf", ),
          head="def _to_pmf(h: np.ndarray) -> np.ndarray:"),
    Entry(11, 6, "binning with the shared edges",
          "xcquinox/pipeline/subset_selection.py", 248, 259,
          ("np.histogram", ),
          head="def _bin_with_edges(arrs: dict, edges: dict) -> dict:"),
    Entry(12, 1, "the probability floor 1e-12",
          "xcquinox/pipeline/subset_selection.py", 54, 54,
          ("KL_PROB_CLIP = 1e-12", ),
          head="KL_PROB_CLIP = 1e-12"),
    Entry(12, 2, "the Kullback-Leibler term",
          "xcquinox/pipeline/subset_selection.py", 196, 210,
          ("np.log", ),
          head="def _kl(p: np.ndarray, q: np.ndarray) -> float:"),
    Entry(12, 3, "the Jensen-Shannon divergence over the three marginals",
          "xcquinox/pipeline/subset_selection.py", 213, 245,
          ("0.5 * (p + q)", "_kl(p, m)", "return float(\"inf\")", ),
          head="def metric_jsd(h_ref: dict, h_cand: dict, weights=None) -> float:"),
    Entry(13, 1, "the exhaustive search over C(26, r)",
          "xcquinox/pipeline/subset_selection.py", 472, 688,
          ("combinations", ),
          head="def select_subset("),
    Entry(15, 1, "the pre-training systems: the DFS inventory and the pool atoms",
          "xcquinox/pipeline/pretrain_data_gen.py", 334, 363,
          ("dfs_set", "pool_atoms", ),
          head="def resolve_pretrain_systems(*, atoms=None, dfs_set=False, pool_atoms=False,"),
    Entry(15, 2, "the pool atoms",
          "xcquinox/pipeline/pretrain_data_gen.py", 214, 240,
          ("BH76", "W4-11", "load_full_held_out_pools", ),
          head="def pool_atom_systems():"),
    Entry(15, 3, "the exchange rows per spin channel at the doubled density",
          "xcquinox/pipeline/pretrain_data_gen.py", 379, 479,
          ("2.0 * rho_gga_s[0]", ),
          head="def spin_channel_exchange_rows(mol, mf, ao, dm_ab, *, descriptors=True,"),
    Entry(15, 4, "the pointwise targets F_parent - 1",
          "xcquinox/pipeline/pretrain_data_gen.py", 981, 1044,
          ("lda", ),
          head="def _x_block_lda(block):"),
    Entry(15, 5, "the synthetic (r_s, s, alpha) mesh at 30 percent of the weight",
          "xcquinox/pipeline/pretrain_data_gen.py", 1068, 1140,
          ("MESH_WEIGHT_FRACTION = 0.3", "MESH_RS", "MESH_ALPHA", ),
          head="MESH_RS = (0.1, 0.3, 0.7, 1.5, 3.0, 5.0, 10.0)"),
    Entry(15, 6, "the integration weights abs(n eps_LDA) w_grid",
          "xcquinox/pipeline/pretrain.py", 104, 167,
          ("eps_x_lda", "grid_weights", ),
          head="def _compute_integration_weights(rho, grid_weights=None):"),
    Entry(15, 7, "the objective: pointwise term plus energy term",
          "xcquinox/pipeline/pretrain.py", 256, 341,
          ("energy_weight", "jnp.sum(w * residual_sq) / (jnp.sum(w) + 1e-12)", ),
          head="class _PretrainLoss(eqx.Module):"),
    Entry(15, 8, "the learning-rate schedule",
          "xcquinox/pipeline/pretrain.py", 1132, 1207,
          ("lr_decay_start", "lr_end", ),
          head="def _lr_schedule("),
    Entry(15, 9, "Adam with the global-norm clip",
          "xcquinox/pipeline/pretrain.py", 1210, 1235,
          ("optax.clip_by_global_norm(grad_clip)", "optax.adam", ),
          head="def _build_optimizer("),
    Entry(15, 10, "the loop: validation every validate_every steps, patience, the best model kept",
          "xcquinox/pipeline/pretrain.py", 739, 825,
          ("step % every", "patience", "best_model", ),
          head="    @eqx.filter_jit"),
    Entry(16, 1, "dE_xc per system in mHa",
          "xcquinox/pipeline/cluster/fidelity.py", 1171, 1189,
          ("(e_xc_nn - e_xc_parent) * HA_TO_MHA", ),
          head="    return {"),
    Entry(16, 2, "dAE = dE_xc(mol) - sum of the atoms' dE_xc",
          "xcquinox/pipeline/cluster/fidelity.py", 1411, 1460,
          ("d_ae_mha = ok[mol_spec.name][\"dE_xc_mHa\"] - sum(atom_terms)", "HA_TO_KCAL", ),
          head="    per_atomization = []"),
    Entry(16, 3, "the atomization gate: mean and the max backstop",
          "xcquinox/pipeline/cluster/fidelity.py", 1616, 1687,
          ("tol_AE", "tol_AE_max_backstop", ),
          head="def _ae_gate_terms(per_atomization, fid_cfg):"),
    Entry(16, 4, "the tolerances 1.0 mHa, 1.0 and 2.0 kcal/mol",
          "xcquinox/pipeline/cluster/grid_config.py", 290, 338,
          ("tol_AE: float = 1.0", "tol_atom: float = 1.0", "tol_AE_max_backstop", ),
          head="@dataclass(frozen=True)"),
    Entry(16, 5, "the gate that holds the training array",
          "xcquinox/pipeline/cluster/fidelity.py", 292, 320,
          ("PASS", ),
          head="def gate_certificate(run_dir: str, arch: str) -> tuple[bool, str]:"),
    Entry(21, 1, "the five channels and their assembly",
          "xcquinox/pipeline/losses.py", 1336, 1417,
          ("loss_AE", "loss_BH76", "loss_IP13", "loss_vxc", "loss_rho", "step_w2 = step_w ** 2", ),
          head="    def compute_components(self, model, batch, relative=False):"),
    Entry(21, 2, "the fixed channel weights 1, 1, 1, 1, 20",
          "xcquinox/pipeline/train.py", 1829, 1835,
          ("\"loss_rho\": 20.0", ),
          head="_DEFAULT_CHANNEL_WEIGHTS = {"),
    Entry(21, 3, "the reaction residual (BH76 barriers, W4-11 atomizations as reactions)",
          "xcquinox/pipeline/losses.py", 530, 578,
          ("jnp.mean(step_w2 * (e_rxn - e_rxn_ref) ** 2)", ),
          head="def _rxn_residual_term("),
    Entry(21, 4, "the BH76 channel",
          "xcquinox/pipeline/losses.py", 1282, 1310,
          ("_rxn_residual_term", ),
          head="    def _bh76_channel(self, E_nn, relative=False, step_w2=None) -> jnp.ndarray:"),
    Entry(21, 5, "the ionization channel",
          "xcquinox/pipeline/losses.py", 581, 605,
          ("e_cation - e_neutral - ip_ref", ),
          head="def _ip_residual_term("),
    Entry(21, 6, "the atom anchors: relative squared error at weight 0.01",
          "xcquinox/pipeline/losses.py", 223, 243,
          ("atom_energies[Z] ** 2", ),
          head="def _atomic_reg(E_nn, atom_mol_idx_dict, atom_energies, step_w2=None):"),
    Entry(21, 7, "the atomization channel with the network's own atoms",
          "xcquinox/pipeline/losses.py", 246, 273,
          ("_ae_from_atoms", ),
          head="def _ae_losses(E_nn, compound_idx, comp_dicts, mol_names, targets, atom_energies,"),
    Entry(21, 8, "the potential channel: the Frobenius residual over n_AO^2",
          "xcquinox/pipeline/losses.py", 408, 483,
          ("n_ao", ),
          head="def _vxc_term(model, mol_data, iter_idx, relative=False):"),
    Entry(21, 9, "the density channel per electron",
          "xcquinox/pipeline/losses.py", 366, 405,
          ("n_e ** 2", ),
          head="def _grid_term(model, mol_data, iter_idx, solver_config=None, relative=False,"),
    Entry(21, 10, "the convergence-tail weights (t/(N-1))^2",
          "xcquinox/pipeline/oneshot.py", 632, 654,
          ("np.linspace(0.0, 1.0, n) ** p", ),
          head="def scf_tail_window(n_cycles, tail, power):"),
    Entry(21, 11, "one optimizer step per training group per epoch",
          "xcquinox/pipeline/train.py", 2001, 2061,
          ("ONE optimizer step per group", ),
          head="def _run_per_molecule_loop(spec, model, batch, loss, progress_callback):"),
    Entry(21, 12, "the group's scoped loss",
          "xcquinox/pipeline/train.py", 1919, 1965,
          ("bh76_reactions", "ip13_pairs", ),
          head="def _build_group_loss_and_batch(spec: TrainingSpec, group: dict, batch: dict):"),
    Entry(22, 1, "the SCF: three cycles, the DFS mixing schedule, the tail loss",
          "hpcjobs/configs/dfs_step7.dfs6311_grid3_v7g1_size.yaml", 131, 140,
          ("max_cycles: 3", "decaying_linear", "scf_loss_tail", ),
          head="solvers:"),
    Entry(22, 2, "the mixing schedule a_t = 0.3^t + 0.3",
          "xcquinox/pipeline/solver.py", 303, 358,
          ("base**step + floor", ),
          head="@register_mixer"),
    Entry(22, 3, "the PBE seed",
          "xcquinox/pipeline/solver.py", 88, 125,
          ("seed_source: str = \"pbe\"", ),
          head="    # SCF. \"pbe\" (default) -> the converged PBE dm from precompute"),
    Entry(22, 4, "the SCF cycle: the mixed density scored, gradients through every cycle",
          "xcquinox/pipeline/solver_manual.py", 404, 463,
          ("D_mixed", "freeze_on_convergence", ),
          head="    def body(state, _):"),
    Entry(22, 5, "the cycles as a scan (gradients through every cycle)",
          "xcquinox/pipeline/solver_manual.py", 263, 289,
          ("jax.lax.scan", ),
          head="def _iterate_scf(config: SolverConfig, body, init_state, forward_only: bool):"),
    Entry(22, 6, "AdamW, the linear decay over the second half, weight decay, the clip",
          "xcquinox/pipeline/train.py", 112, 202,
          ("optax.adamw", "clip_by_global_norm", ),
          head="def build_optimizer("),
    Entry(23, 1, "the hyperparameters of the run (200 epochs, 1e-3 to 1e-5, weight decay 1e-4, clip 1.0, validation every 25)",
          "hpcjobs/configs/dfs_step7.dfs6311_grid3_v7g1_size.yaml", 153, 183,
          ("n_steps: 200", "weight_decay: 0.0001", "validate_every: 25", "patience: 5", ),
          head="hyperparams:"),
    Entry(23, 2, "the validation slice",
          "xcquinox/pipeline/train.py", 745, 817,
          ("validation_molecules", "validation_reactions_path", ),
          head="def _build_validation_data(spec):"),
    Entry(23, 3, "the validation reaction-energy MAE",
          "xcquinox/pipeline/train.py", 820, 862,
          ("kcal", ),
          head="def _validation_reaction_mae(model, val_mol_data, val_reactions,"),
    Entry(23, 4, "the validation-best tracker",
          "xcquinox/pipeline/train.py", 682, 742,
          ("min_delta", "patience", ),
          head="class _BestValidationTracker:"),
    Entry(23, 5, "the validation-best checkpoint among the three saved",
          "xcquinox/pipeline/train.py", 1017, 1052,
          ("model_best.eqx", "model_val_best.eqx", ),
          head="def _save_artifacts(spec, model, losses, aux_log, duration, best_model=None,"),
    Entry(27, 1, "the full pools (76 BH76, 140 W4-11) before the validation split",
          "xcquinox/pipeline/full_benchmark_pools.py", 523, 550,
          ("76 + 140 = 216", ),
          head="def load_full_held_out_pools("),
    Entry(27, 2, "the validation slice, split by reaction identity",
          "xcquinox/pipeline/eval_holdout.py", 204, 245,
          ("reaction_identity_key", "hashlib", ),
          head="def reaction_identity_key(rxn: Dict[str, Any]) -> str:"),
    Entry(27, 3, "the strict held-out filter",
          "xcquinox/pipeline/eval_holdout.py", 173, 201,
          ("strict", ),
          head="def filter_reactions("),
    Entry(27, 4, "the cell's trained reactions removed",
          "xcquinox/pipeline/eval_holdout.py", 283, 338,
          ("trained", ),
          head="def trained_reaction_exclusion(training_spec, pool_specs"),
    Entry(27, 5, "the energy legs average one term per reaction identity",
          "xcquinox/pipeline/eval_holdout.py", 402, 442,
          ("61 rows over 54", ),
          head="def reaction_mae_kcalmol("),
    Entry(27, 6, "eps_n per electron",
          "xcquinox/pipeline/evaluation.py", 196, 218,
          ("jnp.abs(rho - rho_ref)) / n_e", ),
          head="def density_eps_terms(rho, rho_ref, w):"),
    Entry(27, 7, "the grid RMSE",
          "xcquinox/pipeline/evaluation.py", 293, 320,
          ("jnp.sqrt(jnp.sum(w * diff ** 2) / jnp.sum(w))", ),
          head="        w = mol_data[\"grid_weights\"]"),
    Entry(27, 8, "WTMAD-2 with the GMTKN55 scale",
          "tools/analysis/make_ablation_arch_figure.py", 3827, 3860,
          ("56.84", ),
          head="_GMTKN55_SCALE = 56.84  # kcal/mol, global mean |dE| over GMTKN55 (Goerigk 2017)"),
    Entry(27, 9, "the harmonic mean",
          "tools/analysis/make_ablation_arch_figure.py", 4838, 4844,
          ("2.0 / (1.0 / a + 1.0 / b)", ),
          head="def _harmonic_mean(a: float, b: float) -> float:"),
    Entry(27, 10, "the self-calibrated gamma = E_PBE / D_PBE",
          "tools/analysis/make_ablation_arch_figure.py", 4871, 4946,
          ("gamma = float(e_pbe) / float(d_pbe)", ),
          head="def combined_ed_by_cell(energy_by_cell: Dict[Tuple[str, int], float],"),
    Entry(27, 11, "the DFS gamma 1084.87",
          "tools/analysis/make_ablation_arch_figure.py", 4955, 4978,
          ("_DFS_GAMMA_KCAL = 1084.87", ),
          head="_DFS_GAMMA_KCAL = 1084.87"),
)

PAGES: Tuple[int, ...] = tuple(sorted({e.page for e in MANIFEST}))

_SLUG_TITLE_MAX = 40
_OUTDIR_NAME = "slides_code"


def _entries(entries: Optional[Iterable[Entry]]) -> Tuple[Entry, ...]:
    """The rows to work on: the given ones, or the manifest (read at call time)."""
    return tuple(entries) if entries is not None else MANIFEST


def slug(entry: Entry) -> str:
    """``p<page>_<order>_<title>``: the title lowercased, runs of anything but ``[a-z0-9]``
    joined by ``_``, the title part cut at 40 characters."""
    title = re.sub(r"[^a-z0-9]+", "_", entry.title.lower()).strip("_")
    title = title[:_SLUG_TITLE_MAX].rstrip("_")
    return f"p{entry.page}_{entry.order}_{title}"


def _label(entry: Entry) -> str:
    return f"p{entry.page}.{entry.order} ({entry.path}:{entry.first}-{entry.last})"


def excerpt_lines(root: Path, entry: Entry) -> List[str]:
    """The verbatim lines ``first..last`` of ``root/entry.path``.

    Refused with ``ValueError`` when the range starts before line 1, is inverted, runs past the
    end of the file, does not start on the row's anchor, or when a token occurs on no line of
    the excerpt.
    """
    source = Path(root) / entry.path
    if not source.is_file():
        raise ValueError(f"{_label(entry)}: {source} is not a file")
    lines = source.read_text(encoding="utf-8").splitlines()
    if entry.first < 1:
        raise ValueError(f"{_label(entry)}: first line {entry.first} is below 1")
    if entry.last < entry.first:
        raise ValueError(f"{_label(entry)}: last line {entry.last} is before first {entry.first}")
    if entry.last > len(lines):
        raise ValueError(
            f"{_label(entry)}: last line {entry.last} is past the end ({len(lines)} lines)")
    excerpt = lines[entry.first - 1:entry.last]
    if entry.head is not None and excerpt[0] != entry.head:
        raise ValueError(
            f"{_label(entry)}: the range starts on {excerpt[0]!r}, not on its anchor "
            f"{entry.head!r}: the source has shifted above the range")
    joined = "\n".join(excerpt)
    for token in entry.tokens:
        if token not in joined:
            raise ValueError(f"{_label(entry)}: token {token!r} is absent from the excerpt")
    return excerpt


def header(entry: Entry) -> str:
    """The first line of an excerpt file: the file and the range."""
    return f"# {entry.path}:{entry.first}-{entry.last}"


def write_excerpts(root: Path, outdir: Path,
                   entries: Optional[Iterable[Entry]] = None) -> List[Path]:
    """Write ``outdir/<slug>.txt`` (header plus the verbatim lines) for every entry.

    Every entry is validated before the first file is written, so a refused row leaves the
    directory as it was; after a successful write, excerpt files (``p*.txt``) that no entry
    produces are removed.
    """
    rows = _entries(entries)
    prepared = [(slug(e), header(e), excerpt_lines(root, e)) for e in rows]
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []
    for name, head, lines in prepared:
        target = outdir / f"{name}.txt"
        target.write_text("\n".join([head, *lines]) + "\n", encoding="utf-8")
        written.append(target)
    keep = {p.name for p in written}
    for stale in outdir.glob("p*.txt"):
        if stale.name not in keep:
            stale.unlink()
    return written


_ESCAPES = (
    ("\\", "\\textbackslash{}"),
    ("{", "\\{"), ("}", "\\}"),
    ("_", "\\_"), ("&", "\\&"), ("%", "\\%"), ("#", "\\#"), ("$", "\\$"),
    ("~", "\\~{}"), ("^", "\\^{}"),
    ("|", "$|$"), ("<", "$<$"), (">", "$>$"),
)


def latex_escape(text: str) -> str:
    """The text as LaTeX prose (the frame title and the file citation)."""
    out = text
    for raw, esc in _ESCAPES:
        out = out.replace(raw, esc)
    return out


APPENDIX_NAME = "SLIDES_v7_2026-09-15_code_appendix.tex"


def code_label(entry: Entry) -> str:
    """The LaTeX label of a row's appendix frame, ``code:<slug>``, the target of the page
    references in the notes."""
    return f"code:{slug(entry)}"


def appendix_lines(entries: Optional[Iterable[Entry]] = None,
                   outdir_name: str = _OUTDIR_NAME) -> List[str]:
    """The code appendix of the annotated deck: a section heading, then one
    ``allowframebreaks`` frame per manifest row in (page, order) order, titled by the row's
    page, order, title and file range, labelled by :func:`code_label` and setting the row's
    excerpt verbatim. One frame per row is an invariant the pointer lines rest on: a page's
    rows are contiguous in the appendix, so its first and last labels bound its pages."""
    out = ["\\section*{Appendix: code excerpts}"]
    for e in sorted(_entries(entries), key=lambda e: (e.page, e.order)):
        out.extend([
            f"\\begin{{frame}}[allowframebreaks]{{Code p{e.page}.{e.order}: "
            f"{latex_escape(e.title)} [{latex_escape(e.path)}:{e.first}--{e.last}]}}",
            f"\\label{{{code_label(e)}}}",
            f"\\VerbatimInput[fontsize=\\tiny]{{{outdir_name}/{slug(e)}.txt}}",
            "\\end{frame}",
        ])
    return out


def pointer_line(page: int, entries: Optional[Iterable[Entry]] = None) -> str:
    """The sentence that closes the notes of a manifest page: the appendix page of the page's
    one excerpt, or the range from its first to its last; a page with no row is refused."""
    rows = sorted((e for e in _entries(entries) if e.page == page), key=lambda e: e.order)
    if not rows:
        raise ValueError(f"page {page} has no manifest row, so no appendix to point at")
    if len(rows) == 1:
        return f"Code: appendix p. \\pageref{{{code_label(rows[0])}}}."
    return (f"Code: appendix pp. \\pageref{{{code_label(rows[0])}}} to "
            f"\\pageref{{{code_label(rows[-1])}}}.")


def write_appendix(path, entries: Optional[Iterable[Entry]] = None) -> Path:
    """Write the appendix file (:func:`appendix_lines`, one line each)."""
    target = Path(path)
    target.write_text("\n".join(appendix_lines(entries)) + "\n", encoding="utf-8")
    return target


def main(argv: Optional[Sequence[str]] = None) -> int:
    here = Path(__file__).resolve().parent
    reports = here.parents[1] / "reports" / "v7"
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true",
                    help="write the excerpt files and the appendix (default: only print the pointer lines)")
    ap.add_argument("--root", type=Path, default=here.parents[1],
                    help="repository root the manifest paths are relative to")
    ap.add_argument("--outdir", type=Path, default=reports / _OUTDIR_NAME,
                    help="directory of the excerpt files (default: beside the deck under reports/v7)")
    ap.add_argument("--appendix", type=Path, default=reports / APPENDIX_NAME,
                    help="the appendix file (default: the tracked one beside the deck, "
                         "which --write rewrites)")
    args = ap.parse_args(argv)
    if args.write:
        written = write_excerpts(args.root, args.outdir)
        print(f"% {len(written)} excerpts written to {args.outdir}")
        print(f"% appendix written to {write_appendix(args.appendix)}")
    for page in PAGES:
        print(f"% page {page}")
        print(pointer_line(page))
    return 0


if __name__ == "__main__":
    sys.exit(main())
