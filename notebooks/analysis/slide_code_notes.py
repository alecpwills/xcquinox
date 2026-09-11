#!/usr/bin/env python3
"""Code notes of the annotated results deck: verbatim excerpts traced from the repository.

The annotated deck (``SLIDES_v7_2026-09-09_annotated.tex``) follows every slide with a notes
frame. For the slides that state equations, descriptors, architectures, the training pool, the
subset metric, the pre-training objective, the fidelity certificate, the training objective, the
solver and the held-out metrics, the notes are followed by the code that implements each claim:
one ``\\codenote{<title> [<path>:<first>--<last>]}{slides_code/<slug>.txt}`` frame per manifest
row, rendered verbatim by the annotated wrapper and dropped by the plain one.

``MANIFEST`` is the trace: for each row the slide page (of the plain deck), the order on that
page, a title, the repository file, the 1-based inclusive line range and the tokens the claim
rests on. A row is written only when its range lies inside the file and every token occurs on
some line of the excerpt; a claim that its block does not carry is a defect and is refused. The
excerpt files are tracked, and ``test_slide_code_notes.py`` compares each against the current
source, so a later code edit fails the test instead of silently dating the deck.

Usage, from the repository root, after a change to any excerpted file::

    python3 notebooks/analysis/slide_code_notes.py --write

writes ``notebooks/analysis/slides_code/*.txt`` and prints the ``\\codenote`` lines per page (the
lines already placed in ``SLIDES_v7_2026-09-09_frames.tex``); without ``--write`` it only prints.
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


# The frames file holds this many \begin{frame} lines; page N of the plain deck is the N-th.
FRAME_COUNT = 24

# The rows: page, order, title, path, first, last, tokens. Five rows differ from the first
# draft of this manifest, corrected against the code on 2026-09-10 (p5.2, p11.2, p11.6, p16.2,
# p16.8: seven tokens and one title that did not occur in their ranges as written).
MANIFEST: Tuple[Entry, ...] = (
    Entry(3, 1, "k_F, s and the density floor (exchange network)",
          "xcquinox/alec/networks.py", 319, 324,
          ("lower_rho_cutoff", "k_F = (3 * jnp.pi**2 * rho) ** (1 / 3)", "s = jnp.sqrt(sigma) / (2 * k_F * rho)", )),
    Entry(3, 2, "r_s, k_F, s (correlation network)",
          "xcquinox/alec/networks.py", 589, 597,
          ("rs = (3 / (4 * jnp.pi * rho)) ** (1 / 3)", )),
    Entry(3, 3, "the DFS coordinates x_0, x_1, x_s (correlation network)",
          "xcquinox/alec/networks.py", 626, 654,
          ("spinscale", "_DFS_LOG_EPS", "_dfs_log_transform(s)", "_dfs_indicator_coordinate", )),
    Entry(3, 4, "the exchange coordinate x_s",
          "xcquinox/alec/networks.py", 341, 352,
          ("_dfs_log_transform(s)", )),
    Entry(3, 5, "the log transform and the 1e-5 constant",
          "xcquinox/alec/networks.py", 27, 34,
          ("_DFS_LOG_EPS = 1e-5", "jnp.log(x + 1.0)", )),
    Entry(3, 6, "the meta-GGA coordinate ln((alpha + 1)/2)",
          "xcquinox/alec/networks.py", 48, 80,
          ("(alpha + 1)", )),
    Entry(3, 7, "the exchange network on the doubled spin channel",
          "xcquinox/alec/oneshot.py", 465, 537,
          ("2.0 * rho_a, 4.0 * sigma_aa", "2.0 * rho_b, 4.0 * sigma_bb", )),
    Entry(4, 1, "tau from the live density matrix",
          "xcquinox/alec/metagga.py", 115, 139,
          ("einsum", )),
    Entry(4, 2, "the smooth positive part",
          "xcquinox/alec/metagga.py", 142, 152,
          ("jnp.sqrt(x * x + width * width)", )),
    Entry(4, 3, "alpha = (tau - tau_W) / tau_unif",
          "xcquinox/alec/metagga.py", 163, 274,
          ("tau_w", "tau_unif", "smooth_positive_part", )),
    Entry(4, 4, "the alpha descriptor of the network input",
          "xcquinox/alec/descriptors.py", 432, 472,
          ("compute_alpha", )),
    Entry(4, 5, "the cusp pair x_4, x_5",
          "xcquinox/alec/descriptors.py", 225, 258,
          ("exp(-2 Z_nearest r_min)", "tanh", )),
    Entry(5, 1, "the bounded map L_lambda",
          "xcquinox/alec/networks.py", 129, 169,
          ("jax.nn.sigmoid(x - jnp.log(self.limit - 1.0)) - 1.0", )),
    Entry(5, 2, "the MLP, GELU, the attention block after the first hidden layer, the gate and the output",
          "xcquinox/alec/networks.py", 366, 405,
          ("jnp.tanh(s) ** 2", "jax.nn.gelu", "if i == 0", "self.attention(x)", "lobterm = self.lobf(gated)", )),
    Entry(5, 3, "the network construction and the zero-initialized last layer",
          "xcquinox/alec/networks.py", 283, 300,
          ("zero_init_final_layer", "jnp.zeros_like(self.net.layers[-1].weight)", )),
    Entry(5, 4, "the exchange and correlation energy densities (LDA and PW92 factors)",
          "xcquinox/alec/models.py", 175, 215,
          ("lda_x", "pw92", )),
    Entry(5, 5, "the spin-scaled exchange energy (Oliver and Perdew)",
          "xcquinox/alec/oneshot.py", 465, 537,
          ("2.0 * rho_a, 4.0 * sigma_aa", )),
    Entry(5, 6, "the self-attention block",
          "xcquinox/net.py", 50, 136,
          ("softmax", "sqrt", "LayerNorm", "return out + residual", )),
    Entry(6, 1, "the architecture record",
          "xcquinox/alec/config.py", 103, 159,
          ("depth: int", "nodes: int", "num_heads: int = 1", "zero_init_final_layer: bool = False", )),
    Entry(6, 2, "the registry entries medium and medium_attn (shown deep_3x16, deep_attn_3x16)",
          "xcquinox/alec/config.py", 512, 515,
          ("\"medium\"", "num_heads=4", )),
    Entry(6, 3, "the registry entries deep_3x16, deep_attn_3x16, deep_cusp_3x16 (shown deep0_*)",
          "xcquinox/alec/config.py", 579, 592,
          ("zero_init_final_layer=True", "num_heads=4", )),
    Entry(6, 4, "the registry entry deep_cusp_mgga_3x16",
          "xcquinox/alec/config.py", 670, 674,
          ("deep_cusp_mgga_3x16", )),
    Entry(6, 5, "from_spec: the fields an entry sets",
          "xcquinox/alec/config.py", 385, 470,
          ("zero_init_final_layer", )),
    Entry(7, 1, "the 21 atomization points (names, spins, charges, references)",
          "xcquinox/alec/dfs_pool.py", 106, 225,
          ("\"H2O\"", "\"spin\": 2", )),
    Entry(7, 2, "the three BH76 reaction points",
          "xcquinox/alec/dfs_pool.py", 308, 397,
          ("OH+N2_to_H+N2O", "OH+CH3_to_O+CH4", "HF+F_to_H+F2", )),
    Entry(7, 3, "the two ionization points",
          "xcquinox/alec/dfs_pool.py", 427, 466,
          ("Li_IP", "C_IP", )),
    Entry(7, 4, "the H and Li anchors",
          "xcquinox/alec/dfs_pool.py", 472, 477,
          ("\"H\"", "\"Li\"", )),
    Entry(7, 5, "the pool assembly",
          "xcquinox/alec/dfs_pool.py", 547, 639,
          ("DFS_AE_DATA", "DFS_BH76_REACTIONS", "DFS_IP13_PAIRS", )),
    Entry(7, 6, "the 26 points and their species",
          "xcquinox/alec/training_points.py", 347, 436,
          ("21 AE + 3 BH76 + 2 IP13", )),
    Entry(8, 1, "the descriptor triple (rho^(1/3), s, alpha) and the clip to [0, 100]",
          "xcquinox/alec/subset_selection.py", 81, 119,
          ("rho_third", "kf_factor", "np.clip", "100.0", )),
    Entry(8, 2, "one PBE SCF per species at def2-svp, grid level 1",
          "xcquinox/alec/subset_selection.py", 375, 406,
          ("def2-svp", "grid_level: int = 1", )),
    Entry(8, 3, "a point's sample is the concatenation over its species",
          "xcquinox/alec/subset_selection.py", 409, 445,
          ("concatenate", )),
    Entry(8, 4, "the reference histograms: 200 bins between the 0.1 and 99.9 percentiles",
          "xcquinox/alec/subset_selection.py", 448, 469,
          ("np.percentile(full[k], [0.1, 99.9])", "200", )),
    Entry(8, 5, "binning with the shared edges and the mass function",
          "xcquinox/alec/subset_selection.py", 175, 193,
          ("def _to_pmf", )),
    Entry(8, 6, "binning with the shared edges",
          "xcquinox/alec/subset_selection.py", 248, 259,
          ("np.histogram", )),
    Entry(9, 1, "the probability floor 1e-12",
          "xcquinox/alec/subset_selection.py", 54, 54,
          ("KL_PROB_CLIP = 1e-12", )),
    Entry(9, 2, "the Kullback-Leibler term",
          "xcquinox/alec/subset_selection.py", 196, 210,
          ("np.log", )),
    Entry(9, 3, "the Jensen-Shannon divergence over the three marginals",
          "xcquinox/alec/subset_selection.py", 213, 245,
          ("0.5 * (p + q)", "_kl(p, m)", "return float(\"inf\")", )),
    Entry(9, 4, "the exhaustive search over C(26, r)",
          "xcquinox/alec/subset_selection.py", 472, 688,
          ("combinations", )),
    Entry(11, 1, "the pre-training systems: the DFS inventory and the pool atoms",
          "xcquinox/alec/pretrain_data_gen.py", 334, 363,
          ("dfs_set", "pool_atoms", )),
    Entry(11, 2, "the pool atoms",
          "xcquinox/alec/pretrain_data_gen.py", 214, 240,
          ("BH76", "W4-11", "load_full_held_out_pools", )),
    Entry(11, 3, "the exchange rows per spin channel at the doubled density",
          "xcquinox/alec/pretrain_data_gen.py", 379, 479,
          ("2.0 * rho_gga_s[0]", )),
    Entry(11, 4, "the pointwise targets F_parent - 1",
          "xcquinox/alec/pretrain_data_gen.py", 981, 1044,
          ("lda", )),
    Entry(11, 5, "the synthetic (r_s, s, alpha) mesh at 30 percent of the weight",
          "xcquinox/alec/pretrain_data_gen.py", 1068, 1140,
          ("MESH_WEIGHT_FRACTION = 0.3", "MESH_RS", "MESH_ALPHA", )),
    Entry(11, 6, "the integration weights abs(n eps_LDA) w_grid",
          "xcquinox/alec/pretrain.py", 104, 167,
          ("eps_x_lda", "grid_weights", )),
    Entry(11, 7, "the objective: pointwise term plus energy term",
          "xcquinox/alec/pretrain.py", 256, 341,
          ("energy_weight", "jnp.sum(w * residual_sq) / (jnp.sum(w) + 1e-12)", )),
    Entry(11, 8, "the learning-rate schedule",
          "xcquinox/alec/pretrain.py", 1133, 1208,
          ("lr_decay_start", "lr_end", )),
    Entry(11, 9, "Adam with the global-norm clip",
          "xcquinox/alec/pretrain.py", 1211, 1236,
          ("optax.clip_by_global_norm(grad_clip)", "optax.adam", )),
    Entry(11, 10, "the loop: validation every validate_every steps, patience, the best model kept",
          "xcquinox/alec/pretrain.py", 740, 826,
          ("step % every", "patience", "best_model", )),
    Entry(12, 1, "dE_xc per system in mHa",
          "xcquinox/alec/cluster/fidelity.py", 1171, 1189,
          ("(e_xc_nn - e_xc_parent) * HA_TO_MHA", )),
    Entry(12, 2, "dAE = dE_xc(mol) - sum of the atoms' dE_xc",
          "xcquinox/alec/cluster/fidelity.py", 1411, 1460,
          ("d_ae_mha = ok[mol_spec.name][\"dE_xc_mHa\"] - sum(atom_terms)", "HA_TO_KCAL", )),
    Entry(12, 3, "the atomization gate: mean and the max backstop",
          "xcquinox/alec/cluster/fidelity.py", 1616, 1687,
          ("tol_AE", "tol_AE_max_backstop", )),
    Entry(12, 4, "the tolerances 1.0 mHa, 1.0 and 2.0 kcal/mol",
          "xcquinox/alec/cluster/grid_config.py", 290, 338,
          ("tol_AE: float = 1.0", "tol_atom: float = 1.0", "tol_AE_max_backstop", )),
    Entry(12, 5, "the gate that holds the training array",
          "xcquinox/alec/cluster/fidelity.py", 292, 320,
          ("PASS", )),
    Entry(15, 1, "the five channels and their assembly",
          "xcquinox/alec/losses.py", 1336, 1417,
          ("loss_AE", "loss_BH76", "loss_IP13", "loss_vxc", "loss_rho", "step_w2 = step_w ** 2", )),
    Entry(15, 2, "the fixed channel weights 1, 1, 1, 1, 20",
          "xcquinox/alec/train.py", 1829, 1835,
          ("\"loss_rho\": 20.0", )),
    Entry(15, 3, "the reaction residual (BH76 barriers, W4-11 atomizations as reactions)",
          "xcquinox/alec/losses.py", 530, 578,
          ("jnp.mean(step_w2 * (e_rxn - e_rxn_ref) ** 2)", )),
    Entry(15, 4, "the BH76 channel",
          "xcquinox/alec/losses.py", 1282, 1310,
          ("_rxn_residual_term", )),
    Entry(15, 5, "the ionization channel",
          "xcquinox/alec/losses.py", 581, 605,
          ("e_cation - e_neutral - ip_ref", )),
    Entry(15, 6, "the atom anchors: relative squared error at weight 0.01",
          "xcquinox/alec/losses.py", 223, 243,
          ("atom_energies[Z] ** 2", )),
    Entry(15, 7, "the atomization channel with the network's own atoms",
          "xcquinox/alec/losses.py", 246, 273,
          ("_ae_from_atoms", )),
    Entry(15, 8, "the potential channel: the Frobenius residual over n_AO^2",
          "xcquinox/alec/losses.py", 408, 483,
          ("n_ao", )),
    Entry(15, 9, "the density channel per electron",
          "xcquinox/alec/losses.py", 366, 405,
          ("n_e ** 2", )),
    Entry(15, 10, "the convergence-tail weights (t/(N-1))^2",
          "xcquinox/alec/oneshot.py", 632, 654,
          ("np.linspace(0.0, 1.0, n) ** p", )),
    Entry(15, 11, "one optimizer step per training group per epoch",
          "xcquinox/alec/train.py", 2001, 2061,
          ("ONE optimizer step per group", )),
    Entry(15, 12, "the group's scoped loss",
          "xcquinox/alec/train.py", 1919, 1965,
          ("bh76_reactions", "ip13_pairs", )),
    Entry(16, 1, "the SCF: three cycles, the DFS mixing schedule, the tail loss",
          "hpcjobs/configs/dfs_step7.dfs6311_grid3_v7g1_size.yaml", 131, 140,
          ("max_cycles: 3", "decaying_linear", "scf_loss_tail", )),
    Entry(16, 2, "the mixing schedule a_t = 0.3^t + 0.3",
          "xcquinox/alec/solver.py", 303, 358,
          ("base**step + floor", )),
    Entry(16, 3, "the PBE seed",
          "xcquinox/alec/solver.py", 88, 125,
          ("seed_source: str = \"pbe\"", )),
    Entry(16, 4, "the SCF cycle: the mixed density scored, gradients through every cycle",
          "xcquinox/alec/solver_manual.py", 404, 463,
          ("D_mixed", "freeze_on_convergence", )),
    Entry(16, 5, "the cycles as a scan (gradients through every cycle)",
          "xcquinox/alec/solver_manual.py", 263, 289,
          ("jax.lax.scan", )),
    Entry(16, 6, "AdamW, the linear decay over the second half, weight decay, the clip",
          "xcquinox/alec/train.py", 112, 202,
          ("optax.adamw", "clip_by_global_norm", )),
    Entry(16, 7, "the hyperparameters of the run (200 epochs, 1e-3 to 1e-5, weight decay 1e-4, clip 1.0, validation every 25)",
          "hpcjobs/configs/dfs_step7.dfs6311_grid3_v7g1_size.yaml", 153, 183,
          ("n_steps: 200", "weight_decay: 0.0001", "validate_every: 25", "patience: 5", )),
    Entry(16, 8, "the validation slice",
          "xcquinox/alec/train.py", 745, 817,
          ("validation_molecules", "validation_reactions_path", )),
    Entry(16, 9, "the validation reaction-energy MAE",
          "xcquinox/alec/train.py", 820, 862,
          ("kcal", )),
    Entry(16, 10, "the validation-best tracker",
          "xcquinox/alec/train.py", 682, 742,
          ("min_delta", "patience", )),
    Entry(16, 11, "the validation-best checkpoint among the three saved",
          "xcquinox/alec/train.py", 1017, 1052,
          ("model_best.eqx", "model_val_best.eqx", )),
    Entry(19, 1, "the full pools (76 BH76, 140 W4-11) before the validation split",
          "xcquinox/alec/full_benchmark_pools.py", 516, 543,
          ("76 + 140 = 216", )),
    Entry(19, 2, "the validation slice, split by reaction identity",
          "xcquinox/alec/eval_holdout.py", 204, 245,
          ("reaction_identity_key", "hashlib", )),
    Entry(19, 3, "the strict held-out filter",
          "xcquinox/alec/eval_holdout.py", 173, 201,
          ("strict", )),
    Entry(19, 4, "the cell's trained reactions removed",
          "xcquinox/alec/eval_holdout.py", 283, 338,
          ("trained", )),
    Entry(19, 5, "the energy legs average one term per reaction identity",
          "xcquinox/alec/eval_holdout.py", 402, 442,
          ("61 rows over 54", )),
    Entry(19, 6, "eps_n per electron",
          "xcquinox/alec/evaluation.py", 196, 218,
          ("jnp.abs(rho - rho_ref)) / n_e", )),
    Entry(19, 7, "the grid RMSE",
          "xcquinox/alec/evaluation.py", 293, 320,
          ("jnp.sqrt(jnp.sum(w * diff ** 2) / jnp.sum(w))", )),
    Entry(19, 8, "WTMAD-2 with the GMTKN55 scale",
          "notebooks/analysis/make_ablation_arch_figure.py", 3824, 3857,
          ("56.84", )),
    Entry(19, 9, "the harmonic mean",
          "notebooks/analysis/make_ablation_arch_figure.py", 4835, 4841,
          ("2.0 / (1.0 / a + 1.0 / b)", )),
    Entry(19, 10, "the self-calibrated gamma = E_PBE / D_PBE",
          "notebooks/analysis/make_ablation_arch_figure.py", 4868, 4943,
          ("gamma = float(e_pbe) / float(d_pbe)", )),
    Entry(19, 11, "the DFS gamma 1084.87",
          "notebooks/analysis/make_ablation_arch_figure.py", 4952, 4975,
          ("_DFS_GAMMA_KCAL = 1084.87", )),
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
    end of the file, or when a token occurs on no line of the excerpt.
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


def codenote_lines(page: int, entries: Optional[Iterable[Entry]] = None,
                   outdir_name: str = _OUTDIR_NAME) -> List[str]:
    """The ``\\codenote`` lines of one page, in manifest order."""
    rows = sorted((e for e in _entries(entries) if e.page == page), key=lambda e: e.order)
    return [
        f"\\codenote{{{latex_escape(e.title)} [{latex_escape(e.path)}:{e.first}--{e.last}]}}"
        f"{{{outdir_name}/{slug(e)}.txt}}"
        for e in rows
    ]


def main(argv: Optional[Sequence[str]] = None) -> int:
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--write", action="store_true",
                    help="write the excerpt files (default: only print the codenote lines)")
    ap.add_argument("--root", type=Path, default=here.parents[1],
                    help="repository root the manifest paths are relative to")
    ap.add_argument("--outdir", type=Path, default=here / _OUTDIR_NAME,
                    help="directory of the excerpt files")
    args = ap.parse_args(argv)
    if args.write:
        written = write_excerpts(args.root, args.outdir)
        print(f"% {len(written)} excerpts written to {args.outdir}")
    for page in PAGES:
        print(f"% page {page}")
        for line in codenote_lines(page):
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
