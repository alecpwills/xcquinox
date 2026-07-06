---
title: "Figure glossary -- what every label on the 3×3 plots means"
author: "Alec Wills"
date: "2026-05-28"
geometry: margin=0.9in
fontsize: 11pt
mainfont: "DejaVu Serif"
sansfont: "DejaVu Sans"
monofont: "DejaVu Sans Mono"
header-includes:
  - \usepackage{booktabs}
  - \usepackage{longtable}
---

# Purpose

This note explains every label, code word, and visual element appearing on the
3×3 grids saved in `notebooks/analysis/`:

- `multimode_polarized_3x3.png`, `multimode_unpolarized_3x3.png`
  (and their `_convergence.png` companions)
- `constraint_pretrain_gmtkn55_demo_3x3.png` (the earlier
  `pretraining-recipe × metric` version)

It is intended as a stand-alone reference: open the figure on one screen and
this file on another.

---

# The big picture

We train a small **GGA-style exchange-correlation neural network** (call it the
*nn-functional*) to reproduce reference exchange and correlation enhancement
factors at every grid point of a precomputed PBE density. We then ask two
questions about that nn-functional:

1. **How well does it reproduce real chemistry references** (BH76 reaction
   energies and W4-11 atomization energies from GMTKN55)?
2. **How much do physical-constraint priors and the pretraining target choice
   matter** -- both for accuracy and for run-to-run reproducibility across random
   seeds?

The 3×3 plots answer those questions by sweeping *three independent axes*
(self-consistency depth × constraint stack × pretraining-loss weighting) on
*three independent metrics*. Each plot is one specific functional configuration
(polarized or unpolarized correlation baseline).

---

# Decoding the figure title

> **"Self-consistency ladder × constraints × pretraining vs GMTKN55 (polarized,
> 16 seeds)"**

| Phrase | Meaning |
|---|---|
| **Self-consistency ladder** | The *rows* (`fixed-ρ`, `one-shot`, `3-step SCF`) -- increasing the number of SCF cycles the nn-functional is run through before we read off the energy. |
| **× constraints** | The *x-axis* of each panel -- a progression of physical-constraint priors switched on cumulatively (`unconstrained → +LO → +LO+UEG → +LO+UEG+NNc`). |
| **× pretraining** | The *colored bars* inside each x-position -- random initialization versus two flavors of pretrained network. |
| **vs GMTKN55** | The reference values for the left and right columns come from the GMTKN55 benchmark database (the Grimme group's gold-standard collection of accurate post-HF energies). |
| **polarized / unpolarized** | Which version of the **PW92** correlation baseline the nn-functional is built on. *Polarized* uses the spin-resolved ζ-dependent baseline (correct for open-shell atoms); *unpolarized* uses the ζ = 0 total-density baseline. The two PNGs let us compare them directly. |
| **16 seeds** | The "random init" bars summarize 16 independent random network initializations (different seeds). The pretrained bars are a single pretrain seed (this is a demo, not a production benchmark). |

---

# Rows -- the **self-consistency ladder**

The same trained network is evaluated three ways, in increasing self-consistency.
ρ denotes the electron density; "ρ_PBE" is the precomputed PBE density.

| Row label on the y-axis | Code term | What is actually evaluated |
|---|---|---|
| **fixed-ρ** | `fixed_rho` | The nn-functional is **never solved self-consistently**. We just evaluate its XC energy on the *frozen* PBE density. Zero Roothaan steps. The fastest, cheapest test; this is what the original constraint demo used. |
| **one-shot (1 step)** | `one_shot` / `SolverMode.FIXED_J` with `max_cycles=1` and `FeaturePolicy.FROZEN` | Build the nn-functional's Fock matrix from the PBE density once, with the Coulomb (J) term **pinned to J[ρ_PBE]**, then diagonalize **once**. The orbitals get to relax, but ρ does not feed back into J or the descriptor features. |
| **3-step SCF** | `3step` / `SolverMode.FULL` with `max_cycles=3` and `FeaturePolicy.REASSEMBLE` | A genuine 3-cycle self-consistent SCF: at every cycle the Coulomb term J[ρ] *and* the nn-functional's descriptor features are rebuilt from the current density. This is "real" self-consistency, truncated to 3 cycles for cost. |

**Why a ladder.** Reading the figure top-to-bottom tells you whether the
conclusions you draw from the cheap fixed-ρ evaluation survive when you turn the
SCF machinery back on. Empirically (see the report) `3-step ≈ fixed-ρ` almost
exactly on this set, so the cheap evaluation is faithful.

### Equations -- what each mode literally computes

Equations below have been cross-checked against the actual implementation
(`xcquinox/alec/oneshot.py`, `xcquinox/alec/solver.py`,
`xcquinox/alec/solver_manual.py`) -- not just the docstring.

**Notation.** Let $\hat h_\text{core}$ be the one-electron core Hamiltonian
(kinetic + nuclear-attraction), $J[D]_{\mu\nu} = \sum_{\lambda\sigma}
(\mu\nu|\lambda\sigma)\,D_{\lambda\sigma}$ the Coulomb (Hartree) matrix built
from an AO density matrix $D$, $V_\text{xc}^\text{NN}[\rho]$ the
exchange-correlation potential matrix produced by the trained nn-functional, and
$E_\text{xc}^\text{NN}[\rho] = \int
\epsilon_\text{xc}^\text{NN}(\rho, |\nabla\rho|^2, \dots)\, d\mathbf r$ its
energy functional. $E_\text{nuc}$ is the nuclear repulsion. Subscripts
$\text{PBE}$ denote the precomputed PBE reference solution (frozen);
subscripts $n$ index Roothaan cycles. The linear-mixer parameter
$\alpha = \tfrac{1}{2}$ (the `SolverConfig` default) throughout.

**1. fixed-ρ** -- no SCF, energy-only evaluation on the frozen PBE density:

$$
E_\text{fixed-}\rho \;=\; E_\text{nuc} \;+\; \mathrm{Tr}\!\bigl[\hat h_\text{core}\,D_\text{PBE}\bigr]
\;+\; \tfrac{1}{2}\,\mathrm{Tr}\!\bigl[J[D_\text{PBE}]\,D_\text{PBE}\bigr]
\;+\; E_\text{xc}^\text{NN}\!\bigl[\rho_\text{PBE}\bigr].
$$

The nn-functional contributes only its *energy* -- no Fock matrix is built,
no diagonalization is run, the orbitals do not relax.

**2. one-shot** -- `SolverMode.FIXED_J` + `FeaturePolicy.FROZEN`,
exactly one Roothaan step. Build the Fock matrix once with $J$ pinned to
$J[D_\text{PBE}]$ and $V_\text{xc}^\text{NN}$ evaluated on the frozen
PBE-density grid inputs:

$$
F \;=\; \hat h_\text{core} \;+\; \underbrace{J[D_\text{PBE}]}_{\text{pinned}}
\;+\; \underbrace{V_\text{xc}^\text{NN}[\rho_\text{PBE}]}_{\text{frozen}\ \rho,\ \text{frozen descriptors}}
,\qquad F C = S C \varepsilon \;\Longrightarrow\; D_\text{new},
$$

then linear-mix once and read off the energy on the mixed density (with $J$
still pinned):

$$
D_\text{mix} \;=\; \tfrac{1}{2}\,D_\text{new} + \tfrac{1}{2}\,D_\text{PBE},
$$
$$
E_\text{one-shot} \;=\; E_\text{nuc}
+ \mathrm{Tr}\!\bigl[\hat h_\text{core}\,D_\text{mix}\bigr]
+ \tfrac{1}{2}\,\mathrm{Tr}\!\bigl[J[D_\text{PBE}]\,D_\text{mix}\bigr]
+ E_\text{xc}^\text{NN}\!\bigl[\rho(D_\text{mix})\bigr].
$$

The orbitals are allowed to relax exactly once, but neither $J$ nor the
nn-functional's potential is rebuilt.

**3. 3-step SCF** -- `SolverMode.FULL` + `FeaturePolicy.REASSEMBLE`,
at most 3 Roothaan steps. Starting from $D_0 = D_\text{PBE}$, for
$n = 0, 1, 2$:

$$
F_n \;=\; \hat h_\text{core} \;+\; J[D_n] \;+\; V_\text{xc}^\text{NN}[\rho_n]
,\qquad F_n C_{n+1} = S C_{n+1} \varepsilon_{n+1}
\;\Longrightarrow\; D_{n+1}^\text{new},
$$
$$
D_{n+1} \;=\; \tfrac{1}{2}\,D_{n+1}^\text{new} + \tfrac{1}{2}\,D_n
,\qquad \text{halt early if}\ \bigl|E_{n+1} - E_n\bigr| < 10^{-6}\,\text{Ha}.
$$

Each cycle recomputes the Coulomb matrix from the live density and reassembles
the nn-functional's descriptor features from the live density. The final
energy is the standard Kohn-Sham expression at the converged (or cycle-3)
density:

$$
E_\text{3-step} \;=\; E_\text{nuc}
+ \mathrm{Tr}\!\bigl[\hat h_\text{core}\,D_\text{final}\bigr]
+ \tfrac{1}{2}\,\mathrm{Tr}\!\bigl[J[D_\text{final}]\,D_\text{final}\bigr]
+ E_\text{xc}^\text{NN}\!\bigl[\rho_\text{final}\bigr].
$$

The 3-cycle cap is the unroll length passed to `jax.lax.scan` inside
`run_manual_scf`; early-converged states are simply held constant for the
remaining iterations.

**Footnote -- UKS and the descriptor list in this study.** For
spin-polarized (UKS) evaluation the Fock matrix is built per spin channel,
$F^\sigma = \hat h_\text{core} + J[D^\alpha + D^\beta]
+ V_\text{xc}^{\text{NN},\sigma}[\rho^\alpha, \rho^\beta]$, with exchange
spin-scaled per Oliver-Perdew and correlation taken from the spin-resolved
(or ζ = 0) baseline -- this is the `split_exc_energy_uks` branch whose gate is
exactly what the **polarized vs unpolarized** PNG pair toggles. For the
figures in this glossary the descriptor list is empty
(`precompute_fixed_density_data(spec, descriptors=(), ...)` in
`multimode_constraint_eval.py:379`), so the distinction between `FROZEN` and
`REASSEMBLE` in this run reduces to whether $\rho$ and $|\nabla\rho|^2$ on the
integration grid come from $D_\text{PBE}$ (frozen) or from the live $D_n$
(reassembled).

### Code-word translation

- **Roothaan step** = one diagonalization of the Fock matrix (one SCF cycle).
- **Fock matrix** = the one-electron effective Hamiltonian whose eigenvectors are
  the molecular orbitals; for KS-DFT it contains the XC potential V_xc.
- **J** = the classical Coulomb (Hartree) matrix. "**Fixed-J / FIXED\_J**" means
  J is computed once from ρ_PBE and never recomputed.
- **FROZEN / REASSEMBLE feature policy** = whether the nn-functional's
  descriptor inputs (ρ, |∇ρ|, ...) are recomputed each SCF cycle (`REASSEMBLE`)
  or held fixed at their PBE-density values (`FROZEN`). `FROZEN` must accompany
  `FIXED_J`; `REASSEMBLE` must accompany `FULL`.

---

# Columns -- the three metrics

Each column is a different MAE (mean absolute error), in **kcal/mol**, against a
different reference set. Lower bars = better.

## Left column: **BH76 reaction-energy MAE vs GMTKN55-BH76RC**

- **BH76** = a 76-reaction benchmark of thermochemical reaction barriers,
  curated by Goerigk & Grimme. **BH76RC** is its *reaction-energy channel* (ΔE
  for products − reactants), as opposed to BH76's standard *barrier heights*.
- We use **6 of the 76** BH76RC reactions here -- a deliberately held-out
  "probe-C" subset whose three companion reactions are used in training. The 6
  shown are therefore a **transfer test**, not a training metric.
- Reference values are the GMTKN55 W2-F12 / CCSD(T) reaction energies.
- **PBE = 8.1 kcal/mol** (dashed horizontal line); this is the standard PBE
  functional's error on these 6 reactions and is our "what a respectable
  GGA looks like" yardstick.

## Middle column: **per-species |E_nn − E_PBE| MAE**

- For every single species in the pool (29 atoms + small molecules), compute
  the absolute difference between the *nn-functional's* total energy and *PBE's*
  total energy, then mean over species.
- This is a **deviation-from-PBE diagnostic, not an accuracy benchmark.** No
  ground-truth reference goes into it. It answers: *how close to the
  reasonable-looking PBE total-energy surface does the nn-functional land?*
- Why it is informative: reaction-energy MAEs cancel any systematic per-species
  XC error (Σ coefficients = 0 for a balanced reaction). The per-species metric
  refuses that cancellation and exposes how *individually* good or bad each
  total energy is -- which is where constraints actually act.
- There is **no PBE baseline line** on this panel by construction (PBE deviates
  from itself by 0).

## Right column: **atomization-energy MAE vs GMTKN55 W4-11**

- **W4-11** = a 140-molecule subset of GMTKN55 with very high-accuracy
  CCSD(T)-derived atomization energies (the W4 protocol, ~sub-kcal/mol).
- We use a **10-molecule closed-shell subset** (H₂, H₂O, CH₄, NH₃, CO, N₂, CO₂,
  HF, C₂H₂, C₂H₄). Each atomization energy is computed as
  E(molecule) − Σ E(constituent atoms), and the MAE is taken against the W4-11
  reference value (kcal/mol).
- **PBE ≈ 10.45 kcal/mol** (dashed line) on this subset.
- This is the metric that most aggressively exposes **open-shell-atom physics**
  -- the atoms (H, C, N, O, F) enter with no cancelling partner on the other
  side, so any error in their spin-resolved correlation shows up at full
  weight. That is why polarized-vs-unpolarized matters so much here.

---

# x-axis (within each panel) -- the **constraint ladder**

These labels are the cumulative stack of *physical-constraint priors* baked
into the nn-functional's architecture. Each level adds one more constraint to
the previous; constraints are mathematical guarantees, not soft losses.

| x-axis label | Code term | Physical meaning |
|---|---|---|
| **unconstrained** | `lob_lim=None`, empty constraint list | No physics priors imposed: the network's outputs are free real numbers. (The built-in Lieb-Oxford "squash" is also disabled -- otherwise even "no constraints" would already be LO-bounded.) |
| **+LO** | `lieb_oxford` on the exchange net | Enforces the **Lieb-Oxford bound** -- a rigorous lower bound on the exchange-correlation energy (E_xc ≥ −C · ∫ρ^{4/3}). Implemented as a smooth output squash that prevents the network from emitting unphysically negative enhancement factors. |
| **+LO+UEG** | `lieb_oxford` + `ueg_limit` on the exchange net | Adds the **uniform-electron-gas limit**: in the slowly-varying (uniform) density limit, the exchange enhancement factor F_x → 1 (i.e. the functional reduces to LDA). This is a Levy-Perdew-Sahni-style consistency condition. |
| **+LO+UEG+NNc** | adds `non_negative_correlation` on the *correlation* net | **NNc = non-negative correlation enhancement** -- guarantees the correlation enhancement factor stays non-negative, ruling out unphysically positive correlation energies. ("NNc" reads as "Non-Negative correlation".) |

### Letter suffixes you may see in older labels

- The "(x)" or "(c)" suffix in some legends (`+LO(x)`, `+LO+UEG+NNc(c)`) marks
  which *network* the constraint acts on: **x = exchange net** (`xnet`),
  **c = correlation net** (`cnet`). They are the two halves of the nn-functional
  and are constrained separately.

### Quick reading

- **+LO** = "make exchange not blow up."
- **+LO+UEG** = "...and make it reduce to LDA in the uniform limit."
- **+LO+UEG+NNc** = "...and make correlation not be positive." This is the
  fully-constrained level.

---

# Bars and error indicators (the legend)

Inside each panel you see three colored bar series at every x-tick plus several
error indicators:

| Visual | Code label | What it represents |
|---|---|---|
| **Red bar** (dark red, leftmost) | `random init (mean)` | Arithmetic *mean* of the metric over **16 independently random-initialized networks** (same architecture/constraints, different seeds, no pretraining). Tells you what an untrained network of this constraint level looks like on average. |
| **Faint gray cap** at the top of the red bar | `worst of seeds` | The *worst* (largest, i.e. worst-MAE) value across the 16 seeds. Drawn as an upward-only error cap because we only care how bad the tail gets. Pulls back as constraints add. |
| **Bold dark-red whisker** centered on the red bar | `± std (seeds)` | One standard deviation across the 16 seeds. Shrinks as constraints add → constraints buy *robustness*, even when they don't move the mean. |
| **Blue bar** (slightly to the right) | `pretrained [unweighted]` | A *single*, pretrained network, where the pretraining loss is a **plain unweighted mean of squared residuals** between the network and the PBE/LDA target enhancement factors at every grid point. (One seed, hence no error bar.) |
| **Teal bar** (rightmost) | `pretrained [integration]` | Same pretrained network architecture, but the pretraining loss is **integration-weighted** -- each grid-point residual is multiplied by the DFT integration weight × density factors, so the loss approximates the *integrated* exchange/correlation-energy error, not a uniform point-by-point error. |
| **Black dashed line** | `PBE (8.1)` (or `PBE (10.45)`) | The PBE functional's MAE on this metric. Bars below the line beat PBE. The PBE line is omitted in the deviation-from-PBE column. |

### Loss-weighting code translation

- **`unweighted`** -- `L = (1/N) Σ_i (f_nn(x_i) − f_target(x_i))²` over grid
  points i. Treats every point equally.
- **`integration`** -- `L = Σ_i w_i · (f_nn(x_i) − f_target(x_i))²`, with
  `w_i` ∝ DFT integration weight × density-weighting that converts the
  per-point residual into an approximate energy-integral residual. Tilts the
  optimizer toward getting *energy-relevant* regions right rather than tail
  regions where ρ is tiny.

---

# The convergence figure (`*_convergence.png`)

These are the two-panel companion figures `multimode_polarized_convergence.png`
and `multimode_unpolarized_convergence.png`. They diagnose **pretraining
optimization difficulty**, separately for exchange and correlation.

| Element | Meaning |
|---|---|
| **Left panel title:** `pretraining steps-to-converge [unweighted]` | Same metric as below, but with the *unweighted* pretraining loss. |
| **Right panel title:** `pretraining steps-to-converge [integration]` | ...with the *integration-weighted* loss. |
| **Red bar (xnet)** | Number of gradient steps the **exchange network** took to first reach 1.05× its eventual minimum loss (i.e. "within 5% of the best loss it ever achieves"). Lower = pretraining converged faster. |
| **Blue bar (cnet)** | Same, for the **correlation network**. |
| **y-axis:** `steps to reach 1.05× min loss` | Out of 1000 total pretraining steps. A bar at ≈1000 means the network never really converged within budget. |
| **x-axis** | The same `unconstrained → +LO+UEG+NNc` constraint ladder. |

This figure answers: *do constraints help or hurt the pretraining optimizer?*
For the polarized config the picture is non-monotonic and interesting (LO
accelerates exchange dramatically; NNc accelerates correlation dramatically).
For the unpolarized config most levels hit the budget -- a landscape
difference between the two functional families.

---

# The other 3×3 -- `constraint_pretrain_gmtkn55_demo_3x3.png`

This older figure is **structured differently**. It is the
*fixed-ρ-only* version, but with the rows reinterpreted as **pretraining
recipe**, not self-consistency depth:

| Row | Meaning |
|---|---|
| **150-step pretrain (unpolarized)** | Short pretraining run, unpolarized correlation baseline. |
| **1000-step pretrain (unpolarized)** | Longer pretraining, same unpolarized baseline. |
| **1000-step pretrain + spin-polarized PW92c** | Longer pretraining, **switched to the polarized correlation baseline**. This is the row that drops atomization MAE from ~27 → ~10 kcal/mol -- the "polarized baseline is what fixes atomization" punchline. |

The **columns**, **x-axis**, **bar/error semantics**, and **PBE dashed lines**
are identical to the multimode 3×3. Only the row meaning changes.

---

# Software / data anchors (for cross-reference)

| Term you may see | Where it lives |
|---|---|
| **GMTKN55-BH76RC** | Benchmark file `bh76rc.json`, GMTKN55 distribution. |
| **GMTKN55 W4-11** | Benchmark file `w411.json` (atomization-energy subset). |
| **PBE** | Perdew-Burke-Ernzerhof GGA, the reference DFT functional throughout. |
| **PW92** / **PW92c** | Perdew-Wang 1992 LDA correlation. "PW92c" emphasizes the *correlation* part; the spin-polarized form is the ζ-dependent VWN/PW92 expression. |
| **GGA** | Generalized-gradient-approximation rung of Jacob's ladder -- the functional class our nn lives in (uses ρ and ∇ρ but no kinetic-energy density). |
| **`xcquinox.alec.solver`** | Where `SolverConfig`, `SolverMode`, `FeaturePolicy`, and `SolverBackend` are defined. |
| **`xcquinox.alec.pretrain`** | Defines `PretrainSpec`, `loss_weighting`, integration weights. |
| **`make_constraint_levels()`** | In `constraint_pretrain_gmtkn55_demo.py` -- produces the four constraint-ladder labels. |
| **`make_multimode_figure.py`** | The plotting script that *literally produces* the multimode 3×3 and convergence PNGs from the saved JSON; this glossary is faithful to its label strings. |

---

# Advisor summary -- what to take away from the multimode 3×3 figures

> Across all three self-consistency rungs (`fixed-ρ → one-shot → 3-step SCF`)
> the picture is essentially identical -- running the functional
> self-consistently barely moves any bar, so the cheap fixed-ρ conclusions
> hold. Within each rung, layering on physical constraints does not change
> the random-init *mean* accuracy but cuts seed-to-seed standard deviation by
> roughly 4× and the worst-of-16-seeds tail by roughly 2×, while pretraining
> (especially the integration-weighted variant) is what actually closes the
> absolute gap to PBE on BH76 and W4-11. Comparing the two PNGs, the
> spin-polarized correlation baseline is the binding factor for atomization
> energies: the polarized functional lands on PBE (≈10 kcal/mol) at the
> fully-constrained pretrained level, while the unpolarized one is stuck at
> ~27 kcal/mol and actually *worsens* as constraints are added -- because
> open-shell atoms need ζ-resolved correlation the unpolarized functional
> cannot represent.
