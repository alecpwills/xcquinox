# Physical constraints + pretraining vs GMTKN55 — demo analysis

**Scope.** A small, self-contained demonstration (not a production benchmark) probing how three
ingredients pull a randomly-initialized `xcquinox.alec` GGA exchange–correlation network toward
correct physics: (i) **physical constraints**, (ii) **pretraining** to PBE/LDA enhancement factors,
and (iii) a **spin-polarized correlation baseline**. Settings: `def2-svp`, DFT grid level 1, a
single pretrain seed, 16 random-init seeds for the random bars. Evaluation sets: 6 GMTKN55-BH76RC
reactions, a per-species PBE-deviation diagnostic, and a small closed-shell GMTKN55 **W4-11**
atomization subset. Directional evidence about *mechanism*, not absolute accuracy.

**Figure.** Rows = pretraining scenario (150-step → 1000-step → 1000-step + spin-polarized PW92c
baseline); columns = metric.

![3x3 comparison](constraint_pretrain_gmtkn55_demo_3x3.png)

PBE baselines: **BH76 = 8.08**, **W4-11 = 10.45** kcal/mol. The middle column is a *deviation from
PBE* diagnostic (no GMTKN55 absolute-energy reference exists for it) — not an absolute-accuracy
benchmark. The left and right columns are scored against real GMTKN55 reference values.

---

## Datasets and coverage

These are **curated subsets**, not the full benchmarks: **6 of 76 BH76** reactions and **10 of 140
W4-11** atomizations (29 species computed). The subsets are chosen to keep the demo fast and
clean (closed-shell W4-11 molecules avoid molecular open-shell bookkeeping; the BH76 reactions are a
held-out transfer set). Conclusions are about *mechanism*, not full-benchmark accuracy.

**BH76 — 6 reaction energies** (the BH76**RC** reaction-energy channel ΔE, *not* barrier heights),
from the out-of-training Probe-C set — it deliberately **excludes the 3 BH76 reactions used in
training**, so this is a transfer test. References: GMTKN55-BH76RC (W2-F12), kcal/mol.

| reaction | ΔE_ref (kcal/mol) |
|---|---|
| OH + H₂ → H₂O + H    | −16.39 |
| H + HCl → H₂ + Cl    | −1.90 |
| CH₃ + H₂ → CH₄ + H   | −3.11 |
| OH + NH₃ → H₂O + NH₂ | −10.32 |
| H + N₂O → OH + N₂    | −64.91 |
| H + H₂S → H₂ + HS    | −13.26 |

**W4-11 — 10 closed-shell molecules**, each scored as molecule → constituent atoms against the
GMTKN55 W4-11 reference atomization energy (kcal/mol):

| molecule | atomization ref | molecule | atomization ref |
|---|---|---|---|
| H₂  | 109.493 | N₂   | 228.485 |
| H₂O | 232.974 | CO₂  | 390.141 |
| CH₄ | 420.420 | HF   | 141.640 |
| NH₃ | 298.018 | C₂H₂ | 405.525 |
| CO  | 259.727 | C₂H₄ | 564.095 |

The **29 species** are everything precomputed (one PBE SCF + grid each): the BH76 molecules/radicals
above, the 10 W4-11 molecules, the W4-11 constituent atoms (C, H, N, O, F), and atomic Cl (from
H + HCl → H₂ + Cl). The BH76 and W4-11 pools key species independently (Hill formula vs the W4-11
`.res` tokens), so six small species common to both — H₂, H₂O, CH₄, NH₃, N₂, and atomic H — are
computed in *both* pools; the 29 entries are therefore ~23 chemically distinct (a harmless
redundancy: the duplicates get near-identical PBE energies). Note there is no atomic sulfur — sulfur
appears only inside H₂S/HS.

---

## Takeaways

**1. The metric can hide the entire constraint benefit.** On BH76 *reaction energies*, constraints
look useless — random-init mean barely moves (unconstrained 21.4 → fully-constrained 20.9 kcal/mol).
A balanced reaction (Σ coeff = 0) **cancels the systematic per-species XC error** that constraints
act on. Reaction-energy benchmarks alone systematically under-report the value of constraints.

**2. Constraints buy robustness, not mean accuracy.** On the non-cancelling per-species metric the
random-init *mean* improves only modestly (435 → 380), but the tails collapse:
- **worst-of-16-seeds: 910 → 490 kcal/mol (~1.9×)**
- **seed-to-seed std: 288 → 70 kcal/mol (~4.1×)**

Constraints make a randomly-initialized network **reliably reasonable instead of occasionally
catastrophic** — the real payoff is training stability and reproducibility.

**3. Lieb–Oxford + UEG are the variance-killers; non-negativity is a fine-tuner.** The std collapse
is staircase-like: +LO 288 → 144, +UEG 144 → 72 (each ~2×), +NNc essentially flat. The *exchange*
constraints do the heavy lifting on robustness.

**4. Headline physics result — the spin-polarized correlation baseline fixes atomization energies.**
Atomization energies are dominated by open-shell **atoms** (H, C, N, O, F), where spin polarization
ζ matters and does not cancel. With the unpolarized (ζ = 0) PW92c baseline the pretrained W4-11
error is stuck at **~27 kcal/mol — 2.6× worse than PBE — and more training does not fix it**
(150-step 30.1 → 1000-step 27.3). Switching to the ζ-resolved PW92c baseline drops the pretrained,
fully-constrained W4-11 error to **10.2 kcal/mol ≈ PBE (10.45)** — a 2.7× improvement that is a
*physical-correctness* fix, not a capacity or training-length fix. It also helps BH76 (random-init
mean 21.4 → 14.7).

**5. A correct baseline restores "more constraints → better" monotonicity.** With the wrong
(unpolarized) baseline, adding constraints made pretrained W4-11 *worse* and non-monotonic
(19.6 → 21.1 → 27.3). With the spin-polarized baseline it is cleanly monotonic and improving
(12.9 → 12.3 → **10.2**), and the fully-constrained pretrained model **beats PBE on BH76** (7.19 vs
8.08).

**6. A capacity-vs-grounding nuance.** The spin-polarized model is *worse at random init* on W4-11
(unconstrained 42.7 vs 20.7) — the extra ζ input is unconstrained noise before training — but far
better *after* pretraining (10.2 vs 27.3). Added physical expressiveness costs at initialization and
pays off only once trained.

### Mechanism — why atomization improves *only* with the polarized baseline

A natural question: in the unpolarized rows, pretraining + constraints make W4-11 *worse*
(19.6 → 21.1 → 27.3), yet with the spin-polarized baseline the same machinery improves it
monotonically to PBE (12.9 → 12.3 → 10.2). The reason is **model representability**, not training:

- **Atomization isolates open-shell atoms.** AE = E(molecule, closed-shell) − Σ E(atoms,
  open-shell). The molecules are closed-shell (ζ = 0 is exact for them); the atoms (H, C, N, O, F)
  are open-shell (ζ ≠ 0) and are subtracted with full weight and **no cancelling partner** — unlike
  BH76, where Σ coeff = 0 and radicals appear on both sides so ζ errors largely cancel. AE is
  therefore the metric that maximally exposes any open-shell-atom error.
- **The unpolarized model cannot represent a spin-polarized atom.** At evaluation
  (`xcquinox/alec/oneshot.py`, `split_exc_energy_uks`) correlation uses the ζ-dependent PW92 baseline
  **only if `cnet.use_spin_polarization` is True**; otherwise it is the ζ = 0 total-density
  (PW92/von Barth–Hedin) baseline, and the unpolarized cnet has **no ζ input feature**. So for an
  open-shell atom the correct, spin-reduced correlation is **outside the model class** — the atom
  error is *irreducible* within the unpolarized functional.
- **Hence optimizing harder hurts.** Pretraining fits the network faithfully to its target and the
  constraints (UEG limit, non-negative correlation) pull it onto a tighter manifold — but for the
  atoms that manifold is the wrong (ζ = 0) one. "Fitting better" = committing harder to an answer
  the model cannot make correct for atoms, so AE is flat-to-worse and **degrades monotonically with
  more constraints**.
- **The polarized model adds the missing degree of freedom.** `use_spin_polarization=True` switches
  the baseline to the real per-point ζ = (ρ↑−ρ↓)/ρ *and* gives the cnet a ζ input. The correct
  open-shell-atom physics is now representable, so the same pretraining + constraints converge toward
  the right atom energies → AE improves monotonically to PBE-level (10.2 ≈ 10.45).

**Isolation corollary.** The pretrain atom set is identical across all rows (H, He, N, O — no C or
F), yet the polarized row reaches PBE-level AE anyway. This rules out "needs more training" or "needs
broader pretrain coverage" and points squarely at **representability of spin polarization**.

**Honest caveat.** The polarized row changes three things together (ζ-dependent eval baseline, ζ
network input, spin-resolved pretrain targets). The argument above says the eval-time ζ-baseline +
ζ-input is the binding cause (consistent with the `split_exc_energy_uks` gate); a fully clean proof
would ablate the three factors independently.

**One-line summary.** *Constraints don't lower mean reaction-energy error (it cancels) — they cut
worst-case per-species error ~2× and seed variance ~4×; and getting the spin-polarized correlation
baseline right is what pulls pretrained atomization energies from 2.6× worse than PBE down onto PBE.*

---

## Data

### Metric 1 — BH76 reaction-energy MAE (kcal/mol, vs GMTKN55-BH76RC; PBE = 8.08)

*Unpolarized baseline (rows 1 & 2 — random init is identical; pretraining differs):*

| constraint level | random mean | worst-of-16 | std | pretrained (150) | pretrained (1000) |
|---|---|---|---|---|---|
| unconstrained   | 21.37 | 30.79 | 5.84 | — | — |
| +LO             | 20.94 | 25.18 | 2.63 | 11.09 | 10.55 |
| +LO+UEG         | 20.93 | 24.95 | 2.50 | 10.21 | 9.89 |
| +LO+UEG+NNc     | 20.90 | 24.91 | 2.47 | **8.21** | **8.61** |

*Spin-polarized baseline (row 3):*

| constraint level | random mean | worst-of-16 | std | pretrained (1000) |
|---|---|---|---|---|
| unconstrained   | 14.72 | 23.79 | 5.78 | — |
| +LO             | 14.17 | 18.36 | 2.68 | 9.46 |
| +LO+UEG         | 14.15 | 18.20 | 2.55 | 8.94 |
| +LO+UEG+NNc     | 14.12 | 17.96 | 2.48 | **7.19** |

### Metric 2 — per-species |E_nn − E_pbe| MAE (kcal/mol; deviation from PBE; lower = closer to PBE)

*Unpolarized baseline (rows 1 & 2):*

| constraint level | random mean | worst-of-16 | std | pretrained (150) | pretrained (1000) |
|---|---|---|---|---|---|
| unconstrained   | 434.85 | 910.10 | 287.83 | — | — |
| +LO             | 389.11 | 612.92 | 143.91 | 13.31 | 2.55 |
| +LO+UEG         | 381.07 | 495.26 | 71.73 | 142.24 | 7.20 |
| +LO+UEG+NNc     | 380.28 | 490.21 | 70.46 | 104.22 | 42.23 |

*Spin-polarized baseline (row 3):*

| constraint level | random mean | worst-of-16 | std | pretrained (1000) |
|---|---|---|---|---|
| unconstrained   | 436.44 | 905.03 | 286.83 | — |
| +LO             | 391.48 | 611.95 | 143.56 | 55.98 |
| +LO+UEG         | 383.52 | 494.73 | 71.24 | 64.72 |
| +LO+UEG+NNc     | 382.96 | 490.74 | 70.20 | **20.49** |

### Metric 3 — atomization-energy MAE (kcal/mol, vs GMTKN55 W4-11; PBE = 10.45)

*Unpolarized baseline (rows 1 & 2):*

| constraint level | random mean | worst-of-16 | std | pretrained (150) | pretrained (1000) |
|---|---|---|---|---|---|
| unconstrained   | 20.71 | 29.98 | 4.54 | — | — |
| +LO             | 17.83 | 20.56 | 1.35 | 19.44 | 19.59 |
| +LO+UEG         | 17.87 | 20.73 | 1.41 | 20.23 | 21.10 |
| +LO+UEG+NNc     | 17.85 | 20.66 | 1.29 | 30.12 | 27.27 |

*Spin-polarized baseline (row 3):*

| constraint level | random mean | worst-of-16 | std | pretrained (1000) |
|---|---|---|---|---|
| unconstrained   | 42.70 | 62.38 | 12.31 | — |
| +LO             | 41.89 | 50.69 | 5.48 | 12.94 |
| +LO+UEG         | 41.89 | 50.80 | 5.55 | 12.32 |
| +LO+UEG+NNc     | 41.87 | 50.89 | 5.56 | **10.20** |

---

## Reproduce

The figure is rebuilt from three saved run-logs with **no recomputation**:

```bash
python notebooks/analysis/make_constraint_3x3.py \
  notebooks/analysis/demo_logs/constraint_demo_pretrain150step.log \
  notebooks/analysis/demo_logs/constraint_demo_pretrain1000step.log \
  notebooks/analysis/demo_logs/constraint_demo_pretrain1000step_polc.log \
  notebooks/analysis/constraint_pretrain_gmtkn55_demo_3x3.png
```

To regenerate a run-log from scratch (~12 min for 150 steps, ~60 min for the 1000-step rows):
`python notebooks/analysis/constraint_pretrain_gmtkn55_demo.py`. The demo source defines the
constraint levels, metrics, and the spin-polarized baseline toggle.

**Caveat for presentation.** This is the demo configuration (small basis, coarse grid, single
pretrain seed, small reaction/atomization sets). It is directional evidence about the *mechanism*;
the cluster runs are the production test.
