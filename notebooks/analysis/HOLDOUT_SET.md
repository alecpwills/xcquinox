# Held-out evaluation set -- exact constituents (dfs_step7)

This note enumerates precisely what the held-out ("hold-out") evaluation contains: which
reactions the figures' energy metrics run over, which reactions are withheld as the
validation slice, and which species carry the density comparison. Extracted 2026-07-29
from the canonical pool builder (`xcquinox.alec.full_benchmark_pools.load_full_held_out_pools`)
and the pulled run `dfs6311_grid3_v3/run_20260723T161502Z` (`validation/val_reactions.json` + the
`eval_holdout_val_best/` outputs). The same split applies to every dfs_step7 sweep -- the
figures' dataset footer line carries the live counts per run, and this file is the
name-by-name expansion of that line.

## 1. Composition at a glance

| Quantity | Count | Notes |
|---|---|---|
| Canonical pool, reaction entries | 216 (76 BH76 + 140 W4-11) | the full benchmark pool the harness builds |
| Canonical pool, unique reaction names | 212 | 4 BH76 entries share a name with another entry (Sec. 2) |
| TEST slice (what `per_reaction.json` carries) | 165 names (52 BH76 + 113 W4-11) | the reported metrics/figures run on this slice minus the four validation twins (identity exclusion) and minus each cell's VERBATIM supervised reactions (Sec. 2; about 162 surviving names at ss1, 147 at ss26). NN metrics reduce the SCORED subset of a cell's slice (reactions with finite NN energies; incomplete cells are starred on the figures and named in the note band); every PBE/SCAN comparator, pooled and per-cell, reduces the full slice regardless of NN convergence |
| Validation slice (early-stop / val-best selection) | 47 names (20 BH76 + 27 W4-11; 49 entries: 22+27) | withheld from every reported TEST metric |
| Test/validation overlap | 0 by name; 4 by physical identity | four BH76 barriers appear twice in the pool under permuted-reactant names, one copy per slice (`bh76_h_hf_to_hfhts`/`bh76_hf_h_to_hfhts` and the three analogous pairs); the figure layer drops the four test-side twins on read, since validation-best selection saw those barriers |
| Pool species (molecules + atoms) | 214 | reactants/products of the 216 entries |
| Species evaluated per spec (this run) | 213 | pool minus `c2` (Sec. 4) |
| Density species (finite NN + PBE channels) | 198 | atoms are skipped by design (Sec. 4) |
| Atomic species skipped for density | 15 | `O`, `al`, `b`, `be`, `c`, `cl`, `cl-`, `f`, `f-`, `h`, `n`, `o`, `p`, `s`, `si` |

## 2. How the split works

Hold-out is VERBATIM: a spec's test slice excludes exactly its supervised reactions --
the reaction-form training points recorded in its `train_metadata.json`
(`loss_kwargs["bh76_reactions"]`: the AE-as-reactions, whose `w411_*_atomization` pool
twins leave under cross-vocabulary identity, and the trained barrier reactions) -- plus
the recorded validation slice (identity-level, so permuted-name twins leave with it). A
reaction merely CONTAINING a trained molecule stays: it is a generalization target, not
a training target. The figure layer reconstructs each spec's full slice from its
per-species energies over the canonical pool (the training vocabulary is ASE Hill
formulas -- `CHN`, `H3N`, `HO` -- while the pool uses GMTKN55-style names -- `hcn`,
`nh3`, `oh`; identity matching is by element composition + charge + spin with geometric
isomer classes, `xcquinox.alec.species_matching`). Per-cell reaction counts on the
figures reflect exactly these exclusions, so they are near-uniform across subset sizes
(about 162 at ss1, 147 at ss26).

A slice row needs a finite COMPARATOR (PBE) leg only: reactions whose NN energy is NaN
(the model's own SCF failures) stay in the slice with NaN NN columns, so reference
reductions never follow a single arch's NN-scored subset -- at a fixed subset size every
arch shares one slice and hence one comparator anchor. NN metrics reduce the scored
subset; the shortfall is starred on the figures, named with scored/slice counts in the
note band, and recorded as `n_reactions` vs `n_reactions_slice` in the ED CSVs. On the
2026-08-18 v4gga pull three cells are affected, pending re-evaluation:
`deep_rung35_3x16`/ss18 (final checkpoint only) and `deep_rung35_attn_3x16`/ss1 and ss2
(both channels).

The canonical pool is the union of two GMTKN55-style subsets (sources in Sec. 5): the BH76
barrier heights and the W4-11 atomization energies, 216 reaction entries over
214 species. A fixed validation slice of 47 reaction names is withheld for
early stopping and validation-best checkpoint selection (`validation/val_reactions.json`,
staged per run); the remaining 165 names form the TEST slice that
`eval_holdout*/per_reaction.json` records and every energy figure/CSV consumes. The two
slices are disjoint and together cover the pool exactly. Counting is by NAME everywhere
(the figures' dataset line says "name-dedup"): four BH76 entries share a name with a second
entry -- forward/reverse barriers of the same transition state:

- `bh76_C5H8_to_RKT22` (2 entries, validation slice)
- `bh76_clch3clcomp_to_clch3clts` (2 entries, test slice)
- `bh76_fch3fcomp_to_fch3fts` (2 entries, test slice)
- `bh76_h_H2_to_RKT06` (2 entries, validation slice)

so the 212 names expand to 216 entries: the test slice's 165 names cover
167 entries and the validation slice's 47 names cover 49 entries.

## 3. Reaction tables

### 3.1a TEST slice -- BH76 barrier heights (52 names)

| # | Reaction name | Reaction | Ref (kcal/mol) |
|---|---|---|---|
| 1 | `bh76_C2H5_2_NH3_to_RKT20` | C2H5_2 + NH3 -> RKT20 | 17.80 |
| 2 | `bh76_H2O_C2H5_2_to_RKT09` | H2O + C2H5_2 -> RKT09 | 20.40 |
| 3 | `bh76_H2O_NH2_to_RKT07` | H2O + NH2 -> RKT07 | 13.70 |
| 4 | `bh76_H2O_ch3_to_RKT04` | H2O + ch3 -> RKT04 | 19.50 |
| 5 | `bh76_H2_HS_to_RKT16` | H2 + HS -> RKT16 | 17.20 |
| 6 | `bh76_H2_O_to_RKT14` | H2 + O -> RKT14 | 13.20 |
| 7 | `bh76_H2_PH2_to_RKT12` | H2 + PH2 -> RKT12 | 24.70 |
| 8 | `bh76_NH2_C2H5_2_to_RKT19` | NH2 + C2H5_2 -> RKT19 | 9.80 |
| 9 | `bh76_NH2_CH4_to_RKT21` | NH2 + CH4 -> RKT21 | 13.90 |
| 10 | `bh76_NH2_ch3_to_RKT18` | NH2 + ch3 -> RKT18 | 8.90 |
| 11 | `bh76_NH3_ch3_to_RKT21` | NH3 + ch3 -> RKT21 | 16.90 |
| 12 | `bh76_NH_C2H6_to_RKT19` | NH + C2H6 -> RKT19 | 19.40 |
| 13 | `bh76_NH_CH4_to_RKT18` | NH + CH4 -> RKT18 | 22.00 |
| 14 | `bh76_O_CH4_to_RKT11` | O + CH4 -> RKT11 | 14.40 |
| 15 | `bh76_O_hcl_to_RKT17` | O + hcl -> RKT17 | 10.40 |
| 16 | `bh76_c2h5_1_to_c2h5ts` | c2h5_1 -> c2h5ts | 42.00 |
| 17 | `bh76_c3h7_to_c3h7ts` | c3h7 -> c3h7ts | 33.00 |
| 18 | `bh76_ch3_H2_to_RKT03` | ch3 + H2 -> RKT03 | 11.90 |
| 19 | `bh76_ch3_c2h4_to_c3h7ts` | ch3 + c2h4 -> c3h7ts | 6.40 |
| 20 | `bh76_ch3_clf_to_ch3fclts` | ch3 + clf -> ch3fclts | 7.10 |
| 21 | `bh76_ch3cl_cl-_to_clch3clts` | ch3cl + cl- -> clch3clts | 2.50 |
| 22 | `bh76_ch3f_cl_to_ch3fclts` | ch3f + cl -> ch3fclts | 59.80 |
| 23 | `bh76_ch3f_f-_to_fch3fts` | ch3f + f- -> fch3fts | -0.60 |
| 24 | `bh76_ch3oh_f-_to_hoch3fts` | ch3oh + f- -> hoch3fts | 17.60 |
| 25 | `bh76_cl_CH4_to_RKT08` | cl + CH4 -> RKT08 | 6.80 |
| 26 | `bh76_clch3clcomp_to_clch3clts` (x2) | clch3clcomp -> clch3clts | 13.50 |
| 27 | `bh76_f-_ch3cl_to_fch3clts` | f- + ch3cl -> fch3clts | -12.30 |
| 28 | `bh76_f_H2_to_RKT10` | f + H2 -> RKT10 | 1.60 |
| 29 | `bh76_fch3clcomp1_to_fch3clts` | fch3clcomp1 -> fch3clts | 3.50 |
| 30 | `bh76_fch3clcomp2_to_fch3clts` | fch3clcomp2 -> fch3clts | 29.60 |
| 31 | `bh76_fch3fcomp_to_fch3fts` (x2) | fch3fcomp -> fch3fts | 13.40 |
| 32 | `bh76_h_H2S_to_RKT16` | h + H2S -> RKT16 | 3.90 |
| 33 | `bh76_h_PH3_to_RKT12` | h + PH3 -> RKT12 | 2.90 |
| 34 | `bh76_h_co_to_hcots` | h + co -> hcots | 3.20 |
| 35 | `bh76_h_f2_to_hf2ts` | h + f2 -> hf2ts | 1.50 |
| 36 | `bh76_h_hcl_to_RKT01` | h + hcl -> RKT01 | 6.10 |
| 37 | `bh76_h_hcl_to_hclhts` | h + hcl -> hclhts | 17.80 |
| 38 | `bh76_h_hf_to_hfhts` | h + hf -> hfhts | 42.10 |
| 39 | `bh76_h_oh_to_RKT14` | h + oh -> RKT14 | 10.90 |
| 40 | `bh76_hcn_to_hcnts` | hcn -> hcnts | 48.10 |
| 41 | `bh76_hco_to_hcots` | hco -> hcots | 22.80 |
| 42 | `bh76_hf_ch3_to_hfch3ts` | hf + ch3 -> hfch3ts | 56.90 |
| 43 | `bh76_hf_f_to_hf2ts` | hf + f -> hf2ts | 104.80 |
| 44 | `bh76_hn2_to_hn2ts` | hn2 -> hn2ts | 10.90 |
| 45 | `bh76_hnc_to_hcnts` | hnc -> hcnts | 33.00 |
| 46 | `bh76_hoch3fcomp1_to_hoch3fts` | hoch3fcomp1 -> hoch3fts | 47.70 |
| 47 | `bh76_oh-_ch3f_to_hoch3fts` | oh- + ch3f -> hoch3fts | -2.70 |
| 48 | `bh76_oh_CH4_to_RKT04` | oh + CH4 -> RKT04 | 6.30 |
| 49 | `bh76_oh_H2_to_RKT02` | oh + H2 -> RKT02 | 5.20 |
| 50 | `bh76_oh_NH3_to_RKT07` | oh + NH3 -> RKT07 | 3.40 |
| 51 | `bh76_oh_ch3_to_RKT11` | oh + ch3 -> RKT11 | 8.90 |
| 52 | `bh76_oh_n2_to_n2ohts` | oh + n2 -> n2ohts | 82.60 |

### 3.1b TEST slice -- W4-11 atomization energies (113 names)

| # | Reaction name | Reaction | Ref (kcal/mol) |
|---|---|---|---|
| 1 | `w411_acetaldehyde_atomization` | acetaldehyde -> 2 c + o + 4 h | 677.86 |
| 2 | `w411_acetic_atomization` | acetic -> 2 c + 2 o + 4 h | 804.02 |
| 3 | `w411_alcl3_atomization` | alcl3 -> al + 3 cl | 312.65 |
| 4 | `w411_alcl_atomization` | alcl -> al + cl | 122.62 |
| 5 | `w411_alf3_atomization` | alf3 -> al + 3 f | 430.97 |
| 6 | `w411_alf_atomization` | alf -> al + f | 163.78 |
| 7 | `w411_alh3_atomization` | alh3 -> al + 3 h | 213.17 |
| 8 | `w411_alh_atomization` | alh -> al + h | 73.57 |
| 9 | `w411_b2_atomization` | b2 -> 2 b | 67.46 |
| 10 | `w411_b2h6_atomization` | b2h6 -> 2 b + 6 h | 607.02 |
| 11 | `w411_be2_atomization` | be2 -> 2 be | 2.67 |
| 12 | `w411_becl2_atomization` | becl2 -> be + 2 cl | 225.27 |
| 13 | `w411_bef2_atomization` | bef2 -> be + 2 f | 309.10 |
| 14 | `w411_bf3_atomization` | bf3 -> b + 3 f | 470.97 |
| 15 | `w411_bf_atomization` | bf -> b + f | 182.52 |
| 16 | `w411_bhf2_atomization` | bhf2 -> b + h + 2 f | 410.97 |
| 17 | `w411_bn3pi_atomization` | bn3pi -> b + n | 105.81 |
| 18 | `w411_bn_atomization` | bn -> b + n | 105.24 |
| 19 | `w411_c-hcoh_atomization` | c-hcoh -> c + o + 2 h | 317.65 |
| 20 | `w411_c-hono_atomization` | c-hono -> h + n + 2 o | 312.22 |
| 21 | `w411_c-hooo_atomization` | c-hooo -> h + 3 o | 233.09 |
| 22 | `w411_c-n2h2_atomization` | c-n2h2 -> 2 h + 2 n | 291.13 |
| 23 | `w411_c2h2_atomization` | c2h2 -> 2 c + 2 h | 405.52 |
| 24 | `w411_c2h3f_atomization` | c2h3f -> 2 c + f + 3 h | 573.89 |
| 25 | `w411_c2h5f_atomization` | c2h5f -> 2 c + 5 h + f | 721.50 |
| 26 | `w411_c2h6_atomization` | c2h6 -> 2 c + 6 h | 713.08 |
| 27 | `w411_cch_atomization` | cch -> 2 c + h | 266.16 |
| 28 | `w411_cf2_atomization` | cf2 -> c + 2 f | 258.78 |
| 29 | `w411_cf4_atomization` | cf4 -> c + 4 f | 478.76 |
| 30 | `w411_cf_atomization` | cf -> c + f | 132.72 |
| 31 | `w411_ch2-sing_atomization` | ch2-sing -> c + 2 h | 181.46 |
| 32 | `w411_ch2-trip_atomization` | ch2-trip -> c + 2 h | 190.75 |
| 33 | `w411_ch2c_atomization` | ch2c -> 2 c + 2 h | 359.93 |
| 34 | `w411_ch2nh2_atomization` | ch2nh2 -> c + n + 4 h | 482.28 |
| 35 | `w411_ch2nh_atomization` | ch2nh -> c + n + 3 h | 439.44 |
| 36 | `w411_ch3_atomization` | ch3 -> c + 3 h | 307.87 |
| 37 | `w411_ch3f_atomization` | ch3f -> c + f + 3 h | 422.96 |
| 38 | `w411_ch3nh2_atomization` | ch3nh2 -> c + n + 5 h | 582.30 |
| 39 | `w411_ch3nh_atomization` | ch3nh -> c + n + 4 h | 474.63 |
| 40 | `w411_ch4_atomization` | ch4 -> c + 4 h | 420.42 |
| 41 | `w411_ch_atomization` | ch -> c + h | 84.22 |
| 42 | `w411_cl2_atomization` | cl2 -> 2 cl | 59.75 |
| 43 | `w411_cl2o_atomization` | cl2o -> 2 cl + o | 101.46 |
| 44 | `w411_clcn_atomization` | clcn -> cl + c + n | 285.45 |
| 45 | `w411_clf_atomization` | clf -> cl + f | 62.80 |
| 46 | `w411_clo_atomization` | clo -> cl + o | 65.45 |
| 47 | `w411_cloo_atomization` | cloo -> cl + 2 o | 126.39 |
| 48 | `w411_cn_atomization` | cn -> c + n | 181.35 |
| 49 | `w411_co_atomization` | co -> c + o | 259.73 |
| 50 | `w411_cs2_atomization` | cs2 -> c + 2 s | 280.78 |
| 51 | `w411_cs_atomization` | cs -> c + s | 172.22 |
| 52 | `w411_f2_atomization` | f2 -> 2 f | 39.04 |
| 53 | `w411_f2co_atomization` | f2co -> c + o + 2 f | 420.64 |
| 54 | `w411_f2o_atomization` | f2o -> 2 f + o | 93.78 |
| 55 | `w411_fccf_atomization` | fccf -> 2 c + 2 f | 386.09 |
| 56 | `w411_fo2_atomization` | fo2 -> f + 2 o | 134.72 |
| 57 | `w411_foof_atomization` | foof -> 2 f + 2 o | 152.37 |
| 58 | `w411_formic_atomization` | formic -> c + 2 o + 2 h | 501.90 |
| 59 | `w411_glyoxal_atomization` | glyoxal -> 2 c + 2 o + 2 h | 635.10 |
| 60 | `w411_h2cn_atomization` | h2cn -> 2 h + c + n | 343.75 |
| 61 | `w411_h2o_atomization` | h2o -> 2 h + o | 232.97 |
| 62 | `w411_h2s_atomization` | h2s -> 2 h + s | 183.91 |
| 63 | `w411_hccf_atomization` | hccf -> 2 c + f + h | 398.47 |
| 64 | `w411_hcl_atomization` | hcl -> h + cl | 107.50 |
| 65 | `w411_hcn_atomization` | hcn -> h + c + n | 313.42 |
| 66 | `w411_hcnh_atomization` | hcnh -> c + n + 2 h | 336.25 |
| 67 | `w411_hf_atomization` | hf -> h + f | 141.64 |
| 68 | `w411_hnc_atomization` | hnc -> h + c + n | 298.20 |
| 69 | `w411_hnco_atomization` | hnco -> c + o + n + h | 434.74 |
| 70 | `w411_hnnn_atomization` | hnnn -> h + 3 n | 331.79 |
| 71 | `w411_hocl_atomization` | hocl -> h + o + cl | 166.23 |
| 72 | `w411_hocn_atomization` | hocn -> c + o + n + h | 410.07 |
| 73 | `w411_hof_atomization` | hof -> h + o + f | 158.65 |
| 74 | `w411_honc_atomization` | honc -> c + o + n + h | 350.15 |
| 75 | `w411_hoo_atomization` | hoo -> h + 2 o | 175.53 |
| 76 | `w411_hooh_atomization` | hooh -> 2 h + 2 o | 269.09 |
| 77 | `w411_hs_atomization` | hs -> h + s | 87.73 |
| 78 | `w411_ketene_atomization` | ketene -> 2 c + o + 2 h | 533.46 |
| 79 | `w411_methanol_atomization` | methanol -> c + o + 4 h | 513.50 |
| 80 | `w411_n2_atomization` | n2 -> 2 n | 228.49 |
| 81 | `w411_n2h4_atomization` | n2h4 -> 4 h + 2 n | 438.28 |
| 82 | `w411_n2o_atomization` | n2o -> 2 n + o | 270.85 |
| 83 | `w411_nccn_atomization` | nccn -> 2 n + 2 c | 502.04 |
| 84 | `w411_nh2_atomization` | nh2 -> n + 2 h | 182.59 |
| 85 | `w411_nh3_atomization` | nh3 -> n + 3 h | 298.02 |
| 86 | `w411_nh_atomization` | nh -> n + h | 83.10 |
| 87 | `w411_no2_atomization` | no2 -> n + 2 o | 227.88 |
| 88 | `w411_no_atomization` | no -> n + o | 152.75 |
| 89 | `w411_o2_atomization` | o2 -> 2 o | 120.82 |
| 90 | `w411_o3_atomization` | o3 -> 3 o | 147.43 |
| 91 | `w411_oclo_atomization` | oclo -> 2 o + cl | 128.12 |
| 92 | `w411_ocs_atomization` | ocs -> o + c + s | 335.75 |
| 93 | `w411_oh_atomization` | oh -> o + h | 107.21 |
| 94 | `w411_oxirane_atomization` | oxirane -> 2 c + o + 4 h | 651.53 |
| 95 | `w411_oxirene_atomization` | oxirene -> 2 c + o + 2 h | 456.07 |
| 96 | `w411_p2_atomization` | p2 -> 2 p | 117.59 |
| 97 | `w411_p4_atomization` | p4 -> 4 p | 290.58 |
| 98 | `w411_ph3_atomization` | ph3 -> p + 3 h | 242.27 |
| 99 | `w411_propane_atomization` | propane -> 3 c + 8 h | 1007.91 |
| 100 | `w411_propene_atomization` | propene -> 3 c + 6 h | 861.58 |
| 101 | `w411_propyne_atomization` | propyne -> 3 c + 4 h | 705.61 |
| 102 | `w411_s2_atomization` | s2 -> 2 s | 104.25 |
| 103 | `w411_s3_atomization` | s3 -> 3 s | 168.36 |
| 104 | `w411_si2h6_atomization` | si2h6 -> 2 si + 6 h | 535.88 |
| 105 | `w411_sif4_atomization` | sif4 -> si + 4 f | 577.78 |
| 106 | `w411_sih3f_atomization` | sih3f -> si + 3 h + f | 382.75 |
| 107 | `w411_sih4_atomization` | sih4 -> si + 4 h | 324.94 |
| 108 | `w411_sih_atomization` | sih -> si + h | 73.92 |
| 109 | `w411_so2_atomization` | so2 -> s + 2 o | 260.62 |
| 110 | `w411_so_atomization` | so -> s + o | 126.47 |
| 111 | `w411_t-hcoh_atomization` | t-hcoh -> c + o + 2 h | 322.48 |
| 112 | `w411_t-hooo_atomization` | t-hooo -> h + 3 o | 233.30 |
| 113 | `w411_t-n2h2_atomization` | t-n2h2 -> 2 h + 2 n | 296.53 |

### 3.2a Validation slice -- BH76 barrier heights (20 names)

| # | Reaction name | Reaction | Ref (kcal/mol) |
|---|---|---|---|
| 1 | `bh76_C2H6_NH2_to_RKT20` | C2H6 + NH2 -> RKT20 | 11.30 |
| 2 | `bh76_C5H8_to_RKT22` (x2) | C5H8 -> RKT22 | 39.70 |
| 3 | `bh76_CH4_h_to_RKT03` | CH4 + h -> RKT03 | 15.00 |
| 4 | `bh76_H2O_h_to_RKT02` | H2O + h -> RKT02 | 21.60 |
| 5 | `bh76_H2_cl_to_RKT01` | H2 + cl -> RKT01 | 8.00 |
| 6 | `bh76_cl-_ch3cl_to_clch3clts` | cl- + ch3cl -> clch3clts | 2.50 |
| 7 | `bh76_cl-_ch3f_to_fch3clts` | cl- + ch3f -> fch3clts | 19.80 |
| 8 | `bh76_f-_ch3f_to_fch3fts` | f- + ch3f -> fch3fts | -0.60 |
| 9 | `bh76_h_H2_to_RKT06` (x2) | h + H2 -> RKT06 | 9.70 |
| 10 | `bh76_h_c2h4_to_c2h5ts` | h + c2h4 -> c2h5ts | 2.00 |
| 11 | `bh76_h_ch3f_to_hfch3ts` | h + ch3f -> hfch3ts | 30.50 |
| 12 | `bh76_h_n2_to_hn2ts` | h + n2 -> hn2ts | 14.60 |
| 13 | `bh76_h_n2o_to_n2ohts` | h + n2o -> n2ohts | 17.70 |
| 14 | `bh76_hcl_ch3_to_RKT08` | hcl + ch3 -> RKT08 | 1.80 |
| 15 | `bh76_hcl_h_to_hclhts` | hcl + h -> hclhts | 17.80 |
| 16 | `bh76_hf_h_to_RKT10` | hf + h -> RKT10 | 33.80 |
| 17 | `bh76_hf_h_to_hfhts` | hf + h -> hfhts | 42.10 |
| 18 | `bh76_hoch3fcomp2_to_hoch3fts` | hoch3fcomp2 -> hoch3fts | 11.00 |
| 19 | `bh76_oh_C2H6_to_RKT09` | oh + C2H6 -> RKT09 | 3.50 |
| 20 | `bh76_oh_cl_to_RKT17` | oh + cl -> RKT17 | 9.90 |

### 3.2b Validation slice -- W4-11 atomization energies (27 names)

| # | Reaction name | Reaction | Ref (kcal/mol) |
|---|---|---|---|
| 1 | `w411_allene_atomization` | allene -> 3 c + 4 h | 704.10 |
| 2 | `w411_bh3_atomization` | bh3 -> b + 3 h | 281.29 |
| 3 | `w411_bh_atomization` | bh -> b + h | 85.00 |
| 4 | `w411_c2_atomization` | c2 -> 2 c | 147.02 |
| 5 | `w411_c2h4_atomization` | c2h4 -> 2 c + 4 h | 564.10 |
| 6 | `w411_ccl2_atomization` | ccl2 -> c + 2 cl | 177.36 |
| 7 | `w411_ch2ch_atomization` | ch2ch -> 2 c + 3 h | 446.08 |
| 8 | `w411_ch2f2_atomization` | ch2f2 -> c + 2 f + 2 h | 437.67 |
| 9 | `w411_co2_atomization` | co2 -> c + 2 o | 390.14 |
| 10 | `w411_dioxirane_atomization` | dioxirane -> c + 2 o + 2 h | 410.03 |
| 11 | `w411_ethanol_atomization` | ethanol -> 2 c + o + 6 h | 811.24 |
| 12 | `w411_h2_atomization` | h2 -> 2 h | 109.49 |
| 13 | `w411_h2co_atomization` | h2co -> 2 h + c + o | 374.66 |
| 14 | `w411_hcno_atomization` | hcno -> c + o + n + h | 364.97 |
| 15 | `w411_hco_atomization` | hco -> h + c + o | 279.42 |
| 16 | `w411_hcof_atomization` | hcof -> c + o + f + h | 403.74 |
| 17 | `w411_hno_atomization` | hno -> h + n + o | 205.89 |
| 18 | `w411_n2h_atomization` | n2h -> 2 n + h | 224.86 |
| 19 | `w411_nh2cl_atomization` | nh2cl -> n + cl + 2 h | 248.06 |
| 20 | `w411_of_atomization` | of -> o + f | 53.08 |
| 21 | `w411_s2o_atomization` | s2o -> 2 s + o | 208.78 |
| 22 | `w411_s4-c2v_atomization` | s4-c2v -> 4 s | 234.35 |
| 23 | `w411_sif_atomization` | sif -> si + f | 142.71 |
| 24 | `w411_sio_atomization` | sio -> si + o | 193.05 |
| 25 | `w411_so3_atomization` | so3 -> s + 3 o | 346.94 |
| 26 | `w411_ssh_atomization` | ssh -> 2 s + h | 165.13 |
| 27 | `w411_t-hono_atomization` | t-hono -> h + n + 2 o | 312.65 |

## 4. Density species

The density comparison runs per species, not per reaction: each spec's held-out eval
records `density_rmse` (NN vs CCSD) and `density_rmse_pbe` (PBE vs CCSD, model-free) for
every non-atomic species it evaluated. On this run 213 of the 214 pool species were
evaluated -- `c2` had no benchmark reference density at eval time
(the c2 reference-grid drift documented in HISTORY Phase 35; regenerated 2026-07-26, so c2
returns in future evals). The 15 atomic species are skipped by design (`skip_reason:
"atomic_system"`) because the atomization-energy anchors make lone-atom densities
redundant, leaving 198 density species:

`C2H5_2`, `C2H6`, `C5H8`, `CH4`, `H2`, `H2O`, `H2S`, `HS`, `NH`, `NH2`, `NH3`, `PH2`, `PH3`, `RKT01`, `RKT02`, `RKT03`, `RKT04`, `RKT06`, `RKT07`, `RKT08`, `RKT09`, `RKT10`, `RKT11`, `RKT12`, `RKT14`, `RKT16`, `RKT17`, `RKT18`, `RKT19`, `RKT20`, `RKT21`, `RKT22`, `acetaldehyde`, `acetic`, `alcl`, `alcl3`, `alf`, `alf3`, `alh`, `alh3`, `allene`, `b2`, `b2h6`, `be2`, `becl2`, `bef2`, `bf`, `bf3`, `bh`, `bh3`, `bhf2`, `bn`, `bn3pi`, `c-hcoh`, `c-hono`, `c-hooo`, `c-n2h2`, `c2h2`, `c2h3f`, `c2h4`, `c2h5_1`, `c2h5f`, `c2h5ts`, `c2h6`, `c3h7`, `c3h7ts`, `cch`, `ccl2`, `cf`, `cf2`, `cf4`, `ch`, `ch2-sing`, `ch2-trip`, `ch2c`, `ch2ch`, `ch2f2`, `ch2nh`, `ch2nh2`, `ch3`, `ch3cl`, `ch3f`, `ch3fclts`, `ch3nh`, `ch3nh2`, `ch3oh`, `ch4`, `cl2`, `cl2o`, `clch3clcomp`, `clch3clts`, `clcn`, `clf`, `clo`, `cloo`, `cn`, `co`, `co2`, `cs`, `cs2`, `dioxirane`, `ethanol`, `f2`, `f2co`, `f2o`, `fccf`, `fch3clcomp1`, `fch3clcomp2`, `fch3clts`, `fch3fcomp`, `fch3fts`, `fo2`, `foof`, `formic`, `glyoxal`, `h2`, `h2cn`, `h2co`, `h2o`, `h2s`, `hccf`, `hcl`, `hclhts`, `hcn`, `hcnh`, `hcno`, `hcnts`, `hco`, `hcof`, `hcots`, `hf`, `hf2ts`, `hfch3ts`, `hfhts`, `hn2`, `hn2ts`, `hnc`, `hnco`, `hnnn`, `hno`, `hoch3fcomp1`, `hoch3fcomp2`, `hoch3fts`, `hocl`, `hocn`, `hof`, `honc`, `hoo`, `hooh`, `hs`, `ketene`, `methanol`, `n2`, `n2h`, `n2h4`, `n2o`, `n2ohts`, `nccn`, `nh`, `nh2`, `nh2cl`, `nh3`, `no`, `no2`, `o2`, `o3`, `oclo`, `ocs`, `of`, `oh`, `oh-`, `oxirane`, `oxirene`, `p2`, `p4`, `ph3`, `propane`, `propene`, `propyne`, `s2`, `s2o`, `s3`, `s4-c2v`, `si2h6`, `sif`, `sif4`, `sih`, `sih3f`, `sih4`, `sio`, `so`, `so2`, `so3`, `ssh`, `t-hcoh`, `t-hono`, `t-hooo`, `t-n2h2`

## 5. Sources

- BH76: GMTKN55-BH76 forward barrier heights (Goerigk, Hansen, Bauer, Ehrlich, Najibi, Grimme, PCCP 19 32184 (2017); scripts/script_data/gmtkn55/BH76/.res)
- W4-11: GMTKN55-W4-11 zero-point-exclusive nonrelativistic atomization energies (Karton, Daon, Martin, Chem. Phys. Lett. 510, 165 (2011); scripts/script_data/gmtkn55/W4-11/.res)

Reference densities for the density channels are CCSD (not CCSD(T)) at the run's own
basis/grid (`xcquinox.alec.benchmark_refs`); reaction reference energies are the pool
values above (kcal/mol).

## 6. Cross-references

- `README_density_figures.md` -- what every figure panel/marker/footer band means; the
  dataset footer line on held-out figures carries the live counts from Sec. 1.
- `RUNBOOK_pull_and_figures.md` -- how to pull runs and regenerate the figures.
- `xcquinox/alec/HISTORY.md` -- Phase 36 (metric + figure rationale), Phase 35 (c2 grid
  drift).
