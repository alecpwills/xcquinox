# GMTKN55 -- source of truth for BH76 / BH76RC references

The full GMTKN55 benchmark is cloned (gitignored, 190M) at
`scripts/script_data/gmtkn55/` for local reference-value extraction.

- **Upstream:** https://github.com/grimme-lab/GMTKN55
- **Pinned commit:** `354c3ded3371f43545a33e7077866f4254e04917` (2025-09-17)
- Re-fetch: `git clone https://github.com/grimme-lab/GMTKN55.git scripts/script_data/gmtkn55`

The BH76RC reaction energies live in `BH76/.resRC` (W2-F12 best estimates, the
GMTKN55-BH76RC subset); barrier heights in `BH76/.res`; citations in `BH76/.bib`.

## Citations (from `BH76/.bib`)
- **Goerigk2017** -- Goerigk, Hansen, Bauer, Ehrlich, Najibi, Grimme, *Phys. Chem.
  Chem. Phys.* **19**, 32184 (2017). doi:10.1039/C7CP04913G. (GMTKN55 database)
- **Zhao2005-1** -- Zhao, Lynch, Truhlar, *PCCP* **7**, 43 (2005). doi:10.1039/B416937A.
- **Zhao2015-2** -- Zhao, González-García, Truhlar, *J. Phys. Chem. A* **109**, 2012
  (2005). doi:10.1021/jp045141s. (NHTBH barrier database)

## BH76RC reaction energies used by `eval_probes.PROBE_C_BH76_OUT_OF_TRAINING`
All in kcal/mol, reactant→product as written (W2-F12, `BH76/.resRC`):

| Reaction | ΔE (GMTKN55-BH76RC) |
|----------|---------------------|
| OH + H2 → H2O + H   | −16.39 |
| H + HCl → H2 + Cl   |  −1.90 |
| CH3 + H2 → CH4 + H  |  −3.11 |
| OH + NH3 → H2O + NH2| −10.32 |
| H + N2O → OH + N2   | −64.91 |
| H + H2S → H2 + HS   | −13.26 |

Cross-checked against Minnesota HTBH38/08 + NHTBH38/08 (REF1) Vf−Vr: agree to
<0.7 kcal/mol (the W2-F12-vs-barrier-database difference). GMTKN55 is authoritative.

## Note -- training (dfs_pool) BH76 values differ slightly from GMTKN55
`xcquinox/alec/dfs_pool.py` stores the 3 Dick training BH76 reaction energies from
the Minnesota REF1 barrier difference (Vr−Vf): +65.14 / −5.57 / +103.53. The
GMTKN55-BH76RC (W2-F12) values for the same reactions are +64.91 / −5.44 / +103.28.
Aligning training to GMTKN55 is checkpoint-affecting and is pending a user decision.
