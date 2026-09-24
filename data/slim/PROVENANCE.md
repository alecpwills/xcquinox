# The Slim composition files

The three files beside this note are byte-identical copies of the composition files of the
published functional-cloning study, read by `xcquinox.pipeline.gmtkn55_sets` to build the
tracked pool caches `xcquinox/pipeline/data/slim05_pool.json` and `slim16_pool.json`
(`tools/rebuild_full_benchmark_pools.py`).

- **Upstream:** https://gitlab.com/saru1799/xcquinox-clone, branch `public`, commit
  `d6453db76c391d2c1598a3cc667e578c7cbd9106` (reference 37 of arXiv:2605.10331v1; MIT No
  Attribution).
- **Paths in the upstream tree:** `DataBases/GMTKN55/Slim100M_05_composition.txt`,
  `DataBases/GMTKN55/Slim100M_16_composition.txt`, `DataBases/GMTKN55/Slim100M_20_composition.txt`.
- **md5 of each copy:**

| file | md5 |
|---|---|
| `Slim100M_05_composition.txt` | `656d1712426dc5117212943444b0ae85` |
| `Slim100M_16_composition.txt` | `dd8e559e70e75074cf8378cb83c45bec` |
| `Slim100M_20_composition.txt` | `89167d2255e37fccf7db09b5f585b05a` |

The sets are the Slim100M sets of Gould and Vuckovic, J. Chem. Theory Comput. 21, 6517
(2025); the compositions are those of the study's `DataBases/GMTKN55` directory.

## The file grammar

One line per GMTKN55 subset: the subset's name, then the 1-based indices of the subset's
reactions that the set contains, in the order of the subset's `.res` file (`BH76/.resRC` for
BH76RC). The lines are in alphabetical order of the subset name. The systems of a set are the
species of the listed reactions; the checkout under `data/gmtkn55/` supplies their geometries,
charges and spins (`data/gmtkn55/PROVENANCE.md`).

## The study's molecule list and pretraining draw

The study's scripts (`DataBases/GMTKN55/filter_slim_all.py`, `module_utils_GMTKN55.py`,
`create_superdicts_and_polconvinfo.py`, `do_cloning/scripts_clone/clone_using_slim_densities.py`
in the upstream tree) build the molecule list of a set as follows, and
`gmtkn55_sets.paper_molecules` reproduces the rule:

- the subsets are walked in the order of the composition file, the reactions in the order
  listed, the systems of a reaction in the order of the reaction line;
- within one subset, a standard subset keeps the first system of each Hill formula and the
  conformer subsets (IDISP, ICONF, ACONF, Amino20x4, PCONF21, MCONF, SCONF, UPU23, BUT14DIOL)
  keep the first of each system name; the seen-set is reset at every subset, so a formula
  repeated in a later subset is kept again.

The cloning script draws its pretraining molecules from that list with the legacy numpy
generator seeded at `42 + i` for repetition `i` (`np.random.choice(range(N), 25,
replace=False)`, sorted). The pool cache pins the first repetition (seed 42) as
`pretrain_draw` and the drawn molecules as `pretrain_molecules`. The study's own runs set a
molecule's spin by electron parity (a single neutral atom from a table); the checkout's
`coord` metadata is used here, and each drawn molecule's record carries the study's value
as `spin_parity_rule` beside it.

The within-reaction system order of the study comes from the tables of the GMTKN55 web
pages rather than from the `.res` files; the two are taken as equal here, which the study's
authors can confirm against their logged selection.
