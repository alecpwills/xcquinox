# Package data

## `haunschild_g2/`: the G2/97 set with the Haunschild and Klopper reference energies

The G2/97 test set as the pipeline uses it: the geometries the training pool and the held-out
probes are built from, and the atomization energies of Haunschild and Klopper, "New accurate
reference energies for the G2/97 test set", J. Chem. Phys. 136, 164102 (2012),
DOI 10.1063/1.4704796 (Table I, column E_ref,non-rel: frozen-core, non-relativistic
CCSD(T)(F12)/cc-pVQZ-F12 atomization energies with higher-excitation and core/valence
corrections, in kJ/mol). The paper itself is not carried; the code cites the DOI.

| file | what it is | read by |
|---|---|---|
| `g2_97.traj` | the 148 molecules as an ASE trajectory; each entry carries its name in `atoms.info["name"]` | `xcquinox.pipeline.dfs_pool` (the training pool geometries), `xcquinox.pipeline.eval_probes` (the probe geometries), `tools/generate_dfs_pretrain_set.py` (the pretraining set exported to `xcquinox/pipeline/data/dfs_pretrain_set.json`) |
| `g2_97.csv` | Table I as parsed, 148 rows: `xyz_idx` (the entry's index in the trajectory), `Formula`, `Name`, `E` (kJ/mol) | the citations in `dfs_pool` and `eval_probes`; their kcal/mol values were converted from this column with 1 kcal = 4.184 kJ |
| `g2_97.xyz`, `haunshild_coords.xyz` | the same 148 geometries as extended XYZ and as plain XYZ | nothing at run time; the trajectory carries the geometries |
| `G297I.traj` | the same 148 entries with the fields of a 2024 evaluation in `atoms.info` (energy, atomization energy in three units, grid level, open-shell flag) | nothing at run time |
| `single_atoms.traj` | 14 single atoms as an ASE trajectory | nothing at run time |
| `haunshild_parsing.ipynb` | the 2024 notebook that parsed the paper's table and coordinates into the files above, stored without outputs | nothing |

The files are package data (`pyproject.toml`, `[tool.setuptools.package-data]`), so an
installed wheel carries them, and the loaders resolve the trajectory relative to the package
(`Path(__file__).resolve().parents[1] / "data" / "haunschild_g2"`), never to the repository.
