# Repository data

Reference data of the campaigns that is not regenerable from the package and is kept in the
tree beside the code. The package data the loaders read is under `xcquinox/data/`.

| tree | what it is | provenance |
|---|---|---|
| `training_subsets/` | the CCSD(T) reference sets of the step-7 subset study: one directory per subset (`01` to `07`, each with a `wf` variant) holding per-species trajectories with the reference energies in `atoms.info`, the CCSD(T) density matrices, orbital coefficients and occupations (`*.dm.npy`, `*.mo_coeff.npy`, `*.mo_occ.npy`) and the PySCF checkpoints | the README in each directory |
| `dietgmtkn55-150/` | the Diet GMTKN55 subset at 150 systems: the subset list `SubsetGMTKN55_150.yaml`, the element list `AllElements-150.yaml` and the ASE trajectory `diet150.traj` | the header of the subset list records the selection run (1499 systems and 213 methods in, 150 out) |
| `dietgmtkn55-50/` | the 50-system subset: the element list `AllElements_050.yaml` and the ASE trajectory `diet50.traj` | no subset list and no record of its selection run are kept; the trajectory is the definition |
| `gmtkn55/` | the GMTKN55 benchmark checkout the BH76 and W4-11 pools were parsed from; ignored except `PROVENANCE.md`, which records the upstream, the pinned commit and how to fetch it | `gmtkn55/PROVENANCE.md` |

Nothing under `data/` is read by the package at run time; the pools the package reads are the
JSON caches under `xcquinox/pipeline/data/`, rebuilt from `gmtkn55/` by
`tools/rebuild_full_benchmark_pools.py`.
