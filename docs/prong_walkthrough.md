# The proton-transfer prong notebooks as a tour of the package

The two prong notebooks (`notebooks/registry_pt_prong_A.ipynb`,
`notebooks/registry_pt_prong_B.ipynb`) train the same 26 architectures on
the same corpus and differ in exactly one axis: how often the optimizer
steps. Read as a tour, they exercise every layer of `xcquinox` in the
order a real study would touch it. This page walks that order and names
the package API behind each stage, so the notebooks double as an
introduction to the package itself.

The functional under study is an exchange-correlation kernel: two small
networks (an exchange head `Fx` and a correlation head `Fc`) over the
density's reduced ingredients, defined in `xcquinox/net.py` (the model
classes and the checkpoint format), `xcquinox/xc.py` (the functional
evaluation and its spin handling) and `xcquinox/pipeline/networks.py` (the
network pair constructor `create_network_pair`, the GELU MLPs, the
constraints applied inside the forward pass). Everything below builds on
those three modules.

## 1. The campaign and its identity

A run is described by one campaign dict: the architecture set, the basis,
grid level, density fitting, the orientation lock, and the epoch budget.
The notebooks build theirs with `make_campaign_dict`
(`notebooks/registry_pt_prongs.py`) and materialize it per architecture
into a `TrainingSpec` through
`TrainingSpec.from_dicts(xcquinox/pipeline/config.py)` -- the single
constructor every training entry consumes, and the place the validation
fields (`validate_every`, `patience`, `validation_molecules`) are staged.

## 2. The solver identity and the seed cache

Each rung gets its own SCF solver: `make_solver`
(`notebooks/registry_pt_prongs.py`) wraps `SolverConfig`
(`xcquinox/pipeline/solver.py`) with the rung's seed source (PBE for the
GGA rung, SCAN for the meta-GGA rung). SCAN seeds are generated into the
shared cache on first use (`data.precompute_fixed_density_data`, the
`seed_cache_dir` argument), so the first prep job pays the seed cost once
and every later job reads the cache.

## 3. The corpus and its references

The training corpus is 16 species (atoms, radicals, small molecules) with
CCSD(T) reference densities at the run identity. Each species is a
`MoleculeSpec` whose `external_data_path` points at a reference npz; the
npz schema (allowed keys, the method stamp, the lock stamp) is enforced by
the loader in `xcquinox/pipeline/data.py`. The references themselves come
from the (T)-lambda CCSD(T) driver in `xcquinox/pipeline/external_refs.py`
-- the same machinery the held-out validation set's references use, so one
density method trains and selects.

## 4. Training groups and the two update schemes

`_training_groups` (`xcquinox/pipeline/train.py`) decomposes the spec into
per-target groups: one per barrier reaction, one per ionization pair, one
per atom anchor. Each group gets a scoped loss and sub-batch from
`_build_group_loss_and_batch`, built so the channel weights
(`channel_weights` on the spec) are the sole weighting control.

The two prongs then differ only here:

- prong A steps once per group (twelve steps per epoch) -- the DFS
  per-molecule scheme, the package's own `run_training` loop reached
  through `xcquinox.pipeline._train_one_spec`;
- prong B averages the twelve group gradients and steps once per epoch --
  `train_accum_main` (`notebooks/registry_pt_prongs.py`), built from the
  package's own loop elements (the same groups, the same seed mixture, the
  same plateau controller).

## 5. The executed validation protocol

Every 10 epochs both prongs score a disjoint validation set -- the 10
reactions and 23 species of the tracked manifest
(`xcquinox/pipeline/data/si_letter_validation.json`) -- on the metric the
production campaign selected on: reaction MAE plus the square root of the
density weight times the mean relative density deviation. The machinery is
the package's: `_build_validation_data` stages the species and reactions,
`_validation_parity_metric` scores the check (its coverage semantics: a
check that loses a species or a density is ineligible and advances the
patience streak), `_BestValidationTracker` keeps the best snapshot, and
the winning model lands beside the final one as `model_val_best.eqx`.

## 6. The readout and the result tables

`readout_main` (`notebooks/registry_pt_prongs.py`) validates one network:
one self-consistent SCF per species with the training solver, energy and
density from the same solve, the per-reaction errors, the WTMAD-2 over
the training reactions, and the reference-density errors -- every score
reported with its coverage counts, non-finite energies refused rather than
averaged. Both the final and the validation-best model get their own pass.
`assemble` then writes the result tables (`summary.csv` with both legs,
`reactions.csv`, `density.csv`) from the ledger.

## Running it

The cluster path is two sbatch scripts: `hpcjobs/registry_pt_prep.sbatch`
(shared preparation; every artifact cached and resumable) then
`hpcjobs/registry_pt_train.sbatch` per prong, chained on the prep job. The
workstation path is the same notebooks at the smoke identity
(`STEP_SMOKE=1`, minutes per architecture), which is also the fastest way
to try a code change end to end.
