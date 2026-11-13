# Notebooks

Every notebook in this directory is the product of a builder script and is tracked as the
builder writes it, without outputs or execution counts. `test_notebook_hygiene.py` holds the
directory to that rule: a tracked notebook has a builder in the table below, stores no outputs
and is named here, and no `checkpoints*` tree is tracked. To change a notebook, edit its builder
and regenerate it; run the builders from the repository root.

| notebook | builder | content |
|---|---|---|
| `gga_training_pipeline.ipynb` | `python notebooks/_build_gga_training_pipeline_notebook.py` | The library-driven training of the GGA networks (pretraining, training and evaluation through `xcquinox.pipeline`). |
| `gga_training_scf_solvers.ipynb` | `python notebooks/_build_gga_training_scf_solvers_notebook.py` | SCF self-consistency, the eight deep architectures across three loss approaches and three solver configurations. Its pretraining uses the unweighted objective, the step's own finding (the integration-weighted objective under-fit the exchange factor at the bound tail in this setup); the pipeline's objective is integration-weighted. |
| `gga_training_anchor_transfer.ipynb` | `python notebooks/_build_gga_training_anchor_transfer_notebook.py` | Data expansion, the PBE-anchor regularization term and the overfitting checks. Its specs cell writes the fidelity certificate the training stage requires beside every pretrained pair (the pipeline's writer, at the harness layout, on the notebook's own entities), with the example's waiver on the record: the verdict is computed and printed, never enforced, so the demonstration cannot enter the record layers. |
| `gga_training_dfs_subsets.ipynb` | `python notebooks/_build_gga_training_dfs_subsets_notebook.py` | Training and evaluation on histogram-matched subsets of the Dick 2021 pool; the flow the cluster harness mirrors (`hpcjobs/SEAWULF_RUNBOOK.md`). |
| `gga_subset_generation.ipynb` | `python notebooks/_build_subset_generation_notebook.py` | Step 7a: the histogram-matched subset generation whose ledgers the step-7 notebook and the harness consume; `tools/generate_step7_subsets.py` is its standalone form. It shares the step-7 builder's cells. |
| `dfs_selfconsistent_density/train_dfs_density.ipynb` | `python notebooks/dfs_selfconsistent_density/_build_notebook.py` | The self-consistent density training demonstration in the differentiable-programming functional form; `dfs_selfconsistent_density/README.md` is its primer and companion notes. |
| `registry_pt_transfer.ipynb` | `python notebooks/_build_registry_pt_transfer_notebook.py` | The package's miniature and tutorial: the registry's 44 architectures pretrained on the six corpus atoms and fine-tuned on the DFS-Letter miniature (16 species: the atoms, the Li/C cations, H2O, the DFS barrier trio with its transition states) under the production Letter pathway -- 25 differentiable SCF cycles per update, the Letter's loss weights with the atomic-total term, the per-update seed mixture, relaxed CCSD(T) density references -- through the campaign yaml, the cluster harness (dry-run submit, datagen, pretrain, fidelity) and the library route, then read out on proton-transfer barriers (water-dimer cation, Zundel scan, neutral dimer) against frozen CCSD(T) densities with parent-functional baselines and the oneshot-run comparison. |
| `registry_pt_prong_A.ipynb` | `python notebooks/_build_registry_pt_prong_A_notebook.py` | The registry proton-transfer case study at the production identity (6-311++G(3df,2pd), grid level 3, density fitting, SCAN seeds on the meta-GGA rung), prong A: 26 architectures (the MLPs 3x16, the three KANs 2x8, each KAN trained alone on its node) trained with per-group updates (twelve optimizer steps per epoch, the DFS scheme) for 100 epochs, each validated self-consistently as soon as it finishes (training reactions, the water-dimer cation barrier, density errors against relaxed CCSD(T), the DFS combined energy-density score). Runs on the cluster in stages (`hpcjobs/registry_pt_prep.sbatch`, `hpcjobs/registry_pt_train.sbatch`, driver `tools/run_registry_pt_stage.py`); the cells live in `_registry_pt_prongs_cells.py`, the runtime support in `registry_pt_prongs.py`. |
| `registry_pt_prong_B.ipynb` | `python notebooks/_build_registry_pt_prong_B_notebook.py` | Prong B of the same case study: the same architectures, clones and references, trained with gradient accumulation (the twelve group gradients averaged into one optimizer step per epoch). |

The tracked notebooks are held to their builders by `notebooks/test_notebook_hygiene.py` (each
product byte for byte what its builder writes) and executed end to end, slow-marked, by
`xcquinox/pipeline/tests/test_notebooks_end_to_end.py`; the density directory carries its own test (`test_dfs_demo.py`) and its own ignore file for the references, runs and
reports the notebook writes. The checkpoints the training notebooks write go under
`notebooks/checkpoints*/`, which `.gitignore` excludes.

The figure and report scripts live in `analysis/`, with their own README and runbook.
