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

The tracked notebooks are held to their builders by `notebooks/test_notebook_hygiene.py` (each
product byte for byte what its builder writes) and executed end to end, slow-marked, by
`xcquinox/pipeline/tests/test_notebooks_end_to_end.py`; the density directory carries its own test (`test_dfs_demo.py`) and its own ignore file for the references, runs and
reports the notebook writes. The checkpoints the training notebooks write go under
`notebooks/checkpoints*/`, which `.gitignore` excludes.

The figure and report scripts live in `analysis/`, with their own README and runbook.
