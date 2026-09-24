<div align="center" class="margin: 0 auto;">

![xcquinox-image](./xcquinox.png)

# xcquinox

</div>

[//]: # "Badges"

[![GitHub Actions Build Status](https://github.com/alecpwills/xcquinox/workflows/CI/badge.svg)](https://github.com/alecpwills/xcquinox/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/alecpwills/xcquinox/branch/main/graph/badge.svg)](https://codecov.io/gh/alecpwills/xcquinox/branch/main)
[![ReadTheDocs](https://readthedocs.org/projects/xcquinox/badge/?version=latest)](https://xcquinox.readthedocs.io/en/latest/)

Machine-learned exchange-correlation functionals: neural exchange and correlation
enhancement factors, trained against reference densities and energies through a
differentiable self-consistent field. The networks are written in JAX with the
[equinox](https://github.com/patrick-kidger/equinox) library and evaluated through
[pyscfad](https://github.com/fishjojo/pyscfad), the autodifferentiable build of PySCF. The
architecture follows [Dick and Fernandez-Serra](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.104.L161109)
(Phys. Rev. B **104**, L161109), reworked for JAX and extended with parent-anchored
enhancement factors, a pretraining-fidelity certificate and a density channel supervised
against coupled-cluster references.

Full documentation: <https://xcquinox.readthedocs.io/en/latest/>.

## Installation

```bash
git clone https://github.com/alecpwills/xcquinox.git
cd xcquinox
pip install .
```

Python 3.12 or newer. The install brings the runtime stack with it; the floors are the pins
of the cluster parity environment, the current stack around pyscfad 0.3.4, and the
declaration writes no upper bound of its own. The walkthrough of a campaign through the
cluster harness is the [getting-started page](docs/getting_started.md).

## What is here

**The package.** `xcquinox` carries the networks (`net`), the functional forms (`xc`), the
descriptor features (`features`), the training utilities (`train`) and the PySCF interface
(`utils`).

**The pipeline.** `xcquinox.pipeline` is the research pipeline built on the package: an
architecture registry, pretraining against a parent functional with a fidelity certificate,
training through a differentiable SCF with energy, potential and density channels, held-out
evaluation over GMTKN55 and W4-11 pools, and the analysis that turns a campaign into figures
and tables. The walkthrough of a campaign, from the configuration file to the pulled
results, is the [getting-started page](docs/getting_started.md).

**The cluster harness.** One configuration file drives a whole campaign as a five-stage
SLURM job graph (data generation, pretraining, preflight, training, evaluation), with the
job records, the certificates and the pulled artifacts kept beside the runs:

```bash
xcquinox-cluster prepare hpcjobs/configs/dfs_step7.dfs6311_grid3_v7g1_size.yaml
xcquinox-cluster submit  <run directory>
xcquinox-cluster status  <run directory>
```

The same commands are reachable as `python -m xcquinox.pipeline.cluster`. The cluster paths
and the queue conventions are in [hpcjobs/SEAWULF_RUNBOOK.md](hpcjobs/SEAWULF_RUNBOOK.md).

**The reproduction bundle.** `hpcjobs/` holds the job scripts, the campaign configurations
and the ledgers of the published runs; `data/` holds the reference data that is not
regenerable (the CCSD(T) references of the training subsets, the diet-set sources, the
provenance of the GMTKN55 checkout) with the small source data the package reads shipped
inside it; `reports/v7/` holds the results summary and the slides with the figure sets they
cite; `notebooks/` holds the notebooks of the development steps.

## Citation

If this work is useful to you, cite it with the metadata in
[CITATION.cff](CITATION.cff).

## Copyright

Copyright (c) 2024-2026, Alec Wills. Released under the MIT licence; see
[LICENSE](LICENSE).

#### Acknowledgements

Project structure based on the
[Computational Molecular Science Python Cookiecutter](https://github.com/MolSSI/cookiecutter-cms)
version 1.1.
