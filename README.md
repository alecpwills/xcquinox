<div align="center" class="margin: 0 auto;">

![xcquinox-image](./xcquinox.png)

# xcquinox

</div>

[//]: # "Badges"

[![GitHub Actions Build Status](https://github.com/alecpwills/xcquinox/workflows/CI/badge.svg)](https://github.com/alecpwills/xcquinox/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/alecpwills/xcquinox/branch/main/graph/badge.svg)](https://codecov.io/gh/alecpwills/xcquinox/branch/main)
[![ReadTheDocs](https://readthedocs.org/projects/xcquinox/badge/?version=latest)](https://xcquinox.readthedocs.io/en/latest/)

A machine learning package using the equinox library for learning XC functionals with JAX.

You may find the [documentation here](https://xcquinox.readthedocs.io/en/latest/).

**WARNING: This package is still in active development and changes are frequently made. Until an official release is made, user beware.**

### Installation

To install `xcquinox`, navigate to the root package directory. Install the dependencies first with `pip install -r requirements.txt`. After installation of the dependencies is complete, install the `xcquinox` package with `pip install -e .`.

**WARNING:** Due to the active development of this package, it is possible the below patches will be deprecated or break without warning until the development cycle gets to the point where we will re-incorporate the below functionality.
To ensure integrability of the [non-local CIDER descriptors](https://github.com/mir-group/CiderPress2022), modifications to various routine calls in PySCF-AD have been made. These changes are located in the `xcquinox/patch/pyscfad.dft.patch` and `xcquinox/patch/pyscfad.scf.patch` files. You may apply these patches in whatever manner you deem easiest, but CI testing employs the use of the [pypatch](https://github.com/sitkatech/pypatch) package to update the packages with the needed patches, via

```
pypatch apply ./patch/pyscfad.dft.patch pyscfad.dft
pypatch apply ./patch/pyscfad.scf.patch pyscfad.scf
```
[pylibxc](https://libxc.gitlab.io/) The package `pylibxc` is required for some utility functions. **This package uses v4.3.4** in our utility functions. Please follow the base package's installation instructions to install this into your environment.

[PySCF-AD](https://github.com/fishjojo/pyscfad) ~~requires a specific set-up for the PySCF configuration file that tells the program to use the JAX backend. Copy `xcquinox/patch/pyscf_conf.py` into your home directory, or wherever you have PySCF looking for this configuration file.~~ no longer requires a separate `~/.pyscf_conf.py` configuration file, for versions > 0.1.4.

Once the dependencies have been installed, you may install the `xcquinox` package via `pip install -e .`.

After the installation, configuration, and patching have been completed, you may verify the functionality of the package by running `pytest -v .` in the `xcquinox/xcquinox/tests` directory.

### Copyright

Copyright (c) 2024-2025, Alec Wills

#### Acknowledgements

Project based on the
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.
