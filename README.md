<div align="center" class="margin: 0 auto;"> 

 ![xcquinox-image](./xcquinox.png)

xcquinox
==============================

</div>

[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/alecpwills/xcquinox/workflows/CI/badge.svg)](https://github.com/alecpwills/xcquinox/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/alecpwills/xcquinox/branch/main/graph/badge.svg)](https://codecov.io/gh/alecpwills/xcquinox/branch/main)
[![ReadTheDocs](https://readthedocs.org/projects/xcquinox/badge/?version=latest)](https://xcquinox.readthedocs.io/en/latest/)


A machine learning package using the equinox library for learning XC functionals with JAX.

You may find the [documentation here](https://xcquinox.readthedocs.io/en/latest/).

### Installation

PLEASE NOTE: the version of `pip` that has been proven to work correctly in installing this package is `25.0`. `pip` version `22.0.2` (and perhaps others) may not function correctly. This version is reflected in the `requirements.txt` file, but please make sure to upgrade your `pip` version manually before installing, as `pip install -r requirements.txt` will install all packages with your current version of `pip` BEFORE updating `pip` itself. 

To install `xcquinox`, navigate to the root package directory. Install the dependencies first with `pip install -r requirements.txt`. After installation of the dependencies is complete, install the `xcquinox` package with `pip install -e .`.

To ensure integrability of the [non-local CIDER descriptors](https://github.com/mir-group/CiderPress2022), modifications to various routine calls in PySCF-AD have been made. These changes are located in the `xcquinox/patch/pyscfad.dft.patch` and `xcquinox/patch/pyscfad.scf.patch` files. You may apply these patches in whatever manner you deem easiest, but CI testing employs the use of the [pypatch](https://github.com/sitkatech/pypatch) package to update the packages with the needed patches, via
```
pypatch apply ./patch/pyscfad.dft.patch pyscfad.dft
pypatch apply ./patch/pyscfad.scf.patch pyscfad.scf
```

[PySCF-AD](https://github.com/fishjojo/pyscfad) requires a specific set-up for the PySCF configuration file that tells the program to use the JAX backend. Copy `xcquinox/patch/pyscf_conf.py` into your home directory, or wherever you have PySCF looking for this configuration file. 

Once the dependencies have been installed, you may install the `xcquinox` package via `pip install -e .`.

After the installation, configuration, and patching have been completed, you may verify the functionality of the package by running `pytest -v .` in the `xcquinox/xcquinox/tests` directory.

### Copyright

Copyright (c) 2024, Alec Wills


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.
