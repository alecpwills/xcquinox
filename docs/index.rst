Welcome to xcquinox's documentation!
=========================================================
Getting Started
===============

This package is a re-implementation of the `xc-diff`_ network architecture, originally implemented in PyTorch as a stand-alone SCF framework for Kohn-Sham density functional theory.

.. _xc-diff: https://journals.aps.org/prb/abstract/10.1103/PhysRevB.104.L161109

The architecture has been modified and translated to `JAX`-compliant methods and objects, based on the `equinox`_ library.

.. _equinox: https://github.com/patrick-kidger/equinox

This package is designed to work well with the `pyscfad`_ package, `an end-to-end autodifferentiable version of pyscf`_ written with JAX functions.

.. _pyscfad: https://github.com/fishjojo/pyscfad
.. _an end-to-end autodifferentiable version of pyscf: https://pubs.aip.org/aip/jcp/article/157/20/204801/2842264/Differentiable-quantum-chemistry-with-PySCF-for

*This is a work in progress.*

.. toctree::
   :maxdepth: 5
   :numbered:
   :hidden:
   :caption: Contents:

   getting_started
   ./docpages/net
   ./docpages/xc
   ./docpages/utils
   ./docpages/train
   ./docpages/loss
   ./docpages/pyscf

..
   api
..
   ./docpages/utils




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
