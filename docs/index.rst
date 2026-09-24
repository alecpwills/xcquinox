xcquinox
========

A machine-learning framework for exchange-correlation functionals: neural exchange and
correlation enhancement factors, trained against reference densities and energies through a
differentiable self-consistent field, written in `JAX`_ with the `equinox`_ library and
evaluated through `pyscfad`_, the autodifferentiable build of PySCF.

.. _JAX: https://github.com/jax-ml/jax
.. _equinox: https://github.com/patrick-kidger/equinox
.. _pyscfad: https://github.com/fishjojo/pyscfad

The package has two halves. ``xcquinox`` carries the networks, the functional forms and the
training utilities. ``xcquinox.pipeline`` carries the research pipeline built on them: the
architecture registry, the pretraining and training stages, the held-out evaluation, the
benchmark pools, and a SLURM harness that drives a whole campaign from one configuration
file.

The architecture follows `Dick and Fernandez-Serra`_ (Phys. Rev. B 104, L161109), reworked
for JAX and extended with parent-anchored enhancement factors, a pretraining-fidelity
certificate and a density channel supervised against coupled-cluster references.

.. _Dick and Fernandez-Serra: https://journals.aps.org/prb/abstract/10.1103/PhysRevB.104.L161109

.. toctree::
   :maxdepth: 2
   :caption: Documentation

   getting_started
   api

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
