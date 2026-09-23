API reference
=============

The package
-----------

The networks, the functional forms, the trainers and the PySCF interface, class by class.

.. toctree::
   :maxdepth: 2

   docpages/net
   docpages/xc
   docpages/train
   docpages/utils

The descriptor features:

.. automodule:: xcquinox.features
   :members:
   :undoc-members:
   :show-inheritance:

The pipeline
------------

The modules a caller reaches directly: the architecture registry and the specifications, the
two training stages, the evaluations, the networks and models the registry builds, the loss
channels, the reference data and the checkpoint records.

.. automodule:: xcquinox.pipeline.config
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: xcquinox.pipeline.networks
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: xcquinox.pipeline.models
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: xcquinox.pipeline.losses
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: xcquinox.pipeline.pretrain
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: xcquinox.pipeline.train
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: xcquinox.pipeline.evaluation
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: xcquinox.pipeline.eval_holdout
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: xcquinox.pipeline.data
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: xcquinox.pipeline.checkpoint_class
   :members:
   :undoc-members:
   :show-inheritance:

The cluster harness
-------------------

.. automodule:: xcquinox.pipeline.cluster.grid_config
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: xcquinox.pipeline.cluster.materialize
   :members:
   :undoc-members:
   :show-inheritance:

.. automodule:: xcquinox.pipeline.cluster.fidelity
   :members:
   :undoc-members:
   :show-inheritance:
