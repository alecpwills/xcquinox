"""A machine learning framework using the equinox library for learning XC functionals with JAX."""

import xcquinox.net
import xcquinox.xc
import xcquinox.utils
import xcquinox.train
import xcquinox.features

from xcquinox._version import __version__

# The pipeline is NOT imported here: it pulls the quantum-chemistry stack behind it, and
# nothing reaches it through the package's namespace. Import it by name:
# ``import xcquinox.pipeline as pipeline``.
