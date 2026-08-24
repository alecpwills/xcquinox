"""The orientation lock's calibrated strength, and nothing else.

The number lives on its own because two very different readers need it. The
lock operator itself (:mod:`xcquinox.alec.orientation_lock`) builds a
quadrupole integral and carries numpy for it; the HPC harness parser
(``cluster/grid_config.py``) needs only the number, and is imported by
``cluster status``, ``validate_run`` and every certificate reader, whose import
closure must stay free of numeric stacks. Restating the value in the parser
would give a physical constant two definitions and two chances to disagree --
the disagreement that already existed, a harness default of 0.0 against a
generator default of 3e-5, rebuilt a run's pretraining rows at a Hamiltonian
the run was not solved at -- so both import it from here instead. Deferring
the import into a call does not work: the parser's default is a dataclass
field, so the call runs on every configuration load.

Calibration (the argument is in ``orientation_lock``'s module docstring): the
strength is chosen so the induced pi splitting, of order 1e-6..1e-5 Ha, sits
about four orders above float64/BLAS noise and above the intrinsic
finite-basis pi asymmetry, so the lock reliably selects one representative of
the degenerate manifold, while a closed-shell PBE total energy moves by less
than 0.1 kcal/mol and the shift cancels in the like-for-like NN-vs-parent and
density comparisons, which use the same biased h_core on both sides.

This module must keep importing nothing at all.
"""

#: Recommended lock strength, in Hartree per Bohr^2 as a coefficient on the
#: traceless quadrupole of ``orientation_lock._W``. Changing it invalidates
#: every reference density, pretraining file and certificate built at the old
#: value: reference generation, training and evaluation must all apply the
#: IDENTICAL operator for the density comparison to be well defined.
DEFAULT_STRENGTH: float = 3e-5
