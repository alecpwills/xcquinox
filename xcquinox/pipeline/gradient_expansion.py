"""The second-order gradient expansion of exchange: the coefficients an
exchange network's small-s curvature can be fixed at, and the resolver the
architecture registry and the network factory share.

For a slowly varying density F_x = 1 + mu s^2 + O(s^4). The exact expansion
has mu = 10/81 (Antoniewicz and Kleinman, Phys. Rev. B 31, 6779 (1985)); PBE
takes mu = beta pi^2 / 3 with libxc's beta, the value at which its exchange
cancels the gradient term of its correlation in the linear response of the
uniform gas (Perdew, Burke and Ernzerhof, Phys. Rev. Lett. 77, 3865 (1996));
SCAN keeps 10/81 (Sun, Ruzsinszky and Perdew, Phys. Rev. Lett. 115, 036402
(2015)). ``parents.py`` takes its PBE and SCAN constants from here, so each
number is defined once.
"""
from __future__ import annotations

import math

#: libxc's beta of PBE correlation.
PBE_BETA = 0.06672455060314922
#: PBE's mu = beta pi^2 / 3 (0.2195149727645171).
PBE_MU = PBE_BETA * math.pi ** 2 / 3.0
#: The exact second-order coefficient.
MU_GE = 10.0 / 81.0

#: The named coefficients an architecture may fix its curvature at
#: (``ArchitectureConfig.gea_mu``): the parent functional's own on the GGA
#: rung, or the exact expansion.
COEFFICIENTS = {"pbe": PBE_MU, "gea": MU_GE}


def resolve(value) -> float:
    """The coefficient ``value`` names: a key of :data:`COEFFICIENTS` or a
    positive finite number. Anything else raises ``ValueError`` naming the
    choices."""
    choices = sorted(COEFFICIENTS)
    if isinstance(value, str):
        if value not in COEFFICIENTS:
            raise ValueError(
                f"gea_mu must name one of {choices} or be a positive finite "
                f"number, got {value!r}")
        return COEFFICIENTS[value]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"gea_mu must name one of {choices} or be a positive finite "
            f"number, got {value!r}")
    mu = float(value)
    if not math.isfinite(mu) or mu <= 0.0:
        raise ValueError(
            f"gea_mu must be a positive finite number, got {value!r}")
    return mu
