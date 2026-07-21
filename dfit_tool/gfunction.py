"""Nolte G-function and G-time.

The G-function transforms shut-in elapsed time into a dimensionless variable in which the
pre-closure pressure decline of an ideal fracture is linear. Two leakoff-exponent forms are in
common use; per the ResFrac practical guidelines, alpha=1 (low-leakoff / high-efficiency) is the
default unless permeability exceeds ~1 md, in which case alpha=0.5 is used.

Definitions (Nolte):
    dtD  = dt / te                       dimensionless shut-in time
    te                                   effective injection ("pump") time = Vinj / qmax
    dt                                   elapsed time since shut-in

    alpha = 1.0 :  g(dtD) = (4/3) [ (1+dtD)^1.5 - dtD^1.5 ]      g0 = g(0) = 4/3
    alpha = 0.5 :  g(dtD) = (1+dtD) asin((1+dtD)^-0.5) + dtD^0.5  g0 = g(0) = pi/2

    G(dtD) = (4/pi) [ g(dtD) - g0 ]
"""

from __future__ import annotations

import numpy as np

VALID_ALPHA = (1.0, 0.5)


def g_of_dtd(dtd: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """The Nolte g-function of dimensionless time dtd, for alpha in {1.0, 0.5}."""
    dtd = np.asarray(dtd, dtype=float)
    if alpha == 1.0:
        return (4.0 / 3.0) * ((1.0 + dtd) ** 1.5 - dtd ** 1.5)
    if alpha == 0.5:
        return (1.0 + dtd) * np.arcsin((1.0 + dtd) ** -0.5) + np.sqrt(dtd)
    raise ValueError(f"alpha must be one of {VALID_ALPHA}, got {alpha}")


def g0(alpha: float = 1.0) -> float:
    """g(0), the g-function at the instant of shut-in."""
    if alpha == 1.0:
        return 4.0 / 3.0
    if alpha == 0.5:
        return np.pi / 2.0
    raise ValueError(f"alpha must be one of {VALID_ALPHA}, got {alpha}")


def g_time(dt: np.ndarray, te: float, alpha: float = 1.0) -> np.ndarray:
    """G-time G(dtD) = (4/pi)[g(dtD) - g0], for shut-in elapsed time dt and pump time te.

    dt and te must be in the same units. G is zero at shut-in (dt = 0) and increases monotonically.
    """
    if te <= 0:
        raise ValueError(f"te (pump time) must be positive, got {te}")
    dt = np.asarray(dt, dtype=float)
    dtd = dt / te
    return (4.0 / np.pi) * (g_of_dtd(dtd, alpha) - g0(alpha))
