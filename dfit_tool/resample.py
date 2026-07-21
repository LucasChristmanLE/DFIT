"""Pressure-increment resampling and diagnostic derivatives.

After shut-in the pressure declines monotonically. Sampling at a fixed *pressure* step (default
30 psi) instead of a fixed time step collapses ~10^5 raw rows to a few hundred that are dense early
and sparse late, which is exactly what makes the numerical derivatives (dP/dG, t*dP/dt) stable. This
replaces time-domain rolling-mean smoothing.

Sign convention for the diagnostic curves: pressure declines after shut-in, so d(BHP)/dG < 0. Every
derivative curve here is reported **positive-up for a declining pressure** (i.e. negated), matching
the way G-function and log-log diagnostic plots are conventionally drawn.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .gfunction import g_time


@dataclass
class Resampled:
    dt: np.ndarray   # elapsed seconds since shut-in (>= 0), pressure-increment spaced
    p: np.ndarray    # BHP at those points (psi), monotonically decreasing
    n_raw: int       # number of raw post-shut-in samples considered
    guarded_at: int | None = None  # resampled index where the tail guard stopped, if it did


def resample_pressure_increment(
    dt: np.ndarray,
    p: np.ndarray,
    step: float = 30.0,
    rise_tol: float | None = None,
) -> Resampled:
    """Keep a point each time BHP has dropped >= ``step`` psi below the last kept point.

    ``dt`` and ``p`` are the post-shut-in samples (dt >= 0, increasing). ``rise_tol`` (default =
    ``step``) is the tail guard: once the pressure rises more than ``rise_tol`` above the running
    minimum, the late data has gone non-monotonic (gauge noise floor / temperature drift) and
    resampling stops there.
    """
    dt = np.asarray(dt, dtype=float)
    p = np.asarray(p, dtype=float)
    if rise_tol is None:
        rise_tol = step

    keep_dt: list[float] = []
    keep_p: list[float] = []
    guarded_at: int | None = None

    running_min = np.inf
    last_kept = np.inf
    for i in range(len(p)):
        pi = p[i]
        if not np.isfinite(pi):
            continue
        # First finite point is always kept as the reference.
        if not keep_dt:
            keep_dt.append(dt[i])
            keep_p.append(pi)
            last_kept = pi
            running_min = pi
            continue
        # Tail guard: sustained rise above the running minimum -> stop.
        if pi > running_min + rise_tol:
            guarded_at = len(keep_p)
            break
        running_min = min(running_min, pi)
        if pi <= last_kept - step:
            keep_dt.append(dt[i])
            keep_p.append(pi)
            last_kept = pi

    return Resampled(
        dt=np.array(keep_dt),
        p=np.array(keep_p),
        n_raw=int(len(p)),
        guarded_at=guarded_at,
    )


@dataclass
class Diagnostics:
    G: np.ndarray         # G-time
    dPdG: np.ndarray      # first derivative dP/dG, positive-up
    GdPdG: np.ndarray     # superposition semilog derivative G*dP/dG, positive-up
    # log-log falloff diagnostics vs actual shut-in time:
    t: np.ndarray         # shut-in elapsed time (> 0 only)
    p: np.ndarray         # BHP aligned with t (psi)
    dp: np.ndarray        # pressure drop since first resampled point, positive-up
    tdpdt: np.ndarray     # t * dP/dt (log-log derivative), positive-up


def diagnostics(rs: Resampled, te: float, alpha: float = 1.0) -> Diagnostics:
    """Compute G-function and log-log diagnostic curves from a resampled falloff.

    All derivatives use ``np.gradient`` against the actual abscissa (G or t), so they are correct on
    the non-uniform pressure-increment spacing.
    """
    dt = rs.dt
    p = rs.p
    G = g_time(dt, te, alpha)

    # G-function derivatives (positive-up: negate because p declines as G grows).
    dPdG = -np.gradient(p, G)
    GdPdG = G * dPdG

    # Log-log falloff: use strictly positive shut-in times.
    pos = dt > 0
    t = dt[pos]
    p_pos = p[pos]
    dp = p_pos[0] - p_pos if len(p_pos) else p_pos
    tdpdt = -t * np.gradient(p_pos, t) if len(p_pos) > 1 else np.zeros_like(t)

    return Diagnostics(G=G, dPdG=dPdG, GdPdG=GdPdG, t=t, p=p_pos, dp=dp, tdpdt=tdpdt)
