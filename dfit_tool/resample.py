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

# Tail rise guard defaults. Fixed rather than derived from ``step`` -- a smaller resample step
# (endorsed by the reference methodology) must not silently tighten the guard and start
# discarding ordinary water-hammer ringing.
RISE_GUARD_PSI = 30.0
RISE_GUARD_SUSTAIN_S = 60.0
# On its own, RISE_GUARD_SUSTAIN_S can be satisfied by just two samples at >= 60 s spacing --
# effectively the old single-excursion behavior. Also require this many above-tolerance samples
# in the run (counting its first) before the duration check is allowed to fire.
RISE_GUARD_SUSTAIN_SAMPLES = 5


@dataclass
class Resampled:
    dt: np.ndarray   # elapsed seconds since shut-in (>= 0), pressure-increment spaced
    p: np.ndarray    # BHP at those points (psi), monotonically decreasing
    n_raw: int       # number of raw post-shut-in samples considered
    guarded_at: int | None = None  # resampled index where the tail guard stopped, if it did
    # dt of the FIRST sample of the sustained run that tripped the rise guard -- not the
    # confirming sample -- or None if the guard never fired. This is still the true boundary of
    # what the resampler actually consumed: the run itself contributes nothing to keep_dt, so up
    # to a full sustain window (minus one sample) can separate the last *kept* point (dt[-1])
    # from this boundary. Callers that need "everything the resampler looked at" (e.g. the
    # low-surface-pressure scan) should bound by guard_dt, not dt[-1].
    guard_dt: float | None = None


def resample_pressure_increment(
    dt: np.ndarray,
    p: np.ndarray,
    step: float = 30.0,
    rise_tol: float | None = None,
    sustain_s: float = RISE_GUARD_SUSTAIN_S,
    sustain_samples: int = RISE_GUARD_SUSTAIN_SAMPLES,
) -> Resampled:
    """Keep a point each time BHP has dropped >= ``step`` psi below the last kept point.

    ``dt`` and ``p`` are the post-shut-in samples (dt >= 0, increasing). ``rise_tol`` (default =
    ``RISE_GUARD_PSI``, independent of ``step``) is the tail guard's tolerance above the running
    minimum. A single sample past that tolerance is not enough on its own -- gauge noise and
    water-hammer rebounds routinely poke above it for a few seconds. The guard only fires once a
    run of samples stays continuously above tolerance for >= ``sustain_s`` seconds AND contains
    >= ``sustain_samples`` such samples (counting the run's first) -- the sample-count floor
    matters at coarse (>= ``sustain_s``) sample spacing, where duration alone would already be
    satisfied by the run's second sample. Any sample at or below tolerance resets the run (and is
    itself processed normally); so does a non-finite sample -- continuity can't be confirmed
    across a dropout, so two excursions separated by missing data must not bridge into one fire.
    On fire, ``guard_dt`` is the dt of the FIRST sample of the run (not the confirming sample) and
    ``guarded_at`` is the count of points already kept at that moment -- both stay ``None`` unless
    a run actually satisfies both conditions before the record ends. Samples inside a candidate
    run are neither kept nor allowed to lower ``running_min``.
    """
    dt = np.asarray(dt, dtype=float)
    p = np.asarray(p, dtype=float)
    if rise_tol is None:
        rise_tol = RISE_GUARD_PSI

    keep_dt: list[float] = []
    keep_p: list[float] = []
    guarded_at: int | None = None
    guard_dt: float | None = None

    running_min = np.inf
    last_kept = np.inf
    run_start_dt: float | None = None
    run_start_guarded_at: int | None = None
    run_count = 0
    for i in range(len(p)):
        pi = p[i]
        if not np.isfinite(pi):
            # A dropout mid-run breaks continuity -- can't confirm the rise stayed sustained
            # across it, so reset conservatively rather than let two spikes separated by missing
            # data bridge into a false fire. Outside a run this is a no-op, same as always.
            run_start_dt = None
            run_count = 0
            continue
        # First finite point is always kept as the reference.
        if not keep_dt:
            keep_dt.append(dt[i])
            keep_p.append(pi)
            last_kept = pi
            running_min = pi
            continue
        if pi > running_min + rise_tol:
            # Above tolerance: extend the current run, or start a new one. Either way this
            # sample is excluded from keep_p/running_min -- it can't satisfy either condition.
            if run_start_dt is None:
                run_start_dt = float(dt[i])
                run_start_guarded_at = len(keep_p)
                run_count = 1
            else:
                run_count += 1
                if dt[i] - run_start_dt >= sustain_s and run_count >= sustain_samples:
                    guard_dt = run_start_dt
                    guarded_at = run_start_guarded_at
                    break
            continue
        # At or below tolerance: reset any in-progress run and fall through to normal processing.
        run_start_dt = None
        run_count = 0
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
        guard_dt=guard_dt,
    )


@dataclass
class Diagnostics:
    G: np.ndarray         # G-time
    dPdG: np.ndarray      # first derivative dP/dG, positive-up
    GdPdG: np.ndarray     # superposition semilog derivative G*dP/dG, positive-up
    d2PdG2: np.ndarray    # slope of the (positive-up) dP/dG curve: np.gradient(dPdG, G)
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
    d2PdG2 = np.gradient(dPdG, G) if len(G) > 1 else np.zeros_like(G)

    # Log-log falloff: use strictly positive shut-in times.
    pos = dt > 0
    t = dt[pos]
    p_pos = p[pos]
    dp = p_pos[0] - p_pos if len(p_pos) else p_pos
    tdpdt = -t * np.gradient(p_pos, t) if len(p_pos) > 1 else np.zeros_like(t)

    return Diagnostics(G=G, dPdG=dPdG, GdPdG=GdPdG, d2PdG2=d2PdG2, t=t, p=p_pos, dp=dp, tdpdt=tdpdt)
