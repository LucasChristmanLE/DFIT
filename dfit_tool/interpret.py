"""DFIT interpretation math: te, ISIP (literal/effective), Shmin (compliance/tangent),
net pressure, pore pressure, and auto-suggestion helpers for the interactive picks.

Functions here are pure: they take data arrays and pick parameters and return numbers. The
interactive layer (picks.py / ui.py) owns the pick state and calls these to recompute live.

Scenario tables (closure C-A..C-D, postclosure PC-A..PC-F) and offsets (75 psi compliance,
rapid-closure 100-250 psi) are defined in ../plan.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

BBL_PER_MIN = 1.0  # BPM is bbl/min; time integrated in minutes gives bbl.
COMPLIANCE_OFFSET_PSI = 75.0


# --------------------------------------------------------------------------------------------------
# injection window + te
# --------------------------------------------------------------------------------------------------
def detect_injection_window(rate: np.ndarray, threshold: float = 0.1) -> tuple[int, int]:
    """Return (start_idx, shutin_idx) from the rate channel.

    start_idx = first sample above ``threshold``; shutin_idx = one past the last sample above
    ``threshold`` (the instant pumping stops). Raises if the rate never exceeds the threshold.
    """
    active = np.where(np.asarray(rate, dtype=float) > threshold)[0]
    if active.size == 0:
        raise ValueError("Rate never exceeds threshold; cannot auto-detect injection window")
    start_idx = int(active[0])
    shutin_idx = int(active[-1]) + 1
    return start_idx, min(shutin_idx, len(rate) - 1)


def suggest_injection_window(
    rate: np.ndarray, volume: Optional[np.ndarray] = None, threshold: float = 0.1
) -> tuple[int, int]:
    """Best-guess (start, shutin) for the *main* injection when a file has many cycles.

    A DFIT export often contains breakdown pulses, step-rate cycles, and the main injection. When a
    cumulative-volume channel is available, the main injection is taken to be the contiguous
    rate-on cycle with the largest volume gain; its start and end bound the window. Falls back to
    the whole first->last rate-on span otherwise. This is only a default -- the interpreter picks
    the true window interactively.
    """
    rate = np.asarray(rate, dtype=float)
    active = rate > threshold
    if not active.any():
        raise ValueError("Rate never exceeds threshold; cannot auto-detect injection window")
    edges = np.diff(active.astype(int))
    starts = np.where(edges == 1)[0] + 1
    ends = np.where(edges == -1)[0] + 1
    if active[0]:
        starts = np.r_[0, starts]
    if active[-1]:
        ends = np.r_[ends, len(active) - 1]

    if volume is not None and len(starts):
        v = np.asarray(volume, dtype=float)
        gains = [v[e] - v[s] for s, e in zip(starts, ends)]
        k = int(np.argmax(gains))
        return int(starts[k]), int(ends[k])
    return int(starts[0]), int(ends[-1])


def _rolling_mean(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or x.size < w:
        return x
    kernel = np.ones(w) / w
    return np.convolve(x, kernel, mode="same")


def max_sustained_rate(rate: np.ndarray, start: int, shutin: int, smooth_w: int = 15) -> float:
    """Max sustained rate over [start, shutin): peak of a short rolling mean (ignores spikes)."""
    seg = np.asarray(rate, dtype=float)[start:shutin]
    if seg.size == 0:
        return float("nan")
    return float(np.nanmax(_rolling_mean(seg, min(smooth_w, seg.size))))


@dataclass
class VolumeResult:
    vinj: float             # value used (bbl) -- delta if a volume channel exists, else integral
    vinj_delta: Optional[float]
    vinj_integral: float
    source: str             # "volume_channel" or "rate_integral"
    disagreement_frac: Optional[float]  # |delta - integral| / delta, if both available


def injected_volume(
    t_s: np.ndarray,
    rate: np.ndarray,
    start: int,
    shutin: int,
    volume: Optional[np.ndarray] = None,
) -> VolumeResult:
    """Injected volume over [start, shutin).

    Primary = cumulative-volume-channel delta when a volume channel is present; the rate integral is
    always computed as a QC cross-check (and is the fallback when no volume channel exists).
    """
    t_min = np.asarray(t_s, dtype=float) / 60.0
    q = np.asarray(rate, dtype=float)
    integral = float(np.trapezoid(q[start:shutin], t_min[start:shutin]))

    delta = None
    if volume is not None:
        v = np.asarray(volume, dtype=float)
        delta = float(v[shutin] - v[start])

    if delta is not None:
        disagree = abs(delta - integral) / delta if delta else None
        return VolumeResult(vinj=delta, vinj_delta=delta, vinj_integral=integral,
                            source="volume_channel", disagreement_frac=disagree)
    return VolumeResult(vinj=integral, vinj_delta=None, vinj_integral=integral,
                        source="rate_integral", disagreement_frac=None)


def effective_te_seconds(vinj_bbl: float, qmax_bpm: float) -> float:
    """te = Vinj / qmax, returned in **seconds** (Vinj in bbl, qmax in bbl/min)."""
    if qmax_bpm <= 0:
        raise ValueError("qmax must be positive")
    te_min = vinj_bbl / qmax_bpm
    return te_min * 60.0


# --------------------------------------------------------------------------------------------------
# lines / extrapolation
# --------------------------------------------------------------------------------------------------
def fit_line(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Least-squares fit; returns (slope, intercept)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m, b = np.polyfit(x, y, 1)
    return float(m), float(b)


def extrapolate(anchor_x: float, anchor_y: float, slope: float, target_x: float) -> float:
    """Value of the line (anchor, slope) at target_x."""
    return anchor_y + slope * (target_x - anchor_x)


# --------------------------------------------------------------------------------------------------
# ISIP
# --------------------------------------------------------------------------------------------------
def literal_isip(anchor_t: float, anchor_p: float, slope_psi_per_s: float, t_shutin: float) -> float:
    """Literal ISIP: early BHP-decline tangent extrapolated back to the shut-in instant."""
    return extrapolate(anchor_t, anchor_p, slope_psi_per_s, t_shutin)


def effective_isip(anchor_G: float, anchor_P: float, slope_P_per_G: float) -> float:
    """Effective ISIP: the P-vs-G straight line (from the min-dP/dG point) at G = 0."""
    return extrapolate(anchor_G, anchor_P, slope_P_per_G, 0.0)


# --------------------------------------------------------------------------------------------------
# Shmin + net pressure
# --------------------------------------------------------------------------------------------------
def shmin_compliance(contact_pressure: float, offset: float = COMPLIANCE_OFFSET_PSI) -> float:
    """Compliance-method Shmin = contact pressure - offset (default 75 psi)."""
    return contact_pressure - offset


def shmin_tangent(closure_pressure: float) -> float:
    """Tangent-method Shmin = BHP at the closure (departure) point."""
    return closure_pressure


def net_pressure(reference_isip: float, shmin: float) -> float:
    """Net pressure = reference ISIP - Shmin."""
    return reference_isip - shmin


# --------------------------------------------------------------------------------------------------
# pore pressure (postclosure)
# --------------------------------------------------------------------------------------------------
def pore_pressure(x_transform: np.ndarray, P: np.ndarray) -> float:
    """Pore pressure = intercept (x -> 0) of the late-time line on the chosen reciprocal-time axis.

    ``x_transform`` is t**(-1/2) or t**(-1) for the selected window; P the corresponding BHP.
    x -> 0 corresponds to infinite shut-in time.
    """
    m, b = fit_line(x_transform, P)
    return b


# --------------------------------------------------------------------------------------------------
# auto-suggestions for interactive picks
# --------------------------------------------------------------------------------------------------
def suggest_min_dpdg_index(G: np.ndarray, dPdG: np.ndarray, g_min: float = 1.0) -> int:
    """Index of the minimum of dP/dG for G >= g_min (skips the early water-hammer spike)."""
    G = np.asarray(G, dtype=float)
    mask = G >= g_min
    if not mask.any():
        mask = np.ones_like(G, dtype=bool)
    idx_local = int(np.nanargmin(np.where(mask, dPdG, np.inf)))
    return idx_local


def suggest_closure_tangent(
    G: np.ndarray, GdPdG: np.ndarray, tol_frac: float = 0.10
) -> tuple[float, int]:
    """Through-origin line fit to the early G*dP/dG data and the departure point.

    Fits slope m (through the origin) to the near-linear early segment, then walks outward and flags
    the first index where G*dP/dG departs the line by more than ``tol_frac`` of the line value. That
    index is the suggested closure. Returns (slope, departure_index).
    """
    G = np.asarray(G, dtype=float)
    y = np.asarray(GdPdG, dtype=float)
    n = len(G)
    # Use the first ~30% of points (past the very first) to define the through-origin slope.
    lo, hi = max(1, n // 20), max(2, n // 3)
    seg_G, seg_y = G[lo:hi], y[lo:hi]
    good = np.isfinite(seg_G) & np.isfinite(seg_y) & (seg_G > 0)
    if good.sum() < 2:
        return float("nan"), n - 1
    slope = float(np.sum(seg_G[good] * seg_y[good]) / np.sum(seg_G[good] ** 2))  # through-origin LS
    line = slope * G
    for i in range(hi, n):
        if line[i] > 0 and abs(y[i] - line[i]) > tol_frac * line[i]:
            return slope, i
    return slope, n - 1


def loglog_slope(t: np.ndarray, dp: np.ndarray, i0: int, i1: int) -> float:
    """Average log-log slope of dp vs t over [i0, i1] (both > 0)."""
    t = np.asarray(t, dtype=float)[i0:i1]
    dp = np.asarray(dp, dtype=float)[i0:i1]
    good = (t > 0) & (dp > 0)
    if good.sum() < 2:
        return float("nan")
    m, _ = fit_line(np.log10(t[good]), np.log10(dp[good]))
    return m
