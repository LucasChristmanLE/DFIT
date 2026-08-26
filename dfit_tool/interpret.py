"""DFIT interpretation math: te, ISIP (apparent/effective), Shmin (compliance/tangent),
net pressure, near-wellbore complexity, pore pressure, and auto-suggestion helpers for the
interactive picks.

Functions here are pure: they take data arrays and pick parameters and return numbers. The
interactive layer (picks.py / ui.py) owns the pick state and calls these to recompute live.

Scenario tables (closure C-A..C-D, postclosure PC-A..PC-F) and offsets (75 psi compliance,
rapid-closure 100-250 psi) are defined in ../CLAUDE.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

BBL_PER_MIN = 1.0  # BPM is bbl/min; time integrated in minutes gives bbl.
COMPLIANCE_OFFSET_PSI = 75.0
LIBERTY_OFFSET_PSI = 200.0  # Liberty-internal method: anchor BHP - 200 psi (see shmin_liberty)
RAPID_CLOSURE_RANGE_PSI = (100.0, 250.0)  # C-D: Shmin ~= apparent ISIP - (100-250 psi)
RAPID_CLOSURE_OFFSET_PSI = 175.0          # midpoint of RAPID_CLOSURE_RANGE_PSI
MIN_SURFACE_PRESSURE_PSI = 100.0  # below this, the hydrostatic BHP conversion is unreliable
                                  # (the WHP signal is too small to trust); see model.compute_all
ISIP_ANCHOR_HALF = 5  # +/- sample half-window for the apparent-ISIP tangent's local line fit:
                      # small enough to stay a true local tangent on the curving early decline,
                      # large enough to reject single-sample gauge noise.


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


def suggest_injection_window_pressure(p: np.ndarray) -> tuple[int, int]:
    """Best-guess (start, shutin) from the pressure curve alone, for rate-less datasets.

    shutin = the global pressure maximum (in a DFIT the max sits at/just before shut-in);
    start = the last upcross of baseline + 10% of the rise at/before that max (the *last*
    upcross tolerates breakdown pulses and step-rate cycles earlier in the record). The
    heuristic runs on the finite samples only -- a gauge dropout (NaN) is ignored rather
    than treated as below-threshold, so it can't fake an upcross and collapse the window
    (a run above the threshold stays one run across a dropout). Degenerate shapes (flat
    record, max at the first finite sample) fall back to positional defaults at 2% / 25%
    of the record. Raises ``ValueError`` only when fewer than 2 finite samples exist;
    otherwise always returns ``start < shutin``. This is only a default -- the interpreter
    drags the lines to the true window.
    """
    p = np.asarray(p, dtype=float)
    n = len(p)
    fin = np.where(np.isfinite(p))[0]
    if fin.size < 2:
        raise ValueError("Need at least 2 finite pressure samples to suggest a window")
    pf = p[fin]

    k = int(np.argmax(pf))  # position of the max within the finite subsequence
    shutin = int(fin[k])
    start = None
    if k > 0:
        baseline = float(np.min(pf[:k + 1]))
        rise = float(pf[k]) - baseline
        if rise > 0:
            thresh = baseline + 0.1 * rise
            above = pf[:k + 1] >= thresh
            upcross = np.where(above[1:] & ~above[:-1])[0] + 1
            if upcross.size:
                j = int(upcross[-1])
            else:  # unreachable in practice (the max is above, the baseline is below)
                j = int(np.where(above)[0][0]) if above.any() else None
            if j is not None:
                if j >= k:
                    # A single-sample jump straight to the max: keep the max, back off one
                    # finite sample.
                    j = k - 1
                start = int(fin[j])

    if start is None:
        # Flat or declining-only record: positional defaults, coerced so start < shutin.
        start = max(int(0.02 * (n - 1)), 0)
        shutin = min(max(int(0.25 * (n - 1)), start + 1), n - 1)
        start = min(start, shutin - 1)
    return start, shutin


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


def _local_line_fit(x: np.ndarray, y: np.ndarray, idx: int, half: int) -> tuple[float, float]:
    """Least-squares line over the +/-``half`` neighborhood of sample ``idx``: returns
    ``(slope, anchor_y)`` where ``anchor_y`` is the fitted line's value AT ``x[idx]``. The fit
    runs on x centered on ``x[idx]`` (numerical conditioning), so the centered intercept IS the
    anchor y and the slope is unchanged by the shift. A degenerate window (<2 points) returns
    ``(0.0, y[idx])`` -- no fit is possible, so the anchor falls back to the raw sample."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    lo, hi = max(0, idx - half), min(len(x), idx + half + 1)
    if hi - lo < 2:
        return 0.0, float(y[idx])
    m, b = fit_line(x[lo:hi] - x[idx], y[lo:hi])
    return m, b


def local_slope(x: np.ndarray, y: np.ndarray, idx: int, half: int = 4) -> float:
    """Slope of a least-squares fit over the +/-``half`` neighborhood of sample ``idx``."""
    slope, _ = _local_line_fit(x, y, idx, half)
    return slope


def tangent_from_index(x_arr: np.ndarray, y_arr: np.ndarray, idx: int,
                       half: int = 4) -> tuple[float, float, float]:
    """The tangent line anchored at sample ``idx``: ``(anchor_x, anchor_y, slope)``. Both come
    from one least-squares fit over the +/-``half`` neighborhood (``_local_line_fit``);
    ``anchor_y`` is the fitted value AT the anchor -- a true local tangent -- not the raw
    (possibly noisy) sample ``y_arr[idx]``, except in the degenerate <2-point window where it
    falls back to the raw sample (matching ``local_slope``'s slope-0.0 guard)."""
    slope, anchor_y = _local_line_fit(x_arr, y_arr, idx, half)
    return float(x_arr[idx]), float(anchor_y), slope


# --------------------------------------------------------------------------------------------------
# ISIP
# --------------------------------------------------------------------------------------------------
def apparent_isip(anchor_t: float, anchor_p: float, slope_psi_per_s: float, t_shutin: float) -> float:
    """Apparent ISIP: early BHP-decline tangent extrapolated back to the shut-in instant."""
    return extrapolate(anchor_t, anchor_p, slope_psi_per_s, t_shutin)


def effective_isip(anchor_G: float, anchor_P: float, slope_P_per_G: float) -> float:
    """Effective ISIP: the P-vs-G straight line (from the min-dP/dG point) at G = 0."""
    return extrapolate(anchor_G, anchor_P, slope_P_per_G, 0.0)


# --------------------------------------------------------------------------------------------------
# Shmin + net pressure + near-wellbore complexity
# --------------------------------------------------------------------------------------------------
def shmin_compliance(contact_pressure: float, offset: float = COMPLIANCE_OFFSET_PSI) -> float:
    """Compliance-method Shmin = contact pressure - offset (default 75 psi)."""
    return contact_pressure - offset


def shmin_liberty(anchor_pressure: float, offset: float = LIBERTY_OFFSET_PSI) -> float:
    """Liberty-internal Shmin = BHP at the method's anchor (the min-dP/dG pick for C-A, the
    inflection/contact pick for C-B) - offset (default 200 psi)."""
    return anchor_pressure - offset


def shmin_tangent(closure_pressure: float) -> float:
    """Tangent-method Shmin = BHP at the closure (departure) point."""
    return closure_pressure


def shmin_rapid(apparent_isip: float, offset: float = RAPID_CLOSURE_OFFSET_PSI) -> float:
    """C-D "rapid" Shmin = apparent (literal) ISIP - offset (default the 175 psi midpoint of
    the 100-250 psi range the ResFrac guidelines give as an approximate range; no contact pick,
    so no compliance effective ISIP, is constructed for this scenario -- the tangent effective
    ISIP still exists)."""
    return apparent_isip - offset


def format_shmin_rapid(value: float, verbose: bool = False) -> str:
    """Panel/title string for the C-D Shmin. Short form (default, e.g. ``"9325 ±75"``) fits the
    result panel's narrow value column; ``verbose=True`` appends the range it stands in for
    (e.g. ``"9325 ±75 (ISIP − 100–250)"``), used in the G-function title where width isn't
    constrained."""
    lo, hi = RAPID_CLOSURE_RANGE_PSI
    half_range = (hi - lo) / 2.0
    short = f"{value:.0f} ±{half_range:.0f}"
    if not verbose:
        return short
    return f"{short} (ISIP − {lo:.0f}–{hi:.0f})"


def net_pressure(reference_isip: float, shmin: float) -> float:
    """Net pressure = reference ISIP - Shmin."""
    return reference_isip - shmin


def near_wellbore_complexity(apparent_isip: float, reference_isip: float) -> float:
    """Near-wellbore complexity = apparent ISIP - reference (effective) ISIP: the near-wellbore
    friction/tortuosity present in the early-decline extrapolation but already dissipated by the
    time the P-vs-G line is fit. Closes the identity

        Shmin + net pressure + complexity = apparent ISIP

    for every method, since each net pressure subtracts its own Shmin from that same reference.
    """
    return apparent_isip - reference_isip


def pressure_gradient(pressure_psi: float, tvd_ft: float) -> float:
    """Pressure gradient in psi/ft = pressure / TVD. Callers guard tvd_ft > 0."""
    return pressure_psi / tvd_ft


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
def suggest_hump_index(G: np.ndarray, dPdG: np.ndarray) -> Optional[int]:
    """Index of the dP/dG "hump" -- the post-elbow rise in a C-A closure signature.

    Among interior local maxima of dP/dG (``y[i] > y[i-1] and y[i] >= y[i+1]``, finite triples --
    same neighbor test ``suggest_min_dpdg_index`` uses for its local minima), returns the one
    with the largest ``G * dPdG`` value. Multiplying by G is what lets the real hump beat the
    early water-hammer spike (which sits at G -> 0 and is itself an interior local max whenever
    it doesn't decay monotonically from index 0): the spike's own G*dPdG is suppressed by its
    tiny G, while the hump's is not.

    A monotone decaying tail past the hump has no interior local max (it only ever falls), so it
    is never a candidate here -- unlike a plain ``argmax(G * dPdG)`` over the whole curve, which
    keeps climbing for as long as the tail decays slower than 1/G and can land on a tail sample
    far past the real hump instead of the hump itself.

    Falls back to the raw finite ``argmax(dPdG)`` when no interior local max exists at all (e.g.
    a monotonic decline with no hump to find -- the old placeholder behavior, kept as the
    degenerate case). ``None`` only when nothing is finite."""
    G = np.asarray(G, dtype=float)
    y = np.asarray(dPdG, dtype=float)
    finite = np.isfinite(y)
    if not finite.any():
        return None
    interior = np.zeros(len(y), dtype=bool)
    if len(y) >= 3:
        finite3 = np.isfinite(y[:-2]) & np.isfinite(y[1:-1]) & np.isfinite(y[2:])
        local_max = (y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:])
        interior[1:-1] = finite3 & local_max
    if interior.any():
        candidates = np.where(interior)[0]
        return int(candidates[np.argmax(G[candidates] * y[candidates])])
    return int(np.flatnonzero(finite)[np.argmax(y[finite])])


def suggest_min_dpdg_index(G: np.ndarray, dPdG: np.ndarray, g_min: float = 1.0) -> int:
    """Index of the relative minimum of dP/dG -- the compliance "elbow" the effective-ISIP
    tangent should anchor at.

    Interior local minima (``dPdG[i] < dPdG[i-1] and dPdG[i] <= dPdG[i+1]``) are searched over
    the *whole* array, unmasked -- the neighbor test itself skips the spike's monotone descent,
    so no ``g_min`` mask is needed to find an elbow that sits below it (e.g. G < 1.0). When more
    than one candidate exists, those before ``suggest_hump_index``'s hump are preferred (a dip
    planted in the tail, past the hump, is not the true elbow); if that subset is empty (no hump,
    or the hump sits at index 0), the full candidate set is kept. The smallest-valued candidate
    (of whichever set applies) wins.

    Falls back to the ``g_min``-masked global minimum -- byte-identical to the pre-rewrite
    behavior -- when the curve has no interior local min at all (e.g. a monotonic decline, the
    C-C no-contact shape). An all-non-finite curve returns 0."""
    G = np.asarray(G, dtype=float)
    y = np.asarray(dPdG, dtype=float)
    if not np.isfinite(y).any():
        return 0
    interior = np.zeros(len(y), dtype=bool)
    if len(y) >= 3:
        finite3 = np.isfinite(y[:-2]) & np.isfinite(y[1:-1]) & np.isfinite(y[2:])
        local_min = (y[1:-1] < y[:-2]) & (y[1:-1] <= y[2:])
        interior[1:-1] = finite3 & local_min
    if interior.any():
        candidates = np.where(interior)[0]
        hump = suggest_hump_index(G, y)
        if hump is not None and hump > 0:
            before_hump = candidates[candidates < hump]
            if before_hump.size:
                candidates = before_hump
        return int(candidates[np.argmin(y[candidates])])
    mask = G >= g_min
    if not mask.any():
        mask = np.ones_like(G, dtype=bool)
    return int(np.nanargmin(np.where(mask, y, np.inf)))


def suggest_tail_trim_dt(
    dt_post: np.ndarray,
    p_surface_post: Optional[np.ndarray],
    guard_dt: Optional[float],
    floor_psi: float = MIN_SURFACE_PRESSURE_PSI,
) -> tuple[Optional[float], str]:
    """Default boundary for the Overview step's always-on tail-trim line (shut-in-relative
    seconds), and the reason it landed there. ``dt_post``/``p_surface_post`` are the raw
    post-shut-in record (dt >= 0), aligned sample for sample.

    Two independent candidates, earliest wins:
      - ``(guard_dt, "rise_guard")`` -- the tail guard already stopped
        ``resample.resample_pressure_increment`` there; that data never entered the resampled
        record, so this candidate is informational only (see picks.seed_tail_trim).
      - ``(crash_dt, "low_pressure")`` -- the first ``dt_post`` whose finite surface pressure
        drops below ``floor_psi``. Skipped entirely when ``p_surface_post`` is None: a caller
        passes None when the mapped channel is already BHP, where a sub-100-psi test is
        meaningless (the same gate model.compute_all's own low-pressure warning uses).

    A tie goes to "rise_guard" (checked first below). ``(None, "")`` when neither candidate
    exists -- the line parks at the end of the data, nothing trimmed.
    """
    candidates: list[tuple[float, str]] = []
    if guard_dt is not None:
        candidates.append((float(guard_dt), "rise_guard"))
    if p_surface_post is not None:
        dt_post = np.asarray(dt_post, dtype=float)
        p_surface_post = np.asarray(p_surface_post, dtype=float)
        below = np.isfinite(p_surface_post) & (p_surface_post < floor_psi)
        if below.any():
            candidates.append((float(dt_post[below][0]), "low_pressure"))
    if not candidates:
        return None, ""
    candidates.sort(key=lambda c: c[0])  # stable: a tie keeps rise_guard's earlier list position
    return candidates[0]


def min_index_in_window(G: np.ndarray, dPdG: np.ndarray, lo: float, hi: float) -> Optional[int]:
    """Index of the sample with the smallest dP/dG within ``[lo, hi]`` of G (inclusive), among
    finite samples only. ``None`` when the window holds no finite sample -- the Shift+drag
    window-correction gesture's C-A finder (``picks.handle_min_dpdg_window``)."""
    G = np.asarray(G, dtype=float)
    y = np.asarray(dPdG, dtype=float)
    mask = (G >= lo) & (G <= hi) & np.isfinite(y)
    if not mask.any():
        return None
    candidates = np.where(mask)[0]
    return int(candidates[np.argmin(y[candidates])])


def suggest_contact_clear_index(
    dPdG: np.ndarray, min_idx: int, rise_frac: float = 0.10
) -> Optional[int]:
    """C-A "clear" contact rule (URTeC-2019-123 3.1.2): the contact is the first sample right
    of the min-dP/dG pick where dP/dG has risen ``rise_frac`` (10%) above the min value.
    Returns None when the curve never rises that much -- the shape is not a clear contact."""
    y = np.asarray(dPdG, dtype=float)
    if min_idx < 0 or min_idx >= len(y) or not np.isfinite(y[min_idx]):
        return None
    threshold = y[min_idx] * (1.0 + rise_frac)
    for i in range(min_idx + 1, len(y)):
        if np.isfinite(y[i]) and y[i] >= threshold:
            return int(i)
    return None


def suggest_contact_inflection_index(
    G: np.ndarray,
    dPdG: np.ndarray,
    g_min: float = 1.0,
    seed: Optional[float] = None,
    d2: Optional[np.ndarray] = None,
    g_range: Optional[tuple] = None,
) -> Optional[int]:
    """C-B "adequate" contact rule: the inflection of a monotonically declining dP/dG -- the
    flattest point of the decline, i.e. an interior local maximum of d(dP/dG)/dG over
    G >= ``g_min`` (masking the early water-hammer region, mirroring
    ``suggest_min_dpdg_index``). Returns None when no interior local max exists (a shape with
    no flattening, e.g. a pure exponential-style decline).

    ``g_range`` (``(lo, hi)``), when given, replaces the ``g_min`` mask with
    ``lo <= G <= hi`` -- the Shift+drag window-correction gesture's C-B finder
    (``picks.handle_min_dpdg_window``) uses this to search only the hand-picked interval, and
    returns None (no fallback to the full curve) when that window holds no candidate. Default
    ``None`` preserves the ``g_min`` behavior exactly.

    ``d2`` lets a caller reuse an already-computed d2P/dG2 (e.g. ``resample.Diagnostics.d2PdG2``)
    instead of recomputing ``np.gradient`` here. ``seed`` (a G value), when given, picks the
    candidate local max **nearest** the seed instead of the tallest one -- the triangle-drag
    re-derive path (``picks.re_derive_contact_from_min``) uses this so the analyst's dragged
    seed drives which inflection is picked when the curve has more than one."""
    G = np.asarray(G, dtype=float)
    y = np.asarray(dPdG, dtype=float)
    if len(y) < 3:
        return None
    d2 = np.asarray(d2, dtype=float) if d2 is not None else np.gradient(y, G)
    if g_range is not None:
        lo, hi = g_range
        mask = (G >= lo) & (G <= hi)
    else:
        mask = G >= g_min
        if not mask.any():
            mask = np.ones_like(G, dtype=bool)
    interior = np.zeros(len(d2), dtype=bool)
    finite3 = np.isfinite(d2[:-2]) & np.isfinite(d2[1:-1]) & np.isfinite(d2[2:])
    local_max = (d2[1:-1] > d2[:-2]) & (d2[1:-1] >= d2[2:])
    interior[1:-1] = finite3 & local_max & mask[1:-1]
    if not interior.any():
        return None
    candidates = np.where(interior)[0]
    if seed is not None:
        return int(candidates[np.argmin(np.abs(G[candidates] - seed))])
    return int(candidates[np.argmax(d2[candidates])])


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


# --------------------------------------------------------------------------------------------------
# relative stiffness (URTeC-2019-123 A.8/A.9): the h-function is a time-convolution leakoff
# integral against a pore-pressure estimate; relative stiffness S = -dP_eff/dh, and its upturn
# off the minimum marks the fracture walls coming into contact -- a fourth, comparison-only
# Shmin estimate (see model.compute_all's stiffness block).
# --------------------------------------------------------------------------------------------------
def h_function(dt_s: np.ndarray, p_eff: np.ndarray, pore_pressure_psi: float,
              te_s: float) -> np.ndarray:
    """McClure's O(n^2) reference construction (URTeC-2019-123 A.8), vectorized:

        h[i] = (p_eff[0] - Pres)*sqrt(dt[i] + te/2) + sum_{j<i} (p_eff[j+1]-p_eff[j])*sqrt(dt[i]-dt[j])

    His reference code carries dt in minutes and pressure in MPa (psi/145.04); both factors
    cancel out of this relative quantity, so this implementation takes dt in seconds and
    pressure in psi directly, matching every other unit in this module. ``dt_s`` must be
    non-decreasing (post-shut-in elapsed time) so every ``dt[i]-dt[j]`` term (j < i) is >= 0.
    """
    dt_s = np.asarray(dt_s, dtype=float)
    p_eff = np.asarray(p_eff, dtype=float)
    n = len(dt_s)
    term0 = (p_eff[0] - pore_pressure_psi) * np.sqrt(dt_s + te_s / 2.0)
    dp = np.diff(p_eff)
    # outer[i, j] = dt[i] - dt[j], for j in [0, n-2] (dp's own indices); masked to the strictly
    # lower triangle (j < i) so only the j<i terms of the sum contribute. Masking (as 0/1) BEFORE
    # the sqrt, rather than after, keeps every sqrt argument >= 0 -- the masked-out (j >= i)
    # entries would otherwise be negative (dt increasing) and raise/NaN for no reason, since
    # they're zeroed out immediately after anyway.
    outer = dt_s[:, None] - dt_s[None, :-1]
    mask = np.tril(np.ones((n, max(n - 1, 0))), k=-1)
    sqrt_term = np.sqrt(outer * mask)
    return term0 + sqrt_term @ dp


def relative_stiffness(p_eff: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Relative system stiffness S[i] = -(p_eff[i+1]-p_eff[i]) / (h[i+1]-h[i]), length n-1,
    aligned with ``p_eff[1:]``/``h[1:]``. ``np.errstate`` suppresses the divide/invalid warnings
    a dh == 0 sample would otherwise raise -- np.where still evaluates -dp/dh everywhere before
    selecting, so those warnings fire on the full arrays regardless of the mask."""
    p_eff = np.asarray(p_eff, dtype=float)
    h = np.asarray(h, dtype=float)
    dp = np.diff(p_eff)
    dh = np.diff(h)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(dh != 0, -dp / dh, np.nan)


def suggest_stiffness_upturn_index(
    S: np.ndarray, G: Optional[np.ndarray] = None, g_min: float = 1.0, rise_frac: float = 0.10
) -> Optional[int]:
    """Index of the stiffness upturn: the global minimum of S among the masked candidates
    below, then the same C-A +10%-rise rule ``suggest_contact_clear_index`` uses for the
    dP/dG elbow, applied to S instead. Falls back to the min itself when S never rises that
    much to its right (a shape with no clear upturn).

    S is a ratio of first differences and is noisy near G=0, where an early-noise sample can
    land below the true upturn's minimum (the unmasked global min then sits near the highest
    p_eff instead of at the actual contact). ``G``, when given, is the G-time array aligned
    with S (S[i] pairs with grid sample i+1, so pass ``res.diagnostics.G[1:]``) and masks the
    min-search candidates to ``G >= g_min`` -- the same convention ``suggest_min_dpdg_index``
    uses to keep dP/dG's own elbow search clear of the water-hammer spike. Falls back to all
    finite positive samples when that mask is empty (a record entirely below G=1) or when
    ``G`` is not given at all.

    ``None`` only when no finite positive sample exists at all -- S can be negative or zero
    near a dh sign flip, which is never a legitimate stiffness minimum to anchor on."""
    S = np.asarray(S, dtype=float)
    positive = np.isfinite(S) & (S > 0)
    if not positive.any():
        return None
    mask = positive
    if G is not None:
        G = np.asarray(G, dtype=float)
        g_masked = positive & (G >= g_min)
        if g_masked.any():
            mask = g_masked
    candidates = np.where(mask)[0]
    min_idx = int(candidates[np.argmin(S[candidates])])
    idx = suggest_contact_clear_index(S, min_idx, rise_frac=rise_frac)
    return idx if idx is not None else min_idx
