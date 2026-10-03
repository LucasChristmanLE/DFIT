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
TAIL_ONSET_LOOKBACK_S = 300.0  # suggest_tail_trim_dt: how far before a sub-floor sample to look
                               # for the start of the collapse (bounds how much real decline is cut)
TAIL_ONSET_LOOKBACK_FRAC = 0.02  # ...and at most this fraction of the crash's elapsed time, so an
                                 # early crash does not eat the steep early decline before it
SEED_MIN_G = 1.0  # picks.seed_gfunction: the auto-seeded min dP/dG, contact, and auto C-A never sit
                  # below this G (early-decline noise); manual picks are not limited
TERMINAL_SPIKE_FACTOR = 10.0  # terminal_spike_start: trailing dP/dG beyond this x the curve's
                              # median is an end-of-record crash, not a hump
TERMINAL_SPIKE_MAX_GAP = 2  # terminal_spike_start: small samples it steps over inside a spike
TERMINAL_SPIKE_REF_G_FRAC = 0.95  # terminal_spike_start: the reference median uses G up to this
                                  # fraction of the last G, so the crash itself is left out
LEADING_SPIKE_FACTOR = 5.0  # leading_spike_end: early dP/dG beyond this x the median over
                            # G >= SEED_MIN_G is the water-hammer decay
ISIP_ANCHOR_HALF = 5  # +/- sample half-window for the apparent-ISIP tangent's local line fit:
                      # small enough to stay a true local tangent on the curving early decline,
                      # large enough to reject single-sample gauge noise.
CLOSURE_TANGENT_TOL_FRAC = 0.05  # suggest_closure_tangent: departure tolerance as a fraction of
                                 # the tangent line's value at each sample.
CLOSURE_TANGENT_MIN_PROMINENCE = 0.08  # suggest_closure_tangent: a candidate hump must stand at
                                       # least this fraction of its own height above the higher
                                       # of its two flanking bases to be eligible as the tangent
                                       # anchor -- rejects a small noise-driven local max past the
                                       # real hump, which would otherwise win on G*dP/dG alone.
INJECTION_MIN_RUN_FRAC = 0.10  # suggest_injection_window: a rate-on run smaller than this
                               # fraction of the largest run's size is dropped as a pulse/blip.
INJECTION_MAX_PLAUSIBLE_BPM = 150.0  # suggest_injection_window: a run whose median rate exceeds
                                     # this is not pumping (Cream 2C-21HZ fill reads a flat ~432).
INJECTION_MIN_SURFACE_PSI = 500.0  # suggest_injection_window: a run whose median surface pressure
                                   # is below this is a prime/fill into an open system, not the
                                   # injection (WRP Anderson 18-3-11HC: ~90 bpm at ~0 psi).
CLEAR_RISE_FRAC = 0.10  # C-A contact: dP/dG rise above the min-dP/dG value
CLEAR_RISE_MIN_POINTS = 3  # is_clear_closure: consecutive samples the rise must hold


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
    rate: np.ndarray, volume: Optional[np.ndarray] = None, threshold: float = 0.1,
    surface_p: Optional[np.ndarray] = None,
) -> tuple[int, int]:
    """Best-guess (start, shutin) for the *main* injection when a file has many cycles.

    A DFIT export often contains breakdown pulses, step-rate cycles, and the main injection, and
    the record can also carry a trailing post-shut-in rate blip (e.g. a gauge pull or bleed-off
    that nudges the rate channel briefly above ``threshold``). The rate-on mask (a non-finite
    sample counts as "not above threshold") is split into contiguous runs; each run is sized by
    its volume gain (``volume`` at one-past-its-last-active-sample minus ``volume`` at its first
    sample) when ``volume`` is given and every run's gain is finite and positive, else every run
    is sized by the sum of ``rate`` over it (one basis for all runs, never mixed). Runs whose
    median rate exceeds ``INJECTION_MAX_PLAUSIBLE_BPM`` are dropped first (a fill or prime read
    at a non-physical rate), unless every run does. When ``surface_p`` (surface or
    surface-equivalent pressure, psi) is given, runs whose median pressure is below
    ``INJECTION_MIN_SURFACE_PSI`` are dropped next (a prime or fill pumps into an open system),
    again unless every run does; a run with no finite pressure is kept. Runs smaller than ``INJECTION_MIN_RUN_FRAC`` of the largest run's size are dropped
    -- this is what rejects both an early breakdown pulse and a trailing blip, since either is
    tiny next to the main injection. The window is the *last* surviving run: start is its first
    sample, shut-in is one past its last sample (clamped to ``len(rate) - 1``). This is only a
    default -- the interpreter drags the lines to the true window.

    Raises ``ValueError`` when the rate never exceeds ``threshold`` at all.
    """
    rate = np.asarray(rate, dtype=float)
    active = rate > threshold
    active_idx = np.flatnonzero(active)
    if active_idx.size == 0:
        raise ValueError("Rate never exceeds threshold; cannot auto-detect injection window")
    gaps = np.where(np.diff(active_idx) > 1)[0]
    run_first = np.r_[0, gaps + 1]
    run_last = np.r_[gaps, active_idx.size - 1]
    runs = [(int(active_idx[a]), int(active_idx[b])) for a, b in zip(run_first, run_last)]

    # Drop runs at a non-physical rate before sizing, or their summed rate outsizes the real
    # injection. If every run is implausible the channel is likely in the wrong unit: keep all.
    plausible = [r for r in runs if np.median(rate[r[0]:r[1] + 1]) <= INJECTION_MAX_PLAUSIBLE_BPM]
    if plausible:
        runs = plausible
    if surface_p is not None:
        sp = np.asarray(surface_p, dtype=float)

        def _pressured(s: int, e: int) -> bool:
            seg = sp[s:e + 1]
            seg = seg[np.isfinite(seg)]
            return seg.size == 0 or float(np.median(seg)) >= INJECTION_MIN_SURFACE_PSI

        pressured = [r for r in runs if _pressured(*r)]
        if pressured:
            runs = pressured

    # One sizing basis for every run: volume gain only when every run's gain is finite and
    # positive (a NaN cell, a dead/flat channel, or a counter reset would otherwise put runs on
    # different scales or tie them all at 0), else summed rate for all.
    sizes = None
    if volume is not None:
        v = np.asarray(volume, dtype=float)
        gains = np.array([v[min(e + 1, len(rate) - 1)] - v[s] for s, e in runs])
        if np.all(np.isfinite(gains)) and np.all(gains > 0):
            sizes = gains
    if sizes is None:
        sizes = np.array([float(np.sum(rate[s:e + 1])) for s, e in runs])

    keep = np.where(sizes >= INJECTION_MIN_RUN_FRAC * sizes.max())[0]
    start, last = runs[int(keep[-1])]
    return start, min(last + 1, len(rate) - 1)


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
def pore_pressure_fit(x_transform: np.ndarray, P: np.ndarray) -> tuple[float, float]:
    """Late-time line on the chosen reciprocal-time axis: ``(slope, intercept)``. The intercept
    is the pore pressure (see ``pore_pressure``); the slope is reported in the results window."""
    return fit_line(x_transform, P)


def pore_pressure(x_transform: np.ndarray, P: np.ndarray) -> float:
    """Pore pressure = intercept (x -> 0) of the late-time line on the chosen reciprocal-time axis.

    ``x_transform`` is t**(-1/2) or t**(-1) for the selected window; P the corresponding BHP.
    x -> 0 corresponds to infinite shut-in time.
    """
    return pore_pressure_fit(x_transform, P)[1]


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


def terminal_spike_start(G: np.ndarray, dPdG: np.ndarray,
                         factor: float = TERMINAL_SPIKE_FACTOR) -> int:
    """Start index of a trailing crash spike in dP/dG, or ``len(dPdG)`` when there is none.

    A record that ends mid-bleed-off (or a BHP channel the low-pressure tail trim never looks at)
    leaves its last resampled points packed into a tiny G span, so dP/dG there runs 100x the
    settled curve; the final sample can even go negative where pressure ticks back up. Walking
    back from the last finite sample (which must itself qualify), a sample is part of the spike
    while its magnitude exceeds
    ``factor`` x the reference: the median |dP/dG| over G <= TERMINAL_SPIKE_REF_G_FRAC x the
    last G. Taking the reference by G, not by sample count, keeps a long bleed-off (Delphi
    46702: more crash points than falloff points, all in the last 0.03% of G) from setting its
    own reference. A genuine C-A hump is a 10-100% rise, nowhere near 10x. Non-finite samples
    (e.g. a seed mask) are ignored."""
    G = np.asarray(G, dtype=float)
    y = np.asarray(dPdG, dtype=float)
    fin = np.isfinite(y) & np.isfinite(G)
    if fin.sum() < 3:
        return len(y)
    last = int(np.flatnonzero(fin)[-1])
    ref_mask = fin & (G <= TERMINAL_SPIKE_REF_G_FRAC * G[last])
    ref_vals = np.abs(y[ref_mask]) if ref_mask.sum() >= 3 else np.abs(y[fin])
    med = float(np.median(ref_vals))
    if med <= 0:
        return len(y)
    big = np.isfinite(y) & (np.abs(y) > factor * med)
    if not big[last]:
        return len(y)
    # Walk back over the spike. Up to TERMINAL_SPIKE_MAX_GAP small or non-finite samples inside it
    # are stepped over when a spike sample lies just before them: at a down-then-up reversal the
    # central difference puts a ~0 sample in the middle of the spike.
    start = last
    i = last - 1
    while i >= 0:
        if big[i]:
            start = i
            i -= 1
            continue
        gap_end = i
        while i >= 0 and not big[i] and gap_end - i < TERMINAL_SPIKE_MAX_GAP:
            i -= 1
        if i >= 0 and big[i]:
            continue
        break
    return start


def leading_spike_end(G: np.ndarray, dPdG: np.ndarray, g_min: float = SEED_MIN_G,
                      factor: float = LEADING_SPIKE_FACTOR) -> int:
    """First index at G >= ``g_min`` past the leading water-hammer decay, i.e. the first finite
    sample whose |dP/dG| is <= ``factor`` x the median |dP/dG| over G >= ``g_min``. The decay can
    still run 100x the settled curve past G = 1; a C-A hump (10-100% rise) or a concave-up C-D
    decline never trips the threshold. Returns ``len(dPdG)`` when nothing qualifies."""
    G = np.asarray(G, dtype=float)
    y = np.asarray(dPdG, dtype=float)
    fin = np.isfinite(y) & np.isfinite(G) & (G >= g_min)
    if not fin.any():
        return len(y)
    med = float(np.median(np.abs(y[fin])))
    ok = np.flatnonzero(fin & (np.abs(y) <= factor * med))
    return int(ok[0]) if ok.size else len(y)


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
    onset_tol_psi: float = 30.0,
) -> tuple[Optional[float], str]:
    """Default boundary for the Overview step's always-on tail-trim line (shut-in-relative
    seconds), and the reason it landed there. ``dt_post``/``p_surface_post`` are the raw
    post-shut-in record (dt >= 0), aligned sample for sample.

    Two independent candidates, earliest wins:
      - ``(guard_dt, "rise_guard")`` -- the tail guard fired there. This candidate is
        informational only (see picks.seed_tail_trim): it sets no pick of its own, since the
        guard boundary is already the effective cutoff by default (interpret.resolve_tail_cut_dt)
        without one. Data past ``guard_dt`` is not guaranteed excluded from the resampled record
        any more -- ``resample.resample_pressure_increment(stop_at_guard=False)`` still keeps any
        genuine further decline past it -- but nothing past the guard is admitted into the
        diagnostics unless the analyst drags an explicit override past it
        (``PickState.tail_guard_override``).
      - ``(crash_dt, "low_pressure")`` -- the onset of the collapse that ends below
        ``floor_psi``. Found by taking the first ``dt_post`` k whose finite surface pressure is
        below ``floor_psi``, then backing up over the raw samples in
        ``[dt[k] - lookback, dt[k])``, ``lookback = min(TAIL_ONSET_LOOKBACK_S,
        TAIL_ONSET_LOOKBACK_FRAC * dt[k])``: ``p_ref`` is their finite max and the cut is
        the dt of the first sample after the LAST one with ``p >= p_ref - onset_tol_psi``, so the
        whole multi-sample collapse (not just its sub-floor end) lies past the cut. With no
        window samples, or none within tolerance, the cut stays ``dt[k]``. Skipped entirely when ``p_surface_post`` is None: a caller
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
            k = int(np.flatnonzero(below)[0])
            cut = float(dt_post[k])
            lookback = min(TAIL_ONSET_LOOKBACK_S, TAIL_ONSET_LOOKBACK_FRAC * max(cut, 0.0))
            win = np.flatnonzero((dt_post >= dt_post[k] - lookback)
                                 & (np.arange(len(dt_post)) < k) & np.isfinite(p_surface_post))
            if win.size:
                p_ref = float(np.max(p_surface_post[win]))
                ok = win[p_surface_post[win] >= p_ref - onset_tol_psi]
                if ok.size and ok[-1] + 1 < len(dt_post):
                    cut = float(dt_post[ok[-1] + 1])
            candidates.append((cut, "low_pressure"))
    if not candidates:
        return None, ""
    candidates.sort(key=lambda c: c[0])  # stable: a tie keeps rise_guard's earlier list position
    return candidates[0]


def resolve_tail_cut_dt(
    tail_trim_dt: Optional[float], guard_dt: Optional[float], override: bool
) -> Optional[float]:
    """Effective tail-trim cutoff (shut-in-relative seconds), shared by model.compute_all's
    masking and plots.render_overview's display cut. No trim set -> the guard's cutoff (or
    None if there's no guard either). A trim at/before the guard (or no guard at all) -> the
    trim, unchanged. A trim past the guard -> respected only when ``override`` is True (a
    deliberate drag past the guard, tracked by PickState.tail_guard_override); otherwise it's
    treated as stale (e.g. left behind by a shut-in move before a resync ran) and clamped back
    to guard_dt.
    """
    if tail_trim_dt is None:
        return guard_dt
    if guard_dt is None or tail_trim_dt <= guard_dt or override:
        return tail_trim_dt
    return guard_dt


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
    dPdG: np.ndarray, min_idx: int, rise_frac: float = CLEAR_RISE_FRAC
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


def is_clear_closure(G: np.ndarray, dPdG: np.ndarray, min_idx: int) -> bool:
    """True when dP/dG shows a clean C-A signature at ``min_idx``: the gate for auto-assigning
    C-A on the gfunction seed (picks.seed_gfunction).
      - ``min_idx`` is a genuine interior local min (the neighbor test suggest_min_dpdg_index
        uses), not its global-min fallback for a curve with no interior min;
      - dP/dG first reaches CLEAR_RISE_FRAC above the min at or before the hump
        (suggest_hump_index), so a late-tail rise does not count;
      - it stays at or above that level for CLEAR_RISE_MIN_POINTS consecutive samples from
        that crossing, so a single noisy sample does not count."""
    G = np.asarray(G, dtype=float)
    y = np.asarray(dPdG, dtype=float)
    if not 0 < min_idx < len(y) - 1:
        return False
    trio = y[min_idx - 1:min_idx + 2]
    if not (np.isfinite(trio).all() and trio[1] < trio[0] and trio[1] <= trio[2]):
        return False
    if trio[1] <= 0:  # a fractional rise above a min <= 0 is not a rise
        return False
    hump = suggest_hump_index(G, y)
    if hump is None or hump <= min_idx:
        return False
    c = suggest_contact_clear_index(y, min_idx)
    if c is None or c > hump:
        return False
    run = y[c:c + CLEAR_RISE_MIN_POINTS]
    threshold = y[min_idx] * (1.0 + CLEAR_RISE_FRAC)
    return len(run) == CLEAR_RISE_MIN_POINTS and bool(np.all(run >= threshold))


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


def _prominent_hump_index(
    G: np.ndarray, dPdG: np.ndarray, min_prominence: float = CLOSURE_TANGENT_MIN_PROMINENCE
) -> Optional[int]:
    """Like ``suggest_hump_index``, but restricted to interior local maxima whose *relative
    topographic prominence* is at least ``min_prominence`` -- a private helper for
    ``suggest_closure_tangent`` only; ``suggest_hump_index`` itself (the contact seed) is
    untouched.

    Past the real dP/dG hump, G*dP/dG keeps climbing even as dP/dG itself decays (as long as it
    decays slower than 1/G), so a small, noise-driven local max well past the hump can still
    have a larger G*dP/dG product than the genuine hump and win ``suggest_hump_index``'s ranking
    outright. Prominence rejects those: for a candidate at index ``i``, the left base is the
    lowest finite ``dPdG`` value between ``i`` and the nearest finite sample to its left that
    exceeds ``dPdG[i]`` (or the array start); the right base mirrors that to the right. Relative
    prominence is ``(dPdG[i] - max(left_base, right_base)) / dPdG[i]`` -- how far the candidate
    stands above the higher of its two flanking valleys, as a fraction of its own height. A
    shallow bump sitting on the shoulder of the real hump's decay has a low prominence and is
    dropped; the genuine hump, rising off a much lower flanking minimum, is not. Non-finite
    samples are skipped when walking to each base (a dropout doesn't count as a flank).

    Ranks the surviving candidates by ``G * dPdG`` exactly as ``suggest_hump_index`` does.
    Returns ``None`` when no interior local max exists at all, or none survives the prominence
    filter.
    """
    G = np.asarray(G, dtype=float)
    y = np.asarray(dPdG, dtype=float)
    n = len(y)
    finite = np.isfinite(y)
    interior = np.zeros(n, dtype=bool)
    if n >= 3:
        finite3 = finite[:-2] & finite[1:-1] & finite[2:]
        local_max = (y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:])
        interior[1:-1] = finite3 & local_max
    candidates = np.where(interior)[0]
    if candidates.size == 0:
        return None

    survivors = []
    for i in candidates:
        left_base = np.inf
        j = i - 1
        while j >= 0:
            yj = y[j]
            if np.isfinite(yj):
                if yj > y[i]:
                    break
                left_base = min(left_base, yj)
            j -= 1
        right_base = np.inf
        j = i + 1
        while j < n:
            yj = y[j]
            if np.isfinite(yj):
                if yj > y[i]:
                    break
                right_base = min(right_base, yj)
            j += 1
        base = max(left_base, right_base)
        prominence = (y[i] - base) / y[i] if np.isfinite(base) and y[i] > 0 else 0.0
        if prominence >= min_prominence:
            survivors.append(int(i))
    if not survivors:
        return None
    survivors_arr = np.array(survivors)
    return int(survivors_arr[np.argmax(G[survivors_arr] * y[survivors_arr])])


def suggest_closure_tangent(
    G: np.ndarray, GdPdG: np.ndarray, tol_frac: float = CLOSURE_TANGENT_TOL_FRAC, g_min: float = 1.0
) -> tuple[float, int]:
    """Through-origin tangent line to G*dP/dG, and the closure (departure) point.

    A line through the origin is tangent to G*dP/dG where d(G*dP/dG / G)/dG = 0, i.e. at a local
    extremum of dP/dG = G*dP/dG / G. This anchors the tangent at the dP/dG "hump" instead of
    fitting the early samples directly, because the resampled grid is densest across the early
    water-hammer spike (G -> 0) and a raw early-segment fit is dominated by it.

    ``dPdG`` is computed from the inputs (``GdPdG / G`` for finite ``G > 0``, NaN elsewhere) and
    masked to NaN below ``g_min`` -- the same water-hammer-rejection convention
    ``suggest_min_dpdg_index``/``suggest_contact_inflection_index`` use -- so the spike can never
    be mistaken for the hump. The tangent index is ``_prominent_hump_index``'s pick: the interior
    local max of that masked ``dPdG`` with the largest ``G * dPdG``, among those with relative
    prominence >= ``CLOSURE_TANGENT_MIN_PROMINENCE`` (rejecting a small noise-driven local max
    past the real hump, which the raw ``G * dPdG`` ranking alone would prefer). Slope is ``dPdG``
    at that index.

    Falls back to the through-origin least-squares fit over the first third of the ``G >= g_min``
    samples when no candidate survives (e.g. a classic monotonic dP/dG decline with no hump at
    all). Falls back further to the unmasked first-third-of-all-samples segment when that masked
    segment has fewer than 2 finite, positive-``G`` samples (e.g. a very short record, or one
    entirely below ``g_min``).

    Closure: walking forward from the tangent index (fallback: from the last index of the fit
    segment), returns the last index with ``line > 0`` and ``|G*dP/dG - line| <= tol_frac *
    line`` before the first departure -- a non-finite sample is skipped rather than counted as a
    departure, but does not extend the in-tolerance run either. This is deliberately the *first*
    contiguous in-tolerance run: a later re-crossing of the line (e.g. a rising tail) is never
    picked up. If the walk's own start index is itself out of tolerance (or non-finite, with
    nothing later ever in tolerance), the walk's start index is returned anyway -- it is never
    treated as "in tolerance" internally, but it is the closest thing to a closure this data
    offers. Returns ``n - 1`` if the curve never departs. Returns ``(nan, n - 1)`` if the slope
    could not be determined at all.
    """
    G = np.asarray(G, dtype=float)
    y = np.asarray(GdPdG, dtype=float)
    n = len(G)

    with np.errstate(divide="ignore", invalid="ignore"):
        dPdG = np.where((G > 0) & np.isfinite(G), y / G, np.nan)
    dPdG = np.where(np.isfinite(G) & (G >= g_min), dPdG, np.nan)

    tangent_idx = _prominent_hump_index(G, dPdG)

    if tangent_idx is not None:
        slope = float(dPdG[tangent_idx])
        walk_start = tangent_idx
    else:
        mask_gmin = np.isfinite(G) & (G >= g_min)
        idxs = np.where(mask_gmin)[0]
        seg_idx = None
        if idxs.size:
            m = idxs.size
            lo, hi = max(1, m // 20), max(2, m // 3)
            candidate_seg = idxs[lo:hi]
            good = np.isfinite(G[candidate_seg]) & np.isfinite(y[candidate_seg]) & (G[candidate_seg] > 0)
            if good.sum() >= 2:
                seg_idx = candidate_seg
        if seg_idx is None:
            lo, hi = max(1, n // 20), max(2, n // 3)
            hi = min(hi, n)
            lo = min(lo, hi)
            seg_idx = np.arange(lo, hi)
        seg_G, seg_y = G[seg_idx], y[seg_idx]
        good = np.isfinite(seg_G) & np.isfinite(seg_y) & (seg_G > 0)
        if good.sum() < 2:
            return float("nan"), n - 1
        gg, yy = seg_G[good], seg_y[good]
        slope = float(np.sum(gg * yy) / np.sum(gg ** 2))  # through-origin LS
        walk_start = int(seg_idx[-1]) if seg_idx.size else n - 1

    if not np.isfinite(slope):
        return float("nan"), n - 1

    line = slope * G
    last_good = None
    for i in range(walk_start, n):
        yi = y[i]
        if not np.isfinite(yi):
            continue  # a dropout is skipped, not treated as a departure
        li = line[i]
        if li > 0 and abs(yi - li) <= tol_frac * li:
            last_good = i
        else:
            break
    if last_good is None:
        last_good = walk_start
    return slope, last_good


LOGLOG_HALF_SLOPE_TOL = 0.10  # |slope + 1/2| that counts as a -1/2 (PC-A) window
LOGLOG_MIN_POINTS = 5  # samples in a suggested log-log window
LOGLOG_MAX_RMS_DECADES = 0.05  # straightness cap: RMS of the log10 fit residual
LOGLOG_MIN_SPAN_DECADES = 0.3  # a -1/2 window shorter than this is not trusted
LOGLOG_PEAK_PROMINENCE_DECADES = 0.15  # a t*dP/dt local max must stand this far above its bases
LOGLOG_PEAK_HALF_WIDTH = 3  # a peak is the max over +-this many samples


def _loglog_peak(v: np.ndarray) -> Optional[int]:
    """Index of the latest prominent local maximum of log10 t*dP/dt ``v`` that still has
    >= LOGLOG_MIN_POINTS samples after it, or None. Prominence is the usual topographic one:
    the peak minus the higher of its two bases, each base being the minimum between the peak
    and the nearest higher sample on that side (or the array end). "Latest" rather than
    "tallest" so a tall early-time spike or a pre-closure hump never outranks the postclosure
    peak the log-log window has to follow."""
    n = v.size
    w = LOGLOG_PEAK_HALF_WIDTH
    for i in range(n - 1 - LOGLOG_MIN_POINTS, 0, -1):
        if v[i] < np.max(v[max(0, i - w):i + w + 1]):
            continue
        higher_l = np.flatnonzero(v[:i] > v[i])
        left_base = np.min(v[(higher_l[-1] + 1 if higher_l.size else 0):i])
        higher_r = np.flatnonzero(v[i + 1:] > v[i])
        right_base = np.min(v[i + 1:(i + 1 + higher_r[0] if higher_r.size else n)])
        if v[i] - max(left_base, right_base) >= LOGLOG_PEAK_PROMINENCE_DECADES:
            return i
    return None


def loglog_window_slope(t: np.ndarray, y: np.ndarray, lo: float, hi: float) -> float:
    """Log-log slope of y vs t over samples with lo <= t <= hi (both edges included). NaN with
    fewer than 2 usable (t > 0, y > 0) samples."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    good = (t >= lo) & (t <= hi) & (t > 0) & (y > 0) & np.isfinite(y)
    if good.sum() < 2:
        return float("nan")
    m, _ = fit_line(np.log10(t[good]), np.log10(y[good]))
    return m


def suggest_loglog_window(t: np.ndarray,
                          y: np.ndarray) -> Optional[tuple[int, int, bool]]:
    """Suggested log-log window (i0, i1, half_slope) on t*dP/dt ``y``; i0/i1 are inclusive
    indices into ``t``.

    The window always starts after the postclosure peak (``_loglog_peak``: the latest
    prominent local maximum, so an early-time spike or a pre-closure hump is skipped). Among
    post-peak windows of >= LOGLOG_MIN_POINTS samples, the widest (in decades) that is straight
    (LOGLOG_MAX_RMS_DECADES), spans >= LOGLOG_MIN_SPAN_DECADES, and has a slope within
    LOGLOG_HALF_SLOPE_TOL of -1/2 wins, with half_slope True. Without one, the straightest
    section at any slope, with half_slope False: the widest straight window over the span
    floor, else the lowest-RMS window. None when no such peak exists (derivative still
    rising)."""
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    valid = np.flatnonzero(np.isfinite(t) & np.isfinite(y) & (t > 0) & (y > 0))
    if valid.size < LOGLOG_MIN_POINTS + 1:
        return None
    x = np.log10(t[valid])
    v = np.log10(y[valid])
    peak = _loglog_peak(v)
    if peak is None:
        return None
    xs, vs = x[peak + 1:], v[peak + 1:]
    m = xs.size

    # Closed-form LS over every (a, b) window via prefix sums: O(m^2), no per-window fit.
    def _cum(z):
        return np.concatenate([[0.0], np.cumsum(z)])
    cx, cv, cxx, cxv, cvv = _cum(xs), _cum(vs), _cum(xs * xs), _cum(xs * vs), _cum(vs * vs)
    a, b = np.meshgrid(np.arange(m), np.arange(m), indexing="ij")
    n = np.maximum(b - a + 1, 1).astype(float)
    ok = (b - a + 1) >= LOGLOG_MIN_POINTS
    sx = cx[b + 1] - cx[a]
    sv = cv[b + 1] - cv[a]
    sxx_c = cxx[b + 1] - cxx[a] - sx * sx / n
    sxv_c = cxv[b + 1] - cxv[a] - sx * sv / n
    svv_c = cvv[b + 1] - cvv[a] - sv * sv / n
    ok &= sxx_c > 0
    with np.errstate(divide="ignore", invalid="ignore"):
        slope = np.where(ok, sxv_c / np.where(sxx_c > 0, sxx_c, 1.0), np.nan)
        rms = np.sqrt(np.maximum(svv_c - slope * sxv_c, 0.0) / n)
    span = xs[b] - xs[a]
    miss = np.abs(slope + 0.5)
    good = (ok & (miss <= LOGLOG_HALF_SLOPE_TOL) & (rms <= LOGLOG_MAX_RMS_DECADES)
            & (span >= LOGLOG_MIN_SPAN_DECADES))
    straight = ok & (rms <= LOGLOG_MAX_RMS_DECADES) & (span >= LOGLOG_MIN_SPAN_DECADES)
    if good.any() or straight.any():
        score = np.where(good if good.any() else straight, span, -np.inf)
        ia, ib = np.unravel_index(int(np.argmax(score)), score.shape)
    else:
        # Nothing is straight enough: the lowest-RMS window, over the span floor when any
        # window reaches it.
        pool = ok & (span >= LOGLOG_MIN_SPAN_DECADES)
        score = np.where(pool if pool.any() else ok, rms, np.inf)
        if not np.isfinite(score).any():
            return None
        ia, ib = np.unravel_index(int(np.argmin(score)), score.shape)
    off = peak + 1
    return int(valid[off + ia]), int(valid[off + ib]), bool(good.any())


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
