"""Tests for two interpret.py auto-seed helpers:

- ``suggest_injection_window``: sizes each contiguous rate-on run (by volume gain, else by
  summed rate) and returns the last run whose size clears ``INJECTION_MIN_RUN_FRAC`` of the
  largest -- so a multi-cycle DFIT export seeds the main injection, skipping both an earlier
  breakdown pulse and a trailing post-shut-in rate blip.
- ``suggest_closure_tangent``: anchors the through-origin tangent at the dP/dG hump (a local
  extremum of GdPdG/G, filtered to candidates with at least ``CLOSURE_TANGENT_MIN_PROMINENCE``
  relative prominence) when one exists at G >= g_min, instead of fitting the early,
  spike-dominated samples directly.
"""

from __future__ import annotations

import numpy as np
import pytest

from dfit_tool import interpret, picks
from dfit_tool.model import compute_all
from tests.helpers import injection_state, make_testdata


# --------------------------------------------------------------------------------------------------
# suggest_injection_window: last rate-on period
# --------------------------------------------------------------------------------------------------
def test_suggest_injection_window_picks_last_of_several_pulses():
    n = 1000
    rate = np.zeros(n)
    # Three short breakdown pulses, each separated by a long zero gap.
    rate[10:20] = 5.0
    rate[100:115] = 6.0
    rate[250:270] = 4.0
    # A long zero gap, then a sustained final period.
    rate[700:900] = 8.0
    start, shutin = interpret.suggest_injection_window(rate)
    assert start == 700
    assert shutin == 900  # one past the last above-threshold sample


def test_suggest_injection_window_last_surviving_run_wins_over_a_bigger_earlier_volume_gain():
    """An earlier cycle can have a much larger volume gain and still lose to the last run, as
    long as the last run's own size clears INJECTION_MIN_RUN_FRAC of the largest -- the last
    surviving run wins, not simply "the last run no matter how small"."""
    n = 1000
    rate = np.zeros(n)
    volume = np.zeros(n)
    rate[100:200] = 10.0
    volume[200:] = 5000.0     # first cycle's volume gain: 5000
    rate[700:750] = 10.0
    volume[750:] += 600.0     # final period's own gain: 600 (12% of 5000 -- survives the floor)

    start, shutin = interpret.suggest_injection_window(rate, volume=volume)
    assert (start, shutin) == (700, 750)


def test_suggest_injection_window_drops_a_tiny_final_run_below_the_size_floor():
    """A final run whose size is well under INJECTION_MIN_RUN_FRAC of the largest run's size
    (a trailing blip, e.g. a gauge pull) is dropped; the window falls back to the last run that
    still clears the floor."""
    n = 1000
    rate = np.zeros(n)
    volume = np.zeros(n)
    rate[100:200] = 10.0
    volume[200:] = 5000.0   # the main injection: volume gain 5000
    rate[900:903] = 0.5     # a tiny trailing blip
    volume[903:] += 1.0     # negligible gain: 1.0, well under 10% of 5000

    start, shutin = interpret.suggest_injection_window(rate, volume=volume)
    assert (start, shutin) == (100, 200)


def test_suggest_injection_window_single_ramp_matches_old_behavior():
    """The helpers.make_testdata shape (one contiguous rate-on span): unchanged result."""
    td = make_testdata()
    rate = td.column("RATE")
    start, shutin = interpret.suggest_injection_window(rate)
    active = np.where(np.asarray(rate) > 0.1)[0]
    assert start == int(active[0])
    assert shutin == min(int(active[-1]) + 1, len(rate) - 1)


def test_suggest_injection_window_active_through_end_of_record():
    n = 50
    rate = np.full(n, 5.0)
    start, shutin = interpret.suggest_injection_window(rate)
    assert start == 0
    assert shutin == n - 1


def test_suggest_injection_window_all_zero_raises():
    rate = np.zeros(100)
    with pytest.raises(ValueError):
        interpret.suggest_injection_window(rate)


def test_suggest_injection_window_drops_a_trailing_post_shutin_blip_rate_only():
    """The motivating real-record shape: main injection, a long quiet gap, then a 2-3 sample
    rate blip (e.g. a gauge pull) just above threshold. No volume channel -- sizing falls back
    to summed rate, and the blip's tiny sum is still well under the size floor."""
    n = 1000
    rate = np.zeros(n)
    rate[100:700] = 6.5          # the main injection: 600 samples at 6.5 bpm
    rate[850:853] = np.array([0.18, 0.18, 0.69])  # a trailing blip, same magnitude as the real file
    start, shutin = interpret.suggest_injection_window(rate)
    assert (start, shutin) == (100, 700)


def _pulse_main_blip():
    """Early pulse, main injection, trailing blip; volume is the running integral of rate."""
    n = 1000
    rate = np.zeros(n)
    rate[50:80] = 3.0            # early pulse
    rate[100:700] = 6.5          # main injection
    rate[850:853] = np.array([0.18, 0.18, 0.69])  # trailing blip
    volume = np.cumsum(rate) / 60.0
    return rate, volume


def test_suggest_injection_window_nan_volume_cell_does_not_mix_sizing_units():
    """One NaN volume cell at a run edge must not put that run on the summed-rate scale while
    the others use bbl gain -- every run falls back to summed rate together."""
    rate, volume = _pulse_main_blip()
    volume[80] = np.nan
    assert interpret.suggest_injection_window(rate, volume) == (100, 700)


@pytest.mark.parametrize("volume_kind", ["zeros", "reset"])
def test_suggest_injection_window_dead_or_reset_volume_falls_back_to_rate(volume_kind):
    """A flat channel (every gain 0) or a counter reset (a negative gain) would otherwise tie or
    shrink the main run and let the trailing blip win."""
    rate, volume = _pulse_main_blip()
    if volume_kind == "zeros":
        volume = np.zeros_like(volume)
    else:
        volume[700:] -= volume[700]  # counter resets at shut-in
    assert interpret.suggest_injection_window(rate, volume) == (100, 700)


def test_suggest_injection_window_nan_counts_as_not_above_threshold():
    n = 200
    rate = np.zeros(n)
    rate[10:30] = 5.0
    rate[30:40] = np.nan  # a dropout inside would-be-active data
    rate[40:60] = 5.0
    start, shutin = interpret.suggest_injection_window(rate)
    # The NaN samples aren't > threshold, so they split the run: last "above" run is [40, 60).
    assert start == 40
    assert shutin == 60


# --------------------------------------------------------------------------------------------------
# suggest_closure_tangent: hump-anchored tangent, first-in-tolerance-run closure
# --------------------------------------------------------------------------------------------------
def test_suggest_closure_tangent_anchors_at_hump_not_the_early_spike():
    """Dense grid near G=0 (mimicking the resampled grid), a big spike in GdPdG near G=0.2, a
    min near G=2, then a hump. The old fit (first ~30% of all samples) would be dominated by the
    spike; the new one must ignore it and anchor near the hump, with closure at G >= 1."""
    G = np.r_[np.linspace(0.01, 1.0, 80), np.linspace(1.1, 40.0, 40)]
    dPdG = np.full_like(G, 5.0)
    # Water-hammer spike below G=1.
    spike_mask = np.abs(G - 0.2) < 0.05
    dPdG[spike_mask] = 3000.0
    # Declining dP/dG with a hump (local max) around G=8, then declining again.
    decline_mask = (G >= 1.0) & (G <= 8.0)
    dPdG[decline_mask] = 40.0 - 3.0 * (G[decline_mask] - 1.0)
    hump_mask = (G > 8.0) & (G <= 12.0)
    dPdG[hump_mask] = 21.0 + 1.5 * (G[hump_mask] - 8.0)  # rises to a peak
    tail_mask = G > 12.0
    dPdG[tail_mask] = 27.0 - 0.3 * (G[tail_mask] - 12.0)  # then falls off

    GdPdG = G * dPdG
    slope, idx = interpret.suggest_closure_tangent(G, GdPdG)

    # The tangent slope must equal the hump's own dP/dG value (~27, the peak the hump_mask/
    # tail_mask ramps approach at G=12), not the spike (dPdG ~ 3000 below G=1).
    assert slope == pytest.approx(27.0, abs=1.0)
    assert 1.0 <= G[idx] <= 25.0  # closure sits past the hump, never in the sub-G=1 spike


def test_suggest_closure_tangent_realistic_spike_shape_closure_at_or_above_g1():
    """The exact shape called out in the plan: dense grid near 0, a big GdPdG spike near G=0.2,
    a min near G=2, and a hump -- assert the returned closure G is >= 1."""
    G = np.r_[np.linspace(0.01, 1.0, 80), np.linspace(1.1, 40.0, 40)]
    dPdG = np.empty_like(G)
    for i, g in enumerate(G):
        if g < 1.0:
            dPdG[i] = 5.0
        elif g < 2.0:
            dPdG[i] = 5.0 - 3.0 * (g - 1.0)  # decline toward a min near G=2
        elif g < 10.0:
            dPdG[i] = 2.0 + 2.0 * (g - 2.0) / 8.0  # rise to a hump
        else:
            dPdG[i] = 4.0 - 0.05 * (g - 10.0)  # gentle decline past the hump
    # Big spike near G=0.2, well below g_min -- must never win the tangent.
    dPdG[np.abs(G - 0.2) < 0.03] = 4000.0
    GdPdG = G * dPdG

    slope, idx = interpret.suggest_closure_tangent(G, GdPdG)
    # The hump peaks at dPdG=4.0 (G=10); the tangent slope must come from there, not the spike.
    assert slope == pytest.approx(4.0, abs=0.2)
    assert 1.0 <= G[idx] <= 20.0


def test_suggest_closure_tangent_fallback_matches_known_line_when_no_hump():
    """A classic monotonic decline with no local max of dP/dG at G >= 1 (dP/dG is flat, i.e. a
    perfect through-origin line, then bends DOWN -- never up, so no interior local max is ever
    formed): the fallback through-origin LS should recover the known slope, and closure = last
    in-tolerance sample, right around the bend."""
    n = 300
    G = np.linspace(0.01, 30.0, n)
    slope_true = 50.0
    dPdG = np.full(n, slope_true)
    bend = G > 20.0
    dPdG[bend] = np.clip(slope_true - 4.0 * (G[bend] - 20.0), 0.1, None)
    GdPdG = G * dPdG

    slope, idx = interpret.suggest_closure_tangent(G, GdPdG)
    assert slope == pytest.approx(slope_true, rel=0.05)
    assert G[idx] < 22.0  # closure lands right at/just past the bend
    assert G[idx] > 18.0


def test_suggest_closure_tangent_late_recrossing_does_not_extend_closure():
    """A curve that departs the tangent line (dP/dG dips), then rises back toward the tangent
    value late in the record (a "re-crossing" on the G*dP/dG scale): the closure must stay on
    the first contiguous in-tolerance run, not the later re-crossing. dP/dG only ever decreases
    or increases monotonically within each segment and never turns over before the array ends,
    so no interior local max is ever formed and the fallback fit is exercised throughout."""
    n = 300
    G = np.linspace(0.01, 40.0, n)
    slope_true = 20.0
    dPdG = np.empty(n)
    flat = G <= 15.0
    dip = (G > 15.0) & (G <= 30.0)
    rise = G > 30.0
    dPdG[flat] = slope_true
    dPdG[dip] = slope_true - 0.7 * slope_true * (G[dip] - 15.0) / (30.0 - 15.0)
    dPdG[rise] = 0.3 * slope_true + 0.7 * slope_true * (G[rise] - 30.0) / (40.0 - 30.0)
    GdPdG = G * dPdG

    slope, idx = interpret.suggest_closure_tangent(G, GdPdG)
    assert slope == pytest.approx(slope_true, rel=0.05)
    assert G[idx] <= 20.0  # closure sits at the first departure (just past G=15)
    assert G[idx] < 30.0   # nowhere near the late re-crossing region


def test_suggest_closure_tangent_short_record_below_g_min_uses_unmasked_fallback():
    """All G < 1 (or a very short record): still returns a finite slope and a valid index via
    the unmasked fallback."""
    G = np.linspace(0.01, 0.9, 10)
    slope_true = 12.0
    GdPdG = slope_true * G
    slope, idx = interpret.suggest_closure_tangent(G, GdPdG)
    assert np.isfinite(slope)
    assert 0 <= idx < len(G)


def test_suggest_closure_tangent_never_departs_returns_last_index():
    n = 50
    G = np.linspace(0.01, 10.0, n)
    slope_true = 30.0
    GdPdG = slope_true * G
    slope, idx = interpret.suggest_closure_tangent(G, GdPdG)
    assert idx == n - 1


@pytest.mark.parametrize("n_at_gmin", [2, 3, 5, 8])
def test_suggest_closure_tangent_few_samples_at_g_min_still_finite(n_at_gmin):
    """2-8 samples at G >= g_min (a short record, no hump): the masked fallback fit segment can
    itself end up with fewer than 2 finite samples once sliced to "the first third" -- that must
    drop to the unmasked segment rather than return NaN (the bug: checking finiteness only
    *before* slicing to the first third was too weak a trigger)."""
    G = np.r_[np.linspace(0.01, 0.95, 40), np.linspace(1.1, 4.0, n_at_gmin)]
    dPdG = 20.0 / np.sqrt(G + 0.1)  # monotonically decreasing -- no hump anywhere
    slope, idx = interpret.suggest_closure_tangent(G, G * dPdG)
    assert np.isfinite(slope)
    assert 0 <= idx < len(G)


def test_suggest_closure_tangent_nan_inside_walk_is_skipped_not_extended():
    """A single non-finite G*dP/dG sample partway through an otherwise-in-tolerance walk region
    must be skipped -- it does not count as a departure (halting the walk early) and does not
    get recorded as the closure (extending it past where the real data actually stays in
    tolerance). The result must be byte-identical to the same curve with no NaN inserted."""
    n = 300
    G = np.linspace(0.01, 30.0, n)
    slope_true = 50.0
    dPdG = np.full(n, slope_true)
    bend = G > 20.0
    dPdG[bend] = np.clip(slope_true - 4.0 * (G[bend] - 20.0), 0.1, None)
    GdPdG = G * dPdG

    slope0, idx0 = interpret.suggest_closure_tangent(G, GdPdG)

    GdPdG_with_nan = GdPdG.copy()
    GdPdG_with_nan[155] = np.nan  # well inside the in-tolerance run, before the departure
    slope1, idx1 = interpret.suggest_closure_tangent(G, GdPdG_with_nan)

    assert slope1 == slope0
    assert idx1 == idx0


def test_suggest_closure_tangent_fallback_walk_start_out_of_tolerance_returns_start():
    """When the fallback fit segment's own last sample (the walk's start index) is itself
    outside tolerance of the fitted line -- and nothing later ever comes back into tolerance --
    the function returns that start index anyway. It must not be silently credited as "in
    tolerance" internally; it is returned only because there is nothing better to report."""
    # G/dPdG chosen so g_min=1.0 masks nothing (G starts at 1.0); with m=10 samples at G>=1,
    # the fit segment is exactly G=[2,3] (indices 1,2 of this array) and the walk starts at
    # G=3. dP/dG increases monotonically throughout (5,10,50,60,...,120) so no interior local
    # max ever forms and the fallback fit is used. The through-origin LS over G=[2,3] pulls the
    # line to slope (2*20+3*150)/(2**2+3**2) = 37.69..., which G=3's own y=150 departs from by
    # far more than the 5% tolerance -- immediately, at the walk's very first checked sample.
    G = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    dPdG = np.array([5, 10, 50, 60, 70, 80, 90, 100, 110, 120], dtype=float)
    GdPdG = G * dPdG

    slope, idx = interpret.suggest_closure_tangent(G, GdPdG)
    expected_slope = (2 * 20.0 + 3 * 150.0) / (2.0 ** 2 + 3.0 ** 2)
    assert slope == pytest.approx(expected_slope)
    assert G[idx] == 3.0  # the walk's own start index, returned as the fallback


def test_suggest_closure_tangent_noisy_late_tail_stays_near_the_real_hump():
    """The real motivating record (G, GdPdG pairs from a resampled DFIT diagnostic curve, hard-
    coded here so this test needs no data file): past the genuine hump (~G=31, dP/dG ~18.2),
    G*dP/dG keeps rising for a long stretch even as dP/dG itself declines. A noise-driven local
    max anywhere in that stretch can out-rank the real hump on raw G*dP/dG alone; prominence
    filtering must keep the seed near the real hump across many noisy trials."""
    G = np.array([
        0.0, 0.03, 0.037, 0.044, 0.051, 0.085, 0.105, 0.233, 0.561, 1.065, 1.923, 3.277, 5.196,
        7.422, 9.568, 11.607, 13.619, 15.575, 17.474, 19.337, 21.156, 22.942, 24.685, 26.391,
        28.065, 29.732, 31.378, 33.025, 34.681, 36.361, 38.061, 39.77, 41.508, 43.284, 45.122,
        47.006, 48.926, 50.902, 52.943, 55.073, 57.299, 59.61, 62.128, 64.797, 67.664, 70.683,
        73.912, 77.576, 81.63, 86.213, 91.437, 97.383, 104.37, 112.932, 123.807, 138.038,
        155.086, 178.293,
    ])
    GdPdG = np.array([
        0.0, 191.86, 300.56, 327.67, 270.32, 145.24, 190.11, 45.8, 44.47, 53.84, 57.72, 63.84,
        76.07, 101.94, 137.37, 171.96, 206.08, 242.52, 278.7, 315.26, 352.29, 390.19, 429.57,
        468.67, 504.17, 538.67, 571.71, 599.93, 623.84, 645.45, 669.86, 692.34, 709.21, 719.42,
        728.31, 742.72, 754.35, 761.03, 762.83, 760.0, 759.05, 743.69, 719.97, 704.31, 690.76,
        680.38, 648.77, 606.29, 571.37, 531.98, 495.34, 457.77, 411.07, 358.63, 306.68, 269.15,
        242.24, 230.48,
    ])
    assert len(G) == len(GdPdG) == 58

    clean_slope, clean_idx = interpret.suggest_closure_tangent(G, GdPdG)
    assert clean_slope == pytest.approx(18.2, abs=0.5)
    assert G[clean_idx] < 45.0

    for sig in (0.005, 0.01, 0.02):
        rng = np.random.default_rng(0)
        closure_Gs = []
        for _ in range(200):
            noisy = GdPdG * (1.0 + sig * rng.standard_normal(len(GdPdG)))
            _, idx = interpret.suggest_closure_tangent(G, noisy)
            closure_Gs.append(G[idx])
        frac_past_50 = float(np.mean(np.array(closure_Gs) > 50.0))
        assert frac_past_50 <= 0.05, f"sig={sig}: {frac_past_50:.3f} of seeds landed past G=50"


# --------------------------------------------------------------------------------------------------
# seed_tangent stays draggable: a committed pick survives a re-seed
# --------------------------------------------------------------------------------------------------
def test_seed_tangent_dragged_closure_survives_reseed():
    td = make_testdata()
    st = injection_state(td)
    picks.seed_injection(st, td)
    res = compute_all(st, td)
    picks.seed_isip(st, td, res)
    res = compute_all(st, td)
    picks.seed_gfunction(st, res)
    picks.seed_tangent(st, res)
    assert st.closure_slope is not None and st.closure_G is not None

    dg = res.diagnostics
    dragged_G = float(dg.G[len(dg.G) // 2])
    picks.commit_closure_point(st, dragged_G)
    assert st.closure_G == dragged_G

    picks.seed_tangent(st, res)
    assert st.closure_G == dragged_G
