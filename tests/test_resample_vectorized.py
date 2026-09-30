"""``resample.resample_pressure_increment``'s vectorized path must be bit-identical to the
per-sample reference loop it replaced (``resample._resample_loop_reference``, kept only for this
comparison and as the public function's own fallback for a handful of degenerate parameter
combinations) -- same ``dt``/``p`` arrays, same ``n_raw``/``guard_dt``/``guarded_at``, for every
input. This mirrors how ``tests/test_dropouts.py`` fuzzes ``resample._dropout_scan_candidates``
against ``resample._dropout_scan_loop``.
"""

from __future__ import annotations

import numpy as np
import pytest

from dfit_tool import resample
from dfit_tool.resample import (
    RISE_GUARD_PSI,
    RISE_GUARD_SUSTAIN_S,
    RISE_GUARD_SUSTAIN_SAMPLES,
    resample_pressure_increment,
)

# Parameter grid from the spec.
_STEPS = [1, 5, 30]
_SUSTAIN_SAMPLES = [1, 5]
_SUSTAIN_S = [0, 60]
_STOP_AT_GUARD = [True, False]
_RISE_TOL = [None, 10]


def _assert_same(rs_fast: resample.Resampled, rs_ref: resample.Resampled):
    np.testing.assert_array_equal(rs_fast.dt, rs_ref.dt)
    np.testing.assert_array_equal(rs_fast.p, rs_ref.p)
    assert rs_fast.n_raw == rs_ref.n_raw
    assert rs_fast.guard_dt == rs_ref.guard_dt
    assert rs_fast.guarded_at == rs_ref.guarded_at


def _check(dt, p, step, rise_tol, sustain_s, sustain_samples, stop_at_guard):
    dt = np.asarray(dt, dtype=float)
    p = np.asarray(p, dtype=float)
    rs_fast = resample_pressure_increment(dt, p, step=step, rise_tol=rise_tol,
                                          sustain_s=sustain_s, sustain_samples=sustain_samples,
                                          stop_at_guard=stop_at_guard)
    rs_ref = resample._resample_loop_reference(dt, p, step, rise_tol, sustain_s,
                                               sustain_samples, stop_at_guard)
    _assert_same(rs_fast, rs_ref)


# --------------------------------------------------------------------------------------------------
# Randomized series generator covering the shapes called out in the spec.
# --------------------------------------------------------------------------------------------------
def _random_resample_series(rng: np.random.Generator, length: int):
    """One synthetic post-shut-in (dt, p) pair -- a monotone decline with noise, optionally
    perturbed by a water-hammer spike, a sustained rise (which may or may not later resume a
    decline below the prior minimum, or stay permanently elevated/stuck), constant segments, and
    NaN runs (including a leading run or an all-NaN series), on either regular or irregular
    (including coarse >= 60 s) dt spacing."""
    level0 = float(rng.uniform(1000.0, 6000.0))
    level1 = float(rng.uniform(200.0, level0))
    p = np.linspace(level0, level1, length) + rng.normal(0.0, float(rng.uniform(0.5, 8.0)), length)

    # Constant segment somewhere in the middle.
    if length >= 6 and rng.random() < 0.4:
        seg_len = int(rng.integers(2, min(20, length - 2) + 1))
        seg_start = int(rng.integers(1, length - seg_len))
        p[seg_start:seg_start + seg_len] = p[seg_start - 1]

    # Water-hammer spike: a short 1-3 sample poke above tolerance that does not sustain.
    if length >= 5 and rng.random() < 0.4:
        spike_start = int(rng.integers(1, length - 3))
        spike_len = int(rng.integers(1, 3))
        p[spike_start:spike_start + spike_len] += float(rng.uniform(RISE_GUARD_PSI * 1.2,
                                                                    RISE_GUARD_PSI * 4.0))

    # Sustained rise: long enough to plausibly trip the guard under some grid point.
    if length >= 10 and rng.random() < 0.5:
        rise_start = int(rng.integers(length // 3, max(length // 3 + 1, length - 3)))
        rise_len = int(rng.integers(2, length - rise_start))
        delta = float(rng.uniform(RISE_GUARD_PSI * 1.5, RISE_GUARD_PSI * 6.0))
        mode = rng.random()
        if mode < 0.4:
            # Stays permanently elevated -- "never comes back down".
            p[rise_start:] = p[rise_start - 1] + delta
        elif mode < 0.8:
            # Held for a while, then resumes a further decline below the prior minimum.
            hold = min(rise_len, length - rise_start)
            p[rise_start:rise_start + hold] = p[rise_start - 1] + delta
            resume = rise_start + hold
            if resume < length:
                p[resume:] = np.linspace(p[resume - 1], level1 - abs(delta) - 50.0,
                                         length - resume)
        else:
            # A brief excursion that dips back below tolerance on its own (no sustained fire).
            hold = min(int(rng.integers(1, 4)), length - rise_start)
            p[rise_start:rise_start + hold] = p[rise_start - 1] + delta

    # Bidirectional oscillation: a run of samples that swing back and forth by more than the
    # largest tested step (30), well under rise_tol/sustain so it never trips the guard on its
    # own -- exercises the ordinary +-step keep rule in both directions, independent of any
    # guard/run bookkeeping.
    if length >= 8 and rng.random() < 0.4:
        osc_start = int(rng.integers(0, length - 6))
        osc_len = int(rng.integers(4, min(20, length - osc_start) + 1))
        swing = float(rng.uniform(RISE_GUARD_PSI * 1.5, RISE_GUARD_PSI * 3.0))
        signs = np.where(np.arange(osc_len) % 2 == 0, 1.0, -1.0)
        p[osc_start:osc_start + osc_len] = p[osc_start - 1 if osc_start else 0] + signs * swing

    # NaN runs: leading, mid-record, or (occasionally) the whole series.
    r = rng.random()
    if r < 0.1:
        p[:] = np.nan
    elif r < 0.3 and length >= 3:
        lead = int(rng.integers(1, min(length - 1, 5) + 1))
        p[:lead] = np.nan
    if length >= 4 and rng.random() < 0.3:
        nan_len = int(rng.integers(1, min(6, length // 2) + 1))
        nan_start = int(rng.integers(0, length - nan_len))
        p[nan_start:nan_start + nan_len] = np.nan

    # +-inf samples, scattered singly -- non-finite exactly like NaN, but `abs(inf - x)` is
    # `inf` (not NaN), which is >= any finite step: a search against raw (unmasked) p would
    # therefore wrongly keep one, unlike a NaN. Placed after the NaN runs above so an inf can
    # land inside, adjacent to, or clear of one.
    if length >= 3 and rng.random() < 0.3:
        n_inf = int(rng.integers(1, min(4, length) + 1))
        inf_positions = rng.choice(length, size=n_inf, replace=False)
        signs = rng.choice([np.inf, -np.inf], size=n_inf)
        p[inf_positions] = signs

    if rng.random() < 0.5:
        dt = np.cumsum(rng.uniform(0.5, 5.0, size=length))
    elif rng.random() < 0.5:
        dt = np.cumsum(rng.uniform(60.0, 300.0, size=length))  # coarse spacing
    else:
        dt = np.arange(length, dtype=float)
    return dt, p


def test_vectorized_fuzz_matches_reference_loop():
    """A few thousand randomized (series, parameter) checks -- monotone declines with noise,
    water-hammer spikes, sustained rises (some that resume a decline, some permanently elevated),
    constant segments, NaN runs (including leading and all-NaN), lengths 0-300, regular/irregular/
    coarse dt spacing -- crossed against the full parameter grid (step, sustain_samples,
    sustain_s, stop_at_guard, rise_tol)."""
    rng = np.random.default_rng(20260927)
    n_series = 60
    checked = 0
    for _ in range(n_series):
        length = int(rng.integers(0, 301))
        dt, p = _random_resample_series(rng, length)
        for step in _STEPS:
            for sustain_samples in _SUSTAIN_SAMPLES:
                for sustain_s in _SUSTAIN_S:
                    for stop_at_guard in _STOP_AT_GUARD:
                        for rise_tol in _RISE_TOL:
                            _check(dt, p, step, rise_tol, sustain_s, sustain_samples,
                                  stop_at_guard)
                            checked += 1
    assert checked >= 2000


# --------------------------------------------------------------------------------------------------
# Very short arrays, explicitly.
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("length", [0, 1, 2])
def test_very_short_arrays(length):
    dt = np.arange(length, dtype=float)
    p = np.linspace(5000.0, 4000.0, length) if length else np.array([])
    for step in _STEPS:
        for stop_at_guard in _STOP_AT_GUARD:
            _check(dt, p, step, None, RISE_GUARD_SUSTAIN_S, RISE_GUARD_SUSTAIN_SAMPLES,
                  stop_at_guard)


def test_all_nan_array():
    dt = np.arange(20, dtype=float)
    p = np.full(20, np.nan)
    _check(dt, p, 30.0, None, RISE_GUARD_SUSTAIN_S, RISE_GUARD_SUSTAIN_SAMPLES, True)


def test_single_finite_sample_with_trailing_nans():
    dt = np.arange(10, dtype=float)
    p = np.full(10, np.nan)
    p[3] = 4000.0
    _check(dt, p, 30.0, None, RISE_GUARD_SUSTAIN_S, RISE_GUARD_SUSTAIN_SAMPLES, True)


def test_empty_array_matches_public_function_directly():
    """The public function's own early return (no delegation) for an all-empty/no-finite input --
    pinned directly against Resampled's own defaults, not just against the reference loop."""
    rs = resample_pressure_increment(np.array([]), np.array([]))
    assert rs.n_raw == 0
    assert rs.guard_dt is None
    assert rs.guarded_at is None
    np.testing.assert_array_equal(rs.dt, np.array([]))
    np.testing.assert_array_equal(rs.p, np.array([]))


# --------------------------------------------------------------------------------------------------
# Explicit named shapes for each category called out in the spec, beyond what the fuzz covers.
# --------------------------------------------------------------------------------------------------
def test_monotone_decline_with_noise():
    rng = np.random.default_rng(1)
    n = 400
    dt = np.arange(n, dtype=float)
    p = np.linspace(5000.0, 500.0, n) + rng.normal(0.0, 3.0, n)
    for step in _STEPS:
        _check(dt, p, step, None, RISE_GUARD_SUSTAIN_S, RISE_GUARD_SUSTAIN_SAMPLES, True)


def test_water_hammer_spike_shape():
    n = 300
    dt = np.arange(n, dtype=float) * 5.0
    p = np.linspace(4000.0, 1000.0, n)
    p[100:102] += 100.0
    for stop_at_guard in _STOP_AT_GUARD:
        _check(dt, p, 30.0, None, RISE_GUARD_SUSTAIN_S, RISE_GUARD_SUSTAIN_SAMPLES,
              stop_at_guard)


def test_sustained_rise_that_resumes_decline():
    n = 500
    dt = np.arange(n, dtype=float)
    p = np.linspace(4000.0, 2000.0, n)
    rise_start, rise_len = 150, 80
    p[rise_start:rise_start + rise_len] = p[rise_start - 1] + 50.0
    resume = rise_start + rise_len
    p[resume:] = np.linspace(p[resume - 1], 500.0, n - resume)
    for stop_at_guard in _STOP_AT_GUARD:
        _check(dt, p, 30.0, None, RISE_GUARD_SUSTAIN_S, RISE_GUARD_SUSTAIN_SAMPLES,
              stop_at_guard)


def test_stuck_elevated_tail_never_comes_back():
    n = 600
    dt = np.arange(n, dtype=float)
    p = np.linspace(4000.0, 2000.0, n)
    p[300:] = p[299] + np.linspace(50.0, 300.0, n - 300)
    for stop_at_guard in _STOP_AT_GUARD:
        _check(dt, p, 30.0, None, RISE_GUARD_SUSTAIN_S, RISE_GUARD_SUSTAIN_SAMPLES,
              stop_at_guard)


def test_infinite_samples_never_kept():
    """Regression: the fast path's kept walk searched raw `p`, where `abs(inf - x)` is `inf`
    (not NaN) and so satisfies `>= step` -- wrongly keeping a +-inf sample that the reference
    loop's plain `np.isfinite` guard skips outright, same as a NaN. Exact example from the
    review that caught it: dt=0..5, p=[1000, 960, inf, 930, -inf, 900] -- both +inf and -inf
    must be skipped, leaving only the four finite samples, every one a genuine >= 30 psi move
    off the last kept point."""
    dt = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
    p = np.array([1000.0, 960.0, np.inf, 930.0, -np.inf, 900.0])

    rs = resample_pressure_increment(dt, p, step=30.0)

    np.testing.assert_array_equal(rs.dt, np.array([0.0, 1.0, 3.0, 5.0]))
    np.testing.assert_array_equal(rs.p, np.array([1000.0, 960.0, 930.0, 900.0]))
    _check(dt, p, 30.0, None, RISE_GUARD_SUSTAIN_S, RISE_GUARD_SUSTAIN_SAMPLES, True)


def test_nan_run_mid_record():
    n = 400
    dt = np.arange(n, dtype=float)
    p = np.linspace(5000.0, 1000.0, n)
    p[150:158] = np.nan
    _check(dt, p, 30.0, None, RISE_GUARD_SUSTAIN_S, RISE_GUARD_SUSTAIN_SAMPLES, True)


def test_leading_nan_run():
    n = 300
    dt = np.arange(n, dtype=float)
    p = np.linspace(5000.0, 1000.0, n)
    p[:10] = np.nan
    _check(dt, p, 30.0, None, RISE_GUARD_SUSTAIN_S, RISE_GUARD_SUSTAIN_SAMPLES, True)


def test_constant_segment():
    n = 200
    dt = np.arange(n, dtype=float)
    p = np.linspace(4000.0, 2000.0, n)
    p[80:120] = p[79]
    _check(dt, p, 30.0, None, RISE_GUARD_SUSTAIN_S, RISE_GUARD_SUSTAIN_SAMPLES, True)


def test_bidirectional_oscillation():
    """A run of samples that swing up and down by more than ``step``, well under rise_tol/
    sustain so it never fires the guard -- pins fast-vs-reference agreement on the ordinary
    +-step keep rule in both directions, with no run bookkeeping involved at all."""
    n = 200
    dt = np.arange(n, dtype=float)
    p = np.linspace(4000.0, 3000.0, n)
    osc_start = 80
    swing = 50.0
    for k in range(20):
        p[osc_start + k] = p[osc_start - 1] + (swing if k % 2 == 0 else -swing)
    for stop_at_guard in _STOP_AT_GUARD:
        _check(dt, p, 30.0, None, RISE_GUARD_SUSTAIN_S, RISE_GUARD_SUSTAIN_SAMPLES,
              stop_at_guard)


def test_irregular_coarse_spacing():
    rng = np.random.default_rng(2)
    n = 150
    dt = np.cumsum(rng.uniform(60.0, 300.0, size=n))  # >= 60 s spacing throughout
    p = np.linspace(4500.0, 1200.0, n) + rng.normal(0.0, 5.0, n)
    rise_start = 80
    p[rise_start:] = p[rise_start - 1] + 100.0
    for sustain_samples in _SUSTAIN_SAMPLES:
        _check(dt, p, 30.0, None, RISE_GUARD_SUSTAIN_S, sustain_samples, True)


# --------------------------------------------------------------------------------------------------
# Fallback path: a non-finite/non-positive step or a non-finite/negative rise_tol must delegate
# to the reference loop outright, matching it exactly (trivially, since it IS the reference loop
# -- but pinned so the dispatch itself is exercised and can't silently regress).
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("step, rise_tol", [
    (0.0, None), (-5.0, None), (float("nan"), None), (float("inf"), None),
    (30.0, -1.0), (30.0, float("nan")),
])
def test_degenerate_params_delegate_to_reference(step, rise_tol):
    n = 200
    dt = np.arange(n, dtype=float)
    p = np.linspace(5000.0, 1000.0, n)
    p[100:] = p[99] + 50.0  # a sustained rise, so the reference loop has real work to do too

    rs_public = resample_pressure_increment(dt, p, step=step, rise_tol=rise_tol)
    rs_ref = resample._resample_loop_reference(dt, p, step, rise_tol, RISE_GUARD_SUSTAIN_S,
                                               RISE_GUARD_SUSTAIN_SAMPLES, True)
    _assert_same(rs_public, rs_ref)
