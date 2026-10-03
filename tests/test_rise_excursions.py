"""Returning upward gauge excursions (see resample.detect_rise_excursions).

A sustained upward excursion that comes back down to the pre-excursion level (Arkansas 1BH's
+50 psi bump at 975-1075 min, Akbary's +2277 psi excursion at 954-1046 min) otherwise trips the
tail rise guard in `resample.resample_pressure_increment` and ends the resample there, losing the
rest of the falloff. The detector masks those samples; `model.compute_all` folds the mask into
`res.dropout_mask` together with the manual mask/keep intervals in `PickState`.
"""

from __future__ import annotations

import numpy as np
import pytest

from dfit_tool import resample
from dfit_tool.resample import (
    RISE_EXCURSION_CRASH_FRAC,
    RISE_EXCURSION_MAX_FRAC,
    RISE_EXCURSION_MIN_START_S,
    RiseExcursion,
    _rise_excursion_loop,
    detect_rise_excursions,
)


def _falloff(n: int = 30000) -> tuple[np.ndarray, np.ndarray]:
    """1 Hz, strictly declining falloff from 5000 psi (every sample a new running min)."""
    dt = np.arange(n, dtype=float)
    p = 3000.0 + 2000.0 * np.exp(-dt / 20000.0)
    return dt, p


def test_constant():
    assert RISE_EXCURSION_MAX_FRAC == 0.25
    assert RISE_EXCURSION_MIN_START_S == 180.0
    assert RISE_EXCURSION_CRASH_FRAC == 0.5


def test_early_excursion_is_not_masked_and_stops_scanning():
    # Corpus: excursions in the first minutes are mostly pressure steps from a mis-picked
    # shut-in, not glitches, so they stay with the guard (and the analyst).
    # With the default max_frac a >= 60-s run can't be masked before 240 s anyway; the gate is
    # the explicit floor, so lift max_frac to see it act on its own.
    dt, p = _falloff()
    p[120:200] += 300.0      # starts at 2 min, 80 s long
    p[20000:20600] += 400.0  # fine on its own, but scanning stopped
    mask, events = detect_rise_excursions(dt, p, max_frac=5.0)
    assert not mask.any() and events == []
    mask, events = detect_rise_excursions(dt, p, max_frac=5.0, min_start_s=0.0)
    assert len(events) == 2


def test_crash_return_is_not_masked():
    # Rush 4CH / Raindance: the "return" is the gauge falling to ~0, not the pre-rise level.
    dt, p = _falloff()
    p[7200:8400] += 500.0
    p[8400:] = 2.0
    mask, events = detect_rise_excursions(dt, p)
    assert not mask.any() and events == []


def test_return_slightly_below_base_is_still_masked():
    dt, p = _falloff()
    p[7200:8400] += 500.0
    p[8400:] -= 200.0        # lands below base but far above half of it
    mask, events = detect_rise_excursions(dt, p)
    assert len(events) == 1


def test_many_events_runtime_is_linear():
    import time
    n = 1_000_000
    dt = np.arange(n, dtype=float) + 1000.0
    p = 3000.0 + 2000.0 * np.exp(-dt / 400000.0)
    for s in range(20000, n - 1000, 1900):   # ~515 returning 120-s +100 psi bumps
        p[s:s + 120] += 100.0
    t0 = time.perf_counter()
    mask, events = detect_rise_excursions(dt, p)
    elapsed = time.perf_counter() - t0
    assert len(events) > 400
    assert elapsed < 1.5, elapsed


def test_returning_bump_is_masked_with_correct_fields():
    dt, p = _falloff()
    clean = p.copy()
    p[7200:8400] += 500.0   # +500 psi, 20 min, starting at 120 min
    mask, events = detect_rise_excursions(dt, p)
    assert np.flatnonzero(mask).tolist() == list(range(7200, 8400))
    assert len(events) == 1
    e = events[0]
    assert isinstance(e, RiseExcursion)
    assert e.dt_start == 7200.0
    assert e.dt_end == 8400.0
    assert e.n_samples == 1200
    assert e.base == clean[7199]
    assert e.p_max == pytest.approx(np.max(p[7200:8400]))
    assert not mask[:7200].any() and not mask[8400:].any()


def test_non_returning_rise_is_not_masked():
    dt, p = _falloff()
    p[7200:] += 500.0
    mask, events = detect_rise_excursions(dt, p)
    assert not mask.any()
    assert events == []


def test_too_long_bump_masks_nothing_and_stops_scanning():
    dt, p = _falloff()
    p[600:1000] += 400.0       # 400 s > 0.25 * 600 s
    p[20000:20600] += 400.0    # would be fine on its own, but scanning stopped
    mask, events = detect_rise_excursions(dt, p)
    assert not mask.any()
    assert events == []


def test_short_spike_is_not_masked():
    dt, p = _falloff()
    p[7200:7230] += 500.0      # 30 s < 60 s sustain
    mask, events = detect_rise_excursions(dt, p)
    assert not mask.any()
    assert events == []


def test_water_hammer_ringing_is_not_masked():
    dt, p = _falloff()
    ring = dt < 120.0
    p[ring] += 200.0 * np.sin(2 * np.pi * dt[ring] / 10.0)
    mask, events = detect_rise_excursions(dt, p)
    assert not mask.any()
    assert events == []


def test_two_bumps_are_both_masked():
    dt, p = _falloff()
    p[7200:8400] += 500.0
    p[20000:21500] += 300.0
    mask, events = detect_rise_excursions(dt, p)
    assert len(events) == 2
    assert np.flatnonzero(mask).tolist() == list(range(7200, 8400)) + list(range(20000, 21500))
    assert [e.dt_start for e in events] == [7200.0, 20000.0]


def test_slow_ramp_start_is_included():
    """Samples that rise off the last new minimum but are still within tol of it do not start the
    run; they are still part of the excursion (lo = last new-min + 1)."""
    dt, p = _falloff()
    base = p[7199]
    ramp = np.arange(1, 1201, dtype=float)
    p[7200:8400] = base + np.minimum(ramp * 5.0, 400.0)   # 5 psi/s: first >30 psi at k=7
    mask, events = detect_rise_excursions(dt, p)
    assert len(events) == 1
    assert events[0].dt_start == 7200.0
    assert mask[7200] and not mask[7199]


def test_nan_inside_bump_is_masked_with_it():
    dt, p = _falloff()
    p[7200:8400] += 500.0
    p[7800] = np.nan   # one gap: each side must be sustained on its own, both are
    mask, events = detect_rise_excursions(dt, p)
    assert len(events) == 1
    assert mask[7800]
    assert np.flatnonzero(mask).tolist() == list(range(7200, 8400))


def test_frequent_nans_break_every_run():
    dt, p = _falloff()
    p[7200:8400] += 500.0
    p[7200:8400:30] = np.nan   # run never lasts 60 s
    mask, events = detect_rise_excursions(dt, p)
    assert not mask.any()
    assert events == []


def test_nan_gaps_elsewhere_do_not_matter():
    dt, p = _falloff()
    p[7200:8400] += 500.0
    p[100:140] = np.nan
    p[15000:15100] = np.nan
    mask, events = detect_rise_excursions(dt, p)
    assert len(events) == 1
    assert np.flatnonzero(mask).tolist() == list(range(7200, 8400))


def test_all_nan_and_empty():
    mask, events = detect_rise_excursions(np.arange(5.0), np.full(5, np.nan))
    assert not mask.any() and events == []
    mask, events = detect_rise_excursions(np.array([]), np.array([]))
    assert mask.shape == (0,) and events == []


# --------------------------------------------------------------------------------------------------
# Fuzz: vectorized path vs the reference loop
# --------------------------------------------------------------------------------------------------
def _random_series(rng: np.random.Generator, length: int):
    """Integer-valued (so ties are common) declines with plateaus, bumps, ramps, NaN/inf."""
    kind = rng.integers(0, 3)
    if kind == 0:
        base = np.cumsum(-rng.integers(0, 4, size=length)).astype(float)
    elif kind == 1:
        base = np.round(-np.linspace(0.0, float(rng.uniform(0, 300)), length))
    else:
        base = np.round(rng.normal(0.0, 5.0, size=length))
    p = 1000.0 + base
    for _ in range(int(rng.integers(0, 4))):
        if length < 4:
            break
        blen = int(rng.integers(1, max(2, length // 3)))
        s = int(rng.integers(0, length - blen))
        height = float(rng.choice([20.0, 31.0, 40.0, 100.0, 500.0]))
        if rng.random() < 0.4:
            p[s:s + blen] += np.minimum(np.arange(1, blen + 1) * float(rng.integers(1, 20)), height)
        else:
            p[s:s + blen] += height
    if rng.random() < 0.5:
        n_bad = int(rng.integers(0, max(1, length // 8) + 1))
        for bi in rng.choice(length, size=min(n_bad, length), replace=False):
            p[int(bi)] = float(rng.choice([np.nan, np.nan, np.inf, -np.inf]))
    step = float(rng.choice([1.0, 1.0, 5.0, 15.0, 30.0, 61.0]))
    if rng.random() < 0.3:
        dt = np.cumsum(rng.uniform(0.5, 90.0, size=length))
    else:
        dt = np.arange(length, dtype=float) * step
    if rng.random() < 0.3:
        dt = dt + 1.0   # dt[0] > 0 as in real data
    return dt, p


def _events_as_tuples(events):
    return [(e.dt_start, e.dt_end, e.n_samples, e.base, e.p_max) for e in events]


def test_fast_path_fuzz_matches_reference_loop():
    rng = np.random.default_rng(20261002)
    with_events = 0
    for _ in range(4000):
        length = int(rng.integers(1, 400))
        dt, p = _random_series(rng, length)
        kw = {}
        if rng.random() < 0.5:
            kw["sustain_s"] = float(rng.choice([0.0, 3.0, 10.0, 60.0]))
            kw["sustain_samples"] = int(rng.choice([1, 2, 5]))
        if rng.random() < 0.3:
            kw["tol"] = float(rng.choice([0.0, 10.0, 30.0]))
        if rng.random() < 0.5:
            kw["max_frac"] = float(rng.choice([0.05, 0.25, 1.0, 5.0]))
        if rng.random() < 0.7:
            kw["min_start_s"] = float(rng.choice([0.0, 0.0, 30.0, 180.0]))
        if rng.random() < 0.7:
            kw["crash_frac"] = float(rng.choice([0.0, 0.5, 0.99]))
        args = dict(tol=resample.RISE_GUARD_PSI, sustain_s=resample.RISE_GUARD_SUSTAIN_S,
                    sustain_samples=resample.RISE_GUARD_SUSTAIN_SAMPLES,
                    max_frac=RISE_EXCURSION_MAX_FRAC,
                    min_start_s=RISE_EXCURSION_MIN_START_S,
                    crash_frac=RISE_EXCURSION_CRASH_FRAC)
        args.update(kw)
        mf, ef = detect_rise_excursions(dt, p, **kw)
        ml, el = _rise_excursion_loop(dt, p, args["tol"], args["sustain_s"],
                                      args["sustain_samples"], args["max_frac"],
                                      args["min_start_s"], args["crash_frac"])
        np.testing.assert_array_equal(mf, ml)
        assert _events_as_tuples(ef) == _events_as_tuples(el)
        with_events += bool(el)
    assert with_events > 150   # the generator really exercises the event path
