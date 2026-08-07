"""Low-surface-pressure warning (CLAUDE.md TODO #4): a surface-pressure channel whose post-
shut-in reading falls below interpret.MIN_SURFACE_PRESSURE_PSI makes BHP unreliable there
(whether or not a hydrostatic conversion is even happening). Only fires when the mapped channel
is surface pressure (state.pressure_is_bhp is False); the mask's upper bound is where the
resampler actually stopped consuming raw samples (resampled_full.guard_dt, when the rise guard
fired) intersected with the trim, if any -- NOT the last *kept* resampled point, which can sit
up to one resample_step above the true minimum and so miss a crash sitting just past it."""

from __future__ import annotations

import numpy as np
import pandas as pd

from dfit_tool import picks
from dfit_tool.io_load import TestData as IoTestData
from dfit_tool.model import PickState, compute_all
from tests.helpers import PRESSURE_COL, make_testdata, overview_state, pre_crash_trim_dt

_WARNING_SNIPPET = "Surface pressure fell below 100 psi"


def _seeded_with_crash(zero_crash_at: float = 0.5):
    td = make_testdata(n=1200, zero_crash_at=zero_crash_at)
    st = overview_state(td)
    picks.seed_overview(st, td)
    res = compute_all(st, td)
    picks.seed_isip(st, td, res)
    res = compute_all(st, td)
    return td, st, res


def test_fires_on_untrimmed_crash():
    td, st, res = _seeded_with_crash()
    assert any(_WARNING_SNIPPET in w for w in res.warnings)


def test_clears_when_trim_excludes_the_crash():
    td, st, res = _seeded_with_crash()
    last_normal_dt = pre_crash_trim_dt(res)
    st.tail_trim_dt = last_normal_dt
    res2 = compute_all(st, td)
    assert not any(_WARNING_SNIPPET in w for w in res2.warnings)


def test_absent_when_pressure_is_bhp():
    """Gated on state.pressure_is_bhp (the mapped channel), not res.pressure_is_bhp -- that
    flips True after hydrostatic conversion too, which would hide the very condition that makes
    the conversion unreliable."""
    td, st, res = _seeded_with_crash()
    st.pressure_is_bhp = True
    res2 = compute_all(st, td)
    assert not any(_WARNING_SNIPPET in w for w in res2.warnings)


def test_absent_when_pressure_stays_high():
    td = make_testdata()
    st = overview_state(td)
    picks.seed_overview(st, td)
    res = compute_all(st, td)
    picks.seed_isip(st, td, res)
    res = compute_all(st, td)
    assert not any(_WARNING_SNIPPET in w for w in res.warnings)


def test_absent_when_shutin_unpicked():
    td = make_testdata(n=1200, zero_crash_at=0.5)
    st = overview_state(td)
    st.shutin_idx = None
    res = compute_all(st, td)
    assert not any(_WARNING_SNIPPET in w for w in res.warnings)


def test_post_shutin_only_mask_low_reading_before_start_does_not_fire():
    """A low raw-pressure reading well before shut-in must not trip the warning -- the mask is
    dt_ws >= 0 (post-shut-in only)."""
    td = make_testdata()
    td.df.loc[10, PRESSURE_COL] = 5.0  # before injection even starts
    st = overview_state(td)
    picks.seed_overview(st, td)
    res = compute_all(st, td)
    picks.seed_isip(st, td, res)
    res = compute_all(st, td)
    assert not any(_WARNING_SNIPPET in w for w in res.warnings)


def test_absent_when_rise_guard_excludes_the_low_reading():
    """A late reading below 100 psi that the resampler's own rise guard (a sustained >30 psi
    rise above the running minimum, held >= 60 s) already excludes from resampled/resampled_full
    must not trigger the warning -- trimming wouldn't change any computed value, since that data
    was never used."""
    n = 400
    t_s = np.arange(n, dtype=float)
    start_idx, shutin_idx = 50, 100
    rate = np.zeros(n)
    rate[start_idx:shutin_idx] = 5.0

    pressure = np.full(n, 2000.0)
    pressure[start_idx:shutin_idx] = np.linspace(2000.0, 5000.0, shutin_idx - start_idx)
    post = n - shutin_idx
    # Phase 1: a normal smooth decline that feeds the resampler.
    phase1_len = post // 2
    pressure[shutin_idx:shutin_idx + phase1_len] = np.linspace(5000.0, 500.0, phase1_len)
    # Rise: a +100 psi jump (> the fixed 30 psi RISE_GUARD_PSI), held for >= 60 s so the
    # sustained-rise guard (resample.RISE_GUARD_SUSTAIN_S) still fires here, not just a single
    # over-tolerance sample.
    rise_idx = shutin_idx + phase1_len
    rise_len = 70
    pressure[rise_idx:rise_idx + rise_len] = pressure[rise_idx - 1] + 100.0
    # Phase 2: falls well below 100 psi -- must never count; the rise guard already stopped
    # resampling at rise_idx, so nothing from here on feeds resampled/resampled_full.
    phase2_start = rise_idx + rise_len
    phase2_len = n - phase2_start
    pressure[phase2_start:] = np.linspace(pressure[phase2_start - 1], 5.0, phase2_len)

    df = pd.DataFrame({PRESSURE_COL: pressure, "RATE": rate})
    td = IoTestData(path="<synthetic>", df=df, datetime_col="DATETIME", t_s=t_s,
                    columns=list(df.columns))
    st = PickState(pressure_col=PRESSURE_COL, rate_col="RATE", start_idx=start_idx,
                   shutin_idx=shutin_idx)
    res = compute_all(st, td)

    # Sanity: confirm the rise guard actually fired, so this test exercises what it claims to.
    assert res.resampled_full.guarded_at is not None
    assert not any(_WARNING_SNIPPET in w for w in res.warnings)


def test_fires_with_a_coarse_resample_step_that_would_have_missed_it():
    """Regression: bounding the scan by the last *kept* resampled point (rather than where the
    resampler actually stopped) let a coarse resample_step silently swallow the warning even
    with no rise guard and no trim -- e.g. resample_step=400 on the standard crashed dataset."""
    td, st, res = _seeded_with_crash()
    st.resample_step = 400.0
    res2 = compute_all(st, td)

    # Sanity: the crash is monotonic, so the rise guard never fires here -- this is testing the
    # coarse-step gap, not the rise-guard exclusion covered above.
    assert res2.resampled_full.guard_dt is None
    assert any(_WARNING_SNIPPET in w for w in res2.warnings)
