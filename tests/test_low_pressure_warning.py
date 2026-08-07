"""Low-surface-pressure warning (CLAUDE.md TODO #4): a surface-pressure channel whose post-
shut-in reading falls below interpret.MIN_SURFACE_PRESSURE_PSI makes BHP unreliable there
(whether or not a hydrostatic conversion is even happening). Only fires when the mapped channel
is surface pressure (state.pressure_is_bhp is False); the mask's upper bound is the actually-
kept resampled window (trim, if any, already applied there -- and it also respects the
resampler's own rise guard), so a low reading the rise guard already excluded from every
computed value never triggers a warning that trimming would change nothing."""

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
    """A late reading below 100 psi that the resampler's own rise guard (a >30 psi rise above
    the running minimum) already excludes from resampled/resampled_full must not trigger the
    warning -- trimming wouldn't change any computed value, since that data was never used."""
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
    # Rise: a +100 psi jump (> the 30 psi rise_tol) trips the tail guard right here.
    rise_idx = shutin_idx + phase1_len
    pressure[rise_idx] = pressure[rise_idx - 1] + 100.0
    # Phase 2: falls well below 100 psi -- must never count; the rise guard already stopped
    # resampling at rise_idx, so nothing from here on feeds resampled/resampled_full.
    phase2_len = n - rise_idx - 1
    pressure[rise_idx + 1:] = np.linspace(pressure[rise_idx], 5.0, phase2_len)

    df = pd.DataFrame({PRESSURE_COL: pressure, "RATE": rate})
    td = IoTestData(path="<synthetic>", df=df, datetime_col="DATETIME", t_s=t_s,
                    columns=list(df.columns))
    st = PickState(pressure_col=PRESSURE_COL, rate_col="RATE", start_idx=start_idx,
                   shutin_idx=shutin_idx)
    res = compute_all(st, td)

    # Sanity: confirm the rise guard actually fired, so this test exercises what it claims to.
    assert res.resampled_full.guarded_at is not None
    assert not any(_WARNING_SNIPPET in w for w in res.warnings)
