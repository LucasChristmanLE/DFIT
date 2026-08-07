"""No-rate fallback: pressure-based overview window seeding + te = pump duration.

Covers docs/superpowers/specs/2026-08-07-no-rate-fallback-design.md: datasets with no rate
channel (or a dead one) must still get draggable start/shut-in vlines (seeded from the
pressure shape) and a usable te, so the downstream steps work without rate.
"""

from __future__ import annotations

import numpy as np
import pytest

from dfit_tool import interpret
from tests.helpers import make_testdata, PRESSURE_COL, START_IDX, SHUTIN_IDX
from dfit_tool import picks
from dfit_tool.model import PickState
from tests.helpers import RATE_COL
from dfit_tool.model import compute_all
from tests.helpers import overview_state
from matplotlib.figure import Figure

from dfit_tool import plots


# --------------------------------------------------------------------------------------------------
# interpret.suggest_injection_window_pressure
# --------------------------------------------------------------------------------------------------
def test_pressure_window_lands_on_the_dfit_shape():
    td = make_testdata()
    p = td.column(PRESSURE_COL)
    start, shutin = interpret.suggest_injection_window_pressure(p)
    # shut-in at the pressure max (the linspace peak sits at SHUTIN_IDX - 1)
    assert SHUTIN_IDX - 2 <= shutin <= SHUTIN_IDX + 1
    # start at the rise onset: baseline 2000, rise 3000, 10% threshold crosses ~20 samples in
    assert START_IDX <= start <= START_IDX + 40
    assert start < shutin


def test_pressure_window_skips_an_early_breakdown_pulse():
    # Baseline 2000 with a pulse to 4000 at [50, 60), then the main rise at [100, 300) to
    # 5000 and a decline. The last-upcross rule must put start on the main rise, not the pulse.
    n = 600
    p = np.full(n, 2000.0)
    p[50:60] = 4000.0
    p[100:300] = 2000.0 + 3000.0 * np.linspace(0.0, 1.0, 200)
    p[300:] = np.linspace(5000.0, 3500.0, n - 300)
    start, shutin = interpret.suggest_injection_window_pressure(p)
    assert start >= 100
    assert 295 <= shutin <= 301
    assert start < shutin


def test_pressure_window_dropout_before_the_max_does_not_collapse_the_window():
    # A NaN read as "below threshold" would fake an upcross right before the peak and
    # collapse the window to one sample; finite-only handling must keep the real onset.
    td = make_testdata()
    p = td.column(PRESSURE_COL).copy()
    clean_start, clean_shutin = interpret.suggest_injection_window_pressure(p)
    p[SHUTIN_IDX - 2] = np.nan  # one gauge dropout right before the peak
    start, shutin = interpret.suggest_injection_window_pressure(p)
    assert start == clean_start
    assert abs(shutin - clean_shutin) <= 2


def test_pressure_window_dropout_block_mid_rise_does_not_relocate_start():
    td = make_testdata()
    p = td.column(PRESSURE_COL).copy()
    clean_start, _ = interpret.suggest_injection_window_pressure(p)
    p[200:203] = np.nan  # dropout block mid-injection, well above the 10% threshold
    start, _ = interpret.suggest_injection_window_pressure(p)
    assert start == clean_start


def test_pressure_window_flat_record_falls_back_to_positional():
    p = np.full(50, 1000.0)
    assert interpret.suggest_injection_window_pressure(p) == (0, 12)  # 2% / 25% of n-1


def test_pressure_window_declining_record_falls_back_to_positional():
    p = np.linspace(5000.0, 1000.0, 50)  # max at index 0
    assert interpret.suggest_injection_window_pressure(p) == (0, 12)  # 2% / 25% of n-1


def test_pressure_window_too_few_finite_samples_raises():
    with pytest.raises(ValueError):
        interpret.suggest_injection_window_pressure(np.array([np.nan, np.nan, 5.0]))


# --------------------------------------------------------------------------------------------------
# picks.seed_overview fallback
# --------------------------------------------------------------------------------------------------
def test_seed_overview_no_rate_col_seeds_from_pressure():
    td = make_testdata()
    st = PickState(pressure_col=PRESSURE_COL)  # no rate_col
    picks.seed_overview(st, td)
    assert st.start_idx is not None and st.shutin_idx is not None
    assert st.start_idx < st.shutin_idx
    exp = interpret.suggest_injection_window_pressure(td.column(PRESSURE_COL))
    assert (st.start_idx, st.shutin_idx) == exp


def test_seed_overview_dead_rate_channel_falls_back_to_pressure():
    td = make_testdata()
    td.df[RATE_COL] = 0.0  # rate never exceeds threshold -> suggest_injection_window raises
    st = PickState(pressure_col=PRESSURE_COL, rate_col=RATE_COL)
    picks.seed_overview(st, td)
    assert st.start_idx is not None and st.shutin_idx is not None
    assert st.start_idx < st.shutin_idx
    exp = interpret.suggest_injection_window_pressure(td.column(PRESSURE_COL))
    assert (st.start_idx, st.shutin_idx) == exp


def test_seed_overview_fallback_never_clobbers_existing_picks():
    td = make_testdata()
    st = PickState(pressure_col=PRESSURE_COL, start_idx=5, shutin_idx=7)
    picks.seed_overview(st, td)
    assert (st.start_idx, st.shutin_idx) == (5, 7)


def test_seed_overview_no_pressure_col_is_a_noop():
    td = make_testdata()
    st = PickState()  # neither rate nor pressure mapped
    picks.seed_overview(st, td)
    assert st.start_idx is None and st.shutin_idx is None


# --------------------------------------------------------------------------------------------------
# compute_all: te falls back to pump duration without rate
# --------------------------------------------------------------------------------------------------
def test_compute_all_no_rate_sets_t_shutin_and_fallback_te():
    td = make_testdata()
    st = PickState(pressure_col=PRESSURE_COL, start_idx=START_IDX, shutin_idx=SHUTIN_IDX)
    res = compute_all(st, td)
    assert res.t_shutin_s == pytest.approx(float(td.t_s[SHUTIN_IDX]))
    assert res.te_s == pytest.approx(float(td.t_s[SHUTIN_IDX] - td.t_s[START_IDX]))
    assert res.qmax_bpm is None and res.vinj is None
    assert any("pump duration" in w for w in res.warnings)
    # te works end to end: the resample + diagnostics pipeline runs
    assert res.resampled is not None
    assert res.diagnostics is not None


def test_compute_all_with_rate_is_unchanged():
    td = make_testdata()
    st = overview_state(td)
    res = compute_all(st, td)
    # effective te = Vinj/qmax, not the wall-clock duration
    assert res.te_s == pytest.approx(interpret.effective_te_seconds(res.vinj, res.qmax_bpm))
    assert res.vinj is not None and res.qmax_bpm is not None
    assert not any("pump duration" in w for w in res.warnings)


def test_compute_all_dead_rate_channel_falls_back_to_pump_duration():
    td = make_testdata()
    td.df[RATE_COL] = 0.0  # rate present but never pumps: qmax = 0 -> no effective te
    st = overview_state(td)
    res = compute_all(st, td)
    assert res.te_s == pytest.approx(float(td.t_s[SHUTIN_IDX] - td.t_s[START_IDX]))
    assert any("pump duration" in w for w in res.warnings)


# --------------------------------------------------------------------------------------------------
# plots.render_overview: title must not assume te implies Vinj/qmax
# --------------------------------------------------------------------------------------------------
def test_render_overview_title_with_fallback_te_and_no_rate():
    td = make_testdata()
    st = PickState(pressure_col=PRESSURE_COL, start_idx=START_IDX, shutin_idx=SHUTIN_IDX)
    res = compute_all(st, td)
    assert res.te_s is not None and res.vinj is None  # the crash precondition
    ax = Figure().add_subplot(111)
    plots.render_overview(ax, td, st, res)  # must not raise
    title = ax.get_title()
    assert "te=" in title
    assert "Vinj" not in title and "qmax" not in title


def test_render_overview_no_rate_still_draws_the_draggable_vlines():
    td = make_testdata()
    st = PickState(pressure_col=PRESSURE_COL)  # no rate_col
    picks.seed_overview(st, td)
    res = compute_all(st, td)
    ax = Figure().add_subplot(111)
    plots.render_overview(ax, td, st, res)
    gids = {ln.get_gid() for ln in ax.lines}
    assert {"start", "shutin"} <= gids
