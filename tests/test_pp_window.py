"""Pore-pressure (after-closure) window bugfixes:

1. ``handle_pp_span`` must convert the dragged span -- which arrives in the *plotted* transform
   domain (x = t**expo) -- back to shut-in seconds before storing it in ``state.pp_window``
   (contractually seconds; see ``model.py`` ``compute_all`` / ``plots.render_porepressure``).
2. ``seed_pp`` should reuse ``loglog_window`` (same units, same late-time physical window) when
   available, falling back to the last decade of shut-in time -- widened so it always covers at
   least the last two samples even when the late-time data is sparse.
3. ``render_porepressure`` shades the selected window (converted back to plot-transform
   coordinates, including the open-ended/inf-``hi`` case) and flags an unphysical pore-pressure
   (>= the observed minimum in the fitted window) in the title. Exercised directly here against a
   real ``Figure``/``Axes`` (Agg backend, forced globally by ``tests/conftest.py``), plus a
   ``model.compute_all`` round-trip with an inf-``hi`` window from a simulated lo=0 drag.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from dfit_tool import interpret, picks, plots
from dfit_tool.model import DerivedResults, PickState, compute_all
from dfit_tool.resample import Diagnostics
from tests.helpers import make_testdata, overview_state


# --------------------------------------------------------------------------------------------------
# handle_pp_span: span (plotted transform domain) -> pp_window (shut-in seconds)
# --------------------------------------------------------------------------------------------------
def test_handle_pp_span_tm12_open_ended_late_time():
    """Dragging from the axis origin out to x=0.1 on the t^-1/2 axis means "everything from
    t=100s to infinity" in seconds."""
    state = PickState(pp_axis="tm12")
    picks.handle_pp_span(state, 0.0, 0.1)
    lo, hi = state.pp_window
    assert lo == pytest.approx(100.0)
    assert hi == float("inf")


def test_handle_pp_span_tm12_bounded_window():
    state = PickState(pp_axis="tm12")
    picks.handle_pp_span(state, 0.05, 0.1)
    lo, hi = state.pp_window
    assert lo == pytest.approx(100.0)
    assert hi == pytest.approx(400.0)


def test_handle_pp_span_tm1_open_ended_late_time():
    state = PickState(pp_axis="tm1")
    picks.handle_pp_span(state, 0.0, 0.1)
    lo, hi = state.pp_window
    assert lo == pytest.approx(10.0)
    assert hi == float("inf")


def test_handle_pp_span_degenerate_hi_zero_leaves_window_unchanged():
    state = PickState(pp_axis="tm12", pp_window=(5.0, 50.0))
    picks.handle_pp_span(state, 0.2, 0.0)
    assert state.pp_window == (5.0, 50.0)


def test_handle_pp_span_degenerate_hi_negative_leaves_window_unchanged():
    state = PickState(pp_axis="tm12", pp_window=(5.0, 50.0))
    picks.handle_pp_span(state, 0.2, -0.01)
    assert state.pp_window == (5.0, 50.0)


# --------------------------------------------------------------------------------------------------
# helper: minimal Diagnostics with only t/p populated (mirrors test_scenario_contact.py's
# ``_res_with`` helper); G-function fields are unused by the pore-pressure path.
# --------------------------------------------------------------------------------------------------
def _pp_diagnostics(t: np.ndarray, p: np.ndarray) -> Diagnostics:
    z = np.zeros_like(t)
    return Diagnostics(G=t, dPdG=z, GdPdG=z, d2PdG2=z, t=t, p=p, dp=z, tdpdt=z)


# --------------------------------------------------------------------------------------------------
# seed_pp: prefer the loglog_window (same units/late-time region); else last decade of shut-in
# time, widened to guarantee >=2 samples even on sparse late-time data.
# --------------------------------------------------------------------------------------------------
def test_seed_pp_copies_loglog_window_when_present():
    td = make_testdata()
    st = overview_state(td)
    res = compute_all(st, td)
    st.loglog_window = (11.0, 22.0)
    picks.seed_pp(st, res)
    assert st.pp_window == (11.0, 22.0)


def test_seed_pp_falls_back_to_last_decade_of_shutin_time():
    td = make_testdata()
    st = overview_state(td)
    res = compute_all(st, td)
    assert st.loglog_window is None
    picks.seed_pp(st, res)
    dg = res.diagnostics
    expected = (float(dg.t[-1]) / 10.0, float(dg.t[-1]))
    assert st.pp_window == pytest.approx(expected)


def test_seed_pp_widens_last_decade_to_guarantee_two_samples_on_sparse_data():
    """t[-1]/10 .. t[-1] holds only the very last sample here (1000/10=100, and the only other
    sample below 1000 is at t=7). Without widening, the seeded window would fit on a single
    point; the fix pulls lo down to the second-to-last sample so the mask always has >=2."""
    t = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 1000.0])
    p = np.linspace(9000.0, 8000.0, t.size)
    res = DerivedResults(diagnostics=_pp_diagnostics(t, p))
    st = PickState()
    picks.seed_pp(st, res)
    assert st.pp_window == pytest.approx((7.0, 1000.0))
    lo, hi = st.pp_window
    mask = (t >= lo) & (t <= hi)
    assert mask.sum() >= 2


# --------------------------------------------------------------------------------------------------
# round-trip on a synthetic monotone falloff (hand-built Diagnostics, masking replicated from
# model.compute_all): dragged late-time window recovers pore pressure below the window minimum.
# --------------------------------------------------------------------------------------------------
def test_pp_span_drag_recovers_pore_pressure_below_window_minimum():
    p0, m = 8000.0, 20000.0
    t = np.logspace(1.0, 4.0, 200)  # 10s .. 10,000s
    p = p0 + m * t ** -0.5  # monotone falloff declining toward the asymptote p0

    dg = _pp_diagnostics(t, p)
    res = DerivedResults(diagnostics=dg)
    state = PickState(pp_axis="tm12")

    # Drag the late-time (small-x) region: t^-1/2 at t=1000s out to the axis origin (t -> inf).
    x_at_1000s = 1000.0 ** -0.5
    picks.handle_pp_span(state, 0.0, x_at_1000s)
    lo, hi = state.pp_window
    assert lo == pytest.approx(1000.0)
    assert hi == float("inf")

    # Replicate compute_all's pore-pressure masking (model.py ~lines 312-319).
    m_sel = (dg.t >= lo) & (dg.t <= hi) & (dg.t > 0)
    assert m_sel.sum() >= 2
    expo = -0.5 if state.pp_axis == "tm12" else -1.0
    x = dg.t[m_sel] ** expo
    pore_pressure = interpret.pore_pressure(x, dg.p[m_sel])

    assert pore_pressure < float(dg.p[m_sel].min())
    assert pore_pressure == pytest.approx(p0, rel=1e-6)


def test_compute_all_round_trip_with_open_ended_window_from_lo_zero_drag():
    """A real model.compute_all pass (not a hand-replicated mask) with pp_window's hi == inf,
    produced the way the UI actually produces it: a span dragged from the plot's x origin
    (lo=0) out to some finite x."""
    td = make_testdata()
    st = overview_state(td)
    res = compute_all(st, td)
    dg = res.diagnostics
    assert dg is not None

    picks.handle_pp_span(st, 0.0, 0.1)  # lo=0 -> pp_window hi becomes inf
    lo, hi = st.pp_window
    assert hi == float("inf")

    res2 = compute_all(st, td)
    assert res2.pore_pressure is not None
    assert math.isfinite(res2.pore_pressure)
    mask = (dg.t >= lo) & (dg.t > 0)
    assert mask.sum() >= 2
    # Extrapolating to infinite shut-in time should land below every observed pressure.
    assert res2.pore_pressure < float(np.min(dg.p))


# --------------------------------------------------------------------------------------------------
# render_porepressure: window shading (finite + open-ended) and the unphysical-fit title flag,
# against a real Figure/Axes (Agg backend forced globally by tests/conftest.py).
# --------------------------------------------------------------------------------------------------
def _render_pp(state: PickState, res: DerivedResults):
    fig = Figure()
    ax = fig.add_subplot(111)
    defaults = plots.render_porepressure(ax, None, state, res)
    return ax, defaults


def _only_axvspan(ax) -> Rectangle:
    rects = [p for p in ax.patches if isinstance(p, Rectangle)]
    assert len(rects) == 1, f"expected exactly one axvspan Rectangle, got {len(rects)}"
    return rects[0]


def test_render_porepressure_shades_finite_window_and_plain_title():
    p0, m = 8000.0, 20000.0
    t = np.logspace(1.0, 4.0, 200)
    p = p0 + m * t ** -0.5
    dg = _pp_diagnostics(t, p)

    lo, hi = 1000.0, 5000.0
    expo = -0.5
    mask = (dg.t >= lo) & (dg.t <= hi) & (dg.t > 0)
    pore_pressure = interpret.pore_pressure(dg.t[mask] ** expo, dg.p[mask])

    state = PickState(pp_axis="tm12", pp_window=(lo, hi))
    res = DerivedResults(diagnostics=dg, pore_pressure=pore_pressure)

    ax, _ = _render_pp(state, res)
    rect = _only_axvspan(ax)
    x_lo = hi ** expo
    x_hi = lo ** expo
    assert rect.get_x() == pytest.approx(x_lo)
    assert rect.get_x() + rect.get_width() == pytest.approx(x_hi)
    assert ax.get_title() == f"Pore pressure = {pore_pressure:.0f} psi"
    assert ">= observed" not in ax.get_title()


def test_render_porepressure_open_ended_window_shades_from_zero():
    p0, m = 8000.0, 20000.0
    t = np.logspace(1.0, 4.0, 200)
    p = p0 + m * t ** -0.5
    dg = _pp_diagnostics(t, p)

    lo, hi = 1000.0, float("inf")
    expo = -0.5
    mask = (dg.t >= lo) & (dg.t <= hi) & (dg.t > 0)
    pore_pressure = interpret.pore_pressure(dg.t[mask] ** expo, dg.p[mask])

    state = PickState(pp_axis="tm12", pp_window=(lo, hi))
    res = DerivedResults(diagnostics=dg, pore_pressure=pore_pressure)

    ax, _ = _render_pp(state, res)
    rect = _only_axvspan(ax)
    assert rect.get_x() == pytest.approx(0.0)
    assert rect.get_x() + rect.get_width() == pytest.approx(lo ** expo)


def test_render_porepressure_flags_title_when_fit_is_unphysical():
    """A window where the fitted BHP is perfectly flat: the extrapolated (x->0) pore pressure
    equals the window's observed minimum exactly, tripping the ">= observed" sanity flag."""
    t = np.array([10.0, 50.0, 100.0, 500.0, 1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0])
    p = np.array([9500.0, 9400.0, 9300.0, 9200.0, 9100.0, 9000.0, 9000.0, 9000.0, 8500.0, 8400.0])
    dg = _pp_diagnostics(t, p)

    lo, hi = 2000.0, 4000.0  # covers the three flat samples (t=2000,3000,4000; p=9000 each)
    expo = -0.5
    mask = (dg.t >= lo) & (dg.t <= hi) & (dg.t > 0)
    assert mask.sum() == 3
    pore_pressure = interpret.pore_pressure(dg.t[mask] ** expo, dg.p[mask])
    assert pore_pressure == pytest.approx(9000.0)  # flat fit -> intercept == the flat value

    state = PickState(pp_axis="tm12", pp_window=(lo, hi))
    res = DerivedResults(diagnostics=dg, pore_pressure=pore_pressure)

    ax, _ = _render_pp(state, res)
    assert ">= observed" in ax.get_title()
