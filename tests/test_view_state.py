"""Renderer view contract: render_* never sets Axes limits; it returns a ViewDefaults that the
caller (ui.py) applies. Also covers the pure view-resolution helper ui.py uses in refresh()."""

import types

import numpy as np
import pytest
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from dfit_tool import picks
from dfit_tool.model import DerivedResults, PickState, compute_all
from dfit_tool.resample import Diagnostics, Resampled
from dfit_tool import plots
from dfit_tool.plots import ViewDefaults
from dfit_tool.ui import DfitApp, ViewState, _resolve_view
from tests.helpers import PRESSURE_COL, make_testdata, overview_state


def _render(renderer, td, state, res):
    fig = Figure()
    ax = fig.add_subplot(111)
    defaults = renderer(ax, td, state, res)
    return fig, ax, defaults


def test_all_renderers_return_view_defaults():
    td = make_testdata()
    state = overview_state(td)
    res = compute_all(state, td)
    for name, renderer in plots.RENDERERS.items():
        _, _, defaults = _render(renderer, td, state, res)
        assert isinstance(defaults, ViewDefaults), f"{name} did not return a ViewDefaults"


def test_gfunction_leaves_twin_axes_unclipped_but_returns_percentile_y2lim():
    td = make_testdata()
    state = overview_state(td)
    res = compute_all(state, td)
    fig, ax, defaults = _render(plots.render_gfunction, td, state, res)
    dg = res.diagnostics
    finite = np.isfinite(dg.dPdG)
    hi = np.percentile(dg.dPdG[finite], 95)
    expected_clip = (0, min(max(hi * 1.5, 1.0), 50.0))

    assert defaults.y2lim == pytest.approx(expected_clip)

    twin = next(a for a in fig.axes if a is not ax)
    actual = twin.get_ylim()
    # The renderer no longer clips the twin axes itself -- the full (unclipped) data stays
    # visible, including whatever sits above the percentile clip that only the returned
    # ViewDefaults carries.
    assert actual != pytest.approx(expected_clip)
    assert actual[1] >= float(np.nanmax(dg.dPdG[finite]))


def test_gfunction_y2lim_default_capped_at_50_for_spiky_dpdg():
    """The 95th-pct*1.5 default would otherwise scale to whatever the early water-hammer spike
    demands; a hard 50 cap keeps the default view tightly zoomed to the meaningful early-time
    derivative regardless."""
    G = np.linspace(0.1, 20.0, 60)
    dPdG = np.full(60, 5.0)
    dPdG[:3] = 20000.0  # water-hammer spike
    p = np.linspace(5000.0, 4000.0, 60)
    res = DerivedResults()
    res.resampled = Resampled(dt=np.linspace(0.0, 3000.0, 60), p=p, n_raw=60)
    res.diagnostics = Diagnostics(G=G, dPdG=dPdG, GdPdG=G * dPdG, d2PdG2=np.gradient(dPdG, G),
                                  t=np.linspace(1.0, 3000.0, 60),
                                  p=p, dp=np.zeros(60), tdpdt=np.zeros(60))
    fig = Figure()
    ax = fig.add_subplot(111)
    defaults = plots.render_gfunction(ax, None, PickState(), res)
    assert defaults.y2lim[1] == pytest.approx(50.0)


def test_porepressure_does_not_force_axes_xlim_to_zero():
    td = make_testdata()
    state = overview_state(td)
    res = compute_all(state, td)
    fig, ax, defaults = _render(plots.render_porepressure, td, state, res)
    dg = res.diagnostics
    x = dg.t ** -0.5
    xmax = float(np.nanmax(x))

    assert defaults.xlim == pytest.approx((0.0, xmax))
    # Autoscaled (not forced to start at 0 by the renderer).
    assert ax.get_xlim()[0] != 0.0


def test_overview_returns_injection_window_instead_of_setting_it():
    td = make_testdata()
    state = overview_state(td)
    res = compute_all(state, td)
    fig, ax, defaults = _render(plots.render_overview, td, state, res)

    t_h = td.t_s / 3600.0
    span_h = max(t_h[state.shutin_idx] - t_h[state.start_idx], 0.25)
    last_active = int(np.where(res.rate_all > 0)[0][-1])
    t_end_h = t_h[last_active] + 0.25
    expected = (t_h[state.start_idx] - 0.5 * span_h,
               min(t_h[state.shutin_idx] + 2.0 * span_h, t_end_h))

    assert defaults.xlim == pytest.approx(expected)
    # The Axes itself is left autoscaled to the full file extent (a few hundred seconds here),
    # not the injection-window default -- whose 0.25 h floor dwarfs this short synthetic file.
    actual = ax.get_xlim()
    assert actual != pytest.approx(expected)
    assert actual[1] < float(t_h[-1]) + 0.05


def test_overview_clamps_plotted_data_to_last_nonzero_rate_plus_15_min():
    # A long falloff tail (dt=1s, n=3000 -> ~50 min) so the +15-min-past-last-rate clamp binds.
    td = make_testdata(n=3000, dt=1.0)
    state = overview_state(td)
    res = compute_all(state, td)
    fig, ax, defaults = _render(plots.render_overview, td, state, res)

    t_h = td.t_s / 3600.0
    last_active = int(np.where(res.rate_all > 0)[0][-1])
    t_end_h = t_h[last_active] + 0.25
    assert t_end_h < t_h[-1]  # the clamp really is binding for this file

    press_line = ax.get_lines()[0]
    assert press_line.get_xdata().max() <= t_end_h + 1e-9
    twin = next(a for a in fig.axes if a is not ax)
    rate_line = twin.get_lines()[0]
    assert rate_line.get_xdata().max() <= t_end_h + 1e-9
    assert defaults.xlim[1] == pytest.approx(t_end_h)


def test_overview_no_clamp_when_rate_is_none():
    td = make_testdata()
    state = PickState(pressure_col=PRESSURE_COL)
    res = compute_all(state, td)
    assert res.rate_all is None
    fig, ax, defaults = _render(plots.render_overview, td, state, res)

    t_h = td.t_s / 3600.0
    press_line = ax.get_lines()[0]
    assert press_line.get_xdata().max() == pytest.approx(float(t_h[-1]))
    assert not any(a is not ax for a in fig.axes)  # no rate -> no twin either


def test_resolve_view_first_visit_seeds_from_defaults_falling_back_to_full_extent():
    defaults = ViewDefaults(xlim=(1.0, 2.0), ylim=None, y2lim=None)
    view = _resolve_view(None, defaults, full_x=(0.0, 10.0), full_y=(0.0, 5.0), full_y2=(0.0, 3.0))
    assert view.xlim == (1.0, 2.0)  # renderer had an opinion
    assert view.ylim == (0.0, 5.0)  # renderer left it None -> full autoscaled extent
    assert view.y2lim == (0.0, 3.0)  # same for the twin axes


def test_resolve_view_revisit_reuses_stored_view_unchanged():
    stored = ViewState(xlim=(3.0, 4.0), ylim=(1.0, 2.0), y2lim=(0.0, 1.0))
    defaults = ViewDefaults(xlim=(100.0, 200.0))
    view = _resolve_view(stored, defaults, full_x=(0.0, 10.0), full_y=(0.0, 5.0), full_y2=None)
    assert view is stored


def test_gfunction_ylim_default_scales_from_pressure_data_only():
    # The effective-ISIP tangent's dashed extension can swing far outside the real BHP range;
    # defaults.ylim must come from the pressure data alone, not the Axes' full autoscale.
    td = make_testdata()
    state = overview_state(td)
    res = compute_all(state, td)
    picks.seed_isip(state, td, res)
    res = compute_all(state, td)
    picks.seed_gfunction(state, res)
    res = compute_all(state, td)
    fig, ax, defaults = _render(plots.render_gfunction, td, state, res)

    rs = res.resampled
    finite_p = np.isfinite(rs.p)
    p_lo, p_hi = float(np.nanmin(rs.p[finite_p])), float(np.nanmax(rs.p[finite_p]))
    pad = 0.05 * max(p_hi - p_lo, 1.0)

    assert defaults.ylim == pytest.approx((p_lo - pad, p_hi + pad))
    # The tangent construction is drawn on this same Axes, so its dashed extension really can
    # push the Axes' own autoscale far outside the pressure-data-only ylim above.
    actual = ax.get_ylim()
    assert actual[0] < defaults.ylim[0] or actual[1] > defaults.ylim[1]


# --------------------------------------------------------------------------------------------------
# DfitApp.refresh(): step-specific full_y/full_y2 clamps for the gfunction step (drives the
# slider's outer/full range, not just the default view -- ``_build_sliders`` reads these).
# --------------------------------------------------------------------------------------------------
def _refresh_stub(td, state, step):
    """Duck-typed DfitApp stand-in exposing only what refresh()/_build_sliders/_twin_axes touch,
    same headless-Agg approach as test_build_sliders.py's _make_app_stub."""
    stub = types.SimpleNamespace()
    stub.fig = Figure()
    stub.ax = stub.fig.add_subplot(111)
    stub.canvas = FigureCanvasAgg(stub.fig)
    stub.td = td
    stub.state = state
    stub.step = step
    stub._views = {}
    stub.txt_notes = types.SimpleNamespace(get=lambda *a, **kw: "")
    stub._attach_controllers = lambda: None
    stub._update_stepbar = lambda: None
    stub._update_panel_visibility = lambda: None
    stub._update_panel = lambda: None
    stub._make_range_slider = types.MethodType(DfitApp._make_range_slider, stub)
    stub._build_sliders = types.MethodType(DfitApp._build_sliders, stub)
    stub._twin_axes = types.MethodType(DfitApp._twin_axes, stub)
    stub._reconcile_pp_axis = types.MethodType(DfitApp._reconcile_pp_axis, stub)
    stub.refresh = types.MethodType(DfitApp.refresh, stub)
    return stub


def test_refresh_clamps_gfunction_full_y_to_pressure_data_and_full_y2_to_0_500():
    td = make_testdata()
    state = overview_state(td)
    res = compute_all(state, td)
    picks.seed_isip(state, td, res)
    res = compute_all(state, td)
    picks.seed_gfunction(state, res)
    res = compute_all(state, td)
    stub = _refresh_stub(td, state, "gfunction")

    stub.refresh()

    fig = Figure()
    ax = fig.add_subplot(111)
    defaults = plots.render_gfunction(ax, td, state, stub.res)

    # full_y: the y-slider's outer range must equal the renderer's data-driven ylim, not
    # whatever the Axes autoscaled to (which includes the tangent construction).
    assert (stub._y_slider.valmin, stub._y_slider.valmax) == pytest.approx(defaults.ylim)

    # full_y2: the dP/dG slider's outer range must never exceed 0-500.
    assert stub._y2_slider.valmin >= 0.0
    assert stub._y2_slider.valmax <= 500.0
