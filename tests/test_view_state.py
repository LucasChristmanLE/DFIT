"""Renderer view contract: render_* never sets Axes limits; it returns a ViewDefaults that the
caller (ui.py) applies. Also covers the pure view-resolution helper ui.py uses in refresh()."""

import types

import numpy as np
import pytest
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from dfit_tool import picks
from dfit_tool import ui as ui_module
from dfit_tool.model import DerivedResults, PickState, compute_all
from dfit_tool.resample import Diagnostics, Resampled
from dfit_tool import plots
from dfit_tool.plots import ViewDefaults
from dfit_tool.ui import DfitApp, ViewState, _resolve_view
from tests.helpers import PRESSURE_COL, make_testdata, injection_state


def _make_full_y2_mismatch_res():
    """A dP/dG curve that reproduces the full_y2-too-narrow bug: G>=1 max ~100 (so the
    renderer's own default y2lim is (0, 110.0)), a spike near G=0 (below Y2_SCALE_G_MIN, so it
    is excluded from that default) that isn't the record's global max, and a nonzero data
    minimum -- both of which make the twin Axes' own raw autoscale (~(0.84, 104.7), via
    mpl's default 5% margin vs the renderer's 10%) fall short of the default at BOTH ends."""
    G = np.linspace(0.1, 20.0, 60)
    dPdG = np.linspace(0.5, 100.0, 60)
    dPdG[:3] = 50.0  # spike near G=0, below the record's own max of 100
    p = np.linspace(5000.0, 4000.0, 60)
    res = DerivedResults()
    res.resampled = Resampled(dt=np.linspace(0.0, 3000.0, 60), p=p, n_raw=60)
    res.diagnostics = Diagnostics(G=G, dPdG=dPdG, GdPdG=G * dPdG, d2PdG2=np.gradient(dPdG, G),
                                  t=np.linspace(1.0, 3000.0, 60),
                                  p=p, dp=np.zeros(60), tdpdt=np.zeros(60))
    return res


def _render(renderer, td, state, res):
    fig = Figure()
    ax = fig.add_subplot(111)
    defaults = renderer(ax, td, state, res)
    return fig, ax, defaults


def test_all_renderers_return_view_defaults():
    td = make_testdata()
    state = injection_state(td)
    res = compute_all(state, td)
    for name, renderer in plots.RENDERERS.items():
        _, _, defaults = _render(renderer, td, state, res)
        assert isinstance(defaults, ViewDefaults), f"{name} did not return a ViewDefaults"


def test_gfunction_leaves_twin_axes_unclipped_but_returns_autoscaled_y2lim():
    td = make_testdata()
    state = injection_state(td)
    res = compute_all(state, td)
    fig, ax, defaults = _render(plots.render_gfunction, td, state, res)
    dg = res.diagnostics
    finite_all = np.isfinite(dg.dPdG)
    masked = finite_all & (dg.G >= plots.Y2_SCALE_G_MIN)
    if not masked.any():
        masked = finite_all
    hi = float(np.nanmax(dg.dPdG[masked]))
    expected_clip = (0, min(max(hi * 1.10, 1.0), plots.DPDG_VIEW_MAX))

    assert defaults.y2lim == pytest.approx(expected_clip)

    twin = next(a for a in fig.axes if a is not ax)
    actual = twin.get_ylim()
    # The renderer no longer clips the twin axes itself -- the full (unclipped) data stays
    # visible, including whatever sits above the autoscaled default that only the returned
    # ViewDefaults carries.
    assert actual != pytest.approx(expected_clip)
    assert actual[1] >= float(np.nanmax(dg.dPdG[finite_all]))


def test_gfunction_y2lim_default_excludes_early_g_spike():
    """The default view autoscales from dP/dG at G >= Y2_SCALE_G_MIN only, masking the early
    water-hammer spike (which sits below G=1) out of the scale entirely rather than clipping
    the whole default view to a fixed 50 cap."""
    G = np.linspace(0.1, 20.0, 60)
    dPdG = np.full(60, 5.0)
    dPdG[:3] = 20000.0  # water-hammer spike, at G ~ 0.1/0.44/0.78, all below G=1
    p = np.linspace(5000.0, 4000.0, 60)
    res = DerivedResults()
    res.resampled = Resampled(dt=np.linspace(0.0, 3000.0, 60), p=p, n_raw=60)
    res.diagnostics = Diagnostics(G=G, dPdG=dPdG, GdPdG=G * dPdG, d2PdG2=np.gradient(dPdG, G),
                                  t=np.linspace(1.0, 3000.0, 60),
                                  p=p, dp=np.zeros(60), tdpdt=np.zeros(60))
    fig = Figure()
    ax = fig.add_subplot(111)
    defaults = plots.render_gfunction(ax, None, PickState(), res)
    assert defaults.y2lim[1] == pytest.approx(5.5)


def test_gfunction_y2lim_default_no_longer_capped_at_50():
    """A record whose post-G=1 dP/dG genuinely runs to ~300 should autoscale there instead of
    being squashed by the old hard 50 cap."""
    G = np.linspace(0.1, 20.0, 60)
    dPdG = np.linspace(10.0, 300.0, 60)
    dPdG[:3] = 20000.0  # water-hammer spike, still below G=1
    p = np.linspace(5000.0, 4000.0, 60)
    res = DerivedResults()
    res.resampled = Resampled(dt=np.linspace(0.0, 3000.0, 60), p=p, n_raw=60)
    res.diagnostics = Diagnostics(G=G, dPdG=dPdG, GdPdG=G * dPdG, d2PdG2=np.gradient(dPdG, G),
                                  t=np.linspace(1.0, 3000.0, 60),
                                  p=p, dp=np.zeros(60), tdpdt=np.zeros(60))
    fig = Figure()
    ax = fig.add_subplot(111)
    defaults = plots.render_gfunction(ax, None, PickState(), res)
    assert defaults.y2lim[1] == pytest.approx(330.0)


def test_gfunction_y2lim_default_clamped_at_dpdg_view_max():
    """A record whose real dP/dG runs even higher (~900) still clamps at DPDG_VIEW_MAX so the
    default view can never exceed the 0-500 hard bound."""
    G = np.linspace(0.1, 20.0, 60)
    dPdG = np.linspace(10.0, 900.0, 60)
    dPdG[:3] = 20000.0  # water-hammer spike, still below G=1
    p = np.linspace(5000.0, 4000.0, 60)
    res = DerivedResults()
    res.resampled = Resampled(dt=np.linspace(0.0, 3000.0, 60), p=p, n_raw=60)
    res.diagnostics = Diagnostics(G=G, dPdG=dPdG, GdPdG=G * dPdG, d2PdG2=np.gradient(dPdG, G),
                                  t=np.linspace(1.0, 3000.0, 60),
                                  p=p, dp=np.zeros(60), tdpdt=np.zeros(60))
    fig = Figure()
    ax = fig.add_subplot(111)
    defaults = plots.render_gfunction(ax, None, PickState(), res)
    assert defaults.y2lim[1] == pytest.approx(plots.DPDG_VIEW_MAX)


def test_gfunction_y2lim_falls_back_to_all_finite_when_whole_record_below_g_min():
    """A record whose entire G range sits below Y2_SCALE_G_MIN has nothing surviving the mask,
    so the default falls back to scaling from all finite dP/dG instead of returning None."""
    G = np.linspace(0.01, 0.5, 60)
    dPdG = np.linspace(10.0, 40.0, 60)
    p = np.linspace(5000.0, 4000.0, 60)
    res = DerivedResults()
    res.resampled = Resampled(dt=np.linspace(0.0, 3000.0, 60), p=p, n_raw=60)
    res.diagnostics = Diagnostics(G=G, dPdG=dPdG, GdPdG=G * dPdG, d2PdG2=np.gradient(dPdG, G),
                                  t=np.linspace(1.0, 3000.0, 60),
                                  p=p, dp=np.zeros(60), tdpdt=np.zeros(60))
    fig = Figure()
    ax = fig.add_subplot(111)
    defaults = plots.render_gfunction(ax, None, PickState(), res)
    assert defaults.y2lim is not None
    assert defaults.y2lim[1] == pytest.approx(44.0)


def test_porepressure_does_not_force_axes_xlim_to_zero():
    td = make_testdata()
    state = injection_state(td)
    res = compute_all(state, td)
    fig, ax, defaults = _render(plots.render_porepressure, td, state, res)

    # The default view is the fixed per-axis early-time zoom (tm12: 0..0.05, tm1: 0..0.0025),
    # returned via ViewDefaults for ui.py to apply -- not written onto the Axes here.
    expected_hi = 0.05 if state.pp_axis == "tm12" else 0.0025
    assert defaults.xlim == pytest.approx((0.0, expected_hi))
    # Autoscaled (not forced to start at 0 by the renderer).
    assert ax.get_xlim()[0] != 0.0


def test_render_overview_plots_full_dataset_unmasked():
    # Same long-falloff-tail file that binds render_injection's "last nonzero rate + 15 min"
    # clamp -- render_overview must plot the whole thing, unclamped, with the full autoscaled
    # extent as its default xlim, a 3x-max-rate y2lim, but a pressure ylim pinned to 0 at the bottom (CLAUDE.md
    # TODO: "Overview tab make pressure ymin always 0").
    td = make_testdata(n=3000, dt=1.0)
    state = injection_state(td)
    res = compute_all(state, td)
    fig, ax, defaults = _render(plots.render_overview, td, state, res)

    t_h = td.t_s / 3600.0
    press_line = ax.get_lines()[0]
    assert press_line.get_xdata().max() == pytest.approx(float(t_h[-1]))
    assert defaults.xlim is None
    assert defaults.y2lim == (0.0, plots.RATE_VIEW_FACTOR * float(np.nanmax(res.rate_all)))
    assert defaults.ylim[0] == 0.0

    gids = {ln.get_gid() for ln in ax.get_lines()}
    assert "start_ref" in gids
    assert "shutin_ref" in gids


def test_injection_returns_injection_window_instead_of_setting_it():
    td = make_testdata()
    state = injection_state(td)
    res = compute_all(state, td)
    fig, ax, defaults = _render(plots.render_injection, td, state, res)

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


def test_injection_clamps_plotted_data_to_last_nonzero_rate_plus_15_min():
    # A long falloff tail (dt=1s, n=3000 -> ~50 min) so the +15-min-past-last-rate clamp binds.
    td = make_testdata(n=3000, dt=1.0)
    state = injection_state(td)
    res = compute_all(state, td)
    fig, ax, defaults = _render(plots.render_injection, td, state, res)

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


def test_injection_no_clamp_when_rate_is_none():
    td = make_testdata()
    state = PickState(pressure_col=PRESSURE_COL)
    res = compute_all(state, td)
    assert res.rate_all is None
    fig, ax, defaults = _render(plots.render_injection, td, state, res)

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
    state = injection_state(td)
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
    stub.gate_lbl = types.SimpleNamespace(config=lambda **kw: None)
    stub._attach_controllers = lambda: None
    stub._update_stepbar = lambda: None
    stub._update_panel_visibility = lambda: None
    stub._update_panel = lambda: None
    stub._update_unit_labels = lambda: None
    stub._x_slider = None
    stub._y_slider = None
    stub._y2_slider = None
    stub._make_range_slider = types.MethodType(DfitApp._make_range_slider, stub)
    stub._build_sliders = types.MethodType(DfitApp._build_sliders, stub)
    stub._twin_axes = types.MethodType(DfitApp._twin_axes, stub)
    stub._d2_axes = types.MethodType(DfitApp._d2_axes, stub)
    stub._layout_sliders = types.MethodType(DfitApp._layout_sliders, stub)
    stub._x_track_geometry_px = types.MethodType(DfitApp._x_track_geometry_px, stub)
    stub._get_renderer = types.MethodType(DfitApp._get_renderer, stub)
    stub._measure_overhang_px = types.MethodType(DfitApp._measure_overhang_px, stub)
    stub._measure_bottom_overhang_px = types.MethodType(DfitApp._measure_bottom_overhang_px, stub)
    stub._measure_slider_text_col_px = types.MethodType(
        DfitApp._measure_slider_text_col_px, stub)
    stub._reconcile_pp_axis = types.MethodType(DfitApp._reconcile_pp_axis, stub)
    stub.refresh = types.MethodType(DfitApp.refresh, stub)
    return stub


def test_refresh_clamps_gfunction_full_y_to_pressure_data_and_full_y2_to_0_500():
    td = make_testdata()
    state = injection_state(td)
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


def test_refresh_unions_overview_full_y_down_to_zero():
    """render_overview's ViewDefaults.ylim pins the bottom at 0 psi (CLAUDE.md TODO), which sits
    below the Axes' own autoscaled extent for a normal BHP trace -- refresh() must UNION that
    into full_y (not replace it, that's gfunction-only) so the y-slider's outer range still
    reaches 0 instead of silently clamping the default view back up on the first slider touch."""
    td = make_testdata()
    state = injection_state(td)
    stub = _refresh_stub(td, state, "overview")

    stub.refresh()

    assert stub._y_slider.valmin == pytest.approx(0.0)


# --------------------------------------------------------------------------------------------------
# decision D3: the gfunction step's d2P/dG2 axis gets no slider and no persisted view -- refresh()
# applies the renderer's fresh y3lim every time instead.
# --------------------------------------------------------------------------------------------------
def test_refresh_applies_fresh_y3lim_to_d2_axis():
    td = make_testdata()
    state = injection_state(td)
    res = compute_all(state, td)
    picks.seed_isip(state, td, res)
    res = compute_all(state, td)
    picks.seed_gfunction(state, res)
    state.show_d2pdg2 = True
    res = compute_all(state, td)
    stub = _refresh_stub(td, state, "gfunction")

    stub.refresh()

    fig = Figure()
    ax = fig.add_subplot(111)
    defaults = plots.render_gfunction(ax, td, state, stub.res)
    assert defaults.y3lim is not None
    d2_axis = stub._d2_axes()
    assert d2_axis is not None
    assert d2_axis.get_ylim() == pytest.approx(defaults.y3lim)


def test_refresh_builds_no_third_slider_for_d2_axis():
    """No y3 slider exists at all -- _build_sliders only ever makes x/y/y2, and _twin_axes
    (which the y2 slider is built from) excludes the d2 axis (D2_AXIS_GID)."""
    td = make_testdata()
    state = injection_state(td)
    res = compute_all(state, td)
    picks.seed_isip(state, td, res)
    res = compute_all(state, td)
    picks.seed_gfunction(state, res)
    state.show_d2pdg2 = True
    res = compute_all(state, td)
    stub = _refresh_stub(td, state, "gfunction")

    stub.refresh()

    assert not hasattr(stub, "_y3_slider")
    # the y2 slider still targets the dP/dG twin, not the d2 axis
    assert stub._y2_slider is not None
    assert stub._twin_axes().get_gid() != plots.D2_AXIS_GID


def test_refresh_d2_ylim_not_persisted_across_refreshes():
    """Panning/zooming the d2 axis is impossible (no slider), but even if the underlying data
    changed between refreshes the axis must show the renderer's current default, never a value
    left over from ``_views`` (which never stores it, per decision D3)."""
    td = make_testdata()
    state = injection_state(td)
    res = compute_all(state, td)
    picks.seed_isip(state, td, res)
    res = compute_all(state, td)
    picks.seed_gfunction(state, res)
    state.show_d2pdg2 = True
    res = compute_all(state, td)
    stub = _refresh_stub(td, state, "gfunction")

    stub.refresh()
    d2_axis = stub._d2_axes()
    stale_ylim = (-9999.0, 9999.0)
    d2_axis.set_ylim(stale_ylim)  # simulate a leftover/stale limit

    stub.refresh()  # a fresh Figure/Axes is built each refresh -- get the new d2 axis
    d2_axis = stub._d2_axes()
    assert d2_axis.get_ylim() != stale_ylim

    fig = Figure()
    ax = fig.add_subplot(111)
    defaults = plots.render_gfunction(ax, td, state, stub.res)
    assert d2_axis.get_ylim() == pytest.approx(defaults.y3lim)


# --------------------------------------------------------------------------------------------------
# The y2 slider's full/travel range must CONTAIN the renderer's own default y2lim, not just the
# twin Axes' raw autoscale -- the raw autoscale (mpl's own ~5% margin) can be narrower than the
# renderer's masked-and-10%-padded default at either end, and _make_range_slider's valinit
# clamping would otherwise silently snap the view off the default on the first slider touch.
# --------------------------------------------------------------------------------------------------
def test_refresh_unions_gfunction_full_y2_with_default_view(monkeypatch):
    td = make_testdata()
    state = injection_state(td)
    fake_res = _make_full_y2_mismatch_res()
    monkeypatch.setattr(ui_module, "compute_all", lambda st, t: fake_res)
    stub = _refresh_stub(td, state, "gfunction")

    stub.refresh()

    fig = Figure()
    ax = fig.add_subplot(111)
    defaults = plots.render_gfunction(ax, td, state, fake_res)
    assert defaults.y2lim is not None

    # full_y2 (the slider's outer/travel range) must contain the default at both ends.
    assert stub._y2_slider.valmin <= defaults.y2lim[0]
    assert stub._y2_slider.valmax >= defaults.y2lim[1]

    # And the initial view actually applied is the unclamped default, not a valinit-clamped
    # version of it -- this is what silently overwrote the stored ViewState.y2lim before the fix.
    assert stub._y2_slider.val == pytest.approx(defaults.y2lim)


def test_render_step_figure_unions_gfunction_full_y2_with_default_view():
    """Same containment property as test_refresh_unions_gfunction_full_y2_with_default_view,
    covering the headless export path's own full_y2-based fallback (plots.render_step_figure
    with stored_view=None) so both lockstep blocks are exercised."""
    td = make_testdata()
    state = injection_state(td)
    res = _make_full_y2_mismatch_res()

    fig = plots.render_step_figure("gfunction", td, state, res, stored_view=None)

    twin = next(a for a in fig.axes if a is not fig.axes[0])
    fig2 = Figure()
    ax2 = fig2.add_subplot(111)
    defaults = plots.render_gfunction(ax2, td, state, res)
    assert defaults.y2lim is not None

    applied = twin.get_ylim()
    assert applied == pytest.approx(defaults.y2lim)
