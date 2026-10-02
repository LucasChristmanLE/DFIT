"""Headless coverage for DfitApp._build_sliders / _make_range_slider / _twin_axes /
_layout_sliders, and for the pure ``sliders.right_margin_layout`` helper and
``sliders.PanRangeSlider``'s split value text / color.

DfitApp itself needs a real tk.Tk() root (it's built in __init__), which this headless (Agg,
no display) suite can't construct. These tests instead exercise the methods directly against a
duck-typed stand-in exposing only what the methods touch (self.fig/self.ax/self.canvas), binding
the real DfitApp methods onto it via types.MethodType -- same headless-Agg-plus-real-widgets
approach as test_drag_controller.py, just without a full DfitApp instance backing it.
"""
from __future__ import annotations

import types

import matplotlib.colors as mcolors
import pytest
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from dfit_tool import plots
from dfit_tool.model import compute_all
from dfit_tool.sliders import (PanRangeSlider, TRACK_PX, bottom_margin_layout,
                               right_margin_layout, to_log_bounds)
from dfit_tool.plots import ViewState
from dfit_tool.ui import (DfitApp, _FALLBACK_BOTTOM_OVERHANG_PX,
                         _FALLBACK_OVERHANG_PX, _SLIDER_NEUTRAL_COLOR)
from tests.helpers import injection_state, make_testdata


def _bare_stub(figsize, dpi=100.0):
    """A duck-typed DfitApp stand-in with only self.fig/self.ax/self.canvas and the slider-
    layout methods bound -- the caller plots its own Axes content before building sliders."""
    stub = types.SimpleNamespace()
    stub.fig = Figure(figsize=figsize, dpi=dpi)
    stub.ax = stub.fig.add_subplot(111)
    stub.canvas = FigureCanvasAgg(stub.fig)
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
    return stub

# Deterministic figure size for every stub -- _build_sliders/_layout_sliders now compute rects
# from the figure's pixel width, so a test asserting an exact rect needs a known fig_w_px.
_FIG_W_IN, _FIG_H_IN = 9.0, 6.0
_DPI = 100.0
_FIG_W_PX = _FIG_W_IN * _DPI


def _make_app_stub():
    return _bare_stub((_FIG_W_IN, _FIG_H_IN), _DPI)


# --------------------------------------------------------------------------------------------------
# rects / y2-only-with-twin -- computed from the same pure helper _build_sliders now uses (with
# its fallback overhang, since it has no renderer yet), rather than the old fixed fractions.
# --------------------------------------------------------------------------------------------------
def test_x_and_y_slider_rects_no_twin():
    stub = _make_app_stub()
    view = ViewState(xlim=(2.0, 8.0), ylim=(1.0, 9.0), y2lim=None)
    stub._build_sliders(full_x=(0.0, 10.0), full_y=(0.0, 10.0), full_y2=None, view=view, twin=None)

    axes_right_frac, x_fracs = right_margin_layout(_FIG_W_PX, 1, _FALLBACK_OVERHANG_PX)
    track_w = TRACK_PX / _FIG_W_PX
    fig_h_px = _FIG_H_IN * _DPI
    default_track_y0_px, track_h_px, text_below_px, _ = stub._x_track_geometry_px(fig_h_px)
    _, track_y0_frac = bottom_margin_layout(
        fig_h_px, default_track_y0_px, track_h_px, text_below_px, _FALLBACK_BOTTOM_OVERHANG_PX)

    assert stub._x_slider.ax.get_position().bounds == pytest.approx(
        (0.10, track_y0_frac, axes_right_frac - 0.10, track_h_px / fig_h_px))
    assert stub._y_slider.ax.get_position().bounds == pytest.approx(
        (x_fracs[0], 0.16, track_w, 0.74))
    assert stub._y2_slider is None


def test_y2_slider_only_built_when_twin_present_with_correct_rect():
    stub = _make_app_stub()
    twin = stub.ax.twinx()
    twin.set_ylim(0.0, 100.0)
    view = ViewState(xlim=(0.0, 10.0), ylim=(0.0, 10.0), y2lim=(10.0, 90.0))
    stub._build_sliders(full_x=(0.0, 10.0), full_y=(0.0, 10.0), full_y2=(0.0, 100.0),
                        view=view, twin=twin)

    axes_right_frac, x_fracs = right_margin_layout(_FIG_W_PX, 2, _FALLBACK_OVERHANG_PX)
    track_w = TRACK_PX / _FIG_W_PX

    assert stub._y2_slider is not None
    assert stub._y2_slider.ax.get_position().bounds == pytest.approx(
        (x_fracs[1], 0.16, track_w, 0.74))


# --------------------------------------------------------------------------------------------------
# gid tagging / _twin_axes exclusion
# --------------------------------------------------------------------------------------------------
def test_slider_axes_tagged_slider_gid_and_twin_axes_excludes_them():
    stub = _make_app_stub()
    twin = stub.ax.twinx()
    twin.set_ylim(0.0, 100.0)
    view = ViewState(xlim=(0.0, 10.0), ylim=(0.0, 10.0), y2lim=(10.0, 90.0))
    stub._build_sliders(full_x=(0.0, 10.0), full_y=(0.0, 10.0), full_y2=(0.0, 100.0),
                        view=view, twin=twin)

    for slider in (stub._x_slider, stub._y_slider, stub._y2_slider):
        assert slider.ax.get_gid() == "slider"

    # _twin_axes() must still find the real twin, not one of the three slider Axes now sharing
    # the figure with it.
    assert stub._twin_axes() is twin


def test_twin_axes_returns_none_when_only_slider_axes_present():
    stub = _make_app_stub()
    view = ViewState(xlim=(0.0, 10.0), ylim=(0.0, 10.0), y2lim=None)
    stub._build_sliders(full_x=(0.0, 10.0), full_y=(0.0, 10.0), full_y2=None, view=view, twin=None)
    assert stub._twin_axes() is None


# --------------------------------------------------------------------------------------------------
# valinit == current view, range == full extent
# --------------------------------------------------------------------------------------------------
def test_valinit_is_current_view_range_is_full_extent():
    stub = _make_app_stub()
    view = ViewState(xlim=(2.0, 8.0), ylim=(1.0, 9.0), y2lim=None)
    stub._build_sliders(full_x=(0.0, 10.0), full_y=(0.0, 10.0), full_y2=None, view=view, twin=None)

    assert stub._x_slider.val == pytest.approx((2.0, 8.0))
    assert (stub._x_slider.valmin, stub._x_slider.valmax) == pytest.approx((0.0, 10.0))
    assert stub._y_slider.val == pytest.approx((1.0, 9.0))
    assert (stub._y_slider.valmin, stub._y_slider.valmax) == pytest.approx((0.0, 10.0))


def test_valinit_equals_valmin_valmax_when_view_equals_full_extent():
    # Common case per the brief: first visit to a step, current view == the full autoscaled
    # extent. RangeSlider must still build/respond normally with valinit == (valmin, valmax).
    stub = _make_app_stub()
    view = ViewState(xlim=(0.0, 10.0), ylim=(0.0, 10.0), y2lim=None)
    stub._build_sliders(full_x=(0.0, 10.0), full_y=(0.0, 10.0), full_y2=None, view=view, twin=None)

    assert stub._x_slider.val == pytest.approx((0.0, 10.0))
    stub._x_slider.set_val((1.0, 9.0))
    assert view.xlim == pytest.approx((1.0, 9.0))


# --------------------------------------------------------------------------------------------------
# log-scaled axes
# --------------------------------------------------------------------------------------------------
def test_log_scale_axis_builds_in_log10_space_and_callback_exponentiates():
    stub = _make_app_stub()
    stub.ax.set_xscale("log")
    view = ViewState(xlim=(1.0, 1000.0), ylim=(0.0, 10.0), y2lim=None)
    stub._build_sliders(full_x=(1.0, 1000.0), full_y=(0.0, 10.0), full_y2=None, view=view, twin=None)

    expected_lo, expected_hi = to_log_bounds(1.0, 1000.0)
    assert (stub._x_slider.valmin, stub._x_slider.valmax) == pytest.approx((expected_lo, expected_hi))
    assert stub._x_slider.val == pytest.approx((expected_lo, expected_hi))  # valinit round-trips

    stub._x_slider.set_val(to_log_bounds(10.0, 100.0))
    assert stub.ax.get_xlim() == pytest.approx((10.0, 100.0))
    assert view.xlim == pytest.approx((10.0, 100.0))


# --------------------------------------------------------------------------------------------------
# degenerate / non-finite extents
# --------------------------------------------------------------------------------------------------
def test_degenerate_extent_skips_that_slider_without_crashing():
    stub = _make_app_stub()
    view = ViewState(xlim=(5.0, 5.0), ylim=(0.0, 10.0), y2lim=None)
    stub._build_sliders(full_x=(5.0, 5.0), full_y=(0.0, 10.0), full_y2=None, view=view, twin=None)

    assert stub._x_slider is None
    assert stub._y_slider is not None


def test_nonfinite_extent_skips_that_slider_without_crashing():
    stub = _make_app_stub()
    view = ViewState(xlim=(0.0, float("inf")), ylim=(0.0, 10.0), y2lim=None)
    stub._build_sliders(full_x=(0.0, float("inf")), full_y=(0.0, 10.0), full_y2=None,
                        view=view, twin=None)

    assert stub._x_slider is None
    assert stub._y_slider is not None


# --------------------------------------------------------------------------------------------------
# on_changed: sets limits + mutates ViewState in place, never clears the figure
# --------------------------------------------------------------------------------------------------
def test_on_changed_sets_axes_limits_mutates_view_and_does_not_clear_figure():
    stub = _make_app_stub()
    view = ViewState(xlim=(2.0, 8.0), ylim=(1.0, 9.0), y2lim=None)
    stub._build_sliders(full_x=(0.0, 10.0), full_y=(0.0, 10.0), full_y2=None, view=view, twin=None)
    axes_before = list(stub.fig.axes)

    stub._x_slider.set_val((3.0, 7.0))

    assert stub.ax.get_xlim() == pytest.approx((3.0, 7.0))
    assert view.xlim == pytest.approx((3.0, 7.0))
    # unrelated fields untouched
    assert view.ylim == pytest.approx((1.0, 9.0))
    # fig.clf() would have dropped every Axes (including the slider that just fired) -- assert
    # the exact same Axes objects are still there.
    assert stub.fig.axes == axes_before


# --------------------------------------------------------------------------------------------------
# sliders.right_margin_layout: pure pixel layout
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("fig_w_px", [900.0, 1440.0])
@pytest.mark.parametrize("n_vertical,overhang_px", [(1, 40.0), (2, 75.0)])
def test_right_margin_layout_columns_dont_overlap_and_clear_overhang(fig_w_px, n_vertical,
                                                                      overhang_px):
    col_px, pad_px = 60.0, 12.0
    axes_right_frac, x_fracs = right_margin_layout(fig_w_px, n_vertical, overhang_px,
                                                    col_px=col_px, pad_px=pad_px)
    assert len(x_fracs) == n_vertical
    axes_right_px = axes_right_frac * fig_w_px
    margin_start_px = fig_w_px - n_vertical * col_px

    # The plot's right edge, plus the measured overhang, must clear where the slider columns
    # begin (with room left over for the pad gap) -- never sit on top of the twin's labels.
    assert axes_right_px + overhang_px <= margin_start_px + 1e-9

    # Each track is centered in its own col_px-wide column and stays inside it, so adjacent
    # tracks (and the gap between columns) never overlap.
    x0_px = [f * fig_w_px for f in x_fracs]
    for i, x0 in enumerate(x0_px):
        col_left = margin_start_px + i * col_px
        assert col_left <= x0 and x0 + TRACK_PX <= col_left + col_px
    for i in range(len(x0_px) - 1):
        assert x0_px[i] + TRACK_PX <= x0_px[i + 1]


def test_right_margin_layout_zero_vertical_reserves_no_columns():
    axes_right_frac, x_fracs = right_margin_layout(900.0, 0, 40.0)
    assert x_fracs == []
    assert axes_right_frac * 900.0 == pytest.approx(900.0 - 12.0 - 40.0)


# --------------------------------------------------------------------------------------------------
# _layout_sliders: real Agg render, tracks stay clear of the twin's tick labels, and the two
# sliders' split value texts don't collide.
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("figsize", [(9.0, 6.0), (14.0, 8.5)])
def test_layout_sliders_no_overlap_with_twin_labels_or_between_value_texts(figsize):
    stub = _bare_stub(figsize)
    stub.ax.plot([0, 1, 2], [1000.0, 5000.0, 9368.0])
    stub.ax.set_ylabel("pressure (psi)")
    twin = stub.ax.twinx()
    twin.plot([0, 1, 2], [-0.54, 5.0, 11.31])
    twin.set_ylabel("dP/dG", color="tab:red")
    twin.tick_params(axis="y", labelcolor="tab:red")

    view = ViewState(xlim=(0.0, 2.0), ylim=(0.0, 9368.0), y2lim=(-0.54, 11.31))
    stub._build_sliders(full_x=(0.0, 2.0), full_y=(0.0, 9368.0), full_y2=(-0.54, 11.31),
                        view=view, twin=twin, y_color="black", y2_color="tab:red")
    stub._layout_sliders()
    stub.canvas.draw()
    renderer = stub.canvas.get_renderer()

    twin_bbox = twin.get_tightbbox(renderer)
    y_bbox = stub._y_slider.ax.get_window_extent(renderer)
    y2_bbox = stub._y2_slider.ax.get_window_extent(renderer)
    assert twin_bbox.x1 < y_bbox.x0
    assert y_bbox.x1 <= y2_bbox.x0

    for t1 in (stub._y_slider._hi_text, stub._y_slider._lo_text):
        for t2 in (stub._y2_slider._hi_text, stub._y2_slider._lo_text):
            assert not t1.get_window_extent(renderer).overlaps(t2.get_window_extent(renderer))


@pytest.mark.parametrize("fig_w_px", [700.0, 1400.0])
def test_layout_sliders_x_slider_split_text_stays_inside_figure_no_twin(fig_w_px):
    """Regression: the x slider's rect now matches the plot's own width (right up to
    axes_right_frac), which is close to the figure's right edge on a no-twin step (one vertical
    column only) -- the stock single valtext used to sit to the right of the WHOLE track and
    could clip off the figure edge there. The split lo/hi text anchors inward off each end
    instead (see sliders.py), so it must stay inside the figure at both a narrow (700px) and a
    wide (1400px) width."""
    stub = _bare_stub((fig_w_px / 100.0, 6.0))
    stub.ax.plot([0, 1, 2], [1000.0, 5000.0, 9368.0])
    stub.ax.set_ylabel("BHP (psi)")

    view = ViewState(xlim=(-1.0, 3.0), ylim=(1850.0, 5150.0), y2lim=None)
    stub._build_sliders(full_x=(-1.0, 3.0), full_y=(1850.0, 5150.0), full_y2=None,
                        view=view, twin=None, y_color="black")
    stub._layout_sliders()
    stub.canvas.draw()
    renderer = stub.canvas.get_renderer()

    fig_bbox = stub.fig.bbox
    for text in (stub._x_slider._lo_text, stub._x_slider._hi_text):
        bbox = text.get_window_extent(renderer)
        assert bbox.x0 >= 0.0
        assert bbox.x1 <= fig_bbox.width


@pytest.mark.parametrize("fig_h_px", [350.0, 400.0, 600.0, 850.0])
def test_layout_sliders_x_slider_text_inside_figure_bottom_and_track_below_tick_labels(fig_h_px):
    """Short canvases: the x slider's split text must not clip off the figure bottom, and the
    track must sit below the x-axis tick labels/xlabel rather than on top of them."""
    stub = _bare_stub((9.0, fig_h_px / 100.0))
    stub.ax.plot([0, 1, 2], [1000.0, 5000.0, 9368.0])
    stub.ax.set_xlabel("time from file start (h)")
    stub.ax.set_ylabel("BHP (psi)")

    view = ViewState(xlim=(0.0, 2.0), ylim=(0.0, 9368.0), y2lim=None)
    stub._build_sliders(full_x=(0.0, 2.0), full_y=(0.0, 9368.0), full_y2=None,
                        view=view, twin=None, y_color="black")
    stub._layout_sliders()
    stub.canvas.draw()
    renderer = stub.canvas.get_renderer()

    for text in (stub._x_slider._lo_text, stub._x_slider._hi_text):
        assert text.get_window_extent(renderer).y0 >= 0.0
    track_top = stub._x_slider.ax.get_window_extent(renderer).y1
    assert track_top <= stub.ax.xaxis.get_tightbbox(renderer).y0


@pytest.mark.parametrize("dpi", [150.0, 200.0])
@pytest.mark.parametrize("with_twin", [False, True])
def test_layout_sliders_high_dpi_texts_inside_figure_and_columns_dont_collide(dpi, with_twin):
    """Windows display scaling raises fig.dpi under TkAgg; the pixel layout constants must scale
    with it, so long labels (log-format "1.3e+03", 5-digit psi) stay inside the figure and the
    two vertical sliders' texts don't collide."""
    stub = _bare_stub((9.0, 6.0), dpi=dpi)
    stub.ax.set_yscale("log")
    stub.ax.plot([1, 10, 100], [1.0, 100.0, 1300.0])
    stub.ax.set_ylabel("dp, t*dP/dt (psi)")
    twin = None
    if with_twin:
        twin = stub.ax.twinx()
        twin.plot([1, 10, 100], [0.0, 5000.0, 12345.0])
        twin.set_ylabel("dP/dG", color="tab:red")

    view = ViewState(xlim=(1.0, 100.0), ylim=(1.0, 1300.0),
                     y2lim=(0.0, 12345.0) if with_twin else None)
    stub._build_sliders(full_x=(1.0, 100.0), full_y=(1.0, 1300.0),
                        full_y2=(0.0, 12345.0) if with_twin else None,
                        view=view, twin=twin, y_color=None, y2_color="tab:red")
    stub._layout_sliders()
    stub.canvas.draw()
    renderer = stub.canvas.get_renderer()

    fig_w, fig_h = stub.fig.bbox.width, stub.fig.bbox.height
    verticals = [s for s in (stub._y_slider, stub._y2_slider) if s is not None]
    for s in verticals + [stub._x_slider]:
        for text in (s._lo_text, s._hi_text):
            bb = text.get_window_extent(renderer)
            assert bb.x0 >= 0.0 and bb.x1 <= fig_w
            assert bb.y0 >= 0.0 and bb.y1 <= fig_h
    if with_twin:
        assert twin.get_tightbbox(renderer).x1 < stub._y_slider.ax.get_window_extent(renderer).x0
        for t1 in (stub._y_slider._hi_text, stub._y_slider._lo_text):
            for t2 in (stub._y2_slider._hi_text, stub._y2_slider._lo_text):
                assert not t1.get_window_extent(renderer).overlaps(t2.get_window_extent(renderer))


@pytest.mark.parametrize("dpi", [100.0, 200.0])
def test_layout_sliders_x_track_matches_axes_right_edge_at_pathological_width(dpi):
    """A figure narrow enough that the 0.3 right-margin clamp fires: building the sliders must
    not raise (the unclamped x-slider width used to go negative and crash fig.add_axes), the
    clamp must actually be in effect, and the x track's right edge must match the axes'."""
    stub = _bare_stub((2.0, 6.0), dpi=dpi)
    stub.ax.plot([0, 1, 2], [1000.0, 5000.0, 9368.0])
    twin = stub.ax.twinx()
    twin.plot([0, 1, 2], [0.0, 50.0, 100.0])
    twin.set_ylabel("dP/dG")

    view = ViewState(xlim=(0.0, 2.0), ylim=(0.0, 9368.0), y2lim=(0.0, 100.0))
    stub._build_sliders(full_x=(0.0, 2.0), full_y=(0.0, 9368.0), full_y2=(0.0, 100.0),
                        view=view, twin=twin, y_color="black", y2_color="tab:red")
    stub._layout_sliders()
    x_pos = stub._x_slider.ax.get_position()
    ax_pos = stub.ax.get_position()
    assert ax_pos.x1 == pytest.approx(0.3, abs=1e-6)  # the clamp fired
    assert x_pos.x1 == pytest.approx(ax_pos.x1, abs=1e-6)


def test_layout_sliders_d2_axis_widen_settles_after_resize():
    """Regression: the d2P/dG2 axis's third spine sits at a FIXED FRACTION of the primary Axes'
    own width (("axes", 1.12)), so widening the figure moves its absolute pixel overhang too --
    a single measure-then-apply pass under-reserves for it after a big widen. Render at 7 in,
    resize to 19 in, and re-run _layout_sliders: the d2 axis must still clear the y slider."""
    td = make_testdata()
    st = injection_state(td)
    st.show_d2pdg2 = True
    res = compute_all(st, td)

    stub = _bare_stub((7.0, 6.0))
    defaults = plots.render_gfunction(stub.ax, td, st, res)
    sv = plots.apply_step_view("gfunction", stub.ax, defaults)
    d2 = stub._d2_axes()

    stub._build_sliders(sv.full_x, sv.full_y, sv.full_y2, sv.view, sv.twin,
                        y_color=defaults.y_color, y2_color=defaults.y2_color)
    stub._layout_sliders()
    stub.canvas.draw()

    stub.fig.set_size_inches(19.0, 6.0)
    stub._layout_sliders()
    stub.canvas.draw()
    renderer = stub.canvas.get_renderer()

    d2_bbox = d2.get_tightbbox(renderer)
    y_bbox = stub._y_slider.ax.get_window_extent(renderer)
    assert d2_bbox.x1 < y_bbox.x0


def test_layout_sliders_runs_subplots_adjust_even_with_no_sliders():
    """Regression: _layout_sliders used to return early (skipping subplots_adjust entirely)
    whenever both the x and y slider refs were None, which also skipped it for a degenerate/
    not-yet-loaded step -- leaving the plot's margin at whatever a previous step happened to
    set. It must now always adjust the margin, even with all three slider refs (x/y/y2) None."""
    stub = _bare_stub((9.0, 6.0))
    # No _build_sliders() call at all -- every slider ref stays None (its __init__ default).
    assert stub._x_slider is None and stub._y_slider is None and stub._y2_slider is None
    before = stub.ax.get_position().bounds

    stub._layout_sliders()

    assert stub.ax.get_position().bounds != before


def test_measure_overhang_px_skips_an_axes_whose_tightbbox_is_none(monkeypatch):
    """get_tightbbox can come back None for some Axes (matplotlib's own documented behavior for
    an empty/degenerate one) -- _measure_overhang_px must skip that Axes rather than crash on
    ``None.x1``, falling back to whatever the other Axes (or the fixed guess) give it."""
    stub = _bare_stub((9.0, 6.0))
    stub.ax.plot([0, 1], [0, 1])
    twin = stub.ax.twinx()
    twin.plot([0, 1], [0, 1])
    stub.canvas.draw()

    monkeypatch.setattr(twin, "get_tightbbox", lambda renderer: None)
    overhang = stub._measure_overhang_px()  # must not raise
    assert isinstance(overhang, float)


# --------------------------------------------------------------------------------------------------
# PanRangeSlider: split value text (vertical) vs. stock text (horizontal)
# --------------------------------------------------------------------------------------------------
def test_vertical_split_text_shows_hi_above_and_lo_below_after_set_val():
    fig = Figure()
    ax = fig.add_subplot(111)
    slider = PanRangeSlider(ax, "", 0.0, 10.0, valinit=(2.0, 8.0), orientation="vertical")

    assert slider.valtext.get_visible() is False
    assert slider._hi_text.get_text() == "8"
    assert slider._lo_text.get_text() == "2"

    slider.set_val((3.0, 7.0))
    assert slider._hi_text.get_text() == "7"
    assert slider._lo_text.get_text() == "3"


def test_vertical_split_text_log_valfmt_shows_linear_values():
    valfmt = lambda v: f"{10.0 ** v:.3g}"  # noqa: E731 -- mirrors ui.py's own log valfmt
    lo, hi = to_log_bounds(1.0, 1000.0)
    fig = Figure()
    ax = fig.add_subplot(111)
    slider = PanRangeSlider(ax, "", lo, hi, valinit=(lo, hi), orientation="vertical", valfmt=valfmt)

    slider.set_val(to_log_bounds(10.0, 100.0))
    assert slider._hi_text.get_text() == "100"
    assert slider._lo_text.get_text() == "10"


def test_horizontal_slider_gets_split_left_right_text_and_hides_stock_valtext():
    """Regression: the stock single "(lo, hi)" valtext sits to the right of the WHOLE track, and
    can clip off the figure edge once ui.py widens the x slider's track to match the plot (see
    test_layout_sliders_x_slider_split_text_stays_inside_figure_no_twin) -- the horizontal slider
    now gets the same lo/hi split as vertical, anchored inward at each end instead."""
    fig = Figure()
    ax = fig.add_subplot(111)
    slider = PanRangeSlider(ax, "", 0.0, 10.0, valinit=(2.0, 8.0), orientation="horizontal")
    assert slider.valtext.get_visible() is False
    assert slider._lo_text.get_text() == "2"
    assert slider._hi_text.get_text() == "8"
    assert slider._lo_text.get_ha() == "left"
    assert slider._hi_text.get_ha() == "right"

    slider.set_val((3.0, 7.0))
    assert slider._lo_text.get_text() == "3"
    assert slider._hi_text.get_text() == "7"


def test_vertical_split_text_does_not_overlap_handle_at_full_extent():
    """Regression: the split texts used to sit only ~4px off the track (an Axes-fraction
    offset), closer than a thumb handle's own ~5-7px marker radius -- so a handle pinned right
    at the top/bottom (valmax/valmin) visually collided with the text there. The fixed
    points-based offset (sliders.TEXT_OFFSET_PT) must clear it regardless of the track's own
    pixel height."""
    fig = Figure(figsize=(2.0, 4.0), dpi=100.0)
    ax = fig.add_subplot(111)
    canvas = FigureCanvasAgg(fig)
    slider = PanRangeSlider(ax, "", 0.0, 10.0, valinit=(0.0, 10.0), orientation="vertical")
    canvas.draw()
    renderer = canvas.get_renderer()

    lo_handle, hi_handle = slider._handles  # index 0 = valmin (lo), 1 = valmax (hi)
    assert not slider._hi_text.get_window_extent(renderer).overlaps(
        hi_handle.get_window_extent(renderer))
    assert not slider._lo_text.get_window_extent(renderer).overlaps(
        lo_handle.get_window_extent(renderer))


# --------------------------------------------------------------------------------------------------
# Color: _make_range_slider paints the poly/handles/split text; None falls back to neutral gray.
# --------------------------------------------------------------------------------------------------
def _color_slider(color):
    stub = _make_app_stub()
    return stub._make_range_slider(
        rect=[0.90, 0.16, 0.02, 0.74], orientation="vertical",
        full_range=(0.0, 10.0), cur_range=(2.0, 8.0), scale="linear",
        apply=lambda lo, hi: None, store=lambda lo, hi: None, color=color)


def test_make_range_slider_colors_poly_and_split_text_when_given():
    slider = _color_slider("tab:red")
    assert mcolors.same_color(slider.poly.get_facecolor(), mcolors.to_rgba("tab:red"))
    assert mcolors.same_color(slider._hi_text.get_color(), mcolors.to_rgba("tab:red"))
    assert mcolors.same_color(slider._lo_text.get_color(), mcolors.to_rgba("tab:red"))


def test_make_range_slider_none_color_falls_back_to_neutral_gray():
    slider = _color_slider(None)
    assert mcolors.same_color(slider.poly.get_facecolor(), mcolors.to_rgba(_SLIDER_NEUTRAL_COLOR))


# --------------------------------------------------------------------------------------------------
# Renderers: ViewDefaults.y_color/y2_color for the two-twin steps color coding is most visible on.
# --------------------------------------------------------------------------------------------------
def test_render_overview_returns_rate_blue_y2_color_when_rate_present():
    td = make_testdata()
    st = injection_state(td)
    res = compute_all(st, td)
    fig = Figure()
    ax = fig.add_subplot(111)
    defaults = plots.render_overview(ax, td, st, res)
    assert defaults.y2_color == "tab:blue"


def test_render_gfunction_returns_black_and_red_colors():
    td = make_testdata()
    st = injection_state(td)
    res = compute_all(st, td)
    fig = Figure()
    ax = fig.add_subplot(111)
    defaults = plots.render_gfunction(ax, td, st, res)
    assert defaults.y_color == ("black" if res.pressure_is_bhp else "tab:red")
    assert defaults.y2_color == "tab:red"
