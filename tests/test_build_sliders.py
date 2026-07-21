"""Headless coverage for DfitApp._build_sliders / _make_range_slider / _twin_axes.

DfitApp itself needs a real tk.Tk() root (it's built in __init__), which this headless (Agg,
no display) suite can't construct. These tests instead exercise the methods directly against a
duck-typed stand-in exposing only what the methods touch (self.fig/self.ax/self.canvas), binding
the real DfitApp methods onto it via types.MethodType -- same headless-Agg-plus-real-widgets
approach as test_drag_controller.py, just without a full DfitApp instance backing it.
"""
from __future__ import annotations

import types

import pytest
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from dfit_tool.sliders import to_log_bounds
from dfit_tool.ui import DfitApp, ViewState


def _make_app_stub():
    stub = types.SimpleNamespace()
    stub.fig = Figure()
    stub.ax = stub.fig.add_subplot(111)
    stub.canvas = FigureCanvasAgg(stub.fig)
    stub._make_range_slider = types.MethodType(DfitApp._make_range_slider, stub)
    stub._build_sliders = types.MethodType(DfitApp._build_sliders, stub)
    stub._twin_axes = types.MethodType(DfitApp._twin_axes, stub)
    return stub


# --------------------------------------------------------------------------------------------------
# rects / y2-only-with-twin
# --------------------------------------------------------------------------------------------------
def test_x_and_y_slider_rects_no_twin():
    stub = _make_app_stub()
    view = ViewState(xlim=(2.0, 8.0), ylim=(1.0, 9.0), y2lim=None)
    stub._build_sliders(full_x=(0.0, 10.0), full_y=(0.0, 10.0), full_y2=None, view=view, twin=None)

    assert stub._x_slider.ax.get_position().bounds == pytest.approx((0.10, 0.04, 0.68, 0.03))
    assert stub._y_slider.ax.get_position().bounds == pytest.approx((0.87, 0.16, 0.02, 0.74))
    assert stub._y2_slider is None


def test_y2_slider_only_built_when_twin_present_with_correct_rect():
    stub = _make_app_stub()
    twin = stub.ax.twinx()
    twin.set_ylim(0.0, 100.0)
    view = ViewState(xlim=(0.0, 10.0), ylim=(0.0, 10.0), y2lim=(10.0, 90.0))
    stub._build_sliders(full_x=(0.0, 10.0), full_y=(0.0, 10.0), full_y2=(0.0, 100.0),
                        view=view, twin=twin)

    assert stub._y2_slider is not None
    assert stub._y2_slider.ax.get_position().bounds == pytest.approx((0.93, 0.16, 0.02, 0.74))


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
