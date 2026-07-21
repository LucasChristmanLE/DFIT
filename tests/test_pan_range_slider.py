import pytest
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backend_bases import MouseEvent

from dfit_tool.sliders import PanRangeSlider, to_log_bounds, from_log_bounds


def _built(valmin=0.0, valmax=10.0, valinit=(2.0, 8.0), orientation="horizontal"):
    fig = Figure()
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    canvas = FigureCanvasAgg(fig)
    slider = PanRangeSlider(ax, "", valmin, valmax, valinit=valinit, orientation=orientation)
    canvas.draw()  # realize transforms/bboxes
    return canvas, slider


def _px_for_value(slider, value):
    """Pixel coords for a press at ``value`` along the slider's active axis, centered on the
    other axis so the point falls inside the slider's Axes bbox."""
    ax = slider.ax
    if slider.orientation == "horizontal":
        x = ax.transData.transform((value, 0.0))[0]
        y = (ax.bbox.y0 + ax.bbox.y1) / 2.0
    else:
        y = ax.transData.transform((0.0, value))[1]
        x = (ax.bbox.x0 + ax.bbox.x1) / 2.0
    return x, y


def _event(name, canvas, px, py, button=1):
    return MouseEvent(name, canvas, px, py, button=button)


def _press_move_release(slider, canvas, values):
    """Feed a press at ``values[0]`` then motions/releases at the rest through the slider's own
    ``_update`` (the method SliderBase wires to press/motion/release events)."""
    px, py = _px_for_value(slider, values[0])
    slider._update(_event("button_press_event", canvas, px, py))
    for v in values[1:]:
        px, py = _px_for_value(slider, v)
        slider._update(_event("motion_notify_event", canvas, px, py))


# --------------------------------------------------------------------------------------------------
# thumb presses defer to stock RangeSlider behavior
# --------------------------------------------------------------------------------------------------
def test_press_on_thumb_moves_only_that_thumb():
    canvas, slider = _built()
    _press_move_release(slider, canvas, [2.0, 4.0])
    assert slider._pan_active is False
    assert slider.drag_active is True
    assert slider.val[0] == pytest.approx(4.0, abs=0.05)
    assert slider.val[1] == pytest.approx(8.0)  # untouched thumb unaffected


def test_press_near_thumb_within_tolerance_defers_to_stock():
    canvas, slider = _built()
    lo_px, _ = slider._thumb_pixels()
    x, y = lo_px + slider.THUMB_TOL_PX - 2.0, _px_for_value(slider, 2.0)[1]
    slider._update(_event("button_press_event", canvas, x, y))
    assert slider._pan_active is False
    assert slider.drag_active is True


# --------------------------------------------------------------------------------------------------
# bar-drag panning
# --------------------------------------------------------------------------------------------------
def test_press_mid_bar_and_drag_pans_both_thumbs_preserving_width():
    canvas, slider = _built()
    _press_move_release(slider, canvas, [5.0, 6.0])
    assert slider._pan_active is True
    assert slider.val[0] == pytest.approx(3.0, abs=0.05)
    assert slider.val[1] == pytest.approx(9.0, abs=0.05)


def test_pan_release_ends_pan_mode():
    canvas, slider = _built()
    _press_move_release(slider, canvas, [5.0, 6.0])
    px, py = _px_for_value(slider, 6.0)
    slider._update(_event("button_release_event", canvas, px, py))
    assert slider._pan_active is False


def test_pan_clamps_at_valmax_without_shrinking_window():
    canvas, slider = _built()
    width = slider.val[1] - slider.val[0]
    _press_move_release(slider, canvas, [5.0, 50.0])  # way past valmax=10
    assert slider.val[1] == pytest.approx(10.0)
    assert slider.val[1] - slider.val[0] == pytest.approx(width)


def test_pan_clamps_at_valmin_without_shrinking_window():
    canvas, slider = _built()
    width = slider.val[1] - slider.val[0]
    _press_move_release(slider, canvas, [5.0, -50.0])  # way past valmin=0
    assert slider.val[0] == pytest.approx(0.0)
    assert slider.val[1] - slider.val[0] == pytest.approx(width)


def test_vertical_orientation_pans_too():
    canvas, slider = _built(orientation="vertical")
    _press_move_release(slider, canvas, [5.0, 6.0])
    assert slider._pan_active is True
    assert slider.val[0] == pytest.approx(3.0, abs=0.05)
    assert slider.val[1] == pytest.approx(9.0, abs=0.05)


# --------------------------------------------------------------------------------------------------
# log-space helper (pure, used by ui.py for the loglog step)
# --------------------------------------------------------------------------------------------------
def test_log_bounds_round_trip():
    lo, hi = to_log_bounds(1.0, 1000.0)
    assert (lo, hi) == pytest.approx((0.0, 3.0))
    assert from_log_bounds(lo, hi) == pytest.approx((1.0, 1000.0))


def test_log_bounds_clamps_nonpositive_floor():
    lo, hi = to_log_bounds(-5.0, 0.0)
    assert lo == pytest.approx(-12.0)  # log10(1e-12)
    assert hi == pytest.approx(-12.0)
