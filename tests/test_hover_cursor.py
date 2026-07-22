"""Hover-cursor feedback: HoverCursorController probes an ordered list of controllers'
hover_kind()/active_kind() and sets the canvas cursor via matplotlib's backend-agnostic
Cursors/set_cursor API (a no-op on FigureCanvasBase, real on FigureCanvasTkAgg)."""

import numpy as np
from matplotlib.backend_tools import Cursors
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backend_bases import MouseEvent
from matplotlib.figure import Figure

from dfit_tool import picks
from dfit_tool.model import TangentPick

GIDS = {"segment": "segment", "tick": "tick", "extension": "extension"}


class RecordingCanvas(FigureCanvasAgg):
    """Records every set_cursor call instead of touching any real GUI cursor."""

    def __init__(self, fig):
        super().__init__(fig)
        self.cursor_calls = []

    def set_cursor(self, cursor):
        self.cursor_calls.append(cursor)


def _event(name, canvas, ax, x, y, button=1):
    px, py = ax.transData.transform((x, y))
    return MouseEvent(name, canvas, px, py, button=button)


def _build_tangent_axes():
    """A steep tangent construction -- anchor (20, 20), slope 8, over an axes with an exact
    480x480 px bbox (scale_x=9.6, scale_y=4.0 px/data-unit) -- tall enough on screen that the
    tick, the segment ends, and a mid-body point are all cleanly separable by tol_px (12)."""
    fig = Figure(figsize=(6.0, 6.0), dpi=100)
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)
    ax = fig.add_subplot(111)
    canvas = RecordingCanvas(fig)
    ax.set_xlim(0.0, 50.0)
    ax.set_ylim(-40.0, 80.0)
    ax.set_autoscale_on(False)

    anchor_x, anchor_y, slope, half = 20.0, 20.0, 8.0, 5.0
    x0, x1 = anchor_x - half, anchor_x + half
    y0, y1 = anchor_y + slope * (x0 - anchor_x), anchor_y + slope * (x1 - anchor_x)
    ax.plot([x0, x1], [y0, y1], gid="segment")
    ax.plot([anchor_x, anchor_x], [anchor_y - 8.0, anchor_y + 8.0], gid="tick")
    canvas.draw()
    return fig, ax, canvas, (anchor_x, anchor_y, slope)


def _anchor_ctrl(ax, canvas, anchor):
    anchor_x, anchor_y, slope = anchor
    pick = TangentPick(anchor_x=anchor_x, anchor_y=anchor_y, slope=slope)
    x_arr = np.linspace(-5.0, 45.0, 101)
    y_arr = anchor_y + slope * (x_arr - anchor_x)
    return picks.AnchorLineController(canvas, ax, GIDS, get_pick=lambda: pick,
                                      commit_fn=lambda *a: None, curve=(x_arr, y_arr))


# --------------------------------------------------------------------------------------------------
# (a)-(d): the three AnchorLineController zones + empty space, over one construction
# --------------------------------------------------------------------------------------------------
def test_hover_over_tick_shows_resize_horizontal():
    fig, ax, canvas, anchor = _build_tangent_axes()
    ctrl = _anchor_ctrl(ax, canvas, anchor)
    hover = picks.HoverCursorController(canvas, [ctrl])

    hover._on_motion(_event("motion_notify_event", canvas, ax, 20.0, 27.5))  # on the tick's tip
    assert canvas.cursor_calls[-1] == Cursors.RESIZE_HORIZONTAL


def test_hover_over_segment_end_shows_hand():
    fig, ax, canvas, anchor = _build_tangent_axes()
    ctrl = _anchor_ctrl(ax, canvas, anchor)
    hover = picks.HoverCursorController(canvas, [ctrl])

    hover._on_motion(_event("motion_notify_event", canvas, ax, 25.0, 60.0))  # the far segment end
    assert canvas.cursor_calls[-1] == Cursors.HAND


def test_hover_over_segment_body_away_from_tick_and_ends_shows_move():
    fig, ax, canvas, anchor = _build_tangent_axes()
    ctrl = _anchor_ctrl(ax, canvas, anchor)
    hover = picks.HoverCursorController(canvas, [ctrl])

    hover._on_motion(_event("motion_notify_event", canvas, ax, 22.0, 36.0))  # mid-body
    assert canvas.cursor_calls[-1] == Cursors.MOVE


def test_hover_over_empty_space_shows_pointer():
    fig, ax, canvas, anchor = _build_tangent_axes()
    ctrl = _anchor_ctrl(ax, canvas, anchor)
    hover = picks.HoverCursorController(canvas, [ctrl])

    hover._on_motion(_event("motion_notify_event", canvas, ax, 5.0, 5.0))  # nowhere near anything
    assert canvas.cursor_calls[-1] == Cursors.POINTER


# --------------------------------------------------------------------------------------------------
# (e): an active drag holds its cursor even over empty space
# --------------------------------------------------------------------------------------------------
def test_active_body_drag_holds_move_cursor_over_empty_space():
    fig, ax, canvas, anchor = _build_tangent_axes()
    ctrl = _anchor_ctrl(ax, canvas, anchor)
    hover = picks.HoverCursorController(canvas, [ctrl])

    ctrl._on_press(_event("button_press_event", canvas, ax, 22.0, 36.0))  # captures "body"
    assert ctrl._active == "body"

    hover._on_motion(_event("motion_notify_event", canvas, ax, 5.0, 5.0))  # empty space mid-drag
    assert canvas.cursor_calls[-1] == Cursors.MOVE


# --------------------------------------------------------------------------------------------------
# (f): a DragLineController vertical line
# --------------------------------------------------------------------------------------------------
def test_hover_over_drag_line_shows_resize_horizontal():
    fig = Figure(figsize=(6.4, 4.8), dpi=100)
    ax = fig.add_subplot(111)
    canvas = RecordingCanvas(fig)
    ax.set_xlim(0.0, 20.0)
    ax.set_ylim(-5.0, 5.0)
    ax.set_autoscale_on(False)
    ax.axvline(10.0, gid="start")
    canvas.draw()

    ctrl = picks.DragLineController(canvas, ax, handlers={"start": lambda x: None})
    hover = picks.HoverCursorController(canvas, [ctrl])

    hover._on_motion(_event("motion_notify_event", canvas, ax, 10.0, 0.0))
    assert canvas.cursor_calls[-1] == Cursors.RESIZE_HORIZONTAL


# --------------------------------------------------------------------------------------------------
# no churn: unchanged cursor across consecutive motions is set only once
# --------------------------------------------------------------------------------------------------
def test_set_cursor_not_recalled_when_cursor_is_unchanged():
    fig, ax, canvas, anchor = _build_tangent_axes()
    ctrl = _anchor_ctrl(ax, canvas, anchor)
    hover = picks.HoverCursorController(canvas, [ctrl])

    hover._on_motion(_event("motion_notify_event", canvas, ax, 22.0, 36.0))  # body -> MOVE
    assert len(canvas.cursor_calls) == 1
    hover._on_motion(_event("motion_notify_event", canvas, ax, 22.1, 36.8))  # still on the body
    assert len(canvas.cursor_calls) == 1  # no second set_cursor call -- cursor didn't change

    hover._on_motion(_event("motion_notify_event", canvas, ax, 5.0, 5.0))  # empty space -> POINTER
    assert len(canvas.cursor_calls) == 2  # a real change: set_cursor called again


def test_disconnect_unbinds_motion_callback():
    fig, ax, canvas, anchor = _build_tangent_axes()
    ctrl = _anchor_ctrl(ax, canvas, anchor)
    hover = picks.HoverCursorController(canvas, [ctrl])
    assert hover._cids
    hover.disconnect()
    assert hover._cids == []
