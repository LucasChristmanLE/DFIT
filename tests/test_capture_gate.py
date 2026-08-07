"""``_CaptureGate`` arbitration between controllers whose hit zones overlap on one Axes -- the
scenario the eff-ISIP anchor sitting near the contact marker would otherwise double-capture."""

import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backend_bases import MouseEvent

from dfit_tool import picks


def _build_shared_marker_axes():
    """Two point markers at the exact same data location, on one Axes/canvas, so a single press
    pixel is a hit for both controllers' tolerance zones."""
    fig = Figure(figsize=(6.4, 4.8), dpi=100)
    ax = fig.add_subplot(111)
    canvas = FigureCanvasAgg(fig)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_autoscale_on(False)
    ax.plot([5.0], [5.0], "o", gid="pt_a")
    ax.plot([5.0], [5.0], "s", gid="pt_b")
    canvas.draw()
    return fig, ax, canvas


def _press_event(canvas, ax, x, y, button=1):
    px, py = ax.transData.transform((x, y))
    return MouseEvent("button_press_event", canvas, px, py, button=button)


def _release_event(canvas, ax, x, y, button=1):
    px, py = ax.transData.transform((x, y))
    return MouseEvent("button_release_event", canvas, px, py, button=button)


def test_first_registered_claims_second_no_ops():
    fig, ax, canvas = _build_shared_marker_axes()
    curve_x = np.linspace(0.0, 10.0, 11)
    curve_y = np.linspace(0.0, 10.0, 11)
    gate = picks._CaptureGate()
    got_a, got_b = [], []

    ctrl_a = picks.DraggablePointController(canvas, ax, "pt_a", curve_x, curve_y,
                                            commit_fn=got_a.append, gate=gate)
    ctrl_b = picks.DraggablePointController(canvas, ax, "pt_b", curve_x, curve_y,
                                            commit_fn=got_b.append, gate=gate)

    ev = _press_event(canvas, ax, 5.0, 5.0)
    ctrl_a._on_press(ev)
    assert ctrl_a._dragging is True
    assert gate._owner is ctrl_a

    ctrl_b._on_press(ev)
    assert ctrl_b._dragging is False  # gate already claimed by ctrl_a for this gesture

    rel = _release_event(canvas, ax, 5.0, 5.0)
    ctrl_a._on_release(rel)
    ctrl_b._on_release(rel)  # no-op: was never dragging

    assert got_a == [5.0]
    assert got_b == []


def test_gate_released_after_gesture_lets_the_other_controller_win_next_press():
    fig, ax, canvas = _build_shared_marker_axes()
    curve_x = np.linspace(0.0, 10.0, 11)
    curve_y = np.linspace(0.0, 10.0, 11)
    gate = picks._CaptureGate()
    got_a, got_b = [], []

    ctrl_a = picks.DraggablePointController(canvas, ax, "pt_a", curve_x, curve_y,
                                            commit_fn=got_a.append, gate=gate)
    ctrl_b = picks.DraggablePointController(canvas, ax, "pt_b", curve_x, curve_y,
                                            commit_fn=got_b.append, gate=gate)

    ev = _press_event(canvas, ax, 5.0, 5.0)
    ctrl_a._on_press(ev)
    ctrl_b._on_press(ev)
    rel = _release_event(canvas, ax, 5.0, 5.0)
    ctrl_a._on_release(rel)
    ctrl_b._on_release(rel)

    assert gate._owner is None  # freed for the next contest

    # Fresh press: ctrl_a is still registered/checked first and wins again, but the point is that
    # the gate itself did not stay latched onto ctrl_a -- try_claim is live for a fresh owner too.
    assert gate.try_claim(ctrl_b) is True
    gate.release()
    assert gate._owner is None


def test_drag_line_controller_and_draggable_point_controller_share_a_gate():
    """The G-function step's tail-trim line (DragLineController) shares a gate with the contact-
    point marker (DraggablePointController) -- same first-hit-wins contract as two
    DraggablePointControllers above."""
    fig, ax, canvas = _build_shared_marker_axes()
    ax.axvline(5.0, gid="tail_trim")
    canvas.draw()
    curve_x = np.linspace(0.0, 10.0, 11)
    curve_y = np.linspace(0.0, 10.0, 11)
    gate = picks._CaptureGate()
    got_point, got_line = [], []

    point_ctrl = picks.DraggablePointController(canvas, ax, "pt_a", curve_x, curve_y,
                                                commit_fn=got_point.append, gate=gate)
    line_ctrl = picks.DragLineController(canvas, ax, handlers={"tail_trim": got_line.append},
                                         gate=gate)

    ev = _press_event(canvas, ax, 5.0, 5.0)
    point_ctrl._on_press(ev)
    assert point_ctrl._dragging is True
    assert gate._owner is point_ctrl

    line_ctrl._on_press(ev)
    assert line_ctrl._active is None  # gate already claimed by point_ctrl -- no-op

    rel = _release_event(canvas, ax, 5.0, 5.0)
    point_ctrl._on_release(rel)
    line_ctrl._on_release(rel)  # no-op: was never active

    assert gate._owner is None  # freed for the next contest
    assert got_point == [5.0]
    assert got_line == []
