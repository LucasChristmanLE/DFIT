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
