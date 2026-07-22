import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backend_bases import MouseEvent

from dfit_tool import picks


def _build_axes(marker_x=None, marker_y=None, gid="marker"):
    fig = Figure(figsize=(6.4, 4.8), dpi=100)
    ax = fig.add_subplot(111)
    canvas = FigureCanvasAgg(fig)
    ax.set_xlim(0.0, 20.0)
    ax.set_ylim(-5.0, 5.0)
    ax.set_autoscale_on(False)
    if marker_x is not None:
        ax.plot([marker_x], [marker_y], "o", gid=gid)
    canvas.draw()
    return fig, ax, canvas


def _event(name, canvas, ax, x, y, button=1):
    px, py = ax.transData.transform((x, y))
    return MouseEvent(name, canvas, px, py, button=button)


def _curve():
    x = np.linspace(0.0, 20.0, 41)  # step 0.5
    y = np.sin(x)
    return x, y


def test_press_near_marker_captures_and_motion_snaps_to_nearest_sample():
    x, y = _curve()
    fig, ax, canvas = _build_axes(marker_x=x[10], marker_y=y[10])
    got = []
    ctrl = picks.DraggablePointController(canvas, ax, "marker", x, y, commit_fn=got.append)

    ctrl._on_press(_event("button_press_event", canvas, ax, x[10], y[10]))
    assert ctrl._dragging is True

    ctrl._on_motion(_event("motion_notify_event", canvas, ax, x[25] + 0.01, y[25] + 0.3))
    marker = ctrl._artist()
    assert marker.get_xdata()[0] == x[25]
    assert marker.get_ydata()[0] == y[25]

    ctrl._on_release(_event("button_release_event", canvas, ax, x[25], y[25]))
    assert ctrl._dragging is False
    assert got == [float(x[25])]


def test_press_far_from_marker_does_not_capture():
    x, y = _curve()
    fig, ax, canvas = _build_axes(marker_x=x[10], marker_y=y[10])
    got = []
    ctrl = picks.DraggablePointController(canvas, ax, "marker", x, y, commit_fn=got.append)

    ctrl._on_press(_event("button_press_event", canvas, ax, x[30], y[30]))
    assert ctrl._dragging is False

    ctrl._on_release(_event("button_release_event", canvas, ax, x[30], y[30]))
    assert got == []


def test_missing_marker_artist_is_a_no_op():
    x, y = _curve()
    fig, ax, canvas = _build_axes(marker_x=None)  # never drawn
    got = []
    ctrl = picks.DraggablePointController(canvas, ax, "marker", x, y, commit_fn=got.append)

    ctrl._on_press(_event("button_press_event", canvas, ax, x[10], y[10]))
    assert ctrl._dragging is False
    ctrl._on_release(_event("button_release_event", canvas, ax, x[10], y[10]))
    assert got == []


def test_commit_called_exactly_once_on_release():
    x, y = _curve()
    fig, ax, canvas = _build_axes(marker_x=x[10], marker_y=y[10])
    got = []
    ctrl = picks.DraggablePointController(canvas, ax, "marker", x, y, commit_fn=got.append)

    ctrl._on_press(_event("button_press_event", canvas, ax, x[10], y[10]))
    ctrl._on_motion(_event("motion_notify_event", canvas, ax, x[12], y[12]))
    ctrl._on_motion(_event("motion_notify_event", canvas, ax, x[15], y[15]))
    ctrl._on_release(_event("button_release_event", canvas, ax, x[15], y[15]))

    assert got == [float(x[15])]
