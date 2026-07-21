import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backend_bases import MouseEvent
from dfit_tool.model import compute_all
from dfit_tool import plots, picks
from tests.helpers import make_testdata, overview_state


def _built_overview():
    td = make_testdata(); st = overview_state(td); res = compute_all(st, td)
    fig = Figure(); ax = fig.add_subplot(111)
    canvas = FigureCanvasAgg(fig)
    plots.render_overview(ax, td, st, res)
    canvas.draw()  # realize transforms / bboxes
    return td, st, ax, canvas


def _line(ax, gid):
    return next(l for l in ax.get_lines() if l.get_gid() == gid)


def _pixel_of(ax, xdata):
    ymid = float(np.mean(ax.get_ylim()))
    return ax.transData.transform((xdata, ymid))


def _event(name, canvas, ax, xdata, button=1):
    px, py = _pixel_of(ax, xdata)
    return MouseEvent(name, canvas, px, py, button=button)


def test_overview_rate_twin_owns_inaxes_regression():
    # The rate twinx overlays the primary axis; a real event resolves inaxes to the twin.
    # This guards the bug where the controller checked event.inaxes is self.ax and never captured.
    td, st, ax, canvas = _built_overview()
    twins = [a for a in canvas.figure.axes if a is not ax]
    assert twins, "expected a twinx rate axis on the overview"
    ev = _event("button_press_event", canvas, ax, _line(ax, "start").get_xdata()[0])
    assert ev.inaxes is not ax  # matplotlib assigns the topmost (twin) axis


def test_press_captures_and_release_commits_over_twin():
    td, st, ax, canvas = _built_overview()
    got = {}
    ctrl = picks.DragLineController(
        canvas, ax, handlers={"start": lambda xd: got.__setitem__("start", xd),
                              "shutin": lambda xd: got.__setitem__("shutin", xd)})
    start_line = _line(ax, "start")
    x0 = start_line.get_xdata()[0]
    ctrl._on_press(_event("button_press_event", canvas, ax, x0))
    assert ctrl._active is start_line
    target_x = float(np.mean(ax.get_xlim()))
    ctrl._on_motion(_event("motion_notify_event", canvas, ax, target_x))
    assert abs(start_line.get_xdata()[0] - target_x) < 1e-6
    ctrl._on_release(_event("button_release_event", canvas, ax, target_x))
    assert ctrl._active is None
    assert abs(got["start"] - target_x) < 1e-6


def test_guard_blocks_capture():
    td, st, ax, canvas = _built_overview()
    ctrl = picks.DragLineController(canvas, ax, handlers={"start": lambda xd: None},
                                    guard=lambda: True)
    x0 = _line(ax, "start").get_xdata()[0]
    ctrl._on_press(_event("button_press_event", canvas, ax, x0))
    assert ctrl._active is None


def test_press_far_from_any_line_captures_nothing():
    td, st, ax, canvas = _built_overview()
    ctrl = picks.DragLineController(canvas, ax, handlers={"start": lambda xd: None,
                                                          "shutin": lambda xd: None})
    sx = _pixel_of(ax, _line(ax, "start").get_xdata()[0])[0]
    hx = _pixel_of(ax, _line(ax, "shutin").get_xdata()[0])[0]
    far_px = min(max(sx, hx) + 40.0, ax.bbox.x1 - 1.0)
    py = (ax.bbox.y0 + ax.bbox.y1) / 2.0
    ctrl._on_press(MouseEvent("button_press_event", canvas, far_px, py, button=1))
    assert ctrl._active is None


def test_disconnect_unbinds_all_callbacks():
    td, st, ax, canvas = _built_overview()
    ctrl = picks.DragLineController(canvas, ax, handlers={"start": lambda xd: None})
    assert ctrl._cids
    ctrl.disconnect()
    assert ctrl._cids == []
