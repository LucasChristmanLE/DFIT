import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backend_bases import MouseEvent
from dfit_tool.model import compute_all
from dfit_tool import plots, picks
from tests.helpers import make_testdata, injection_state


def _built_injection():
    td = make_testdata(); st = injection_state(td); res = compute_all(st, td)
    fig = Figure(); ax = fig.add_subplot(111)
    canvas = FigureCanvasAgg(fig)
    plots.render_injection(ax, td, st, res)
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


def test_injection_rate_twin_owns_inaxes_regression():
    # The rate twinx overlays the primary axis; a real event resolves inaxes to the twin.
    # This guards the bug where the controller checked event.inaxes is self.ax and never captured.
    td, st, ax, canvas = _built_injection()
    twins = [a for a in canvas.figure.axes if a is not ax]
    assert twins, "expected a twinx rate axis on the injection"
    ev = _event("button_press_event", canvas, ax, _line(ax, "start").get_xdata()[0])
    assert ev.inaxes is not ax  # matplotlib assigns the topmost (twin) axis


def test_press_captures_and_release_commits_over_twin():
    td, st, ax, canvas = _built_injection()
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
    td, st, ax, canvas = _built_injection()
    ctrl = picks.DragLineController(canvas, ax, handlers={"start": lambda xd: None},
                                    guard=lambda: True)
    x0 = _line(ax, "start").get_xdata()[0]
    ctrl._on_press(_event("button_press_event", canvas, ax, x0))
    assert ctrl._active is None


def test_press_far_from_any_line_captures_nothing():
    td, st, ax, canvas = _built_injection()
    ctrl = picks.DragLineController(canvas, ax, handlers={"start": lambda xd: None,
                                                          "shutin": lambda xd: None})
    sx = _pixel_of(ax, _line(ax, "start").get_xdata()[0])[0]
    hx = _pixel_of(ax, _line(ax, "shutin").get_xdata()[0])[0]
    far_px = min(max(sx, hx) + 40.0, ax.bbox.x1 - 1.0)
    py = (ax.bbox.y0 + ax.bbox.y1) / 2.0
    ctrl._on_press(MouseEvent("button_press_event", canvas, far_px, py, button=1))
    assert ctrl._active is None


def test_disconnect_unbinds_all_callbacks():
    td, st, ax, canvas = _built_injection()
    ctrl = picks.DragLineController(canvas, ax, handlers={"start": lambda xd: None})
    assert ctrl._cids
    ctrl.disconnect()
    assert ctrl._cids == []


def test_default_gate_is_private_per_instance():
    td, st, ax, canvas = _built_injection()
    ctrl1 = picks.DragLineController(canvas, ax, handlers={"start": lambda xd: None})
    ctrl2 = picks.DragLineController(canvas, ax, handlers={"start": lambda xd: None})
    assert ctrl1.gate is not ctrl2.gate


def test_press_denied_when_shared_gate_pre_claimed():
    td, st, ax, canvas = _built_injection()
    gate = picks._CaptureGate()
    other_owner = object()
    gate.try_claim(other_owner)  # simulate a sibling controller already holding the gate
    ctrl = picks.DragLineController(canvas, ax, handlers={"start": lambda xd: None}, gate=gate)
    x0 = _line(ax, "start").get_xdata()[0]
    ctrl._on_press(_event("button_press_event", canvas, ax, x0))
    assert ctrl._active is None  # hit-test found the line, but the gate denied the claim


def test_release_frees_the_gate():
    td, st, ax, canvas = _built_injection()
    gate = picks._CaptureGate()
    ctrl = picks.DragLineController(canvas, ax, handlers={"start": lambda xd: None}, gate=gate)
    x0 = _line(ax, "start").get_xdata()[0]
    ctrl._on_press(_event("button_press_event", canvas, ax, x0))
    assert gate._owner is ctrl
    ctrl._on_release(_event("button_release_event", canvas, ax, x0))
    assert gate._owner is None
