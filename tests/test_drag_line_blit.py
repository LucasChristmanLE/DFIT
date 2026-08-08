"""Blitting during DragLineController drags: motion restores + redraws only the active line via
Agg's copy_from_bbox/restore_region/blit instead of a full canvas.draw_idle() per event; release
leaves the canvas in a normal (non-animated) state. Modeled on
tests/test_anchor_line_blit.py's canvas-instrumentation approach."""

import pytest
from matplotlib.backend_bases import MouseEvent
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from dfit_tool import picks


class RecordingCanvas(FigureCanvasAgg):
    """Real Agg rendering (so copy_from_bbox/restore_region/blit actually work), with call
    counters -- same pattern as tests/test_anchor_line_blit.py's RecordingCanvas."""

    def __init__(self, fig):
        super().__init__(fig)
        self.draw_idle_calls = 0
        self.restore_region_calls = 0
        self.blit_calls = 0

    def draw_idle(self, *a, **kw):
        self.draw_idle_calls += 1
        return super().draw_idle(*a, **kw)

    def restore_region(self, *a, **kw):
        self.restore_region_calls += 1
        return super().restore_region(*a, **kw)

    def blit(self, *a, **kw):
        self.blit_calls += 1
        return super().blit(*a, **kw)


class NoBlitCanvas(RecordingCanvas):
    supports_blit = False


def _build_axes(canvas_cls=RecordingCanvas):
    fig = Figure(figsize=(6.4, 4.8), dpi=100)
    ax = fig.add_subplot(111)
    canvas = canvas_cls(fig)
    ax.set_xlim(0.0, 20.0)
    ax.set_ylim(-5.0, 5.0)
    ax.set_autoscale_on(False)
    ax.axvline(10.0, gid="start")
    canvas.draw()
    return fig, ax, canvas


def _event(name, canvas, ax, xdata, ydata=0.0, button=1):
    px, py = ax.transData.transform((xdata, ydata))
    return MouseEvent(name, canvas, px, py, button=button)


def _line(ax, gid):
    return next(l for l in ax.get_lines() if l.get_gid() == gid)


def _pressed(canvas_cls=RecordingCanvas):
    fig, ax, canvas = _build_axes(canvas_cls=canvas_cls)
    got = {}
    ctrl = picks.DragLineController(canvas, ax, handlers={"start": lambda xd: got.setdefault("start", xd)})
    ctrl._on_press(_event("button_press_event", canvas, ax, 10.0))
    assert ctrl._active is not None
    return ctrl, ax, canvas, got


def test_press_sets_the_active_line_animated():
    ctrl, ax, canvas, got = _pressed()
    assert ctrl._active.get_animated() is True


def test_press_paints_the_line_immediately_via_restore_and_blit():
    # Without a first paint at press, the just-animated line is skipped by the preceding
    # canvas.draw() and stays invisible until the first motion event.
    fig, ax, canvas = _build_axes()
    ctrl = picks.DragLineController(canvas, ax, handlers={"start": lambda xd: None})
    restore_before, blit_before = canvas.restore_region_calls, canvas.blit_calls

    ctrl._on_press(_event("button_press_event", canvas, ax, 10.0))

    assert ctrl._active is not None
    assert canvas.restore_region_calls == restore_before + 1
    assert canvas.blit_calls == blit_before + 1


def test_motion_during_drag_blits_instead_of_draw_idle():
    ctrl, ax, canvas, got = _pressed()
    # Press already did one restore/blit (the first paint) -- see
    # test_press_paints_the_line_immediately_via_restore_and_blit -- so check the delta.
    idle_before = canvas.draw_idle_calls
    restore_before, blit_before = canvas.restore_region_calls, canvas.blit_calls

    ctrl._on_motion(_event("motion_notify_event", canvas, ax, 12.0))

    assert canvas.draw_idle_calls == idle_before
    assert canvas.restore_region_calls == restore_before + 1
    assert canvas.blit_calls == blit_before + 1
    xs = ctrl._active.get_xdata()
    assert xs[0] == pytest.approx(12.0, abs=0.05)
    assert xs[1] == pytest.approx(12.0, abs=0.05)


def test_supports_blit_false_falls_back_to_draw_idle():
    ctrl, ax, canvas, got = _pressed(canvas_cls=NoBlitCanvas)
    assert ctrl._bg is None
    idle_before = canvas.draw_idle_calls

    ctrl._on_motion(_event("motion_notify_event", canvas, ax, 12.0))

    assert canvas.draw_idle_calls == idle_before + 1
    assert canvas.restore_region_calls == 0
    assert canvas.blit_calls == 0


def test_release_unanimates_the_line_and_schedules_a_final_draw():
    ctrl, ax, canvas, got = _pressed()
    ctrl._on_motion(_event("motion_notify_event", canvas, ax, 12.0))
    line = ctrl._active
    assert line.get_animated() is True

    idle_before = canvas.draw_idle_calls
    ctrl._on_release(_event("button_release_event", canvas, ax, 12.0))

    assert line.get_animated() is False
    assert canvas.draw_idle_calls == idle_before + 1
    assert ctrl._bg is None
    assert got["start"] == pytest.approx(12.0, abs=0.05)


def test_disconnect_mid_drag_unanimates_and_clears_state():
    ctrl, ax, canvas, got = _pressed()
    ctrl._on_motion(_event("motion_notify_event", canvas, ax, 12.0))
    line = ctrl._active
    assert line.get_animated() is True

    ctrl.disconnect()

    assert line.get_animated() is False
    assert ctrl._active is None
    assert ctrl._bg is None
    assert ctrl._cids == []
    assert ctrl.gate.try_claim(object())  # gate was released, so a fresh claim succeeds
    assert "start" not in got  # disconnect mid-drag never commits
