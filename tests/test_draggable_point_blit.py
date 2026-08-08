"""Blitting during DraggablePointController drags: motion restores + redraws only the managed
marker (and companion vline, when present) via Agg's copy_from_bbox/restore_region/blit instead
of a full canvas.draw_idle() per event; release leaves the canvas in a normal (non-animated)
state. Modeled on tests/test_anchor_line_blit.py's canvas-instrumentation approach."""

import numpy as np
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


def _curve():
    x = np.linspace(0.0, 20.0, 41)  # step 0.5
    y = np.sin(x)
    return x, y


def _build_axes(marker_x, marker_y, vline_x=None, canvas_cls=RecordingCanvas):
    fig = Figure(figsize=(6.4, 4.8), dpi=100)
    ax = fig.add_subplot(111)
    canvas = canvas_cls(fig)
    ax.set_xlim(0.0, 20.0)
    ax.set_ylim(-5.0, 5.0)
    ax.set_autoscale_on(False)
    ax.plot([marker_x], [marker_y], "o", gid="marker")
    if vline_x is not None:
        ax.axvline(vline_x, gid="closure_vline")
    canvas.draw()
    return fig, ax, canvas


def _event(name, canvas, ax, x, y, button=1):
    px, py = ax.transData.transform((x, y))
    return MouseEvent(name, canvas, px, py, button=button)


def _pressed(canvas_cls=RecordingCanvas, with_vline=False):
    x, y = _curve()
    vline_x = x[20] if with_vline else None
    fig, ax, canvas = _build_axes(x[10], y[10], vline_x=vline_x, canvas_cls=canvas_cls)
    got = []
    ctrl = picks.DraggablePointController(canvas, ax, "marker", x, y, commit_fn=got.append,
                                          vline_gid="closure_vline" if with_vline else None)
    ctrl._on_press(_event("button_press_event", canvas, ax, x[10], y[10]))
    assert ctrl._dragging is True
    return ctrl, ax, canvas, got, x, y


def test_press_sets_the_marker_animated():
    ctrl, ax, canvas, got, x, y = _pressed()
    assert ctrl._blit_artists is not None
    assert all(a.get_animated() for a in ctrl._blit_artists)


def test_press_with_vline_animates_both_artists():
    ctrl, ax, canvas, got, x, y = _pressed(with_vline=True)
    assert len(ctrl._blit_artists) == 2
    assert all(a.get_animated() for a in ctrl._blit_artists)


def test_press_paints_the_marker_immediately_via_restore_and_blit():
    # Without a first paint at press, the just-animated marker is skipped by the preceding
    # canvas.draw() and stays invisible until the first motion event.
    x, y = _curve()
    fig, ax, canvas = _build_axes(x[10], y[10])
    ctrl = picks.DraggablePointController(canvas, ax, "marker", x, y, commit_fn=lambda v: None)
    restore_before, blit_before = canvas.restore_region_calls, canvas.blit_calls

    ctrl._on_press(_event("button_press_event", canvas, ax, x[10], y[10]))

    assert ctrl._dragging is True
    assert canvas.restore_region_calls == restore_before + 1
    assert canvas.blit_calls == blit_before + 1


def test_motion_during_drag_blits_instead_of_draw_idle():
    ctrl, ax, canvas, got, x, y = _pressed()
    # Press already did one restore/blit (the first paint) -- see
    # test_press_paints_the_marker_immediately_via_restore_and_blit -- so check the delta.
    idle_before = canvas.draw_idle_calls
    restore_before, blit_before = canvas.restore_region_calls, canvas.blit_calls

    ctrl._on_motion(_event("motion_notify_event", canvas, ax, x[25], y[25]))

    assert canvas.draw_idle_calls == idle_before
    assert canvas.restore_region_calls == restore_before + 1
    assert canvas.blit_calls == blit_before + 1
    marker = ctrl._artist()
    assert marker.get_xdata()[0] == x[25]


def test_supports_blit_false_falls_back_to_draw_idle():
    ctrl, ax, canvas, got, x, y = _pressed(canvas_cls=NoBlitCanvas)
    assert ctrl._bg is None
    idle_before = canvas.draw_idle_calls

    ctrl._on_motion(_event("motion_notify_event", canvas, ax, x[25], y[25]))

    assert canvas.draw_idle_calls == idle_before + 1
    assert canvas.restore_region_calls == 0
    assert canvas.blit_calls == 0


def test_release_unanimates_artists_and_schedules_a_final_draw():
    ctrl, ax, canvas, got, x, y = _pressed()
    ctrl._on_motion(_event("motion_notify_event", canvas, ax, x[25], y[25]))
    artists = ctrl._blit_artists
    assert all(a.get_animated() for a in artists)

    idle_before = canvas.draw_idle_calls
    ctrl._on_release(_event("button_release_event", canvas, ax, x[25], y[25]))

    assert all(a.get_animated() is False for a in artists)
    assert ctrl._blit_artists is None
    assert ctrl._bg is None
    assert canvas.draw_idle_calls == idle_before + 1
    assert got == [float(x[25])]


def test_disconnect_mid_drag_unanimates_and_clears_state():
    ctrl, ax, canvas, got, x, y = _pressed(with_vline=True)
    ctrl._on_motion(_event("motion_notify_event", canvas, ax, x[25], y[25]))
    artists = ctrl._blit_artists
    assert all(a.get_animated() for a in artists)

    ctrl.disconnect()

    assert all(a.get_animated() is False for a in artists)
    assert ctrl._blit_artists is None
    assert ctrl._bg is None
    assert ctrl._dragging is False
    assert ctrl._cids == []
    assert ctrl.gate.try_claim(object())  # gate was released, so a fresh claim succeeds
    assert got == []  # disconnect mid-drag never commits
