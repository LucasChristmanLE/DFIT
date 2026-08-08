"""Blitting during ModifierSpanController drags: motion restores + redraws only the span patch
via Agg's copy_from_bbox/restore_region/blit instead of a full canvas.draw_idle() per event; the
patch is a single persistent artist whose geometry updates rather than being recreated each
motion; release/disconnect leave no animated artists and no patch. Modeled on
tests/test_anchor_line_blit.py's canvas-instrumentation approach."""

import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backend_bases import MouseEvent
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


def _built_axes(canvas_cls=RecordingCanvas):
    fig = Figure(figsize=(6.4, 4.8), dpi=100)
    ax = fig.add_subplot(111)
    canvas = canvas_cls(fig)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_autoscale_on(False)
    canvas.draw()
    return fig, ax, canvas


def _event(name, canvas, ax, xdata, ydata=5.0, button=1, key=None):
    px, py = ax.transData.transform((xdata, ydata))
    return MouseEvent(name, canvas, px, py, button=button, key=key)


def _pressed(canvas_cls=RecordingCanvas):
    fig, ax, canvas = _built_axes(canvas_cls=canvas_cls)
    got = []
    ctrl = picks.ModifierSpanController(canvas, ax, lambda lo, hi: got.append((lo, hi)))
    ctrl._on_press(_event("button_press_event", canvas, ax, 7.0, key="shift"))
    assert ctrl._press_x is not None
    return ctrl, ax, canvas, got


def test_press_creates_the_patch_once_with_zero_width():
    ctrl, ax, canvas, got = _pressed()
    assert ctrl._patch is not None
    assert ctrl._patch.get_width() == 0.0


def test_press_paints_the_patch_immediately_via_restore_and_blit():
    # Without a first paint at press, the just-animated (zero-width, so invisible either way)
    # patch is skipped by the preceding canvas.draw(); asserted here uniformly with the other
    # three controllers.
    fig, ax, canvas = _built_axes()
    ctrl = picks.ModifierSpanController(canvas, ax, lambda lo, hi: None)
    restore_before, blit_before = canvas.restore_region_calls, canvas.blit_calls

    ctrl._on_press(_event("button_press_event", canvas, ax, 7.0, key="shift"))

    assert ctrl._press_x is not None
    assert canvas.restore_region_calls == restore_before + 1
    assert canvas.blit_calls == blit_before + 1


def test_motion_during_drag_blits_instead_of_draw_idle():
    ctrl, ax, canvas, got = _pressed()
    # Press already did one restore/blit (the first paint) -- see
    # test_press_paints_the_patch_immediately_via_restore_and_blit -- so check the delta.
    idle_before = canvas.draw_idle_calls
    restore_before, blit_before = canvas.restore_region_calls, canvas.blit_calls

    ctrl._on_motion(_event("motion_notify_event", canvas, ax, 3.0, key="shift"))

    assert canvas.draw_idle_calls == idle_before
    assert canvas.restore_region_calls == restore_before + 1
    assert canvas.blit_calls == blit_before + 1


def test_patch_identity_persists_across_motion_events_geometry_updates():
    ctrl, ax, canvas, got = _pressed()
    patch0 = ctrl._patch

    ctrl._on_motion(_event("motion_notify_event", canvas, ax, 3.0, key="shift"))
    patch1 = ctrl._patch
    ctrl._on_motion(_event("motion_notify_event", canvas, ax, 4.0, key="shift"))
    patch2 = ctrl._patch

    assert patch0 is patch1 is patch2  # same object throughout the gesture, geometry just moves
    assert patch2.get_x() == pytest.approx(4.0, abs=0.05)
    assert patch2.get_width() == pytest.approx(3.0, abs=0.05)  # sorted (4.0, 7.0)


def test_supports_blit_false_falls_back_to_draw_idle():
    ctrl, ax, canvas, got = _pressed(canvas_cls=NoBlitCanvas)
    assert ctrl._bg is None
    idle_before = canvas.draw_idle_calls

    ctrl._on_motion(_event("motion_notify_event", canvas, ax, 3.0, key="shift"))

    assert canvas.draw_idle_calls == idle_before + 1
    assert canvas.restore_region_calls == 0
    assert canvas.blit_calls == 0


def test_release_removes_patch_and_clears_animated_state():
    ctrl, ax, canvas, got = _pressed()
    ctrl._on_motion(_event("motion_notify_event", canvas, ax, 3.0, key="shift"))
    patch = ctrl._patch
    assert patch.get_animated() is True

    ctrl._on_release(_event("button_release_event", canvas, ax, 3.0, key="shift"))

    assert ctrl._patch is None
    assert ctrl._bg is None
    assert patch not in ax.patches
    assert len(got) == 1
    lo, hi = got[0]
    assert lo == pytest.approx(3.0, abs=0.05)
    assert hi == pytest.approx(7.0, abs=0.05)


def test_motion_with_press_x_set_but_no_patch_does_not_raise():
    # Defensive guard for the state _on_release/disconnect never actually produce (both clear
    # _press_x alongside _patch), but which a stray motion event should survive anyway.
    ctrl, ax, canvas, got = _pressed()
    ctrl._patch = None

    ctrl._on_motion(_event("motion_notify_event", canvas, ax, 3.0, key="shift"))  # no raise


def test_disconnect_mid_drag_removes_patch_and_clears_bg():
    ctrl, ax, canvas, got = _pressed()
    ctrl._on_motion(_event("motion_notify_event", canvas, ax, 3.0, key="shift"))
    assert ctrl._patch is not None

    ctrl.disconnect()

    assert ctrl._patch is None
    assert ctrl._bg is None
    assert ctrl._cids == []
    assert ctrl._press_x is None
    assert ctrl.active_kind() is None
    assert ctrl.gate.try_claim(object())  # gate was released, so a fresh claim succeeds
