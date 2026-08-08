"""Blitting during AnchorLineController drags: motion restores + redraws only the managed
artists via Agg's copy_from_bbox/restore_region/blit instead of a full canvas.draw_idle() per
event; release leaves the canvas in a normal (non-animated) state."""

from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backend_bases import MouseEvent
from matplotlib.figure import Figure

from dfit_tool import picks
from dfit_tool.model import TangentPick

GIDS = {"segment": "segment", "tick": "tick", "extension": "extension"}


class RecordingCanvas(FigureCanvasAgg):
    """Real Agg rendering (so copy_from_bbox/restore_region/blit actually work), with call
    counters -- same pattern as tests/test_hover_cursor.py's RecordingCanvas."""

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


def _build_axes(anchor_x, anchor_y, slope, half_len=5.0, canvas_cls=RecordingCanvas):
    fig = Figure(figsize=(6.4, 4.8), dpi=100)
    ax = fig.add_subplot(111)
    canvas = canvas_cls(fig)
    ax.set_xlim(-5.0, 25.0)
    ax.set_ylim(-20.0, 120.0)
    ax.set_autoscale_on(False)

    far_x, far_y = anchor_x + half_len, anchor_y + slope * half_len
    ax.plot([anchor_x, far_x], [anchor_y, far_y], gid="segment")
    ax.plot([anchor_x - 0.3, anchor_x + 0.3], [anchor_y - 0.3, anchor_y + 0.3], gid="tick")
    ext_x = far_x + half_len
    ax.plot([far_x, ext_x], [far_y, far_y + slope * half_len], ls="--", gid="extension")
    canvas.draw()
    return fig, ax, canvas


def _event(name, canvas, ax, x, y, button=1):
    px, py = ax.transData.transform((x, y))
    return MouseEvent(name, canvas, px, py, button=button)


def _recorder():
    calls = []
    return calls, lambda *args: calls.append(args)


def _pressed_body(canvas_cls=RecordingCanvas):
    anchor_x0, anchor_y0, slope0 = 5.0, 50.0, -3.0
    pick = TangentPick(anchor_x=anchor_x0, anchor_y=anchor_y0, slope=slope0)
    fig, ax, canvas = _build_axes(anchor_x0, anchor_y0, slope0, canvas_cls=canvas_cls)
    calls, commit = _recorder()
    ctrl = picks.AnchorLineController(canvas, ax, GIDS, get_pick=lambda: pick, commit_fn=commit)
    mid_x, mid_y = anchor_x0 + 2.5, anchor_y0 + slope0 * 2.5
    ctrl._on_press(_event("button_press_event", canvas, ax, mid_x, mid_y))
    assert ctrl._active == "body"
    return ctrl, ax, canvas, mid_x, mid_y


def test_press_paints_managed_artists_immediately_via_restore_and_blit():
    # Without a first paint at press, the just-animated artists are skipped by the preceding
    # canvas.draw() and stay invisible until the first motion event.
    anchor_x0, anchor_y0, slope0 = 5.0, 50.0, -3.0
    pick = TangentPick(anchor_x=anchor_x0, anchor_y=anchor_y0, slope=slope0)
    fig, ax, canvas = _build_axes(anchor_x0, anchor_y0, slope0)
    ctrl = picks.AnchorLineController(canvas, ax, GIDS, get_pick=lambda: pick,
                                      commit_fn=lambda *a: None)
    mid_x, mid_y = anchor_x0 + 2.5, anchor_y0 + slope0 * 2.5
    restore_before, blit_before = canvas.restore_region_calls, canvas.blit_calls

    ctrl._on_press(_event("button_press_event", canvas, ax, mid_x, mid_y))

    assert ctrl._active == "body"
    assert canvas.restore_region_calls == restore_before + 1
    assert canvas.blit_calls == blit_before + 1


def test_disconnect_mid_drag_unanimates_and_clears_state():
    ctrl, ax, canvas, mid_x, mid_y = _pressed_body()
    ctrl._on_motion(_event("motion_notify_event", canvas, ax, mid_x + 3.0, mid_y + 4.0))
    artists = ctrl._blit_artists
    assert all(a.get_animated() for a in artists)

    ctrl.disconnect()

    assert all(a.get_animated() is False for a in artists)
    assert ctrl._active is None
    assert ctrl._blit_artists is None
    assert ctrl._bg is None
    assert ctrl._cids == []
    assert ctrl.gate.try_claim(object())  # gate was released, so a fresh claim succeeds


def test_motion_during_drag_blits_instead_of_draw_idle():
    ctrl, ax, canvas, mid_x, mid_y = _pressed_body()
    # Press already did one restore/blit (the first paint) -- see
    # test_press_paints_managed_artists_immediately_via_restore_and_blit -- so check the delta.
    idle_before = canvas.draw_idle_calls
    restore_before, blit_before = canvas.restore_region_calls, canvas.blit_calls

    ctrl._on_motion(_event("motion_notify_event", canvas, ax, mid_x + 3.0, mid_y + 4.0))

    assert canvas.draw_idle_calls == idle_before
    assert canvas.restore_region_calls == restore_before + 1
    assert canvas.blit_calls == blit_before + 1


def test_release_unanimates_artists_and_schedules_a_draw():
    ctrl, ax, canvas, mid_x, mid_y = _pressed_body()
    ctrl._on_motion(_event("motion_notify_event", canvas, ax, mid_x + 3.0, mid_y + 4.0))
    seg = next(l for l in ax.get_lines() if l.get_gid() == "segment")
    assert seg.get_animated() is True

    idle_before = canvas.draw_idle_calls
    ctrl._on_release(_event("button_release_event", canvas, ax, mid_x + 3.0, mid_y + 4.0))

    assert seg.get_animated() is False
    assert canvas.draw_idle_calls == idle_before + 1
    assert ctrl._bg is None


def test_supports_blit_false_falls_back_to_draw_idle():
    ctrl, ax, canvas, mid_x, mid_y = _pressed_body(canvas_cls=NoBlitCanvas)
    assert ctrl._bg is None
    idle_before = canvas.draw_idle_calls

    ctrl._on_motion(_event("motion_notify_event", canvas, ax, mid_x + 3.0, mid_y + 4.0))

    assert canvas.draw_idle_calls == idle_before + 1
    assert canvas.restore_region_calls == 0
    assert canvas.blit_calls == 0
