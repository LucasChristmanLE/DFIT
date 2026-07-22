import math

import numpy as np
import pytest
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backend_bases import MouseEvent

from dfit_tool import picks, plots
from dfit_tool.model import TangentPick, compute_all
from tests.helpers import make_testdata, overview_state

XLIM = (-5.0, 25.0)
YLIM = (-20.0, 120.0)


def _build_axes(anchor_x, anchor_y, slope, half_len=5.0, with_tick=True, with_extension=True):
    """A gid-tagged tangent construction: segment anchor->far-end, an anchor tick, and a dashed
    extension past the far end -- the shape AnchorLineController expects from a renderer."""
    fig = Figure(figsize=(6.4, 4.8), dpi=100)
    ax = fig.add_subplot(111)
    canvas = FigureCanvasAgg(fig)
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.set_autoscale_on(False)

    far_x, far_y = anchor_x + half_len, anchor_y + slope * half_len
    ax.plot([anchor_x, far_x], [anchor_y, far_y], gid="segment")
    if with_tick:
        ax.plot([anchor_x - 0.3, anchor_x + 0.3], [anchor_y - 0.3, anchor_y + 0.3], gid="tick")
    if with_extension:
        ext_x = far_x + half_len
        ext_y = far_y + slope * half_len
        ax.plot([far_x, ext_x], [far_y, ext_y], ls="--", gid="extension")
    canvas.draw()
    return fig, ax, canvas


def _line(ax, gid):
    return next(l for l in ax.get_lines() if l.get_gid() == gid)


def _event(name, canvas, ax, x, y, button=1):
    px, py = ax.transData.transform((x, y))
    return MouseEvent(name, canvas, px, py, button=button)


def _roundtrip(ax, x, y):
    """Data (x, y) after the same event.x/event.y int-pixel rounding that matplotlib's
    ``LocationEvent`` applies (``self.x = int(x)``) -- what the controller actually receives for
    a press/motion at (x, y). Used to compute exact expectations instead of fighting sub-pixel
    rounding noise in assertions."""
    px, py = ax.transData.transform((x, y))
    xr, yr = ax.transData.inverted().transform((int(px), int(py)))
    return float(xr), float(yr)


GIDS = {"segment": "segment", "tick": "tick", "extension": "extension"}


def _recorder():
    calls = []
    return calls, lambda *args: calls.append(args)


# --------------------------------------------------------------------------------------------------
def test_anchor_drag_snaps_to_nearest_sample_and_refits_slope():
    x_arr = np.linspace(0.0, 20.0, 41)  # step 0.5
    y_arr = 100.0 - 3.0 * x_arr  # perfectly linear: any local fit gives slope -3 exactly
    anchor_x0, anchor_y0 = float(x_arr[10]), float(y_arr[10])
    pick = TangentPick(anchor_x=anchor_x0, anchor_y=anchor_y0, slope=0.0)  # deliberately stale

    fig, ax, canvas = _build_axes(anchor_x0, anchor_y0, slope=0.0)
    calls, commit = _recorder()
    ctrl = picks.AnchorLineController(canvas, ax, GIDS, get_pick=lambda: pick, commit_fn=commit,
                                      curve=(x_arr, y_arr), anchor_half=4)

    ctrl._on_press(_event("button_press_event", canvas, ax, anchor_x0, anchor_y0))
    assert ctrl._active == "anchor"

    target_idx = 25
    ctrl._on_motion(_event("motion_notify_event", canvas, ax,
                           float(x_arr[target_idx]), float(y_arr[target_idx])))
    seg = _line(ax, "segment")
    assert seg.get_xdata()[0] == pytest.approx(x_arr[target_idx])
    assert seg.get_ydata()[0] == pytest.approx(y_arr[target_idx])
    kind, ax1, ay1, slope1 = ctrl._final
    assert kind == "anchor"
    assert ax1 == pytest.approx(x_arr[target_idx])
    assert ay1 == pytest.approx(y_arr[target_idx])
    assert slope1 == pytest.approx(-3.0)

    ctrl._on_release(_event("button_release_event", canvas, ax,
                            float(x_arr[target_idx]), float(y_arr[target_idx])))
    assert ctrl._active is None
    assert len(calls) == 1
    kind, cax, cay, cslope = calls[0]
    assert kind == "anchor"
    assert cax == pytest.approx(x_arr[target_idx])
    assert cay == pytest.approx(y_arr[target_idx])
    assert cslope == pytest.approx(-3.0)


def test_body_drag_translates_anchor_slope_unchanged():
    anchor_x0, anchor_y0, slope0 = 5.0, 50.0, -3.0
    pick = TangentPick(anchor_x=anchor_x0, anchor_y=anchor_y0, slope=slope0)
    fig, ax, canvas = _build_axes(anchor_x0, anchor_y0, slope0, half_len=5.0)
    calls, commit = _recorder()
    ctrl = picks.AnchorLineController(canvas, ax, GIDS, get_pick=lambda: pick, commit_fn=commit)

    mid_x, mid_y = anchor_x0 + 2.5, anchor_y0 + slope0 * 2.5  # a point on the segment body
    ctrl._on_press(_event("button_press_event", canvas, ax, mid_x, mid_y))
    assert ctrl._active == "body"
    press_x, press_y = _roundtrip(ax, mid_x, mid_y)  # what the controller actually saw

    new_x, new_y = mid_x + 3.0, mid_y + 4.0
    ctrl._on_motion(_event("motion_notify_event", canvas, ax, new_x, new_y))
    motion_x, motion_y = _roundtrip(ax, new_x, new_y)
    expected_ax1 = anchor_x0 + (motion_x - press_x)
    expected_ay1 = anchor_y0 + (motion_y - press_y)

    kind, ax1, ay1, slope1 = ctrl._final
    assert kind == "body"
    assert ax1 == pytest.approx(expected_ax1)
    assert ay1 == pytest.approx(expected_ay1)
    assert slope1 == pytest.approx(slope0)

    seg = _line(ax, "segment")
    assert seg.get_xdata()[0] == pytest.approx(expected_ax1)
    assert seg.get_ydata()[0] == pytest.approx(expected_ay1)

    ctrl._on_release(_event("button_release_event", canvas, ax, new_x, new_y))
    assert len(calls) == 1
    assert calls[0] == ("body", pytest.approx(expected_ax1), pytest.approx(expected_ay1),
                        pytest.approx(slope0))


def test_end_drag_rotates_about_unchanged_anchor():
    anchor_x0, anchor_y0, slope0 = 5.0, 50.0, -3.0
    pick = TangentPick(anchor_x=anchor_x0, anchor_y=anchor_y0, slope=slope0)
    fig, ax, canvas = _build_axes(anchor_x0, anchor_y0, slope0, half_len=5.0)
    calls, commit = _recorder()
    ctrl = picks.AnchorLineController(canvas, ax, GIDS, get_pick=lambda: pick, commit_fn=commit)

    far_x, far_y = anchor_x0 + 5.0, anchor_y0 + slope0 * 5.0
    ctrl._on_press(_event("button_press_event", canvas, ax, far_x, far_y))
    assert ctrl._active == "end"

    target_x, target_y = anchor_x0 + 4.0, anchor_y0 + 8.0
    ctrl._on_motion(_event("motion_notify_event", canvas, ax, target_x, target_y))
    motion_x, motion_y = _roundtrip(ax, target_x, target_y)
    expected_slope = (motion_y - anchor_y0) / (motion_x - anchor_x0)

    kind, ax1, ay1, slope1 = ctrl._final
    assert kind == "end"
    assert ax1 == pytest.approx(anchor_x0)
    assert ay1 == pytest.approx(anchor_y0)
    assert slope1 == pytest.approx(expected_slope)

    seg = _line(ax, "segment")
    assert seg.get_xdata()[0] == pytest.approx(anchor_x0)  # anchor endpoint unmoved
    assert seg.get_ydata()[1] == pytest.approx(anchor_y0 + expected_slope * 5.0)  # far end re-rotated

    ctrl._on_release(_event("button_release_event", canvas, ax, target_x, target_y))
    assert calls == [("end", pytest.approx(anchor_x0), pytest.approx(anchor_y0),
                      pytest.approx(expected_slope))]


def test_pinned_mode_ignores_anchor_and_body_but_rotates():
    anchor_x0, anchor_y0, slope0 = 0.0, 0.0, 1.0
    pick = TangentPick(anchor_x=anchor_x0, anchor_y=anchor_y0, slope=slope0)
    fig, ax, canvas = _build_axes(anchor_x0, anchor_y0, slope0, half_len=5.0)
    calls, commit = _recorder()
    ctrl = picks.AnchorLineController(canvas, ax, {"segment": "segment"}, get_pick=lambda: pick,
                                      commit_fn=commit, curve=None, allow_anchor=False,
                                      allow_body=False)

    # a press on the segment body: allow_body is False, and it's not near either endpoint either.
    mid_x, mid_y = anchor_x0 + 2.5, anchor_y0 + slope0 * 2.5
    ctrl._on_press(_event("button_press_event", canvas, ax, mid_x, mid_y))
    assert ctrl._active is None

    far_x, far_y = anchor_x0 + 5.0, anchor_y0 + slope0 * 5.0
    ctrl._on_press(_event("button_press_event", canvas, ax, far_x, far_y))
    assert ctrl._active == "end"

    target_x, target_y = anchor_x0 + 5.0, anchor_y0 + 10.0
    ctrl._on_motion(_event("motion_notify_event", canvas, ax, target_x, target_y))
    motion_x, motion_y = _roundtrip(ax, target_x, target_y)
    expected_slope = (motion_y - anchor_y0) / (motion_x - anchor_x0)

    kind, ax1, ay1, slope1 = ctrl._final
    assert (ax1, ay1) == (anchor_x0, anchor_y0)
    assert slope1 == pytest.approx(expected_slope)

    ctrl._on_release(_event("button_release_event", canvas, ax, target_x, target_y))
    assert calls == [("end", pytest.approx(0.0), pytest.approx(0.0), pytest.approx(expected_slope))]


def test_twinx_regression_still_captures_via_pixel_not_inaxes():
    td = make_testdata()
    st = overview_state(td)
    res = compute_all(st, td)
    fig = Figure()
    ax = fig.add_subplot(111)
    canvas = FigureCanvasAgg(fig)
    plots.render_overview(ax, td, st, res)
    canvas.draw()
    twins = [a for a in canvas.figure.axes if a is not ax]
    assert twins, "expected a twinx rate axis on the overview"

    xlo, xhi = ax.get_xlim()
    anchor_x, anchor_y = xlo + 0.25 * (xhi - xlo), 0.0
    far_x = anchor_x + 0.1 * (xhi - xlo)
    pick = TangentPick(anchor_x=anchor_x, anchor_y=anchor_y, slope=1.0)
    ax.plot([anchor_x, far_x], [anchor_y, anchor_y + (far_x - anchor_x)], gid="segment")
    canvas.draw()

    ev = _event("button_press_event", canvas, ax, anchor_x, anchor_y)
    assert ev.inaxes is not ax  # matplotlib assigns the topmost (twin) axis, as in DragLineController's regression

    calls, commit = _recorder()
    ctrl = picks.AnchorLineController(canvas, ax, {"segment": "segment"}, get_pick=lambda: pick,
                                      commit_fn=commit, curve=None, allow_anchor=False,
                                      allow_body=False)
    ctrl._on_press(ev)
    assert ctrl._active == "end"  # pressed exactly on the near (anchor) endpoint of the segment


def test_readout_created_on_press_updated_on_motion_removed_on_release():
    anchor_x0, anchor_y0, slope0 = 5.0, 50.0, -3.0
    pick = TangentPick(anchor_x=anchor_x0, anchor_y=anchor_y0, slope=slope0)
    fig, ax, canvas = _build_axes(anchor_x0, anchor_y0, slope0, half_len=5.0)
    calls, commit = _recorder()

    def readout(kind, ax_, ay_, slope_):
        return f"{kind} {ax_:.2f} {ay_:.2f} {slope_:.3f}"

    ctrl = picks.AnchorLineController(canvas, ax, GIDS, get_pick=lambda: pick, commit_fn=commit,
                                      readout_fn=readout)

    mid_x, mid_y = anchor_x0 + 2.5, anchor_y0 + slope0 * 2.5
    ctrl._on_press(_event("button_press_event", canvas, ax, mid_x, mid_y))
    assert len(ax.texts) == 1
    assert ax.texts[0].get_text() == f"body {anchor_x0:.2f} {anchor_y0:.2f} {slope0:.3f}"
    press_x, press_y = _roundtrip(ax, mid_x, mid_y)

    new_x, new_y = mid_x + 3.0, mid_y + 4.0
    ctrl._on_motion(_event("motion_notify_event", canvas, ax, new_x, new_y))
    motion_x, motion_y = _roundtrip(ax, new_x, new_y)
    exp_ax1 = anchor_x0 + (motion_x - press_x)
    exp_ay1 = anchor_y0 + (motion_y - press_y)
    expected = f"body {exp_ax1:.2f} {exp_ay1:.2f} {slope0:.3f}"
    assert ax.texts[0].get_text() == expected

    ctrl._on_release(_event("button_release_event", canvas, ax, new_x, new_y))
    assert len(ax.texts) == 0


def test_missing_segment_artist_is_a_no_op():
    pick = TangentPick(anchor_x=5.0, anchor_y=50.0, slope=-3.0)
    fig, ax, canvas = _build_axes(0.0, 0.0, 0.0, with_tick=False, with_extension=False)
    # "segment" gid points at nothing drawn on this axes.
    calls, commit = _recorder()
    ctrl = picks.AnchorLineController(canvas, ax, {"segment": "does-not-exist"},
                                      get_pick=lambda: pick, commit_fn=commit)
    ctrl._on_press(_event("button_press_event", canvas, ax, 5.0, 50.0))
    assert ctrl._active is None
    assert calls == []


def test_missing_pick_is_a_no_op():
    fig, ax, canvas = _build_axes(5.0, 50.0, -3.0)
    calls, commit = _recorder()
    ctrl = picks.AnchorLineController(canvas, ax, GIDS, get_pick=lambda: None, commit_fn=commit)
    ctrl._on_press(_event("button_press_event", canvas, ax, 5.0, 50.0))
    assert ctrl._active is None
    assert calls == []


def test_tick_slide_grabs_whole_tick_not_just_the_anchor_point():
    """Regression: the visible vertical tick is drawn taller on screen (``+/-tick_half_y`` data
    units, often much more than ``tol_px`` pixels) than the ``tol_px`` circle the old hit-test drew
    around only the anchor *point*. Pressing near the tick's tip -- farther than ``tol_px`` from
    the anchor point, but still squarely on the tick -- must still enter "anchor" mode. Before the
    fix, a steep segment passing within ``tol_px`` of that same pixel let the press fall through to
    "body" instead (a pan when a slide was intended)."""
    fig = Figure(figsize=(6.0, 6.0), dpi=100)
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9)  # exact 480x480 px axes bbox
    ax = fig.add_subplot(111)
    canvas = FigureCanvasAgg(fig)
    ax.set_xlim(0.0, 50.0)      # scale_x = 480 / 50  = 9.6 px per data-unit
    ax.set_ylim(-40.0, 80.0)    # scale_y = 480 / 120 = 4.0 px per data-unit
    ax.set_autoscale_on(False)

    # A steep segment (slope=8) through the anchor: at the pixel above the anchor where the tick's
    # tip sits, the segment's finite body -- point-to-segment, not just its two endpoints -- passes
    # within tol_px, which is exactly what let the old point-only "anchor" test fall through to
    # "body" instead.
    anchor_x, anchor_y, slope = 20.0, 20.0, 8.0
    half = 5.0
    x0, x1 = anchor_x - half, anchor_x + half
    y0, y1 = anchor_y + slope * (x0 - anchor_x), anchor_y + slope * (x1 - anchor_x)
    ax.plot([x0, x1], [y0, y1], gid="segment")
    tick_half_y = 8.0  # 8 data-units * 4.0 px/unit = 32 px each way on screen, vs tol_px=12
    ax.plot([anchor_x, anchor_x], [anchor_y - tick_half_y, anchor_y + tick_half_y], gid="tick")
    canvas.draw()

    pick = TangentPick(anchor_x=anchor_x, anchor_y=anchor_y, slope=slope)
    x_arr = np.linspace(x0 - 2.0, x1 + 2.0, 61)
    y_arr = anchor_y + slope * (x_arr - anchor_x)  # perfectly linear, matching the segment's slope
    calls, commit = _recorder()
    ctrl = picks.AnchorLineController(canvas, ax, GIDS, get_pick=lambda: pick, commit_fn=commit,
                                      curve=(x_arr, y_arr))

    press_x, press_y = anchor_x, anchor_y + 7.5  # near the tick's tip, straight up from the anchor
    apx = ax.transData.transform((anchor_x, anchor_y))
    ppx = ax.transData.transform((press_x, press_y))
    # Sanity: this press really is outside the old anchor-point-only circle.
    assert math.hypot(ppx[0] - apx[0], ppx[1] - apx[1]) > ctrl.tol_px

    ev = _event("button_press_event", canvas, ax, press_x, press_y)
    ctrl._on_press(ev)
    assert ctrl._active == "anchor"  # grabbed via the tick, not missed and fallen through to "body"

    ctrl._on_motion(ev)
    ctrl._on_release(ev)
    assert len(calls) == 1
    assert calls[0][0] == "anchor"  # the old bug's fallthrough result ("body") must not occur
