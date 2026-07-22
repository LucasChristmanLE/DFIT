"""Interactive picking on the matplotlib canvas.

Generic controllers capture mouse events and forward data coordinates (or, for the anchored-line
controllers, a finished line/point geometry) to a callback. Pure ``commit_*`` functions translate
that geometry into a ``PickState`` change; they touch no matplotlib object, so they are unit-
testable without a canvas. Everything else here depends only on matplotlib (never Tkinter), so the
picking layer is reusable if the shell is ported.

Interaction model: drag-to-move for lines/points (snap-to-sample where a backing curve exists),
drag-select for windows. Every controller hit-tests through its own Axes' pixel transforms
(``_axes_contains_pixel`` / ``_data_from_pixel``) rather than ``event.inaxes`` -- a twinned Axes
(e.g. the overview's rate ``twinx``) owns ``inaxes`` over the shared region, so an identity check
against a specific Axes would never match. See ``test_overview_rate_twin_owns_inaxes_regression``.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
from matplotlib.widgets import SpanSelector

from . import interpret
from .model import DerivedResults, PickState, TangentPick
from .io_load import TestData


# --------------------------------------------------------------------------------------------------
# pixel/data helpers shared by every controller below
# --------------------------------------------------------------------------------------------------
def _axes_contains_pixel(ax, event) -> bool:
    """True if the event pixel falls within ``ax``'s bbox, regardless of which (possibly
    twinned) axes matplotlib assigned to ``event.inaxes``. A twin axes overlaid on the same
    region owns ``inaxes`` over the shared area, so an identity check against a specific axes
    would never match; testing the pixel against this axes' own bbox is robust either way."""
    return (event.x is not None and event.y is not None
            and ax.bbox.contains(event.x, event.y))


def _data_from_pixel(ax, event) -> tuple[float, float]:
    """Data (x, y) on ``ax`` for the event pixel (robust to a twin axes owning ``event.inaxes``)."""
    x, y = ax.transData.inverted().transform((event.x, event.y))
    return float(x), float(y)


def _nearest_index_by_pixel(ax, x_arr: np.ndarray, event) -> int:
    """Index into ``x_arr`` whose on-screen (transData) x-pixel is nearest ``event.x``. Matching
    by pixel rather than raw data-x keeps snapping correct on a log-scaled axis too."""
    xs = np.asarray(x_arr, dtype=float)
    px = ax.transData.transform(np.column_stack([xs, np.zeros_like(xs)]))[:, 0]
    return int(np.nanargmin(np.abs(px - event.x)))


def _segment_distance_px(ax, p0_data, p1_data, event) -> float:
    """Pixel distance from the event to the finite segment between two data points (point-to-
    segment, not point-to-infinite-line), via this axes' own transform."""
    p0 = np.asarray(ax.transData.transform(p0_data), dtype=float)
    p1 = np.asarray(ax.transData.transform(p1_data), dtype=float)
    p = np.array([event.x, event.y], dtype=float)
    ab = p1 - p0
    denom = float(np.dot(ab, ab))
    t = 0.0 if denom == 0 else float(np.clip(np.dot(p - p0, ab) / denom, 0.0, 1.0))
    closest = p0 + t * ab
    return float(np.hypot(*(p - closest)))


# --------------------------------------------------------------------------------------------------
# capture arbitration
# --------------------------------------------------------------------------------------------------
class _CaptureGate:
    """Per-gesture press arbiter shared by controllers whose hit zones overlap on one Axes (e.g.
    the eff-ISIP anchor sitting near the contact marker).

    Ordering contract: a controller calls ``try_claim(self)`` only *after* its own hit-test at
    press has already succeeded -- claiming before hit-testing would let a miss on one controller
    block a hit on another sharing the gate. Whichever controller claims first holds the gate for
    the rest of the gesture (motion + release); ``release()`` on button-release frees it so the
    next press is a fresh contest (first hit-test to succeed wins again, not necessarily the same
    controller).
    """

    def __init__(self):
        self._owner = None

    def try_claim(self, owner) -> bool:
        if self._owner is None or self._owner is owner:
            self._owner = owner
            return True
        return False

    def release(self):
        self._owner = None


# --------------------------------------------------------------------------------------------------
# generic controllers
# --------------------------------------------------------------------------------------------------
class ClickController:
    """Forwards left/right clicks inside a target Axes as (xdata, ydata, button)."""

    def __init__(self, canvas, ax, on_click: Callable[[float, float, int], None]):
        self.canvas = canvas
        self.ax = ax
        self.on_click = on_click
        self._cid = canvas.mpl_connect("button_press_event", self._handle)

    def _handle(self, event):
        if event.inaxes is not self.ax or event.xdata is None:
            return
        if event.button not in (1, 3):  # left / right
            return
        self.on_click(event.xdata, event.ydata, event.button)

    def disconnect(self):
        self.canvas.mpl_disconnect(self._cid)


class SpanController:
    """Horizontal drag-select on a target Axes; forwards (xmin, xmax)."""

    def __init__(self, ax, on_span: Callable[[float, float], None]):
        self.on_span = on_span
        self.selector = SpanSelector(
            ax, self._handle, "horizontal", useblit=True,
            props=dict(alpha=0.2, facecolor="tab:orange"), interactive=True,
        )

    def _handle(self, xmin, xmax):
        if xmax > xmin:
            self.on_span(xmin, xmax)

    def disconnect(self):
        self.selector.disconnect_events()


class DragLineController:
    """Drag gid-tagged vertical lines within an Axes; commit the released x per gid.

    ``handlers`` maps an axvline gid to ``on_release(x_data)``. During a drag the line is moved
    live (no recompute); on release the matching handler is called with the final x. ``guard()``
    returning True blocks capture (e.g. while the toolbar zoom/pan mode is active).

    Hit-tests and reads the cursor through ``_axes_contains_pixel``/``_data_from_pixel`` (this
    axes' own transforms) rather than ``event.inaxes``/``event.xdata``: an overlaid twin axes
    (e.g. the overview's rate ``twinx``) owns ``inaxes`` over the shared region, so an identity
    check against ``self.ax`` would never match. Twinned axes share the x-scale, so converting the
    event pixel through ``self.ax.transData`` yields the correct data-x regardless.
    """

    def __init__(self, canvas, ax, handlers, guard=None, tol_px: float = 6.0):
        self.canvas = canvas
        self.ax = ax
        self.handlers = handlers
        self.guard = guard or (lambda: False)
        self.tol_px = tol_px
        self._active = None
        self._cids = [
            canvas.mpl_connect("button_press_event", self._on_press),
            canvas.mpl_connect("motion_notify_event", self._on_motion),
            canvas.mpl_connect("button_release_event", self._on_release),
        ]

    def _lines(self):
        return [l for l in self.ax.get_lines() if l.get_gid() in self.handlers]

    def _on_press(self, event):
        if event.button != 1 or not _axes_contains_pixel(self.ax, event):
            return
        if self.guard():
            return
        best, best_d = None, self.tol_px
        for line in self._lines():
            lx = line.get_xdata()[0]
            px = self.ax.transData.transform((lx, 0.0))[0]
            d = abs(px - event.x)
            if d <= best_d:
                best, best_d = line, d
        self._active = best

    def _on_motion(self, event):
        if self._active is None or event.x is None or event.y is None:
            return
        x, _ = _data_from_pixel(self.ax, event)
        self._active.set_xdata([x, x])
        self.canvas.draw_idle()

    def _on_release(self, event):
        if self._active is None:
            return
        line = self._active
        self._active = None
        if event.x is not None and event.y is not None:
            x, _ = _data_from_pixel(self.ax, event)
        else:
            x = float(line.get_xdata()[0])
        handler = self.handlers.get(line.get_gid())
        if handler is not None:
            handler(float(x))

    def disconnect(self):
        for cid in self._cids:
            self.canvas.mpl_disconnect(cid)
        self._cids = []


class AnchorLineController:
    """Drag a gid-tagged anchored line: geometry is ``(anchor_x, anchor_y, slope)``. The current
    anchor+slope at press time come from ``get_pick()`` -- the caller's own PickState accessor
    (e.g. ``lambda: state.isip_tangent``) -- never reverse-engineered from the artists.

    ``gids`` names the renderer-drawn pieces of the tangent construction:
      - ``"segment"`` (required): the finite Line2D drawn through the anchor.
      - ``"tick"`` (optional): a short mark at the anchor.
      - ``"extension"`` (optional): a Line2D continuing the segment (e.g. dashed, to a reference
        vertical).
    A missing pick (``get_pick() is None``) or a missing "segment" artist makes every press a
    no-op -- the pick simply hasn't been placed yet.

    Hit priority at press (tolerance ``tol_px``, all through this axes' own transforms -- see
    ``_axes_contains_pixel``/``_data_from_pixel`` -- never ``event.inaxes``):
      1. the anchor/tick zone -- only if ``allow_anchor`` and ``curve`` is given and a "tick"
         artist is present.
      2. either segment endpoint -- only if ``allow_rotate``.
      3. the segment body (point-to-segment pixel distance) -- only if ``allow_body``.

    Motion (nothing is written to ``PickState`` until release; artists update live via
    ``canvas.draw_idle()``):
      - **"anchor"**: snaps to the nearest ``curve`` sample by x-pixel and refits
        ``(anchor_x, anchor_y, slope) = _tangent_from_index(curve_x, curve_y, idx, anchor_half)``.
      - **"body"**: translates the anchor by the press-to-cursor *data* delta; slope unchanged.
      - **"end"**: rotates about the (unchanged) anchor: ``slope = (y - anchor_y) / (x -
        anchor_x)`` for the cursor's current data position. A press-to-cursor horizontal pixel
        delta under ``_ROTATE_DX_EPS_PX`` keeps the prior slope (guards a vertical/zero-width
        drag rather than blowing up or dividing by zero).
      "segment"/"extension" are redrawn each motion from their *press-time* x-offset-from-anchor
      (every point stays at the same signed x-distance from the anchor; y is recomputed from the
      live anchor+slope) -- this reproduces pure translation exactly and gives rotation a stable
      visual length. "tick" translates by the anchor's data-delta during "anchor"/"body" motion
      and is left untouched during "end" motion, since the anchor -- and so the tick's position --
      does not move while rotating.

    On release, ``commit_fn(kind, anchor_x, anchor_y, slope)`` is called exactly once with the
    *final* geometry -- never the raw cursor position -- where ``kind`` is one of "anchor" /
    "body" / "end". Pinned mode (``curve=None, allow_anchor=False, allow_body=False``) leaves only
    "end" reachable: a through-origin (or otherwise fixed-anchor) rotate-only line, with the
    anchor staying at whatever ``get_pick()`` returns (e.g. anchor (0, 0)).

    ``readout_fn(kind, anchor_x, anchor_y, slope) -> str | None``, if given, drives a small Text
    artist: created on press, updated every motion, removed on release/disconnect (returning
    ``None`` from it also removes the text early, mid-drag).

    ``gate`` (a shared ``_CaptureGate``) is claimed only *after* this controller's own hit-test at
    press succeeds, and released on button-release -- so a miss here never blocks a sibling
    controller sharing the gate, and whichever controller's hit-test succeeds first (in
    ``mpl_connect``/construction order) wins the gesture. Defaults to a private gate (no sharing)
    when omitted.
    """

    _ROTATE_DX_EPS_PX = 1.0

    def __init__(self, canvas, ax, gids: dict, get_pick: Callable[[], Optional[TangentPick]],
                 commit_fn: Callable[[str, float, float, float], None], curve=None,
                 anchor_half: int = 4, allow_anchor: bool = True, allow_body: bool = True,
                 allow_rotate: bool = True, tol_px: float = 8.0, readout_fn=None,
                 gate: Optional[_CaptureGate] = None):
        self.canvas = canvas
        self.ax = ax
        self.gids = gids
        self.get_pick = get_pick
        self.commit_fn = commit_fn
        self.curve = curve
        self.anchor_half = anchor_half
        self.allow_anchor = allow_anchor and curve is not None
        self.allow_body = allow_body
        self.allow_rotate = allow_rotate
        self.tol_px = tol_px
        self.readout_fn = readout_fn
        self.gate = gate if gate is not None else _CaptureGate()

        self._active: Optional[str] = None
        self._press_anchor = None   # (anchor_x, anchor_y, slope) snapshot at press
        self._press_data = None     # (x, y) data coords under the cursor at press
        self._seg_offsets = None
        self._ext_offsets = None
        self._tick_orig = None
        self._final = None          # (kind, anchor_x, anchor_y, slope) -- last-known-good geometry
        self._readout = None
        self._cids = [
            canvas.mpl_connect("button_press_event", self._on_press),
            canvas.mpl_connect("motion_notify_event", self._on_motion),
            canvas.mpl_connect("button_release_event", self._on_release),
        ]

    # ---- artist lookup ----
    def _artist(self, gid):
        if gid is None:
            return None
        for line in self.ax.get_lines():
            if line.get_gid() == gid:
                return line
        return None

    # ---- hit testing (priority: anchor tick zone -> segment ends -> line body) ----
    def _hit_test(self, event, pick, segment) -> Optional[str]:
        if self.allow_anchor:
            tick = self._artist(self.gids.get("tick"))
            if tick is not None:
                apx = self.ax.transData.transform((pick.anchor_x, pick.anchor_y))
                if math.hypot(event.x - apx[0], event.y - apx[1]) <= self.tol_px:
                    return "anchor"
        if self.allow_rotate:
            xs, ys = segment.get_xdata(), segment.get_ydata()
            best = None
            for x, y in zip(xs, ys):
                epx = self.ax.transData.transform((x, y))
                d = math.hypot(event.x - epx[0], event.y - epx[1])
                if d <= self.tol_px and (best is None or d < best):
                    best = d
            if best is not None:
                return "end"
        if self.allow_body:
            xs, ys = segment.get_xdata(), segment.get_ydata()
            d = _segment_distance_px(self.ax, (xs[0], ys[0]), (xs[-1], ys[-1]), event)
            if d <= self.tol_px:
                return "body"
        return None

    def _on_press(self, event):
        if event.button != 1 or not _axes_contains_pixel(self.ax, event):
            return
        pick = self.get_pick()
        if pick is None:
            return
        segment = self._artist(self.gids.get("segment"))
        if segment is None:
            return
        kind = self._hit_test(event, pick, segment)
        if kind is None:
            return
        if not self.gate.try_claim(self):
            return

        self._active = kind
        self._press_anchor = (pick.anchor_x, pick.anchor_y, pick.slope)
        self._press_data = _data_from_pixel(self.ax, event)
        self._seg_offsets = [float(x) - pick.anchor_x for x in segment.get_xdata()]
        ext = self._artist(self.gids.get("extension"))
        self._ext_offsets = ([float(x) - pick.anchor_x for x in ext.get_xdata()]
                             if ext is not None else None)
        tick = self._artist(self.gids.get("tick"))
        self._tick_orig = ((list(tick.get_xdata()), list(tick.get_ydata()))
                           if tick is not None else None)
        self._final = (kind, pick.anchor_x, pick.anchor_y, pick.slope)

        if self.readout_fn is not None:
            text = self.readout_fn(kind, pick.anchor_x, pick.anchor_y, pick.slope)
            if text is not None:
                self._readout = self.ax.text(0.02, 0.95, text, transform=self.ax.transAxes,
                                             fontsize=8, va="top", ha="left")

    def _on_motion(self, event):
        if self._active is None or event.x is None or event.y is None:
            return
        kind = self._active
        ax0, ay0, slope0 = self._press_anchor
        if kind == "anchor":
            x_arr, y_arr = self.curve
            idx = _nearest_index_by_pixel(self.ax, x_arr, event)
            ax1, ay1, slope1 = _tangent_from_index(x_arr, y_arr, idx, self.anchor_half)
        elif kind == "body":
            cx, cy = _data_from_pixel(self.ax, event)
            px0, py0 = self._press_data
            ax1, ay1, slope1 = ax0 + (cx - px0), ay0 + (cy - py0), slope0
        else:  # "end" -- rotate about the (unchanged) anchor
            cx, cy = _data_from_pixel(self.ax, event)
            apx = self.ax.transData.transform((ax0, ay0))
            if abs(event.x - apx[0]) < self._ROTATE_DX_EPS_PX or cx == ax0:
                slope1 = slope0  # near-vertical/zero-width drag: keep the prior slope
            else:
                slope1 = (cy - ay0) / (cx - ax0)
            ax1, ay1 = ax0, ay0

        self._apply_geometry(ax1, ay1, slope1)
        self._final = (kind, ax1, ay1, slope1)

        if self._readout is not None:
            text = self.readout_fn(kind, ax1, ay1, slope1)
            if text is None:
                self._readout.remove()
                self._readout = None
            else:
                self._readout.set_text(text)
        self.canvas.draw_idle()

    def _apply_geometry(self, ax1, ay1, slope1):
        segment = self._artist(self.gids.get("segment"))
        if segment is not None and self._seg_offsets is not None:
            segment.set_data([ax1 + t for t in self._seg_offsets],
                             [ay1 + slope1 * t for t in self._seg_offsets])
        ext = self._artist(self.gids.get("extension"))
        if ext is not None and self._ext_offsets is not None:
            ext.set_data([ax1 + t for t in self._ext_offsets],
                         [ay1 + slope1 * t for t in self._ext_offsets])
        tick = self._artist(self.gids.get("tick"))
        if tick is not None and self._tick_orig is not None and self._active in ("anchor", "body"):
            ax0, ay0, _ = self._press_anchor
            dx, dy = ax1 - ax0, ay1 - ay0
            txo, tyo = self._tick_orig
            tick.set_data([x + dx for x in txo], [y + dy for y in tyo])

    def _on_release(self, event):
        if self._active is None:
            return
        kind, ax1, ay1, slope1 = self._final
        self._active = None
        self._final = None
        self._seg_offsets = None
        self._ext_offsets = None
        self._tick_orig = None
        self._press_anchor = None
        self._press_data = None
        if self._readout is not None:
            self._readout.remove()
            self._readout = None
        self.gate.release()
        self.commit_fn(kind, float(ax1), float(ay1), float(slope1))

    def disconnect(self):
        for cid in self._cids:
            self.canvas.mpl_disconnect(cid)
        self._cids = []
        if self._readout is not None:
            self._readout.remove()
            self._readout = None


class DraggablePointController:
    """Drag a single gid-tagged marker, snapped along a backing curve.

    Press within ``tol_px`` (this axes' own transforms, never ``event.inaxes``) of the gid-tagged
    marker's current position captures it. Motion snaps the marker to the nearest
    ``(curve_x, curve_y)`` sample by x-pixel (``_nearest_index_by_pixel``) and moves it live. On
    release, ``commit_fn(float(snapped_x))`` is called once with the final snapped x. A missing
    marker artist (gid not found) makes press a no-op.

    ``gate`` follows the same claim-after-hit-test / release-on-release contract as
    ``AnchorLineController`` -- see ``_CaptureGate``.
    """

    def __init__(self, canvas, ax, gid: str, curve_x: np.ndarray, curve_y: np.ndarray,
                 commit_fn: Callable[[float], None], tol_px: float = 8.0,
                 gate: Optional[_CaptureGate] = None):
        self.canvas = canvas
        self.ax = ax
        self.gid = gid
        self.curve_x = curve_x
        self.curve_y = curve_y
        self.commit_fn = commit_fn
        self.tol_px = tol_px
        self.gate = gate if gate is not None else _CaptureGate()

        self._dragging = False
        self._final_x = None
        self._cids = [
            canvas.mpl_connect("button_press_event", self._on_press),
            canvas.mpl_connect("motion_notify_event", self._on_motion),
            canvas.mpl_connect("button_release_event", self._on_release),
        ]

    def _artist(self):
        for line in self.ax.get_lines():
            if line.get_gid() == self.gid:
                return line
        return None

    def _on_press(self, event):
        if event.button != 1 or not _axes_contains_pixel(self.ax, event):
            return
        marker = self._artist()
        if marker is None:
            return
        xs, ys = marker.get_xdata(), marker.get_ydata()
        if len(xs) == 0:
            return
        mpx = self.ax.transData.transform((xs[0], ys[0]))
        if math.hypot(event.x - mpx[0], event.y - mpx[1]) > self.tol_px:
            return
        if not self.gate.try_claim(self):
            return
        self._dragging = True
        self._final_x = float(xs[0])

    def _on_motion(self, event):
        if not self._dragging or event.x is None or event.y is None:
            return
        idx = _nearest_index_by_pixel(self.ax, self.curve_x, event)
        marker = self._artist()
        if marker is not None:
            marker.set_data([float(self.curve_x[idx])], [float(self.curve_y[idx])])
        self._final_x = float(self.curve_x[idx])
        self.canvas.draw_idle()

    def _on_release(self, event):
        if not self._dragging:
            return
        self._dragging = False
        self.gate.release()
        self.commit_fn(float(self._final_x))

    def disconnect(self):
        for cid in self._cids:
            self.canvas.mpl_disconnect(cid)
        self._cids = []


# --------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------
def _nearest(arr: np.ndarray, value: float) -> int:
    return int(np.nanargmin(np.abs(np.asarray(arr, dtype=float) - value)))


def _local_slope(x: np.ndarray, y: np.ndarray, idx: int, half: int = 4) -> float:
    lo, hi = max(0, idx - half), min(len(x), idx + half + 1)
    if hi - lo < 2:
        return 0.0
    m, _ = interpret.fit_line(x[lo:hi], y[lo:hi])
    return m


def _tangent_from_index(x_arr: np.ndarray, y_arr: np.ndarray, idx: int,
                        half: int = 4) -> tuple[float, float, float]:
    """The tangent line anchored at sample ``idx``: ``(anchor_x, anchor_y, slope)``, the slope
    from a local fit of the +/-``half`` neighborhood (``_local_slope``)."""
    slope = _local_slope(x_arr, y_arr, idx, half=half)
    return float(x_arr[idx]), float(y_arr[idx]), slope


# --------------------------------------------------------------------------------------------------
# step pick handlers  (mutate PickState in place)
# --------------------------------------------------------------------------------------------------
def handle_isip_click(state: PickState, td: TestData, res: DerivedResults,
                      x_min: float, button: int) -> None:
    """Deprecated: superseded by commit_* controllers; removed when ui.py is rewired (Task 4).

    Set the literal-ISIP tangent: anchor at the nearest BHP sample, slope from a local fit."""
    if res.t_shutin_s is None or res.bhp_all is None:
        return
    t_target = res.t_shutin_s + x_min * 60.0
    idx = _nearest(td.t_s, t_target)
    slope = _local_slope(td.t_s, res.bhp_all, idx, half=30)  # psi/s over ~1 min of 1 Hz data
    state.isip_tangent = TangentPick(anchor_x=float(td.t_s[idx]),
                                     anchor_y=float(res.bhp_all[idx]), slope=slope)


def handle_gfunction_click(state: PickState, res: DerivedResults,
                           x_G: float, button: int) -> None:
    """Deprecated: superseded by commit_* controllers; removed when ui.py is rewired (Task 4).

    Left-click sets the effective-ISIP line (anchor + local P-vs-G slope); right-click contact."""
    if res.diagnostics is None:
        return
    G = res.diagnostics.G
    p = res.resampled.p
    j = _nearest(G, x_G)
    if button == 1:
        slope = _local_slope(G, p, j, half=4)
        state.eff_isip_line = TangentPick(anchor_x=float(G[j]), anchor_y=float(p[j]), slope=slope)
    else:
        state.contact_G = float(G[j])


def handle_tangent_click(state: PickState, res: DerivedResults, x_G: float, button: int) -> None:
    """Deprecated: superseded by commit_* controllers; removed when ui.py is rewired (Task 4).

    Set the tangent-method closure (departure) point; refresh the through-origin slope."""
    if res.diagnostics is None:
        return
    dg = res.diagnostics
    j = _nearest(dg.G, x_G)
    state.closure_G = float(dg.G[j])
    if state.closure_slope is None:
        slope, _ = interpret.suggest_closure_tangent(dg.G, dg.GdPdG)
        state.closure_slope = slope


def handle_loglog_span(state: PickState, lo: float, hi: float) -> None:
    state.loglog_window = (float(lo), float(hi))


def handle_pp_span(state: PickState, lo: float, hi: float) -> None:
    state.pp_window = (float(lo), float(hi))


# --------------------------------------------------------------------------------------------------
# pure commit functions -- mutate PickState only, no matplotlib.
#
# Convention shared by every AnchorLineController.commit_fn: called as
# ``commit_fn(kind, anchor_x, anchor_y, slope)`` with the controller's *final* line geometry
# (never the raw cursor position); ``kind`` is one of "anchor" / "body" / "end". For "anchor",
# the commit function independently re-derives the anchor+slope from ``anchor_x`` (nearest sample
# + local refit at the step's own half-window) rather than trusting the controller's live-preview
# values -- so it stays correct even when called directly (as the commit-function unit tests do),
# and is insulated from a controller ``anchor_half`` that doesn't match this half.
# --------------------------------------------------------------------------------------------------
def commit_isip_tangent(state: PickState, td: TestData, res: DerivedResults,
                        kind: str, anchor_x: float, anchor_y: float, slope: float) -> None:
    """Commit the literal-ISIP tangent (BHP vs time-seconds) after an AnchorLineController drag."""
    if kind == "anchor":
        idx = _nearest(td.t_s, anchor_x)
        ax_, ay_, sl_ = _tangent_from_index(td.t_s, res.bhp_all, idx, half=30)
        state.isip_tangent = TangentPick(anchor_x=ax_, anchor_y=ay_, slope=sl_)
    elif kind == "body":
        prev = state.isip_tangent
        sl_ = prev.slope if prev is not None else float(slope)
        state.isip_tangent = TangentPick(anchor_x=float(anchor_x), anchor_y=float(anchor_y),
                                         slope=sl_)
    elif kind == "end":
        prev = state.isip_tangent
        ax_, ay_ = (prev.anchor_x, prev.anchor_y) if prev is not None else (anchor_x, anchor_y)
        state.isip_tangent = TangentPick(anchor_x=float(ax_), anchor_y=float(ay_),
                                         slope=float(slope))


def commit_eff_isip_line(state: PickState, res: DerivedResults,
                         kind: str, anchor_x: float, anchor_y: float, slope: float) -> None:
    """Commit the effective-ISIP line (BHP vs G) after an AnchorLineController drag."""
    if res.diagnostics is None or res.resampled is None:
        return
    G, p = res.diagnostics.G, res.resampled.p
    if kind == "anchor":
        idx = _nearest(G, anchor_x)
        ax_, ay_, sl_ = _tangent_from_index(G, p, idx, half=4)
        state.eff_isip_line = TangentPick(anchor_x=ax_, anchor_y=ay_, slope=sl_)
    elif kind == "body":
        prev = state.eff_isip_line
        sl_ = prev.slope if prev is not None else float(slope)
        state.eff_isip_line = TangentPick(anchor_x=float(anchor_x), anchor_y=float(anchor_y),
                                          slope=sl_)
    elif kind == "end":
        prev = state.eff_isip_line
        ax_, ay_ = (prev.anchor_x, prev.anchor_y) if prev is not None else (anchor_x, anchor_y)
        state.eff_isip_line = TangentPick(anchor_x=float(ax_), anchor_y=float(ay_),
                                          slope=float(slope))


def commit_closure_line(state: PickState, res: DerivedResults,
                        kind: str, anchor_x: float, anchor_y: float, slope: float) -> None:
    """Commit the through-origin tangent-closure line (tangent-method step). The controller for
    this line is always constructed pinned (``curve=None, allow_anchor=False, allow_body=False``),
    so only "end" is ever reached and only the slope is meaningful -- the anchor stays fixed at
    the origin. ``res``/``kind``/``anchor_x``/``anchor_y`` are accepted for signature symmetry
    with the other AnchorLineController commit functions but are not needed here."""
    state.closure_slope = float(slope)


def commit_contact_point(state: PickState, x: float) -> None:
    """DraggablePointController commit for the compliance-method contact pick (G-function step)."""
    state.contact_G = float(x)


def commit_closure_point(state: PickState, x: float) -> None:
    """DraggablePointController commit for the tangent-method closure (departure) pick."""
    state.closure_G = float(x)


# --------------------------------------------------------------------------------------------------
# default seeding (auto-suggestions used as starting picks)
# --------------------------------------------------------------------------------------------------
def seed_defaults(state: PickState, td: TestData, res_fn: Callable[[PickState], DerivedResults]) -> None:
    """Populate sensible starting picks after a file loads. ``res_fn`` recomputes DerivedResults."""
    if state.rate_col:
        rate = td.column(state.rate_col)
        vol = td.column(state.volume_col) if state.volume_col else None
        try:
            state.start_idx, state.shutin_idx = interpret.suggest_injection_window(rate, vol)
        except ValueError:
            pass

    res = res_fn(state)

    # literal-ISIP tangent: anchor ~1 min after shut-in, slope from a local fit of the early decline
    if res.t_shutin_s is not None and res.bhp_all is not None:
        idx = _nearest(td.t_s, res.t_shutin_s + 60.0)
        slope = _local_slope(td.t_s, res.bhp_all, idx, half=30)
        state.isip_tangent = TangentPick(anchor_x=float(td.t_s[idx]),
                                         anchor_y=float(res.bhp_all[idx]), slope=slope)

    dg = res.diagnostics
    if dg is not None and res.resampled is not None and len(dg.G) > 5:
        # effective-ISIP line: anchor where the P-vs-G curve straightens (past the dP/dG hump peak)
        hump = int(np.nanargmax(dg.dPdG))
        anchor = min(hump + max(2, len(dg.G) // 10), len(dg.G) - 2)
        slope = _local_slope(dg.G, res.resampled.p, anchor, half=4)
        state.eff_isip_line = TangentPick(anchor_x=float(dg.G[anchor]),
                                          anchor_y=float(res.resampled.p[anchor]), slope=slope)
        # compliance contact at the dP/dG hump peak
        state.contact_G = float(dg.G[hump])
        # tangent closure from the through-origin departure
        cslope, idep = interpret.suggest_closure_tangent(dg.G, dg.GdPdG)
        state.closure_slope = cslope
        state.closure_G = float(dg.G[idep])
        # late-time windows for log-log + pore pressure
        if len(dg.t) > 6:
            win = (float(dg.t[int(len(dg.t) * 0.6)]), float(dg.t[-1]))
            state.loglog_window = win
            state.pp_window = win
