"""Interactive picking on the matplotlib canvas.

Generic controllers capture mouse events and forward data coordinates (or, for the anchored-line
controllers, a finished line/point geometry) to a callback. Pure ``commit_*`` functions translate
that geometry into a ``PickState`` change; they touch no matplotlib object, so they are unit-
testable without a canvas. Everything else here depends only on matplotlib (never Tkinter), so the
picking layer is reusable if the shell is ported.

Interaction model: drag-to-move for lines/points (snap-to-sample where a backing curve exists),
drag-select for windows. Every controller hit-tests through its own Axes' pixel transforms
(``_axes_contains_pixel`` / ``_data_from_pixel``) rather than ``event.inaxes`` -- a twinned Axes
(e.g. the injection's rate ``twinx``) owns ``inaxes`` over the shared region, so an identity check
against a specific Axes would never match. See ``test_injection_rate_twin_owns_inaxes_regression``.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
from matplotlib.backend_tools import Cursors
from matplotlib.patches import Rectangle
from matplotlib.widgets import SpanSelector

from . import interpret
from .model import NO_CONTACT_SCENARIOS, DerivedResults, PickState, TangentPick
from .io_load import TestData


# --------------------------------------------------------------------------------------------------
# hover cursor mapping -- shared by every controller's hover_kind()/active_kind() and
# HoverCursorController. Uses matplotlib's backend-agnostic Cursors enum + canvas.set_cursor, which
# FigureCanvasBase implements as a no-op (so the Agg-backed test suite never touches Tkinter) and
# FigureCanvasTkAgg implements for real.
# --------------------------------------------------------------------------------------------------
_HOVER_CURSORS = {
    "anchor": Cursors.RESIZE_HORIZONTAL,   # slide along the backing curve
    "body": Cursors.MOVE,                  # pan the line
    "end": Cursors.HAND,                   # rotate about the anchor
    "line": Cursors.RESIZE_HORIZONTAL,     # drag a vertical injection-start/shut-in line
    "point": Cursors.HAND,                 # drag a contact/closure marker
}


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


def _pixel_xs(ax, x_arr: np.ndarray) -> np.ndarray:
    """On-screen (transData) x-pixel for every sample in ``x_arr``, one vectorized transform.
    Shared by ``_nearest_index_by_pixel`` (recomputes every call) and AnchorLineController's
    press-time cache, which computes it once per gesture -- view limits are frozen for a
    gesture's duration (sliders grab the mouse), so the array is drag-invariant."""
    xs = np.asarray(x_arr, dtype=float)
    return ax.transData.transform(np.column_stack([xs, np.zeros_like(xs)]))[:, 0]


def _nearest_index_by_pixel(ax, x_arr: np.ndarray, event) -> int:
    """Index into ``x_arr`` whose on-screen (transData) x-pixel is nearest ``event.x``. Matching
    by pixel rather than raw data-x keeps snapping correct on a log-scaled axis too."""
    px = _pixel_xs(ax, x_arr)
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
    the min-dP/dG marker sitting near the contact marker).

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


class ModifierSpanController:
    """Modifier-armed (default Shift) horizontal drag-select on a target Axes; forwards the
    sorted ``(lo, hi)`` window on release. Backs the G-function step's window-correction gesture
    (Shift+drag re-finds the min-dP/dG pick, or the C-B inflection, within a hand-picked
    interval) -- a plain drag never captures, so ordinary point/line dragging on the same Axes is
    untouched by this controller sitting alongside it.

    Press captures only when ``modifier`` is present in the event's held-modifier set (matplotlib
    reports it as a "+"-joined string on ``event.key``, e.g. ``"shift"`` or
    ``"shift+control"`` -- absent entirely, i.e. ``None``, when nothing is held), the pixel is
    inside ``ax`` via ``_axes_contains_pixel`` (never ``event.inaxes`` -- see module docstring),
    and ``guard()`` is not active; then ``gate.try_claim(self)`` -- same claim-after-hit-test
    contract as every other controller here, so a Shift-press claims a shared gate before sibling
    point controllers get a chance when this is registered ahead of them.

    Motion updates a translucent span patch (same look as ``SpanController``'s ``axvspan``)
    between the press-x and the cursor-x (via ``_data_from_pixel``). The patch is created ONCE at
    press-capture time (zero-width at the press-x) rather than being removed and rebuilt every
    motion event -- a ``Rectangle`` added via ``ax.add_patch`` with
    ``transform=ax.get_xaxis_transform(which="grid")`` (x in data coords, y spanning 0..1 axes
    coords -- exactly what ``Axes.axvspan`` builds internally, dataLim-protection dance included),
    so motion only needs to ``patch.set_x(lo)``/``patch.set_width(hi - lo)``. When the canvas
    supports blitting (``canvas.supports_blit``), the patch is set animated and a single
    ``canvas.draw()`` + background snapshot happen at press, followed by one immediate
    restore/redraw/blit so the zero-width patch is painted at press rather than staying invisible
    (harmless here since it's zero-width anyway, but keeps the pattern uniform with the other
    three controllers) -- then motion only restores the snapshot, redraws the patch, and blits --
    never a full ``draw_idle()``. Without blit support, motion falls back to ``draw_idle()`` after
    updating the patch's geometry (no remove/re-add either way). Release removes the patch either
    way (the selection disappears) and calls ``on_span(lo, hi)`` with the sorted bounds, but only
    when ``hi > lo``. ``disconnect()`` mid-drag also releases the gate and resets ``_press_x``, so
    a stray motion event after disconnection can't raise on the now-gone patch.
    """

    def __init__(self, canvas, ax, on_span: Callable[[float, float], None],
                 modifier: str = "shift", gate: Optional[_CaptureGate] = None, guard=None):
        self.canvas = canvas
        self.ax = ax
        self.on_span = on_span
        self.modifier = modifier
        self.gate = gate if gate is not None else _CaptureGate()
        self.guard = guard or (lambda: False)

        self._press_x: Optional[float] = None
        self._patch = None
        self._bg = None  # blit background snapshot for the current gesture
        self._cids = [
            canvas.mpl_connect("button_press_event", self._on_press),
            canvas.mpl_connect("motion_notify_event", self._on_motion),
            canvas.mpl_connect("button_release_event", self._on_release),
        ]

    def _on_press(self, event):
        if event.button != 1 or not _axes_contains_pixel(self.ax, event):
            return
        # event.key only reflects a modifier when the canvas had keyboard focus at key-press time;
        # event.modifiers comes from the mouse event's own state, so it works on the first click.
        held = set((event.key or "").split("+")) | set(getattr(event, "modifiers", None) or ())
        if self.modifier not in held:
            return
        if self.guard():
            return
        if not self.gate.try_claim(self):
            return
        x, _ = _data_from_pixel(self.ax, event)
        self._press_x = x

        self._patch = Rectangle((x, 0.0), 0.0, 1.0, alpha=0.2, facecolor="tab:orange")
        self._patch.set_transform(self.ax.get_xaxis_transform(which="grid"))
        # Mirror Axes.axvspan: adding an xaxis-transformed Rectangle can otherwise perturb
        # dataLim's y-interval, so snapshot/restore it around add_patch.
        iy = self.ax.dataLim.intervaly.copy()
        my = self.ax.dataLim.minposy
        self.ax.add_patch(self._patch)
        self.ax.dataLim.intervaly = iy
        self.ax.dataLim.minposy = my
        self.ax._request_autoscale_view("x")

        if getattr(self.canvas, "supports_blit", False):
            self._patch.set_animated(True)
            self.canvas.draw()
            self._bg = self.canvas.copy_from_bbox(self.ax.bbox)
            # First paint: the draw() above skipped the now-animated patch, so without this the
            # patch is invisible from press until the first motion event.
            self.canvas.restore_region(self._bg)
            self.ax.draw_artist(self._patch)
            self.canvas.blit(self.ax.bbox)
        else:
            self._bg = None

    def _on_motion(self, event):
        if self._press_x is None or self._patch is None or event.x is None or event.y is None:
            return
        x, _ = _data_from_pixel(self.ax, event)
        lo, hi = sorted((self._press_x, x))
        self._patch.set_x(lo)
        self._patch.set_width(hi - lo)
        if self._bg is not None:
            self.canvas.restore_region(self._bg)
            self.ax.draw_artist(self._patch)
            self.canvas.blit(self.ax.bbox)
        else:
            self.canvas.draw_idle()

    def _on_release(self, event):
        if self._press_x is None:
            return
        press_x = self._press_x
        self._press_x = None
        if self._patch is not None:
            self._patch.remove()
            self._patch = None
        self._bg = None
        self.gate.release()
        if event.x is not None and event.y is not None:
            x, _ = _data_from_pixel(self.ax, event)
        else:
            x = press_x
        self.canvas.draw_idle()
        lo, hi = sorted((press_x, x))
        if hi > lo:
            self.on_span(lo, hi)

    def disconnect(self):
        for cid in self._cids:
            self.canvas.mpl_disconnect(cid)
        self._cids = []
        if self._press_x is not None:
            self.gate.release()
        self._press_x = None
        if self._patch is not None:
            self._patch.remove()
            self._patch = None
        self._bg = None

    # ---- hover probes (no side effects) -- see HoverCursorController ----
    def hover_kind(self, event) -> Optional[str]:
        """Always None: the modifier gesture needs a held key that a plain motion event carries
        no information about, so there is no hover affordance to show before a Shift+press."""
        return None

    def active_kind(self) -> Optional[str]:
        """"body" (same kind ``AnchorLineController`` reports for its own pan-the-whole-line
        drag) while a span drag is in progress, else None -- lets ``HoverCursorController`` hold
        a move cursor for the duration of the gesture instead of raising on the probe it doesn't
        implement."""
        return "body" if self._press_x is not None else None


class DragLineController:
    """Drag gid-tagged vertical lines within an Axes; commit the released x per gid.

    ``handlers`` maps an axvline gid to ``on_release(x_data)``. During a drag the line is moved
    live (no recompute); on release the matching handler is called with the final x. ``guard()``
    returning True blocks capture (e.g. while the toolbar zoom/pan mode is active).

    Blits during the drag when the canvas supports it (``canvas.supports_blit``): press sets the
    captured line animated, does one ``canvas.draw()``, snapshots the background, and immediately
    restores/redraws/blits once more so the line stays visible at press instead of going blank
    until the first motion event (the preceding ``draw()`` skips animated artists); motion
    restores that snapshot, redraws the line, and blits instead of a full ``draw_idle()``; release
    un-animates the line and issues one final ``draw_idle()``. Falls back to a plain
    ``draw_idle()`` per motion event when blitting isn't supported. ``disconnect()`` mid-drag
    un-animates the line, clears the background snapshot, and releases the gate, so an active
    line never stays animated (and so invisible to any later full draw) past disconnection. Same
    pattern as ``AnchorLineController``.

    Hit-tests and reads the cursor through ``_axes_contains_pixel``/``_data_from_pixel`` (this
    axes' own transforms) rather than ``event.inaxes``/``event.xdata``: an overlaid twin axes
    (e.g. the injection's rate ``twinx``) owns ``inaxes`` over the shared region, so an identity
    check against ``self.ax`` would never match. Twinned axes share the x-scale, so converting the
    event pixel through ``self.ax.transData`` yields the correct data-x regardless.

    ``gate`` (a shared ``_CaptureGate``) is claimed only *after* this controller's own hit-test at
    press finds a line, and released on button-release -- same claim-after-hit-test contract as
    ``AnchorLineController``/``DraggablePointController``, so this controller can share a gate
    with either of them if a step ever overlays a draggable line on top of a draggable point or
    anchor. Defaults to a private gate (no sharing) when omitted, which covers every current
    usage: the injection step's start/shut-in vlines and the Overview step's tail-trim line each
    have their axes to themselves.
    """

    def __init__(self, canvas, ax, handlers, guard=None, tol_px: float = 6.0,
                 gate: Optional[_CaptureGate] = None):
        self.canvas = canvas
        self.ax = ax
        self.handlers = handlers
        self.guard = guard or (lambda: False)
        self.tol_px = tol_px
        self.gate = gate if gate is not None else _CaptureGate()
        self._active = None
        self._bg = None  # blit background snapshot for the current gesture
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
        if best is None:
            return
        if not self.gate.try_claim(self):
            return
        self._active = best

        # Blitting: one full draw now (instead of one per motion event), then snapshot the
        # background so motion only restores it and redraws the active line.
        if getattr(self.canvas, "supports_blit", False):
            self._active.set_animated(True)
            self.canvas.draw()
            self._bg = self.canvas.copy_from_bbox(self.ax.bbox)
            # First paint: the draw() above skipped the now-animated line, so without this the
            # line is invisible from press until the first motion event.
            self.canvas.restore_region(self._bg)
            self.ax.draw_artist(self._active)
            self.canvas.blit(self.ax.bbox)
        else:
            self._bg = None

    def _on_motion(self, event):
        if self._active is None or event.x is None or event.y is None:
            return
        x, _ = _data_from_pixel(self.ax, event)
        self._active.set_xdata([x, x])
        if self._bg is not None:
            self.canvas.restore_region(self._bg)
            self.ax.draw_artist(self._active)
            self.canvas.blit(self.ax.bbox)
        else:
            self.canvas.draw_idle()

    def _on_release(self, event):
        if self._active is None:
            return
        line = self._active
        self._active = None
        line.set_animated(False)
        self._bg = None
        self.gate.release()
        if event.x is not None and event.y is not None:
            x, _ = _data_from_pixel(self.ax, event)
        else:
            x = float(line.get_xdata()[0])
        handler = self.handlers.get(line.get_gid())
        if handler is not None:
            handler(float(x))
        # The last blit already painted the line at its final x; this un-animates it for any
        # subsequent full draw (e.g. the commit-path refresh most handlers trigger) rather than
        # leaving an animated artist behind.
        self.canvas.draw_idle()

    def disconnect(self):
        for cid in self._cids:
            self.canvas.mpl_disconnect(cid)
        self._cids = []
        if self._active is not None:
            self._active.set_animated(False)
            self._active = None
            self._bg = None
            self.gate.release()

    # ---- hover probes (no side effects) -- see HoverCursorController ----
    def hover_kind(self, event) -> Optional[str]:
        """"line" if the event pixel is within ``tol_px`` of one of this controller's gid lines
        (same x-distance test as ``_on_press``), else None."""
        if not _axes_contains_pixel(self.ax, event) or event.x is None:
            return None
        for line in self._lines():
            lx = line.get_xdata()[0]
            px = self.ax.transData.transform((lx, 0.0))[0]
            if abs(px - event.x) <= self.tol_px:
                return "line"
        return None

    def active_kind(self) -> Optional[str]:
        return "line" if self._active is not None else None


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
         artist is present. Hits if the event pixel is within ``tol_px`` of either the anchor
         *point* or the tick's own drawn *segment* (point-to-segment distance, via
         ``_segment_distance_px``) -- the tick is often drawn taller on screen than ``tol_px``, so
         testing only its center point would miss presses on the rest of the visible tick.
      2. either segment endpoint -- only if ``allow_rotate``.
      3. the segment body (point-to-segment pixel distance) -- only if ``allow_body``. If
         ``allow_body`` is false but ``allow_rotate`` is true, this same body distance test is
         still run as a fallback and reported as "end" rather than "body" -- see "Pinned mode"
         below.

    Motion (nothing is written to ``PickState`` until release; artists update live via
    ``canvas.draw_idle()``):
      - **"anchor"**: snaps to the nearest ``curve`` sample by x-pixel and refits
        ``(anchor_x, anchor_y, slope) = interpret.tangent_from_index(curve_x, curve_y, idx,
        anchor_half)``.
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
      does not move while rotating. When ``pin_x`` is set, the dashed extension is instead
      re-anchored to that x every motion event rather than replaying its press-time offsets, so
      it stays glued to a reference vertical (e.g. shut-in) throughout the drag; an optional
      "point" gid (e.g. a value dot) rides along too, redrawn each motion at
      (``pin_x``, the drag's current line value there) instead of freezing at its press position.

    On release, ``commit_fn(kind, anchor_x, anchor_y, slope)`` is called exactly once with the
    *final* geometry -- never the raw cursor position -- where ``kind`` is one of "anchor" /
    "body" / "end". Pinned mode (``curve=None, allow_anchor=False, allow_body=False``) leaves only
    "end" reachable: a through-origin (or otherwise fixed-anchor) rotate-only line, with the
    anchor staying at whatever ``get_pick()`` returns (e.g. anchor (0, 0)). Because the far
    endpoint of such a line is often clipped off-screen, a press anywhere on the visible body also
    hits as "end" in this mode (see hit priority #3 above), so the whole line is grabbable and a
    body drag rotates about the anchor exactly like an endpoint drag would.

    ``readout_fn(kind, anchor_x, anchor_y, slope) -> str | None``, if given, drives a small Text
    artist: created on press, updated every motion, removed on release/disconnect (returning
    ``None`` from it also removes the text early, mid-drag).

    ``gate`` (a shared ``_CaptureGate``) is claimed only *after* this controller's own hit-test at
    press succeeds, and released on button-release -- so a miss here never blocks a sibling
    controller sharing the gate, and whichever controller's hit-test succeeds first (in
    ``mpl_connect``/construction order) wins the gesture. Defaults to a private gate (no sharing)
    when omitted. ``disconnect()`` mid-drag mirrors release's cleanup: it un-animates the blit
    artists, clears the background snapshot, and releases the gate, so a dragged line never stays
    animated (and so invisible to any later full draw) past disconnection.
    """

    _ROTATE_DX_EPS_PX = 1.0

    def __init__(self, canvas, ax, gids: dict, get_pick: Callable[[], Optional[TangentPick]],
                 commit_fn: Callable[[str, float, float, float], None], curve=None,
                 anchor_half: int = 4, allow_anchor: bool = True, allow_body: bool = True,
                 allow_rotate: bool = True, tol_px: float = 12.0, readout_fn=None,
                 gate: Optional[_CaptureGate] = None, pin_x: Optional[float] = None):
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
        self.pin_x = pin_x

        self._active: Optional[str] = None
        self._press_anchor = None   # (anchor_x, anchor_y, slope) snapshot at press
        self._press_data = None     # (x, y) data coords under the cursor at press
        self._seg_offsets = None
        self._ext_offsets = None
        self._tick_orig = None
        self._final = None          # (kind, anchor_x, anchor_y, slope) -- last-known-good geometry
        self._readout = None
        self._curve_px = None       # per-gesture cache of _pixel_xs(curve[0]) for anchor drags
        self._bg = None            # blit background snapshot for the current gesture
        self._blit_artists = None  # animated artists for the current gesture
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
                txs, tys = tick.get_xdata(), tick.get_ydata()
                d = _segment_distance_px(self.ax, (txs[0], tys[0]), (txs[-1], tys[-1]), event)
                if d <= self.tol_px:
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
        elif self.allow_rotate:
            # Pinned mode (allow_body=False): the through-origin segment's far endpoint is often
            # clipped off-screen, leaving no reachable endpoint to grab. Fall back to a
            # point-to-segment test against the visible body and still hand back "end" -- a press
            # anywhere on the line rotates it about the (unchanged) anchor.
            xs, ys = segment.get_xdata(), segment.get_ydata()
            d = _segment_distance_px(self.ax, (xs[0], ys[0]), (xs[-1], ys[-1]), event)
            if d <= self.tol_px:
                return "end"
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
        self._curve_px = (_pixel_xs(self.ax, self.curve[0])
                          if kind == "anchor" and self.curve is not None else None)

        if self.readout_fn is not None:
            text = self.readout_fn(kind, pick.anchor_x, pick.anchor_y, pick.slope)
            if text is not None:
                self._readout = self.ax.text(0.02, 0.95, text, transform=self.ax.transAxes,
                                             fontsize=8, va="top", ha="left")

        # Blitting: one full draw now (instead of one per motion event), then snapshot the
        # background so motion only restores it and redraws the managed artists. The readout
        # (if any) was created above, so it joins the animated set and stays out of the
        # captured background.
        if getattr(self.canvas, "supports_blit", False):
            self._blit_artists = [a for a in (
                segment, self._artist(self.gids.get("tick")),
                self._artist(self.gids.get("extension")), self._artist(self.gids.get("point")),
                self._readout) if a is not None]
            for artist in self._blit_artists:
                artist.set_animated(True)
            self.canvas.draw()
            self._bg = self.canvas.copy_from_bbox(self.ax.bbox)
            # First paint: the draw() above skipped the now-animated artists, so without this
            # they are invisible from press until the first motion event.
            self.canvas.restore_region(self._bg)
            for artist in self._blit_artists:
                self.ax.draw_artist(artist)
            self.canvas.blit(self.ax.bbox)
        else:
            self._blit_artists = None
            self._bg = None

    def _on_motion(self, event):
        if self._active is None or event.x is None or event.y is None:
            return
        kind = self._active
        ax0, ay0, slope0 = self._press_anchor
        if kind == "anchor":
            x_arr, y_arr = self.curve
            if self._curve_px is not None:
                idx = int(np.nanargmin(np.abs(self._curve_px - event.x)))
            else:
                idx = _nearest_index_by_pixel(self.ax, x_arr, event)
            ax1, ay1, slope1 = interpret.tangent_from_index(x_arr, y_arr, idx, self.anchor_half)
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
                removed = self._readout
                self._readout.remove()
                self._readout = None
                if self._blit_artists is not None:
                    # The captured background still shows the now-gone readout; recapture a
                    # clean one before the next blit rather than leave a ghost image.
                    self._blit_artists = [a for a in self._blit_artists if a is not removed]
                    self.canvas.draw()
                    self._bg = self.canvas.copy_from_bbox(self.ax.bbox)
            else:
                self._readout.set_text(text)

        if self._bg is not None and self._blit_artists is not None:
            self.canvas.restore_region(self._bg)
            for artist in self._blit_artists:
                self.ax.draw_artist(artist)
            self.canvas.blit(self.ax.bbox)
        else:
            self.canvas.draw_idle()

    def _apply_geometry(self, ax1, ay1, slope1):
        segment = self._artist(self.gids.get("segment"))
        seg_xs = None
        if segment is not None and self._seg_offsets is not None:
            seg_xs = [ax1 + t for t in self._seg_offsets]
            segment.set_data(seg_xs, [ay1 + slope1 * t for t in self._seg_offsets])
        ext = self._artist(self.gids.get("extension"))
        if ext is not None:
            if self.pin_x is not None and seg_xs is not None:
                # Mirror plots._draw_tangent_construction: run the dashed extension from the
                # pinned reference to the *current* dragged segment's nearest endpoint, so it
                # never drifts off the reference vertical mid-drag.
                near_x = min(seg_xs, key=lambda x: abs(x - self.pin_x))
                ext_x = [self.pin_x, near_x]
                ext.set_data(ext_x, [ay1 + slope1 * (x - ax1) for x in ext_x])
            elif self._ext_offsets is not None:
                ext.set_data([ax1 + t for t in self._ext_offsets],
                             [ay1 + slope1 * t for t in self._ext_offsets])
        tick = self._artist(self.gids.get("tick"))
        if tick is not None and self._tick_orig is not None and self._active in ("anchor", "body"):
            ax0, ay0, _ = self._press_anchor
            dx, dy = ax1 - ax0, ay1 - ay0
            txo, tyo = self._tick_orig
            tick.set_data([x + dx for x in txo], [y + dy for y in tyo])
        point = self._artist(self.gids.get("point"))
        if point is not None and self.pin_x is not None:
            # A value dot pinned to the same reference x as the extension (e.g. the apparent-ISIP
            # dot): its y is the drag's current line evaluated at pin_x, for every drag kind --
            # anchor, body, and rotate all change that value.
            point.set_data([self.pin_x], [ay1 + slope1 * (self.pin_x - ax1)])

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
        self._curve_px = None
        if self._readout is not None:
            self._readout.remove()
            self._readout = None
        if self._blit_artists is not None:
            for artist in self._blit_artists:
                artist.set_animated(False)
            self._blit_artists = None
        self._bg = None
        self.gate.release()
        self.commit_fn(kind, float(ax1), float(ay1), float(slope1))
        self.canvas.draw_idle()

    def disconnect(self):
        for cid in self._cids:
            self.canvas.mpl_disconnect(cid)
        self._cids = []
        if self._readout is not None:
            self._readout.remove()
            self._readout = None
        if self._active is not None:
            if self._blit_artists is not None:
                for artist in self._blit_artists:
                    artist.set_animated(False)
                self._blit_artists = None
            self._bg = None
            self._active = None
            self.gate.release()

    # ---- hover probes (no side effects) -- see HoverCursorController ----
    def hover_kind(self, event) -> Optional[str]:
        """Reuses ``_hit_test`` to report what a press at ``event`` would capture ("anchor" /
        "end" / "body"), without capturing anything. None when there's no pick or no "segment"
        artist to test against (mirrors ``_on_press``'s own no-op guards)."""
        if not _axes_contains_pixel(self.ax, event):
            return None
        pick = self.get_pick()
        if pick is None:
            return None
        segment = self._artist(self.gids.get("segment"))
        if segment is None:
            return None
        return self._hit_test(event, pick, segment)

    def active_kind(self) -> Optional[str]:
        return self._active


class DraggablePointController:
    """Drag a single gid-tagged marker, snapped along a backing curve.

    Press within ``tol_px`` (this axes' own transforms, never ``event.inaxes``) of the gid-tagged
    marker's current position captures it. Motion snaps the marker to the nearest
    ``(curve_x, curve_y)`` sample by x-pixel (``_nearest_index_by_pixel``) and moves it live. On
    release, ``commit_fn(float(snapped_x))`` is called once with the final snapped x. A missing
    marker artist (gid not found) makes press a no-op.

    ``vline_gid`` optionally names a companion gid-tagged vertical line (an ``axvline``) that
    tracks the same pick, e.g. the tangent page's dotted closure line. When given, a press within
    ``tol_px`` of that line's x-pixel (same test as ``DragLineController``) also captures the
    marker -- so the whole line body is draggable, not just the dot -- and motion moves the vline
    to the same snapped x alongside the marker. A missing marker artist still makes press a no-op
    even if the vline would hit, since the marker is what gets dragged. Hit-testing looks up the
    vline on ``self.ax`` (the axes this controller was built against), so it only works when the
    marker and vline share an axes.

    ``gate`` follows the same claim-after-hit-test / release-on-release contract as
    ``AnchorLineController`` -- see ``_CaptureGate``.

    Blits during the drag when the canvas supports it (``canvas.supports_blit``): press sets the
    marker (and the companion vline, when ``vline_gid`` is set and present) animated, does one
    ``canvas.draw()``, snapshots the background, and immediately restores/redraws/blits once more
    so the marker stays visible at press instead of going blank until the first motion event (the
    preceding ``draw()`` skips animated artists); motion restores that snapshot, redraws the
    animated artists, and blits instead of a full ``draw_idle()``; release un-animates them and
    issues one final ``draw_idle()``. Falls back to a plain ``draw_idle()`` per motion event when
    blitting isn't supported. ``disconnect()`` mid-drag un-animates the managed artists, clears the
    background snapshot, and releases the gate, so a dragged marker never stays animated (and so
    invisible to any later full draw) past disconnection. Same pattern as ``AnchorLineController``.
    """

    def __init__(self, canvas, ax, gid: str, curve_x: np.ndarray, curve_y: np.ndarray,
                 commit_fn: Callable[[float], None], tol_px: float = 8.0,
                 vline_gid: Optional[str] = None, gate: Optional[_CaptureGate] = None):
        self.canvas = canvas
        self.ax = ax
        self.gid = gid
        self.curve_x = curve_x
        self.curve_y = curve_y
        self.commit_fn = commit_fn
        self.tol_px = tol_px
        self.vline_gid = vline_gid
        self.gate = gate if gate is not None else _CaptureGate()

        self._dragging = False
        self._final_x = None
        self._bg = None            # blit background snapshot for the current gesture
        self._blit_artists = None  # animated artists (marker + optional vline) for the gesture
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

    def _vline(self):
        if self.vline_gid is None:
            return None
        for line in self.ax.get_lines():
            if line.get_gid() == self.vline_gid:
                return line
        return None

    def _hit_marker(self, event) -> bool:
        marker = self._artist()
        if marker is None:
            return False
        xs, ys = marker.get_xdata(), marker.get_ydata()
        if len(xs) == 0:
            return False
        mpx = self.ax.transData.transform((xs[0], ys[0]))
        return math.hypot(event.x - mpx[0], event.y - mpx[1]) <= self.tol_px

    def _hit_vline(self, event) -> bool:
        vline = self._vline()
        if vline is None:
            return False
        vx = vline.get_xdata()[0]
        px = self.ax.transData.transform((vx, 0.0))[0]
        return abs(px - event.x) <= self.tol_px

    def _on_press(self, event):
        if event.button != 1 or not _axes_contains_pixel(self.ax, event):
            return
        marker = self._artist()
        if marker is None:
            return
        xs = marker.get_xdata()
        if len(xs) == 0:
            return
        if not (self._hit_marker(event) or self._hit_vline(event)):
            return
        if not self.gate.try_claim(self):
            return
        self._dragging = True
        self._final_x = float(xs[0])

        # Blitting: one full draw now (instead of one per motion event), then snapshot the
        # background so motion only restores it and redraws the managed artists.
        if getattr(self.canvas, "supports_blit", False):
            vline = self._vline()
            self._blit_artists = [a for a in (marker, vline) if a is not None]
            for artist in self._blit_artists:
                artist.set_animated(True)
            self.canvas.draw()
            self._bg = self.canvas.copy_from_bbox(self.ax.bbox)
            # First paint: the draw() above skipped the now-animated artists, so without this
            # they are invisible from press until the first motion event.
            self.canvas.restore_region(self._bg)
            for artist in self._blit_artists:
                self.ax.draw_artist(artist)
            self.canvas.blit(self.ax.bbox)
        else:
            self._blit_artists = None
            self._bg = None

    def _on_motion(self, event):
        if not self._dragging or event.x is None or event.y is None:
            return
        idx = _nearest_index_by_pixel(self.ax, self.curve_x, event)
        sx = float(self.curve_x[idx])
        marker = self._artist()
        if marker is not None:
            marker.set_data([sx], [float(self.curve_y[idx])])
        vline = self._vline()
        if vline is not None:
            vline.set_xdata([sx, sx])
        self._final_x = sx
        if self._bg is not None and self._blit_artists is not None:
            self.canvas.restore_region(self._bg)
            for artist in self._blit_artists:
                self.ax.draw_artist(artist)
            self.canvas.blit(self.ax.bbox)
        else:
            self.canvas.draw_idle()

    def _on_release(self, event):
        if not self._dragging:
            return
        self._dragging = False
        if self._blit_artists is not None:
            for artist in self._blit_artists:
                artist.set_animated(False)
            self._blit_artists = None
        self._bg = None
        self.gate.release()
        self.commit_fn(float(self._final_x))
        self.canvas.draw_idle()

    def disconnect(self):
        for cid in self._cids:
            self.canvas.mpl_disconnect(cid)
        self._cids = []
        if self._dragging:
            if self._blit_artists is not None:
                for artist in self._blit_artists:
                    artist.set_animated(False)
                self._blit_artists = None
            self._bg = None
            self._dragging = False
            self.gate.release()

    # ---- hover probes (no side effects) -- see HoverCursorController ----
    def hover_kind(self, event) -> Optional[str]:
        """"point" if the event pixel is within ``tol_px`` of this controller's marker (same test
        as ``_on_press``); else "line" if within ``tol_px`` of the companion vline (when
        ``vline_gid`` is set); else None."""
        if not _axes_contains_pixel(self.ax, event):
            return None
        if self._hit_marker(event):
            return "point"
        if self._hit_vline(event):
            return "line"
        return None

    def active_kind(self) -> Optional[str]:
        return "point" if self._dragging else None


class HoverCursorController:
    """Sets the canvas cursor to indicate what a press at the current pointer position would do,
    by probing an ordered list of controllers (first hit wins). Each controller in ``controllers``
    must expose ``hover_kind(event) -> Optional[str]`` and ``active_kind() -> Optional[str]``
    (``AnchorLineController``, ``DragLineController``, ``DraggablePointController`` all do).

    On every ``motion_notify_event``: if any controller reports an in-progress drag via
    ``active_kind()``, that kind's cursor wins outright (held even if the pointer strays over empty
    space mid-drag, e.g. panning); otherwise the first controller (in list order) whose
    ``hover_kind(event)`` is not None sets the cursor; otherwise the cursor falls back to
    ``Cursors.POINTER``. ``_HOVER_CURSORS`` maps kind -> ``matplotlib.backend_tools.Cursors``.

    Uses the backend-agnostic ``canvas.set_cursor`` -- a no-op on ``FigureCanvasBase`` (so this
    stays crash-free under the Agg backend the test suite runs on) and real on
    ``FigureCanvasTkAgg`` -- so this module stays Tkinter-free. ``set_cursor`` is only called when
    the resolved cursor differs from the previous event's, to avoid needless churn.
    """

    def __init__(self, canvas, controllers):
        self.canvas = canvas
        self.controllers = list(controllers)
        self._last_cursor = None
        self._cids = [canvas.mpl_connect("motion_notify_event", self._on_motion)]

    def _resolve_kind(self, event) -> Optional[str]:
        for ctrl in self.controllers:
            kind = ctrl.active_kind()
            if kind is not None:
                return kind
        for ctrl in self.controllers:
            kind = ctrl.hover_kind(event)
            if kind is not None:
                return kind
        return None

    def _on_motion(self, event):
        cursor = _HOVER_CURSORS.get(self._resolve_kind(event), Cursors.POINTER)
        if cursor is not self._last_cursor:
            self.canvas.set_cursor(cursor)
            self._last_cursor = cursor

    def disconnect(self):
        for cid in self._cids:
            self.canvas.mpl_disconnect(cid)
        self._cids = []


# --------------------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------------------
def _nearest(arr: np.ndarray, value: float) -> int:
    return int(np.nanargmin(np.abs(np.asarray(arr, dtype=float) - value)))


# --------------------------------------------------------------------------------------------------
# step pick handlers  (mutate PickState in place)
# --------------------------------------------------------------------------------------------------
def handle_loglog_span(state: PickState, lo: float, hi: float) -> None:
    state.loglog_window = (float(lo), float(hi))


def handle_pp_span(state: PickState, lo: float, hi: float) -> None:
    # Span arrives in the plotted transform domain (x = t**expo); pp_window is shut-in seconds.
    if hi <= 0:
        return
    expo = -0.5 if state.pp_axis == "tm12" else -1.0
    t_lo = float(hi) ** (1.0 / expo)
    t_hi = float(lo) ** (1.0 / expo) if lo > 0 else float("inf")
    state.pp_window = (t_lo, t_hi)


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
    """Commit the apparent-ISIP tangent (BHP vs time-seconds) after an AnchorLineController drag."""
    if kind == "anchor":
        idx = _nearest(td.t_s, anchor_x)
        ax_, ay_, sl_ = interpret.tangent_from_index(td.t_s, res.bhp_all, idx,
                                                     half=interpret.ISIP_ANCHOR_HALF)
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


def commit_closure_line(state: PickState, res: DerivedResults,
                        kind: str, anchor_x: float, anchor_y: float, slope: float) -> None:
    """Commit the through-origin tangent-closure line (tangent-method step). The controller for
    this line is always constructed pinned (``curve=None, allow_anchor=False, allow_body=False``),
    so only "end" is ever reached and only the slope is meaningful -- the anchor stays fixed at
    the origin. ``res``/``kind``/``anchor_x``/``anchor_y`` are accepted for signature symmetry
    with the other AnchorLineController commit functions but are not needed here."""
    state.closure_slope = float(slope)


def commit_min_dpdg_point(state: PickState, x: float) -> None:
    """DraggablePointController commit for the min-dP/dG pick (G-function step, dP/dG twin axis).
    A diagnostic pick only -- it does not feed the effective-ISIP tangent (that's contact_G,
    see commit_contact_point / model.compute_all)."""
    state.min_dpdg_G = float(x)


def commit_contact_point(state: PickState, x: float) -> None:
    """DraggablePointController commit for the compliance-method contact pick (G-function step).
    The effective-ISIP tangent is derived from this in model.compute_all, not stored here."""
    state.contact_G = float(x)


def commit_closure_point(state: PickState, x: float) -> None:
    """DraggablePointController commit for the tangent-method closure (departure) pick."""
    state.closure_G = float(x)


def commit_tail_trim(state: PickState, dt: Optional[float], guard_dt: Optional[float] = None) -> None:
    """DragLineController commit for the Overview step's always-visible tail-trim line: ``dt``
    is shut-in-relative seconds, or None to clear the trim (drag released at/past the last
    point). Resets state.tail_trim_reason to "" -- a manual drag or clear always overrides
    whatever auto-attribution put the trim where it was; only seed_tail_trim (immediately after
    calling this itself) sets a non-"" reason back.
    Orthogonal to the closure-scenario flows -- never touched by apply_closure_scenario, the
    triangle-drag commit, or the Shift+drag window commit.

    ``guard_dt`` (the rise guard's boundary, shut-in-relative seconds, or None if it never fired)
    sets state.tail_guard_override: True when this commit lands a real trim strictly past
    guard_dt -- a deliberate drag past the guard, which interpret.resolve_tail_cut_dt then lets
    stick instead of clamping back. Any other case (no trim, no guard, or a trim at/before it)
    leaves it False. The default guard_dt=None means "no guard to compare against", so override
    always resolves False unless a caller passes the real boundary -- seed_tail_trim's own call
    site omits it on purpose (see its docstring), since it never seeds a trim past the guard."""
    state.tail_trim_dt = float(dt) if dt is not None else None
    state.tail_trim_reason = ""
    state.tail_guard_override = (
        state.tail_trim_dt is not None and guard_dt is not None and state.tail_trim_dt > guard_dt
    )


def commit_stiffness_point(state: PickState, x: float) -> None:
    """DragLineController commit for the relative-stiffness upturn pick (stiffness step): the
    picked pressure feeds Shmin(stiffness) = x - 75 psi (model.compute_all,
    interpret.shmin_compliance)."""
    state.stiffness_pick_P = float(x)


def apply_closure_scenario(state: PickState, res: DerivedResults) -> Optional[str]:
    """Re-suggest the contact pick from the just-selected closure scenario (an explicit user
    action, so it may overwrite a previous contact pick). Pure state mutation -- no matplotlib.

    Rules (URTeC-2019-123 / ../CLAUDE.md scenario table):
      - C-A clear: contact = first sample right of the min-dP/dG pick where dP/dG >= 110% of
        the value at that pick. Anchors at ``state.min_dpdg_G`` (suggesting it first if unset)
        so a dragged min pick drives the rule.
      - C-B adequate: contact = the dP/dG inflection (flattest point of the decline). Also seeds
        ``state.min_dpdg_G`` if unset -- for this scenario the triangle is the inflection *seed*,
        not a rel-min pick (decision 4), and it must exist for the marker to be drawn/draggable.
      - C-C no-contact / C-D rapid / C-X uninterpretable: no contact pick -> Shmin(compliance)
        and the *compliance* effective ISIP become None downstream (model.compute_all). The
        tangent effective ISIP
        is unaffected -- it builds off ``closure_G``, so it still feeds the shared
        net-pressure/complexity reference.

    Returns a user-facing hint string when the rule finds nothing (picks left unchanged),
    else None. Degrades to a no-op when diagnostics aren't ready.
    """
    scen = state.closure_scenario
    if not scen:
        return None
    if scen.startswith(NO_CONTACT_SCENARIOS):
        state.contact_G = None
        return None
    dg = res.diagnostics
    if dg is None or len(dg.G) < 3:
        return None
    if scen.startswith(("C-A", "C-B")) and state.min_dpdg_G is None:
        idx = interpret.suggest_min_dpdg_index(dg.G, dg.dPdG)
        state.min_dpdg_G = float(dg.G[idx])
    return re_derive_contact_from_min(state, res)


def re_derive_contact_from_min(state: PickState, res: DerivedResults) -> Optional[str]:
    """Re-derive the contact pick from the current ``state.min_dpdg_G`` under the active closure
    scenario, without moving the min-dP/dG marker itself.

    This is the rule engine shared by ``apply_closure_scenario`` (scenario just picked) and the
    triangle-drag commit path (ui.py's gfunction wiring, decision D4): dragging the triangle
    re-runs the same rule from its new position so the contact pick always matches the
    scenario's answer for wherever the analyst has put the anchor. It is not the *only* rule
    engine for this pick, though: the Shift+drag window-correction gesture
    (``handle_min_dpdg_window``) runs its own parallel version of the same C-A/C-B rules, seeded
    from a hand-picked window instead of a g_min-masked search or a dragged seed, and is the
    complete commit for that gesture -- ui.py does not call this function afterward.

      - C-A clear: nearest-sample lookup of the dragged min, then the +10% rule from there.
      - C-B adequate: the interior inflection of d2P/dG2 nearest the dragged seed.
      - blank / C-C / C-D / C-X / missing min pick or diagnostics: no-op (nothing to re-derive).

    Returns a user-facing hint string on failure (same convention as ``apply_closure_scenario``),
    else None.
    """
    scen = state.closure_scenario
    if not scen or scen.startswith(NO_CONTACT_SCENARIOS) or state.min_dpdg_G is None:
        return None
    dg = res.diagnostics
    if dg is None or len(dg.G) < 3:
        return None
    if scen.startswith("C-A"):
        min_idx = _nearest(dg.G, state.min_dpdg_G)
        idx = interpret.suggest_contact_clear_index(dg.dPdG, min_idx)
        if idx is None:
            return ("dP/dG never rises 10% above the min -- not a clear contact "
                    "(consider C-B or C-C).")
        state.contact_G = float(dg.G[idx])
        return None
    if scen.startswith("C-B"):
        idx = interpret.suggest_contact_inflection_index(
            dg.G, dg.dPdG, seed=state.min_dpdg_G, d2=dg.d2PdG2)
        if idx is None:
            return "No inflection found on dP/dG -- drag the contact marker manually."
        state.contact_G = float(dg.G[idx])
        return None
    return None


def handle_min_dpdg_window(state: PickState, res: DerivedResults, lo: float,
                           hi: float) -> Optional[str]:
    """Shift+drag window-correction commit for the min-dP/dG pick (ui.py's
    ``ModifierSpanController`` on the gfunction step): finds the relevant point within
    ``[lo, hi]`` under the active closure scenario and moves the triangle there. Mirrors
    ``re_derive_contact_from_min``'s scenario dispatch, but the window narrows the *search*
    itself rather than seeding a nearest-sample/seed lookup from an existing pick.

    Unlike the triangle-drag commit path (which moves only the triangle and leaves ui.py to
    call ``re_derive_contact_from_min`` afterward), this is the *complete* commit for the window
    gesture -- it sets both ``state.min_dpdg_G`` and ``state.contact_G`` itself, rather than
    handing off to ``re_derive_contact_from_min``. That re-derive runs the C-A rule from a
    ``g_min``-masked (>= 1.0) search, which can silently re-derive the contact to a DIFFERENT
    inflection/rise point outside a window the analyst deliberately narrowed below G=1.0 --
    the window itself is the more specific answer and must win.

      - C-A clear: the relative minimum of dP/dG within the window
        (``interpret.min_index_in_window``) becomes the triangle; the +10%-rise contact rule
        (``interpret.suggest_contact_clear_index``) then runs from *that* index (unmasked, right
        of it), same as ``re_derive_contact_from_min``'s C-A branch -- just seeded from the
        window-found min instead of a g_min-masked search.
      - C-B adequate: the dP/dG inflection within the window
        (``interpret.suggest_contact_inflection_index`` with ``g_range``) IS the contact -- both
        picks are set to it directly, with no further re-derive over the (possibly sub-1.0)
        window.
      - blank / C-C / C-D / C-X / missing diagnostics: no-op (the controller isn't wired then
        anyway).

    Returns a user-facing hint string when the window (or, for C-A, the rise rule from the
    window's min) holds nothing usable, else None. On a C-A rise-rule failure, ``min_dpdg_G`` is
    still moved to the window's min (``contact_G`` is left unchanged) -- same partial-failure
    shape as the triangle-drag path. The caller (ui.py) does not need to re-derive the contact
    afterward; this function is the whole commit.
    """
    scen = state.closure_scenario
    if not scen or scen.startswith(NO_CONTACT_SCENARIOS):
        return None
    dg = res.diagnostics
    if dg is None or len(dg.G) < 3:
        return None
    if scen.startswith("C-A"):
        idx = interpret.min_index_in_window(dg.G, dg.dPdG, lo, hi)
        if idx is None:
            return "No finite dP/dG sample inside the window."
        state.min_dpdg_G = float(dg.G[idx])
        contact_idx = interpret.suggest_contact_clear_index(dg.dPdG, idx)
        if contact_idx is None:
            return ("dP/dG never rises 10% above the min -- not a clear contact "
                    "(consider C-B or C-C).")
        state.contact_G = float(dg.G[contact_idx])
        return None
    if scen.startswith("C-B"):
        idx = interpret.suggest_contact_inflection_index(
            dg.G, dg.dPdG, g_range=(lo, hi), d2=dg.d2PdG2)
        if idx is None:
            return "No inflection inside the window."
        state.min_dpdg_G = float(dg.G[idx])
        state.contact_G = float(dg.G[idx])
        return None
    return None


_GFUNCTION_HINT_DEFAULT = ("Drag the contact marker (the effective-ISIP tangent follows it) or "
                           "the min-dP/dG marker.")


_MIN_DPDG_WINDOW_HINT = " Shift+drag a window on the plot to re-find it there."


def gfunction_hint_text(scenario: str) -> str:
    """Scenario-aware hint-label text for the G-function step."""
    if scenario.startswith("C-A"):
        return ("Contact auto-positioned at rel-min +10%; drag the triangle to move the anchor."
                + _MIN_DPDG_WINDOW_HINT)
    if scenario.startswith("C-B"):
        return ("The triangle is the inflection seed; drag it to re-find the nearest inflection."
                + _MIN_DPDG_WINDOW_HINT)
    if scenario.startswith("C-X"):
        return ("G-function marked uninterpretable: no compliance, Liberty, variable, or "
                "stiffness Shmin is reported.")
    if scenario.startswith(NO_CONTACT_SCENARIOS):
        return "No contact pick applies for this closure scenario."
    return _GFUNCTION_HINT_DEFAULT


_PP_AXIS_BY_SCENARIO = {"PC-A": "tm12", "PC-B": "tm1", "PC-C": "tm12", "PC-E": "tm12"}
# PC-D "either" and PC-F "none" are intentionally absent -> axis left to the analyst.


def suggest_pp_axis(scenario: str) -> Optional[str]:
    """Pore-pressure axis dictated by a postclosure scenario ("tm12"/"tm1"), or None when
    the scenario leaves the axis to the analyst (unset, PC-D 'either', PC-F 'none')."""
    if not scenario:
        return None
    return _PP_AXIS_BY_SCENARIO.get(scenario[:4])


# --------------------------------------------------------------------------------------------------
# per-step seeding (auto-suggestions used as starting picks)
#
# Each seeder below fires exactly once per step, from ``ui.DfitApp._seed_step`` on that step's
# first visit (never on a revisit). They are made non-destructive anyway -- each early-returns
# if its target pick(s) are already set -- so a state loaded from JSON with real picks but an
# un-visited step (e.g. an old save resumed via ``first_not_visited_step``) is never clobbered by
# arriving at that step. Every seeder also degrades to a no-op (never a crash) when the inputs it
# needs (``res.t_shutin_s``, ``res.diagnostics``, etc.) aren't ready yet -- out-of-order entry into
# a step whose prerequisites weren't picked simply seeds nothing.
# --------------------------------------------------------------------------------------------------
def seed_overview(state: PickState, td: TestData) -> None:
    """The Overview step owns no picks of its own -- it just seeds the injection window early
    (delegating to ``seed_injection``, which is non-destructive) so the start/shut-in reference
    lines exist from step 1, not just once the analyst reaches Injection. Also gives a later
    trim tool a shut-in anchor to work from."""
    seed_injection(state, td)


def seed_tail_trim(state: PickState, td: TestData, res: DerivedResults) -> None:
    """Overview-step tail-trim default: park-and-apply interpret.suggest_tail_trim_dt's cut as a
    real pick on the step's first visit, per ../CLAUDE.md's Tail trim section, rather than leaving
    it a suggestion the analyst has to notice and act on.

    NOT in SEEDERS -- it needs a ``res`` computed *after* the injection window is seeded (the
    ``res`` at the top of ``ui._seed_step`` predates that seed), so ``ui._seed_step`` calls this
    explicitly instead.

    Non-destructive, same contract as every SEEDERS entry: returns immediately if a trim already
    exists (a manual drag, or a reloaded save) or the inputs it needs
    (res.resampled_full/res.t_shutin_s) aren't ready yet."""
    if state.tail_trim_dt is not None:
        return
    if res.resampled_full is None or res.t_shutin_s is None:
        return
    dt_full = res.resampled_full.dt
    post = td.t_s >= res.t_shutin_s
    dt_post = td.t_s[post] - res.t_shutin_s
    p_surface_post = None
    if not state.pressure_is_bhp:  # state, not res -- see model.compute_all's same gate
        p_surface_post = td.pressure_surface(state.channel_config())[post]
        if res.dropout_mask is not None:
            # A masked gauge glitch is not a real sub-floor reading -- suggest_tail_trim_dt
            # already ignores non-finite samples (see model.compute_all's dropout block and
            # ../CLAUDE.md's "Pressure dropouts" section), so this alone keeps a dropout from
            # triggering the "low_pressure" auto-trim.
            p_surface_post = p_surface_post.copy()
            p_surface_post[res.dropout_mask[post]] = np.nan
    cut_dt, reason = interpret.suggest_tail_trim_dt(dt_post, p_surface_post,
                                                     res.resampled_full.guard_dt)
    if reason != "low_pressure":
        # "" (no candidate at all) sets nothing -- the line parks at the end of the data.
        # "rise_guard" sets no pick either: the guard boundary is already the effective cutoff
        # by default (interpret.resolve_tail_cut_dt, with no trim in state) -- only the rendered
        # line position and gray-out need to reflect it (plots.render_overview), not a pick
        # here. Data past the guard may or may not have entered resampled_full (it does whenever
        # a genuine further decline resumes; resample.resample_pressure_increment's
        # stop_at_guard=False), but either way nothing past it is admitted into the diagnostics
        # unless the analyst drags an explicit override (PickState.tail_guard_override).
        return
    # Snap to the last resampled sample STRICTLY BEFORE the cut (side="left", not "right"), and
    # not _nearest -- both of those can leave the crash sample itself in the record, which is the
    # one sample that must go. The resampler keeps a point at every >=30 psi drop, so a crash
    # cliff is almost always kept, and it is usually the LAST point kept (the flat ~0 psi tail
    # after it never drops another 30 psi). "<= cut" would therefore land on dt_full[-1] and hit
    # the bail below on the ordinary crashed record, making this seeder a near-no-op.
    idx = int(np.searchsorted(dt_full, cut_dt, side="left")) - 1
    # Bail rather than clamp. With side="left", idx < 2 happens exactly when fewer than 3
    # resampled samples precede the crash -- clamping idx UP to 2 (the old behavior) can then
    # set a trim at dt_full[2], which can sit AT OR PAST the cut and so still keep a sub-floor
    # sample; the "auto-trimmed" message would then name a point where pressure never actually
    # fell below the floor. idx >= len(dt_full) - 1 happens when the cut sits past the last kept
    # point (crash beyond where resampling reached) -- nothing in the record to remove. Both
    # halves are covered without a pick here by the separate low-surface-pressure warning, which
    # scans raw (not kept) samples, so bailing never goes silent.
    if idx < 2 or idx >= len(dt_full) - 1:
        return
    commit_tail_trim(state, float(dt_full[idx]))
    state.tail_trim_reason = "low_pressure"


def resync_auto_tail_trim(state: PickState, td: TestData, res: DerivedResults) -> None:
    """Re-derive an auto-applied ("low_pressure") tail trim against a new shut-in pick.

    ``tail_trim_dt`` is shut-in-relative, but the shut-in pick itself can move after the trim
    was seeded -- the Injection step exists specifically to let the analyst drag it. Dragging
    shut-in later by delta moves the auto trim's cut delta later in absolute time too (it never
    re-ran), quietly re-admitting whatever crashed tail it was supposed to exclude while the
    "Tail auto-trimmed" warning keeps claiming the crash is handled. A manual trim (or no trim
    at all) is never touched here -- ``tail_trim_reason == ""`` covers both, and only the
    analyst's own drag/clear may set or clear those.

    Clears the existing pick first so ``seed_tail_trim``'s non-destructive guard (it returns
    immediately when ``tail_trim_dt is not None``) doesn't block the re-derive, then reseeds
    from scratch against the new window. If the new window has no crash (and no guard fire),
    the correct outcome is no trim at all, which falling through to seed_tail_trim gives for
    free.

    ``res`` is expected to have been computed with the STALE trim still in state (the caller
    calls this before clearing anything) -- that's fine, because seed_tail_trim only reads
    ``res.resampled_full``/``res.t_shutin_s``/``res.resampled_full.guard_dt``, none of which are
    masked by ``tail_trim_dt`` (only ``res.resampled``/``res.diagnostics`` are; see the
    resample block in ``model.compute_all``)."""
    if state.tail_trim_reason != "low_pressure":
        return
    state.tail_trim_dt = None
    state.tail_trim_reason = ""
    seed_tail_trim(state, td, res)


def seed_injection(state: PickState, td: TestData) -> None:
    """Injection window (start/shut-in indices) from the rate (+ optional volume) curve,
    falling back to the pressure shape when no usable rate channel exists -- the vlines must
    always exist for the analyst to drag, rate or not."""
    if state.start_idx is not None or state.shutin_idx is not None:
        return
    if state.rate_col:
        rate = td.column(state.rate_col)
        vol = td.column(state.volume_col) if state.volume_col else None
        try:
            state.start_idx, state.shutin_idx = interpret.suggest_injection_window(rate, vol)
            return
        except ValueError:
            pass  # e.g. an all-zero rate channel -- fall through to the pressure fallback
    if not state.pressure_col:
        return
    try:
        state.start_idx, state.shutin_idx = interpret.suggest_injection_window_pressure(
            td.column(state.pressure_col))
    except ValueError:
        pass


def seed_isip(state: PickState, td: TestData, res: DerivedResults) -> None:
    """Apparent-ISIP tangent: anchor ~1 min after shut-in, slope from a local fit of the early
    decline."""
    if state.isip_tangent is not None:
        return
    if res.t_shutin_s is None or res.bhp_all is None:
        return
    idx = _nearest(td.t_s, res.t_shutin_s + 60.0)
    anchor_x, anchor_y, slope = interpret.tangent_from_index(
        td.t_s, res.bhp_all, idx, half=interpret.ISIP_ANCHOR_HALF)
    state.isip_tangent = TangentPick(anchor_x=anchor_x, anchor_y=anchor_y, slope=slope)


def seed_gfunction(state: PickState, res: DerivedResults) -> None:
    """The min-dP/dG point (a diagnostic pick), plus the compliance contact pick at the dP/dG
    hump -- the contact pick feeds the derived effective-ISIP tangent, see
    model.compute_all/DerivedResults.eff_isip_line_compliance."""
    if state.min_dpdg_G is not None and state.contact_G is not None:
        return
    dg = res.diagnostics
    if dg is None or res.resampled is None or len(dg.G) <= 5:
        return
    if state.min_dpdg_G is None:
        idx = interpret.suggest_min_dpdg_index(dg.G, dg.dPdG)
        state.min_dpdg_G = float(dg.G[idx])
    if state.contact_G is None:
        hump = interpret.suggest_hump_index(dg.G, dg.dPdG)
        state.contact_G = float(dg.G[hump]) if hump is not None else float(dg.G[-1])


def seed_tangent(state: PickState, res: DerivedResults) -> None:
    """Tangent-method closure (slope + departure pick) from the through-origin departure."""
    if state.closure_slope is not None and state.closure_G is not None:
        return
    dg = res.diagnostics
    if dg is None or res.resampled is None or len(dg.G) <= 5:
        return
    cslope, idep = interpret.suggest_closure_tangent(dg.G, dg.GdPdG)
    if state.closure_slope is None:
        state.closure_slope = cslope
    if state.closure_G is None:
        state.closure_G = float(dg.G[idep])


def seed_loglog(state: PickState, res: DerivedResults) -> None:
    """Late-time window for the log-log diagnostic plot."""
    if state.loglog_window is not None:
        return
    dg = res.diagnostics
    if dg is None or len(dg.t) <= 6:
        return
    state.loglog_window = (float(dg.t[int(len(dg.t) * 0.6)]), float(dg.t[-1]))


def seed_pp(state: PickState, res: DerivedResults) -> None:
    """Late-time window for the pore-pressure diagnostic plot."""
    if state.pp_window is not None:
        return
    dg = res.diagnostics
    if dg is None or len(dg.t) <= 6:
        return
    if state.loglog_window is not None:
        state.pp_window = state.loglog_window
    else:
        # The last decade alone can hold <2 samples on sparsely-resampled falloffs; widen lo to
        # at least the second-to-last sample so the seeded window always has >=2 points to fit.
        lo = min(float(dg.t[-1]) / 10.0, float(dg.t[-2]))
        state.pp_window = (lo, float(dg.t[-1]))


def seed_stiffness(state: PickState, res: DerivedResults) -> None:
    """Relative-stiffness upturn pick, from interpret.suggest_stiffness_upturn_index -- same
    park-and-refine pattern as every other step. No-op when the stiffness arrays aren't
    computed (res.stiffness_S is None: no min-dP/dG pick yet, no pore-pressure estimate, PC-F,
    or too few resampled points -- see model.compute_all's stiffness block), or when the
    analyst has already recorded "no slope change apparent" -- a reloaded save with that flag
    must not get a line parked on first visit."""
    if state.stiffness_pick_P is not None:
        return
    if state.stiffness_no_upturn:
        return
    if res.stiffness_S is None:
        return
    # stiffness_S[i] pairs with grid sample i+1, so the G slice aligned with it is G[1:] --
    # stiffness_G, not diagnostics.G: the two only differ when STIFFNESS_MAX_POINTS decimated
    # the stiffness arrays to a subset of the full resampled grid (model.compute_all).
    idx = interpret.suggest_stiffness_upturn_index(res.stiffness_S, res.stiffness_G[1:])
    if idx is None:
        return
    state.stiffness_pick_P = float(res.stiffness_p_eff[1:][idx])


# seed_tail_trim is deliberately absent here -- see its docstring for why ui._seed_step calls
# it directly instead.
SEEDERS = {
    "overview": seed_overview,
    "injection": seed_injection,
    "isip": seed_isip,
    "gfunction": seed_gfunction,
    "tangent": seed_tangent,
    "loglog": seed_loglog,
    "porepressure": seed_pp,
    "stiffness": seed_stiffness,
}
