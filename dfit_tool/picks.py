"""Interactive picking on the matplotlib canvas.

Two generic controllers (click and span-select) capture mouse events and forward data coordinates
to a callback. The step-specific ``handle_*`` functions translate a coordinate into a PickState
change. Everything here depends only on matplotlib (never Tkinter), so the picking layer is
reusable if the shell is ported.

Interaction model (first build): click-to-place for verticals/points, drag-select for windows.
Numeric fine-tuning of a placed pick is done in the side panel, not by dragging handles.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from matplotlib.widgets import SpanSelector

from . import interpret
from .model import DerivedResults, PickState, TangentPick
from .io_load import TestData


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

    Works off the event pixel rather than ``event.inaxes``/``event.xdata``: an overlaid twin axes
    (e.g. the overview's rate ``twinx``) owns ``inaxes`` over the shared region, so identity checks
    against ``self.ax`` would never match. Twinned axes share the x-scale, so converting the event
    pixel through ``self.ax.transData`` yields the correct data-x regardless.
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

    def _in_axes(self, event) -> bool:
        """True if the event pixel falls within self.ax, regardless of which (possibly twinned)
        axes matplotlib assigned to event.inaxes."""
        return (event.x is not None and event.y is not None
                and self.ax.bbox.contains(event.x, event.y))

    def _x_from_pixel(self, event) -> float:
        """Data-x on self.ax for the event pixel (robust to a twin axes owning inaxes)."""
        return float(self.ax.transData.inverted().transform((event.x, event.y))[0])

    def _on_press(self, event):
        if event.button != 1 or not self._in_axes(event):
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
        x = self._x_from_pixel(event)
        self._active.set_xdata([x, x])
        self.canvas.draw_idle()

    def _on_release(self, event):
        if self._active is None:
            return
        line = self._active
        self._active = None
        if event.x is not None and event.y is not None:
            x = self._x_from_pixel(event)
        else:
            x = float(line.get_xdata()[0])
        handler = self.handlers.get(line.get_gid())
        if handler is not None:
            handler(float(x))

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


# --------------------------------------------------------------------------------------------------
# step pick handlers  (mutate PickState in place)
# --------------------------------------------------------------------------------------------------
def handle_isip_click(state: PickState, td: TestData, res: DerivedResults,
                      x_min: float, button: int) -> None:
    """Set the literal-ISIP tangent: anchor at the nearest BHP sample, slope from a local fit."""
    if res.t_shutin_s is None or res.bhp_all is None:
        return
    t_target = res.t_shutin_s + x_min * 60.0
    idx = _nearest(td.t_s, t_target)
    slope = _local_slope(td.t_s, res.bhp_all, idx, half=30)  # psi/s over ~1 min of 1 Hz data
    state.isip_tangent = TangentPick(anchor_x=float(td.t_s[idx]),
                                     anchor_y=float(res.bhp_all[idx]), slope=slope)


def handle_gfunction_click(state: PickState, res: DerivedResults,
                           x_G: float, button: int) -> None:
    """Left-click sets the effective-ISIP line (anchor + local P-vs-G slope); right-click contact."""
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
    """Set the tangent-method closure (departure) point; refresh the through-origin slope."""
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
