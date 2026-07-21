"""Pan-capable RangeSlider and log-space helpers. Matplotlib-only -- no Tkinter -- so it is
headless-testable (Agg + synthetic MouseEvents), same as picks.py.

Stock ``matplotlib.widgets.RangeSlider`` treats every press on its track as "jump the nearest
thumb", so there is no way to drag the whole selected window at once -- exactly what the
per-axis zoom sliders in ui.py need. ``PanRangeSlider`` adds that: a press landing strictly
between the two thumbs (beyond a pixel tolerance from each) grabs the bar and pans both thumbs
together on drag, preserving the window width and clamping to [valmin, valmax]. A press at/near
either thumb is left entirely to stock ``RangeSlider`` behavior, so thumb-drag and click-to-jump
are unchanged.
"""

from __future__ import annotations

import math

from matplotlib.widgets import RangeSlider, _call_with_reparented_event

_LOG_FLOOR = 1e-12


def to_log_bounds(lo: float, hi: float, floor: float = _LOG_FLOOR) -> tuple[float, float]:
    """Clamp both bounds to ``floor`` (log10 is undefined at/below 0) and convert to log10 space,
    for feeding a log-scaled axis's data extent into a linear-valued RangeSlider."""
    return math.log10(max(lo, floor)), math.log10(max(hi, floor))


def from_log_bounds(log_lo: float, log_hi: float) -> tuple[float, float]:
    """Inverse of ``to_log_bounds``: log10-space slider values back to linear axis limits."""
    return 10.0 ** log_lo, 10.0 ** log_hi


class PanRangeSlider(RangeSlider):
    """A ``RangeSlider`` whose track also supports bar-drag panning (see module docstring)."""

    #: presses within this many pixels of a thumb defer entirely to stock thumb-drag/jump.
    THUMB_TOL_PX = 8.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pan_active = False
        self._pan_ref_val = None
        self._pan_ref_pos = None

    # -- geometry -------------------------------------------------------------------------------
    def _press_pos(self, event):
        """The event's data coordinate along this slider's active axis."""
        return event.xdata if self.orientation == "horizontal" else event.ydata

    def _press_pixel(self, event):
        """The event's pixel coordinate along this slider's active axis."""
        return event.x if self.orientation == "horizontal" else event.y

    def _thumb_pixels(self):
        """Pixel positions of the two thumbs, ascending. Uses transData on this slider's own
        axis only -- transData is separable in x/y, so the unused coordinate is irrelevant."""
        lo, hi = self.val
        if self.orientation == "horizontal":
            lo_px = self.ax.transData.transform((lo, 0.0))[0]
            hi_px = self.ax.transData.transform((hi, 0.0))[0]
        else:
            lo_px = self.ax.transData.transform((0.0, lo))[1]
            hi_px = self.ax.transData.transform((0.0, hi))[1]
        return (lo_px, hi_px) if lo_px <= hi_px else (hi_px, lo_px)

    def _in_pan_zone(self, event) -> bool:
        """True if the press falls strictly between the thumbs, beyond ``THUMB_TOL_PX`` of each.
        Also False when the thumbs are within ``2 * THUMB_TOL_PX`` of each other (degenerate/
        collapsed window) -- there is no bar to grab, so stock nearest-thumb behavior applies."""
        p = self._press_pixel(event)
        if p is None:
            return False
        lo_px, hi_px = self._thumb_pixels()
        return (lo_px + self.THUMB_TOL_PX) < p < (hi_px - self.THUMB_TOL_PX)

    # -- event handling --------------------------------------------------------------------------
    @_call_with_reparented_event
    def _update(self, event):
        """Single entry point for press/motion/release, mirroring stock RangeSlider._update
        (SliderBase connects all three event names to this one method)."""
        if self.ignore(event) or event.button != 1:
            return

        if (event.name == "button_press_event" and not self._pan_active
                and self.ax.contains(event)[0] and self._in_pan_zone(event)):
            self._pan_active = True
            self._pan_ref_val = self.val
            self._pan_ref_pos = self._press_pos(event)
            event.canvas.grab_mouse(self.ax)
            return

        if not self._pan_active:
            super()._update(event)
            return

        if event.name == "motion_notify_event":
            pos = self._press_pos(event)
            if pos is None or self._pan_ref_pos is None:
                return
            ref_lo, ref_hi = self._pan_ref_val
            d = pos - self._pan_ref_pos
            d = max(self.valmin - ref_lo, min(d, self.valmax - ref_hi))
            self.set_val((ref_lo + d, ref_hi + d))
            return

        if (event.name == "button_release_event"
                or (event.name == "button_press_event" and not self.ax.contains(event)[0])):
            self._pan_active = False
            event.canvas.release_mouse(self.ax)
            return
        # Any other event while panning (e.g. a stray second press inside the axes) is ignored.
