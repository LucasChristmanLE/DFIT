"""Pan-capable RangeSlider and log-space helpers. Matplotlib-only -- no Tkinter -- so it is
headless-testable (Agg + synthetic MouseEvents), same as picks.py.

Stock ``matplotlib.widgets.RangeSlider`` treats every press on its track as "jump the nearest
thumb", so there is no way to drag the whole selected window at once -- exactly what the
per-axis zoom sliders in ui.py need. ``PanRangeSlider`` adds that: a press landing strictly
between the two thumbs (beyond a pixel tolerance from each) grabs the bar and pans both thumbs
together on drag, preserving the window width and clamping to [valmin, valmax]. A press at/near
either thumb is left entirely to stock ``RangeSlider`` behavior, so thumb-drag and click-to-jump
are unchanged.

``PanRangeSlider`` also replaces stock ``RangeSlider``'s single "(lo, hi)" value text with two
split texts, one per bound: for a vertical slider, hi above the track and lo below (both
centered) -- the stock single, centered string is wider than the fixed-pixel column ``ui.py``
gives each vertical slider and collides with its neighbor when two sit side by side (see
``right_margin_layout`` below). For a horizontal slider, lo at the track's left end and hi at
its right end (both below the track, each anchored so its text extends inward, never past the
track's own footprint) -- the stock single string sits to the right of the WHOLE track, which
can run past the figure's edge once the track is widened to match the plot (``ui.py``'s x
slider). Both texts sit a fixed point offset (not an Axes-fraction one) off the track, so the
gap from the track -- and from a thumb handle sitting right at that edge -- doesn't shrink as
the track's own pixel size changes.

``right_margin_layout``/``bottom_margin_layout`` are the pure pixel-math half of the plot/slider
layout: given the figure size and how far a twin axis's tick labels (right) or the x-axis's own
tick labels/xlabel (bottom) overhang the main Axes' edge (measured by ``ui.DfitApp._layout_sliders``
via ``get_tightbbox``), they return the main Axes' right/bottom-edge fraction and each slider's
track position, so the tracks always clear those labels -- and the x slider's own split text
always clears the figure's bottom edge -- regardless of window size or DPI. Both take already
dpi-scaled pixel inputs (``col_px``/``pad_px``/``track_px``/etc.) rather than doing any dpi
scaling themselves, since a raw pixel count doesn't track a points-based quantity like rendered
text as dpi rises (e.g. Windows 150%/200% display scaling under TkAgg) -- ``ui.py`` is what
knows the current ``fig.dpi`` to scale by.
"""

from __future__ import annotations

import math

from matplotlib.transforms import offset_copy
from matplotlib.widgets import RangeSlider, _call_with_reparented_event

_LOG_FLOOR = 1e-12

#: fixed on-screen width (px) of a vertical slider's track, centered within its layout column.
TRACK_PX = 14.0

#: fixed point offset of a split value text off its track edge -- points (not an Axes fraction)
#: so the gap stays constant regardless of the track's own pixel height/width, and comfortably
#: clears a thumb handle (~5pt marker radius, the ``handle_style`` default ``size=10``) sitting
#: right at that edge.
TEXT_OFFSET_PT = 9.0

#: extra breathing room (nominal 100dpi px) added on each side of a vertical slider's column
#: when it's sized from its own text's real rendered width (``ui.DfitApp._layout_sliders``)
#: rather than the flat ``col_px`` default, so the text doesn't sit flush against the column
#: edge (or its neighbor's) once that width is used instead of a guess.
COLUMN_TEXT_PAD_PX = 6.0

#: generous single-line height estimate (points) for the split text's own ~8pt font, used to
#: reserve room for it below the horizontal (x) slider's track. Both ``_build_sliders`` and
#: ``_layout_sliders`` use this estimate; the text height is never measured -- see
#: ``bottom_margin_layout``.
TEXT_HEIGHT_PT = 12.0


def to_log_bounds(lo: float, hi: float, floor: float = _LOG_FLOOR) -> tuple[float, float]:
    """Clamp both bounds to ``floor`` (log10 is undefined at/below 0) and convert to log10 space,
    for feeding a log-scaled axis's data extent into a linear-valued RangeSlider."""
    return math.log10(max(lo, floor)), math.log10(max(hi, floor))


def from_log_bounds(log_lo: float, log_hi: float) -> tuple[float, float]:
    """Inverse of ``to_log_bounds``: log10-space slider values back to linear axis limits."""
    return 10.0 ** log_lo, 10.0 ** log_hi


def right_margin_layout(fig_w_px: float, n_vertical: int, axis_overhang_px: float,
                        col_px: float = 60.0, pad_px: float = 12.0, track_px: float = TRACK_PX
                        ) -> tuple[float, list[float]]:
    """Pure pixel layout for the plot's right margin and its vertical zoom sliders.

    Reserves, right to left from the figure's right edge: nothing (if ``n_vertical`` is 0);
    else one fixed ``col_px``-wide column per vertical slider (in the order the caller wants
    them, left to right), each holding a ``track_px``-wide track centered in its column; then a
    ``pad_px`` gap; then ``axis_overhang_px`` -- the measured overhang of a twin/third axis's
    tick labels past the main Axes' own right edge, so reserving it is what keeps a slider
    track off of those labels. Whatever's left, left of all that, is the main Axes' width.

    ``col_px``/``pad_px``/``track_px`` all default to their nominal (100dpi) values, but the
    caller is expected to pass dpi-scaled ones at another dpi (``ui.DfitApp._build_sliders``/
    ``_layout_sliders``) -- this function stays a plain function of whatever pixel values it's
    given, with no dpi/Figure awareness of its own, so it can't silently use the wrong one of
    the two internally the way an earlier version did (hardcoding the module-level ``TRACK_PX``
    regardless of what ``col_px`` it was actually asked to center within).

    Returns ``(axes_right_frac, [track_x0_frac, ...])``, both as fractions of ``fig_w_px``: the
    main Axes' right edge (for ``Figure.subplots_adjust(right=...)``) and each vertical slider's
    track left-x (for ``Axes.set_position``), in the same left-to-right column order as given.
    A plain function of pixel measurements -- no Figure/Axes/renderer -- so it is unit-testable
    without building any matplotlib state, and reusable from both the initial build (no
    renderer yet -- see ``ui.DfitApp._build_sliders``) and the post-draw refinement / resize
    hook (measured overhang -- see ``ui.DfitApp._layout_sliders``).
    """
    n_vertical = max(int(n_vertical), 0)
    margin_start_px = fig_w_px - n_vertical * col_px
    axes_right_px = margin_start_px - pad_px - max(axis_overhang_px, 0.0)
    x_fracs = [
        (margin_start_px + i * col_px + (col_px - track_px) / 2.0) / fig_w_px
        for i in range(n_vertical)
    ]
    return axes_right_px / fig_w_px, x_fracs


def bottom_margin_layout(fig_h_px: float, default_track_y0_px: float, track_h_px: float,
                         text_below_px: float, bottom_overhang_px: float,
                         label_gap_px: float = 6.0, bottom_floor_frac: float = 0.16
                         ) -> tuple[float, float]:
    """Pure pixel layout for the plot's bottom margin and the horizontal (x) slider's track.

    The track's bottom edge sits at whichever is larger: ``default_track_y0_px`` (a look-and-
    feel baseline the caller sizes as a fraction of the figure height, e.g. 4%) or
    ``text_below_px`` (the vertical clearance the split lo/hi text needs below the track --
    offset + text height + a small pad, all already in device pixels -- see
    ``ui.DfitApp._layout_sliders``). On a tall-enough figure the baseline already clears the
    text and wins; on a short one, ``text_below_px`` wins instead and the track moves up rather
    than letting the text clip off the figure's bottom edge.

    The plot's own bottom margin then clears the track's top edge, plus ``bottom_overhang_px``
    (how far the x-axis's own tick labels + xlabel extend below the primary Axes' bottom edge --
    measured, not guessed, by the caller, mirroring ``right_margin_layout``'s twin/d2 overhang)
    and a small ``label_gap_px``, floored at ``bottom_floor_frac`` so a step with very little
    bottom content still gets its usual margin.

    Returns ``(bottom_frac, track_y0_frac)``, both fractions of ``fig_h_px``. A plain function
    of pixel measurements -- no Figure/Axes/renderer -- mirroring ``right_margin_layout``.
    """
    track_y0_px = max(default_track_y0_px, text_below_px)
    track_top_px = track_y0_px + track_h_px
    bottom_px = track_top_px + max(bottom_overhang_px, 0.0) + label_gap_px
    bottom_frac = max(bottom_floor_frac, bottom_px / fig_h_px)
    return bottom_frac, track_y0_px / fig_h_px


class PanRangeSlider(RangeSlider):
    """A ``RangeSlider`` whose track also supports bar-drag panning (see module docstring)."""

    #: presses within this many pixels of a thumb defer entirely to stock thumb-drag/jump.
    THUMB_TOL_PX = 8.0

    #: a press outside the track's Axes within this many nominal (100 dpi) pixels of a thumb
    #: grabs it (see ``_outside_thumb_hit``).
    HANDLE_GRAB_PX = 10.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._pan_active = False
        self._pan_ref_val = None
        self._pan_ref_pos = None
        # Split hi/lo value text -- see module docstring. ``set_val`` (which stock
        # ``RangeSlider.__init__`` already called once, above, before either attribute existed
        # at all -- see ``_sync_split_text``'s ``getattr`` guard) has nothing to land on until
        # both are assigned below.
        self.valtext.set_visible(False)
        fig = self.ax.figure
        below = offset_copy(self.ax.transAxes, fig=fig, y=-TEXT_OFFSET_PT, units="points")
        if self.orientation == "vertical":
            above = offset_copy(self.ax.transAxes, fig=fig, y=TEXT_OFFSET_PT, units="points")
            self._hi_text = self.ax.text(0.5, 1.0, "", transform=above,
                                         va="bottom", ha="center", fontsize=8)
            self._lo_text = self.ax.text(0.5, 0.0, "", transform=below,
                                         va="top", ha="center", fontsize=8)
        else:
            # lo at the left end, hi at the right end, both below the track and each anchored so
            # its own text extends INWARD (never past the track's own left/right edge, and so
            # never past the figure edge just because the track itself was widened to match the
            # plot -- see module docstring).
            self._lo_text = self.ax.text(0.0, 0.0, "", transform=below,
                                         va="top", ha="left", fontsize=8)
            self._hi_text = self.ax.text(1.0, 0.0, "", transform=below,
                                         va="top", ha="right", fontsize=8)
        self._sync_split_text()

    def _format_single(self, v: float) -> str:
        """Pretty-print one bound -- same convention as stock ``RangeSlider._format``, but for
        a single value rather than the "(lo, hi)" pair, since each split text shows only one
        bound. ``valfmt`` (e.g. the log-axis ``10**v`` display) takes precedence when given."""
        if self.valfmt is not None:
            return self.valfmt(v) if callable(self.valfmt) else (self.valfmt % v)
        return f"{v:.0f}" if abs(v) >= 1000 else f"{v:.3g}"

    def _sync_split_text(self):
        """Update the two split texts from ``self.val``. A no-op, via the ``getattr`` guard, for
        the one ``set_val`` stock ``RangeSlider.__init__`` makes before either text exists yet."""
        hi_text = getattr(self, "_hi_text", None)
        if hi_text is None:
            return
        lo, hi = self.val
        hi_text.set_text(self._format_single(hi))
        self._lo_text.set_text(self._format_single(lo))

    def set_val(self, val):
        super().set_val(val)
        self._sync_split_text()

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
        collapsed window) -- there is no bar to grab, so stock nearest-thumb behavior applies.

        Only called after ``self.ax.contains(event)[0]`` is already True (see ``_update``), which
        itself requires real ``event.x``/``.y`` pixel coordinates -- so ``_press_pixel`` can't
        return None here."""
        p = self._press_pixel(event)
        lo_px, hi_px = self._thumb_pixels()
        return (lo_px + self.THUMB_TOL_PX) < p < (hi_px - self.THUMB_TOL_PX)

    def _outside_thumb_hit(self, event):
        """The thumb handle a press OUTSIDE this slider's Axes grabs, or None.

        Stock ``RangeSlider`` only reacts to presses inside its Axes, but a thumb at either end
        of the track is drawn half past it, and the tracks are only ~14-18 px across, so much of
        a thumb's visible marker is dead. A press counts if it is within ``HANDLE_GRAB_PX``
        (dpi-scaled) of a thumb along the track and of the track's own extent across it."""
        if event.x is None or event.y is None:
            return None
        tol = self.HANDLE_GRAB_PX * self.ax.figure.dpi / 100.0
        bb = self.ax.bbox
        if self.orientation == "horizontal":
            along, across, lo_edge, hi_edge = event.x, event.y, bb.y0, bb.y1
        else:
            along, across, lo_edge, hi_edge = event.y, event.x, bb.x0, bb.x1
        if not (lo_edge - tol <= across <= hi_edge + tol):
            return None
        lo, hi = self.val
        axis = 0 if self.orientation == "horizontal" else 1
        best, best_d = None, tol
        for handle, v in zip(self._handles, (lo, hi)):
            pt = (v, 0.0) if axis == 0 else (0.0, v)
            d = abs(self.ax.transData.transform(pt)[axis] - along)
            if d <= best_d:
                best, best_d = handle, d
        return best

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

        if (event.name == "button_press_event" and not self._pan_active
                and not self.drag_active and not self.ax.contains(event)[0]):
            handle = self._outside_thumb_hit(event)
            if handle is not None:
                # Start a stock thumb drag without moving the value on the press itself; the
                # following motion/release events go through stock ``_update`` as usual.
                self.drag_active = True
                self._active_handle = handle
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
