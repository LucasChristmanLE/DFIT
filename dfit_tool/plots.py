"""Static rendering of each workflow step onto a matplotlib Axes/Figure.

Each ``render_*`` takes an Axes, the loaded ``TestData``, the ``PickState``, and the
``DerivedResults`` and draws the plot plus whatever picks currently exist. The interactive layer
(picks.py) updates the PickState and calls the matching render to refresh.

Renderers no longer set view limits (``set_xlim``/``set_ylim``) on the Axes -- they leave the
Axes autoscaled to the full data extent and instead return a ``ViewDefaults`` describing the
view the caller should apply on first visit to a step. ``apply_step_view`` is the one place that
resolves the autoscaled extent against ``ViewDefaults`` (or a stored ``ViewState``) and applies
it; ui.refresh and the headless PNG export both call it. ui.py owns the per-step stored views.

Matplotlib only -- no Tkinter -- so figures can be produced headlessly (Agg) for verification.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

from . import colors as C
from . import interpret, summary
from .model import (STEP_KEYS, DerivedResults, PickState, closure_uninterpretable,
                    loglog_window_suppressed, pp_from_peak, skipped_steps)
from .io_load import TestData

_MAX_POINTS = 6000  # display decimation cap for the raw (dense) traces

# Solid-segment half-width (minutes) for the apparent-ISIP tangent construction; must keep both
# endpoints inside render_isip's default (-1, 3)-min view for the seeded anchor (~+1 min) so the
# AnchorLineController rotate handles stay reachable.
_ISIP_TANGENT_HALF_MIN = 0.75


D2_AXIS_GID = "d2pdg2_axis"  # gid on the gfunction step's optional third (d2P/dG2) twin axes

RATE_VIEW_FACTOR = 3.0  # default rate-axis ceiling = this x the max rate plotted on that step, so
                        # the rate trace rides in the bottom third, clear of the pressure trace
DPDG_VIEW_MAX = 2000.0  # hard ceiling on the gfunction dP/dG axis: default view AND slider range
Y2_SCALE_G_MIN = 1.0    # G below this is the water-hammer spike -- excluded from the autoscale


@dataclass
class ViewDefaults:
    """The view a renderer suggests for first-visit display; ``None`` means autoscaled full
    extent. Callers apply these to the Axes -- renderers never set limits themselves.

    ``y_color``/``y2_color`` name the color of the primary/twin trace this step's y/y2 zoom
    slider should be painted to match (``ui.py``'s ``_build_sliders``/``_make_range_slider``);
    ``None`` means no single color dominates that axis (e.g. log-log's two same-axis series),
    and the slider falls back to a neutral gray. Not a view limit, so setting them does not
    violate "renderers never set view limits" -- an early-return ``ViewDefaults()`` (no data
    to color a slider for) stays colorless."""
    xlim: Optional[tuple[float, float]] = None
    ylim: Optional[tuple[float, float]] = None
    y2lim: Optional[tuple[float, float]] = None
    y3lim: Optional[tuple[float, float]] = None
    y_color: Optional[str] = None
    y2_color: Optional[str] = None


_NICE_LOCATOR = MaxNLocator(nbins=10,steps=[1, 2, 2.5, 5, 10])


def nice_limits(lo: float, hi: float) -> tuple[float, float]:
    """Round a linear default range outward to tick-aligned values (1/2/2.5/5 x 10^k steps).
    Non-finite or empty ranges pass through unchanged."""
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return (lo, hi)
    ticks = _NICE_LOCATOR.tick_values(lo, hi)  # first tick <= lo, last >= hi
    tol = 1e-9 * (hi - lo)
    below, above = ticks[ticks <= lo + tol], ticks[ticks >= hi - tol]
    if not (below.size and above.size):
        return (lo, hi)
    return (float(below.max()), float(above.min()))


def nice_log_limits(lo: float, hi: float) -> tuple[float, float]:
    """Round a positive log-axis default range outward to whole decades."""
    if not (np.isfinite(lo) and np.isfinite(hi)) or lo <= 0 or hi <= lo:
        return (lo, hi)
    return (10.0 ** np.floor(np.log10(lo) + 1e-9), 10.0 ** np.ceil(np.log10(hi) - 1e-9))


_ISIP_DEFAULT_XLIM = (-1.0, 3.0)  # ISIP first-visit view, minutes from shut-in


def _pressure_ylim(p_plotted) -> Optional[tuple[float, float]]:
    """Default pressure-axis range over the pressure actually plotted on a step: 5% pad, then
    rounded outward to ticks. None when nothing finite is plotted."""
    p = np.asarray(p_plotted, dtype=float)
    p = p[np.isfinite(p)]
    if p.size == 0:
        return None
    p_lo, p_hi = float(p.min()), float(p.max())
    pad = 0.05 * max(p_hi - p_lo, 1.0)
    return nice_limits(p_lo - pad, p_hi + pad)


def _rate_y2lim(rate_plotted) -> Optional[tuple[float, float]]:
    """Default rate-axis range (0, RATE_VIEW_FACTOR x max) over the rate actually plotted on a
    step, rounded up to a tick, or None when there is no finite positive rate. Samples above
    ``interpret.INJECTION_MAX_PLAUSIBLE_BPM`` (a fill read at a non-physical rate) are left out
    unless no sample at or below it is positive; the slider's outer range still reaches them."""
    if rate_plotted is None:
        return None
    r = np.asarray(rate_plotted, dtype=float)
    r = r[np.isfinite(r)]
    plausible = r[r <= interpret.INJECTION_MAX_PLAUSIBLE_BPM]
    if plausible.size and plausible.max() > 0:
        r = plausible
    if r.size == 0 or r.max() <= 0:
        return None
    return nice_limits(0.0, RATE_VIEW_FACTOR * float(r.max()))


def _decimate(x: np.ndarray, *ys: np.ndarray):
    """Uniformly thin long arrays for display without distorting shape."""
    n = len(x)
    if n <= _MAX_POINTS:
        return (x, *ys)
    step = int(np.ceil(n / _MAX_POINTS))
    return (x[::step], *(y[::step] for y in ys))


def _hours(t_s: np.ndarray, t0: float = 0.0) -> np.ndarray:
    return (np.asarray(t_s, dtype=float) - t0) / 3600.0


def _split_dropouts(p: np.ndarray, mask: Optional[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Split a pressure trace into ``(p_clean, p_masked)``: each holds NaN wherever the other one
    has real data. Plotting ``p_clean`` breaks the main trace's line across a masked dropout
    instead of drawing a spike through it; ``p_masked`` is what a caller scatters as markers
    (see render_overview/render_isip). ``mask`` may be ``None`` (no dropout detection ran, e.g. a
    hand-built ``DerivedResults`` in a test) -- both counterparts then degrade to ``(p, all-NaN)``.
    """
    p = np.asarray(p, dtype=float)
    if mask is None or not np.any(mask):
        return p, np.full_like(p, np.nan)
    p_clean = p.copy()
    p_clean[mask] = np.nan
    p_masked = np.full_like(p, np.nan)
    p_masked[mask] = p[mask]
    return p_clean, p_masked


_MAX_DROPOUT_MARKERS = 20000


def _plot_manual_spans(ax, state: PickState) -> None:
    """Shade the analyst's manual mask/keep intervals (absolute seconds) as hour-axis bands.
    dataLim is snapshotted and restored so the bands never move autoscale or ViewDefaults."""
    saved_points = ax.dataLim.get_points().copy()
    for intervals, gid, color in ((state.mask_intervals, "manual_mask", C.MANUAL_MASK),
                                  (state.keep_intervals, "manual_keep", C.MANUAL_KEEP)):
        for lo, hi in intervals:
            span = ax.axvspan(lo / 3600.0, hi / 3600.0, color=color, alpha=C.MANUAL_SPAN_ALPHA,
                              lw=0, gid=gid)
            span.set_label("_nolegend_")
    ax.dataLim.set_points(saved_points)


def _plot_dropout_markers(ax, x: np.ndarray, y: np.ndarray) -> None:
    """Scatter masked-dropout samples (``x``/``y`` already filtered to the finite/masked subset --
    see render_overview/render_isip) as their own small markers, without letting their real (and
    possibly extreme, e.g. a -9999 psi sentinel) value drag the Axes' autoscale down to them --
    the warning already reports the event, so the marker is allowed to sit off-screen below the
    view. ``scalex=False, scaley=False`` stops this ``plot`` call from requesting a fresh
    autoscale of the VIEW, but ``Axes.add_line`` still unconditionally folds the marker into
    ``ax.dataLim`` (the bounding box other code reads back, e.g. the ISIP step's default view or
    the Overview y-slider's full range) -- so ``dataLim`` is snapshotted first and restored after,
    leaving the marker plotted at its true position but invisible to both the view and dataLim.
    A no-op when there's nothing to plot.
    """
    if not len(x):
        return
    if len(x) > _MAX_DROPOUT_MARKERS:
        # A rise excursion can mask tens of thousands of samples; thin evenly.
        stride = int(np.ceil(len(x) / _MAX_DROPOUT_MARKERS))
        x, y = x[::stride], y[::stride]
    saved_points = ax.dataLim.get_points().copy()
    ax.plot(x, y, color=C.DROPOUT, marker="o", ms=3, ls="none", label="masked dropout",
            gid="dropout_masked", scalex=False, scaley=False)
    ax.dataLim.set_points(saved_points)


def _draw_tangent_construction(ax, anchor_x: float, anchor_y: float, slope: float, *,
                               ref_x: float, half: float, color: str, gids: dict,
                               tick_half_y: float, label: Optional[str] = None,
                               lw: float = 1.6, draw_tick: bool = True) -> None:
    """Draw one gid-tagged tangent construction (see the workflow steps in ../CLAUDE.md): a finite ``segment`` through
    the anchor, a short vertical ``tick`` at the anchor, and a dashed ``extension`` running from
    the segment's near end back to the reference vertical ``ref_x`` (the shut-in line for the
    apparent-ISIP construction, G=0 for the effective-ISIP construction) -- the ISIP marker sits
    where the extension crosses ``ref_x``. ``gids`` maps "segment"/"tick"/"extension" to the exact
    gid string each piece is drawn with, matched by ``picks.AnchorLineController``. ``draw_tick``
    suppresses the vertical anchor tick where the construction is not user-draggable (the
    G-function effective-ISIP line, which follows the contact marker) and the tick would just be a
    fixed vertical mark.
    """
    x0, x1 = anchor_x - half, anchor_x + half
    xs = np.array([x0, x1])
    ys = anchor_y + slope * (xs - anchor_x)
    ax.plot(xs, ys, color=color, lw=lw, label=label, gid=gids["segment"])
    if draw_tick:
        ax.plot([anchor_x, anchor_x], [anchor_y - tick_half_y, anchor_y + tick_half_y],
                color=color, lw=lw, gid=gids["tick"])
    near_x = x0 if abs(x0 - ref_x) <= abs(x1 - ref_x) else x1
    ext_x = np.array([ref_x, near_x])
    ext_y = anchor_y + slope * (ext_x - anchor_x)
    ax.plot(ext_x, ext_y, color=color, lw=max(lw - 0.3, 1.0), ls="--", gid=gids["extension"])


# --------------------------------------------------------------------------------------------------
def render_overview(ax, td: TestData, state: PickState, res: DerivedResults,
                    interactive: bool = True) -> ViewDefaults:
    """Step 1: the entire dataset, unmasked -- BHP (or surface P) and rate vs time, for the whole
    record from file start to its last raw sample.

    ``render_injection`` (the next step) clamps its default view to the active-injection region so
    a multi-week falloff tail doesn't dwarf it; that's useful once the analyst is working the
    injection window, but it hides the tail's true length/shape up front. This step shows the full
    record with no clamp, so that context is visible before the tool zooms in. The start/shut-in
    picks are drawn here only as thin reference lines (gids "start_ref"/"shutin_ref", distinct from
    Injection's draggable "start"/"shutin") -- they are owned and dragged on the Injection step;
    this step draws no controllers of its own.

    The tail-trim line is always on: ``picks.seed_tail_trim`` parks it at the earliest of the
    rise-guard boundary or a sub-50-psi surface-pressure crash on the step's first visit (or at
    the end of the data when neither exists), and it stays draggable from there -- there is no
    more "Show trim tool" toggle. ``interactive`` (default True) means "this is the live canvas,
    not an export": it gates only the draggable vline itself (gid "tail_trim"), so
    ``render_step_figure`` can pass ``interactive=False`` and an exported PNG never carries a
    line the analyst can't actually drag.

    The effective display cut, ``cut_dt``, comes from ``interpret.resolve_tail_cut_dt(
    state.tail_trim_dt, res.resampled_full.guard_dt, state.tail_guard_override)`` -- the same
    helper ``model.compute_all`` uses to mask ``res.resampled``, so the line the analyst sees
    always matches what was actually diagnosed. No trim set -> the guard's boundary (or None).
    A trim at/before the guard (or no guard at all) -> the trim, unchanged. A trim past the
    guard is respected only when ``state.tail_guard_override`` is True -- set by
    ``picks.commit_tail_trim`` when a committed drag deliberately lands past ``guard_dt`` --
    otherwise it's presumed stale (e.g. left behind by a shut-in move before ``ui.py``'s
    ``resync_auto_tail_trim`` ran) and clamped back to ``guard_dt``. Both ``tail_trim_dt`` and
    ``guard_dt`` live in shut-in-relative dt, so dragging the Injection shut-in line later
    shrinks ``guard_dt`` while an existing ``tail_trim_dt`` is unchanged until resynced -- and
    because a guard-fired record parks its trim right at ``guard_dt``, a trim within a resample
    step of the guard is the ordinary case, not an edge case. Taking the plain max of the two
    (or just ``tail_trim_dt``) can leave a stale trim that sits PAST the guard, which would
    render the guard-excluded region as kept -- the opposite of this feature's purpose -- so the
    clamp is required whenever the override isn't set. The raw pressure trace is split and
    grayed out beyond ``cut_dt`` (gid
    "tail_excluded") whenever it's not None -- this is what makes a guard-excluded tail visible
    on Overview too, not just in the G-function plot's own ``guard_excluded`` preview. It is
    drawn as its own segment of the real raw trace (not an overlay of the coarser post-shut-in
    resample), so it is visible in front of, not under, the kept portion. The draggable vline
    itself sits at ``cut_dt`` when set, else the last raw sample time (nothing to cut yet).
    """
    ax.clear()
    p = res.bhp_all if res.bhp_all is not None else np.full(td.n, np.nan)
    t_h = _hours(td.t_s)
    press_color, press_ylabel, press_label = _pressure_style(res)

    has_trim_context = (res.resampled_full is not None and res.t_shutin_s is not None
                        and len(res.resampled_full.dt))
    cut_dt = None
    if has_trim_context:
        cut_dt = interpret.resolve_tail_cut_dt(state.tail_trim_dt, res.resampled_full.guard_dt,
                                                state.tail_guard_override)

    kept = np.ones_like(t_h, dtype=bool)
    if cut_dt is not None:
        t_trim_s = res.t_shutin_s + cut_dt
        kept = td.t_s <= t_trim_s

    p_clean, p_masked = _split_dropouts(p, res.dropout_mask)
    xt, xp = _decimate(t_h[kept], p_clean[kept])
    ax.plot(xt, xp, color=press_color, lw=0.8, label=press_label)
    excluded = ~kept
    if excluded.any():
        xte, xpe = _decimate(t_h[excluded], p_clean[excluded])
        ax.plot(xte, xpe, color=C.EXCLUDED, alpha=C.EXCLUDED_ALPHA, lw=0.8, gid="tail_excluded")
    # Converted-BHP record: overlay the raw surface pressure it came from, thin red, same axis.
    ps = res.p_surface_all if res.pressure_is_bhp else None
    if ps is not None:
        ps_clean, _ = _split_dropouts(ps, res.dropout_mask)
        xst, xsp = _decimate(t_h[kept], ps_clean[kept])
        ax.plot(xst, xsp, color=C.SURFACE_PRESSURE, lw=0.5, label="Surface Pressure",
                gid="surface_pressure")
        press_ylabel = "Pressure (psi)"  # the axis now carries both traces
        if excluded.any():
            xste, xspe = _decimate(t_h[excluded], ps_clean[excluded])
            ax.plot(xste, xspe, color=C.EXCLUDED, alpha=C.EXCLUDED_ALPHA, lw=0.5,
                    gid="surface_tail_excluded")
    # Masked dropouts as their own markers, not decimated with the main trace -- a handful of
    # masked samples inside a record with 10^5+ points would almost certainly fall between the
    # main trace's decimation stride and never get drawn.
    masked_idx = np.flatnonzero(np.isfinite(p_masked))
    _plot_dropout_markers(ax, t_h[masked_idx], p_masked[masked_idx])
    _plot_manual_spans(ax, state)

    ax.set_xlabel("Time from File Start (h)")
    ax.set_ylabel(press_ylabel, color=press_color)
    ax.tick_params(axis="y", labelcolor=press_color)
    ax.grid(True, alpha=0.3)

    if res.rate_all is not None:
        ax2 = ax.twinx()
        xrt, xr = _decimate(t_h, res.rate_all)
        ax2.plot(xrt, xr, color=C.RATE, lw=0.7, alpha=0.7)
        ax2.set_ylabel("Rate (bpm)", color=C.RATE)
        ax2.tick_params(axis="y", labelcolor=C.RATE)

    if state.start_idx is not None:
        ax.axvline(t_h[state.start_idx], color=C.INJECTION_START, ls=":", lw=1.0, alpha=0.6,
                   label="injection start", gid="start_ref")
    if state.shutin_idx is not None:
        ax.axvline(t_h[state.shutin_idx], color=C.SHUTIN, ls=":", lw=1.0, alpha=0.6,
                   label="shut-in", gid="shutin_ref")

    if interactive and has_trim_context:
        # Nothing to cut yet -> park at the last RAW sample, not resampled_full.dt[-1]. The
        # resampler only keeps a point per 30-psi drop, so on a slow falloff its last kept point
        # can sit well short of the record's end; parking there would show the line mid-plot with
        # ungrayed data to its right, implying a cut that isn't in effect. Releasing a drag at
        # that raw edge still clears (ui's commit clears at idx >= len(dt_full) - 1).
        trim_x_h = (cut_dt + res.t_shutin_s) / 3600.0 if cut_dt is not None else t_h[-1]
        ax.axvline(trim_x_h, color=C.TAIL_TRIM, ls="--", lw=1.8, gid="tail_trim")

    ax.set_title("Overview", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)

    # Pinned y-min 0: the pressure trace never reads below 0 psi, and a
    # gauge/BHP-conversion floor should always be visible relative to true zero, even though
    # that squashes a converted-BHP trace (~4800-6200 psi) into the top of the axes -- the
    # y-slider and Reset view are the escape. p_hi/pad use the whole record (kept + grayed),
    # same data-span idiom render_gfunction uses for its own pressure ylim.
    finite_p = np.isfinite(p)
    if not finite_p.any():
        return ViewDefaults()
    p_lo, p_hi = float(np.nanmin(p[finite_p])), float(np.nanmax(p[finite_p]))
    pad = 0.05 * max(p_hi - p_lo, 1.0)
    y2_color = C.RATE if res.rate_all is not None else None
    return ViewDefaults(ylim=nice_limits(0.0, p_hi + pad), y2lim=_rate_y2lim(res.rate_all),
                        y_color=press_color, y2_color=y2_color)


def render_injection(ax, td: TestData, state: PickState, res: DerivedResults,
                     full_record: bool = False) -> ViewDefaults:
    """Step 2: BHP (or surface P) and rate vs time, with injection-start / shut-in markers.

    The falloff tail can run for weeks and would otherwise dwarf the active-injection region in
    both the autoscaled extent and the x-slider's full range, so -- following ``render_isip``'s
    precedent of clamping the *plotted data* -- every trace is masked to the last nonzero rate +
    15 min before decimation when a rate channel exists and pumped at all; otherwise the full
    record is plotted, unclamped. When shut-in sits past the clamp, the clamp moves to shut-in +
    15 min instead, so the picks are never cut off.

    ``full_record`` (the "Show all data" button) plots the whole record with no default xlim, so
    the analyst can find a missed injection and drag the lines to it.
    """
    ax.clear()
    p = res.bhp_all if res.bhp_all is not None else np.full(td.n, np.nan)
    t_h = _hours(td.t_s)

    t_end_h = None
    if not full_record and res.rate_all is not None and np.any(res.rate_all > 0):
        last_active = int(np.where(res.rate_all > 0)[0][-1])
        t_end_h = t_h[last_active] + 0.25
        if state.shutin_idx is not None and t_h[state.shutin_idx] > t_end_h:
            t_end_h = t_h[state.shutin_idx] + 0.25
    m = (t_h <= t_end_h) if t_end_h is not None else np.ones_like(t_h, dtype=bool)

    xt, xp = _decimate(t_h[m], p[m])
    press_color, press_ylabel, press_label = _pressure_style(res)
    ax.plot(xt, xp, color=press_color, lw=0.8, label=press_label)
    ax.set_xlabel("Time from File Start (h)")
    ax.set_ylabel(press_ylabel, color=press_color)
    ax.tick_params(axis="y", labelcolor=press_color)
    ax.grid(True, alpha=0.3)

    if res.rate_all is not None:
        ax2 = ax.twinx()
        _, xr = _decimate(t_h[m], res.rate_all[m])
        ax2.plot(xt, xr, color=C.RATE, lw=0.7, alpha=0.7)
        ax2.set_ylabel("Rate (bpm)", color=C.RATE)
        ax2.tick_params(axis="y", labelcolor=C.RATE)

    if state.start_idx is not None:
        ax.axvline(t_h[state.start_idx], color=C.INJECTION_START, ls="--", lw=1.6,
                   label="injection start", gid="start")
    if state.shutin_idx is not None:
        ax.axvline(t_h[state.shutin_idx], color=C.SHUTIN, ls="-", lw=1.8,
                   label="shut-in", gid="shutin")

    # The default view zooms to the active injection region (the falloff tail can be weeks long);
    # the full autoscaled extent stays available for the caller to zoom back out to.
    xlim = None
    if full_record:
        pass  # no default: autoscale over the whole record
    elif state.start_idx is not None and state.shutin_idx is not None:
        span_h = max(t_h[state.shutin_idx] - t_h[state.start_idx], 0.25)
        xlim = (t_h[state.start_idx] - 0.5 * span_h, t_h[state.shutin_idx] + 2.0 * span_h)
    elif res.rate_all is not None:
        act = np.where(res.rate_all > 0.1)[0]
        if act.size:
            xlim = (max(0, t_h[act[0]] - 0.2), t_h[act[-1]] + 0.5)
    if xlim is not None and t_end_h is not None:
        xlim = (xlim[0], min(xlim[1], t_end_h))

    title = "Injection"
    if res.te_s:
        title += f"   te={res.te_s/60:.2f} min"
        if res.vinj is not None:
            title += f"   Vinj={res.vinj:.1f} bbl"
        if res.qmax_bpm is not None:
            title += f"   qmax={res.qmax_bpm:.2f} bpm"
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    y2_color = C.RATE if res.rate_all is not None else None
    y2lim = _rate_y2lim(res.rate_all[m]) if res.rate_all is not None else None
    return ViewDefaults(xlim=xlim, ylim=_pressure_ylim(p[m]), y2lim=y2lim, y_color=press_color,
                        y2_color=y2_color)


def _pressure_style(res: DerivedResults) -> tuple[str, str, str]:
    """(color, y-axis label, legend label) for the primary pressure trace. Unconverted surface
    pressure (res.pressure_is_bhp False: surface channel without both density and TVD) is red
    and never called BHP, on every step."""
    if res.pressure_is_bhp:
        return C.PRESSURE, "Bottomhole Pressure (psi)", "Bottomhole Pressure"
    return C.SURFACE_PRESSURE, "Surface Pressure (psi)", "Surface Pressure"


def render_isip(ax, td: TestData, state: PickState, res: DerivedResults) -> ViewDefaults:
    """Step 3: BHP vs time after shut-in; the apparent-ISIP tangent + extension to shut-in.

    The apparent ISIP always occurs just after shut-in, so the plotted data (and therefore the
    maximum extent the x-slider can zoom within) is deliberately clamped to shut-in -5 min .. +15
    min rather than the full falloff tail (which can run for days) -- the slider zooms further
    within that fixed window. The *default* view on first visit is a tighter -1..3 min, so the
    early-time shape near shut-in is visible without the user having to zoom in manually; the
    slider can still pan/zoom back out to the full -5..15 clamp.
    """
    ax.clear()
    if res.bhp_all is None or res.t_shutin_s is None:
        ax.set_title("Apparent ISIP -- Set Injection Window First", fontsize=10)
        return ViewDefaults()
    t_min = (td.t_s - res.t_shutin_s) / 60.0
    m = (t_min >= -5.0) & (t_min <= 15.0)
    p_clean, p_masked = _split_dropouts(res.bhp_all, res.dropout_mask)
    xt, xp = _decimate(t_min[m], p_clean[m])
    press_color, press_ylabel, _ = _pressure_style(res)
    ax.plot(xt, xp, color=press_color, lw=0.9)
    masked_idx = np.flatnonzero(m & np.isfinite(p_masked))
    _plot_dropout_markers(ax, t_min[masked_idx], p_masked[masked_idx])
    ax.axvline(0.0, color=C.SHUTIN, lw=1.2, label="shut-in")
    ax.set_xlabel("Time from Shut-In (min)")
    ax.set_ylabel(press_ylabel, color=press_color)
    ax.tick_params(axis="y", labelcolor=press_color)
    ax.grid(True, alpha=0.3)

    y2lim = None
    y2_color = None
    if res.rate_all is not None:
        ax2 = ax.twinx()
        xrt, xr = _decimate(t_min[m], res.rate_all[m])
        ax2.plot(xrt, xr, color=C.RATE, lw=0.7, alpha=0.7)
        ax2.set_ylabel("Rate (bpm)", color=C.RATE)
        ax2.tick_params(axis="y", labelcolor=C.RATE)
        y2lim = _rate_y2lim(res.rate_all[m])
        y2_color = C.RATE

    # The pressure default fits the default -1..3 min view, not the whole -5..15 min clamp:
    # injection pressure before -1 min would otherwise set the ceiling. The apparent-ISIP dot
    # is included so it is always on screen. The slider's outer range still spans everything.
    in_view = m & (t_min >= _ISIP_DEFAULT_XLIM[0]) & (t_min <= _ISIP_DEFAULT_XLIM[1])
    p_view = p_clean[in_view] if np.isfinite(p_clean[in_view]).any() else p_clean[m]
    if res.apparent_isip is not None:
        p_view = np.append(p_view, res.apparent_isip)
    ylim = _pressure_ylim(p_view)

    tg = state.isip_tangent
    if not state.isip_use_tangent:
        # Default: the apparent ISIP is the BHP at the shut-in sample, drawn at (0, P).
        if res.apparent_isip is not None:
            ax.plot(0.0, res.apparent_isip, "o", color=C.ISIP_LINE, gid="isip_shutin_dot")
            ax.set_title(f"Apparent ISIP = {res.apparent_isip:.0f} psi", fontsize=10)
        else:
            ax.set_title("Apparent ISIP -- No BHP at the Shut-In Sample", fontsize=10)
        ax.legend(loc="upper right", fontsize=8)
        return ViewDefaults(xlim=_ISIP_DEFAULT_XLIM, ylim=ylim, y2lim=y2lim,
                            y_color=press_color, y2_color=y2_color)
    if tg is not None:
        # tg lives on the seconds-since-file-start / psi-per-second convention td.t_s uses; this
        # axes plots minutes-from-shut-in, so convert before drawing -- ui.py's controller wiring
        # converts the same way (see _isip_pick_in_minutes/_isip_minutes_to_seconds).
        anchor_x_min = (tg.anchor_x - res.t_shutin_s) / 60.0
        slope_per_min = tg.slope * 60.0
        y_span = float(np.nanmax(xp) - np.nanmin(xp)) if xp.size else max(abs(tg.anchor_y), 1.0)
        _draw_tangent_construction(
            ax, anchor_x_min, tg.anchor_y, slope_per_min, ref_x=0.0, half=_ISIP_TANGENT_HALF_MIN,
            color=C.ISIP_LINE,
            gids={"segment": "isip_tangent_segment", "tick": "isip_tangent_tick",
                  "extension": "isip_tangent_extension"},
            tick_half_y=0.04 * y_span, label="ISIP tangent")
        ax.plot(0.0, res.apparent_isip, "o", color=C.ISIP_LINE, gid="isip_value_dot")
    if res.apparent_isip is not None:
        ax.set_title(f"Apparent ISIP = {res.apparent_isip:.0f} psi", fontsize=10)
    else:
        ax.set_title("Apparent ISIP -- Place the Tangent", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    return ViewDefaults(xlim=_ISIP_DEFAULT_XLIM, ylim=ylim, y2lim=y2lim,
                        y_color=press_color, y2_color=y2_color)


def render_gfunction(ax, td: TestData, state: PickState, res: DerivedResults) -> ViewDefaults:
    """Step 4: P and dP/dG vs G-time; contact + min-dP/dG markers; effective-ISIP line to G=0."""
    ax.clear()
    if res.diagnostics is None:
        ax.set_title("G-Function -- Need te and a Falloff", fontsize=10)
        # Recovery path: a pathological saved tail trim can leave <3 resampled points (the
        # diagnostics guard in compute_all fails), so there is nothing left to diagnose here.
        # The trim tool itself now lives on the Overview tab -- dragging it back right there is
        # the way out of this state.
        # A guard fire severe enough to leave <3 kept points lands here too (no trim needed) --
        # draw its excluded-tail preview the same as the main path, so it isn't silent just
        # because there were too few points left to diagnose.
        if res.guard_excluded_G is not None and len(res.guard_excluded_G):
            ax.plot(res.guard_excluded_G, res.guard_excluded_p, color=C.EXCLUDED,
                    alpha=C.GUARD_EXCLUDED_ALPHA, lw=0.8, gid="guard_excluded", zorder=0.5)
        return ViewDefaults()
    dg = res.diagnostics
    rs = res.resampled
    # Fainter still, and drawn under everything else: the raw tail past the guard boundary, not
    # admitted into these diagnostics (they're masked to interpret.resolve_tail_cut_dt's cutoff,
    # which stays at guard_dt unless the analyst drags an explicit override past it) -- it may or
    # may not have entered resampled_full itself (it does whenever a genuine further decline
    # resumes there), but either way it's excluded here. Left in autoscale on purpose -- the
    # 2x-G cap applied in compute_all is what bounds a runaway tail, not a view-limit clamp here.
    if res.guard_excluded_G is not None and len(res.guard_excluded_G):
        ax.plot(res.guard_excluded_G, res.guard_excluded_p, color=C.EXCLUDED,
                alpha=C.GUARD_EXCLUDED_ALPHA, lw=0.8, gid="guard_excluded", zorder=0.5)
    press_color, press_ylabel, press_label = _pressure_style(res)
    ax.plot(dg.G, rs.p, color=press_color, lw=1.2, marker=".", ms=3, label=press_label)
    ax.set_xlabel("G-Time")
    ax.set_ylabel(press_ylabel, color=press_color)
    ax.tick_params(axis="y", labelcolor=press_color)
    ax.grid(True, alpha=0.3)

    # The pressure axis must scale from the BHP data only -- the effective-ISIP tangent's dashed
    # extension (drawn below, on this same Axes) can swing to extreme psi values far outside the
    # real data, and the Axes' own autoscale would otherwise pick that up too.
    ylim = _pressure_ylim(rs.p)

    ax2 = ax.twinx()
    ax2.plot(dg.G, dg.dPdG, color=C.DERIVATIVE, lw=1.0, label="dP/dG")
    ax2.set_ylabel("dP/dG", color=C.DERIVATIVE)
    ax2.tick_params(axis="y", labelcolor=C.DERIVATIVE)
    y2lim = None
    # Autoscale the default view to the real derivative, masking the early water-hammer spike
    # out by G-time (same g_min convention as interpret.suggest_min_dpdg_index and the d2P/dG2
    # block below). A percentile over all samples was dominated by the spike (the resampled grid
    # is densest there), and the old hard 50 cap squashed any record whose real dP/dG ran higher.
    # The decay can still run past G=1, so its leading run is cut too (interpret.leading_spike_end),
    # as is a trailing end-of-record crash spike (interpret.terminal_spike_start).
    finite = np.isfinite(dg.dPdG) & (dg.G >= Y2_SCALE_G_MIN)
    idx = np.arange(len(dg.dPdG))
    trimmed = (finite & (idx >= interpret.leading_spike_end(dg.G, dg.dPdG, Y2_SCALE_G_MIN))
               & (idx < interpret.terminal_spike_start(dg.G, dg.dPdG)))
    if trimmed.any():
        finite = trimmed
    if not finite.any():        # whole record sits below G=1 -- scale from everything finite
        finite = np.isfinite(dg.dPdG)
    if finite.any():
        hi = float(np.nanmax(dg.dPdG[finite]))
        y2lim = (0.0, min(nice_limits(0.0, max(hi * 1.10, 1.0))[1], DPDG_VIEW_MAX))

    y3lim = None
    if state.show_d2pdg2:
        # A third y-axis, offset further right so it doesn't collide with the dP/dG twin's
        # ticks/label -- see D2_AXIS_GID (ui.py excludes it from the twin lookup/y2 slider, and
        # gives it no slider/persisted view of its own, decision D3).
        ax3 = ax.twinx()
        ax3.set_gid(D2_AXIS_GID)
        ax3.spines["right"].set_position(("axes", 1.12))
        ax3.plot(dg.G, dg.d2PdG2, color=C.SECOND_DERIVATIVE, lw=0.9, label="d2P/dG2",
                 gid="d2pdg2_curve")
        ax3.set_ylabel("d2P/dG2", color=C.SECOND_DERIVATIVE)
        ax3.tick_params(axis="y", labelcolor=C.SECOND_DERIVATIVE)
        # Scale from G >= 1 only (same g_min convention as interpret.suggest_min_dpdg_index):
        # the resampled grid is densest across the early water-hammer spike, so percentiles
        # over all samples would still be dominated by its huge |d2| values.
        finite_d2 = np.isfinite(dg.d2PdG2) & (dg.G >= 1.0)
        if not finite_d2.any():
            finite_d2 = np.isfinite(dg.d2PdG2)
        if finite_d2.any():
            lo, hi = np.percentile(dg.d2PdG2[finite_d2], [5, 95])
            pad = 0.10 * max(hi - lo, 1e-9)
            y3lim = nice_limits(lo - pad, hi + pad)

    if res.eff_isip_line_compliance is not None and res.effective_isip_compliance is not None:
        ln = res.eff_isip_line_compliance
        g_span = float(np.nanmax(dg.G) - np.nanmin(dg.G)) if len(dg.G) else 1.0
        y_span = (float(np.nanmax(rs.p) - np.nanmin(rs.p)) if len(rs.p)
                 else max(abs(ln.anchor_y), 1.0))
        _draw_tangent_construction(
            ax, ln.anchor_x, ln.anchor_y, ln.slope, ref_x=0.0, half=max(0.06 * g_span, 1e-6),
            color=C.ISIP_LINE,
            gids={"segment": "eff_isip_segment", "tick": "eff_isip_tick",
                  "extension": "eff_isip_extension"},
            tick_half_y=0.04 * y_span, label="effective-ISIP line", draw_tick=False)
        ax.plot(0.0, res.effective_isip_compliance, "o", color=C.ISIP_LINE)
    # The triangle is only meaningful for C-A (rel-min anchor) / C-B (inflection seed) -- C-C/C-D
    # have no contact rule and blank leaves it hidden until a scenario is chosen (decision 4).
    if state.min_dpdg_G is not None and state.closure_scenario.startswith(("C-A", "C-B")):
        y = float(np.interp(state.min_dpdg_G, dg.G, dg.dPdG))
        ax2.plot(state.min_dpdg_G, y, marker="v", color=C.DERIVATIVE, ms=8, label="min dP/dG",
                gid="min_dpdg_point")
    # C-A only: if dP/dG never rises 10% above the picked min, suggest_contact_clear_index finds
    # no contact -- draw the 110% threshold it's checking against so the analyst can see the
    # curve never reaches it. No text label: it sat behind the legend.
    if state.closure_scenario.startswith("C-A") and state.min_dpdg_G is not None:
        min_idx = int(np.argmin(np.abs(dg.G - state.min_dpdg_G)))
        threshold = 1.10 * dg.dPdG[min_idx]
        # An all-NaN dPdG at the pick makes threshold itself NaN -- suggest_contact_clear_index
        # already returns None for it (never rises), but an axhline at NaN is invisible, so gate
        # on finiteness too.
        if np.isfinite(threshold) and interpret.suggest_contact_clear_index(dg.dPdG, min_idx) is None:
            ax2.axhline(threshold, ls="--", color=C.DERIVATIVE, alpha=0.6,
                        gid="clear_threshold_line")
    if state.contact_G is not None and res.contact_pressure is not None:
        ax.plot(state.contact_G, res.contact_pressure, "s", color=C.PICK, ms=7, label="contact",
               gid="contact_point")
    if state.contact_G is not None:
        ax.axvline(state.contact_G, color=C.PICK, ls=":", lw=1.2, gid="contact_vline")

    title = "G-Function"
    if res.effective_isip_compliance is not None:
        title += f"   eff.ISIP={res.effective_isip_compliance:.0f}"
    if res.shmin_compliance is not None:
        title += f"   Shmin(compl)={res.shmin_compliance:.0f}"
    if res.shmin_rapid is not None:
        title += f"   Shmin(rapid)={interpret.format_shmin_rapid(res.shmin_rapid, verbose=True)}"
    title += f"   ({state.closure_scenario or '?'})"
    ax.set_title(title, fontsize=10)
    ax.legend(loc="lower left", fontsize=8)
    return ViewDefaults(ylim=ylim, y2lim=y2lim, y3lim=y3lim, y_color=press_color,
                        y2_color=C.DERIVATIVE)


def render_tangent(ax, td: TestData, state: PickState, res: DerivedResults) -> ViewDefaults:
    """Step 5: BHP and G*dP/dG vs G-time -- mirrors ``render_gfunction``'s twinx layout (BHP on
    the primary/left axis, G*dP/dG on the twin/right). The through-origin line is still picked
    on the G*dP/dG curve and lives on the twin axis, but the closure marker now rides the BHP
    curve on the primary axis."""
    ax.clear()
    if res.diagnostics is None:
        ax.set_title("Tangent Method -- Need a Falloff", fontsize=10)
        return ViewDefaults()
    dg = res.diagnostics
    rs = res.resampled
    press_color, press_ylabel, press_label = _pressure_style(res)
    ax.plot(dg.G, rs.p, color=press_color, lw=1.2, marker=".", ms=3, label=press_label)
    ax.set_xlabel("G-Time")
    ax.set_ylabel(press_ylabel, color=press_color)
    ax.tick_params(axis="y", labelcolor=press_color)
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(dg.G, dg.GdPdG, color=C.DERIVATIVE, lw=1.0, marker=".", ms=3, label="G*dP/dG")
    ax2.set_ylabel("G*dP/dG", color=C.DERIVATIVE)
    ax2.tick_params(axis="y", labelcolor=C.DERIVATIVE)
    y2lim = None
    # Clip the early water-hammer spike off-scale (default view only). Same scale the closure
    # tolerance is measured in (interpret.tangent_visual_gap).
    top = interpret.tangent_view_y_top(dg.G, dg.GdPdG)
    if np.isfinite(top):
        y2lim = nice_limits(0.0, max(top, 1.0))

    if state.tangent_uninterpretable:
        # Explicit negative finding, same precedent as render_stiffness's stiffness_no_upturn:
        # the curves stay as evidence, but no line/marker, and shmin_tangent is None.
        title = "Tangent Method -- Uninterpretable (Shmin Not Reported)"
    else:
        if state.closure_slope is not None:
            gg = np.array([0.0, float(dg.G.max())])
            ax2.plot(gg, state.closure_slope * gg, color=C.GUIDE, ls="--", lw=1.2,
                    label="through-origin", gid="closure_line_segment")
        if state.closure_G is not None:
            yv = float(np.interp(state.closure_G, dg.G, rs.p))
            ax.plot(state.closure_G, yv, "o", color=C.PICK, ms=7, label="closure",
                    gid="closure_point")
            ax.axvline(state.closure_G, color=C.PICK, ls=":", lw=1.2, gid="closure_vline")
        title = "Tangent Method"
        if res.shmin_tangent is not None:
            title += f"   Shmin(tangent)={res.shmin_tangent:.0f}"
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    return ViewDefaults(ylim=_pressure_ylim(rs.p), y2lim=y2lim, y_color=press_color,
                        y2_color=C.DERIVATIVE)


def render_loglog(ax, td: TestData, state: PickState, res: DerivedResults) -> ViewDefaults:
    """Step 6: log-log dp and t*dP/dt vs shut-in time; selected window + fitted slope."""
    ax.clear()
    if res.diagnostics is None:
        ax.set_title("Log-Log -- Need a Falloff", fontsize=10)
        return ViewDefaults()
    dg = res.diagnostics
    good = (dg.t > 0) & (dg.dp > 0)
    ax.loglog(dg.t[good], dg.dp[good], color=C.PRESSURE, lw=1.0, marker=".", ms=3, label="dp")
    tgood = (dg.t > 0) & (dg.tdpdt > 0)
    ax.loglog(dg.t[tgood], dg.tdpdt[tgood], color=C.DERIVATIVE, lw=1.0, marker=".", ms=3,
              label="t*dP/dt")
    ax.set_xlabel("Shut-In Time (s)")
    ax.set_ylabel("dp, t*dP/dt (psi)")
    ax.grid(True, which="both", alpha=0.3)

    if pp_from_peak(state) and res.pce_peak_t is not None:
        # PC-E: the -1/2 line from the peak that the pore pressure is extrapolated on, drawn
        # one decade past the last sample. dataLim is snapshotted and restored (same pattern as
        # _plot_dropout_markers) so the extension never widens the default view.
        t_pk, d_pk = res.pce_peak_t, res.pce_peak_tdpdt
        tt = np.geomspace(t_pk, max(float(dg.t[-1]), t_pk) * 10.0, 50)
        saved_points = ax.dataLim.get_points().copy()
        ax.plot(tt, d_pk * (tt / t_pk) ** -0.5, color=C.PORE_PRESSURE, ls="--", lw=1.3,
                label="-1/2 from peak", gid="pce_halfslope", scalex=False, scaley=False)
        ax.dataLim.set_points(saved_points)
        ax.plot(t_pk, d_pk, "o", color=C.PICK, ms=7, label="peak", gid="pce_peak")
        ax.set_title(f"Log-Log   ({state.postclosure_scenario})   -1/2 From Peak", fontsize=10)
    elif loglog_window_suppressed(state):
        # PC-E/PC-F: no straight trend, so no window or slope (the pick stays in state).
        ax.set_title(f"Log-Log   ({state.postclosure_scenario})", fontsize=10)
    elif state.loglog_window is not None:
        lo, hi = state.loglog_window
        ax.axvspan(lo, hi, color=C.WINDOW, alpha=C.WINDOW_ALPHA)
        s = "?" if res.loglog_slope is None else f"{res.loglog_slope:.2f}"
        ax.set_title(f"Log-Log   Window Slope={s}   ({state.postclosure_scenario or '?'})",
                     fontsize=10)
    else:
        ax.set_title("Log-Log -- Select the Late-Time Window", fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    return ViewDefaults()


def render_porepressure(ax, td: TestData, state: PickState, res: DerivedResults) -> ViewDefaults:
    """Step 7: P vs t^-1/2 or t^-1 with the fitted line extended to the intercept."""
    ax.clear()
    if res.diagnostics is None:
        ax.set_title("Pore Pressure -- Need a Falloff", fontsize=10)
        return ViewDefaults()
    dg = res.diagnostics
    expo = -0.5 if state.pp_axis == "tm12" else -1.0
    x = dg.t ** expo
    press_color, press_ylabel, _ = _pressure_style(res)
    ax.plot(x, dg.p, color=press_color, lw=1.0, marker=".", ms=3)
    ax.set_xlabel("t^(-1/2)" if state.pp_axis == "tm12" else "t^(-1)")
    ax.set_ylabel(press_ylabel, color=press_color)
    ax.tick_params(axis="y", labelcolor=press_color)
    ax.grid(True, alpha=0.3)
    xmax = float(np.nanmax(x)) if x.size else 1.0

    if pp_from_peak(state):
        if res.pce_peak_t is not None and res.pore_pressure is not None:
            # P = Pp + m*t^(-1/2) from the peak to t -> inf (x = 0); a curve on the t^-1 axis.
            tt = np.geomspace(res.pce_peak_t, res.pce_peak_t * 1e8, 60)
            xs = np.append(tt ** expo, 0.0)
            ys = np.append(res.pore_pressure + res.pore_pressure_slope * tt ** -0.5,
                           res.pore_pressure)
            ax.plot(xs, ys, color=C.PORE_PRESSURE, ls="--", lw=1.3, gid="pce_extrapolation")
            ax.plot(res.pce_peak_t ** expo, res.pce_peak_p, "o", color=C.PICK, ms=6)
            ax.plot(0.0, res.pore_pressure, "o", color=C.PORE_PRESSURE)
            ax.set_title(f"Pore Pressure = {res.pore_pressure:.0f} psi  (-1/2 From Peak)",
                         fontsize=10)
        else:
            ax.set_title("Pore Pressure -- Pick the Peak on Log-Log", fontsize=10)
        return _porepressure_view(state, dg, res, press_color)

    if state.pp_window is not None:
        lo, hi = state.pp_window
        x_lo = 0.0 if not np.isfinite(hi) else hi ** expo
        x_hi = lo ** expo if lo > 0 else xmax
        ax.axvspan(x_lo, x_hi, color=C.WINDOW, alpha=C.WINDOW_ALPHA)

    if state.pp_window is not None and res.pore_pressure is not None:
        lo, hi = state.pp_window
        m = (dg.t >= lo) & (dg.t <= hi)
        if m.sum() >= 2:
            from .interpret import fit_line
            slope, intercept = fit_line(x[m], dg.p[m])
            xr = np.array([0.0, x[m].max()])
            ax.plot(xr, intercept + slope * xr, color=C.PORE_PRESSURE, ls="--", lw=1.3)
            ax.plot(0.0, res.pore_pressure, "o", color=C.PORE_PRESSURE)
            pmin = float(dg.p[m].min())
            if res.pore_pressure >= pmin:
                ax.set_title(
                    f"Pore Pressure = {res.pore_pressure:.0f} psi  (>= Observed -- Adjust Window)",
                    fontsize=10)
            else:
                ax.set_title(f"Pore Pressure = {res.pore_pressure:.0f} psi", fontsize=10)
        else:
            ax.set_title(f"Pore Pressure = {res.pore_pressure:.0f} psi", fontsize=10)
    else:
        ax.set_title("Pore Pressure -- Select the Late-Time Window", fontsize=10)
    return _porepressure_view(state, dg, res, press_color)


def _porepressure_view(state: PickState, dg, res: DerivedResults, press_color) -> ViewDefaults:
    xhi = 0.05 if state.pp_axis == "tm12" else 0.0025
    # The pick sits at x = 0, normally below every observed sample; keep it in the default view.
    p_view = dg.p if res.pore_pressure is None else np.append(dg.p, res.pore_pressure)
    return ViewDefaults(xlim=(0.0, xhi), ylim=_pressure_ylim(p_view), y_color=press_color)


def render_stiffness(ax, td: TestData, state: PickState, res: DerivedResults) -> ViewDefaults:
    """Step 8: relative system stiffness (semilog-y) vs effective pressure (URTeC-2019-123
    A.8/A.9) -- the upturn where the fracture walls come into contact gives a fourth,
    comparison-only Shmin estimate. Needs the min-dP/dG pick and a pore-pressure estimate (the
    h-function's Pres term); skipped end to end under PC-F/PC-X (model.skipped_steps), which
    never yields one. state.stiffness_no_upturn records the negative finding "no slope change
    apparent" -- the curve still draws, but no pick vline/marker and a title saying so."""
    ax.clear()
    if res.stiffness_S is None:
        ax.set_title("Stiffness -- Requires the min-dP/dG Pick and a Pore-Pressure Estimate",
                     fontsize=10)
        return ViewDefaults()
    p_eff, S = res.stiffness_p_eff[1:], res.stiffness_S
    finite_pos = np.isfinite(S) & (S > 0)
    if not finite_pos.any():
        # A log-scaled axis needs at least one positive sample to have anything to draw --
        # set_yscale("log") ahead of this check drew an axis-only plot plus a matplotlib
        # UserWarning ("Data has no positive values..."). Guard branch instead, same style as
        # the missing-arrays branch above.
        title = "Stiffness -- No Positive Relative-Stiffness Samples to Plot"
        if closure_uninterpretable(state):
            title += " -- G-Function Uninterpretable (Shmin Not Reported)"
        elif state.stiffness_no_upturn:
            # The recorded finding still belongs in the title even though there is nothing to
            # plot -- otherwise this branch silently drops it and looks like a data problem
            # rather than the analyst's own "no slope change apparent" call.
            title += " -- No Slope Change Apparent (Shmin Not Reported)"
        ax.set_title(title, fontsize=10)
        return ViewDefaults()
    ax.plot(p_eff, S, color=C.PRESSURE, lw=1.0, marker=".", ms=3)
    ax.set_yscale("log")
    ax.set_xlabel("Effective Pressure (psi)")
    ax.set_ylabel("Relative Stiffness")
    ax.grid(True, which="both", alpha=0.3)

    if closure_uninterpretable(state):
        # C-X: the min-dP/dG anchor of this curve can't be trusted, so shmin_stiffness is None
        # (model.compute_all). Same no-vline/no-marker treatment as stiffness_no_upturn.
        ax.set_title("Relative Stiffness -- G-Function Uninterpretable (Shmin Not Reported)",
                     fontsize=10)
    elif state.stiffness_no_upturn:
        # Explicit negative finding, same precedent as closure scenario C-C's "no contact ->
        # no Shmin": the curve is still evidence (kept on the plot and in PNG exports), but
        # there is no upturn to mark, so no vline/marker -- and shmin_stiffness is None
        # (model.compute_all), so no title should imply otherwise.
        ax.set_title("Relative Stiffness -- No Slope Change Apparent (Shmin Not Reported)",
                     fontsize=10)
    elif state.stiffness_pick_P is not None:
        ax.axvline(state.stiffness_pick_P, color=C.STIFFNESS_PICK, ls="--", lw=1.4,
                   gid="stiffness_pick")
        # A small non-draggable marker at the curve intersection, for readability only. p_eff is
        # no longer monotonic (rs.p can rise as well as fall), so np.interp -- which needs an
        # ascending x -- is no longer valid here; snap to the nearest finite sample's S value
        # instead (S aligns with p_eff[1:] one for one, so the same index into either array is
        # correct), same style as picks._nearest.
        finite_eff = np.isfinite(p_eff)
        if finite_eff.any():
            i_near = int(np.argmin(np.abs(p_eff[finite_eff] - state.stiffness_pick_P)))
            s_at_pick = float(S[finite_eff][i_near])
            ax.plot(state.stiffness_pick_P, s_at_pick, "o", color=C.STIFFNESS_PICK, ms=5)
        if res.shmin_stiffness is not None:
            ax.set_title(f"Stiffness   Shmin(stiffness)={res.shmin_stiffness:.0f} psi",
                         fontsize=10)
        else:
            ax.set_title("Stiffness", fontsize=10)
    else:
        ax.set_title("Stiffness -- Pick the Upturn", fontsize=10)

    finite_p = p_eff[np.isfinite(p_eff)]
    xlim = None
    if finite_p.size:
        p_lo, p_hi = float(np.nanmin(finite_p)), float(np.nanmax(finite_p))
        pad = 0.05 * max(p_hi - p_lo, 1.0)
        xlim = (p_lo - pad, p_hi + pad)
    # finite_pos is already known non-empty -- the all-non-positive case returned above.
    y_lo, y_hi = float(np.nanmin(S[finite_pos])), float(np.nanmax(S[finite_pos]))
    ylim = nice_log_limits(y_lo * 0.8, y_hi * 1.25)  # log-safe pad, then whole decades
    return ViewDefaults(xlim=xlim, ylim=ylim, y_color=C.PRESSURE)


RENDERERS = {
    "overview": render_overview,
    "injection": render_injection,
    "isip": render_isip,
    "gfunction": render_gfunction,
    "tangent": render_tangent,
    "loglog": render_loglog,
    "porepressure": render_porepressure,
    "stiffness": render_stiffness,
}


@dataclass
class ViewState:
    """The resolved (non-optional) view actually applied to a step's Axes: primary xlim/ylim,
    and the twin axes' ylim if that step has one."""
    xlim: tuple[float, float]
    ylim: tuple[float, float]
    y2lim: Optional[tuple[float, float]] = None


@dataclass
class StepView:
    """What ``apply_step_view`` resolved and applied. ``full_*`` are the outer (slider) ranges;
    ``twin`` is the step's secondary-y Axes (never the d2P/dG2 axis), or None."""
    view: ViewState
    full_x: tuple[float, float]
    full_y: tuple[float, float]
    full_y2: Optional[tuple[float, float]]
    twin: Optional[object]


def apply_step_view(step_key: str, ax, defaults: ViewDefaults,
                    stored: Optional[ViewState] = None) -> StepView:
    """Resolve and apply a step's view to ``ax`` and its twins. Call right after the renderer,
    before any other Axes (sliders) are added to the figure.

    Outer ranges start from the renderer's autoscale. A concrete ``defaults.ylim``/``y2lim`` is
    unioned in, so a default reaching outside the autoscale (Overview's y-min 0, the 3x rate
    ceiling) stays inside a slider's range; otherwise the slider's valinit clamping snaps the view
    off the default on first touch. Gfunction instead REPLACES full_y with ``defaults.ylim``
    (the effective-ISIP tangent extension can swing the Axes' autoscale to extreme psi) and
    clamps full_y2 to 0..``DPDG_VIEW_MAX`` after the union.

    The view is ``stored`` unchanged when given (pan/zoom survives a recompute), else each axis's
    default, falling back to its outer range. The d2P/dG2 axis gets ``defaults.y3lim`` fresh
    every call and is never part of the stored view (decision D3)."""
    def union(a, b):
        return (min(a[0], b[0]), max(a[1], b[1]))

    others = [a for a in ax.figure.axes if a is not ax]
    twin = next((a for a in others if a.get_gid() != D2_AXIS_GID), None)
    d2 = next((a for a in others if a.get_gid() == D2_AXIS_GID), None)
    gfunction = step_key == "gfunction"

    full_x = ax.get_xlim()
    full_y = ax.get_ylim()
    if defaults.ylim is not None:
        full_y = defaults.ylim if gfunction else union(full_y, defaults.ylim)
    full_y2 = twin.get_ylim() if twin is not None else None
    if full_y2 is not None:
        if defaults.y2lim is not None:
            full_y2 = union(full_y2, defaults.y2lim)
        if gfunction:
            full_y2 = (max(full_y2[0], 0.0), min(full_y2[1], DPDG_VIEW_MAX))

    view = stored if stored is not None else ViewState(
        xlim=defaults.xlim if defaults.xlim is not None else full_x,
        ylim=defaults.ylim if defaults.ylim is not None else full_y,
        y2lim=defaults.y2lim if defaults.y2lim is not None else full_y2,
    )
    ax.set_xlim(view.xlim)
    ax.set_ylim(view.ylim)
    if twin is not None and view.y2lim is not None:
        twin.set_ylim(view.y2lim)
    if d2 is not None and defaults.y3lim is not None:
        d2.set_ylim(defaults.y3lim)
    return StepView(view=view, full_x=full_x, full_y=full_y, full_y2=full_y2, twin=twin)


def render_step_figure(step_key: str, td: TestData, state: PickState, res: DerivedResults,
                       stored_view: Optional[tuple] = None,
                       figsize: tuple[float, float] = (9, 6)) -> Figure:
    """Render one step onto an offscreen ``Figure`` through ``apply_step_view``, the same
    view resolution ui.refresh uses, so an exported PNG matches what the analyst was looking at
    (``stored_view`` as an ``(xlim, ylim, y2lim)`` tuple) or the renderer's own default.

    No Tkinter -- this and ``save_all_step_pngs`` are called by ``ui._finish`` but could equally
    run headlessly for tests, per the module-level invariant.
    """
    fig = Figure(figsize=figsize)
    ax = fig.add_subplot(111)
    # Overview's tail-trim line is a live-canvas control, not part of the interpretation, so an
    # exported PNG never carries a line the analyst can't actually drag.
    kwargs = {"interactive": False} if step_key == "overview" else {}
    defaults = RENDERERS[step_key](ax, td, state, res, **kwargs)
    stored = ViewState(*stored_view) if stored_view is not None else None
    apply_step_view(step_key, ax, defaults, stored)

    d2_on = any(a.get_gid() == D2_AXIS_GID for a in fig.axes)
    right = 0.80 if (step_key == "gfunction" and d2_on) else 0.90
    fig.subplots_adjust(left=0.10, right=right, bottom=0.16, top=0.90)
    return fig


SUMMARY_SIZE = (11.0, 8.0)  # inches, the Expanded results chart and its exported PNG


def _summary_ladder(ax, cv) -> bool:
    """Horizontal dot chart: one row per reported pressure on a shared psi axis. Missing values
    are omitted, never 0."""
    shmin_label, shmin = (("Shmin compliance", cv.shmin_compliance)
                          if cv.shmin_compliance is not None or cv.shmin_rapid is None
                          else ("Shmin rapid", cv.shmin_rapid))
    entries = [  # (label, value, color)
        ("apparent ISIP", cv.apparent_isip, C.ISIP_LINE),
        ("eff ISIP compliance", cv.eff_isip_compliance, C.ISIP_LINE),
        ("eff ISIP tangent", cv.eff_isip_tangent, C.ISIP_LINE),
        ("eff ISIP variable", cv.eff_isip_variable, C.ISIP_LINE),
        (shmin_label, shmin, C.LBRT_RED),
        ("Shmin tangent", cv.shmin_tangent, C.LBRT_RED),
        ("Shmin variable", cv.shmin_variable, C.LBRT_RED),
        ("Shmin Liberty", cv.shmin_liberty, C.LBRT_RED),
        ("Shmin stiffness", cv.shmin_stiffness, C.LBRT_RED),
        ("pore pressure", cv.pore_pressure, C.PORE_PRESSURE),
    ]
    entries = [e for e in entries if e[1] is not None]
    ax.set_gid("summary_ladder")
    ax.set_title("Reported Pressures", fontsize=10, pad=36)
    if not entries:
        ax.text(0.5, 0.5, "no pressures yet", ha="center", va="center",
                transform=ax.transAxes, color=C.MID_GREY)
        ax.set_xticks([]); ax.set_yticks([])
        return False
    ys = np.arange(len(entries))[::-1]
    for y, (_, v, color) in zip(ys, entries):
        ax.plot([v], [y], "o", color=color, ms=7)
        ax.annotate(f"{v:.0f}", (v, y), xytext=(0, 7), textcoords="offset points",
                    ha="center", fontsize=8)
    ax.set_yticks(ys)
    ax.set_yticklabels([e[0] for e in entries])
    ax.set_ylim(-0.7, len(entries) - 0.3)
    vals = [e[1] for e in entries]
    span = max(max(vals) - min(vals), 1.0)
    ax.set_xlim(min(vals) - 0.1 * span, max(vals) + 0.1 * span)
    ax.set_xlabel("Pressure (psi)")
    ax.grid(axis="x", alpha=0.3)
    tvd = cv.tvd_ft
    if tvd is not None:
        sec = ax.secondary_xaxis("top", functions=(lambda x: x / tvd, lambda g: g * tvd))
        sec.set_xlabel("psi/ft")
    return True


def _summary_breakdown(ax, cv) -> bool:
    """Per method, a floating bar Shmin -> reference ISIP (net pressure) and reference ISIP ->
    apparent ISIP (complexity): Shmin + net + complexity = apparent ISIP."""
    ax.set_gid("summary_breakdown")
    ax.set_title("Shmin + Net + Complexity\n= Apparent ISIP", fontsize=10)
    rows = []
    for name, shmin, net in (("compliance", cv.shmin_compliance, cv.net_compliance),
                             ("tangent", cv.shmin_tangent, cv.net_tangent),
                             ("variable", cv.shmin_variable, cv.net_variable)):
        if shmin is not None and net is not None:
            rows.append((name, shmin, net))
    if not rows:
        ax.text(0.5, 0.5, "no net pressure yet", ha="center", va="center",
                transform=ax.transAxes, color=C.MID_GREY)
        ax.set_xticks([]); ax.set_yticks([])
        return False
    cx = cv.complexity
    ys = np.arange(len(rows))[::-1]
    lo = hi = None
    for y, (name, shmin, net) in zip(ys, rows):
        ax.barh(y, net, left=shmin, height=0.5, color=C.NET_PRESSURE,
                label="net pressure" if y == ys[0] else None)
        ax.text(shmin + net / 2, y, f"{net:.0f}", ha="center", va="center", fontsize=8,
                color=C.BAR_LABEL)
        edges = [shmin, shmin + net]
        if cx is not None:
            ax.barh(y, cx, left=shmin + net, height=0.5, color=C.COMPLEXITY,
                    label="complexity" if y == ys[0] else None)
            ax.text(shmin + net + cx / 2, y, f"{cx:.0f}", ha="center", va="center",
                    fontsize=8)
            edges.append(shmin + net + cx)
        lo = min(edges) if lo is None else min(lo, min(edges))
        hi = max(edges) if hi is None else max(hi, max(edges))
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows])
    span = max(hi - lo, 1.0)
    ax.set_xlim(lo - 0.1 * span, hi + 0.1 * span)
    ax.set_xlabel("Pressure (psi)")
    # Below the axes so it never covers a bar; constrained layout makes room for it.
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False)
    ax.grid(axis="x", alpha=0.3)
    return True


def _summary_closure(ax, cv) -> bool:
    """BHP vs G-time with each closure pick marked at its reported pressure, labelled with G and
    the closure time in minutes. The compliance pick is drawn at Shmin (contact - 75 psi), so it
    sits below the curve; the others are read off the curve."""
    ax.set_gid("summary_closure")
    ax.set_title("Closure Picks in G-Time", fontsize=10)
    entries = [("min dP/dG", cv.min_dpdg_G, cv.min_dpdg_p, cv.min_dpdg_tc_s, C.MID_GREY),
               ("compliance (contact - 75)", cv.G_compliance, cv.shmin_compliance,
                cv.tc_compliance_s, C.LBRT_RED),
               ("variable", cv.G_variable, cv.shmin_variable, cv.tc_variable_s, C.LBRT_DEEP_RED),
               ("tangent closure", cv.G_tangent, cv.shmin_tangent, cv.tc_tangent_s, C.RATE_BLUE)]
    entries = [e for e in entries if e[1] is not None and e[2] is not None]
    if cv.curve_G is None and not entries:
        ax.text(0.5, 0.5, "no closure picks yet", ha="center", va="center",
                transform=ax.transAxes, color=C.MID_GREY)
        ax.set_xticks([]); ax.set_yticks([])
        return False
    if cv.curve_G is not None:
        ax.plot(cv.curve_G, cv.curve_p, "-", color=C.PRESSURE, lw=1.2, label="BHP")
    for name, G, p, tc_s, color in entries:
        ax.plot([G], [p], "o", color=color, ms=8, label=name, zorder=3)
        text = f"G = {G:.2f}" + (f", tc = {tc_s / 60.0:.2f} min" if tc_s is not None else "")
        ax.annotate(text, (G, p), xytext=(8, 6), textcoords="offset points", fontsize=8)
    ax.set_xlabel("G-time")
    ax.set_ylabel("BHP (psi)")
    ax.grid(alpha=0.3)
    if entries:
        ax.legend(fontsize=8, loc="upper right")
    return True


def render_summary(fig: Figure, state: PickState, res: DerivedResults) -> None:
    """Three summary charts on ``fig`` (cleared first): the pressure ladder, the Shmin / net /
    complexity breakdown, and the closure picks in G-time. Builds its own axes -- it is not a
    step renderer, so the step view-limit rules do not apply. Values that are None are omitted,
    never plotted at 0, and values whose owning step is not visited are omitted
    like the tables (summary.chart_values); with nothing to show the figure carries a "No results yet" message."""
    fig.clear()
    cv = summary.chart_values(state, res)
    if not cv.has_any():
        fig.text(0.5, 0.5, "No results yet", ha="center", va="center", fontsize=14,
                 color=C.MID_GREY)
        return
    # Constrained layout sizes the margins to the tick labels and titles, so the charts fit
    # whatever width the results window gives the canvas (fixed fractions clipped the labels).
    fig.set_layout_engine("constrained", h_pad=0.08, w_pad=0.08)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1.0])
    _summary_ladder(fig.add_subplot(gs[0, 0]), cv)
    _summary_breakdown(fig.add_subplot(gs[0, 1]), cv)
    _summary_closure(fig.add_subplot(gs[1, :]), cv)


def save_all_step_pngs(out_dir: str, td: TestData, state: PickState, res: DerivedResults,
                       views: dict[str, Optional[tuple]], dpi: int = 150) -> list[str]:
    """Render every step's current view to a numbered PNG in ``out_dir``, in ``STEP_KEYS``
    order, then the Expanded-results chart as ``<n+1>_summary.png`` (always written).
    Returns the written paths in that order. Steps the workflow leaves out
    (``skipped_steps``; PC-F/PC-X drop porepressure and stiffness) are omitted, but the numbering
    still runs over every step so the other filenames are unaffected. Used by ``ui._finish``,
    but headless/Tkinter-free like the rest of this module."""
    paths = []
    skipped = skipped_steps(state)
    for i, key in enumerate(STEP_KEYS, start=1):
        if key in skipped:
            continue
        fig = render_step_figure(key, td, state, res, views.get(key))
        path = os.path.join(out_dir, f"{i}_{key}.png")
        fig.savefig(path, dpi=dpi)
        paths.append(path)
    fig = Figure(figsize=SUMMARY_SIZE)
    render_summary(fig, state, res)
    path = os.path.join(out_dir, f"{len(STEP_KEYS) + 1}_summary.png")
    fig.savefig(path, dpi=dpi)
    paths.append(path)
    return paths
