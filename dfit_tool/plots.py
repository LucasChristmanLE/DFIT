"""Static rendering of each workflow step onto a matplotlib Axes/Figure.

Each ``render_*`` takes an Axes, the loaded ``TestData``, the ``PickState``, and the
``DerivedResults`` and draws the plot plus whatever picks currently exist. The interactive layer
(picks.py) updates the PickState and calls the matching render to refresh.

Renderers no longer set view limits (``set_xlim``/``set_ylim``) on the Axes -- they leave the
Axes autoscaled to the full data extent and instead return a ``ViewDefaults`` describing the
view the caller should apply on first visit to a step. Callers (ui.py) own view state from
there: they read the autoscaled extent, resolve it against ``ViewDefaults``, and apply the
result to the Axes themselves.

Matplotlib only -- no Tkinter -- so figures can be produced headlessly (Agg) for verification.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from matplotlib.figure import Figure

from . import interpret
from .model import (DerivedResults, PickState, closure_uninterpretable, porepressure_skipped,
                    stiffness_skipped)
from .io_load import TestData

_MAX_POINTS = 6000  # display decimation cap for the raw (dense) traces

# Solid-segment half-width (minutes) for the apparent-ISIP tangent construction; must keep both
# endpoints inside render_isip's default (-1, 3)-min view for the seeded anchor (~+1 min) so the
# AnchorLineController rotate handles stay reachable.
_ISIP_TANGENT_HALF_MIN = 0.75


D2_AXIS_GID = "d2pdg2_axis"  # gid on the gfunction step's optional third (d2P/dG2) twin axes

DPDG_VIEW_MAX = 500.0   # hard ceiling on the gfunction dP/dG axis: default view AND slider range
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
    saved_points = ax.dataLim.get_points().copy()
    ax.plot(x, y, color="magenta", marker="o", ms=3, ls="none", label="masked dropout",
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
    rise-guard boundary or a sub-100-psi surface-pressure crash on the step's first visit (or at
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
    press_color = "black" if res.pressure_is_bhp else "tab:red"
    press_label = "bottomhole pressure" if res.pressure_is_bhp else "pressure"

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
        ax.plot(xte, xpe, color="0.75", lw=0.8, gid="tail_excluded")
    # Converted-BHP record: overlay the raw surface pressure it came from, thin red, same axis.
    ps = res.p_surface_all if res.pressure_is_bhp else None
    if ps is not None:
        ps_clean, _ = _split_dropouts(ps, res.dropout_mask)
        xst, xsp = _decimate(t_h[kept], ps_clean[kept])
        ax.plot(xst, xsp, color="tab:red", lw=0.5, label="surface pressure", gid="surface_pressure")
        if excluded.any():
            xste, xspe = _decimate(t_h[excluded], ps_clean[excluded])
            ax.plot(xste, xspe, color="0.75", lw=0.5, gid="surface_tail_excluded")
    # Masked dropouts as their own markers, not decimated with the main trace -- a handful of
    # masked samples inside a record with 10^5+ points would almost certainly fall between the
    # main trace's decimation stride and never get drawn.
    masked_idx = np.flatnonzero(np.isfinite(p_masked))
    _plot_dropout_markers(ax, t_h[masked_idx], p_masked[masked_idx])

    ax.set_xlabel("time from file start (h)")
    ax.set_ylabel("pressure (psi)", color=press_color)
    ax.tick_params(axis="y", labelcolor=press_color)
    ax.grid(True, alpha=0.3)

    if res.rate_all is not None:
        ax2 = ax.twinx()
        xrt, xr = _decimate(t_h, res.rate_all)
        ax2.plot(xrt, xr, color="tab:blue", lw=0.7, alpha=0.7)
        ax2.set_ylabel("rate (bpm)", color="tab:blue")
        ax2.tick_params(axis="y", labelcolor="tab:blue")

    if state.start_idx is not None:
        ax.axvline(t_h[state.start_idx], color="tab:orange", ls=":", lw=1.0, alpha=0.6,
                   label="injection start", gid="start_ref")
    if state.shutin_idx is not None:
        ax.axvline(t_h[state.shutin_idx], color="tab:red", ls=":", lw=1.0, alpha=0.6,
                   label="shut-in", gid="shutin_ref")

    if interactive and has_trim_context:
        # Nothing to cut yet -> park at the last RAW sample, not resampled_full.dt[-1]. The
        # resampler only keeps a point per 30-psi drop, so on a slow falloff its last kept point
        # can sit well short of the record's end; parking there would show the line mid-plot with
        # ungrayed data to its right, implying a cut that isn't in effect. Releasing a drag at
        # that raw edge still clears (ui's commit clears at idx >= len(dt_full) - 1).
        trim_x_h = (cut_dt + res.t_shutin_s) / 3600.0 if cut_dt is not None else t_h[-1]
        ax.axvline(trim_x_h, color="tab:blue", ls="--", lw=1.4, gid="tail_trim")

    ax.set_title("Overview — entire dataset", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)

    # Pinned y-min 0 (CLAUDE.md TODO): the pressure trace never reads below 0 psi, and a
    # gauge/BHP-conversion floor should always be visible relative to true zero, even though
    # that squashes a converted-BHP trace (~4800-6200 psi) into the top of the axes -- the
    # y-slider and Reset view are the escape. p_hi/pad use the whole record (kept + grayed),
    # same data-span idiom render_gfunction uses for its own pressure ylim.
    finite_p = np.isfinite(p)
    if not finite_p.any():
        return ViewDefaults()
    p_lo, p_hi = float(np.nanmin(p[finite_p])), float(np.nanmax(p[finite_p]))
    pad = 0.05 * max(p_hi - p_lo, 1.0)
    y2_color = "tab:blue" if res.rate_all is not None else None
    return ViewDefaults(ylim=(0.0, p_hi + pad), y_color=press_color, y2_color=y2_color)


def render_injection(ax, td: TestData, state: PickState, res: DerivedResults) -> ViewDefaults:
    """Step 2: BHP (or surface P) and rate vs time, with injection-start / shut-in markers.

    The falloff tail can run for weeks and would otherwise dwarf the active-injection region in
    both the autoscaled extent and the x-slider's full range, so -- following ``render_isip``'s
    precedent of clamping the *plotted data* -- every trace is masked to the last nonzero rate +
    15 min before decimation when a rate channel exists and pumped at all; otherwise the full
    record is plotted, unclamped.
    """
    ax.clear()
    p = res.bhp_all if res.bhp_all is not None else np.full(td.n, np.nan)
    t_h = _hours(td.t_s)

    t_end_h = None
    if res.rate_all is not None and np.any(res.rate_all > 0):
        last_active = int(np.where(res.rate_all > 0)[0][-1])
        t_end_h = t_h[last_active] + 0.25
    m = (t_h <= t_end_h) if t_end_h is not None else np.ones_like(t_h, dtype=bool)

    xt, xp = _decimate(t_h[m], p[m])
    press_color = "black" if res.pressure_is_bhp else "tab:red"
    ax.plot(xt, xp, color=press_color, lw=0.8,
            label="bottomhole pressure" if res.pressure_is_bhp else "pressure")
    ax.set_xlabel("time from file start (h)")
    ax.set_ylabel("BHP (psi)" if res.pressure_is_bhp else "pressure (psi)", color=press_color)
    ax.tick_params(axis="y", labelcolor=press_color)
    ax.grid(True, alpha=0.3)

    if res.rate_all is not None:
        ax2 = ax.twinx()
        _, xr = _decimate(t_h[m], res.rate_all[m])
        ax2.plot(xt, xr, color="tab:blue", lw=0.7, alpha=0.7)
        ax2.set_ylabel("rate (bpm)", color="tab:blue")
        ax2.tick_params(axis="y", labelcolor="tab:blue")

    if state.start_idx is not None:
        ax.axvline(t_h[state.start_idx], color="tab:orange", ls="--", lw=1.6,
                   label="injection start", gid="start")
    if state.shutin_idx is not None:
        ax.axvline(t_h[state.shutin_idx], color="tab:red", ls="-", lw=1.8,
                   label="shut-in", gid="shutin")

    # The default view zooms to the active injection region (the falloff tail can be weeks long);
    # the full autoscaled extent stays available for the caller to zoom back out to.
    xlim = None
    if state.start_idx is not None and state.shutin_idx is not None:
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
    y2_color = "tab:blue" if res.rate_all is not None else None
    return ViewDefaults(xlim=xlim, y_color=press_color, y2_color=y2_color)


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
        ax.set_title("Apparent ISIP -- set injection window first", fontsize=10)
        return ViewDefaults()
    t_min = (td.t_s - res.t_shutin_s) / 60.0
    m = (t_min >= -5.0) & (t_min <= 15.0)
    p_clean, p_masked = _split_dropouts(res.bhp_all, res.dropout_mask)
    xt, xp = _decimate(t_min[m], p_clean[m])
    ax.plot(xt, xp, color="black", lw=0.9)
    masked_idx = np.flatnonzero(m & np.isfinite(p_masked))
    _plot_dropout_markers(ax, t_min[masked_idx], p_masked[masked_idx])
    ax.axvline(0.0, color="tab:red", lw=1.2, label="shut-in")
    ax.set_xlabel("time from shut-in (min)")
    ax.set_ylabel("BHP (psi)")
    ax.grid(True, alpha=0.3)

    tg = state.isip_tangent
    if tg is not None:
        # tg lives on the seconds-since-file-start / psi-per-second convention td.t_s uses; this
        # axes plots minutes-from-shut-in, so convert before drawing -- ui.py's controller wiring
        # converts the same way (see _isip_pick_in_minutes/_isip_minutes_to_seconds).
        anchor_x_min = (tg.anchor_x - res.t_shutin_s) / 60.0
        slope_per_min = tg.slope * 60.0
        y_span = float(np.nanmax(xp) - np.nanmin(xp)) if xp.size else max(abs(tg.anchor_y), 1.0)
        _draw_tangent_construction(
            ax, anchor_x_min, tg.anchor_y, slope_per_min, ref_x=0.0, half=_ISIP_TANGENT_HALF_MIN,
            color="tab:purple",
            gids={"segment": "isip_tangent_segment", "tick": "isip_tangent_tick",
                  "extension": "isip_tangent_extension"},
            tick_half_y=0.04 * y_span, label="ISIP tangent")
        ax.plot(0.0, res.apparent_isip, "o", color="tab:purple", gid="isip_value_dot")
    if res.apparent_isip is not None:
        ax.set_title(f"Apparent ISIP = {res.apparent_isip:.0f} psi", fontsize=10)
    else:
        ax.set_title("Apparent ISIP -- place the tangent", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    return ViewDefaults(xlim=(-1.0, 3.0), y_color="black")


def render_gfunction(ax, td: TestData, state: PickState, res: DerivedResults) -> ViewDefaults:
    """Step 4: P and dP/dG vs G-time; contact + min-dP/dG markers; effective-ISIP line to G=0."""
    ax.clear()
    if res.diagnostics is None:
        ax.set_title("G-function -- need te and a falloff", fontsize=10)
        # Recovery path: a pathological saved tail trim can leave <3 resampled points (the
        # diagnostics guard in compute_all fails), so there is nothing left to diagnose here.
        # The trim tool itself now lives on the Overview tab -- dragging it back right there is
        # the way out of this state.
        # A guard fire severe enough to leave <3 kept points lands here too (no trim needed) --
        # draw its excluded-tail preview the same as the main path, so it isn't silent just
        # because there were too few points left to diagnose.
        if res.guard_excluded_G is not None and len(res.guard_excluded_G):
            ax.plot(res.guard_excluded_G, res.guard_excluded_p, color="0.85", alpha=0.5, lw=0.8,
                    gid="guard_excluded", zorder=0.5)
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
        ax.plot(res.guard_excluded_G, res.guard_excluded_p, color="0.85", alpha=0.5, lw=0.8,
                gid="guard_excluded", zorder=0.5)
    ax.plot(dg.G, rs.p, color="black", lw=1.2, marker=".", ms=3, label="BHP")
    ax.set_xlabel("G-time")
    ax.set_ylabel("BHP (psi)")
    ax.grid(True, alpha=0.3)

    # The pressure axis must scale from the BHP data only -- the effective-ISIP tangent's dashed
    # extension (drawn below, on this same Axes) can swing to extreme psi values far outside the
    # real data, and the Axes' own autoscale would otherwise pick that up too.
    finite_p = np.isfinite(rs.p)
    ylim = None
    if finite_p.any():
        p_lo, p_hi = float(np.nanmin(rs.p[finite_p])), float(np.nanmax(rs.p[finite_p]))
        pad = 0.05 * max(p_hi - p_lo, 1.0)
        ylim = (p_lo - pad, p_hi + pad)

    ax2 = ax.twinx()
    ax2.plot(dg.G, dg.dPdG, color="tab:red", lw=1.0, label="dP/dG")
    ax2.set_ylabel("dP/dG", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    y2lim = None
    # Autoscale the default view to the real derivative, masking the early water-hammer spike
    # out by G-time (same g_min convention as interpret.suggest_min_dpdg_index and the d2P/dG2
    # block below). A percentile over all samples was dominated by the spike (the resampled grid
    # is densest there), and the old hard 50 cap squashed any record whose real dP/dG ran higher.
    finite = np.isfinite(dg.dPdG) & (dg.G >= Y2_SCALE_G_MIN)
    if not finite.any():        # whole record sits below G=1 -- scale from everything finite
        finite = np.isfinite(dg.dPdG)
    if finite.any():
        hi = float(np.nanmax(dg.dPdG[finite]))
        y2lim = (0, min(max(hi * 1.10, 1.0), DPDG_VIEW_MAX))

    y3lim = None
    if state.show_d2pdg2:
        # A third y-axis, offset further right so it doesn't collide with the dP/dG twin's
        # ticks/label -- see D2_AXIS_GID (ui.py excludes it from the twin lookup/y2 slider, and
        # gives it no slider/persisted view of its own, decision D3).
        ax3 = ax.twinx()
        ax3.set_gid(D2_AXIS_GID)
        ax3.spines["right"].set_position(("axes", 1.12))
        ax3.plot(dg.G, dg.d2PdG2, color="tab:purple", lw=0.9, label="d2P/dG2",
                 gid="d2pdg2_curve")
        ax3.set_ylabel("d2P/dG2", color="tab:purple")
        ax3.tick_params(axis="y", labelcolor="tab:purple")
        # Scale from G >= 1 only (same g_min convention as interpret.suggest_min_dpdg_index):
        # the resampled grid is densest across the early water-hammer spike, so percentiles
        # over all samples would still be dominated by its huge |d2| values.
        finite_d2 = np.isfinite(dg.d2PdG2) & (dg.G >= 1.0)
        if not finite_d2.any():
            finite_d2 = np.isfinite(dg.d2PdG2)
        if finite_d2.any():
            lo, hi = np.percentile(dg.d2PdG2[finite_d2], [5, 95])
            pad = 0.10 * max(hi - lo, 1e-9)
            y3lim = (lo - pad, hi + pad)

    if res.eff_isip_line_compliance is not None and res.effective_isip_compliance is not None:
        ln = res.eff_isip_line_compliance
        g_span = float(np.nanmax(dg.G) - np.nanmin(dg.G)) if len(dg.G) else 1.0
        y_span = (float(np.nanmax(rs.p) - np.nanmin(rs.p)) if len(rs.p)
                 else max(abs(ln.anchor_y), 1.0))
        _draw_tangent_construction(
            ax, ln.anchor_x, ln.anchor_y, ln.slope, ref_x=0.0, half=max(0.06 * g_span, 1e-6),
            color="tab:green",
            gids={"segment": "eff_isip_segment", "tick": "eff_isip_tick",
                  "extension": "eff_isip_extension"},
            tick_half_y=0.04 * y_span, label="effective-ISIP line", draw_tick=False)
        ax.plot(0.0, res.effective_isip_compliance, "o", color="tab:green")
    # The triangle is only meaningful for C-A (rel-min anchor) / C-B (inflection seed) -- C-C/C-D
    # have no contact rule and blank leaves it hidden until a scenario is chosen (decision 4).
    if state.min_dpdg_G is not None and state.closure_scenario.startswith(("C-A", "C-B")):
        y = float(np.interp(state.min_dpdg_G, dg.G, dg.dPdG))
        ax2.plot(state.min_dpdg_G, y, marker="v", color="tab:red", ms=8, label="min dP/dG",
                gid="min_dpdg_point")
    # C-A only: if dP/dG never rises 10% above the picked min, suggest_contact_clear_index finds
    # no contact -- draw the 110% threshold it's checking against so the analyst can see the
    # curve never reaches it (and knows to try C-B instead).
    if state.closure_scenario.startswith("C-A") and state.min_dpdg_G is not None:
        min_idx = int(np.argmin(np.abs(dg.G - state.min_dpdg_G)))
        threshold = 1.10 * dg.dPdG[min_idx]
        # An all-NaN dPdG at the pick makes threshold itself NaN -- suggest_contact_clear_index
        # already returns None for it (never rises), but drawing an axhline at NaN leaves an
        # invisible line with a visible "never reached" label, so gate on finiteness too.
        if np.isfinite(threshold) and interpret.suggest_contact_clear_index(dg.dPdG, min_idx) is None:
            ax2.axhline(threshold, ls="--", color="tab:red", alpha=0.6, gid="clear_threshold_line")
            ax2.text(0.02, 0.02, "min +10% -- never reached; consider C-B",
                     transform=ax2.transAxes, fontsize=7, color="tab:red", va="bottom", ha="left")
    if state.contact_G is not None and res.contact_pressure is not None:
        ax.plot(state.contact_G, res.contact_pressure, "s", color="black", ms=7, label="contact",
               gid="contact_point")
    if state.contact_G is not None:
        ax.axvline(state.contact_G, color="black", ls=":", lw=1.2, gid="contact_vline")

    title = "G-function"
    if res.effective_isip_compliance is not None:
        title += f"   eff.ISIP={res.effective_isip_compliance:.0f}"
    if res.shmin_compliance is not None:
        title += f"   Shmin(compl)={res.shmin_compliance:.0f}"
    if res.shmin_rapid is not None:
        title += f"   Shmin(rapid)={interpret.format_shmin_rapid(res.shmin_rapid, verbose=True)}"
    title += f"   ({state.closure_scenario or '?'})"
    ax.set_title(title, fontsize=10)
    ax.legend(loc="lower left", fontsize=8)
    return ViewDefaults(ylim=ylim, y2lim=y2lim, y3lim=y3lim, y_color="black", y2_color="tab:red")


def render_tangent(ax, td: TestData, state: PickState, res: DerivedResults) -> ViewDefaults:
    """Step 5: BHP and G*dP/dG vs G-time -- mirrors ``render_gfunction``'s twinx layout (BHP on
    the primary/left axis, G*dP/dG on the twin/right). The through-origin line is still picked
    on the G*dP/dG curve and lives on the twin axis, but the closure marker now rides the BHP
    curve on the primary axis."""
    ax.clear()
    if res.diagnostics is None:
        ax.set_title("Tangent method -- need a falloff", fontsize=10)
        return ViewDefaults()
    dg = res.diagnostics
    rs = res.resampled
    ax.plot(dg.G, rs.p, color="black", lw=1.2, marker=".", ms=3, label="BHP")
    ax.set_xlabel("G-time")
    ax.set_ylabel("BHP (psi)")
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(dg.G, dg.GdPdG, color="tab:red", lw=1.0, marker=".", ms=3, label="G*dP/dG")
    ax2.set_ylabel("G*dP/dG", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    y2lim = None
    finite = np.isfinite(dg.GdPdG)
    if finite.any():  # clip early water-hammer spike off-scale (in the default view only)
        hi = np.percentile(dg.GdPdG[finite], 95)
        y2lim = (0, max(hi * 1.5, 1.0))

    if state.tangent_uninterpretable:
        # Explicit negative finding, same precedent as render_stiffness's stiffness_no_upturn:
        # the curves stay as evidence, but no line/marker, and shmin_tangent is None.
        title = "Tangent method -- uninterpretable (Shmin not reported)"
    else:
        if state.closure_slope is not None:
            gg = np.array([0.0, float(dg.G.max())])
            ax2.plot(gg, state.closure_slope * gg, color="tab:gray", ls="--", lw=1.2,
                    label="through-origin", gid="closure_line_segment")
        if state.closure_G is not None:
            yv = float(np.interp(state.closure_G, dg.G, rs.p))
            ax.plot(state.closure_G, yv, "o", color="black", ms=7, label="closure",
                    gid="closure_point")
            ax.axvline(state.closure_G, color="black", ls=":", lw=1.2, gid="closure_vline")
        title = "Tangent method"
        if res.shmin_tangent is not None:
            title += f"   Shmin(tangent)={res.shmin_tangent:.0f}"
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    return ViewDefaults(y2lim=y2lim, y_color="black", y2_color="tab:red")


def render_loglog(ax, td: TestData, state: PickState, res: DerivedResults) -> ViewDefaults:
    """Step 6: log-log dp and t*dP/dt vs shut-in time; selected window + fitted slope."""
    ax.clear()
    if res.diagnostics is None:
        ax.set_title("Log-log -- need a falloff", fontsize=10)
        return ViewDefaults()
    dg = res.diagnostics
    good = (dg.t > 0) & (dg.dp > 0)
    ax.loglog(dg.t[good], dg.dp[good], color="tab:blue", lw=1.0, marker=".", ms=3, label="dp")
    tgood = (dg.t > 0) & (dg.tdpdt > 0)
    ax.loglog(dg.t[tgood], dg.tdpdt[tgood], color="tab:red", lw=1.0, marker=".", ms=3,
              label="t*dP/dt")
    ax.set_xlabel("shut-in time (s)")
    ax.set_ylabel("dp, t*dP/dt (psi)")
    ax.grid(True, which="both", alpha=0.3)

    if state.loglog_window is not None:
        lo, hi = state.loglog_window
        ax.axvspan(lo, hi, color="tab:orange", alpha=0.15)
        from .interpret import loglog_slope
        i0 = int(np.searchsorted(dg.t, lo))
        i1 = int(np.searchsorted(dg.t, hi))
        s = loglog_slope(dg.t, dg.tdpdt, i0, i1)
        ax.set_title(f"Log-log   window slope={s:.2f}   ({state.postclosure_scenario or '?'})",
                     fontsize=10)
    else:
        ax.set_title("Log-log -- select the late-time window", fontsize=10)
    ax.legend(loc="upper left", fontsize=8)
    return ViewDefaults()


def render_porepressure(ax, td: TestData, state: PickState, res: DerivedResults) -> ViewDefaults:
    """Step 7: P vs t^-1/2 or t^-1 with the fitted line extended to the intercept."""
    ax.clear()
    if res.diagnostics is None:
        ax.set_title("Pore pressure -- need a falloff", fontsize=10)
        return ViewDefaults()
    dg = res.diagnostics
    expo = -0.5 if state.pp_axis == "tm12" else -1.0
    x = dg.t ** expo
    ax.plot(x, dg.p, color="black", lw=1.0, marker=".", ms=3)
    ax.set_xlabel("t^(-1/2)" if state.pp_axis == "tm12" else "t^(-1)")
    ax.set_ylabel("BHP (psi)")
    ax.grid(True, alpha=0.3)
    xmax = float(np.nanmax(x)) if x.size else 1.0

    if state.pp_window is not None:
        lo, hi = state.pp_window
        x_lo = 0.0 if not np.isfinite(hi) else hi ** expo
        x_hi = lo ** expo if lo > 0 else xmax
        ax.axvspan(x_lo, x_hi, color="tab:orange", alpha=0.15)

    if state.pp_window is not None and res.pore_pressure is not None:
        lo, hi = state.pp_window
        m = (dg.t >= lo) & (dg.t <= hi)
        if m.sum() >= 2:
            from .interpret import fit_line
            slope, intercept = fit_line(x[m], dg.p[m])
            xr = np.array([0.0, x[m].max()])
            ax.plot(xr, intercept + slope * xr, color="tab:green", ls="--", lw=1.3)
            ax.plot(0.0, res.pore_pressure, "o", color="tab:green")
            pmin = float(dg.p[m].min())
            if res.pore_pressure >= pmin:
                ax.set_title(
                    f"Pore pressure = {res.pore_pressure:.0f} psi  (>= observed -- adjust window)",
                    fontsize=10)
            else:
                ax.set_title(f"Pore pressure = {res.pore_pressure:.0f} psi", fontsize=10)
        else:
            ax.set_title(f"Pore pressure = {res.pore_pressure:.0f} psi", fontsize=10)
    else:
        ax.set_title("Pore pressure -- select the late-time window", fontsize=10)
    xhi = 0.05 if state.pp_axis == "tm12" else 0.0025
    return ViewDefaults(xlim=(0.0, xhi), y_color="black")


def render_stiffness(ax, td: TestData, state: PickState, res: DerivedResults) -> ViewDefaults:
    """Step 8: relative system stiffness (semilog-y) vs effective pressure (URTeC-2019-123
    A.8/A.9) -- the upturn where the fracture walls come into contact gives a fourth,
    comparison-only Shmin estimate. Needs the min-dP/dG pick and a pore-pressure estimate (the
    h-function's Pres term); skipped end to end under PC-F (model.stiffness_skipped), which
    never yields one. state.stiffness_no_upturn records the negative finding "no slope change
    apparent" -- the curve still draws, but no pick vline/marker and a title saying so."""
    ax.clear()
    if res.stiffness_S is None:
        ax.set_title("Stiffness -- requires the min-dP/dG pick and a pore-pressure estimate",
                     fontsize=10)
        return ViewDefaults()
    p_eff, S = res.stiffness_p_eff[1:], res.stiffness_S
    finite_pos = np.isfinite(S) & (S > 0)
    if not finite_pos.any():
        # A log-scaled axis needs at least one positive sample to have anything to draw --
        # set_yscale("log") ahead of this check drew an axis-only plot plus a matplotlib
        # UserWarning ("Data has no positive values..."). Guard branch instead, same style as
        # the missing-arrays branch above.
        title = "Stiffness -- no positive relative-stiffness samples to plot"
        if closure_uninterpretable(state):
            title += " -- G-function uninterpretable (Shmin not reported)"
        elif state.stiffness_no_upturn:
            # The recorded finding still belongs in the title even though there is nothing to
            # plot -- otherwise this branch silently drops it and looks like a data problem
            # rather than the analyst's own "no slope change apparent" call.
            title += " -- no slope change apparent (Shmin not reported)"
        ax.set_title(title, fontsize=10)
        return ViewDefaults()
    ax.plot(p_eff, S, color="black", lw=1.0, marker=".", ms=3)
    ax.set_yscale("log")
    ax.set_xlabel("effective pressure (psi)")
    ax.set_ylabel("relative stiffness")
    ax.grid(True, which="both", alpha=0.3)

    if closure_uninterpretable(state):
        # C-X: the min-dP/dG anchor of this curve can't be trusted, so shmin_stiffness is None
        # (model.compute_all). Same no-vline/no-marker treatment as stiffness_no_upturn.
        ax.set_title("Relative stiffness -- G-function uninterpretable (Shmin not reported)",
                     fontsize=10)
    elif state.stiffness_no_upturn:
        # Explicit negative finding, same precedent as closure scenario C-C's "no contact ->
        # no Shmin": the curve is still evidence (kept on the plot and in PNG exports), but
        # there is no upturn to mark, so no vline/marker -- and shmin_stiffness is None
        # (model.compute_all), so no title should imply otherwise.
        ax.set_title("Relative stiffness -- no slope change apparent (Shmin not reported)",
                     fontsize=10)
    elif state.stiffness_pick_P is not None:
        ax.axvline(state.stiffness_pick_P, color="tab:blue", ls="--", lw=1.4,
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
            ax.plot(state.stiffness_pick_P, s_at_pick, "o", color="tab:blue", ms=5)
        if res.shmin_stiffness is not None:
            ax.set_title(f"Stiffness   Shmin(stiffness)={res.shmin_stiffness:.0f} psi",
                         fontsize=10)
        else:
            ax.set_title("Stiffness", fontsize=10)
    else:
        ax.set_title("Stiffness -- pick the upturn", fontsize=10)

    finite_p = p_eff[np.isfinite(p_eff)]
    xlim = None
    if finite_p.size:
        p_lo, p_hi = float(np.nanmin(finite_p)), float(np.nanmax(finite_p))
        pad = 0.05 * max(p_hi - p_lo, 1.0)
        xlim = (p_lo - pad, p_hi + pad)
    # finite_pos is already known non-empty -- the all-non-positive case returned above.
    y_lo, y_hi = float(np.nanmin(S[finite_pos])), float(np.nanmax(S[finite_pos]))
    ylim = (y_lo * 0.5, y_hi * 2.0)  # log-safe floor/pad
    return ViewDefaults(xlim=xlim, ylim=ylim, y_color="black")


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


def render_step_figure(step_key: str, td: TestData, state: PickState, res: DerivedResults,
                       stored_view: Optional[tuple] = None,
                       figsize: tuple[float, float] = (9, 6)) -> Figure:
    """Render one step onto an offscreen ``Figure`` with the same view-resolution logic
    ``ui.refresh()``/``ui._resolve_view`` apply to the live canvas, so an exported PNG matches
    what the analyst was looking at (stored_view) or the renderer's own default.

    No Tkinter -- this and ``save_all_step_pngs`` are called by ``ui._finish`` but could equally
    run headlessly for tests, per the module-level invariant.
    """
    fig = Figure(figsize=figsize)
    ax = fig.add_subplot(111)
    # Overview's tail-trim line is a live-canvas control, not part of the interpretation --
    # render_step_figure (the only caller of save_all_step_pngs) always opts out explicitly,
    # mirroring the step_key == "gfunction" special cases just below, so an exported PNG never
    # carries a line the analyst can't actually drag.
    kwargs = {"interactive": False} if step_key == "overview" else {}
    defaults = RENDERERS[step_key](ax, td, state, res, **kwargs)

    full_x = ax.get_xlim()
    full_y = ax.get_ylim()
    if defaults.ylim is not None:
        # There is no slider on this offscreen Figure -- full_y here only feeds this function's
        # own stored_view-is-None fallback a few lines down, nothing in ui.py. This union/replace
        # split is kept in textual lockstep with the near-identical block in ui.refresh anyway
        # (deliberately, so the two view-resolution paths can't silently drift apart), even
        # though the REPLACE-vs-slider reasoning that motivates it there (shielding the y-slider
        # from the effective-ISIP tangent's dashed extension) doesn't apply here, and the
        # gfunction branch specifically is moot for "overview" (the case this split was added
        # for). Every non-gfunction step still UNIONS so a concrete ViewDefaults.ylim that
        # reaches outside the autoscaled extent (e.g. Overview's pinned y-min 0 against a
        # converted-BHP trace) survives into the fallback instead of being silently dropped.
        full_y = (defaults.ylim if step_key == "gfunction"
                 else (min(full_y[0], defaults.ylim[0]), max(full_y[1], defaults.ylim[1])))
    # Exclude the d2P/dG2 axis (D2_AXIS_GID) from the twin lookup -- it gets no slider/persisted
    # view of its own (decision D3) and must never be mistaken for the dP/dG twin here.
    twin = next((a for a in fig.axes if a is not ax and a.get_gid() != D2_AXIS_GID), None)
    full_y2 = twin.get_ylim() if twin is not None else None
    if step_key == "gfunction" and full_y2 is not None:
        # UNION the renderer's own y2 default in first (kept in textual lockstep with the
        # near-identical block in ui.refresh), then hard-clamp to 0-500 -- see that block for
        # the full reasoning.
        if defaults.y2lim is not None:
            full_y2 = (min(full_y2[0], defaults.y2lim[0]), max(full_y2[1], defaults.y2lim[1]))
        full_y2 = (max(full_y2[0], 0.0), min(full_y2[1], DPDG_VIEW_MAX))

    if stored_view is not None:
        xlim, ylim, y2lim = stored_view
    else:
        xlim = defaults.xlim if defaults.xlim is not None else full_x
        ylim = defaults.ylim if defaults.ylim is not None else full_y
        y2lim = defaults.y2lim if defaults.y2lim is not None else full_y2

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    if twin is not None and y2lim is not None:
        twin.set_ylim(y2lim)

    # The d2 axis is never part of stored_view (D3) -- always apply the renderer's fresh default.
    d2_axis = next((a for a in fig.axes if a.get_gid() == D2_AXIS_GID), None)
    d2_on = d2_axis is not None
    if d2_axis is not None and defaults.y3lim is not None:
        d2_axis.set_ylim(defaults.y3lim)

    right = 0.80 if (step_key == "gfunction" and d2_on) else 0.90
    fig.subplots_adjust(left=0.10, right=right, bottom=0.16, top=0.90)
    return fig


def save_all_step_pngs(out_dir: str, td: TestData, state: PickState, res: DerivedResults,
                       views: dict[str, Optional[tuple]], dpi: int = 150) -> list[str]:
    """Render every step's current view to a numbered PNG in ``out_dir`` (RENDERERS' insertion
    order: overview -> injection -> isip -> gfunction -> tangent -> loglog -> porepressure ->
    stiffness). Returns the written paths in that order. Two independent skips, each PC-F-gated
    (porepressure_skipped/stiffness_skipped): "porepressure" (no postclosure line, nothing to
    render) and "stiffness" (no pore-pressure estimate for the h-function) -- the numbering from
    ``enumerate`` still runs over all of RENDERERS so the other filenames are unaffected; the
    skipped ones are simply absent. Used by ``ui._finish``, but headless/Tkinter-free like the
    rest of this module."""
    paths = []
    skip_pp = porepressure_skipped(state)
    skip_stiff = stiffness_skipped(state)
    for i, key in enumerate(RENDERERS, start=1):
        if (key == "porepressure" and skip_pp) or (key == "stiffness" and skip_stiff):
            continue
        fig = render_step_figure(key, td, state, res, views.get(key))
        path = os.path.join(out_dir, f"{i}_{key}.png")
        fig.savefig(path, dpi=dpi)
        paths.append(path)
    return paths
