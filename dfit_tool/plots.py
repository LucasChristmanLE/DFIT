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

from .model import DerivedResults, PickState
from .io_load import TestData

_MAX_POINTS = 6000  # display decimation cap for the raw (dense) traces


@dataclass
class ViewDefaults:
    """The view a renderer suggests for first-visit display; ``None`` means autoscaled full
    extent. Callers apply these to the Axes -- renderers never set limits themselves."""
    xlim: Optional[tuple[float, float]] = None
    ylim: Optional[tuple[float, float]] = None
    y2lim: Optional[tuple[float, float]] = None


def _decimate(x: np.ndarray, *ys: np.ndarray):
    """Uniformly thin long arrays for display without distorting shape."""
    n = len(x)
    if n <= _MAX_POINTS:
        return (x, *ys)
    step = int(np.ceil(n / _MAX_POINTS))
    return (x[::step], *(y[::step] for y in ys))


def _hours(t_s: np.ndarray, t0: float = 0.0) -> np.ndarray:
    return (np.asarray(t_s, dtype=float) - t0) / 3600.0


def _draw_tangent_construction(ax, anchor_x: float, anchor_y: float, slope: float, *,
                               ref_x: float, half: float, color: str, gids: dict,
                               tick_half_y: float, label: Optional[str] = None,
                               lw: float = 1.6) -> None:
    """Draw one gid-tagged tangent construction (see the workflow steps in ../CLAUDE.md): a finite ``segment`` through
    the anchor, a short vertical ``tick`` at the anchor, and a dashed ``extension`` running from
    the segment's near end back to the reference vertical ``ref_x`` (the shut-in line for the
    apparent-ISIP construction, G=0 for the effective-ISIP construction) -- the ISIP marker sits
    where the extension crosses ``ref_x``. ``gids`` maps "segment"/"tick"/"extension" to the exact
    gid string each piece is drawn with, matched by ``picks.AnchorLineController``.
    """
    x0, x1 = anchor_x - half, anchor_x + half
    xs = np.array([x0, x1])
    ys = anchor_y + slope * (xs - anchor_x)
    ax.plot(xs, ys, color=color, lw=lw, label=label, gid=gids["segment"])
    ax.plot([anchor_x, anchor_x], [anchor_y - tick_half_y, anchor_y + tick_half_y],
            color=color, lw=lw, gid=gids["tick"])
    near_x = x0 if abs(x0 - ref_x) <= abs(x1 - ref_x) else x1
    ext_x = np.array([ref_x, near_x])
    ext_y = anchor_y + slope * (ext_x - anchor_x)
    ax.plot(ext_x, ext_y, color=color, lw=max(lw - 0.3, 1.0), ls="--", gid=gids["extension"])


# --------------------------------------------------------------------------------------------------
def render_overview(ax, td: TestData, state: PickState, res: DerivedResults) -> ViewDefaults:
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
    ax.set_ylabel("pressure (psi)", color=press_color)
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

    title = "Overview"
    if res.te_s:
        title += f"   te={res.te_s/60:.2f} min   Vinj={res.vinj:.1f} bbl   qmax={res.qmax_bpm:.2f} bpm"
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    return ViewDefaults(xlim=xlim)


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
    xt, xp = _decimate(t_min[m], res.bhp_all[m])
    ax.plot(xt, xp, color="black", lw=0.9)
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
            ax, anchor_x_min, tg.anchor_y, slope_per_min, ref_x=0.0, half=3.0,
            color="tab:purple",
            gids={"segment": "isip_tangent_segment", "tick": "isip_tangent_tick",
                  "extension": "isip_tangent_extension"},
            tick_half_y=0.04 * y_span, label="ISIP tangent")
        ax.plot(0.0, res.apparent_isip, "o", color="tab:purple")
    if res.apparent_isip is not None:
        ax.set_title(f"Apparent ISIP = {res.apparent_isip:.0f} psi", fontsize=10)
    else:
        ax.set_title("Apparent ISIP -- place the tangent", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    return ViewDefaults(xlim=(-1.0, 3.0))


def render_gfunction(ax, td: TestData, state: PickState, res: DerivedResults) -> ViewDefaults:
    """Step 5: P and dP/dG vs G-time; contact + min-dP/dG markers; effective-ISIP line to G=0."""
    ax.clear()
    if res.diagnostics is None:
        ax.set_title("G-function -- need te and a falloff", fontsize=10)
        return ViewDefaults()
    dg = res.diagnostics
    rs = res.resampled
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
    finite = np.isfinite(dg.dPdG)
    if finite.any():  # clip early water-hammer spike off-scale (in the default view only)
        hi = np.percentile(dg.dPdG[finite], 95)
        y2lim = (0, min(max(hi * 1.5, 1.0), 50.0))
    if state.show_d2pdg2:
        ax2.plot(dg.G, dg.d2PdG2, color="tab:purple", lw=0.9, label="d2P/dG2",
                 gid="d2pdg2_curve")

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
            tick_half_y=0.04 * y_span, label="effective-ISIP line")
        ax.plot(0.0, res.effective_isip_compliance, "o", color="tab:green")
    if state.min_dpdg_G is not None:
        y = float(np.interp(state.min_dpdg_G, dg.G, dg.dPdG))
        ax2.plot(state.min_dpdg_G, y, marker="v", color="tab:red", ms=8, label="min dP/dG",
                gid="min_dpdg_point")
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
    ax.set_title(title, fontsize=10)
    ax.legend(loc="lower left", fontsize=8)
    return ViewDefaults(ylim=ylim, y2lim=y2lim)


def render_tangent(ax, td: TestData, state: PickState, res: DerivedResults) -> ViewDefaults:
    """Step 6: BHP and G*dP/dG vs G-time -- mirrors ``render_gfunction``'s twinx layout (BHP on
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
    return ViewDefaults(y2lim=y2lim)


def render_loglog(ax, td: TestData, state: PickState, res: DerivedResults) -> ViewDefaults:
    """Step 7: log-log dp and t*dP/dt vs shut-in time; selected window + fitted slope."""
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
    """Step 8: P vs t^-1/2 or t^-1 with the fitted line extended to the intercept."""
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
    return ViewDefaults(xlim=(0.0, xhi))


RENDERERS = {
    "overview": render_overview,
    "isip": render_isip,
    "gfunction": render_gfunction,
    "tangent": render_tangent,
    "loglog": render_loglog,
    "porepressure": render_porepressure,
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
    defaults = RENDERERS[step_key](ax, td, state, res)

    full_x = ax.get_xlim()
    full_y = ax.get_ylim()
    if step_key == "gfunction" and defaults.ylim is not None:
        full_y = defaults.ylim
    twin = next((a for a in fig.axes if a is not ax), None)
    full_y2 = twin.get_ylim() if twin is not None else None
    if step_key == "gfunction" and full_y2 is not None:
        full_y2 = (max(full_y2[0], 0.0), min(full_y2[1], 500.0))

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

    fig.subplots_adjust(left=0.10, right=0.90, bottom=0.16, top=0.90)
    return fig


def save_all_step_pngs(out_dir: str, td: TestData, state: PickState, res: DerivedResults,
                       views: dict[str, Optional[tuple]], dpi: int = 150) -> list[str]:
    """Render every step's current view to a numbered PNG in ``out_dir`` (RENDERERS' insertion
    order: overview -> isip -> gfunction -> tangent -> loglog -> porepressure). Returns the
    written paths in that order. Used by ``ui._finish``, but headless/Tkinter-free like the
    rest of this module."""
    paths = []
    for i, key in enumerate(RENDERERS, start=1):
        fig = render_step_figure(key, td, state, res, views.get(key))
        path = os.path.join(out_dir, f"{i}_{key}.png")
        fig.savefig(path, dpi=dpi)
        paths.append(path)
    return paths
