"""Static rendering of each workflow step onto a matplotlib Axes/Figure.

Each ``render_*`` takes an Axes, the loaded ``TestData``, the ``PickState``, and the
``DerivedResults`` and draws the plot plus whatever picks currently exist. The interactive layer
(picks.py) updates the PickState and calls the matching render to refresh.

Matplotlib only -- no Tkinter -- so figures can be produced headlessly (Agg) for verification.
"""

from __future__ import annotations

import numpy as np

from .model import DerivedResults, PickState
from .io_load import TestData

_MAX_POINTS = 6000  # display decimation cap for the raw (dense) traces


def _decimate(x: np.ndarray, *ys: np.ndarray):
    """Uniformly thin long arrays for display without distorting shape."""
    n = len(x)
    if n <= _MAX_POINTS:
        return (x, *ys)
    step = int(np.ceil(n / _MAX_POINTS))
    return (x[::step], *(y[::step] for y in ys))


def _hours(t_s: np.ndarray, t0: float = 0.0) -> np.ndarray:
    return (np.asarray(t_s, dtype=float) - t0) / 3600.0


# --------------------------------------------------------------------------------------------------
def render_overview(ax, td: TestData, state: PickState, res: DerivedResults) -> None:
    """Step 2: BHP (or surface P) and rate vs time, with injection-start / shut-in markers."""
    ax.clear()
    p = res.bhp_all if res.bhp_all is not None else np.full(td.n, np.nan)
    t_h = _hours(td.t_s)
    xt, xp = _decimate(t_h, p)
    press_color = "black" if state.pressure_is_bhp else "tab:red"
    ax.plot(xt, xp, color=press_color, lw=0.8,
            label="BHP" if state.pressure_is_bhp else "pressure")
    ax.set_xlabel("time from file start (h)")
    ax.set_ylabel("pressure (psi)", color=press_color)
    ax.tick_params(axis="y", labelcolor=press_color)
    ax.grid(True, alpha=0.3)

    if res.rate_all is not None:
        ax2 = ax.twinx()
        _, xr = _decimate(t_h, res.rate_all)
        ax2.plot(xt, xr, color="tab:blue", lw=0.7, alpha=0.7)
        ax2.set_ylabel("rate (bpm)", color="tab:blue")
        ax2.tick_params(axis="y", labelcolor="tab:blue")

    if state.start_idx is not None:
        ax.axvline(t_h[state.start_idx], color="tab:orange", ls="--", lw=1.6,
                   label="injection start", gid="start")
    if state.shutin_idx is not None:
        ax.axvline(t_h[state.shutin_idx], color="tab:red", ls="-", lw=1.8,
                   label="shut-in", gid="shutin")

    # Auto-zoom to the active injection region (the falloff tail can be weeks long).
    if state.start_idx is not None and state.shutin_idx is not None:
        span_h = max(t_h[state.shutin_idx] - t_h[state.start_idx], 0.25)
        ax.set_xlim(t_h[state.start_idx] - 0.5 * span_h, t_h[state.shutin_idx] + 2.0 * span_h)
    elif res.rate_all is not None:
        act = np.where(res.rate_all > 0.1)[0]
        if act.size:
            ax.set_xlim(max(0, t_h[act[0]] - 0.2), t_h[act[-1]] + 0.5)

    title = "Overview"
    if res.te_s:
        title += f"   te={res.te_s/60:.2f} min   Vinj={res.vinj:.1f} bbl   qmax={res.qmax_bpm:.2f} bpm"
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper right", fontsize=8)


def render_isip(ax, td: TestData, state: PickState, res: DerivedResults, window_min: float = 30.0) -> None:
    """Step 3: BHP vs time zoomed after shut-in; the literal-ISIP tangent + extension to shut-in."""
    ax.clear()
    if res.bhp_all is None or res.t_shutin_s is None:
        ax.set_title("Literal ISIP -- set injection window first", fontsize=10)
        return
    t_min = (td.t_s - res.t_shutin_s) / 60.0
    m = (t_min >= -2.0) & (t_min <= window_min)
    xt, xp = _decimate(t_min[m], res.bhp_all[m])
    ax.plot(xt, xp, color="black", lw=0.9)
    ax.axvline(0.0, color="tab:red", lw=1.2, label="shut-in")
    ax.set_xlabel("time from shut-in (min)")
    ax.set_ylabel("BHP (psi)")
    ax.grid(True, alpha=0.3)

    tg = state.isip_tangent
    if tg is not None:
        # tangent lives on a seconds abscissa; draw over the zoom window
        x_s = np.array([res.t_shutin_s, tg.anchor_x + 8 * 60])
        y = tg.anchor_y + tg.slope * (x_s - tg.anchor_x)
        ax.plot((x_s - res.t_shutin_s) / 60.0, y, color="tab:purple", lw=1.4, label="ISIP tangent")
        ax.plot(0.0, res.literal_isip, "o", color="tab:purple")
    if res.literal_isip is not None:
        ax.set_title(f"Literal ISIP = {res.literal_isip:.0f} psi", fontsize=10)
    else:
        ax.set_title("Literal ISIP -- place the tangent", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)


def render_gfunction(ax, td: TestData, state: PickState, res: DerivedResults) -> None:
    """Step 5: P and dP/dG vs G-time; contact + min-dP/dG markers; effective-ISIP line to G=0."""
    ax.clear()
    if res.diagnostics is None:
        ax.set_title("G-function -- need te and a falloff", fontsize=10)
        return
    dg = res.diagnostics
    rs = res.resampled
    ax.plot(dg.G, rs.p, color="black", lw=1.2, marker=".", ms=3, label="BHP")
    ax.set_xlabel("G-time")
    ax.set_ylabel("BHP (psi)")
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(dg.G, dg.dPdG, color="tab:red", lw=1.0, label="dP/dG")
    ax2.set_ylabel("dP/dG", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")
    finite = np.isfinite(dg.dPdG)
    if finite.any():  # clip early water-hammer spike off-scale
        hi = np.percentile(dg.dPdG[finite], 95)
        ax2.set_ylim(0, max(hi * 1.5, 1.0))

    if state.eff_isip_line is not None and res.effective_isip is not None:
        ln = state.eff_isip_line
        gg = np.array([0.0, ln.anchor_x])
        yy = ln.anchor_y + ln.slope * (gg - ln.anchor_x)
        ax.plot(gg, yy, color="tab:green", lw=1.3, ls="--", label="effective-ISIP line")
        ax.plot(0.0, res.effective_isip, "o", color="tab:green")
    if state.contact_G is not None and res.contact_pressure is not None:
        ax.plot(state.contact_G, res.contact_pressure, "s", color="black", ms=7, label="contact")

    title = "G-function"
    if res.effective_isip is not None:
        title += f"   eff.ISIP={res.effective_isip:.0f}"
    if res.shmin_compliance is not None:
        title += f"   Shmin(compl)={res.shmin_compliance:.0f}"
    ax.set_title(title, fontsize=10)
    ax.legend(loc="lower left", fontsize=8)


def render_tangent(ax, td: TestData, state: PickState, res: DerivedResults) -> None:
    """Step 6: G*dP/dG vs G with the through-origin line and the closure departure point."""
    ax.clear()
    if res.diagnostics is None:
        ax.set_title("Tangent method -- need a falloff", fontsize=10)
        return
    dg = res.diagnostics
    ax.plot(dg.G, dg.GdPdG, color="tab:red", lw=1.0, marker=".", ms=3, label="G*dP/dG")
    ax.set_xlabel("G-time")
    ax.set_ylabel("G*dP/dG")
    ax.grid(True, alpha=0.3)

    if state.closure_slope is not None:
        gg = np.array([0.0, dg.G.max()])
        ax.plot(gg, state.closure_slope * gg, color="tab:gray", ls="--", lw=1.2,
                label="through-origin")
    if state.closure_G is not None:
        yv = float(np.interp(state.closure_G, dg.G, dg.GdPdG))
        ax.plot(state.closure_G, yv, "o", color="black", ms=7, label="closure")
    title = "Tangent method"
    if res.shmin_tangent is not None:
        title += f"   Shmin(tangent)={res.shmin_tangent:.0f}"
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper left", fontsize=8)


def render_loglog(ax, td: TestData, state: PickState, res: DerivedResults) -> None:
    """Step 7: log-log dp and t*dP/dt vs shut-in time; selected window + fitted slope."""
    ax.clear()
    if res.diagnostics is None:
        ax.set_title("Log-log -- need a falloff", fontsize=10)
        return
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
        s = loglog_slope(dg.t, dg.dp, i0, i1)
        ax.set_title(f"Log-log   window slope={s:.2f}   ({state.postclosure_scenario or '?'})",
                     fontsize=10)
    else:
        ax.set_title("Log-log -- select the late-time window", fontsize=10)
    ax.legend(loc="upper left", fontsize=8)


def render_porepressure(ax, td: TestData, state: PickState, res: DerivedResults) -> None:
    """Step 8: P vs t^-1/2 or t^-1 with the fitted line extended to the intercept."""
    ax.clear()
    if res.diagnostics is None:
        ax.set_title("Pore pressure -- need a falloff", fontsize=10)
        return
    dg = res.diagnostics
    expo = -0.5 if state.pp_axis == "tm12" else -1.0
    x = dg.t ** expo
    ax.plot(x, dg.p, color="black", lw=1.0, marker=".", ms=3)
    ax.set_xlabel("t^(-1/2)" if state.pp_axis == "tm12" else "t^(-1)")
    ax.set_ylabel("BHP (psi)")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    if state.pp_window is not None and res.pore_pressure is not None:
        lo, hi = state.pp_window
        m = (dg.t >= lo) & (dg.t <= hi)
        if m.sum() >= 2:
            from .interpret import fit_line
            slope, intercept = fit_line(x[m], dg.p[m])
            xr = np.array([0.0, x[m].max()])
            ax.plot(xr, intercept + slope * xr, color="tab:green", ls="--", lw=1.3)
            ax.plot(0.0, res.pore_pressure, "o", color="tab:green")
        ax.set_title(f"Pore pressure = {res.pore_pressure:.0f} psi", fontsize=10)
    else:
        ax.set_title("Pore pressure -- select the late-time window", fontsize=10)


RENDERERS = {
    "overview": render_overview,
    "isip": render_isip,
    "gfunction": render_gfunction,
    "tangent": render_tangent,
    "loglog": render_loglog,
    "porepressure": render_porepressure,
}
