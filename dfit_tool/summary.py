"""Headless tables for the Expanded results window: groups and compares what ``compute_all``
produced. Tk-free; imports model and interpret only. Every cell is a pre-formatted string, "-"
for a missing value or a value whose owning step is still not visited.

Computes nothing new: every number comes from ``DerivedResults`` or ``PickState``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import interpret
from .model import DerivedResults, PickState, pp_from_peak, skipped_steps

DASH = "-"


@dataclass
class Section:
    title: str
    columns: list[str]
    rows: list[list[str]]


def visited(state: PickState, step: str) -> bool:
    """True once the analyst has opened ``step`` (any status except "not_visited"). Values
    whose owning step is not visited display "-" even when ``compute_all`` produced them, e.g.
    from a loaded picks file."""
    return state.step_status.get(step, "not_visited") != "not_visited"


def finite_or_none(v) -> Optional[float]:
    """``v`` as a finite float, else None. Guards hand-edited saves where ``tvd_ft`` or
    ``density_ppg`` can be a string, NaN or inf (same coercion as model._resolve_gradients)."""
    try:
        x = float(v)
    except (TypeError, ValueError, OverflowError):
        return None
    return x if math.isfinite(x) else None


def _fmt(v, f: str = "{:.0f}") -> str:
    v = finite_or_none(v) if not isinstance(v, int) else v
    return f.format(v) if v is not None else DASH


def _min(v_s: Optional[float]) -> Optional[float]:
    return v_s / 60.0 if v_s is not None else None


def _gate(state: PickState, step: str, cells: list[str]) -> list[str]:
    """Blank every cell after the label when ``step`` is not visited."""
    if visited(state, step):
        return cells
    return [cells[0]] + [DASH] * (len(cells) - 1)


def _injection(state: PickState, res: DerivedResults) -> Section:
    vinj = _fmt(res.vinj, "{:.1f}")
    if res.vinj is not None and res.vinj_source:
        vinj = f"{vinj} ({res.vinj_source})"
    rows = [
        ["te (min)", _fmt(_min(res.te_s) if res.te_s else None, "{:.2f}")],
        ["Vinj (bbl)", vinj],
        ["Vinj by volume delta (bbl)", _fmt(res.vinj_delta, "{:.1f}")],
        ["Vinj by rate integral (bbl)", _fmt(res.vinj_integral, "{:.1f}")],
        ["Vinj disagreement (%)",
         _fmt(res.vinj_disagreement * 100.0 if res.vinj_disagreement is not None else None,
              "{:.1f}")],
        ["qmax (bpm)", _fmt(res.qmax_bpm, "{:.2f}")],
    ]
    return Section("Injection", ["Item", "Value"], [_gate(state, "injection", r) for r in rows])


def _isip(state: PickState, res: DerivedResults) -> Section:
    app = _fmt(res.apparent_isip)
    label = "apparent ISIP"
    if res.apparent_isip is not None and res.apparent_isip_method:
        app = f"{app} ({res.apparent_isip_method})"
    rows = [
        _gate(state, "isip", [label, app, _fmt(res.apparent_isip_gradient, "{:.3f}")]),
        _gate(state, "gfunction", ["eff ISIP (compliance)", _fmt(res.effective_isip_compliance),
                                   _fmt(res.effective_isip_compliance_gradient, "{:.3f}")]),
        _gate(state, "tangent", ["eff ISIP (tangent)", _fmt(res.effective_isip_tangent),
                                 _fmt(res.effective_isip_tangent_gradient, "{:.3f}")]),
        _gate(state, "tangent", ["eff ISIP (variable)", _fmt(res.effective_isip_variable),
                                 _fmt(res.effective_isip_variable_gradient, "{:.3f}")]),
        _gate(state, "gfunction", ["reference ISIP source",
                                   res.net_pressure_isip_source or DASH, ""]),
    ]
    # Complexity is referenced to the shared ISIP; mark it when that fell back to the tangent
    # one. The asterisk follows the same gate as the value, so it never sits next to "-".
    cx = _gate(state, "gfunction", ["NWB complexity", _fmt(res.near_wellbore_complexity), ""])
    if (res.near_wellbore_complexity is not None and res.net_pressure_isip_source == "tangent"
            and cx[1] != DASH):
        cx[0] = "NWB complexity*"
    rows.append(cx)
    return Section("ISIP", ["Item", "Value (psi)", "Gradient (psi/ft)"], rows)


# Short headers so eight columns fit the tables pane; pressures are psi, gradient psi/ft.
CLOSURE_COLUMNS = ["Method", "G", "tc (min)", "P pick", "Shmin", "psi/ft", "Eff ISIP", "Net"]


def _closure(state: PickState, res: DerivedResults) -> Section:
    def r(label, step, G, tc_s, p, shmin, grad, eff=None, net=None):
        return _gate(state, step, [label, _fmt(G, "{:.3f}"), _fmt(_min(tc_s), "{:.2f}"),
                                   _fmt(p), _fmt(shmin), _fmt(grad, "{:.3f}"),
                                   _fmt(eff), _fmt(net)])
    rows = [
        r("compliance (contact)", "gfunction", res.closure_G_compliance,
          res.closure_time_compliance_s, res.contact_pressure, res.shmin_compliance,
          res.shmin_compliance_gradient, res.effective_isip_compliance,
          res.net_pressure_compliance),
        r("tangent (closure)", "tangent", res.closure_G_tangent, res.closure_time_tangent_s,
          res.closure_pressure, res.shmin_tangent, res.shmin_tangent_gradient,
          res.effective_isip_tangent, res.net_pressure_tangent),
        r("variable", "tangent", res.closure_G_variable,
          res.closure_time_variable_s, res.shmin_variable, res.shmin_variable,
          res.shmin_variable_gradient, res.effective_isip_variable,
          res.net_pressure_variable),
        r("Liberty", "gfunction", res.liberty_anchor_G, res.liberty_anchor_time_s,
          res.liberty_anchor_pressure, res.shmin_liberty, res.shmin_liberty_gradient),
        r("stiffness", "stiffness", None, None,
          state.stiffness_pick_P if res.shmin_stiffness is not None else None,
          res.shmin_stiffness, res.shmin_stiffness_gradient),
    ]
    if res.shmin_rapid is not None or state.closure_scenario.startswith("C-D"):
        rows.append(r("rapid (C-D)", "gfunction", None, None, None, res.shmin_rapid,
                      res.shmin_rapid_gradient))
    pad = [""] * (len(CLOSURE_COLUMNS) - 2)
    footer = [
        _gate(state, "gfunction", ["closure scenario", state.closure_scenario or DASH] + pad),
        _gate(state, "gfunction", ["min-dP/dG G", _fmt(state.min_dpdg_G, "{:.3f}")] + pad),
        _gate(state, "tangent", ["delta closure (psi)", _fmt(res.delta_closure)] + pad),
        ["alpha", f"{state.alpha:g}"] + pad,
    ]
    return Section("Closure comparison", CLOSURE_COLUMNS, rows + footer)


def _postclosure(state: PickState, res: DerivedResults) -> Section:
    rows = [_gate(state, "loglog", ["scenario", state.postclosure_scenario or DASH])]
    if "porepressure" in skipped_steps(state):
        rows.append(["pore pressure fit", f"skipped ({state.postclosure_scenario[:4]})"])
        return Section("Postclosure", ["Item", "Value"], rows)
    if pp_from_peak(state):
        # PC-E: no window (model.compute_all). The peak is picked on Log-log; the values gate on
        # porepressure like chart_values and the sidebar.
        peak = None if res.pce_peak_t is None else res.pce_peak_t / 60.0
        rows += [
            _gate(state, "loglog", ["method", "-1/2 from peak"]),
            _gate(state, "loglog", ["peak (min)", _fmt(peak, "{:.2f}")]),
            _gate(state, "porepressure", ["slope (psi per t^(-1/2))",
                                          _fmt(res.pore_pressure_slope, "{:.1f}")]),
            _gate(state, "porepressure", ["pore pressure (psi)", _fmt(res.pore_pressure)]),
            _gate(state, "porepressure", ["gradient (psi/ft)",
                                          _fmt(res.pore_pressure_gradient, "{:.3f}")]),
        ]
        return Section("Postclosure", ["Item", "Value"], rows)
    axis = {"tm12": "t^(-1/2)", "tm1": "t^(-1)"}.get(state.pp_axis, state.pp_axis)
    if state.pp_window is not None:
        lo, hi = state.pp_window
        window = f"{lo / 60.0:.2f} to " + (f"{hi / 60.0:.2f}" if math.isfinite(hi) else "end")
    else:
        window = DASH
    rows += [
        _gate(state, "porepressure", ["axis", axis]),
        _gate(state, "porepressure", ["window (min)", window]),
        _gate(state, "porepressure", ["points in window",
                                      _fmt(res.pore_pressure_n_points, "{:d}")]),
        _gate(state, "porepressure", ["fit slope (psi per axis unit)",
                                      _fmt(res.pore_pressure_slope, "{:.1f}")]),
        _gate(state, "porepressure", ["pore pressure (psi)", _fmt(res.pore_pressure)]),
        _gate(state, "porepressure", ["gradient (psi/ft)",
                                      _fmt(res.pore_pressure_gradient, "{:.3f}")]),
    ]
    return Section("Postclosure", ["Item", "Value"], rows)


def _inputs(state: PickState, res: DerivedResults) -> Section:
    if res.bhp_all is None:
        source = DASH
    elif res.p_surface_all is not None:
        source = "BHP converted from surface"
    elif res.pressure_is_bhp:
        source = "BHP channel"
    else:
        source = "surface pressure (not converted)"
    cut = None
    if res.resampled_full is not None:
        cut = interpret.resolve_tail_cut_dt(state.tail_trim_dt, res.resampled_full.guard_dt,
                                            state.tail_guard_override)
    rows = [
        ["TVD (ft)", _fmt(finite_or_none(state.tvd_ft))],
        ["fluid density (ppg)", _fmt(finite_or_none(state.density_ppg), "{:.2f}")],
        ["pressure source", source],
        ["units note", res.unit_conversion_note or "none"],
        ["resampled points", str(len(res.resampled.p)) if res.resampled is not None else DASH],
        ["dropouts masked", str(len(res.dropouts)) if res.bhp_all is not None else DASH],
        ["rises masked", str(len(res.rise_excursions)) if res.bhp_all is not None else DASH],
        ["manual masks", (f"{len(state.mask_intervals)} mask, {len(state.keep_intervals)} keep"
                          if res.bhp_all is not None else DASH)],
        ["tail cut (min after shut-in)", _fmt(_min(cut), "{:.2f}")],
    ]
    return Section("Inputs and data", ["Item", "Value"], rows)


@dataclass
class ChartValues:
    """The values ``plots.render_summary`` draws, already passed through the same not-visited
    gate as the tables (a gated value is None). ``tvd_ft`` is a finite float or None."""
    apparent_isip: Optional[float] = None
    eff_isip_compliance: Optional[float] = None
    eff_isip_tangent: Optional[float] = None
    eff_isip_variable: Optional[float] = None
    shmin_compliance: Optional[float] = None
    shmin_rapid: Optional[float] = None
    shmin_tangent: Optional[float] = None
    shmin_variable: Optional[float] = None
    shmin_liberty: Optional[float] = None
    shmin_stiffness: Optional[float] = None
    pore_pressure: Optional[float] = None
    net_compliance: Optional[float] = None
    net_tangent: Optional[float] = None
    net_variable: Optional[float] = None
    complexity: Optional[float] = None
    min_dpdg_G: Optional[float] = None
    min_dpdg_tc_s: Optional[float] = None
    G_compliance: Optional[float] = None
    G_tangent: Optional[float] = None
    G_variable: Optional[float] = None
    tc_compliance_s: Optional[float] = None
    tc_tangent_s: Optional[float] = None
    tc_variable_s: Optional[float] = None
    min_dpdg_p: Optional[float] = None
    # P-vs-G curve the closure chart draws the picks on (gated on gfunction like the picks).
    curve_G: Optional[np.ndarray] = field(default=None, repr=False)
    curve_p: Optional[np.ndarray] = field(default=None, repr=False)
    tvd_ft: Optional[float] = None

    def has_any(self) -> bool:
        return any(v is not None for k, v in self.__dict__.items()
                   if k not in ("tvd_ft", "curve_G", "curve_p"))


def chart_values(state: PickState, res: DerivedResults) -> ChartValues:
    """Gate every charted value by its owning step (the same steps the tables use)."""
    def g(step, v):
        return v if visited(state, step) else None
    cv = ChartValues(
        apparent_isip=g("isip", res.apparent_isip),
        eff_isip_compliance=g("gfunction", res.effective_isip_compliance),
        eff_isip_tangent=g("tangent", res.effective_isip_tangent),
        eff_isip_variable=g("tangent", res.effective_isip_variable),
        shmin_compliance=g("gfunction", res.shmin_compliance),
        shmin_rapid=g("gfunction", res.shmin_rapid),
        shmin_tangent=g("tangent", res.shmin_tangent),
        shmin_variable=g("tangent", res.shmin_variable),
        shmin_liberty=g("gfunction", res.shmin_liberty),
        shmin_stiffness=g("stiffness", res.shmin_stiffness),
        pore_pressure=g("porepressure", res.pore_pressure),
        net_compliance=g("gfunction", res.net_pressure_compliance),
        net_tangent=g("tangent", res.net_pressure_tangent),
        net_variable=g("tangent", res.net_pressure_variable),
        complexity=g("gfunction", res.near_wellbore_complexity),
        # Positioned on the curve only when compute_all reported a pressure for it (not C-X).
        min_dpdg_G=g("gfunction", state.min_dpdg_G if res.min_dpdg_pressure is not None
                     else None),
        min_dpdg_p=g("gfunction", res.min_dpdg_pressure),
        min_dpdg_tc_s=g("gfunction", res.min_dpdg_time_s),
        G_compliance=g("gfunction", res.closure_G_compliance),
        G_tangent=g("tangent", res.closure_G_tangent),
        G_variable=g("tangent", res.closure_G_variable),
        tc_compliance_s=g("gfunction", res.closure_time_compliance_s),
        tc_tangent_s=g("tangent", res.closure_time_tangent_s),
        tc_variable_s=g("tangent", res.closure_time_variable_s),
    )
    if (visited(state, "gfunction") and res.diagnostics is not None
            and res.resampled is not None and len(res.diagnostics.G)):
        cv.curve_G, cv.curve_p = res.diagnostics.G, res.resampled.p
    tvd = finite_or_none(state.tvd_ft)
    cv.tvd_ft = tvd if tvd is not None and tvd > 0 else None
    return cv


def summary_sections(state: PickState, res: DerivedResults) -> list[Section]:
    """The five tables of the Expanded results window, in display order."""
    return [_injection(state, res), _isip(state, res), _closure(state, res),
            _postclosure(state, res), _inputs(state, res)]
