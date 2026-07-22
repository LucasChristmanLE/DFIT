"""Pick state and derived results.

``PickState`` is the complete, JSON-serializable set of choices an interpreter makes for one test:
channel mapping, BHP inputs, the injection/shut-in picks, the ISIP tangent, closure/contact picks,
and the log-log / pore-pressure selections. ``compute_all`` turns a PickState plus a loaded
``TestData`` into a ``DerivedResults`` bundle (all numbers + the arrays the plots need).

This module is pure Python + numpy: no matplotlib, no Tkinter, so it is fully unit-testable and is
the single source of truth for every reported value.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict, fields
from typing import Optional

import numpy as np

from . import interpret, resample
from .io_load import ChannelConfig, TestData


# --------------------------------------------------------------------------------------------------
# pick state (serializable)
# --------------------------------------------------------------------------------------------------
@dataclass
class TangentPick:
    """A line defined by an anchor point and a slope, in whatever axes it lives on."""
    anchor_x: float
    anchor_y: float
    slope: float


@dataclass
class PickState:
    # --- channel mapping / BHP inputs ---
    pressure_col: str = ""
    rate_col: Optional[str] = None
    volume_col: Optional[str] = None
    pressure_is_bhp: bool = False
    density_ppg: Optional[float] = None
    tvd_ft: Optional[float] = None

    # --- G-function / resampling ---
    alpha: float = 1.0
    resample_step: float = 30.0

    # --- step 2: injection window ---
    start_idx: Optional[int] = None
    shutin_idx: Optional[int] = None
    qmax_bpm: Optional[float] = None  # auto-detected; overridable

    # --- step 3: literal ISIP tangent (BHP vs time-seconds axis) ---
    isip_tangent: Optional[TangentPick] = None

    # --- step 5: effective ISIP line (P vs G axis) + compliance contact ---
    eff_isip_line: Optional[TangentPick] = None
    contact_G: Optional[float] = None
    closure_scenario: str = ""  # C-A..C-D

    # --- step 6: tangent-method closure (G*dP/dG through-origin departure) ---
    closure_G: Optional[float] = None
    closure_slope: Optional[float] = None

    # --- step 7-8: log-log window + postclosure + pore pressure ---
    loglog_window: Optional[tuple[float, float]] = None  # (t_lo, t_hi) shut-in seconds
    postclosure_scenario: str = ""  # PC-A..PC-F
    pp_axis: str = "tm12"  # "tm12" (t^-1/2) or "tm1" (t^-1)
    pp_window: Optional[tuple[float, float]] = None  # (t_lo, t_hi) shut-in seconds

    notes: str = ""

    # --- step-bar breadcrumb: absent key means "not_visited"; other values are "visited"/
    # "done"/"skipped". Owned by the UI (DfitApp._goto/_next/_skip); rides along in to_json/
    # from_json like everything else in this dataclass. ---
    step_status: dict[str, str] = field(default_factory=dict)

    def channel_config(self) -> ChannelConfig:
        return ChannelConfig(
            pressure_col=self.pressure_col,
            pressure_is_bhp=self.pressure_is_bhp,
            rate_col=self.rate_col,
            volume_col=self.volume_col,
            mw_ppg=self.density_ppg,
            tvd_ft=self.tvd_ft,
        )

    # ---- persistence ----
    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(_encode(self), fh, indent=2)

    @staticmethod
    def from_json(path: str) -> "PickState":
        with open(path, encoding="utf-8") as fh:
            return _decode(json.load(fh))


def _encode(state: PickState) -> dict:
    d = asdict(state)
    return d


def _decode(d: dict) -> PickState:
    for key in ("isip_tangent", "eff_isip_line"):
        if d.get(key) is not None:
            d[key] = TangentPick(**d[key])
    for key in ("loglog_window", "pp_window"):
        if d.get(key) is not None:
            d[key] = tuple(d[key])
    # Filter to known field names so an old save (missing step_status -> falls to its default)
    # or a foreign/future save (extra keys we don't understand yet) never raises a TypeError
    # from an unexpected/missing keyword argument.
    known = {f.name for f in fields(PickState)}
    filtered = {k: v for k, v in d.items() if k in known}
    return PickState(**filtered)


def infer_step_status(state: PickState) -> dict[str, str]:
    """Best-effort backfill for picks files saved before ``step_status`` existed: an old file
    has real picks but no breadcrumb history, which -- left as ``{}`` -- would present as every
    step being unreached and lock the whole breadcrumb. Mark a step "done" when the pick(s) that
    define it are present; steps with no picks are left absent ("not_visited"). Used by
    ``DfitApp._load_picks`` only when the loaded ``step_status`` is empty -- an explicitly saved
    ``{}`` from a workflow that never advanced past overview is indistinguishable from "never
    recorded", and treating it as "infer" is the safer default either way.
    """
    status: dict[str, str] = {}
    if state.start_idx is not None or state.shutin_idx is not None:
        status["overview"] = "done"
    if state.isip_tangent is not None:
        status["isip"] = "done"
    if state.eff_isip_line is not None or state.contact_G is not None:
        status["gfunction"] = "done"
    if state.closure_G is not None:
        status["tangent"] = "done"
    if state.loglog_window is not None:
        status["loglog"] = "done"
    if state.pp_window is not None:
        status["porepressure"] = "done"
    return status


# --------------------------------------------------------------------------------------------------
# derived results
# --------------------------------------------------------------------------------------------------
@dataclass
class DerivedResults:
    # timing / volume
    te_s: Optional[float] = None
    vinj: Optional[float] = None
    vinj_delta: Optional[float] = None
    vinj_integral: Optional[float] = None
    vinj_source: str = ""
    vinj_disagreement: Optional[float] = None
    qmax_bpm: Optional[float] = None
    t_shutin_s: Optional[float] = None

    # pressures
    literal_isip: Optional[float] = None
    effective_isip: Optional[float] = None
    contact_pressure: Optional[float] = None
    shmin_compliance: Optional[float] = None
    shmin_tangent: Optional[float] = None
    closure_pressure: Optional[float] = None
    net_pressure_compliance: Optional[float] = None
    net_pressure_tangent: Optional[float] = None
    delta_closure: Optional[float] = None
    pore_pressure: Optional[float] = None

    # arrays for plotting (not serialized)
    t_all_s: Optional[np.ndarray] = field(default=None, repr=False)
    bhp_all: Optional[np.ndarray] = field(default=None, repr=False)
    rate_all: Optional[np.ndarray] = field(default=None, repr=False)
    resampled: Optional[resample.Resampled] = field(default=None, repr=False)
    diagnostics: Optional[resample.Diagnostics] = field(default=None, repr=False)

    warnings: list[str] = field(default_factory=list)


def compute_all(state: PickState, td: TestData) -> DerivedResults:
    """Compute every derived value that the current PickState supports. Missing picks -> None."""
    res = DerivedResults()
    cfg = state.channel_config()

    if not state.pressure_col:
        res.warnings.append("No pressure channel selected")
        return res
    if not cfg.bhp_inputs_ready():
        res.warnings.append("Surface pressure selected but density/TVD not set")

    # Full-length channels
    res.t_all_s = td.t_s
    try:
        res.bhp_all = td.bhp(cfg) if cfg.bhp_inputs_ready() else td.pressure_surface(cfg)
    except Exception as e:  # pragma: no cover - defensive
        res.warnings.append(f"BHP computation failed: {e}")
        res.bhp_all = td.pressure_surface(cfg)
    if cfg.rate_col:
        res.rate_all = td.column(cfg.rate_col)

    # Injection window + te
    if state.start_idx is not None and state.shutin_idx is not None and res.rate_all is not None:
        start, shutin = state.start_idx, state.shutin_idx
        res.t_shutin_s = float(td.t_s[shutin])
        res.qmax_bpm = state.qmax_bpm or interpret.max_sustained_rate(res.rate_all, start, shutin)
        vol = td.column(cfg.volume_col) if cfg.volume_col else None
        vr = interpret.injected_volume(td.t_s, res.rate_all, start, shutin, volume=vol)
        res.vinj, res.vinj_delta = vr.vinj, vr.vinj_delta
        res.vinj_integral, res.vinj_source = vr.vinj_integral, vr.source
        res.vinj_disagreement = vr.disagreement_frac
        if vr.disagreement_frac is not None and vr.disagreement_frac > 0.05:
            res.warnings.append(f"Volume delta vs rate-integral disagree {vr.disagreement_frac:.0%}")
        if res.qmax_bpm and res.qmax_bpm > 0:
            res.te_s = interpret.effective_te_seconds(res.vinj, res.qmax_bpm)

    # Literal ISIP (needs shut-in time)
    if state.isip_tangent and res.t_shutin_s is not None:
        tg = state.isip_tangent
        res.literal_isip = interpret.literal_isip(tg.anchor_x, tg.anchor_y, tg.slope, res.t_shutin_s)

    # Resample + diagnostics (needs te)
    if res.te_s and res.t_shutin_s is not None and res.bhp_all is not None:
        dt_all = td.t_s - res.t_shutin_s
        post = dt_all >= 0
        rs = resample.resample_pressure_increment(dt_all[post], res.bhp_all[post],
                                                  step=state.resample_step)
        res.resampled = rs
        if len(rs.p) >= 3:
            res.diagnostics = resample.diagnostics(rs, res.te_s, state.alpha)
            if len(rs.p) < 20:
                res.warnings.append(f"Only {len(rs.p)} resampled points; consider a smaller step")

    # Effective ISIP (P-vs-G line to G=0)
    if state.eff_isip_line:
        ln = state.eff_isip_line
        res.effective_isip = interpret.effective_isip(ln.anchor_x, ln.anchor_y, ln.slope)

    # Compliance contact -> Shmin
    if state.contact_G is not None and res.diagnostics is not None:
        res.contact_pressure = float(np.interp(state.contact_G, res.diagnostics.G, res.resampled.p))
        res.shmin_compliance = interpret.shmin_compliance(res.contact_pressure)

    # Tangent closure -> Shmin
    if state.closure_G is not None and res.diagnostics is not None:
        res.closure_pressure = float(np.interp(state.closure_G, res.diagnostics.G, res.resampled.p))
        res.shmin_tangent = interpret.shmin_tangent(res.closure_pressure)

    # Net pressures (effective ISIP is the reference per plan)
    ref = res.effective_isip if res.effective_isip is not None else res.literal_isip
    if ref is not None and res.shmin_compliance is not None:
        res.net_pressure_compliance = interpret.net_pressure(ref, res.shmin_compliance)
    if ref is not None and res.shmin_tangent is not None:
        res.net_pressure_tangent = interpret.net_pressure(ref, res.shmin_tangent)
    if res.shmin_compliance is not None and res.shmin_tangent is not None:
        res.delta_closure = res.shmin_compliance - res.shmin_tangent

    # Pore pressure (postclosure)
    if state.pp_window and res.diagnostics is not None:
        dg = res.diagnostics
        lo, hi = state.pp_window
        m = (dg.t >= lo) & (dg.t <= hi) & (dg.t > 0)
        if m.sum() >= 2:
            expo = -0.5 if state.pp_axis == "tm12" else -1.0
            x = dg.t[m] ** expo
            res.pore_pressure = interpret.pore_pressure(x, dg.p[m])

    return res
