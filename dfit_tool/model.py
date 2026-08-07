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
from .gfunction import g_time
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
    well_name: str = ""
    formation: str = ""

    # --- G-function / resampling ---
    alpha: float = 1.0
    resample_step: float = 30.0
    # Manual-only tail trim (no auto-detect seeder): shut-in-relative seconds beyond which the
    # post-shut-in record is discarded before diagnostics. Stored in seconds (not an index or a
    # G value) so it stays stable across alpha/resample_step changes. None = no trim (use the
    # full record). Old saves lack this key and take the default via _decode's known-field
    # filter -- no migration needed.
    tail_trim_dt: Optional[float] = None

    # --- step 2: injection window ---
    start_idx: Optional[int] = None
    shutin_idx: Optional[int] = None
    qmax_bpm: Optional[float] = None  # auto-detected; overridable

    # --- step 3: apparent ISIP tangent (BHP vs time-seconds axis) ---
    isip_tangent: Optional[TangentPick] = None

    # --- step 5: min-dP/dG point (P vs G axis; a diagnostic pick) + compliance contact (feeds
    # the derived effective-ISIP tangent, see DerivedResults.eff_isip_line) ---
    min_dpdg_G: Optional[float] = None
    contact_G: Optional[float] = None
    closure_scenario: str = ""  # C-A..C-D
    show_d2pdg2: bool = False  # overlay d2P/dG2 on the G-function step (helps spot the C-B inflection)

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

    # --- folder mode (store.py): which data file this test's picks were made against, and the
    # whole-test Skip-test flag -- a user override of the recomputed status (store.status_for)
    # for the one thing step_status can't express: parking a test outright regardless of how
    # far its steps got. "done" is never a manual choice; it is always derived from step_status.
    # Old saves lack both keys and take these defaults via _decode's known-field filter, no
    # migration needed (a legacy "done" value is normalized to None in _decode below). ---
    active_source: str = "csv"  # "csv" or "dbs"
    explicit_status: Optional[str] = None  # "skipped"/None

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
    # Migrate an old save's eff_isip_line (a stored, draggable pick on P-vs-G) to min_dpdg_G: its
    # anchor sat on the P-vs-G curve at the same G the min-dP/dG point now lives at.
    if d.get("min_dpdg_G") is None and isinstance(d.get("eff_isip_line"), dict):
        d["min_dpdg_G"] = d["eff_isip_line"].get("anchor_x")
    if d.get("isip_tangent") is not None:
        d["isip_tangent"] = TangentPick(**d["isip_tangent"])
    for key in ("loglog_window", "pp_window"):
        if d.get(key) is not None:
            d[key] = tuple(d[key])
    # A foreign/corrupted save can carry an explicit JSON null for a string field that
    # PickState defaults to "" -- e.g. compute_all calls state.closure_scenario.startswith(...)
    # unconditionally, which raises AttributeError on None. Coerce null -> "" for every scenario
    # field so a null here never raises downstream, matching this module's "old or foreign JSON
    # never raises" contract.
    for key in ("closure_scenario", "postclosure_scenario"):
        if key in d and d[key] is None:
            d[key] = ""
    # Old saves store the postclosure scenario's pre-rename label (the combobox values in
    # ui.py:POSTCLOSURE_SCENARIOS changed to the descriptive ResFrac guide titles); map the four
    # renamed labels to their current form so a loaded save matches the combobox. Unrecognized
    # strings (a current label, or a foreign value) pass through untouched -- same "old or
    # foreign JSON never raises" contract as the coercion above.
    pc_label_migrations = {
        "PC-C mixed": "PC-C false radial to genuine linear",
        "PC-D mixed": "PC-D genuine linear to genuine radial",
        "PC-E none": "PC-E no trend",
        "PC-F none": "PC-F no peak",
    }
    if d.get("postclosure_scenario") in pc_label_migrations:
        d["postclosure_scenario"] = pc_label_migrations[d["postclosure_scenario"]]
    # Old saves made with the since-removed Mark combobox can carry explicit_status == "done";
    # that value space narrowed to "skipped"/None (see the field comment above), so normalize
    # the stale "done" to None rather than let it linger as a value no other code expects.
    if d.get("explicit_status") == "done":
        d["explicit_status"] = None
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
    ``{}`` from a workflow that never advanced past injection is indistinguishable from "never
    recorded", and treating it as "infer" is the safer default either way.
    """
    status: dict[str, str] = {}
    # "overview" is deliberately absent from this backfill: it owns no picks of its own (it just
    # mirrors the injection window), and this function only exists to reconstruct step_status for
    # pre-step_status saves, which predate the "overview" step entirely.
    if state.start_idx is not None or state.shutin_idx is not None:
        status["injection"] = "done"
    if state.isip_tangent is not None:
        status["isip"] = "done"
    if state.min_dpdg_G is not None or state.contact_G is not None:
        status["gfunction"] = "done"
    if state.closure_G is not None:
        status["tangent"] = "done"
    if state.loglog_window is not None:
        status["loglog"] = "done"
    if state.pp_window is not None:
        status["porepressure"] = "done"
    return status


def step_gate_error(state: PickState, step: str) -> Optional[str]:
    """Message describing what must be completed before advancing FORWARD from ``step``,
    or ``None`` if forward navigation is allowed. Only the two scenario selections are
    enforced -- every other step's picks are auto-seeded on first visit, so they are never
    "incomplete". Gates only the "Next >" button (``DfitApp._advance``); Back, Skip, and
    breadcrumb jumps are unaffected."""
    if step == "gfunction" and not state.closure_scenario:
        return "Select a closure scenario before continuing to Tangent."
    if step == "loglog" and not state.postclosure_scenario:
        return "Select a postclosure scenario before continuing to Pore pressure."
    return None


def porepressure_skipped(state: PickState) -> bool:
    """PC-F (no peak): the derivative never peaks, so no postclosure line exists and
    the pore-pressure step is skipped entirely."""
    return state.postclosure_scenario.startswith("PC-F")


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
    apparent_isip: Optional[float] = None
    effective_isip_compliance: Optional[float] = None
    effective_isip_tangent: Optional[float] = None
    effective_isip_variable: Optional[float] = None
    contact_pressure: Optional[float] = None
    shmin_compliance: Optional[float] = None
    shmin_tangent: Optional[float] = None
    shmin_variable: Optional[float] = None
    shmin_rapid: Optional[float] = None
    closure_time_compliance_s: Optional[float] = None
    closure_time_tangent_s: Optional[float] = None
    closure_time_variable_s: Optional[float] = None
    closure_pressure: Optional[float] = None
    net_pressure_compliance: Optional[float] = None
    net_pressure_tangent: Optional[float] = None
    net_pressure_variable: Optional[float] = None
    # Which effective-ISIP source fed the shared net-pressure reference:
    # "compliance", "tangent", or "" when no reference was available.
    net_pressure_isip_source: Optional[str] = None
    # Apparent ISIP - that same shared reference ISIP. One value per test (not per method);
    # None when either the apparent ISIP or the reference is missing. Negative values are
    # reported as-is -- see _resolve_net_pressures.
    near_wellbore_complexity: Optional[float] = None
    delta_closure: Optional[float] = None
    pore_pressure: Optional[float] = None

    # arrays for plotting (not serialized)
    t_all_s: Optional[np.ndarray] = field(default=None, repr=False)
    bhp_all: Optional[np.ndarray] = field(default=None, repr=False)
    pressure_is_bhp: bool = field(default=False, repr=False)  # bhp_all holds true BHP, not surface
    rate_all: Optional[np.ndarray] = field(default=None, repr=False)
    resampled: Optional[resample.Resampled] = field(default=None, repr=False)
    # Untrimmed counterparts of resampled/diagnostics.G, kept so the renderer can draw the
    # trimmed-away tail grayed out and ui.py can convert a drag back to seconds. resampled/
    # diagnostics themselves are the trimmed arrays -- see the tail-trim block in compute_all.
    resampled_full: Optional[resample.Resampled] = field(default=None, repr=False)
    G_full: Optional[np.ndarray] = field(default=None, repr=False)  # g_time over resampled_full.dt
    diagnostics: Optional[resample.Diagnostics] = field(default=None, repr=False)
    # A gray preview of the raw tail the rise guard excluded, in G-time -- None unless the guard
    # actually fired. Built from raw (not resampled) post-shut-in samples past guard_dt, capped
    # at 2x the kept G-range and decimated to <= 500 points so a runaway tail can't blow out the
    # plot or the point count. See the resample block in compute_all.
    guard_excluded_G: Optional[np.ndarray] = field(default=None, repr=False)
    guard_excluded_p: Optional[np.ndarray] = field(default=None, repr=False)

    # The effective-ISIP tangent (P vs G): derived from state.contact_G, not a stored pick --
    # see compute_all. Not serialized (DerivedResults never is).
    eff_isip_line_compliance: Optional[TangentPick] = field(default=None, repr=False)

    warnings: list[str] = field(default_factory=list)


def _resolve_net_pressures(res: "DerivedResults") -> "DerivedResults":
    """Resolve the single shared reference ISIP -- compliance eff ISIP, else tangent eff ISIP,
    else none (no apparent-ISIP fallback) -- and set everything derived from it on ``res``:
    ``net_pressure_isip_source``, the three ``net_pressure_*``, and
    ``near_wellbore_complexity``. Each net pressure keeps its own per-method Shmin guard, so it
    stays None when the shared reference is None or its own Shmin is None. Complexity is
    guarded on the apparent ISIP instead, and a negative result is returned as-is (no clamp, no
    warning) so the identity Shmin + net + complexity = apparent ISIP stays exact."""
    if res.effective_isip_compliance is not None:
        ref, res.net_pressure_isip_source = res.effective_isip_compliance, "compliance"
    elif res.effective_isip_tangent is not None:
        ref, res.net_pressure_isip_source = res.effective_isip_tangent, "tangent"
    else:
        ref, res.net_pressure_isip_source = None, ""
    if ref is not None:
        if res.apparent_isip is not None:
            res.near_wellbore_complexity = interpret.near_wellbore_complexity(res.apparent_isip,
                                                                              ref)
        if res.shmin_compliance is not None:
            res.net_pressure_compliance = interpret.net_pressure(ref, res.shmin_compliance)
        if res.shmin_tangent is not None:
            res.net_pressure_tangent = interpret.net_pressure(ref, res.shmin_tangent)
        if res.shmin_variable is not None:
            res.net_pressure_variable = interpret.net_pressure(ref, res.shmin_variable)
    return res


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
        res.pressure_is_bhp = cfg.bhp_inputs_ready()
    except Exception as e:  # pragma: no cover - defensive
        res.warnings.append(f"BHP computation failed: {e}")
        res.bhp_all = td.pressure_surface(cfg)
        res.pressure_is_bhp = False
    if cfg.rate_col:
        res.rate_all = td.column(cfg.rate_col)

    # Injection window + te. t_shutin_s needs only the two picks; qmax/Vinj (and the
    # effective te = Vinj/qmax) need a rate channel. When the effective te is unavailable
    # (no rate channel, or a degenerate one), te falls back to the wall-clock pump duration
    # (TODO pair: rate-less datasets), with a warning so the analyst knows te is not the
    # Vinj/qmax effective time.
    if state.start_idx is not None and state.shutin_idx is not None:
        start, shutin = state.start_idx, state.shutin_idx
        res.t_shutin_s = float(td.t_s[shutin])
        if res.rate_all is not None:
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
        if res.te_s is not None and not (np.isfinite(res.te_s) and res.te_s > 0):
            res.te_s = None  # a NaN/<=0 effective te must not leak into the truthy te gate below
        if res.te_s is None and shutin > start:
            dur = float(td.t_s[shutin] - td.t_s[start])
            if dur > 0:
                res.te_s = dur
                res.warnings.append(
                    "te = pump duration (shut-in - start); no usable rate for Vinj/qmax")

    # Apparent ISIP (needs shut-in time)
    if state.isip_tangent and res.t_shutin_s is not None:
        tg = state.isip_tangent
        res.apparent_isip = interpret.apparent_isip(tg.anchor_x, tg.anchor_y, tg.slope, res.t_shutin_s)

    # Resample + diagnostics (needs te). Resample the full post-shut-in record first, then mask
    # to the tail trim (if any) before diagnostics -- everything downstream (effective ISIP,
    # Shmin, log-log, pore pressure) reads only res.resampled/res.diagnostics, so the trim
    # propagates everywhere with no other changes. resampled_full/G_full stay available
    # (untrimmed) so the renderer can gray out the excluded tail and ui.py can convert a drag
    # back to seconds.
    if res.te_s and res.t_shutin_s is not None and res.bhp_all is not None:
        dt_all = td.t_s - res.t_shutin_s
        post = dt_all >= 0
        rs_full = resample.resample_pressure_increment(dt_all[post], res.bhp_all[post],
                                                        step=state.resample_step)
        res.resampled_full = rs_full
        res.G_full = g_time(rs_full.dt, res.te_s, state.alpha)
        if rs_full.guard_dt is not None:
            # A gray preview of what the guard threw away, in G-time -- built from the raw (not
            # resampled) samples past guard_dt, since the resampler itself kept none of them.
            # Non-finite pressures are left in on purpose: matplotlib skips NaN when drawing, so
            # filtering here would just be extra work for the same visual result. Capped at 2x
            # the kept G-range so a runaway crash-to-zero tail can't blow out the plot's
            # autoscale, and decimated (ceiling stride, so the result is always <= 500 -- a
            # floor stride via `n // 500` can leave up to 999) so a very long raw tail can't
            # blow out the point count.
            dt_raw, p_raw = dt_all[post], res.bhp_all[post]
            tail_dt = dt_raw[dt_raw > rs_full.guard_dt]
            tail_p = p_raw[dt_raw > rs_full.guard_dt]
            if len(res.G_full):
                tail_G = g_time(tail_dt, res.te_s, state.alpha)
                within = tail_G <= 2.0 * res.G_full[-1]
                tail_G, tail_p = tail_G[within], tail_p[within]
                n = len(tail_G)
                if n:
                    stride = -(-n // 500)
                    res.guard_excluded_G = tail_G[::stride]
                    res.guard_excluded_p = tail_p[::stride]
            # Inserted at the front (not appended) so this stays the topmost line in warn_lbl's
            # stacked display, ahead of any earlier-queued warning (density/TVD, volume
            # disagreement, ...) -- a firing guard must never be buried in the UI. The tail-trim
            # escape warning below (its own insert(0), for a diagnostics-starving trim) runs after
            # this in code order, so it still lands frontmost of the two when both fire.
            res.warnings.insert(0,
                f"Tail guard stopped resampling {rs_full.guard_dt/60:.0f} min after shut-in "
                "(sustained pressure rise); later data excluded")
        if state.tail_trim_dt is not None:
            mask = rs_full.dt <= state.tail_trim_dt
            rs = resample.Resampled(dt=rs_full.dt[mask], p=rs_full.p[mask], n_raw=rs_full.n_raw,
                                    guarded_at=rs_full.guarded_at, guard_dt=rs_full.guard_dt)
        else:
            rs = rs_full
        res.resampled = rs
        if len(rs.p) >= 3:
            res.diagnostics = resample.diagnostics(rs, res.te_s, state.alpha)
            if len(rs.p) < 20:
                res.warnings.append(f"Only {len(rs.p)} resampled points; consider a smaller step")
        elif state.tail_trim_dt is not None:
            # A pathological trim leaves too few points to diagnose -- warn rather than let it be
            # a silent dead end (the renderer/controller recovery path still lets the analyst drag
            # the trim line back right from here). Inserted at the front so it stays the topmost
            # line in warn_lbl's stacked display even when other warnings already queued ahead of
            # it -- this is the escape instruction for an otherwise-blank plot.
            res.warnings.insert(0, f"Tail trim leaves only {len(rs.p)} resampled point(s); drag "
                                   "the trim line back right")

    # Low-surface-pressure warning: only when the mapped channel is surface pressure
    # (state.pressure_is_bhp, not res.pressure_is_bhp -- that flips True after hydrostatic
    # conversion too, which would hide the very condition that makes the conversion unreliable).
    # The upper bound is where the resampler actually STOPPED CONSUMING raw samples
    # (resampled_full.guard_dt, strict < to exclude the rising sample itself), not the last
    # *kept* point -- dt[-1] can sit up to one resample_step below guard_dt, so bounding there
    # would leave a gap of un-scanned raw samples right where a crash typically lives (the
    # literal motivating case: a record bottoming out just above the last kept point never
    # warned). A trim, if set, narrows the window further on top of that. Falls back to trim-
    # only (or no upper bound at all) when the resample block above never ran.
    if not state.pressure_is_bhp and res.t_shutin_s is not None:
        surf = td.pressure_surface(cfg)          # raw WHP column -- NOT res.bhp_all (converted)
        dt_ws = td.t_s - res.t_shutin_s
        m = dt_ws >= 0
        if res.resampled_full is not None and res.resampled_full.guard_dt is not None:
            m &= dt_ws < res.resampled_full.guard_dt
        if state.tail_trim_dt is not None:
            m &= dt_ws <= state.tail_trim_dt
        surf_kept = surf[m]
        finite = np.isfinite(surf_kept)
        if finite.any() and float(np.min(surf_kept[finite])) < interpret.MIN_SURFACE_PRESSURE_PSI:
            res.warnings.append("Surface pressure fell below 100 psi post-shut-in -- BHP "
                                "unreliable there; consider trimming the tail")

    # Stale-pick warning: only meaningful once a trim is actually in effect -- gating on
    # state.tail_trim_dt avoids misleadingly reporting picks as "beyond the tail trim" for a
    # reloaded save against a shorter *source* with no trim set at all. np.interp silently
    # clamps an out-of-range G to the trimmed edge, which would otherwise produce a stale-but-
    # plausible number for a pick that now lies beyond the trimmed tail.
    if state.tail_trim_dt is not None and res.diagnostics is not None and len(res.diagnostics.G):
        edge = res.diagnostics.G[-1]
        stale = [name for name, g in (("contact", state.contact_G), ("min dP/dG", state.min_dpdg_G),
                                      ("closure", state.closure_G)) if g is not None and g > edge]
        # The pore-pressure fit masks its window against the diagnostics' post-shut-in time
        # array (dg.t), not G -- see the pp block below -- so a pp_window affected by the trim
        # can silently shrink (still fits, just off fewer samples) or empty outright (blanking
        # pore_pressure) with no other warning. Checked by outcome, not bound-vs-edge: a lower
        # bound sitting inside the data but in the last inter-sample gap is still "<= edge" yet
        # leaves the same <2-sample mask the pp block itself requires to fit at all, so the
        # emptied case is the pp block's own `m.sum() >= 2` guard, mirrored here. An open-ended
        # *upper* bound (t_hi = inf, "to the end of the data") naturally shrinks with the trim
        # instead of emptying, so it's exempt from the shrunk-but-still-fitting check.
        if state.pp_window is not None and len(res.diagnostics.t):
            pp_lo, pp_hi = state.pp_window
            t = res.diagnostics.t
            edge_t = t[-1]
            emptied = int(((t >= pp_lo) & (t <= pp_hi)).sum()) < 2
            shrunk = np.isfinite(pp_hi) and pp_hi > edge_t
            if emptied or shrunk:
                stale.append("pore-pressure window")
        if stale:
            res.warnings.append(f"{', '.join(stale)} pick(s) lie beyond the tail trim -- values "
                                "may be stale")

    # Effective ISIP: tangent to P-vs-G at the contact point, extrapolated to G=0. Derived here
    # (not a stored pick) -- the anchor is the diagnostics sample nearest state.contact_G, the
    # slope a local fit (half=4) around it, same math the old draggable "anchor" commit used. The
    # min-dP/dG point (state.min_dpdg_G) stays a separate diagnostic pick -- it no longer feeds
    # this line.
    if state.contact_G is not None and res.diagnostics is not None and res.resampled is not None:
        dg = res.diagnostics
        idx = int(np.nanargmin(np.abs(dg.G - state.contact_G)))
        anchor_x, anchor_y, slope = interpret.tangent_from_index(dg.G, res.resampled.p, idx,
                                                                  half=4)
        res.eff_isip_line_compliance = TangentPick(anchor_x=anchor_x, anchor_y=anchor_y, slope=slope)
        res.effective_isip_compliance = interpret.effective_isip(anchor_x, anchor_y, slope)

    # Compliance contact -> Shmin
    if state.contact_G is not None and res.diagnostics is not None:
        res.contact_pressure = float(np.interp(state.contact_G, res.diagnostics.G, res.resampled.p))
        res.shmin_compliance = interpret.shmin_compliance(res.contact_pressure)
        res.closure_time_compliance_s = float(np.interp(state.contact_G, res.diagnostics.G,
                                                          res.resampled.dt))

    # C-D rapid closure: Shmin ~= apparent ISIP - 175 psi (no contact pick, so no *compliance*
    # effective ISIP -- the tangent one still exists and still feeds the shared reference; this
    # is a separate field so it doesn't feed net_pressure_compliance/delta_closure, see
    # ../CLAUDE.md and the plan's decision D2).
    if state.closure_scenario.startswith("C-D") and res.apparent_isip is not None:
        res.shmin_rapid = interpret.shmin_rapid(res.apparent_isip)

    # Tangent closure -> Shmin
    if state.closure_G is not None and res.diagnostics is not None:
        res.closure_pressure = float(np.interp(state.closure_G, res.diagnostics.G, res.resampled.p))
        res.shmin_tangent = interpret.shmin_tangent(res.closure_pressure)
        res.closure_time_tangent_s = float(np.interp(state.closure_G, res.diagnostics.G,
                                                       res.resampled.dt))

    # Effective ISIP (tangent method): same construction as the compliance block above, anchored
    # at state.closure_G instead of state.contact_G.
    if state.closure_G is not None and res.diagnostics is not None and res.resampled is not None:
        dg = res.diagnostics
        idx = int(np.nanargmin(np.abs(dg.G - state.closure_G)))
        x, y, slope = interpret.tangent_from_index(dg.G, res.resampled.p, idx, half=4)
        res.effective_isip_tangent = interpret.effective_isip(x, y, slope)

    # Variable compliance method: average the raw contact/closure picks in G-time, then read
    # Shmin (variable) off the P-vs-G curve at that midpoint and build its own effective ISIP the
    # same way as the other two methods. Guarded on both picks being present.
    if (state.contact_G is not None and state.closure_G is not None
            and res.diagnostics is not None and res.resampled is not None):
        dg = res.diagnostics
        G_var = (state.contact_G + state.closure_G) / 2.0
        res.shmin_variable = float(np.interp(G_var, dg.G, res.resampled.p))
        res.closure_time_variable_s = float(np.interp(G_var, dg.G, res.resampled.dt))
        idx = int(np.nanargmin(np.abs(dg.G - G_var)))
        x, y, slope = interpret.tangent_from_index(dg.G, res.resampled.p, idx, half=4)
        res.effective_isip_variable = interpret.effective_isip(x, y, slope)

    _resolve_net_pressures(res)
    if res.shmin_compliance is not None and res.shmin_tangent is not None:
        res.delta_closure = res.shmin_compliance - res.shmin_tangent

    # Pore pressure (postclosure). PC-F ("no peak") means the derivative never peaks, so no
    # postclosure line exists -- suppress the fit even if a stale pp_window pick is present.
    if state.pp_window and res.diagnostics is not None and not porepressure_skipped(state):
        dg = res.diagnostics
        lo, hi = state.pp_window
        m = (dg.t >= lo) & (dg.t <= hi) & (dg.t > 0)
        if m.sum() >= 2:
            expo = -0.5 if state.pp_axis == "tm12" else -1.0
            x = dg.t[m] ** expo
            res.pore_pressure = interpret.pore_pressure(x, dg.p[m])

    return res
