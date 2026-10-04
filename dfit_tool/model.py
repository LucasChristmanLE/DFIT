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
import math
from dataclasses import dataclass, field, asdict, fields
from typing import Optional

import numpy as np

from . import interpret, io_load, resample
from .gfunction import g_time
from .io_load import ChannelConfig, TestData


# Closure scenarios with no contact pick, so no compliance Shmin/effective ISIP, no Liberty, no
# variable. C-X ("uninterpretable") is the analyst's explicit "this G-function can't be read" --
# it additionally blanks Shmin(stiffness), whose curve is anchored on the min-dP/dG pick.
NO_CONTACT_SCENARIOS = ("C-C", "C-D", "C-X")


def closure_uninterpretable(state: "PickState") -> bool:
    """C-X: the G-function step was marked uninterpretable."""
    return state.closure_scenario.startswith("C-X")


# Postclosure scenarios with no straight late-time trend: the log-log window is neither drawn
# nor fitted (the pick stays in state). PC-X ("uninterpretable") keeps its window and slope: a
# test can show a clear slope that is far from both -1/2 and -1.
NO_TREND_POSTCLOSURE = ("PC-E", "PC-F")


def loglog_window_suppressed(state: "PickState") -> bool:
    """PC-E/PC-F: no log-log window is shown or fitted."""
    return state.postclosure_scenario.startswith(NO_TREND_POSTCLOSURE)


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
    # Per-channel unit override, "auto" (the default) meaning "detect it" -- see
    # io_load.detect_channel_unit/refresh_unit_detection. Old saves lack these keys and take the
    # "auto" default via _decode's known-field filter, no migration needed.
    pressure_unit: str = "auto"
    rate_unit: str = "auto"
    volume_unit: str = "auto"
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
    # Attribution for tail_trim_dt: "low_pressure" when picks.seed_tail_trim auto-applied it
    # (a sub-100-psi surface-pressure crash), "" when unset, set manually, or cleared --
    # commit_tail_trim always resets this to "" on a manual drag/clear. Never "rise_guard": that
    # reason sets no pick at all, since the resampler already excluded that data (see the
    # resample block in compute_all below). Old saves lack this key and take the default via
    # _decode's known-field filter -- no migration needed.
    tail_trim_reason: str = ""
    # Set by picks.commit_tail_trim when a committed drag lands past guard_dt: True means the
    # analyst deliberately dragged the trim line past the rise guard's boundary on purpose, so
    # interpret.resolve_tail_cut_dt lets tail_trim_dt stick there instead of clamping back to
    # guard_dt. False (the default, and what any other commit/clear leaves it at) means a trim
    # sitting past guard_dt is presumed stale -- e.g. left behind by a shut-in move before
    # picks.resync_auto_tail_trim ran -- and gets clamped. Old saves lack this key and take the
    # default via _decode's known-field filter, no migration needed.
    tail_guard_override: bool = False
    # Analyst-drawn pressure-mask intervals, absolute td.t_s seconds (not shut-in-relative, so a
    # shut-in move never shifts them): (lo, hi) inclusive, applied post-shut-in only. A mask
    # interval marks a gauge glitch the auto detectors missed; a keep interval overrides the
    # auto-masking (dropouts and rise excursions) over the samples it covers. Both feed
    # DerivedResults.dropout_mask in compute_all. Old saves lack these keys and take the default;
    # _decode drops malformed entries.
    mask_intervals: list[tuple[float, float]] = field(default_factory=list)
    keep_intervals: list[tuple[float, float]] = field(default_factory=list)

    # --- step 2: injection window ---
    start_idx: Optional[int] = None
    shutin_idx: Optional[int] = None
    qmax_bpm: Optional[float] = None  # auto-detected; overridable

    # --- step 3: apparent ISIP tangent (BHP vs time-seconds axis) ---
    isip_tangent: Optional[TangentPick] = None
    # Take the apparent ISIP as the BHP at the shut-in sample instead of the tangent (tests with
    # no water hammer). The tangent pick is left in state so unchecking restores it. Old saves
    # take the default via _decode's known-field filter, no migration needed.
    isip_at_shutin: bool = False

    # --- step 5: min-dP/dG point (P vs G axis; a diagnostic pick) + compliance contact (feeds
    # the derived effective-ISIP tangent, see DerivedResults.eff_isip_line) ---
    min_dpdg_G: Optional[float] = None
    contact_G: Optional[float] = None
    closure_scenario: str = ""  # C-A..C-D
    show_d2pdg2: bool = False  # overlay d2P/dG2 on the G-function step (helps spot the C-B inflection)

    # --- step 6: tangent-method closure (G*dP/dG through-origin departure) ---
    closure_G: Optional[float] = None
    closure_slope: Optional[float] = None
    # Explicit negative finding: the tangent method can't be read on this test. Same pattern as
    # stiffness_no_upturn: suppresses only the reported tangent values (Shmin tangent, tangent
    # effective ISIP, and the variable method, which needs closure_G) in compute_all; the pick is
    # left in state so unchecking restores it. Old saves take the default via _decode. ---
    tangent_uninterpretable: bool = False

    # --- step 7-8: log-log window + postclosure + pore pressure ---
    loglog_window: Optional[tuple[float, float]] = None  # (t_lo, t_hi) shut-in seconds
    postclosure_scenario: str = ""  # PC-A..PC-F
    # True while postclosure_scenario is PC-A set by picks.auto_assign_postclosure (a near -1/2
    # window slope) rather than by the analyst; a manual scenario change clears it.
    postclosure_auto: bool = False
    pp_axis: str = "tm12"  # "tm12" (t^-1/2) or "tm1" (t^-1)
    pp_window: Optional[tuple[float, float]] = None  # (t_lo, t_hi) shut-in seconds

    # --- step 9: relative stiffness (URTeC-2019-123 A.8/A.9) -- a draggable vline pick on the
    # semilog-y stiffness-vs-effective-pressure plot, in psi. Comparison-only: feeds
    # Shmin(stiffness) = this - 75 psi (interpret.shmin_compliance) and nothing else. Needs the
    # pore-pressure estimate (see skipped_steps/compute_all), so it is skipped end to end
    # under PC-F exactly like porepressure. Old saves lack this key and take the default via
    # _decode's known-field filter, no migration needed. ---
    stiffness_pick_P: Optional[float] = None
    # Explicit negative finding: "no slope change apparent" on the relative-stiffness plot --
    # same precedent as closure scenario C-C's "no contact -> no Shmin". Suppresses only the
    # reported shmin_stiffness (compute_all); the pick, arrays, and curve are untouched, so
    # unchecking restores whatever was already picked. Old saves lack this key and take the
    # default via _decode's known-field filter, no migration needed. ---
    stiffness_no_upturn: bool = False

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
    active_source: str = "csv"  # "csv", "dbs", or "xlsx"
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


def _clean_intervals(raw) -> list[tuple[float, float]]:
    """Mask/keep intervals from a save as (float, float) tuples. Drops any entry that is not a
    2-sequence of finite real numbers with lo < hi; a null or non-list value gives []. Never
    raises (the module's "old or foreign JSON never raises" contract)."""
    if not isinstance(raw, (list, tuple)):
        return []
    out: list[tuple[float, float]] = []
    for item in raw:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            continue
        if any(isinstance(x, bool) or not isinstance(x, (int, float)) for x in item):
            continue
        lo, hi = float(item[0]), float(item[1])
        if math.isfinite(lo) and math.isfinite(hi) and lo < hi:
            out.append((lo, hi))
    return out


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
    for key in ("mask_intervals", "keep_intervals"):
        if key in d:
            d[key] = _clean_intervals(d[key])
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
    if state.isip_tangent is not None or state.isip_at_shutin:
        status["isip"] = "done"
    if state.min_dpdg_G is not None or state.contact_G is not None:
        status["gfunction"] = "done"
    if state.closure_G is not None or state.tangent_uninterpretable:
        status["tangent"] = "done"
    if state.loglog_window is not None:
        status["loglog"] = "done"
    if state.pp_window is not None:
        status["porepressure"] = "done"
    if state.stiffness_pick_P is not None or state.stiffness_no_upturn:
        status["stiffness"] = "done"
    return status


def step_gate_error(state: PickState, step: str) -> Optional[str]:
    """Message describing what must be completed before advancing FORWARD from ``step``,
    or ``None`` if forward navigation is allowed. Only the two scenario selections are
    enforced -- every other step's picks are auto-seeded on first visit, so they are never
    "incomplete". Gates only the "Next >" button (``DfitApp._advance``); Back, Skip, and
    breadcrumb jumps are unaffected, except that a blocking issue (``blocking_issues``) also
    gates Skip on Overview, and ``DfitApp._goto`` redirects every other step to Overview while
    one exists."""
    if step == "overview":
        issues = blocking_issues(state)
        if issues == [SURFACE_NEEDS_BHP_INPUTS]:
            return "Enter density and TVD (or map a BHP channel) before continuing."
        if issues:
            return f"{issues[0]}; fix it before continuing."
    if step == "gfunction" and not state.closure_scenario:
        return "Select a closure scenario before continuing to Tangent."
    if step == "loglog" and not state.postclosure_scenario:
        return "Select a postclosure scenario before continuing to Pore pressure."
    return None


NO_PRESSURE_CHANNEL = "No pressure channel selected"
SURFACE_NEEDS_BHP_INPUTS = "Surface pressure selected but density/TVD not set"


def blocking_issues(state: PickState) -> list[str]:
    """Conditions that make every downstream step meaningless, so navigation past Overview is
    blocked while any holds: no pressure channel, or surface pressure with no density/TVD to
    convert it to BHP. ``compute_all`` reports these (plus a failed BHP conversion) as
    ``DerivedResults.blockers``."""
    if not state.pressure_col:
        return [NO_PRESSURE_CHANNEL]
    if not state.channel_config().bhp_inputs_ready():
        return [SURFACE_NEEDS_BHP_INPUTS]
    return []


# --------------------------------------------------------------------------------------------------
# workflow steps
# --------------------------------------------------------------------------------------------------
# The eight workflow steps, in order, with their breadcrumb labels. The one step list: ui's
# breadcrumbs, store.status_for, and the PNG export all read it from here.
STEPS = [
    ("overview", "Overview"),
    ("injection", "Injection"),
    ("isip", "Apparent ISIP"),
    ("gfunction", "G-function"),
    ("tangent", "Tangent"),
    ("loglog", "Log-log"),
    ("porepressure", "Pore pressure"),
    ("stiffness", "Stiffness"),
]
STEP_KEYS = tuple(k for k, _ in STEPS)


def step_index(key: str) -> int:
    """Position of ``key`` in ``STEPS``."""
    return STEP_KEYS.index(key)


def next_step(key: str) -> str:
    """The step after ``key``, or ``key`` itself if it is already the last one."""
    return STEP_KEYS[min(step_index(key) + 1, len(STEP_KEYS) - 1)]


def prev_step(key: str) -> str:
    """The step before ``key``, or ``key`` itself if it is already the first one."""
    return STEP_KEYS[max(step_index(key) - 1, 0)]


def first_not_visited_step(step_status: dict[str, str]) -> str:
    """Where ``_load_picks`` should land after loading a file: the first (in ``STEPS`` order)
    step that is still ``not_visited``, so the breadcrumb resumes wherever the saved workflow
    left off. If every step already has some status -- an old file whose picks cover the whole
    workflow -- there is no natural "resume point", so the simplest sensible fallback is the
    first step, "overview"."""
    for key in STEP_KEYS:
        if step_status.get(key, "not_visited") == "not_visited":
            return key
    return STEP_KEYS[0]


def skipped_steps(state: PickState) -> frozenset[str]:
    """Steps this test's interpretation leaves out end to end. PC-F (no peak): the derivative
    never peaks, so there is no postclosure line (porepressure) and no pore-pressure estimate
    for the h-function's Pres term (stiffness). PC-X (uninterpretable) drops the same two
    steps. The one place the skip rule lives: a skipped step is not computed, not navigable,
    not exported, and counts as accounted for."""
    if state.postclosure_scenario.startswith(("PC-F", "PC-X")):
        return frozenset({"porepressure", "stiffness"})
    return frozenset()


def last_step(state: PickState) -> str:
    """The effective last step: where Next becomes Finish."""
    skipped = skipped_steps(state)
    return next(k for k in reversed(STEP_KEYS) if k not in skipped)


def resolve_step(state: PickState, step: str) -> str:
    """``step`` itself when it is part of this test's workflow, else the nearest earlier step
    that is (PC-F/PC-X send porepressure and stiffness to loglog)."""
    skipped = skipped_steps(state)
    i = step_index(step)
    while STEP_KEYS[i] in skipped and i > 0:
        i -= 1
    return STEP_KEYS[i]


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
    apparent_isip_method: str = ""  # "tangent" / "shutin" / "" (no shut-in pick)
    effective_isip_compliance: Optional[float] = None
    effective_isip_tangent: Optional[float] = None
    effective_isip_variable: Optional[float] = None
    contact_pressure: Optional[float] = None
    shmin_compliance: Optional[float] = None
    shmin_tangent: Optional[float] = None
    shmin_variable: Optional[float] = None
    shmin_rapid: Optional[float] = None
    shmin_liberty: Optional[float] = None
    # Comparison-only fourth Shmin estimate (URTeC-2019-123 A.8/A.9 relative stiffness) --
    # picked pressure - 75 psi (interpret.shmin_compliance). Never feeds net pressure or the
    # shared reference ISIP; see the stiffness block in compute_all.
    shmin_stiffness: Optional[float] = None
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

    # Depth-normalized (psi/ft) forms of the pressures above, each strictly its own source value
    # / TVD -- no cross-field fallback. shmin_compliance_gradient is therefore None under C-D, in
    # step with shmin_compliance, and shmin_rapid_gradient is the only one set there; the panel's
    # rapid-substitution display (ui.py's use_rapid) is a display choice, not a property of these
    # fields. See _resolve_gradients.
    apparent_isip_gradient: Optional[float] = None
    shmin_compliance_gradient: Optional[float] = None
    shmin_variable_gradient: Optional[float] = None
    shmin_tangent_gradient: Optional[float] = None
    shmin_liberty_gradient: Optional[float] = None
    shmin_rapid_gradient: Optional[float] = None
    shmin_stiffness_gradient: Optional[float] = None
    pore_pressure_gradient: Optional[float] = None
    effective_isip_compliance_gradient: Optional[float] = None
    effective_isip_tangent_gradient: Optional[float] = None
    effective_isip_variable_gradient: Optional[float] = None

    # Pick details behind the reported numbers, shown in the Expanded results window (not
    # logged): G-time of each closure pick, the Liberty anchor, and the pore-pressure line fit.
    closure_G_compliance: Optional[float] = None
    closure_G_tangent: Optional[float] = None
    closure_G_variable: Optional[float] = None
    min_dpdg_time_s: Optional[float] = None  # shut-in time at the min-dP/dG pick; None for C-X
    min_dpdg_pressure: Optional[float] = None  # BHP at the min-dP/dG pick; None for C-X
    liberty_anchor_G: Optional[float] = None
    liberty_anchor_pressure: Optional[float] = None
    liberty_anchor_time_s: Optional[float] = None  # shut-in time at the Liberty anchor G
    pore_pressure_slope: Optional[float] = None
    pore_pressure_n_points: Optional[int] = None
    loglog_slope: Optional[float] = None  # t*dP/dt log-log slope over state.loglog_window

    # arrays for plotting (not serialized)
    t_all_s: Optional[np.ndarray] = field(default=None, repr=False)
    bhp_all: Optional[np.ndarray] = field(default=None, repr=False)
    pressure_is_bhp: bool = field(default=False, repr=False)  # bhp_all holds true BHP, not surface
    # Raw surface pressure, set only when bhp_all was converted from it hydrostatically
    # (Overview overlays it on the BHP trace); None when the mapped channel is already BHP.
    p_surface_all: Optional[np.ndarray] = field(default=None, repr=False)
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

    # Masked pressure samples on the raw mapped pressure channel (see resample.detect_dropouts,
    # resample.detect_rise_excursions and the "Pressure dropouts" section in ../CLAUDE.md).
    # dropout_mask is the COMBINED mask from all sources (near-zero dropouts, returning rise
    # excursions, PickState.mask_intervals, minus PickState.keep_intervals); the field name is
    # kept for its consumers. Full-length (aligned with td samples), False before shut-in and
    # wherever nothing was masked. dropouts / rise_excursions are the resample.Dropout /
    # resample.RiseExcursion records found; n_manual_masked counts samples masked by a manual
    # interval and n_keep_restored counts auto-masked samples a keep interval restored. Every
    # consumer of raw post-shut-in pressure applies this mask by treating a masked sample as
    # missing (NaN) -- res.bhp_all itself is never mutated.
    dropout_mask: Optional[np.ndarray] = field(default=None, repr=False)
    dropouts: list = field(default_factory=list, repr=False)
    rise_excursions: list = field(default_factory=list, repr=False)
    n_manual_masked: int = 0
    n_keep_restored: int = 0

    # The effective-ISIP tangent (P vs G): derived from state.contact_G, not a stored pick --
    # see compute_all. Not serialized (DerivedResults never is).
    eff_isip_line_compliance: Optional[TangentPick] = field(default=None, repr=False)

    # Relative-stiffness plot arrays (URTeC-2019-123 A.8/A.9): stiffness_p_eff and stiffness_G
    # are aligned with EACH OTHER (same length, same selected samples), and stiffness_S is one
    # shorter than both (aligned with stiffness_p_eff[1:]/stiffness_G[1:]) -- see the stiffness
    # block in compute_all. All three equal the full resampled grid (res.resampled.p/
    # res.diagnostics.G) unless STIFFNESS_MAX_POINTS capped it, in which case they are an evenly
    # spaced (plus index 0/i_min/last) subset of it, and res.warnings carries a "decimated to
    # limit memory" line. Not serialized (DerivedResults never is).
    stiffness_p_eff: Optional[np.ndarray] = field(default=None, repr=False)
    stiffness_G: Optional[np.ndarray] = field(default=None, repr=False)
    stiffness_S: Optional[np.ndarray] = field(default=None, repr=False)

    # Compact summary of any non-1.0 unit conversion applied this compute (see
    # io_load.refresh_unit_detection / UnitDetection) -- e.g. "pressure: kpa×0.145038 (header)".
    # Empty when every mapped channel resolved to factor 1.0. Logged to dfit_log.csv's
    # units_note column.
    unit_conversion_note: str = ""

    # Three levels, most severe first (ui.issue_sections renders them in this order):
    # blockers stop navigation past Overview (see blocking_issues/step_gate_error); warnings
    # flag anything that changed or may invalidate a reported number; notes are informational.
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


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


def _resolve_gradients(state: "PickState", res: "DerivedResults") -> "DerivedResults":
    """Depth-normalize every reported pressure to psi/ft. Bails, leaving every gradient None,
    unless ``state.tvd_ft`` is set, finite, and > 0 -- nothing upstream guarantees this
    (``io_load.bhp_inputs_ready`` accepts ``tvd_ft = 0.0`` and the UI entry is free-form), so
    this guard is the only thing between the feature and a divide-by-zero. Each gradient is
    strictly its own source value / TVD -- no cross-field fallback. Bailing is never silent: it
    appends a warning naming the reason, gated on there being at least one source value to
    normalize."""
    pairs = (
        ("apparent_isip", "apparent_isip_gradient"),
        ("shmin_compliance", "shmin_compliance_gradient"),
        ("shmin_variable", "shmin_variable_gradient"),
        ("shmin_tangent", "shmin_tangent_gradient"),
        ("shmin_liberty", "shmin_liberty_gradient"),
        ("shmin_rapid", "shmin_rapid_gradient"),
        ("shmin_stiffness", "shmin_stiffness_gradient"),
        ("pore_pressure", "pore_pressure_gradient"),
        ("effective_isip_compliance", "effective_isip_compliance_gradient"),
        ("effective_isip_tangent", "effective_isip_tangent_gradient"),
        ("effective_isip_variable", "effective_isip_variable_gradient"),
    )
    # Gates both warnings below: a just-opened test has no picks yet, so there is nothing to
    # normalize and nothing worth saying. Neither is suppressed when compute_all's "Surface
    # pressure selected but density/TVD not set" has already fired -- that one is about BHP
    # reliability, these are about the gradients not being reported, and the panel stacks
    # warnings one per line. Do not collapse them.
    has_source = any(getattr(res, src_attr) is not None for src_attr, _ in pairs)
    if state.tvd_ft is None:
        if has_source:
            res.notes.append("TVD not set; gradients blank")
        return res
    # Coerce rather than crash: `_decode`'s contract is that old or foreign JSON never raises,
    # and this is the first code path that reads `tvd_ft` unconditionally, so a hand-edited or
    # foreign save carrying a string (or other non-numeric) `tvd_ft` has to reach the same
    # "no gradients" bail instead of raising out of `math.isfinite`. It reports the value it
    # got, not "not set" -- the two are different analyst-facing problems.
    try:
        tvd = float(state.tvd_ft)
    except (TypeError, ValueError, OverflowError):
        # OverflowError is not hypothetical: json.load turns a huge integer literal into a Python
        # int, and float() raises on one too large for a double (a float literal like 1e400
        # parses to inf instead and takes the isfinite path below).
        tvd = math.nan
    if not math.isfinite(tvd) or tvd <= 0:
        if has_source:
            res.notes.append(f"TVD {state.tvd_ft!r} is not a positive number; gradients blank")
        return res
    for src_attr, grad_attr in pairs:
        src = getattr(res, src_attr)
        if src is not None:
            setattr(res, grad_attr, interpret.pressure_gradient(src, tvd))
    return res


# Hard cap on the number of points fed into the stiffness construction (interpret.h_function
# builds dense n x n arrays -- ~1 GB at n=~11k). Bidirectional resampling (keeps rises as well
# as declines) can leave 10k-40k resampled points on a noisy gauge, well past what h_function
# can afford; above this cap, compute_all decimates to an even subset before building p_eff/h/S
# (see the stiffness block below) rather than resampling more coarsely -- that would move every
# other reported value too, not just this comparison-only plot.
STIFFNESS_MAX_POINTS = 2000


def _min_label(dt_s: float) -> str:
    """Whole minutes for a warning. "at 0 min" reads as "no time elapsed at all" -- true for
    anything in the first ~30 s, not just the exact instant of shut-in -- so "<1" is printed
    instead; gated on the formatted (rounded) string, since that's what's actually printed."""
    s = f"{dt_s/60:.0f}"
    return "<1" if s == "0" else s


def compute_all(state: PickState, td: TestData) -> DerivedResults:
    """Compute every derived value that the current PickState supports. Missing picks -> None.

    Side effect: this refreshes `td`'s unit-detection cache (`unit_factors`/`unit_detections`/
    `unit_warnings`, via `io_load.refresh_unit_detection`) from the state's current column
    mapping and per-channel unit overrides, before anything reads a channel through `td.column`/
    `td.pressure_surface`/`td.bhp` (including the `picks` seeders, which run compute_all first).
    It is idempotent -- rebuilt from the untouched raw df every call, so repeated calls never
    compound a conversion -- so this is safe to do unconditionally on every recompute rather than
    only on a mapping/override change.

    Also folds in `td.load_warnings` -- a load-time anomaly unrelated to unit detection, e.g. a
    `.DBS` file whose trailing padding records `load_dbs` had to truncate -- so it surfaces in
    the same warnings panel. Unlike `unit_warnings`, `load_warnings` is set once at load and
    never rebuilt here; it is just re-appended to the fresh `res.warnings` every recompute.
    """
    res = DerivedResults()
    cfg = state.channel_config()

    res.warnings.extend(td.load_warnings)

    unit_warnings = io_load.refresh_unit_detection(
        td, state.pressure_col, state.rate_col, state.volume_col,
        state.pressure_unit, state.rate_unit, state.volume_unit)
    res.warnings.extend(unit_warnings)
    notes = []
    for kind, col in (("pressure", state.pressure_col), ("rate", state.rate_col),
                      ("volume", state.volume_col)):
        det = td.unit_detections.get(col) if col else None
        if det is not None and det.factor != 1.0:
            notes.append(f"{kind}: {det.unit}×{det.factor:.6g} ({det.source})")
    res.unit_conversion_note = "; ".join(notes)

    res.blockers.extend(blocking_issues(state))
    if not state.pressure_col:
        return res

    # Full-length channels
    res.t_all_s = td.t_s
    try:
        res.bhp_all = td.bhp(cfg) if cfg.bhp_inputs_ready() else td.pressure_surface(cfg)
        res.pressure_is_bhp = cfg.bhp_inputs_ready()
    except Exception as e:  # pragma: no cover - defensive
        res.blockers.append(f"BHP computation failed: {e}")
        res.bhp_all = td.pressure_surface(cfg)
        res.pressure_is_bhp = False
    if res.pressure_is_bhp and not state.pressure_is_bhp:
        res.p_surface_all = td.pressure_surface(cfg)
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
                res.warnings.append(
                    f"Vinj: volume and rate integral disagree by {vr.disagreement_frac:.0%}")
            if res.qmax_bpm and res.qmax_bpm > 0:
                res.te_s = interpret.effective_te_seconds(res.vinj, res.qmax_bpm)
        if res.te_s is not None and not (np.isfinite(res.te_s) and res.te_s > 0):
            res.te_s = None  # a NaN/<=0 effective te must not leak into the truthy te gate below
        if res.te_s is None and shutin > start:
            dur = float(td.t_s[shutin] - td.t_s[start])
            if dur > 0:
                res.te_s = dur
                res.warnings.append("No usable rate: te = pump duration, no Vinj/qmax")

    # Apparent ISIP (needs shut-in time). The at-shut-in branch needs res.dropout_mask, so it
    # is evaluated just after the mask is built below. apparent_isip_method is set only when a
    # value results, so the log never pairs a method with a blank ISIP.
    if state.isip_tangent and res.t_shutin_s is not None and not state.isip_at_shutin:
        tg = state.isip_tangent
        res.apparent_isip = interpret.apparent_isip(tg.anchor_x, tg.anchor_y, tg.slope, res.t_shutin_s)
        res.apparent_isip_method = "tangent"

    # Pressure dropouts: detect momentary near-zero gauge glitches after shut-in, on the raw
    # mapped pressure channel (before any hydrostatic offset -- "near zero" means near zero on
    # the gauge, not on a converted BHP). Full-length and False before shut-in so every consumer
    # below (the resample block, the guard-excluded preview, the low-surface-pressure scan, and
    # picks.seed_tail_trim) applies the same mask by treating a masked sample as missing; res.
    # bhp_all itself stays raw and unmodified. See ../CLAUDE.md's "Pressure dropouts" section.
    res.dropout_mask = np.zeros(td.n, dtype=bool)
    if res.t_shutin_s is not None:
        dt_all_dropout = td.t_s - res.t_shutin_s
        post_dropout = dt_all_dropout >= 0
        if post_dropout.any():
            raw_pressure = td.column(state.pressure_col)
            dt_post = dt_all_dropout[post_dropout]
            p_post_raw = raw_pressure[post_dropout]
            auto_drop, res.dropouts = resample.detect_dropouts(dt_post, p_post_raw)
            # Analyst intervals (absolute td.t_s, inclusive). Manual masks add to the auto mask;
            # keep intervals override it. Rise excursions are detected on a copy of the raw
            # pressure with the dropouts and manual masks removed (so a masked false low cannot
            # pin the running min), except where a keep interval covers them (kept samples stay
            # live input to the rise detector). A keep then unmasks in the combined mask.
            t_post = td.t_s[post_dropout]
            manual = np.zeros(len(t_post), dtype=bool)
            keep = np.zeros(len(t_post), dtype=bool)
            n_touching = 0
            for lo, hi in state.mask_intervals:
                hit = (t_post >= lo) & (t_post <= hi)
                n_touching += bool(hit.any())
                manual |= hit
            for lo, hi in state.keep_intervals:
                keep |= (t_post >= lo) & (t_post <= hi)
            p_for_rise = np.array(p_post_raw, dtype=float)
            p_for_rise[(auto_drop | manual) & ~keep] = np.nan
            auto_rise, res.rise_excursions = resample.detect_rise_excursions(dt_post, p_for_rise)
            final = (auto_drop | auto_rise | manual) & ~keep
            res.dropout_mask[post_dropout] = final
            res.n_manual_masked = int(np.count_nonzero(manual & ~keep))
            res.n_keep_restored = int(np.count_nonzero((auto_drop | auto_rise) & keep))
            if keep.any():
                # An event a keep interval fully restored is not masked, so it is not reported as
                # masked (warning, event list, log count); the keep warning covers it instead. An
                # event's samples are the contiguous run of its own source mask containing its
                # first sample: that covers a dropout's lead-in (which precedes dt_start) and
                # stops at a rise's return sample (which is not masked).
                def _still_masked(source: np.ndarray, dt_start: float) -> bool:
                    k = int(np.searchsorted(dt_post, dt_start, side="left"))
                    if k >= len(source) or not source[k]:
                        return True   # can't locate it: keep reporting rather than go silent
                    lo = k
                    while lo > 0 and source[lo - 1]:
                        lo -= 1
                    off = np.flatnonzero(~source[k:])
                    hi = k + int(off[0]) if off.size else len(source)
                    return bool(final[lo:hi].any())
                res.dropouts = [d for d in res.dropouts if _still_masked(auto_drop, d.dt_start)]
                res.rise_excursions = [e for e in res.rise_excursions
                                       if _still_masked(auto_rise, e.dt_start)]
            if len(res.dropouts) == 1:
                d = res.dropouts[0]
                # "1 sample" not "1 samples" (the common case, a single-sample glitch). The
                # duration clause is dropped when it ROUNDS to 0 s (gate on the formatted string,
                # not the raw duration>0 -- a raw check would still print the useless "0 s" for
                # a sub-0.5 s multi-sample dip that rounds down).
                n_word = "sample" if d.n_samples == 1 else "samples"
                detail = f"{d.n_samples} {n_word}"
                duration_str = f"{d.dt_end - d.dt_start:.0f}"
                if duration_str != "0":
                    detail += f", {duration_str} s"
                detail += f", to {d.p_min:.0f} psi"
                res.warnings.insert(0,
                    f"Pressure dropout masked at {_min_label(d.dt_start)} min ({detail})")
            elif len(res.dropouts) > 1:
                first = res.dropouts[0]
                res.warnings.insert(0,
                    f"{len(res.dropouts)} pressure dropouts masked, first at "
                    f"{_min_label(first.dt_start)} min")
            # Front-inserted in reverse so they read rise, manual mask, manual keep, then the
            # dropout line.
            if res.n_keep_restored > 0:
                n_word = "sample" if res.n_keep_restored == 1 else "samples"
                res.warnings.insert(0, f"Manual keep: {res.n_keep_restored} auto-masked "
                                       f"{n_word} restored")
            if res.n_manual_masked > 0:
                ivl = "interval" if n_touching == 1 else "intervals"
                s_word = "sample" if res.n_manual_masked == 1 else "samples"
                res.warnings.insert(0, f"Manual mask: {res.n_manual_masked} {s_word} in "
                                       f"{n_touching} {ivl}")
            if len(res.rise_excursions) == 1:
                e = res.rise_excursions[0]
                res.warnings.insert(0,
                    f"Pressure rise masked at {_min_label(e.dt_start)} min "
                    f"({_min_label(e.dt_end - e.dt_start)} min, +{e.p_max - e.base:.0f} psi)")
            elif len(res.rise_excursions) > 1:
                first = res.rise_excursions[0]
                res.warnings.insert(0,
                    f"{len(res.rise_excursions)} pressure rises masked, first at "
                    f"{_min_label(first.dt_start)} min")

    if state.isip_at_shutin and res.t_shutin_s is not None and state.shutin_idx is not None:
        p_shutin = (float(res.bhp_all[state.shutin_idx])
                    if res.bhp_all is not None else float("nan"))
        if res.dropout_mask[state.shutin_idx] or not np.isfinite(p_shutin):
            res.apparent_isip = None
            res.warnings.append(
                "Apparent ISIP at shut-in: BHP missing at the shut-in sample")
        else:
            res.apparent_isip = p_shutin
            res.apparent_isip_method = "shutin"

    # Resample + diagnostics (needs te). Resample the full post-shut-in record first
    # (stop_at_guard=False, so resampled_full/G_full always span the whole record, guard or no
    # guard), then mask to the effective cutoff (interpret.resolve_tail_cut_dt, which folds in
    # the guard boundary and PickState.tail_guard_override) before diagnostics -- everything
    # downstream (effective ISIP, Shmin, log-log, pore pressure) reads only
    # res.resampled/res.diagnostics, so the trim propagates everywhere with no other changes.
    # resampled_full/G_full stay available (unmasked) so the renderer can gray out the excluded
    # tail and ui.py can convert a drag back to seconds.
    #
    # override_extends_past_guard / admitted_new_data: declared here (default False) so the
    # low-surface-pressure scan further down -- a separate, independently-gated block -- can
    # safely reference them even when this resample block never runs (e.g. no te_s yet). Set
    # for real inside the block below.
    override_extends_past_guard = False
    admitted_new_data = False
    if res.te_s and res.t_shutin_s is not None and res.bhp_all is not None:
        dt_all = td.t_s - res.t_shutin_s
        post = dt_all >= 0
        # Masked dropouts (res.dropout_mask, computed above) are treated as missing here --
        # res.bhp_all itself is never mutated. NaN is already handled by
        # resample_pressure_increment (skipped, resets any rise run), so no resampler change
        # is needed.
        p_post = res.bhp_all[post].copy()
        p_post[res.dropout_mask[post]] = np.nan
        rs_full = resample.resample_pressure_increment(dt_all[post], p_post,
                                                        step=state.resample_step,
                                                        stop_at_guard=False)
        res.resampled_full = rs_full
        res.G_full = g_time(rs_full.dt, res.te_s, state.alpha)
        # Resolved once and reused for both the guard-warning gate below and the actual masking
        # a few lines down -- avoid calling resolve_tail_cut_dt twice with different arguments
        # in the same function.
        cutoff = interpret.resolve_tail_cut_dt(state.tail_trim_dt, rs_full.guard_dt,
                                                state.tail_guard_override)
        # override_extends_past_guard: the resolved cutoff sits past the guard at all -- which
        # resolve_tail_cut_dt only ever does when state.tail_guard_override is set (see its
        # docstring). admitted_new_data: whether res.resampled actually gains any point it
        # doesn't already have in the un-overridden default -- i.e. whether the override's mask
        # (dt <= cutoff) admits anything the default mask (dt < guard_dt) didn't, which is
        # exactly the range [guard_dt, cutoff], INCLUSIVE of guard_dt itself: the bidirectional
        # +-step rule can keep a sample AT guard_dt (the excursion's own first sample) in
        # resampled_full, even though the default mask always excludes it. With the bidirectional
        # rule this is now usually True whenever the excursion moves >= step off of whatever was
        # last kept before the guard, so an override past the guard typically does admit at least
        # that one boundary sample, let alone a rising tail's continued climb. It's False only
        # when NOTHING in [guard_dt, cutoff] ever moves >= step from the last pre-guard kept
        # point -- e.g. a rise whose height clears the fixed 30-psi rise_tol (enough to fire the
        # guard) but not resample_step, when the latter is configured larger than that height.
        # Both feed the guard-warning gate right below and the low-surface-pressure scan further
        # down.
        override_extends_past_guard = (rs_full.guard_dt is not None and cutoff is not None
                                        and cutoff > rs_full.guard_dt)
        admitted_new_data = (override_extends_past_guard
                             and bool(np.any((rs_full.dt >= rs_full.guard_dt)
                                              & (rs_full.dt <= cutoff))))
        if rs_full.guard_dt is not None:
            # A gray preview of what the guard threw away, in G-time -- built from the raw (not
            # resampled) samples past guard_dt, since the resampler may keep only the excursion's
            # own first sample (a truly flat tail that never moves again after its initial jump)
            # or many more, if the excursion's own climb, or a later decline, keeps moving >= step
            # off the last pre-guard kept point -- either way the raw samples are what the preview
            # needs to show.
            # Non-finite pressures are left in on purpose: matplotlib skips NaN when drawing, so
            # filtering here would just be extra work for the same visual result. Capped at 2x
            # the G-range that was
            # actually kept BEFORE the guard fired (the last resampled_full sample strictly
            # before guard_dt), not the whole (possibly guard-spanning) record -- see
            # ../CLAUDE.md -- so a runaway crash-to-zero tail can't blow out the plot's
            # autoscale, and decimated (ceiling stride, so the result is always <= 500 -- a
            # floor stride via `n // 500` can leave up to 999) so a very long raw tail can't
            # blow out the point count.
            dt_raw, p_raw = dt_all[post], p_post
            tail_dt = dt_raw[dt_raw > rs_full.guard_dt]
            tail_p = p_raw[dt_raw > rs_full.guard_dt]
            pre_guard = rs_full.dt < rs_full.guard_dt
            cap_G = float(res.G_full[pre_guard][-1]) if pre_guard.any() else 0.0
            if len(res.G_full):
                tail_G = g_time(tail_dt, res.te_s, state.alpha)
                within = tail_G <= 2.0 * cap_G
                tail_G, tail_p = tail_G[within], tail_p[within]
                n = len(tail_G)
                if n:
                    stride = -(-n // 500)
                    res.guard_excluded_G = tail_G[::stride]
                    res.guard_excluded_p = tail_p[::stride]
            # Only worth reporting in its original form while the guard is still actually
            # binding -- i.e. the resolved cutoff hasn't been overridden past it at all
            # (cutoff <= guard_dt). Once an override DOES extend the cutoff past the guard, one
            # of two things is true: either it actually admitted at least one new point into the
            # diagnostics for real (admitted_new_data -- with the bidirectional keep rule this is
            # now the common case, since even a flat excursion's own jump usually moves >= step
            # off the last pre-guard kept point, let alone a rising tail's continued climb), in
            # which case this warning would contradict the separate "Tail trimmed" warning below
            # (which reports that later, real cutoff) and must be suppressed entirely; or it found
            # nothing new there at all (only reachable when resample_step is configured larger
            # than the excursion's own height above the last pre-guard kept point), in which case
            # silently dropping this warning would leave the analyst thinking the override worked
            # when Shmin/effective ISIP/pore pressure are still exactly the guard-clamped values
            # -- so it's replaced with an honest admission-failed warning instead (the elif below)
            # rather than just vanishing.
            # Inserted at the front (not appended) so this stays the topmost line in warn_lbl's
            # stacked display, ahead of any earlier-queued warning (density/TVD, volume
            # disagreement, ...) -- a firing, still-binding (or falsely-believed-overridden)
            # guard must never be buried in the UI. The tail-trim escape warning below (its own
            # insert(0), for a diagnostics-starving trim) runs after this in code order, so it
            # still lands frontmost of the two when both fire.
            if cutoff is not None and cutoff <= rs_full.guard_dt:
                res.warnings.insert(0,
                    f"Tail guard stopped resampling at {rs_full.guard_dt/60:.0f} min "
                    "(sustained pressure rise)")
            elif override_extends_past_guard and not admitted_new_data:
                res.warnings.insert(0,
                    f"Tail-guard override to {cutoff/60:.0f} min added no points past "
                    f"{rs_full.guard_dt/60:.0f} min; results unchanged")
        if cutoff is not None:
            # cutoff == guard_dt is exactly the "no trim narrower than the guard" case (either
            # state.tail_trim_dt is None, or an explicit trim got clamped back to guard_dt for
            # sitting past it with no override -- resolve_tail_cut_dt) -- the bidirectional
            # resampler can now keep a sample exactly at guard_dt (the first rising sample of
            # the guarded excursion, if it's a >= step move off the last pre-guard kept point),
            # so this uses strict < to exclude it, matching the guard's own historic "nothing at
            # or past guard_dt is consumed" contract. Any other cutoff is an explicit trim value
            # (whether below the guard or, with an override, past it) and keeps <= as before --
            # narrowing it to a strict < would wrongly re-admit points between the trim and the
            # guard.
            if rs_full.guard_dt is not None and cutoff == rs_full.guard_dt:
                mask = rs_full.dt < cutoff
            else:
                mask = rs_full.dt <= cutoff
            rs = resample.Resampled(dt=rs_full.dt[mask], p=rs_full.p[mask], n_raw=rs_full.n_raw,
                                    guarded_at=rs_full.guarded_at, guard_dt=rs_full.guard_dt)
        else:
            rs = rs_full
        if state.tail_trim_dt is not None:
            # A trim in effect is never silent -- it moves Shmin/pore pressure with no other
            # visible signal when it was auto-applied (picks.seed_tail_trim), and even a manual
            # drag deserves the raw-sample count. insert(0), same as the guard warning above, so
            # it can't be buried under an earlier-queued warning.
            if state.tail_trim_reason == "low_pressure":
                # The cut itself sits at a still-above-floor sample by construction (picks.
                # seed_tail_trim snaps to the last resampled sample STRICTLY BEFORE the crash,
                # side="left") -- so this must point at the crash beginning just past the cut,
                # not claim the cut is where pressure first read low.
                msg = (f"Tail auto-trimmed at {state.tail_trim_dt/60:.0f} min "
                       "(surface pressure drops below 100 psi)")
                # A guard existing at all in this record means it fired LATER than this
                # low-pressure cut (suggest_tail_trim_dt's earliest-wins rule -- low_pressure is
                # never chosen over an earlier rise_guard), so dragging to the right edge to undo
                # this crosses the guard too -- flag that consequence. Left off entirely when
                # there's no guard in this record (the common case), so the warning stays no
                # noisier than it has to be.
                if rs_full.guard_dt is not None:
                    msg += "; undoing it also overrides the tail guard"
                res.warnings.insert(0, msg)
            elif not (override_extends_past_guard and not admitted_new_data):
                # Suppressed when the override-admitted-nothing warning above already fired for
                # this exact cut (see that warning's own comment) -- pairing it with "(0 raw
                # samples excluded)" here would read as a confusing, near-contradictory two-line
                # combo about the same cut, and the honest warning already says everything this
                # one would.
                n_excluded = int(np.sum(dt_all[post] > state.tail_trim_dt))
                res.warnings.insert(0,
                    f"Tail trimmed at {state.tail_trim_dt/60:.0f} min "
                    f"({n_excluded} samples excluded)")
        res.resampled = rs
        if len(rs.p) >= 3:
            res.diagnostics = resample.diagnostics(rs, res.te_s, state.alpha)
            if len(rs.p) < 20:
                res.warnings.append(f"Only {len(rs.p)} resampled points; consider a smaller step")
        elif state.tail_trim_dt is not None:
            # A pathological trim leaves too few points to diagnose -- warn rather than let it be
            # a silent dead end (the trim tool lives on the Overview tab now, so the recovery is
            # this warning text plus dragging that tab's trim line back right, not anything on
            # this step). Inserted at the front so it stays the topmost line in warn_lbl's stacked
            # display even when other warnings already queued ahead of it -- this is the escape
            # instruction for an otherwise-blank plot.
            res.warnings.insert(0, f"Tail trim leaves {len(rs.p)} resampled point(s); move "
                                   "the Overview trim line right")

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
        m &= ~res.dropout_mask  # a masked gauge glitch is not a real low-pressure reading
        # The guard-based restriction is skipped when an active override has actually extended
        # the effective cutoff past the guard (reusing the same override_extends_past_guard
        # computed above -- not re-deriving it here) -- a sub-100-psi crash the override admits
        # into the diagnostics (a genuine further decline resumed there) must not be hidden from
        # this scan just because it happens to sit past guard_dt. The tail_trim_dt bound right
        # below still narrows the window by the explicit trim either way.
        if (res.resampled_full is not None and res.resampled_full.guard_dt is not None
                and not override_extends_past_guard):
            m &= dt_ws < res.resampled_full.guard_dt
        if state.tail_trim_dt is not None:
            m &= dt_ws <= state.tail_trim_dt
        surf_kept = surf[m]
        finite = np.isfinite(surf_kept)
        if finite.any() and float(np.min(surf_kept[finite])) < interpret.MIN_SURFACE_PRESSURE_PSI:
            res.warnings.append("Surface pressure fell below 100 psi after shut-in; BHP "
                                "unreliable there")

    # Stale-pick warning: only meaningful once a trim is actually in effect -- gating on
    # state.tail_trim_dt avoids misleadingly reporting picks as "beyond the tail trim" for a
    # reloaded save against a shorter *source* with no trim set at all. np.interp silently
    # clamps an out-of-range G to the trimmed edge, which would otherwise produce a stale-but-
    # plausible number for a pick that now lies beyond the trimmed tail.
    if state.tail_trim_dt is not None and res.diagnostics is not None and len(res.diagnostics.G):
        edge = res.diagnostics.G[-1]
        # Picks suppressed by an uninterpretable finding report nothing, so they can't be stale:
        # C-X blanks everything the min-dP/dG pick feeds (Liberty, stiffness), and
        # tangent_uninterpretable blanks everything closure_G feeds.
        min_g = None if closure_uninterpretable(state) else state.min_dpdg_G
        closure_g = None if state.tangent_uninterpretable else state.closure_G
        stale = [name for name, g in (("contact", state.contact_G), ("min dP/dG", min_g),
                                      ("closure", closure_g)) if g is not None and g > edge]
        # stiffness_pick_P is stored in pressure, not G, so it can't be compared against
        # edge -- compare against the trimmed record's lowest kept pressure instead. rs.p is no
        # longer monotonic (a sustained rise is kept too, same as a decline), so its lowest kept
        # value isn't necessarily rs.p[-1] any more -- np.nanmin finds it regardless of where it
        # sits. p_eff's tail equals rs.p past the min-dP/dG pick, so a pick below that low end no
        # longer sits on the curve.
        # A suppressed pick (stiffness_no_upturn) reports no value at all, so it can't be
        # reported stale -- there's nothing downstream for a clamp to silently corrupt.
        if (state.stiffness_pick_P is not None and not state.stiffness_no_upturn
                and not closure_uninterpretable(state)
                and res.resampled is not None
                and len(res.resampled.p)
                and state.stiffness_pick_P < np.nanmin(res.resampled.p)):
            stale.append("stiffness")
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
            res.warnings.append(f"{', '.join(stale)} pick(s) beyond the tail trim; may be stale")

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

    if (state.min_dpdg_G is not None and not closure_uninterpretable(state)
            and res.diagnostics is not None and res.resampled is not None):
        res.min_dpdg_time_s = float(np.interp(state.min_dpdg_G, res.diagnostics.G,
                                              res.resampled.dt))
        res.min_dpdg_pressure = float(np.interp(state.min_dpdg_G, res.diagnostics.G,
                                                res.resampled.p))

    # Compliance contact -> Shmin
    if state.contact_G is not None and res.diagnostics is not None:
        res.closure_G_compliance = state.contact_G
        res.contact_pressure = float(np.interp(state.contact_G, res.diagnostics.G, res.resampled.p))
        res.shmin_compliance = interpret.shmin_compliance(res.contact_pressure)
        res.closure_time_compliance_s = float(np.interp(state.contact_G, res.diagnostics.G,
                                                          res.resampled.dt))

    # Liberty-internal Shmin: the Liberty variant of the compliance method. The anchor is the
    # min-dP/dG pick for C-A (and a blank scenario); for C-B the inflection is the contact pick
    # itself -- the min pick there is only the inflection *seed* and can sit at a nearby rel-min,
    # so anchoring on it would read the wrong point. Gated on contact_G so it blanks whenever the
    # scenario's contact construction failed (e.g. the min pick fell back to the global min on a
    # record with no interior rel-min) -- the same states that blank the compliance row. Display +
    # log only: never feeds net pressure or the shared reference ISIP.
    if (state.contact_G is not None and res.diagnostics is not None
            and not state.closure_scenario.startswith(NO_CONTACT_SCENARIOS)):
        anchor_G = (state.contact_G if state.closure_scenario.startswith("C-B")
                    else state.min_dpdg_G)
        if anchor_G is not None:
            p_anchor = float(np.interp(anchor_G, res.diagnostics.G, res.resampled.p))
            res.shmin_liberty = interpret.shmin_liberty(p_anchor)
            res.liberty_anchor_G, res.liberty_anchor_pressure = anchor_G, p_anchor
            res.liberty_anchor_time_s = float(np.interp(anchor_G, res.diagnostics.G,
                                                        res.resampled.dt))

    # C-D rapid closure: Shmin ~= apparent ISIP - 175 psi (no contact pick, so no *compliance*
    # effective ISIP -- the tangent one still exists and still feeds the shared reference; this
    # is a separate field so it doesn't feed net_pressure_compliance/delta_closure, see
    # ../CLAUDE.md and the plan's decision D2).
    if state.closure_scenario.startswith("C-D") and res.apparent_isip is not None:
        res.shmin_rapid = interpret.shmin_rapid(res.apparent_isip)

    # Tangent closure -> Shmin. tangent_uninterpretable blanks this, the tangent effective ISIP,
    # and the variable method below; the pick itself stays in state.
    tangent_ok = state.closure_G is not None and not state.tangent_uninterpretable
    if tangent_ok and res.diagnostics is not None:
        res.closure_G_tangent = state.closure_G
        res.closure_pressure = float(np.interp(state.closure_G, res.diagnostics.G, res.resampled.p))
        res.shmin_tangent = interpret.shmin_tangent(res.closure_pressure)
        res.closure_time_tangent_s = float(np.interp(state.closure_G, res.diagnostics.G,
                                                       res.resampled.dt))

    # Effective ISIP (tangent method): same construction as the compliance block above, anchored
    # at state.closure_G instead of state.contact_G.
    if tangent_ok and res.diagnostics is not None and res.resampled is not None:
        dg = res.diagnostics
        idx = int(np.nanargmin(np.abs(dg.G - state.closure_G)))
        x, y, slope = interpret.tangent_from_index(dg.G, res.resampled.p, idx, half=4)
        res.effective_isip_tangent = interpret.effective_isip(x, y, slope)

    # Variable compliance method: average the raw contact/closure picks in G-time, then read
    # Shmin (variable) off the P-vs-G curve at that midpoint and build its own effective ISIP the
    # same way as the other two methods. Guarded on both picks being present.
    if (state.contact_G is not None and tangent_ok
            and res.diagnostics is not None and res.resampled is not None):
        dg = res.diagnostics
        G_var = (state.contact_G + state.closure_G) / 2.0
        res.closure_G_variable = G_var
        res.shmin_variable = float(np.interp(G_var, dg.G, res.resampled.p))
        res.closure_time_variable_s = float(np.interp(G_var, dg.G, res.resampled.dt))
        idx = int(np.nanargmin(np.abs(dg.G - G_var)))
        x, y, slope = interpret.tangent_from_index(dg.G, res.resampled.p, idx, half=4)
        res.effective_isip_variable = interpret.effective_isip(x, y, slope)

    _resolve_net_pressures(res)
    if res.shmin_compliance is not None and res.shmin_tangent is not None:
        res.delta_closure = res.shmin_compliance - res.shmin_tangent

    if (state.loglog_window is not None and res.diagnostics is not None
            and not loglog_window_suppressed(state)):
        s = interpret.loglog_window_slope(res.diagnostics.t, res.diagnostics.tdpdt,
                                          *state.loglog_window)
        res.loglog_slope = s if np.isfinite(s) else None

    # Pore pressure (postclosure). PC-F ("no peak") means the derivative never peaks, so no
    # postclosure line exists (PC-X: none can be read) -- suppress the fit even if a stale
    # pp_window pick is present.
    if (state.pp_window and res.diagnostics is not None
            and "porepressure" not in skipped_steps(state)):
        dg = res.diagnostics
        lo, hi = state.pp_window
        m = (dg.t >= lo) & (dg.t <= hi) & (dg.t > 0)
        if m.sum() >= 2:
            expo = -0.5 if state.pp_axis == "tm12" else -1.0
            x = dg.t[m] ** expo
            res.pore_pressure_slope, res.pore_pressure = interpret.pore_pressure_fit(x, dg.p[m])
            res.pore_pressure_n_points = int(m.sum())

    # Relative stiffness (URTeC-2019-123 A.8/A.9): a fourth, comparison-only Shmin estimate at
    # the upturn where the h-function-derived relative stiffness S rises off its minimum --
    # i.e. where the fracture walls come into contact. p_eff (effective pressure, paper 3.1.1)
    # is the actual resampled pressure at/after the min-dP/dG pick, and the P-vs-G tangent
    # extrapolation from that pick before it (same construction as eff_isip_line_compliance,
    # anchored at min_dpdg_G instead of contact_G). Gated on the pore-pressure estimate (the
    # h-function's Pres term) existing -- which transitively covers PC-F, see
    # skipped_steps -- and on >= 4 resampled points, the minimum this O(n^2) construction
    # needs to be meaningful. shmin_stiffness is set INSIDE this gate: a stale pick whose
    # arrays are no longer computable must report nothing.
    if (state.min_dpdg_G is not None and res.pore_pressure is not None and res.te_s
            and res.resampled is not None and res.diagnostics is not None
            and len(res.resampled.p) >= 4):
        dg = res.diagnostics
        rs = res.resampled
        i_min = int(np.nanargmin(np.abs(dg.G - state.min_dpdg_G)))
        anchor_x, anchor_y, slope = interpret.tangent_from_index(dg.G, rs.p, i_min, half=4)
        line = anchor_y + slope * (dg.G - anchor_x)
        idx = np.arange(len(rs.p))
        p_eff = np.where(idx < i_min, line, rs.p)
        n = len(p_eff)
        if n > STIFFNESS_MAX_POINTS:
            # Decimate to an even subset before the O(n^2) h-function construction -- bidirectional
            # resampling can leave far more points than this plot can afford. np.unique both sorts
            # and dedupes, so the always-included 0/i_min/last indices collapsing onto an evenly
            # spaced one (or each other, on a tiny n) costs nothing extra; the result is <=
            # STIFFNESS_MAX_POINTS + 2 (the 3 extras, minus whatever the linspace already covered).
            even = np.linspace(0, n - 1, STIFFNESS_MAX_POINTS).round().astype(int)
            sel = np.unique(np.concatenate([even, [0, i_min, n - 1]]))
            res.notes.append(f"Stiffness plot uses {len(sel)} of {n} resampled points")
        else:
            sel = np.arange(n)
        h = interpret.h_function(rs.dt[sel], p_eff[sel], res.pore_pressure, res.te_s)
        res.stiffness_p_eff = p_eff[sel]
        res.stiffness_G = dg.G[sel]
        res.stiffness_S = interpret.relative_stiffness(res.stiffness_p_eff, h)
        # stiffness_no_upturn is an explicit negative finding ("no slope change apparent") --
        # it blanks only the reported value; the pick itself is left in state so unchecking
        # restores it rather than losing it. C-X (G-function uninterpretable) blanks it the same
        # way: the curve is anchored on the min-dP/dG pick, which C-X says can't be trusted.
        if (state.stiffness_pick_P is not None and not state.stiffness_no_upturn
                and not closure_uninterpretable(state)):
            res.shmin_stiffness = interpret.shmin_compliance(state.stiffness_pick_P)

    _resolve_gradients(state, res)

    return res
