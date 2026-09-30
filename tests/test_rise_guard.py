"""Tail rise guard fixes: a fixed 30-psi tolerance independent of resample_step
(``resample.RISE_GUARD_PSI``), a sustained-rise confirmation window before the guard fires --
both a minimum duration (``resample.RISE_GUARD_SUSTAIN_S``) and a minimum sample count
(``resample.RISE_GUARD_SUSTAIN_SAMPLES``, so coarse sample spacing can't satisfy the duration
alone after just two samples) -- a non-finite sample mid-run resetting the run (continuity can't
be confirmed across a dropout), and a visible gray preview of whatever raw tail the guard
excluded (``DerivedResults.guard_excluded_G``/``guard_excluded_p``) plus a frontmost warning, so
a firing guard is never silent.

Before the fixes pinned here: ``rise_tol`` defaulted to ``step`` (lowering the resample step
silently tightened the guard); the guard fired on a single over-tolerance sample; a NaN dropout
mid-run didn't reset it, letting two unrelated excursions separated by missing data bridge into
a false fire; sample count wasn't checked, so coarse (>= 60 s) sample spacing let a 2-sample
excursion satisfy the duration window alone; a firing guard's warning was appended (not
inserted), so earlier warnings could push it below the fold in the UI's stacked warnings
display; and the
preview's decimation stride (``n // 500``) could leave up to 999 points, not <= 500. Tests
called out below as "must fail against the pre-fix code" pin exactly those regressions.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from dfit_tool import picks, plots, resample
from dfit_tool.io_load import TestData as IoTestData
from dfit_tool.model import DerivedResults, PickState, compute_all
from dfit_tool.resample import (
    RISE_GUARD_PSI,
    RISE_GUARD_SUSTAIN_S,
    RISE_GUARD_SUSTAIN_SAMPLES,
    resample_pressure_increment,
)
from tests.helpers import PRESSURE_COL, make_testdata, injection_state

_WARNING_SNIPPET = "Tail guard stopped resampling"


# ------------------------------------------------------------------------------------------------
# resample_pressure_increment (Parts 1 + 2)
# ------------------------------------------------------------------------------------------------
def test_water_hammer_immunity_at_reduced_step():
    """The motivating bug: lowering resample_step below the old rise_tol=step default made brief
    water-hammer rebounds trip the guard seconds after shut-in and discard the whole falloff.

    Each rebound is constructed as `p[idx] = p[idx - 1] + delta` (not `+=`), so its height above
    the running minimum is exactly `delta` by construction -- the fixture asserts that height
    lands in the (step, RISE_GUARD_PSI) band the pre-fix bug depended on, rather than trusting a
    raw `+=` amount that the underlying decline eats into. Must fail against the pre-fix code."""
    n = 600
    dt = np.arange(n, dtype=float)
    p = np.linspace(5000.0, 500.0, n)
    step = 20.0
    rebounds = [22.0, 25.0, 23.0, 24.0]
    for k, idx in enumerate(range(20, n - 3, 80)):
        delta = rebounds[k % len(rebounds)]
        assert step < delta < RISE_GUARD_PSI  # self-check: in the band the old bug depended on
        p[idx] = p[idx - 1] + delta
        p[idx + 1] = p[idx - 1] + delta  # hold for 2 samples -- well under the 60 s sustain window

    rs = resample_pressure_increment(dt, p, step=step)

    assert rs.guard_dt is None
    assert rs.guarded_at is None
    assert rs.dt[-1] > 0.95 * dt[-1]


def test_rise_tol_decoupled_from_step():
    """rise_tol no longer defaults to step: a rebound sized (by construction, not a raw `+=`) to
    sit provably in the (step, RISE_GUARD_PSI) band comes in comfortably under the fixed 30-psi
    tolerance even with a 20-psi resample step."""
    n = 200
    dt = np.arange(n, dtype=float)
    p = np.linspace(3000.0, 1000.0, n)
    step = 20.0
    idx = 100
    delta = 25.0
    assert step < delta < RISE_GUARD_PSI  # self-check: in the band the old rise_tol=step bug hit
    p[idx] = p[idx - 1] + delta
    p[idx + 1] = p[idx - 1] + delta
    p[idx + 2] = p[idx - 1] + delta

    rs = resample_pressure_increment(dt, p, step=step)

    assert rs.guard_dt is None


def test_high_step_still_guards_with_fixed_tolerance():
    """The other direction of the decoupling: with step=200, the old rise_tol=step default would
    have set the tolerance to 200 psi and never fired on a 50-psi rise. The fixed RISE_GUARD_PSI
    (30) must still catch it."""
    n = 300
    dt = np.arange(n, dtype=float)
    p = np.linspace(4000.0, 2000.0, n)
    rise_start = 150
    p[rise_start:] = p[rise_start - 1] + 50.0  # sustained rise, well past the fixed 30 psi tolerance

    rs = resample_pressure_increment(dt, p, step=200.0)

    assert rs.guard_dt == pytest.approx(dt[rise_start])


def test_single_large_spike_immunity():
    """A brief spike, however large, must not fire the guard on its first over-tolerance sample
    -- only a sustained rise fires. Must fail against the pre-fix code, which breaks immediately
    on the first sample past running_min + rise_tol."""
    n = 300
    dt = np.arange(n, dtype=float) * 5.0  # 5 s sample spacing
    p = np.linspace(4000.0, 1000.0, n)
    spike_start = 100
    p[spike_start:spike_start + 2] += 100.0  # ~10 s spike

    rs = resample_pressure_increment(dt, p, step=30.0)

    assert rs.guard_dt is None
    assert rs.guarded_at is None
    assert rs.dt[-1] > 0.9 * dt[-1]


def test_sustained_rise_fires():
    """A rise that stays > rise_tol above the running min for >= sustain_s (and >=
    sustain_samples) fires the guard, with guard_dt pinned to the FIRST sample of the run (not
    the confirming sample) and guarded_at equal to the count of points kept before the run
    started."""
    n = 300
    dt = np.arange(n, dtype=float)
    p = np.linspace(4000.0, 2000.0, n)
    rise_start = 150
    p[rise_start:] = p[rise_start - 1] + 50.0  # flat, well above the 30-psi tolerance, to the end

    rs = resample_pressure_increment(dt, p, step=30.0)

    assert rs.guard_dt == pytest.approx(dt[rise_start])
    assert rs.guarded_at == len(rs.p)
    assert rs.dt[-1] < dt[rise_start]


def test_run_reset_on_dip_below_tolerance():
    """Two short above-tolerance excursions separated by a dip back to the running min must not
    accumulate into a fire -- each is well under sustain_s on its own. A later genuine >= 60 s
    run does fire, anchored at ITS start, not either earlier excursion."""
    n = 500
    dt = np.arange(n, dtype=float)
    p = np.linspace(5000.0, 3000.0, n)
    floor = float(p[99])  # running_min just before the first excursion

    exc1_start = 100
    p[exc1_start:exc1_start + 30] = floor + 50.0          # ~30 s excursion #1
    dip_start = exc1_start + 30
    p[dip_start:dip_start + 2] = floor                     # dip resets the run
    exc2_start = dip_start + 2
    p[exc2_start:exc2_start + 30] = floor + 50.0           # ~30 s excursion #2
    resume = exc2_start + 30
    p[resume:resume + 100] = np.linspace(floor, 2500.0, 100)  # decline resumes and continues
    rise_start = resume + 100
    p[rise_start:] = p[rise_start - 1] + 50.0              # genuine sustained rise to the end

    rs = resample_pressure_increment(dt, p, step=30.0)

    assert rs.guard_dt == pytest.approx(dt[rise_start])
    assert rs.guarded_at == len(rs.p)


def test_record_ends_mid_run_no_fire():
    """A rise that starts too close to the end of the record to reach sustain_s never fires --
    guard_dt/guarded_at stay None."""
    n = 200
    dt = np.arange(n, dtype=float)
    p = np.linspace(4000.0, 2000.0, n)
    rise_start = n - 20  # only 20 s of record left -- short of the 60 s sustain window
    p[rise_start:] = p[rise_start - 1] + 50.0

    rs = resample_pressure_increment(dt, p, step=30.0)

    assert rs.guard_dt is None
    assert rs.guarded_at is None


def test_module_constants():
    assert RISE_GUARD_PSI == 30.0
    assert RISE_GUARD_SUSTAIN_S == 60.0
    assert RISE_GUARD_SUSTAIN_SAMPLES == 5


# ------------------------------------------------------------------------------------------------
# stop_at_guard=False: guard_dt/guarded_at still fire, but resampling continues past them
# ------------------------------------------------------------------------------------------------
def test_stop_at_guard_false_still_records_guard_dt():
    """guard_dt/guarded_at get set exactly the same way regardless of stop_at_guard -- only what
    happens to the loop afterward (break vs. continue) differs."""
    n = 300
    dt = np.arange(n, dtype=float)
    p = np.linspace(4000.0, 2000.0, n)
    rise_start = 150
    p[rise_start:] = p[rise_start - 1] + 50.0  # sustained rise, held to the end

    rs_stop = resample_pressure_increment(dt, p, step=30.0, stop_at_guard=True)
    rs_continue = resample_pressure_increment(dt, p, step=30.0, stop_at_guard=False)

    assert rs_continue.guard_dt == pytest.approx(dt[rise_start])
    assert rs_continue.guard_dt == rs_stop.guard_dt
    assert rs_continue.guarded_at == rs_stop.guarded_at


def test_stop_at_guard_false_second_excursion_does_not_overwrite_guard_dt():
    """Post-review Finding 1: with stop_at_guard=False, a SECOND, later sustained excursion must
    not overwrite guard_dt/guarded_at -- they stay pinned to the FIRST qualifying run. Must fail
    against the pre-fix code, which reassigns guard_dt on every qualifying run."""
    dt = np.arange(600, dtype=float)
    p = np.linspace(4000.0, 1000.0, 600)
    p[200:280] = p[199] + 100.0  # first sustained excursion -- fires the guard
    p[400:480] = p[399] + 100.0  # second, later sustained excursion

    rs_stop = resample_pressure_increment(dt, p, step=30.0, stop_at_guard=True)
    assert rs_stop.guard_dt == pytest.approx(dt[200])

    rs_continue = resample_pressure_increment(dt, p, step=30.0, stop_at_guard=False)
    assert rs_continue.guard_dt == pytest.approx(dt[200])  # NOT dt[400]
    assert rs_continue.guarded_at == rs_stop.guarded_at


def test_stop_at_guard_false_captures_only_the_jump_when_tail_then_stays_flat():
    """A rise that fires the guard and then holds perfectly flat forever (a stuck sensor, not a
    ramp) has exactly ONE new thing to offer past guard_dt under the bidirectional (+-step) keep
    rule: the jump into the elevated level itself, since it's a >= step move off the last
    pre-guard kept point. Once flat, nothing later ever moves by >= step again, so no further
    points get kept after that -- this is still mathematically honest (no new INFORMATION exists
    past the single jump), just no longer "nothing at all" the way the old, decline-only rule
    made it. stop_at_guard=True still discards even that one point (truncated along with the
    rest of the run), so the two modes diverge by exactly one point, not zero.

    An earlier round (predating the bidirectional rule) tried to paper over the "never comes
    back" gap by appending the confirming sample as a fresh anchor, which broke the then-
    strictly-decreasing resampler's monotonicity invariant without even fixing the real-world
    case that mattered (the anchor sat at dt_full[-1], so a UI drag there still fell back to
    clearing the trim); that anchor-insertion approach stays reverted -- the jump captured here
    is a genuine, non-synthesized sample, not a fake anchor."""
    n = 600
    dt = np.arange(n, dtype=float)
    p = np.linspace(4000.0, 2000.0, n)
    p[300:] = p[299] + 250.0  # jumps once, then holds perfectly flat -- never moves again

    rs_stop = resample_pressure_increment(dt, p, step=30.0, stop_at_guard=True)
    rs_continue = resample_pressure_increment(dt, p, step=30.0, stop_at_guard=False)

    assert rs_continue.guard_dt == pytest.approx(dt[300])
    assert rs_continue.guard_dt == rs_stop.guard_dt
    assert rs_continue.guarded_at == rs_stop.guarded_at

    assert len(rs_continue.p) == len(rs_stop.p) + 1
    assert rs_continue.dt[-1] == pytest.approx(dt[300])
    assert rs_continue.p[-1] == pytest.approx(p[300])
    # Everything before that one extra point is identical between the two modes.
    np.testing.assert_array_equal(rs_continue.dt[:-1], rs_stop.dt)
    np.testing.assert_array_equal(rs_continue.p[:-1], rs_stop.p)


def test_stop_at_guard_false_resamples_past_the_guard():
    """A rise held for a while and then followed by a further decline: with stop_at_guard=True
    (the default) resampling stops dead at the rise and never sees the decline that follows. With
    stop_at_guard=False the guard still fires (same guard_dt), but under the bidirectional
    (+-step) keep rule the rise ITSELF is now captured too (a >= step move off the last pre-guard
    kept point), not just the decline that resumes afterward -- so rs_continue.p is no longer
    strictly decreasing throughout: there is exactly one upward step (the held rise being kept),
    then it holds flat (no more new points while it doesn't move), then the resumed decline is
    strictly decreasing among itself, same as ever."""
    n = 500
    dt = np.arange(n, dtype=float)
    p = np.linspace(4000.0, 2000.0, n)
    rise_start = 150
    rise_len = 80
    delta = 100.0  # comfortably clears `step` (30) off the last pre-guard kept point
    p[rise_start:rise_start + rise_len] = p[rise_start - 1] + delta  # sustained rise, then...
    resume = rise_start + rise_len
    p[resume:] = np.linspace(p[resume - 1], 500.0, n - resume)      # ...decline resumes

    rs_stop = resample_pressure_increment(dt, p, step=30.0, stop_at_guard=True)
    rs_continue = resample_pressure_increment(dt, p, step=30.0, stop_at_guard=False)

    assert rs_continue.guard_dt == pytest.approx(dt[rise_start])
    assert rs_continue.guard_dt == rs_stop.guard_dt

    assert rs_continue.dt[-1] > rs_stop.dt[-1]
    assert len(rs_continue.dt) > len(rs_stop.dt)

    # The held rise's own jump is kept, at exactly guard_dt, and it's a genuine rise (not a new
    # low) relative to the last pre-guard kept point.
    at_guard = rs_continue.dt == rs_continue.guard_dt
    assert at_guard.any()
    idx = int(np.flatnonzero(at_guard)[0])
    assert rs_continue.p[idx] > rs_continue.p[idx - 1]

    # The resumed decline (everything from `resume` onward) is still strictly decreasing among
    # itself -- the bidirectional rule doesn't change how an ordinary decline resamples.
    post_resume = rs_continue.p[rs_continue.dt >= dt[resume]]
    assert np.all(np.diff(post_resume) < 0)


# ------------------------------------------------------------------------------------------------
# non-finite samples mid-run (Part 1 fix: a dropout resets the run)
# ------------------------------------------------------------------------------------------------
def test_nan_dropout_resets_the_run():
    """Verified failure: a 2-sample +60 psi spike at dt=2000, 98 s of NaN, then one more +60
    sample at dt=2100 bridged into a single "sustained" run under the pre-fix code (which just
    `continue`s through non-finite samples without touching the run), firing with guard_dt=2000
    and truncating half the record. A dropout can't confirm continuity across itself -- it must
    reset the run instead. Must fail against the pre-fix code."""
    n = 2200
    dt = np.arange(n, dtype=float)
    p = np.linspace(5000.0, 3000.0, n)
    spike_start = 2000
    p[spike_start:spike_start + 2] = p[spike_start - 1] + 60.0   # 2-sample spike at dt=2000..2001
    nan_start = spike_start + 2
    nan_len = 98
    p[nan_start:nan_start + nan_len] = np.nan                     # 98 s dropout
    p[nan_start + nan_len] = p[spike_start - 1] + 60.0            # one more +60 sample, at dt=2100

    rs = resample_pressure_increment(dt, p, step=30.0)

    assert rs.guard_dt is None


def test_continuous_run_across_same_span_fires():
    """Same span as the dropout case above, but with the gap filled by continuous above-
    tolerance samples instead of NaN -- a genuine sustained run over that time, which must
    fire (confirms the reset in the test above isn't just suppressing every fire)."""
    n = 2200
    dt = np.arange(n, dtype=float)
    p = np.linspace(5000.0, 3000.0, n)
    spike_start = 2000
    p[spike_start:2101] = p[spike_start - 1] + 60.0  # continuous run, dt 2000..2100

    rs = resample_pressure_increment(dt, p, step=30.0)

    assert rs.guard_dt == pytest.approx(float(dt[spike_start]))


def test_nan_outside_a_run_still_just_skipped():
    """A NaN sample that isn't part of any candidate run keeps today's behavior -- it's simply
    skipped, not treated as a reset event that affects anything (there's no run to reset)."""
    n = 300
    dt = np.arange(n, dtype=float)
    p = np.linspace(4000.0, 2000.0, n)
    p[150] = np.nan  # an isolated dropout, well clear of any excursion

    rs = resample_pressure_increment(dt, p, step=30.0)

    assert rs.guard_dt is None
    assert 150.0 not in rs.dt  # the NaN sample itself was never kept


# ------------------------------------------------------------------------------------------------
# minimum sample count (Part 2 fix: duration alone isn't enough at coarse spacing)
# ------------------------------------------------------------------------------------------------
def test_sustain_requires_min_sample_count_at_coarse_spacing():
    """At 60 s+ sample spacing, duration alone reaches sustain_s after just the second sample --
    RISE_GUARD_SUSTAIN_SAMPLES (5) additionally requires the run to actually contain that many
    above-tolerance samples before firing. Must fail against the pre-fix code (duration-only)."""
    dt = np.arange(0, 600, 60, dtype=float)  # 1-minute spacing
    p = np.linspace(4000.0, 3000.0, len(dt))
    rise_start = 5
    # A 2-sample excursion spans exactly one 60 s gap -- duration condition alone would fire.
    p[rise_start:rise_start + 2] = p[rise_start - 1] + 100.0
    p[rise_start + 2:] = np.linspace(p[rise_start - 1] - 10.0, 2500.0, len(dt) - rise_start - 2)

    rs = resample_pressure_increment(dt, p, step=30.0)

    assert rs.guard_dt is None


def test_sustain_fires_with_enough_samples_at_coarse_spacing():
    """A 1-minute-spaced run of >= 5 above-tolerance samples, spanning >= 60 s, still fires."""
    dt = np.arange(0, 600, 60, dtype=float)  # 1-minute spacing
    p = np.linspace(4000.0, 3000.0, len(dt))
    rise_start = 5
    p[rise_start:] = p[rise_start - 1] + 100.0  # 5 samples to the end, spanning 240 s

    rs = resample_pressure_increment(dt, p, step=30.0)

    assert rs.guard_dt == pytest.approx(dt[rise_start])


# ------------------------------------------------------------------------------------------------
# compute_all (Part 3): guard_excluded_G/p + warning
# ------------------------------------------------------------------------------------------------
def _seeded_with_sustained_rise():
    """Post-shut-in record: a long decline (400 kept-worthy samples, so the excluded-tail
    preview's dt window is wide enough to survive the 2x-G cap for ~900 raw samples -- enough to
    actually exercise decimation to <= 500) followed by a sustained +100 psi rise to the end."""
    start_idx, shutin_idx = 50, 100
    decline_len = 400
    rise_len = 1000
    n = shutin_idx + decline_len + rise_len
    t_s = np.arange(n, dtype=float)
    rate = np.zeros(n)
    rate[start_idx:shutin_idx] = 5.0

    pressure = np.full(n, 2000.0)
    pressure[start_idx:shutin_idx] = np.linspace(2000.0, 5000.0, shutin_idx - start_idx)
    pressure[shutin_idx:shutin_idx + decline_len] = np.linspace(5000.0, 1500.0, decline_len)
    rise_idx = shutin_idx + decline_len
    pressure[rise_idx:] = pressure[rise_idx - 1] + 100.0

    df = pd.DataFrame({PRESSURE_COL: pressure, "RATE": rate})
    td = IoTestData(path="<synthetic>", df=df, datetime_col="DATETIME", t_s=t_s,
                    columns=list(df.columns))
    st = PickState(pressure_col=PRESSURE_COL, rate_col="RATE", start_idx=start_idx,
                   shutin_idx=shutin_idx)
    res = compute_all(st, td)
    return td, st, res


def test_compute_all_populates_guard_excluded_preview_and_warns():
    td, st, res = _seeded_with_sustained_rise()

    # Sanity: confirm the guard actually fired, so this test exercises what it claims to.
    assert res.resampled_full.guard_dt is not None

    assert any(_WARNING_SNIPPET in w for w in res.warnings)
    assert res.guard_excluded_G is not None and len(res.guard_excluded_G) > 0
    assert res.guard_excluded_p is not None and len(res.guard_excluded_p) == len(res.guard_excluded_G)
    tiny_eps = 1e-6
    assert float(np.nanmax(res.guard_excluded_G)) <= 2.0 * float(res.G_full[-1]) + tiny_eps
    # Non-vacuous: the raw survivor count here is well over 500 (~900), so this actually
    # exercises decimation rather than just happening to land under the cap already.
    assert 1 < len(res.guard_excluded_G) <= 500


def test_compute_all_no_rise_no_preview_no_warning():
    td = make_testdata()
    st = injection_state(td)
    picks.seed_injection(st, td)
    res = compute_all(st, td)
    picks.seed_isip(st, td, res)
    res = compute_all(st, td)

    assert res.resampled_full.guard_dt is None
    assert not any(_WARNING_SNIPPET in w for w in res.warnings)
    assert res.guard_excluded_G is None
    assert res.guard_excluded_p is None


def _seeded_with_sustained_rise_and_two_earlier_warnings():
    """Same shape as `_seeded_with_sustained_rise`, plus a volume channel that deliberately
    disagrees with the rate integral by > 5% -- alongside the density/TVD warning that fires by
    default (pressure_is_bhp defaults False with no density/tvd set), this queues TWO warnings
    ahead of the guard's in compute_all's append order, which is what exposed the append-vs-
    insert(0) bug (plain append pushed the guard's message to warnings[2], below the top of the
    UI's stacked warnings display)."""
    start_idx, shutin_idx = 50, 100
    decline_len = 200
    rise_len = 300
    n = shutin_idx + decline_len + rise_len
    t_s = np.arange(n, dtype=float)
    rate = np.zeros(n)
    rate[start_idx:shutin_idx] = 5.0
    volume = np.zeros(n)
    volume[start_idx:shutin_idx] = np.linspace(0.0, 1000.0, shutin_idx - start_idx)
    volume[shutin_idx:] = 1000.0  # flat post-shut-in; delta (1000 bbl) wildly disagrees with the
                                   # rate integral (~4 bbl), forcing the disagreement warning

    pressure = np.full(n, 2000.0)
    pressure[start_idx:shutin_idx] = np.linspace(2000.0, 5000.0, shutin_idx - start_idx)
    pressure[shutin_idx:shutin_idx + decline_len] = np.linspace(5000.0, 1500.0, decline_len)
    rise_idx = shutin_idx + decline_len
    pressure[rise_idx:] = pressure[rise_idx - 1] + 100.0

    df = pd.DataFrame({PRESSURE_COL: pressure, "RATE": rate, "VOLUME": volume})
    td = IoTestData(path="<synthetic>", df=df, datetime_col="DATETIME", t_s=t_s,
                    columns=list(df.columns))
    st = PickState(pressure_col=PRESSURE_COL, rate_col="RATE", volume_col="VOLUME",
                   start_idx=start_idx, shutin_idx=shutin_idx)
    res = compute_all(st, td)
    return res


def test_guard_warning_survives_two_earlier_warnings():
    """Must fail against the pre-fix code: with two warnings already queued ahead of it (density/
    TVD, volume disagreement), a plain `append` for the guard's own warning lands it at index 2,
    below the top of the UI's stacked warn_lbl display -- insert(0) keeps it topmost instead."""
    res = _seeded_with_sustained_rise_and_two_earlier_warnings()

    assert res.resampled_full.guard_dt is not None
    # Sanity: this only exercises the ordering bug if there really are >= 2 other warnings queued.
    assert len(res.warnings) >= 3
    assert any("density/TVD" in w for w in res.warnings)
    assert any("disagree" in w for w in res.warnings)

    assert any(_WARNING_SNIPPET in w for w in res.warnings[:2])


# ------------------------------------------------------------------------------------------------
# compute_all (post-review Finding 3, corrected by a later round's Finding 1): the guard-stopped-
# resampling warning is suppressed only once an override actually admits new resampled data past
# the guard; the separate "Tail trimmed" warning covers that case instead, with the later, real
# cutoff. When an override is requested but admits NOTHING (no sample in [guard_dt, cutoff] clears
# ±step off the last kept value -- the tests below force this with a large resample_step -- so
# resampled_full keeps zero points at or past guard_dt regardless of how far past it the cutoff
# reaches), the original warning is replaced by a distinct, honest one instead of just vanishing
# -- the earlier round's version of this test asserted the plain-vanishing behavior, which turned
# out to be misleading (it made an ineffective override look like it had worked); see
# test_guard_warning_honest_when_override_admits_nothing below, which replaces it.
# ------------------------------------------------------------------------------------------------
_TRIMMED_SNIPPET = "Tail trimmed"
_OVERRIDE_NO_DATA_SNIPPET = "Tail-guard override requested to"


def test_guard_warning_present_when_guard_fires_with_no_trim_or_override():
    td, st, res = _seeded_with_sustained_rise()
    assert res.resampled_full.guard_dt is not None
    assert st.tail_trim_dt is None
    assert st.tail_guard_override is False

    assert any(_WARNING_SNIPPET in w for w in res.warnings)


def test_guard_warning_present_when_trim_is_stale_and_not_overridden():
    """A trim sitting past guard_dt without the override flag is presumed stale and clamped back
    to guard_dt (interpret.resolve_tail_cut_dt) -- the guard is still binding, so its warning
    must still fire, same as today."""
    td, st, res = _seeded_with_sustained_rise()
    guard_dt = res.resampled_full.guard_dt
    assert guard_dt is not None

    st.tail_trim_dt = guard_dt + 3600.0  # stale: past the guard, no override
    assert st.tail_guard_override is False
    res2 = compute_all(st, td)

    assert any(_WARNING_SNIPPET in w for w in res2.warnings)


def test_guard_warning_honest_when_override_admits_nothing():
    """Finding 1 (this round): overriding past a guard on a tail that never resumes a decline
    admits NOTHING new into the diagnostics -- res.resampled/res.diagnostics stay numerically
    identical to the un-overridden default. The original guard-stopped-resampling warning is
    still suppressed (it would otherwise misleadingly claim the guard is still binding when the
    analyst explicitly asked to override it), but it must not just silently vanish either -- that
    left the earlier round's behavior claiming, by omission, that the override worked. It's
    replaced by a distinct, honest warning naming both the requested cutoff and the guard's
    original (still-in-effect) one. The plain "Tail trimmed ... (0 raw samples excluded)" message
    is ALSO suppressed here (not just the guard one) -- showing it right next to the honest
    warning would read as a confusing, near-contradictory pair about the same cut.

    Under the bidirectional (+-step) keep rule, a fired guard's own excursion is now generally
    keepable too -- the excursion's first sample is itself a >= step move off the last pre-guard
    kept point whenever its height clears both the fixed 30-psi rise_tol AND resample_step, which
    default to the same 30 psi. To get a fixture that genuinely admits NOTHING (not even that one
    boundary sample), this test bumps resample_step well above the fixture's fixed +100 psi rise
    -- resample_step doesn't affect guard detection at all (that's still the fixed RISE_GUARD_PSI),
    so guard_dt is unaffected, but nothing in the record ever again moves >= 300 psi from its last
    kept point, so (post-revert) ``resampled_full`` correctly keeps NOTHING new past ``guard_dt``.
    That's fine: the override (ui.py's raw-sample snap) sets an explicit cutoff past the guard
    directly against the raw record, and model.compute_all's masking (``rs_full.dt <= cutoff``)
    doesn't require the cutoff to land on an existing resampled point, so an arbitrary dt past the
    guard exercises this exactly the same way."""
    td, st, _ = _seeded_with_sustained_rise()
    st.resample_step = 300.0  # comfortably above the fixture's own +100 psi rise (see docstring)
    baseline = compute_all(st, td)  # no trim/override -- the guard-clamped default
    guard_dt = baseline.resampled_full.guard_dt
    assert guard_dt is not None
    assert not np.any(baseline.resampled_full.dt >= guard_dt)  # sanity: truly nothing admitted

    st.tail_trim_dt = guard_dt + 100.0  # a deliberate override cutoff past the guard
    st.tail_guard_override = True
    res2 = compute_all(st, td)

    # No data was actually admitted: the regression this round's Finding 1 caught.
    assert len(res2.resampled.dt) == len(baseline.resampled.dt)
    np.testing.assert_array_equal(res2.resampled.dt, baseline.resampled.dt)
    np.testing.assert_array_equal(res2.resampled.p, baseline.resampled.p)
    assert len(res2.diagnostics.G) == len(baseline.diagnostics.G)
    np.testing.assert_array_equal(res2.diagnostics.G, baseline.diagnostics.G)

    assert not any(_WARNING_SNIPPET in w for w in res2.warnings)      # superseded, not shown
    assert not any(_TRIMMED_SNIPPET in w for w in res2.warnings)      # would be a confusing pair
    assert any(_OVERRIDE_NO_DATA_SNIPPET in w for w in res2.warnings)
    honest = next(w for w in res2.warnings if _OVERRIDE_NO_DATA_SNIPPET in w)
    assert f"{st.tail_trim_dt/60:.0f} min" in honest
    assert f"{guard_dt/60:.0f} min" in honest


def test_guard_warning_trimmed_present_and_data_changes_when_override_admits_real_data():
    """Sibling of the test above: when the tail DOES resume a decline past the guard (a genuine
    further decline, not a permanently-elevated one), an override with a cutoff that reaches into
    that decline actually changes res.resampled/res.diagnostics, the original guard warning stays
    suppressed (unchanged from the prior round), and the honest "admits nothing" warning above
    must NOT appear -- this is the case it exists to distinguish from."""
    td, st, res = _seeded_with_guard_then_long_further_decline()
    guard_dt = res.resampled_full.guard_dt
    assert guard_dt is not None
    past_guard = res.resampled_full.dt[res.resampled_full.dt > guard_dt]
    assert len(past_guard) > 0  # sanity: real data really is admitted here

    baseline = compute_all(st, td)  # no trim/override -- the guard-clamped default

    st.tail_trim_dt = float(past_guard[-1])  # override reaching well into the further decline
    st.tail_guard_override = True
    res2 = compute_all(st, td)

    assert len(res2.resampled.dt) > len(baseline.resampled.dt)
    assert res2.resampled.dt[-1] > guard_dt

    assert not any(_WARNING_SNIPPET in w for w in res2.warnings)
    assert not any(_OVERRIDE_NO_DATA_SNIPPET in w for w in res2.warnings)
    assert any(_TRIMMED_SNIPPET in w for w in res2.warnings)
    assert any(f"{st.tail_trim_dt/60:.0f} min" in w for w in res2.warnings if
               _TRIMMED_SNIPPET in w)


# ------------------------------------------------------------------------------------------------
# compute_all (post-review Finding 4): the guard-excluded preview's 2x-G cap stays anchored to
# the G-range actually kept BEFORE the guard fired, not to G_full[-1] -- which, now that
# stop_at_guard=False can keep points well past the guard, can span much further than what the
# guard originally excluded.
# ------------------------------------------------------------------------------------------------
def _seeded_with_guard_then_long_further_decline():
    """Guard fires partway through a short decline, but a long further decline resumes
    afterward -- G_full[-1] then spans far more G-time than what was kept before the guard
    fired, which is exactly the scenario that would loosen a G_full[-1]-anchored cap."""
    start_idx, shutin_idx = 50, 100
    decline1_len = 200
    rise_len = 80
    decline2_len = 2000
    n = shutin_idx + decline1_len + rise_len + decline2_len
    t_s = np.arange(n, dtype=float)
    rate = np.zeros(n)
    rate[start_idx:shutin_idx] = 5.0

    pressure = np.full(n, 1500.0)
    pressure[start_idx:shutin_idx] = np.linspace(2000.0, 5000.0, shutin_idx - start_idx)
    pressure[shutin_idx:shutin_idx + decline1_len] = np.linspace(5000.0, 4000.0, decline1_len)
    rise_idx = shutin_idx + decline1_len
    pressure[rise_idx:rise_idx + rise_len] = pressure[rise_idx - 1] + 100.0
    decline2_start = rise_idx + rise_len
    pressure[decline2_start:] = np.linspace(pressure[decline2_start - 1], 10.0, decline2_len)

    df = pd.DataFrame({PRESSURE_COL: pressure, "RATE": rate})
    td = IoTestData(path="<synthetic>", df=df, datetime_col="DATETIME", t_s=t_s,
                    columns=list(df.columns))
    st = PickState(pressure_col=PRESSURE_COL, rate_col="RATE", start_idx=start_idx,
                   shutin_idx=shutin_idx)
    res = compute_all(st, td)
    return td, st, res


def test_guard_excluded_preview_cap_anchored_to_pre_guard_g_range():
    """Must fail against a G_full[-1]-anchored cap: with stop_at_guard=False now keeping many
    points well past guard_dt (the long further decline), G_full[-1] sits far beyond the G-range
    that was actually kept before the guard fired. The preview's 2x cap must stay anchored to
    that earlier, tighter range."""
    td, st, res = _seeded_with_guard_then_long_further_decline()
    rf = res.resampled_full
    assert rf.guard_dt is not None
    pre_guard = rf.dt < rf.guard_dt
    assert pre_guard.any()
    cap_G_correct = float(res.G_full[pre_guard][-1])
    loose_cap_G = float(res.G_full[-1])
    # Sanity: this fixture actually distinguishes the two -- the loose cap is measurably larger.
    assert loose_cap_G > 2.0 * cap_G_correct

    assert res.guard_excluded_G is not None and len(res.guard_excluded_G) > 0
    tiny_eps = 1e-6
    assert float(np.nanmax(res.guard_excluded_G)) <= 2.0 * cap_G_correct + tiny_eps


# ------------------------------------------------------------------------------------------------
# render_gfunction: gray "guard_excluded" preview artist
# ------------------------------------------------------------------------------------------------
def _gids(ax):
    return {ln.get_gid() for ln in ax.get_lines() if ln.get_gid()}


def _gid(ax, gid):
    return next(ln for ln in ax.get_lines() if ln.get_gid() == gid)


def test_render_gfunction_draws_guard_excluded_when_guard_fired():
    td, st, res = _seeded_with_sustained_rise()
    assert res.diagnostics is not None
    fig, ax = plt.subplots()
    plots.render_gfunction(ax, td, st, res)
    assert "guard_excluded" in _gids(ax)
    plt.close(fig)


def test_render_gfunction_no_guard_excluded_when_guard_did_not_fire():
    td = make_testdata()
    st = injection_state(td)
    picks.seed_injection(st, td)
    res = compute_all(st, td)
    picks.seed_isip(st, td, res)
    res = compute_all(st, td)
    assert res.diagnostics is not None
    fig, ax = plt.subplots()
    plots.render_gfunction(ax, td, st, res)
    assert "guard_excluded" not in _gids(ax)
    plt.close(fig)


def test_render_gfunction_recovery_path_draws_guard_excluded_preview():
    """A guard fire severe enough to leave <3 kept points (no trim needed) reaches the same
    diagnostics-None recovery path as a pathological trim -- the excluded-tail preview must not
    go missing just because there weren't enough points left to diagnose. Built directly (not via
    compute_all) since the point is to pin the renderer's recovery-path branch in isolation, the
    same way test_render_constructions.py's own recovery-path test does."""
    dt_full = np.array([0.0, 10.0])
    p_full = np.array([5000.0, 4970.0])
    G_full = np.array([0.0, 1.0])
    guard_G = np.array([1.2, 1.3, 1.5])
    guard_p = np.array([5050.0, 5060.0, 5055.0])
    res = DerivedResults(
        diagnostics=None, resampled=None,
        resampled_full=resample.Resampled(dt=dt_full, p=p_full, n_raw=5, guard_dt=11.0, guarded_at=2),
        G_full=G_full, guard_excluded_G=guard_G, guard_excluded_p=guard_p)
    st = PickState()
    fig, ax = plt.subplots()
    plots.render_gfunction(ax, None, st, res)

    preview = _gid(ax, "guard_excluded")
    assert np.allclose(preview.get_xdata(), guard_G)
    assert np.allclose(preview.get_ydata(), guard_p)
    plt.close(fig)
