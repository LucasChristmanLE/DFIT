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
inserted), so earlier warnings could push it out of the UI's warnings[:2] display; and the
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
from tests.helpers import PRESSURE_COL, make_testdata, overview_state

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
    st = overview_state(td)
    picks.seed_overview(st, td)
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
    insert(0) bug (plain append pushed the guard's message to warnings[2], outside the UI's
    warnings[:2] display)."""
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
    outside warnings[:2] -- the UI's warn_lbl would never show it."""
    res = _seeded_with_sustained_rise_and_two_earlier_warnings()

    assert res.resampled_full.guard_dt is not None
    # Sanity: this only exercises the ordering bug if there really are >= 2 other warnings queued.
    assert len(res.warnings) >= 3
    assert any("density/TVD" in w for w in res.warnings)
    assert any("disagree" in w for w in res.warnings)

    assert any(_WARNING_SNIPPET in w for w in res.warnings[:2])


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
    st = overview_state(td)
    picks.seed_overview(st, td)
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
