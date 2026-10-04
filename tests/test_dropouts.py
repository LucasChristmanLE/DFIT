"""Momentary near-zero pressure dropout masking (see ../CLAUDE.md's "Pressure dropouts" section).

A brief gauge dropout -- pressure reads near zero for a few seconds, then returns -- otherwise
pins `resample.resample_pressure_increment`'s `running_min` at the dip and fires the rise guard
on the recovery, starving every downstream diagnostic. `resample.detect_dropouts` finds and masks
these automatically; `model.compute_all` wires the mask into the resample block, the guard-
excluded preview, the low-surface-pressure scan, and the dropout warning; `picks.seed_tail_trim`
treats masked samples as missing so a dropout never triggers the "low_pressure" auto-trim;
`plots.render_overview`/`render_isip` draw the masked samples as their own markers and break the
main trace across them.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from matplotlib.figure import Figure

from dfit_tool import picks, plots, resample
from dfit_tool.io_load import TestData as IoTestData
from dfit_tool.model import PickState, compute_all
from dfit_tool.resample import (
    DROPOUT_FLOOR_FRAC,
    DROPOUT_LEADIN_MATCH_PSI,
    DROPOUT_MAX_S,
    DROPOUT_MIN_REF_PSI,
    RISE_GUARD_PSI,
    RISE_GUARD_SUSTAIN_S,
    RISE_GUARD_SUSTAIN_SAMPLES,
    detect_dropouts,
)
from tests.helpers import PRESSURE_COL


def test_module_constants():
    assert DROPOUT_FLOOR_FRAC == 0.10
    assert DROPOUT_MIN_REF_PSI == 100.0
    assert DROPOUT_MAX_S == 600.0
    assert DROPOUT_LEADIN_MATCH_PSI == 150.0


# --------------------------------------------------------------------------------------------------
# detect_dropouts -- synthetic arrays
# --------------------------------------------------------------------------------------------------
def test_encore_shaped_masks_leadin_and_dip():
    """Shape of the motivating Encore record (see the plan's Context table): a flat baseline, a
    slow lead-in slide toward zero, a momentary near-zero dip, and a full recovery. The lead-in
    walk stops at the sample closest to the flat baseline (2978.5, 6.5 psi from the ~2985 psi
    recovery level) and that stopping sample is NOT itself masked -- only the samples below it
    (from 2953 down) plus the dip itself."""
    dt = np.array([0, 1, 2, 657, 660, 663, 666, 669,
                   672, 675, 678, 681, 684, 687, 690, 693,
                   696, 699, 702], dtype=float)
    p = np.array([2988, 2988, 2988, 2978.5, 2953, 2933, 2914, 2866.5,
                  63.5, 15, 14, 13, 14, 15, 14, 13,
                  2985, 2970, 2986], dtype=float)

    mask, events = detect_dropouts(dt, p)

    assert not mask[:3].any()
    assert mask[3] == False       # 2978.5 -- the lead-in walk's stopping sample -- stays unmasked
    assert mask[4:16].all()       # 2953...2866.5 (lead-in) + 63.5...13 (dip)
    assert not mask[16:].any()

    assert len(events) == 1
    ev = events[0]
    assert ev.dt_start == pytest.approx(672.0)
    assert ev.dt_end == pytest.approx(693.0)
    assert ev.n_samples == 8
    assert ev.p_min == pytest.approx(13.0)


def test_legit_big_single_sample_drops_never_recover_nothing_masked():
    """Big single-sample drops (5090->4570, 4423->4111) that never come back up must not be
    mistaken for dropouts -- neither reaches DROPOUT_FLOOR_FRAC (10%) of the level before it, so
    onset never even triggers."""
    dt = np.arange(50, dtype=float)
    p = np.concatenate([np.full(10, 5090.0), np.full(15, 4570.0),
                        np.full(10, 4423.0), np.full(15, 4111.0)])

    mask, events = detect_dropouts(dt, p)

    assert not mask.any()
    assert events == []


def test_crash_to_zero_never_recovers_nothing_masked():
    """A monotone crash to ~0 psi that never comes back is the low-pressure tail trim's job, not
    this detector's -- no recovery before the record ends means "not a dropout"."""
    dt = np.arange(50, dtype=float)
    p = np.concatenate([np.linspace(5000.0, 3000.0, 30), np.zeros(20)])

    mask, events = detect_dropouts(dt, p)

    assert not mask.any()
    assert events == []


def test_sustained_dip_60s_5samples_not_masked():
    """A dip that meets the rise guard's own sustain definition (>= 60 s AND >= 5 samples) is a
    genuine event (a dead channel, a real quiet period), not a glitch -- left alone."""
    dt = np.array([0, 15, 30, 45, 60, 75, 90, 105, 120, 135], dtype=float)
    p = np.array([3000, 3000, 5, 4, 3, 4, 5, 3005, 2995, 3000], dtype=float)

    mask, events = detect_dropouts(dt, p)

    assert not mask.any()
    assert events == []


def test_partial_dip_above_floor_not_masked():
    """A dip to 60% of the pre-dip level never reaches DROPOUT_FLOOR_FRAC (10%) -- above the
    floor, not a dropout, even though it fully recovers."""
    dt = np.array([0, 10, 20, 30, 40], dtype=float)
    p = np.array([3000, 3000, 1800, 1800, 3000], dtype=float)

    mask, events = detect_dropouts(dt, p)

    assert not mask.any()
    assert events == []


def test_water_hammer_ringing_not_masked():
    """A +-300 psi oscillation around a high baseline never comes close to the 10% floor --
    water-hammer ringing must never be mistaken for a dropout."""
    dt = np.arange(0, 31, 5, dtype=float)
    p = np.array([5000, 5300, 4700, 5300, 4700, 5300, 5000], dtype=float)

    mask, events = detect_dropouts(dt, p)

    assert not mask.any()
    assert events == []


def test_two_separate_dropouts_both_masked():
    dt = np.arange(40, dtype=float)
    p = np.full(40, 3000.0)
    p[5] = 30.0
    p[20] = 25.0

    mask, events = detect_dropouts(dt, p)

    assert mask[5] and mask[20]
    assert int(mask.sum()) == 2
    assert len(events) == 2
    assert events[0].dt_start == pytest.approx(5.0)
    assert events[0].p_min == pytest.approx(30.0)
    assert events[1].dt_start == pytest.approx(20.0)
    assert events[1].p_min == pytest.approx(25.0)


def test_nan_inside_dip_same_decision_as_without_it():
    """A NaN sample inside the dip run must not change the detector's decision -- it's skipped,
    never part of a decision, per the algorithm's own contract."""
    dt = np.array([0, 1, 2, 657, 660, 663, 666, 669,
                   672, 675, 678, 681, 684, 687, 690, 693,
                   696, 699, 702], dtype=float)
    p = np.array([2988, 2988, 2988, 2978.5, 2953, 2933, 2914, 2866.5,
                  63.5, 15, 14, 13, 14, 15, 14, 13,
                  2985, 2970, 2986], dtype=float)
    p_nan = p.copy()
    p_nan[11] = np.nan  # inside the dip run, not the onset or last-dip sample

    mask, events = detect_dropouts(dt, p)
    mask_nan, events_nan = detect_dropouts(dt, p_nan)

    assert len(events) == len(events_nan) == 1
    assert events[0].dt_start == events_nan[0].dt_start
    assert events[0].dt_end == events_nan[0].dt_end
    unaffected = np.ones(len(p), dtype=bool)
    unaffected[11] = False
    np.testing.assert_array_equal(mask[unaffected], mask_nan[unaffected])


def test_clean_monotone_decline_empty_mask():
    dt = np.arange(50, dtype=float)
    p = np.linspace(5000.0, 1000.0, 50)

    mask, events = detect_dropouts(dt, p)

    assert not mask.any()
    assert events == []


def test_recovery_through_intermediate_ramp_sample_fully_masked():
    """The recovery can pass through an intermediate ramp sample (Encore's 1530 psi at dt=694)
    that is still well below the recovery level -- a single recover_tol threshold off the
    original pre-dip ref (not a separate "still near zero" band) correctly keeps the run open
    through it instead of prematurely calling it recovered."""
    dt = np.array([0, 1, 2, 657, 660, 663, 666, 669,
                   672, 675, 694, 696, 699], dtype=float)
    p = np.array([2988, 2988, 2988, 2978.5, 2953, 2933, 2914, 2866.5,
                  63.5, 13, 1530, 2985, 2970], dtype=float)

    mask, events = detect_dropouts(dt, p)

    assert len(events) == 1
    ev = events[0]
    assert ev.dt_end == pytest.approx(694.0)   # the ramp sample, not the earlier 675 sample
    assert ev.p_min == pytest.approx(13.0)
    assert mask[10]  # the 1530 ramp sample (index of dt=694) is masked


def test_dead_channel_min_ref_guard_no_events():
    """A channel already reading under DROPOUT_MIN_REF_PSI (100 psi), with noise dipping toward
    zero, must never trigger onset -- without this guard, noise on an already-dead channel
    re-triggers onset endlessly."""
    dt = np.arange(10, dtype=float)
    p = np.array([50, 52, 48, 5, 49, 3, 51, 47, 4, 50], dtype=float)

    mask, events = detect_dropouts(dt, p)

    assert not mask.any()
    assert events == []


def test_step_change_straddling_glitch_leadin_rejected():
    """A real step change (916 -> 2707 psi) that happens to straddle a one-sample glitch must not
    have its whole leading plateau swept into the mask -- the backward lead-in walk runs into the
    DROPOUT_LEADIN_MATCH_PSI-independent time cap (sustain_s) before it can reach a level close
    to the recovery, so only the glitch itself is masked."""
    dt = np.arange(141, dtype=float)
    p = np.concatenate([np.full(70, 916.0), [5.0], np.full(70, 2707.0)])

    mask, events = detect_dropouts(dt, p)

    assert int(mask.sum()) == 1
    assert mask[70]                 # only the glitch
    assert not mask[:70].any()      # the 916 psi plateau is untouched
    assert len(events) == 1
    assert events[0].n_samples == 1
    assert events[0].p_min == pytest.approx(5.0)


def test_single_sample_sentinel_spike_masked():
    dt = np.arange(11, dtype=float)
    p = np.full(11, 3000.0)
    p[5] = -9999.0

    mask, events = detect_dropouts(dt, p)

    assert mask[5]
    assert int(mask.sum()) == 1
    assert len(events) == 1
    assert events[0].p_min == pytest.approx(-9999.0)


def test_stuck_negative_33s_masked_under_60s():
    dt = np.array([0, 11, 22, 33, 44, 55, 66], dtype=float)
    p = np.array([3000, 3000, -50, -50, -50, -50, 3000], dtype=float)

    mask, events = detect_dropouts(dt, p)

    assert mask[2:6].all()
    assert len(events) == 1
    assert events[0].dt_start == pytest.approx(22.0)
    assert events[0].dt_end == pytest.approx(55.0)


def test_dip_spanning_more_than_max_s_not_masked():
    """A 3-sample dip whose timestamps span > DROPOUT_MAX_S is never "momentary", regardless of
    the small sample count -- a corrupted time base must not produce a multi-hundred-year
    "momentary" dip."""
    dt = np.array([0.0, 50.0, 700.0, 1301.0, 1900.0])
    p = np.array([3000.0, 3000.0, 10.0, 8.0, 3000.0])

    mask, events = detect_dropouts(dt, p)

    assert not mask.any()
    assert events == []


# --------------------------------------------------------------------------------------------------
# Fast path (resample._dropout_candidates / _dropout_scan_candidates): must never disagree with
# _dropout_scan_loop, the reference implementation it replaces -- kept only for this comparison.
# --------------------------------------------------------------------------------------------------
def _run_scan_loop_directly(dt, p):
    """Calls the reference per-sample scan (resample._dropout_scan_loop) with detect_dropouts'
    own default parameters -- what the fast candidate-jump path in detect_dropouts must always
    agree with."""
    return resample._dropout_scan_loop(
        np.asarray(dt, dtype=float), np.asarray(p, dtype=float),
        DROPOUT_FLOOR_FRAC, DROPOUT_MIN_REF_PSI, RISE_GUARD_PSI, RISE_GUARD_SUSTAIN_S,
        RISE_GUARD_SUSTAIN_SAMPLES, DROPOUT_MAX_S, DROPOUT_LEADIN_MATCH_PSI)


def _events_as_tuples(events):
    return [(e.dt_start, e.dt_end, e.n_samples, e.p_min) for e in events]


@pytest.mark.parametrize("dt, p", [
    # A clean record: the fast path's whole reason to exist -- must actually skip the loop
    # (checked separately below) and still agree with it.
    (np.arange(50, dtype=float), np.linspace(5000.0, 1000.0, 50)),
    # Encore-shaped: onset + lead-in + dip + recovery.
    (np.array([0, 1, 2, 657, 660, 663, 666, 669, 672, 675, 678, 681, 684, 687, 690, 693,
              696, 699, 702], dtype=float),
     np.array([2988, 2988, 2988, 2978.5, 2953, 2933, 2914, 2866.5, 63.5, 15, 14, 13, 14, 15,
              14, 13, 2985, 2970, 2986], dtype=float)),
    # A crash that never recovers.
    (np.arange(50, dtype=float),
     np.concatenate([np.linspace(5000.0, 3000.0, 30), np.zeros(20)])),
    # A sustained (not masked) dip.
    (np.array([0, 15, 30, 45, 60, 75, 90, 105, 120, 135], dtype=float),
     np.array([3000, 3000, 5, 4, 3, 4, 5, 3005, 2995, 3000], dtype=float)),
    # A real step change straddling a one-sample glitch.
    (np.arange(141, dtype=float),
     np.concatenate([np.full(70, 916.0), [5.0], np.full(70, 2707.0)])),
    # A single-sample sentinel spike.
    (np.arange(11, dtype=float),
     np.concatenate([np.full(5, 3000.0), [-9999.0], np.full(5, 3000.0)])),
    # Two separate dropouts.
    (np.arange(40, dtype=float),
     np.concatenate([np.full(5, 3000.0), [30.0], np.full(14, 3000.0), [25.0],
                    np.full(19, 3000.0)])),
], ids=["clean", "encore", "crash_never_recovers", "sustained_dip", "step_change_glitch",
        "sentinel_spike", "two_dropouts"])
def test_fast_path_agrees_with_loop(dt, p):
    mask_fast, events_fast = detect_dropouts(dt, p)
    mask_loop, events_loop = _run_scan_loop_directly(dt, p)

    np.testing.assert_array_equal(mask_fast, mask_loop)
    assert _events_as_tuples(events_fast) == _events_as_tuples(events_loop)


def test_detect_dropouts_never_calls_the_reference_loop():
    """detect_dropouts must never call _dropout_scan_loop (the reference implementation) -- not
    on a clean record (skipped outright, no candidates) and, this round's actual fix, not on a
    record that DOES have a dropout either: _dropout_scan_candidates jumps between candidates
    instead of falling back to the full per-sample loop just because a candidate exists. Pinned
    via monkeypatching the reference loop to raise if it's ever called, against one clean and
    one dropout-containing record."""
    original = resample._dropout_scan_loop
    def _boom(*a, **kw):
        raise AssertionError("_dropout_scan_loop must not be called by detect_dropouts")
    resample._dropout_scan_loop = _boom
    try:
        clean_dt = np.arange(50, dtype=float)
        clean_p = np.linspace(5000.0, 1000.0, 50)
        assert not resample._dropout_onset_possible(clean_p, DROPOUT_FLOOR_FRAC,
                                                     DROPOUT_MIN_REF_PSI)
        mask, events = detect_dropouts(clean_dt, clean_p)
        assert not mask.any()
        assert events == []

        dirty_dt = np.arange(11, dtype=float)
        dirty_p = np.full(11, 3000.0)
        dirty_p[5] = -9999.0
        mask2, events2 = detect_dropouts(dirty_dt, dirty_p)
        assert mask2[5]
        assert len(events2) == 1
    finally:
        resample._dropout_scan_loop = original


def _random_dropout_series(rng: np.random.Generator, length: int):
    """One synthetic post-shut-in (dt, p) pair mixing declines, glitch runs to 0/12/-9999, an
    occasional exact onset-boundary pair, NaN/+-inf, and irregular dt spacing -- for the fuzz
    test below."""
    level = float(rng.uniform(200.0, 5000.0))
    p = rng.normal(loc=level, scale=float(rng.uniform(1.0, 50.0)), size=length)
    if rng.random() < 0.5:
        p = p + np.linspace(0.0, -float(rng.uniform(0.0, 3000.0)), length)
    if length >= 3 and rng.random() < 0.6:
        glitch_len = int(rng.integers(1, min(40, length - 1) + 1))
        start = int(rng.integers(0, length - glitch_len))
        glitch_level = float(rng.choice([0.0, 12.0, -9999.0, float(rng.uniform(-100.0, 50.0))]))
        p[start:start + glitch_len] = glitch_level
    if length >= 2 and rng.random() < 0.3:
        idx = int(rng.integers(1, length))
        if rng.random() < 0.5:
            p[idx - 1] = 100.0   # exactly AT min_ref -- must NOT trigger ('>' not '>=')
            p[idx] = 1.0
        else:
            p[idx - 1] = 150.0
            p[idx] = 0.1 * 150.0  # exactly AT floor_frac -- must trigger ('<=' not '<')
    if length and rng.random() < 0.5:
        n_bad = int(rng.integers(0, max(1, length // 10) + 1))
        if n_bad:
            bad_idx = rng.choice(length, size=min(n_bad, length), replace=False)
            for bi in bad_idx:
                p[int(bi)] = float(rng.choice([np.nan, np.inf, -np.inf]))
    if rng.random() < 0.5:
        dt = np.cumsum(rng.uniform(0.5, 120.0, size=length))
    else:
        dt = np.arange(length, dtype=float)
    return dt, p


def test_fast_path_fuzz_matches_reference_loop():
    """A few thousand randomized series -- declines, glitch runs to 0/12/-9999, NaN/+-inf,
    exact onset-boundary pairs, irregular dt, lengths 1-300 -- must all agree exactly between the
    fast candidate-jump scan and the reference per-sample loop."""
    rng = np.random.default_rng(20260925)
    n_series = 3000
    for _ in range(n_series):
        length = int(rng.integers(1, 301))
        dt, p = _random_dropout_series(rng, length)

        mask_fast, events_fast = detect_dropouts(dt, p)
        mask_loop, events_loop = _run_scan_loop_directly(dt, p)

        np.testing.assert_array_equal(mask_fast, mask_loop)
        assert _events_as_tuples(events_fast) == _events_as_tuples(events_loop)


# --------------------------------------------------------------------------------------------------
# End to end through compute_all
# --------------------------------------------------------------------------------------------------
_START_IDX, _SHUTIN_IDX = 50, 100
_DECLINE1_LEN, _DECLINE2_LEN = 200, 200


def _dropout_fixture(glitch: bool = True):
    """A post-shut-in record that declines smoothly from 5000 to 500 psi, with (if ``glitch``) a
    single momentary near-zero gauge glitch inserted partway through -- recovering at the very
    next sample, so masking it can never perturb which OTHER samples the resampler keeps (see
    test_dropout_masking_reproduces_result_with_the_row_absent below for why that matters).
    Returns ``(td, st, glitch_idx)``, the absolute index of the (possibly inserted) glitch."""
    decline_len = _DECLINE1_LEN + 1 + _DECLINE2_LEN
    n = _SHUTIN_IDX + decline_len
    t_s = np.arange(n, dtype=float)
    rate = np.zeros(n)
    rate[_START_IDX:_SHUTIN_IDX] = 5.0

    pressure = np.full(n, 2000.0)
    pressure[_START_IDX:_SHUTIN_IDX] = np.linspace(2000.0, 5000.0, _SHUTIN_IDX - _START_IDX)
    pressure[_SHUTIN_IDX:_SHUTIN_IDX + decline_len] = np.linspace(5000.0, 500.0, decline_len)
    glitch_idx = _SHUTIN_IDX + _DECLINE1_LEN
    if glitch:
        pressure[glitch_idx] = 15.0

    df = pd.DataFrame({PRESSURE_COL: pressure, "RATE": rate})
    td = IoTestData(path="<synthetic>", df=df, datetime_col="DATETIME", t_s=t_s,
                    columns=list(df.columns))
    st = PickState(pressure_col=PRESSURE_COL, rate_col="RATE", start_idx=_START_IDX,
                   shutin_idx=_SHUTIN_IDX)
    return td, st, glitch_idx


def test_compute_all_masks_dropout_no_guard_no_low_pressure_warning():
    td, st, glitch_idx = _dropout_fixture(glitch=True)
    res = compute_all(st, td)

    assert res.dropout_mask is not None
    assert res.dropout_mask[glitch_idx]
    assert int(res.dropout_mask.sum()) == 1
    # The rise guard must never fire at the recovery -- masking removed the artificial pin/jump
    # that the un-masked bug shape would have produced.
    assert res.resampled_full.guard_dt is None
    assert any("Pressure dropout masked" in w for w in res.warnings)
    assert not any("Surface pressure fell below 50 psi" in w for w in res.warnings)
    # Resampling continues well past the dropout -- not truncated to just the pre-dropout decline.
    assert res.resampled_full.dt[-1] > (_DECLINE1_LEN + _DECLINE2_LEN) * 0.5


def test_single_sample_dropout_warning_uses_singular_and_omits_zero_duration():
    """A single-sample glitch (the most common case, e.g. this fixture's) reads "1 sample" not
    "1 samples", and the duration clause is dropped entirely (a single-sample dip has dt_end ==
    dt_start == 0 s) rather than printing the useless "0 s"."""
    td, st, glitch_idx = _dropout_fixture(glitch=True)
    res = compute_all(st, td)

    msg = next(w for w in res.warnings if w.startswith("Pressure dropout masked"))
    assert "1 sample," in msg
    assert "1 samples" not in msg
    assert " s," not in msg
    assert "to 15 psi" in msg


def test_multi_sample_dropout_warning_keeps_plural_and_duration():
    """Sibling of the singular-form test above: a >1-sample, >0 s dip keeps the ordinary plural
    form with its duration clause."""
    start_idx, shutin_idx = 50, 100
    n = shutin_idx + 60
    t_s = np.arange(n, dtype=float)
    rate = np.zeros(n)
    rate[start_idx:shutin_idx] = 5.0
    pressure = np.full(n, 2000.0)
    pressure[start_idx:shutin_idx] = np.linspace(2000.0, 3000.0, shutin_idx - start_idx)
    pressure[shutin_idx:shutin_idx + 55] = np.linspace(3000.0, 2500.0, 55)
    pressure[shutin_idx + 55:shutin_idx + 58] = 20.0   # 3-sample dip
    pressure[shutin_idx + 58:] = 2470.0                # recovers and stays flat

    df = pd.DataFrame({PRESSURE_COL: pressure, "RATE": rate})
    td = IoTestData(path="<synthetic>", df=df, datetime_col="DATETIME", t_s=t_s,
                    columns=list(df.columns))
    st = PickState(pressure_col=PRESSURE_COL, rate_col="RATE", start_idx=start_idx,
                   shutin_idx=shutin_idx)
    res = compute_all(st, td)

    msg = next(w for w in res.warnings if w.startswith("Pressure dropout masked"))
    assert "3 samples," in msg
    assert "2 s," in msg
    assert "to 20 psi" in msg


def test_dropout_warning_omits_zero_duration_for_subsecond_multi_sample_dip():
    """A multi-sample dip lasting under 0.5 s (a sub-1-second sample rate) rounds to "0 s" via
    the warning's `:.0f` formatting -- the gate must be on that rounded value, not the raw
    `duration > 0`, or this prints the useless "(3 samples, 0 s, ...)"."""
    start_idx, shutin_idx = 50, 100
    n = shutin_idx + 60
    dt_step = 0.2
    t_s = np.arange(n, dtype=float) * dt_step
    rate = np.zeros(n)
    rate[start_idx:shutin_idx] = 5.0
    pressure = np.full(n, 2000.0)
    pressure[start_idx:shutin_idx] = np.linspace(2000.0, 3000.0, shutin_idx - start_idx)
    pressure[shutin_idx:shutin_idx + 55] = np.linspace(3000.0, 2500.0, 55)
    pressure[shutin_idx + 55:shutin_idx + 58] = 20.0   # 3-sample dip, 0.2 s spacing -> 0.4 s
    pressure[shutin_idx + 58:] = 2470.0

    df = pd.DataFrame({PRESSURE_COL: pressure, "RATE": rate})
    td = IoTestData(path="<synthetic>", df=df, datetime_col="DATETIME", t_s=t_s,
                    columns=list(df.columns))
    st = PickState(pressure_col=PRESSURE_COL, rate_col="RATE", start_idx=start_idx,
                   shutin_idx=shutin_idx)
    res = compute_all(st, td)

    msg = next(w for w in res.warnings if w.startswith("Pressure dropout masked"))
    assert "3 samples," in msg   # plural retained (n_samples=3)
    assert " s," not in msg      # duration (0.4 s) rounds to 0 -- omitted, not printed as "0 s"
    assert "to 20 psi" in msg


def test_dropout_warning_shows_less_than_1_min_for_early_events():
    """A dropout starting well within the first minute after shut-in must read "at <1 min", not
    the misleading "at 0 min" (reads as "no time elapsed at all"). Covers both the single-event
    and the "N pressure dropouts masked (first at ...)" multi-event message, via a second, later
    dropout in the same record."""
    start_idx, shutin_idx = 20, 50
    n = shutin_idx + 120
    t_s = np.arange(n, dtype=float)
    rate = np.zeros(n)
    rate[start_idx:shutin_idx] = 5.0
    pressure = np.full(n, 2000.0)
    pressure[start_idx:shutin_idx] = np.linspace(2000.0, 3000.0, shutin_idx - start_idx)
    pressure[shutin_idx:shutin_idx + 10] = np.linspace(3000.0, 2900.0, 10)
    pressure[shutin_idx + 10] = 15.0                     # first glitch, dt=10 s (< 1 min)
    pressure[shutin_idx + 11:shutin_idx + 90] = 2895.0
    pressure[shutin_idx + 90] = 12.0                     # second glitch, later in the record
    pressure[shutin_idx + 91:] = 2880.0

    df = pd.DataFrame({PRESSURE_COL: pressure, "RATE": rate})
    td = IoTestData(path="<synthetic>", df=df, datetime_col="DATETIME", t_s=t_s,
                    columns=list(df.columns))
    st = PickState(pressure_col=PRESSURE_COL, rate_col="RATE", start_idx=start_idx,
                   shutin_idx=shutin_idx)
    res = compute_all(st, td)

    assert len(res.dropouts) == 2
    msg = next(w for w in res.warnings if "pressure dropouts masked" in w)
    assert "first at <1 min" in msg
    assert "at 0 min" not in msg


def test_compute_all_no_dropout_no_warning_no_mask():
    td, st, glitch_idx = _dropout_fixture(glitch=False)
    res = compute_all(st, td)

    assert res.dropout_mask is not None
    assert not res.dropout_mask.any()
    assert res.dropouts == []
    assert not any("Pressure dropout masked" in w or "pressure dropouts masked" in w
                   for w in res.warnings)


def test_dropout_masking_reproduces_result_with_the_row_absent():
    """Masking a sample to NaN and having resample_pressure_increment skip it must give exactly
    the same kept points as if that row were never in the record at all -- NaN handling there is
    a pure `continue` (no state touched), identical either way. Built by literally deleting the
    glitch row from the "clean" record (same start_idx/shutin_idx, since the glitch sits after
    shut-in) rather than trying to guess a replacement value that wouldn't itself perturb which
    samples get kept."""
    td_dirty, st_dirty, glitch_idx = _dropout_fixture(glitch=True)
    res_dirty = compute_all(st_dirty, td_dirty)

    df_clean = td_dirty.df.drop(index=glitch_idx).reset_index(drop=True)
    t_s_clean = np.delete(td_dirty.t_s, glitch_idx)
    td_clean = IoTestData(path="<synthetic>", df=df_clean, datetime_col="DATETIME",
                          t_s=t_s_clean, columns=list(df_clean.columns))
    st_clean = PickState(pressure_col=PRESSURE_COL, rate_col="RATE", start_idx=_START_IDX,
                         shutin_idx=_SHUTIN_IDX)
    res_clean = compute_all(st_clean, td_clean)

    assert res_clean.dropouts == []  # sanity: the deleted-row record has no dropout of its own
    np.testing.assert_array_equal(res_dirty.resampled_full.dt, res_clean.resampled_full.dt)
    np.testing.assert_array_equal(res_dirty.resampled_full.p, res_clean.resampled_full.p)


def test_seed_tail_trim_ignores_masked_dropout():
    """A record whose only sub-100-psi samples sit inside a masked dropout must not get the
    "low_pressure" auto-trim -- picks.seed_tail_trim masks p_surface_post the same way
    model.compute_all masks res.bhp_all before resampling."""
    td, st, glitch_idx = _dropout_fixture(glitch=True)
    res = compute_all(st, td)

    picks.seed_tail_trim(st, td, res)

    assert st.tail_trim_dt is None
    assert st.tail_trim_reason == ""


# --------------------------------------------------------------------------------------------------
# render_overview / render_isip: the "dropout_masked" artist
# --------------------------------------------------------------------------------------------------
def test_render_overview_draws_dropout_masked_and_breaks_main_trace():
    td, st, glitch_idx = _dropout_fixture(glitch=True)
    res = compute_all(st, td)
    fig = Figure()
    ax = fig.add_subplot(111)
    plots.render_overview(ax, td, st, res, interactive=True)

    assert any(l.get_gid() == "dropout_masked" for l in ax.get_lines())
    main = next(l for l in ax.get_lines() if l.get_gid() is None)
    assert len(main.get_ydata()) == td.n  # no decimation at this record size
    assert np.isnan(main.get_ydata()[glitch_idx])


def test_render_overview_no_dropout_masked_when_no_events():
    td, st, _ = _dropout_fixture(glitch=False)
    res = compute_all(st, td)
    fig = Figure()
    ax = fig.add_subplot(111)
    plots.render_overview(ax, td, st, res, interactive=True)

    assert not any(l.get_gid() == "dropout_masked" for l in ax.get_lines())


def test_render_isip_draws_dropout_masked_and_breaks_main_trace():
    """render_isip's -5..+15 min window contains the fixture's glitch (200 s = 3.3 min after
    shut-in), mirroring the motivating Encore record."""
    td, st, glitch_idx = _dropout_fixture(glitch=True)
    res = compute_all(st, td)
    fig = Figure()
    ax = fig.add_subplot(111)
    plots.render_isip(ax, td, st, res)

    assert any(l.get_gid() == "dropout_masked" for l in ax.get_lines())
    main = ax.get_lines()[0]  # the pressure trace is drawn first

    t_min = (td.t_s - res.t_shutin_s) / 60.0
    m = (t_min >= -5.0) & (t_min <= 15.0)
    assert m[glitch_idx]  # sanity: the glitch really is inside the plotted window
    pos = int(np.sum(m[:glitch_idx]))
    assert np.isnan(main.get_ydata()[pos])


def test_render_isip_no_dropout_masked_when_no_events():
    td, st, _ = _dropout_fixture(glitch=False)
    res = compute_all(st, td)
    fig = Figure()
    ax = fig.add_subplot(111)
    plots.render_isip(ax, td, st, res)

    assert not any(l.get_gid() == "dropout_masked" for l in ax.get_lines())


# --------------------------------------------------------------------------------------------------
# Markers must not affect autoscale (a -9999 sentinel, or Encore's ~12 psi, otherwise drags the
# ISIP default view and the Overview y-slider's full range down to it).
# --------------------------------------------------------------------------------------------------
def test_plot_dropout_markers_does_not_affect_datalim():
    fig = Figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(10.0), np.linspace(5000.0, 4000.0, 10), color="black")
    before = ax.dataLim.get_points().copy()

    plots._plot_dropout_markers(ax, np.array([3.0]), np.array([-9999.0]))

    np.testing.assert_array_equal(ax.dataLim.get_points(), before)
    # The marker is still a real, findable artist at its true (here, off-screen) position --
    # only the Axes' bookkeeping is protected, not the artist itself.
    marker = next(l for l in ax.get_lines() if l.get_gid() == "dropout_masked")
    assert marker.get_ydata()[0] == pytest.approx(-9999.0)


def test_plot_dropout_markers_noop_when_empty():
    fig = Figure()
    ax = fig.add_subplot(111)
    ax.plot(np.arange(10.0), np.linspace(5000.0, 4000.0, 10), color="black")
    before = ax.dataLim.get_points().copy()

    plots._plot_dropout_markers(ax, np.array([]), np.array([]))

    np.testing.assert_array_equal(ax.dataLim.get_points(), before)
    assert not any(l.get_gid() == "dropout_masked" for l in ax.get_lines())


def test_render_isip_dropout_marker_does_not_pull_datalim_down():
    """End-to-end sibling of the unit test above: render_isip's -5..+15 min window holds the
    fixture's whole post-shut-in decline (500-5000 psi) alongside the masked 15 psi glitch --
    without the dataLim protection the axis would autoscale down to ~15, exactly the "ISIP view
    spans to ~12 psi on Encore" symptom."""
    td, st, glitch_idx = _dropout_fixture(glitch=True)
    res = compute_all(st, td)
    fig = Figure()
    ax = fig.add_subplot(111)
    plots.render_isip(ax, td, st, res)

    assert any(l.get_gid() == "dropout_masked" for l in ax.get_lines())
    assert ax.dataLim.intervaly[0] > 100.0  # the masked 15 psi sample must not pull this down


def test_render_overview_dropout_marker_does_not_pull_datalim_down():
    td, st, glitch_idx = _dropout_fixture(glitch=True)
    res = compute_all(st, td)
    fig = Figure()
    ax = fig.add_subplot(111)
    plots.render_overview(ax, td, st, res, interactive=True)

    assert any(l.get_gid() == "dropout_masked" for l in ax.get_lines())
    assert ax.dataLim.intervaly[0] > 100.0


# --------------------------------------------------------------------------------------------------
# Encore-style lead-in (flat baseline -> multi-sample slide -> multi-sample near-zero dip -> an
# intermediate ramp sample -> recovery), end to end, plus NaN robustness near its boundaries.
# --------------------------------------------------------------------------------------------------
def _encore_leadin_post_array() -> np.ndarray:
    """Post-shut-in shape mirroring the motivating Encore record (see the plan's Context table):
    a flat baseline (30 samples), a 5-sample lead-in slide (~120 psi total, well short of onset
    on its own), a 20-sample near-zero dip, one intermediate ramp sample partway through
    recovery, then a full recovery and a long further decline -- so resampled_full has real data
    to extend into, and nothing here could independently fire the rise guard."""
    baseline = 3000.0
    flat = np.full(30, baseline)
    leadin = baseline - np.array([24.0, 48.0, 72.0, 96.0, 120.0])
    dip = np.full(20, 10.0)
    ramp = np.array([1500.0])
    decline = np.linspace(2985.0, 400.0, 250)
    return np.concatenate([flat, leadin, dip, ramp, decline])


# Indices into _encore_leadin_post_array(), hand-derived from the algorithm (see
# test_encore_style_leadin_masks_leadin_and_dip's assertions for the walk that produces them).
_LEADIN_ONSET, _LEADIN_MASKED_FROM, _LEADIN_RAMP, _LEADIN_RECOVERED = 35, 32, 55, 56


def test_encore_style_leadin_masks_leadin_and_dip():
    post = _encore_leadin_post_array()
    dt = np.arange(len(post), dtype=float)

    mask, events = detect_dropouts(dt, post)

    assert not mask[:_LEADIN_MASKED_FROM].any()             # the flat baseline + first 2 of the
                                                              # 5 lead-in samples are untouched
    assert mask[_LEADIN_MASKED_FROM:_LEADIN_RECOVERED].all()  # 3 lead-in + 20 dip + the ramp
    assert not mask[_LEADIN_RECOVERED:].any()

    assert len(events) == 1
    assert events[0].dt_start == pytest.approx(float(_LEADIN_ONSET))
    assert events[0].dt_end == pytest.approx(float(_LEADIN_RAMP))
    assert events[0].n_samples == 21   # the 20 near-zero samples + the intermediate ramp sample
    assert events[0].p_min == pytest.approx(10.0)


def test_encore_style_leadin_nan_near_boundaries_same_decision():
    """NaN just before onset, inside the lead-in, and after recovery must not crash and must not
    change the mask anywhere except at those exact (already-non-finite) positions. The
    after-recovery NaN is placed at _LEADIN_RECOVERED + 7, outside the 5-finite-sample window
    detect_dropouts' R (the recovery-level reference for the lead-in walk) is computed from --
    a NaN inside that specific window legitimately shifts R by construction (it changes which
    samples are "the first 5 finite ones"), which is documented, correct behavior, not something
    this robustness check is about."""
    post = _encore_leadin_post_array()
    dt = np.arange(len(post), dtype=float)
    mask_ref, events_ref = detect_dropouts(dt, post)

    post_nan = post.copy()
    just_before_onset = _LEADIN_ONSET - 1
    inside_leadin = _LEADIN_ONSET - 2
    after_recovery = _LEADIN_RECOVERED + 7
    post_nan[[just_before_onset, inside_leadin, after_recovery]] = np.nan

    mask_nan, events_nan = detect_dropouts(dt, post_nan)

    assert len(events_ref) == len(events_nan) == 1
    assert events_ref[0].dt_start == events_nan[0].dt_start
    assert events_ref[0].dt_end == events_nan[0].dt_end
    assert events_ref[0].n_samples == events_nan[0].n_samples
    assert events_ref[0].p_min == pytest.approx(events_nan[0].p_min)
    unaffected = np.ones(len(post), dtype=bool)
    unaffected[[just_before_onset, inside_leadin, after_recovery]] = False
    np.testing.assert_array_equal(mask_ref[unaffected], mask_nan[unaffected])


def _encore_style_leadin_fixture():
    """Embeds _encore_leadin_post_array() into a full synthetic record (injection ramp, then the
    lead-in/dip/ramp/recovery shape) for an end-to-end compute_all/seed_tail_trim/render_overview
    check. Returns (td, st, leadin_lo, ramp_idx, recovered_idx) -- absolute indices."""
    start_idx, shutin_idx = 20, 50
    post = _encore_leadin_post_array()
    n_pre = shutin_idx
    pressure_pre = np.full(n_pre, 2000.0)
    pressure_pre[start_idx:shutin_idx] = np.linspace(2000.0, 3000.0, shutin_idx - start_idx)
    pressure = np.concatenate([pressure_pre, post])
    n = len(pressure)
    rate = np.zeros(n)
    rate[start_idx:shutin_idx] = 5.0
    t_s = np.arange(n, dtype=float)

    df = pd.DataFrame({PRESSURE_COL: pressure, "RATE": rate})
    td = IoTestData(path="<synthetic>", df=df, datetime_col="DATETIME", t_s=t_s,
                    columns=list(df.columns))
    st = PickState(pressure_col=PRESSURE_COL, rate_col="RATE", start_idx=start_idx,
                   shutin_idx=shutin_idx)
    return (td, st, shutin_idx + _LEADIN_MASKED_FROM, shutin_idx + _LEADIN_RAMP,
            shutin_idx + _LEADIN_RECOVERED)


def test_encore_style_leadin_end_to_end_through_compute_all_and_render():
    td, st, leadin_lo, ramp_idx, recovered_idx = _encore_style_leadin_fixture()
    res = compute_all(st, td)

    assert res.dropout_mask[leadin_lo:ramp_idx + 1].all()
    assert not res.dropout_mask[:leadin_lo].any()
    assert not res.dropout_mask[recovered_idx:].any()
    # The rise guard must not fire at the recovery -- masking removed the artificial pin/jump.
    assert res.resampled_full.guard_dt is None
    assert any("Pressure dropout masked" in w for w in res.warnings)

    picks.seed_tail_trim(st, td, res)
    assert st.tail_trim_dt is None
    assert st.tail_trim_reason == ""

    fig = Figure()
    ax = fig.add_subplot(111)
    plots.render_overview(ax, td, st, res, interactive=True)
    assert any(l.get_gid() == "dropout_masked" for l in ax.get_lines())
    main = next(l for l in ax.get_lines() if l.get_gid() is None)
    assert np.isnan(main.get_ydata()[ramp_idx])       # the intermediate ramp sample is masked
    assert np.isnan(main.get_ydata()[leadin_lo])       # so is the masked lead-in
