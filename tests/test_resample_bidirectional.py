"""Direct semantic tests for the bidirectional (+-step) resampling keep rule:
``resample.resample_pressure_increment`` keeps a point whenever ``abs(p_i - last_kept) >= step``,
in EITHER direction, not just on a decline. ``tests/test_resample_vectorized.py`` fuzzes the fast
path against the reference loop for agreement; these tests instead pin the actual, expected
numbers against a handful of small, hand-worked fixtures, so a regression that made both
implementations agreeably wrong would still be caught here.
"""

from __future__ import annotations

import numpy as np
import pytest

from dfit_tool.resample import resample_pressure_increment


def test_rise_of_exactly_step_is_kept():
    """After a decline, a rise of exactly ``step`` off the last kept point is kept -- the keep
    rule is ``>=``, not ``>``, in the rising direction too."""
    dt = np.array([0.0, 1.0, 2.0])
    p = np.array([1000.0, 970.0, 1000.0])  # -30 then +30

    rs = resample_pressure_increment(dt, p, step=30.0)

    np.testing.assert_array_equal(rs.dt, dt)
    np.testing.assert_array_equal(rs.p, p)


def test_rise_of_just_under_step_is_not_kept():
    """A +29 rise (one psi under step) is not kept -- the last kept point stays at the low."""
    dt = np.array([0.0, 1.0, 2.0])
    p = np.array([1000.0, 970.0, 999.0])  # -30 then +29

    rs = resample_pressure_increment(dt, p, step=30.0)

    np.testing.assert_array_equal(rs.dt, np.array([0.0, 1.0]))
    np.testing.assert_array_equal(rs.p, np.array([1000.0, 970.0]))


def test_subsequent_decline_is_measured_from_the_kept_peak():
    """Once a rise is kept, a later decline is measured from that NEW kept value (the peak),
    not from whatever low preceded the rise. Constructed so the two readings disagree: measured
    from the pre-rise low (970), the final sample's -59 drop from the peak would only be a -29
    drop from 970 and would NOT be kept; measured correctly from the kept peak (1000), it is a
    genuine -59 drop and IS kept."""
    dt = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    p = np.array([1000.0, 970.0, 1000.0, 975.0, 941.0])
    #              kept    kept   kept    skip    kept (|941-1000| = 59 >= 30;
    #                                              |941-970| would only be 29 < 30)

    rs = resample_pressure_increment(dt, p, step=30.0)

    np.testing.assert_array_equal(rs.dt, np.array([0.0, 1.0, 2.0, 4.0]))
    np.testing.assert_array_equal(rs.p, np.array([1000.0, 970.0, 1000.0, 941.0]))


@pytest.mark.parametrize("stop_at_guard", [True, False])
def test_stop_at_guard_true_keeps_nothing_at_or_after_guard_dt(stop_at_guard):
    """A fired guard's own excursion is now keepable under the bidirectional rule (the jump into
    the elevated level is itself a >= step move) -- but stop_at_guard=True must still discard
    every kept index at or after the run's first sample (guard_dt), same as the historic
    hard-stop contract. stop_at_guard=False, by contrast, keeps the jump."""
    n = 600
    dt = np.arange(n, dtype=float)
    p = np.linspace(4000.0, 2000.0, n)
    rise_start = 300
    p[rise_start:] = p[rise_start - 1] + 250.0  # jumps once, then holds perfectly flat forever

    rs = resample_pressure_increment(dt, p, step=30.0, stop_at_guard=stop_at_guard)

    assert rs.guard_dt == pytest.approx(dt[rise_start])
    if stop_at_guard:
        assert np.all(rs.dt < rs.guard_dt)
    else:
        # The jump itself (at exactly guard_dt) is the one new thing a flat-forever tail has to
        # offer -- nothing later, since it never moves again once elevated.
        assert rs.dt[-1] == pytest.approx(dt[rise_start])
        assert rs.p[-1] == pytest.approx(p[rise_start])
        assert np.all(rs.dt[:-1] < rs.guard_dt)
