"""Falloff duration: shut-in to the effective tail cutoff (the tail-trim line), or to the end of
the record when nothing cuts it. Reported in days in Expanded results and the log."""

from __future__ import annotations

import os

import pytest

from dfit_tool import store, summary
from dfit_tool.model import compute_all
from tests.helpers import injection_state, make_testdata

LABEL = "falloff duration (days)"


def _setup():
    td = make_testdata()
    st = injection_state(td)
    res = compute_all(st, td)
    t_end = float((td.t_s - res.t_shutin_s).max())
    return td, st, t_end


def _inputs_row(st, res):
    sec = next(s for s in summary.summary_sections(st, res) if s.title == "Inputs and data")
    return next(r for r in sec.rows if r[0] == LABEL)


def _log_row(tmp_path, st, td):
    res = compute_all(st, td)
    entry = store.TestEntry(test_id="well1", folder=str(tmp_path))
    return store.build_log_row(entry, os.path.join(str(tmp_path), "well1.csv"), str(tmp_path),
                               st, td, res)


def test_no_trim_is_end_of_record():
    td, st, t_end = _setup()
    res = compute_all(st, td)
    assert res.resampled_full.guard_dt is None
    assert res.falloff_duration_s == pytest.approx(t_end)


def test_trim_sets_duration():
    td, st, t_end = _setup()
    st.tail_trim_dt = 0.5 * t_end
    assert compute_all(st, td).falloff_duration_s == pytest.approx(0.5 * t_end)
    st.tail_trim_dt = 0.25 * t_end
    assert compute_all(st, td).falloff_duration_s == pytest.approx(0.25 * t_end)


def test_trim_past_end_clamps_to_last_sample():
    td, st, t_end = _setup()
    st.tail_trim_dt = 2.0 * t_end
    st.tail_guard_override = True
    assert compute_all(st, td).falloff_duration_s == pytest.approx(t_end)


def test_summary_row_in_days():
    td, st, t_end = _setup()
    st.tail_trim_dt = 0.5 * t_end
    res = compute_all(st, td)
    assert _inputs_row(st, res) == [LABEL, f"{0.5 * t_end / 86400.0:.2f}"]


def test_log_row_writes_days(tmp_path):
    td, st, t_end = _setup()
    st.tail_trim_dt = 0.5 * t_end
    assert _log_row(tmp_path, st, td)["falloff_duration_days"] == pytest.approx(
        0.5 * t_end / 86400.0)
    assert store.LOG_COLUMNS[-1] == "falloff_duration_days"
