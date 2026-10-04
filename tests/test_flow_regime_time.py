"""Time to flow regime: shut-in-relative time of the first sample in the log-log window, or of
the t*dP/dt peak under PC-E. n/a (None) under PC-F and PC-X."""

from __future__ import annotations

import os

import numpy as np
import pytest

from dfit_tool import store, summary
from dfit_tool.model import compute_all
from tests.helpers import injection_state, make_testdata

LABEL = "time to flow regime (min)"


def _state():
    td = make_testdata()
    st = injection_state(td)
    res = compute_all(st, td)
    dg = res.diagnostics
    n = len(dg.t)
    # Window low edge between two samples: the value snaps to the later one.
    lo = 0.5 * (dg.t[n // 2] + dg.t[n // 2 + 1])
    st.loglog_window = (float(lo), float(dg.t[-1]))
    st.postclosure_scenario = "PC-A linear"
    st.step_status = {"loglog": "done", "porepressure": "done"}
    return td, st, float(dg.t[n // 2 + 1])


def _post_row(st, res):
    post = next(s for s in summary.summary_sections(st, res) if s.title == "Postclosure")
    return next(r for r in post.rows if r[0] == LABEL)


def _log_row(tmp_path, st, td):
    res = compute_all(st, td)
    entry = store.TestEntry(test_id="well1", folder=str(tmp_path))
    return store.build_log_row(entry, os.path.join(str(tmp_path), "well1.csv"), str(tmp_path),
                               st, td, res)


def test_window_snaps_to_first_sample_inside():
    td, st, t_first = _state()
    res = compute_all(st, td)
    assert res.flow_regime_time_s == pytest.approx(t_first)
    assert _post_row(st, res) == [LABEL, f"{t_first / 60.0:.2f}"]


def test_window_past_last_sample_is_none():
    td, st, _ = _state()
    t_end = float(compute_all(st, td).diagnostics.t[-1])
    st.loglog_window = (2.0 * t_end, 3.0 * t_end)
    assert compute_all(st, td).flow_regime_time_s is None


def test_pce_uses_peak():
    td, st, _ = _state()
    st.postclosure_scenario = "PC-E no trend"
    dg = compute_all(st, td).diagnostics
    st.pce_peak_t = float(dg.t[int(np.nanargmax(np.where(dg.t > 0, dg.tdpdt, -np.inf)))])
    res = compute_all(st, td)
    assert res.pce_peak_t is not None
    assert res.flow_regime_time_s == pytest.approx(res.pce_peak_t)
    assert _post_row(st, res) == [LABEL, f"{res.pce_peak_t / 60.0:.2f}"]


@pytest.mark.parametrize("scen", ["PC-F no peak", "PC-X uninterpretable"])
def test_pcf_pcx_are_na(scen, tmp_path):
    td, st, _ = _state()
    st.postclosure_scenario = scen
    res = compute_all(st, td)
    assert res.flow_regime_time_s is None
    assert _post_row(st, res) == [LABEL, f"n/a ({scen[:4]})"]
    assert _log_row(tmp_path, st, td)["flow_regime_time_min"] is None


def test_summary_gated_until_loglog_visited():
    td, st, _ = _state()
    st.step_status = {}
    assert _post_row(st, compute_all(st, td)) == [LABEL, "-"]


def test_log_row_writes_minutes(tmp_path):
    td, st, t_first = _state()
    row = _log_row(tmp_path, st, td)
    assert row["flow_regime_time_min"] == pytest.approx(t_first / 60.0)
    assert store.LOG_COLUMNS[-1] == "flow_regime_time_min"
