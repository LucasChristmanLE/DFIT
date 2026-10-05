"""The log-log window is the only late-time window: the pore-pressure fit reads loglog_window,
and a drag on the Pore pressure step writes it. There is no separate pp_window to go stale."""

from __future__ import annotations

import numpy as np
import pytest
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from dfit_tool import model, picks, plots
from dfit_tool.model import PickState, compute_all
from tests.helpers import make_testdata, injection_state


def _state():
    td = make_testdata()
    st = injection_state(td)
    st.postclosure_scenario = "PC-A linear"
    return td, st, compute_all(st, td)


def test_pore_pressure_fit_uses_the_loglog_window():
    td, st, res = _state()
    t = res.diagnostics.t
    st.loglog_window = (float(t[len(t) // 2]), float(t[-3]))
    res = compute_all(st, td)
    lo, hi = st.loglog_window
    assert res.pore_pressure is not None
    assert res.pore_pressure_n_points == int(((t >= lo) & (t <= hi) & (t > 0)).sum())


def test_pp_span_drag_writes_the_loglog_window():
    st = PickState(pp_axis="tm12")
    picks.handle_pp_span(st, 0.05, 0.1)
    assert st.loglog_window == pytest.approx((100.0, 400.0))


def test_pp_window_is_not_a_pick_and_old_saves_drop_it():
    assert not hasattr(PickState(), "pp_window")
    st = model._decode({"loglog_window": [10.0, 20.0], "pp_window": [1.0, 2.0]})
    assert st.loglog_window == (10.0, 20.0)


def test_seed_pp_seeds_the_loglog_window_when_missing():
    td, st, res = _state()
    assert st.loglog_window is None
    picks.seed_pp(st, res)
    assert st.loglog_window is not None


def test_stale_warning_names_the_loglog_window():
    td, st, res = _state()
    t = res.diagnostics.t
    st.loglog_window = (float(t[len(t) // 2]), float(t[-1]))
    st.tail_trim_dt = float(t[-5])
    res = compute_all(st, td)
    assert any(w.startswith("log-log window") and "beyond the tail trim" in w
               for w in res.warnings)
    assert not any("pore-pressure window" in w for w in res.warnings)


def test_window_inside_the_trimmed_data_is_not_stale():
    td, st, res = _state()
    t = res.diagnostics.t
    st.loglog_window = (float(t[len(t) // 2]), float(t[-8]))
    st.tail_trim_dt = float(t[-5])
    res = compute_all(st, td)
    assert not any("beyond the tail trim" in w for w in res.warnings)


def test_render_loglog_clamps_an_open_ended_window_to_the_last_sample():
    td, st, res = _state()
    t = res.diagnostics.t
    st.loglog_window = (float(t[len(t) // 2]), float("inf"))
    res = compute_all(st, td)
    fig = Figure()
    ax = fig.add_subplot(111)
    plots.render_loglog(ax, td, st, res)
    rects = [p for p in ax.patches if isinstance(p, Rectangle)]
    assert len(rects) == 1
    assert rects[0].get_x() + rects[0].get_width() == np.float64(t[-1])


def test_loglog_drag_past_the_last_sample_means_to_the_end():
    """SpanSelector doesn't clamp to the data, so a drag to the plot's right edge lands past the
    last sample. That is "to the end" (inf), never a stale pick on a trimmed record."""
    td, st, res = _state()
    t = res.diagnostics.t
    st.tail_trim_dt = float(t[-5])
    res = compute_all(st, td)
    edge = float(res.diagnostics.t[-1])
    picks.handle_loglog_span(st, float(res.diagnostics.t[len(t) // 2]), edge * 1.3,
                             t_end=edge)
    assert st.loglog_window[1] == float("inf")
    res = compute_all(st, td)
    assert not any("beyond the tail trim" in w for w in res.warnings)
    picks.handle_loglog_span(st, 10.0, edge * 0.5, t_end=edge)
    assert st.loglog_window == (10.0, edge * 0.5)
