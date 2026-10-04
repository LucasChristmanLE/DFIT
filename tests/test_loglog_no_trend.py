"""PC-E ("no trend") and PC-F ("no peak") show no log-log window or slope; PC-X
("uninterpretable") keeps them but skips porepressure and stiffness like PC-F.

Covers: model.compute_all leaving loglog_slope blank; plots.render_loglog drawing no window;
model.skipped_steps/last_step/resolve_step under PC-X; store.build_log_row/status_for; the
summary row; and ui.DfitApp._attach_controllers wiring (duck-typed stand-in, no tk.Tk()).
"""

from __future__ import annotations

import os
import types

import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from dfit_tool import picks, plots, store, summary, ui
from dfit_tool.model import (STEP_KEYS, PickState, compute_all, last_step, resolve_step,
                             skipped_steps)
from dfit_tool.ui import DfitApp
from tests.helpers import injection_state, make_testdata

NO_TREND = ["PC-E no trend", "PC-F no peak"]
WITH_WINDOW = ["PC-A linear", "PC-X uninterpretable"]


def _seeded(scenario):
    td = make_testdata()
    st = injection_state(td)
    picks.seed_loglog(st, compute_all(st, td))
    assert st.loglog_window is not None
    st.postclosure_scenario = scenario
    st.postclosure_auto = False
    return td, st, compute_all(st, td)


def _render(td, st, res):
    fig = Figure()
    ax = fig.add_subplot(111)
    plots.render_loglog(ax, td, st, res)
    return ax


@pytest.mark.parametrize("scenario", NO_TREND)
def test_no_trend_scenarios_report_no_slope_and_keep_window_in_state(scenario):
    _, st, res = _seeded(scenario)
    assert res.loglog_slope is None
    assert st.loglog_window is not None


@pytest.mark.parametrize("scenario", WITH_WINDOW)
def test_other_scenarios_report_slope(scenario):
    _, _, res = _seeded(scenario)
    assert res.loglog_slope is not None


@pytest.mark.parametrize("scenario", NO_TREND)
def test_render_loglog_draws_no_window_under_no_trend(scenario):
    td, st, res = _seeded(scenario)
    ax = _render(td, st, res)
    assert not ax.patches
    assert "Slope" not in ax.get_title()
    assert scenario in ax.get_title()


@pytest.mark.parametrize("scenario", WITH_WINDOW)
def test_render_loglog_draws_window_and_slope(scenario):
    td, st, res = _seeded(scenario)
    ax = _render(td, st, res)
    assert ax.patches
    assert f"Slope={res.loglog_slope:.2f}" in ax.get_title()


def test_pcx_skips_like_pcf():
    pcx = PickState(postclosure_scenario="PC-X uninterpretable")
    pcf = PickState(postclosure_scenario="PC-F no peak")
    assert skipped_steps(pcx) == skipped_steps(pcf) == {"porepressure", "stiffness"}
    assert last_step(pcx) == "loglog"
    assert resolve_step(pcx, "stiffness") == "loglog"


def test_pcx_in_combobox():
    assert "PC-X uninterpretable" in ui.POSTCLOSURE_SCENARIOS
    assert picks.suggest_pp_axis("PC-X uninterpretable") is None


def test_pcx_log_row(tmp_path):
    td, st, _ = _seeded("PC-X uninterpretable")
    picks.seed_pp(st, compute_all(st, td))
    st.stiffness_no_upturn = True
    res = compute_all(st, td)
    entry = store.TestEntry(test_id="well1", folder=str(tmp_path))
    row = store.build_log_row(entry, os.path.join(str(tmp_path), "well1.csv"), str(tmp_path),
                              st, td, res)
    assert row["postclosure_trend"] == "uninterpretable"
    assert row["pore_pressure"] is None
    assert row["Shmin_stiffness"] is None
    assert row["stiffness_no_upturn"] == ""


def test_pcx_status_counts_skipped_steps_as_accounted_for():
    st = PickState(postclosure_scenario="PC-X uninterpretable",
                   step_status={k: "done" for k in STEP_KEYS
                                if k not in ("porepressure", "stiffness")})
    assert store.status_for(st) == "done"


def test_pcx_summary_row():
    td, st, res = _seeded("PC-X uninterpretable")
    post = next(s for s in summary.summary_sections(st, res) if s.title == "Postclosure")
    assert ["pore pressure fit", "skipped (PC-X)"] in post.rows


def _ctrl_stub(td, st, res):
    stub = types.SimpleNamespace()
    fig = Figure()
    stub.fig = fig
    stub.ax = fig.add_subplot(111)
    stub.canvas = FigureCanvasAgg(fig)
    plots.render_loglog(stub.ax, td, st, res)
    stub.canvas.draw()
    stub.td, stub.res, stub.state, stub.step = td, res, st, "loglog"
    stub._controllers = []
    stub.hints = []
    stub.hint_lbl = types.SimpleNamespace(config=lambda **kw: stub.hints.append(kw.get("text")))
    stub.refresh = lambda: None
    stub._twin_axes = types.MethodType(DfitApp._twin_axes, stub)
    return stub


@pytest.mark.parametrize("scenario", NO_TREND)
def test_no_span_controller_under_no_trend(scenario):
    stub = _ctrl_stub(*_seeded(scenario))
    DfitApp._attach_controllers(stub)
    assert not any(isinstance(c, picks.SpanController) for c in stub._controllers)
    assert scenario in stub.hints[-1]


@pytest.mark.parametrize("scenario", WITH_WINDOW)
def test_span_controller_wired_otherwise(scenario):
    stub = _ctrl_stub(*_seeded(scenario))
    DfitApp._attach_controllers(stub)
    assert any(isinstance(c, picks.SpanController) for c in stub._controllers)
