import numpy as np
from matplotlib.figure import Figure

from dfit_tool.model import compute_all
from dfit_tool import plots
from tests.helpers import make_testdata, overview_state


def _overview_axis(state):
    td = make_testdata()
    res = compute_all(state, td)
    fig = Figure(); ax = fig.add_subplot(111)
    plots.render_overview(ax, td, state, res)
    return fig, ax


def test_rate_trace_is_blue_and_bhp_black_when_bhp():
    st = overview_state(make_testdata()); st.pressure_is_bhp = True
    fig, ax = _overview_axis(st)
    # pressure trace on primary axis is black
    press_line = ax.get_lines()[0]
    assert press_line.get_color() == "black"
    # rate trace lives on the twin axis; find it among figure axes
    twins = [a for a in fig.axes if a is not ax]
    rate_line = twins[0].get_lines()[0]
    assert rate_line.get_color() == "tab:blue"


def test_pressure_trace_is_red_when_surface():
    st = overview_state(make_testdata()); st.pressure_is_bhp = False
    _, ax = _overview_axis(st)
    assert ax.get_lines()[0].get_color() == "tab:red"
