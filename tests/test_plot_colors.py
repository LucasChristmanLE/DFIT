import numpy as np
from matplotlib.figure import Figure

from dfit_tool.model import compute_all
from dfit_tool import plots
from tests.helpers import make_testdata, injection_state


def _injection_axis(state):
    td = make_testdata()
    res = compute_all(state, td)
    fig = Figure(); ax = fig.add_subplot(111)
    plots.render_injection(ax, td, state, res)
    return fig, ax


def test_rate_trace_is_blue_and_bhp_black_when_bhp():
    st = injection_state(make_testdata()); st.pressure_is_bhp = True
    fig, ax = _injection_axis(st)
    # pressure trace on primary axis is black
    press_line = ax.get_lines()[0]
    assert press_line.get_color() == "black"
    # rate trace lives on the twin axis; find it among figure axes
    twins = [a for a in fig.axes if a is not ax]
    rate_line = twins[0].get_lines()[0]
    assert rate_line.get_color() == "tab:blue"


def test_pressure_trace_is_red_when_surface():
    st = injection_state(make_testdata()); st.pressure_is_bhp = False
    _, ax = _injection_axis(st)
    assert ax.get_lines()[0].get_color() == "tab:red"


def test_pressure_trace_is_black_and_labeled_bhp_when_density_and_tvd_convert_surface_to_bhp():
    # pressure_is_bhp stays False (the checkbox is unchecked) but density/TVD are set, so
    # model.compute_all derives true BHP from surface pressure -- the trace should follow the
    # derived result (res.pressure_is_bhp), not the raw checkbox state.
    st = injection_state(make_testdata())
    st.pressure_is_bhp = False
    st.density_ppg = 9.0
    st.tvd_ft = 8000.0
    _, ax = _injection_axis(st)
    press_line = ax.get_lines()[0]
    assert press_line.get_color() == "black"
    assert press_line.get_label() == "bottomhole pressure"


def _overview_axis(state):
    td = make_testdata()
    res = compute_all(state, td)
    fig = Figure(); ax = fig.add_subplot(111)
    plots.render_overview(ax, td, state, res)
    return ax


def test_overview_overlays_thin_red_surface_pressure_when_converted_to_bhp():
    st = injection_state(make_testdata())
    st.pressure_is_bhp = False
    st.density_ppg = 9.0
    st.tvd_ft = 8000.0
    ax = _overview_axis(st)
    bhp = ax.get_lines()[0]
    surf = [l for l in ax.get_lines() if l.get_gid() == "surface_pressure"]
    assert bhp.get_color() == "black" and bhp.get_label() == "bottomhole pressure"
    assert len(surf) == 1
    assert surf[0].get_color() == "tab:red"
    assert surf[0].get_linewidth() < bhp.get_linewidth()
    assert np.nanmax(surf[0].get_ydata()) < np.nanmax(bhp.get_ydata())
    assert ax.get_ylabel() == "pressure (psi)"


def test_overview_no_surface_overlay_when_channel_is_bhp_or_unconverted():
    for is_bhp in (True, False):
        st = injection_state(make_testdata()); st.pressure_is_bhp = is_bhp
        ax = _overview_axis(st)
        assert not [l for l in ax.get_lines() if l.get_gid() == "surface_pressure"]
        assert ax.get_ylabel() == "pressure (psi)"
