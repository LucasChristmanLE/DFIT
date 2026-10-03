import numpy as np
from matplotlib.figure import Figure

from dfit_tool.model import compute_all
from dfit_tool import colors as C
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
    assert press_line.get_color() == C.PRESSURE
    # rate trace lives on the twin axis; find it among figure axes
    twins = [a for a in fig.axes if a is not ax]
    rate_line = twins[0].get_lines()[0]
    assert rate_line.get_color() == C.RATE


def test_pressure_trace_is_red_when_surface():
    st = injection_state(make_testdata()); st.pressure_is_bhp = False
    _, ax = _injection_axis(st)
    assert ax.get_lines()[0].get_color() == C.SURFACE_PRESSURE


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
    assert press_line.get_color() == C.PRESSURE
    assert press_line.get_label() == "Bottomhole Pressure"


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
    assert bhp.get_color() == C.PRESSURE and bhp.get_label() == "Bottomhole Pressure"
    assert len(surf) == 1
    assert surf[0].get_color() == C.SURFACE_PRESSURE
    assert surf[0].get_linewidth() < bhp.get_linewidth()
    assert np.nanmax(surf[0].get_ydata()) < np.nanmax(bhp.get_ydata())
    assert ax.get_ylabel() == "Pressure (psi)"


def test_overview_no_surface_overlay_when_channel_is_bhp_or_unconverted():
    for is_bhp in (True, False):
        st = injection_state(make_testdata()); st.pressure_is_bhp = is_bhp
        ax = _overview_axis(st)
        assert not [l for l in ax.get_lines() if l.get_gid() == "surface_pressure"]
        assert ax.get_ylabel() == ("Bottomhole Pressure (psi)" if is_bhp else "Surface Pressure (psi)")


def _step_axis(renderer, state):
    td = make_testdata()
    res = compute_all(state, td)
    fig = Figure(); ax = fig.add_subplot(111)
    defaults = renderer(ax, td, state, res)
    return ax, defaults


def _surface_state(converted):
    st = injection_state(make_testdata())
    st.pressure_is_bhp = False
    if converted:
        st.density_ppg = 9.0
        st.tvd_ft = 8000.0
    return st


def test_isip_unconverted_surface_pressure_is_red_and_not_called_bhp():
    # Surface channel, no density/TVD: Overview and Injection draw it red as "Surface Pressure"; the
    # ISIP step must too, not a black "BHP" trace.
    ax, defaults = _step_axis(plots.render_isip, _surface_state(converted=False))
    assert ax.get_lines()[0].get_color() == C.SURFACE_PRESSURE
    assert ax.get_ylabel() == "Surface Pressure (psi)"
    assert defaults.y_color == C.SURFACE_PRESSURE


def test_isip_converted_surface_pressure_is_black_bhp():
    ax, defaults = _step_axis(plots.render_isip, _surface_state(converted=True))
    assert ax.get_lines()[0].get_color() == C.PRESSURE
    assert ax.get_ylabel() == "Bottomhole Pressure (psi)"
    assert defaults.y_color == C.PRESSURE


def test_later_steps_never_call_unconverted_surface_pressure_bhp():
    st = _surface_state(converted=False)
    for renderer in (plots.render_gfunction, plots.render_tangent, plots.render_porepressure):
        ax, _ = _step_axis(renderer, st)
        assert ax.get_ylabel() == "Surface Pressure (psi)", renderer.__name__
        if renderer is not plots.render_porepressure:  # pore-pressure trace has no legend entry
            assert ax.get_lines()[0].get_label() == "Surface Pressure", renderer.__name__


def test_gfunction_and_tangent_draw_unconverted_surface_pressure_red():
    # Red marks unconverted surface pressure on every step. The derivative stays red too: an
    # analyst should never reach these steps without converting to BHP.
    st = _surface_state(converted=False)
    for renderer in (plots.render_gfunction, plots.render_tangent):
        ax, defaults = _step_axis(renderer, st)
        assert ax.get_lines()[0].get_color() == C.SURFACE_PRESSURE, renderer.__name__
        assert defaults.y_color == C.SURFACE_PRESSURE, renderer.__name__
        twin = next(a for a in ax.figure.axes if a is not ax)
        assert twin.get_lines()[0].get_color() == C.DERIVATIVE, renderer.__name__
        assert defaults.y2_color == C.DERIVATIVE, renderer.__name__


def test_gfunction_and_tangent_keep_black_bhp_and_red_derivative_when_converted():
    st = _surface_state(converted=True)
    for renderer in (plots.render_gfunction, plots.render_tangent):
        ax, defaults = _step_axis(renderer, st)
        assert ax.get_lines()[0].get_color() == C.PRESSURE, renderer.__name__
        twin = next(a for a in ax.figure.axes if a is not ax)
        assert twin.get_lines()[0].get_color() == C.DERIVATIVE, renderer.__name__
        assert defaults.y2_color == C.DERIVATIVE, renderer.__name__


def test_later_steps_label_converted_pressure_bhp():
    st = _surface_state(converted=True)
    for renderer in (plots.render_gfunction, plots.render_tangent, plots.render_porepressure):
        ax, _ = _step_axis(renderer, st)
        assert ax.get_ylabel() == "Bottomhole Pressure (psi)", renderer.__name__
