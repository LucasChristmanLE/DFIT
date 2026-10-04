"""The axis-only overhang measurements in ``ui._measure_overhang_px`` /
``_measure_bottom_overhang_px`` must equal the old whole-Axes ``get_tightbbox`` measurements on
every step (the cheaper form only changed what is walked, not the extent)."""
import types

import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from dfit_tool import plots
from dfit_tool.model import STEPS, compute_all
from dfit_tool.plots import ViewState
from dfit_tool.ui import DfitApp
from tests.helpers import injection_state, make_testdata

_METHODS = ("_make_range_slider", "_build_sliders", "_twin_axes", "_d2_axes", "_layout_sliders",
            "_x_track_geometry_px", "_get_renderer", "_measure_overhang_px",
            "_measure_bottom_overhang_px", "_measure_slider_text_col_px")


def _reference_overhangs(stub):
    """The pre-optimization measurement: whole-Axes tightbbox for every axis."""
    renderer = stub._get_renderer()
    right = max(ax.get_tightbbox(renderer).x1 - stub.ax.bbox.x1
                for ax in (stub.ax, stub._twin_axes(), stub._d2_axes()) if ax is not None)
    bottom = max(stub.ax.bbox.y0 - stub.ax.get_tightbbox(renderer).y0, 0.0)
    return right, bottom


@pytest.mark.parametrize("size", [(9, 6), (6.5, 4.5)])
@pytest.mark.parametrize("step", [k for k, _ in STEPS])
def test_axis_only_overhang_matches_whole_axes_tightbbox(step, size):
    td = make_testdata()
    state = injection_state(td)
    state.show_d2pdg2 = step == "gfunction"
    res = compute_all(state, td)
    stub = types.SimpleNamespace()
    stub.fig = Figure(figsize=size, dpi=100.0)
    stub.ax = stub.fig.add_subplot(111)
    stub.canvas = FigureCanvasAgg(stub.fig)
    stub._x_slider = stub._y_slider = stub._y2_slider = None
    for name in _METHODS:
        setattr(stub, name, types.MethodType(getattr(DfitApp, name), stub))
    kw = {"interactive": False} if step == "overview" else {}
    dflt = plots.RENDERERS[step](stub.ax, td, state, res, **kw)
    plots.apply_step_view(step, stub.ax, dflt, None)
    twin = stub._twin_axes()
    xl, yl = stub.ax.get_xlim(), stub.ax.get_ylim()
    y2 = twin.get_ylim() if twin is not None else None
    stub._build_sliders(full_x=xl, full_y=yl, full_y2=y2,
                        view=ViewState(xlim=xl, ylim=yl, y2lim=y2), twin=twin)
    stub._layout_sliders()
    stub.canvas.draw()

    ref_right, ref_bottom = _reference_overhangs(stub)
    assert stub._measure_overhang_px() == pytest.approx(ref_right, abs=1e-6)
    assert stub._measure_bottom_overhang_px() == pytest.approx(ref_bottom, abs=1e-6)
