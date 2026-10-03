"""ui.DfitApp wiring for the manual mask tool: controllers, hint, clear button."""

import types

from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backend_bases import MouseEvent

from dfit_tool import picks, plots
from dfit_tool.model import PickState, compute_all
from dfit_tool.ui import DfitApp
from tests.helpers import injection_state, make_testdata


def _stub(td, st, res, step="overview"):
    stub = types.SimpleNamespace()
    fig = Figure()
    stub.fig = fig
    stub.ax = fig.add_subplot(111)
    stub.canvas = FigureCanvasAgg(fig)
    plots.RENDERERS[step](stub.ax, td, st, res)
    stub.canvas.draw()
    stub.td, stub.res, stub.state, stub.step = td, res, st, step
    stub._controllers = []
    stub.hints = []
    stub.hint_lbl = types.SimpleNamespace(config=lambda **kw: stub.hints.append(kw["text"]))
    stub.refreshed = 0
    stub.refresh = lambda: setattr(stub, "refreshed", stub.refreshed + 1)
    stub._twin_axes = types.MethodType(DfitApp._twin_axes, stub)
    return stub


def _seeded():
    td = make_testdata()
    st = injection_state(td)
    picks.seed_injection(st, td)
    return td, st, compute_all(st, td)


def _names(stub):
    return [type(c).__name__ for c in stub._controllers]


def test_overview_attaches_mask_controllers_with_trim():
    td, st, res = _seeded()
    stub = _stub(td, st, res)
    DfitApp._attach_controllers(stub)
    names = _names(stub)
    assert names.count("ModifierSpanController") == 2
    assert names.count("IntervalRemoveController") == 1
    assert "DragLineController" in names
    assert "Shift+drag to mask" in stub.hints[-1]


def test_overview_attaches_mask_controllers_without_resampled_data():
    td = make_testdata()
    st = PickState(pressure_col="PRESSURE")
    stub = _stub(td, st, compute_all(st, td))
    DfitApp._attach_controllers(stub)
    names = _names(stub)
    assert names.count("ModifierSpanController") == 2
    assert "DragLineController" not in names
    assert "Shift+drag to mask" in stub.hints[-1]


def test_shift_and_ctrl_drag_commit_seconds_and_refresh():
    td, st, res = _seeded()
    stub = _stub(td, st, res)
    DfitApp._attach_controllers(stub)
    shift_c, ctrl_c = [c for c in stub._controllers
                       if isinstance(c, picks.ModifierSpanController)]
    ax, canvas = stub.ax, stub.canvas
    x0, x1 = ax.get_xlim()
    a, b = x0 + 0.2 * (x1 - x0), x0 + 0.4 * (x1 - x0)
    c_, d_ = x0 + 0.6 * (x1 - x0), x0 + 0.8 * (x1 - x0)
    y = sum(ax.get_ylim()) / 2

    def ev(name, x, key):
        px, py = ax.transData.transform((x, y))
        return MouseEvent(name, canvas, px, py, button=1, key=key)

    shift_c._on_press(ev("button_press_event", a, "shift"))
    shift_c._on_release(ev("button_release_event", b, "shift"))
    ctrl_c._on_press(ev("button_press_event", c_, "control"))
    ctrl_c._on_release(ev("button_release_event", d_, "control"))
    assert len(st.mask_intervals) == 1 and len(st.keep_intervals) == 1
    lo, hi = st.mask_intervals[0]
    assert abs(lo - a * 3600.0) < 5.0 and abs(hi - b * 3600.0) < 5.0
    klo, khi = st.keep_intervals[0]
    assert abs(klo - c_ * 3600.0) < 5.0 and abs(khi - d_ * 3600.0) < 5.0
    assert stub.refreshed == 2


def test_right_click_removes_band_and_refreshes():
    td, st, res = _seeded()
    mid_s = float(td.t_s[len(td.t_s) // 2])
    st.mask_intervals = [(mid_s - 10.0, mid_s + 10.0)]
    stub = _stub(td, st, compute_all(st, td))
    DfitApp._attach_controllers(stub)
    rc = next(c for c in stub._controllers if isinstance(c, picks.IntervalRemoveController))
    px, py = stub.ax.transData.transform((mid_s / 3600.0, sum(stub.ax.get_ylim()) / 2))
    rc._on_press(MouseEvent("button_press_event", stub.canvas, px, py, button=3))
    assert st.mask_intervals == []
    assert stub.refreshed == 1


def test_clear_button_update_and_command():
    st = PickState()
    btn_calls = []
    stub = types.SimpleNamespace(
        state=st, refreshed=0,
        btn_clear_masks=types.SimpleNamespace(config=lambda **kw: btn_calls.append(kw)))
    stub.refresh = lambda: setattr(stub, "refreshed", stub.refreshed + 1)
    DfitApp._update_masks_button(stub)
    assert btn_calls[-1] == {"text": "Clear manual masks (0)", "state": "disabled"}
    st.mask_intervals = [(1.0, 2.0), (4.0, 5.0)]
    st.keep_intervals = [(7.0, 8.0)]
    DfitApp._update_masks_button(stub)
    assert btn_calls[-1] == {"text": "Clear manual masks (3)", "state": "normal"}
    DfitApp._on_clear_masks(stub)
    assert st.mask_intervals == [] and st.keep_intervals == []
    assert stub.refreshed == 1


def test_frm_masks_packed_on_overview_only_below_notes():
    from tests.test_stiffness import _panel_visibility_stub
    stub = _panel_visibility_stub()
    for key, _ in __import__("dfit_tool.ui", fromlist=["STEPS"]).STEPS:
        stub.step = key
        stub._update_panel_visibility()
        assert stub.frm_masks.packed == (key == "overview")
    kw = stub.frm_masks.pack_calls[-1]
    assert kw.get("side") == "bottom" and kw.get("after") is stub.frm_notes
