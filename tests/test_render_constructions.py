"""Task 4: the tangent-step twinx relayout, the gid-tagged tangent constructions (segment/tick/
dashed-extension) on render_isip/render_gfunction/render_tangent, and the controller wiring that
drags them -- including the literal-ISIP seconds<->minutes coordinate reconciliation ui.py's
wiring performs (render_isip plots minutes-from-shut-in; state.isip_tangent/commit_isip_tangent
store seconds-since-file-start / psi-per-second, per interpret.literal_isip's signature).
"""

from __future__ import annotations

import types

import numpy as np
import pytest
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backend_bases import MouseEvent

from dfit_tool import picks, plots, ui
from dfit_tool.model import TangentPick, compute_all
from dfit_tool.ui import DfitApp
from tests.helpers import make_testdata, overview_state


def _seeded():
    """A PickState with every step-3/5/6 pick populated via the real seeding logic, so the
    tangent constructions under test actually have something to draw."""
    td = make_testdata()
    st = overview_state(td)
    picks.seed_overview(st, td)
    res = compute_all(st, td)
    picks.seed_isip(st, td, res)
    res = compute_all(st, td)
    picks.seed_gfunction(st, res)
    picks.seed_tangent(st, res)
    res = compute_all(st, td)
    assert st.isip_tangent is not None
    assert st.min_dpdg_G is not None and st.contact_G is not None
    assert res.eff_isip_line is not None
    assert st.closure_slope is not None and st.closure_G is not None
    return td, st, res


def _gid(ax, gid):
    return next(l for l in ax.get_lines() if l.get_gid() == gid)


# --------------------------------------------------------------------------------------------------
# render_tangent: twinx relayout
# --------------------------------------------------------------------------------------------------
def test_render_tangent_pressure_primary_gdpdg_secondary_mirrors_gfunction():
    td, st, res = _seeded()
    fig = Figure()
    ax = fig.add_subplot(111)
    defaults = plots.render_tangent(ax, td, st, res)

    press_line = ax.get_lines()[0]
    assert press_line.get_color() == "black"
    assert press_line.get_lw() == pytest.approx(1.2)
    assert press_line.get_marker() == "."
    assert press_line.get_label() == "BHP"

    twins = [a for a in fig.axes if a is not ax]
    assert len(twins) == 1
    ax2 = twins[0]
    gdpdg = next(l for l in ax2.get_lines() if l.get_label() == "G*dP/dG")
    assert gdpdg.get_color() == "tab:red"

    dg = res.diagnostics
    finite = np.isfinite(dg.GdPdG)
    hi = np.percentile(dg.GdPdG[finite], 95)
    assert defaults.y2lim == pytest.approx((0, max(hi * 1.5, 1.0)))


def test_render_tangent_closure_artists_are_gid_tagged_on_the_twin_not_primary():
    td, st, res = _seeded()
    fig = Figure()
    ax = fig.add_subplot(111)
    plots.render_tangent(ax, td, st, res)
    ax2 = next(a for a in fig.axes if a is not ax)

    seg = _gid(ax2, "closure_line_segment")
    pt = _gid(ax2, "closure_point")
    assert seg.get_xdata()[0] == pytest.approx(0.0)  # through-origin
    assert pt.get_xdata()[0] == pytest.approx(st.closure_G)
    assert all(l.get_gid() not in ("closure_line_segment", "closure_point")
              for l in ax.get_lines())


def test_render_tangent_early_return_still_returns_view_defaults():
    td = make_testdata()
    st = overview_state(td)
    st.start_idx = st.shutin_idx = None  # no te -> no diagnostics
    res = compute_all(st, td)
    fig = Figure()
    ax = fig.add_subplot(111)
    defaults = plots.render_tangent(ax, td, st, res)
    assert defaults == plots.ViewDefaults()


# --------------------------------------------------------------------------------------------------
# render_isip: literal-ISIP tangent construction
# --------------------------------------------------------------------------------------------------
def test_render_isip_tangent_construction_gids_colors_and_extension_reaches_shutin():
    td, st, res = _seeded()
    fig = Figure()
    ax = fig.add_subplot(111)
    plots.render_isip(ax, td, st, res)

    seg = _gid(ax, "isip_tangent_segment")
    tick = _gid(ax, "isip_tangent_tick")
    ext = _gid(ax, "isip_tangent_extension")
    for line in (seg, tick, ext):
        assert line.get_color() == "tab:purple"
    assert ext.get_linestyle() == "--"
    assert 0.0 in ext.get_xdata()  # dashed extension reaches the shut-in vertical (x=0 minutes)

    anchor_x_min = (st.isip_tangent.anchor_x - res.t_shutin_s) / 60.0
    assert tick.get_xdata()[0] == pytest.approx(anchor_x_min)
    assert tick.get_xdata()[1] == pytest.approx(anchor_x_min)
    assert tick.get_ydata()[0] != tick.get_ydata()[1]  # a short vertical mark, not a point


def test_render_isip_clamps_view_and_plotted_data_to_shutin_window():
    # n=1800 gives a 25-min falloff tail (shut-in at 300 s), so the +15-min clamp is binding.
    td = make_testdata(n=1800)
    st = overview_state(td)
    picks.seed_overview(st, td)
    res = compute_all(st, td)
    picks.seed_isip(st, td, res)
    res = compute_all(st, td)
    assert (td.t_s.max() - res.t_shutin_s) / 60.0 > 15.0  # raw data really extends past the clamp
    fig = Figure()
    ax = fig.add_subplot(111)
    defaults = plots.render_isip(ax, td, st, res)

    assert defaults.xlim == pytest.approx((-5.0, 15.0))
    press_line = ax.get_lines()[0]  # the BHP decline trace (first line drawn)
    xdata = press_line.get_xdata()
    assert xdata.max() <= 15.0
    assert xdata.min() >= -5.0


def test_render_isip_early_return_still_returns_view_defaults():
    td = make_testdata()
    st = overview_state(td)
    st.shutin_idx = None
    res = compute_all(st, td)
    fig = Figure()
    ax = fig.add_subplot(111)
    defaults = plots.render_isip(ax, td, st, res)
    assert defaults == plots.ViewDefaults()


# --------------------------------------------------------------------------------------------------
# render_gfunction: effective-ISIP construction (derived) + contact_point + min_dpdg_point gids
# --------------------------------------------------------------------------------------------------
def test_render_gfunction_eff_isip_construction_gids_and_extension_reaches_g_zero():
    td, st, res = _seeded()
    fig = Figure()
    ax = fig.add_subplot(111)
    plots.render_gfunction(ax, td, st, res)

    seg = _gid(ax, "eff_isip_segment")
    tick = _gid(ax, "eff_isip_tick")
    ext = _gid(ax, "eff_isip_extension")
    contact = _gid(ax, "contact_point")
    for line in (seg, tick, ext):
        assert line.get_color() == "tab:green"
    assert ext.get_linestyle() == "--"
    assert 0.0 in ext.get_xdata()
    assert contact.get_xdata()[0] == pytest.approx(st.contact_G)


def test_render_gfunction_min_dpdg_point_gid_on_twin_axis():
    """The min-dP/dG marker is a real gid-tagged artist on the twin (dP/dG) axis. The eff-ISIP
    construction gids (asserted above) are still drawn from the derived res.eff_isip_line, but no
    AnchorLineController wiring exists for them anywhere in ui.py any more -- see
    test_gfunction_wiring_attaches_two_point_controllers_sharing_one_gate below, which asserts
    the step's only two controllers are both DraggablePointControllers."""
    td, st, res = _seeded()
    fig = Figure()
    ax = fig.add_subplot(111)
    plots.render_gfunction(ax, td, st, res)
    ax2 = next(a for a in fig.axes if a is not ax)

    marker = _gid(ax2, "min_dpdg_point")
    assert marker.get_xdata()[0] == pytest.approx(st.min_dpdg_G)
    assert marker.get_marker() == "v"
    assert marker.get_color() == "tab:red"


# --------------------------------------------------------------------------------------------------
# twinx interaction regressions
# --------------------------------------------------------------------------------------------------
def test_gfunction_dpdg_twin_owns_inaxes_but_contact_point_controller_still_hit_tests_by_pixel():
    """The contact-point DraggablePointController lives on the primary (P-vs-G) axis, but the
    later-created dP/dG twin sits on top and owns ``event.inaxes`` over the shared region -- the
    same twin-owns-inaxes hazard ``test_overview_rate_twin_owns_inaxes_regression`` guards for
    the overview's start/shut-in lines. The controller must still capture via its own pixel bbox."""
    td, st, res = _seeded()
    fig = Figure()
    ax = fig.add_subplot(111)
    canvas = FigureCanvasAgg(fig)
    plots.render_gfunction(ax, td, st, res)
    canvas.draw()
    ax2 = next(a for a in fig.axes if a is not ax)

    pt = _gid(ax, "contact_point")
    px, py = ax.transData.transform((pt.get_xdata()[0], pt.get_ydata()[0]))
    ev = MouseEvent("button_press_event", canvas, px, py, button=1)
    assert ev.inaxes is ax2
    assert ev.inaxes is not ax

    calls = []
    ctrl = picks.DraggablePointController(canvas, ax, "contact_point", res.diagnostics.G,
                                          res.resampled.p, commit_fn=calls.append)
    ctrl._on_press(ev)
    assert ctrl._dragging is True  # captured despite inaxes pointing at the twin


def test_tangent_closure_point_controller_on_the_twin_hit_tests_correctly():
    """The tangent step's closure controllers live on ax2 (the G*dP/dG twin) itself. Empirically
    ax2 -- created after the primary axis via ``ax.twinx()`` -- is the one matplotlib assigns
    ``event.inaxes`` to over the shared region (same "later-created axes wins" rule as the
    overview/gfunction twins above), so this is the safe direction; the assertion below confirms
    the wiring still works from a real ``render_tangent`` figure rather than assuming it."""
    td, st, res = _seeded()
    fig = Figure()
    ax = fig.add_subplot(111)
    canvas = FigureCanvasAgg(fig)
    plots.render_tangent(ax, td, st, res)
    canvas.draw()
    ax2 = next(a for a in fig.axes if a is not ax)

    pt = _gid(ax2, "closure_point")
    px, py = ax2.transData.transform((pt.get_xdata()[0], pt.get_ydata()[0]))
    ev = MouseEvent("button_press_event", canvas, px, py, button=1)
    assert ev.inaxes is ax2

    dg = res.diagnostics
    calls = []
    ctrl = picks.DraggablePointController(canvas, ax2, "closure_point", dg.G, dg.GdPdG,
                                          commit_fn=calls.append)
    ctrl._on_press(ev)
    assert ctrl._dragging is True


# --------------------------------------------------------------------------------------------------
# ISIP minutes<->seconds coordinate reconciliation (pure functions)
# --------------------------------------------------------------------------------------------------
def test_isip_pick_in_minutes_converts_units():
    pick = TangentPick(anchor_x=1230.0, anchor_y=4500.0, slope=-2.0)  # psi/s, seconds-abscissa
    t_shutin_s = 1200.0
    converted = ui._isip_pick_in_minutes(pick, t_shutin_s)
    assert converted.anchor_x == pytest.approx(0.5)      # (1230 - 1200) / 60
    assert converted.anchor_y == pytest.approx(4500.0)   # psi unchanged
    assert converted.slope == pytest.approx(-120.0)      # -2 psi/s * 60 s/min


def test_isip_pick_in_minutes_passes_through_none():
    assert ui._isip_pick_in_minutes(None, 100.0) is None


def test_isip_minutes_to_seconds_is_the_exact_inverse():
    pick = TangentPick(anchor_x=1230.0, anchor_y=4500.0, slope=-2.0)
    t_shutin_s = 1200.0
    converted = ui._isip_pick_in_minutes(pick, t_shutin_s)
    back_x, back_slope = ui._isip_minutes_to_seconds(converted.anchor_x, converted.slope,
                                                      t_shutin_s)
    assert back_x == pytest.approx(pick.anchor_x)
    assert back_slope == pytest.approx(pick.slope)


# --------------------------------------------------------------------------------------------------
# ui.py controller wiring (end to end against the real _attach_controllers closures)
# --------------------------------------------------------------------------------------------------
def _stub(td, st, res, step):
    stub = types.SimpleNamespace()
    fig = Figure()
    stub.fig = fig
    stub.ax = fig.add_subplot(111)
    stub.canvas = FigureCanvasAgg(fig)
    plots.RENDERERS[step](stub.ax, td, st, res)
    stub.canvas.draw()
    stub.td = td
    stub.res = res
    stub.state = st
    stub.step = step
    stub._controllers = []
    stub.hint_lbl = types.SimpleNamespace(config=lambda **kw: None)
    stub.refresh = lambda: None
    stub._twin_axes = types.MethodType(DfitApp._twin_axes, stub)
    return stub


def test_isip_wiring_get_pick_and_commit_round_trip_seconds_through_axes_minutes():
    td, st, res = _seeded()
    stub = _stub(td, st, res, "isip")
    DfitApp._attach_controllers(stub)
    assert len(stub._controllers) == 2  # AnchorLineController + its HoverCursorController
    ctrl = stub._controllers[0]

    p_min = ctrl.get_pick()
    assert p_min.anchor_x == pytest.approx((st.isip_tangent.anchor_x - res.t_shutin_s) / 60.0)
    assert p_min.anchor_y == pytest.approx(st.isip_tangent.anchor_y)
    assert p_min.slope == pytest.approx(st.isip_tangent.slope * 60.0)

    orig_slope = st.isip_tangent.slope
    # "body" (pan) commit reported in the controller's own (axes-minutes) coordinates must land
    # the anchor back on state.isip_tangent in seconds-since-file-start / psi; slope is untouched
    # by a pan (commit_isip_tangent's "body" branch keeps the previously stored slope).
    ctrl.commit_fn("body", 5.0, 4321.0, 999.0)
    assert st.isip_tangent.anchor_x == pytest.approx(5.0 * 60.0 + res.t_shutin_s)
    assert st.isip_tangent.anchor_y == pytest.approx(4321.0)
    assert st.isip_tangent.slope == pytest.approx(orig_slope)

    # "end" (rotate) commit's reported slope (axes-minutes, psi/min) must convert back to the
    # stored psi/s convention.
    ctrl.commit_fn("end", 0.0, 0.0, 12.0)
    assert st.isip_tangent.slope == pytest.approx(12.0 / 60.0)


def test_isip_wiring_no_op_when_bhp_or_shutin_missing():
    td = make_testdata()
    st = overview_state(td)
    st.shutin_idx = None
    res = compute_all(st, td)
    stub = _stub(td, st, res, "isip")
    DfitApp._attach_controllers(stub)
    assert stub._controllers == []


def test_gfunction_wiring_attaches_two_point_controllers_sharing_one_gate():
    td, st, res = _seeded()
    stub = _stub(td, st, res, "gfunction")
    DfitApp._attach_controllers(stub)
    assert len(stub._controllers) == 3  # min-dP/dG point + contact point + hover
    min_dpdg_ctrl, contact_ctrl, hover_ctrl = stub._controllers
    assert isinstance(min_dpdg_ctrl, picks.DraggablePointController)
    assert isinstance(contact_ctrl, picks.DraggablePointController)
    assert min_dpdg_ctrl.gate is contact_ctrl.gate
    assert isinstance(hover_ctrl, picks.HoverCursorController)
    ax2 = stub._twin_axes()
    assert min_dpdg_ctrl.ax is ax2
    assert contact_ctrl.ax is stub.ax

    contact_ctrl.commit_fn(3.5)
    assert st.contact_G == pytest.approx(3.5)

    min_dpdg_ctrl.commit_fn(4.2)
    assert st.min_dpdg_G == pytest.approx(4.2)


def test_gfunction_wiring_no_op_when_diagnostics_missing():
    td = make_testdata()
    st = overview_state(td)
    st.start_idx = st.shutin_idx = None
    res = compute_all(st, td)
    stub = _stub(td, st, res, "gfunction")
    DfitApp._attach_controllers(stub)
    assert stub._controllers == []


def test_tangent_wiring_attaches_to_the_twin_axes_sharing_one_gate():
    td, st, res = _seeded()
    stub = _stub(td, st, res, "tangent")
    DfitApp._attach_controllers(stub)
    assert len(stub._controllers) == 3  # AnchorLineController + DraggablePointController + hover
    anchor_ctrl, point_ctrl, hover_ctrl = stub._controllers
    assert anchor_ctrl.gate is point_ctrl.gate
    assert isinstance(hover_ctrl, picks.HoverCursorController)
    ax2 = stub._twin_axes()
    assert anchor_ctrl.ax is ax2
    assert point_ctrl.ax is ax2

    pick = anchor_ctrl.get_pick()
    assert (pick.anchor_x, pick.anchor_y) == (0.0, 0.0)
    assert pick.slope == pytest.approx(st.closure_slope)

    point_ctrl.commit_fn(2.5)
    assert st.closure_G == pytest.approx(2.5)


def test_tangent_wiring_no_op_when_diagnostics_missing():
    td = make_testdata()
    st = overview_state(td)
    st.start_idx = st.shutin_idx = None
    res = compute_all(st, td)
    stub = _stub(td, st, res, "tangent")
    DfitApp._attach_controllers(stub)
    assert stub._controllers == []
