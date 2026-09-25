"""Unit tests for ``picks.ModifierSpanController`` -- the Shift+drag window-correction gesture on
the G-function step. Modeled on tests/test_drag_controller.py's press/motion/release pattern
(synthesized ``MouseEvent``s); gate-contention cases mirror tests/test_capture_gate.py. A held
modifier is simulated with ``key="shift"``.
"""

import numpy as np
import pytest
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backend_bases import MouseEvent

from dfit_tool import picks, plots
from dfit_tool.model import compute_all
from tests.helpers import make_testdata, injection_state


def _built_axes():
    fig = Figure(figsize=(6.4, 4.8), dpi=100)
    ax = fig.add_subplot(111)
    canvas = FigureCanvasAgg(fig)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_autoscale_on(False)
    canvas.draw()
    return fig, ax, canvas


def _event(name, canvas, ax, xdata, ydata=5.0, button=1, key=None):
    px, py = ax.transData.transform((xdata, ydata))
    return MouseEvent(name, canvas, px, py, button=button, key=key)


def test_shift_press_drag_release_fires_on_span_sorted_and_removes_patch():
    fig, ax, canvas = _built_axes()
    got = []
    ctrl = picks.ModifierSpanController(canvas, ax, lambda lo, hi: got.append((lo, hi)))

    ctrl._on_press(_event("button_press_event", canvas, ax, 7.0, key="shift"))
    assert ctrl._press_x is not None
    ctrl._on_motion(_event("motion_notify_event", canvas, ax, 3.0, key="shift"))
    assert ctrl._patch is not None
    ctrl._on_release(_event("button_release_event", canvas, ax, 3.0, key="shift"))
    assert ctrl._patch is None
    assert len(got) == 1
    lo, hi = got[0]  # sorted, regardless of drag direction (pixel round-trip: approximate)
    assert lo == pytest.approx(3.0, abs=0.05)
    assert hi == pytest.approx(7.0, abs=0.05)


def test_shift_via_event_modifiers_captures_without_canvas_focus():
    # An unfocused Tk canvas never sees the Shift key_press, so event.key stays None; the
    # mouse event's own modifier state (event.modifiers) must still arm the span.
    fig, ax, canvas = _built_axes()
    got = []
    ctrl = picks.ModifierSpanController(canvas, ax, lambda lo, hi: got.append((lo, hi)))

    px, py = ax.transData.transform((7.0, 5.0))
    ctrl._on_press(MouseEvent("button_press_event", canvas, px, py, button=1,
                              modifiers={"shift"}))
    assert ctrl._press_x is not None
    ctrl._on_release(_event("button_release_event", canvas, ax, 3.0))
    assert len(got) == 1


def test_plain_press_never_captures():
    fig, ax, canvas = _built_axes()
    got = []
    ctrl = picks.ModifierSpanController(canvas, ax, lambda lo, hi: got.append((lo, hi)))

    ctrl._on_press(_event("button_press_event", canvas, ax, 7.0, key=None))
    assert ctrl._press_x is None
    ctrl._on_motion(_event("motion_notify_event", canvas, ax, 3.0, key=None))
    assert ctrl._patch is None
    ctrl._on_release(_event("button_release_event", canvas, ax, 3.0, key=None))
    assert got == []


def test_zero_width_span_does_not_fire_on_span():
    fig, ax, canvas = _built_axes()
    got = []
    ctrl = picks.ModifierSpanController(canvas, ax, lambda lo, hi: got.append((lo, hi)))

    ctrl._on_press(_event("button_press_event", canvas, ax, 5.0, key="shift"))
    ctrl._on_release(_event("button_release_event", canvas, ax, 5.0, key="shift"))
    assert got == []


def test_guard_blocks_capture():
    fig, ax, canvas = _built_axes()
    ctrl = picks.ModifierSpanController(canvas, ax, lambda lo, hi: None, guard=lambda: True)
    ctrl._on_press(_event("button_press_event", canvas, ax, 5.0, key="shift"))
    assert ctrl._press_x is None


def test_disconnect_unbinds_all_callbacks_and_removes_patch():
    fig, ax, canvas = _built_axes()
    ctrl = picks.ModifierSpanController(canvas, ax, lambda lo, hi: None)
    ctrl._on_press(_event("button_press_event", canvas, ax, 5.0, key="shift"))
    ctrl._on_motion(_event("motion_notify_event", canvas, ax, 6.0, key="shift"))
    assert ctrl._patch is not None
    ctrl.disconnect()
    assert ctrl._cids == []
    assert ctrl._patch is None


def test_custom_modifier_name_is_honored():
    fig, ax, canvas = _built_axes()
    got = []
    ctrl = picks.ModifierSpanController(canvas, ax, lambda lo, hi: got.append((lo, hi)),
                                        modifier="control")
    ctrl._on_press(_event("button_press_event", canvas, ax, 3.0, key="shift"))
    assert ctrl._press_x is None  # wrong modifier held
    ctrl._on_press(_event("button_press_event", canvas, ax, 3.0, key="control"))
    assert ctrl._press_x is not None


# --------------------------------------------------------------------------------------------------
# gate contention -- mirrors tests/test_capture_gate.py's shared-gate contract
# --------------------------------------------------------------------------------------------------
def _built_shared_marker_axes():
    fig, ax, canvas = _built_axes()
    ax.plot([5.0], [5.0], "o", gid="pt_a")
    canvas.draw()
    return fig, ax, canvas


def test_shift_press_blocks_a_draggable_point_controller_sharing_the_gate():
    fig, ax, canvas = _built_shared_marker_axes()
    curve_x = np.linspace(0.0, 10.0, 11)
    curve_y = np.linspace(0.0, 10.0, 11)
    gate = picks._CaptureGate()
    got_point = []

    span_ctrl = picks.ModifierSpanController(canvas, ax, lambda lo, hi: None, gate=gate)
    point_ctrl = picks.DraggablePointController(canvas, ax, "pt_a", curve_x, curve_y,
                                                commit_fn=got_point.append, gate=gate)

    ev = _event("button_press_event", canvas, ax, 5.0, ydata=5.0, key="shift")
    span_ctrl._on_press(ev)  # registered/pressed first, as in ui.py's wiring order
    assert span_ctrl._press_x is not None
    assert gate._owner is span_ctrl

    point_ctrl._on_press(ev)
    assert point_ctrl._dragging is False  # gate already claimed by span_ctrl


def test_plain_press_leaves_the_point_controller_working_when_sharing_the_gate():
    fig, ax, canvas = _built_shared_marker_axes()
    curve_x = np.linspace(0.0, 10.0, 11)
    curve_y = np.linspace(0.0, 10.0, 11)
    gate = picks._CaptureGate()
    got_point = []

    span_ctrl = picks.ModifierSpanController(canvas, ax, lambda lo, hi: None, gate=gate)
    point_ctrl = picks.DraggablePointController(canvas, ax, "pt_a", curve_x, curve_y,
                                                commit_fn=got_point.append, gate=gate)

    ev = _event("button_press_event", canvas, ax, 5.0, ydata=5.0, key=None)
    span_ctrl._on_press(ev)
    assert span_ctrl._press_x is None  # no modifier held -- never even tries to claim the gate

    point_ctrl._on_press(ev)
    assert point_ctrl._dragging is True
    assert gate._owner is point_ctrl


# --------------------------------------------------------------------------------------------------
# FIX 1 regression -- HoverCursorController probes every controller's active_kind()/hover_kind()
# on every motion event (picks.py's _resolve_kind); ModifierSpanController used to expose neither,
# so wiring it into ui.py's gfunction hover controller (as ui.py actually does) raised
# AttributeError on the very first mouse move over the step.
# --------------------------------------------------------------------------------------------------
def test_hover_controller_over_span_and_point_controllers_does_not_raise_on_motion():
    td = make_testdata()
    st = injection_state(td)
    picks.seed_injection(st, td)
    res = compute_all(st, td)
    picks.seed_isip(st, td, res)
    res = compute_all(st, td)
    picks.seed_gfunction(st, res)
    st.closure_scenario = "C-A clear"
    res = compute_all(st, td)
    assert st.min_dpdg_G is not None

    fig = Figure()
    ax = fig.add_subplot(111)
    plots.render_gfunction(ax, td, st, res)
    canvas = FigureCanvasAgg(fig)
    canvas.draw()

    ax2 = next(a for a in fig.axes if a is not ax)  # dP/dG twin -- carries the min_dpdg_point
    dg = res.diagnostics
    span_ctrl = picks.ModifierSpanController(canvas, ax2, lambda lo, hi: None)
    point_ctrl = picks.DraggablePointController(canvas, ax2, "min_dpdg_point", dg.G, dg.dPdG,
                                                commit_fn=lambda x: None)
    hover_ctrl = picks.HoverCursorController(canvas, [span_ctrl, point_ctrl])

    marker = next(l for l in ax2.get_lines() if l.get_gid() == "min_dpdg_point")
    mx, my = marker.get_xdata()[0], marker.get_ydata()[0]
    px, py = ax2.transData.transform((mx, my))
    ev = MouseEvent("motion_notify_event", canvas, px, py)

    hover_ctrl._on_motion(ev)  # must not raise (the AttributeError this test guards against)
    assert hover_ctrl._resolve_kind(ev) == "point"
