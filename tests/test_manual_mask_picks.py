"""Manual mask/keep interval helpers and controllers (picks.py), driven with real MouseEvents."""

import pytest
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.backend_bases import MouseEvent

from dfit_tool import picks
from dfit_tool.model import PickState


# ---- pure helpers ---------------------------------------------------------------------------------
def _st(mask=(), keep=()):
    return PickState(mask_intervals=list(mask), keep_intervals=list(keep))


def test_commit_merges_overlapping_touching_contained_and_sorts():
    st = _st()
    picks.commit_mask_interval(st, "mask", 50.0, 60.0)
    picks.commit_mask_interval(st, "mask", 10.0, 20.0)
    assert st.mask_intervals == [(10.0, 20.0), (50.0, 60.0)]
    picks.commit_mask_interval(st, "mask", 15.0, 30.0)       # overlap
    assert st.mask_intervals == [(10.0, 30.0), (50.0, 60.0)]
    picks.commit_mask_interval(st, "mask", 30.0, 40.0)       # touching
    assert st.mask_intervals == [(10.0, 40.0), (50.0, 60.0)]
    picks.commit_mask_interval(st, "mask", 12.0, 14.0)       # contained
    assert st.mask_intervals == [(10.0, 40.0), (50.0, 60.0)]
    picks.commit_mask_interval(st, "mask", 5.0, 100.0)       # bridges everything
    assert st.mask_intervals == [(5.0, 100.0)]


def test_commit_sorts_swapped_bounds_and_keep_kind():
    st = _st()
    picks.commit_mask_interval(st, "keep", 9.0, 3.0)
    assert st.keep_intervals == [(3.0, 9.0)]
    assert st.mask_intervals == []


def test_commit_subtracts_from_other_kind_straddle_split():
    st = _st(keep=[(0.0, 100.0)])
    picks.commit_mask_interval(st, "mask", 40.0, 60.0)
    assert st.keep_intervals == [(0.0, 40.0), (60.0, 100.0)]
    assert st.mask_intervals == [(40.0, 60.0)]


def test_commit_subtracts_full_cover_and_partial_edges():
    st = _st(mask=[(10.0, 20.0), (30.0, 50.0), (70.0, 90.0)])
    picks.commit_mask_interval(st, "keep", 5.0, 40.0)
    # (10,20) fully covered, (30,50) left edge cut, (70,90) untouched
    assert st.mask_intervals == [(40.0, 50.0), (70.0, 90.0)]
    assert st.keep_intervals == [(5.0, 40.0)]


def test_commit_drops_zero_width_pieces():
    st = _st(keep=[(0.0, 10.0)])
    picks.commit_mask_interval(st, "mask", 10.0, 20.0)        # touches only at 10
    assert st.keep_intervals == [(0.0, 10.0)]
    picks.commit_mask_interval(st, "mask", 0.0, 10.0)         # exact cover
    assert st.keep_intervals == []


@pytest.mark.parametrize("lo,hi", [(5.0, 5.0), (float("nan"), 3.0), (3.0, float("inf")),
                                   (float("-inf"), 3.0)])
def test_commit_ignores_degenerate_input(lo, hi):
    st = _st(mask=[(1.0, 2.0)], keep=[(8.0, 9.0)])
    picks.commit_mask_interval(st, "mask", lo, hi)
    assert st.mask_intervals == [(1.0, 2.0)]
    assert st.keep_intervals == [(8.0, 9.0)]


def test_lists_never_overlap_invariant():
    st = _st()
    for kind, lo, hi in [("mask", 0, 50), ("keep", 20, 70), ("mask", 60, 90), ("keep", 10, 30)]:
        picks.commit_mask_interval(st, kind, float(lo), float(hi))
    for a_lo, a_hi in st.mask_intervals:
        for b_lo, b_hi in st.keep_intervals:
            assert min(a_hi, b_hi) <= max(a_lo, b_lo)


def test_interval_at():
    iv = [(1.0, 2.0), (5.0, 6.0)]
    assert picks.interval_at(iv, 1.5) == 0
    assert picks.interval_at(iv, 5.0) == 1
    assert picks.interval_at(iv, 6.0) == 1
    assert picks.interval_at(iv, 3.0) is None
    assert picks.interval_at([], 3.0) is None


def test_remove_interval_and_clear():
    st = _st(mask=[(1.0, 2.0), (5.0, 6.0)], keep=[(8.0, 9.0)])
    picks.remove_interval(st, "mask", 0)
    assert st.mask_intervals == [(5.0, 6.0)]
    picks.remove_interval(st, "keep", 0)
    assert st.keep_intervals == []
    picks.remove_interval(st, "mask", 7)   # out of range: no-op
    assert st.mask_intervals == [(5.0, 6.0)]
    st.keep_intervals = [(1.0, 2.0)]
    picks.clear_manual_masks(st)
    assert st.mask_intervals == [] and st.keep_intervals == []


# ---- controllers ----------------------------------------------------------------------------------
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


def _drag(ctrl, canvas, ax, key, x0=2.0, x1=6.0):
    ctrl._on_press(_event("button_press_event", canvas, ax, x0, key=key))
    ctrl._on_motion(_event("motion_notify_event", canvas, ax, x1, key=key))
    ctrl._on_release(_event("button_release_event", canvas, ax, x1, key=key))


def _pair(gate=None):
    fig, ax, canvas = _built_axes()
    shift_got, ctrl_got = [], []
    gate = gate or picks._CaptureGate()
    sc = picks.ModifierSpanController(canvas, ax, lambda lo, hi: shift_got.append((lo, hi)),
                                      modifier="shift", exclude=("ctrl",), gate=gate)
    cc = picks.ModifierSpanController(canvas, ax, lambda lo, hi: ctrl_got.append((lo, hi)),
                                      modifier="ctrl", exclude=("shift",), gate=gate)
    return fig, ax, canvas, sc, cc, shift_got, ctrl_got


def test_shift_drag_fires_shift_controller_only():
    fig, ax, canvas, sc, cc, shift_got, ctrl_got = _pair()
    _drag(sc, canvas, ax, "shift")
    _drag(cc, canvas, ax, "shift")
    assert len(shift_got) == 1 and ctrl_got == []


def test_ctrl_drag_fires_ctrl_controller_only_and_control_alias():
    fig, ax, canvas, sc, cc, shift_got, ctrl_got = _pair()
    for key in ("ctrl", "control"):
        _drag(sc, canvas, ax, key)
        _drag(cc, canvas, ax, key)
    assert shift_got == [] and len(ctrl_got) == 2


def test_ctrl_shift_fires_neither():
    fig, ax, canvas, sc, cc, shift_got, ctrl_got = _pair()
    for key in ("shift+control", "ctrl+shift"):
        _drag(sc, canvas, ax, key)
        _drag(cc, canvas, ax, key)
    assert shift_got == [] and ctrl_got == []


def test_modifiers_set_also_normalizes_control():
    fig, ax, canvas, sc, cc, shift_got, ctrl_got = _pair()
    px, py = ax.transData.transform((2.0, 5.0))
    cc._on_press(MouseEvent("button_press_event", canvas, px, py, button=1,
                            modifiers={"control"}))
    assert cc._press_x is not None


def test_exclude_default_empty_keeps_existing_behavior():
    fig, ax, canvas = _built_axes()
    got = []
    c = picks.ModifierSpanController(canvas, ax, lambda lo, hi: got.append((lo, hi)))
    _drag(c, canvas, ax, "shift+control")
    assert len(got) == 1


def test_remove_controller_hit_miss_and_button():
    fig, ax, canvas = _built_axes()
    removed = []
    spans = [("mask", 0, 1.0, 3.0), ("keep", 2, 6.0, 8.0)]
    ctrl = picks.IntervalRemoveController(canvas, ax, lambda: spans,
                                          lambda k, i: removed.append((k, i)))
    for x, button in ((2.0, 3), (7.0, 3), (4.5, 3), (2.0, 1)):
        ctrl._on_press(_event("button_press_event", canvas, ax, x, button=button))
        ctrl._on_release(_event("button_release_event", canvas, ax, x, button=button))
    assert removed == [("mask", 0), ("keep", 2)]


def test_remove_controller_gate_claimed_then_released():
    fig, ax, canvas = _built_axes()
    gate = picks._CaptureGate()
    ctrl = picks.IntervalRemoveController(canvas, ax, lambda: [("mask", 0, 1.0, 3.0)],
                                          lambda k, i: None, gate=gate)
    ctrl._on_press(_event("button_press_event", canvas, ax, 2.0, button=3))
    assert gate._owner is ctrl
    ctrl._on_release(_event("button_release_event", canvas, ax, 2.0, button=3))
    assert gate._owner is None


def test_remove_controller_miss_does_not_claim_and_blocked_by_other_owner():
    fig, ax, canvas = _built_axes()
    gate = picks._CaptureGate()
    removed = []
    ctrl = picks.IntervalRemoveController(canvas, ax, lambda: [("mask", 0, 1.0, 3.0)],
                                          lambda k, i: removed.append((k, i)), gate=gate)
    ctrl._on_press(_event("button_press_event", canvas, ax, 8.0, button=3))
    assert gate._owner is None
    gate.try_claim(object())
    ctrl._on_press(_event("button_press_event", canvas, ax, 2.0, button=3))
    assert removed == []


def test_remove_controller_hover_probes_and_disconnect():
    fig, ax, canvas = _built_axes()
    ctrl = picks.IntervalRemoveController(canvas, ax, lambda: [], lambda k, i: None)
    ev = _event("motion_notify_event", canvas, ax, 2.0)
    assert ctrl.hover_kind(ev) is None and ctrl.active_kind() is None
    picks.HoverCursorController(canvas, [ctrl])._on_motion(ev)   # must not raise
    ctrl.disconnect()
    assert ctrl._cids == []
