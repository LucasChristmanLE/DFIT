"""The tail-trim tool, moved from the G-function step to the Overview step behind an ephemeral
"Show trim tool" toggle (CLAUDE.md TODO). ``PickState.tail_trim_dt`` semantics (shut-in-relative
seconds) and ``model.compute_all``'s masking are unchanged -- see tests/test_tail_trim.py, which
covers those and is untouched by this move. This file covers the new home for the tool:
``plots.render_overview``'s ``show_trim`` kwarg, ``ui.DfitApp``'s overview controller wiring
(the ``x_hours``-coordinate commit closure), and the toggle/panel-visibility plumbing.
"""

from __future__ import annotations

import types

import numpy as np
import pytest
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from dfit_tool import picks, plots, resample
from dfit_tool.model import DerivedResults, PickState, compute_all
from dfit_tool.ui import DfitApp
from tests.helpers import make_testdata, injection_state, pre_crash_trim_dt


def _gid(ax, gid):
    return next(l for l in ax.get_lines() if l.get_gid() == gid)


def _seeded_with_crash(trim_dt=None):
    """A PickState with the injection window seeded and a monotone zero-crash in the
    post-shut-in tail (helpers.make_testdata's zero_crash_at), optionally trimmed. Mirrors
    test_render_constructions.py's (now-removed) gfunction fixture of the same name."""
    td = make_testdata(n=1200, zero_crash_at=0.5)
    st = injection_state(td)
    picks.seed_injection(st, td)
    res = compute_all(st, td)
    picks.seed_isip(st, td, res)
    res = compute_all(st, td)
    if trim_dt is not None:
        st.tail_trim_dt = trim_dt
        res = compute_all(st, td)
    return td, st, res


# --------------------------------------------------------------------------------------------------
# render_overview: tail_excluded (informational, gated on a trim being set) vs. tail_trim
# (the draggable line, gated on show_trim)
# --------------------------------------------------------------------------------------------------
def test_render_overview_trim_set_show_false_draws_excluded_not_trim_line():
    td, st, res = _seeded_with_crash()
    last_normal_dt = pre_crash_trim_dt(res)
    st.tail_trim_dt = last_normal_dt
    res2 = compute_all(st, td)
    fig = Figure()
    ax = fig.add_subplot(111)
    plots.render_overview(ax, td, st, res2, show_trim=False)

    t_trim_h = (last_normal_dt + res2.t_shutin_s) / 3600.0
    # The main pressure trace (no gid) is split at the trim time -- it never reaches past it.
    main = next(l for l in ax.get_lines() if l.get_gid() is None)
    assert main.get_xdata().max() <= t_trim_h

    # The excluded tail is the real raw trace's remainder, not a hidden overlay: it starts right
    # at the trim boundary and runs all the way to the record's last raw sample.
    tail = _gid(ax, "tail_excluded")
    tail_x = tail.get_xdata()
    assert len(tail_x)
    last_raw_h = float(td.t_s[-1]) / 3600.0
    assert t_trim_h < tail_x.min() < t_trim_h + 0.01
    assert tail_x.max() == pytest.approx(last_raw_h)
    assert not any(l.get_gid() == "tail_trim" for l in ax.get_lines())


def test_render_overview_trim_set_show_true_draws_trim_line_at_trim_x():
    td, st, res = _seeded_with_crash()
    last_normal_dt = pre_crash_trim_dt(res)
    st.tail_trim_dt = last_normal_dt
    res2 = compute_all(st, td)
    fig = Figure()
    ax = fig.add_subplot(111)
    plots.render_overview(ax, td, st, res2, show_trim=True)

    vline = _gid(ax, "tail_trim")
    assert vline.get_xdata()[0] == pytest.approx((last_normal_dt + res2.t_shutin_s) / 3600.0)
    # The excluded-tail preview is informational and still drawn regardless of show_trim.
    assert _gid(ax, "tail_excluded") is not None


def test_render_overview_no_trim_show_true_vline_at_last_sample_no_excluded_tail():
    td, st, res = _seeded_with_crash()  # no trim set
    fig = Figure()
    ax = fig.add_subplot(111)
    plots.render_overview(ax, td, st, res, show_trim=True)

    vline = _gid(ax, "tail_trim")
    expected_x = (float(res.resampled_full.dt[-1]) + res.t_shutin_s) / 3600.0
    assert vline.get_xdata()[0] == pytest.approx(expected_x)
    assert not any(l.get_gid() == "tail_excluded" for l in ax.get_lines())
    # With no trim, the main trace is unsplit and reaches the full record's last raw sample.
    main = next(l for l in ax.get_lines() if l.get_gid() is None)
    last_raw_h = float(td.t_s[-1]) / 3600.0
    assert main.get_xdata().max() == pytest.approx(last_raw_h)


def test_render_overview_no_resampled_full_no_gids_no_crash():
    td = make_testdata()
    st = PickState(pressure_col="PRESSURE")  # no start/shutin picked -> no t_shutin_s/resampled_full
    res = compute_all(st, td)
    fig = Figure()
    ax = fig.add_subplot(111)
    plots.render_overview(ax, td, st, res, show_trim=True)

    assert all(l.get_gid() not in ("tail_excluded", "tail_trim") for l in ax.get_lines())


# --------------------------------------------------------------------------------------------------
# export: render_step_figure never passes show_trim, so an exported PNG never carries the line
# --------------------------------------------------------------------------------------------------
def test_render_step_figure_overview_never_draws_trim_line_but_draws_excluded_tail():
    td, st, res = _seeded_with_crash()
    st.tail_trim_dt = pre_crash_trim_dt(res)
    res2 = compute_all(st, td)

    fig = plots.render_step_figure("overview", td, st, res2)
    ax = fig.axes[0]
    assert not any(l.get_gid() == "tail_trim" for l in ax.get_lines())
    assert any(l.get_gid() == "tail_excluded" for l in ax.get_lines())


# --------------------------------------------------------------------------------------------------
# ui.py controller wiring (end to end against the real _attach_controllers closure)
# --------------------------------------------------------------------------------------------------
def _stub(td, st, res, step, show_trim=False):
    """Same duck-typed stand-in convention as test_render_constructions.py's ``_stub``, plus the
    ``show_trim`` ephemeral toggle _attach_controllers' overview branch reads."""
    stub = types.SimpleNamespace()
    fig = Figure()
    stub.fig = fig
    stub.ax = fig.add_subplot(111)
    stub.canvas = FigureCanvasAgg(fig)
    stub.show_trim = show_trim
    kwargs = {"show_trim": show_trim} if step == "overview" else {}
    plots.RENDERERS[step](stub.ax, td, st, res, **kwargs)
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


def test_overview_wiring_show_trim_true_attaches_drag_and_hover():
    td, st, res = _seeded_with_crash()
    stub = _stub(td, st, res, "overview", show_trim=True)
    DfitApp._attach_controllers(stub)
    assert len(stub._controllers) == 2
    ctrl, hover_ctrl = stub._controllers
    assert isinstance(ctrl, picks.DragLineController)
    assert isinstance(hover_ctrl, picks.HoverCursorController)


def test_overview_wiring_show_trim_false_no_controllers():
    td, st, res = _seeded_with_crash()
    stub = _stub(td, st, res, "overview", show_trim=False)
    DfitApp._attach_controllers(stub)
    assert stub._controllers == []


def test_overview_wiring_no_resampled_full_no_controllers_even_with_show_trim():
    td = make_testdata()
    st = PickState(pressure_col="PRESSURE")  # no start/shutin -> no t_shutin_s/resampled_full
    res = compute_all(st, td)
    stub = _stub(td, st, res, "overview", show_trim=True)
    DfitApp._attach_controllers(stub)
    assert stub._controllers == []


# --------------------------------------------------------------------------------------------------
# commit closure (moved from the old gfunction wiring tests, adapted to x_hours coordinates:
# x_hours = (dt + t_shutin_s) / 3600)
# --------------------------------------------------------------------------------------------------
def _trim_commit(td, st, res):
    stub = _stub(td, st, res, "overview", show_trim=True)
    DfitApp._attach_controllers(stub)
    trim_ctrl = next(c for c in stub._controllers if isinstance(c, picks.DragLineController))
    return stub, trim_ctrl.handlers["tail_trim"]


def test_overview_trim_commit_release_at_last_sample_clears():
    td, st, res = _seeded_with_crash()
    stub, commit = _trim_commit(td, st, res)
    x_hours = (float(res.resampled_full.dt[-1]) + res.t_shutin_s) / 3600.0
    commit(x_hours)
    assert st.tail_trim_dt is None


def test_overview_trim_commit_release_mid_record_snaps_to_nearest_sample():
    td, st, res = _seeded_with_crash()
    stub, commit = _trim_commit(td, st, res)
    mid = len(res.resampled_full.dt) // 2
    x_hours = (float(res.resampled_full.dt[mid]) + res.t_shutin_s) / 3600.0
    commit(x_hours)
    assert st.tail_trim_dt == pytest.approx(float(res.resampled_full.dt[mid]))


def test_overview_trim_commit_release_before_shutin_clamps_to_index_2():
    td, st, res = _seeded_with_crash()
    stub, commit = _trim_commit(td, st, res)
    x_hours = (res.t_shutin_s - 3600.0) / 3600.0  # 1h before shut-in -> negative dt_target
    commit(x_hours)
    assert st.tail_trim_dt == pytest.approx(float(res.resampled_full.dt[2]))


def test_overview_trim_commit_recovery_two_point_resample_never_indexerrors():
    """Regression: with an exactly-2-point resampled_full, clamping to index 2 BEFORE the clear
    check must never index past the end of dt_full. A mid-plot drag on a record this short can
    only ever clear the trim -- pinned here by pre-setting a real trim and asserting it's
    actually cleared, not just that nothing raised."""
    td = make_testdata()
    dt_full = np.array([0.0, 10.0])
    res = DerivedResults(diagnostics=None, resampled=None,
                         resampled_full=resample.Resampled(dt=dt_full, p=np.array([500.0, 300.0]),
                                                            n_raw=2),
                         t_shutin_s=1000.0)
    st = PickState(tail_trim_dt=5.0)
    stub, commit = _trim_commit(td, st, res)

    commit((res.t_shutin_s + 5.0) / 3600.0)  # a mid-plot x -- must not raise, and must clear

    assert st.tail_trim_dt is None


# --------------------------------------------------------------------------------------------------
# toggle / panel visibility (duck-typed stand-ins, no real tk.Tk())
# --------------------------------------------------------------------------------------------------
class _Var:
    def __init__(self, value=None):
        self.value = value

    def set(self, v):
        self.value = v

    def get(self):
        return self.value


def test_on_show_trim_flips_show_trim_and_refreshes():
    stub = types.SimpleNamespace()
    stub.var_show_trim = _Var(True)
    stub.show_trim = False
    calls = []
    stub.refresh = lambda: calls.append(True)
    stub._on_show_trim = types.MethodType(DfitApp._on_show_trim, stub)

    stub._on_show_trim()

    assert stub.show_trim is True
    assert calls == [True]

    stub.var_show_trim.set(False)
    stub._on_show_trim()
    assert stub.show_trim is False
    assert calls == [True, True]


class _FakeFrame:
    def __init__(self):
        self.packed = False

    def pack(self, **kw):
        self.packed = True

    def pack_forget(self):
        self.packed = False


def _panel_stub(step):
    stub = types.SimpleNamespace()
    stub.frm_overview = _FakeFrame()
    stub.frm_cscen = _FakeFrame()
    stub.frm_pcscen = _FakeFrame()
    stub.sep_before_notes = object()
    stub.step = step
    stub.state = PickState()
    stub.btn_gfunction_reset = types.SimpleNamespace(configure=lambda **kw: None)
    stub.rb_ppaxis = []
    stub._update_ppaxis_enabled = lambda: None
    stub._update_panel_visibility = types.MethodType(DfitApp._update_panel_visibility, stub)
    return stub


def test_update_panel_visibility_packs_frm_overview_only_on_overview_step():
    stub = _panel_stub("overview")
    stub._update_panel_visibility()
    assert stub.frm_overview.packed is True
    assert stub.frm_cscen.packed is False
    assert stub.frm_pcscen.packed is False


def test_update_panel_visibility_does_not_pack_frm_overview_on_other_steps():
    for step in ("injection", "isip", "gfunction", "tangent", "loglog", "porepressure"):
        stub = _panel_stub(step)
        stub._update_panel_visibility()
        assert stub.frm_overview.packed is False, step
