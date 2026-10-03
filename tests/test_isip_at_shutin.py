"""Apparent ISIP taken directly at the shut-in sample (PickState.isip_at_shutin), plus the
low-riding rate axis (plots.RATE_VIEW_FACTOR) on overview/injection/ISIP.
"""

from __future__ import annotations

import types

import numpy as np
import pytest
from matplotlib.figure import Figure

from dfit_tool import model, picks, plots, store
from dfit_tool import colors as C
from dfit_tool.model import PickState, compute_all, infer_step_status
from dfit_tool.ui import DfitApp
from tests.helpers import SHUTIN_IDX, injection_state, make_testdata


def _seeded(at_shutin=False):
    td = make_testdata()
    st = injection_state(td)
    picks.seed_injection(st, td)
    res = compute_all(st, td)
    picks.seed_isip(st, td, res)
    st.isip_at_shutin = at_shutin
    res = compute_all(st, td)
    assert st.isip_tangent is not None
    return td, st, res


def _no_rate(td, st):
    st.rate_col = None
    st.volume_col = None
    return compute_all(st, td)


# ---- model ----------------------------------------------------------------------------------------
def test_default_is_tangent_method():
    td, st, res = _seeded()
    assert st.isip_at_shutin is False
    assert res.apparent_isip_method == "tangent"
    assert res.apparent_isip is not None


def test_at_shutin_uses_bhp_at_shutin_sample_and_keeps_tangent():
    td, st, res_t = _seeded()
    tangent_pick = st.isip_tangent
    st.isip_at_shutin = True
    res = compute_all(st, td)
    assert res.apparent_isip == res.bhp_all[st.shutin_idx]
    assert res.apparent_isip_method == "shutin"
    assert st.isip_tangent is tangent_pick
    assert res.apparent_isip != res_t.apparent_isip


def test_at_shutin_needs_no_tangent_pick():
    td, st, _ = _seeded()
    st.isip_tangent = None
    st.isip_at_shutin = True
    res = compute_all(st, td)
    assert res.apparent_isip == res.bhp_all[st.shutin_idx]


def test_at_shutin_follows_shutin_line():
    td, st, _ = _seeded(at_shutin=True)
    st.shutin_idx = SHUTIN_IDX + 5
    res = compute_all(st, td)
    assert res.apparent_isip == res.bhp_all[SHUTIN_IDX + 5]


def test_at_shutin_nan_sample_is_blank_with_warning():
    td, st, _ = _seeded(at_shutin=True)
    td.df.loc[st.shutin_idx, "PRESSURE"] = np.nan
    res = compute_all(st, td)
    assert res.apparent_isip is None
    assert any("Apparent ISIP at shut-in" in w for w in res.warnings)


def test_at_shutin_dropout_masked_sample_is_blank_with_warning(monkeypatch):
    # detect_dropouts never masks the very first post-shut-in sample (no lead-in reference), so
    # force the mask to cover it: the branch must still treat a masked sample as missing.
    from dfit_tool import resample
    real = resample.detect_dropouts

    def fake(dt, p, *a, **kw):
        mask, events = real(dt, p, *a, **kw)
        mask = mask.copy()
        mask[0] = True
        return mask, events

    monkeypatch.setattr(resample, "detect_dropouts", fake)
    td, st, _ = _seeded(at_shutin=True)
    res = compute_all(st, td)
    assert res.dropout_mask[st.shutin_idx]
    assert res.apparent_isip is None
    assert any("Apparent ISIP at shut-in" in w for w in res.warnings)


def test_downstream_sees_shutin_value():
    td, st, _ = _seeded(at_shutin=True)
    st.tvd_ft = 10000.0
    res = compute_all(st, td)
    assert res.apparent_isip_gradient == pytest.approx(res.apparent_isip / 10000.0)


def test_method_blank_without_shutin():
    res = compute_all(PickState(), make_testdata())
    assert res.apparent_isip_method == ""


def test_method_blank_whenever_value_blank():
    # The log must never pair a method with a blank ISIP: no tangent placed, or an unusable
    # shut-in sample, both leave the method blank.
    td, st, _ = _seeded()
    st.isip_tangent = None
    res = compute_all(st, td)
    assert res.apparent_isip is None and res.apparent_isip_method == ""
    td, st, _ = _seeded(at_shutin=True)
    td.df.loc[st.shutin_idx, "PRESSURE"] = np.nan
    res = compute_all(st, td)
    assert res.apparent_isip is None and res.apparent_isip_method == ""


def test_json_round_trip_and_old_save_default(tmp_path):
    path = str(tmp_path / "p.json")
    PickState(isip_at_shutin=True).to_json(path)
    assert PickState.from_json(path).isip_at_shutin is True
    assert model._decode({"well_name": "w1"}).isip_at_shutin is False


def test_infer_step_status_isip_done_for_shutin_flag():
    assert infer_step_status(PickState(isip_at_shutin=True))["isip"] == "done"
    assert "isip" not in infer_step_status(PickState())


# ---- plots ----------------------------------------------------------------------------------------
def _render(fn, td, st, res):
    fig = Figure()
    ax = fig.add_subplot(111)
    return fig, ax, fn(ax, td, st, res)


def test_overview_y2lim_is_3x_max_rate():
    td, st, res = _seeded()
    _, _, d = _render(plots.render_overview, td, st, res)
    assert d.y2lim == plots.nice_limits(0.0, 3.0 * float(np.nanmax(res.rate_all)))


def test_injection_y2lim_is_3x_max_plotted_rate():
    td, st, res = _seeded()
    _, _, d = _render(plots.render_injection, td, st, res)
    assert d.y2lim == plots.nice_limits(0.0, 3.0 * float(np.nanmax(res.rate_all)))


def test_isip_y2lim_uses_rate_in_window_only():
    td, st, res = _seeded()
    _, _, d = _render(plots.render_isip, td, st, res)
    t_min = (td.t_s - res.t_shutin_s) / 60.0
    m = (t_min >= -5.0) & (t_min <= 15.0)
    assert d.y2lim == plots.nice_limits(0.0, 3.0 * float(np.nanmax(res.rate_all[m])))
    assert d.y2_color == C.RATE


@pytest.mark.parametrize("fn", [plots.render_overview, plots.render_injection, plots.render_isip])
def test_y2lim_none_without_rate(fn):
    td, st, _ = _seeded()
    res = _no_rate(td, st)
    fig, ax, d = _render(fn, td, st, res)
    assert d.y2lim is None
    assert len(fig.axes) == 1


def test_isip_has_rate_twin():
    td, st, res = _seeded()
    fig, ax, _ = _render(plots.render_isip, td, st, res)
    assert len(fig.axes) == 2


def _gids(ax):
    return {l.get_gid() for l in ax.get_lines()}


def test_isip_tangent_mode_draws_tangent():
    td, st, res = _seeded()
    _, ax, _ = _render(plots.render_isip, td, st, res)
    assert "isip_tangent_segment" in _gids(ax)
    assert "isip_shutin_dot" not in _gids(ax)


def test_isip_at_shutin_draws_dot_not_tangent():
    td, st, res = _seeded(at_shutin=True)
    _, ax, _ = _render(plots.render_isip, td, st, res)
    gids = _gids(ax)
    assert "isip_shutin_dot" in gids
    assert not any(g and g.startswith("isip_tangent_") for g in gids)
    assert "(at shut-in)" in ax.get_title()
    assert f"{res.apparent_isip:.0f}" in ax.get_title()


def test_isip_at_shutin_none_value_has_placeholder_title():
    td, st, _ = _seeded(at_shutin=True)
    td.df.loc[st.shutin_idx, "PRESSURE"] = np.nan
    res = compute_all(st, td)
    _, ax, _ = _render(plots.render_isip, td, st, res)
    assert "isip_shutin_dot" not in _gids(ax)
    assert "Apparent ISIP" in ax.get_title()


@pytest.mark.parametrize("step", ["overview", "injection", "isip"])
def test_render_step_figure_twin_ylim_equals_default(step):
    td, st, res = _seeded()
    fig = plots.render_step_figure(step, td, st, res, None)
    twin = fig.axes[1]
    _, _, d = _render(plots.RENDERERS[step], td, st, res)
    assert twin.get_ylim() == pytest.approx(d.y2lim)


# ---- store ----------------------------------------------------------------------------------------
def test_log_column_is_last_and_filled(tmp_path):
    assert store.LOG_COLUMNS[-6] == "apparent_isip_method"
    for at_shutin, want in ((False, "tangent"), (True, "shutin")):
        td, st, res = _seeded(at_shutin=at_shutin)
        entry = store.TestEntry(test_id="w", folder=str(tmp_path))
        row = store.build_log_row(entry, str(tmp_path / "w.csv"), str(tmp_path), st, td, res)
        assert list(row.keys()) == store.LOG_COLUMNS
        assert row["apparent_isip_method"] == want


# ---- ui -------------------------------------------------------------------------------------------
def test_on_isip_at_shutin_sets_state_and_refreshes():
    td, st, res = _seeded()
    calls = []
    fake = types.SimpleNamespace(
        state=st, td=td, res=res,
        var_isip_at_shutin=types.SimpleNamespace(get=lambda: True),
        refresh=lambda: calls.append("refresh"))
    DfitApp._on_isip_at_shutin(fake)
    assert st.isip_at_shutin is True
    assert calls == ["refresh"]


def test_update_panel_visibility_frm_isip_only_on_isip_step_and_resyncs_var():
    from dfit_tool import ui
    from tests.test_stiffness import _panel_visibility_stub

    stub = _panel_visibility_stub()
    stub.state.isip_at_shutin = True
    for key, _ in ui.STEPS:
        stub.step = key
        stub._update_panel_visibility()
        assert stub.frm_isip.packed == (key == "isip")
    stub.step = "isip"
    stub._update_panel_visibility()
    assert stub.var_isip_at_shutin.get() is True
