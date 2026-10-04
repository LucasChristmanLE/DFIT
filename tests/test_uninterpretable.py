"""Explicit "uninterpretable" negative findings on the G-function and tangent steps.

- Closure scenario C-X ("uninterpretable"): no contact pick, so no compliance Shmin/effective
  ISIP, no Liberty, no variable -- same as C-C -- and additionally no Shmin(stiffness), whose
  curve is anchored on the min-dP/dG pick.
- PickState.tangent_uninterpretable: blanks the tangent Shmin, tangent effective ISIP, and the
  variable method (which needs closure_G). The pick stays in state, so unsetting restores it.
"""

from __future__ import annotations

import types

from matplotlib.figure import Figure

from dfit_tool import interpret, model, picks, plots, store, ui
from dfit_tool.model import PickState, compute_all, infer_step_status
from dfit_tool.ui import DfitApp
from tests.test_stiffness import _panel_visibility_stub, _state_with_pore_pressure


def _with_stiffness_pick():
    td, st, res = _state_with_pore_pressure()
    idx = interpret.suggest_stiffness_upturn_index(res.stiffness_S, res.diagnostics.G[1:])
    st.stiffness_pick_P = float(res.stiffness_p_eff[1:][idx])
    return td, st, compute_all(st, td)


# --------------------------------------------------------------------------------------------------
# C-X closure scenario
# --------------------------------------------------------------------------------------------------
def test_baseline_has_every_value_c_x_blanks():
    """Sanity: the fixture reports every value the C-X tests below expect to go blank."""
    td, st, res = _with_stiffness_pick()
    assert res.shmin_compliance is not None
    assert res.effective_isip_compliance is not None
    assert res.shmin_liberty is not None
    assert res.shmin_variable is not None
    assert res.shmin_stiffness is not None
    assert res.shmin_tangent is not None


def test_c_x_blanks_compliance_liberty_variable_and_stiffness_keeps_tangent():
    td, st, res = _with_stiffness_pick()
    st.closure_scenario = "C-X uninterpretable"
    assert picks.apply_closure_scenario(st, res) is None
    assert st.contact_G is None

    res2 = compute_all(st, td)

    assert res2.shmin_compliance is None
    assert res2.effective_isip_compliance is None
    assert res2.contact_pressure is None
    assert res2.shmin_liberty is None
    assert res2.shmin_variable is None
    assert res2.shmin_rapid is None
    assert res2.shmin_stiffness is None
    # the stiffness curve is still evidence on the plot
    assert res2.stiffness_S is not None
    # tangent values stand and become the shared reference
    assert res2.shmin_tangent == res.shmin_tangent
    assert res2.effective_isip_tangent == res.effective_isip_tangent
    assert res2.net_pressure_isip_source == "tangent"
    assert res2.net_pressure_tangent is not None


def test_c_x_contact_handlers_are_no_ops():
    td, st, res = _with_stiffness_pick()
    st.closure_scenario = "C-X uninterpretable"
    st.contact_G = None
    min_before = st.min_dpdg_G

    assert picks.re_derive_contact_from_min(st, res) is None
    dg = res.diagnostics
    assert picks.handle_min_dpdg_window(st, res, float(dg.G[0]), float(dg.G[-1])) is None

    assert st.contact_G is None
    assert st.min_dpdg_G == min_before


def test_c_x_hint_text():
    text = picks.gfunction_hint_text("C-X uninterpretable")
    assert "uninterpretable" in text


def test_c_x_is_in_the_closure_dropdown():
    assert "C-X uninterpretable" in ui.CLOSURE_SCENARIOS


def test_c_x_passes_the_gfunction_gate():
    st = PickState(closure_scenario="C-X uninterpretable")
    assert model.step_gate_error(st, "gfunction") is None


def test_render_stiffness_under_c_x_draws_no_pick_and_titles_the_finding():
    td, st, res = _with_stiffness_pick()
    st.closure_scenario = "C-X uninterpretable"
    picks.apply_closure_scenario(st, res)
    res = compute_all(st, td)
    fig = Figure()
    ax = fig.add_subplot(111)

    plots.render_stiffness(ax, td, st, res)

    assert not any(l.get_gid() == "stiffness_pick" for l in ax.get_lines())
    assert "uninterpretable" in ax.get_title().lower()
    assert len(ax.get_lines()) == 1


def test_render_gfunction_under_c_x_draws_no_contact_or_triangle():
    td, st, res = _with_stiffness_pick()
    st.closure_scenario = "C-X uninterpretable"
    picks.apply_closure_scenario(st, res)
    res = compute_all(st, td)
    fig = Figure()
    ax = fig.add_subplot(111)

    plots.render_gfunction(ax, td, st, res)

    gids = {l.get_gid() for a in fig.axes for l in a.get_lines()}
    assert not gids & {"contact_point", "contact_vline", "min_dpdg_point"}
    assert "C-X" in ax.get_title()


# --------------------------------------------------------------------------------------------------
# tangent_uninterpretable
# --------------------------------------------------------------------------------------------------
def test_tangent_flag_blanks_tangent_and_variable_keeps_compliance():
    td, st, res = _with_stiffness_pick()
    st.tangent_uninterpretable = True

    res2 = compute_all(st, td)

    assert res2.shmin_tangent is None
    assert res2.closure_pressure is None
    assert res2.closure_time_tangent_s is None
    assert res2.effective_isip_tangent is None
    assert res2.shmin_tangent_gradient is None
    assert res2.net_pressure_tangent is None
    assert res2.shmin_variable is None
    assert res2.effective_isip_variable is None
    assert res2.closure_time_variable_s is None
    assert res2.net_pressure_variable is None
    assert res2.delta_closure is None
    # compliance side untouched
    assert res2.shmin_compliance == res.shmin_compliance
    assert res2.effective_isip_compliance == res.effective_isip_compliance
    assert res2.net_pressure_compliance == res.net_pressure_compliance
    assert res2.shmin_stiffness == res.shmin_stiffness
    # the pick is preserved
    assert st.closure_G is not None


def test_tangent_flag_unset_restores_the_original_numbers():
    td, st, res = _with_stiffness_pick()
    st.tangent_uninterpretable = True
    compute_all(st, td)
    st.tangent_uninterpretable = False

    res2 = compute_all(st, td)

    assert res2.shmin_tangent == res.shmin_tangent
    assert res2.shmin_variable == res.shmin_variable
    assert res2.net_pressure_tangent == res.net_pressure_tangent


def test_both_uninterpretable_leaves_no_reference_isip():
    td, st, res = _with_stiffness_pick()
    st.closure_scenario = "C-X uninterpretable"
    picks.apply_closure_scenario(st, res)
    st.tangent_uninterpretable = True

    res2 = compute_all(st, td)

    assert res2.net_pressure_isip_source == ""
    assert res2.near_wellbore_complexity is None
    assert res2.net_pressure_compliance is None
    assert res2.net_pressure_tangent is None
    assert res2.net_pressure_variable is None
    assert res2.apparent_isip is not None


def test_render_tangent_flag_draws_no_line_or_marker_and_titles_the_finding():
    td, st, res = _with_stiffness_pick()
    st.tangent_uninterpretable = True
    res = compute_all(st, td)
    fig = Figure()
    ax = fig.add_subplot(111)

    plots.render_tangent(ax, td, st, res)

    gids = {l.get_gid() for a in fig.axes for l in a.get_lines()}
    assert not gids & {"closure_point", "closure_vline", "closure_line_segment"}
    assert "uninterpretable" in ax.get_title().lower()


def test_suppressed_picks_beyond_the_trim_do_not_warn_stale():
    td, st, res = _with_stiffness_pick()
    dg = res.diagnostics
    # trim well before the closure and min-dP/dG picks so both would normally be stale
    st.tail_trim_dt = float(res.resampled.dt[dg.G.size // 8])
    st.closure_scenario = "C-X uninterpretable"
    st.contact_G = None
    st.tangent_uninterpretable = True

    def stale_names(res):
        msgs = [w for w in res.warnings if "beyond the tail trim" in w]
        return {n for n in ("closure", "min dP/dG", "stiffness") for w in msgs if n in w}

    assert stale_names(compute_all(st, td)) == set()

    st.tangent_uninterpretable = False
    st.closure_scenario = "C-A clear"
    assert {"closure", "min dP/dG"} <= stale_names(compute_all(st, td))


def test_infer_step_status_marks_tangent_done_on_flag_alone():
    assert infer_step_status(PickState(tangent_uninterpretable=True))["tangent"] == "done"
    assert "tangent" not in infer_step_status(PickState())


def test_tangent_flag_round_trips_and_old_saves_default_false(tmp_path):
    path = str(tmp_path / "picks.json")
    PickState(tangent_uninterpretable=True).to_json(path)
    assert PickState.from_json(path).tangent_uninterpretable is True

    assert model._decode({"well_name": "w1"}).tangent_uninterpretable is False


def test_on_tangent_uninterpretable_uncheck_seeds_a_missing_pick():
    td, st, res = _with_stiffness_pick()
    st.closure_G = None
    st.closure_slope = None
    st.tangent_uninterpretable = True
    refreshed = []
    fake = types.SimpleNamespace(
        state=st, res=res,
        var_tangent_uninterpretable=types.SimpleNamespace(get=lambda: False),
        refresh=lambda: refreshed.append(True))

    DfitApp._on_tangent_uninterpretable(fake)

    assert st.tangent_uninterpretable is False
    assert st.closure_G is not None
    assert refreshed == [True]


def test_update_panel_visibility_frm_tangent_only_on_tangent_step():
    stub = _panel_visibility_stub()
    for key, _ in ui.STEPS:
        stub.step = key
        stub._update_panel_visibility()
        assert stub.frm_tangent.packed == (key == "tangent")


def test_update_panel_visibility_resyncs_tangent_var_on_tangent_entry():
    stub = _panel_visibility_stub()
    stub.state.tangent_uninterpretable = True
    stub.step = "tangent"

    stub._update_panel_visibility()

    assert stub.var_tangent_uninterpretable.get() is True


# --------------------------------------------------------------------------------------------------
# dfit_log.csv
# --------------------------------------------------------------------------------------------------
def test_log_row_records_both_findings(tmp_path):
    assert store.LOG_COLUMNS[-10] == "tangent_uninterpretable"

    td, st, res = _with_stiffness_pick()
    st.closure_scenario = "C-X uninterpretable"
    picks.apply_closure_scenario(st, res)
    st.tangent_uninterpretable = True
    res = compute_all(st, td)

    entry = store.TestEntry(test_id="well1", folder=str(tmp_path))
    row = store.build_log_row(entry, str(tmp_path / "well1.csv"), str(tmp_path), st, td, res)

    assert list(row.keys()) == store.LOG_COLUMNS
    assert row["closure_quality"] == "uninterpretable"
    assert row["tangent_uninterpretable"] is True
    assert row["Shmin_compliance"] is None
    assert row["Shmin_tangent"] is None
    assert row["tangent_Gc"] is None
    assert row["Shmin_stiffness"] is None
    assert st.closure_G is not None  # still in state, only the log blanks it
