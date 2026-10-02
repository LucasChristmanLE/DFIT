"""Expanded-results support: new DerivedResults fields, the headless summary tables
(dfit_tool/summary.py), the summary chart (plots.render_summary), the 9_summary.png export, and
the sidebar changes that go with them (gradient rows removed, step frames packed after the
results separator, refresh hook for the results window)."""

from __future__ import annotations

import io
import types

import numpy as np
import pytest
from matplotlib.figure import Figure

from dfit_tool import interpret, plots, summary, ui
from dfit_tool.model import DerivedResults, PickState, STEPS, compute_all
from tests.helpers import make_testdata, injection_state
from tests.test_gradients import _full_state
from tests.test_stiffness import _panel_visibility_stub

ALL_DONE = {k: "done" for k, _ in STEPS}


def _visited(st):
    st.step_status = dict(ALL_DONE)
    return st


def _section(secs, title):
    return next(s for s in secs if s.title == title)


def _row(sec, label):
    return next(r for r in sec.rows if r[0] == label)


# ---- new DerivedResults fields -------------------------------------------------------------------
def test_closure_g_fields_equal_picks_and_variable_is_midpoint():
    td, st, res = _full_state("C-A clear")
    assert res.closure_G_compliance == pytest.approx(st.contact_G)
    assert res.closure_G_tangent == pytest.approx(st.closure_G)
    assert res.closure_G_variable == pytest.approx((st.contact_G + st.closure_G) / 2)


def test_liberty_anchor_fields_match_pick_and_pressure():
    td, st, res = _full_state("C-A clear")
    assert res.liberty_anchor_G == pytest.approx(st.min_dpdg_G)
    assert res.liberty_anchor_pressure == pytest.approx(
        res.shmin_liberty + interpret.LIBERTY_OFFSET_PSI)


def test_liberty_anchor_time_matches_its_anchor_pick():
    # C-A anchors on the min-dP/dG pick, C-B on the contact pick; the time is the shut-in time
    # at that G, the same interp the other closure times use.
    td, st, res = _full_state("C-A clear")
    assert res.liberty_anchor_time_s == pytest.approx(res.min_dpdg_time_s)
    st.closure_scenario = "C-B adequate"
    res_b = compute_all(st, td)
    if res_b.shmin_liberty is not None:
        assert res_b.liberty_anchor_time_s == pytest.approx(res_b.closure_time_compliance_s)
    for scen in ("C-C no-contact", "C-D rapid", "C-X uninterpretable"):
        st.closure_scenario = scen
        assert compute_all(st, td).liberty_anchor_time_s is None
    sec = _section(summary.summary_sections(_visited(_full_state("C-A clear")[1]), res),
                   "Closure comparison")
    assert _row(sec, "Liberty")[sec.columns.index("tc (min)")] == \
        f"{res.liberty_anchor_time_s / 60:.2f}"


def test_pore_pressure_fit_fields_set_and_consistent():
    td, st, res = _full_state("C-A clear")
    dg = res.diagnostics
    lo, hi = st.pp_window
    m = (dg.t >= lo) & (dg.t <= hi) & (dg.t > 0)
    assert res.pore_pressure_n_points == int(m.sum())
    assert res.pore_pressure_n_points >= 2
    expo = -0.5 if st.pp_axis == "tm12" else -1.0
    slope, icpt = interpret.pore_pressure_fit(dg.t[m] ** expo, dg.p[m])
    assert res.pore_pressure_slope == pytest.approx(slope)
    assert res.pore_pressure == pytest.approx(icpt)
    assert interpret.pore_pressure(dg.t[m] ** expo, dg.p[m]) == pytest.approx(icpt)


def test_new_fields_default_none_on_empty_state():
    td = make_testdata()
    res = compute_all(injection_state(td), td)
    for f in ("closure_G_compliance", "closure_G_tangent", "closure_G_variable",
              "liberty_anchor_G", "liberty_anchor_pressure", "pore_pressure_slope",
              "pore_pressure_n_points", "effective_isip_compliance_gradient"):
        assert getattr(res, f) is None


def test_effective_isip_gradients_are_value_over_tvd():
    td, st, res = _full_state("C-A clear")
    for kind in ("compliance", "tangent", "variable"):
        v = getattr(res, f"effective_isip_{kind}")
        assert v is not None
        assert getattr(res, f"effective_isip_{kind}_gradient") == pytest.approx(v / st.tvd_ft)


def test_effective_isip_gradients_blank_without_tvd():
    td, st, res = _full_state("C-A clear", tvd_ft=None)
    assert res.effective_isip_compliance_gradient is None
    assert res.effective_isip_tangent_gradient is None


# ---- summary tables ------------------------------------------------------------------------------
def test_sections_titles_in_order():
    td, st, res = _full_state("C-A clear")
    secs = summary.summary_sections(_visited(st), res)
    assert [s.title for s in secs] == ["Injection", "ISIP", "Closure comparison",
                                       "Postclosure", "Inputs and data"]
    for s in secs:
        for r in s.rows:
            assert len(r) == len(s.columns)
            assert all(isinstance(c, str) for c in r)


def test_closure_comparison_rows_and_values():
    td, st, res = _full_state("C-A clear")
    sec = _section(summary.summary_sections(_visited(st), res), "Closure comparison")
    comp = _row(sec, "compliance (contact)")
    assert comp[sec.columns.index("Shmin")] == f"{res.shmin_compliance:.0f}"
    assert comp[sec.columns.index("G")] == f"{st.contact_G:.3f}"
    var = _row(sec, "variable")
    assert var[sec.columns.index("G")] == f"{res.closure_G_variable:.3f}"
    lib = _row(sec, "Liberty")
    assert lib[sec.columns.index("Shmin")] == f"{res.shmin_liberty:.0f}"
    assert not any(r[0].startswith("rapid") for r in sec.rows)


def test_cd_state_shows_rapid_row():
    td, st, res = _full_state("C-D rapid")
    assert res.shmin_rapid is not None
    sec = _section(summary.summary_sections(_visited(st), res), "Closure comparison")
    rapid = next(r for r in sec.rows if r[0].startswith("rapid"))
    assert rapid[sec.columns.index("Shmin")] == f"{res.shmin_rapid:.0f}"


def test_pcf_postclosure_reads_skipped():
    td, st, res = _full_state("C-A clear")
    st.postclosure_scenario = "PC-F no peak"
    res = compute_all(st, td)
    sec = _section(summary.summary_sections(_visited(st), res), "Postclosure")
    assert any("skipped (PC-F)" in c for r in sec.rows for c in r)
    assert not any(r[0] == "pore pressure (psi)" for r in sec.rows)


def test_postclosure_rows_show_fit_details():
    td, st, res = _full_state("C-A clear")
    sec = _section(summary.summary_sections(_visited(st), res), "Postclosure")
    assert _row(sec, "points in window")[1] == str(res.pore_pressure_n_points)
    assert _row(sec, "pore pressure (psi)")[1] == f"{res.pore_pressure:.0f}"
    assert _row(sec, "gradient (psi/ft)")[1] == f"{res.pore_pressure_gradient:.3f}"


def test_not_visited_steps_show_dashes():
    td, st, res = _full_state("C-A clear")
    st.step_status = {}
    secs = summary.summary_sections(st, res)
    comp = _section(secs, "Closure comparison")
    assert all(c == "-" for c in _row(comp, "compliance (contact)")[1:])
    assert all(c == "-" for c in _row(comp, "tangent (closure)")[1:])
    assert _row(_section(secs, "ISIP"), "apparent ISIP")[1] == "-"
    assert _row(_section(secs, "Injection"), "te (min)")[1] == "-"


def test_visited_helper():
    assert summary.visited(PickState(step_status={"isip": "done"}), "isip")
    assert summary.visited(PickState(step_status={"isip": "in_progress"}), "isip")
    assert not summary.visited(PickState(step_status={}), "isip")
    assert not summary.visited(PickState(step_status={"isip": "not_visited"}), "isip")


def test_asterisk_never_next_to_dash():
    # No compliance contact: the reference ISIP falls back to the tangent one.
    td, st, res = _full_state("C-C no-contact")
    assert res.net_pressure_isip_source == "tangent"
    secs = summary.summary_sections(_visited(st), res)
    row = next(r for r in _section(secs, "ISIP").rows if r[0].startswith("NWB complexity"))
    assert row[0] == "NWB complexity*" and row[1] != "-"
    for s in secs:
        for r in s.rows:
            if "*" in r[0]:
                assert r[1] != "-"
    st.step_status = {}
    row = next(r for r in _section(summary.summary_sections(st, res), "ISIP").rows
               if r[0].startswith("NWB complexity"))
    assert row[:2] == ["NWB complexity", "-"]


def test_gradients_in_tables_come_from_res():
    td, st, res = _full_state("C-A clear")
    isip = _section(summary.summary_sections(_visited(st), res), "ISIP")
    assert _row(isip, "apparent ISIP")[2] == f"{res.apparent_isip_gradient:.3f}"
    assert _row(isip, "eff ISIP (tangent)")[2] == f"{res.effective_isip_tangent_gradient:.3f}"


def test_inputs_section():
    td, st, res = _full_state("C-A clear")
    sec = _section(summary.summary_sections(_visited(st), res), "Inputs and data")
    assert _row(sec, "TVD (ft)")[1] == f"{st.tvd_ft:.0f}"
    assert _row(sec, "resampled points")[1] == str(len(res.resampled.p))
    assert _row(sec, "dropouts masked")[1] == str(len(res.dropouts))


# ---- render_summary ------------------------------------------------------------------------------
def _ytick_labels(ax):
    return [t.get_text() for t in ax.get_yticklabels()]


def _by_gid(fig, gid):
    return next((a for a in fig.axes if a.get_gid() == gid), None)


def test_render_summary_full_state():
    td, st, res = _full_state("C-A clear")
    _visited(st)
    fig = Figure()
    plots.render_summary(fig, st, res)
    ladder = _by_gid(fig, "summary_ladder")
    assert ladder is not None
    labels = _ytick_labels(ladder)
    assert "apparent ISIP" in labels and "pore pressure" in labels
    assert "Shmin variable" in labels
    assert _by_gid(fig, "summary_breakdown") is not None
    assert _by_gid(fig, "summary_closure") is not None
    assert any(a.get_xlabel() == "psi/ft" for a in ladder.child_axes)  # TVD > 0
    fig.savefig(io.BytesIO(), format="png")


def test_render_summary_omits_none_values():
    td, st, res = _full_state("C-A clear")
    res.shmin_tangent = None
    res.pore_pressure = None
    _visited(st)
    fig = Figure()
    plots.render_summary(fig, st, res)
    ladder = _by_gid(fig, "summary_ladder")
    labels = _ytick_labels(ladder)
    assert "Shmin tangent" not in labels
    assert "pore pressure" not in labels
    for ln in ladder.lines:
        assert 0.0 not in list(np.asarray(ln.get_xdata(), dtype=float))


def test_render_summary_no_tvd_has_no_secondary_axis():
    td, st, res = _full_state("C-A clear", tvd_ft=None)
    _visited(st)
    fig = Figure()
    plots.render_summary(fig, st, res)
    assert not _by_gid(fig, "summary_ladder").child_axes


def test_render_summary_empty_draws_message():
    fig = Figure()
    plots.render_summary(fig, PickState(), DerivedResults())
    assert any("No results yet" in t.get_text() for t in fig.texts)
    fig.savefig(io.BytesIO(), format="png")


def test_render_summary_cd_and_pcf_states_run():
    td, st, res = _full_state("C-D rapid")
    _visited(st)
    fig = Figure()
    plots.render_summary(fig, st, res)
    assert "Shmin rapid" in _ytick_labels(_by_gid(fig, "summary_ladder"))
    st.postclosure_scenario = "PC-F no peak"
    res = compute_all(st, td)
    plots.render_summary(Figure(), st, res)


# ---- export --------------------------------------------------------------------------------------
def test_save_all_step_pngs_writes_summary(tmp_path):
    td, st, res = _full_state("C-A clear")
    paths = plots.save_all_step_pngs(str(tmp_path), td, st, res, views={})
    assert paths[-1].endswith("9_summary.png")
    assert (tmp_path / "9_summary.png").stat().st_size > 0


# ---- ui: sidebar ---------------------------------------------------------------------------------
def test_gradient_rows_gone_from_sidebar():
    assert len(ui.PANEL_FIELDS) == 20
    assert not any("grad" in f for f in ui.PANEL_FIELDS)
    assert set(ui.FIELD_STEP) == set(ui.PANEL_FIELDS)


def test_step_frames_pack_bottom_above_notes():
    stub = _panel_visibility_stub()
    for key, _ in ui.STEPS:
        stub.step = key
        stub._update_panel_visibility()
    packed = [kw for fr in (stub.frm_cscen, stub.frm_isip, stub.frm_tangent, stub.frm_pcscen,
                            stub.frm_stiffness) for kw in fr.pack_calls]
    assert packed
    for kw in packed:
        # Slotted right after frm_notes in the pack order. A bare pack() appends after the
        # Results rows, so the frame would get space last and be the first thing clipped.
        assert kw.get("side") == "bottom"
        assert kw.get("after") is stub.frm_notes


def test_overview_issues_list_outranks_notes_for_space():
    # On Overview the issues list is the only full list of blockers; it is slotted ahead of
    # frm_notes in the pack order so a short panel squeezes Notes, not the issues.
    stub = _panel_visibility_stub()
    stub.step = "overview"
    stub._update_panel_visibility()
    assert stub.frm_issues.pack_calls[-1].get("before") is stub.frm_notes


def test_refresh_updates_results_window_only_when_open():
    from tests.test_view_state import _refresh_stub
    td = make_testdata()
    st = injection_state(td)
    calls = []
    stub = _refresh_stub(td, st, "overview")
    stub._update_results_window = lambda: calls.append(1)
    stub._results_win = None
    stub.refresh()
    assert calls == []
    stub._results_win = object()
    stub.refresh()
    assert calls == [1]


# ---- review fixes --------------------------------------------------------------------------------
@pytest.mark.parametrize("tvd,dens", [("abc", "x"), (float("nan"), None), (float("inf"), 8.6)])
def test_bad_tvd_density_do_not_crash_summary_charts_or_export(tmp_path, tvd, dens):
    td, st, res = _full_state("C-A clear")
    _visited(st)
    st.tvd_ft, st.density_ppg = tvd, dens
    secs = summary.summary_sections(st, res)
    inputs = _section(secs, "Inputs and data")
    assert _row(inputs, "TVD (ft)")[1] == "-"
    if isinstance(dens, str):
        assert _row(inputs, "fluid density (ppg)")[1] == "-"
    for sec in secs:
        assert not any("nan" in c.lower() or "inf" in c.lower() for r in sec.rows for c in r)
    fig = Figure()
    plots.render_summary(fig, st, res)
    assert not _by_gid(fig, "summary_ladder").child_axes
    paths = plots.save_all_step_pngs(str(tmp_path), td, st, res, views={})
    assert paths[-1].endswith("9_summary.png")


def test_charts_respect_not_visited_gate():
    td, st, res = _full_state("C-A clear")
    _visited(st)
    st.step_status["tangent"] = "not_visited"
    fig = Figure()
    plots.render_summary(fig, st, res)
    labels = _ytick_labels(_by_gid(fig, "summary_ladder"))
    assert "Shmin tangent" not in labels and "Shmin variable" not in labels
    assert "eff ISIP tangent" not in labels and "eff ISIP variable" not in labels
    assert "Shmin compliance" in labels
    assert "compliance" in _ytick_labels(_by_gid(fig, "summary_breakdown"))
    assert "tangent" not in _ytick_labels(_by_gid(fig, "summary_breakdown"))
    closure = _pick_markers(_by_gid(fig, "summary_closure"))
    assert "variable" not in closure and "tangent closure" not in closure
    assert "compliance (contact - 75)" in closure


def _pick_markers(ax):
    """{label: (G, P)} for the single-point pick markers on the closure chart."""
    return {ln.get_label(): (ln.get_xdata()[0], ln.get_ydata()[0]) for ln in ax.lines
            if len(ln.get_xdata()) == 1 and not ln.get_label().startswith("_")}


def test_closure_chart_draws_pressure_curve_and_picks_at_reported_pressures():
    td, st, res = _full_state("C-A clear")
    _visited(st)
    fig = Figure()
    plots.render_summary(fig, st, res)
    ax = _by_gid(fig, "summary_closure")
    assert ax.get_xlabel() == "G-time" and ax.get_ylabel() == "BHP (psi)"
    curve = next(ln for ln in ax.lines if ln.get_label() == "BHP")
    np.testing.assert_allclose(curve.get_xdata(), res.diagnostics.G)
    np.testing.assert_allclose(curve.get_ydata(), res.resampled.p)
    marks = _pick_markers(ax)
    assert marks["min dP/dG"] == pytest.approx((st.min_dpdg_G, res.min_dpdg_pressure))
    # Compliance is plotted at Shmin (contact - 75 psi), so it sits below the curve.
    assert marks["compliance (contact - 75)"] == pytest.approx(
        (res.closure_G_compliance, res.shmin_compliance))
    assert marks["tangent closure"] == pytest.approx((res.closure_G_tangent, res.shmin_tangent))
    assert marks["variable"] == pytest.approx((res.closure_G_variable, res.shmin_variable))


def test_min_dpdg_pressure_reads_the_curve_and_blanks_for_cx():
    td, st, res = _full_state("C-A clear")
    assert res.min_dpdg_pressure == pytest.approx(
        float(np.interp(st.min_dpdg_G, res.diagnostics.G, res.resampled.p)))
    st.closure_scenario = "C-X uninterpretable"
    assert compute_all(st, td).min_dpdg_pressure is None


def test_charts_all_unvisited_show_no_results():
    td, st, res = _full_state("C-A clear")
    st.step_status = {}
    fig = Figure()
    plots.render_summary(fig, st, res)
    assert any("No results yet" in t.get_text() for t in fig.texts)


def test_min_dpdg_time_set_and_blank_for_cx():
    td, st, res = _full_state("C-A clear")
    dg = res.diagnostics
    assert res.min_dpdg_time_s == pytest.approx(
        float(np.interp(st.min_dpdg_G, dg.G, res.resampled.dt)))
    st.closure_scenario = "C-X uninterpretable"
    assert compute_all(st, td).min_dpdg_time_s is None
