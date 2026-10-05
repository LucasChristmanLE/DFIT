"""C-A inflection fallback: when dP/dG never rises 10% above a genuine interior min, the contact
goes to the inflection on the rising limb (the effective ISIP stays anchored at the min)."""

import types

import numpy as np

from dfit_tool import interpret, picks, store
from dfit_tool.model import DerivedResults, PickState, compute_all
from dfit_tool.resample import Diagnostics
from tests.helpers import injection_state, make_testdata


def _small_rise_curve():
    """Interior min at G=5 (value 50), then an S-shaped rise of ~2.4 (< 10% = 5) whose slope
    peaks (the inflection) at G=7, then a hump at G=9."""
    G = np.linspace(0.0, 12.0, 241)
    sig = lambda g: 1.0 / (1.0 + np.exp(-(g - 7.0) * 1.5))
    dPdG = np.where(G <= 5.0, 50.0 + (5.0 - G) ** 2, 50.0 + 2.5 * (sig(G) - sig(5.0)))
    dPdG = dPdG - 0.4 * np.clip(G - 9.0, 0.0, None)  # hump at G=9, gentle decline after
    return G, dPdG


def _monotonic():
    G = np.linspace(0.0, 12.0, 241)
    return G, 100.0 * np.exp(-G / 3.0)


def _res_with(G, dPdG):
    z = np.zeros_like(G)
    dg = Diagnostics(G=G, dPdG=dPdG, GdPdG=G * dPdG, d2PdG2=np.gradient(dPdG, G),
                     t=G, p=z, dp=z, tdpdt=z)
    return DerivedResults(diagnostics=dg)


# interpret ---------------------------------------------------------------------------------
def test_fallback_returns_rising_limb_inflection():
    G, dPdG = _small_rise_curve()
    min_idx = int(np.argmin(dPdG))
    assert interpret.suggest_contact_clear_index(dPdG, min_idx) is None
    idx = interpret.suggest_contact_ca_fallback_index(G, dPdG, min_idx)
    assert idx is not None and idx > min_idx
    assert abs(G[idx] - 7.0) < 0.1


def test_fallback_none_when_min_not_interior():
    G, dPdG = _monotonic()
    assert interpret.suggest_contact_ca_fallback_index(G, dPdG, len(G) - 1) is None
    assert interpret.suggest_contact_ca_fallback_index(G, dPdG, 0) is None


def test_fallback_hi_caps_search():
    G, dPdG = _small_rise_curve()
    min_idx = int(np.argmin(dPdG))
    assert interpret.suggest_contact_ca_fallback_index(G, dPdG, min_idx, hi=6.0) is None


# picks -------------------------------------------------------------------------------------
def test_re_derive_ca_small_rise_sets_inflection_contact():
    G, dPdG = _small_rise_curve()
    state = PickState(closure_scenario="C-A clear", min_dpdg_G=5.0, contact_G=1.0)
    hint = picks.re_derive_contact_from_min(state, _res_with(G, dPdG))
    assert hint == picks._CA_INFLECTION_HINT
    assert abs(state.contact_G - 7.0) < 0.1
    assert state.min_dpdg_G == 5.0


def test_re_derive_ca_monotonic_still_clears():
    G, dPdG = _monotonic()
    state = PickState(closure_scenario="C-A clear", min_dpdg_G=5.0, contact_G=1.0)
    hint = picks.re_derive_contact_from_min(state, _res_with(G, dPdG))
    assert hint == picks._CA_NO_RISE_HINT
    assert state.contact_G is None


def test_window_ca_small_rise_sets_inflection_contact():
    G, dPdG = _small_rise_curve()
    state = PickState(closure_scenario="C-A clear", min_dpdg_G=1.0)
    hint = picks.handle_min_dpdg_window(state, _res_with(G, dPdG), 4.0, 11.0)
    assert hint == picks._CA_INFLECTION_HINT
    assert abs(state.min_dpdg_G - 5.0) < 0.1
    assert abs(state.contact_G - 7.0) < 0.1


def test_window_ca_inflection_search_capped_at_window_hi():
    G, dPdG = _small_rise_curve()
    state = PickState(closure_scenario="C-A clear", min_dpdg_G=1.0, contact_G=9.0)
    hint = picks.handle_min_dpdg_window(state, _res_with(G, dPdG), 4.0, 6.0)
    assert hint == picks._CA_NO_RISE_HINT
    assert state.contact_G is None


def test_window_ca_monotonic_still_clears():
    G, dPdG = _monotonic()
    state = PickState(closure_scenario="C-A clear", min_dpdg_G=1.0, contact_G=99.0)
    assert picks.handle_min_dpdg_window(state, _res_with(G, dPdG), 4.0, 6.0) == picks._CA_NO_RISE_HINT
    assert state.contact_G is None


def test_gfunction_alert_text_only_for_ca_inflection_on_gfunction():
    res = DerivedResults(contact_method="inflection")
    ca = PickState(closure_scenario="C-A clear")
    text = picks.gfunction_alert_text(ca, res, "gfunction")
    assert text and "inflection" in text
    assert picks.gfunction_alert_text(ca, res, "isip") == ""
    assert picks.gfunction_alert_text(ca, DerivedResults(contact_method="rise10"), "gfunction") == ""
    assert picks.gfunction_alert_text(ca, DerivedResults(), "gfunction") == ""
    cb = PickState(closure_scenario="C-B adequate")
    assert picks.gfunction_alert_text(cb, res, "gfunction") == ""


# model -------------------------------------------------------------------------------------
def _ca_state(td):
    state = injection_state(td)
    res = compute_all(state, td)
    picks.seed_gfunction(state, res)
    dg = res.diagnostics
    state.closure_scenario = "C-A clear"
    state.min_dpdg_G = float(dg.G[len(dg.G) // 3])
    state.contact_G = float(dg.G[2 * len(dg.G) // 3])
    return state


def test_compute_all_inflection_method_warns_and_anchors_at_min():
    td = make_testdata()
    state = _ca_state(td)
    state.contact_rule = "inflection"
    res = compute_all(state, td)
    assert res.contact_method == "inflection"
    assert any("inflection" in w and len(w) <= 90 for w in res.warnings)
    anchor_x = res.eff_isip_line_compliance.anchor_x
    dg = res.diagnostics
    assert abs(anchor_x - dg.G[np.argmin(np.abs(dg.G - state.min_dpdg_G))]) < 1e-9
    assert abs(anchor_x - state.contact_G) > 1e-6


def test_compute_all_rise10_method_no_warning():
    td = make_testdata()
    state = _ca_state(td)
    state.contact_rule = "rise10"
    res = compute_all(state, td)
    assert res.contact_method == "rise10"
    assert not any("inflection" in w for w in res.warnings)


def test_compute_all_old_save_rule_blank_reads_by_scenario():
    td = make_testdata()
    state = _ca_state(td)
    assert state.contact_rule == ""
    assert compute_all(state, td).contact_method == "rise10"
    state.closure_scenario = "C-B adequate"
    assert compute_all(state, td).contact_method == "inflection"
    state.contact_G = None
    assert compute_all(state, td).contact_method is None


def test_contact_rule_roundtrip_and_old_json_loads():
    from dfit_tool.model import _decode
    import dataclasses
    d = dataclasses.asdict(PickState(closure_scenario="C-A clear", contact_rule="inflection"))
    assert _decode(dict(d)).contact_rule == "inflection"
    d.pop("contact_rule")
    assert _decode(dict(d)).contact_rule == ""
    d["contact_rule"] = None
    assert _decode(dict(d)).contact_rule == ""


def test_rule_set_by_commit_paths():
    G, dPdG = _small_rise_curve()
    res = _res_with(G, dPdG)
    state = PickState(closure_scenario="C-A clear", min_dpdg_G=5.0)
    picks.re_derive_contact_from_min(state, res)
    assert state.contact_rule == "inflection"
    state = PickState(closure_scenario="C-A clear", min_dpdg_G=1.0)
    picks.handle_min_dpdg_window(state, res, 4.0, 11.0)
    assert state.contact_rule == "inflection"
    sG = np.linspace(0.0, 12.0, 241)
    sy = 50.0 + (sG - 5.0) ** 2
    state = PickState(closure_scenario="C-A clear", min_dpdg_G=5.0)
    picks.re_derive_contact_from_min(state, _res_with(sG, sy))
    assert state.contact_rule == "rise10"
    mG, my = _monotonic()
    picks.re_derive_contact_from_min(state, _res_with(mG, my))
    assert state.contact_rule == ""
    state = PickState(closure_scenario="C-B adequate", min_dpdg_G=5.0, contact_rule="rise10")
    cG = np.linspace(0.0, 12.0, 241)
    cy = 300.0 - (cG + (cG - 6.0) ** 3 / 3.0)
    picks.re_derive_contact_from_min(state, _res_with(cG, cy))
    assert state.contact_rule == "inflection"
    state = PickState(closure_scenario="C-C no-contact", contact_rule="inflection", contact_G=1.0)
    picks.apply_closure_scenario(state, res)
    assert state.contact_rule == ""
    state = PickState(closure_scenario="C-A clear", contact_rule="inflection")
    picks.commit_contact_point(state, 3.0)
    assert state.contact_rule == "inflection"


def test_hint_text_inflection_variant():
    t = picks.gfunction_hint_text("C-A clear", "inflection")
    assert "inflection" in t and "rel-min +10%" not in t
    assert "rel-min +10%" in picks.gfunction_hint_text("C-A clear", "rise10")
    assert "rel-min +10%" in picks.gfunction_hint_text("C-A clear")


def test_threshold_line_drawn_without_text_label():
    import matplotlib.pyplot as plt
    from dfit_tool import plots
    td = make_testdata()
    state = _ca_state(td)
    def texts(rule):
        state.contact_rule = rule
        res = compute_all(state, td)
        fig, ax = plt.subplots()
        plots.render_gfunction(ax, td, state, res)
        out = [t.get_text() for a in fig.axes for t in a.texts]
        gids = {ln.get_gid() for a in fig.axes for ln in a.get_lines()}
        plt.close(fig)
        return out, gids
    # The label sat behind the legend; the dashed line alone marks the threshold.
    for rule in ("inflection", ""):
        out, gids = texts(rule)
        assert "clear_threshold_line" in gids
        assert not any("never reached" in t for t in out)


def test_fallback_caps_search_at_hump_sparse_curve():
    G = np.array([1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7, 8, 9, 10], dtype=float)
    y = np.array([130, 115, 105, 100, 102, 95, 88, 80, 76, 74.5, 74, 72, 69], dtype=float)
    assert interpret.suggest_contact_ca_fallback_index(G, y, 3) == 4


def test_fallback_none_without_hump_right_of_min():
    G = np.arange(1.0, 8.0)
    y = np.array([9.0, 7.0, 5.0, 4.0, 4.5, 4.4, 4.3])
    # min at idx 3, hump at idx 4 exists -> returns hump; make rising limb absent instead
    y2 = np.array([9.0, 7.0, 5.0, 4.0, 3.9, 3.8, 3.7])
    assert interpret.suggest_contact_ca_fallback_index(G, y2, 3) is None
    assert interpret.suggest_contact_ca_fallback_index(G, y, 3) == 4


def test_fallback_uses_hump_right_of_min_not_global_hump():
    # AEF Fed 05-61-34-5649B shape: a tall early hump (largest G*dP/dG, so suggest_hump_index
    # picks it) left of a late min that rises ~6% to a small local max.
    G = np.array([1, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], dtype=float)
    y = np.array([80, 160, 300, 200, 120, 60, 48, 47, 47.1, 47.6, 48.8, 49.5, 49.0, 48.5])
    assert interpret.suggest_hump_index(G, y) == 2
    idx = interpret.suggest_contact_ca_fallback_index(G, y, 7)
    assert idx is not None and 7 < idx <= 11


# store -------------------------------------------------------------------------------------
def test_contact_method_is_last_log_column_and_written(tmp_path):
    assert store.LOG_COLUMNS[-4] == "contact_method"
    td = make_testdata()
    state = _ca_state(td)
    state.contact_rule = "inflection"
    res = compute_all(state, td)
    entry = store.TestEntry(test_id="well1", folder=str(tmp_path))
    row = store.build_log_row(entry, str(tmp_path / "well1.csv"), str(tmp_path), state, td, res)
    assert row["contact_method"] == "inflection"


# ui wiring ---------------------------------------------------------------------------------
def test_update_alert_shows_and_hides_label():
    from dfit_tool.ui import DfitApp
    calls = []
    lbl = types.SimpleNamespace(
        config=lambda **kw: calls.append(("config", kw["text"])),
        pack=lambda **kw: calls.append(("pack", kw)),
        pack_forget=lambda: calls.append(("forget",)))
    stub = types.SimpleNamespace(
        alert_lbl=lbl, hint_lbl=object(), step="gfunction",
        state=PickState(closure_scenario="C-A clear"),
        res=DerivedResults(contact_method="inflection"))
    DfitApp._update_alert(stub)
    assert calls[0][0] == "config" and calls[0][1]
    assert calls[1][0] == "pack" and calls[1][1].get("after") is stub.hint_lbl
    calls.clear()
    stub.res = DerivedResults()
    DfitApp._update_alert(stub)
    assert calls == [("config", ""), ("forget",)]
