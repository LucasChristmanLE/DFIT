"""Scenario-driven contact pick: d2P/dG2 curve, suggestion rules, scenario application,
and the G-function render additions (dotted contact vline, d2 overlay)."""

import numpy as np

from dfit_tool import resample


def test_diagnostics_has_d2pdg2():
    dt = np.linspace(0.0, 3600.0, 200)
    p = 5000.0 - 1500.0 * (1.0 - np.exp(-dt / 900.0))
    rs = resample.resample_pressure_increment(dt, p, step=5.0)
    dg = resample.diagnostics(rs, te=300.0)
    assert dg.d2PdG2.shape == dg.G.shape
    np.testing.assert_allclose(dg.d2PdG2, np.gradient(dg.dPdG, dg.G))


from dfit_tool import interpret


def _s_curve():
    """C-A shape: dP/dG dips to a min at G=5 then rises (parabola)."""
    G = np.linspace(0.0, 12.0, 241)
    dPdG = 50.0 + (G - 5.0) ** 2
    return G, dPdG


def _monotonic_decline():
    """C-C shape: dP/dG only ever falls."""
    G = np.linspace(0.0, 12.0, 241)
    return G, 100.0 * np.exp(-G / 3.0)


def _decline_with_inflection():
    """C-B shape: monotonic decline, steep -> flat (at G=6) -> steep.
    slope(G) = -(1 + (G-6)^2), so d2 = gradient(dPdG, G) has its interior max at G=6."""
    G = np.linspace(0.0, 12.0, 241)
    dPdG = 300.0 - (G + (G - 6.0) ** 3 / 3.0)
    return G, dPdG


def test_clear_rule_finds_first_10pct_rise():
    G, dPdG = _s_curve()
    min_idx = int(np.argmin(dPdG))
    idx = interpret.suggest_contact_clear_index(dPdG, min_idx)
    assert idx is not None and idx > min_idx
    # threshold: 10% above the min value; the found sample is the FIRST at/above it
    thr = dPdG[min_idx] * 1.10
    assert dPdG[idx] >= thr
    assert np.all(dPdG[min_idx + 1:idx] < thr)
    # analytic crossing: 50*(G-5)^2 rise of 5 -> G = 5 + sqrt(5)
    assert abs(G[idx] - (5.0 + np.sqrt(5.0))) < 0.1


def test_clear_rule_none_on_monotonic_decline():
    G, dPdG = _monotonic_decline()
    min_idx = int(np.argmin(dPdG))  # the last sample
    assert interpret.suggest_contact_clear_index(dPdG, min_idx) is None


def test_inflection_rule_finds_flattening():
    G, dPdG = _decline_with_inflection()
    assert np.all(np.diff(dPdG) < 0)  # sanity: monotone decline
    idx = interpret.suggest_contact_inflection_index(G, dPdG)
    assert idx is not None
    assert abs(G[idx] - 6.0) < 0.2


def test_inflection_rule_none_without_inflection():
    G, dPdG = _monotonic_decline()
    assert interpret.suggest_contact_inflection_index(G, dPdG) is None


def test_inflection_rule_respects_g_min():
    """A flattening before g_min is masked out (early water-hammer region)."""
    G = np.linspace(0.0, 12.0, 241)
    dPdG = 300.0 - (G + (G - 0.5) ** 3 / 3.0)  # flattening at G=0.5 < g_min
    assert interpret.suggest_contact_inflection_index(G, dPdG, g_min=1.0) is None


def _two_bump_decline():
    """Two interior d2 local maxima (flattenings) at roughly G=3 and G=9 -- built by integrating
    a target d2 curve with two Gaussian bumps of different heights, so the tallest-bump default
    (G=3) differs from what a seed near the shorter bump (G=9) picks."""
    G = np.linspace(0.0, 12.0, 481)
    d2_target = (-1.0 + 3.0 * np.exp(-((G - 3.0) ** 2) / (2 * 0.4 ** 2))
                 + 2.0 * np.exp(-((G - 9.0) ** 2) / (2 * 0.4 ** 2)))
    dPdG = np.cumsum(d2_target) * (G[1] - G[0])
    return G, dPdG


def test_inflection_rule_seed_picks_nearest_of_two_bumps():
    G, dPdG = _two_bump_decline()
    idx_default = interpret.suggest_contact_inflection_index(G, dPdG)
    idx_near_low = interpret.suggest_contact_inflection_index(G, dPdG, seed=3.0)
    idx_near_high = interpret.suggest_contact_inflection_index(G, dPdG, seed=9.0)
    assert idx_near_low is not None and idx_near_high is not None
    assert idx_near_low != idx_near_high
    assert abs(G[idx_near_low] - 3.0) < abs(G[idx_near_high] - 3.0)
    assert abs(G[idx_near_high] - 9.0) < abs(G[idx_near_low] - 9.0)
    assert idx_default in (idx_near_low, idx_near_high)  # the tallest bump is one of the two


def test_inflection_rule_uses_passed_in_d2():
    """A passed-in ``d2`` is what actually drives the pick -- not a fresh np.gradient of dPdG."""
    G, dPdG = _decline_with_inflection()
    real_d2 = np.gradient(dPdG, G)
    fake_d2 = np.full_like(real_d2, np.nan)
    fake_d2[100] = 5.0  # a single finite local max, far from the true inflection at G=6
    fake_d2[99] = 1.0
    fake_d2[101] = 1.0
    idx = interpret.suggest_contact_inflection_index(G, dPdG, d2=fake_d2)
    assert idx == 100
    assert abs(G[idx] - 6.0) > 0.2  # not the real-d2 inflection


from dfit_tool import picks
from dfit_tool.model import DerivedResults, PickState, compute_all
from dfit_tool.resample import Diagnostics


def _res_with(G, dPdG):
    z = np.zeros_like(G)
    dg = Diagnostics(G=G, dPdG=dPdG, GdPdG=G * dPdG, d2PdG2=np.gradient(dPdG, G),
                     t=G, p=z, dp=z, tdpdt=z)
    return DerivedResults(diagnostics=dg)


def test_scenario_ca_sets_contact_right_of_min():
    G, dPdG = _s_curve()
    state = PickState(closure_scenario="C-A clear", contact_G=99.0)
    hint = picks.apply_closure_scenario(state, _res_with(G, dPdG))
    assert hint is None
    assert state.min_dpdg_G is not None and abs(state.min_dpdg_G - 5.0) < 0.2
    assert state.contact_G is not None and state.contact_G > state.min_dpdg_G
    assert abs(state.contact_G - (5.0 + np.sqrt(5.0))) < 0.1


def test_scenario_ca_uses_existing_min_pick():
    """The rule anchors at the user's (possibly dragged) min pick, not a re-detected min."""
    G, dPdG = _s_curve()
    state = PickState(closure_scenario="C-A clear", min_dpdg_G=6.0)
    picks.apply_closure_scenario(state, _res_with(G, dPdG))
    assert state.min_dpdg_G == 6.0  # untouched
    # threshold from the value AT the pick: (50 + 1) * 1.1 = 56.1 -> (G-5)^2 >= 6.1
    assert abs(state.contact_G - (5.0 + np.sqrt(6.1))) < 0.1


def test_scenario_ca_hints_when_no_rise():
    G, dPdG = _monotonic_decline()
    state = PickState(closure_scenario="C-A clear", contact_G=3.0)
    hint = picks.apply_closure_scenario(state, _res_with(G, dPdG))
    assert hint is not None
    assert state.contact_G == 3.0  # left unchanged


def test_scenario_cb_sets_contact_at_inflection():
    G, dPdG = _decline_with_inflection()
    state = PickState(closure_scenario="C-B adequate")
    hint = picks.apply_closure_scenario(state, _res_with(G, dPdG))
    assert hint is None
    assert abs(state.contact_G - 6.0) < 0.2


def test_scenario_cb_hints_when_no_inflection():
    G, dPdG = _monotonic_decline()
    state = PickState(closure_scenario="C-B adequate", contact_G=3.0)
    hint = picks.apply_closure_scenario(state, _res_with(G, dPdG))
    assert hint is not None
    assert state.contact_G == 3.0


def test_scenario_cc_cd_clear_contact():
    G, dPdG = _s_curve()
    for scen in ("C-C no-contact", "C-D rapid"):
        state = PickState(closure_scenario=scen, contact_G=7.0, min_dpdg_G=5.0)
        assert picks.apply_closure_scenario(state, _res_with(G, dPdG)) is None
        assert state.contact_G is None
        assert state.min_dpdg_G == 5.0  # the diagnostic pick survives


def test_scenario_noop_cases():
    G, dPdG = _s_curve()
    # empty scenario
    state = PickState(closure_scenario="", contact_G=7.0)
    assert picks.apply_closure_scenario(state, _res_with(G, dPdG)) is None
    assert state.contact_G == 7.0
    # missing diagnostics
    state = PickState(closure_scenario="C-A clear", contact_G=7.0)
    assert picks.apply_closure_scenario(state, DerivedResults()) is None
    assert state.contact_G == 7.0


def test_scenario_cb_seeds_min_when_unset():
    """New intentional side effect (plan step 2): selecting C-B also seeds min_dpdg_G, needed
    for the triangle to appear per decision 4."""
    G, dPdG = _decline_with_inflection()
    state = PickState(closure_scenario="C-B adequate")
    assert state.min_dpdg_G is None
    picks.apply_closure_scenario(state, _res_with(G, dPdG))
    assert state.min_dpdg_G is not None


# --------------------------------------------------------------------------------------------------
# re_derive_contact_from_min: the triangle-drag re-derive path (decision D4)
# --------------------------------------------------------------------------------------------------
def test_re_derive_contact_from_min_ca_follows_dragged_min():
    G, dPdG = _s_curve()
    state = PickState(closure_scenario="C-A clear", min_dpdg_G=6.0)
    hint = picks.re_derive_contact_from_min(state, _res_with(G, dPdG))
    assert hint is None
    # threshold from the value AT the dragged min: (50 + 1) * 1.1 = 56.1 -> (G-5)^2 >= 6.1
    assert abs(state.contact_G - (5.0 + np.sqrt(6.1))) < 0.1


def test_re_derive_contact_from_min_cb_follows_dragged_seed():
    G, dPdG = _two_bump_decline()
    res = _res_with(G, dPdG)
    state_low = PickState(closure_scenario="C-B adequate", min_dpdg_G=3.0)
    state_high = PickState(closure_scenario="C-B adequate", min_dpdg_G=9.0)
    assert picks.re_derive_contact_from_min(state_low, res) is None
    assert picks.re_derive_contact_from_min(state_high, res) is None
    assert abs(state_low.contact_G - 3.0) < abs(state_high.contact_G - 3.0)
    assert abs(state_high.contact_G - 9.0) < abs(state_low.contact_G - 9.0)


def test_re_derive_contact_from_min_no_op_cases():
    G, dPdG = _s_curve()
    res = _res_with(G, dPdG)
    # blank scenario
    state = PickState(closure_scenario="", min_dpdg_G=5.0, contact_G=1.0)
    assert picks.re_derive_contact_from_min(state, res) is None
    assert state.contact_G == 1.0
    # C-C / C-D
    for scen in ("C-C no-contact", "C-D rapid"):
        state = PickState(closure_scenario=scen, min_dpdg_G=5.0, contact_G=1.0)
        assert picks.re_derive_contact_from_min(state, res) is None
        assert state.contact_G == 1.0
    # no min pick yet
    state = PickState(closure_scenario="C-A clear", contact_G=1.0)
    assert picks.re_derive_contact_from_min(state, res) is None
    assert state.contact_G == 1.0
    # missing diagnostics
    state = PickState(closure_scenario="C-A clear", min_dpdg_G=5.0, contact_G=1.0)
    assert picks.re_derive_contact_from_min(state, DerivedResults()) is None
    assert state.contact_G == 1.0


def test_re_derive_contact_from_min_ca_hints_when_no_rise():
    G, dPdG = _monotonic_decline()
    state = PickState(closure_scenario="C-A clear", min_dpdg_G=5.0, contact_G=1.0)
    hint = picks.re_derive_contact_from_min(state, _res_with(G, dPdG))
    assert hint is not None
    assert state.contact_G == 1.0


from dfit_tool.resample import Resampled


def _res_with_resampled(G, dPdG):
    res = _res_with(G, dPdG)
    res.resampled = Resampled(dt=G, p=res.diagnostics.p, n_raw=len(G))
    return res


def test_gfunction_hint_text_per_scenario():
    assert "triangle" in picks.gfunction_hint_text("C-A clear")
    assert "triangle" in picks.gfunction_hint_text("C-B adequate")
    assert picks.gfunction_hint_text("C-C no-contact") == "No contact pick applies for this closure scenario."
    assert picks.gfunction_hint_text("C-D rapid") == "No contact pick applies for this closure scenario."
    assert picks.gfunction_hint_text("") == picks._GFUNCTION_HINT_DEFAULT


import matplotlib.pyplot as plt

from dfit_tool import plots
from tests.helpers import make_testdata, injection_state


def _gfunction_fixture(**state_kw):
    td = make_testdata()
    state = injection_state(td)
    for k, v in state_kw.items():
        setattr(state, k, v)
    res = compute_all(state, td)
    assert res.diagnostics is not None
    fig, ax = plt.subplots()
    plots.render_gfunction(ax, td, state, res)
    return fig


def _gids(fig):
    return {ln.get_gid() for a in fig.axes for ln in a.get_lines() if ln.get_gid()}


def test_render_gfunction_contact_vline():
    fig = _gfunction_fixture(contact_G=1.5)
    assert "contact_vline" in _gids(fig)
    plt.close(fig)
    fig = _gfunction_fixture()  # no contact pick -> no vline
    assert "contact_vline" not in _gids(fig)
    plt.close(fig)


def test_render_gfunction_d2_toggle():
    fig = _gfunction_fixture(show_d2pdg2=True)
    assert "d2pdg2_curve" in _gids(fig)
    plt.close(fig)
    fig = _gfunction_fixture(show_d2pdg2=False)
    assert "d2pdg2_curve" not in _gids(fig)
    plt.close(fig)


def _gfunction_defaults(**state_kw):
    td = make_testdata()
    state = injection_state(td)
    for k, v in state_kw.items():
        setattr(state, k, v)
    res = compute_all(state, td)
    assert res.diagnostics is not None
    fig, ax = plt.subplots()
    defaults = plots.render_gfunction(ax, td, state, res)
    return fig, defaults


def test_render_gfunction_triangle_hidden_for_blank_cc_cd_shown_for_ca_cb():
    for scen in ("", "C-C no-contact", "C-D rapid"):
        fig = _gfunction_fixture(min_dpdg_G=3.0, closure_scenario=scen)
        assert "min_dpdg_point" not in _gids(fig), scen
        plt.close(fig)
    for scen in ("C-A clear", "C-B adequate"):
        fig = _gfunction_fixture(min_dpdg_G=3.0, closure_scenario=scen)
        assert "min_dpdg_point" in _gids(fig), scen
        plt.close(fig)


def test_render_gfunction_d2_axis_has_own_gid():
    fig, defaults = _gfunction_defaults(show_d2pdg2=True)
    d2_axes = [a for a in fig.axes if a.get_gid() == plots.D2_AXIS_GID]
    assert len(d2_axes) == 1
    plt.close(fig)
    fig, defaults = _gfunction_defaults(show_d2pdg2=False)
    assert not any(a.get_gid() == plots.D2_AXIS_GID for a in fig.axes)
    plt.close(fig)


def test_render_gfunction_y3lim_only_returned_when_toggled():
    fig, defaults = _gfunction_defaults(show_d2pdg2=True)
    assert defaults.y3lim is not None
    plt.close(fig)
    fig, defaults = _gfunction_defaults(show_d2pdg2=False)
    assert defaults.y3lim is None
    plt.close(fig)


def test_render_gfunction_y3lim_ignores_early_spike():
    """The d2 axis default scale comes from G >= 1 samples only -- the resampled grid is
    densest across the early water-hammer spike, so unmasked percentiles would blow up the
    (slider-less) default view."""
    td = make_testdata()
    state = injection_state(td)
    state.show_d2pdg2 = True
    res = compute_all(state, td)
    dg = res.diagnostics
    early = dg.G < 1.0
    assert early.any() and (~early).any()
    dg.d2PdG2[early] = 1e6  # simulated water-hammer spike
    fig, ax = plt.subplots()
    defaults = plots.render_gfunction(ax, td, state, res)
    plt.close(fig)
    lo, hi = defaults.y3lim
    assert -1e5 < lo < hi < 1e5


def test_show_d2pdg2_round_trips(tmp_path):
    p = tmp_path / "picks.json"
    state = PickState(show_d2pdg2=True)
    state.to_json(str(p))
    assert PickState.from_json(str(p)).show_d2pdg2 is True
    # an old save without the key defaults False
    state2 = PickState()
    state2.to_json(str(p))
    assert PickState.from_json(str(p)).show_d2pdg2 is False


# --------------------------------------------------------------------------------------------------
# seed_gfunction (blank scenario): the contact placeholder is now interpret.suggest_hump_index
# (G*dP/dG argmax), not a raw nanargmax(dPdG) -- the latter lands on the early water-hammer spike
# whenever the spike outweighs the real hump.
# --------------------------------------------------------------------------------------------------
def _spike_hump_curve():
    """Local fixture: raw argmax(dPdG) lands on the spike near G~=0, but the real hump (found via
    G*dP/dG) sits at G~=3.5 -- same shape as tests/test_interpret.py's
    _spike_elbow_hump_curve, kept local here per this file's convention of file-local fixtures."""
    G = np.linspace(0.001, 8.0, 400)
    spike = 50.0 * np.exp(-G / 0.05)
    elbow_G, hump_G, min_val, hump_val, tail_val = 0.55, 3.5, 2.0, 6.0, 1.0
    left = min_val + 20.0 * (elbow_G - G) ** 2
    right_rise = min_val + (hump_val - min_val) * (G - elbow_G) / (hump_G - elbow_G)
    right_tail = tail_val + (hump_val - tail_val) * np.exp(-(G - hump_G) / 1.5)
    base = np.where(G < elbow_G, left, np.where(G <= hump_G, right_rise, right_tail))
    return G, spike + base


def test_seed_gfunction_contact_at_hump_not_spike():
    G, dPdG = _spike_hump_curve()
    res = _res_with_resampled(G, dPdG)
    # A preset scenario keeps the auto C-A seed (tests/test_closure_autoassign.py) from moving
    # the contact to the +10% point: this pins the hump placeholder itself.
    state = PickState(closure_scenario="C-B adequate")
    picks.seed_gfunction(state, res)
    assert abs(state.min_dpdg_G - 0.55) < 0.15
    assert abs(state.contact_G - 3.5) < 0.3
    assert state.contact_G > 1.0  # nowhere near the spike at G~=0


# --------------------------------------------------------------------------------------------------
# handle_min_dpdg_window: the Shift+drag window-correction commit (ui.py's ModifierSpanController)
# --------------------------------------------------------------------------------------------------
def test_handle_min_dpdg_window_ca_finds_argmin_in_window_and_sets_contact():
    G, dPdG = _s_curve()  # dips to a min at G=5 then rises
    state = PickState(closure_scenario="C-A clear", min_dpdg_G=1.0)
    hint = picks.handle_min_dpdg_window(state, _res_with(G, dPdG), 4.0, 6.0)
    assert hint is None
    assert abs(state.min_dpdg_G - 5.0) < 0.1
    # both picks set from the window-found min -- the +10% rise rule seeded from it, not a
    # separate g_min-masked re-derive that could land on a different point outside the window.
    assert abs(state.contact_G - (5.0 + np.sqrt(5.0))) < 0.1


def test_handle_min_dpdg_window_ca_never_rises_hints_and_leaves_contact():
    """C-A window finds a min but the rise rule never clears -- min still moves (matching the
    drag path's partial-failure shape), contact is left untouched, and the hint is the same
    re-derive failure string used elsewhere."""
    G, dPdG = _monotonic_decline()
    state = PickState(closure_scenario="C-A clear", min_dpdg_G=1.0, contact_G=99.0)
    hint = picks.handle_min_dpdg_window(state, _res_with(G, dPdG), 4.0, 6.0)
    assert hint == ("dP/dG never rises 10% above the min -- not a clear contact "
                    "(consider C-B or C-C).")
    assert state.min_dpdg_G != 1.0  # moved to the window's min
    assert state.contact_G == 99.0  # unchanged


def test_handle_min_dpdg_window_cb_finds_inflection_in_window():
    """Regression for the bug this fix addresses: a window-found C-B inflection (G=0.6) below
    the re-derive's default g_min=1.0 mask used to get silently re-derived to a DIFFERENT
    inflection outside the window (G=6.0) by the old two-call on_span path. Both picks must land
    on the window-found inflection, not the g_min-masked one (same two-bump construction as
    test_inflection_rule_seed_picks_nearest_of_two_bumps above, moved to G=0.6/G=6.0 so one bump
    sits below g_min=1.0)."""
    G = np.linspace(0.0, 12.0, 481)
    d2_target = (-1.0 + 3.0 * np.exp(-((G - 0.6) ** 2) / (2 * 0.15 ** 2))
                 + 2.0 * np.exp(-((G - 6.0) ** 2) / (2 * 0.4 ** 2)))
    dPdG = np.cumsum(d2_target) * (G[1] - G[0])
    state = PickState(closure_scenario="C-B adequate", min_dpdg_G=1.0)
    # sanity: the g_min-masked default (what the old buggy re-derive would have used) lands on
    # the OTHER bump, at G=6.0 -- confirming the window is doing the work, not a coincidence.
    default_idx = interpret.suggest_contact_inflection_index(G, dPdG, d2=np.gradient(dPdG, G))
    assert abs(G[default_idx] - 6.0) < 0.3

    hint = picks.handle_min_dpdg_window(state, _res_with(G, dPdG), 0.2, 1.2)
    assert hint is None
    assert abs(state.min_dpdg_G - 0.6) < 0.1
    assert abs(state.contact_G - 0.6) < 0.1


def test_handle_min_dpdg_window_ca_empty_window_hints_and_leaves_pick():
    G, dPdG = _s_curve()
    state = PickState(closure_scenario="C-A clear", min_dpdg_G=1.0, contact_G=99.0)
    hint = picks.handle_min_dpdg_window(state, _res_with(G, dPdG), 50.0, 60.0)
    assert hint is not None
    assert state.min_dpdg_G == 1.0
    assert state.contact_G == 99.0


def test_handle_min_dpdg_window_cb_empty_window_hints_and_leaves_pick():
    G, dPdG = _decline_with_inflection()
    state = PickState(closure_scenario="C-B adequate", min_dpdg_G=1.0, contact_G=99.0)
    hint = picks.handle_min_dpdg_window(state, _res_with(G, dPdG), 50.0, 60.0)
    assert hint is not None
    assert state.min_dpdg_G == 1.0
    assert state.contact_G == 99.0


def test_handle_min_dpdg_window_cc_cd_no_op():
    G, dPdG = _s_curve()
    for scen in ("C-C no-contact", "C-D rapid"):
        state = PickState(closure_scenario=scen, min_dpdg_G=1.0)
        hint = picks.handle_min_dpdg_window(state, _res_with(G, dPdG), 4.0, 6.0)
        assert hint is None
        assert state.min_dpdg_G == 1.0


def test_handle_min_dpdg_window_blank_and_missing_diagnostics_no_op():
    G, dPdG = _s_curve()
    state = PickState(closure_scenario="", min_dpdg_G=1.0)
    assert picks.handle_min_dpdg_window(state, _res_with(G, dPdG), 4.0, 6.0) is None
    assert state.min_dpdg_G == 1.0
    state2 = PickState(closure_scenario="C-A clear", min_dpdg_G=1.0)
    assert picks.handle_min_dpdg_window(state2, DerivedResults(), 4.0, 6.0) is None
    assert state2.min_dpdg_G == 1.0
