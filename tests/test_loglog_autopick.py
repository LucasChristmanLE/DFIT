"""Log-log window auto-pick (interpret.suggest_loglog_window, picks.seed_loglog) and the PC-A
auto-assignment (picks.auto_assign_postclosure) that follows a near -1/2 window slope."""

from __future__ import annotations

import types

import numpy as np

from dfit_tool import interpret, picks, store
from dfit_tool.model import PickState, _decode, compute_all
from dfit_tool.ui import DfitApp
from tests.helpers import injection_state, make_testdata


def _shaped(late_slope: float, spike: float = 300.0):
    """t*dP/dt shaped like 7170: early spike, trough, rise to a hump at 1.5e5 s, then a
    power-law decline with ``late_slope`` out to 1.2e6 s."""
    t_early = np.array([4.0, 5.0, 6.0, 7.0, 12.0, 15.0, 36.0, 100.0])
    y_early = np.array([200.0, 280.0, spike, 250.0, 130.0, 170.0, 45.0, 42.0])
    t_rise = np.logspace(np.log10(220.0), np.log10(1.5e5), 25)
    y_rise = 42.0 * (t_rise / 100.0) ** (np.log10(390.0 / 42.0) / np.log10(1.5e3))
    t_fall = np.logspace(np.log10(1.5e5), np.log10(1.2e6), 16)[1:]
    y_fall = 390.0 * (t_fall / 1.5e5) ** late_slope
    t = np.concatenate([t_early, t_rise, t_fall])
    y = np.concatenate([y_early, y_rise, y_fall])
    peak = len(t_early) + len(t_rise) - 1
    return t, y, peak


# --------------------------------------------------------------------------------------------------
# suggest_loglog_window
# --------------------------------------------------------------------------------------------------
def test_window_starts_after_peak_on_half_slope():
    t, y, peak = _shaped(-0.5)
    i0, i1, _ = interpret.suggest_loglog_window(t, y)
    assert i0 > peak
    s = interpret.loglog_window_slope(t, y, t[i0], t[i1])
    assert abs(s + 0.5) <= 0.1


def test_early_spike_taller_than_hump_is_skipped():
    t, y, peak = _shaped(-0.5, spike=900.0)
    i0, _, _ = interpret.suggest_loglog_window(t, y)
    assert i0 > peak


def test_latest_prominent_peak_wins_over_taller_early_hump():
    # Caprito 99-202H shape: a tiny first sample, a tall early hump at ~280 s, a deep trough,
    # then a second (postclosure) hump at ~4.8e4 s with a -1/2 decline after it.
    t_a = np.logspace(np.log10(3.0), np.log10(280.0), 20)
    y_a = 60.0 * (t_a / 3.0) ** (np.log10(2300.0 / 60.0) / np.log10(280.0 / 3.0))
    t_b = np.logspace(np.log10(300.0), np.log10(2700.0), 10)
    y_b = 2300.0 * (t_b / 280.0) ** (np.log10(99.0 / 2300.0) / np.log10(2700.0 / 280.0))
    t_c = np.logspace(np.log10(3000.0), np.log10(4.8e4), 15)
    y_c = 99.0 * (t_c / 2700.0) ** (np.log10(398.0 / 99.0) / np.log10(4.8e4 / 2700.0))
    t_d = np.logspace(np.log10(4.8e4), np.log10(5e5), 16)[1:]
    y_d = 398.0 * (t_d / 4.8e4) ** -0.5
    t = np.concatenate([t_a, t_b, t_c, t_d])
    y = np.concatenate([y_a, y_b, y_c, y_d])
    peak = len(t_a) + len(t_b) + len(t_c) - 1
    i0, i1, _ = interpret.suggest_loglog_window(t, y)
    assert i0 > peak
    assert abs(interpret.loglog_window_slope(t, y, t[i0], t[i1]) + 0.5) <= 0.1


def test_no_half_slope_takes_straightest_section_not_closest_slope():
    # Peak, a short curved rollover (slope 0 -> -1 over 0.2 decades, too short to qualify as
    # -1/2), then a long straight -1 decline. The rollover holds the slope closest to -1/2;
    # the straight -1 section must win instead.
    t_rise = np.logspace(1, 4, 20)
    v_rise = np.log10(10.0) + 0.5 * np.log10(t_rise)
    x0, v0 = 4.0, v_rise[-1]
    x_roll = np.linspace(x0, x0 + 0.2, 7)[1:]
    v_roll = v0 - 2.5 * (x_roll - x0) ** 2  # slope -5*(x-x0): 0 -> -1 at x0+0.2
    x_lin = np.linspace(x0 + 0.2, x0 + 1.2, 16)[1:]
    v_lin = v_roll[-1] - (x_lin - (x0 + 0.2))
    t = np.concatenate([t_rise, 10.0 ** x_roll, 10.0 ** x_lin])
    y = 10.0 ** np.concatenate([v_rise, v_roll, v_lin])
    i0, i1, half = interpret.suggest_loglog_window(t, y)
    assert half is False
    assert i1 == len(t) - 1  # spans the straight -1 section, not a piece of the rollover
    assert abs(interpret.loglog_window_slope(t, y, t[i0], t[i1]) + 1.0) < 0.1


def test_rising_derivative_returns_none():
    t = np.logspace(0, 6, 40)
    y = 10.0 * t ** 0.3
    assert interpret.suggest_loglog_window(t, y) is None


def test_unit_slope_decline_still_after_peak():
    t, y, peak = _shaped(-1.0)
    i0, i1, _ = interpret.suggest_loglog_window(t, y)
    assert i0 > peak
    s = interpret.loglog_window_slope(t, y, t[i0], t[i1])
    assert abs(s + 1.0) < 0.1


def test_window_slope_includes_both_edges():
    t = np.array([1.0, 10.0, 100.0])
    y = np.array([1.0, 0.1, 1.0])  # the right edge flips the sign if it is dropped
    assert abs(interpret.loglog_window_slope(t, y, 1.0, 10.0) + 1.0) < 1e-9
    assert np.isnan(interpret.loglog_window_slope(t, y, 2.0, 9.0))


# --------------------------------------------------------------------------------------------------
# auto_assign_postclosure
# --------------------------------------------------------------------------------------------------
def test_auto_assign_sets_pca_on_hit():
    st = PickState(pp_axis="tm1")
    hint = picks.auto_assign_postclosure(st, -0.48)
    assert st.postclosure_scenario == "PC-A linear"
    assert st.postclosure_auto is True
    assert st.pp_axis == "tm12"
    assert "-0.48" in hint and "PC-A" in hint


def test_auto_assign_leaves_blank_on_miss():
    st = PickState()
    assert picks.auto_assign_postclosure(st, -0.95) is None
    assert st.postclosure_scenario == ""
    assert st.postclosure_auto is False


def test_auto_assign_never_overrides_manual_scenario():
    st = PickState(postclosure_scenario="PC-B false-radial")
    assert picks.auto_assign_postclosure(st, -0.5) is None
    assert st.postclosure_scenario == "PC-B false-radial"
    assert st.postclosure_auto is False


def test_auto_assign_clears_auto_pca_on_miss():
    st = PickState()
    picks.auto_assign_postclosure(st, -0.5)
    hint = picks.auto_assign_postclosure(st, -1.0)
    assert st.postclosure_scenario == ""
    assert st.postclosure_auto is False
    assert hint and "cleared" in hint


def test_auto_assign_nan_slope_clears_auto():
    st = PickState()
    picks.auto_assign_postclosure(st, -0.5)
    picks.auto_assign_postclosure(st, float("nan"))
    assert st.postclosure_scenario == ""


def test_decode_old_save_defaults_auto_false():
    st = _decode({"postclosure_scenario": "PC-A linear"})
    assert st.postclosure_auto is False


# --------------------------------------------------------------------------------------------------
# seed_loglog + compute_all
# --------------------------------------------------------------------------------------------------
def _seeded_state():
    td = make_testdata()
    st = injection_state(td)
    picks.seed_injection(st, td)
    res = compute_all(st, td)
    picks.seed_isip(st, td, res)
    res = compute_all(st, td)
    picks.seed_gfunction(st, res)
    picks.seed_tangent(st, res)
    return td, st, compute_all(st, td)


def test_seed_loglog_window_and_auto_pick_agree_with_compute_all():
    td, st, res = _seeded_state()
    picks.seed_loglog(st, res)
    assert st.loglog_window is not None
    res = compute_all(st, td)
    assert res.loglog_slope is not None
    hit = abs(res.loglog_slope + 0.5) <= interpret.LOGLOG_HALF_SLOPE_TOL
    assert (st.postclosure_scenario == "PC-A linear") == hit
    assert st.postclosure_auto == hit


def test_seed_loglog_does_not_touch_manual_scenario():
    td, st, res = _seeded_state()
    st.postclosure_scenario = "PC-E no trend"
    picks.seed_loglog(st, res)
    assert st.postclosure_scenario == "PC-E no trend"


def test_compute_all_loglog_slope_none_without_window():
    td, st, res = _seeded_state()
    assert res.loglog_slope is None


# --------------------------------------------------------------------------------------------------
# ui: a manual scenario change clears the auto flag; the log carries it
# --------------------------------------------------------------------------------------------------
class _Var:
    def __init__(self, v):
        self.v = v

    def get(self):
        return self.v

    def set(self, v):
        self.v = v


def _scenario_stub(st, pcscen):
    stub = types.SimpleNamespace(
        state=st, td=None, step="loglog",
        var_cscen=_Var(""), var_pcscen=_Var(pcscen), var_ppaxis=_Var("tm12"),
        hint_lbl=types.SimpleNamespace(config=lambda **kw: None),
        _update_ppaxis_enabled=lambda: None, refresh=lambda: None, _goto=lambda s: None)
    stub._on_scenario = lambda: DfitApp._on_scenario(stub)
    return stub


def test_on_scenario_manual_change_clears_auto_flag():
    st = PickState()
    picks.auto_assign_postclosure(st, -0.5)
    DfitApp._on_pcscen_selected(_scenario_stub(st, "PC-B false-radial"))
    assert st.postclosure_scenario == "PC-B false-radial"
    assert st.postclosure_auto is False


def test_reselecting_auto_pca_confirms_it():
    st = PickState()
    picks.auto_assign_postclosure(st, -0.5)
    DfitApp._on_pcscen_selected(_scenario_stub(st, "PC-A linear"))
    assert st.postclosure_auto is False
    assert picks.auto_assign_postclosure(st, -1.0) is None
    assert st.postclosure_scenario == "PC-A linear"


def test_closure_scenario_change_keeps_auto_flag():
    st = PickState()
    picks.auto_assign_postclosure(st, -0.5)
    DfitApp._on_scenario(_scenario_stub(st, "PC-A linear"))
    assert st.postclosure_auto is True


def test_seed_does_not_auto_assign_from_fallback_window():
    # Hump, then a -1/2 decline with alternating +-0.1-decade noise: no window passes the
    # straightness check, so the fallback window is used and must not set PC-A even though
    # its slope is near -1/2.
    t_rise = np.logspace(1, 4, 20)
    y_rise = 10.0 * t_rise ** 0.5
    t_fall = 1e4 * np.logspace(0, 0.8, 13)[1:]
    noise = 10.0 ** (0.1 * (-1.0) ** np.arange(t_fall.size))
    y_fall = y_rise[-1] * (t_fall / 1e4) ** -0.5 * noise
    t = np.concatenate([t_rise, t_fall])
    y = np.concatenate([y_rise, y_fall])
    win = interpret.suggest_loglog_window(t, y)
    assert win is not None and win[2] is False
    st = PickState()
    res = types.SimpleNamespace(diagnostics=types.SimpleNamespace(t=t, tdpdt=y))
    picks.seed_loglog(st, res)
    assert st.loglog_window is not None
    assert st.postclosure_scenario == ""


def test_log_column_postclosure_auto_at_end(tmp_path):
    assert store.LOG_COLUMNS[-9] == "postclosure_auto"
    td, st, res = _seeded_state()
    picks.auto_assign_postclosure(st, -0.5)
    entry = store.TestEntry(test_id="well1", folder=str(tmp_path))
    row = store.build_log_row(entry, str(tmp_path / "well1.csv"), str(tmp_path), st, td,
                              compute_all(st, td))
    assert row["postclosure_auto"] is True
