"""Auto C-A on the gfunction seed: interpret.is_clear_closure (genuine interior min, a >= 10% rise
reached at or before the hump and held for CLEAR_RISE_MIN_POINTS samples) and its use in
picks.seed_gfunction. Nothing is flagged: an auto C-A is an ordinary C-A in state."""

import types

import numpy as np

from dfit_tool import interpret, picks
from dfit_tool.model import DerivedResults, PickState
from dfit_tool.resample import Diagnostics, Resampled
from dfit_tool.ui import DfitApp


def _res(G, dPdG):
    z = np.zeros_like(G)
    dg = Diagnostics(G=G, dPdG=dPdG, GdPdG=G * dPdG, d2PdG2=np.gradient(dPdG, G),
                     t=G, p=z, dp=z, tdpdt=z)
    return DerivedResults(diagnostics=dg, resampled=Resampled(dt=G, p=z, n_raw=len(G)))


def _s_curve():
    """Clean C-A: decline to a min at G=5, rise to a hump at G=8, decline after."""
    G = np.linspace(0.0, 12.0, 241)
    rise = 50.0 + (G - 5.0) ** 2
    tail = 59.0 - 2.0 * (G - 8.0)
    return G, np.where(G <= 8.0, rise, tail)


def _monotonic_decline():
    G = np.linspace(0.0, 12.0, 241)
    return G, 100.0 * np.exp(-G / 3.0)


def _blip_curve(n_high):
    """Decline to a min of 50 at G=5, then a flat 52 floor with ``n_high`` consecutive samples at
    58 (+16%) starting at G=6, then a slow decline. Only the blip clears 110% of the min."""
    G = np.linspace(0.0, 12.0, 241)
    y = np.where(G <= 5.0, 50.0 + (G - 5.0) ** 2, 52.0 - 0.1 * np.clip(G - 9.0, 0.0, None))
    start = int(np.searchsorted(G, 6.0))
    y[start:start + n_high] = 58.0
    return G, y


def _late_tail_rise():
    """Min at G=5, a small +4% bump at G=6.5 (the hump), then a decline and a tail that climbs
    past +10% only near the end, after the hump."""
    G = np.linspace(0.0, 12.0, 241)
    y = np.where(G <= 5.0, 50.0 + (G - 5.0) ** 2, 50.0 + 2.0 * np.exp(-((G - 6.5) / 0.5) ** 2))
    late = G > 10.0
    y[late] = 50.0 + 4.0 * (G[late] - 10.0)
    return G, y


def _min_idx(G, y):
    return interpret.suggest_min_dpdg_index(G, y)


# --------------------------------------------------------------------------------------------------
# interpret.is_clear_closure
# --------------------------------------------------------------------------------------------------
def test_clean_s_curve_is_clear():
    G, y = _s_curve()
    assert interpret.is_clear_closure(G, y, _min_idx(G, y))


def test_monotonic_decline_fallback_min_is_not_clear():
    G, y = _monotonic_decline()
    assert not interpret.is_clear_closure(G, y, _min_idx(G, y))


def test_single_sample_spike_is_not_clear():
    G, y = _blip_curve(1)
    assert not interpret.is_clear_closure(G, y, _min_idx(G, y))


def test_two_point_rise_is_not_clear():
    G, y = _blip_curve(2)
    assert not interpret.is_clear_closure(G, y, _min_idx(G, y))


def test_three_point_rise_is_clear():
    G, y = _blip_curve(interpret.CLEAR_RISE_MIN_POINTS)
    assert interpret.is_clear_closure(G, y, _min_idx(G, y))


def test_rise_only_after_hump_is_not_clear():
    G, y = _late_tail_rise()
    assert not interpret.is_clear_closure(G, y, _min_idx(G, y))


def test_non_positive_min_is_not_clear():
    """110% of a min <= 0 is not a rise above it, so a pressure uptick (dP/dG <= 0) never counts."""
    G = np.linspace(0.0, 12.0, 241)
    y = np.where(G <= 5.0, (G - 5.0) ** 2 - 1.0, -1.0 + 0.2 * (G - 5.0))
    y[G > 8.0] = -0.4 - 0.1 * (G[G > 8.0] - 8.0)
    assert y[_min_idx(G, y)] < 0
    assert not interpret.is_clear_closure(G, y, _min_idx(G, y))


def test_all_nan_and_short_curves_are_not_clear():
    G = np.linspace(0.0, 1.0, 10)
    assert not interpret.is_clear_closure(G, np.full(10, np.nan), 0)
    G3 = np.array([0.0, 1.0, 2.0])
    assert not interpret.is_clear_closure(G3, np.array([5.0, 4.0, 9.0]), 1)


# --------------------------------------------------------------------------------------------------
# picks.seed_gfunction
# --------------------------------------------------------------------------------------------------
def test_seed_auto_assigns_ca_and_moves_contact_to_rise_point():
    G, y = _s_curve()
    state = PickState()
    hint = picks.seed_gfunction(state, _res(G, y))
    assert state.closure_scenario == "C-A clear"
    assert hint == picks.closure_auto_hint()
    min_idx = _min_idx(G, y)
    assert state.min_dpdg_G == float(G[min_idx])
    assert state.contact_G == float(G[interpret.suggest_contact_clear_index(y, min_idx)])


def test_seed_leaves_scenario_blank_on_no_contact_shape():
    G, y = _monotonic_decline()
    state = PickState()
    assert picks.seed_gfunction(state, _res(G, y)) is None
    assert state.closure_scenario == ""


def test_seed_never_overrides_a_preset_scenario():
    G, y = _s_curve()
    state = PickState(closure_scenario="C-B adequate")
    assert picks.seed_gfunction(state, _res(G, y)) is None
    assert state.closure_scenario == "C-B adequate"


def test_seed_does_not_auto_assign_over_an_existing_min_pick():
    G, y = _s_curve()
    state = PickState(min_dpdg_G=3.0)
    assert picks.seed_gfunction(state, _res(G, y)) is None
    assert state.closure_scenario == ""
    assert state.min_dpdg_G == 3.0


# --------------------------------------------------------------------------------------------------
# ui wiring: _seed_step pushes the auto scenario into the combobox and stashes the one-shot hint
# --------------------------------------------------------------------------------------------------
def test_dfit_app_seed_step_gfunction_updates_combobox_and_stashes_hint(monkeypatch):
    from tests.helpers import injection_state, make_testdata

    def fake_seed(state, res):
        state.closure_scenario = "C-A clear"
        return "auto hint"

    monkeypatch.setitem(picks.SEEDERS, "gfunction", fake_seed)
    td = make_testdata()
    stub = types.SimpleNamespace(td=td, state=injection_state(td))
    stub.var_cscen = types.SimpleNamespace(value=None)
    stub.var_cscen.set = lambda v: setattr(stub.var_cscen, "value", v)
    stub._seed_step = types.MethodType(DfitApp._seed_step, stub)

    hint = stub._seed_step("gfunction")

    assert stub.var_cscen.value == "C-A clear"
    assert hint == "auto hint"


def test_dfit_app_goto_shows_seed_hint_after_refresh():
    """The seed hint is applied after refresh(), which resets hint_lbl to the step default."""
    from dfit_tool.model import PickState as PS

    shown = []
    stub = types.SimpleNamespace(td=object(), state=PS(), step="overview")
    stub._seed_step = lambda key: "auto hint"
    stub.refresh = lambda: shown.append("refresh")
    stub.hint_lbl = types.SimpleNamespace(config=lambda text: shown.append(text))
    stub._goto = types.MethodType(DfitApp._goto, stub)

    stub._goto("overview")  # a bare PickState blocks every later step

    assert shown == ["refresh", "auto hint"]


# --------------------------------------------------------------------------------------------------
# picks.seed_gfunction: the auto-seed never lands below interpret.SEED_MIN_G (G = 1)
# --------------------------------------------------------------------------------------------------
def _noisy_early_curve():
    """1BH MERGED shape: early-decline noise below G=1 (alternating 800 / 80, so every other
    sample is a local min/max and the G*dP/dG of the noise beats the real bump), then a smooth
    decline to a shallow min of 50 at G=5, a 2% bump (51) at G=8, and a slow decline."""
    G = np.linspace(0.02, 12.0, 600)
    y = np.empty_like(G)
    early = G < 1.0
    y[early] = np.where(np.arange(early.sum()) % 2 == 0, 8000.0, 800.0)
    a = (G >= 1.0) & (G < 5.0)
    y[a] = 50.0 + 30.0 * (G[a] - 5.0) ** 2
    mid = (G >= 5.0) & (G <= 8.0)
    y[mid] = 50.0 + (G[mid] - 5.0) ** 2 / 9.0
    y[G > 8.0] = 51.0 - 0.5 * (G[G > 8.0] - 8.0)
    return G, y


def test_seed_ignores_noise_below_g_1():
    G, y = _noisy_early_curve()
    state = PickState()
    hint = picks.seed_gfunction(state, _res(G, y))
    assert abs(state.min_dpdg_G - 5.0) < 0.05
    assert abs(state.contact_G - 8.0) < 0.05
    assert state.closure_scenario == ""  # a 2% bump is not a clear C-A
    assert hint is None


def test_seed_never_below_g_1_even_for_a_clean_sub_1_elbow():
    """A real elbow below G=1 is not auto-seeded; the analyst drags it. The seed stays >= 1."""
    G = np.linspace(0.02, 8.0, 400)
    y = np.where(G < 0.55, 2.0 + 20.0 * (0.55 - G) ** 2, 2.0 + (G - 0.55))
    state = PickState()
    picks.seed_gfunction(state, _res(G, y))
    assert state.min_dpdg_G >= interpret.SEED_MIN_G
    assert state.contact_G >= interpret.SEED_MIN_G


def test_seed_floor_falls_back_when_the_record_barely_reaches_g_1():
    """Fewer than 6 samples at G >= 1: seed on the whole curve, as before."""
    G = np.linspace(0.0, 1.01, 200)
    y = 50.0 + (G - 0.5) ** 2
    state = PickState()
    picks.seed_gfunction(state, _res(G, y))
    assert state.min_dpdg_G == float(G[_min_idx(G, y)])


def test_seed_ignores_a_terminal_crash_spike():
    """Flaherty / Delphi shape: the record ends mid-bleed-off, so the last samples have dP/dG
    100x the settled curve (the final one negative where pressure ticks back up). The trim cannot
    catch it (BHP channel, or the bleed stops above 100 psi), so the seed must."""
    G, y = _noisy_early_curve()
    G = np.append(G, G[-1] + np.array([0.001, 0.002, 0.003, 0.004, 0.005]))
    y = np.append(y, [9000.0, 9500.0, 8700.0, 5300.0, -7800.0])
    state = PickState()
    hint = picks.seed_gfunction(state, _res(G, y))
    assert abs(state.min_dpdg_G - 5.0) < 0.05
    assert abs(state.contact_G - 8.0) < 0.05
    assert state.closure_scenario == ""
    assert hint is None


def test_seed_ignores_a_long_terminal_crash_that_outnumbers_the_curve():
    """Delphi shape: a BHP channel bleeds off to ~0 psi, and the 30-psi resampler keeps more
    crash points (all packed in the last 0.01% of G) than real falloff points."""
    G, y = _noisy_early_curve()
    keep = G >= 1.0
    G, y = G[keep], y[keep]
    n_crash = 2 * len(G)
    G = np.append(G, G[-1] + np.linspace(1e-4, 1e-2, n_crash))
    y = np.append(y, np.full(n_crash, 48000.0))
    state = PickState()
    hint = picks.seed_gfunction(state, _res(G, y))
    assert abs(state.min_dpdg_G - 5.0) < 0.05
    assert abs(state.contact_G - 8.0) < 0.05
    assert state.closure_scenario == ""
    assert hint is None


def test_seed_spike_with_a_reversal_sample_inside_is_masked_whole():
    """At a down-then-up reversal the central difference gives a ~0 (or NaN) sample inside the
    spike; the walk back must step over it, not stop there."""
    for tail in ([9000.0, 9500.0, 120.0, -7800.0], [9000.0, 9500.0, 8700.0, np.nan, -7800.0]):
        G, y = _noisy_early_curve()
        G = np.append(G, G[-1] + 0.001 * np.arange(1, len(tail) + 1))
        y = np.append(y, tail)
        state = PickState()
        hint = picks.seed_gfunction(state, _res(G, y))
        assert abs(state.min_dpdg_G - 5.0) < 0.05, tail
        assert state.closure_scenario == "", tail
        assert hint is None


def test_terminal_spike_start_cases():
    G = np.linspace(1.0, 10.0, 10)
    assert interpret.terminal_spike_start(G, np.full(10, 5.0)) == 10
    y = np.full(10, 5.0)
    y[-3:] = [900.0, 2.0, -900.0]  # reversal sample inside the spike
    assert interpret.terminal_spike_start(G, y) == 7
    y = np.full(10, 5.0)
    y[-1] = 900.0
    assert interpret.terminal_spike_start(G, y) == 9
    assert interpret.terminal_spike_start(G, np.full(10, np.nan)) == 10
