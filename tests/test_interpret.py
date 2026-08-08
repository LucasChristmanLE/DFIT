"""Unit tests for interpret.py's pure suggest_* auto-suggestion helpers not already covered via
the seeding/render integration tests (tests/test_seed_steps.py, tests/test_view_state.py)."""

import numpy as np
import pytest

from dfit_tool import interpret, picks
from dfit_tool.model import compute_all
from tests.helpers import injection_state, make_testdata


def test_suggest_min_dpdg_index_prefers_interior_relative_min_over_smaller_endpoint():
    G = np.linspace(1.0, 10.0, 20)
    dPdG = np.linspace(10.0, 1.0, 20)
    dPdG[10] = 3.0  # a genuine interior local min (dips below both neighbors), even though the
                    # monotonic tail's endpoint (1.0) is a smaller value overall
    idx = interpret.suggest_min_dpdg_index(G, dPdG)
    assert idx == 10


def test_suggest_min_dpdg_index_falls_back_to_masked_argmin_on_monotonic_data():
    G = np.linspace(1.0, 10.0, 20)
    dPdG = np.linspace(10.0, 1.0, 20)  # no interior local min anywhere (C-C no-contact shape)
    idx = interpret.suggest_min_dpdg_index(G, dPdG)
    assert idx == len(dPdG) - 1


def test_shmin_rapid_is_apparent_isip_minus_175():
    assert interpret.shmin_rapid(9500.0) == 9500.0 - 175.0
    assert interpret.shmin_rapid(9500.0) == interpret.shmin_rapid(9500.0, offset=175.0)


def test_format_shmin_rapid_short_form_fits_the_panel_column():
    s = interpret.format_shmin_rapid(9325.0)
    assert s == "9325 ±75"
    assert len(s) <= 14  # the result panel's value labels are ttk.Label(width=14, anchor="e")


def test_format_shmin_rapid_verbose_form_annotates_range():
    s = interpret.format_shmin_rapid(9325.0, verbose=True)
    assert s == "9325 ±75 (ISIP − 100–250)"


def test_tangent_from_index_true_tangent_on_quadratic_decline():
    """A noiseless quadratic decline: the analytic tangent at idx has a known slope and value.
    The local-line-fit tangent (fit centered on x[idx], anchor_y = fit value AT x[idx]) must
    reproduce both -- a raw-sample anchor is only correct when the curve happens to be linear."""
    x = np.linspace(0.0, 20.0, 401)
    a, b, c = -0.5, -3.0, 100.0
    y = a * x**2 + b * x + c
    idx = 300
    analytic_slope = 2 * a * x[idx] + b
    analytic_y = a * x[idx]**2 + b * x[idx] + c

    ax_, ay_, slope = interpret.tangent_from_index(x, y, idx, half=5)
    assert ax_ == pytest.approx(x[idx])
    assert slope == pytest.approx(analytic_slope, abs=0.05)
    assert ay_ == pytest.approx(analytic_y, abs=0.05)


def test_tangent_from_index_anchor_y_rejects_single_sample_noise():
    """Corrupting only the anchor sample must not corrupt the anchor y: the fitted value sits
    near the clean trend, not the corrupted raw sample."""
    x = np.linspace(0.0, 20.0, 41)
    y = 100.0 - 3.0 * x
    idx = 20
    y_noisy = y.copy()
    y_noisy[idx] += 500.0

    ax_, ay_, slope = interpret.tangent_from_index(x, y_noisy, idx, half=5)
    assert ax_ == pytest.approx(x[idx])
    assert ay_ != pytest.approx(y_noisy[idx])
    assert ay_ == pytest.approx(y[idx], abs=50.0)
    assert slope == pytest.approx(-3.0, abs=5.0)


def test_tangent_from_index_degenerate_window_falls_back_to_raw_sample():
    x = np.array([0.0, 1.0])
    y = np.array([10.0, 20.0])
    ax_, ay_, slope = interpret.tangent_from_index(x, y, 0, half=0)  # window [0:1] -> <2 points
    assert slope == 0.0
    assert ay_ == pytest.approx(y[0])
    assert ax_ == pytest.approx(x[0])


def test_local_slope_and_tangent_from_index_share_the_same_fit():
    """No duplicated window logic: both must produce the identical slope for the same window."""
    x = np.linspace(0.0, 10.0, 21)
    y = np.sin(x) * 5.0 + 2.0 * x
    idx = 10
    slope_only = interpret.local_slope(x, y, idx, half=4)
    _, _, slope_tangent = interpret.tangent_from_index(x, y, idx, half=4)
    assert slope_only == pytest.approx(slope_tangent)


# --------------------------------------------------------------------------------------------------
# suggest_hump_index / suggest_min_dpdg_index (rewrite): a synthetic curve shaped like a real C-A
# closure signature -- early water-hammer spike, a true elbow (relative min) at G=0.55 (below the
# old g_min=1.0 mask on purpose), a rise to a hump at G=3.5, then a decaying tail.
# --------------------------------------------------------------------------------------------------
def _spike_elbow_hump_curve(n=400, g_max=8.0, elbow_G=0.55, hump_G=3.5, min_val=2.0,
                            hump_val=6.0, tail_val=1.0, spike_amp=50.0, spike_decay=0.05,
                            dip=None, noise_std=0.0, seed=0):
    """dPdG(G) = an early spike (decays to ~0 well before G=0.3) + a base shape that dips to a
    relative min at ``elbow_G``, rises to a hump at ``hump_G``, then decays through a tail.
    ``dip=(G, value)`` optionally overwrites one sample -- e.g. a deep dip planted past the hump,
    to check the hump-based candidate narrowing rejects it in favor of the real elbow.
    ``noise_std`` adds Gaussian noise (seeded) for a robustness check."""
    G = np.linspace(0.001, g_max, n)
    spike = spike_amp * np.exp(-G / spike_decay)
    left = min_val + 20.0 * (elbow_G - G) ** 2
    right_rise = min_val + (hump_val - min_val) * (G - elbow_G) / (hump_G - elbow_G)
    right_tail = tail_val + (hump_val - tail_val) * np.exp(-(G - hump_G) / 1.5)
    base = np.where(G < elbow_G, left, np.where(G <= hump_G, right_rise, right_tail))
    dPdG = spike + base
    if dip is not None:
        dip_G, dip_val = dip
        idx = int(np.argmin(np.abs(G - dip_G)))
        dPdG[idx] = dip_val
    if noise_std:
        rng = np.random.default_rng(seed)
        dPdG = dPdG + rng.normal(0.0, noise_std, size=dPdG.shape)
    return G, dPdG


def test_suggest_hump_index_ignores_the_early_spike():
    G, dPdG = _spike_elbow_hump_curve()
    idx = interpret.suggest_hump_index(G, dPdG)
    assert idx is not None
    assert abs(G[idx] - 3.5) < 0.3


def test_suggest_hump_index_none_when_all_non_finite():
    G = np.linspace(0.0, 10.0, 20)
    dPdG = np.full_like(G, np.nan)
    assert interpret.suggest_hump_index(G, dPdG) is None


def test_suggest_hump_index_returns_the_raw_peak_not_a_slow_decaying_tail_sample():
    """Regression: on tests/helpers.make_testdata's diagnostics, the raw peak of dP/dG is a
    genuine interior local max, but the tail past it decays slower than 1/G -- so a plain
    argmax(G*dPdG) lands 4 samples from the end of the array instead of on the real peak. The
    peak (an interior local max) must win over the tail (which has no interior local max)."""
    td = make_testdata()
    st = injection_state(td)
    picks.seed_injection(st, td)
    res = compute_all(st, td)
    picks.seed_isip(st, td, res)
    res = compute_all(st, td)
    dg = res.diagnostics
    idx = interpret.suggest_hump_index(dg.G, dg.dPdG)
    assert idx == int(np.nanargmax(dg.dPdG))


def test_suggest_min_dpdg_index_finds_elbow_below_the_old_g_min_mask():
    G, dPdG = _spike_elbow_hump_curve()
    idx = interpret.suggest_min_dpdg_index(G, dPdG)
    assert abs(G[idx] - 0.55) < 0.1


def test_suggest_min_dpdg_index_ignores_a_planted_dip_past_the_hump():
    G, dPdG = _spike_elbow_hump_curve(dip=(6.0, 0.1))  # deeper than the real elbow, but past hump
    idx = interpret.suggest_min_dpdg_index(G, dPdG)
    assert abs(G[idx] - 0.55) < 0.1


def test_suggest_min_dpdg_index_finds_elbow_with_noise():
    for seed in range(5):
        G, dPdG = _spike_elbow_hump_curve(noise_std=0.05, seed=seed)
        idx = interpret.suggest_min_dpdg_index(G, dPdG)
        assert abs(G[idx] - 0.55) < 0.15


def test_suggest_min_dpdg_index_all_non_finite_returns_zero():
    G = np.linspace(0.0, 10.0, 20)
    dPdG = np.full_like(G, np.nan)
    assert interpret.suggest_min_dpdg_index(G, dPdG) == 0


# --------------------------------------------------------------------------------------------------
# min_index_in_window / suggest_contact_inflection_index's g_range -- the Shift+drag window
# correction's underlying finders.
# --------------------------------------------------------------------------------------------------
def test_min_index_in_window_basic():
    G = np.linspace(0.0, 10.0, 21)
    dPdG = np.abs(G - 5.0)  # min at G=5
    idx = interpret.min_index_in_window(G, dPdG, 3.0, 7.0)
    assert idx is not None
    assert G[idx] == pytest.approx(5.0)


def test_min_index_in_window_empty_window_returns_none():
    G = np.linspace(0.0, 10.0, 21)
    dPdG = np.abs(G - 5.0)
    assert interpret.min_index_in_window(G, dPdG, 20.0, 30.0) is None


def test_min_index_in_window_skips_non_finite_samples():
    G = np.linspace(0.0, 10.0, 21)
    dPdG = np.abs(G - 5.0)
    dPdG[10] = np.nan  # the true min sample (G=5.0) is NaN'd out
    idx = interpret.min_index_in_window(G, dPdG, 3.0, 7.0)
    assert idx is not None
    assert G[idx] != pytest.approx(5.0)
    assert np.isfinite(dPdG[idx])


def test_suggest_contact_inflection_index_g_range_overrides_g_min():
    """A flattening before g_min=1.0 is normally masked out (see
    test_inflection_rule_respects_g_min in test_scenario_contact.py) -- an explicit g_range that
    includes it finds it anyway."""
    G = np.linspace(0.0, 12.0, 241)
    dPdG = 300.0 - (G + (G - 0.5) ** 3 / 3.0)  # flattening at G=0.5
    assert interpret.suggest_contact_inflection_index(G, dPdG, g_min=1.0) is None
    idx = interpret.suggest_contact_inflection_index(G, dPdG, g_range=(0.0, 1.0))
    assert idx is not None
    assert abs(G[idx] - 0.5) < 0.2


def test_suggest_contact_inflection_index_g_range_empty_window_returns_none():
    G = np.linspace(0.0, 12.0, 241)
    dPdG = 300.0 - (G + (G - 6.0) ** 3 / 3.0)  # inflection at G=6, well outside the window below
    assert interpret.suggest_contact_inflection_index(G, dPdG, g_range=(20.0, 30.0)) is None
