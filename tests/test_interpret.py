"""Unit tests for interpret.py's pure suggest_* auto-suggestion helpers not already covered via
the seeding/render integration tests (tests/test_seed_steps.py, tests/test_view_state.py)."""

import numpy as np
import pytest

from dfit_tool import interpret


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
