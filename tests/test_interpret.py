"""Unit tests for interpret.py's pure suggest_* auto-suggestion helpers not already covered via
the seeding/render integration tests (tests/test_seed_steps.py, tests/test_view_state.py)."""

import numpy as np

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
