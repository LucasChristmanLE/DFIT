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
