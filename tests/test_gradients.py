"""Pressure gradients (psi/ft): compute_all's _resolve_gradients depth-normalizes eight reported
pressures -- apparent ISIP, the three compliance-family Shmins (compliance/variable/tangent),
Shmin Liberty, Shmin rapid, Shmin stiffness, and pore pressure -- each strictly its own source
value / state.tvd_ft, with no cross-field fallback. Gated on tvd_ft being not None, finite, and
> 0; every gradient stays None otherwise. Mirrors the pattern in test_shmin_liberty.py.
"""

from __future__ import annotations

import math

import pytest

from dfit_tool import interpret, picks, store
from dfit_tool.model import compute_all
from tests.helpers import make_testdata, injection_state


def _picked_gs(dg):
    """Two distinct G-times, well clear of the array edges -- same helper as
    test_shmin_liberty.py's, used to anchor the min-dP/dG and contact picks directly rather than
    relying on C-A/C-B auto-detection finding a shape in this suite's smooth synthetic decline."""
    return float(dg.G[dg.G.size // 4]), float(dg.G[3 * dg.G.size // 4])


def _full_state(closure_scenario: str = "C-A clear", tvd_ft=10000.0):
    """A state with apparent ISIP (isip step), contact + min-dP/dG (gfunction step), closure
    (tangent step), and a pore-pressure window all set, so every gradient source value that can
    be non-None for the given closure scenario actually is. tvd_ft is set explicitly --
    injection_state leaves it None."""
    td = make_testdata()
    st = injection_state(td)
    picks.seed_injection(st, td)
    res = compute_all(st, td)
    picks.seed_isip(st, td, res)
    st.tvd_ft = tvd_ft
    st.closure_scenario = closure_scenario
    res = compute_all(st, td)
    assert res.apparent_isip is not None

    dg = res.diagnostics
    min_G, contact_G = _picked_gs(dg)
    picks.commit_min_dpdg_point(st, min_G)
    # C-C/C-D never carry a contact pick (apply_closure_scenario clears it) -- committing one
    # here would fake a compliance Shmin/eff-ISIP the real scenario never produces.
    if not closure_scenario.startswith(("C-C", "C-D")):
        picks.commit_contact_point(st, contact_G)
    closure_G = float(dg.G[dg.G.size // 2])
    picks.commit_closure_point(st, closure_G)
    res = compute_all(st, td)

    picks.seed_pp(st, res)
    st.postclosure_scenario = "PC-A linear"
    st.pp_axis = picks.suggest_pp_axis(st.postclosure_scenario) or st.pp_axis
    res = compute_all(st, td)
    return td, st, res


# --------------------------------------------------------------------------------------------------
# Each gradient is strictly its own source value / TVD.
# --------------------------------------------------------------------------------------------------
def test_each_gradient_equals_its_own_source_over_tvd():
    td, st, res = _full_state("C-A clear")
    tvd = st.tvd_ft

    assert res.apparent_isip is not None
    assert res.apparent_isip_gradient == pytest.approx(res.apparent_isip / tvd)
    assert res.shmin_compliance is not None
    assert res.shmin_compliance_gradient == pytest.approx(res.shmin_compliance / tvd)
    assert res.shmin_variable is not None
    assert res.shmin_variable_gradient == pytest.approx(res.shmin_variable / tvd)
    assert res.shmin_tangent is not None
    assert res.shmin_tangent_gradient == pytest.approx(res.shmin_tangent / tvd)
    assert res.shmin_liberty is not None
    assert res.shmin_liberty_gradient == pytest.approx(res.shmin_liberty / tvd)
    assert res.pore_pressure is not None
    assert res.pore_pressure_gradient == pytest.approx(res.pore_pressure / tvd)
    # C-A never sets shmin_rapid (that's C-D only), so its gradient stays None too.
    assert res.shmin_rapid is None
    assert res.shmin_rapid_gradient is None


def test_pressure_gradient_matches_hand_rolled_division():
    assert interpret.pressure_gradient(9000.0, 10000.0) == pytest.approx(0.9)


# --------------------------------------------------------------------------------------------------
# The tvd_ft > 0 guard: None, 0.0, negative, and non-finite all blank every gradient.
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("bad_tvd, expected_warning", [
    (None, "TVD not set"),
    (0.0, "is not a positive number"),
    (-5000.0, "is not a positive number"),
    (math.nan, "is not a positive number"),
    (math.inf, "is not a positive number"),
    ("abc", "is not a positive number"),
    # A hand-edited save carrying an integer too large for a double: json.load yields a Python
    # int and float() raises OverflowError on it. Guards the except tuple in _resolve_gradients.
    (10 ** 400, "is not a positive number"),
])
def test_all_gradients_none_when_tvd_invalid(bad_tvd, expected_warning):
    td, st, res = _full_state("C-A clear", tvd_ft=10000.0)
    st.tvd_ft = bad_tvd
    res2 = compute_all(st, td)

    assert res2.apparent_isip_gradient is None
    assert res2.shmin_compliance_gradient is None
    assert res2.shmin_variable_gradient is None
    assert res2.shmin_tangent_gradient is None
    assert res2.shmin_liberty_gradient is None
    # C-A never sets shmin_rapid (that's C-D only), so this is tautologically None regardless of
    # the guard -- assert the source too so the assertion below reads honestly.
    assert res2.shmin_rapid is None
    assert res2.shmin_rapid_gradient is None
    assert res2.pore_pressure_gradient is None
    # At least one source value (apparent_isip, etc.) is set in this fixture, so the "gradients
    # not reported" warning must fire -- never silent on an unusable TVD once there's something
    # to normalize. Assert the message-specific text, not just the word "gradient": the two
    # messages describe different analyst-facing problems ("not set" for a missing TVD vs a
    # value-naming "not a positive number" for anything else), and only this catches the two
    # appends being swapped.
    assert any("gradient" in w.lower() for w in res2.warnings), res2.warnings
    assert any(expected_warning in w for w in res2.warnings), res2.warnings
    if expected_warning == "is not a positive number":
        # The message names the offending value, so the analyst can see what got read.
        assert any(repr(bad_tvd) in w for w in res2.warnings), res2.warnings


# --------------------------------------------------------------------------------------------------
# The warning above is gated on at least one of the eight source values existing -- a freshly-
# opened test with no picks yet must stay quiet, or the warning would be permanent noise from the
# moment a file loads.
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("bad_tvd", [None, 0.0, -5000.0, math.nan, math.inf, "abc", 10 ** 400])
def test_no_gradient_warning_when_no_source_values_exist(bad_tvd):
    td = make_testdata()
    st = injection_state(td)  # channel mapping only -- no isip/gfunction/tangent/pp picks
    st.tvd_ft = bad_tvd
    res = compute_all(st, td)

    # Precondition: every gradient source is really None, so this isn't a vacuous pass.
    assert res.apparent_isip is None
    assert res.shmin_compliance is None
    assert res.shmin_variable is None
    assert res.shmin_tangent is None
    assert res.shmin_liberty is None
    assert res.shmin_rapid is None
    assert res.pore_pressure is None

    assert not any("gradient" in w.lower() for w in res.warnings), res.warnings


def test_no_gradient_warning_when_tvd_valid():
    td, st, res = _full_state("C-A clear", tvd_ft=10000.0)
    assert not any("gradient" in w.lower() for w in res.warnings), res.warnings


# --------------------------------------------------------------------------------------------------
# The gradient warning deliberately coexists with "Surface pressure selected but density/TVD not
# set" -- the two say different things (BHP reliability vs. gradients not reported) and the panel
# stacks warnings one per line. A future "fix" collapsing them into one must fail this test.
# --------------------------------------------------------------------------------------------------
def test_gradient_warning_coexists_with_density_tvd_warning():
    td, st, res = _full_state("C-A clear", tvd_ft=None)
    assert res.apparent_isip is not None  # a source value exists

    assert "Surface pressure selected but density/TVD not set" in res.warnings
    assert any("gradient" in w.lower() for w in res.warnings), res.warnings


@pytest.mark.parametrize("bad_tvd", [None, 0.0, -5000.0, math.nan, math.inf])
def test_shmin_rapid_gradient_none_when_tvd_invalid(bad_tvd):
    # Unlike the C-A case above, C-D rapid DOES set shmin_rapid, so this is the real regression
    # guard for the row the Fix 1 panel asterisk depends on: the guard must blank
    # shmin_rapid_gradient even though its source value exists.
    td, st, res = _full_state("C-D rapid", tvd_ft=10000.0)
    assert res.shmin_rapid is not None
    st.tvd_ft = bad_tvd
    res2 = compute_all(st, td)
    assert res2.shmin_rapid_gradient is None


# --------------------------------------------------------------------------------------------------
# No cross-field fallback: C-D sets shmin_rapid_gradient but not shmin_compliance_gradient, and
# vice versa for C-A. This is the regression guard for the model layer's per-field-only rule.
# --------------------------------------------------------------------------------------------------
def test_cd_rapid_sets_rapid_gradient_not_compliance_gradient():
    td, st, res = _full_state("C-D rapid")
    assert st.contact_G is None  # C-D clears the contact pick
    assert res.shmin_compliance is None
    assert res.shmin_compliance_gradient is None
    assert res.shmin_rapid is not None
    assert res.shmin_rapid_gradient == pytest.approx(res.shmin_rapid / st.tvd_ft)


def test_ca_sets_compliance_gradient_not_rapid_gradient():
    td, st, res = _full_state("C-A clear")
    assert res.shmin_compliance is not None
    assert res.shmin_compliance_gradient == pytest.approx(res.shmin_compliance / st.tvd_ft)
    assert res.shmin_rapid is None
    assert res.shmin_rapid_gradient is None


# --------------------------------------------------------------------------------------------------
# PC-F: pore_pressure stays None, so pore_pressure_gradient falls out None for free.
# --------------------------------------------------------------------------------------------------
def test_pore_pressure_gradient_none_under_pcf():
    td, st, res = _full_state("C-A clear")
    st.postclosure_scenario = "PC-F no peak"
    res2 = compute_all(st, td)
    assert res2.pore_pressure is None
    assert res2.pore_pressure_gradient is None


# --------------------------------------------------------------------------------------------------
# store.py: LOG_COLUMNS membership/ordering, build_log_row passthrough (no arithmetic there).
# --------------------------------------------------------------------------------------------------
def test_log_columns_gradients_appended_after_units_note_in_order():
    expected = [
        "apparent_ISIP_gradient",
        "Shmin_compliance_gradient",
        "Shmin_variable_gradient",
        "Shmin_tangent_gradient",
        "Shmin_liberty_gradient",
        "Shmin_rapid_gradient",
        "pore_pressure_gradient",
    ]
    units_note_idx = store.LOG_COLUMNS.index("units_note")
    for offset, col in enumerate(expected, start=1):
        assert col in store.LOG_COLUMNS
        assert store.LOG_COLUMNS.index(col) == units_note_idx + offset


def test_build_log_row_passes_gradients_through(tmp_path):
    td = make_testdata()
    state = injection_state(td)
    res = compute_all(state, td)
    res.apparent_isip_gradient = 0.61
    res.shmin_compliance_gradient = 0.58
    res.shmin_variable_gradient = 0.585
    res.shmin_tangent_gradient = 0.57
    res.shmin_liberty_gradient = 0.55
    res.shmin_rapid_gradient = 0.59
    res.pore_pressure_gradient = 0.45
    entry = store.TestEntry(test_id="well1", folder=str(tmp_path))
    active_path = str(tmp_path / "well1.csv")

    row = store.build_log_row(entry, active_path, str(tmp_path), state, td, res)

    assert row["apparent_ISIP_gradient"] == 0.61
    assert row["Shmin_compliance_gradient"] == 0.58
    assert row["Shmin_variable_gradient"] == 0.585
    assert row["Shmin_tangent_gradient"] == 0.57
    assert row["Shmin_liberty_gradient"] == 0.55
    assert row["Shmin_rapid_gradient"] == 0.59
    assert row["pore_pressure_gradient"] == 0.45


def test_build_log_row_does_not_replicate_panel_rapid_fallback(tmp_path):
    # build_log_row computes nothing -- Shmin_compliance_gradient stays None (empty in the CSV)
    # under C-D exactly as Shmin_compliance does, even though the panel substitutes the rapid
    # value/asterisk for display purposes.
    td, st, res = _full_state("C-D rapid")
    entry = store.TestEntry(test_id="well1", folder=str(tmp_path))
    active_path = str(tmp_path / "well1.csv")

    row = store.build_log_row(entry, active_path, str(tmp_path), st, td, res)

    assert row["Shmin_compliance"] is None
    assert row["Shmin_compliance_gradient"] is None
    assert row["Shmin_rapid_gradient"] is not None
