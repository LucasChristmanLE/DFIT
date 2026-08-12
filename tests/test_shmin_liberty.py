"""Shmin Liberty: Liberty's internal variant of the compliance method. compute_all sets
DerivedResults.shmin_liberty = BHP interpolated at the method's anchor minus 200 psi. The
anchor is state.min_dpdg_G for a blank scenario and C-A, but state.contact_G (the inflection)
for C-B -- picks.apply_closure_scenario/re_derive_contact_from_min only move contact_G to the
inflection, leaving min_dpdg_G wherever it was seeded/dragged, so the two picks are NOT
generally the same point under C-B (only the Shift+drag window path makes them equal). The
whole block is gated on state.contact_G, so it blanks in every state that blanks the compliance
row -- including a C-A/C-B whose contact construction failed. Display + log only: it never
feeds net pressure, delta closure, or the shared reference ISIP. Mirrors the pattern in
test_rapid_closure.py / test_variable_compliance.py."""

from __future__ import annotations

import numpy as np
import pytest

from dfit_tool import interpret, picks
from dfit_tool.model import compute_all
from tests.helpers import make_testdata, injection_state


def _res():
    td = make_testdata()
    st = injection_state(td)
    picks.seed_injection(st, td)
    return td, st, compute_all(st, td)


def _picked_gs(dg):
    """Two distinct G-times, well clear of the array edges -- same helper as
    test_variable_compliance.py's, used here to anchor the min-dP/dG and contact picks without
    relying on the C-A/C-B auto-detection rules finding a shape. This suite's synthetic pressure
    is a smooth exponential decline, which never produces an interior dP/dG local min (C-A) or
    a genuine flattening/inflection (C-B) -- confirmed empirically: apply_closure_scenario's
    rules never fire on it (see test_cb_real_apply_closure_scenario_leaves_contact_none_below).
    Direct commits are the only way to test the anchor-selection logic itself in isolation."""
    return float(dg.G[dg.G.size // 4]), float(dg.G[3 * dg.G.size // 4])


# --------------------------------------------------------------------------------------------------
# C-A: anchors on min_dpdg_G
# --------------------------------------------------------------------------------------------------
def test_ca_matches_hand_rolled_interp_and_differs_from_compliance():
    """Anchor strictly between two diagnostics grid points, checked against a hand-rolled linear
    interpolation of the two neighboring (G, p) samples -- catches both an array mix-up (dg.p vs
    res.resampled.p) and an interp-argument-order bug, not just a formula-level regression."""
    td, st, res = _res()
    dg = res.diagnostics
    i = dg.G.size // 2
    G_lo, G_hi = float(dg.G[i]), float(dg.G[i + 1])
    p_lo, p_hi = float(res.resampled.p[i]), float(res.resampled.p[i + 1])
    anchor_G = (G_lo + G_hi) / 2.0
    frac = (anchor_G - G_lo) / (G_hi - G_lo)
    expected_p = p_lo + frac * (p_hi - p_lo)

    st.closure_scenario = "C-A clear"
    picks.commit_min_dpdg_point(st, anchor_G)
    picks.commit_contact_point(st, G_hi)  # gate only -- C-A anchors on min_dpdg_G, not this
    res2 = compute_all(st, td)

    assert res2.shmin_liberty == pytest.approx(expected_p - 200.0)
    assert res2.shmin_liberty == pytest.approx(interpret.shmin_liberty(expected_p))
    # Different anchor point (min-dP/dG, not the contact) and a different offset (200 vs
    # 75 psi), so the two methods disagree.
    assert res2.shmin_compliance is not None
    assert res2.shmin_liberty != pytest.approx(res2.shmin_compliance)


def test_ca_blanks_when_contact_none_even_with_min_pick_set():
    """Regression for blocker 2: a min-dP/dG pick alone is not enough -- the block is gated on
    contact_G, since that's what fails to exist when the scenario's contact construction (the
    +10% rule, here never satisfied since nothing sets contact_G at all) hasn't happened."""
    td, st, res = _res()
    dg = res.diagnostics
    min_G, _ = _picked_gs(dg)
    st.closure_scenario = "C-A clear"
    picks.commit_min_dpdg_point(st, min_G)
    assert st.contact_G is None
    res2 = compute_all(st, td)
    assert res2.shmin_liberty is None


# --------------------------------------------------------------------------------------------------
# C-B: anchors on contact_G (the inflection), not min_dpdg_G
# --------------------------------------------------------------------------------------------------
def test_cb_anchors_on_contact_not_min_when_they_differ():
    """The blocker-1 regression guard: min_dpdg_G and contact_G are set to two distinct picks
    (as they can genuinely be under C-B -- see the module docstring), and Liberty must track
    contact_G, not min_dpdg_G."""
    td, st, res = _res()
    dg = res.diagnostics
    min_G, contact_G = _picked_gs(dg)
    assert min_G != contact_G
    st.closure_scenario = "C-B adequate"
    picks.commit_min_dpdg_point(st, min_G)
    picks.commit_contact_point(st, contact_G)
    res2 = compute_all(st, td)

    expected = float(np.interp(contact_G, dg.G, res.resampled.p)) - 200.0
    assert res2.shmin_liberty == pytest.approx(expected)
    min_anchored = float(np.interp(min_G, dg.G, res.resampled.p)) - 200.0
    assert res2.shmin_liberty != pytest.approx(min_anchored)
    # Also matches the panel's own contact-pressure row, since both read contact_G now.
    assert res2.shmin_liberty == pytest.approx(res2.contact_pressure - 200.0)


def test_cb_real_apply_closure_scenario_leaves_contact_none_below():
    """Regression for blocker 2 through the real scenario-application path (not a hand-built
    state): this suite's synthetic decline has no dP/dG inflection, so
    picks.apply_closure_scenario's C-B rule genuinely fails (a hint is returned, contact_G is
    left None) -- and Liberty must blank there too, in step with shmin_compliance."""
    td, st, res = _res()
    st.closure_scenario = "C-B adequate"
    hint = picks.apply_closure_scenario(st, res)
    assert hint is not None
    assert st.contact_G is None
    res2 = compute_all(st, td)
    assert res2.shmin_compliance is None
    assert res2.shmin_liberty is None


def test_cb_blanks_when_contact_none_even_with_min_pick_set():
    """Same blocker-2 regression as the C-A case above, for C-B."""
    td, st, res = _res()
    dg = res.diagnostics
    min_G, _ = _picked_gs(dg)
    st.closure_scenario = "C-B adequate"
    picks.commit_min_dpdg_point(st, min_G)
    assert st.contact_G is None
    res2 = compute_all(st, td)
    assert res2.shmin_liberty is None


# --------------------------------------------------------------------------------------------------
# blank scenario: anchors on min_dpdg_G, same as C-A
# --------------------------------------------------------------------------------------------------
def test_blank_scenario_computes_off_min_dpdg_anchor():
    """Blank scenario mirrors compliance Shmin's behavior (both show under "", both cleared by
    C-C/C-D) and anchors like C-A. min_dpdg_G and contact_G are set to distinct picks (a blank
    scenario seeds a contact placeholder in the app, same as picks.seed_gfunction would), and
    Liberty must track the min pick, not the placeholder contact."""
    td, st, res = _res()
    dg = res.diagnostics
    min_G, contact_G = _picked_gs(dg)
    picks.commit_min_dpdg_point(st, min_G)
    picks.commit_contact_point(st, contact_G)  # closure_scenario stays "" (blank)
    res2 = compute_all(st, td)

    expected = float(np.interp(min_G, dg.G, res.resampled.p)) - 200.0
    assert res2.shmin_liberty == pytest.approx(expected)


# --------------------------------------------------------------------------------------------------
# C-C / C-D: never computed, regardless of what picks exist
# --------------------------------------------------------------------------------------------------
def test_cc_and_cd_report_none_even_with_min_and_contact_pick_set():
    """The model.compute_all gate on closure_scenario, not just on the picks being set: seed
    both picks first, then switch to C-C/C-D without clearing them, and confirm the Liberty
    value still goes None."""
    for scen in ("C-C no-contact", "C-D rapid"):
        td, st, res = _res()
        dg = res.diagnostics
        min_G, contact_G = _picked_gs(dg)
        picks.commit_min_dpdg_point(st, min_G)
        picks.commit_contact_point(st, contact_G)
        st.closure_scenario = scen
        res2 = compute_all(st, td)
        assert res2.shmin_liberty is None
