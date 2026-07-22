"""Task 6: per-step seed-on-entry functions in picks.py (seed_overview/seed_isip/seed_gfunction/
seed_tangent/seed_loglog/seed_pp + SEEDERS), replacing the old load-time seed_defaults.

Each seeder is exercised progressively -- seed_overview -> compute_all -> seed_isip -> compute_all
-> seed_gfunction/seed_tangent -> ... -- mirroring how ui.DfitApp._seed_step calls them one step
at a time as the user actually visits each step, rather than all at once at load.
"""

from __future__ import annotations

from pathlib import Path

from dfit_tool import picks, ui
from dfit_tool.model import PickState, TangentPick, compute_all
from tests.helpers import make_testdata, overview_state


# --------------------------------------------------------------------------------------------------
# progressive happy-path seeding: each seeder sets exactly its own field(s)
# --------------------------------------------------------------------------------------------------
def test_seed_overview_sets_only_the_injection_window():
    td = make_testdata()
    st = PickState(rate_col="RATE", volume_col="VOLUME")
    assert st.start_idx is None and st.shutin_idx is None
    picks.seed_overview(st, td)
    assert st.start_idx is not None
    assert st.shutin_idx is not None
    # nothing else on the state was touched
    assert st.isip_tangent is None
    assert st.min_dpdg_G is None


def test_seed_isip_sets_only_isip_tangent():
    td = make_testdata()
    st = overview_state(td)
    picks.seed_overview(st, td)
    res = compute_all(st, td)
    assert st.isip_tangent is None
    picks.seed_isip(st, td, res)
    assert st.isip_tangent is not None
    assert isinstance(st.isip_tangent, TangentPick)
    # nothing downstream was touched
    assert st.min_dpdg_G is None
    assert st.closure_slope is None


def test_seed_gfunction_sets_min_dpdg_g_and_contact_g_only():
    td = make_testdata()
    st = overview_state(td)
    picks.seed_overview(st, td)
    res = compute_all(st, td)
    picks.seed_isip(st, td, res)
    res = compute_all(st, td)
    assert st.min_dpdg_G is None and st.contact_G is None
    picks.seed_gfunction(st, res)
    assert st.min_dpdg_G is not None
    assert st.contact_G is not None
    # tangent-step fields untouched
    assert st.closure_slope is None
    assert st.closure_G is None


def test_seed_tangent_sets_closure_slope_and_closure_g_only():
    td = make_testdata()
    st = overview_state(td)
    picks.seed_overview(st, td)
    res = compute_all(st, td)
    picks.seed_isip(st, td, res)
    res = compute_all(st, td)
    picks.seed_gfunction(st, res)
    assert st.closure_slope is None and st.closure_G is None
    picks.seed_tangent(st, res)
    assert st.closure_slope is not None
    assert st.closure_G is not None
    # loglog/pp windows untouched
    assert st.loglog_window is None
    assert st.pp_window is None


def test_seed_loglog_sets_loglog_window_only():
    td = make_testdata()
    st = overview_state(td)
    picks.seed_overview(st, td)
    res = compute_all(st, td)
    picks.seed_isip(st, td, res)
    res = compute_all(st, td)
    picks.seed_gfunction(st, res)
    picks.seed_tangent(st, res)
    assert st.loglog_window is None
    picks.seed_loglog(st, res)
    assert st.loglog_window is not None
    assert st.pp_window is None


def test_seed_pp_sets_pp_window_only():
    td = make_testdata()
    st = overview_state(td)
    picks.seed_overview(st, td)
    res = compute_all(st, td)
    picks.seed_isip(st, td, res)
    res = compute_all(st, td)
    picks.seed_gfunction(st, res)
    picks.seed_tangent(st, res)
    assert st.pp_window is None
    picks.seed_pp(st, res)
    assert st.pp_window is not None
    # loglog_window (a sibling, seeded separately) is untouched by seed_pp
    assert st.loglog_window is None


# --------------------------------------------------------------------------------------------------
# graceful no-op: missing prerequisites never raise and never set anything
# --------------------------------------------------------------------------------------------------
def test_seed_isip_no_op_when_no_injection_window_picked():
    td = make_testdata()
    st = overview_state(td)
    st.start_idx = st.shutin_idx = None  # no shut-in -> res.t_shutin_s stays None
    res = compute_all(st, td)
    assert res.t_shutin_s is None
    picks.seed_isip(st, td, res)
    assert st.isip_tangent is None


def test_seed_gfunction_no_op_when_diagnostics_missing():
    td = make_testdata()
    st = overview_state(td)
    st.start_idx = st.shutin_idx = None  # no te -> no diagnostics
    res = compute_all(st, td)
    assert res.diagnostics is None
    picks.seed_gfunction(st, res)
    assert st.min_dpdg_G is None
    assert st.contact_G is None


def test_seed_tangent_no_op_when_diagnostics_missing():
    td = make_testdata()
    st = overview_state(td)
    st.start_idx = st.shutin_idx = None
    res = compute_all(st, td)
    assert res.diagnostics is None
    picks.seed_tangent(st, res)
    assert st.closure_slope is None
    assert st.closure_G is None


def test_seed_loglog_no_op_when_diagnostics_missing():
    td = make_testdata()
    st = overview_state(td)
    st.start_idx = st.shutin_idx = None
    res = compute_all(st, td)
    assert res.diagnostics is None
    picks.seed_loglog(st, res)
    assert st.loglog_window is None


def test_seed_pp_no_op_when_diagnostics_missing():
    td = make_testdata()
    st = overview_state(td)
    st.start_idx = st.shutin_idx = None
    res = compute_all(st, td)
    assert res.diagnostics is None
    picks.seed_pp(st, res)
    assert st.pp_window is None


# --------------------------------------------------------------------------------------------------
# non-destructive: an already-set pick is left unchanged
# --------------------------------------------------------------------------------------------------
def test_seed_overview_non_destructive():
    td = make_testdata()
    st = PickState(rate_col="RATE", volume_col="VOLUME", start_idx=5, shutin_idx=9)
    picks.seed_overview(st, td)
    assert (st.start_idx, st.shutin_idx) == (5, 9)


def test_seed_isip_non_destructive():
    td = make_testdata()
    st = overview_state(td)
    res = compute_all(st, td)
    pre = TangentPick(anchor_x=1.0, anchor_y=2.0, slope=3.0)
    st.isip_tangent = pre
    picks.seed_isip(st, td, res)
    assert st.isip_tangent is pre


def test_seed_gfunction_non_destructive():
    td = make_testdata()
    st = overview_state(td)
    picks.seed_overview(st, td)
    res = compute_all(st, td)
    picks.seed_isip(st, td, res)
    res = compute_all(st, td)
    st.min_dpdg_G = 1.0
    st.contact_G = 42.0
    picks.seed_gfunction(st, res)
    assert st.min_dpdg_G == 1.0
    assert st.contact_G == 42.0


def test_seed_tangent_non_destructive():
    td = make_testdata()
    st = overview_state(td)
    picks.seed_overview(st, td)
    res = compute_all(st, td)
    picks.seed_isip(st, td, res)
    res = compute_all(st, td)
    st.closure_slope = 7.0
    st.closure_G = 8.0
    picks.seed_tangent(st, res)
    assert st.closure_slope == 7.0
    assert st.closure_G == 8.0


def test_seed_loglog_non_destructive():
    td = make_testdata()
    st = overview_state(td)
    picks.seed_overview(st, td)
    res = compute_all(st, td)
    picks.seed_isip(st, td, res)
    res = compute_all(st, td)
    pre = (11.0, 22.0)
    st.loglog_window = pre
    picks.seed_loglog(st, res)
    assert st.loglog_window == pre


def test_seed_pp_non_destructive():
    td = make_testdata()
    st = overview_state(td)
    picks.seed_overview(st, td)
    res = compute_all(st, td)
    picks.seed_isip(st, td, res)
    res = compute_all(st, td)
    pre = (11.0, 22.0)
    st.pp_window = pre
    picks.seed_pp(st, res)
    assert st.pp_window == pre


# --------------------------------------------------------------------------------------------------
# SEEDERS dict shape + seed_defaults is fully gone
# --------------------------------------------------------------------------------------------------
def test_seeders_covers_exactly_the_six_step_keys():
    assert set(picks.SEEDERS.keys()) == {k for k, _ in ui.STEPS}


def test_seed_defaults_no_longer_exists():
    assert not hasattr(picks, "seed_defaults")


def test_no_source_references_to_seed_defaults_remain():
    repo_root = Path(__file__).resolve().parent.parent
    this_file = Path(__file__).resolve()
    needle = "seed" + "_defaults"  # split so this file's own check isn't a false positive
    offenders = []
    for py_file in list((repo_root / "dfit_tool").rglob("*.py")) + list((repo_root / "tests").rglob("*.py")):
        if py_file.resolve() == this_file:
            continue
        if needle in py_file.read_text(encoding="utf-8"):
            offenders.append(str(py_file))
    assert offenders == []
