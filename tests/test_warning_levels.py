"""Warning hierarchy: blocking issues (stop navigation past Overview), warnings (anything that
changes or may invalidate a reported number), and notes (informational). Plus the Overview
issues list and the message-length budget."""
import types

from dfit_tool import ui
from dfit_tool.model import (DerivedResults, PickState, blocking_issues, compute_all,
                             step_gate_error)
from dfit_tool.ui import DfitApp
from tests.helpers import injection_state, make_testdata, pre_crash_trim_dt

SURFACE_MSG = "Surface pressure selected but density/TVD not set"


# --------------------------------------------------------------------------------------------------
# model: levels
# --------------------------------------------------------------------------------------------------
def test_surface_pressure_without_density_tvd_is_a_blocker_not_a_warning():
    td = make_testdata()
    st = injection_state(td)  # surface pressure, no density/TVD
    res = compute_all(st, td)
    assert SURFACE_MSG in res.blockers
    assert SURFACE_MSG not in res.warnings


def test_no_blocker_once_density_and_tvd_are_set():
    td = make_testdata()
    st = injection_state(td)
    st.density_ppg, st.tvd_ft = 8.4, 8000.0
    assert blocking_issues(st) == []
    assert compute_all(st, td).blockers == []


def test_no_blocker_when_channel_is_bhp():
    td = make_testdata()
    st = injection_state(td)
    st.pressure_is_bhp = True
    assert blocking_issues(st) == []


def test_no_pressure_channel_is_a_blocker():
    st = PickState()
    assert blocking_issues(st) == ["No pressure channel selected"]
    res = compute_all(st, make_testdata())
    assert res.blockers == ["No pressure channel selected"]
    assert "No pressure channel selected" not in res.warnings


def test_missing_tvd_gradient_message_is_a_note():
    td = make_testdata()
    st = injection_state(td)
    st.pressure_is_bhp = True
    seeded = compute_all(st, td)
    assert seeded.t_shutin_s is not None
    # isip_at_shutin gives an apparent ISIP, so there is a gradient source to report against.
    st.isip_at_shutin = True
    res = compute_all(st, td)
    assert res.apparent_isip is not None
    assert any("TVD" in n and "gradient" in n.lower() for n in res.notes), res.notes
    assert not any("gradient" in w.lower() for w in res.warnings), res.warnings


# --------------------------------------------------------------------------------------------------
# model: gate
# --------------------------------------------------------------------------------------------------
def test_step_gate_blocks_overview_on_blocker():
    st = injection_state(make_testdata())
    msg = step_gate_error(st, "overview")
    assert msg and "density" in msg.lower() and "tvd" in msg.lower()
    st.density_ppg, st.tvd_ft = 8.4, 8000.0
    assert step_gate_error(st, "overview") is None


# --------------------------------------------------------------------------------------------------
# ui: navigation is blocked past Overview
# --------------------------------------------------------------------------------------------------
class _Label:
    def __init__(self):
        self.text = ""

    def config(self, **kw):
        if "text" in kw:
            self.text = kw["text"]

    configure = config


def _nav_stub(step, state):
    stub = types.SimpleNamespace()
    stub.td = object()
    stub.state = state
    stub.step = step
    stub.gate_lbl = _Label()
    stub._goto_calls = []
    stub._goto = lambda dest: stub._goto_calls.append(dest)
    stub._finish = lambda: None
    stub.res = DerivedResults(blockers=blocking_issues(state))
    stub._sync_state_from_widgets = lambda: None
    stub.refresh = lambda: setattr(stub, "res",
                                   DerivedResults(blockers=blocking_issues(stub.state)))
    stub._views = {}
    stub._overview_scale_key = types.MethodType(DfitApp._overview_scale_key, stub)
    stub._overview_gate = types.MethodType(DfitApp._overview_gate, stub)
    stub._advance = types.MethodType(DfitApp._advance, stub)
    stub._next = types.MethodType(DfitApp._next, stub)
    stub._skip = types.MethodType(DfitApp._skip, stub)
    return stub


def test_next_blocked_on_overview_with_blocker():
    stub = _nav_stub("overview", injection_state(make_testdata()))
    stub._advance()
    assert stub._goto_calls == []
    assert stub.gate_lbl.text
    assert stub.state.step_status.get("overview") != "done"


def test_skip_blocked_on_overview_with_blocker():
    stub = _nav_stub("overview", injection_state(make_testdata()))
    stub._skip()
    assert stub._goto_calls == []
    assert stub.gate_lbl.text
    assert stub.state.step_status.get("overview") != "skipped"


def test_next_allowed_on_overview_without_blocker():
    st = injection_state(make_testdata())
    st.pressure_is_bhp = True
    stub = _nav_stub("overview", st)
    stub._advance()
    assert stub._goto_calls == ["injection"]


def test_next_on_overview_applies_typed_density_tvd_first():
    st = injection_state(make_testdata())
    stub = _nav_stub("overview", st)

    def sync():
        st.density_ppg, st.tvd_ft = 8.4, 8000.0  # what the user typed, not yet applied
    stub._sync_state_from_widgets = sync
    stub._advance()
    assert stub._goto_calls == ["injection"]
    assert stub.gate_lbl.text == ""


def test_next_blocked_on_overview_when_bhp_conversion_failed():
    st = injection_state(make_testdata())
    st.pressure_is_bhp = True  # blocking_issues is clean; only res reports the failure
    stub = _nav_stub("overview", st)
    stub.res = DerivedResults(blockers=["BHP computation failed: boom"])
    stub._advance()
    assert stub._goto_calls == []
    assert "BHP computation failed" in stub.gate_lbl.text


def test_warn_label_click_inert_on_overview():
    stub = types.SimpleNamespace(warn_lbl=_Label(), _warnings_expanded=False,
                                 _issue_sections=[("Warnings", ["W"])], _warn_lbl_hidden=True)
    stub._toggle_warnings = types.MethodType(DfitApp._toggle_warnings, stub)
    stub._toggle_warnings()
    assert stub._warnings_expanded is False
    assert stub.warn_lbl.text == ""


def test_goto_redirects_to_overview_when_blocked():
    st = injection_state(make_testdata())
    st.step_status = {"overview": "done", "injection": "visited"}
    stub = types.SimpleNamespace(td=object(), state=st, step="overview", gate_lbl=_Label())
    stub._seed_step = lambda key: None
    stub.refresh = lambda: None
    stub._goto = types.MethodType(DfitApp._goto, stub)
    stub._goto("injection")
    assert stub.step == "overview"
    assert "density" in stub.gate_lbl.text.lower()
    assert st.step_status["injection"] == "visited"  # untouched, not re-seeded


# --------------------------------------------------------------------------------------------------
# ui: issue text
# --------------------------------------------------------------------------------------------------
def test_issue_sections_order_and_empty_levels_dropped():
    res = DerivedResults(blockers=["B"], warnings=["W1", "W2"], notes=[])
    assert ui.issue_sections(res) == [("Blocking", ["B"]), ("Warnings", ["W1", "W2"])]
    assert ui.issue_sections(DerivedResults()) == []


def test_collapsed_summary_counts_each_level():
    res = DerivedResults(warnings=["W1", "W2"], notes=["N"])
    assert ui.format_warnings_text(ui.issue_sections(res), expanded=False) == (
        "2 warnings, 1 note (click to expand)")


# --------------------------------------------------------------------------------------------------
# message length budget
# --------------------------------------------------------------------------------------------------
MAX_LEN = 90


def _all_messages(res):
    return res.blockers + res.warnings + res.notes


def test_messages_fit_the_length_budget():
    td = make_testdata(zero_crash_at=0.5)
    st = injection_state(td)
    res = compute_all(st, td)
    st.tail_trim_dt = pre_crash_trim_dt(res)
    st.tail_trim_reason = "low_pressure"
    res = compute_all(st, td)
    msgs = _all_messages(res)
    assert msgs
    long = [m for m in msgs if len(m) > MAX_LEN]
    assert not long, long
