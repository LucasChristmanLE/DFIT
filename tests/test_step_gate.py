import types

from dfit_tool import ui
from dfit_tool.model import PickState, step_gate_error
from dfit_tool.ui import DfitApp


def test_gfunction_gate_blocks_without_closure_scenario():
    st = PickState()
    assert st.closure_scenario == ""
    assert step_gate_error(st, "gfunction") is not None
    st.closure_scenario = "C-A clear"
    assert step_gate_error(st, "gfunction") is None


def test_loglog_gate_blocks_without_postclosure_scenario():
    st = PickState()
    assert st.postclosure_scenario == ""
    assert step_gate_error(st, "loglog") is not None
    st.postclosure_scenario = "PC-A linear"
    assert step_gate_error(st, "loglog") is None


def test_other_steps_never_gated():
    st = PickState()
    for step in ("overview", "isip", "tangent", "porepressure"):
        assert step_gate_error(st, step) is None


# --------------------------------------------------------------------------------------------------
# Behavioral: exercise DfitApp._advance / _next / _skip through a duck-typed stand-in, same
# pattern as test_view_state.py's _refresh_stub. These guard against the gate being bypassed
# (e.g. someone dropping the early ``return`` in _advance) or over-applied to Skip/last-step.
# --------------------------------------------------------------------------------------------------
class _GateLabel:
    """Records the last text passed to config(text=...)."""

    def __init__(self):
        self.texts = []

    def config(self, **kw):
        if "text" in kw:
            self.texts.append(kw["text"])

    @property
    def last_text(self):
        return self.texts[-1] if self.texts else None


def _nav_stub(step):
    """Duck-typed DfitApp stand-in exposing only what _advance/_next/_skip touch."""
    stub = types.SimpleNamespace()
    stub.td = object()  # truthy sentinel -- these methods only check `is None`
    stub.state = PickState()
    stub.step = step
    stub.gate_lbl = _GateLabel()
    stub._goto_calls = []
    stub._goto = lambda dest: stub._goto_calls.append(dest)
    stub._finish_calls = []
    stub._finish = lambda: stub._finish_calls.append(True)
    stub._advance = types.MethodType(DfitApp._advance, stub)
    stub._next = types.MethodType(DfitApp._next, stub)
    stub._skip = types.MethodType(DfitApp._skip, stub)
    return stub


def test_advance_blocked_on_gfunction_without_closure_scenario():
    stub = _nav_stub("gfunction")
    assert stub.state.closure_scenario == ""

    stub._advance()

    assert stub._goto_calls == []
    assert stub.gate_lbl.last_text
    assert "closure" in stub.gate_lbl.last_text.lower() or "scenario" in stub.gate_lbl.last_text.lower()


def test_advance_allowed_on_gfunction_with_closure_scenario():
    stub = _nav_stub("gfunction")
    stub.state.closure_scenario = "C-A clear"

    stub._advance()

    assert stub._goto_calls == [ui.next_step("gfunction")]
    assert not stub.gate_lbl.last_text


def test_advance_blocked_on_loglog_without_postclosure_scenario():
    stub = _nav_stub("loglog")
    assert stub.state.postclosure_scenario == ""

    stub._advance()

    assert stub._goto_calls == []
    assert stub.gate_lbl.last_text


def test_skip_bypasses_gate_on_gfunction():
    stub = _nav_stub("gfunction")
    assert stub.state.closure_scenario == ""

    stub._skip()

    assert stub._goto_calls == [ui.next_step("gfunction")]
    assert stub.state.step_status["gfunction"] == "skipped"


def test_advance_on_last_step_calls_finish_not_gated():
    stub = _nav_stub(ui.STEPS[-1][0])

    stub._advance()

    assert stub._finish_calls == [True]
    assert stub._goto_calls == []
    assert not stub.gate_lbl.last_text
