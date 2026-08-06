"""PC-F ("no peak") skips the pore-pressure step entirely, and the postclosure scenario
labels carry through the rename cleanly.

Covers: model.compute_all suppressing pore_pressure under PC-F; model._decode migrating the
four pre-rename labels; plots.save_all_step_pngs omitting the porepressure PNG; and the
ui.DfitApp navigation helpers (_advance/_goto) routing around the skipped step. Same
duck-typed stand-in pattern as tests/test_step_gate.py -- no real tk.Tk().
"""

from __future__ import annotations

import types

import pytest

from dfit_tool import picks, plots, ui
from dfit_tool.model import PickState, _decode, compute_all, porepressure_skipped
from dfit_tool.ui import DfitApp
from tests.helpers import make_testdata, overview_state


# --------------------------------------------------------------------------------------------------
# compute_all: PC-F suppresses pore_pressure even with a valid pp_window; a non-PC-F scenario
# with the same window still produces a value.
# --------------------------------------------------------------------------------------------------
def _state_with_pp_window(td) -> PickState:
    st = overview_state(td)
    res = compute_all(st, td)
    picks.seed_pp(st, res)
    assert st.pp_window is not None
    return st


def test_compute_all_suppresses_pore_pressure_under_pcf():
    td = make_testdata()
    st = _state_with_pp_window(td)
    st.postclosure_scenario = "PC-F no peak"

    res = compute_all(st, td)

    assert porepressure_skipped(st)
    assert res.pore_pressure is None


def test_compute_all_still_computes_pore_pressure_for_non_pcf_scenario():
    td = make_testdata()
    st = _state_with_pp_window(td)
    st.postclosure_scenario = "PC-A linear"
    st.pp_axis = picks.suggest_pp_axis(st.postclosure_scenario) or st.pp_axis

    res = compute_all(st, td)

    assert not porepressure_skipped(st)
    assert res.pore_pressure is not None


# --------------------------------------------------------------------------------------------------
# model._decode: old labels normalize to the new ones; unrecognized strings pass through.
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("old,new", [
    ("PC-C mixed", "PC-C false radial to genuine linear"),
    ("PC-D mixed", "PC-D genuine linear to genuine radial"),
    ("PC-E none", "PC-E no trend"),
    ("PC-F none", "PC-F no peak"),
])
def test_decode_migrates_old_postclosure_labels(old, new):
    state = _decode({"postclosure_scenario": old})
    assert state.postclosure_scenario == new


@pytest.mark.parametrize("label", ["PC-A linear", "some foreign label"])
def test_decode_leaves_unrecognized_postclosure_labels_unchanged(label):
    state = _decode({"postclosure_scenario": label})
    assert state.postclosure_scenario == label


def test_decode_normalizes_legacy_explicit_status_done_to_none():
    # Old saves made with the since-removed Mark combobox could carry explicit_status ==
    # "done"; that field's value space narrowed to "skipped"/None (model.py), so a legacy
    # "done" must decode to None rather than linger as a value store.status_for no longer
    # trusts (belt-and-braces alongside store.status_for's own guard, see test_store.py).
    state = _decode({"explicit_status": "done"})
    assert state.explicit_status is None


def test_decode_leaves_explicit_status_skipped_unchanged():
    state = _decode({"explicit_status": "skipped"})
    assert state.explicit_status == "skipped"


def test_pickstate_from_json_roundtrip_migrates_label(tmp_path):
    path = tmp_path / "picks.json"
    st = PickState(postclosure_scenario="PC-F none")
    st.to_json(str(path))

    loaded = PickState.from_json(str(path))

    assert loaded.postclosure_scenario == "PC-F no peak"


# --------------------------------------------------------------------------------------------------
# plots.save_all_step_pngs: PC-F omits the porepressure PNG (five files, none named
# "*porepressure*"); a non-PC-F scenario still writes all six.
# --------------------------------------------------------------------------------------------------
def test_save_all_step_pngs_omits_porepressure_under_pcf(tmp_path):
    td = make_testdata()
    state = overview_state(td)
    state.postclosure_scenario = "PC-F no peak"
    res = compute_all(state, td)

    paths = plots.save_all_step_pngs(str(tmp_path), td, state, res, views={})

    assert len(paths) == 5
    assert not any("porepressure" in p for p in paths)
    assert not list(tmp_path.glob("*porepressure*"))


def test_save_all_step_pngs_writes_all_six_for_non_pcf_scenario(tmp_path):
    td = make_testdata()
    state = overview_state(td)
    state.postclosure_scenario = "PC-A linear"
    res = compute_all(state, td)

    paths = plots.save_all_step_pngs(str(tmp_path), td, state, res, views={})

    assert len(paths) == 6
    assert any("porepressure" in p for p in paths)


# --------------------------------------------------------------------------------------------------
# Navigation: DfitApp._advance/_goto route around the skipped pore-pressure step. Same
# duck-typed stand-in pattern as test_step_gate.py's _nav_stub -- _advance needs self._last_step,
# so the real DfitApp._last_step is bound onto the stub too.
# --------------------------------------------------------------------------------------------------
class _GateLabel:
    def __init__(self):
        self.texts = []

    def config(self, **kw):
        if "text" in kw:
            self.texts.append(kw["text"])

    @property
    def last_text(self):
        return self.texts[-1] if self.texts else None


def _nav_stub(step, postclosure_scenario=""):
    stub = types.SimpleNamespace()
    stub.td = object()  # truthy sentinel -- these methods only check `is None`
    stub.state = PickState(postclosure_scenario=postclosure_scenario)
    stub.step = step
    stub.gate_lbl = _GateLabel()
    stub._goto_calls = []
    stub._goto = lambda dest: stub._goto_calls.append(dest)
    stub._finish_calls = []
    stub._finish = lambda: stub._finish_calls.append(True)
    stub._last_step = types.MethodType(DfitApp._last_step, stub)
    stub._advance = types.MethodType(DfitApp._advance, stub)
    stub._next = types.MethodType(DfitApp._next, stub)
    stub._skip = types.MethodType(DfitApp._skip, stub)
    return stub


def test_advance_on_loglog_with_pcf_calls_finish_not_goto():
    stub = _nav_stub("loglog", postclosure_scenario="PC-F no peak")

    stub._advance()

    assert stub._finish_calls == [True]
    assert stub._goto_calls == []


def test_advance_on_loglog_with_non_pcf_gotos_porepressure():
    stub = _nav_stub("loglog", postclosure_scenario="PC-A linear")

    stub._advance()

    assert stub._goto_calls == [ui.next_step("loglog")]
    assert stub._finish_calls == []


def _goto_stub(postclosure_scenario):
    """A stand-in exposing what the real DfitApp._goto touches: self.td, self.state, self.step,
    self._seed_step, and self.refresh. step_status is prefilled so every step looks already
    visited (the real _goto's seed-on-first-visit branch is irrelevant to this redirect logic)."""
    stub = types.SimpleNamespace()
    stub.td = object()
    stub.state = PickState(postclosure_scenario=postclosure_scenario,
                           step_status={k: "visited" for k, _ in ui.STEPS})
    stub.step = "overview"
    stub._seed_step = lambda key: None
    stub._refresh_calls = []
    stub.refresh = lambda: stub._refresh_calls.append(True)
    stub._goto = types.MethodType(DfitApp._goto, stub)
    return stub


def test_goto_porepressure_redirects_to_loglog_under_pcf():
    stub = _goto_stub("PC-F no peak")

    stub._goto("porepressure")

    assert stub.step == "loglog"


def test_goto_porepressure_lands_on_porepressure_under_non_pcf():
    stub = _goto_stub("PC-A linear")

    stub._goto("porepressure")

    assert stub.step == "porepressure"
