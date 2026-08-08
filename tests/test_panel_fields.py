import types

from dfit_tool import ui
from dfit_tool.model import DerivedResults, PickState


def test_extra_eff_isip_rows_removed():
    assert "eff ISIP (compliance)" in ui.PANEL_FIELDS
    assert "eff ISIP (tangent)" not in ui.PANEL_FIELDS
    assert "eff ISIP (variable)" not in ui.PANEL_FIELDS
    assert "eff ISIP (tangent)" not in ui.FIELD_STEP
    assert "eff ISIP (variable)" not in ui.FIELD_STEP


def test_net_pressure_rows_kept():
    for row in ("net (compliance)", "net (tangent)", "net (variable)"):
        assert row in ui.PANEL_FIELDS


def test_nwb_complexity_row_sits_directly_under_eff_isip():
    # The panel then reads apparent ISIP -> eff ISIP -> complexity straight down, so the
    # subtraction is visible.
    assert (ui.PANEL_FIELDS.index("NWB complexity")
            == ui.PANEL_FIELDS.index("eff ISIP (compliance)") + 1)


def test_nwb_complexity_owned_by_gfunction_step():
    # It needs both the isip and gfunction picks; gfunction is the later of the two, the same
    # precedent "net (compliance)" follows.
    assert ui.FIELD_STEP["NWB complexity"] == "gfunction"


def test_shmin_rapid_folded_into_shmin_compliance_row():
    # There is no longer a separate "Shmin rapid" panel row -- _update_panel folds it into
    # "Shmin compliance", marked by an asterisk on the label (the value carries no marker).
    assert "Shmin rapid" not in ui.PANEL_FIELDS
    assert "Shmin rapid" not in ui.FIELD_STEP
    assert "Shmin compliance" in ui.PANEL_FIELDS
    assert "Shmin compliance" in ui.FIELD_STEP


class _FakeLabel:
    """Minimal duck-typed ttk.Label stand-in: records the last .config(text=...) call."""

    def __init__(self, text="-"):
        self.text = text

    def config(self, **kw):
        if "text" in kw:
            self.text = kw["text"]


def _panel_stub(res, state):
    """Duck-typed DfitApp stand-in exposing only what _update_panel touches, same pattern as
    test_view_state.py's _refresh_stub."""
    stub = types.SimpleNamespace()
    stub.res = res
    stub.state = state
    stub.value_lbls = {k: _FakeLabel() for k in ui.PANEL_FIELDS}
    stub.name_lbls = {k: _FakeLabel(text=k) for k in ui.PANEL_FIELDS}
    stub.warn_lbl = _FakeLabel()
    stub._update_panel = types.MethodType(ui.DfitApp._update_panel, stub)
    return stub


def test_update_panel_cd_rapid_marks_shmin_compliance_row():
    # C-D: no compliance Shmin, but shmin_rapid stands in for it in the same row.
    res = DerivedResults(shmin_compliance=None, shmin_rapid=9325.0)
    state = PickState(step_status={"gfunction": "done"})
    stub = _panel_stub(res, state)
    stub._update_panel()
    assert stub.name_lbls["Shmin compliance"].text == "Shmin compliance*"
    assert stub.value_lbls["Shmin compliance"].text == "9325 ±75"


def test_update_panel_non_rapid_shmin_compliance_row_plain():
    res = DerivedResults(shmin_compliance=9200.0, shmin_rapid=None)
    state = PickState(step_status={"gfunction": "done"})
    stub = _panel_stub(res, state)
    stub._update_panel()
    assert stub.name_lbls["Shmin compliance"].text == "Shmin compliance"
    assert stub.value_lbls["Shmin compliance"].text == "9200"


def test_update_panel_not_visited_blanks_value_and_asterisk():
    # gfunction not yet visited: the value is blanked to "-" by the existing visited-gate, and
    # the asterisk must not appear next to it either.
    res = DerivedResults(shmin_compliance=None, shmin_rapid=9325.0)
    state = PickState(step_status={})
    stub = _panel_stub(res, state)
    stub._update_panel()
    assert stub.value_lbls["Shmin compliance"].text == "-"
    assert stub.name_lbls["Shmin compliance"].text == "Shmin compliance"


def test_update_panel_complexity_referenced_to_tangent_marks_row():
    # net_pressure_isip_source == "tangent" means the shared eff ISIP reference fell back from
    # compliance (C-C/C-D clear the contact pick), so the complexity row is marked.
    res = DerivedResults(near_wellbore_complexity=450.0, net_pressure_isip_source="tangent")
    state = PickState(step_status={"gfunction": "done"})
    stub = _panel_stub(res, state)
    stub._update_panel()
    assert stub.name_lbls["NWB complexity"].text == "NWB complexity*"
    assert stub.value_lbls["NWB complexity"].text == "450"


def test_update_panel_complexity_referenced_to_compliance_not_marked():
    # net_pressure_isip_source == "compliance" is the primary reference -- no asterisk.
    res = DerivedResults(near_wellbore_complexity=450.0, net_pressure_isip_source="compliance")
    state = PickState(step_status={"gfunction": "done"})
    stub = _panel_stub(res, state)
    stub._update_panel()
    assert stub.name_lbls["NWB complexity"].text == "NWB complexity"
    assert stub.value_lbls["NWB complexity"].text == "450"


def test_update_panel_complexity_blank_never_marked():
    # No complexity value (blank "-"): the asterisk must not appear even though the source says
    # "tangent" -- same not_visited-style gate the value column uses.
    res = DerivedResults(near_wellbore_complexity=None, net_pressure_isip_source="tangent")
    state = PickState(step_status={"gfunction": "done"})
    stub = _panel_stub(res, state)
    stub._update_panel()
    assert stub.value_lbls["NWB complexity"].text == "-"
    assert stub.name_lbls["NWB complexity"].text == "NWB complexity"


def test_update_panel_both_asterisks_clear_on_a_second_refresh():
    """Both label resets are load-bearing: the name labels are built once in _build_body and
    persist across refreshes, so switching the closure scenario away from C-D (contact pick
    re-derived -> a real compliance Shmin and a compliance-referenced complexity) has to clear
    both asterisks. The single-call tests above can't catch a dropped else branch, since the
    stub's labels start out plain."""
    state = PickState(step_status={"gfunction": "done"})
    stub = _panel_stub(DerivedResults(shmin_compliance=None, shmin_rapid=9325.0,
                                      near_wellbore_complexity=450.0,
                                      net_pressure_isip_source="tangent"), state)
    stub._update_panel()
    assert stub.name_lbls["Shmin compliance"].text == "Shmin compliance*"
    assert stub.name_lbls["NWB complexity"].text == "NWB complexity*"

    stub.res = DerivedResults(shmin_compliance=9200.0, shmin_rapid=None,
                              near_wellbore_complexity=450.0,
                              net_pressure_isip_source="compliance")
    stub._update_panel()
    assert stub.name_lbls["Shmin compliance"].text == "Shmin compliance"
    assert stub.name_lbls["NWB complexity"].text == "NWB complexity"
