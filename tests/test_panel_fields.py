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


def test_shmin_liberty_row_present_and_owned_by_gfunction_step():
    # Display + log only -- no asterisk logic, this row is never a fallback substitution.
    assert "Shmin Liberty" in ui.PANEL_FIELDS
    assert ui.FIELD_STEP["Shmin Liberty"] == "gfunction"


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


def test_update_panel_shmin_liberty_renders_when_visited():
    res = DerivedResults(shmin_liberty=9150.0)
    state = PickState(step_status={"gfunction": "done"})
    stub = _panel_stub(res, state)
    stub._update_panel()
    assert stub.value_lbls["Shmin Liberty"].text == "9150"
    assert stub.name_lbls["Shmin Liberty"].text == "Shmin Liberty"  # never asterisked


def test_update_panel_shmin_liberty_blanks_when_gfunction_not_visited():
    res = DerivedResults(shmin_liberty=9150.0)
    state = PickState(step_status={})
    stub = _panel_stub(res, state)
    stub._update_panel()
    assert stub.value_lbls["Shmin Liberty"].text == "-"


def test_update_panel_collapses_warnings_by_default():
    res = DerivedResults(warnings=["Tail auto-trimmed.", "Pressure dropout masked."])
    state = PickState(step_status={"gfunction": "done"})
    stub = _panel_stub(res, state)

    stub._update_panel()

    assert stub.warn_lbl.text == "2 warnings (click to expand)"
    assert stub._warnings_expanded is False
    assert stub._warnings_list == ["Tail auto-trimmed.", "Pressure dropout masked."]


def test_update_panel_blanks_warnings_label_when_clean():
    res = DerivedResults(warnings=[])
    state = PickState(step_status={"gfunction": "done"})
    stub = _panel_stub(res, state)

    stub._update_panel()

    assert stub.warn_lbl.text == ""
    assert stub._warnings_list == []


# --------------------------------------------------------------------------------------------------
# Gradient rows: "apparent ISIP grad", "Shmin compliance grad", "pore pressure grad" -- each
# inline directly under its parent row.
# --------------------------------------------------------------------------------------------------
def test_gradient_rows_present_directly_under_their_parents():
    for parent, grad in (("apparent ISIP", "apparent ISIP grad"),
                         ("Shmin compliance", "Shmin compliance grad"),
                         ("pore pressure", "pore pressure grad")):
        assert grad in ui.PANEL_FIELDS
        assert ui.PANEL_FIELDS.index(grad) == ui.PANEL_FIELDS.index(parent) + 1


def test_gradient_rows_owned_by_same_step_as_their_parent():
    assert ui.FIELD_STEP["apparent ISIP grad"] == ui.FIELD_STEP["apparent ISIP"] == "isip"
    assert (ui.FIELD_STEP["Shmin compliance grad"] == ui.FIELD_STEP["Shmin compliance"]
            == "gfunction")
    assert (ui.FIELD_STEP["pore pressure grad"] == ui.FIELD_STEP["pore pressure"]
            == "porepressure")


def test_no_shmin_rapid_grad_row():
    # shmin_rapid_gradient (the model field, logged to the CSV) gets no panel row of its own --
    # same precedent as shmin_rapid itself, which folds into "Shmin compliance" via the asterisk.
    assert "Shmin rapid grad" not in ui.PANEL_FIELDS
    assert "Shmin rapid grad" not in ui.FIELD_STEP


def test_update_panel_gradient_rows_render_3_decimals():
    res = DerivedResults(apparent_isip_gradient=0.6543, shmin_compliance_gradient=0.5987,
                         pore_pressure_gradient=0.4321)
    state = PickState(step_status={"isip": "done", "gfunction": "done",
                                   "porepressure": "done"})
    stub = _panel_stub(res, state)
    stub._update_panel()
    assert stub.value_lbls["apparent ISIP grad"].text == "0.654"
    assert stub.value_lbls["Shmin compliance grad"].text == "0.599"
    assert stub.value_lbls["pore pressure grad"].text == "0.432"


def test_update_panel_gradient_rows_blank_when_not_visited():
    res = DerivedResults(apparent_isip_gradient=0.6543, shmin_compliance_gradient=0.5987,
                         pore_pressure_gradient=0.4321)
    state = PickState(step_status={})
    stub = _panel_stub(res, state)
    stub._update_panel()
    assert stub.value_lbls["apparent ISIP grad"].text == "-"
    assert stub.value_lbls["Shmin compliance grad"].text == "-"
    assert stub.value_lbls["pore pressure grad"].text == "-"


def test_update_panel_cd_rapid_marks_shmin_compliance_grad_row_too():
    # The parent row's asterisk pair extends to the grad row: same use_rapid gate, no ±75
    # half-range in the value (a gradient of that band isn't worth rendering).
    res = DerivedResults(shmin_compliance=None, shmin_rapid=9325.0,
                         shmin_compliance_gradient=None, shmin_rapid_gradient=0.9325)
    state = PickState(step_status={"gfunction": "done"})
    stub = _panel_stub(res, state)
    stub._update_panel()
    assert stub.name_lbls["Shmin compliance grad"].text == "Shmin compliance grad*"
    assert stub.value_lbls["Shmin compliance grad"].text == "0.932"


def test_update_panel_non_rapid_shmin_compliance_grad_row_plain():
    res = DerivedResults(shmin_compliance=9200.0, shmin_rapid=None,
                         shmin_compliance_gradient=0.92, shmin_rapid_gradient=None)
    state = PickState(step_status={"gfunction": "done"})
    stub = _panel_stub(res, state)
    stub._update_panel()
    assert stub.name_lbls["Shmin compliance grad"].text == "Shmin compliance grad"
    assert stub.value_lbls["Shmin compliance grad"].text == "0.920"


def test_update_panel_cd_rapid_no_tvd_leaves_grad_row_plain():
    # use_rapid is True and gfunction is visited (the parent row's own asterisk gate), but
    # model._resolve_gradients' tvd_ft > 0 guard blanked shmin_rapid_gradient (e.g. a
    # pressure_is_bhp downhole-gauge test with no TVD entered), so the grad row's value is "-".
    # The label must stay plain -- an asterisk here would sit next to a "-", violating the panel
    # invariant that an asterisk never marks a blank value.
    res = DerivedResults(shmin_compliance=None, shmin_rapid=9325.0,
                         shmin_compliance_gradient=None, shmin_rapid_gradient=None)
    state = PickState(step_status={"gfunction": "done"})
    stub = _panel_stub(res, state)
    stub._update_panel()
    assert stub.name_lbls["Shmin compliance grad"].text == "Shmin compliance grad"
    assert stub.value_lbls["Shmin compliance grad"].text == "-"


def test_update_panel_shmin_compliance_grad_asterisk_clears_on_second_refresh():
    state = PickState(step_status={"gfunction": "done"})
    stub = _panel_stub(DerivedResults(shmin_compliance=None, shmin_rapid=9325.0,
                                      shmin_compliance_gradient=None,
                                      shmin_rapid_gradient=0.9325), state)
    stub._update_panel()
    assert stub.name_lbls["Shmin compliance grad"].text == "Shmin compliance grad*"

    stub.res = DerivedResults(shmin_compliance=9200.0, shmin_rapid=None,
                              shmin_compliance_gradient=0.92, shmin_rapid_gradient=None)
    stub._update_panel()
    assert stub.name_lbls["Shmin compliance grad"].text == "Shmin compliance grad"


def test_format_warnings_text_empty():
    assert ui.format_warnings_text([], expanded=False) == ""
    assert ui.format_warnings_text([], expanded=True) == ""


def test_format_warnings_text_collapsed_singular():
    assert ui.format_warnings_text(["Tail auto-trimmed."], expanded=False) == (
        "1 warning (click to expand)")


def test_format_warnings_text_collapsed_plural():
    warnings = ["Tail auto-trimmed.", "Pressure dropout masked."]
    assert ui.format_warnings_text(warnings, expanded=False) == (
        "2 warnings (click to expand)")


def test_format_warnings_text_expanded_shows_full_list():
    warnings = ["Tail auto-trimmed.", "Pressure dropout masked."]
    assert ui.format_warnings_text(warnings, expanded=True) == (
        "2 warnings (click to collapse)\nTail auto-trimmed.\nPressure dropout masked.")
