from dfit_tool import ui


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
