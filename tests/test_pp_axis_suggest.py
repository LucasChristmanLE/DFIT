"""picks.suggest_pp_axis: postclosure-scenario -> pore-pressure-axis mapping (pure, no Tk)."""

from dfit_tool import picks


def test_tm12_scenarios():
    for scen in ("PC-A linear", "PC-C false radial to genuine linear", "PC-E no trend"):
        assert picks.suggest_pp_axis(scen) == "tm12"


def test_tm1_scenario():
    assert picks.suggest_pp_axis("PC-B false-radial") == "tm1"


def test_no_dictated_axis():
    for scen in ("", "PC-D genuine linear to genuine radial", "PC-F no peak"):
        assert picks.suggest_pp_axis(scen) is None
