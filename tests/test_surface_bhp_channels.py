"""Surface and bottomhole pressure mapped as separate channels (PickState.surface_col +
bhp_col). "Bottomhole gauge" means a mapped downhole pressure channel; a surface transducer is
also a gauge, so the bare word is never used here. Surface checks (overlay, <50 psi warning,
low_pressure trim) key on surface_col being mapped, because surface on vacuum means the fluid
column is no longer full, which invalidates a bottomhole gauge reading too."""

from __future__ import annotations

import os

import numpy as np
import pytest

from dfit_tool import io_load, picks, store
from dfit_tool.io_load import TestData as IoTestData
from dfit_tool.model import (NO_PRESSURE_CHANNEL, SURFACE_NEEDS_BHP_INPUTS, PickState, _decode,
                             _encode, compute_all)
from tests.helpers import (PRESSURE_COL, RATE_COL, SHUTIN_IDX, START_IDX, VOLUME_COL,
                           injection_state, make_testdata)

_LOW_SURFACE = "Surface pressure fell below 50 psi"
BHG = "BHP"          # the bottomhole gauge column
SURF = "SURFACE"     # the surface column


def _column_fn(data: dict):
    def column(name):
        return data[name]
    return column


# --------------------------------------------------------------------------------------------------
# _is_bhp_named keywords
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["Downhole Pressure", "DownholePress", "Down Hole Pressure",
                                  "DH Press (psi)"])
def test_is_bhp_named_downhole_keywords(name):
    assert io_load._is_bhp_named(name)


@pytest.mark.parametrize("name", ["DHCP Pressure", "DHP", "Surface Pressure", "Pump Down Press"])
def test_is_bhp_named_rejects_lookalikes(name):
    assert not io_load._is_bhp_named(name)


@pytest.mark.parametrize("name", ["Bottomhole Press", "BottomholePressure", "Bottom Hole Pressure",
                                  "Bottom-Hole Press (psi)"])
def test_is_bhp_named_bottomhole_forms(name):
    assert io_load._is_bhp_named(name)


@pytest.mark.parametrize("name", ["Pressure Bottom Gauge", "Bottom Pressure", "Pressure Top Gauge"])
def test_is_bhp_named_bottom_alone_is_not_bottomhole(name):
    # APC Badger 11N/12N-3HZ: "Pressure Bottom Gauge" and "Pressure Top Gauge" are both surface
    # transducers; "bottom" without "hole" does not name a bottomhole gauge.
    assert not io_load._is_bhp_named(name)


def test_suggest_channels_badger_bottom_and_top_gauge_are_surface():
    # Badger's own shape: both transducers live (~1840-3910 psi), the named surface channel all zero.
    cols = ["DateTime", "Pressure Bottom Gauge", "Pressure Top Gauge", "Surface Pressure", "Rate"]
    data = {"Pressure Bottom Gauge": np.linspace(3900.0, 1840.0, 50),
            "Pressure Top Gauge": np.linspace(3910.0, 1846.0, 50),
            "Surface Pressure": np.zeros(50)}
    guess = io_load.suggest_channels(cols, column=lambda c: data[c])
    assert guess["bhp"] is None
    assert guess["pressure_is_bhp"] is False
    assert guess["surface"] == "Pressure Bottom Gauge"


def test_dh_max_press_is_candidate_but_demoted_not_suggested_as_bhp():
    name = "DH Max Press"
    assert io_load._is_pressure_candidate(name)
    assert io_load._pressure_tier(name) == 3
    guess = io_load.suggest_channels(["DateTime", name, "Surface Pressure"])
    assert guess["bhp"] is None
    assert guess["surface"] == "Surface Pressure"


# --------------------------------------------------------------------------------------------------
# suggest_channels: surface/bhp keys and per-pool liveness
# --------------------------------------------------------------------------------------------------
def test_suggest_channels_maps_both_surface_and_bhp():
    guess = io_load.suggest_channels(["DateTime", "BHP", "Surface Pressure"])
    assert guess["bhp"] == "BHP"
    assert guess["surface"] == "Surface Pressure"
    assert guess["pressure"] == "BHP"
    assert guess["pressure_is_bhp"] is True


def test_suggest_channels_live_surface_survives_much_higher_bottomhole_gauge():
    # Surface p99 2000 vs BHP p99 9000: judged together, the surface channel fails the relative
    # p99/range bars and is dropped. Liveness runs per pool, so it survives.
    cols = ["DateTime", "BHP", "Surface Pressure"]
    data = {
        "BHP": np.linspace(0.0, 9000.0, 50),
        "Surface Pressure": np.linspace(0.0, 2000.0, 50),
    }
    guess = io_load.suggest_channels(cols, column=_column_fn(data))
    assert guess["bhp"] == "BHP"
    assert guess["surface"] == "Surface Pressure"


def test_suggest_channels_no_bh_file_surface_equals_old_pressure_pick():
    cols = ["DateTime", "Pump Press", "Treating Pressure", "Surf Press [Csg]"]
    guess = io_load.suggest_channels(cols)
    assert guess["surface"] == "Surf Press [Csg]"
    assert guess["pressure"] == "Surf Press [Csg]"
    assert guess["bhp"] is None
    assert guess["pressure_is_bhp"] is False


def test_suggest_channels_no_bh_all_demoted_surface_keeps_old_fallback():
    guess = io_load.suggest_channels(["DateTime", "Max PSI", "Average Pressure"])
    assert guess["surface"] == "Max PSI"
    assert guess["pressure"] == "Max PSI"
    assert guess["bhp"] is None


def test_suggest_channels_no_candidates_has_no_surface_or_bhp():
    guess = io_load.suggest_channels(["DateTime", "Rate", "Volume"])
    assert guess["surface"] is None and guess["bhp"] is None


# --------------------------------------------------------------------------------------------------
# _decode migration
# --------------------------------------------------------------------------------------------------
def test_decode_old_bhp_save_maps_to_bhp_col_and_unit():
    st = _decode({"pressure_col": "X", "pressure_is_bhp": True, "pressure_unit": "kpa"})
    assert st.bhp_col == "X" and st.bhp_unit == "kpa"
    assert st.surface_col == ""


@pytest.mark.parametrize("extra", [{"pressure_is_bhp": False}, {}])
def test_decode_old_surface_save_maps_to_surface_col_and_unit(extra):
    st = _decode({"pressure_col": "X", "pressure_unit": "kpa", **extra})
    assert st.surface_col == "X" and st.surface_unit == "kpa"
    assert st.bhp_col == ""


def test_old_pressure_keys_are_not_reemitted_on_save():
    st = _decode({"pressure_col": "X", "pressure_is_bhp": True})
    d = _encode(st)
    assert "pressure_col" not in d and "pressure_is_bhp" not in d and "pressure_unit" not in d
    assert d["bhp_col"] == "X"


# --------------------------------------------------------------------------------------------------
# compute_all with a bottomhole gauge and surface
# --------------------------------------------------------------------------------------------------
def _two_channel_td(n: int = 1200, surface_crash_at=None, surface_glitch_idx=None,
                    bhg_glitch_idx=None):
    """Bottomhole gauge column (surface + 4000 psi during injection, then a steady linear
    decline so the 30-psi resampler keeps points past the surface crash) next to a surface
    column that can crash to 0 psi or take a single-sample glitch."""
    td = make_testdata(n=n)
    bhg = td.df[PRESSURE_COL].to_numpy().copy() + 4000.0
    bhg[SHUTIN_IDX:] = np.linspace(bhg[SHUTIN_IDX], 6000.0, n - SHUTIN_IDX)
    surf = make_testdata(n=n, zero_crash_at=surface_crash_at).df[PRESSURE_COL].to_numpy().copy()
    if surface_glitch_idx is not None:
        surf[surface_glitch_idx] = 1.0
    if bhg_glitch_idx is not None:
        bhg[bhg_glitch_idx:bhg_glitch_idx + 3] = 5.0
    td.df = td.df.drop(columns=[PRESSURE_COL])
    td.df[BHG] = bhg
    td.df[SURF] = surf
    td.columns = list(td.df.columns)
    return td


def _bh_state(td, with_surface=True):
    st = injection_state(td)
    st.surface_col = SURF if with_surface else ""
    st.bhp_col = BHG
    return st


def _seed(st, td):
    picks.seed_injection(st, td)
    res = compute_all(st, td)
    picks.seed_isip(st, td, res)
    return compute_all(st, td)


def test_bottomhole_gauge_is_bhp_with_no_hydrostatic_and_no_blocker():
    td = _two_channel_td()
    st = _bh_state(td)
    assert st.density_ppg is None and st.tvd_ft is None
    res = compute_all(st, td)
    assert res.blockers == []
    np.testing.assert_allclose(res.bhp_all, td.df[BHG].to_numpy())
    assert res.pressure_is_bhp is True
    np.testing.assert_allclose(res.p_surface_all, td.df[SURF].to_numpy())


def test_surface_crash_warns_and_auto_trims_on_bottomhole_gauge_test():
    td = _two_channel_td(surface_crash_at=0.5)
    st = _bh_state(td)
    res = _seed(st, td)
    assert any(_LOW_SURFACE in w for w in res.warnings)
    picks.seed_tail_trim(st, td, res)
    assert st.tail_trim_dt is not None
    assert st.tail_trim_reason == "low_pressure"


def test_single_surface_glitch_neither_warns_nor_trims():
    td = _two_channel_td(surface_glitch_idx=SHUTIN_IDX + 100)
    st = _bh_state(td)
    res = _seed(st, td)
    assert not any(_LOW_SURFACE in w for w in res.warnings)
    picks.seed_tail_trim(st, td, res)
    assert st.tail_trim_dt is None
    assert res.dropout_mask is not None and not res.dropout_mask.any()  # surface hit never masks BHP


def test_bottomhole_gauge_dropout_still_masks_bhp():
    gi = SHUTIN_IDX + 100
    td = _two_channel_td(bhg_glitch_idx=gi)
    st = _bh_state(td)
    res = _seed(st, td)
    assert res.dropout_mask[gi:gi + 3].all()
    assert not any(_LOW_SURFACE in w for w in res.warnings)


# --------------------------------------------------------------------------------------------------
# bottomhole gauge only
# --------------------------------------------------------------------------------------------------
def test_bottomhole_gauge_only_has_no_surface_checks_and_no_blocker():
    td = _two_channel_td(surface_crash_at=0.5)
    st = _bh_state(td, with_surface=False)
    res = _seed(st, td)
    assert res.blockers == []
    assert res.p_surface_all is None
    assert not any(_LOW_SURFACE in w for w in res.warnings)
    picks.seed_tail_trim(st, td, res)
    assert st.tail_trim_dt is None


# --------------------------------------------------------------------------------------------------
# Calculated (bhp_col empty)
# --------------------------------------------------------------------------------------------------
def test_calculated_without_surface_is_no_pressure_channel():
    td = make_testdata()
    st = injection_state(td)
    st.surface_col = ""
    st.bhp_col = ""
    res = compute_all(st, td)
    assert res.blockers == [NO_PRESSURE_CHANNEL]


def test_calculated_with_surface_but_no_inputs_needs_bhp_inputs():
    td = make_testdata()
    st = injection_state(td)
    assert st.surface_col == PRESSURE_COL and st.bhp_col == ""
    res = compute_all(st, td)
    assert res.blockers == [SURFACE_NEEDS_BHP_INPUTS]
    assert res.p_surface_all is None
    assert res.pressure_is_bhp is False


def test_calculated_with_surface_and_inputs_converts_as_before():
    td = make_testdata()
    st = injection_state(td)
    st.density_ppg, st.tvd_ft = 8.33, 10000.0
    res = compute_all(st, td)
    head = io_load.hydrostatic_head(8.33, 10000.0)
    np.testing.assert_allclose(res.bhp_all, td.df[PRESSURE_COL].to_numpy() + head)
    assert res.pressure_is_bhp is True
    np.testing.assert_allclose(res.p_surface_all, td.df[PRESSURE_COL].to_numpy())
    assert res.blockers == []


# --------------------------------------------------------------------------------------------------
# prime-filter surface pressure
# --------------------------------------------------------------------------------------------------
def test_seed_surface_pressure_uses_real_surface_column_even_with_bottomhole_gauge():
    td = _two_channel_td()
    st = _bh_state(td)
    out = picks._seed_surface_pressure(st, td)
    np.testing.assert_allclose(out, td.df[SURF].to_numpy())


def test_seed_surface_pressure_bottomhole_gauge_only_subtracts_hydrostatic():
    td = _two_channel_td()
    st = _bh_state(td, with_surface=False)
    assert picks._seed_surface_pressure(st, td) is None  # no density/TVD
    st.density_ppg, st.tvd_ft = 8.33, 10000.0
    out = picks._seed_surface_pressure(st, td)
    np.testing.assert_allclose(out, td.df[BHG].to_numpy() - io_load.hydrostatic_head(8.33, 10000.0))


# --------------------------------------------------------------------------------------------------
# log row
# --------------------------------------------------------------------------------------------------
def _log_row(tmp_path, st, td):
    res = compute_all(st, td)
    entry = store.TestEntry(test_id="well1", folder=str(tmp_path))
    return store.build_log_row(entry, os.path.join(str(tmp_path), "well1.csv"), str(tmp_path),
                               st, td, res)


def test_pressure_source_is_bottomhole_gauge_when_bhp_col_set(tmp_path):
    td = _two_channel_td()
    assert _log_row(tmp_path, _bh_state(td), td)["pressure_source"] == "bottomhole gauge"


def test_pressure_source_is_calculated_when_bhp_col_empty(tmp_path):
    td = make_testdata()
    st = injection_state(td)
    assert _log_row(tmp_path, st, td)["pressure_source"] == "calculated"


# --------------------------------------------------------------------------------------------------
# Overview overlay and UI sync
# --------------------------------------------------------------------------------------------------
def test_overview_overlays_surface_on_a_bottomhole_gauge_test():
    from matplotlib.figure import Figure
    from dfit_tool import plots
    td = _two_channel_td()
    st = _bh_state(td)
    res = compute_all(st, td)
    ax = Figure().add_subplot(111)
    plots.render_overview(ax, td, st, res)
    assert [ln for ln in ax.get_lines() if ln.get_gid() == "surface_pressure"]


def test_sync_maps_calculated_sentinel_to_empty_bhp_col():
    import types
    from dfit_tool import ui

    class _Var:
        def __init__(self, v):
            self.v = v

        def get(self):
            return self.v

    stub = types.SimpleNamespace(td=None, state=PickState())
    for name, val in [("var_surface", "S"), ("var_bhp", ui.BHP_CALCULATED), ("var_rate", ""),
                      ("var_volume", ""), ("var_surface_unit", "kpa"), ("var_bhp_unit", "auto"),
                      ("var_rate_unit", "auto"), ("var_volume_unit", "auto"),
                      ("var_density", ""), ("var_density_fallback", False), ("var_tvd", ""),
                      ("var_well", ""), ("var_formation", ""), ("var_alpha", "1.0"),
                      ("var_step", "30")]:
        setattr(stub, name, _Var(val))
    ui.DfitApp._sync_state_from_widgets(stub)
    assert stub.state.surface_col == "S" and stub.state.bhp_col == ""
    assert stub.state.surface_unit == "kpa"
    stub.var_bhp.v = "B"
    ui.DfitApp._sync_state_from_widgets(stub)
    assert stub.state.bhp_col == "B"


# --------------------------------------------------------------------------------------------------
# review round: a BH-named channel must pass the OLD joint liveness filter
# --------------------------------------------------------------------------------------------------
def test_suggest_channels_disconnected_bottomhole_gauge_loses_to_live_surface():
    rng = np.random.default_rng(0)
    data = {"BHP": 200.0 + 120.0 * rng.random(500),
            "Surface Pressure": np.linspace(6000.0, 500.0, 500)}
    guess = io_load.suggest_channels(["DateTime", "BHP", "Surface Pressure"],
                                     column=_column_fn(data))
    assert guess["bhp"] is None
    assert guess["surface"] == "Surface Pressure"
    assert guess["pressure"] == "Surface Pressure"
    assert guess["pressure_is_bhp"] is False


def test_suggest_channels_flat_bottomhole_gauge_loses_to_live_surface():
    data = {"BHP": np.linspace(9000.0, 8800.0, 500),
            "Surface Pressure": np.linspace(6000.0, 500.0, 500)}
    guess = io_load.suggest_channels(["DateTime", "BHP", "Surface Pressure"],
                                     column=_column_fn(data))
    assert guess["bhp"] is None
    assert guess["pressure"] == "Surface Pressure"


# --------------------------------------------------------------------------------------------------
# review round: surface_dropout_mask honors keep/manual intervals and is never silent
# --------------------------------------------------------------------------------------------------
_SURFACE_IGNORED = "dropout"   # "Surface: N dropout sample(s) ignored for the 50 psi check"


def _surface_ignored_warnings(res):
    return [w for w in res.warnings if w.startswith("Surface:") and "ignored" in w]


def test_unkept_surface_dip_adds_one_short_warning_on_bottomhole_gauge_test():
    gi = SHUTIN_IDX + 100
    td = _two_channel_td(surface_glitch_idx=gi)
    st = _bh_state(td)
    res = _seed(st, td)
    found = _surface_ignored_warnings(res)
    assert len(found) == 1
    assert "1 dropout sample ignored for the 50 psi check" in found[0]
    assert len(found[0]) <= 90


def test_no_surface_ignored_warning_when_bhp_is_calculated():
    gi = SHUTIN_IDX + 100
    td = _two_channel_td(surface_glitch_idx=gi)
    st = _bh_state(td)
    st.bhp_col = ""
    st.density_ppg, st.tvd_ft = 8.33, 8000.0
    res = _seed(st, td)
    assert _surface_ignored_warnings(res) == []


def test_keep_interval_over_surface_dip_restores_low_surface_warning():
    gi = SHUTIN_IDX + 100
    td = _two_channel_td(surface_glitch_idx=gi)
    st = _bh_state(td)
    res = _seed(st, td)
    assert not any(_LOW_SURFACE in w for w in res.warnings)
    st.keep_intervals = [(float(td.t_s[gi - 1]), float(td.t_s[gi + 1]))]
    res2 = compute_all(st, td)
    assert any(_LOW_SURFACE in w for w in res2.warnings)
    assert _surface_ignored_warnings(res2) == []


def test_manual_mask_hides_surface_samples_in_overlay():
    from matplotlib.figure import Figure
    from dfit_tool import plots
    td = _two_channel_td()
    st = _bh_state(td)
    lo, hi = float(td.t_s[SHUTIN_IDX + 100]), float(td.t_s[SHUTIN_IDX + 110])
    st.mask_intervals = [(lo, hi)]
    res = compute_all(st, td)
    assert res.surface_dropout_mask[SHUTIN_IDX + 100:SHUTIN_IDX + 111].all()
    ax = Figure().add_subplot(111)
    plots.render_overview(ax, td, st, res)
    surf = [ln for ln in ax.get_lines() if ln.get_gid() == "surface_pressure"][0]
    assert np.isnan(surf.get_ydata()).any()


# --------------------------------------------------------------------------------------------------
# review round: decode with empty new keys, unit note once, Bottomhole unit combo, summary source
# --------------------------------------------------------------------------------------------------
def test_decode_migrates_when_new_keys_are_present_but_empty():
    st = _decode({"pressure_col": "X", "pressure_is_bhp": True, "pressure_unit": "kpa",
                  "surface_col": "", "bhp_col": ""})
    assert st.bhp_col == "X" and st.bhp_unit == "kpa" and st.surface_col == ""
    st2 = _decode({"pressure_col": "X", "surface_col": "", "bhp_col": ""})
    assert st2.surface_col == "X" and st2.bhp_col == ""


def test_decode_does_not_migrate_over_a_real_new_mapping():
    st = _decode({"pressure_col": "X", "pressure_is_bhp": True, "surface_col": "S",
                  "bhp_col": ""})
    assert st.surface_col == "S" and st.bhp_col == ""


def test_unit_note_lists_a_shared_column_once():
    td = make_testdata()
    kpa_col = "PRESSURE (KPA)"
    td.df[kpa_col] = td.df[PRESSURE_COL] / 0.145038
    td.columns = list(td.df.columns)
    st = injection_state(td)
    st.surface_col = st.bhp_col = kpa_col
    res = compute_all(st, td)
    assert res.unit_conversion_note.count("kpa") == 1
    assert len([w for w in res.warnings if "ressure converted from" in w]) == 1


def test_bottomhole_unit_combo_disabled_while_calculated(monkeypatch):
    import types
    from dfit_tool import ui

    class _Font:
        def copy(self):
            return _Font()

        def configure(self, **kw):
            pass

    monkeypatch.setattr(ui.tkfont, "nametofont", lambda name: _Font())

    class _Cmb:
        def __init__(self):
            self.opts = {"font": "TkTextFont"}

        def cget(self, k):
            return self.opts[k]

        def configure(self, **kw):
            self.opts.update(kw)

    stub = types.SimpleNamespace(cmb_bhp=_Cmb(), cmb_bhp_unit=_Cmb(), _bhp_fonts=None,
                                 var_bhp=types.SimpleNamespace(get=lambda: ui.BHP_CALCULATED))
    ui.DfitApp._update_bhp_combo_font(stub)
    assert stub.cmb_bhp_unit.opts["state"] == "disabled"
    stub.var_bhp = types.SimpleNamespace(get=lambda: "BHP")
    ui.DfitApp._update_bhp_combo_font(stub)
    assert stub.cmb_bhp_unit.opts["state"] == "readonly"


def _source_row(state, res):
    from dfit_tool import summary
    sec = [s for s in summary.summary_sections(state, res) if s.title == "Inputs and data"][0]
    return [r for r in sec.rows if r[0] == "pressure source"][0][1]


def test_summary_source_says_bottomhole_gauge_only_when_it_was_used():
    td = _two_channel_td()
    st = _bh_state(td)
    assert _source_row(st, compute_all(st, td)) == "bottomhole gauge"
    # compute_all's failed-BHP-computation fallback leaves surface in bhp_all and
    # pressure_is_bhp False even though bhp_col is mapped.
    res = compute_all(st, td)
    res.pressure_is_bhp = False
    assert _source_row(st, res) != "bottomhole gauge"
