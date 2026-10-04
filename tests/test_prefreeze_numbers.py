"""Pre-freeze fixes for silently changed numbers (batch B): Vinj with non-finite samples, stale-pick
warning without a trim, effective tail-trim cutoff in its warning, unchosen closure scenario, and
an inverted injection window."""
from __future__ import annotations

import numpy as np
import pytest

from dfit_tool import interpret
from dfit_tool.model import compute_all
from tests.helpers import PRESSURE_COL, RATE_COL, VOLUME_COL, injection_state, make_testdata


# ---- Q7: interpret.injected_volume -----------------------------------------------------------
def _flat_rate(n=11, q=2.0):
    t = np.arange(n, dtype=float) * 60.0  # one minute apart
    return t, np.full(n, q)


def test_vinj_nan_rate_sample_is_skipped_not_poisoning():
    t, q = _flat_rate()
    q[4] = np.nan
    vr = interpret.injected_volume(t, q, 0, 10)
    assert np.isfinite(vr.vinj_integral)
    assert vr.vinj_integral == pytest.approx(2.0 * 9.0, rel=1e-6)  # 9 min across the NaN gap
    assert vr.n_nonfinite_skipped == 1


def test_vinj_nan_volume_endpoint_uses_nearest_finite_samples():
    t, q = _flat_rate()
    v = np.arange(11, dtype=float) * 2.0
    v[0] = np.nan
    v[10] = np.nan
    vr = interpret.injected_volume(t, q, 0, 10, volume=v)
    assert vr.vinj_delta == pytest.approx(v[9] - v[1])
    assert vr.source == "volume_channel"


def test_vinj_volume_reset_falls_back_to_rate_integral():
    t, q = _flat_rate()
    v = np.arange(11, dtype=float) * 2.0
    v[6:] -= 100.0  # counter reset mid-window: end < start
    vr = interpret.injected_volume(t, q, 0, 10, volume=v)
    assert vr.source == "rate_integral"
    assert vr.vinj == pytest.approx(vr.vinj_integral)
    assert vr.vinj_delta is None
    assert vr.volume_unusable is True


def test_vinj_clean_inputs_unchanged():
    t, q = _flat_rate()
    v = np.arange(11, dtype=float) * 2.0
    vr = interpret.injected_volume(t, q, 0, 10, volume=v)
    assert vr.vinj == pytest.approx(20.0)
    assert vr.n_nonfinite_skipped == 0 and vr.volume_unusable is False


def test_compute_all_nan_rate_keeps_vinj_te_and_warns_once():
    td = make_testdata()
    st = injection_state(td)
    td.df.loc[150, RATE_COL] = np.nan
    td.df.loc[100, VOLUME_COL] = np.nan
    res = compute_all(st, td)
    assert res.te_s is not None and np.isfinite(res.vinj)
    assert not any("No usable rate" in w for w in res.warnings)
    vinj_w = [w for w in res.warnings if w.startswith("Vinj:") and "skipped" in w]
    assert len(vinj_w) == 1 and len(vinj_w[0]) <= 90


def test_compute_all_volume_reset_warns_and_uses_rate_integral():
    td = make_testdata()
    st = injection_state(td)
    td.df.loc[200:, VOLUME_COL] -= 1000.0
    res = compute_all(st, td)
    assert res.vinj_source == "rate_integral"
    assert any(w.startswith("Vinj:") and "rate integral" in w and len(w) <= 90
               for w in res.warnings)


# ---- Q8 / Q9: tail guard ---------------------------------------------------------------------
def _guarded_td():
    """make_testdata with a sustained +150 psi step up late in the record: the tail guard fires."""
    td = make_testdata(n=1200)
    td.df.loc[900:, PRESSURE_COL] = td.df.loc[899, PRESSURE_COL] + 150.0
    return td


def test_stale_pick_warning_fires_without_a_trim_when_guard_cuts_picks():
    td = _guarded_td()
    st = injection_state(td)
    st.closure_scenario = "C-A clear"
    res = compute_all(st, td)
    assert res.resampled_full.guard_dt is not None and st.tail_trim_dt is None
    edge = res.diagnostics.G[-1]
    st.contact_G = float(edge) * 2.0
    res = compute_all(st, td)
    assert any("contact" in w and "beyond the tail trim" in w for w in res.warnings)


def test_tail_trimmed_warning_reports_effective_cutoff_when_clamped_to_guard():
    td = _guarded_td()
    st = injection_state(td)
    res0 = compute_all(st, td)
    guard_dt = res0.resampled_full.guard_dt
    assert guard_dt is not None
    st.tail_trim_dt = guard_dt + 300.0  # past the guard, no override -> clamped to guard_dt
    res = compute_all(st, td)
    dt_all = td.t_s - res.t_shutin_s
    n_beyond = int(np.sum(dt_all >= guard_dt))
    want = f"Tail trimmed at {guard_dt/60:.0f} min ({n_beyond} samples excluded)"
    assert want in res.warnings


def test_tail_trimmed_warning_unchanged_for_trim_before_guard():
    td = _guarded_td()
    st = injection_state(td)
    guard_dt = compute_all(st, td).resampled_full.guard_dt
    st.tail_trim_dt = guard_dt - 120.0
    res = compute_all(st, td)
    dt_all = td.t_s - res.t_shutin_s
    n_beyond = int(np.sum(dt_all[dt_all >= 0] > st.tail_trim_dt))
    assert f"Tail trimmed at {st.tail_trim_dt/60:.0f} min ({n_beyond} samples excluded)" \
        in res.warnings


# ---- B4 / B5 ---------------------------------------------------------------------------------
def _with_contact(scenario):
    td = make_testdata()
    st = injection_state(td)
    res = compute_all(st, td)
    st.contact_G = float(res.diagnostics.G[len(res.diagnostics.G) // 2])
    st.closure_scenario = scenario
    return compute_all(st, td)


def test_blank_scenario_with_contact_warns():
    res = _with_contact("")
    assert res.shmin_compliance is not None
    msg = "No closure scenario chosen; compliance Shmin uses the seeded contact"
    assert msg in res.warnings and len(msg) <= 90


def test_chosen_scenario_does_not_warn():
    res = _with_contact("C-A clear")
    assert not any("No closure scenario chosen" in w for w in res.warnings)


def test_inverted_injection_window_warns():
    td = make_testdata()
    st = injection_state(td)
    st.start_idx, st.shutin_idx = 300, 100
    res = compute_all(st, td)
    assert res.te_s is None
    msg = "Injection start is at or after shut-in; re-pick the injection window"
    assert msg in res.warnings and len(msg) <= 90


def test_valid_injection_window_no_inverted_warning():
    td = make_testdata()
    res = compute_all(injection_state(td), td)
    assert not any("at or after shut-in" in w for w in res.warnings)
