"""PickState mask/keep intervals and compute_all's combined dropout mask (see
resample.detect_rise_excursions and tests/test_rise_excursions.py for the detector itself)."""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest

from dfit_tool import store
from dfit_tool.io_load import TestData as IoTestData
from dfit_tool.model import PickState, _decode, compute_all
from tests.helpers import PRESSURE_COL

_START_IDX, _SHUTIN_IDX = 50, 100
_POST = 6000   # post-shut-in samples, 1 Hz; linear 5000 -> 500 psi (0.75 psi/s)
BUMP_LO, BUMP_HI = 2000, 2300   # post-shut-in sample offsets of the bump [lo, hi)


def _fixture(bump: bool = True, false_low_at: int | None = None):
    n = _SHUTIN_IDX + _POST
    t_s = np.arange(n, dtype=float)
    rate = np.zeros(n)
    rate[_START_IDX:_SHUTIN_IDX] = 5.0
    pressure = np.full(n, 2000.0)
    pressure[_START_IDX:_SHUTIN_IDX] = np.linspace(2000.0, 5000.0, _SHUTIN_IDX - _START_IDX)
    pressure[_SHUTIN_IDX:] = np.linspace(5000.0, 500.0, _POST)
    if bump:
        pressure[_SHUTIN_IDX + BUMP_LO:_SHUTIN_IDX + BUMP_HI] += 300.0
    if false_low_at is not None:
        pressure[_SHUTIN_IDX + false_low_at] -= 1500.0
    df = pd.DataFrame({PRESSURE_COL: pressure, "RATE": rate})
    td = IoTestData(path="<synthetic>", df=df, datetime_col="DATETIME", t_s=t_s,
                    columns=list(df.columns))
    st = PickState(pressure_col=PRESSURE_COL, rate_col="RATE", start_idx=_START_IDX,
                   shutin_idx=_SHUTIN_IDX)
    return td, st


def _abs_t(post_idx: int) -> float:
    return float(_SHUTIN_IDX + post_idx)


# ---- compute_all: auto detection ------------------------------------------------------------------
def test_returning_bump_is_masked_and_resampling_continues():
    td, st = _fixture()
    raw_before = td.df[PRESSURE_COL].to_numpy().copy()
    res = compute_all(st, td)
    assert res.resampled_full.guard_dt is None
    assert len(res.rise_excursions) == 1
    e = res.rise_excursions[0]
    assert e.dt_start == BUMP_LO and e.n_samples == BUMP_HI - BUMP_LO
    want = (f"Pressure rise masked at 33 min (5 min, "
            f"+{e.p_max - e.base:.0f} psi)")
    assert res.warnings[0] == want
    assert len(want) <= 90
    m = res.dropout_mask
    assert np.flatnonzero(m).tolist() == list(range(_SHUTIN_IDX + BUMP_LO,
                                                    _SHUTIN_IDX + BUMP_HI))
    assert res.resampled_full.dt.max() > BUMP_HI + 1000
    np.testing.assert_array_equal(res.bhp_all, raw_before)   # never mutated
    assert res.n_manual_masked == 0 and res.n_keep_restored == 0


def test_two_rises_warning():
    td, st = _fixture()
    td.df.loc[_SHUTIN_IDX + 4000:_SHUTIN_IDX + 4299, PRESSURE_COL] += 300.0
    res = compute_all(st, td)
    assert len(res.rise_excursions) == 2
    assert res.warnings[0] == "2 pressure rises masked, first at 33 min"


def test_keep_interval_restores_guard_and_warns():
    td, st = _fixture()
    st.keep_intervals = [(_abs_t(BUMP_LO), _abs_t(BUMP_HI - 1))]
    res = compute_all(st, td)
    assert res.resampled_full.guard_dt is not None
    assert not res.dropout_mask.any()
    assert res.n_keep_restored == BUMP_HI - BUMP_LO
    assert "Manual keep: 300 auto-masked samples restored" in res.warnings
    assert all(len(w) <= 90 for w in res.warnings)
    # A fully kept rise is not reported as masked anywhere (warning, event list, log count).
    assert res.rise_excursions == []
    assert not any("rise masked" in w or "rises masked" in w for w in res.warnings)


def test_partially_kept_rise_is_still_reported():
    td, st = _fixture()
    st.keep_intervals = [(_abs_t(BUMP_LO), _abs_t(BUMP_LO + 100))]
    res = compute_all(st, td)
    assert len(res.rise_excursions) == 1
    assert any(w.startswith("Pressure rise masked at 33 min") for w in res.warnings)
    assert res.resampled_full.guard_dt is not None   # the kept part of the bump fires the guard


def test_keep_over_dip_only_still_reports_masked_lead_in():
    # detect_dropouts also masks a lead-in slide before the dip; a keep over just the dip leaves
    # those samples masked, so the dropout must still be reported (never a silent mask).
    td, st = _fixture(bump=False)
    i = _SHUTIN_IDX + 900
    level = float(td.df.loc[i - 4, PRESSURE_COL])
    td.df.loc[i - 3:i - 1, PRESSURE_COL] = [level - 700.0, level - 1500.0, level - 3000.0]
    td.df.loc[i, PRESSURE_COL] = 5.0
    res = compute_all(st, td)
    assert np.flatnonzero(res.dropout_mask).tolist() == [i - 3, i - 2, i - 1, i]   # sanity
    st.keep_intervals = [(float(i) - 0.25, float(i) + 0.25)]
    res = compute_all(st, td)
    assert np.flatnonzero(res.dropout_mask).tolist() == [i - 3, i - 2, i - 1]
    assert len(res.dropouts) == 1
    assert any("dropout masked" in w for w in res.warnings)
    assert "Manual keep: 1 auto-masked sample restored" in res.warnings


def test_fully_kept_dropout_is_not_reported():
    td, st = _fixture(bump=False)
    i = _SHUTIN_IDX + 3000
    td.df.loc[i:i + 2, PRESSURE_COL] = 5.0
    res = compute_all(st, td)
    assert len(res.dropouts) == 1   # sanity: the detector fires without a keep
    st.keep_intervals = [(float(i - 5), float(i + 5))]
    res = compute_all(st, td)
    assert res.dropouts == []
    assert not any("dropout" in w for w in res.warnings)
    assert res.n_keep_restored == 3


def test_manual_mask_interval_masks_counts_and_warns():
    td, st = _fixture(bump=False)
    st.mask_intervals = [(_abs_t(1000), _abs_t(1009)), (_abs_t(3000), _abs_t(3004)),
                         (5.0, 20.0)]   # the last is entirely pre-shut-in
    res = compute_all(st, td)
    assert res.n_manual_masked == 15
    assert res.dropout_mask[_SHUTIN_IDX + 1000:_SHUTIN_IDX + 1010].all()
    assert res.dropout_mask.sum() == 15
    assert not res.dropout_mask[:_SHUTIN_IDX].any()
    assert res.warnings[0] == "Manual mask: 15 samples in 2 intervals"


def test_manual_mask_single_interval_is_singular():
    td, st = _fixture(bump=False)
    st.mask_intervals = [(_abs_t(1000), _abs_t(1009))]
    res = compute_all(st, td)
    assert res.warnings[0] == "Manual mask: 10 samples in 1 interval"


def test_manual_mask_over_false_low_lets_later_bump_be_masked():
    td, st = _fixture(false_low_at=300)
    res = compute_all(st, td)
    # The false low pins the running min ~1500 psi down: the "excursion" is the whole decline
    # after it, far too long to be a glitch, so the bump is not masked and the guard fires.
    assert res.rise_excursions == []
    assert res.resampled_full.guard_dt is not None
    st.mask_intervals = [(_abs_t(300) - 0.5, _abs_t(300) + 0.5)]
    res2 = compute_all(st, td)
    assert len(res2.rise_excursions) == 1
    assert res2.rise_excursions[0].dt_start == BUMP_LO
    assert res2.resampled_full.guard_dt is None
    assert res2.n_manual_masked == 1


def test_no_intervals_no_extra_warnings_on_clean_record():
    td, st = _fixture(bump=False)
    res = compute_all(st, td)
    assert res.rise_excursions == []
    assert not res.dropout_mask.any()
    assert not any("mask" in w.lower() or "keep" in w.lower() for w in res.warnings)


# ---- decode / persistence -------------------------------------------------------------------------
def test_old_save_lacks_keys():
    st = _decode({"pressure_col": "P"})
    assert st.mask_intervals == [] and st.keep_intervals == []


@pytest.mark.parametrize("bad", [
    None, "abc", 5, {"a": 1},
])
def test_non_list_value_becomes_empty(bad):
    st = _decode({"mask_intervals": bad, "keep_intervals": bad})
    assert st.mask_intervals == [] and st.keep_intervals == []


def test_malformed_entries_are_dropped():
    raw = [[1, 2], None, [1, 2, 3], [float("nan"), 5], [0, float("inf")], [5, 5], [9, 3],
           ["a", "b"], "ab", [1], 7, [True, 3.5], [10.5, 20]]
    st = _decode({"mask_intervals": raw, "keep_intervals": [[4, 8], [3]]})
    assert st.mask_intervals == [(1.0, 2.0), (10.5, 20.0)]   # bool is not a number
    assert all(isinstance(iv, tuple) and len(iv) == 2 and all(isinstance(x, float) for x in iv)
               for iv in st.mask_intervals)
    assert all(math.isfinite(a) and math.isfinite(b) and a < b for a, b in st.mask_intervals)
    assert st.keep_intervals == [(4.0, 8.0)]


def test_json_round_trip_preserves_tuples(tmp_path):
    st = PickState(mask_intervals=[(1.5, 2.5), (10.0, 20.0)], keep_intervals=[(3.0, 4.0)])
    path = str(tmp_path / "p.json")
    st.to_json(path)
    json.load(open(path))   # plain JSON lists on disk
    back = PickState.from_json(path)
    assert back.mask_intervals == [(1.5, 2.5), (10.0, 20.0)]
    assert back.keep_intervals == [(3.0, 4.0)]
    assert all(isinstance(iv, tuple) for iv in back.mask_intervals + back.keep_intervals)


# ---- log ------------------------------------------------------------------------------------------
def test_log_columns_are_last_four_and_filled(tmp_path):
    assert store.LOG_COLUMNS[-6:-2] == ["dropouts_masked", "rises_masked", "manual_masks",
                                      "manual_keeps"]
    td, st = _fixture()
    st.mask_intervals = [(_abs_t(10), _abs_t(12)), (_abs_t(20), _abs_t(22))]
    st.keep_intervals = [(_abs_t(30), _abs_t(32))]
    res = compute_all(st, td)
    entry = store.TestEntry(test_id="w", folder=str(tmp_path))
    row = store.build_log_row(entry, str(tmp_path / "w.csv"), str(tmp_path), st, td, res)
    assert list(row.keys()) == store.LOG_COLUMNS
    assert row["dropouts_masked"] == 0
    assert row["rises_masked"] == 1
    assert row["manual_masks"] == 2
    assert row["manual_keeps"] == 1


def test_summary_rows():
    from dfit_tool import summary
    td, st = _fixture()
    st.mask_intervals = [(_abs_t(10), _abs_t(12))]
    res = compute_all(st, td)
    rows = {r[0]: r[1] for s in summary.summary_sections(st, res) for r in s.rows}
    assert rows["rises masked"] == "1"
    assert rows["manual masks"] == "1 mask, 0 keep"
