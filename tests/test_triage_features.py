"""Unit tests for scripts/triage/features.py (extract, signature, scan_folders, save/load_scan),
scripts/triage/basins.py (basin_for), and scripts/triage/figure.py (page_count, render_grid,
render_file_png's error panel). Headless (tests/conftest.py forces Agg) and never touches
`C:\\DFIT Data` or any real corpus file -- every fixture here is a small synthetic CSV written to
`tmp_path`.

`scripts/` is not a package (mirrors tests/test_well_locations.py's fixup), so the repo root and
the scripts directory are added to sys.path here.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCRIPTS_DIR = os.path.join(_REPO_ROOT, "scripts")
for _p in (_REPO_ROOT, _SCRIPTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from triage import basins, features, figure  # noqa: E402


# --------------------------------------------------------------------------------------------------
# synthetic CSV builders
# --------------------------------------------------------------------------------------------------
def _dfit_arrays(
    n: int = 4000, dt: float = 60.0, start_idx: int = 100, shutin_idx: int = 300,
    inj_peak: float = 5000.0, decline_amount: float = 1500.0, decay_s: float = 6000.0,
    flat: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A DFIT-shaped (t_s, pressure, rate): rate ramps up then down between start_idx/shutin_idx,
    pressure rises during injection then decays exponentially to a plateau after shut-in.
    `flat=True` shrinks the post-shut-in decline to a few psi (the "flat" verdict case)."""
    t_s = np.arange(n, dtype=float) * dt
    rate = np.zeros(n)
    half = (shutin_idx - start_idx) // 2
    rate[start_idx:start_idx + half] = np.linspace(0.0, 8.0, half)
    rate[start_idx + half:shutin_idx] = np.linspace(8.0, 0.0, shutin_idx - start_idx - half)

    pressure = np.full(n, 2000.0)
    pressure[start_idx:shutin_idx] = 2000.0 + (inj_peak - 2000.0) * np.linspace(
        0.0, 1.0, shutin_idx - start_idx
    )
    post = n - shutin_idx
    decline_t = np.arange(post, dtype=float) * dt
    amount = 5.0 if flat else decline_amount
    pressure[shutin_idx:] = inj_peak - amount * (1.0 - np.exp(-decline_t / decay_s))
    return t_s, pressure, rate


def _write_csv(path, t_s, pressure, rate=None, datetime_col="DateTime",
               pressure_col="Pressure (psi)", rate_col="Rate (bpm)"):
    start = pd.Timestamp("2020-01-01")
    dt = start + pd.to_timedelta(t_s, unit="s")
    data = {datetime_col: dt.strftime("%m/%d/%Y %H:%M:%S"), pressure_col: pressure}
    if rate is not None:
        data[rate_col] = rate
    pd.DataFrame(data).to_csv(path, index=False)


# --------------------------------------------------------------------------------------------------
# extract: verdicts
# --------------------------------------------------------------------------------------------------
def test_extract_likely_dfit(tmp_path):
    path = tmp_path / "well1.csv"
    t_s, p, r = _dfit_arrays()
    _write_csv(path, t_s, p, rate=r)

    feat = features.extract(str(path))

    assert feat.verdict == "likely_dfit"
    assert feat.load_error == ""
    assert feat.rows is not None and feat.rows >= 50
    assert feat.post_shutin_hr >= 4.0
    assert feat.drop >= 200.0
    assert feat.decline_fraction >= 0.6


def test_extract_flat(tmp_path):
    path = tmp_path / "flat.csv"
    t_s, p, r = _dfit_arrays(flat=True)
    _write_csv(path, t_s, p, rate=r)

    feat = features.extract(str(path))

    assert feat.verdict == "flat"
    assert feat.load_error == ""
    assert feat.drop is not None and feat.drop < 50.0


def test_extract_short_falloff_big_drop_but_short_post_shutin_span(tmp_path):
    """Regression for FIX 2: a file with a big drop (well above the flat threshold) but only a
    couple hours of post-shut-in falloff must be labelled `short_falloff`, not the old
    catch-all `flat` (which the panel annotation, printing the real drop, would contradict)."""
    path = tmp_path / "short_falloff.csv"
    t_s, p, r = _dfit_arrays(n=400, start_idx=100, shutin_idx=300, decay_s=6000.0)
    _write_csv(path, t_s, p, rate=r)

    feat = features.extract(str(path))

    assert feat.duration_hr is not None and feat.duration_hr >= 1.0
    assert feat.drop is not None and feat.drop >= 50.0
    assert feat.post_shutin_hr is not None and feat.post_shutin_hr < 4.0
    assert feat.verdict == "short_falloff"


def test_extract_noisy_verdict_low_decline_fraction(tmp_path):
    """A long, big-drop record whose post-shut-in samples are dominated by noise (more than 40%
    of consecutive diffs are positive) gets `noisy`, distinct from `flat`/`short_falloff`."""
    path = tmp_path / "noisy.csv"
    n, dt = 4000, 60.0
    start_idx, shutin_idx = 100, 300
    t_s = np.arange(n, dtype=float) * dt
    rate = np.zeros(n)
    half = (shutin_idx - start_idx) // 2
    rate[start_idx:start_idx + half] = np.linspace(0.0, 8.0, half)
    rate[start_idx + half:shutin_idx] = np.linspace(8.0, 0.0, shutin_idx - start_idx - half)

    pressure = np.full(n, 2000.0)
    pressure[start_idx:shutin_idx] = 2000.0 + 3000.0 * np.linspace(
        0.0, 1.0, shutin_idx - start_idx
    )
    post = n - shutin_idx
    rng = np.random.default_rng(0)
    trend = np.linspace(5000.0, 3000.0, post)
    noise = rng.normal(scale=400.0, size=post)
    pressure[shutin_idx:] = trend + noise
    _write_csv(path, t_s, pressure, rate=rate)

    feat = features.extract(str(path))

    assert feat.duration_hr is not None and feat.duration_hr >= 1.0
    assert feat.drop is not None and feat.drop >= 50.0
    assert feat.post_shutin_hr is not None and feat.post_shutin_hr >= 4.0
    assert feat.decline_fraction is not None and feat.decline_fraction < 0.6
    assert feat.verdict == "noisy"


def test_extract_too_short(tmp_path):
    path = tmp_path / "short.csv"
    # 100 samples * 30s = 3000s = 50 min < 1 hour, but >= 50 finite samples so it isn't no_pressure.
    t_s = np.arange(100, dtype=float) * 30.0
    p = np.linspace(5000.0, 4000.0, 100)
    _write_csv(path, t_s, p)

    feat = features.extract(str(path))

    assert feat.verdict == "too_short"
    assert feat.load_error == ""
    assert feat.duration_hr < 1.0


def test_extract_no_pressure_no_recognizable_column(tmp_path):
    path = tmp_path / "unrecognizable.csv"
    start = pd.Timestamp("2020-01-01")
    t_s = np.arange(100, dtype=float) * 10.0
    dt = start + pd.to_timedelta(t_s, unit="s")
    # "Time" is recognized as the datetime column (so the load itself succeeds); "Foo"/"Bar" are
    # not recognizable as pressure or rate.
    df = pd.DataFrame({
        "Time": dt.strftime("%m/%d/%Y %H:%M:%S"),
        "Foo": np.random.default_rng(0).normal(size=100),
        "Bar": np.arange(100),
    })
    df.to_csv(path, index=False)

    feat = features.extract(str(path))

    assert feat.verdict == "no_pressure"
    assert feat.load_error == ""


def test_extract_load_error_on_corrupt_file(tmp_path):
    path = tmp_path / "corrupt.csv"
    # Invalid UTF-8 byte sequences that pd.read_csv(encoding="utf-8-sig") cannot decode.
    with open(path, "wb") as fh:
        fh.write(bytes([0xFF, 0xFE, 0x00, 0x01, 0x80, 0x81]) * 200)

    feat = features.extract(str(path))

    assert feat.verdict == "load_error"
    assert feat.load_error != ""
    assert ":" in feat.load_error  # "<ExceptionType>: <message>"


def test_extract_never_raises_on_missing_file(tmp_path):
    """A path that doesn't exist at all must also never raise -- covers the case a folder scan
    races a file being moved/deleted mid-walk."""
    feat = features.extract(str(tmp_path / "does_not_exist.csv"))
    assert feat.verdict == "load_error"
    assert feat.load_error != ""


# --------------------------------------------------------------------------------------------------
# extract: shut-in fallback
# --------------------------------------------------------------------------------------------------
def test_extract_shutin_source_rate_when_rate_column_pumps(tmp_path):
    path = tmp_path / "with_rate.csv"
    t_s, p, r = _dfit_arrays()
    _write_csv(path, t_s, p, rate=r)

    feat = features.extract(str(path))

    assert feat.shutin_source == "rate"
    assert feat.injection_min is not None


def test_extract_shutin_source_pressure_peak_when_gauge_only(tmp_path):
    path = tmp_path / "gauge_only.csv"
    t_s, p, _r = _dfit_arrays()
    _write_csv(path, t_s, p, rate=None)  # no rate column at all

    feat = features.extract(str(path))

    assert feat.shutin_source == "pressure-peak"
    assert feat.injection_min is None


# --------------------------------------------------------------------------------------------------
# extract: unit_guess
# --------------------------------------------------------------------------------------------------
def test_extract_unit_guess_psi_near_10000(tmp_path):
    path = tmp_path / "psi.csv"
    t_s, p, r = _dfit_arrays(inj_peak=10000.0, decline_amount=3000.0)
    _write_csv(path, t_s, p, rate=r)

    feat = features.extract(str(path))
    assert feat.unit_guess == "psi"


def test_extract_unit_guess_kpa_near_100000(tmp_path):
    path = tmp_path / "kpa.csv"
    t_s, p, r = _dfit_arrays(inj_peak=100000.0, decline_amount=30000.0)
    _write_csv(path, t_s, p, rate=r)

    feat = features.extract(str(path))
    assert feat.unit_guess == "kPa"


def test_extract_kpa_record_with_1500_kpa_drop_still_reaches_likely_dfit(tmp_path):
    """The likely_dfit drop threshold is 200 psi, scaled to ~1379 kPa when unit_guess is "kPa" --
    not the raw 200. A record whose peak reads in kPa (unit_guess == "kPa") and whose actual drop
    is ~1500 kPa must still clear that scaled bar. Gauge-only (no rate column) so the shut-in
    instant lands exactly at the array's own pressure peak (pressure-peak fallback), giving an
    exact, known drop rather than one perturbed by the rate-detected window's edge."""
    path = tmp_path / "kpa_dfit.csv"
    t_s, p, _r = _dfit_arrays(inj_peak=100000.0, decline_amount=1500.0)
    _write_csv(path, t_s, p, rate=None)

    feat = features.extract(str(path))

    assert feat.unit_guess == "kPa"
    assert feat.drop is not None and 1400.0 <= feat.drop <= 1600.0
    assert feat.verdict == "likely_dfit"


# --------------------------------------------------------------------------------------------------
# monotonic_prefix
# --------------------------------------------------------------------------------------------------
def test_monotonic_prefix_all_monotonic_is_unchanged():
    t_s = np.arange(10, dtype=float) * 60.0
    assert features.monotonic_prefix(t_s) == slice(0, 10)


def test_monotonic_prefix_trims_trailing_zeros():
    """Mimics real DBS trailing `idx == 0` padding: a clean ramp, then N zero rows."""
    ramp = np.arange(20, dtype=float) * 60.0
    t_s = np.concatenate([ramp, np.zeros(5)])
    assert features.monotonic_prefix(t_s) == slice(0, 20)


def test_monotonic_prefix_single_sample():
    assert features.monotonic_prefix(np.array([5.0])) == slice(0, 1)


def test_monotonic_prefix_empty_array():
    assert features.monotonic_prefix(np.array([])) == slice(0, 0)


def test_monotonic_prefix_all_zero_is_unchanged():
    t_s = np.zeros(6)
    assert features.monotonic_prefix(t_s) == slice(0, 6)


def test_monotonic_prefix_empty_after_slicing():
    """`monotonic_prefix` returns a `slice`, so callers must be able to index with it even on an
    empty array without raising."""
    t_s = np.array([])
    result = t_s[features.monotonic_prefix(t_s)]
    assert result.size == 0


def test_monotonic_prefix_interior_dip_truncates_at_onset_not_global_max(tmp_path):
    """FIX 6: the OLD "last index equal to the running max" shortcut let a dip that recovers
    below the eventual global max pass straight through untouched -- `[0, 60, 120, 0, 0, 180,
    240]` kept its full length because 240 is still the running max at the end. The genuine
    longest-non-decreasing-prefix contract must truncate right where the first decrease happens,
    at the sample before `t_s` drops from 120 back to 0."""
    t_s = np.array([0.0, 60.0, 120.0, 0.0, 0.0, 180.0, 240.0])
    assert features.monotonic_prefix(t_s) == slice(0, 3)


def test_monotonic_prefix_strictly_decreasing_collapses_but_stays_nondecreasing():
    """FIX 1 (third review round): an earlier draft special-cased a mostly/overwhelmingly
    decreasing `t_s` as a reversed export and returned the reversed view. That branch was
    removed rather than hardened further -- a completed scan of the real corpus (402 folders /
    1,237 files) found zero files with a negative `duration_hr`/`post_shutin_hr`, so detecting
    reversal here is a theoretical concern, not an observed one, and `verdict` only orders/
    pre-highlights panels for a human reviewer rather than gating anything. A strictly
    decreasing `t_s` therefore collapses to its 1-sample prefix, same as any other non-monotonic
    record -- and the result is still trivially non-decreasing and its span non-negative."""
    t_s = np.array([500.0, 400.0, 300.0, 200.0, 100.0])
    mono = features.monotonic_prefix(t_s)
    result = t_s[mono]
    assert list(result) == [500.0]
    assert np.all(np.diff(result) >= 0)
    assert (result[-1] - result[0]) >= 0.0


def test_monotonic_prefix_pairwise_swapped_view_stays_nondecreasing():
    """A scrambled merge of two out-of-order channel streams (adjacent rows swapped throughout,
    ~50% of steps decreasing) must not raise and must yield a non-decreasing view -- whatever
    length that truncates to. No assertion about which verdict this produces: this module no
    longer tries to classify a scrambled record as anything special, it is truncated the same as
    any other non-monotonic input."""
    t_s = np.arange(200, dtype=float) * 60.0
    swapped = t_s.copy()
    swapped[0::2], swapped[1::2] = t_s[1::2], t_s[0::2]
    mono = features.monotonic_prefix(swapped)
    result = swapped[mono]
    assert np.all(np.diff(result) >= 0)
    assert (result[-1] - result[0]) >= 0.0


# --------------------------------------------------------------------------------------------------
# extract: trailing DBS-style padding (FIX 1 regression)
# --------------------------------------------------------------------------------------------------
def test_extract_trailing_padding_does_not_corrupt_duration_or_verdict(tmp_path):
    """The regression test that matters most: real DBS files carry a block of trailing
    `idx == 0` padding rows, so `t_s` is not monotonic and its last element is 0.0. Naively
    trusting `t_s[-1]` truncates `duration_hr` to ~0 and makes `post_shutin_hr` negative,
    mislabeling a genuine 205-hour-shaped DFIT as `too_short`. Confirmed (see task notes) to FAIL
    against the old `t_s[-1]` logic before the fix."""
    path = tmp_path / "padded.csv"
    t_s, p, r = _dfit_arrays()
    n_pad = 104
    t_s_padded = np.concatenate([t_s, np.zeros(n_pad)])
    p_padded = np.concatenate([p, np.full(n_pad, p[-1])])
    r_padded = np.concatenate([r, np.zeros(n_pad)])
    _write_csv(path, t_s_padded, p_padded, rate=r_padded)

    feat = features.extract(str(path))

    assert feat.trailing_dropped == n_pad
    assert feat.verdict == "likely_dfit"
    assert feat.duration_hr is not None and feat.duration_hr == pytest.approx(
        t_s[-1] / 3600.0, rel=1e-6
    )
    assert feat.post_shutin_hr is not None and feat.post_shutin_hr > 0


def test_extract_trailing_padding_pressure_excluded_from_drop_and_decline(tmp_path):
    """The padding rows' pressure values must never leak into `drop`/`decline_fraction` -- a
    padded file with garbage rebound pressure in its dropped tail must measure identically to
    the same record with no padding at all."""
    t_s, p, r = _dfit_arrays()
    clean_path = tmp_path / "clean.csv"
    _write_csv(clean_path, t_s, p, rate=r)
    feat_clean = features.extract(str(clean_path))

    n_pad = 50
    t_s_padded = np.concatenate([t_s, np.zeros(n_pad)])
    # Garbage rebound in the padding block -- would corrupt drop/decline_fraction if included.
    p_padded = np.concatenate([p, np.full(n_pad, 50000.0)])
    r_padded = np.concatenate([r, np.zeros(n_pad)])
    padded_path = tmp_path / "padded.csv"
    _write_csv(padded_path, t_s_padded, p_padded, rate=r_padded)
    feat_padded = features.extract(str(padded_path))

    assert feat_padded.trailing_dropped == n_pad
    assert feat_padded.duration_hr == pytest.approx(feat_clean.duration_hr)
    assert feat_padded.post_shutin_hr == pytest.approx(feat_clean.post_shutin_hr)
    assert feat_padded.drop == pytest.approx(feat_clean.drop)
    assert feat_padded.decline_fraction == pytest.approx(feat_clean.decline_fraction)
    assert feat_padded.verdict == feat_clean.verdict == "likely_dfit"


def test_extract_no_padding_trailing_dropped_is_zero(tmp_path):
    path = tmp_path / "unpadded.csv"
    t_s, p, r = _dfit_arrays()
    _write_csv(path, t_s, p, rate=r)

    feat = features.extract(str(path))
    assert feat.trailing_dropped == 0


# --------------------------------------------------------------------------------------------------
# extract: NaN timestamps (FIX 1 regression)
# --------------------------------------------------------------------------------------------------
def _write_csv_with_bad_datetime_rows(path, t_s, pressure, rate, bad_indices):
    """Same shape as `_write_csv`, but the given row indices get an unparseable datetime string
    instead of a real one -- `io_load.parse_datetime` coerces those to NaT (so `t_s` is NaN
    there), mirroring the real corpus failure mode (a corrupt/garbled timestamp cell), rather
    than a NaN *pressure* value, which is a different code path entirely."""
    start = pd.Timestamp("2020-01-01")
    dt = start + pd.to_timedelta(t_s, unit="s")
    dt_strings = list(dt.strftime("%m/%d/%Y %H:%M:%S"))
    for i in bad_indices:
        dt_strings[i] = "not-a-timestamp"
    data = {"DateTime": dt_strings, "Pressure (psi)": pressure, "Rate (bpm)": rate}
    pd.DataFrame(data).to_csv(path, index=False)


def test_extract_nan_timestamp_first_row_does_not_raise(tmp_path):
    """The FIX 1 regression: `np.nonzero(t_s == running_max)[0]` is empty whenever `t_s[0]` is
    NaN (NaN never equals its own running max), so the old `monotonic_prefix` raised
    `ValueError: zero-size array to reduction operation maximum which has no identity`, and
    `extract`'s catch-all turned that into `verdict="load_error"` -- unreviewable, even though
    the rest of the record is perfectly good data."""
    path = tmp_path / "nan_first.csv"
    t_s, p, r = _dfit_arrays()
    _write_csv_with_bad_datetime_rows(path, t_s, p, r, bad_indices=[0])

    feat = features.extract(str(path))

    assert feat.verdict != "load_error"
    assert feat.load_error == ""
    assert feat.verdict == "likely_dfit"
    assert feat.duration_hr is not None and feat.duration_hr > 1.0


def test_extract_nan_timestamp_middle_row_does_not_truncate_record(tmp_path):
    """A single unparseable timestamp partway through a long record must not silently truncate
    everything after it -- the old bug: one bad row at hour 3 of a 200-hour record gave
    `duration_hr ~= 3` and a `too_short`/`short_falloff` verdict with no error surfaced."""
    t_s, p, r = _dfit_arrays()
    clean_path = tmp_path / "clean.csv"
    _write_csv(clean_path, t_s, p, rate=r)
    feat_clean = features.extract(str(clean_path))

    mid = len(t_s) // 2
    bad_path = tmp_path / "nan_middle.csv"
    _write_csv_with_bad_datetime_rows(bad_path, t_s, p, r, bad_indices=[mid])
    feat_bad = features.extract(str(bad_path))

    assert feat_bad.verdict != "load_error"
    assert feat_bad.load_error == ""
    assert feat_bad.duration_hr == pytest.approx(feat_clean.duration_hr, rel=1e-3)
    assert feat_bad.verdict == feat_clean.verdict == "likely_dfit"


def test_extract_nan_timestamp_scattered_does_not_raise_or_truncate(tmp_path):
    """Several unparseable timestamps scattered throughout (not clustered at either end) must
    all be dropped as missing data, not just the first one encountered."""
    t_s, p, r = _dfit_arrays()
    bad_indices = [0, 5, 200, 1500, 3000, len(t_s) - 1]
    path = tmp_path / "nan_scattered.csv"
    _write_csv_with_bad_datetime_rows(path, t_s, p, r, bad_indices=bad_indices)

    feat = features.extract(str(path))

    assert feat.verdict != "load_error"
    assert feat.load_error == ""
    assert feat.verdict == "likely_dfit"
    assert feat.duration_hr is not None and feat.duration_hr > 1.0
    assert feat.post_shutin_hr is not None and feat.post_shutin_hr > 0


def test_extract_nan_timestamp_dropped_count_is_surfaced(tmp_path):
    """FIX 4: the NaN-timestamp drop (distinct from monotonic-prefix truncation, `trailing_
    dropped`) must be counted and exposed on `FileFeatures`, not silently absorbed -- a file
    whose datetime column is largely unparseable must not report confident numbers with nothing
    on screen saying a chunk of it was dropped."""
    t_s, p, r = _dfit_arrays()
    bad_indices = [0, 5, 200, 1500, 3000]
    path = tmp_path / "nan_scattered_count.csv"
    _write_csv_with_bad_datetime_rows(path, t_s, p, r, bad_indices=bad_indices)

    feat = features.extract(str(path))

    assert feat.nonfinite_time_dropped == len(bad_indices)


def test_extract_no_bad_timestamps_nonfinite_time_dropped_is_zero(tmp_path):
    path = tmp_path / "clean.csv"
    t_s, p, r = _dfit_arrays()
    _write_csv(path, t_s, p, rate=r)

    feat = features.extract(str(path))

    assert feat.nonfinite_time_dropped == 0


# --------------------------------------------------------------------------------------------------
# extract: out-of-order t_s (FIX 1, third review round -- non-negative duration guardrail)
# --------------------------------------------------------------------------------------------------
# `monotonic_prefix` no longer special-cases a mostly/overwhelmingly decreasing `t_s` as a
# reversed export (see its docstring) -- these two shapes just confirm `extract`'s defensive
# clamp holds regardless: never raises, and `duration_hr`/`post_shutin_hr` are never negative,
# no matter how short the truncated prefix ends up being. No assertion about which `verdict`
# results -- this module doesn't try to classify these shapes as anything special.
def test_extract_strictly_reverse_chronological_never_negative(tmp_path):
    path = tmp_path / "reversed.csv"
    t_s, p, r = _dfit_arrays()
    _write_csv(path, t_s[::-1].copy(), p, rate=r)  # timestamps strictly decreasing

    feat = features.extract(str(path))

    assert feat.verdict != "load_error"
    assert feat.duration_hr is not None and feat.duration_hr >= 0.0
    assert feat.post_shutin_hr is None or feat.post_shutin_hr >= 0.0


def test_extract_pairwise_swapped_never_negative(tmp_path):
    """A scrambled merge of two out-of-order channel streams: every adjacent pair of rows is
    swapped throughout, leaving ~50% of steps decreasing."""
    path = tmp_path / "swapped.csv"
    t_s, p, r = _dfit_arrays()
    swapped = t_s.copy()
    swapped[0::2], swapped[1::2] = t_s[1::2], t_s[0::2]
    _write_csv(path, swapped, p, rate=r)

    feat = features.extract(str(path))

    assert feat.verdict != "load_error"
    assert feat.duration_hr is not None and feat.duration_hr >= 0.0
    assert feat.post_shutin_hr is None or feat.post_shutin_hr >= 0.0


# --------------------------------------------------------------------------------------------------
# signature
# --------------------------------------------------------------------------------------------------
def test_signature_identical_files_match(tmp_path):
    content = os.urandom(600_000)  # > _SIG_TAIL_MIN_SIZE, exercises the head+tail path
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(content)
    b.write_bytes(content)

    assert features.signature(str(a)) == features.signature(str(b))


def test_signature_small_identical_files_match(tmp_path):
    content = b"hello dfit" * 100  # well under _SIG_TAIL_MIN_SIZE, head-only path
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(content)
    b.write_bytes(content)

    assert features.signature(str(a)) == features.signature(str(b))


def test_signature_different_size_never_collides(tmp_path):
    a = tmp_path / "a.bin"
    b = tmp_path / "b.bin"
    a.write_bytes(os.urandom(1000))
    b.write_bytes(os.urandom(2000))

    assert features.signature(str(a)) != features.signature(str(b))


# --------------------------------------------------------------------------------------------------
# scan_folders: dedup
# --------------------------------------------------------------------------------------------------
def _build_two_folder_root(tmp_path):
    """FolderA/file1.csv and FolderA/file2.csv are byte-identical (in-folder duplicate).
    FolderB/file3.csv is byte-identical to both of FolderA's files, but in a different folder
    (the cross-folder case: a real corpus pattern -- the same raw export copied into two
    separate DFIT-test folders on the same well -- must not be flagged duplicate)."""
    root = tmp_path / "root"
    folder_a = root / "FolderA"
    folder_b = root / "FolderB"
    folder_a.mkdir(parents=True)
    folder_b.mkdir(parents=True)

    t_s, p, r = _dfit_arrays()
    _write_csv(folder_a / "file1.csv", t_s, p, rate=r)
    content = (folder_a / "file1.csv").read_bytes()
    (folder_a / "file2.csv").write_bytes(content)
    (folder_b / "file3.csv").write_bytes(content)

    return str(root)


def test_scan_folders_same_folder_duplicate(tmp_path):
    root = _build_two_folder_root(tmp_path)
    scans = features.scan_folders(root, require_questionnaire=False)

    folder_a = next(s for s in scans if s.rel == "FolderA")

    reps = [f for f in folder_a.files if f.dup_of is None]
    dups = [f for f in folder_a.files if f.dup_of is not None]
    assert len(reps) == 1
    assert len(dups) == 1
    assert dups[0].verdict == "duplicate"
    assert dups[0].rows is None  # never loaded/feature-extracted
    # The representative is the path that sorts first.
    assert dups[0].dup_of == min(f.path for f in folder_a.files)
    assert reps[0].path == dups[0].dup_of
    assert reps[0].rows is not None  # the representative was actually extracted


def test_scan_folders_cross_folder_not_duplicate(tmp_path):
    root = _build_two_folder_root(tmp_path)
    scans = features.scan_folders(root, require_questionnaire=False)

    folder_b = next(s for s in scans if s.rel == "FolderB")
    assert len(folder_b.files) == 1
    feat_b = folder_b.files[0]

    assert feat_b.verdict != "duplicate"
    assert feat_b.dup_of is None
    assert feat_b.rows is not None  # real features, not skipped

    folder_a = next(s for s in scans if s.rel == "FolderA")
    rep_a = next(f for f in folder_a.files if f.dup_of is None)
    assert rep_a.rows is not None

    file3_path = os.path.join(root, "FolderB", "file3.csv")
    file1_path = os.path.join(root, "FolderA", "file1.csv")
    file2_path = os.path.join(root, "FolderA", "file2.csv")

    # feat_b (FolderB) lists FolderA's copies (both file1 and file2, regardless of which one is
    # FolderA's in-folder representative) as same_bytes_as -- cross-folder, informational only.
    assert set(feat_b.same_bytes_as) == {file1_path, file2_path}

    # rep_a (FolderA's representative) lists FolderB's copy, but not its own in-folder duplicate
    # sibling -- that relationship is already captured by dup_of, not same_bytes_as.
    assert rep_a.same_bytes_as == [file3_path]


def test_scan_folders_caches_extraction_by_signature(tmp_path, monkeypatch):
    """The cross-folder copy must not be reloaded: io_load.load is called once for two files
    sharing a signature across different folders."""
    root = _build_two_folder_root(tmp_path)

    calls = []
    real_load = features.io_load.load

    def counting_load(path):
        calls.append(path)
        return real_load(path)

    monkeypatch.setattr(features.io_load, "load", counting_load)

    features.scan_folders(root, require_questionnaire=False)

    # All three files (FolderA's file1.csv/file2.csv and FolderB's file3.csv) share one
    # signature. file2.csv is skipped entirely as an in-folder duplicate; file1.csv (the
    # representative) triggers the one real load that populates the signature cache; file3.csv
    # (a different folder, same signature) reuses that cached result instead of reloading.
    assert len(calls) == 1


# --------------------------------------------------------------------------------------------------
# scan_folders: FIX 4 -- a file disappearing between the signature pass and the extraction pass
# --------------------------------------------------------------------------------------------------
def test_scan_folders_file_disappears_between_signature_and_extraction_passes(tmp_path, monkeypatch):
    """The signature pass and the extraction pass are minutes apart on a full corpus scan. A file
    that vanishes in that window (moved/deleted mid-walk) must yield `load_error`/`size_bytes=0`
    for just that file, not raise `FileNotFoundError` out of `scan_folders` and abort the whole
    scan (the regression: an unguarded `os.path.getsize` outside `extract`'s own try/except)."""
    root = tmp_path / "root"
    folder = root / "FolderA"
    folder.mkdir(parents=True)
    path = folder / "will_vanish.csv"
    t_s, p, r = _dfit_arrays()
    _write_csv(path, t_s, p, rate=r)

    real_signature = features.signature

    def vanishing_signature(p_):
        sig = real_signature(p_)
        if p_ == str(path):
            os.remove(p_)  # simulate the file disappearing right after it was signatured
        return sig

    monkeypatch.setattr(features, "signature", vanishing_signature)

    scans = features.scan_folders(str(root), require_questionnaire=False)

    folder_scan = next(s for s in scans if s.rel == "FolderA")
    assert len(folder_scan.files) == 1
    feat = folder_scan.files[0]
    assert feat.verdict == "load_error"
    assert feat.size_bytes == 0
    assert feat.load_error != ""


# --------------------------------------------------------------------------------------------------
# scan_folders: well-root grouping (replaces the old FIX 2 "revert to entry.folder" rule)
# --------------------------------------------------------------------------------------------------
def test_scan_folders_ambiguous_shared_customer_questionnaire_now_merges(tmp_path):
    """Superseded shape from the old `entry.folder`-only rule: a customer directory holds ONE
    unparseable (hence sentinel-named) questionnaire and two well subdirectories with no
    questionnaire of their own. Well-root grouping's anti-merge guard is a COUNT of distinct well
    names in a candidate parent's subtree, and this subtree holds exactly one name (there is only
    one questionnaire file, full stop) -- so the walk-up from either well subdirectory now merges
    them at the customer level. This is the accepted, intentional behavior change (see
    `scan_folders`' docstring): with only one, unreadable, shared questionnaire, there is no
    distinct-name evidence anywhere that would tell the two subdirectories apart, so the guard
    that exists specifically to protect DIFFERENT wells' own questionnaires from being merged
    does not (and structurally cannot) fire here. The properly-scoped regression for the real
    Bonanza Creek failure -- each well subfolder carrying its OWN distinct questionnaire -- is
    `test_scan_folders_anti_merge_regression_each_well_has_own_questionnaire` below."""
    root = tmp_path / "root"
    customer = root / "CustomerA"
    well1 = customer / "Well1"
    well2 = customer / "Well2"
    well1.mkdir(parents=True)
    well2.mkdir(parents=True)
    (customer / "well_questionnaire.xlsx").write_bytes(b"dummy")  # unparseable -> sentinel name

    t_s, p, r = _dfit_arrays(inj_peak=5000.0)
    _write_csv(well1 / "well1.csv", t_s, p, rate=r)
    content = (well1 / "well1.csv").read_bytes()
    (well2 / "well2.csv").write_bytes(content)  # byte-identical -- now an in-group duplicate

    scans = features.scan_folders(str(root), require_questionnaire=True)

    assert len(scans) == 1
    scan = scans[0]
    assert scan.rel == "CustomerA"
    assert scan.n_wells == 1
    assert len(scan.files) == 2

    reps = [f for f in scan.files if f.dup_of is None]
    dups = [f for f in scan.files if f.dup_of is not None]
    assert len(reps) == 1 and len(dups) == 1
    assert dups[0].verdict == "duplicate"


def test_scan_folders_anti_merge_regression_each_well_has_own_questionnaire(tmp_path, monkeypatch):
    """The real Bonanza Creek failure, properly scoped: a customer directory holds two well
    subdirectories, each with ITS OWN questionnaire (parsing to a DIFFERENT well name) and its
    own data. The customer directory's subtree therefore reports 2 distinct names, so the
    walk-up from either well subdirectory stops immediately -- they must never merge into one
    scan. A byte-identical file living in two different wells' subdirectories (a real corpus
    case: `Latham P-T-14HNC.DBS` / `State Antelope 34-25.DBS`) must stay a full, non-`duplicate`
    candidate in both, cross-referencing the other via `same_bytes_as` -- collapsing either one
    to `duplicate` would quarantine a different well's only file."""
    root = tmp_path / "root"
    customer = root / "CustomerA"
    well1 = customer / "Well1"
    well2 = customer / "Well2"
    well1.mkdir(parents=True)
    well2.mkdir(parents=True)
    q1 = well1 / "well1_questionnaire.xlsx"
    q2 = well2 / "well2_questionnaire.xlsx"
    q1.write_bytes(b"dummy")
    q2.write_bytes(b"dummy")

    def fake_names(paths):
        names = {str(q1): "well one", str(q2): "well two"}
        return {p: names.get(p, f"?{p}") for p in paths}

    monkeypatch.setattr(features, "_questionnaire_well_names", fake_names)

    t_s, p, r = _dfit_arrays(inj_peak=5000.0)
    _write_csv(well1 / "well1.csv", t_s, p, rate=r)
    content = (well1 / "well1.csv").read_bytes()
    (well2 / "well2.csv").write_bytes(content)  # byte-identical, but a DIFFERENT well

    scans = features.scan_folders(str(root), require_questionnaire=True)

    assert len(scans) == 2
    rels = sorted(s.rel for s in scans)
    assert rels == ["CustomerA/Well1", "CustomerA/Well2"]

    scan1 = next(s for s in scans if s.rel == "CustomerA/Well1")
    scan2 = next(s for s in scans if s.rel == "CustomerA/Well2")
    assert scan1.n_wells == 1 and scan2.n_wells == 1
    assert len(scan1.files) == 1
    assert len(scan2.files) == 1

    feat1, feat2 = scan1.files[0], scan2.files[0]
    assert feat1.verdict != "duplicate" and feat1.dup_of is None
    assert feat2.verdict != "duplicate" and feat2.dup_of is None
    assert feat1.rows is not None and feat2.rows is not None  # both fully extracted
    assert feat2.path in feat1.same_bytes_as
    assert feat1.path in feat2.same_bytes_as


def test_scan_folders_strathcona_shape_now_merges_into_one_well_root_group(tmp_path):
    """Well-root grouping's whole point: a well folder with its own questionnaire plus a nested
    `JR` subfolder (which has no questionnaire of its own) is now ONE review screen, not two --
    the fix this feature delivers for exactly the split-across-nested-directories shape the old
    `entry.folder`-only rule (reverted FIX 2) had to accept as a trade-off. Because both files
    are now in the SAME group, the byte-identical pair collapses via in-group dedup (`dup_of`)
    instead of the old cross-group `same_bytes_as` annotation."""
    root = tmp_path / "root"
    well = root / "104-07-32-062-03W6"
    jr = well / "JR"
    well.mkdir(parents=True)
    jr.mkdir(parents=True)
    (well / "well_questionnaire.xlsx").write_bytes(b"dummy")

    parent = well / "parent.csv"
    child = jr / "child.csv"
    t_s, p, r = _dfit_arrays(inj_peak=5000.0)
    _write_csv(parent, t_s, p, rate=r)
    child.write_bytes(parent.read_bytes())  # byte-identical, but a different immediate directory

    scans = features.scan_folders(str(root), require_questionnaire=True)

    assert len(scans) == 1
    scan = scans[0]
    assert scan.rel == "104-07-32-062-03W6"
    assert scan.n_wells == 1
    assert len(scan.files) == 2

    reps = [f for f in scan.files if f.dup_of is None]
    dups = [f for f in scan.files if f.dup_of is not None]
    assert len(reps) == 1 and len(dups) == 1
    assert dups[0].verdict == "duplicate"
    assert dups[0].dup_of == reps[0].path


def test_scan_folders_bkh_shape_merges_well_folder_customer_data_and_old_files(tmp_path):
    """The measured real-corpus blind spot this feature fixes: BKH HDU 9-11AH keeps its
    questionnaire in `Customer Data\\` but has data files in the well folder itself and in
    `OLD Files\\` -- neither of which has any questionnaire of its own, so the old
    `entry.folder`-only rule saw no questionnaire for either and dropped them under
    `require_questionnaire=True`. All three directories' subtrees resolve to exactly one well
    name once aggregated bottom-up, so they all land in ONE well-root group at the well folder."""
    root = tmp_path / "root"
    well = root / "BKH_HDU_9-11AH"
    customer_data = well / "Customer Data"
    old_files = well / "OLD Files"
    well.mkdir(parents=True)
    customer_data.mkdir(parents=True)
    old_files.mkdir(parents=True)
    (customer_data / "well_questionnaire.xlsx").write_bytes(b"dummy")

    t_s, p, r = _dfit_arrays(inj_peak=5000.0)
    _write_csv(well / "well_data.csv", t_s, p, rate=r)
    _write_csv(customer_data / "customer_data.csv", t_s, p, rate=r)
    _write_csv(old_files / "old_data.csv", t_s, p, rate=r)

    scans = features.scan_folders(str(root), require_questionnaire=True)

    assert len(scans) == 1
    scan = scans[0]
    assert scan.folder == str(well)
    assert scan.rel == "BKH_HDU_9-11AH"
    assert scan.n_wells == 1
    assert scan.questionnaire_path == str(customer_data / "well_questionnaire.xlsx")
    assert len(scan.files) == 3  # all three files present, none dropped


def test_scan_folders_pad_shape_wells_stay_separate_loose_file_group_is_the_pad(tmp_path, monkeypatch):
    """A pad directory (itself nested one level under the scan root, so the walk-up from a well
    subfolder genuinely has to stop AT the pad rather than being cut off by "parent == root"
    first) holds two distinct wells in subfolders, plus a loose data file directly in the pad
    itself: `Pad/WellA`/`Pad/WellB` must stay separate groups (each subtree has exactly one name,
    and the pad's own subtree -- 2 names -- stops the walk-up there), and the loose file's group
    is the pad directory itself -- kept under `require_questionnaire=True` because the pad's
    subtree (aggregating both wells) still holds >= 1 questionnaire, even though it can't tell
    which well the loose file belongs to (`n_wells == 2` flags that ambiguity)."""
    root = tmp_path / "root"
    pad = root / "Pad"
    well_a = pad / "WellA"
    well_b = pad / "WellB"
    well_a.mkdir(parents=True)
    well_b.mkdir(parents=True)
    (well_a / "well_a_questionnaire.xlsx").write_bytes(b"dummy")
    (well_b / "well_b_questionnaire.xlsx").write_bytes(b"dummy")

    t_s, p, r = _dfit_arrays(inj_peak=5000.0)
    _write_csv(well_a / "a.csv", t_s, p, rate=r)
    _write_csv(well_b / "b.csv", t_s, p, rate=r)
    _write_csv(pad / "loose.csv", t_s, p, rate=r)  # directly in the pad, belongs to neither

    def fake_names(paths):
        return {p: ("well a" if "WellA" in p else "well b") for p in paths}

    monkeypatch.setattr(features, "_questionnaire_well_names", fake_names)
    scans = features.scan_folders(str(root), require_questionnaire=True)

    assert len(scans) == 3
    by_rel = {s.rel: s for s in scans}
    assert set(by_rel) == {"Pad/WellA", "Pad/WellB", "Pad"}

    assert by_rel["Pad/WellA"].n_wells == 1
    assert by_rel["Pad/WellB"].n_wells == 1
    assert len(by_rel["Pad/WellA"].files) == 1
    assert len(by_rel["Pad/WellB"].files) == 1

    pad_scan = by_rel["Pad"]
    assert pad_scan.folder == str(pad)
    assert pad_scan.n_wells == 2
    assert len(pad_scan.files) == 1
    assert pad_scan.files[0].path == str(pad / "loose.csv")


def test_scan_folders_duplicate_questionnaire_shape_picks_shallowest(tmp_path, monkeypatch):
    """A well folder holding its own data and questionnaire, plus a SECOND questionnaire (parsing
    to the SAME well name) tucked inside `OLD Files\\` -- a real corpus pattern (an `OLD Files`
    copy, or a "Copy of ..." variant). The well's subtree has exactly one distinct name (both
    questionnaires agree), so it's one group, and the shallower questionnaire (directly in the
    well folder) is the one attached, not the one in `OLD Files`."""
    root = tmp_path / "root"
    well = root / "Well1"
    old_files = well / "OLD Files"
    well.mkdir(parents=True)
    old_files.mkdir(parents=True)
    shallow_q = well / "well_questionnaire.xlsx"
    deep_q = old_files / "well_questionnaire_copy.xlsx"
    shallow_q.write_bytes(b"dummy")
    deep_q.write_bytes(b"dummy")

    def fake_names(paths):
        return {p: "well one" for p in paths}  # both parse to the same name

    monkeypatch.setattr(features, "_questionnaire_well_names", fake_names)
    t_s, p, r = _dfit_arrays(inj_peak=5000.0)
    _write_csv(well / "data.csv", t_s, p, rate=r)

    scans = features.scan_folders(str(root), require_questionnaire=True)

    assert len(scans) == 1
    scan = scans[0]
    assert scan.n_wells == 1
    assert scan.questionnaire_path == str(shallow_q)


# --------------------------------------------------------------------------------------------------
# FIX 5 (perf): the naming pass's parse is reused for the group well_name/formation lookup,
# not repeated
# --------------------------------------------------------------------------------------------------
def test_scan_folders_reuses_naming_pass_parse_for_group_lookup(tmp_path, monkeypatch):
    """`_questionnaire_well_names` (the naming pass, run once inside `_subtree_well_names`) already
    calls `questionnaire.parse_questionnaire` on every questionnaire in the tree to get its well
    name. `scan_folders`' later per-group lookup of well_name/formation must reuse that same parse
    (via `_LAST_QUESTIONNAIRE_PARSE_CACHE`) rather than parsing the one questionnaire here a second
    time. MUTATION: remove the `_LAST_QUESTIONNAIRE_PARSE_CACHE` reuse branch in `scan_folders`
    (always fall through to a fresh parse) -> `parse_questionnaire` is called twice for this one
    file instead of once, and this test fails."""
    root = tmp_path / "root"
    well = root / "Well1"
    well.mkdir(parents=True)
    quest = well / "well_questionnaire.xlsx"
    quest.write_bytes(b"dummy")

    t_s, p, r = _dfit_arrays(inj_peak=5000.0)
    _write_csv(well / "data.csv", t_s, p, rate=r)

    import unittest.mock as mock
    fake_result = mock.Mock(well_name="Well One", formation="Niobrara")
    calls: list[str] = []

    def counting_parse(path):
        calls.append(path)
        return fake_result

    monkeypatch.setattr(features.questionnaire, "parse_questionnaire", counting_parse)

    scans = features.scan_folders(str(root), require_questionnaire=True)

    assert len(scans) == 1
    assert scans[0].well_name == "Well One"
    assert scans[0].formation == "Niobrara"
    assert calls == [str(quest)]  # parsed exactly once, not once per naming pass + once per group


# --------------------------------------------------------------------------------------------------
# FIX 4: _group_questionnaire_path's same-depth tie-break is alphabetical, pinned directly
# --------------------------------------------------------------------------------------------------
def test_group_questionnaire_path_same_depth_tie_break_is_alphabetical(tmp_path):
    """Two questionnaires at the SAME depth (unlike the shallowest-wins test above, which only
    exercises depth ordering) must resolve to the alphabetically-first path -- pinned directly
    against `_group_questionnaire_path`, not just indirectly through `scan_folders`.
    MUTATION: reverse the alphabetical order (`max` instead of `min`, or negate the tie-break key)
    -> this test fails."""
    group = str(tmp_path / "Well1")
    quest_a = os.path.join(group, "a_questionnaire.xlsx")
    quest_z = os.path.join(group, "z_questionnaire.xlsx")

    # Order in the input list must not matter -- feed it both ways.
    assert features._group_questionnaire_path(group, [quest_z, quest_a]) == quest_a
    assert features._group_questionnaire_path(group, [quest_a, quest_z]) == quest_a


def test_scan_folders_multi_well_single_folder_stays_one_group_flagged(tmp_path, monkeypatch):
    """One folder holding data plus TWO questionnaires that parse to DIFFERENT well names (a
    real corpus shape: Civitas Bijou "1A and 1B") is kept as its own group -- not split, not
    merged into anything above it -- with `n_wells == 2` flagging it as ambiguous for a human to
    resolve manually."""
    root = tmp_path / "root"
    well = root / "Bijou_1A_and_1B"
    well.mkdir(parents=True)
    q1 = well / "1A_questionnaire.xlsx"
    q2 = well / "1B_questionnaire.xlsx"
    q1.write_bytes(b"dummy")
    q2.write_bytes(b"dummy")

    def fake_names(paths):
        names = {str(q1): "bijou 1a", str(q2): "bijou 1b"}
        return {p: names[p] for p in paths}

    monkeypatch.setattr(features, "_questionnaire_well_names", fake_names)
    t_s, p, r = _dfit_arrays(inj_peak=5000.0)
    _write_csv(well / "data.csv", t_s, p, rate=r)

    scans = features.scan_folders(str(root), require_questionnaire=True)

    assert len(scans) == 1
    scan = scans[0]
    assert scan.folder == str(well)
    assert scan.n_wells == 2


def test_scan_folders_parse_failure_sentinel_prevents_merge(tmp_path, monkeypatch):
    """Two questionnaires under one parent, one of which is unparseable: the sentinel name for
    the failed one must be treated as DISTINCT from the other's real name, so the parent's
    subtree reports 2 names and the walk-up from either subfolder stops there -- no merge.
    Mutating the sentinel to a constant shared across every parse failure (instead of unique per
    path) is exactly the conservatism this guards against."""
    root = tmp_path / "root"
    parent = root / "Parent"
    sub_a = parent / "SubA"
    sub_b = parent / "SubB"
    sub_a.mkdir(parents=True)
    sub_b.mkdir(parents=True)
    q_good = sub_a / "good_questionnaire.xlsx"
    q_bad = sub_b / "bad_questionnaire.xlsx"
    q_good.write_bytes(b"dummy")
    q_bad.write_bytes(b"dummy")

    def fake_names(paths):
        out = {}
        for p in paths:
            if p == str(q_good):
                out[p] = "alpha"
            else:
                out[p] = f"?{p}"  # simulates a parse-failure sentinel, unique per path
        return out

    monkeypatch.setattr(features, "_questionnaire_well_names", fake_names)

    t_s, p, r = _dfit_arrays(inj_peak=5000.0)
    _write_csv(sub_a / "a.csv", t_s, p, rate=r)
    _write_csv(sub_b / "b.csv", t_s, p, rate=r)

    scans = features.scan_folders(str(root), require_questionnaire=True)

    assert len(scans) == 2
    rels = sorted(s.rel for s in scans)
    assert rels == ["Parent/SubA", "Parent/SubB"]


def test_scan_folders_deep_0name_chain_merges_into_1name_well_root(tmp_path):
    """FIX 1 (well-root walk-up must traverse 0-name parents, not stop at the first one): a data
    file three directories below the well folder (`Well1\\Raw Data\\CSVs\\deep.csv`), with the
    well's only questionnaire off in a sibling `Well1\\Customer Data\\` -- none of `Raw Data`,
    `Raw Data\\CSVs` has a questionnaire of its own (0 names each), but climbing through both
    reaches `Well1` itself, whose subtree holds exactly 1 name. The deep file must therefore land
    in the `Well1` well-root group, not be dropped as its own 0-questionnaire group under
    `require_questionnaire=True` -- the exact data-loss class this feature exists to fix.
    MUTATION: revert the 0-name-continue to stop-at-0 -> this test fails (the deep file's group
    would resolve to `Raw Data\\CSVs`, which has no questionnaire, and gets dropped)."""
    root = tmp_path / "root"
    well = root / "Well1"
    customer_data = well / "Customer Data"
    raw_data = well / "Raw Data"
    csvs = raw_data / "CSVs"
    customer_data.mkdir(parents=True)
    csvs.mkdir(parents=True)
    (customer_data / "well_questionnaire.xlsx").write_bytes(b"dummy")

    t_s, p, r = _dfit_arrays(inj_peak=5000.0)
    _write_csv(csvs / "deep.csv", t_s, p, rate=r)

    scans = features.scan_folders(str(root), require_questionnaire=True)

    assert len(scans) == 1
    scan = scans[0]
    assert scan.folder == str(well)
    assert scan.rel == "Well1"
    assert scan.n_wells == 1
    assert len(scan.files) == 1
    deep_feat = scan.files[0]
    assert deep_feat.path == str(csvs / "deep.csv")
    # The file's own immediate directory is unchanged (still 3 levels below the well root) --
    # `review_app.file_subfolder_label` is the provenance guard for this, computed from exactly
    # this pair of paths.
    assert deep_feat.folder == str(csvs)
    assert os.path.relpath(deep_feat.folder, scan.folder) == os.path.join("Raw Data", "CSVs")


def test_scan_folders_0name_chain_with_no_1name_ancestor_still_dropped(tmp_path):
    """The other half of FIX 1: a 0-name chain with NO 1-name ancestor anywhere below `root` must
    still resolve to `entry.folder` itself (never updates `best`), and therefore still gets
    dropped by `require_questionnaire=True`, exactly as before the fix -- climbing through
    ancestors that never resolve any name must not somehow manufacture a group with a
    questionnaire out of nothing."""
    root = tmp_path / "root"
    raw_data = root / "NoQuestWell" / "Raw Data"
    csvs = raw_data / "CSVs"
    csvs.mkdir(parents=True)
    t_s, p, r = _dfit_arrays(inj_peak=5000.0)
    _write_csv(csvs / "deep.csv", t_s, p, rate=r)

    scans_required = features.scan_folders(str(root), require_questionnaire=True)
    assert scans_required == []

    scans_optional = features.scan_folders(str(root), require_questionnaire=False)
    assert len(scans_optional) == 1
    assert scans_optional[0].folder == str(csvs)  # unchanged: no 1-name ancestor to climb to
    assert scans_optional[0].n_wells == 0


def test_scan_folders_no_questionnaire_chain_dropped_and_kept(tmp_path):
    """A folder tree with no questionnaire anywhere: every candidate group's subtree reports 0
    distinct names, so `require_questionnaire=True` drops all of them, and
    `require_questionnaire=False` keeps them (unaffected -- same behavior as before well-root
    grouping)."""
    root = tmp_path / "root"
    folder = root / "NoQuestWell"
    folder.mkdir(parents=True)
    t_s, p, r = _dfit_arrays(inj_peak=5000.0)
    _write_csv(folder / "data.csv", t_s, p, rate=r)

    scans_required = features.scan_folders(str(root), require_questionnaire=True)
    assert scans_required == []

    scans_optional = features.scan_folders(str(root), require_questionnaire=False)
    assert len(scans_optional) == 1
    assert scans_optional[0].n_wells == 0


# --------------------------------------------------------------------------------------------------
# _questionnaire_well_names: parse-failure sentinel
# --------------------------------------------------------------------------------------------------
def test_questionnaire_well_names_parse_failure_sentinel_is_unique_per_path():
    """Two different questionnaire paths that both fail to parse (here: they don't exist at all,
    which `parse_questionnaire` reports the same way as a corrupt workbook -- an exception) must
    get DIFFERENT sentinel names, keyed by their own path -- never the same constant placeholder.
    A shared constant sentinel would make two genuinely different, unreadable questionnaires
    look like the same well name and merge their folders, exactly the failure this module's
    conservatism is built to prevent."""
    names = features._questionnaire_well_names(["/no/such/a.xlsx", "/no/such/b.xlsx"])
    assert names["/no/such/a.xlsx"] != names["/no/such/b.xlsx"]
    assert names["/no/such/a.xlsx"].startswith("?")
    assert names["/no/such/b.xlsx"].startswith("?")


def test_questionnaire_well_names_real_name_is_normalized():
    """A real (mocked) parsed well name is stripped, casefolded, and has internal whitespace
    collapsed -- so " Well   ONE \n" and "well one" are recognized as the same well."""
    import unittest.mock as mock
    from triage import features as features_mod

    fake_result = mock.Mock(well_name=" Well   ONE \n")
    with mock.patch.object(
        features_mod.questionnaire, "parse_questionnaire", return_value=fake_result
    ):
        names = features._questionnaire_well_names(["/data/q.xlsx"])
    assert names["/data/q.xlsx"] == "well one"


# --------------------------------------------------------------------------------------------------
# scan_folders: limit
# --------------------------------------------------------------------------------------------------
def _build_n_folder_root(tmp_path, n: int):
    """`n` folders, each named `Folder{i}` holding one likely-DFIT CSV -- distinct content per
    folder (no cross-folder signature collisions to worry about)."""
    root = tmp_path / "root"
    root.mkdir()
    for i in range(n):
        folder = root / f"Folder{i}"
        folder.mkdir()
        t_s, p, r = _dfit_arrays(inj_peak=5000.0 + i)  # tiny per-folder variation, distinct bytes
        _write_csv(folder / "file.csv", t_s, p, rate=r)
    return str(root)


def test_scan_folders_limit_truncates_before_extraction(tmp_path, monkeypatch):
    """The regression test: with the old "limit applied after the loop" ordering, io_load.load
    would be called once per folder (4 times) even though only 2 `FolderScan`s come back. With
    the fix, only the 2 surviving folders' files are ever loaded."""
    root = _build_n_folder_root(tmp_path, 4)

    calls = []
    real_load = features.io_load.load

    def counting_load(path):
        calls.append(path)
        return real_load(path)

    monkeypatch.setattr(features.io_load, "load", counting_load)

    scans = features.scan_folders(root, require_questionnaire=False, limit=2)

    assert len(scans) == 2
    assert len(calls) == 2  # not 4 -- the other 2 folders' files were never read at all
    loaded_folders = {os.path.dirname(p) for p in calls}
    scanned_folders = {s.folder for s in scans}
    assert loaded_folders == scanned_folders


def test_scan_folders_limit_is_reproducible_rel_sorted_first_n(tmp_path):
    root = _build_n_folder_root(tmp_path, 4)

    all_scans = features.scan_folders(root, require_questionnaire=False)
    expected_rels = sorted(s.rel for s in all_scans)[:2]

    scans_a = features.scan_folders(root, require_questionnaire=False, limit=2)
    scans_b = features.scan_folders(root, require_questionnaire=False, limit=2)

    assert [s.rel for s in scans_a] == expected_rels
    assert [s.rel for s in scans_b] == expected_rels


def test_scan_folders_limit_none_scans_everything(tmp_path):
    root = _build_n_folder_root(tmp_path, 4)
    scans = features.scan_folders(root, require_questionnaire=False, limit=None)
    assert len(scans) == 4


def test_scan_folders_limit_larger_than_folder_count(tmp_path):
    root = _build_n_folder_root(tmp_path, 4)
    scans = features.scan_folders(root, require_questionnaire=False, limit=100)
    assert len(scans) == 4


# --------------------------------------------------------------------------------------------------
# save_scan / load_scan
# --------------------------------------------------------------------------------------------------
def test_save_scan_load_scan_round_trip(tmp_path):
    root = str(tmp_path)
    feat = features.FileFeatures(
        path="/data/a.csv", folder="/data", size_bytes=123, sig="123:abc",
        same_bytes_as=["/data/other.csv"], rows=500, duration_hr=12.5, pressure_col="Pressure",
        rate_col="Rate", unit_guess="psi", p_max=9000.0, p_at_shutin=8900.0,
        shutin_source="rate", injection_min=45.0, post_shutin_hr=11.0, drop=1500.0,
        decline_fraction=0.9, verdict="likely_dfit", load_error="",
    )
    scan = features.FolderScan(
        folder="/data", rel="CustomerA/Well1", well_name="Well 1", formation="Niobrara",
        questionnaire_path="/data/questionnaire.xlsx", n_wells=2, files=[feat],
        suggested=["/data/a.csv"],
    )

    path = features.save_scan(root, [scan])
    assert path == os.path.join(root, "_triage", "features.json")

    loaded = features.load_scan(root)
    assert len(loaded) == 1
    assert loaded[0] == scan
    assert loaded[0].n_wells == 2


def test_load_scan_missing_raises_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        features.load_scan(str(tmp_path))


def test_load_scan_ignores_unknown_key(tmp_path):
    root = str(tmp_path)
    triage_dir = features.triage_dir(root)
    os.makedirs(triage_dir, exist_ok=True)
    with open(os.path.join(triage_dir, "features.json"), "w", encoding="utf-8") as fh:
        json.dump([{
            "folder": "/data", "rel": "CustomerA/Well1", "well_name": "", "formation": "",
            "questionnaire_path": "", "suggested": [], "future_scan_field": "surprise",
            "files": [{
                "path": "/data/a.csv", "folder": "/data", "size_bytes": 1, "sig": "1:x",
                "future_file_field": "surprise",
            }],
        }], fh)

    loaded = features.load_scan(root)
    assert len(loaded) == 1
    assert loaded[0].rel == "CustomerA/Well1"
    assert loaded[0].files[0].path == "/data/a.csv"
    assert not hasattr(loaded[0], "future_scan_field")
    assert not hasattr(loaded[0].files[0], "future_file_field")


def test_load_scan_missing_n_wells_defaults_to_1(tmp_path):
    """A `features.json` written before `n_wells` existed (FolderScan.n_wells's whole reason for
    having a default) must load without crashing, defaulting the field to 1 rather than raising
    a `TypeError` for a missing required argument."""
    root = str(tmp_path)
    triage_dir = features.triage_dir(root)
    os.makedirs(triage_dir, exist_ok=True)
    with open(os.path.join(triage_dir, "features.json"), "w", encoding="utf-8") as fh:
        json.dump([{
            "folder": "/data", "rel": "CustomerA/Well1", "well_name": "", "formation": "",
            "questionnaire_path": "", "suggested": [], "files": [],
            # no "n_wells" key at all -- the pre-FEATURE-1 shape.
        }], fh)

    loaded = features.load_scan(root)
    assert len(loaded) == 1
    assert loaded[0].n_wells == 1


# --------------------------------------------------------------------------------------------------
# basins.basin_for
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("formation", [
    "Niobrara B Chalk", "Nio B", "C-Chalk", "Niobrara - C-Chalk",
])
def test_basin_for_dj_formation_spellings(formation):
    assert basins.basin_for(formation, None) == ("DJ", "formation")


def test_basin_for_unknown_formation_known_customer():
    basin, source = basins.basin_for("Some Unknown Formation", "Bonanza Creek")
    assert (basin, source) == ("DJ", "customer")


def test_basin_for_unknown_both():
    assert basins.basin_for("Some Unknown Formation", "Some Unknown Customer") == (
        basins.DEFAULT_BASIN, "default"
    )


def test_basin_for_empty_formation_falls_through_to_customer():
    basin, source = basins.basin_for("", "Bonanza Creek")
    assert (basin, source) == ("DJ", "customer")


def test_basin_for_multi_basin_operator_resolves_via_formation_not_customer():
    assert basins.basin_for("Wolfcamp A1", "Abraxas Petroleum Corp") == ("Permian", "formation")
    assert basins.basin_for("", "Abraxas Petroleum Corp") == (basins.DEFAULT_BASIN, "default")


def test_basin_for_same_customer_two_basins_by_formation():
    assert basins.basin_for("Turner", "Oxy (formerly APC)") == ("Powder River", "formation")
    assert basins.basin_for("Niobrara", "Oxy (formerly APC)") == ("DJ", "formation")


def test_customer_to_basin_has_all_70_customers():
    expected = {
        "ARC Resources", "Abraxas Petroleum Corp", "Aspect Energy", "B-32 Exploration",
        "Ballard Petroleum", "Birch Resources", "Black Hills", "Bonanza Creek", "Boomtown Oil",
        "Bright Rock", "Catamount Energy Partners", "Centennial Resources", "Civitas",
        "Clear Creek", "Conoco Phillips", "Continental Resources", "Crescent Point",
        "Crestone Peak", "Cygnet", "Devon Energy", "EOG", "Edge Energy", "Elk Mesa",
        "Emerald Oil", "Extraction Oil & Gas", "Fifth Creek Energy", "Fulcrum Energy",
        "GMT Exploration", "Great Western", "Halcon Resources", "Hat Creek Resources",
        "Helis Oil & Gas", "Hess", "Highlands Natural Resources", "Impact E&P", "Kinney Oil",
        "Kiwetinohk", "Koch Exploration", "Koda Resources", "Laramie Energy", "Laredo",
        "Liberty Resources", "Lonestar Operating", "Lucero Energy", "Mallard Exploration",
        "Noble Energy", "North Peak", "North Plains Energy", "North Silo",
        "Oxy (formerly APC)", "PDC Energy", "Phoenix Energy", "Red Willow",
        "Rockies Resources", "Roost Resources", "SM Energy", "Sandpoint Resources",
        "Strathcona Resources", "Synergy Resources", "Tamboran", "Tap Rock Resources",
        "True Oil", "Vermilion", "Vesta Energy", "WPX Energy", "Ward Petroleum",
        "Wave Petroleum", "Whiting Oil & Gas Corp", "Williams", "Zavanna LLC",
    }
    assert expected == set(basins.CUSTOMER_TO_BASIN)
    assert len(expected) == 70


# --------------------------------------------------------------------------------------------------
# figure.page_count / render_grid / render_file_png
# --------------------------------------------------------------------------------------------------
def _dummy_scan(n_files: int) -> features.FolderScan:
    files = [
        features.FileFeatures(
            path=f"/data/f{i}.csv", folder="/data", size_bytes=10, sig=f"10:sig{i}",
            verdict="likely_dfit",
        )
        for i in range(n_files)
    ]
    return features.FolderScan(folder="/data", rel="CustomerA/Well1", files=files)


def test_page_count():
    assert figure.page_count(_dummy_scan(3), per_page=8) == 1
    assert figure.page_count(_dummy_scan(17), per_page=8) == 3
    assert figure.page_count(_dummy_scan(0), per_page=8) == 1  # minimum 1


def test_render_grid_visible_axes_count_3_files(tmp_path):
    scan = _dummy_scan(3)
    fig = figure.render_grid(scan, str(tmp_path), page=0, per_page=8)
    visible = [ax for ax in fig.axes if ax.axison]
    assert len(visible) == 3
    assert len(fig.axes) == 8


def test_render_grid_visible_axes_count_17_files_paginated(tmp_path):
    scan = _dummy_scan(17)
    fig_page0 = figure.render_grid(scan, str(tmp_path), page=0, per_page=8)
    fig_page2 = figure.render_grid(scan, str(tmp_path), page=2, per_page=8)

    assert len([ax for ax in fig_page0.axes if ax.axison]) == 8
    assert len([ax for ax in fig_page2.axes if ax.axison]) == 1  # 17 - 2*8 = 1 file on last page


def test_render_grid_missing_png_shows_placeholder_not_exception(tmp_path):
    scan = _dummy_scan(2)
    fig = figure.render_grid(scan, str(tmp_path), page=0, per_page=8)
    # No PNGs were pre-written under tmp_path/_triage/png -- must not raise, and each populated
    # panel should carry the "(no plot)" placeholder text.
    texts = [t.get_text() for ax in fig.axes if ax.axison for t in ax.texts]
    assert texts.count("(no plot)") == 2


def test_render_grid_keep_and_suggested_spine_colors(tmp_path):
    scan = _dummy_scan(2)
    scan.suggested = ["/data/f1.csv"]
    fig = figure.render_grid(scan, str(tmp_path), page=0, per_page=8, keeps={"/data/f0.csv"})

    visible = [ax for ax in fig.axes if ax.axison]
    kept_ax, suggested_ax = visible[0], visible[1]
    kept_color = kept_ax.spines["top"].get_edgecolor()
    suggested_color = suggested_ax.spines["top"].get_edgecolor()
    assert kept_color != suggested_color


def test_render_grid_duplicate_panel_marked_distinctly(tmp_path):
    """FIX 3: a `verdict == "duplicate"` panel must be unmistakable -- distinct spine, a
    `"DUPLICATE"`-prefixed title, and an overlay naming its representative -- even though its
    cached PNG (keyed by content signature) is the representative's plot/annotation."""
    files = [
        features.FileFeatures(
            path="/data/rep.csv", folder="/data", size_bytes=10, sig="10:sig0",
            verdict="likely_dfit",
        ),
        features.FileFeatures(
            path="/data/dup.csv", folder="/data", size_bytes=10, sig="10:sig0",
            verdict="duplicate", dup_of="/data/rep.csv",
        ),
    ]
    scan = features.FolderScan(folder="/data", rel="CustomerA/Well1", files=files)

    fig = figure.render_grid(scan, str(tmp_path), page=0, per_page=8)
    visible = [ax for ax in fig.axes if ax.axison]
    rep_ax, dup_ax = visible[0], visible[1]

    assert "DUPLICATE" in dup_ax.get_title()
    assert "DUPLICATE" not in rep_ax.get_title()

    rep_color = rep_ax.spines["top"].get_edgecolor()
    dup_color = dup_ax.spines["top"].get_edgecolor()
    assert dup_color != rep_color

    overlay_texts = [t.get_text() for t in dup_ax.texts]
    assert any("rep.csv" in t for t in overlay_texts)


@pytest.mark.parametrize("verdict,load_error", [
    ("load_error", "ValueError: boom"),
    ("no_pressure", ""),
])
def test_render_file_png_error_verdicts_have_no_axes_with_data(tmp_path, verdict, load_error):
    """`load_error`/`no_pressure` files get an axes-free error panel: no plotted line data, just
    the centered filename/message text -- render_file_png builds this via `_error_panel`, so
    exercise that directly against a captured Figure (render_file_png itself only returns the
    saved path, not the Figure)."""
    from matplotlib.figure import Figure as _Figure  # local import, matches figure.py's OO-only use

    feat = features.FileFeatures(
        path="/data/broken.csv", folder="/data", size_bytes=10, sig="10:x",
        verdict=verdict, load_error=load_error,
    )

    fig = _Figure(figsize=(6.0, 3.2))
    figure._error_panel(fig, feat, load_error or "no recognizable pressure channel")
    lines = [ln for ax in fig.axes for ln in ax.get_lines()]
    assert lines == []
    texts = [t.get_text() for ax in fig.axes for t in ax.texts]
    assert any("broken.csv" in t for t in texts)

    out_path = tmp_path / f"{verdict}.png"
    result = figure.render_file_png(feat, str(out_path))
    assert result == str(out_path)
    assert out_path.exists()


# --------------------------------------------------------------------------------------------------
# figure.render_file_png: FIX 4 -- the plot must use the same truncation as the numbers
# --------------------------------------------------------------------------------------------------
def test_render_file_png_plotted_xdata_matches_reported_duration_hr(tmp_path, monkeypatch):
    """Patching `figure.monotonic_prefix` to a no-op left every triage test passing before this
    fix, because nothing checked that the PLOT truncates its x-data the same way `extract`
    truncates `duration_hr` -- exactly the "numbers look sane while the plot disagrees" failure.
    Render a record with real trailing DBS-style padding and assert the plotted line's x-data
    maximum equals the reported `duration_hr`, not the padded record's raw last timestamp."""
    from matplotlib.figure import Figure as _Figure  # local import, matches figure.py's OO-only use

    t_s, p, r = _dfit_arrays()
    n_pad = 104
    t_s_padded = np.concatenate([t_s, np.zeros(n_pad)])
    p_padded = np.concatenate([p, np.full(n_pad, p[-1])])
    r_padded = np.concatenate([r, np.zeros(n_pad)])
    path = tmp_path / "padded.csv"
    _write_csv(path, t_s_padded, p_padded, rate=r_padded)

    feat = features.extract(str(path))
    assert feat.trailing_dropped == n_pad  # sanity: the numbers side did truncate

    captured: list = []
    real_savefig = _Figure.savefig

    def capturing_savefig(self, *args, **kwargs):
        captured.append(self)
        return real_savefig(self, *args, **kwargs)

    monkeypatch.setattr(_Figure, "savefig", capturing_savefig)

    out_path = tmp_path / "padded.png"
    figure.render_file_png(feat, str(out_path))

    assert len(captured) == 1
    fig = captured[0]
    lines = [ln for ax in fig.axes for ln in ax.get_lines()]
    assert lines  # a real plot was drawn, not an error panel
    x_max = max(np.max(ln.get_xdata()) for ln in lines if len(ln.get_xdata()))
    # Without the fix, the plotted x-data would run out to the padded record's raw last
    # timestamp (0.0, since the padding rows reset t_s to 0) rather than stopping at the same
    # truncated point the reported duration_hr uses.
    assert x_max == pytest.approx(feat.duration_hr, rel=1e-6)


# --------------------------------------------------------------------------------------------------
# figure.render_file_png: nonfinite_time_dropped annotation (third review round, FIX 4)
# --------------------------------------------------------------------------------------------------
def _capture_render_file_png(feat, out_path, monkeypatch):
    """Render `feat` through `figure.render_file_png`, returning the Figure it built (which the
    function itself doesn't return -- only the saved path) by capturing `Figure.savefig`, same
    approach as the plotted-xdata parity test above."""
    from matplotlib.figure import Figure as _Figure  # local import, matches figure.py's OO-only use

    captured: list = []
    real_savefig = _Figure.savefig

    def capturing_savefig(self, *args, **kwargs):
        captured.append(self)
        return real_savefig(self, *args, **kwargs)

    monkeypatch.setattr(_Figure, "savefig", capturing_savefig)
    figure.render_file_png(feat, str(out_path))
    assert len(captured) == 1
    return captured[0]


def test_render_file_png_shows_nonfinite_time_dropped_when_nonzero(tmp_path, monkeypatch):
    """FIX 4: a non-zero `nonfinite_time_dropped` must be surfaced in the panel annotation,
    alongside the existing `trailing_dropped` line -- mutating this to drop the `if
    feat.nonfinite_time_dropped:` block (or the append itself) leaves this test failing."""
    t_s, p, r = _dfit_arrays()
    bad_indices = [0, 5, 200, 1500, 3000]
    path = tmp_path / "nan_scattered.csv"
    _write_csv_with_bad_datetime_rows(path, t_s, p, r, bad_indices=bad_indices)
    feat = features.extract(str(path))
    assert feat.nonfinite_time_dropped == len(bad_indices)  # sanity: the count side is non-zero

    fig = _capture_render_file_png(feat, tmp_path / "a.png", monkeypatch)
    texts = [t.get_text() for ax in fig.axes for t in ax.texts]
    assert any(f"nonfinite timestamps dropped: {len(bad_indices)}" in t for t in texts)


def test_render_file_png_omits_nonfinite_time_dropped_line_when_zero(tmp_path, monkeypatch):
    """The annotation line is a no-op whenever the value is zero, regardless of
    PNG_RENDER_VERSION -- pin that here too."""
    t_s, p, r = _dfit_arrays()
    path = tmp_path / "clean.csv"
    _write_csv(path, t_s, p, rate=r)
    feat = features.extract(str(path))
    assert feat.nonfinite_time_dropped == 0  # sanity: the count side is zero

    fig = _capture_render_file_png(feat, tmp_path / "a.png", monkeypatch)
    texts = [t.get_text() for ax in fig.axes for t in ax.texts]
    assert not any("nonfinite timestamps dropped" in t for t in texts)


# --------------------------------------------------------------------------------------------------
# FIX 3 (third review round) -- five regression guards that had no test before this round; each
# is confirmed (see task report) to FAIL under the stated mutation.
# --------------------------------------------------------------------------------------------------
def test_png_render_version_is_an_integer():
    """PNG_RENDER_VERSION must be bumped whenever render_file_png's output or the features
    feeding its annotation change -- most recently 2 -> 3, once a corpus measurement showed the
    FIX 4 nonfinite_time_dropped annotation affects roughly 275 of 1,237 files (~22%), not the
    ~1% originally assumed, so every cached PNG needed invalidating for the annotation to ever
    reach a review panel. This test intentionally does not pin a literal value (the whole point
    is that it keeps changing) -- `png_path_for`'s version-agnostic behavior is covered by the
    two tests below instead."""
    assert isinstance(features.PNG_RENDER_VERSION, int)


def test_png_path_for_includes_render_version_in_filename():
    """A3: `png_path_for` must key the cached filename on `PNG_RENDER_VERSION` in addition to the
    content signature -- mutation: remove the `_v{PNG_RENDER_VERSION}` suffix -- so bumping that
    constant invalidates every previously cached PNG instead of a stale pre-fix render being
    silently reused next to fresh, disagreeing feature numbers."""
    feat = features.FileFeatures(path="/data/a.csv", folder="/data", size_bytes=10, sig="10:abc")
    path = features.png_path_for("/root", feat)
    assert f"_v{features.PNG_RENDER_VERSION}.png" in path


def test_png_path_for_changes_when_render_version_bumped(monkeypatch):
    feat = features.FileFeatures(path="/data/a.csv", folder="/data", size_bytes=10, sig="10:abc")
    path_v2 = features.png_path_for("/root", feat)
    monkeypatch.setattr(features, "PNG_RENDER_VERSION", features.PNG_RENDER_VERSION + 1)
    path_v3 = features.png_path_for("/root", feat)
    assert path_v2 != path_v3


def test_extract_rows_counts_the_truncated_view_not_the_raw_array(tmp_path):
    """A5: `rows` must count finite-pressure samples surviving BOTH the NaN-timestamp drop and
    the monotonic-prefix truncation -- mutation: `int(np.isfinite(p_clean_full).sum())` (the raw,
    pre-truncation array) -- so a real padded DBS reports 739,257 (post-truncation) rather than
    739,361 (raw). Reproduced here with a smaller padded fixture: `rows` must equal the finite
    count AFTER the trailing pad is dropped, not before."""
    path = tmp_path / "padded.csv"
    t_s, p, r = _dfit_arrays()
    n_pad = 104
    t_s_padded = np.concatenate([t_s, np.zeros(n_pad)])
    p_padded = np.concatenate([p, np.full(n_pad, p[-1])])
    r_padded = np.concatenate([r, np.zeros(n_pad)])
    _write_csv(path, t_s_padded, p_padded, rate=r_padded)

    feat = features.extract(str(path))

    assert feat.trailing_dropped == n_pad
    # Every pre-padding sample has finite pressure, so the truncated (correct) count equals the
    # pre-padding length -- and is strictly less than the raw, pre-truncation array's length.
    assert feat.rows == len(t_s)
    assert feat.rows == len(p_padded) - n_pad
    assert feat.rows < len(p_padded)


def test_render_file_png_masks_nan_timestamps_not_just_truncation(tmp_path, monkeypatch):
    """The plot must apply the `finite_t` (NaN-timestamp) mask to the pressure column, not just
    the `mono` truncation -- mutation: `td.column(pressure_col)[mono]`, dropping `[finite_t]`.
    Because `bad_indices` are scattered (not clustered at the start), dropping the mask does NOT
    raise a shape mismatch: `mono` is a plain `slice(0, k)`, and slicing the raw (longer,
    NaN-timestamp-including) pressure array with that same slice yields an array of the SAME
    length `k` as the masked one, just built from the wrong (off-by-however-many-NaNs-precede-
    each-point) elements -- silently misaligning every pressure value against its timestamp
    rather than raising or changing the reported duration. The existing padding-only parity
    fixture has no NaN timestamps, so it can't catch this, and checking only `x_max` (a
    time-axis quantity, unaffected by which pressure values got plotted) doesn't either -- this
    compares the plotted line's actual Y data against independently-recomputed ground truth."""
    t_s, p, r = _dfit_arrays()
    bad_indices = [0, 5, 200, 1500, 3000]
    path = tmp_path / "nan_scattered.csv"
    _write_csv_with_bad_datetime_rows(path, t_s, p, r, bad_indices=bad_indices)

    feat = features.extract(str(path))
    assert feat.verdict != "load_error"

    # Ground truth: the same finite_t-mask-then-mono-truncate steps render_file_png/extract use,
    # computed independently here against the KNOWN synthetic pressure array.
    finite_t = np.ones(len(t_s), dtype=bool)
    finite_t[bad_indices] = False
    t_s_nonan = t_s[finite_t]
    mono = features.monotonic_prefix(t_s_nonan)
    expected_p = p[finite_t][mono]

    fig = _capture_render_file_png(feat, tmp_path / "nan_scattered.png", monkeypatch)
    pressure_ax = fig.axes[0]  # the primary axes, added before any twinx() rate axes
    lines = pressure_ax.get_lines()
    assert lines  # a real plot, not an error panel
    plotted_p = lines[0].get_ydata()
    assert len(plotted_p) == len(expected_p)
    assert np.allclose(plotted_p, expected_p)
