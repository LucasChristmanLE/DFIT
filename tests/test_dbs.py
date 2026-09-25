"""Tests for the Fracpro .DBS binary reader in dfit_tool/io_load.py.

``write_dbs`` builds a minimal, spec-valid .DBS file in memory so the round-trip tests don't
depend on a real Fracpro export. Only the first 4 bytes of the 8-byte header prefix are a
constant magic; bytes 4-7 are a per-file save timestamp that varies between real exports, so
``write_dbs`` deliberately uses a save-timestamp different from any real file's to pin that
those bytes are never validated. The real-file smoke tests at the bottom exercise the actual
Caprito 99-202H merged file and the Argentine State 7170 file on disk and are skipped if those
files aren't present.
"""

import os
import struct

import numpy as np
import pytest

from dfit_tool import io_load

_DBS_MAGIC = bytes.fromhex("77efcdab")
_DBS_SAVE_TIMESTAMP = 0x5C49DE4E  # arbitrary, deliberately != any real file's save timestamp
_HEADER_SIZE = 0x334
_CHANNEL_RECORD_SIZE = 84

REAL = (
    r"C:\Users\LucasChristman\OneDrive - Liberty Oilfield Services\Documents\DFIT Project"
    r"\LOS DFIT Questionnaire_Abraxas_Caprito 99-202H\Caprito 99-202H DFIT Merged.dbs"
)
REAL2 = (
    r"C:\Users\LucasChristman\OneDrive - Liberty Oilfield Services\Documents\DFIT Project"
    r"\2019.02.12_PDC_Argentine State 7170 4U B4H_Final Data"
    r"\PDC_Argentine State 7170 4U B4H DFIT.DBS"
)


def write_dbs(path, names_tags, records, interval_min):
    """Write a minimal valid .DBS file.

    ``names_tags``: list of (display_name, 4-char tag) per channel.
    ``records``: list of (idx, val0, val1, ...) tuples, one per sample, len(vals) == n_channels.
    ``interval_min``: sample interval in minutes (float32 at 0x2C0).
    """
    n_channels = len(names_tags)
    n_samples = len(records)
    data_offset = _HEADER_SIZE + _CHANNEL_RECORD_SIZE * n_channels

    header = bytearray(_HEADER_SIZE)
    header[0:4] = _DBS_MAGIC
    struct.pack_into("<I", header, 4, _DBS_SAVE_TIMESTAMP)
    struct.pack_into("<I", header, 0x2B4, n_channels)
    struct.pack_into("<I", header, 0x2B8, n_samples)
    struct.pack_into("<f", header, 0x2C0, interval_min)
    struct.pack_into("<f", header, 0x2C4, n_samples * interval_min)
    struct.pack_into("<I", header, 0x2CC, data_offset)

    chan_table = bytearray()
    for name, tag in names_tags:
        rec = bytearray(_CHANNEL_RECORD_SIZE)
        tag_bytes = tag.encode("latin-1")[:4]
        rec[0:len(tag_bytes)] = tag_bytes
        name_bytes = name.encode("latin-1") + b"\x00"
        rec[16:16 + len(name_bytes)] = name_bytes
        chan_table += rec

    body = bytearray()
    for rec in records:
        idx = rec[0]
        vals = rec[1:]
        assert len(vals) == n_channels
        body += struct.pack("<I" + "f" * n_channels, idx, *vals)

    with open(path, "wb") as f:
        f.write(bytes(header) + bytes(chan_table) + bytes(body))


# --------------------------------------------------------------------------------------------------
# round-trip
# --------------------------------------------------------------------------------------------------
def test_round_trip(tmp_path):
    path = tmp_path / "test.dbs"
    idxs = [0, 1, 2, 5, 6]
    records = [(i, float(i) * 10.0, float(i) * -1.0) for i in idxs]
    write_dbs(
        path,
        [("Surf Press [Csg]", "THCS"), ("Slurry Flow Rate", "SLRT")],
        records,
        interval_min=1.0 / 60.0,
    )

    td = io_load.load_dbs(str(path))

    assert td.columns == ["DateTime", "Surf Press [Csg]", "Slurry Flow Rate"]
    assert td.datetime_col == "DateTime"
    assert td.n == len(idxs)
    np.testing.assert_allclose(td.t_s, np.array(idxs, dtype=float) * 1.0)
    np.testing.assert_allclose(td.column("Surf Press [Csg]"), [float(i) * 10.0 for i in idxs])
    np.testing.assert_allclose(td.column("Slurry Flow Rate"), [float(i) * -1.0 for i in idxs])


# --------------------------------------------------------------------------------------------------
# error paths
# --------------------------------------------------------------------------------------------------
def test_bad_magic_raises(tmp_path):
    path = tmp_path / "bad.dbs"
    write_dbs(path, [("P", "PRES")], [(0, 1.0)], interval_min=1.0)
    data = bytearray(path.read_bytes())
    data[0:4] = b"\x00" * 4
    path.write_bytes(bytes(data))

    with pytest.raises(ValueError, match="bad magic"):
        io_load.load_dbs(str(path))


def test_truncated_data_raises(tmp_path):
    path = tmp_path / "truncated.dbs"
    write_dbs(path, [("P", "PRES")], [(0, 1.0), (1, 2.0), (2, 3.0)], interval_min=1.0)
    data = path.read_bytes()
    path.write_bytes(data[:-4])  # chop off the last sample's worth of bytes

    with pytest.raises(ValueError):
        io_load.load_dbs(str(path))


def test_wrong_data_offset_raises(tmp_path):
    path = tmp_path / "offset.dbs"
    write_dbs(path, [("P", "PRES")], [(0, 1.0), (1, 2.0)], interval_min=1.0)
    data = bytearray(path.read_bytes())
    struct.pack_into("<I", data, 0x2CC, 999999)
    path.write_bytes(bytes(data))

    with pytest.raises(ValueError):
        io_load.load_dbs(str(path))


def test_nan_interval_raises(tmp_path):
    path = tmp_path / "nan_interval.dbs"
    write_dbs(path, [("P", "PRES")], [(0, 1.0), (1, 2.0)], interval_min=1.0)
    data = bytearray(path.read_bytes())
    struct.pack_into("<f", data, 0x2C0, float("nan"))
    path.write_bytes(bytes(data))

    with pytest.raises(ValueError, match="not a positive finite number"):
        io_load.load_dbs(str(path))


# --------------------------------------------------------------------------------------------------
# duplicate channel names
# --------------------------------------------------------------------------------------------------
def test_duplicate_names_deduped_with_tag_suffix(tmp_path):
    path = tmp_path / "dup.dbs"
    write_dbs(
        path,
        [("Pressure", "AAAA"), ("Pressure", "BBBB")],
        [(0, 1.0, 2.0), (1, 3.0, 4.0)],
        interval_min=1.0,
    )

    td = io_load.load_dbs(str(path))

    assert td.columns == ["DateTime", "Pressure", "Pressure [BBBB]"]
    np.testing.assert_allclose(td.column("Pressure"), [1.0, 3.0])
    np.testing.assert_allclose(td.column("Pressure [BBBB]"), [2.0, 4.0])


def test_empty_display_name_falls_back_to_tag(tmp_path):
    path = tmp_path / "empty_name.dbs"
    write_dbs(path, [("", "PRES")], [(0, 1.0), (1, 2.0)], interval_min=1.0)

    td = io_load.load_dbs(str(path))

    assert td.columns == ["DateTime", "PRES"]
    np.testing.assert_allclose(td.column("PRES"), [1.0, 2.0])


def test_channel_named_datetime_does_not_clobber_synthetic_column(tmp_path):
    path = tmp_path / "datetime_clash.dbs"
    write_dbs(
        path,
        [("DateTime", "AAAA"), ("Pressure", "PRES")],
        [(0, 1.0, 100.0), (1, 2.0, 200.0)],
        interval_min=1.0,
    )

    td = io_load.load_dbs(str(path))

    # The synthetic datetime column keeps its name; the colliding channel gets deduped instead
    # of overwriting it -- and no data is lost.
    assert td.columns == ["DateTime", "DateTime [AAAA]", "Pressure"]
    assert td.datetime_col == "DateTime"
    np.testing.assert_allclose(td.column("DateTime [AAAA]"), [1.0, 2.0])
    np.testing.assert_allclose(td.column("Pressure"), [100.0, 200.0])


def test_triple_collision_same_name_and_tag_all_columns_distinct(tmp_path):
    path = tmp_path / "triple.dbs"
    write_dbs(
        path,
        [("Pressure", "SAME"), ("Pressure", "SAME"), ("Pressure", "SAME")],
        [(0, 1.0, 2.0, 3.0), (1, 4.0, 5.0, 6.0)],
        interval_min=1.0,
    )

    td = io_load.load_dbs(str(path))

    # Same name AND same tag for all three channels: tag-suffix dedupe alone would still collide,
    # so the third channel must fall through to the " (2)" numbered suffix. No column is dropped.
    assert td.columns == ["DateTime", "Pressure", "Pressure [SAME]", "Pressure [SAME] (2)"]
    np.testing.assert_allclose(td.column("Pressure"), [1.0, 4.0])
    np.testing.assert_allclose(td.column("Pressure [SAME]"), [2.0, 5.0])
    np.testing.assert_allclose(td.column("Pressure [SAME] (2)"), [3.0, 6.0])


# --------------------------------------------------------------------------------------------------
# unit detection: a synthetic DBS (no header suffix at all -- .DBS channel names never carry a
# parenthesized unit) with a kPa-scale pressure column must fall through to the magnitude
# heuristic alone.
# --------------------------------------------------------------------------------------------------
def test_dbs_pressure_with_no_header_suffix_detected_via_heuristic(tmp_path):
    path = tmp_path / "metric.dbs"
    records = [(i, 90000.0 + i, 0.0) for i in range(5)]
    write_dbs(path, [("Surf Press [Csg]", "THCS"), ("Rate", "SLRT")], records, interval_min=1.0)

    td = io_load.load_dbs(str(path))
    io_load.refresh_unit_detection(td, "Surf Press [Csg]")

    det = td.unit_detections["Surf Press [Csg]"]
    assert det.unit == "kpa"
    assert det.source == "heuristic"
    assert det.confidence == "high"


# --------------------------------------------------------------------------------------------------
# trailing padding truncation (preallocated-but-not-fully-written recording buffer)
# --------------------------------------------------------------------------------------------------
def test_trailing_zero_idx_padding_is_truncated(tmp_path):
    path = tmp_path / "padded.dbs"
    k = 6
    real = [(i, float(i) * 10.0) for i in range(k)]
    # Trailing padding: idx repeats 0 for every one of these, with arbitrary/garbage channel
    # values (not zero) -- the fix must key off idx alone, never off the channel floats.
    padding = [(0, 999.0), (0, -5.0), (0, 42.0)]
    write_dbs(path, [("Pressure", "PRES")], real + padding, interval_min=1.0 / 60.0)

    td = io_load.load_dbs(str(path))

    assert td.n == k
    np.testing.assert_allclose(td.t_s, np.arange(k, dtype=float))
    assert td.t_s[-1] != 0.0
    np.testing.assert_allclose(td.column("Pressure"), [float(i) * 10.0 for i in range(k)])
    assert len(td.load_warnings) == 1
    assert "ignored 3 empty trailing records" in td.load_warnings[0]


def test_legitimate_idx_gap_not_truncated(tmp_path):
    path = tmp_path / "gap.dbs"
    idxs = [0, 1, 2, 500, 503, 504]  # 500 -> 503 is a legitimate skip, still strictly increasing
    records = [(i, float(i)) for i in idxs]
    write_dbs(path, [("Pressure", "PRES")], records, interval_min=1.0)

    td = io_load.load_dbs(str(path))

    assert td.n == len(idxs)
    np.testing.assert_allclose(td.t_s, np.array(idxs, dtype=float) * 60.0)
    assert td.load_warnings == []


def test_mid_file_idx_jitter_not_truncated(tmp_path):
    # A repeat-then-skip pair in the MIDDLE of an otherwise-normal file: idx repeats 2, then
    # skips straight to 4, then continues normally to a proper terminal value. This is the exact
    # shape a forward (first-violation) scan misidentifies as trailing padding and truncates down
    # to almost nothing -- the backward scan must recognize the counter recovers before EOF and
    # leave the file untouched.
    path = tmp_path / "jitter.dbs"
    idxs = [0, 1, 2, 2, 4, 5, 6, 7]
    records = [(i, float(i)) for i in idxs]
    write_dbs(path, [("Pressure", "PRES")], records, interval_min=1.0)

    td = io_load.load_dbs(str(path))

    assert td.n == len(idxs)
    np.testing.assert_allclose(td.t_s, np.array(idxs, dtype=float) * 60.0)
    assert td.load_warnings == []


def test_clean_file_truncation_is_a_no_op(tmp_path):
    # Same shape as test_round_trip: idx strictly increasing start to end, no padding at all.
    path = tmp_path / "clean.dbs"
    idxs = [0, 1, 2, 5, 6]
    records = [(i, float(i) * 10.0) for i in idxs]
    write_dbs(path, [("Pressure", "PRES")], records, interval_min=1.0)

    td = io_load.load_dbs(str(path))

    assert td.n == len(idxs)
    np.testing.assert_allclose(td.t_s, np.array(idxs, dtype=float) * 60.0)
    np.testing.assert_allclose(td.column("Pressure"), [float(i) * 10.0 for i in idxs])
    assert td.load_warnings == []


def test_load_warnings_fold_into_compute_all_warnings():
    """model.compute_all must surface td.load_warnings in the analyst-visible warnings panel --
    the actual wiring this whole fix exists to reach, not just io_load's own return value."""
    from dfit_tool.model import compute_all
    from tests.helpers import injection_state, make_testdata

    td = make_testdata()
    td.load_warnings = ["3 of 9 declared sample(s) after sample number 6 look like "
                        "unrecorded padding; dropped."]
    st = injection_state(td)

    res = compute_all(st, td)

    assert any("unrecorded padding" in w for w in res.warnings)


# --------------------------------------------------------------------------------------------------
# dispatcher
# --------------------------------------------------------------------------------------------------
def test_load_dispatches_on_extension(tmp_path):
    dbs_path = tmp_path / "well.dbs"
    write_dbs(dbs_path, [("P", "PRES")], [(0, 1.0), (1, 2.0)], interval_min=1.0)
    td_dbs = io_load.load(str(dbs_path))
    assert td_dbs.columns == ["DateTime", "P"]

    csv_path = tmp_path / "well.csv"
    csv_path.write_text("DateTime,Pressure\n01/01/2024 00:00:00,100\n01/01/2024 00:00:01,101\n")
    td_csv = io_load.load(str(csv_path))
    assert "Pressure" in td_csv.columns


# --------------------------------------------------------------------------------------------------
# real-file smoke test
# --------------------------------------------------------------------------------------------------
@pytest.mark.skipif(not os.path.exists(REAL), reason="real Caprito .dbs file not present")
def test_real_caprito_file():
    td = io_load.load_dbs(REAL)

    assert td.n == 474846
    assert td.columns == ["DateTime", "Surf Press [Csg]", "Slurry Flow Rate"]
    assert td.t_s[-1] == pytest.approx(474845.0, abs=0.5)

    pressure = td.column("Surf Press [Csg]")
    assert pressure[0] == pytest.approx(16.9, abs=0.1)
    assert pressure[-1] == pytest.approx(3494.7, abs=0.5)

    rate = td.column("Slurry Flow Rate")
    assert rate.max() > 0


@pytest.mark.skipif(not os.path.exists(REAL2), reason="real Argentine .dbs file not present")
def test_real_argentine_file():
    td = io_load.load_dbs(REAL2)

    assert td.n == 1566146
    assert td.columns == ["DateTime", "Surf Press [Csg]", "Temp", "Injection Rate"]
    assert td.t_s[-1] == pytest.approx(1566145.0, abs=0.5)

    pressure = td.column("Surf Press [Csg]")
    assert pressure[0] == pytest.approx(7.98, abs=0.05)

    temp = td.column("Temp")
    assert temp[0] == pytest.approx(36.17, abs=0.05)
