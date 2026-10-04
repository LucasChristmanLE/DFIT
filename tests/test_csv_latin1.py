"""Downhole-gauge CSVs (``Sn#...``) carry a latin-1 degree byte; load_csv must read them with a
warning instead of crashing on the utf-8-sig decode."""
from dfit_tool import io_load


def _write(path, header: bytes):
    rows = [b"Date/Time," + header + b",Pressure (psi)"]
    for i in range(30):
        rows.append(f"01/01/2021 00:00:{i:02d},70.5,{5000 - i}".encode("ascii"))
    path.write_bytes(b"\r\n".join(rows) + b"\r\n")


def test_latin1_header_loads_with_warning(tmp_path):
    path = tmp_path / "Sn123 gauge.csv"
    _write(path, b"Temp (\xb0F)")
    td = io_load.load_csv(str(path))
    assert len(td.t_s) == 30
    assert "Pressure (psi)" in td.df.columns
    assert "File is not UTF-8; read as Latin-1" in td.load_warnings
    assert all(len(w) <= 90 for w in td.load_warnings)


def test_utf8_file_has_no_encoding_warning(tmp_path):
    path = tmp_path / "plain.csv"
    _write(path, "Temp (°F)".encode("utf-8"))
    td = io_load.load_csv(str(path))
    assert not any("Latin-1" in w for w in td.load_warnings)
