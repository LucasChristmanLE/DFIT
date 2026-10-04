"""store._sniff_xlsx_cached: a rescan of an unchanged workbook must not re-sniff it; a touched
or rewritten file must."""
import os

import pytest

from dfit_tool import io_load, store
from tests.test_store_xlsx import _write_data_xlsx


@pytest.fixture
def sniff_calls(monkeypatch):
    calls = []
    real = io_load.sniff_xlsx_data

    def counting(path):
        calls.append(path)
        return real(path)

    monkeypatch.setattr(io_load, "sniff_xlsx_data", counting)
    store._SNIFF_CACHE.clear()
    return calls


def test_second_scan_does_not_sniff_again(tmp_path, sniff_calls):
    _write_data_xlsx(tmp_path / "a.xlsx")
    first = store.scan_root(str(tmp_path))
    n_first = len(sniff_calls)
    assert n_first >= 1
    second = store.scan_root(str(tmp_path))
    assert len(sniff_calls) == n_first
    assert [e.test_id for e in first] == [e.test_id for e in second]


def test_mtime_change_invalidates(tmp_path, sniff_calls):
    path = tmp_path / "a.xlsx"
    _write_data_xlsx(path)
    store.scan_root(str(tmp_path))
    n = len(sniff_calls)
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    store.scan_root(str(tmp_path))
    assert len(sniff_calls) == n + 1


def test_missing_file_is_not_cached(sniff_calls, tmp_path):
    assert store._sniff_xlsx_cached(str(tmp_path / "nope.xlsx")) is False
    assert store._SNIFF_CACHE == {}
