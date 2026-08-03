"""Headless coverage for app.py's CLI dispatch and DfitApp.__init__'s path branch (Task C):
a directory argument opens folder mode, a file argument loads single-file mode. Neither a real
tk.Tk() nor DfitApp's real widget-building methods are exercised -- same duck-typed stand-in
approach as tests/test_folder_mode.py.
"""
from __future__ import annotations

import types

from dfit_tool import app
from dfit_tool.ui import DfitApp


# --------------------------------------------------------------------------------------------------
# app.main: constructs DfitApp with path= (renamed from csv_path=) and drives its mainloop.
# --------------------------------------------------------------------------------------------------
def test_main_constructs_dfitapp_with_path_and_runs_mainloop(monkeypatch):
    calls = []

    class _FakeRoot:
        def mainloop(self):
            calls.append("mainloop")

    monkeypatch.setattr(app.tk, "Tk", lambda: _FakeRoot())

    class _FakeApp:
        def __init__(self, root, path=None):
            calls.append(("DfitApp", root, path))

    monkeypatch.setattr(app, "DfitApp", _FakeApp)

    rc = app.main(["some/folder"])

    assert rc == 0
    assert calls[0][0] == "DfitApp"
    assert calls[0][2] == "some/folder"
    assert calls[1] == "mainloop"


def test_main_no_argv_passes_none_path(monkeypatch):
    monkeypatch.setattr(app.tk, "Tk", lambda: types.SimpleNamespace(mainloop=lambda: None))
    seen = {}

    class _FakeApp:
        def __init__(self, root, path=None):
            seen["path"] = path

    monkeypatch.setattr(app, "DfitApp", _FakeApp)

    app.main([])

    assert seen["path"] is None


# --------------------------------------------------------------------------------------------------
# DfitApp.__init__: os.path.isdir(path) branches to _open_folder_path, else _load. The widget-
# building calls (_build_top/_build_body/_build_stepbar) are stubbed no-ops on the stand-in.
# --------------------------------------------------------------------------------------------------
def _init_stub():
    stub = types.SimpleNamespace()
    stub._build_top = lambda: None
    stub._build_body = lambda: None
    stub._build_stepbar = lambda: None
    stub._open_folder_calls = []
    stub._open_folder_path = lambda p: stub._open_folder_calls.append(p)
    stub._load_calls = []
    stub._load = lambda p: stub._load_calls.append(p)
    stub.root = types.SimpleNamespace(title=lambda t: None, geometry=lambda g: None)
    return stub


def test_dfitapp_init_directory_path_opens_folder_mode(tmp_path):
    stub = _init_stub()
    init = types.MethodType(DfitApp.__init__, stub)

    init(stub.root, str(tmp_path))

    assert stub._open_folder_calls == [str(tmp_path)]
    assert stub._load_calls == []


def test_dfitapp_init_file_path_loads_single_file(tmp_path):
    stub = _init_stub()
    init = types.MethodType(DfitApp.__init__, stub)
    fake_file = str(tmp_path / "well.csv")  # need not exist -- os.path.isdir is False either way

    init(stub.root, fake_file)

    assert stub._load_calls == [fake_file]
    assert stub._open_folder_calls == []


def test_dfitapp_init_no_path_does_neither():
    stub = _init_stub()
    init = types.MethodType(DfitApp.__init__, stub)

    init(stub.root, None)

    assert stub._load_calls == []
    assert stub._open_folder_calls == []
