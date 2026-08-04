"""Folder-mode persistence: scanning a root folder of tests, the per-test picks JSON, and the
``dfit_log.csv`` master log that rolls interpretations up across an entire folder.

Pure Python + pandas -- no matplotlib, no Tkinter, so this module (like model.py) is fully
unit-testable headless. It never computes an interpreted value itself: everything reported in a
log row comes from ``model.compute_all`` (via ``DerivedResults``, passed in). This module only
scans, formats, and persists what the compute layer already produced.
"""

from __future__ import annotations

import datetime
import getpass
import os
import tempfile
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

from . import model
from .model import PickState
from .questionnaire import find_questionnaire

LOG_FILENAME = "dfit_log.csv"      # lives at <opened_root>/dfit_log.csv
PICKS_SUFFIX = ".dfit_picks.json"  # <test_folder>/<test_id>.dfit_picks.json

# Deliberately duplicated from ui.py's STEPS keys, not imported -- ui.py imports tkinter at
# module level, and store.py must stay importable with no Tk on the path (see module docstring).
STEP_KEYS = ("overview", "isip", "gfunction", "tangent", "loglog", "porepressure")

# Column order: the 33 original schema columns, then the appended per-method columns the tool
# computes beyond that schema. CSV only for now; a parquet mirror alongside dfit_log.csv is a
# future extension point (would need its own load_log/save_log pair, or a format arg on these).
LOG_COLUMNS = [
    "file", "test_id", "well_name", "formation", "status", "interpreter", "review_date",
    "orientation", "fluid_type", "play", "pressure_source", "tvd", "fluid_density",
    "t_start_inj", "t_shutin", "te", "Vinj", "max_rate",
    "apparent_ISIP", "effective_ISIP",
    "closure_scenario", "closure_quality", "contact_pressure", "Shmin_compliance",
    "Shmin_tangent", "tangent_Gc",
    "postclosure_scenario", "postclosure_trend", "pore_pressure", "pp_axis",
    "pp_confidence",
    "net_pressure_compliance", "net_pressure_tangent", "delta_closure", "notes",
    # appended: per-method values the tool computes beyond the original schema
    "effective_ISIP_tangent", "effective_ISIP_variable",
    "Shmin_variable", "Shmin_rapid", "net_pressure_variable",
    "closure_time_compliance_min", "closure_time_tangent_min",
    "closure_time_variable_min",
    "net_pressure_isip_source",
]

_CLOSURE_QUALITY_BY_PREFIX = {
    "C-A": "clear", "C-B": "adequate", "C-C": "no-contact", "C-D": "rapid",
}
_POSTCLOSURE_TREND_BY_PREFIX = {
    "PC-A": "linear", "PC-B": "false-radial", "PC-C": "mixed", "PC-D": "mixed",
    "PC-E": "none", "PC-F": "none",
}


# --------------------------------------------------------------------------------------------------
# TestEntry
# --------------------------------------------------------------------------------------------------
@dataclass
class TestEntry:
    test_id: str                 # path-qualified, forward-slash-separated id, unique within a root
    folder: str                  # dir containing the data files
    csv_path: Optional[str] = None
    dbs_path: Optional[str] = None
    questionnaire_path: Optional[str] = None
    scan_warnings: list[str] = field(default_factory=list)
    status: str = "new"          # recomputed from picks JSON, never trusted from the log CSV
    picks_basename: Optional[str] = None  # local data-file stem; falls back to test_id if unset

    @property
    def picks_path(self) -> str:
        base = self.picks_basename if self.picks_basename is not None else self.test_id
        return os.path.join(self.folder, base + PICKS_SUFFIX)

    @property
    def display_label(self) -> str:
        return self.test_id.replace("/", " / ")

    @property
    def available_sources(self) -> list[str]:
        sources = []
        if self.csv_path:
            sources.append("CSV")
        if self.dbs_path:
            sources.append("DBS")
        return sources

    def data_path(self, source: str) -> str:
        s = source.upper()
        if s == "CSV":
            if not self.csv_path:
                raise ValueError(f"no CSV available for test {self.test_id!r}")
            return self.csv_path
        if s == "DBS":
            if not self.dbs_path:
                raise ValueError(f"no DBS available for test {self.test_id!r}")
            return self.dbs_path
        raise ValueError(f"unknown source {source!r} (expected 'CSV' or 'DBS')")


# --------------------------------------------------------------------------------------------------
# scan_root: arbitrary-depth scan of an opened root -- every directory under it (including the
# root itself) that holds data files becomes one or more tests, keyed by a path-qualified,
# forward-slash-separated test_id unique within the root.
# --------------------------------------------------------------------------------------------------
def _group_data_files(filenames: list[str]) -> dict[str, dict[str, str]]:
    """Group one directory's filenames by stem: `{stem: {"csv": name, "dbs": name}}`. Files that
    are not `.csv`/`.dbs` (case-insensitive) are skipped, as is `LOG_FILENAME` (case-insensitive,
    it is our own log, not a test). Filenames within one directory are unique, so each
    (stem, ext) maps to exactly one name -- no "pick first" ambiguity to warn about."""
    by_stem: dict[str, dict[str, str]] = {}
    for name in filenames:
        if name.lower() == LOG_FILENAME.lower():
            continue
        low = name.lower()
        if low.endswith(".csv"):
            ext = "csv"
        elif low.endswith(".dbs"):
            ext = "dbs"
        else:
            continue
        stem = os.path.splitext(name)[0]
        by_stem.setdefault(stem, {})[ext] = name
    return by_stem


def _entries_for_dir(root: str, dirpath: str, filenames: list[str]) -> list[TestEntry]:
    """One TestEntry per stem group in `dirpath`, per the identity rules: loose files directly in
    `root` get `test_id = stem`; a non-root dir with a single stem group collapses to
    `test_id = rel`; a non-root dir with multiple stem groups gets `test_id = rel + "/" + stem`.
    `picks_basename` is always the local stem, so the picks file stays unique within its folder
    regardless of how test_id is qualified."""
    by_stem = _group_data_files(filenames)
    if not by_stem:
        return []
    rel = os.path.relpath(dirpath, root).replace(os.sep, "/")
    entries = []
    for stem, by_ext in by_stem.items():
        if rel == ".":
            test_id = stem
        elif len(by_stem) == 1:
            test_id = rel
        else:
            test_id = f"{rel}/{stem}"
        entry = TestEntry(test_id=test_id, folder=dirpath, picks_basename=stem)
        if "csv" in by_ext:
            entry.csv_path = os.path.join(dirpath, by_ext["csv"])
        if "dbs" in by_ext:
            entry.dbs_path = os.path.join(dirpath, by_ext["dbs"])
        entries.append(entry)
    return entries


def scan_root(root: str, progress=None) -> list[TestEntry]:
    """Every candidate test anywhere under `root`, at any depth: one entry per stem group of data
    files in each walked directory (including `root` itself), keyed by a path-qualified test_id
    (see `_entries_for_dir`). Everything else (other extensions, dfit_log.csv, directories with no
    data files) is ignored. Export subdirectories named "<stem> DFIT plots" (created by Finish)
    are pruned before descending, so a stray PNG-export folder never becomes a test. Attaches each
    entry's questionnaire (via ``find_questionnaire``) and returns entries sorted by test_id.

    `progress`, if given, is an optional `progress(dirs_scanned: int, tests_found: int) -> None`
    called once per directory visited during the `os.walk` loop below, so a caller (the UI) can
    pump its event loop and show a running count during a slow scan. Not called during the
    dedup/sort/questionnaire-attach step that follows -- default None leaves behavior unchanged.

    A test_id collision (e.g. a loose `well1.csv` in a customer folder alongside a
    `well1/well1.csv` subfolder of the same name) is resolved in favor of the deeper folder --
    it is the richer layout -- with a warning attached to the surviving entry and the shallower
    one dropped. Duplicate iids would otherwise crash the folder-mode queue Treeview (insert with
    a repeated iid)."""
    entries: list[TestEntry] = []
    dirs_scanned = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.endswith(" DFIT plots")]
        entries.extend(_entries_for_dir(root, dirpath, filenames))
        dirs_scanned += 1
        if progress is not None:
            progress(dirs_scanned, len(entries))

    by_id: dict[str, TestEntry] = {}
    for entry in entries:
        existing = by_id.get(entry.test_id)
        if existing is None:
            by_id[entry.test_id] = entry
            continue
        # Same test_id from two different folders: the deeper one wins.
        shallow, deep = sorted((existing, entry), key=lambda e: e.folder.count(os.sep))
        deep.scan_warnings.append(
            f"loose file {os.path.basename(shallow.csv_path or shallow.dbs_path)!r} in "
            f"{shallow.folder!r} ignored: test folder {entry.test_id!r} has the same name"
        )
        by_id[entry.test_id] = deep

    entries = list(by_id.values())
    for entry in entries:
        data_path = entry.csv_path or entry.dbs_path
        entry.questionnaire_path, warns = find_questionnaire(data_path)
        entry.scan_warnings.extend(warns)

    return sorted(entries, key=lambda e: e.test_id)


# --------------------------------------------------------------------------------------------------
# picks persistence
# --------------------------------------------------------------------------------------------------
def load_picks_for(entry: TestEntry) -> Optional[PickState]:
    """The saved PickState for `entry`, or None if there is no picks file, or it exists but is
    unreadable/corrupt -- a broken JSON must never kill a folder scan."""
    try:
        return PickState.from_json(entry.picks_path)
    except Exception:
        return None


def save_picks_for(entry: TestEntry, state: PickState) -> None:
    """Write `state` to `entry.picks_path` atomically: a temp file in the same directory, then
    an os.replace onto the final path, so a crash mid-write never leaves a half-written picks
    file behind."""
    folder = entry.folder
    os.makedirs(folder, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=folder, prefix=".dfit_picks_", suffix=".tmp")
    os.close(fd)
    try:
        state.to_json(tmp_path)
        os.replace(tmp_path, entry.picks_path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


# --------------------------------------------------------------------------------------------------
# status
# --------------------------------------------------------------------------------------------------
def status_for(state: Optional[PickState]) -> str:
    """The folder-mode status for `state`: "new" (no picks / never visited a step), "done",
    "in_progress", or "skipped" -- the latter two only reachable through explicit_status, the
    user override from the Mark combobox."""
    if state is None:
        return "new"
    if state.explicit_status in ("done", "skipped"):
        return state.explicit_status
    if not any(k in state.step_status for k in STEP_KEYS):
        return "new"

    def _step_complete(key: str) -> bool:
        if state.step_status.get(key) in ("done", "skipped"):
            return True
        # PC-F ("no peak") skips the pore-pressure step end to end, so it never gets a
        # step_status entry -- without this, a PC-F test could never reach "done".
        if key == "porepressure" and model.porepressure_skipped(state):
            return True
        return False

    if all(_step_complete(k) for k in STEP_KEYS):
        return "done"
    return "in_progress"


# --------------------------------------------------------------------------------------------------
# master log
# --------------------------------------------------------------------------------------------------
def load_log(root: str) -> pd.DataFrame:
    """The master log at `<root>/dfit_log.csv`, or an empty DataFrame shaped like LOG_COLUMNS if
    it doesn't exist yet -- or exists but is empty/corrupt/unparseable. Never raises, same
    contract as `load_picks_for`: a bad dfit_log.csv must never make a folder unopenable. An
    older log missing newer columns gets them appended (empty), with existing row data
    preserved; the returned column order is always LOG_COLUMNS. `test_id` is forced to a string
    dtype -- otherwise a purely numeric test_id (e.g. a folder named "7170") round-trips as
    int64, and `upsert_log_row`'s string-keyed comparison never matches, silently appending a
    duplicate row on every save instead of updating."""
    path = os.path.join(root, LOG_FILENAME)
    if not os.path.exists(path):
        return pd.DataFrame(columns=LOG_COLUMNS)
    try:
        df = pd.read_csv(path, dtype={"test_id": str})
    except Exception:
        # Bare except like load_picks_for: encoding corruption (UnicodeDecodeError) and OS-level
        # read errors must not make the folder unopenable any more than a parse error does.
        return pd.DataFrame(columns=LOG_COLUMNS)
    for col in LOG_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[LOG_COLUMNS]


def save_log(root: str, df: pd.DataFrame) -> None:
    """Write `df` to `<root>/dfit_log.csv` atomically (temp file + os.replace)."""
    path = os.path.join(root, LOG_FILENAME)
    fd, tmp_path = tempfile.mkstemp(dir=root, prefix=".dfit_log_", suffix=".tmp")
    os.close(fd)
    try:
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, path)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise


def upsert_log_row(df: pd.DataFrame, row: dict) -> pd.DataFrame:
    """Replace the row keyed on `row["test_id"]` if one exists, else append `row`. Returns a new
    DataFrame; does not mutate `df` in place."""
    test_id = row["test_id"]
    df = df.copy()
    mask = df["test_id"] == test_id
    if mask.any():
        idx = df.index[mask][0]
        for col, value in row.items():
            df.at[idx, col] = value
        return df
    new_row = pd.DataFrame([row])
    return pd.concat([df, new_row], ignore_index=True)


def list_tests(root: str, progress=None) -> tuple[list[TestEntry], pd.DataFrame]:
    """`scan_root(root, progress=progress)` with each entry's `status` recomputed from its picks
    file, plus `load_log(root)`. Log rows whose test_id matches no scanned entry are kept as-is
    -- the UI can detect these orphans by comparing the df's test_ids to the returned entries.

    `progress`, if given, is passed straight through to `scan_root` (see its docstring); default
    None leaves behavior unchanged."""
    entries = scan_root(root, progress=progress)
    for entry in entries:
        entry.status = status_for(load_picks_for(entry))
    return entries, load_log(root)


# --------------------------------------------------------------------------------------------------
# build_log_row
# --------------------------------------------------------------------------------------------------
def build_log_row(entry: TestEntry, active_path: str, root: str, state: PickState,
                   td, res: model.DerivedResults) -> dict:
    """One LOG_COLUMNS-shaped dict for `entry`, stamped with the current user/date. `td` is the
    loaded io_load.TestData, `res` the model.DerivedResults for `state` -- every interpreted
    value comes from `res`/`state`; this function only formats and maps, it computes nothing."""

    def _minutes(seconds: Optional[float]) -> Optional[float]:
        return seconds / 60.0 if seconds is not None else None

    closure_scenario = state.closure_scenario or ""
    postclosure_scenario = state.postclosure_scenario or ""
    t_start_inj = float(td.t_s[state.start_idx]) if state.start_idx is not None else None

    return {
        "file": os.path.relpath(active_path, root),
        "test_id": entry.test_id,
        "well_name": state.well_name,
        "formation": state.formation,
        "status": status_for(state),
        "interpreter": getpass.getuser(),
        "review_date": datetime.date.today().isoformat(),
        "orientation": "",
        "fluid_type": "",
        "play": "",
        "pressure_source": "BHP" if res.pressure_is_bhp else "WHP",
        "tvd": state.tvd_ft,
        "fluid_density": state.density_ppg,
        "t_start_inj": t_start_inj,
        "t_shutin": res.t_shutin_s,
        "te": res.te_s,
        "Vinj": res.vinj,
        "max_rate": res.qmax_bpm,
        "apparent_ISIP": res.apparent_isip,
        "effective_ISIP": res.effective_isip_compliance,
        "closure_scenario": state.closure_scenario,
        "closure_quality": _CLOSURE_QUALITY_BY_PREFIX.get(closure_scenario[:3], ""),
        "contact_pressure": res.contact_pressure,
        "Shmin_compliance": res.shmin_compliance,
        "Shmin_tangent": res.shmin_tangent,
        "tangent_Gc": state.closure_G,
        "postclosure_scenario": state.postclosure_scenario,
        "postclosure_trend": _POSTCLOSURE_TREND_BY_PREFIX.get(postclosure_scenario[:4], ""),
        "pore_pressure": res.pore_pressure,
        "pp_axis": state.pp_axis,
        "pp_confidence": "low" if postclosure_scenario.startswith(("PC-E", "PC-F")) else "",
        "net_pressure_compliance": res.net_pressure_compliance,
        "net_pressure_tangent": res.net_pressure_tangent,
        "delta_closure": res.delta_closure,
        "notes": state.notes,
        "effective_ISIP_tangent": res.effective_isip_tangent,
        "effective_ISIP_variable": res.effective_isip_variable,
        "Shmin_variable": res.shmin_variable,
        "Shmin_rapid": res.shmin_rapid,
        "net_pressure_variable": res.net_pressure_variable,
        "closure_time_compliance_min": _minutes(res.closure_time_compliance_s),
        "closure_time_tangent_min": _minutes(res.closure_time_tangent_s),
        "closure_time_variable_min": _minutes(res.closure_time_variable_s),
        "net_pressure_isip_source": res.net_pressure_isip_source,
    }
