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
    "file", "test_id", "status", "interpreter", "review_date",
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
    test_id: str                 # immediate-child dir name, or file stem for flat layout
    folder: str                  # dir containing the data files
    csv_path: Optional[str] = None
    dbs_path: Optional[str] = None
    questionnaire_path: Optional[str] = None
    scan_warnings: list[str] = field(default_factory=list)
    status: str = "new"          # recomputed from picks JSON, never trusted from the log CSV

    @property
    def picks_path(self) -> str:
        return os.path.join(self.folder, self.test_id + PICKS_SUFFIX)

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
# scan_root: depth-1 scan of an opened root -- subfolder tests and flat-layout loose files.
# A stated limitation: nested subfolders (depth 2+) are never scanned.
# --------------------------------------------------------------------------------------------------
def _pick_first(names: list[str], ext_label: str, dirpath: str, warnings: list[str]) -> str:
    """Alphabetically-first (case-insensitive) name among `names`; appends a warning to
    `warnings` if there was more than one."""
    ordered = sorted(names, key=str.lower)
    if len(ordered) > 1:
        warnings.append(
            f"multiple {ext_label} files found in {dirpath!r}; using {ordered[0]!r}"
        )
    return ordered[0]


def _scan_dir(test_id: str, dirpath: str) -> Optional[TestEntry]:
    """One candidate test from an immediate subdirectory, or None if it holds no data files."""
    try:
        names = os.listdir(dirpath)
    except OSError:
        return None
    csvs = [n for n in names if n.lower().endswith(".csv")]
    dbss = [n for n in names if n.lower().endswith(".dbs")]
    if not csvs and not dbss:
        return None
    entry = TestEntry(test_id=test_id, folder=dirpath)
    if csvs:
        entry.csv_path = os.path.join(dirpath, _pick_first(csvs, "CSV", dirpath, entry.scan_warnings))
    if dbss:
        entry.dbs_path = os.path.join(dirpath, _pick_first(dbss, "DBS", dirpath, entry.scan_warnings))
    return entry


def _scan_flat(root: str, names: list[str]) -> list[TestEntry]:
    """Loose *.csv/*.dbs files directly in `root`: one flat-layout test per file stem, with
    same-stem csv+dbs merged into a single entry. dfit_log.csv is excluded (it is our own log,
    not a test); the check is case-insensitive on the filename."""
    by_stem: dict[str, dict[str, list[str]]] = {}
    for name in names:
        full = os.path.join(root, name)
        if not os.path.isfile(full):
            continue
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
        by_stem.setdefault(stem, {}).setdefault(ext, []).append(name)

    entries = []
    for stem, by_ext in by_stem.items():
        entry = TestEntry(test_id=stem, folder=root)
        if "csv" in by_ext:
            entry.csv_path = os.path.join(root, _pick_first(by_ext["csv"], "CSV", root, entry.scan_warnings))
        if "dbs" in by_ext:
            entry.dbs_path = os.path.join(root, _pick_first(by_ext["dbs"], "DBS", root, entry.scan_warnings))
        entries.append(entry)
    return entries


def scan_root(root: str) -> list[TestEntry]:
    """Every candidate test directly under `root`: one entry per immediate subdirectory that
    holds data files, plus one per loose data file (or same-stem csv+dbs pair) directly in
    `root`. Everything else (other extensions, dfit_log.csv, empty subdirectories) is ignored.
    Attaches each entry's questionnaire (via ``find_questionnaire``) and returns entries sorted
    by test_id."""
    names = os.listdir(root)
    entries: list[TestEntry] = []
    for name in names:
        full = os.path.join(root, name)
        if os.path.isdir(full):
            entry = _scan_dir(name, full)
            if entry is not None:
                entries.append(entry)
    entries.extend(_scan_flat(root, names))

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
    it doesn't exist yet. An older log missing newer columns gets them appended (empty), with
    existing row data preserved; the returned column order is always LOG_COLUMNS."""
    path = os.path.join(root, LOG_FILENAME)
    if not os.path.exists(path):
        return pd.DataFrame(columns=LOG_COLUMNS)
    df = pd.read_csv(path)
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


def list_tests(root: str) -> tuple[list[TestEntry], pd.DataFrame]:
    """`scan_root(root)` with each entry's `status` recomputed from its picks file, plus
    `load_log(root)`. Log rows whose test_id matches no scanned entry are kept as-is -- the UI
    can detect these orphans by comparing the df's test_ids to the returned entries."""
    entries = scan_root(root)
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
    }
