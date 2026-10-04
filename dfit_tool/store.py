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

from . import io_load, model
from .model import STEP_KEYS, PickState, skipped_steps
from .questionnaire import find_questionnaire, is_questionnaire_filename

LOG_FILENAME = "dfit_log.csv"      # lives at <opened_root>/dfit_log.csv
PICKS_SUFFIX = ".dfit_picks.json"  # <test_folder>/<test_id>.dfit_picks.json

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
    # appended: values the tool computes beyond the original schema
    "effective_ISIP_tangent", "effective_ISIP_variable",
    "Shmin_variable", "Shmin_rapid", "net_pressure_variable",
    "closure_time_compliance_min", "closure_time_tangent_min",
    "closure_time_variable_min",
    "net_pressure_isip_source",
    "near_wellbore_complexity",
    "tail_trim_s",
    "tail_trim_reason",
    "Shmin_liberty",
    "units_note",
    # Depth-normalized (psi/ft) forms of the pressures above, each strictly its own psi column /
    # tvd -- Shmin_rapid_gradient sits here with the other Shmin gradients rather than adjacent to
    # Shmin_rapid, since the append-only convention forbids inserting mid-list.
    "apparent_ISIP_gradient",
    "Shmin_compliance_gradient",
    "Shmin_variable_gradient",
    "Shmin_tangent_gradient",
    "Shmin_liberty_gradient",
    "Shmin_rapid_gradient",
    "pore_pressure_gradient",
    # Relative-stiffness comparison-only Shmin (URTeC-2019-123 A.8/A.9, the "stiffness" step) --
    # tail-appended per the append-only convention rather than sitting with the other Shmin
    # gradients above.
    "Shmin_stiffness",
    "Shmin_stiffness_gradient",
    # "No slope change apparent" negative finding (stiffness step) -- tail-appended per the
    # append-only convention.
    "stiffness_no_upturn",
    # Deliberate drag of the Overview tail-trim line past the rise guard's boundary (see
    # PickState.tail_guard_override) -- tail-appended per the append-only convention rather than
    # sitting next to tail_trim_reason above.
    "tail_guard_override",
    # "Tangent closure uninterpretable" negative finding (tangent step) -- tail-appended per the
    # append-only convention.
    "tangent_uninterpretable",
    # How the apparent ISIP was taken: "tangent" or "shutin" (BHP at the shut-in sample) --
    # tail-appended per the append-only convention.
    "apparent_isip_method",
    # True when the postclosure scenario is the auto-assigned PC-A (picks.auto_assign_postclosure)
    # -- tail-appended per the append-only convention.
    "postclosure_auto",
    # Counts of masked-pressure sources (resample.detect_dropouts, resample.detect_rise_excursions,
    # PickState.mask_intervals / keep_intervals) -- tail-appended per the append-only convention.
    "dropouts_masked",
    "rises_masked",
    "manual_masks",
    "manual_keeps",
]

_CLOSURE_QUALITY_BY_PREFIX = {
    "C-A": "clear", "C-B": "adequate", "C-C": "no-contact", "C-D": "rapid",
    "C-X": "uninterpretable",
}
_POSTCLOSURE_TREND_BY_PREFIX = {
    "PC-A": "linear", "PC-B": "false-radial", "PC-C": "mixed", "PC-D": "mixed",
    "PC-E": "none", "PC-F": "none", "PC-X": "uninterpretable",
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
    xlsx_path: Optional[str] = None
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
        if self.xlsx_path:
            sources.append("XLSX")
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
        if s == "XLSX":
            if not self.xlsx_path:
                raise ValueError(f"no XLSX available for test {self.test_id!r}")
            return self.xlsx_path
        raise ValueError(f"unknown source {source!r} (expected 'CSV', 'DBS', or 'XLSX')")


# --------------------------------------------------------------------------------------------------
# scan_root: arbitrary-depth scan of an opened root -- every directory under it (including the
# root itself) that holds data files becomes one or more tests, keyed by a path-qualified,
# forward-slash-separated test_id unique within the root.
# --------------------------------------------------------------------------------------------------
def _group_data_files(filenames: list[str], dirpath: str) -> dict[str, dict[str, str]]:
    """Group one directory's filenames by stem: `{stem: {"csv": name, "dbs": name, "xlsx":
    name}}`. Files that are not `.csv`/`.dbs`/`.xlsx` (case-insensitive) are skipped, as is
    `LOG_FILENAME` (case-insensitive, it is our own log, not a test).

    An `.xlsx` file has three more gates before it counts as a data file at all: it must not be
    an Excel lock file (its name starts with `"~$"`, checked explicitly here -- NOT covered by
    `is_questionnaire_filename`, which only ever returns True for a name that also contains
    "questionnaire"; a lock file almost never does, so without this explicit check one would fall
    through to the sniff below and get opened for nothing before failing to parse as a workbook
    at all), must not be a questionnaire (`is_questionnaire_filename`), and must actually sniff
    as time-series data (`io_load.sniff_xlsx_data`, which opens and peeks the workbook, hence
    `dirpath` being needed here at all). The corpus has roughly 640 non-questionnaire `.xlsx`
    files that are summaries, casing tallies, completion calcs, pump schedules, or production
    tallies rather than DFIT records; the sniff is what keeps those out of the queue. `.csv`/
    `.dbs` need no such gate -- every file of either extension in the corpus is a data file.

    Filenames within one directory are unique, so each (stem, ext) maps to exactly one name --
    no "pick first" ambiguity to warn about."""
    by_stem: dict[str, dict[str, str]] = {}
    for name in filenames:
        if name.lower() == LOG_FILENAME.lower():
            continue
        low = name.lower()
        if low.endswith(".csv"):
            ext = "csv"
        elif low.endswith(".dbs"):
            ext = "dbs"
        elif low.endswith(".xlsx"):
            if name.startswith("~$"):
                continue
            if is_questionnaire_filename(name):
                continue
            if not io_load.sniff_xlsx_data(os.path.join(dirpath, name)):
                continue
            ext = "xlsx"
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
    regardless of how test_id is qualified.

    The "single stem group" collapse test counts only csv/dbs stem groups whenever the directory
    has any at all -- an xlsx-only stem group never participates in that count, and never
    collapses to `rel` itself while a csv/dbs group is also present. Adding a data `.xlsx` must
    never change the test_id of an existing csv/dbs test: an `.xlsx` whose stem matches a csv/dbs
    group simply joins that group as its XLSX source (``_group_data_files`` already does this,
    unconditionally, by stem), so it can only ever affect a group's test_id by changing how many
    stem groups the directory has -- which is exactly what this carve-out prevents. A directory
    with no csv/dbs groups at all (xlsx-only) keeps the original one-stem-total rule, unchanged.
    The extra xlsx-only stems in a directory that also has a csv/dbs group each get their own
    `rel/<stem>` id, same as the "multiple stem groups" branch below -- since every id in
    `by_stem` is keyed on a distinct `stem` within this one directory, `rel/<stem>` can never
    collide with another entry from the SAME `_entries_for_dir` call (the collapsed group's own
    `rel` has no stem suffix at all, so it can't collide with one either); a cross-directory
    collision is a pre-existing, separately-handled case (`scan_root`'s deeper-folder-wins dedup).
    """
    by_stem = _group_data_files(filenames, dirpath)
    if not by_stem:
        return []
    rel = os.path.relpath(dirpath, root).replace(os.sep, "/")
    csv_dbs_stems = [s for s, by_ext in by_stem.items() if "csv" in by_ext or "dbs" in by_ext]
    if csv_dbs_stems:
        collapse_stem = csv_dbs_stems[0] if len(csv_dbs_stems) == 1 else None
    else:
        collapse_stem = next(iter(by_stem)) if len(by_stem) == 1 else None
    entries = []
    for stem, by_ext in by_stem.items():
        if rel == ".":
            test_id = stem
        elif stem == collapse_stem:
            test_id = rel
        else:
            test_id = f"{rel}/{stem}"
        entry = TestEntry(test_id=test_id, folder=dirpath, picks_basename=stem)
        if "csv" in by_ext:
            entry.csv_path = os.path.join(dirpath, by_ext["csv"])
        if "dbs" in by_ext:
            entry.dbs_path = os.path.join(dirpath, by_ext["dbs"])
        if "xlsx" in by_ext:
            entry.xlsx_path = os.path.join(dirpath, by_ext["xlsx"])
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
    a repeated iid). The exception is a collision between a csv/dbs entry and an xlsx-only entry:
    the csv/dbs entry keeps the id whatever the depth, and the xlsx-only one is renamed
    ``<id>/<stem>`` (made unique) with a warning, so adding a data .xlsx never displaces an
    existing csv/dbs test."""
    entries: list[TestEntry] = []
    dirs_scanned = 0
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.endswith(" DFIT plots")]
        entries.extend(_entries_for_dir(root, dirpath, filenames))
        dirs_scanned += 1
        if progress is not None:
            progress(dirs_scanned, len(entries))

    by_id: dict[str, TestEntry] = {}
    used_ids = {e.test_id for e in entries}
    for entry in entries:
        existing = by_id.get(entry.test_id)
        if existing is None:
            by_id[entry.test_id] = entry
            continue
        # csv/dbs vs xlsx-only: the csv/dbs entry keeps the id (adding a data .xlsx must never
        # change or displace an existing csv/dbs test); the xlsx-only one gets a distinct,
        # unique id "<id>/<stem>" plus a warning.
        x_only = lambda e: not (e.csv_path or e.dbs_path)
        if x_only(existing) != x_only(entry):
            keeper, moved = (entry, existing) if x_only(existing) else (existing, entry)
            new_id = f"{moved.test_id}/{moved.picks_basename}"
            n = 2
            while new_id in used_ids:
                new_id = f"{moved.test_id}/{moved.picks_basename} ({n})"
                n += 1
            used_ids.add(new_id)
            moved.scan_warnings.append(
                f"xlsx-only test renamed {new_id!r}: id {moved.test_id!r} is taken by "
                f"{os.path.basename(keeper.csv_path or keeper.dbs_path)!r} in "
                f"{keeper.folder!r}"
            )
            moved.test_id = new_id
            by_id[keeper.test_id] = keeper
            by_id[new_id] = moved
            continue
        # Same test_id from two different folders: the deeper one wins.
        shallow, deep = sorted((existing, entry), key=lambda e: e.folder.count(os.sep))
        deep.scan_warnings.append(
            f"loose file "
            f"{os.path.basename(shallow.csv_path or shallow.dbs_path or shallow.xlsx_path)!r} "
            f"in {shallow.folder!r} ignored: test folder {entry.test_id!r} has the same name"
        )
        by_id[entry.test_id] = deep

    entries = list(by_id.values())
    for entry in entries:
        data_path = entry.csv_path or entry.dbs_path or entry.xlsx_path
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
    """The folder-mode status for `state`: "new" (no picks / never visited a step), "done" and
    "in_progress" are purely derived from step_status -- all eight steps accounted for and none
    skipped is "done", all accounted for with >=1 skipped is "skipped", otherwise
    "in_progress" -- except "skipped" can also come from explicit_status, the whole-test
    Skip-test button, which overrides the derivation at any point in the workflow regardless of
    how far the steps got."""
    if state is None:
        return "new"
    if state.explicit_status == "skipped":  # the whole-test Skip button, any point in the workflow
        return "skipped"
    if not any(k in state.step_status for k in STEP_KEYS):
        return "new"
    # A step the workflow leaves out (skipped_steps; PC-F drops porepressure and stiffness) may
    # have no step_status entry at all, and any entry it does have (from a session where the
    # analyst hit Skip before choosing PC-F) is not a user decision worth reporting.
    keys = [k for k in STEP_KEYS if k not in skipped_steps(state)]
    if not all(state.step_status.get(k) in ("done", "skipped") for k in keys):
        return "in_progress"
    return "skipped" if any(state.step_status.get(k) == "skipped" for k in keys) else "done"


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
        # A suppressed pick (tangent_uninterpretable) is kept in the picks JSON so unchecking
        # restores it, but must not read as a real closure pick in the log.
        "tangent_Gc": None if state.tangent_uninterpretable else state.closure_G,
        "postclosure_scenario": state.postclosure_scenario,
        "postclosure_trend": _POSTCLOSURE_TREND_BY_PREFIX.get(postclosure_scenario[:4], ""),
        "pore_pressure": res.pore_pressure,
        "pp_axis": state.pp_axis,
        "pp_confidence": "low" if postclosure_scenario.startswith(("PC-E", "PC-F", "PC-X")) else "",
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
        "near_wellbore_complexity": res.near_wellbore_complexity,
        "tail_trim_s": state.tail_trim_dt,
        "tail_trim_reason": state.tail_trim_reason,
        "Shmin_liberty": res.shmin_liberty,
        "units_note": res.unit_conversion_note,
        "apparent_ISIP_gradient": res.apparent_isip_gradient,
        "Shmin_compliance_gradient": res.shmin_compliance_gradient,
        "Shmin_variable_gradient": res.shmin_variable_gradient,
        "Shmin_tangent_gradient": res.shmin_tangent_gradient,
        "Shmin_liberty_gradient": res.shmin_liberty_gradient,
        "Shmin_rapid_gradient": res.shmin_rapid_gradient,
        "pore_pressure_gradient": res.pore_pressure_gradient,
        "Shmin_stiffness": res.shmin_stiffness,
        "Shmin_stiffness_gradient": res.shmin_stiffness_gradient,
        # Blank, not the stored flag, once PC-F (or PC-X) skips the stiffness step: an analyst can check
        # "No slope change apparent" on the stiffness step, go Back, and switch to PC-F, which
        # makes the step unreachable (so the checkbox can never be unchecked again). The flag
        # itself is left alone in the picks JSON on purpose -- switching back off PC-F revives
        # it -- mirroring how a stale pp_window pick is suppressed, not cleared, under PC-F (see
        # model.py's pore-pressure block comment). Blank here matches every other stiffness
        # output going blank for a test whose stiffness curve never existed under PC-F.
        "stiffness_no_upturn": ("" if "stiffness" in skipped_steps(state)
                                else state.stiffness_no_upturn),
        "tail_guard_override": state.tail_guard_override,
        "tangent_uninterpretable": state.tangent_uninterpretable,
        "apparent_isip_method": res.apparent_isip_method,
        "postclosure_auto": state.postclosure_auto,
        "dropouts_masked": len(res.dropouts),
        "rises_masked": len(res.rise_excursions),
        "manual_masks": len(state.mask_intervals),
        "manual_keeps": len(state.keep_intervals),
    }
