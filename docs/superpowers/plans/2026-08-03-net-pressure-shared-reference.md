# TODO #4: Slim eff-ISIP sidebar + shared net-pressure reference — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove tangent/variable effective-ISIP rows from the sidebar (keep them in the CSV log), and make every net-pressure calculation reference one shared effective ISIP (compliance → tangent → none) with the source recorded in the log.

**Architecture:** All computation stays in `model.compute_all` (single source of truth). One shared reference ISIP replaces the three per-method `ref_*` locals; a new non-serialized `DerivedResults.net_pressure_isip_source` records which source fed it. `ui.py` only removes two display rows. `store.py` gains one log column.

**Tech Stack:** Python 3.14, numpy/pandas, pytest. Project venv: `C:\Users\LucasChristman\.venvs\dfit\Scripts\python.exe`. Run tests from repo root with `-m pytest`.

## Global Constraints

- `compute_all` is the single source of truth: net-pressure logic and the source string are computed there, never in the UI.
- `DerivedResults` is never serialized — the new field needs no migration and no `PickState` change.
- Keep `picks.py`/`plots.py`/`sliders.py` Tkinter-free; this plan does not touch them.
- No apparent-ISIP fallback for net pressure (removing the current one).
- Preserve the existing per-method Shmin guards on each net pressure.
- Append new CSV columns to the end of `LOG_COLUMNS` (documented extension convention).

---

### Task 1: Shared net-pressure reference + source field in `compute_all`

**Files:**
- Modify: `dfit_tool/model.py` — add field at `DerivedResults` (after line 229, `net_pressure_variable`); replace net-pressure block at lines 357-370.
- Test: `tests/test_net_pressure_reference.py` (create)

**Interfaces:**
- Consumes: existing `DerivedResults` fields `effective_isip_compliance`, `effective_isip_tangent`, `shmin_compliance`, `shmin_tangent`, `shmin_variable`; `interpret.net_pressure(reference_isip, shmin)`.
- Produces: `DerivedResults.net_pressure_isip_source: Optional[str]` — `"compliance"`, `"tangent"`, or `""`. `net_pressure_compliance`/`net_pressure_tangent`/`net_pressure_variable` now all reference the shared ISIP.

- [ ] **Step 1: Write the failing test**

Create `tests/test_net_pressure_reference.py`. These tests build a `DerivedResults`-producing state via the existing helpers, then assert the shared-reference semantics. Use the same construction pattern other model tests use (check `tests/` for an existing `compute_all` test to copy the state-building boilerplate — e.g. `test_model.py` or similar; reuse `helpers.make_testdata`).

```python
import numpy as np
from dfit_tool import model, interpret
from dfit_tool.model import DerivedResults


def _res(**kw):
    """A DerivedResults with only the fields the net-pressure block reads."""
    return DerivedResults(**kw)


def test_source_compliance_when_compliance_present():
    # Mirror compute_all's net-pressure block by calling the real function under test.
    # Compliance eff ISIP present -> shared ref is compliance, source "compliance".
    r = model._resolve_net_pressures(
        _res(effective_isip_compliance=9000.0, effective_isip_tangent=8800.0,
             shmin_compliance=7000.0, shmin_tangent=6900.0, shmin_variable=6950.0))
    assert r.net_pressure_isip_source == "compliance"
    assert r.net_pressure_compliance == interpret.net_pressure(9000.0, 7000.0)
    assert r.net_pressure_tangent == interpret.net_pressure(9000.0, 6900.0)
    assert r.net_pressure_variable == interpret.net_pressure(9000.0, 6950.0)


def test_source_tangent_when_compliance_cleared():
    # C-C style: no contact pick -> no compliance eff ISIP and no shmin_compliance/variable.
    r = model._resolve_net_pressures(
        _res(effective_isip_compliance=None, effective_isip_tangent=8800.0,
             shmin_compliance=None, shmin_tangent=6900.0, shmin_variable=None))
    assert r.net_pressure_isip_source == "tangent"
    assert r.net_pressure_compliance is None
    assert r.net_pressure_tangent == interpret.net_pressure(8800.0, 6900.0)
    assert r.net_pressure_variable is None


def test_source_blank_when_both_cleared():
    r = model._resolve_net_pressures(
        _res(effective_isip_compliance=None, effective_isip_tangent=None,
             shmin_compliance=None, shmin_tangent=6900.0, shmin_variable=None))
    assert r.net_pressure_isip_source == ""
    assert r.net_pressure_compliance is None
    assert r.net_pressure_tangent is None
    assert r.net_pressure_variable is None
```

Note: this test targets a small extracted helper `model._resolve_net_pressures(res)` that mutates and returns `res`. Extract it in Step 3 so both the test and `compute_all` call the same code.

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\Users\LucasChristman\.venvs\dfit\Scripts\python.exe -m pytest tests/test_net_pressure_reference.py -v`
Expected: FAIL — `AttributeError: module 'dfit_tool.model' has no attribute '_resolve_net_pressures'`.

- [ ] **Step 3: Add the field and extract the shared-reference helper**

In `dfit_tool/model.py`, add the field to `DerivedResults` immediately after `net_pressure_variable` (line 229):

```python
    net_pressure_variable: Optional[float] = None
    # Which effective-ISIP source fed the shared net-pressure reference:
    # "compliance", "tangent", or "" when no reference was available.
    net_pressure_isip_source: Optional[str] = None
```

Then replace the net-pressure block in `compute_all` (currently lines 357-370, the `ref_compliance`/`ref_tangent`/`ref_variable` assignments through the three `net_pressure_*` assignments — but NOT the `delta_closure` lines 371-372) with a single call:

```python
    _resolve_net_pressures(res)
    if res.shmin_compliance is not None and res.shmin_tangent is not None:
        res.delta_closure = res.shmin_compliance - res.shmin_tangent
```

Add the helper as a module-level function (place it just above `compute_all`):

```python
def _resolve_net_pressures(res: "DerivedResults") -> "DerivedResults":
    """Set net_pressure_* and net_pressure_isip_source on ``res`` from a single shared
    reference ISIP: compliance eff ISIP, else tangent eff ISIP, else none (no apparent-ISIP
    fallback). Each net pressure keeps its own per-method Shmin guard, so it stays None when
    the shared reference is None or its own Shmin is None."""
    if res.effective_isip_compliance is not None:
        ref, res.net_pressure_isip_source = res.effective_isip_compliance, "compliance"
    elif res.effective_isip_tangent is not None:
        ref, res.net_pressure_isip_source = res.effective_isip_tangent, "tangent"
    else:
        ref, res.net_pressure_isip_source = None, ""
    if ref is not None:
        if res.shmin_compliance is not None:
            res.net_pressure_compliance = interpret.net_pressure(ref, res.shmin_compliance)
        if res.shmin_tangent is not None:
            res.net_pressure_tangent = interpret.net_pressure(ref, res.shmin_tangent)
        if res.shmin_variable is not None:
            res.net_pressure_variable = interpret.net_pressure(ref, res.shmin_variable)
    return res
```

Confirm the old comment block (lines 357-358, "Net pressures: each method references its own effective ISIP...") is removed along with the replaced code.

- [ ] **Step 4: Run tests to verify they pass**

Run: `C:\Users\LucasChristman\.venvs\dfit\Scripts\python.exe -m pytest tests/test_net_pressure_reference.py -v`
Expected: 3 passed.

- [ ] **Step 5: Run the full suite (guard against regressions in existing net-pressure tests)**

Run: `C:\Users\LucasChristman\.venvs\dfit\Scripts\python.exe -m pytest`
Expected: all pass. If an existing test asserted the old apparent-ISIP fallback (compliance net pressure computed from apparent ISIP when compliance eff ISIP absent), update it to the new semantics: net pressure is `None` when the shared reference is `None`. Show the diff of any such change in the commit.

- [ ] **Step 6: Commit**

```bash
git add dfit_tool/model.py tests/test_net_pressure_reference.py
git commit -m "TODO #4: shared net-pressure reference (compliance->tangent) + source field"
```

---

### Task 2: Add `net_pressure_isip_source` to the CSV log

**Files:**
- Modify: `dfit_tool/store.py:35-50` (`LOG_COLUMNS`) and the `build_log_row` dict (append near line 394).
- Test: `tests/test_store_log.py` (add a case; if no such file exists, create `tests/test_log_source_column.py`).

**Interfaces:**
- Consumes: `res.net_pressure_isip_source` from Task 1.
- Produces: `"net_pressure_isip_source"` key in `build_log_row`'s output and in `LOG_COLUMNS`.

- [ ] **Step 1: Write the failing test**

Add to the store test file. Build an `entry`/`state`/`res`/`td` the way the existing `build_log_row` tests do (copy that boilerplate; if none exists, construct a `DerivedResults(net_pressure_isip_source="compliance")` and a minimal `PickState`/`TestEntry` matching `build_log_row`'s signature). Assert:

```python
def test_log_row_has_net_pressure_isip_source():
    row = store.build_log_row(entry, active_path, root, state, res, td)
    assert "net_pressure_isip_source" in store.LOG_COLUMNS
    assert row["net_pressure_isip_source"] == "compliance"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\Users\LucasChristman\.venvs\dfit\Scripts\python.exe -m pytest tests/test_store_log.py::test_log_row_has_net_pressure_isip_source -v`
Expected: FAIL — `KeyError: 'net_pressure_isip_source'` (or `assert 'net_pressure_isip_source' in LOG_COLUMNS`).

- [ ] **Step 3: Add the column and row mapping**

In `dfit_tool/store.py`, append to `LOG_COLUMNS` (after `"closure_time_variable_min"` on line 49):

```python
    "closure_time_variable_min",
    "net_pressure_isip_source",
]
```

In `build_log_row`, add the mapping (after `"closure_time_variable_min": _minutes(...)` near line 394):

```python
        "net_pressure_isip_source": res.net_pressure_isip_source,
```

- [ ] **Step 4: Run test to verify it passes**

Run: `C:\Users\LucasChristman\.venvs\dfit\Scripts\python.exe -m pytest tests/test_store_log.py::test_log_row_has_net_pressure_isip_source -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add dfit_tool/store.py tests/test_store_log.py
git commit -m "TODO #4: add net_pressure_isip_source column to dfit_log.csv"
```

---

### Task 3: Remove the tangent/variable eff-ISIP rows from the sidebar

**Files:**
- Modify: `dfit_tool/ui.py` — `PANEL_FIELDS` (lines 63-70), `FIELD_STEP` (lines 78-99), and the `vals` dict in `_update_panel` (lines 1420-1445).
- Test: `tests/test_panel_fields.py` (create).

**Interfaces:**
- Consumes: `ui.PANEL_FIELDS`, `ui.FIELD_STEP` (module-level, importable headless — no `tk.Tk()` needed to read them).
- Produces: no new interface; two rows removed.

- [ ] **Step 1: Write the failing test**

Create `tests/test_panel_fields.py`:

```python
from dfit_tool import ui


def test_extra_eff_isip_rows_removed():
    assert "eff ISIP (compliance)" in ui.PANEL_FIELDS
    assert "eff ISIP (tangent)" not in ui.PANEL_FIELDS
    assert "eff ISIP (variable)" not in ui.PANEL_FIELDS
    assert "eff ISIP (tangent)" not in ui.FIELD_STEP
    assert "eff ISIP (variable)" not in ui.FIELD_STEP


def test_net_pressure_rows_kept():
    for row in ("net (compliance)", "net (tangent)", "net (variable)"):
        assert row in ui.PANEL_FIELDS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `C:\Users\LucasChristman\.venvs\dfit\Scripts\python.exe -m pytest tests/test_panel_fields.py -v`
Expected: `test_extra_eff_isip_rows_removed` FAILS (`eff ISIP (tangent)` still present); `test_net_pressure_rows_kept` passes.

- [ ] **Step 3: Remove the two rows in all three places**

In `dfit_tool/ui.py`:

`PANEL_FIELDS` — change line 65 from:

```python
    "eff ISIP (compliance)", "eff ISIP (tangent)", "eff ISIP (variable)",
```

to:

```python
    "eff ISIP (compliance)",
```

`FIELD_STEP` — delete these two lines (89 and 94):

```python
    "eff ISIP (tangent)": "tangent",
```
```python
    "eff ISIP (variable)": "tangent",
```

`_update_panel` `vals` dict — delete these two lines (1426-1427):

```python
            "eff ISIP (tangent)": s(r.effective_isip_tangent),
            "eff ISIP (variable)": s(r.effective_isip_variable),
```

(Leaving them in `vals` while absent from `PANEL_FIELDS`/`value_lbls` would `KeyError` in the `_update_panel` loop.)

- [ ] **Step 4: Run test to verify it passes**

Run: `C:\Users\LucasChristman\.venvs\dfit\Scripts\python.exe -m pytest tests/test_panel_fields.py -v`
Expected: both pass.

- [ ] **Step 5: Run the full suite**

Run: `C:\Users\LucasChristman\.venvs\dfit\Scripts\python.exe -m pytest`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add dfit_tool/ui.py tests/test_panel_fields.py
git commit -m "TODO #4: drop tangent/variable eff-ISIP rows from sidebar panel"
```

---

### Task 4: Update CLAUDE.md domain docs

**Files:**
- Modify: `CLAUDE.md` — the "Net pressure" paragraph in the Domain and methodology section, and the folder-mode `store.py` `build_log_row`/`LOG_COLUMNS` description if it enumerates columns.

**Interfaces:** none (docs only).

- [ ] **Step 1: Update the net-pressure definition**

Find the line in `CLAUDE.md`:

```
**Net pressure** = reference ISIP − Shmin, using each method's own effective ISIP (falling
back to apparent ISIP per method when that effective ISIP is unavailable).
```

Replace with:

```
**Net pressure** = shared reference ISIP − Shmin. All three methods (compliance, tangent,
variable) subtract their own Shmin from one shared reference ISIP: the compliance effective
ISIP, falling back to the tangent effective ISIP, else undefined (no apparent-ISIP fallback,
so a net pressure is blank when neither effective ISIP exists or that method's Shmin is
absent). `compute_all` records the source that fed the reference in
`DerivedResults.net_pressure_isip_source` ("compliance"/"tangent"/""), logged to the
`net_pressure_isip_source` column of `dfit_log.csv`. The tangent and variable effective ISIPs
are kept in the CSV log but are no longer shown in the sidebar panel.
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "TODO #4: docs for shared net-pressure reference"
```

---

## Self-Review

**Spec coverage:**
- Sidebar drop two eff-ISIP rows → Task 3 (all three edit sites: `PANEL_FIELDS`, `FIELD_STEP`, `_update_panel`).
- Shared reference chain compliance→tangent→None → Task 1 (`_resolve_net_pressures`).
- No apparent-ISIP fallback → Task 1 (explicit `else None`) + Step 5 regression note.
- `net_pressure_isip_source` field → Task 1; CSV column → Task 2.
- Keep per-method eff-ISIP CSV columns → unchanged, no task needed (already present).
- Keep three net-pressure sidebar rows → verified by Task 3 `test_net_pressure_rows_kept`.
- Docs → Task 4.

**Placeholder scan:** the only soft spot is "copy the existing test boilerplate" in Tasks 1-2 for state construction. This is deliberate — the exact `PickState`/`TestData` construction is verbose and already established in `tests/`; the executor must read one existing `compute_all`/`build_log_row` test rather than have it re-invented here. All logic-under-test code is fully specified.

**Type consistency:** `_resolve_net_pressures(res) -> DerivedResults`, field `net_pressure_isip_source: Optional[str]`, values `"compliance"`/`"tangent"`/`""` — consistent across Tasks 1, 2, 4 and both test files.
