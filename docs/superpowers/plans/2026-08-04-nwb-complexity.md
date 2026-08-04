# Near-Wellbore Complexity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Report one new per-test value, near-wellbore complexity = apparent ISIP − the shared reference effective ISIP, in the result panel and the `dfit_log.csv` master log.

**Architecture:** The math is a one-line subtraction in `interpret.py`. `model._resolve_net_pressures` already resolves the shared reference ISIP (compliance eff ISIP → tangent eff ISIP → none) for the three net pressures, so it sets the new `DerivedResults.near_wellbore_complexity` off the same reference — one function owns the reference and everything derived from it. `ui.py` gains one panel row that only displays the value; `store.py` gains one appended log column that only maps it.

**Tech Stack:** Python 3.14, numpy, pandas, pytest. Layered package `dfit_tool/` where lower layers never import higher ones: `interpret` → `model` → `{ui, store}`.

## Global Constraints

- Always use the project venv python: `C:\Users\LucasChristman\.venvs\dfit\Scripts\python.exe`. Dependencies are not on the system Python.
- Run pytest from the repo root: `C:\Users\LucasChristman\.venvs\dfit\Scripts\python.exe -m pytest`.
- `compute_all` is the single source of truth. New reported values are computed there (or in a helper it calls), never in `ui.py`. `ui.py` only displays what `compute_all` returns.
- `store.build_log_row` computes nothing itself; it maps existing `PickState`/`DerivedResults` fields into a row.
- Complexity subtracts the **shared** reference ISIP: `effective_isip_compliance`, else `effective_isip_tangent`, else undefined. There is one complexity value per test, not one per method.
- A negative complexity is reported as-is: no warning appended, no clamp to zero. Clamping would break the identity `Shmin + net pressure + complexity = apparent ISIP`.
- New log columns are **appended** to the end of `LOG_COLUMNS`, never inserted, so existing `dfit_log.csv` files stay loadable (`load_log` backfills missing columns).
- Do not rename `model._resolve_net_pressures`. Three existing tests in `tests/test_net_pressure_reference.py` call it directly.

---

### Task 1: Complexity math and computation

**Files:**
- Modify: `dfit_tool/interpret.py` — add `near_wellbore_complexity` after `net_pressure` (currently ends at line 216)
- Modify: `dfit_tool/model.py:232` — add the `DerivedResults` field after `net_pressure_isip_source`
- Modify: `dfit_tool/model.py:251-269` — `_resolve_net_pressures` sets the new field
- Test: `tests/test_nwb_complexity.py` (create)

**Interfaces:**
- Consumes: `model._resolve_net_pressures(res: DerivedResults) -> DerivedResults`, which mutates and returns `res`. It already reads `res.effective_isip_compliance`, `res.effective_isip_tangent`, `res.shmin_compliance`, `res.shmin_tangent`, `res.shmin_variable` and writes `res.net_pressure_*` and `res.net_pressure_isip_source`.
- Produces: `interpret.near_wellbore_complexity(apparent_isip: float, reference_isip: float) -> float` and `DerivedResults.near_wellbore_complexity: Optional[float]`. Tasks 2 and 3 read that field by name.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_nwb_complexity.py`:

```python
"""Unit tests for near-wellbore complexity (CLAUDE.md TODO #5): apparent ISIP minus the
shared reference effective ISIP, and the additive identity

    Shmin + net pressure + complexity = apparent ISIP

which holds for all three methods because every net pressure subtracts its own Shmin from
that same shared reference. Mirrors the direct-_resolve_net_pressures style of
tests/test_net_pressure_reference.py.
"""

import pytest

from dfit_tool import model, picks
from dfit_tool.model import DerivedResults, TangentPick, compute_all
from tests.helpers import make_testdata, overview_state


def _res(**kw):
    """A DerivedResults with only the fields the shared-reference block reads."""
    return DerivedResults(**kw)


def test_complexity_uses_compliance_reference():
    r = model._resolve_net_pressures(
        _res(apparent_isip=9500.0, effective_isip_compliance=9000.0,
             effective_isip_tangent=8800.0, shmin_compliance=7000.0))
    assert r.net_pressure_isip_source == "compliance"
    assert r.near_wellbore_complexity == pytest.approx(500.0)


def test_complexity_falls_back_to_tangent_reference():
    # C-C style: no contact pick -> no compliance eff ISIP, so the tangent eff ISIP is the
    # shared reference for both net pressure and complexity.
    r = model._resolve_net_pressures(
        _res(apparent_isip=9500.0, effective_isip_compliance=None,
             effective_isip_tangent=8800.0, shmin_tangent=6900.0))
    assert r.net_pressure_isip_source == "tangent"
    assert r.near_wellbore_complexity == pytest.approx(700.0)


def test_complexity_none_when_no_reference_isip():
    r = model._resolve_net_pressures(
        _res(apparent_isip=9500.0, effective_isip_compliance=None,
             effective_isip_tangent=None, shmin_tangent=6900.0))
    assert r.net_pressure_isip_source == ""
    assert r.near_wellbore_complexity is None


def test_complexity_none_when_no_apparent_isip():
    # The isip step has not been picked yet. Net pressure is unaffected; only complexity needs
    # the apparent ISIP.
    r = model._resolve_net_pressures(
        _res(apparent_isip=None, effective_isip_compliance=9000.0, shmin_compliance=7000.0))
    assert r.net_pressure_compliance is not None
    assert r.near_wellbore_complexity is None


def test_complexity_negative_reported_as_is_with_no_warning():
    # Apparent ISIP below the P-vs-G extrapolation is physically odd (bad ISIP tangent pick),
    # but the tool reports the arithmetic rather than clamping or warning.
    r = model._resolve_net_pressures(
        _res(apparent_isip=8960.0, effective_isip_compliance=9000.0, shmin_compliance=7000.0))
    assert r.near_wellbore_complexity == pytest.approx(-40.0)
    assert r.warnings == []


def test_identity_shmin_plus_net_plus_complexity_equals_apparent_isip():
    # End-to-end through compute_all on synthetic data: the identity must close for each of
    # the three Shmin methods.
    td = make_testdata()
    st = overview_state(td)
    res = compute_all(st, td)
    dg = res.diagnostics

    # Two distinct G-times well clear of the array edges, to anchor contact/closure at.
    contact_G = float(dg.G[dg.G.size // 4])
    closure_G = float(dg.G[3 * dg.G.size // 4])

    # apparent_isip needs a stored shut-in tangent; overview_state sets none. Anchoring at the
    # shut-in instant makes apparent_isip == anchor_y (same stand-in as
    # tests/test_variable_compliance.py::test_net_pressure_none_when_no_effective_isip_available).
    st.isip_tangent = TangentPick(anchor_x=res.t_shutin_s, anchor_y=4500.0, slope=-10.0)
    picks.commit_contact_point(st, contact_G)
    picks.commit_closure_point(st, closure_G)
    r = compute_all(st, td)

    assert r.apparent_isip is not None
    assert r.near_wellbore_complexity is not None
    for shmin, net in ((r.shmin_compliance, r.net_pressure_compliance),
                       (r.shmin_tangent, r.net_pressure_tangent),
                       (r.shmin_variable, r.net_pressure_variable)):
        assert shmin is not None
        assert net is not None
        assert shmin + net + r.near_wellbore_complexity == pytest.approx(r.apparent_isip)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `C:\Users\LucasChristman\.venvs\dfit\Scripts\python.exe -m pytest tests/test_nwb_complexity.py -v`

Expected: all 6 tests FAIL with `AttributeError: 'DerivedResults' object has no attribute 'near_wellbore_complexity'`. Every keyword `_res(...)` passes (`apparent_isip`, `effective_isip_*`, `shmin_*`) is an existing `DerivedResults` field, so construction succeeds and only the new attribute is missing.

- [ ] **Step 3: Add the math function**

In `dfit_tool/interpret.py`, immediately after `net_pressure` (which ends at line 216, just before the `# pore pressure (postclosure)` banner comment):

```python
def near_wellbore_complexity(apparent_isip: float, reference_isip: float) -> float:
    """Near-wellbore complexity = apparent ISIP - reference (effective) ISIP: the near-wellbore
    friction/tortuosity present in the early-decline extrapolation but already dissipated by the
    time the P-vs-G line is fit. Closes the identity

        Shmin + net pressure + complexity = apparent ISIP

    for every method, since each net pressure subtracts its own Shmin from that same reference.
    """
    return apparent_isip - reference_isip
```

- [ ] **Step 4: Add the `DerivedResults` field**

In `dfit_tool/model.py`, after the `net_pressure_isip_source` field (line 232) and before `delta_closure`:

```python
    # Which effective-ISIP source fed the shared net-pressure reference:
    # "compliance", "tangent", or "" when no reference was available.
    net_pressure_isip_source: Optional[str] = None
    # Apparent ISIP - that same shared reference ISIP. One value per test (not per method);
    # None when either the apparent ISIP or the reference is missing. Negative values are
    # reported as-is -- see _resolve_net_pressures.
    near_wellbore_complexity: Optional[float] = None
    delta_closure: Optional[float] = None
```

- [ ] **Step 5: Wire it into `_resolve_net_pressures`**

In `dfit_tool/model.py`, replace the whole function (lines 251-269) with:

```python
def _resolve_net_pressures(res: "DerivedResults") -> "DerivedResults":
    """Resolve the single shared reference ISIP -- compliance eff ISIP, else tangent eff ISIP,
    else none (no apparent-ISIP fallback) -- and set everything derived from it on ``res``:
    ``net_pressure_isip_source``, the three ``net_pressure_*``, and
    ``near_wellbore_complexity``. Each net pressure keeps its own per-method Shmin guard, so it
    stays None when the shared reference is None or its own Shmin is None. Complexity is
    guarded on the apparent ISIP instead, and a negative result is returned as-is (no clamp, no
    warning) so the identity Shmin + net + complexity = apparent ISIP stays exact."""
    if res.effective_isip_compliance is not None:
        ref, res.net_pressure_isip_source = res.effective_isip_compliance, "compliance"
    elif res.effective_isip_tangent is not None:
        ref, res.net_pressure_isip_source = res.effective_isip_tangent, "tangent"
    else:
        ref, res.net_pressure_isip_source = None, ""
    if ref is not None:
        if res.apparent_isip is not None:
            res.near_wellbore_complexity = interpret.near_wellbore_complexity(res.apparent_isip,
                                                                              ref)
        if res.shmin_compliance is not None:
            res.net_pressure_compliance = interpret.net_pressure(ref, res.shmin_compliance)
        if res.shmin_tangent is not None:
            res.net_pressure_tangent = interpret.net_pressure(ref, res.shmin_tangent)
        if res.shmin_variable is not None:
            res.net_pressure_variable = interpret.net_pressure(ref, res.shmin_variable)
    return res
```

`compute_all` needs no change: it already sets `res.apparent_isip` (line 313) well before its single `_resolve_net_pressures(res)` call (line 381).

- [ ] **Step 6: Run the new tests to verify they pass**

Run: `C:\Users\LucasChristman\.venvs\dfit\Scripts\python.exe -m pytest tests/test_nwb_complexity.py -v`

Expected: 6 passed.

- [ ] **Step 7: Run the full suite to check for regressions**

Run: `C:\Users\LucasChristman\.venvs\dfit\Scripts\python.exe -m pytest`

Expected: all tests pass. `tests/test_net_pressure_reference.py` in particular must stay green — the three net pressures and the source string are unchanged by this task.

- [ ] **Step 8: Commit**

```bash
git add dfit_tool/interpret.py dfit_tool/model.py tests/test_nwb_complexity.py
git commit -m "TODO #5: near-wellbore complexity in interpret + compute_all"
```

---

### Task 2: Result-panel row

**Files:**
- Modify: `dfit_tool/ui.py:61-70` — the `PANEL_FIELDS` list and its row-count comment
- Modify: `dfit_tool/ui.py:78-96` — the `FIELD_STEP` mapping
- Modify: `dfit_tool/ui.py:1418-1441` — the `vals` dict in `_update_panel`
- Test: `tests/test_panel_fields.py` (modify), `tests/test_step_status.py:103-124` (modify)

**Interfaces:**
- Consumes: `DerivedResults.near_wellbore_complexity` from Task 1.
- Produces: the panel row key string `"NWB complexity"`, present in both `ui.PANEL_FIELDS` (at index 5, directly after `"eff ISIP (compliance)"`) and `ui.FIELD_STEP` (mapped to `"gfunction"`).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_panel_fields.py`:

```python
def test_nwb_complexity_row_sits_directly_under_eff_isip():
    # The panel then reads apparent ISIP -> eff ISIP -> complexity straight down, so the
    # subtraction is visible.
    assert (ui.PANEL_FIELDS.index("NWB complexity")
            == ui.PANEL_FIELDS.index("eff ISIP (compliance)") + 1)


def test_nwb_complexity_owned_by_gfunction_step():
    # It needs both the isip and gfunction picks; gfunction is the later of the two, the same
    # precedent "net (compliance)" follows.
    assert ui.FIELD_STEP["NWB complexity"] == "gfunction"
```

In `tests/test_step_status.py`, add one entry to the hardcoded `expected` dict in
`test_field_step_mapping_matches_spec` (line 104), directly after the
`"eff ISIP (compliance)"` line:

```python
        "eff ISIP (compliance)": "gfunction",
        "NWB complexity": "gfunction",
        "contact P": "gfunction",
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `C:\Users\LucasChristman\.venvs\dfit\Scripts\python.exe -m pytest tests/test_panel_fields.py tests/test_step_status.py -v`

Expected: `test_nwb_complexity_row_sits_directly_under_eff_isip` FAILS with `ValueError: 'NWB complexity' is not in list`; `test_nwb_complexity_owned_by_gfunction_step` FAILS with `KeyError: 'NWB complexity'`; `test_field_step_mapping_matches_spec` FAILS on dict inequality; `test_field_step_covers_exactly_the_panel_fields` still passes (both collections lack the key).

- [ ] **Step 3: Add the panel row**

In `dfit_tool/ui.py`, replace lines 61-70 with:

```python
# The 19 result-panel rows, in display order -- module level (not just a literal inside
# _build_body) so FIELD_STEP below and tests can both refer to the same list.
PANEL_FIELDS = [
    "te (min)", "Vinj (bbl)", "qmax (bpm)", "apparent ISIP",
    "eff ISIP (compliance)", "NWB complexity",
    "contact P", "Shmin compliance", "Shmin tangent", "Shmin variable", "Shmin rapid",
    "tc compliance (min)", "tc tangent (min)", "tc variable (min)",
    "net (compliance)", "net (tangent)", "net (variable)",
    "delta closure", "pore pressure",
]
```

- [ ] **Step 4: Add the `FIELD_STEP` entry**

In `dfit_tool/ui.py`, in the `FIELD_STEP` dict, directly after the
`"eff ISIP (compliance)": "gfunction",` line (line 83):

```python
    "eff ISIP (compliance)": "gfunction",
    # Needs the isip pick (apparent ISIP) and the gfunction pick (the reference eff ISIP);
    # gfunction is the later of the two, same precedent as "net (compliance)".
    "NWB complexity": "gfunction",
```

- [ ] **Step 5: Display the value**

In `dfit_tool/ui.py`, in `_update_panel`'s `vals` dict, directly after the
`"eff ISIP (compliance)": s(r.effective_isip_compliance),` line (line 1423):

```python
            "eff ISIP (compliance)": s(r.effective_isip_compliance),
            "NWB complexity": s(r.near_wellbore_complexity),
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `C:\Users\LucasChristman\.venvs\dfit\Scripts\python.exe -m pytest tests/test_panel_fields.py tests/test_step_status.py -v`

Expected: all pass, including `test_field_step_covers_exactly_the_panel_fields` (the key is now in both).

- [ ] **Step 7: Run the full suite**

Run: `C:\Users\LucasChristman\.venvs\dfit\Scripts\python.exe -m pytest`

Expected: all tests pass.

- [ ] **Step 8: Commit**

```bash
git add dfit_tool/ui.py tests/test_panel_fields.py tests/test_step_status.py
git commit -m "TODO #5: NWB complexity panel row"
```

---

### Task 3: Master-log column

**Files:**
- Modify: `dfit_tool/store.py:35-51` — append to `LOG_COLUMNS`
- Modify: `dfit_tool/store.py:396-397` — map the field in `build_log_row`
- Test: `tests/test_store.py` (modify — append two tests)

**Interfaces:**
- Consumes: `DerivedResults.near_wellbore_complexity` from Task 1.
- Produces: the log column name `"near_wellbore_complexity"`, last element of `store.LOG_COLUMNS`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_store.py` (the module already imports `os`, `pandas as pd`, `store`,
`compute_all`, `make_testdata`, `overview_state`):

```python
def test_log_row_has_near_wellbore_complexity(tmp_path):
    td = make_testdata()
    state = overview_state(td)
    res = compute_all(state, td)
    res.near_wellbore_complexity = 107.0
    entry = store.TestEntry(test_id="well1", folder=str(tmp_path))
    active_path = os.path.join(str(tmp_path), "well1.csv")

    row = store.build_log_row(entry, active_path, str(tmp_path), state, td, res)

    # Appended last so an existing dfit_log.csv stays loadable.
    assert store.LOG_COLUMNS[-1] == "near_wellbore_complexity"
    assert row["near_wellbore_complexity"] == 107.0


def test_load_log_backfills_missing_near_wellbore_complexity(tmp_path):
    old_columns = [c for c in store.LOG_COLUMNS if c != "near_wellbore_complexity"]
    old_df = pd.DataFrame([{c: "" for c in old_columns}])
    old_df.loc[0, "test_id"] = "well1"
    path = tmp_path / store.LOG_FILENAME
    old_df.to_csv(path, index=False)

    loaded = store.load_log(str(tmp_path))

    assert list(loaded.columns) == store.LOG_COLUMNS
    assert loaded.loc[0, "test_id"] == "well1"
    assert pd.isna(loaded.loc[0, "near_wellbore_complexity"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `C:\Users\LucasChristman\.venvs\dfit\Scripts\python.exe -m pytest tests/test_store.py -v -k near_wellbore`

Expected: both FAIL — the first on `assert store.LOG_COLUMNS[-1] == "near_wellbore_complexity"` (it is `"net_pressure_isip_source"`), the second on the column list not containing the new name.

- [ ] **Step 3: Append the column**

In `dfit_tool/store.py`, replace the tail of `LOG_COLUMNS` (lines 48-51) with:

```python
    "closure_time_compliance_min", "closure_time_tangent_min",
    "closure_time_variable_min",
    "net_pressure_isip_source",
    "near_wellbore_complexity",
]
```

- [ ] **Step 4: Map the field in `build_log_row`**

In `dfit_tool/store.py`, in the returned dict, after the
`"net_pressure_isip_source": res.net_pressure_isip_source,` line (line 396):

```python
        "net_pressure_isip_source": res.net_pressure_isip_source,
        "near_wellbore_complexity": res.near_wellbore_complexity,
    }
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `C:\Users\LucasChristman\.venvs\dfit\Scripts\python.exe -m pytest tests/test_store.py -v`

Expected: all pass, including `test_build_log_row_keys_match_log_columns` (line 636), which
asserts `list(row.keys()) == store.LOG_COLUMNS` and therefore checks that the new key was
added in the matching position.

- [ ] **Step 6: Run the full suite**

Run: `C:\Users\LucasChristman\.venvs\dfit\Scripts\python.exe -m pytest`

Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add dfit_tool/store.py tests/test_store.py
git commit -m "TODO #5: near_wellbore_complexity log column"
```

---

### Task 4: Documentation

**Files:**
- Modify: `CLAUDE.md` — the "Net pressure" paragraph in "Domain and methodology", the
  per-test deliverables list above it, and TODO #5 in the "TODO" section

**Interfaces:**
- Consumes: the names established in Tasks 1-3 — `interpret.near_wellbore_complexity`,
  `DerivedResults.near_wellbore_complexity`, panel row `"NWB complexity"`, log column
  `near_wellbore_complexity`.
- Produces: nothing other tasks depend on.

- [ ] **Step 1: Add the deliverable to the per-test list**

In `CLAUDE.md`, in the "Per-test deliverables" bullet list under "Domain and methodology",
add a bullet after the **Shmin, variable** bullet and before **Pore pressure**:

```markdown
- **Near-wellbore complexity** — apparent ISIP − the shared reference effective ISIP. The
  near-wellbore friction and tortuosity that is in the early-decline extrapolation but has
  dissipated by the time the P-vs-G line is fit. Shown as the "NWB complexity" panel row and
  logged to the `near_wellbore_complexity` column.
```

- [ ] **Step 2: Document the identity in the Net pressure paragraph**

In `CLAUDE.md`, append to the **Net pressure** paragraph (the one ending "...are kept in the
CSV log but are no longer shown in the sidebar panel."):

```markdown
Near-wellbore complexity subtracts that same shared reference from the apparent ISIP
(`interpret.near_wellbore_complexity`), which closes the identity `Shmin + net pressure +
complexity = apparent ISIP` for all three methods. It is one value per test, not one per
method, set by `model._resolve_net_pressures` alongside the net pressures and guarded on the
apparent ISIP being present. A negative value is reported as-is -- no warning, no clamp --
since clamping would break the identity. The C-D rapid-closure scenario gets no complexity:
it has no contact pick and so no effective ISIP, and `shmin_rapid` deliberately never feeds
the shared reference.
```

- [ ] **Step 3: Mark TODO #5 done**

In `CLAUDE.md`, in the "TODO" section, replace item 5 with:

```markdown
5) ~~Add new calculation "Near-Wellbore Complexity". Shmin + net pressure + complexity = ISIP.~~
   Done: apparent ISIP − shared reference eff ISIP, panel row "NWB complexity" and log column
   `near_wellbore_complexity`. See "Net pressure" under Domain and methodology.
```

- [ ] **Step 4: Verify the docs match the code**

Run: `C:\Users\LucasChristman\.venvs\dfit\Scripts\python.exe -m pytest`

Expected: all tests pass (no code changed in this task; this is the regression check before
the final commit).

Then confirm each name quoted in the new CLAUDE.md prose exists:

Run: `C:\Users\LucasChristman\.venvs\dfit\Scripts\python.exe -c "from dfit_tool import interpret, ui, store, model; assert callable(interpret.near_wellbore_complexity); assert 'NWB complexity' in ui.PANEL_FIELDS; assert 'near_wellbore_complexity' in store.LOG_COLUMNS; assert hasattr(model.DerivedResults(), 'near_wellbore_complexity'); print('ok')"`

Expected: `ok`

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md
git commit -m "TODO #5: docs for near-wellbore complexity"
```

---

## Manual verification

After Task 4, launch the app once and confirm the row renders:

Run: `C:\Users\LucasChristman\.venvs\dfit\Scripts\python.exe -m dfit_tool.app`

Open a test file, pick start/shut-in on overview, the decline tangent on isip, and the contact
point on gfunction. Expected: the panel shows an "NWB complexity" row under
"eff ISIP (compliance)", reading `-` until the gfunction step is visited, then a number equal
to `apparent ISIP − eff ISIP (compliance)` to the nearest psi (both are displayed rounded, so
allow ±1 psi of rounding in the check).
