# Collapsible Warnings Panel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the pick panel's warnings label from growing unbounded and crowding the
closure/postclosure scenario controls out of view, by collapsing it to a one-line count that
expands on click.

**Architecture:** Entirely inside `dfit_tool/ui.py`. A new pure module-level function,
`format_warnings_text`, decides what text `warn_lbl` shows given the warning list and an
expanded/collapsed flag. `DfitApp` gains two instance attributes (`_warnings_list`,
`_warnings_expanded`) and a click handler (`_toggle_warnings`) that flips the flag and
re-renders without a full recompute. `_update_panel` always resets to collapsed on every
recompute, per the approved design.

**Tech Stack:** Python 3.14, Tkinter/ttk, pytest. No new dependencies.

## Global Constraints

- Use the project venv for every test run: `C:\Users\LucasChristman\.venvs\dfit\Scripts\python.exe`
  (never system Python) -- see repo root `CLAUDE.md`.
- `dfit_tool/ui.py` is the only Tkinter consumer in the package; `picks.py`/`plots.py`/`sliders.py`
  stay untouched by this change (nothing here belongs there).
- New logic should be headless-testable where possible -- `format_warnings_text` is a plain
  function with no Tkinter dependency, and both `DfitApp` methods it wires into are tested via
  the existing duck-typed stub pattern in `tests/test_panel_fields.py`, not a real `tk.Tk()`.
- `DerivedResults.warnings` (model.py) is not touched by this feature -- only how `ui.py` renders
  that list changes.
- Follow the approved design exactly:
  `docs/superpowers/specs/2026-09-25-collapsible-warnings-panel-design.md`.

---

## Task 1: `format_warnings_text` pure helper

**Files:**
- Modify: `dfit_tool/ui.py:211-214` (insert the new function between `_isip_minutes_to_seconds`
  and `class DfitApp:`)
- Test: `tests/test_panel_fields.py` (append new test functions)

**Interfaces:**
- Produces: `format_warnings_text(warnings: list[str], expanded: bool) -> str`, a module-level
  function in `dfit_tool.ui`. Later tasks call it as `ui.format_warnings_text(...)` from tests
  and as `format_warnings_text(...)` from inside `ui.py` itself.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_panel_fields.py`:

```python
def test_format_warnings_text_empty():
    assert ui.format_warnings_text([], expanded=False) == ""
    assert ui.format_warnings_text([], expanded=True) == ""


def test_format_warnings_text_collapsed_singular():
    assert ui.format_warnings_text(["Tail auto-trimmed."], expanded=False) == (
        "1 warning (click to expand)")


def test_format_warnings_text_collapsed_plural():
    warnings = ["Tail auto-trimmed.", "Pressure dropout masked."]
    assert ui.format_warnings_text(warnings, expanded=False) == (
        "2 warnings (click to expand)")


def test_format_warnings_text_expanded_shows_full_list():
    warnings = ["Tail auto-trimmed.", "Pressure dropout masked."]
    assert ui.format_warnings_text(warnings, expanded=True) == (
        "2 warnings (click to collapse)\nTail auto-trimmed.\nPressure dropout masked.")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `C:\Users\LucasChristman\.venvs\dfit\Scripts\python.exe -m pytest tests/test_panel_fields.py -v -k format_warnings_text`
Expected: FAIL with `AttributeError: module 'dfit_tool.ui' has no attribute 'format_warnings_text'`

- [ ] **Step 3: Write the minimal implementation**

In `dfit_tool/ui.py`, insert immediately after `_isip_minutes_to_seconds` (currently ends at
line 211, two blank lines before `class DfitApp:` at line 214):

```python
def format_warnings_text(warnings: list[str], expanded: bool) -> str:
    """Render the pick panel's warnings label text. Collapsed (the default after every
    recompute -- see ``DfitApp._update_panel``) shows just a count, so a long warning list can
    never crowd the closure/postclosure scenario controls above it out of view. Expanded shows
    the full list, same text the label always carried before this feature existed."""
    if not warnings:
        return ""
    n = len(warnings)
    noun = "warning" if n == 1 else "warnings"
    if not expanded:
        return f"{n} {noun} (click to expand)"
    return f"{n} {noun} (click to collapse)\n" + "\n".join(warnings)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `C:\Users\LucasChristman\.venvs\dfit\Scripts\python.exe -m pytest tests/test_panel_fields.py -v -k format_warnings_text`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add dfit_tool/ui.py tests/test_panel_fields.py
git commit -m "Add format_warnings_text helper for the collapsible warnings label"
```

---

## Task 2: Wire `_update_panel` to render collapsed by default

**Files:**
- Modify: `dfit_tool/ui.py:398-402` (`_build_body`, where `warn_lbl` is created)
- Modify: `dfit_tool/ui.py:1800` (`_update_panel`, the line that currently sets `warn_lbl`'s text)
- Test: `tests/test_panel_fields.py` (append new tests using the existing `_panel_stub` helper)

**Interfaces:**
- Consumes: `format_warnings_text(warnings: list[str], expanded: bool) -> str` from Task 1.
- Produces: `DfitApp._warnings_list: list[str]` and `DfitApp._warnings_expanded: bool`, both set
  by `_update_panel` on every call and read by Task 3's `_toggle_warnings`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_panel_fields.py` (uses the existing `_panel_stub(res, state)` helper
already defined in this file, which builds a `PickState`/`DerivedResults`-backed stub and binds
the real `DfitApp._update_panel` onto it):

```python
def test_update_panel_collapses_warnings_by_default():
    res = DerivedResults(warnings=["Tail auto-trimmed.", "Pressure dropout masked."])
    state = PickState(step_status={"gfunction": "done"})
    stub = _panel_stub(res, state)

    stub._update_panel()

    assert stub.warn_lbl.text == "2 warnings (click to expand)"
    assert stub._warnings_expanded is False
    assert stub._warnings_list == ["Tail auto-trimmed.", "Pressure dropout masked."]


def test_update_panel_blanks_warnings_label_when_clean():
    res = DerivedResults(warnings=[])
    state = PickState(step_status={"gfunction": "done"})
    stub = _panel_stub(res, state)

    stub._update_panel()

    assert stub.warn_lbl.text == ""
    assert stub._warnings_list == []
```

This follows the same pattern already used by every other test in this file (see e.g.
`test_update_panel_cd_rapid_marks_shmin_compliance_row` just above): construct `DerivedResults`
and `PickState` directly with keyword args, no `tests/helpers.py` fixtures needed. `DerivedResults`
and `PickState` are already imported at the top of the file (line 4).

- [ ] **Step 2: Run tests to verify they fail**

Run: `C:\Users\LucasChristman\.venvs\dfit\Scripts\python.exe -m pytest tests/test_panel_fields.py -v -k update_panel_collapses`
Expected: FAIL -- `stub.warn_lbl.text` will be `"Tail auto-trimmed.\nPressure dropout masked."`
(today's uncollapsed join), not the new one-line summary; `stub._warnings_expanded` won't exist
as an attribute at all yet.

- [ ] **Step 3: Write the minimal implementation**

In `dfit_tool/ui.py`, `_build_body`, change (currently lines 400-402):

```python
        self.warn_lbl = ttk.Label(panel, text="", foreground="red", wraplength=300,
                                  justify="left")
        self.warn_lbl.pack(side="bottom", anchor="w", fill="x", pady=(6, 0))
```

to:

```python
        self.warn_lbl = ttk.Label(panel, text="", foreground="red", wraplength=300,
                                  justify="left")
        self.warn_lbl.pack(side="bottom", anchor="w", fill="x", pady=(6, 0))
        self._warnings_list: list[str] = []
        self._warnings_expanded: bool = False
```

In `dfit_tool/ui.py`, `_update_panel`, change the current last line (currently line 1800):

```python
        self.warn_lbl.config(text="\n".join(r.warnings) if r.warnings else "")
```

to:

```python
        self._warnings_list = list(r.warnings)
        self._warnings_expanded = False
        self.warn_lbl.config(text=format_warnings_text(self._warnings_list, False),
                             cursor="hand2" if self._warnings_list else "")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `C:\Users\LucasChristman\.venvs\dfit\Scripts\python.exe -m pytest tests/test_panel_fields.py -v`
Expected: PASS, all tests in the file (existing ones plus the new ones from Task 1 and this task)

- [ ] **Step 5: Commit**

```bash
git add dfit_tool/ui.py tests/test_panel_fields.py
git commit -m "Collapse the pick panel's warnings label to a one-line count by default"
```

---

## Task 3: `_toggle_warnings` click handler

**Files:**
- Modify: `dfit_tool/ui.py:398-405` (`_build_body`, bind the click right after the attributes
  added in Task 2)
- Modify: `dfit_tool/ui.py` (insert new method `_toggle_warnings` immediately after
  `_update_panel`, i.e. between `_update_panel`'s closing line and `_update_unit_labels`,
  currently ui.py:1801-1802)
- Test: `tests/test_panel_fields.py` (append new tests)

**Interfaces:**
- Consumes: `DfitApp._warnings_list`, `DfitApp._warnings_expanded`, `format_warnings_text` (all
  from Tasks 1-2).
- Produces: `DfitApp._toggle_warnings(self, event=None) -> None`, bound to `warn_lbl`'s
  `<Button-1>` event.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_panel_fields.py`:

```python
def _toggle_stub(warnings_list, expanded):
    stub = types.SimpleNamespace()
    stub.warn_lbl = _FakeLabel()
    stub._warnings_list = warnings_list
    stub._warnings_expanded = expanded
    stub._toggle_warnings = types.MethodType(ui.DfitApp._toggle_warnings, stub)
    return stub


def test_toggle_warnings_expands_then_collapses():
    stub = _toggle_stub(["A", "B"], expanded=False)

    stub._toggle_warnings()
    assert stub._warnings_expanded is True
    assert stub.warn_lbl.text == "2 warnings (click to collapse)\nA\nB"

    stub._toggle_warnings()
    assert stub._warnings_expanded is False
    assert stub.warn_lbl.text == "2 warnings (click to expand)"


def test_toggle_warnings_noop_when_no_warnings():
    stub = _toggle_stub([], expanded=False)
    stub.warn_lbl.text = ""

    stub._toggle_warnings()

    assert stub._warnings_expanded is False
    assert stub.warn_lbl.text == ""
```

This reuses the module's existing `_FakeLabel` (already defined above in this file for the
`_panel_stub` tests) and the `types`/`ui` imports already at the top of the file.

- [ ] **Step 2: Run tests to verify they fail**

Run: `C:\Users\LucasChristman\.venvs\dfit\Scripts\python.exe -m pytest tests/test_panel_fields.py -v -k toggle_warnings`
Expected: FAIL with `AttributeError: type object 'DfitApp' has no attribute '_toggle_warnings'`

- [ ] **Step 3: Write the minimal implementation**

In `dfit_tool/ui.py`, `_build_body`, change the block added in Task 2 (currently lines 400-405
after that task's edit):

```python
        self.warn_lbl = ttk.Label(panel, text="", foreground="red", wraplength=300,
                                  justify="left")
        self.warn_lbl.pack(side="bottom", anchor="w", fill="x", pady=(6, 0))
        self._warnings_list: list[str] = []
        self._warnings_expanded: bool = False
```

to:

```python
        self.warn_lbl = ttk.Label(panel, text="", foreground="red", wraplength=300,
                                  justify="left")
        self.warn_lbl.pack(side="bottom", anchor="w", fill="x", pady=(6, 0))
        self._warnings_list: list[str] = []
        self._warnings_expanded: bool = False
        self.warn_lbl.bind("<Button-1>", self._toggle_warnings)
```

In `dfit_tool/ui.py`, insert a new method immediately after `_update_panel`'s closing line
(currently ui.py:1800, right before `def _update_unit_labels(self):` at line 1802):

```python
    def _toggle_warnings(self, event=None):
        """Click handler for warn_lbl: flips the collapsed/expanded summary in place, no
        recompute. No-ops on a click when there's nothing to show."""
        if not self._warnings_list:
            return
        self._warnings_expanded = not self._warnings_expanded
        self.warn_lbl.config(text=format_warnings_text(self._warnings_list,
                                                        self._warnings_expanded))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `C:\Users\LucasChristman\.venvs\dfit\Scripts\python.exe -m pytest tests/test_panel_fields.py -v`
Expected: PASS, every test in the file.

- [ ] **Step 5: Run the full test suite**

Run: `C:\Users\LucasChristman\.venvs\dfit\Scripts\python.exe -m pytest`
Expected: full suite passes (no regressions elsewhere -- nothing else in the codebase reads or
sets `warn_lbl`'s text except the unrelated one-off folder-scan summary at `ui.py:676-686`,
which is untouched by this plan).

- [ ] **Step 6: Commit**

```bash
git add dfit_tool/ui.py tests/test_panel_fields.py
git commit -m "Add click-to-expand/collapse toggle for the pick panel warnings label"
```

---

## Manual smoke test (not automated, do once after Task 3)

Launch the app against any sample DFIT file that produces several warnings at once (e.g. one
that trips the tail-guard and a low-surface-pressure warning together), open the G-function or
overview step, and confirm:
1. The warnings label shows a one-line count, and the closure-scenario dropdown above it is
   fully visible.
2. Clicking the label expands it to the full list; clicking again collapses it.
3. Dragging any pick (which triggers a recompute) collapses it back to one line even if it was
   left expanded.
