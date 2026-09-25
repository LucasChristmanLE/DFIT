# Collapsible warnings panel

## Problem

The right-hand pick panel (`ui.py`'s `panel`) packs, bottom-up: `hint_lbl` (per-step gray
instruction text), then `warn_lbl` (red, `"\n".join(r.warnings)`, unbounded height, wraplength
tracks the panel's width). Everything else -- Results rows, the closure/postclosure-scenario
widgets, the stiffness checkbox, Notes -- packs from the top. The panel is a fixed-height pane
inside the outer `PanedWindow` with no scrolling, so when a test accumulates several warnings
(tail trim, rise guard, dropout masking, low surface pressure, TVD, unit conversion, etc. can
all fire at once, each wrapped over 1-3 lines), `warn_lbl`'s growing height eats into the space
the top-packed widgets need, and the closure-scenario dropdown / postclosure combobox /
stiffness checkbox become invisible or clipped.

## Design

Confined entirely to `dfit_tool/ui.py`. `DerivedResults.warnings` (model.py) is untouched --
this only changes how `warn_lbl` renders that list.

### `format_warnings_text(warnings: list[str], expanded: bool) -> str`

A new pure function (no Tkinter), same spirit as `interpret.format_shmin_rapid`:

- `[]` -> `""` (unchanged blank state when a test is clean)
- non-empty, `expanded=False` -> one line: `"{n} warning{s} (click to expand)"`
- non-empty, `expanded=True` -> `"{n} warning{s} (click to collapse)\n"` + the existing
  `"\n".join(warnings)` body

`{s}` is `""` for `n == 1`, `"s"` otherwise.

### New `DfitApp` state

Two attributes, initialized in `_build_body` alongside `warn_lbl`'s creation:

- `self._warnings_list: list[str] = []` -- the current test's full warning list, refreshed on
  every `_update_panel()` call
- `self._warnings_expanded: bool = False` -- reset to `False` on every `_update_panel()` call
  (recompute always re-collapses; only an explicit click expands), flipped only by
  `_toggle_warnings`

### Wiring

- `_update_panel()` (currently `self.warn_lbl.config(text="\n".join(r.warnings) if r.warnings
  else "")` at ui.py:1800) becomes:

  ```python
  self._warnings_list = list(r.warnings)
  self._warnings_expanded = False
  self.warn_lbl.config(text=format_warnings_text(self._warnings_list, False),
                        cursor="hand2" if self._warnings_list else "")
  ```

- New method:

  ```python
  def _toggle_warnings(self, event=None):
      if not self._warnings_list:
          return
      self._warnings_expanded = not self._warnings_expanded
      self.warn_lbl.config(text=format_warnings_text(self._warnings_list, self._warnings_expanded))
  ```

- `_build_body()` adds `self.warn_lbl.bind("<Button-1>", self._toggle_warnings)` right after
  `warn_lbl` is created/packed.

### Behavior

With 1+ warnings, the panel shows one red line (`"3 warnings (click to expand)"`) instead of
the full list, so it can never crowd out the scenario/checkbox controls above it. Clicking
expands it in place to the full wrapped list (with a "(click to collapse)" header line);
clicking again, or the next recompute (a drag, a pick commit, a scenario change -- anything
that calls `refresh()` -> `_update_panel()`), collapses it back to one line. Zero warnings
still renders blank text with a normal (non-hand) cursor, matching today.

### Out of scope, left as-is

- The one-off folder-open scan-warning summary (`ui.py:676-686`, `"{warn_count} scan
  warning(s)..."`) that transiently reuses `warn_lbl` right after `_open_folder_path` scans a
  root, before any test-specific `_update_panel()` call has necessarily run against the newly
  opened test. It sets `warn_lbl`'s text directly, bypassing `format_warnings_text`; the very
  next `_update_panel()` call (from loading the first queue entry, or any later refresh)
  overwrites it with the per-test collapsed summary as usual. Not touched by this design.
- `hint_lbl` (separate per-step gray hint label, already short, not part of this problem).

## Testing

Extend `tests/test_panel_fields.py`'s existing duck-typed `_FakeLabel`/`_panel_stub` pattern
(already used to test `_update_panel` headless):

- Direct tests of `format_warnings_text` for 0/1/N warnings, collapsed and expanded.
- A `_panel_stub`-based test confirming `_update_panel` sets the collapsed one-line summary and
  resets `_warnings_expanded` to `False`.
- A test binding `_toggle_warnings` the same way (`types.MethodType`) onto a stub carrying
  `_warnings_list`/`_warnings_expanded`, confirming it flips state and re-renders the full list,
  and that it no-ops when `_warnings_list` is empty.
