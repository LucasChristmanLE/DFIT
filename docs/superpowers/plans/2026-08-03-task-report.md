# Task report: TODO #4 shared net-pressure reference + slim eff-ISIP sidebar

Plan: `docs/superpowers/plans/2026-08-03-net-pressure-shared-reference.md`
Branch: `todo-4-net-pressure-reference`

## Status

DONE

## Commits

- `8366ea2` — TODO #4: shared net-pressure reference (compliance->tangent) + source field
- `941c2e2` — TODO #4: add net_pressure_isip_source column to dfit_log.csv
- `fdeb92d` — TODO #4: drop tangent/variable eff-ISIP rows from sidebar panel
- `4fc9168` — TODO #4: docs for shared net-pressure reference

## Test summary

Final `pytest` run: **395 passed**, 0 failed.

## Per-task notes

**Task 1** (`dfit_tool/model.py`, `tests/test_net_pressure_reference.py`): added
`DerivedResults.net_pressure_isip_source`, extracted `_resolve_net_pressures(res)` exactly
as specified, replaced the old `ref_compliance`/`ref_tangent`/`ref_variable` block (lines
357-370) with a single call, keeping the `delta_closure` lines untouched. New test file's
3 tests pass against the hand-built-`DerivedResults` pattern the plan specified as simplest.

Full-suite run surfaced two existing tests in `tests/test_variable_compliance.py` that
asserted the old per-method-effective-ISIP / apparent-ISIP-fallback semantics:
- `test_net_pressures_each_use_their_own_effective_isip_and_can_differ` — rewritten as
  `test_net_pressures_all_use_shared_compliance_reference`: all three net pressures now
  reference the shared compliance eff ISIP, and still differ because each method's own
  Shmin differs.
- `test_net_pressure_falls_back_to_apparent_isip_when_effective_isip_unavailable` —
  rewritten as `test_net_pressure_none_when_no_effective_isip_available`: with no
  effective ISIP available at either rung, all three net pressures are now `None` and
  `net_pressure_isip_source == ""` (no apparent-ISIP fallback).
These are exactly the "old apparent-ISIP fallback" test the plan told me to expect and
update; no other regressions.

**Task 2** (`dfit_tool/store.py`, `tests/test_store.py`): appended
`"net_pressure_isip_source"` to `LOG_COLUMNS` and to `build_log_row`'s output dict, as the
last entries in each. Added `test_log_row_has_net_pressure_isip_source` to
`tests/test_store.py` (no separate `test_store_log.py`/`test_log_source_column.py` existed
or was needed — `tests/test_store.py` already holds all `build_log_row` tests, matching the
plan's fallback instruction). Note: the plan's Task 2 test snippet calls
`build_log_row(entry, active_path, root, state, res, td)` — the real signature (confirmed by
reading `store.py` and the existing tests) is `build_log_row(entry, active_path, root,
state, td, res)` (`td` before `res`). I used the real signature; this is a copy-paste
ordering slip in the plan text, not a semantic ambiguity, so I did not stop for
NEEDS_CONTEXT over it.

**Task 3** (`dfit_tool/ui.py`, `tests/test_panel_fields.py`): removed `"eff ISIP (tangent)"`
and `"eff ISIP (variable)"` from `PANEL_FIELDS`, `FIELD_STEP`, and the `vals` dict in
`_update_panel`, verbatim per the plan. Full-suite run surfaced one more pre-existing test
not mentioned in the plan, `tests/test_step_status.py::test_field_step_mapping_matches_spec`,
which asserted the full old `FIELD_STEP` dict including the two removed keys — updated it to
match the new mapping (same edit, different assertion site). No other regressions.

**Task 4** (`CLAUDE.md`): replaced the "Net pressure" paragraph with the plan's exact
replacement text. Left the file's other pre-existing uncommitted edit (TODO item #3 wording,
present before this task started and unrelated to TODO #4) untouched and still uncommitted,
by staging only the relevant hunk for the commit.

## Deferred / out of scope

- `dfit_log.csv` in the repo root is an untracked leftover from a prior manual run; not
  touched, not committed (it predates this session per the initial `git status`).
- Nothing else deferred. All 4 tasks and their self-review checklist items are complete.

## Concerns

None blocking. One minor plan inaccuracy noted above (Task 2 test snippet's parameter
order), resolved by using the actual function signature — no behavior ambiguity resulted.
