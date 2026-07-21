# Draggable start/shut-in lines + plot color preferences

Date: 2026-07-21
Status: approved for planning

## Goal

Two changes to the DFIT interpretation tool's plotting/interaction layer:

1. Apply the user's color convention: rate = blue, surface pressure = red, BHP = black.
2. Replace click-to-place of the injection-start and shut-in markers with drag-to-move,
   snapping to the nearest data sample.

## Background

The tool is Tkinter + matplotlib (`dfit_tool/`). `plots.py` holds stateless `render_*(ax, td,
state, res)` functions; `picks.py` holds interaction controllers and per-step pick handlers that
mutate `PickState`; `ui.py` (`DfitApp`) hosts the canvas and re-attaches controllers after each
render via `_attach_controllers`, and recomputes + redraws through `refresh()`.

Today the injection-start (`state.start_idx`) and shut-in (`state.shutin_idx`) markers are drawn as
`axvline`s in `render_overview` and set only by left/right click through
`picks.ClickController` -> `handle_overview_click`. `picks.py` documents that fine-tuning is done in
the side panel, not by dragging handles. This spec changes that.

## Decisions (from brainstorming)

- Interaction model: **drag-only**. Remove left/right click placement entirely.
- Snapping: **snap to nearest data sample** (state keeps integer `start_idx` / `shutin_idx`).
- BHP color: **black everywhere** it appears as a raw pressure curve.
- View on drag release: **preserve the current view** (do not re-zoom).

## Changes

### 1. Colors

`render_overview` (`plots.py`):
- Rate trace and its right (`ax2`) axis label + ticks -> `tab:blue`.
- Pressure trace and its left (`ax`) axis: `black` when `state.pressure_is_bhp` is true,
  else `tab:red` (surface pressure). Legend label unchanged ("BHP" / "pressure").

BHP curves in the diagnostic renderers -> `black` (from `tab:blue`):
- `render_isip`: the BHP decline curve.
- `render_gfunction`: the BHP-vs-G curve and its left ylabel. `dP/dG` stays red.
- `render_porepressure`: the BHP-vs-t^n curve.
- `render_loglog`: **unchanged.** Its blue curve is `dp` (a derived pressure change), not a raw
  BHP trace; recoloring it black would collide with the red `t*dP/dt` pairing.

The injection-start / shut-in marker colors (orange dashed / red solid) are unchanged.

### 2. Drag-only start/shut-in markers

`render_overview` (`plots.py`):
- Tag the two `axvline`s with a gid: `"start"` and `"shutin"`.
- Increase their linewidth slightly (start ~1.6, shut-in ~1.8) so they are easy to grab.

`picks.py`:
- Remove `handle_overview_click` (retired; no longer wired).
- Add `DragLineController`:
  - Constructed with `(canvas, ax, handlers)` where `handlers` maps gid ->
    `on_release(x_data) -> None`, plus a `guard()` callable returning True when interaction should
    be blocked (toolbar zoom/pan active).
  - On `button_press_event` (button 1) inside `ax`: find the tagged line whose x, projected to
    display pixels, is within a pixel tolerance (~6 px) of the cursor. If found, capture it as the
    active drag target. If `guard()` is true, do nothing.
  - On `motion_notify_event` while dragging: set the captured line's xdata to `event.xdata` and
    `canvas.draw_idle()` for live feedback. No recompute during drag.
  - On `button_release_event`: call the gid's `on_release(final_x)` and release the target. The
    release handler (in `ui.py`) snaps to the nearest sample and commits to state, then refreshes.
  - `disconnect()` unbinds all three cids, mirroring the other controllers.
- Add/keep a nearest-sample helper (reuse `_nearest`) to map released x (hours) -> sample index.

`ui.py`:
- `_attach_controllers` overview branch: replace the `ClickController` with a `DragLineController`
  whose handlers are:
  - `"start"`: `x_hours -> state.start_idx = nearest(td.t_s/3600, x_hours)`
  - `"shutin"`: `x_hours -> state.shutin_idx = nearest(td.t_s/3600, x_hours)`
  - each also sets `state.qmax_bpm = None` (force re-derive from the new window), then calls
    `self.refresh(preserve_view=True)`.
  - guard = `lambda: bool(self.toolbar.mode)`.
- Hint text -> "Drag the injection-start and shut-in lines to adjust the window."

### 3. Preserve view on drag-triggered refresh

`ui.py` `refresh(self, preserve_view: bool = False)`:
- When `preserve_view` is true and an axes already exists, capture the current `get_xlim()` /
  `get_ylim()` before `self.fig.clf()`, and after the renderer draws, re-apply them with
  `set_xlim` / `set_ylim` (before `tight_layout` / `draw_idle`).
- Default `False` preserves existing behavior for step changes, config apply, and pick loads.

## Out of scope

- Dragging picks on the other steps (ISIP tangent, contact, closure, windows) — unchanged.
- Free (continuous) placement between samples — not doing; indices retained.
- Blitting optimization — `draw_idle` is sufficient for the decimated (<=6000 pt) overview.

## Testing / verification

- Headless render check (Agg): after setting `start_idx`/`shutin_idx`, `render_overview` produces
  a figure with two gid-tagged lines and the correct trace colors for both BHP and surface-pressure
  cases. Assert line gids, colors (rate blue, BHP black / SP red), and axis label colors.
- `DragLineController` unit check with synthetic mpl events: press near a tagged line captures it;
  motion updates its xdata; release invokes the mapped handler with the final x; guard True blocks
  capture.
- Manual: load a CSV, confirm the two lines drag and snap, the view stays put on release, results
  (te, Vinj, qmax) update, and the color convention renders in both BHP and surface-pressure modes.

## Notes

- Project is not a git repo, so the design doc is saved but not committed.
