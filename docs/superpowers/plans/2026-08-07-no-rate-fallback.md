# No-Rate Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rate-less DFIT datasets get draggable start/shut-in vlines (seeded from the pressure shape) and a usable te (wall-clock pump duration), so every downstream step works without a rate channel.

**Architecture:** A new pure suggester in `interpret.py` picks a window from the pressure curve alone; `picks.seed_overview` falls back to it when there is no usable rate; `model.compute_all` un-gates `t_shutin_s` from rate and falls back to te = t[shutin] − t[start] when the effective te (Vinj/qmax) is unavailable; `plots.render_overview`'s title stops assuming te implies Vinj/qmax.

**Tech Stack:** Python 3.14, numpy, matplotlib (Agg in tests), pytest.

**Spec:** `docs/superpowers/specs/2026-08-07-no-rate-fallback-design.md`

## Global Constraints

- Test runner: `C:\Users\LucasChristman\.venvs\dfit\Scripts\python.exe -m pytest` from the repo root (system Python lacks the deps).
- New logic lands only in headless layers (`interpret`, `picks`, `model`, `plots`) — no Tkinter.
- `compute_all` stays the single source of truth; the UI is untouched.
- Renderers never set view limits.
- Warning text is ASCII. The fallback-te warning is **appended** to `res.warnings` (the front slot is reserved for the tail-guard warning).
- All new tests go in `tests/test_no_rate_fallback.py`.

---

### Task 1: `interpret.suggest_injection_window_pressure`

**Files:**
- Modify: `dfit_tool/interpret.py` (after `suggest_injection_window`, ~line 76)
- Test: `tests/test_no_rate_fallback.py` (create)

**Interfaces:**
- Produces: `suggest_injection_window_pressure(p: np.ndarray) -> tuple[int, int]` — returns `(start_idx, shutin_idx)` with `start < shutin`; raises `ValueError` only when fewer than 2 finite samples exist. Task 2 calls it.

- [x] **Step 1: Write the failing tests**

Create `tests/test_no_rate_fallback.py`:

```python
"""No-rate fallback: pressure-based overview window seeding + te = pump duration.

Covers docs/superpowers/specs/2026-08-07-no-rate-fallback-design.md: datasets with no rate
channel (or a dead one) must still get draggable start/shut-in vlines (seeded from the
pressure shape) and a usable te, so the downstream steps work without rate.
"""

from __future__ import annotations

import numpy as np
import pytest

from dfit_tool import interpret
from tests.helpers import make_testdata, PRESSURE_COL, START_IDX, SHUTIN_IDX


# --------------------------------------------------------------------------------------------------
# interpret.suggest_injection_window_pressure
# --------------------------------------------------------------------------------------------------
def test_pressure_window_lands_on_the_dfit_shape():
    td = make_testdata()
    p = td.column(PRESSURE_COL)
    start, shutin = interpret.suggest_injection_window_pressure(p)
    # shut-in at the pressure max (the linspace peak sits at SHUTIN_IDX - 1)
    assert SHUTIN_IDX - 2 <= shutin <= SHUTIN_IDX + 1
    # start at the rise onset: baseline 2000, rise 3000, 10% threshold crosses ~20 samples in
    assert START_IDX <= start <= START_IDX + 40
    assert start < shutin


def test_pressure_window_skips_an_early_breakdown_pulse():
    # Baseline 2000 with a pulse to 4000 at [50, 60), then the main rise at [100, 300) to
    # 5000 and a decline. The last-upcross rule must put start on the main rise, not the pulse.
    n = 600
    p = np.full(n, 2000.0)
    p[50:60] = 4000.0
    p[100:300] = 2000.0 + 3000.0 * np.linspace(0.0, 1.0, 200)
    p[300:] = np.linspace(5000.0, 3500.0, n - 300)
    start, shutin = interpret.suggest_injection_window_pressure(p)
    assert start >= 100
    assert 295 <= shutin <= 301
    assert start < shutin


def test_pressure_window_flat_record_falls_back_to_positional():
    p = np.full(50, 1000.0)
    start, shutin = interpret.suggest_injection_window_pressure(p)
    assert 0 <= start < shutin <= 49


def test_pressure_window_declining_record_falls_back_to_positional():
    p = np.linspace(5000.0, 1000.0, 50)  # max at index 0
    start, shutin = interpret.suggest_injection_window_pressure(p)
    assert 0 <= start < shutin <= 49


def test_pressure_window_too_few_finite_samples_raises():
    with pytest.raises(ValueError):
        interpret.suggest_injection_window_pressure(np.array([np.nan, np.nan, 5.0]))
```

- [x] **Step 2: Run the new tests to verify they fail**

Run: `C:\Users\LucasChristman\.venvs\dfit\Scripts\python.exe -m pytest tests/test_no_rate_fallback.py -v`
Expected: FAIL / ERROR with `AttributeError: ... has no attribute 'suggest_injection_window_pressure'`

- [x] **Step 3: Implement the suggester**

In `dfit_tool/interpret.py`, directly after `suggest_injection_window` (after ~line 76):

```python
def suggest_injection_window_pressure(p: np.ndarray) -> tuple[int, int]:
    """Best-guess (start, shutin) from the pressure curve alone, for rate-less datasets.

    shutin = the global pressure maximum (in a DFIT the max sits at/just before shut-in);
    start = the last upcross of baseline + 10% of the rise at/before that max (the *last*
    upcross tolerates breakdown pulses and step-rate cycles earlier in the record).
    Degenerate shapes (flat record, max at the first sample) fall back to positional
    defaults at 2% / 25% of the record. Raises ``ValueError`` only when fewer than 2 finite
    samples exist; otherwise always returns ``start < shutin``. This is only a default --
    the interpreter drags the lines to the true window.
    """
    p = np.asarray(p, dtype=float)
    n = len(p)
    if int(np.isfinite(p).sum()) < 2:
        raise ValueError("Need at least 2 finite pressure samples to suggest a window")

    shutin = int(np.nanargmax(p))
    start = None
    if shutin > 0:
        baseline = float(np.nanmin(p[:shutin + 1]))
        rise = float(p[shutin]) - baseline
        if rise > 0:
            thresh = baseline + 0.1 * rise
            above = p[:shutin + 1] >= thresh  # NaN compares False, so dropouts stay "below"
            upcross = np.where(above[1:] & ~above[:-1])[0] + 1
            if upcross.size:
                start = int(upcross[-1])
            elif above.any():  # unreachable in practice (the max is above); belt and braces
                start = int(np.where(above)[0][0])

    if shutin == 0 or start is None:
        # Flat or declining-only record: positional defaults, coerced so start < shutin.
        start = max(int(0.02 * (n - 1)), 0)
        shutin = min(max(int(0.25 * (n - 1)), start + 1), n - 1)
        start = min(start, shutin - 1)
    elif start >= shutin:
        # A single-sample jump straight to the max: keep the max, back start off one.
        start = shutin - 1
    return start, shutin
```

- [x] **Step 4: Run the tests to verify they pass**

Run: `C:\Users\LucasChristman\.venvs\dfit\Scripts\python.exe -m pytest tests/test_no_rate_fallback.py -v`
Expected: 5 PASS

- [x] **Step 5: Commit**

```bash
git add dfit_tool/interpret.py tests/test_no_rate_fallback.py
git commit -m "Add pressure-based injection-window suggester for rate-less datasets"
```

---

### Task 2: `picks.seed_overview` pressure fallback

**Files:**
- Modify: `dfit_tool/picks.py:1011-1022` (`seed_overview`)
- Test: `tests/test_no_rate_fallback.py` (append)

**Interfaces:**
- Consumes: `interpret.suggest_injection_window_pressure(p) -> tuple[int, int]` (Task 1).
- Produces: `seed_overview(state, td)` now seeds `state.start_idx`/`state.shutin_idx` when `state.rate_col` is empty (or the rate suggestion raises `ValueError`), provided `state.pressure_col` is set. Signature unchanged.

- [x] **Step 1: Write the failing tests**

Append to `tests/test_no_rate_fallback.py` (extend the imports at the top):

```python
from dfit_tool import picks
from dfit_tool.model import PickState
from tests.helpers import RATE_COL
```

```python
# --------------------------------------------------------------------------------------------------
# picks.seed_overview fallback
# --------------------------------------------------------------------------------------------------
def test_seed_overview_no_rate_col_seeds_from_pressure():
    td = make_testdata()
    st = PickState(pressure_col=PRESSURE_COL)  # no rate_col
    picks.seed_overview(st, td)
    assert st.start_idx is not None and st.shutin_idx is not None
    assert st.start_idx < st.shutin_idx


def test_seed_overview_dead_rate_channel_falls_back_to_pressure():
    td = make_testdata()
    td.df[RATE_COL] = 0.0  # rate never exceeds threshold -> suggest_injection_window raises
    st = PickState(pressure_col=PRESSURE_COL, rate_col=RATE_COL)
    picks.seed_overview(st, td)
    assert st.start_idx is not None and st.shutin_idx is not None
    assert st.start_idx < st.shutin_idx


def test_seed_overview_fallback_never_clobbers_existing_picks():
    td = make_testdata()
    st = PickState(pressure_col=PRESSURE_COL, start_idx=5, shutin_idx=7)
    picks.seed_overview(st, td)
    assert (st.start_idx, st.shutin_idx) == (5, 7)


def test_seed_overview_no_pressure_col_is_a_noop():
    td = make_testdata()
    st = PickState()  # neither rate nor pressure mapped
    picks.seed_overview(st, td)
    assert st.start_idx is None and st.shutin_idx is None
```

- [x] **Step 2: Run the new tests to verify they fail**

Run: `C:\Users\LucasChristman\.venvs\dfit\Scripts\python.exe -m pytest tests/test_no_rate_fallback.py -k seed_overview -v`
Expected: `test_seed_overview_no_rate_col_seeds_from_pressure` and `test_seed_overview_dead_rate_channel_falls_back_to_pressure` FAIL (picks stay None); the other two PASS (existing behavior).

- [x] **Step 3: Replace `seed_overview`**

In `dfit_tool/picks.py`, replace the whole function (currently lines 1011-1022):

```python
def seed_overview(state: PickState, td: TestData) -> None:
    """Injection window (start/shut-in indices) from the rate (+ optional volume) curve,
    falling back to the pressure shape when no usable rate channel exists -- the vlines must
    always exist for the analyst to drag, rate or not."""
    if state.start_idx is not None or state.shutin_idx is not None:
        return
    if state.rate_col:
        rate = td.column(state.rate_col)
        vol = td.column(state.volume_col) if state.volume_col else None
        try:
            state.start_idx, state.shutin_idx = interpret.suggest_injection_window(rate, vol)
            return
        except ValueError:
            pass  # e.g. an all-zero rate channel -- fall through to the pressure fallback
    if not state.pressure_col:
        return
    try:
        state.start_idx, state.shutin_idx = interpret.suggest_injection_window_pressure(
            td.column(state.pressure_col))
    except ValueError:
        pass
```

- [x] **Step 4: Run the tests to verify they pass**

Run: `C:\Users\LucasChristman\.venvs\dfit\Scripts\python.exe -m pytest tests/test_no_rate_fallback.py tests/test_seed_steps.py -v`
Expected: all PASS (test_seed_steps.py guards the rate-present path against regression).

- [x] **Step 5: Commit**

```bash
git add dfit_tool/picks.py tests/test_no_rate_fallback.py
git commit -m "Seed the overview start/shut-in vlines from pressure when no usable rate exists"
```

---

### Task 3: `compute_all` te fallback

**Files:**
- Modify: `dfit_tool/model.py:331-344` (the "Injection window + te" block in `compute_all`)
- Test: `tests/test_no_rate_fallback.py` (append)

**Interfaces:**
- Consumes: nothing new (pure restructure inside `compute_all`).
- Produces: `res.t_shutin_s` set whenever both picks exist (rate no longer required); `res.te_s` = wall-clock pump duration + an appended warning containing `"pump duration"` when the effective te is unavailable; `res.qmax_bpm`/`res.vinj*` stay `None` without rate. Task 4's test relies on the te-set/vinj-None combination.

- [x] **Step 1: Write the failing tests**

Append to `tests/test_no_rate_fallback.py` (extend imports):

```python
from dfit_tool.model import compute_all
from tests.helpers import overview_state
```

```python
# --------------------------------------------------------------------------------------------------
# compute_all: te falls back to pump duration without rate
# --------------------------------------------------------------------------------------------------
def test_compute_all_no_rate_sets_t_shutin_and_fallback_te():
    td = make_testdata()
    st = PickState(pressure_col=PRESSURE_COL, start_idx=START_IDX, shutin_idx=SHUTIN_IDX)
    res = compute_all(st, td)
    assert res.t_shutin_s == pytest.approx(float(td.t_s[SHUTIN_IDX]))
    assert res.te_s == pytest.approx(float(td.t_s[SHUTIN_IDX] - td.t_s[START_IDX]))
    assert res.qmax_bpm is None and res.vinj is None
    assert any("pump duration" in w for w in res.warnings)
    # te works end to end: the resample + diagnostics pipeline runs
    assert res.resampled is not None
    assert res.diagnostics is not None


def test_compute_all_with_rate_is_unchanged():
    td = make_testdata()
    st = overview_state(td)
    res = compute_all(st, td)
    # effective te = Vinj/qmax, not the wall-clock duration
    assert res.te_s == pytest.approx(interpret.effective_te_seconds(res.vinj, res.qmax_bpm))
    assert res.vinj is not None and res.qmax_bpm is not None
    assert not any("pump duration" in w for w in res.warnings)


def test_compute_all_dead_rate_channel_falls_back_to_pump_duration():
    td = make_testdata()
    td.df[RATE_COL] = 0.0  # rate present but never pumps: qmax = 0 -> no effective te
    st = overview_state(td)
    res = compute_all(st, td)
    assert res.te_s == pytest.approx(float(td.t_s[SHUTIN_IDX] - td.t_s[START_IDX]))
    assert any("pump duration" in w for w in res.warnings)
```

- [x] **Step 2: Run the new tests to verify they fail**

Run: `C:\Users\LucasChristman\.venvs\dfit\Scripts\python.exe -m pytest tests/test_no_rate_fallback.py -k compute_all -v`
Expected: `no_rate` and `dead_rate` FAIL (te_s/t_shutin_s are None); `with_rate` PASSES.

- [x] **Step 3: Restructure the block**

In `dfit_tool/model.py`, replace lines 331-344 (the `# Injection window + te` block) with:

```python
    # Injection window + te. t_shutin_s needs only the two picks; qmax/Vinj (and the
    # effective te = Vinj/qmax) need a rate channel. When the effective te is unavailable
    # (no rate channel, or a degenerate one), te falls back to the wall-clock pump duration
    # (TODO pair: rate-less datasets), with a warning so the analyst knows te is not the
    # Vinj/qmax effective time.
    if state.start_idx is not None and state.shutin_idx is not None:
        start, shutin = state.start_idx, state.shutin_idx
        res.t_shutin_s = float(td.t_s[shutin])
        if res.rate_all is not None:
            res.qmax_bpm = state.qmax_bpm or interpret.max_sustained_rate(res.rate_all, start, shutin)
            vol = td.column(cfg.volume_col) if cfg.volume_col else None
            vr = interpret.injected_volume(td.t_s, res.rate_all, start, shutin, volume=vol)
            res.vinj, res.vinj_delta = vr.vinj, vr.vinj_delta
            res.vinj_integral, res.vinj_source = vr.vinj_integral, vr.source
            res.vinj_disagreement = vr.disagreement_frac
            if vr.disagreement_frac is not None and vr.disagreement_frac > 0.05:
                res.warnings.append(f"Volume delta vs rate-integral disagree {vr.disagreement_frac:.0%}")
            if res.qmax_bpm and res.qmax_bpm > 0:
                res.te_s = interpret.effective_te_seconds(res.vinj, res.qmax_bpm)
        if res.te_s is not None and not (np.isfinite(res.te_s) and res.te_s > 0):
            res.te_s = None  # a NaN/<=0 effective te must not leak into the truthy te gate below
        if res.te_s is None and shutin > start:
            dur = float(td.t_s[shutin] - td.t_s[start])
            if dur > 0:
                res.te_s = dur
                res.warnings.append(
                    "te = pump duration (shut-in - start); no usable rate for Vinj/qmax")
```

- [x] **Step 4: Run the tests to verify they pass**

Run: `C:\Users\LucasChristman\.venvs\dfit\Scripts\python.exe -m pytest tests/test_no_rate_fallback.py tests/test_model.py -v`
Expected: all PASS.

- [x] **Step 5: Commit**

```bash
git add dfit_tool/model.py tests/test_no_rate_fallback.py
git commit -m "Fall back te to the wall-clock pump duration when no usable rate exists"
```

---

### Task 4: overview title without Vinj/qmax

**Files:**
- Modify: `dfit_tool/plots.py:146-148` (`render_overview` title)
- Test: `tests/test_no_rate_fallback.py` (append)

**Interfaces:**
- Consumes: a `DerivedResults` with `te_s` set but `vinj`/`qmax_bpm` `None` (Task 3's no-rate path).

- [x] **Step 1: Write the failing test**

Append to `tests/test_no_rate_fallback.py` (extend imports):

```python
from matplotlib.figure import Figure

from dfit_tool import plots
```

```python
# --------------------------------------------------------------------------------------------------
# plots.render_overview: title must not assume te implies Vinj/qmax
# --------------------------------------------------------------------------------------------------
def test_render_overview_title_with_fallback_te_and_no_rate():
    td = make_testdata()
    st = PickState(pressure_col=PRESSURE_COL, start_idx=START_IDX, shutin_idx=SHUTIN_IDX)
    res = compute_all(st, td)
    assert res.te_s is not None and res.vinj is None  # the crash precondition
    ax = Figure().add_subplot(111)
    plots.render_overview(ax, td, st, res)  # must not raise
    title = ax.get_title()
    assert "te=" in title
    assert "Vinj" not in title and "qmax" not in title
```

- [x] **Step 2: Run the new test to verify it fails**

Run: `C:\Users\LucasChristman\.venvs\dfit\Scripts\python.exe -m pytest tests/test_no_rate_fallback.py -k render_overview -v`
Expected: FAIL with `TypeError` (formatting `None` with `:.1f`).

- [x] **Step 3: Build the title from conditional pieces**

In `dfit_tool/plots.py`, replace lines 146-148:

```python
    title = "Overview"
    if res.te_s:
        title += f"   te={res.te_s/60:.2f} min"
        if res.vinj is not None:
            title += f"   Vinj={res.vinj:.1f} bbl"
        if res.qmax_bpm is not None:
            title += f"   qmax={res.qmax_bpm:.2f} bpm"
```

- [x] **Step 4: Run the full suite**

Run: `C:\Users\LucasChristman\.venvs\dfit\Scripts\python.exe -m pytest`
Expected: all PASS, no new warnings about the overview renderer.

- [x] **Step 5: Commit**

```bash
git add dfit_tool/plots.py tests/test_no_rate_fallback.py
git commit -m "Render the overview title without assuming te implies Vinj/qmax"
```

---

### Task 5: docs — retire the two TODOs

**Files:**
- Modify: `CLAUDE.md` (TODO section + Resampling/domain notes)

**Interfaces:** none (docs only).

- [ ] **Step 1: Update CLAUDE.md**

Remove the two completed TODO bullets:

```
- When no rate is auto-detected, the start/shut-in vlines never appear
- Some datasets don't have rate. Fallback in this case should simply set injection time by the true time between user-marked start and shut-in
```

Add to the domain section (after the **Resampling** paragraph, before **Tail trim**):

```markdown
**No-rate fallback.** When a dataset has no rate channel (or a dead one that never exceeds
the detection threshold), `picks.seed_overview` seeds the start/shut-in vlines from the
pressure shape instead (`interpret.suggest_injection_window_pressure`: shut-in at the
pressure max, start at the last upcross of a 10%-of-rise threshold, positional defaults for
degenerate shapes), so the draggable lines always exist. `compute_all` sets `t_shutin_s`
from the picks alone and, when the effective te (Vinj/qmax) is unavailable, falls back to
te = wall-clock pump duration (shut-in − start) with an appended warning. Vinj and qmax stay
blank without rate; the overview title shows only the pieces that exist.
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "Document the no-rate fallback and retire its two TODOs"
```
