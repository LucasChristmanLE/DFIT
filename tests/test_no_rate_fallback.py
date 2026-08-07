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
