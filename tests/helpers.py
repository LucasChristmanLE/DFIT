"""Synthetic-data builders shared across the test suite.

``make_testdata`` builds an ``io_load.TestData`` directly from in-memory arrays (the same
fields ``load_csv`` populates) instead of round-tripping through a temp CSV -- ``TestData``
is a plain dataclass with no validation logic in ``__init__``, so constructing it directly
is both faster and exercises the same object the app uses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from dfit_tool.io_load import TestData
from dfit_tool.model import PickState

PRESSURE_COL = "PRESSURE"
RATE_COL = "RATE"
VOLUME_COL = "VOLUME"
DATETIME_COL = "DATETIME"

START_IDX = 100
SHUTIN_IDX = 300


def make_testdata(n: int = 600, dt: float = 1.0) -> TestData:
    """A synthetic DFIT-shaped `TestData`: rate ramps up then down between START_IDX and
    SHUTIN_IDX, pressure rises during injection then falls off after shut-in, volume is the
    running integral of rate.
    """
    t_s = np.arange(n, dtype=float) * dt

    rate = np.zeros(n)
    # Ramp 0 -> 8 bpm over the first half of the injection window, 8 -> 0 over the second half.
    half = (SHUTIN_IDX - START_IDX) // 2
    rate[START_IDX:START_IDX + half] = np.linspace(0.0, 8.0, half)
    rate[START_IDX + half:SHUTIN_IDX] = np.linspace(8.0, 0.0, SHUTIN_IDX - START_IDX - half)

    volume = np.cumsum(rate) * dt / 60.0

    pressure = np.full(n, 2000.0)
    pressure[START_IDX:SHUTIN_IDX] = 2000.0 + 3000.0 * np.linspace(0.0, 1.0, SHUTIN_IDX - START_IDX)
    post = n - SHUTIN_IDX
    decline_t = np.arange(post, dtype=float) * dt
    pressure[SHUTIN_IDX:] = 5000.0 - 1500.0 * (1.0 - np.exp(-decline_t / 200.0))

    df = pd.DataFrame({
        PRESSURE_COL: pressure,
        RATE_COL: rate,
        VOLUME_COL: volume,
    })
    columns = list(df.columns)

    return TestData(
        path="<synthetic>",
        df=df,
        datetime_col=DATETIME_COL,
        t_s=t_s,
        columns=columns,
    )


def overview_state(td: TestData) -> PickState:
    """A minimal `PickState` pointing at the synthetic channels with start/shut-in picked."""
    return PickState(
        pressure_col=PRESSURE_COL,
        rate_col=RATE_COL,
        volume_col=VOLUME_COL,
        start_idx=START_IDX,
        shutin_idx=SHUTIN_IDX,
    )
