"""parse_datetime fast paths 2-4 run on the still-NaT subset only; the output must equal the
original whole-column form (``io_load._fill_na_exact_reference``) on mixed-format columns."""
import random

import pandas as pd
import pytest

from dfit_tool import io_load


def _random_column(rng: random.Random, n: int) -> list:
    shapes = [
        lambda d: f"{d.month}/{d.day}/{d.year} {d.hour:02d}:{d.minute:02d}:{d.second:02d}",
        lambda d: f"{d.day:02d}/{d.month:02d}/{d.year} {d.hour:02d}:{d.minute:02d}:{d.second:02d}",
        lambda d: f"{d.month}/{d.day}/{d.year} {(d.hour % 12) or 12:02d}:{d.minute:02d}:"
                  f"{d.second:02d} {'AM' if d.hour < 12 else 'PM'}",
        lambda d: f"{d.month}/{d.day}/{d.year} {d.hour:02d}:{d.minute:02d}:{d.second:02d}.{d.microsecond // 1000:03d}",
        lambda d: f"{d.month:02d}-{d.day:02d}-{d.year}_{d.hour:02d}:{d.minute:02d}:{d.second:02d}",
        lambda d: "",
        lambda d: "(date time)",
        lambda d: "45123.5",
        lambda d: "2019-08-29 14:13:35",
        lambda d: "garbage",
    ]
    # Weighted so the common case is mostly one shape with a handful of odd rows.
    main = rng.choice(shapes[:5])
    base = pd.Timestamp("2021-03-01") + pd.to_timedelta(rng.randrange(0, 400), unit="D")
    out = []
    for i in range(n):
        d = base + pd.to_timedelta(i * rng.choice([1, 1, 30]), unit="s")
        f = main if rng.random() < 0.8 else rng.choice(shapes)
        out.append(f(d))
    return out


@pytest.mark.parametrize("seed", range(60))
def test_subset_fast_paths_match_whole_column(seed, monkeypatch):
    rng = random.Random(seed)
    col = pd.Series(_random_column(rng, rng.choice([5, 40, 300])), dtype="object")
    new = io_load.parse_datetime(col)
    monkeypatch.setattr(io_load, "_fill_na_exact", io_load._fill_na_exact_reference)
    old = io_load.parse_datetime(col)
    pd.testing.assert_series_equal(new, old)
