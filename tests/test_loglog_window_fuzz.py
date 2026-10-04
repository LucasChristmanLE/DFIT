"""interpret.suggest_loglog_window (loop over start, vectorize over end; O(m) memory) must match
the original all-pairs implementation exactly on random log-log curves."""
import numpy as np
import pytest

from dfit_tool import interpret


def _curve(rng: np.random.Generator, n: int):
    t = np.cumsum(rng.exponential(1.0, n)) * rng.uniform(1, 30) + 1.0
    lt = np.log10(t)
    # Piecewise log-log shape: rise to a peak, then 1-3 segments of random slope, plus noise.
    peak = int(rng.integers(n // 10 + 1, max(n // 3, n // 10 + 2)))
    knots = np.sort(rng.choice(np.arange(peak + 1, n), size=min(2, max(n - peak - 1, 0)),
                               replace=False)) if n - peak > 2 else np.array([], dtype=int)
    edges = [peak, *knots.tolist(), n - 1]
    lv = np.empty(n)
    lv[: peak + 1] = 1.0 + 0.8 * (lt[: peak + 1] - lt[0]) * rng.uniform(0.3, 1.5)
    level = lv[peak]
    for lo, hi in zip(edges[:-1], edges[1:]):
        slope = rng.choice([-0.5, -1.0, rng.uniform(-1.5, 0.2)])
        seg = np.arange(lo + 1, hi + 1)
        lv[seg] = level + slope * (lt[seg] - lt[lo])
        level = lv[hi]
    lv += rng.normal(0, rng.choice([0.0, 0.01, 0.05, 0.2]), n)
    y = 10.0 ** lv
    if rng.random() < 0.3:  # some non-positive / non-finite samples
        y[rng.integers(0, n, size=3)] = rng.choice([0.0, -1.0, np.nan], size=3)
    return t, y


@pytest.mark.parametrize("seed", range(120))
def test_suggest_loglog_window_matches_reference(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.choice([4, 8, 15, 40, 120, 400]))
    t, y = _curve(rng, n)
    assert (interpret.suggest_loglog_window(t, y)
            == interpret._suggest_loglog_window_reference(t, y))


def test_suggest_loglog_window_matches_reference_early_peak_full_size():
    """Peak at sample 6 so the post-peak search really spans ~1500 samples."""
    n = 1500
    t = np.logspace(0, 4, n)
    lv = np.where(np.arange(n) < 6, 1 + 0.3 * np.arange(n),
                  2.5 - 0.5 * (np.log10(t) - np.log10(t[6])))
    lv[6] = 2.6
    rng = np.random.default_rng(7)
    y = 10.0 ** (lv + rng.normal(0, 0.01, n))
    new = interpret.suggest_loglog_window(t, y)
    assert new is not None and new[1] - new[0] > 1000
    assert new == interpret._suggest_loglog_window_reference(t, y)


@pytest.mark.parametrize("seed", range(3))
def test_suggest_loglog_window_matches_reference_large(seed):
    rng = np.random.default_rng(1000 + seed)
    t, y = _curve(rng, 1500)
    assert (interpret.suggest_loglog_window(t, y)
            == interpret._suggest_loglog_window_reference(t, y))
