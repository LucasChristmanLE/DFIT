"""interpret.h_function (mask-free, in place) vs the original masked implementation."""
import numpy as np
import pytest

from dfit_tool import interpret


@pytest.mark.parametrize("seed", range(40))
def test_h_function_matches_reference(seed):
    rng = np.random.default_rng(seed)
    n = int(rng.choice([1, 2, 3, 10, 57, 300]))
    steps = rng.exponential(5.0, size=n)
    steps[rng.random(n) < 0.2] = 0.0  # tied timestamps
    dt_s = np.cumsum(steps) + rng.uniform(0, 50)
    p_eff = rng.uniform(500, 6000, size=n)
    pp = float(rng.uniform(0, 4000))
    te = float(rng.uniform(1, 2000))
    new = interpret.h_function(dt_s, p_eff, pp, te)
    old = interpret._h_function_reference(dt_s, p_eff, pp, te)
    np.testing.assert_allclose(new, old, rtol=1e-12, atol=0)
    assert np.array_equal(new, old)  # bitwise in practice: same sqrt args, same matmul


def test_h_function_nan_propagates_like_reference():
    dt_s = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    p_eff = np.array([3000.0, 2900.0, 2800.0, 2700.0, 2600.0])
    new = interpret.h_function(dt_s, p_eff, 1000.0, 100.0)
    old = interpret._h_function_reference(dt_s, p_eff, 1000.0, 100.0)
    np.testing.assert_array_equal(new, old)
