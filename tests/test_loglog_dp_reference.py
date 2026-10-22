"""Log-log dp reference: dp = apparent ISIP - P, so a water-hammer trough at the first
post-shut-in sample cannot push the whole early falloff negative (and off the log axis)."""

from __future__ import annotations

import numpy as np

from dfit_tool import picks, resample
from dfit_tool.model import compute_all
from tests.helpers import injection_state, make_testdata


def _hammer_falloff():
    """Shut-in at 5000, a trough to 4700 at t = 1 s, then settled near 4950 and declining."""
    dt = np.arange(0.0, 3601.0, 60.0)
    dt[1] = 1.0
    p = 4950.0 - 300.0 * (1.0 - np.exp(-dt / 2000.0))
    p[0] = 5000.0
    p[1] = 4700.0
    # Built directly: the tail guard would stop resample_pressure_increment at the rebound.
    return resample.Resampled(dt=dt, p=p, n_raw=len(dt), guarded_at=None, guard_dt=None)


def test_dp_uses_reference_pressure():
    rs = _hammer_falloff()
    dg = resample.diagnostics(rs, te=300.0, p_ref=5000.0)
    np.testing.assert_allclose(dg.dp, 5000.0 - dg.p)
    assert (dg.dp > 0).all()


def test_dp_without_reference_keeps_first_sample():
    rs = _hammer_falloff()
    dg = resample.diagnostics(rs, te=300.0)
    np.testing.assert_allclose(dg.dp, dg.p[0] - dg.p)
    dg_nan = resample.diagnostics(rs, te=300.0, p_ref=float("nan"))
    np.testing.assert_allclose(dg_nan.dp, dg.dp)


def test_compute_all_references_dp_to_apparent_isip():
    td = make_testdata()
    st = injection_state(td)
    picks.seed_injection(st, td)
    res = compute_all(st, td)
    picks.seed_isip(st, td, res)
    for use_tangent in (False, True):
        st.isip_use_tangent = use_tangent
        res = compute_all(st, td)
        assert res.apparent_isip is not None
        np.testing.assert_allclose(res.diagnostics.dp, res.apparent_isip - res.diagnostics.p)
