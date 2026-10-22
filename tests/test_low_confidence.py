"""Low confidence flags (G-function and Tangent steps): informational, logged, persisted."""
from __future__ import annotations

import json

from dfit_tool import model, store
from dfit_tool.model import PickState, compute_all
from tests.test_stiffness import _panel_visibility_stub
from tests.test_uninterpretable import _with_stiffness_pick


def _row(tmp_path, st, td, res):
    entry = store.TestEntry(test_id="well1", folder=str(tmp_path))
    return store.build_log_row(entry, str(tmp_path / "well1.csv"), str(tmp_path), st, td, res)


def test_defaults_false_and_columns_at_tail():
    st = PickState()
    assert st.compliance_low_confidence is False
    assert st.tangent_low_confidence is False
    assert st.loglog_low_confidence is False
    assert store.LOG_COLUMNS[-4:-1] == ["compliance_low_confidence", "tangent_low_confidence",
                                      "loglog_low_confidence"]


def test_log_row_carries_flags(tmp_path):
    td, st, res = _with_stiffness_pick()
    row = _row(tmp_path, st, td, res)
    assert row["compliance_low_confidence"] is False
    assert row["tangent_low_confidence"] is False
    assert row["loglog_low_confidence"] is False

    st.compliance_low_confidence = True
    st.tangent_low_confidence = True
    st.loglog_low_confidence = True
    row = _row(tmp_path, st, td, compute_all(st, td))
    assert row["compliance_low_confidence"] is True
    assert row["tangent_low_confidence"] is True
    assert row["loglog_low_confidence"] is True


def test_flags_do_not_change_results(tmp_path):
    td, st, res = _with_stiffness_pick()
    base = _row(tmp_path, st, td, res)
    st.compliance_low_confidence = True
    st.tangent_low_confidence = True
    st.loglog_low_confidence = True
    flagged = _row(tmp_path, st, td, compute_all(st, td))
    for k in base:
        if k not in ("compliance_low_confidence", "tangent_low_confidence",
                     "loglog_low_confidence"):
            assert flagged[k] == base[k] or (flagged[k] != flagged[k] and base[k] != base[k]), k


def test_round_trip_and_legacy_save():
    st = PickState()
    st.compliance_low_confidence = True
    st.tangent_low_confidence = True
    st.loglog_low_confidence = True
    back = model._decode(json.loads(json.dumps(model._encode(st))))
    assert back.compliance_low_confidence is True
    assert back.tangent_low_confidence is True
    assert back.loglog_low_confidence is True

    legacy = model._decode({"well_name": "w1"})
    assert legacy.compliance_low_confidence is False
    assert legacy.tangent_low_confidence is False
    assert legacy.loglog_low_confidence is False


def test_panel_visibility_resyncs_vars():
    stub = _panel_visibility_stub()
    stub.state.compliance_low_confidence = True
    stub.state.tangent_low_confidence = True
    stub.state.loglog_low_confidence = True

    stub.step = "gfunction"
    stub._update_panel_visibility()
    assert stub.var_compliance_low_conf.get() is True

    stub.step = "tangent"
    stub._update_panel_visibility()
    assert stub.var_tangent_low_conf.get() is True

    stub.step = "loglog"
    stub._update_panel_visibility()
    assert stub.var_loglog_low_conf.get() is True
