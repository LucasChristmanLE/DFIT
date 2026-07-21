from tests.helpers import make_testdata, overview_state


def test_harness_builds_testdata():
    td = make_testdata()
    assert td.n == 600
    assert "PRESSURE" in td.columns
    st = overview_state(td)
    assert st.start_idx == 100 and st.shutin_idx == 300
