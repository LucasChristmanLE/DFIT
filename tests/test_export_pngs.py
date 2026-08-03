import os

from matplotlib.figure import Figure

from dfit_tool.model import compute_all
from dfit_tool import plots
from tests.helpers import make_testdata, overview_state


def test_save_all_step_pngs_writes_six_nonempty_files(tmp_path):
    td = make_testdata()
    state = overview_state(td)
    res = compute_all(state, td)

    paths = plots.save_all_step_pngs(str(tmp_path), td, state, res, views={})

    assert len(paths) == 6
    expected = [
        "1_overview.png", "2_isip.png", "3_gfunction.png",
        "4_tangent.png", "5_loglog.png", "6_porepressure.png",
    ]
    for name in expected:
        full = tmp_path / name
        assert full.exists()
        assert full.stat().st_size > 0


def test_render_step_figure_applies_stored_view():
    td = make_testdata()
    state = overview_state(td)
    res = compute_all(state, td)

    fig = plots.render_step_figure(
        "overview", td, state, res,
        stored_view=((0.0, 5.0), (100.0, 200.0), None))

    ax = fig.axes[0]
    assert ax.get_xlim() == (0.0, 5.0)
    assert ax.get_ylim() == (100.0, 200.0)


def test_render_step_figure_gfunction_clamps_default_view():
    td = make_testdata()
    state = overview_state(td)
    res = compute_all(state, td)

    # Reproduce the renderer's own defaults to compare against.
    probe = Figure().add_subplot(111)
    defaults = plots.RENDERERS["gfunction"](probe, td, state, res)

    fig = plots.render_step_figure("gfunction", td, state, res)
    # gfunction overrides full_y with the renderer's data-driven ylim (ui.py:646-651).
    assert fig.axes[0].get_ylim() == defaults.ylim
    # ...and clamps the derivative twin to 0..500 (ui.py:654-657).
    twin = next(a for a in fig.axes if a is not fig.axes[0])
    lo, hi = twin.get_ylim()
    assert lo >= 0.0 and hi <= 500.0


def test_render_step_figure_applies_stored_view_twin():
    td = make_testdata()
    state = overview_state(td)
    res = compute_all(state, td)

    fig = plots.render_step_figure(
        "gfunction", td, state, res,
        stored_view=((1.0, 3.0), (500.0, 900.0), (10.0, 120.0)))

    assert fig.axes[0].get_xlim() == (1.0, 3.0)
    assert fig.axes[0].get_ylim() == (500.0, 900.0)
    twin = next(a for a in fig.axes if a is not fig.axes[0])
    assert twin.get_ylim() == (10.0, 120.0)
