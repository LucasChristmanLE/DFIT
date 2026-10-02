"""The headless workflow-steps interface in model: the step list, and which steps a test's
interpretation leaves out (PC-F skips porepressure and stiffness). ui, store, and plots all
answer "is this step part of the workflow" through these functions."""

from __future__ import annotations

import pytest

from dfit_tool import model, plots, store, ui
from dfit_tool.model import PickState, STEP_KEYS, last_step, resolve_step, skipped_steps


def _pcf() -> PickState:
    return PickState(postclosure_scenario="PC-F no peak")


@pytest.mark.parametrize("scenario", ["", "PC-A linear", "PC-B false-radial", "PC-E no trend"])
def test_skipped_steps_empty_unless_pcf(scenario):
    assert skipped_steps(PickState(postclosure_scenario=scenario)) == frozenset()


def test_skipped_steps_pcf_skips_porepressure_and_stiffness():
    assert skipped_steps(_pcf()) == frozenset({"porepressure", "stiffness"})


def test_last_step_is_stiffness_normally_and_loglog_under_pcf():
    assert last_step(PickState()) == "stiffness"
    assert last_step(_pcf()) == "loglog"


def test_resolve_step_passes_active_steps_through():
    for key in STEP_KEYS:
        assert resolve_step(PickState(), key) == key


@pytest.mark.parametrize("key", ["porepressure", "stiffness"])
def test_resolve_step_redirects_skipped_step_to_loglog_under_pcf(key):
    assert resolve_step(_pcf(), key) == "loglog"


def test_step_keys_is_the_single_step_list():
    assert STEP_KEYS == tuple(k for k, _ in model.STEPS)
    assert store.STEP_KEYS is STEP_KEYS
    assert ui.STEPS is model.STEPS
    assert set(plots.RENDERERS) == set(STEP_KEYS)
