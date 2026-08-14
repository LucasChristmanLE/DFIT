"""Tests for dfit_tool/units.py: the conversion constants, factor tables, token normalization,
and the alias lookups (lookup_pressure/lookup_rate/lookup_volume).
"""

import pytest

from dfit_tool import units


# --------------------------------------------------------------------------------------------------
# constants / factor tables
# --------------------------------------------------------------------------------------------------
def test_pressure_factors_exact_values():
    assert units.PRESSURE_FACTORS["psi"] == 1.0
    assert units.PRESSURE_FACTORS["kpa"] == pytest.approx(0.1450377377)
    assert units.PRESSURE_FACTORS["mpa"] == pytest.approx(145.0377377)
    assert units.PRESSURE_FACTORS["bar"] == pytest.approx(14.50377)


def test_rate_factors_exact_values():
    assert units.RATE_FACTORS["bpm"] == 1.0
    assert units.RATE_FACTORS["m3/min"] == pytest.approx(6.2898107704)


def test_volume_factors_exact_values():
    assert units.VOLUME_FACTORS["bbl"] == 1.0
    assert units.VOLUME_FACTORS["m3"] == pytest.approx(6.2898107704)


def test_m_to_ft_and_kgm3_to_ppg_constants():
    assert units.M_TO_FT == pytest.approx(3.280839895)
    assert units.KGM3_TO_PPG == pytest.approx(1.0 / 119.8264)


# --------------------------------------------------------------------------------------------------
# _normalize_token
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("raw,expected", [
    ("KPAg", "kpag"),
    ("kPa g", "kpag"),
    ("m³/min", "m3/min"),
    ("m^3/min", "m3/min"),
    ("M3 / MIN", "m3/min"),
    ("kg·m-3", "kgm-3"),
    ("  psi  ", "psi"),
])
def test_normalize_token(raw, expected):
    assert units._normalize_token(raw) == expected


# --------------------------------------------------------------------------------------------------
# lookup_pressure
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("token", ["psi", "PSI", "psia", "psig"])
def test_lookup_pressure_psi_variants(token):
    assert units.lookup_pressure(token) == ("psi", 1.0)


@pytest.mark.parametrize("token", ["kpa", "KPAg", "kPa g", "kpaa"])
def test_lookup_pressure_kpa_variants(token):
    canon, factor = units.lookup_pressure(token)
    assert canon == "kpa"
    assert factor == pytest.approx(0.1450377377)


@pytest.mark.parametrize("token", ["mpa", "MPAg", "mpaa"])
def test_lookup_pressure_mpa_variants(token):
    canon, factor = units.lookup_pressure(token)
    assert canon == "mpa"
    assert factor == pytest.approx(145.0377377)


@pytest.mark.parametrize("token", ["bar", "barg", "BARA"])
def test_lookup_pressure_bar_variants(token):
    canon, factor = units.lookup_pressure(token)
    assert canon == "bar"
    assert factor == pytest.approx(14.50377)


def test_lookup_pressure_unrecognized_returns_none():
    assert units.lookup_pressure("furlongs") is None
    assert units.lookup_pressure("auto") is None


# --------------------------------------------------------------------------------------------------
# lookup_rate
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("token", ["bpm", "BPM", "bbl/min", "bbls/min", "bbl / min"])
def test_lookup_rate_bpm_variants(token):
    assert units.lookup_rate(token) == ("bpm", 1.0)


@pytest.mark.parametrize("token", ["m3/min", "M3/MIN", "m^3/min", "m³/min", "m3min", "m3/m"])
def test_lookup_rate_m3min_variants(token):
    canon, factor = units.lookup_rate(token)
    assert canon == "m3/min"
    assert factor == pytest.approx(6.2898107704)


def test_lookup_rate_unrecognized_returns_none():
    assert units.lookup_rate("gpm") is None


# --------------------------------------------------------------------------------------------------
# lookup_volume
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("token", ["bbl", "BBL", "bbls"])
def test_lookup_volume_bbl_variants(token):
    assert units.lookup_volume(token) == ("bbl", 1.0)


@pytest.mark.parametrize("token", ["m3", "M3", "m^3", "m³"])
def test_lookup_volume_m3_variants(token):
    canon, factor = units.lookup_volume(token)
    assert canon == "m3"
    assert factor == pytest.approx(6.2898107704)


def test_lookup_volume_unrecognized_returns_none():
    assert units.lookup_volume("gal") is None
