"""Tests for dfit_tool/questionnaire.py: label-anchored Q&A parsing of the DFIT Questionnaire
template. Fixtures are built in-test with openpyxl, mimicking the two real-world layouts on hand
(Abraxas: bare numbers, no unit labels; PDC: MD:/TVD: labeled rows, SG-labeled density) without
referencing the actual sample files.
"""

import openpyxl
import pytest

from dfit_tool import units
from dfit_tool.questionnaire import find_questionnaire, parse_questionnaire


def _make_xlsx(path, rows, sheet_name="Sheet1"):
    """Write `rows` (a list of column-A strings/numbers) one per row into a new workbook."""
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = sheet_name
    for i, value in enumerate(rows, start=1):
        ws.cell(row=i, column=1, value=value)
    wb.save(path)
    return path


def _make_multi_sheet_xlsx(path, sheets):
    """`sheets`: list of `(sheet_name, rows)` pairs -- one worksheet per pair, column-A rows
    written the same way `_make_xlsx` does for a single sheet."""
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for name, rows in sheets:
        ws = wb.create_sheet(name)
        for i, value in enumerate(rows, start=1):
            ws.cell(row=i, column=1, value=value)
    wb.save(path)
    return path


# --------------------------------------------------------------------------------------------------
# Abraxas-style layout: bare numbers, lbs/gal density
# --------------------------------------------------------------------------------------------------
def test_abraxas_style_density_and_tvd(tmp_path):
    path = _make_xlsx(tmp_path / "LOS DFIT Questionnaire_Foo.xlsx", [
        "Well Name:",
        "Foo State 1H",
        "Formation:",
        "Eagle Ford",
        "Type and density of fluid in the wellbore?",
        "3% KCl - 8.4 lbs/gal",
        "Planned Perforations (MD and TVD):",
        "15887'",
        "10958'",
        "Section to be completed by LOS Service Leader",
        "Actual perforation depth used for the DFIT:",
        "15887'",
    ])
    result = parse_questionnaire(str(path))

    assert result.density_ppg == pytest.approx(8.4)
    assert result.density_source == "3% KCl - 8.4 lbs/gal"
    assert result.tvd_ft == pytest.approx(10958.0)
    assert result.tvd_source == "10958'"
    assert result.well_name == "Foo State 1H"
    assert result.formation == "Eagle Ford"
    # the "actual perforation depth" block has only the one bare MD number, which must not be
    # mistaken for TVD -- the parser should fall through to the "planned perforations" block and
    # note why it skipped the preferred block.
    assert any("one depth value" in w for w in result.warnings)


# --------------------------------------------------------------------------------------------------
# PDC-style layout: MD:/TVD: labeled rows, SG-labeled density coerced to ppg
# --------------------------------------------------------------------------------------------------
def test_pdc_style_density_coerced_from_specific_gravity(tmp_path):
    path = _make_xlsx(tmp_path / "PDC Energy DFIT Questionnaire_Foo.xlsx", [
        "Type and density of fluid in the wellbore?",
        "Saturated Oil",
        "Planned Perforations (MD and TVD): Toesleeve Conversions",
        "MD: 21833'",
        "TVD: 10929.8'",
        "Section to be completed by LOS Service Leader",
        "What type of fluid was pumped:",
        "Claypex 650 - 8.41 Specific gravity",
        "Actual perforation depth used for the DFIT",
        "MD: 21833'",
        "TVD: 10929.8'",
    ], sheet_name="Questionnaire")
    result = parse_questionnaire(str(path))

    assert result.density_ppg == pytest.approx(8.41)
    assert result.density_source == "Claypex 650 - 8.41 Specific gravity"
    assert result.tvd_ft == pytest.approx(10929.8)
    assert result.tvd_source == "TVD: 10929.8'"
    # 8.41 is in the ppg range, not the SG range, so using it as ppg-as-is is a coercion that
    # should be flagged rather than silently assumed.
    assert any("Specific gravity" in w for w in result.warnings)


# --------------------------------------------------------------------------------------------------
# density interpretation edge cases
# --------------------------------------------------------------------------------------------------
def test_true_specific_gravity_converted_to_ppg(tmp_path):
    path = _make_xlsx(tmp_path / "Questionnaire.xlsx", [
        "Type and density of fluid in the wellbore?",
        "Produced water - 1.02 SG",
    ])
    result = parse_questionnaire(str(path))
    assert result.density_ppg == pytest.approx(1.02 * 8.345)
    assert result.density_source == "Produced water - 1.02 SG"


def test_density_gradient_psi_per_ft_converted_to_ppg(tmp_path):
    path = _make_xlsx(tmp_path / "Questionnaire.xlsx", [
        "Type and density of fluid in the wellbore?",
        "3% KCl - 0.433 psi/ft",
    ])
    result = parse_questionnaire(str(path))
    assert result.density_ppg == pytest.approx(0.433 / 0.052)
    assert result.density_source == "3% KCl - 0.433 psi/ft"


def test_density_gradient_psi_per_ft_word_variant(tmp_path):
    path = _make_xlsx(tmp_path / "Questionnaire.xlsx", [
        "Type and density of fluid in the wellbore?",
        "0.45 psi per ft",
    ])
    result = parse_questionnaire(str(path))
    assert result.density_ppg == pytest.approx(0.45 / 0.052)


@pytest.mark.parametrize("cell", ["0.44 psi/foot", "0.44 psi per foot"])
def test_density_gradient_foot_variants(tmp_path, cell):
    path = _make_xlsx(tmp_path / "Questionnaire.xlsx", [
        "Type and density of fluid in the wellbore?",
        cell,
    ])
    result = parse_questionnaire(str(path))
    assert result.density_ppg == pytest.approx(0.44 / 0.052)


def test_density_gradient_out_of_range_ignored(tmp_path):
    path = _make_xlsx(tmp_path / "Questionnaire.xlsx", [
        "Type and density of fluid in the wellbore?",
        "2.0 psi/ft",
    ])
    result = parse_questionnaire(str(path))
    assert result.density_ppg is None
    assert any("outside the expected ppg range" in w for w in result.warnings)


def test_no_density_anywhere_returns_none_with_warning(tmp_path):
    path = _make_xlsx(tmp_path / "Questionnaire.xlsx", [
        "Type and density of fluid in the wellbore?",
        "Saturated Oil",
        "What type of fluid was pumped:",
        "Slickwater",
    ])
    result = parse_questionnaire(str(path))
    assert result.density_ppg is None
    assert result.density_source is None
    assert any("no parseable fluid density" in w for w in result.warnings)


# --------------------------------------------------------------------------------------------------
# TVD edge cases
# --------------------------------------------------------------------------------------------------
def test_single_bare_number_under_perfs_gives_no_tvd(tmp_path):
    path = _make_xlsx(tmp_path / "Questionnaire.xlsx", [
        "Actual perforation depth used for the DFIT:",
        "15887'",
    ])
    result = parse_questionnaire(str(path))
    assert result.tvd_ft is None
    assert any("one depth value" in w for w in result.warnings)


def test_tvd_greater_than_md_warns_but_still_used(tmp_path):
    path = _make_xlsx(tmp_path / "Questionnaire.xlsx", [
        "Planned Perforations (MD and TVD):",
        "10000'",
        "12000'",
    ])
    result = parse_questionnaire(str(path))
    assert result.tvd_ft == pytest.approx(12000.0)
    assert any("exceeds MD" in w for w in result.warnings)


def test_no_tvd_anywhere_returns_none_with_warning(tmp_path):
    path = _make_xlsx(tmp_path / "Questionnaire.xlsx", [
        "Well Name:",
        "Foo",
    ])
    result = parse_questionnaire(str(path))
    assert result.tvd_ft is None
    assert any("no parseable TVD" in w for w in result.warnings)
    assert result.well_name == "Foo"


def test_combined_md_tvd_cell_uses_number_after_tvd(tmp_path):
    path = _make_xlsx(tmp_path / "Questionnaire.xlsx", [
        "Planned Perforations (MD and TVD):",
        "MD 21833' / TVD 10929.8'",
    ])
    result = parse_questionnaire(str(path))
    assert result.tvd_ft == pytest.approx(10929.8)
    assert result.tvd_source == "MD 21833' / TVD 10929.8'"


def test_malformed_perfs_cell_is_harmlessly_skipped(tmp_path):
    # "," used to match the bare-footage regex's old [\d,]+ class with no digits, so float() on
    # it would raise and get caught by parse_questionnaire's per-field try/except -- but that
    # blanket catch also aborted the loop over answer blocks, discarding a good TVD living in a
    # later block (see test_junk_comma_cell_does_not_block_labeled_tvd_in_next_block below).
    # _NUMBER_RE/_BARE_FOOTAGE_RE now both require a leading digit, so this cell is just skipped
    # (no bare-footage candidate at all), never raising in the first place.
    path = _make_xlsx(tmp_path / "Questionnaire.xlsx", [
        "Actual perforation depth used for the DFIT:",
        ",",
    ])
    result = parse_questionnaire(str(path))
    assert result.tvd_ft is None
    assert any("no parseable TVD" in w for w in result.warnings)


def test_junk_comma_cell_does_not_block_labeled_tvd_in_next_block(tmp_path):
    # Regression guard: a junk "," cell under the preferred (actual-perfs) block must not raise
    # and abort the whole _extract_tvd loop before the fallback (planned-perfs) block, which
    # carries a perfectly good labeled TVD, ever gets tried.
    path = _make_xlsx(tmp_path / "Questionnaire.xlsx", [
        "Actual perforation depth used for the DFIT:",
        ",",
        "Planned Perforations (MD and TVD):",
        "TVD: 10929.8'",
    ])
    result = parse_questionnaire(str(path))
    assert result.tvd_ft == pytest.approx(10929.8)


# --------------------------------------------------------------------------------------------------
# well name / formation
# --------------------------------------------------------------------------------------------------
def test_well_name_and_formation_absent_returns_none(tmp_path):
    path = _make_xlsx(tmp_path / "Questionnaire.xlsx", [
        "Type and density of fluid in the wellbore?",
        "Saturated Oil",
    ])
    result = parse_questionnaire(str(path))
    assert result.well_name is None
    assert result.formation is None


def test_formation_hydrocarbon_label_does_not_spill_into_formation(tmp_path):
    # "Formation Hydrocarbon GOR:" is a longer, more specific known label than "Formation", so the
    # longest-prefix rule in _label_key must route its answer to its own block, not "formation".
    path = _make_xlsx(tmp_path / "Questionnaire.xlsx", [
        "Formation:",
        "Eagle Ford",
        "Formation Hydrocarbon GOR:",
        "1200",
    ])
    result = parse_questionnaire(str(path))
    assert result.formation == "Eagle Ford"


# --------------------------------------------------------------------------------------------------
# find_questionnaire
# --------------------------------------------------------------------------------------------------
def test_find_questionnaire_same_directory(tmp_path):
    data_dir = tmp_path / "well_data"
    data_dir.mkdir()
    quest = data_dir / "DFIT Questionnaire_Well.xlsx"
    quest.write_text("placeholder")
    csv_path = data_dir / "well.csv"
    csv_path.write_text("t,p\n")

    found, warns = find_questionnaire(str(csv_path))
    assert found == str(quest)
    assert warns == []


def test_find_questionnaire_parent_directory(tmp_path):
    quest = tmp_path / "Questionnaire_Well.xlsx"
    quest.write_text("placeholder")
    data_dir = tmp_path / "well_data"
    data_dir.mkdir()
    csv_path = data_dir / "well.csv"
    csv_path.write_text("t,p\n")

    found, warns = find_questionnaire(str(csv_path))
    assert found == str(quest)
    assert warns == []


def test_find_questionnaire_prefers_same_directory_over_parent(tmp_path):
    parent_quest = tmp_path / "Questionnaire_Well.xlsx"
    parent_quest.write_text("placeholder")
    data_dir = tmp_path / "well_data"
    data_dir.mkdir()
    same_dir_quest = data_dir / "DFIT Questionnaire_Well.xlsx"
    same_dir_quest.write_text("placeholder")
    csv_path = data_dir / "well.csv"
    csv_path.write_text("t,p\n")

    found, warns = find_questionnaire(str(csv_path))
    assert found == str(same_dir_quest)
    assert warns == []


def test_find_questionnaire_skips_lock_files(tmp_path):
    data_dir = tmp_path / "well_data"
    data_dir.mkdir()
    lock = data_dir / "~$Questionnaire_Well.xlsx"
    lock.write_text("placeholder")
    csv_path = data_dir / "well.csv"
    csv_path.write_text("t,p\n")

    found, warns = find_questionnaire(str(csv_path))
    assert found is None
    assert warns == []


def test_find_questionnaire_miss(tmp_path):
    data_dir = tmp_path / "well_data"
    data_dir.mkdir()
    csv_path = data_dir / "well.csv"
    csv_path.write_text("t,p\n")

    found, warns = find_questionnaire(str(csv_path))
    assert found is None
    assert warns == []


def test_find_questionnaire_multiple_matches_warns_and_picks_first_sorted(tmp_path):
    data_dir = tmp_path / "well_data"
    data_dir.mkdir()
    (data_dir / "A Questionnaire.xlsx").write_text("placeholder")
    (data_dir / "B Questionnaire.xlsx").write_text("placeholder")
    csv_path = data_dir / "well.csv"
    csv_path.write_text("t,p\n")

    found, warns = find_questionnaire(str(csv_path))

    assert found == str(data_dir / "A Questionnaire.xlsx")
    assert len(warns) == 1
    assert "multiple questionnaire files" in warns[0]


# --------------------------------------------------------------------------------------------------
# meter-token TVD (Strathcona shape: "TVD: 3368 mTVD" -- today's parser silently accepts 3368 as
# feet, a ~7,700 ft error feeding the hydrostatic head; see ../CLAUDE.md's Context)
# --------------------------------------------------------------------------------------------------
def test_tvd_meters_with_mtvd_suffix_converts_to_feet(tmp_path):
    path = _make_xlsx(tmp_path / "Questionnaire.xlsx", [
        "Planned Perforations (MD and TVD):",
        "TVD: 3368 mTVD",
    ])
    result = parse_questionnaire(str(path))

    # Regression guard: the value must change vs. today's silent 3368-as-ft acceptance --
    # 3368 alone is inside the 1000-25000 ft window, so a bug here would pass unnoticed.
    assert result.tvd_ft == pytest.approx(3368 * units.M_TO_FT, rel=1e-6)
    assert result.tvd_ft == pytest.approx(11050.0, abs=1.0)
    assert any("converted" in w.lower() for w in result.warnings)


def test_tvd_meters_no_space_mkb_suffix_converts_to_feet(tmp_path):
    path = _make_xlsx(tmp_path / "Questionnaire.xlsx", [
        "Planned Perforations (MD and TVD):",
        "TVD: 6399.88mKB",
    ])
    result = parse_questionnaire(str(path))

    assert result.tvd_ft == pytest.approx(6399.88 * units.M_TO_FT, rel=1e-6)


def test_combined_metric_md_tvd_cell_uses_tvd_not_md(tmp_path):
    # A combined "MD: 6656 mMD / TVD: 3368 mTVD" cell has two "tvd" occurrences: the genuine
    # "TVD:" label, and the unit-suffix tail of the trailing "mTVD" token (m-preceded, attached to
    # the "3368" before it). The label occurrence is used directly -- the "after tvd" branch finds
    # "3368" right after it -- so this takes the label path, not the whole-cell fallback; the fix
    # is what keeps the genuine label from being shadowed by the later suffix occurrence.
    path = _make_xlsx(tmp_path / "Questionnaire.xlsx", [
        "Planned Perforations (MD and TVD):",
        "MD: 6656 mMD / TVD: 3368 mTVD",
    ])
    result = parse_questionnaire(str(path))

    assert result.tvd_ft == pytest.approx(3368 * units.M_TO_FT, rel=1e-6)
    assert result.tvd_ft == pytest.approx(11050.0, abs=1.0)


def test_tvd_before_md_combined_cell_uses_tvd_not_md(tmp_path):
    # Reverse ordering of test_combined_metric_md_tvd_cell_uses_tvd_not_md -- TVD first, MD
    # second. The only non-suffix "tvd" occurrence is the leading "TVD:" label (not preceded by
    # "m"); the parser must anchor on it rather than on whichever "tvd" substring is last.
    path = _make_xlsx(tmp_path / "Questionnaire.xlsx", [
        "Planned Perforations (MD and TVD):",
        "TVD: 3368 mTVD / MD: 6656 mMD",
    ])
    result = parse_questionnaire(str(path))

    assert result.tvd_ft == pytest.approx(3368 * units.M_TO_FT, rel=1e-6)
    assert result.tvd_ft == pytest.approx(11050.0, abs=1.0)


def test_tvd_labeled_cell_with_only_meter_suffix_occurrence_of_tvd(tmp_path):
    # 'TVD: 3368 mTVD' has two "tvd" occurrences (the label, and the "mTVD" suffix); the label
    # one (index 0, not preceded by "m") must be used to anchor the search for the number.
    path = _make_xlsx(tmp_path / "Questionnaire.xlsx", [
        "Planned Perforations (MD and TVD):",
        "TVD: 3368 mTVD",
    ])
    result = parse_questionnaire(str(path))

    assert result.tvd_ft == pytest.approx(3368 * units.M_TO_FT, rel=1e-6)


def test_legacy_combined_md_tvd_feet_cell_unaffected(tmp_path):
    # Legacy shape (feet, apostrophe unit, no attached meter suffix) must keep working exactly as
    # before the finding-1 fix.
    path = _make_xlsx(tmp_path / "Questionnaire.xlsx", [
        "Planned Perforations (MD and TVD):",
        "MD: 21833' / TVD: 10929.8'",
    ])
    result = parse_questionnaire(str(path))

    assert result.tvd_ft == pytest.approx(10929.8)


def test_tvd_labeled_bare_number_no_slash_md(tmp_path):
    path = _make_xlsx(tmp_path / "Questionnaire.xlsx", [
        "Planned Perforations (MD and TVD):",
        "TVD 10929.8",
    ])
    result = parse_questionnaire(str(path))

    assert result.tvd_ft == pytest.approx(10929.8)


def test_mtvd_spelled_label_with_no_preceding_number_treated_as_meters(tmp_path):
    # Regression guard: the "m" in "mTVD:" starts the token with no number before it at all, so
    # it must NOT be mistaken for a unit-suffix "m" attached to some earlier number (there is
    # none) -- it's the label itself, spelling out that the number which follows is in meters.
    path = _make_xlsx(tmp_path / "Questionnaire.xlsx", [
        "Planned Perforations (MD and TVD):",
        "mTVD: 3368",
    ])
    result = parse_questionnaire(str(path))

    assert result.tvd_ft == pytest.approx(3368 * units.M_TO_FT, rel=1e-6)
    assert result.tvd_ft == pytest.approx(11049.9, abs=1.0)
    assert any("converted" in w.lower() for w in result.warnings)


def test_mtvd_spelled_label_combined_with_mmd_uses_tvd_not_md(tmp_path):
    # Same as above, but with a companion "mMD:" value after it -- the "after tvd" branch must
    # find the label's own number (3368) rather than falling through to the whole-cell fallback
    # and silently reporting the MD (6656).
    path = _make_xlsx(tmp_path / "Questionnaire.xlsx", [
        "Planned Perforations (MD and TVD):",
        "mTVD: 3368 / mMD: 6656",
    ])
    result = parse_questionnaire(str(path))

    assert result.tvd_ft == pytest.approx(3368 * units.M_TO_FT, rel=1e-6)
    assert result.tvd_ft == pytest.approx(11049.9, abs=1.0)


def test_tvd_label_with_parenthesized_meter_unit_before_number(tmp_path):
    # A meter unit interposed between the "TVD" label and the number, rather than attached to the
    # number itself, must still be detected -- "(mKB)" sits in the gap after the label and before
    # "3368", not right after the number the way "_is_meters" checks.
    path = _make_xlsx(tmp_path / "Questionnaire.xlsx", [
        "Planned Perforations (MD and TVD):",
        "TVD (mKB): 3368",
    ])
    result = parse_questionnaire(str(path))

    assert result.tvd_ft == pytest.approx(3368 * units.M_TO_FT, rel=1e-6)
    assert result.tvd_ft == pytest.approx(11049.9, abs=1.0)


def test_tvd_label_with_parenthesized_bare_m_before_number(tmp_path):
    path = _make_xlsx(tmp_path / "Questionnaire.xlsx", [
        "Planned Perforations (MD and TVD):",
        "TVD (m): 3368",
    ])
    result = parse_questionnaire(str(path))

    assert result.tvd_ft == pytest.approx(3368 * units.M_TO_FT, rel=1e-6)
    assert result.tvd_ft == pytest.approx(11049.9, abs=1.0)


def test_bare_meters_candidate_rejected_by_range_filter_is_not_warned_about(tmp_path):
    # A bare meterage candidate the _TVD_MIN/_TVD_MAX filter rejects (50000 m converts to
    # ~164,000 ft, way out of range) must not get a "converted to N ft" warning of its own -- the
    # conversion itself still has to happen before the filter (that's the actual finding-2-
    # adjacent fix), just the warning is deferred until after. And of the two surviving
    # candidates (6656 m the MD, 3368 m the TVD), only the one actually used as the TVD gets a
    # conversion warning -- warning about the MD's conversion too would mislabel it as a TVD one.
    path = _make_xlsx(tmp_path / "Questionnaire.xlsx", [
        "Planned Perforations (MD and TVD):",
        "50000 m",
        "6656 m",
        "3368 m",
    ])
    result = parse_questionnaire(str(path))

    assert result.tvd_ft == pytest.approx(3368 * units.M_TO_FT, rel=1e-6)
    converted_warnings = [w for w in result.warnings if "converted to" in w]
    assert len(converted_warnings) == 1
    assert not any("50000" in w for w in converted_warnings)
    assert not any("6656" in w for w in converted_warnings)
    assert any("3368" in w for w in converted_warnings)


def test_tvd_meters_bare_footage_fallback_branch_converts_to_feet(tmp_path):
    # Two unlabeled rows under one question (the Abraxas-style layout) -- MD then TVD, both in
    # meters -- with no "tvd"/"mtvd" substring in either cell at all, so `_tvd_from_block`'s
    # per-cell "tvd" check finds nothing and this genuinely falls through to the bare-footage
    # fallback branch, exercising its own meter handling. (A fixture using "mTVD"/"mMD" suffixes,
    # as in test_combined_metric_md_tvd_cell_uses_tvd_not_md, would be caught by the labeled loop
    # instead -- those substrings contain "tvd" and never reach this branch.)
    path = _make_xlsx(tmp_path / "Questionnaire.xlsx", [
        "Planned Perforations (MD and TVD):",
        "6656 m",
        "3368 m",
    ])
    result = parse_questionnaire(str(path))

    assert result.tvd_ft == pytest.approx(3368 * units.M_TO_FT, rel=1e-6)


# --------------------------------------------------------------------------------------------------
# kg/m3 density
# --------------------------------------------------------------------------------------------------
def test_density_kgm3_converted_to_ppg(tmp_path):
    path = _make_xlsx(tmp_path / "Questionnaire.xlsx", [
        "Type and density of fluid in the wellbore?",
        "1000 kg/m3",
    ])
    result = parse_questionnaire(str(path))
    assert result.density_ppg == pytest.approx(1000 * units.KGM3_TO_PPG, rel=1e-6)


def test_density_kgm3_superscript_variant(tmp_path):
    path = _make_xlsx(tmp_path / "Questionnaire.xlsx", [
        "Type and density of fluid in the wellbore?",
        "1000 kg/m³",
    ])
    result = parse_questionnaire(str(path))
    assert result.density_ppg == pytest.approx(1000 * units.KGM3_TO_PPG, rel=1e-6)


def test_density_kgm3_out_of_range_ignored(tmp_path):
    path = _make_xlsx(tmp_path / "Questionnaire.xlsx", [
        "Type and density of fluid in the wellbore?",
        "50 kg/m3",
    ])
    result = parse_questionnaire(str(path))
    assert result.density_ppg is None
    assert any("outside the expected" in w for w in result.warnings)


# --------------------------------------------------------------------------------------------------
# fresh-water whole-cell density fallback
# --------------------------------------------------------------------------------------------------
def test_fresh_water_whole_cell_fallback(tmp_path):
    path = _make_xlsx(tmp_path / "Questionnaire.xlsx", [
        "Type and density of fluid in the wellbore?",
        "Fresh Water ",
    ])
    result = parse_questionnaire(str(path))
    assert result.density_ppg == pytest.approx(8.34)
    assert any("assumed" in w.lower() for w in result.warnings)


@pytest.mark.parametrize("cell", ["fresh water", "Freshwater", "WATER"])
def test_fresh_water_fallback_name_variants(tmp_path, cell):
    path = _make_xlsx(tmp_path / "Questionnaire.xlsx", [
        "What type of fluid was pumped:",
        cell,
    ])
    result = parse_questionnaire(str(path))
    assert result.density_ppg == pytest.approx(8.34)


def test_fresh_water_with_real_number_uses_the_number_not_the_fallback(tmp_path):
    # Regression guard: a cell that both names the fluid and carries a real density number must
    # use the number, not the 8.34 ppg fresh-water fallback.
    path = _make_xlsx(tmp_path / "Questionnaire.xlsx", [
        "Type and density of fluid in the wellbore?",
        "Fresh Water - 8.6 lbs/gal",
    ])
    result = parse_questionnaire(str(path))
    assert result.density_ppg == pytest.approx(8.6)


# --------------------------------------------------------------------------------------------------
# multi-sheet workbooks (real Strathcona shape: 3 sheets, one well each, sheet title = well name)
# --------------------------------------------------------------------------------------------------
def test_multi_sheet_well_hint_matches_sheet_title(tmp_path):
    path = _make_multi_sheet_xlsx(tmp_path / "Questionnaire.xlsx", [
        ("100-01-01", ["Well Name:", "Well A",
                      "Type and density of fluid in the wellbore?", "8.4 lbs/gal"]),
        ("100-01-28-061-03W6", ["Well Name:", "Well B",
                                "Type and density of fluid in the wellbore?", "9.0 lbs/gal"]),
        ("100-01-99", ["Well Name:", "Well C",
                      "Type and density of fluid in the wellbore?", "9.5 lbs/gal"]),
    ])

    result = parse_questionnaire(str(path), well_hint="100-01-28")

    assert result.well_name == "Well B"
    assert result.density_ppg == pytest.approx(9.0)


def test_multi_sheet_well_hint_matches_well_name_cell_not_title(tmp_path):
    # The hint matches no sheet TITLE, only the second sheet's own "Well Name" answer cell.
    path = _make_multi_sheet_xlsx(tmp_path / "Questionnaire.xlsx", [
        ("Sheet1", ["Well Name:", "Alpha Well"]),
        ("Sheet2", ["Well Name:", "Bravo Well 1H"]),
    ])

    result = parse_questionnaire(str(path), well_hint="Bravo Well 1H")

    assert result.well_name == "Bravo Well 1H"


def test_multi_sheet_no_match_reads_whole_workbook_with_warning(tmp_path):
    path = _make_multi_sheet_xlsx(tmp_path / "Questionnaire.xlsx", [
        ("Alpha", ["Well Name:", "Well A"]),
        ("Beta", ["Well Name:", "Well B"]),
    ])

    result = parse_questionnaire(str(path), well_hint="Zzz-not-a-match")

    assert result.well_name == "Well A"
    assert any("none matched well hint" in w.lower() for w in result.warnings)


def test_multi_sheet_no_hint_reads_whole_workbook_with_warning(tmp_path):
    # scripts/ callers that pass no hint at all -- same fallback, a differently-worded warning.
    path = _make_multi_sheet_xlsx(tmp_path / "Questionnaire.xlsx", [
        ("Alpha", ["Well Name:", "Well A"]),
        ("Beta", ["Well Name:", "Well B"]),
    ])

    result = parse_questionnaire(str(path))

    assert result.well_name == "Well A"
    assert any("no well hint given" in w.lower() for w in result.warnings)


def test_multi_sheet_no_hint_finds_data_on_later_sheet(tmp_path):
    # Regression guard (finding 1): with no hint, a cover/instructions sheet first must not blank
    # out data that actually lives on a later sheet -- the no-hint fallback has to flatten the
    # whole workbook, not just sheets[0] (which would return no well name/density/TVD at all).
    path = _make_multi_sheet_xlsx(tmp_path / "Questionnaire.xlsx", [
        ("Instructions", ["Please fill this out carefully."]),
        ("100-01-28-061-03W6", ["Well Name:", "Well B",
                                "Type and density of fluid in the wellbore?", "9.0 lbs/gal"]),
    ])

    result = parse_questionnaire(str(path))

    assert result.well_name == "Well B"
    assert result.density_ppg == pytest.approx(9.0)


def test_multi_sheet_positive_hint_still_narrows_to_matched_sheet(tmp_path):
    # A positive hint match still narrows to just that one sheet (not a whole-workbook flatten) --
    # the first sheet's conflicting answer must not leak through.
    path = _make_multi_sheet_xlsx(tmp_path / "Questionnaire.xlsx", [
        ("100-01-01", ["Well Name:", "Well A",
                      "Type and density of fluid in the wellbore?", "8.4 lbs/gal"]),
        ("100-01-28-061-03W6", ["Well Name:", "Well B",
                                "Type and density of fluid in the wellbore?", "9.0 lbs/gal"]),
    ])

    result = parse_questionnaire(str(path), well_hint="100-01-28")

    assert result.well_name == "Well B"
    assert result.density_ppg == pytest.approx(9.0)


def test_multi_sheet_hint_disambiguates_similar_well_numbers(tmp_path):
    # Finding 4 regression: unbounded substring matching picked "1H" as a substring match inside
    # "21H" before "21H" itself was even tried -- delimiter-bounded containment must skip that
    # false match and land on the real "21H" sheet.
    path = _make_multi_sheet_xlsx(tmp_path / "Questionnaire.xlsx", [
        ("1H", ["Well Name:", "Well A"]),
        ("21H", ["Well Name:", "Well B"]),
        ("31H", ["Well Name:", "Well C"]),
    ])

    result = parse_questionnaire(str(path), well_hint="Smith 21H DFIT")

    assert result.well_name == "Well B"


def test_multi_sheet_hint_prefers_longest_match_and_warns(tmp_path):
    # Two sheet titles both match (bounded) the hint; the longer/more specific one should win,
    # with a warning naming the chosen sheet.
    path = _make_multi_sheet_xlsx(tmp_path / "Questionnaire.xlsx", [
        ("DFIT", ["Well Name:", "Cover Sheet"]),
        ("Smith 21H DFIT", ["Well Name:", "Well B"]),
    ])

    result = parse_questionnaire(str(path), well_hint="Smith 21H DFIT")

    assert result.well_name == "Well B"
    assert any("sheets matched well hint" in w.lower() for w in result.warnings)
    assert any("Smith 21H DFIT" in w for w in result.warnings)


def test_multi_sheet_no_match_finds_data_on_non_first_sheet(tmp_path):
    # Finding 5 regression: the earlier no-match test used two sheets that BOTH carried data, so
    # it would still pass even if the fallback silently reverted to sheets[0]. Here sheet 1 (the
    # cover sheet) carries no usable data at all, and only sheet 2 does -- a non-matching hint
    # must still find sheet 2's data via the whole-workbook fallback, not just sheet 1's (empty)
    # answer.
    path = _make_multi_sheet_xlsx(tmp_path / "Questionnaire.xlsx", [
        ("Cover", ["Please fill this out carefully."]),
        ("100-01-28-061-03W6", ["Well Name:", "Well B",
                                "Type and density of fluid in the wellbore?", "9.0 lbs/gal",
                                "Planned Perforations (MD and TVD):", "TVD: 10929.8'"]),
    ])

    result = parse_questionnaire(str(path), well_hint="Zzz-not-a-match")

    assert result.well_name == "Well B"
    assert result.density_ppg == pytest.approx(9.0)
    assert result.tvd_ft == pytest.approx(10929.8)
    assert any("none matched well hint" in w.lower() for w in result.warnings)


def test_single_sheet_ignores_well_hint(tmp_path):
    # A single-sheet workbook is unambiguous -- well_hint isn't even consulted, so a hint that
    # would match nothing must not produce a warning.
    path = _make_xlsx(tmp_path / "Questionnaire.xlsx", [
        "Well Name:",
        "Solo Well",
    ])
    result = parse_questionnaire(str(path), well_hint="Zzz-not-a-match")
    assert result.well_name == "Solo Well"
    assert not any("sheet" in w.lower() for w in result.warnings)


# --------------------------------------------------------------------------------------------------
# unitless density, leading-decimal gradient, "FW" token (real DJ Basin cells)
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("cell, expected", [
    ("8.4", 8.4),
    (8.33, 8.33),                 # numeric cell, as openpyxl returns it
    ("Fresh water, 8.33", 8.33),
    ("Fresh Water.  8.34", 8.34),
    ("water 8.4", 8.4),
    ("Fresh 8.33/lb", 8.33),
])
def test_unitless_density_in_fluid_answer_read_as_ppg(tmp_path, cell, expected):
    path = _make_xlsx(tmp_path / "Questionnaire.xlsx", [
        "Type and density of fluid in the wellbore?",
        cell,
    ])
    result = parse_questionnaire(str(path))
    assert result.density_ppg == pytest.approx(expected)
    assert any("no unit" in w for w in result.warnings)


@pytest.mark.parametrize("cell", ["6% KCl brine", "10 bbls", "Slickwater 15", "12"])
def test_unitless_rule_skips_percent_volume_and_integers(tmp_path, cell):
    path = _make_xlsx(tmp_path / "Questionnaire.xlsx", [
        "What type of fluid was pumped:",
        cell,
    ])
    assert parse_questionnaire(str(path)).density_ppg is None


def test_unitless_rule_only_reads_fluid_answers(tmp_path):
    path = _make_xlsx(tmp_path / "Questionnaire.xlsx", [
        "Porosity",
        "8.5",
        "Type and density of fluid in the wellbore?",
        "Saturated Oil",
    ])
    assert parse_questionnaire(str(path)).density_ppg is None


def test_unit_density_beats_earlier_unitless_number(tmp_path):
    path = _make_xlsx(tmp_path / "Questionnaire.xlsx", [
        "Type and density of fluid in the wellbore?",
        "8.4",
        "What type of fluid was pumped:",
        "Brine 9.2 ppg",
    ])
    assert parse_questionnaire(str(path)).density_ppg == pytest.approx(9.2)


def test_leading_decimal_gradient_converted_to_ppg(tmp_path):
    path = _make_xlsx(tmp_path / "Questionnaire.xlsx", [
        "Type and density of fluid in the wellbore?",
        ".44 psi/ft",
    ])
    result = parse_questionnaire(str(path))
    assert result.density_ppg == pytest.approx(0.44 / 0.052)


@pytest.mark.parametrize("cell", ["Clay treated FW", "FW", "treated fresh water"])
def test_fresh_water_token_fallback(tmp_path, cell):
    path = _make_xlsx(tmp_path / "Questionnaire.xlsx", [
        "Type and density of fluid in the wellbore?",
        cell,
    ])
    result = parse_questionnaire(str(path))
    assert result.density_ppg == pytest.approx(8.34)
    assert any("assumed" in w.lower() for w in result.warnings)


@pytest.mark.parametrize("cell", ["Produced water", "FWKO water", "LibertyFR"])
def test_fresh_water_token_fallback_needs_fresh_water_token(tmp_path, cell):
    path = _make_xlsx(tmp_path / "Questionnaire.xlsx", [
        "Type and density of fluid in the wellbore?",
        cell,
    ])
    assert parse_questionnaire(str(path)).density_ppg is None
