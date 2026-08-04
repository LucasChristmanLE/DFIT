"""Tests for dfit_tool/questionnaire.py: label-anchored Q&A parsing of the DFIT Questionnaire
template. Fixtures are built in-test with openpyxl, mimicking the two real-world layouts on hand
(Abraxas: bare numbers, no unit labels; PDC: MD:/TVD: labeled rows, SG-labeled density) without
referencing the actual sample files.
"""

import openpyxl
import pytest

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


def test_malformed_perfs_cell_never_raises(tmp_path):
    # "," matches the bare-footage regex's [\d,]+ class but has no digits, so float() on it would
    # raise -- this must be caught by parse_questionnaire's per-field try/except, not propagate.
    path = _make_xlsx(tmp_path / "Questionnaire.xlsx", [
        "Actual perforation depth used for the DFIT:",
        ",",
    ])
    result = parse_questionnaire(str(path))
    assert result.tvd_ft is None
    assert any("TVD parsing failed" in w for w in result.warnings)


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
