"""Tests for dfit_tool/guide_content.py: pure-data content for the interpretation guide window.

Headless -- no Tkinter. Checks the two Guide instances are non-empty and that every figure they
reference resolves to a real, non-empty PNG under dfit_tool/assets/guide/ (catches a missing or
failed-download asset at test time rather than at runtime in front of a user).
"""

import pathlib

import dfit_tool
from dfit_tool import guide_content

ASSETS_DIR = pathlib.Path(dfit_tool.__file__).parent / "assets" / "guide"

GUIDES = [guide_content.CLOSURE_GUIDE, guide_content.POSTCLOSURE_GUIDE]


def test_guides_have_title_intro_source_and_sections():
    for guide in GUIDES:
        assert guide.title
        assert guide.intro
        assert guide.source
        assert len(guide.sections) >= 1


def test_sections_have_title_and_body():
    for guide in GUIDES:
        for section in guide.sections:
            assert section.title
            assert section.body


def test_figure_assets_exist_and_are_nonempty():
    for guide in GUIDES:
        for section in guide.sections:
            for fig in section.figures:
                path = ASSETS_DIR / fig.image
                assert path.is_file(), f"missing guide asset: {path}"
                assert path.stat().st_size > 0, f"empty guide asset: {path}"
                assert fig.caption
