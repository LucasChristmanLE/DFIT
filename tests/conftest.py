"""Pytest configuration shared by all tests.

Forces the non-interactive Agg backend before matplotlib.pyplot is ever imported, so the
suite runs headless (no Tk, no display) regardless of what backend is installed/active.
"""

import matplotlib

matplotlib.use("Agg")
