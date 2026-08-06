"""DFIT data triage: classify, review, and reorganize the ~2,400-file ``C:\\DFIT Data`` tree.

See ``docs/superpowers/plans`` for the approved plan. Layered like ``dfit_tool``: ``features.py``,
``basins.py``, ``ledger.py``, and ``figure.py`` are Tkinter-free and unit-testable headless;
``review_app.py`` (built separately) is the only module in this package allowed to import
tkinter.
"""

from __future__ import annotations
