"""One shared atomic-replace helper, retried on Windows' transient PermissionError.

Every writer in this package (the ledger, the features cache, the provenance records) wants the
same contract `dfit_tool/store.py:save_picks_for` uses -- write a temp file in the destination
directory, then `os.replace` onto the final path, so a crash mid-write never leaves a half-written
file behind. On Windows that last step is the fragile one: `os.replace` raises `PermissionError`
(WinError 5) whenever any process holds a transient handle on either path, which in practice means
Defender real-time scanning the file that was just written. Measured at roughly 1 failure in 6 runs
of this repo's test suite.

The retry lives here rather than in each caller because the cost of an unretried failure differs
wildly by caller and all of them are unacceptable: the ledger saves after every one of ~402 folder
decisions in a review session, and the features cache is the single output of a scan that takes
tens of minutes.
"""

from __future__ import annotations

import os
import time

REPLACE_ATTEMPTS = 5
REPLACE_BACKOFF_S = 0.05


def replace_with_retry(src: str, dst: str) -> None:
    """`os.replace(src, dst)`, retried on `PermissionError` with a short linear backoff.

    Only `PermissionError` is retried -- every other `OSError` is a real problem (a bad path, a
    cross-device move) that retrying cannot fix, so it propagates on the first attempt. After
    `REPLACE_ATTEMPTS` failures the last `PermissionError` is re-raised, so a genuine permissions
    problem still surfaces instead of being swallowed.
    """
    for attempt in range(REPLACE_ATTEMPTS):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(REPLACE_BACKOFF_S * (attempt + 1))
