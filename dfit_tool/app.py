"""Entry point: python -m dfit_tool.app [path/to/file.csv | path/to/folder]"""

from __future__ import annotations

import sys
import tkinter as tk

from .ui import DfitApp


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    path = argv[0] if argv else None
    root = tk.Tk()
    DfitApp(root, path=path)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
