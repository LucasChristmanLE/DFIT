"""PyInstaller entry point for FracClosure. dfit_tool.app uses relative imports, so it cannot be
the frozen script itself."""

from dfit_tool.app import main

raise SystemExit(main())
