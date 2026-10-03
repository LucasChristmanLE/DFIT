"""Entry point: python -m dfit_tool.app [path/to/file.csv | path/to/folder]"""

from __future__ import annotations

import pathlib
import sys
import tkinter as tk

from .ui import DfitApp

_ASSETS = pathlib.Path(__file__).parent / "assets"
ICON_PNG = _ASSETS / "app_icon.png"
ICON_ICO = _ASSETS / "app_icon.ico"


def _set_windows_app_id() -> None:
    """Give the process its own taskbar identity so Windows shows our icon, not python.exe's."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("Liberty.FracClosure")
    except Exception:
        pass


def set_app_icon(root) -> None:
    """Apply the app icon to `root` and every later Toplevel. A missing icon is never fatal.

    Windows gets only the multi-size .ico: iconphoto would replace it with one large image
    that Windows downscales itself, which is blurry on the taskbar.
    """
    if sys.platform == "win32":
        try:
            root.iconbitmap(default=str(ICON_ICO))
        except Exception:
            pass
        return
    try:
        root._app_icon = tk.PhotoImage(file=str(ICON_PNG))  # Tk keeps no strong ref
        root.iconphoto(True, root._app_icon)
    except Exception:
        pass


def taskbar_icon_px(dpi: int) -> int:
    """The taskbar's big-icon size at `dpi` (32 px at 100% scaling)."""
    return round(32 * dpi / 96)


def _real_system_dpi() -> int:
    """The display DPI as a DPI-aware thread sees it; this process itself is DPI-unaware,
    so a plain query would always report 96."""
    import ctypes
    user32 = ctypes.windll.user32
    user32.SetThreadDpiAwarenessContext.restype = ctypes.c_void_p
    user32.SetThreadDpiAwarenessContext.argtypes = [ctypes.c_void_p]
    old = user32.SetThreadDpiAwarenessContext(ctypes.c_void_p(-4))  # per-monitor aware v2
    try:
        return int(user32.GetDpiForSystem()) or 96
    finally:
        if old:
            user32.SetThreadDpiAwarenessContext(ctypes.c_void_p(old))


def set_taskbar_icon(root) -> None:
    """Send the taskbar an icon frame at the real display size.

    Tk is DPI-unaware, so it hands Windows a 32 px big icon that the taskbar stretches
    (blurry at 125%+). Load the matching .ico frame and set it as the window's big icon.
    Must run once the root is mapped: Tk replaces the wrapper HWND on first map, so an
    earlier call sets the icon on a window that is then discarded. Never fatal.
    """
    if sys.platform != "win32":
        return
    try:
        import ctypes
        user32 = ctypes.windll.user32
        user32.LoadImageW.restype = ctypes.c_void_p
        user32.LoadImageW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_uint,
                                      ctypes.c_int, ctypes.c_int, ctypes.c_uint]
        user32.SendMessageW.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p,
                                        ctypes.c_void_p]
        IMAGE_ICON, LR_LOADFROMFILE, WM_SETICON, ICON_BIG = 1, 0x10, 0x80, 1
        hicon = getattr(root, "_taskbar_hicon", None)
        if hicon is None:
            px = taskbar_icon_px(_real_system_dpi())
            hicon = user32.LoadImageW(None, str(ICON_ICO), IMAGE_ICON, px, px, LR_LOADFROMFILE)
            if not hicon:
                return
            root._taskbar_hicon = hicon  # loaded once, kept alive with the window
        hwnd = int(root.wm_frame(), 16)
        user32.SendMessageW(ctypes.c_void_p(hwnd), WM_SETICON, ctypes.c_void_p(ICON_BIG),
                            ctypes.c_void_p(hicon))
    except Exception:
        pass


def _on_root_map(root, event) -> None:
    """<Map> handler: the root's binding also sees every child's Map, so filter to the root."""
    if event.widget is root:
        set_taskbar_icon(root)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    path = argv[0] if argv else None
    _set_windows_app_id()
    root = tk.Tk()
    set_app_icon(root)
    DfitApp(root, path=path)
    root.bind("<Map>", lambda e: _on_root_map(root, e), add="+")
    if root.winfo_ismapped():  # DfitApp may already have shown it (folder-open progress)
        set_taskbar_icon(root)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
