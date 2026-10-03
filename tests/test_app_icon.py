"""The app icon assets exist and app.set_app_icon applies them to the Tk root."""

from dfit_tool import app


def test_icon_assets_exist():
    assert app.ICON_PNG.is_file()
    assert app.ICON_ICO.is_file()


class _FakeRoot:
    def __init__(self):
        self.calls = []

    def iconbitmap(self, **kw):
        self.calls.append(("iconbitmap", kw))

    def iconphoto(self, default, img):
        self.calls.append(("iconphoto", default))


def test_set_app_icon_windows_uses_ico_only(monkeypatch):
    # iconphoto after iconbitmap would replace the multi-size .ico with one 256 px image
    # that Windows downscales itself, which renders blurry on the taskbar.
    monkeypatch.setattr(app.sys, "platform", "win32")
    monkeypatch.setattr(app.tk, "PhotoImage", lambda **kw: object())
    root = _FakeRoot()
    app.set_app_icon(root)
    assert root.calls == [("iconbitmap", {"default": str(app.ICON_ICO)})]


def test_set_app_icon_elsewhere_uses_png(monkeypatch):
    monkeypatch.setattr(app.sys, "platform", "linux")
    monkeypatch.setattr(app.tk, "PhotoImage", lambda **kw: object())
    root = _FakeRoot()
    app.set_app_icon(root)
    assert root.calls == [("iconphoto", True)]


def test_ico_has_distinct_small_frames():
    from PIL import Image
    ico = Image.open(app.ICON_ICO)
    assert {(16, 16), (24, 24), (32, 32), (256, 256)} <= set(ico.info["sizes"])


def test_set_app_icon_never_raises():
    class Broken:
        def iconbitmap(self, **kw):
            raise RuntimeError("no display")

        def iconphoto(self, *a):
            raise RuntimeError("no display")

    app.set_app_icon(Broken())


def test_taskbar_icon_px_matches_display_scaling():
    # A DPI-unaware Tk window gets a 32 px big icon that the taskbar stretches; pick the
    # frame for the real scaling instead (125% -> 40 px, 150% -> 48 px).
    assert app.taskbar_icon_px(96) == 32
    assert app.taskbar_icon_px(120) == 40
    assert app.taskbar_icon_px(144) == 48
    assert app.taskbar_icon_px(192) == 64


def test_ico_has_a_frame_for_common_scalings():
    from PIL import Image
    sizes = set(Image.open(app.ICON_ICO).info["sizes"])
    for dpi in (96, 120, 144, 168, 192):
        px = app.taskbar_icon_px(dpi)
        assert (px, px) in sizes, px


def test_set_taskbar_icon_is_noop_off_windows(monkeypatch):
    monkeypatch.setattr(app.sys, "platform", "linux")
    app.set_taskbar_icon(object())  # must not touch the root or raise


def test_taskbar_icon_applied_on_root_map_only(monkeypatch):
    # Tk replaces the wrapper HWND when the root is first mapped, so the icon must be sent
    # from <Map>, not from after_idle. Child widgets' Map events bubble to the root binding.
    import types
    seen = []
    monkeypatch.setattr(app, "set_taskbar_icon", lambda r: seen.append(r))
    root = object()
    app._on_root_map(root, types.SimpleNamespace(widget=object()))
    assert seen == []
    app._on_root_map(root, types.SimpleNamespace(widget=root))
    assert seen == [root]


def test_icon_corners_are_transparent():
    # The tile is a rounded square; the area outside it must not be opaque white.
    from PIL import Image
    ico = Image.open(app.ICON_ICO)
    for size in ico.info["sizes"]:
        ico.size = size
        frame = ico.copy().convert("RGBA")
        assert frame.getpixel((0, 0))[3] < 16, size  # 16 px: tile edge antialiases to ~4
    assert Image.open(app.ICON_PNG).convert("RGBA").getpixel((0, 0))[3] == 0
