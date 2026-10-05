"""Plot and UI colors: the Liberty brand palette plus two blues and a gold accent.

Leaf module (no ``dfit_tool`` imports). Renderers and the Tk shell use the role names, not the
brand names, so a role can move to another color in one place. Gold (1.7:1 on white) is used for
fills, the tail trim, and the ISIP constructions; Light Grey (1.2:1) is not used for marks.
"""

# Liberty brand.
LBRT_RED = "#EE2827"
LBRT_BLACK = "#262626"
LBRT_DEEP_RED = "#A91E22"
LIGHT_GREY = "#E7E7E7"
MID_GREY = "#7C7B7A"

# Complementary accents.
RATE_BLUE = "#2E86AB"
PP_BLUE = "#5C80BC"
GOLD = "#E8C547"

# Plot roles.
PRESSURE = LBRT_BLACK               # BHP trace
SURFACE_PRESSURE = LBRT_DEEP_RED    # unconverted surface pressure, overview surface overlay
DERIVATIVE = LBRT_RED               # dP/dG, G*dP/dG, t*dP/dt and their picks
SECOND_DERIVATIVE = MID_GREY        # d2P/dG2
SHUTIN = LBRT_RED
INJECTION_START = MID_GREY
RATE = RATE_BLUE
ISIP_LINE = GOLD                    # apparent ISIP tangent, effective ISIP line
WINDOW = GOLD                       # shaded windows and drag-select spans
WINDOW_ALPHA = 0.25                 # gold is lighter than the old orange, so a bit denser
TAIL_TRIM = GOLD
PICK = LBRT_BLACK                   # contact/closure markers and lines
GUIDE = MID_GREY                    # through-origin line
PORE_PRESSURE = PP_BLUE
STIFFNESS_PICK = LBRT_RED
EXCLUDED = MID_GREY                 # trimmed / guard-excluded samples, drawn at EXCLUDED_ALPHA
EXCLUDED_ALPHA = 0.35
GUARD_EXCLUDED_ALPHA = 0.2          # the raw tail past the guard boundary, fainter still
DROPOUT = "magenta"                 # alarm marker, must not match any series
NET_PRESSURE = RATE_BLUE
COMPLEXITY = GOLD
BAR_LABEL = "white"                 # value text drawn on a net-pressure bar

# UI roles.
UI_ERROR = LBRT_RED                 # blockers, gate label, warning label
UI_WARNING = "#b35c00"              # amber; gold is unreadable as text
UI_MUTED = MID_GREY                 # notes, hints, skipped rows, neutral slider
UI_TEXT = LBRT_BLACK
UI_DONE = "#137333"
UI_CURRENT = LBRT_RED               # folder-mode sidebar row of the open test
UI_CURRENT_TEXT = "white"

# Manual mask/keep bands on Overview.
MANUAL_MASK = DROPOUT               # analyst-masked glitch, same hue as auto-masked samples
MANUAL_KEEP = UI_DONE               # analyst-restored data, distinct from gold and grey
MANUAL_SPAN_ALPHA = 0.15
