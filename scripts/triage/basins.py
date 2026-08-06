"""Basin lookup for triage's `apply` step: formation first, then a customer fallback, else
`DEFAULT_BASIN`.

`FORMATION_TO_BASIN` is seeded from formation spellings actually observed across the 402
questionnaire-bearing folders in the real `C:\\DFIT Data` corpus (measured this session).
`CUSTOMER_TO_BASIN` is seeded from the same measurement, grouped by which of the 70 top-level
customer directories had questionnaires pointing consistently at one basin. Both tables are
best-effort and the user is expected to correct them once real values are in hand -- see the
`DEFAULT_BASIN` notes on each table below for where that's already been done deliberately.
"""

from __future__ import annotations

import re

DEFAULT_BASIN = "Unassigned"

# --------------------------------------------------------------------------------------------------
# formation normalization
# --------------------------------------------------------------------------------------------------
_NON_ALNUM_RE = re.compile(r"[^A-Z0-9]+")


def normalize_formation(s: str | None) -> str:
    """Uppercase, strip, and collapse every run of non-alphanumeric characters (spaces, dashes,
    parentheses, punctuation) to a single space. `"Niobrara - C-Chalk"`, `"Niobrara C-Chalk"`,
    and `"NIOBRARA   C   CHALK"` all normalize to the same `"NIOBRARA C CHALK"` key -- the
    leading `"NIOBRARA -"` style separator some questionnaires use disappears into that same
    collapse, no separate rule needed."""
    if not s:
        return ""
    return _NON_ALNUM_RE.sub(" ", s.upper()).strip()


# --------------------------------------------------------------------------------------------------
# formation -> basin
# --------------------------------------------------------------------------------------------------
# Raw spellings observed in the real corpus, grouped by the basin assigned to them. Built into
# FORMATION_TO_BASIN below by running each raw spelling through normalize_formation(), so keys in
# the resulting table are exactly what a parsed questionnaire's formation field would normalize
# to. Several raw spellings collapse to the same normalized key (e.g. "C Chalk" / "C-Chalk") --
# harmless, they map to the same basin anyway.
_FORMATION_BASIN_RAW: dict[str, list[str]] = {
    "DJ": [
        "Codell", "Niobrara", "Niobrara A", "Niobrara B", "Niobrara C",
        "Niobrara B Chalk", "Niobrara - B-Chalk", "Niobrara C Bench",
        "Nio B", "Nio C", "Nio - B", "C Chalk", "C-Chalk", "Niobrara - C-Chalk",
        "J Sand", "Malory",
    ],
    "Powder River": [
        "Turner", "Frontier", "Parkman", "Shannon", "Parker", "Sussex", "Belle Fourche",
    ],
    "Williston": [
        "Bakken", "Middle Bakken", "Three Forks", "Three Forks 1st bench",
        "Lower Lodgepole", "Mission Canyon?",  # the "?" is in the observed data, not a typo here
    ],
    "Permian": [
        "Wolfcamp A", "Wolfcamp B", "Wolfcamp A1", "Wolfcamp A2", "Wolfcamp D",
        "Lower Wolfcamp A", "Lower Wolfcamp B", "3rd Bone Spring Carb",
    ],
    "Montney": ["Montney", "Montney (D1)", "Montney (D2)"],
    "Piceance": ["Mancos"],
    # Genuinely ambiguous or parse-noise formation strings observed in the real corpus. Mapped
    # here explicitly (rather than simply absent from the table) so basin_for reports source
    # "formation" for them -- i.e. it does NOT fall through to a customer guess, since a customer
    # guess would be no better informed than these strings already are. "Stage Info:" is a
    # questionnaire parse artifact (the formation label-block parse picked up a neighboring
    # cell, see questionnaire.py's _build_blocks) rather than a real formation name; not worth
    # fixing questionnaire.py over, so it's just mapped to DEFAULT_BASIN here.
    DEFAULT_BASIN: ["Shale 3 (Decorah)", "Canyon Creek", "Stage Info:"],
}

FORMATION_TO_BASIN: dict[str, str] = {}
for _basin, _names in _FORMATION_BASIN_RAW.items():
    for _name in _names:
        FORMATION_TO_BASIN[normalize_formation(_name)] = _basin


# --------------------------------------------------------------------------------------------------
# customer -> basin
# --------------------------------------------------------------------------------------------------
# Seeded from observed questionnaire formations across all 402 questionnaire-bearing folders
# (measured this session), one entry required for each of the 70 top-level customer directories.
# DEFAULT_BASIN here is deliberate in two different senses -- read `basin_for`'s docstring for how
# that interacts with the formation table above:
#   - the operator's own wells span more than one basin (e.g. Abraxas: Permian Caprito wells AND
#     Williston Stenehjem wells; Oxy: Powder River Turner wells AND DJ Niobrara wells) -- a
#     customer fallback here would silently guess wrong for roughly half of that operator's wells,
#     so the formation table is left to decide and this table intentionally offers no guess;
#   - or there's no formation evidence for that customer in the corpus at all (an international
#     operator -- Aspect Energy is Croatia/Hungary, Tamboran is Australia -- or a Canadian one
#     with no US-basin formation on file -- ARC Resources, B-32 Exploration, Vesta Energy,
#     Kiwetinohk).
# The user is expected to correct any of these once real values are in hand.
CUSTOMER_TO_BASIN: dict[str, str] = {
    "ARC Resources": DEFAULT_BASIN,
    "Abraxas Petroleum Corp": DEFAULT_BASIN,          # spans Permian + Williston, see above
    "Aspect Energy": DEFAULT_BASIN,                   # international (Croatia/Hungary)
    "B-32 Exploration": DEFAULT_BASIN,                # Canadian, no formation evidence on file
    "Ballard Petroleum": "Powder River",
    "Birch Resources": DEFAULT_BASIN,
    "Black Hills": DEFAULT_BASIN,                     # spans Powder River (Shannon) + Piceance (Mancos)
    "Bonanza Creek": "DJ",
    "Boomtown Oil": "DJ",
    "Bright Rock": "DJ",
    "Catamount Energy Partners": "Powder River",
    "Centennial Resources": "Permian",
    "Civitas": "DJ",
    "Clear Creek": "DJ",
    "Conoco Phillips": DEFAULT_BASIN,
    "Continental Resources": "Williston",
    "Crescent Point": "Williston",
    "Crestone Peak": "DJ",
    "Cygnet": DEFAULT_BASIN,
    "Devon Energy": "DJ",
    "EOG": "Williston",
    "Edge Energy": "DJ",
    "Elk Mesa": "DJ",
    "Emerald Oil": "Williston",
    "Extraction Oil & Gas": "DJ",
    "Fifth Creek Energy": "DJ",
    "Fulcrum Energy": "DJ",
    "GMT Exploration": DEFAULT_BASIN,
    "Great Western": "DJ",
    "Halcon Resources": DEFAULT_BASIN,
    "Hat Creek Resources": DEFAULT_BASIN,
    "Helis Oil & Gas": "DJ",
    "Hess": "Williston",
    "Highlands Natural Resources": DEFAULT_BASIN,
    "Impact E&P": DEFAULT_BASIN,
    "Kinney Oil": DEFAULT_BASIN,
    "Kiwetinohk": DEFAULT_BASIN,                      # Canadian, no formation evidence on file
    "Koch Exploration": "Permian",
    "Koda Resources": "Williston",
    "Laramie Energy": DEFAULT_BASIN,
    "Laredo": DEFAULT_BASIN,
    "Liberty Resources": "Williston",
    "Lonestar Operating": DEFAULT_BASIN,
    "Lucero Energy": "Williston",
    "Mallard Exploration": "DJ",
    "Noble Energy": "DJ",
    "North Peak": "DJ",
    "North Plains Energy": DEFAULT_BASIN,
    "North Silo": DEFAULT_BASIN,
    "Oxy (formerly APC)": DEFAULT_BASIN,              # spans Powder River (Turner) + DJ (Niobrara)
    "PDC Energy": DEFAULT_BASIN,                      # DJ mostly, but Wolfcamp (Permian) also on file
    "Phoenix Energy": "Williston",
    "Red Willow": DEFAULT_BASIN,
    "Rockies Resources": DEFAULT_BASIN,
    "Roost Resources": DEFAULT_BASIN,
    "SM Energy": DEFAULT_BASIN,                       # spans Powder River (Frontier) + Permian (Wolfcamp D)
    "Sandpoint Resources": DEFAULT_BASIN,
    "Strathcona Resources": "Montney",
    "Synergy Resources": "DJ",
    "Tamboran": DEFAULT_BASIN,                        # international (Australia)
    "Tap Rock Resources": "Permian",
    "True Oil": "Williston",
    "Vermilion": DEFAULT_BASIN,
    "Vesta Energy": DEFAULT_BASIN,                    # Canadian, no formation evidence on file
    "WPX Energy": "Williston",
    "Ward Petroleum": "DJ",
    "Wave Petroleum": DEFAULT_BASIN,                  # Codell + Parkman on file, i.e. mixed
    "Whiting Oil & Gas Corp": DEFAULT_BASIN,
    "Williams": DEFAULT_BASIN,
    "Zavanna LLC": "Williston",
}


# --------------------------------------------------------------------------------------------------
# lookup
# --------------------------------------------------------------------------------------------------
def basin_for(formation: str | None, customer: str | None) -> tuple[str, str]:
    """`(basin, source)`: the formation table wins whenever the normalized formation is a known
    key -- including the keys deliberately mapped to `DEFAULT_BASIN` above (see that table's
    comment), which is why those still return `source == "formation"` rather than falling
    through to a customer guess: a customer guess would be no better informed.

    Only when the formation is blank or unrecognized does the customer table get a turn -- and
    only a customer mapped to a *real* basin counts as `source == "customer"`; a customer
    mapped to `DEFAULT_BASIN` (an operator that spans multiple basins, or one with no formation
    evidence on file -- see that table's comment) is reported the same as no customer match at
    all, `(DEFAULT_BASIN, "default")`.
    """
    key = normalize_formation(formation)
    if key and key in FORMATION_TO_BASIN:
        return FORMATION_TO_BASIN[key], "formation"
    if customer and customer in CUSTOMER_TO_BASIN:
        basin = CUSTOMER_TO_BASIN[customer]
        if basin != DEFAULT_BASIN:
            return basin, "customer"
    return DEFAULT_BASIN, "default"
