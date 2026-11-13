"""Recover the info-page table grid from ruled lines -- no OCR, no language model.

Page 1 is a machine-generated form, so its rules land on the same pixels every time.
Measured across AC1, AC10, AC12 and AC100:

* horizontal rules at ``124, 157, 191, 225, 399, 433, 483, 816, 850, 896`` are present on
  every page, pixel-identical;
* the section-4 elector table is always the **last five** rules, with fixed relative
  spacing (``+34, +80, +116, +170``);
* its six columns sit at ``24, 156, 286, 497, 723, 947, 1162``, identical within 1px.

Two page variants exist. Some pages carry an extra rule around y=171, and the section-3
address block is 26px taller when the address wraps to a second line -- which shifts
everything below it. So cells are addressed by **detected structure** (stable rules from
the top, the elector table from the end), never by hardcoded absolute offsets.

Recovering the grid mechanically means label text never has to be read: the pipeline
knows where each value lives and OCRs only that cell. That is what makes a free,
local-only pipeline viable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from PIL import Image, ImageDraw

#: The **upper** rules, present on every Assamese page observed, used to verify a page
#: matches the template. Bengali shares these exactly; English does not (its rows are
#: taller, putting them at 133, 185, 243, 295, 488, 540, 596). Each language carries its
#: own set in its ``LanguageProfile``; this is the Assamese one and the default.
#:
#: The rule at y=171 is deliberately excluded: it appears on some Assamese pages and not
#: others, so requiring it would reject valid pages.
UPPER_H_RULES: Tuple[int, ...] = (124, 157, 191, 225, 399, 433, 483)

#: Kept as the name earlier versions used, for anything still importing it.
STABLE_H_RULES = UPPER_H_RULES

#: Indices into the matched upper rules. Named because the cells below read far better
#: as ``upper[S1_TOP]`` than as ``upper[3]``.
HEADER_TOP, HEADER_MID, HEADER_BOTTOM = 0, 1, 2
S1_TOP, S1_BOTTOM = 3, 4
S2_TOP = 6  # index 5 is a stability anchor only, not a cell boundary

#: The three rules between section 2 and section 4: the bottom of the area list, and the
#: pair bracketing the polling-station name. Located from the end of the rule list rather
#: than by pixel, since they shift with page content and with language.
LOWER_RULE_COUNT = 3

#: The elector table is always the last five horizontal rules.
S4_RULE_COUNT = 5

#: Tolerance when matching a detected rule to its expected position.
RULE_TOLERANCE = 4

#: Ink threshold: below this gray value a pixel counts as part of a rule.
INK = 128

#: A row/column is a rule when this share of the sampled span is ink.
H_RULE_FILL = 0.5
V_RULE_FILL = 0.6

#: Page frame, used to bound sampling.
FRAME_LEFT, FRAME_RIGHT = 24, 1162

#: Height of the footer strip carrying the total-pages line.
FOOTER_HEIGHT = 100

#: Native page size of these scans; used by tests and as a sanity hint.
EXPECTED_SIZE_HINT = (1187, 1679)

#: Width every pixel constant in this module is expressed against. A page issued at a
#: different size is read at its own resolution with the constants scaled, never rescaled
#: to match them -- see ``render.normalize_page`` for what resampling costs.
CANONICAL_WIDTH = 1187


class LayoutError(RuntimeError):
    """Raised when a page does not match the info-page template."""


@dataclass(frozen=True)
class Box:
    """A cell in page pixel coordinates."""

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def size(self) -> Tuple[int, int]:
        return self.x1 - self.x0, self.y1 - self.y0

    def pad(self, inset: int = 2) -> "Box":
        """Shrink by ``inset`` px so the surrounding rules are not fed to OCR."""
        return Box(self.x0 + inset, self.y0 + inset, self.x1 - inset, self.y1 - inset)


@dataclass(frozen=True)
class Grid:
    """A page's detected geometry and its named cells."""

    width: int
    height: int
    h_rules: Tuple[int, ...]
    cells: Dict[str, Box]

    #: Page width relative to the canonical 1187px. The publisher issues some pages at
    #: 0.80x, and they are read at their own resolution rather than rescaled -- resampling
    #: either invents pixels or discards them, and both corrupt digits. Callers with their
    #: own pixel offsets (the parser's value columns) must scale them by this.
    scale: float = 1.0

    def crop(self, image: Image.Image, name: str, inset: int = 2) -> Image.Image:
        box = self.cells[name].pad(inset)
        return image.crop((box.x0, box.y0, box.x1, box.y1))


# --------------------------------------------------------------------------- detection


def _group(indices: Sequence[int], gap: int = 2) -> List[int]:
    """Collapse runs of adjacent indices into their midpoints."""
    groups: List[List[int]] = []
    for index in indices:
        if groups and index - groups[-1][-1] <= gap:
            groups[-1].append(index)
        else:
            groups.append([index])
    return [sum(group) // len(group) for group in groups]


def detect_h_rules(image: Image.Image, scale: float = 1.0) -> List[int]:
    """Y positions of full-width horizontal rules."""
    grey = image.convert("L")
    width, height = grey.size
    pixels = grey.load()
    left = int(round(FRAME_LEFT * scale))
    right = min(int(round(FRAME_RIGHT * scale)), width)
    step = 3
    samples = len(range(left, right, step))
    hits = [
        y
        for y in range(height)
        if sum(1 for x in range(left, right, step) if pixels[x, y] < INK) > samples * H_RULE_FILL
    ]
    return _group(hits)


def detect_v_rules(image: Image.Image, y0: int, y1: int) -> List[int]:
    """X positions of vertical rules spanning the band ``y0..y1``."""
    grey = image.convert("L")
    width, _ = grey.size
    pixels = grey.load()
    span = max(1, y1 - y0)
    hits = [
        x
        for x in range(width)
        if sum(1 for y in range(y0, y1) if pixels[x, y] < INK) > span * V_RULE_FILL
    ]
    return _group(hits)


def _match(rules: Sequence[int], expected: int, scale: float = 1.0) -> Optional[int]:
    """The detected rule closest to ``expected``, within tolerance.

    The tolerance scales with the page: a 4px allowance on a 1187px page is 3px on a
    949px one, and holding it fixed would make the smaller pages match more loosely than
    intended rather than equivalently.
    """
    tolerance = max(2, round(RULE_TOLERANCE * scale))
    candidates = [r for r in rules if abs(r - expected) <= tolerance]
    return min(candidates, key=lambda r: abs(r - expected)) if candidates else None


def _column(v_rules: Sequence[int], expected: int, fallback: int, scale: float = 1.0) -> int:
    """Snap to a detected vertical rule, falling back to the template position."""
    match = _match(v_rules, expected, scale)
    return match if match is not None else fallback


# ------------------------------------------------------------------------------- grid


def build_grid(image: Image.Image, stable_rules: Sequence[int] = UPPER_H_RULES) -> Grid:
    """Detect the grid and map it to named cells.

    ``stable_rules`` are the upper anchors for the language the page is printed in, taken
    from its ``LanguageProfile``. Only the *upper* rules are language-specific; everything
    below section 2 is located structurally, because its position depends on how much text
    the page carries rather than on the language.

    Raises ``LayoutError`` when the stable rules are absent, which is the signal that a
    page is not the standard info-page template and should go to review rather than
    silently produce garbage cells.
    """
    width, height = image.size
    scale = width / CANONICAL_WIDTH
    px = lambda value: int(round(value * scale))  # noqa: E731 -- canonical px -> this page

    h_rules = detect_h_rules(image, scale)

    upper: List[int] = []
    missing: List[int] = []
    for expected in stable_rules:
        found = _match(h_rules, px(expected), scale)
        if found is None:
            missing.append(expected)
        else:
            upper.append(found)
    if missing:
        raise LayoutError(
            f"page does not match the info-page template; missing rules at {missing} "
            f"(found {len(h_rules)}: {h_rules[:16]})"
        )

    if len(h_rules) < len(stable_rules) + LOWER_RULE_COUNT + S4_RULE_COUNT:
        raise LayoutError(
            f"expected at least {len(stable_rules) + LOWER_RULE_COUNT + S4_RULE_COUNT} "
            f"rules, got {len(h_rules)}"
        )

    # Section 4 is anchored from the end: its header, two header rows, then the values.
    s4_rules = h_rules[-S4_RULE_COUNT:]
    s4_top, value_top, value_bottom = s4_rules[0], s4_rules[-2], s4_rules[-1]

    # The three rules between section 2's top and section 4 -- the bottom of the area
    # list and the two that bracket the polling-station name. They are located by
    # position from the end rather than by absolute pixel, because they move: on English
    # pages three variants appear 16px apart depending on how far the address wraps, and
    # Bengali sits 24px below Assamese throughout. Anchoring them to the end of the rule
    # list makes one rule work for every language and every variant.
    lower = h_rules[-(S4_RULE_COUNT + LOWER_RULE_COUNT) : -S4_RULE_COUNT]
    if lower[0] <= upper[-1]:
        raise LayoutError(f"section-2 block is empty or inverted: rules {upper[-1]}..{lower[0]}")

    # Vertical rules are detected per band; the page-wide scan misses short dividers.
    # Measured identical to within 3px across all three languages, so they need no
    # per-language table -- `_column` snaps to whatever is detected anyway.
    header_v = detect_v_rules(image, upper[HEADER_TOP], upper[HEADER_MID])
    s1_v = detect_v_rules(image, upper[S1_TOP], upper[S1_BOTTOM])
    s2_v = detect_v_rules(image, upper[S2_TOP], lower[0])
    s3_v = detect_v_rules(image, lower[1], lower[2])
    s4_v = detect_v_rules(image, value_top, value_bottom)

    part_split = _column(header_v, px(947), px(947), scale)
    s1_split = _column(s1_v, px(497), px(497), scale)
    s2_split = _column(s2_v, px(502), px(502), scale)
    s3_split = _column(s3_v, px(497), px(497), scale)
    s3_right = _column(s3_v, px(947), px(947), scale)

    # Six elector columns; fall back to template positions if a divider is faint.
    defaults = (24, 156, 286, 497, 723, 947, 1162)
    cols = [_column(s4_v, px(expected), px(expected), scale) for expected in defaults]

    cells: Dict[str, Box] = {
        # header
        "header_ac": Box(px(FRAME_LEFT), upper[HEADER_TOP], part_split, upper[HEADER_MID]),
        "header_part_no": Box(part_split, upper[HEADER_TOP], px(FRAME_RIGHT), upper[HEADER_MID]),
        "header_pc": Box(px(FRAME_LEFT), upper[HEADER_MID], px(FRAME_RIGHT), upper[HEADER_BOTTOM]),
        # section 1 -- revision
        "s1_revision": Box(px(FRAME_LEFT), upper[S1_TOP], s1_split, upper[S1_BOTTOM]),
        "s1_description": Box(s1_split, upper[S1_TOP], px(FRAME_RIGHT), upper[S1_BOTTOM]),
        # section 2 -- areas and locality
        "s2_areas": Box(px(FRAME_LEFT), upper[S2_TOP], s2_split, lower[0]),
        "s2_locality": Box(s2_split, upper[S2_TOP], px(FRAME_RIGHT), lower[0]),
        # section 3 -- polling station
        # The left-hand station cell spans the whole block, label rows included. The
        # right-hand column is divided at lower[2], but the left is not -- the name and
        # address run continuously beneath their labels. Splitting the left side at that
        # divider only works if it happens to fall in whitespace, which it does in
        # Assamese and does not in English, where it cuts the station name in half.
        # `parse_station` finds the label rows by reading them, which does not care.
        "s3_ps_name": Box(px(FRAME_LEFT), lower[1], s3_split, lower[2]),
        "s3_type_value": Box(s3_right, lower[1], px(FRAME_RIGHT), lower[2]),
        "s3_address": Box(px(FRAME_LEFT), lower[1], s3_split, s4_top),
        "s3_aux_value": Box(s3_right, lower[2], px(FRAME_RIGHT), s4_top),
        # section 4 -- electors (pure digits)
        "s4_start_serial": Box(cols[0], value_top, cols[1], value_bottom),
        "s4_end_serial": Box(cols[1], value_top, cols[2], value_bottom),
        "s4_male": Box(cols[2], value_top, cols[3], value_bottom),
        "s4_female": Box(cols[3], value_top, cols[4], value_bottom),
        "s4_third_gender": Box(cols[4], value_top, cols[5], value_bottom),
        "s4_total": Box(cols[5], value_top, cols[6], value_bottom),
        # footer -- "মুঠ পৃষ্ঠা <N> - পৃষ্ঠা 1" sits in the last inch of the page
        "footer": Box(px(FRAME_LEFT), height - px(FOOTER_HEIGHT), px(FRAME_RIGHT), height),
    }

    return Grid(width=width, height=height, h_rules=tuple(h_rules), cells=cells, scale=scale)


#: Cells whose content is digits only -- OCR'd with a digit whitelist and verified by
#: the hard checks in ``validate.py``.
NUMERIC_CELLS = (
    "s4_start_serial",
    "s4_end_serial",
    "s4_male",
    "s4_female",
    "s4_third_gender",
    "s4_total",
    "s3_aux_value",
)

#: Cells holding Assamese script, where engine choice actually matters.
TEXT_CELLS = (
    "header_ac",
    "header_pc",
    "s1_revision",
    "s1_description",
    "s2_areas",
    "s2_locality",
    "s3_ps_name",
    "s3_address",
    "s3_type_value",
)


def debug_overlay(image: Image.Image, grid: Grid) -> Image.Image:
    """Draw the detected cells on a copy of the page, so a bad grid is obvious."""
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    for name, box in grid.cells.items():
        colour = (200, 30, 30) if name in NUMERIC_CELLS else (30, 90, 200)
        draw.rectangle([box.x0, box.y0, box.x1, box.y1], outline=colour, width=2)
        draw.text((box.x0 + 4, box.y0 + 2), name, fill=colour)
    return canvas
