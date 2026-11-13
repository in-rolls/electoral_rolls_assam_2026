"""Turn a detected grid plus an OCR engine into schema rows.

Every field is read from a cell whose position is known, so the parser never has to
recognize a label to find a value. Two structures recur and are handled positionally:

* **Stacked label/value blocks** (section 1's revision details, section 2's locality
  block) are split into text rows by ink profile, then each row is cropped to the value
  column. Field identity comes from the **row index**, not from reading the label --
  Tesseract mangles labels (``ব্লক`` becomes ``বক``) and sometimes reads the colon as
  ``£``, both of which would break a text-based split.
* **Wrapped values.** The polling-station name is printed under its label, so it lands
  in the ``s3_address`` cell rather than ``s3_ps_name``; the address follows its own
  label row. Both are addressed by row index within that cell.

Numbers are only ever read with ``read_digits`` from digit-only crops -- see ``ocr`` for
why mixing the Assamese model into number reading corrupts results silently.
"""

from __future__ import annotations

import difflib
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from .languages import LanguageProfile
from .layout import Grid
from .ocr import (
    ELECTOR_RETRY_SCALES,
    MULTILINE_TEXT_SCALE,
    TEXT_FALLBACK_SCALES,
    Engine,
    int_or_none,
    value_after_label,
)
from .render import PartRef
from .schema import (
    PIPELINE_VERSION,
    clean_text,
    derive_columns,
    empty_part_row,
    normalize_digits,
    parse_source_filename,
)

#: Row order of the section-2 locality block, top to bottom. The label the form prints
#: beside each row comes from the page's ``LanguageProfile`` -- it differs by language,
#: while the row *order* does not. The label is what makes the mapping checkable: if row
#: 3 does not read as the police-station label, the rows have shifted and every field
#: below it is suspect.
#: The Assamese edition's rows -- the core eight, printed by every language.
LOCALITY_FIELDS = (
    "main_town_village",
    "ward_no",
    "post_office",
    "police_station",
    "block",
    "revenue_circle",
    "district",
    "pin_code",
)

#: Every locality field any edition prints. The core eight are common to all three; the
#: Bengali form adds ``gram_panchayat`` after the police station and the English form adds
#: ``subdivision`` after the revenue circle (which it labels "Tehsil"). Confirmed against
#: the rendered pages, not inferred from OCR.
#:
#: A language that does not print a row leaves it **blank**, not null: the absence is a
#: property of the form, not a failure to read it.
ALL_LOCALITY_FIELDS = (
    "main_town_village",
    "ward_no",
    "post_office",
    "police_station",
    "gram_panchayat",
    "block",
    "revenue_circle",
    "subdivision",
    "district",
    "pin_code",
)

#: Row order of the section-1 revision block.
REVISION_FIELDS = (
    "revision_year",
    "qualifying_date",
    "revision_type",
    "publication_date",
)

#: Minimum similarity for an OCR'd label to be accepted as a given field's label.
#: Tesseract mangles labels (ব্লক -> বক), so this is deliberately forgiving; it only has
#: to be sharp enough to tell one label from the seven others in its block.
LABEL_MATCH_THRESHOLD = 0.55

#: How far short of the value column to stop when counting label rows.
LABEL_STRIP_MARGIN = 14

#: A column inked down at least this share of a cell's height is a rule, not text. The
#: tallest glyph in this corpus covers well under half a cell; a border covers all of it.
VERTICAL_RULE_FRACTION = 0.8

#: A label is separated from its value by a wide run of whitespace; word spaces inside a
#: label are far narrower. Measured gaps are ~95-131px against ~10px word spacing.
MIN_LABEL_GAP = 40

DATE_RE = re.compile(r"(\d{2})-(\d{2})-(\d{4})")
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
#: "<n> - <name> (<reservation>)" as printed in the AC/PC header and the station name.
NUMBERED_NAME_RE = re.compile(r"^\s*(\d+)\s*[-–]\s*(.*?)\s*$")


def text_rows(
    image: Image.Image,
    gap: int = 4,
    min_ink: int = 2,
    ink: int = 150,
    right: Optional[int] = None,
) -> List[Tuple[int, int]]:
    """Y extents of each text row in a cell, found from the horizontal ink profile.

    ``right`` limits the scan to a left-hand strip. Passing the value column's x makes
    this count **label** rows only, which is what a stacked block actually has: on the
    English form a long value wraps to a second line, and scanning the whole cell counts
    that wrap as its own row -- giving 9, 10 or 11 rows for the same 9 fields and making
    positional mapping impossible. The label column never wraps.
    """
    width, height = image.size
    if right is not None:
        width = max(1, min(width, right))
    pixels = image.convert("L").load()

    # A cell border that survives the crop inset is ink on *every* row, so it makes the
    # whole cell read as one continuous text row: six parts came back with their entire
    # locality block unread because a vertical rule sat inside the left edge. Columns that
    # are inked down almost the full height are rules, not glyphs -- no letter is -- so
    # they are excluded before rows are counted.
    columns = [
        x
        for x in range(width)
        if sum(1 for y in range(height) if pixels[x, y] < ink) < height * VERTICAL_RULE_FRACTION
    ]
    hits = [y for y in range(height) if sum(1 for x in columns if pixels[x, y] < ink) > min_ink]
    groups: List[List[int]] = []
    for y in hits:
        if groups and y - groups[-1][-1] <= gap:
            groups[-1].append(y)
        else:
            groups.append([y])
    return [(g[0], g[-1]) for g in groups]


def _row_crop(cell: Image.Image, span: Tuple[int, int], x0: int = 0, pad: int = 4):
    top = max(0, span[0] - pad)
    bottom = min(cell.height, span[1] + pad + 1)
    return cell.crop((max(0, x0), top, cell.width, bottom))


def value_start_x(
    cell: Image.Image, span: Tuple[int, int], default: int, ink: int = 150, scale: float = 1.0
) -> int:
    """Where the value begins on a ``label   value`` row.

    Found from the widest run of whitespace in the row: a label is separated from its
    value by a gap far wider than the spaces inside it. This self-calibrates, which
    matters because the revision block's values are identical corpus-wide and so cannot
    be located by comparing pages. Falls back to ``default`` when no wide gap exists --
    the signature of an empty value.
    """
    pixels = cell.convert("L").load()
    columns = [
        x for x in range(cell.width) if any(pixels[x, y] < ink for y in range(span[0], span[1] + 1))
    ]
    if len(columns) < 2:
        return default
    width, start = max(((b - a, a) for a, b in zip(columns, columns[1:])), default=(0, 0))
    return start + width if width >= MIN_LABEL_GAP * scale else default


#: The section-4 cells, in the order the elector table prints them.
S4_CELLS = (
    ("s4_start_serial", "start_serial"),
    ("s4_end_serial", "end_serial"),
    ("s4_male", "electors_male"),
    ("s4_female", "electors_female"),
    ("s4_third_gender", "electors_third_gender"),
    ("s4_total", "electors_total"),
)


def electors_balance(values: Dict[str, Any]) -> bool:
    """Whether ``male + female + third == total`` for a set of readings."""
    parts = [
        values.get(k)
        for k in ("electors_male", "electors_female", "electors_third_gender", "electors_total")
    ]
    if any(v is None for v in parts):
        return False
    return parts[0] + parts[1] + parts[2] == parts[3]


def balance_electors(grid: Grid, image: Image.Image, engine: Engine, row: Dict[str, Any]):
    """Re-read the elector table at another scale when its arithmetic does not hold.

    This is the one place the pipeline can tell a *wrong* reading from a right one without
    a human, because the four numbers must sum. That makes a retry safe in a way it is not
    elsewhere: a candidate reading is accepted **only if it balances**, so a second guess
    can never replace a correct row with a plausible wrong one.

    It is needed because the failure here is a wrong value rather than an empty one, which
    the empty-read fallbacks cannot see. Tesseract renders the printed ``779`` as ``7719``
    at every scale above 1 -- deterministically, on 37 parts across eleven unrelated
    constituencies, always the same misread of the same string. Scale 1 reads it correctly
    and reads the controls correctly too.

    Returns the corrected fields, or ``None`` when the row already balances or no rescaled
    reading does -- in which case the row keeps its original values and stays flagged.
    """
    if electors_balance(row):
        return None

    readings = [{field: row.get(field) for _, field in S4_CELLS}]
    for scale in ELECTOR_RETRY_SCALES:
        readings.append(
            {
                field: int_or_none(engine.read_digits(grid.crop(image, cell), scale=scale))
                for cell, field in S4_CELLS
            }
        )

    # A whole reading from one scale, first. This is the common case and the safest: every
    # number comes from the same pass.
    for candidate in readings[1:]:
        if electors_balance(candidate):
            return candidate

    # Otherwise the misread may be confined to one side. A scale can resolve the total and
    # mangle a component, or the reverse -- one page reads 779 correctly at scale 1 while
    # turning 411 into 4. Pairing the components from one pass with the total from another
    # is still gated on the arithmetic holding, so a mixed reading is only accepted when
    # the three components independently sum to the total read. That is a far stronger
    # constraint than either half satisfies alone.
    components = ("electors_male", "electors_female", "electors_third_gender")
    for triple in readings:
        if any(triple.get(k) is None for k in components):
            continue
        summed = sum(triple[k] for k in components)
        for other in readings:
            if other.get("electors_total") != summed:
                continue
            merged = {**triple, "electors_total": summed}
            merged["start_serial"] = triple.get("start_serial") or other.get("start_serial")
            merged["end_serial"] = triple.get("end_serial") or other.get("end_serial")
            return merged
    return None


#: Fraction of a ``label : number`` cell to search for the label/value boundary. The gap
#: before the number is only ~9px -- far narrower than the locality block's ~95-131px --
#: so it competes with the gaps *inside* the label. Restricting the search to the right of
#: the cell removes that competition.
TRAILING_NUMBER_SEARCH_FROM = 0.45


def read_trailing_number(cell: Image.Image, engine: Engine, ink: int = 150) -> str:
    """Read a number printed after a label in the same cell, using the digit engine.

    The part-number cell reads ``খণ্ড নং : 103``. Handing the whole thing to the language
    model and pulling the integer out of the result is wrong for the same reason the
    elector cells are read separately: a language model does not transcribe digits
    faithfully. It rendered ``103`` as ``1093`` and ``133`` as ``1393`` -- a spurious 9 --
    and on Assamese pages it turned ``8`` into ``৪``, which the ASCII-only guard then
    correctly refused, losing the value entirely.

    Cropping to the right of the label and reading *that* with ``read_digits`` fixes both:
    measured over 216 parts sampled across 24 constituencies and all three languages,
    98.15% -> 100.00%.
    """
    pixels = cell.convert("L").load()
    columns = [x for x in range(cell.width) if any(pixels[x, y] < ink for y in range(cell.height))]
    if len(columns) < 2:
        return ""
    threshold = int(cell.width * TRAILING_NUMBER_SEARCH_FROM)
    gaps = [(b - a, a) for a, b in zip(columns, columns[1:]) if a >= threshold]
    if not gaps:
        gaps = [(b - a, a) for a, b in zip(columns, columns[1:])]
    width, start = max(gaps)
    return engine.read_digits(cell.crop((start + width, 0, cell.width, cell.height)))


#: Characters Tesseract produces where the form prints a colon. It reads the locality
#: block's colon as "£" often enough that stripping only ":" leaves it in the value.
COLON_LIKE = ":;£¢|"


def strip_leading_colon(text: str) -> str:
    """Drop a leading colon (however it was transcribed) from a value."""
    return clean_text(text or "").lstrip(COLON_LIKE + " ").strip()


def iso_date(text: str) -> str:
    """``01-01-2026`` as printed becomes ``2026-01-01``."""
    match = DATE_RE.search(text or "")
    if not match:
        return ""
    day, month, year = match.groups()
    return f"{year}-{month}-{day}"


def split_numbered_name(value: str) -> Tuple[Optional[int], str]:
    """``"1 - গোসাইগাঁও (সাধাৰণ)"`` becomes ``(1, "গোসাইগাঁও (সাধাৰণ)")``."""
    match = NUMBERED_NAME_RE.match(value or "")
    if not match:
        return int_or_none(value), clean_text(value)
    return int(match.group(1)), clean_text(match.group(2))


def split_reservation(name: str, profile: LanguageProfile) -> Tuple[str, str]:
    """Separate a trailing ``(reservation)`` from a constituency name."""
    match = re.match(r"^(.*?)\s*\(([^)]*)\)\s*$", name or "")
    if not match:
        return clean_text(name), ""
    bare, bracket = clean_text(match.group(1)), clean_text(match.group(2))
    return bare, profile.reservation_map.get(bracket, "")


def normalize_ps_type(value: str, profile: LanguageProfile) -> str:
    return profile.ps_type_map.get(clean_text(value), "")


# ------------------------------------------------------------------------------ blocks


def has_ink(image: Image.Image, ink: int = 150, min_pixels: int = 6) -> bool:
    """Whether a crop contains enough dark pixels to be holding text.

    This is what separates the two kinds of missing value. If the value region carries
    ink but OCR returned nothing, the pipeline failed to read something that is there
    (``None``). If there is no ink, the form simply prints nothing there (``""``) --
    rural parts have no ward number, and that absence is data, not an error.
    """
    grey = image.convert("L")
    pixels = grey.load()
    width, height = grey.size
    seen = 0
    for y in range(height):
        for x in range(width):
            if pixels[x, y] < ink:
                seen += 1
                if seen >= min_pixels:
                    return True
    return False


def read_text_retrying(engine: Engine, crop: Image.Image, scale: Optional[int] = None) -> str:
    """Read a text crop, retrying at other scales when the first attempt is empty.

    The same failure the digit cells have, on text: a row flush against the top of its
    cell reads cleanly at scale 1 and returns **empty** at the default scale 2. It is not
    a marginal glyph -- ``1-কলিয়াগাওঁ`` is plainly legible, and scale 2 returns "".

    The retries fire only when the first read yields nothing, so a value that already read
    is never re-read and never changed. Left unhandled this lost ~600 values across the
    corpus: 364 ``block``, 162 ``post_office``, 24 ``main_town_village`` and 52 whole
    section lists, all of them present on the page.
    """
    text = clean_text(engine.read_text(crop, scale=scale) if scale else engine.read_text(crop))
    if text:
        return text
    for fallback in TEXT_FALLBACK_SCALES:
        if fallback == scale:
            continue
        text = clean_text(engine.read_text(crop, scale=fallback))
        if text:
            return text
    return ""


def read_value(engine: Engine, crop: Image.Image, scale: Optional[int] = None):
    """Read one value crop, distinguishing "unreadable" from "blank".

    Returns ``""`` when the crop holds no ink, ``None`` when it holds ink but yields no
    text, and the text otherwise.
    """
    text = read_text_retrying(engine, crop, scale)
    if text:
        return text
    return None if has_ink(crop) else ""


def _read_region(cell: Image.Image, engine: Engine, y0: int, y1: int) -> str:
    """OCR a whole vertical region as one block, joining its lines with spaces.

    Used for free-form multi-line values (station name, address) where there is no
    field mapping to preserve and slicing into rows risks cutting glyphs.
    """
    top, bottom = max(0, y0), min(cell.height, y1)
    if bottom - top < 4:
        return ""
    # A taller region needs more upscaling: at 2x, Tesseract silently drops one line of
    # a two-line address. Short single-line values are read at the default scale, where
    # 3x instead loses long values entirely.
    return engine.read_text(cell.crop((0, top, cell.width, bottom)), scale=MULTILINE_TEXT_SCALE)


def label_similarity(seen: str, expected: str) -> float:
    """How closely an OCR'd label matches the label the form prints."""
    return difflib.SequenceMatcher(None, clean_text(seen), clean_text(expected)).ratio()


def _align_by_label(
    cell: Image.Image,
    engine: Engine,
    spans: List[Tuple[int, int]],
    fields: Tuple[str, ...],
    labels: Tuple[str, ...],
    value_x: int,
) -> Tuple[Dict[str, Tuple[int, int]], List[str]]:
    """Assign rows to fields by reading each row's *label*.

    Only used when the row count is not the expected one -- reading the key as well as
    the value costs an OCR call per row, and positional mapping is already correct when
    every row is present. When a row is missing (a blank value can merge or vanish),
    positional mapping would shift every subsequent field silently, so here the label
    decides.
    """
    assigned: Dict[str, Tuple[int, int]] = {}
    notes: List[str] = []
    for span in spans:
        label_text = engine.read_text(cell.crop((0, max(0, span[0] - 4), value_x, span[1] + 5)))
        scores = [(label_similarity(label_text, lab), i) for i, lab in enumerate(labels)]
        best, index = max(scores)
        if best < LABEL_MATCH_THRESHOLD:
            notes.append(f"unmatched_label:{label_text[:20]}")
            continue
        field = fields[index]
        if field not in assigned:
            assigned[field] = span
    return assigned, notes


def _stacked_block(
    cell: Image.Image,
    engine: Engine,
    fields: Tuple[str, ...],
    labels: Tuple[str, ...],
    default_x: int,
    dynamic: bool = False,
    scale: float = 1.0,
) -> Tuple[Dict[str, Image.Image], List[str]]:
    """Crop a stacked label/value block to one value image per field.

    Fast path: when the number of detected rows equals the number of fields, rows map to
    fields by index. Otherwise the labels are read and used to align, which is what keeps
    a missing row from shifting every field beneath it.

    ``dynamic`` locates the label/value boundary from the widest whitespace run, which
    the revision block needs because its values are constant corpus-wide and so cannot be
    located by comparing pages. The locality block instead has a fixed colon column, and
    a dynamic boundary there would land *before* the colon and pull it into the value.
    """
    # Rows are counted from the label column only, stopping short of the printed colon.
    # A wrapped value continues at the value column, and scanning as far as that column
    # lets its continuation line register as a row of its own -- English part 1 came back
    # with 11 rows for 9 fields that way.
    spans = text_rows(cell, right=max(20, default_x - round(LABEL_STRIP_MARGIN * scale)))
    notes: List[str] = []

    if len(spans) == len(fields):
        by_field = {fields[i]: span for i, span in enumerate(spans)}
    else:
        notes.append(f"row_count:{len(spans)}!={len(fields)}")
        by_field, extra = _align_by_label(cell, engine, spans, fields, labels, default_x)
        notes += extra

    crops: Dict[str, Image.Image] = {}
    for field, span in by_field.items():
        x0 = value_start_x(cell, span, default_x, scale=scale) if dynamic else default_x
        crops[field] = _row_crop(cell, span, x0)
    return crops, notes


def parse_locality(
    cell: Image.Image, engine: Engine, profile: LanguageProfile, scale: float = 1.0
) -> Tuple[Dict[str, Any], List[str]]:
    crops, notes = _stacked_block(
        cell,
        engine,
        profile.locality_fields,
        profile.locality_labels,
        round(profile.locality_value_x * scale),
        scale=scale,
    )
    # Every locality field in the schema starts as "" -- the form for this language simply
    # does not print it (gram_panchayat outside Bengali, subdivision outside English), which
    # is blank rather than unread. Rows this language *does* print but that were not
    # detected become None below, which is the honest "could not read".
    parsed: Dict[str, Any] = {name: "" for name in ALL_LOCALITY_FIELDS}
    for name in profile.locality_fields:
        parsed[name] = None
    for name, crop in crops.items():
        # The pincode is printed in Latin digits, so read it with the digit engine
        # rather than the Assamese one, which can render an 8 as ৪.
        if name == "pin_code":
            digits = engine.read_digits(crop)
            parsed[name] = int_or_none(digits) if digits else (None if has_ink(crop) else "")
        else:
            value = read_value(engine, crop)
            parsed[name] = strip_leading_colon(value) if value else value
    return parsed, notes


def parse_revision(
    cell: Image.Image, engine: Engine, profile: LanguageProfile, scale: float = 1.0
) -> Tuple[Dict[str, Any], List[str]]:
    crops, notes = _stacked_block(
        cell,
        engine,
        REVISION_FIELDS,
        profile.revision_labels,
        round(profile.revision_value_x * scale),
        dynamic=True,
        scale=scale,
    )
    values = {name: engine.read_text(crop) for name, crop in crops.items()}
    return {
        "revision_year": int_or_none(values.get("revision_year", "")),
        "qualifying_date": iso_date(values.get("qualifying_date", "")),
        "revision_type": clean_text(values.get("revision_type", "")),
        "publication_date": iso_date(values.get("publication_date", "")),
    }, notes


#: A leading "<n> - " on the station-name line, where <n> may be garbled by OCR. The
#: number is discarded (it is the part number, known authoritatively from the filename),
#: but the prefix must be stripped or the garbage lands in the name.
LEADING_SERIAL_RE = re.compile(r"^\s*\S{0,6}?\s*[-–]\s*")


def parse_station(
    grid: Grid, image: Image.Image, engine: Engine, profile: LanguageProfile
) -> Tuple[Dict[str, Any], list]:
    """Polling station name and address.

    Both wrap. The name is printed *below* its label so it lands in the address cell,
    and either the name or the address can run to a second line. Rows before the address
    label belong to the name; rows from the label onward belong to the address. Taking
    only the first row (an earlier version) truncated wrapped names and, when the name
    wrapped, dropped the start of the address.

    ``ps_no`` is deliberately **not** parsed here. It always equals the part number --
    verified against source images for every mismatch in the corpus -- and reading it
    off the page misread the leading digits on 10% of pages, corrupting ``ps_name`` too.
    The caller supplies it from the filename.
    """
    cell = grid.crop(image, "s3_address")
    spans = text_rows(cell)
    notes: List[str] = []
    if not spans:
        return {"ps_name": "", "ps_address": ""}, ["address_cell_empty"]

    # Locate the address label row. Only this row is read individually; the name and
    # address are then read as whole regions.
    lines = [engine.read_text(_row_crop(cell, span)) for span in spans]

    # The *best* match, not the first one over the threshold. The cell also opens with the
    # station-name label, and the two labels are similar enough to be confused -- Assamese
    # prints "ভোটকেন্দ্ৰৰ নং আৰু নাম" above "ভোটগ্ৰহন কেন্দ্ৰৰ ঠিকনা", English "No. and Name of
    # Polling Station" above "Address of Polling Station". Taking the first match over the
    # line put the whole record into ps_address and left ps_name empty.
    scores = [
        (label_similarity(line.split(":")[0], profile.address_label), i)
        for i, line in enumerate(lines)
    ]
    best_score, label_index = max(scores)
    if best_score < LABEL_MATCH_THRESHOLD:
        label_index = next((i for i, line in enumerate(lines) if ":" in line), 1)
        notes.append("address_label_not_found")

    # The cell opens with the station-name label ("No. and Name of Polling Station :"),
    # printed above the name it introduces. Skip it, or it lands in ps_name.
    name_top = 0
    if label_index > 0 and ":" in lines[0]:
        name_top = spans[0][1] + 2

    # Read each side as ONE region rather than row by row. Assamese hangs glyphs below
    # the headline, so a boundary drawn from the ink profile can slice through
    # descenders and render a line unreadable -- observed on AC12 part 47, where the
    # first address line OCR'd to an empty string and the address lost its opening.
    name = _read_region(cell, engine, name_top, spans[label_index][0] - 2)
    address_tail = value_after_label(lines[label_index]) if label_index < len(lines) else ""
    below = _read_region(cell, engine, spans[label_index][1] + 2, cell.height)
    address = " ".join(part for part in (address_tail, below) if part)

    return {
        "ps_name": clean_text(LEADING_SERIAL_RE.sub("", clean_text(name))),
        "ps_address": clean_text(address),
    }, notes


def parse_sections(grid: Grid, image: Image.Image, engine: Engine) -> List[Dict[str, Any]]:
    """The numbered area list, one entry per printed line."""
    cell = grid.crop(image, "s2_areas")
    rows: List[Dict[str, Any]] = []
    for index, span in enumerate(text_rows(cell), start=1):
        line = read_text_retrying(engine, _row_crop(cell, span))
        if not line:
            continue
        number, name = split_numbered_name(line)
        rows.append(
            {
                "section_no": number if number is not None else index,
                "section_name": name,
                "section_name_digits": normalize_digits(name),
            }
        )
    return rows


# -------------------------------------------------------------------------------- page


def failed_row(
    ref: PartRef,
    engine_name: str,
    reason: str,
    provenance: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """A row for a page that could not be parsed at all.

    Emitted rather than dropped so the part still appears in the dataset, flagged for
    review, with its provenance intact. A silently missing part is far harder to notice
    than one that says why it failed.
    """
    row = empty_part_row()
    row.update(
        {
            "source_zip": ref.zip_name,
            "source_pdf": ref.pdf_name,
            "ac_no_file": ref.ac_no,
            "part_no_file": ref.part_no,
            "engine": engine_name,
            "pipeline_version": PIPELINE_VERSION,
            "extracted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "template_match": False,
            "flags": "layout_failed",
            "needs_review": True,
            "anomaly_notes": reason[:200],
        }
    )
    parsed_name = parse_source_filename(Path(ref.pdf_name).name)
    if parsed_name:
        row.update(
            {
                "roll_year": parsed_name["year"],
                "state_code": parsed_name["state"],
                "roll_type": parsed_name["roll_type"],
                "revision_no": parsed_name["revision"],
                "lang": parsed_name["lang"],
            }
        )
    row.update(provenance or {})
    return row


def parse_page(
    image: Image.Image,
    grid: Grid,
    ref: PartRef,
    engine: Engine,
    profile: LanguageProfile,
    provenance: Optional[Dict[str, Any]] = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Read one page into a part row plus its section rows.

    ``profile`` says which language the page is printed in, which decides the labels the
    row-alignment fallback matches against and the vocabulary the categorical fields map
    through. It is required rather than defaulted: 14 of the 126 constituencies are not
    Assamese, and defaulting would read them with the wrong model in silence.

    ``provenance`` carries the source hashes and run metadata the caller has but the
    parser does not (it never sees the PDF bytes or the page file). Everything needed to
    stitch a row back to its source file is written here rather than bolted on later.
    """
    row = empty_part_row()

    # ac_no/part_no come from the filename, which is authoritative publisher metadata;
    # the OCR readings below are kept only as a quality signal.
    row.update(
        {
            "source_zip": ref.zip_name,
            "source_pdf": ref.pdf_name,
            "ac_no_file": ref.ac_no,
            "part_no_file": ref.part_no,
            "engine": getattr(engine, "name", "unknown"),
            "pipeline_version": PIPELINE_VERSION,
            "extracted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
    )

    # The filename encodes the roll year, state, roll type, revision and language.
    parsed_name = parse_source_filename(Path(ref.pdf_name).name)
    if parsed_name:
        row.update(
            {
                "roll_year": parsed_name["year"],
                "state_code": parsed_name["state"],
                "roll_type": parsed_name["roll_type"],
                "revision_no": parsed_name["revision"],
                "lang": parsed_name["lang"],
            }
        )
    row.update(provenance or {})

    ac_no, ac_rest = split_numbered_name(
        value_after_label(engine.read_text(grid.crop(image, "header_ac")))
    )
    ac_name, ac_reservation = split_reservation(ac_rest, profile)
    pc_no, pc_rest = split_numbered_name(
        value_after_label(engine.read_text(grid.crop(image, "header_pc")))
    )
    pc_name, pc_reservation = split_reservation(pc_rest, profile)

    row.update(
        {
            "ac_no": ac_no,
            "ac_name": ac_name,
            "ac_reservation": ac_reservation,
            "pc_no": pc_no,
            "pc_name": pc_name,
            "pc_reservation": pc_reservation,
            "part_no": int_or_none(
                read_trailing_number(grid.crop(image, "header_part_no"), engine)
            ),
        }
    )

    notes: List[str] = []

    revision, revision_notes = parse_revision(
        grid.crop(image, "s1_revision"), engine, profile, grid.scale
    )
    row.update(revision)
    notes += revision_notes

    description = engine.read_text(grid.crop(image, "s1_description"))
    row["roll_description"] = clean_text(description)
    years = [int(y.group()) for y in YEAR_RE.finditer(description)]
    row["mother_roll_year"] = min(years) if years else None

    locality, locality_notes = parse_locality(
        grid.crop(image, "s2_locality"), engine, profile, grid.scale
    )
    row.update(locality)
    notes += locality_notes

    station, station_notes = parse_station(grid, image, engine, profile)
    row.update(station)
    notes += station_notes

    # The station number always equals the part number; see parse_station.
    row["ps_no"] = ref.part_no
    row["ps_type"] = normalize_ps_type(engine.read_text(grid.crop(image, "s3_type_value")), profile)
    row["auxiliary_ps_count"] = int_or_none(engine.read_digits(grid.crop(image, "s3_aux_value")))

    for cell_name, field in S4_CELLS:
        row[field] = int_or_none(engine.read_digits(grid.crop(image, cell_name)))
    reread = balance_electors(grid, image, engine, row)
    if reread:
        row.update(reread)
        notes.append("electors_rescaled")

    row["total_pages"] = int_or_none(engine.read_text(grid.crop(image, "footer")))
    row["template_match"] = True
    row["anomaly_notes"] = ";".join(notes)
    row.update(derive_columns(row))

    sections = [
        {"ac_no": ref.ac_no, "part_no": ref.part_no, **section}
        for section in parse_sections(grid, image, engine)
    ]
    return row, sections
