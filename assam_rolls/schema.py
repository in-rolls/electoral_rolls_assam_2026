"""Schema for Assam 2026 electoral roll info pages.

Single source of truth for three things that must never drift apart:

1. ``PART_COLUMNS`` / ``SECTION_COLUMNS`` -- the CSV output headers.
2. ``PAGE1_JSON_SCHEMA`` -- the JSON Schema handed to Claude's structured outputs,
   which guarantees the response parses without regex salvage.
3. ``parse_source_filename`` -- recovers the AC and part number from the PDF filename.
   The publisher encodes both in every filename, so they are independently-known
   ground truth on every page at zero labeling cost (see ``validate.py``).

Text conventions
----------------
Assamese text is stored **verbatim**, including Assamese numerals where the source
uses them (``ps_address`` reads ``৫৯০ বনগাঁও এল. পি. স্কুল, কঠা নং ১, বনগাঁও``).
Each such field has a ``*_roman`` companion holding a model-produced romanization.
The verbatim column is the archival record; ``*_roman`` is a convenience for joining
and is explicitly model-generated -- it is scored separately from the verbatim fields
because transliteration has no single correct answer.

The three categorical fields are normalized to a controlled English vocabulary rather
than transliterated, so they can be filtered on directly.
"""

import re
import unicodedata
from typing import Any, Dict, List, Mapping, Optional, Tuple

# --------------------------------------------------------------------------- filename


def source_file_key(row: Mapping[str, Any]) -> Tuple[int, int]:
    """The authoritative AC/part key encoded in an info-page source filename.

    ``ac_no`` and ``part_no`` are OCR outputs kept to measure extraction accuracy. Joining on
    them makes an OCR error change record identity, so consumers must use the independently
    known ``*_file`` fields instead.
    """
    return int(row["ac_no_file"]), int(row["part_no_file"])


# 2026-EROLLGEN-S03-100-FinalRoll-Revision1-ASM-45-WI_INFO.pdf
#                   ^AC                          ^part
SOURCE_FILENAME_RE = re.compile(
    r"^(?P<year>\d{4})-EROLLGEN-(?P<state>S\d{2})-(?P<ac_no>\d+)-"
    r"(?P<roll_type>[A-Za-z]+)-Revision(?P<revision>\d+)-"
    r"(?P<lang>[A-Z]+)-(?P<part_no>\d+)-WI(?P<info>_INFO)?\.pdf$"
)

STATE_CODE = "S03"
STATE_NAME = "Assam"
N_ASSEMBLY_CONSTITUENCIES = 126


def parse_source_filename(filename: str) -> Optional[Dict[str, Any]]:
    """Recover AC number, part number and roll metadata from a source PDF filename.

    Returns ``None`` when the name does not match the publisher's pattern, which is
    itself worth flagging -- an unrecognized name means an unexpected file in the zip.
    """
    match = SOURCE_FILENAME_RE.match(filename.strip())
    if not match:
        return None
    fields = match.groupdict()
    return {
        # The roll PDF and its info pages differ only by this suffix, which is what makes
        # the elector rows join to the part rows for free on (ac_no, part_no).
        "info_pages": bool(fields["info"]),
        "year": int(fields["year"]),
        "state": fields["state"],
        "ac_no": int(fields["ac_no"]),
        "part_no": int(fields["part_no"]),
        "roll_type": fields["roll_type"],
        "revision": int(fields["revision"]),
        "lang": fields["lang"],
    }


# -------------------------------------------------------------------- digit normalization

# Assamese uses the Bengali-Assamese numerals, U+09E6..U+09EF. The mapping to Western
# digits is a pure bijection, so it is done in code rather than asked of the model:
# exact, free, and testable. Devanagari (U+0966..U+096F) is included because roll
# addresses occasionally mix it in.
ASSAMESE_DIGITS = "০১২৩৪৫৬৭৮৯"
DEVANAGARI_DIGITS = "०१२३४५६७८९"
_DIGIT_TRANSLATION = {
    **{ord(c): str(i) for i, c in enumerate(ASSAMESE_DIGITS)},
    **{ord(c): str(i) for i, c in enumerate(DEVANAGARI_DIGITS)},
}


def normalize_digits(text: Optional[str]) -> str:
    """Rewrite Assamese/Devanagari numerals as Western digits, leaving all else intact.

    ``'৫৯০ বনগাঁও এল. পি. স্কুল, কঠা নং ১'`` -> ``'590 বনগাঁও এল. পি. স্কুল, কঠা নং 1'``
    """
    if not text:
        return ""
    return str(text).translate(_DIGIT_TRANSLATION)


def first_int(text: Optional[str]) -> Optional[int]:
    """First integer appearing in ``text`` after digit normalization, else ``None``.

    Used to derive ``ward_no_num`` from a verbatim ward label such as ``ৱাৰ্ড নং ৮``.
    """
    match = re.search(r"\d+", normalize_digits(text))
    return int(match.group()) if match else None


def clean_text(text: Optional[str]) -> str:
    """Normalize Unicode to NFC and collapse whitespace, preserving script.

    Assamese combining marks admit multiple encodings for the same grapheme; NFC makes
    verbatim fields comparable across extraction runs and against the gold set.
    """
    if not text:
        return ""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(text))).strip()


# ---------------------------------------------------------------- controlled vocabulary

RESERVATION_VALUES = ["GENERAL", "SC", "ST"]
PS_TYPE_VALUES = ["MALE", "FEMALE", "GENERAL"]
CONFIDENCE_VALUES = ["HIGH", "MEDIUM", "LOW"]

# The surface forms that map onto this vocabulary differ by language, so they live in
# the per-language profiles (``languages.py``) rather than here. The controlled values
# above are language-independent and are what the dataset actually stores.

# ------------------------------------------------------------------- model-facing fields

#: Fields the model reads off page 1, in the order they appear on the form.
#: (name, json_type, description) -- ``json_type`` is "string", "integer" or an enum list.
MODEL_FIELDS: List[Dict[str, Any]] = [
    # header
    {"name": "ac_no", "type": "integer", "desc": "Assembly constituency number."},
    {"name": "ac_name", "type": "string", "desc": "AC name, verbatim Assamese."},
    {"name": "ac_name_roman", "type": "string", "desc": "Romanization of ac_name."},
    {"name": "ac_reservation", "type": RESERVATION_VALUES, "desc": "AC reservation status."},
    {"name": "pc_no", "type": "integer", "desc": "Parliamentary constituency number."},
    {"name": "pc_name", "type": "string", "desc": "PC name, verbatim Assamese."},
    {"name": "pc_name_roman", "type": "string", "desc": "Romanization of pc_name."},
    {"name": "pc_reservation", "type": RESERVATION_VALUES, "desc": "PC reservation status."},
    {"name": "part_no", "type": "integer", "desc": "Part number (খণ্ড নং), top right."},
    # section 1 -- revision
    {"name": "revision_year", "type": "integer", "desc": "Year of revision (সংশোধনৰ বছৰ)."},
    {
        "name": "qualifying_date",
        "type": "string",
        "desc": "Qualifying date (ভিত্তি তাৰিখ) as ISO YYYY-MM-DD.",
    },
    {
        "name": "revision_type",
        "type": "string",
        "desc": "Type of revision (সংশোধনৰ প্ৰকাৰ), verbatim.",
    },
    {"name": "revision_type_roman", "type": "string", "desc": "English/romanized revision_type."},
    {
        "name": "publication_date",
        "type": "string",
        "desc": "Date of publication (প্ৰকাশনৰ তাৰিখ) as ISO YYYY-MM-DD.",
    },
    {
        "name": "roll_description",
        "type": "string",
        "desc": "Roll identification paragraph (তালিকাৰ চিনাক্তকৰণ), verbatim Assamese.",
    },
    # section 2 -- locality
    {"name": "main_town_village", "type": "string", "desc": "মূল চহৰ/গাঁও, verbatim."},
    {"name": "main_town_village_roman", "type": "string", "desc": "Romanization."},
    {"name": "ward_no", "type": "string", "desc": "ৱাৰ্ড নং, verbatim (may be empty)."},
    {"name": "ward_no_roman", "type": "string", "desc": "Romanization of ward_no."},
    {"name": "post_office", "type": "string", "desc": "ডাকঘৰ, verbatim."},
    {"name": "post_office_roman", "type": "string", "desc": "Romanization."},
    {"name": "police_station", "type": "string", "desc": "পুলিচ থানা, verbatim."},
    {"name": "police_station_roman", "type": "string", "desc": "Romanization."},
    # Printed only on the Bengali edition of the form (গ্রাম পঞ্চায়েত), between the police
    # station and the block. Blank -- not null -- on the Assamese and English editions,
    # which do not print the row at all.
    {
        "name": "gram_panchayat",
        "type": "string",
        "desc": "গ্রাম পঞ্চায়েত, verbatim (Bengali only).",
    },
    {"name": "block", "type": "string", "desc": "ব্লক (development block / পৌৰসভা), verbatim."},
    {"name": "block_roman", "type": "string", "desc": "Romanization."},
    {"name": "revenue_circle", "type": "string", "desc": "ৰাজহ চক্ৰ (revenue circle), verbatim."},
    {"name": "revenue_circle_roman", "type": "string", "desc": "Romanization."},
    # Printed only on the English edition, between the revenue circle (there labeled
    # "Tehsil") and the district.
    {"name": "subdivision", "type": "string", "desc": "Subdivision, verbatim (English only)."},
    {"name": "district", "type": "string", "desc": "জিলা, verbatim."},
    {"name": "district_roman", "type": "string", "desc": "Romanization."},
    {"name": "pin_code", "type": "integer", "desc": "পিনকোড, 6 digits."},
    # section 3 -- polling station
    {"name": "ps_no", "type": "integer", "desc": "Polling station number."},
    {"name": "ps_name", "type": "string", "desc": "Polling station name, verbatim Assamese."},
    {"name": "ps_name_roman", "type": "string", "desc": "Romanization of ps_name."},
    {"name": "ps_address", "type": "string", "desc": "Polling station address, verbatim."},
    {"name": "ps_address_roman", "type": "string", "desc": "Romanization of ps_address."},
    {"name": "ps_type", "type": PS_TYPE_VALUES, "desc": "Polling station type."},
    {
        "name": "auxiliary_ps_count",
        "type": "integer",
        "desc": "Number of auxiliary polling stations in the part.",
    },
    # section 4 -- electors
    {"name": "start_serial", "type": "integer", "desc": "আৰম্ভনি ক্ৰমিক নং."},
    {"name": "end_serial", "type": "integer", "desc": "শেষ ক্ৰমিক নং."},
    {"name": "electors_male", "type": "integer", "desc": "পুৰুষ."},
    {"name": "electors_female", "type": "integer", "desc": "মহিলা."},
    {"name": "electors_third_gender", "type": "integer", "desc": "৩য় লিঙ্গ."},
    {"name": "electors_total", "type": "integer", "desc": "মুঠ."},
    # footer
    {"name": "total_pages", "type": "integer", "desc": "Total pages of the full roll, footer."},
    {
        "name": "mother_roll_year",
        "type": "integer",
        "desc": (
            "Year of the mother roll named in the roll_description paragraph "
            "(e.g. 2025 in 'পূৰ্বৰ 2025 চনৰ মূল তালিকা'). Null if not stated."
        ),
    },
    # triage -- model self-report, used to route pages to human review
    {
        "name": "template_match",
        "type": "boolean",
        "desc": (
            "True if the page is the standard 4-section info page. False for blank, "
            "rotated, truncated or otherwise anomalous pages."
        ),
    },
    {
        "name": "extraction_confidence",
        "type": CONFIDENCE_VALUES,
        "desc": (
            "Overall legibility of this page. LOW if the scan is faint, skewed or "
            "partly cut off, or if several values had to be guessed."
        ),
    },
    {
        "name": "anomaly_notes",
        "type": "string",
        "desc": (
            "Short English note about anything unusual -- handwritten corrections, "
            "struck-through values, illegible cells, unexpected extra text. Null if none."
        ),
    },
]

MODEL_FIELD_NAMES = [f["name"] for f in MODEL_FIELDS]

#: Where the row came from -- enough to find the exact source file again.
#:
#: ``pdf_sha256`` hashes the **source PDF**, not the rendered page image. The image is
#: derived, so hashing it would not notice the publisher re-issuing a part; hashing the
#: PDF both identifies the input and drives cache invalidation on a resumed run.
SOURCE_COLUMNS = [
    "source_zip",
    "source_zip_dir",
    "source_pdf",
    "pdf_sha256",
    "pdf_bytes",
]

#: Recovered from the source filename, which encodes them all. ``parse_source_filename``
#: has always returned these; earlier versions kept only ac_no/part_no and dropped the
#: rest, which are exactly the keys needed to stitch across roll types and revisions.
FILENAME_COLUMNS = [
    "roll_year",
    "state_code",
    "roll_type",
    "revision_no",
    "lang",
    "ac_no_file",
    "part_no_file",
]

#: The rendered page this row was read from.
RENDER_COLUMNS = [
    "page_png",
    "page_sha256",
]

#: Which code and which engine produced the row.
RUN_COLUMNS = [
    "engine",
    "engine_version",
    "pipeline_version",
    "extracted_at",
]

PROVENANCE_COLUMNS = SOURCE_COLUMNS + FILENAME_COLUMNS + RENDER_COLUMNS + RUN_COLUMNS

#: Columns derived in code from verbatim model output -- deterministic, no model call.
DERIVED_COLUMNS = [
    "ward_no_num",
    "ps_address_digits",
]

#: QA columns written by ``validate.py``.
QA_COLUMNS = [
    "checks_passed",
    # The denominator varies by row: corpus-consensus checks only run once a mode
    # exists, so checks_passed is uninterpretable without it.
    "checks_total",
    "flags",
    "needs_review",
    "anomaly_notes",
]


def _ordered_unique(*groups: List[str]) -> List[str]:
    """Concatenate column groups, keeping the first occurrence of each name.

    ``anomaly_notes`` deliberately appears in two groups: it is a field the model can
    report *and* one ``validate.py`` writes when a check fires. That is one output
    column with two possible producers, so it is listed in both groups for honesty and
    collapsed here rather than deleted from either.
    """
    seen, columns = set(), []
    for group in groups:
        for name in group:
            if name not in seen:
                seen.add(name)
                columns.append(name)
    return columns


PART_COLUMNS = _ordered_unique(PROVENANCE_COLUMNS, MODEL_FIELD_NAMES, DERIVED_COLUMNS, QA_COLUMNS)

#: Fields the dictionary canonicalises; each gains a ``*_canonical`` companion in the
#: shipped output. Raw values are always kept -- canonical is "made internally
#: consistent", not "corrected", and conflating the two would overstate its authority.
CANONICAL_SOURCE_FIELDS = (
    "district",
    "ac_name",
    "pc_name",
    "revenue_circle",
    "block",
    "police_station",
    "post_office",
)
CANONICAL_COLUMNS = [f"{name}_canonical" for name in CANONICAL_SOURCE_FIELDS]

#: Columns the Tesseract pipeline cannot populate, excluded from the shipped output.
#: The 11 romanisations were produced by the (now dormant) model path, and
#: ``extraction_confidence`` is a model self-report. They stay in ``MODEL_FIELDS`` so
#: re-enabling that path needs no schema migration, but shipping them empty would leave
#: a quarter of every row blank.
UNSUPPORTED_BY_OCR = tuple(name for name in MODEL_FIELD_NAMES if name.endswith("_roman")) + (
    "extraction_confidence",
)


def output_columns() -> List[str]:
    """Columns for the shipped dataset: everything the OCR pipeline actually fills."""
    return [c for c in PART_COLUMNS if c not in UNSUPPORTED_BY_OCR] + CANONICAL_COLUMNS


OUTPUT_COLUMNS = output_columns()

SECTION_COLUMNS = [
    "ac_no",
    "part_no",
    "section_no",
    "section_name",
    "section_name_digits",
]


def derive_columns(row: Dict[str, Any]) -> Dict[str, Any]:
    """Compute ``DERIVED_COLUMNS`` from a part row's verbatim fields.

    Kept deterministic on purpose: digit transliteration is a bijection, so doing it in
    code avoids both the token cost and the failure mode of asking the model for it.
    """
    ward = row.get("ward_no")
    address = row.get("ps_address")
    return {
        # A derived field inherits its source's kind of emptiness: blank in, blank out;
        # unread in, unread out. Reporting "" as NA would claim a read failure that did
        # not happen.
        "ward_no_num": None if ward is None else (first_int(ward) if ward else ""),
        "ps_address_digits": None if address is None else normalize_digits(address),
    }


# ------------------------------------------------------------------------- JSON Schema


def _property_schema(field: Dict[str, Any]) -> Dict[str, Any]:
    """Build one JSON Schema property, nullable so the model can report an empty cell.

    Nullability uses ``anyOf`` rather than a type union because structured outputs
    validates against the documented subset, which supports ``anyOf`` and ``enum``
    but not numeric or length constraints.
    """
    spec = field["type"]
    if isinstance(spec, list):
        base: Dict[str, Any] = {"type": "string", "enum": spec}
    else:
        base = {"type": spec}
    return {
        "anyOf": [base, {"type": "null"}],
        "description": field["desc"],
    }


def build_page1_json_schema() -> Dict[str, Any]:
    """JSON Schema for one page-1 extraction, for ``output_config.format``."""
    properties = {f["name"]: _property_schema(f) for f in MODEL_FIELDS}
    properties["sections"] = {
        "type": "array",
        "description": (
            "The numbered area list under 'খণ্ডৰ অন্তৰ্গত এলেকা সমূহৰ নম্বৰ আৰু নাম'. "
            "One entry per numbered line, in page order."
        ),
        "items": {
            "type": "object",
            "properties": {
                "section_no": {
                    "anyOf": [{"type": "integer"}, {"type": "null"}],
                    "description": "Leading number of the area line.",
                },
                "section_name": {
                    "type": "string",
                    "description": "Area name, verbatim Assamese.",
                },
                "section_name_roman": {
                    "type": "string",
                    "description": "Romanization of section_name.",
                },
            },
            "required": ["section_no", "section_name", "section_name_roman"],
            "additionalProperties": False,
        },
    }
    return {
        "type": "object",
        "properties": properties,
        "required": MODEL_FIELD_NAMES + ["sections"],
        "additionalProperties": False,
    }


PAGE1_JSON_SCHEMA = build_page1_json_schema()


#: Bumped when a change alters extracted values, so rows can be traced to the code that
#: produced them and a partially-rebuilt corpus is never silently mixed.
#:
#: 2.0.0 -- multi-language support. Each page is read with the Tesseract model and label
#: table of the language its filename names; the grid is anchored on per-language upper
#: rules with the lower ones located structurally; ``gram_panchayat`` and ``subdivision``
#: join the schema. Parsing changed for every language, so 1.x cache entries are stale.
#:
#: 2.1.0 -- a digit cell holding a single glyph is retried as a single character.
#: 2.2.0 -- that retry now varies the *scale*, which is what actually decides a lone
#: digit. 2.1.0 fixed the lone "0" but left 6.6% of Bengali parts with no start serial.
#: 2.3.0 -- part_no is read with the digit engine instead of being pulled out of the
#: language model's transcription of the whole cell. 98.15% -> 100.00%.
PIPELINE_VERSION = "2.3.0"


def empty_part_row() -> Dict[str, Any]:
    """A part row with every column present and ``None``.

    ``None`` is the correct default because it means *not read*. Fields the form leaves
    genuinely blank are set to ``""`` explicitly by the parser, which is what keeps the
    two distinguishable downstream -- an unreadable pincode and an absent ward number
    are different facts and should not both arrive as empty strings.
    """
    return {column: None for column in PART_COLUMNS}
