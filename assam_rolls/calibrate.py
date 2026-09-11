"""Derive a language profile by reading the form's own printed labels.

The Bengali and English constituencies use the same form as the Assamese ones, but its
labels are printed in their own language, and the row-alignment fallback matches against
those labels on most pages. They therefore have to be known exactly.

Typing them from memory would be guessing. Instead they are **recovered from the corpus**,
which is possible because a label is *printed* rather than filled in: it is byte-identical
on every page of that language. So OCR the label column across a sample of pages and take
the modal reading per row. Errors that are random across pages are outvoted; the agreement
fraction says how much to trust each result.

What consensus cannot fix is a *systematic* misreading -- if the model mangles a glyph the
same way on every page, the mode is mangled too. That is why the agreement figure is
reported rather than assumed, and why the per-language accuracy in ``validate`` is the
real backstop: a label table that is wrong in a way consensus cannot see shows up there as
a language whose numbers are worse than its neighbours'.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image

from .languages import TESSERACT_LANG, LanguageProfile
from .layout import (
    LOWER_RULE_COUNT,
    RULE_TOLERANCE,
    S4_RULE_COUNT,
    Grid,
    LayoutError,
    build_grid,
    detect_h_rules,
)
from .log import get_logger
from .ocr import Engine, get_engine
from .parse import (
    COLON_LIKE,
)
from .parse import LOCALITY_FIELDS as CORE_LOCALITY_FIELDS
from .parse import MIN_LABEL_GAP, REVISION_FIELDS, text_rows
from .schema import clean_text

logger = get_logger(__name__)

#: Below this, a derived label is reported as weak. Consensus on a printed label should be
#: near-total; anything much under this means the model is struggling with the glyphs
#: rather than that the pages disagree.
MIN_AGREEMENT = 0.80

#: The locality rows each edition prints, keyed by (language, row count). Confirmed by
#: looking at the rendered pages of all three languages rather than by trusting OCR of the
#: labels -- which is the one thing calibration cannot check itself.
LOCALITY_VARIANTS: Dict[Tuple[str, int], Tuple[str, ...]] = {
    ("BEN", 9): (
        "main_town_village",
        "ward_no",
        "post_office",
        "police_station",
        "gram_panchayat",
        "block",
        "revenue_circle",
        "district",
        "pin_code",
    ),
    ("ENG", 9): (
        "main_town_village",
        "ward_no",
        "post_office",
        "police_station",
        "block",
        "revenue_circle",  # printed as "Tehsil"; the same slot and the same field
        "subdivision",
        "district",
        "pin_code",
    ),
}

#: Surface forms that map onto the controlled vocabulary, per language.
#:
#: These are *values*, not printed labels, so consensus across pages cannot recover them
#: the way it recovers labels -- they legitimately differ page to page. They were instead
#: read off the corpus directly. Reservation is a property of the constituency, so
#: sampling every Bengali AC and the single English one saw every value that exists:
#: 11 Bengali ACs are সাধারণ and 2 are তপশীলি জাতি; none is ST. Entries marked "unobserved"
#: follow the construction of the forms that *were* seen and are there so a reissued roll
#: does not silently produce a blank.
#:
#: An unrecognised value yields ``""`` rather than a guess, which keeps a misread from
#: being filed as a real category.
VOCAB: Dict[str, Dict[str, Dict[str, str]]] = {
    "BEN": {
        "reservation": {
            "সাধারণ": "GENERAL",
            "তপশীলি জাতি": "SC",
            "তপশীলি জনজাতি": "ST",  # unobserved: no Bengali AC is ST-reserved
        },
        "ps_type": {
            "সাধারণ": "GENERAL",
            "সাধারন": "GENERAL",  # unnormalised spelling, as in Assamese
            "পুরুষ": "MALE",  # unobserved in the sample
            "মহিলা": "FEMALE",  # unobserved in the sample
        },
    },
    "ENG": {
        "reservation": {
            "GENERAL": "GENERAL",
            "General": "GENERAL",
            "SC": "SC",
            "ST": "ST",
        },
        "ps_type": {
            "GENERAL": "GENERAL",
            "General": "GENERAL",
            "MALE": "MALE",
            "Male": "MALE",
            "FEMALE": "FEMALE",
            "Female": "FEMALE",
        },
    },
}

#: How far right of the cell edge to look for a label. Wide enough to hold the longest
#: label in any of the three languages, narrow enough to exclude the value.
LABEL_SEARCH_X = 420

#: How far short of the printed colon to stop the label crop.
COLON_MARGIN = 8

#: Whitespace, in px, that separates the printed colon from the value after it.
COLON_TO_VALUE_GAP = 4


@dataclass
class RowConsensus:
    """What a sample of pages agreed the label on one row says."""

    index: int
    label: str
    agreement: float
    readings: int
    value_x: Optional[int]

    @property
    def is_weak(self) -> bool:
        return self.agreement < MIN_AGREEMENT


def _after_colon(columns: Sequence[int], colon_x: int) -> Optional[int]:
    """Where the value begins: past the colon's ink and the whitespace following it.

    ``columns`` are the ink column positions of one row, ``colon_x`` the first of them
    after the label. Returns ``None`` when nothing follows the colon, which is what an
    empty value looks like.
    """
    after = [c for c in columns if c >= colon_x]
    for a, b in zip(after, after[1:]):
        if b - a >= COLON_TO_VALUE_GAP:
            return b
    return None


def _label_and_gap(
    cell: Image.Image, span: Tuple[int, int], engine: Engine, ink: int = 150
) -> Tuple[str, Optional[int]]:
    """Read one row's label, and where the value after it begins.

    The gap is located **first**, and the label is then cropped to end where the gap
    begins. Reading a fixed-width strip instead pulls the value in with the label -- the
    revision rows came back as ``সংশোধনের বছর 2026``, label and value fused.

    The crop runs to the *end* of the gap, not its start. Stopping at the start makes the
    crop only as wide as the label itself, and a short one -- Bengali ``ব্লক`` is about
    45px -- is then too small a strip for Tesseract to read, which returned ``a``. Running
    to the end of the gap costs nothing (the gap is whitespace) and leaves the recogniser
    some room. The printed colon lands at that boundary, so it is split off by text rather
    than by pixels, which is more reliable than either edge.
    """
    pixels = cell.convert("L").load()
    columns = [
        x for x in range(cell.width) if any(pixels[x, y] < ink for y in range(span[0], span[1] + 1))
    ]
    value_x, label_right = None, LABEL_SEARCH_X
    if len(columns) >= 2:
        width, start = max(((b - a, a) for a, b in zip(columns, columns[1:])), default=(0, 0))
        if width >= MIN_LABEL_GAP:
            colon_x = start + width
            # Stop a few pixels short of the colon. Including it adds a glyph Tesseract
            # transcribes inconsistently -- as H, 1, ( or ] depending on the row -- which
            # then differs across pages and destroys the consensus. The gap is ~95-131px,
            # so backing off still leaves a wide, readable crop.
            label_right = max(start + 20, colon_x - COLON_MARGIN)
            # The value starts after the colon, found by walking past the colon's ink to
            # the next whitespace. A fixed offset cannot do this: too small and the colon
            # lands in the value as a stray "'" or ">", too large and it shears the first
            # letter off (HAFLONG became 4~AFLONG).
            value_x = _after_colon(columns, colon_x)
    top, bottom = max(0, span[0] - 4), span[1] + 5
    raw = engine.read_text(cell.crop((0, top, min(cell.width, label_right), bottom)))
    label = clean_text(raw).split(":", 1)[0].rstrip(COLON_LIKE + " ").strip()
    # A lone trailing character is punctuation the recogniser turned into a letter, never
    # part of a real label.
    parts = label.split()
    if len(parts) > 1 and len(parts[-1]) == 1:
        label = " ".join(parts[:-1])
    return label, value_x


def _consensus(readings: Sequence[Tuple[str, Optional[int]]], index: int) -> RowConsensus:
    """The modal label across pages, with its agreement fraction."""
    labels = [text for text, _ in readings if text]
    if not labels:
        return RowConsensus(index, "", 0.0, 0, None)
    label, count = Counter(labels).most_common(1)[0]
    offsets = [x for text, x in readings if text == label and x is not None]
    return RowConsensus(
        index=index,
        label=label,
        agreement=count / len(labels),
        readings=len(labels),
        value_x=int(median(offsets)) if offsets else None,
    )


def _block_rows(
    cell: Image.Image, engine: Engine, expected: int
) -> Optional[List[Tuple[str, Optional[int]]]]:
    """Read every row of a stacked block, or ``None`` if the row count is unexpected.

    A page whose block has the wrong number of rows is skipped rather than aligned:
    during calibration there is no label table yet to align *with*, so a short block
    would shift the sample's rows against each other and corrupt the consensus.

    This is why calibration needs a generous sample. Most parts are rural and print no
    ward number, so their locality block has seven rows rather than eight and cannot be
    used here; only the urban minority contributes to the locality labels.
    """
    spans = text_rows(cell, right=LABEL_SEARCH_X)
    if len(spans) != expected:
        return None
    return [_label_and_gap(cell, span, engine) for span in spans]


@dataclass
class Calibration:
    """The outcome of calibrating one language."""

    code: str
    pages_attempted: int
    grids_built: int
    stable_h_rules: Tuple[int, ...]
    locality: List[RowConsensus]
    revision: List[RowConsensus]
    address_label: str
    address_agreement: float
    locality_value_x: int
    revision_value_x: int
    reservation_values: Counter
    ps_type_values: Counter

    @property
    def weak_rows(self) -> List[RowConsensus]:
        return [row for row in self.locality + self.revision if row.is_weak]

    @property
    def min_agreement(self) -> float:
        rows = self.locality + self.revision
        return min([row.agreement for row in rows] + [self.address_agreement], default=0.0)

    @property
    def unmapped_values(self) -> Dict[str, List[str]]:
        """Observed categorical values the vocabulary does not cover.

        Surfaced rather than swallowed: an unmapped value reads as blank, and a blank that
        nobody was told about looks identical to a field the form does not print.
        """
        vocab = VOCAB.get(self.code, {})
        found = {}
        for name, observed in (
            ("reservation", self.reservation_values),
            ("ps_type", self.ps_type_values),
        ):
            missing = [value for value in observed if value not in vocab.get(name, {})]
            if missing:
                found[name] = missing
        return found

    @property
    def locality_fields(self) -> Tuple[str, ...]:
        """Which schema field each detected locality row corresponds to.

        The row *count* identifies the edition, which is enough because the difference
        between editions is a single inserted row at a known place -- verified against the
        rendered pages of all three languages:

        * 8 rows: the core set (Assamese);
        * 9 rows starting after the police station: Bengali, which adds ``gram_panchayat``;
        * 9 rows with the extra one after the revenue circle: English, which adds
          ``subdivision`` (and labels the revenue circle "Tehsil").

        Anything else is refused rather than guessed at, because a wrong field order here
        misfiles every value below the mistake instead of failing visibly.
        """
        count = len(self.locality)
        if count == len(CORE_LOCALITY_FIELDS):
            return CORE_LOCALITY_FIELDS
        variant = LOCALITY_VARIANTS.get((self.code, count))
        if variant is None:
            raise ValueError(
                f"{self.code}: {count} locality rows does not match any known edition "
                f"({sorted({n for _, n in LOCALITY_VARIANTS})} or "
                f"{len(CORE_LOCALITY_FIELDS)}); the form has changed and the field order "
                f"must be confirmed against a rendered page before extracting"
            )
        return variant

    def to_profile(self) -> LanguageProfile:
        return LanguageProfile(
            code=self.code,
            tesseract_lang=TESSERACT_LANG[self.code],
            stable_h_rules=self.stable_h_rules,
            locality_fields=self.locality_fields,
            locality_labels=tuple(row.label for row in self.locality),
            revision_labels=tuple(row.label for row in self.revision),
            address_label=self.address_label,
            locality_value_x=self.locality_value_x,
            revision_value_x=self.revision_value_x,
            reservation_map=dict(VOCAB.get(self.code, {}).get("reservation", {})),
            ps_type_map=dict(VOCAB.get(self.code, {}).get("ps_type", {})),
            derived_from_pages=self.grids_built,
            min_label_agreement=round(self.min_agreement, 4),
        )

    def to_record(self) -> Dict[str, Any]:
        """The full audit trail, written beside the profile."""
        return {
            "code": self.code,
            "pages_attempted": self.pages_attempted,
            "grids_built": self.grids_built,
            "stable_h_rules": list(self.stable_h_rules),
            "locality": [
                {
                    "field": self.locality_fields[row.index],
                    "label": row.label,
                    "agreement": round(row.agreement, 4),
                    "readings": row.readings,
                    "value_x": row.value_x,
                    "weak": row.is_weak,
                }
                for row in self.locality
            ],
            "revision": [
                {
                    "field": REVISION_FIELDS[row.index],
                    "label": row.label,
                    "agreement": round(row.agreement, 4),
                    "readings": row.readings,
                    "value_x": row.value_x,
                    "weak": row.is_weak,
                }
                for row in self.revision
            ],
            "address_label": self.address_label,
            "address_agreement": round(self.address_agreement, 4),
            "locality_value_x": self.locality_value_x,
            "revision_value_x": self.revision_value_x,
            "observed_reservation_values": dict(self.reservation_values.most_common()),
            "observed_ps_type_values": dict(self.ps_type_values.most_common()),
            "unmapped_values": self.unmapped_values,
            "min_agreement": round(self.min_agreement, 4),
            "weak_rows": [self.locality_fields[r.index] for r in self.locality if r.is_weak]
            + [REVISION_FIELDS[r.index] for r in self.revision if r.is_weak],
        }


def _address_label(grid: Grid, image: Image.Image, engine: Engine) -> str:
    """The station-address label, which sits on its own row inside the station cell.

    The cell carries **two** labels: the station's number-and-name label, then the name,
    then the address label, then the address. Taking the first labelled row returns the
    wrong one -- it produced "No. and Name of Polling Station" as the English address
    label, which then matched the name row and left every ``ps_name`` empty. The address
    label is the second labelled row; the name between them does not carry a colon.
    """
    cell = grid.crop(image, "s3_address")
    labelled = []
    for span in text_rows(cell):
        line = clean_text(
            engine.read_text(cell.crop((0, max(0, span[0] - 4), cell.width, span[1] + 5)))
        )
        if ":" in line:
            labelled.append(clean_text(line.split(":")[0]))
    return labelled[1] if len(labelled) > 1 else (labelled[0] if labelled else "")


def derive_stable_rules(images: Sequence[Image.Image]) -> Tuple[int, ...]:
    """The upper rules this language's form prints on *every* sample page.

    Taken as an intersection rather than a mode, which is what drops rules that appear
    only sometimes -- Assamese has one such rule at y=171, and requiring it would reject
    the pages that lack it.

    Only the rules above the last eight are considered: the final five are the elector
    table and the three before them move with page content, so neither belongs in a
    fixed table.
    """
    per_page: List[List[int]] = []
    for image in images:
        rules = detect_h_rules(image)
        if len(rules) >= LOWER_RULE_COUNT + S4_RULE_COUNT + 4:
            per_page.append(rules[: -(LOWER_RULE_COUNT + S4_RULE_COUNT)])
    if not per_page:
        return ()

    stable = []
    for candidate in per_page[0]:
        if all(
            any(abs(candidate - rule) <= RULE_TOLERANCE for rule in page) for page in per_page[1:]
        ):
            stable.append(candidate)
    return tuple(stable)


def calibrate(
    code: str,
    images: Sequence[Image.Image],
    engine: Optional[Engine] = None,
) -> Calibration:
    """Derive a language profile from sample page images.

    ``images`` are rendered page-1 images from that language's constituencies; the caller
    decides how they are sampled.
    """
    engine = engine or get_engine("tesseract", lang=TESSERACT_LANG[code])

    # The rules have to be derived before any grid can be built, since they are what
    # identifies the template in the first place.
    stable_rules = derive_stable_rules(images)
    logger.info(
        "%s: derived %d stable upper rules: %s", code, len(stable_rules), list(stable_rules)
    )

    # How many rows this edition's locality block prints. Taken as the mode over the
    # sample rather than assumed, since it is exactly what differs between editions --
    # Assamese prints 8, Bengali and English 9. Rows are counted from the label column
    # only, so a wrapped value does not inflate the count.
    grids: List[Tuple[Any, Any]] = []
    row_counts: Counter = Counter()
    for image in images:
        try:
            grid = build_grid(image, stable_rules)
        except LayoutError as exc:
            logger.warning("%s: grid not found on a sample page: %s", code, exc)
            continue
        grids.append((image, grid))
        row_counts[len(text_rows(grid.crop(image, "s2_locality"), right=LABEL_SEARCH_X))] += 1

    locality_rows = row_counts.most_common(1)[0][0] if row_counts else len(CORE_LOCALITY_FIELDS)
    logger.info("%s: locality row counts across the sample: %s", code, dict(row_counts))

    locality_readings: List[List[Tuple[str, Optional[int]]]] = []
    revision_readings: List[List[Tuple[str, Optional[int]]]] = []
    address_labels: List[str] = []
    reservations: Counter = Counter()
    ps_types: Counter = Counter()
    grids_built = len(grids)

    for image, grid in grids:
        rows = _block_rows(grid.crop(image, "s2_locality"), engine, locality_rows)
        if rows:
            locality_readings.append(rows)
        rows = _block_rows(grid.crop(image, "s1_revision"), engine, len(REVISION_FIELDS))
        if rows:
            revision_readings.append(rows)

        label = _address_label(grid, image, engine)
        if label:
            address_labels.append(label)

        ps_type = clean_text(engine.read_text(grid.crop(image, "s3_type_value")))
        if ps_type:
            ps_types[ps_type] += 1
        header = clean_text(engine.read_text(grid.crop(image, "header_ac")))
        if "(" in header and ")" in header:
            reservations[clean_text(header[header.rfind("(") + 1 : header.rfind(")")])] += 1

    def rows_for(readings, count):
        return [
            (
                _consensus([page[i] for page in readings], i)
                if readings
                else RowConsensus(i, "", 0.0, 0, None)
            )
            for i in range(count)
        ]

    locality = rows_for(locality_readings, locality_rows)
    revision = rows_for(revision_readings, len(REVISION_FIELDS))

    address_label, address_agreement = "", 0.0
    if address_labels:
        address_label, hits = Counter(address_labels).most_common(1)[0]
        address_agreement = hits / len(address_labels)

    def offset_for(rows: List[RowConsensus], fallback: int) -> int:
        found = [row.value_x for row in rows if row.value_x]
        return int(median(found)) if found else fallback

    return Calibration(
        code=code,
        pages_attempted=len(images),
        grids_built=grids_built,
        stable_h_rules=stable_rules,
        locality=locality,
        revision=revision,
        address_label=address_label,
        address_agreement=address_agreement,
        # Fall back to the Assamese offsets only if nothing could be measured, which
        # would itself mean the sample was unusable.
        locality_value_x=offset_for(locality, 320),
        revision_value_x=offset_for(revision, 213),
        reservation_values=reservations,
        ps_type_values=ps_types,
    )


def render_report(calibration: Calibration) -> str:
    """A human-readable summary, printed after calibration."""
    lines = [
        f"# {calibration.code}",
        f"grid found on {calibration.grids_built}/{calibration.pages_attempted} sample pages",
        f"stable upper rules: {list(calibration.stable_h_rules)}",
        "",
        f"{'field':<20} {'agree':>6}  label",
    ]
    for row in calibration.locality:
        mark = " (weak)" if row.is_weak else ""
        lines.append(
            f"{calibration.locality_fields[row.index]:<20} {row.agreement:>6.0%}  {row.label}{mark}"
        )
    for row in calibration.revision:
        mark = " (weak)" if row.is_weak else ""
        lines.append(f"{REVISION_FIELDS[row.index]:<20} {row.agreement:>6.0%}  {row.label}{mark}")
    lines += [
        f"{'address_label':<20} {calibration.address_agreement:>6.0%}  {calibration.address_label}",
        "",
        f"locality_value_x = {calibration.locality_value_x}",
        f"revision_value_x = {calibration.revision_value_x}",
        f"observed ps_type values     : {dict(calibration.ps_type_values.most_common(6))}",
        f"observed reservation values : {dict(calibration.reservation_values.most_common(6))}",
    ]
    unmapped = calibration.unmapped_values
    if unmapped:
        lines.append("")
        lines.append(f"UNMAPPED values (will read as blank): {unmapped}")
    if calibration.weak_rows:
        lines.append("")
        lines.append(
            f"WEAK: {len(calibration.weak_rows)} row(s) below {MIN_AGREEMENT:.0%} agreement"
        )
    return "\n".join(lines)


def write_record(out_dir: Path, calibration: Calibration) -> Path:
    import json

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{calibration.code}.json"
    path.write_text(
        json.dumps(calibration.to_record(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return path
