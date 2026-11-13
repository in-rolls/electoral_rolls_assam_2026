"""Reading a whole part through Cloud Vision.

The tesseract path in :mod:`electors.extract` is built around one page at a time, because that
is the unit tesseract charges for. Vision charges per **image submitted**, so the unit here is
several pages stacked into one tall PNG -- four at 400 dpi -- and a whole part is seven or
eight calls rather than nine hundred process spawns. That is the difference between $1,400 and
$368 for the state, and it is why this is a separate reader rather than an engine slotted into
the existing one.

**No tesseract anywhere in it.** The section header and the closing summary were the two places
the old path still needed a second engine, and Vision has already returned the words for both
by the time they are wanted -- the header sits above the first box, and the summary page is
just another page in the stack. Reading them from words that were already paid for removes the
engine rather than adding a fallback.
"""

from __future__ import annotations

import re
import string
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

from PIL import Image

from assam_rolls import languages, render, schema

from . import extract, grid, pages, repack
from . import summary as summary_page
from . import vision

#: One page of a stack, and everything Vision said about it.
PageWords = Tuple[int, Image.Image, List[vision.Word]]
PriorBoxReads = Dict[Tuple[int, int, int], Dict[str, Any]]


def _chunks(items: Sequence[Any], size: int) -> Iterator[Sequence[Any]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def words_per_page(
    images: Sequence[Image.Image],
    api_key: str,
    pages_per_image: Optional[int] = None,
    images_per_request: int = vision.IMAGES_PER_REQUEST,
) -> Tuple[List[List[vision.Word]], int]:
    """Every page's words, and the number of images billed.

    Pages are stacked, the stacks are batched, and the words are shifted back into each page's
    own coordinates on the way out. That last step is the one that would fail silently: a word
    on page two of a stack comes back in stack coordinates, and left unshifted it lands in
    whichever box happens to sit at that height on page one -- a plausible wrong answer on every
    page but the first.

    The stacking factor is measured from the pages unless it is given, because it depends on the
    render resolution and a constant that disagreed with it refused every part.
    """
    if not images:
        return [], 0
    if pages_per_image is None:
        widest = max(image.width for image in images)
        tallest = max(image.height for image in images)
        pages_per_image = vision.pages_that_fit(
            widest, tallest, bytes_per_page=vision.encoded_size(images[0])
        )
    stacks = [vision.stack_pages(group) for group in _chunks(images, pages_per_image)]
    for stack in stacks:
        problem = stack.check()
        if problem:
            raise ValueError(f"stack rejected before spending a request: {problem}")

    # Batched by payload, not by count. Sixteen images are legal individually and 96 MB
    # together, and the request ceiling is 40 MB -- which rejected every 400 dpi and 300 dpi
    # part while letting native through, and would have read as "the expensive arms do not
    # work" rather than "the batcher does not".
    encoded = [vision.encode(stack.image) for stack in stacks]
    groups = vision.batch_images(encoded)

    # Re-chunked rather than truncated. ``batch_images`` already caps at the module's own
    # per-request count, but a caller passing a smaller one would otherwise have the surplus
    # stacks silently dropped -- pages missing from the output with nothing raised.
    groups = [
        g[i : i + images_per_request] for g in groups for i in range(0, len(g), images_per_request)
    ]

    out: List[List[vision.Word]] = []
    billed = 0
    for group in groups:
        batch = [stacks[i] for i in group]
        responses = vision.annotate([stack.image for stack in batch], api_key=api_key)
        billed += len(batch)
        if len(responses) != len(batch):
            raise RuntimeError(f"asked for {len(batch)} images, got {len(responses)} responses")
        for stack, response in zip(batch, responses):
            words = vision.words_from(response)
            bounds = list(zip(stack.offsets, stack.offsets[1:] + [stack.image.height]))
            for top, bottom in bounds:
                page = [w.shifted(top) for w in words if top <= w.middle_y < bottom]
                out.append(page)
    return out[: len(images)], billed


def _header_text(
    words: Sequence[vision.Word], image: Image.Image, boxes: Sequence[grid.Box]
) -> str:
    """The band above the first box, which says which list the page belongs to."""
    top = min(boxes[0].top, int(image.height * 0.09))
    above = [w for w in words if w.bottom <= top]
    above.sort(key=lambda w: (w.top, w.left))
    return " ".join(w.text for w in above)


def _page_text(words: Sequence[vision.Word], image: Image.Image) -> str:
    """A page as lines, for the closing-total parser.

    Real lines, not one long string: ``summary.parse`` takes the last balancing triple *on a
    line*, because the row's prose runs ahead of its figures and a year can precede the real
    numbers. Flattened to a single line, a page with several rows would offer up the last row's
    figures for the main list.
    """
    return "\n".join(vision.grouped_lines(words, 0, image.width))


def _record_summary(
    result: extract.PartResult, words: Sequence[vision.Word], image: Image.Image
) -> None:
    """Keep a Vision closing total only when its three components balance."""
    if result.summary_total is not None:
        return
    found = summary_page.parse(_page_text(words, image))
    if not found:
        return
    male, female, third, total = found
    closing = summary_page.RollSummary(male=male, female=female, third=third, total=total, scale=1)
    if closing.balances:
        result.summary_male = male
        result.summary_female = female
        result.summary_third = third
        result.summary_total = total


def _box_header(
    image: Image.Image,
    box: grid.Box,
    key: Tuple[int, int, int],
    prior_box_reads: Optional[PriorBoxReads],
) -> Dict[str, Any]:
    """Reuse a header read only when both OCR crops describe the identical source pixels."""
    cached = (prior_box_reads or {}).get(key)
    if cached is not None and cached.get("header_geometry") == _header_geometry(image, box):
        return dict(cached)
    return _header_of(image, box)


def tiles_for(
    images: Sequence[Image.Image],
    lang: str,
    prior_box_reads: Optional[PriorBoxReads] = None,
) -> Tuple[List[Tuple[Image.Image, Tuple[int, int, int, int], Tuple[int, int, int]]], Dict]:
    """Every box's text lines as a tile, plus what the CPU needs to read for itself.

    Only the four labeled lines go to Vision. Three other things are read here on hardware
    already paid for, because none of them is worth a paid pixel:

    - the **EPIC** and the serial, which sit to the right of ``text_right`` in the photo column.
      Read from their own cells with tesseract ``eng``, the EPIC comes back well formed on 96.7%
      of boxes -- against 62.3% for the single whole-strip pass this replaces, which asked for a
      boxed serial and an EPIC as though they were one line.

      **The cost case for this was checked and is thinner than it reads.** Submitting the whole
      box instead would let Vision read all three, and the side-reads are 60% of stage one -- an
      hour of CPU a constituency, about $1.00. But a full-box tile is **1.71x** the area of a
      text-only one once the header strip is included, taking Vision from $1.32 a constituency to
      $2.25. The saving is $0.07, which is noise.

      **And it would buy no accuracy.** Read both ways over 90 boxes of AC10 part 29, tesseract
      with its charset constrained and Vision on the whole box each returned 100% well-formed
      EPICs and **agreed on every one**. The 4.77% malformed rate in the shipped data was
      tesseract being allowed to answer with characters an EPIC cannot contain, which is one word
      of configuration, not an architecture.

      What the change would still buy is wall clock -- stage one roughly halves -- and the same
      hours can be bought by adding machines at the same total cost, without a second vintage in
      the dataset.
    - the **section header**, which distinguishes the main roll from its supplements.
    - the **closing summary**, which is the number every completeness check is measured against.
    """
    from . import crops as crop_module

    entries: List[Tuple[Image.Image, Tuple[int, int, int, int], Tuple[int, int, int]]] = []
    side: Dict[str, Any] = {
        "pages": {},
        "unknown": [],
        "summary": None,
        "unread_summary": None,
        "header_reader_version": HEADER_READER_VERSION,
    }

    # Classify every page first, then reconsider the ones that came out as anything but elector
    # pages. A part's last page can hold a single elector, and one page's rules cannot tell that
    # box from a row of the closing summary's table -- but the part's other pages can, because
    # they draw their columns in the same places. Part 18's serial 661 was lost exactly here.
    signatures = pages.recover_partial(
        [pages.classify(image, n) for n, image in enumerate(images, start=1)]
    )

    current_section = pages.Section.MAIN
    for signature, image in zip(signatures, images):
        index = signature.number
        if signature.kind is pages.PageKind.UNKNOWN:
            side["unknown"].append(index)
            continue
        if signature.kind is pages.PageKind.SUMMARY and side["summary"] is None:
            side["summary"] = _summary_from(image, lang)
            # Keep the page tesseract could not read, so stage two can spend on it. The closing
            # row wraps across four lines because its description cell is tall, and the parser
            # wants a balancing triple on one line -- so about a fifth of parts end up with no
            # total, and therefore with no check on their extracted rows at all.
            #
            # Kept here rather than re-rendered later because this is the only moment the page
            # is in memory, and a page is a few hundred kilobytes against the several gigabytes
            # of composites this stage already writes.
            if side["summary"] is None:
                side["unread_summary"] = image.copy()
        if not signature.is_elector:
            continue
        boxes = grid.build(signature.h_rules, signature.v_rules, columns=signature.columns)
        if not boxes:
            side["unknown"].append(index)
            continue

        header = _header_strip_text(image, boxes, lang)
        if summary_page.is_summary(header):
            if side["summary"] is None:
                side["summary"] = _summary_from(image, lang)
                if side["summary"] is None:
                    side["unread_summary"] = image.copy()
            continue

        section, recognised = pages.section_of(header)
        section = pages.section_after(current_section, section)
        current_section = section
        side["pages"][index] = {
            "section": section,
            "recognised": recognised,
            "boxes": {},
        }
        for box in boxes:
            if not grid.has_ink(image, box):
                continue
            bands = grid.text_bands(image, box)
            if not bands:
                continue
            top = crop_module.band_window(bands, 0, box)[0]
            bottom = crop_module.band_window(bands, len(bands) - 1, box)[1]
            key = (index, box.row, box.column)
            entries.append((image, (box.left, top, box.text_right, bottom), key))
            side["pages"][index]["boxes"][key] = _box_header(image, box, key, prior_box_reads)
    return entries, side


#: The letters the roll uses to mark an entry that is no longer a live elector, defined by its
#: own legend: E dead, S shifted, R duplicate, M missing, Q ineligible. A closed set, so anything
#: else in that cell is a misread digit rather than a marking.
STATUS_CODES = "ESRMQ"

#: Compatibility key for cached serial, EPIC, and status reads. Geometry and source-byte checks
#: protect against changed inputs; this key protects against changed OCR or parsing code when the
#: same rectangles are deliberately re-read differently.
HEADER_READER_VERSION = "1.0.0"


#: The serial cell's own ruled borders, as fractions of the box. Measured off the rules rather
#: than guessed: at 400 dpi a 1004x415 box rules that cell at x 15..328, y 15..77. Two earlier
#: attempts at these numbers by eye read 0 of 30 serials.
SERIAL_CELL = (0.020, 0.042, 0.325, 0.182)

#: Where the EPIC sits, as fractions of the box, with the right edge extended past it.
#:
#: The extension is the point. The EPIC is printed hard against the box's right rule, so cropping
#: at the rule slices the last character -- ``HHK0001457`` came back as ``HHK000145/``. Forty
#: pixels of margin costs nothing and takes the field from 88.9% to 96.7%; the whole-strip read
#: this replaces managed 62.3%.
EPIC_REGION = (0.55, 0.0, 1.0, 0.19)
EPIC_PAD = 40


def _read(
    image: Image.Image,
    rect: Tuple[int, int, int, int],
    psm: int,
    whitelist: Optional[str] = None,
) -> str:
    from assam_rolls import ocr

    try:
        engine = ocr.get_engine("tesseract", lang="eng")
        return " ".join(engine._run(image.crop(rect), "eng", whitelist, 2, psm=psm).split())
    except Exception:
        return ""


#: The only characters an EPIC can contain: three uppercase letters and seven digits.
#:
#: Unconstrained, tesseract answers with characters the field cannot hold, and one of them costs a
#: digit. ``FXW1/56134`` is a ten-character read whose fifth character came back as ``/``; the
#: repair strips non-alphanumerics before matching, so the slash is deleted rather than treated as
#: the digit position it occupies, and a recoverable value becomes a nine-character one.
#:
#: Measured on one part: of 31 malformed EPICs, 28 are explained entirely by an interior artifact,
#: and in all 30 occurrences that artifact is ``/``. It is also why the defect tracks the box row
#: -- a rule intrudes into the strip at some row positions and not others, giving 0.0% malformed on
#: rows 2, 6 and 7 against 7.5-9.4% on row 9, near-identically across eight constituencies.
#:
#: The serial cell beside it has been whitelisted to digits since it was written. The EPIC was not,
#: which is the whole defect: ~250,000 rows of the state.
EPIC_ALPHABET = string.ascii_uppercase + string.digits

#: A serial number is an elector's position within a part, and no part of this roll holds more
#: than about twelve hundred electors. A longer number is not a serial read badly; it is the OCR
#: of a cell that held something else.
MAX_SERIAL = 9999


def plausible_serial(text: str) -> Optional[int]:
    """The serial the cell claims, or None where the number cannot be one.

    Accepted unbounded, it reached the Parquet schema's int32 and took a whole constituency with
    it. AC24 part 95 held a box whose name was the printed placeholder ``ফটো উপলব্ধ`` -- "photo
    available", not a person -- and whose serial cell read 3,801,114,767. One value, and the write
    of all 170,726 rows raised ``ArrowInvalid`` and the constituency was re-claimed and re-failed
    by every machine that picked it up.

    None rather than a truncation. Keeping the first four digits would have turned that read into
    a confident 3801, and this field exists only to be an independent check on the derived row
    order -- a check that invents a plausible answer is worse than one that abstains.
    """
    digits = text.strip()
    if not digits.isdigit():
        return None
    value = int(digits)
    return value if 0 < value <= MAX_SERIAL else None


def _header_geometry(image: Image.Image, box: grid.Box) -> List[List[int]]:
    """The exact serial/status and EPIC rectangles consumed by Tesseract."""
    width = box.right - box.left
    height = box.bottom - box.top
    serial_left, serial_top, serial_right, serial_bottom = SERIAL_CELL
    epic_left, epic_top, epic_right, epic_bottom = EPIC_REGION
    return [
        [
            box.left + int(width * serial_left),
            box.top + int(height * serial_top),
            box.left + int(width * serial_right),
            box.top + int(height * serial_bottom),
        ],
        [
            box.left + int(width * epic_left),
            box.top + int(height * epic_top),
            min(image.width, box.left + int(width * epic_right) + EPIC_PAD),
            box.top + int(height * epic_bottom),
        ],
    ]


def _header_of(image: Image.Image, box: grid.Box) -> Dict[str, Any]:
    """The serial, the EPIC and the status code, each read from its own cell.

    Read with the English model: ``asm`` renders ``HHK0001471`` as ``1414140001471``, being
    trained on a script with no Latin letters and mapping them onto lookalike digits.

    Three crops rather than one. The old version OCR'd the whole strip in a single pass and got
    62.3% well-formed EPICs over 14,486 boxes -- the boxed serial and the EPIC are different
    kinds of thing sitting in different cells, and asking for both at once as a single line lost
    a fifth of them entirely.

    The **status code** is a letter the roll prints inside the serial cell, to the left of the
    number: ``E   15`` where a live entry shows ``15``. The roll defines them itself, in a legend
    at the foot of every closing page::

        E-মৃত, S-স্থানান্তৰিত/ বাসস্থান পৰিবৰ্তন, R-প্ৰতিলিপি, M-নিখোজ, Q-অযোগ্য

    dead, shifted, duplicate, missing, ineligible. All five mean the same thing for counting
    purposes -- not a live elector -- and nothing outside that set is a code. Accepting any
    letter read 5,167 across AC1 against 3,816 implied by the roll's own arithmetic, 34% over;
    restricting to the five leaves 3,826, which is 0.3% over. The 1,341 rejected were #, A, H
    and D: mangled digits, not markings.

    This is the signal because the watermark is not. Vision returns five Latin fragments from a
    whole 67 MP composite -- 'D', 'ED', 'RIZED' -- so reading the stamp itself finds nothing.
    """
    geometry = _header_geometry(image, box)
    cell = _read(image, tuple(geometry[0]), psm=7)
    epic = _read(
        image,
        tuple(geometry[1]),
        psm=7,
        whitelist=EPIC_ALPHABET,
    ).replace(" ", "")

    # A code only counts when it is a lone letter *followed by* the serial. Anything else in this
    # cell is a mangled digit, and treating those as codes would mark live electors dead.
    found = re.match(rf"^\s*([{STATUS_CODES}])[\s.]+\d", cell)
    return {
        "serial": re.sub(r"\D", "", cell),
        "epic": epic,
        "code": found.group(1).upper() if found else "",
        "header_geometry": geometry,
    }


def _header_strip_text(image: Image.Image, boxes: Sequence[grid.Box], lang: str) -> str:
    top = min(boxes[0].top, int(image.height * 0.09))
    if top <= 0:
        return ""
    from assam_rolls import ocr

    try:
        tesseract_lang = languages.profile_for(lang).tesseract_lang
        engine = ocr.get_engine("tesseract", lang=tesseract_lang)
        return engine._run(image.crop((0, 0, image.width, top)), tesseract_lang, None, 1, psm=6)
    except Exception:
        return ""


def _summary_from(image: Image.Image, lang: str):
    """The closing totals, read on the CPU and only kept if the arithmetic balances."""
    from assam_rolls import ocr

    try:
        tesseract_lang = languages.profile_for(lang).tesseract_lang
        engine = ocr.get_engine("tesseract", lang=tesseract_lang)
        return summary_page.read(engine, image, tesseract_lang)
    except Exception:
        return None


def rasterize(dpi: int) -> Callable[[bytes, Path], List[Image.Image]]:
    """A renderer that rasterizes every page at ``dpi``."""

    def renderer(pdf_bytes: bytes, workdir: Path) -> List[Image.Image]:
        paths = extract._render_pages(pdf_bytes, workdir, dpi=dpi)
        return [Image.open(path).convert("L") for path in paths]

    return renderer


def read_part(
    zip_path: Path,
    pdf_name: str,
    api_key: str,
    pages_per_image: Optional[int] = None,
    renderer: Optional[Callable[[bytes, Path], List[Image.Image]]] = None,
) -> extract.PartResult:
    """Extract every elector from one part PDF, using Cloud Vision for all of it.

    ``renderer`` decides what pixels Vision is shown. It is pluggable because the source PDFs
    are 144 dpi scans -- rendering at 400 upsamples them -- and what resolution to submit is a
    measured question rather than a settled one. Everything after the renderer is identical
    across resolutions, which is what makes the arms comparable.
    """
    meta = schema.parse_source_filename(pdf_name) or {}
    pdf_bytes = render.read_pdf_bytes(zip_path, pdf_name)
    result = extract.PartResult(
        ac_no=meta.get("ac_no", 0),
        part_no=meta.get("part_no", 0),
        lang=meta.get("lang", ""),
        source_zip=zip_path.name,
        source_pdf=pdf_name,
        pdf_sha256=render.sha256_bytes(pdf_bytes),
    )

    try:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            images = (renderer or rasterize(extract.DPI))(pdf_bytes, workdir)
            result.page_count = len(images)
            per_page, result.images_billed = words_per_page(
                images, api_key=api_key, pages_per_image=pages_per_image
            )
            _fill(result, images, per_page, meta)
            for image in images:
                image.close()
    except (subprocess.CalledProcessError, OSError, RuntimeError, ValueError) as exc:
        result.error = f"{type(exc).__name__}: {exc}"
    return result


def read_repacked(
    zip_path: Path,
    pdf_name: str,
    api_key: str,
    renderer: Optional[Callable[[bytes, Path], List[Image.Image]]] = None,
) -> extract.PartResult:
    """One part, submitting only the text lines and nothing else.

    Same engine, same pixels of text, same parser as :func:`read_part` -- only the packing
    differs, which is why the two must agree almost exactly and any real gap is a mapping bug.
    """
    meta = schema.parse_source_filename(pdf_name) or {}
    pdf_bytes = render.read_pdf_bytes(zip_path, pdf_name)
    result = extract.PartResult(
        ac_no=meta.get("ac_no", 0),
        part_no=meta.get("part_no", 0),
        lang=meta.get("lang", ""),
        source_zip=zip_path.name,
        source_pdf=pdf_name,
        pdf_sha256=render.sha256_bytes(pdf_bytes),
    )
    try:
        with tempfile.TemporaryDirectory() as tmp:
            images = (renderer or rasterize(extract.DPI))(pdf_bytes, Path(tmp))
            result.page_count = len(images)
            entries, side = tiles_for(images, result.lang)
            result.unknown_pages = side["unknown"]
            if side["summary"] is not None:
                closing = side["summary"]
                result.summary_male = closing.male
                result.summary_female = closing.female
                result.summary_third = closing.third
                result.summary_total = closing.total

            words: Dict[Tuple[int, int, int], List[vision.Word]] = {}
            budget = int(vision.MAX_PIXELS * vision.PIXEL_MARGIN)
            for batch in repack.batches(entries, budget):
                words.update(_read_composite(batch, api_key))
                result.images_billed += 1

            _fill_repacked(result, side, words, meta)
            for image in images:
                image.close()
    except (subprocess.CalledProcessError, OSError, RuntimeError, ValueError) as exc:
        result.error = f"{type(exc).__name__}: {exc}"
    return result


def _read_composite(batch: Sequence, api_key: str) -> Dict[Tuple[int, int, int], List]:
    composite, placed = repack.compose(batch)
    responses = vision.annotate([composite], api_key=api_key)
    if not responses:
        raise RuntimeError("no response for a composite")
    return repack.words_for(placed, vision.words_from(responses[0]))


def _fill_repacked(
    result: extract.PartResult,
    side: Dict[str, Any],
    words: Dict[Tuple[int, int, int], List],
    meta: Dict[str, Any],
) -> None:
    """Rows from the tiles' words, in page order, with the serial counted the same way."""
    serial = 0
    current_section = pages.Section.MAIN
    for index in sorted(side["pages"]):
        page = side["pages"][index]
        result.elector_pages += 1
        if not page["recognised"]:
            result.unrecognised_headers.append(index)
        section = pages.section_after(current_section, page["section"])
        current_section = section
        if section is not pages.Section.MAIN:
            result.supplement_pages.append(index)

        for key in sorted(page["boxes"]):
            found = words.get(key, [])
            if not found:
                continue
            width = max((w.right for w in found), default=0) + 1
            # Stage one sends only the body column to Vision; the header strip is read
            # separately and never enters the composite. Splitting on the first
            # Bengali-Assamese line here discarded every line of AC113's English boxes.
            body = vision.grouped_lines(found, 0, width)
            # The header strip never went to Vision -- the EPIC is outside the text column, and
            # tesseract's English model reads it for nothing on hardware already paid for.
            side_read = page["boxes"][key]
            elector = vision.elector_from([side_read["epic"]], body)
            # The roll's own markings, read from their own cells rather than inferred from the
            # words: a status code means this entry is no longer a live elector.
            elector.status_code = side_read.get("code", "")
            elector.deleted = bool(elector.status_code)
            elector.serial_no_ocr = plausible_serial(side_read.get("serial", ""))
            if elector.is_empty:
                elector.flags.append("unreadable")
            serial += 1
            elector.serial_no = serial
            box = grid.Box(
                row=key[1],
                column=key[2],
                left=0,
                top=0,
                right=0,
                bottom=0,
                text_right=0,
            )
            result.electors.append(
                extract._row(
                    result,
                    elector,
                    page=index,
                    box=box,
                    meta=meta,
                    section=section,
                    engine=extract.MIXED_ENGINE,
                )
            )


def _fill(
    result: extract.PartResult,
    images: Sequence[Image.Image],
    per_page: Sequence[Sequence[vision.Word]],
    meta: Dict[str, Any],
) -> None:
    """Turn each page's words into rows, in page order."""
    serial = 0
    current_section = pages.Section.MAIN

    for index, (image, words) in enumerate(zip(images, per_page), start=1):
        signature = pages.classify(image, index)
        if signature.kind is pages.PageKind.UNKNOWN:
            result.unknown_pages.append(index)
            continue
        if signature.kind is pages.PageKind.SUMMARY:
            _record_summary(result, words, image)
        if not signature.is_elector:
            continue

        boxes = grid.build(signature.h_rules, signature.v_rules)
        if not boxes:
            # Classified as an elector page but its geometry did not resolve. Recorded rather
            # than parsed: guessing thirty box positions here would produce thirty plausible
            # rows of nothing.
            result.unknown_pages.append(index)
            continue
        header_text = _header_text(words, image, boxes)
        if summary_page.is_summary(header_text):
            _record_summary(result, words, image)
            continue
        result.elector_pages += 1

        section, recognised = pages.section_of(header_text)
        section = pages.section_after(current_section, section)
        current_section = section
        if not recognised:
            result.unrecognised_headers.append(index)
        if section is not pages.Section.MAIN:
            result.supplement_pages.append(index)
        for box in boxes:
            inside = vision.words_within(words, box.left, box.top, box.text_right, box.bottom)
            header, body = vision.header_and_body(inside, box.left, box.text_right)
            if not header and not body:
                # Vision found no words in the box at all. A blank box at the end of a part is
                # the publisher's doing, and there is nothing in it to read.
                continue
            elector = vision.elector_from(header, body)
            if elector.is_empty:
                # Words but nothing readable. Emitted anyway: the row exists in the source, and
                # dropping it makes the count agree with nothing.
                elector.flags.append("unreadable")
            serial += 1
            elector.serial_no = serial
            result.electors.append(
                extract._row(
                    result,
                    elector,
                    page=index,
                    box=box,
                    meta=meta,
                    section=section,
                    engine=extract.VISION_ENGINE,
                )
            )


def parts_cost(page_counts: Sequence[int], pages_per_image: int = vision.PAGES_PER_IMAGE) -> float:
    """Dollars to read parts of these page counts, billed as Vision bills."""
    images = sum(-(-count // pages_per_image) for count in page_counts)
    return vision.cost(images)


def summarise(results: Sequence[extract.PartResult]) -> Dict[str, Any]:
    """What a run of parts came to, in the terms the bill and the roll are both in."""
    done = [r for r in results if not r.error]
    billed = sum(r.images_billed for r in results)
    return {
        "parts": len(results),
        "failed": len(results) - len(done),
        "electors": sum(r.count for r in done),
        "pages": sum(r.page_count for r in done),
        "images_billed": billed,
        "dollars": vision.cost(billed),
    }
