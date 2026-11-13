"""One part PDF in, one list of electors out.

The unit of work is the **part**, not the page, for two reasons that both come from the data
rather than from convenience. Serial numbers run consecutively through a part and restart at
the next one, so numbering can only be assigned with the whole part in view. And each part
has a published elector total on its info page, so the part is also the smallest unit that
can be *checked* -- which makes it the right thing to cache, retry and resume.

Rendering is the expensive half. ``pdftoppm`` costs about a second a page at 400 dpi, and a
part is thirty-odd pages, so pages are rasterized once in a single invocation and read from
disk rather than re-rendered per field.
"""

from __future__ import annotations

import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from PIL import Image

from assam_rolls import languages, ocr, render, schema

from . import fields, grid, pages, replay
from . import summary as summary_page

#: Everything downstream is calibrated to this. The boxes are ~1000px wide here, which is
#: where the Assamese conjuncts in a name become legible; at 200 dpi they are not.
DPI = 400

#: Bumped whenever extraction changes what a part yields, because the cache is keyed on it.
#: 1.1.0 -- partial last pages, which draw no photo dividers, are no longer discarded whole.
#: 1.2.0 -- supplement pages are labeled instead of merged into the main roll.
#: 1.3.0 -- the roll's own closing totals are read, so counts have a real target.
#: 1.4.0 -- the age line is found by its লিঙ্গ label, recovering the male rows it dropped.
#: 1.5.0 -- the name consensus compares the two scales on the *same band*, so it can disagree.
#:          Bumped despite the cost of discarding 10,245 cached rows: the flag it feeds was
#:          raised zero times under 1.4.0, and a cache mixing the two would report a
#:          disagreement rate that is part real and part an artifact of when a part was read.
#: 1.6.0 -- bands anchor on the relation label rather than counting inwards from the ends, the
#:          two scales are compared once against a measured threshold instead of twice against
#:          two different rules, debris stranded after punctuation is stripped, and an age zone
#:          holding more digits than an age can account for is flagged rather than guessed.
#:
#: Deliberately *not* bumped for the narrowed serial zone. The version exists so a resume does
#: not serve rows produced by code since fixed, and that concern is about data. The serial crop
#: feeds only ``serial_no_ocr`` -- an independent check on the derived row order, itself noisy
#: enough to disagree on most rows -- and no field a consumer reads. Bumping would have thrown
#: away a seventeen-hour constituency to improve a diagnostic, so a 1.6.0 shard may carry
#: serial_no_ocr from either crop and nothing else differs.
#: 1.7.0 -- Latin letters and digits are stripped from names and relations, where they are
#:          scanner debris by definition: the roll prints those fields in Assamese only.
#: 1.8.0 -- relation labels are matched by shape and scored against the three relatives, rather
#:          than enumerated. The enumeration was not converging.
#: 1.9.0 -- repacked English boxes retain their body lines instead of passing through the
#:          Assamese-script header splitter, which discarded every field in AC113.
#: 2.0.0 -- serial numbers continue through supplements instead of restarting at section
#:          boundaries, and sparse supplement pages inherit the preceding addition section.
#: 2.1.0 -- row provenance records the Google Cloud Vision body reader and Tesseract header
#:          reader instead of incorrectly attributing every extracted field to Tesseract.
PIPELINE_VERSION = "2.1.0"

#: The production path sends elector body fields to Google Cloud Vision and reads the header
#: cells locally with Tesseract. A single-engine label would misstate either half of the row.
MIXED_ENGINE = "google-cloud-vision+tesseract"
VISION_ENGINE = "google-cloud-vision"


@dataclass
class PartResult:
    """Every elector in one part, with what the run could and could not establish."""

    ac_no: int
    part_no: int
    lang: str
    source_zip: str
    source_pdf: str
    pdf_sha256: str
    electors: List[Dict[str, Any]] = field(default_factory=list)
    page_count: int = 0
    elector_pages: int = 0
    unknown_pages: List[int] = field(default_factory=list)
    #: Elector pages belonging to a supplement rather than the main roll.
    supplement_pages: List[int] = field(default_factory=list)
    #: Pages whose header band could not be read as either. Worth a look: an unrecognized
    #: supplement would otherwise be counted as main-roll electors.
    unrecognised_headers: List[int] = field(default_factory=list)
    #: What the closing page says this part contains -- the number the main-roll rows are
    #: measured against. ``None`` when it could not be read, which is reported rather than
    #: quietly replaced by the info page's (different) net.
    summary_male: Optional[int] = None
    summary_female: Optional[int] = None
    summary_third: Optional[int] = None
    summary_total: Optional[int] = None
    #: Images submitted to a per-image engine. Zero for tesseract, which bills nothing, and the
    #: unit the Cloud Vision bill is actually in -- eight pages stacked into one image is one
    #: image here and one image on the invoice.
    images_billed: int = 0
    error: str = ""

    @property
    def count(self) -> int:
        return len(self.electors)


def _render_pages(pdf_bytes: bytes, workdir: Path, dpi: int = DPI) -> List[Path]:
    """Rasterize the whole PDF once. Returns page images in page order."""
    pdf = workdir / "part.pdf"
    pdf.write_bytes(pdf_bytes)
    subprocess.run(
        # -gray, because the source is a bilevel scan: every page in out/pages has R == G == B
        # on every pixel, so the color channels are two exact copies. Rendering them costs three
        # times the temp bytes and three times the decode, and every consumer here immediately
        # calls convert("L") anyway.
        ["pdftoppm", "-gray", "-r", str(dpi), "-png", str(pdf), str(workdir / "page")],
        check=True,
        capture_output=True,
    )
    return sorted(workdir.glob("page-*.png"), key=lambda p: int(p.stem.split("-")[-1]))


def read_part(
    zip_path: Path,
    pdf_name: str,
    engine: Optional[Any] = None,
    capture_lines: bool = False,
) -> PartResult:
    """Extract every elector from one part PDF.

    ``capture_lines`` writes everything the OCR said to ``dataset/lines`` on the way past, so
    a later parsing change can be scored against identical text in seconds instead of another
    half-hour of tesseract. It costs one JSON file per part and no extra OCR.
    """
    meta = schema.parse_source_filename(pdf_name) or {}
    pdf_bytes = render.read_pdf_bytes(zip_path, pdf_name)
    result = PartResult(
        ac_no=meta.get("ac_no", 0),
        part_no=meta.get("part_no", 0),
        lang=meta.get("lang", ""),
        source_zip=zip_path.name,
        source_pdf=pdf_name,
        pdf_sha256=render.sha256_bytes(pdf_bytes),
    )
    tesseract_lang = languages.profile_for(result.lang).tesseract_lang
    engine = engine or ocr.get_engine("tesseract", lang=tesseract_lang)
    captured = (
        replay.PartLines(
            part=result.part_no,
            ac=result.ac_no,
            capture_version=replay.CAPTURE_VERSION,
            scale=fields.NAME_SCALES[0],
            psm=7,
            lang=tesseract_lang,
        )
        if capture_lines
        else None
    )

    try:
        with tempfile.TemporaryDirectory() as tmp:
            workdir = Path(tmp)
            page_paths = _render_pages(pdf_bytes, workdir)
            result.page_count = len(page_paths)
            serial = 0
            current_section = pages.Section.MAIN

            for index, path in enumerate(page_paths, start=1):
                image = Image.open(path)
                signature = pages.classify(image, index)
                if signature.kind is pages.PageKind.UNKNOWN:
                    result.unknown_pages.append(index)
                    continue
                if signature.kind is pages.PageKind.SUMMARY and result.summary_total is None:
                    found = summary_page.read(engine, image, tesseract_lang)
                    if found:
                        result.summary_male = found.male
                        result.summary_female = found.female
                        result.summary_third = found.third
                        result.summary_total = found.total
                if not signature.is_elector:
                    continue

                boxes = grid.build(signature.h_rules, signature.v_rules)
                if not boxes:
                    # Classified as an elector page but its geometry did not resolve. Recorded
                    # rather than parsed: guessing thirty box positions here would produce
                    # thirty plausible rows of nothing.
                    result.unknown_pages.append(index)
                    continue
                header = _page_header(engine, image, boxes, tesseract_lang)
                if summary_page.is_summary(header):
                    if result.summary_total is None:
                        found = summary_page.read(engine, image, tesseract_lang)
                        if found:
                            result.summary_male = found.male
                            result.summary_female = found.female
                            result.summary_third = found.third
                            result.summary_total = found.total
                    image.close()
                    continue

                result.elector_pages += 1
                section, recognised = pages.section_of(header)
                section = pages.section_after(current_section, section)
                current_section = section
                if not recognised:
                    result.unrecognised_headers.append(index)
                if section is not pages.Section.MAIN:
                    result.supplement_pages.append(index)
                for box, elector, reads in fields.read_page(image, boxes):
                    if captured is not None:
                        captured.boxes.append(
                            replay.BoxLines(
                                page=index,
                                section=section.value,
                                col=box.column,
                                row=box.row,
                                lines=list(reads.lines),
                                name_second=list(reads.name_second),
                                epic_raw=reads.epic_raw,
                                serial_raw=reads.serial_raw,
                            )
                        )
                    if elector.is_empty:
                        # Ink but nothing readable. Emitted anyway: the row exists in the
                        # source, and dropping it makes the count agree with nothing.
                        elector.flags.append("unreadable")
                    serial += 1
                    elector.serial_no = serial
                    # The OCR'd serial is *recorded*, not used to flag. It disagreed on 68%
                    # of the first run's rows -- because it is a small number alone in a wide
                    # strip, not because the derived sequence was wrong -- and flagging it
                    # pushed needs_review to 95%, drowning every signal that mattered.
                    result.electors.append(
                        _row(
                            result,
                            elector,
                            page=index,
                            box=box,
                            meta=meta,
                            section=section,
                        )
                    )
                image.close()
    except (subprocess.CalledProcessError, OSError, ocr.OCRError) as exc:
        result.error = f"{type(exc).__name__}: {exc}"
    if captured is not None and captured.boxes and not result.error:
        replay.save(captured)
    return result


def _page_header(engine, image: Image.Image, boxes: List[grid.Box], lang: str) -> str:
    """Read the title band used to distinguish sections from closing summaries."""
    top = min(boxes[0].top, int(image.height * 0.09))
    if top <= 0:
        return ""
    return engine._run(image.crop((0, 0, image.width, top)), lang, None, 1, psm=6)


def _row(
    result: PartResult,
    elector: fields.Elector,
    page: int,
    box: grid.Box,
    meta: Dict[str, Any],
    section: "pages.Section",
    engine: str = "tesseract",
) -> Dict[str, Any]:
    """One output row: the elector, where it came from, and how it was read."""
    row = asdict(elector)
    row["flags"] = ",".join(elector.flags)
    row.update(
        {
            "ac_no": result.ac_no,
            "part_no": result.part_no,
            "lang": result.lang,
            "page_no": page,
            "roll_section": section.value,
            "box_row": box.row,
            "box_col": box.column,
            "needs_review": elector.needs_review,
            "source_zip": result.source_zip,
            "source_pdf": result.source_pdf,
            "pdf_sha256": result.pdf_sha256,
            "roll_type": meta.get("roll_type", ""),
            "revision": meta.get("revision"),
            "year": meta.get("year"),
            "engine": engine,
            "pipeline_version": PIPELINE_VERSION,
            "extracted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
    )
    return row


#: Column order for the Parquet shards. Identity first, then content, then provenance --
#: the same shape as ``dataset/parts.jsonl.gz`` so the two read alike.
COLUMNS = [
    "ac_no",
    "part_no",
    "serial_no",
    "epic_no",
    "name",
    "relation_name",
    "relation_type",
    "house_no",
    "age",
    "sex",
    # Struck off the roll: printed, numbered and counted in the part's closing total, but no
    # longer an elector. A row count is not an elector count until these are excluded.
    "deleted",
    "status_code",
    "lang",
    "roll_section",
    "page_no",
    "box_row",
    "box_col",
    "serial_no_ocr",
    "flags",
    "needs_review",
    "source_zip",
    "source_pdf",
    "pdf_sha256",
    "roll_type",
    "revision",
    "year",
    "engine",
    "pipeline_version",
    "extracted_at",
]
