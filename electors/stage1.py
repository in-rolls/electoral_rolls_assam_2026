"""Stage one: everything the CPU can do, and nothing that costs money.

The pipeline splits at the only expensive step. This stage renders each part, finds the boxes
geometrically from the ruled lines, finds the text lines inside them, reads the three fields
tesseract is good enough for -- the EPIC, the section header and the closing totals -- and packs
the four labeled lines into composites for Cloud Vision.

Everything it produces is what stage two needs and nothing more:

- **composites**, the images Vision is billed for. Repacking cuts a rendered page from 15.5 MP
  to 30% of that, which is 3.3x fewer images: $345 for the state against about $105.
- **placements**, every tile's rectangle inside its composite. Vision answers in the composite's
  coordinates and a word has to come back to the box it was cut from; without this, stage two
  would have to re-derive the geometry and could silently disagree with the geometry that
  produced the image.
- **a manifest**, one row per tile carrying ac, part, page, box, band, rectangle, section and
  the source sha, so a reading can be put in the right row and traced to exact source bytes.
- **side reads**, the EPIC and serial per box and the part's printed closing total, which is the
  only number in the source that the extracted rows can be measured against.

It costs about 142 ms of CPU per box -- 985 core-hours for all 126 constituencies, roughly $8 of
spot CPU or five days on a laptop -- and nothing at all in API charges.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from assam_rolls import render, schema

from . import crops, extract, pages, repack, summary, vision, vision_part

#: Bumped whenever stage one changes what it writes, because ``done`` is keyed on it.
#:
#: The concern is data, not code. ``done`` proved a part had been prepared, never *by what*, so
#: a resume happily served side reads produced before a fix -- twice in one evening. Once when
#: composites on disk let a rerun skip parts whose EPICs were still at 62%, and again when a new
#: instance pulled that same output back out of the bucket and reported "resumed with 111 parts
#: already prepared". Both were caught by reading a log line, which is not a control.
#:
#: ``extract.PIPELINE_VERSION`` does exactly this for the row cache and is why that cache has
#: never had the same problem.
#:
#: 2.0.0 -- the EPIC and serial are read from their own cells rather than one strip pass, and a
#:          struck-off entry's status code is recorded.
#: 2.1.0 -- a section marker must be a whole word outside the station-name field, a partial
#:          page's second row is no longer rejected by a gutter estimated from header rules,
#:          and all of a part's columns are built with ink deciding which hold an elector.
#:          Together these recovered 625 rows in AC1 that reconciliation had reported missing.
#: 2.2.0 -- a status code must be one of the five the roll defines in its own legend.
#: 2.3.0 -- a closing page tesseract could not read is kept, so stage two can spend on it.
#: 2.4.0 -- the Bengali genitive section titles (সংযোজনের তালিকা) classify as supplements, and the
#:          Bengali র is accepted where only the Assamese ৰ was. AC126's sections were written by
#:          the older classifier and are exactly the stale output this gate exists to invalidate.
#: 2.5.0 -- section markers have a left word boundary, so -বাদ in place names is not a deletion,
#:          and sparse continuation pages inherit an addition section.
#: 2.6.0 -- section headers and closing summaries use the source roll's declared language;
#:          English pages are no longer sent through the Assamese OCR model.
#: 2.7.0 -- a single ruled elector row is recovered, and an English closing table whose rules
#:          resemble elector boxes is identified by its explicit title instead of emitted.
#: 2.8.0 -- a page whose three detected text columns have impossible unequal widths inherits
#:          the part's established geometry; stray vertical rules no longer become dividers.
#: 2.9.0 -- cached serial, EPIC, and status reads require both identical source bytes and an
#:          identical crop rectangle; a geometry repair cannot retain reads from the old box.
STAGE1_VERSION = "2.9.0"


@dataclass
class PartPrep:
    """What stage one produced for one part, and what it could not."""

    ac_no: int
    part_no: int
    pages: int = 0
    elector_pages: int = 0
    boxes: int = 0
    main_boxes: int = 0
    supplement_boxes: int = 0
    composites: int = 0
    unknown_pages: List[int] = field(default_factory=list)
    summary_total: Optional[int] = None
    seconds: float = 0.0
    error: str = ""

    @property
    def matches_roll(self) -> Optional[bool]:
        """Whether the **main-list** boxes equal the part's own printed total.

        Main list only. The closing page totals ``মূল তালিকা`` -- the main roll -- and a part's
        supplements are counted separately, so comparing every box against it makes a part with
        supplements look over-extracted. Part 4 read 511 boxes against a printed 483 and the 28
        were its supplement, correctly found.

        ``None`` when the closing page could not be read -- reported as unmeasured rather than
        scored against a number that was never established.
        """
        if self.summary_total is None:
            return None
        return self.main_boxes == self.summary_total


def clear_stale(part_dir: Path) -> None:
    """Drop every downstream cache before this part is laid out again.

    Cached words are keyed by image name and nothing else, and a re-render is under no obligation
    to reproduce the old layout -- measured across two runs, only 17.2% of tiles kept identical
    geometry. Words left from the old layout would be attributed to whatever tiles now occupy
    those coordinates, which shuffles fields between electors *silently*: the row counts still
    match, reconciliation still balances, and every name is somebody else's.

    Only reached for a part being re-prepared, so a part resumed intact keeps its words; one being
    rebuilt loses caches that describe an image which is about to stop existing.
    """
    for stale in part_dir.glob("composite*.words.json"):
        stale.unlink()
    (part_dir / "rows.jsonl").unlink(missing_ok=True)


def prepare_part(
    zip_path: Path,
    pdf_name: str,
    part_no: int,
    out_dir: Path,
    dpi: int = extract.DPI,
) -> PartPrep:
    """Render one part and write everything stage two will need for it."""
    started = time.time()
    meta = schema.parse_source_filename(pdf_name) or {}
    prep = PartPrep(ac_no=meta.get("ac_no", 0), part_no=part_no)
    part_dir = out_dir / f"part{part_no:04d}"

    try:
        import tempfile

        pdf_bytes = render.read_pdf_bytes(zip_path, pdf_name)
        sha = render.sha256_bytes(pdf_bytes)
        with tempfile.TemporaryDirectory() as tmp:
            images = vision_part.rasterize(dpi)(pdf_bytes, Path(tmp))
            prep.pages = len(images)
            entries, side = vision_part.tiles_for(
                images,
                meta.get("lang", ""),
                prior_box_reads=_prior_box_reads(part_dir, sha),
            )
            prep.unknown_pages = side["unknown"]
            prep.elector_pages = len(side["pages"])
            prep.boxes = len(entries)
            for _, _, key in entries:
                if side["pages"][key[0]]["section"] is pages.Section.MAIN:
                    prep.main_boxes += 1
                else:
                    prep.supplement_boxes += 1
            if side["summary"] is not None:
                prep.summary_total = side["summary"].total

            part_dir.mkdir(parents=True, exist_ok=True)
            clear_stale(part_dir)
            budget = int(vision.MAX_PIXELS * vision.PIXEL_MARGIN)
            placements: Dict[str, Any] = {}
            written: List[crops.Crop] = []
            for number, batch in enumerate(repack.batches(entries, budget)):
                composite, placed = repack.compose(batch)
                # Keyed by the filename itself, not a stem the reader has to know how to
                # complete. The first version keyed on "composite000" and the verifier went
                # looking for a file of that name.
                name = f"composite{number:03d}.png"
                composite.save(part_dir / name, format="PNG", optimize=True)
                placements[name] = [
                    {
                        "page": tile.key[0],
                        "box_row": tile.key[1],
                        "box_col": tile.key[2],
                        "left": tile.left,
                        "top": tile.top,
                        "right": tile.right,
                        "bottom": tile.bottom,
                    }
                    for tile in placed
                ]
                prep.composites += 1
                for _, rect, key in batch:
                    page_side = side["pages"][key[0]]
                    written.append(
                        crops.Crop(
                            part=part_no,
                            page=key[0],
                            row=key[1],
                            column=key[2],
                            path=part_dir / name,
                            band=None,
                            left=rect[0],
                            top=rect[1],
                            right=rect[2],
                            bottom=rect[3],
                            ac_no=prep.ac_no,
                            section=page_side["section"].value,
                            source_zip=zip_path.name,
                            source_pdf=pdf_name,
                            pdf_sha256=sha,
                        )
                    )

            _write(part_dir / "placements.json", placements)
            # The closing page tesseract could not read, left for stage two to buy a reading of.
            if side.get("unread_summary") is not None:
                side["unread_summary"].save(
                    part_dir / "summary_page.png", format="PNG", optimize=True
                )
            write_side(part_dir / "side.json", side)
            crops.write_manifest(written, part_dir, append=False)
            for image in images:
                image.close()
    except Exception as exc:  # a bad part must not stop the constituency
        prep.error = f"{type(exc).__name__}: {exc}"
    prep.seconds = time.time() - started
    return prep


def _prior_box_reads(part_dir: Path, pdf_sha256: str) -> vision_part.PriorBoxReads:
    """Reusable serial/EPIC reads from an earlier layout of the identical source PDF.

    A stage-one version bump can change page classification without changing an existing box's
    header. Re-running two Tesseract processes for every unchanged box makes a safe cache
    invalidation take hours. Position keys are reused only when the old manifest proves that it
    came from the same source bytes and records the exact header OCR rectangles. The caller
    compares those rectangles with the current serial/status and EPIC crops before reuse.
    """
    side_path = part_dir / "side.json"
    if not side_path.exists():
        return {}
    try:
        manifest = crops.read_manifest(part_dir)
        if not manifest or {row.get("pdf_sha256") for row in manifest.values()} != {pdf_sha256}:
            return {}
        previous = read_side(side_path)
        if previous.get("header_reader_version") != vision_part.HEADER_READER_VERSION:
            return {}
    except (KeyError, OSError, ValueError):
        return {}
    return {key: read for page in previous["pages"].values() for key, read in page["boxes"].items()}


def _write(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def write_side(path: Path, side: Dict[str, Any]) -> None:
    """The CPU's own readings, in a form that survives the trip to disk unchanged.

    A round trip, not a summary. The first version wrote a flattened digest -- EPICs, section
    values, a total -- and stage two would have had to re-render the part to recover the sex
    breakdown and the unreadable pages, which is the re-rendering the split exists to avoid.
    """
    _write(
        path,
        {
            "pages": {
                str(number): {
                    "section": page["section"].value,
                    "recognised": page["recognised"],
                    "boxes": {f"{k[1]},{k[2]}": read for k, read in page["boxes"].items()},
                }
                for number, page in side["pages"].items()
            },
            # Preserved, never re-stamped. Stage two rewrites this file when it buys a closing
            # total, and stamping the current version there would mark output produced by older
            # stage-one code as fresh -- the exact failure done() exists to prevent.
            "stage1_version": side.get("stage1_version") or STAGE1_VERSION,
            # Unlike stage1_version, never default this while rewriting an older side file in
            # stage two. A purchased summary must not make stale header OCR look compatible.
            "header_reader_version": side.get("header_reader_version", ""),
            "unknown": side["unknown"],
            "summary": (
                None
                if side["summary"] is None
                else {
                    "male": side["summary"].male,
                    "female": side["summary"].female,
                    "third": side["summary"].third,
                    "total": side["summary"].total,
                    "scale": side["summary"].scale,
                }
            ),
        },
    )


def read_side(path: Path) -> Dict[str, Any]:
    """``write_side`` inverted, giving back exactly what ``tiles_for`` produced."""
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        "pages": {
            int(number): {
                "section": pages.Section(page["section"]),
                "recognised": page["recognised"],
                "boxes": {
                    (int(number), int(k.split(",")[0]), int(k.split(",")[1])): read
                    for k, read in page["boxes"].items()
                },
            }
            for number, page in raw["pages"].items()
        },
        "stage1_version": raw.get("stage1_version", ""),
        "header_reader_version": raw.get("header_reader_version", ""),
        "unknown": raw["unknown"],
        "summary": (None if raw["summary"] is None else summary.RollSummary(**raw["summary"])),
    }


def image_names(part_dir: Path) -> List[str]:
    """The composites this part is made of, from the record rather than from the disk.

    ``placements.json`` names every image stage one wrote and where each tile landed in it, so it
    knows the part's shape whether or not the pixels are still here. Listing the directory instead
    conflates "this part has no images" with "this part's images were deleted", and those need
    opposite responses.
    """
    path = part_dir / "placements.json"
    if not path.exists():
        return []
    try:
        return sorted(json.loads(path.read_text(encoding="utf-8")))
    except (ValueError, OSError):
        return []


def readable(part_dir: Path) -> bool:
    """Whether every composite this part names can still be read -- as pixels or as cached words.

    Composites are not synced to the bucket: they are a terabyte for the state and cheap to
    rebuild. So a machine resuming from the bucket gets a part's metadata and its *words* without
    its images, and there are three states here, not two:

    - images present: readable, and re-readable if the parser changes
    - images gone, words cached: still readable. The words carry their own coordinates; the image
      was only ever opened to learn how big it was, and ``placements.json`` records that too
    - neither: not readable, and the part must be prepared again

    Treating the second state as unprepared is what has been happening, and it is expensive in
    both directions. Calling it *un*readable makes every machine re-render and re-side-read every
    part of every constituency it resumes -- four machines spent six hours after one reboot
    redoing work whose results were sitting in the bucket. Calling it readable when the words are
    *not* there is the older failure: AC101 shipped 113 of its 210 parts and reported no error,
    because stage one skipped the parts as prepared and stage two skipped them as empty.
    """
    names = image_names(part_dir)
    if not names:
        return False
    return all(
        (part_dir / name).exists() or _has_words(part_dir / f"{Path(name).stem}.words.json")
        for name in names
    )


#: ``[]`` and nothing else. A words file this small holds no words, and a composite with no words
#: on it does not exist -- it is a read that failed and was written down anyway.
EMPTY_WORDS = 2


def _has_words(path: Path) -> bool:
    """Whether a cached read actually found anything.

    Existence is not enough. AC33 kept 214 empty words files, and a part standing on those has
    the paperwork of a finished read with none of the result: no image to go back to, nothing to
    parse, and no error raised by either.
    """
    try:
        return path.stat().st_size > EMPTY_WORDS
    except OSError:
        return False


def done(out_dir: Path, part_no: int) -> bool:
    """Whether this part is already prepared, so a run can be resumed.

    Requires the manifest *and* the placements: a part interrupted between writing composites
    and writing where the tiles are cannot be read by stage two, and half-finished work that
    looks finished is worse than work that was never started.

    And requires the side reads to have been written by *this* version of stage one. Without
    that, a rerun after a fix serves the output the fix was meant to replace, and the resulting
    constituency is a silent mixture of two vintages with nothing in the data to tell them
    apart. Re-preparing a part costs a few seconds of CPU; not re-preparing it costs the fix.
    """
    part_dir = out_dir / f"part{part_no:04d}"
    if not (part_dir / crops.MANIFEST).exists() or not (part_dir / "placements.json").exists():
        return False
    # And something stage two can actually read -- pixels or cached words. done() and
    # stage2.ready() must not disagree about what a finished part is, so they ask the same
    # function.
    if not readable(part_dir):
        return False
    side = part_dir / "side.json"
    if not side.exists():
        return False
    try:
        return json.loads(side.read_text(encoding="utf-8")).get("stage1_version") == STAGE1_VERSION
    except (ValueError, OSError):
        return False


def summarise(preps: Sequence[PartPrep]) -> Dict[str, Any]:
    """What the run established, in the terms the next stage and the bill are both in."""
    ok = [p for p in preps if not p.error]
    measured = [p for p in ok if p.matches_roll is not None]
    return {
        "parts": len(preps),
        "failed": len(preps) - len(ok),
        "pages": sum(p.pages for p in ok),
        "boxes": sum(p.boxes for p in ok),
        "main_boxes": sum(p.main_boxes for p in ok),
        "supplement_boxes": sum(p.supplement_boxes for p in ok),
        "composites": sum(p.composites for p in ok),
        "parts_measured": len(measured),
        "parts_matching_roll": sum(1 for p in measured if p.matches_roll),
        "parts_unmeasured": [p.part_no for p in ok if p.matches_roll is None],
        "residuals": [
            {
                "part_no": p.part_no,
                "main_boxes": p.main_boxes,
                "supplement_boxes": p.supplement_boxes,
                "roll_total": p.summary_total,
                "diff": p.main_boxes - (p.summary_total or 0),
            }
            for p in measured
            if not p.matches_roll
        ],
        "unknown_pages": {p.part_no: p.unknown_pages for p in ok if p.unknown_pages},
        "seconds": sum(p.seconds for p in preps),
        "vision_cost": vision.cost(sum(p.composites for p in ok)),
    }
