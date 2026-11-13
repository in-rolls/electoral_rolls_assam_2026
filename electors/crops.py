"""Cutting box images out for a reader that runs somewhere else.

Cloud Vision reads the state, and once it has, **nothing checks it**. There are no labels at
this scale, so the only automatic error signal is two independent engines disagreeing --
dots.ocr is the strongest second reader available and it fails differently, being pure OCR with
no language prior where Vision has one.

It cannot run the state: it reads per box, so 25 million inferences, which is months of free
GPU quota for accuracy it already ties. On a sample it is minutes, and a sample is all a check
needs. That means getting a few thousand box images to a machine with a GPU, which is what this
module is for.

**The filename is the key.** ``p13_pg4_r2c0.png`` is part 13, page 4, box row 2, column 0 --
the same spelling ``dataset/eval`` uses and the same thing :data:`electors.resolution.KEY`
parses. So a folder of crops and a JSON of readings need nothing else to be lined up against
what Vision said about the same boxes; there is no separate manifest to fall out of step.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from PIL import Image

from assam_rolls import render, schema

from . import extract, grid, pages

#: ``p13_pg4_r2c0`` for a whole box, ``p13_pg4_r2c0_b0`` for one line inside it -- the same
#: spelling ``dataset/eval`` and :mod:`electors.resolution` use, with the band appended.
NAME = re.compile(r"p(\d+)_pg(\d+)_r(\d+)c(\d+)(?:_b(\d+))?$")


def name_for(part: int, page: int, row: int, column: int, band: Optional[int] = None) -> str:
    stem = f"p{part}_pg{page}_r{row}c{column}"
    return stem if band is None else f"{stem}_b{band}"


def key_of(name: str) -> Optional[Tuple[int, int, int, int]]:
    """``(part, page, row, column)`` from a crop's filename, or None.

    The band is deliberately *not* part of the key: several bands belong to one elector, and
    this is what joins them back to that one row.
    """
    found = NAME.match(Path(name).stem)
    if not found:
        return None
    return tuple(int(g) for g in found.groups()[:4])  # type: ignore[return-value]


def band_of(name: str) -> Optional[int]:
    """Which line of the box a crop is, or None if it is the whole box."""
    found = NAME.match(Path(name).stem)
    if not found or found.group(5) is None:
        return None
    return int(found.group(5))


#: How far past the ink a band crop reaches, as a fraction of the band's own height.
#:
#: ``grid.text_bands`` pads by a flat 5px, which is enough for tesseract and **not** enough to
#: hand a line to a model on its own: at 400 dpi that clips the top of a tall matra and the tail
#: of a descender. Cut at the bands as given, ``বিনয়`` lost the head of its ``ি`` and ``অৰ্জুন``
#: lost its hanger -- read as a different name, with nothing to show anything had gone wrong.
#:
#: 0.45 recovers both. It is deliberately more than needed and then clamped, because the cost of
#: too little is a wrong name and the cost of too much is nothing.
BAND_PAD_FRACTION = 0.45


def band_window(bands: Sequence[Tuple[int, int]], index: int, box: grid.Box) -> Tuple[int, int]:
    """The crop rectangle for one band: padded generously, never into its neighbor.

    Padding alone bleeds the next line into the crop -- at 0.45 the relation line arrives under
    the name, and a model handed two lines may answer with either. Clamping each side to the
    midpoint of the gap takes everything available and stops exactly where the neighbour's claim
    begins.
    """
    top, bottom = bands[index]
    pad = int((bottom - top) * BAND_PAD_FRACTION)
    ceiling = box.top if index == 0 else (bands[index - 1][1] + top) // 2
    floor = box.bottom if index == len(bands) - 1 else (bottom + bands[index + 1][0]) // 2
    return max(ceiling, top - pad), min(floor, bottom + pad)


@dataclass
class Crop:
    """One image and everything needed to put its reading back in the right row.

    The GPU stage sees only a folder of PNGs; this is what the CPU stage must have written down
    for the join to be reconstructable. The rectangle is here so any crop can be re-cut or
    checked against the source without re-deriving the geometry, and the sha ties the row to
    exact source bytes rather than to a filename that may be reissued.
    """

    part: int
    page: int
    row: int
    column: int
    path: Path
    band: Optional[int] = None
    left: int = 0
    top: int = 0
    right: int = 0
    bottom: int = 0
    ac_no: int = 0
    section: str = ""
    source_zip: str = ""
    source_pdf: str = ""
    pdf_sha256: str = ""

    @property
    def key(self) -> Tuple[int, int, int, int]:
        """The elector this crop belongs to. Bands of one box share it."""
        return (self.part, self.page, self.row, self.column)

    @property
    def name(self) -> str:
        """The row's key: where the box is, never which file it landed in.

        These were the same thing while every crop was its own PNG. Once tiles are packed many
        to a composite they are not: every tile in ``composite000`` shared that stem, so 873
        manifest rows keyed down to 3 and 870 electors disappeared between the stages without a
        word. The position is unique by construction; the filename is not.
        """
        return name_for(self.part, self.page, self.row, self.column, self.band)

    def as_row(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            # Which image the tile is in -- one composite holds hundreds.
            "image": self.path.name,
            "ac_no": self.ac_no,
            "part_no": self.part,
            "page_no": self.page,
            "box_row": self.row,
            "box_col": self.column,
            "band": self.band,
            "left": self.left,
            "top": self.top,
            "right": self.right,
            "bottom": self.bottom,
            "roll_section": self.section,
            "source_zip": self.source_zip,
            "source_pdf": self.source_pdf,
            "pdf_sha256": self.pdf_sha256,
        }


def boxes_of(image: Image.Image, number: int) -> List[grid.Box]:
    """The elector boxes on one page, or nothing if it is not an elector page."""
    signature = pages.classify(image, number)
    if not signature.is_elector:
        return []
    return grid.build(signature.h_rules, signature.v_rules)


def export_part(
    zip_path: Path,
    pdf_name: str,
    part_no: int,
    out_dir: Path,
    dpi: int = extract.DPI,
    limit: Optional[int] = None,
    bands: Optional[Sequence[int]] = None,
) -> List[Crop]:
    """Write one part's boxes -- or single lines inside them -- as PNGs named after their position.

    Cropped to ``text_right`` rather than the box's full width: everything past it is the photo
    placeholder, and it is the same region the pipeline reads. Sending the photo would change
    what the reader is being asked.

    ``bands`` selects lines instead of whole boxes -- ``(0,)`` is the name line alone, which is
    84 vision tokens against the box's 420. That is the whole point of the split: the model is
    wanted for the name, and showing it the borders, the photo panel and three lines of digits
    costs five times as much prefill for nothing.

    Boxes with no ink are skipped before anything is written -- a blank box at the end of a part
    is the publisher's doing and there is nothing in it to read.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = schema.parse_source_filename(pdf_name) or {}
    pdf_bytes = render.read_pdf_bytes(zip_path, pdf_name)
    sha = render.sha256_bytes(pdf_bytes)
    written: List[Crop] = []

    def record(image, box, number, section, rect, band):
        name = name_for(part_no, number, box.row, box.column, band)
        target = out_dir / f"{name}.png"
        image.crop(rect).save(target, format="PNG", optimize=True)
        written.append(
            Crop(
                part=part_no,
                page=number,
                row=box.row,
                column=box.column,
                path=target,
                band=band,
                left=rect[0],
                top=rect[1],
                right=rect[2],
                bottom=rect[3],
                ac_no=meta.get("ac_no", 0),
                section=section,
                source_pdf=pdf_name,
                pdf_sha256=sha,
            )
        )

    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        for number, path in enumerate(extract._render_pages(pdf_bytes, workdir, dpi=dpi), start=1):
            with Image.open(path) as page_image:
                image = page_image.convert("L")
                found = boxes_of(image, number)
                section = _section_of(image, found) if found else ""
                for box in found:
                    if limit is not None and len(written) >= limit:
                        return written
                    if not grid.has_ink(image, box):
                        continue
                    if bands is None:
                        rect = (box.left, box.top, box.text_right, box.bottom)
                        record(image, box, number, section, rect, None)
                        continue
                    found_bands = grid.text_bands(image, box)
                    for index in bands:
                        if index >= len(found_bands):
                            # The band was not found on this box. Skipped rather than guessed at
                            # from a fixed offset: a crop cut at the wrong height reads as a
                            # confident wrong name.
                            continue
                        top, bottom = band_window(found_bands, index, box)
                        rect = (box.left, top, box.text_right, bottom)
                        record(image, box, number, section, rect, index)
    return written


def _section_of(image: Image.Image, boxes: Sequence[grid.Box]) -> str:
    """Main roll or supplement, from the header band above the first box.

    Carried into the manifest because serial numbers restart at 1 in each list, so a row without
    it cannot be numbered correctly later.
    """
    top = min(boxes[0].top, int(image.height * 0.09))
    if top <= 0:
        return pages.Section.MAIN.value
    from assam_rolls import ocr

    try:
        engine = ocr.get_engine("tesseract", lang="asm")
        header = engine._run(image.crop((0, 0, image.width, top)), "asm", None, 1, psm=6)
    except Exception:
        return pages.Section.MAIN.value
    return pages.section_of(header)[0].value


def export(
    zip_path: Path,
    parts: Sequence,
    out_dir: Path,
    dpi: int = extract.DPI,
    limit: Optional[int] = None,
) -> Iterator[Tuple[int, List[Crop]]]:
    """Every part in turn, yielding as each finishes so a long export shows progress."""
    total = 0
    for part in parts:
        remaining = None if limit is None else max(0, limit - total)
        if remaining == 0:
            return
        found = export_part(
            zip_path, part.pdf_name, part.part_no, out_dir, dpi=dpi, limit=remaining
        )
        total += len(found)
        yield part.part_no, found


MANIFEST = "manifest.jsonl"


def write_manifest(written: Sequence[Crop], out_dir: Path, append: bool = True) -> Path:
    """One row per crop, written beside the crops by whoever cut them.

    The filename carries the join key, but not the rectangle, the section or the source sha. A
    manifest generated separately from the images can disagree with them, so this is called from
    the same loop that writes the PNGs.
    """
    import json

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / MANIFEST
    with path.open("a" if append else "w", encoding="utf-8") as handle:
        for crop in written:
            handle.write(json.dumps(crop.as_row(), ensure_ascii=False) + "\n")
    return path


def read_manifest(out_dir: Path) -> Dict[str, Dict[str, Any]]:
    """The manifest keyed by crop name, which is how readings are joined back."""
    import json

    path = out_dir / MANIFEST
    if not path.exists():
        return {}
    rows: Dict[str, Dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row["name"] in rows:
            # Keys collide only if the writer used something that is not a position. Raised
            # rather than overwritten: silently keeping the last row is how 873 tiles became 3.
            raise ValueError(f"duplicate manifest key {row['name']!r} in {path}")
        rows[row["name"]] = row
    return rows


def readings_to_arm(readings: dict, arm: str = "dots.ocr") -> List:
    """Turn ``{crop name: terse text}`` into the arms :mod:`electors.resolution` compares.

    The readings come back from whatever GPU ran them as one JSON object. Parsed here with the
    *same* parser Vision's output goes through, because two readers of one format is the bug
    that put the elector's own name in the relation field, and comparing engines through two
    different parsers would measure the parsers.
    """
    from . import resolution, second_pass

    by_part: dict = {}
    for name, text in readings.items():
        key = key_of(name)
        if not key:
            continue
        part, page, row, column = key
        found = second_pass.Reading(str(text or "")).fields()
        by_part.setdefault(part, {})[(page, row, column)] = {
            "name": found.get("name", ""),
            "age": found.get("age"),
            "house": found.get("house_no", ""),
            "sex": found.get("sex", ""),
        }
    return [
        resolution.ArmResult(arm=arm, part=part, boxes=boxes)
        for part, boxes in sorted(by_part.items())
    ]


def zip_dir(directory: Path, target: Path) -> Path:
    """Zip a crop folder for upload. Kaggle takes a dataset far more happily than 10,000 files."""
    subprocess.run(
        ["zip", "-q", "-r", "-0", str(target), "."],
        cwd=directory,
        check=True,
        capture_output=True,
    )
    return target
