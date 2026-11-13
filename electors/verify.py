"""Proving the repack is right, rather than assuming it.

The composites are what Cloud Vision is billed for and what every extracted row derives from, so
"the packing is correct" cannot rest on having looked at a contact sheet. Two failures matter and
neither announces itself:

- **a tile lost**, so an elector who is plainly printed never reaches the output
- **a tile misplaced**, so a name comes back attributed to somebody else's row

Both are checkable exactly, because stage one records two rectangles for every tile: where it was
cut from the page (``manifest.jsonl``) and where it landed in the composite (``placements.json``).
Cropping the composite at the second must give **byte-identical pixels** to cropping the
re-rendered page at the first. Nothing about that is statistical -- it either matches or it names
the tile that does not.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from assam_rolls import render

from . import crops, extract, vision_part


@dataclass
class PartCheck:
    """What one part's repack survived, and what it did not."""

    part_no: int
    tiles: int = 0
    placements: int = 0
    compared: int = 0
    identical: int = 0
    problems: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """No problems **and** something actually compared.

        An audit pointed a part directory with an empty manifest and ``{}`` placements at this
        and got "every check passed": nothing was compared, so nothing disagreed. A proof that
        rests on zero observations is not a proof.
        """
        return not self.problems and self.tiles > 0 and self.compared > 0

    def note(self, problem: str) -> None:
        self.problems.append(problem)


def _rect(row: Dict[str, Any]) -> Tuple[int, int, int, int]:
    return (row["left"], row["top"], row["right"], row["bottom"])


def _overlaps(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return not (
        a["right"] <= b["left"]
        or b["right"] <= a["left"]
        or a["bottom"] <= b["top"]
        or b["bottom"] <= a["top"]
    )


def check_structure(part_dir: Path) -> PartCheck:
    """Everything checkable without re-rendering: counts, duplicates, overlaps.

    Runs first because it is instant, and because a structural failure makes the pixel check
    meaningless -- there is no point comparing images if the rows they belong to are wrong.
    """
    part_no = int(part_dir.name.replace("part", ""))
    check = PartCheck(part_no=part_no)

    try:
        manifest = crops.read_manifest(part_dir)
    except ValueError as exc:
        check.note(str(exc))
        return check
    placements = json.loads((part_dir / "placements.json").read_text(encoding="utf-8"))

    check.tiles = len(manifest)
    check.placements = sum(len(v) for v in placements.values())
    if check.tiles != check.placements:
        check.note(f"{check.tiles} manifest rows against {check.placements} placements")

    # Every placement must name a box the manifest knows, and each box exactly once.
    seen: Dict[str, str] = {}
    for image, tiles in placements.items():
        for tile in tiles:
            key = crops.name_for(part_no, tile["page"], tile["box_row"], tile["box_col"])
            if key not in manifest:
                check.note(f"{image} places {key}, which is not in the manifest")
            elif key in seen:
                check.note(f"{key} is placed twice: {seen[key]} and {image}")
            else:
                seen[key] = image
    for key, row in manifest.items():
        if key not in seen:
            check.note(f"{key} is in the manifest but was never placed -- an elector lost")

    # Inside one composite no two tiles may overlap: a word in the overlap belongs to both.
    for image, tiles in placements.items():
        for i, a in enumerate(tiles):
            for b in tiles[i + 1 : i + 4]:  # sorted by position, so neighbors suffice
                if _overlaps(a, b):
                    check.note(f"{image}: tiles overlap at {_rect(a)} and {_rect(b)}")
    return check


def check_pixels(
    zip_path: Path,
    pdf_name: str,
    part_dir: Path,
    check: PartCheck,
    sample: Optional[int] = 40,
    dpi: int = extract.DPI,
) -> PartCheck:
    """Compare the composite against the source page, tile by tile.

    The decisive test. If the crop taken from the composite is not the same bytes as the crop
    taken from the page, the tile is either the wrong pixels or in the wrong place, and every row
    derived from it is wrong in a way no downstream check would notice.
    """
    import tempfile

    from PIL import Image

    manifest = crops.read_manifest(part_dir)
    placements = json.loads((part_dir / "placements.json").read_text(encoding="utf-8"))

    wanted: Dict[int, List[Tuple[str, Dict[str, Any], Dict[str, Any]]]] = {}
    for image, tiles in placements.items():
        for tile in tiles:
            key = crops.name_for(check.part_no, tile["page"], tile["box_row"], tile["box_col"])
            if key in manifest:
                wanted.setdefault(tile["page"], []).append((image, tile, manifest[key]))

    chosen = _sample(wanted, sample)
    if not chosen:
        return check

    opened: Dict[str, Any] = {}
    pdf_bytes = render.read_pdf_bytes(zip_path, pdf_name)
    with tempfile.TemporaryDirectory() as tmp:
        images = vision_part.rasterize(dpi)(pdf_bytes, Path(tmp))
        for page, rows in chosen.items():
            if page > len(images):
                check.note(f"page {page} is in the manifest but the part has {len(images)} pages")
                continue
            source = images[page - 1]
            for image, tile, row in rows:
                if image not in opened:
                    opened[image] = Image.open(part_dir / image).convert("L")
                    opened[image].load()
                got = opened[image].crop(_rect(tile))
                want = source.crop(_rect(row))
                check.compared += 1
                if got.tobytes() == want.tobytes():
                    check.identical += 1
                else:
                    check.note(
                        f"{row['name']} differs: composite {image}{_rect(tile)} "
                        f"is not page {page}{_rect(row)}"
                    )
        for handle in opened.values():
            handle.close()
        for handle in images:
            handle.close()
    return check


def _sample(wanted: Dict[int, List[Any]], sample: Optional[int]) -> Dict[int, List[Any]]:
    """A spread of tiles across pages, or all of them.

    Spread rather than the first N: the first N are one page, and a mapping that breaks on the
    second composite would pass. Deterministic, so a failure can be reproduced.
    """
    if sample is None:
        return wanted
    pages = sorted(wanted)
    if not pages:
        return {}
    per_page = max(1, sample // len(pages))
    out: Dict[int, List[Any]] = {}
    taken = 0
    for page in pages:
        rows = wanted[page][:per_page]
        if rows:
            out[page] = rows
            taken += len(rows)
        if taken >= sample:
            break
    return out


def render_report(checks: Sequence[PartCheck]) -> str:
    """What held and what did not, in the order a reader needs it."""
    tiles = sum(c.tiles for c in checks)
    compared = sum(c.compared for c in checks)
    identical = sum(c.identical for c in checks)
    bad = [c for c in checks if not c.ok]

    lines = [
        f"REPACK VERIFICATION over {len(checks)} parts",
        "",
        f"   tiles in manifests        {tiles:,}",
        f"   tiles placed              {sum(c.placements for c in checks):,}",
        f"   pixel-compared            {compared:,}",
        f"   byte-identical            {identical:,}"
        + (f"  ({identical / compared:.1%})" if compared else ""),
        "",
    ]
    if not tiles or not compared:
        lines.append("   NOTHING WAS CHECKED: no tiles, or no pixels compared. This is not a pass.")
        return "\n".join(lines)
    if not bad:
        lines.append("   every check passed: no tile lost, none placed twice, none overlapping,")
        lines.append("   and every compared crop is the same bytes as its source page.")
        return "\n".join(lines)

    lines.append(f"   {len(bad)} parts with problems:")
    for c in bad:
        lines.append(f"      part {c.part_no}: {len(c.problems)} problem(s)")
        for problem in c.problems[:5]:
            lines.append(f"         {problem}")
        if len(c.problems) > 5:
            lines.append(f"         ... and {len(c.problems) - 5} more")
    return "\n".join(lines)
