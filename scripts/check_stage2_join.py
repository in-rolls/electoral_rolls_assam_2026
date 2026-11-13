"""Prove stage two files every word in the box it came from, without spending anything.

The pixel check in ``electors.verify`` proves the *image* is right: every tile in a composite is
byte-identical to its source page. It says nothing about what happens after Vision answers, and
that is where the remaining risk is -- Vision replies in the composite's coordinates, and a word
in tile 200 has to come back to page 14, box (3, 1).

So this feeds stage two a synthetic reply instead of a real one: for every tile, a full elector
box whose name is that tile's own position. If the join is right, the row for page 14 box (3, 1)
reads back the name minted for page 14 box (3, 1) and no other. A mapping that is off by one tile,
or that hands a word to all three tiles in its row, fails on almost every box rather than subtly.

Free, exact, and it runs over every tile rather than a sample.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from electors import stage2  # noqa: E402

#: Ten consonants standing in for the ten digits. The first version wrote the position as digits
#: and every marker came back as "কখগ" -- the name parser strips digits, correctly, because a name
#: does not contain them. None of these is a RA, so the Bengali/Assamese RA folding cannot touch
#: them either.
DIGIT_LETTERS = "কখগঘঙচছজঝঞ"


def minted(key) -> str:
    """A name no OCR would produce, so a row carrying it can only have got it from its own tile."""
    return "".join(DIGIT_LETTERS[int(c)] for c in f"{key[0]:03d}{key[1]:02d}{key[2]:01d}")


def _word(text: str, left: int, top: int, right: int, bottom: int) -> Dict[str, Any]:
    box = [
        {"x": left, "y": top},
        {"x": right, "y": top},
        {"x": right, "y": bottom},
        {"x": left, "y": bottom},
    ]
    return {"symbols": [{"text": c} for c in text], "boundingBox": {"vertices": box}}


def reply_for(placed: Sequence[Any]) -> List[Dict[str, Any]]:
    """One synthetic Vision response: four labeled lines inside each tile's rectangle."""
    words: List[Dict[str, Any]] = []
    for tile in placed:
        lines = [
            f"নাম : {minted(tile.key)}",
            "পিতাৰ নাম : ৰাম",
            "ঘৰ নং : 12",
            "বয়স : 45 লিঙ্গ : পুৰুষ",
        ]
        step = max(1, tile.height // (len(lines) + 1))
        for n, line in enumerate(lines, start=1):
            y = tile.top + n * step
            x = tile.left + 4
            for piece in line.split(" "):
                if not piece:
                    continue
                width = max(6, 10 * len(piece))
                words.append(_word(piece, x, y - step // 3, x + width, y + step // 3))
                x += width + 6
    return [{"fullTextAnnotation": {"pages": [{"blocks": [{"paragraphs": [{"words": words}]}]}]}}]


def main(stage_dir: Path) -> int:
    parts = sorted(p for p in stage_dir.glob("part*") if stage2.ready(p))
    if not parts:
        print(f"no prepared parts under {stage_dir}", file=sys.stderr)
        return 1

    total = right = 0
    wrong: List[str] = []
    for part_dir in parts:
        placements = stage2.placements_of(part_dir)
        expected = {tile.key: minted(tile.key) for tiles in placements.values() for tile in tiles}

        # Keyed on the image being annotated, so a response can only reach the composite it was
        # minted for -- otherwise this would pass even if stage two paired them up at random.
        by_size = {}
        for name, tiles in placements.items():
            by_size[name] = reply_for(tiles)
        order = list(placements)
        seen = {"n": 0}

        def annotate(_images, order=order, by_size=by_size, seen=seen):
            reply = by_size[order[seen["n"]]]
            seen["n"] += 1
            return reply

        result = stage2.read_part(part_dir, api_key="", annotate=annotate)
        if result.error:
            print(f"  part {part_dir.name}: {result.error}")
            return 1

        for row in result.electors:
            key = (row["page_no"], row["box_row"], row["box_col"])
            total += 1
            if row.get("name") == expected.get(key):
                right += 1
            elif len(wrong) < 8:
                wrong.append(f"{key} read {row.get('name')!r}, expected {expected.get(key)!r}")
        print(
            f"  {part_dir.name}  {len(result.electors):>4} rows  "
            f"{len(expected):>4} tiles  images {result.images_billed}"
        )

    print()
    print(f"  rows                 {total:,}")
    print(f"  name matches own tile {right:,}  ({right / total:.1%})" if total else "  no rows")
    for line in wrong:
        print(f"     {line}")
    return 0 if total and right == total else 1


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1])))
