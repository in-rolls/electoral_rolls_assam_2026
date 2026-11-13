"""Packing the text tighter, because Cloud Vision charges by the image.

A rendered page is 15.5 MP and **30% of it is text**. The rest is photo placeholders, ruled
borders, page margins and the white space under the last line of every box. Vision bills per
image submitted rather than per page, so those pixels are the bill: four pages fit in one image
as rendered, and fourteen fit once the dead space is gone -- $345 for the state against $99.

This is the same idea that just failed for dots.ocr, and it should work here for a different
reason. dots.ocr broke on a 776x70 strip because 11:1 is a shape nothing in its training looks
like: it looped on 76% of crops and switched script on a third. Vision is a detector and
recogniser, not a generative model -- no token cap to run into, no script to hallucinate -- and
it already reads four-page composites. **The risk here is not that it reads worse. It is that a
word comes back attributed to the wrong box**, which is a plausible answer with nothing raised.

So every tile's rectangle in the composite is recorded, and words are assigned by that rectangle
on both axes. Two tiles side by side never share a word even if Vision groups them into one
visual line, because the filter is geometric and not a reading-order guess.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Tuple

#: White space between tiles, in pixels at 400 dpi. Roughly half a line height.
#:
#: Nothing depends on it for correctness -- words are assigned by rectangle, so tiles could abut
#: and still be separated exactly. It is here so the composite still looks like a document to
#: whatever layout analysis Vision runs before recognizing anything.
GUTTER = 24

#: Tiles per row. Three keeps the composite's proportions close to the page it came from, which
#: is a shape Vision is known to read: a single column of 400 boxes would be 776 x 87,000, and an
#: image that thin has never been tested here.
COLUMNS = 3


@dataclass
class Placed:
    """One box's tile, and where it landed in the composite."""

    key: Tuple[int, int, int]
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


def compose(
    entries: Sequence[Tuple[Any, Tuple[int, int, int, int], Tuple[int, int, int]]],
    columns: int = COLUMNS,
    gutter: int = GUTTER,
) -> Tuple[Any, List[Placed]]:
    """Lay tiles out in a grid and return the image with every tile's rectangle.

    ``entries`` is ``(page_image, crop_rect, key)``. The rectangles come from the caller because
    the caller is the one that knows the geometry -- this only does the packing and the
    bookkeeping.

    The offsets are the point. Vision answers in the composite's coordinates, and a word in tile
    200 has to come back to tile 200; that is the same problem as the page-stack offsets one
    level down, and getting it wrong misfiles rows silently.
    """
    from PIL import Image

    if not entries:
        raise ValueError("nothing to compose")

    crops = [(image.crop(rect), key) for image, rect, key in entries]
    cell_w = max(c.width for c, _ in crops)
    rows = (len(crops) + columns - 1) // columns
    row_heights = [
        max(c.height for c, _ in crops[r * columns : (r + 1) * columns]) for r in range(rows)
    ]

    width = columns * cell_w + (columns - 1) * gutter
    height = sum(row_heights) + (rows - 1) * gutter
    canvas = Image.new("L", (width, height), "white")

    placed: List[Placed] = []
    y = 0
    for r in range(rows):
        for c, (crop, key) in enumerate(crops[r * columns : (r + 1) * columns]):
            x = c * (cell_w + gutter)
            canvas.paste(crop, (x, y))
            placed.append(Placed(key, x, y, x + crop.width, y + crop.height))
        y += row_heights[r] + gutter
    return canvas, placed


def words_for(placed: Sequence[Placed], words: Sequence[Any]) -> Dict[Tuple[int, int, int], List]:
    """Every tile's words, in that tile's own coordinates.

    Assignment is by rectangle on **both** axes. Tiles sit side by side, so a word must be
    matched horizontally as well: filtering on height alone would hand a word to all three tiles
    in its row, and each of them would answer with somebody else's name.

    Matched on the word's middle rather than its whole box, because a tall glyph can overhang the
    tile it belongs to and requiring containment would drop it from every tile.
    """
    from .vision import Word

    out: Dict[Tuple[int, int, int], List] = {tile.key: [] for tile in placed}
    for word in words:
        middle_x = (word.left + word.right) / 2
        middle_y = (word.top + word.bottom) / 2
        for tile in placed:
            if tile.left <= middle_x < tile.right and tile.top <= middle_y < tile.bottom:
                out[tile.key].append(
                    Word(
                        word.text,
                        word.left - tile.left,
                        word.top - tile.top,
                        word.right - tile.left,
                        word.bottom - tile.top,
                    )
                )
                break
    return out


def batches(
    entries: Sequence[Tuple[Any, Tuple[int, int, int, int], Any]], budget: int
) -> List[List[Tuple[Any, Tuple[int, int, int, int], Any]]]:
    """Split tiles into composites that each fit under the pixel ceiling.

    Grown one tile at a time and measured with :func:`pixels`, rather than divided by an
    estimated tiles-per-image. A row takes the height of its tallest tile, so the cost of adding
    one is not constant -- the tile that starts a new row adds its whole height to the total.
    An estimate would be right on average and over the ceiling sometimes, and a composite over
    the ceiling is a rejected request rather than a slow one.

    A single tile too large to fit goes alone; it is still submitted, because dropping it would
    silently lose those electors.
    """
    out: List[List[Any]] = []
    current: List[Any] = []
    for entry in entries:
        if current and pixels(current + [entry]) > budget:
            out.append(current)
            current = [entry]
        else:
            current.append(entry)
    if current:
        out.append(current)
    return out


def pixels(entries: Sequence[Tuple[Any, Tuple[int, int, int, int], Any]]) -> int:
    """What a composite of these tiles would cost in pixels, before building it."""
    if not entries:
        return 0
    widths = [rect[2] - rect[0] for _, rect, _ in entries]
    heights = [rect[3] - rect[1] for _, rect, _ in entries]
    rows = (len(entries) + COLUMNS - 1) // COLUMNS
    return (COLUMNS * max(widths) + (COLUMNS - 1) * GUTTER) * (
        sum(sorted(heights, reverse=True)[:rows]) + (rows - 1) * GUTTER
    )
