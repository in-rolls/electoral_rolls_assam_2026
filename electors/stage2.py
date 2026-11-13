"""Stage two: the only step that costs money, and nothing else.

Stage one already rendered the part, found the boxes, read the EPICs and packed the text into
composites. This stage submits those composites to Cloud Vision and puts the words back in the
rows they came from. It never opens the source PDF.

That is the whole point of the split. ``vision_part.read_repacked`` does both halves in one pass,
which means re-running Vision -- after a parser change, or with a second engine -- re-renders 121
pages of PDF and re-runs tesseract over every box to get back to pixels that are already sitting
on disk. Here the composites are the input, so a second reading costs the API call and nothing
else.

The join is the load-bearing part. Vision answers in each **composite's** coordinates, and a word
in tile 200 has to come back to box 200 of page 14. ``placements.json`` holds every tile's
rectangle and ``repack.words_for`` assigns by that rectangle on both axes, so two tiles side by
side never share a word even when Vision groups them into one visual line. That the rectangles
are right is not assumed: ``electors.verify`` compares every tile's pixels against the source
page, and on the first five parts of AC1 all 2,964 were byte-identical.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from assam_rolls import schema

from . import crops, extract, repack, stage1, vision, vision_part


def composites(part_dir: Path) -> List[Path]:
    """The images stage one wrote, named from its own record of them.

    Named, not necessarily present: a part resumed from the bucket has the words these images were
    read into but not the images. ``read_part`` opens the file only when the words are missing, so
    the path is a key here more often than it is a file.
    """
    return [part_dir / name for name in stage1.image_names(part_dir)]


def placements_of(part_dir: Path) -> Dict[str, List[repack.Placed]]:
    """Each composite's tiles, back as the objects the mapper expects.

    Rebuilt rather than re-derived. Re-deriving the layout here would let stage two disagree with
    the image stage one actually produced -- and the disagreement would look like a reading error,
    not a geometry error, because the words would come back plausible and misfiled.
    """
    raw = json.loads((part_dir / "placements.json").read_text(encoding="utf-8"))
    return {
        image: [
            repack.Placed(
                key=(tile["page"], tile["box_row"], tile["box_col"]),
                left=tile["left"],
                top=tile["top"],
                right=tile["right"],
                bottom=tile["bottom"],
            )
            for tile in tiles
        ]
        for image, tiles in raw.items()
    }


def identity(part_dir: Path) -> Dict[str, Any]:
    """Which part this is, taken from the manifest rather than from the directory name.

    A directory can be moved or renamed; the manifest carries the source filename and its sha, so
    the identity travels with the data and a row can still be traced to exact source bytes.
    """
    manifest = crops.read_manifest(part_dir)
    if not manifest:
        raise ValueError(f"{part_dir} has an empty manifest")
    row = next(iter(manifest.values()))
    return {
        "ac_no": row["ac_no"],
        "part_no": row["part_no"],
        "source_zip": row.get("source_zip", ""),
        "source_pdf": row["source_pdf"],
        "pdf_sha256": row["pdf_sha256"],
    }


def read_part(
    part_dir: Path,
    api_key: str,
    annotate: Optional[Callable[[Sequence[Any]], Sequence[Any]]] = None,
) -> extract.PartResult:
    """One prepared part through Vision, as rows.

    ``annotate`` is injectable so the mapping can be tested without spending anything: the join
    is the part that can be wrong, and it should not take an API key to check it.
    """
    who = identity(part_dir)
    meta = schema.parse_source_filename(who["source_pdf"]) or {}
    result = extract.PartResult(
        ac_no=who["ac_no"],
        part_no=who["part_no"],
        lang=meta.get("lang", ""),
        source_zip=who["source_zip"],
        source_pdf=who["source_pdf"],
        pdf_sha256=who["pdf_sha256"],
    )
    try:
        side = stage1.read_side(part_dir / "side.json")
        result.unknown_pages = list(side["unknown"])
        result.page_count = len(side["pages"]) + len(side["unknown"])
        if side["summary"] is None:
            # Bought only where the free reading failed, and only once: the words are cached
            # beside the composites', so a later parser change re-reads them for nothing.
            bought, paid = summary_from_vision(part_dir, api_key, annotate)
            result.images_billed += int(paid)
            if bought is not None:
                side["summary"] = bought
                # Written back, because reconciliation reads the closing total from side.json
                # and not from this result. Without this the purchase happened, the number was
                # correct, and it was thrown away -- AC100 bought 37 readings and still reported
                # 80% of its parts unmeasured.
                stage1.write_side(part_dir / "side.json", side)
        if side["summary"] is not None:
            closing = side["summary"]
            result.summary_male = closing.male
            result.summary_female = closing.female
            result.summary_third = closing.third
            result.summary_total = closing.total

        placements = placements_of(part_dir)
        words: Dict[Tuple[int, int, int], List[vision.Word]] = {}
        for path in composites(part_dir):
            placed = placements.get(path.name)
            if placed is None:
                raise ValueError(f"{path.name} has no placements -- cannot attribute its words")
            found = cached_words(part_dir, path.name)
            if found is None:
                found = _buy_words(part_dir, path, api_key, annotate)
                result.images_billed += 1
            words.update(repack.words_for(placed, found))

        vision_part._fill_repacked(result, side, words, meta)
    except (OSError, RuntimeError, ValueError) as exc:
        result.error = f"{type(exc).__name__}: {exc}"
    return result


def summary_from_vision(
    part_dir: Path, api_key: str, annotate: Optional[Callable] = None
) -> Tuple[Optional[Any], bool]:
    """The closing totals for a part tesseract could not read, bought from Vision.

    About a fifth of parts end up with no printed total, so a fifth of the dataset has **no
    check on it at all** -- the rows are extracted and nothing says whether the right number of
    them came out. The cause is not a hard page: it is that the closing row's description cell
    wraps across four lines, and ``summary.parse`` wants a balancing triple on one line. The
    numbers are there; tesseract reconstructs the line without them.

    One image a part, on a fifth of parts: about $9 for the state, against $148 already spent on
    the boxes. Cheap for taking validation from four parts in five to all of them.

    The arithmetic still decides. A triple that does not balance is refused here exactly as it is
    on the tesseract path, so a bought reading can no more replace a right answer with a
    plausible wrong one than a free one could.

    Returns the totals and **whether Vision was actually charged for them**. On the cached path
    nothing is bought, and a caller that counts the read anyway reports a cost that was never
    incurred -- one phantom image per resumed part, which is the whole state once the fleet
    starts resuming rather than re-rendering.
    """
    from . import summary as summary_module

    page = part_dir / "summary_page.png"
    cached = cached_words(part_dir, page.name)
    paid = cached is None
    if cached is None:
        if not page.exists():
            return None, False
        from PIL import Image

        with Image.open(page) as image:
            image.load()
            width = image.width
            responses = (
                annotate([image]) if annotate is not None else vision.annotate([image], api_key)
            )
        if not responses:
            return None, False
        cached = vision.words_from(responses[0])
        save_words(part_dir, page.name, cached)
    else:
        width = max((word.right for word in cached), default=0) + 1

    text = "\n".join(vision.grouped_lines(cached, 0, width))
    found = summary_module.parse(text)
    if not found:
        return None, paid
    male, female, third, total = found
    return (
        summary_module.RollSummary(male=male, female=female, third=third, total=total, scale=1),
        paid,
    )


#: Re-asks for one image whose answer came back empty or errored. Separate from the transport
#: retries inside ``vision.annotate``: those cover the connection, these cover the *answer*.
#:
#: Five, not three. AC43 died on "Backend deadline exceeded." outlasting three attempts over six
#: seconds -- a backend having a bad minute needs to be waited out in minutes, and the exponential
#: backoff makes the extra attempts nearly free when the answer comes sooner.
IMAGE_ATTEMPTS = 5


def _buy_words(
    part_dir: Path, path: Path, api_key: str, annotate: Optional[Callable]
) -> List[vision.Word]:
    """One composite through Vision, insisting on an actual answer.

    The batch endpoint reports per-image failure inside an HTTP 200: the response carries an
    ``error`` object and no annotation. ``annotate`` does not raise for it, ``words_from`` finds
    nothing to walk, and the failure used to come out indistinguishable from a blank image --
    which a composite cannot be, since it is a stack of text lines cut from elector boxes.

    One night of that cost twenty-two constituencies: each had parts come up hundreds of rows
    short, each was correctly refused at the gate, and each was parked as failed. The rows were
    never wrong; the reads had failed and nothing said so.

    So an errored or empty answer is retried with backoff -- the plausible causes are transient,
    rate limits and internal errors -- and if it still will not answer, the part fails loudly and
    takes its constituency to a retry, rather than publishing short.
    """
    from PIL import Image

    last = ""
    for attempt in range(1, IMAGE_ATTEMPTS + 1):
        with Image.open(path) as image:
            image.load()
            responses = (
                annotate([image])
                if annotate is not None
                else vision.annotate([image], api_key=api_key)
            )
        if not responses:
            last = "no response"
        else:
            problem = responses[0].get("error")
            if problem:
                last = f"per-image error: {problem.get('message', problem)}"
            else:
                found = vision.words_from(responses[0])
                if found:
                    save_words(part_dir, path.name, found)
                    return found
                last = "no words in a non-blank composite"
        if attempt < IMAGE_ATTEMPTS:
            time.sleep(2**attempt)
    raise RuntimeError(f"vision gave no usable answer for {path.name} ({last})")


def words_path(part_dir: Path, image: str) -> Path:
    return part_dir / f"{Path(image).stem}.words.json"


def save_words(part_dir: Path, image: str, found: Sequence[vision.Word]) -> None:
    """Keep what Vision said, so saying it again never has to be bought twice.

    Rows were the only thing kept, and rows are a *parse* of the answer rather than the answer.
    Every improvement to the parser therefore cost the whole API bill again: the EPIC fix that
    took well-formed reads from 62% to 99% would have meant re-buying every image in the state
    to apply it. Words are the last point at which the money has been spent and no judgment has
    yet been made, so they are what to keep.

    About 6,000 words a composite, which is a megabyte or so -- next to nothing beside the
    composite itself, and a tiny fraction of what re-reading it would cost.

    **An empty answer is not saved.** A composite is a stack of text lines cut from elector boxes;
    there is no such thing as one with no words on it, so an empty result is a failed read rather
    than a finding. Cached, it becomes permanent: AC33 holds 214 of them across 110 parts, and
    every subsequent run served those parts from a "successful" read of nothing. It lost 55,359
    rows and each retry reproduced the loss exactly, because the failure had been written down as
    the answer.
    """
    if not found:
        return
    temporary = words_path(part_dir, image).with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(
            [[w.text, w.left, w.top, w.right, w.bottom] for w in found],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    temporary.replace(words_path(part_dir, image))


def cached_words(part_dir: Path, image: str) -> Optional[List[vision.Word]]:
    """What Vision said last time, or None if this image has never been usefully read.

    An empty file counts as never read, which is what heals the ones already written. They cannot
    be a real answer -- see :func:`save_words` -- and leaving them in place makes the loss
    permanent and invisible, since a part reading zero words reports no error at all.
    """
    path = words_path(part_dir, image)
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        # A file that will not parse is a read that did not finish. AC20 part 24 holds a
        # zero-length one, and raising here killed the constituency from inside the branch whose
        # job is to decide whether the image needs reading again.
        return None
    found = [vision.Word(text, left, top, right, bottom) for text, left, top, right, bottom in raw]
    return found or None


def ready(part_dir: Path) -> bool:
    """Whether stage one left this part complete enough to read."""
    return (
        (part_dir / crops.MANIFEST).exists()
        and (part_dir / "placements.json").exists()
        and (part_dir / "side.json").exists()
        and stage1.readable(part_dir)
    )
