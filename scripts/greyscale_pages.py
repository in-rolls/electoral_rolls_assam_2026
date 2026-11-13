"""Re-encode rendered pages as grayscale, in place, losslessly.

The source PDFs are bilevel scans. Every page in ``out/pages`` carries R == G == B on every
pixel, so two of the three channels are exact copies and cost a third of the file each. The
renderers now pass ``-gray`` to ``pdftoppm`` so this stops being produced; this is for the
31,486 pages already on disk.

**Lossless is checked, not assumed.** A page whose channels are not identical is left alone and
counted, because converting it would be a real loss rather than a saving. On this corpus nothing
has ever failed that test, which is exactly why it is worth keeping -- the check costs a read
that is happening anyway, and the day it fires is the day the assumption stopped holding.

Written to a temporary file and renamed over the original, so an interruption leaves either the
old page or the new one and never a truncated one.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image


def channels_identical(image: Image.Image) -> bool:
    """Whether dropping to one channel throws nothing away."""
    data = np.asarray(image.convert("RGB"))
    return bool((data[:, :, 0] == data[:, :, 1]).all() and (data[:, :, 1] == data[:, :, 2]).all())


def convert(path: Path) -> int:
    """Bytes saved, 0 if the page was already gray or must not be touched."""
    with Image.open(path) as image:
        if image.mode == "L":
            return 0
        if not channels_identical(image):
            return -1
        grey = image.convert("L")
        before = path.stat().st_size
        temporary = path.with_suffix(".png.tmp")
        grey.save(temporary, format="PNG", optimize=True)
    saved = before - temporary.stat().st_size
    temporary.replace(path)
    return saved


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", nargs="?", default="out/pages")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    # Nice, because this runs beside OCR jobs that are the actual work.
    try:
        os.nice(10)
    except OSError:
        pass

    pages = sorted(Path(args.directory).glob("*.png"))
    if args.limit:
        pages = pages[: args.limit]
    if not pages:
        print(f"no pages under {args.directory}", file=sys.stderr)
        return 1

    saved = done = already = refused = 0
    for number, path in enumerate(pages, start=1):
        result = convert(path)
        if result < 0:
            refused += 1
        elif result == 0:
            already += 1
        else:
            saved += result
            done += 1
        if number % 2000 == 0:
            print(f"   {number:,}/{len(pages):,}  {saved / 1e9:.2f} GB freed", flush=True)

    print(f"\n   converted        {done:,}")
    print(f"   already grey     {already:,}")
    print(f"   left alone       {refused:,}  (channels differ -- converting would lose data)")
    print(f"   freed            {saved / 1e9:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
