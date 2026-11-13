"""A held-out sample of boxes, cropped and laid out for a person to mark.

Every accuracy figure quoted for the elector fields so far traces back to
``dataset/eval/truth.json`` -- 36 boxes that have been used to tune the parser repeatedly. A set
you tune on stops measuring anything, so the name accuracy of this pipeline is, honestly, not
known. This draws a fresh one.

**Random, never from the disagreements.** ``electors.evaluate`` already carries the scar of a
version that scored the subset where two engines differed, which measures that subset and would
have reported the disagreement rate as the error rate. The sample here is seeded and uniform.

**The whole box, not the tile that goes to Vision.** The pipeline sends only the four labeled
lines; a reviewer needs the header strip too, because the EPIC and the serial live there and
because context is what lets somebody judge a name they cannot otherwise verify.

**Local file, never published.** These are real electors' names, ages and identity numbers.

    python scripts/build_review_sheet.py out/electors/AC010.parquet out/stage1/AC010 \\
        --zip data/keep/AC10_ASM.zip -n 150 --out out/review_sheet.html
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import random
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from assam_rolls import render  # noqa: E402
from electors import crops, grid, output, pages, vision_part  # noqa: E402

#: In the order they matter, which is the order they get looked at.
#:
#: ``relation`` is the father's or husband's name shown with its label -- one mark covers both,
#: because the reviewer is looking at a line that reads "পিতাৰ নাম : ৰাম" and judging the whole
#: of it. Splitting them would double the marking for a distinction the crop already makes plain.
FIELDS = ("name", "relation", "age", "sex", "house_no", "epic_no")


def shown_value(row: Dict[str, Any], field: str) -> Any:
    """What to print beside the crop. Only ``relation`` is assembled rather than read off."""
    # The serial is derived by counting rows, not read, and the OCR that could check it agrees
    # only 42% of the time -- while forming a consecutive run on 2.7% of pages, which is what a
    # 10%-per-box error rate looks like. So the OCR is probably the wrong one, and "probably" is
    # not good enough for a published column: the printed serial sits in the top-left of every
    # box, so a person can settle it.
    if field != "relation":
        return row.get(field)
    name = row.get("relation_name")
    if not name:
        return None
    kind = row.get("relation_type") or "relation?"
    return f"{kind}: {name}"


#: Seeded so the same draw can be rebuilt, and so a second sample can deliberately differ.
SEED = 20260811

STYLE = """
body { font: 15px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       margin: 0; padding: 24px; background: #f6f7f9; color: #16191d; }
h1 { font-size: 20px; margin: 0 0 4px; }
.note { color: #5b6470; margin-bottom: 20px; max-width: 60em; }
.row { background: #fff; border: 1px solid #dfe3e8; border-radius: 8px;
       padding: 14px; margin-bottom: 18px; }
.row img { width: 100%; max-width: 900px; border: 1px solid #dfe3e8; border-radius: 4px;
           display: block; margin-bottom: 10px; image-rendering: crisp-edges; }
.key { color: #5b6470; font-size: 12px; font-family: ui-monospace, Menlo, monospace; }
table { border-collapse: collapse; }
td { padding: 3px 10px 3px 0; vertical-align: middle; }
td.label { color: #5b6470; width: 90px; }
td.value { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 16px;
           min-width: 22em; }
.empty { color: #b00; font-style: italic; font-family: inherit; }
button.mark { border: 1px solid #cfd5dc; background: #fff; border-radius: 5px;
              padding: 2px 11px; cursor: pointer; font-size: 15px; margin-right: 5px; }
button.mark.on-yes { background: #1f7a3d; color: #fff; border-color: #1f7a3d; }
button.mark.on-no  { background: #b00; color: #fff; border-color: #b00; }
#bar { position: sticky; top: 0; background: #16191d; color: #fff; padding: 12px 16px;
       border-radius: 8px; margin-bottom: 18px; display: flex; gap: 18px; align-items: center; }
#bar button { padding: 6px 14px; border-radius: 6px; border: 0; cursor: pointer;
              background: #2f6fdb; color: #fff; font-size: 14px; }
#out { width: 100%; height: 130px; font-family: ui-monospace, Menlo, monospace; font-size: 11px;
       margin-top: 10px; display: none; }
"""

SCRIPT = """
const marks = {};
function mark(btn, key, field, value) {
  marks[key] = marks[key] || {};
  marks[key][field] = value;
  const parent = btn.parentElement;
  for (const b of parent.querySelectorAll('button.mark')) {
    b.classList.remove('on-yes', 'on-no');
  }
  btn.classList.add(value ? 'on-yes' : 'on-no');
  let done = 0;
  for (const k in marks) done += Object.keys(marks[k]).length;
  document.getElementById('done').textContent = done + ' of ' + (TOTAL * FIELDS.length);
}
function emit() {
  const box = document.getElementById('out');
  box.style.display = 'block';
  box.value = JSON.stringify(marks);
  box.select();
}
"""


CROP_IN_SHEET = re.compile(
    r"<img src='(data:image/png;base64,[^']+)'>\s*<div class='key'>[^<]*?&middot; (\S+) &middot;"
)


def reuse_crops(sheet: Path) -> List[Tuple[str, str]]:
    """``(key, data uri)`` for every crop already inlined in a sheet."""
    text = sheet.read_text(encoding="utf-8")
    return [(key, uri) for uri, key in CROP_IN_SHEET.findall(text)]


def full_box(page_image: Any, signature: Any, row: int, column: int) -> Any:
    """The whole elector box, header strip and photo panel included.

    The manifest records the *text tile* -- what the pipeline sends to Vision -- which excludes
    the serial and the EPIC. A reviewer needs those, so the box is rebuilt from the page's own
    geometry rather than padded out from the tile.
    """
    for box in grid.build(signature.h_rules, signature.v_rules, columns=signature.columns):
        if box.row == row and box.column == column:
            return page_image.crop((box.left, box.top, box.right, box.bottom))
    return None


#: Parts to draw from, and boxes from each. Their product is the sample size.
#:
#: A uniform draw of 150 rows from 200,000 lands in about 120 different parts, and a crop cannot
#: be cut without rendering that part's thirty pages at 400 dpi -- two hours of rendering for 150
#: images. Clustering costs precision rather than correctness: boxes within one part share a scan,
#: so they are not independent, and the true interval is perhaps half again as wide as the formula
#: for a simple random sample would say. Thirty clusters is enough that no single bad part can
#: dominate, and it renders in half an hour.
PARTS_DRAWN = 30


def sample(rows: List[Dict[str, Any]], count: int) -> List[Dict[str, Any]]:
    """A two-stage draw: parts at random, then boxes at random within them."""
    rng = random.Random(SEED)
    by_part: Dict[int, List[Dict[str, Any]]] = {}
    for row in rows:
        by_part.setdefault(row["part_no"], []).append(row)

    parts = rng.sample(sorted(by_part), min(PARTS_DRAWN, len(by_part)))
    per_part = max(1, count // len(parts))
    picked: List[Dict[str, Any]] = []
    for part in parts:
        picked.extend(rng.sample(by_part[part], min(per_part, len(by_part[part]))))
    rng.shuffle(picked)
    return picked[:count]


def crop_uri(image: Any) -> str:
    buffer = io.BytesIO()
    # Downscaled: a box is ~1000x415 at 400 dpi and the sheet holds 150 of them. Half size is
    # still comfortably legible and keeps the file openable.
    image.resize((image.width // 2, image.height // 2)).save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.standard_b64encode(buffer.getvalue()).decode("ascii")


def build(picked: List[Tuple[Dict[str, Any], str]], out: Path) -> None:
    parts = [
        "<style>",
        STYLE,
        "</style>",
        f"<script>const TOTAL={len(picked)};const FIELDS={list(FIELDS)!r};</script>".replace(
            "'", '"'
        ),
        "<h1>Elector fields: does the reading match the scan?</h1>",
        "<p class='note'>For each box, tick the fields that are right and cross the ones that "
        "are wrong. A field the pipeline left empty is shown in red -- cross it if the scan "
        "plainly says something, tick it if the box really is blank there. When you are done, "
        "press <b>Copy results</b> and paste them back into the chat.</p>",
        "<div id='bar'><b>Marked:</b> <span id='done'>0</span>"
        "<button onclick='emit()'>Copy results</button></div>",
        "<textarea id='out'></textarea>",
    ]
    for row, uri in picked:
        key = crops.name_for(row["part_no"], row["page_no"], row["box_row"], row["box_col"])
        parts.append("<div class='row'>")
        parts.append(f"<img src='{uri}'>")
        parts.append(
            f"<div class='key'>AC{row['ac_no']} &middot; {key} &middot; "
            f"{html.escape(row.get('roll_section') or '')}</div><table>"
        )
        for field in FIELDS:
            value = shown_value(row, field)
            shown = (
                "<span class='empty'>(empty)</span>"
                if value in (None, "")
                else html.escape(str(value))
            )
            tick = f"mark(this,&quot;{key}&quot;,&quot;{field}&quot;,true)"
            cross = f"mark(this,&quot;{key}&quot;,&quot;{field}&quot;,false)"
            parts.append(
                f"<tr><td class='label'>{field}</td><td class='value'>{shown}</td><td>"
                f"<button class='mark' onclick='{tick}'>&#10003;</button>"
                f"<button class='mark' onclick='{cross}'>&#10007;</button></td></tr>"
            )
        parts.append("</table></div>")
    parts.append(f"<script>{SCRIPT}</script>")
    out.write_text("\n".join(parts), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("shard")
    parser.add_argument("stage1_dir")
    parser.add_argument("--zip", required=True)
    parser.add_argument("-n", type=int, default=150)
    parser.add_argument("--out", default="out/review_sheet.html")
    parser.add_argument(
        "--reuse",
        default="",
        help="an existing sheet to take crops from, so adding a field needs no re-render",
    )
    args = parser.parse_args()

    rows = output.read_shard(Path(args.shard))

    # Rendering is half an hour and the crops are already inlined in a sheet built earlier, so
    # adding a column to the form does not mean rasterizing 30 parts again.
    if args.reuse:
        existing = dict(reuse_crops(Path(args.reuse)))
        keyed = {
            crops.name_for(r["part_no"], r["page_no"], r["box_row"], r["box_col"]): r for r in rows
        }
        picked = [(keyed[k], uri) for k, uri in existing.items() if k in keyed]
        print(f"reused {len(picked)} crops from {args.reuse}")
        out = Path(args.out)
        build(picked, out)
        print(f"{len(picked)} crops -> {out}  ({out.stat().st_size / 1e6:.1f} MB)")
        return 0

    picked_rows = sample(rows, args.n)
    by_part: Dict[int, List[Dict[str, Any]]] = {}
    for row in picked_rows:
        by_part.setdefault(row["part_no"], []).append(row)
    print(f"drew {len(picked_rows)} boxes from {len(rows):,} across {len(by_part)} parts")

    zip_path = Path(args.zip)
    refs = {r.part_no: r.pdf_name for r in render.iter_zip_parts(zip_path)}
    picked: List[Tuple[Dict[str, Any], str]] = []
    for part_no, wanted in sorted(by_part.items()):
        payload = render.read_pdf_bytes(zip_path, refs[part_no])
        with tempfile.TemporaryDirectory() as tmp:
            images = vision_part.rasterize(400)(payload, Path(tmp))
            signatures = pages.recover_partial(
                [pages.classify(image, n) for n, image in enumerate(images, start=1)]
            )
            for row in wanted:
                index = row["page_no"] - 1
                if index >= len(images):
                    continue
                crop = full_box(images[index], signatures[index], row["box_row"], row["box_col"])
                if crop is not None:
                    picked.append((row, crop_uri(crop)))
            for image in images:
                image.close()
        print(f"   part {part_no}: {len(wanted)} boxes")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    build(picked, out)
    print(f"\n{len(picked)} crops -> {out}  ({out.stat().st_size / 1e6:.1f} MB)")
    keys = [
        crops.name_for(r["part_no"], r["page_no"], r["box_row"], r["box_col"]) for r, _ in picked
    ]
    Path("dataset/eval/sample_v2.json").write_text(
        json.dumps({"seed": SEED, "shard": args.shard, "keys": keys}, indent=1), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
