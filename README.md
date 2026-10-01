# Assam 2026 Electoral Rolls

Extracts the Assam 2026 Final Electoral Roll from scanned PDFs into two linked datasets:

- a part-level dataset of polling stations, localities, and published elector counts; and
- a statewide Parquet file containing one row per printed elector box, with source-page
  provenance.

The elector rows join to the authoritative info-page records with
`electors.(ac_no, part_no) = parts.(ac_no_file, part_no_file)`. The filename-derived keys on the
right are deliberate: OCR-read `ac_no` and `part_no` are validation fields, not join keys.

## Get the data

The part-level dataset is in [`dataset/`](dataset/) — **31,486 parts, all 126 constituencies,
24,958,139 published net electors**. Gzipped, it is about 12 MB total; every archive is verified
to decompress byte-identical to what the pipeline wrote.

| file | rows | what it is |
|---|--:|---|
| `parts.jsonl.gz` | 31,486 | **authoritative** — real integers, `null` distinct from `""`, sections nested per part |
| `parts.csv.gz` | 31,486 | the same rows, flat, `utf-8-sig` so Excel renders Assamese |
| `part_sections.csv.gz` | 47,290 | one row per numbered area, joins on `(ac_no, part_no)` |
| `transliteration.csv.gz` | 46,644 | native → romanized lookup for all seven name fields, anchored to India Post where possible ([details](docs/ROMANIZATION.md)) |
| `report.json` | — | accuracy, overall and per language |
| `SHA256SUMS` | — | check a download |

The verified elector-level release artifact is staged for publication:

| file | what it is |
|---|---|
| `assam_electoral_rolls_2026.parquet` | all 126 constituency shards assembled in source order |
| `assam_electoral_rolls_2026.sha256` | SHA-256 checksum |

The permanent download location will be added when `v2026.2` is tagged. The release report records
the artifact's exact byte size and checksum independently of its eventual host.

See the [extraction report](docs/EXTRACTION_REPORT.md) for measured completeness, known error
bounds, and the exact release checks. The per-constituency pipeline and schema are documented in
[docs/ELECTORS.md](docs/ELECTORS.md).

No decompression step needed:

```python
import gzip, json
import pandas as pd

parts = pd.read_csv("dataset/parts.csv.gz", keep_default_na=False, na_values=["NA"])

# or the authoritative form, which keeps types and the null/blank distinction
with gzip.open("dataset/parts.jsonl.gz", "rt", encoding="utf-8") as fh:
    parts = [json.loads(line) for line in fh]
```

```r
parts <- read.csv(gzfile("dataset/parts.csv.gz"), fileEncoding = "UTF-8-BOM")
```

`keep_default_na=False` matters in pandas: without it an empty cell is read as missing,
which destroys the distinction between *the form prints nothing here* (`""`) and *this
could not be read* (`NA`). See [`DATA.md`](DATA.md) for every column, the null convention
and the known limitations, and [Measured results](#measured-results) for accuracy.

## Why this needs OCR

The source PDFs carry **no text layer**. Each page is a single embedded 1187×1679 RGB
image at 144 dpi:

```console
$ pdftotext -layout part.pdf -   # 2 bytes
$ pdffonts part.pdf              # no fonts
$ pdfimages -list part.pdf       # 1 image per page, 1187x1679, 144 dpi
```

So extraction runs OCR over the page image rather than parsing text, with mixed Western
(`2026`, `783350`) and Bengali-Assamese (`৫৯০`, `১`) numerals.

The roll is **not monolingual**. The publisher names the language of every constituency
in its filenames:

| language | ACs | parts | which |
|---|--:|--:|---|
| Assamese | 112 | 27,683 | 1–112 |
| Bengali | 13 | 3,542 | 114–126 (Barak Valley) |
| English | 1 | 261 | 113 |

Each is read with its own Tesseract model, its own grid anchors and its own label table,
derived by `assam-rolls calibrate` from the form's own printed labels. The three editions
are not pixel-identical and do not print quite the same fields — see
[`docs/ANALYSIS.md`](docs/ANALYSIS.md#10-the-roll-is-printed-in-three-languages-and-the-form-is-not-the-same-in-each).

## What gets extracted

Page 1 is a rigid four-section form, stable across rural and urban constituencies:

| Section | Fields |
|---|---|
| Header | AC no/name/reservation, PC no/name/reservation, part no |
| 1 — revision | year, qualifying date, revision type, publication date, roll description, mother roll year |
| 2 — locality | numbered area list, main town/village, ward, post office, police station, block, revenue circle, district, pincode |
| 3 — polling station | number, name, address, type, auxiliary station count |
| 4 — electors | start/end serial, male, female, third gender, total |
| Footer | total pages of the full roll |

Output is two tables:

- **`parts.jsonl`** — one JSON object per part, sections nested inside it.
  **Authoritative**: real integers, `null` distinct from `""`, Assamese unescaped.
- **`parts.csv`** — the same rows, flat and `utf-8-sig` so Excel renders Assamese.
- **`part_sections.csv`** — one row per numbered area within a part (the list is
  one-to-many, so it does not belong in a wide row).

CSV cannot represent `null`, so it writes `NA`; where the two formats disagree, the
JSONL is right. Every row carries the source zip, PDF name and `pdf_sha256` needed to
stitch it back to the exact bytes it came from. See [`DATA.md`](DATA.md).

### Text conventions

Assamese is stored **verbatim**, including Assamese numerals — `ps_address` reads
`৫৯০ বনগাঁও এল. পি. স্কুল, কঠা নং ১`. Each such field has:

- a `*_roman` companion — a model-produced romanization, for joining. Explicitly
  model-generated, and scored separately from the verbatim fields because
  transliteration has no single correct answer.
- where numerals matter, a `*_digits` companion produced **in code**, not by the model:
  Assamese→Western digit mapping is a pure bijection, so it is done deterministically
  (`ward_no_num`, `ps_address_digits`, `section_name_digits`).

The three categorical fields are normalized rather than transliterated:
`ac_reservation`/`pc_reservation` → `GENERAL`/`SC`/`ST`, `ps_type` →
`MALE`/`FEMALE`/`GENERAL`.

## Quality: ground truth on every page, for free

Every source filename encodes the AC and part number:

```
2026-EROLLGEN-S03-100-FinalRoll-Revision1-ASM-45-WI_INFO.pdf
                  └── AC 100                     └── part 45
```

The model **never sees the filename** — it reads both numbers off the image. Comparing
the two gives a true accuracy measurement on *every page in the corpus*, not just on a
sampled gold set. Together with the elector arithmetic
(`male + female + third_gender == total`), that is four independent hard checks per
page.

Checks are **hard** (a failure means the row is wrong) or **soft** (suspicious but
legitimately possible). Only hard failures — plus an explicit low-confidence or
off-template self-report from the model — set `needs_review`.

`assam-rolls review` renders flagged rows as an HTML sheet with the page image beside
the extracted values.

## Install

Requires [poppler](https://poppler.freedesktop.org/), Tesseract 5 with the Assamese,
Bengali and English language models, and Python 3.10+.

```bash
brew install poppler tesseract      # Debian: apt-get install poppler-utils tesseract-ocr

for lang in asm ben eng; do
  curl -L -o "$(brew --prefix)/share/tessdata/$lang.traineddata" \
    "https://github.com/tesseract-ocr/tessdata_best/raw/main/$lang.traineddata"
done

make install                        # uv venv + editable install with dev extras
```

No API key is required.

## Run the part-level extraction

```bash
assam-rolls calibrate     # derive the Bengali and English label tables (once)
assam-rolls render        # zips  -> out/pages/{ac}-{part}.png
assam-rolls ocr           # pages -> out/cache/*.json          (local, free)
assam-rolls build         # cache -> parts.jsonl + CSVs + report.json
assam-rolls review        # flagged rows -> out/review.html
```

`render` and `ocr` both run across a process pool.

Every stage is resumable and idempotent, keyed on `{ac:03d}-{part:04d}`. `ocr` skips any
part whose cached result still matches its source PDF's hash, so an interrupted run
resumes where it stopped, and a re-issued PDF is re-extracted automatically. Pass
`--overwrite` to force a clean run, and `--log-file` for a durable record.

**No API key is needed for this part-level flow.** The elector pipeline uses Google Cloud Vision
for the text inside elector boxes; downloading either published dataset requires no credentials.
The Claude path in `extract.py` is retained but not wired into the default flow — see
[why it was abandoned](docs/ANALYSIS.md#8-why-the-claude-api-path-was-measured-and-then-abandoned).

## Development

```bash
make test        # pytest
make lint        # black --check, isort --check-only, flake8
make ci          # both
```

## Measured results

Full run over all **126 constituencies — 31,486 parts, 47,290 area entries, 24,958,139
electors** — Tesseract, 9 workers, **no API calls**:

| Metric | Result |
|---|---|
| Grid detected | **31,486 / 31,486** |
| `ac_no` vs filename | **100.00%** |
| `part_no` vs filename | 99.41% |
| `male + female + third == total` | **100.00%** |
| Rows needing review | **185 / 31,486** (0.59%) |
| Rows with no failed check at all | 30,695 (97.49%) |
| Unread numeric fields | **0** |
| Cost | **$0** |

Per language, since each has its own model, label table and grid anchors:

| language | parts | grid | `ac_no` | `part_no` | elector sum | flagged |
|---|--:|--:|--:|--:|--:|--:|
| Assamese | 27,683 | 100% | 100% | 99.3% | **100%** | 183 |
| Bengali | 3,542 | 100% | 100% | **100%** | **100%** | 1 |
| English | 261 | 100% | 100% | 99.6% | **100%** | 1 |

The corpus is 88% Assamese, so a defect confined to one of the others barely moves the
overall figure — at one point Bengali's elector sum was 64% while the corpus-wide number
read 96%. `report.json` carries the breakdown, and it is the number to read first.

Field fill rates are 100% for `district`, `ac_name` and `total_pages`, and 99.9%+ for
`revenue_circle`, `police_station`, `post_office`, `pin_code`, `ps_type`, `ps_name` and
`block`. `gram_panchayat` and `subdivision` are low overall by design — only the Bengali
and English editions print them.

`scripts/verify_dataset.py` re-reads the shipped files and checks coverage, provenance, a
stitching round trip back to each source PDF's bytes, encoding, the null convention and
the arithmetic. It passes.

### Where it still gets things wrong

Fill rate is not accuracy. Known failure modes, all in the Assamese text:

- **`ঝ` (jha) is systematically misread.** `কোকৰাঝাৰ` (Kokrajhar) comes out as
  `কোকৰাব্মাৰ`, `কোকৰাব্বাৰ` or `কোকৰাবাৰ`. Since every part of an AC shares a district,
  spelling variance within an AC is a free consistency probe: AC10, AC12 and AC100 are
  100% internally consistent, AC1 is 92.2% across 4 variants. **Consistency is not
  correctness** — AC1's *modal* spelling is itself wrong.
- **Diacritic ordering** — `গাঁও` is often returned as `গাওঁ`.
- **Latin letters inside Assamese runs** — `(P-1)` becomes `(2-1)`.

Numeric fields are verified on every page by the checks above; the text fields are not,
and their accuracy is not yet established. That is what the engine bake-off and the gold
set are for.

## Documentation

- [`DATA.md`](DATA.md) — every column, the null convention, accuracy, and limitations.
- [`docs/ANALYSIS.md`](docs/ANALYSIS.md) — how the pipeline was arrived at: grid
  detection, the digit-script trap, per-region upscaling, dictionary thresholds, the
  Tesseract-vs-Surya bake-off, and the cost analysis that ruled out the API.

## Status

Page 1 is done and its numerics are provably correct on the corpus at hand. The engine
bake-off is settled — **Tesseract**, on error *shape* rather than raw agreement. What
remains is a human-labelled gold set: Assamese **text** accuracy is still unmeasured.

Page 2 — polling-station imagery, GPS, facilities — is deliberately sequenced **after**
page 1 ships; its CAD drawings are non-standard across districts and need a coverage
survey before a schema can be fixed.

## Known limitations

- Source is 144 dpi scanned imagery. There is no text layer to cross-check against, so
  accuracy is measured, not assumed.
- `*_roman` fields are model-generated and unverified; the verbatim column is the record.
- Page-2 CAD drawings vary by district: GPS coordinates are present in only some, and
  the facilities table has several incompatible schemas. Coverage will be reported
  rather than assumed.
- CAD headers carry **pre-delimitation** AC numbers (AC 1 Gossaigaon appears as "28").
  When page 2 is parsed these are captured as a separate `ac_no_predelimitation`
  column, never merged into `ac_no`.

## Related

- [`in-rolls/parse_unsearchable_rolls`](https://github.com/in-rolls/parse_unsearchable_rolls) — shared roll-parsing core
- [`in-rolls/electoral_rolls`](https://github.com/in-rolls/electoral_rolls) — scrapers
- [`in-rolls/savitr`](https://github.com/in-rolls/savitr) — Surya OCR on Apple Silicon

## Maintenance

This is a point-in-time data collection; see the [shared maintenance policy](https://github.com/soodoku/data-repos#maintenance-policy). Run the affected parser tests when code changes and the relevant data validators when inputs or outputs change. Full-data checks and publication are explicit operations. Routine edits do not require hosted CI, Docker, a Python-version matrix, Preen or pre-commit.
