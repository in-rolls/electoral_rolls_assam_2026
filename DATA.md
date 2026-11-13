# Dataset documentation

One row per **part** (polling-station area) of the Assam 2026 Final Electoral Roll,
read from page 1 of each part's info page.

Covers **all 126 constituencies — 31,486 parts**.

The roll is printed in three languages, and the language of each constituency is stated in
the publisher's own filenames:

| language | ACs | parts | which |
|---|--:|--:|---|
| Assamese | 112 | 27,683 | 1–112 |
| Bengali | 13 | 3,542 | 114–126 (Barak Valley) |
| English | 1 | 261 | 113 |

Each is read with its own Tesseract model and its own label table, and each is scored
separately (see [Accuracy](#accuracy)). **Every row has the same columns regardless of
language** — the core fields are identical across the three editions.

## Files

| file | grain | rows | size | notes |
|---|---|--:|--:|---|
| `parts.jsonl` | one part | 31,486 | 98 MB | **authoritative** — real types, real nulls, sections nested |
| `parts.csv` | one part | 31,486 | 52 MB | flat, `utf-8-sig` |
| `part_sections.csv` | one section | 47,290 | 4 MB | joins on `(ac_no, part_no)` |
| `report.json` | — | — | — | accuracy figures, overall and per language |
| `review.html` | flagged parts | 185 | — | page image beside extracted values |

Covering **24,958,139 electors** across 126 constituencies.

**`parts.jsonl` is authoritative.** Where it and the CSVs disagree, JSONL is right — CSV
cannot represent `null`, and rounds a missing value and a blank one toward each other.

### Reading it

```python
import json, pandas as pd

parts = [json.loads(line) for line in open("parts.jsonl", encoding="utf-8")]

# or, if you want a frame and accept the CSV caveats:
parts = pd.read_csv("parts.csv", keep_default_na=False, na_values=["NA"])
```

```r
parts <- jsonlite::stream_in(file("parts.jsonl"))
parts <- read.csv("parts.csv", fileEncoding = "UTF-8-BOM")   # NA is R-native
```

`keep_default_na=False` matters in pandas: without it, pandas also treats an empty cell as
missing, which destroys the null/blank distinction described below.

---

## The null convention

This dataset distinguishes two kinds of missing, and the distinction carries information:

| JSONL | CSV | meaning |
|---|---|---|
| `""` | *(empty)* | **the form prints nothing there.** A rural part has no ward number. |
| `null` | `NA` | **the pipeline could not read a value that appears to be there.** |

The call is made mechanically from the ink profile of the value region, not guessed: ink
present but no text recognized → `null`; no ink → `""`.

Why it matters: 729 of 890 parts have `ward_no == ""`. If unread values were also `""`,
a genuine failure would be invisible among them. In this release **0 parts have a null
ward** and **4 have a null `main_town_village`** — those four are real failures and are
identifiable as such.

Derived columns (`ward_no_num`, `ps_address_digits`) inherit the kind of emptiness of
their source. A blank ward yields a blank ward number, not `NA`.

---

## Columns

### Provenance — stitching a row back to its source (18)

Every row carries enough to locate and verify the exact bytes it came from.

| column | example | notes |
|---|---|---|
| `source_zip` | `AC100_ASM Roll Info Pages.zip` | archive filename |
| `source_zip_dir` | `data/ac_info` | where that archive was read from |
| `source_pdf` | `2026-EROLLGEN-S03-100-…-1-WI_INFO.pdf` | member name inside the zip |
| `pdf_sha256` | `c3b34ef4…` | **hash of the source PDF**, not the rendered image |
| `pdf_bytes` | `1170600` | |
| `roll_year` `state_code` `roll_type` `revision_no` `lang` | `2026` `S03` `FinalRoll` `1` `ASM` | parsed from the filename |
| `ac_no_file` `part_no_file` | `100` `1` | **from the filename — authoritative** |
| `page_png` `page_sha256` | `out/pages/100-0001.png` | the rendered page actually read |
| `engine` `engine_version` | `tesseract` `tesseract 5.5.2 (asm=synth20170629)` | recogniser *and* language model |
| `pipeline_version` | `1.0.0` | bumping it invalidates the cache |
| `extracted_at` | `2026-08-04T07:54:52+00:00` | UTC |

To verify a row, using only what the row carries:

```python
import hashlib, zipfile
from pathlib import Path

with zipfile.ZipFile(Path(row["source_zip_dir"]) / row["source_zip"]) as z:
    payload = z.read(row["source_pdf"])
assert hashlib.sha256(payload).hexdigest() == row["pdf_sha256"]
```

`(source_zip, source_pdf, part_no_file)` uniquely identifies every row.

> **`ac_no_file`/`part_no_file` vs `ac_no`/`part_no`.** The `_file` pair comes from the
> publisher's filename and is authoritative — **use it as the key**. The unsuffixed pair
> is what OCR read off the image, kept only as a quality signal. Comparing the two is
> what produces the accuracy figures below.

### Header (7)

`ac_no`, `ac_name`, `ac_reservation`, `pc_no`, `pc_name`, `pc_reservation`, `part_no`

Reservation is normalized to `GENERAL` / `SC` / `ST`.

### Section 1 — revision (6)

`revision_year`, `qualifying_date`, `revision_type`, `publication_date`,
`roll_description`, `mother_roll_year`

Dates are ISO `YYYY-MM-DD`. `roll_description` is the verbatim identification paragraph.

### Section 2 — locality (10)

`main_town_village`, `ward_no`, `post_office`, `police_station`, `gram_panchayat`,
`block`, `revenue_circle`, `subdivision`, `district`, `pin_code`

Verbatim, in the language the constituency is printed in. `ward_no` is blank on rural
parts (see the null convention above).

Eight of these are printed by all three editions. Two are not, and are **blank** (`""`,
not `NA`) where the form does not print them — the absence is a property of the form, not
a failure to read it:

| column | printed by | note |
|---|---|---|
| `gram_panchayat` | Bengali only | `গ্রাম পঞ্চায়েত`, between the police station and the block |
| `subdivision` | English only | after the revenue circle |

The English form labels the revenue circle **Tehsil**. It occupies the identical slot —
between Block and District — in all three editions, so it is stored as `revenue_circle`
rather than as a fourth column, keeping the revenue unit joinable across languages.

### Section 3 — polling station (5)

`ps_no`, `ps_name`, `ps_address`, `ps_type`, `auxiliary_ps_count`

`ps_type` is normalized to `MALE` / `FEMALE` / `GENERAL`. `ps_no` is taken from the
filename's part number, which the form's own numbering follows.

### Section 4 — electors (6)

`start_serial`, `end_serial`, `electors_male`, `electors_female`,
`electors_third_gender`, `electors_total`

`male + female + third_gender == total` holds on **99.89%** of parts and is one of the
four hard checks.

### Derived, in code (2)

`ward_no_num`, `ps_address_digits` — Assamese/Devanagari numerals rewritten as Western
digits. A pure bijection, so it is done deterministically rather than by a model.
`section_name_digits` is the same thing for section names.

### Canonicalised (7)

`district_canonical`, `ac_name_canonical`, `pc_name_canonical`,
`revenue_circle_canonical`, `block_canonical`, `police_station_canonical`,
`post_office_canonical`

Within-AC clustering promotes the dominant spelling of fields that are constant across an
AC. **Written alongside the raw value, never over it.**

> `*_canonical` means *made internally consistent*, not *corrected*. Clustering fixes
> random OCR noise; it cannot fix systematic error. Where every page in an AC shares a
> misreading, the wrong spelling is the mode and is what gets promoted. This release makes
> 4 substitutions with 0 contested clusters.

### Quality (6)

| column | meaning |
|---|---|
| `template_match` | the page matched the expected form layout |
| `checks_passed` / `checks_total` | the denominator varies by row — corpus-consensus checks only run once a mode exists, so `checks_passed` alone is uninterpretable |
| `flags` | `;`-separated names of checks that failed |
| `needs_review` | a **hard** check failed — the row is wrong |
| `anomaly_notes` | why parsing failed, when it did (empty across this release) |

### `part_sections.csv`

`ac_no`, `part_no`, `section_no`, `section_name`, `section_name_digits`

The numbered area list within a part — one-to-many (1 to 9 per part here), which is why it
is a separate table. In `parts.jsonl` it is nested under `sections` instead, so each line
stays a self-contained record.

---

## Accuracy

Every source filename encodes the AC and part number, and the engine never sees the
filename — it reads both off the image. That gives **independently-known ground truth on
every page**, not on a sampled subset. With the elector arithmetic, that is four hard
checks per page.

| check | result |
|---|---|
| Grid detected | **31,486 / 31,486** |
| `ac_no` vs filename | **100.00%** |
| `part_no` vs filename | 99.41% |
| `male + female + third == total` | **100.00%** |
| Rows needing review | **185 / 31,486** (0.59%) |
| Rows with no failed check at all | 30,695 (97.49%) |
| Cost | **$0** |

Per language — each is read with its own Tesseract model, its own derived label table and
its own grid anchors, so each can fail independently:

| language | parts | grid | `ac_no` | `part_no` | elector sum | flagged |
|---|--:|--:|--:|--:|--:|--:|
| Assamese | 27,683 | 100% | 100% | 99.3% | **100%** | 183 |
| Bengali | 3,542 | 100% | 100% | **100%** | **100%** | 1 |
| English | 261 | 100% | 100% | 99.6% | **100%** | 1 |

This breakdown is not decoration. The corpus is 88% Assamese, so a defect confined to
Bengali moves the overall figure by a fraction of a point: at one stage the Bengali elector
sum was **64%** while the corpus-wide number read a healthy 96%. Look at the per-language
table first.

### Field fill rates

100% for `district`, `ac_name` and `total_pages`; 99.9%+ for `revenue_circle`,
`police_station`, `post_office`, `pin_code`, `ps_type`, `ps_name` and `block`; 99.5% for
`main_town_village` and 98.9% for `ps_address`.

`gram_panchayat` (10.1%) and `subdivision` (0.8%) are low **by design** — only the Bengali
and English editions print them. Within their own language they are 89.8% and 100%.

**Fill rate is not accuracy.** A field can be fully populated and wrong; see the
limitations below.

### The 185 flagged rows

All are `part_no` disagreements — the OCR reading of the part number differs from the
filename. `part_no_file`, taken from the filename, is the dataset's key and is correct on
every one of them, so joins and stitching are unaffected. The residual is a hard OCR
effect on particular digit strings, described in
[`docs/ANALYSIS.md`](docs/ANALYSIS.md#11-three-digit-bugs-none-of-which-a-corpus-wide-average-would-have-shown).

---

## Limitations

**Numeric accuracy is measured; text accuracy is not, in any language.** The four checks
above verify numbers on every page. There is no human-labelled gold set for the text, and
cross-engine agreement (86.7% with Surya, on Assamese only) is not accuracy — both engines
can share a failure. Fill rate is not accuracy either: a field can be 100% filled and wrong.

**Bengali and English carry their own unquantified error profiles.** They are read with
`ben` and `eng` traineddata, neither of which has been characterized here the way the
Assamese failures below have. All three models share the same 2017 synthetic vintage
(`synth20170629`) — the English one is no fresher than the Indic ones.

**The Bengali and English label tables are machine-derived.** They were recovered by
consensus across sampled pages rather than typed, at 98–100% agreement with no weak rows,
and are checked in under `assam_rolls/profiles/` with their audit records in
`out/calibration/`. Consensus outvotes random error but cannot detect a *systematic*
misreading of a label. The per-language accuracy table above is the backstop.

**Known text failure modes:**

- **`ঝ` (jha) is systematically misread.** `কোকৰাঝাৰ` (Kokrajhar) → `কোকৰাব্মাৰ`,
  `কোকৰাব্বাৰ`, `কোকৰাবাৰ`. AC10, AC12 and AC100 are 100% internally consistent on
  district; AC1 is 92.2% across 4 variants — and AC1's *modal* spelling is itself wrong.
- **`ৰ` / `র` are confused inconsistently** — Assamese `ra` versus Bengali `ra`.
- **Diacritic ordering** — `গাঁও` often returned as `গাওঁ`. Both engines produce this, so
  it may be faithful to the source; without ground truth it is unresolved.
- **Latin inside Assamese runs** — `(P-1)` becomes `(2-1)`.

**The language model is frozen.** `asm.traineddata` is `4.00.00alpha:asm:synth20170629` —
trained in 2017 on synthetic data, unchanged since. These failures will not improve on
their own.

**No transliteration.** `*_roman` columns are not shipped. OCR cannot produce them, and
shipping them empty would imply data that does not exist.

**Page 2 is not parsed.** Polling-station imagery, GPS and facilities live on page 2,
whose CAD drawings vary by district. Its headers also carry **pre-delimitation** AC
numbers (AC 1 Gossaigaon appears as "28") — when parsed these will be a separate
`ac_no_predelimitation` column, never merged into `ac_no`.

**Source is 144 dpi scanned imagery** with no text layer, so nothing can be cross-checked
against embedded text.

---

## Reproducing

```bash
assam-rolls render        # zips -> out/pages/{ac}-{part}.png
assam-rolls ocr           # pages -> out/cache/*.json   (resumable)
assam-rolls build         # cache -> parts.jsonl + CSVs + report.json
assam-rolls review        # flagged rows -> out/review.html
```

`ocr` is resumable and skips any part whose cached entry still matches its source PDF's
hash — interrupt it freely. A re-issued PDF is re-extracted automatically because its
hash changed. `--overwrite` forces a clean run.

---

## Source and attribution

Derived from the **Election Commission of India's** published 2026 Assam final electoral
roll *info pages* — the two-page cover sheet that opens each part's roll. These are not
the elector lists; no individual elector's details appear anywhere in this dataset.

The extraction code is MIT licensed (see `LICENSE`). The underlying records are a public
government record; this repository reproduces them, it does not originate them.

**Values are reproduced verbatim, including the source's own errors.** Seventeen parts in
AC47 carry the pin code `748113`, which is outside Assam's 78xxxx/79xxxx range — the page
prints it, on every part of that locality, so it is almost certainly a typo in the
published roll. It is recorded as printed and flagged, not silently changed to `784113`.
Correcting it here would mean publishing a number the source does not contain, which is
the opposite of what this dataset is for.

If you find a discrepancy, check it against the source PDF first: every row carries
`source_zip`, `source_pdf` and `pdf_sha256`, which locate and verify the exact bytes it
came from.
