# Transliteration

`dataset/transliteration.csv.gz` maps every native-script value in **all seven** name
fields to a romanization. It is a **lookup table, not a model**: under a megabyte, plain
CSV, editable in a spreadsheet, joined onto the dataset in one merge.

| tier | fields | distinct values |
|---|---|--:|
| 1 | `district`, `revenue_circle`, `police_station`, `block`, `post_office` | 2,817 |
| 2 | `main_town_village`, `ps_name` | 43,827 |

**Transliteration, never translation.** `গোসাইগাও ৰাজহ চক্ৰ` is *Gossaigaon Rajah Chakra*,
not *Gossaigaon Revenue Circle*. A check over the whole table enforces this.

```python
import csv, gzip
lut = {(r["field"], r["native"]): r["roman"]
       for r in csv.DictReader(gzip.open("dataset/transliteration.csv.gz", "rt", encoding="utf-8"))}
lut[("district", "কোকৰাব্মাৰ")]        # -> 'Kokrajhar'
```

| column | meaning |
|---|---|
| `roman` | the romanization to use |
| `variant` | a **competing spelling** an official source displaced — present on 568 rows |
| `source` | how `roman` was produced |
| `authority` | set when an official gazetteer produced **or independently confirmed** it |

`authority` is worth reading separately from `source`. "Nobody has checked this" and "India
Post spells it the same way" are very different claims, and only one of them is visible if
agreement is recorded silently. **25.7% of row-weight** carries one.

`variant` is only populated where the displaced spelling had a human behind it. An earlier
version filled it from whatever happened to be there, so 617 of 1,186 variants were rejected
machine guesses published as if they were alternative English spellings — 113 still carrying
the `x`/`aa` artifacts this pipeline calls wrong. A guess an authority overruled is not a
variant; it is in `out/indicxlit_words.json` for anyone who wants it.

## Why a table and not a model

No free offline tool produces conventional English spellings reliably. Measured on twelve
well-known Assam districts, top-1 exact:

| tool | exact |
|---|--:|
| **IndicXlit** (AI4Bharat, neural) | **5/12** |
| Aksharamukha `RomanColloquial` | 2/12 |
| Aksharamukha `RomanReadable` / IAST / ISO | 0–2/12 |

IndicXlit wins, but its misses are *systematic*: it renders Assamese **phonetics** —
`xunitpur` because শ is /x/, `zurhat` because যো is /z/, `dixpur` for Dispur — where
conventional spellings are older anglicisations. It optimizes a different target, so its
output seeds a reviewable table rather than shipping directly.

`indicate` was tried and is **not broken** — it has no Assamese weights. Fed Bengali script
its Hindi model errors per character and returns `""`. On the script it *was* trained for it
gives `kokarajhar` and `jorhat`, better anglicisation than IndicXlit manages on Assamese, so
training `asm` weights on Aksharantar looks worth doing. `fetch_aksharantar.py` already maps
`asm`.

LLMs and hosted APIs (Bhashini/ULCA) were out of scope: free and offline only.

## What a table can do that no transliterator can

`district` holds 41 native spellings for 35 real districts, because `ঝ` is misread
systematically:

| native | rows | IndicXlit | table |
|---|--:|---|---|
| কোকৰাব্মাৰ | 862 | `kukorabmar` | **Kokrajhar** |
| কোকৰাব্াৰ | 33 | `kukorabrar` | **Kokrajhar** |
| কোকৰাব্বাৰ | 29 | `kukorabbar` | **Kokrajhar** |
| কোকৰাবাৰ | 20 | `kukorabar` | **Kokrajhar** |

**944 rows** now name their district correctly. No transliterator could do
this — the `ঝ` is gone from the source. Only a lookup can.

The verbatim column is **not** patched. Every row carries `pdf_sha256` so a reader can check
it against the source page; editing the text would make that provenance a lie. The table is
the same repair as a third column — derived, reversible, checkable.

## Anchored to an official gazetteer

Completeness was never the problem; **correctness** was. Scored against the 725 hand-checked
tokens the model also saw, raw IndicXlit is exactly right **29.4%** of the time. Most
unreviewed spellings were plausible rather than right, and nothing in the pipeline could tell
the two apart.

What fixed that was a join key the corpus already carried and nothing had used: **`pin_code`,
present on 100% of rows**, 554 distinct, with 1,408 of 1,420 post offices mapping to exactly
one pin. India Post publishes the official English name of every office under a pincode, so
the question stops being *"what does this say"* and becomes *"which of these twelve is it"* —
which edit distance can actually answer.

3,917 official post offices were fetched for 547 of the 554 pincodes. Matching is **always
scoped**: a post office against the offices sharing its pincode, a block against the blocks
of its district. Nothing is ever compared against the whole gazetteer, because at that size
a nearest match is meaningless.

**25.7% of row-weight is now backed by an official source** — 1,186 values corrected and
561 independently confirmed.

### Where the threshold came from, measured honestly

A first version of this section swept the threshold and reported its score on **the same
values it was chosen from** — an in-sample number presented as if it were a validation.

The split that was missing: gold restricted to values where every token is hand-checked
(1,396 of them), shuffled, swept on half A, scored on **held-out half B**.

| threshold | margin | A applied | A precision | B applied | **B precision** |
|--:|--:|--:|--:|--:|--:|
| 0.85 | 0.05 | 331 | 85.5% | 336 | 87.2% |
| 0.88 | 0.05 | 302 | 93.4% | 310 | 94.2% |
| **0.90** | **0.05** | **301** | **93.7%** | **305** | **95.7%** |
| 0.92 | 0.05 | 291 | 96.9% | 301 | 97.0% |

Held-out precision is *higher* than in-sample, so the threshold is not overfit — the
**reporting** was. 0.90 is where the last *wrong place* disappears: at 0.85 the matcher
confidently turned `ধমধমা` into **Nizdhamdhama** and `মধুপুৰ` into **Madhapur**, different
places applied without hesitation. Every disagreement surviving at 0.90 is the same place
spelled differently, which is the point rather than a defect.

### Official versus conventional

These are real conflicts, not errors. India Post wins and the displaced spelling is kept in
`variant`, because "official" is genuinely not a single thing: the district administration
writes *Sivasagar* where the Department of Posts writes *Sibsagar*.

| native | conventional (`variant`) | official (`roman`) | rows |
|---|---|---|--:|
| `কামৰুপ (মহানগৰ)` | Kamrup Metropolitan | **Kamrup (Mahanagar)** | 1,218 |
| `তামুলপুৰ` | Tamulpur | **Tambulpur** | 482 |
| `পশ্চিম কাৰ্বি আংলং` | West Karbi Anglong | **Paschim Karbi Anglong** | 358 |
| `চিদলা-চৰাং (অংশ-১)` | Sidli-Chirang (Ansh-1) | **Sidli Chirang (Ansh-1)** | 214 |
| `ডুমডুমা` | Doomdooma | **Doom Dooma** | 173 |
| `নিলামবাজার` | Nilambazar | **Nilam Bazar** | 167 |
| `লাহাৰঘাচ (অংশ-২)` | Laharighat (Ansh-2) | **Lahorighat (Ansh-2)** | 128 |
| `হয়বৰগাও অংশ` | Hoiborgaon Ansh | **Haiborgaon Ansh** | 124 |
| `ভৱানাপুৰ` | Bhawanipur | **Bhowanipur** | 122 |
| `চাপৰ-শালকোচা` | Chapar-Salkocha | **Chapar Salkocha** | 116 |
| `।বজনা` | ।Bijni | **Bijni** | 108 |
| `মুছলপুৰ` | Musalpur | **Mussalpur** | 101 |
| `উধারবন্দ` | Udharbond | **Udarbond** | 81 |
| `কাকপথাৰ` | Kakopathar | **Kakapathar** | 79 |
| `আমবাগান` | aambagan | **Ambagan** | 78 |
| `বিহগুৰী পি-1` | bihguri P-1 | **Bihaguri P-1** | 75 |

### Model artifacts, repaired

IndicXlit writes Assamese phonetics. Two habits are wrong in every English context this table
will be joined against, and both are deterministic: of 1,831 cached tokens whose output holds
an `x`, **1,830** have শ, ষ or স in the source.

| repair | occurrences | gold-set effect |
|---|--:|--:|
| `x` → `s` (`xalbari` → `salbari`) | 8,650 | |
| `aa` → `a` (`aambari` → `ambari`) | 4,928 | |
| **both** | | **29.4% → 33.0%** |

`z` → `j` was measured and **rejected**: it scored −0.3 points, corrupting as many tokens as
it fixed. A rule that fails its own measurement does not ship.

## Review status

The table is complete. It is **not** fully reviewed, and the `source` column says so per row:

| provenance | meaning | entries | row-weight |
|---|---|--:|--:|
| `lexicon+indicxlit` | some tokens hand-checked, rest from the model | 33,121 | 21.2% |
| `indicxlit` | model output, unreviewed | 7,027 | 15.6% |
| `lexicon` | every token hand-checked | 4,749 | 43.4% |
| `indiapost` | **matched to India Post's official name** | 1,185 | 5.6% |
| `already-latin` | source already Latin, passed through | 524 | 0.8% |
| `manual` | written by hand | 38 | 13.4% |

**63.2% of row-weight is hand-checked, officially matched, or already Latin**, and
**77.2% of token-occurrences are hand-checked**. The second number is higher because a
value can be nine parts reviewed and one part not, and the first counts that as unreviewed.
Both are in the file: `source` and `authority` per row, and the lexicon itself.

Capitalization happens to make this legible. Reviewed and official tokens are title-cased,
model output is not, so `puthimari ombikagiri High School (Baon Ansh)` shows at a glance
which half someone looked at.

Review works at the **token** level, which is why it goes this fast. The 46,644 values
decompose into 20,600 tokens, and they repeat hard: `অংশ` appears in 362 values, `স্কুল` in
19,600 rows. Correcting `ৰাজহ` once fixes every revenue circle in the corpus.

The leverage is steep and measurable. Four review passes over
[`romanize/tokens.py`](../romanize/tokens.py) — 750 tokens in total — each took the
tokens ranked by how many rows they touch rather than by where they happened to appear:

| pass | tokens | tier 1 row-weight checked |
|---|--:|--:|
| 1 | ~120 | 42.5% |
| 2 | +232 | 71.8% |
| 3 | +101 | 78.0% |
| 4 | +286 | 78.5% (and tier 2 from nothing to 70.9% of tokens) |

Pass three bought 6.2 points for 101 tokens and pass four bought 0.5 on tier 1 — the tier-1
curve is flat and what remains there is a genuine long tail. Tier 2 is earlier on its own
curve.

## Tier 2 was cheaper than its size suggested

`ps_name` and `main_town_village` are 43,827 distinct values against tier 1's 2,817 — 15×
the volume. They cost roughly one more review pass, because a station name is a **place name
plus a school**, and the school half is a closed vocabulary:

| token | rows |
|---|--:|
| `স্কুল` (school) | 19,600 |
| `এল` / `পি` (the L and P of "L.P. School") | 11,928 / 790 |
| `নং` (no.) | 9,931 |
| `বিদ্যালয়` (vidyalaya) | 7,489 |
| `শাখা` (branch) | 3,201 |

200 tokens covered 47.8% of tier-2 token-occurrences before any of them were looked at. The
place-name half is the real long tail — 19,216 distinct tokens — and that is where the
unreviewed remainder sits.

Two distinctions the vocabulary forced:

**`বিদ্যালয়` is *Vidyalaya*, not "School".** The source uses both words, sometimes in the
same name, so collapsing them would destroy information the source took care to record.
Only words the source itself borrowed come back as English — `স্কুল`, `ৰুম`, `কলেজ`.

**`১০ম` is *10ma*, not "10th".** Bengali ordinal suffixes attach to Western numerals here.
IndicXlit reads them as syllables and returns `10maha`, `2yoy`, `4ortho`, `6stho`; four
lexicon entries fix every one.

## Known limits

- Transliteration inherits OCR error. Only hand-checked entries repair it, and only where
  someone looked.
- Conventional spellings are not stable: Karimganj is now officially Sribhumi, and both
  appear here. This column is a convenience key, never an identifier.
- **Coverage of the anchor is partial.** India Post reaches post offices, villages and
  blocks. `revenue_circle` and `police_station` have no gazetteer here at all: the Local
  Government Directory holds them, but its export sits behind a CSRF-bearing form and no
  bulk download was obtained, so those fields remain lexicon-and-model only.
- **4,100 values sit in the 0.60–0.85 review band** and were deliberately *not* applied.
  They are written to `out/anchor_review.csv` for a human, not silently accepted.
- The India Post directory carries **24 postal districts**, not Assam's 35 revenue
  districts, so it is a weak authority on district names and was used mainly as a scope.
