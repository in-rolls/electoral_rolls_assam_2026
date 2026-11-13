# Extracting the electoral rolls

The `electors` stage turns each part PDF in `data/ac_rolls/` into one row per elector. It skips
the two info pages the `assam_rolls` stage already reads and starts at the roll proper.

```
python -m electors parse data/ac_rolls/AC1_ASM.zip      # extract one constituency
python -m electors report out/electors/AC001.parquet    # reconcile against the info pages
```

Output is one Parquet shard per constituency, zstd-compressed and gitignored; the committed
manifest carries each shard's row count and SHA-256. Rows join to the info-page tables on
elector `(ac_no, part_no)` = info-page `(ac_no_file, part_no_file)`. These keys come from the
source filenames; the OCR-read info-page `ac_no` and `part_no` are validation fields.

## Running one, and what it costs

```
python -m electors parse data/ac_rolls/AC1_ASM.zip --workers 5 --capture
```

Progress is logged with timestamps to the console and to a dated file under `out/logs/`, one
line per part as it **finishes** rather than in submission order — a pool that yields in order
lets one slow part at the front hold back everything behind it, and a run that looks silent for
twenty minutes cannot be told from a wedged one.

Each part is timed inside its own worker, because the wall-clock gap between two results is
whatever the slowest earlier part was still doing. The run's wall clock, mean and median per
part, and worker count are written into the manifest beside the row counts, so what a
constituency cost is recorded rather than remembered.

Two things the accounting deliberately refuses to do. Cached parts are kept out of the per-part
mean, since mixing them in makes a resumed run look faster than the pipeline is. And no
estimate is offered until each worker has actually read a part — a resumed run serves cached
parts in about a second each, and counting those as throughput put the first estimate for a
two-hour job at 57 seconds.

### Measured cost, and three optimizations that did not survive measurement

On a 10-core machine shared with other work (load ~25), 5 workers: **a part takes about 47
minutes** (822 electors in 46m58s; 873 in 47m34s) and a constituency projects to **16-17
hours**. One page of thirty electors costs about 53 seconds, and it divides:

| | share of a page |
|---|--:|
| text bands, 123 crops at scale 2 | 56% |
| name band, 30 crops at scale 3 | 17% |
| serial strips | 11% |
| rasterizing the page | 11% |
| EPIC strips | 2% |

Four attempts to cut it, three of which failed and are recorded because a failed
optimization tried twice is worse than one written down:

- **`OMP_THREAD_LIMIT=1`.** Tesseract multithreads, and five workers spawn far more threads
  than there are cores. Interleaved A/B: **0.97x, faster in 5 rounds of 8** — noise. An earlier
  version of the same test said 15%, but it ran the default arm first every round while the
  machine was quietening, so "second in the round" was worth the entire effect.
- **Whole text column at `psm 6` instead of four band crops at `psm 7`.** Five times faster,
  4.1s against 20.0s — and it recovered 34 of 120 fields against 104. Rejected.
- **Rasterizing in gray rather than color.** Every consumer converts to gray anyway, so the
  color channels looked like pure waste, and a first A/B said 13% faster with byte-identical
  text on 123 of 123 crops. Isolating the stages that actually change killed it: the PNG is the
  **same size either way** (2.88 MB), because the source PDFs are already gray. Render 0.99x,
  load 0.97x, 0.04s saved per page. The 13% was the machine.
- **The serial zone.** This one worked: see above.

#### Establish the resolution before believing any of it

Three of the four failed and all three looked like wins first, which should have prompted the
obvious question sooner: **what does a null result look like here?** Running the same
configuration in both arms answers it.

    arm A median 23.3s   arm B median 21.0s   (identical configuration)
    apparent effect 0.90x, arm B "faster" in 5 of 8 rounds
    single measurements span 1.87x with nothing changed
    median-of-8 resolves to about 10%

An A/A pair produced a 10% "improvement" and won five rounds of eight -- the same score
`OMP_THREAD_LIMIT=1` got. That result was not evidence of anything, and neither was the
grayscale 13%.

The rule is not that large effects are unreachable; the serial zone's 2.1x sat far outside this
band and was readable directly. It is that **the resolution has to be measured rather than
assumed**, because it tracks whatever else the machine is doing -- load swung between 6 and 62
in one evening here. Below the resolution, more rounds help slowly and isolating the stage that
differs helps immediately: grayscale was settled in one step by noticing the output file was
the same size either way.

*Applied honestly to the change that shipped:* the serial zone's timing gain (0.78x on the
second part) is near this noise floor and should not be leaned on. Its justification is the
read rate -- 1.15-1.34x more serials recovered on both parts -- which is a count and not
subject to timing noise at all.

**Cost is dominated by OCR and by whatever else the machine is doing.** A part is roughly 30
pages, and each page costs about four tesseract invocations, so a constituency is a multi-hour
job. Any per-constituency figure quoted from a loaded machine describes the load. The manifest
records the worker count alongside the timing for that reason.

## What a page looks like

An elector page is a ruled grid, three columns by ten boxes. Each box holds six fields in two
places: a **header strip** across the full width with the serial number at the left and the
EPIC at the right, and a **text column** below it with four labeled lines.

```
নাম : খাদৰাম ৰাভা              name
পিতাৰ নাম : গংগাৰাম ৰাভা       relation, and its type from the label
ঘৰ নং : 21                     house number
বয়স : 46 লিঙ্গ : পুৰুষ          age and sex
```

Four measured facts shape how it is read.

**The EPIC needs the English model.** `asm` renders `HHK0001471` as `1414140001471` — it is
trained on a script with no Latin letters and maps them onto lookalike digits. Read with `eng`
the same crop comes back exactly right. The EPIC is the only globally unique field here, so
this one substitution is worth more than any other tuning.

**Scale, not page-segmentation mode, decides a marginal read.** Native resolution was tried:
counting how many crops still contained the *label* said scale 1 was as good as scale 2 for
half the cost. That measured the wrong thing. Extracting the *values* cost age 77% → 63% and
sex 83% → 73% — the printed label survives a coarse read, the digits beside it do not.

**Adjacent boxes share their borders.** Three columns produce seven rule clusters, not nine, so
column geometry is derived from cluster positions rather than assumed. Partial last pages draw
no photo dividers and produce four; both forms are handled, and the divider is derived from a
measured fraction of box width when it was never printed.

**Row height comes from page area explained, not from frequency.** Taking the modal gap between
horizontal rules returned six rows for a thirty-elector page, because internal rules outnumber
real ones.

## Ground truth, and what stands in for it

The roll prints its own totals on a closing page — `মূল তালিকা ... 453 420 873` — and they are
self-validating, since `male + female + third == total`. That page is the only real ground
truth in the source, and finding it mattered more than any single fix: the pipeline had been
reconciling against the info page's *net* figure instead, and residuals collapsed from ±27 per
part to ±1 the moment the right pair was compared.

Everything else is bracketed rather than known.

**Floor — provably wrong.** Values that cannot be right whatever the source says: Latin letters
or digits in an Assamese name, a name identical to the relation, a field's own label leaking
into its value, a malformed EPIC, an age outside 18–120, and a supposedly unique EPIC repeating.

**Ceiling — disagreement.** The name band is read at two scales and compared. Where they agree
the value is very likely right; where they differ by more than scanner noise the row is
flagged. The finer scale is not assumed better, so the primary reading is kept and the
disagreement is recorded rather than resolved silently.

**A fill rate is never reported as accuracy.** `epic = 83%` means 83% non-empty, which is
equally consistent with 83% correct and 40% correct.

## The improvement loop

Fixes here were once justified by the page that revealed them, which is how the sex bias was
declared fixed twice before its real cause was found. Four modules now carry the loop.

### `capture` / `replay` — separate the expensive half from the cheap half

Most fixes change how a line is *interpreted*, not how it is read. Scoring one used to mean
re-rasterising the PDF and re-running tesseract over every crop — half an hour per variant, for
a change that cannot alter a single character tesseract returns.

```
python -m electors capture data/ac_rolls/AC1_ASM.zip --parts 12 13 14
python -m electors replay --diagnose
```

`capture` writes down every line of text with the geometry that produced it. `replay` re-parses
it with whatever the code says today. Two variants then score against **identical** input,
which also removes a confound: where OCR is re-run per variant, some of the difference is just
tesseract. Replay calls the pipeline's own `fields.assemble`, so it cannot drift from what
production does, and a test asserts the two agree.

It is exact for anything downstream of the text and **blind to crop geometry, scale and
engine**, which change the text itself. Captures are stamped, and a stale one is refused rather
than replayed — replaying text the pipeline would not produce answers the wrong question.

### `diagnose` — compute the co-occurrence, do not eyeball it

Every root cause found here started as a co-occurrence noticed by eye. Noticing by eye does not
scale and does not repeat, so this computes the failure rate of each class within each value of
each feature and reports the slices well above base rate.

Both missing *and* wrong values are failure classes. For a while only the first kind was, and
the largest error class in the corpus was invisible to the scan while the detectors that find
it sat in another module.

The feature set includes derived positional features, because **"diffuse" is a claim about the
feature set, not about the data** — and it is the claim that routes a class to the expensive
second pass. Adding page position moved `no_house` (3,550 rows) from diffuse to concentrated
and surfaced that names fail at 24.0% on the first elector page of a part against an 11.7%
base. `no_age` stayed diffuse under the richer features, which makes escalating it a measured
verdict rather than a gap in what was measured.

### `bench` — four gates, all required

| gate | means |
|---|---|
| in-sample | the target metric improves on the parts it was diagnosed against |
| out-of-sample | it improves by a comparable margin on parts never inspected |
| no degradation | every other field, and both ground-truth measures, stay within tolerance |
| cost | seconds per part does not rise materially |

Splits are written down rather than drawn fresh, because an out-of-sample set that gets looked
at stops being one. DIAGNOSE (parts 12–16) may be opened and stared at; VALIDATE is seeded and
used only to score; REGRESSION (parts 1–11) has measured roll totals.

Three details that took a mistake each to get right:

- **Guarded on soundness, not fill.** A fill rate *falls* when a provably wrong value is
  correctly cleared, so guarding it rejects the one move that unambiguously improves the data.
  `*_sound` counts a field only when present and not provably wrong.
- **A metric absent from either side is a failure, not a skip.** Otherwise the two checks with
  real ground truth pass by not being measured.
- **Targets have a direction.** An error rate improves by falling. Without saying so, the only
  way to gate a fix aimed at reducing wrong values is to gate it on some other metric that
  happens to rise — judging a change on something it never set out to do.

### What the loop produced

The band-assignment work is the first change taken through all four gates end to end, scored by
replaying identical cached OCR text through the old code and the new. 5,664 rows, eight parts,
splits disjoint.

| | in-sample (3,567) | out-of-sample (2,097) |
|---|--:|--:|
| name present and not provably wrong | 72.0% → **86.9%** | 73.4% → **89.2%** |
| relation present and not provably wrong | 77.5% → **93.2%** | 76.1% → **94.3%** |
| provably wrong | 19.2% → **6.3%** | 19.2% → **4.0%** |
| name identical to relation | 486 → 2 | 325 → 3 |

Out-of-sample matched and slightly exceeded in-sample, which is what says the diagnosis set was
not fitted. Nothing degraded: age, EPIC and sex were flat, completeness stayed at every
measured part exact, and the sex ratio did not move. Parsing cost rose 1.05× — under 0.1% of
pipeline time, which is OCR.

### Which engine, measured against the page

Four engines on 16 boxes sampled at random and read by eye -- the only real ground truth here:

| field | tesseract | savitr terse | surya-full | gemini flash-lite |
|---|--:|--:|--:|--:|
| name exactly right | 0% | 19% | **62%** | 38% |
| name nearly right | 38% | 38% | 81% | **94%** |
| age right | 50% | 44% | 94% | **100%** |
| house no right | 0% | 44% | 75% | **88%** |
| sex right | 94% | 44% | 94% | **100%** |

Tesseract reads no name exactly right in sixteen. Its Assamese is not broken -- it returns
recognizable text with wrong details -- and it is the engine rather than the cropping: the
pipeline's band finder, a re-derived band and a fixed slice of the box all give the same 6/16.

**Bengali traineddata was tried and is worse.** Assamese and Bengali share one script and
Bengali has far more data behind it, so `ben` should plausibly beat `asm`. Measured: names 12%
near-right against 38%, sex 69% against 94%, age 56% against 50%. Shared script, more data,
worse reading.

**Send a column, not a page.** Gemini's image tokens are relative -- a single box and a whole
page both cost about 324 input tokens -- so a page is thirty times the area at the same budget.
Names are 80% near-right per box and per column, 20% per page; house numbers 100%, 80%, 0%. At
page scale it still returns all thirty electors with correct serials, ages and sexes, and gives
every one of them the same invented surname. Confidently wrong at the granularity that looks
cheapest.

**Cost, measured.** 324 input and ~2,100 output tokens per page of thirty electors. One
constituency at column granularity is about **$2** on Flash-Lite through the Batch API, $11 on
2.5 Flash; all 126 about $230. Against roughly 150 hours of local GPU time for one.

#### Two scoring bugs, both understating the models

- **Bengali RA (U+09B0) and Assamese RA (U+09F0) are the same letter.** The models emit the
  Bengali codepoint; scoring them apart marked correct readings wrong and cost surya-full four
  exact names. Only RA is folded -- `ব` and `ৱ` look like a pair but are distinct letters, and
  folding them changed no score.
- **House numbers came back in Bengali digits**, because the prompt asks for the page exactly as
  printed. Scored against Latin digits, that marked Gemini wrong on 14 of 16 it had read right.

And one remaining miss was the ground truth itself: `ঘৰ নং : 20 ক` was recorded as `20`.
Eyeball truth has its own error rate, and 16 boxes is enough to separate 0% from 90%, not to
rank two engines ten points apart.

### Choosing an engine for 25 million electors

31,486 parts, **24,958,139 electors**, about 921,000 pages. At that scale the accuracy winner
and the usable engine are not the same thing.

Scored on the 25 boxes every engine answered -- the only basis on which they can be ranked:

| field | tesseract | savitr | surya-full | gemini | dots.ocr | cloud-vision |
|---|--:|--:|--:|--:|--:|--:|
| name exactly right | 0% | **76%** | 60% | 28% | 72% | 72% |
| name nearly right | 36% | 92% | 76% | 88% | **100%** | **100%** |
| age right | 60% | 92% | 92% | **100%** | **100%** | **100%** |
| house no right | 0% | 80% | 68% | 60% | **88%** | **88%** |
| sex right | 96% | 92% | 92% | **100%** | **100%** | **100%** |

**dots.ocr and Cloud Vision are tied**, field for field, and the tie is what decides the state:

| | whole state | wall clock |
|---|--:|---|
| Cloud Vision, 400 dpi, repacked | **$105** | hours |
| Gemini Flash-Lite, batch | ~$1,500 | a day per batch |
| dots.ocr, per box | **$1,860** measured, ~$370 if vLLM gives 5x | 16,909 T4-hours at $0.11/hr |

Cloud Vision reads as well, costs $105, needs no hardware, and is the only engine here whose
throughput at scale is a measured quantity rather than a guess spanning thirty-fold. **It runs
the state.**

#### Sending only the text: 3.3x cheaper, and it reads the same

Vision bills per **image submitted**, and a rendered page is 15.5 MP of which **30% is text**.
The rest is photo placeholders, ruled borders, page margins and the white space under the last
line of every box. So the page is repacked before submission: each box's four labeled lines are
cut to their band extent and tiled three to a row, and words are assigned back to a box by their
rectangle on both axes.

Measured on parts 1-3, 2,091 boxes: **23 images as rendered against 7 repacked**, 3.3x fewer.
Across the state that is **$105 against $345**.

It costs nothing. Scored against the 34 hand-read boxes both passes answered:

| field | as rendered | repacked |
|---|--:|--:|
| name exactly right | 68% | **68%** |
| name nearly right | 97% | 94% |
| first name right | 91% | **91%** |
| age right | 97% | **97%** |
| house no right | 88% | **88%** |
| sex right | 97% | **97%** |

Identical on five of six, and the sixth is one box.

**The two passes disagree more than the scores suggest, and it does not matter.** Over 2,091
boxes they agree on 85% of names, 96% of ages, 97% of house numbers and 99% of sexes -- but the
name disagreements are single matras of the same person (`বসুমাতাৰী` against `বসুমতাৰী`), and
against truth neither pass is better. It is recognition jitter, not misfiled rows: a mapping
error would have collapsed all four fields together, and the other three sit at 96-99%.

Three things stay on the CPU because none is worth a paid pixel -- the **EPIC**, which sits
outside the text column and would cost most of the saving to include; the **section header**,
which distinguishes the main roll from supplements; and the **closing summary**, which
completeness is measured against. Tesseract reads all three, and its English model gets the EPIC
at 96.6%.

#### What resolution to feed it

The part PDFs carry **no fonts at all**: 34 images, every page a single 1187x1679 raster. They
are 144 dpi scans, so rendering at 400 dpi is a 2.78x upsample of pixels that already exist.
Since Vision bills per *image submitted* and how many pages fit in one is set by their size,
resolution is the whole cost lever -- and it had never been measured.

Three arms, same parts, same parser, scored on the 25 hand-read boxes all of them returned:

| field | 400 dpi | 300 dpi | native 1187x1679 |
|---|--:|--:|--:|
| name exactly right | **72%** | 68% | 56% |
| name nearly right | **100%** | 96% | 88% |
| age right | **100%** | 96% | 84% |
| house no right | **88%** | **88%** | 52% |
| sex right | **100%** | **100%** | **100%** |
| pages per image | 4.2 | 6.4 | 31.0 |
| **whole state** | $368 | $217 | $45 |

**Native is worse, and not by noise.** House numbers halve and names fall 16 points -- four of
five fields degrade together, which is what separates a real effect from sampling error. A box
is ~1000px wide at 400 dpi and ~395px at native, and the note in `extract.py` about conjuncts
ceasing to be legible turns out not to have been only about tesseract. The $45 is real and so
is what it costs.

**300 dpi looked free on this table and is not.** Twenty-five boxes cannot separate 68% from
72%, and reading that as "within noise" was the wrong conclusion -- it was an absence of
evidence. Two measurements over 7,431 boxes, neither needing ground truth, found the difference:

| | 400 dpi | 300 dpi |
|---|--:|--:|
| names the two arms agree on | \-- | 76.6% |
| **name left empty** | **0.44%** | **5.77%** |
| age left empty | 2.69% | 2.22% |
| house left empty | 2.17% | 3.15% |
| sex left empty | 0.73% | 1.00% |

The arms differ on **23% of names**, so they are not interchangeable, and they differ
*asymmetrically*: 300 dpi returns no name at all thirteen times as often. Of the 1,742 names
that differ, 300 dpi is empty on 420 and 400 dpi on 24.

A missing value is a failure whichever spelling would have been right, which is why this ranks
the arms without any hand-reading. Over 25 million electors it is **1.3 million missing names
against 110,000**, and the difference costs $151. **The pipeline stays at 400 dpi.**

Set against that, `render.py` records that interpolating these scans invents ink badly enough to
turn a "1" into a "4" for tesseract. Vision evidently does not mind: the upsampled arms win.
Recorded because the prediction was the other way round.

**Four API constraints, all found by running rather than by reading:**

- **The 40 MB limit is on the request, not the image.** 20 MB per image was already checked;
  sixteen 6 MB images are legal individually and 96 MB together. Batching by count rejected
  every 400 dpi and 300 dpi part while letting native through, which would have become the
  experiment's finding.
- **`PAGES_PER_IMAGE = 8` is no limit of the API.** At native, 33 pages fit under the pixel
  ceiling, and the constant -- not the ceiling -- was capping the cheapest arm.
- **The stacking factor cannot be a constant.** Eight pages is right at 300 dpi and 124 MP at
  400, against a 75 MP ceiling; it is measured from the pages now.
- **Requests fail transiently.** DNS failures and broken pipes cost two arms of one run, so
  `annotate` retries 5xx and network errors and never retries a 4xx.

#### The fourth measurement artifact, and the one that decided this

Vision scored **44% on exact names until the parser was fixed, and 72% after** -- and the
recognition never changed. Vision reads the printed `নাম` label as `নামু`, a matra that is not
on the page, on 12 of 34 boxes. `NAME_RE` required the correct spelling, so the label did not
match, the line fell through to the unlabelled-name branch, and the elector was recorded as
`নামু : অঙেলা মুছাহাৰী`. Ten names Vision had read perfectly were scored wrong.

Every other engine's parse is byte-identical across the fix -- `নামু` appears only in Vision's
output -- so this was one engine's column being wrong, in the direction of looking worse.

That is four for four. **Every measurement artifact found in this stage understated an engine**:
unequal token budgets, a truth set attached to the wrong boxes, a loop guard that returned the
padding instead of the answer, and now a label the parser could not see. Three of the four were
found by refusing to accept a number that looked too bad, which is the only method that has
worked here.

#### What dots.ocr actually costs, measured

Every earlier figure here came from single-stream MLX on a Mac, which describes the laptop. On a
Kaggle Tesla T4 -- transformers 4.56.1, float16, sdpa -- the batched rate is:

| batch | boxes/sec |
|--:|--:|
| 1 | 0.19 |
| 2 | 0.32 |
| **4** | **0.41** |
| 8 | out of memory |

24,958,139 electors at 0.41 boxes/sec is **16,909 GPU-hours**. At a spot T4 price of $0.11/hr
that is **$1,860**, against Cloud Vision's measured $368.

**But this is a floor, not a verdict.** The sweep was stopped by *memory*, not compute -- batch 8
would not fit in 16 GB. vLLM's paged attention would hold a far larger batch on the same card,
and 25 million short uniform prompts is precisely what continuous batching is for. Break-even
against Vision is **2.07 boxes/sec, 5x the measured rate**, which is inside the range vLLM
plausibly delivers:

| | boxes/sec | GPU-hours | at $0.11/hr | vs Vision |
|---|--:|--:|--:|--:|
| transformers, measured | 0.41 | 16,909 | $1,860 | 5.1x |
| vLLM, 3x | 1.23 | 5,636 | $620 | 1.7x |
| vLLM, 5x | 2.05 | 3,382 | $372 | 1.0x |
| vLLM, 10x | 4.10 | 1,691 | $186 | 0.5x |

So **whether dots.ocr can run the state is open**, and turns on a vLLM measurement that has not
been made. Two things price does not settle either way: 25 million inferences need checkpointing,
retries and preemption handling, where Vision is about 36 calls per constituency; and on the
240-box sample dots.ocr left **no field empty at all** where Vision left 3.33% of ages blank --
an edge, unless it is guessing rather than admitting ignorance, which a tie on exact names does
not rule out.

#### Cropping to one line is 6.2x cheaper and does not work

Prefill is where dots.ocr's cost is, and the vision tower is where prefill is: 42 layers and
0.94B parameters against a 1.27B language model, so **the ViT is 70% of the work**.

| | box crop | name band |
|---|--:|--:|
| vision tokens | 378 | 54 |
| ViT forward | 2.85 TFLOPs (70%) | 0.41 (62%) |
| LLM prefill | 1.01 (25%) | 0.19 (29%) |
| LLM decode | 0.23 (6%) | 0.06 (10%) |
| **total** | **4.08** | **0.66** |

So showing the model one 776x70 line instead of the whole 776x415 box is **6.2x less work**, and
`grid.text_bands` already finds the lines on the CPU for nothing. Measured on the same 240 boxes:

| | Bengali script | Devanagari | runaway | name agreement with Vision |
|---|--:|--:|--:|--:|
| box | 100% | 0% | 0.8% | **52.1%** |
| band | 83% | **32%** | **76%** | **26.7%** |

**It comes apart two ways at once.** On 76% of band crops the model repeats the line to the token
cap, and on a third it switches script entirely -- `নাম : মহিমা গয়াবী` returned as
`नाम : रमिमा गयाबी`, Assamese transliterated into Devanagari. 17% yield no parseable name.

The likely cause is the aspect ratio: 11:1 is not a shape a document model sees in training, and
one line with no page around it gives nothing to anchor the script on and nothing to signal an
end. The saving is real and unusable.

Two things this cost nothing to learn, because the crops were checked before the speed was:

- The band as `grid.text_bands` returns it **clips glyphs**. Its 5px pad suits tesseract; at 400
  dpi it takes the head off a tall matra and the tail off a descender, so `বিনয়` loses its `ি`
  and `অৰ্জুন` its hanger -- read as different names with nothing to show for it. Padding by 0.45
  of band height fixes that and then bleeds the *relation* line in underneath, so each side is
  clamped to the midpoint of the gap.
- Quality was measured first, not last. The prefill arithmetic was right and the conclusion drawn
  from it was wrong, and only the order of the measurements caught that.

#### dots.ocr agreed with Vision on 240 boxes

| field | agreement | Vision left empty | dots.ocr left empty |
|---|--:|--:|--:|
| name | 52.1% | 0.42% | **0.00%** |
| age | 95.4% | 3.33% | **0.00%** |
| house | 92.5% | 1.67% | 0.42% |
| sex | 99.6% | 0.42% | **0.00%** |

Agreement, never accuracy: both can be wrong together. The name disagreements are single matras
of the same name -- `অৰ্জন`/`অৰ্জুন`, `সীতাশ্বৰী`/`সীতাশুৰী` -- and one stray hyphen Vision
appended, not different people. Where the two agree the value is very likely right, and the rows
where they differ are the ones worth review. That is the error signal there would otherwise be
none of.

**dots.ocr still cannot read a page**, which is why it was never the cheap option:

- Asked to extract the text of a page, it returns **13 of 30 electors** and then repeats one
  EPIC until the token cap.
- Asked with its own layout prompt, it classifies **the entire elector grid as a single
  `"Picture"`** and extracts nothing from it -- only the page header and footer come back. A
  ruled grid of boxes is not text to it.

That is the same under-generation every vision-language model here shows at page scale. Gemini
fell from 94% to 20% on names for the related reason that its image tokens are relative to the
image, so a page is thirty times the area at one budget. Cloud Vision is the only engine that
reads a page as well as it reads a box, which is what makes eight-pages-per-image viable.

**dots.ocr's remaining job is to check Vision.** Once Vision has run the state nothing else
verifies it, and with no labels the only automatic error signal is two independent engines
disagreeing. dots.ocr is the strongest available second reader and it fails differently, so
running it over a sample and reporting per-field agreement is worth the local time -- as
agreement, never as accuracy, because two engines can be wrong together.

### Combining engines: three or one, nothing in between

The engines fail differently -- Gemini regularizes spellings toward commoner forms because it
is a language model, while dots.ocr, Surya and Vision are pure OCR with no such prior -- which
is the condition under which a vote beats a single reader. Majority vote, ties broken by the
preferred engine, on the 33 boxes all three answered:

| engines | name exact | name near | age | house | sex |
|---|--:|--:|--:|--:|--:|
| dots.ocr | 70% | 100% | 100% | 85% | 100% |
| gemini | 30% | 82% | 100% | 61% | 100% |
| vision | 67% | 97% | 97% | 88% | 97% |
| dots + gemini | 70% | 100% | 100% | 85% | 100% |
| dots + vision | 70% | 100% | 100% | 85% | 100% |
| gemini + vision | 30% | 82% | 100% | 61% | 100% |
| **all three** | **73%** | 100% | 100% | 88% | 100% |

**Every pair scores exactly like its stronger member.** With two readers a majority cannot
exist, so a disagreement falls back to the preferred engine and the second one changes nothing.
Voting needs three, which makes this all-or-nothing rather than a gradient.

The three-way gain over dots.ocr alone is +3 points on exact names -- **one box in thirty-three**
-- and every other field is already at ceiling or within a box. Three engines for one box is not
a result; it is a sample too small to see one. Nothing here justifies 3x the compute.

### The second pass, and what it revealed about the first

Tesseract cannot read the house number on a third of boxes or the age on a fifth, and the line
cache showed the crop yields no digits at all -- so no parser change recovers them. savitr's
distilled Surya reads the same crops. Over a whole part (636 electors, 334 boxes re-read):

| | before | after |
|---|--:|--:|
| house no | 59.6% | **82.1%** |
| age | 80.3% | **92.3%** |
| sex | 95.8% | **98.6%** |

**And it exposed an error the floor could not see.** The engines both produced an age on 20
boxes of one page and disagreed on 7. The crops were read by eye: the second pass was right on
every one of the six checked -- 24 against 25, 59 against 92, 52 against 92, 25 against 85, 54
against 25, 45 against 55.

That means roughly **a third of the ages the first pass reports are wrong**, against an error
floor of 0.5% which never saw them, because an age of 92 is perfectly plausible on its own. It
is also the mechanism behind an anomaly detected independently weeks of measurement earlier:
the excess of nonagenarians in the age distribution. Age is now taken from the second pass
where the two disagree, and the first reading is kept beside it.

**The cost is the constraint, and batching does not relieve it.** One call per box is not an
oversight -- stacking crops makes the model ignore all but one box, and layout-preserving crops
(a grid column, a page) run away until the token cap stops them. At 2-6 seconds a box a full
second pass is tens of hours per constituency, so which rows to send is the only lever.

### `escalate` — know which rows are wrong rather than perfecting the pass

A cheap pass will always leave errors; what makes that acceptable is knowing which rows they
are, so an expensive engine runs on those and only those. Four families feed the router:
provably wrong, two-scale disagreement, a relation whose label was never recognized, and two or
more missing core fields. It flags **28.3%** of rows.

The disagreement threshold is measured rather than chosen. Over 1,733 boxes where both scales
produced a name they differ on 53%, but 83% of those differences are a character or two — a
stray matra, a speck of punctuation. Flagging all of them took the router to 56%, which is a
re-run rather than a triage. Below 0.90 similarity the readings differ by more than noise.

Its precision is deliberately not reported. The floor detectors are half the router, so scoring
it against them is a check that cannot fail. Volume and per-family contribution are reported
instead, and precision becomes measurable once a second engine has read the flagged rows.

**Cost is counted in bands, not rows.** A re-read settles one field, not a whole box, so
flagging 39.8% of rows costs 2,718 band re-reads against 22,656 for a full second pass — 12%
of one. Counting rows alone made a precise detector look like it had pushed the router past
being a triage.

**A detector can be falsified even where precision cannot be measured.** If the rows it picks
are no likelier to be wrong than the rows it passes over, it is flagging at random however
sensible its reason sounds. `separation` scores that against a *distributional* property, which
is the only handle on a value that is individually plausible but collectively impossible.

The age detector is the case in point. `age_ambiguous` fires where the age zone holds more
digits than an age can account for — `বয়স ' 9526`. Flagged rows put 15.2% of ages in the
nineties; unflagged rows put 2.2% there, and fall away from 28.6% in the twenties exactly as a
roll should. That is a **6.8× concentration** of demographically impossible ages, established
without a single labeled row.

Which end of a long run is the age was left undecided on purpose. Read from the left it puts
15.3% of those rows in the nineties, from the right 13.6%, against the 2.3% the unambiguous
two-digit runs show — so neither end is the age, and picking the marginally better rule would
be tuning on a signal that does not support it. An age of 95 that is really 26 sits inside
every range check, so no floor detector sees it and the soundness metric counts it as good.
The honest response to a value the parser cannot recover is to route it, not to invent a rule.

Measuring that composition immediately found a dead signal: `name_disagreement` had been raised
**zero times in 10,245 rows**, because the one-line second-scale read was passed through the
band assigner, where a single line lands in `house`. A scale-3 pass over every name crop was
being computed and discarded.

savitr's distilled model was checked as the expensive tier and rejected on evidence: it emits
this schema at 17.5 s/page with 96–98% reported fidelity, but 400 of 400 sampled training
records are Latin-script English rolls. It says nothing about Bengali-Assamese. General Surya
remains the candidate.

## What is not measured

None of this measures name accuracy against truth. The bracket — provably wrong at the floor,
two-scale disagreement at the ceiling — is what exists without labeled data, and cross-engine
agreement will narrow it without closing it. Agreement is reported as agreement, never as
accuracy: two engines can be wrong together.

The full 154-part run for AC1 has not completed, so no constituency-wide figure is published
here. The numbers above are from sampled parts and say so.
