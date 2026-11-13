# How a constituency is checked, and what the checks cannot see

This is a record of what "verified" means for this dataset, so the limits travel with the data
instead of living in somebody's memory. Everything described here is in `electors/reconcile.py`
and runs on every constituency.

The short version: **a PASS is a statement about completeness, not accuracy.** A constituency can
pass every check on this page with every name misread. Read the last section before quoting a
number.

## Why this exists

The pipeline reported success three times while producing incomplete data.

- **AC101 shipped 113 of its 210 parts** and reconciled every one of the 113. The check that
  scored parts only scored the parts that arrived, so 97 that vanished entirely were invisible to
  it. A check that cannot see absence is not a completeness check.
- **An earlier run extracted 44.2% male against a published 50.8%.** Every field was populated,
  every count was self-consistent, and the only thing that noticed was a ratio compared against a
  number the pipeline did not produce.
- **Six constituencies came home judged `UNVERIFIABLE`** — not failing, but with no evidence to
  fail against, which is operationally the same as unchecked.

Each of those is a check that now exists. The pattern in all three: the failure was invisible to
anything computed *from* the extracted rows, and visible immediately against a number printed on
the roll.

## The three tiers, and why the distinction is the whole point

Checks are grouped by **where their answer comes from**, because that determines what they can
prove.

### Tier 1 — against the roll (these can contradict us)

Numbers the publisher printed and this pipeline did not produce. Only these decide a verdict.

| Check | Compares | Catches |
|---|---|---|
| `parts_complete` | parts extracted vs. parts on the info pages | a whole part silently missing — the AC101 failure |
| `rows_against_printed` | main-list rows per part vs. that part's printed মূল তালিকা total | boxes lost within a part |
| `measured_coverage` | how many parts carry a printed total at all | how much of the check above actually applies |
| `sex_against_published` | extracted male share vs. published male/female | systematic mis-assignment across the whole roll |

Two of these are compared as **sets of parts, not counts**. An audit reproduced rows for parts
{1, 2} scored against totals for parts {1, 3} and it passed every check: part 2 was never
measured, and phantom part 3 "matched" its zero rows. Counting the two collections agrees; asking
whether they describe the same parts does not.

`measured_coverage` is reported separately from `rows_against_printed` on purpose. "205 of 205
parts match" means something very different when 205 is every part than when it is four fifths of
them — and four constituencies are currently in exactly that second position.

### Tier 2 — between the stages (these catch us losing our own work)

`rows_match_boxes` compares boxes that stage one found against rows that reached the shard. It
cannot tell you the rows are right, only that none went missing between finding and writing. It
has never failed on its own, but run by hand it is what showed AC101 holding 113 parts of rows
for 210 parts of boxes.

### Tier 3 — self-graded (these never decide anything)

`fill_rates` and `duplicate_epics` are computed entirely from the extracted rows, so they cannot
contradict the extraction. **A fill rate of 99% is equally consistent with 99% correct and 40%
correct.** They are reported because they are diagnostic, and marked `external=False` so they
cannot vote.

`duplicate_epics` earned its place indirectly: rows sharing an EPIC turned out to be eleven times
more likely to have an unreadable name, which is what implicated the digit-position repair. That
is a lead, not a verdict.

## What a verdict requires

`Verdict.ok` is deliberately narrow (`reconcile.py`):

```python
external = [c for c in self.checks if c.external]
return bool(external) and all(c.passed for c in external)
```

The `bool(external)` is not defensive padding. An audit found `ok` returning **true for a verdict
with no checks at all** — nothing was checked, so nothing was wrong, which is this project's
entire failure mode compressed into one line. Passing requires evidence, not the absence of
complaint.

## Where verification happens, and why twice

A constituency is judged on the machine that produced it, and **judged again on the laptop after
it comes home** (`electors/deliver.py`). The second pass is not redundancy: the machine that
produced a constituency is not a witness to its own correctness, and the cloud side is meant to
be deleted.

For the local judgment to be possible without the cloud, a **verification bundle** travels beside
each shard — the printed totals and box counts per part, a few kilobytes. Re-judging from
`stage1/` would mean pulling 383 MB per constituency to check 5 MB of rows, and `stage1/` is one
of the things being deleted.

The shard's **SHA-256 is checked before it is accepted**, and a shard failing its checksum is
deleted rather than left on disk — a truncated download is otherwise a silently short
constituency, which is precisely the failure this project keeps producing.

## Release results and the error floor

Everything above compares counts. `electors floor` asks a different question — **how many rows are
provably wrong**, needing no ground truth and no page to be read:

| detector | why it is proof, not suspicion |
|---|---|
| Latin characters in a name | the roll prints names in Assamese/Bengali script |
| `name == relation_name` | one line read into two fields; nobody is their own father |
| a printed label inside a value | the label/value split failed |
| EPIC failing `^[A-Z]{3}[0-9]{7}$` | the printed format is fixed |
| **the same EPIC on two rows of one constituency** | unique by definition, so one of them is wrong |
| age outside 18–120 | an electoral roll holds no seven-year-olds |

The [extraction report](EXTRACTION_REPORT.md) is the canonical release snapshot. It records the
statewide row and part counts, reconciliation coverage, field fill by edition, and each floor
detector's measured rate. Keeping those values in one generated page prevents this design note
from preserving an intermediate fleet result after the data changes.

The floor remains independent of the completeness verdict. A row no detector condemns is
*unmeasured*, not correct. Completeness and transcription accuracy are different questions.

## What none of this can see

**Whether a name is right.** No check on this page reads a name, and the floor above only catches
names wrong in a way a rule can state. A constituency can pass every check here, and clear every
detector, with names that are plausible and wrong — which is the failure that matters most to
anyone using the data.

Of the fields, exactly one has external truth:

- **sex** — the roll publishes male/female per part. But this bounds **bias, not error**:
  symmetric misreads cancel and leave the ratio untouched, so it can pass while a tenth of rows
  are individually wrong.
- **name, relation, age, house number** — no external check exists. Fill rate is not accuracy.
- **serial number** is derived by counting rows, never read. The OCR'd serial cannot check it,
  because a main-roll box has one numbered cell and a supplement box has two.

The only measurement of name, relation and age accuracy is a **hand-read sample** —
`scripts/build_review_sheet.py` draws 150 boxes across 30 parts and `scripts/score_review.py`
scores them with Wilson intervals, because 90% of 150 is 90% ± 5 points and quoting the point
estimate alone invites reading noise as improvement.

That sample is **held out**. Its predecessor, `dataset/eval/truth.json` (36 boxes), was tuned
against until it stopped measuring anything — the "72% exact names" figure came from it and should
not be quoted.

## Running it

```bash
python -m electors fleet     # where the run is: done, in flight, queued, stalled
python -m electors pull      # bring finished constituencies home and re-judge them here
python -m electors status    # the full verdict table with every check's detail
python -m electors floor     # rows provably wrong, per edition, without reading a page
python -m electors bundle    # backfill evidence for constituencies that finished without it
python -m electors reap      # delete source archives, but only for what is home and passing
```
