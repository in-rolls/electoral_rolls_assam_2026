"""Re-reading the boxes the cheap pass could not, with a model that can.

Tesseract loses the house number on a third of boxes and the age on a fifth, and the line cache
showed why: the crop yields no digits at all, so no parser change recovers them. savitr's
distilled Surya reads the same crops -- on one page it took the house number from 14 of 30 to
29 and the age from 21 to 29.

**It is also right where the two disagree.** On the seven ages both engines read differently,
tesseract gave 92, 92, 85 and 96 while Surya gave 59, 52, 25 and 61. That is the same excess of
nonagenarians the age distribution had already flagged as an artifact -- 220 rows in the
nineties against 126 in the seventies -- so the disagreement is not a coin toss between two
opinions, it is one engine reproducing a known defect and the other not.

It costs 2-6 seconds a box against tesseract's 1.7, depending on what else the machine is
doing, which is why this runs on flagged rows rather than on everything.

**One call per box is not an oversight; it is the only thing that works.** The obvious saving
is to answer for several electors per call, and it was measured two ways and failed both:

- *Stacking crops into one tile.* The model returns exactly one record however many boxes are
  in the image -- 6 ages from 12 boxes at two per call, 3 from 12 at four per call, 2 from 12
  at six. The extra boxes are simply ignored, so the saving is imaginary.
- *Keeping the roll's own layout* -- a whole grid column, or three columns of four rows. These
  run away instead: 13 ages from 10 boxes, then 0, then 12, then 0, every one of them stopped
  by the token cap. A whole page behaves the same way, producing one voter in 106 seconds.

So the cost per elector is fixed at one model call, and the only lever left is sending fewer
boxes.

Two things the model does that have to be handled rather than hoped away:

*It runs away.* Eleven generations of thirty repeated themselves until the token cap stopped
them. The loop repeats a *complete* record, so the first cycle is kept and the rest discarded;
a response that hit the cap without ever repeating stopped mid-record and is refused outright.

*Its format varies.* The same prompt returns bare pipe-separated fields, the same wrapped in
``<p>``, and table rows padded with ``||``. The fields are found by label rather than by
position, because position is the thing that moves.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from assam_rolls import schema

#: Fields worth taking from the second pass. Name and relation are deliberately absent: the
#: cheap pass already returns them for 92% and 97% of rows and this model rewrites spellings
#: (নার্জাৰি for নাজ্জাৰা), so accepting them would churn values that were not in question.
RECOVERABLE = ("age", "house_no", "sex")

#: Fields where the second pass **overrules** a value the first pass already produced.
#:
#: Only ``age``, and only on evidence. The two engines disagreed on 7 of the 20 ages both read
#: on one page; the age lines were cropped and read by eye, and the second pass was right on
#: every one of the six checked -- 24 against 25, 59 against 92, 52 against 92, 25 against 85,
#: 54 against 25, 45 against 55. Tesseract's errors are also the *shape* already flagged from
#: the distribution: implausibly old readings, which is where the excess of nonagenarians came
#: from.
#:
#: That measurement is small (six crops) but one-sided, and it is corroborated by a corpus-wide
#: anomaly it explains. House and sex are not on this list because the second pass has only
#: ever filled gaps in them, never contradicted a value.
OVERRULED = ("age",)


#: Label-anchored, because the layout of the response is what varies. Bengali and Latin digits
#: both appear, sometimes in the same line.
def _either_ra(pattern: str) -> str:
    """Accept the Bengali and Assamese forms of RA anywhere in a label.

    They are one letter written two ways, and the engines pick either: dots.ocr writes ``ঘর নং``
    where the page prints ``ঘৰ নং``, and a pattern that knew only one of them found no house
    number at all on those rows.
    """
    # One pass. Replacing ৰ first and র second mangles the class it just inserted, because
    # "[ৰর]" contains র -- it becomes "[ৰ[ৰর]]" and matches nothing.
    return re.sub("[ৰর]", "[ৰর]", pattern)


#: A combining vowel sign, optionally hung off the last letter of a printed label.
#:
#: Cloud Vision reads the printed ``নাম`` as ``নামু`` on 12 of 34 boxes -- a matra that is not on
#: the page, attached to the ``ম``. A pattern that knew only the correct spelling did not match
#: the label at all, so the whole line fell through to the unlabelled-name branch and the elector
#: was recorded as "নামু : অঙেলা মুছাহাৰী". Ten names Vision had read perfectly were scored wrong,
#: which is the difference between Vision reading 44% of names exactly and 76%.
#:
#: Only on labels, never on values: a label is a known word and a stray matra on it is provably
#: noise, whereas a matra in a name is the name.
MATRA = "[া-ৌৗ]?"

#: Up to six non-digit characters between the label and its number, rather than a colon and
#: whitespace. The scan litters the gap: ``বয়স ' : 30``, ``বয়স ? 42 : ১``, ``ঘৰ বয়স : : 78`` all
#: carry a perfectly readable age that the stricter pattern walked straight past.
#:
#: Six, not unlimited. ``বয়স ঘৰ নং : ৪৪ 58`` is a merged age and house line where the first number
#: is the house number and the second the age; nine characters separate the label from ৪৪, so a
#: wider window would confidently record the house number as the age. Refusing that line loses one
#: age and invents none.
#: Assam prints one constituency in English -- AC113 -- and its labels are Latin words. They are
#: added to these patterns rather than given a parallel reader: the two scripts cannot collide, so
#: an alternation costs nothing and a second reader would be a second thing to keep correct. The
#: house-number bug is the argument: two label sets existed, one was dead code for the path that
#: ran, and it stayed wrong through 2.2 million rows.
AGE_RE = re.compile(r"(?:বয়স[^0-9০-৯]{0,6}|AGE[^0-9]{0,6})([0-9০-৯]{1,3})", re.IGNORECASE)
#: The word for "house", in both editions the state publishes. Assamese prints ``ঘৰ নং``; Bengali
#: prints ``বাড়ী নং``, and knowing only the Assamese form cost the house number on **every row of
#: all ten Bengali constituencies** -- about 2.2 million of them. The label is on the page and
#: Vision read it: 282 of 316 boxes on one AC114 page carry ``বাড়ী নং : 01`` or similar. Nothing
#: matched it, so ``fields()`` never recorded a value and the column shipped empty.
#:
#: ``fields.HOUSE_LABELS`` also lists these forms, and that is not where the bug was: the whole
#: pipeline reads through :func:`vision.elector_from`, which routes to this module deliberately
#: (labels rather than positions) and never consults that tuple. Two label sets, one of which was
#: dead code for this path.
HOUSE_WORD = r"(?:ঘৰ|বাড়ী|বাড়ি)"

#: ``House No. 1`` on the English roll. Vision often separates punctuation into its own token,
#: yielding ``House No .: 1``; whitespace before the period is formatting, not a missing label.
HOUSE_EN = r"HOUSE\s*N[O0]\s*\.?"

#: ``[ \t]*`` rather than ``\s*`` before the suffix: the terse answers are line-delimited, and a
#: pattern that crossed the newline captured the ``ব`` of the ``বয়স`` on the next line, turning
#: house 13 into "13\nব".
HOUSE_RE = re.compile(
    _either_ra(
        rf"(?:{HOUSE_WORD}\s*নং|{HOUSE_EN})\s*[:：.]?\s*" rf"([0-9০-৯]{{1,6}}[ \t]*[ক-হA-Z]?)"
    ),
    re.IGNORECASE,
)
SEX_RE = re.compile(r"(?:লিঙ্গ|GENDER)\s*[:：.]?\s*(\S+)", re.IGNORECASE)

RELATION_WORDS = _either_ra(r"পিতাৰ|স্বামীৰ|মাতাৰ|মাতৃৰ|FATHER'?S|MOTHER'?S|HUSBAND'?S")
RELATION_RE = re.compile(
    rf"(?P<kind>{RELATION_WORDS})\s*(?:নাম{MATRA}|NAME)\s*[:：.]?\s*(?P<value>[^|<\n]{{2,40}})",
    re.IGNORECASE,
)

#: The elector's own name is the ``নাম`` label with no relation word in front of it. Written as
#: an optional prefix rather than as a negative lookbehind: the lookbehinds it replaces had to be
#: fixed width, so they could not have tolerated :data:`MATRA` on the relation words too.
NAME_RE = re.compile(
    rf"(?P<kind>{RELATION_WORDS})?\s*(?:নাম{MATRA}|NAME)\s*[:：.]?\s*(?P<value>[^|<\n]{{2,40}})",
    re.IGNORECASE,
)

RELATION_KIND = {
    "fathers": "father",
    "father's": "father",
    "mothers": "mother",
    "mother's": "mother",
    "husbands": "husband",
    "husband's": "husband",
    "পিতাৰ": "father",
    "স্বামীৰ": "husband",
    "মাতাৰ": "mother",
    "মাতৃৰ": "mother",
}

#: Assamese text long enough to be a name, used when the model omits the ``নাম :`` label.
BARE_NAME = re.compile(r"^[^|<>\n]*[ঀ-৿]{3,}[^|<>\n]*$")

#: Matched by substring, so **order matters**: ``MALE`` is inside ``FEMALE``, and listed first it
#: reads every woman on the English roll as a man. The Assamese pair has no such overlap, which is
#: why the hazard only appeared when English was added. Longest and most specific first.
SEX_WORDS = (
    ("FEMALE", "F"),
    ("MALE", "M"),
    ("THIRD", "T"),
    ("OTHER", "T"),
    ("পুৰুষ", "M"),
    ("পুরুষ", "M"),
    ("মহিলা", "F"),
    ("তৃতীয়", "T"),
)

MIN_AGE, MAX_AGE = 18, 120

#: A response repeating itself is a runaway even when it stopped short of the cap.
REPEAT = re.compile(r"(.{20,}?)\1{2,}", re.S)


#: The full Surya model answers in HTML and the distilled one in pipes. Both are normalized to
#: the same shape here rather than parsed twice: two readers for one format is the bug that put
#: the elector's own name in the relation field, and it would be the fourth time.
#: Only these end a line. ``<span>`` is inline -- treating it as a break severed ``নাম`` from
#: the value after it and the label became the name.
BREAK = re.compile(r"<br\s*/?>|</?p>|</?div>|</?tr>|</?li>", re.I)
ANY_TAG = re.compile(r"<[^>]+>")


def normalise(text: str) -> str:
    """One line-delimited shape, whether the engine answered in HTML or in pipes."""
    out = BREAK.sub("|", text)
    out = ANY_TAG.sub("", out)
    return out.replace("\xa0", " ")


def _clean_value(text: str) -> str:
    """Trim the punctuation the terse format leaves around a value."""
    return re.sub(r"\s+", " ", text).strip(" :|<>\"'৷,.").strip()


@dataclass
class Reading:
    """What the second pass made of one box, and whether it can be believed."""

    text: str
    tokens: int = 0
    capped: bool = False

    @property
    def settled(self) -> str:
        """The part of the response worth reading.

        A loop repeats a *complete* record -- ``বয়স: 54`` appears in every cycle with the same
        value -- so the first cycle is the answer and everything after it is the model failing
        to stop.

        On the page this was measured against it recovered no extra fields, because those rows
        already had values; what it bought was eleven more rows where the two engines could be
        compared at all. A response that hit the cap without ever repeating is different: it
        stopped mid-record and its tail may be half a number, so that one is refused.
        """
        repeat = REPEAT.search(self.text)
        if repeat:
            # Keep what came *before* the loop, not the loop itself. These models finish the
            # record and then pad -- "||\n||\n||\n" for hundreds of characters -- and returning
            # the repeated cycle handed back the padding and discarded the answer. Three
            # electors' names came out empty from responses that plainly contained them.
            head = self.text[: repeat.start()]
            return head if head.strip() else repeat.group(1)
        return "" if self.capped else self.text

    @property
    def usable(self) -> bool:
        return bool(self.settled.strip())

    def fields(self) -> Dict[str, Any]:
        text = normalise(self.settled)
        if not text.strip():
            return {}
        found: Dict[str, Any] = {}

        relation = RELATION_RE.search(text)
        if relation:
            value = _clean_value(relation.group("value"))
            if value:
                found["relation_name"] = value
                # The label may have arrived with either RA, so normalize before looking it
                # up -- and lowercased, because the English labels are matched case-insensitively
                # and arrive as printed ("Fathers"), which no key would match.
                kind = relation.group("kind").replace("র", "ৰ").lower()
                found["relation_type"] = RELATION_KIND.get(kind, "")

        name = next((m for m in NAME_RE.finditer(text) if not m.group("kind")), None)
        if name:
            value = _clean_value(name.group("value"))
            if value:
                found["name"] = value
        else:
            # The model often drops the label and leads with the name. Take the first field
            # that is Assamese text and is not one of the labeled ones.
            for part in (p.strip() for p in re.split(r"\||\n", text)):
                if not part or part.isdigit():
                    continue
                if re.search(_either_ra(rf"{HOUSE_WORD}|বয়স|লিঙ্গ|নাম{MATRA}\s*[:：]|<"), part):
                    continue
                if BARE_NAME.match(part):
                    value = _clean_value(part)
                    if value:
                        found["name"] = value
                    break

        age = AGE_RE.search(text)
        if age:
            digits = schema.normalize_digits(age.group(1))
            if digits.isdigit() and MIN_AGE <= int(digits) <= MAX_AGE:
                found["age"] = int(digits)
        house = HOUSE_RE.search(text)
        if house:
            value = schema.normalize_digits(house.group(1)).strip()
            if value:
                found["house_no"] = value
        sex = SEX_RE.search(text)
        if sex:
            value = sex.group(1).upper()
            for word, code in SEX_WORDS:
                if word.upper() in value:
                    found["sex"] = code
                    break
        return found


def wanted(row: Dict[str, Any]) -> bool:
    """Whether this row has something the second pass could plausibly recover.

    Deliberately narrower than the escalation router. The router flags anything doubtful; this
    asks only whether a field it can actually supply is missing, because a second read that
    returns what we already had is 6 seconds spent on nothing.
    """
    return row.get("age") is None or not row.get("house_no") or not row.get("sex")


def merge(row: Dict[str, Any], reading: Reading) -> Dict[str, Any]:
    """The row with recovered fields filled in, never overwritten.

    Gaps are filled from the second pass. A *disagreement* is resolved in its favor only for
    the fields in :data:`OVERRULED`, and only because the crops were read by eye and it won
    every one. Everywhere else the first pass's value stands.

    Both the old and the new value are recorded on an overrule, because the evidence behind it
    is six crops on one page: enough to act on, not enough to discard what it replaced.
    """
    out = dict(row)
    recovered, disagreed = [], []
    for field, value in reading.fields().items():
        if field not in RECOVERABLE:
            continue
        if out.get(field) in (None, ""):
            out[field] = value
            recovered.append(field)
        elif str(out[field]) != str(value):
            disagreed.append(f"{field}:{out[field]}!={value}")
            if field in OVERRULED:
                out[f"{field}_first_pass"] = out[field]
                out[field] = value

    flags = [f for f in (out.get("flags") or "").split(",") if f]
    if recovered:
        flags.append("second_pass_" + "_".join(recovered))
    if disagreed:
        flags.append("second_pass_disagreed")
    if any(d.split(":")[0] in OVERRULED for d in disagreed):
        flags.append("second_pass_overruled")
    out["flags"] = ",".join(flags)
    out["second_pass_disagreements"] = ";".join(disagreed)
    return out


def summarise(before: Sequence[Dict[str, Any]], after: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """What the second pass added, per field, and where it contradicted the first."""
    out: Dict[str, Any] = {"rows": len(after)}
    for field in RECOVERABLE:
        had = sum(1 for r in before if r.get(field) not in (None, ""))
        has = sum(1 for r in after if r.get(field) not in (None, ""))
        out[field] = {"before": had, "after": has, "recovered": has - had}
    out["disagreements"] = sum(1 for r in after if r.get("second_pass_disagreements"))
    out["overruled"] = sum(1 for r in after if "second_pass_overruled" in (r.get("flags") or ""))
    return out


def render(found: Dict[str, Any]) -> str:
    lines = [f"SECOND PASS over {found['rows']:,} rows", ""]
    for field in RECOVERABLE:
        stat = found[field]
        total = found["rows"] or 1
        lines.append(
            f"   {field:<10} {stat['before']:>6,} -> {stat['after']:>6,} "
            f"({stat['before'] / total:.1%} -> {stat['after'] / total:.1%}, "
            f"+{stat['recovered']:,})"
        )
    lines += [
        "",
        f"   rows where the two engines contradicted each other: {found['disagreements']:,}",
        f"   of those, ages the second pass overruled: {found['overruled']:,}",
        "   the first pass's value is kept alongside, in <field>_first_pass",
    ]
    return "\n".join(lines)


def parse_reading(payload: Dict[str, Any]) -> Optional[Reading]:
    if "text" not in payload:
        return None
    return Reading(
        text=payload.get("text") or "",
        tokens=int(payload.get("tokens") or 0),
        capped=bool(payload.get("capped")),
    )


def readings_from(payloads: Sequence[Dict[str, Any]]) -> List[Reading]:
    return [r for r in (parse_reading(p) for p in payloads) if r is not None]


class TerseEngine:
    """savitr's distilled roll model, held resident in its own virtualenv.

    Its dependencies conflict with this package's, so it runs as a subprocess and speaks one
    JSON object per line. The model stays loaded between calls, which is the whole point: a
    process per box would pay about a second of model load for six seconds of work, on boxes
    numbering in the tens of thousands.
    """

    def __init__(
        self,
        python: str = ".venv-surya/bin/python",
        worker: str = "scripts/surya_worker.py",
        max_tokens: int = 160,
    ) -> None:
        self._python, self._worker, self._max_tokens = python, worker, max_tokens
        self._process: Optional[Any] = None

    def _start(self) -> Any:
        import subprocess
        from pathlib import Path

        if self._process and self._process.poll() is None:
            return self._process
        if not Path(self._python).exists():
            raise RuntimeError(
                f"{self._python} not found; create it with `uv venv --python 3.12 .venv-surya "
                "&& uv pip install --python .venv-surya/bin/python 'savitr[backend]'`"
            )
        self._process = subprocess.Popen(
            [
                self._python,
                self._worker,
                "--terse",
                "--max-tokens",
                str(self._max_tokens),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        ready = self._process.stdout.readline()
        if not ready or "ready" not in ready:
            raise RuntimeError(f"surya worker failed to start: {ready!r}")
        return self._process

    def read(self, image: Any) -> Reading:
        import base64
        import io
        import json as _json

        process = self._start()
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        process.stdin.write(
            _json.dumps({"png": base64.b64encode(buffer.getvalue()).decode("ascii")}) + "\n"
        )
        process.stdin.flush()
        reply = _json.loads(process.stdout.readline())
        if "error" in reply:
            # One unreadable box must not end a run of tens of thousands.
            return Reading(text="", capped=True)
        return parse_reading(reply) or Reading(text="", capped=True)

    def close(self) -> None:
        if self._process and self._process.poll() is None:
            self._process.terminate()
            self._process.wait(timeout=10)
