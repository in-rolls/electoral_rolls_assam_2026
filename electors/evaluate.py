"""Scoring an engine against what the page actually says.

Everything else in this stage measures agreement, coverage or self-consistency, because no
labeled data exists. That was a real limit and it hid a real problem: the name field sat at
"91.5% present and not provably wrong" for the whole of this stage's development, and when a
second engine finally made comparison possible, the two disagreed on 59% of names. Reading the
crops settled which was right, and it was not the first pass.

So this is the small piece of genuine ground truth: a random sample of boxes, cropped and read
by eye, scored per field. Sixteen boxes across four parts is not a benchmark -- it is enough to
tell a 0% from a 90%, which is what the numbers turned out to need.

**Sampled at random, never from the disagreements.** Scoring the subset where two engines differ
measures that subset. The first version of this comparison did exactly that and would have
reported the disagreement rate as the error rate.
"""

from __future__ import annotations

import difflib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

#: Bengali numerals, which both engines emit for the same fields as Latin ones.
BENGALI_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

#: How close a name has to be to count as "nearly right" -- the same reading with a different
#: matra, rather than a different person.
NEARLY = 0.80

#: Bengali RA (U+09B0) and Assamese RA (U+09F0) are the same letter written two ways, and the
#: models emit the Bengali codepoint for Assamese text. Scoring them apart marked correct
#: readings wrong: ``পবিত্র লাক্রা`` against ``পবিত্ৰ লাক্ৰা`` differs in nothing else. Folding it
#: moves surya-full from 6 of 16 exact names to 10.
#:
#: Only RA. Bengali ``ব`` and Assamese ``ৱ`` look like a matching pair but are distinct letters
#: in Assamese, and folding them changed no score -- which is the evidence for leaving them
#: alone rather than an argument for it.
SCRIPT_VARIANTS = str.maketrans({"র": "ৰ"})


def fold(text: str) -> str:
    """Normalise script variants that are the same letter, and collapse whitespace."""
    return " ".join(str(text or "").translate(SCRIPT_VARIANTS).split())


@dataclass
class Score:
    """One engine's tally over the sample."""

    engine: str
    total: int = 0
    name_exact: int = 0
    name_nearly: int = 0
    first_name: int = 0
    age: int = 0
    house: int = 0
    sex: int = 0

    def rates(self) -> Dict[str, float]:
        n = self.total or 1
        return {
            "name exactly right": self.name_exact / n,
            "name nearly right": self.name_nearly / n,
            "first name right": self.first_name / n,
            "age right": self.age / n,
            "house no right": self.house / n,
            "sex right": self.sex / n,
        }


def terse_fields(text: str) -> Dict[str, Any]:
    """The fields out of an engine's answer, whatever shape it arrived in.

    Delegates to :mod:`electors.second_pass`, which is the one place this format is read. An
    earlier version of this module carried its own copy, and two readers for one format is
    exactly the bug that put the elector's own name in the relation field.
    """
    from . import second_pass

    found = second_pass.Reading(text).fields()
    return {
        "name": found.get("name", ""),
        "age": found.get("age"),
        "house": found.get("house_no", ""),
        "sex": found.get("sex", ""),
    }


def score_one(got: Dict[str, Any], want: Dict[str, Any], tally: Score) -> None:
    tally.total += 1
    name, truth = fold(got.get("name")), fold(want["name"])
    tally.name_exact += bool(name) and name == truth
    tally.name_nearly += bool(name) and difflib.SequenceMatcher(None, name, truth).ratio() >= NEARLY
    tally.first_name += bool(name) and name.split()[0] == truth.split()[0]
    tally.age += got.get("age") == want["age"]
    # Whitespace folded: a house number reads "20 ক" or "20ক" on the same page.
    tally.house += fold(got.get("house")).replace(" ", "") == fold(want["house"]).replace(" ", "")
    tally.sex += (got.get("sex") or "") == want["sex"]


def score(
    truth: Dict[str, Dict[str, Any]],
    sample: Sequence[Dict[str, Any]],
) -> Dict[str, Score]:
    """Both engines over every box that has been read by eye."""
    by_key = {row["key"]: row for row in sample}
    tallies = {
        "tesseract": Score("tesseract"),
        "savitr-terse": Score("savitr-terse"),
        "surya-full": Score("surya-full"),
        "gemini": Score("gemini"),
        "dots.ocr": Score("dots.ocr"),
        "cloud-vision": Score("cloud-vision"),
    }
    for key, want in truth.items():
        row = by_key.get(key)
        if not row:
            continue
        score_one(
            {
                "name": row.get("t_name"),
                "age": row.get("t_age"),
                "house": row.get("t_house"),
                "sex": row.get("t_sex"),
            },
            want,
            tallies["tesseract"],
        )
        score_one(terse_fields(row.get("surya") or ""), want, tallies["savitr-terse"])
        if row.get("surya_full"):
            score_one(terse_fields(row["surya_full"]), want, tallies["surya-full"])
        if row.get("gemini"):
            from . import gemini as gemini_engine

            score_one(gemini_engine.fields_of(row["gemini"]), want, tallies["gemini"])
        if row.get("dots"):
            score_one(terse_fields(row["dots"]), want, tallies["dots.ocr"])
        if row.get("vision"):
            score_one(terse_fields(row["vision"]), want, tallies["cloud-vision"])
    return {name: t for name, t in tallies.items() if t.total}


def common_keys(truth: Dict[str, Any], sample: Sequence[Dict[str, Any]]) -> List[str]:
    """Boxes every engine answered, which is the only fair basis for comparing them.

    Engines arrive at different times and the sample grows, so one column can be scored on 16
    boxes and another on 36. Comparing those is the same error as scoring two models at
    different token budgets -- which happened here, and reversed the ranking.
    """
    fields = ("t_name", "surya", "surya_full", "gemini", "dots", "vision")
    by_key = {row["key"]: row for row in sample}
    return [key for key in truth if key in by_key and all(by_key[key].get(f) for f in fields)]


def render(tallies: Dict[str, Score]) -> str:
    names = list(tallies)
    total = max(t.total for t in tallies.values())
    uneven = {t.total for t in tallies.values() if t.total}
    lines = [
        f"ENGINE ACCURACY over {total} boxes read off the page",
        "",
        f"   {'field':<22}" + "".join(f"{n:>14}" for n in names),
    ]
    for field in tallies[names[0]].rates():
        row = f"   {field:<22}"
        for name in names:
            tally = tallies[name]
            hits = int(round(tally.rates()[field] * (tally.total or 1)))
            row += f"{hits:>6}/{tally.total} ({tally.rates()[field]:>3.0%})".rjust(14)
        lines.append(row)
    lines += [""]
    if len(uneven) > 1:
        lines += [
            "   *** COLUMNS ARE NOT COMPARABLE: they were scored on different numbers of",
            f"   *** boxes ({sorted(uneven)}). Score the common subset before ranking these.",
            "",
        ]
    lines += [
        "   Ground truth is the crop, read by eye. Small, and the only real truth here.",
        "   Compare engines only at the same token budget: a smaller one truncates fewer",
        "   answers into loops, and the loop, not the model, is what the score then measures.",
    ]
    return "\n".join(lines)


def load(directory: Path = Path("out/eval")) -> Optional[Dict[str, Any]]:
    truth_path, sample_path = directory / "truth.json", directory / "sample.json"
    if not truth_path.exists() or not sample_path.exists():
        return None
    return {
        "truth": json.loads(truth_path.read_text(encoding="utf-8")),
        "sample": json.loads(sample_path.read_text(encoding="utf-8")),
    }
