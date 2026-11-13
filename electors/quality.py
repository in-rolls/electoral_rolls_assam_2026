"""How good the extracted fields actually are, without ground truth.

Every quality figure quoted for this stage until now was a **fill rate** -- whether the field
came back non-empty. ``epic=83%`` says 83% of rows have something in the EPIC column. It says
nothing about whether that something is the elector's EPIC, and an 83% fill is equally
consistent with 83% correct and 40% correct.

There is no labeled data here, so accuracy cannot be measured directly. It can be
**bracketed**, which is worth more than another fill rate:

**Lower bound -- values that cannot be right.** An Assamese name does not contain Latin
letters. An elector's own name is not identical to their father's. An EPIC does not repeat
within a constituency. Each of these is certainly an error, so their combined rate is a floor
under the true error rate.

**Upper bound -- disagreement.** Names are read at two scales and compared; where two
independent passes produce the same string the value is very unlikely to be wrong. The
disagreement rate is therefore a ceiling.

**And two fields do have real ground truth**, which is why they are reported apart from the
bracket rather than mixed into it. The roll's closing page publishes its own male/female
split and its own total, so the sex ratio and completeness can be checked outright. That is
not a nicety: fill rate showed sex at a healthy 87% while it was running 44.2% male against a
published 51%, and only the ratio caught it.

**Distributions catch what neither bound sees.** A per-row check cannot notice that ages
ending in one digit are over-represented, but the model doubles digits it is unsure of --
``55`` read as ``525`` -- so systematic digit confusion shows up in the histogram and nowhere
else.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence

from . import floor

#: Assamese text. A name made only of these is at least plausible; one containing Latin
#: letters or digits is not.
LATIN = re.compile(r"[A-Za-z]")
DIGIT = re.compile(r"[0-9০-৯]")

EPIC_RE = re.compile(r"^[A-Z]{2,4}\d{6,9}$")

MIN_AGE, MAX_AGE = 18, 120
MIN_NAME, MAX_NAME = 2, 60


def problems_with(row: Dict[str, Any]) -> List[str]:
    """Every way this row is certainly wrong, whatever the source says."""
    problems: List[str] = []
    name = row.get("name") or ""
    relation = row.get("relation_name") or ""
    epic = row.get("epic_no") or ""
    age = row.get("age")
    latin_is_wrong = row.get("lang") != "ENG"

    if name and ((latin_is_wrong and LATIN.search(name)) or DIGIT.search(name)):
        problems.append("name_has_latin_or_digits")
    if relation and ((latin_is_wrong and LATIN.search(relation)) or DIGIT.search(relation)):
        problems.append("relation_has_latin_or_digits")
    if name and floor.has_leaked_label(row, name):
        problems.append("name_contains_label")
    if relation and floor.has_leaked_label(row, relation):
        problems.append("relation_contains_label")
    if name and relation and name == relation:
        # Seen in real output: তন্ডা ৰাভা as both. One band was read twice.
        problems.append("name_equals_relation")
    if name and not (MIN_NAME <= len(name) <= MAX_NAME):
        problems.append("name_implausible_length")
    if epic and not EPIC_RE.match(epic):
        problems.append("epic_malformed")
    if age is not None and not (MIN_AGE <= age <= MAX_AGE):
        problems.append("age_out_of_range")
    return problems


def definite_errors(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """The floor under the error rate: rows provably wrong, by kind."""
    kinds: Counter = Counter()
    affected = 0
    for row in rows:
        problems = problems_with(row)
        if problems:
            affected += 1
            kinds.update(problems)

    # EPIC uniqueness is a property of the set, not of a row, so it is counted separately.
    epics = [r.get("epic_no") for r in rows if r.get("epic_no")]
    repeats = sum(count - 1 for count in Counter(epics).values() if count > 1)
    if repeats:
        kinds["epic_duplicated"] = repeats

    total = len(rows) or 1
    return {
        "rows_definitely_wrong": affected,
        "rate": affected / total,
        "by_kind": dict(kinds.most_common()),
        "duplicate_epics": repeats,
    }


def disagreement(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """The ceiling: how often two independent reads of the same crop differed."""
    flags = Counter(f for r in rows for f in (r.get("flags") or "").split(",") if f)
    total = len(rows) or 1
    return {
        "name_disagreement": flags.get("name_disagreement", 0) / total,
        "relation_disagreement": flags.get("relation_disagreement", 0) / total,
        "unreadable": flags.get("unreadable", 0) / total,
        "flags": dict(flags.most_common()),
    }


def distributions(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Shapes that reveal systematic misreading no per-row check can see."""
    ages = [r["age"] for r in rows if r.get("age")]
    decades = Counter((a // 10) * 10 for a in ages)
    last_digits = Counter(a % 10 for a in ages)

    houses = Counter((r.get("part_no"), r.get("house_no")) for r in rows if r.get("house_no"))
    household_sizes = Counter(houses.values())
    singles = household_sizes.get(1, 0)

    return {
        "age_by_decade": dict(sorted(decades.items())),
        # A flat spread is expected; a spike means digits are being confused for one another.
        "age_last_digit": dict(sorted(last_digits.items())),
        "age_last_digit_spread": (
            (max(last_digits.values()) - min(last_digits.values())) / max(1, len(ages))
            if last_digits
            else 0.0
        ),
        "single_occupant_share": singles / max(1, len(houses)),
    }


FIELDS: Sequence[str] = ("epic_no", "name", "relation_name", "house_no", "age", "sex")


def fill_rates(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Kept only so it can be shown beside the error rates it was mistaken for."""
    total = len(rows) or 1
    return {field: sum(1 for r in rows if r.get(field)) / total for field in FIELDS}


def sound_rates(rows: Sequence[Dict[str, Any]]) -> Dict[str, float]:
    """Per field: present **and not provably wrong**. The right thing to guard on.

    A fill rate punishes the correct move. Clearing a name that is provably not the
    elector's name -- because it is identical to the father's, or contains Latin letters --
    lowers ``name_present`` while making the data better, so a gate guarding fill rate would
    reject it. This does not: a wrong value contributes nothing either way, so removing one
    is neutral and replacing it with a right one is a gain.
    """
    total = len(rows) or 1
    out: Dict[str, float] = {}
    for field in FIELDS:
        good = 0
        for row in rows:
            if not row.get(field):
                continue
            problems = problems_with(row)
            if not any(field.split("_")[0] in problem for problem in problems):
                good += 1
        out[f"{field}_sound"] = good / total
    return out


def report(rows: Iterable[Dict[str, Any]], reconciliation: Dict[str, Any]) -> Dict[str, Any]:
    """Everything measurable about a sample, grouped by how much each group proves."""
    rows = list(rows)
    return {
        "rows": len(rows),
        # Real ground truth: the roll publishes these itself.
        "measured": {
            "parts_matching_roll": reconciliation.get("parts_matching_roll"),
            "parts_measured": reconciliation.get("parts_measured"),
            "male_share": reconciliation.get("male_share"),
            "roll_male_share": reconciliation.get("roll_male_share"),
        },
        "definitely_wrong": definite_errors(rows),
        "disagreement": disagreement(rows),
        "distributions": distributions(rows),
        "fill_rates": fill_rates(rows),
        # Present *and* not provably wrong -- what a gate should guard, because a fill rate
        # falls when a wrong value is correctly removed.
        "sound_rates": sound_rates(rows),
    }


def format_report(data: Dict[str, Any]) -> str:
    """The report as text, with each group labeled by what it can prove."""
    measured = data["measured"]
    wrong = data["definitely_wrong"]
    disagree = data["disagreement"]
    dist = data["distributions"]

    lines = [f"{data['rows']:,} rows", "", "MEASURED (the roll publishes these itself)"]
    if measured.get("parts_measured"):
        lines.append(
            f"   completeness   {measured['parts_matching_roll']}/{measured['parts_measured']}"
            f" parts match the roll's own total"
        )
    if measured.get("roll_male_share"):
        lines.append(
            f"   male share     {measured['male_share']:.1%} extracted vs "
            f"{measured['roll_male_share']:.1%} on the roll"
        )

    lines += ["", f"ERROR FLOOR -- provably wrong: {wrong['rate']:.1%} of rows"]
    for kind, count in wrong["by_kind"].items():
        lines.append(f"   {kind:<32} {count:,}")

    lines += [
        "",
        "ERROR CEILING -- two reads of the same crop disagreed",
        f"   name           {disagree['name_disagreement']:.1%}",
        f"   relation       {disagree['relation_disagreement']:.1%}",
        "",
        "DISTRIBUTIONS",
        f"   age by decade      {dist['age_by_decade']}",
        f"   age last digit     {dist['age_last_digit']}",
        f"   single-occupant houses {dist['single_occupant_share']:.1%}",
        "",
        "FILL RATES (not accuracy -- shown for contrast)",
    ]
    for field, rate in data["fill_rates"].items():
        lines.append(f"   {field:<16} {rate:.1%}")
    lines += [
        "",
        "Name accuracy is not measured here: the true rate lies between the floor and the",
        "ceiling above, and no labelled data exists to pin it down.",
    ]
    return "\n".join(lines)
