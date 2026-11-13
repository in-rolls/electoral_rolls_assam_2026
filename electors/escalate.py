"""Which rows are worth paying more to re-read.

The cheap pass will always leave errors, and pushing it to perfection has sharply diminishing
returns. What makes leaving them acceptable is **knowing which rows they are**, so an
expensive engine runs on those and only those. That makes the detector as important as the
extractor, and it has to be measured before it is trusted.

Two independent families of signal feed it:

``certain``  the floor detectors -- Latin letters in an Assamese name, a name identical to
             the relation, a malformed or duplicated EPIC, a leaked label. These are provably
             wrong without reference to the source.
``doubtful`` the two-scale disagreement flags. Nothing here is known to be wrong; the two
             upscales simply did not produce the same string.

**On measuring this honestly.** Scoring the router's precision against the floor detectors
would be a check that cannot fail: the floor detectors *are* half the router, so it would
report 100% and mean nothing. That trap has already been walked into twice in this stage, so
what is reported here is what can be established without a second opinion -- the volume, and
how much each family contributes on its own -- and precision is left explicitly unmeasured
until an independent engine has read the same rows. :func:`agreement_against` is where that
number comes from once it has.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence, Set

from . import quality

#: Flags raised by the extractor when two upscales disagreed. Independent of the floor
#: detectors, which look only at the final value.
DOUBT_FLAGS = (
    "name_disagreement",
    "relation_disagreement",
    "unreadable",
    "age_ambiguous",
    "age_unreadable",
    "name_repaired",
    "relation_repaired",
)


def certain(row: Dict[str, Any]) -> List[str]:
    """Reasons this row is provably wrong, needing no second opinion to establish."""
    return list(quality.problems_with(row))


def doubtful(row: Dict[str, Any]) -> List[str]:
    """Reasons to doubt the row that are not proof of anything."""
    flags = (row.get("flags") or "").split(",")
    return [flag for flag in DOUBT_FLAGS if flag in flags]


def unlabelled_relation(row: Dict[str, Any]) -> List[str]:
    """A relation value whose label was never recognized -- so it may be the elector's own.

    Not a floor detector: a genuinely unreadable label produces this too, and the value is
    usually right. It earns a place because of how it fails when it is wrong. When the label
    is missed, the relation line can be filed as the *name* and the name line as the relation,
    and the fallback then fills ``relation_name`` with the elector's own name. Both fields
    come out populated and plausible, so nothing in the floor catches it and ``name`` never
    equals ``relation`` -- the swap is invisible to every check that exists.

    ``relation_type`` empty beside a populated ``relation_name`` is the one visible trace.
    """
    if row.get("relation_name") and not row.get("relation_type"):
        return ["relation_type_unknown"]
    return []


def missing_core(row: Dict[str, Any]) -> List[str]:
    """Fields absent that the row plainly ought to have.

    Empty is not wrong in the way a Latin-lettered name is wrong -- the box may genuinely be
    blank -- but an elector with no name and no EPIC is not a usable record either, and a
    second pass is exactly what might recover it.
    """
    absent = [field for field in ("epic_no", "name", "age", "sex") if not row.get(field)]
    return [f"no_{field}" for field in absent] if len(absent) >= 2 else []


def reasons(row: Dict[str, Any]) -> List[str]:
    return certain(row) + doubtful(row) + unlabelled_relation(row) + missing_core(row)


def needs_escalation(row: Dict[str, Any]) -> bool:
    return bool(reasons(row))


#: Bands of text per elector box: name, relation, house, age.
BANDS_PER_BOX = 4

#: Which band a reason implicates. The expensive pass does not need to re-read a whole box to
#: settle one field, so the cost that matters is bands re-read, not rows flagged. Counting rows
#: alone made a precise age detector look like it had pushed the router past the point of being
#: a triage, when it had added one band to 16% of boxes.
REASON_FIELD: Dict[str, str] = {
    "name_has_latin_or_digits": "name",
    "name_contains_label": "name",
    "name_equals_relation": "name",
    "name_implausible_length": "name",
    "name_disagreement": "name",
    "name_repaired": "name",
    "no_name": "name",
    "relation_has_latin_or_digits": "relation",
    "relation_contains_label": "relation",
    "relation_type_unknown": "relation",
    "relation_disagreement": "relation",
    "relation_repaired": "relation",
    "no_relation_name": "relation",
    "age_out_of_range": "age",
    "age_ambiguous": "age",
    "age_unreadable": "age",
    "no_age": "age",
    "no_sex": "age",
    "epic_malformed": "epic",
    "epic_duplicated": "epic",
    "no_epic_no": "epic",
    "unreadable": "box",
}


@dataclass
class RouterReport:
    """What the router does to a corpus, and what about it is not yet known."""

    rows: int
    flagged: int
    by_family: Dict[str, int]
    only_family: Dict[str, int]
    top_reasons: List[tuple]
    by_field: Dict[str, int]

    @property
    def volume(self) -> float:
        return self.flagged / self.rows if self.rows else 0.0

    def __str__(self) -> str:
        lines = [
            "ESCALATION ROUTER",
            "",
            f"   volume            {self.flagged:,} of {self.rows:,} rows ({self.volume:.1%})",
            "",
            "   by signal family (rows may appear in more than one)",
        ]
        for family, count in self.by_family.items():
            alone = self.only_family.get(family, 0)
            lines.append(
                f"      {family:<10} {count:>7,}  ({alone:,} flagged by this family alone)"
            )
        lines += ["", "   bands the second pass would re-read (not whole boxes)"]
        for band, count in sorted(self.by_field.items(), key=lambda kv: -kv[1]):
            lines.append(f"      {band:<10} {count:>7,}  ({count / max(1, self.rows):.1%} of rows)")
        bands = sum(self.by_field.values())
        share = bands / (BANDS_PER_BOX * self.rows) if self.rows else 0.0
        lines += [
            "",
            f"      {'total':<10} {bands:>7,}  bands, against {BANDS_PER_BOX * self.rows:,} "
            f"for a full re-run -- {share:.1%} of one",
            "",
            "   commonest reasons",
        ]
        for reason, count in self.top_reasons:
            lines.append(f"      {reason:<28} {count:>7,}")
        lines += [
            "",
            "   precision is NOT reported. The floor detectors are half the router, so",
            "   scoring the router against them is a check that cannot fail. It becomes",
            "   measurable once a second engine has read the flagged rows.",
        ]
        return "\n".join(lines)


def report(rows: Sequence[Dict[str, Any]]) -> RouterReport:
    """Volume and composition -- everything establishable without a second engine."""
    families = {
        "certain": certain,
        "doubtful": doubtful,
        "unlabelled": unlabelled_relation,
        "missing": missing_core,
    }
    hits: Dict[str, Set[int]] = {name: set() for name in families}
    reason_counts: Dict[str, int] = {}

    for index, row in enumerate(rows):
        for name, detect in families.items():
            found = detect(row)
            if found:
                hits[name].add(index)
            for reason in found:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1

    flagged = set().union(*hits.values()) if hits else set()
    only = {
        name: len(indexes - set().union(*(other for key, other in hits.items() if key != name)))
        for name, indexes in hits.items()
    }
    fields_hit: Dict[str, Set[int]] = {}
    for index, row in enumerate(rows):
        for reason in reasons(row):
            band = REASON_FIELD.get(reason, "box")
            fields_hit.setdefault(band, set()).add(index)

    return RouterReport(
        by_field={band: len(indexes) for band, indexes in fields_hit.items()},
        rows=len(rows),
        flagged=len(flagged),
        by_family={name: len(indexes) for name, indexes in hits.items()},
        only_family=only,
        top_reasons=sorted(reason_counts.items(), key=lambda kv: -kv[1])[:10],
    )


def separation(
    rows: Sequence[Dict[str, Any]],
    flag: str,
    field: str,
    implausible: Callable[[Any], bool],
) -> Dict[str, Any]:
    """How much more often a flagged row holds an implausible value than an unflagged one.

    The router's precision cannot be measured without a second engine, but a detector can
    still be falsified: if the rows it picks are no likelier to be wrong than the rows it
    passes over, it is flagging at random however sensible its reason sounds. Where a value is
    individually plausible but collectively impossible -- an age of 95 that is really 26 sits
    inside every range check -- this is the only handle there is.

    ``implausible`` is a property of the *distribution*, not of the row: "in the nineties" is
    not an error, but 15% of a roll being in the nineties is.
    """
    picked = [r for r in rows if flag in (r.get("flags") or "").split(",")]
    passed = [r for r in rows if flag not in (r.get("flags") or "").split(",")]
    rates = {}
    for label, subset in (("flagged", picked), ("unflagged", passed)):
        values = [r[field] for r in subset if r.get(field) is not None]
        rates[label] = {
            "rows": len(values),
            "implausible": (
                sum(1 for v in values if implausible(v)) / len(values) if values else 0.0
            ),
        }
    base = rates["unflagged"]["implausible"]
    picked_rate = rates["flagged"]["implausible"]
    if base:
        concentration = picked_rate / base
    elif picked_rate:
        # Every implausible value is inside the flagged set and none outside it. That is
        # perfect separation, and the first version of this reported it as 0.0 -- the score a
        # useless detector gets -- because it divided by the base without asking whether the
        # base was zero.
        concentration = float("inf")
    else:
        concentration = 1.0
    return {
        **rates,
        # Above 1 the detector is finding something; at 1 it is flagging at random.
        "concentration": concentration,
    }


#: Fields compared when a second engine re-reads a row. The EPIC is excluded: it is read with
#: a different model in a different script and its agreement says nothing about the Assamese.
COMPARED = ("name", "relation_name", "age", "sex")


def agreement_against(
    cheap: Sequence[Dict[str, Any]],
    expensive: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Where the two passes agree, keyed by whether the router flagged the row.

    This is what turns the router from asserted into measured. If the flagged rows disagree
    far more often than the unflagged ones, the router is finding something; if they disagree
    at the same rate, it is flagging at random however sensible its reasons sound.

    Agreement is still not accuracy -- two engines can be wrong together -- and it is reported
    under that name.
    """
    paired = list(zip(cheap, expensive))
    out: Dict[str, Any] = {"pairs": len(paired)}
    for group, subset in (
        ("flagged", [p for p in paired if needs_escalation(p[0])]),
        ("unflagged", [p for p in paired if not needs_escalation(p[0])]),
    ):
        scores = {}
        for field in COMPARED:
            comparable = [(a, b) for a, b in subset if a.get(field) or b.get(field)]
            if comparable:
                agreed = sum(1 for a, b in comparable if a.get(field) == b.get(field))
                scores[field] = agreed / len(comparable)
        out[group] = {"rows": len(subset), "agreement": scores}

    lift = {}
    for field in COMPARED:
        unflagged = out["unflagged"]["agreement"].get(field)
        flagged = out["flagged"]["agreement"].get(field)
        if unflagged and flagged is not None:
            # Below 1.0 means the router is picking rows the engines argue about, which is
            # what it is supposed to do.
            lift[field] = flagged / unflagged
    out["flagged_agreement_ratio"] = lift
    return out
