"""Cross-engine field-level benchmark.

Modeled on ``parse_unsearchable_rolls/scripts/manipur/benchmark/score.py``. Every engine
is driven through the **same** grid detection and the same field parser, so the numbers
isolate OCR quality rather than parsing.

**These are agreement figures, not accuracy figures.** There is no human-labelled gold
set here: a candidate is scored on how often it matches a *declared reference* engine.
The Manipur benchmark did the same thing and said so plainly ("the share of each field
that exactly matches Surya"). Where reference and candidate share a failure they will
agree and both be wrong, so a high score is a necessary but not sufficient condition.

Three metrics per field, mirroring the ``name`` / ``name_nospace`` / ``name_charsim``
split used there, adapted to Assamese:

``exact``
    byte-equal after Unicode NFC normalization and whitespace collapse.
``skeleton``
    compares with all combining marks stripped, so a misplaced diacritic (গাঁও/গাওঁ)
    counts as a match while a wrong letter does not -- the same job ``name_nospace``
    does for ALL-CAPS spacing in the Manipur benchmark.
``charsim``
    mean character-level similarity, so near-misses are visible rather than binary.
"""

from __future__ import annotations

import difflib
import json
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

#: Fields worth comparing: the Assamese text where engines actually differ.
TEXT_FIELDS = (
    "ac_name",
    "pc_name",
    "district",
    "revenue_circle",
    "block",
    "police_station",
    "post_office",
    "main_town_village",
    "ps_name",
    "ps_address",
)

#: Numeric fields, kept in the report because they are independently verifiable.
NUMERIC_FIELDS = (
    "electors_male",
    "electors_female",
    "electors_third_gender",
    "electors_total",
    "pin_code",
)


def nfc(text: Any) -> str:
    """NFC-normalise and collapse whitespace."""
    if text is None:
        return ""
    return " ".join(unicodedata.normalize("NFC", str(text)).split())


def skeleton(text: Any) -> str:
    """The letter skeleton: combining marks removed.

    Engines disagree about diacritic **placement** -- গাঁও versus গাওঁ, where the
    candrabindu attaches to a different base character. Those are genuinely different
    codepoint sequences (গ া ঁ ও vs গ া ও ঁ), so no amount of reordering within a cluster
    makes them equal; only dropping the marks does.

    Comparing skeletons therefore separates "right letters, misplaced diacritic" from
    "wrong letter", which is the distinction that matters when judging an engine:

        গাঁও   vs গাওঁ       -> same skeleton  (recoverable placement artifact)
        কোকৰাঝাৰ vs কোকৰাব্মাৰ -> different     (a real misread, ঝ became ব্ম)

    This is the Assamese analog of the ``name_nospace`` metric in the Manipur
    benchmark, which isolated an ALL-CAPS spacing artifact the same way.
    """
    decomposed = unicodedata.normalize("NFD", nfc(text))
    return "".join(
        char
        for char in decomposed
        if not (unicodedata.combining(char) or unicodedata.category(char) in ("Mn", "Mc"))
    )


def charsim(a: Any, b: Any) -> float:
    return difflib.SequenceMatcher(None, nfc(a), nfc(b)).ratio()


@dataclass
class FieldScore:
    field: str
    n: int = 0
    exact: int = 0
    skeleton_match: int = 0
    similarity: float = 0.0
    both_blank: int = 0

    def add(self, ref: Any, cand: Any) -> None:
        self.n += 1
        if not nfc(ref) and not nfc(cand):
            self.both_blank += 1
        self.exact += nfc(ref) == nfc(cand)
        self.skeleton_match += skeleton(ref) == skeleton(cand)
        self.similarity += charsim(ref, cand)

    def as_row(self) -> Dict[str, Any]:
        n = max(1, self.n)
        return {
            "field": self.field,
            "n": self.n,
            "exact": self.exact / n,
            "skeleton": self.skeleton_match / n,
            "charsim": self.similarity / n,
            "both_blank": self.both_blank,
        }


@dataclass
class EngineScore:
    engine: str
    seconds_per_page: float = 0.0
    fields: Dict[str, FieldScore] = field(default_factory=dict)

    def add(self, ref_row: Dict[str, Any], cand_row: Dict[str, Any], names: Sequence[str]) -> None:
        for name in names:
            self.fields.setdefault(name, FieldScore(name)).add(
                ref_row.get(name), cand_row.get(name)
            )

    def rows(self) -> List[Dict[str, Any]]:
        return [self.fields[name].as_row() for name in self.fields]

    def mean(self, metric: str) -> float:
        rows = self.rows()
        return sum(r[metric] for r in rows) / max(1, len(rows))


def score_engine(
    reference: Dict[Any, Dict[str, Any]],
    candidate: Dict[Any, Dict[str, Any]],
    engine_name: str,
    names: Sequence[str] = TEXT_FIELDS,
    seconds_per_page: float = 0.0,
) -> EngineScore:
    """Score ``candidate`` against ``reference``, both keyed by (ac_no, part_no)."""
    score = EngineScore(engine_name, seconds_per_page=seconds_per_page)
    for key in sorted(set(reference) & set(candidate)):
        score.add(reference[key], candidate[key], names)
    return score


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def render_report(
    scores: Iterable[EngineScore],
    reference_name: str,
    n_pages: int,
    notes: Sequence[str] = (),
) -> str:
    """Render the benchmark as markdown."""
    scores = list(scores)
    lines = [
        "# OCR engine benchmark — Assam 2026 roll info pages",
        "",
        f"**Reference (declared, not ground truth):** `{reference_name}` over {n_pages} pages.",
        "",
        "Scores are the share of each field **matching the reference**, so they measure",
        "agreement, not correctness — where both engines share a failure they agree and",
        "are both wrong. `skeleton` additionally ignores combining marks entirely",
        "(গাঁও/গাওঁ), isolating misplaced diacritics from genuinely wrong letters.",
        "",
    ]
    for score in scores:
        lines += [
            f"## {score.engine}",
            "",
            f"{score.seconds_per_page:.2f} s/page · mean exact {_pct(score.mean('exact'))} · "
            f"mean skeleton {_pct(score.mean('skeleton'))} · "
            f"mean charsim {score.mean('charsim'):.3f}",
            "",
            "| field | n | exact | skeleton | charsim | both blank |",
            "|---|--:|--:|--:|--:|--:|",
        ]
        for row in score.rows():
            lines.append(
                f"| {row['field']} | {row['n']} | {_pct(row['exact'])} | "
                f"{_pct(row['skeleton'])} | {row['charsim']:.3f} | {row['both_blank']} |"
            )
        lines.append("")
    if notes:
        lines += ["## Notes", ""] + [f"- {n}" for n in notes] + [""]
    return "\n".join(lines)


def write_report(path: Path, text: str, scores: Iterable[EngineScore]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    payload = [
        {
            "engine": s.engine,
            "seconds_per_page": s.seconds_per_page,
            "fields": s.rows(),
        }
        for s in scores
    ]
    path.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
