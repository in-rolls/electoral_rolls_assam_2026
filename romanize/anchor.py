"""Match a native value to an official English name, within a scope small enough to be safe.

Fuzzy matching a name against a national gazetteer is a good way to produce confident
nonsense. What makes it defensible here is that every match is **scoped by a key the corpus
already carries**: a post office is compared only against the offices sharing its pincode --
a median of twelve candidates, not 155,000. At that size the question stops being "what does
this say" and becomes "which of these twelve is it", which edit distance can answer.

Two preparations are needed before the comparison is even meaningful.

**Strip the generic suffix.** ``বচদ্ৰৱা অংশ`` romanizes to *Batadrava Ansh*, and the
gazetteer says *Batadrava*. A first probe scored these as failures purely because of the
suffix, so the value is reduced to its **core** -- the tokens that are not administrative
vocabulary -- before matching, and the suffix is put back afterwards.

**Repair the model's artifacts.** ``xalbari`` cannot match *Salbari* until the ``x`` is dealt
with (see ``normalize``).

A match replaces only the core. ``ৰহা অংশ`` becomes *Raha Ansh*, never *Raha* -- the official
source is an authority on the place name, not a license to drop what the roll printed.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from . import normalize, tokens
from .backends.indicxlit import DIGITS, SCRIPT_RUN

#: Accept without review. Chosen by sweeping against 643 values whose answer was already
#: known by hand, not because it sounds strict::
#:
#:     threshold  margin   applied  agrees with hand review
#:          0.85    0.00       420                    86.9%
#:          0.88    0.05       383                    93.5%
#:          0.90    0.05       379                    94.5%
#:          0.92    0.05       369                    97.0%
#:
#: 0.90 is where the last *wrong place* disappears. At 0.85 the matcher was confidently
#: turning ধমধমা into "Nizdhamdhama" and মধুপুৰ into "Madhapur" -- different places. Every
#: disagreement that survives at 0.90 is the same place spelled differently (Tamulpur /
#: Tambulpur, Lahowal / Lahoal), which is the conflict this whole exercise exists to
#: surface, not an error.
ACCEPT = 0.90

#: How far the best candidate must beat the runner-up. Guards the case where a pincode
#: holds two similar names and the true one is not obviously nearer.
MARGIN = 0.05

#: Below this, the gazetteer has nothing useful to say and the value is left alone. Between
#: the two, a human should look -- the run writes those out instead of applying them.
REVIEW = 0.60

#: Sentinel marking a core run the official name absorbed, so its separator can go too.
DROPPED = "\x00"

#: How much a field's scope is worth as evidence, strongest first. A post office matched
#: against the offices under its own pincode is a near-certain identification; a village
#: matched against those same office names is an inference. When one native string is
#: anchored in two fields at once, this -- not dictionary iteration order -- decides.
SCOPE_STRENGTH = ("post_office", "block", "district", "main_town_village")


def scope_rank(field: str) -> int:
    """Lower is stronger. Unknown scopes sort last."""
    return SCOPE_STRENGTH.index(field) if field in SCOPE_STRENGTH else len(SCOPE_STRENGTH)


#: Administrative vocabulary: everything that is *not* the name of the place.
GENERIC_TOKENS = (
    frozenset(tokens.GENERIC)
    | frozenset(tokens.INSTITUTIONS)
    | frozenset(tokens.ABBREVIATIONS)
    | frozenset(tokens.QUALIFIERS)
)


@dataclass(frozen=True)
class Match:
    """One value's best official candidate, whether or not it was good enough to apply."""

    native: str
    core: str
    official: str
    score: float
    scope: str
    runner_up: float = 0.0

    @property
    def accepted(self) -> bool:
        return self.score >= ACCEPT and (self.score - self.runner_up) >= MARGIN

    @property
    def needs_review(self) -> bool:
        return not self.accepted and self.score >= REVIEW


def split_runs(native: str) -> List[Tuple[str, bool]]:
    """Each native script run, paired with whether it is administrative vocabulary."""
    return [(run, run in GENERIC_TOKENS) for run in SCRIPT_RUN.findall(native.translate(DIGITS))]


def core_runs(native: str) -> List[str]:
    """The runs that name the place, with the administrative vocabulary removed."""
    return [run for run, generic in split_runs(native) if not generic]


def romanize_run(run: str, words: Dict[str, Sequence[str]]) -> str:
    """The best romanization of one run: hand-checked if we have it, else repaired model."""
    if run in tokens.LEXICON:
        return tokens.LEXICON[run]
    candidates = words.get(run)
    if isinstance(candidates, dict) or not candidates:
        return run
    return normalize.repair(candidates[0])


def core_text(native: str, words: Dict[str, Sequence[str]]) -> str:
    """The romanized place-name core -- what actually gets compared to the gazetteer."""
    return " ".join(romanize_run(run, words) for run in core_runs(native))


def best(core: str, candidates: Iterable[str]) -> Tuple[str, float, float]:
    """The closest official name, its score, and the runner-up's, comparing on letters alone."""
    target = normalize.key(core)
    if not target:
        return "", 0.0, 0.0
    ranked = sorted(
        (
            (name, difflib.SequenceMatcher(None, target, normalize.key(name)).ratio())
            for name in candidates
            if normalize.key(name)
        ),
        key=lambda pair: -pair[1],
    )
    if not ranked:
        return "", 0.0, 0.0
    return ranked[0][0], ranked[0][1], (ranked[1][1] if len(ranked) > 1 else 0.0)


def apply(native: str, official: str, words: Dict[str, Sequence[str]]) -> str:
    """Rebuild the value with the official spelling in place of the core.

    Substituted **in position**, over the original string, so every separator and digit
    survives. A first version rebuilt the value by joining the script runs with spaces and
    silently destroyed the part numbers -- ``বড়োবজাৰ (অংশ-২)`` came back as *Boro Bazar Ansh*,
    losing the ``-2`` that distinguishes it from ``(অংশ-১)``. The official source is an
    authority on the place name and on nothing else in the string.

    The core may be several runs (``চিদলা চৰাং``); the official name replaces the group as a
    whole, so the first core run carries it and the rest drop out.
    """
    seen_core = False

    def substitute(match: "re.Match") -> str:
        nonlocal seen_core
        run = match.group()
        if run in GENERIC_TOKENS:
            return romanize_run(run, words)
        if seen_core:
            return DROPPED
        seen_core = True
        return official

    out = SCRIPT_RUN.sub(substitute, native.translate(DIGITS))
    # A dropped run leaves its joining punctuation behind: ``চিদলা-চৰাং`` would end up as
    # ``Sidli-``. Take the separator with it, and the brackets if it was parenthesised --
    # ``সুখচৰ (ধুবুৰা)`` matched an office literally named "Sukchar (Dhubri)" and came out
    # as ``Sukchar (Dhubri) ()``.
    out = re.sub(rf"[\s\-–—]*{DROPPED}", "", out)
    out = re.sub(r"[([{]\s*[)\]}]", "", out)
    return normalize.trim(re.sub(r"\s{2,}", " ", out))


def match(
    native: str,
    candidates: Sequence[str],
    words: Dict[str, Sequence[str]],
    scope: str,
) -> Optional[Match]:
    """Best official candidate for one value, or ``None`` when there is nothing to compare."""
    core = core_text(native, words)
    if not core or not candidates:
        return None
    official, score, runner_up = best(core, candidates)
    if not official:
        return None
    return Match(
        native=native,
        core=core,
        official=official,
        score=score,
        scope=scope,
        runner_up=runner_up,
    )


def calibrate(
    known: Sequence[Tuple[str, str, Sequence[str]]],
    words: Dict[str, Sequence[str]],
) -> Dict[str, object]:
    """Score the thresholds on values whose right answer is already known.

    ``known`` is ``(native, hand_checked_roman, candidates)``. Reports how often an accepted
    match agrees with the hand-checked spelling, so the cutoff is defended by a measurement
    rather than by assertion. Disagreements are the interesting output: they are the
    official-versus-common conflicts the report has to name.
    """
    agree: List[Tuple[str, str, str, float]] = []
    differ: List[Tuple[str, str, str, float]] = []
    missed = 0
    for native, hand, candidates in known:
        found = match(native, candidates, words, scope="calibration")
        if found is None or not found.accepted:
            missed += 1
            continue
        target = normalize.key(found.official)
        (
            agree
            if target == normalize.key(core_text(native, words)) or target == normalize.key(hand)
            else differ
        ).append((native, hand, found.official, found.score))
    total = len(agree) + len(differ)
    return {
        "accepted": total,
        "below_threshold": missed,
        "agree": agree,
        "differ": differ,
        "precision": (len(agree) / total) if total else 0.0,
    }
