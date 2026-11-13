"""Repair of IndicXlit's systematic artifacts. Model output only, never reviewed text.

IndicXlit renders Assamese **phonetics**, and two of its habits are wrong in every English
context this dataset will be joined against:

``x`` for শ / ষ / স
    Assamese pronounces all three as /x/, so the model writes ``xalbari``, ``xapotgram``,
    ``boxistho`` where English has always written *Salbari*, *Sapatgram*, *Basistha*. This is
    not a judgment call: of 1,831 cached tokens whose output contains ``x``, **1,830** have
    শ, ষ or স in the source. The one exception is ``৷৷``, pure OCR noise.

``aa`` for a single আ
    ``aambari``, ``aajij``, ``aaloga``. English doubles that vowel in almost no Assam place
    name.

Both together are worth 13,578 token-occurrences and gain +2.7 points on the hand-checked
gold set. That gain is small because exact-match is a harsh measure of a change whose real
effect is legibility: ``salbari`` is not certainly the conventional spelling, but it is never
worse than ``xalbari`` and a reader can recognize it.

**``z`` is deliberately left alone.** য/জ do surface as *j* conventionally (*Jorhat* for
``zurhat``), but applying ``z``→``j`` scored **-0.3 points**: it corrupts as many tokens as it
fixes. A rule that does not survive its own measurement does not ship.

The other repairs here are OCR debris the transliterator faithfully carried through: the
Bengali danda ``৷`` (U+09F7) sits inside the script-run range and was being handed to a word
transliterator, and zero-width joiners survive into the output invisibly.
"""

from __future__ import annotations

import re

#: U+200C/U+200D. Invisible in the roman column and meaningless there.
ZERO_WIDTH = dict.fromkeys(map(ord, "‌‍"))

#: Punctuation that is only ever noise at the edge of a name. The inner ones are kept --
#: ``Kamrup (Metropolitan)`` and ``PT-II`` need their brackets and hyphen.
EDGE_NOISE = "।|৷॥,-;:_ \t"

#: Indic sentence punctuation, dropped **wherever** it appears. It carries no sound, so it
#: has no romanization, and once ``৷`` stopped being treated as a letter it began surviving
#: into the roman column -- ``Golakganj৷ Paurasabha`` -- which the leaked-script guard
#: correctly refused to write.
DANDA = dict.fromkeys(map(ord, "।৷॥"))

_X = re.compile(r"x")
_AA = re.compile(r"aa")


def repair(roman: str) -> str:
    """Apply every model-artifact repair. Safe to call twice; it is idempotent."""
    if not roman:
        return roman
    out = roman.translate(ZERO_WIDTH)
    out = _X.sub("s", out).replace("X", "S")
    out = _AA.sub("a", out).replace("Aa", "A").replace("AA", "A")
    return trim(out)


def trim(roman: str) -> str:
    """Drop Indic punctuation, collapse whitespace, and strip noise from the edges."""
    out = re.sub(r"\s+", " ", roman.translate(DANDA)).strip()
    return out.strip(EDGE_NOISE).strip()


#: A romanization is comparable to a gazetteer entry only after both are reduced to letters.
#: ``Batadrava Ansh`` and ``Batadrava`` must be able to meet.
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def key(text: str) -> str:
    """Fold to a comparison key: lowercase, letters and digits only."""
    return _NON_ALNUM.sub("", text.lower())
