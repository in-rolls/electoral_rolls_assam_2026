"""Which page is which, decided by what is ruled on it rather than by its number.

A part PDF is three kinds of page in sequence: the two info pages already parsed into
``dataset/parts.jsonl.gz``, then the elector pages, then a summary. Page *count* varies with
how many electors a part has, so only the first two positions are fixed and everything else
has to be recognized.

Recognizing it by rule signature rather than by position is not fussiness. The info-page
pipeline was scoped as "ACs 64-73" from a one-page-per-AC sample and turned out to be 28 ACs
and 2,601 pages once every page was measured; layout in this corpus varies more than a
sample suggests. Only one AC of rolls is downloadable today, so any constant read off AC 1
is a guess about the other 125.

The signature is robust because it is structural:

    elector page   a grid can be built from its rules  ->  columns and rows both resolve
    summary page   ruled rows, no columns              ->  no vertical rules
    info page      the form grid                       ->  positions 1 and 2, always

An unrecognized page is **flagged, never parsed**. A page this module cannot classify is a
page whose geometry we do not understand, and guessing at 30 box positions on it would
produce 30 plausible rows of nonsense.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from enum import Enum
from typing import List, Optional, Sequence, Tuple

from PIL import Image

from assam_rolls import layout

from . import grid

#: Pages 1 and 2 are the info pages, already extracted. Everything after is roll content.
INFO_PAGES = 2


class PageKind(str, Enum):
    INFO = "info"
    ELECTOR = "elector"
    SUMMARY = "summary"
    UNKNOWN = "unknown"


class Section(str, Enum):
    """Which list a page of electors belongs to.

    A final roll is the main list plus supplements published with it, and they use the
    **identical box layout**. Part 8's page 31 is headed ``যোগ তালিকা 1 (27-12-2025
    04-02-2026)`` -- an addition list -- and without reading that header its 23 electors are
    indistinguishable from main-roll rows. Parts 1, 5 and 9 have no supplement at all, which
    is what makes this dangerous: it is occasional, so nothing downstream would ever notice
    the two being mixed.
    """

    MAIN = "main"
    ADDITION = "addition"
    DELETION = "deletion"
    MODIFICATION = "modification"


#: Header words that name a supplement.
#:
#: Matched as **whole words**, not substrings. A supplement's header reads ``যোগ তালিকা 1``, and
#: matching ``যোগ`` anywhere filed every page of part 38 as an addition list -- because its
#: polling station is যোগেন্দ্ৰপুৰ, Jogendrapur, and the marker is the first three letters of the
#: village. 534 rows of a 1,048-row main list were labeled a supplement, and the part then read
#: 514 against its own printed total.
#:
#: ``বাদ`` would have been worse in a way nobody would have noticed: countless place names end
#: in -abad. That one never fired in AC1 only by luck of which villages are in it.
#: The English edition writes these out: ``List of Addition 1 ( 27-12-2025 04-02-2026 )``. Matched
#: without the Bengali-letter boundary below, which cannot apply to Latin words -- the boundary is
#: there to stop যোগ matching inside যোগেন্দ্ৰপুৰ, and "Addition" has no such neighbor.
SECTION_MARKERS = (
    ("List of Addition", Section.ADDITION),
    ("List of Deletion", Section.DELETION),
    ("List of Modification", Section.MODIFICATION),
    ("যোগ", Section.ADDITION),
    ("সংযোজন", Section.ADDITION),
    # The Bengali genitive. AC126 titles its supplement সংযোজনের তালিকা -- "list of additions",
    # the possessive -ের on the same word -- and the whole-word boundary below rightly refuses to
    # see সংযোজন inside it. The inflected forms are still whole words, so they are markers of
    # their own rather than a loosened boundary: relaxing the boundary is what once let যোগ match
    # inside a village name. One wording missed here cost 5,602 supplement electors their section,
    # every one filed into the main roll, and the constituency its reconciliation.
    ("সংযোজনের", Section.ADDITION),
    ("বিয়োজনের", Section.DELETION),
    ("সংশোধনের", Section.MODIFICATION),
    ("বিয়োজন", Section.DELETION),
    ("লোপ", Section.DELETION),
    ("বাদ", Section.DELETION),
    ("সংশোধন", Section.MODIFICATION),
    ("পৰিৱৰ্তন", Section.MODIFICATION),
)


#: Where a main-roll page names its polling station. Everything after this is a place name and
#: cannot contain a section marker.
STATION_FIELD = re.compile(r"অংশৰ[^:]{0,24}নাম\s*[:：]")


def section_of(header: str) -> Tuple[Section, bool]:
    """``(section, recognized)`` for one page's header band.

    An unrecognized header is reported as ``MAIN`` with ``recognised=False`` rather than
    guessed at. A supplement worded in a way this does not know about then shows up as a
    flagged page -- something to look at -- instead of as electors quietly merged into the
    main roll, which is the failure that matters.
    """
    text = " ".join(header.split())

    # The station name is not a section marker. A page of the main roll names its polling
    # station after ``অংশৰ ... নাম :``; a supplement page has no station field at all, just the
    # list's own title. So the station name is removed before looking for markers, which is what
    # separates যোগেন্দ্ৰপুৰ the village from যোগ তালিকা the addition list.
    searchable = STATION_FIELD.split(text)[0] if STATION_FIELD.search(text) else text

    for marker, section in SECTION_MARKERS:
        # Whole words. Bengali letters or matras on either side make this part of a longer word:
        # যোগ inside যোগেন্দ্ৰপুৰ, or বাদ at the end of ইসলামাবাদ. The missing left boundary
        # labeled 15,102 main-list rows as deletions in place names ending in -abad.
        if re.search(r"(?<![\u0980-\u09FF])" + marker + r"(?![\u0980-\u09FF])", searchable):
            return section, True
    # Whether the header was understood at all, in either language. A main-roll page that is not
    # recognized is counted as an unrecognized header and reported, so leaving the English forms
    # out would have flagged every one of AC113's pages as unreadable while they parsed fine.
    # Both ra's: Assamese ৰ and Bengali র. AC126 writes সমষ্টির and every one of its 2,328 pages
    # was reported unreadable while parsing fine.
    recognised = any(mark in text for mark in ("অংশৰ", "সমষ্টিৰ", "অংশের", "সমষ্টির")) or bool(
        re.search(r"Assembly\s+Constituency|Part\s+number", text, re.IGNORECASE)
    )
    return Section.MAIN, recognised


def section_after(previous: Section, found: Section) -> Section:
    """Carry an addition onto sparse continuation pages whose title could not be read.

    Main-list pages do not resume after the addition list starts. The last addition page is
    often sparse and its header OCR falls back to the generic main-page form; carrying the
    section forward is independently confirmed by the printed main total. Deletion sheets are
    different in this corpus: short ones can be inserted before the main list resumes.
    """
    if previous is Section.ADDITION and found is Section.MAIN:
        return previous
    return found


@dataclass(frozen=True)
class PageSignature:
    """What the rule detector saw, kept so a misclassification can be diagnosed."""

    number: int
    kind: PageKind
    h_rules: Sequence[int]
    v_rules: Sequence[int]
    #: Set only where the page's own rules could not settle its columns and the part's geometry
    #: did -- the last page of a part, which may draw one column where the sheet holds three.
    columns: Optional[Sequence[Tuple[int, int, int]]] = None

    @property
    def is_elector(self) -> bool:
        return self.kind is PageKind.ELECTOR


def classify(image: Image.Image, number: int) -> PageSignature:
    """Decide what a rendered page is, from its rules alone."""
    h_rules = layout.detect_h_rules(image)
    # Look for columns **only between the first and last horizontal rule**, not down the whole
    # page. A supplement holding eighteen electors rules six rows across the top fifth of the
    # sheet, so its vertical lines are a fifth of the height and none of them survives a
    # full-page threshold: parts 2 and 3 had their addition lists classified as summaries and
    # came out 18 and 8 electors short, matching those lists exactly.
    v_rules = layout.detect_v_rules(image, h_rules[0], h_rules[-1]) if len(h_rules) > 1 else []

    columns = grid.column_triples(v_rules)
    rows = grid.row_bands(h_rules)

    if number <= INFO_PAGES:
        kind = PageKind.INFO
    elif columns and rows:
        # The test is whether a grid can actually be built, not whether enough rules were
        # counted. A rule-count floor is a guess about how many electors a page holds: at
        # eight it called a two-row addition list "unknown" and dropped it. Asking the grid
        # builder is the same question the extractor will ask, so the two cannot disagree.
        kind = PageKind.ELECTOR
    elif columns:
        # Columns but no rows is a grid half-resolved -- geometry we do not understand, and
        # the only case worth flagging. Everything else here is a page with no electors.
        kind = PageKind.UNKNOWN
    elif h_rules:
        # Ruled, but no column structure: the closing summary of net electors. It carries a
        # couple of stray vertical marks, so testing for *no* vertical rules at all put every
        # summary page into "unknown" and drowned the signal that flag exists to give.
        kind = PageKind.SUMMARY
    else:
        kind = PageKind.UNKNOWN

    return PageSignature(number=number, kind=kind, h_rules=tuple(h_rules), v_rules=tuple(v_rules))


def recover_partial(signatures: Sequence[PageSignature]) -> List[PageSignature]:
    """Reconsider the pages that were not elector pages, knowing what this part's columns look like.

    One pass cannot decide a page holding a single elector. Three vertical rules are a box or they
    are a row of a table, and nothing on the page itself says which -- so the first pass calls it
    a summary and the elector is gone. The part's *other* pages settle it: a part draws its
    columns at the same x positions on every page, so a page with rules on those positions is a
    page of that part's boxes.

    This is not only about the main list's last page. Part 5's page 15 is a three-elector addition
    list -- serials 352, 353, 354 -- and it was dropped whole. **Reconciliation would never have
    found that**: supplements are counted separately from the printed ``মূল তালিকা`` total, so the
    main list still balanced at 351 while three real electors were missing.

    Run as a second pass over the signatures already computed, so it costs no rule detection and
    no re-render. Only non-elector pages are reconsidered, so a page already understood cannot be
    changed by this.
    """
    established = typical_columns(signatures)
    if not established:
        return list(signatures)

    out: List[PageSignature] = []
    for signature in signatures:
        if signature.is_elector:
            # A page may resolve to three columns only because an extra vertical rule was
            # mistaken for a photo divider. When their repeated widths disagree, use the
            # geometry established by the rest of this same part. Keeping the bad page-local
            # triples produces plausible rows from photo placeholders, so row counts cannot
            # detect the damage downstream.
            own = grid.column_triples(signature.v_rules)
            if established and not grid.balanced_columns(own):
                signature = replace(signature, columns=tuple(established))
            out.append(signature)
            continue
        if signature.kind is PageKind.INFO or not signature.v_rules:
            out.append(signature)
            continue
        # Two different questions, and conflating them lost electors. Whether *any* of the
        # part's columns are drawn decides whether this is an elector page at all -- a strict
        # test, because a ruled table must not be promoted. How many columns to *build* is not
        # that question: the part draws three, and which of them hold an elector is settled by
        # ink, not by whether the publisher happened to print a particular edge.
        #
        # Part 43's page 25 holds five electors across three columns. Its third column's left
        # edge was not inked -- only the neighboring line of the shared border -- so the strict
        # test called that column absent and serial 663 was dropped from a page whose other two
        # columns read fine.
        if grid.columns_present(signature.v_rules, established) and grid.row_bands(
            signature.h_rules
        ):
            signature = replace(signature, kind=PageKind.ELECTOR, columns=tuple(established))
        out.append(signature)
    return out


def typical_columns(signatures: Sequence[PageSignature]) -> List[Tuple[int, int, int]]:
    """Where this part draws its columns, from the pages that drew all three."""
    return grid.typical_columns(
        [grid.column_triples(s.v_rules) for s in signatures if s.is_elector]
    )


def elector_pages(signatures: Sequence[PageSignature]) -> List[PageSignature]:
    return [s for s in signatures if s.is_elector]


def unknown_pages(signatures: Sequence[PageSignature]) -> List[int]:
    """Page numbers that could not be classified -- the thing to look at, not to parse."""
    return [s.number for s in signatures if s.kind is PageKind.UNKNOWN]
