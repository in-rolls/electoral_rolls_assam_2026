"""AC113 is printed in English, and the reader was built for one script.

Its 161,750 rows shipped with name, relation, age, sex and house number blank on every one --
EPIC alone survived, because that is read from its own cell by a Latin-alphabet engine. These
tests pin the English path and, just as importantly, pin the two Assamese-script paths beside it,
because the labels now live in shared patterns.
"""

import pytest
from PIL import Image

from assam_rolls import ocr
from electors import extract, fields, grid, pages, vision, vision_part

ENGLISH = [
    "Name : PROKAS RAI",
    "Fathers Name : RAJ BAHADUR RAI",
    "House No. 1",
    "Age 45 Gender : Male",
]
ASSAMESE = [
    "নাম : ৰমেন দাস",
    "পিতাৰ নাম : হৰেন দাস",
    "ঘৰ নং : 12",
    "বয়স : 45 লিঙ্গ : পুৰুষ",
]
BENGALI = [
    "নাম : বিটন ভূমিজ",
    "পিতার নাম : লাল বাবু ভুমিজ",
    "বাড়ী নং : 01",
    "বয়স : 34 লিঙ্গ : পুরুষ",
]


class TestLanguageAwareSideReads:
    class Engine:
        def __init__(self, answer="List of Addition 1"):
            self.answer = answer
            self.calls = []

        def _run(self, image, lang, _config, scale, *, psm):
            self.calls.append((lang, scale, psm))
            return self.answer

    def test_english_header_uses_the_english_model(self, monkeypatch):
        engine = self.Engine()
        requested = []
        monkeypatch.setattr(
            ocr,
            "get_engine",
            lambda name, lang: requested.append((name, lang)) or engine,
        )
        box = grid.Box(0, 0, 10, 50, 90, 100, 70)

        text = vision_part._header_strip_text(Image.new("L", (100, 120)), [box], "ENG")

        assert text == "List of Addition 1"
        assert requested == [("tesseract", "eng")]
        assert engine.calls == [("eng", 1, 6)]

    def test_english_summary_uses_the_english_model(self, monkeypatch):
        engine = self.Engine("Mother Roll Basic Roll 361 328 0 689")
        requested = []
        monkeypatch.setattr(
            ocr,
            "get_engine",
            lambda name, lang: requested.append((name, lang)) or engine,
        )

        found = vision_part._summary_from(Image.new("L", (100, 120)), "ENG")

        assert found is not None and found.total == 689
        assert requested == [("tesseract", "eng")]
        assert engine.calls == [("eng", 1, 6)]


def test_an_english_box_yields_every_field():
    e = vision.elector_from(["FXW1756134"], ENGLISH)
    assert e.name == "PROKAS RAI"
    assert e.relation_name == "RAJ BAHADUR RAI"
    assert e.relation_type == "father"
    assert (e.age, e.sex, e.house_no) == (45, "M", "1")


def test_an_english_house_label_tolerates_ocr_separated_punctuation():
    lines = [
        "Name : ARINGLE NRIAME",
        "Fathers Name : GIARIAYING NRIAME",
        "House No .: 220A",
        "Age : 27 Gender : Female",
    ]

    assert vision.elector_from(["ABC1234567"], lines).house_no == "220A"


def test_an_english_repacked_box_reaches_the_field_reader():
    """Stage one's composites contain only body lines, so stage two must not strip them again."""

    words = []
    for row, line in enumerate(ENGLISH):
        left = 0
        for token in line.split():
            right = left + max(8, len(token) * 8)
            words.append(vision.Word(token, left, row * 30, right, row * 30 + 20))
            left = right + 5

    result = extract.PartResult(
        ac_no=113,
        part_no=1,
        lang="ENG",
        source_zip="AC113_ENG.zip",
        source_pdf="2026-EROLLGEN-S03-113-FinalRoll-Revision1-ENG-1-WI.pdf",
        pdf_sha256="a" * 64,
    )
    side = {
        "pages": {
            3: {
                "recognised": True,
                "section": pages.Section.MAIN,
                "boxes": {(3, 0, 0): {"epic": "FXW1756134", "serial": "1", "code": ""}},
            }
        },
        "unknown": [],
        "summary": None,
    }
    meta = {"year": 2026, "roll_type": "FinalRoll", "revision": 1, "lang": "ENG"}

    vision_part._fill_repacked(result, side, {(3, 0, 0): words}, meta)

    row = result.electors[0]
    assert row["name"] == "PROKAS RAI"
    assert row["relation_name"] == "RAJ BAHADUR RAI"
    assert row["relation_type"] == "father"
    assert (row["age"], row["sex"], row["house_no"]) == (45, "M", "1")
    assert row["lang"] == "ENG"
    assert row["engine"] == extract.MIXED_ENGINE


def test_the_all_vision_reader_skips_an_explicit_summary_and_records_its_engine(monkeypatch):
    image = Image.new("L", (400, 500), "white")
    box = grid.Box(0, 0, 10, 100, 390, 500, 300)
    signature = pages.PageSignature(1, pages.PageKind.ELECTOR, (), ())
    monkeypatch.setattr(pages, "classify", lambda _image, _number: signature)
    monkeypatch.setattr(grid, "build", lambda *_args, **_kwargs: [box])

    summary_words = [
        vision.Word("SUMMARY", 10, 10, 80, 30),
        vision.Word("OF", 90, 10, 110, 30),
        vision.Word("ELECTORS", 120, 10, 200, 30),
        vision.Word("নাম", 10, 150, 50, 175),
    ]
    result = extract.PartResult(1, 1, "ASM", "AC1.zip", "part1.pdf", "a" * 64)

    vision_part._fill(result, [image], [summary_words], {})

    assert not result.electors

    ordinary_words = [
        vision.Word("ABC1234567", 10, 110, 100, 130),
        vision.Word("নাম", 10, 150, 50, 175),
        vision.Word("ৰমেন", 60, 150, 120, 175),
    ]
    result = extract.PartResult(1, 1, "ASM", "AC1.zip", "part1.pdf", "a" * 64)
    vision_part._fill(result, [image], [ordinary_words], {})

    assert result.electors[0]["engine"] == extract.VISION_ENGINE


def test_repacked_serials_continue_through_supplements():
    """The printed serial is one cumulative part sequence, independently confirmed by OCR."""
    word = vision.Word("Name", 0, 0, 40, 20)
    result = extract.PartResult(1, 1, "ASM", "AC1.zip", "part1.pdf", "a" * 64)
    side = {
        "pages": {
            3: {
                "recognised": True,
                "section": pages.Section.MAIN,
                "boxes": {(3, 0, 0): {"epic": "A", "serial": "1", "code": ""}},
            },
            4: {
                "recognised": True,
                "section": pages.Section.ADDITION,
                "boxes": {(4, 0, 0): {"epic": "B", "serial": "2", "code": ""}},
            },
            5: {
                "recognised": True,
                "section": pages.Section.MAIN,
                "boxes": {(5, 0, 0): {"epic": "C", "serial": "3", "code": ""}},
            },
        }
    }
    words = {(page, 0, 0): [word] for page in (3, 4, 5)}

    vision_part._fill_repacked(result, side, words, {})

    assert [row["serial_no"] for row in result.electors] == [1, 2, 3]
    assert [row["roll_section"] for row in result.electors] == ["main", "addition", "addition"]


@pytest.mark.parametrize(
    "lines,name,kind",
    [
        (ASSAMESE, "ৰমেন দাস", "father"),
        (BENGALI, "বিটন ভূমিজ", "father"),
    ],
)
def test_the_assamese_script_paths_are_untouched(lines, name, kind):
    """Adding English to the shared patterns must not disturb the editions that already worked."""
    e = vision.elector_from(["ABC1234567"], lines)
    assert e.name == name
    assert e.relation_type == kind
    assert e.age is not None and e.sex == "M" and e.house_no


@pytest.mark.parametrize(
    "printed,expected",
    [
        ("Male", "M"),
        ("Female", "F"),
        ("Other", "T"),
        ("Third", "T"),
    ],
)
def test_female_is_not_read_as_male(printed, expected):
    """``MALE`` is a substring of ``FEMALE``.

    Listed first it matched every woman on the roll as a man -- silently, since both values are
    valid. The Assamese pair has no such overlap, so the hazard arrived with English.
    """
    lines = ["Name : A B", "Fathers Name : C D", "House No. 3", f"Age 30 Gender : {printed}"]
    assert vision.elector_from(["ABC1234567"], lines).sex == expected


@pytest.mark.parametrize(
    "relation,kind",
    [
        ("Fathers", "father"),
        ("Mothers", "mother"),
        ("Husbands", "husband"),
    ],
)
def test_english_relations_are_typed(relation, kind):
    lines = ["Name : A B", f"{relation} Name : C D", "House No. 3", "Age 30 Gender : Male"]
    assert vision.elector_from(["ABC1234567"], lines).relation_type == kind


def test_a_latin_name_survives_the_foreign_stripper():
    """``strip_foreign`` removes Latin as debris, which empties an English name outright.

    A value with no Assamese in it is not an Assamese name carrying debris; it is Latin text.
    """
    assert fields.strip_foreign("PROKAS RAI")[0] == "PROKAS RAI"
    # and the original behavior, where there is something for the Latin to be debris of
    assert fields.strip_foreign("ৰমেন দাস 12")[0] == "ৰমেন দাস"


def test_the_english_closing_page_yields_the_main_list():
    """AC113 carried a readable closing page and none of it was used.

    ``MAIN_ROW`` looked only for ``মূল তালিকা``, so 0 of 261 parts had a printed total and nothing
    about the constituency could be checked. The English edition calls the row ``Mother Roll ...
    Basic Roll``. The net row must not win: it is a different quantity.
    """
    from electors import summary

    page = (
        "SUMMARY OF ELECTORS\n"
        "Male Female Third Total\n"
        "Mother Roll Mother Roll Basic Roll of Special Revision , 361 328 0 689\n"
        "List of Addition Supplement 1 Special Summary Revision 12 3 0 15\n"
        "Net Elector in this Roll after this Revision ( I + II - III + IV ) 354 322 0 676"
    )
    assert summary.parse(page) == (361, 328, 0, 689)


def test_english_supplement_and_main_headers():
    from electors import pages

    supplement = (
        "Assembly Constituency No and Name : 113 - HAFLONG List of Addition 1 ( 27-12-2025 )"
    )
    assert pages.section_of(supplement)[0] is pages.Section.ADDITION

    main = "Assembly Constituency No and Name : 113 - HAFLONG Part number : 103"
    section, recognised = pages.section_of(main)
    assert section is pages.Section.MAIN
    assert recognised, "an English main-roll header must not count as unrecognised"


def test_the_assamese_village_is_still_not_a_section_marker():
    """যোগ sits inside যোগেন্দ্ৰপুৰ. Adding Latin markers must not loosen that boundary."""
    from electors import pages

    assert pages.section_of("অংশৰ নাম : যোগেন্দ্ৰপুৰ")[0] is pages.Section.MAIN
    assert pages.section_of("যোগ তালিকা 1")[0] is pages.Section.ADDITION


class TestBengaliGenitiveHeaders:
    """AC126's real headers, transcribed from a rendered page of its part 1.

    The supplement title uses the genitive -- সংযোজনের তালিকা, "list of additions" -- and the
    whole-word boundary rightly refuses to see সংযোজন inside it, so the inflected form must be a
    marker in its own right. Every one of 5,602 supplement electors was filed into the main roll,
    and every one of 2,328 pages was reported unreadable because the main-page tokens knew only
    the Assamese ৰ and this roll writes the Bengali র.
    """

    def test_the_genitive_supplement_title_is_an_addition(self):
        section, recognised = pages.section_of("সংযোজনের তালিকা 1 (27-12-2025 04-02-2026 )")
        assert section is pages.Section.ADDITION
        assert recognised

    def test_the_bengali_main_header_is_recognised(self):
        section, recognised = pages.section_of("বিধানসভা সমষ্টির নম্বর এবং নাম : 126-রাম কৃষ্ণ নগর")
        assert section is pages.Section.MAIN
        assert recognised

    def test_the_village_name_still_does_not_become_a_section(self):
        """The boundary that caused this stays: যোগ inside a village name is not an addition."""
        section, _ = pages.section_of("অংশৰ নাম : যোগেন্দ্ৰপুৰ")
        assert section is pages.Section.MAIN
