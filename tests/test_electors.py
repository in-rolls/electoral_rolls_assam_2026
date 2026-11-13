"""The elector stage: geometry, field parsing, and the check that proves completeness.

Every constant here was read off real pages of ``AC1_ASM.zip`` and each test names the
failure it exists to prevent, because all of them were failures first.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import pytest

from electors import (
    against,
    bench,
    crops,
    deliver,
    diagnose,
    escalate,
    evaluate,
    extract,
    fields,
    fleet,
    gemini,
    grid,
    output,
    pages,
    quality,
    reconcile,
    repack,
    replay,
    resolution,
    run,
    second_pass,
    stage1,
    stage2,
    summary,
    timing,
    validate,
    vision,
    vision_part,
)

#: One real elector page from part 1, at 400 dpi. Vertical rules come in clusters because
#: adjacent boxes share a border; the extras at 749, 1189 ... are the internal rule some
#: boxes carry under the EPIC line.
V_RULES = (78, 854, 1082, 1108, 1133, 1909, 2137, 2163, 2189, 2965, 3193, 3219)
H_RULES = (
    38,
    153,
    568,
    593,
    1008,
    1033,
    1451,
    1476,
    1891,
    1916,
    2331,
    2356,
    2771,
    2796,
    3214,
    3236,
    3654,
    3679,
    4094,
    4119,
    4534,
)
H_RULES_WITH_INTERNALS = (
    38,
    153,
    568,
    593,
    749,
    1008,
    1033,
    1189,
    1451,
    1476,
    1629,
    1891,
    1916,
    2331,
    2356,
    2771,
    2796,
    2951,
    3214,
    3236,
    3392,
    3654,
    3679,
    4094,
    4119,
    4273,
    4534,
)


class TestGridGeometry:
    def test_columns_come_from_shared_borders(self):
        """Adjacent boxes share a border, so three columns give seven clusters, not nine rules.

        Reading the rules as consecutive triples finds nothing at all.
        """
        columns = grid.column_triples(V_RULES)
        assert len(columns) == grid.COLUMNS
        for left, divider, right in columns:
            assert left < divider < right
            assert right - left == 1004, "every box is the same width"

    def test_a_page_with_no_dividers_still_gives_three_columns(self):
        """The last page of a part prints no photo panels. Requiring the divider discarded
        those pages whole, and part 5 came out fourteen electors short."""
        bare = [28, 388, 407, 767, 786, 1146]
        assert len(grid.column_triples(bare)) == grid.COLUMNS

    def test_four_rules_that_are_not_columns_are_refused(self):
        """The closing summary page at the scan's native size yields four vertical rules, and
        the divider-less branch built columns of 745, 196 and 197 pixels out of them --
        interpolating dividers that were never printed.

        The part then loses its closing total, which is the number every completeness check is
        measured against, and gains about six rows of nothing. A real page's columns are equal.
        """
        summary_page = [24, 769, 965, 1162]
        assert grid.column_triples(summary_page) == []

    def test_a_slightly_uneven_page_is_still_a_page(self):
        """Loose enough that one missed edge does not disqualify a good page."""
        nearly = [0, 300, 310, 600, 610, 890]
        assert len(grid.column_triples(nearly)) == grid.COLUMNS

    def test_ten_rows_on_a_full_page(self):
        assert len(grid.row_bands(H_RULES)) == grid.ROWS

    def test_internal_rules_do_not_break_the_rows(self):
        """Where most boxes carry the internal rule, its spans outnumber the real row height.

        Choosing the row height by frequency picks 156 and returns six rows for a page
        holding thirty electors; choosing it by the page area it explains picks 415.
        """
        assert len(grid.row_bands(H_RULES_WITH_INTERNALS)) == grid.ROWS

    def test_rows_are_separated_by_a_real_gutter(self):
        """Without this the bottom rule of one row pairs with the bottom of the next."""
        rows = grid.row_bands(H_RULES)
        for (_, bottom), (top, _) in zip(rows, rows[1:]):
            assert top > bottom

    def test_a_full_page_yields_thirty_boxes(self):
        assert len(grid.build(H_RULES, V_RULES)) == grid.COLUMNS * grid.ROWS

    def test_the_photo_column_is_excluded_from_the_text_crop(self):
        """OCR of a whole box drops the name: ফটো উপলব্ধ merges into the text lines."""
        box = grid.build(H_RULES, V_RULES)[0]
        assert box.text_right < box.right


class TestPageClassification:
    def test_an_elector_page_is_recognised_by_its_columns(self):
        signature = pages.PageSignature(3, pages.PageKind.ELECTOR, H_RULES, V_RULES)
        assert signature.is_elector

    def test_unknown_pages_are_reported_not_parsed(self):
        signatures = [
            pages.PageSignature(4, pages.PageKind.ELECTOR, H_RULES, V_RULES),
            pages.PageSignature(5, pages.PageKind.UNKNOWN, (), ()),
        ]
        assert pages.unknown_pages(signatures) == [5]
        assert [s.number for s in pages.elector_pages(signatures)] == [4]


class TestFieldParsing:
    def test_a_trailing_matra_is_part_of_the_name(self):
        """Stripping it truncated ৰাভা to ৰাভ -- almost every surname on the page."""
        assert fields._clean("খাদৰাম ৰাভা |") == "খাদৰাম ৰাভা"

    def test_a_leading_matra_is_scanner_debris(self):
        assert fields._clean("ু ' বাসন্তা ৰাভা") == "বাসন্তা ৰাভা"

    def test_relation_type_comes_from_the_label(self):
        assert fields.relation_of("পিতাৰ নাম : গংগাৰাম ৰাভা") == ("গংগাৰাম ৰাভা", "father")
        assert fields.relation_of("স্বামীৰ নাম : খাদৰাম ৰাভা") == ("খাদৰাম ৰাভা", "husband")

    def test_a_label_missing_its_space_still_matches(self):
        """One character of whitespace cost the relation on 1,588 rows -- 24% of the corpus.

        OCR returns স্বামৰনাম for স্বামীৰ নাম: same letters, no space. A literal substring
        test misses it while the value sits legibly on the same line.
        """
        assert fields.relation_of("স্বামৰনাম' বভজময") == ("বভজময", "husband")
        assert fields.relation_of("স্বামাৰনাম' হুলশা নাজাৰা") == ("হুলশা নাজাৰা", "husband")

    def test_an_unreadable_label_still_yields_the_name(self):
        """The relation name is usually legible even when the word in front of it is not.

        The type is left empty to say the relationship is unknown, rather than dropping a
        value the page plainly holds.
        """
        name, kind = fields.relation_of("গলিত়াৰ মায়: য়ককাই ৰাভা")
        assert name and not kind

    def test_debris_does_not_become_a_relation(self):
        assert fields.relation_of("৷ ক্ৰ ভৰা ক্ম লে *, = |") == ("", "")

    def test_sex_survives_a_damaged_scan(self):
        """পুৰুষ comes back as পৰষ or পৰম; মহিলা as মাহলা."""
        assert fields.age_and_sex("বয়স ' 46 [লঙ্গ ' পৰষ") == (46, "M")
        assert fields.age_and_sex("বয়স : 55 লিঙ্গ : মাহলা") == (55, "F")
        assert fields.sex_of("বয়স ' এ4এ4 ললক্ষ ' পৰম") == "M"
        assert fields.sex_of("বয়স : 30 লিঙ্গ : তৃতীয়") == "T"

    def test_sex_scoring_is_symmetric_between_the_words(self):
        """An ordered fragment list skewed the dataset and this is the regression test.

        মহিলা is longer and more distinctive than পুৰুষ, so it survived damage the male word
        did not, and the female fragments were checked first. Against a published 51/49 male
        split the fragment matcher returned 41/59 -- on rows whose age had read perfectly, so
        it was the matcher and not the scan. Scoring all three words and taking the best is
        symmetric by construction.
        """
        male = fields.sex_of("লিঙ্গ ' পৰম")
        female = fields.sex_of("লিঙ্গ ' মহল")
        assert (male, female) == ("M", "F")

    def test_a_line_with_no_sex_word_yields_nothing(self):
        """Scoring must not invent a sex for the name line."""
        assert fields.sex_of("নাম : খাদৰাম ৰাভা") == ""
        assert fields.sex_of("") == ""

    def test_a_garbled_age_line_is_still_found_by_its_sex_label(self):
        """The whole sex bias lived here, not in the sex matcher.

        Five boxes on one page had no age line identified and all five were male: their line
        read ``বৈবলমলস ' 2/ লঙ্গ ' পৰম`` -- বয়স garbled, digits unreadable -- so age and sex
        were both lost. লিঙ্গ survives where বয়স does not.
        """
        for line in (
            "বৈবলমলস ‘*2/ লঙ্গ ‘ পৰম",
            "বৈবযস * ৫48ক8 [লল্গী ‘' পৰম",
            "বৈবলমস ‘*2/ [ল্গ ‘' পৰম",
        ):
            assert fields._is_age_line(line), line
            assert fields.sex_of(line) == "M", line

    def test_a_name_line_is_not_mistaken_for_the_age_line(self):
        """The widened test must not let a name hijack the anchor."""
        assert not fields._is_age_line("নাম * দিপক নাজাখৰা 0")
        assert not fields._is_age_line("পতাৰ নাম‘ কামৰাজ নাজাৰা")

    def test_the_age_line_is_taken_from_the_end(self):
        """It is always last; searching from the front lets a widened match win too early."""
        assigned = fields.assign_bands(
            ["নাম ' লগন বৰুৱা", "পতাৰ নাম' গংগাৰাম", "ছাৰ ন ' 21", "বয়স ' 46 লঙ্গ ' পৰষ"]
        )
        assert "46" in assigned["age"]

    def test_a_one_character_tail_does_not_raise(self):
        """``range(2, 2)`` is empty and ``max()`` raised, killing the whole part.

        Found by a real run dying on it. Every short input must return, not raise.
        """
        for text in ("", "প", "ম", "1", "  ", "-"):
            assert fields.sex_of(text) == ""

    def test_an_implausible_age_is_not_recorded(self):
        assert fields.age_and_sex("বয়স : 7 লিঙ্গ : পুৰুষ")[0] is None

    def test_a_doubled_digit_still_yields_an_age(self):
        """The Assamese model doubles digits it is unsure of: 55 comes back as 525."""
        assert fields.age_from("বয়স ' 525 !লঙ্গ ' মাহলা") == 52
        assert fields.age_from("বয়স : 4 6") == 46

    def test_an_unreadable_box_is_still_a_row(self):
        """Ink decides emptiness, not whether any field parsed.

        Conflating "the publisher left this blank" with "OCR failed here" drops the second
        silently, and every part then comes up short of its published total.
        """
        assert fields.Elector(flags=["unreadable"]).is_empty is True

    def test_a_box_with_only_a_relation_is_not_treated_as_blank(self):
        """Testing only EPIC, name and age left the first run nine electors short."""
        assert not fields.Elector(relation_name="গংগাৰাম ৰাভা").is_empty
        assert fields.Elector().is_empty

    def test_bands_are_anchored_on_the_age_line(self):
        """The house line scans worst of all, so it cannot be found by its own label."""
        assigned = fields.assign_bands(
            [
                "নাম ' খাদৰাম ৰাভা",
                "পতাৰ নাম' গংগাৰাম ৰাভা",
                "ছাৰ ন ' 21",
                "বয়স ' 46 [লঙ্গ ' পৰষ",
            ]
        )
        assert "46" in assigned["age"]
        assert assigned["house"] == "ছাৰ ন ' 21"
        assert "গংগাৰাম" in assigned["relation"]

    def test_house_number_falls_back_to_its_digits(self):
        assert fields.house_number("ছাৰ ন ' 21") == "21"

    def test_scales_that_agree_are_trusted_and_disagreement_is_flagged(self):
        assert fields.consensus(["ৰাভা", "ৰাভা"]) == ("ৰাভা", True)
        value, agreed = fields.consensus(["ৰাভ", "ৰাভা"])
        assert value == "ৰাভা" and not agreed


class TestRollSections:
    """A final roll is the main list plus supplements, in the identical box layout."""

    def test_a_supplement_header_is_recognised(self):
        """Part 8 page 31: যোগ তালিকা 1 (27-12-2025 04-02-2026) -- an addition list."""
        header = "বিধানসভা সমষ্টিৰ নম্বৰ আৰু নাম : 1-গোসাইগাওঁ খণ্ড নং :8 | যোগ তালিকা 1"
        assert pages.section_of(header) == (pages.Section.ADDITION, True)

    def test_a_main_roll_header_is_recognised(self):
        header = "বিধানসভা সমষ্টিৰ নম্বৰ আৰু নাম : 1-গোসাইগাওঁ | অংশৰ নম্বৰ আৰু নাম : 1-বনগাওঁ"
        assert pages.section_of(header) == (pages.Section.MAIN, True)

    def test_an_unreadable_header_is_flagged_not_guessed(self):
        """An unknown supplement wording must surface, not merge silently into the roll."""
        section, recognised = pages.section_of("garbled nonsense")
        assert section is pages.Section.MAIN and not recognised


class TestPageKindsAreSpecific:
    """``unknown`` has to mean something, or it stops being a signal."""

    SUMMARY_H = (38, 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000, 2200, 2400)

    def test_a_summary_with_stray_vertical_marks_is_not_unknown(self):
        """Testing for *no* vertical rules put every summary page into unknown.

        The closing page carries a couple of stray vertical marks, so the seven-part run came
        back with unknown=[25], [30, 32] and so on -- all of them ordinary summaries, drowning
        the flag that exists to surface geometry we do not understand.
        """
        signature = pages.PageSignature(25, pages.PageKind.SUMMARY, self.SUMMARY_H, (100, 900))
        assert signature.kind is pages.PageKind.SUMMARY

    def test_a_two_row_supplement_is_an_elector_page(self):
        """A rule-count floor is a guess about how many electors a page holds.

        At eight horizontal rules it classified an eighteen-elector addition list as a
        summary, and parts 2 and 3 came out 18 and 8 short -- exactly those lists.
        """
        h = (38, 153, 568, 593, 1008)
        v = (78, 854, 1082, 1108, 1133, 1909, 2137, 2189, 2965, 3193)
        assert grid.column_triples(v) and grid.row_bands(h)


class TestResume:
    """A part is durable the moment it is read.

    The first version wrote nothing until all 154 parts finished, so stopping a four-hour run
    discarded four hours. Worse, the commit message claimed the stage was "cached and
    resumable" before any of it existed.
    """

    def payload(self, tmp_path, zip_path="z.zip", pdf="p.pdf", engine="tesseract", key=""):
        """The worker's argument tuple, built the way cmd_parse builds it.

        Spelled out rather than defaulted: the tuple gained an engine and a key, and a worker
        that quietly filled in "tesseract" for a short tuple would have run the wrong engine
        rather than failing.
        """
        return (str(zip_path), pdf, str(tmp_path), False, engine, key)

    def test_a_read_part_is_cached_and_not_read_twice(self, tmp_path, monkeypatch):
        from assam_rolls import cache
        from electors import cli, extract

        calls = []

        def fake_read_part(zip_path, pdf_name, engine=None, capture_lines=False):
            calls.append((pdf_name, capture_lines))
            result = extract.PartResult(1, 7, "ASM", "z.zip", pdf_name, "sha")
            result.electors = [{"ac_no": 1, "part_no": 7, "name": "x"}]
            return result

        monkeypatch.setattr(extract, "read_part", fake_read_part)
        monkeypatch.setattr(cli.render, "read_pdf_bytes", lambda *a: b"bytes")
        monkeypatch.setattr(cli.render, "sha256_bytes", lambda *a: "sha")
        monkeypatch.setattr(cli.ocr, "get_engine", lambda *a, **k: None)

        first = cli._one_part(self.payload(tmp_path, tmp_path / "z.zip", "part7.pdf"))
        second = cli._one_part(self.payload(tmp_path, tmp_path / "z.zip", "part7.pdf"))

        assert calls == [("part7.pdf", False)], "the second call must come from cache"
        assert not first["cached"] and second["cached"]
        assert second["electors"] == first["electors"]
        assert cache.read_entry(tmp_path, "part7") is not None

    def test_the_engines_do_not_share_a_cache_directory(self):
        """Keyed on the AC alone, a Vision run is served rows tesseract produced -- same part
        number, same source bytes, same stage version, completely different text.

        The engine is in the directory name rather than checked inside the entry so that both
        engines' results can exist at once; a run of one does not discard the other's.
        """
        from electors import cli

        parser = cli.build_parser()
        base = parser.parse_args(["parse", "z.zip"])
        vision_run = parser.parse_args(["parse", "z.zip", "--engine", "vision"])
        assert base.engine == "tesseract"
        assert vision_run.engine == "vision"
        assert base.cache == vision_run.cache, "the difference is applied to the AC directory"

    def test_vision_refuses_to_start_without_a_key(self, tmp_path, monkeypatch, capsys):
        """Better than 154 parts each failing on their own request."""
        from electors import cli

        monkeypatch.delenv("VISION_API_KEY", raising=False)
        monkeypatch.setattr(cli.render, "require_poppler", lambda: None)
        monkeypatch.setattr(
            cli.render, "iter_zip_parts", lambda p: [type("P", (), {"ac_no": 1, "pdf_name": "a"})()]
        )
        args = cli.build_parser().parse_args(
            ["parse", str(tmp_path / "z.zip"), "--engine", "vision"]
        )
        (tmp_path / "z.zip").write_bytes(b"")
        assert cli.cmd_parse(args) == 1
        assert "key" in capsys.readouterr().err

    def test_reissued_source_bytes_invalidate_the_entry(self, tmp_path, monkeypatch):
        """Resumption must never serve data derived from a file that no longer exists."""
        from assam_rolls import cache

        cache.write_entry(tmp_path, "part7", {"part_no": 7}, [], "old-sha")
        entry = cache.read_entry(tmp_path, "part7")
        assert cache.is_fresh(entry, "old-sha")
        assert not cache.is_fresh(entry, "new-sha")


class TestReconciliation:
    """The check that makes this stage trustworthy: the source published the answer."""

    def rows(self, n, ac=1, part=1, sex="M"):
        return [
            {"ac_no": ac, "part_no": part, "serial_no": i + 1, "sex": sex, "name": "x"}
            for i in range(n)
        ]

    def test_a_part_that_matches_its_published_total(self):
        checks = validate.reconcile(self.rows(850), {(1, 1): {"total": 850, "male": 850}})
        assert checks[0].counts_match and checks[0].shortfall == 0

    def test_a_short_part_is_caught(self):
        """The failure this whole design exists to detect: a page that did not resolve."""
        checks = validate.reconcile(self.rows(820), {(1, 1): {"total": 850}})
        assert not checks[0].counts_match
        assert checks[0].shortfall == -30

    def test_the_serial_check_compares_ocr_against_position(self):
        """The old check could not fail: serial_no is assigned by a counter, so it is 1..N
        by construction. It reported "0 parts with serial gaps" and I quoted that as
        evidence. What can disagree is the serial read off the page.
        """
        rows = self.rows(10)
        for r in rows:
            r["serial_no_ocr"] = r["serial_no"]
        assert validate.reconcile(rows, {})[0].serial_gaps == 0
        rows[5]["serial_no_ocr"] = 99
        assert validate.reconcile(rows, {})[0].serial_gaps == 1

    def test_the_roll_total_is_the_target_not_the_info_page_net(self):
        """Part 1 prints 873 and its closing page says 873; the info page says 850.

        Measuring against the net made a correct extraction look 23 over.
        """
        rows = self.rows(873)
        checks = validate.reconcile(
            rows, {(1, 1): {"total": 850}}, {(1, 1): {"total": 873, "male": 453, "female": 420}}
        )
        assert checks[0].matches_roll and checks[0].roll_shortfall == 0
        summary = validate.summarize(checks, rows)
        assert summary["parts_matching_roll"] == 1
        assert summary["parts_unmeasured"] == []

    def test_a_part_whose_roll_total_is_unreadable_is_excluded_not_guessed(self):
        rows = self.rows(600)
        checks = validate.reconcile(rows, {(1, 1): {"total": 618}})
        assert checks[0].matches_roll is None
        summary = validate.summarize(checks, rows)
        assert summary["parts_measured"] == 0
        assert summary["parts_unmeasured"] == [1]

    def test_the_sex_ratio_is_reported_against_the_published_one(self):
        """A ratio catches what no count can: a matcher that skews the dataset.

        The fragment matcher returned 41% male against a published 51% while the totals
        looked fine, so this is the check that would have caught it.
        """
        rows = self.rows(6, sex="M") + self.rows(4, sex="F")[:4]
        for i, r in enumerate(rows):
            r["serial_no"] = i + 1
        checks = validate.reconcile(rows, {(1, 1): {"total": 10, "male": 5, "female": 5}})
        summary = validate.summarize(checks, rows)
        assert summary["male_share"] == 0.6
        assert summary["published_male_share"] == 0.5

    def test_the_published_gap_is_reported_not_scored(self):
        """The printed roll does not equal the published net, so this is a difference.

        Part 1 prints 870 against a published 850 with no supplement and no blank boxes.
        Treating that as failure would mean deleting real rows.
        """
        rows = self.rows(870)
        summary = validate.summarize(validate.reconcile(rows, {(1, 1): {"total": 850}}), rows)
        assert summary["printed_minus_published_mean"] == 20.0
        assert summary["gap_outliers"] == [], "a single part cannot be an outlier"

    def test_serial_gaps_are_counted_within_each_list(self):
        """Supplements restart at 1, so a part-wide sequence check sees a false gap."""
        rows = self.rows(3)
        extra = self.rows(2, part=1)
        for i, r in enumerate(extra):
            r["serial_no"] = i + 1
            r["roll_section"] = "addition"
        checks = validate.reconcile(rows + extra, {(1, 1): {"total": 5}})
        assert checks[0].serial_gaps == 0
        assert checks[0].supplement_rows == 2


class TestQualityIsNotFillRate:
    """Fill rate says a field is non-empty. These say whether it can possibly be right.

    Every string here came out of a real run; none is invented.
    """

    def test_latin_in_an_assamese_name_is_certainly_wrong(self):
        rows = [{"name": "Kokrajhar midancipalus"}, {"name": "খাদৰাম ৰাভা"}]
        assert quality.definite_errors(rows)["by_kind"]["name_has_latin_or_digits"] == 1

    def test_a_name_equal_to_the_relation_is_certainly_wrong(self):
        """Seen as তন্ডা ৰাভা in both columns -- one band read twice."""
        rows = [{"name": "তন্ডা ৰাভা", "relation_name": "তন্ডা ৰাভা"}]
        assert "name_equals_relation" in quality.definite_errors(rows)["by_kind"]

    def test_a_leaked_label_is_certainly_wrong(self):
        rows = [{"name": "নাম খাদৰাম"}]
        assert "name_contains_label" in quality.definite_errors(rows)["by_kind"]

    def test_a_duplicate_epic_is_certainly_wrong(self):
        """EPICs are unique by construction, so a repeat inside one AC is a misread."""
        rows = [{"epic_no": "HHK0001471"}, {"epic_no": "HHK0001471"}, {"epic_no": "HHK0001472"}]
        assert quality.definite_errors(rows)["duplicate_epics"] == 1

    def test_a_clean_row_trips_nothing(self):
        rows = [
            {
                "name": "খাদৰাম ৰাভা",
                "relation_name": "গংগাৰাম ৰাভা",
                "epic_no": "HHK0001471",
                "age": 46,
            }
        ]
        assert quality.definite_errors(rows)["rows_definitely_wrong"] == 0

    def test_the_floor_and_ceiling_are_reported_separately(self):
        """They bound the error rate from opposite sides and must not be conflated."""
        rows = [
            {"name": "Latin here", "flags": ""},
            {"name": "খাদৰাম ৰাভা", "flags": "name_disagreement"},
        ]
        assert quality.definite_errors(rows)["rate"] == 0.5
        assert quality.disagreement(rows)["name_disagreement"] == 0.5

    def test_the_age_last_digit_histogram_exposes_digit_confusion(self):
        """No per-row check can see this: every age is individually plausible."""
        rows = [{"age": a} for a in (25, 35, 45, 55, 65, 25, 35, 45)]
        dist = quality.distributions(rows)
        assert dist["age_last_digit"] == {5: 8}


class TestTheGate:
    """A fix is accepted only if it clears all four conditions.

    The sex bias was declared fixed twice before its cause was found, because each attempt
    improved the case in front of me. Improving the case in front of you is not evidence.
    """

    def test_a_fix_that_only_works_in_sample_is_rejected(self):
        results = bench.run(
            "epic_present",
            in_sample={"before": 0.83, "after": 0.98},
            out_of_sample={"before": 0.79, "after": 0.80},
            regression_before={},
            regression_after={},
            guarded=(),
        )
        assert not bench.verdict(results)

    def test_a_fix_that_breaks_something_else_is_rejected(self):
        """The gate that was missing: collateral damage to a field nobody was watching."""
        results = bench.run(
            "epic_present",
            in_sample={"before": 0.83, "after": 0.98},
            out_of_sample={"before": 0.79, "after": 0.97},
            regression_before={"parts_matching_roll_rate": 0.60, "name_present": 0.90},
            regression_after={"parts_matching_roll_rate": 0.40, "name_present": 0.90},
            guarded=("parts_matching_roll_rate", "name_present"),
        )
        assert not bench.verdict(results)
        assert any("parts_matching_roll_rate" in r.detail for r in results)

    def test_an_unmeasured_guard_is_a_failure_not_a_pass(self):
        """Skipping a metric absent from either side is how the checks that matter clear
        the gate by being missing. The two with real ground truth are exactly the ones a
        harness is most likely to fail to compute."""
        result = bench.gate_no_damage(
            {"name_present": 0.90},
            {"name_present": 0.90},
            guarded=("name_present", "parts_matching_roll_rate", "male_share"),
        )
        assert not result.passed
        assert "not measured" in result.detail

    def test_guarding_soundness_does_not_punish_removing_wrong_data(self):
        """A fill rate falls when a provably wrong value is correctly cleared.

        Guarding fill would reject the one move that unambiguously improves the data, so the
        gate guards *present and not provably wrong* instead.
        """
        wrong = [{"name": "তন্ডা ৰাভা", "relation_name": "তন্ডা ৰাভা"}]
        cleared = [{"name": "", "relation_name": "তন্ডা ৰাভা"}]
        assert quality.fill_rates(cleared)["name"] < quality.fill_rates(wrong)["name"]
        assert (
            quality.sound_rates(cleared)["name_sound"] == quality.sound_rates(wrong)["name_sound"]
        )

    def test_a_fix_clearing_everything_is_accepted(self):
        results = bench.run(
            "epic_present",
            in_sample={"before": 0.826, "after": 0.981},
            out_of_sample={"before": 0.792, "after": 0.977},
            regression_before={"name_present": 0.90},
            regression_after={"name_present": 0.90},
            guarded=("name_present",),
            seconds=(800.0, 780.0),
        )
        assert bench.verdict(results)

    def test_the_splits_are_disjoint(self):
        """An out-of-sample set that overlaps the others is not out of sample."""
        parts = list(range(1, 155))
        splits = bench.split_parts(parts)
        assert not set(splits["validate"]) & set(splits["diagnose"])
        assert not set(splits["validate"]) & set(splits["regression"])

    def test_the_held_out_split_is_reproducible(self):
        parts = list(range(1, 155))
        assert bench.validate_parts(parts) == bench.validate_parts(parts)

    def test_a_ratio_is_scored_by_closeness_not_by_size(self):
        """Overshooting the target is not an improvement."""
        under = bench.metrics_from({}, {"male_share": 0.45, "roll_male_share": 0.508})
        near = bench.metrics_from({}, {"male_share": 0.505, "roll_male_share": 0.508})
        over = bench.metrics_from({}, {"male_share": 0.60, "roll_male_share": 0.508})
        assert near["male_share"] > under["male_share"]
        assert near["male_share"] > over["male_share"]


class TestTheProposalEngine:
    """Root causes are co-occurrences. Found by eye they neither repeat nor scale."""

    def rows(self, failing_column):
        out = []
        for col in (0, 1, 2):
            for i in range(60):
                lost = col == failing_column and i < 30
                out.append(
                    {
                        "box_col": col,
                        "box_row": i % 10,
                        "roll_section": "main",
                        "sex": "" if lost else "M",
                        "age": None if lost else 40,
                        "epic_no": "HHK0001471",
                        "name": "ৰাভা",
                        "relation_name": "গংগাৰাম",
                        "house_no": "2",
                        "flags": "",
                    }
                )
        return out

    def test_it_finds_a_failure_concentrated_in_one_column(self):
        found = diagnose.associations(self.rows(2), min_support=10)
        assert found and found[0].feature == "box_col" and found[0].value == 2
        assert found[0].lift > 2

    def test_it_proposes_the_crop_when_the_signature_is_spatial(self):
        found = diagnose.associations(self.rows(2), min_support=10)
        assert any("crop" in p for p in diagnose.proposals(found))

    def test_evenly_spread_failures_produce_nothing(self):
        """It must not invent a cause when there is no pattern.

        The first version of this test used ``i % 5`` for failures and ``i % 10`` for the row,
        which puts every failure in rows 0 and 5 -- a real association. The engine found it and
        the test was wrong. Randomized so the null really is null.
        """
        rng = random.Random(3)
        rows = [
            {
                "box_col": rng.randrange(3),
                "box_row": rng.randrange(10),
                "sex": "" if rng.random() < 0.2 else "M",
                "flags": "",
            }
            for _ in range(3000)
        ]
        assert diagnose.associations(rows, min_support=100) == []

    def test_empty_and_garbled_reads_are_separated(self):
        """They want opposite fixes and are identical in a fill rate."""
        shapes = dict(diagnose.raw_read_shapes(["", "", "", "S 106 -, HHK3535/704", "", "HH"]))
        assert shapes["empty"] == 4 and shapes["garbled"] == 1


class TestReplay:
    """The line cache is only worth having if it reproduces the pipeline exactly."""

    def _box(self):
        return replay.BoxLines(
            page=7,
            section="main",
            col=1,
            row=2,
            lines=[
                "নাম : খাদৰাম ৰাভা",
                "পিতাৰ নাম : গংগাৰাম ৰাভা",
                "ঘৰ নং : 42",
                "বয়স : 35 লিঙ্গ : পুৰুষ",
            ],
            name_second=["নাম : খাদৰাম ৰাভা"],
            epic_raw="S 106 -, HHK3535704",
            serial_raw="106",
        )

    def test_replay_reproduces_the_pipeline_assembler(self):
        """Replay must call the pipeline's own assembler, not a copy of it.

        A harness that reimplements the parser measures the harness. This asserts the two
        paths produce identical values from identical text.
        """
        box = self._box()
        direct = fields.assemble((box.lines, box.name_second), box.epic_raw, box.serial_raw)
        replayed = replay.parse_box(box, serial=1)
        assert replayed["name"] == direct.name
        assert replayed["relation_name"] == direct.relation_name
        assert replayed["age"] == direct.age
        assert replayed["sex"] == direct.sex
        assert replayed["epic_no"] == direct.epic_no

    def test_a_capture_from_older_code_is_refused_not_replayed(self, tmp_path):
        """Replaying text the current pipeline would not produce answers the wrong question."""
        stale = replay.PartLines(
            part=13,
            ac=1,
            capture_version=replay.CAPTURE_VERSION - 1,
            scale=2,
            psm=7,
            lang="asm",
            boxes=[self._box()],
        )
        replay.save(stale, root=tmp_path)
        assert replay.load(13, 1, tmp_path) is None
        assert replay.missing([13], 1, tmp_path) == [13]

    def test_a_current_capture_round_trips(self, tmp_path):
        fresh = replay.PartLines(
            part=13,
            ac=1,
            capture_version=replay.CAPTURE_VERSION,
            scale=2,
            psm=7,
            lang="asm",
            boxes=[self._box()],
        )
        replay.save(fresh, root=tmp_path)
        rows = replay.replay_parts([13], 1, tmp_path)
        assert len(rows) == 1
        assert rows[0]["name"] and rows[0]["box_col"] == 1
        assert replay.cached_parts(tmp_path) == [13]

    def test_replayed_rows_keep_the_capture_language(self):
        captured = replay.PartLines(
            part=13,
            ac=113,
            capture_version=replay.CAPTURE_VERSION,
            scale=2,
            psm=7,
            lang="eng",
            boxes=[self._box()],
        )

        assert replay.replay(captured)[0]["lang"] == "ENG"


class TestEscalation:
    """The router decides what the expensive pass reads, so it is measured before it is used."""

    CLEAN = {
        "epic_no": "HHK0001471",
        "name": "খাদৰাম ৰাভা",
        "relation_name": "গংগাৰাম ৰাভা",
        "relation_type": "father",
        "age": 35,
        "sex": "M",
        "flags": "",
    }

    def test_a_clean_row_is_not_escalated(self):
        assert not escalate.needs_escalation(dict(self.CLEAN))

    def test_a_provably_wrong_row_is_escalated(self):
        row = dict(self.CLEAN, name="গংগাৰাম ৰাভা")
        assert "name_equals_relation" in escalate.certain(row)
        assert escalate.needs_escalation(row)

    def test_disagreement_alone_is_enough_to_doubt_a_row(self):
        """Nothing here is known to be wrong -- the two upscales just did not agree."""
        row = dict(self.CLEAN, flags="name_disagreement")
        assert escalate.certain(row) == []
        assert escalate.doubtful(row) == ["name_disagreement"]
        assert escalate.needs_escalation(row)

    def test_one_missing_field_is_not_enough(self):
        """A blank box is the publisher's doing; two missing core fields is a broken read."""
        assert escalate.missing_core(dict(self.CLEAN, age=None)) == []
        assert escalate.missing_core(dict(self.CLEAN, age=None, sex="")) != []

    def test_volume_is_reported_because_it_bounds_the_second_pass_cost(self):
        rows = [dict(self.CLEAN) for _ in range(9)] + [dict(self.CLEAN, name="গংগাৰাম ৰাভা")]
        found = escalate.report(rows)
        assert found.flagged == 1 and found.volume == 0.1

    def test_a_router_that_flags_at_random_scores_no_better_than_chance(self):
        """The check that makes the router falsifiable.

        Scoring precision against the floor detectors could only ever return 100%, because
        they are half the router. This compares agreement on flagged rows against agreement
        on unflagged ones, which a useless router cannot win.
        """
        cheap = [dict(self.CLEAN, flags="name_disagreement" if i % 2 else "") for i in range(20)]
        # The second engine disagrees on every other row, uncorrelated with the flag.
        expensive = [dict(r, name="অন্য নাম" if i % 3 else r["name"]) for i, r in enumerate(cheap)]
        scored = escalate.agreement_against(cheap, expensive)
        assert 0.5 < scored["flagged_agreement_ratio"]["name"] < 1.6

    def test_a_router_that_finds_real_trouble_scores_below_one(self):
        cheap = [dict(self.CLEAN, flags="name_disagreement" if i < 10 else "") for i in range(20)]
        # The second engine disagrees only where the router flagged.
        expensive = [dict(r, name="অন্য নাম" if i < 10 else r["name"]) for i, r in enumerate(cheap)]
        scored = escalate.agreement_against(cheap, expensive)
        assert scored["flagged_agreement_ratio"]["name"] == 0.0


class TestNameConsensus:
    """The second scale must be able to disagree, or it is a check that cannot fail."""

    LINES = [
        "নাম : খাদৰাম ৰাভা",
        "পিতাৰ নাম : গংগাৰাম ৰাভা",
        "ঘৰ নং : 5",
        "বয়স : 35 লিঙ্গ : পুৰুষ",
    ]

    def test_the_second_scale_reading_the_same_band_differently_is_flagged(self):
        """Under the previous version this flag was raised zero times in 10,245 rows.

        The one-line second read was passed through assign_bands, where a single line lands
        in ``house`` -- so the second reading of the *name* was always empty, consensus always
        saw one value, and a full scale-3 pass over every name crop was computed and discarded.
        """
        elector = fields.assemble((self.LINES, ["নাম : লঙ্ষু্ন কঙস্ত্ু"]), "HHK0001471", "106")
        assert "name_disagreement" in elector.flags

    def test_a_one_character_difference_is_noise_and_not_flagged(self):
        """53% of boxes differ between the scales; 83% of those by a character or two.

        Flagging all of them took the escalation router to 56% of rows, which is a re-run
        rather than a triage.
        """
        elector = fields.assemble((self.LINES, ["নাম : খাদৰম ৰাভা"]), "HHK0001471", "106")
        assert elector.flags == []

    def test_agreement_is_not_flagged(self):
        elector = fields.assemble((self.LINES, ["নাম : খাদৰাম ৰাভা"]), "HHK0001471", "106")
        assert elector.flags == []

    def test_no_second_opinion_when_the_name_did_not_come_from_the_top_band(self):
        """Comparing anyway would report the name line disagreeing with the relation line."""
        shifted = self.LINES[1:]
        assigned = fields.assign_bands(shifted)
        assert fields.second_name(shifted, ["নাম : খাদৰাম ৰাভা"], assigned) == ""


class TestDiagnosingWrongValues:
    """An engine that localizes only what is absent will never point at what is false."""

    def test_a_wrong_value_is_a_failure_class_the_engine_can_localise(self):
        rows = [
            {
                "name": "গংগাৰাম ৰাভা" if i % 3 == 0 else "খাদৰাম ৰাভা",
                "relation_name": "গংগাৰাম ৰাভা",
                "box_col": i % 3,
                "box_row": i % 10,
            }
            for i in range(90)
        ]
        ranked = {p.failure: p for p in diagnose.priorities(rows)}
        assert "name_equals_relation" in ranked, "the largest error class must be rankable"
        assert ranked["name_equals_relation"].affected == 30

    def test_it_finds_the_slice_a_wrong_value_concentrates_in(self):
        rows = [
            {
                "name": "গংগাৰাম ৰাভা" if i % 3 == 0 else "খাদৰাম ৰাভা",
                "relation_name": "গংগাৰাম ৰাভা",
                "box_col": i % 3,
                "box_row": 0,
            }
            for i in range(90)
        ]
        found = diagnose.associations(rows, features=("box_col",))
        hit = next(a for a in found if a.failure == "name_equals_relation")
        assert hit.value == 0 and hit.rate == 1.0


class TestDerivedFeatures:
    """`diffuse` is a claim about the feature set, not about the data."""

    def _part(self, pages, per_page=3, part=5):
        rows = []
        for page in pages:
            for box_row in range(per_page):
                rows.append({"ac_no": 1, "part_no": part, "page_no": page, "box_row": box_row})
        return rows

    def test_first_and_last_pages_are_distinguished_from_the_middle(self):
        found = diagnose.derive_features(self._part([3, 4, 5]))
        positions = {(r["page_no"], r["page_position"]) for r in found}
        assert positions == {(3, "first"), (4, "middle"), (5, "last")}

    def test_the_bottom_row_of_each_page_is_marked(self):
        found = diagnose.derive_features(self._part([3], per_page=4))
        assert [r["in_last_row"] for r in found] == [False, False, False, True]

    def test_a_cause_invisible_to_the_base_features_is_found_by_the_derived_ones(self):
        """Without page_position this failure looks evenly spread and routes to escalation.

        Routing it there quietly buys a second OCR pass for something a geometry fix solves,
        which is why the thinness of the feature set is a correctness concern and not a
        matter of taste.
        """
        # Four parts, so the first-page slice clears MIN_SUPPORT. A slice below it is not
        # reported at all, which is the intended behavior: a rate over five rows is noise.
        rows = [r for part in range(1, 5) for r in self._part([1, 2, 3, 4, 5], 10, part)]
        enriched = diagnose.derive_features(rows)
        for row in enriched:
            row["name"] = "" if row["page_position"] == "first" else "খাদৰাম"
        blind = diagnose.associations(enriched, features=("box_col", "box_row"))
        assert not [a for a in blind if a.failure == "no_name"]
        seeing = diagnose.associations(enriched, features=("page_position",))
        hit = next(a for a in seeing if a.failure == "no_name")
        assert hit.value == "first" and hit.rate == 1.0


class TestSwapDetection:
    """A swapped name and relation is populated, plausible, and invisible to the floor."""

    def test_a_relation_without_a_recognised_label_is_doubted(self):
        row = {"name": "হুলশা নাজাৰা", "relation_name": "খাদৰাম ৰাভা", "relation_type": ""}
        assert escalate.certain(row) == [], "nothing here is provably wrong -- that is the point"
        assert escalate.unlabelled_relation(row) == ["relation_type_unknown"]
        assert escalate.needs_escalation(row)

    def test_a_labelled_relation_is_not_doubted(self):
        row = {"name": "খাদৰাম ৰাভা", "relation_name": "গংগাৰাম", "relation_type": "father"}
        assert escalate.unlabelled_relation(row) == []

    def test_an_empty_relation_is_not_a_swap_suspect(self):
        assert escalate.unlabelled_relation({"relation_name": "", "relation_type": ""}) == []


class TestCleaningValues:
    """A trailing matra is usually the word; stripping it truncates almost every surname."""

    def test_real_trailing_matras_survive(self):
        for name in ("খাদৰাম ৰাভা", "শৰ্মা", "বৰুৱা", "গংগাৰাম ৰাভা", "চন্দা মাড"):
            assert fields._clean(name) == name

    def test_a_matra_stranded_after_punctuation_is_debris(self):
        """No Assamese syllable puts a vowel sign after a full stop.

        Left in, it survived every strip, and consensus -- which keeps the longer of two
        readings -- then preferred the dirtier reading over the clean one beside it.
        """
        assert fields._clean("ৰেহমাত বসমাতাৰা .ে") == "ৰেহমাত বসমাতাৰা"

    def test_the_punctuation_class_is_escaped(self):
        """EDGE_PUNCTUATION contains a hyphen, which is a range operator in a character class.

        Unescaped it spanned U+0022 to U+2018 -- every Bengali letter -- so the debris pattern
        matched any trailing mark and truncated শৰ্মা to শ.
        """
        assert fields.ORPHAN_MARK.search("শৰ্মা") is None
        assert fields.ORPHAN_MARK.search("নাম .ে") is not None


class TestDisagreementThreshold:
    """A flag that fires on half the corpus carries no information."""

    def test_noise_is_not_disagreement(self):
        assert not fields.materially_different("ৰেহমাত বসমাতাৰা", "ৰেহমাত বসমাতাৰা")
        assert not fields.materially_different("নলৈেন বসমাতাৰা", "নলৈন বসমাতাৰা")

    def test_genuinely_different_readings_are(self):
        assert fields.materially_different("ল্ষ্মা কস্ষু", "লঙ্ষু্ন কঙস্ত্ু")

    def test_the_threshold_is_the_measured_one(self):
        assert fields.NAME_AGREEMENT == 0.90


class TestGateDirection:
    """An error rate improves by falling; a gate that cannot say so judges the wrong thing."""

    def test_a_falling_error_rate_passes_a_lower_is_better_gate(self):
        result = bench.gate_target("wrong", 0.069, 0.063, minimum=0.005, higher_is_better=False)
        assert result.passed

    def test_a_rising_error_rate_fails_it(self):
        result = bench.gate_target("wrong", 0.063, 0.069, minimum=0.005, higher_is_better=False)
        assert not result.passed

    def test_the_default_is_still_higher_is_better(self):
        assert bench.gate_target("present", 0.80, 0.85).passed
        assert not bench.gate_target("present", 0.85, 0.80).passed


class TestAmbiguousAges:
    """An age of 95 that is really 26 is in range, so no floor detector can see it."""

    def test_a_contaminated_run_is_flagged(self):
        assert fields.age_is_ambiguous("বয়স ' 9526 !লঙ্গ ' মাহলা")

    def test_a_clean_two_digit_age_is_not(self):
        assert not fields.age_is_ambiguous("বয়স ' 46 [লঙ্গ ' মাহলা")

    def test_the_flag_reaches_the_router(self):
        lines = [
            "নাম : খাদৰাম ৰাভা",
            "পিতাৰ নাম : গংগাৰাম",
            "ঘৰ নং : 5",
            "বয়স ' 9526 !লঙ্গ ' মাহলা",
        ]
        elector = fields.assemble((lines, [lines[0]]), "HHK0001471", "106")
        assert "age_ambiguous" in elector.flags
        row = {"flags": ",".join(elector.flags), "relation_type": "father", "relation_name": "x"}
        assert "age_ambiguous" in escalate.doubtful(row)

    def test_the_threshold_follows_the_unambiguous_lines(self):
        """Two-digit runs put 2.3% of ages in the nineties; long runs put 13-15% there
        whichever end is read, so neither end is the age."""
        assert fields.AGE_DIGITS == 2


class TestEscalationCost:
    """Rows flagged overstates the cost: a re-read settles one band, not a whole box."""

    def test_cost_is_counted_in_bands_not_rows(self):
        rows = [
            {
                "name": "খাদৰাম",
                "relation_name": "গংগাৰাম",
                "relation_type": "father",
                "epic_no": "HHK0001471",
                "age": 35,
                "sex": "M",
                "flags": "age_ambiguous" if i % 2 else "",
            }
            for i in range(100)
        ]
        found = escalate.report(rows)
        assert found.volume == 0.5, "half the rows are flagged"
        assert found.by_field == {"age": 50}, "but only one band of each is in question"

    def test_every_reason_the_detectors_can_raise_maps_to_a_band(self):
        """An unmapped reason silently becomes 'box' and overstates the second pass's cost.

        Enumerated from the detectors rather than from a hand-written list, so a reason added
        later without a band fails here instead of quietly inflating the cost estimate.
        """
        broken = {
            "name": "Latin",
            "relation_name": "Latin",
            "relation_type": "",
            "epic_no": "nope",
            "age": 900,
            "sex": "",
            "flags": ",".join(escalate.DOUBT_FLAGS),
        }
        raised = set(escalate.reasons(broken)) | set(quality.problems_with(broken))
        unmapped = sorted(reason for reason in raised if reason not in escalate.REASON_FIELD)
        assert not unmapped, f"reasons with no band: {unmapped}"


class TestDetectorSeparation:
    """A detector can be falsified without ground truth, if not measured."""

    @staticmethod
    def _old(value):
        return value >= 90

    def test_a_detector_that_finds_trouble_concentrates_it(self):
        """Implausible ages sit in the flagged half far more often, but not only there."""
        rows = [
            {
                "age": 95 if (i % 4 == 0 if i % 2 == 0 else i % 20 == 1) else 30,
                "flags": "age_ambiguous" if i % 2 == 0 else "",
            }
            for i in range(200)
        ]
        found = escalate.separation(rows, "age_ambiguous", "age", self._old)
        assert found["concentration"] > 2.0
        assert found["unflagged"]["implausible"] > 0, "a zero base is a different case"

    def test_a_detector_flagging_at_random_scores_about_one(self):
        """The check that makes the whole thing falsifiable.

        Implausible ages are spread evenly across both halves, so a detector that splits the
        corpus in two has found nothing however sensible its reason sounds.
        """
        rows = [
            {"age": 95 if i % 10 < 2 else 30, "flags": "age_ambiguous" if i % 2 else ""}
            for i in range(200)
        ]
        found = escalate.separation(rows, "age_ambiguous", "age", self._old)
        assert 0.8 < found["concentration"] < 1.25

    def test_perfect_separation_is_not_reported_as_failure(self):
        """Dividing by a zero base returned 0.0 -- the score a useless detector gets."""
        rows = [
            {"age": 95 if i % 2 else 30, "flags": "age_ambiguous" if i % 2 else ""}
            for i in range(100)
        ]
        found = escalate.separation(rows, "age_ambiguous", "age", self._old)
        assert found["concentration"] == float("inf")

    def test_no_implausible_values_anywhere_is_no_information(self):
        rows = [{"age": 30, "flags": "age_ambiguous" if i % 2 else ""} for i in range(100)]
        found = escalate.separation(rows, "age_ambiguous", "age", self._old)
        assert found["concentration"] == 1.0


class TestRunTiming:
    """A multi-hour job has to be able to say what it cost."""

    def test_durations_read_the_way_a_long_run_is_discussed(self):
        assert timing.human(5) == "5s"
        assert timing.human(95) == "1m 35s"
        assert timing.human(3600 * 2 + 840) == "2h 14m"
        assert timing.human(-1) == "0s"

    def test_cached_parts_are_kept_out_of_the_per_part_cost(self):
        """Mixing them in makes a resumed run look faster than the pipeline is."""
        clock = timing.RunClock(total=10)
        clock.record(60.0, was_cached=False)
        clock.record(None, was_cached=True)
        assert clock.done == 2 and clock.cached == 1
        assert clock.worked == [60.0]
        assert "1 served from cache" in clock.summary()[0]

    def test_no_estimate_until_each_worker_has_read_a_part(self):
        """A resumed run serves cached parts in a second each.

        Counting those as throughput put the first estimate for a two-hour job at 57 seconds.
        """
        clock = timing.RunClock(total=154, workers=5)
        for _ in range(18):
            clock.record(None, was_cached=True)
        assert clock.eta is None, "eighteen instant parts are not evidence of throughput"
        for _ in range(5):
            clock.record(300.0, was_cached=False)
        assert clock.eta is not None

    def test_the_estimate_comes_from_throughput_not_per_part_cost(self):
        """Per-part cost ignores how many workers are running and what else the machine does."""
        clock = timing.RunClock(total=100, workers=5)
        clock.started -= 600
        for _ in range(10):
            clock.record(300.0, was_cached=False)
        # Ten parts in ten minutes means ninety more take ninety minutes, whatever each cost.
        assert clock.eta is not None
        assert 5300 < clock.eta < 5700

    def test_no_estimate_before_anything_finishes_or_after_everything_has(self):
        assert timing.RunClock(total=10).eta is None
        done = timing.RunClock(total=1)
        done.record(1.0, was_cached=False)
        assert done.eta is None

    def test_progress_names_the_part_and_its_duration(self):
        clock = timing.RunClock(total=154)
        clock.record(61.0, was_cached=False)
        line = clock.progress(7, 873, 61.0, False)
        assert "part 7" in line and "873 electors" in line and "1m 01s" in line

    def test_a_cached_part_says_so_instead_of_reporting_a_duration(self):
        clock = timing.RunClock(total=154)
        clock.record(None, was_cached=True)
        assert "(cached)" in clock.progress(8, 850, None, True)


class TestGatingAgainstARevision:
    """Git already stores every previous parser, so there is no baseline file to go stale."""

    def test_the_three_written_down_sets_stay_separate(self):
        """A regression part has been looked at; quoting it as held-out evidence is a lie."""
        parts = [1, 5, 12, 16, 41, 64]
        found = against.split(parts)
        assert found["in-sample"] == [12, 16]
        assert found["regression"] == [1, 5]
        assert found["out-of-sample"] == [41, 64]

    def test_an_empty_split_is_named_rather_than_dropped(self):
        assert against.split([41, 64])["in-sample"] == []

    def test_a_split_with_nothing_in_it_cannot_pass_its_gate(self):
        """A gate that quietly skips an unmeasured split passes by omission."""
        before, after = against._pair(None, "definitely_wrong")
        assert before == after == 0.0
        gate = bench.gate_target("out-of-sample", before, after, 0.005, higher_is_better=False)
        assert not gate.passed

    def test_the_baseline_parser_is_loaded_not_reimplemented(self):
        """A rewrite that drifts makes the fix look better or worse than it was."""
        module = against.field_module("HEAD")
        assert hasattr(module, "assemble") and hasattr(module, "assign_bands")
        assert module.assemble is not fields.assemble


class TestSerialZone:
    """The serial is a small number in a wide strip -- the shape that defeated the EPIC."""

    def test_the_zone_stays_clear_of_the_cliff(self):
        """Below 40% of the text column the serial is cut off and nothing reads at all.

        Measured on part 70: 35% of the width read 0 of 120 boxes. The chosen fraction has to
        keep a margin above that, which is why the per-part optimum was not taken.
        """
        assert 0.45 <= fields.SERIAL_FRACTION <= 0.65

    def test_the_zone_is_narrower_than_the_text_column(self):
        box = grid.build(H_RULES, V_RULES)[0]
        assert box.text_width == box.text_right - box.left
        zone = int(box.text_width * fields.SERIAL_FRACTION)
        assert zone < box.text_width, "a full-width crop is what made this slow"
        assert zone > box.text_width // 3, "too narrow and the serial falls outside it"


class TestForeignDebris:
    """The roll prints names in Assamese only, so a Latin letter in one is scanner debris."""

    def test_a_speck_beside_a_word_is_removed(self):
        assert fields.strip_foreign("দিপক নাজাখৰা 0") == ("দিপক নাজাখৰা", False)
        assert fields.strip_foreign("1পলাৰ ৰাভা") == ("পলাৰ ৰাভা", False)

    def test_a_digit_between_two_letters_closes_up_and_is_flagged(self):
        """It may have displaced a letter, so the value is repaired but the row stays routable.

        Removing rather than replacing with a space is what makes this a repair: নেম2্ৰা reads
        as নেম্ৰা, not as নেম ্ৰা.
        """
        value, uncertain = fields.strip_foreign("নেম2্ৰা নাজাৰা")
        assert value == "নেম্ৰা নাজাৰা"
        assert uncertain

    def test_separate_words_are_not_joined(self):
        assert fields.strip_foreign("ৰাম 1 কুমাৰ") == ("ৰাম কুমাৰ", False)

    def test_a_clean_name_is_untouched(self):
        assert fields.strip_foreign("খাদৰাম ৰাভা") == ("খাদৰাম ৰাভা", False)

    def test_bengali_digits_count_as_debris_too(self):
        """quality tests names with \\d, which in Python matches any Unicode decimal.

        While this pattern was Latin-only, ২ was provably wrong to the detector and invisible
        to the repair, and 261 rows stayed flagged after the strip meant to clear them.
        """
        cleaned = fields.strip_foreign("ৰোমোছ বসমতাুূৰ ২ ৷")[0]
        assert cleaned == "ৰোমোছ বসমতাুূৰ ৷"
        assert "name_has_latin_or_digits" not in quality.problems_with({"name": cleaned})

    def test_only_the_uncertain_case_reaches_the_router(self):
        """A speck beside a word is repaired with confidence and does not need re-reading."""
        clean = [
            "নাম : দিপক নাজাখৰা 0",
            "পিতাৰ নাম : গংগাৰাম",
            "ঘৰ নং : 5",
            "বয়স : 35 লিঙ্গ : পুৰুষ",
        ]
        assert "name_repaired" not in fields.assemble((clean, []), "HHK0001471", "1").flags
        risky = [
            "নাম : নেম2্ৰা নাজাৰা",
            "পিতাৰ নাম : গংগাৰাম",
            "ঘৰ নং : 5",
            "বয়স : 35 লিঙ্গ : পুৰুষ",
        ]
        assert "name_repaired" in fields.assemble((risky, []), "HHK0001471", "1").flags

    def test_the_house_number_and_epic_are_left_alone(self):
        """Both are digits or Latin by nature; stripping would empty them."""
        lines = [
            "নাম : খাদৰাম ৰাভা",
            "পিতাৰ নাম : গংগাৰাম",
            "ঘৰ নং : 42",
            "বয়স : 35 লিঙ্গ : পুৰুষ",
        ]
        elector = fields.assemble((lines, []), "HHK0001471", "106")
        assert elector.house_no == "42"
        assert elector.epic_no == "HHK0001471"


class TestFuzzyRelationLabel:
    """Enumerating damaged labels does not converge; the shape of the label is stable."""

    def test_the_variants_the_enumeration_missed_now_match(self):
        """1,227 boxes in 21,669 carried a ...নাম label matching none of the eight listed."""
        for line, kind in (
            ('মাতৰ নাম" গিলাশ্বৰা ৰাভা', "mother"),
            ("স্বামূৰ নাম : হুলশা", "husband"),
            ("স্সামাৰ নাম : হুলশা", "husband"),
            ("পৈতাৰ নাম : গংগাৰাম", "father"),
        ):
            found = fields.relation_label(line)
            assert found, line
            assert found[1] == kind, line

    def test_the_name_line_is_never_a_relation(self):
        """The one thing that must never match: it would swap the elector with a relative."""
        for line in ("নাম : খাদৰাম ৰাভা", "ঘৰ নং : 42", "বয়স : 35 লিঙ্গ : পুৰুষ", ""):
            assert fields.relation_label(line) is None, line

    def test_an_ambiguous_relative_yields_a_value_but_no_type(self):
        """1লতাৰ resembles মাতাৰ at 0.60 and is almost certainly a damaged পিতাৰ.

        Publishing a father as a mother is worse than publishing the relationship as unknown,
        and unknown routes the row for a second opinion.
        """
        found = fields.relation_label("1লতাৰ নাম : ৰাম")
        assert found and found[1] == ""

    def test_the_thresholds_keep_their_measured_margins(self):
        """Real variants scored 0.60 and above; ঘৰ, বয়স and a bare name scored 0.33 and below."""
        assert 0.33 < fields.RELATION_MATCH < 0.60
        assert fields.RELATION_TYPE_CONFIDENCE > fields.RELATION_MATCH


class TestCircularAssociations:
    """A feature derived from the failed field explains it by definition, not by evidence."""

    def test_relation_type_does_not_explain_a_missing_relation(self):
        """relation_type is empty because there is no relation. Reported as an 18.2x lift.

        The same family as the serial-gap check that reported zero forever because the serial
        was assigned by a counter: ask what would have to be true for the slice not to light
        up, and if the answer is "nothing", it is not a finding.
        """
        rows = [
            {
                "relation_name": "" if i % 2 else "গংগাৰাম",
                "relation_type": "" if i % 2 else "father",
            }
            for i in range(200)
        ]
        found = diagnose.associations(rows, features=("relation_type",))
        assert not [a for a in found if a.failure == "no_relation"]

    def test_a_real_association_on_the_same_feature_still_reports(self):
        """Only the definitional pair is suppressed, not the feature."""
        rows = [
            {
                "relation_name": "গংগাৰাম",
                "relation_type": "" if i % 2 else "father",
                "name": "" if i % 2 else "খাদৰাম",
            }
            for i in range(200)
        ]
        found = diagnose.associations(rows, features=("relation_type",))
        assert [a for a in found if a.failure == "no_name"]


class TestUnreadableAges:
    """The page plainly holds an age and the cheap pass could not read it."""

    def test_a_two_digit_age_with_one_digit_lost_is_flagged(self):
        """64% of missing ages look like this: বয়স ' 3/ -- one digit came back a symbol."""
        assert fields.age_is_unreadable("বয়স ' 3/ [লঙ্গ ' মাহলা")
        assert fields.age_is_unreadable("বয়স ' 4€ !লঙ্গ ' পৰষ")

    def test_a_readable_age_is_not_flagged(self):
        assert not fields.age_is_unreadable("বয়স ' 46 [লঙ্গ ' মাহলা")

    def test_a_band_with_no_digits_is_a_different_failure(self):
        """No digits means the OCR found nothing to read, not that a value was lost.

        The distinction is what makes escalation worth paying for: one says a better engine
        might recover the value, the other says there is nothing on the crop to recover.
        """
        assert not fields.age_is_unreadable("৷ ক্ল ভকঁনু কুঁমক্র ৷")
        assert not fields.age_is_unreadable("")

    def test_the_flag_routes_the_age_band_and_not_the_box(self):
        lines = ["নাম : খাদৰাম ৰাভা", "পিতাৰ নাম : গংগাৰাম", "ঘৰ নং : 5", "বয়স ' 3/ [লঙ্গ ' মাহলা"]
        elector = fields.assemble((lines, []), "HHK0001471", "1")
        assert "age_unreadable" in elector.flags
        assert escalate.REASON_FIELD["age_unreadable"] == "age"


class TestSecondPass:
    """A second engine is only worth its cost if what it returns can be trusted."""

    def test_it_reads_every_field_out_of_a_pipe_separated_answer(self):
        """Names and relations included: the first pass gets 0 of 16 names exactly right."""
        reading = second_pass.Reading(
            "61|মৌকথাং|পিতাৰ নাম: বীজেন্দ্ৰ|ঘৰ নং : ১৩|বয়স : 24 লিঙ্গ : পুৰুষ|"
        )
        assert reading.fields() == {
            "name": "মৌকথাং",
            "relation_name": "বীজেন্দ্ৰ",
            "relation_type": "father",
            "age": 24,
            "house_no": "13",
            "sex": "M",
        }

    def test_the_same_answer_wrapped_in_markup_reads_the_same(self):
        """The model returns bare pipes, <p> wrappers and table rows for one prompt.

        Fields are found by label because position is the thing that moves.
        """
        wrapped = second_pass.Reading("<p>63|নাম : কৌশিলা|ঘৰ নং : ১৪|বয়স : 52 লিঙ্গ : মহিলা</p>")
        assert wrapped.fields() == {"name": "কৌশিলা", "age": 52, "house_no": "14", "sex": "F"}

    def test_html_from_the_full_model_reads_the_same_as_pipes(self):
        """The full model answers in HTML and the distilled one in pipes; one reader for both.

        A span is inline, not a line break -- treating it as one severed নাম from the value
        after it, and the label itself became the name.
        """
        html = (
            "<p>নাম<span> </span>: কৌশিলা<br/>ঘৰ নং<span> </span>: ১৪<br/>"
            "বয়স<span> </span>: 52 লিঙ্গ<span> </span>: মহিলা</p>"
        )
        assert second_pass.Reading(html).fields() == {
            "name": "কৌশিলা",
            "age": 52,
            "house_no": "14",
            "sex": "F",
        }

    def test_a_truncated_answer_is_refused(self):
        """A cap landing mid-value yields a plausible wrong number, which is the whole point."""
        cut = second_pass.Reading("61|মৌকথাং|ঘৰ নং : ১৩|বয়স : 2", tokens=160, capped=True)
        assert not cut.usable and cut.fields() == {}

    def test_a_looping_answer_keeps_its_first_cycle(self):
        text = "|76|সিবিৰানী|ঘৰ নং: ১৭|বয়স: 54|লিঙ্গ: মহিলা|" * 3
        looped = second_pass.Reading(text, tokens=160, capped=True)
        assert looped.usable
        assert looped.fields()["age"] == 54

    def test_an_implausible_age_is_not_accepted_from_the_second_engine_either(self):
        assert "age" not in second_pass.Reading("ঘৰ নং : ১৪|বয়স : 7 লিঙ্গ : পুৰুষ").fields()

    def test_it_fills_gaps(self):
        row = {"age": 35, "house_no": "", "sex": "M", "flags": ""}
        merged = second_pass.merge(row, second_pass.Reading("ঘৰ নং : ১৪|বয়স : 35 লিঙ্গ : পুৰুষ"))
        assert merged["house_no"] == "14"

    def test_a_disputed_age_goes_to_the_second_pass_and_keeps_the_first(self):
        """The crops were read by eye and the second pass won every one: 59 against 92.

        Both values are kept, because the evidence is six crops on one page -- enough to act
        on, not enough to throw away what it replaced.
        """
        row = {"age": 92, "house_no": "", "sex": "M", "flags": ""}
        merged = second_pass.merge(row, second_pass.Reading("ঘৰ নং : ১৪|বয়স : 59 লিঙ্গ : পুৰুষ"))
        assert merged["age"] == 59
        assert merged["age_first_pass"] == 92
        assert "second_pass_overruled" in merged["flags"]

    def test_a_disputed_house_number_is_not_overruled(self):
        """The second pass has only ever filled gaps in house numbers, never contradicted one."""
        row = {"age": 35, "house_no": "42", "sex": "M", "flags": ""}
        merged = second_pass.merge(row, second_pass.Reading("ঘৰ নং : ১৪|বয়স : 35 লিঙ্গ : পুৰুষ"))
        assert merged["house_no"] == "42"
        assert "house_no:42!=14" in merged["second_pass_disagreements"]

    def test_only_rows_missing_something_it_can_supply_are_sent(self):
        """A second read returning what we already had is six seconds spent on nothing."""
        assert not second_pass.wanted({"age": 35, "house_no": "14", "sex": "M"})
        assert second_pass.wanted({"age": None, "house_no": "14", "sex": "M"})
        assert second_pass.wanted({"age": 35, "house_no": "", "sex": "M"})


class TestEngineEvaluation:
    """The only real ground truth here is a crop read by eye."""

    TRUTH = {"a": {"name": "সমে নাৰ্জাৰী", "house": "7", "age": 54, "sex": "F"}}

    def test_it_scores_an_engine_against_the_page(self):
        sample = [
            {
                "key": "a",
                "t_name": "সমে নাজাৰা",
                "t_age": 25,
                "t_house": "",
                "t_sex": "F",
                "surya": (
                    "31|নাম : সমে নাৰ্জাৰী|স্বামীৰ নাম: তিৰেন" "|ঘৰ নং : ৭|বয়স : 54 লিঙ্গ : মহিলা|"
                ),
            }
        ]
        tallies = evaluate.score(self.TRUTH, sample)
        assert tallies["tesseract"].name_exact == 0 and tallies["savitr-terse"].name_exact == 1
        assert tallies["tesseract"].age == 0 and tallies["savitr-terse"].age == 1
        assert tallies["tesseract"].house == 0 and tallies["savitr-terse"].house == 1
        assert tallies["tesseract"].sex == 1 and tallies["savitr-terse"].sex == 1

    def test_a_name_off_by_a_matra_counts_as_nearly_right(self):
        """Distinguishes a different spelling from a different person."""
        sample = [
            {
                "key": "a",
                "t_name": "সমে নাৰ্জাৰি",
                "t_age": None,
                "t_house": "",
                "t_sex": "",
                "surya": "",
            }
        ]
        tallies = evaluate.score(self.TRUTH, sample)
        assert tallies["tesseract"].name_exact == 0
        assert tallies["tesseract"].name_nearly == 1

    def test_fields_are_found_whatever_shape_the_answer_arrives_in(self):
        bare = evaluate.terse_fields("31|সমে নাৰ্জাৰী|ঘৰ নং : ৭|বয়স : 54 লিঙ্গ : মহিলা|")
        wrapped = evaluate.terse_fields(
            "<p>31|নাম : সমে নাৰ্জাৰী|ঘৰ নং : ৭|বয়স : 54 লিঙ্গ : মহিলা</p>"
        )
        assert bare["age"] == wrapped["age"] == 54
        assert bare["house"] == wrapped["house"] == "7"
        assert bare["sex"] == wrapped["sex"] == "F"

    def test_a_relation_line_is_not_taken_for_the_name(self):
        found = evaluate.terse_fields("31|নাম : সমে|স্বামীৰ নাম: তিৰেন|বয়স : 54 লিঙ্গ : মহিলা|")
        assert found["name"] == "সমে"


class TestGeminiFields:
    """The schema removes the parser as a variable; what is left is the digit script."""

    def test_house_numbers_come_back_in_the_script_they_are_printed_in(self):
        """The prompt asks for the page exactly as printed, and ১৩ is what is printed.

        Scoring those against Latin digits marked the model wrong on 14 of 16 house numbers
        it had read correctly.
        """
        assert gemini.fields_of({"house_no": "১৩"})["house"] == "13"
        assert gemini.fields_of({"house_no": "13"})["house"] == "13"

    def test_the_unknown_sentinel_becomes_an_empty_field(self):
        """The API rejects an empty enum member, so the schema names the sentinel instead."""
        assert gemini.fields_of({"sex": "unknown"})["sex"] == ""
        assert gemini.fields_of({"name": "unknown"})["name"] == ""
        assert gemini.fields_of({"sex": "F"})["sex"] == "F"

    def test_cost_halves_in_batch_mode(self):
        usage = gemini.Usage(input_tokens=1_000_000, output_tokens=1_000_000)
        assert usage.cost(0.10, 0.40, batch=False) == 0.5
        assert usage.cost(0.10, 0.40, batch=True) == 0.25

    def test_a_request_covers_a_column_not_a_page(self):
        """A page costs the same tokens as a box and loses the detail: 20% of names near-right
        against 80%, and no house numbers at all."""
        assert gemini.BOXES_PER_REQUEST == 10


class TestScriptFolding:
    """Two codepoints for one letter is not a reading error."""

    def test_bengali_and_assamese_ra_are_the_same_letter(self):
        """পবিত্র লাক্রা against পবিত্ৰ লাক্ৰা differs in nothing else.

        Scoring them apart marked correct readings wrong and moved surya-full from 10 of 16
        exact names down to 6.
        """
        assert evaluate.fold("পবিত্র লাক্রা") == evaluate.fold("পবিত্ৰ লাক্ৰা")

    def test_ba_and_wa_are_left_alone(self):
        """They look like a matching pair but are distinct letters in Assamese, and folding
        them changed no score -- which is the evidence for leaving them, not an argument."""
        assert evaluate.fold("বৰুৱা") != evaluate.fold("ৱৰুৱা")

    def test_a_genuinely_different_name_still_fails(self):
        assert evaluate.fold("সমে নাৰ্জাৰী") != evaluate.fold("দৌমেল নাৰ্জাৰী")

    def test_house_numbers_fold_whitespace(self):
        """The same page prints "20 ক" and "20ক"."""
        tally = evaluate.Score("x")
        evaluate.score_one(
            {"house": "20 ক", "name": "", "age": None, "sex": ""},
            {"house": "20ক", "name": "x", "age": 1, "sex": "M"},
            tally,
        )
        assert tally.house == 1

    def test_an_empty_name_is_never_exactly_right(self):
        tally = evaluate.Score("x")
        evaluate.score_one(
            {"name": "", "age": None, "house": "", "sex": ""},
            {"name": "", "age": 1, "house": "1", "sex": "M"},
            tally,
        )
        assert tally.name_exact == 0


class TestVisionStacking:
    """Vision bills per image, so pages are stacked -- and then coordinates have to come back."""

    def _pages(self, count=3, width=100, height=200):
        from PIL import Image

        return [Image.new("L", (width, height), "white") for _ in range(count)]

    def test_pages_stack_and_record_where_each_one_starts(self):
        stack = vision.stack_pages(self._pages(3))
        assert stack.image.size == (100, 600)
        assert stack.offsets == [0, 200, 400]

    def test_a_word_on_the_second_page_comes_back_to_the_second_page(self):
        """The one off-by-one that would misfile every field on every page but the first.

        Vision answers in the stacked image's coordinates; page two's first line sits at y=210
        there and at y=10 on its own page.
        """
        stack = vision.stack_pages(self._pages(3))
        word = vision.Word("খাদৰাম", left=5, top=210, right=60, bottom=230)
        moved = word.shifted(stack.offsets[1])
        assert (moved.top, moved.bottom) == (10, 30)
        assert moved.text == word.text

    def test_a_stack_over_the_pixel_ceiling_is_refused_before_it_is_sent(self):
        from PIL import Image

        huge = [Image.new("L", (9000, 9000), "white")]
        assert vision.stack_pages(huge).check() is not None
        assert vision.stack_pages(self._pages(2)).check() is None

    def test_too_many_images_in_one_request_is_an_error_not_a_silent_truncation(self):
        import pytest

        with pytest.raises(ValueError):
            vision.annotate(self._pages(vision.IMAGES_PER_REQUEST + 1), "key")

    def test_cost_is_counted_in_images_because_that_is_what_is_billed(self):
        """4,500 pages stacked eight deep is 563 images, not 4,500."""
        assert vision.cost(1000) == 1.50
        pages, per = 4500, vision.PAGES_PER_IMAGE
        images = -(-pages // per)
        assert vision.cost(images) < 1.0


class TestVisionWordMapping:
    """Words are matched into the geometry this stage already derives, not cropped and re-read."""

    @staticmethod
    def _word(text, left, top, right, bottom):
        return {
            "symbols": [{"text": text}],
            "boundingBox": {
                "vertices": [
                    {"x": left, "y": top},
                    {"x": right, "y": top},
                    {"x": right, "y": bottom},
                    {"x": left, "y": bottom},
                ]
            },
        }

    @property
    def RESPONSE(self):
        """A response in Vision's page/block/paragraph/word shape, built rather than nested."""
        words = [
            self._word("নাম", 10, 10, 40, 30),
            self._word("খাদৰাম", 50, 10, 110, 30),
            self._word("গংগাৰাম", 10, 60, 90, 80),
        ]
        return {"fullTextAnnotation": {"pages": [{"blocks": [{"paragraphs": [{"words": words}]}]}]}}

    def test_words_are_read_out_of_the_response_with_their_boxes(self):
        words = vision.words_from(self.RESPONSE)
        assert [w.text for w in words] == ["নাম", "খাদৰাম", "গংগাৰাম"]
        assert (words[0].left, words[0].top) == (10, 10)

    def test_words_land_in_the_band_they_sit_in(self):
        words = vision.words_from(self.RESPONSE)
        lines = vision.lines_in(words, [(5, 35), (55, 85)], left=0, right=200)
        assert lines == ["নাম খাদৰাম", "গংগাৰাম"]

    def test_a_word_grazing_the_band_above_stays_where_it_belongs(self):
        """Bands are padded and adjacent, so a tall glyph can touch its neighbor."""
        grazing = [vision.Word("ৰাভা", left=10, top=50, right=60, bottom=80)]
        assert vision.lines_in(grazing, [(5, 55)], 0, 200) == [""]
        assert vision.lines_in(grazing, [(55, 85)], 0, 200) == ["ৰাভা"]

    def test_words_outside_the_column_are_not_pulled_in(self):
        """The photo strip and the neighboring box sit at the same heights."""
        words = vision.words_from(self.RESPONSE)
        assert vision.lines_in(words, [(5, 35)], left=0, right=45) == ["নাম"]


class TestVisionPageMapping:
    """A word on page two of a stack has to come back to page two.

    Unshifted it lands in whichever box sits at that height on page one -- a plausible wrong
    answer on every page but the first, with nothing raised and nothing to notice.
    """

    @staticmethod
    def _response(words):
        boxed = [
            {
                "symbols": [{"text": text}],
                "boundingBox": {
                    "vertices": [
                        {"x": 10, "y": top},
                        {"x": 90, "y": top},
                        {"x": 90, "y": top + 20},
                        {"x": 10, "y": top + 20},
                    ]
                },
            }
            for text, top in words
        ]
        return {"fullTextAnnotation": {"pages": [{"blocks": [{"paragraphs": [{"words": boxed}]}]}]}}

    def _run(self, monkeypatch, words, pages=3, height=100, per_image=8, width=200):
        from PIL import Image

        captured = {"images": []}

        def fake_annotate(images, api_key, **_kwargs):
            captured["images"] += list(images)
            return [self._response(words) for _ in images]

        monkeypatch.setattr(vision, "annotate", fake_annotate)
        images = [Image.new("L", (width, height), "white") for _ in range(pages)]
        found, billed = vision_part.words_per_page(images, api_key="x", pages_per_image=per_image)
        return found, billed, captured

    def test_each_page_gets_the_words_that_sit_on_it(self, monkeypatch):
        found, _, _ = self._run(monkeypatch, [("এক", 10), ("দুই", 110), ("তিনি", 210)])
        assert [[w.text for w in page] for page in found] == [["এক"], ["দুই"], ["তিনি"]]

    def test_words_come_back_in_their_own_page_coordinates(self, monkeypatch):
        """Page two's word sits at y=110 in the stack and at y=10 on its own page."""
        found, _, _ = self._run(monkeypatch, [("এক", 10), ("দুই", 110)])
        assert found[0][0].top == 10
        assert found[1][0].top == 10

    def test_the_last_page_keeps_words_below_the_final_offset(self, monkeypatch):
        """An open-ended last band. Bounded by the offset of a page that does not exist, the
        bottom page would come back empty."""
        found, _, _ = self._run(monkeypatch, [("শেষ", 250)], pages=3)
        assert [w.text for w in found[2]] == ["শেষ"]

    def test_pages_are_stacked_and_billed_as_one_image(self, monkeypatch):
        """The whole reason to use Vision: it bills per image, not per page."""
        _, billed, captured = self._run(monkeypatch, [("এক", 10)], pages=8, per_image=8)
        assert billed == 1
        assert len(captured["images"]) == 1
        assert captured["images"][0].height == 800

    def test_a_part_longer_than_one_stack_is_billed_per_stack(self, monkeypatch):
        found, billed, _ = self._run(monkeypatch, [("এক", 10)], pages=9, per_image=8)
        assert billed == 2
        assert len(found) == 9

    def test_a_smaller_per_request_limit_splits_rather_than_drops(self, monkeypatch):
        """Truncating a batch loses pages with nothing raised. Every page must come back."""
        from PIL import Image

        seen = {"images": 0}

        def fake_annotate(images, api_key, **_kwargs):
            seen["images"] += len(images)
            return [self._response([("এক", 10)]) for _ in images]

        monkeypatch.setattr(vision, "annotate", fake_annotate)
        pages = [Image.new("L", (200, 100), "white") for _ in range(6)]
        found, billed = vision_part.words_per_page(
            pages, api_key="x", pages_per_image=1, images_per_request=2
        )
        assert seen["images"] == 6
        assert billed == 6
        assert len(found) == 6

    def test_an_oversized_stack_is_refused_before_a_request_is_spent(self, monkeypatch):
        from PIL import Image

        def fail(*a, **k):
            raise AssertionError("a request was sent for a stack that would be rejected")

        monkeypatch.setattr(vision, "annotate", fail)
        huge = [Image.new("L", (9000, 9000), "white") for _ in range(2)]
        with pytest.raises(ValueError, match="pixels"):
            vision_part.words_per_page(huge, api_key="x", pages_per_image=2)

    def test_cost_is_counted_in_images_not_pages(self):
        """A 30-page part is four images, and the state is hundreds rather than thousands."""
        assert vision_part.parts_cost([30], pages_per_image=8) == pytest.approx(
            4 / 1000 * vision.COST_PER_1000
        )

    def test_the_stacking_factor_is_measured_from_the_pages(self, monkeypatch):
        """It was the constant 8, which is right at 300 dpi. The pipeline renders at 400, where
        eight pages is 124 MP against a 75 MP ceiling, and every part was refused.

        Real 400-dpi dimensions, because that is the size the constant got wrong.
        """
        found, billed, captured = self._run(
            monkeypatch, [("এক", 10)], pages=8, width=3400, height=4400, per_image=None
        )
        assert all(image.height * image.width <= vision.MAX_PIXELS for image in captured["images"])
        # Eight pages at three per image under the 0.6 margin.
        assert billed == 3
        assert len(found) == 8


class TestVisionStackingFactor:
    """How many pages fit in one image, derived rather than assumed."""

    def test_a_400_dpi_page_stacks_three_deep(self):
        """3,400 x 4,400 is a 400-dpi A4 page. Eight would be 120 MP."""
        # Three, not four, since the margin came down to 0.6: a 67 MP image under the
        # old 0.9 budget hit the OCR backend's deadline on every attempt.
        assert vision.pages_that_fit(3400, 4400) == 3

    def test_a_300_dpi_page_stacks_five_deep(self):
        """2,480 x 3,509 is what these pages actually render to at 300 dpi -- 8.7 MP, so seven
        fit under the ceiling with the margin."""
        assert vision.pages_that_fit(2480, 3509) == 5

    def test_the_native_scan_stacks_a_whole_part_into_one_image(self):
        """1187x1679 is the source's own resolution, and the reason native costs $55 against
        $387: a 25-page part is one submitted image."""
        assert vision.pages_that_fit(1187, 1679) >= 20

    def test_the_ceiling_binds_only_when_the_limits_do_not(self):
        """It was 8, which is no limit of the API, and at native it -- not the pixel ceiling --
        was what capped the cheapest arm."""
        assert vision.pages_that_fit(10, 10) == vision.PAGES_PER_IMAGE
        assert vision.PAGES_PER_IMAGE >= 32

    def test_a_page_too_large_to_stack_still_goes_one_at_a_time(self):
        """Refusing outright would lose the part; one page an image merely costs more."""
        assert vision.pages_that_fit(9000, 9000) == 1

    def test_the_stack_leaves_room_for_a_page_larger_than_the_median(self):
        """The factor is chosen from the pages in hand, and a supplement page renders taller.
        Filling the ceiling exactly would have a later page tip a stack over it."""
        fit = vision.pages_that_fit(3400, 4400)
        assert fit * 3400 * 4400 <= vision.MAX_PIXELS * vision.PIXEL_MARGIN


class TestResolutionComparison:
    """Three renderings of one 144 dpi scan, compared without lying about it."""

    @staticmethod
    def _arm(name, boxes, pages=8, images=2):
        return resolution.ArmResult(
            arm=name, part=13, boxes=boxes, pages=pages, images_billed=images
        )

    def test_only_boxes_every_arm_returned_are_compared(self):
        """A resolution that resolves fewer boxes has failed at geometry, and scoring it on the
        ones it did resolve would flatter it."""
        arms = [
            self._arm("400 dpi", {(4, 0, 0): {}, (4, 1, 0): {}}),
            self._arm("native", {(4, 0, 0): {}}),
        ]
        assert resolution.common_boxes(arms) == [(4, 0, 0)]

    def test_agreement_needs_every_arm_to_say_the_same_thing(self):
        """Two of three agreeing is a disagreement: the question is whether resolution changes
        the answer at all, not which answer wins a vote."""
        box = (4, 0, 0)
        arms = [
            self._arm("400 dpi", {box: {"name": "ৰাম", "age": 40, "house": "1", "sex": "M"}}),
            self._arm("300 dpi", {box: {"name": "ৰাম", "age": 40, "house": "1", "sex": "M"}}),
            self._arm("native", {box: {"name": "ৰাম", "age": 41, "house": "1", "sex": "M"}}),
        ]
        found = resolution.agreement(arms)
        assert found.boxes == 1
        assert found.rate("name") == 1.0
        assert found.rate("age") == 0.0
        assert found.examples["age"], "a disagreement must be shown, not just counted"

    def test_agreement_folds_the_script_variants_the_scorer_folds(self):
        """Bengali RA and Assamese RA are one letter. Counting them apart would report a
        disagreement the scorer would call a match."""
        box = (4, 0, 0)
        arms = [
            self._arm("400 dpi", {box: {"name": "পবিত্র", "house": "20 ক"}}),
            self._arm("native", {box: {"name": "পবিত্ৰ", "house": "20ক"}}),
        ]
        found = resolution.agreement(arms)
        assert found.rate("name") == 1.0
        assert found.rate("house") == 1.0

    def test_cost_comes_from_the_arm_s_own_measured_pages_per_image(self):
        """Not from a constant. The constant is what got the stacking factor wrong."""
        arms = [
            self._arm("400 dpi", {}, pages=34, images=9),
            self._arm("native", {}, pages=34, images=2),
        ]
        costs = resolution.cost_for_state(arms, pages_in_state=1000)
        assert costs["400 dpi"] > costs["native"]
        assert costs["native"] == pytest.approx(vision.cost(round(1000 / 17)))

    def test_truth_is_matched_by_box_position_not_by_order(self):
        key = "p13_pg4_r4c0"
        truth = {key: {"name": "ৰাম", "age": 40, "house": "1", "sex": "M"}}
        arms = [
            self._arm("native", {(4, 4, 0): {"name": "ৰাম", "age": 40, "house": "1", "sex": "M"}})
        ]
        tallies = resolution.score_against_truth(arms, truth)
        assert tallies["native"].total == 1
        assert tallies["native"].name_exact == 1

    def test_the_truth_pages_are_found_from_the_keys(self):
        """Running whole parts for six pages would cost thirty times what the truth arm needs."""
        truth = {"p13_pg4_r4c0": {}, "p13_pg4_r1c0": {}, "p25_pg5_r2c0": {}}
        assert resolution.truth_parts(truth) == {13: [4], 25: [5]}

    def test_box_keys_are_only_unique_within_a_part(self):
        """(page, row, column) repeats in every part. Pooling six parts and intersecting
        compares part 7's page 4 with part 64's, and the first real run reported 0 boxes."""
        arms = [
            self._arm("400 dpi", {(4, 0, 0): {"name": "ৰাম"}}),
            resolution.ArmResult("400 dpi", 64, {(4, 0, 0): {"name": "শ্যাম"}}),
            self._arm("native", {(4, 0, 0): {"name": "ৰাম"}}),
            resolution.ArmResult("native", 64, {(4, 0, 0): {"name": "শ্যাম"}}),
        ]
        found = resolution.agreement(arms)
        assert found.boxes == 2
        assert found.rate("name") == 1.0

    def test_a_part_whose_arm_failed_does_not_void_the_others(self):
        """Part 25's native request died on DNS and part 33's 400 dpi on a broken pipe.
        Intersecting across the whole run then yields nothing, which reads as "the arms never
        agree" rather than "two requests failed".
        """
        arms = [
            self._arm("400 dpi", {(4, 0, 0): {"name": "ৰাম"}}),
            self._arm("native", {(4, 0, 0): {"name": "ৰাম"}}),
            resolution.ArmResult("400 dpi", 25, {(4, 0, 0): {"name": "শ্যাম"}}),
            resolution.ArmResult("native", 25, {}, error="DNS"),
        ]
        assert [p for p, _ in resolution.comparable_parts(arms)] == [13]
        assert resolution.agreement(arms).boxes == 1

    def test_readings_survive_a_round_trip_to_disk(self, tmp_path):
        """The first run of this comparison spent three hours and every request, then reported
        nothing because of an arithmetic bug over readings that were fine."""
        arms = [self._arm("native", {(4, 2, 1): {"name": "ৰাম", "age": 40}}, pages=8, images=1)]
        back = resolution.load(resolution.save(arms, tmp_path / "arms.json"))
        assert back[0].boxes == arms[0].boxes
        assert back[0].arm == "native" and back[0].part == 13
        assert back[0].images_billed == 1

    def test_an_empty_field_is_counted_against_the_arm_that_left_it_empty(self):
        """The measurement that settled the resolution question, and it needs no truth: a
        missing value is a failure whichever spelling would have been right.

        Agreement said only that the arms differ on 23% of names; 25 hand-read boxes could not
        say which was right at that margin. The blank rates could -- 300 dpi leaves the name
        empty thirteen times as often.
        """
        box = (4, 0, 0)
        arms = [
            self._arm("400 dpi", {box: {"name": "ৰাম", "age": 40, "house": "1", "sex": "M"}}),
            self._arm("300 dpi", {box: {"name": "", "age": 40, "house": "1", "sex": "M"}}),
        ]
        rates = resolution.blank_rates(arms)
        assert rates["400 dpi"]["name"] == 0.0
        assert rates["300 dpi"]["name"] == 1.0

    def test_a_narrowed_comparison_ignores_fields_the_crop_cannot_hold(self):
        """A name-band crop is one line, so its age, house and sex are empty by construction.
        Counting those as disagreements measures the crop, not the reader."""
        box = (4, 0, 0)
        arms = [
            self._arm("cloud-vision", {box: {"name": "ৰাম", "age": 40, "house": "1", "sex": "M"}}),
            self._arm("dots.ocr band", {box: {"name": "ৰাম", "age": None, "house": "", "sex": ""}}),
        ]
        assert resolution.agreement(arms).rate("age") == 0.0
        narrowed = resolution.agreement(arms, fields=("name",))
        assert narrowed.rate("name") == 1.0
        rates = resolution.blank_rates(arms, fields=("name",))
        assert rates["dots.ocr band"] == {"name": 0.0}

    def test_a_disagreement_names_its_part_as_well_as_its_page(self):
        """Box positions repeat in every part, so a page number alone points at ten boxes."""
        box = (4, 1, 2)
        arms = [
            self._arm("400 dpi", {box: {"name": "ৰাম"}}),
            self._arm("300 dpi", {box: {"name": "শ্যাম"}}),
        ]
        text = resolution.report(arms)
        assert "part 13 page 4 r1c2" in text

    def test_the_report_never_calls_agreement_accuracy(self):
        """Three arms of one engine can be wrong together, and are likelier to be than three
        different engines would be."""
        text = resolution.report([self._arm("400 dpi", {(4, 0, 0): {"name": "ৰাম"}})])
        assert "AGREEMENT BETWEEN ARMS" in text
        assert "agreement, not accuracy" in text


class TestRequestPayloadCeiling:
    """The limit that bit: 40 MB per request, separate from 20 MB per image.

    Sixteen 6 MB images are legal one at a time and 96 MB together. Batching by count alone
    rejected every 400 dpi and 300 dpi part while letting native through, which reads as "the
    expensive arms do not work" rather than "the batcher does not".
    """

    def test_a_batch_is_split_before_it_exceeds_the_request_limit(self):
        six_mb = [b"x" * (6 * 1024 * 1024)] * 8
        groups = vision.batch_images(six_mb)
        budget = vision.MAX_REQUEST_BYTES / vision.BASE64_GROWTH
        assert len(groups) > 1
        for group in groups:
            assert sum(len(six_mb[i]) for i in group) <= budget

    def test_the_budget_allows_for_base64_growth(self):
        """The ceiling is on what goes on the wire, and base64 costs four bytes for three."""
        just_under_raw = [b"x" * (30 * 1024 * 1024)] * 2
        assert len(vision.batch_images(just_under_raw)) == 2

    def test_small_images_still_batch_up_to_the_count_limit(self):
        tiny = [b"x" * 1024] * (vision.IMAGES_PER_REQUEST + 3)
        groups = vision.batch_images(tiny)
        assert [len(g) for g in groups] == [vision.IMAGES_PER_REQUEST, 3]

    def test_every_image_lands_in_exactly_one_batch(self):
        sizes = [b"x" * (i * 1024 * 1024) for i in (1, 9, 2, 15, 7, 3)]
        flat = [i for group in vision.batch_images(sizes) for i in group]
        assert flat == list(range(len(sizes)))

    def test_an_image_too_big_to_share_goes_alone(self):
        """Legal as long as it is under the per-image limit, which encode already checked.

        30 MiB is the whole budget once base64 growth is taken off the 40 MB request ceiling,
        so it can carry nothing else.
        """
        alone = int(vision.MAX_REQUEST_BYTES / vision.BASE64_GROWTH)
        sizes = [b"x" * 1024, b"x" * alone, b"x" * 1024]
        assert vision.batch_images(sizes) == [[0], [1], [2]]


class TestStackByteCeiling:
    """Two ceilings, and the one that binds depends on the scan."""

    def test_the_byte_limit_can_bind_before_the_pixel_limit(self):
        """On these pages pixels bind at every resolution tried. A noisier scan reverses it, and
        a 21 MB image is a failed request rather than a wrong answer."""
        loose = vision.pages_that_fit(1187, 1679)
        tight = vision.pages_that_fit(1187, 1679, bytes_per_page=4 * 1024 * 1024)
        assert tight < loose
        assert tight == int(vision.MAX_BYTES * vision.PIXEL_MARGIN) // (4 * 1024 * 1024)

    def test_a_measured_page_size_is_what_it_costs_as_png(self):
        from PIL import Image

        assert vision.encoded_size(Image.new("L", (50, 50), "white")) > 0


class TestSectionMarkersAreWords:
    """A village is not a supplement.

    Part 38's polling station is যোগেন্দ্ৰপুৰ -- Jogendrapur -- and the marker for an addition
    list is যোগ, its first three letters. Matched as a substring, every page of that station was
    filed as a supplement: 534 rows of a 1,048-row main list, and the part then reconciled at
    514 against its own printed total.
    """

    STATION = (
        "বিধানসভা সমষ্টিৰ নম্বৰ আৰু নাম : 1-গোসাইগাওঁ খণ্ড নং : 38 "
        "অংশৰ নম্বৰ আৰু নাম : 1-যোগেন্দ্ৰপুৰ এফ ভি"
    )
    SUPPLEMENT = (
        "বিধানসভা সমষ্টিৰ নম্বৰ আৰু নাম : 1-গোসাইগাওঁ খণ্ড নং : 38 "
        "যোগ তালিকা 1 (27-12-2025 04-02-2026 )"
    )

    def test_a_station_named_jogendrapur_is_a_main_roll_page(self):
        assert pages.section_of(self.STATION)[0] is pages.Section.MAIN

    def test_the_real_addition_list_is_still_found(self):
        assert pages.section_of(self.SUPPLEMENT)[0] is pages.Section.ADDITION

    def test_a_station_ending_in_abad_is_not_a_deletion_list(self):
        """বাদ would have been the worse one: countless place names end in -abad."""
        header = "সমষ্টিৰ নাম : 1-ক অংশৰ নম্বৰ আৰু নাম : 5-আহমদাবাদ এফ ভি"
        assert pages.section_of(header)[0] is pages.Section.MAIN

    def test_a_bengali_station_ending_in_abad_is_not_a_deletion_list(self):
        header = "সমষ্টির নাম : 1-ক অংশের নম্বর এবং নাম : 5-ইসলামাবাদ"
        assert pages.section_of(header)[0] is pages.Section.MAIN

    def test_a_sparse_page_after_an_addition_keeps_the_addition_section(self):
        assert (
            pages.section_after(pages.Section.ADDITION, pages.Section.MAIN)
            is pages.Section.ADDITION
        )

    def test_the_main_list_can_resume_after_a_short_deletion_sheet(self):
        assert pages.section_after(pages.Section.DELETION, pages.Section.MAIN) is pages.Section.MAIN

    def test_a_deletion_list_is_still_found(self):
        assert pages.section_of("খণ্ড নং : 7 বিয়োজন তালিকা 1")[0] is pages.Section.DELETION

    def test_a_marker_inside_the_station_name_never_counts(self):
        """Even a whole-word marker is not a marker when it sits in the station field."""
        header = "খণ্ড নং : 3 অংশৰ নম্বৰ আৰু নাম : 9-যোগ এফ ভি"
        assert pages.section_of(header)[0] is pages.Section.MAIN

    def test_an_unreadable_header_is_flagged_not_guessed(self):
        section, recognised = pages.section_of("qqq zzz")
        assert section is pages.Section.MAIN and not recognised


class TestStatusCodesAreTheRollsOwn:
    """The roll defines its own codes, in a legend at the foot of every closing page.

        E-মৃত, S-স্থানান্তৰিত/ বাসস্থান পৰিবৰ্তন, R-প্ৰতিলিপি, M-নিখোজ, Q-অযোগ্য

    Dead, shifted, duplicate, missing, ineligible. A closed set, which is what makes anything
    else in that cell a misread digit rather than a marking -- and the difference between
    detecting 5,167 struck-off entries in AC1 against an implied 3,816, and detecting 3,826.
    """

    @pytest.mark.parametrize(
        "cell,code", [("E 15", "E"), ("S 134", "S"), ("R 3", "R"), ("M 7", "M"), ("Q 9", "Q")]
    )
    def test_each_code_the_roll_defines_is_read(self, cell, code):
        found = re.match(rf"^\s*([{vision_part.STATUS_CODES}])[\s.]+\d", cell)
        assert found and found.group(1) == code

    @pytest.mark.parametrize("cell", ["# 139", "A 12", "H 5", "D 8", "2C", "13"])
    def test_a_letter_outside_the_legend_is_a_mangled_digit(self, cell):
        assert not re.match(rf"^\s*([{vision_part.STATUS_CODES}])[\s.]+\d", cell)

    def test_the_legend_is_exactly_the_five(self):
        assert set(vision_part.STATUS_CODES) == set("ESRMQ")


class TestStruckOffEntries:
    """A struck-off elector stays printed, numbered and counted -- and is not an elector.

    Part 1 publishes 873 boxes against a roll whose net is 850. Nothing read the DELETED
    watermark, so all 873 went out as live electors. Reconciliation cannot catch this: 873 is
    the number being reconciled *to*.
    """

    @pytest.mark.parametrize("token", ["DELETED", "DELETEO", "OELETED", "DELTED", "DELETED."])
    def test_a_damaged_stamp_is_still_the_stamp(self, token):
        """The stamp is hollow, diagonal and drawn over the elector's own text.

        Its glyphs are broken by whatever they cross, so a substring test would find only the
        clean reads -- and the damaged ones are not a random sample of the rows.
        """
        assert fields.is_struck_off([token])

    def test_it_is_found_among_the_electors_own_words(self):
        assert fields.is_struck_off("নাম : হাইনা বসুমাতাৰী DELETED".split())

    @pytest.mark.parametrize("token", ["HHK0001464", "QTD0043265"])
    def test_an_epic_is_not_mistaken_for_the_stamp(self, token):
        """The EPIC is the only other Latin string on these pages."""
        assert not fields.is_struck_off([token])

    def test_assamese_never_triggers_it(self):
        assert not fields.is_struck_off(["পুৰুষ", "মহিলা", "ফটো", "উপলব্ধ"])

    def test_short_noise_never_triggers_it(self):
        assert not fields.is_struck_off(["OA", "TEBE", "E", "15"])

    def test_deleted_is_a_column_not_a_flag(self):
        """Anyone counting electors has to exclude these, and flags are not where people look."""
        assert "deleted" in output.COLUMNS

    def test_the_roll_arithmetic_gives_an_independent_expected_count(self):
        """net = main + additions - struck off, from two numbers printed on different pages.

        The watermark is the one field with no other check on it. A detector that silently
        found nothing would look exactly like a roll with no deletions.
        """
        rows = [{"part_no": 4, "roll_section": "main", "deleted": False} for _ in range(483)] + [
            {"part_no": 4, "roll_section": "addition", "deleted": False} for _ in range(28)
        ]
        for row in rows[:14]:
            row["deleted"] = True
        found = run.struck_off_check(Path("."), rows, {4: 497})
        assert found["implied"] == 14 and found["detected"] == 14
        assert found["per_part"][0]["diff"] == 0

    def test_a_detector_finding_nothing_is_visible_not_silent(self):
        rows = [{"part_no": 1, "roll_section": "main", "deleted": False} for _ in range(873)]
        rows += [{"part_no": 1, "roll_section": "addition", "deleted": False}]
        found = run.struck_off_check(Path("."), rows, {1: 850})
        assert found["implied"] == 24 and found["detected"] == 0
        assert found["per_part"][0]["diff"] == -24

    def test_a_deletion_sheet_does_not_count_as_an_addition(self):
        rows = [{"part_no": 1, "roll_section": "main", "deleted": False} for _ in range(100)]
        rows += [{"part_no": 1, "roll_section": "deletion", "deleted": False} for _ in range(4)]

        found = run.struck_off_check(Path("."), rows, {1: 100})

        assert found["implied"] == 0


class TestPartialLastPage:
    """The last page of a part holds whatever is left, and it used to be thrown away.

    Part 18's page 25 is a single box -- serial 661, ৰাজেশ নুজ্জাৰী, age 25, plainly printed --
    and it was read as a closing summary and dropped. Four of the first twenty parts lost exactly
    one elector this way; part 6 lost nineteen. Nothing raised anything: the count was simply
    short against a printed total nobody was comparing it to.

    The fix is that column geometry belongs to the *part*, not the page. Real x positions below,
    from AC1 at 400 dpi.
    """

    #: The three columns AC1 draws, from pages that draw all three.
    ESTABLISHED = [(78, 854, 1082), (1133, 1909, 2137), (2189, 2965, 3193)]

    #: Part 18 page 25: one box in the first column, and nothing else on the sheet.
    ONE_COLUMN = [78, 855, 1080, 1108]

    #: Part 6 page 39: nineteen electors, with the *middle* divider not drawn.
    MIDDLE_DIVIDER_MISSING = [78, 854, 1082, 1108, 1133, 2135, 2164, 2189, 2968, 3192, 3220]

    #: Part 5 page 15: a three-elector addition list. The border it shares between columns two and
    #: three prints as 2164, 2189 -- the 2137 line, which is where the full pages put that edge,
    #: is simply not inked.
    SUPPLEMENT = [78, 854, 1080, 1108, 1133, 2164, 2189, 3192, 3220]

    #: Part 5 page 16: the real closing summary.
    REAL_SUMMARY = [67, 3236]

    #: AC24 part 95 page 4 has one extra vertical rule. Read consecutively, the clusters make
    #: text cells 856, 137 and 228 pixels wide even though all three printed columns are equal.
    STRAY_DIVIDER = [78, 934, 1162, 1189, 1326, 2009, 2237, 2264, 2289, 2971, 3199, 3225]

    def test_a_page_of_one_box_is_read_against_the_parts_geometry(self):
        found = grid.columns_present(self.ONE_COLUMN, self.ESTABLISHED)
        assert found == [(78, 854, 1082)]

    def test_a_missing_middle_divider_does_not_lose_the_page(self):
        """Part 3's page 23 lacks its last divider and part 6's page 39 its middle one.

        Both yield six clusters, so no positional reading of the cluster count is right for
        both -- which is why the edges are matched against the part instead of counted.
        """
        assert grid.columns_present(self.MIDDLE_DIVIDER_MISSING, self.ESTABLISHED) == (
            self.ESTABLISHED
        )

    def test_a_shared_border_may_be_inked_anywhere_across_its_width(self):
        """Part 5's page 15 held three electors and an exact edge test found two.

        Neighboring columns share a border drawn as two or three lines, and which of them the
        publisher inks varies by page. Demanding a rule within a few pixels of where the full
        pages put that edge dropped the middle column, and the elector on it.
        """
        assert len(grid.columns_present(self.SUPPLEMENT, self.ESTABLISHED)) == 3

    def test_a_supplement_page_lost_here_would_still_reconcile(self):
        """Why this cannot be left to the printed total to catch.

        Part 5's page 15 is an addition list. Supplements are counted separately from the
        closing মূল তালিকা total, so losing it left the main list balancing at 351 while three
        real electors -- serials 352, 353, 354 -- were missing and nothing said so.
        """
        assert grid.columns_present(self.SUPPLEMENT, self.ESTABLISHED) != []

    def test_the_real_closing_summary_is_still_a_summary(self):
        assert grid.columns_present(self.REAL_SUMMARY, self.ESTABLISHED) == []

    def test_columns_must_be_a_prefix_because_boxes_fill_in_reading_order(self):
        """A page can hold columns 1, or 1-2, or 1-2-3 -- never 2 and 3 with 1 empty."""
        assert grid.columns_present([1133, 2137, 2189, 3193], self.ESTABLISHED) == []

    def test_the_page_alone_cannot_settle_it(self):
        """Without the part's geometry these rules are a table, and are read as one."""
        assert grid.column_triples(self.ONE_COLUMN) == []

    def test_the_closing_summary_is_not_promoted_to_an_elector_page(self):
        """The check that keeps this safe: a summary's rules are not on the column edges.

        Getting this wrong costs the part its printed total, which is the only number in the
        source that the extraction is measured against.
        """
        assert grid.columns_present([120, 300, 1600, 2400], self.ESTABLISHED) == []

    def test_a_page_with_no_vertical_rules_yields_nothing(self):
        assert grid.columns_present([], self.ESTABLISHED) == []

    def test_geometry_is_taken_only_from_pages_that_drew_all_three(self):
        partial = [(78, 854, 1082)]
        assert grid.typical_columns([self.ESTABLISHED, partial, self.ESTABLISHED]) == (
            self.ESTABLISHED
        )

    def test_a_part_with_no_full_page_establishes_nothing(self):
        """Refused rather than extrapolated from one column."""
        assert grid.typical_columns([[(78, 854, 1082)]]) == []

    def test_build_uses_the_given_columns_over_the_pages_own_rules(self):
        boxes = grid.build(H_RULES, self.ONE_COLUMN, columns=[(78, 854, 1082)])
        assert boxes and {b.column for b in boxes} == {0}
        assert boxes[0].left == 78 and boxes[0].text_right == 854

    def test_recovery_only_touches_pages_that_were_not_elector_pages(self):
        good = pages.PageSignature(
            number=3, kind=pages.PageKind.ELECTOR, h_rules=H_RULES, v_rules=V_RULES
        )
        last = pages.PageSignature(
            number=4, kind=pages.PageKind.SUMMARY, h_rules=H_RULES, v_rules=self.ONE_COLUMN
        )
        before, after = pages.recover_partial([good, last])
        assert before is good
        # All of the part's columns, not only the ones whose edges happened to be inked: which
        # of them hold an elector is settled by ink, in grid.has_ink.
        assert after.kind is pages.PageKind.ELECTOR
        assert after.columns == tuple(self.ESTABLISHED)

    def test_an_elector_page_with_impossible_column_widths_uses_part_geometry(self):
        bad = pages.PageSignature(
            number=4,
            kind=pages.PageKind.ELECTOR,
            h_rules=H_RULES,
            v_rules=self.STRAY_DIVIDER,
        )
        good = pages.PageSignature(
            number=3, kind=pages.PageKind.ELECTOR, h_rules=H_RULES, v_rules=V_RULES
        )

        recovered = pages.recover_partial([good, bad])[1]

        assert not grid.balanced_columns(grid.column_triples(self.STRAY_DIVIDER))
        assert recovered.columns == tuple(self.ESTABLISHED)

    def test_every_column_is_built_even_when_one_edge_was_not_inked(self):
        """Part 43's page 25 holds five electors and lost serial 663 to a missing edge line."""
        page25 = [78, 854, 1082, 1108, 1133, 1909, 2137, 2164]
        good = pages.PageSignature(
            number=3, kind=pages.PageKind.ELECTOR, h_rules=H_RULES, v_rules=V_RULES
        )
        last = pages.PageSignature(
            number=4, kind=pages.PageKind.SUMMARY, h_rules=H_RULES, v_rules=page25
        )
        assert len(grid.columns_present(page25, self.ESTABLISHED)) == 2
        assert len(pages.recover_partial([good, last])[1].columns) == 3

    def test_one_row_page_is_recovered_from_its_part_geometry(self):
        """Two horizontal rules are a complete one-row page, not insufficient evidence."""
        good = pages.PageSignature(
            number=3, kind=pages.PageKind.ELECTOR, h_rules=H_RULES, v_rules=V_RULES
        )
        sparse = pages.PageSignature(
            number=18,
            kind=pages.PageKind.SUMMARY,
            h_rules=(153, 568),
            v_rules=self.ONE_COLUMN,
        )

        recovered = pages.recover_partial([good, sparse])[1]

        assert recovered.kind is pages.PageKind.ELECTOR
        assert grid.build(recovered.h_rules, recovered.v_rules, recovered.columns)


def test_an_explicit_english_summary_title_is_not_an_elector_page():
    assert summary.is_summary("SUMMARY OF ELECTORS A) NUMBER OF ELECTORS")
    assert not summary.is_summary("Assembly Constituency No and Name: 113 - HAFLONG")


def test_the_direct_tesseract_reader_does_not_emit_an_explicit_summary(tmp_path, monkeypatch):
    class Engine:
        def _run(self, _image, _lang, _config, _scale, *, psm):
            assert psm == 6
            return "SUMMARY OF ELECTORS A) NUMBER OF ELECTORS"

    def rendered(_pdf, workdir, dpi=extract.DPI):
        from PIL import Image

        page = workdir / "page-1.png"
        Image.new("L", (400, 500), "white").save(page)
        return [page]

    signature = pages.PageSignature(1, pages.PageKind.ELECTOR, (), ())
    box = grid.Box(0, 0, 10, 100, 390, 500, 300)
    monkeypatch.setattr(extract.render, "read_pdf_bytes", lambda *_args: b"pdf")
    monkeypatch.setattr(extract, "_render_pages", rendered)
    monkeypatch.setattr(pages, "classify", lambda _image, _number: signature)
    monkeypatch.setattr(grid, "build", lambda *_args, **_kwargs: [box])
    monkeypatch.setattr(
        summary,
        "read",
        lambda *_args: summary.RollSummary(male=10, female=10, third=0, total=20, scale=1),
    )
    monkeypatch.setattr(
        fields,
        "read_page",
        lambda *_args: pytest.fail("a closing summary must not reach the elector reader"),
    )

    result = extract.read_part(
        tmp_path / "AC113_ENG.zip",
        "2026-EROLLGEN-S03-113-FinalRoll-Revision1-ENG-1-WI.pdf",
        engine=Engine(),
    )

    assert not result.electors
    assert result.summary_total == 20


class TestRunnerAcrossConstituencies:
    """The AC number must be the only thing that changes between runs.

    Everything here is the plumbing that decides whether AC6-AC10 slot in behind AC1-AC5 or
    quietly overwrite them. It is testable without a second zip, which is the point -- the
    failure would be a shard filed under the wrong constituency, and that is not the kind of
    thing to discover after paying for the reading.
    """

    @pytest.mark.parametrize(
        "name,expected",
        [("AC1_ASM.zip", 1), ("AC17_ASM.zip", 17), ("AC126_ASM.zip", 126)],
    )
    def test_the_ac_number_comes_from_the_archive_name(self, name, expected):
        assert run._ac_number(Path("data/ac_rolls") / name) == expected

    def test_an_archive_with_no_number_is_refused_not_guessed(self):
        with pytest.raises(ValueError, match="cannot read an AC number"):
            run._ac_number(Path("rolls.zip"))

    def test_the_manifest_accumulates_rather_than_replacing(self, tmp_path):
        """Five constituencies must leave five entries, not the last one."""
        manifest = tmp_path / "manifest.json"
        for ac_no in (6, 7, 8):
            shard = output.write_shard([], ac_no, directory=tmp_path)
            output.update_manifest(ac_no, shard, {"rows": ac_no * 10}, manifest=manifest)
        entries = json.loads(manifest.read_text(encoding="utf-8"))
        assert sorted(entries) == ["6", "7", "8"]
        assert entries["7"]["rows"] == 70

    def test_rerunning_one_ac_replaces_only_its_own_entry(self, tmp_path):
        manifest = tmp_path / "manifest.json"
        for ac_no in (6, 7):
            shard = output.write_shard([], ac_no, directory=tmp_path)
            output.update_manifest(ac_no, shard, {"rows": 1}, manifest=manifest)
        output.update_manifest(7, output.shard_path(7, tmp_path), {"rows": 999}, manifest=manifest)
        entries = json.loads(manifest.read_text(encoding="utf-8"))
        assert entries["6"]["rows"] == 1 and entries["7"]["rows"] == 999

    def test_cached_rows_are_reused_instead_of_re_read(self, tmp_path):
        """A part already read is a part already paid for."""
        part_dir = tmp_path / "part0003"
        part_dir.mkdir()
        (part_dir / run.ROWS).write_text(
            json.dumps(
                {
                    "part_no": 3,
                    "ac_no": 1,
                    "roll_section": "main",
                    "pipeline_version": extract.PIPELINE_VERSION,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        billed = []
        rows = run.read(tmp_path, "", on_part=lambda n, c, b: billed.append(b))
        assert len(rows) == 1 and billed == [0]

    def test_reconciliation_separates_unmeasured_from_failing(self, tmp_path):
        """A part whose closing page could not be read is not a part that passed."""
        for number, closing in ((1, 2), (2, None)):
            part_dir = tmp_path / f"part{number:04d}"
            part_dir.mkdir()
            side = {"pages": {}, "unknown": [], "summary": None}
            if closing is not None:
                side["summary"] = summary.RollSummary(
                    male=1, female=1, third=0, total=closing, scale=400
                )
            stage1.write_side(part_dir / "side.json", side)
        rows = [
            {"part_no": 1, "roll_section": "main"},
            {"part_no": 1, "roll_section": "main"},
            {"part_no": 2, "roll_section": "main"},
        ]
        found = run.reconcile(tmp_path, rows)
        assert found["measured"] == 1 and found["matching"] == 1
        assert found["residuals"] == []

    def test_supplement_rows_are_not_scored_against_the_main_total(self, tmp_path):
        """Part 4 read 511 boxes against a printed 483; the 28 were its supplement."""
        part_dir = tmp_path / "part0004"
        part_dir.mkdir()
        stage1.write_side(
            part_dir / "side.json",
            {
                "pages": {},
                "unknown": [],
                "summary": summary.RollSummary(male=240, female=243, third=0, total=483, scale=400),
            },
        )
        rows = [{"part_no": 4, "roll_section": "main"} for _ in range(483)]
        rows += [{"part_no": 4, "roll_section": "addition"} for _ in range(28)]
        found = run.reconcile(tmp_path, rows)
        assert found["matching"] == 1 and found["residuals"] == []


class TestBoughtSummaryPersists:
    """A closing total bought from Vision has to survive the process that bought it.

    Reconciliation reads the total from side.json, not from the result object stage two builds.
    So the first version of the fallback read 37 pages of AC100 correctly, at real cost, and
    reported 80% of the constituency unmeasured anyway -- the purchase was made and discarded.
    """

    def test_a_rewrite_keeps_the_version_that_produced_the_output(self, tmp_path):
        """Stage two rewrites this file, and must not bless stale stage-one output as fresh."""
        stage1.write_side(tmp_path / "side.json", {"pages": {}, "unknown": [], "summary": None})
        old = stage1.read_side(tmp_path / "side.json")
        old["stage1_version"] = "1.9.0"
        stage1.write_side(tmp_path / "side.json", old)
        assert json.loads((tmp_path / "side.json").read_text())["stage1_version"] == "1.9.0"

    def test_a_fresh_write_stamps_the_current_version(self, tmp_path):
        stage1.write_side(tmp_path / "side.json", {"pages": {}, "unknown": [], "summary": None})
        stamped = json.loads((tmp_path / "side.json").read_text())["stage1_version"]
        assert stamped == stage1.STAGE1_VERSION

    def test_the_bought_total_is_on_disk_for_reconciliation_to_find(self, tmp_path):
        side = {"pages": {}, "unknown": [], "summary": None}
        stage1.write_side(tmp_path / "side.json", side)
        side = stage1.read_side(tmp_path / "side.json")
        side["summary"] = summary.RollSummary(male=435, female=477, third=0, total=912, scale=1)
        stage1.write_side(tmp_path / "side.json", side)
        assert stage1.read_side(tmp_path / "side.json")["summary"].total == 912

    def test_cached_summary_words_do_not_need_the_old_image(self, tmp_path):
        """The reparse tool keeps paid words and omits the image they already describe."""
        lines = [
            "Mother Roll Basic Roll 361 328 0 689",
            "Net Elector in this Roll 354 322 0 676",
        ]
        words = []
        for row, line in enumerate(lines):
            left = 0
            for token in line.split():
                right = left + len(token) * 10
                words.append(vision.Word(token, left, row * 40, right, row * 40 + 20))
                left = right + 8
        stage2.save_words(tmp_path, "summary_page.png", words)

        found, paid = stage2.summary_from_vision(tmp_path, "")

        assert found is not None and found.total == 689
        assert not paid


class TestStageOneSideReadReuse:
    def test_only_identical_source_bytes_reuse_box_headers(self, tmp_path, monkeypatch):
        (tmp_path / "side.json").write_text("{}")
        key = (3, 0, 0)
        read = {"epic": "ABC1234567", "header_geometry": [[10, 20, 30, 40]]}
        monkeypatch.setattr(
            crops,
            "read_manifest",
            lambda _part: {
                "box": {
                    "pdf_sha256": "same",
                    "page_no": key[0],
                    "box_row": key[1],
                    "box_col": key[2],
                }
            },
        )
        monkeypatch.setattr(
            stage1,
            "read_side",
            lambda _path: {
                "header_reader_version": vision_part.HEADER_READER_VERSION,
                "pages": {3: {"boxes": {key: read}}},
            },
        )

        assert stage1._prior_box_reads(tmp_path, "same") == {key: read}
        assert stage1._prior_box_reads(tmp_path, "different") == {}

    def test_a_different_header_reader_version_forces_fresh_reads(self, tmp_path, monkeypatch):
        (tmp_path / "side.json").write_text("{}")
        key = (3, 0, 0)
        monkeypatch.setattr(
            crops,
            "read_manifest",
            lambda _part: {"box": {"pdf_sha256": "same"}},
        )
        monkeypatch.setattr(
            stage1,
            "read_side",
            lambda _path: {
                "header_reader_version": "older",
                "pages": {3: {"boxes": {key: {"epic": "stale"}}}},
            },
        )

        assert stage1._prior_box_reads(tmp_path, "same") == {}

    def test_a_changed_rectangle_forces_a_fresh_header_read(self, monkeypatch):
        key = (3, 0, 0)
        monkeypatch.setattr(vision_part, "_header_of", lambda _image, _box: {"epic": "fresh"})
        monkeypatch.setattr(
            vision_part, "_header_geometry", lambda _image, _box: [[10, 20, 31, 40]]
        )

        found = vision_part._box_header(
            object(),
            object(),
            key,
            {key: {"epic": "stale", "header_geometry": [[10, 20, 30, 40]]}},
        )

        assert found == {"epic": "fresh"}

    def test_an_identical_rectangle_reuses_the_header_read(self, monkeypatch):
        key = (3, 0, 0)
        geometry = [[10, 20, 30, 40]]
        monkeypatch.setattr(vision_part, "_header_geometry", lambda _image, _box: geometry)
        monkeypatch.setattr(
            vision_part,
            "_header_of",
            lambda _image, _box: pytest.fail("identical geometry must use its cached read"),
        )

        found = vision_part._box_header(
            object(),
            object(),
            key,
            {key: {"epic": "cached", "header_geometry": geometry}},
        )

        assert found == {"epic": "cached", "header_geometry": geometry}

    def test_header_geometry_changes_when_only_the_outer_box_changes(self):
        from PIL import Image

        image = Image.new("L", (4000, 5000), "white")
        first = grid.Box(0, 0, 100, 200, 1100, 600, 850)
        wider = grid.Box(0, 0, 100, 200, 1200, 600, 850)

        assert vision_part._header_geometry(image, first) != vision_part._header_geometry(
            image, wider
        )


class TestStageHandoff:
    """What stage one writes must be exactly what stage two needs -- no re-rendering.

    The split only saves anything if stage two can work from the files alone. Anything the
    handoff drops, stage two has to recover by opening the PDF again, which is the cost the
    split exists to avoid.
    """

    def _side(self):
        return {
            "header_reader_version": vision_part.HEADER_READER_VERSION,
            "pages": {
                3: {
                    "section": pages.Section.MAIN,
                    "recognised": True,
                    "boxes": {(3, 0, 0): "HHK0001471", (3, 0, 1): "HHK0001472"},
                },
                4: {"section": pages.Section.ADDITION, "recognised": False, "boxes": {}},
            },
            "unknown": [7],
            "summary": summary.RollSummary(male=315, female=302, third=1, total=618, scale=400),
        }

    def test_side_survives_the_round_trip_whole(self, tmp_path):
        stage1.write_side(tmp_path / "side.json", self._side())
        back = stage1.read_side(tmp_path / "side.json")
        assert {k: v for k, v in back.items() if k != "stage1_version"} == self._side()

    def test_the_side_read_records_which_stage_one_wrote_it(self, tmp_path):
        """Resume is keyed on this, so a fix invalidates its own stale output."""
        stage1.write_side(tmp_path / "side.json", self._side())
        assert stage1.read_side(tmp_path / "side.json")["stage1_version"] == stage1.STAGE1_VERSION

    def test_a_part_prepared_by_older_code_is_not_treated_as_done(self, tmp_path):
        """The failure this prevents: a rerun after a fix serving what the fix replaced.

        It happened twice in one evening -- once from composites left on disk, once from the
        same output pulled back out of a bucket by a fresh machine -- and both times the only
        thing that caught it was a human reading a log line.
        """
        part_dir = tmp_path / "part0007"
        part_dir.mkdir()
        (part_dir / "placements.json").write_text('{"composite000.png": []}', encoding="utf-8")
        crops.write_manifest(
            [crops.Crop(part=7, page=1, row=0, column=0, path=part_dir / "c.png")],
            part_dir,
            append=False,
        )
        stage1.write_side(part_dir / "side.json", self._side())
        (part_dir / "composite000.png").write_bytes(b"x")
        assert stage1.done(tmp_path, 7)

        stale = json.loads((part_dir / "side.json").read_text(encoding="utf-8"))
        stale["stage1_version"] = "1.0.0"
        (part_dir / "side.json").write_text(json.dumps(stale), encoding="utf-8")
        assert not stage1.done(tmp_path, 7)

    def test_metadata_with_nothing_readable_is_not_done(self, tmp_path):
        """The shape AC101 took when it shipped 113 of its 210 parts and reported no error.

        Composites are not synced to the bucket -- they are cheap to rebuild and would be a
        terabyte for the state -- so a machine resuming from the bucket gets a part's metadata
        without its images. Every other test passed, the part was skipped as prepared, and
        stage two skipped it again for having nothing to read. done() and stage2.ready() must
        not disagree about what a finished part is.

        What makes a part readable is pixels *or* cached words, and the third state is the common
        one -- see ``tests/test_resume.py``. Here only the empty-handed case is pinned.
        """
        part_dir = tmp_path / "part0009"
        part_dir.mkdir()
        (part_dir / "placements.json").write_text('{"composite000.png": []}', encoding="utf-8")
        crops.write_manifest(
            [crops.Crop(part=9, page=1, row=0, column=0, path=part_dir / "c.png")],
            part_dir,
            append=False,
        )
        stage1.write_side(part_dir / "side.json", self._side())
        assert not stage1.done(tmp_path, 9)
        (part_dir / "composite000.png").write_bytes(b"x")
        assert stage1.done(tmp_path, 9)

    def test_an_unreadable_part_stops_the_run_rather_than_being_skipped(self, tmp_path):
        """A bare continue is what let 97 missing parts look like success."""
        part_dir = tmp_path / "part0003"
        part_dir.mkdir()
        (part_dir / "placements.json").write_text("{}", encoding="utf-8")
        crops.write_manifest(
            [crops.Crop(part=3, page=1, row=0, column=0, path=part_dir / "c.png")],
            part_dir,
            append=False,
        )
        with pytest.raises(RuntimeError, match="no composites to read"):
            run.read(tmp_path, "")

    def test_a_part_with_no_side_read_at_all_is_not_done(self, tmp_path):
        part_dir = tmp_path / "part0008"
        part_dir.mkdir()
        (part_dir / "placements.json").write_text("{}", encoding="utf-8")
        crops.write_manifest(
            [crops.Crop(part=8, page=1, row=0, column=0, path=part_dir / "c.png")],
            part_dir,
            append=False,
        )
        assert not stage1.done(tmp_path, 8)

    def test_the_sex_breakdown_is_not_flattened_to_a_total(self, tmp_path):
        """An earlier version kept only the total, so stage two could not report male/female."""
        stage1.write_side(tmp_path / "side.json", self._side())
        closing = stage1.read_side(tmp_path / "side.json")["summary"]
        assert (closing.male, closing.female, closing.third) == (315, 302, 1)

    def test_sections_come_back_as_the_enum_not_a_string(self, tmp_path):
        stage1.write_side(tmp_path / "side.json", self._side())
        back = stage1.read_side(tmp_path / "side.json")
        assert back["pages"][4]["section"] is pages.Section.ADDITION

    def test_placements_rebuild_with_the_keys_they_were_written_with(self, tmp_path):
        (tmp_path / "placements.json").write_text(
            json.dumps(
                {
                    "composite000.png": [
                        {
                            "page": 3,
                            "box_row": 0,
                            "box_col": 1,
                            "left": 800,
                            "top": 0,
                            "right": 1576,
                            "bottom": 300,
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        placed = stage2.placements_of(tmp_path)["composite000.png"]
        assert [p.key for p in placed] == [(3, 0, 1)]
        assert (placed[0].left, placed[0].bottom) == (800, 300)

    def test_identity_is_read_from_the_manifest_not_the_directory_name(self, tmp_path):
        """A directory can be renamed; the sha that ties a row to source bytes cannot."""
        crops.write_manifest(
            [
                crops.Crop(
                    part=9,
                    page=1,
                    row=0,
                    column=0,
                    path=tmp_path / "composite000.png",
                    ac_no=42,
                    source_zip="AC42_ASM.zip",
                    source_pdf="2026-EROLLGEN-S03-42-FinalRoll-Revision1-ASM-9-WI.pdf",
                    pdf_sha256="c" * 64,
                )
            ],
            tmp_path,
            append=False,
        )
        who = stage2.identity(tmp_path)
        assert who["ac_no"] == 42 and who["part_no"] == 9
        assert who["source_zip"] == "AC42_ASM.zip" and who["pdf_sha256"] == "c" * 64


class TestManifestKeys:
    """The manifest key is where the box is, never which file it landed in.

    These were the same thing while every crop was its own PNG. Packed many to a composite they
    are not: 873 tiles all carried the stem ``composite000``, keyed down to 3 rows, and 870
    electors would have vanished between the stages without a word.
    """

    @staticmethod
    def _crop(row, column, image="composite000.png"):
        return crops.Crop(
            part=1,
            page=3,
            row=row,
            column=column,
            path=Path("/x") / image,
            left=0,
            top=0,
            right=10,
            bottom=10,
            ac_no=1,
            section="main",
        )

    def test_tiles_sharing_one_composite_keep_distinct_keys(self):
        a, b = self._crop(0, 0), self._crop(0, 1)
        assert a.name != b.name
        assert a.name == "p1_pg3_r0c0" and b.name == "p1_pg3_r0c1"

    def test_the_row_still_records_which_image_holds_it(self):
        assert self._crop(2, 1).as_row()["image"] == "composite000.png"

    def test_a_duplicate_key_is_raised_not_overwritten(self, tmp_path):
        """Keeping the last row silently is exactly how 873 tiles became 3."""
        crops.write_manifest([self._crop(0, 0), self._crop(0, 0)], tmp_path, append=False)
        with pytest.raises(ValueError, match="duplicate manifest key"):
            crops.read_manifest(tmp_path)

    def test_a_manifest_of_distinct_tiles_reads_back_whole(self, tmp_path):
        made = [self._crop(r, c) for r in range(10) for c in range(3)]
        crops.write_manifest(made, tmp_path, append=False)
        assert len(crops.read_manifest(tmp_path)) == 30


class TestRepack:
    """Tiles packed to cut the bill, and words found again afterwards.

    Vision bills per image and 70% of a page is white space, borders and empty photo boxes. The
    danger is not that a tighter image reads worse -- it is that a word comes back attributed to
    the wrong box, which is a plausible row with nothing raised.
    """

    @staticmethod
    def _tiles(n, w=776, h=200):
        from PIL import Image

        page = Image.new("L", (w, h * n), "white")
        return [(page, (0, i * h, w, (i + 1) * h), (3, i, 0)) for i in range(n)]

    def test_every_tile_gets_a_rectangle_in_the_composite(self):
        image, placed = repack.compose(self._tiles(7), columns=3)
        assert len(placed) == 7
        assert image.width == 3 * 776 + 2 * repack.GUTTER
        for tile in placed:
            assert tile.width == 776 and tile.height == 200

    def test_a_word_comes_back_to_the_tile_it_was_placed_in(self):
        """The page-offset bug one level down: unshifted, a word lands in whichever tile sits at
        that height and the row is wrong with nothing to show for it."""
        image, placed = repack.compose(self._tiles(9), columns=3)
        for tile in placed:
            middle = vision.Word("x", tile.left + 5, tile.top + 5, tile.left + 40, tile.top + 40)
            found = repack.words_for(placed, [middle])
            assert [k for k, v in found.items() if v] == [tile.key], f"{tile.key} misfiled"

    def test_words_are_matched_horizontally_as_well_as_vertically(self):
        """Tiles sit side by side. Filtering on height alone hands a word to all three tiles in
        its row, and each answers with somebody else's name."""
        image, placed = repack.compose(self._tiles(3), columns=3)
        first, second, third = placed
        word = vision.Word(
            "x", second.left + 10, second.top + 10, second.left + 60, second.top + 50
        )
        found = repack.words_for(placed, [word])
        assert len(found[second.key]) == 1
        assert not found[first.key] and not found[third.key]

    def test_a_word_returns_in_its_own_tile_coordinates(self):
        """The parser downstream expects a box starting at zero, not at wherever it was pasted."""
        image, placed = repack.compose(self._tiles(6), columns=3)
        tile = placed[4]
        word = vision.Word("x", tile.left + 12, tile.top + 30, tile.left + 90, tile.top + 70)
        back = repack.words_for(placed, [word])[tile.key][0]
        assert (back.left, back.top, back.right, back.bottom) == (12, 30, 90, 70)

    def test_a_word_in_a_gutter_belongs_to_nobody(self):
        image, placed = repack.compose(self._tiles(6), columns=3)
        first = placed[0]
        stray = vision.Word("x", first.right + 4, first.top + 10, first.right + 18, first.top + 40)
        assert not any(repack.words_for(placed, [stray]).values())

    def test_the_last_row_may_be_short(self):
        image, placed = repack.compose(self._tiles(4), columns=3)
        assert len(placed) == 4
        assert placed[3].top > placed[0].top, "the fourth tile starts a second row"

    def test_every_composite_stays_under_the_pixel_ceiling(self):
        """A composite over the ceiling is a rejected request, not a slow one."""
        entries = self._tiles(500)
        budget = 20_000_000
        made = repack.batches(entries, budget)
        assert len(made) > 1
        for batch in made:
            assert repack.pixels(batch) <= budget

    def test_batching_grows_a_tile_at_a_time_rather_than_estimating(self):
        """A row takes the height of its tallest tile, so the cost of one more tile is not
        constant -- the one that starts a row adds its whole height."""
        entries = self._tiles(30)
        total = sum(len(b) for b in repack.batches(entries, 20_000_000))
        assert total == 30, "every tile must land in exactly one composite"

    def test_a_tile_too_large_to_fit_is_still_submitted(self):
        """Dropping it would lose those electors with nothing raised."""
        entries = self._tiles(3)
        made = repack.batches(entries, 1)
        assert sum(len(b) for b in made) == 3

    def test_the_saving_can_be_costed_before_the_image_is_built(self):
        """A composite over the pixel ceiling is a lost request, so the size is known first."""
        entries = self._tiles(30)
        assert repack.pixels(entries) > 0
        image, _ = repack.compose(entries)
        assert repack.pixels(entries) == image.width * image.height


class TestCropExport:
    """Box images cut for a reader that runs on someone else's GPU."""

    def test_the_filename_is_the_key(self):
        """A crop folder and a JSON of readings line up with nothing else between them. A
        separate manifest is one more thing that can fall out of step with the images."""
        assert crops.name_for(13, 4, 2, 0) == "p13_pg4_r2c0"
        assert crops.key_of("p13_pg4_r2c0.png") == (13, 4, 2, 0)
        assert crops.key_of("p13_pg4_r2c0") == (13, 4, 2, 0)

    def test_the_key_spelling_matches_the_eval_set_and_the_comparison(self):
        """Three places parse this shape. If they ever disagree the comparison silently
        compares nothing, which is exactly what an empty intersection looks like."""
        name = crops.name_for(7, 5, 9, 2)
        assert resolution.KEY.match(name), "resolution must parse what crops writes"
        assert crops.key_of(name) == (7, 5, 9, 2)

    def test_a_name_that_is_not_a_key_is_refused_rather_than_guessed(self):
        assert crops.key_of("random.png") is None
        assert crops.key_of("IMG_1234.png") is None

    def test_a_band_crop_keeps_the_key_and_adds_the_line(self):
        """Several bands belong to one elector, so the band must not be part of the key --
        that is what joins them back to a single row."""
        name = crops.name_for(13, 4, 2, 0, band=0)
        assert name == "p13_pg4_r2c0_b0"
        assert crops.key_of(name) == (13, 4, 2, 0)
        assert crops.band_of(name) == 0
        assert crops.band_of("p13_pg4_r2c0") is None

    def test_a_band_is_padded_past_its_ink(self):
        """grid.text_bands pads 5px, which is enough for tesseract and not for handing a line
        to a model alone: at 400 dpi it clips the head of a matra and the tail of a descender.
        বিনয় lost its ি and অৰ্জুন lost its hanger, read as different names."""
        box = grid.Box(row=0, column=0, left=0, top=0, right=800, bottom=400, text_right=776)
        bands = [(100, 148), (200, 248), (300, 348)]
        top, bottom = crops.band_window(bands, 1, box)
        assert top < 200 and bottom > 248, "must reach past the ink on both sides"

    def test_padding_never_crosses_into_the_neighbouring_line(self):
        """Padded far enough to catch a matra, the relation line arrives under the name and a
        model handed two lines may answer with either."""
        box = grid.Box(row=0, column=0, left=0, top=0, right=800, bottom=400, text_right=776)
        bands = [(100, 148), (200, 248), (300, 348)]
        top, bottom = crops.band_window(bands, 1, box)
        assert top >= (148 + 200) // 2, "must not reach into the band above"
        assert bottom <= (248 + 300) // 2, "must not reach into the band below"

    def test_the_outer_bands_are_clamped_to_the_box(self):
        box = grid.Box(row=0, column=0, left=0, top=90, right=800, bottom=360, text_right=776)
        bands = [(100, 148), (300, 348)]
        assert crops.band_window(bands, 0, box)[0] >= box.top
        assert crops.band_window(bands, 1, box)[1] <= box.bottom

    def test_the_manifest_round_trips_and_carries_what_the_filename_cannot(self, tmp_path):
        """A filename identifies a box but does not say which roll section contains it."""
        crop = crops.Crop(
            part=13,
            page=4,
            row=2,
            column=0,
            path=tmp_path / "p13_pg4_r2c0_b0.png",
            band=0,
            left=78,
            top=228,
            right=854,
            bottom=299,
            ac_no=1,
            section="supplement",
            source_pdf="x.pdf",
            pdf_sha256="abc123",
        )
        crops.write_manifest([crop], tmp_path, append=False)
        back = crops.read_manifest(tmp_path)
        assert set(back) == {"p13_pg4_r2c0_b0"}
        row = back["p13_pg4_r2c0_b0"]
        assert row["roll_section"] == "supplement"
        assert (row["left"], row["top"], row["right"], row["bottom"]) == (78, 228, 854, 299)
        assert row["pdf_sha256"] == "abc123" and row["ac_no"] == 1

    def test_readings_become_arms_through_the_same_parser_vision_uses(self):
        """Comparing two engines through two parsers measures the parsers. This is the bug
        that put the elector's own name in the relation field, and it recurred three times."""
        readings = {
            "p13_pg4_r0c0.png": "নাম : খাদৰাম ৰাভা\nঘৰ নং : 20\nবয়স : 45 লিঙ্গ : পুৰুষ",
            "p13_pg4_r1c0.png": "নাম : গংগাৰাম\nবয়স : 30 লিঙ্গ : মহিলা",
        }
        arms = crops.readings_to_arm(readings)
        assert len(arms) == 1 and arms[0].part == 13
        assert arms[0].boxes[(4, 0, 0)] == {
            "name": "খাদৰাম ৰাভা",
            "age": 45,
            "house": "20",
            "sex": "M",
        }
        assert arms[0].boxes[(4, 1, 0)]["sex"] == "F"

    def test_readings_from_different_parts_become_different_arms(self):
        """Box positions repeat in every part, so one arm per part is what keeps them apart."""
        readings = {"p7_pg1_r0c0.png": "নাম : ক", "p64_pg1_r0c0.png": "নাম : খ"}
        assert sorted(a.part for a in crops.readings_to_arm(readings)) == [7, 64]

    def test_an_unreadable_reading_still_produces_a_box(self):
        """A box the second reader could not read is evidence about the second reader. Dropping
        it would quietly shrink the denominator instead."""
        arms = crops.readings_to_arm({"p1_pg1_r0c0.png": ""})
        assert arms[0].boxes[(1, 0, 0)] == {"name": "", "age": None, "house": "", "sex": ""}


class TestVisionPipelineMatchesWhatWasScored:
    """The pipeline must build electors the same way the accuracy was measured.

    Vision's 72% exact names came from ``second_pass``, which reads the printed labels. The
    tesseract path uses ``assign_bands``, which infers what a line is from its position among
    the others. Both are reasonable and they do not agree, so routing the pipeline through the
    second one would publish a number nobody measured -- the same class of error as scoring two
    engines at different token budgets, which reversed a ranking here once already.
    """

    @staticmethod
    def _rows():
        import json
        from pathlib import Path

        directory = Path(__file__).resolve().parent.parent / "dataset" / "eval"
        sample = json.loads((directory / "sample.json").read_text(encoding="utf-8"))
        return [row for row in sample if row.get("vision")]

    def test_the_builder_agrees_with_the_scorer_on_every_box(self):
        rows = self._rows()
        assert rows, "dataset/eval carries no Vision output to check against"
        for row in rows:
            elector = vision.elector_from([], row["vision"].split("\n"))
            built = {
                "name": elector.name,
                "age": elector.age,
                "house": elector.house_no,
                "sex": elector.sex,
            }
            assert built == evaluate.terse_fields(row["vision"]), row["key"]


class TestParserScriptAndLines:
    """Two parser bugs the dots.ocr run exposed, both silent."""

    def test_a_house_number_does_not_swallow_the_next_line(self):
        """`\\s*` before the optional suffix crossed the newline and took the ব of বয়স.

        House 13 came out as "13\\nব" on every terse answer, and read as a wrong value rather
        than a missing one.
        """
        found = second_pass.Reading("ঘৰ নং : 13\nবয়স : 22 লিঙ্গ : পুৰুষ").fields()
        assert found["house_no"] == "13"

    def test_a_real_suffix_on_the_same_line_survives(self):
        assert second_pass.Reading("ঘৰ নং : ২০ ক").fields()["house_no"] == "20 ক"

    def test_labels_are_matched_with_either_form_of_ra(self):
        """dots.ocr writes ঘর নং where the page prints ঘৰ নং -- one letter, two codepoints.

        A pattern knowing only the Assamese form found no house number at all on those rows.
        """
        assert second_pass.Reading("ঘর নং : ৮").fields()["house_no"] == "8"
        assert second_pass.Reading("ঘৰ নং : ৮").fields()["house_no"] == "8"

    def test_a_relation_label_with_either_ra_still_names_the_relative(self):
        assert second_pass.Reading("পিতার নাম: ৰাসু").fields()["relation_type"] == "father"
        assert second_pass.Reading("পিতাৰ নাম: ৰাসু").fields()["relation_type"] == "father"

    def test_a_stray_matra_on_the_name_label_does_not_join_the_name(self):
        """Cloud Vision reads the printed নাম as নামু on a third of boxes.

        Unmatched, the label fell through to the unlabelled-name branch and was recorded as part
        of the elector's name. Ten names Vision had read perfectly scored wrong, which is the
        whole of the gap between Vision at 44% exact names and Vision at 72%.
        """
        assert second_pass.Reading("নামু : অঙেলা মুছাহাৰী").fields()["name"] == "অঙেলা মুছাহাৰী"
        assert second_pass.Reading("নাম : অঙেলা মুছাহাৰী").fields()["name"] == "অঙেলা মুছাহাৰী"

    def test_a_relative_keeps_the_relation_field_even_with_a_stray_matra(self):
        """The tolerance must not turn the father's name into the elector's.

        The negative lookbehinds this replaced were fixed width, so they could not have been
        widened to allow the matra -- which is why the prefix is matched rather than excluded.
        """
        found = second_pass.Reading("পিতাৰ নামু : ৰাসু বসুমতাৰী").fields()
        assert found["relation_name"] == "ৰাসু বসুমতাৰী"
        assert found["relation_type"] == "father"
        assert "name" not in found

    def test_the_elector_is_named_when_both_labels_are_present(self):
        found = second_pass.Reading("নামু : অলনি\nপিতাৰ নাম : ৰাসু").fields()
        assert found["name"] == "অলনি"
        assert found["relation_name"] == "ৰাসু"

    def test_building_the_variant_class_is_a_single_pass(self):
        """Replacing ৰ then র mangles the class just inserted: "[ৰর]" contains র."""
        assert second_pass._either_ra("ঘৰ") == "ঘ[ৰর]"
        assert second_pass._either_ra("ঘর") == "ঘ[ৰর]"


class TestFairComparison:
    """Columns scored on different numbers of boxes are not a ranking."""

    def test_uneven_denominators_are_called_out_loudly(self):
        """One engine on 16 boxes and another on 36 is the same error as unequal token
        budgets, which reversed the ranking once already."""
        a, b = evaluate.Score("a"), evaluate.Score("b")
        a.total, b.total = 16, 36
        assert "NOT COMPARABLE" in evaluate.render({"a": a, "b": b})

    def test_equal_denominators_pass_without_the_warning(self):
        a, b = evaluate.Score("a"), evaluate.Score("b")
        a.total = b.total = 36
        assert "NOT COMPARABLE" not in evaluate.render({"a": a, "b": b})

    def test_the_common_subset_is_what_every_engine_answered(self):
        """Built from one complete row so adding an engine cannot silently empty the subset.

        Spelling every engine out per row meant that adding cloud-vision left every fixture row
        missing a field, and the subset went to nothing without the test saying why.
        """
        answered = {
            "t_name": "a",
            "surya": "a",
            "surya_full": "a",
            "gemini": {"name": "a"},
            "dots": "a",
            "vision": "a",
        }
        truth = {"x": {}, "y": {}, "z": {}}
        sample = [
            dict(answered, key="x"),
            dict(answered, key="y", gemini={}),
            dict(answered, key="z", vision=""),
        ]
        assert evaluate.common_keys(truth, sample) == ["x"]


class TestVisionLinesByPosition:
    """Lines from the words' own coordinates, with no ink projection involved."""

    def _w(self, text, top, left=10):
        return vision.Word(text, left, top, left + 40, top + 20)

    def test_words_on_one_line_group_together(self):
        words = [self._w("খাদৰাম", 10, 10), self._w("ৰাভা", 12, 60)]
        assert vision.lines_by_position(words, 0, 200) == ["খাদৰাম ৰাভা"]

    def test_separate_lines_stay_separate(self):
        """At a permissive threshold the name and relation lines welded into one and exact
        names collapsed from 46% to 33%."""
        words = [self._w("খাদৰাম", 10), self._w("গংগাৰাম", 40)]
        assert vision.lines_by_position(words, 0, 200) == ["খাদৰাম", "গংগাৰাম"]

    def test_lines_come_back_top_to_bottom_and_left_to_right(self):
        words = [self._w("দুই", 40, 60), self._w("এক", 40, 10), self._w("শীৰ্ষ", 5)]
        assert vision.lines_by_position(words, 0, 200) == ["শীৰ্ষ", "এক দুই"]

    def test_the_header_strip_is_dropped(self):
        """It carries the serial and the EPIC in Latin; left in, every field reads one line
        down."""
        words = [self._w("HHK0001471", 5), self._w("খাদৰাম", 40)]
        assert vision.lines_by_position(words, 0, 200) == ["খাদৰাম"]

    def test_words_outside_the_column_are_excluded(self):
        words = [self._w("খাদৰাম", 10, 10), self._w("photo", 10, 500)]
        assert vision.lines_by_position(words, 0, 200) == ["খাদৰাম"]

    def test_the_threshold_sits_on_the_measured_plateau(self):
        assert 0.6 <= vision.LINE_SHARE <= 0.8

    def test_the_header_is_kept_when_it_is_asked_for(self):
        """Dropped for the fields, needed for the EPIC. Same split, both sides returned."""
        words = [self._w("HHK0001471", 5), self._w("খাদৰাম", 40)]
        header, body = vision.header_and_body(words, 0, 200)
        assert header == ["HHK0001471"]
        assert body == ["খাদৰাম"]

    def test_a_box_gets_only_the_words_inside_it(self):
        """Constrained on left and right alone, every box in a column got the whole column and
        reported the *first* elector in it -- thirty identical rows per column, each one
        individually plausible. A stubbed run could not see it; the real values could.
        """
        words = [self._w("ৰাম", 10), self._w("শ্যাম", 500)]
        top_box = vision.words_within(words, left=0, top=0, right=200, bottom=100)
        low_box = vision.words_within(words, left=0, top=400, right=200, bottom=600)
        assert [w.text for w in top_box] == ["ৰাম"]
        assert [w.text for w in low_box] == ["শ্যাম"]

    def test_a_glyph_crossing_a_shared_border_still_belongs_to_one_box(self):
        """Box borders are shared, so requiring full containment would drop a tall glyph from
        both neighbors."""
        crossing = [vision.Word("ৰাম", left=10, top=90, right=60, bottom=130)]
        assert vision.words_within(crossing, 0, 0, 200, 100) == []
        assert len(vision.words_within(crossing, 0, 100, 200, 200)) == 1

    def test_a_two_line_header_is_split_by_script_not_by_count(self):
        words = [self._w("101", 5), self._w("HHK0001471", 25), self._w("খাদৰাম", 60)]
        header, body = vision.header_and_body(words, 0, 200)
        assert header == ["101", "HHK0001471"]
        assert body == ["খাদৰাম"]


class TestVisionElector:
    """One elector out of the lines Vision returned for its box."""

    def _lines(self, **over):
        found = {
            "name": "নাম : খাদৰাম ৰাভা",
            "relation": "পিতাৰ নাম : গংগাৰাম ৰাভা",
            "house": "ঘৰ নং : 20 ক",
            "age": "বয়স : 45 লিঙ্গ : পুৰুষ",
        }
        found.update(over)
        return [v for v in found.values() if v]

    def test_every_field_comes_off_the_labels(self):
        elector = vision.elector_from(["101 HHK0001471"], self._lines())
        assert elector.name == "খাদৰাম ৰাভা"
        assert elector.relation_name == "গংগাৰাম ৰাভা"
        assert elector.relation_type == "father"
        assert elector.house_no == "20 ক"
        assert elector.age == 45
        assert elector.sex == "M"
        assert elector.epic_no == "HHK0001471"
        assert elector.serial_no_ocr == 101

    def test_the_epic_is_read_out_of_the_header_strip(self):
        """It is Latin and digits, and it is the only field the body cannot supply."""
        elector = vision.elector_from(["HHK 0001471"], self._lines())
        assert elector.epic_no == "HHK0001471"

    def test_a_missing_field_is_flagged_rather_than_invented(self):
        elector = vision.elector_from(["101"], self._lines(age=""))
        assert elector.age is None
        assert "no_age" in elector.flags
        assert "no_epic" in elector.flags

    def test_an_empty_box_is_empty_and_carries_no_missing_field_flags(self):
        """Flagging four missing fields on a box that has nothing in it buries the boxes that
        do."""
        elector = vision.elector_from([], [])
        assert elector.is_empty
        assert elector.flags == []


class TestTruthAlignment:
    """Ground truth attached to the wrong box is worse than no ground truth.

    Half a truth set was once misattributed: crop sheets were drawn from one ordered list, and
    the list was recomputed between recording sheets, so after twelve boxes gained truth the
    slice no longer pointed at the boxes the sheet had shown. Every engine scored ~20 points
    low and the pipeline looked broken.

    The tell is a name appearing as both a reading and an expectation on different rows.
    """

    def test_a_name_appearing_on_both_sides_signals_misalignment(self):
        pairs = [("ছেকেন্দাৰ আলী", "কৃষ্টপাৰ মুৰ্মু"), ("আবনেৰ লামা", "ছেকেন্দাৰ আলী")]
        got = {evaluate.fold(a) for a, _ in pairs}
        want = {evaluate.fold(b) for _, b in pairs}
        assert got & want, "a value on both sides is a shifted alignment, not a bad engine"

    def test_a_correctly_aligned_set_has_no_such_overlap(self):
        pairs = [("খাদৰাম ৰাভা", "খাদৰাম ৰাভা"), ("গংগাৰাম", "গংগাৰাম ৰাভা")]
        got = {evaluate.fold(a) for a, _ in pairs}
        want = {evaluate.fold(b) for _, b in pairs}
        assert not (got - want) & (want - got)


class TestReconcileChecks:
    """Every check, shown failing.

    A check nobody has seen fail is a check nobody has tested, and three of the worst defects in
    this stage were checks quietly reporting success: done() proving a part was prepared but not
    by what, a bought closing total discarded before reconciliation read it, and an unreadable
    part skipped without a word.
    """

    @staticmethod
    def _rows(parts=(1, 2), per_part=3, sex="M", epic="ABC1234567"):
        return [
            {
                "part_no": p,
                "roll_section": "main",
                "sex": sex,
                "epic_no": epic,
                "name": "ক",
                "relation_name": "খ",
                "age": 30,
            }
            for p in parts
            for _ in range(per_part)
        ]

    def test_a_missing_part_fails(self):
        """AC101's failure: 113 parts extracted of 210 published, reported as success."""
        check = reconcile.parts_complete(self._rows(parts=(1, 2)), published=3)
        assert not check.passed and "2 of 3" in check.detail

    def test_all_parts_present_passes(self):
        assert reconcile.parts_complete(self._rows(parts=(1, 2)), published=2).passed

    def test_a_part_short_of_its_printed_total_fails(self):
        check = reconcile.rows_against_printed(self._rows(per_part=3), {1: 3, 2: 5})
        assert not check.passed and "1 of 2" in check.detail

    def test_matching_printed_totals_passes(self):
        assert reconcile.rows_against_printed(self._rows(per_part=3), {1: 3, 2: 3}).passed

    def test_a_part_with_no_printed_total_fails_coverage(self):
        check = reconcile.measured_coverage(self._rows(parts=(1, 2)), {1: 3})
        assert not check.passed and "1 of 2" in check.detail

    def test_a_skewed_sex_ratio_fails(self):
        """The real failure: 44.2% male extracted against a published 50.8%."""
        check = reconcile.sex_against_published(self._rows(sex="M"), male=50, female=50)
        assert not check.passed

    def test_a_matching_sex_ratio_passes(self):
        rows = self._rows(parts=(1,), per_part=1, sex="M") + self._rows(
            parts=(2,), per_part=1, sex="F"
        )
        assert reconcile.sex_against_published(rows, male=50, female=50).passed

    def test_rows_lost_between_the_stages_fails(self):
        check = reconcile.rows_match_boxes(self._rows(parts=(1, 2), per_part=3), {1: 3, 2: 9})
        assert not check.passed

    def test_a_self_graded_check_never_decides_the_verdict(self):
        """A 99% fill rate is equally consistent with 99% correct and 40% correct."""
        verdict = reconcile.Verdict(ac_no=1)
        verdict.add("parts complete", True, "3 of 3", external=True)
        verdict.add("fill rates", False, "poor", external=False)
        assert verdict.ok

    def test_a_verdict_with_no_external_evidence_fails(self):
        """Found by the second-model audit: ok was true for a verdict that checked nothing.

        The failure mode of this whole project in one line -- nothing was checked, so nothing
        was wrong. Passing has to require evidence, not the absence of complaint.
        """
        assert not reconcile.Verdict(ac_no=1).ok
        only_self = reconcile.Verdict(ac_no=1)
        only_self.add("fill rates", True, "fine", external=False)
        assert not only_self.ok

    def test_totals_for_parts_with_no_rows_fail(self):
        """The audit's reproduction: rows for parts {1,2} against totals for {1,3} passed.

        Counting the two collections agrees; asking whether they describe the same parts does
        not. Part 2 was never measured and phantom part 3 matched zero rows.
        """
        rows = self._rows(parts=(1, 2), per_part=1)
        assert not reconcile.judge(1, rows, published_parts=2, totals={1: 1, 3: 0}).ok
        assert not reconcile.measured_coverage(rows, {1: 1, 3: 0}).passed
        assert not reconcile.rows_against_printed(rows, {1: 1, 3: 0}).passed

    def test_no_totals_at_all_is_not_a_pass(self):
        assert not reconcile.rows_against_printed(self._rows(), {}).passed

    def test_an_external_check_does_decide_it(self):
        verdict = reconcile.Verdict(ac_no=1)
        verdict.add("parts complete", False, "2 of 3", external=True)
        assert not verdict.ok

    def test_an_empty_shard_never_passes(self):
        assert not reconcile.judge(1, [], published_parts=5, totals={}).ok


class TestPublishedPartIdentity:
    def test_info_page_totals_use_filename_keys_not_ocr_keys(self, tmp_path):
        import gzip

        path = tmp_path / "parts.jsonl.gz"
        record = {
            "ac_no": 999,
            "part_no": 777,
            "ac_no_file": 12,
            "part_no_file": 34,
            "electors_total": 500,
        }
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

        assert run.published_totals(12, path) == {34: 500}

    def test_every_info_page_consumer_uses_filename_keys(self, tmp_path):
        import gzip

        path = tmp_path / "parts.jsonl.gz"
        record = {
            "ac_no": 999,
            "part_no": 777,
            "ac_no_file": 12,
            "part_no_file": 34,
            "electors_total": 500,
            "electors_male": 260,
            "electors_female": 240,
        }
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

        assert validate.load_part_totals(path) == {
            (12, 34): {"total": 500, "male": 260, "female": 240, "third": None}
        }
        assert fleet.published_parts(path) == {12: 1}

    def test_a_stale_row_cache_is_reparsed(self, tmp_path, monkeypatch):
        part = tmp_path / "part1"
        part.mkdir()
        stale = {"pipeline_version": "1.8.0", "serial_no_ocr": 1}
        (part / run.ROWS).write_text(json.dumps(stale) + "\n")
        fresh = {"pipeline_version": extract.PIPELINE_VERSION, "serial_no_ocr": 1}

        monkeypatch.setattr(stage2, "ready", lambda _part: True)
        monkeypatch.setattr(
            stage2,
            "read_part",
            lambda _part, _key: extract.PartResult(1, 1, "ASM", "zip", "pdf", "0" * 64, [fresh]),
        )

        assert run.read(tmp_path, "", workers=1) == [fresh]
        assert (
            json.loads((part / run.ROWS).read_text())["pipeline_version"]
            == extract.PIPELINE_VERSION
        )


class TestDelivery:
    """What comes home has to be usable and checkable without the cloud it came from."""

    def test_a_shard_matching_its_sha_is_accepted(self, tmp_path):
        shard = tmp_path / "AC010.parquet"
        shard.write_bytes(b"rows")
        assert deliver.already_here(10, output.sha256_file(shard), tmp_path)

    def test_a_truncated_shard_is_not_accepted(self, tmp_path):
        """A broken download has the right filename. Only the bytes say otherwise."""
        shard = tmp_path / "AC010.parquet"
        shard.write_bytes(b"rows and more rows")
        sha = output.sha256_file(shard)
        shard.write_bytes(b"rows")
        assert not deliver.already_here(10, sha, tmp_path)

    def test_an_absent_shard_is_not_accepted(self, tmp_path):
        assert not deliver.already_here(10, "abc", tmp_path)

    def test_no_recorded_sha_means_not_accepted(self, tmp_path):
        """Without a checksum there is nothing to verify, so it is re-fetched rather than kept."""
        (tmp_path / "AC010.parquet").write_bytes(b"rows")
        assert not deliver.already_here(10, "", tmp_path)

    def test_a_shard_with_no_bundle_is_unverifiable_not_passing(self, tmp_path):
        """The distinction that matters: we cannot judge it, which is not the same as fine."""
        output.write_shard([], 10, directory=tmp_path)
        assert deliver.judge_local(10, tmp_path).verdict == "UNVERIFIABLE"

    def test_the_index_merges_rather_than_replaces(self, tmp_path):
        deliver.record([deliver.Arrival(ac_no=10, rows=5, verdict="PASS", reasons=[])], tmp_path)
        deliver.record([deliver.Arrival(ac_no=11, rows=7, verdict="FAIL", reasons=["x"])], tmp_path)
        import json

        entries = json.loads((tmp_path / "index.json").read_text(encoding="utf-8"))
        assert sorted(entries) == ["10", "11"] and entries["10"]["verdict"] == "PASS"
