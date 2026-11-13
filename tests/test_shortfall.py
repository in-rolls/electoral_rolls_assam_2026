"""What may be published, and what may not.

The old rule was exact-or-nothing: one missing row discarded a constituency. Measured on AC8 that
meant throwing away 231,131 rows to avoid 28, and at a 99.4% per-part match rate a 232-part
constituency cleared the bar about a quarter of the time -- which is what a queue stuck at fifty
looks like from outside.

An absolute five-row per-part bar was tried next and repeated the mistake at a higher altitude:
AC24, one part in 177 off by nine rows out of 738, was still discarded whole. Both cases are
pinned below with their real numbers, because the temptation each time was to move the bar until
the thing in front of me passed, and the only defense against that is measured cases in a test.

The rule that survives: whole parts going missing is caught upstream against the published part
count; here a part may be mangled but not badly mangled, and the constituency's total shortfall is
capped as a proportion.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from electors import run


def _found(parts: List[Tuple[int, int]], main_rows: int, measured: int) -> Dict[str, Any]:
    """``parts`` is (printed total, diff) for each mismatching part."""
    residuals = [
        {"part_no": i, "main_rows": total + diff, "roll_total": total, "diff": diff}
        for i, (total, diff) in enumerate(parts, start=1)
    ]
    residuals.sort(key=lambda r: (-abs(r["diff"]), r["part_no"]))
    return {
        "measured": measured,
        "matching": measured - len(residuals),
        "residuals": residuals,
        "main_rows": main_rows,
        "short": sum(abs(d) for _, d in parts),
        "worst": max((abs(d) for _, d in parts), default=0),
        "worst_fraction": max(
            (abs(d) / t for t, d in parts if t),
            default=0.0,
        ),
    }


def test_an_exact_constituency_publishes() -> None:
    assert run.shortfall_verdict(_found([], 231_131, 232)) == ""


def test_ac8_publishes() -> None:
    """Measured: 14 parts short by one to three rows, 28 of 231,131 = 0.012%."""
    parts = [(783, -3), (633, -3)] + [(1052, -2)] * 10 + [(1051, -1), (1021, -1)]
    assert run.shortfall_verdict(_found(parts, 231_131, 232)) == ""


def test_ac24_publishes() -> None:
    """Measured: one part in 177 off by nine rows. 9 of 170,726 = 0.0053%.

    The cleanest constituency measured, discarded by an absolute five-row bar.
    """
    assert run.shortfall_verdict(_found([(738, -9)], 170_726, 177)) == ""


def test_ac19_publishes() -> None:
    """Measured: six parts, worst off by 25 of 985. 45 of 216,865 = 0.021%."""
    parts = [(985, -25), (900, -2), (800, 4), (800, 4), (800, 5), (800, 5)]
    assert run.shortfall_verdict(_found(parts, 216_865, 222)) == ""


def test_a_mangled_part_fails() -> None:
    """A part missing a third of itself was not marginally missed; it was mis-processed."""
    verdict = run.shortfall_verdict(_found([(900, -300)], 216_865, 222))
    assert "off by" in verdict and "of its printed total" in verdict


def test_many_small_shortfalls_still_fail() -> None:
    """Marginal loss is tolerated, not unlimited: 0.1% of the constituency is the cap."""
    verdict = run.shortfall_verdict(_found([(800, -3)] * 400, 216_865, 222))
    assert "short of the printed totals" in verdict


def test_extra_rows_are_judged_the_same_as_missing_ones() -> None:
    """Inventing a third of a part is exactly as wrong as losing a third."""
    assert "off by" in run.shortfall_verdict(_found([(900, 300)], 216_865, 222))


def test_the_worst_residual_is_the_one_reported() -> None:
    """It used to report the first two by part number under the word "worst"."""
    verdict = run.shortfall_verdict(_found([(800, -2), (900, -400), (800, -3)], 216_865, 222))
    assert "-400" in verdict


def test_an_impossible_serial_is_refused_rather_than_written() -> None:
    """AC24's crash: one OCR'd serial overflowed int32 and discarded 170,726 rows.

    None rather than a truncation to four digits -- this field exists only as an independent
    check on the derived row order, and a check that invents a plausible answer is worse than
    one that abstains.
    """
    from electors import vision_part

    assert vision_part.plausible_serial("32") == 32
    assert vision_part.plausible_serial("3801114767") is None
    assert vision_part.plausible_serial("") is None
    assert vision_part.plausible_serial("0") is None
    # Bengali digits are what the roll prints, and int() reads them.
    assert vision_part.plausible_serial("১২") == 12


class TestGeometryAsASecondOpinion:
    """Stage one counts the boxes it cuts. Nothing used to compare anything against that.

    The count comes from the page geometry and is written before Vision is called, so it is
    independent of both the parse and the OCR of the closing page -- which makes it the only
    thing on hand able to catch a cached parse that covers half a part, or a printed total that
    was misread in a way that still adds up.
    """

    def _part(self, tmp_path, boxes_per_page, pages_count, total, rows_cached=None):
        from electors import pages as pages_mod
        from electors import stage1, summary

        part = tmp_path / "part0001"
        part.mkdir(exist_ok=True)
        stage1.write_side(
            part / "side.json",
            {
                "pages": {
                    3
                    + i: {
                        "section": pages_mod.Section.MAIN,
                        "recognised": True,
                        "boxes": {(3 + i, r, 0): "" for r in range(boxes_per_page)},
                    }
                    for i in range(pages_count)
                },
                "unknown": [],
                "summary": summary.RollSummary(
                    male=total // 2, female=total - total // 2, third=0, total=total, scale=400
                ),
            },
        )
        if rows_cached is not None:
            (part / "rows.jsonl").write_text(
                "".join('{"part_no": 1, "roll_section": "main"}\n' for _ in range(rows_cached)),
                encoding="utf-8",
            )
        return part

    def test_boxes_are_counted_from_the_pages(self, tmp_path):
        part = self._part(tmp_path, boxes_per_page=30, pages_count=29, total=870)
        assert run.boxes_cut(part) == 870
        assert run.boxes_cut(part, main_only=True) == 870

    def test_a_total_matching_the_geometry_is_measured(self, tmp_path):
        self._part(tmp_path, boxes_per_page=30, pages_count=29, total=870)
        rows = [{"part_no": 1, "roll_section": "main"} for _ in range(870)]
        found = run.reconcile(tmp_path, rows)
        assert found["measured"] == 1 and found["matching"] == 1
        assert found["summary_unusable"] == []

    def test_ac22_part_103_the_total_that_lost_a_digit(self, tmp_path):
        """Printed male 424, female 417, total 841; read as 424, 17, 441.

        Both cells dropped the same leading digit so the triple still balanced. The part is
        correctly extracted -- 841 rows against 841 boxes -- and the printed total is the thing
        that is wrong, so it is unmeasured rather than 400 rows over-extracted.
        """
        self._part(tmp_path, boxes_per_page=29, pages_count=29, total=441)
        rows = [{"part_no": 1, "roll_section": "main"} for _ in range(841)]
        found = run.reconcile(tmp_path, rows)
        assert found["summary_unusable"] == [1]
        assert found["measured"] == 0 and found["residuals"] == []
        assert run.shortfall_verdict(found) == "" if found["measured"] else True

    def test_a_part_with_no_geometry_is_still_measured(self, tmp_path):
        """Silence is not evidence against the printed total."""
        from electors import stage1, summary

        part = tmp_path / "part0001"
        part.mkdir()
        stage1.write_side(
            part / "side.json",
            {
                "pages": {},
                "unknown": [],
                "summary": summary.RollSummary(male=1, female=1, third=0, total=2, scale=400),
            },
        )
        rows = [{"part_no": 1, "roll_section": "main"} for _ in range(2)]
        found = run.reconcile(tmp_path, rows)
        assert found["measured"] == 1 and found["summary_unusable"] == []


def test_ac110_publishes() -> None:
    """Measured: 75 parts short by one to six rows, 191 of 186,712 = 0.1023%.

    Four rows over the old 0.1% aggregate cap, with a distribution in which nothing structural
    can hide -- a page is thirty rows and the worst part here is six short. The cap moved to 0.2%
    for this case; the per-part fraction still refuses mangled parts.
    """
    parts = (
        [(800, -1)] * 20
        + [(800, -2)] * 15
        + [(800, -3)] * 28
        + [(800, -4)] * 6
        + [(800, -5)] * 3
        + [(800, -6)] * 3
    )
    assert run.shortfall_verdict(_found(parts, 186_712, 237)) == ""
