"""The transliteration stage: the guard, and the table's memory of hand review."""

from __future__ import annotations

from romanize import guards, lookup, vocabulary


class TestTransliterationNotTranslation:
    """The one rule that must not drift, and the loanword case that broke v1."""

    def test_a_translated_suffix_is_caught(self):
        assert guards.violations("গোসাইগাঁও থানা", "gossaigaon police station")

    def test_the_transliterated_suffix_is_fine(self):
        assert not guards.violations("গোসাইগাঁও থানা", "gossaigaon thana")

    def test_a_loanword_is_not_a_translation(self):
        """টাউন *is* English 'town' borrowed into Assamese; rendering it 'town' is right.

        The first version banned the English word outright and failed on this the first
        time it ran over real data.
        """
        assert not guards.violations("লামডিং ৰেলৱে টাউন", "lumding relway town")

    def test_the_native_word_for_town_is_still_caught(self):
        assert guards.violations("গুৱাহাটী চহৰ", "guwahati town")

    def test_a_banned_word_inside_a_name_does_not_trip(self):
        assert not guards.violations("পাৰ্টাবঘৰ", "partabghar")

    def test_a_loanword_survives_next_to_the_word_it_would_translate(self):
        """ভিলেজ earns "village" even when a গাওঁ is also in the value.

        Listing loanwords by hand was not enough: ``বৰবাৰী গাওঁ ফৰেষ্ট ভিলেজ`` contains
        both the borrowed word and the word it would be the translation of.
        """
        assert not guards.violations("বৰবাৰী গাওঁ ফৰেষ্ট ভিলেজ", "Barbari Gaon Forest Village")
        assert guards.violations("বৰবাৰী গাওঁ", "Barbari Village")

    def test_a_phrase_needs_every_word_borrowed(self):
        """পুলিচ earns "police"; nothing in the source earns "station", so থানা still trips."""
        assert guards.violations("পুলিচ থানা", "Police Station")

    def test_loanword_evidence_comes_from_the_lexicon(self):
        """Adding a loanword to ``tokens`` must be all it takes -- no second list here."""
        for english in ("village", "town", "part"):
            assert any(guards.loanwords_for(english)), english

    def test_check_reports_every_offender(self):
        """Distinct natives, so this isolates the translation rule from the consistency one."""
        problems = guards.check(
            [
                ("block", "কচুগাও উন্নয়ন খণ্ড", "kachugaon development block"),
                ("block", "কোকৰাঝাৰ উন্নয়ন খণ্ড", "kokrajhar unnayan khanda"),
            ]
        )
        assert len(problems) == 1


class TestLookupKeepsHandReview:
    """Hand review is the expensive part; a re-run must never undo it."""

    def entry(self, native, freq=1, field="district"):
        return vocabulary.Entry(field=field, native=native, frequency=freq, lang="ASM")

    def test_manual_rows_survive_a_machine_refill(self):
        existing = {
            ("district", "কোকৰাব্মাৰ"): lookup.Row(
                "district", "কোকৰাব্মাৰ", 862, "Kokrajhar", lookup.MANUAL
            )
        }
        merged = lookup.merge(
            existing,
            [self.entry("কোকৰাব্মাৰ", 862)],
            filled={("district", "কোকৰাব্মাৰ"): ("kukorabmar", "indicxlit")},
        )
        assert merged[0].roman == "Kokrajhar"
        assert merged[0].source == lookup.MANUAL

    def test_manual_rows_still_track_corpus_frequency(self):
        """The romanization belongs to the review; the count belongs to the corpus."""
        existing = {("district", "ক"): lookup.Row("district", "ক", 10, "K", lookup.MANUAL)}
        merged = lookup.merge(existing, [self.entry("ক", 999)])
        assert merged[0].frequency == 999 and merged[0].roman == "K"

    def test_machine_rows_are_refreshed(self):
        existing = {("district", "ক"): lookup.Row("district", "ক", 1, "old", "indicxlit")}
        merged = lookup.merge(
            existing, [self.entry("ক")], filled={("district", "ক"): ("new", "indicxlit")}
        )
        assert merged[0].roman == "new"

    def test_round_trip_through_the_file(self, tmp_path):
        path = tmp_path / "t.csv.gz"
        rows = [lookup.Row("district", "কোকৰাঝাৰ", 5, "Kokrajhar", lookup.MANUAL, "note")]
        lookup.write(rows, path)
        back = lookup.read(path)
        assert back[("district", "কোকৰাঝাৰ")] == rows[0]

    def test_pending_is_the_review_queue_most_common_first(self):
        rows = [
            lookup.Row("district", "a", 1, "", ""),
            lookup.Row("district", "b", 99, "", ""),
            lookup.Row("district", "c", 50, "done", "indicxlit"),
        ]
        assert [r.native for r in lookup.pending(rows)] == ["b", "a"]


class TestOfficialNamesOutrankHandReview:
    """The one thing allowed to overwrite a hand-checked row, and only without destroying it.

    India Post spells it *Sibsagar* where the reviewer wrote *Sivasagar*. A published
    gazetteer beats one person's recollection -- but "official" is not a single thing here,
    so the displaced spelling has to survive.
    """

    def test_an_authority_replaces_the_hand_checked_spelling(self):
        row = lookup.Row("post_office", "শিৱসাগৰ", 100, "Sivasagar", lookup.MANUAL)
        got = lookup.adopt(row, "Sibsagar", "indiapost")
        assert got.roman == "Sibsagar"
        assert got.variant == "Sivasagar"
        assert got.authority == "indiapost"

    def test_adopting_the_same_spelling_changes_nothing(self):
        row = lookup.Row("post_office", "ৰহা", 10, "Raha", lookup.MANUAL)
        assert lookup.adopt(row, "Raha", "indiapost") == row

    def test_an_official_row_survives_a_machine_refill(self):
        """Having outranked hand review, it must also outrank the model."""
        existing = {
            ("post_office", "শিৱসাগৰ"): lookup.Row(
                "post_office", "শিৱসাগৰ", 100, "Sibsagar", "indiapost", "Sivasagar", "indiapost"
            )
        }
        entry = vocabulary.Entry("post_office", "শিৱসাগৰ", 100, "ASM")
        merged = lookup.merge(
            existing, [entry], filled={("post_office", "শিৱসাগৰ"): ("xiboxagor", "indicxlit")}
        )
        assert merged[0].roman == "Sibsagar"
        assert merged[0].variant == "Sivasagar"

    def test_the_machine_still_never_overwrites_hand_review(self):
        """The original invariant, unchanged: only an authority may displace a manual row."""
        existing = {("district", "ক"): lookup.Row("district", "ক", 1, "Kokrajhar", lookup.MANUAL)}
        merged = lookup.merge(
            existing,
            [vocabulary.Entry("district", "ক", 1, "ASM")],
            filled={("district", "ক"): ("kukorabmar", "indicxlit")},
        )
        assert merged[0].roman == "Kokrajhar"


class TestSecondModelAudit:
    """Defects an independent review found in code its author had just declared finished.

    Each of these was reproduced against the shipped 46,644-row table before being fixed, so
    each test names the number it observed.
    """

    def test_check_catches_cross_field_inconsistency(self):
        """It lived only inside the anchoring run, so hand-editing the CSV could break it.

        Shipping a reviewable table is the point; a check that only fires during a machine
        pass does not cover the reviewer.
        """
        problems = guards.check(
            [
                ("post_office", "ডুমডুমা", "Doom Dooma"),
                ("police_station", "ডুমডুমা", "Doomdooma"),
            ]
        )
        assert problems and "inconsistently" in problems[0]
        assert not guards.check(
            [("post_office", "ডুমডুমা", "Doom Dooma"), ("police_station", "ডুমডুমা", "Doom Dooma")]
        )

    def test_the_school_rule_is_enforced_not_just_documented(self):
        """বিদ্যালয় is Vidyalaya. The report led with this and nothing checked it."""
        assert guards.violations("উচ্চ মাধ্যমিক বিদ্যালয়", "Uchcha Madhyamik School")
        assert not guards.violations("উচ্চ মাধ্যমিক বিদ্যালয়", "Uchcha Madhyamik Vidyalaya")

    def test_a_borrowed_school_is_still_allowed(self):
        """The source borrowed স্কুল, so "School" is the faithful reading of 21,355 rows."""
        assert not guards.violations("অৰুণোদয় হাই স্কুল", "Arunodoy High School")

    def test_an_official_confirmation_survives_a_refill(self):
        """561 confirmations were being erased by the next `fill`.

        A row an authority confirmed but did not change keeps source="lexicon", so it failed
        the is_official test in merge, fell through, and had authority rewritten to source.
        """
        prior = lookup.Row("post_office", "ৰহা", 10, "Raha", "lexicon", "", "indiapost")
        merged = lookup.merge(
            {prior.key: prior},
            [vocabulary.Entry("post_office", "ৰহা", 10, "ASM")],
            filled={prior.key: ("roha", "indicxlit")},
        )
        assert merged[0].authority == "indiapost"

    def test_a_discarded_model_guess_is_not_published_as_a_variant(self):
        """617 of 1,186 variants were rejected machine output presented as alternatives."""
        row = lookup.Row("main_town_village", "শিঙৰী", 5, "xingori", "indicxlit")
        assert lookup.adopt(row, "Singri", "indiapost", keep_variant=False).variant == ""
        hand = lookup.Row("post_office", "শিৱসাগৰ", 5, "Sivasagar", lookup.MANUAL)
        assert lookup.adopt(hand, "Sibsagar", "indiapost").variant == "Sivasagar"

    def test_scope_strength_orders_conflicting_authorities(self):
        """Two fields anchoring one name differently used to deadlock the whole run.

        Official rows were skipped by propagation, so neither could be corrected, the
        consistency check fired, and the run exited having written nothing.
        """
        from romanize import anchor

        assert anchor.scope_rank("post_office") < anchor.scope_rank("main_town_village")
        assert anchor.scope_rank("ps_name") == len(anchor.SCOPE_STRENGTH)

    def test_a_dropped_parenthesised_run_leaves_no_empty_brackets(self):
        """সুখচৰ (ধুবুৰা) matched an office literally named "Sukchar (Dhubri)"."""
        from romanize import anchor

        got = anchor.apply("সুখচৰ (ধুবুৰা)", "Sukchar (Dhubri)", {"ধুবুৰা": ["dhuburi"]})
        assert got == "Sukchar (Dhubri)"


class TestModelArtifactRepair:
    """IndicXlit renders Assamese phonetics; two of its habits are wrong in every English
    context this table will be joined against."""

    def test_x_becomes_s(self):
        """শ/ষ/স are /x/ in Assamese, but English has always written Salbari, not xalbari."""
        from romanize import normalize

        assert normalize.repair("xalbari") == "salbari"
        assert normalize.repair("Xantonogor") == "Santonogor"

    def test_doubled_vowels_collapse(self):
        from romanize import normalize

        assert normalize.repair("aambari") == "ambari"

    def test_zero_width_joiners_do_not_survive(self):
        from romanize import normalize

        assert normalize.repair("dh‌su") == "dhsu"

    def test_edge_punctuation_is_trimmed_but_inner_kept(self):
        from romanize import normalize

        assert normalize.repair("Gossaigaon boad-") == "Gossaigaon boad"
        assert normalize.repair("Kamrup (Metropolitan)") == "Kamrup (Metropolitan)"

    def test_repair_is_idempotent(self):
        from romanize import normalize

        once = normalize.repair("xaapotgram-")
        assert normalize.repair(once) == once

    def test_the_danda_is_not_fed_to_the_model(self):
        """U+09F7 is punctuation that happens to sit inside the Bengali block."""
        from romanize.backends.indicxlit import SCRIPT_RUN

        assert SCRIPT_RUN.findall("৷") == []


class TestScopedMatching:
    """Matching is only safe because the candidate set is small and chosen by a real key."""

    WORDS = {"বচদ্ৰৱা": ["botodrowa"], "ৰহা": ["roha"], "লাহাৰঘাচ": ["lahorighat"]}

    def test_the_generic_suffix_is_stripped_before_matching(self):
        """A first probe failed on 'Batadrava Ansh' purely because of the suffix."""
        from romanize import anchor

        assert anchor.core_runs("বচদ্ৰৱা অংশ") == ["বচদ্ৰৱা"]
        assert "Ansh" not in anchor.core_text("বচদ্ৰৱা অংশ", self.WORDS)

    def test_the_suffix_is_put_back_after_matching(self):
        """An authority on the place name is not a license to drop what the roll printed."""
        from romanize import anchor

        assert anchor.apply("ৰহা অংশ", "Raha", self.WORDS) == "Raha Ansh"

    def test_matching_ignores_case_and_punctuation(self):
        from romanize import anchor

        name, score, runner_up = anchor.best("Bilasipara", ["Bilashipara", "Dhubri"])
        assert name == "Bilashipara"
        assert score >= anchor.ACCEPT and score - runner_up >= anchor.MARGIN

    def test_an_empty_scope_yields_no_match(self):
        from romanize import anchor

        assert anchor.match("ৰহা", [], self.WORDS, scope="test") is None

    def test_a_near_tie_is_not_accepted(self):
        """Two similar names under one pincode: being nearest is not enough to be right.

        At 0.85 with no margin the matcher turned ধমধমা into "Nizdhamdhama" and মধুপুৰ into
        "Madhapur" -- different places, applied with confidence.
        """
        from romanize import anchor

        found = anchor.match("লাহাৰঘাচ", ["Lahorighat", "Lahorighatt"], self.WORDS, scope="test")
        assert found is not None
        assert found.score >= anchor.ACCEPT, "the best candidate is above the threshold"
        assert not found.accepted, "but not far enough ahead of the runner-up"
        assert found.needs_review

    def test_dropped_core_runs_take_their_separator(self):
        from romanize import anchor

        assert anchor.apply("চিদলা-চৰাং (অংশ-১)", "Sidli", self.WORDS) == "Sidli (Ansh-1)"

    def test_part_numbers_survive_substitution(self):
        """The official source is an authority on the place name and nothing else."""
        from romanize import anchor

        got = anchor.apply("বড়োবজাৰ (অংশ-২)", "Boro Bazar", self.WORDS)
        assert got == "Boro Bazar (Ansh-2)"


class TestScriptRunTokenization:
    """Split on script runs, not whitespace.

    Splitting on whitespace left native characters in the roman column of 112 entries
    covering 2,426 rows, because ``চক্ৰ(অংশ)`` and ``চিদলা-চৰাং`` are each one whitespace
    token: IndicXlit received them whole and converted only part.
    """

    def test_punctuation_separates_runs(self):
        from romanize.backends.indicxlit import SCRIPT_RUN

        assert SCRIPT_RUN.findall("চিদলা-চৰাং") == ["চিদলা", "চৰাং"]
        assert SCRIPT_RUN.findall("চক্ৰ(অংশ)") == ["চক্ৰ", "অংশ"]

    def test_native_digits_are_not_sent_to_the_model(self):
        """They are a pure bijection, so they convert directly rather than being guessed."""
        from romanize.backends.indicxlit import SCRIPT_RUN

        assert "১" not in "".join(SCRIPT_RUN.findall("অংশ-১"))

    def test_rejoin_keeps_every_separator(self):
        from romanize.backends.indicxlit import IndicXlitBackend

        got = IndicXlitBackend.join(
            "চিদলা-চৰাং (অংশ-১)", {"চিদলা": ["sidla"], "চৰাং": ["sorang"], "অংশ": ["ansh"]}
        )
        assert got == "sidla-sorang (ansh-1)"

    def test_already_latin_text_passes_through(self):
        from romanize.backends.indicxlit import IndicXlitBackend

        assert IndicXlitBackend.join("HARANGAJAO ITDP BLOCK", {}) == "HARANGAJAO ITDP BLOCK"


class TestModelOutputIsCached:
    """A review pass changes the lexicon, not the model. It must not re-ask the model.

    Without this, editing ``tokens`` cost a twenty-minute run over 19,216 words to get
    19,216 identical answers back -- which is what makes review something you batch up and
    dread instead of something you iterate on.
    """

    def backend(self, tmp_path):
        from romanize.backends.indicxlit import IndicXlitBackend

        return IndicXlitBackend(cache=tmp_path / "words.json")

    def test_a_fully_cached_run_never_starts_the_model(self, tmp_path, monkeypatch):
        import subprocess

        backend = self.backend(tmp_path)
        backend._save_cache({"কোকৰাঝাৰ": ["kukorajhar"]})

        def explode(*args, **kwargs):
            raise AssertionError("the model was started for a word already known")

        monkeypatch.setattr(subprocess, "run", explode)
        assert backend.romanize_many([("কোকৰাঝাৰ", "ASM")]) == {"কোকৰাঝাৰ": ["kukorajhar"]}

    def test_the_cache_round_trips(self, tmp_path):
        backend = self.backend(tmp_path)
        backend._save_cache({"কোকৰাঝাৰ": ["kukorajhar", "kokrajhar"]})
        assert backend._load_cache()["কোকৰাঝাৰ"] == ["kukorajhar", "kokrajhar"]

    def test_a_missing_cache_is_not_an_error(self, tmp_path):
        assert self.backend(tmp_path)._load_cache() == {}


class TestLeaks:
    def test_leaked_native_script_is_caught(self):
        assert guards.leaked_native("চিদলা-sorang")
        assert not guards.leaked_native("sidla-sorang")
        assert not guards.leaked_native("HAFLONG")
