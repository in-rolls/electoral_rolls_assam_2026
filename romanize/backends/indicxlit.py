"""IndicXlit (AI4Bharat) -- the best free option measured, at 5/12 on known districts.

Runs in an **isolated Python 3.10 environment**, because ``ai4bharat-transliteration``
needs ``fairseq`` and an old TensorFlow/torch stack that will not coexist with this
project's Python 3.12. The recipe is lifted from ``indicate/training/baseline_indicxlit.py``
rather than reinvented.

The vocabulary goes through in **chunks of a few thousand words**, not one word at a time
and not all at once. Model load is ~10 seconds and per-word inference is milliseconds, so a
process per word would spend all its time loading; but a single process over all 19,216
words writes nothing until it finishes, and lost twenty minutes of work the first time it
was interrupted. A chunk pays ~5% overhead for progress that survives.

Output is cached (see ``CACHE``), so a second run asks only about words never seen.

Quality, measured on twelve well-known Assam districts (top-1 exact):

    guwahati, udalguri, hailakandi, dhemaji, bongaon            correct
    dibrugorh (dibrugarh at rank 2)                             near
    kukorajhar, zurhat, kasar, xunitpur, nogaon, tinisukiya     wrong

The misses are systematic, not noise: IndicXlit renders Assamese **phonetics** -- ``xunitpur``
because Assamese শ is /x/, ``zurhat`` because যো is /z/ -- where conventional English
spellings are older anglicisations. It is not undertrained; it is optimizing a different
target. That is why its output seeds a reviewable table instead of shipping directly.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

NAME = "indicxlit"

#: Runs of Bengali-script letters. Everything else -- spaces, hyphens, parentheses,
#: apostrophes, digits, stray punctuation -- is a separator kept verbatim.
#:
#: Splitting on whitespace alone was wrong and measurably so: ``চক্ৰ(অংশ)``,
#: ``চিদলা-চৰাং`` and ``রা'ঈরখাড়া`` are each a single whitespace token, so IndicXlit got
#: them whole, transliterated part and returned the rest in native script. That left
#: native characters in the roman column of 112 entries covering 2,426 rows.
SCRIPT_RUN = re.compile(r"[\u0980-\u09E5\u09F0-\u09F6\u09F8-\u09FF]+")

#: Bengali-Assamese digits are a pure bijection to Latin, so they are converted directly
#: rather than sent to a model -- exact, free and testable, the same call the extraction
#: pipeline makes. They are excluded from ``SCRIPT_RUN`` so a lone ``১`` is not handed to a
#: word transliterator that has never seen one.
#:
#: U+09F7, the Bengali danda, is excluded for the same reason: it is punctuation that happens
#: to sit inside the script block, and feeding it to a word transliterator produced ``xou``
#: and ``ugh`` in 87 values.
DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

#: IndicXlit language codes, by the corpus's language code.
XLIT_LANG = {"ASM": "as", "BEN": "bn", "ENG": None}

#: The dependency set that actually resolves. Pinned because the loose versions pull a
#: torch/TF combination that fails to import.
UV_ARGS = [
    "uv",
    "run",
    "--python",
    "3.10",
    "--no-project",
    "--with",
    "ai4bharat-transliteration",
    "--with",
    "gevent==24.11.1",
    "--with",
    "tensorflow==2.15.1",
    "--with",
    "keras==2.15.0",
    "--with",
    "tensorflow-addons==0.23.0",
    "--with",
    "torch==2.2.2",
    "python",
]

#: Runs inside the isolated interpreter: read {lang: [words]} as JSON, write {word: [top-k]}.
WORKER = """
import json, sys
from ai4bharat.transliteration import XlitEngine

payload = json.load(open(sys.argv[1], encoding="utf-8"))
topk = int(sys.argv[3])
out = {}
for lang, words in payload.items():
    engine = XlitEngine(lang, beam_width=4, src_script_type="indic")
    for word in words:
        try:
            out[word] = engine.translit_word(word, lang, topk=topk)
        except Exception as exc:
            out[word] = {"_error": f"{type(exc).__name__}: {exc}"}
json.dump(out, open(sys.argv[2], "w", encoding="utf-8"), ensure_ascii=False)
"""


#: Model output for words already seen, so a review pass costs nothing.
#:
#: The model is deterministic and the review loop only changes which words the *lexicon*
#: overrides -- re-running it after editing ``tokens`` asks 19,216 identical questions and
#: gets 19,216 identical answers, at twenty minutes a pass. Cached, a pass is instant, which
#: is what makes reviewing iterative instead of a thing you batch up and dread.
#:
#: Deliberately not shipped: it is derivable from the model and the corpus, and the useful
#: artifact is the table.
CACHE = Path("out/indicxlit_words.json")


class IndicXlitBackend:
    """Batch-romanize through the isolated environment."""

    name = NAME

    #: Words per subprocess invocation. Model load is ~10 seconds, so a chunk this size
    #: spends under 5% of its time loading -- cheap enough to buy durability. A single
    #: invocation over all 19,216 words was faster on paper and lost twenty minutes of
    #: work the first time it was interrupted, because nothing was written until the end.
    CHUNK = 4000

    def __init__(self, topk: int = 3, timeout: int = 3600, cache: Optional[Path] = CACHE) -> None:
        self.topk = topk
        self.timeout = timeout
        self.cache = cache

    def _load_cache(self) -> Dict[str, List[str]]:
        if self.cache is None or not self.cache.exists():
            return {}
        return json.loads(self.cache.read_text(encoding="utf-8"))

    def _save_cache(self, words: Dict[str, List[str]]) -> None:
        if self.cache is None:
            return
        self.cache.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.cache.with_suffix(".tmp")
        tmp.write_text(json.dumps(words, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.cache)

    def romanize_many(
        self,
        items: Iterable[Tuple[str, str]],
        progress: Optional[Callable[[str], None]] = None,
    ) -> Dict[str, List[str]]:
        """``items`` is ``(text, lang)``. Returns ``{text: [candidates]}``.

        Values are split into **runs of native script** and transliterated run by run.
        IndicXlit is a *word* transliterator: a phrase fed in whole comes back mangled,
        and a run glued to punctuation comes back half-converted.
        """
        known = self._load_cache()
        by_lang: Dict[str, set] = {}
        for text, lang in items:
            code = XLIT_LANG.get(lang)
            if code is None:
                continue
            for run in SCRIPT_RUN.findall(text.translate(DIGITS)):
                if run not in known:
                    by_lang.setdefault(code, set()).add(run)
        if not by_lang:
            return known

        todo = [(lang, sorted(words)) for lang, words in by_lang.items()]
        remaining = sum(len(words) for _, words in todo)
        for lang, words in todo:
            for start in range(0, len(words), self.CHUNK):
                chunk = words[start : start + self.CHUNK]
                fresh = self._run({lang: chunk})
                # Errors are not cached: a word that failed once should be retried, not
                # remembered as unanswerable.
                known.update({w: c for w, c in fresh.items() if not isinstance(c, dict)})
                self._save_cache(known)  # after every chunk, so an interrupted run resumes
                remaining -= len(chunk)
                if progress:
                    progress(f"    {len(known):,} words known, {remaining:,} to go")
        return known

    def _run(self, payload: Dict[str, List[str]]) -> Dict[str, List[str]]:
        with tempfile.TemporaryDirectory() as tmp:
            src, dst, script = (Path(tmp) / n for n in ("in.json", "out.json", "worker.py"))
            src.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            script.write_text(WORKER, encoding="utf-8")
            result = subprocess.run(
                [*UV_ARGS, str(script), str(src), str(dst), str(self.topk)],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            if not dst.exists():
                raise RuntimeError(
                    f"IndicXlit worker produced nothing (rc={result.returncode}): "
                    f"{result.stderr.strip()[-400:]}"
                )
            return json.loads(dst.read_text(encoding="utf-8"))

    @staticmethod
    def join(text: str, words: Dict[str, List[str]], rank: int = 0) -> str:
        """Rebuild a value, replacing each native run and keeping every separator.

        A string with no native run -- the English constituency's ``HAFLONG``,
        ``HARANGAJAO ITDP BLOCK`` -- comes back unchanged, which is the right answer: it
        is already Latin and there is nothing to transliterate.
        """

        def replace(match: "re.Match") -> str:
            candidates = words.get(match.group())
            if isinstance(candidates, dict) or not candidates:
                return match.group()  # error or unknown: leave the original
            return candidates[min(rank, len(candidates) - 1)]

        return SCRIPT_RUN.sub(replace, text.translate(DIGITS))


def available() -> bool:
    """Whether ``uv`` is present; the environment itself is built on first use."""
    try:
        subprocess.run(["uv", "--version"], capture_output=True, timeout=30)
        return True
    except (OSError, subprocess.SubprocessError):
        return False
