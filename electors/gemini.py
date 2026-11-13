"""Reading elector boxes and pages with Gemini, under a schema.

Every engine before this one was parsed out of prose, HTML or pipe-delimited text, and that
parsing was a recurring source of bugs -- a ``<span>`` treated as a line break made a label into
a name, and two copies of one reader drifted apart three separate times. Gemini can be given a
response schema, so the model returns the record already shaped and there is nothing to parse.
That removes the parser as a variable when comparing it against the others.

**Cost, which is the reason it is being considered at all.** Local Surya reads a box in 4.6
seconds, so a pass over one constituency is on the order of 150 hours. The same constituency
through the Batch API is a few dollars and a day of waiting, and the waiting is not this
machine's.

**Failure mode to watch.** Every vision model tried here under-generates confidently: savitr's
distilled model returns one voter for a whole page of thirty and reports nothing wrong. So a
page result is checked for how many electors came back, not only for whether the ones that did
look plausible.

**Send a column, not a page.** Gemini's image tokens are *relative*: it tiles whatever it is
given into a fixed number of crops, so one box and one whole page both cost about 324 input
tokens. A page is thirty times the area, and the detail goes. Measured on the same boxes:

    per box    per column    per page
    name nearly right    80%       80%        20%
    house no right      100%       80%         0%
    age, sex            100%      100%       100%

At page scale it still returns all thirty electors with the right serials, ages and sexes -- and
gives every one of them the same invented surname, ``নৰাৱী`` where the page reads ``নাৰ্জাৰী``,
and no house numbers at all. Confidently wrong at exactly the granularity that looks cheapest.

A grid column (ten boxes) reads as well as a single box and costs a third as much, because the
output tokens dominate and those scale with electors rather than with requests.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

#: One elector, in the shape the rest of the stage already uses.
ELECTOR_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "serial_no": {
            "type": "integer",
            "description": "the number printed in the box header",
        },
        "epic_no": {
            "type": "string",
            "description": "voter id, Latin letters and digits",
        },
        "name": {"type": "string", "description": "elector's name, in Assamese script"},
        "relation_name": {
            "type": "string",
            "description": "the relative's name, Assamese",
        },
        # "unknown" rather than "": the API rejects an empty enum member, and naming the
        # sentinel is better anyway -- it gives the model a way to say "not legible" instead of
        # picking one of three relatives at random.
        "relation_type": {
            "type": "string",
            "enum": ["father", "husband", "mother", "unknown"],
            "description": "which relative the label names, or unknown if not legible",
        },
        "house_no": {"type": "string"},
        "age": {"type": "integer"},
        "sex": {"type": "string", "enum": ["M", "F", "T", "unknown"]},
    },
    "required": ["name", "age", "sex"],
}

PAGE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {"electors": {"type": "array", "items": ELECTOR_SCHEMA}},
    "required": ["electors"],
}

#: Deliberately spare. The schema carries the structure, so the prompt only has to say what the
#: image is and what not to invent -- a model that fills a blank box with a plausible elector is
#: worse than one that leaves it out.
#:
#: **A stricter prompt was tried and made it worse.** Every exact-name miss is the model
#: normalizing toward a commoner spelling -- ``প্ৰাণজিততা`` for the ``প্ৰানজিতা`` the roll
#: prints, ``সৌমন্ত্ৰী`` for ``সৌমশ্ৰী`` -- which is a language model doing what language models
#: do, and looks like something a prompt should fix. Telling it in detail not to correct
#: spellings dropped names from 94% near-right to 75% and cost a perfect age and sex score
#: besides. The instruction seems to buy hesitancy rather than fidelity.
#:
#: (16 boxes cannot resolve two or three boxes of difference. The near-right drop is the part
#: of that result worth believing.)
PROMPT = (
    "This is a box (or page) from an Assamese electoral roll. Transcribe every elector you can "
    "see, exactly as printed, keeping Assamese script for names. Do not translate or "
    "transliterate. Do not guess: leave a field empty if it is not legible, and do not invent "
    "electors that are not there."
)

DEFAULT_MODEL = "gemini-2.5-flash-lite"


@dataclass
class Usage:
    """Token counts, so the cost estimate can be replaced with a measurement."""

    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.calls += other.calls

    def cost(self, per_million_in: float, per_million_out: float, batch: bool = True) -> float:
        """Dollars at the given rates. Batch is half price."""
        discount = 0.5 if batch else 1.0
        return discount * (
            self.input_tokens / 1e6 * per_million_in + self.output_tokens / 1e6 * per_million_out
        )


@dataclass
class Result:
    """What one image produced."""

    electors: List[Dict[str, Any]] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    error: str = ""


def client(api_key: Optional[str] = None):
    """A Gemini client. The key comes from the environment and is never written down."""
    from google import genai

    key = api_key or os.environ.get("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    return genai.Client(api_key=key)


def read_image(
    image_path: str,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    prompt: str = PROMPT,
) -> Result:
    """One image -> the electors on it, already shaped by the schema."""
    from google.genai import types

    api = client(api_key)
    with open(image_path, "rb") as handle:
        data = handle.read()

    try:
        response = api.models.generate_content(
            model=model,
            contents=[
                types.Part.from_bytes(data=data, mime_type="image/png"),
                prompt,
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=PAGE_SCHEMA,
                temperature=0,
            ),
        )
    except Exception as exc:  # one bad image must not end a run of thousands
        return Result(error=f"{type(exc).__name__}: {exc}")

    meta = getattr(response, "usage_metadata", None)
    usage = Usage(
        input_tokens=getattr(meta, "prompt_token_count", 0) or 0,
        output_tokens=getattr(meta, "candidates_token_count", 0) or 0,
        calls=1,
    )
    parsed = getattr(response, "parsed", None) or {}
    electors = parsed.get("electors", []) if isinstance(parsed, dict) else []
    return Result(electors=[dict(e) for e in electors], usage=usage)


#: The schema's sentinel, mapped back to the empty string the rest of the stage uses.
UNKNOWN = "unknown"


def _known(value: Any) -> str:
    text = str(value or "").strip()
    return "" if text == UNKNOWN else text


def fields_of(elector: Dict[str, Any]) -> Dict[str, Any]:
    """One record in the shape :mod:`electors.evaluate` scores.

    House numbers come back in Bengali digits -- ``১৩`` -- because the prompt asks for the page
    exactly as printed, and that is what is printed. Scoring them against Latin digits marked
    the model wrong on 14 of 16 house numbers it had in fact read correctly. Normalized here,
    as everywhere else in this stage.
    """
    from assam_rolls import schema

    return {
        "name": _known(elector.get("name")),
        "age": elector.get("age"),
        "house": schema.normalize_digits(_known(elector.get("house_no"))),
        "sex": _known(elector.get("sex")),
    }


def per_page_cost(usage: Usage, pages: int, rates: Sequence[tuple]) -> List[str]:
    """What a measured page costs, extrapolated -- with the measurement stated."""
    if not pages:
        return []
    per_in = usage.input_tokens / pages
    per_out = usage.output_tokens / pages
    lines = [
        f"measured over {pages} pages: {per_in:,.0f} input and {per_out:,.0f} output tokens each",
        "",
        f"   {'model':<22}{'per page':>12}{'AC1 (4,100 pp)':>18}{'126 ACs':>12}",
    ]
    for name, cost_in, cost_out in rates:
        page = 0.5 * (per_in / 1e6 * cost_in + per_out / 1e6 * cost_out)
        lines.append(
            f"   {name:<22}{'$' + format(page, '.5f'):>12}"
            f"{'$' + format(page * 4100, ',.2f'):>18}{'$' + format(page * 4100 * 126, ',.0f'):>12}"
        )
    return lines


#: Electors per request at the granularity that measured best. A page is 30 and reads badly.
BOXES_PER_REQUEST = 10

#: Batch rates, halved inside :meth:`Usage.cost`. Listed rather than looked up so a price change
#: is a visible edit rather than a silent drift in what gets quoted.
RATES = (
    ("gemini-2.5-flash-lite", 0.10, 0.40),
    ("gemini-2.5-flash", 0.30, 2.50),
    ("gemini-3-flash-preview", 0.50, 3.00),
)
