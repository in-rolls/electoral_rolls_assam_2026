"""Official English names, fetched once and cached.

The table's weakest claim was that its spellings are *correct*. IndicXlit matches the
hand-checked tokens 29.4% of the time, so most unreviewed spellings are plausible rather
than right, and nothing in the pipeline could tell the difference.

What makes this fixable is that the corpus carries a **join key nobody had used**:
``pin_code`` is present on 100% of rows and 1,408 of 1,420 post offices map to exactly one
pin. India Post publishes, per pincode, the official English post office name along with its
block and district. That turns "guess the spelling" into "pick from twelve candidates", which
is a problem fuzzy matching can actually solve.

One source is implemented:

``indiapost``
    ``api.postalpincode.in``, a mirror of the Department of Posts directory. One request per
    pincode, 554 of them. Authoritative for post office names, and a useful secondary
    witness for block and district. Only Assam offices are kept -- border pincodes serve
    across the state line, and a candidate pool should not hold names from a state this
    corpus never covers.

The Local Government Directory would cover ``revenue_circle`` and ``police_station``, which
nothing else here reaches. Its bulk export sits behind a CSRF-bearing form and no working
download was found, so those two fields remain lexicon-and-model only. ``lookup.AUTHORITIES``
already reserves the ``lgd`` name for whenever that changes.

Everything is cached under ``out/gazetteer/``. The network is hit once; every later run of
the audit is offline and reproducible, the same discipline as the IndicXlit cache.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

CACHE_DIR = Path("out/gazetteer")

POSTAL_API = "https://api.postalpincode.in/pincode/{pin}"

#: Deliberately slow. This is a free public API being asked 554 questions for a one-off
#: audit; there is no deadline that justifies hammering it.
REQUEST_PAUSE = 0.4

USER_AGENT = "electoral_rolls_assam_2026 (open data audit; github.com/in-rolls)"


#: Border pincodes serve offices across the state line. Two Meghalaya offices reached the
#: Assam gazetteer this way -- harmless so far, but a candidate pool should not contain
#: names from a state the corpus never covers.
STATE = "Assam"


@dataclass(frozen=True)
class PostOffice:
    """One office as the Department of Posts spells it."""

    name: str
    pincode: str
    block: str
    district: str
    branch_type: str
    state: str = ""

    @classmethod
    def from_api(cls, row: dict) -> "PostOffice":
        return cls(
            # The API returns trailing spaces on some names ("Guwahati ").
            name=(row.get("Name") or "").strip(),
            pincode=(row.get("Pincode") or "").strip(),
            block=(row.get("Block") or "").strip(),
            district=(row.get("District") or "").strip(),
            branch_type=(row.get("BranchType") or "").strip(),
            state=(row.get("State") or "").strip(),
        )


@dataclass
class Gazetteer:
    """Official names, grouped by the scope a value can be matched within.

    Scoping is the whole safety argument. Matching a post office against every name in
    India would confidently return nonsense; matching it against the twelve offices sharing
    its pincode is a question with a real answer.
    """

    offices_by_pin: Dict[str, List[PostOffice]] = field(default_factory=dict)
    blocks_by_district: Dict[str, List[str]] = field(default_factory=dict)
    districts: List[str] = field(default_factory=list)

    def offices(self, pin: str) -> List[PostOffice]:
        return self.offices_by_pin.get(str(pin).strip(), [])

    def __len__(self) -> int:
        return sum(len(v) for v in self.offices_by_pin.values())


def _get(url: str, timeout: int = 30) -> Optional[object]:
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8", "replace"))
    except (urllib.error.URLError, json.JSONDecodeError, OSError):
        return None


def fetch_post_offices(
    pincodes: Iterable[str],
    cache_dir: Path = CACHE_DIR,
    progress: Optional[Callable[[str], None]] = None,
) -> Dict[str, List[PostOffice]]:
    """Official post offices per pincode, cached one file per pincode.

    One file each rather than one big file, so an interrupted fetch keeps everything it
    already paid for -- the same lesson the IndicXlit run taught expensively.
    """
    directory = cache_dir / "pincode"
    directory.mkdir(parents=True, exist_ok=True)

    out: Dict[str, List[PostOffice]] = {}
    wanted = sorted({str(p).strip() for p in pincodes if str(p).strip()})
    fetched = 0
    for index, pin in enumerate(wanted, start=1):
        path = directory / f"{pin}.json"
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
        else:
            payload = _get(POSTAL_API.format(pin=pin))
            time.sleep(REQUEST_PAUSE)
            fetched += 1
            # A miss is cached as an empty list: a pincode the directory does not know is a
            # fact about the directory, not a transient failure, and re-asking every run
            # would be 94 wasted requests each time.
            path.write_text(json.dumps(payload or [], ensure_ascii=False), encoding="utf-8")
        rows = []
        if isinstance(payload, list) and payload:
            rows = payload[0].get("PostOffice") or []
        offices = [o for o in map(PostOffice.from_api, rows) if o.state == STATE]
        if offices:
            out[pin] = offices
        if progress and index % 50 == 0:
            progress(f"    {index}/{len(wanted)} pincodes ({fetched} fetched, rest cached)")
    return out


def load(
    pincodes: Iterable[str],
    cache_dir: Path = CACHE_DIR,
    progress: Optional[Callable[[str], None]] = None,
) -> Gazetteer:
    """Assemble every official source we have into one scoped lookup."""
    offices = fetch_post_offices(pincodes, cache_dir, progress)

    blocks: Dict[str, set] = {}
    districts: set = set()
    for group in offices.values():
        for office in group:
            if office.district:
                districts.add(office.district)
            if office.block and office.district:
                blocks.setdefault(office.district, set()).add(office.block)

    return Gazetteer(
        offices_by_pin=offices,
        blocks_by_district={d: sorted(b) for d, b in blocks.items()},
        districts=sorted(districts),
    )
