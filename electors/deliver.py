"""Bringing constituencies home, and re-judging them once they arrive.

The rows live in a bucket that is meant to be deleted when the run is over. So what comes home
has to stand on its own -- usable without the cloud, and **checkable** without it either.

Three things travel, and the reasons differ:

- **the shard**, which is the product. 3-7 MB; the whole state is about 700 MB.
- **the verification bundle**, a few kilobytes of printed totals and box counts, so the verdict
  can be recomputed here rather than believed. Re-judging from ``stage1`` would mean pulling
  412 MB a constituency, and pulling fifty gigabytes to check seven hundred megabytes is the
  wrong shape -- but more to the point, ``stage1`` is one of the things being deleted.
- **the Vision words**, ~120 MB a constituency, which are the only artifact that cost money.
  $148 to re-read for the state. Composites, renders and rows are CPU we can spend again; the
  words are not. With them here the bucket becomes disposable and a better parser costs nothing.

**The SHA-256 is checked before a shard is accepted.** A truncated download is otherwise a
silently short constituency, and silently short is exactly what this project keeps producing.

**The verdict is recomputed, not copied.** The machine that produced a constituency is not a
witness to its own correctness.
"""

from __future__ import annotations

import json
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from assam_rolls import schema

from . import output, reconcile

#: Where the finished data lives. Deliberately not ``out/``, which this project treats as
#: regenerable and empties without ceremony.
HOME = Path("data/electors")
WORDS = Path("data/words")


@dataclass
class Arrival:
    """One constituency's trip home."""

    ac_no: int
    rows: int = 0
    verdict: str = ""
    reasons: List[str] = None  # type: ignore[assignment]
    fetched: bool = False
    note: str = ""


def _run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True)


def remote_shards(bucket: str) -> List[int]:
    """Constituency numbers with a shard in the bucket."""
    found = _run("gcloud", "storage", "ls", f"{bucket}/shards/").stdout
    return sorted(
        int(Path(line).name[2:5])
        for line in found.splitlines()
        if line.strip().endswith(".parquet")
    )


def remote_sha(bucket: str, ac_no: int) -> str:
    """What the entry file says the shard's SHA-256 is, or empty if there is no entry."""
    got = _run("gcloud", "storage", "cat", f"{bucket}/shards/AC{ac_no:03d}.entry.json")
    if got.returncode != 0 or not got.stdout.strip():
        return ""
    try:
        return json.loads(got.stdout).get("sha256", "")
    except ValueError:
        return ""


def already_here(ac_no: int, sha: str, home: Path = HOME) -> bool:
    """Whether this constituency is present and intact.

    Compared by content, not by filename: a half-written file has the right name.
    """
    shard = home / f"AC{ac_no:03d}.parquet"
    if not shard.exists() or not sha:
        return False
    return output.sha256_file(shard) == sha


def fetch(bucket: str, ac_no: int, home: Path = HOME, words: Optional[Path] = WORDS) -> Arrival:
    """Bring one constituency home, and refuse it if the bytes are wrong."""
    arrival = Arrival(ac_no=ac_no, reasons=[])
    home.mkdir(parents=True, exist_ok=True)
    sha = remote_sha(bucket, ac_no)

    for suffix in ("parquet", "entry.json", "verify.json"):
        source = f"{bucket}/shards/AC{ac_no:03d}.{suffix}"
        if (
            _run("gcloud", "storage", "cp", source, str(home)).returncode != 0
            and suffix == "parquet"
        ):
            arrival.note = f"could not fetch the shard from {source}"
            return arrival

    shard = home / f"AC{ac_no:03d}.parquet"
    if sha and output.sha256_file(shard) != sha:
        # Removed rather than left in place: a corrupt shard that stays on disk is one somebody
        # eventually reads.
        shard.unlink(missing_ok=True)
        arrival.note = "the downloaded shard does not match its recorded SHA-256; discarded"
        return arrival

    if words is not None:
        target = words / f"AC{ac_no:03d}"
        target.mkdir(parents=True, exist_ok=True)
        _run(
            "gcloud",
            "storage",
            "rsync",
            "-r",
            "-x",
            r".*(?<!\.words)\.json$|.*\.jsonl$",
            f"{bucket}/stage1/AC{ac_no:03d}",
            str(target),
        )
    arrival.fetched = True
    return arrival


def judge_local(
    ac_no: int, home: Path = HOME, info: Path = Path("dataset/parts.jsonl.gz")
) -> Arrival:
    """Re-judge a constituency from what is on this machine, and nothing else."""

    import gzip

    arrival = Arrival(ac_no=ac_no, reasons=[])
    shard = home / f"AC{ac_no:03d}.parquet"
    bundle_path = home / f"AC{ac_no:03d}.verify.json"
    if not shard.exists():
        arrival.verdict, arrival.note = "MISSING", "no shard"
        return arrival
    if not bundle_path.exists():
        arrival.verdict, arrival.note = "UNVERIFIABLE", "no verification bundle to judge against"
        return arrival

    rows = output.read_shard(shard)
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    male = female = 0
    if info.exists():
        with gzip.open(info, "rt", encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                if schema.source_file_key(row)[0] == ac_no:
                    male += row.get("electors_male") or 0
                    female += row.get("electors_female") or 0

    verdict = reconcile.judge(
        ac_no,
        rows,
        bundle.get("published_parts", 0),
        {int(k): v for k, v in bundle.get("printed_totals", {}).items()},
        male,
        female,
        {int(k): v for k, v in bundle.get("boxes_per_part", {}).items()} or None,
    )
    arrival.rows = len(rows)
    arrival.verdict = "PASS" if verdict.ok else "FAIL"
    arrival.reasons = [
        f"{c.name}: {c.detail}" for c in verdict.checks if c.external and not c.passed
    ]
    return arrival


#: Everything in a ``stage1`` tree except the two files a verdict is made of. The words are 120 MB
#: a constituency and the renders more; ``side.json`` and ``manifest.jsonl`` together are under a
#: megabyte. Pulling the difference is what makes backfilling six constituencies a minute's work
#: rather than an afternoon's.
NOT_EVIDENCE = r".*\.words\.json$|.*rows\.jsonl$|.*placements\.json$|.*\.png$"


def backfill_bundle(
    bucket: str, ac_no: int, home: Path = HOME, info: Path = Path("dataset/parts.jsonl.gz")
) -> Optional[Dict[str, Any]]:
    """Build the verification bundle for a constituency that finished before bundles existed.

    Six constituencies came home judged ``UNVERIFIABLE``: not failing, but with no evidence to
    fail against, which is the same thing as unchecked. Their evidence is still in the bucket, so
    the fix is to fetch the two small files per part and compute the bundle that should have been
    written at the time -- rather than re-running $25 of Vision to recover numbers we already have.

    Writes the bundle beside the shard **and back to the bucket**, so a later ``pull`` on another
    machine gets a judgeable constituency too.
    """
    import gzip
    import tempfile

    from . import run as run_module

    shard = home / f"AC{ac_no:03d}.parquet"
    if not shard.exists():
        return None

    published = 0
    if info.exists():
        with gzip.open(info, "rt", encoding="utf-8") as handle:
            published = sum(
                1 for line in handle if schema.source_file_key(json.loads(line))[0] == ac_no
            )

    with tempfile.TemporaryDirectory() as scratch:
        work = Path(scratch)
        pulled = _run(
            "gcloud",
            "storage",
            "rsync",
            "-r",
            "-x",
            NOT_EVIDENCE,
            f"{bucket}/stage1/AC{ac_no:03d}",
            str(work),
        )
        if pulled.returncode != 0 or not any(work.glob("part*")):
            return None
        rows = output.read_shard(shard)
        bundle = run_module.verification_bundle(work, rows, published)

    target = home / f"AC{ac_no:03d}.verify.json"
    target.write_text(json.dumps(bundle, indent=1) + "\n", encoding="utf-8")
    _run("gcloud", "storage", "cp", str(target), f"{bucket}/shards/{target.name}")
    return bundle


def index_path(home: Path = HOME) -> Path:
    return home / "index.json"


def record(arrivals: Sequence[Arrival], home: Path = HOME) -> Dict[str, Any]:
    """What is here, and what it is worth. Merged, so a partial run never drops earlier work."""
    path = index_path(home)
    entries: Dict[str, Any] = {}
    if path.exists():
        entries = json.loads(path.read_text(encoding="utf-8"))
    for arrival in arrivals:
        entries[str(arrival.ac_no)] = {
            "rows": arrival.rows,
            "verdict": arrival.verdict,
            "reasons": arrival.reasons or [],
            "pulled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }
    ordered = dict(sorted(entries.items(), key=lambda kv: int(kv[0])))
    path.write_text(json.dumps(ordered, indent=1) + "\n", encoding="utf-8")
    return ordered


def pull(
    bucket: str,
    home: Path = HOME,
    words: Optional[Path] = WORDS,
    say: Optional[Any] = None,
) -> List[Arrival]:
    """Every constituency in the bucket that is not already here, intact and re-judged."""
    talk = say or (lambda _message: None)
    arrivals: List[Arrival] = []
    for ac_no in remote_shards(bucket):
        sha = remote_sha(bucket, ac_no)
        if already_here(ac_no, sha, home):
            continue
        talk(f"   fetching AC{ac_no}")
        arrival = fetch(bucket, ac_no, home, words)
        if not arrival.fetched:
            talk(f"   AC{ac_no}: {arrival.note}")
            arrivals.append(arrival)
            continue
        judged = judge_local(ac_no, home)
        judged.fetched = True
        arrivals.append(judged)
        talk(f"   AC{ac_no}: {judged.rows:,} rows, {judged.verdict}")
        for reason in judged.reasons:
            talk(f"      {reason}")
    if arrivals:
        record(arrivals, home)
    return arrivals


def reapable(home: Path = HOME) -> List[int]:
    """Constituencies whose source archive is safe to delete.

    Safe means three things at once, and all three are needed:

    - the shard is **here**, and its bytes match the SHA-256 recorded when it was written
    - the local verdict is **PASS**, judged from the bundle rather than taken on trust
    - the Vision **words** are here, so a better parser costs CPU rather than $148

    An archive is 3 GB and a day of somebody's uplink; a shard is 5 MB. Deleting the expensive,
    slow-to-replace thing on the strength of the cheap one having arrived is only reasonable when
    the cheap one has been checked. AC101 looked finished at 113 of 210 parts.

    Anything short of all three keeps its archive. Re-running stage one needs the PDFs, and two
    constituencies have already needed exactly that.
    """
    index = index_path(home)
    if not index.exists():
        return []
    entries = json.loads(index.read_text(encoding="utf-8"))
    safe: List[int] = []
    for key, entry in entries.items():
        ac_no = int(key)
        if entry.get("verdict") != "PASS":
            continue
        if not (home / f"AC{ac_no:03d}.parquet").exists():
            continue
        if not any((WORDS / f"AC{ac_no:03d}").glob("**/*.words.json")):
            continue
        safe.append(ac_no)
    return sorted(safe)


def reap(bucket: str, home: Path = HOME, dry_run: bool = True, say: Optional[Any] = None) -> int:
    """Delete the source archives of constituencies that are provably home and sound."""
    talk = say or (lambda _message: None)
    freed = 0
    for ac_no in reapable(home):
        found = _run("gcloud", "storage", "ls", f"{bucket}/source/AC{ac_no}_*.zip").stdout.split()
        for archive in found:
            talk(f"   {'would delete' if dry_run else 'deleting'} {archive}")
            if not dry_run:
                _run("gcloud", "storage", "rm", archive)
            freed += 1
    return freed


def watch(
    bucket: str,
    home: Path = HOME,
    words: Optional[Path] = WORDS,
    every: int = 300,
    say: Optional[Any] = None,
) -> None:
    """Poll, so constituencies land as they finish rather than when somebody remembers."""
    talk = say or (lambda _message: None)
    while True:
        found = pull(bucket, home, words, say=talk)
        if not found:
            talk(f"   nothing new; checking again in {every}s")
        time.sleep(every)
