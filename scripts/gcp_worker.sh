#!/usr/bin/env bash
#
# Startup script for a rented CPU box that prepares and reads constituencies.
#
# Three properties matter more than anything else here, and each is one line of this script.
#
#   Nothing secret lands on the box. The instance's service account is read from the metadata
#   server, so Cloud Vision is called with a short-lived token and there is no key to leak, to
#   rotate, or to find in a shell history.
#
#   The work survives the machine. A spot VM is reclaimed with no warning and its disk goes with
#   it. So state is pushed to a bucket every few minutes, and pulled back before work starts --
#   a replacement instance resumes where the last one stopped rather than re-doing, and more to
#   the point, re-paying for, the parts already read.
#
#   The log is durable and readable while it runs. Everything goes to a file that is synced with
#   the state, so a run can be watched from a laptop and post-mortemed after the box is gone.
#
# Set by the launcher through instance metadata: BUCKET, ACS, PARTS, WORKERS.
set -uo pipefail

BUCKET=$(curl -sf -H "Metadata-Flavor: Google" \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/bucket)
ACS=$(curl -sf -H "Metadata-Flavor: Google" \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/acs)
PARTS=$(curl -sf -H "Metadata-Flavor: Google" \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/parts || echo 0)
WORKERS=$(curl -sf -H "Metadata-Flavor: Google" \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/workers || echo 0)

ROOT=/mnt/work
LOG=$ROOT/run.log
# Named for the machine. Four workers sharing one object each overwrote the others every 90
# seconds, so the only log of a four-machine run was whichever machine synced last -- and it was
# empty when it mattered. A log that four writers race to clobber is not a log.
REMOTE_LOG=$BUCKET/logs/$(hostname).log
mkdir -p "$ROOT"
exec > >(tee -a "$LOG") 2>&1

say() { echo "$(date -u +%H:%M:%S) | $*"; }

#: How long a claim may go unrefreshed before another machine may take the constituency.
#:
#: Six missed heartbeats, not a multiple of how long the work takes. The original three hours was
#: sized against the slowest constituency observed, which confused two different things: a machine
#: can be slow *and* alive, and the heartbeat is what tells them apart -- it runs on its own timer
#: and keeps firing however long the work grinds on. So a machine that has gone quiet for thirty
#: minutes is not busy, it is gone.
#:
#: The cost of getting this wrong was measured rather than imagined. assam-rolls-8 was preempted
#: two hours into AC113 with all 261 parts already synced, and the three-hour threshold would have
#: left that constituency untouchable for another 1.9 hours while seven idle-capable machines took
#: newer work. Robbing a live machine is the opposite risk, and six consecutive missed beats makes
#: it vanishingly unlikely.
STALE=1800

# Claim a constituency, atomically. --if-generation-match=0 creates the object only if it does
# not exist and returns 412 otherwise, so the winner is decided by Cloud Storage rather than by
# a check followed by a write, which two machines can both pass.
#
# The file is "host started last-beat". The start time is carried through every heartbeat rather
# than overwritten, because it is the only record of how long a constituency actually takes: with
# just a beat time, throughput has to be inferred from the gaps between finished shards, and that
# inference reads a fleet that grew from one machine to four as a fleet running at a quarter of
# its real speed. Two extra fields turn an estimate into a measurement.
CLAIM_START=0
claim() {
  local ac=$1 file=/tmp/claim.$$ now
  now=$(date +%s)
  echo "$(hostname) $now $now" > "$file"
  if gcloud storage cp --if-generation-match=0 "$file" "$BUCKET/claims/AC$ac.claim" 2>/dev/null; then
    rm -f "$file"; CLAIM_START=$now; return 0
  fi
  # Held by somebody. Stale only if it has not been refreshed -- a live worker rewrites it.
  local held age
  held=$(gcloud storage cat "$BUCKET/claims/AC$ac.claim" 2>/dev/null | awk '{print $NF}')
  age=$(( now - ${held:-0} ))
  if [ -n "$held" ] && [ "$age" -gt "$STALE" ]; then
    say "AC$ac claimed ${age}s ago and not refreshed -- taking it over"
    gcloud storage cp "$file" "$BUCKET/claims/AC$ac.claim" 2>/dev/null && {
      rm -f "$file"; CLAIM_START=$now; return 0; }
  fi
  rm -f "$file"
  return 1
}

# Rewritten while the constituency is being worked, so its claim never looks abandoned.
heartbeat() {
  local ac=$1 started=$2 file=/tmp/beat.$$
  while true; do
    sleep 300
    echo "$(hostname) $started $(date +%s)" > "$file"
    gcloud storage cp "$file" "$BUCKET/claims/AC$ac.claim" 2>/dev/null
  done
}

# One tesseract thread per worker. Tesseract uses OpenMP and will otherwise take about four
# threads inside each of the 32 worker processes: the first run on this machine sat at a load
# average of 113 on 32 cores and finished 4 parts in 37 minutes, which extrapolates to 37 hours
# for a constituency that should take half an hour. The cores are already saturated by having
# one process per core; the threads only add contention.
export OMP_THREAD_LIMIT=1
export OMP_NUM_THREADS=1

# Unbuffered, or Python block-buffers stdout into a pipe and the log shows nothing for the
# length of the run -- which is indistinguishable from a wedged job at exactly the moment you
# need to tell the difference.
export PYTHONUNBUFFERED=1

say "host $(hostname), $(nproc --all) vCPU, $(free -g | awk '/^Mem:/{print $2}') GB RAM"
[ "$WORKERS" -eq 0 ] && WORKERS=$(nproc --all)

say "installing poppler, tesseract and python"
export DEBIAN_FRONTEND=noninteractive
apt-get -qq update
# poppler renders the PDFs; tesseract reads the EPIC strip and the section headers. Both are the
# same versions the pipeline was measured against on a laptop, which is the point of naming them.
apt-get -qq install -y poppler-utils tesseract-ocr tesseract-ocr-asm tesseract-ocr-eng \
  python3-pip python3-venv git >/dev/null

# Removed first, because a stopped instance keeps its disk and this script runs again on every
# boot. git clone into an existing directory fails, the failure is not fatal, and the machine
# then runs whatever it was cloned with last time -- which is how AC10 spent two hours being
# processed by code three fixes out of date while the log cheerfully said it had cloned.
say "cloning the pipeline at its current main"
rm -rf "$ROOT/repo"
git clone -q https://github.com/in-rolls/electoral_rolls_assam_2026.git "$ROOT/repo" || {
  say "clone failed -- refusing to run stale code"; exit 1; }
cd "$ROOT/repo"
COMMIT=$(git rev-parse --short HEAD)
say "at commit $COMMIT"
python3 -m venv "$ROOT/venv"
"$ROOT/venv/bin/pip" -q install --upgrade pip
"$ROOT/venv/bin/pip" -q install -e ".[gcp]"

# Resume, part one: whatever a previous instance got through.
#
# Stage one only. Shards are deliberately *not* pulled back, and this is the difference between
# a machine that publishes its own work and one that republishes everybody's.
#
# Pulling every shard at boot and then rsyncing the whole directory back every ninety seconds made
# each machine a publisher of sixty constituencies it did not produce. Correcting a shard then
# became impossible: eleven were deleted from the bucket so the workers would redo them with a
# fixed parser, and all eleven were restored from stale local copies within six minutes. The same
# shape had already reverted the fix on the laptop twice, from the other direction.
#
# A machine needs stage one to resume a half-finished constituency. It has never needed another
# machine's shards for anything.
say "pulling any existing state from $BUCKET"
# Emptied, not merely left unpulled. A reset keeps the disk, so shards downloaded by an earlier
# boot survive it -- and the periodic sync would push them back to the bucket exactly as before,
# which is the whole behaviour this is meant to stop. Nothing is lost: a shard is written at the
# very end of a constituency, so a machine interrupted mid-run has none of its own to keep, and
# every finished one is already in the bucket.
rm -rf "$ROOT/shards"
mkdir -p "$ROOT/stage1" "$ROOT/shards"
gsutil -q -m rsync -r "$BUCKET/stage1" "$ROOT/stage1" 2>/dev/null || true
say "resumed with $(find "$ROOT/stage1" -name manifest.jsonl 2>/dev/null | wc -l) parts already prepared"

# Resume, part two: keep pushing, so the next instance has something to resume from. Started
# before the work rather than after it, because the failure this guards against is the work
# never reaching its own end.
# Composites are deliberately not synced. They are a pure function of the archive and this
# code -- about $12 of CPU to rebuild for the whole state -- while Vision's words cost $146 and
# cannot be regenerated at any price. So the cheap-to-reproduce bulk stays on the instance's
# disk and dies with it, and only what was paid for goes to the bucket: 0.6 KB a box against
# 36 KB. It is also the difference between rsyncing two gigabytes every ninety seconds and
# rsyncing a few megabytes.
(
  while true; do
    sleep 90
    gsutil -q -m rsync -r -x '.*composite.*\.png$' "$ROOT/stage1" "$BUCKET/stage1" 2>/dev/null
    gsutil -q -m rsync -r "$ROOT/shards" "$BUCKET/shards" 2>/dev/null
    gsutil -q cp "$LOG" "$REMOTE_LOG" 2>/dev/null
  done
) &
SYNC=$!

LIMIT=""
[ "$PARTS" -gt 0 ] && LIMIT="--limit $PARTS"

# "auto" means: take whatever is in the bucket that has no shard yet, and keep taking. The
# alternative -- a fixed list walked in order -- stalls on the first constituency that has not
# been uploaded yet while later ones sit ready, and it needs the uploader and the worker to
# agree on an order. Discovery needs them to agree on nothing.
IDLE=0
while true; do
  NAME=""
  for candidate in $(gsutil ls "$BUCKET/source/AC*.zip" 2>/dev/null | xargs -r -n1 basename | sort -V); do
    AC=$(echo "$candidate" | sed -E 's/^AC0*([0-9]+)_.*/\1/')
    # A shard in the bucket is the only record that a constituency is finished, and it is
    # written last, so a half-done AC is picked up again rather than skipped.
    PADDED=$(printf %03d "$AC")
    gsutil -q stat "$BUCKET/shards/AC$PADDED.parquet" 2>/dev/null && continue
    # A constituency this build already failed. Without this the picker takes the lowest
    # unfinished number every time, so one that fails in eleven seconds -- everything cached,
    # refused at the gate -- is re-claimed by every machine forever and the queue behind it
    # never moves. Keyed by commit, so fixing the cause and redeploying retries it by itself.
    gsutil -q stat "$BUCKET/failed/AC$PADDED.$COMMIT" 2>/dev/null && continue
    claim "$PADDED" || continue
    NAME=$candidate
    break
  done

  if [ -z "$NAME" ]; then
    if [ "$ACS" != "auto" ]; then break; fi
    IDLE=$((IDLE + 60))
    if [ "$IDLE" -ge 3600 ]; then
      say "nothing new for an hour -- stopping so the machine can be shut down"
      break
    fi
    [ $((IDLE % 600)) -eq 0 ] && say "every constituency in the bucket is done; waiting for more"
    sleep 60
    continue
  fi
  IDLE=0

  AC=$(echo "$NAME" | sed -E 's/^AC0*([0-9]+)_.*/\1/')
  say "fetching $NAME"
  gsutil -q cp "$BUCKET/source/$NAME" "$ROOT/$NAME" || { say "could not fetch $NAME"; sleep 60; continue; }

  heartbeat "$(printf %03d "$AC")" "$CLAIM_START" &
  BEAT=$!
  say "running AC$AC with $WORKERS workers $LIMIT"
  "$ROOT/venv/bin/python" -m electors run "$ROOT/$NAME" \
    --work "$ROOT/stage1" --shards "$ROOT/shards" \
    --manifest "$ROOT/shards/manifest.json" --workers "$WORKERS" $LIMIT
  STATUS=$?
  say "AC$AC finished with status $STATUS"
  if [ "$STATUS" -ne 0 ]; then
    # A real file, not a pipe: gsutil cp does not read stdin, so "cp -" failed silently
    # into 2>/dev/null and every failed constituency went straight back into the queue.
    echo "$COMMIT" > "$ROOT/failed.marker"
    gsutil -q cp "$ROOT/failed.marker" \
      "$BUCKET/failed/AC$(printf %03d "$AC").$COMMIT" 2>/dev/null
  fi

  # Push this AC's results before touching anything, so a machine lost now loses no money.
  gsutil -q -m rsync -r -x '.*composite.*\.png$' "$ROOT/stage1" "$BUCKET/stage1" 2>/dev/null
  gsutil -q -m rsync -r "$ROOT/shards" "$BUCKET/shards" 2>/dev/null

  # The archive stays in the bucket; the local copy and the composites are finished with. The
  # composites are ~6 GB an AC and would fill the disk by the fifth constituency.
  kill $BEAT 2>/dev/null
  # Released only after the shard is in the bucket, so a machine that dies mid-constituency
  # leaves its claim to expire rather than handing out work it has not finished.
  gcloud storage rm "$BUCKET/claims/$(printf AC%03d "$AC").claim" 2>/dev/null

  rm -f "$ROOT/$NAME"
  find "$ROOT/stage1" -name 'composite*.png' -delete
  say "AC$AC done; $(df -h "$ROOT" | awk 'NR==2{print $4}') free on the instance"
done

kill $SYNC 2>/dev/null
say "final sync"
gsutil -q -m rsync -r -x '.*composite.*\.png$' "$ROOT/stage1" "$BUCKET/stage1"
gsutil -q -m rsync -r "$ROOT/shards" "$BUCKET/shards"
gsutil -q cp "$LOG" "$REMOTE_LOG"
say "done -- results under $BUCKET/shards, composites under $BUCKET/stage1"
touch "$ROOT/FINISHED"
gsutil -q cp "$ROOT/FINISHED" "$BUCKET/FINISHED"
