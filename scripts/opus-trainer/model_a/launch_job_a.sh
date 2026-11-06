#!/usr/bin/env bash
# MODEL A COPY of helpers/launch_job.sh (the original is untouched): passes --image/--disk-gb/--deadline-hours 12, state dir /tmp/km-model-a/state.
# Create ONE Vast instance, wait for it, ship a bundle, start a detached job, start its pull loop.
# Replaces decode-km2en/launch_job_cmd.sh and launch_box.sh (same steps, one script, boot cap as an argument).
#
# usage: BUNDLE_DIR=<dir with MANIFEST.sha256> CMD="<remote command run from /w>" \
#        launch_job.sh <K> <offer_id> <hourly_total> [--boot-cap-minutes 90] [--expect-job-minutes 30]
#                      [--age-cap-minutes 210] [--credit-floor 5] [--check-only]
#   K            box number: writes decode-km2en/box<K>.iid and syncs into <DATA_ROOT>/box<K> (the guard maps box*.iid)
#   --boot-cap-minutes   absolute limit for create -> ssh+GPU ready (default 90). 2026-11-06 boots took up to ~35 min and one host
#                        never became usable, so the cap is generous but finite; passed to vast_wait_ready as --hard-cap-minutes.
#   --expect-job-minutes how long the job itself should run once started (default 30), used only for the pre-create check.
#   --age-cap-minutes    the safety net's age limit per instance (default 210 = the 3.5 h watchdog). BEFORE CREATING we require
#                        boot-cap + expected job + 10 min margin <= age cap, otherwise the watchdog could destroy the box mid-job
#                        (the failure Tom wants never to happen). We also require credit - estimated spend >= --credit-floor.
#   --check-only         run the pre-create checks and exit without creating anything.
# vast_wait_ready --destroy-on-timeout is safe now: it destroys only its own instance id (fixed 2026-11-06 after the sweep).
set -euo pipefail
H=/tmp/km-smoke/gpu-session/helpers; W="${W:-/tmp/km-model-a/state}"
DATA_ROOT="${DATA_ROOT:-/Users/tom/src/github/translator-rs/scripts/opus-trainer/data/model-a-2026-11-06}"
IMAGE="${IMAGE:-ghcr.io/davidventura/offline-translator/marian-cuda:fp16-1a743582}"; DISK_GB="${DISK_GB:-80}"
K="${1:?box number}"; OFFER="${2:?offer id}"; DPH="${3:?hourly total}"; shift 3
BOOT_CAP=90; EXPECT_JOB=30; AGE_CAP=210; FLOOR=5; CHECK_ONLY=0
while [[ $# -gt 0 ]]; do case "$1" in
  --boot-cap-minutes) BOOT_CAP="$2"; shift 2;;
  --expect-job-minutes) EXPECT_JOB="$2"; shift 2;;
  --age-cap-minutes) AGE_CAP="$2"; shift 2;;
  --credit-floor) FLOOR="$2"; shift 2;;
  --check-only) CHECK_ONLY=1; shift;;
  *) echo "unknown arg $1" >&2; exit 2;;
esac; done
: "${BUNDLE_DIR:?set BUNDLE_DIR}"; : "${CMD:?set CMD}"
[[ -f "$BUNDLE_DIR/MANIFEST.sha256" ]] || { echo "no $BUNDLE_DIR/MANIFEST.sha256" >&2; exit 2; }

# ---- pre-create checks (nothing is created, nothing billed, if any fails) ----
NEED=$(python3 -c "print(int($BOOT_CAP + $EXPECT_JOB + 10))")
if (( NEED > AGE_CAP )); then
  echo "REFUSING to create: boot cap ${BOOT_CAP}m + expected job ${EXPECT_JOB}m + 10m margin = ${NEED}m exceeds the age cap ${AGE_CAP}m;" >&2
  echo "the safety net could destroy the box mid-job. Lower --boot-cap-minutes / --expect-job-minutes, or raise the age cap everywhere first." >&2
  exit 3
fi
CREDIT=$("$H/.venv/bin/python" -c "import vast_client as v; print(round(float(v.get_credit()),2))")
EST=$(python3 -c "print(round(($EXPECT_JOB/60.0)*$DPH + 0.15,2))")   # job hours at the hourly rate + bandwidth/disk margin
if python3 -c "import sys; sys.exit(0 if $CREDIT - $EST >= $FLOOR else 1)"; then :; else
  echo "REFUSING to create: credit \$$CREDIT minus estimated spend \$$EST is below the floor \$$FLOOR" >&2; exit 4; fi
echo "pre-create OK: boot cap ${BOOT_CAP}m, job ${EXPECT_JOB}m, need ${NEED}m <= age cap ${AGE_CAP}m; credit \$$CREDIT, est spend \$$EST, floor \$$FLOOR"
[[ "$CHECK_ONLY" == 1 ]] && exit 0

# ---- create, wait, ship, start, pull ----
D="$DATA_ROOT/box$K"; mkdir -p "$D"; cd "$H"
OUT=$("$H/.venv/bin/python" vast_create.py "$OFFER" --hourly-total "$DPH" --image "$IMAGE" --disk-gb "$DISK_GB" --deadline-hours 12 --really-create)
IID=$(echo "$OUT" | grep new_contract | grep -o '[0-9]\+' || true)
if [[ -z "$IID" ]]; then echo "box$K: CREATE FAILED (offer $OFFER gone or refused):"; echo "$OUT" | tail -8; exit 1; fi
echo "box$K instance $IID" | tee "$W/box$K.iid"
BASE=$(( BOOT_CAP < 45 ? BOOT_CAP : 45 ))
"$H/.venv/bin/python" vast_wait_ready.py "$IID" --timeout-minutes "$BASE" --hard-cap-minutes "$BOOT_CAP" --destroy-on-timeout --sync-dest "$D"
bash /tmp/km-model-a/start_job_a.sh "$IID" "$BUNDLE_DIR" "$CMD"
nohup "$H/.venv/bin/python" "$H/vast_pull.py" "$IID" --sync-dest "$D" --loop 300 >> "$W/pull_box$K.log" 2>&1 &
echo "box$K launched"
