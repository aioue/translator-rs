#!/usr/bin/env bash
# Mac-side: ship ANY small bundle to a ready Vast instance and start one detached command.
# Why this exists: the 2026-11-06 GPU bench was launched by hand and hit four avoidable
# mistakes (zsh arrays are 1-indexed so the host/port/user split broke; /w/markers was not
# created so the pre-destroy sync failed; sacrebleu is not in the image so the job crashed
# on import; a one-line ssh that backgrounded a job hung for minutes). This script does the
# safe version once. Use it instead of typing ssh/rsync by hand.
#
# usage: start_job.sh <instance_id> <bundle_dir> <remote command run from /w, e.g.
#        "python3 -u /w/bundle/scripts/bench_kd.py --src ... --out /w/out/bench">
# <bundle_dir> must contain MANIFEST.sha256 (sha256sum format, relative paths).
set -euo pipefail
DIR="/tmp/km-smoke/gpu-session/helpers"  # model-a copy lives elsewhere; helper modules stay in the original dir
# shellcheck source=bash_compat.sh
source "$DIR/bash_compat.sh"   # needs bash 4+: macOS /bin/bash 3.2 and zsh both break array indexing

INSTANCE_ID="${1:?instance id}"; BUNDLE="${2:?bundle dir}"; CMD="${3:?remote command}"
[[ -f "$BUNDLE/MANIFEST.sha256" ]] || { echo "no $BUNDLE/MANIFEST.sha256" >&2; exit 2; }

readarray -t EP < <(python3 "$DIR/ssh_endpoint_for_instance.py" "$INSTANCE_ID")   # lines: host, port, user
HOST="${EP[0]:-}"; PORT="${EP[1]:-}"; USER_="${EP[2]:-root}"
[[ -n "$HOST" && -n "$PORT" ]] || { echo "no ssh endpoint for $INSTANCE_ID" >&2; exit 1; }
SSH=(ssh -n -p "$PORT" -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 "$USER_@$HOST")

# /w/markers must exist or the pre-destroy sync (vast_sync.DEFAULT_REMOTE_DIRS) reports a failure.
"${SSH[@]}" "mkdir -p /w/out /w/hf /w/logs /w/markers /w/bundle && nvidia-smi -L"
rsync -az -e "ssh -p $PORT -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20" \
  "$BUNDLE/" "$USER_@$HOST:/w/bundle/"
"${SSH[@]}" "cd /w/bundle && sha256sum -c MANIFEST.sha256"
# The hy-kd image has vLLM/torch but not sacrebleu; gate_flores_plus imports it at module load.
# (model-a copy: sacrebleu install dropped; the marian-cuda image has no need for it, scoring runs on the Mac)
# Detach fully (stdin/stdout/stderr redirected, setsid, pid file), as start_session.sh does, so
# the ssh returns at once; then prove the process is alive instead of trusting an echo.
"${SSH[@]}" "cd /w; nohup setsid bash -c $(printf %q "$CMD") </dev/null >> /w/logs/job.log 2>&1 & echo \$! > /w/job.pid; disown"
sleep 5
PID="$("${SSH[@]}" "cat /w/job.pid")"
"${SSH[@]}" "kill -0 $PID" || { echo "job pid $PID died at once; see /w/logs/job.log" >&2; "${SSH[@]}" "tail -20 /w/logs/job.log" >&2; exit 1; }
echo "started on instance $INSTANCE_ID pid=$PID (log /w/logs/job.log); poll with ssh -p $PORT $USER_@$HOST"
