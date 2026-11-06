#!/usr/bin/env bash
# Guard loop for MY Model A boxes only (copy of decode-km2en/guard_loop4.sh, minus the decode-only parts).
# Each pass: rebuild instance->dest map from state/box*.iid, then ONE idle_guard pass (retires, after a final pull, a box whose job
# finished (ALL_DONE/PARTIAL_DONE) or died/never started for 3 polls, or hung: live pid + stale log 45 min + idle GPU).
# idle_guard only ever touches ids that are in the map (= ids I recorded in state/); unmapped instances are reported, never destroyed.
# State file persists the 3-poll counters across passes (each pass is a new process).
# Does NOT exit when the instance list is empty (2026-11-06 lesson). Stops only when /tmp/km-model-a/state/GUARD_STOP exists.
W=/tmp/km-model-a/state; H=/tmp/km-smoke/gpu-session/helpers
DEST=/Users/tom/src/github/translator-rs/scripts/opus-trainer/data/model-a-2026-11-06
cd "$H" || exit 1
while true; do
  [ -f "$W/GUARD_STOP" ] && { echo "$(date -u +%FT%TZ) GUARD_STOP present; guard exits"; break; }
  python3 - <<PY
import glob,json,re
m={}
for f in glob.glob("$W/box*.iid"):
    k=re.search(r"box(\d+)",f).group(1); ids=re.findall(r"\d{6,}",open(f).read())
    if ids: m[ids[-1]]="$DEST/box"+k
json.dump(m,open("$W/guard_map.json","w"))
PY
  $H/.venv/bin/python $H/idle_guard.py --map $W/guard_map.json --once --state-file $W/guard_state.json --really-destroy 2>&1 | grep -v "Welcome\|Have fun"
  echo "$(date -u +%FT%TZ) pass done; credit=$($H/.venv/bin/python -c 'import vast_client as v;print(round(float(v.get_credit()),2))' 2>&1 | tail -1) instances=$($H/.venv/bin/python -c 'import vast_client as v;print(len(v.list_instances()))' 2>&1 | tail -1)"
  sleep 300
done
