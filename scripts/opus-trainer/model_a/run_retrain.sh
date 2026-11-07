#!/bin/bash
# Model A RETRAIN (number augmentation + UI/dialogue weighting): data is prepared on the km box (bundle), this runs
# gpu_smoke -> KD student -> decode -> finetune -> decode on one rented box. Same trainer scripts/configs as run 1 (run_model_a.sh).  Runs on a rented NVIDIA box in the
# marian-cuda:fp16-1a743582 image (or on the km box in SMOKE=1 mode with a CPU marian wrapper).
#
# Resumability: every stage drops /w/markers/<stage>.done; a restart skips finished stages and Marian resumes from its own
# model.npz checkpoints (train_student.sh/finetune_student.sh pass --overwrite and resume from the last save).
# Guard contract: this script is started by start_job.sh (so /w/job.pid exists), appends to /w/logs/job.log every 2 minutes
# (heartbeat) and ends with ALL_DONE.  A failed stage writes FAILED_<stage> to the log and exits non-zero WITHOUT a done marker.
#
# Env: ROOT (default /w) ; SMOKE=1 (tiny subsets; MARIAN must be a wrapper that makes the binary CPU-only and short)
#      KD_TIMEOUT_MIN (default 250) FT_TIMEOUT_MIN (default 60) BW_TIMEOUT_MIN (default 120)
set -uo pipefail
ROOT=${ROOT:-/w}; B=$ROOT/bundle; D=$B/data; S=$B/scripts; O=$ROOT/out; M=$ROOT/markers; L=$ROOT/logs/job.log
# $O (= /w/out) is what the Mac pull loop rsyncs every 5 minutes: keep only models, logs, scores, reports and rejects there.
# Multi-GB working TSVs live in $WK (= /w/work) and are never pulled; they are regenerable from the bundle + ce_scores + rejects.
WK=$ROOT/work; MT=$O/meta
mkdir -p "$O" "$M" "$ROOT/logs" "$ROOT/tmp" "$WK" "$MT"
export PATH=$S:$PATH PYTHONUNBUFFERED=1
export MA_REPO=$S MA_EVAL=$D/eval MA_HASH=$D/hashes MA_PROBES_KM=$D/probes_en_to_km.tsv MA_VALID=$D/valid
export OPUSTRAINER_CFG=$S/configs/opustrainer.student.km.yml
SMOKE=${SMOKE:-0}
GPUS=${GPUS:-0}
log() { echo "$(date -u +%FT%TZ) $*" >> "$L"; }
( while true; do sleep 120; echo "$(date -u +%FT%TZ) hb gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>/dev/null | head -1) $(ls -t $O/*/train.log 2>/dev/null | head -1 | xargs -r tail -qn1 | cut -c1-170)" >> "$L"; done ) &
HB=$!
trap 'kill $HB 2>/dev/null' EXIT
stage() { local n=$1; shift
  if [ -f "$M/$n.done" ]; then log "skip $n (done marker)"; return 0; fi
  log "start $n"; local t0=$SECONDS; "$@"; local rc=$?
  if [ $rc -eq 0 ] || [ $rc -eq 124 ]; then touch "$M/$n.done"; log "done $n rc=$rc secs=$((SECONDS-t0))"; return 0; fi
  log "FAILED_$n rc=$rc secs=$((SECONDS-t0))"; exit $rc; }
lim() { local mins=$1; shift; if [ "$SMOKE" = 1 ]; then "$@"; else timeout --kill-after=180 $((mins*60)) "$@"; fi; }

log "job start; host=$(hostname) smoke=$SMOKE"; log "marian: $(marian --version 2>&1 | head -1)"; nvidia-smi -L >> "$L" 2>&1 || true
( cd "$B" && sha256sum -c MANIFEST.sha256 > "$O/manifest_check.txt" 2>&1 ) || { log "FAILED_manifest"; exit 3; }

s_inputs() {
  mkdir -p "$WK/data" "$WK/kd" && cd "$WK/kd" || return 1
  zcat "$D/kd.v2.clean.tsv.gz" > kd.clean.tsv && zcat "$D/ft.v2.clean.tsv.gz" > ft.clean.tsv || return 1
  cp "$D/kd.v2.clean.tsv.evalclean.json" kd.clean.tsv.evalclean.json; cp "$D/ft.v2.clean.tsv.evalclean.json" ft.clean.tsv.evalclean.json
  cp kd.clean.tsv.evalclean.json ft.clean.tsv.evalclean.json "$MT/"; wc -l kd.clean.tsv ft.clean.tsv >> "$L"; }
s_gpu_smoke() { local G="$WK/gpu_smoke"; rm -rf "$G"; mkdir -p "$G" && cd "$G" || return 1
  awk 'NR%80==0' "$WK/kd/kd.clean.tsv" > kd.tsv; awk 'NR%40==0' "$WK/kd/ft.clean.tsv" > ft.tsv
  python3 "$S/model_a_exclude.py" ft.tsv ft.clean.tsv --text-columns 2 || return 11
  export SHORT_AFTER=${SHORT_AFTER:-150u}
  MARIAN=$S/marian_short.sh bash "$S/train_student.sh" "$G/kd.tsv" "$D/vocab.spm" "$D/valid.km2en.tsv" "$G/kd/model.npz" "$GPUS" "--fp16 --workspace 12000 --mini-batch 4000" > kd.log 2>&1 || { tail -40 kd.log >> "$L"; return 15; }
  MARIAN=$S/marian_short.sh bash "$S/finetune_student.sh" "$G/ft.clean.tsv" "$D/vocab.spm" "$G/kd/model.npz.best-ce-mean-words.npz" "$D/valid.km2en.tsv" "$G/ft/model.npz" "$GPUS" > ft.log 2>&1 || { tail -40 ft.log >> "$L"; return 16; }
  head -n 20 "$D/eval/flores-plus-devtest.km.zwsp-stripped" > d.src
  bash "$S/decode_flores.sh" "$G/ft/model.npz.best-ce-mean-words.npz" "$D/vocab.spm" d.src d.hyp "$GPUS" > dec.log 2>&1 || { tail -20 dec.log >> "$L"; return 17; }
  [ "$(wc -l < d.hyp)" = 20 ] || return 18
  { echo "gpu_smoke ok"; head -n 2 d.hyp; } >> "$L" 2>&1; return 0; }
s_kd() { cd "$O" && mkdir -p student_kd && lim ${KD_TIMEOUT_MIN:-330} bash "$S/train_student.sh" "$WK/kd/kd.clean.tsv" "$D/vocab.spm" "$D/valid.km2en.tsv" "$O/student_kd/model.npz" "$GPUS" "${KD_EXTRA:---fp16 --workspace 12000 --mini-batch 4000}" 2>&1 | tee -a "$O/student_kd/train.log" > /dev/null; return ${PIPESTATUS[0]}; }
s_ft() { cd "$O" && mkdir -p student_ft; KB="$O/student_kd/model.npz.best-ce-mean-words.npz"; [ -f "$KB" ] || KB="$O/student_kd/model.npz"
  lim ${FT_TIMEOUT_MIN:-60} bash "$S/finetune_student.sh" "$WK/kd/ft.clean.tsv" "$D/vocab.spm" "$KB" "$D/valid.km2en.tsv" "$O/student_ft/model.npz" "$GPUS" 2>&1 | tee -a "$O/student_ft/train.log" > /dev/null; return ${PIPESTATUS[0]}; }
s_decode() { local who=$1; cd "$O" && mkdir -p decode
  for _once in 1; do
    M1="$O/student_$who/model.npz.best-ce-mean-words.npz"; [ -f "$M1" ] || { log "no best checkpoint for $who; skipping decode"; continue; }
    mkdir -p "decode/$who"
    for s in flores-plus-dev flores-plus-devtest holdout-ui-300 holdout-dialogue-300 alt-dev alt-test ntrex-128; do
      SRC="$D/eval/$s.km.zwsp-stripped"; [ "$SMOKE" = 1 ] && { head -n 20 "$SRC" > "$ROOT/tmp/$s.src"; SRC="$ROOT/tmp/$s.src"; }
      ( cd "$ROOT/tmp" && bash "$S/decode_flores.sh" "$M1" "$D/vocab.spm" "$SRC" "$O/decode/$who/$s.hyp" "$GPUS" ) 2>> "$O/decode/decode.log" || { log "decode failed $who $s"; return 1; }
    done
  done; }

stage inputs s_inputs
if [ "${GPU_SMOKE:-1}" = 1 ]; then stage gpu_smoke s_gpu_smoke; fi
stage kd s_kd
stage decode_kd s_decode kd
stage ft s_ft
stage decode_ft s_decode ft
( cd "$O" && sha256sum student_kd/model.npz.best-ce-mean-words.npz student_ft/model.npz.best-ce-mean-words.npz meta/* > hashes.sha256 2>/dev/null; sha256sum "$WK/kd/kd.clean.tsv" "$WK/kd/ft.clean.tsv" >> hashes.sha256 2>/dev/null )
log "ALL_DONE"
