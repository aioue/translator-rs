#!/bin/bash
# Model A (km->en) one-box pipeline: backward model -> CE score -> bucketed cut -> eval-exclusion -> rehearsal/finetune set ->
# KD student -> finetune -> decode every eval slice with both checkpoints.  Runs on a rented NVIDIA box in the
# marian-cuda:fp16-1a743582 image (or on the km box in SMOKE=1 mode with a CPU marian wrapper).
#
# Resumability: every stage drops /w/markers/<stage>.done; a restart skips finished stages and Marian resumes from its own
# model.npz checkpoints (train_student.sh/finetune_student.sh pass --overwrite and resume from the last save).
# Guard contract: this script is started by start_job.sh (so /w/job.pid exists), appends to /w/logs/job.log every 2 minutes
# (heartbeat) and ends with ALL_DONE.  A failed stage writes FAILED_<stage> to the log and exits non-zero WITHOUT a done marker.
#
# Env: ROOT (default /w) ; SMOKE=1 (tiny subsets; MARIAN must be a wrapper that makes the binary CPU-only and short)
#      KD_TIMEOUT_MIN (default 330) FT_TIMEOUT_MIN (default 120) BW_TIMEOUT_MIN (default 120)
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
  mkdir -p "$WK/data" && cd "$WK/data" || return 1
  if [ -f "$D/kd_aligned.tsv.gz" ]; then zcat "$D/kd_aligned.tsv.gz" > kd_aligned.tsv; else cp "$D/kd_aligned.tsv" kd_aligned.tsv; fi
  zcat "$D/ft_nonkd_aligned.tsv.gz" > ft_nonkd_aligned.tsv 2>/dev/null || cp "$D/ft_nonkd_aligned.tsv" ft_nonkd_aligned.tsv
  cp "$D/pass.reg" "$D/pass.idx" .; [ "$SMOKE" = 1 ] || true
  zcat "$D/bw_clean.tsv.gz" > bw_clean.tsv 2>/dev/null || cp "$D/bw_clean.tsv" bw_clean.tsv
  if [ "$SMOKE" = 1 ]; then head -n 3000 bw_clean.tsv > x && mv x bw_clean.tsv; head -n 4000 kd_aligned.tsv > x && mv x kd_aligned.tsv
     head -n 4000 pass.reg > x && mv x pass.reg; head -n 4000 pass.idx > x && mv x pass.idx; head -n 1500 ft_nonkd_aligned.tsv > x && mv x ft_nonkd_aligned.tsv; fi
  cut -f1 kd_aligned.tsv > kd.km; cut -f2 kd_aligned.tsv > kd.en
  cut -f2 bw_clean.tsv > bw.km; cut -f1 bw_clean.tsv > bw.en
  wc -l kd_aligned.tsv bw_clean.tsv ft_nonkd_aligned.tsv pass.reg >> "$L"
  [ "$(wc -l < kd.km)" = "$(wc -l < pass.reg)" ]; }
s_gpu_smoke() { local G="$WK/gpu_smoke"; rm -rf "$G"; mkdir -p "$G" && cd "$G" || return 1
  # tiny deterministic samples of the real inputs
  awk 'NR%80==0' "$WK/data/kd_aligned.tsv" > kd.tsv; awk 'NR%8==0' "$WK/data/ft_nonkd_aligned.tsv" > ftn.tsv
  awk 'NR%20==0' "$WK/data/bw.en" > bw.en; awk 'NR%20==0' "$WK/data/bw.km" > bw.km
  cat kd.tsv ftn.tsv > ft.tsv
  python3 "$S/model_a_exclude.py" ft.tsv ft.clean.tsv --text-columns 2 || return 11
  export SHORT_AFTER=${SHORT_AFTER:-150u}
  MARIAN=$S/marian_short.sh bash "$S/backward_step.sh" "$G/bw.en" "$G/bw.km" "$D/vocab.spm" "$D/valid.km2en.tsv" "$G/bw/model.npz" "$GPUS" > bw.log 2>&1 || { tail -30 bw.log >> "$L"; return 12; }
  head -n 300 "$WK/data/kd.en" > s.en; head -n 300 "$WK/data/kd.km" > s.km
  bash "$S/cefilter_score.sh" "$G/bw/model.npz.best-ce-mean-words.npz" "$D/vocab.spm" s.en s.km s.scores "$GPUS" > score.log 2>&1 || { tail -30 score.log >> "$L"; return 13; }
  [ "$(wc -l < s.scores)" = 300 ] || return 14
  MARIAN=$S/marian_short.sh bash "$S/train_student.sh" "$G/kd.tsv" "$D/vocab.spm" "$D/valid.km2en.tsv" "$G/kd/model.npz" "$GPUS" "--fp16 --workspace 12000 --mini-batch 4000" > kd.log 2>&1 || { tail -40 kd.log >> "$L"; return 15; }
  MARIAN=$S/marian_short.sh bash "$S/finetune_student.sh" "$G/ft.clean.tsv" "$D/vocab.spm" "$G/kd/model.npz.best-ce-mean-words.npz" "$D/valid.km2en.tsv" "$G/ft/model.npz" "$GPUS" > ft.log 2>&1 || { tail -40 ft.log >> "$L"; return 16; }
  head -n 20 "$D/eval/flores-plus-devtest.km.zwsp-stripped" > d.src
  bash "$S/decode_flores.sh" "$G/ft/model.npz.best-ce-mean-words.npz" "$D/vocab.spm" d.src d.hyp "$GPUS" > dec.log 2>&1 || { tail -20 dec.log >> "$L"; return 17; }
  [ "$(wc -l < d.hyp)" = 20 ] || return 18
  { echo "gpu_smoke ok; sample hyp:"; head -n 3 d.hyp; grep -a "Up\. 150\|valid\]" kd.log | tail -3; grep -a "words/s" kd.log | tail -2; } >> "$L" 2>&1
  cp bw.log score.log kd.log ft.log dec.log "$MT/" 2>/dev/null; return 0; }
s_bw() { cd "$O" && mkdir -p bw && lim ${BW_TIMEOUT_MIN:-120} bash "$S/backward_step.sh" "$WK/data/bw.en" "$WK/data/bw.km" "$D/vocab.spm" "$D/valid.km2en.tsv" "$O/bw/model.npz" "$GPUS" 2>&1 | tee -a "$O/bw/train.log" > /dev/null; rc=${PIPESTATUS[0]}; if [ $rc -eq 0 ] || [ $rc -eq 124 ]; then rm -f "$O/bw/model.npz.optimizer.npz"; fi; return $rc; }  # optimizer state (800 MB) is only for resuming; drop it once the stage is done so the pull loop does not carry it
s_score() { cd "$O" && mkdir -p scores
  BW="$O/bw/model.npz.best-ce-mean-words.npz"; [ -f "$BW" ] || BW="$O/bw/model.npz"
  bash "$S/cefilter_score.sh" "$BW" "$D/vocab.spm" "$WK/data/kd.en" "$WK/data/kd.km" "$O/scores/ce_scores.txt" "$GPUS" 2> "$O/scores/score.log"; rc=$?
  [ $rc -eq 0 ] && [ "$(wc -l < "$O/scores/ce_scores.txt")" = "$(wc -l < "$WK/data/kd.km")" ]; }
s_sanity() { cd "$WK/data" && python3 "$S/ce_sanity.py" prep kd.km kd.en san 127 >> "$L" 2>&1 \
  && bash "$S/cefilter_score.sh" "$BWM" "$D/vocab.spm" san.en.shuf san.km san.shuf.scores "$GPUS" 2> "$O/scores/sanity.log" \
  && python3 "$S/ce_sanity.py" judge "$O/scores/ce_scores.txt" 127 san.shuf.scores | tee "$MT/ce_sanity.txt" >> "$L" || [ "$SMOKE" = 1 ]; }
s_cut() { cd "$WK" && mkdir -p kd && python3 "$S/ce_cut.py" --tsv data/kd_aligned.tsv --scores "$O/scores/ce_scores.txt" --reg data/pass.reg --idx data/pass.idx \
    --out kd/kd.cut.tsv --rejects "$MT/rejects.ce.tsv" --summary "$MT/ce_cut_summary.json" --pct 5 >> "$L" 2>&1; }
s_excl_kd() { cd "$WK/kd" && python3 "$S/model_a_exclude.py" kd.cut.tsv kd.clean.tsv --text-columns 2 >> "$L" 2>&1 && [ -s kd.clean.tsv.evalclean.json ] \
  && python3 "$S/diff_rejects.py" kd.cut.tsv kd.clean.tsv "$MT/rejects.evalleak.tsv" >> "$L" 2>&1 && cp kd.clean.tsv.evalclean.json "$MT/"; }
s_ft_set() { cd "$WK/kd" && N=$(wc -l < "$WK/data/ft_nonkd_aligned.tsv") \
  && python3 "$S/sample_rehearsal.py" --pool kd.clean.tsv --rows "$N" --out kd.rehearsal.tsv --seed model-a-km2en-ft1 --long-words 3 --boost 3 --report "$MT/kd.rehearsal.json" >> "$L" 2>&1 \
  && cat "$WK/data/ft_nonkd_aligned.tsv" kd.rehearsal.tsv > ft.tsv && python3 "$S/model_a_exclude.py" ft.tsv ft.clean.tsv --text-columns 2 >> "$L" 2>&1 && [ -s ft.clean.tsv.evalclean.json ] \
  && python3 "$S/diff_rejects.py" ft.tsv ft.clean.tsv "$MT/rejects.ft_evalleak.tsv" >> "$L" 2>&1 && cp ft.clean.tsv.evalclean.json "$MT/"; }
s_kd() { cd "$O" && mkdir -p student_kd && lim ${KD_TIMEOUT_MIN:-330} bash "$S/train_student.sh" "$WK/kd/kd.clean.tsv" "$D/vocab.spm" "$D/valid.km2en.tsv" "$O/student_kd/model.npz" "$GPUS" "${KD_EXTRA:---fp16 --workspace 12000 --mini-batch 4000}" 2>&1 | tee -a "$O/student_kd/train.log" > /dev/null; return ${PIPESTATUS[0]}; }
s_ft() { cd "$O" && mkdir -p student_ft; KB="$O/student_kd/model.npz.best-ce-mean-words.npz"; [ -f "$KB" ] || KB="$O/student_kd/model.npz"
  lim ${FT_TIMEOUT_MIN:-120} bash "$S/finetune_student.sh" "$WK/kd/ft.clean.tsv" "$D/vocab.spm" "$KB" "$D/valid.km2en.tsv" "$O/student_ft/model.npz" "$GPUS" 2>&1 | tee -a "$O/student_ft/train.log" > /dev/null; return ${PIPESTATUS[0]}; }
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
stage bw s_bw
stage score s_score
BWM="$O/bw/model.npz.best-ce-mean-words.npz"; [ -f "$BWM" ] || BWM="$O/bw/model.npz"
stage sanity s_sanity
stage cut s_cut
stage excl_kd s_excl_kd
stage ft_set s_ft_set
stage kd s_kd
stage decode_kd s_decode kd
stage ft s_ft
stage decode_ft s_decode ft
( cd "$O" && sha256sum student_kd/model.npz.best-ce-mean-words.npz student_ft/model.npz.best-ce-mean-words.npz bw/model.npz.best-ce-mean-words.npz meta/* scores/ce_scores.txt > hashes.sha256 2>/dev/null; sha256sum "$WK/kd/kd.clean.tsv" "$WK/kd/ft.clean.tsv" >> hashes.sha256 2>/dev/null )
log "ALL_DONE"
