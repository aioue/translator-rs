#!/bin/bash
# Retrain bundle (km box): scripts + eval/valid/vocab + the augmented KD and finetune TSVs (gz) with their exclude_eval reports.
set -euo pipefail
W=/home/km/work/model-a; P=/home/km/work/pool-v3; OUT=$W/bundle2
rm -rf "$OUT.tmp"; mkdir -p "$OUT.tmp/scripts" "$OUT.tmp/data/valid"
cp -r "$W/scripts/." "$OUT.tmp/scripts/"; rm -rf "$OUT.tmp/scripts/__pycache__"; D=$OUT.tmp/data
cp "$P/vocab.spm" "$D/"; cp "$W/valid.km2en.tsv" "$D/"; cp "$W/valid/valid.en" "$W/valid/valid.km" "$D/valid/"
cp -r "$P/eval" "$D/eval"; cp -r "$P/hashes" "$D/hashes"; cp "$P/probes_en_to_km.tsv" "$D/"
gzip -1 -c "$W/retrain/kd.v2.clean.tsv" > "$D/kd.v2.clean.tsv.gz"; gzip -1 -c "$W/retrain/ft.v2.clean.tsv" > "$D/ft.v2.clean.tsv.gz"
cp "$W/retrain/kd.v2.clean.tsv.evalclean.json" "$W/retrain/ft.v2.clean.tsv.evalclean.json" "$D/"
chmod +x "$OUT.tmp/scripts/"*.sh
( cd "$OUT.tmp" && find . -type f ! -name MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256 )
rm -rf "$OUT"; mv "$OUT.tmp" "$OUT"; du -sh "$OUT"
