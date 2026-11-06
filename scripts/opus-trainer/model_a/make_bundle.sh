#!/bin/bash
# Build the Model A upload bundle on the km box (CPU): /home/km/work/model-a/bundle  (MANIFEST.sha256 inside, sha256sum format).
# usage: make_bundle.sh [OUT=/home/km/work/model-a/bundle]
set -euo pipefail
W=/home/km/work/model-a; P=/home/km/work/pool-v3; OUT=${1:-$W/bundle}
rm -rf "$OUT.tmp"; mkdir -p "$OUT.tmp/scripts" "$OUT.tmp/data/valid"
cp -r "$W/scripts/." "$OUT.tmp/scripts/"; rm -rf "$OUT.tmp/scripts/__pycache__"
D=$OUT.tmp/data
cp "$P/vocab.spm" "$D/vocab.spm"; cp "$W/valid.km2en.tsv" "$D/"; cp "$W/valid/valid.en" "$W/valid/valid.km" "$D/valid/"
cp -r "$P/eval" "$D/eval"; cp -r "$P/hashes" "$D/hashes"; cp "$P/probes_en_to_km.tsv" "$D/"
gzip -1 -c "$W/kd/align/train.tsv" > "$D/kd_aligned.tsv.gz"
gzip -1 -c "$W/ft/align/train.tsv" > "$D/ft_nonkd_aligned.tsv.gz"
gzip -1 -c "$W/bw/bw_clean.tsv" > "$D/bw_clean.tsv.gz"
cp "$W/kd/pass.reg" "$W/kd/pass.idx" "$D/"
chmod +x "$OUT.tmp/scripts/"*.sh
( cd "$OUT.tmp" && find . -type f ! -name MANIFEST.sha256 -print0 | sort -z | xargs -0 sha256sum > MANIFEST.sha256 )
rm -rf "$OUT"; mv "$OUT.tmp" "$OUT"; du -sh "$OUT"; wc -l "$OUT/MANIFEST.sha256"
