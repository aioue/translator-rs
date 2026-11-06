#!/usr/bin/env python3
"""Stage-1/2 prep: build the CE-filter inputs and the rule screen over the FULL teacher output (km box, CPU, streaming).

Inputs (line-aligned, 2,566,243 rows): baseline-prep/kd-km2en/kd_src.v2 (Khmer), teacher-km2en/teacher_hyp.txt (teacher English),
labse/scores.tsv (line, register, cos).
Outputs in /home/km/work/model-a/kd/ (nothing overwritten, nothing dropped here, only flagged):
  km.txt   Khmer source, ZWSP removed (runtime strips U+200B before translation; training must match), tabs/CR -> space
  en.txt   teacher English, tabs/CR -> space
  reg.txt  register per row
  flags.tsv  idx(0-based) \t register \t flags(comma list, '-' if none) \t km_chars \t en_chars
  screen_summary.json  counts per flag per register
Flags: EMPTY (either side blank), COPY (en == km), KHMER_IN_EN (>15% Khmer-script chars in English),
LEN_LOW/LEN_HIGH (en chars / km chars < 0.12 or > 2.5), REPEAT (loop in English), NUM_MISMATCH (digit multiset differs after
Khmer->ASCII digits and removing thousands separators), SRC_EMPTY_AFTER_STRIP.
The same flag definitions as runbook/cpu_side_screen.py (which covers only 191 of 257 shards), re-run on all rows here.
"""
import collections, json, os, re, sys
W = "/home/km/work/model-a"; B = "/home/km/work/baseline-prep"
os.makedirs(f"{W}/kd", exist_ok=True)
KH = re.compile(r"[ក-៿]")
KHD = str.maketrans("០១២៣៤៥៦៧៨៩", "0123456789")
REP_W = re.compile(r"(\b\S+(?:\s+\S+){0,2})(?:\s+\1){3,}", re.I)
REP_C = re.compile(r"(.{12,}?)\1{3,}")
NUM = re.compile(r"\d+(?:[.,]\d+)*")
def nums(t):
    t = t.translate(KHD)
    return collections.Counter(re.sub(r"[.,](?=\d{3}\b)", "", m) for m in NUM.findall(t))
cnt = collections.defaultdict(collections.Counter); tot = collections.Counter()
op = lambda p, m="r": open(p, m, encoding="utf-8", newline="\n")
with op(f"{B}/kd-km2en/kd_src.v2") as fs, op(f"{B}/teacher-km2en/teacher_hyp.txt") as fh, op(f"{B}/labse/scores.tsv") as fr, \
     op(f"{W}/kd/km.txt", "w") as ok, op(f"{W}/kd/en.txt", "w") as oe, op(f"{W}/kd/reg.txt", "w") as orr, op(f"{W}/kd/flags.tsv", "w") as of:
    for i, (s, h, r) in enumerate(zip(fs, fh, fr)):
        s = s.rstrip("\n"); h = h.rstrip("\n")
        li, reg, cos = r.rstrip("\n").split("\t")
        assert int(li) in (i, i + 1), (i, li)
        s = s.replace("​", "").replace("\t", " ").replace("\r", " ").strip()
        h = h.replace("\t", " ").replace("\r", " ").strip()
        f = []
        if not s or not h: f.append("EMPTY")
        else:
            if s == h: f.append("COPY")
            kh = len(KH.findall(h))
            if kh / max(len(h), 1) > 0.15: f.append("KHMER_IN_EN")
            ratio = len(h) / max(len(s), 1)
            if ratio < 0.12: f.append("LEN_LOW")
            if ratio > 2.5: f.append("LEN_HIGH")
            if REP_W.search(h) or REP_C.search(h): f.append("REPEAT")
            if nums(s) != nums(h): f.append("NUM_MISMATCH")
        ok.write(s + "\n"); oe.write(h + "\n"); orr.write(reg + "\n")
        of.write(f"{i}\t{reg}\t{','.join(f) or '-'}\t{len(s)}\t{len(h)}\n")
        tot[reg] += 1
        for x in f: cnt[reg][x] += 1
        if f: cnt[reg]["ANY"] += 1
        if i % 500000 == 0: print(i, flush=True)
n = sum(tot.values())
json.dump({"rows": n, "per_register_rows": tot, "flags": cnt}, open(f"{W}/kd/screen_summary.json", "w"), indent=1)
print("rows", n); 
for reg in tot: print(reg, tot[reg], dict(cnt[reg]))
