#!/usr/bin/env python3
"""Stage-1 prep: valid files + backward-model training data. Runs on the km box.
Outputs (all new files under /home/km/work/model-a/, nothing overwritten):
  valid/valid.en valid/valid.km, valid.km2en.tsv (km\ten, what the km->en student validates on; valid.tsv is en\tkm)
  bw/bw_all.tsv   en\tkm  from LaBSE trusted refs (ZWSP stripped), exact-dup removed; rejects -> bw/bw_rejects.tsv
"""
import os, re, collections
W = "/home/km/work/model-a"; B = "/home/km/work/baseline-prep"
os.makedirs(f"{W}/valid", exist_ok=True); os.makedirs(f"{W}/bw", exist_ok=True)
ZW = "​"
en, km = [], []
for l in open(f"{B}/valid.tsv", encoding="utf-8"):
    a, b = l.rstrip("\n").split("\t"); en.append(a); km.append(b)
open(f"{W}/valid/valid.en", "w", encoding="utf-8").write("\n".join(en) + "\n")
open(f"{W}/valid/valid.km", "w", encoding="utf-8").write("\n".join(km) + "\n")
with open(f"{W}/valid.km2en.tsv", "w", encoding="utf-8") as f:
    for a, b in zip(en, km):
        assert ZW not in b and "\t" not in a + b
        f.write(f"{b}\t{a}\n")
seen = set(); n = kept = 0; reg = collections.Counter(); rej = collections.Counter()
with open(f"{B}/labse/trusted_refs.tsv", encoding="utf-8") as fi, open(f"{W}/bw/bw_all.tsv", "w", encoding="utf-8") as fo, \
     open(f"{W}/bw/bw_rejects.tsv", "w", encoding="utf-8") as fr:
    for l in fi:
        n += 1
        p = l.rstrip("\n").split("\t")
        if len(p) != 4: rej["badcols"] += 1; fr.write("badcols\t" + l); continue
        e, k, r, c = p
        k = k.replace(ZW, "").strip(); e = e.strip()
        if not e or not k: rej["empty"] += 1; fr.write("empty\t" + l); continue
        if (e, k) in seen: rej["dup"] += 1; fr.write("dup\t" + l); continue
        seen.add((e, k)); kept += 1; reg[r] += 1
        fo.write(f"{e}\t{k}\n")
print("trusted rows", n, "kept", kept, dict(reg), "rejected", dict(rej))
