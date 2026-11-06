#!/usr/bin/env python3
"""Inspect the OpusTrainer-augmented Khmer stream on the REAL KD training TSV (km box, inside the marian-cuda image, CPU).
usage (inside image): inspect_aug.py SAMPLE_TSV(3col) OUT_DIR [N=30000]
Runs opustrainer-train with the student km config (__TSV__ -> SAMPLE), --no-shuffle, collects the first N augmented lines through a
sink, pairs them with the originals in order (the first pass is sequential; synthetic Noise lines are detected as lines whose English
is not the original's and are skipped), and counts NEW malformed Khmer tokens (leading combining mark, coeng not before a consonant,
double dependent vowel) that augmentation introduced, plus how many lines each modifier touched. Writes examples.txt for reading."""
import re, subprocess, sys, unicodedata, pathlib, collections, textwrap
sample, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]); N = int(sys.argv[3]) if len(sys.argv) > 3 else 30000
out.mkdir(parents=True, exist_ok=True)
cfg_src = pathlib.Path(__import__("os").environ.get("OT_CFG", "/scripts/configs/opustrainer.student.km.yml")).read_text()
cfg = out / "ot.yml"; cfg.write_text(cfg_src.replace("__TSV__", str(sample)))
sink = out / "sink.py"; aug = out / "aug.tsv"
sink.write_text(textwrap.dedent(f"""
    import sys
    n=0
    with open({str(aug)!r},'w',encoding='utf-8',newline='\\n') as f:
        for l in sys.stdin:
            f.write(l); n+=1
            if n>={N}: break
"""))
subprocess.run(["opustrainer-train", "--config", str(cfg), "--no-shuffle", "--do-not-resume", "--log-level", "WARNING", "python3", str(sink)],
               stderr=open(out / "ot.log", "w"), timeout=1800)
def kh(c): return 0x1780 <= ord(c) <= 0x17FF
def mn(c): return kh(c) and unicodedata.category(c) in ("Mn", "Mc")
def bad(t):
    f = set()
    for tok in t.split():
        p = None
        for i, c in enumerate(tok):
            if mn(c) and (i == 0 or (p and not kh(p))): f.add("leading_combining")
            if ord(c) == 0x17D2:
                n = tok[i + 1] if i + 1 < len(tok) else None
                if n is None or not (0x1780 <= ord(n) <= 0x17A2): f.add("bad_coeng")
            if p and 0x17B6 <= ord(c) <= 0x17C5 and 0x17B6 <= ord(p) <= 0x17C5: f.add("double_vowel")
            p = c
    return f
orig = [l.rstrip("\n").split("\t") for l in sample.read_text(encoding="utf-8").split("\n") if l]
augl = [l.rstrip("\n").split("\t") for l in aug.read_text(encoding="utf-8").split("\n") if l]
i = 0; st = collections.Counter(); ex = collections.defaultdict(list); skipped = 0
for a in augl:
    # find the original this line came from: advance i while the English differs (synthetic Noise lines don't consume a row)
    strip = lambda x: x.rstrip(" .!?,;:\u17d4\u17d5")
    if i < len(orig) and (a[1] == orig[i][1] or strip(a[1]) == strip(orig[i][1])):
        o = orig[i]; i += 1
    else:
        skipped += 1; st["synthetic_or_unpaired"] += 1; continue
    st["paired"] += 1
    if a[0] != o[0]:
        st["khmer_changed"] += 1
        new = bad(a[0]) - bad(o[0])
        for k in new: st["new_" + k] += 1
        if new: st["new_any"] += 1
        if len(ex["changed"]) < 25 or (new and len(ex["new"]) < 40):
            (ex["new"] if new else ex["changed"]).append((o[0], a[0], sorted(new)))
    if a[1] != o[1]: st["english_changed"] += 1
    if a[1] != o[1] or (len(a) > 2 and a[2] != o[2]): st["endpunct_or_align_changed"] += 1
    if len(a) > 2 and a[2] == "" and o[2] != "": st["alignment_blanked"] += 1
    if bad(o[0]): st["orig_bad_any"] += 1
print(dict(st), "rows consumed", i)
with open(out / "examples.txt", "w", encoding="utf-8") as f:
    f.write(str(dict(st)) + "\n\n== NEW MALFORMED ==\n")
    for o, a, n in ex["new"]: f.write(f"{n}\n  before: {o}\n  after:  {a}\n")
    f.write("\n== CHANGED (not malformed) ==\n")
    for o, a, n in ex["changed"]: f.write(f"  before: {o}\n  after:  {a}\n")
