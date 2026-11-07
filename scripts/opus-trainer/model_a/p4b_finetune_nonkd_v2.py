#!/usr/bin/env python3
"""Finetune block that is NOT KD rehearsal, in km->en orientation (km \t en). km box, CPU.
Sources (all ZWSP-stripped on the Khmer side):
  human     all 4,574 LaBSE-trusted human bitext rows
  dialogue  100,000 of 242,521 trusted OpenSubtitles rows (LaBSE >= 0.60, ~1% wrong in the blind sample)
  crawl     60,000 of 403,540 trusted crawl rows (LaBSE >= 0.75, ~4% wrong) - mined English is the target
  ui        all 26,838 trusted UI rows (LaBSE >= 0.70, ~12% wrong: weak spot, but UI is the camera/app register)
  gen       38,334 generated short rows (finetune_gen.v2.tsv: Luna + Qwen/Hy-MT2, GEL-example rows removed), columns SWAPPED from en\tkm
Not used: entity (romanisation risk), generated-menu rows are already absent from finetune_gen (held back upstream).
Seeded sampling (random.Random(20261106)). Output ft/ft_nonkd.tsv + ft/ft_nonkd.reg + ft/ft_nonkd.summary.json.
Rows containing a tab, empty sides or exact duplicates are skipped (counted)."""
import json, os, random, collections
W = "/home/km/work/model-a"; os.makedirs(f"{W}/retrain/ft2", exist_ok=True)
rng = random.Random(20261107); ZW = "​"
by = collections.defaultdict(list)
for l in open(f"{W}/bw/bw_clean.tsv", encoding="utf-8", newline="\n"):   # en \t km, already trusted/stripped/eval-excluded
    e, k = l.rstrip("\n").split("\t"); by["x"].append((e, k))
# need register: re-read trusted_refs for the register column of the same rows
reg_of = {}
for l in open("/home/km/work/baseline-prep/labse/trusted_refs.tsv", encoding="utf-8", newline="\n"):
    e, k, r, c = l.rstrip("\n").split("\t"); reg_of[(e.strip(), k.replace(ZW, "").strip())] = r
pools = collections.defaultdict(list)
for e, k in by["x"]:
    pools[reg_of.get((e, k), "?")].append((e, k))
take = {"human": None, "dialogue": 150000, "crawl": 30000, "ui": None}  # RETRAIN: dialogue 100k->150k, crawl 60k->30k
rows = []; summ = {}
for r, n in take.items():
    p = pools[r]
    if n and len(p) > n: p = rng.sample(p, n)
    for e, k in p: rows.append((k, e, r))
    summ[r] = len(p)
gen_skip = collections.Counter()
for l in open(f"{W}/ft_gen_v2_en_km.tsv", encoding="utf-8", newline="\n"):
    p = l.rstrip("\n").split("\t")
    if len(p) != 2: gen_skip["cols"] += 1; continue
    e, k = p[0].strip(), p[1].replace(ZW, "").strip()
    if not e or not k: gen_skip["empty"] += 1; continue
    rows.append((k, e, "gen"))
summ["gen"] = sum(1 for x in rows if x[2] == "gen")
# RETRAIN: UI and generated short rows enter TWICE (exact copies; number rows are redrawn by perturb_filtered afterwards).
seen = set(); out = []
for k, e, r in rows:
    if (k, e) in seen: gen_skip["dup"] += 1; continue
    seen.add((k, e)); out.append((k, e, r))
dup2 = [x for x in out if x[2] in ("ui", "gen")]; out = out + dup2; summ["extra_copies_ui_gen"] = len(dup2)
rng.shuffle(out)
with open(f"{W}/retrain/ft2/ft_nonkd.tsv", "w", encoding="utf-8", newline="\n") as f, open(f"{W}/retrain/ft2/ft_nonkd.reg", "w", newline="\n") as g:
    for k, e, r in out:
        f.write(f"{k}\t{e}\n"); g.write(r + "\n")
json.dump({"rows": len(out), "per_source": summ, "skipped": gen_skip}, open(f"{W}/retrain/ft2/ft_nonkd.summary.json", "w"), indent=1)
print(len(out), summ, dict(gen_skip))
