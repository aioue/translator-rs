#!/usr/bin/env python3
"""CE-filter cut, bucketed (NEW_LANGUAGE.md section 7: "rank within length buckets, drop only the configured worst fraction,
preserve row order, record coverage"). David's cefilter_cut.sh cuts one global tail; per-token cross-entropy favours short
lines and registers differ, so this cuts the worst PCT% inside every (register x English-length bucket) instead.
Score = marian-scorer --normalize output (log-prob per token of the Khmer source given the teacher English): HIGHER is better,
so the worst are the lowest. Non-finite/unparsable scores count as worst. Dropped rows are never deleted: they go to REJECTS.

usage: ce_cut.py --tsv aligned_kd.tsv --scores scores.txt --reg pass.reg --idx pass.idx --out kd.cut.tsv --rejects rejects.ce.tsv
                 --summary summary.json [--pct 5]
Buckets (English whitespace words): 1, 2, 3-4, 5-8, 9-16, 17-32, 33+.  Output keeps the 3-column rows in the original order."""
import argparse, json, math, collections
ap = argparse.ArgumentParser()
for a in ("tsv", "scores", "reg", "idx", "out", "rejects", "summary"): ap.add_argument("--" + a, required=True)
ap.add_argument("--pct", type=float, default=5.0)
a = ap.parse_args()
rd = lambda p: open(p, encoding="utf-8", newline="\n")
tsv = [l.rstrip("\n") for l in rd(a.tsv)]
sc = []
for l in rd(a.scores):
    try: v = float(l.strip().split()[0])
    except Exception: v = float("-inf")
    sc.append(v if math.isfinite(v) else float("-inf"))
reg = [l.rstrip("\n") for l in rd(a.reg)]; idx = [l.rstrip("\n") for l in rd(a.idx)]
assert len(tsv) == len(sc) == len(reg) == len(idx), (len(tsv), len(sc), len(reg), len(idx))
def bucket(n): return 0 if n <= 1 else 1 if n == 2 else 2 if n <= 4 else 3 if n <= 8 else 4 if n <= 16 else 5 if n <= 32 else 6
groups = collections.defaultdict(list)
for i, row in enumerate(tsv):
    groups[(reg[i], bucket(len(row.split("\t")[1].split())))].append(i)
drop = set()
for key, ids in groups.items():
    k = int(len(ids) * a.pct / 100.0)
    ids.sort(key=lambda i: (sc[i], i))   # worst first, stable
    drop.update(ids[:k])
stat = collections.defaultdict(lambda: [0, 0])
with open(a.out, "w", encoding="utf-8", newline="\n") as fo, open(a.rejects, "w", encoding="utf-8", newline="\n") as fr:
    for i, row in enumerate(tsv):
        stat[reg[i]][0] += 1
        if i in drop:
            km, en = row.split("\t")[:2]
            fr.write(f"{idx[i]}\t{reg[i]}\t{sc[i]}\t{km}\t{en}\n")
        else:
            fo.write(row + "\n"); stat[reg[i]][1] += 1
fin = [x for x in sc if x == float("-inf")]
out = {"pct": a.pct, "rows": len(tsv), "kept": len(tsv) - len(drop), "dropped": len(drop), "nonfinite_scores": len(fin),
       "per_register": {r: {"rows": v[0], "kept": v[1], "keep_rate": round(v[1] / v[0], 4)} for r, v in stat.items()}}
json.dump(out, open(a.summary, "w"), indent=1); print(json.dumps(out))
